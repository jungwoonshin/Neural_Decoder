
from functions import *
import gc

from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math, time
from copy import deepcopy
import multiprocessing as mp

import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader

import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class MyDataset(InMemoryDataset):
    def __init__(self, data_list, root,transform=None, pre_transform=None):
        self.data_list = data_list

        super(MyDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = self.data_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        del self.data_list


def subtraction(tensor1, tensor2):
    left = torch.logical_xor(tensor1, tensor2)
    return torch.logical_and(tensor1, left)

def divide_to_three(subgraph,node1,node2):
    subgraph[0,1] = 0
    subgraph[1,0] = 0
    subgraph.fill_diagonal_(0)

    common = torch.logical_and(subgraph[0,:],subgraph[1,:])
    common_neighbor = torch.nonzero(common)
    node1_neighbor = torch.nonzero(subtraction(subgraph[0,:],common))
    node2_neighbor = torch.nonzero(subtraction(subgraph[1,:],common))
    return subgraph, common_neighbor, node1_neighbor, node2_neighbor


def one_hot(dist_tensor, length, max_nodes_per_hop):
    x = torch.zeros((max_nodes_per_hop*2+2),5)
    x[torch.arange(dist_tensor.shape[0]), dist_tensor] = 1.0
    return x

def edge_index_emb(g_label, node1, node2, A,two_hop,h,max_nodes_per_hop=10):
    node1 = torch.tensor([node1]).to(node1.device).long()
    node2 = torch.tensor([node2]).to(node2.device).long()

    node1_neighbor = A.index_select(0, node1)
    node2_neighbor = A.index_select(0, node2)
    
    two_hop_adj = node1_neighbor * node2_neighbor
    three_hop_adj = two_hop.index_select(0, node1) * node2_neighbor
    
    hop_adj = torch.add(two_hop_adj, three_hop_adj)
    hop_adj[0, node2] = 0
    hop_adj[0, node1] = 0
    hop_adj_top_k = torch.topk(hop_adj, k=max_nodes_per_hop*2, dim=1).indices

    node_list = [node1,node2, hop_adj_top_k.view(-1)]
    node_list = torch.cat(node_list, 0)
    # print('node_list:',node_list.shape)
    subgraph = A[node_list,:][:,node_list]
    # print('subgraph.size',subgraph.shape)
    subgraph, n12, n1,n2 = divide_to_three(subgraph, node1, node2)
    edge_index = torch.nonzero(subgraph).to(node1.device).T
    dist =  torch.tensor([0,1] +[2]*n12.shape[0]+[3]*n1.shape[0]+[4]*n2.shape[0])
    emb = one_hot(dist, 5, max_nodes_per_hop).to(node1.device)
    
    return edge_index, emb

def links2subgraphs(
        A,
        split_edge,
        h=1, 
        sample_ratio=1.0, 
        max_nodes_per_hop=10, 
        u_features=None, 
        v_features=None, 
        max_node_label=5, 
        class_values=None, 
        testing=False, 
        parallel=True,
        positive=False):
    # extract enclosing subgraphs
    if max_node_label is None:  # if not provided, infer from graphs
        max_n_label = {'max_node_label': 0}

    two_hop = A@A

    def helper(A, links,positive=False):
        g_labels = torch.ones(int(links.shape[0])).to(A.device).view(-1,1) if positive else torch.zeros(int(links.shape[0])).to(A.device).view(-1,1)

        # x = [edge_index_emb(g_label, i, j, A,two_hop,h,max_nodes_per_hop) for i, j,g_label  in zip(links[:,0], links[:,1], g_labels)]
        # g_labels = torch.ones(int(links.shape[0])).to(A.device) if positive else torch.zeros(int(links.shape[0])).to(A.device)

        g_list = []
        start = time.time()
        pool = mp.Pool(mp.cpu_count()) # (g_label, ind, A, h=1, max_nodes_per_hop=10)
        results = pool.starmap_async(parallel_worker, [(g_label, i, j, A,two_hop,h,max_nodes_per_hop) for i, j,g_label  in zip(links[:,0], links[:,1], g_labels)])
        remaining = results._number_left
        pbar = tqdm(total=remaining)
        while True:
            pbar.update(remaining - results._number_left)
            if results.ready(): break
            remaining = results._number_left
            time.sleep(1)
        results = results.get()
        pool.close()
        pbar.close()
        end = time.time()
        # print("Time eplased for subgraph extraction: {}s".format(end-start))
        # print("Transforming to pytorch_geometric graphs...".format(end-start))
        g_list += [Data(emb, edge_index, y=g_label) for g_label, edge_index, emb in tqdm(results)]
        # print([(data.x.shape, data.y.shape, data.edge_index.shape) for data in tqdm(g_list)])

        del results
        end2 = time.time()
        # print("Time eplased for transforming to pytorch_geometric graphs: {}s".format(end2-end))
        return g_list

    print('Enclosing subgraph extraction begins...')

    pos_train_edge = split_edge['train']['edge'].to(A.device)
    neg_train_edge = get_train_neg_edge(A, pos_train_edge)
    pos_valid_edge = split_edge['valid']['edge'].to(A.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(A.device)
    pos_test_edge = split_edge['test']['edge'].to(A.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(A.device)
    eval_train_edge = split_edge['eval_train']['edge'].to(A.device)

    print('train_loader:',neg_train_edge.shape, pos_train_edge.shape)
    print('valid_loader:',neg_valid_edge.shape, pos_valid_edge.shape)
    print('test_loader:',neg_test_edge.shape, pos_test_edge.shape)

    data_combo = ('ogbl-ddi', '', 'testmode')
    print('begin train')
    for index,perm in enumerate(DataLoader(range(eval_train_edge.size(0)), 10000,
                           shuffle=True)):
        edge = eval_train_edge[perm]
        eval_train_graphs = helper(A,edge,positive=True)
        eval_train_graphs = MyDataset(eval_train_graphs, root='data/{}{}/{}/eval_train'.format(*data_combo),index=index)
        del eval_train_graphs
        gc.collect()
    print('end train')
    
    print('begin train')
    for index, perm in enumerate(DataLoader(range(pos_train_edge.size(0)), 10000,
                           shuffle=True)):
        edge = pos_train_edge[perm]
        edge2 = neg_train_edge[perm]
        train_graphs = helper(A,edge,positive=True) + helper(A,edge2)
        train_graphs = MyDataset(train_graphs, root='data/{}{}/{}/train'.format(*data_combo),index=index)
        print('type(train_graphs): ',type(train_graphs))
        print('type(train_graphs): ',len(train_graphs))

        del train_graphs
        gc.collect()
    print('end train')
   
    print('begin valid')

    for index, perm in enumerate(DataLoader(range(pos_valid_edge.size(0)), 10000,
                           shuffle=False)):
        edge = pos_valid_edge[perm]
        val_graphs = helper(A,edge,positive=True)
        val_graphs = MyDataset(val_graphs, root='data/{}{}/{}/val'.format(*data_combo),index=index)

        print(len(val_graphs))
        del val_graphs
        gc.collect()

    print('end valid')
    print('begin valid')

    for index, perm in enumerate(DataLoader(range(neg_valid_edge.size(0)), 10000,
                           shuffle=False)):
        edge = neg_valid_edge[perm]
        val_graphs = helper(A,edge)
        val_graphs = MyDataset(val_graphs, root='data/{}{}/{}/val_false'.format(*data_combo),index=index)
        print(len(val_graphs))
        del val_graphs
        gc.collect()

    print('end valid')

    print('begin test')
    for index, perm in enumerate(DataLoader(range(pos_test_edge.size(0)), 10000,
                           shuffle=False)):
        edge = pos_test_edge[perm]
        test_graphs = helper(A,edge,positive=True)
        test_graphs = MyDataset(test_graphs, root='data/{}{}/{}/test'.format(*data_combo),index=index)
        print(len(test_graphs))
        del test_graphs
        gc.collect()

    print('end test')
    print('begin test')

    for index,perm in enumerate(DataLoader(range(neg_test_edge.size(0)), 10000,
                           shuffle=False)):
        edge = neg_test_edge[perm]
        test_graphs = helper(A,edge)
        test_graphs = MyDataset(test_graphs, root='data/{}{}/{}/test_false'.format(*data_combo),index=index)
        print(len(test_graphs))
        del test_graphs
        gc.collect()

    print('end test')
    exit('finished reprocessing #############################')
    return 

def parallel_worker(g_label, i,j, A,two_hop, h=1, max_nodes_per_hop=10):
    edge_index, emb = edge_index_emb(g_label,i,j, A,two_hop, h, max_nodes_per_hop)
    # g, node_labels, node_features = subgraph_extraction_labeling(ind, A, h, sample_ratio, max_nodes_per_hop, u_features, v_features, class_values)
    return g_label, edge_index, emb

def get_data_set(split_edge,A):

    pos_train_edge = split_edge['train']['edge'].to(A.device)
    neg_train_edge = get_train_neg_edge(A, pos_train_edge)
    pos_valid_edge = split_edge['valid']['edge'].to(A.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(A.device)
    pos_test_edge = split_edge['test']['edge'].to(A.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(A.device)
    eval_train_edge = split_edge['eval_train']['edge'].to(A.device)

    print('train_loader:',neg_train_edge.shape, pos_train_edge.shape)
    print('valid_loader:',neg_valid_edge.shape, pos_valid_edge.shape)
    print('test_loader:',neg_test_edge.shape, pos_test_edge.shape)

    data_combo = ('ogbl-ddi', '', 'testmode')
    print('begin train')
    eval_train_graphs = []
    for index,perm in enumerate(tqdm(DataLoader(range(eval_train_edge.size(0)), 10000,
                           shuffle=True))):
        eval_train_graphs += [MyDataset(None, root='data/{}{}/{}/eval_train'.format(*data_combo),index=index)]

    print('end train')

    print('begin train')
    train_graphs = []
    for index, perm in enumerate(tqdm(DataLoader(range(pos_train_edge.size(0)), 10000,
                           shuffle=True))):
        train_graphs += [MyDataset(None, root='data/{}{}/{}/train'.format(*data_combo),index=index)]

    print('end train')
   
    print('begin valid')
    val_graphs = []
    for index, perm in enumerate(tqdm(DataLoader(range(pos_valid_edge.size(0)), 10000,
                           shuffle=False))):
        val_graphs += [MyDataset(None, root='data/{}{}/{}/val'.format(*data_combo),index=index)]

    print('end valid')
    print('begin valid')

    for index,perm in enumerate(tqdm(DataLoader(range(neg_valid_edge.size(0)), 10000,
                           shuffle=False))):
        val_graphs += [MyDataset(None, root='data/{}{}/{}/val_false'.format(*data_combo),index=index)]
    print('end valid')

    print('begin test')
    test_graphs = []
    for index,perm in enumerate(tqdm(DataLoader(range(pos_test_edge.size(0)), 10000,
                           shuffle=False))):
        test_graphs +=  [MyDataset(None, root='data/{}{}/{}/test'.format(*data_combo),index=index)]

    print('end test')
    print('begin test')

    for index,perm in enumerate(tqdm(DataLoader(range(neg_test_edge.size(0)), 10000,
                           shuffle=False))):
        test_graphs += [MyDataset(None, root='data/{}{}/{}/test_false'.format(*data_combo),index=index)]
    train_graphs = torch.utils.data.ConcatDataset(train_graphs)
    eval_train_graphs = torch.utils.data.ConcatDataset(eval_train_graphs)
    test_graphs = torch.utils.data.ConcatDataset(test_graphs)
    val_graphs = torch.utils.data.ConcatDataset(val_graphs)

    return train_graphs, val_graphs, test_graphs, eval_train_graphs



