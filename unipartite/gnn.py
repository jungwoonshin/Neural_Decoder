import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling, to_networkx
import torch_geometric.utils as utils
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, VGAE
from torch_geometric.nn.models import InnerProductDecoder
from torch_geometric.data import Data

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from logger import Logger
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import os
from torch.utils.data import random_split
from torch_geometric.datasets import CitationFull
from model import *
import networkx as nx


def train(model, x, adj_t, split_edge, optimizer, batch_size):
    model.train()

    row, col, _ = adj_t.coo()
    edge_index = torch.stack([col, row], dim=0)

    pos_train_edge = split_edge.train_pos_edge_index.to(x.device)

    total_loss = total_examples = 0
    for index, perm in enumerate(DataLoader(range(pos_train_edge.size(1)), batch_size,
                           shuffle=True)):

        optimizer.zero_grad()
        
        pos_edge = pos_train_edge[:,perm]

        neg_edge = negative_sampling(pos_train_edge, num_nodes=x.size(0),
                                 num_neg_samples=perm.size(0), method='dense')

        loss = model.loss(x, pos_train_edge, pos_edge, neg_edge)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        num_examples = perm.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

        # if 
        
    return total_loss / total_examples



@torch.no_grad()
def test(model, x, adj_t,split_edge, evaluator, batch_size):
    model.eval()

    pos_train_edge = split_edge.train_pos_edge_index.to(x.device)
    pos_valid_edge = split_edge.val_pos_edge_index.to(x.device)
    neg_valid_edge = split_edge.val_neg_edge_index.to(x.device)
    pos_test_edge = split_edge.test_pos_edge_index.to(x.device)
    neg_test_edge = split_edge.test_neg_edge_index.to(x.device)
    
    h = model(x, pos_train_edge)
    predictor = model.decode

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(1)), batch_size):
        edge = pos_train_edge[:,perm]
        pos_train_preds += [predictor(h, edge).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(1)), batch_size):
        edge = pos_valid_edge[:,perm]
        pos_valid_preds += [predictor(h, edge).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(1)), batch_size):
        edge = neg_valid_edge[:,perm]
        neg_valid_preds += [predictor(h, edge).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(1)), batch_size):
        edge = pos_test_edge[:,perm]
        pos_test_preds += [predictor(h, edge).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(1)), batch_size):
        edge = neg_test_edge[:,perm]
        neg_test_preds += [predictor(h, edge).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)
    results = {}
    K_list = [10,20,30]
    for K in K_list:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        val_roc, val_ap = roc_ap_score(pos_valid_pred, neg_valid_pred)
        test_roc, test_ap = roc_ap_score(pos_test_pred, neg_test_pred)
        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits, test_roc, test_ap, val_roc, val_ap)

    return results

def roc_ap_score(edge_pos, edge_neg):
    preds = []
    pos = []
    for index, e in enumerate(edge_pos):
        preds.append(e)
        pos.append(1)

    preds_neg = []
    neg = []
    for index, e in enumerate(edge_neg):
        index += len(edge_pos)
        preds_neg.append(e)
        neg.append(0)

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score

def run(file, data_name, model_name,lr):
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64*1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--use_nd', action='store_true')
    parser.add_argument('--use_lgae', action='store_true')
    parser.add_argument('--use_vgae', action='store_true')
    parser.add_argument('--model', type=str, default='')

    parser.add_argument('--dataset', type=str, default='Citeseer')

    args = parser.parse_args()
    if data_name != None and model_name != None and lr != None:
        args.dataset = data_name
        args.model = model_name
        args.lr = lr
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    device = torch.device(device)

    dataset = CitationFull(os.path.join('citation_data',args.dataset),name=args.dataset,transform=T.ToSparseTensor())
    num_training = int(dataset.__len__()*0.8)
    num_val = int(dataset.__len__()*0.1)
    num_test = dataset.__len__() - (num_training+num_val)

    data = dataset[0]
    adj_t = data.adj_t.to(device)
    edge_index, edge_type = utils.dense_to_sparse(adj_t.to_dense())
    data.edge_index = edge_index
    data.x = data.x.to(device)
    num_nodes = data.x.shape[0]
    num_edges = data.edge_index.shape[1]
    print(data)
    # nx_data = to_networkx(data, to_undirected=True)
    # print('graph density='+str(2*num_edges/(num_nodes*(num_nodes-1))))
    # print('clustering coefficient='+str(nx.average_clustering(nx_data)))


    decoder_enable = args.model[-3:]
    if args.model[-3:] == '-nd': model_name = args.model[:-3]
    
    if model_name == 'lgae':
        model = LGAE(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout)
    elif model_name == 'vgae':
        model = DeepVGAE(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout)

    elif model_name == 'gae':
        model = GraphAutoEncoder(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout)

    elif model_name == 'arga':
        model = AdversarialGAE(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout)

    elif model_name == 'arvga':
        model = AdversarialVGAE(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout)
    elif model_name == 'lrga':
        model = LRGA(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout)
    elif model_name == 'sage':
        model = SAGEAutoEncoder(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout)

    if decoder_enable == '-nd':
        model.decoder = NeuralDecoder( args.hidden_channels,  
            args.hidden_channels, 1, args.num_layers, args.dropout)
    
    evaluator = Evaluator(name='ogbl-ddi')

    model = model.to(device)

    loggers = {}
    K_list = ['10','20','30']
    for k in K_list:
        loggers['Hits@'+k] = Logger(args.runs, args)

    for run in range(args.runs):
        torch.manual_seed(run)
        split_edge = utils.train_test_split_edges(data)
        # print(split_edge.train_pos_edge_index.shape)
        # print(split_edge.val_pos_edge_index.shape)


        # exit()
        split_edge.edge_index = edge_index

        # emb.weight.data = features
        model.reset_parameters()

        if args.model in ['arga','arga-nd','arvga','arvga-nd']:
            args.lr=0.005
        optimizer = torch.optim.Adam(
                list(model.parameters()), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data.x, adj_t, split_edge,
                         optimizer, args.batch_size)

            if epoch % args.eval_steps == 0:
                results = test(model, data.x, adj_t, split_edge,
                               evaluator, args.batch_size)
                for key, result in results.items():
                    loggers[key].add_result(run, result)
            

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits, test_auc, test_ap, val_auc, val_ap = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'auc: {100 * test_auc:.2f}%, '
                              f'ap: {100 * test_ap:.2f}%, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%', )
                    print('---')



        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        toWrite = loggers[key].print_statistics()

        file.write(str(args.lr)+' ' +key + ' ' +args.model+"'"+str(toWrite)+'\n')
        file.flush()

if __name__ == "__main__":
    data = ['cora','cora_ml','citeseer','pubmed']
    model_list = ['gae','gae-nd','vgae','vgae-nd','lgae','lgae-nd','sage','sage-nd','arga','arga-nd','arvga','arvga-nd','lrga','lrga-nd']

    for data_name in data:
       for model_name in model_list:
            file = open('output/'+ data_name+'.txt','a')
            run(file,data_name,model_name,0.001)
            file.close()
    

