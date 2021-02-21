import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.models import  VGAE, GAE, InnerProductDecoder, ARGA,ARGVA
from torch_geometric.nn.conv import GCNConv, SAGEConv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from graph_global_attention_layer import LowRankAttention, weight_init


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.gcn_shared = GCNConv(in_channels, hidden_channels, normalize=False)
        self.gcn_mu = GCNConv(hidden_channels, out_channels, normalize=False)
        self.gcn_logvar = GCNConv(hidden_channels, out_channels, normalize=False)

    def reset_parameters(self):
        self.gcn_shared.reset_parameters()
        self.gcn_mu.reset_parameters()
        self.gcn_logvar.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.gcn_shared(x, edge_index))
        mu = self.gcn_mu(x, edge_index)
        logvar = self.gcn_logvar(x, edge_index)
        return mu, logvar

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels,normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels,normalize=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class LGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LGCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs:
            x = conv(x, adj_t)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class NeuralDecoder(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(NeuralDecoder, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels//2))

        self.lins2 = torch.nn.ModuleList()
        self.lins2.append(torch.nn.Linear(int(hidden_channels*2), hidden_channels))
        self.lins2.append(torch.nn.Linear(hidden_channels, hidden_channels//2))

        self.lins_final = torch.nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            torch.nn.init.xavier_uniform_(lin.weight)
            # lin.reset_parameters()
        for lin in self.lins2:
            torch.nn.init.xavier_uniform_(lin.weight)
            # lin.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lins_final.weight)
        # self.lins_final.reset_parameters()

    def forward(self, z, edge_index,sigmoid=True):
        x_i = z[edge_index[0]] 
        x_j = z[edge_index[1]]
        x_dot = x_i * x_j
        for lin in self.lins:
            x_dot = lin(x_dot)
            x_dot = F.relu(x_dot)
            x_dot = F.dropout(x_dot, p=self.dropout, training=self.training)
        x_pair = torch.cat([x_i,x_j],1)
        for lin in self.lins2:
            x_pair = lin(x_pair)
            x_pair = F.relu(x_pair)
            x_pair = F.dropout(x_pair, p=self.dropout, training=self.training)
        x = torch.cat([x_dot, x_pair],1)
        x = self.lins_final(x)
        return torch.sigmoid(x)
class VariationalNeuralDecoder(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(VariationalNeuralDecoder, self).__init__()

        self.lins = torch.nn.ModuleList()
        # self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels//2))

        self.lins2 = torch.nn.ModuleList()
        self.lins2.append(torch.nn.Linear(int(hidden_channels*2), hidden_channels//2))
        # self.lins2.append(torch.nn.Linear(hidden_channels, hidden_channels//2))

        self.lins_final = torch.nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            torch.nn.init.xavier_uniform_(lin.weight)
            # lin.reset_parameters()
        for lin in self.lins2:
            torch.nn.init.xavier_uniform_(lin.weight)
            # lin.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lins_final.weight)
        # self.lins_final.reset_parameters()

    def forward(self, z, edge_index,sigmoid=True):
        x_i = z[edge_index[0]] 
        x_j = z[edge_index[1]]
        x_dot = x_i * x_j
        for lin in self.lins:
            x_dot = lin(x_dot)
            x_dot = F.relu(x_dot)
            x_dot = F.dropout(x_dot, p=self.dropout, training=self.training)
        x_pair = torch.cat([x_i,x_j],1)
        for lin in self.lins2:
            x_pair = lin(x_pair)
            x_pair = F.relu(x_pair)
            x_pair = F.dropout(x_pair, p=self.dropout, training=self.training)
        x = torch.cat([x_dot, x_pair],1)
        x = self.lins_final(x)
        return torch.sigmoid(x)
class DeepVGAE(VGAE):
    def __init__(self, in_channels, hidden_channels,
                    out_channels, num_layers,
                    dropout):
        super(DeepVGAE, self).__init__(encoder=GCNEncoder(in_channels, hidden_channels, hidden_channels),
                                       decoder=None)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        if type(self.decoder) != type(InnerProductDecoder()):
            self.decoder.reset_parameters()

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        # adj_pred = self.decoder.forward(z, pos_edge_index)
        return z

    def loss(self, x, edge_index, pos_edge_index, neg_edge_index):
        z = self.encode(x, edge_index)

        pos_loss = -torch.log(
            self.decode(z, pos_edge_index, sigmoid=True) + 1e-15).mean()
        neg_loss = -torch.log(1 - self.decode(z, neg_edge_index, sigmoid=True) + 1e-15).mean()
        kl_loss = 1 / x.size(0) * self.kl_loss()

        return pos_loss + neg_loss + kl_loss

class GraphAutoEncoder(GAE):
    def __init__(self, in_channels, hidden_channels,
                    out_channels, num_layers,
                    dropout):
        super(GraphAutoEncoder, self).__init__(encoder=GCN(in_channels, hidden_channels, hidden_channels, num_layers,dropout),
                                    decoder=None)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        # print('type(self.decoder)',type(self.decoder))
        if type(self.decoder) != type(InnerProductDecoder()):
            self.decoder.reset_parameters()

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return z

    def loss(self, x, edge_index, pos_edge_index, neg_edge_index):
        z = self.encode(x, edge_index)
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()
        return pos_loss + neg_loss

class SageEncoder(GAE):
    def __init__(self, in_channels, hidden_channels,
                    out_channels, num_layers,
                    dropout):
        super(SageEncoder, self).__init__(encoder=SAGE(in_channels, hidden_channels, hidden_channels, num_layers,dropout),
                                    decoder=None)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        # print('type(self.decoder)',type(self.decoder))
        if type(self.decoder) != type(InnerProductDecoder()):
            self.decoder.reset_parameters()

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return z

    def loss(self, x, edge_index, pos_edge_index, neg_edge_index):
        z = self.encode(x, edge_index)
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()
        return pos_loss + neg_loss


class LGAE(GAE):
    def __init__(self, in_channels, hidden_channels,
                    out_channels, num_layers,
                    dropout):
        super(LGAE, self).__init__(encoder=LGCN(in_channels, hidden_channels, hidden_channels, num_layers,dropout),
                                    decoder=None)
    
    def reset_parameters(self):
        self.encoder.reset_parameters()
        # print('type(self.decoder)',type(self.decoder))
        if type(self.decoder) != type(InnerProductDecoder()):
            self.decoder.reset_parameters()

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return z

    def loss(self, x, edge_index, pos_edge_index, neg_edge_index):
        z = self.encode(x, edge_index)
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()
        return pos_loss + neg_loss


class AdversarialGAE(ARGA):
    def __init__(self, in_channels, hidden_channels,
                    out_channels, num_layers,
                    dropout):
        encoder = GCN(in_channels, hidden_channels, hidden_channels, num_layers,dropout)
        discriminator = Discriminator(hidden_channels, hidden_channels, hidden_channels)
        super(AdversarialGAE, self).__init__(encoder=encoder,discriminator=discriminator,
                                    decoder=None)
        self.discrim_optimizer = torch.optim.Adam(
                list(self.discriminator.parameters()), lr=0.001)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        # print('type(self.decoder)',type(self.decoder))
        if type(self.decoder) != type(InnerProductDecoder()):
            self.decoder.reset_parameters()

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return z

    def loss(self, x, edge_index, pos_edge_index, neg_edge_index):
        z = self.encode(x, edge_index)

        for i in range(5):
            self.discriminator.train()
            self.discrim_optimizer.zero_grad()
            discriminator_loss = self.discriminator_loss(z)
            discriminator_loss.backward()
            self.discrim_optimizer.step()

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()
        return pos_loss + neg_loss

class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Discriminator, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x

class AdversarialVGAE(ARGVA):
    def __init__(self, in_channels, hidden_channels,
                    out_channels, num_layers,
                    dropout):
        encoder = GCNEncoder(in_channels, hidden_channels, hidden_channels)
        discriminator = Discriminator(hidden_channels, hidden_channels, hidden_channels)
        super(AdversarialVGAE, self).__init__(encoder=encoder,discriminator=discriminator,
                                    decoder=None)
        self.discrim_optimizer = torch.optim.Adam(
                list(self.discriminator.parameters()), lr=0.001)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        # print('type(self.decoder)',type(self.decoder))
        if type(self.decoder) != type(InnerProductDecoder()):
            self.decoder.reset_parameters()

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return z

    def loss(self, x, edge_index, pos_edge_index, neg_edge_index):
        z = self.encode(x, edge_index)

        for i in range(5):
            self.discriminator.train()
            self.discrim_optimizer.zero_grad()
            discriminator_loss = self.discriminator_loss(z)
            discriminator_loss.backward()
            self.discrim_optimizer.step()

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()
        kl_loss = 1 / x.size(0) * self.kl_loss()
        return pos_loss + neg_loss + kl_loss

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, z, edge_index,sigmoid=True):
        x_i = z[edge_index[0]] 
        x_j = z[edge_index[1]]
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

class LRGA(GAE):
    def __init__(self, in_channels, hidden_channels,
                    out_channels, num_layers,
                    dropout):
        encoder = GCNWithAttention(in_channels, hidden_channels,
                hidden_channels, num_layers,
                dropout, 50)
        decoder = LinkPredictor(hidden_channels, hidden_channels, 1, num_layers,
                 dropout)

        super(LRGA, self).__init__(encoder=encoder,
                                    decoder=None)
    def reset_parameters(self):
        self.encoder.reset_parameters()
        # print('type(self.decoder)',type(self.decoder))
        if type(self.decoder) != type(InnerProductDecoder()):
            self.decoder.reset_parameters()

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return z

    def loss(self, x, edge_index, pos_edge_index, neg_edge_index):
        z = self.encode(x, edge_index)
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()
        return pos_loss + neg_loss


class GCNWithAttention(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, k):
        super(GCNWithAttention, self).__init__()
        self.k = k
        self.hidden = hidden_channels
        self.num_layer = num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels,normalize=False))
        self.attention = torch.nn.ModuleList()
        self.dimension_reduce = torch.nn.ModuleList()
        self.attention.append(LowRankAttention(self.k, in_channels, dropout))
        self.dimension_reduce.append(nn.Sequential(nn.Linear(2*(self.k + hidden_channels),\
        hidden_channels),nn.ReLU()))
        self.dimension_reduce[0] = nn.Sequential(nn.Linear(2*self.k + hidden_channels + in_channels,\
        hidden_channels),nn.ReLU())
        self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers-1)])      
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels,normalize=False))
            self.attention.append(LowRankAttention(self.k,hidden_channels, dropout))
            self.dimension_reduce.append(nn.Sequential(nn.Linear(2*(self.k + hidden_channels),\
            hidden_channels)))
        self.dimension_reduce[-1] = nn.Sequential(nn.Linear(2*(self.k + hidden_channels),\
            out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for glob_attention in self.attention:
            glob_attention.apply(weight_init)
        for dim_reduce in self.dimension_reduce:
            dim_reduce.apply(weight_init)
        for batch_norm in self.bn:
            batch_norm.reset_parameters()

    def forward(self, x, adj):
        for i, conv in enumerate(self.convs[:-1]):
            x_local = F.relu(conv(x, adj))
            x_local = F.dropout(x_local, p=self.dropout, training=self.training)
            x_global = self.attention[i](x)
            x = self.dimension_reduce[i](torch.cat((x_global, x_local, x),dim=1))
            x = F.relu(x)
            x = self.bn[i](x)
        x_local = F.relu(self.convs[-1](x, adj))
        x_local = F.dropout(x_local, p=self.dropout, training=self.training)
        x_global = self.attention[-1](x)
        x = self.dimension_reduce[-1](torch.cat((x_global, x_local, x),dim=1))
        return x