# -*- coding: utf-8 -*-

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import global_mean_pool, global_add_pool, GINConv, GATConv,ChebConv, SAGEConv,SGConv,GraphConv, BatchNorm, TopKPooling, EdgePooling, ASAPooling, SAGPooling
from models2.gcn_conv import GCNConv
import random
import pdb


class CLUB(torch.nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim]
    '''

    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        # print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = torch.nn.Sequential(torch.nn.Linear(x_dim, hidden_size // 2),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hidden_size // 2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = torch.nn.Sequential(torch.nn.Linear(x_dim, hidden_size // 2),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(hidden_size // 2, y_dim),
                                            torch.nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples) ** 2 / 2. / logvar.exp()

        prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2. / logvar.exp()

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def loglikeli(self, x_samples, y_samples):  # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class CausalGCN(torch.nn.Module):
    """GCN with BN and residual connection."""

    def __init__(self, num_features,
                 num_classes, args,
                 gfn=False,
                 edge_norm=True):
        super(CausalGCN, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args
        self.global_pool = global_add_pool
        # self.dropout = dropout
        self.with_random = args.with_random # True
        self.without_node_attention = args.without_node_attention # False
        self.without_edge_attention = args.without_edge_attention # False
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = num_features
        self.num_classes = num_classes
        hidden_out = num_classes
        self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True)  # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GConv(hidden, hidden))

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        self.node_att_mlp = nn.Linear(hidden, 2)
        self.bnc = BatchNorm1d(hidden)
        self.bno = BatchNorm1d(hidden)
        self.context_convs = GConv(hidden, hidden)
        self.objects_convs = GConv(hidden, hidden)

        # context mlp
        self.fc1_bn_c = BatchNorm1d(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random=True):

        x = data.x if data.x is not None else data.feat
        # if eval_stage:
        #     x = x + torch.randn(x.size()).cuda() * 0.1
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index
        # print(edge_index.size())
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))

        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))

        edge_rep = torch.cat([x[row], x[col]], dim=-1)

        if self.without_edge_attention:
            edge_att = 0.5 * torch.ones(edge_rep.shape[0], 2).cuda()
        else:
            edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        if self.without_node_attention:
            node_att = 0.5 * torch.ones(x.shape[0], 2).cuda()
        else:
            node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        xc_att = node_att[:, 0].view(-1, 1) * x
        xo_att = node_att[:, 1].view(-1, 1) * x
        xc = F.relu(self.context_convs(self.bnc(xc_att), edge_index, edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo_att), edge_index, edge_weight_o))

        xc = self.global_pool(xc, batch)
        xo = self.global_pool(xo, batch)

        xc_logis = self.context_readout_layer(xc)
        xo_logis = self.objects_readout_layer(xo)
        xco_logis = self.random_readout_layer(xc, xo, eval_random=eval_random)

        return xc_logis, xo_logis, xco_logis, xc_att, xo_att,edge_att,node_att

    def context_readout_layer(self, x):

        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x):

        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def random_readout_layer(self, xc, xo, eval_random):

        num = xc.shape[0]
        l = [i for i in range(num)]
        if self.with_random:
            if eval_random:
                random.shuffle(l)
        random_idx = torch.tensor(l)
        if self.args.cat_or_add == "cat":
            x = torch.cat((xc[random_idx], xo), dim=1)
        else:
            x = xc[random_idx] + xo

        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis
