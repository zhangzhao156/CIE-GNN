# -*- coding: utf-8 -*-
# @Time    : 2023/5/10 20:48
# @Author  : zhao
# @File    : CausalGCN.py

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import global_mean_pool, global_add_pool, GINConv, GATConv,ChebConv, SAGEConv,SGConv,GraphConv, BatchNorm, TopKPooling, EdgePooling, ASAPooling, SAGPooling
from models2.gcn_conv import GCNConv
import random
import pdb


# def pearson_correlation_coefficient(tensor1, tensor2):
#     # 计算两个tensor的平均值
#     tensor1_mean = tensor1.mean(dim=1, keepdim=True)
#     tensor2_mean = tensor2.mean(dim=1, keepdim=True)
#     # 计算两个tensor的方差
#     tensor1_std = tensor1.std(dim=1, keepdim=True)
#     tensor2_std = tensor2.std(dim=1, keepdim=True)
#     # 计算两个tensor的协方差
#     covariance = ((tensor1 - tensor1_mean) * (tensor2 - tensor2_mean)).mean(dim=1, keepdim=True)
#     # 计算Pearson相关系数
#     pearson_corr = covariance / (tensor1_std * tensor2_std)
#     return pearson_corr

def pearson_correlation_coefficient(tensor1, tensor2):
    # 计算两个 tensor 的平均值
    tensor1_mean = tensor1.mean()
    tensor2_mean = tensor2.mean()
    # 计算两个 tensor 与其对应平均值的差
    tensor1_diff = tensor1 - tensor1_mean
    tensor2_diff = tensor2 - tensor2_mean
    # 计算 Pearson 相关系数的分子，即协方差
    covariance = (tensor1_diff * tensor2_diff).sum()
    # 计算 Pearson 相关系数的分母，即两个 tensor 的标准差的乘积
    denominator = torch.sqrt((tensor1_diff ** 2).sum()) * torch.sqrt((tensor2_diff ** 2).sum())
    # 计算 Pearson 相关系数
    pearson_corr = covariance / denominator
    return pearson_corr

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


# class CausalGCN(torch.nn.Module):
#     """GCN with BN and residual connection."""
#
#     def __init__(self, num_features,
#                  num_classes, args,
#                  gfn=False,
#                  edge_norm=True):
#         super(CausalGCN, self).__init__()
#         num_conv_layers = args.layers
#         hidden = args.hidden
#         self.args = args
#         self.global_pool = global_add_pool
#         # self.dropout = dropout
#         self.with_random = args.with_random # True
#         self.without_node_attention = args.without_node_attention # False
#         self.without_edge_attention = args.without_edge_attention # False
#         GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)
#
#         hidden_in = num_features
#         self.num_classes = num_classes
#         hidden_out = num_classes
#         self.fc_num = args.fc_num
#         self.bn_feat = BatchNorm1d(hidden_in)
#         self.conv_feat = GCNConv(hidden_in, hidden, gfn=True)  # linear transform
#         self.bns_conv = torch.nn.ModuleList()
#         self.convs = torch.nn.ModuleList()
#
#         for i in range(num_conv_layers):
#             self.bns_conv.append(BatchNorm1d(hidden))
#             self.convs.append(GConv(hidden, hidden))
#
#         self.edge_att_mlp = nn.Linear(hidden * 2, 2)
#         self.node_att_mlp = nn.Linear(hidden, 2)
#         self.bnc = BatchNorm1d(hidden)
#         self.bno = BatchNorm1d(hidden)
#         self.context_convs = GConv(hidden, hidden)
#         self.objects_convs = GConv(hidden, hidden)
#
#         # context mlp
#         self.fc1_bn_c = BatchNorm1d(hidden)
#         self.fc1_c = Linear(hidden, hidden)
#         self.fc2_bn_c = BatchNorm1d(hidden)
#         self.fc2_c = Linear(hidden, hidden_out)
#         # object mlp
#         self.fc1_bn_o = BatchNorm1d(hidden)
#         self.fc1_o = Linear(hidden, hidden)
#         self.fc2_bn_o = BatchNorm1d(hidden)
#         self.fc2_o = Linear(hidden, hidden_out)
#         # random mlp
#         if self.args.cat_or_add == "cat":
#             self.fc1_bn_co = BatchNorm1d(hidden * 2)
#             self.fc1_co = Linear(hidden * 2, hidden)
#             self.fc2_bn_co = BatchNorm1d(hidden)
#             self.fc2_co = Linear(hidden, hidden_out)
#
#         elif self.args.cat_or_add == "add":
#             self.fc1_bn_co = BatchNorm1d(hidden)
#             self.fc1_co = Linear(hidden, hidden)
#             self.fc2_bn_co = BatchNorm1d(hidden)
#             self.fc2_co = Linear(hidden, hidden_out)
#         else:
#             assert False
#
#         # BN initialization.
#         for m in self.modules():
#             if isinstance(m, (torch.nn.BatchNorm1d)):
#                 torch.nn.init.constant_(m.weight, 1)
#                 torch.nn.init.constant_(m.bias, 0.0001)
#
#     def forward(self, data, eval_random=True):
#
#         x = data.x if data.x is not None else data.feat
#         # if eval_stage:
#         #     x = x + torch.randn(x.size()).cuda() * 0.1
#         edge_index, batch = data.edge_index, data.batch
#         row, col = edge_index
#         # print(edge_index.size())
#         x = self.bn_feat(x)
#         x = F.relu(self.conv_feat(x, edge_index))
#
#         for i, conv in enumerate(self.convs):
#             x = self.bns_conv[i](x)
#             x = F.relu(conv(x, edge_index))
#
#         edge_rep = torch.cat([x[row], x[col]], dim=-1)
#
#         if self.without_edge_attention:
#             edge_att = 0.5 * torch.ones(edge_rep.shape[0], 2).cuda()
#         else:
#             edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
#         edge_weight_c = edge_att[:, 0]
#         edge_weight_o = edge_att[:, 1]
#
#         if self.without_node_attention:
#             node_att = 0.5 * torch.ones(x.shape[0], 2).cuda()
#         else:
#             node_att = F.softmax(self.node_att_mlp(x), dim=-1)
#         xc_att = node_att[:, 0].view(-1, 1) * x
#         xo_att = node_att[:, 1].view(-1, 1) * x
#         xc = F.relu(self.context_convs(self.bnc(xc_att), edge_index, edge_weight_c))
#         xo = F.relu(self.objects_convs(self.bno(xo_att), edge_index, edge_weight_o))
#
#         xc = self.global_pool(xc, batch)
#         xo = self.global_pool(xo, batch)
#
#         xc_logis = self.context_readout_layer(xc)
#         xo_logis = self.objects_readout_layer(xo)
#         xco_logis = self.random_readout_layer(xc, xo, eval_random=eval_random)
#
#         return xc_logis, xo_logis, xco_logis, xc_att, xo_att,edge_att,node_att
#
#     def context_readout_layer(self, x):
#
#         x = self.fc1_bn_c(x)
#         x = self.fc1_c(x)
#         x = F.relu(x)
#         x = self.fc2_bn_c(x)
#         x = self.fc2_c(x)
#         x_logis = F.log_softmax(x, dim=-1)
#         return x_logis
#
#     def objects_readout_layer(self, x):
#
#         x = self.fc1_bn_o(x)
#         x = self.fc1_o(x)
#         x = F.relu(x)
#         x = self.fc2_bn_o(x)
#         x = self.fc2_o(x)
#         x_logis = F.log_softmax(x, dim=-1)
#         return x_logis
#
#     def random_readout_layer(self, xc, xo, eval_random):
#
#         num = xc.shape[0]
#         l = [i for i in range(num)]
#         if self.with_random:
#             if eval_random:
#                 random.shuffle(l)
#         random_idx = torch.tensor(l)
#         if self.args.cat_or_add == "cat":
#             x = torch.cat((xc[random_idx], xo), dim=1)
#         else:
#             x = xc[random_idx] + xo
#
#         x = self.fc1_bn_co(x)
#         x = self.fc1_co(x)
#         x = F.relu(x)
#         x = self.fc2_bn_co(x)
#         x = self.fc2_co(x)
#         x_logis = F.log_softmax(x, dim=-1)
#         return x_logis

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

        self.edge_att_mlp_c = nn.Linear(hidden * 2, 1)
        self.node_att_mlp_c = nn.Linear(hidden, 1)
        self.edge_att_mlp_o = nn.Linear(hidden * 2, 1)
        self.node_att_mlp_o = nn.Linear(hidden, 1)
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
            edge_att_c = 0.5 * torch.ones(edge_rep.shape[0], 1).cuda()
            edge_att_o = 0.5 * torch.ones(edge_rep.shape[0], 1).cuda()
        else:
            edge_att_c = F.softmax(self.edge_att_mlp_c(edge_rep), dim=-1)
            edge_att_o = F.softmax(self.edge_att_mlp_o(edge_rep), dim=-1)
        edge_weight_c = edge_att_c[:, 0]
        edge_weight_o = edge_att_o[:, 0]

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

        return xc_logis, xo_logis, xco_logis, xc_att, xo_att,edge_att_c, node_att_c

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


# class CausalGCN_w(torch.nn.Module):
#     """GCN with BN and residual connection."""
#
#     def __init__(self, num_features,
#                  num_classes, args,
#                  gfn=False,
#                  edge_norm=True):
#         super(CausalGCN_w, self).__init__()
#         num_conv_layers = args.layers
#         hidden = args.hidden
#         self.args = args
#         self.global_pool = global_add_pool
#         # self.dropout = dropout
#         self.with_random = args.with_random # True
#         self.without_node_attention = args.without_node_attention # False
#         self.without_edge_attention = args.without_edge_attention # False
#         GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)
#
#         hidden_in = num_features
#         self.num_classes = num_classes
#         hidden_out = num_classes
#         self.fc_num = args.fc_num
#         self.bn_feat = BatchNorm1d(hidden_in)
#         self.conv_feat = GCNConv(hidden_in, hidden, gfn=True)  # linear transform
#         self.bns_conv = torch.nn.ModuleList()
#         self.convs = torch.nn.ModuleList()
#
#         for i in range(num_conv_layers):
#             self.bns_conv.append(BatchNorm1d(hidden))
#             self.convs.append(GConv(hidden, hidden))
#
#         self.edge_att_mlp = nn.Linear(hidden * 2, 1)
#         self.node_att_mlp = nn.Linear(hidden, 1)
#         self.bnc = BatchNorm1d(hidden)
#         self.bno = BatchNorm1d(hidden)
#         self.context_convs = GConv(hidden, hidden)
#         self.objects_convs = GConv(hidden, hidden)
#
#         # context mlp
#         self.fc1_bn_c = BatchNorm1d(hidden)
#         self.fc1_c = Linear(hidden, hidden)
#         self.fc2_bn_c = BatchNorm1d(hidden)
#         self.fc2_c = Linear(hidden, hidden_out)
#         # object mlp
#         self.fc1_bn_o = BatchNorm1d(hidden)
#         self.fc1_o = Linear(hidden, hidden)
#         self.fc2_bn_o = BatchNorm1d(hidden)
#         self.fc2_o = Linear(hidden, hidden_out)
#         # random mlp
#         if self.args.cat_or_add == "cat":
#             self.fc1_bn_co = BatchNorm1d(hidden * 2)
#             self.fc1_co = Linear(hidden * 2, hidden)
#             self.fc2_bn_co = BatchNorm1d(hidden)
#             self.fc2_co = Linear(hidden, hidden_out)
#
#         elif self.args.cat_or_add == "add":
#             self.fc1_bn_co = BatchNorm1d(hidden)
#             self.fc1_co = Linear(hidden, hidden)
#             self.fc2_bn_co = BatchNorm1d(hidden)
#             self.fc2_co = Linear(hidden, hidden_out)
#         else:
#             assert False
#
#         # BN initialization.
#         for m in self.modules():
#             if isinstance(m, (torch.nn.BatchNorm1d)):
#                 torch.nn.init.constant_(m.weight, 1)
#                 torch.nn.init.constant_(m.bias, 0.0001)
#
#     def forward(self, data, eval_random=True):
#
#         x = data.x if data.x is not None else data.feat
#         # if eval_stage:
#         #     x = x + torch.randn(x.size()).cuda() * 0.1
#         edge_index, batch = data.edge_index, data.batch
#         row, col = edge_index
#         # print(edge_index.size())
#         x = self.bn_feat(x)
#         x = F.relu(self.conv_feat(x, edge_index))
#
#         for i, conv in enumerate(self.convs):
#             x = self.bns_conv[i](x)
#             x = F.relu(conv(x, edge_index))
#
#         edge_rep = torch.cat([x[row], x[col]], dim=-1)
#
#         if self.without_edge_attention:
#             edge_att = 0.5 * torch.ones(edge_rep.shape[0], 2).cuda()
#         else:
#             edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
#         edge_weight_c = edge_att[:, 0]
#         edge_weight_o = edge_att[:, 1]
#
#         if self.without_node_attention:
#             node_att = 0.5 * torch.ones(x.shape[0], 2).cuda()
#         else:
#             node_att = F.softmax(self.node_att_mlp(x), dim=-1)
#         xc_att = node_att[:, 0].view(-1, 1) * x
#         xo_att = node_att[:, 1].view(-1, 1) * x
#         xc = F.relu(self.context_convs(self.bnc(xc_att), edge_index, edge_weight_c))
#         xo = F.relu(self.objects_convs(self.bno(xo_att), edge_index, edge_weight_o))
#
#         xc = self.global_pool(xc, batch)
#         xo = self.global_pool(xo, batch)
#
#         xc_logis = self.context_readout_layer(xc)
#         xo_logis = self.objects_readout_layer(xo)
#         xco_logis, corrw = self.random_readout_layer(xc, xo, eval_random=eval_random)
#
#         return xc_logis, xo_logis, xco_logis, xc_att, xo_att,corrw
#
#     def context_readout_layer(self, x):
#
#         x = self.fc1_bn_c(x)
#         x = self.fc1_c(x)
#         x = F.relu(x)
#         x = self.fc2_bn_c(x)
#         x = self.fc2_c(x)
#         x_logis = F.log_softmax(x, dim=-1)
#         return x_logis
#
#     def objects_readout_layer(self, x):
#
#         x = self.fc1_bn_o(x)
#         x = self.fc1_o(x)
#         x = F.relu(x)
#         x = self.fc2_bn_o(x)
#         x = self.fc2_o(x)
#         x_logis = F.log_softmax(x, dim=-1)
#         return x_logis
#
#     def random_readout_layer(self, xc, xo, eval_random):
#
#         num = xc.shape[0]
#         l = [i for i in range(num)]
#         if self.with_random:
#             if eval_random:
#                 random.shuffle(l)
#         random_idx = torch.tensor(l)
#         if self.args.cat_or_add == "cat":
#             x = torch.cat((xc[random_idx], xo), dim=1)
#         else:
#             x = xc[random_idx] + xo
#
#         corrw = pearson_correlation_coefficient(xc[random_idx], xo)
#         x = self.fc1_bn_co(x)
#         x = self.fc1_co(x)
#         x = F.relu(x)
#         x = self.fc2_bn_co(x)
#         x = self.fc2_co(x)
#         x_logis = F.log_softmax(x, dim=-1)
#         return x_logis,corrw

class GCNNet(torch.nn.Module):
    """GCN with BN and residual connection."""

    def __init__(self, num_features,
                 num_classes,
                 args,
                 gfn=False,
                 edge_norm=True):
        super(GCNNet, self).__init__()
        self.args = args
        num_conv_layers = args.layers
        hidden = args.hidden
        num_fc_layers = args.fc_num
        self.global_pool = global_add_pool
        # self.dropout = dropout
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = num_features
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True)  # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GConv(hidden, hidden))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, num_classes)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data):

        x = data.x if data.x is not None else data.feat
        # if eval_stage:
        #     x = x + torch.randn(x.size()).cuda() * 0.1
        edge_index, batch = data.edge_index, data.batch

        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))

        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))

        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))

        x = self.bn_hidden(x)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)


class GCNLSTMNet(torch.nn.Module):
    """GCN with BN and residual connection."""

    def __init__(self, num_features,
                 num_classes,
                 args,
                 gfn=False,
                 edge_norm=True):
        super(GCNLSTMNet, self).__init__()
        self.args = args
        num_conv_layers = args.layers
        hidden = args.hidden
        num_fc_layers = args.fc_num
        self.global_pool = global_add_pool
        # self.dropout = dropout
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = num_features
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True)  # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GConv(hidden, hidden))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        self.lstm = torch.nn.LSTM(hidden, hidden,
                            1, True)

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, num_classes)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data):

        x = data.x if data.x is not None else data.feat
        # if eval_stage:
        #     x = x + torch.randn(x.size()).cuda() * 0.1
        edge_index, batch = data.edge_index, data.batch

        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))

        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))
        # print('x', x.size())
        x = self.global_pool(x, batch)
        # print('x',x.size())
        x, (hn, cn) = self.lstm(x)
        x = x.view(x.shape[0], -1)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))

        x = self.bn_hidden(x)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)



class CausalGIN(torch.nn.Module):
    """GCN with BN and residual connection."""

    def __init__(self, num_features,
                 num_classes, args,
                 gfn=False,
                 edge_norm=True):
        super(CausalGIN, self).__init__()

        hidden = args.hidden
        num_conv_layers = args.layers
        self.args = args
        self.global_pool = global_add_pool
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
            self.convs.append(GINConv(
                Sequential(
                    Linear(hidden, hidden),
                    BatchNorm1d(hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    ReLU())))

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

    def forward(self, data, eval_random=True, train_type="base"):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        node_weight_c = node_att[:, 0]
        node_weight_o = node_att[:, 1]

        xc = node_weight_c.view(-1, 1) * x
        xo = node_weight_o.view(-1, 1) * x
        xc = F.relu(self.context_convs(self.bnc(xc), edge_index, edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o))

        xc = self.global_pool(xc, batch)
        xo = self.global_pool(xo, batch)

        xc_logis = self.context_readout_layer(xc)
        xco_logis = self.random_readout_layer(xc, xo, eval_random=eval_random)
        # return xc_logis, xo_logis, xco_logis
        xo_logis = self.objects_readout_layer(xo, train_type)
        return xc_logis, xo_logis, xco_logis, xc, xo,edge_att,node_att

    def context_readout_layer(self, x):

        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x, train_type):

        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        if train_type == "irm":
            return x, x_logis
        else:
            return x_logis

    def random_readout_layer(self, xc, xo, eval_random):

        num = xc.shape[0]
        l = [i for i in range(num)]
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


class GINNet(torch.nn.Module):
    def __init__(self, num_features,
                 num_classes,
                 args):
        super(GINNet, self).__init__()
        self.args = args
        num_conv_layers = args.layers
        hidden = args.hidden
        num_fc_layers = args.fc_num
        self.global_pool = global_add_pool
        hidden_in = num_features
        hidden_out = num_classes
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True)  # linear transform

        self.convs = torch.nn.ModuleList()
        for i in range(num_conv_layers):
            self.convs.append(GINConv(
                Sequential(Linear(hidden, hidden),
                           BatchNorm1d(hidden),
                           ReLU(),
                           Linear(hidden, hidden),
                           ReLU())))

        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, hidden_out)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data):
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        # x, edge_index, batch = data.feat, data.edge_index, data.batch
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))
        x = self.bn_hidden(x)
        x = self.lin_class(x)

        prediction = F.log_softmax(x, dim=-1)
        return prediction

class CausalGAT(torch.nn.Module):
    def __init__(self, num_features,
                 num_classes,
                 args,
                 head=4,
                 dropout=0.2):
        super(CausalGAT, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args
        self.global_pool = global_add_pool
        self.dropout = dropout
        GConv = partial(GCNConv, edge_norm=True, gfn=False)

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
            self.convs.append(GATConv(hidden, int(hidden / head), heads=head, dropout=dropout))

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
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))

        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))

        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        xc = node_att[:, 0].view(-1, 1) * x
        xo = node_att[:, 1].view(-1, 1) * x
        xc = F.relu(self.context_convs(self.bnc(xc), edge_index, edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o))

        xc = self.global_pool(xc, batch)
        xo = self.global_pool(xo, batch)

        xc_logis = self.context_readout_layer(xc)
        xo_logis = self.objects_readout_layer(xo)
        xco_logis = self.random_readout_layer(xc, xo, eval_random=eval_random)
        return xc_logis, xo_logis, xco_logis, xc, xo,edge_att,node_att

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





class GATNet(torch.nn.Module):
    def __init__(self, num_features,
                 num_classes,
                 args,
                 head=8,
                 dropout=0.2):

        super(GATNet, self).__init__()
        self.args = args
        num_conv_layers = args.layers
        hidden = args.hidden
        num_fc_layers = args.fc_num
        self.global_pool = global_add_pool
        self.dropout = dropout
        hidden_in = num_features
        hidden_out = num_classes

        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True)  # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GATConv(hidden, int(hidden / head), heads=head, dropout=dropout))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, hidden_out)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch

        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))

        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))

        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))

        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

class ChebyNet(torch.nn.Module):
    def __init__(self, num_features, num_classes, args):
        super(ChebyNet, self).__init__()

        self.args = args
        hidden = args.hidden
        self.GConv1 = ChebConv(num_features,hidden,K=1)
        self.bn1 = BatchNorm(hidden)

        self.GConv2 = ChebConv(hidden,hidden,K=1)
        self.bn2 = BatchNorm(hidden)

        self.fc = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Sequential(nn.Linear(hidden, num_classes))

    def forward(self, data):
        x, edge_index, batch= data.x, data.edge_index, data.batch

        x = self.GConv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.GConv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = self.fc(x)
        x = self.dropout(x)
        x = self.fc1(x)

        return F.log_softmax(x, dim=-1)


class GraphSage(torch.nn.Module):
    def __init__(self, num_features, num_classes, args):
        super(GraphSage, self).__init__()
        self.args = args
        hidden = args.hidden
        self.GConv1 = SAGEConv(num_features,hidden)
        self.bn1 = BatchNorm(hidden)

        self.GConv2 = SAGEConv(hidden,hidden)
        self.bn2 = BatchNorm(hidden)

        self.fc = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Sequential(nn.Linear(hidden, num_classes))

    def forward(self, data):
        x, edge_index, batch= data.x, data.edge_index, data.batch

        x = self.GConv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.GConv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.fc1(x)

        return F.log_softmax(x, dim=-1)



class HoGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes,args):
        super(HoGCN, self).__init__()
        self.args = args
        hidden = args.hidden
        self.GConv1 = GraphConv(num_features,hidden)
        self.bn1 = BatchNorm(hidden)

        self.GConv2 = GraphConv(hidden,hidden)
        self.bn2 = BatchNorm(hidden)

        self.fc = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Sequential(nn.Linear(hidden, num_classes))

    def forward(self, data):
        x, edge_index, batch= data.x, data.edge_index, data.batch

        x = self.GConv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.GConv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.fc1(x)

        return F.log_softmax(x, dim=-1)

class SGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes,args):
        super(SGCN, self).__init__()
        self.args = args
        hidden = args.hidden
        self.GConv1 = SGConv(num_features,hidden)
        self.bn1 = BatchNorm(hidden)

        self.GConv2 = SGConv(hidden,hidden)
        self.bn2 = BatchNorm(hidden)

        self.fc = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Sequential(nn.Linear(hidden, num_classes))

    def forward(self, data):
        x, edge_index, batch= data.x, data.edge_index, data.batch

        x = self.GConv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.GConv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.fc1(x)

        return F.log_softmax(x, dim=-1)


# class ChebyNet(torch.nn.Module):
#     def __init__(self, num_features, num_classes, args):
#         super(ChebyNet, self).__init__()
#
#         self.args = args
#         self.pool1, self.pool2 = self.poollayer(args.pooltype)
#
#         self.GConv1 = ChebConv(num_features,128,K=1)
#         self.bn1 = BatchNorm(128)
#
#         self.GConv2 = ChebConv(128,128,K=1)
#         self.bn2 = BatchNorm(128)
#
#         self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True))
#         self.dropout = nn.Dropout(0.2)
#         self.fc1 = nn.Sequential(nn.Linear(64, num_classes))
#
#     def forward(self, data):
#         x, edge_index, batch= data.x, data.edge_index, data.batch
#
#         x = self.GConv1(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x, edge_index, batch = self.poolresult(self.pool1,self.args.pooltype,x, edge_index, batch)
#         x1 = global_mean_pool(x, batch)
#
#         x = self.GConv2(x, edge_index)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x, edge_index, batch = self.poolresult(self.pool2, self.args.pooltype, x, edge_index, batch)
#         x2 = global_mean_pool(x, batch)
#
#         x = x1 + x2
#
#         x = self.fc(x)
#         x = self.dropout(x)
#         x = self.fc1(x)
#
#         return F.log_softmax(x, dim=-1)
#
#     def poollayer(self, pooltype):
#
#         self.pooltype = pooltype
#
#         if self.pooltype == 'TopKPool':
#             self.pool1 = TopKPooling(128)
#             self.pool2 = TopKPooling(128)
#         elif self.pooltype == 'EdgePool':
#             self.pool1 = EdgePooling(128)
#             self.pool2 = EdgePooling(128)
#         elif self.pooltype == 'ASAPool':
#             self.pool1 = ASAPooling(128)
#             self.pool2 = ASAPooling(128)
#         elif self.pooltype == 'SAGPool':
#             self.pool1 = SAGPooling(128)
#             self.pool2 = SAGPooling(128)
#         else:
#             print('Such graph pool method is not implemented!!')
#
#         return self.pool1, self.pool2
#
#     def poolresult(self,pool,pooltype,x,edge_index,batch):
#
#         self.pool = pool
#
#         if pooltype == 'TopKPool':
#             x, edge_index, _, batch, _, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
#         elif pooltype == 'EdgePool':
#             x, edge_index, batch, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
#         elif pooltype == 'ASAPool':
#             x, edge_index, _, batch, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
#         elif pooltype == 'SAGPool':
#             x, edge_index, _, batch, _, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
#         else:
#             print('Such graph pool method is not implemented!!')
#
#         return x, edge_index, batch
#
#
# class GraphSage(torch.nn.Module):
#     def __init__(self, num_features, num_classes, args):
#         super(GraphSage, self).__init__()
#         self.args = args
#         self.pool1, self.pool2 = self.poollayer(args.pooltype)
#         hidden = args.hidden
#         self.GConv1 = SAGEConv(num_features,128)
#         self.bn1 = BatchNorm(128)
#
#         self.GConv2 = SAGEConv(128,128)
#         self.bn2 = BatchNorm(128)
#
#         self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True))
#         self.dropout = nn.Dropout(0.2)
#         self.fc1 = nn.Sequential(nn.Linear(64, num_classes))
#
#     def forward(self, data):
#         x, edge_index, batch= data.x, data.edge_index, data.batch
#
#         x = self.GConv1(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x, edge_index, batch = self.poolresult(self.pool1, self.args.pooltype, x, edge_index, batch)
#         x1 = global_mean_pool(x, batch)
#
#         x = self.GConv2(x, edge_index)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x, edge_index, batch = self.poolresult(self.pool2, self.args.pooltype, x, edge_index, batch)
#         x2 = global_mean_pool(x, batch)
#         x = x1 + x2
#         x = self.fc(x)
#         x = self.dropout(x)
#         x = self.fc1(x)
#
#         return F.log_softmax(x, dim=-1)
#
#     def poollayer(self, pooltype):
#
#         self.pooltype = pooltype
#
#         if self.pooltype == 'TopKPool':
#             self.pool1 = TopKPooling(128)
#             self.pool2 = TopKPooling(128)
#         elif self.pooltype == 'EdgePool':
#             self.pool1 = EdgePooling(128)
#             self.pool2 = EdgePooling(128)
#         elif self.pooltype == 'ASAPool':
#             self.pool1 = ASAPooling(128)
#             self.pool2 = ASAPooling(128)
#         elif self.pooltype == 'SAGPool':
#             self.pool1 = SAGPooling(128)
#             self.pool2 = SAGPooling(128)
#         else:
#             print('Such graph pool method is not implemented!!')
#
#         return self.pool1, self.pool2
#
#     def poolresult(self,pool,pooltype,x,edge_index,batch):
#
#         self.pool = pool
#
#         if pooltype == 'TopKPool':
#             x, edge_index, _, batch, _, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
#         elif pooltype == 'EdgePool':
#             x, edge_index, batch, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
#         elif pooltype == 'ASAPool':
#             x, edge_index, _, batch, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
#         elif pooltype == 'SAGPool':
#             x, edge_index, _, batch, _, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
#         else:
#             print('Such graph pool method is not implemented!!')
#
#         return x, edge_index, batch
#
# class HoGCN(torch.nn.Module):
#     def __init__(self, num_features, num_classes,args):
#         super(HoGCN, self).__init__()
#         self.args = args
#         self.pool1, self.pool2 = self.poollayer(args.pooltype)
#         hidden = args.hidden
#         self.GConv1 = GraphConv(num_features,128)
#         self.bn1 = BatchNorm(128)
#
#         self.GConv2 = GraphConv(128,128)
#         self.bn2 = BatchNorm(128)
#
#         self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True))
#         self.dropout = nn.Dropout(0.2)
#         self.fc1 = nn.Sequential(nn.Linear(64, num_classes))
#
#
#     def forward(self, data):
#         x, edge_index, batch= data.x, data.edge_index, data.batch
#
#         x = self.GConv1(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x, edge_index, batch = self.poolresult(self.pool1,self.args.pooltype,x, edge_index, batch)
#         x1 = global_mean_pool(x, batch)
#
#         x = self.GConv2(x, edge_index)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x, edge_index, batch = self.poolresult(self.pool2, self.args.pooltype, x, edge_index, batch)
#         x2 = global_mean_pool(x, batch)
#         x = x1 + x2
#         x = self.fc(x)
#         x = self.dropout(x)
#         x = self.fc1(x)
#
#         return F.log_softmax(x, dim=-1)
#
#     def poollayer(self, pooltype):
#
#         self.pooltype = pooltype
#
#         if self.pooltype == 'TopKPool':
#             self.pool1 = TopKPooling(128)
#             self.pool2 = TopKPooling(128)
#         elif self.pooltype == 'EdgePool':
#             self.pool1 = EdgePooling(128)
#             self.pool2 = EdgePooling(128)
#         elif self.pooltype == 'ASAPool':
#             self.pool1 = ASAPooling(128)
#             self.pool2 = ASAPooling(128)
#         elif self.pooltype == 'SAGPool':
#             self.pool1 = SAGPooling(128)
#             self.pool2 = SAGPooling(128)
#         else:
#             print('Such graph pool method is not implemented!!')
#
#         return self.pool1, self.pool2
#
#     def poolresult(self,pool,pooltype,x,edge_index,batch):
#
#         self.pool = pool
#
#         if pooltype == 'TopKPool':
#             x, edge_index, _, batch, _, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
#         elif pooltype == 'EdgePool':
#             x, edge_index, batch, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
#         elif pooltype == 'ASAPool':
#             x, edge_index, _, batch, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
#         elif pooltype == 'SAGPool':
#             x, edge_index, _, batch, _, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
#         else:
#             print('Such graph pool method is not implemented!!')
#
#         return x, edge_index, batch
#
# class SGCN(torch.nn.Module):
#     def __init__(self, feature, out_channel,args):
#         super(SGCN, self).__init__()
#         self.args = args
#         self.pool1, self.pool2 = self.poollayer(args.pooltype)
#
#         self.GConv1 = SGConv(feature,128)
#         self.bn1 = BatchNorm(128)
#
#         self.GConv2 = SGConv(128,128)
#         self.bn2 = BatchNorm(128)
#
#         self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True))
#         self.dropout = nn.Dropout(0.2)
#         self.fc1 = nn.Sequential(nn.Linear(64, out_channel))
#
#     def forward(self, data):
#         x, edge_index, batch= data.x, data.edge_index, data.batch
#
#         x = self.GConv1(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x, edge_index, batch = self.poolresult(self.pool1,self.args.pooltype,x, edge_index, batch)
#         x1 = global_mean_pool(x, batch)
#
#         x = self.GConv2(x, edge_index)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x, edge_index, batch = self.poolresult(self.pool2, self.args.pooltype, x, edge_index, batch)
#         x2 = global_mean_pool(x, batch)
#         x = x1 + x2
#         x = self.fc(x)
#         x = self.dropout(x)
#         x = self.fc1(x)
#
#         return F.log_softmax(x, dim=-1)
#
#     def poollayer(self, pooltype):
#
#         self.pooltype = pooltype
#
#         if self.pooltype == 'TopKPool':
#             self.pool1 = TopKPooling(128)
#             self.pool2 = TopKPooling(128)
#         elif self.pooltype == 'EdgePool':
#             self.pool1 = EdgePooling(128)
#             self.pool2 = EdgePooling(128)
#         elif self.pooltype == 'ASAPool':
#             self.pool1 = ASAPooling(128)
#             self.pool2 = ASAPooling(128)
#         elif self.pooltype == 'SAGPool':
#             self.pool1 = SAGPooling(128)
#             self.pool2 = SAGPooling(128)
#         else:
#             print('Such graph pool method is not implemented!!')
#
#         return self.pool1, self.pool2
#
#     def poolresult(self,pool,pooltype,x,edge_index,batch):
#
#         self.pool = pool
#
#         if pooltype == 'TopKPool':
#             x, edge_index, _, batch, _, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
#         elif pooltype == 'EdgePool':
#             x, edge_index, batch, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
#         elif pooltype == 'ASAPool':
#             x, edge_index, _, batch, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
#         elif pooltype == 'SAGPool':
#             x, edge_index, _, batch, _, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
#         else:
#             print('Such graph pool method is not implemented!!')
#
#         return x, edge_index, batch
