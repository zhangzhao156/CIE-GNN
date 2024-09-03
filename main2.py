# -*- coding: utf-8 -*-
# @Time    : 2023/6/1 16:31
# @Author  : zhao
# @File    : main2.py


import argparse
import os
from datetime import datetime
import logging
#from utils.train_graph_utils import train_utils
from collections import Iterable
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support,accuracy_score
import time
import warnings
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
from models2.CausalGCN import GCNNet, GATNet, GINNet, ChebyNet,GraphSage,HoGCN,SGCN
from TEgraph import TEgraph

args = None

def parse_args():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser(description='Train')
    # basic parameters
    parser.add_argument('--model_name', type=str, default='SGCN', help='the name of the model')
    parser.add_argument('--sample_length', type=int, default=100, help='batchsize of the training process')
    parser.add_argument('--overlap', type=int, default=1, help='overlap')
    parser.add_argument('--noise_std', type=float, default=0.1, help='the noise added in test data')
    parser.add_argument('--Input_type', choices=['TD', 'FD','other'],type=str, default='TD', help='the input type decides the length of input')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')

    # Define the tasks
    parser.add_argument('--task', choices=['Node', 'Graph'], type=str,
                        default='Graph', help='Node classification or Graph classification')
    parser.add_argument('--pooltype', choices=['TopKPool', 'EdgePool', 'ASAPool', 'SAGPool'],type=str,
                        default='EdgePool', help='For the Graph classification task')

    # optimization information
    parser.add_argument('--layer_num_last', type=int, default=0, help='the number of last layers which unfreeze')
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='sgd', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='step', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='9', help='the learning rate decay for step and stepLR')

    # save, load and display information
    parser.add_argument('--resume', type=str, default='', help='the directory of the resume training model')
    parser.add_argument('--max_model_num', type=int, default=1, help='the number of most recent models to save')
    parser.add_argument('--max_epochs', type=int, default=100, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=100, help='the interval of log training information')

    ###
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--with_random', type=str2bool, default=True)
    parser.add_argument('--eval_random', type=str2bool, default=False)
    parser.add_argument('--without_node_attention', type=str2bool, default=False)
    parser.add_argument('--without_edge_attention', type=str2bool, default=False)
    parser.add_argument('--fc_num', type=int, default=2)
    parser.add_argument('--cat_or_add', type=str, default="add")
    parser.add_argument('--c', type=float, default=0.001)  ##0.5
    parser.add_argument('--o', type=float, default=1.0)
    parser.add_argument('--co', type=float, default=0.5)
    parser.add_argument('--harf_hidden', type=float, default=0.5)

    args = parser.parse_args()
    return args

# convert a list of list to a list [[],[],[]]->[,,]
def flatten(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x

def train(model, optimizer, loader, device):
    model.train()
    total_loss = 0
    correct = 0

    for it, data in enumerate(loader):
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def train2(model, optimizer, loader, device):
    model.train()
    total_loss = 0
    correct = 0
    data_pred = []
    data_label = []
    for it, data in enumerate(loader):
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
        data_pred.append(pred.cpu().detach().data.tolist())
        data_label.append(data.y.cpu().detach().data.tolist())
    list_data_label = list(flatten(data_label))
    list_data_pred = list(flatten(data_pred))
    all_report = precision_recall_fscore_support(list_data_label, list_data_pred, average='weighted')
    all_precision = all_report[0]
    all_recall = all_report[1]
    all_fscore = all_report[2]
    print('Training all_precision', all_precision, 'all_recall', all_recall, 'all_fscore', all_fscore)
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def eval_acc(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)

def eval_acc2(model, loader, device):
    model.eval()
    correct = 0
    data_pred = []
    data_label = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        data_pred.append(pred.cpu().detach().data.tolist())
        data_label.append(data.y.cpu().detach().data.tolist())
    list_data_label = list(flatten(data_label))
    list_data_pred = list(flatten(data_pred))
    all_report = precision_recall_fscore_support(list_data_label, list_data_pred, average='weighted')
    all_precision = all_report[0]
    all_recall = all_report[1]
    all_fscore = all_report[2]
    print('Testing all_precision', all_precision, 'all_recall', all_recall, 'all_fscore', all_fscore)
    print(classification_report(list_data_label, list_data_pred))
    print(confusion_matrix(list_data_label, list_data_pred))
    return correct / len(loader.dataset)


if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()

    device = torch.device("cuda")
    # Dataset = getattr(datasets, args.data_name)
    datasets = {}
    datasets['train'], datasets['val'] = TEgraph(args.sample_length, args.overlap, args.noise_std).data_preprare()
    # print('datasets[val]',len(datasets['val']))
    # print('datasets[train]', len(datasets['train']))

    InputType = args.Input_type
    if InputType == "TD":
        feature = args.sample_length
    elif InputType == "FD":
        feature = int(args.sample_length / 2)
    elif InputType == "other":
        feature = 1
    else:
        print("The InputType is wrong!!")

    train_accs, test_accs, test_accs_c, test_accs_o = [], [], [], []
    # model = CausalGCN(num_features=feature, num_classes=Dataset.num_classes)
    random_guess = 1.0 / TEgraph.num_classes

    best_test_acc, best_epoch, best_test_acc_c, best_test_acc_o = 0, 0, 0, 0
    train_dataset = datasets['train']
    test_dataset = datasets['val']
    print('train_dataset',len(train_dataset))
    print('test_dataset', len(test_dataset))
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)
    seed = 10
    torch.manual_seed(seed)

    if args.model_name == "GCN":
        model = GCNNet(num_features=feature, num_classes=TEgraph.num_classes, args=args).to(device)
    elif args.model_name == "GIN":
        model = GINNet(num_features=feature, num_classes=TEgraph.num_classes, args=args).to(device)
    elif args.model_name == "GAT":
        model = GATNet(num_features=feature, num_classes=TEgraph.num_classes, args=args).to(device)
    elif args.model_name == "ChebyNet":
        model =ChebyNet(num_features=feature, num_classes=TEgraph.num_classes, args=args).to(device)
    elif args.model_name == "GraphSage":
        model =GraphSage(num_features=feature, num_classes=TEgraph.num_classes, args=args).to(device)
    elif args.model_name == "SGCN":
        model =SGCN(num_features=feature, num_classes=TEgraph.num_classes, args=args).to(device)
    elif args.model_name == "HoGCN":
        model =HoGCN(num_features=feature, num_classes=TEgraph.num_classes, args=args).to(device)
    else:
        print('MODEL ERROR')
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.max_epochs + 1):
        if epoch>95:
            train_loss, train_acc = train2(model, optimizer, train_loader, device)
            test_acc = eval_acc2(model, test_loader, device)
        else:
            test_acc = eval_acc(model, test_loader, device)
            train_loss, train_acc = train(model, optimizer, train_loader, device)

        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch

        print(
            "NoCausal | Epoch:[{}/{}] Loss:[{:.4f}] Train:[{:.4f}] Test:[{:.2f}]  (RG:{:.2f}) | Best Test:[{:.2f}] at Epoch:[{}] "
            .format(
                    epoch, args.max_epochs,
                    train_loss,
                    train_acc * 100,
                    test_acc * 100,
                    random_guess * 100,
                    best_test_acc * 100,
                    best_epoch))

    print(
        "syd: NoCausal | Best Test:[{:.2f}] at epoch [{}] | (RG:{:.2f})"
        .format(
                best_test_acc * 100,
                best_epoch,
                random_guess * 100))


