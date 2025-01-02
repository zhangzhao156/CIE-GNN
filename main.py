import argparse
import os
from datetime import datetime
# from utils.logger import setlogger
import logging
# from utils.train_graph_utils import train_utils
from collections import Iterable
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support,accuracy_score
import numpy as np
import time
import warnings
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
from CausalGCN import CLUB, CausalGCN, CausalGAT, CausalGIN
from TEgraph import TEgraph

args = None

def parse_args():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser(description='Train')
    # basic parameters
    parser.add_argument('--model_name', type=str, default='CausalGCN', help='the name of the model') #'CausalGCN'
    parser.add_argument('--sample_length', type=int, default=100, help='sample_lengths')
    parser.add_argument('--overlap', type=int, default=1, help='overlap')
    parser.add_argument('--noise_std', type=float, default=0.05, help='the noise added in test data')
    parser.add_argument('--Input_type', choices=['TD', 'FD','other'],type=str, default='TD', help='the input type decides the length of input')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--batch_size', type=int, default=128, help='batchsize of the training process')
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
    parser.add_argument('--eval_random', type=str2bool, default=False) #False
    parser.add_argument('--without_node_attention', type=str2bool, default=False)
    parser.add_argument('--without_edge_attention', type=str2bool, default=False)
    parser.add_argument('--fc_num', type=str, default=2)
    parser.add_argument('--cat_or_add', type=str, default="add")#"add"
    parser.add_argument('--c', type=float, default=0.001)  ##0.5
    parser.add_argument('--o', type=float, default=1.0)
    parser.add_argument('--co', type=float, default=1.0)
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

def train_causal_epoch(epoch, model, optimizer, mi_estimator, mi_optimizer, loader, device, args):
    model.train()
    mi_estimator.eval()
    total_loss = 0
    total_loss_c = 0
    total_loss_o = 0
    total_loss_co = 0
    correct_o = 0
    for it, data in enumerate(loader):
        optimizer.zero_grad()
        data = data.to(device)

        one_hot_target = data.y.view(-1)
        c_logs, o_logs, co_logs, c_f, o_f,_,_ = model(data, eval_random=args.with_random)
        c_loss = mi_estimator(o_f, c_f)
        # uniform_target = torch.ones_like(c_logs, dtype=torch.float).to(device) / model.num_classes
        # c_loss = F.kl_div(c_logs, uniform_target, reduction='batchmean')

        o_loss = F.nll_loss(o_logs, one_hot_target)
        co_loss = F.nll_loss(co_logs, one_hot_target)
        if epoch < 0:
            c = 0.0
            co = 0.0
        else:
            c = args.c
            co = args.co
        loss = c * c_loss + args.o * o_loss + co * co_loss

        pred_o = o_logs.max(1)[1]
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        total_loss_c += c * (c_loss.item()) * data.num_graphs
        total_loss_o += args.o * (o_loss.item()) * data.num_graphs
        total_loss_co += co * (co_loss.item()) * data.num_graphs
        optimizer.step()

        for inter_mi in range(20):
            mi_estimator.train()
            _, _, _, c_f, o_f,_,_ = model(data, eval_random=args.with_random)
            mi_loss = mi_estimator.learning_loss(o_f, c_f)
            mi_optimizer.zero_grad()
            mi_loss.backward()
            mi_optimizer.step()

    num = len(loader.dataset)
    total_loss = total_loss / num
    total_loss_c = total_loss_c / num
    total_loss_o = total_loss_o / num
    total_loss_co = total_loss_co / num
    correct_o = correct_o / num
    return total_loss, total_loss_c, total_loss_o, total_loss_co, correct_o


def train_causal_epoch2(epoch, model, optimizer, mi_estimator, mi_optimizer, loader, device, args):
    model.train()
    mi_estimator.eval()
    total_loss = 0
    total_loss_c = 0
    total_loss_o = 0
    total_loss_co = 0
    correct_o = 0
    data_pred = []
    data_label = []
    for it, data in enumerate(loader):
        optimizer.zero_grad()
        data = data.to(device)

        one_hot_target = data.y.view(-1)
        c_logs, o_logs, co_logs, c_f, o_f,_,_ = model(data, eval_random=args.with_random)
        c_loss = mi_estimator(o_f, c_f)
        # uniform_target = torch.ones_like(c_logs, dtype=torch.float).to(device) / model.num_classes
        # c_loss = F.kl_div(c_logs, uniform_target, reduction='batchmean')

        o_loss = F.nll_loss(o_logs, one_hot_target)
        co_loss = F.nll_loss(co_logs, one_hot_target)
        if epoch < 0:
            c = 0.0
            co = 0.0
        else:
            c = args.c
            co = args.co
        loss = c * c_loss + args.o * o_loss + co * co_loss

        pred_o = o_logs.max(1)[1]
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()
        pred = co_logs.max(1)[1]
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        total_loss_c += c * (c_loss.item()) * data.num_graphs
        total_loss_o += args.o * (o_loss.item()) * data.num_graphs
        total_loss_co += co * (co_loss.item()) * data.num_graphs
        optimizer.step()
        data_pred.append(pred.cpu().detach().data.tolist())
        data_label.append(data.y.cpu().detach().data.tolist())

        for inter_mi in range(20):
            mi_estimator.train()
            _, _, _, c_f, o_f,_,_ = model(data, eval_random=args.with_random)
            mi_loss = mi_estimator.learning_loss(o_f, c_f)
            mi_optimizer.zero_grad()
            mi_loss.backward()
            mi_optimizer.step()

    list_data_label = list(flatten(data_label))
    list_data_pred = list(flatten(data_pred))
    all_report = precision_recall_fscore_support(list_data_label, list_data_pred, average='weighted')
    all_precision = all_report[0]
    all_recall = all_report[1]
    all_fscore = all_report[2]
    print('Training all_precision', all_precision, 'all_recall', all_recall, 'all_fscore', all_fscore)

    num = len(loader.dataset)
    total_loss = total_loss / num
    total_loss_c = total_loss_c / num
    total_loss_o = total_loss_o / num
    total_loss_co = total_loss_co / num
    correct_o = correct_o / num
    return total_loss, total_loss_c, total_loss_o, total_loss_co, correct_o

def eval_acc_causal(model, loader, device, args):
    model.eval()
    eval_random = args.eval_random
    correct = 0
    correct_c = 0
    correct_o = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            c_logs, o_logs, co_logs, _, _,_,_ = model(data, eval_random=eval_random)
            pred = co_logs.max(1)[1]
            pred_c = c_logs.max(1)[1]
            pred_o = o_logs.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        correct_c += pred_c.eq(data.y.view(-1)).sum().item()
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()

    acc_co = correct / len(loader.dataset)
    acc_c = correct_c / len(loader.dataset)
    acc_o = correct_o / len(loader.dataset)
    return acc_co, acc_c, acc_o

def eval_acc_causal2(model, loader, device, args):
    model.eval()
    eval_random = args.eval_random
    correct = 0
    correct_c = 0
    correct_o = 0
    data_pred = []
    data_label = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            c_logs, o_logs, co_logs, _, _,_,_ = model(data, eval_random=eval_random)
            pred = co_logs.max(1)[1]
            pred_c = c_logs.max(1)[1]
            pred_o = o_logs.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        correct_c += pred_c.eq(data.y.view(-1)).sum().item()
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()
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

    acc_co = correct / len(loader.dataset)
    acc_c = correct_c / len(loader.dataset)
    acc_o = correct_o / len(loader.dataset)
    return acc_co, acc_c, acc_o

def eval_acc_causal3(model, loader, device, args):
    model.eval()
    eval_random = args.eval_random
    correct = 0
    correct_c = 0
    correct_o = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            c_logs, o_logs, co_logs, _, _, edgatt, nodeatt = model(data, eval_random=eval_random)
            pred = co_logs.max(1)[1]
            pred_c = c_logs.max(1)[1]
            pred_o = o_logs.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        correct_c += pred_c.eq(data.y.view(-1)).sum().item()
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()

    acc_co = correct / len(loader.dataset)
    acc_c = correct_c / len(loader.dataset)
    acc_o = correct_o / len(loader.dataset)
    np.save('edgatt_TE', edgatt.cpu().numpy())
    np.save('nodeatt_TE', nodeatt.cpu().numpy())
    np.save('labels_TE', (data.y.view(-1)).cpu().numpy())

    return acc_co, acc_c, acc_o

def eval_acc_causal4(model, loader, device, args):
    model.eval()
    eval_random = args.eval_random
    correct = 0
    correct_c = 0
    correct_o = 0
    data_pred_c = []
    data_pred_o = []
    data_pred_co = []
    data_label = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            c_logs, o_logs, co_logs, _, _,_,_ = model(data, eval_random=eval_random)
            pred = co_logs.max(1)[1]
            pred_c = c_logs.max(1)[1]
            pred_o = o_logs.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        correct_c += pred_c.eq(data.y.view(-1)).sum().item()
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()
        data_pred_co.append(pred.cpu().detach().data.tolist())
        data_pred_c.append(pred_c.cpu().detach().data.tolist())
        data_pred_o.append(pred_o.cpu().detach().data.tolist())
        data_label.append(data.y.cpu().detach().data.tolist())
    list_data_label = list(flatten(data_label))
    list_data_pred_co = list(flatten(data_pred_co))
    list_data_pred_c = list(flatten(data_pred_c))
    list_data_pred_o = list(flatten(data_pred_o))
    all_report = precision_recall_fscore_support(list_data_label, list_data_pred_co, average='weighted')
    all_precision = all_report[0]
    all_recall = all_report[1]
    all_fscore = all_report[2]
    print('CO Testing all_precision', all_precision, 'all_recall', all_recall, 'all_fscore', all_fscore)
    print(classification_report(list_data_label, list_data_pred_co))
    print(confusion_matrix(list_data_label, list_data_pred_co))
    ##########
    all_report = precision_recall_fscore_support(list_data_label, list_data_pred_c, average='weighted')
    all_precision = all_report[0]
    all_recall = all_report[1]
    all_fscore = all_report[2]
    print('C Testing all_precision', all_precision, 'all_recall', all_recall, 'all_fscore', all_fscore)
    print(classification_report(list_data_label, list_data_pred_c))
    print(confusion_matrix(list_data_label, list_data_pred_c))
    ##############
    all_report = precision_recall_fscore_support(list_data_label, list_data_pred_o, average='weighted')
    all_precision = all_report[0]
    all_recall = all_report[1]
    all_fscore = all_report[2]
    print('O Testing all_precision', all_precision, 'all_recall', all_recall, 'all_fscore', all_fscore)
    print(classification_report(list_data_label, list_data_pred_o))
    print(confusion_matrix(list_data_label, list_data_pred_o))
    acc_co = correct / len(loader.dataset)
    acc_c = correct_c / len(loader.dataset)
    acc_o = correct_o / len(loader.dataset)
    return acc_co, acc_c, acc_o

if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()

    device = torch.device("cuda")
    # Dataset = getattr(datasets, args.data_name)
    datasets = {}
    datasets['train'], datasets['val'] = TEgraph(args.sample_length, args.overlap, args.noise_std).data_preprare()

    # Define the model
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
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)
    seed = 10
    torch.manual_seed(seed)

    if args.model_name == "CausalGCN":
        model = CausalGCN(num_features=feature, num_classes=TEgraph.num_classes, args=args,gfn=False,edge_norm=True).to(device)
    elif args.model_name == "CausalGIN":
        model = CausalGIN(num_features=feature, num_classes=TEgraph.num_classes, args=args, gfn=False, edge_norm=True).to(
        device)
    elif args.model_name == "CausalGAT":
        model = CausalGAT(num_features=feature, num_classes=TEgraph.num_classes, args=args).to(
        device)
    else:
        print('No model')
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mi_estimator = CLUB(x_dim=args.hidden, y_dim=args.hidden, hidden_size=args.hidden).to(device)
    mi_optimizer = Adam(mi_estimator.parameters(), lr=0.01)#args.lr

    for epoch in range(1, args.max_epochs + 1):
        if epoch >40:
            train_loss, loss_c, loss_o, loss_co, train_acc = train_causal_epoch(epoch, model, optimizer, mi_estimator,
                                                                                mi_optimizer, train_loader, device,
                                                                                args)
            test_acc, test_acc_c, test_acc_o = eval_acc_causal4(model, test_loader, device, args)
        # if epoch>95:
        #     train_loss, loss_c, loss_o, loss_co, train_acc = train_causal_epoch2(epoch, model, optimizer, mi_estimator,
        #                                                                         mi_optimizer, train_loader, device,
        #                                                                         args)
        #     test_acc, test_acc_c, test_acc_o = eval_acc_causal2(model, test_loader, device, args)
        else:
            train_loss, loss_c, loss_o, loss_co, train_acc = train_causal_epoch(epoch, model, optimizer, mi_estimator,
                                                                            mi_optimizer, train_loader, device, args)
            test_acc, test_acc_c, test_acc_o = eval_acc_causal(model, test_loader, device, args)

        train_accs.append(train_acc)
        test_accs.append(test_acc)
        test_accs_c.append(test_acc_c)
        test_accs_o.append(test_acc_o)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            best_test_acc_c = test_acc_c
            best_test_acc_o = test_acc_o

        print(
            "Causal | Epoch:[{}/{}] Loss:[{:.4f}={:.4f}+{:.4f}+{:.4f}] Train:[{:.4f}] Test:[{:.2f}] Test_o:[{:.2f}] Test_c:[{:.2f}] (RG:{:.2f}) | Best Test:[{:.2f}] at Epoch:[{}] | Test_o:[{:.2f}] Test_c:[{:.2f}]"
            .format(
                    epoch, args.max_epochs,
                    train_loss, loss_c, loss_o, loss_co,
                    train_acc * 100,
                    test_acc * 100,
                    test_acc_o * 100,
                    test_acc_c * 100,
                    random_guess * 100,
                    best_test_acc * 100,
                    best_epoch,
                    best_test_acc_o * 100,
                    best_test_acc_c * 100))

    print(
        "syd: Causal | Best Test:[{:.2f}] at epoch [{}] | Test_o:[{:.2f}] Test_c:[{:.2f}] (RG:{:.2f})"
        .format(
                best_test_acc * 100,
                best_epoch,
                best_test_acc_o * 100,
                best_test_acc_c * 100,
                random_guess * 100))

