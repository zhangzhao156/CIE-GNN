# -*- coding: utf-8 -*-
# @Time    : 2023/5/31 21:41
# @Author  : zhao
# @File    : TEgraph.py

from sklearn.model_selection import train_test_split
import numpy as np
import copy
from scipy.spatial.distance import pdist
import torch
from torch_geometric.data import Data

# generate Training Dataset and Testing Dataset
def get_files(sample_length, overlap):
    data = np.load("data.npz")
    traindata_x, traindata_y, testdata_x, testdata_y = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print(traindata_x.shape, traindata_y.shape, testdata_x.shape, testdata_y.shape)
    # print(traindata_x.dtype, traindata_y.dtype, testdata_x.dtype, testdata_y.dtype)
    normal_samples = np.vstack((traindata_x[:500, :], testdata_x[:960, :]))
    # normal_labels = np.hstack((traindata_y[:500], testdata_y[:960]))
    # print(np.unique(normal_labels))
    # print(normal_samples.shape)
    mean = np.mean(normal_samples, axis=0)
    # print(mean.shape, mean)
    std = np.std(normal_samples, axis=0)
    # print(std.shape, std)
    # print('origan:',traindata_x[:2,:])
    traindata_x_norm = (traindata_x - mean) / std
    testdata_x_norm = (testdata_x - mean) / std

    edge_index, edge_fe = SensorNetwork(data=traindata_x_norm,graphType='RadiusGraph')

    train_graph_data=[]
    test_graph_data = []
    for i in range(22):
        # print('cls',i)
        traindata_indexes = np.where(traindata_y == i)
        traindata_x_cls = traindata_x_norm[traindata_indexes, :].reshape(-1,52)
        # print('traindata_x_cls',traindata_x_cls.shape)
        traindata_x_wins = data_win_split(traindata_x_cls,sample_length,overlap)
        # print('traindata_x_wins', len(traindata_x_wins))
        traindata_x_graph = Gen_graph(traindata_x_wins, edge_index, edge_fe, i)
        # print('traindata_x_graph', len(traindata_x_graph))
        train_graph_data += traindata_x_graph
        testdata_indexes = np.where(testdata_y == i)
        testdata_x_cls = testdata_x_norm[testdata_indexes, :].reshape(-1,52)
        testdata_x_wins = data_win_split(testdata_x_cls, sample_length, overlap)
        testdata_x_graph = Gen_graph(testdata_x_wins, edge_index, edge_fe, i)
        test_graph_data += testdata_x_graph
    return train_graph_data, test_graph_data

def data_win_split(datax,signal_size,overlap):
    data = []
    start, end = 0, signal_size
    # seg_num = 0
    while start <= (datax.shape[0]-signal_size):
        x = datax[start:end,:]
        data.append(x)
        start += overlap
        end += overlap
    return data

def Gen_graph(data, node_edge, w, label):
    data_list = []
    for i in range(len(data)):
        graph_feature = data[i].T
        # print('graph_feature',graph_feature.shape)
        labels = [label]
        node_features = torch.tensor(graph_feature, dtype=torch.float)
        graph_label = torch.tensor(labels, dtype=torch.long)  # 获得图标签
        edge_index = torch.tensor(node_edge, dtype=torch.long)
        edge_features = torch.tensor(w, dtype=torch.float)
        graph = Data(x=node_features, y=graph_label, edge_index=edge_index, edge_attr=edge_features)
        data_list.append(graph)
    return data_list

def SensorNetwork(data,graphType):
    if graphType == 'KNNGraph':
         edge_index,edge_fe = KNN_attr(data)
    elif graphType == 'RadiusGraph':
        edge_index, edge_fe = Radius_attr(data)
    else:
        print("There is no such graphType!")
    return edge_index, edge_fe

def Radius_attr(data):
    s1 = range(data.shape[1])
    s2 = copy.deepcopy(s1)
    edge_index = np.array([[], []])  # 一个故障样本与其他故障样本匹配生成一次图
    edge_fe = []
    for i in s1:
        for j in s2:
            local_edge, w = cal_sim(data, i, j)
            edge_index = np.hstack((edge_index, local_edge))
            if any(w):
                edge_fe.append(w[0])
    return edge_index,edge_fe

def KNN_attr(data):
    '''
    for KNNgraph
    :param data:
    :return:
    '''
    data = data.T
    edge_raw0 = []
    edge_raw1 = []
    edge_fea = []
    for i in range(len(data)):
        x = data[i]
        node_index, topK_x= KNN_classify(5,data,x)
        loal_weigt = KNN_weigt(x,topK_x)
        local_index = np.zeros(5)+i

        edge_raw0 = np.hstack((edge_raw0,local_index))
        edge_raw1 = np.hstack((edge_raw1,node_index))
        edge_fea = np.hstack((edge_fea,loal_weigt))

    edge_index = [edge_raw0, edge_raw1]

    return edge_index, edge_fea

#--------------------------------------------------------------------------------
def cal_sim(data,s1,s2):
    edge_index = [[],[]]
    edge_feature = []
    if s1 != s2:
        v_1 = data[:,s1]
        v_2 = data[:,s2]
        combine = np.vstack([v_1, v_2])
        likely = 1- pdist(combine, 'cosine')
        # print('likely',likely)
        # w = 1
        # edge_index[0].append(s1)
        # edge_index[1].append(s2)
        # edge_feature.append(w)
        if likely.item() >= 0:
            w = 1
            edge_index[0].append(s1)
            edge_index[1].append(s2)
            edge_feature.append(w)
    return edge_index,edge_feature

#-------------------------------------------------------------------------------
def KNN_classify(k,X_set,x):
    """
    k:number of neighbours
    X_set: the datset of x
    x: to find the nearest neighbor of data x
    """

    distances = [sqrt(np.sum((x_compare-x)**2)) for x_compare in X_set]
    nearest = np.argsort(distances)
    node_index  = [i for i in nearest[1:k+1]]
    topK_x = [X_set[i] for i in nearest[1:k+1]]
    return  node_index,topK_x


def KNN_weigt(x,topK_x):
    distance = []
    v_1 = x
    data_2 = topK_x
    for i in range(len(data_2)):
        v_2 = data_2[i]
        combine = np.vstack([v_1, v_2])
        likely = pdist(combine, 'euclidean')
        distance.append(likely[0])
    beata = np.mean(distance)
    w = np.exp((-(np.array(distance)) ** 2) / (2 * (beata ** 2)))
    return w

# def gaussian_noise(x,std):
#     add_mu = 0.0
#     add_std = std
#     scale_mu = 1.0
#     scale_std = std
#     add_noise = torch.normal(add_mu, add_std, size=(1,1))#x.size(0), x.size(1)
#     scale_noise = torch.normal(scale_mu, scale_std, size=(1,1))
#     # print('add_noise',add_noise)
#     # x_noisy = scale_noise*x + add_noise
#     x_noisy=scale_noise*x + add_noise
#     return x_noisy

def gaussian_noise(x,add_noise,scale_noise):

    # print('add_noise',add_noise)
    # x_noisy = scale_noise*x + add_noise
    x_noisy=scale_noise*x + add_noise
    return x_noisy

class TEgraph(object):
    num_classes = 22
    def __init__(self, sample_length, overlap,noise_std):
        self.sample_length = sample_length
        self.overlap = overlap
        self.noise_std = noise_std

    def data_preprare(self):
        train_dataset, val_dataset = get_files(self.sample_length, self.overlap)
        all_dataset = train_dataset+val_dataset
        train_dataset, val_dataset = train_test_split(all_dataset, test_size=0.20, random_state=40)
        if self.noise_std > 0:
            add_mu = 0.0
            add_std = self.noise_std
            scale_mu = 1.0
            scale_std = self.noise_std
            add_noise = torch.normal(add_mu, add_std, size=(52, 1))  # x.size(0), x.size(1)
            scale_noise = torch.normal(scale_mu, scale_std, size=(52, 1))
            for i in range(len(val_dataset)):
                # print('before', val_dataset[i].x)
                # val_dataset[i].x = gaussian_noise(val_dataset[i].x, self.noise_std)
                val_dataset[i].x = gaussian_noise(val_dataset[i].x, add_noise, scale_noise)
                # print('after', val_dataset[i].x)
        return train_dataset, val_dataset

# train_dataset, val_dataset = get_files(100, 1)
# print('train_dataset',len(train_dataset),'val_dataset',len(val_dataset))
