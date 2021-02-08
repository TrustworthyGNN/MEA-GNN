# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:51:25 2020
This code is for creat a sub-graph
@author: Bang
"""

#alpha_list = [0,0.2,0.4,0.6,0.8,1,0,0.2,0.4,0.6,0.8,1,0,0.2,0.4,0.6,0.8,1,0,0.2,0.4,0.6,0.8,1,0,0.2,0.4,0.6,0.8,1]#,0,0.2,0.4,0.6,0.8,1,0,0.2,0.4,0.6,0.8,1,0,0.2,0.4,0.6,0.8,1,0,0.2,0.4,0.6,0.8,1,0,0.2,0.4,0.6,0.8,1,0,0.2,0.4,0.6,0.8,1]
alpha_list = [0.8,0.8,0.8,0.8,0.8]

from dgl.data import citation_graph as citegrh
import networkx as nx
import numpy as np

import torch as th

import math

import random

from scipy.stats import wasserstein_distance

import dgl
import dgl.function as fn
#import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import time

from dgl.nn.pytorch import GraphConv

node_number=2708 #3327
feature_number = 1433 #3703
label_number = 7 #6
attack_node_number = 677 #832

def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

class Gcn_Net(nn.Module):
    def __init__(self):
        super(Gcn_Net, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(feature_number, 16, activation=F.relu))
        # output layer
        self.layers.append(GraphConv(16, label_number))
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, g, features):
        x = F.relu(self.layers[0](g, features))
        x = self.layers[1](g, x)
        return x

acc_result_list = []
fed_result_list = []


    
for alpha in alpha_list:
    data = citegrh.load_cora()
    features = data.features
    labels = data.labels
    train_mask = data.train_mask
    test_mask = data.test_mask
    g = nx.to_numpy_array(data.graph)
    g_matrix = np.asmatrix(g.copy())
    
    
    total_num_0 = labels.tolist().count(0)
    total_num_1 = labels.tolist().count(1)
    total_num_2 = labels.tolist().count(2)
    total_num_3 = labels.tolist().count(3)
    total_num_4 = labels.tolist().count(4)
    total_num_5 = labels.tolist().count(5)
    
    
    
    
    # =============================================================================
    # def distribution(target):
    #     #original dataset distribution
    #     dis_source = th.tensor([298/2708, 418/2708, 818/2708, 426/2708, 217/2708, 180/2708, 351/2708],dtype=th.float32)
    #     
    #     
    #     
    #     total_num_0 = target.tolist().count(0)
    #     total_num_1 = target.tolist().count(1)
    #     total_num_2 = target.tolist().count(2)
    #     total_num_3 = target.tolist().count(3)
    #     total_num_4 = target.tolist().count(4)
    #     total_num_5 = target.tolist().count(5)
    #     total_num_6 = target.tolist().count(6)
    #     
    #     total = total_num_0 + total_num_1 + total_num_2 + total_num_3 + total_num_4 + total_num_5 + total_num_6
    #     dis_target = th.tensor([total_num_0/total, total_num_1/total, total_num_2/total, total_num_3/total, total_num_4/total, total_num_5/total, total_num_6/total],dtype=th.float32)
    #     
    #     kl = th.nn.functional.kl_div(dis_target.log(), dis_source, size_average=None, reduce=None, reduction='mean')
    #     
    #     return kl.item()
    # =============================================================================
    #generate a sub-graph index
    done_nodes = []
    
    sub_graph_node_index = []
    for i in range(attack_node_number):
        sub_graph_node_index.append(random.randint(0,node_number-1))
    
    #sub_graph_node_index = [482,523,962,1128,1560,1819,2273]
    
    data1 = citegrh.load_cora()
    labels1 = th.LongTensor(data1.labels)
    test_mask1 = th.ByteTensor(data1.test_mask)
    
# =============================================================================
#     while len(sub_graph_node_index)<832:
#         protential_nodes = []
#         for node_index in sub_graph_node_index:
#             #get nodes
#             one_step_node_index = g_matrix[node_index, :].nonzero()[1].tolist()
#             for first_order_node_index in one_step_node_index:
#                 if first_order_node_index in sub_graph_node_index:
#                     continue
#                 if first_order_node_index in protential_nodes:
#                     continue
#                 protential_nodes.append(first_order_node_index)
#         #distribution distance
#         min_dis = 1000
#         min_index = one_step_node_index[0]
#         for first_order_node_index in protential_nodes:
#             #caculate distance
#             current_node_index = sub_graph_node_index.copy()
#             current_node_index.append(first_order_node_index)
#             #current_nodes = labels[current_node_index]
#             c = wasserstein_distance(labels[current_node_index],[1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
#             if c < min_dis:
#                 min_index = first_order_node_index
#             #sub_graph_node_index.append(first_order_node_index)
#         sub_graph_node_index.append(min_index)
# =============================================================================
        
    #generate sub-graph adj-matrix, features, labels
    sub_labels = labels[sub_graph_node_index]
    # =============================================================================
    # #==================================Plot the distribution of the generated label============
    # import matplotlib.pyplot as plt
    # # =============================================================================
    # # import numpy as np
    # # =============================================================================
    # # sort the data:
    # data_sorted = np.sort(sub_labels)
    # # calculate the proportional values of samples
    # p = 1. * np.arange(len(sub_labels)) / (len(sub_labels) - 1)
    # # plot the sorted data:
    # fig = plt.figure()
    # ax1 = fig.add_subplot(121)
    # ax1.plot(p, data_sorted)
    # ax1.set_xlabel('$p$')
    # ax1.set_ylabel('$x$')
    # # 
    # ax2 = fig.add_subplot(122)
    # ax2.plot(data_sorted, p)
    # ax2.set_xlabel('$x$')
    # ax2.set_ylabel('$p$')
    # =============================================================================
    
    num_0 = labels[sub_graph_node_index].tolist().count(0)
    num_1 = labels[sub_graph_node_index].tolist().count(1)
    num_2 = labels[sub_graph_node_index].tolist().count(2)
    num_3 = labels[sub_graph_node_index].tolist().count(3)
    num_4 = labels[sub_graph_node_index].tolist().count(4)
    num_5 = labels[sub_graph_node_index].tolist().count(5)
    num_6 = labels[sub_graph_node_index].tolist().count(6)
    
    
    #generate syn nodes for this sub-graph index
    done_syn_nodes = []
    syn_nodes = []
    for node_index in sub_graph_node_index:
        #get nodes
        one_step_node_index = g_matrix[node_index, :].nonzero()[1].tolist()
        two_step_node_index = []
        for first_order_node_index in one_step_node_index:
            syn_nodes.append(first_order_node_index)
            two_step_node_index = g_matrix[first_order_node_index, :].nonzero()[1].tolist()
# =============================================================================
#             for second_order_node_index in two_step_node_index:
#                 syn_nodes.append(second_order_node_index)
# =============================================================================
    
    sub_graph_syn_node_index = list(set(syn_nodes) - set(sub_graph_node_index))
    
    total_sub_nodes = list(set(sub_graph_syn_node_index + sub_graph_node_index))
    
    np_features_query = features.copy()
    
    for node_index in sub_graph_syn_node_index:
        #initialized as zero
        np_features_query[node_index] = np_features_query[node_index] * 0
        #get one step and two steps nodes
        one_step_node_index = g_matrix[node_index, :].nonzero()[1].tolist()
        one_step_node_index = list(set(one_step_node_index).intersection(set(sub_graph_node_index)))
        
        total_two_step_node_index = []
        num_one_step = len(one_step_node_index)
        for first_order_node_index in one_step_node_index:
            
            #caculate the feature: features =  0.8 * average_one + 0.8^2 * average_two
            #new_array = features[first_order_node_index]*0.8/num_one_step
            this_node_degree = len(g_matrix[first_order_node_index, :].nonzero()[1].tolist())
            np_features_query[node_index] = np.sum(
                    [np_features_query[node_index],features[first_order_node_index]*alpha/math.sqrt(num_one_step*this_node_degree)], 
                    axis = 0)
            
            two_step_node_index = g_matrix[first_order_node_index, :].nonzero()[1].tolist()
            total_two_step_node_index = list(set(total_two_step_node_index + two_step_node_index) - set(one_step_node_index))
        total_two_step_node_index = list(set(total_two_step_node_index).intersection(set(sub_graph_node_index)))
        
        num_two_step = len(total_two_step_node_index)
        for second_order_node_index in total_two_step_node_index:
            
            #caculate the feature: features =  0.8 * average_one + 0.8^2 * average_two
            this_node_second_step_nodes = []
            this_node_first_step_nodes = g_matrix[second_order_node_index, :].nonzero()[1].tolist()
            for nodes_in_this_node in this_node_first_step_nodes:
                this_node_second_step_nodes = list(set(this_node_second_step_nodes + g_matrix[nodes_in_this_node, :].nonzero()[1].tolist()))
            this_node_second_step_nodes = list(set(this_node_second_step_nodes) - set(this_node_first_step_nodes))
            
            this_node_second_degree = len(this_node_second_step_nodes)
            np_features_query[node_index] = np.sum(
                    [np_features_query[node_index],features[second_order_node_index]*(1-alpha)/math.sqrt(num_two_step*this_node_second_degree)], 
                    axis = 0)
            
    #np.where(np_features_query > 1, np_features_query, 1)
    
    
    
    
    features_query = th.FloatTensor(np_features_query)
    
    # =============================================================================
    # np_nomal_query = np_features_query.copy()
    # #normalize random query
    # for i in range(2708):
    #     np_nomal_query[i] = np_features_query[i]/sum(np_features_query[i])
    # features_query = th.FloatTensor(np_nomal_query)
    # =============================================================================
    
    
    
    # use original features
    
    #generate sub-graph adj-matrix, features, labels
    
    total_sub_nodes = list(set(sub_graph_syn_node_index + sub_graph_node_index))
    sub_g = np.zeros((len(total_sub_nodes),len(total_sub_nodes)))
    for sub_index in range(len(total_sub_nodes)):
        sub_g[sub_index] = g_matrix[total_sub_nodes[sub_index], total_sub_nodes]
    
    for i in range(node_number):
        if i in sub_graph_node_index:
            test_mask[i] = 0
            train_mask[i] = 1
            continue
        if i in sub_graph_syn_node_index:
            test_mask[i] = 1
            train_mask[i] = 0
        else:
            test_mask[i] = 1
            train_mask[i] = 0
    
    sub_train_mask = train_mask[total_sub_nodes]
    
    sub_features = features_query[total_sub_nodes]
    sub_labels = labels[total_sub_nodes]
    
    
    #num_0 = sub_labels.tolist().count(0)
    
    
    
    
    
    gcn_msg = fn.copy_src(src='h', out='m')
    gcn_reduce = fn.sum(msg='m', out='h')
    
    
    sub_features = th.FloatTensor(sub_features)
    sub_labels = th.LongTensor(sub_labels)
    sub_train_mask = th.ByteTensor(sub_train_mask)
    sub_test_mask = th.ByteTensor(test_mask)
    #sub_g = DGLGraph(nx.from_numpy_matrix(sub_g))
    
    data = citegrh.load_cora()
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    train_mask = th.ByteTensor(data.train_mask)
    test_mask = th.ByteTensor(data.test_mask)
    g = DGLGraph(data.graph)
    
    gcn_Net = Gcn_Net()
    #print(gcn_Net)
    
    gcn_Net.load_state_dict(th.load("./models/improved_target_model_cora_new_test.pkl"))
    gcn_Net.eval()
    
    #=================Generate Label===================================================
    logits_query = gcn_Net(g, features)
    _, labels_query = th.max(logits_query, dim=1)
    
    # =============================================================================
    # logits_query_numpy = logits_query.detach().numpy()
    # sub_logits_query = np.zeros((len(sub_graph_node_index),7))
    # for sub_index in range(len(sub_graph_node_index)):
    #     sub_logits_query[sub_index] = logits_query_numpy[sub_graph_node_index[sub_index]]
    # sub_logits_query = th.FloatTensor(sub_logits_query)
    # =============================================================================
    
    sub_labels_query = labels_query[total_sub_nodes]
    
    # =============================================================================
    # net = Net_shadow()
    # print(net)
    # =============================================================================
    
    # graph preprocess and calculate normalization factor
    sub_g = nx.from_numpy_array(sub_g)
    # add self loop
    
    sub_g.remove_edges_from(nx.selfloop_edges(sub_g))
    sub_g.add_edges_from(zip(sub_g.nodes(), sub_g.nodes()))
    
    
    sub_g = DGLGraph(sub_g)
    n_edges = sub_g.number_of_edges()
    # normalization
    degs = sub_g.in_degrees().float()
    norm = th.pow(degs, -0.5)
    norm[th.isinf(norm)] = 0
    
    sub_g.ndata['norm'] = norm.unsqueeze(1)
    
    # create GCN model
    
    net = Gcn_Net()
    
    
    optimizer = th.optim.Adam(net.parameters(), lr=1e-2, weight_decay=5e-4)
    dur = []
    
    
    testing_num_0 = labels[sub_test_mask].tolist().count(0)
    testing_num_1 = labels[sub_test_mask].tolist().count(1)
    testing_num_2 = labels[sub_test_mask].tolist().count(2)
    testing_num_3 = labels[sub_test_mask].tolist().count(3)
    testing_num_4 = labels[sub_test_mask].tolist().count(4)
    testing_num_5 = labels[sub_test_mask].tolist().count(5)
    
    
    
    # =============================================================================
    # for i in range(2708):
    #     if labels[i] == 0 and i in
    # testing_num_0_mask = labels[test_mask].tolist().count(0)
    # testing_num_1_mask = labels[test_mask].tolist().count(1)
    # testing_num_2_mask = labels[test_mask].tolist().count(2)
    # testing_num_3_mask = labels[test_mask].tolist().count(3)
    # testing_num_4_mask = labels[test_mask].tolist().count(4)
    # testing_num_5_mask = labels[test_mask].tolist().count(5)
    # testing_num_6_mask = labels[test_mask].tolist().count(6)
    # =============================================================================
    
    max_acc1 = 0
    max_acc2 = 0
    for epoch in range(200):
        if epoch >=3:
            t0 = time.time()
    
        net.train()
        logits = net(sub_g, sub_features)
        
    # =============================================================================
    #     logp = th.nn.functional.log_softmax(logits, dim = 1)
    #     loss = th.nn.functional.nll_loss(logp[sub_train_mask], sub_labels_query[sub_train_mask])
    # =============================================================================
        
        logp = F.log_softmax(logits, dim = 1)
        loss = F.nll_loss(logp[sub_train_mask], sub_labels_query[sub_train_mask])
        
    # =============================================================================
    #     weights = [1/num_0, 1/num_1, 1/num_2, 1/num_3, 1/num_4, 1/num_5]
    #     class_weights = th.FloatTensor(weights)
    #     criterion = nn.CrossEntropyLoss(weight=class_weights)
    #     loss = criterion(logp[sub_train_mask], sub_labels_query[sub_train_mask])
    # =============================================================================
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if epoch >=3:
            dur.append(time.time() - t0)
    
        acc1 = evaluate(net, g, features, labels_query, test_mask)
        acc2 = evaluate(net, g, features, labels, test_mask)
        if acc1>max_acc1:
            max_acc1 = acc1
        if acc2>max_acc2:
            max_acc2 = acc2
        #acc2 = evaluate(net, g, features, labels_query, test_mask)
    # =============================================================================
    #     print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
    #             epoch, loss.item(), acc, np.mean(dur)))
    # =============================================================================
    acc_result_list.append(max_acc2)
    fed_result_list.append(max_acc1)
# =============================================================================
#     print("=========================================================================")
#     print("Final one:" + str(max_acc2) + "Fedility:" + str(max_acc1))
#     print("=========================================================================")
#     th.save(net.state_dict(), "./models/test_sub_graph_B_526labeled_1448syn_combine_start_from_1_model_citeseer.pkl")
#     #"""
# =============================================================================
print("=================================================================")
print(acc_result_list)
print(fed_result_list)