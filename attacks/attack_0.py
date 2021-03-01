# -*- coding: utf-8 -*-


from dgl.data import citation_graph as citegrh
import networkx as nx
import numpy as np
import torch as th
import math
import random
from scipy.stats import wasserstein_distance
import dgl
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import time
from dgl.nn.pytorch import GraphConv

def load_data(dataset_name):
    if dataset_name == 'cora':
        data = citegrh.load_cora()
    if dataset_name == 'citeseer':
        data = citegrh.load_citeseer()
    if dataset_name == 'pubmed':
        data = citegrh.load_pubmed()
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    train_mask = th.ByteTensor(data.train_mask)
    test_mask = th.ByteTensor(data.test_mask)
    g = DGLGraph(data.graph)
    return g, features, labels, train_mask, test_mask

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
    def __init__(self,feature_number, label_number):
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

def attack0(dataset_name, attack_node_arg, cuda):
    
    if dataset_name == 'cora':
        node_number=2708
        feature_number = 1433
        label_number = 7
        data = citegrh.load_cora()
        data1 = citegrh.load_cora()
    if dataset_name == 'citeseer':
        node_number=3327
        feature_number =3703
        label_number =6
        data = citegrh.load_citeseer()
        data1 = citegrh.load_citeseer()
    if dataset_name == 'pubmed':
        node_number=19717
        feature_number = 500
        label_number = 3
        data = citegrh.load_pubmed()
        data1 = citegrh.load_pubmed()
    
    attack_node_number = int(node_number * attack_node_arg)
    
    #train target model:
    
    gcn_msg = fn.copy_src(src='h', out='m')
    gcn_reduce = fn.sum(msg='m', out='h')
    
    g, features, labels, train_mask, test_mask = load_data(dataset_name)
    
    # graph preprocess and calculate normalization factor
    g = data.graph
    # add self loop
    if 1:
        g.remove_edges_from(nx.selfloop_edges(g))
        g.add_edges_from(zip(g.nodes(), g.nodes()))
    g = DGLGraph(g)
    n_edges = g.number_of_edges()
    # normalization
    degs = g.in_degrees().float()
    norm = th.pow(degs, -0.5)
    norm[th.isinf(norm)] = 0
    if cuda != None:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)
    gcn_Net = Gcn_Net(feature_number, label_number)
    optimizer = th.optim.Adam(gcn_Net.parameters(), lr=1e-2, weight_decay=5e-4)
    dur = []
    
    print("=========Target Model Generating==========================")
    for epoch in range(200):
        if epoch >=3:
            t0 = time.time()
    
        gcn_Net.train()
        logits = gcn_Net(g, features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if epoch >=3:
            dur.append(time.time() - t0)
    
        acc = evaluate(gcn_Net, g, features, labels, test_mask)
        print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), acc, np.mean(dur)))
    
    
    
    
    alpha = 0.8
    
    features = data.features
    labels = data.labels
    train_mask = data.train_mask
    test_mask = data.test_mask
    g = nx.to_numpy_array(data.graph)
    g_matrix = np.asmatrix(g.copy())
    
    done_nodes = []
    
    sub_graph_node_index = []
    for i in range(attack_node_number):
        sub_graph_node_index.append(random.randint(0,node_number-1))
    
    
    labels1 = th.LongTensor(data1.labels)
    test_mask1 = th.ByteTensor(data1.test_mask)
    
    sub_labels = labels[sub_graph_node_index]
    
    
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
    
    features_query = th.FloatTensor(np_features_query)
    
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
    gcn_msg = fn.copy_src(src='h', out='m')
    gcn_reduce = fn.sum(msg='m', out='h')
    
    
    sub_features = th.FloatTensor(sub_features)
    sub_labels = th.LongTensor(sub_labels)
    sub_train_mask = th.ByteTensor(sub_train_mask)
    sub_test_mask = th.ByteTensor(test_mask)
    #sub_g = DGLGraph(nx.from_numpy_matrix(sub_g))
    
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    train_mask = th.ByteTensor(data.train_mask)
    test_mask = th.ByteTensor(data.test_mask)
    g = DGLGraph(data.graph)
    
    gcn_Net.eval()
    
    #=================Generate Label===================================================
    logits_query = gcn_Net(g, features)
    _, labels_query = th.max(logits_query, dim=1)
    
    
    sub_labels_query = labels_query[total_sub_nodes]
    
    
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
    
    net = Gcn_Net(feature_number, label_number)
    
    
    optimizer = th.optim.Adam(net.parameters(), lr=1e-2, weight_decay=5e-4)
    dur = []
    
    print("=========Model Extracting==========================")
    max_acc1 = 0
    max_acc2 = 0
    for epoch in range(200):
        if epoch >=3:
            t0 = time.time()
    
        net.train()
        logits = net(sub_g, sub_features)
        
        
        logp = F.log_softmax(logits, dim = 1)
        loss = F.nll_loss(logp[sub_train_mask], sub_labels_query[sub_train_mask])
        
    
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
        print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Test Fid {:.4f} | Time(s) {:.4f}".format(
             epoch, loss.item(), acc2, acc1, np.mean(dur)))
    
    print("========================Final results:=========================================")
    print("Accuracy:" + str(max_acc2) + "Fedility:" + str(max_acc1))


