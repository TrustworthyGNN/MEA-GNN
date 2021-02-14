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
#import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import time
from dgl.nn.pytorch import GraphConv

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(1433, 16, activation=F.relu))
        # output layer
        self.layers.append(GraphConv(16, 7))
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, g, features):
        x = F.relu(self.layers[0](g, features))
        x = self.layers[1](g, x)
        return x

class Net_attack(nn.Module):
    def __init__(self,feature_number, label_number):
        super(Net_attack, self).__init__()
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

def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

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

def attack2(dataset_name, attack_node_arg, cuda):
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
        
    attack_nodes = []
    for i in range(attack_node_number):
        candidate_node = random.randint(0,node_number - 1)
        if candidate_node not in attack_nodes:
            attack_nodes.append(candidate_node)
    
    
    features_np = data.features
    labels = data.labels
    train_mask = data.train_mask
    test_mask = data.test_mask
    g = nx.to_numpy_array(data.graph)
    g_matrix = np.asmatrix(g.copy())
    
    test_num = 0
    for i in range(node_number):
        if i in attack_nodes:
            test_mask[i] = 0
            train_mask[i] = 1
            continue
        else:
            if test_num < 1000:
                test_mask[i] = 1
                train_mask[i] = 0
                test_num = test_num + 1
            else:
                test_mask[i] = 0
                train_mask[i] = 0
    
    
    
    gcn_Net.eval()
    
    
    features = th.FloatTensor(features_np)
    g_graph = DGLGraph(g)
    
    #=================Generate Label===================================================
    logits_query = gcn_Net(g_graph, features)
    _, labels_query = th.max(logits_query, dim=1)
    
    # =============================================================================
    # logits_query_numpy = logits_query.detach().numpy()
    # sub_logits_query = np.zeros((len(sub_graph_node_index),7))
    # for sub_index in range(len(sub_graph_node_index)):
    #     sub_logits_query[sub_index] = logits_query_numpy[sub_graph_node_index[sub_index]]
    # sub_logits_query = th.FloatTensor(sub_logits_query)
    # =============================================================================
    
    syn_features_np = np.eye(node_number)
    
    syn_features = th.FloatTensor(syn_features_np)
    
    g = nx.from_numpy_matrix(g)
    # graph preprocess and calculate normalization factor
    #sub_g_b = nx.from_numpy_array(sub_g_b)
    # add self loop
    
    g.remove_edges_from(nx.selfloop_edges(g))
    g.add_edges_from(zip(g.nodes(), g.nodes()))
    
    
    g = DGLGraph(g)
    n_edges = g.number_of_edges()
    # normalization
    degs = g.in_degrees().float()
    norm = th.pow(degs, -0.5)
    norm[th.isinf(norm)] = 0
    
    g.ndata['norm'] = norm.unsqueeze(1)
    
    
    #train target model
    
    labels = th.LongTensor(labels)
    train_mask = th.ByteTensor(train_mask)
    test_mask = th.ByteTensor(test_mask)
    
    net_attack = Net_attack(node_number, label_number)
    
    optimizer_original = th.optim.Adam(net_attack.parameters(), lr=5e-2, weight_decay=5e-4)
    dur = []
    
    max_acc1 = 0
    max_acc2 = 0
    
    print("=========Model Extracting==========================")
    
    for epoch in range(200):
        if epoch >=3:
            t0 = time.time()
            
        net_attack.train()
        logits = net_attack(g, syn_features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels_query[train_mask])
        
        optimizer_original.zero_grad()
        loss.backward()
        optimizer_original.step()
        
        
        if epoch >=3:
            dur.append(time.time() - t0)
            
        acc1 = evaluate(net_attack, g, syn_features, labels, test_mask)
        acc2 = evaluate(net_attack, g, syn_features, labels_query, test_mask)
        print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Test Fid  {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), acc1, acc2, np.mean(dur)))
        
        if acc1>max_acc1:
            max_acc1 = acc1
        if acc2>max_acc2:
            max_acc2 = acc2
    
    print("Accuracy: " + str(acc1) + " /Fidelity: " + str(acc2))
    