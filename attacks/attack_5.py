# -*- coding: utf-8 -*-

import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

import numpy as np

from dgl.nn.pytorch import GraphConv
from dgl.data import citation_graph as citegrh
import networkx as nx

import random

import time

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)
        
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
    
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
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

def load_cora_data():
    data = citegrh.load_cora()
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

def attack5(dataset_name, attack_node_arg, cuda):
    #devide into two graph
    
    
    # Create a graph given in the above diagram
    if dataset_name == 'cora':
        node_number=2708
        feature_number = 1433
        label_number = 7
        data = citegrh.load_cora()
        data1 = citegrh.load_cora()
        model_path = './models/attack_3_subgraph_shadow_model_cora_8159.pkl'
    if dataset_name == 'citeseer':
        node_number=3327
        feature_number =3703
        label_number =6
        data = citegrh.load_citeseer()
        data1 = citegrh.load_citeseer()
        model_path = './models/attack_3_subgraph_shadow_model_citeseer_6966.pkl'
    if dataset_name == 'pubmed':
        node_number=19717
        feature_number = 500
        label_number = 3
        data = citegrh.load_pubmed()
        data1 = citegrh.load_pubmed()
        model_path = './models/attack_3_subgraph_shadow_model_pubmed_8044.pkl'
    features = data.features
    labels = data.labels
    train_mask = data.train_mask
    test_mask = data.test_mask
    
    G = data.graph
    g_numpy = nx.to_numpy_array(data.graph)
    g_matrix = np.asmatrix(g_numpy.copy())
    
    sub_graph_index_b = []
    fileObject = open('./data/' + dataset_name + '/target_graph_index.txt', 'r')
    contents = fileObject.readlines()
    for ip in contents:
        sub_graph_index_b.append(int(ip))
    fileObject.close()
    
    sub_graph_index_a = []
    fileObject = open('./data/' + dataset_name + '/protential_1200_shadow_graph_index.txt', 'r')
    contents = fileObject.readlines()
    for ip in contents:
        sub_graph_index_a.append(int(ip))
    fileObject.close()
    
    #choose attack features in graphA
    attack_node = []
    while len(attack_node) < 60:
        protential_node_index = random.randint(0,len(sub_graph_index_b) - 1)
        protential_node = sub_graph_index_b[protential_node_index]
        if protential_node not in attack_node:
            attack_node.append(int(protential_node))
    
    attack_features = features[attack_node]
    attack_labels = labels[attack_node]
    shadow_features = features[sub_graph_index_a]
    shadow_labels = labels[sub_graph_index_a]
    
    sub_graph_g_A = g_numpy[sub_graph_index_a]
    sub_graph_g_a = sub_graph_g_A[:,sub_graph_index_a]
    
    sub_graph_Attack = np.zeros((len(attack_node),len(attack_node)))
    
    
    #sub_graph_attack = 
    generated_graph_1 = np.hstack((sub_graph_Attack,np.zeros((len(attack_node),len(sub_graph_index_a)))))
    generated_graph_2 = np.hstack((np.zeros((len(sub_graph_g_a),len(attack_node))),sub_graph_g_a))
    generated_graph = np.vstack((generated_graph_1,generated_graph_2))
    
    distance = []
    for i in range(100):
        index1 = i
        index2_list = np.nonzero(sub_graph_g_a[i])[0].tolist()
        for index2 in index2_list:
            distance.append(float(np.linalg.norm(shadow_features[index1]-shadow_features[int(index2)])))
    
    threshold = np.mean(distance)
    max_threshold = max(distance)
    
    #caculate the distance of this features to every node in graphA
    generated_features = np.vstack((attack_features, shadow_features))
    generated_labels = np.hstack((attack_labels, shadow_labels))
    
    for i in range(len(attack_features)):
        for loop in range(1000):
            j = random.randint(0,len(shadow_features) - 1)
        #for j in range(len(shadow_features)):
            if np.linalg.norm(generated_features[i] - generated_features[len(attack_features) + j])<threshold:
                generated_graph[i][len(attack_features) + j] = 1
                generated_graph[len(attack_features) + j][i] = 1
                break
            if loop > 500:
                #for k in range(len(shadow_features)):
                if np.linalg.norm(generated_features[i] - generated_features[len(attack_features) + j])<max_threshold :
                    generated_graph[i][len(attack_features) + j] = 1
                    generated_graph[len(attack_features) + j][i] = 1
                    break
            if loop == 999:
                
                print("one isolated node!")
    
    # train two model and evaluate
    generated_train_mask = np.ones(len(generated_features))
    generated_test_mask = np.ones(len(generated_features))
    
    generated_features = th.FloatTensor(generated_features)
    generated_labels = th.LongTensor(generated_labels)
    generated_train_mask = th.ByteTensor(generated_train_mask)
    generated_test_mask = th.ByteTensor(generated_test_mask)
    
    generated_g = nx.from_numpy_matrix(generated_graph)
    
    # graph preprocess and calculate normalization factor
    #sub_g = nx.from_numpy_array(sub_g)
    # add self loop
    
    generated_g.remove_edges_from(nx.selfloop_edges(generated_g))
    generated_g.add_edges_from(zip(generated_g.nodes(), generated_g.nodes()))
    
    
    generated_g = DGLGraph(generated_g)
    n_edges = generated_g.number_of_edges()
    # normalization
    degs = generated_g.in_degrees().float()
    norm = th.pow(degs, -0.5)
    norm[th.isinf(norm)] = 0
    
    generated_g.ndata['norm'] = norm.unsqueeze(1)
    
    dur = []
    
    net1 = Net()
    optimizer_b = th.optim.Adam(net1.parameters(), lr=1e-2, weight_decay=5e-4)
    
    max_acc1 = 0
    max_acc2 = 0
    max_acc3 = 0
    
    net1.load_state_dict(th.load(model_path))
    #th.save(net.state_dict(), "./models/attack_3_subgraph_shadow_model_pubmed.pkl")
    
    #for sub_graph_B
    sub_graph_g_B = g_numpy[sub_graph_index_b]
    sub_graph_g_b = sub_graph_g_B[:,sub_graph_index_b]
    
    sub_graph_features_b = features[sub_graph_index_b]
    
    sub_graph_labels_b = labels[sub_graph_index_b]
    
    sub_graph_train_mask_b = train_mask[sub_graph_index_b]
    
    sub_graph_test_mask_b = test_mask[sub_graph_index_b]
    
    
    for i in range(len(sub_graph_test_mask_b)):
        if i >= 300:
            sub_graph_train_mask_b[i] = 0
            sub_graph_test_mask_b[i] = 1
        else:
            sub_graph_train_mask_b[i] = 1
            sub_graph_test_mask_b[i] = 0
    # =============================================================================
    # for i in range(len(sub_graph_test_mask_b)):
    #     #if sub_graph_test_mask_b[i] == 0:
    #     sub_graph_test_mask_b[i] = 1
    # =============================================================================
    
    sub_features_b = th.FloatTensor(sub_graph_features_b)
    sub_labels_b = th.LongTensor(sub_graph_labels_b)
    sub_train_mask_b = th.ByteTensor(sub_graph_train_mask_b)
    sub_test_mask_b = th.ByteTensor(sub_graph_test_mask_b)
    #sub_g_b = DGLGraph(nx.from_numpy_matrix(sub_graph_g_b))
    
    sub_g_b = nx.from_numpy_matrix(sub_graph_g_b)
    
    # graph preprocess and calculate normalization factor
    #sub_g_b = nx.from_numpy_array(sub_g_b)
    # add self loop
    
    sub_g_b.remove_edges_from(nx.selfloop_edges(sub_g_b))
    sub_g_b.add_edges_from(zip(sub_g_b.nodes(), sub_g_b.nodes()))
    
    
    sub_g_b = DGLGraph(sub_g_b)
    n_edges = sub_g_b.number_of_edges()
    # normalization
    degs = sub_g_b.in_degrees().float()
    norm = th.pow(degs, -0.5)
    norm[th.isinf(norm)] = 0
    
    sub_g_b.ndata['norm'] = norm.unsqueeze(1)
    
    net1.eval()
    logits_b = net1(sub_g_b, sub_features_b)
    #logits_b = F.log_softmax(logits_b, 1)
    _, query_b = th.max(logits_b, dim=1)
    
    net2 = Net2()
    optimizer_a = th.optim.Adam(net2.parameters(), lr=1e-2, weight_decay=5e-4)
    
    for epoch in range(300):
        if epoch >=3:
            t0 = time.time()
    
        net2.train()
        logits_a = net2(generated_g, generated_features)
        logp_a = F.log_softmax(logits_a, 1)
        loss_a = F.nll_loss(logp_a[generated_train_mask], generated_labels[generated_train_mask])
    
        optimizer_a.zero_grad()
        loss_a.backward()
        optimizer_a.step()
        
    
        if epoch >=3:
            dur.append(time.time() - t0)
    
        acc2 = evaluate(net2, sub_g_b, sub_features_b, sub_labels_b, sub_test_mask_b)
        acc3 = evaluate(net2, sub_g_b, sub_features_b, query_b, sub_test_mask_b)
        
        print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
                epoch, loss_a.item(), acc2, np.mean(dur)))
        
        if acc2>max_acc2:
            max_acc2 = acc2
        if acc3>max_acc3:
            max_acc3 = acc3
    
    print("===============" + str(max_acc1) + "===========================================")
    print(str(max_acc2) + " fedility: " + str(max_acc3))
    
    #"""
