# -*- coding: utf-8 -*-

import numpy as np
from dgl.data import citation_graph as citegrh
import torch
import torch.nn
import time

import dgl.function as fn
from dgl import DGLGraph
import networkx as nx
from dgl.nn.pytorch import GraphConv
import torch.nn as nn
import torch.nn.functional as F



def attack1(dataset_name, attack_node_arg, cuda):
    
    # you should replace the file with the structure you generated
    # file 1 is for the attack node index you selected here we use 700 nodes as defult
    # file 2 is the query labeled of all the nodes
    # filre 3 is the generated graph structure
    
    print("==================attack nodes and their queried labels/generated structure loading================================================")
    
    if dataset_name == 'cora':
        node_number=2708
        feature_number = 1433
        label_number = 7
        data = citegrh.load_cora()
        data1 = citegrh.load_cora()
        filename1 = "./data/attack2_generated_graph/cora/selected_index.txt"
        filename2 = "./data/attack2_generated_graph/cora/query_labels_cora.txt"
        filename3 = "./data/attack2_generated_graph/cora/graph_label0_564_541.txt"
    if dataset_name == 'citeseer':
        node_number=3327
        feature_number =3703
        label_number =6
        data = citegrh.load_citeseer()
        data1 = citegrh.load_citeseer()
        filename1 = "./data/attack2_generated_graph/citeseer/selected_index.txt"
        filename2 = "./data/attack2_generated_graph/citeseer/query_labels_citeseer.txt"
        filename3 = "./data/attack2_generated_graph/citeseer/graph_label0_604_525.txt"
    if dataset_name == 'pubmed':
        node_number=19717
        feature_number = 500
        label_number = 3
        data = citegrh.load_pubmed()
        data1 = citegrh.load_pubmed()
        filename1 = "./data/attack2_generated_graph/pubmed/selected_index.txt"
        filename2 = "./data/attack2_generated_graph/pubmed/query_labels_pubmed.txt"
        filename3 = "./data/attack2_generated_graph/pubmed/graph_label0_0.657_667_.txt"
    
    #attack_node_number = int(node_number * attack_node_arg)
    attack_node_number = 700
    
    print("==================Done !====================================")
    
    # get sample nodes
    
    my_file1 = open(filename1,"r")
    lines1 = my_file1.readlines()
    attack_nodes = []
    for line_1 in lines1:
        attack_nodes.append(int(line_1))
    my_file1.close()
    
    testing_nodes = []
    for i in range(node_number):
        if i not in attack_nodes:
            testing_nodes.append(i)
    
    
    # get their features
    features = data.features
    features_numpy = features
    labels = data.labels
    labels_numpy = labels
    train_mask = data.train_mask
    test_mask = data.test_mask
    
    #features = torch.FloatTensor(data.features)
    #labels = torch.LongTensor(data.labels)
    
    attack_features = torch.FloatTensor(features[attack_nodes])
    test_features = torch.FloatTensor(features[testing_nodes])
    test_labels = torch.LongTensor(labels_numpy[testing_nodes])
    
    for i in range(node_number):
        if i in attack_nodes:
            test_mask[i] = 0
            train_mask[i] = 1
        else:
            test_mask[i] = 1
            train_mask[i] = 0
    
    sub_test_mask = torch.ByteTensor(test_mask)
    
    # get their labels
    my_file2 = open(filename2,"r")
    lines2 = my_file2.readlines()
    all_query_labels = []
    attack_query = []
    for line_2 in lines2:
        all_query_labels.append(int(line_2.split()[1]))
        if int(line_2.split()[0]) in attack_nodes:
            attack_query.append(int(line_2.split()[1]))
    
    my_file2.close()
    
    attack_query = torch.LongTensor(attack_query)
    all_query_labels = torch.LongTensor(all_query_labels)
    
    #build shadow graph
    my_file2 = open(filename3,"r")
    lines2 = my_file2.readlines()
    adj_matrix = np.zeros((attack_node_number,attack_node_number))
    for line_2 in lines2:
        list_line = line_2.split()
        adj_matrix[int(list_line[0])][int(list_line[1])]=1
        adj_matrix[int(list_line[1])][int(list_line[0])]=1
    
    my_file2.close()
    
    g_shadow = np.asmatrix(adj_matrix)
    #sub_g = DGLGraph(nx.from_numpy_matrix(g_shadow))
    
    sub_g = nx.from_numpy_matrix(g_shadow)
    
    # graph preprocess and calculate normalization factor
    #sub_g_b = nx.from_numpy_array(sub_g_b)
    # add self loop
    
    sub_g.remove_edges_from(nx.selfloop_edges(sub_g))
    sub_g.add_edges_from(zip(sub_g.nodes(), sub_g.nodes()))
    
    
    sub_g = DGLGraph(sub_g)
    n_edges = sub_g.number_of_edges()
    # normalization
    degs = sub_g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    
    sub_g.ndata['norm'] = norm.unsqueeze(1)
    
    #build GCN
    
    
    
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    #g = DGLGraph(data.graph)
    g_numpy = nx.to_numpy_array(data.graph)
    sub_g_b = nx.from_numpy_matrix(g_numpy)
    
    # graph preprocess and calculate normalization factor
    #sub_g_b = nx.from_numpy_array(sub_g_b)
    # add self loop
    
    sub_g_b.remove_edges_from(nx.selfloop_edges(sub_g_b))
    sub_g_b.add_edges_from(zip(sub_g_b.nodes(), sub_g_b.nodes()))
    
    
    sub_g_b = DGLGraph(sub_g_b)
    n_edges = sub_g_b.number_of_edges()
    # normalization
    degs = sub_g_b.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    
    sub_g_b.ndata['norm'] = norm.unsqueeze(1)
    
    #train the DNN
            
    net = Net_shadow(feature_number, label_number)
    print(net)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=5e-4)
    dur = []
    
    max_acc1 = 0
    max_acc2 = 0
    
    print("===================Model Extracting================================")
    
    for epoch in range(200):
        if epoch >=3:
            t0 = time.time()
    
        net.train()
        logits = net(sub_g, attack_features)
        logp = torch.nn.functional.log_softmax(logits, dim = 1)
        loss = torch.nn.functional.nll_loss(logp, attack_query)
        
        #weights = [1/num_0, 1/num_1, 1/num_2, 1/num_3, 1/num_4, 1/num_5, 1/num_6]
        #class_weights = th.FloatTensor(weights)
    # =============================================================================
    #     criterion = torch.nn.CrossEntropyLoss()
    #     loss = criterion(logp, attack_query)
    # =============================================================================
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if epoch >=3:
            dur.append(time.time() - t0)
    
        acc1 = evaluate(net, sub_g_b, features, labels, sub_test_mask)
        acc2 = evaluate(net, sub_g_b, features, all_query_labels, sub_test_mask)
        
        if acc1>max_acc1:
            max_acc1 = acc1
        if acc2>max_acc2:
            max_acc2 = acc2
        print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Test Fid {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), acc1, acc2, np.mean(dur)))
    
    print("Final one:" + str(max_acc1) + "Fiderity: " + str(max_acc2))

class GCNLayer(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = torch.nn.Linear(in_feats, out_feats)
        

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        gcn_msg = fn.copy_src(src='h', out='m')
        gcn_reduce = fn.sum(msg='m', out='h')
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)

class Net_shadow(torch.nn.Module):
    def __init__(self, feature_number, label_number):
        super(Net_shadow, self).__init__()
        self.layer1 = GCNLayer(feature_number, 16)
        self.layer2 = GCNLayer(16, label_number)

    def forward(self, g, features):
        x = torch.nn.functional.relu(self.layer1(g, features))
        #x = torch.nn.functional.dropout(x, 0.2)
        #x = F.dropout(x, training=self.training)
        x = self.layer2(g, x)
        return x

#build a simple DNN
class MyNet(torch.nn.Module):
    """
    Input - 1433
    Output - 7
    """
    def __init__(self, in_feats, out_feats):
        super(MyNet, self).__init__()

        self.fc1 = torch.nn.Linear(in_feats, 16)
        self.fc2 = torch.nn.Linear(16, out_feats) 

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)



