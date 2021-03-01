# -*- coding: utf-8 -*-

#import library
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from dgl.data import citation_graph as citegrh
import networkx as nx
import numpy as np
import torch as th
from dgl import DGLGraph
import time



class Net_sub_attack_2(nn.Module):
    def __init__(self):
        super(Net_sub_attack_2, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(SUB_ATTACK_2_NODE_NUM, 16, activation=F.relu))
        # output layer
        self.layers.append(GraphConv(16, LABEL_NUM))
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, g, features):
        x = F.relu(self.layers[0](g, features))
        x = self.layers[1](g, x)
        return x

class Net_sub_attack_3(nn.Module):
    def __init__(self):
        super(Net_sub_attack_3, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(FEATURE_NUM, 16, activation=F.relu))
        # output layer
        self.layers.append(GraphConv(16, LABEL_NUM))
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, g, features):
        x = F.relu(self.layers[0](g, features))
        x = self.layers[1](g, x)
        return x

class Net_attack_6(nn.Module):
    def __init__(self):
        super(Net_attack_6, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(LABEL_NUM*2, LABEL_NUM))
    def forward(self, features):
        x = F.log_softmax(self.layers[0](features))
        return x

class Net_attack_2(nn.Module):
    def __init__(self):
        super(Net_attack_2, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(TARGET_NODE_NUM, 16, activation=F.relu))
        # output layer
        self.layers.append(GraphConv(16, LABEL_NUM))
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, g, features):
        x = F.relu(self.layers[0](g, features))
        x = self.layers[1](g, x)
        return x

class Net_attack_3(nn.Module):
    def __init__(self):
        super(Net_attack_3, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(FEATURE_NUM, 16, activation=F.relu))
        # output layer
        self.layers.append(GraphConv(16, LABEL_NUM))
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, g, features):
        x = F.relu(self.layers[0](g, features))
        x = self.layers[1](g, x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(FEATURE_NUM, 16, activation=F.relu))
        # output layer
        self.layers.append(GraphConv(16, LABEL_NUM))
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

def attack_model_evaluate(model, features, labels):
    model.eval()
    with th.no_grad():
        logits = model(features)
        logits = logits
        labels = labels
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def attack_model_evaluate_print(model, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def attack6(dataset_name, attack_node_arg, cuda):
    #define parameters
    TARGET_NODE_NUM = 17108 #cora:1408 citeseer:2325
    NODE_NUM = 19717 #cora:2708 citeseer:3327 pubmed:19717
    SHADOW_NODE_NUM = 2609 #cora:1300 citeseer:1002 pubmed:2609
    SUB_ATTACK_2_NODE_NUM = 2609 #cora:653 citeseer:506 pubmed:521
    SUB_ATTACK_3_NODE_NUM = 2609 #cora:647 citeseer:496 pubmed:2558
    LABEL_NUM = 3 #cora:7 citeseer:6 pubmed:3
    FEATURE_NUM = 500 #cora:1433 citeseer:3703 pubmed:500
    
    #generating graphs
    
    ##read target graph
    shadow_graph_index = []
    fileObject = open('./data/pubmed/protential_1300_shadow_graph_index.txt', 'r')
    contents = fileObject.readlines()
    for ip in contents:
        shadow_graph_index.append(int(ip))
    fileObject.close()
    
    target_graph_index = []
    for i in range(NODE_NUM):
        if i not in shadow_graph_index:
            target_graph_index.append(i)
    
    sub_shadow_graph_index_attack_2 = shadow_graph_index.copy()
    # =============================================================================
    # fileObject = open('./data/pubmed/attack_6_sub_shadow_graph_index_attack_2.txt', 'r')
    # contents = fileObject.readlines()
    # for ip in contents:
    #     sub_shadow_graph_index_attack_2.append(int(ip))
    # fileObject.close()
    # =============================================================================
    
    sub_shadow_graph_index_attack_3 = shadow_graph_index.copy()
    # =============================================================================
    # fileObject = open('./data/pubmed/attack_6_sub_shadow_graph_index_attack_3.txt', 'r')
    # contents = fileObject.readlines()
    # for ip in contents:
    #     sub_shadow_graph_index_attack_3.append(int(ip))
    # fileObject.close()
    # =============================================================================
    ##
    
    #train the emsemble model
    
    ##train sub attack-2 model
    
    data = citegrh.load_pubmed()
    features_np = data.features
    labels = data.labels
    train_mask = data.train_mask
    test_mask = data.test_mask
    g = nx.to_numpy_array(data.graph)
    g_matrix = np.asmatrix(g.copy())
    
    for i in sub_shadow_graph_index_attack_2:
        test_mask[i] = 1
        train_mask[i] = 0
    
    train_num = 0
    while train_num < 140:
        random_i = np.random.randint(0,len(sub_shadow_graph_index_attack_2)-1)
        i = sub_shadow_graph_index_attack_2[random_i]
        if train_mask[i] == 0:
            test_mask[i] = 1
            train_mask[i] = 1
            train_num = train_num + 1
    
    
    syn_features_np = np.eye(SUB_ATTACK_2_NODE_NUM)
    
    syn_features = th.FloatTensor(syn_features_np)
    
    G = data.graph
    g_numpy = nx.to_numpy_array(data.graph)
    shadow_graph_index_attack_2_A = g_numpy[sub_shadow_graph_index_attack_2]
    shadow_graph_attack_2_numpy = shadow_graph_index_attack_2_A[:,sub_shadow_graph_index_attack_2]
    shadow_graph_attack_2 = nx.from_numpy_matrix(shadow_graph_attack_2_numpy)
    
    # graph preprocess and calculate normalization factor
    #sub_g_b = nx.from_numpy_array(sub_g_b)
    # add self loop
    
    shadow_graph_attack_2.remove_edges_from(nx.selfloop_edges(shadow_graph_attack_2))
    shadow_graph_attack_2.add_edges_from(zip(shadow_graph_attack_2.nodes(), shadow_graph_attack_2.nodes()))
    
    
    shadow_graph_attack_2 = DGLGraph(shadow_graph_attack_2)
    shadow_graph_attack_2_edges = shadow_graph_attack_2.number_of_edges()
    # normalization
    degs = shadow_graph_attack_2.in_degrees().float()
    norm = th.pow(degs, -0.5)
    norm[th.isinf(norm)] = 0
    
    shadow_graph_attack_2.ndata['norm'] = norm.unsqueeze(1)
    
    
    #train target model
    
    sub_shadow_graph_index_attack_2_labels = th.LongTensor(labels[sub_shadow_graph_index_attack_2])
    sub_shadow_graph_index_attack_2_train_mask = th.ByteTensor(train_mask[sub_shadow_graph_index_attack_2])
    sub_shadow_graph_index_attack_2_test_mask = th.ByteTensor(test_mask[sub_shadow_graph_index_attack_2])
    
    # =============================================================================
    # net_sub_attack_2 = Net_sub_attack_2()
    # 
    # optimizer_sub_attack_2 = th.optim.Adam(net_sub_attack_2.parameters(), lr=1e-2, weight_decay=5e-4)
    # dur = []
    # 
    # max_acc1 = 0
    # max_acc2 = 0
    # for epoch in range(300):
    #     if epoch >=3:
    #         t0 = time.time()
    #         
    #     net_sub_attack_2.train()
    #     logits = net_sub_attack_2(shadow_graph_attack_2, syn_features)
    #     logp = F.log_softmax(logits, 1)
    #     #print(logp[sub_shadow_graph_index_attack_2_train_mask])
    #     #print(labels[sub_shadow_graph_index_attack_2_train_mask])
    #     loss = F.nll_loss(logp[sub_shadow_graph_index_attack_2_train_mask], sub_shadow_graph_index_attack_2_labels[sub_shadow_graph_index_attack_2_train_mask])
    #     
    #     optimizer_sub_attack_2.zero_grad()
    #     loss.backward()
    #     optimizer_sub_attack_2.step()
    #     
    #     
    #     if epoch >=3:
    #         dur.append(time.time() - t0)
    #         
    #     acc1 = evaluate(net_sub_attack_2, shadow_graph_attack_2, syn_features, sub_shadow_graph_index_attack_2_labels, sub_shadow_graph_index_attack_2_test_mask)
    #     print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
    #             epoch, loss.item(), acc1, np.mean(dur)))
    #     
    #     if acc1>max_acc1:
    #         max_acc1 = acc1
    # 
    # print("Accuracy: " + str(acc1))
    # th.save(net_sub_attack_2.state_dict(), "./models/attack_6_pubmed_attack_2_model_all_shadow.pkl")
    # =============================================================================
    
    
    ##train sub attack-3 model
    
    # =============================================================================
    # data = citegrh.load_pubmed()
    # features_np = data.features
    # labels = data.labels
    # train_mask = data.train_mask
    # test_mask = data.test_mask
    # g = nx.to_numpy_array(data.graph)
    # g_matrix = np.asmatrix(g.copy())
    # 
    # for i in sub_shadow_graph_index_attack_3:
    #     test_mask[i] = 1
    #     train_mask[i] = 0
    # 
    # # =============================================================================
    # # for i in sub_shadow_graph_index_attack_2:
    # #     test_mask[i] = 1
    # #     train_mask[i] = 0
    # # =============================================================================
    # 
    # train_num = 0
    # while train_num < 70:
    #     random_i = np.random.randint(0,len(sub_shadow_graph_index_attack_3)-1)
    #     i = sub_shadow_graph_index_attack_3[random_i]
    #     if train_mask[i] == 0:
    #         test_mask[i] = 0
    #         train_mask[i] = 1
    #         train_num = train_num + 1
    # 
    # sub_shadow_graph_index_attack_2_test_mask = th.ByteTensor(test_mask[sub_shadow_graph_index_attack_2])
    # 
    # sub_shadow_graph_index_attack_3_features_np = features_np[sub_shadow_graph_index_attack_3]
    # sub_shadow_graph_index_attack_3_features = th.FloatTensor(sub_shadow_graph_index_attack_3_features_np)
    # 
    # sub_shadow_graph_index_attack_2_features_np = features_np[sub_shadow_graph_index_attack_2]
    # sub_shadow_graph_index_attack_2_features = th.FloatTensor(sub_shadow_graph_index_attack_2_features_np)
    # 
    # G = data.graph
    # g_numpy = nx.to_numpy_array(data.graph)
    # shadow_graph_index_attack_3_A = g_numpy[sub_shadow_graph_index_attack_3]
    # shadow_graph_attack_3_numpy = shadow_graph_index_attack_3_A[:,sub_shadow_graph_index_attack_3]
    # shadow_graph_attack_3 = nx.from_numpy_matrix(shadow_graph_attack_3_numpy)
    # 
    # # graph preprocess and calculate normalization factor
    # #sub_g_b = nx.from_numpy_array(sub_g_b)
    # # add self loop
    # 
    # shadow_graph_attack_3.remove_edges_from(nx.selfloop_edges(shadow_graph_attack_3))
    # shadow_graph_attack_3.add_edges_from(zip(shadow_graph_attack_3.nodes(), shadow_graph_attack_3.nodes()))
    # 
    # 
    # shadow_graph_attack_3 = DGLGraph(shadow_graph_attack_3)
    # shadow_graph_attack_3_edges = shadow_graph_attack_3.number_of_edges()
    # # normalization
    # degs = shadow_graph_attack_3.in_degrees().float()
    # norm = th.pow(degs, -0.5)
    # norm[th.isinf(norm)] = 0
    # 
    # shadow_graph_attack_3.ndata['norm'] = norm.unsqueeze(1)
    # 
    # 
    # #train target model
    # 
    # labels = th.LongTensor(labels[sub_shadow_graph_index_attack_3])
    # train_mask = th.ByteTensor(train_mask[sub_shadow_graph_index_attack_3])
    # test_mask = th.ByteTensor(test_mask[sub_shadow_graph_index_attack_3])
    # 
    # net_sub_attack_3 = Net_sub_attack_3()
    # 
    # optimizer_sub_attack_3 = th.optim.Adam(net_sub_attack_3.parameters(), lr=5e-3, weight_decay=5e-4)
    # dur = []
    # 
    # max_acc1 = 0
    # max_acc2 = 0
    # for epoch in range(400):
    #     if epoch >=3:
    #         t0 = time.time()
    #         
    #     net_sub_attack_3.train()
    #     logits = net_sub_attack_3(shadow_graph_attack_3, sub_shadow_graph_index_attack_3_features)
    #     logp = F.log_softmax(logits, 1)
    #     #print(logp[train_mask].size())
    #     #print(labels[train_mask].size())
    #     loss = F.nll_loss(logp[train_mask], labels[train_mask])
    #     
    #     optimizer_sub_attack_3.zero_grad()
    #     loss.backward()
    #     optimizer_sub_attack_3.step()
    #     
    #     
    #     if epoch >=3:
    #         dur.append(time.time() - t0)
    #     
    #     #print(sub_shadow_graph_index_attack_3_features[test_mask].size())
    #     #print(labels.size())
    #     #print(test_mask.size())
    #     
    #     acc1 = evaluate(net_sub_attack_3, shadow_graph_attack_3, sub_shadow_graph_index_attack_3_features, labels, test_mask)
    #     acc2 = evaluate(net_sub_attack_3, shadow_graph_attack_2, sub_shadow_graph_index_attack_2_features, sub_shadow_graph_index_attack_2_labels, sub_shadow_graph_index_attack_2_test_mask)
    #     print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} // {:.4f}| Time(s) {:.4f}".format(
    #             epoch, loss.item(), acc1, acc2, np.mean(dur)))
    #     
    #     if acc1>max_acc1:
    #         max_acc1 = acc1
    # 
    # print("Accuracy: " + str(acc1))
    # th.save(net_sub_attack_3.state_dict(), "./models/attack_6_pubmed_attack_3_model_all_shadow.pkl")
    # =============================================================================
    
    
    ##train the combined factors
    
    ###load the two model
    model_sub_attack_2 = Net_sub_attack_2()
    model_sub_attack_2.load_state_dict(th.load("./models/attack_6_pubmed_attack_2_model_all_shadow.pkl"))
    model_sub_attack_2.eval()
    
    model_sub_attack_3 = Net_sub_attack_3()
    model_sub_attack_3.load_state_dict(th.load("./models/attack_6_pubmed_attack_3_model_all_shadow.pkl"))
    model_sub_attack_3.eval()
    
    ###prepare parameters:
    
    
    syn_features_np = np.eye(SUB_ATTACK_2_NODE_NUM)
    
    syn_features = th.FloatTensor(syn_features_np)
    
    
    data = citegrh.load_pubmed()
    train_mask = data.train_mask
    test_mask = data.test_mask
    
    
    G = data.graph
    g_numpy = nx.to_numpy_array(data.graph)
    shadow_graph_index_attack_2_A = g_numpy[sub_shadow_graph_index_attack_2]
    shadow_graph_attack_2_numpy = shadow_graph_index_attack_2_A[:,sub_shadow_graph_index_attack_2]
    shadow_graph_attack_2 = nx.from_numpy_matrix(shadow_graph_attack_2_numpy)
    
    # graph preprocess and calculate normalization factor
    #sub_g_b = nx.from_numpy_array(sub_g_b)
    # add self loop
    
    shadow_graph_attack_2.remove_edges_from(nx.selfloop_edges(shadow_graph_attack_2))
    shadow_graph_attack_2.add_edges_from(zip(shadow_graph_attack_2.nodes(), shadow_graph_attack_2.nodes()))
    
    shadow_graph_attack_2 = DGLGraph(shadow_graph_attack_2)
    shadow_graph_attack_2_edges = shadow_graph_attack_2.number_of_edges()
    # normalization
    degs = shadow_graph_attack_2.in_degrees().float()
    norm = th.pow(degs, -0.5)
    norm[th.isinf(norm)] = 0
    
    shadow_graph_attack_2.ndata['norm'] = norm.unsqueeze(1)
    
    features_np = data.features
    sub_shadow_graph_index_attack_2_features_np = features_np[sub_shadow_graph_index_attack_2]
    sub_shadow_graph_index_attack_2_features = th.FloatTensor(sub_shadow_graph_index_attack_2_features_np)
    
    labels = data.labels
    sub_shadow_graph_index_attack_2_labels = th.LongTensor(labels[sub_shadow_graph_index_attack_2])
    
    
    
    attack_6_train_mask_np = train_mask[sub_shadow_graph_index_attack_2]
    attack_6_test_mask_np = test_mask[sub_shadow_graph_index_attack_2]
    
    for i in range(len(sub_shadow_graph_index_attack_2)):
        attack_6_train_mask_np[i] = 0
        attack_6_test_mask_np[i] = 1
    
    attack_6_train_num = 0
    while attack_6_train_num < 140:
        i = np.random.randint(0,len(sub_shadow_graph_index_attack_2)-1)
        #i = sub_shadow_graph_index_attack_2[random_i]
        if attack_6_train_mask_np[i] == 0:
            attack_6_train_mask_np[i] = 1
            attack_6_test_mask_np[i] = 0
            attack_6_train_num = attack_6_train_num + 1
        
    
    attack_6_train_mask = th.ByteTensor(attack_6_train_mask_np)
    attack_6_test_mask = th.ByteTensor(attack_6_train_mask_np)
    
    ###define a new model
    net_attack_6 = Net_attack_6()
    
    optimizer_attack_6 = th.optim.Adam(net_attack_6.parameters(), lr=5e-3, weight_decay=5e-4)
    #optimizer_attack_6 = th.optim.SGD(net_attack_6.parameters(), lr=0.01)
    criterion = th.nn.NLLLoss()
    max_acc1 = 0
    
    ###
    for epoch in range(100):
        net_attack_6.train()
        optimizer_attack_6.zero_grad()
        # Forward pass
        #print(syn_features.size())
        #print(sub_shadow_graph_index_attack_2_features.size())
        y1_pred = model_sub_attack_2(shadow_graph_attack_2, syn_features)
        y2_pred = model_sub_attack_3(shadow_graph_attack_2, sub_shadow_graph_index_attack_2_features)
        #print(y1_pred.size())
        #print(y2_pred.size())
        
        x_data = th.cat([y1_pred,y2_pred],1)
        
        #print(x_data.size())
        
        y_pred = net_attack_6(x_data[attack_6_train_mask])
        
        #print(y_pred.size())
        #print(sub_shadow_graph_index_attack_2_labels.size())
        
        # Compute Loss
        loss = criterion(y_pred, sub_shadow_graph_index_attack_2_labels[attack_6_train_mask])
        # Backward pass
        loss.backward()
        optimizer_attack_6.step()
        
        #print(y_pred.size())
        #print(sub_shadow_graph_index_attack_2_labels.size())
        acc1 = attack_model_evaluate_print(net_attack_6,x_data,sub_shadow_graph_index_attack_2_labels, attack_6_test_mask)
        
        print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f}".format(
                epoch, loss.item(), acc1))
    
    th.save(net_attack_6.state_dict(), "./models/attack_6_pubmed_attack_6_model_correct_all_shadow.pkl")
    
    #read attack-2 model
    model_attack_2 = Net_attack_2()
    model_attack_2.load_state_dict(th.load("./models/attack_6_pubmed_attack_2_model_no_sub.pkl"))
    model_attack_2.eval()
    
    #read attack-3 model
    model_attack_3 = Net_attack_3()
    model_attack_3.load_state_dict(th.load("./models/attack_3_subgraph_shadow_model_pubmed_8063.pkl"))
    model_attack_3.eval()
    
    #generate the parameters
    shadow_graph_index = []
    fileObject = open('./data/pubmed/protential_1300_shadow_graph_index.txt', 'r')
    contents = fileObject.readlines()
    for ip in contents:
        shadow_graph_index.append(int(ip))
    fileObject.close()
    
    target_graph_index = []
    for i in range(NODE_NUM):
        if i not in shadow_graph_index:
            target_graph_index.append(i)
    
    data = citegrh.load_pubmed()
    features = data.features
    labels = data.labels
    train_mask = data.train_mask
    test_mask = data.test_mask
    
    model = Net()
    model.load_state_dict(th.load("./models/improved_target_model_pubmed_8000.pkl"))
    model.eval()
    
    
    features_query = th.FloatTensor(features)
    g_graph = DGLGraph(data.graph)
    
    #=================Generate Label===================================================
    logits_query = model(g_graph, features_query)
    _, labels_query = th.max(logits_query, dim=1)
    
    target_graph_index_query = th.LongTensor(labels_query.detach().numpy()[target_graph_index])
    
    G = data.graph
    g_numpy = nx.to_numpy_array(data.graph)
    
    target_graph_index_g_A = g_numpy[target_graph_index]
    target_graph_index_g = target_graph_index_g_A[:,target_graph_index]
    
    target_graph_index_g = nx.from_numpy_array(target_graph_index_g)
    
    target_graph_index_g.remove_edges_from(nx.selfloop_edges(target_graph_index_g))
    target_graph_index_g.add_edges_from(zip(target_graph_index_g.nodes(), target_graph_index_g.nodes()))
    
    target_graph_index_g = DGLGraph(target_graph_index_g)
    target_graph_index_g_edges = target_graph_index_g.number_of_edges()
    # normalization
    degs = target_graph_index_g.in_degrees().float()
    norm = th.pow(degs, -0.5)
    norm[th.isinf(norm)] = 0
    
    target_graph_index_g.ndata['norm'] = norm.unsqueeze(1)
    
    target_graph_index_features = features[target_graph_index]
    
    target_graph_index_labels = labels[target_graph_index]
    
    target_graph_index_train_mask = train_mask[target_graph_index]
    
    target_graph_index_test_mask = test_mask[target_graph_index]
    
    for i in range(len(target_graph_index_test_mask)):
        target_graph_index_test_mask[i] = 1
    
    # =============================================================================
    # train_num = 0
    # while train_num < 250:
    #     random_i = np.random.randint(0,len(sub_shadow_graph_index_attack_2)-1)
    #     i = sub_shadow_graph_index_attack_2[random_i]
    #     if train_mask[i] == 0:
    #         test_mask[i] = 1
    #         train_mask[i] = 1
    #         train_num = train_num + 1
    # =============================================================================
    
    syn_features_test_np = np.eye(len(target_graph_index))
    syn_features_test = th.FloatTensor(syn_features_test_np)
    
    target_graph_index_features = th.FloatTensor(target_graph_index_features)
    target_graph_index_labels = th.LongTensor(target_graph_index_labels)
    
    y1_pred = model_attack_2(target_graph_index_g, syn_features_test)
    y2_pred = model_attack_3(target_graph_index_g, target_graph_index_features)
    #print(y1_pred.size())
    #print(y2_pred.size())
    
    x_data = th.cat([y1_pred,y2_pred],1)
    
    #print(x_data.size())
    
    y_pred = net_attack_6(x_data)
    
    #print(y_pred.size())
    #print(sub_shadow_graph_index_attack_2_labels.size())
    
    acc1 = attack_model_evaluate(net_attack_6,x_data,target_graph_index_labels)
    acc2 = attack_model_evaluate(net_attack_6,x_data,target_graph_index_query)
    
    print(acc1)
    print(acc2)
    #test the performance
