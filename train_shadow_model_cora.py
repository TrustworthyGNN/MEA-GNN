# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 17:27:39 2020
This code is for training a GCN based on 
https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html
@author: Bang
"""
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

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
        self.layer1 = GCNLayer(1433, 16)
        self.layer2 = GCNLayer(16, 7)

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x
net = Net()
print(net)

from dgl.data import citation_graph as citegrh
import networkx as nx
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

import time
import numpy as np
g, features, labels, train_mask, test_mask = load_cora_data()
# =============================================================================
# data = citegrh.load_cora()
# o_features = data.features
# o_feature = o_features[1]
# data = citegrh.load_cora()
# o_labels = data.labels
# o_label = o_labels[1]
# n_labels = o_label.numpy()
# a = 0
# b = 0
# for i in range(1433):
#     a = a + o_feature[i]
#     if (o_feature[i]==0):
#         b = b + 1
# n_features = features.numpy()
# =============================================================================
model = Net()
model.load_state_dict(th.load("./models/target_model_cora_7840.pkl"))
model.eval()

print("====================================================================================")

#=================Generate the random query(normal distribution)========================================
np_query = np.random.randint(0,2,size=[2708,1433])
np_nomal_query = np.random.random((2708,1433))
for i in range(2708):
    np_nomal_query[i] = np_query[i]/sum(np_query[i])
features_query = th.FloatTensor(np_nomal_query)


#=================Generate Label===================================================
logits_query = model(g, features_query)
_, labels_query = th.max(logits_query, dim=1)
n_features = features.numpy()
"""
net_shadow = Net()
optimizer = th.optim.Adam(net_shadow.parameters(), lr=1e-2)
dur = []

for epoch in range(50):
    if epoch >=3:
        t0 = time.time()

    net_shadow.train()
    logits = net_shadow(g, features_query)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[train_mask], labels_query[train_mask])
# =============================================================================
#     loss = F.nll_loss(logits[train_mask], logits_query[train_mask])
# =============================================================================

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >=3:
        dur.append(time.time() - t0)

    acc = evaluate(net_shadow, g, features, labels, test_mask)
    acc1 = evaluate(net_shadow, g, features_query, labels_query, test_mask)
    print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} |Synthetical Test Acc {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), acc, acc1, np.mean(dur)))
    

th.save(net_shadow.state_dict(), "./models/shadow_model_UniformDistribution_LabelOnly_cora.pkl")

"""
print("====================================================================================")


net_shadow1 = Net()
optimizer = th.optim.Adam(net_shadow1.parameters(), lr=1e-2)
dur = []
for epoch in range(50):
    if epoch >=3:
        t0 = time.time()

    net_shadow1.train()
    logits = net_shadow1(g, features_query)
# =============================================================================
#     logp = F.log_softmax(logits, 1)
#     loss = F.nll_loss(logp[train_mask], labels_query[train_mask])
# =============================================================================
    loss = F.l1_loss(logits[train_mask], logits_query[train_mask])

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    if epoch >=3:
        dur.append(time.time() - t0)

    acc = evaluate(net_shadow1, g, features, labels, test_mask)
    acc1 = evaluate(net_shadow1, g, features_query, labels_query, test_mask)
    print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} |Synthetical Test Acc {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), acc, acc1, np.mean(dur)))
    
th.save(net_shadow1.state_dict(), "./models/shadow_model_UniformDistribution_ConfidentialValues_cora.pkl")
#"""