# -*- coding: utf-8 -*-
import argparse
from attacks.attack_0 import attack0

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', default=None, help="To use cuda, set \
                    to a specific GPU ID. Default set to use CPU.")

parser.add_argument('--attack_type', type=int, default=0,
                    help="int id of attack type")

parser.add_argument('--dataset', type=str, default='cora',
                    help="Dataset for the target model: (cora, citeseer, pubmed)")

parser.add_argument('--attack_node', type=float, default=0.25,
                    help='proportion of the attack nodes')

parser.add_argument('--shadow_dataset_size', type=float, default=1,
                    help='size of the shadow datasets')

args = parser.parse_args()

print(args.attack_node)

if args.attack_type == 0:
    attack0(args.dataset, args.attack_node,args.gpu)