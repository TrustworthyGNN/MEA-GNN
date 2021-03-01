# -*- coding: utf-8 -*-
import argparse
from attacks.attack_0 import attack0
from attacks.attack_1 import attack1
from attacks.attack_2 import attack2
from attacks.attack_3 import attack3
from attacks.attack_4 import attack4
from attacks.attack_5 import attack5
from attacks.attack_6 import attack6

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', default=None, help="To use cuda, set \
                    to a specific GPU ID. Default set to use CPU.")

parser.add_argument('--attack_type', type=int, default=2,
                    help="int id of attack type")

parser.add_argument('--dataset', type=str, default='citeseer',
                    help="Dataset for the target model: (cora, citeseer, pubmed)")

parser.add_argument('--attack_node', type=float, default=0.25,
                    help='proportion of the attack nodes')

parser.add_argument('--shadow_dataset_size', type=float, default=1,
                    help='size of the shadow datasets')

args = parser.parse_args()

print(args.attack_node)

if args.attack_type == 0:
    attack0(args.dataset, args.attack_node,args.gpu)
if args.attack_type == 1:
    attack1(args.dataset, args.attack_node,args.gpu)
if args.attack_type == 2:
    attack2(args.dataset, args.attack_node,args.gpu)
if args.attack_type == 3:
    attack3(args.dataset, args.attack_node,args.gpu)
if args.attack_type == 4:
    attack4(args.dataset, args.attack_node,args.gpu)
if args.attack_type == 5:
    attack5(args.dataset, args.attack_node,args.gpu)
if args.attack_type == 6:
    attack6(args.dataset, args.attack_node,args.gpu)