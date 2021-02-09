# MEA-GNN

## Enviroments Requires

* Pytorch
* Dgl

## Usage

### Parameters

* attack types

Please specify the attack you propose to run. They are list as the following table:
| Attack Types | Node Attribute | Graph Structure | Shadow Dataset |
| ----------   | ------------   | --------------- | -------------- |
| Attack-0 | Partially Known | Partially Known | Unknown |
| Attack-1 | Partially Known | Unknown | Unknown |
| Attack-2 | Unknown | Known | Unknown |
| Attack-3 | Unknown | Unknown | Known |
| Attack-4 | Partially Known | Partially Known | Known |
| Attack-5 | Partially Known | Unknown | Known |
| Attack-6 | Unknown | Known | Known |

* target model dataset

Please specify the dataset among Cora, Citeseer, Pubmed for your target model training.

* attack node number

Please specify the proportion of the nodes obtained by the adversary. 

* shadow dataset size (required for attack3/4/5/6)

For attacks with knowledge about the shadow dataset, please specify the size of shadow dataset comparing with training dataset as ratios.
For example, for shadow dataset with half size as the target dataset, please use 0.5.

### Example

For runing the attack-0 in Cora with 5% attack nodes obtained by the adversary, you can run the comment as:

`` python main.py --attack 0 --``

