# Model Extraction Attacks against Graph Neural Network

The source code for AsiaCCS2022 paper: "Model Extraction Attacks on Graph Neural Networks: Taxonomy and Realization ".
The paper can be found in [https://arxiv.org/abs/2010.12751](https://arxiv.org/abs/2010.12751)

If you make use of this code in your work, please cite the following paper:
<pre>
@inproceedings{wypy2022meagnn,
  title={Model Extraction Attacks on Graph Neural Networks: Taxonomy and Realization},
  author={Bang, Wu and Xiangwen, Yang and Shirui, Pan and Xingliang, Yuan},
  booktitle = {{ASIA} {CCS} '22: {ACM} Asia Conference on Computer and Communications
               Security, Nagasaki, May 30 - June 3, 2022},
  year={2022},
  publisher = {{ACM}}
}
</pre>

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

For runing the attack-0 in Cora with 25% attack nodes obtained by the adversary, you can run the comment as:

`` python main.py --attack_type 0 --dataset cora --attack_node 0.25``


If you have any questions, please send an email to us.
