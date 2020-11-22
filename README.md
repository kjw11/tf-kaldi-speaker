# TF-kaldi-speaker

This code is forked from [entn-at/tf-kaldi-speaker](https://github.com/entn-at/tf-kaldi-speaker).
It is a speaker verification system based on [Kaldi](https://github.com/kaldi-asr/kaldi) and [TensorFlow](https://github.com/tensorflow/tensorflow).
More detail please refer: [entn-at/tf-kaldi-speaker](https://github.com/entn-at/tf-kaldi-speaker).

## Features
This version has two features compared with the original branch:  

### Resnet-34 Topology
This is a famouse Resnet topology, the blocks are: **[3/32, 3/32], [3/64, 3/64], [3/128, 3/128], [3/256, 3/256]**, and the number of blocks is: **[3, 4, 6, 3]**

The code is [/model/resnet.py](/model/resnet.py).

### SITW Recipe
A SITW recipe is added in [egs/sitw](./egs/sitw), 
which is largely based on [SITW offical x-vector recipe](https://github.com/kaldi-asr/kaldi/tree/master/).

There are 8 exprimental settings in the SITW recipe, with different network topologies, pooling methods and loss functions. 
See [./egs/sitw/v1/nnet_conf](./egs/sitw/v1/nnet_conf). Note that the training and test data are the same as SITW offical recipe.

Some of the experimental results are shown below:

| Topoloy | Pooling | Loss func | EER(%) |
| :-----| :----: | :----: | :----: |
| TDNN | Statistic Pooling | Softmax | 2.43 |
| TDNN | Attention Pooling | AAM-Softmax | 2.49 |
| TDNN | Statistic Pooling | Softmax | 2.41 |
| TDNN | Attention Pooling | AAM-Softmax | 2.57 |
| Resnet-34 | Statistic Pooling | Softmax | 2.41 |
| Resnet-34 | Attention Pooling | AAM-Softmax | 1.96 |
| Resnet-34 | Statistic Pooling | Softmax | 2.16 |
| Resnet-34 | Attention Pooling | AAM-Softmax | 2.30 |

Where "AAM" means additive angular margin.

