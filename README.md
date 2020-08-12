### NEWS: a PyTorch version of the Weighted Batch Norm layers is available [here](https://github.com/mancinimassimiliano/pytorch_wbn)!


-----------------------------------------------------------------------------------------------------------------------------------------------


This is the official Caffe implementation of [Boosting Domain Adaptation by Discovering Latent Domains](http://research.mapillary.com/img/publications/CVPR18b.pdf). 

This code is forked from [BVLC/caffe](https://github.com/BVLC/caffe). For any issue not directly related to our additional layers, please refer to the upstream repository.

## Additional layers

In this Caffe version, two additional layers are provided:

### MultiModalBatchNormLayer

Allows to perform a weighted normalization with respect to one domain.
Differently form standard BatchNormLayer it takes one more input, which is a weight vector of dimension equal to the batch size. This vector represents the probability that each sample belongs to the domain represented by this MultiModalBatchNormLayer. As an example, the syntax is the following: 
	
    layer{
        name: "wbn"
        type: "MultiModalBatchNorm"
        bottom: "input"
        bottom: "weights"
        top: "output"
    }

In case we have 2 latent domains, the full mDA layer would be:

    layer{
        name: "wbn1"
        type: "MultiModalBatchNorm"
        bottom: "input_1"
        bottom: "weights_1"
        top: "output_1"
    }

    layer{
        name: "wbn2"
        type: "MultiModalBatchNorm"
        bottom: "input_2"
        bottom: "weights_2"
        top: "output_2"
    }

    layer{
        name: "wbn"
        type: "Eltwise"
        bottom: "output_1"
        bottom: "output_2"
        top: "output"
        eltwise_param{
            operation: SUM
        }
    }

Since the output of a MultiModalBatchNormLayer for each sample is already scaled for its probability, the final layer is a simple element-wise sum.

### EntropyLossLayer 

A simple entropy loss implementation with integrated softmax computation. We used the implementation of [AutoDIAL](https://github.com/ducksoup/autodial/).

## Networks and solvers
Under models/latentDA we provide prototxts and solvers for the experiments reported in the paper. In particular the folder contains:

* `resnet18_k3.prototxt` : the ResNet architecture used for the PACS experiments, with 3 latent domains.
* `alexnet_k2.prototxt` : the AlexNet architecture used for the Office31 experiments, with 2 latent domains.
* `alexnet_sourcek2_targetk2`.prototxt : the AlexNet architecture used for the Office-Caltech experiments in the multi-target scenario, with 2 latent domains for both source and target.
* `alexnet_k3.prototxt` : the AlexNet architecture used for the Office-Caltech experiments in the multi-target scenario, with 3 latent domains.
* `solver_pacs.prototxt` : the solver used for the PACS experiments.
* `solver_alexnet.prototxt` : the solver used for both the Office31 and Office-Caltech experiments.

Notice that each of these files have some fields delimited by `%` which must be specified before their usage.


## Abstract and citation

Current Domain Adaptation (DA) methods based on deep architectures assume that the source samples arise from a single distribution. However, in practice most datasets can be regarded as mixtures of multiple domains. In these cases exploiting single-source DA methods for learning target classifiers may lead to sub-optimal, if not poor, results. In addition, in many applications it is difficult to manually provide the domain labels for all source data points, i.e. latent domains should be automatically discovered. This paper introduces a novel Convolutional Neural Network (CNN) architecture which (i) automatically discovers latent domains in visual datasets and (ii) exploits this information to learn robust target classifiers. Our approach is based on the introduction of two main components, which can be embedded into any existing CNN architecture: (i) a side branch that automatically computes the assignment of a source sample to a latent domain and (ii) novel layers that exploit domain membership information to appropriately align the distribution of the CNN internal feature representations to a reference distribution. We test our approach on publicly-available datasets, showing that it outperforms state-of-the-art multi-source DA methods by a large margin.

    @inProceedings{mancini2018boosting,
	author = {Mancini, Massimilano and Porzi, Lorenzo and Rota Bul\`o, Samuel and Caputo, Barbara and Ricci, Elisa},
  	title  = {Boosting Domain Adaptation by Discovering Latent Domains},
  	booktitle = {Computer Vision and Pattern Recognition (CVPR)},
  	year      = {2018},
  	month     = {June}
    }


