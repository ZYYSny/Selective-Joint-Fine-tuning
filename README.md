# Selective Joint Fine-tuning

By [Weifeng Ge], [Yizhou Yu](http://i.cs.hku.hk/~yzyu/)

Department of Computer Science, The University of Hong Kong

### Table of Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Codes and Installation](#codes-and-installation)
0. [Models](#models)
0. [Results](#results)
0. [Third-party re-implementations](#third-party-re-implementations)

### Introduction

This repository contains the codes and models described in the paper "Borrowing Treasures from the Wealthy: Deep Transfer Learning through Selective Joint Fine-tuning"(https://arxiv.org/abs/1702.08690). These models are those used in [Stanford Dogs 120](http://vision.stanford.edu/aditya86/ImageNetDogs/), [Oxford Flowers 102](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/), [Caltech 256](http://authors.library.caltech.edu/7694/) and [MIT Indoor 67](http://web.mit.edu/torralba/www/indoor.html).

**Note**

0. All algorithms are implemented based on the deep learning framework [Caffe](https://github.com/BVLC/caffe).
0. Please add the additional layers used into your own Caffe to run the training codes.

### Citation

If you use these codes and models in your research, please cite:

	@article{ge2017borrowing,
	        title = {Borrowing Treasures from the Wealthy: Deep Transfer Learning through Selective Joint Fine-tuning},
		author = {Ge, Weifeng and Yu, Yizhou},
		journal = {arXiv preprint arXiv:1702.08690},
		year = {2017}
	}

### Codes and Installation



### Models

0. Visualizations of network structures (tools from [ethereon](http://ethereon.github.io/netscope/quickstart.html)):
	- [Selective Joint Fine-tuning: ResNet-152] (http://ethereon.github.io/netscope/#/gist/d38f3e6091952b45198b)

0. Model files:
	- Stanford Dogs 120: [link](https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777)
        - Oxford Flowers 102: [link](https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777)
	- Caltech 256: [link](https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777)
	- Mit Indoor 67: [link](https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777)
### Results
0. Curves on ImageNet (solid lines: 1-crop val error; dashed lines: training error):
	![Training curves](https://cloud.githubusercontent.com/assets/11435359/13046277/e904c04c-d412-11e5-9260-efc5b8301e2f.jpg)

0. 1-crop validation error on ImageNet (center 224x224 crop from resized image with shorter side=256):

	model|top-1|top-5
	:---:|:---:|:---:
	[VGG-16](http://www.vlfeat.org/matconvnet/pretrained/)|[28.5%](http://www.vlfeat.org/matconvnet/pretrained/)|[9.9%](http://www.vlfeat.org/matconvnet/pretrained/)
	ResNet-50|24.7%|7.8%
	ResNet-101|23.6%|7.1%
	ResNet-152|23.0%|6.7%
	
0. 10-crop validation error on ImageNet (averaging softmax scores of 10 224x224 crops from resized image with shorter side=256), the same as those in the paper:

	model|top-1|top-5
	:---:|:---:|:---:
	ResNet-50|22.9%|6.7%
	ResNet-101|21.8%|6.1%
	ResNet-152|21.4%|5.7%
	
