# Selective Joint Fine-tuning

By [Weifeng Ge], [Yizhou Yu](http://i.cs.hku.hk/~yzyu/)

Department of Computer Science, The University of Hong Kong

### Table of Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Pipeline](#pipeline)
0. [Codes and Installation](#codes-and-installation)
0. [Models](#models)
0. [Results](#results)

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

### Pipeline
0. Pipeline of the proposed selective joint fine-tuning:
	![Selective Joint Fine-tuning Pipeline](https://github.com/ZYYSzj/Selective-Joint-Fine-tuning/blob/master/selective_joint_ft/cvpr2017_img1.png)


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

0. Multi crop testing error on Stanford Dogs 120 (in the same manner with that in [VGG-net](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)):

	Method|mean Accuracy(%)
	:---:|:---:
	[HAR-CNN](http://www.linyq.com/hyper-cvpr2015.pdf)|49.4
	[Local Alignment](https://link.springer.com/article/10.1007/s11263-014-0741-5)|57.0
	[Multi Scale Metric Learning](https://arxiv.org/abs/1402.0453)|70.3
	[MagNet](https://arxiv.org/abs/1511.05939)|75.1
	[Web Data + Original Data](https://arxiv.org/abs/1511.06789)|85.9
	Target Only Training from Scratch|53.8
	Selective Joint Training from Scratch|83.4
	Fine-tuning w/o source domain|80.4
	Selective Joint FT with all source samples|85.6
	Selective Joint FT with random source samples|85.5
	Selective Joint FT w/o iterative NN retrieval|88.3
	Selective Joint FT with Gabor filter bank|87.5
	Selective Joint FT|90.2
	Selective Joint FT with Model Fusion|90.3
	
0. Multi crop testing error on Oxford Flowers 102 (in the same manner with that in [VGG-net](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)):

	Method|mean Accuracy(%)
	:---:|:---:
	[MPP](http://ieeexplore.ieee.org/document/7301274/)|91.3
	[Multi-model Feature Concat](https://arxiv.org/abs/1406.5774)|91.3
	[MagNet](https://arxiv.org/abs/1511.05939)|91.4
	[VGG-19 + GoogleNet + AlexNet](https://arxiv.org/abs/1506.02565)|94.5
	Target Only Training from Scratch|58.2
	Selective Joint Training from Scratch|80.6
	Fine-tuning w/o source domain|90.2
	Selective Joint FT with all source samples|93.4
	Selective Joint FT with random source samples|93.2
	Selective Joint FT w/o iterative NN retrieval|94.2
	Selective Joint FT with Gabor filter bank|93.8
	Selective Joint FT|94.7
	Selective Joint FT with Model Fusion|95.8
	[VGG-19 + Part Constellation Model](https://arxiv.org/abs/1504.08289)|95.3
	Selective Joint FT with val set|97.0

0. Multi crop testing error on Caltech 256 (in the same manner with that in [VGG-net](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)):

	Method|mean Acc(%) 15/class|mean Acc(%) 30/class|mean Acc(%) 45/class|mean Acc(%) 60/class
	:---:|:---:|:---:|:---:|:---:
	[HAR-CNN](http://www.linyq.com/hyper-cvpr2015.pdf)|49.4%
	[Local Alignment](https://link.springer.com/article/10.1007/s11263-014-0741-5)|57.0%
	[Multi Scale Metric Learning](https://arxiv.org/abs/1402.0453)|70.3%
	[MagNet](https://arxiv.org/abs/1511.05939)|75.1%
	[Web Data + Original Data](https://arxiv.org/abs/1511.06789)|85.9%
	Target Only Training from Scratch|53.8%
	Selective Joint Training from Scratch|83.4%
	Fine-tuning w/o source domain|80.4%
	Selective Joint FT with all source samples|85.6%
	Selective Joint FT with random source samples|85.5%
	Selective Joint FT w/o iterative NN retrieval|88.3%
	Selective Joint FT with Gabor filter bank|87.5%
	Selective Joint FT|90.2%
	Selective Joint FT with Model Fusion|90.3%
