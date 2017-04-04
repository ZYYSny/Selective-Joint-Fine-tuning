# Selective-Joint-Fine-tuning

By [Weifeng Ge], [Yizhou Yu](http://i.cs.hku.hk/~yzyu/)

The University of Hong Kong

### Table of Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Disclaimer and known issues](#disclaimer-and-known-issues)
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
