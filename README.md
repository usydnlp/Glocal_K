# GLocal-K: Global and Local Kernels for Recommender Systems

This repository contains code for paper GLocal-K: Global and Local Kernels for Recommender Systems.

### <div align="center"> Han, C.\*, Lim, T.\*, Long, S., Burgstaller, B., & Poon, J. (2021, August). <br> GLocal-K: Global and Local Kernels for Recommender Systems <br> The 30th ACM International Conference on Information and Knowledge Management </div>

![GLocal_K_overview](https://user-images.githubusercontent.com/41948621/131093771-39d86126-6be6-4fc8-bcda-3eab8fd2c181.png)

## 1. Introduction
The proposed matrix completion framework based on global and local kernels, called GLocal-K, includes two stages: 1) pre-training an autoencoder using the local kernelised weight matrix, and 2) fine-tuning the pre-trained auto encoder with the rating matrix, produced by the global convolutional kernel. This repository provides the integrated implementation of two stages with two types of kernels on three benchmarks: ML-100K, ML-1M, and Douban.

## 2. Setup
Download this repository. As the code format is .ipynb, there are no settings but the Jupyter notebook with GPU.

## 3. Requirements
* numpy
* scipy
* tensorflow (converted to version 1.x automatically in the main code)

## 4. Run
1. Insert the path of a data directory on the main code by yourself (e.g., '/content/.../data').
2. Write down a dataset correctly among 'ML-1M', 'ML-100K', and 'Douban' on the main code.
3. There are no other things to do anymore, just try running the code and see it.
