#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/13 9:56
# @Author  : Zhaohang
'''
下面的代码会自动加载数据，如果是第一次调用，数据会保存在你的hone目录下~/.keras/datasets/mnist.pkl.gz，大约15MB。
代码加载了数据集并画出了前4个图片
'''
# Plot ad hoc mnist instances
from keras.datasets import mnist
import matplotlib.pyplot as plt
# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()


