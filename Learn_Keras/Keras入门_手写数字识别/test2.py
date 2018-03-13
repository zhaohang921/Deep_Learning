#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/13 10:15
# @Author  : Zhaohang
'''

'''
#第一步依然是导入需要的函数库
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

#设定随机数种子
seed = 7
numpy.random.seed(seed)

#将数据集reshape，CNN的输入是4维的张量（可看做多维的向量），
# 第一维是样本规模，第二维是像素通道，第三维和第四维是长度和宽度。并将数值归一化和类别标签向量化。
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# 接下来构造CNN。
# 第一层是卷积层。该层有32个feature map,或者叫滤波器，作为模型的输入层，接受[pixels][width][height]大小的输入数据。feature map的大小是5*5，其输出接一个‘relu’激活函数。
# 下一层是pooling层，使用了MaxPooling，大小为2*2。
# 下一层是Dropout层，该层的作用相当于对参数进行正则化来防止模型过拟合。
# 接下来是全连接层，有128个神经元，激活函数采用‘relu’。
# 最后一层是输出层，有10个神经元，每个神经元对应一个类别，输出值表示样本属于该类别的概率大小。
def baseline_model():
    # create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#接着开始训练模型
# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))




