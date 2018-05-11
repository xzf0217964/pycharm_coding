#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
""" 
 @desc: 
 @author: ronny 
 @contact: set@aliyun.com 
 @site: www.xxxx.com 
 @software: PyCharm  @since:python 3.6.3 on 2018/05/5 21:20
 """

import tensorflow as ft
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("C:\\Users\\xuzhenfan\\tensorflow\\MNIST_data\\", one_hot=True)
print('train', mnist.train.num_examples)

import matplotlib.pyplot as plt
import numpy as np


def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(np.reshape(images[idx], (28, 28)), cmap='binary')
        title = 'label=' + str(np.argmax(labels[idx]))
        if len(prediction) > 0:
            title += ', predict=' + str(prediction[idx])
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()


plot_images_labels_prediction(mnist.train.images, mnist.train.labels, [], 0)