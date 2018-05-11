#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
""" 
 @desc: 
 @author: ronny 
 @contact: set@aliyun.com 
 @site: www.xxxx.com 
 @software: PyCharm  @since:python 3.6.3 on 2018/05/5 21:20
 """
import tensorflow as tf

x = tf.Variable([[0.4, 0.2, 0.4]])
w = tf.Variable([[-0.5, -0.2], [-0.3, 0.4], [-0.5, 0.2]])
b = tf.Variable([[0.1, 0.2]])
xwb = tf.matmul(x, w) + b
y = tf.nn.sigmoid(tf.matmul(x, w) + b)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('xwb:', sess.run(xwb))
    print('y:', sess.run(y))
    print('x:', sess.run(x))
