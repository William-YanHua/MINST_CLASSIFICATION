# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 12:34:45 2018

@author: william
"""

import tensorflow as tf
from readUbyte import decodeIdx1UByte, decodeIdx3UByte
# 训练集
train_image = './trainning_set/train-images.idx3-ubyte'
train_label = './trainning_set/train-labels.idx1-ubyte'
# 测试集
test_image = './test_set/t10k-images.idx3-ubyte'
test_label = './test_set/t10k-labels.idx1-ubyte'


