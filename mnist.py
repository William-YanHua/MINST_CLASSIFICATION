# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 11:43:24 2018

@author: william(Hua Yan)
"""
# 所有引入的全局包位置
import numpy as np
from readUbyte import decodeIdx3UByte, decodeIdx1UByte
#import matplotlib.pyplot as plt
#训练集文件
train_image = './trainning_set/train-images.idx3-ubyte'
train_label = './trainning_set/train-labels.idx1-ubyte'

test_image = './test_set/t10k-images.idx3-ubyte'
test_label = './test_set/t10k-labels.idx1-ubyte'

def logisticFunc(X, W, b):
    X_W = -np.dot(X,W)
    result = 1/(1 + np.exp(X_W + b))
    return result

def gradient(X, y_hat, y):
    grads_w = X * (y_hat - y)
    grads_b = y_hat - y
    grads_w = np.sum(grads_w, axis = 0)
#    print("grads_w shape", np.shape(grads_w))
    grads_b = np.sum(grads_b)
    grads = {}
    grads.setdefault('W', grads_w)
    grads.setdefault('b', grads_b)
    return grads

def gradient_reg(X, y_hat, y, W, lambda_ = 0.01):
    grads_w = X * (y_hat - y)
    grads_b = y_hat - y
    grads_w = np.sum(grads_w, axis = 0) + lambda_ * W.T
#    print("grads_w shape", np.shape(grads_w))
    grads_b = np.sum(grads_b)
    grads = {}
    grads.setdefault('W', grads_w)
    grads.setdefault('b', grads_b)
    return grads

def update(W, b, grads, learning_rate = 0.7):
    w,b = grads['W'], grads['b']
    w = w.reshape((784, 1))
    temp_w = W - learning_rate * w
    temp_b = b - learning_rate * b
    params = {}
    params.setdefault('W', temp_w)
    params.setdefault('b', temp_b)
    return params

def difference(Y, y_hat):
    result =  1-(Y == y_hat)
    return np.sum(result)
  
def reshapeX(X):    
    print("type of X %s" % (type(X[0])))
    m, n = np.shape(X[0])
    print("old_shape %d * %d" % (m, n))
    result = np.zeros((len(X), m * n))
    for i in range(len(X)):
        temp_x = X[i].flatten()
        result[i] = temp_x
    print("new shape ", np.shape(result))
    return result

def reshapeY(Y):
    result = np.zeros((len(Y), 1))
    for i in range(len(Y)):
        result[i] = Y[i]
    return result

def logisticRegression(X, Y,threshold = 0.02, learing_rate = 1, alpha = 0.99, alphaNode = 100, baseModel = None, reg = False):
    (m,n) = np.shape(X[0])
    trainning_size = len(Y)
    # 重构X
    X = reshapeX(X)
    X = (X-np.mean(X)) / (np.max(X) - np.min(X))
    
    # 初始化参数
    if (baseModel == None):
        W = np.ones((m * n, 1))
        b = 0
    else:
        W = baseModel['W']
        b = baseModel['b']
    # 进行梯度下降优化
    predict = logisticFunc(X, W, b)
    global grads
    if (reg):
        grads = gradient_reg(X, predict, Y, W)
    else:
        grads = gradient(X, predict, Y)
    i = 0
#    alpha = 0.999
    diff = difference(Y, predict)
    print("difference ", diff)
# =============================================================================
#     # 此处进行随机梯度下降
#     while (diff > threshold * trainning_size):
#         for x in range(trainning_size):
#             predict = logisticFunc(X[x], W, b)
#             grads = gradient(X[i], predict, Y[i])
#             params = update(W, b, grads, learing_rate)
#             W, b = params['W'], params['b']
#             if (x % 100 == 0 and x > 0):
#                 predict = logisticFunc(X, W, b)
#                 diff = difference(Y, predict)
#                 print('difference ', diff)
#                 if (x % 1000 == 0 and x > 0):
#                     predict[predict >= 0.5] = 1
#                     predict[predict < 0.5] = 0
#                     diff = difference(Y, predict)
#                     print("difference after regularize", diff)
#                     if (diff <= 0.02 * trainning_size):
#                         diff = threshold * trainning_size
#                         break
# =============================================================================
        
#    此处是普通的梯度下降，针对整个数据集
    old_different = diff
    print("grads_w shape ", np.shape(grads['W']))
    while(diff > threshold * trainning_size):
        if (i % alphaNode == 0 and i > 0):
            print("iteration times : %d" % i)
            print("difference ", diff)
            learing_rate = learing_rate * alpha
            if(i % (5 * alphaNode) == 0):
                predict[predict > 0.5] = 1
                predict[predict <= 0.5] = 0
                diff_reg = difference(Y, predict)
                print("diff after regularize ", diff_reg)
                if (old_different < diff):
                    break
            if (i >= 1000):
                break
        params = update(W, b, grads, learing_rate)
        W,b = params['W'], params['b']
        old_different = diff
        predict = logisticFunc(X, W, b)
        if (reg):
            grads = gradient_reg(X, predict, Y, W)
        else:
            grads = gradient(X, predict, Y)
        i = i + 1
        diff = difference(Y, predict)
    print("different ", diff)
    predict[predict > 0.5] = 1
    predict[predict <= 0.5] = 0
    diff_reg = difference(Y, predict)
    print("diff after regularize ", diff_reg)
    return W, b



def f1Measure(prediction, Y):
    f1_scores = []
    for i in range(10):
        TP = 0
        FN = 0
        TN = 0
        FP = 0
        for j in range(len(Y)):
            if (Y[j] == i and prediction[j] == i):
                TP =TP + 1
            elif (Y[j] == i and prediction[j] != i):
                FN = FN + 1
            elif (prediction[j] == i and Y[j] != i):
                FP = FP + 1
            else:
                TN = TN + 1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)
        print("%d 的识别精确度为 %f, 召回率为 %f, F1-Score为 %f" % (i, precision, recall, f1))
    return f1_scores

def predict(X):
    W, b = loadModel()
    X = reshapeX(X)
    y = logisticFunc(X, W, b)
    result = y.argmax(axis = 1)
    return result

def trainI(i, X, Y, threshold = 0.02, learing_rate = 1, alpha = 0.99, alphaNode = 100, baseModel = None, reg = False):
    print("=================get params for %d =================" % i)
    temp_Y = (Y == i)
    temp_Y[temp_Y == True] = 1
    temp_Y[temp_Y == False] = 0
    print(temp_Y)
    temp_W, temp_b = logisticRegression(X, temp_Y, threshold, \
                                        learing_rate, alpha, \
                                        alphaNode, baseModel, reg)
    return temp_W, temp_b

def saveModel(W, b, filename = 'lr_model_one_vs_all', backward = ''):
    import pickle
    params = {'W': W, 'b': b}
    with open(filename + backward, 'wb') as file:
        pickle.dump(params, file)
        
def loadModel(filename = 'lr_model_one_vs_all', backward = ''):
    import pickle
    with open(filename, 'rb') as file:
        params = pickle.load(file)
        W,b = params['W'], params['b']
    return W,b

def oneVsAll(X, Y, train = True):
    import os
    global W, b
    Y = reshapeY(Y)
    if (os.path.exists('lr_model_one_vs_all') == False or train):
        # 初始化10个W
        m, n = np.shape(X[0])
        W = np.zeros((10, m * n, 1))
        b = np.zeros((10,1))
        for i in range(10):
            temp_W, temp_b = trainI(i, X, Y, reg = True)
            W[i] = temp_W
            b[i] = temp_b
        saveModel(W, b)
    else:
        W,b = loadModel()
    prediction = predict(X)
    diff = difference(prediction, Y)
    print("difference ", diff)
    print("precision ", 1 - diff / len(X))
#    进行F1 measure
    f1_scores = f1Measure(prediction, Y)
    for i in range(len(f1_scores)):
        if (f1_scores[i] <= 0.6):
            print("再训练 %d " % i)
            temp_W, temp_b = trainI(i, X, Y, threshold = 0.02,\
                   learing_rate = 1, alpha=0.9,\
                   alphaNode = 150, baseModel={'W': W[i], 'b': b[i]}, reg = True)
            W[i] = temp_W
            b[i] = temp_b
    saveModel(W, b)
    prediction = predict(X)
    f1Measure(prediction, Y)
# =============================================================================
#     for i in range(100):
#         print("----%d   ----%d     ----" % (Y[i], prediction[i]))
# =============================================================================
    return W, b

def main():
    # 读取训练集
    images = decodeIdx3UByte(train_image)
    labels = decodeIdx1UByte(train_label)
    test_images = decodeIdx3UByte(test_image) 
    test_labels = decodeIdx1UByte(test_label)
    # 进行分类匹配
    print(labels)
    W,b = oneVsAll(images, labels, train=False)
    # 查看分类效果
    # 进行测试
    prediction = predict(test_images)
    test_labels = reshapeY(test_labels)
    f1Measure(prediction, test_labels)
if __name__=='__main__':
    main()