#!/usr/bin/env python
# coding=utf-8
import os
import urllib2
import numpy as np


def getRawDataSet(url):
    dataSet = urllib2.urlopen(url)
    filename = 'MLFex1_' + url.split('_')[1] + '_' + url.split('_')[2]
    with open(filename, 'w') as fr:
        fr.write(dataSet.read())
    return filename

def getDataSet(filename):
    # 从本地文件读取训练数据或测试数据，保存X,y两个变量中
    dataSet = open(filename, 'r')
    dataSet = dataSet.readlines()   # 将训练数据读出，存入dataSet变量中
    num = len(dataSet)  # 训练数据的组数
    # 提取X, Y
    X = np.zeros((num, 5))
    Y = np.zeros((num, 1))
    for i in range(num):
        data = dataSet[i].strip().split()
        X[i, 0] = 1.0
        X[i, 1] = np.float(data[0])
        X[i, 2] = np.float(data[1])
        X[i, 3] = np.float(data[2])
        X[i, 4] = np.float(data[3])
        Y[i, 0] = np.int(data[4])
    return X, Y


def sign(x, w):
    # sigmoid函数，返回函数值
    if np.dot(x, w)[0] >= 0:
        return 1
    else:
        return -1


def trainPLA_Naive(X, Y, w, eta, updates):
    iterations = 0  # 记录实际迭代次数
    num = len(X)    # 训练数据的个数
    flag = True
    for i in range(updates):
        flag = True
        for j in range(num):
            if sign(X[j], w) != Y[j, 0]:
                flag = False
                w += eta * Y[j, 0] * np.matrix(X[j]).T
                break
            else:
                continue
        if flag == True:
            iterations = i
            break
    return flag, iterations, w


def question15():
    # 使用此函数可直接解答15题
    url = 'https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_15_train.dat'
    filename = getRawDataSet(url)
    X, y = getDataSet(filename)
    w0 = np.zeros((5, 1))
    eta = 1
    updates = 80
    flag, iterations, w = trainPLA_Naive(X, y, w0, eta, updates)
    print flag
    print iterations
    print w

question15()