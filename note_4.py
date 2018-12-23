#!/usr/bin/env python
# coding=utf-8
# linear regression regularization
import urllib2
import os
import numpy as np


hw4_train = 'https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw4_train.dat'
hw4_test = 'https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw4_test.dat'


def get_dataset(url, f):
    content = urllib2.urlopen(url)
    dataset_f = os.path.join(os.getcwd(), f)
    fr = open(dataset_f, 'w')
    for line in content:
        print line
        fr.write(line)

    fr.close()


# get_dataset(hw4_test, 'hw4_test.dat')
# get_dataset(hw4_train, 'hw4_train.dat')


def get_sample(f):
    x = list()
    y = list()
    with open(f) as f:
        for line in f:
            raw_x = line.strip().split(' ')[:-1]
            raw_y = line.strip().split(' ')[-1]
            item_x = [float(v) for v in raw_x]
            item_y = 1 if not raw_y.startswith('-') else -1
            x.append([1] + item_x)
            y.append([item_y])
    return np.asarray(x), np.asarray(y)


# W_reg =  inv(X.T * X + lambda*I) * X.T * Y

def train_w_reg(x_train, y_train, _lambda):
    samples, columns = x_train.shape
    x_train.reshape((samples, columns))
    y_train.reshape((samples, 1))

    f1 = np.linalg.inv(np.dot(x_train.T, x_train) + _lambda * np.identity(columns))
    return np.dot(np.dot(f1, x_train.T), y_train)


def question_13(_lambda=10):
    x_train, y_train = get_sample('hw4_train.dat')
    x_test, y_test = get_sample('hw4_test.dat')

    # w_reg, columns * 1
    w_reg = train_w_reg(x_train, y_train, _lambda)

    # Ein
    y_pred = np.dot(x_train, w_reg)    # samples * 1
    # y_pred = np.asarray([1 if y[0]>=0 else -1 for y in y_pred])
    y_pred = np.where(y_pred >= 0, 1, -1)
    combine_res = np.where(y_pred == y_train, 0, 1)
    Ein = float(combine_res.sum()) / len(x_train)

    #Eout
    y_pred = np.dot(x_test, w_reg)
    y_pred = np.where(y_pred >= 0, 1, -1)
    combine_res = np.where(y_pred == y_test, 0, 1)
    Eout = float(combine_res.sum()) / len(x_test)

    print 'lambda: {}; Ein: {}; Eout: {}'.format(_lambda, Ein, Eout)


# question_13()


def question_14():
    lambda_set = [2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
    for _lambda in lambda_set:
        print 'lambda: ', _lambda
        _lambda = 10 ** _lambda
        question_13(_lambda)


# 14, 15
# question_14()


def validation(_lambda):
    x, y = get_sample('hw4_train.dat')
    x_test, y_test = get_sample('hw4_test.dat')

    x_train = x[:120]
    y_train = y[:120]
    x_val = x[120:]
    y_val = y[120:]

    # w_reg, columns * 1
    w_reg = train_w_reg(x_train, y_train, _lambda)

    # Ein
    y_pred = np.dot(x_train, w_reg)    # samples * 1
    # y_pred = np.asarray([1 if y[0]>=0 else -1 for y in y_pred])
    y_pred = np.where(y_pred >= 0, 1, -1)
    combine_res = np.where(y_pred == y_train, 0, 1)
    Ein = float(combine_res.sum()) / len(x_train)

    # Eval
    y_pred = np.dot(x_val, w_reg)
    y_pred = np.where(y_pred >= 0, 1, -1)
    combine_res = np.where(y_pred == y_val, 0, 1)
    Eval = float(combine_res.sum()) / len(x_val)

    #Eout
    y_pred = np.dot(x_test, w_reg)
    y_pred = np.where(y_pred >= 0, 1, -1)
    combine_res = np.where(y_pred == y_test, 0, 1)
    Eout = float(combine_res.sum()) / len(x_test)

    print 'lambda: {}; Ein: {}; Eval: {}; Eout: {}'.format(_lambda, Ein, Eval, Eout)


def question_16():
    lambda_set = [2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
    for _lambda in lambda_set:
        print 'lambda: ', _lambda
        _lambda = 10 ** _lambda
        validation(_lambda)


# question 17 : lambda=0
# question_16()

def question_18(_lambda=10):
    x_train, y_train = get_sample('hw4_train.dat')
    x_test, y_test = get_sample('hw4_test.dat')
    # x = np.concatenate((x_train, x_test) ,axis=0)
    # y = np.concatenate((y_train, y_test), axis=0)
    # w_reg, columns * 1
    w_reg = train_w_reg(x_train, y_train, _lambda)

    # Ein
    y_pred = np.dot(x_train, w_reg)    # samples * 1
    # y_pred = np.asarray([1 if y[0]>=0 else -1 for y in y_pred])
    y_pred = np.where(y_pred >= 0, 1, -1)
    combine_res = np.where(y_pred == y_train, 0, 1)
    Ein = float(combine_res.sum()) / len(x_train)

    #Eout
    y_pred = np.dot(x_test, w_reg)
    y_pred = np.where(y_pred >= 0, 1, -1)
    combine_res = np.where(y_pred == y_test, 0, 1)
    Eout = float(combine_res.sum()) / len(x_test)

    print 'lambda: {}; Ein: {}; Eout: {}'.format(_lambda, Ein, Eout)


# question_18(10 ** 0)


def cross_validation(_lambda):
    x, y = get_sample('hw4_train.dat')
    # x_test, y_test = get_sample('hw4_test.dat')

    Eval = 0
    for i in range(5):
        left = i*40
        right = (i+1)*40
        x_val = x[left: right]
        y_val = y[left: right]

        if left > 0 and right < len(x):
            x_top = x[0: left]
            x_bottom = x[right:]
            x_train = np.concatenate((x_top, x_bottom), axis=0)
            y_top = y[0: left]
            y_bottom = y[right:]
            y_train = np.concatenate((y_top, y_bottom), axis=0)

        elif left >0:
            x_train = x[0: left]
            y_train = y[0: left]
        elif right < len(x):
            x_train = x[right:]
            y_train = y[right:]
        # w_reg, columns * 1
        w_reg = train_w_reg(x_train, y_train, _lambda)

        # Ein
        # y_pred = np.dot(x_train, w_reg)    # samples * 1
        # # y_pred = np.asarray([1 if y[0]>=0 else -1 for y in y_pred])
        # y_pred = np.where(y_pred >= 0, 1, -1)
        # combine_res = np.where(y_pred == y_train, 0, 1)
        # Ein += float(combine_res.sum()) / len(x_train)

        # Eval
        y_pred = np.dot(x_val, w_reg)
        y_pred = np.where(y_pred >= 0, 1, -1)
        combine_res = np.where(y_pred == y_val, 0, 1)
        Eval += float(combine_res.sum()) / len(x_val)

    # #Eout
    # y_pred = np.dot(x_test, w_reg)
    # y_pred = np.where(y_pred >= 0, 1, -1)
    # combine_res = np.where(y_pred == y_test, 0, 1)
    # Eout = float(combine_res.sum()) / len(x_test)

    print 'lambda: {}; Eval: {};'.format(_lambda,  Eval / 5)


def question_19():
    lambda_set = [2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
    for _lambda in lambda_set:
        print 'lambda: ', _lambda
        _lambda = 10 ** _lambda
        cross_validation(_lambda)


# question_19()

question_18(10 ** -8)