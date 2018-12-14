#!/usr/bin/env python
# coding=utf-8

import numpy as np
import random
import urllib2
import os

def d_u(u, v):
    return np.exp(u) + v* np.exp(u*v) + 2*u -2*v - 3


def d_v(u, v):
    return 2*np.exp(2*v) + u*np.exp(u*v) -2*u + 4*v - 2


def e_func(u, v):
    return np.exp(u) + np.exp(2*v) + np.exp(u*v) + np.power(u, 2) - 2*u*v + 2*np.power(v, 2) -3*u -2*v


def ques_seven():
    u = 0
    v = 0
    for i in range(5):
        b_u = u
        b_v = v
        u = u - 0.01 * d_u(b_u, b_v)
        v = v - 0.01 * d_v(b_u, b_v)

    print u, v
    print e_func(u, v)


# ques_seven()
# print e_func(0, 0)

def hessian(u, v):
    return np.array([
        [np.exp(u) + np.power(v, 2)*np.exp(u*v) + 2, np.exp(u*v)+u*v*np.exp(u*v) - 2],
        [np.exp(u*v)+u*v*np.exp(u*v) - 2, 4*np.exp(2*v) + np.power(u, 2)*np.exp(u*v) + 4]
    ])


def gradient(u, v):
    return np.array([
        [np.exp(u) + v* np.exp(u*v) + 2*u -2*v - 3],
        [2 * np.exp(2 * v) + u * np.exp(u * v) - 2 * u + 4 * v - 2],
    ])


def ques_ten():
    u = 0
    v = 0
    for i in range(5):
        b_u = u
        b_v = v
        delta = np.dot(np.linalg.inv(hessian(b_u, b_v)), gradient(b_u, b_v)).transpose()
        u = u - delta[0][0]
        v = v - delta[0][1]
    print u, v
    print e_func(u, v)

# ques_ten()


def sign(x):
    if x>= 0:
        return 1
    else:
        return -1


def generate_circle_sample(size):
    x = np.random.uniform(-1, 1, (size, 3))
    for i in x:
        t = sign(np.power(i[0], 2) + np.power(i[1], 2) - 0.6)
        prob_noise = np.random.random()
        if prob_noise < 0.1:
            t = -t
        i[2] = t
    return x


def ques_thirt():
    e_in = []
    for i in range(1000):
        raw = generate_circle_sample(1000)
        sample = np.mat([[1, v[0], v[1]] for v in raw])
        y = np.mat([[int(v[2])] for v in raw])
        # 3 * 1
        w_lin = np.dot(np.dot(np.linalg.inv(np.dot(sample.T, sample)), sample.T), y)
        print w_lin
        y_hat = np.dot(sample, w_lin)
        y_hat = [sign(v) for v in y_hat]

        error = 0
        for j, pred in enumerate(y_hat):
            if pred != int(y[j]):
                error += 1
        e_in.append(float(error) / 1000)
    print np.mean(e_in)

# ques_thirt()


def ques_fourt():
    raw = generate_circle_sample(1000)
    sample = np.mat([[1, v[0], v[1], v[0]*v[1], v[0]**2, v[1]**2] for v in raw])
    y = np.mat([[int(v[2])] for v in raw])
    w_lin = np.dot(np.dot(np.linalg.inv(np.dot(sample.T, sample)), sample.T), y)
    print w_lin

    # e_in = list()
    # for i in range(1000):
    #     raw = generate_circle_sample(1000)
    #     sample = np.mat([[1, v[0], v[1], v[0]*v[1], v[0]**2, v[1]**2] for v in raw])
    #     y = np.mat([[int(v[2])] for v in raw])
    #     y_hat = np.dot(sample, w_lin)
    #     y_hat = [sign(v) for v in y_hat]
    #
    #     error = 0
    #     for j, pred in enumerate(y_hat):
    #         if pred != int(y[j]):
    #             error += 1
    #     e_in.append(float(error) / 1000)
    # print np.mean(e_in)


# ques_fourt()

train_url = 'https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw3_train.dat'
test_url = 'https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw3_test.dat'


def get_dataset(url, f):
    content = urllib2.urlopen(url)
    dataset_f = os.path.join(os.getcwd(), f)
    fr = open(dataset_f, 'w')
    for line in content:
        print line
        fr.write(line)

    fr.close()


# get_dataset(train_url, 'hdw3_train.dat')
# get_dataset(test_url, 'hdw3_test.dat')

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
            y.append(item_y)
    return np.asarray(x), np.asarray(y)


def lr_gradient_descent(train_x, train_y, learning_rate=0.001, iter_num=2000):
    print 'train'
    columns = len(train_x[0])
    rows = len(train_y)
    w = np.zeros((1, columns))
    # print w

    train_y = train_y.reshape((1, rows))
    for i in range(iter_num):
        y_hat = calc(train_x, w.T).T
        # print y_hat
        # print train_y
        hx_y = y_hat - train_y
        # print y_hat.shape
        # print train_y.T.shape
        assert y_hat.shape == train_y.shape
        # 1 * n   n * d = 1 * d
        w = w - learning_rate * np.dot(hx_y, train_x)

    return w


def lr_sgd(train_x, train_y, learning_rate=0.001, iter_num=2000):
    print 'train'
    columns = len(train_x[0])
    rows = len(train_y)
    w = np.zeros((1, columns))

    train_y = train_y.reshape((1, rows))
    for i in range(iter_num):
        sample_x = train_x[i % rows, :]
        sample_y = train_y[:, i % rows]
        y_hat = calc(sample_x, w.T).T
        hx_y = y_hat - sample_y
        # print y_hat
        # print sample_y
        assert y_hat.shape == sample_y.shape
        # 1 * n   n * d = 1 * d
        # print hx_y
        # print sample_x
        w = w - learning_rate * np.multiply(hx_y, sample_x)

    return w


def calc(x, w):
    return 1.0 / (1 + np.exp(-1 * np.dot(x, w)))


def predict(w, test_x, test_y):
    print 'predict'
    err_rate = 0
    test_len = len(test_x)

    print w
    for x, y in zip(test_x, test_y):
        y_hat = sign(calc(x, w.T) - 0.5)
        # y_hat = 0 if y_hat < 0 else y_hat
        if y_hat != y:
            err_rate += 1
    return 1.0*err_rate / test_len


train_x, train_y = get_sample('hdw3_train.dat')
print train_x[0]
print train_y[0]
test_x, test_y = get_sample('hdw3_test.dat')
# w = lr_gradient_descent(train_x, train_y, learning_rate=0.001)
w = lr_sgd(train_x, train_y, learning_rate=0.001)
err_rate = predict(w, test_x, test_y)
print err_rate

