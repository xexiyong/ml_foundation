#!/usr/bin/env python
# coding=utf-8

import numpy as np
import random


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


ques_fourt()
