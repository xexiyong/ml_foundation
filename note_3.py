#!/usr/bin/env python
# coding=utf-8

import numpy as np


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
print e_func(0, 0)
