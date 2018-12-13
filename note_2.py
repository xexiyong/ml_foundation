#!/usr/bin/env python
# coding=utf-8

import random
import numpy as np


def sign(x, threshold, s):
    if x >= threshold:
        return 1*s
    return -1*s


def generate_dataset(size):
    '''生成样本
    '''
    samples = list()
    for i in range(size):
        x = random.uniform(-1, 1)
        y = sign(x, 0, 1)
        seed = random.random()
        y = -1*y if seed < 0.2 else y
        samples.append([x, y])
    return samples


def program_17():
    threshold_list = list()
    error_rate_list = list()
    for i in range(5000):
        samples = generate_dataset(20)
        best_thresh = 0
        best_precision = 20
        best_s = 0
        for j in samples:
            # 选某个阈值
            thresh_x = j[0]
            precison = 0
            for v in samples:
                x = v[0]
                y = v[1]
                for s in [-1, 1]:
                    pred = sign(x, thresh_x, s)
                    if pred != y:
                        precison += 1
            if best_precision > precison:
                best_precision = precison
                best_thresh = thresh_x
                best_s = 0
        error_rate_list.append(best_precision)
        threshold_list.append(best_thresh)
    print np.median(np.array(threshold_list))
    print np.asarray(error_rate_list).mean()

program_17()

