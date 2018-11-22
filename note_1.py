#!/usr/bin/env python
# coding=utf-8
import os
import urllib2
import numpy as np

dataset_15 = 'https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_15_train.dat'
dataset_18 = 'https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_18_train.dat'
dataset_18_ = 'https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_18_test.dat'


def down_dataset(url, f):
    # donwload the dataset
    # response = requests.get(dataset_url)
    content = urllib2.urlopen(url)
    dataset_f = os.path.join(os.getcwd(), f)
    fr = open(dataset_f, 'w')
    for line in content:
        print line
        fr.write(line)

    fr.close()

# down_dataset(dataset_15, '15.dat')
# down_dataset(dataset_15, '18.dat')
# down_dataset(dataset_15, '18_.dat')


def load_15():
    sample = []
    with open(os.path.join(os.getcwd(), '15.dat'), 'r') as f:
        for line in f:
            x1, x2, x3, x4, y = line.split()
            sample.append([float(x1), float(x2), float(x3), float(x4), int(y)])
    return sample


def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


def train_pla_naive():
    iteration_num = 0
    w = np.array([0, 0, 0, 0, 0])
    sample = load_15()

    while True:
        converge = True
        for f_label in sample:
            f = np.array([1] + f_label[:4])
            y = f_label[4]
            if sign(np.dot(f, w)) != y:
                converge = False
                w = w + y*f
                break
        iteration_num += 1
        print w

        if converge:
            print('converge!')
            break
    print('iter times', iteration_num)

# question 15 converge after 61 times
# train_pla_naive()


def train_pla_randomly(learning_rate=1.0):
    iteration_num = 0
    w = np.array([0, 0, 0, 0, 0])
    sample = load_15()

    # make random seed
    sequence = range(len(sample))
    np.random.shuffle(sequence)

    while True:
        converge = True
        for v in sequence:
            f = np.array([1] + sample[v][:4])
            y = sample[v][4]
            if sign(np.dot(f, w)) != y:
                converge = False
                w = w + learning_rate*y*f
                break
        iteration_num += 1
        if converge:
            print('converge!')
            break

    return iteration_num

# question 16 after 2000 experiment average times is 2.732  66.9475
# iter_array = 0
# for i in range(2000):
#     iter_times = train_pla_randomly()
#     iter_array += iter_times
#
# print float(iter_array) / 2000

# question 17 after 2000 experiment with learning rate average times is 2.7455 66.2705
# iter_array = 0
# for i in range(2000):
#     iter_times = train_pla_randomly(learning_rate=0.5)
#     iter_array += iter_times
#
# print float(iter_array) / 2000


def load_18():
    sample = []
    with open(os.path.join(os.getcwd(), '18.dat'), 'r') as f:
        for line in f:
            x1, x2, x3, x4, y = line.split()
            sample.append([float(x1), float(x2), float(x3), float(x4), int(y)])
    return sample


def load_18_test():
    sample = []
    with open(os.path.join(os.getcwd(), '18_.dat'), 'r') as f:
        for line in f:
            x1, x2, x3, x4, y = line.split()
            sample.append([float(x1), float(x2), float(x3), float(x4), int(y)])
    return sample


def train_pla_pure_randomly(train, test, use_pocket=True, update_num=50):
    w = np.array([0, 0, 0, 0, 0])

    for i in range(update_num):
        seed = np.random.randint(0, len(train))
        f = np.array([1] + train[seed][:4])
        y = train[seed][4]
        if sign(np.dot(f, w)) != y:
            wt = w + y*f
            if use_pocket:
                wt_err = test_error_rate(wt, test)
                w_err = test_error_rate(w, test)
                if wt_err < w_err:
                    w = wt
            else:
                w = wt

    return w


def test_error_rate(w, test):
    error_num = 0
    for v in test:
        f = np.array([1] + v[:4])
        y = v[4]
        if sign(np.dot(f, w)) != y:
            error_num += 1
    return error_num


def pocket_algo(use_pocket=True, update_num=50):
    train_sample = load_18()
    test_sample = load_18_test()

    w = train_pla_pure_randomly(train_sample, test_sample, use_pocket, update_num)

    err_num = test_error_rate(w, test_sample)
    print err_num
    return float(err_num) / len(test_sample)

#  question 18: the err rate is 0.29
err_rate = 0
for i in range(2000):
    np.random.seed(i)
    err_rate += pocket_algo()
print err_rate / 2000

#  question 19: the err rate is 0.169
# err_rate = 0
# for i in range(2000):
#     np.random.seed(i)
#     err_rate += pocket_algo(use_pocket=False)
#
# print err_rate / 2000

#  question 20: the err rate is 0.29
# err_rate = 0
# for i in range(2000):
#     np.random.seed(i)
#     err_rate += pocket_algo(use_pocket=True, update_num=100)
#
# print err_rate / 2000
