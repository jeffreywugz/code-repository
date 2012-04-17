#!/usr/bin/env python2

import random
import numpy as np
import matplotlib.pyplot as plt

def make_hash_stat(n_buckets, n_items):
    bucket_list = [0] * n_buckets
    for i in range(n_items):
        bucket_list[random.randint(0, n_buckets-1)] += 1
    return bucket_list

def plot_hist(n_per_step=10000, n_step=4):
    for i in range(1, n_step+1):
        hist, _ = np.histogram(make_hash_stat(n_per_step*n_step, i*n_per_step), bins=range(20))
        hist = map(lambda (x,y): 1.0 * x*y/(n_per_step*i), enumerate(hist))
        plt.plot(hist, label='%d/%d'%(i, n_step))
    plt.xlabel('bucket list len')
    plt.ylabel('probability')
    plt.legend()
    return plt

if __name__ == '__main__':
    plot_hist().savefig('search-len.png')
