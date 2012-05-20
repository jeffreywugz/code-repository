#!/usr/bin/env python2
from subprocess import Popen, PIPE, STDOUT
import re
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.scale as scale
import matplotlib.pyplot as plt

def popen(cmd):
    return Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT).communicate()[0]

def get_maccess_time(size, stride, n_access):
    m = re.search('access_time=.*=([.0-9]+)ns', popen('./mem-param.exe %d %d %d'%(size, stride, n_access)))
    if not m: raise Exception('no output match for mem-param.exe!')
    return float(m.group(1))

def gen_prefetch_ts(size, max_stride, n_access=10240000):
    return [get_maccess_time(size, 1<<i, n_access) for i in range(int(math.log(min(size, max_stride), 2)))]

def gen_workingset_ts(size, stride=512, n_access=10240000):
    return [get_maccess_time(1<<i, stride, n_access) for i in range(int(math.log(stride, 2)), int(math.log(size, 2)))]

def plot_prefetch():
    '''same working set, different stride'''
    ts0 = gen_prefetch_ts(512, 10240)
    ts1 = gen_prefetch_ts(1024, 10240)
    ts2 = gen_prefetch_ts(10240, 10240)
    ts3 = gen_prefetch_ts(10240000, 10240)
    plt.plot(range(len(ts0)), ts0, 'ro-', label='block=512B')
    plt.plot(range(len(ts1)), ts1, 'g^-', label='block=1K')
    plt.plot(range(len(ts2)), ts2, 'bh-', label='block=10K')
    plt.plot(range(len(ts3)), ts3, 'rs-', label='block=10M')
    #plt.yscale('log')
    plt.xlabel('stride(2**x)')
    plt.ylabel('access time(ns)')
    plt.legend(loc='upper left')
    return plt

def plot_workingset():
    ts0 = gen_workingset_ts(1<<28, stride=8)
    ts1 = gen_workingset_ts(1<<28, stride=16)
    ts2 = gen_workingset_ts(1<<28, stride=32)
    ts3 = gen_workingset_ts(1<<28, stride=512)
    plt.plot([8 * (1<<i) for i in range(len(ts0))], ts0, 'ro-', label='stride=8')
    plt.plot([16 * (1<<i) for i in range(len(ts1))], ts1, 'g^-', label='stride=16')
    plt.plot([32 * (1<<i) for i in range(len(ts2))], ts2, 'bh-', label='stride=32')
    plt.plot([512 * (1<<i) for i in range(len(ts3))], ts3, 'rs-', label='stride=512')
    plt.xscale('log', basex=2)
    plt.xlabel('size(2**x)')
    plt.ylabel('access time(ns)')
    plt.legend(loc='upper left')
    return plt

#plot_prefetch().savefig('prefetch.png')
plot_workingset().savefig('workingset.png')
