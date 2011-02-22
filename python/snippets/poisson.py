#!/usr/bin/python2

import numpy as np

lambd, size = 100, 100000
intervals = np.random.exponential(1.0/lambd, size)
times = np.add.accumulate(intervals)
bins = int(times[-1]-times[0])
print 'bins: ', bins
rates = np.histogram(times, bins)[0]
rates2 = np.random.poisson(lambd, bins)
print 'rates: ', rates, 'rates2: ', rates2
rates = np.histogram(rates, range(100))[0]
rates2 = np.histogram(rates2, range(100))[0]
print rates-rates2
