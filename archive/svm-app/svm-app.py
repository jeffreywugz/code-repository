#!/usr/bin/env python

import sys
sys.path.append("libsvm-2.89/python")
import exceptions
import time
from svm import *

inf = 1e10

class ContinuousNormalizer:
    def __init__(self):
        self.min = inf
        self.max = -inf

    def stat_item_add(self, item):
        item = float(item)
        self.min = min(self.min, item)
        self.max = max(self.max, item)

    def stat_calculate(self):
        self.range = self.max - self.min
        if self.range == 0.0:
            self.range = 1.0
    
    def item_normalize(self, item):
        item = float(item)
        return (item-self.min)/self.range

class CategoricalNormalizer:
    def __init__(self):
        self.categories = set()

    def stat_item_add(self, item):
        self.categories.add(item)

    def stat_calculate(self):
        li = list(self.categories)
        self.n_items = len(li)
        self.cate = dict(map(None, li, range(self.n_items)))
        
    def item_normalize(self, item):
        return self.cate[item]

class ListNormalizer:
    def __init__(self, normalizers):
        self.normalizers = normalizers

    def stat_item_add(self, item):
        map(lambda obj, args: obj.stat_item_add(args), self.normalizers, item)

    def stat_calculate(self):
        map(lambda obj: obj.stat_calculate(), self.normalizers)

    def item_normalize(self, item):
        return map(lambda obj, args: obj.item_normalize(args), self.normalizers, item)

normalizers = (
    ContinuousNormalizer(),
    CategoricalNormalizer(),
    ContinuousNormalizer(),
    CategoricalNormalizer(),
    ContinuousNormalizer(),
    CategoricalNormalizer(),
    CategoricalNormalizer(),
    CategoricalNormalizer(),
    CategoricalNormalizer(),
    CategoricalNormalizer(),
    ContinuousNormalizer(),
    ContinuousNormalizer(),
    ContinuousNormalizer(),
    CategoricalNormalizer(),
    CategoricalNormalizer(),
)

samples = [line.split(',') for line in open("data/adult.data").readlines()]
list_normalizer = ListNormalizer(normalizers)
map(list_normalizer.stat_item_add, samples)
list_normalizer.stat_calculate()
samples = map(list_normalizer.item_normalize, samples)
kname = ['linear','polynomial','rbf']

def test_svm(kernel, train_samples, test_samples):
    start = time.time()
    
    labels = [item[-1] for item in train_samples ]
    samples = [item[:-1] for item in train_samples]
    problem = svm_problem(labels, samples)
    param = svm_parameter(C = 5, kernel_type = kernel)
    model = svm_model(problem,param)
    
    errors = 0
    for item in test_samples:
        prediction = model.predict(item[:-1])
        if (item[-1] != prediction):
            errors = errors + 1
            
    end = time.time()
    print "##kernel: %s; number: %d; error rate: %f; time: %f; " % (kname[param.kernel_type],
                                                    len(test_samples), 1.0*errors/len(test_samples), end-start)
    
    
for kernel in (LINEAR, POLY, RBF):
    for n in (100,200,400,800,1600):
        train_samples, test_samples = samples[:n], samples[n:2*n]
        test_svm(kernel, train_samples, test_samples)
