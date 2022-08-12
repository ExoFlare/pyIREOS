#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 22:47:38 2022

@author: exoflare
"""
import time
from numpy import random, ascontiguousarray
import minicore as mc
from sklearn.datasets import make_blobs
import numpy as np

dat = np.genfromtxt('data/dens-diff_12', delimiter=' ')

for i in range(5):
    dat = np.random.rand(50000, 1000)
    
    start = time.time()
    
    #https://github.com/dnbaker/minicore/blob/8a76640229f25059664a5d56edae1e131d93daaf/python/README.md
    res = mc.kmeanspp(dat, k=10)
    out, asn, costs  = res
    
    cs = mc.CoresetSampler()
    
    sensid = mc.constants.SENSDICT["LBK"] # This uses coresets for Bregman divergences
    
    cs.make_sampler(10, costs=costs, assignments=asn)
    
    weights, ids = cs.sample(5000)
    end = time.time()
    print("Time consumed in working: ",end - start)


top = sorted(zip(ids, weights), key=lambda x: -x[1])[:10]
print("top 10: ", top)

bottom = sorted(zip(ids, weights), key=lambda x: x[1])[:10]
print("bottom 10: ", bottom)