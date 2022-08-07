#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 13:54:00 2022

@author: exoflare
"""

import pandas as pd
from sklearn.metrics import pairwise_distances
from numpy import random, ascontiguousarray
import minicore as mc
from sklearn.datasets import make_blobs
import numpy as np
import logging, time
import ireos as ir

logging.getLogger().setLevel(logging.INFO)

#read data and solutions
data = np.genfromtxt('data/WBC_withoutdupl_norm', delimiter=' ')
solutions = np.genfromtxt('data/solutions.csv', delimiter=',')

#data = np.random.rand(3000, 2)
#solutions = np.random.rand(10, 3000)

res = mc.kmeanspp(data, k=25, msr=2)
out, asn, costs = res

cs = mc.CoresetSampler()

# This uses coresets for Bregman divergences
sensid = mc.constants.SENSDICT["LBK"]

cs.make_sampler(25, costs=costs, assignments=asn, sens=4)

weights, ids = cs.sample(100)



clfs = ['svc', 'logreg', 'klr']
tol = 0.005
gamma_max = pairwise_distances(data).max()

results = pd.DataFrame(columns = ['dataset', 'classifier', 'solution', 'solution_index', 'ireos', 'runtime'])

for clf in clfs:
    logging.info('Starting classifier {}'.format(clf))
    start_time = time.time()
    ireos = ir.IREOS(data, clf, gamma_max, tol, max_recursion_depth=3)
    ireos.run()
    eval = ir.IREOSEvaluation(ireos, solutions)
    eval.run()
    runtime =  time.time() - start_time
    for j in range(len(eval.results)):
        results.loc[len(results.index)] = ['WBC_withoutdupl_norm', clf, 'solutions.csv', j, eval.results[j], runtime]

results_coreset = pd.DataFrame(columns = ['dataset', 'classifier', 'solution', 'solution_index', 'ireos', 'runtime'])

for clf in clfs:
    logging.info('Starting classifier {}'.format(clf))
    start_time = time.time()
    
    res = mc.kmeanspp(data, k=25, msr=2)
    out, asn, costs = res
    cs = mc.CoresetSampler()
    # This uses coresets for Bregman divergences
    sensid = mc.constants.SENSDICT["LBK"]
    cs.make_sampler(25, costs=costs, assignments=asn, sens=4)
    weights, ids = cs.sample(500)
    
    data_coreset = data[ids, :]
    solutions_coreset = solutions[:,ids]
    
    ireos = ir.IREOS(data_coreset, clf, gamma_max, tol, max_recursion_depth=3)
    ireos.run()
    eval = ir.IREOSEvaluation(ireos, solutions_coreset)
    eval.run()
    runtime =  time.time() - start_time
    for j in range(len(eval.results)):
        results_coreset.loc[len(results_coreset.index)] = ['WBC_withoutdupl_norm', clf, 'solutions.csv', j, eval.results[j], runtime]