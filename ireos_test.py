# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 15:52:55 2021

@author: ExoFlare
"""

import logging
import numpy as np

import pandas as pd

from sklearn.metrics import pairwise_distances
import ireos as ir

import time

data_files = np.array(['data/WBC_withoutdupl_norm', 'data/mytest.txt', 'data/dens-diff_12'])
solution_files = np.array(['data/solutions.csv', 'data/mytest_solution.txt', np.zeros(5000)])

#data_files = np.array(['data/WBC_withoutdupl_norm', 'data/mytest.txt'])
#solution_files = np.array(['data/solutions.csv', 'data/mytest_solution.txt'])

logging.getLogger().setLevel(logging.INFO)

tol = 0.005

res = pd.DataFrame(columns = ['dataset', 'classifier', 'solution', 'solution_index', 'ireos', 'runtime'])

for i in range(len(data_files)):
     
     logging.info('Starting file {}'.format(data_files[i]))
     
     data = np.genfromtxt(data_files[i],delimiter=' ')
     solutions = np.genfromtxt(solution_files[i],delimiter=',')

     gamma_max = pairwise_distances(data).max()
     
     clfs = ['svc', 'logreg', 'klr']
     
     for clf in clfs:
          logging.info('Starting classifier {}'.format(clf))
          start_time = time.time()
          ireos = ir.IREOS(data, clf, gamma_max, tol, max_recursion_depth=3)
          ireos.run()
          runtime =  time.time() - start_time
          
          eval = ir.IREOSEvaluation(ireos, solutions)
          eval.run()
          for j in range(len(eval.results)):
               res.loc[len(res.index)] = [data_files[i], clf, solution_files[i], j, eval.results[j], runtime]
                                     
res.to_csv('mydata_results.csv')
