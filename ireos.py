# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 16:28:21 2021

@author: ExoFlare
"""

import logging
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics.pairwise import rbf_kernel

from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import concurrent.futures

DATA = None
# Cache of R matrices for different gammas
R_CACHE = None
CACHE_SIZE = 256
MAX_RECURSION_DEPTH = -1

class IREOS():
     
     def __init__(self, data, clf, gamma_max, tol, chunk_size=64, num_workers=multiprocessing.cpu_count(), max_recursion_depth = 3):
          
          global DATA
          DATA = data
          global R_CACHE
          R_CACHE = {}
          global MAX_RECURSION_DEPTH
          MAX_RECURSION_DEPTH = max_recursion_depth
          self.clf = clf
          if self.clf == 'svc':
               self.gamma_min = 0.0001
          elif self.clf == 'klr' or self.clf == 'logreg':
               self.gamma_min = 0
          else:
               raise ValueError('Invalid classifier!')
          self.gamma_max = gamma_max
          self.tol = tol
          self.num_workers = num_workers
          self.chunk_size=chunk_size
          self.num_samples = len(data)
          self.results = {}
          
     def run(self):
          with ThreadPoolExecutor(max_workers=self.num_workers) as exe:
               for chunk in range(int(self.num_samples / self.chunk_size)+1):
                    futures = []
                    logging.info('Starting chunk {} / {}'.format(chunk+1, int(self.num_samples / self.chunk_size)+1))
                    for x in range(chunk * self.chunk_size, (chunk + 1) * self.chunk_size 
                                   if (chunk + 1) * self.chunk_size <= self.num_samples 
                                   else self.num_samples):
                                   logging.debug('Start chunk {}, row {}'.format(chunk, x))
                                   ireos_example = IREOSExample(self.clf, x)
                                   futures.append(exe.submit(ireos_example.run, gamma_min=self.gamma_min,
                                              gamma_max=self.gamma_max, tol=self.tol))
                    for future in concurrent.futures.as_completed(futures):
                         self.results[future.result().ireos_data.get_index()] = future.result().get_auc()
                         logging.debug('Auc retreived for row {}: {}'.format(future.result().ireos_data.get_index()
                                                                            , future.result().get_auc()))
                    del futures
          global R_CACHE
          #del R_CACHE

                    
     def get_auc_of_sample(self, sample_index):
          return self.results[sample_index]
     def get_gamma_range(self):
          return self.gamma_max - self.gamma_min
               
     
     
class IREOSData:
     __inlier_class = -1
     __outlier_class = 1
     def __init__(self, index):
          self.index = index
          self.separabilities = {}
          self.num_samples = len(DATA)
          self.y = np.full(self.num_samples, self.__inlier_class)
          self.y[index] = self.__outlier_class
          self.current_sample = DATA[index].reshape(1, -1)
          
     def get_index(self):
          return self.index
     def get_seperability(self, key):
          return self.separabilities[key]
     def get_separabilities(self):
          return self.separabilities
     def add_separability(self, key, value):
          self.separabilities[key] = value
     def get_num_samples(self):
          return self.num_samples()
     def get_current_sample(self):
          if self.current_sample is None:
               raise ValueError("No current sample!")
          return self.current_sample
     def set_current_sample(self, current_sample):
          self.current_sample=current_sample
     def get_y(self):
          return self.y
     def get_outlier_class(self):
          return self.__outlier_class
          


class IREOSExample:
     
     def __init__(self, clf, index):
          self.ireos_data = IREOSData(index)
          self.clf=clf
          self.current_recursion_depth = 0
          
     def run(self, gamma_min, gamma_max, tol):
          #logging.info('Start auc calculation for index {}'.format(self.ireos_data.get_index()))
          self.auc = self.adaptive_quads(gamma_min, gamma_max, tol)
          logging.debug('Stop auc calculation for index {}'.format(self.ireos_data.get_index()))
          return self
     
     def adaptive_quads(self, a, b, tol):
          m = (a + b) / 2
          err_all = self.simpson_rule(a, b)
          err_new = self.simpson_rule(a, m) + self.simpson_rule(m, b)
          calculated_error = abs(err_all - err_new) / 15
          if tol < calculated_error and self.current_recursion_depth < MAX_RECURSION_DEPTH:
               logging.debug('Iteration depth: {}. Criterion not reached: {} > {}'
                             .format(self.current_recursion_depth, calculated_error, tol))
               self.current_recursion_depth += 1
               return self.adaptive_quads(a, m, tol / 2) + self.adaptive_quads(m, b, tol/2)
          else:
               logging.debug('Termination criterion of {} < {} reached.'.format(calculated_error, tol))
               return err_new
     
     def simpson_rule(self, a, b):
          h = (b - a) / 2
          for i in np.array([a, a+h, b]):
               if i in self.ireos_data.get_separabilities():
                    continue
               if self.clf == 'klr':
                    clf = self.get_klr_clf(i)
                    # calculate alpha weights and kernel value differently -> look at KLR#predict
                    # intercept = b, coefficients = alphas, R-Matrix = Kernel values
                    #res = clf.intercept_ + sum(sum(clf.coef_ * self.ireos_data.get_current_sample()))
                    #logging.debug('{}: weighted sum: {}'.format(self.ireos_data.index, res))
                    #p_outlier = 1 / (1 +math.exp(-res))
                    p_outlier = clf.predict_proba(self.ireos_data.get_current_sample())[0,1]
               else:
                    if self.clf == 'svc':
                         clf = self.get_svm_clf(i)
                    elif self.clf == 'logreg':
                         clf = self.get_logreg_clf(i)
                    
                    p_index = list(clf.classes_).index(self.ireos_data.get_outlier_class())
                    p_outlier = clf.predict_proba(self.ireos_data.get_current_sample())[0, p_index]
                    
               logging.debug('{}: gamma {} : p-value: {}'.format(self.ireos_data.index, float(i), p_outlier))
               self.ireos_data.add_separability(float(i), p_outlier)
          return (h / 3) * (self.ireos_data.get_seperability(a) 
                            + 4 * self.ireos_data.get_seperability(a + h) 
                            + self.ireos_data.get_seperability(b))
     
     def get_logreg_clf(self, gamma):
          clf = LogisticRegression(random_state=123, tol=0.0095, max_iter=1000000)
          global R_CACHE
          if(gamma not in R_CACHE):
               logging.debug('Cache Miss for gamma {}'.format(gamma))
               R = rbf_kernel(DATA, gamma=gamma)
               if(len(R_CACHE) < CACHE_SIZE):
                    R_CACHE[gamma] = R
          else:
               logging.debug('Cache Hit for gamma {}'.format(gamma))
               R = R_CACHE[gamma]
               
          self.ireos_data.set_current_sample(R[self.ireos_data.get_index()].reshape(1, -1))
          clf.fit(R, self.ireos_data.get_y())
          return clf
     def get_svm_clf(self, gamma):
          clf = SVC(gamma=gamma, probability=True, C=100, random_state=123, tol=0.0095, max_iter=1000000)
          clf.fit(DATA, self.ireos_data.get_y())
          return clf
     def get_klr_clf(self, gamma):
          # param set closest to the paper (liblinear returns non-zero values for gamma = 0)
          clf = LogisticRegression(class_weight='balanced', tol=0.0095, solver = 'saga', C=100, max_iter=1000000, random_state=123)
          global R_CACHE
          if(gamma not in R_CACHE):
               logging.debug('Cache Miss for gamma {}'.format(gamma))
               R = rbf_kernel(DATA, gamma=gamma)
               if(len(R_CACHE) < CACHE_SIZE):
                    R_CACHE[gamma] = R
          else:
               logging.debug('Cache Hit for gamma {}'.format(gamma))
               R = R_CACHE[gamma]
               
          self.ireos_data.set_current_sample(R[self.ireos_data.get_index()].reshape(1, -1))
          clf.fit(R, self.ireos_data.get_y())
          return clf
          
     def get_auc(self):
          return self.auc
     
class IREOSEvaluation:
     
     def __init__(self, ireos, solutions, num_workers=multiprocessing.cpu_count()):
          self.ireos = ireos
          self.solutions = solutions
          if solutions.ndim == 1:
               self.num_solutions = 1
          else:
               self.num_solutions = len(solutions)
          self.num_workers = num_workers
          self.results = {}
     
     def run(self):
          with ThreadPoolExecutor(max_workers=self.num_workers) as exe:
               futures = []
               for x in range(0, self.num_solutions):
                    futures.append(exe.submit(self.evaluate_solution, index=x))
               for future in concurrent.futures.as_completed(futures):
                    logging.info('Solution evaluated.')
                    
     def evaluate_solution(self, index):
          if self.num_solutions > 1:
               solution = self.solutions[index]
          else:
               solution = self.solutions
          sum_weights = 0
          res = 0
          for i in range(self.solutions.shape[0]):
               sum_weights = sum_weights + solution[i]
               res = res + solution[i] * self.ireos.get_auc_of_sample(i)
          self.results[index] = res / sum_weights / self.ireos.get_gamma_range()
          