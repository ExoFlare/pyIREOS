# -*- coding: utf-8 -*-
"""
Created on Fri May 27 15:43:05 2022

@author: ExoFlare
"""
"""
==============================================
Comparison of anomaly detection algorithms 
with multi-dimensional synthetic data 
with GT
 
FIV, May 2021
==============================================
"""

#!/usr/bin/env python3

print(__doc__)

import os

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

from pyod.models.abod import ABOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.loci import LOCI
from pyod.models.ocsvm import OCSVM
from pyod.models.cof import COF
from pyod.models.sod import SOD
from pysdo import SDO
from indices import get_indices
import logging

import sys

data_path = os.getcwd() + '/data/'

np.random.seed(100)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def abod(c):
    model = ABOD(contamination=c)
    return model
 
def hbos(c):
    model = HBOS(contamination=c)
    return model

def iforest(c):
    model = IForest(contamination=c, random_state=100)
    return model

def knn(c):
    model = KNN(contamination=c)
    return model

def lof(c):
    model = LOF(contamination=c, n_neighbors=35)
    return model

def ocsvm(c):
    model = OCSVM(contamination=c)
    return model

def sdo(c):
    model = SDO(contamination=c, return_scores=True)
    return model

def select_algorithm(argument,k):
    switcher = {"abod":abod, "hbos":hbos, "iforest":iforest, "knn":knn, "lof":lof, "ocsvm":ocsvm, "sdo":sdo}
    model = switcher.get(argument, lambda: "Invalid algorithm")
    return model(k)

data_names =['complex_1', 'complex_2', 'complex_3', 'complex_4', 'complex_5', 'complex_6', 'complex_7', 'complex_8', 'complex_9', 'complex_10',
	'complex_11', 'complex_12', 'complex_13', 'complex_14', 'complex_15', 'complex_16',
    'complex_17', 'complex_18', 'complex_19', 'complex_20', 'high-noise_1', 'high-noise_2', 'high-noise_3',
	'high-noise_4', 'high-noise_5', 'high-noise_6', 'high-noise_7', 'high-noise_8', 'high-noise_9', 
	'high-noise_10', 'high-noise_11', 'high-noise_12', 'high-noise_13', 
    'high-noise_14', 'high-noise_15', 'high-noise_16', 'high-noise_17', 'high-noise_18', 'high-noise_19', 'high-noise_20', 
    'dens-diff_1', 'dens-diff_2', 'dens-diff_3', 'dens-diff_4', 'dens-diff_5', 'dens-diff_6', 'dens-diff_7', 'dens-diff_8',
	'dens-diff_9', 'dens-diff_10', 'dens-diff_11', 'dens-diff_12', 'dens-diff_13', 'dens-diff_14', 'dens-diff_15', 'dens-diff_16', 
    'dens-diff_17', 'dens-diff_18', 'dens-diff_19', 'dens-diff_20',
	'low-noise_1', 'low-noise_2', 'low-noise_3', 'low-noise_4', 'low-noise_5', 'low-noise_6', 'low-noise_7', 'low-noise_8', 'low-noise_9',
	'low-noise_10', 'low-noise_11', 'low-noise_12', 'low-noise_13', 'low-noise_14', 'low-noise_15', 'low-noise_16', 
    'low-noise_17', 'low-noise_18', 'low-noise_19', 'low-noise_20']

algs = ["sdo","abod", "hbos", "iforest", "knn", "lof", "ocsvm"]
df_columns = ["adj_Patn", "adj_maxf1", "adj_ap", "auc", "AMI"]

iterables = [data_names,algs]
df_index = pd.MultiIndex.from_product(iterables, names=['Data', 'Alg.'])
df_val = pd.DataFrame(columns=df_columns,index=df_index)

for d_ind, d_name in enumerate(data_names):

    file_name = data_path + d_name
    dataset = np.genfromtxt(file_name, delimiter=',')
    print("\n**************************************** DATASET: ", d_name, "**********")

    X, ygt = dataset[:,0:-1], dataset[:,-1].astype(int)
    # Transform multi-class labels with -1 for outliers into: 0-inlier, 1-outlier
    ygt[ygt>-1] = 0
    ygt[ygt==-1] = 1

    n_samples = len(ygt)
    outliers_fraction = sum(ygt)/len(ygt)
    

    # normalize dataset
    X = StandardScaler().fit_transform(X)

    ### OUTLIER DET. ALGORITHMS 

    scorings = pd.DataFrame(columns=algs)
    for a_name in algs:

        print("-----------------------------")
        print("Algorithm:", a_name)

        algorithm = select_algorithm(a_name,outliers_fraction)
        algorithm.fit(X)
        if a_name == "sdo":
            scores = algorithm.predict(X)
            threshold = np.quantile(scores, 1-outliers_fraction)
            y = scores > threshold
        else:
            y = algorithm.predict(X)
            scores = algorithm.decision_scores_

        scorings[a_name] = scores
        AMI = adjusted_mutual_info_score(ygt, y)
        RES = get_indices(ygt, scores)

        df_val.loc[(d_name,a_name), 'adj_Patn'] = RES['adj_Patn']
        df_val.loc[(d_name,a_name), 'adj_maxf1'] = RES['adj_maxf1']
        df_val.loc[(d_name,a_name), 'adj_ap'] = RES['adj_ap']
        df_val.loc[(d_name,a_name), 'auc'] = RES['auc']
        df_val.loc[(d_name,a_name), 'AMI'] = AMI
        
        
        print("Adj P@n: ", RES['adj_Patn'])
        print("Adj MaxF1: ", RES['adj_maxf1'])
        print("Adj AP: ", RES['adj_ap'])
        print("Adj ROC-AUC: ", RES['auc'])
        print("Adj AMI: ", AMI)
        print("-----------------------------\n")
        scorings.to_csv(os.getcwd() + '/scores/' + d_name +'.csv', index=False)

#df_val.to_csv('scorings__all.csv')



