# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 22:16:21 2021

@author: ExoFlare
"""
import matplotlib.pyplot as plt  
import pandas as pd

data = pd.read_csv('mydata_results.csv')

data.drop('Unnamed: 0', axis=1, inplace=True)

data = data.replace('klr.*', 'klr', regex=True)
data = data.replace('logreg.*', 'logreg', regex=True)
data = data.replace('svc.*', 'svc', regex=True)

mytest = data[data['dataset'] == 'data/mytest.txt']
wbc = data[data['dataset'] == 'data/WBC_withoutdupl_norm']
densdiff = data[data['dataset'] == 'data/dens-diff_12']

mytest.boxplot(by='classifier', figsize=(6,5))

title_boxplot = 'Runtimes for dataset (n=201, d=2) in seconds'
plt.title( title_boxplot )
plt.suptitle('') # that's what you're after
plt.show()

wbc.boxplot(by='classifier', figsize=(6,5))

title_boxplot = 'Runtimes for dataset (n=223, d=9) in seconds'
plt.title( title_boxplot )
plt.suptitle('') # that's what you're after
plt.show()

densdiff.boxplot(by='classifier', figsize=(6,5))

title_boxplot = 'Runtimes for dataset (n=5000, d=4) in seconds'
plt.title( title_boxplot )
plt.suptitle('') # that's what you're after
plt.show()

