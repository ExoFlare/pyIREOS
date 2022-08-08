# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 11:43:25 2022

@author: ExoFlare
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

datasets = ['complex_1', 'complex_2', 'complex_3', 'complex_4', 'complex_5', 'complex_6', 'complex_7', 'complex_8', 'complex_9', 'complex_10',
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

base_dir = os.getcwd()
results_dir = '/results/'
ireos_java_dir = 'ireos_java/'
ireos_dir = 'ireos/'

# define algorithm of java implementation -> column name later
java_predictors = ['Kernel Logistic Regression']

# calculate mean and standard deviation for kernel logistic regression
result_java = []
result_java_std = []
for data_set in datasets:
     mean_ = pd.read_csv(base_dir + results_dir + ireos_java_dir + data_set + '.csv').to_numpy().mean()
     std_ = pd.read_csv(base_dir + results_dir + ireos_java_dir + data_set + '.csv').to_numpy().std()
     result_java = np.append(result_java, mean_)
     result_java_std = np.append(result_java_std, std_)
     
     
fireos_predictors = ['Decision Tree', 'Random Forest', 'Support Vector Machine', 'LibLinear']
fireos_suffixes = ['-decision_tree_native-nothing-false-sequential', '-random_forest_native-nothing-false-sequential'
            ,'-libsvm-parallel', '-liblinear-nothing-false-sequential']

num_predictors = len(fireos_suffixes)
num_datasets = len(datasets)

result_fireos = np.zeros((num_datasets, num_predictors))
result_fireos_std = np.zeros((num_datasets, num_predictors))

# calculate mean and standard deviation for fireos predictors
for i in range(0, num_predictors):
     for j in range(0, num_datasets):
          mean_ = pd.read_csv(base_dir + results_dir + ireos_dir + datasets[j] + fireos_suffixes[i] + '.csv', header=0).to_numpy().mean()
          std_ = pd.read_csv(base_dir + results_dir + ireos_dir + datasets[j] + fireos_suffixes[i] + '.csv', header=0).to_numpy().std()
          result_fireos[j][i] = mean_
          result_fireos_std[j][i] = std_
          
# concat java and fireos solutions
predictors = np.append(java_predictors, fireos_predictors)
results = np.column_stack((result_java, result_fireos))
stds = np.column_stack((result_java_std, result_fireos_std))
stds_df = pd.DataFrame(stds, columns=predictors, index=datasets)

result_df = pd.DataFrame(results, columns=predictors, index=datasets)
result_df.index.name='Dataset'

sns.set_style('darkgrid')
for predictor in predictors:
     ax = sns.lineplot(x="Dataset", y=predictor, data=result_df, marker='o')
     ax.tick_params(axis='x', rotation=90)
     
ax.legend(predictors)
plt.title('Separabilities of different predictors over different datasets')
plt.ylabel('Separability')
plt.show()