# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 15:47:51 2022

@author: ExoFlare
"""
import os
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt

base_path = os.getcwd()

julia_extension = '/results/ireos/'
java_extension = '/results/ireos_java/'

# set dataset
dataset_name = 'complex_1'

x = Path(base_path + julia_extension).glob(dataset_name+'-*')

julia_ireos_df = pd.concat([pd.read_csv(csv_name, sep=',', header=None, names=[csv_name]) for csv_name in Path(base_path + julia_extension).glob(dataset_name+'-*')], axis =1)
java_ireos_df = pd.read_csv(base_path + java_extension + dataset_name + '.csv', sep=',', header=None, names = ['klr'])


ireos_df = pd.concat([java_ireos_df, julia_ireos_df], axis=1)
num_samples = ireos_df.shape[0]

labels = [str(x) for x in java_ireos_df.columns] + [str(x).replace(base_path + julia_extension, '') for x in julia_ireos_df.columns]

plt.boxplot(ireos_df)
plt.show()

indices = [0, 3, 9, 14]
num_plots = 4

fig, ax = plt.subplots(nrows=1, ncols=num_plots, figsize=(18, 5))
num_bins=10

for i in range(num_plots):
     ax[i].hist(ireos_df.iloc[:, indices[i]], label=labels[indices[i]], bins=num_bins)
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=num_plots, figsize=(18, 5))

for i in range(num_plots):
     ax[i].bar(list(range(num_samples)), ireos_df.iloc[:, indices[i]], label=labels[indices[i]])
plt.show()

