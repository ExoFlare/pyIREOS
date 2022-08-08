# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 16:14:38 2022

@author: ExoFlare
"""
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


base_dir = os.getcwd()
data_dir = '/data/'
results_dir = '/results/'
ireos_java_dir = 'ireos_java/'
ireos_dir = 'ireos/'

# set dataset name
data_set_name = 'complex_1'

df = pd.read_csv(base_dir + data_dir + data_set_name)
num_samples = df.shape[0]
one_percent = int(num_samples / 100)

#transform data into 2D space
pca2D = PCA(n_components=2, svd_solver='auto')
principalComponents2D = pca2D.fit_transform(df)
principalDf2D = pd.DataFrame(data = principalComponents2D, columns = ['PC1', 'PC2'])

#transform data into 3D space
pca3D = PCA(n_components=3, svd_solver='auto')
principalComponents3D = pca3D.fit_transform(df)
principalDf3D = pd.DataFrame(data = principalComponents3D, columns = ['PC1', 'PC2', 'PC3'])


ireos_java = pd.read_csv(base_dir + results_dir + ireos_java_dir + data_set_name + '.csv')

#plot kernel logistic regression results
outlier_amount = ireos_java.to_numpy().sum() / num_samples
avg_top_outliers = np.mean(ireos_java.iloc[:,0].nlargest(n=one_percent))
title = data_set_name.upper() + ': IREOS JAVA\n Outlier Amount: ' + str(outlier_amount) + ' ,\nAverage Outlierfactor Top 1%: ' + str(avg_top_outliers)
sns.set(style='whitegrid')
sns.scatterplot(x='PC1', y='PC2', data=principalDf2D, hue=ireos_java.to_numpy().flatten()).set(title=title)
plt.show()
fig = px.scatter_3d(principalDf3D, x='PC1', y='PC2', z='PC3', color=ireos_java.to_numpy().flatten(), title=title)
fig.show(renderer="browser")

#plot decision tree results
dt = '-decision_tree_native-nothing-false-sequential'
ireos_dt = pd.read_csv(base_dir + results_dir + ireos_dir + data_set_name + dt + '.csv', header=0)
outlier_amount = ireos_dt.to_numpy().sum() / num_samples
avg_top_outliers = np.mean(ireos_dt.iloc[:,0].nlargest(n=one_percent))
title = data_set_name.upper() + ': IREOS DECISION TREE\n Outlier Amount: ' + str(outlier_amount) + ' ,\nAverage Outlierfactor Top 1%: ' + str(avg_top_outliers)
plt.figure()
sns.set(style='whitegrid')
sns.scatterplot(x='PC1', y='PC2', data=principalDf2D, hue=ireos_dt.to_numpy().flatten()).set(title=title)
plt.show()
fig = px.scatter_3d(principalDf3D, x='PC1', y='PC2', z='PC3', color=ireos_dt.to_numpy().flatten(), title=title)
fig.show(renderer="browser")

#plot random forests results
rf = '-random_forest_native-nothing-false-sequential'
ireos_rf = pd.read_csv(base_dir + results_dir + ireos_dir + data_set_name + rf + '.csv', header=0)
outlier_amount = ireos_rf.to_numpy().sum() / num_samples
avg_top_outliers = np.mean(ireos_rf.iloc[:,0].nlargest(n=one_percent))
title = data_set_name.upper() + ': IREOS RANDOM FOREST\n Outlier Amount: ' + str(outlier_amount) + ' ,\nAverage Outlierfactor Top 1%: ' + str(avg_top_outliers)
plt.figure()
sns.set(style='whitegrid')
sns.scatterplot(x='PC1', y='PC2', data=principalDf2D, hue=ireos_rf.to_numpy().flatten()).set(title=title)
plt.show()
fig = px.scatter_3d(principalDf3D, x='PC1', y='PC2', z='PC3', color=ireos_rf.to_numpy().flatten(), title=title)
fig.show(renderer="browser")

#plot support vector machine results
svm = '-libsvm-parallel'
ireos_svm = pd.read_csv(base_dir + results_dir + ireos_dir + data_set_name + svm + '.csv', header=0)
outlier_amount = ireos_svm.to_numpy().sum() / num_samples
avg_top_outliers = np.mean(ireos_svm.iloc[:,0].nlargest(n=one_percent))
title = data_set_name.upper() + ': IREOS SUPPORT VECTOR MACHINE\n Outlier Amount: ' + str(outlier_amount) + ' ,\nAverage Outlierfactor Top 1%: ' + str(avg_top_outliers)
plt.figure()
sns.set(style='whitegrid')
sns.scatterplot(x='PC1', y='PC2', data=principalDf2D, hue=ireos_svm.to_numpy().flatten()).set(title=title)
plt.show()
fig = px.scatter_3d(principalDf3D, x='PC1', y='PC2', z='PC3', color=ireos_svm.to_numpy().flatten(), title=title)
fig.show(renderer="browser")

#plot liblinear results
liblinear = '-liblinear-nothing-false-sequential'
ireos_liblinear = pd.read_csv(base_dir + results_dir + ireos_dir + data_set_name + liblinear + '.csv', header=0)
outlier_amount = ireos_liblinear.to_numpy().sum() / num_samples
avg_top_outliers = np.mean(ireos_liblinear.iloc[:,0].nlargest(n=one_percent))
title = data_set_name.upper() + ': IREOS LIBLINEAR\n Outlier Amount: ' + str(outlier_amount) + ' ,\nAverage Outlierfactor Top 1%: ' + str(avg_top_outliers)
plt.figure()
sns.set(style='whitegrid')
sns.scatterplot(x='PC1', y='PC2', data=principalDf2D, hue=ireos_liblinear.to_numpy().flatten()).set(title=title)
plt.show()
fig = px.scatter_3d(principalDf3D, x='PC1', y='PC2', z='PC3', color=ireos_liblinear.to_numpy().flatten(), title=title)
fig.show(renderer="browser")



