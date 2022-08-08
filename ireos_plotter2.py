# -*- coding: utf-8 -*-
"""
Created on Mon May 23 23:14:23 2022

@author: ExoFlare
"""

import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(os.getcwd() + '/wbc_results.csv')


sns.boxplot(x="clf", y="mae", hue="adaptive_quads_enabled", data=df)

# distinct by algorithm and adaptive quads enabled
klr = df[(df['clf'] == 'klr') & (df['adaptive_quads_enabled'] == True)]
svc = df[(df['clf'] == 'svc') & (df['adaptive_quads_enabled'] == True)]
logreg = df[(df['clf'] == 'logreg') & (df['adaptive_quads_enabled'] == True)]
libsvm = df[(df['clf'] == 'libsvm') & (df['adaptive_quads_enabled'] == True)]
liblinear = df[(df['clf'] == 'liblinear') & (df['adaptive_quads_enabled'] == False)]
dt = df[(df['clf'] == 'decision_tree') & (df['adaptive_quads_enabled'] == False)]

#concat all datasets with adaptive quads enabled -> compare window ratios
df_2 = pd.concat([klr, svc, logreg, libsvm, liblinear, dt])
sns.boxplot(x="clf", y="mae", hue="window_ratio", data=df_2).set(title='MAE of different models having multiple window sizes to IREOS-Java implementation')

df = pd.read_csv(os.getcwd() + '/test2.csv')

# filter SVM results
svc = df[(df['clf'] == 'svc')]
svc_par = df.filter(items = [14, 52, 90], axis=0)
svc_seq = df.filter(items = [36, 74, 112], axis=0)
libsvm = df[(df['clf'] == 'libsvm')]
libsvm_par = df.filter(items = [37, 75, 113], axis=0)
libsvm_seq = df.filter(items = [32, 70, 108], axis=0)

#plot car plot of different runtimes of parallel and sequential SVM implementations
plt.bar(['Sequential', 'Parallel SVC', 'Parallel LIBSVM'], 
         [1.0, svc_seq['time'].mean() / svc_par['time'].mean(),libsvm_seq['time'].mean() / libsvm_par['time'].mean()])
plt.xlabel("Classifier")
plt.ylabel("Speedup")
plt.show()

klr = df[(df['clf'] == 'klr') & (df['adaptive_quads_enabled'] == True)]
svc = df[(df['clf'] == 'svc') & (df['adaptive_quads_enabled'] == True)]
logreg = df[(df['clf'] == 'logreg') & (df['adaptive_quads_enabled'] == True)]
libsvm = df[(df['clf'] == 'libsvm') & (df['adaptive_quads_enabled'] == True)]
liblinear = df[(df['clf'] == 'liblinear') & (df['adaptive_quads_enabled'] == False)]
dt = df[(df['clf'] == 'decision_tree') & (df['adaptive_quads_enabled'] == False)]

df_2 = pd.concat([svc, libsvm, liblinear, dt])
sns.boxplot(x="clf", y="time", hue="window_ratio", data=df_2).set(title='Runtimes of different models having multiple window sizes for n = 5000')

svc = df[(df['clf'] == 'svc') & (df['adaptive_quads_enabled'] == True)]
libsvm = df[(df['clf'] == 'libsvm') & (df['adaptive_quads_enabled'] == True)]
liblinear = df[(df['clf'] == 'liblinear') & (df['adaptive_quads_enabled'] == False)]
dt = df[(df['clf'] == 'decision_tree') & (df['adaptive_quads_enabled'] == False)]

svc_mean = svc['time'].mean()
y = [1.0, svc_mean / libsvm['time'].mean(), svc_mean / liblinear['time'].mean(), svc_mean / dt['time'].mean()]

plt.figure()
bars = plt.bar(['SVC', 'LIBSVM', 'LIBLINEAR', 'DECISION TREE'], y)
plt.title("Relative Speedup in comparison to SVC over different parameter settings")
plt.xlabel("Classifier")
plt.ylabel("Speedup")

xlocs, xlabs = plt.xticks()
for i, v in enumerate(y):
    plt.text(xlocs[i] - 0.25, v + 0.01, str(v))
plt.show()

