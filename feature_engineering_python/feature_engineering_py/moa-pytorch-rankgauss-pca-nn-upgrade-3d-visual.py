#!/usr/bin/env python
# coding: utf-8

# <a class="anchor" id="0"></a>
# # [Mechanisms of Action (MoA) Prediction](https://www.kaggle.com/c/lish-moa)

# ### I use the notebook [Pytorch CV|0.0145| LB| 0.01839 |](https://www.kaggle.com/riadalmadani/pytorch-cv-0-0145-lb-0-01839) from [riadalmadani](https://www.kaggle.com/riadalmadani) as a basis and will try to tune its various parameters. 

# # Acknowledgements
# 
# * [MoA | Pytorch | 0.01859 | RankGauss | PCA | NN](https://www.kaggle.com/kushal1506/moa-pytorch-0-01859-rankgauss-pca-nn)
# * [[MoA] Pytorch NN+PCA+RankGauss](https://www.kaggle.com/nayuts/moa-pytorch-nn-pca-rankgauss)
# * [Pytorch CV|0.0145| LB| 0.01839 |](https://www.kaggle.com/riadalmadani/pytorch-cv-0-0145-lb-0-01839)
# * [[New Baseline] Pytorch | MoA](https://www.kaggle.com/namanj27/new-baseline-pytorch-moa)
# * [Deciding (n_components) in PCA](https://www.kaggle.com/kushal1506/deciding-n-components-in-pca)
# * [Titanic - Featuretools (automatic FE&FS)](https://www.kaggle.com/vbmokin/titanic-featuretools-automatic-fe-fs)
# * tuning and visualization from [Higher LB score by tuning mloss - upgrade & visual](https://www.kaggle.com/vbmokin/higher-lb-score-by-tuning-mloss-upgrade-visual)
# * [Data Science for tabular data: Advanced Techniques](https://www.kaggle.com/vbmokin/data-science-for-tabular-data-advanced-techniques)

# ## My upgrade:
# 
# * PCA parameters
# * Feature Selection methods, including QuantileTransformer tuning
# * Dropout
# * Structuring of the notebook
# * Tuning visualization
# * Number of folds
# 
# I used the code from sources (please see above). 
# 
# I have completed the improvement of this notebook. Commit 9 is optimal.
# 
# I switch to improving my private notebook and other tasks.

# ## My Conclusions in posts:
# 
# * [FE for Pytorch-RankGauss-PCA-NN model with LB=0.01839](https://www.kaggle.com/c/lish-moa/discussion/194345)
# * [FE : VarianceThreshold - what else is there?](https://www.kaggle.com/c/lish-moa/discussion/194973)
# * [QuantileTransformer parameters tuning](https://www.kaggle.com/c/lish-moa/discussion/195788)

# <a class="anchor" id="0.1"></a>
# ## Table of Contents
# 
# 1. [Import libraries](#1)
# 1. [My upgrade](#2)
#     -  [Commit now](#2.1)
#     -  [Previous commits](#2.2)
#     -  [Parameters and LB score visualization](#2.3)
# 1. [Download data](#3)
# 1. [FE & Data Preprocessing](#4)
#     - [RankGauss](#4.1)
#     - [Seed](#4.2)    
#     - [PCA features](#4.3)
#     - [Feature selection](#4.4)
#     - [CV folds](#4.5)
#     - [Dataset Classes](#4.6)
#     - [Smoothing](#4.7)
#     - [Preprocessing](#4.8)
# 1. [Modeling](#5)
# 1. [Prediction & Submission](#6)

# ## 1. Import libraries<a class="anchor" id="1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[1]:


import sys
sys.path.append('../input/iterativestratification')

import numpy as np
import random
import pandas as pd
import os
import copy
import gc

import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import scipy.stats as stats
from scipy.stats import kurtosis

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import _WeightedLoss

import warnings
warnings.filterwarnings('ignore')

os.listdir('../input/lish-moa')

pd.set_option('max_columns', 2000)


# ## 2. My upgrade <a class="anchor" id="2"></a>
# 
# [Back to Table of Contents](#0.1)

# ### 2.1. Commit now <a class="anchor" id="2.1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[2]:


# From optimal commit 9
n_comp_GENES = 463
n_comp_CELLS = 60
VarianceThreshold_for_FS = 0.9
Dropout_Model = 0.25
#QT_n_quantile_min=50, 
#QT_n_quantile_max=1000,
print('n_comp_GENES', n_comp_GENES, 'n_comp_CELLS', n_comp_CELLS, 'total', n_comp_GENES + n_comp_CELLS)


# ### 2.2 Previous commits <a class="anchor" id="2.2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[3]:


commits_df = pd.DataFrame(columns = ['n_commit', 'n_comp_GENES', 'n_comp_CELLS', 'train_features','VarianceThreshold_for_FS', 'Dropout_Model', 'LB_score', 'CV_logloss'])


# ### Commit 0 (parameters from https://www.kaggle.com/riadalmadani/pytorch-cv-0-0145-lb-0-01839, commit 8)

# In[4]:


n=0
commits_df.loc[n, 'n_commit'] = 0                       # Number of commit
commits_df.loc[n, 'n_comp_GENES'] = 600                 # Number of output features for PCA for g-features
commits_df.loc[n, 'n_comp_CELLS'] = 50                  # Number of output features for PCA for c-features
commits_df.loc[n, 'VarianceThreshold_for_FS'] = 0.8     # Threshold for VarianceThreshold for feature selection
commits_df.loc[n, 'train_features'] = 1245              # Number features in the training dataframe after FE and before modeling
commits_df.loc[n, 'Dropout_Model'] = 0.2619422201258426 # Dropout in Model
commits_df.loc[n, 'CV_logloss'] = 0.01458269555140327   # Result CV logloss metrics
commits_df.loc[n, 'LB_score'] = 0.01839                 # LB score after submitting


# ### Commit 4

# In[5]:


n=1
commits_df.loc[n, 'n_commit'] = 4
commits_df.loc[n, 'n_comp_GENES'] = 610
commits_df.loc[n, 'n_comp_CELLS'] = 55
commits_df.loc[n, 'VarianceThreshold_for_FS'] = 0.82
commits_df.loc[n, 'train_features'] = 1240
commits_df.loc[n, 'Dropout_Model'] = 0.25
commits_df.loc[n, 'CV_logloss'] =  0.014584545081734047
commits_df.loc[n, 'LB_score'] = 0.01839


# ### Commit 5

# In[6]:


n=2
commits_df.loc[n, 'n_commit'] = 5
commits_df.loc[n, 'n_comp_GENES'] = 670
commits_df.loc[n, 'n_comp_CELLS'] = 67
commits_df.loc[n, 'VarianceThreshold_for_FS'] = 0.67
commits_df.loc[n, 'train_features'] = 1298
commits_df.loc[n, 'Dropout_Model'] = 0.25
commits_df.loc[n, 'CV_logloss'] =  0.014588561242139069
commits_df.loc[n, 'LB_score'] = 0.01840


# ### Commit 6

# In[7]:


n=3
commits_df.loc[n, 'n_commit'] = 6
commits_df.loc[n, 'n_comp_GENES'] = 450
commits_df.loc[n, 'n_comp_CELLS'] = 45
commits_df.loc[n, 'VarianceThreshold_for_FS'] = 0.67
commits_df.loc[n, 'train_features'] = 1297
commits_df.loc[n, 'Dropout_Model'] = 0.25
commits_df.loc[n, 'CV_logloss'] =  0.014586229676302227
commits_df.loc[n, 'LB_score'] = 0.01840


# ### Commit 9

# In[8]:


n=4
commits_df.loc[n, 'n_commit'] = 9
commits_df.loc[n, 'n_comp_GENES'] = 463
commits_df.loc[n, 'n_comp_CELLS'] = 60
commits_df.loc[n, 'VarianceThreshold_for_FS'] = 0.9
commits_df.loc[n, 'train_features'] = 1219
commits_df.loc[n, 'Dropout_Model'] = 0.25
commits_df.loc[n, 'CV_logloss'] =  0.014572358066092783
commits_df.loc[n, 'LB_score'] = 0.01839


# ### Commit 10

# In[9]:


n=5
commits_df.loc[n, 'n_commit'] = 10
commits_df.loc[n, 'n_comp_GENES'] = 463
commits_df.loc[n, 'n_comp_CELLS'] = 80
commits_df.loc[n, 'VarianceThreshold_for_FS'] = 0.92
commits_df.loc[n, 'train_features'] = 1214
commits_df.loc[n, 'Dropout_Model'] = 0.25
commits_df.loc[n, 'CV_logloss'] =  0.014571552074579226
commits_df.loc[n, 'LB_score'] = 0.01841


# ### Commit 12

# In[10]:


n=6
commits_df.loc[n, 'n_commit'] = 12
commits_df.loc[n, 'n_comp_GENES'] = 450
commits_df.loc[n, 'n_comp_CELLS'] = 65
commits_df.loc[n, 'VarianceThreshold_for_FS'] = 0.9
commits_df.loc[n, 'train_features'] = 1219
commits_df.loc[n, 'Dropout_Model'] = 0.25
commits_df.loc[n, 'CV_logloss'] = 0.01458043214513875
commits_df.loc[n, 'LB_score'] = 0.01840


# ### Commit 13

# In[11]:


n=7
commits_df.loc[n, 'n_commit'] = 13
commits_df.loc[n, 'n_comp_GENES'] = 463
commits_df.loc[n, 'n_comp_CELLS'] = 60
commits_df.loc[n, 'VarianceThreshold_for_FS'] = 0.9
commits_df.loc[n, 'train_features'] = 1219
commits_df.loc[n, 'Dropout_Model'] = 0.4
commits_df.loc[n, 'CV_logloss'] = 0.014625250378417162
commits_df.loc[n, 'LB_score'] = 0.01844


# ### Commit 14

# In[12]:


n=8
commits_df.loc[n, 'n_commit'] = 14
commits_df.loc[n, 'n_comp_GENES'] = 463
commits_df.loc[n, 'n_comp_CELLS'] = 60
commits_df.loc[n, 'VarianceThreshold_for_FS'] = 0.01
commits_df.loc[n, 'train_features'] = 1604
commits_df.loc[n, 'Dropout_Model'] = 0.25
commits_df.loc[n, 'CV_logloss'] = 0.014713482787703418
commits_df.loc[n, 'LB_score'] = 0.01849


# ### Commit 18

# In[13]:


n=9
commits_df.loc[n, 'n_commit'] = 18
commits_df.loc[n, 'n_comp_GENES'] = 363
commits_df.loc[n, 'n_comp_CELLS'] = 60
commits_df.loc[n, 'VarianceThreshold_for_FS'] = 0.9
commits_df.loc[n, 'train_features'] = 1219
commits_df.loc[n, 'Dropout_Model'] = 0.25
commits_df.loc[n, 'CV_logloss'] = 0.014568689235607534
commits_df.loc[n, 'LB_score'] = 0.01841


# ### Commit 19

# In[14]:


n=10
commits_df.loc[n, 'n_commit'] = 19
commits_df.loc[n, 'n_comp_GENES'] = 550
commits_df.loc[n, 'n_comp_CELLS'] = 60
commits_df.loc[n, 'VarianceThreshold_for_FS'] = 0.91
commits_df.loc[n, 'train_features'] = 1218
commits_df.loc[n, 'Dropout_Model'] = 0.25
commits_df.loc[n, 'CV_logloss'] = 0.014577509066710863
commits_df.loc[n, 'LB_score'] = 0.01841


# ### Commit 20

# In[15]:


n=11
commits_df.loc[n, 'n_commit'] = 20
commits_df.loc[n, 'n_comp_GENES'] = 463
commits_df.loc[n, 'n_comp_CELLS'] = 60
commits_df.loc[n, 'VarianceThreshold_for_FS'] = 0.9
commits_df.loc[n, 'train_features'] = 1218
commits_df.loc[n, 'Dropout_Model'] = 0.25
commits_df.loc[n, 'CV_logloss'] = 0.014572358066092783
commits_df.loc[n, 'LB_score'] = 0.01839


# In[16]:


# Commits 0-20
commits_df['QT_n_quantile_max'] = 100


# ### Commit 21

# In[17]:


n=12
commits_df.loc[n, 'n_commit'] = 21
commits_df.loc[n, 'n_comp_GENES'] = 463
commits_df.loc[n, 'n_comp_CELLS'] = 60
commits_df.loc[n, 'VarianceThreshold_for_FS'] = 0.9
commits_df.loc[n, 'train_features'] = 1223
commits_df.loc[n, 'Dropout_Model'] = 0.25
commits_df.loc[n, 'QT_n_quantile_max'] = 200
commits_df.loc[n, 'CV_logloss'] = 0.014585887029392697
commits_df.loc[n, 'LB_score'] = 0.01841


# ### Commit 22

# In[18]:


n=13
commits_df.loc[n, 'n_commit'] = 22
commits_df.loc[n, 'n_comp_GENES'] = 463
commits_df.loc[n, 'n_comp_CELLS'] = 60
commits_df.loc[n, 'VarianceThreshold_for_FS'] = 0.9
commits_df.loc[n, 'train_features'] = 1224
commits_df.loc[n, 'Dropout_Model'] = 0.25
commits_df.loc[n, 'QT_n_quantile_max'] = 500
commits_df.loc[n, 'CV_logloss'] = 0.014572447523411875
commits_df.loc[n, 'LB_score'] = 0.01840


# ### Commit 23

# In[19]:


n=14
commits_df.loc[n, 'n_commit'] = 23
commits_df.loc[n, 'n_comp_GENES'] = 463
commits_df.loc[n, 'n_comp_CELLS'] = 60
commits_df.loc[n, 'VarianceThreshold_for_FS'] = 0.9
commits_df.loc[n, 'train_features'] = 1212
commits_df.loc[n, 'Dropout_Model'] = 0.25
commits_df.loc[n, 'QT_n_quantile_max'] = 50
commits_df.loc[n, 'CV_logloss'] = 0.014581633680902033
commits_df.loc[n, 'LB_score'] = 0.01840


# In[20]:


# Commits 0-23
commits_df['QT_n_quantile_min'] = commits_df['QT_n_quantile_max']


# ### Commit 24

# In[21]:


n=15
commits_df.loc[n, 'n_commit'] = 24
commits_df.loc[n, 'n_comp_GENES'] = 463
commits_df.loc[n, 'n_comp_CELLS'] = 60
commits_df.loc[n, 'VarianceThreshold_for_FS'] = 0.84
commits_df.loc[n, 'train_features'] = 1215
commits_df.loc[n, 'Dropout_Model'] = 0.25
commits_df.loc[n, 'QT_n_quantile_min'] = 10
commits_df.loc[n, 'QT_n_quantile_max'] = 200
commits_df.loc[n, 'CV_logloss'] = 0.014578913453054567
commits_df.loc[n, 'LB_score'] = 0.01840


# ### Commit 25

# In[22]:


n=15
commits_df.loc[n, 'n_commit'] = 24
commits_df.loc[n, 'n_comp_GENES'] = 463
commits_df.loc[n, 'n_comp_CELLS'] = 60
commits_df.loc[n, 'VarianceThreshold_for_FS'] = 0.9
commits_df.loc[n, 'train_features'] = 1223
commits_df.loc[n, 'Dropout_Model'] = 0.25
commits_df.loc[n, 'QT_n_quantile_min'] = 50
commits_df.loc[n, 'QT_n_quantile_max'] = 1000
commits_df.loc[n, 'CV_logloss'] = 0.014576369890002664
commits_df.loc[n, 'LB_score'] = 0.01842


# ### 2.3 Parameters and LB score visualization <a class="anchor" id="2.3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[23]:


commits_df['n_comp_total'] = commits_df['n_comp_GENES'] + commits_df['n_comp_CELLS']
commits_df['seed'] = 42


# In[24]:


commits_df['l_rate'] = 1e-3
commits_df.loc[11, 'l_rate'] = 5e-4


# In[25]:


# Find and mark minimun value of LB score
commits_df['LB_score'] = pd.to_numeric(commits_df['LB_score'])
commits_df = commits_df.sort_values(by=['LB_score', 'CV_logloss'], ascending = True).reset_index(drop=True)
commits_df['min'] = 0
commits_df.loc[0, 'min'] = 1
commits_df


# In[26]:


commits_df.sort_values(by=['CV_logloss'], ascending = True)


# In[27]:


# Interactive plot with results of parameters tuning
fig = px.scatter_3d(commits_df, x='n_comp_GENES', y='n_comp_CELLS', z='LB_score', color = 'min', 
                    symbol = 'Dropout_Model',
                    title='Parameters and LB score visualization of MoA solutions')
fig.update(layout=dict(title=dict(x=0.1)))


# In[28]:


# Interactive plot with results of parameters tuning
fig = px.scatter_3d(commits_df, x='train_features', y='VarianceThreshold_for_FS', z='LB_score', color = 'min', 
                    symbol = 'l_rate',
                    title='Parameters and LB score visualization of MoA solutions')
fig.update(layout=dict(title=dict(x=0.1)))


# In[29]:


# Interactive plot with results of parameters tuning
fig = px.scatter_3d(commits_df, x='train_features', y='CV_logloss', z='LB_score', color = 'min', 
                    symbol = 'l_rate',
                    title='Parameters and LB score visualization of MoA solutions')
fig.update(layout=dict(title=dict(x=0.1)))


# In[30]:


# Interactive plot with results of parameters tuning
commits_df_1841 = commits_df[commits_df.LB_score <= 0.01841]
fig = px.scatter_3d(commits_df_1841, x='train_features', y='CV_logloss', z='LB_score', color = 'min', 
                    symbol = 'l_rate',
                    title='Parameters and LB score visualization of MoA solutions')
fig.update(layout=dict(title=dict(x=0.1)))


# In[31]:


# Interactive plot with results of parameters tuning
commits_df_1840 = commits_df[commits_df.LB_score <= 0.01840]
fig = px.scatter_3d(commits_df_1840, x='QT_n_quantile_max', y='train_features', z='LB_score', color = 'min', 
                    symbol = 'seed',
                    title='Parameters and LB score visualization of MoA solutions')
fig.update(layout=dict(title=dict(x=0.1)))


# In[32]:


# Interactive plot with results of parameters tuning
commits_df_1842 = commits_df[commits_df.LB_score <= 0.01842]
fig = px.scatter_3d(commits_df_1842, x='QT_n_quantile_min', y='QT_n_quantile_max', z='LB_score', color = 'min',
                    title='Parameters and LB score visualization of MoA solutions')
fig.update(layout=dict(title=dict(x=0.1)))


# As already noted, **LEARNING_RATE** is adaptively adjusted. Therefore, it makes no sense to tune it - the result is the same.

# ### It is recommended:
# * **n_comp_GENES** smaller, 
# * **n_comp_CELLS** more,
# * **VarianceThreshold_for_FS** more, so that **train_features** is less.

# ## 3. Download data<a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[33]:


train_features = pd.read_csv('../input/lish-moa/train_features.csv')
train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')
sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')


# ## 4. FE & Data Preprocessing <a class="anchor" id="4"></a>
# 
# [Back to Table of Contents](#0.1)

# In[34]:


GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]


# ### 4.1 RankGauss<a class="anchor" id="4.1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[35]:


# Search for minimum and maximum values
# df_kurt = pd.DataFrame(columns=['col','train', 'test'])
# i = 0
# for col in (GENES + CELLS):
#     df_kurt.loc[i, 'col'] = col
#     df_kurt.loc[i, 'train'] = kurtosis(train_features[col])
#     df_kurt.loc[i, 'test'] = kurtosis(test_features[col])
#     i += 1
# print(df_kurt.min())
# print(df_kurt.max())


# In[36]:


def calc_QT_par_kurt(QT_n_quantile_min=10, QT_n_quantile_max=200):
    # Calculation parameters of function: n_quantile(kurtosis) = k1*kurtosis + k0
    # For Train & Test datasets (GENES + CELLS features): minimum kurtosis = 1.53655, maximum kurtosis = 30.4929
    
    a = np.array([[1.53655,1], [30.4929,1]])
    b = np.array([QT_n_quantile_min, QT_n_quantile_max])
    
    return np.linalg.solve(a, b)


# In[37]:


def n_quantile_for_kurt(kurt, calc_QT_par_kurt_transform):
    # Calculation parameters of function: n_quantile(kurtosis) = calc_QT_par_kurt_transform[0]*kurtosis + calc_QT_par_kurt_transform[1]
    return int(calc_QT_par_kurt_transform[0]*kurt + calc_QT_par_kurt_transform[1])


# In[38]:


# RankGauss - transform to Gauss

for col in (GENES + CELLS):

    #kurt = max(kurtosis(train_features[col]), kurtosis(test_features[col]))
    #QuantileTransformer_n_quantiles = n_quantile_for_kurt(kurt, calc_QT_par_kurt(QT_n_quantile_min, QT_n_quantile_max))
    #transformer = QuantileTransformer(n_quantiles=QuantileTransformer_n_quantiles,random_state=0, output_distribution="normal")
    
    transformer = QuantileTransformer(n_quantiles=100,random_state=0, output_distribution="normal")   # from optimal commit 9
    vec_len = len(train_features[col].values)
    vec_len_test = len(test_features[col].values)
    raw_vec = train_features[col].values.reshape(vec_len, 1)
    transformer.fit(raw_vec)

    train_features[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
    test_features[col] = transformer.transform(test_features[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]


# ### 4.2 Seed<a class="anchor" id="4.2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[39]:


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)


# ### 4.3 PCA features<a class="anchor" id="4.3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[40]:


len(GENES)


# In[41]:


# GENES

data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])
data2 = (PCA(n_components=n_comp_GENES, random_state=42).fit_transform(data[GENES]))
train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp_GENES)])
test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp_GENES)])

train_features = pd.concat((train_features, train2), axis=1)
test_features = pd.concat((test_features, test2), axis=1)


# In[42]:


len(CELLS)


# In[43]:


# CELLS

data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])
data2 = (PCA(n_components=n_comp_CELLS, random_state=42).fit_transform(data[CELLS]))
train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp_CELLS)])
test2 = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp_CELLS)])

train_features = pd.concat((train_features, train2), axis=1)
test_features = pd.concat((test_features, test2), axis=1)


# In[44]:


train_features.shape


# In[45]:


train_features.head(5)


# ### 4.4 Feature selection<a class="anchor" id="4.4"></a>
# 
# [Back to Table of Contents](#0.1)

# In[46]:


data = train_features.append(test_features)
data


# In[47]:


var_thresh = VarianceThreshold(VarianceThreshold_for_FS)
data = train_features.append(test_features)
data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])

train_features_transformed = data_transformed[ : train_features.shape[0]]
test_features_transformed = data_transformed[-test_features.shape[0] : ]


train_features = pd.DataFrame(train_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                              columns=['sig_id','cp_type','cp_time','cp_dose'])

train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)


test_features = pd.DataFrame(test_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                             columns=['sig_id','cp_type','cp_time','cp_dose'])

test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)

train_features.shape


# In[48]:


train_features.head(5)


# In[49]:


train = train_features.merge(train_targets_scored, on='sig_id')
train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)

target = train[train_targets_scored.columns]


# In[50]:


train = train.drop('cp_type', axis=1)
test = test.drop('cp_type', axis=1)


# In[51]:


train.head(5)


# In[52]:


target_cols = target.drop('sig_id', axis=1).columns.values.tolist()


# ### 4.5 CV folds<a class="anchor" id="4.5"></a>
# 
# [Back to Table of Contents](#0.1)

# In[53]:


folds = train.copy()

mskf = MultilabelStratifiedKFold(n_splits=7)

for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):
    folds.loc[v_idx, 'kfold'] = int(f)

folds['kfold'] = folds['kfold'].astype(int)
folds


# In[54]:


print(train.shape)
print(folds.shape)
print(test.shape)
print(target.shape)
print(sample_submission.shape)


# ### 4.6 Dataset Classes<a class="anchor" id="4.6"></a>
# 
# [Back to Table of Contents](#0.1)

# In[55]:


class MoADataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
            'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)            
        }
        return dct
    
class TestDataset:
    def __init__(self, features):
        self.features = features
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct    


# In[56]:


def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0
    
    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        final_loss += loss.item()
        
    final_loss /= len(dataloader)
    
    return final_loss


def valid_fn(model, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []
    
    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        final_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())
        
    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)
    
    return final_loss, valid_preds

def inference_fn(model, dataloader, device):
    model.eval()
    preds = []
    
    for data in dataloader:
        inputs = data['x'].to(device)

        with torch.no_grad():
            outputs = model(inputs)
        
        preds.append(outputs.sigmoid().detach().cpu().numpy())
        
    preds = np.concatenate(preds)
    
    return preds


# ### 4.7 Smoothing<a class="anchor" id="4.7"></a>
# 
# [Back to Table of Contents](#0.1)

# In[57]:


class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


# ### 4.8 Preprocessing<a class="anchor" id="4.8"></a>
# 
# [Back to Table of Contents](#0.1)

# In[58]:


def process_data(data):
    data = pd.get_dummies(data, columns=['cp_time','cp_dose'])
    return data


# In[59]:


feature_cols = [c for c in process_data(folds).columns if c not in target_cols]
feature_cols = [c for c in feature_cols if c not in ['kfold','sig_id']]
len(feature_cols)


# ## 5. Modeling<a class="anchor" id="5"></a>
# 
# [Back to Table of Contents](#0.1)

# In[60]:


# HyperParameters

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 25
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NFOLDS = 7
EARLY_STOPPING_STEPS = 10
EARLY_STOP = False

num_features=len(feature_cols)
num_targets=len(target_cols)
hidden_size=1500


# In[61]:


class Model(nn.Module):
    
    def __init__(self, num_features, num_targets, hidden_size):
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))
        
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(Dropout_Model)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))
        
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(Dropout_Model)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))
    
    
    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))
        
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))
        
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)
        
        return x
    
    
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))    


# In[62]:


def run_training(fold, seed):
    
    seed_everything(seed)
    
    train = process_data(folds)
    test_ = process_data(test)
    
    trn_idx = train[train['kfold'] != fold].index
    val_idx = train[train['kfold'] == fold].index
    
    train_df = train[train['kfold'] != fold].reset_index(drop=True)
    valid_df = train[train['kfold'] == fold].reset_index(drop=True)
    
    x_train, y_train  = train_df[feature_cols].values, train_df[target_cols].values
    x_valid, y_valid =  valid_df[feature_cols].values, valid_df[target_cols].values
    
    train_dataset = MoADataset(x_train, y_train)
    valid_dataset = MoADataset(x_valid, y_valid)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )
    
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                              max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))
    
    loss_fn = nn.BCEWithLogitsLoss()
    loss_tr = SmoothBCEwLogits(smoothing =0.001)
    
    early_stopping_steps = EARLY_STOPPING_STEPS
    early_step = 0
   
    oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))
    best_loss = np.inf
    
    for epoch in range(EPOCHS):
        
        train_loss = train_fn(model, optimizer,scheduler, loss_tr, trainloader, DEVICE)
        print(f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}")
        valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)
        print(f"FOLD: {fold}, EPOCH: {epoch}, valid_loss: {valid_loss}")
        
        if valid_loss < best_loss:
            
            best_loss = valid_loss
            oof[val_idx] = valid_preds
            torch.save(model.state_dict(), f"FOLD{fold}_.pth")
        
        elif(EARLY_STOP == True):
            
            early_step += 1
            if (early_step >= early_stopping_steps):
                break
            
    
    #--------------------- PREDICTION---------------------
    x_test = test_[feature_cols].values
    testdataset = TestDataset(x_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,

    )
    
    model.load_state_dict(torch.load(f"FOLD{fold}_.pth"))
    model.to(DEVICE)
    
    predictions = np.zeros((len(test_), target.iloc[:, 1:].shape[1]))
    predictions = inference_fn(model, testloader, DEVICE)
    
    return oof, predictions


# ## 6. Prediction & Submission <a class="anchor" id="6"></a>
# 
# [Back to Table of Contents](#0.1)

# In[63]:


def run_k_fold(NFOLDS, seed):
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))
    
    for fold in range(NFOLDS):
        oof_, pred_ = run_training(fold, seed)
        
        predictions += pred_ / NFOLDS
        oof += oof_
        
    return oof, predictions


# In[64]:


# Averaging on multiple SEEDS

SEED = [0, 1, 2, 3, 4, 5, 6]
oof = np.zeros((len(train), len(target_cols)))
predictions = np.zeros((len(test), len(target_cols)))

for seed in SEED:
    
    oof_, predictions_ = run_k_fold(NFOLDS, seed)
    oof += oof_ / len(SEED)
    predictions += predictions_ / len(SEED)

train[target_cols] = oof
test[target_cols] = predictions


# In[65]:


train_targets_scored


# In[66]:


len(target_cols)


# In[67]:


valid_results = train_targets_scored.drop(columns=target_cols).merge(train[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)

y_true = train_targets_scored[target_cols].values
y_pred = valid_results[target_cols].values

score = 0
for i in range(len(target_cols)):
    score_ = log_loss(y_true[:, i], y_pred[:, i])
    score += score_ / target.shape[1]
    
print("CV log_loss: ", score)    


# In[68]:


sub = sample_submission.drop(columns=target_cols).merge(test[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)
sub.to_csv('submission.csv', index=False)


# In[69]:


sub.shape


# [Go to Top](#0)
