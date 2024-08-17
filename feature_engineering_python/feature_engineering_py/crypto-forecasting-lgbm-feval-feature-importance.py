#!/usr/bin/env python
# coding: utf-8

# # Crypto Forecasting - Basic LGBM
# 
# Basic lgbm, using standard Feature Engineering from here: 

# In[1]:


import gresearch_crypto

import pandas as pd
import numpy as np
import os
import gc
import pickle

import time
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb

seed = 2021

DEBUG = False


# In[2]:


# https://stackoverflow.com/questions/38641691/weighted-correlation-coefficient-with-pandas
def wmean(x, w):
    return np.sum(x * w) / np.sum(w)

def wcov(x, y, w):
    return np.sum(w * (x - wmean(x, w)) * (y - wmean(y, w))) / np.sum(w)

def wcorr(x, y, w):
    return wcov(x, y, w) / np.sqrt(wcov(x, x, w) * wcov(y, y, w))

def eval_wcorr(preds, train_data):
    w = train_data.add_w.values.flatten()
    y_true = train_data.get_label()
    return 'eval_wcorr', wcorr(preds, y_true, w), True


# In[3]:


n_fold = 5

importances = []

for fold in range(n_fold):
    print('Fold: '+str(fold))

    train = pd.read_parquet('../input/crypto-forecasting-static-feature-engineering/train_fold_'+str(fold)+'.parquet')
    test = pd.read_parquet('../input/crypto-forecasting-static-feature-engineering/test_fold_'+str(fold)+'.parquet')
    
    if DEBUG:
        timestamp_sample_train = train.timestamp.unique()[:np.int(len(train.timestamp.unique())*0.05)]
        timestamp_sample_test = test.timestamp.unique()[:np.int(len(test.timestamp.unique())*0.05)]
        train = train[train.timestamp.isin(timestamp_sample_train)]
        test = test[test.timestamp.isin(timestamp_sample_test)]

    y_train = train['Target']
    y_test = test['Target']

    features = [col for col in train.columns if col not in {'timestamp', 'Target', 'Target_M','weights'}]

    weights_train = train[['weights']]
    weights_test = test[['weights']]

    train = train[features]
    test = test[features]
    
    train_dataset = lgb.Dataset(train, y_train, feature_name = features, categorical_feature= ['Asset_ID'])
    val_dataset = lgb.Dataset(test, y_test, feature_name = features, categorical_feature= ['Asset_ID'])

    train_dataset.add_w = weights_train
    val_dataset.add_w = weights_test

    val_data = test
    val_y = y_test

    del train
    
    evals_result = {}
    
    # parameters
    params = {'n_estimators': 2000,
            'objective': 'regression',
            'metric': 'None',
            'boosting_type': 'gbdt',
            'max_depth': -1,
            'learning_rate': 0.05,
            'subsample': 0.72,
            'subsample_freq': 4,
            'feature_fraction': 0.4,
            'lambda_l1': 1,
            'lambda_l2': 1,
            'seed': 46,
            'verbose': -1,
            }

    model = lgb.train(params = params,
                      train_set = train_dataset, 
                      valid_sets = [val_dataset],
                      #early_stopping_rounds=1000,
                      verbose_eval = 100,
                      feval=eval_wcorr,
                      evals_result = evals_result 
                     )
    
    importances.append(model.feature_importance(importance_type='gain'))
    
    plt.plot(np.array(evals_result['valid_0']['eval_wcorr']), label='fold '+str(fold))
    
plt.legend(loc="upper left")
plt.show()


# from nyanp's Optiver solution.

# In[4]:


def plot_importance(importances, features_names = features, PLOT_TOP_N = 20, figsize=(10, 10)):
    importance_df = pd.DataFrame(data=importances, columns=features)
    sorted_indices = importance_df.median(axis=0).sort_values(ascending=False).index
    sorted_importance_df = importance_df.loc[:, sorted_indices]
    plot_cols = sorted_importance_df.columns[:PLOT_TOP_N]
    _, ax = plt.subplots(figsize=figsize)
    ax.grid()
    ax.set_xscale('log')
    ax.set_ylabel('Feature')
    ax.set_xlabel('Importance')
    sns.boxplot(data=sorted_importance_df[plot_cols],
                orient='h',
                ax=ax)
    plt.show()


# In[5]:


plot_importance(np.array(importances),features, PLOT_TOP_N = 20, figsize=(10, 20))

