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

import warnings

base_seed = 0

DEBUG = False

if ~DEBUG:
    warnings.filterwarnings("ignore")


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

asset_details = pd.read_csv('../input/g-research-crypto-forecasting/asset_details.csv')

#create dictionnary of weights
dict_weights = {}
for i in range(asset_details.shape[0]):
    dict_weights[asset_details.iloc[i,0]] = asset_details.iloc[i,1]


# In[3]:


n_fold = 2 if DEBUG else 5
n_seed = 2 if DEBUG else 5

importances = []
models = {}
ES_it = {}
df_scores = []

#start early stopping after this number of rows
low = 100

SAMPLE = False

for fold in range(n_fold):
    
    train = pd.read_parquet('../input/on-line-feature-engineering/train_fold_'+str(fold)+'.parquet')
    test = pd.read_parquet('../input/on-line-feature-engineering/test_fold_'+str(fold)+'.parquet')

    if DEBUG or SAMPLE:
        timestamp_sample_train = train.timestamp.unique()[:np.int(len(train.timestamp.unique())*0.1)]
        timestamp_sample_test = test.timestamp.unique()[:np.int(len(test.timestamp.unique())*0.1)]
        train = train[train.timestamp.isin(timestamp_sample_train)]
        test = test[test.timestamp.isin(timestamp_sample_test)]


    train['weights'] = train.Asset_ID.map(dict_weights).astype('float32')
    test['weights'] = test.Asset_ID.map(dict_weights).astype('float32')    

    y_train = train['Target']
    y_test = test['Target']

    features = [col for col in train.columns if col not in {'timestamp', 'Target', 'Target_M','weights','Asset_ID'}]

    weights_train = train[['weights']]
    weights_test = test[['weights']]

    train = train[features]
    test = test[features]

    for seed in range(n_seed):    
        print('Fold: '+str(fold)+ ' - seed: '+str(seed))

        train_dataset = lgb.Dataset(train, y_train, feature_name = features)#, categorical_feature= ['Asset_ID'])
        val_dataset = lgb.Dataset(test, y_test, feature_name = features)#, categorical_feature= ['Asset_ID'])

        train_dataset.add_w = weights_train
        val_dataset.add_w = weights_test

        val_data = test
        val_y = y_test
        
        evals_result = {}

        # parameters
        # objective_params = [0.0001,0.001,0.01,0.1,1,10]

        params = {'n_estimators': 2500,
                'objective': 'regression',  #objectives = ['regression','regression_l1', 'huber', 'fair','quantile', 'mape', 'gamma','tweedie']
                #'fair_c': 100,
                'metric': 'None',
                'boosting_type': 'gbdt',
                'max_depth': -1,
                'learning_rate': 0.005,
                'subsample': 0.4,
                'subsample_freq': 4,
                'feature_fraction': 0.4,
                'lambda_l1': 1,
                'lambda_l2': 1,
                'seed': base_seed+seed,
                'verbose': -1,
                'min_data_in_leaf':100
                }

        model = lgb.train(params = params,
                          train_set = train_dataset, 
                          valid_sets = [val_dataset],
                          #early_stopping_rounds=1000,
                          verbose_eval = 100,
                          feval= eval_wcorr,
                          evals_result = evals_result 
                         )
        
        key = str(fold)+'-'+str(seed) 
        
        early_stopping_it = low + np.argmax(np.array(evals_result['valid_0']['eval_wcorr'])[low:])
        
        models[key] = model
        ES_it[key] = early_stopping_it
        
        df_scores.append((fold, seed, np.max(np.array(evals_result['valid_0']['eval_wcorr'])[low:])))

        importances.append(model.feature_importance(importance_type='gain'))

        plt.plot(np.array(evals_result['valid_0']['eval_wcorr']), label= 'fold '+str(fold)+' seed '+str(seed))
        
    plt.legend(loc="upper left", bbox_to_anchor=(1, 0.5))
    plt.show()


# In[4]:


df_results = pd.DataFrame(df_scores,columns=['fold','seed','score']).pivot(index='fold',columns='seed',values='score')

df_results.loc['seed_mean']= df_results.mean(numeric_only=True, axis=0)
df_results.loc[:,'fold_mean'] = df_results.mean(numeric_only=True, axis=1)
df_results


# from nyanp's Optiver solution.

# In[5]:


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


# In[6]:


plot_importance(np.array(importances),features, PLOT_TOP_N = 20, figsize=(10, 20))


# In[7]:


pickle.dump(models, open('lgbm_models.pkl', 'wb'))
pickle.dump(df_scores, open('scores.pkl', 'wb'))
pickle.dump(ES_it, open('ES_it.pkl', 'wb'))
pickle.dump(importances, open('importances.pkl', 'wb'))
pickle.dump(features, open('features.pkl', 'wb'))

