#!/usr/bin/env python
# coding: utf-8

# # AMEX - lgbm + Feature Engineering
# 
# Baseline lgbm approach + blanket feature eng.
# 
# The magic (Feature Engineering) is here: https://www.kaggle.com/code/lucasmorin/amex-feature-engineering
# 
# Some stuff from my G-research lgbm baseline: https://www.kaggle.com/code/lucasmorin/online-fe-lgbm-feval-importances
# 
# And my JPX XGBRanker baseline: https://www.kaggle.com/code/lucasmorin/jpx-all-data-xgbranker
# 
# Some other stuff from @ambrosm lgbm baseline here: https://www.kaggle.com/code/ambrosm/amex-lightgbm-quickstart
# 
# **This Notebook is part of a serie built for the AMEX competition:**
# - [Base Feature engineering](https://www.kaggle.com/code/lucasmorin/amex-feature-engineering-base)
# - [Baseline lgbm](https://www.kaggle.com/code/lucasmorin/amex-lgbm-features-eng)
# - [Feature Engineering 2: aggregation function](https://www.kaggle.com/code/lucasmorin/amex-feature-engineering-2-aggreg-functions)
# - [Feature Engineering 3: transformation function](https://www.kaggle.com/code/lucasmorin/amex-feature-engineering-3-transform-functions)
# 
# **With associated Data Sets:**
# - [Base Feature engineering](https://www.kaggle.com/datasets/lucasmorin/amex-base-fe)
# - [Feature Engineering 2 - aggregation function](https://www.kaggle.com/datasets/lucasmorin/amex-fe2)
# - [Feature Engineering 3 - transform function](https://www.kaggle.com/datasets/lucasmorin/amex-fe3)
# 
# 
# 
# **Please make sure to upvote everything you use / find interesting / usefull**

# In[1]:


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

from sklearn.model_selection import StratifiedKFold

base_seed = 0

DEBUG = False
TEST = True

if ~DEBUG:
    warnings.filterwarnings("ignore")


# In[2]:


train_data = pd.read_pickle('../input/amex-feature-engineering/train_data_agg.pkl')
train_labels = pd.read_csv('../input/amex-default-prediction/train_labels.csv').set_index('customer_ID').loc[train_data.index]

Features = train_data.columns


# In[3]:


def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)

# from ambrosm notebook
def lgb_amex_metric(y_true, y_pred):
    """The competition metric with lightgbm's calling convention"""
    return ('amex',
            amex_metric(pd.DataFrame({'target': y_true}), pd.Series(y_pred, name='prediction')),
            True)

#see : https://www.kaggle.com/competitions/amex-default-prediction/discussion/327534
def amex_metric_mod_lgbm(y_pred: np.ndarray, data: lgb.Dataset):

    y_true = data.get_label()
    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 'AMEX', 0.5 * (gini[1]/gini[0]+ top_four), True


# In[4]:


n_fold = 2 if DEBUG else 3
n_seed = 1 if TEST else (2 if DEBUG else 3)
n_estimators = 100 if DEBUG else 10000

kf = StratifiedKFold(n_splits=n_fold)

importances = []
importances_split = []
models = {}
df_scores = []

ids_folds = {}
preds_tr_va = {}

SAMPLE = False


for fold, (idx_tr, idx_va) in enumerate(kf.split(train_data, train_labels)):
    
    if TEST:
        if fold>0:
            continue
    
    ids_folds[fold] = (idx_tr, idx_va)
    
    X_tr = train_data[Features].iloc[idx_tr]
    X_va = train_data[Features].iloc[idx_va]
    y_tr = train_labels.iloc[idx_tr]
    y_va = train_labels.iloc[idx_va]
    
    lgb_train_data = lgb.Dataset(X_tr, label=y_tr)
    lgb_val_data = lgb.Dataset(X_va, label=y_va)
    
    for seed in range(n_seed):
        print('Fold: '+str(fold)+ ' - seed: '+str(seed))
        key = str(fold)+'-'+str(seed)
        
        parameters = {
            'objective': 'cross_entropy_lambda',
            'boosting': 'dart',
            'learning_rate': 0.005,
            #'min_child_samples': 1000,
            'reg_alpha':10,
            'feature_fraction':0.3,
            'bagging_fraction':0.3,
            'max_depth': 6,
            'seed':seed,
            'n_estimators':n_estimators,
            'verbose':-1,
            'linear_tree': True
        }

        clf = lgb.train(parameters,
                               lgb_train_data,
                               valid_sets = [lgb_train_data,lgb_val_data],
                               verbose_eval = 100,
                               feval=amex_metric_mod_lgbm,
                               early_stopping_rounds=200)

        
        preds_tr = pd.Series(clf.predict(X_tr)).rename('prediction')
        preds_va = pd.Series(clf.predict(X_va)).rename('prediction')
        
        preds_tr_va[(fold, seed)] = (preds_tr, preds_va)
        
        score = amex_metric(y_va.reset_index(drop=True), preds_va)
        models[key] = clf
        df_scores.append((fold, seed, score))
        print(f'Fold: {fold} - seed: {seed} - score {score:.2%}')
        importances.append(clf.feature_importance(importance_type='gain'))
        importances_split.append(clf.feature_importance(importance_type='split'))


# In[5]:


pickle.dump(ids_folds, open('ids_folds.p', 'wb'))
pickle.dump(preds_tr_va, open('preds_tr_va.p', 'wb'))


# In[6]:


df_results = pd.DataFrame(df_scores,columns=['fold','seed','score']).pivot(index='fold',columns='seed',values='score')

df_results.loc['seed_mean']= df_results.mean(numeric_only=True, axis=0)
df_results.loc[:,'fold_mean'] = df_results.mean(numeric_only=True, axis=1)
df_results


# In[7]:


def plot_importance(importances, features, PLOT_TOP_N = 20, figsize=(10, 10)):
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
    
plot_importance(np.array(importances),train_data.columns, PLOT_TOP_N = 20, figsize=(10, 20))


# In[8]:


importance_split_df = pd.DataFrame(data=importances_split, columns=train_data.columns)
sorted_split = importance_split_df.median(axis=0).sort_values(ascending=False)
sorted_split.head()

print(f'nb features used:{len(sorted_split[sorted_split>1])}')


# In[9]:


def plot_importance_groups(importances, features_names = train_data.columns, PLOT_TOP_N = 20, figsize=(4, 8)):
    importance_df = pd.DataFrame(data=importances, columns=features_names)
    sorted_indices = importance_df.median(axis=0).sort_values(ascending=False).index
    sorted_importance_df = importance_df.loc[:, sorted_indices]
    plot_cols = sorted_importance_df.columns[:PLOT_TOP_N]
    
    t = sorted_importance_df.transpose()
    t['groups'] = [s.split('_')[-1] for s in sorted_importance_df.columns]
    
    t = t.groupby('groups').sum().transpose()
    t = t.loc[:,t.columns.isin(['last','max','mean','min','std'])]

    _, ax = plt.subplots(figsize=figsize)
    ax.grid()
    #ax.set_xscale('log')
    ax.set_ylabel('Feature')
    ax.set_xlabel('Importance')
    sns.boxplot(data=t,
                orient='h',
                ax=ax)
    plt.show()
    
plot_importance_groups(np.array(importances))


# indeed 'last' features play a major role. 

# In[10]:


del train_data, train_labels, X_tr, X_va, y_tr, y_va 


# # submission
# 

# In[11]:


test_data = pd.read_pickle('../input/amex-feature-engineering/test_data_agg.pkl').astype('float16')

missing_cols = [f for f in Features if f not in test_data.columns]
test_data[missing_cols] = 0


# In[12]:


# https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

split_ids = split(test_data.index.unique(),10)

df_list_preds = []


# In[13]:


preds_sub = []

for (i,ids) in enumerate(split_ids):
    print(f'chunk {i}')
    test_data_ids = test_data[test_data.index.isin(ids)][Features]
    preds_ids_sub = []
    for k in models:
        print(f'model key {k}')
        preds_ids_sub.append(models[k].predict(test_data_ids, raw_score=True))
    preds_sub.append(np.nanmean(np.array(preds_ids_sub),axis=0))
    gc.collect()
    
preds_sub = np.hstack(preds_sub)
preds_series = pd.Series(preds_sub,index = test_data.index.unique())
proba_series = np.exp(preds_series)/(1+np.exp(preds_series))


# In[14]:


# preds_series.hist(bins=100)
# plt.axvline(x=np.log(0.04/(1-0.04)),color='black');


# In[15]:


# proba_series.hist(bins=100)
# plt.axvline(x=0.04,color='black');


# In[16]:


df_sub = pd.read_csv('../input/amex-default-prediction/sample_submission.csv')
df_sub.prediction = proba_series.loc[df_sub.customer_ID].values

df_sub = df_sub.set_index('customer_ID')
df_sub.to_csv('submission.csv')

