#!/usr/bin/env python
# coding: utf-8

# Simple weight of evidence baseline; WoE is a target encoding technique replacing values by an associated value that has nice additive properties.
# Give strong baseline, generally at the cost of feature interactions.
# 
# **Don't Forget to upvote if you find this interesting or usefull**

# In[1]:


import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin

# matplotlib setting
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

# pandas setting
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 1000)

DEBUG = False


# ## Aggregate_data

# In[2]:


get_ipython().run_cell_magic('time', '', "train_data_grp = pd.read_pickle('../input/amex-feature-engineering/train_data_agg.pkl')\ntrain_labels = pd.read_csv('../input/amex-default-prediction/train_labels.csv').set_index('customer_ID').loc[train_data_grp.index]\n")


# In[3]:


get_ipython().run_cell_magic('time', '', "test_data_grp = pd.read_pickle('../input/amex-feature-engineering/test_data_agg.pkl')\n")


# # weight of evidence
# 
# Standard Credit Scoring Technique. Target encoding technique that replace feature value by an additive value that helps build credit Scorecards. Personal sklearn implementation (doesn't handle edge case very well).

# In[4]:


class WoE_Imputer(BaseEstimator, TransformerMixin):
# Bins the features and impute Weight of Evidence associated with each bin
# Weight of Evidence is calculated as the log ratio of positive outcome to negative ones in each bin
# This imputation technique is adapted to the specific functionnal form of logistic regression
# Allows to impute missing values
# Also allows to calculate Information Values for feature selection
    def __init__(self, feature_name, n_bin = 100, Categorical = False, verbosity = 1):  
        self.feature_name = feature_name
        self.n_bin = n_bin
        self.bins = []
        self.WoE_values = []
        self.Categorical = Categorical 
        self.verbosity = verbosity
        self.IV = 0

    def fit(self, X, y = None):
        if y is None:
            raise ValueError('Woe Imputer is a supervised imputer. It needs a target')

        if self.Categorical:
            values_quantiles = X[self.feature_name].astype('category')
            self.bins = values_quantiles.cat
        else:
            values_quantiles, self.bins = pd.qcut(X[self.feature_name], q=self.n_bin, duplicates = 'drop', retbins=True)   
            self.bins[0] = -np.Inf
            self.bins[-1] = np.Inf
            values_quantiles = pd.cut(X[self.feature_name], bins = self.bins)

        values_quantiles = values_quantiles.cat.add_categories('missing_value')
        values_quantiles.fillna('missing_value', inplace = True) 

        df = pd.DataFrame({'group': values_quantiles, 'val': X[self.feature_name], 'target': y.values.flatten()})

        sum_positive_by_quantile = df.groupby('group').sum().target
        sum_negative_by_quantile = df.groupby('group').count().target - df.groupby('group').sum().target

        data = np.log(sum_positive_by_quantile / sum_negative_by_quantile)
        
        #interpolate in case of na - there are other tricks
        mask = np.isnan(data)
        data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])

        self.WoE_values =  data

        self.IV = ((sum_positive_by_quantile - sum_negative_by_quantile) * self.WoE_values / df.shape[0]).sum()

        if self.verbosity>0:
            print('Information Value ' + str(self.feature_name)+': ' + str(round(self.IV,5)))
            
        return self

    def transform(self, X):
        feature_to_transform = X[self.feature_name].copy()
        transformed_feature = pd.cut(feature_to_transform, bins =  self.bins, labels = np.array(self.WoE_values[:-1]), ordered = False).astype('float32')
        transformed_feature = transformed_feature.replace(np.nan, self.WoE_values[-1])
        X[self.feature_name] = transformed_feature
        return X

    def __get_val__(self):  
        return self.feature_name, self.n_bin, self.bins, self.WoE_values, self.IV


# In[5]:


from tqdm import tqdm_notebook

Features = train_data_grp.columns
Features = [f for f in Features if not f.startswith('target')]

if DEBUG:
    Features = Features[:20]

IV_list = []
Features_list = []

Features_list_not_ok = []

for f in tqdm_notebook(Features):
    try:
        WoE_imp = WoE_Imputer(f, n_bin = 50, verbosity = 0)
        WoE_imp.fit(train_data_grp, y = train_labels.target)
        train_data_grp = WoE_imp.transform(train_data_grp)
        test_data_grp = WoE_imp.transform(test_data_grp)
        feature_name, n_bin, bins, WoE_values, IV = WoE_imp.__get_val__()
        Features_list.append(feature_name)
        IV_list.append(IV)
    except:
        Features_list_not_ok.append(f)


# In[6]:


sorted_IV = pd.DataFrame({'Features':Features_list,'IV':IV_list}).sort_values('IV',ascending=False).reset_index(drop=True)
plt.plot(sorted_IV.IV);


# In[7]:


inf_value = sorted_IV.IV[0]

sorted_IV = sorted_IV[sorted_IV.IV != inf_value]


# In[8]:


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


# pos / (pos + neg) = 1/(1+neg/pos) = (1/1+exp(-log(pos/neg)))

# In[9]:


top_n = 625

list_features = sorted_IV.Features[:top_n].to_list()
pred_train = train_data_grp[list_features].mean(axis=1)
prob_train = 1/(1+np.exp(-pred_train))
metric = amex_metric(train_labels, prob_train.rename('prediction'))
print(f'n {top_n} - {metric:.2%}')


# # submission

# In[10]:


df_sub = pd.read_csv('../input/amex-default-prediction/sample_submission.csv')

pred_test = test_data_grp[list_features].mean(axis=1)
prob_test = 1/(1+np.exp(-pred_test))
df_sub.prediction = prob_test.loc[df_sub.customer_ID].values

df_sub.set_index('customer_ID').to_csv('submission.csv')

