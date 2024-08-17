#!/usr/bin/env python
# coding: utf-8

# # Realized volatilities aggregation functions
# 
# Some realized volatility aggregation functions. The functions that currently do not appears in other notebooks mostly come from this - well-documented - Git-Hub repo: (https://github.com/BayerSe/RealizedQuantities/blob/master/main.py). I added the metrics that are discusses here(https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/267096) and some idea from personnal experience. The idea is to study those metrics regarding the target, the relative error of the naive baseline prediction and the variation of the metrics (in the last 200 seconds) v.s. the target.
# 
# # Other Feature Engineering Notebooks: 
# 
# This notebook is part of a serie on basic Feature Engineering / visual variable selection notebooks:
# 
# 1) Base Features: https://www.kaggle.com/lucasmorin/feature-engineering-1-base-features
# 
# 2) Aggregation Functions: https://www.kaggle.com/lucasmorin/feature-engineering-2-aggregation-functions
# 
# 3) RV aggregation: https://www.kaggle.com/lucasmorin/feature-engineering-3-rv-aggregation/

# In[1]:


from builtins import range
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import seaborn as sns
import numpy as np
import pandas as pd
import stumpy
from IPython.core.display import display, HTML
import glob
import os
import gc
from joblib import Parallel, delayed


# In[2]:


path_submissions = '/'
target_name = 'target'
scores_folds = {}


# # Tools

# In[3]:


def calc_wap(df):
    wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1'])/(df['bid_size1'] + df['ask_size1'])
    return wap

def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff()

def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return**2))


# In[4]:


book_example = pd.read_parquet('../input/optiver-realized-volatility-prediction/book_train.parquet/stock_id=0')


# In[5]:


book_example = pd.read_parquet('../input/optiver-realized-volatility-prediction/book_train.parquet/stock_id=0')
trade_example =  pd.read_parquet('../input/optiver-realized-volatility-prediction/trade_train.parquet/stock_id=0')

stock_id = '0'
time_id = book_example.time_id.unique()

book_example = book_example[book_example['time_id'].isin(time_id)]
book_example.loc[:,'stock_id'] = stock_id
trade_example = trade_example[trade_example['time_id'].isin(time_id)]
trade_example.loc[:,'stock_id'] = stock_id

book_example['wap'] = calc_wap(book_example)

#book_example.groupby('time_id', as_index=False).apply(lambda x: x.reset_index())['wap'].unstack(level=0).plot(legend=None)

book_example.loc[:,'log_return'] = log_return(book_example['wap'])
book_example = book_example[~book_example['log_return'].isnull()]

book_example = book_example.merge(trade_example, on=['seconds_in_bucket','time_id'],how='left', suffixes=('', '_y'))
book_example = book_example.loc[:, ~book_example.columns.str.endswith('_y')]


# In[6]:


df = book_example


# In[7]:


rv = pd.DataFrame(book_example[['log_return','time_id']].groupby(['time_id']).agg(realized_volatility)).reset_index()

train = pd.read_csv('../input/optiver-realized-volatility-prediction/train.csv', dtype = {'stock_id': np.int32, 'time_id': np.int32, 'target': np.float64})
train.head()

train_0 = train[train['stock_id']==0]
df_rv_train = train_0.merge(rv, on = ['time_id'], how = 'right')

df_rv_train['error'] = (df_rv_train['target'] - df_rv_train['log_return'])
df_rv_train['percentage_error'] = (df_rv_train['target'] - df_rv_train['log_return'])/df_rv_train['target']


# # realised volatilities metrics

# In[8]:


from scipy import stats
from scipy.special import gamma

# code from : https://github.com/BayerSe/RealizedQuantities/blob/master/main.py
# do we care about constants ? 

trading_seconds = 600
avg_sampling_frequency = 1
original_sampling_frequency = 1
M = trading_seconds / original_sampling_frequency


mu_1 = np.sqrt((2 / np.pi))
mu_43 = 2 ** (2 / 3) * gamma(7 / 6) * gamma(1 / 2) ** (-1)

def log_return(series_log_return):
    series_log_return =  series_log_return[1:]
    return np.log(series_log_return).diff()

#Realized Variance (Andersen and Bollerslev, 1998) - realized vol without sqrt
def realized_variance(series_log_return):
    series_log_return =  series_log_return[1:]
    return np.sum(series_log_return**2)

# Realized absolute variation (Forsberg and Ghysels, 2007)
#rav = mu_1 ** (-1) * M ** (-.5) * realized_quantity(lambda x: x.abs().sum())
def realized_aboslute_variation(series_log_return):
    series_log_return =  series_log_return[1:]
    return np.sum(np.abs(series_log_return))

# Realized bipower variation (Barndorff-Nielsen and Shephard; 2004, 2006)
#bv = mu_1 ** (-2) * realized_quantity(lambda x: (x.abs() * x.shift(1).abs()).sum())
def realized_bipower_variation(series_log_return):
    series_log_return =  series_log_return[1:]
    return np.sum(np.abs(series_log_return)*np.abs(series_log_return.shift(1)))

# Standardized tri-power quarticity (see e.g. Forsberg & Ghysels, 2007)
#tq = M * mu_43 ** (-3) * realized_quantity(lambda x: (x.abs() ** (4 / 3) * x.shift(1).abs() ** (4 / 3) * x.shift(2).abs() ** (4 / 3)).sum())
def realized_tri_power_quarticity(series_log_return):
    series_log_return =  series_log_return[1:]
    return np.sum(np.abs(series_log_return) ** (4 / 3) * np.abs(series_log_return.shift(1)) ** (4 / 3) * np.abs(series_log_return).shift(2)** (4 / 3))

# simple jump test
def is_jumping(series_log_return):
    series_log_return =  series_log_return[1:]
    rv = realized_variance(series_log_return)
    bv = realized_aboslute_variation(series_log_return)
    return np.max(rv-bv,0)

# Statistical Jump test by Huang and Tauchen (2005) - see if the first test is usefull
#j = (np.log(rv) - np.log(bv)) / ((mu_1 ** -4 + 2 * mu_1 ** -2 - 5) / (M * tq * bv ** -2)) ** 0.5
#jump = j.abs() >= stats.norm.ppf(0.999)

# Separate continuous and discontinuous parts of the quadratic variation
#iv = pd.Series(0, index=index)
#iv[jump] = bv[jump] ** 0.5
#iv[~jump] = rv[~jump] ** 0.5

#jv = pd.Series(0, index=index)
#jv[jump] = rv[jump] ** 0.5 - bv[jump] ** 0.5
#jv[jv < 0] = 0

# Realized Semivariance (Barndorff-Nielsen, Kinnebrock and Shephard, 2010)
def realized_variance_m(series_log_return):
    series_log_return =  series_log_return[1:]
    return np.sum(series_log_return**2 * (series_log_return < 0))

def realized_variance_p(series_log_return):
    series_log_return =  series_log_return[1:]
    return np.sum(series_log_return**2 * (series_log_return > 0))

# Signed jump variation (Patton and Sheppard, 2015)
def Signed_jump_variation(series_log_return):
    series_log_return =  series_log_return[1:]
    rv_p = realized_variance_p(series_log_return)
    rv_m = realized_variance_m(series_log_return)
    return rv_p - rv_m

def Signed_jump_variation_p(series_log_return):
    series_log_return =  series_log_return[1:]
    sjv = Signed_jump_variation(series_log_return)
    return sjv * (sjv > 0)

def Signed_jump_variation_m(series_log_return):
    series_log_return =  series_log_return[1:]
    sjv = Signed_jump_variation(series_log_return)
    return sjv * (sjv < 0)

# Realized Skewness and Kurtosis  (see, e.g. Amaya, Christoffersen, Jacobs and Vasquez, 2015)
#rm3 = realized_quantity(lambda x: (x ** 3).sum())
#rm4 = realized_quantity(lambda x: (x ** 4).sum())
#rs = np.sqrt(M) * rm3 / rv ** (3 / 2)
#rk = M * rm4 / rv ** 2

def realized_skewness(series_log_return):
    series_log_return =  series_log_return[1:]
    return np.sum(series_log_return**3)

def realized_kurtosis(series_log_return):
    series_log_return =  series_log_return[1:]
    return np.sum(series_log_return**4)

#shared in discussion

def realized_quarticity(series_log_return):
    series_log_return =  series_log_return[1:]
    return np.sum(series_log_return**4)*series_log_return.shape[0]/3

def realized_quadpower_quarticity(series_log_return):
    series_log_return =  series_log_return[1:]
    series = series_log_return.rolling(window=4).apply(np.product, raw=True)
    return (np.sum(series_log_return) * series_log_return.shape[0] * (np.pi**2))/4

def realized_1(series_log_return):
    series_log_return =  series_log_return[1:]
    return np.sqrt(np.sum(series_log_return**4)/(6*np.sum(series_log_return**2)))

def realized_2(series_log_return):
    series_log_return =  series_log_return[1:]
    return np.sqrt(((np.pi**2)*np.sum(series_log_return.rolling(window=4).apply(np.product, raw=True)))/(8*np.sum(series_log_return**2)))


# In[9]:


all_vol_functions = [realized_variance,realized_aboslute_variation,realized_bipower_variation,realized_tri_power_quarticity,is_jumping,realized_variance_m,realized_variance_p,Signed_jump_variation,Signed_jump_variation_p,Signed_jump_variation_m,realized_skewness,realized_kurtosis,realized_quarticity,realized_quadpower_quarticity,realized_1, realized_2]


# In[10]:


create_feature_dict_vol = {
        'log_return': all_vol_functions,
    }


# In[11]:


get_ipython().run_cell_magic('time', '', "\ndf_train_stock_0_vol = df.groupby('time_id').agg(create_feature_dict_vol)\ndf_train_stock_0_vol.columns = ['_'.join(col) for col in df_train_stock_0_vol.columns]\n\ndf_train_stock_0_400_vol = df[df['seconds_in_bucket'] >= 400].groupby('time_id').agg(create_feature_dict_vol)\ndf_train_stock_0_400_vol.columns = ['_'.join(col) for col in df_train_stock_0_400_vol.columns]\n\ndf_train_stock_0_vol_diff = df_train_stock_0_400_vol - df_train_stock_0_vol\n\ndf_train_stock_0_feat_na0_vol = df_train_stock_0_vol_diff.fillna(0)\n")


# In[12]:


import random

sns.set(rc={'figure.figsize':(24,8)})
sns.set_style(style='white')

columns = [columns for columns in df_train_stock_0_vol.columns if columns not in ['time_id','stock_id','seconds_in_bucket']]

for col in columns:
    color = (random.random(), random.random(), random.random())
    
    fig, axs = plt.subplots(ncols=3)
    
    sns.regplot(x=df_train_stock_0_vol[col], y=df_rv_train['target'], color=color, order = 2, line_kws={"color": 'black'}, ax=axs[0]).set(ylim=(0, None),title= 'Variable v.s. target')
    sns.regplot(x=df_train_stock_0_vol[col], y=df_rv_train['percentage_error'], color=color, order = 2, line_kws={"color": 'black'}, ax=axs[1]).set(title= 'Variable v.s. percenatge error')
    sns.regplot(x=df_train_stock_0_vol_diff[col], y=df_rv_train['target'], color=color, order = 2, line_kws={"color": 'black'}, ax=axs[2]).set(ylim=(0, None),title= 'Variation v.s target')
    
    fig.suptitle(col+' v.s. target',size=30) 
    
    plt.show()

