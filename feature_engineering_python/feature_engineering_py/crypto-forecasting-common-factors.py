#!/usr/bin/env python
# coding: utf-8

# # Crypto Forecasting - Feature engineering

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

DEBUG = False


# ### Training data is in the competition dataset as usual

# In[2]:


nrows = 100000 if DEBUG else None

dtype={'Asset_ID': 'int8', 'Count': 'int32', 'row_id': 'int32', 'Count': 'int32',
       'Open': 'float32', 'High': 'float32', 'Low': 'float32', 'Close': 'float32',
       'Volume': 'float32', 'VWAP': 'float32'}

train_df = pd.read_csv('../input/g-research-crypto-forecasting/train.csv', low_memory=False, dtype=dtype, nrows=nrows)
asset_details = pd.read_csv('../input/g-research-crypto-forecasting/asset_details.csv')

#create dictionnary of weights
dict_weights = {}
for i in range(asset_details.shape[0]):
    dict_weights[asset_details.iloc[i,0]] = asset_details.iloc[i,1]

# remove rows with missing targets - DO THIS AT THE END
# train_df = train_df[~train_df.Target.isna()]

# replace infinite VWAP with close price
train_df.VWAP = np.where(np.isinf(train_df.VWAP),train_df.Close,train_df.VWAP)

#filter to avoid time leakage with the data 
filter_leakage = pd.to_datetime(train_df['timestamp'], unit='s') < '2021-06-01 00:00:00'
train_df = train_df[filter_leakage]


# In[3]:


train_df[train_df.Asset_ID == 2].head()


# In[4]:


ref_col = 'Close'


# In[5]:


#Standardise prices

def standardise_prices(df,cols=['Open','High','Low','Close','VWAP'],by='Close'):
    base = train_df[by].copy()
    df['Open'] = train_df['Open'] / base
    df['High'] = train_df['High'] / base
    df['Low'] = train_df['Low'] /  base
    df['Close'] = train_df['Close'] / base
    df['VWAP'] = train_df['VWAP'] / base
    df['Price'] = base
    return df

def calc_dollar_features(df, by='Price'):
    train_df['Volume_dollar'] = train_df['Volume']*train_df[by]
    train_df['volume_per_trade'] = train_df['Volume']/train_df['Count']
    train_df['dollar_per_trade'] = train_df['Volume_dollar']/train_df['Count']
    return df

train_df['weights'] = train_df.Asset_ID.map(dict_weights).astype('float32')
train_df = standardise_prices(train_df)
train_df = calc_dollar_features(train_df)

# log returns and estimated volatilities
train_df['log_ret'] = np.log(train_df.Close/train_df.Open)
train_df['GK_vol'] = (1 / 2 * np.log(train_df.High / train_df.Low) ** 2 - \
    (2 * np.log(2) - 1) * np.log(train_df.Close / train_df.Open) ** 2).astype('float32')
train_df['RS_vol'] = np.log(train_df.High/train_df.Close)*np.log(train_df.High/train_df.Open) + \
    np.log(train_df.Low/train_df.Close)*np.log(train_df.Low/train_df.Open)


# # Market Aggregation
# 
# code inspired from Slawek Biel work in optiver competition.

# In[6]:


get_ipython().run_cell_magic('time', '', "\nfeatures_to_aggregate = ['Count','Open','High','Low','Close','Price','Volume','VWAP','Target','Volume_dollar','volume_per_trade','dollar_per_trade','log_ret','GK_vol','RS_vol']\n\nt, w, A_id = (train_df[col].values for col in ['timestamp','weights','Asset_ID'])\nids, index = np.unique(t, return_index=True)\n\nValues = train_df[features_to_aggregate].values\nsplits = np.split(Values, index[1:])\nsplits_w = np.split(w, index[1:])\nsplits_A_id = np.split(A_id, index[1:])\n\nout = []\n\nfor time_id, x, w, A_id in zip(ids.tolist(), splits, splits_w, splits_A_id):\n    outputs = np.float32(np.sum((x.T*w),axis=1)/sum(w))\n    outputs = np.tile(outputs, (len(w), 1))\n    out.append(outputs)\n    \nout = np.concatenate(out,axis=0)\n")


# In[7]:


train_df[[s+'_M' for s in features_to_aggregate]] = out

del out, Values
gc.collect()


# In[8]:


train_df = train_df.drop([ref_col,ref_col+'_M'],axis=1)


# In[9]:


train_df.head()


# # time encoding

# In[10]:


def timestamp_to_date(timestamp):
    return(datetime.fromtimestamp(timestamp))

ts = train_df.timestamp
ts = ts.apply(timestamp_to_date)


# In[11]:


ts


# In[12]:


train_df['sin_month'] = (np.sin(2 * np.pi * ts.dt.month/12)).astype('float32')
train_df['cos_month'] = (np.cos(2 * np.pi * ts.dt.month/12)).astype('float32')
train_df['sin_day'] = (np.sin(2 * np.pi * ts.dt.day/31)).astype('float32')
train_df['cos_day'] = (np.cos(2 * np.pi * ts.dt.day/31)).astype('float32')
train_df['sin_hour'] = (np.sin(2 * np.pi * ts.dt.hour/24)).astype('float32')
train_df['cos_hour'] = (np.cos(2 * np.pi * ts.dt.hour/24)).astype('float32')
train_df['sin_minute'] = (np.sin(2 * np.pi * ts.dt.minute/60)).astype('float32')
train_df['cos_minute'] = (np.cos(2 * np.pi * ts.dt.minute/60)).astype('float32')


# # Cut data set in five
# 
# First step to build independant folds. the idea is to end with something like that:

# In[13]:


# Generate the class/group data

time_ids = train_df.timestamp.unique()

n_fold = 5
splits = 0.6
ntimes = len(time_ids)

embargo_train_test = 60*24*30
embargo_fold = 60*24*30

time_per_fold = (ntimes - 5*embargo_train_test - 5*embargo_fold)/5
train_len = splits*time_per_fold 
test_len = (1-splits)*time_per_fold

fold_start = [np.int(i*(len(time_ids)+1)/5) for i in range(6)]

for i in range(n_fold):
    time_folds = time_ids[fold_start[i]:fold_start[i+1]-1]
    df_fold = train_df[train_df.timestamp.isin(time_folds)]
    df_fold.to_parquet('df_fold_'+str(i)+'.parquet')
    
del train_df


# In[14]:


time_folds


# In[15]:


gc.collect()


# # lagged Features

# In[16]:


get_ipython().run_cell_magic('time', '', "\nfeatures_to_lag = ['Price','Volume','VWAP','log_ret','RS_vol']\nlags = [2,5,15,30,60,120,300,1800,3750,10*24*60,30*24*60]\n\nfor fold in range(n_fold):\n    print('fold:'+str(fold))\n    df_fold = pd.read_parquet('df_fold_'+str(fold)+'.parquet')\n    \n    tmp = pd.DataFrame()\n    \n    for l in lags:\n        #print('lag:'+str(l))\n        tmp2 = df_fold[features_to_lag+['Asset_ID']].groupby('Asset_ID').transform(lambda s: s.rolling(l, min_periods=1).mean())\n        tmp2.columns = [str(c)+'_l_'+str(l) for c in tmp2.columns]\n        tmp = pd.concat([tmp,tmp2],axis=1)\n        \n    tmp.astype('float32').to_parquet('df_fold_'+str(fold)+'_lag.parquet')\n")


# In[17]:


get_ipython().run_cell_magic('time', '', "\nfeatures_to_lag = ['Price_M','Volume_M','VWAP_M','log_ret_M','RS_vol_M']\nlags = [2,5,15,30,60,120,300,1800,3750,10*24*60,30*24*60]\n\nfor fold in range(n_fold):\n    print('fold:'+str(fold))\n    df_fold = pd.read_parquet('df_fold_'+str(fold)+'.parquet')\n    \n    tmp = pd.DataFrame()\n    \n    for l in lags:\n        #print('lag:'+str(l))\n        tmp2 = df_fold[features_to_lag+['Asset_ID']].groupby('Asset_ID').transform(lambda s: s.rolling(l, min_periods=1).mean())\n        tmp2.columns = [str(c)+'_l_'+str(l) for c in tmp2.columns]\n        tmp = pd.concat([tmp,tmp2],axis=1)\n        \n    tmp.astype('float32').to_parquet('df_fold_'+str(fold)+'_lag_M.parquet')\n")


# # Beta Features

# In[18]:


get_ipython().run_cell_magic('time', '', "\nfold = 0\nlags = [60,300,1800,3750,10*24*60,30*24*60]\n\nfor fold in range(n_fold):\n    print('fold:'+str(fold))\n    df_fold = pd.read_parquet('df_fold_'+str(fold)+'.parquet')\n    df_fold = df_fold[['Asset_ID','log_ret_M','log_ret']]\n    df_fold['log_ret_M2'] = df_fold['log_ret_M']**2\n    df_fold['log_ret_Mr'] = df_fold['log_ret_M']*df_fold['log_ret']\n    tmp = pd.DataFrame()\n    \n    for l in lags:\n        #print(l)\n        features_to_lag = ['log_ret_M2','log_ret_Mr']\n        #use min periods = l to match definition of target ?\n        tmp2 = df_fold[features_to_lag+['Asset_ID']].groupby('Asset_ID').transform(lambda s: s.rolling(l, min_periods=1).mean())\n        tmp2['beta'] = tmp2['log_ret_Mr'] / tmp2['log_ret_M2']\n        tmp2 = tmp2.drop(['log_ret_Mr','log_ret_M2'],axis=1)\n        \n        tmp2.columns = [str(c)+'_l_'+str(l) for c in tmp2.columns]\n        tmp = pd.concat([tmp,tmp2],axis=1)\n        tmp = tmp.loc[:,~tmp.columns.duplicated()]\n    \n    tmp.astype('float32').to_parquet('df_fold_'+str(fold)+'_beta.parquet')\n\ndel tmp2\ndel tmp\n")


# In[19]:


import sys

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                         key= lambda x: -x[1])[:20]:
    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


# In[20]:


del ts, df_fold, splits_w, splits_A_id, filter_leakage, ids, index, time_ids
gc.collect()


# In[21]:


for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                         key= lambda x: -x[1])[:20]:
    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


# # Merge data

# In[22]:


import os

dict_fold = {}

for fold in range(n_fold):
    print('fold:'+str(fold))
    
    df_fold = pd.read_parquet('df_fold_'+str(fold)+'.parquet')
    time_ids = df_fold.timestamp.unique()
    
    test_train_len = len(time_ids) - embargo_train_test - embargo_fold
    
    train_start = embargo_fold + 1
    train_end = embargo_fold + np.int(test_train_len*0.6) + 1
    test_start = embargo_fold + np.int(test_train_len*0.6) + embargo_train_test + 1
    test_end = len(df_fold.timestamp.unique())
    
    dict_fold['train_fold_'+str(fold)] = time_ids[train_start:train_end]
    dict_fold['test_fold_'+str(fold)] = time_ids[test_start:test_end]

del df_fold


# In[23]:


#scipy.stats.qmc.Halton


# In[24]:


import os

for fold in range(n_fold):
    
    df_train_fold = pd.DataFrame()
    df_test_fold = pd.DataFrame()
    
    df_read = pd.read_parquet("df_fold_"+str(fold)+'.parquet')
    
    ind_train = df_read.timestamp.isin(dict_fold['train_fold_'+str(fold)])
    ind_test = df_read.timestamp.isin(dict_fold['test_fold_'+str(fold)])
    
    df_train_fold = df_read[ind_train]
    df_test_fold = df_read[ind_test]
    
    for file in os.listdir('./'):
        if file == "df_fold_"+str(fold)+'.parquet':
            continue
            
        elif file.startswith("df_fold_"+str(fold)):
            print(file)
            df_read = pd.read_parquet(file)
            
            df_train_read = df_read[ind_train]
            df_test_read = df_read[ind_test]
            
            print(df_read.info())
            df_train_fold = pd.concat([df_train_fold,df_train_read],axis=1)
            df_train_fold = df_train_fold.loc[:,~df_train_fold.columns.duplicated()]
            
            df_test_fold = pd.concat([df_test_fold,df_test_read],axis=1)
            df_test_fold = df_test_fold.loc[:,~df_test_fold.columns.duplicated()]
            os.remove('./'+file)
            
    df_train_fold.to_parquet('train_fold_'+str(fold)+'.parquet')
    df_test_fold.to_parquet('test_fold_'+str(fold)+'.parquet')

