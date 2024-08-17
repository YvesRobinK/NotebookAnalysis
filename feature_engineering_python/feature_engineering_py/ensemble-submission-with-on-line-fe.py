#!/usr/bin/env python
# coding: utf-8

# # Model Submission notebook
# 
# This notebook use an online feature engineering framework to submit previoulsy calibrated model. 
# 
# For the feature engineering part see here: https://www.kaggle.com/lucasmorin/on-line-feature-engineering
# 
# For the model calibration part see here: https://www.kaggle.com/lucasmorin/online-fe-lgbm-feval-importances

# In[1]:


import numpy as np
import pandas as pd

import gresearch_crypto

from tqdm import tqdm
import os
import gc
import pickle

import time
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb

import collections
from collections import deque

seed = 2021
DEBUG = False

def timestamp_to_date(timestamp):
    return(datetime.fromtimestamp(timestamp))

env = gresearch_crypto.make_env()

iter_test = env.iter_test()


# # load lgbm models

# In[2]:


model_lgbm = pickle.load(open('../input/k/lucasmorin/online-fe-lgbm-feval-importances/lgbm_models.pkl', 'rb'))
ES_it = pickle.load(open('../input/k/lucasmorin/online-fe-lgbm-feval-importances/ES_it.pkl', 'rb'))


# In[3]:


model_lgbm 


# # Running Mean (hidden)

# In[4]:


class RunningMean:
    def __init__(self, WIN_SIZE=20, n_size = 1):
        self.n = 0
        self.mean = np.zeros(n_size)
        self.cum_sum = 0
        self.past_value = 0
        self.WIN_SIZE = WIN_SIZE
        self.windows = collections.deque(maxlen=WIN_SIZE+1)
        
    def clear(self):
        self.n = 0
        self.windows.clear()

    def push(self, x):
        #currently fillna with past value, might want to change that
        x = fillna_npwhere(x, self.past_value)
        self.past_value = x
        
        self.windows.append(x)
        self.cum_sum += x
        
        if self.n < self.WIN_SIZE:
            self.n += 1
            self.mean = self.cum_sum / float(self.n)
            
        else:
            self.cum_sum -= self.windows.popleft()
            self.mean = self.cum_sum / float(self.WIN_SIZE)

    def get_mean(self):
        return self.mean if self.n else np.zeros(n_size)

    def __str__(self):
        return "Current window values: {}".format(list(self.windows))

# Temporary removing njit as it cause many bugs down the line
# Problems mainly due to data types, I have to find where I need to constraint types so as not to make njit angry
#@njit
def fillna_npwhere(array, values):
    if np.isnan(array.sum()):
        array = np.where(np.isnan(array), values, array)
    return array


# In[5]:


get_ipython().run_cell_magic('time', '', "\n#not building the weights each loops\nasset_details = pd.read_csv('../input/g-research-crypto-forecasting/asset_details.csv')\ndict_weights = {}\n\nfor i in range(asset_details.shape[0]):\n    dict_weights[asset_details.iloc[i,0]] = asset_details.iloc[i,1]\nweigths = np.array([dict_weights[i] for i in range(14)])\n\n# only needed when saving ?\ndtype={'Asset_ID': 'int8', 'Count': 'int32', 'row_id': 'int32', 'Count': 'int32',\n       'Open': 'float32', 'High': 'float32', 'Low': 'float32', 'Close': 'float32',\n       'Volume': 'float32', 'VWAP': 'float32'}\n#test_df = test_df.astype(dtype)\n\n#refactoring functions:\n\ndef Clean_df(x):\n    Asset_ID = x[:,1]\n    timestamp = x[0,0]\n\n    if len(Asset_ID)<14:\n        missing_ID = [i for i in range(14) if i not in Asset_ID]\n        for i in missing_ID:\n            row = np.array((timestamp,i,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan))\n            x = np.concatenate((x,np.expand_dims(row,axis=0)))\n\n    x = x[np.argsort(x[:,1])]\n    return (x[:,i] for i in range(x.shape[1]))\n\ndef Base_Feature_fn(timestamp,Asset_ID,Count,O,H,L,C,Volume,VWAP):\n\n    VWAP = np.where(np.isinf(VWAP),(C+O)/2,VWAP)\n    base = C\n    O = O/base\n    H = H/base\n    L = L/base\n    C = C/base\n    VWAP = VWAP/base\n    Price = base\n\n    Dollars = Volume * Price\n    Volume_per_trade = Volume/Count\n    Dollars_per_trade = Dollars/Count\n\n    log_ret = np.log(C/O)\n    GK_vol = (1 / 2 * np.log(H/L) ** 2 - (2 * np.log(2) - 1) * np.log(C/O) ** 2)\n    RS_vol = np.log(H/C)*np.log(H/O) + np.log(L/C)*np.log(L/O)\n\n    return(np.transpose(np.array([Count,O,H,L,C,Price,Volume,VWAP,Dollars,Volume_per_trade,Dollars_per_trade,log_ret,GK_vol,RS_vol])))\n\ndef Time_Feature_fn(timestamp):\n    \n    sin_month = (np.sin(2 * np.pi * timestamp.month/12))\n    cos_month = (np.cos(2 * np.pi * timestamp.month/12))\n    sin_day = (np.sin(2 * np.pi * timestamp.day/31))\n    cos_day = (np.cos(2 * np.pi * timestamp.day/31))\n    sin_hour = (np.sin(2 * np.pi * timestamp.hour/24))\n    cos_hour = (np.cos(2 * np.pi * timestamp.hour/24))\n    sin_minute = (np.sin(2 * np.pi * timestamp.minute/60))\n    cos_minute = (np.cos(2 * np.pi * timestamp.minute/60))\n\n    return(np.array((sin_month,cos_month,sin_day,cos_day,sin_hour,cos_hour,sin_minute,cos_minute)))\n\nMA_lags = [2,5,15,30,60,120,300,1800,3750,10*24*60,30*24*60]\n\n# #instantiation Moving average features dict\n# dict_RM = {}\n# dict_RM_M = {}\n\n# for lag in MA_lags:\n#     dict_RM[lag] = RunningMean(lag)\n#     dict_RM_M[lag] = RunningMean(lag)\n    \n\nbeta_lags = [60,300,1800,3750,10*24*60,30*24*60]\n\n# #instantiation dict betas\n# dict_MM = {}\n# dict_Mr = {}\n\ndict_RM = pickle.load(open('../input/on-line-feature-engineering/dict_RM_4.pkl', 'rb'))\ndict_RM_M = pickle.load(open('../input/on-line-feature-engineering/dict_RM_M_4.pkl', 'rb'))\ndict_MM = pickle.load(open('../input/on-line-feature-engineering/dict_MM_4.pkl', 'rb'))\ndict_Mr = pickle.load(open('../input/on-line-feature-engineering/dict_MR_4.pkl', 'rb'))\n\n\n# for lag in beta_lags:\n#     dict_MM[lag] = RunningMean(lag)\n#     dict_Mr[lag] = RunningMean(lag)\n\nfor (test_df, sample_prediction_df) in iter_test:\n    \n    timestamp,Asset_ID,Count,O,H,L,C,Volume,VWAP,row_id = Clean_df(test_df.values)\n    \n    # np.array([Count,O,H,L,C,Price,Volume,VWAP,Dollars,Volume_per_trade,Dollars_per_trade,log_ret,GK_vol,RS_vol])\n    Features = Base_Feature_fn(timestamp,Asset_ID,Count,O,H,L,C,Volume,VWAP)\n    \n    #removing wieghts when data is missing so that they don't appears in market\n    weigths = np.where(np.isnan(O),O,weigths)\n    Market_Features = np.nansum(Features*np.expand_dims(weigths,axis=1)/np.nansum(weigths),axis=0)\n    #Market_Features = np.tile(Market_Features,(14,1))\n    \n    #np.array((sin_month,cos_month,sin_day,cos_day,sin_hour,cos_hour,sin_minute,cos_minute))\n    timestamp = timestamp_to_date(timestamp[0])\n    Time_Features = Time_Feature_fn(timestamp)\n    #Time_Features = np.tile(Time_Features,(14,1))\n    \n    MA_Features = []\n    MA_Features_M  = [] \n    \n    for lag in MA_lags:\n        dict_RM[lag].push(Features)\n        dict_RM_M[lag].push(Market_Features)\n        \n        MA_Features.append(dict_RM[lag].get_mean())\n        MA_Features_M.append(dict_RM_M[lag].get_mean())\n        \n    MA_Features = np.concatenate(MA_Features,axis=1)\n    MA_Features_M = np.concatenate(MA_Features_M)\n    #MA_Features_M = np.tile(MA_Features_M,(14,1))\n    \n    betas = []\n    \n    for lag in beta_lags:\n        dict_MM[lag].push(Market_Features[11]**2)\n        dict_Mr[lag].push(Market_Features[11]*Features[11])\n        betas.append(np.expand_dims(dict_Mr[lag].get_mean()/dict_MM[lag].get_mean(),axis=1))\n        \n    betas = np.concatenate(betas,axis=1)\n    \n    values = np.concatenate((np.expand_dims(Asset_ID,axis=1), Features,np.tile(Market_Features,(14,1)),np.tile(Time_Features,(14,1)),MA_Features,np.tile(MA_Features_M,(14,1)),betas),axis=1)\n    \n    \n    #preds = model_lgbm[0].predict(values)\n    \n    preds = np.median(np.array([model_lgbm[str(i)+'-'+str(j)].predict(values, num_iteration = ES_it[str(i)+'-'+str(j)]) for i in range(5) for j in range(5)]),axis=0)\n    \n    sample_prediction_df['Target'] = [preds[(row_id == rid)][0] for rid in sample_prediction_df.row_id.values]\n    env.predict(sample_prediction_df)\n")

