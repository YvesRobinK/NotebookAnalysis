#!/usr/bin/env python
# coding: utf-8

# # On-line Feature Engineering
# 
# The goal of this notebook is to provide a framework for online feature engineering that seems to be needed for this competition.
# 
# Note that:
# 
# 
# - The notebook mainly rely on my previous work in the JaneStreet competition. (See: https://www.kaggle.com/lucasmorin/running-algos-fe-for-fast-inference)
# - The notebook implement an online version of my previous exploratory notebook (See: https://www.kaggle.com/lucasmorin/crypto-forecasting-common-factors) and associated discussion (https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/288555).
# - The data is then used to calibrate a lgbm with custom feval and an importance framework (see: https://www.kaggle.com/lucasmorin/online-fe-lgbm-feval-importances). 
# - The online FE framework is then used for submission (See: https://www.kaggle.com/lucasmorin/ensemble-submission/edit/run/81219507)

# ## Features engineering techniques :
# 
# - [Start with the end](#Start) 
# - [Get Data](#Get_Data)
# - [Reorder Data](#Reorder_Data)
# - [Missing Values](#Missing_Values)
# - [Base Feature Engineering](#Base_FE)
# - [Market Features](#Market_Features)
# - [Time Features](#Time_Features)
# - [Running Moving Average](#RMA) (<- Magic)
# - [Moving Average Features](#MA_FE)
# - [Betas](#Betas)
# - [Putting it all together](#All) (<- All the features)
# - [Building Folds](#Folds)
# - [Running Variance](#Variance)
# - [Complete Feature Exploration](#FE_exploration) (New)

# In[1]:


import gresearch_crypto
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
from datetime import datetime

def timestamp_to_date(timestamp):
    return(datetime.fromtimestamp(timestamp))

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

env = gresearch_crypto.make_env()

iter_test = env.iter_test()

(test_df, sample_prediction_df) = next(iter_test)


# <a id='Start'></a>
# # Start with the end
# Looking at iterator submission data.

# In[2]:


test_df


# <a id='Get_Data'></a>
# # Get Data
# Change data from pandas to numpy.

# In[3]:


timestamp,Asset_ID,Count,O,H,L,C,Volume,VWAP,row_id = (test_df[col].values for col in ['timestamp','Asset_ID','Count','Open','High','Low','Close','Volume','VWAP','row_id'])


# <a id='Reorder_Data'></a>
# # Reorder data
# 
# Not sure this is entirely necessary depending on your model.
# I do that so that I can handle data of constant size.

# In[4]:


order = np.argsort(Asset_ID)
order


# <a id='Missing_Values'></a>
# # Missing value ?
# 
# Handling missing assets: adding rows with nan. 

# In[5]:


test_df_missing = test_df[test_df.Asset_ID.isin([1,2,3])]

missing_ID = [i for i in range(14) if i not in [1,2,3]]

val = test_df_missing.values

for i in missing_ID:
    val = np.append(val,np.expand_dims(np.array((timestamp[0],i,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)),axis=0),axis=0)
    
val


# <a id='Base_FE'></a>
# # Base Feature Enginerring

# In[6]:


asset_details = pd.read_csv('../input/g-research-crypto-forecasting/asset_details.csv')

#create dictionnary of weights
dict_weights = {}
for i in range(asset_details.shape[0]):
    dict_weights[asset_details.iloc[i,0]] = asset_details.iloc[i,1]
    
weigths = np.array([dict_weights[i] for i in range(14)])


# In[7]:


VWAP = np.where(np.isinf(VWAP),(C+O)/2,VWAP)


# In[8]:


dtype={'Asset_ID': 'int8', 'Count': 'int32', 'row_id': 'int32', 'Count': 'int32',
       'Open': 'float32', 'High': 'float32', 'Low': 'float32', 'Close': 'float32',
       'Volume': 'float32', 'VWAP': 'float32'}

test_df = test_df.astype(dtype)


# In[9]:


# Standardising Features
base = C
O = O/base
H = H/base
L = L/base
C = C/base
VWAP = VWAP/base
Price = base

# Using dollars 
Dollars = Volume * Price
Volume_per_trade = Volume/Count
Dollars_per_trade = Dollars/Count

# log returns and volatility estimators
log_ret = np.log(C/O)
GK_vol = (1 / 2 * np.log(H/L) ** 2 - (2 * np.log(2) - 1) * np.log(C / O) ** 2)
RS_vol = np.log(H/C)*np.log(H/O) + np.log(L/C)*np.log(L/O)


# <a id='Market_Features'></a>
# # Market Features

# In[10]:


#get back missing values in weights 
weigths = np.where(np.isnan(O),O,weigths)
Market_Features = np.nansum(np.array([Count,O,H,L,C,Price,Volume,VWAP,Dollars,Volume_per_trade,Dollars_per_trade,log_ret,GK_vol,RS_vol])*weigths/np.nansum(weigths),axis=1)


# <a id='Time_Features'></a>
# # Time Features

# In[11]:


timestamp = timestamp_to_date(timestamp[0])

sin_month = (np.sin(2 * np.pi * timestamp.month/12))
cos_month = (np.cos(2 * np.pi * timestamp.month/12))
sin_day = (np.sin(2 * np.pi * timestamp.day/31))
cos_day = (np.cos(2 * np.pi * timestamp.day/31))
sin_hour = (np.sin(2 * np.pi * timestamp.hour/24))
cos_hour = (np.cos(2 * np.pi * timestamp.hour/24))
sin_minute = (np.sin(2 * np.pi * timestamp.minute/60))
cos_minute = (np.cos(2 * np.pi * timestamp.minute/60))

time_features = np.array((sin_month,cos_month,sin_day,cos_day,sin_hour,cos_hour,sin_minute,cos_minute))
time_features


# <a id='RMA'></a>
# # Running Moving Average
# 
# Standard pandas moving average implementation would look like this:

# In[12]:


#rw = 10000
#train_data_rolled = train_data.rolling(window=rw).mean()


# But that wouldn't be practical. One idea is to get values in memory, then perform the mean. This would be rather inefficient too. 
# A better approach is to keep track of the cumulated sum. Only adding the last instance / removing the further one in time at each time step.

# In[13]:


import collections
from collections import deque

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


# <a id='MA_FE'></a>
# # Moving Average Features

# In[14]:


get_ipython().run_cell_magic('time', '', '\nMA_lags = [2,5,15,30,60,120,300,1800,3750,10*24*60,30*24*60]\n\n\nFeatures = np.transpose(np.array([Count,O,H,L,C,Price,Volume,VWAP,Dollars,Volume_per_trade,Dollars_per_trade,log_ret,GK_vol,RS_vol]))\n\n#instantiation Moving average features dict\ndict_RM = {}\ndict_RM_M = {}\nfor lag in MA_lags:\n    dict_RM[lag] = RunningMean(lag)\n    dict_RM_M[lag] = RunningMean(lag)\n\nfor i in tqdm(range(10000)):\n    \n    MA_Features = []\n    MA_Features_M  = [] \n    \n    for lag in MA_lags:\n        dict_RM[lag].push(Features)\n        dict_RM_M[lag].push(Market_Features)\n        \n        MA_Features.append(dict_RM[lag].get_mean())\n        MA_Features_M.append(dict_RM_M[lag].get_mean())\n        \n    MA_Features = np.concatenate(MA_Features,axis=1)\n    MA_Features_M = np.concatenate(MA_Features_M)\n')


# <a id='Betas'></a>
# # Betas
# 
# For a lack of a better implementation I start with just two memories. 

# In[15]:


get_ipython().run_cell_magic('time', '', '\nbeta_lags = [30,60,120,300,600,1800,3750,10*24*60,30*24*60]\n\n#instantiation dict betas\ndict_MM = {}\ndict_Mr = {}\nfor lag in beta_lags:\n    dict_MM[lag] = RunningMean(lag)\n    dict_Mr[lag] = RunningMean(lag)\n\nfor i in tqdm(range(10000)):\n    \n    betas = []\n    \n    for lag in beta_lags:\n        dict_MM[lag].push(Market_Features[11]**2)\n        dict_Mr[lag].push(Market_Features[11]*Features[11])\n        betas.append(np.expand_dims(dict_Mr[lag].get_mean()/dict_MM[lag].get_mean(),axis=1))\n        \n    betas = np.concatenate(betas,axis=1)\n')


# <a id='All'></a>
# # Putting it all together - cleaning and testing

# In[16]:


get_ipython().run_cell_magic('time', '', "\n#not building the weights each loops\nasset_details = pd.read_csv('../input/g-research-crypto-forecasting/asset_details.csv')\ndict_weights = {}\nfor i in range(asset_details.shape[0]):\n    dict_weights[asset_details.iloc[i,0]] = asset_details.iloc[i,1]\nweigths = np.array([dict_weights[i] for i in range(14)])\n\n# only needed when saving ?\ndtype={'Asset_ID': 'int8', 'Count': 'int32', 'row_id': 'int32', 'Count': 'int32',\n       'Open': 'float32', 'High': 'float32', 'Low': 'float32', 'Close': 'float32',\n       'Volume': 'float32', 'VWAP': 'float32'}\n#test_df = test_df.astype(dtype)\n\n#refactoring functions:\n\ndef Clean_df(x):\n    Asset_ID = x[:,1]\n    timestamp = x[0,0]\n    if len(Asset_ID)<14:\n        missing_ID = [i for i in range(14) if i not in Asset_ID]\n        for i in missing_ID:\n            row = np.array((timestamp,i,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan))\n            x = np.concatenate((x,np.expand_dims(row,axis=0)))\n    x = x[np.argsort(x[:,1])]\n    return (x[:,i] for i in range(x.shape[1]))\n\ndef Base_Feature_fn(timestamp,Asset_ID,Count,O,H,L,C,Volume,VWAP):\n    VWAP = np.where(np.isinf(VWAP),(C+O)/2,VWAP)\n    base = C\n    O = O/base\n    H = H/base\n    L = L/base\n    C = C/base\n    VWAP = VWAP/base\n    Price = base\n\n    Dollars = Volume * Price\n    Volume_per_trade = Volume/Count\n    Dollars_per_trade = Dollars/Count\n\n    log_ret = np.log(C/O)\n    log_ret_H = np.log(H/C)\n    log_ret_L = np.log(C/L)\n    log_ret_VWAP = np.log(C/VWAP)\n    \n    GK_vol = (1 / 2 * np.log(H/L) ** 2 - (2 * np.log(2) - 1) * np.log(C/O) ** 2)\n    RS_vol = np.log(H/C)*np.log(H/O) + np.log(L/C)*np.log(L/O)\n\n    #return(np.transpose(np.array([Count,O,H,L,C,Price,Volume,VWAP,Dollars,Volume_per_trade,Dollars_per_trade,log_ret,GK_vol,RS_vol])))\n    \n    log_Count,log_Volume,log_Dollars,log_Volume_per_trade,log_Dollars_per_trade = np.log([Count,Volume,Dollars,Volume_per_trade,Dollars_per_trade])\n\n    return(np.transpose(np.array([log_ret,log_ret_H,log_ret_L,log_ret_VWAP,GK_vol,RS_vol,log_Count,log_Volume,log_Dollars,log_Volume_per_trade,log_Dollars_per_trade])))\n\ndef Time_Feature_fn(timestamp):\n    \n    sin_month = (np.sin(2 * np.pi * timestamp.month/12))\n    cos_month = (np.cos(2 * np.pi * timestamp.month/12))\n    sin_day = (np.sin(2 * np.pi * timestamp.day/31))\n    cos_day = (np.cos(2 * np.pi * timestamp.day/31))\n    sin_hour = (np.sin(2 * np.pi * timestamp.hour/24))\n    cos_hour = (np.cos(2 * np.pi * timestamp.hour/24))\n    sin_minute = (np.sin(2 * np.pi * timestamp.minute/60))\n    cos_minute = (np.cos(2 * np.pi * timestamp.minute/60))\n\n    return(np.array((sin_month,cos_month,sin_day,cos_day,sin_hour,cos_hour,sin_minute,cos_minute)))\n\nMA_lags = [2,5,15,30,60,120,300,1800,3750,10*24*60,30*24*60]\n\n#instantiation Moving average features dict\ndict_RM = {}\ndict_RM_M = {}\n\nfor lag in MA_lags:\n    dict_RM[lag] = RunningMean(lag)\n    dict_RM_M[lag] = RunningMean(lag)\n    \nbeta_lags = [60,300,1800,3750,10*24*60,30*24*60]\n\n#instantiation dict betas\ndict_MM = {}\ndict_Mr = {}\nfor lag in beta_lags:\n    dict_MM[lag] = RunningMean(lag)\n    dict_Mr[lag] = RunningMean(lag)\n\nfor i in tqdm(range(10000)):\n    \n    timestamp,Asset_ID,Count,O,H,L,C,Volume,VWAP,row_id = Clean_df(test_df.values)\n    \n    # np.array([Count,O,H,L,C,Price,Volume,VWAP,Dollars,Volume_per_trade,Dollars_per_trade,log_ret,GK_vol,RS_vol])\n    Features = Base_Feature_fn(timestamp,Asset_ID,Count,O,H,L,C,Volume,VWAP)\n    \n    #removing wieghts when data is missing so that they don't appears in market\n    weigths = np.where(np.isnan(O),O,weigths)\n    Market_Features = np.nansum(Features*np.expand_dims(weigths,axis=1)/np.nansum(weigths),axis=0)\n    #Market_Features = np.tile(Market_Features,(14,1))\n    \n    #np.array((sin_month,cos_month,sin_day,cos_day,sin_hour,cos_hour,sin_minute,cos_minute))\n    timestamp = timestamp_to_date(timestamp[0])\n    Time_Features = Time_Feature_fn(timestamp)\n    #Time_Features = np.tile(Time_Features,(14,1))\n    \n    MA_Features = []\n    MA_Features_M  = [] \n    \n    for lag in MA_lags:\n        dict_RM[lag].push(Features)\n        dict_RM_M[lag].push(Market_Features)\n        \n        MA_Features.append(dict_RM[lag].get_mean())\n        MA_Features_M.append(dict_RM_M[lag].get_mean())\n        \n    MA_Features = np.concatenate(MA_Features,axis=1)\n    MA_Features_M = np.concatenate(MA_Features_M)\n    #MA_Features_M = np.tile(MA_Features_M,(14,1))\n    \n    betas = []\n    \n    for lag in beta_lags:\n        dict_MM[lag].push(Market_Features[0]**2)\n        dict_Mr[lag].push(Market_Features[0]*Features[:,0])\n        betas.append(np.expand_dims(dict_Mr[lag].get_mean()/dict_MM[lag].get_mean(),axis=1))\n        \n    betas = np.concatenate(betas,axis=1)\n    \n    #print(values)\n    #for data in [Features,np.tile(Market_Features,(14,1)),np.tile(Time_Features,(14,1)),MA_Features,np.tile(MA_Features_M,(14,1)),betas]:\n        #print(data.shape)\n    \n    values = np.concatenate((Features,np.tile(Market_Features,(14,1)),np.tile(Time_Features,(14,1)),MA_Features,np.tile(MA_Features_M,(14,1)),betas),axis=1)\n")


# In[17]:


Features_names = ['log_ret','log_ret_H','log_ret_L','log_ret_VWAP','GK_vol','RS_vol','log_Count','log_Volume','log_Dollars','log_Volume_per_trade','log_Dollars_per_trade']
Market_Features_names = [s+'_M' for s in Features_names]
Time_Features_names = ['sin_month','cos_month','sin_day','cos_day','sin_hour','cos_hour','sin_minute','cos_minute']
MA_Features_names = [s+'_'+str(lag) for lag in MA_lags for s in Features_names ]
MA_Features_M_names = [s+'_'+str(lag) for lag in MA_lags for s in Market_Features_names]
betas_names = ['betas_'+str(lag) for lag in beta_lags]

All_names = Features_names + Market_Features_names + Time_Features_names + MA_Features_names + MA_Features_M_names + betas_names
df_values = pd.DataFrame(values, columns = All_names)


# <a id='Folds'></a>
# # Creating Training Folds
# 
# For the design of the folds, see discussion here: https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/288555

# In[18]:


DEBUG  = False
nrows = 100000 if DEBUG else None

dtype={'Asset_ID': 'int8', 'Count': 'int32', 'row_id': 'int32', 'Count': 'int32',
       'Open': 'float32', 'High': 'float32', 'Low': 'float32', 'Close': 'float32',
       'Volume': 'float32', 'VWAP': 'float32'}

train_df = pd.read_csv('../input/g-research-crypto-forecasting/train.csv', low_memory=False, dtype=dtype, nrows=nrows)
asset_details = pd.read_csv('../input/g-research-crypto-forecasting/asset_details.csv')

#filter to avoid time leakage with the data 
filter_leakage = pd.to_datetime(train_df['timestamp'], unit='s') < '2021-06-13 00:00:00'
train_df = train_df[filter_leakage]


# In[19]:


# Generate the class/group data

import os
time_ids = train_df.timestamp.unique()

n_fold = 5
splits = 0.6
ntimes = len(time_ids)

embargo_train_test = 100 if DEBUG else 60*24*30
embargo_fold = 100 if DEBUG else 60*24*30

time_per_fold = (ntimes - 5*embargo_train_test - 5*embargo_fold)/5
train_len = splits*time_per_fold 
test_len = (1-splits)*time_per_fold

fold_start = [np.int(i*(len(time_ids)+1)/5) for i in range(6)]

for i in range(n_fold):
    time_folds = time_ids[fold_start[i]:fold_start[i+1]-1]
    df_fold = train_df[train_df.timestamp.isin(time_folds)]
    df_fold.to_parquet('df_fold_'+str(i)+'.parquet')
    
del train_df

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


# In[20]:


get_ipython().run_cell_magic('time', '', '\nimport os\nfrom random import random\n\nsampling = 0.05\n\nMA_lags = [2,5,15,30,60,120,300,1800,3750,2*3750,7*24*60]\nbeta_lags = [15,30,60,120,300,600,1800,3750,2*3750,7*24*60]\n\nFeatures_names = [\'log_ret\',\'log_ret_H\',\'log_ret_L\',\'log_ret_VWAP\',\'GK_vol\',\'RS_vol\',\'log_Count\',\'log_Volume\',\'log_Dollars\',\'log_Volume_per_trade\',\'log_Dollars_per_trade\']\nMarket_Features_names = [s+\'_M\' for s in Features_names]\nTime_Features_names = [\'sin_month\',\'cos_month\',\'sin_day\',\'cos_day\',\'sin_hour\',\'cos_hour\',\'sin_minute\',\'cos_minute\']\nMA_Features_names = [s+\'_\'+str(lag) for lag in MA_lags for s in Features_names ]\nMA_Features_M_names = [s+\'_\'+str(lag) for lag in MA_lags for s in Market_Features_names]\nbetas_names = [\'betas_\'+str(lag) for lag in beta_lags]\n\nAll_names = Features_names + Market_Features_names + Time_Features_names + MA_Features_names + MA_Features_M_names + betas_names\n#df_values = pd.DataFrame(values, columns = All_names)\n\nfor fold in range(n_fold):\n    \n    df_train_fold = pd.DataFrame()\n    df_test_fold = pd.DataFrame()\n    \n    df_read = pd.read_parquet("df_fold_"+str(fold)+\'.parquet\')\n    \n    #instantiation Moving average features dict\n    dict_RM = {}\n    dict_RM_M = {}\n\n    for lag in MA_lags:\n        dict_RM[lag] = RunningMean(lag)\n        dict_RM_M[lag] = RunningMean(lag)\n\n    #instantiation dict betas\n    dict_MM = {}\n    dict_Mr = {}\n    for lag in beta_lags:\n        dict_MM[lag] = RunningMean(lag)\n        dict_Mr[lag] = RunningMean(lag)\n\n    f = [\'timestamp\',\'Asset_ID\',\'Count\',\'Open\',\'High\',\'Low\',\'Close\',\'Volume\',\'VWAP\',\'Target\']\n    t = df_read[\'timestamp\'].values\n    ids, index = np.unique(t, return_index=True)\n\n    Values = df_read[f].values\n    splits = np.split(Values, index[1:])\n    out = []\n\n    for time_id, x in tqdm(zip(ids.tolist(), splits)):\n        #df = Clean_df(pd.DataFrame(x,columns=f))\n\n        #timestamp,Asset_ID,Count,O,H,L,C,Volume,VWAP,row_id = (test_df[col].values for col in [\'timestamp\',\'Asset_ID\',\'Count\',\'Open\',\'High\',\'Low\',\'Close\',\'Volume\',\'VWAP\',\'row_id\'])\n        timestamp,Asset_ID,Count,O,H,L,C,Volume, VWAP,Target = Clean_df(x)\n\n        # np.array([Count,O,H,L,C,Price,Volume,VWAP,Dollars,Volume_per_trade,Dollars_per_trade,log_ret,GK_vol,RS_vol])\n        Features = Base_Feature_fn(timestamp,Asset_ID,Count,O,H,L,C,Volume,VWAP)\n\n        #removing wieghts when data is missing so that they don\'t appears in market\n        weigths_curr = np.where(np.isnan(O),O,weigths)\n        Market_Features = np.nansum(Features*np.expand_dims(weigths_curr,axis=1)/np.nansum(weigths_curr),axis=0)\n        #Market_Features = np.tile(Market_Features,(14,1))\n\n        #np.array((sin_month,cos_month,sin_day,cos_day,sin_hour,cos_hour,sin_minute,cos_minute))\n        time = timestamp_to_date(timestamp[0])\n        Time_Features = Time_Feature_fn(time)\n        #Time_Features = np.tile(Time_Features,(14,1))\n\n        MA_Features = []\n        MA_Features_M  = [] \n\n        for lag in MA_lags:\n            dict_RM[lag].push(Features.copy())\n            dict_RM_M[lag].push(Market_Features.copy())\n\n            MA_Features.append(dict_RM[lag].get_mean())\n            MA_Features_M.append(dict_RM_M[lag].get_mean())\n        \n        #standardise w/ 3750 lag\n        ref = 3750\n        \n        for i in range(len(MA_lags)):\n            if MA_lags[i] == ref:\n                MA_ref = dict_RM[MA_lags[i]].get_mean().copy()\n                MA_M_ref = dict_RM_M[MA_lags[i]].get_mean().copy()\n        \n        \n                \n                \n        Features[:,-6:] = (Features[:,-6:] - MA_ref[:,-6:]).copy()\n        Market_Features[-6:] = (Market_Features[-6:] - MA_M_ref[-6:]).copy()\n                \n        for i in range(len(MA_lags)):\n            MA_Features[i][:,-6:] = (MA_Features[i][:,-6:] - MA_ref[:,-6:]).copy()\n            MA_Features_M[i][-6:] = (MA_Features_M[i][-6:] - MA_M_ref[-6:]).copy()\n\n        MA_Features_agg = np.concatenate(MA_Features,axis=1)\n        MA_Features_M_agg = np.concatenate(MA_Features_M)\n\n        betas = []\n\n        for lag in beta_lags:\n            dict_MM[lag].push(Market_Features[0]**2)\n            dict_Mr[lag].push(Market_Features[0]*Features[:,0])\n            betas.append(np.expand_dims(dict_Mr[lag].get_mean()/dict_MM[lag].get_mean(),axis=1))\n\n        betas = np.concatenate(betas,axis=1)\n        betas = np.nan_to_num(betas, nan=0., posinf=0., neginf=0.) \n\n        values = np.concatenate((Features,np.tile(Market_Features,(14,1)),np.tile(Time_Features,(14,1)),MA_Features_agg,np.tile(MA_Features_M_agg,(14,1)),betas, np.expand_dims(Target,axis=1)),axis=1)\n        \n        if random() < sampling:\n            out.append(np.concatenate((np.expand_dims(timestamp,axis=1),np.expand_dims(Asset_ID,axis=1),np.float32(values)),axis=1))\n    \n    df_out = pd.DataFrame(np.concatenate(out), columns = [\'timestamp\',\'Asset_ID\'] + All_names + [\'Target\']).astype({\'timestamp\': \'int64\',\'Asset_ID\': \'int64\'})\n    \n    df_out = df_out[~np.isnan(df_out.Target)]\n\n    ind_train = df_out.timestamp.isin(dict_fold[\'train_fold_\'+str(fold)])\n    ind_test = df_out.timestamp.isin(dict_fold[\'test_fold_\'+str(fold)])\n    \n    df_train_fold = df_out[ind_train]\n    df_test_fold = df_out[ind_test]\n    \n    df_train_fold.to_parquet(\'train_fold_\'+str(fold)+\'.parquet\')\n    df_test_fold.to_parquet(\'test_fold_\'+str(fold)+\'.parquet\')\n    \n    pd.DataFrame(df_train_fold.mean(),columns=[\'mean\']).to_parquet(\'mean_fold_\'+str(fold)+\'.parquet\')\n    pd.DataFrame(df_train_fold.std(),columns=[\'std\']).to_parquet(\'std_fold_\'+str(fold)+\'.parquet\')\n    \n')


# In[ ]:





# In[ ]:





# In[21]:


import pickle

pickle.dump(dict_RM, open('dict_RM_4.pkl', 'wb'))
pickle.dump(dict_RM_M, open('dict_RM_M_4.pkl', 'wb'))
pickle.dump(dict_MM, open('dict_MM_4.pkl', 'wb'))
pickle.dump(dict_Mr, open('dict_MR_4.pkl', 'wb'))


# <a id='Variance'></a>
# # Variance
# 
# For the moment I get volatility estimators from Garman-Klass estimation on OLHC data. In the future a better estimation might be needed. I share some code below for the second moment (adapted from same stack overflow post). Some code were added to deal with bug but it might not be entirely clean.

# In[22]:


from __future__ import division
import collections
import math

class RunningStats:
    def __init__(self, WIN_SIZE=20, n_size = 1):
        self.n = 0
        self.mean = 0
        self.run_var = 0
        self.n_size = n_size
        self.WIN_SIZE = WIN_SIZE
        self.past_value = 0
        self.windows = collections.deque(maxlen=WIN_SIZE+1)

    def clear(self):
        self.n = 0
        self.windows.clear()

    def push(self, x):
        
        x = fillna_npwhere(x, self.past_value)
        self.past_value = x

        self.windows.append(x)

        if self.n < self.WIN_SIZE:
            # Calculating first variance
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            self.run_var += delta * (x - self.mean)
        else:
            # Adjusting variance
            x_removed = self.windows.popleft()
            old_m = self.mean
            self.mean += (x - x_removed) / self.WIN_SIZE
            self.run_var += (x + x_removed - old_m - self.mean) * (x - x_removed)

    def get_mean(self):
        return self.mean if self.n else np.zeros(n_size)

    def get_var(self):
        return self.run_var / (self.n) if self.n > 1 else np.zeros(self.n_size)

    def get_std(self):
        return np.sqrt(self.get_var())

    def get_all(self):
        return list(self.windows)

    def __str__(self):
        return "Current window values: {}".format(list(self.windows))


# In[23]:


get_ipython().run_cell_magic('time', '', '\nfrom tqdm import tqdm\nimport random\n\nFeatures = np.array([Count,O,H,L,C,Price,Volume,VWAP,Dollars,Volume_per_trade,Dollars_per_trade,log_ret,GK_vol,RS_vol])\n\nMarket_Features\nlags = [2,5,15,30,60,120,300,1800,3750,10*24*60,30*24*60]\n\ndict_vol = {}\ndict_vol_M = {}\n\n#instantiation\nfor lag in lags:\n    dict_vol[lag] = RunningStats(lag)\n    dict_vol_M[lag] = RunningStats(lag)\n\nfor i in tqdm(range(10000)):\n    \n    vol = []\n    vol_M = []\n    \n    for lag in lags:\n        dict_vol[lag].push(Features[0]+0.001*np.array([random.random() for i in range(14)]))\n        dict_vol_M[lag].push(Market_Features[0]+0.005*np.array([random.random() for i in range(14)]))\n        \n        vol.append(dict_vol[lag].get_var())\n        vol_M.append(dict_vol_M[lag].get_var())\n')


# In[24]:


dict_vol_M[lag].get_std()


# Seems to work (better use a non constant input)

# <a id='FE_exploration'></a>
# # Complete Feature Exploration

# In[25]:


import matplotlib.pyplot as plt
for c in df_train_fold.columns:
    if c == 'Asset_ID':
        continue
    print(c)
    print(df_train_fold[c].describe())
    df_plot = df_train_fold[[c,'Asset_ID']].pivot(columns='Asset_ID')
    df_plot[c].plot(kind = 'hist', stacked=True, bins=100).set_xlim((np.min(df_plot[c].quantile(0.025)),np.max(df_plot[c].quantile(0.975))))
    #plt.hist(df_train_fold[c],bins=100)
    plt.show()


# In[ ]:




