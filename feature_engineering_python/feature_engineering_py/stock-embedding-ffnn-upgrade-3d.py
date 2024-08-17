#!/usr/bin/env python
# coding: utf-8

# <a class="anchor" id="0"></a>
# # [Optiver Realized Volatility Prediction](https://www.kaggle.com/c/optiver-realized-volatility-prediction)

# ### I use the notebook [Stock Embedding - FFNN - My features](https://www.kaggle.com/alexioslyon/stock-embedding-ffnn-my-features) from [alexioslyon](https://www.kaggle.com/alexioslyon) as a basis and tried to tune its various parameters. 

# # Acknowledgements
# 
# * [Stock Embedding - FFNN - My features](https://www.kaggle.com/alexioslyon/stock-embedding-ffnn-my-features) from @alexioslyon
# * [Stock Embedding - FFNN - My features](https://www.kaggle.com/tatudoug/stock-embedding-ffnn-my-features) from @tatudoug
# * [Stock Embedding - FFNN - features of the best lgbm](https://www.kaggle.com/tatudoug/stock-embedding-ffnn-features-of-the-best-lgbm) from @tatudoug
# * [NN Starter - Stock Embedding](https://www.kaggle.com/lucasmorin/tf-keras-nn-with-stock-embedding)
# * [Embedding Layers](https://www.kaggle.com/colinmorris/embedding-layers)
# * [Optiver Realized Volatility LGBM Baseline](https://www.kaggle.com/ragnar123/optiver-realized-volatility-lgbm-baseline)
# * tuning and visualization from [Higher LB score by tuning mloss - upgrade & visual](https://www.kaggle.com/vbmokin/higher-lb-score-by-tuning-mloss-upgrade-visual) and [MoA: Pytorch-RankGauss-PCA-NN upgrade & 3D visual](https://www.kaggle.com/vbmokin/moa-pytorch-rankgauss-pca-nn-upgrade-3d-visual)
# * [Data Science for tabular data: Advanced Techniques](https://www.kaggle.com/vbmokin/data-science-for-tabular-data-advanced-techniques)

# ## My upgrade:
# 
# Improved notebook structure.
# 
# Tuning and 3D visualization of prediction results is performed for different:
# 
# * Feature engineering
# * Learning rate
# * Number of epochs
# 
# FE: I tried to use 350 instead of 300 - it worsened the result to LB = 0.20209.

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
# 1. [Modeling and prediction](#5)
# 1. [Submission](#6)

# ## 1. Import libraries<a class="anchor" id="1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[1]:


from IPython.core.display import display, HTML

import glob
import os
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy.matlib

import plotly.express as px
import plotly.graph_objects as go

from joblib import Parallel, delayed

from sklearn import preprocessing, model_selection
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans

from numpy.random import seed
seed(42)

import tensorflow as tf
tf.random.set_seed(42)
from tensorflow import keras
from keras import backend as K
from keras.backend import sigmoid
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation

import warnings
warnings.filterwarnings('ignore')


# In[2]:


path_submissions = '/'

target_name = 'target'
scores_folds = {}


# ## 2. My upgrade <a class="anchor" id="2"></a>
# 
# [Back to Table of Contents](#0.1)

# ### 2.1. Commit now <a class="anchor" id="2.1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[3]:


# From the best version (commit) 5
learning_rate = 0.006
num_epochs = 200


# ### 2.2 Previous commits <a class="anchor" id="2.2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[4]:


commits_df = pd.DataFrame(columns = ['n_commit', 'learning_rate', 'num_epochs', 'FE', 'target4', 'target32_34', 'LB_score'])


# ### Commit 0 (parameters from [Stock Embedding - FFNN - My features](https://www.kaggle.com/alexioslyon/stock-embedding-ffnn-my-features), version 4)

# In[5]:


n=0
commits_df.loc[n, 'n_commit'] = 0                   # Number of version
commits_df.loc[n, 'learning_rate'] = 0.005          # Learning rate
commits_df.loc[n, 'num_epochs'] = 1000              # Number of epochs
commits_df.loc[n, 'FE'] = 0                         # Was there a replacement of 300 for 350?
commits_df.loc[n, 'target4'] = 0.000935             # Target 0-4
commits_df.loc[n, 'target32_34'] = 0.002423         # Target 0-32 & 0-34
commits_df.loc[n, 'LB_score'] = 0.20157             # LB score after submitting


# ### Commit 3

# In[6]:


n=1
commits_df.loc[n, 'n_commit'] = 3                   # Number of version
commits_df.loc[n, 'learning_rate'] = 0.005          # Learning rate
commits_df.loc[n, 'num_epochs'] = 1100              # Number of epochs (but all calculations 
                                                    # are completed much earlier)
commits_df.loc[n, 'FE'] = 1                         # Was there a replacement of 300 for 350?
commits_df.loc[n, 'target4'] = 0.001048             # Target 0-4
commits_df.loc[n, 'target32_34'] = 0.002394         # Target 0-32 & 0-34
commits_df.loc[n, 'LB_score'] = 0.20209             # LB score after submitting


# ### Commit 4

# In[7]:


n=2
commits_df.loc[n, 'n_commit'] = 4                   # Number of version
commits_df.loc[n, 'learning_rate'] = 0.004          # Learning rate
commits_df.loc[n, 'num_epochs'] = 100               # Number of epochs
commits_df.loc[n, 'FE'] = 0                         # Was there a replacement of 300 for 350?
commits_df.loc[n, 'target4'] = 0.000968             # Target 0-4
commits_df.loc[n, 'target32_34'] = 0.002465         # Target 0-32 & 0-34
commits_df.loc[n, 'LB_score'] = 0.20219             # LB score after submitting


# ### Commit 5

# In[8]:


n=3
commits_df.loc[n, 'n_commit'] = 5                   # Number of version
commits_df.loc[n, 'learning_rate'] = 0.006          # Learning rate
commits_df.loc[n, 'num_epochs'] = 200               # Number of epochs
commits_df.loc[n, 'FE'] = 0                         # Was there a replacement of 300 for 350?
commits_df.loc[n, 'target4'] = 0.000829             # Target 0-4
commits_df.loc[n, 'target32_34'] = 0.002325         # Target 0-32 & 0-34
commits_df.loc[n, 'LB_score'] = 0.20012             # LB score after submitting


# ### Commit 7

# In[9]:


n=4
commits_df.loc[n, 'n_commit'] = 7                   # Number of version
commits_df.loc[n, 'learning_rate'] = 0.007          # Learning rate
commits_df.loc[n, 'num_epochs'] = 200               # Number of epochs
commits_df.loc[n, 'FE'] = 0                         # Was there a replacement of 300 for 350?
commits_df.loc[n, 'target4'] = 0.001622             # Target 0-4
commits_df.loc[n, 'target32_34'] = 0.002353         # Target 0-32 & 0-34
commits_df.loc[n, 'LB_score'] = 0.20101             # LB score after submitting


# ### Commit 8

# In[10]:


n=5
commits_df.loc[n, 'n_commit'] = 8                   # Number of version
commits_df.loc[n, 'learning_rate'] = 0.0055         # Learning rate
commits_df.loc[n, 'num_epochs'] = 200               # Number of epochs
commits_df.loc[n, 'FE'] = 0                         # Was there a replacement of 300 for 350?
commits_df.loc[n, 'target4'] = 0.001225             # Target 0-4
commits_df.loc[n, 'target32_34'] = 0.002434         # Target 0-32 & 0-34
commits_df.loc[n, 'LB_score'] = 0.20147             # LB score after submitting


# ### Commit 9

# In[11]:


n=6
commits_df.loc[n, 'n_commit'] = 9                   # Number of version
commits_df.loc[n, 'learning_rate'] = 0.0065         # Learning rate
commits_df.loc[n, 'num_epochs'] = 200               # Number of epochs
commits_df.loc[n, 'FE'] = 0                         # Was there a replacement of 300 for 350?
commits_df.loc[n, 'target4'] = 0.001466             # Target 0-4
commits_df.loc[n, 'target32_34'] = 0.002362         # Target 0-32 & 0-34
commits_df.loc[n, 'LB_score'] = 0.20057             # LB score after submitting


# ### Commit 11

# In[12]:


n=7
commits_df.loc[n, 'n_commit'] = 11                  # Number of version
commits_df.loc[n, 'learning_rate'] = 0.0059         # Learning rate
commits_df.loc[n, 'num_epochs'] = 200               # Number of epochs
commits_df.loc[n, 'FE'] = 0                         # Was there a replacement of 300 for 350?
commits_df.loc[n, 'target4'] = 0.001019             # Target 0-4
commits_df.loc[n, 'target32_34'] = 0.002347         # Target 0-32 & 0-34
commits_df.loc[n, 'LB_score'] = 0.20161             # LB score after submitting


# ### Commit 12

# In[13]:


n=8
commits_df.loc[n, 'n_commit'] = 12                  # Number of version
commits_df.loc[n, 'learning_rate'] = 0.0061         # Learning rate
commits_df.loc[n, 'num_epochs'] = 200               # Number of epochs
commits_df.loc[n, 'FE'] = 0                         # Was there a replacement of 300 for 350?
commits_df.loc[n, 'target4'] = 0.001295             # Target 0-4
commits_df.loc[n, 'target32_34'] = 0.002450         # Target 0-32 & 0-34
commits_df.loc[n, 'LB_score'] = 0.20148             # LB score after submitting


# ### Commit 13

# In[14]:


n=9
commits_df.loc[n, 'n_commit'] = 13                  # Number of version
commits_df.loc[n, 'learning_rate'] = 0.00601        # Learning rate
commits_df.loc[n, 'num_epochs'] = 200               # Number of epochs
commits_df.loc[n, 'FE'] = 0                         # Was there a replacement of 300 for 350?
commits_df.loc[n, 'target4'] = 0.000979             # Target 0-4
commits_df.loc[n, 'target32_34'] = 0.002414         # Target 0-32 & 0-34
commits_df.loc[n, 'LB_score'] = 0.20119             # LB score after submitting


# ### Commit 14

# In[15]:


n=10
commits_df.loc[n, 'n_commit'] = 14                  # Number of version
commits_df.loc[n, 'learning_rate'] = 0.00599        # Learning rate
commits_df.loc[n, 'num_epochs'] = 200               # Number of epochs
commits_df.loc[n, 'FE'] = 0                         # Was there a replacement of 300 for 350?
commits_df.loc[n, 'target4'] = 0.001537             # Target 0-4
commits_df.loc[n, 'target32_34'] = 0.002432         # Target 0-32 & 0-34
commits_df.loc[n, 'LB_score'] = 0.20216             # LB score after submitting


# ### Commit 15

# In[16]:


n=11
commits_df.loc[n, 'n_commit'] = 15                  # Number of version
commits_df.loc[n, 'learning_rate'] = 0.0049         # Learning rate
commits_df.loc[n, 'num_epochs'] = 200               # Number of epochs
commits_df.loc[n, 'FE'] = 0                         # Was there a replacement of 300 for 350?
commits_df.loc[n, 'target4'] = 0.001378             # Target 0-4
commits_df.loc[n, 'target32_34'] = 0.002514         # Target 0-32 & 0-34
commits_df.loc[n, 'LB_score'] = 0.20208             # LB score after submitting


# ### Commit 16

# In[17]:


n=12
commits_df.loc[n, 'n_commit'] = 16                  # Number of version
commits_df.loc[n, 'learning_rate'] = 0.0067         # Learning rate
commits_df.loc[n, 'num_epochs'] = 200               # Number of epochs
commits_df.loc[n, 'FE'] = 0                         # Was there a replacement of 300 for 350?
commits_df.loc[n, 'target4'] = 0.001633             # Target 0-4
commits_df.loc[n, 'target32_34'] = 0.002473         # Target 0-32 & 0-34
commits_df.loc[n, 'LB_score'] = 0.20150             # LB score after submitting


# ### Commit 17

# In[18]:


n=13
commits_df.loc[n, 'n_commit'] = 17                  # Number of version
commits_df.loc[n, 'learning_rate'] = 0.006          # Learning rate
commits_df.loc[n, 'num_epochs'] = 65                # Number of epochs
commits_df.loc[n, 'FE'] = 0                         # Was there a replacement of 300 for 350?
commits_df.loc[n, 'target4'] = 0.000776             # Target 0-4
commits_df.loc[n, 'target32_34'] = 0.002251         # Target 0-32 & 0-34
commits_df.loc[n, 'LB_score'] = 0.20050             # LB score after submitting


# ### 2.3 Parameters and LB score visualization <a class="anchor" id="2.3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[19]:


# Find and mark minimun value of LB score
commits_df['LB_score'] = pd.to_numeric(commits_df['LB_score'])
commits_df = commits_df.sort_values(by=['LB_score'], ascending = True).reset_index(drop=True)
commits_df['min'] = 0
commits_df.loc[0, 'min'] = 1
commits_df


# In[20]:


# Interactive plot with results of parameters tuning
fig = px.scatter_3d(commits_df, x='learning_rate', y='num_epochs', z='LB_score', color = 'min', 
                    symbol = 'FE',
                    title='    Parameters and LB score visualization of "ORV Prediction" solutions')
fig.update(layout=dict(title=dict(x=0.1)))


# In[21]:


# Interactive plot with targets
fig = px.scatter_3d(commits_df, x='target4', y='target32_34', z='LB_score', color = 'min', 
                    symbol = 'learning_rate',
                    title='     Targets and LB score visualization of "ORV Prediction" solutions')
fig.update(layout=dict(title=dict(x=0.2)))


# In[22]:


# Interactive plot with learning rate and LB score
commits_df = commits_df.sort_values(by=['learning_rate'])
fig = px.line(commits_df, x='learning_rate', y="LB_score", text='n_commit',
              title="Learning rate and LB score with number of version",
              log_y=True,template='gridon',width=800, height=500)
fig.update_traces(textposition="bottom right")
fig.show()


# In[23]:


# Interactive plot with target for 0-4 and LB score
commits_df = commits_df.sort_values(by=['target4'])
fig = px.line(commits_df, x='target4', y="LB_score", text='n_commit',
              title="Target4 (for 0-4) and LB score with number of version",
              log_y=True,template='gridon',width=800, height=500)
fig.update_traces(textposition="bottom right")
fig.show()


# In[24]:


# Interactive plot with target for 0-32 & 0-34 and LB score
commits_df = commits_df.sort_values(by=['target32_34'])
fig = px.line(commits_df, x='target32_34', y="LB_score", text='n_commit',
              title="Target32 (for 0-32 and 0-34) and LB score with number of version",
              log_y=True,template='gridon',width=800, height=500)
fig.update_traces(textposition="bottom right")
fig.show()


# ## 3. Download data<a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[25]:


def read_train_test():
    # Function to read our base train and test set
    
    train = pd.read_csv('../input/optiver-realized-volatility-prediction/train.csv')
    test = pd.read_csv('../input/optiver-realized-volatility-prediction/test.csv')

    # Create a key to merge with book and trade data
    train['row_id'] = train['stock_id'].astype(str) + '-' + train['time_id'].astype(str)
    test['row_id'] = test['stock_id'].astype(str) + '-' + test['time_id'].astype(str)
    print(f'Our training set has {train.shape[0]} rows')
    
    return train, test


# In[26]:


# Read train and test
train, test = read_train_test()


# ## 4. FE & Data Preprocessing <a class="anchor" id="4"></a>
# 
# [Back to Table of Contents](#0.1)

# In[27]:


# data directory
data_dir = '../input/optiver-realized-volatility-prediction/'

def calc_wap1(df):
    # Function to calculate first WAP
    wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap

def calc_wap2(df):
    # Function to calculate second WAP
    wap = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])
    return wap

def log_return(series):
    # Function to calculate the log of the return
    return np.log(series).diff()

def realized_volatility(series):
    # Calculate the realized volatility
    return np.sqrt(np.sum(series**2))

def count_unique(series):
    # Function to count unique elements of a series
    return len(np.unique(series))

def book_preprocessor(file_path):
    # Function to preprocess book data (for each stock id)
    
    df = pd.read_parquet(file_path)
    
    # Calculate Wap
    df['wap1'] = calc_wap1(df)
    df['wap2'] = calc_wap2(df)
    
    # Calculate log returns
    df['log_return1'] = df.groupby(['time_id'])['wap1'].apply(log_return)
    df['log_return2'] = df.groupby(['time_id'])['wap2'].apply(log_return)
    
    # Calculate wap balance
    df['wap_balance'] = abs(df['wap1'] - df['wap2'])
    
    # Calculate spread
    df['price_spread'] = (df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1']) / 2)
    df['price_spread2'] = (df['ask_price2'] - df['bid_price2']) / ((df['ask_price2'] + df['bid_price2']) / 2)
    df['bid_spread'] = df['bid_price1'] - df['bid_price2']
    df['ask_spread'] = df['ask_price1'] - df['ask_price2']
    df["bid_ask_spread"] = abs(df['bid_spread'] - df['ask_spread'])
    df['total_volume'] = (df['ask_size1'] + df['ask_size2']) + (df['bid_size1'] + df['bid_size2'])
    df['volume_imbalance'] = abs((df['ask_size1'] + df['ask_size2']) - (df['bid_size1'] + df['bid_size2']))
    
    # Dict for aggregations
    create_feature_dict = {
        'wap1': [np.sum, np.mean, np.std],
        'wap2': [np.sum, np.mean, np.std],
        'log_return1': [np.sum, realized_volatility, np.mean, np.std],
        'log_return2': [np.sum, realized_volatility, np.mean, np.std],
        'wap_balance': [np.sum, np.mean, np.std],
        'price_spread':[np.sum, np.mean, np.std],
        'price_spread2':[np.sum, np.mean, np.std],
        'bid_spread':[np.sum, np.mean, np.std],
        'ask_spread':[np.sum, np.mean, np.std],
        'total_volume':[np.sum, np.mean, np.std],
        'volume_imbalance':[np.sum, np.mean, np.std],
        "bid_ask_spread":[np.sum, np.mean, np.std],
    }
    
    def get_stats_window(seconds_in_bucket, add_suffix = False):
        # Function to get group stats for different windows (seconds in bucket)
        
        # Group by the window
        df_feature = df[df['seconds_in_bucket'] >= seconds_in_bucket].groupby(['time_id']).agg(create_feature_dict).reset_index()
        
        # Rename columns joining suffix
        df_feature.columns = ['_'.join(col) for col in df_feature.columns]
        
        # Add a suffix to differentiate windows
        if add_suffix:
            df_feature = df_feature.add_suffix('_' + str(seconds_in_bucket))
        return df_feature
    
    # Get the stats for different windows
    df_feature = get_stats_window(seconds_in_bucket = 0, add_suffix = False)
    df_feature_400 = get_stats_window(seconds_in_bucket = 400, add_suffix = True)
    df_feature_300 = get_stats_window(seconds_in_bucket = 300, add_suffix = True)
    df_feature_200 = get_stats_window(seconds_in_bucket = 200, add_suffix = True)
    
    # Merge all
    df_feature = df_feature.merge(df_feature_400, how = 'left', left_on = 'time_id_', right_on = 'time_id__400')
    df_feature = df_feature.merge(df_feature_300, how = 'left', left_on = 'time_id_', right_on = 'time_id__300')
    df_feature = df_feature.merge(df_feature_200, how = 'left', left_on = 'time_id_', right_on = 'time_id__200')

    # Drop unnecesary time_ids
    df_feature.drop(['time_id__400', 'time_id__300', 'time_id__200'], axis = 1, inplace = True)
    
    
    # Create row_id so we can merge
    stock_id = file_path.split('=')[1]
    df_feature['row_id'] = df_feature['time_id_'].apply(lambda x: f'{stock_id}-{x}')
    df_feature.drop(['time_id_'], axis = 1, inplace = True)
    
    return df_feature


def trade_preprocessor(file_path):
    # Function to preprocess trade data (for each stock id)
    
    df = pd.read_parquet(file_path)
    df['log_return'] = df.groupby('time_id')['price'].apply(log_return)
    
    # Dict for aggregations
    create_feature_dict = {
        'log_return':[realized_volatility],
        'seconds_in_bucket':[count_unique],
        'size':[np.sum, realized_volatility, np.mean, np.std, np.max, np.min],
        'order_count':[np.mean,np.sum,np.max],
    }
    
    def get_stats_window(seconds_in_bucket, add_suffix = False):
        # Function to get group stats for different windows (seconds in bucket)
        
        # Group by the window
        df_feature = df[df['seconds_in_bucket'] >= seconds_in_bucket].groupby(['time_id']).agg(create_feature_dict).reset_index()
        
        # Rename columns joining suffix
        df_feature.columns = ['_'.join(col) for col in df_feature.columns]
        
        # Add a suffix to differentiate windows
        if add_suffix:
            df_feature = df_feature.add_suffix('_' + str(seconds_in_bucket))
        return df_feature
    
    # Get the stats for different windows
    df_feature = get_stats_window(seconds_in_bucket = 0, add_suffix = False)
    df_feature_400 = get_stats_window(seconds_in_bucket = 400, add_suffix = True)
    df_feature_300 = get_stats_window(seconds_in_bucket = 300, add_suffix = True)
    df_feature_200 = get_stats_window(seconds_in_bucket = 200, add_suffix = True)
    
    def tendency(price, vol):    
        df_diff = np.diff(price)
        val = (df_diff/price[1:])*100
        power = np.sum(val*vol[1:])
        return(power)
    
    lis = []
    for n_time_id in df['time_id'].unique():
        df_id = df[df['time_id'] == n_time_id]        
        tendencyV = tendency(df_id['price'].values, df_id['size'].values)      
        f_max = np.sum(df_id['price'].values > np.mean(df_id['price'].values))
        f_min = np.sum(df_id['price'].values < np.mean(df_id['price'].values))
        df_max =  np.sum(np.diff(df_id['price'].values) > 0)
        df_min =  np.sum(np.diff(df_id['price'].values) < 0)
        abs_diff = np.median(np.abs( df_id['price'].values - np.mean(df_id['price'].values)))        
        energy = np.mean(df_id['price'].values**2)
        iqr_p = np.percentile(df_id['price'].values,75) - np.percentile(df_id['price'].values,25)
        abs_diff_v = np.median(np.abs( df_id['size'].values - np.mean(df_id['size'].values)))        
        energy_v = np.sum(df_id['size'].values**2)
        iqr_p_v = np.percentile(df_id['size'].values,75) - np.percentile(df_id['size'].values,25)
        
        lis.append({'time_id':n_time_id,'tendency':tendencyV,'f_max':f_max,'f_min':f_min,'df_max':df_max,'df_min':df_min,
                   'abs_diff':abs_diff,'energy':energy,'iqr_p':iqr_p,'abs_diff_v':abs_diff_v,'energy_v':energy_v,'iqr_p_v':iqr_p_v})
    
    df_lr = pd.DataFrame(lis)
        
   
    df_feature = df_feature.merge(df_lr, how = 'left', left_on = 'time_id_', right_on = 'time_id')
    
    # Merge all
    df_feature = df_feature.merge(df_feature_400, how = 'left', left_on = 'time_id_', right_on = 'time_id__400')
    df_feature = df_feature.merge(df_feature_300, how = 'left', left_on = 'time_id_', right_on = 'time_id__300')
    df_feature = df_feature.merge(df_feature_200, how = 'left', left_on = 'time_id_', right_on = 'time_id__200')

    # Drop unnecesary time_ids
    df_feature.drop(['time_id__400', 'time_id__300', 'time_id__200','time_id'], axis = 1, inplace = True)
    df_feature = df_feature.add_prefix('trade_')
    stock_id = file_path.split('=')[1]
    df_feature['row_id'] = df_feature['trade_time_id_'].apply(lambda x:f'{stock_id}-{x}')
    df_feature.drop(['trade_time_id_'], axis = 1, inplace = True)
    
    return df_feature


def get_time_stock(df):
    # Function to get group stats for the stock_id and time_id
    
    # Get realized volatility columns
    vol_cols = ['log_return1_realized_volatility', 'log_return2_realized_volatility', 'log_return1_realized_volatility_400', 'log_return2_realized_volatility_400', 
                'log_return1_realized_volatility_300', 'log_return2_realized_volatility_300', 'log_return1_realized_volatility_200', 'log_return2_realized_volatility_200', 
                'trade_log_return_realized_volatility', 'trade_log_return_realized_volatility_400', 'trade_log_return_realized_volatility_300', 'trade_log_return_realized_volatility_200']

    # Group by the stock id
    df_stock_id = df.groupby(['stock_id'])[vol_cols].agg(['mean', 'std', 'max', 'min', ]).reset_index()
    
    # Rename columns joining suffix
    df_stock_id.columns = ['_'.join(col) for col in df_stock_id.columns]
    df_stock_id = df_stock_id.add_suffix('_' + 'stock')

    # Group by the stock id
    df_time_id = df.groupby(['time_id'])[vol_cols].agg(['mean', 'std', 'max', 'min', ]).reset_index()
    
    # Rename columns joining suffix
    df_time_id.columns = ['_'.join(col) for col in df_time_id.columns]
    df_time_id = df_time_id.add_suffix('_' + 'time')
    
    # Merge with original dataframe
    df = df.merge(df_stock_id, how = 'left', left_on = ['stock_id'], right_on = ['stock_id__stock'])
    df = df.merge(df_time_id, how = 'left', left_on = ['time_id'], right_on = ['time_id__time'])
    df.drop(['stock_id__stock', 'time_id__time'], axis = 1, inplace = True)
    
    return df
    
    
def preprocessor(list_stock_ids, is_train = True):
    # Funtion to make preprocessing function in parallel (for each stock id)
    
    # Parrallel for loop
    def for_joblib(stock_id):
        # Train
        if is_train:
            file_path_book = data_dir + "book_train.parquet/stock_id=" + str(stock_id)
            file_path_trade = data_dir + "trade_train.parquet/stock_id=" + str(stock_id)
        # Test
        else:
            file_path_book = data_dir + "book_test.parquet/stock_id=" + str(stock_id)
            file_path_trade = data_dir + "trade_test.parquet/stock_id=" + str(stock_id)
    
        # Preprocess book and trade data and merge them
        df_tmp = pd.merge(book_preprocessor(file_path_book), trade_preprocessor(file_path_trade), on = 'row_id', how = 'left')
        
        # Return the merge dataframe
        return df_tmp
    
    # Use parallel api to call paralle for loop
    df = Parallel(n_jobs = -1, verbose = 1)(delayed(for_joblib)(stock_id) for stock_id in list_stock_ids)
    
    # Concatenate all the dataframes that return from Parallel
    df = pd.concat(df, ignore_index = True)
    
    return df


def rmspe(y_true, y_pred):
    # Function to calculate the root mean squared percentage error
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

def feval_rmspe(y_pred, lgb_train):
    # Function to early stop with root mean squared percentage error
    y_true = lgb_train.get_label()
    return 'RMSPE', rmspe(y_true, y_pred), False


# In[28]:


# Get unique stock ids 
train_stock_ids = train['stock_id'].unique()

# Preprocess them using Parallel and our single stock id functions
train_ = preprocessor(train_stock_ids, is_train = True)
train = train.merge(train_, on = ['row_id'], how = 'left')

# Get unique stock ids 
test_stock_ids = test['stock_id'].unique()

# Preprocess them using Parallel and our single stock id functions
test_ = preprocessor(test_stock_ids, is_train = False)
test = test.merge(test_, on = ['row_id'], how = 'left')

# Get group stats of time_id and stock_id
train = get_time_stock(train)
test = get_time_stock(test)


# In[29]:


# replace by order sum (tau)
train['size_tau'] = np.sqrt(1/train['trade_seconds_in_bucket_count_unique'])
test['size_tau'] = np.sqrt(1/test['trade_seconds_in_bucket_count_unique'])
train['size_tau_400'] = np.sqrt(1/train['trade_seconds_in_bucket_count_unique_400'])
test['size_tau_400'] = np.sqrt(1/test['trade_seconds_in_bucket_count_unique_400'])
train['size_tau_300'] = np.sqrt(1/train['trade_seconds_in_bucket_count_unique_300'])
test['size_tau_300'] = np.sqrt(1/test['trade_seconds_in_bucket_count_unique_300'])
train['size_tau_200'] = np.sqrt(1/train['trade_seconds_in_bucket_count_unique_200'])
test['size_tau_200'] = np.sqrt(1/test['trade_seconds_in_bucket_count_unique_200'])


# In[30]:


# tau2 
train['size_tau2'] = np.sqrt(1/train['trade_order_count_sum'])
test['size_tau2'] = np.sqrt(1/test['trade_order_count_sum'])
train['size_tau2_400'] = np.sqrt(0.25/train['trade_order_count_sum'])
test['size_tau2_400'] = np.sqrt(0.25/test['trade_order_count_sum'])
train['size_tau2_300'] = np.sqrt(0.5/train['trade_order_count_sum'])
test['size_tau2_300'] = np.sqrt(0.5/test['trade_order_count_sum'])
train['size_tau2_200'] = np.sqrt(0.75/train['trade_order_count_sum'])
test['size_tau2_200'] = np.sqrt(0.75/test['trade_order_count_sum'])

# delta tau
train['size_tau2_d'] = train['size_tau2_400'] - train['size_tau2']
test['size_tau2_d'] = test['size_tau2_400'] - test['size_tau2']


# ## 5. Modeling and prediction<a class="anchor" id="5"></a>
# 
# [Back to Table of Contents](#0.1)

# In[31]:


# kfold based on the knn++ algorithm

out_train = pd.read_csv('../input/optiver-realized-volatility-prediction/train.csv')
out_train = out_train.pivot(index='time_id', columns='stock_id', values='target')

# out_train[out_train.isna().any(axis=1)]
out_train = out_train.fillna(out_train.mean())
out_train.head()

# Code to add the just the read data after first execution

# Data separation based on knn ++
nfolds = 5 # number of folds
index = []
totDist = []
values = []

# Generates a matriz with the values of 
mat = out_train.values
scaler = MinMaxScaler(feature_range=(-1, 1))
mat = scaler.fit_transform(mat)
nind = int(mat.shape[0]/nfolds) # number of individuals

# Adds index in the last column
mat = np.c_[mat,np.arange(mat.shape[0])]
lineNumber = np.random.choice(np.array(mat.shape[0]), size=nfolds, replace=False)
lineNumber = np.sort(lineNumber)[::-1]
for n in range(nfolds):
    totDist.append(np.zeros(mat.shape[0]-nfolds))

# Saves index
for n in range(nfolds):    
    values.append([lineNumber[n]])

s=[]
for n in range(nfolds):
    s.append(mat[lineNumber[n],:])
    mat = np.delete(mat, obj=lineNumber[n], axis=0)

for n in range(nind-1):    
    luck = np.random.uniform(0,1,nfolds)
    
    for cycle in range(nfolds):
        # Saves the values of index           
        s[cycle] = np.matlib.repmat(s[cycle], mat.shape[0], 1)
        sumDist = np.sum( (mat[:,:-1] - s[cycle][:,:-1])**2 , axis=1)   
        totDist[cycle] += sumDist        
                
        # Probabilities
        f = totDist[cycle]/np.sum(totDist[cycle]) # normalizing the totDist
        j = 0
        kn = 0
        for val in f:
            j += val        
            if (j > luck[cycle]): # the column was selected
                break
            kn +=1
        lineNumber[cycle] = kn
        
        # Delete line of the value added    
        for n_iter in range(nfolds):
            totDist[n_iter] = np.delete(totDist[n_iter],obj=lineNumber[cycle], axis=0)
            j= 0
        
        s[cycle] = mat[lineNumber[cycle],:]
        values[cycle].append(int(mat[lineNumber[cycle],-1]))
        mat = np.delete(mat, obj=lineNumber[cycle], axis=0)

for n_mod in range(nfolds):
    values[n_mod] = out_train.index[values[n_mod]]


# In[32]:


def root_mean_squared_per_error(y_true, y_pred):
         return K.sqrt(K.mean(K.square( (y_true - y_pred)/ y_true )))
    
es = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=20, verbose=0,
    mode='min',restore_best_weights=True)

plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=7, verbose=0,
    mode='min')


# In[33]:


colNames = list(train)
colNames.remove('time_id')
colNames.remove('target')
colNames.remove('row_id')
colNames.remove('stock_id')


# In[34]:


train.replace([np.inf, -np.inf], np.nan,inplace=True)
test.replace([np.inf, -np.inf], np.nan,inplace=True)
qt_train = []

for col in colNames:
    qt = QuantileTransformer(random_state=21,n_quantiles=2000, output_distribution='normal')
    train[col] = qt.fit_transform(train[[col]])
    test[col] = qt.transform(test[[col]])    
    qt_train.append(qt)


# In[35]:


# Making agg features

train_p = pd.read_csv('../input/optiver-realized-volatility-prediction/train.csv')
train_p = train_p.pivot(index='time_id', columns='stock_id', values='target')
corr = train_p.corr()
ids = corr.index
kmeans = KMeans(n_clusters=7, random_state=0).fit(corr.values)
print(kmeans.labels_)
l = []
for n in range(7):
    l.append ( [ (x-1) for x in ( (ids+1)*(kmeans.labels_ == n)) if x > 0] )

mat = []
matTest = []
n = 0
for ind in l:
    print(ind)
    newDf = train.loc[train['stock_id'].isin(ind) ]
    newDf = newDf.groupby(['time_id']).agg(np.nanmean)
    newDf.loc[:,'stock_id'] = str(n)+'c1'
    mat.append ( newDf )
    newDf = test.loc[test['stock_id'].isin(ind) ]    
    newDf = newDf.groupby(['time_id']).agg(np.nanmean)
    newDf.loc[:,'stock_id'] = str(n)+'c1'
    matTest.append ( newDf )
    n+=1
    
mat1 = pd.concat(mat).reset_index()
mat1.drop(columns=['target'],inplace=True)
mat2 = pd.concat(matTest).reset_index()


# In[36]:


matTest = []
mat = []
kmeans = []


# In[37]:


mat2 = pd.concat([mat2,mat1.loc[mat1.time_id==5]])


# In[38]:


mat1 = mat1.pivot(index='time_id', columns='stock_id')
mat1.columns = ["_".join(x) for x in mat1.columns.ravel()]
mat1.reset_index(inplace=True)

mat2 = mat2.pivot(index='time_id', columns='stock_id')
mat2.columns = ["_".join(x) for x in mat2.columns.ravel()]
mat2.reset_index(inplace=True)


# In[39]:


nnn = ['time_id',
     'log_return1_realized_volatility_0c1',
     'log_return1_realized_volatility_1c1',     
     'log_return1_realized_volatility_3c1',
     'log_return1_realized_volatility_4c1',     
     'log_return1_realized_volatility_6c1',
     'total_volume_mean_0c1',
     'total_volume_mean_1c1', 
     'total_volume_mean_3c1',
     'total_volume_mean_4c1', 
     'total_volume_mean_6c1',
     'trade_size_mean_0c1',
     'trade_size_mean_1c1', 
     'trade_size_mean_3c1',
     'trade_size_mean_4c1', 
     'trade_size_mean_6c1',
     'trade_order_count_mean_0c1',
     'trade_order_count_mean_1c1',
     'trade_order_count_mean_3c1',
     'trade_order_count_mean_4c1',
     'trade_order_count_mean_6c1',      
     'price_spread_mean_0c1',
     'price_spread_mean_1c1',
     'price_spread_mean_3c1',
     'price_spread_mean_4c1',
     'price_spread_mean_6c1',   
     'bid_spread_mean_0c1',
     'bid_spread_mean_1c1',
     'bid_spread_mean_3c1',
     'bid_spread_mean_4c1',
     'bid_spread_mean_6c1',       
     'ask_spread_mean_0c1',
     'ask_spread_mean_1c1',
     'ask_spread_mean_3c1',
     'ask_spread_mean_4c1',
     'ask_spread_mean_6c1',   
     'volume_imbalance_mean_0c1',
     'volume_imbalance_mean_1c1',
     'volume_imbalance_mean_3c1',
     'volume_imbalance_mean_4c1',
     'volume_imbalance_mean_6c1',       
     'bid_ask_spread_mean_0c1',
     'bid_ask_spread_mean_1c1',
     'bid_ask_spread_mean_3c1',
     'bid_ask_spread_mean_4c1',
     'bid_ask_spread_mean_6c1',
     'size_tau2_0c1',
     'size_tau2_1c1',
     'size_tau2_3c1',
     'size_tau2_4c1',
     'size_tau2_6c1'] 


# In[40]:


train = pd.merge(train,mat1[nnn],how='left',on='time_id')


# In[41]:


test = pd.merge(test,mat2[nnn],how='left',on='time_id')


# In[42]:


mat1 = []
mat2 = []


# In[43]:


# Thanks to https://bignerdranch.com/blog/implementing-swish-activation-function-in-keras/
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

get_custom_objects().update({'swish': Activation(swish)})


# In[44]:


hidden_units = (128,64,32)
stock_embedding_size = 24
cat_data = train['stock_id']

def base_model():
    
    # Each instance will consist of two inputs: a single user id, and a single movie id
    stock_id_input = keras.Input(shape=(1,), name='stock_id')
    num_input = keras.Input(shape=(362,), name='num_data')

    # Embedding, flatenning and concatenating
    stock_embedded = keras.layers.Embedding(max(cat_data)+1, stock_embedding_size, 
                                           input_length=1, name='stock_embedding')(stock_id_input)
    stock_flattened = keras.layers.Flatten()(stock_embedded)
    out = keras.layers.Concatenate()([stock_flattened, num_input])
    
    # Add one or more hidden layers
    for n_hidden in hidden_units:
        out = keras.layers.Dense(n_hidden, activation='swish')(out)

    # A single output: our predicted rating
    out = keras.layers.Dense(1, activation='linear', name='prediction')(out)
    
    model = keras.Model(
    inputs = [stock_id_input, num_input],
    outputs = out,
    )
    
    return model


# In[45]:


model_name = 'NN'
pred_name = 'pred_{}'.format(model_name)

n_folds = 5
kf = model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=2020)
scores_folds[model_name] = []
counter = 1

features_to_consider = list(train)

features_to_consider.remove('time_id')
features_to_consider.remove('target')
features_to_consider.remove('row_id')
try:
    features_to_consider.remove('pred_NN')
except:
    pass

train[features_to_consider] = train[features_to_consider].fillna(train[features_to_consider].mean())
test[features_to_consider] = test[features_to_consider].fillna(train[features_to_consider].mean())

train[pred_name] = 0
test['target'] = 0

for n_count in range(n_folds):
    print('CV {}/{}'.format(counter, n_folds))
    
    indexes = np.arange(nfolds).astype(int)    
    indexes = np.delete(indexes,obj=n_count, axis=0) 
    indexes = np.r_[values[indexes[0]],values[indexes[1]],values[indexes[2]],values[indexes[3]]]
    
    X_train = train.loc[train.time_id.isin(indexes), features_to_consider]
    y_train = train.loc[train.time_id.isin(indexes), target_name]
    X_test = train.loc[train.time_id.isin(values[n_count]), features_to_consider]
    y_test = train.loc[train.time_id.isin(values[n_count]), target_name]
    
    # NN
    model = base_model()
    
    model.compile(
        keras.optimizers.Adam(learning_rate=learning_rate),
        loss=root_mean_squared_per_error
    )
    
    try:
        features_to_consider.remove('stock_id')
    except:
        pass
    
    num_data = X_train[features_to_consider]
    
    scaler = MinMaxScaler(feature_range=(-1, 1))         
    num_data = scaler.fit_transform(num_data.values)    
    
    cat_data = X_train['stock_id']    
    target =  y_train
    
    num_data_test = X_test[features_to_consider]
    num_data_test = scaler.transform(num_data_test.values)
    cat_data_test = X_test['stock_id']

    model.fit([cat_data, num_data], 
              target,               
              batch_size=2048,
              epochs=num_epochs,
              validation_data=([cat_data_test, num_data_test], y_test),
              callbacks=[es, plateau],
              validation_batch_size=len(y_test),
              shuffle=True,
             verbose = 1)

    preds = model.predict([cat_data_test, num_data_test]).reshape(1,-1)[0]
    
    score = round(rmspe(y_true = y_test, y_pred = preds),5)
    print('Fold {} {}: {}'.format(counter, model_name, score))
    scores_folds[model_name].append(score)
    
    tt =scaler.transform(test[features_to_consider].values)
    test[target_name] += model.predict([test['stock_id'], tt]).reshape(1,-1)[0].clip(0,1e10)
       
    counter += 1
    features_to_consider.append('stock_id')


# ## 6. Submission <a class="anchor" id="6"></a>
# 
# [Back to Table of Contents](#0.1)

# In[46]:


# Postprocessing
test[target_name] = test[target_name]/n_folds

score = round(rmspe(y_true = train[target_name].values, y_pred = train[pred_name].values),5)
print('RMSPE {}: {} - Folds: {}'.format(model_name, score, scores_folds[model_name]))

display(test[['row_id', target_name]].head(2))

# Submission
test[['row_id', target_name]].to_csv('submission.csv',index = False)


# [Go to Top](#0)
