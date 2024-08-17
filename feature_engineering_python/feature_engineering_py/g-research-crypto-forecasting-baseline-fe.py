#!/usr/bin/env python
# coding: utf-8

# <a class="anchor" id="0"></a>
# # Competition "[G-Research Crypto Forecasting](https://www.kaggle.com/c/g-research-crypto-forecasting)"
# 
# ## Baseline code with advFE

# ## Acknowledgements:
# * the main code - the notebook [💲💲G-Research- Starter LGBM Pipeline](https://www.kaggle.com/julian3833/g-research-starter-lgbm-pipeline)
# * the model tuning - the notebook [G-Research: XGBoost with GPU (Fit in 1min)](https://www.kaggle.com/yamqwe/g-research-xgboost-with-gpu-fit-in-1min)
# * FE - the notebook [GResearch Simple LGB Starter](https://www.kaggle.com/code1110/gresearch-simple-lgb-starter)
# * remove inf, NaN - [[Crypto] Beginner's Try for simple LGBM (En/Jp)](https://www.kaggle.com/junjitakeshima/crypto-beginner-s-try-for-simple-lgbm-en-jp)

# <a class="anchor" id="0.1"></a>
# 
# ## Table of Contents
# 
# 1. [Import libraries](#1)
# 1. [Download datasets](#2)
# 1. [FE](#3)
# 1. [Model training](#4)
# 1. [Prediction and submission](#5)

# ## 1. Import libraries <a class="anchor" id="1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[1]:


import os
import numpy as np 
import pandas as pd 
import random
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor

import gresearch_crypto

import warnings
warnings.filterwarnings("ignore")


# In[2]:


def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

fix_all_seeds(42)


# ## 2. Download datasets <a class="anchor" id="2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[3]:


df_train = pd.read_csv('/kaggle/input/g-research-crypto-forecasting/train.csv')
df_train.head()


# In[4]:


df_asset_details = pd.read_csv('/kaggle/input/g-research-crypto-forecasting/asset_details.csv')
df_asset_details


# ## 3. FE <a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[5]:


def get_features(data):
    # FE for data as row of DataFrame
    
    # Two new features from the competition tutorial
    df_feat = data[['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']].copy()
    df_feat['Upper_Shadow'] = df_feat['High'] - np.maximum(df_feat['Close'], df_feat['Open'])
    df_feat['Lower_Shadow'] = np.minimum(df_feat['Close'], df_feat['Open']) - df_feat['Low']
    
    # Thanks to https://www.kaggle.com/code1110/gresearch-simple-lgb-starter
    df_feat['high2low'] = df_feat['High'] / df_feat['Low']
    df_feat['volume2count'] = df_feat['Volume'] / (df_feat['Count'] + 1)
    
    return df_feat


# In[6]:


def get_data_for_asset(df_train, asset_id):
    # Get X and y
    
    df = df_train[df_train["Asset_ID"] == asset_id]    
    df_proc = get_features(df)
    df_proc['y'] = df['Target']
    #df_proc = df_proc[~df_proc.isin([np.nan, np.inf, -np.inf]).any(1)].reset_index(drop=True)
    df_proc = df_proc.dropna(how="any")
    
    X = df_proc.drop("y", axis=1)
    y = df_proc["y"]
    
    return X, y


# ## 4. Model training and prediction <a class="anchor" id="4"></a>
# 
# [Back to Table of Contents](#0.1)

# In[7]:


def model_training(X,y):
    # Model training
    
    model = LGBMRegressor(n_estimators=5000,num_leaves=700,learning_rate=0.1)
    model.fit(X, y)
    
    return model


# In[8]:


get_ipython().run_cell_magic('time', '', 'Xs = {}\nys = {}\nmodels = {}\n\nfor asset_id, asset_name in zip(df_asset_details[\'Asset_ID\'], df_asset_details[\'Asset_Name\']):\n    print(f"Training model for {asset_name:<16} (ID={asset_id:<2})")\n    X, y = get_data_for_asset(df_train, asset_id)    \n    model = model_training(X,y)\n    Xs[asset_id], ys[asset_id], models[asset_id] = X, y, model\n')


# In[9]:


# Check the model and it's possibility for the prediction 
print("Check the model and it's possibility for the prediction")
x = get_features(df_train.iloc[1])
y_pred = models[0].predict([x])
y_pred[0]


# ## 5. Prediction and submission <a class="anchor" id="5"></a>
# 
# [Back to Table of Contents](#0.1)

# In[10]:


# Prediction and submission
env = gresearch_crypto.make_env()
iter_test = env.iter_test()

for i, (df_test, df_pred) in enumerate(iter_test):
    for j, row in df_test.iterrows():
        
        try:
            model = models[row['Asset_ID']]
            x_test = get_features(row)
            y_pred = model.predict([x_test])[0]

            df_pred.loc[df_pred['row_id'] == row['row_id'], 'Target'] = y_pred
        
        except:
            print(f'{i}-th iteration of the test dataset, {j}-th row - there was the exception, then set Target = 0')
            df_pred.loc[df_pred['row_id'] == row['row_id'], 'Target'] = 0
            
        # Print just one sample row to get a feeling of what it looks like        
        if i == 0 and j == 0:
            print('Example of the x_test data')
            display(x_test)

    # Display the first prediction dataframe
    if i == 0:
        print('Example of the prediction for test data')
        display(df_pred)
    df_pred['Target'] = df_pred['Target'].fillna(0)

    # Send submissions
    env.predict(df_pred)


# I hope you find this kernel useful and enjoyable.
# 
# Your comments and feedback are most welcome.

# [Go to Top](#0)
