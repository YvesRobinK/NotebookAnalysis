#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


get_ipython().run_cell_magic('capture', '', '!pip install pycaret[full]\n')


# In[3]:


from pycaret.regression import *


# In[4]:


train = pd.read_csv('../input/tabular-playground-series-mar-2022/train.csv', index_col=0)
test = pd.read_csv('../input/tabular-playground-series-mar-2022/test.csv', index_col=0)
sub = pd.read_csv('../input/tabular-playground-series-mar-2022/sample_submission.csv')


# In[5]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)  
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[6]:


reduce_mem_usage(train)
reduce_mem_usage(test)


# # Simple Feature Engineering on Datetime
# * Convert time strings to datetime
# * Extract Month from datetime
# * Extract day of week from datetime
# * Extract time of day from datetime. The time is transformed into the number of minutes, e.g. 14:20 is transformed to 14x60+20 = 860
# * Cyclical Features is given by [INVERSION](https://www.kaggle.com/inversion/tps-mar-22-cyclical-features)
# 

# In[7]:


from datetime import datetime
from math import sin, cos, pi


# In[8]:


def fe(data):
    data['time'] = pd.to_datetime(data['time'])
    data['month'] = data['time'].dt.month
    data['dayofweek'] = data['time'].dt.dayofweek
    data['hourminute'] = data['time'].dt.hour *60 + data['time'].dt.minute
    data = data.drop(['time'], axis=1)
    
    sin_vals = {
        'NB': 0.0,
        'NE': sin(1 * pi/4),
        'EB': 1.0,
        'SE': sin(3 * pi/4),
        'SB': 0.0,
        'SW': sin(5 * pi/4),    
        'WB': -1.0,    
        'NW': sin(7 * pi/4),  
    }

    cos_vals = {
        'NB': 1.0,
        'NE': cos(1 * pi/4),
        'EB': 0.0,
        'SE': cos(3 * pi/4),
        'SB': -1.0,
        'SW': cos(5 * pi/4),    
        'WB': 0.0,    
        'NW': cos(7 * pi/4),  
    }


    data['sin'] = data['direction'].map(sin_vals)
    data['cos'] = data['direction'].map(cos_vals)

    encoded_vals = {
        'NB': 0,
        'NE': 1,
        'EB': 2,
        'SE': 3,
        'SB': 4,
        'SW': 5,
        'WB': 6, 
        'NW': 7,
    }

    data['direction'] = data['direction'].map(encoded_vals)
    
    
    return data


# In[9]:


train = fe(train)
test = fe(test)


# In[10]:


train.head()


# In[11]:


print('Train data shape:', train.shape)
print('Train data shape:', test.shape)


# # PyCaret AutoML
# ## 1. Setting up the regression
# Pass the complete training dataset (including the target) as data and the congestion to be predicted as target

# In[12]:


tps_123 = setup(data = train, 
                target = 'congestion', 
                use_gpu = True,
                n_jobs = -1,
                silent = True,
                data_split_shuffle = True,
                fold_shuffle = True,
                train_size = 0.7,
                fold = 5,
                session_id=123
                ) 


# In[13]:


# best = compare_models()


# ## 2. Creating the initial models
# Three models are created, i.e.
# * CatBoost
# * XGBoost
# * LightGBM

# In[14]:


cat = create_model('catboost')
xgb = create_model('xgboost')
lgbm = create_model('lightgbm')


# ## 3. Model Tuning with Optuna

# In[15]:


cat = tune_model(cat, search_library = 'scikit-learn', search_algorithm = 'random', optimize = 'MAE', n_iter = 50)
xgb = tune_model(xgb, search_library = 'scikit-learn', search_algorithm = 'random', optimize = 'MAE', n_iter = 50)
lgbm = tune_model(lgbm, search_library = 'scikit-learn', search_algorithm = 'random', optimize = 'MAE', n_iter = 50)


# ## 4. Model Blending
# The three tuned model are blended.

# In[16]:


blender = blend_models(estimator_list = [cat, xgb, lgbm])


# ## 5. Finalize model
# The model parameters are updated by the entire training data including the hold-out set.

# In[17]:


model = finalize_model(blender)
y_pred = predict_model(model, data = test)['Label']


# # Submission

# In[18]:


sub["congestion"] = y_pred.values
sub.to_csv("submission.csv", index=False)
sub


# In[19]:


sub["congestion"] = y_pred.values.round()
sub.to_csv("submission1.csv", index=False)
sub

