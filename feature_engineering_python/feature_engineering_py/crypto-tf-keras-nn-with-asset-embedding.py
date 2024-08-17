#!/usr/bin/env python
# coding: utf-8

# A simple NN starter using asset Embedding. Hope it will be useful for you.
# 
# Heavily inspired from this notebook: https://www.kaggle.com/lucasmorin/tf-keras-nn-with-stock-embedding.
#         
# This code is only trained with 100,000 data for time consuming. It can be change to all data with the debug pamameter. 
# 

# In[1]:


import io
import json
import requests
import functools
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import backend as K

debug = True


# ## Load data

# In[2]:


if debug:
    train = pd.read_csv('../input/g-research-crypto-forecasting/train.csv', nrows=100000)
else:
    train = pd.read_csv('../input/g-research-crypto-forecasting/train.csv')


train = train[~train.isin([np.nan, np.inf, -np.inf]).any(1)].reset_index(drop=True)


print('train shape:',train.shape)
train.head() 


# In[3]:


test = pd.read_csv('../input/g-research-crypto-forecasting/example_test.csv')
test.head()


# The columns in the train data is just the same as the real market, so we can make very practical models. Besides, the test date is part of the train data, so the LB is not trustable, we should rely on the CV score.

# ## Feature Engineering
# 
# Only use np.log1p to price data now. Some features will be add in the future.

# In[4]:


def feature_engineer(df):
    for col in ['Open', 'High', 'Low', 'Close', 'VWAP']:
        df[col] = np.log1p(df[col])
    df = df.fillna(0)
    return df

train = feature_engineer(train)
test = feature_engineer(test)


# In[5]:


from datetime import datetime
train_date = pd.DataFrame(train.timestamp.unique())
train_date.columns = ['timestamp']
train_date['date'] = [datetime.fromtimestamp(u) for u in train_date['timestamp']]

print('train data begin date:',train_date.head(1)['date'].values[0])
print('train data end date:',train_date.tail(1)['date'].values[0])


# In[6]:


test_date = pd.DataFrame(test.timestamp.unique())
test_date.columns = ['timestamp']
test_date['date'] = [datetime.fromtimestamp(u) for u in test_date['timestamp']]
print('example test data begin date:',test_date.head(1)['date'].values[0])
print('example test data end date:',test_date.tail(1)['date'].values[0])


# ## Reduce train memory
# Reduce the memory of train data in case of OOM problem.

# In[7]:


def reduce_mem_usage(df, verbose=True):
    end_mem_ori = df.memory_usage().sum() / 1024**2
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
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased from {:5.2f} to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem_ori, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

train = reduce_mem_usage(train)


# In[8]:


numerical_columns =  ['Count', 'Open', 'High', 'Low', 'Close',
       'Volume', 'VWAP']
category_columns = ['Asset_ID']
target_columns = ['Target']


# In[9]:


asset_nunique = train['Asset_ID'].nunique()
print('asset_nunique:',asset_nunique)


# In[10]:


scaler = RobustScaler()
train[numerical_columns] = scaler.fit_transform(train[numerical_columns])
test[numerical_columns] = scaler.transform(test[numerical_columns])


# ## NN model
# 
# NN model with asset embedding.

# In[11]:


hidden_units = (32,16,8,4,2)

cat_data = train['Asset_ID']

def base_model():
    
    # Each instance will consist of two inputs: a single user id, and a single movie id
    stock_id_input = keras.Input(shape=(1,), name='stock_id')

    
    num_input = keras.Input(shape=(len(numerical_columns),), name='num_data')


    #embedding, flatenning and concatenating
    stock_embedded = keras.layers.Embedding(16, 8, 
                                           input_length=1, name='stock_embedding')(stock_id_input)
    
    
    stock_flattened = keras.layers.Flatten()(stock_embedded)


    out = keras.layers.Concatenate()([stock_flattened,num_input])
    hidden_units = (32,16,8,4,2)



    # Add one or more hidden layers
    for n_hidden in hidden_units:

        out = keras.layers.Dense(n_hidden, activation='selu')(out)        

    # A single output: our predicted rating
    out = keras.layers.Dense(1, activation='linear', name='prediction')(out)
    
    model = keras.Model(
    inputs = [stock_id_input, num_input],
    outputs = out,
    )
    
    return model


# In[12]:


model_name = 'NN'
pred_name = 'pred_{}'.format(model_name)
n_folds = 5

features_to_consider = numerical_columns + category_columns

train[pred_name] = 0
modellist = []

from sklearn.model_selection import KFold,GroupKFold
# kfold = KFold(n_splits = 5, random_state = 2021, shuffle = True)
kfoldgroup = GroupKFold(n_splits = 5)
oof_predictions = np.zeros(train.shape[0])
test_predictions = np.zeros(test.shape[0])
        
# for fold, (trn_ind, val_ind) in enumerate(kfold.split(train)):
for fold, (trn_ind, val_ind) in enumerate(kfoldgroup.split(range(len(train)),train.Target,train.timestamp)):
    X_train = train.loc[trn_ind, features_to_consider]
    y_train = train.loc[trn_ind, target_columns].values
    X_test = train.loc[val_ind, features_to_consider]
    y_test = train.loc[val_ind, target_columns].values
    
    
    model = base_model()
    
    model.compile(
        keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.metrics.mean_squared_error,
        metrics=['MSE'],
    )

    num_data = X_train[numerical_columns]
    stock_data = X_train['Asset_ID']
    

    
    num_data_test = X_test[numerical_columns]
    stock_data_test = X_test['Asset_ID']
    
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=1e-05, patience=15, verbose=0,
        mode='min', baseline=0.25)

    plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=7, verbose=0,
        mode='min')

    model.fit([stock_data,num_data], 
              y_train, 
              batch_size=4096,
              epochs=100,
              validation_data=([stock_data_test, num_data_test], y_test),
              callbacks=[es, plateau],
              shuffle=True,
             verbose = 0)

    preds = model.predict([stock_data_test, num_data_test]).reshape(1,-1)[0]
    oof_predictions[val_ind] = preds
    score_fold = round(pearsonr(y_test.reshape(1,-1)[0], preds)[0],5)
    print(f'fold {fold} oof score:',score_fold)

    test_predictions += model.predict([test['Asset_ID'], test[numerical_columns]]).reshape(1,-1)[0]/5
    modellist.append(model)
    
    


# In[13]:


test['Target'] = test_predictions
score = round(pearsonr(train[target_columns].values.reshape(1,-1)[0], oof_predictions)[0],5)
print('oof score all:',score)


# ### Submission

# In[14]:


import gresearch_crypto
env = gresearch_crypto.make_env()
iter_test = env.iter_test()


# In[15]:


for (test_df, sample_prediction_df) in iter_test:
    test_df = feature_engineer(test_df)
    test_predictions = 0
    test_df[numerical_columns] = scaler.transform(test_df[numerical_columns])
    for model in modellist:
        test_predictions += model.predict([test_df['Asset_ID'], test_df[numerical_columns]]).reshape(1,-1)[0]/len(modellist)
    sample_prediction_df['Target'] = test_predictions
    env.predict(sample_prediction_df)


# In[ ]:





# In[ ]:




