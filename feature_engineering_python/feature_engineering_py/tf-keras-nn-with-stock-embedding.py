#!/usr/bin/env python
# coding: utf-8

# ## NN starter
# 
# A simple NN starter using stock Embedding. 
# 
# Heavily inspired from this notebook for the feature engineering part:
# https://www.kaggle.com/manels/lgb-starter
# 
# Embedding layer from :
# https://www.kaggle.com/colinmorris/embedding-layers
# 
# Also see:
# * https://www.kaggle.com/jiashenliu/introduction-to-financial-concepts-and-data
# * https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/250324
# 
# **I hope it will be useful for other beginners.**

# In[1]:


from IPython.core.display import display, HTML

import pandas as pd
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import os
import gc

from joblib import Parallel, delayed

from sklearn import preprocessing, model_selection

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt 
import seaborn as sns

path_root = '../input/optiver-realized-volatility-prediction'
path_data = '../input/optiver-realized-volatility-prediction'
path_submissions = '/'

target_name = 'target'
scores_folds = {}


# In[2]:


def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff() 

def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return**2))

def rmspe(y_true, y_pred):
    return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))

def get_stock_stat(stock_id : int, dataType = 'train'):
    key = ['stock_id', 'time_id', 'seconds_in_bucket']
    
    #Book features
    df_book = pd.read_parquet(os.path.join(path_data, 'book_{}.parquet/stock_id={}/'.format(dataType, stock_id)))
    df_book['stock_id'] = stock_id
    cols = key + [col for col in df_book.columns if col not in key]
    df_book = df_book[cols]
    
    df_book['wap1'] = (df_book['bid_price1'] * df_book['ask_size1'] +
                                    df_book['ask_price1'] * df_book['bid_size1']) / (df_book['bid_size1'] + df_book['ask_size1'])
    df_book['wap2'] = (df_book['bid_price2'] * df_book['ask_size2'] +
                                    df_book['ask_price2'] * df_book['bid_size2']) / (df_book['bid_size2'] + df_book['ask_size2'])
    df_book['log_return1'] = df_book.groupby(by = ['time_id'])['wap1'].apply(log_return).fillna(0)
    df_book['log_return2'] = df_book.groupby(by = ['time_id'])['wap2'].apply(log_return).fillna(0)
    
    features_to_apply_realized_volatility = ['log_return'+str(i+1) for i in range(2)]
    stock_stat = df_book.groupby(by = ['stock_id', 'time_id'])[features_to_apply_realized_volatility]\
                        .agg(realized_volatility).reset_index()

    #Trade features
    trade_stat =  pd.read_parquet(os.path.join(path_data,'trade_{}.parquet/stock_id={}'.format(dataType, stock_id)))
    trade_stat = trade_stat.sort_values(by=['time_id', 'seconds_in_bucket']).reset_index(drop=True)
    trade_stat['stock_id'] = stock_id
    cols = key + [col for col in trade_stat.columns if col not in key]
    trade_stat = trade_stat[cols]
    trade_stat['trade_log_return1'] = trade_stat.groupby(by = ['time_id'])['price'].apply(log_return).fillna(0)
    trade_stat = trade_stat.groupby(by = ['stock_id', 'time_id'])[['trade_log_return1']]\
                           .agg(realized_volatility).reset_index()
    #Joining book and trade features
    stock_stat = stock_stat.merge(trade_stat, on=['stock_id', 'time_id'], how='left').fillna(-999)
    
    return stock_stat

def get_dataSet(stock_ids : list, dataType = 'train'):

    stock_stat = Parallel(n_jobs=-1)(
        delayed(get_stock_stat)(stock_id, dataType) 
        for stock_id in stock_ids
    )
    
    stock_stat_df = pd.concat(stock_stat, ignore_index = True)

    return stock_stat_df


# ## Train and test datasets

# In[3]:


train = pd.read_csv(os.path.join(path_data, 'train.csv'))
get_ipython().run_line_magic('time', "train_stock_stat_df = get_dataSet(stock_ids = train['stock_id'].unique(), dataType = 'train')")
train = pd.merge(train, train_stock_stat_df, on = ['stock_id', 'time_id'], how = 'left')
print('Train shape: {}'.format(train.shape))
display(train.head(2))

test = pd.read_csv(os.path.join(path_data, 'test.csv'))
test_stock_stat_df = get_dataSet(stock_ids = test['stock_id'].unique(), dataType = 'test')
test = pd.merge(test, test_stock_stat_df, on = ['stock_id', 'time_id'], how = 'left').fillna(0)
print('Test shape: {}'.format(test.shape))
display(test.head(2))


# ## Training model and making predictions

# In[4]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import backend as K


# In[5]:


hidden_units = (32,16,8,4,2)
stock_embedding_size = 16

cat_data = train['stock_id']

def base_model():
    
    # Each instance will consist of two inputs: a single user id, and a single movie id
    stock_id_input = keras.Input(shape=(1,), name='stock_id')
    num_input = keras.Input(shape=(3,), name='num_data')


    #embedding, flatenning and concatenating
    stock_embedded = keras.layers.Embedding(max(cat_data)+1, stock_embedding_size, 
                                           input_length=1, name='stock_embedding')(stock_id_input)
    stock_flattened = keras.layers.Flatten()(stock_embedded)
    out = keras.layers.Concatenate()([stock_flattened, num_input])
    
    # Add one or more hidden layers
    for n_hidden in hidden_units:

        out = keras.layers.Dense(n_hidden, activation='selu')(out)
        

    #out = keras.layers.Concatenate()([out, num_input])

    # A single output: our predicted rating
    out = keras.layers.Dense(1, activation='linear', name='prediction')(out)
    
    model = keras.Model(
    inputs = [stock_id_input, num_input],
    outputs = out,
    )
    
    return model


# In[6]:


es = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=1e-05, patience=10, verbose=1,
    mode='min', baseline=0.25)

plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=3, verbose=1,
    mode='min')


# In[7]:


model_name = 'NN'
pred_name = 'pred_{}'.format(model_name)

n_folds = 4
kf = model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=2020)
scores_folds[model_name] = []
counter = 1

features_to_consider = ['stock_id','log_return1','log_return2','trade_log_return1']

train[pred_name] = 0
test['target'] = 0

for dev_index, val_index in kf.split(range(len(train))):
    print('CV {}/{}'.format(counter, n_folds))
    
    #Bottleneck ? 
    X_train = train.loc[dev_index, features_to_consider]
    y_train = train.loc[dev_index, target_name].values
    X_test = train.loc[val_index, features_to_consider]
    y_test = train.loc[val_index, target_name].values
    
    #############################################################################################
    # NN
    #############################################################################################
    
    model = base_model()
    
    model.compile(
        keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.metrics.mean_squared_error,
        metrics=['MSE'],
    )


    num_data = X_train[['log_return1','log_return2','trade_log_return1']]
    cat_data = X_train['stock_id']
    target =  y_train
    
    num_data_test = X_test[['log_return1','log_return2','trade_log_return1']]
    cat_data_test = X_test['stock_id']

    model.fit([cat_data, num_data], 
              target, 
              sample_weight = 1/np.square(target),
              batch_size=1024,
              epochs=100,
              validation_data=([cat_data_test, num_data_test], y_test, 1/np.square(y_test)),
              callbacks=[es, plateau],
              shuffle=True,
             verbose = 1)

    preds = model.predict([cat_data_test, num_data_test]).reshape(1,-1)[0]
    
    score = round(rmspe(y_true = y_test, y_pred = preds),5)
    print('Fold {} {}: {}'.format(counter, model_name, score))
    scores_folds[model_name].append(score)
    test[target_name] += model.predict([test['stock_id'], test[['log_return1','log_return2','trade_log_return1']]]).reshape(1,-1)[0].clip(0,1e10)
       
    counter += 1


# In[8]:


test[target_name] = test[target_name]/n_folds

score = round(rmspe(y_true = train[target_name].values, y_pred = train[pred_name].values),5)
print('RMSPE {}: {} - Folds: {}'.format(model_name, score, scores_folds[model_name]))

display(test[['row_id', target_name]].head(2))
test[['row_id', target_name]].to_csv('submission.csv',index = False)


# 
