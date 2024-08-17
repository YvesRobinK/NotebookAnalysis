#!/usr/bin/env python
# coding: utf-8

# # GRU with Simplified Feature Engineering

# In this notebook, we experiment with using solely target values as the primary feature, eliminating the need for complex feature engineering. This method simplifies the modeling process, allowing us to concentrate directly on the inherent patterns and trends within the target data itself.

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random as python_random

from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, GRU, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


# In[2]:


SEED = 42

N_LAGS = 55

BATCH_SIZE = 32
EPOCHS = 10000
PATIENCE = 25
DROPOUT = 0.5
LEARNING_RATE = 1e-4

SPLIT_DAY = 390

N_STOCKS = 200
N_DATES = 481
N_SECONDS = 55

RUN_TRAINING = True
RUN_FOR_SUBMISSION = True


# In[3]:


os.environ['PYTHONHASHSEED'] = str(SEED)
tf.keras.utils.set_random_seed(SEED)


# In[4]:


df = pd.read_csv("/kaggle/input/optiver-trading-at-the-close/train.csv")
df = df[["stock_id", "date_id", "seconds_in_bucket", "target"]]
df.shape


# In[5]:


def reduce_mem_usage(df, verbose=0):
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
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
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)

    print(f"Memory usage of dataframe is {start_mem:.2f} MB")
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after optimization is: {end_mem:.2f} MB")
    decrease = 100 * (start_mem - end_mem) / start_mem
    print(f"Decreased by {decrease:.2f}%")

    return df

df = reduce_mem_usage(df, verbose=1)


# The following step is to create a DataFrame that includes all possible combinations of stock IDs, 
# date IDs, and time intervals (in seconds). This ensures that the DataFrame contains 
# a complete set of data across the specified ranges and intervals, with any missing 
# data points filled with zeros.
# 

# In[6]:


all_stock_ids = range(N_STOCKS)
all_date_ids = range(N_DATES)
all_seconds = [i * 10 for i in range(N_SECONDS)]

multi_index = pd.MultiIndex.from_product([all_stock_ids, all_date_ids, all_seconds], 
                                         names=['stock_id', 'date_id', 'seconds_in_bucket'])

df_full = df.set_index(['stock_id', 'date_id', 'seconds_in_bucket']).reindex(multi_index)
df_full = df_full.fillna(0)
df_full = df_full.reset_index()

assert(df_full.shape[0] == N_STOCKS * N_DATES * N_SECONDS)

df_full


# This function `windowed_dataset` takes a time series data, converts it into a TensorFlow Dataset, 
# and processes it into overlapping windows. Each window contains a sequence of data 
# points for the model to learn from (features) and the next data point as the target 
# (label). The dataset is then batched and prefetched for efficient training.

# In[7]:


def windowed_dataset(series, window_size=N_LAGS, batch_size=BATCH_SIZE):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    return ds.batch(batch_size).prefetch(1)


# 
# The function `build_features` takes a DataFrame containing time series data for various stocks and 
# transforms it into a pivoted format suitable for time series analysis and modeling. 
# The function ensures comprehensive coverage by including all combinations of stock IDs, 
# date IDs, and time intervals, and then restructures the data into a pivot table format.

# In[8]:


def build_features(df):

    all_stock_ids = range(N_STOCKS)
    all_date_ids = df["date_id"].unique()
    all_seconds = [i * 10 for i in range(N_SECONDS)]
    
    multi_index = pd.MultiIndex.from_product([all_stock_ids, all_date_ids, all_seconds], 
                                             names=['stock_id', 'date_id', 'seconds_in_bucket'])
    df_full = df.set_index(['stock_id', 'date_id', 'seconds_in_bucket']).reindex(multi_index)
    df_full = df_full.fillna(0)
    df_full = df_full.reset_index()
    
    df_pivoted = df_full.pivot_table(
                values='target', 
                index=['date_id', 'seconds_in_bucket'], 
                columns='stock_id')

    df_pivoted = df_pivoted.reset_index(drop=True)
    df_pivoted.columns.name = None
    
    return df_pivoted

build_features(df_full)


# In[9]:


def build_model(dropout=DROPOUT):
    model = Sequential()
    model.add(Input(shape=(N_LAGS, N_STOCKS)))
    model.add(Dropout(dropout))
    model.add(GRU(25, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(N_STOCKS))
    model.compile(loss='mae',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
    return model


# # Training

# In[10]:


get_ipython().run_cell_magic('time', '', 'if RUN_TRAINING:\n    \n    split = df_full[\'date_id\'] > SPLIT_DAY\n    df_train = df_full[~split]\n    df_valid = df_full[split]\n    \n    df_train_features = build_features(df_train)\n    df_valid_features = build_features(df_valid)\n\n    scaler = StandardScaler()\n    train_features = scaler.fit_transform(df_train_features)\n    valid_features = scaler.transform(df_valid_features)\n \n    train_dataset = windowed_dataset(train_features, N_LAGS, BATCH_SIZE)\n    valid_dataset = windowed_dataset(valid_features, N_LAGS, BATCH_SIZE)\n\n    model = build_model()\n\n    early_stopping = EarlyStopping(monitor=\'val_loss\',\n                      mode=\'min\',\n                      patience=PATIENCE,\n                      restore_best_weights=True,\n                      verbose=True)\n\n    history = model.fit(train_dataset,\n                        validation_data=valid_dataset,\n                        epochs=EPOCHS,\n                        batch_size=BATCH_SIZE,\n                        callbacks=[early_stopping],\n                        verbose=True)\n\n    ## Evaluate ## \n    y_pred = model.predict(valid_dataset)\n\n    y_pred = scaler.inverse_transform(y_pred)\n    y_true = df_valid_features[N_LAGS:]\n\n    mae = mean_absolute_error(y_true, y_pred)\n    print(f"MAE score: {mae}")\n\n    ## Plots ##\n    plt.plot(history.history[\'loss\'])\n    plt.plot(history.history[\'val_loss\'])\n    plt.title(\'model train vs validation loss\')\n    plt.ylabel(\'loss\')\n    plt.xlabel(\'epoch\')\n    plt.legend([\'train\', \'validation\'], loc=\'upper right\')\n    plt.show()\n')


# # Submission 

# In[11]:


if RUN_FOR_SUBMISSION:
    
    import optiver2023
    optiver2023.make_env.func_dict['__called__'] = False
    env = optiver2023.make_env()
    iter_test = env.iter_test()
    
    counter = 0
    
    for i, (test, revealed_targets, sample_prediction) in enumerate(iter_test):
        
        if test.currently_scored.iloc[0]== False:
            sample_prediction['target'] = 0
            env.predict(sample_prediction)
            counter += 1
            continue        
            
        if test.seconds_in_bucket.unique()[0] == 0:

            df_revealed_targets = revealed_targets[["stock_id", "revealed_date_id", "seconds_in_bucket", "revealed_target"]]
            df_revealed_targets = df_revealed_targets.rename(columns={'revealed_date_id': 'date_id', 'revealed_target': 'target'})

            df_features = build_features(df_revealed_targets)

            history_scaled = scaler.transform(df_features)

        y_pred_scaled = model.predict(
            history_scaled[-N_LAGS:][np.newaxis, :, :],
            verbose=True)
        
        y_pred = scaler.inverse_transform(y_pred_scaled)
        
        sample_prediction['target'] = y_pred[0]
        env.predict(sample_prediction)
        counter += 1
        
        history_scaled = np.concatenate([history_scaled, y_pred_scaled])

else:
    print("Run for submission skipped")

