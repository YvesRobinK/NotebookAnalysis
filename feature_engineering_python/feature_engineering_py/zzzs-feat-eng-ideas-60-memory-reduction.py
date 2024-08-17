#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_ns_dtype
import gc

import warnings
warnings.filterwarnings("ignore")


# I used the the training data from the reduced dataset ["Zzzs: Lightweight training dataset + target"](https://www.kaggle.com/datasets/carlmcbrideellis/zzzs-lightweight-training-dataset-target)
# 
# ### Feature Engineering
# 
# Feature engineering is the backbone of effective AI models. It involves crafting meaningful input variables that enable models to learn and make accurate predictions. Well-engineered features extract crucial information from raw data, enhancing model performance and interpretability. In AI, the saying "garbage in, garbage out" holds true; thoughtful feature engineering transforms data into valuable insights, making it an indispensable step in building robust and effective models. However, it's important to note that feature engineering is a highly context-dependent process.
# 
# 
# This notebook performs feature engineering by adding various derived features. Here's a description of some ideas:
# 
# -**Timestamp Derivatives**: It extracts time-related information from the 'timestamp' column, including hour, weekday, whether it's a weekend, month, and day.
# 
# -**Feature Cross**: It creates a new feature by multiplying 'anglez' and 'enmo' columns.
# 
# -**Difference**: For the 'enmo' and 'anglez' columns, it calculates difference for different time periods.
# 
# -**Rolling Statistics**: For the 'enmo' and 'anglez' columns, it calculates various rolling statistics like mean, median, max, min, skewness, and kurtosis for different time periods.
# 
# -**Time Lag and Lead**: It creates lag and lead features for 'enmo' and 'anglez' columns, allowing the model to consider past and future values.
# 
# This notebook serves as a template, demonstrating some feature engineering techniques that can be adapted and extended. The data with this feaure engineering can e download [here](https://www.kaggle.com/datasets/renatoreggiani/zzzs-training-dataset-target-feat-eng)

# In[2]:


train = pd.read_parquet('/kaggle/input/zzzs-lightweight-training-dataset-target/Zzzs_train_multi.parquet')


# In[3]:


def reduce_mem_usage(df):
    
    """ 
    Iterate through all numeric columns of a dataframe and modify the data type
    to reduce memory usage.        
    """
    
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage of dataframe is {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and not is_datetime64_ns_dtype(df[col]):
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
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        
    df['series_id'] = df['series_id'].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage after optimization is: {end_mem:.2f} MB')
    decrease = 100 * (start_mem - end_mem) / start_mem
    print(f'Decreased by {decrease:.2f}%')
    
    return df


# In[4]:


signal_awake = dict(zip(range(1440), np.sin(np.linspace(0, np.pi, 1440) + 0.208 * np.pi) ** 24))
signal_onset = dict(zip(range(1440), np.sin(np.linspace(0, np.pi, 1440) + 0.555 * np.pi) ** 24))


def feat_eng(df):
    
    print('Start Feat Eng')
    df['timestamp'] = pd.to_datetime(df['timestamp']).apply(lambda t: t.tz_localize(None))
    df.sort_values(['series_id', 'timestamp'], inplace=True)
    
    df['hour'] = df['timestamp'].dt.hour.astype(np.int8)
    
    if df['hour'].nunique()==24:
        df = pd.concat([df, pd.get_dummies(df['hour'], dtype=np.int8, prefix='hour', drop_first=True)], axis=1)
    else:
        for h in range(1, 24):
            in_hour = 1 if h in df['hour'].unique() else 0
            df[f'hour_{h}'] = in_hour
    
    df['month'] = df['timestamp'].dt.month.astype(np.int8)
    
    df['signal_onset'] = (df.timestamp.dt.hour * 60 + df.timestamp.dt.minute).map(signal_onset).astype(np.float32)
    df['signal_awake'] = (df.timestamp.dt.hour * 60 + df.timestamp.dt.minute).map(signal_awake).astype(np.float32)
    df["anglez"] = abs(df["anglez"])
    
    print('Created timestamp derivates')
    gc.collect()
    
    # feature cross
    df['anglez_x_enmo'] = df['anglez'] * df['enmo']
    df['anglezabs_x_enmo'] = abs(df['anglez']) * df['enmo']
    
    
    z = np.maximum(0., df.enmo - 0.02)
    z = z.to_frame(name='lids').assign(series_id=df.series_id)
    z = z.groupby('series_id')['lids'].rolling(120, center=True).agg('sum')
    z = 100 / (z + 1)
    z = z.to_frame().reset_index()
    z = z.groupby('series_id')['lids'].rolling(360, center=True).agg('mean')
    df['lids'] = z.astype(np.float32).values
    
    for col in ['enmo', 'anglez', 'anglez_x_enmo']:
        for n in [21, 61, 121]:
            
            for agg in ['median', 'mean', 'max', 'min', 'std']:
            
                df[f'{col}_{agg}_{n}'] = df.groupby('series_id')[col].rolling(n, center=True).agg(agg).astype(np.float32).values
                gc.collect()
            
            df[f'{col}_diff_{n}'] = df.groupby('series_id')[col].diff(periods=n).astype(np.float32)
            df[f'{col}_emavg_{n}'] = df[col].ewm(span=n).mean().astype(np.float32)
            
            
            df[f'{col}_mad_{n}'] = (df[col] - df[f'{col}_median_{n}']).abs()
            df[f'{col}_mad_{n}'] = df.groupby('series_id')[f'{col}_mad_{n}'].rolling(n, center=True).agg('median').astype(np.float32).values
            
#             df[f'{col}_skew_{n}'] = df.groupby('series_id')[col].rolling(n).skew().reset_index(drop=True).astype(np.float32)
#             df[f'{col}_kurt_{n}'] = df.groupby('series_id')[col].rolling(n).kurt().reset_index(drop=True).astype(np.float32)
            
            df[f'{col}_amplit_{n}'] = df[f'{col}_max_{n}']-df[f'{col}_min_{n}'].astype(np.float32)
            
            df[f'{col}_diff_{n}_mean'] = df[f'{col}_diff_{n}'].rolling(n,center=True).mean().astype(np.float32)
            df[f'{col}_diff_{n}_max'] = df[f'{col}_max_{n}'].rolling(n,center=True).max().astype(np.float32)
    
            gc.collect()
            
            
        print(f'Created diff and smoothed derivates from {col}')

    gc.collect() 


    df.bfill(inplace=True, limit=15)
    df.ffill(inplace=True, limit=15)
    df.dropna(inplace=True)

    df = reduce_mem_usage(df)

    return df


# In[5]:


get_ipython().run_cell_magic('time', '', "\ntrain['awake'] = train['awake'].astype(np.int8)\ntrain = feat_eng(train)\n")


# In[6]:


train.shape


# In[7]:


train.to_parquet('train_plus_feat.parquet', compression='gzip')


# In[8]:


test  = pd.read_parquet('/kaggle/input/child-mind-institute-detect-sleep-states/test_series.parquet')


# In[9]:


get_ipython().run_cell_magic('time', '', "\ntest['timestamp'] = pd.to_datetime(test['timestamp'],utc=True)\ntest = feat_eng(test)\n")


# In[10]:


train['awake'].value_counts()


# In[ ]:




