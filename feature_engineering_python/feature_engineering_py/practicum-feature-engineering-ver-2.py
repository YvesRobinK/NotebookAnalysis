#!/usr/bin/env python
# coding: utf-8

# ## Kernel description

# Adding new features to the `train_meta` dataframe for one batch only:  
#   
# 1) Statistical, such as the total number of impulses or the average relative time within one event and so one;  
# 2) Predictions of different models as features (***work in progress***).    
#   
# Polars library was used for feature engineering, because it allows to process all 660 batches and 131,953,924 events many times faster than Pandas.  
#   
# This Kernel has separate functions that you can use and modify to create your own features.  
#   
# The resulting feature table is shown at the end of the notebook.  
# 
# Please don't hesitate to leave your comments on this Kernel: use the features table for your models and share the results.

# ## Updates
# 
# **Ver. 2:** Removed the use of Pandas functions to create features (now Polars only). Separate functions added for feature engineering. Added new features in the `train_meta` data as well. Feature engineering was implemented to one batch only due to memory error in all batches case.

# ## Sources

# For this Kernel, [[日本語/Eng]🧊: FeatureEngineering](https://www.kaggle.com/code/utm529fg/eng-featureengineering) kernel was used, as well as separate articles about Polars library and feature engineering:  
# 1) [📊 Построение и отбор признаков. Часть 1: feature engineering (RUS)](https://proglib.io/p/postroenie-i-otbor-priznakov-chast-1-feature-engineering-2021-09-15)  
# 2) [Polars: Pandas DataFrame but Much Faster](https://towardsdatascience.com/pandas-dataframe-but-much-faster-f475d6be4cd4)  
# 3) [Polars: calm](https://calmcode.io/polars/calm.html)  
# 4) [Polars - User Guide](https://pola-rs.github.io/polars-book/user-guide/coming_from_pandas.html) 

# ## Import libraries

# In[1]:


# List all installed packages and package versions
#!pip freeze


# In[2]:


get_ipython().system('pip install polars==0.16.8')


# In[3]:


get_ipython().system(' pip install memory_profiler')
get_ipython().run_line_magic('load_ext', 'memory_profiler')


# In[4]:


import numpy as np
import os
import pandas as pd
import polars as pl
from tqdm.notebook import tqdm


# Check types info for memory usage optimization:

# In[5]:


int_types = ["uint64", "int64"]
for it in int_types:
    print(np.iinfo(it))


# Check existing paths:

# In[6]:


# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# Will try to use Polars as one of the fastest libraries because it uses parallelization and cache efficient algorithms to speed up tasks.  
#   
# Let's create LazyFrames from all existing inputs:

# In[7]:


INPUT_DIR = '/kaggle/input/icecube-neutrinos-in-deep-ice'

train_meta = pl.scan_parquet(f'{INPUT_DIR}/train_meta.parquet')
sensor_geometry = pl.scan_csv(f'{INPUT_DIR}/sensor_geometry.csv')

batches_dict = {}

for i in range(1, train_meta.collect()['batch_id'].max() + 1):
    key=str('train_batch_'+str(i))
    batches_dict[key] = pl.scan_parquet(f'{INPUT_DIR}/train/batch_{i}.parquet')


# ## Feature engineering

# In[8]:


def add_cols_to_sensor_geometry(sensor):
    '''
    add new columns for groupby.sum() function
    
    Parameters:
    -----------
    sensor : LazyFrame
        existing 'sensor_geometry' data
    
    Returns:
    -----------
    sensor : LazyFrame
        updated 'sensor_geometry' data with additional 
        null-filled columns
    '''
    sensor=sensor.with_columns(
        [(pl.col('sensor_id') * 0).alias('sensor_count'),
         (pl.col('sensor_id') * 0.0).alias('charge_sum'),
         (pl.col('sensor_id') * 0).alias('time_sum'),
         (pl.col('sensor_id') * 0).alias('auxiliary_sum')]
    )
    
    return sensor


# In[9]:


sensor_geometry = add_cols_to_sensor_geometry(sensor_geometry).collect()
sensor_geometry.head()


# In[10]:


def add_stat_features(meta):
    '''
    add new statistics features into the selected data
    
    Parameters:
    -----------
    meta : LazyFrame
        existing 'train_meta' data
    
    Returns:
    -----------
    meta : LazyFrame
        updated 'train_meta' data with additional columns:
        * 'n_events_per_batch' (can be useful when creating 
        features for all batches at once)
        * 'pulse_count' - count of pulses detected
    '''
    return (meta
            .with_columns([
                pl.col('event_id').count().over('batch_id').cast(pl.UInt64).alias('n_events_per_batch'),
                (pl.col('last_pulse_index') - pl.col('first_pulse_index') + 1).alias('pulse_count')
             ]))


# In[11]:


train_meta = train_meta.pipe(add_stat_features).collect()
train_meta.head()


# There is not enough memory to execute these codes for all batches - the reason why I commented out this code. Perhaps after some additional work with memory in the future, I can use one of these options to create features for all batches at once.

# In[12]:


# def add_time_mean(train_meta, batches_dict):
#     batches = []
#     for batch_name, batch in tqdm(batches_dict.items()):
#         batch_id = int(batch_name.split("_")[-1])
#         batch_df = batch.select(['sensor_id', 'time', 'event_id']).collect()
#         batch_len = len(batch_df)
        
#         batch = batch_df.with_columns((pl.Series([batch_id] * batch_len)).alias('batch_id'))
#         batches.append(batch)

#     all_batches = pl.concat(batches)

#     time_mean = all_batches.groupby('event_id').agg(
#         pl.col('time').mean().alias('time_mean'))

#     train_meta_with_time_mean = train_meta.join(
#         time_mean, on='event_id', how='inner')

#     return train_meta_with_time_mean


# In[13]:


# %%time
# add_time_mean(train_meta, batches_dict)


# In[14]:


# def create_batch_1_features(batch_dict, sensor, meta):
#     '''
#     Creating new meta_total data including statistics features from batches
    
#     Parameters:
#     -----------
#     batch_dict : dictionary where keys are polars LazyFrames
#     sensor : LazyFrame
#     meta : train_meta pl.DataFrame
    
#     Returns:
#     -----------
#     sensor : polars DataFrame
#     '''
    
#     meta_tmp = pl.DataFrame()
#     meta_total = pl.DataFrame() # for output
    
#     for key in tqdm(batch_dict):
#         batch = batch_dict[key].collect()
        
#         # count detected sensor
#         batch_tmp = batch['sensor_id'].value_counts()
        
#         # cast and join
#         batch_tmp = batch_tmp.with_columns([
#             pl.col('sensor_id').cast(pl.Int64),
#             pl.col('counts').cast(pl.Int64)
#         ])
        
#         sensor = sensor.join(batch_tmp, on='sensor_id', how='left')
    
#         # groupby sensor_id and sum
#         batch_tmp = batch.select(pl.col(['sensor_id','time','charge','auxiliary'])).groupby(['sensor_id']).sum()
        
#         # cast and join
#         batch_tmp = batch_tmp.with_columns(
#             [pl.col('sensor_id').cast(pl.Int64),
#              pl.col('auxiliary').cast(pl.Int64)])
        
#         sensor = sensor.join(batch_tmp,on='sensor_id',how='left')
#         sensor = sensor.fill_null(0)
        
#         # add total value
#         sensor = sensor.with_columns(
#             [(pl.col('sensor_count')  + pl.col('counts')).alias('sensor_count'),
#              (pl.col('time_sum')      + pl.col('time')).alias('time_sum'),
#              (pl.col('charge_sum')    + pl.col('charge')).alias('charge_sum'),
#              (pl.col('auxiliary_sum') + pl.col('auxiliary')).alias('auxiliary_sum')])

#         # exclude unnecessary columns
#         sensor = sensor.select(pl.exclude(['counts','time','charge','auxiliary']))
        
#         # groupby event_id
#         batch_tmp = batch.select(pl.col(['event_id','time','charge','auxiliary'])).groupby(['event_id']).sum()
    
#         # cast and join
#         batch_tmp = batch_tmp.with_columns(pl.col('auxiliary').cast(pl.Int64))
#         meta_tmp = meta.join(batch_tmp,on='event_id',how='inner')

#         # add total value
#         meta_tmp = meta_tmp.with_columns(
#             [(pl.col('time')).alias('time_sum'),
#              (pl.col('charge')).alias('charge_sum'),
#              (pl.col('auxiliary')).alias('auxiliary_sum')])

#         # exclude unnecessary columns
#         meta_tmp = meta_tmp.select(pl.exclude(['time','charge','auxiliary']))

#         # append to output
#         meta_total = pl.concat([meta_total, meta_tmp])
    
#     return meta_total


# In[15]:


def create_batch_features(batch_dict, key, sensor, meta):
    '''
    Creating new meta_total data including statistics features 
    for one selected batch only
    
    Parameters:
    -----------
    batch_dict : dict 
        keys - str, values - LazyFrames
    key : str
        name of batch
    sensor : LazyFrame
    meta : polars DataFrame
        existing 'train_meta' data
    
    Returns:
    -----------
    sensor : polars DataFrame
    meta_total : polars DataFrame
    '''
    # for output
    meta_tmp = pl.DataFrame()
    meta_total = pl.DataFrame() 
    
    batch = batch_dict[key].collect()
    
    # count detected sensor
    batch_tmp = batch['sensor_id'].value_counts()
        
    # cast and join
    batch_tmp = batch_tmp.with_columns([
        pl.col('sensor_id').cast(pl.Int64),
        pl.col('counts').cast(pl.Int64)
    ])
         
    sensor = sensor.join(batch_tmp, on='sensor_id', how='left')
    
    # groupby sensor_id and sum
    batch_tmp = batch.select(pl.col(['sensor_id','time','charge','auxiliary'])).groupby(['sensor_id']).sum()
        
    # cast and join
    batch_tmp = batch_tmp.with_columns(
        [pl.col('sensor_id').cast(pl.Int64),
         pl.col('auxiliary').cast(pl.Int64)])
        
    sensor = sensor.join(batch_tmp,on='sensor_id',how='left')
    sensor = sensor.fill_null(0)
        
    # add total value
    sensor = sensor.with_columns(
        [(pl.col('sensor_count')  + pl.col('counts')).alias('sensor_count'),
         (pl.col('time_sum')      + pl.col('time')).alias('time_sum'),
         (pl.col('charge_sum')    + pl.col('charge')).alias('charge_sum'),
         (pl.col('auxiliary_sum') + pl.col('auxiliary')).alias('auxiliary_sum')])

    # exclude unnecessary columns
    sensor = sensor.select(pl.exclude(['counts','time','charge','auxiliary']))
        
    # groupby event_id
    batch_tmp = batch.select(pl.col(['event_id','time','charge','auxiliary'])).groupby(['event_id']).sum()
    
    # cast and join
    batch_tmp = batch_tmp.with_columns(pl.col('auxiliary').cast(pl.Int64))
    meta_tmp = meta.join(batch_tmp,on='event_id',how='inner')

    # add total value
    meta_tmp = meta_tmp.with_columns(
        [(pl.col('time')).alias('time_sum'),
         (pl.col('charge')).alias('charge_sum'),
         (pl.col('auxiliary')).alias('auxiliary_sum')])

    # exclude unnecessary columns
    meta_tmp = meta_tmp.select(pl.exclude(['time','charge','auxiliary']))

    # append to output
    meta_total = pl.concat([meta_total, meta_tmp])
    
    return sensor, meta_total


# In[16]:


get_ipython().run_line_magic('memit', "sensor_geometry, meta_batch = create_batch_features(batches_dict, 'train_batch_1', sensor_geometry, train_meta)")

meta_batch.head()


# In[17]:


# feature engineering

def feature_engineering(sensor, meta_total):
    '''
    Creating new meta_total data including statistics features
    for one selected batch only
    
    Parameters:
    -----------
    sensor : polars DataFrame
    meta_total : polars DataFrame
    
    Returns:
    -----------
    sensor : polars DataFrame
    meta_total : polars DataFrame
    '''
    sensor = sensor.with_columns(
            [(pl.col('sensor_count')  / len(meta_total)).alias('sensor_count_mean'),
             (pl.col('time_sum')      / pl.col('sensor_count')).alias('time_mean'),
             (pl.col('charge_sum')    / pl.col('sensor_count')).alias('charge_mean'),
             (pl.col('auxiliary_sum') / pl.col('sensor_count')).alias('auxiliary_ratio')])
    meta_total = meta_total.with_columns(
            [(pl.col('time_sum')      / pl.col('pulse_count')).alias('time_mean'),
             (pl.col('charge_sum')    / pl.col('pulse_count')).alias('charge_mean'),
             (pl.col('auxiliary_sum') / pl.col('pulse_count')).alias('auxiliary_ratio')])

    # select and sort columns
    sensor = sensor.select(pl.col(['sensor_id',
                                   'x',
                                   'y',
                                   'z',
                                   'sensor_count',
                                   'sensor_count_mean',
                                   'time_mean',
                                   'charge_mean',
                                   'auxiliary_ratio'])
                          )
    meta_total = meta_total.select(pl.col(['batch_id',
                                           'event_id',
                                           'first_pulse_index',
                                           'last_pulse_index',
                                           'azimuth',
                                           'zenith',
                                           'n_events_per_batch',
                                           'pulse_count',
                                           'charge_sum',
                                           'auxiliary_sum',
                                           'time_mean',
                                           'charge_mean',
                                           'auxiliary_ratio'])
                                  )
    return sensor, meta_total


# In[18]:


sensor_geometry, meta_batch = feature_engineering(sensor_geometry, meta_batch)

display(sensor_geometry.head())
meta_batch.head()


# <font color='red'>
# If you know how to reduce memory usage for above code, please feel free to share your ideas in the comments of this notebook. I've found the good notebook:</font>    
# [14 Simple Tips to save RAM memory for 1+GB dataset](https://www.kaggle.com/code/pavansanagapati/14-simple-tips-to-save-ram-memory-for-1-gb-dataset/notebook)  
# <font color='red'>
# Tried to change Int64 to UInt64 to reduce memory a little, but it need more precise work with all data to be sure that all data fit same type.
# </font>

# ## 
