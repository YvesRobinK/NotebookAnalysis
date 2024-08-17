#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import polars as pl
import gc
from tqdm.auto import tqdm 
import joblib
from datetime import timedelta, date



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


# def feature_extraction(df, istrain = True, step_rolling = 5):
#     ids = df['series_id'].unique(maintain_order = True)
    
#     if (istrain):
#         df = df.with_columns(
#             awake = df["awake"].cast(pl.Int8)
#         )

    
#     df = df.with_columns(        
#         year = (df["timestamp"].str.slice(0,4).cast(pl.Int16) - 2000).cast(pl.UInt8),
#         month = df["timestamp"].str.slice(5,2).cast(pl.Int8),
#         day = df["timestamp"].str.slice(8,2).cast(pl.Int8),
#         hour = df["timestamp"].str.slice(11,2).cast(pl.Int8),
#         minute = df["timestamp"].str.slice(14,2).cast(pl.Int8),
# #         second = df["timestamp"].str.slice(17,2).cast(pl.Int8),
#         time_zone = df["timestamp"].str.slice(-5,3).cast(pl.Int32),
        
#         enmo =  (pl.col('enmo')*1000).cast(pl.UInt16),
        
#         weekday = train["timestamp"].str.slice(0,10).str.to_date().dt.weekday()
        
#     )
    
#     signal_awake = dict(zip(range(1440), np.sin(np.linspace(0, np.pi, 1440) + 0.208 * np.pi) ** 24))
#     signal_onset = dict(zip(range(1440), np.sin(np.linspace(0, np.pi, 1440) + 0.555 * np.pi) ** 24))
#     df = df.with_columns(
#         (pl.col("hour").cast(pl.Int32)*60 + pl.col("minute").cast(pl.Int32))
#         .apply(signal_onset.get, return_dtype=pl.Float32)
#         .alias("signal_onset")
#     )    
#     df = df.with_columns(
#         (pl.col("hour").cast(pl.Int32)*60 + pl.col("minute").cast(pl.Int32))
#         .apply(signal_awake.get, return_dtype=pl.Float32)
#         .alias("signal_awake")
#     )

#     df = df.with_columns(lids = np.maximum(0., tmp['enmo'] - 0.02))
#     df = df.with_columns(
#         lids = pl.col('lids').rolling_sum(121, center=True, min_periods=1)
#     )
#     df = df.with_columns(lids = 100 / (pl.col('lids') + 1))
#     df = df.with_columns(
#         lids = pl.col('lids').rolling_mean(361, center=True, min_periods=1).cast(pl.Float32)
#     )
#     gc.collect()
#     return df

# train = feature_extraction(train)
# train
# # list(train['signal_onset'][:10])


# In[3]:


def feature_extraction(df, istrain = True, step_rolling = 5):    
# def feature_extraction(df, istrain = True):    

    ids = df['series_id'].unique(maintain_order = True)
    
    if (istrain):
        df = df.with_columns(
            awake = df["awake"].cast(pl.Int8)
        )

    df = df.with_columns(
        lids = np.maximum(0., df['enmo'] - 0.02),
        year = (df["timestamp"].str.slice(0,4).cast(pl.Int16) - 2000).cast(pl.UInt8),
        month = df["timestamp"].str.slice(5,2).cast(pl.Int8),
        day = df["timestamp"].str.slice(8,2).cast(pl.Int8),
        hour = df["timestamp"].str.slice(11,2).cast(pl.Int8),
        minute = df["timestamp"].str.slice(14,2).cast(pl.Int8),
        second = df["timestamp"].str.slice(17,2).cast(pl.Int8),
        time_zone = df["timestamp"].str.slice(-5,3).cast(pl.Int32),
        weekday = df["timestamp"].str.slice(0,10).str.to_date().dt.weekday(),

        enmo =  (pl.col('enmo')*1000).cast(pl.UInt16),
        )
    
    signal_awake = dict(zip(range(1440), np.sin(np.linspace(0, np.pi, 1440) + 0.208 * np.pi) ** 24))
    signal_onset = dict(zip(range(1440), np.sin(np.linspace(0, np.pi, 1440) + 0.555 * np.pi) ** 24))
    df = df.with_columns(
        (pl.col("hour").cast(pl.Int32)*60 + pl.col("minute").cast(pl.Int32))
        .apply(signal_onset.get, return_dtype=pl.Float32)
        .alias("signal_onset"),
        (pl.col("hour").cast(pl.Int32)*60 + pl.col("minute").cast(pl.Int32))
        .apply(signal_awake.get, return_dtype=pl.Float32)
        .alias("signal_awake"),
        (df["hour"].cast(pl.Int32)*60 + df["minute"].cast(pl.Int32)).alias('hour_minute')
    )
    
    features, feature_cols = [], []

    for mins in [1, 3, 5, 30, 60*2, 60*8] :
        for var in ['enmo', 'anglez'] :
            features += [
                pl.col(var).rolling_mean(12 * mins, center=True, min_periods=1).abs().cast(pl.UInt16).alias(f'{var}_{mins}m_mean'),
                pl.col(var).rolling_max(12 * mins, center=True, min_periods=1).abs().cast(pl.UInt16).alias(f'{var}_{mins}m_max'),
                pl.col(var).rolling_std(12 * mins, center=True, min_periods=1).abs().cast(pl.UInt16).alias(f'{var}_{mins}m_std')
            ]
            feature_cols += [ 
                f'{var}_{mins}m_mean', f'{var}_{mins}m_max', f'{var}_{mins}m_std'
            ]
            # Getting first variations
            features += [
                (pl.col(var).diff().abs().rolling_mean(12 * mins, center=True, min_periods=1)*10).abs().cast(pl.UInt32).alias(f'{var}_1v_{mins}m_mean'),
                (pl.col(var).diff().abs().rolling_max(12 * mins, center=True, min_periods=1)*10).abs().cast(pl.UInt32).alias(f'{var}_1v_{mins}m_max'),
                (pl.col(var).diff().abs().rolling_std(12 * mins, center=True, min_periods=1)*10).abs().cast(pl.UInt32).alias(f'{var}_1v_{mins}m_std')
            ]
            feature_cols += [ 
                f'{var}_1v_{mins}m_mean', f'{var}_1v_{mins}m_max', f'{var}_1v_{mins}m_std'
            ]
    # id_cols = ['series_id', 'step', 'timestamp']

    dict_ids2data = {}    
    for i in tqdm(range(len(ids))):
    # for i in range(2):
        print (i,': ', ids[i])
        dict_ids2data[ids[i]] = df.filter(pl.col("series_id") == ids[i])
        dict_ids2data[ids[i]] = dict_ids2data[ids[i]].with_columns(
            features,
            lids = pl.col('lids').rolling_sum(121, center=True, min_periods=1)
        )
                
        dict_ids2data[ids[i]] = dict_ids2data[ids[i]].with_columns(
            (100 / (pl.col('lids') + 1)).alias('lids'),
            pl.col("anglez").rolling_mean(window_size = step_rolling, center = True).cast(pl.Float32).alias(f'anglez_{step_rolling*5}s_smooth'),
            pl.col("enmo").rolling_mean(window_size = step_rolling, center = True).cast(pl.Float32).alias(f'enmo_{step_rolling*5}s_smooth')
        )
        
        dict_ids2data[ids[i]] = dict_ids2data[ids[i]].fill_null(strategy="forward").fill_null(strategy="backward")
        
        dict_ids2data[ids[i]] = dict_ids2data[ids[i]].with_columns(
            lids = pl.col('lids').rolling_mean(361, center=True, min_periods=1).cast(pl.Float32),
#             anglez_lag_5s = dict_ids2data[ids[i]]['anglez'].shift(periods=1),
#             anglez_lag_10s = dict_ids2data[ids[i]]['anglez'].shift(periods=2),
#             anglez_lag_15s = dict_ids2data[ids[i]]['anglez'].shift(periods=3),
#             anglez_lag_30s = dict_ids2data[ids[i]]['anglez'].shift(periods=6),
#             anglez_lag_60s = dict_ids2data[ids[i]]['anglez'].shift(periods=12),
            anglez_smooth_lag_30s = dict_ids2data[ids[i]][f'anglez_{step_rolling*5}s_smooth'].cast(pl.Float32).shift(periods=6),
            anglez_smooth_lag_60s = dict_ids2data[ids[i]][f'anglez_{step_rolling*5}s_smooth'].cast(pl.Float32).shift(periods=12),

#             enmo_lag_5s = dict_ids2data[ids[i]]['enmo'].shift(periods=1),
#             enmo_lag_10s = dict_ids2data[ids[i]]['enmo'].shift(periods=2),
#             enmo_lag_15s = dict_ids2data[ids[i]]['enmo'].shift(periods=3),
#             enmo_lag_30s = dict_ids2data[ids[i]]['enmo'].shift(periods=6),
#             enmo_lag_60s = dict_ids2data[ids[i]]['enmo'].shift(periods=12),
            enmo_smooth_lag_30s = dict_ids2data[ids[i]][f'enmo_{step_rolling*5}s_smooth'].cast(pl.Float32).shift(periods=6),
            enmo_smooth_lag_60s = dict_ids2data[ids[i]][f'enmo_{step_rolling*5}s_smooth'].cast(pl.Float32).shift(periods=12)
        )
        
        dict_ids2data[ids[i]] = dict_ids2data[ids[i]].fill_null(strategy="forward").fill_null(strategy="backward")

        
    df = pl.concat([i for i in dict_ids2data.values()], rechunk=True)

    del dict_ids2data
    gc.collect()

    return df



# In[4]:


get_ipython().run_cell_magic('time', '', "train = pl.read_parquet('/kaggle/input/zzzs-lightweight-training-dataset-target/Zzzs_train_multi.parquet')\ntrain.head()\n")


# In[5]:


train_fe = feature_extraction(train)

train_fe


# In[6]:


c_types = train_fe.dtypes
c_name = train_fe.columns

print(len(c_name))
print('\n'.join(["{}\t-\t{}".format(str(a_), str(b_)) for a_, b_ in zip(c_name, c_types)]))


# In[7]:


get_ipython().run_cell_magic('time', '', "test_raw = pl.read_parquet('/kaggle/input/child-mind-institute-detect-sleep-states/test_series.parquet')\ntest_raw.head()\n")


# In[8]:


test_fe = feature_extraction(test_raw, False)

test_fe


# In[9]:


gc.collect()


# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["figure.figsize"] = [12.00, 5]
plt.rcParams["figure.autolayout"] = True

df = pd.DataFrame(dict(day=np.array(train_fe["day"].take_every(12*60))))

ax = sns.countplot(x="day", data=df)

for p in ax.patches:
   ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))

plt.show()


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["figure.figsize"] = [12.00, 5]
plt.rcParams["figure.autolayout"] = True

df = pd.DataFrame(dict(month=np.array(train_fe["month"].take_every(12*60*24))))

ax = sns.countplot(x="month", data=df)

for p in ax.patches:
   ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))

plt.show()


# In[12]:


get_ipython().run_cell_magic('time', '', "test_fe.write_csv('test_series.csv')\ntest_fe.write_parquet('test_series.parquet')\ntrain_fe.write_parquet('train_series.parquet')\n")


# In[13]:


# data_dtype = {
#         "series_id": "category",
#         "step": "uint32",
#         "timestamp": "str",
#         "anglez": "float32",
#         "enmo": "float32",
#     }
# data = pd.read_parquet('/kaggle/input/child-mind-institute-detect-sleep-states/test_series.parquet').astype(data_dtype, copy=False)


# In[ ]:





# In[ ]:




