#!/usr/bin/env python
# coding: utf-8

# # Objective
# 
# In this notebook, I have optimised some of the preprocessing methods that you can find [here](https://www.kaggle.com/code/raimondomelis/preprocessing-of-data-in-chunks-right-way). In the previous notebook I explained the theory, but now we will see how to perform preprocessing with the GPU accelerator. In particular, we are going to compare GPU P100 vs GPU T4 x2. 
# 
# **TIP:** never perform all preprocessing on the GPU (in Kaggle it is limited), but it is convenient to also use the CPU simultaneously.

# # Calculations performed
# 
# I performed some of the standard feature engineering calculations: **count, last, nunique**

# # Importing Libraries

# In[1]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


import gc
import operator as op
import numpy as np
import cupy as cp
import pandas as pd
from tqdm.auto import tqdm
import cudf
import time


import warnings 
warnings.filterwarnings('ignore')


# # GPU P100 
# 
# In the right-hand panel, set P100 as GPU accelerator

# In[3]:


#Dividing our dataset into N° parts
num_parts = 4
def read_preprocess_divide(num_parts):
    #wanted columns
    columns = ['ip', 'channel', 'click_time']
    dtypes = {
             'ip'      : 'int32',
             'channel' : 'int16',
             'click_time' : 'datetime64[us]',
             }
    df = cudf.read_csv('../input/talkingdata-adtracking-fraud-detection/train.csv', usecols=columns, dtype=dtypes)
    all_rows = len(df)
    chunk = all_rows//num_parts
    #sort the dataset by ip and reset the index
    df = df.sort_values(by=['ip', 'click_time']).reset_index(drop = True)
    return df, all_rows, chunk 
    

def window(df):
    #calculate the most common value with the "mode", and the "window"
    most_common = df['ip'].mode().values.tolist()[0]
    window = len(df[df['ip'] == most_common])+1
    return window

def feature_engineering(df,start,new_end):
    if new_end is not None:
        end = new_end+1
    else:
        end = None
    features = [c for c in list(df.columns) if c not in ['ip','click_time']]
    cat_function = ['count', 'last', 'nunique']    
    new_chunk = df[start:end].groupby('ip')[features].agg(cat_function)
    new_chunk.columns = ['_'.join(x) for x in new_chunk.columns]
    new_chunk.reset_index(inplace = True)
    diff_num_features = [f'diff_{col}' for col in features]
    df = df.to_pandas()
    ips = df[start:end]['ip'].values
    new_chunk_diff = df[start:end].groupby('ip')[features].diff().add_prefix('diff_')
    new_chunk_diff.insert(0,'ip',ips)
    new_chunk_diff = cudf.DataFrame(new_chunk_diff)
    new_chunk_diff = new_chunk_diff.groupby('ip')[diff_num_features].agg(cat_function)
    new_chunk_diff.columns = ['_'.join(x) for x in new_chunk_diff.columns]
    new_chunk_diff.reset_index(inplace = True)
    new_chunk = new_chunk.merge(new_chunk_diff, how = 'inner', on = 'ip')
    new_chunk = new_chunk.sort_values(by=['ip']).reset_index(drop = True)
    return new_chunk


# In[4]:


get_ipython().run_cell_magic('time', '', 'df, all_rows, chunk = read_preprocess_divide(num_parts)\n')


# In[5]:


get_ipython().run_cell_magic('time', '', '#function to select a safe window of rows\nwindow = window(df)\n#new dataframe to append the results of the for loop\nnew_df=cudf.DataFrame()\n#set start = 0\nstart = 0\nfor p in range(0,num_parts):\n    end = p*chunk + chunk\n    if end < all_rows:\n        chunk_window = df[start:end].tail(window)\n        second_last_unique = chunk_window[\'ip\'].unique().values.tolist()[-2]\n        new_end = chunk_window[chunk_window[\'ip\'] == second_last_unique].tail(1).index[0]\n        print(f"Processing {(new_end+1)-start} rows of chunk N° {p+1}")\n        new_chunk = feature_engineering(df,start,new_end)\n    else:\n        print(f"Processing {all_rows-(new_end+1)} rows of chunk N° {p+1}")\n        new_chunk = feature_engineering(df,start,None)\n    start = new_end+1\n    new_df = new_df.append(new_chunk, ignore_index=True)\n')


# In[6]:


new_df


# # GPU T4 x2
# 
# In the right-hand panel, set T4 x2 as GPU accelerator

# # Importing Libraries

# In[1]:


import gc
import operator as op
import numpy as np
import cupy as cp
import pandas as pd
from tqdm.auto import tqdm
import cudf
import time


import warnings 
warnings.filterwarnings('ignore')


# In[2]:


#Dividing our dataset into N° parts
num_parts = 4
def read_preprocess_divide(num_parts):
    #wanted columns
    columns = ['ip', 'channel', 'click_time']
    dtypes = {
             'ip'      : 'int32',
             'channel' : 'int16',
             'click_time' : 'datetime64[us]',
             }
    df = cudf.read_csv('../input/talkingdata-adtracking-fraud-detection/train.csv', usecols=columns, dtype=dtypes)
    all_rows = len(df)
    chunk = all_rows//num_parts
    #sort the dataset by ip and reset the index
    df = df.sort_values(by=['ip', 'click_time']).reset_index(drop = True)
    return df, all_rows, chunk 
    

def window(df):
    #calculate the most common value with the "mode", and the "window"
    most_common = df['ip'].mode().values.tolist()[0]
    window = len(df[df['ip'] == most_common])+1
    return window

def feature_engineering(df,start,new_end):
    if new_end is not None:
        end = new_end+1
    else:
        end = None
    features = [c for c in list(df.columns) if c not in ['ip','click_time']]
    cat_function = ['count', 'last', 'nunique']    
    new_chunk = df[start:end].groupby('ip')[features].agg(cat_function)
    new_chunk.columns = ['_'.join(x) for x in new_chunk.columns]
    new_chunk.reset_index(inplace = True)
    diff_num_features = [f'diff_{col}' for col in features]
    df = df.to_pandas()
    ips = df[start:end]['ip'].values
    new_chunk_diff = df[start:end].groupby('ip')[features].diff().add_prefix('diff_')
    new_chunk_diff.insert(0,'ip',ips)
    new_chunk_diff = cudf.DataFrame(new_chunk_diff)
    new_chunk_diff = new_chunk_diff.groupby('ip')[diff_num_features].agg(cat_function)
    new_chunk_diff.columns = ['_'.join(x) for x in new_chunk_diff.columns]
    new_chunk_diff.reset_index(inplace = True)
    new_chunk = new_chunk.merge(new_chunk_diff, how = 'inner', on = 'ip')
    new_chunk = new_chunk.sort_values(by=['ip']).reset_index(drop = True)
    return new_chunk


# In[3]:


get_ipython().run_cell_magic('time', '', 'df, all_rows, chunk = read_preprocess_divide(num_parts)\n')


# In[4]:


get_ipython().run_cell_magic('time', '', '#function to select a safe window of rows\nwindow = window(df)\n#new dataframe to append the results of the for loop\nnew_df=cudf.DataFrame()\n#set start = 0\nstart = 0\nfor p in range(0,num_parts):\n    end = p*chunk + chunk\n    if end < all_rows:\n        chunk_window = df[start:end].tail(window)\n        second_last_unique = chunk_window[\'ip\'].unique().values.tolist()[-2]\n        new_end = chunk_window[chunk_window[\'ip\'] == second_last_unique].tail(1).index[0]\n        print(f"Processing {(new_end+1)-start} rows of chunk N° {p+1}")\n        new_chunk = feature_engineering(df,start,new_end)\n    else:\n        print(f"Processing {all_rows-(new_end+1)} rows of chunk N° {p+1}")\n        new_chunk = feature_engineering(df,start,None)\n    start = new_end+1\n    new_df = new_df.append(new_chunk, ignore_index=True)\n')


# In[5]:


new_df


# # RESULTS
# 
# **GPU P100**
# 
# - Read, preprocess and divide took **1min 8s**
# - The processing with feature engineering took **2min 17s**
# 
# **GPU T4 x2**
# 
# - Read, preprocess and divide took **1min 34s**
# - The processing with feature engineering took **2min 28s**
# 
# 
# GPU P100 apparently seems more powerful, but let us remember that GPU T4 x2 was designed for image processing and neural networks. My advice for novices is to use only CPU and GPU P100 for tabular datasets, and GPU T4 x2 for images
