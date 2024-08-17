#!/usr/bin/env python
# coding: utf-8

# ---
# # [Predict Student Performance from Game Play][1]
# 
# The goal of this competition is to predict student performance during game-based learning in real-time.
# 
# ---
# #### **The aim of this notebook is to...**
# - **1. devide the data of game play logs on 'level_group' level or 'session_id' level by using clustering methods for categorical features. (<a href="#4">Chapter4</a>)**
# - **2. aggregate the game log data and make features for the purpose of clustering (<a href="#3">Chapter3</a>)**
# 
# **Please note that the aim of this notebook is not just as same as the competition's one. Thus, I will not make predictions for submission in this notebook.**
# 
# ---
# #### **Results**
# - **1. Data points created by aggregations on 'level_group' level are clearly devided into clusters. (<a href="#4.1">Chapter4.1</a>)**
# - **2. A large part of data points created by aggregations on 'session_id' level are still not clearly devided into clusters, but there might be a specific small group. (<a href="#4.3">Chapter4.3</a>)**
# 
# 
# ---
# **References:** Thanks to previous great codes, blogs and notebooks.
# 
# - Japanese tech-blogs.
#     - [質的（カテゴリカルな）な特徴量があるときのクラスタリング手法とPythonでの実装について][2]
#     - [読了：Ahmad & Khan (2019) 量質混在データのクラスタリング手法レビュー ][3]
#     - [Oisixのお客様をクラスタリングしてみた][4]
# 
# ---
# **If you find this notebook useful, or when you copy&edit this notebook, please give me an upvote. It helps me keep up my motivation.**
# 
# ---
# [1]: https://www.kaggle.com/competitions/predict-student-performance-from-game-play
# [2]: https://qiita.com/shinji_komine/items/b634146bd873d8d876e5
# [3]: https://elsur.jpn.org/mt/2019/12/002808.html
# [4]: https://creators.oisix.co.jp/entry/2020/06/25/194031#%E3%81%9D%E3%81%AE%E4%BB%96%E6%89%8B%E6%B3%95K-modesK-prototype

# <h1 style="background:#05445E; border:0; border-radius: 12px; color:#D3D3D3"><center>0. TOC</center></h1>
# 
# <ul class="list-group" style="list-style-type:none;">
#     <li><a href="#1" class="list-group-item list-group-item-action">1. Settings</a></li>
#     <li><a href="#2" class="list-group-item list-group-item-action">2. Data Loading</a></li>
#     <li><a href="#3" class="list-group-item list-group-item-action">3. Feature Engineering</a>
#         <ul class="list-group" style="list-style-type:none;">
#             <li><a href="#3.1" class="list-group-item list-group-item-action">3.1 Aggregate on 'level' levevl</a></li>
#             <li><a href="#3.2" class="list-group-item list-group-item-action">3.2 Aggregate on 'level_group' levevl</a></li>
#             <li><a href="#3.3" class="list-group-item list-group-item-action">3.3 Load and aggregate additional data</a></li>
#             <li><a href="#3.4" class="list-group-item list-group-item-action">3.4 Aggregate on 'session_id' levevl</a></li>
#         </ul>
#     </li>
#     <li><a href="#4" class="list-group-item list-group-item-action">4. Clustering</a>
#         <ul class="list-group" style="list-style-type:none;">
#             <li><a href="#4.1" class="list-group-item list-group-item-action">4.1 Clustering on 'level_group' level </a></li>
#             <li><a href="#4.2" class="list-group-item list-group-item-action">4.2 Clustering on 'session_id' level</a></li>
#             <li><a href="#4.3" class="list-group-item list-group-item-action">4.3 Convert categorical features from sparse matrix to dense matrix</a></li>
#         </ul>
#     </li>
# </ul>

# <a id ="1"></a><h1 style="background:#05445E; border:0; border-radius: 12px; color:#D3D3D3"><center>1. Settings</center></h1>

# In[1]:


## Parameters
data_config = {
    'train_csv_path': '/kaggle/input/predict-student-performance-from-game-play/train.csv',
    'train_labels_csv_path': '/kaggle/input/predict-student-performance-from-game-play/train_labels.csv',
    'test_csv_path': '/kaggle/input/predict-student-performance-from-game-play/test.csv',
    'sample_submission_path': '/kaggle/input/predict-student-performance-from-game-play/sample_submission.csv',
}

exp_config = {
    'sample_limits': 100000, 
    'data_loading_iter': 25,
    'level_group_n_clusters': 3,
    'session_id_n_clusters': 5,
}

print('Parameters setted!')


# In[2]:


## Import dependencies 
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
import matplotlib.ticker as ticker
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os, sys, pathlib, gc
import re, math, random, time
import datetime as dt
from tqdm import tqdm
from typing import Optional, Union, Tuple
from collections import OrderedDict

import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

from kmodes.kprototypes import KPrototypes
get_ipython().system('pip install gower -q')
import gower
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

print('import done!')


# In[3]:


## For reproducible results    
def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['PYTHONHASHSEED'] = str(s) 
    print('Seeds setted!')
global_seed = 42
seed_all(global_seed)

## Limit GPU Memory in TensorFlow
## Because TensorFlow, by default, allocates the full amount of available GPU memory when it is launched. 
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")
    
## For Seaborn Setting
custom_params = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    'grid.alpha': 0.3,
    'figure.figsize': (16, 6),
    'axes.titlesize': 'Large',
    'axes.labelsize': 'Large',
    'figure.facecolor': '#fdfcf6',
    'axes.facecolor': '#fdfcf6',
}
cluster_colors = ['#b4d2b1', '#568f8b', '#1d4a60', '#cd7e59', '#ddb247', '#d15252']
sns.set_theme(
    style='whitegrid',
    #palette=sns.color_palette(cluster_colors),
    rc=custom_params,)


# <a id ="2"></a><h1 style="background:#05445E; border:0; border-radius: 12px; color:#D3D3D3"><center>2. Data Loading</center></h1>

# ---
# ### [File and Data Field Descriptions](https://www.kaggle.com/competitions/predict-student-performance-from-game-play/data)
# 
# - **train.csv** - the training set
#  - `session_id` - the ID of the session the event took place in
#  - `index` - the index of the event for the session
#  - `elapsed_time` - how much time has passed (in milliseconds) between the start of the session and when the event was recorded
#  - `event_name` - the name of the event type
#  - `name` - the event name (e.g. identifies whether a notebook_click is is opening or closing the notebook)
#  - `level` - what level of the game the event occurred in (0 to 22)
#  - `page` - the page number of the event (only for notebook-related events)
#  - `room_coor_x` - the coordinates of the click in reference to the in-game room (only for click events)
#  - `room_coor_y` - the coordinates of the click in reference to the in-game room (only for click events)
#  - `screen_coor_x` - the coordinates of the click in reference to the player’s screen (only for click events)
#  - `screen_coor_y` - the coordinates of the click in reference to the player’s screen (only for click events)
#  - `hover_duration` - how long (in milliseconds) the hover happened for (only for hover events)
#  - `text` - the text the player sees during this event
#  - `fqid` - the fully qualified ID of the event
#  - `room_fqid` - the fully qualified ID of the room the event took place in
#  - `text_fqid` - the fully qualified ID of the
#  - `fullscreen` - whether the player is in fullscreen mode
#  - `hq` - whether the game is in high-quality
#  - `music` - whether the game music is on or off
#  - `level_group` - which group of levels - and group of questions - this row belongs to (0-4, 5-12, 13-22)
# 
# 
# - **test.csv** -  the test set
# 
# - **sample_submission.csv** - a sample submission file in the correct format
# 
# - **train_labels.csv** - `correct` value for all 18 questions for each session in the training set
# 
# ---
# ### [Submission & Evaluation](https://www.kaggle.com/competitions/predict-student-performance-from-game-play/overview/evaluation)
# 
# - Submissions will be evaluated based on their F1 score.
# 
# ---

# ---
# **Note**
# - train.csv is too large to load fully on VM. (In this competision's constraint, VMs will have only 2 CPUs, 8GB of RAM, and no GPU available.)
# - Thus, I will load the first 100,000 records for the present. (I will load additional data in later.) 
# - P.S. Thanks for [this discussion](https://www.kaggle.com/competitions/predict-student-performance-from-game-play/discussion/384359), I added the code for full loading of train.csv. However, the full record of train.csv is too huge to aggregate or compute clustering. Thus, I limited the loading data to a part of it. 
# 
# ---

# In[4]:


## Data Loading

if exp_config['sample_limits'] is None:
    # Load the whole train.csv
    dtypes={
        'session_id':'category', 
        'elapsed_time':np.int32,
        'event_name':'category',
        'name':'category',
        'level':np.uint8,
        'page':'category',
        'room_coor_x':np.float32,
        'room_coor_y':np.float32,
        'screen_coor_x':np.float32,
        'screen_coor_y':np.float32,
        'hover_duration':np.float32,
         'text':'category',
         'fqid':'category',
         'room_fqid':'category',
         'text_fqid':'category',
         'fullscreen':'category',
         'hq':'category',
         'music':'category',
         'level_group':'category'
        }
    train_df = pd.read_csv(data_config['train_csv_path'], dtype=dtypes)
else:
    train_df = pd.read_csv(data_config['train_csv_path'], nrows=exp_config['sample_limits'])

train_labels_df = pd.read_csv(data_config['train_labels_csv_path'], nrows=exp_config['sample_limits'])
test_df = pd.read_csv(data_config['test_csv_path'])
submission_df = pd.read_csv(data_config['sample_submission_path'])

print(f'train_length: {len(train_df)}')
print(f'train_labels_length: {len(train_labels_df)}')
print(f'test_lenght: {len(test_df)}')
print(f'submission_length: {len(submission_df)}')


# In[5]:


## Null Value Check
print('train_df.info()'); print(train_df.info(), '\n')
print('train_labels_df.info()'); print(train_labels_df.info(), '\n')
print('test_df.info()'); print(test_df.info(), '\n')

## train_df Check
train_df.head()


# In[6]:


# Unique number of categorical features
for feature in train_df.columns:
    if train_df[feature].dtype=='object':
        print(f"unique_#_of_{feature}: {train_df[feature].nunique()}")

print()
        
for feature in ['session_id', 'level', 'fullscreen', 'hq', 'music']:
    print(f"unique_#_of_{feature}: {train_df[feature].nunique()}")


# In[7]:


# Basic statistics of tain_df
train_df.describe()


# In[8]:


# Unique values of categorical features
session_ids = train_df['session_id'].unique()
levels = train_df['level'].unique()
event_names = train_df['event_name'].unique()
names = train_df['name'].unique()
fqids = train_df['fqid'].unique()
room_fqids = train_df['room_fqid'].unique()
text_fqids = train_df['text_fqid'].unique()
level_groups = train_df['level_group'].unique() # array(['0-4', '5-12', '13-22'], dtype=object)


# <a id ="3"></a><h1 style="background:#05445E; border:0; border-radius: 12px; color:#D3D3D3"><center>3. Feature Engineering</center></h1>

# <a id ="3.1"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>3.1 Aggregate on 'level' levevl</center></h2>

# In[9]:


# Features
level_features = [
    'session_id', 
    'level',
    'level_elapsed_time',     
    'max_room_coor_x',
    'min_room_coor_x',
    'mean_room_coor_x',
    'std_room_coor_x',
    'max_room_coor_y',
    'min_room_coor_y',
    'mean_room_coor_y',
    'std_room_coor_y',
    'fullscreen',
    'hq',
    'music',
    'level_group',
    ]
event_flg_features = [f"{event_name}_flg" for event_name in event_names]
fqid_flg_features = [f"{fqid}_flg" for fqid in fqids]
level_features.extend(event_flg_features)
level_features.extend(fqid_flg_features)
print(len(level_features))


# In[10]:


# This code took some minutes to compute

# Aggregation
def aggregate_level_behavior(train_df, level_features):
    agg_level_df = pd.DataFrame(columns=level_features)
    session_ids = train_df['session_id'].unique()
    for session_id in session_ids:
        ex_df = train_df[train_df['session_id']==session_id]
        session_df = pd.DataFrame(columns=level_features)

        for i, level in enumerate(levels):
            value_dict = {}
            value_dict['session_id'] = session_id
            value_dict['level'] = level
            ex_level_df = ex_df[ex_df['level']==level]
            if len(ex_level_df) != 0:
                value_dict['level_elapsed_time'] = ex_level_df['elapsed_time'].iloc[-1] - ex_level_df['elapsed_time'].iloc[0]
                value_dict['fullscreen'] = ex_level_df['fullscreen'].mode()[0]
                value_dict['hq'] = ex_level_df['hq'].mode()[0]
                value_dict['music'] = ex_level_df['music'].mode()[0]
                value_dict['level_group'] = ex_level_df['level_group'].iloc[0]

                value_dict['max_room_coor_x'] = ex_level_df['room_coor_x'].max()
                value_dict['min_room_coor_x'] = ex_level_df['room_coor_x'].min()
                value_dict['mean_room_coor_x'] = ex_level_df['room_coor_x'].mean()
                value_dict['std_room_coor_x'] = ex_level_df['room_coor_x'].std()

                value_dict['max_room_coor_y'] = ex_level_df['room_coor_y'].max()
                value_dict['min_room_coor_y'] = ex_level_df['room_coor_y'].min()
                value_dict['mean_room_coor_y'] = ex_level_df['room_coor_y'].mean()
                value_dict['std_room_coor_y'] = ex_level_df['room_coor_y'].std()

                for event_name in event_names:
                    tmp_count = ex_level_df[ex_level_df['event_name']==event_name]['event_name'].count()
                    if tmp_count > 0:
                        value_dict[f"{event_name}_flg"] = 1
                    else:
                        value_dict[f"{event_name}_flg"] = 0

                for fqid in fqids:
                    tmp_count = ex_level_df[ex_level_df['fqid']==fqid]['fqid'].count()
                    if tmp_count > 0:
                        value_dict[f"{fqid}_flg"] = 1
                    else:
                        value_dict[f"{fqid}_flg"] = 0

                session_df.loc[i] = value_dict
        agg_level_df = pd.concat([agg_level_df, session_df], axis=0).reset_index(drop=True)
    return agg_level_df

# Operation check
agg_level_df = aggregate_level_behavior(train_df, level_features)
print(agg_level_df.shape)
display(agg_level_df.describe())


# <a id ="3.2"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>3.2 Aggregate on 'level_group' levevl</center></h2>

# In[11]:


# Features
level_group_features = [
    'session_id', 
    'mean_level_elapsed_time',     
    'max_room_coor_x',
    'min_room_coor_x',
    'mean_mean_room_coor_x',
    'mean_std_room_coor_x',
    'max_room_coor_y',
    'min_room_coor_y',
    'mean_mean_room_coor_y',
    'mean_std_room_coor_y',
    'fullscreen',
    'hq',
    'music',
    'level_group',
    ]
event_flg_features = [f"{event_name}_flg" for event_name in event_names]
fqid_flg_features = [f"{fqid}_flg" for fqid in fqids]
level_group_features.extend(event_flg_features)
level_group_features.extend(fqid_flg_features)
print(len(level_group_features))


# In[12]:


# Aggregation
def aggregate_level_group_behavior(agg_level_df, level_group_features):
    agg_group_df = pd.DataFrame(columns=level_group_features)
    session_ids = agg_level_df['session_id'].unique()
    for session_id in session_ids:
        ex_df = agg_level_df[agg_level_df['session_id']==session_id]
        session_df = pd.DataFrame(columns=level_group_features)

        for i, level_group in enumerate(level_groups):
            value_dict = {}
            value_dict['session_id'] = session_id
            value_dict['level_group'] = level_group
            ex_group_df = ex_df[ex_df['level_group']==level_group]
            if len(ex_group_df) != 0:
                value_dict['mean_level_elapsed_time'] = ex_group_df['level_elapsed_time'].mean()
                value_dict['fullscreen'] = ex_group_df['fullscreen'].mode()[0]
                value_dict['hq'] = ex_group_df['hq'].mode()[0]
                value_dict['music'] = ex_group_df['music'].mode()[0]

                value_dict['max_room_coor_x'] = ex_group_df['max_room_coor_x'].max()
                value_dict['min_room_coor_x'] = ex_group_df['min_room_coor_x'].min()
                value_dict['mean_mean_room_coor_x'] = ex_group_df['mean_room_coor_x'].mean()
                value_dict['mean_std_room_coor_x'] = ex_group_df['std_room_coor_x'].mean()

                value_dict['max_room_coor_y'] = ex_group_df['max_room_coor_y'].max()
                value_dict['min_room_coor_y'] = ex_group_df['min_room_coor_y'].min()
                value_dict['mean_mean_room_coor_y'] = ex_group_df['mean_room_coor_y'].mean()
                value_dict['mean_std_room_coor_y'] = ex_group_df['std_room_coor_y'].mean()

                for event_name in event_names:
                    tmp_count = ex_group_df[f'{event_name}_flg'].sum()
                    if tmp_count > 0:
                        value_dict[f"{event_name}_flg"] = 1
                    else:
                        value_dict[f"{event_name}_flg"] = 0

                for fqid in fqids:
                    tmp_count = ex_group_df[f'{fqid}_flg'].sum()
                    if tmp_count > 0:
                        value_dict[f"{fqid}_flg"] = 1
                    else:
                        value_dict[f"{fqid}_flg"] = 0

            else:
                empty_group_features = [
                        #'session_id', 
                        'mean_level_elapsed_time',     
                        'max_room_coor_x',
                        'min_room_coor_x',
                        'mean_mean_room_coor_x',
                        'mean_std_room_coor_x',
                        'max_room_coor_y',
                        'min_room_coor_y',
                        'mean_mean_room_coor_y',
                        'mean_std_room_coor_y',
                        'fullscreen',
                        'hq',
                        'music',
                        #'level_group',
                        ]
                event_flg_features = [f"{event_name}_flg" for event_name in event_names]
                fqid_flg_features = [f"{fqid}_flg" for fqid in fqids]
                empty_group_features.extend(event_flg_features)
                empty_group_features.extend(fqid_flg_features)
                for feature in empty_group_features:
                    value_dict[feature] = np.nan

            session_df.loc[i] = value_dict
        agg_group_df = pd.concat([agg_group_df, session_df], axis=0).reset_index(drop=True)
    return agg_group_df
    
# Operation check
agg_group_df = aggregate_level_group_behavior(agg_level_df, level_group_features)
print(agg_group_df.shape)
display(agg_group_df.describe())


# <a id ="3.3"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>3.3 Load and aggregate additional data</center></h2>

# In[13]:


sample_limits = exp_config['sample_limits']
data_loading_iter = exp_config['data_loading_iter']
train_df_columns = train_df.columns

all_agg_group_df = pd.DataFrame(columns=level_group_features)

# The data on the edge is excluded from the aggregation.
gap_session_ids = []

if exp_config['sample_limits'] is not None:
    if data_loading_iter > 1:
        for i in range(data_loading_iter):    
            train_df = pd.read_csv(data_config['train_csv_path'],
                                   skiprows=sample_limits*i,
                                   nrows=sample_limits,
                                   header=None,
                                   )
            train_df.columns = train_df_columns

            agg_level_df = aggregate_level_behavior(train_df, level_features)
            agg_df = aggregate_level_group_behavior(agg_level_df, level_group_features)

            gap_session_ids.append(agg_df['session_id'].iloc[-1])
            all_agg_group_df = pd.concat([all_agg_group_df, agg_df], axis=0).reset_index(drop=True)

        # The data on the edge is excluded from the aggregation.
        all_agg_group_df = all_agg_group_df[~all_agg_group_df['session_id'].isin(gap_session_ids)]
        all_agg_group_df = all_agg_group_df.reset_index(drop=True)

    else:
        all_agg_group_df = agg_group_df
else:
    all_agg_group_df = agg_group_df
        
print(all_agg_group_df.shape)
display(all_agg_group_df.describe())


# <a id ="3.4"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>3.4 Aggregate on 'session_id' levevl</center></h2>

# In[14]:


# Features
session_id_features = [
    'session_id',     
    ]
quality_features = [
    'fullscreen',
    'hq',
    'music',
    ]
level_group_quality_features = \
    [f"{level_group}_{quality}" for quality in quality_features for level_group in level_groups]
coor_x_features = [
    'max_room_coor_x',
    'min_room_coor_x',
    'mean_mean_room_coor_x',
    'mean_std_room_coor_x',
    ]
level_group_coor_x_features = \
    [f"{level_group}_{coor_x}" for coor_x in coor_x_features for level_group in level_groups]
coor_y_features = [
    'max_room_coor_y',
    'min_room_coor_y',
    'mean_mean_room_coor_y',
    'mean_std_room_coor_y',
    ]
level_group_coor_y_features = \
    [f"{level_group}_{coor_y}" for coor_y in coor_y_features for level_group in level_groups]
level_group_elapsed_time_features = \
    [f"{level_group}_mean_level_elapsed_time" for level_group in level_groups]
level_group_event_flg_features = \
    [f"{level_group}_{event_name}_flg" for event_name in event_names for level_group in level_groups]
level_group_fqid_flg_features = \
    [f"{level_group}_{fqid}_flg" for fqid in fqids for level_group in level_groups]
session_id_features.extend(level_group_quality_features)
session_id_features.extend(level_group_coor_x_features)
session_id_features.extend(level_group_coor_y_features)
session_id_features.extend(level_group_elapsed_time_features)
session_id_features.extend(level_group_event_flg_features)
session_id_features.extend(level_group_fqid_flg_features)
print(len(session_id_features))


# In[15]:


# Aggregation
def aggregate_session_behavior(
    agg_level_group_df,
    session_id_features,
    level_group_features,
    level_groups=level_groups,
    ):
    agg_session_df = pd.DataFrame(columns=session_id_features)
    level_group_features_copy = level_group_features.copy()
    level_group_features_copy.remove('session_id')
    session_ids = agg_level_group_df['session_id'].unique()
    for i, session_id in enumerate(session_ids):
        ex_df = agg_level_group_df[agg_level_group_df['session_id']==session_id].reset_index(drop=True)
        value_dict = {}
        value_dict['session_id'] = session_id
        if len(ex_df) > 0:
            for level_group in level_groups:
                level_group_df = ex_df[ex_df['level_group']==level_group].reset_index(drop=True)
                if len(level_group_df) > 0:
                    for level_group_feature in level_group_features_copy:
                        value_dict[f"{level_group}_{level_group_feature}"] = \
                            level_group_df[level_group_feature].iloc[0]
        else:
            for level_group in level_groups:
                for level_group_feature in level_group_features_copy:
                    value_dict[f"{level_group}_{level_group_feature}"] = np.nan

        agg_session_df.loc[i] = value_dict
    agg_session_df = agg_session_df.reset_index(drop=True)
    return agg_session_df

# Operation check
agg_session_df = aggregate_session_behavior(
    all_agg_group_df, session_id_features, level_group_features)
print(agg_session_df.shape)
display(agg_session_df.describe())


# <a id ="4"></a><h1 style="background:#05445E; border:0; border-radius: 12px; color:#D3D3D3"><center>4. Clustering</center></h1>

# ---
# ## How to clustering the data points including both numerical and categorical features.
# 
# ---
# #### 【Gower’s Distance】
# **<u>Calculation</u>**  
# Gower’s Distance is calculated based on Gower's similarity score.  
# Gower's similarity score represents the similarity between two data points, $i$ and $j$.  
# Please note that the larger gower's similarity score means that the two samples are more similar.  
# Gower's similarity $S_{\text{Gower}}(x_i, x_j)$ is caluculated as follows:  
# 
# - for each feature $k$, 
# 
#   $S_{ij} = \frac{\sum_k s_{ijk}}{\sum_k \delta_{ijk}}$
# 
#   How to calculate the $s_{ijk}$ and $\delta_{ijk}$ is as follows:
#   
#   - $s_{ijk}$ :  
#     - for numerical features -> $s_{ijk} = 1 - \frac{\|x_i-x_j\|}{R_k}$  
#     - for binary features -> 1 if both are 1, else 0  
#     - for categorical features -> 1 if $x_{ij} = x_{jk}$, 0 if $x_{ij} \neq x_{jk}$  
#     - note: $R_k$ represents the range of the values of feature $k$ (max - min)
#     
#   - $\delta_{ijk}$ :  
#     - for numerical features -> 1  
#     - for binary features -> 0 if both are 0, else 1  
#     - for categorical features -> 1  
#     
#   <br>
#   
#   - note1: for binary features -> It means that 0-0 is not treated as 'match' case  
#   - note2: for categorical features -> It is similar to Hamming distance or Sørensen–Dice coefficient  
#   - note3: When the valuable has hierarchical structure, you can (somehow) allocate the weights $w_k$ and calculate as follows:  
#     $S_{ij} = \frac{\sum_k s_{ijk} w_k}{\sum_k \delta_{ijk} w_k}$  
#   
# Now, Gower's distance $S_{\text{Gower}}(x_i, x_j)$ is caluculated as follows: 
# 
# $ d_{\text{Gower}} = \sqrt{1-S_{\text{Gower}}} $
#     
# <br>
# 
# **<u>Library</u>** : gower
#   ```python
#   pip install gower
#   ```
# <br>
# 
# ---
# #### 【K-modes】
# **<u>Calculation</u>**  
# K-modes is the application of K-means to categorical features.  
# The distance of two data points $d_{ij}$ are calculated as follows:
# 
# - When $x$ contains $p$-dimentional numerical features and $m$-dimentional categorical features,  
# 
#   $d_{ij} = \sum^p_{k=1}(x_{ik} - x_{jk})^2 + \gamma \sum^m_{k=p+1} \delta_{ijk}$  
#   $\delta_{ijk} = 0 \, \text{if} \, x_{ik} = x_{jk} \, , \text{else} \, 1$  
# 
#   - note: $\gamma$ is the parameter which controls the weight balances between numerical and categorical features.  
#   
# 
# <br>
# 
# **<u>Library</u>** : kmodes
#   ```python
#   pip install kmodes
#   ```   
# - note1: The parameter $\gamma$ is automatically computed based on the variances of the categorical features by default.  
# - note2: We can use K-prototype algorithm which combines K-means and K-modes for the clustering based on both numerical and categorical features.
# 
# <br>
# 
# ---

# <a id ="4.1"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>4.1 Clustering on 'level_group' features</center></h2>

# In[16]:


# Features
categorical_features = [ 
    'fullscreen',
    'hq',
    'music',
    'level_group'
    ]
categorical_features.extend(event_flg_features)
categorical_features.extend(fqid_flg_features)
# Of course, I will not use 'level_group' as a feature for clustering.

numerical_features = [
    'mean_level_elapsed_time',     
    'max_room_coor_x',
    'min_room_coor_x',
    'mean_mean_room_coor_x',
    'mean_std_room_coor_x',
    'max_room_coor_y',
    'min_room_coor_y',
    'mean_mean_room_coor_y',
    'mean_std_room_coor_y',
    ]
    
print(len(numerical_features), len(categorical_features))


# In[17]:


# The number of clusters
n_clusters = exp_config['level_group_n_clusters']

# Data preprocessing
def data_preprocessing(
    df, numerical_features, categorical_features):

    X_df = df.copy()

    # Drop the records which contain null values in numerical features
    X_df[numerical_features] = X_df[numerical_features].astype('float64')
    X_df.dropna(subset=numerical_features, axis=0, inplace=True)
    X_df = X_df.reset_index(drop=True)

    # Numerical and categorical features for clustering
    X_numerical = X_df[numerical_features]
    X_categorical = X_df[categorical_features]

    # Standardization of numerical features
    stdscl = StandardScaler()
    X_numerical_scl = pd.DataFrame(stdscl.fit_transform(X_numerical) , columns=X_numerical.columns)

    # Standardization of categorical features (convert to str type)
    X_categorical_scl = X_categorical.astype(str)

    # Preprocessed features
    X_scl = pd.concat([X_numerical_scl, X_categorical_scl], axis=1, join='inner')
    
    return X_scl
    
# Operation check
X_scl = data_preprocessing(
    all_agg_group_df, 
    numerical_features=numerical_features, 
    categorical_features=categorical_features)
print(X_scl.shape)


# ---
# Clustering by K-Prototype method
# 
# ---

# In[18]:


# Positions of categorical columns (this is nedded as a parameter of K-Prototype) 
catColumnsPos = [X_scl.columns.get_loc(col) for col in list(X_scl.select_dtypes('object').columns)]

# Execution of clustering
kprototype = KPrototypes(n_jobs=-1, n_clusters=n_clusters, init='Huang')
kprototype.fit_predict(X_scl, categorical=catColumnsPos)
all_agg_group_df['k_proto_label'] = kprototype.labels_
display(all_agg_group_df.groupby('k_proto_label')['session_id'].count())


# In[19]:


# Elbow method by K-Prototype
def plot_elbow(X_scl, n_max=8):
    catColumnsPos = [ X_scl.columns.get_loc(col) for col in list(X_scl.select_dtypes('object').columns) ]
    n_clusters_list = []
    cost_list = []
    for n in tqdm(range(1, n_max+1)):
        try:
            kprototype = KPrototypes(n_jobs=-1, n_clusters=n, init='Huang')
            kprototype.fit_predict(X_scl, categorical=catColumnsPos)
            n_clusters_list.append(n)
            cost_list.append(kprototype.cost_)
        except:
            break
    plt.plot(n_clusters_list, cost_list)
    plt.xlabel('n_clusters')
    plt.ylabel('cost')
    return 

# Operation check
plot_elbow(X_scl, n_max=6)


# ---
# **Results:**
# - n_clusters should be 3.
# 
# ---

# ---
# Clustering based on gower's distance
# 
# ---

# In[20]:


# Computation of gower's distance
X_distance = gower.gower_matrix(X_scl)
print(X_distance.shape)

# Hierarchical clustering
AggCluster = AgglomerativeClustering(
    n_clusters=n_clusters, linkage='complete',
    affinity='precomputed', compute_distances=True)
AggCluster.fit(X_distance)
all_agg_group_df['gower_label'] = AggCluster.fit_predict(X_distance) 
display(all_agg_group_df.groupby('gower_label')['session_id'].count())


# In[21]:


# The function for the execution of clustering on both numerical and categorical features
# This fuction executes the clustering based on gower's distance by default
def make_clustering_labels(df,
                           numerical_features,
                           categorical_features,
                           n_clusters=5,
                           K_proto_flg=False, 
                           gower_flg=True, 
                           verbose=True):
    X_df = df.copy()
    X_df[numerical_features] = X_df[numerical_features].astype('float')
    X_df[categorical_features] =X_df[categorical_features].astype('str')
    
    if verbose:
        print(f'------ before dropna for numerical features ------: \n{X_df.shape}')
    X_df.dropna(subset=numerical_features, inplace=True)
    X_df = X_df.reset_index(drop=True)
    if verbose:
        print(f'------ after dropna for numerical features ------: \n{X_df.shape}')

    X_numerical = X_df[numerical_features]
    X_categorical = X_df[categorical_features]
    
    stdscl = StandardScaler()
    X_numerical_scl = pd.DataFrame(stdscl.fit_transform(X_numerical) , columns=X_numerical.columns)
    X_categorical_scl = X_categorical.astype(str)

    if verbose:
        for categorical_feature in categorical_features:
            print('------ unique numbers for categorical features ------')
            print(f"{categorical_feature} unique_number: {X_df[categorical_feature].nunique()}")

    X_scl =pd.concat([X_numerical_scl, X_categorical_scl], axis=1, join='inner')

    if K_proto_flg: # Clustering by K-Prototype method
        catColumnsPos = [X_scl.columns.get_loc(col) for col in list(X_scl.select_dtypes('object').columns)]
        kprototype = KPrototypes(n_jobs=-1, n_clusters=n_clusters, init='Huang', random_state=0)
        kprototype.fit_predict(X_scl, categorical=catColumnsPos)
        X_df['k_proto_label'] = kprototype.labels_

    if gower_flg: # Clustering based on gower's distance
        X_distance = gower.gower_matrix(X_scl)
        AggCluster = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete',
                                    affinity='precomputed', compute_distances=True)
        AggCluster.fit(X_distance)
        X_df['gower_label'] = AggCluster.fit_predict(X_distance)
        return X_df, X_distance
    
    return X_df

# Operation check
all_agg_group_df, X_distance = make_clustering_labels(
    all_agg_group_df,
    numerical_features=numerical_features, 
    categorical_features=categorical_features, 
    n_clusters=n_clusters,
    verbose=False,
    )
display(all_agg_group_df.groupby('gower_label')['session_id'].count())


# ---
# Dimensionality reduction with PCA, t-SNE and UMAP
# 
# ---

# In[22]:


# Dimensionality reduction with PCA
pca = PCA()
pca.fit(X_distance)
X_distance_pca = pca.transform(X_distance)
#print(X_distance.shape)

# Visualization of the variances
#var = pd.DataFrame(X_distance_pca).var()
#plt.plot(np.arange(0,len(X_distance_pca)), var)
#plt.show()
#print(pca.explained_variance_ratio_)

# Plot in a 2D graph
plt.figure(figsize=(4,4))
plt.scatter(X_distance_pca[:,0],X_distance_pca[:,1])
plt.show()


# In[23]:


# Dimensionality reduction with t-SNE
tsne = TSNE(random_state=0)
X_distance_tsne = tsne.fit_transform(X_distance)

# Plot in a 2D graph
plt.figure(figsize=(4,4))
plt.scatter(X_distance_tsne[:,0],X_distance_tsne[:,1])
plt.show()


# In[24]:


# Dimensionality reduction with UMAP
mapper = umap.UMAP(random_state=0)
X_distance_umap = mapper.fit_transform(X_distance)

# Plot in a 2D graph
plt.figure(figsize=(4,4))
plt.scatter(X_distance_umap[:,0],X_distance_umap[:,1])
plt.show()


# In[25]:


# Color-coded display according to a categorical feature.
def plot_category(X, X_distance, label_feature, figsize=(4,4), mapper='umap'):
    if mapper == 'tsne':
        mapper = TSNE(random_state=0)
    elif mapper == 'pca':
        mapper = PCA()
    else:
        mapper = umap.UMAP(random_state=0)
    X_distance_emb = mapper.fit_transform(X_distance)
    cmap_keyword = "jet"
    cmap = plt.get_cmap(cmap_keyword)
    plt.figure(figsize=figsize)
    labels = list(X[label_feature].unique())
    labels = sorted(labels)
    for idx, label in enumerate(labels):
        condition = X[label_feature] == label
        c = cmap(idx/(X[label_feature].nunique()-1))
        plt.scatter(X_distance_emb[condition,0],X_distance_emb[condition,1], color=c, label=label)
    plt.title(label_feature)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()
    
# Operation check
plot_category(all_agg_group_df, X_distance,
              'gower_label', mapper='umap')
plot_category(all_agg_group_df, X_distance,
              'k_proto_label', mapper='umap')
plot_category(all_agg_group_df, X_distance,
              'level_group', mapper='umap')


# ---
# **Results:**
# - Data points created by aggregations on 'level_group' are clearly devided by level_groups.  
# - Please note that I didn't use 'level_group' as a feature for clustering.  
# - It means that there are distinct differences in user experiences by level_groups.
# 
# ---

# In[26]:


# Display basic statistics on a numerical feature
def make_numerical_table(
        cluster_df, 
        numerical_feature,
        label_feature='gower_label',
        label_groups=None):
  
    value_list = []
    if label_groups is None:
        label_groups = list(cluster_df.groupby(label_feature)[numerical_feature].groups.keys())
        label_groups = sorted(label_groups)

    for group in label_groups:
        value_list.append(cluster_df.groupby(label_feature)[numerical_feature] \
                        .get_group(group).describe())
    value_df = pd.concat(value_list, axis=1, join='inner')
    value_df.columns = label_groups
    return value_df 
# Operation check
#value_df = make_numerical_table(all_agg_group_df, 'mean_level_elapsed_time', )
#display(value_df)

# Display specified statistics on numerical features in clusters
def make_stats_numerical(
        df,
        numerical_features,
        stats='mean',
        label_feature='gower_label'):
    
    label_groups = list(df.groupby(label_feature)[numerical_features[0]].groups.keys())
    label_groups = sorted(label_groups)
    stats_df = pd.DataFrame(columns=label_groups)

    for numerical_feature in numerical_features:
        value_df = make_numerical_table(df, 
                    numerical_feature,
                    label_feature,
                    label_groups=label_groups)
        stats_df.loc[numerical_feature] = value_df.loc[stats]
    stats_df.columns = label_groups
    return stats_df

# Operation check
mean_df = make_stats_numerical(
    all_agg_group_df,
    numerical_features=numerical_features,
    stats='mean',
)
print('mean: ')
display(mean_df)

std_df = make_stats_numerical(
    all_agg_group_df,
    numerical_features=numerical_features,
    stats='std',
)
print('std: ')
display(std_df)


# In[27]:


# Aggregation of common items in a categorical feature
def make_categorical_table(
        cluster_df, 
        categorical_feature,
        label_feature='gower_label', 
        join='inner'):
    value_list = []
    label_groups = list(cluster_df.groupby(label_feature)[categorical_feature].groups.keys())
    label_groups = sorted(label_groups)
    for group in label_groups:
        value_list.append(cluster_df.groupby(label_feature)[categorical_feature] \
                        .get_group(group).value_counts(sort=False, normalize=True))
    value_df = pd.concat(value_list, axis=1, join=join)
    value_df.columns = label_groups
    return value_df 
# Operation check
#value_df = make_categorical_table(all_agg_group_df, 'fullscreen', join='inner')
#display(value_df)

# Display ratios of common items in each categorical feature as a bar plot
def plot_bar_category(n_rows, n_cols,
                      df, categorical_cols,
                      label_feature='gower_label', color_set=False,
                      figsize=(4, 4)):
    figsize = tuple([l*x for l, x in zip(figsize, [n_cols, n_rows])])
    fig, axes = plt.subplots(n_rows, n_cols,
                            figsize=figsize,
                            tight_layout=True,
                            sharey=True)
    if color_set:
        cmap_keyword = "jet"
        cmap = plt.get_cmap(cmap_keyword)

    for i in range(n_rows):
        for j in range(n_cols):
            idx = (i * n_cols) + j
            if idx >= len(categorical_cols):
                break
            feature = categorical_cols[idx]
            # Aggregation of categorical features
            value_list = []
            label_groups = list(df.groupby(label_feature)[feature].groups.keys())
            label_groups = sorted(label_groups)
            for group in label_groups:
                value_list.append(df.groupby(label_feature)[feature] \
                                .get_group(group).value_counts(sort=False, normalize=True))
            value_df = pd.concat(value_list, axis=1, join='inner')
            value_df.columns = label_groups
            
            # display
            x = np.arange(len(value_df))
            labels = value_df.index
            margin = 0.2  #0 <margin< 1
            totoal_width = 1 - margin
            for k, group in enumerate(label_groups):
                height = value_df[group]
                pos = x - totoal_width * ( 1- (2*k+1)/len(label_groups) )/2
                if color_set:
                    unique_minus_one = df[label_feature].nunique() - 1
                    if unique_minus_one <= 0:
                        unique_minus_one = 1
                    c = cmap(k / unique_minus_one)
                    if (n_rows != 1) and (n_cols != 1):
                        axes[i][j].bar(pos, height, width=totoal_width/len(label_groups), color=c)
                    elif (n_rows == 1 ):
                        axes[j].bar(pos, height, width=totoal_width/len(label_groups), color=c)
                    elif (n_cols == 1 ):
                        axes[i].bar(pos, height, width=totoal_width/len(label_groups), color=c)
                else:
                    if (n_rows != 1) and (n_cols != 1):
                        axes[i][j].bar(pos, height, width=totoal_width/len(label_groups))
                    elif (n_rows == 1 ):
                         axes[j].bar(pos, height, width=totoal_width/len(label_groups))
                    elif (n_cols == 1 ):
                         axes[i].bar(pos, height, width=totoal_width/len(label_groups))

            # tick labels settings
            if (n_rows != 1) and (n_cols != 1):
                axes[i][j].set_xticks(x)
                axes[i][j].set_xticklabels(labels, rotation=45)
                axes[i][j].set_title(feature)
                axes[i][0].set_ylabel("ratio")
            elif (n_rows == 1 ):
                axes[j].set_xticks(x)
                axes[j].set_xticklabels(labels, rotation=45)
                axes[j].set_title(feature)
                axes[0].set_ylabel('ratio')
            elif (n_cols == 1 ):
                axes[i].set_xticks(x)
                axes[i].set_xticklabels(labels, rotation=45)
                axes[i].set_title(feature)
                axes[i].set_ylabel("ratio")
        else:
            continue
        break
    
    plt.show()
    
# Operation check
showing_cat_features = [ 
    'fullscreen',
    'hq',
    'music',
    ]
plot_bar_category(
    n_rows=2,
    n_cols=2,
    df=all_agg_group_df,
    categorical_cols=showing_cat_features,
    )


# <a id ="4.2"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>4.2 Clustering on 'session_id' features</center></h2>

# In[28]:


# Features
numerical_features = []
numerical_features.extend(level_group_coor_x_features)
numerical_features.extend(level_group_coor_y_features)
numerical_features.extend(level_group_elapsed_time_features)

categorical_features = []
categorical_features.extend(level_group_quality_features)
categorical_features.extend(level_group_event_flg_features)
categorical_features.extend(level_group_fqid_flg_features)

print(len(numerical_features), len(categorical_features))


# In[29]:


# The number of clusters
n_clusters = exp_config['session_id_n_clusters']

# Data preprocessing
X_scl = data_preprocessing(
    agg_session_df, 
    numerical_features=numerical_features, 
    categorical_features=categorical_features)
print(X_scl.shape)


# In[30]:


# Clustering by K-Prototype method
agg_session_df = make_clustering_labels(
    agg_session_df,
    numerical_features=numerical_features, 
    categorical_features=categorical_features, 
    n_clusters=n_clusters,
    K_proto_flg=True, 
    gower_flg=False,
    verbose=False,
    )
print('Clustering by K-Prototype method')
display(agg_session_df.groupby('k_proto_label')['session_id'].count())

print()

# Clustering based on gower's distance
agg_session_df, X_distance = make_clustering_labels(
    agg_session_df,
    numerical_features=numerical_features, 
    categorical_features=categorical_features, 
    n_clusters=n_clusters,
    verbose=False,
    )
print('Clustering by K-Prototype method')
display(agg_session_df.groupby('gower_label')['session_id'].count())


# In[31]:


# Display means and stds on numerical features in clusters
mean_df = make_stats_numerical(
    agg_session_df,
    numerical_features=numerical_features,
    stats='mean',
)
print('mean: ')
display(mean_df)

std_df = make_stats_numerical(
    agg_session_df,
    numerical_features=numerical_features,
    stats='std',
)
print('std: ')
display(std_df)


# In[32]:


# Dimensionality reduction with PCA
plot_category(agg_session_df, X_distance,
              'gower_label', mapper='pca')

plot_category(agg_session_df, X_distance,
              'k_proto_label', mapper='pca')


# In[33]:


# Dimensionality reduction with t-SNE
plot_category(agg_session_df, X_distance,
              'gower_label', mapper='tsne')

plot_category(agg_session_df, X_distance,
              'k_proto_label', mapper='tsne')


# In[34]:


# Dimensionality reduction with UMAP
plot_category(agg_session_df, X_distance,
              'gower_label', mapper='umap')
plot_category(agg_session_df, X_distance,
              'k_proto_label', mapper='umap')


# ---
# **Results:** 
# - Data points created by aggregations on 'session_id' level are not clearly devided into clusters.
# 
# ---

# <a id ="4.3"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>4.3 Convert categorical features from sparse matrix to dense matrix</center></h2>

# In[35]:


# Dimensionality reduction on categorical features with SVD
svd = sklearn.decomposition.TruncatedSVD(n_components=50)
decomposed_features = svd.fit_transform(agg_session_df[categorical_features])
X_num_cat_decompose = np.concatenate([X_scl[numerical_features], decomposed_features], axis=1)

# Data preprocessing
stdscl = StandardScaler()
X_num_cat_decompose_scl = stdscl.fit_transform(X_num_cat_decompose)

# Clustering by K-means
kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters)
pred = kmeans.fit_predict(X_num_cat_decompose_scl)
agg_session_df['decomposed_kmeans_label'] = pred

display(agg_session_df.groupby('decomposed_kmeans_label')['session_id'].count())


# In[36]:


# Dimensionality reduction with PCA
plot_category(agg_session_df, X_num_cat_decompose_scl,
              'decomposed_kmeans_label', mapper='pca')


# In[37]:


# Dimensionality reduction with t-SNE
plot_category(agg_session_df, X_num_cat_decompose_scl,
              'decomposed_kmeans_label', mapper='tsne')


# In[38]:


# Dimensionality reduction with UMAP
plot_category(agg_session_df, X_num_cat_decompose_scl,
              'decomposed_kmeans_label', mapper='umap')


# ---
# **Results:** 
# - A large part of data points created by aggregations on 'session_id' level are still not clearly devided into clusters, but there might be a specific small group.
# 
# ---

# In[ ]:




