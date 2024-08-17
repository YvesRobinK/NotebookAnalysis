#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

get_ipython().system('pip install feature_engine 2>/dev/null 1>&2')
get_ipython().system('pip install fastparquet 2>/dev/null 1>&2')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random

from sklearn.preprocessing import StandardScaler
from feature_engine.wrappers import SklearnTransformerWrapper as SKWrapper
from sklearn.model_selection import train_test_split


from matplotlib import pyplot as plt


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# **Downloading data**

# In[2]:


INPUT = '../input/tabular-playground-series-oct-2022/'

df_train_dtypes = pd.read_csv(INPUT + 'train_dtypes.csv')
df_test_dtypes = pd.read_csv(INPUT + 'test_dtypes.csv')
train_dtypes = {k: v for (k, v) in zip(df_train_dtypes.column, df_train_dtypes.dtype)}
test_dtypes = {k: v for (k, v) in zip(df_test_dtypes.column, df_test_dtypes.dtype)}

all_columns =list(pd.read_csv("/kaggle/input/tabular-playground-series-oct-2022/train_0.csv",nrows=1))
dropcols = ['game_num', 'event_id', 'event_time', 'player_scoring_next', 'team_scoring_next']
boost_t = ['boost0_timer', 'boost1_timer', 'boost2_timer', 'boost3_timer', 'boost4_timer', 'boost5_timer']
dropcols = dropcols + boost_t
usecols = [i for i in all_columns if i not in dropcols]


# In[3]:


train = []
num = 4
t_list = [3, 5, 7, 9]
for i in t_list:
#     df = pd.read_csv(INPUT + f'train_{i}.csv', dtype = train_dtypes)
    df = pd.read_csv(INPUT + f'train_{i}.csv', usecols = usecols)
    df.to_parquet(f'train_{i}.parquet.gzip', compression='gzip')
    print('Done with File', i)
    train.append(pd.read_parquet(f'train_{i}.parquet.gzip'))

# dft = pd.read_csv(INPUT + 'test.csv', dtype = test_dtypes)
dft = pd.read_csv(INPUT + 'test.csv')
dft.to_parquet('test.parquet.gzip', compression='gzip')
print('Done with File test')
test = pd.read_parquet('test.parquet.gzip')
df_sample = pd.read_csv(INPUT + 'sample_submission.csv')

test = test.drop(boost_t, axis=1)


# **Preprocessing data**
# 
# Thanks to @Jose Cáliz for feature engineering ideas!

# In[4]:


# for i in range(num):
#     print(train_list[i].shape)
#     games = random.sample(list(train_list[i].game_num.unique()), 300)
#     train_list[i] = train_list[i][train_list[i].game_num.isin(games)]
#     print(train_list[i].shape)


# In[5]:


for i in range(num):
    train[i]['label'] = train[i].team_A_scoring_within_10sec + train[i].team_B_scoring_within_10sec.replace(1, 2)
    train[i].label.value_counts(True).to_frame(name='label proportion')


# In[6]:


for i in range(num):
#     train[i] = train[i].fillna(train[i].median())
    train[i] = train[i].fillna(0)
    
# test = test.fillna(test.median())
test = test.fillna(0)


# **Feature engineering**

# In[7]:


for i in range(num):
    train[i]['ball_distance_to_goal_A'] = np.sqrt(
        (train[i].ball_pos_x)**2 + (train[i].ball_pos_y + 100)**2 + (train[i].ball_pos_z)**2
    )
    train[i]['ball_distance_to_goal_B'] = np.sqrt(
        (train[i].ball_pos_x)**2 + (train[i].ball_pos_y - 100)**2 + (train[i].ball_pos_z)**2
    )
    
    train[i]['ball_pos'] = np.sqrt(train[i]['ball_pos_x']**2 + train[i]['ball_pos_y']**2 + train[i]['ball_pos_z']**2)
    train[i]['ball_vel'] = np.sqrt(train[i]['ball_vel_x']**2 + train[i]['ball_vel_y']**2 + train[i]['ball_vel_z']**2)
    
    train[i]['p0_pos'] = np.sqrt(train[i]['p0_pos_x']**2 + train[i]['p0_pos_y']**2 + train[i]['p0_pos_z']**2)
    train[i]['p1_pos'] = np.sqrt(train[i]['p1_pos_x']**2 + train[i]['p1_pos_y']**2 + train[i]['p1_pos_z']**2)
    train[i]['p2_pos'] = np.sqrt(train[i]['p2_pos_x']**2 + train[i]['p2_pos_y']**2 + train[i]['p2_pos_z']**2)
    train[i]['p3_pos'] = np.sqrt(train[i]['p3_pos_x']**2 + train[i]['p3_pos_y']**2 + train[i]['p3_pos_z']**2)
    train[i]['p4_pos'] = np.sqrt(train[i]['p4_pos_x']**2 + train[i]['p4_pos_y']**2 + train[i]['p4_pos_z']**2)
    train[i]['p5_pos'] = np.sqrt(train[i]['p5_pos_x']**2 + train[i]['p5_pos_y']**2 + train[i]['p5_pos_z']**2)
    
    train[i]['p0_vel'] = np.sqrt(train[i]['p0_vel_x']**2 + train[i]['p0_vel_y']**2 + train[i]['p0_vel_z']**2)
    train[i]['p1_vel'] = np.sqrt(train[i]['p1_vel_x']**2 + train[i]['p1_vel_y']**2 + train[i]['p1_vel_z']**2)
    train[i]['p2_vel'] = np.sqrt(train[i]['p2_vel_x']**2 + train[i]['p2_vel_y']**2 + train[i]['p2_vel_z']**2)
    train[i]['p3_vel'] = np.sqrt(train[i]['p3_vel_x']**2 + train[i]['p3_vel_y']**2 + train[i]['p3_vel_z']**2)
    train[i]['p4_vel'] = np.sqrt(train[i]['p4_vel_x']**2 + train[i]['p4_vel_y']**2 + train[i]['p4_vel_z']**2)
    train[i]['p5_vel'] = np.sqrt(train[i]['p5_vel_x']**2 + train[i]['p5_vel_y']**2 + train[i]['p5_vel_z']**2)
    
    train[i]['p0_to_ball'] = np.sqrt((train[i]['p0_pos_x']-train[i]['ball_pos_x'])**2 + (train[i]['p0_pos_y'] - train[i]['ball_pos_y'])**2 + (train[i]['p0_pos_z'] - train[i]['ball_pos_z'])**2)
    train[i]['p1_to_ball'] = np.sqrt((train[i]['p1_pos_x']-train[i]['ball_pos_x'])**2 + (train[i]['p1_pos_y'] - train[i]['ball_pos_y'])**2 + (train[i]['p1_pos_z'] - train[i]['ball_pos_z'])**2)
    train[i]['p2_to_ball'] = np.sqrt((train[i]['p2_pos_x']-train[i]['ball_pos_x'])**2 + (train[i]['p2_pos_y'] - train[i]['ball_pos_y'])**2 + (train[i]['p2_pos_z'] - train[i]['ball_pos_z'])**2)
    train[i]['p3_to_ball'] = np.sqrt((train[i]['p3_pos_x']-train[i]['ball_pos_x'])**2 + (train[i]['p3_pos_y'] - train[i]['ball_pos_y'])**2 + (train[i]['p3_pos_z'] - train[i]['ball_pos_z'])**2)
    train[i]['p4_to_ball'] = np.sqrt((train[i]['p4_pos_x']-train[i]['ball_pos_x'])**2 + (train[i]['p4_pos_y'] - train[i]['ball_pos_y'])**2 + (train[i]['p4_pos_z'] - train[i]['ball_pos_z'])**2)
    train[i]['p5_to_ball'] = np.sqrt((train[i]['p5_pos_x']-train[i]['ball_pos_x'])**2 + (train[i]['p5_pos_y'] - train[i]['ball_pos_y'])**2 + (train[i]['p5_pos_z'] - train[i]['ball_pos_z'])**2)
    
    train[i]['p0_to_goal'] = np.sqrt((train[i]['p0_pos_x'])**2 + (train[i]['p0_pos_y'] + 100)**2 + (train[i]['p0_pos_z'])**2)
    train[i]['p1_to_goal'] = np.sqrt((train[i]['p1_pos_x'])**2 + (train[i]['p1_pos_y'] + 100)**2 + (train[i]['p1_pos_z'])**2)
    train[i]['p2_to_goal'] = np.sqrt((train[i]['p2_pos_x'])**2 + (train[i]['p2_pos_y'] + 100)**2 + (train[i]['p2_pos_z'])**2)
    train[i]['p3_to_goal'] = np.sqrt((train[i]['p3_pos_x'])**2 + (train[i]['p3_pos_y'] - 100)**2 + (train[i]['p3_pos_z'])**2)
    train[i]['p4_to_goal'] = np.sqrt((train[i]['p4_pos_x'])**2 + (train[i]['p4_pos_y'] - 100)**2 + (train[i]['p4_pos_z'])**2)
    train[i]['p5_to_goal'] = np.sqrt((train[i]['p5_pos_x'])**2 + (train[i]['p5_pos_y'] - 100)**2 + (train[i]['p5_pos_z'])**2)
    
test['ball_distance_to_goal_A'] = np.sqrt(
        (test.ball_pos_x)**2 + (test.ball_pos_y + 100)**2 + (test.ball_pos_z)**2
    )
test['ball_distance_to_goal_B'] = np.sqrt(
        (test.ball_pos_x)**2 + (test.ball_pos_y - 100)**2 + (test.ball_pos_z)**2
    )

test['ball_pos'] = np.sqrt(test['ball_pos_x']**2 + test['ball_pos_y']**2 + test['ball_pos_z']**2)
test['ball_vel'] = np.sqrt(test['ball_vel_x']**2 + test['ball_vel_y']**2 + test['ball_vel_z']**2)

test['p0_pos'] = np.sqrt(test['p0_pos_x']**2 + test['p0_pos_y']**2 + test['p0_pos_z']**2)
test['p1_pos'] = np.sqrt(test['p1_pos_x']**2 + test['p1_pos_y']**2 + test['p1_pos_z']**2)
test['p2_pos'] = np.sqrt(test['p2_pos_x']**2 + test['p2_pos_y']**2 + test['p2_pos_z']**2)
test['p3_pos'] = np.sqrt(test['p3_pos_x']**2 + test['p3_pos_y']**2 + test['p3_pos_z']**2)
test['p4_pos'] = np.sqrt(test['p4_pos_x']**2 + test['p4_pos_y']**2 + test['p4_pos_z']**2)
test['p5_pos'] = np.sqrt(test['p5_pos_x']**2 + test['p5_pos_y']**2 + test['p5_pos_z']**2)
    
test['p0_vel'] = np.sqrt(test['p0_vel_x']**2 + test['p0_vel_y']**2 + test['p0_vel_z']**2)
test['p1_vel'] = np.sqrt(test['p1_vel_x']**2 + test['p1_vel_y']**2 + test['p1_vel_z']**2)
test['p2_vel'] = np.sqrt(test['p2_vel_x']**2 + test['p2_vel_y']**2 + test['p2_vel_z']**2)
test['p3_vel'] = np.sqrt(test['p3_vel_x']**2 + test['p3_vel_y']**2 + test['p3_vel_z']**2)
test['p4_vel'] = np.sqrt(test['p4_vel_x']**2 + test['p4_vel_y']**2 + test['p4_vel_z']**2)
test['p5_vel'] = np.sqrt(test['p5_vel_x']**2 + test['p5_vel_y']**2 + test['p5_vel_z']**2)
    
test['p0_to_ball'] = np.sqrt((test['p0_pos_x'] - test['ball_pos_x'])**2 + (test['p0_pos_y'] - test['ball_pos_y'])**2 + (test['p0_pos_z'] - test['ball_pos_z'])**2)
test['p1_to_ball'] = np.sqrt((test['p1_pos_x'] - test['ball_pos_x'])**2 + (test['p1_pos_y'] - test['ball_pos_y'])**2 + (test['p1_pos_z'] - test['ball_pos_z'])**2)
test['p2_to_ball'] = np.sqrt((test['p2_pos_x'] - test['ball_pos_x'])**2 + (test['p2_pos_y'] - test['ball_pos_y'])**2 + (test['p2_pos_z'] - test['ball_pos_z'])**2)
test['p3_to_ball'] = np.sqrt((test['p3_pos_x'] - test['ball_pos_x'])**2 + (test['p3_pos_y'] - test['ball_pos_y'])**2 + (test['p3_pos_z'] - test['ball_pos_z'])**2)
test['p4_to_ball'] = np.sqrt((test['p4_pos_x'] - test['ball_pos_x'])**2 + (test['p4_pos_y'] - test['ball_pos_y'])**2 + (test['p4_pos_z'] - test['ball_pos_z'])**2)
test['p5_to_ball'] = np.sqrt((test['p5_pos_x'] - test['ball_pos_x'])**2 + (test['p5_pos_y'] - test['ball_pos_y'])**2 + (test['p5_pos_z'] - test['ball_pos_z'])**2)

test['p0_to_goal'] = np.sqrt((test['p0_pos_x'])**2 + (test['p0_pos_y'] + 100)**2 + (test['p0_pos_z'])**2)
test['p1_to_goal'] = np.sqrt((test['p1_pos_x'])**2 + (test['p1_pos_y'] + 100)**2 + (test['p1_pos_z'])**2)
test['p2_to_goal'] = np.sqrt((test['p2_pos_x'])**2 + (test['p2_pos_y'] + 100)**2 + (test['p2_pos_z'])**2)
test['p3_to_goal'] = np.sqrt((test['p3_pos_x'])**2 + (test['p3_pos_y'] - 100)**2 + (test['p3_pos_z'])**2)
test['p4_to_goal'] = np.sqrt((test['p4_pos_x'])**2 + (test['p4_pos_y'] - 100)**2 + (test['p4_pos_z'])**2)
test['p5_to_goal'] = np.sqrt((test['p5_pos_x'])**2 + (test['p5_pos_y'] - 100)**2 + (test['p5_pos_z'])**2)
    


# In[8]:


# from sklearn.decomposition import PCA

# ball_transform = PCA(4)

# ball_cols = ['ball_vel_x', 'ball_vel_y', 'ball_vel_z', 'ball_pos_x', 'ball_pos_y', 'ball_pos_z']

# ball_transform.fit(train[0][ball_cols])

# for i in range(num):
#     x = ball_transform.transform(train[i][ball_cols])
#     train[i] = train[i].drop(ball_cols, axis=1)
#     x = x.T
#     train[i]['ball0'], train[i]['ball1'], train[i]['ball2'], train[i]['ball3'] = x[0], x[1], x[2], x[3]
    
# x = ball_transform.transform(test[ball_cols])
# test = test.drop(ball_cols, axis=1)
# x = x.T
# test['ball0'], test['ball1'], test['ball2'], test['ball3'] = x[0], x[1], x[2], x[3]

# ball_transform.explained_variance_ratio_


# In[9]:


# pl0_cols = ['p0_pos_x', 'p0_pos_y', 'p0_pos_z']
# pl1_cols = ['p1_pos_x', 'p1_pos_y', 'p1_pos_z']
# pl2_cols = ['p2_pos_x', 'p2_pos_y', 'p2_pos_z']
# pl3_cols = ['p3_pos_x', 'p3_pos_y', 'p3_pos_z']
# pl4_cols = ['p4_pos_x', 'p4_pos_y', 'p4_pos_z']
# pl5_cols = ['p5_pos_x', 'p5_pos_y', 'p5_pos_z']

# pl0_v_cols = ['p0_vel_x', 'p0_vel_y', 'p0_vel_z']
# pl1_v_cols = ['p1_vel_x', 'p1_vel_y', 'p1_vel_z']
# pl2_v_cols = ['p2_vel_x', 'p2_vel_y', 'p2_vel_z']
# pl3_v_cols = ['p3_vel_x', 'p3_vel_y', 'p3_vel_z']
# pl4_v_cols = ['p4_vel_x', 'p4_vel_y', 'p4_vel_z']
# pl5_v_cols = ['p5_vel_x', 'p5_vel_y', 'p5_vel_z']

# pl_v_z_cols = ['p0_vel_z', 'p1_vel_z', 'p2_vel_z', 'p3_vel_z', 'p4_vel_z', 'p5_vel_z']

# pl = pl0_cols + pl1_cols + pl2_cols + pl3_cols + pl4_cols + pl5_cols
# pl_v = pl0_v_cols + pl1_v_cols + pl2_v_cols + pl3_v_cols + pl4_v_cols + pl5_v_cols

# players_transform = PCA(2)

# players_transform.fit(train[0][pl])

# for i in range(num):
#     x = players_transform.transform(train[i][pl])
#     x = x.T
#     train[i] = train[i].drop(pl, axis=1)
#     train[i]['pl0'], train[i]['pl1'] = x[0], x[1]
    
# x = players_transform.transform(test[pl])
# x = x.T
# test = test.drop(pl, axis=1)
# test['pl0'], test['pl1'] = x[0], x[1]

# for i in range(num):
#     train[i] = train[i].drop(pl_v_z_cols, axis=1)
# test = test.drop(pl_v_z_cols, axis=1)

# players_transform.explained_variance_ratio_


# In[10]:


# pl_boost = ['p0_boost', 'p1_boost', 'p2_boost', 'p3_boost', 'p4_boost', 'p5_boost']

# pl_boost_transform = PCA(6)

# pl_boost_transform.fit(train[0][pl_boost])

# for i in range(num):
#     x = pl_boost_transform.transform(train[i][pl_boost])
#     x = x.T
#     train[i] = train[i].drop(pl_boost, axis=1)
#     train[i]['pl_b1'], train[i]['pl_b2'] = x[0], x[1]
    
# x = pl_boost_transform.transform(test[pl_boost])
# x = x.T
# test = test.drop(pl_boost, axis=1)
# test['pl_b1'], test['pl_b2'] = x[0], x[1]

# pl_boost_transform.explained_variance_ratio_


# In[11]:


# boost_t = ['boost0_timer', 'boost1_timer', 'boost2_timer', 'boost3_timer', 'boost4_timer', 'boost5_timer']

# boost_t_transform = PCA(6)

# boost_t_transform.fit(train[0][boost_t])

# for i in range(num):
#     x = boost_t_transform.transform(train[i][boost_t])
#     x = x.T
#     train[i] = train[i].drop(boost_t, axis=1)
#     train[i]['b1'], train[i]['b2'] = x[0], x[1]
    
# x = boost_t_transform.transform(test[boost_t])
# x = x.T
# test = test.drop(boost_t, axis=1)
# test['b1'], test['b2'] = x[0], x[1]

# boost_t_transform.explained_variance_ratio_

# for i in range(num):
#     train[i] = train[i].drop(boost_t, axis=1)
# test = test.drop(boost_t, axis=1)


# In[12]:


test


# In[13]:


def mirror_x(df, columns):
    data = pd.DataFrame(df, columns = columns)
    data['ball_pos_x'] = - data['ball_pos_x']
    data['ball_vel_x'] = - data['ball_vel_x']
    data['p0_pos_x'] = - data['p0_pos_x']
    data['p0_vel_x'] = - data['p0_vel_x']
    data['p1_pos_x'] = - data['p1_pos_x']
    data['p1_vel_x'] = - data['p1_vel_x']
    data['p2_pos_x'] = - data['p2_pos_x']
    data['p2_vel_x'] = - data['p2_vel_x']
    data['p3_pos_x'] = - data['p3_pos_x']
    data['p3_vel_x'] = - data['p3_vel_x']
    data['p4_pos_x'] = - data['p4_pos_x']
    data['p4_vel_x'] = - data['p4_vel_x']
    data['p5_pos_x'] = - data['p5_pos_x']
    data['p5_vel_x'] = - data['p5_vel_x']
    return data.to_numpy() 


# In[14]:


def change_players(df, columns):
    data = pd.DataFrame(df, columns = columns)
    data['p0_pos_x'], data['p1_pos_x'], data['p2_pos_x'] = data['p1_pos_x'], data['p2_pos_x'], data['p0_pos_x']
    data['p0_pos_y'], data['p1_pos_y'], data['p2_pos_y'] = data['p1_pos_y'], data['p2_pos_y'], data['p0_pos_y']
    data['p0_pos_z'], data['p1_pos_z'], data['p2_pos_z'] = data['p1_pos_z'], data['p2_pos_z'], data['p0_pos_z']
    data['p0_vel_x'], data['p1_vel_x'], data['p2_vel_x'] = data['p1_vel_x'], data['p2_vel_x'], data['p0_vel_x']
    data['p0_vel_y'], data['p1_vel_y'], data['p2_vel_y'] = data['p1_vel_y'], data['p2_vel_y'], data['p0_vel_y']
    data['p0_vel_z'], data['p1_vel_z'], data['p2_vel_z'] = data['p1_vel_z'], data['p2_vel_z'], data['p0_vel_z']
    data['p0_boost'], data['p1_boost'], data['p2_boost'] = data['p1_boost'], data['p2_boost'], data['p0_boost']
    
    data['p3_pos_x'], data['p4_pos_x'], data['p5_pos_x'] = data['p4_pos_x'], data['p5_pos_x'], data['p3_pos_x']
    data['p3_pos_y'], data['p4_pos_y'], data['p5_pos_y'] = data['p4_pos_y'], data['p5_pos_y'], data['p3_pos_y']
    data['p3_pos_z'], data['p4_pos_z'], data['p5_pos_z'] = data['p4_pos_z'], data['p5_pos_z'], data['p3_pos_z']
    data['p3_vel_x'], data['p4_vel_x'], data['p5_vel_x'] = data['p4_vel_x'], data['p5_vel_x'], data['p3_vel_x']
    data['p3_vel_y'], data['p4_vel_y'], data['p5_vel_y'] = data['p4_vel_y'], data['p5_vel_y'], data['p3_vel_y']
    data['p3_vel_z'], data['p4_vel_z'], data['p5_vel_z'] = data['p4_vel_z'], data['p5_vel_z'], data['p3_vel_z']
    data['p3_boost'], data['p4_boost'], data['p5_boost'] = data['p4_boost'], data['p5_boost'], data['p3_boost']
    return data.to_numpy()


# In[15]:


def make_inv(df, columns):
    data = pd.DataFrame(df, columns = columns)
    data['ball_distance_to_goal_A'], data['ball_distance_to_goal_B'] = data['ball_distance_to_goal_B'], data['ball_distance_to_goal_A']
    data['ball_pos_y'] = - data['ball_pos_y']
    data['ball_vel_y'] = - data['ball_vel_y']
    data['p0_pos_x'], data['p3_pos_x'] = - data['p3_pos_x'], - data['p0_pos_x']
    data['p0_pos_y'], data['p3_pos_y'] = - data['p3_pos_y'], - data['p0_pos_y']
    data['p0_pos_z'], data['p3_pos_z'] = data['p3_pos_z'], data['p0_pos_z']
    data['p0_vel_x'], data['p3_vel_x'] = - data['p3_vel_x'], - data['p0_vel_x']
    data['p0_vel_y'], data['p3_vel_y'] = - data['p3_vel_y'], - data['p0_vel_y']
    data['p0_vel_z'], data['p3_vel_z'] = data['p3_vel_z'], data['p0_vel_z']
    data['p1_pos_x'], data['p4_pos_x'] = - data['p4_pos_x'], - data['p1_pos_x']
    data['p1_pos_y'], data['p4_pos_y'] = - data['p4_pos_y'], - data['p1_pos_y']
    data['p1_pos_z'], data['p4_pos_z'] = data['p4_pos_z'], data['p1_pos_z']
    data['p1_vel_x'], data['p4_vel_x'] = - data['p4_vel_x'], - data['p1_vel_x']
    data['p1_vel_y'], data['p4_vel_y'] = - data['p4_vel_y'], - data['p1_vel_y']
    data['p1_vel_z'], data['p4_vel_z'] = data['p4_vel_z'], data['p1_vel_z']
    data['p2_pos_x'], data['p5_pos_x'] = - data['p5_pos_x'], - data['p2_pos_x']
    data['p2_pos_y'], data['p5_pos_y'] = - data['p5_pos_y'], - data['p2_pos_y']
    data['p2_pos_z'], data['p5_pos_z'] = data['p5_pos_z'], data['p2_pos_z']
    data['p2_vel_x'], data['p5_vel_x'] = - data['p5_vel_x'], - data['p2_vel_x']
    data['p2_vel_y'], data['p5_vel_y'] = - data['p5_vel_y'], - data['p2_vel_y']
    data['p2_vel_z'], data['p5_vel_z'] = data['p5_vel_z'], data['p2_vel_z']
    data['p0_boost'], data['p3_boost'] = data['p3_boost'], data['p0_boost']
    data['p1_boost'], data['p4_boost'] = data['p4_boost'], data['p1_boost']
    data['p2_boost'], data['p5_boost'] = data['p5_boost'], data['p2_boost']
    return data.to_numpy()


# In[16]:


test


# In[17]:


target = []
# train = []
for i in range(num):
    target.append(pd.get_dummies(train[i]['label']))
#     target.append(train_list[i][['team_A_scoring_within_10sec','team_B_scoring_within_10sec']])
#     train.append(train_list[i].drop(['game_num', 'event_id', 'event_time', 'player_scoring_next', 'team_scoring_next', 'team_A_scoring_within_10sec', 'team_B_scoring_within_10sec', 'label'], axis = 1))
    train[i] = train[i].drop(['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec', 'label'], axis = 1)

for i in range(num):
    target[i].columns = ['nobody_scores', 'team_A_scores', 'team_B_scores']
    
test = test.drop(['id'], axis = 1)


# In[18]:


col_list = test.columns


# In[19]:


# from sklearn.feature_selection import mutual_info_regression

# def make_mi_scores(X, y):
#     mi_scores = mutual_info_regression(X, y)
#     mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
#     mi_scores = mi_scores.sort_values(ascending=False)
#     return mi_scores

# mi_scores = make_mi_scores(train[0].head(500000), target[0]['team_B_scores'].head(500000))
# mi_scores


# In[20]:


# mi_scores.tail(20)
# p_to_ball, ball_pos, ball_pos_z, p_vel_z, p_pos_x


# In[21]:


# scaler = StandardScaler()
scaler = SKWrapper(StandardScaler(), variables=train[0].columns.tolist())
scaler.fit(train[0])
for i in range(num):
    train[i] = scaler.transform(train[i])
test = scaler.transform(test)


# In[22]:


# X_train, X_valid, y_train, y_valid = train_test_split(train[0], target[0], test_size = 0.2, shuffle=True)


# **Model evaluation**
# 
# Thanks to @WEI XIE for Catboost details :)

# In[23]:


import catboost

MAX_ITER = 6000
PATIENCE = 100
DISPLAY_FREQ = 100

MODEL_PARAMS = {'random_seed': 1234,    
                'learning_rate': 0.01,                
                'iterations': MAX_ITER,
                'early_stopping_rounds': PATIENCE,
                'metric_period': DISPLAY_FREQ,
                'use_best_model': True,
                'eval_metric': 'Logloss',
                'task_type': 'GPU'
               }


predsA = []
predsB = []

for i in range(num-1):
    mdl = catboost.CatBoostClassifier(**MODEL_PARAMS)
    mdl.fit(X=train[i], y=target[i]['team_A_scores'],
          eval_set=[(train[num-1], target[num-1]['team_A_scores'])],
          early_stopping_rounds = PATIENCE,
          metric_period = DISPLAY_FREQ)
    predsA.append(mdl.predict_proba(test)[:,1].T)
    mdl = catboost.CatBoostClassifier(**MODEL_PARAMS)
    mdl.fit(X=mirror_x(train[i], col_list), y=target[i]['team_A_scores'],
          eval_set=[(train[num-1], target[num-1]['team_A_scores'])],
          early_stopping_rounds = PATIENCE,
          metric_period = DISPLAY_FREQ)
    predsA.append(mdl.predict_proba(test)[:,1].T)
    
    
#     mdl.fit(X=change_players(train[i], col_list), y=target[i]['team_A_scores'],
#           eval_set=[(train[num-1], target[num-1]['team_A_scores'])],
#           early_stopping_rounds = PATIENCE,
#           metric_period = DISPLAY_FREQ)
#     predsA.append(mdl.predict_proba(test)[:,1].T)
#     mdl.fit(X=change_players(change_players(train[i], col_list), col_list), y=target[i]['team_A_scores'],
#           eval_set=[(train[num-1], target[num-1]['team_A_scores'])],
#           early_stopping_rounds = PATIENCE,
#           metric_period = DISPLAY_FREQ)
#     predsA.append(mdl.predict_proba(test)[:,1].T)
    
#     mdl.fit(X=make_inv(train[i], col_list), y=target[i]['team_B_scores'],
#           eval_set=[(train[num-1], target[num-1]['team_A_scores'])],
#           early_stopping_rounds = PATIENCE,
#           metric_period = DISPLAY_FREQ)
#     predsA.append(mdl.predict_proba(test)[:,1].T)


for i in range(num-1):
    mdl = catboost.CatBoostClassifier(**MODEL_PARAMS)
    mdl.fit(X=train[i], y=target[i]['team_B_scores'],
          eval_set=[(train[num-1], target[num-1]['team_B_scores'])],
          early_stopping_rounds = PATIENCE,
          metric_period = DISPLAY_FREQ)
    predsB.append(mdl.predict_proba(test)[:,1].T)
    mdl = catboost.CatBoostClassifier(**MODEL_PARAMS)
    mdl.fit(X=mirror_x(train[i], col_list), y=target[i]['team_B_scores'],
          eval_set=[(train[num-1], target[num-1]['team_B_scores'])],
          early_stopping_rounds = PATIENCE,
          metric_period = DISPLAY_FREQ)
    predsB.append(mdl.predict_proba(test)[:,1].T)
    
    
#     mdl.fit(X=change_players(train[i],col_list), y=target[i]['team_B_scores'],
#           eval_set=[(train[num-1], target[num-1]['team_B_scores'])],
#           early_stopping_rounds = PATIENCE,
#           metric_period = DISPLAY_FREQ)
#     predsB.append(mdl.predict_proba(test)[:,1].T)
#     mdl.fit(X=change_players(change_players(train[i],col_list),col_list), y=target[i]['team_B_scores'],
#           eval_set=[(train[num-1], target[num-1]['team_B_scores'])],
#           early_stopping_rounds = PATIENCE,
#           metric_period = DISPLAY_FREQ)
#     predsB.append(mdl.predict_proba(test)[:,1].T)
    
#     mdl.fit(X=make_inv(train[i],col_list), y=target[i]['team_A_scores'],
#           eval_set=[(train[num-1], target[num-1]['team_B_scores'])],
#           early_stopping_rounds = PATIENCE,
#           metric_period = DISPLAY_FREQ)
#     predsB.append(mdl.predict_proba(test)[:,1].T)
    
#  ++   
    
mdl = catboost.CatBoostClassifier(**MODEL_PARAMS)
mdl.fit(X=train[num-1], y=target[num-1]['team_A_scores'],
          eval_set=[(train[0], target[0]['team_A_scores'])],
          early_stopping_rounds = PATIENCE,
          metric_period = DISPLAY_FREQ)
predsA.append(mdl.predict_proba(test)[:,1].T)
mdl = catboost.CatBoostClassifier(**MODEL_PARAMS)
mdl.fit(X=mirror_x(train[num-1], col_list), y=target[num-1]['team_A_scores'],
          eval_set=[(train[0], target[0]['team_A_scores'])],
          early_stopping_rounds = PATIENCE,
          metric_period = DISPLAY_FREQ)
predsA.append(mdl.predict_proba(test)[:,1].T)

# mdl.fit(X=change_players(train[num-1],col_list), y=target[num-1]['team_A_scores'],
#           eval_set=[(train[0], target[0]['team_A_scores'])],
#           early_stopping_rounds = PATIENCE,
#           metric_period = DISPLAY_FREQ)
# predsA.append(mdl.predict_proba(test)[:,1].T)
# mdl.fit(X=change_players(change_players(train[num-1],col_list),col_list), y=target[num-1]['team_A_scores'],
#           eval_set=[(train[0], target[0]['team_A_scores'])],
#           early_stopping_rounds = PATIENCE,
#           metric_period = DISPLAY_FREQ)
# predsA.append(mdl.predict_proba(test)[:,1].T)

# mdl.fit(X=make_inv(train[num-1], col_list), y=target[num-1]['team_B_scores'],
#           eval_set=[(train[0], target[0]['team_A_scores'])],
#           early_stopping_rounds = PATIENCE,
#           metric_period = DISPLAY_FREQ)
# predsA.append(mdl.predict_proba(test)[:,1].T)

mdl = catboost.CatBoostClassifier(**MODEL_PARAMS)
mdl.fit(X=train[num-1], y=target[num-1]['team_B_scores'],
          eval_set=[(train[0], target[0]['team_B_scores'])],
          early_stopping_rounds = PATIENCE,
          metric_period = DISPLAY_FREQ)
predsB.append(mdl.predict_proba(test)[:,1].T)
mdl = catboost.CatBoostClassifier(**MODEL_PARAMS)
mdl.fit(X=mirror_x(train[num-1], col_list), y=target[num-1]['team_B_scores'],
          eval_set=[(train[0], target[0]['team_B_scores'])],
          early_stopping_rounds = PATIENCE,
          metric_period = DISPLAY_FREQ)
predsB.append(mdl.predict_proba(test)[:,1].T)

# mdl.fit(X=change_players(train[num-1], col_list), y=target[num-1]['team_B_scores'],
#           eval_set=[(train[0], target[0]['team_B_scores'])],
#           early_stopping_rounds = PATIENCE,
#           metric_period = DISPLAY_FREQ)
# predsB.append(mdl.predict_proba(test)[:,1].T)
# mdl.fit(X=change_players(change_players(train[num-1], col_list),col_list), y=target[num-1]['team_B_scores'],
#           eval_set=[(train[0], target[0]['team_B_scores'])],
#           early_stopping_rounds = PATIENCE,
#           metric_period = DISPLAY_FREQ)
# predsB.append(mdl.predict_proba(test)[:,1].T)

# mdl.fit(X=make_inv(train[num-1],col_list), y=target[num-1]['team_A_scores'],
#           eval_set=[(train[0], target[0]['team_B_scores'])],
#           early_stopping_rounds = PATIENCE,
#           metric_period = DISPLAY_FREQ)
# predsB.append(mdl.predict_proba(test)[:,1].T)

# --

# mdlA = catboost.CatBoostClassifier(**MODEL_PARAMS)
# for i in range(num-1):
#     mdlA.fit(X=train[i], y=target[i]['team_A_scores'],
#           eval_set=[(train[num-1], target[num-1]['team_A_scores'])],
#           early_stopping_rounds = PATIENCE,
#           metric_period = DISPLAY_FREQ)
# mdlA.fit(X=train[num-1], y=target[num-1]['team_A_scores'],
#           eval_set=[(train[0], target[0]['team_A_scores'])],
#           early_stopping_rounds = PATIENCE,
#           metric_period = DISPLAY_FREQ)

# mdlB = catboost.CatBoostClassifier(**MODEL_PARAMS)
# for i in range(num-1):
#     mdlB.fit(X=train[i], y=target[i]['team_B_scores'],
#           eval_set=[(train[num-1], target[num-1]['team_B_scores'])],
#           early_stopping_rounds = PATIENCE,
#           metric_period = DISPLAY_FREQ)
# mdlB.fit(X=train[num-1], y=target[num-1]['team_B_scores'],
#           eval_set=[(train[0], target[0]['team_B_scores'])],
#           early_stopping_rounds = PATIENCE,
#           metric_period = DISPLAY_FREQ)


# In[24]:


# from catboost import CatBoost

# model = CatBoost()

# grid = {'learning_rate': [0.5],
#         'depth': [6],
#         'l2_leaf_reg': [1, 3, 5, 7]}

# grid_search_result = model.grid_search(grid, 
#                                        X=train[0], 
#                                        y=target[0]['team_A_scores'],
#                                        cv=3,
#                                        plot=True, 
#                                        verbose=100)


# **Making prediction**

# In[25]:


predictionA = np.average(np.array(predsA),axis=0)
predictionB = np.average(np.array(predsB),axis=0)
# predictionA = mdlA.predict_proba(test)[:,1].T
# predictionB = mdlB.predict_proba(test)[:,1].T

preds = pd.DataFrame([predictionA, predictionB])
preds = preds.T
preds.columns = ['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec']
preds


# In[26]:


df_sample[['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec']] = preds[['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec']]


# In[27]:


df_sample.to_csv('submission.csv', index = False)

