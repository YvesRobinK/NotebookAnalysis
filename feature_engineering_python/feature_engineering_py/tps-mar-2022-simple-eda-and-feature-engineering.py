#!/usr/bin/env python
# coding: utf-8

# # ***[Tabular Playground Series - Mar 2022] EDA***

# <img src="https://www.bigdata-navi.com/aidrops/wp-content/uploads/2019/07/0Kehovl-1024x585.jpg" width="500">

# The purpose of this notebook is to analyze the data of this competition so that even beginners can understand. I've written other simple EDA notebooks like this. If you have time, please check it.
# * [House Price - Simple EDA and XGBRegressor](https://www.kaggle.com/code/packinman/house-price-simple-eda-and-xgbregressor)
# * [Titanic - Simple EDA and RandomForest](https://www.kaggle.com/code/packinman/titanic-simple-eda-and-randomforest)

# # Import the libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # EDA

# ## 1. Training Data

# ### load the training data

# In[2]:


train_df = pd.read_csv('../input/tabular-playground-series-mar-2022/train.csv')
train_df


# training data consists of measurements of traffic congestion across 65 roadways from April through September of 1991.
# 
# * row_id - a unique identifier for this instance (int64)
# * time - the 20-minute period in which each measurement was taken (object)
# * x - the east-west midpoint coordinate of the roadway (int64)
# * y - the north-south midpoint coordinate of the roadway (int64)
# * direction - the direction of travel of the roadway. EB indicates "eastbound" travel, for example, while SW indicates a "southwest" direction of travel. (object)
# * congestion - congestion levels for the roadway during each hour; the target. (int64)

# In[3]:


train_df.describe()


# ### Check the missing values

# In[4]:


train_df.isnull().sum()


# There is no missing value.

# ### Check the value of each feature

# In[5]:


train_df['x'].unique()


# In[6]:


train_df['y'].unique()


# In[7]:


train_df['direction'].unique()


# ### Visualize the distribution of direction

# In[8]:


sns.histplot(train_df['direction'])
plt.tight_layout()


# ### Visualize the distribution of congestion

# In[9]:


sns.histplot(train_df['congestion'], kde=True)
plt.tight_layout()


# It seems to be normalized to the range 0 to 100.

# ### Feature engineering

# In[10]:


def feature_engineering(data):
    data['time'] = pd.to_datetime(data['time'])
    data['month'] = data['time'].dt.month
    data['weekday'] = data['time'].dt.weekday
    data['hour'] = data['time'].dt.hour
    data['minute'] = data['time'].dt.minute
    data['is_month_start'] = data['time'].dt.is_month_start.astype('int')
    data['is_month_end'] = data['time'].dt.is_month_end.astype('int')
    data['is_weekend'] = (data['time'].dt.dayofweek > 5).astype('int')
    data['is_afternoon'] = (data['time'].dt.hour > 12).astype('int')
    data['road'] = data['x'].astype(str) + data['y'].astype(str) + data['direction']
    
    data = data.drop(['row_id', 'direction', 'x', 'y'], axis=1)
    
    return data


# In[11]:


train_df = feature_engineering(train_df)
train_df


# I decomposed 'time' into 'month', 'weekday', 'hour', and 'minute'. And I combined 'x', ' y', and 'direction' into 'road'.
# I also generated features that can identify the end of the month, the beginning of the month, the weekend, and the afternoon.

# In[12]:


train_df.describe()


# ### Visualize the effect of each feature on congestion

# In[13]:


features = ['month', 'weekday', 'hour', 'is_month_start', 'is_month_end', 'is_weekend', 'is_afternoon']

plt.subplots(figsize=(16,16))

for i, feature in enumerate(features):
    plt.subplot(4, 3, i+1)
    sns.boxplot(x=train_df[feature], y=train_df['congestion'])
    plt.tight_layout()


# There is not much difference in month, is_month_end and is_month_start

# ### Visualize the effect of hour on congestion group by road

# In[14]:


roads = train_df['road'].unique()

plt.subplots(figsize=(12,80))

for i, road in enumerate(roads):
    plt.subplot(30, 3, i+1)
    plt.title(roads[i])
    sns.barplot(x=train_df['hour'][train_df['road']==road], y=train_df['congestion'])
    plt.tight_layout()


# ### Visualize the effect of weekday on congestion group by road

# In[15]:


plt.subplots(figsize=(12,80))

for i, road in enumerate(roads):
    plt.subplot(30, 3, i+1)
    plt.title(roads[i])
    sns.barplot(x=train_df['weekday'][train_df['road']==road], y=train_df['congestion'])
    plt.tight_layout()


# ### Visualize the effect of is_afternoon on congestion group by road

# In[16]:


plt.subplots(figsize=(12,80))

for i, road in enumerate(roads):
    plt.subplot(30, 3, i+1)
    plt.title(roads[i])
    sns.barplot(x=train_df['is_afternoon'][train_df['road']==road], y=train_df['congestion'])
    plt.tight_layout()


# Only 00NB, 00SB, 21SB, 21SW, 22NE, 23NE have more traffic in the morning

# ## 2. Test_data

# In[17]:


test_df = pd.read_csv('../input/tabular-playground-series-mar-2022/test.csv')
test_df


# Test data is from 12:00 to 23:40 on September 30, 1991, if you prepare validation data to check the score of the model, it needs to be at the same time on the same weekday. So, it is good to create validation data from 12:00 to 23:40 on September 23, 1991.

# In[18]:


# tst_start = pd.to_datetime('1991-09-23 12:00')
# tst_finish = pd.to_datetime('1991-09-23 23:40')

# X_train = train_df[train_df['time'] < tst_start]
# y_train = X_train['congestion']
# X_train = X_train.drop(['congestion', 'time'], axis=1)

# X_valid = train_df[(train_df['time'] >= tst_start) & (train_df['time'] <= tst_finish)]
# y_valid = X_valid['congestion']
# X_valid = X_valid.drop(['time', 'congestion'], axis=1)


# In[ ]:




