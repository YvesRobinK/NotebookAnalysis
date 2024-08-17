#!/usr/bin/env python
# coding: utf-8

# # Intro
# Welcome to the [Store Sales - Time Series Forecasting](https://www.kaggle.com/c/store-sales-time-series-forecasting/data) competition.
# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/29781/logos/header.png)
# 
# In this competition, we have to predict sales for the thousands of product families sold at Favorita stores located in [Ecuador](https://en.wikipedia.org/wiki/Ecuador).
# 
# <span style="color: royalblue;">Please vote the notebook up if it helps you. Feel free to leave a comment above the notebook. Thank you. </span>

# # Libraries

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


# # Path

# In[2]:


path = '/kaggle/input/store-sales-time-series-forecasting/'
os.listdir(path)


# # Load Data

# In[3]:


data_oil = pd.read_csv(path+'oil.csv')
train_data = pd.read_csv(path+'train.csv', index_col=0)
test_data = pd.read_csv(path+'test.csv', index_col=0)
samp_subm = pd.read_csv(path+'sample_submission.csv')
data_holi = pd.read_csv(path+'holidays_events.csv')
data_store =  pd.read_csv(path+'stores.csv')
data_trans = pd.read_csv(path+'transactions.csv')


# # Overview

# In[4]:


print('Number of train samples: ', len(train_data.index))
print('Number of test samples: ', len(test_data.index))
print('Number of features: ', len(train_data.columns))


# In[5]:


train_data.head()


# In[6]:


test_data.head()


# # Exploratory Data Analysis

# ## Feature family
# The feature family has 33 categorical values which we have to encode later. The values are evenly distributed.

# In[7]:


train_data['family'].value_counts()[0:3]


# ## Oil Data
# Daily oil price. Includes values during both the train and test data timeframes. (Ecuador is an oil-dependent country and it's economical health is highly vulnerable to shocks in oil prices.)

# In[8]:


data_oil.head()


# ## Store Data
# * Store metadata, including city, state, type, and cluster.
# * cluster is a grouping of similar stores.

# In[9]:


data_store.head()


# In[10]:


data_store['city'].value_counts()[0:3]


# ## Transaction Data

# In[11]:


data_trans.head()


# ## Holiday Data
# * Holidays and Events, with metadata
# * NOTE: Pay special attention to the transferred column. A holiday that is transferred officially falls on that calendar day, but was moved to another date by the government. A transferred day is more like a normal day than a holiday. To find the day that it was actually celebrated, look for the corresponding row where type is Transfer. For example, the holiday Independencia de Guayaquil was transferred from 2012-10-09 to 2012-10-12, which means it was celebrated on 2012-10-12. Days that are type Bridge are extra days that are added to a holiday (e.g., to extend the break across a long weekend). These are frequently made up by the type Work Day which is a day not normally scheduled for work (e.g., Saturday) that is meant to payback the Bridge.
# * Additional holidays are days added a regular calendar holiday, for example, as typically happens around Christmas (making Christmas Eve a holiday).

# In[12]:


data_holi.head()


# # Feature Engineering

# In[13]:


features = ['store_nbr', 'family', 'onpromotion']
target = 'sales'


# ## Create Feature Weekday
# Based on the feature date we can create the features weekday, month or year.

# In[14]:


def extract_weekday(s):
    return s.dayofweek

def extract_month(s):
    return s.month

def extract_year(s):
    return s.year


# In[15]:


train_data['date'] = pd.to_datetime(train_data['date'])
train_data['weekday'] = train_data['date'].apply(extract_weekday)
train_data['year'] = train_data['date'].apply(extract_year)
train_data['month'] = train_data['date'].apply(extract_month)

test_data['date'] = pd.to_datetime(test_data['date'])
test_data['weekday'] = test_data['date'].apply(extract_weekday)
test_data['year'] = test_data['date'].apply(extract_year)
test_data['month'] = test_data['date'].apply(extract_month)


# In[16]:


features.append('weekday')
features.append('year')
features.append('month')


# ## Encode Categorical Labels

# In[17]:


enc = preprocessing.LabelEncoder()
enc.fit(train_data['family'])


# In[18]:


train_data['family'] = enc.transform(train_data['family'])
test_data['family'] = enc.transform(test_data['family'])


# # Define Train, Val And Test Data

# In[19]:


X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]


# In[20]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.33, random_state=2021)


# # Simple Model
# First we start with a simple model based on the feature in the train and test data.

# XGB Regression:

# In[21]:


model = XGBRegressor(objective='reg:squaredlogerror', n_estimators=200)
model.fit(X_train, y_train)
y_val_pred = model.predict(X_val)
y_val_pred = np.where(y_val_pred<0, 0, y_val_pred)
print('Root Mean Squared Logaritmic Error:', np.sqrt(mean_squared_log_error(y_val, y_val_pred)))


# Linear Regression:

# In[22]:


reg = LinearRegression(normalize=True).fit(X_train, y_train)
y_val_pred = reg.predict(X_val)
y_val_pred = np.where(y_val_pred<0, 0, y_val_pred)
print('Root Mean Squared Logaritmic Error:', np.sqrt(mean_squared_log_error(y_val, y_val_pred)))


# # Predict Test Data

# In[23]:


y_test_XGB = model.predict(X_test)
y_test_REG = model.predict(X_test)
samp_subm[target] = (0.8*y_test_XGB+0.2*y_test_REG)


# In[24]:


samp_subm[target] = np.where(samp_subm[target]<0, 0, samp_subm[target])


# # Export

# In[25]:


samp_subm.to_csv('submission.csv', index=False)

