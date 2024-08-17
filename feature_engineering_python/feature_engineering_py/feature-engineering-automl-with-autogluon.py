#!/usr/bin/env python
# coding: utf-8

# # Time Series Forecasting with AutoGluon
# 
# This notebook is dedicated to solving the July kaggle playground challenge (https://www.kaggle.com/c/tabular-playground-series-jul-2021).
# 
# We will first start by analyzing the data, and perform some basic feature engineering. Then we will perform AutoML with AutoGluon

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas_profiling import ProfileReport

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Exploratory Data Analysis
# 
# The first step in our analysis is to take a look at the data. Since we are dealing with time series, seasonalities will be fundamental to improve the performance of our machine learning models. For the sake of concisiveness, we also pass the datetime as the index and we parse it as a date.

# In[2]:


df = pd.read_csv("/kaggle/input/tabular-playground-series-jul-2021/train.csv", index_col="date_time", parse_dates=True)
df.tail()


# All the rows in the dataframe seems to be space by time intervals of one hour. Lets continue our analysis using a pandas profiler, to check for missing values, outliers and correlations.

# In[3]:


profile = ProfileReport(df, title="Pandas Profiling Report")
profile


# Lets take a closer look on the targets time series:

# In[4]:


for column in df.columns:
    if "target" in column:
        fig = plt.figure()
        _ = df[column].plot(figsize=(12, 8), title=column)


# Each single one of the target has strong fluctuations. If we think about the problem statement, we are dealing with pollution level, that will most likely be influenced by the movement of peoples. 
# 
# Thus, we should check the following assumptions:
# 
# - In a single day we will have stronger pollution levels during rush hours
# - In a week working days will be characterized by higer pollution levels

# In[5]:


from sklearn.preprocessing import MinMaxScaler #JUST TO NORMALIZE THE TARGET IN A SIGNLE GRAPH


# ## Feature Engineering
# 
# We shall now try to perform some very basic feature engineering to improve the performance of our ML models

# In[6]:


df['dayoftheweek'] = df.index.dayofweek #Weekly Seasonality
df['hourofday'] = df.index.hour #Daily Seasonality
df['time'] = df.index.astype(np.int64) #Yearly Trend


# In[7]:


def view_seasonality(view):
    
    fig = plt.figure(figsize=(12, 8))
    legends=[]
    for column in df.columns:
        if "target" in column:
            plt.plot(1+view.index, MinMaxScaler().fit_transform(view[column].values.reshape(-1, 1)))
            legends.append(column)
    plt.xlabel(view.index.name)
    plt.legend(legends)


# In[8]:


view = df.groupby(by='hourofday').mean()
view_seasonality(view)


# As expected, rush hours have higer levels of pollution, also during the night the targets are much lower (few peoples drive at 4 am)

# In[9]:


view = df.groupby(by='dayoftheweek').mean()
view_seasonality(view)


# Again, in the weekends, the pollution level decrease as expected!
# 
# One very promising feature to add is then something realted to holidays (like a binary flag). Sadly, no information is available on the country of origin for this data, so for the time being I avoided including it.
# 
# Lets now take a look at sensor data

# In[10]:


# Resample the entire dataset by daily average
rollmean = df.resample(rule='D').mean()
rollstd = df.resample(rule='D').std()
# Plot time series for each sensor with its mean and standard deviation
names = ['sensor_4']
for name in names:
    _ = plt.figure(figsize=(18,3))
    _ = plt.plot(df[name], color='blue', label='Original')
    _ = plt.plot(rollmean[name], color='red', label='Rolling Mean')
    _ = plt.plot(rollstd[name], color='black', label='Rolling Std' )
    _ = plt.legend(loc='best')
    _ = plt.title(name)
    plt.show()


# It is clear that is not impossible for the sensor to be broken for a long period of time. Another possible feature to insert is then a binary flag about the current status of the various sensors. This chould be achieved with SPRT or other anomalies detection strategies. If someone has a suggestion please put them in the comment ;)

# ## Model Selection
# 
# We can now begin the AutoML process.
# Note that in this kaggle notebook I severely reduced the time limit on the autoML process, so please increase it if you intend to follow it

# In[11]:


get_ipython().system('pip install "mxnet<2.0.0"')
get_ipython().system('pip install autogluon')


# In[12]:


from autogluon.tabular import TabularDataset, TabularPredictor


# In[13]:


TIME_LIMIT = 60 #Simulation time (IN SECONDS) SET IT TO AT LEAST 3600 IN A REAL SCENARIO
#all target columns have target in the column name
target = [column for column in df.columns if "target" in column] 
features = df.columns.drop(target)


# In[14]:


def multi_target_prediction(i):
    
    '''
    Simple function to iterate the AutoML process for the 3 targets
    '''
    
    train_data = TabularDataset(df[features.tolist()+[target[i]]])
    label = target[i]
    
    save_path = f'agModels-predictClass{i}'
    
    predictor = TabularPredictor(
        label = label,
        path = save_path).fit(train_data, 
                              presets='best_quality', 
                              num_stack_levels = 3,
                              num_bag_folds = 5,
                              num_bag_sets = 1,
                              time_limit=TIME_LIMIT)

    return predictor


# Perform the AutoML for the 3 different targets

# In[15]:


predictors = [multi_target_prediction(i) for i in range(3)]


# We can now cast our predictions on the test dataset using the optimized models!

# In[16]:


test = pd.read_csv(dirname + '/test.csv', index_col='date_time', parse_dates=True)
test['dayoftheweek'] = test.index.dayofweek
test['hourofday'] = test.index.hour
test['time'] = test.index.astype(np.int64)
t = TabularDataset(test)


# In[17]:


predictions = [predictor.predict(t) for predictor in predictors]


# In[18]:


submission = pd.DataFrame(test.index.values,  columns=['date_time'])
submission[target] = np.vstack(predictions).T


# In[19]:


submission.to_csv('./sumbission.csv', index=False)


# In[ ]:




