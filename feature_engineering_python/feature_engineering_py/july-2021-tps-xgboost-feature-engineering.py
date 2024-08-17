#!/usr/bin/env python
# coding: utf-8

# This is my code for the July 2021 Tabular Playground Series. I got a public score of ~0.23 and a private score of about ~0.18

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import datetime
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


data = pd.read_csv('/kaggle/input/tabular-playground-series-jul-2021/train.csv')


# In[3]:


data['temp_humidity_combined'] = data['deg_C'] / data['relative_humidity']

data['date_parsed'] = pd.to_datetime(data['date_time'], infer_datetime_format=True)
data['time'] = data['date_time'].astype('datetime64[ns]').astype(np.int64)//10**9
data['hour'] = data['date_parsed'].dt.hour
data['month_of_year'] = data['date_parsed'].dt.month
data['year'] = data['date_parsed'].dt.year
data['quarter'] = data['date_parsed'].dt.quarter
data['total_months_since_2010_start'] = ((data['year'] - 2010) * 12) + data['month_of_year'] #Gets the total amounts of months since start of 2010, to avoid months being 0 in a different year.
data["dt-6"] = data["deg_C"] - data["deg_C"].shift(periods=6, fill_value=0)
data["dt-3"] = data["deg_C"] - data["deg_C"].shift(periods=3, fill_value=0)

total_features = ['temp_humidity_combined', 'absolute_humidity', 'relative_humidity', 'deg_C', 'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'hour', 'month_of_year', 'quarter', 'total_months_since_2010_start', 'dt-6', 'dt-3', 'time']

labels = ['target_carbon_monoxide', 'target_benzene', 'target_nitrogen_oxides'] #Not used


# In[4]:


data.head()


# In[5]:


from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y):
    mi_scores = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

print(make_mi_scores(data[total_features], data['target_carbon_monoxide']), "\n\n")  # show a few features with their MI scores
print(make_mi_scores(data[total_features], data['target_benzene']), "\n\n")  # show a few features with their MI scores
print(make_mi_scores(data[total_features], data['target_nitrogen_oxides']), "\n\n")  # show a few features with their MI scores


# In[6]:


best_features = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'hour', 'time'] #Features used to train the model -- high mutual information scores


# In[7]:


X = data[best_features]
y = data[labels]


# In[8]:


X.tail() #Last row has total months as 13 -- because it's January in 2011. 13 months since start of 2010. 


# In[9]:


y.head()


# In[10]:


X.describe()


# Split into train and validation

# In[11]:


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8)


# In[12]:


len(X_train)


# In[13]:


len(y_train)


# In[14]:


len(X_valid)


# In[15]:


len(y_valid)


# In[16]:


len(X)


# XGBoost Implementation

# In[17]:


from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

carbon_monoxide_model = XGBRegressor(n_estimators=500, learning_rate=0.01)
carbon_monoxide_model.fit(X_train, y_train['target_carbon_monoxide'],early_stopping_rounds=10, eval_set=[(X_valid, y_valid['target_carbon_monoxide'])])
print("done")

target_benzene_model = XGBRegressor(n_estimators=500, learning_rate=0.01)
target_benzene_model.fit(X_train, y_train['target_benzene'], early_stopping_rounds=10, eval_set=[(X_valid, y_valid['target_benzene'])])
print("done")

target_nitrogen_model = XGBRegressor(n_estimators=1000, learning_rate=0.01)
target_nitrogen_model.fit(X_train, y_train['target_nitrogen_oxides'], early_stopping_rounds=10, eval_set=[(X_valid, y_valid['target_nitrogen_oxides'])])
print("done")


# In[18]:


from sklearn.metrics import mean_absolute_error

valid_predictions = carbon_monoxide_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(valid_predictions, y_valid['target_carbon_monoxide'])))

valid_predictions = target_benzene_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(valid_predictions, y_valid['target_benzene'])))

valid_predictions = target_nitrogen_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(valid_predictions, y_valid['target_nitrogen_oxides'])))


# In[19]:


test = pd.read_csv('/kaggle/input/tabular-playground-series-jul-2021/test.csv')

test['date_parsed'] = pd.to_datetime(test['date_time'], infer_datetime_format=True)
test['time'] = test['date_time'].astype('datetime64[ns]').astype(np.int64)//10**9
test["dt-6"] = test["deg_C"] - test["deg_C"].shift(periods=6, fill_value=0)
test['hour'] = test['date_parsed'].dt.hour
test['month_of_year'] = test['date_parsed'].dt.month
test['year'] = test['date_parsed'].dt.year
test['total_months_since_2010_start'] = ((test['year'] - 2010) * 12) + test['month_of_year'] #Gets the total amounts of months since start of 2010, to avoid months being 0 in a different year.

nitrogen_test_X = test[best_features]
benzene_test_X = test[best_features]
carbon_monoxide_test_X = test[best_features]

target_carbon_monoxide = carbon_monoxide_model.predict(carbon_monoxide_test_X)
target_benzene = target_benzene_model.predict(benzene_test_X)
target_nitrogen_oxides = target_nitrogen_model.predict(nitrogen_test_X)


# In[20]:


test.head()


# In[21]:


output = pd.DataFrame({'date_time': test.date_time, 
                        'target_carbon_monoxide': target_carbon_monoxide,
                        'target_benzene' : target_benzene,
                        'target_nitrogen_oxides' : target_nitrogen_oxides})


# In[22]:


output.head()


# In[23]:


output.to_csv('submission.csv', index=False)

