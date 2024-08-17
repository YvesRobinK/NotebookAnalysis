#!/usr/bin/env python
# coding: utf-8

# # Bike Sharing
# 
# On this notebook, we will try to predict number of total rental using machine learning algorithms. Before this one, we will do feature engineering and exploratory data analysis for examine the data.
# 
# **Let's explore the data.**
# 
# 
# * datetime - hourly date + timestamp  
# * season -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
# * holiday - whether the day is considered a holiday
# * workingday - whether the day is neither a weekend nor holiday
# * weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy 
# * 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 
# * 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 
# * 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
# * temp - temperature in Celsius
# * atemp - "feels like" temperature in Celsius
# * humidity - relative humidity
# * windspeed - wind speed
# * casual - number of non-registered user rentals initiated
# * registered - number of registered user rentals initiated
# * count - number of total rentals
# 
# 

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[3]:


train.head() # we can see  first 5 samples with this function


# In[4]:


test.head()


# In[5]:


print(train.isnull().sum().sort_values(ascending = False)) #as you can see there is no null value in the columns
print("**"*50)
print(test.isnull().sum().sort_values(ascending = False))


# In[6]:


print(train.info()) # we can see type of features with this info()
print('**'*50)
print(test.info())


# As we can see, datetime's type is object. We should convert it to datetime.

# In[7]:


train.datetime = pd.to_datetime(train.datetime)
test.datetime = pd.to_datetime(test.datetime)


# let's see again !

# In[8]:


print(train.info())
print('**'*50)
print(test.info())


# Yes! we converted it. Now, we will separate the datetime column as year,month,day,hour and week

# In[9]:


train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour
train['dayofweek'] = train['datetime'].dt.weekday_name


test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['day'] = test['datetime'].dt.day
test['hour'] = test['datetime'].dt.hour
test['dayofweek'] = test['datetime'].dt.weekday_name


# In[10]:


train.tail() # we can see last 5 samples with this function! 


# We did it too. Now we can start to data exploratory. Let's start !

# In[11]:


train.describe().T # we can see statistical results with this function 


# # EDA & Feature Engineering

# In[12]:


plt.figure(figsize=(16,8))
sns.heatmap(train.corr(), annot=True)
plt.show()


# In[13]:


plt.figure(figsize=(16,8))
sns.pairplot(train)
plt.show()


# In[14]:


plt.figure(figsize=(16,8))
sns.distplot(train['count'])
plt.show()


# In[15]:


def scatter_plot():
    for i in test.columns:
        plt.scatter(train[i],train['count'])
        plt.title(f"Scatter plot for {i}")
        plt.show()


# In[16]:


scatter_plot()


# In[17]:


plt.figure(figsize=(16,8))
plt.plot(train.set_index('datetime')["count"][0:300])
plt.show()


# In[18]:


plt.figure(figsize=(16,8))
sns.boxplot(x='dayofweek',y='count', data=train)


# In[19]:


# we need to convert categorical data to numeric data.

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['dayofweek'] = le.fit_transform(train['dayofweek'])
test['dayofweek'] = le.transform(test['dayofweek'])


# As we can see from above graph. This is positively(right) skewed data.Now we will look the box plot and outliers value.

# Box plot use the IQR method for finding display data and outliers.
# 
# 
# 
# #### Wikipedia Definition
# 
# The interquartile range (IQR), also called the midspread or middle 50%, or technically H-spread, is a measure of statistical dispersion, being equal to the difference between 75th and 25th percentiles, or between upper and lower quartiles, IQR = Q3 − Q1.
# In other words, the IQR is the first quartile subtracted from the third quartile; these quartiles can be clearly seen on a box plot on the data.
# It is a measure of the dispersion similar to standard deviation or variance, but is much more robust against outliers.
# 
# 
# * We will clear the outliers values.
# > Okay, let's check!

# In[20]:


plt.figure(figsize=(16,8))
sns.boxplot(x='season', y='count', data=train)


# we can say of seein this graph, people more rent bike on summer and fall.

# In[21]:


plt.figure(figsize=(16,8))
sns.boxplot(x='hour',y='count', data=train) # as we can see there is difference for each hour. We need to use it !


# In[22]:


plt.figure(figsize=(16,8))
sns.boxplot(x='year',y='count', data=train) # bike were rented in 2012!


# We can say that people prefer the morning and evening times for renting bike

# In[23]:


plt.figure(figsize=(16,8))
plt.hist(train['count'][train['year'] == 2011], alpha=0.5, label='2011')
plt.hist(train['count'][train['year'] == 2012], alpha=0.5, label='2012', color='red')


# Rented more bike in 2012 than 2011.
# 
# 
# ####  Now, let's find the outliers

# In[24]:


train.head()


# In[25]:


train.set_index('datetime', inplace=True)


# In[26]:


train['2011-01-19 23:00:00':]


# In[27]:


Q1 = train.quantile(0.25)
Q3 = train.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[28]:


train_without_outliers =train[~((train < (Q1 - 1.5 * IQR)) |(train > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[29]:


train_without_outliers.dropna(inplace=True)


# In[30]:


print(train.info())
print('*********************************************************************************')
print(train_without_outliers.info())


# > We removed outliers data points.

# In[31]:


train_without_outliers.head(2)


# We are going to fill the row that wind speed is equal zero.

# In[32]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='season',y='windspeed',data=train_without_outliers,palette='winter')


# In[33]:


train_without_outliers['windspeed'] = train_without_outliers['windspeed'].replace(0,np.NaN)
test['windspeed'] = test['windspeed'].replace(0,np.NaN) 


# Now, We repalced zero as NaN. We will fill NaN with interpolate. Interpolate is using fill NaN value for time series data.

# In[34]:


train_without_outliers['windspeed'].fillna(method='bfill',inplace=True)
train_without_outliers['windspeed'] = train_without_outliers['windspeed'].interpolate()
test['windspeed'] = test['windspeed'].interpolate()


# In[35]:


train_without_outliers['windspeed'].isnull().sum()


# In[36]:


test.head()


# In[37]:


train_without_outliers.head(5)


# Now e are going to convert cateforical data to categorical columns.

# In[38]:


train_without_outliers[['season','holiday','workingday','weather', 'year','month','day','hour','dayofweek']] = train_without_outliers[['season','holiday','workingday','weather', 'year','month','day','hour','dayofweek']].astype('category')
test[['season','holiday','workingday','weather', 'year','month','day','hour','dayofweek']] = test[['season','holiday','workingday','weather', 'year','month','day','hour','dayofweek']].astype('category')


# In[39]:


train_without_outliers.info()


# Now, we can start to make predictions

# # Random Forest Regression

# A Random Forest is an ensemble technique capable of performing both regression and classification tasks with the use of multiple decision trees and a technique called Bootstrap Aggregation, commonly known as bagging. What is bagging you may ask? Bagging, in the Random Forest method, involves training each decision tree on a different data sample where sampling is done with replacement.
# 
# [](http://www.google.com/url?sa=i&url=https%3A%2F%2Fmedium.com%2Fdatadriveninvestor%2Frandom-forest-regression-9871bc9a25eb&psig=AOvVaw01y2FFgzla0z_xdAC60_j8&ust=1608717087340000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCLjXl6io4e0CFQAAAAAdAAAAABAH)

# ### Train test split

# In[40]:


from sklearn.model_selection import train_test_split


# In[41]:


X = train_without_outliers[['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp','humidity', 'year', 'month', 'day', 'hour', 'dayofweek','windspeed']]
y = train_without_outliers['count']


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1)


# In[43]:


y_train


# In[44]:


from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


# In[45]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)


# In[46]:


rf_prediction = rf.predict(X_test)


# In[47]:


from sklearn.metrics import mean_squared_error
from sklearn import metrics
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, rf_prediction)))


# In[48]:


plt.scatter(y_test,rf_prediction)


# In[49]:


plt.figure(figsize=(16,8))
plt.plot(rf_prediction[0:200],'r')
plt.plot(y_test[0:200].values)


# # Decision Tree Regression

# The decision tree is a simple machine learning model for getting started with regression tasks.
# 
# Background A decision tree is a flow-chart-like structure, where each internal (non-leaf) node denotes a test on an attribute, each branch represents the outcome of a test, and each leaf (or terminal) node holds a class label. The topmost node in a tree is the root node. (see here for more details).

# In[50]:


from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor()
dt_reg.fit(X_train, y_train)


# In[51]:


dt_prediction = dt_reg.predict(X_test)


# In[52]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dt_prediction)))


# In[53]:


plt.scatter(y_test,dt_prediction)


# Now, We will use the test data.

# In[54]:


test.head()


# In[55]:


test[['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp','humidity', 'year', 'month', 'day', 'hour', 'dayofweek','windspeed']] = sc_X.fit_transform(test[['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp','humidity', 'year', 'month', 'day', 'hour', 'dayofweek','windspeed']])


# In[56]:


test_pred= rf.predict(test[['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp','humidity', 'year', 'month', 'day', 'hour', 'dayofweek','windspeed']])


# In[57]:


test_pred


# In[58]:


test_pred=test_pred.reshape(-1,1)


# In[59]:


test_pred = pd.DataFrame(test_pred, columns=['count'])


# In[60]:


df = pd.concat([test['datetime'], test_pred],axis=1)


# In[61]:


df.head()


# In[62]:


df['count'] = df['count'].astype('int')


# In[63]:


df.to_csv('submission1.csv' , index=False)

