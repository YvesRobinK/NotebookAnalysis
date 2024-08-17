#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# # Load Datasets

# In[2]:


holiday=pd.read_csv("../input/store-sales-time-series-forecasting/holidays_events.csv")
oil=pd.read_csv("../input/store-sales-time-series-forecasting/oil.csv")
sample_submission=pd.read_csv("../input/store-sales-time-series-forecasting/sample_submission.csv")
stores=pd.read_csv("../input/store-sales-time-series-forecasting/stores.csv")
test=pd.read_csv("../input/store-sales-time-series-forecasting/test.csv")
train=pd.read_csv("../input/store-sales-time-series-forecasting/train.csv")
transactions=pd.read_csv("../input/store-sales-time-series-forecasting/transactions.csv")


# In[3]:


print ("Training Data Shape: ", train.shape)
print ("Testing Data Shape: ", test.shape)
train.head()


# # Merging Datasets

# Several supplement files are provided which contain addition features, which can be cobined to training and test datasets(original).

# In[4]:


train1 = train.merge(oil, on = 'date', how='left')
train1 = train1.merge(holiday, on = 'date', how='left')
train1 = train1.merge(stores, on = 'store_nbr', how='left')
train1 = train1.merge(transactions, on = ['date', 'store_nbr'], how='left')
train1 = train1.rename(columns = {"type_x" : "holiday_type", "type_y" : "store_type"})

test1 = test.merge(oil, on = 'date', how='left')
test1 = test1.merge(holiday, on = 'date', how='left')
test1 = test1.merge(stores, on = 'store_nbr', how='left')
test1 = test1.merge(transactions, on = ['date', 'store_nbr'], how='left')
test1 = test1.rename(columns = {"type_x" : "holiday_type", "type_y" : "store_type"})
train1.head()


# In[5]:


test1.head()


# # Value Count for each feature

# In[6]:


train1["family"].value_counts()


# In[7]:


train1["city"].value_counts()


# In[8]:


train1["state"].value_counts()


# In[9]:


train1["onpromotion"].value_counts()


# In[10]:


train1["store_type"].value_counts()


# # Correlation b/w features

# In[11]:


import seaborn as sns
corr = train1.corr()
sns.heatmap(corr)


# # Visualizing closely correlated features

# In[12]:


sns.set(rc={'figure.figsize':(20,8.27)})
sns.barplot(x = 'store_nbr',y = 'sales',data = train1,palette = "Blues")


# In[13]:


sns.set(rc={'figure.figsize':(20,8.27)})
sns.barplot(x = 'store_nbr',y = 'transactions',data = train1,palette = "Blues")


# In[14]:


sns.set(rc={'figure.figsize':(20,8.27)})
sns.lineplot(x = "transactions",y = 'sales',data = train1,palette = "Blues")


# In[15]:


sns.set(rc={'figure.figsize':(20,8.27)})
sns.lineplot(x = "onpromotion",y = 'sales',data = train1,palette = "Blues")


# In[16]:


sns.set(rc={'figure.figsize':(20,8.27)})
sns.barplot(x = 'cluster',y = 'transactions',data = train1,palette = "Blues")


# # Spilliting Dataset

# In[17]:


from sklearn.model_selection import train_test_split
features=['date','store_nbr','family','onpromotion','dcoilwtico','holiday_type','locale','locale_name','description','transferred','city','state','store_type','cluster','transactions']
X=train1[features]
y=train1.sales
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)


# # Feature importance plot

# In[18]:


# linear regression feature importance
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
# define the model
model = LinearRegression()
# fit the model
model.fit(X, y)
# get importance
importance = model.coef_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()


# # Feature Engineering

# In[19]:


def feature_eng(data):
    data['date'] = pd.to_datetime(data['date'])
    data['dayofweek'] = data['date'].dt.dayofweek
    data['quarter'] = data['date'].dt.quarter
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year
    data['dayofyear'] = data['date'].dt.dayofyear
    data['dayofmonth'] = data['date'].dt.day
    return data
    
train1 = feature_eng(train1)
test1 = feature_eng(test1)
train1.head()


# In[20]:


train1.to_csv("train_m_fe.csv", index = False)
test1.to_csv("test_m_fe.csv", index = False)

