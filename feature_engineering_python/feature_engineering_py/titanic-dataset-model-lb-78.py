#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install -U lightautoml


# ## Introduction
# In this kernel I will be creating machine learning model usnig LightAutoML (LAMA) library and will be doing Auotmatic EDA using Pandas Profilling on the vary famous **Titanic Dataset**.
# Hope you all enjoy it and please let me know about the mistakes I made in this kernel. I would appreciate your remarks and glad to improve. ; )
# 
# If you like the notebook and find it helpful then **PLEASE UPVOTE**. It motivates me a lot. 

# ## Load the DataSet

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas_profiling 
import datetime as dt

import os
import time
import re

# Installed libraries
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch

# Imports from our package
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.dataset.roles import DatetimeRole
from lightautoml.tasks import Task
from lightautoml.utils.profiler import Profiler


# In[ ]:


N_THREADS = 4 # threads cnt for lgbm and linear models
N_FOLDS = 5 # folds cnt for AutoML
RANDOM_STATE = 42 # fixed random state for various reasons
TEST_SIZE = 0.2 # Test size for metric check
TIMEOUT = 1800 # Time in seconds for automl run


# In[ ]:


train_df=pd.read_csv("../input/train.csv")
test_df=pd.read_csv("../input/test.csv")


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# ## Identifying & Filling Missing Value 

# In[ ]:


def missingdata(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    ms=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    ms= ms[ms["Percent"] > 0]
    f,ax =plt.subplots(figsize=(8,6))
    plt.xticks(rotation='90')
    fig=sns.barplot(ms.index, ms["Percent"],color="green",alpha=0.8)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    return ms


# In[ ]:


missingdata(train_df)


# In[ ]:


missingdata(test_df)


# ### Filling missing Values

# In[ ]:


test_df['Age'].mean()


# In[ ]:


train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace = True)


# In[ ]:


test_df['Fare'].fillna(test_df['Fare'].median(), inplace = True)


# ### Cabin Featueres has more than 75% of missing data in both Test and train data so we are remove the Cabin 

# In[ ]:


drop_column = ['Cabin']
train_df.drop(drop_column, axis=1, inplace = True)
test_df.drop(drop_column,axis=1,inplace=True)


# ### Both the test and train Age features contains more the 15% of missing Data so we are fill with the median

# In[ ]:


test_df['Age'].fillna(test_df['Age'].median(), inplace = True)
train_df['Age'].fillna(train_df['Age'].median(), inplace = True)


# In[ ]:


print('Checking the nan value in train data')
print(train_df.isnull().sum())
print('___'*20)
print('Checking the nan value in test data')
print(test_df.isnull().sum())


# ## Feature engineering
# 
# Feature engineering is the art of converting raw data into useful features. 

# In[ ]:


## combine test and train as single to apply some function
all_data=[train_df,test_df]


# In[ ]:


# Create new feature FamilySize as a combination of SibSp and Parch
for dataset in all_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1


# In[ ]:


import re
# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
# Create a new feature Title, containing the titles of passenger names
for dataset in all_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
for dataset in all_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 
                                                 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


# In[ ]:


## create bin for age features
for dataset in all_data:
    dataset['Age_bin'] = pd.cut(dataset['Age'], bins=[0,12,20,40,120], labels=['Children','Teenage','Adult','Elder'])


# In[ ]:


## create bin for fare features
for dataset in all_data:
    dataset['Fare_bin'] = pd.cut(dataset['Fare'], bins=[0,7.91,14.45,31,120], labels=['Low_fare','median_fare',
                                                                                      'Average_fare','high_fare'])


# In[ ]:


### for our reference making a copy of both DataSet start working for copy of dataset
traindf=train_df
testdf=test_df


# In[ ]:


all_dat=[traindf,testdf]


# In[ ]:


for dataset in all_dat:
    drop_column = ['Age','Fare','Name','Ticket']
    dataset.drop(drop_column, axis=1, inplace = True)


# In[ ]:


drop_column = ['PassengerId']
traindf.drop(drop_column, axis=1, inplace = True)


# #### Now every thing almost ready only one step we converted the catergical features in numerical by using dummy variable

# In[ ]:


testdf.head(2)


# Creating dummies : )

# In[ ]:


traindf = pd.get_dummies(traindf, columns = ["Sex","Title","Age_bin","Embarked","Fare_bin"],
                             prefix=["Sex","Title","Age_type","Em_type","Fare_type"])


# In[ ]:


testdf = pd.get_dummies(testdf, columns = ["Sex","Title","Age_bin","Embarked","Fare_bin"],
                             prefix=["Sex","Title","Age_type","Em_type","Fare_type"])


# ### Correlation Between The Features

# In[ ]:


sns.heatmap(traindf.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(20,12)
plt.show()


# Interpreting The Heatmap : 
# The first thing to note is that only the numeric features are compared as it is obvious that we cannot correlate between alphabets or strings. Before understanding the plot, let us see what exactly correlation is.
# 
# POSITIVE CORRELATION: If an increase in feature A leads to increase in feature B, then they are positively correlated. A value 1 means perfect positive correlation.
# 
# NEGATIVE CORRELATION: If an increase in feature A leads to decrease in feature B, then they are negatively correlated. A value -1 means perfect negative correlation.
# 
# Now lets say that two features are highly or perfectly correlated, so the increase in one leads to increase in the other. This means that both the features are containing highly similar information and there is very little or no variance in information. This is known as MultiColinearity as both of them contains almost the same information.
# 
# So do you think we should use both of them as one of them is redundant. While making or training models, we should try to eliminate redundant features as it reduces training time and many such advantages.

# ### Pairplots
# Finally let us generate some pairplots to observe the distribution of data from one feature to the other. Once again we use Seaborn to help us.

# In[ ]:


g = sns.pairplot(data=train_df, hue='Survived', palette = 'seismic',
                 size=2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
g.set(xticklabels=[])


# # 📊 Automatic EDA using Pandas Profiling 📚
# Pandas profiling is an open source Python module with which we can quickly do an exploratory data analysis with just a few lines of code. In short, what pandas profiling does is save us all the work of visualizing and understanding the distribution of each variable.

# In[ ]:


# Start of Pandas Profiling process
start_time = dt.datetime.now()
print("Started at ", start_time)
report = pandas_profiling.ProfileReport(traindf)
report


# In[ ]:


print('Pandas Profling finished!!')
finish_time = dt.datetime.now()
print("Finished at ", finish_time)
elapsed = finish_time - start_time
print("Elapsed time: ", elapsed)


# In[ ]:


def acc_score(y_true, y_pred, **kwargs):
    return accuracy_score(y_true, (y_pred > 0.5).astype(int), **kwargs)

def f1_metric(y_true, y_pred, **kwargs):
    return f1_score(y_true, (y_pred > 0.5).astype(int), **kwargs)

task = Task('binary', metric = f1_metric)


# In[ ]:


roles = {
    'target': 'Survived',
    'drop': ['PassengerId', 'Name','Ticket'],
}


# # Model Training 
# Here we are using lightAutoML (LAMA) library. Learn and explore more about it : [here](https://lightautoml.readthedocs.io/en/latest/automl.html)

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nautoml = TabularUtilizedAutoML(task = task, \n                       timeout = TIMEOUT,\n                       cpu_limit = N_THREADS,\n                       general_params = {'use_algos': [['linear_l2', 'lgb', 'lgb_tuned',]]},\n                       reader_params = {'n_jobs': N_THREADS})\noof_pred = automl.fit_predict(traindf, roles = roles)\nprint('oof_pred:\\n{}\\nShape = {}'.format(oof_pred[:10], oof_pred.shape))\n")


# # Prediction

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntest_pred = automl.predict(testdf)\nprint('Prediction for test data:\\n{}\\nShape = {}'.format(test_pred[:10], test_pred.shape))\n\nprint('Check scores...')\nprint('OOF score: {}'.format(acc_score(traindf['Survived'].values, oof_pred.data[:, 0])))\n")


# # Submission

# In[ ]:


submission = pd.DataFrame()
submission["PassengerId"] = test_df["PassengerId"]
submission['Survived'] = (test_pred.data[:, 0] > 0.5).astype(int)
submission.to_csv('automl_utilized_3600_f1_metric.csv', index = False)


# In[ ]:


submission


# Hope you liked this kernel, if you, then please **UPVOTE** and **COMMENT** below your remarks, I would be glad to listen to you all ; )
# Every suggestion and correction is accepted happily 😁
# # Thank You 😊
