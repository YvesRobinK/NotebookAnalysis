#!/usr/bin/env python
# coding: utf-8

# Hello.
# 
# I find AutoML tools the best for baseline models, so here I'm trying another one called EvalML. You may find my other AutoML notebooks [here ](https://www.kaggle.com/kritidoneria/code?userId=1260510&sortBy=dateRun&tab=profile&language=Python&privacy=public)
# 
# A huge shoutout to [this](https://www.kaggle.com/gauravduttakiit/automate-the-ml-pipelines-with-evalml) Notebook for introducing me to this library.
# [I've also used EvalML to compete in TPS May](https://www.kaggle.com/kritidoneria/automl-tps-may21-using-evalml)
# Another reference for this work is [here](https://www.kaggle.com/tsnarendran14/jane-street-simple-xgb-model/data)

# <h1> Introduction to library </h1>

# Source: https://github.com/alteryx/evalml
# 
# EvalML is an AutoML library which builds, optimizes, and evaluates machine learning pipelines using domain-specific objective functions.
# 
# **Key Functionality**
# 
# 1. Automation - Makes machine learning easier. Avoid training and tuning models by hand. Includes data quality checks, cross-validation and more.
# 2. Data Checks - Catches and warns of problems with your data and problem setup before modeling.
# 3. End-to-end - Constructs and optimizes pipelines that include state-of-the-art preprocessing, feature engineering, feature selection, and a variety of modeling techniques.
# 4. Model Understanding - Provides tools to understand and introspect on models, to learn how they'll behave in your problem domain.
# 5. Domain-specific - Includes repository of domain-specific objective functions and an interface to define your own.

# <h1> Installation from Pypi </h1>

# In[1]:


get_ipython().system('pip install evalml')


# <h1> Load the Dataset </h1>

# In[2]:


import evalml
from evalml import AutoMLSearch
import pandas as pd


# In[3]:


X = pd.read_csv('/kaggle/input/jane-street-market-prediction/train.csv',nrows=1000)
#limiting rows here because of computational bottlenecks
y = pd.read_csv('/kaggle/input/jane-street-market-prediction/example_test.csv')


# <h2> Preprocessing</h2>

# In[4]:


# Only selecting the columns where missing values is less than7 percent
final_cols = X.isnull().mean()[X.isnull().mean() < 0.07]


# In[5]:


# Selecting only the required columns
X = X[final_cols.index]


# In[6]:


# Filling NA values with median
X = X.fillna(X.median())
import numpy as np


# In[7]:


X['action'] = np.where((X.resp_1 > 0) & (X.resp_2 > 0) & (X.resp_3 > 0) & (X.resp_4 > 0) & (X.resp > 0),1,0)


# In[8]:


X_train, X_test, y_train, y_test = evalml.preprocessing.split_data(X.drop(columns = ['date', 'weight', 'resp_1', 'resp_2', 'resp_3', 'resp_4','resp', 'ts_id','action']),X['action'], problem_type='binary')
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# # Run the search for the best classification model.

# In[9]:


#limiting search for efficiency
automl = AutoMLSearch(X_train=X_train, y_train=y_train,   problem_type='binary',allowed_model_families=['xgboost', 'lightgbm','catboost'],max_batches=5)
automl.search() 


# <h1> Model rankings and best pipeline </h1>

# In[10]:


automl.rankings


# In[11]:


automl.describe_pipeline(automl.rankings.iloc[0]["id"])


# <h1> Making predictions </h1>

# In[12]:


winner = automl.best_pipeline
df_submission = winner.predict_proba(y.drop(columns=['ts_id'])).to_dataframe()
df_submission['ts_id'] = y['ts_id']


# In[13]:


df_submission.set_index('ts_id').to_csv('submission.csv')


# In[ ]:




