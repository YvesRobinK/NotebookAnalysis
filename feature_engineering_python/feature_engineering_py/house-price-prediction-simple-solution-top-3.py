#!/usr/bin/env python
# coding: utf-8

# ## Introduction

# ![Housing Prices Competition](https://i.imgur.com/JRIWMD2.png)

# This is my submission to the [Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/c/home-data-for-ml-course/overview). I have used **XGBoost** for prediction. As I'm writing this , I am ranked among the **top 2%** of all Kagglers. 
# 
# I hope you enjoy while reading it! And if you liked this kernel feel free to **upvote** and leave **feedback**, thanks!

# Let's start...

# In[1]:


# importing libraries
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# In[2]:


# Read the data
train = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')
test = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')


# ## Exploratory Data Analysis

# #### Train data

# In[3]:


# print first five rows
train.head()


# In[4]:


# column names
train.columns


# In[5]:


# columns with null values
train_col_null = train.columns[train.isnull().any()==True].tolist()
# null values in these columns
train[train_col_null].isnull().sum()


# #### Test data

# In[6]:


# print first five rows
test.head()


# In[7]:


# column names
test.columns


# In[8]:


# columns with null values
test_col_null = test.columns[test.isnull().any()==True].tolist()
# null values in these columns
test[test_col_null].isnull().sum()


# ## Feature Engineering

# In[9]:


# Remove rows with missing target
X = train.dropna(axis=0, subset=['SalePrice'])

# separate target from predictors
y = X.SalePrice              
X.drop(['SalePrice'], axis=1, inplace=True)


# In[10]:


# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y,
                                                                train_size=0.8,
                                                                test_size=0.2,
                                                                random_state=0)


# In[11]:


# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in X_train_full.columns 
                        if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]


# In[12]:


# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns
                if X_train_full[cname].dtype in ['int64', 'float64']]


# In[13]:


# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols

X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# for test data also
X_test = test[my_cols].copy()


# In[14]:


# One-hot encode the data
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)


# ## Model Fitting and Prediction

# In[15]:


# Define the model
xgb =  XGBRegressor(n_estimators=1000,
                    learning_rate=0.05)


# In[16]:


# Fit the model
xgb.fit(X_train, y_train)


# In[17]:


# Get predictions
y_pred = xgb.predict(X_valid)


# In[18]:


# Calculate MAE
mae = mean_absolute_error(y_pred, y_valid)
print("Mean Absolute Error:" , mae)


# In[19]:


# prediction
prediction = xgb.predict(X_test)


# In[20]:


# Submission file

output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': prediction})
output.to_csv('submission.csv', index=False)
output.head()


# ### Thank You
