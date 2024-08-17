#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this notebook series, I'll be sharing my typical approach to machine learning (ML) problems (in this case competitions), in an end-to-end ML pipeline starting with the data analyses, to feature engineering, validation strategies, model training and finally inference (with post-processing techniques).
# 
# My notebooks in this series can be found in the links below:
# - [Exploratory Data Analysis (EDA)](https://www.kaggle.com/khoongweihao/part-1-exploratory-data-analysis-eda)
# - [Preprocessing & Feature Engineering](https://www.kaggle.com/khoongweihao/part-2-preprocessing-feature-engineering)
# - [Model Training & Validation Strategies](https://www.kaggle.com/khoongweihao/part-3-model-training-validation-strategies)
# - [Inference & Post-processing Techniques](https://www.kaggle.com/khoongweihao/part-4-inference-and-post-processing-techniques)
# 
# Bonus notebooks include adoption of recent research in terms of models, hyperparameter search, etc. They can be found in the links below:
# - Hyperparameter optimization with Optuna
# - TabNet 

# # Imports

# In[1]:


import numpy as np 
import pandas as pd 
import os
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import category_encoders as ce


# In[2]:


df_train = pd.read_csv('../input/titanic/train.csv')
df_test = pd.read_csv('../input/titanic/test.csv')
df_sub = pd.read_csv('../input/titanic/gender_submission.csv')


# # Preprocessing The Data

# In[3]:


df_train.head()


# In[4]:


df_test.head()


# Firstly, let's see which features could be used to train the ML model in predicting passenger survival. The passenger name (`Name`) probably can't be used right off the bat as each passenger have a unique name (most likely, can be verified) and having the number of categories equal to the number of entries in the dataset may not be useful for the model. A possible way to incorporate the passenger name as a feature will be to encode it into say a binary category, where `1` indicates that the passenger is a person of status (e.g. monarchy, president, etc) and `0` otherwise. 
# 
# The ticket number `Ticket` is similar in that sense, and may be dropped for initial ML modeling.

# In[5]:


df_train.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)


# In[6]:


df_test.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)


# In[7]:


df_train.head()


# In[8]:


df_test.head()


# Next, we encode the categorical features which are strings.For which we use Scikit-learn's `LabelEncoder`:

# In[9]:


df_train['Cabin'] = df_train['Cabin'].replace(np.NaN, 'UNKNOWN', regex=True)
df_test['Cabin'] = df_test['Cabin'].replace(np.NaN, 'UNKNOWN', regex=True)


# In[10]:


cabin_vals = np.unique(list(df_train['Cabin'].values) + list(df_test['Cabin'].values))
cabin_vals


# In[11]:


mapping_d = {}
for i, feature in enumerate(cabin_vals):
    mapping_d[feature] = i
    
encoder= ce.OrdinalEncoder(cols=['Cabin'], return_df=True,
                           mapping=[{'col':'Cabin',
                                     'mapping': mapping_d}])


# In[12]:


mapping_d


# In[13]:


df_train = encoder.fit_transform(df_train)


# In[14]:


df_test = encoder.fit_transform(df_test)


# In[15]:


df_train.head()


# In[16]:


df_test.head()


# For the other categorical variables, there are no null values from the EDA. So we can proceed to encode them with Scikit-learn's `LabelEncoder`:

# In[17]:


cols_to_encode = ['Sex', 'Embarked']
for col in cols_to_encode:
    le = LabelEncoder()
    le.fit(df_train[col])
    
    df_train[col] = le.transform(df_train[col])
    df_test[col] = le.transform(df_test[col])


# In[18]:


df_train.head()


# In[19]:


df_test.head()


# Note that there are missing values in `Age` and `Fare`. For such cases, we impute them with `-1`.

# In[20]:


df_train['Age'] = df_train['Age'].replace(np.NaN, -1)
df_test['Age'] = df_test['Age'].replace(np.NaN, -1)
df_train['Fare'] = df_train['Fare'].replace(np.NaN, -1)
df_test['Fare'] = df_test['Fare'].replace(np.NaN, -1)


# Some sanity check...

# In[21]:


df_train[df_train.isna().any(axis=1)]


# In[22]:


df_test[df_test.isna().any(axis=1)]


# # Feature Engineering

# ## Stratify Age Into Groups
# 
# Looking at our [EDA](https://www.kaggle.com/khoongweihao/part-1-exploratory-data-analysis-eda) for `Age`, we can see that most of the passengers are between 18-30 years of age. We can generate new features that group the passengers into different age groups.

# In[23]:


def age_group(x):
    if x < 18:
        return 'under_18'
    elif x >= 18 and x <=30:
        return '18_to_30'
    else:
        return 'above_30'


# In[24]:


df_train['Age_Group'] = df_train['Age'].apply(lambda x: age_group(x))
df_train.head()


# In[25]:


df_test['Age_Group'] = df_test['Age'].apply(lambda x: age_group(x))
df_test.head()


# Note that we have to encode again!

# In[26]:


cols_to_encode = ['Age_Group']
for col in cols_to_encode:
    le = LabelEncoder()
    le.fit(df_train[col])
    
    df_train[col] = le.transform(df_train[col])
    df_test[col] = le.transform(df_test[col])


# In[27]:


df_train.head()


# In[28]:


df_test.head()


# # More Preprocessing
# 
# Before we begin to train our model for the classification task, we first scale the data with Scikit-learn's `StandardScaler`. And before that, we extract the target column from `df_train`.

# In[29]:


y = df_train['Survived']
#df_train.drop('Survived', axis=1, inplace=True)


# Now we are ready to scale the data and use them for model training.

# In[30]:


cols_to_scale = ['Pclass', 'Age', 'Fare', 'Cabin', 'Embarked', 'Age_Group']
for col in cols_to_scale:
    sc = MinMaxScaler()
    df_train[col] = sc.fit_transform(df_train[col].values.reshape(-1,1))
    sc = MinMaxScaler()
    df_test[col] = sc.fit_transform(df_test[col].values.reshape(-1,1))


# In[31]:


df_train.head()


# In[32]:


df_test.head()


# # Save Preprocessed Datasets For Modeling

# In[33]:


df_train.to_csv('train_preprocessed.csv', index=False)
df_test.to_csv('test_preprocessed.csv', index=False)


# # Finishing Remarks
# 
# Thanks for reading and I welcome your feedback and suggestions for improvement. The notebook will be updated periodically as well.
# 
# Happy Kaggling!
# 
# ---------------------------------------------------------------------
# My notebooks in this series can be found in the links below:
# - [Exploratory Data Analysis (EDA)](https://www.kaggle.com/khoongweihao/part-1-exploratory-data-analysis-eda)
# - [Preprocessing & Feature Engineering](https://www.kaggle.com/khoongweihao/part-2-preprocessing-feature-engineering)
# - [Model Training & Validation Strategies](https://www.kaggle.com/khoongweihao/part-3-model-training-validation-strategies)
# - [Inference & Post-processing Techniques](https://www.kaggle.com/khoongweihao/part-4-inference-and-post-processing-techniques)
# 
# Bonus notebooks include adoption of recent research in terms of models, hyperparameter search, etc. They can be found in the links below:
# - Hyperparameter optimization with Optuna
# - TabNet 
