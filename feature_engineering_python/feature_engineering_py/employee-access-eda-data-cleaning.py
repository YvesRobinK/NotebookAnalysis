#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import warnings
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import xgboost as xgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
plt.style.use(style='ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train = pd.read_csv('../input/amazon-employee-access-challenge/train.csv')
test = pd.read_csv('../input/amazon-employee-access-challenge/test.csv')
sample = pd.read_csv('../input/amazon-employee-access-challenge/sampleSubmission.csv')
test_id = test['id']


# In[4]:


print(train.shape)
train.head()


# In[5]:


print(test.shape)
test.head()


# In[6]:


sample.head()


# # Exploration

# ACTION: ACTION is 1 if the resource was approved, 0 if the resource was not.
# 
# RESOURCE: An ID for each resource.
# 
# MGR_ID: The EMPLOYEE ID of the manager of the current EMPLOYEE ID record; an employee may have only one manager at a time.
# 
# ROLE_ROLLUP_1: Company role grouping category id 1 (e.g. US Engineering).
# 
# ROLE_ROLLUP_2: Company role grouping category id 2 (e.g. US Retail).
# 
# ROLE_DEPTNAME: Company role department description (e.g. Retail).
# 
# ROLE_TITLE: Company role business title description (e.g. Senior Engineering Retail Manager).
# 
# ROLE_FAMILY_DESC: Company role family extended description (e.g. Retail Manager, Software Engineering).
# 
# ROLE_FAMILY: Company role family description (e.g. Retail Manager).
# 
# ROLE_CODE: Company role code; this code is unique to each role (e.g. Manager).

# In[7]:


print(train.shape)
print(train.isnull().sum())
train.info()


# In[8]:


print(test.shape)
print(test.isnull().sum())
test.info()


# In[9]:


print("----------Box Plots for detecting outliers---------")
cols = train.columns
for i in cols:
    plt.figure()
    sns.boxplot(train[i])


# In[10]:


# Resource
sns.boxplot(test['RESOURCE'])


# In[11]:


# Role Rollup 1
sns.boxplot(test['ROLE_ROLLUP_1'])


# In[12]:


# Role Rollup 2
sns.boxplot(test['ROLE_ROLLUP_2'])


# In[13]:


# Role Code
sns.boxplot(test['ROLE_CODE'])


# - I guess it's gonna be better if we do nothing with ('resource', 'role rollup 1', 'role_detpname', 'role_code') columns
# - We handle the rest of the outliers

# In[14]:


# resource_outliers = train[train['RESOURCE'] > 150000]
# role_rollup_1_outliers1 = train[train['ROLE_ROLLUP_1'] > 150000]  
# role_rollup_1_outliers2 = train[train['ROLE_ROLLUP_1'] < 100000]
# role_rollup_1_outliers = pd.concat([role_rollup_1_outliers1, role_rollup_1_outliers2], axis=0)
role_rollup_2_outliers1 = train[train['ROLE_ROLLUP_2'] > 200000]
# role_rollup_2_outliers2 = train[train['ROLE_ROLLUP_2'] < 100000]
# role_rollup_2_outliers = pd.concat([role_rollup_2_outliers1, role_rollup_2_outliers2], axis=0)
# role_deptname_outliers = train[train['ROLE_DEPTNAME'] < 100000]
# role_code_outliers = train[train['ROLE_CODE'] > 200000]
# print("Rsource outliers:", len(resource_outliers))
# print("Role Rollup 1 outliers:", len(role_rollup_1_outliers))
print("Role Rollup 2 outliers:", len(role_rollup_2_outliers1))
# print("Role detname outliers:", len(role_deptname_outliers))
# print("Role code outliers:", len(role_code_outliers))


# In[15]:


# print(train.shape)
# lst = [resource_outliers, role_rollup_1_outliers, role_rollup_2_outliers, role_deptname_outliers, role_code_outliers]
# for i in lst:
#     todrop = list(i.index)
#     try:
#         train.drop(todrop, axis=0, inplace=True)
#     except Exception as e:
#         print(len(i))
#         print(e)
#         continue
# print(train.shape)


# In[16]:


print(train.shape)
todrop = list(role_rollup_2_outliers1.index)
train.drop(todrop, axis=0, inplace=True)
print(train.shape)


# In[17]:


train.reset_index(drop=True)


# # EDA

# In[18]:


train.describe().T


# In[19]:


plt.figure(figsize=(10, 8))
sns.heatmap(train.corr(), annot=True)


# ### Resource

# In[20]:


train['RESOURCE'][:10]


# In[21]:


plt.figure(figsize=(9, 6))
sns.catplot('ACTION', 'RESOURCE', data=train)


# ### MGR_ID

# In[22]:


train['MGR_ID'][:10]


# In[23]:


plt.figure(figsize=(9, 6))
sns.catplot('ACTION', 'MGR_ID', data=train)


# We could generate a new feature here called 'is_MGR_ID_BiggerThan150000'

# In[24]:


temp = pd.DataFrame(columns=['is_MGR_ID_BiggerThan150000'], dtype=np.float32)
train = pd.concat([train, temp], axis=1)
for i in range(train.shape[0]):
    try:
        if train['MGR_ID'][i] > 150000:
            train['is_MGR_ID_BiggerThan150000'][i] = 1
        else:
            train['is_MGR_ID_BiggerThan150000'][i] = 0
    except Exception:
        train['is_MGR_ID_BiggerThan150000'][i] = 0
        pass
    
    
temp = pd.DataFrame(columns=['is_MGR_ID_BiggerThan150000'], dtype=np.float32)
test = pd.concat([test, temp], axis=1)
for i in range(test.shape[0]):
    try:
        if test['MGR_ID'][i] > 150000:
            test['is_MGR_ID_BiggerThan150000'][i] = 1
        else:
            test['is_MGR_ID_BiggerThan150000'][i] = 0
    except Exception:
        test['is_MGR_ID_BiggerThan150000'][i] = 0
        pass


# ### ROLE_ROLLUP_1

# In[25]:


train['ROLE_ROLLUP_1'][:10]


# In[26]:


plt.figure(figsize=(12, 12))
sns.catplot('ACTION', 'ROLE_ROLLUP_1', data=train)


# We could generate a new feature here called 'is_ROLE_ROLLUP_1_BiggerThan150000'

# In[27]:


temp = pd.DataFrame(columns=['is_ROLE_ROLLUP_1_BiggerThan150000'], dtype=np.float32)
train = pd.concat([train, temp], axis=1)
for i in range(train.shape[0]):
    try:
        if train['ROLE_ROLLUP_1'][i] > 150000:
            train['is_ROLE_ROLLUP_1_BiggerThan150000'][i] = 1
        else:
            train['is_ROLE_ROLLUP_1_BiggerThan150000'][i] = 0
    except Exception:
        train['is_ROLE_ROLLUP_1_BiggerThan150000'][i] = 0
        pass
    
temp = pd.DataFrame(columns=['is_ROLE_ROLLUP_1_BiggerThan150000'], dtype=np.float32)
test = pd.concat([test, temp], axis=1)
for i in range(test.shape[0]):
    try:
        if test['ROLE_ROLLUP_1'][i] > 150000:
            test['is_ROLE_ROLLUP_1_BiggerThan150000'][i] = 1
        else:
            test['is_ROLE_ROLLUP_1_BiggerThan150000'][i] = 0
    except Exception:
        test['is_ROLE_ROLLUP_1_BiggerThan150000'][i] = 0
        pass


# ### ROLE_ROLLUP_2

# In[28]:


train['ROLE_ROLLUP_2'][:10]


# In[29]:


train['ROLE_ROLLUP_2'].value_counts()


# In[30]:


sns.catplot('ACTION', 'ROLE_ROLLUP_2', data=train)


# We could generate a new feature here called 'is_ROLE_ROLLUP_2_BiggerThan140000'

# In[31]:


temp = pd.DataFrame(columns=['is_ROLE_ROLLUP_2_BiggerThan140000'], dtype=np.float32)
train = pd.concat([train, temp], axis=1)
for i in range(train.shape[0]):
    try:
        if train['ROLE_ROLLUP_2'][i] > 140000:
            train['is_ROLE_ROLLUP_2_BiggerThan140000'][i] = 1
        else:
            train['is_ROLE_ROLLUP_2_BiggerThan140000'][i] = 0
    except Exception:
        train['is_ROLE_ROLLUP_2_BiggerThan140000'][i] = 0
        pass
    
temp = pd.DataFrame(columns=['is_ROLE_ROLLUP_2_BiggerThan140000'], dtype=np.float32)
test = pd.concat([test, temp], axis=1)
for i in range(test.shape[0]):
    try:
        if test['ROLE_ROLLUP_2'][i] > 140000:
            test['is_ROLE_ROLLUP_2_BiggerThan140000'][i] = 1
        else:
            test['is_ROLE_ROLLUP_2_BiggerThan140000'][i] = 0
    except Exception:
        test['is_ROLE_ROLLUP_2_BiggerThan140000'][i] = 0
        pass


# ### ROLE_DEPTNAME

# In[32]:


train['ROLE_DEPTNAME'][:10]


# In[33]:


train['ROLE_DEPTNAME'].value_counts()


# In[34]:


plt.figure(figsize=(9, 6))
sns.catplot('ACTION', 'ROLE_DEPTNAME', data=train)


# In[35]:


len(train[train['ROLE_DEPTNAME'] < 100000])


# Not enough to generate some feature for it

# ### ROLE_TITLE

# In[36]:


train['ROLE_TITLE'][:10]


# In[37]:


train['ROLE_TITLE'].value_counts()


# In[38]:


plt.figure(figsize=(9, 6))
sns.catplot('ACTION', 'ROLE_TITLE', data=train)


# ### ROLE_FAMILY_DESC

# In[39]:


train['ROLE_FAMILY_DESC'][:10]


# In[40]:


train['ROLE_FAMILY_DESC'].value_counts()


# In[41]:


plt.figure(figsize=(9, 7))
sns.catplot('ACTION', 'ROLE_FAMILY_DESC', data=train)


# ### ROLE_FAMILY

# In[42]:


train['ROLE_FAMILY'][:10]


# In[43]:


train['ROLE_FAMILY'].value_counts()


# In[44]:


plt.figure(figsize=(9, 6))
sns.catplot('ACTION', 'ROLE_FAMILY', data=train)


# ### ROLE_CODE

# In[45]:


train['ROLE_CODE'][:10]


# In[46]:


train['ROLE_CODE'].value_counts()


# In[47]:


plt.figure(figsize=(9, 6))
sns.catplot('ACTION', 'ROLE_CODE', data=train)


# We could generate a new feature here called 'is_ROLE_CODE_BiggerThan200000'

# In[48]:


temp = pd.DataFrame(columns=['is_ROLE_CODE_BiggerThan200000'], dtype=np.float32)
train = pd.concat([train, temp], axis=1)
for i in range(train.shape[0]):
    try:
        if train['ROLE_CODE'][i] > 140000:
            train['is_ROLE_CODE_BiggerThan200000'][i] = 1
        else:
            train['is_ROLE_CODE_BiggerThan200000'][i] = 0
    except Exception:
        train['is_ROLE_CODE_BiggerThan200000'][i] = 0
        pass
    
temp = pd.DataFrame(columns=['is_ROLE_CODE_BiggerThan200000'], dtype=np.float32)
test = pd.concat([test, temp], axis=1)
for i in range(test.shape[0]):
    try:
        if test['ROLE_CODE'][i] > 140000:
            test['is_ROLE_CODE_BiggerThan200000'][i] = 1
        else:
            test['is_ROLE_CODE_BiggerThan200000'][i] = 0
    except Exception:
        test['is_ROLE_CODE_BiggerThan200000'][i] = 0
        pass


# In[49]:


train['is_ROLE_CODE_BiggerThan200000'].value_counts()


# In[50]:


train.isnull().sum()


# In[51]:


train[train['is_MGR_ID_BiggerThan150000'].isnull()]
train.drop(32768, axis=0, inplace=True)


# # Modelling Advice
# Your main goal here is to avoid overfitting, I recommend you build a Neural Network (using Pytorch/TF/Keras) and use Dropout(for Regularization) because it's very effective. This is how you get the best results.
# Here is a good introduction for Dropoutt (https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/).

# In[ ]:




