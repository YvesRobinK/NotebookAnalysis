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


# **Downloading data**

# In[2]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[3]:


train


# **Feature engineering**
# - first letter of 'Cabin'
# - number of 'Cabin'
# - quantity of cabins

# In[4]:


import re
train['Cabin'] = train.Cabin.fillna('None')
train['Cabin1'] = train.Cabin.apply(lambda x: x[0])
train['Cabin_num'] = train.Cabin.apply(lambda x: re.findall(r'\d+', x))
train['Cabin_'] = train.Cabin_num.apply(lambda x: len(x))
train['Cabin_num'] = train.Cabin_num.apply(lambda x: int(x[0]) if len(x)>0 else 0)

test['Cabin'] = test.Cabin.fillna('None')
test['Cabin1'] = test.Cabin.apply(lambda x: x[0])
test['Cabin_num'] = test.Cabin.apply(lambda x: re.findall(r'\d+', x))
test['Cabin_'] = test.Cabin_num.apply(lambda x: len(x))
test['Cabin_num'] = test.Cabin_num.apply(lambda x: int(x[0]) if len(x)>0 else 0)

train = train.drop('Cabin', axis=1)
test = test.drop('Cabin', axis=1)


# In[5]:


train


# **Filling missing values**

# In[6]:


train = train.fillna(train.median())
test = test.fillna(test.median())


# In[7]:


train.isna().sum()


# In[8]:


train = train.apply(lambda x:x.fillna(x.value_counts().index[0]))
test = test.apply(lambda x:x.fillna(x.value_counts().index[0]))


# In[9]:


train


# **Dropping 'useless' columns**

# In[10]:


train = train.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
test = test.drop(['PassengerId', 'Name', 'Ticket'], axis=1)


# **Some information and plots**

# In[11]:


train.info()


# In[12]:


train.Sex.hist()


# In[13]:


# train.Cabin.hist()
# train.Cabin.unique()


# In[14]:


test.Embarked.hist()


# **One-hot encoding of categorical features**

# In[15]:


train1 = pd.get_dummies(train)
test1 = pd.get_dummies(test)


# **Creating train dataset and target**

# In[16]:


target = train1.Survived
train_df = train1.drop(['Survived','Cabin1_T'], axis=1)
# train = train.drop('Survived', axis=1)


# In[17]:


test1


# **The model - 
# CatBoostClassifier**

# In[18]:


from sklearn.model_selection import cross_val_score
import catboost
# from catboost import Pool

MAX_ITER = 6000
PATIENCE = 100
DISPLAY_FREQ = 100

MODEL_PARAMS = {'random_seed': 1234,    
                'learning_rate': 0.001,                
                'iterations': MAX_ITER,
                'early_stopping_rounds': PATIENCE,
                'metric_period': DISPLAY_FREQ,
#                 'use_best_model': True,
                'eval_metric': 'Accuracy',
#                 'task_type': 'GPU',
                'one_hot_max_size': 9
               }

# pool_train = Pool(train, target,
#                   cat_features = ['Sex', 'Embarked'])
# pool_test = Pool(test, cat_features = ['Sex', 'Embarked'])

model = catboost.CatBoostClassifier(**MODEL_PARAMS)
model.fit(train_df, target,
        eval_set=[(train_df, target)],
#         early_stopping_rounds = PATIENCE,
#         metric_period = DISPLAY_FREQ
         )


# In[19]:


# from sklearn.linear_model import LogisticRegression

# model = LogisticRegression()

# model.fit(train_df, target)


# In[20]:


# scores = cross_val_score(model, train, target, cv=5)
# scores


# **Making prediction**

# In[21]:


prediction = model.predict(test1)


# In[22]:


prediction


# **Making submission**

# In[23]:


sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')


# In[24]:


sub.Survived = prediction


# In[25]:


sub


# In[26]:


sub.to_csv('submission.csv', index = False)

