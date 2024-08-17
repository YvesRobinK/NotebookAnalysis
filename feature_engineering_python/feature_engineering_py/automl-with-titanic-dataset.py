#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## **Load Data**

# In[26]:


train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
train_df.head()


# In[27]:


test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
test_df.head()


# In[28]:


# train_test_split
X_train, y_train = train_df.drop(['Survived'], axis=1), train_df['Survived']


# ## **Install MLJAR**

# In[29]:


get_ipython().system('pip install -q -U git+https://github.com/mljar/mljar-supervised.git@master')


# ## **Extended EDA**

# In[30]:


from supervised.preprocessing.eda import EDA

EDA.extensive_eda(X_train, y_train, save_path='./')


# ## **AutoML with raw data**

# In[31]:


from supervised.automl import AutoML

automl_raw = AutoML(total_time_limit = 60*15,
                        model_time_limit = 60,
                        mode='Compete',
                        train_ensemble=True)
automl_raw.fit(X_train, y_train)


# In[32]:


pd.set_option('display.max_rows', None)
automl_raw.get_leaderboard()


# In[33]:


automl_raw.report()


# In[34]:


automl_raw.predict(test_df)


# In[35]:


submission_raw = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
submission_raw['Survived'] = automl_raw.predict(test_df)


# In[57]:


submission_raw.to_csv('submission_raw.csv')
submission_raw.head()


# ## **Additional Feature Engineering**

# Since there are lots of null values in 'Age', filling age by using other features may be helpful.  
# In this notebook, **'Name'** is used to fill values.

# In[37]:


train_df.info()


# In[38]:


train_df.isnull().sum()


# In[40]:


# Extract 'Initial' from 'Name'

train_df['Initial'] = 0
for i in train_df:
    train_df['Initial'] = train_df.Name.str.extract('([A-Za-z]+)\.')
    
train_df['Initial'].value_counts()


# In[41]:


# replace misspelled initials

train_df['Initial'].replace([
    'Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess', 'Jonkheer', 'Col',
    'Rev', 'Capt', 'Sir', 'Don'], 
    ['Miss', 'Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs', 'Other', 'Other',
    'Other', 'Mr', 'Mr', 'Mr'], inplace=True)


# In[42]:


train_df.groupby('Initial')['Age'].mean()


# In[43]:


# fill Age with mean age of Initials

train_df.loc[(train_df.Age.isnull()) & (train_df.Initial == 'Mr'), 'Age'] = 33
train_df.loc[(train_df.Age.isnull()) & (train_df.Initial == 'Mrs'), 'Age'] = 36
train_df.loc[(train_df.Age.isnull()) & (train_df.Initial == 'Master'), 'Age'] = 5
train_df.loc[(train_df.Age.isnull()) & (train_df.Initial == 'Miss'), 'Age'] = 22
train_df.loc[(train_df.Age.isnull()) & (train_df.Initial == 'Other'), 'Age'] = 46


# In[44]:


train_df.Age.isnull().any()


# In[45]:


# drop 'Name' and 'Initial'

train_df.drop(['Name', 'Initial'], axis=1, inplace=True)


# In[50]:


train_df.head()


# ## **AutoML with preprocessed data**

# In[47]:


from supervised.automl import AutoML

automl_preprocessed = AutoML(total_time_limit = 60*15,
                        model_time_limit = 60,
                        mode='Compete',
                        train_ensemble=True)
automl_preprocessed.fit(X_train, y_train)


# In[49]:


pd.set_option('display.max_rows', None)
automl_preprocessed.get_leaderboard()


# In[52]:


automl_preprocessed.report()


# In[53]:


automl_preprocessed.predict(test_df)


# In[54]:


submission_preprocessed = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
submission_preprocessed['Survived'] = automl_preprocessed.predict(test_df)


# In[56]:


submission_raw.to_csv('submission_preprocessed.csv')
submission_raw.head()


# AutoML with **preprocessed data** gives a better model.  
# Still, understanding and preprocessing data is important!!⭐
