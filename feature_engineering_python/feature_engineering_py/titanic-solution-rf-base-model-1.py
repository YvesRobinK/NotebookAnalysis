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


# New version

# In[2]:


#Import 
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[3]:


#Check train
train.isnull().sum()


# In[4]:


#Check test
test.isnull().sum()


# In[5]:


from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score


# # Feature Engineering

# In[6]:


full_data = [train, test]


# In[7]:


for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
# Create new feature IsAlone from FamilySize
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
# Remove all NULLS in the Embarked column
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[8]:


import re


# In[9]:


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Create a new feature Title, containing the titles of passenger names
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)


# In[10]:


for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


# In[11]:


for dataset in full_data:
    dataset['Cabin'] = dataset['Cabin'].astype(str).str[0]


# In[12]:


train.head()


# In[13]:


y = train['Survived']
X = train.drop(['Survived','Name','Ticket'], axis = 1)


# In[14]:


numerical_features = [c for c, dtype in zip(X.columns, X.dtypes)
                     if dtype.kind in ['i','f'] and c !='PassengerId']
categorical_features = [c for c, dtype in zip(X.columns, X.dtypes)
                     if dtype.kind not in ['i','f']]


# In[15]:


numerical_features


# In[16]:


categorical_features


# In[17]:


#import train_test_split library
from sklearn.model_selection import train_test_split

# create train test split
X_train, X_val, y_train, y_val = train_test_split( X,  y, test_size=0.3, random_state=0, stratify = y)


# In[18]:


preprocessor = make_column_transformer(
    
    (make_pipeline(
    SimpleImputer(strategy = 'median')
        #,KBinsDiscretizer(n_bins=3)
    ), numerical_features),
    
    (make_pipeline(
    SimpleImputer(strategy = 'constant', fill_value = 'missing'),
    OneHotEncoder(categories = 'auto', handle_unknown = 'ignore')), categorical_features),
)


# In[19]:


#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[20]:


model_pipeline = make_pipeline(preprocessor,RandomForestClassifier(n_estimators = 200) )


# In[21]:


model_pipeline.fit(X_train, y_train)


# In[22]:


X_prediction = model_pipeline.predict(X_train)


# In[23]:


print(f'Train : {model_pipeline.score(X_train, y_train):.3f}')


# In[24]:


print(f'Test : {model_pipeline.score(X_val, y_val):.3f}')


# In[25]:


submission_prediction = model_pipeline.predict(test.drop(['Name','Ticket'], axis = 1)).astype(int)
#


# In[26]:


AllSub = pd.DataFrame({ 'PassengerId': test['PassengerId'],
                       'Survived' : submission_prediction
    
})

AllSub.to_csv("Solution_Pipeline_RF_IMproved.csv", index = False)

