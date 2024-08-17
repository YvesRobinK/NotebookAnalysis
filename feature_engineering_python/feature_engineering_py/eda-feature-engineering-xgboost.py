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


# ## EDA

# In[2]:


train = pd.read_csv('/kaggle/input/tabular-playground-series-dec-2021/train.csv')
test = pd.read_csv('/kaggle/input/tabular-playground-series-dec-2021/test.csv')
submission = pd.read_csv('/kaggle/input/tabular-playground-series-dec-2021/sample_submission.csv')
train


# In[3]:


train.drop(['Id'], axis = 1, inplace = True)
test.drop(['Id'], axis = 1, inplace = True)
TARGET  = 'Cover_Type'
FEATURES = [col for col in train.columns if col not in ['Id', TARGET]]
train.info()


# In[4]:


df = pd.concat([train[FEATURES], test[FEATURES]], axis = 0)

cat_features = [col for col in df.columns if df[col].nunique() < 10]
cont_features = [col for col in df.columns if df[col].nunique() >= 10]

del df
print('Categorical features:', len(cat_features))
print('Continuous features:', len(cont_features))


# In[5]:


import matplotlib.pyplot as plt

plt.pie([len(cat_features), len(cont_features)], 
       labels = ['Categorical', 'Continuous'], 
       autopct = '%.2f%%')
plt.title('Continuous vs Categorical features')
plt.show()


# ## Continuous Features

# In[6]:


import seaborn as sns

ncols = 5
nrows = 2

fig, axes = plt.subplots(nrows, ncols, figsize = (20, 10))

for r in range(nrows):
    for c in range(ncols):
        col = cont_features[r * ncols + c]
        sns.kdeplot(x = train[col], ax = axes[r, c], label = 'Train data')
        sns.kdeplot(x = test[col], ax = axes[r, c], label = 'Test data')
plt.show()


# ## Categorical Features

# In[7]:


ncols = 5
nrows = int(len(cat_features) / ncols + (len(FEATURES) % ncols > 0)) 

fig, axes = plt.subplots(nrows, ncols, figsize = (18, 45))

for r in range(nrows):
    for c in range(ncols):
        if r * ncols + c >= len(cat_features):
            break
        col = cat_features[r * ncols + c]
        sns.countplot(x = train[col], ax = axes[r, c], label = 'Train data')
        sns.countplot(x = test[col], ax = axes[r, c], label = 'Test data')
plt.show()


# In[8]:


# Since Soil_Type7 and Soil_Type15 are all 0 values
train = train.drop(labels = ["Soil_Type7" , "Soil_Type15"] ,axis = 1)
FEATURES.remove('Soil_Type7')
FEATURES.remove('Soil_Type15')


# ## Target Distribution

# In[9]:


sns.countplot(x = train[TARGET])
plt.show()


# In[10]:


train[TARGET].value_counts().sort_index()


# In[11]:


train.loc[train['Cover_Type'] == 5]


# In[12]:


# Since Cover_Type = 5 has only one sample, it's prob. safe to remove it. 
train.drop(train.loc[train['Cover_Type'] == 5].index, inplace = True)

train['Cover_Type'].value_counts()


# ## Feature Engineering

# In[13]:


train["mean"] = train[FEATURES].mean(axis = 1)
train["std"] = train[FEATURES].std(axis = 1)
train["min"] = train[FEATURES].min(axis = 1)
train["max"] = train[FEATURES].max(axis = 1)

test["mean"] = test[FEATURES].mean(axis = 1)
test["std"] = test[FEATURES].std(axis = 1)
test["min"] = test[FEATURES].min(axis = 1)
test["max"] = test[FEATURES].max(axis = 1)

FEATURES.extend(['mean', 'std', 'min', 'max'])


# In[14]:


X = train.drop([TARGET], axis = 1)
y = train[TARGET]
X_test = test[FEATURES]


# In[15]:


# from xgboost import XGBClassifier
# from sklearn.model_selection import cross_val_score

# xgb_params = {
#     'objective': 'multi:softmax',
#     'eval_metric': 'mlogloss',
#     'tree_method': 'gpu_hist',
#     'predictor': 'gpu_predictor',
#     }

# model = XGBClassifier(**xgb_params)
# cross_val_score(model, X, y, cv = 5, scoring = 'accuracy')


# In[16]:


# Since a very large dataset(4000000 rows), one validation set is enough instead of cross-validation
# But remember to do stratified sampling
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.25, stratify = train['Cover_Type'])
# test_size = 0.25 since the data is very large


# ## XGBoost Classifier

# In[17]:


from xgboost import XGBClassifier

xgb_params = {
    'objective': 'multi:softmax',
    'eval_metric': 'mlogloss', 
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    }

model = XGBClassifier(**xgb_params)


# In[18]:


from sklearn.metrics import accuracy_score

model.fit(X_train, y_train)
y_val_pred = model.predict(X_val)
accuracy_score(y_val, y_val_pred)


# In[19]:


y_pred = model.predict(X_test)

submission[TARGET] = y_pred
submission.to_csv('submission.csv', index = False)


# In[ ]:




