#!/usr/bin/env python
# coding: utf-8

# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/3338/media/gate.png)

# ## Amazon.com - Employee Access Challenge
# 
# When an employee at any company starts work, they first need to obtain the computer access necessary to fulfill their role. This access may allow an employee to read/manipulate resources through various applications or web portals. It is assumed that employees fulfilling the functions of a given role will access the same or similar resources. It is often the case that employees figure out the access they need as they encounter roadblocks during their daily work (e.g. not able to log into a reporting portal). A knowledgeable supervisor then takes time to manually grant the needed access in order to overcome access obstacles. As employees move throughout a company, this access discovery/recovery cycle wastes a nontrivial amount of time and money.
# 
# There is a considerable amount of data regarding an employee’s role within an organization and the resources to which they have access. Given the data related to current employees and their provisioned access, models can be built that automatically determine access privileges as employees enter and leave roles within a company. These auto-access models seek to minimize the human involvement required to grant or revoke employee access.

# ## Part 1. Get started.
# 
# In any competition it is always a good idea to have a look at your data first, so you won't blindly rush into the coding/tuning.

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


# Unfortunately due to some glitch file `test.csv` on Kaggle is broken, it has almost 1M records instead of 56K. We have to use another source for our data, luckily for us `CatBoost` library has Amazon dataset build in.

# In[2]:


# Loading data directly from CatBoost
from catboost.datasets import amazon

train, test = amazon()


# In[3]:


print("Train shape: {}, Test shape: {}".format(train.shape, test.shape))


# In[4]:


train.head(5)


# In[5]:


test.head(5)


# Ok, so dataset has 9 columns, plus target (`ACTION`) for train and `id` for test. All these columns are categorical encoded as integers.
# 
# Let's count unique category values.

# In[6]:


train.apply(lambda x: len(x.unique()))


# First, I'd like to draw your attention to columns `RESOURCE`,`MGR_ID` and `ROLE_FAMILY_DESC`. These 3 columns are high-cardinality categorical features. That means they have a lot of unique values and that makes them harder to encode. 
# 
# Also, take a look on `ROLE_CODE` and `ROLE_TITLE`. These 2 columns have exactly the same amount of unique values. That’s suspicious.
# 
# Let's have a closer look.

# In[7]:


import itertools
target = "ACTION"
col4train = [x for x in train.columns if x!=target]

col1 = 'ROLE_CODE'
col2 = 'ROLE_TITLE'

pair = len(train.groupby([col1,col2]).size())
single = len(train.groupby([col1]).size())

print(col1, col2, pair, single)


# It seems like these 2 columns have 1:1 relationship. For each unique value in column `ROLE_CODE` there is 1 and only 1 unique value in column `ROLE_TITLE`. In other words we don't need both columns to build a model, so I'm removing `ROLE_TITLE`.

# In[8]:


col4train = [x for x in col4train if x!='ROLE_TITLE']


# Ok, that's it for our very short data analysis, unfortunately deeper analysis is out-of-scope of our today's topic.
# 
# At the end let's transform our data using the most straight-forward approach - one-hot encoding using [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) transformation from [scikit-learn](https://scikit-learn.org/stable/index.html) package.
# 
# After transformation we will fit [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) and check it performance using 5-fold cross-validation.

# In[9]:


#linear - OHE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=True, dtype=np.float32, handle_unknown='ignore')


# In[10]:


X = ohe.fit_transform(train[col4train])
y = train["ACTION"].values


# For CV I'm going to use [cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html) function, this function is quite useful, you can even pass fit parameters to estimator.

# In[11]:


from sklearn.model_selection import cross_validate

model = LogisticRegression(
                penalty='l2',  
                C=1.0, 
                fit_intercept=True, 
                random_state=432,
                solver = 'liblinear',
                max_iter = 1000,
        )
stats = cross_validate(model, X, y, groups=None, scoring='roc_auc', 
                       cv=5, n_jobs=2, return_train_score = True)
stats = pd.DataFrame(stats)
stats.describe().transpose()


# Our linear model gets AUC score of 0.8636, which is pretty ok, but not good enough for Kaggle competition. 
# Let's find out how we can transform/engineer our features to get better score.

# Let's check LB score for our baseline.

# In[12]:


X = ohe.fit_transform(train[col4train])
y = train["ACTION"].values
X_te = ohe.transform(test[col4train])

model.fit(X,y)
predictions = model.predict_proba(X_te)[:,1]

submit = pd.DataFrame()
submit["Id"] = test["id"]
submit["ACTION"] = predictions

submit.to_csv("submission.csv", index = False)


# In[13]:




