#!/usr/bin/env python
# coding: utf-8

# ### **This kernel will serve as a starting point with a potential ensemble model.**
# ### **Future work on feature engineering will be incorporated in the future**

# # 1. **Import necessary libraries**

# In[1]:


import janestreet
env = janestreet.make_env() # initialize the environment
iter_test = env.iter_test() # an iterator which loops over the test set


# In[2]:


import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
#tree classifier
import xgboost as xgb
import lightgbm as lgb 
from sklearn.ensemble import RandomForestClassifier as rf
#stacking
from sklearn.ensemble import StackingClassifier

import warnings
warnings.filterwarnings("ignore")
print("XGBoost version:", xgb.__version__)
print("XGBoost version:", lgb.__version__)


# # 2. **Load training and testing data**

# In[3]:


get_ipython().run_cell_magic('time', '', 'train = pd.read_csv(\'/kaggle/input/jane-street-market-prediction/train.csv\')\nfeatures = pd.read_csv(\'../input/jane-street-market-prediction/features.csv\')\nexample_test = pd.read_csv(\'../input/jane-street-market-prediction/example_test.csv\')\nsample_prediction_df = pd.read_csv(\'../input/jane-street-market-prediction/example_sample_submission.csv\')\nprint ("Data is loaded!")\n')


# # 3. **Data pre-processing**
# The naive approach has been taken as a starting point. More will be added in the future.

# In[4]:


# I have taked this cell from https://www.kaggle.com/drcapa/jane-street-market-prediction-starter-xgb
# but I am not sure about the choice
train = train[train['weight'] != 0]
train['action'] = (train['resp']>0)*1

X_train = train.loc[:, train.columns.str.contains('feature')]
y_train = train.loc[:, 'action']

X_train = X_train.fillna(-999)
del train


# # 4.1. **Base models defined**

# In[5]:


lgbclf = lgb.LGBMClassifier(
        n_estimators=64,
        max_depth=8,
        learning_rate=0.01,
        subsample=0.85,
        colsample_bytree=0.85,
        boosting_type= "gbdt",
        nthread=-1,
        metric="AUC",
        random_state=2020
    )
xgbclf = xgb.XGBClassifier(
        n_estimators=64,
        max_depth=8,
        learning_rate=0.01,
        subsample=0.85,
        colsample_bytree=0.85,
        missing=-999,
        tree_method='gpu_hist',
        nthread=-1,
        random_state=2020
    )
rfclf = rf(
            n_estimators=64,
            max_depth=8, 
            max_features='sqrt',
            n_jobs=-1,
            random_state=2020
    )


# # 4.2. **Models stacking**

# In[6]:


# level-1 ensemble bass models
models = [
    ('xgb',xgbclf),
    #('lgb',lgbclf),
    #('rf',rfclf)
    
]
# level-2 random forest is stacked over the base models
stack_clf = StackingClassifier(models,final_estimator=rfclf,cv=2)   


# # 5. **Training and submission generation**

# In[7]:


get_ipython().run_line_magic('time', 'stack_clf.fit(X_train, y_train)')


# In[8]:


for (test_df, sample_prediction_df) in iter_test:
    X_test = test_df.loc[:, test_df.columns.str.contains('feature')]
    X_test.fillna(-999)
    y_preds = stack_clf.predict(X_test)
    sample_prediction_df.action = y_preds
    env.predict(sample_prediction_df)

