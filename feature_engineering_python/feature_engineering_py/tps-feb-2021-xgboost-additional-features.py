#!/usr/bin/env python
# coding: utf-8

# ### Trying to create new features when the features are anonymised, is taking a stab in the dark. It is more of a hope than expectations. So I did!
# 
# #### Highlights: 
# * Features I combined: **cat1, cat8, cat9** (cat9 x cat1, cat8 x cat1)
# * Cat features are label encoded first
# 
# ||5 fold local CV|||
# |---|---|---|---|
# | Model|lr =0.1 | lr =0.01 |lr=0.005|
# | base-model | 0.843853 | 0.842680 |0.842935|
# | with new features | 0.843737 | 0.842648 |0.842927|
# |score gain|0.000116|0.000032|0.000008|
# 
# ### Remark:
# The result is based on realtively coarse model. Outcome may be different for fine-tuned models. If you find something different with your models please let me know. 
# 
# The score-gained due to the additional features has decreased with lowering the learning rate. Why?  
# 
#          
# 

# # 0. Set-up

# In[1]:


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # 1. Load data

# In[2]:


train = pd.read_csv(r'/kaggle/input/tabular-playground-series-feb-2021/train.csv', index_col= 'id')
test = pd.read_csv(r'/kaggle/input/tabular-playground-series-feb-2021/test.csv', index_col= 'id')
submission = pd.read_csv(r'/kaggle/input/tabular-playground-series-feb-2021/sample_submission.csv', index_col= 'id')


# # 2. Data processing

# In[3]:


target = train.pop('target')
y = target


# In[4]:


cat_features = [col for col in train.columns if train[col].dtype=='object']
num_features = [col for col in train.columns if train[col].dtype=='float']


le = LabelEncoder()

le_train = train.copy()
le_test = test.copy()

for col in cat_features:
    le_train[col] = le.fit_transform(train[col])
    le_test[col] = le.transform(test[col])


# In[5]:


train = le_train
test = le_test


# In[6]:


features = train.columns
len(features)


# # 3. Base model: xgboost

# In[7]:


Nfold = 5
SEED = 100

kfold = KFold(n_splits=Nfold, shuffle=True, random_state=SEED)
oof_preds = np.zeros(train.shape[0])
subm_preds_xgb = np.zeros(test.shape[0])

for n_fold, (trn_idx, val_idx) in enumerate(kfold.split(train)):
    trn_x, trn_y = train[features].iloc[trn_idx], y.iloc[trn_idx]
    val_x, val_y = train[features].iloc[val_idx], y.iloc[val_idx]
    
    xgb = XGBRegressor(max_depth=6,
        learning_rate=0.005,
        n_estimators=5000,
        verbosity=1,
        silent=None,
        objective='reg:squarederror',
        booster='gbtree',
        n_jobs=-1,
        nthread=None,
        gamma=0.0,
        min_child_weight= 133, 
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=7.5,
        reg_lambda=0.25,
        random_state=SEED,
        tree_method = 'gpu_hist',
        predictor = 'gpu_predictor',
    )                      

    
    xgb.fit(trn_x, trn_y,
           eval_set =[(trn_x, trn_y), (val_x, val_y)],
           eval_metric="rmse", verbose=1000, early_stopping_rounds=40
           )
   
    oof_preds[val_idx] = xgb.predict(val_x)
    subm_preds_xgb += xgb.predict(test[features])/kfold.n_splits
    
    print('Fold {} MSE : {:.6f}'.format(n_fold + 1, mean_squared_error(val_y, oof_preds[val_idx], squared=False)))   
      
    
print("*****************************************************************")
print('{} fold local CV= {:.6f}'.format(Nfold, mean_squared_error(y, oof_preds, squared=False)))


# # 4. Feature engineering
# ## 4.1 Mutual Info Regression

# In[8]:


# the following two code snippets are adapted from the "feature engineering kaggle min-course"

from sklearn.feature_selection import mutual_info_regression

features = train.dtypes == int

def make_mi_scores(train, y, discrete_features):
    mi_scores = mutual_info_regression(train, y, discrete_features=features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=train.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(train, y, features)
mi_scores


# In[9]:


def plot_utility_scores(scores):
    y = scores.sort_values(ascending=True)
    width = np.arange(len(y))
    ticks = list(y.index)
    plt.barh(width, y, color='#d1aeab', alpha=0.9)
    plt.yticks(width, ticks)
    plt.grid()
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_utility_scores(mi_scores)


# ## 4.2 Additional Features Created

# In[10]:


# I tried other combinations of features based on the mutual information score 
# but these two gave the best improvement

train['9t1'] = train['cat9']*train['cat1']
train['8t1'] = train['cat8']*train['cat1']

test['9t1'] = test['cat9']*test['cat1']
test['8t1'] = test['cat8']*test['cat1']


# In[11]:


features = train.columns
len(features)


# # 5. Model with the additional features

# In[12]:


Nfold = 5
SEED = 100

kfold = KFold(n_splits=Nfold, shuffle=True, random_state=SEED)
oof_preds = np.zeros(train.shape[0])
subm_preds_xgb = np.zeros(test.shape[0])

for n_fold, (trn_idx, val_idx) in enumerate(kfold.split(train)):
    trn_x, trn_y = train[features].iloc[trn_idx], y.iloc[trn_idx]
    val_x, val_y = train[features].iloc[val_idx], y.iloc[val_idx]
    
    xgb = XGBRegressor(max_depth=6,
        learning_rate=0.005,
        n_estimators=5000,
        verbosity=1,
        silent=None,
        objective='reg:squarederror',
        booster='gbtree',
        n_jobs=-1,
        nthread=None,
        gamma=0.0,
        min_child_weight= 133, 
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=7.5,
        reg_lambda=0.25,
        random_state=SEED,
        tree_method = 'gpu_hist',
        predictor = 'gpu_predictor',
    )                      

    
    xgb.fit(trn_x, trn_y,
           eval_set =[(trn_x, trn_y), (val_x, val_y)],
           eval_metric="rmse", verbose=1000, early_stopping_rounds=40
           )
   
    oof_preds[val_idx] = xgb.predict(val_x)
    subm_preds_xgb += xgb.predict(test[features])/kfold.n_splits
    
    print('Fold {} MSE : {:.6f}'.format(n_fold + 1, mean_squared_error(val_y, oof_preds[val_idx], squared=False)))   
      
    
print("*****************************************************************")
print('{} fold local CV= {:.6f}'.format(Nfold, mean_squared_error(y, oof_preds, squared=False)))


# ### Thank you for your interest in this notebook!
