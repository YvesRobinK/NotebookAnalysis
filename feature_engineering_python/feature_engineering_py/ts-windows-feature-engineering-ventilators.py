#!/usr/bin/env python
# coding: utf-8

# ## Actual time-series and window features + some more original feature engineering. 
# * Big boost compared to merely aggregate features
# * 3X speedup and better results by modelling  without the expiratory phase
# * Features applied in a modular fashion for easier, safer changes.
# * Many base features based on the notebook: https://www.kaggle.com/shivansh002/lgbm-lover-s

# In[1]:


FAST_RUN = False #False#True ## if true, run with 223k rows for faster debugging

if FAST_RUN:
    n_splits = 3
else:
    n_splits = 5


# In[2]:


get_ipython().run_cell_magic('time', '', 'import pandas as pd\nimport numpy as np\nif FAST_RUN:\n    train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv",nrows=223456)\nelse:\n    train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")\ntest = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")\nsubmission = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")\ntrain\n\nfeatures = ["R", "C", "time_step", "u_in", "u_out"]\ntarget = train["pressure"]\n')


# In[3]:


train


# In[4]:


train.head(200000).describe().round(2)


# In[5]:


train.head(100000)["breath_id"].value_counts().describe() ## ~Most all series have 80 time steps


# In[6]:


### we see that R,C are fixed for a given breath id - no point in making TS feats from them
train.head(100000).groupby("breath_id").std().describe()


# In[7]:


train.dtypes


# In[8]:


def add_feats(train):
    # # rewritten calculation of lag features from this notebook: https://www.kaggle.com/patrick0302/add-lag-u-in-as-new-feat
# # some of ideas from this notebook: https://www.kaggle.com/mst8823/google-brain-lightgbm-baseline
    # train[["15_out_mean"]] = train.groupby('breath_id')['u_out'].rolling(window=15,min_periods=1).agg({"15_out_mean":"mean"}).reset_index(level=0,drop=True)
    train['last_value_u_in'] = train.groupby('breath_id')['u_in'].transform('last')
    train['u_in_lag1'] = train.groupby('breath_id')['u_in'].shift(1)
    train['u_out_lag1'] = train.groupby('breath_id')['u_out'].shift(1)
    train['u_in_lag_back1'] = train.groupby('breath_id')['u_in'].shift(-1)
    train['u_out_lag_back1'] = train.groupby('breath_id')['u_out'].shift(-1)
    train['u_in_lag2'] = train.groupby('breath_id')['u_in'].shift(2)
    train['u_out_lag2'] = train.groupby('breath_id')['u_out'].shift(2)
    train['u_in_lag3'] = train.groupby('breath_id')['u_in'].shift(3)
    train['u_out_lag3'] = train.groupby('breath_id')['u_out'].shift(3)
    train['u_in_lag_back2'] = train.groupby('breath_id')['u_in'].shift(-2)
    train['u_out_lag_back2'] = train.groupby('breath_id')['u_out'].shift(-2)
    train['u_in_lag_back3'] = train.groupby('breath_id')['u_in'].shift(-3)
    train['u_out_lag_back3'] = train.groupby('breath_id')['u_out'].shift(-3)
    train['u_in_lag_back10'] = train.groupby('breath_id')['u_in'].shift(-10)
    train['u_out_lag_back10'] = train.groupby('breath_id')['u_out'].shift(-10)

    train['u_in_first'] = train.groupby('breath_id')['u_in'].first()
    train['u_out_first'] = train.groupby('breath_id')['u_out'].first()

    ## time since last step
    train['time_step_diff'] = train.groupby('breath_id')['time_step'].diff().fillna(0)
    ### rolling window ts feats
    train['ewm_u_in_mean'] = train.groupby('breath_id')['u_in'].ewm(halflife=9).mean().reset_index(level=0,drop=True)
    train['ewm_u_in_std'] = train.groupby('breath_id')['u_in'].ewm(halflife=10).std().reset_index(level=0,drop=True) ## could add covar?
    train['ewm_u_in_corr'] = train.groupby('breath_id')['u_in'].ewm(halflife=15).corr().reset_index(level=0,drop=True) # self umin corr
    # train['ewm_u_in_corr'] = train.groupby('breath_id')['u_in'].ewm(halflife=6).corr(train.groupby('breath_id')["u_out"]).reset_index(level=0,drop=True) # corr with u_out # error
    ## rolling window of 15 periods
    train[["15_in_sum","15_in_min","15_in_max","15_in_mean","15_out_std"]] = train.groupby('breath_id')['u_in'].rolling(window=15,min_periods=1).agg({"15_in_sum":"sum","15_in_min":"min","15_in_max":"max","15_in_mean":"mean","15_in_std":"std"}).reset_index(level=0,drop=True)
#     train[["45_in_sum","45_in_min","45_in_max","45_in_mean","45_out_std"]] = train.groupby('breath_id')['u_in'].rolling(window=45,min_periods=1).agg({"45_in_sum":"sum","45_in_min":"min","45_in_max":"max","45_in_mean":"mean","45_in_std":"std"}).reset_index(level=0,drop=True)
    train[["45_in_sum","45_in_min","45_in_max","45_in_mean","45_out_std"]] = train.groupby('breath_id')['u_in'].rolling(window=45,min_periods=1).agg({"45_in_sum":"sum","45_in_min":"min","45_in_max":"max","45_in_mean":"mean","45_in_std":"std"}).reset_index(level=0,drop=True)

    train[["15_out_mean"]] = train.groupby('breath_id')['u_out'].rolling(window=15,min_periods=1).agg({"15_out_mean":"mean"}).reset_index(level=0,drop=True)

    print(train.shape[0])
    display(train)
    train = train.fillna(0) # ORIG

    # max, min, mean value of u_in and u_out for each breath
    train['breath_id__u_in__max'] = train.groupby(['breath_id'])['u_in'].transform('max')
    train['breath_id__u_out__max'] = train.groupby(['breath_id'])['u_out'].transform('max')

    train['breath_id__u_out__mean'] =train.groupby(['breath_id'])['u_out'].mean()
    train['breath_id__u_in__mean'] =train.groupby(['breath_id'])['u_in'].mean()

    train['breath_id__u_in__min'] = train.groupby(['breath_id'])['u_in'].transform('min')
    train['breath_id__u_out__min'] = train.groupby(['breath_id'])['u_out'].transform('min')

    train['R_div_C'] = train["R"].div(train["C"])

    # difference between consequitive values
    train['R__C'] = train["R"].astype(str) + '__' + train["C"].astype(str)
    train['u_in_diff1'] = train['u_in'] - train['u_in_lag1']
    train['u_out_diff1'] = train['u_out'] - train['u_out_lag1']
    train['u_in_diff2'] = train['u_in'] - train['u_in_lag2']
    train['u_out_diff2'] = train['u_out'] - train['u_out_lag2']
    train['u_in_diff3'] = train['u_in'] - train['u_in_lag3']
    train['u_out_diff3'] = train['u_out'] - train['u_out_lag3']
    ## diff between last 2 steps
    train['u_in_diff_1_2'] = train['u_in_lag1'] - train['u_in_lag2']
    train['u_out_diff_1_2'] = train['u_out_lag1'] - train['u_out_lag2']
    train['u_in_lagback_diff_1_2'] = train['u_in_lag_back1'] - train['u_in_lag_back2']
    train['u_out_lagback_diff_1_2'] = train['u_out_lag_back1'] - train['u_out_lag_back2']

    train['u_in_lagback_diff1'] = train['u_in'] - train['u_in_lag_back1']
    train['u_out_lagback_diff1'] = train['u_out'] - train['u_out_lag_back1']
    train['u_in_lagback_diff2'] = train['u_in'] - train['u_in_lag_back2']
    train['u_out_lagback_diff2'] = train['u_out'] - train['u_out_lag_back2']

    # from here: https://www.kaggle.com/yasufuminakama/ventilator-pressure-lstm-starter
    train.loc[train['time_step'] == 0, 'u_in_diff'] = 0
    train.loc[train['time_step'] == 0, 'u_out_diff'] = 0

    # difference between the current value of u_in and the max value within the breath
    train['breath_id__u_in__diffmax'] = train.groupby(['breath_id'])['u_in'].transform('max') - train['u_in']
    train['breath_id__u_in__diffmean'] = train.groupby(['breath_id'])['u_in'].transform('mean') - train['u_in']

    print("before OHE")
    display(train)

    # OHE
    train = train.merge(pd.get_dummies(train['R'], prefix='R'), left_index=True, right_index=True).drop(['R'], axis=1)
    train = train.merge(pd.get_dummies(train['C'], prefix='C'), left_index=True, right_index=True).drop(['C'], axis=1)
    train = train.merge(pd.get_dummies(train['R__C'], prefix='R__C'), left_index=True, right_index=True).drop(['R__C'], axis=1)

    # https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/273974
    train['u_in_cumsum'] = train.groupby(['breath_id'])['u_in'].cumsum()
    train['time_step_cumsum'] = train.groupby(['breath_id'])['time_step'].cumsum()

    # feature by u in or out (ideally - make 2 sep columns for each state) # dan
    train['u_in_partition_out_sum'] = train.groupby(['breath_id',"u_out"])['u_in'].transform("sum")

    train = train.fillna(0) # add for consistency with how test is done - dan

    return train


# In[9]:


get_ipython().run_cell_magic('time', '', 'train = add_feats(train)\n')


# In[10]:


get_ipython().run_cell_magic('time', '', 'test = add_feats(test)\n')


# In[11]:


from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os
import numpy as np
import time
import lightgbm as lgb

from sklearn.model_selection import GroupKFold 
from sklearn.model_selection import  KFold
from sklearn import metrics


# In[12]:


#remove expiratory phase from training #dan
print(train.shape[0])
train = train.loc[train["u_out"] != 1]
print(train.shape[0])


# In[13]:


scores = []
feature_importance = pd.DataFrame()
models = []
columns = [col for col in train.columns if col not in ['id', 'breath_id', 'pressure']]
X = train[columns]
y = train['pressure']
params = {'objective': 'regression',
          'learning_rate': 0.25,
          "boosting_type": "gbdt",
          'min_data_in_leaf': 120,#600,
          'max_bin': 210, #196,
#           'device':'gpu',
          'feature_fraction': 0.5, #0.4,
          'lambda_l1':36, 'lambda_l2':80,
          'max_depth':16,
          'num_leaves':1000,
          "metric": 'mae',
          'n_jobs': -1
         }
folds = GroupKFold(n_splits=n_splits)
for fold_n, (train_index, valid_index) in enumerate(folds.split(train, y, groups=train['breath_id'])):
    print(f'Fold {fold_n} started at {time.ctime()}')
    X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    model = lgb.LGBMRegressor(**params, n_estimators=6000) # 8000
    model.fit(X_train, y_train, 
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=100, early_stopping_rounds=15)
    score = metrics.mean_absolute_error(y_valid, model.predict(X_valid))
    
    models.append(model)
    scores.append(score)

    fold_importance = pd.DataFrame()
    fold_importance["feature"] = columns
    fold_importance["importance"] = model.feature_importances_
    fold_importance["fold"] = fold_n + 1
    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
    
print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))


# In[14]:


feature_importance


# In[15]:


for model in models:
    submission['pressure'] += model.predict(test[columns])
submission['pressure'] /= n_splits

submission.to_csv('first_sub.csv', index=False)


# In[ ]:




