#!/usr/bin/env python
# coding: utf-8

# <a class="anchor" id="0"></a>
# # [Google Cloud & NCAA® ML Competition 2020-NCAAM](https://www.kaggle.com/c/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament)

# ## In this notebook, I'm just training some models (LGB, XGB etc.).
# 
# ## Commits from 33-th don't contain leaks for 2015-2019 and are trained only by data from 1985-2014.

# # My older commits:
# ## The organizers warned that solutions with LB < 0.20 will be deleted, so I have a solution (with all data) with LB=0.20125.
# 
# ## Experienced participants in previous such competitions suggest that the best solution should be around 0.5, so I have a solution (commit 16) with LB = 0.49.
# 
# 

# # Acknowledgements
# 
# This kernel uses such good kernels: 
# * [Merging FE & Prediction - xgb, lgb, logr, linr](https://www.kaggle.com/vbmokin/merging-fe-prediction-xgb-lgb-logr-linr)
# * [Basic Starter Kernel](https://www.kaggle.com/addisonhoward/basic-starter-kernel-ncaa-men-s-dataset-2019)
# * [2020 Basic Starter Kernel](https://www.kaggle.com/hiromoon166/2020-basic-starter-kernel)
# * [March Madness 2020 NCAAM EDA and baseline](https://www.kaggle.com/artgor/march-madness-2020-ncaam-eda-and-baseline)
# * [March Madness 2020 NCAAM:Simple Lightgbm on KFold](https://www.kaggle.com/ratan123/march-madness-2020-ncaam-simple-lightgbm-on-kfold)
# * [NCAAM2020: XGBoost + LightGBM K-Fold (Baseline)](https://www.kaggle.com/khoongweihao/ncaam2020-xgboost-lightgbm-k-fold-baseline)

# <a class="anchor" id="0.1"></a>
# ## Table of Contents
# 
# 1. [My upgrade](#1)
#     -  [Commit now](#1.1)
#     -  [Previous commits: LGB (1985-2014)](#1.2)
#     -  [Previous commits: LGB (all data)](#1.3)
#     -  [Previous commits: Merging of LGB, XGB, Logistic Regression (all data)](#1.4)
# 1. [Import libraries](#2)
# 1. [Download data & FE](#3)
# 1. [Models tuning](#4)
#     -  [LGB](#4.1)
#     -  [XGB](#4.2)    
#     -  [Logistic Regression](#4.3)
# 1. [Showing Confusion Matrices](#5)
# 1. [Comparison and merging solutions](#6)
# 1. [Submission](#7)

# ## 1. My upgrade<a class="anchor" id="1"></a>
# 
# [Back to Table of Contents](#0.1)

# ## 1.1. Commit now <a class="anchor" id="1.1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[1]:


# LGB
lgb_num_leaves_max = 255
lgb_in_leaf = 50
lgb_lr = 0.0001
lgb_bagging = 7

# XGB
xgb_max_depth = 20
xgb_min_child_weight = 75
xgb_lr = 0.0005
xgb_num_boost_round_max = 4000
# without early_stopping_rounds

# Set weight of models
w_lgb = 0.6
w_xgb = 0.3
w_logreg = 1 - w_lgb - w_xgb
w_logreg


# ## 1.2. Previous commits: LGB (1985-2014) <a class="anchor" id="1.2"></a>
# 
# [Back to Table of Contents](#0.1)

# ### Commit 32 (as Commit 14 for all data)
# 
# w_lgb = 1
# 
#     params = {'num_leaves': 127,
#               'min_data_in_leaf': 10,
#               'objective': 'binary',
#               'max_depth': -1,
#               'learning_rate': 0.001,
#               "boosting_type": "gbdt",
#               "bagging_seed": 7,
#               "metric": 'logloss',
#               "verbosity": -1,
#               'random_state': 42,
#              }
#              
# **LB = 0.64267**

# ### Commit 34
# 
# * LGB
# * lgb_num_leaves_max = 127
# * lgb_in_leaf = 10
# * lgb_lr = 0.001
# * lgb_bagging = 7
# * 
# * XGB
# * xgb_max_depth = 35
# * xgb_min_child_weight = 75
# * xgb_lr = 0.0004
# * xgb_n_estimators = 4000
# * 
# * w_lgb = 0.55
# * w_xgb = 0.35
# * x_logreg = 0.1
# 
# **LB = 0.56026**

# ### Commit 38
# 
# * LGB
# * lgb_num_leaves_max = 127
# * lgb_in_leaf = 10
# * lgb_lr = 0.001
# * lgb_bagging = 7
# * 
# * XGB
# * xgb_max_depth = 35
# * xgb_min_child_weight = 75
# * xgb_lr = 0.0004
# * xgb_n_estimators = 4000
# * 
# * w_lgb = 0.5
# * w_xgb = 0.3
# * x_logreg = 0.2
# 
# **LB = 0.57203**

# ### Commit 39
# 
# * LGB
# * lgb_num_leaves_max = 200
# * lgb_in_leaf = 10
# * lgb_lr = 0.01
# * lgb_bagging = 7
# * 
# * XGB
# * xgb_max_depth = 7
# * xgb_min_child_weight = 10
# * xgb_lr = 0.01
# * xgb_n_estimators = 2000
# * 
# * w_lgb = 0.55
# * w_xgb = 0.35
# * x_logreg = 0.1
# 
# **LB = 0.60422**

# ### Commit 40
# 
# * LGB
# * lgb_num_leaves_max = 200
# * lgb_in_leaf = 10
# * lgb_lr = 0.01
# * lgb_bagging = 7
# * 
# * XGB
# * xgb_max_depth = 7
# * xgb_min_child_weight = 10
# * xgb_lr = 0.01
# * xgb_n_estimators = 2000
# * 
# * w_lgb = 0.8
# * w_xgb = 0.1
# * x_logreg = 0.1
# 
# **LB = 0.73089**

# ### Commit 42
# 
# * LGB
# * lgb_num_leaves_max = 200
# * lgb_in_leaf = 10
# * lgb_lr = 0.01
# * lgb_bagging = 7
# * 
# * XGB
# * xgb_max_depth = 7
# * xgb_min_child_weight = 10
# * xgb_lr = 0.001
# * xgb_n_estimators = 4000
# * 
# * w_lgb = 0.8
# * w_xgb = 0.1
# * x_logreg = 0.1
# 
# **LB = 0.74658**

# ### Commit 43
# 
# * LGB
# * lgb_num_leaves_max = 200
# * lgb_in_leaf = 10
# * lgb_lr = 0.001
# * lgb_bagging = 7
# * 
# * XGB
# * xgb_max_depth = 7
# * xgb_min_child_weight = 10
# * xgb_lr = 0.0001
# * xgb_n_estimators = 4000
# * 
# * w_lgb = 0.8
# * w_xgb = 0.1
# * x_logreg = 0.1
# 
# **LB = 0.61163**

# ### Commit 44
# 
# * LGB
# * lgb_num_leaves_max = 255
# * lgb_in_leaf = 50
# * lgb_lr = 0.001
# * lgb_bagging = 7
# * 
# * XGB
# * xgb_max_depth = 20
# * xgb_min_child_weight = 75
# * xgb_lr = 0.0004
# * xgb_n_estimators = 4000
# * 
# * w_lgb = 0.6
# * w_xgb = 0.3
# * x_logreg = 0.1
# 
# **LB = 0.55582**

# ### Commit 45
# 
# * LGB
# * lgb_num_leaves_max = 300
# * lgb_in_leaf = 20
# * lgb_lr = 0.001
# * lgb_bagging = 7
# * 
# * XGB
# * xgb_max_depth = 20
# * xgb_min_child_weight = 50
# * xgb_lr = 0.0004
# * xgb_n_estimators = 4000
# * 
# * w_lgb = 0.6
# * w_xgb = 0.3
# * x_logreg = 0.1
# 
# **LB = 0.56801**

# ### Commit 46
# 
# * LGB
# * lgb_num_leaves_max = 255
# * lgb_in_leaf = 50
# * lgb_lr = 0.0001
# * lgb_bagging = 7
# * 
# * XGB
# * xgb_max_depth = 20
# * xgb_min_child_weight = 75
# * xgb_lr = 0.0004
# * xgb_n_estimators = 4000
# * 
# * w_lgb = 0.6
# * w_xgb = 0.3
# * x_logreg = 0.1
# 
# **LB = 0.56177**

# ## 1.3. Previous commits: LGB (all data) <a class="anchor" id="1.3"></a>
# 
# [Back to Table of Contents](#0.1)

# ### Commit 1
# 
#     params = {'num_leaves': 255,
#               'min_data_in_leaf': 100,
#               'objective': 'binary',
#               'max_depth': -1,
#               'learning_rate': 0.005,
#               "boosting_type": "gbdt",
#               "bagging_seed": 11,
#               "metric": 'logloss',
#               "verbosity": -1,
#               'random_state': 42,
#              }
#              
# **LB = 0.29445**

# ### Commit 2
# 
#     params = {'num_leaves': 127,
#               'min_data_in_leaf': 100,
#               'objective': 'binary',
#               'max_depth': -1,
#               'learning_rate': 0.005,
#               "boosting_type": "gbdt",
#               "bagging_seed": 11,
#               "metric": 'logloss',
#               "verbosity": -1,
#               'random_state': 42,
#              }
#              
# **LB = 0.29445**

# ### Commit 6
# 
#     params = {'num_leaves': 127,
#               'min_data_in_leaf': 50,
#               'objective': 'binary',
#               'max_depth': -1,
#               'learning_rate': 0.005,
#               "boosting_type": "gbdt",
#               "bagging_seed": 11,
#               "metric": 'logloss',
#               "verbosity": -1,
#               'random_state': 42,
#              }
#              
# **LB = 0.14726**

# ### Commit 8
# 
#     params = {'num_leaves': 127,
#               'min_data_in_leaf': 10,
#               'objective': 'binary',
#               'max_depth': -1,
#               'learning_rate': 0.005,
#               "boosting_type": "gbdt",
#               "bagging_seed": 11,
#               "metric": 'logloss',
#               "verbosity": -1,
#               'random_state': 42,
#              }
#              
# **LB = 0.05038**

# ### Commit 9
# 
#     params = {'num_leaves': 63,
#               'min_data_in_leaf': 10,
#               'objective': 'binary',
#               'max_depth': -1,
#               'learning_rate': 0.005,
#               "boosting_type": "gbdt",
#               "bagging_seed": 11,
#               "metric": 'logloss',
#               "verbosity": -1,
#               'random_state': 42,
#              }
#              
# **LB = 0.08311**

# ### Commit 13
# 
#     params = {'num_leaves': 127,
#               'min_data_in_leaf': 10,
#               'objective': 'binary',
#               'max_depth': -1,
#               'learning_rate': 0.01,
#               "boosting_type": "gbdt",
#               "bagging_seed": 11,
#               "metric": 'logloss',
#               "verbosity": -1,
#               'random_state': 42,
#              }
#              
# **LB = 0.04872**

# ### Commit 14
# 
#     params = {'num_leaves': 127,
#               'min_data_in_leaf': 10,
#               'objective': 'binary',
#               'max_depth': -1,
#               'learning_rate': 0.01,
#               "boosting_type": "gbdt",
#               "bagging_seed": 7,
#               "metric": 'logloss',
#               "verbosity": -1,
#               'random_state': 42,
#              }
#              
# **LB = 0.04872**

# ### Commit 15
# 
#     params = {'num_leaves': 127,
#               'min_data_in_leaf': 10,
#               'objective': 'binary',
#               'max_depth': -1,
#               'learning_rate': 0.001,
#               "boosting_type": "gbdt",
#               "bagging_seed": 7,
#               "metric": 'logloss',
#               "verbosity": -1,
#               'random_state': 42,
#              }
#              
# **LB = 0.17908**

# ## 1.4. Previous commits: Merging of LGB, XGB, Logistic Regression (all data) <a class="anchor" id="1.4"></a>
# 
# [Back to Table of Contents](#0.1)

# ### Commit 20
# 
# **LGB**
# 
#     params_lgb = {'num_leaves': 127,
#                   'min_data_in_leaf': 10,
#                   'objective': 'binary',
#                   'max_depth': -1,
#                   'learning_rate': 0.0001,
#                   "boosting_type": "gbdt",
#                   "bagging_seed": 7,
#                   "metric": 'logloss',
#                   "verbosity": -1,
#                   'random_state': 42,
#                  }
# 
# **XGB**
# 
#     params_xgb = {'max_depth':63,
#                   'objective':'binary:logistic',
#                   'min_child_weight': 10,
#                   'learning_rate': 0.0001,
#                   'eta'      :0.3,
#                   'subsample':0.8,
#                   'lambda '  :4,
#                   'eval_metric':'logloss',
#                   'n_estimators':2000,
#                   'colsample_bytree ':0.9,
#                   'colsample_bylevel':1
#                   }
#                   
#     w_lgb = 0.48
#     w_xgb = 0.48
#     w_logreg = 1 - w_lgb - w_xgb
#     
#     y_preds = w_lgb*y_preds_lgb + w_xgb*y_preds_xgb + w_logreg*y_logreg_pred
#     
# 
# **LB = 0.34462**

# ### Commit 22
# 
# **LGB**
# 
#     params_lgb = {'num_leaves': 63,
#                   'min_data_in_leaf': 10,
#                   'objective': 'binary',
#                   'max_depth': -1,
#                   'learning_rate': 0.005,
#                   "boosting_type": "gbdt",
#                   "bagging_seed": 7,
#                   "metric": 'logloss',
#                   "verbosity": -1,
#                   'random_state': 42,
#                  }
# 
# **XGB**
# 
#     params_xgb = {'max_depth':63,
#                   'objective':'binary:logistic',
#                   'min_child_weight': 10,
#                   'learning_rate': 0.005,
#                   'eta'      :0.3,
#                   'subsample':0.8,
#                   'lambda '  :4,
#                   'eval_metric':'logloss',
#                   'n_estimators':2000,
#                   'colsample_bytree ':0.9,
#                   'colsample_bylevel':1
#                   }
#                   
#     w_lgb = 0.48
#     w_xgb = 0.48
#     w_logreg = 1 - w_lgb - w_xgb
#     
#     y_preds = w_lgb*y_preds_lgb + w_xgb*y_preds_xgb + w_logreg*y_logreg_pred
#     
# 
# **LB = 0.27677**

# ### Commit 27
# 
# #### LGB
# * lgb_num_leaves_max = 100
# * lgb_in_leaf = 75
# * lgb_lr = 0.007
# * lgb_bagging = 7
# 
# #### XGB
# * xgb_max_depth = 40
# * xgb_min_child_weight = 75
# * xgb_lr = 0.0004
# * xgb_n_estimators = 4000
# 
# #### Set weight of models
# * w_lgb = 0.8
# * w_xgb = 0.15
# * w_logreg = 1 - w_lgb - w_xgb    
# 
# **LB = 0.24733**

# ## 2. Import libraries <a class="anchor" id="2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import eli5

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
import xgboost as xgb

import gc

import warnings
warnings.filterwarnings("ignore")


# ## 3. Download data & FE <a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[3]:


tourney_result = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')
tourney_result = tourney_result[tourney_result['Season'] < 2015]
tourney_seed = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneySeeds.csv')
tourney_seed = tourney_seed[tourney_seed['Season'] < 2015]
tourney_result = tourney_result.drop(['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], axis=1)
tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
tourney_result.rename(columns={'Seed':'WSeed'}, inplace=True)
tourney_result = tourney_result.drop('TeamID', axis=1)
tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
tourney_result.rename(columns={'Seed':'LSeed'}, inplace=True)
tourney_result = tourney_result.drop('TeamID', axis=1)


# In[4]:


tourney_seed


# In[5]:


def get_seed(x):
    return int(x[1:3])

tourney_result['WSeed'] = tourney_result['WSeed'].map(lambda x: get_seed(x))
tourney_result['LSeed'] = tourney_result['LSeed'].map(lambda x: get_seed(x))


# In[6]:


season_result = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')
season_result = season_result[season_result['Season'] < 2015]
season_win_result = season_result[['Season', 'WTeamID', 'WScore']]
season_lose_result = season_result[['Season', 'LTeamID', 'LScore']]
season_win_result.rename(columns={'WTeamID':'TeamID', 'WScore':'Score'}, inplace=True)
season_lose_result.rename(columns={'LTeamID':'TeamID', 'LScore':'Score'}, inplace=True)
season_result = pd.concat((season_win_result, season_lose_result)).reset_index(drop=True)
season_score = season_result.groupby(['Season', 'TeamID'])['Score'].sum().reset_index()


# In[7]:


tourney_result = pd.merge(tourney_result, season_score, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
tourney_result.rename(columns={'Score':'WScoreT'}, inplace=True)
tourney_result = tourney_result.drop('TeamID', axis=1)
tourney_result = pd.merge(tourney_result, season_score, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
tourney_result.rename(columns={'Score':'LScoreT'}, inplace=True)
tourney_result = tourney_result.drop('TeamID', axis=1)


# In[8]:


tourney_win_result = tourney_result.drop(['Season', 'WTeamID', 'LTeamID'], axis=1)
tourney_win_result.rename(columns={'WSeed':'Seed1', 'LSeed':'Seed2', 'WScoreT':'ScoreT1', 'LScoreT':'ScoreT2'}, inplace=True)


# In[9]:


tourney_lose_result = tourney_win_result.copy()
tourney_lose_result['Seed1'] = tourney_win_result['Seed2']
tourney_lose_result['Seed2'] = tourney_win_result['Seed1']
tourney_lose_result['ScoreT1'] = tourney_win_result['ScoreT2']
tourney_lose_result['ScoreT2'] = tourney_win_result['ScoreT1']


# ## Prepare Training Data

# In[10]:


tourney_win_result['Seed_diff'] = tourney_win_result['Seed1'] - tourney_win_result['Seed2']
tourney_win_result['ScoreT_diff'] = tourney_win_result['ScoreT1'] - tourney_win_result['ScoreT2']
tourney_lose_result['Seed_diff'] = tourney_lose_result['Seed1'] - tourney_lose_result['Seed2']
tourney_lose_result['ScoreT_diff'] = tourney_lose_result['ScoreT1'] - tourney_lose_result['ScoreT2']


# In[11]:


tourney_win_result['result'] = 1
tourney_lose_result['result'] = 0
train_df = pd.concat((tourney_win_result, tourney_lose_result)).reset_index(drop=True)
train_df


# In[12]:


season_result


# In[13]:


tourney_result


# # Preparing testing data

# In[14]:


test_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')
sub = test_df.copy()


# In[15]:


test_df


# In[16]:


test_df['Season'] = test_df['ID'].map(lambda x: int(x[:4]))
test_df['WTeamID'] = test_df['ID'].map(lambda x: int(x[5:9]))
test_df['LTeamID'] = test_df['ID'].map(lambda x: int(x[10:14]))


# In[17]:


tourney_seed = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneySeeds.csv')
tourney_seed = tourney_seed[tourney_seed['Season'] > 2014]


# In[18]:


season_result = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')
season_result = season_result[season_result['Season'] > 2014]
season_win_result = season_result[['Season', 'WTeamID', 'WScore']]
season_lose_result = season_result[['Season', 'LTeamID', 'LScore']]
season_win_result.rename(columns={'WTeamID':'TeamID', 'WScore':'Score'}, inplace=True)
season_lose_result.rename(columns={'LTeamID':'TeamID', 'LScore':'Score'}, inplace=True)
season_result = pd.concat((season_win_result, season_lose_result)).reset_index(drop=True)
season_score = season_result.groupby(['Season', 'TeamID'])['Score'].sum().reset_index()


# In[19]:


test_df = pd.merge(test_df, tourney_seed, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Seed':'Seed1'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)
test_df = pd.merge(test_df, tourney_seed, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Seed':'Seed2'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)
test_df = pd.merge(test_df, season_score, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Score':'ScoreT1'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)
test_df = pd.merge(test_df, season_score, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Score':'ScoreT2'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)


# In[20]:


test_df['Seed1'] = test_df['Seed1'].map(lambda x: get_seed(x))
test_df['Seed2'] = test_df['Seed2'].map(lambda x: get_seed(x))
test_df['Seed_diff'] = test_df['Seed1'] - test_df['Seed2']
test_df['ScoreT_diff'] = test_df['ScoreT1'] - test_df['ScoreT2']
test_df = test_df.drop(['ID', 'Pred', 'Season', 'WTeamID', 'LTeamID'], axis=1)
test_df


# ## 4. Model tuning <a class="anchor" id="4"></a>
# 
# [Back to Table of Contents](#0.1)

# In[21]:


X = train_df.drop('result', axis=1)
y = train_df.result


# ### 4.1 LGB <a class="anchor" id="4.1"></a>
# 
# [Back to Table of Contents](#0.1)

# Thanks to:
# * [March Madness 2020 NCAAM EDA and baseline](https://www.kaggle.com/artgor/march-madness-2020-ncaam-eda-and-baseline)
# * [March Madness 2020 NCAAM:Simple Lightgbm on KFold](https://www.kaggle.com/ratan123/march-madness-2020-ncaam-simple-lightgbm-on-kfold)

# In[22]:


params_lgb = {'num_leaves': lgb_num_leaves_max,
              'min_data_in_leaf': lgb_in_leaf,
              'objective': 'binary',
              'max_depth': -1,
              'learning_rate': lgb_lr,
              "boosting_type": "gbdt",
              "bagging_seed": lgb_bagging,
              "metric": 'logloss',
              "verbosity": -1,
              'random_state': 42,
             }


# In[23]:


NFOLDS = 10
folds = KFold(n_splits=NFOLDS)

columns = X.columns
splits = folds.split(X, y)
y_preds_lgb = np.zeros(test_df.shape[0])
y_train_lgb = np.zeros(X.shape[0])
y_oof = np.zeros(X.shape[0])

feature_importances = pd.DataFrame()
feature_importances['feature'] = columns
  
for fold_n, (train_index, valid_index) in enumerate(splits):
    print('Fold:',fold_n+1)
    X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    clf = lgb.train(params_lgb, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200)
    
    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()
    
    y_pred_valid = clf.predict(X_valid)
    y_oof[valid_index] = y_pred_valid
    
    y_train_lgb += clf.predict(X) / NFOLDS
    y_preds_lgb += clf.predict(test_df) / NFOLDS
    
    del X_train, X_valid, y_train, y_valid
    gc.collect()


# ### 4.2 XGB <a class="anchor" id="4.2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[24]:


params_xgb = {'max_depth': xgb_max_depth,
              'objective': 'binary:logistic',
              'min_child_weight': xgb_min_child_weight,
              'learning_rate': xgb_lr,
              'eta'      : 0.3,
              'subsample': 0.8,
              'lambda '  : 4,
              'eval_metric': 'logloss',
              'colsample_bytree ': 0.9,
              'colsample_bylevel': 1
              }


# In[25]:


X


# In[26]:


y


# In[27]:


# Thanks to https://www.kaggle.com/khoongweihao/ncaam2020-xgboost-lightgbm-k-fold-baseline
NFOLDS = 10
folds = KFold(n_splits=NFOLDS)

columns = X.columns
splits = folds.split(X, y)

y_preds_xgb = np.zeros(test_df.shape[0])
y_train_xgb = np.zeros(X.shape[0])
y_oof_xgb = np.zeros(X.shape[0])

train_df_set = xgb.DMatrix(X)
test_set = xgb.DMatrix(test_df)
  
for fold_n, (train_index, valid_index) in enumerate(splits):
    print('Fold:',fold_n+1)
    X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    train_set = xgb.DMatrix(X_train, y_train)
    val_set = xgb.DMatrix(X_valid, y_valid)
    
    clf = xgb.train(params_xgb, train_set, num_boost_round=xgb_num_boost_round_max, evals=[(train_set, 'train'), (val_set, 'val')], verbose_eval=100)
    
    y_train_xgb += clf.predict(train_df_set) / NFOLDS
    y_preds_xgb += clf.predict(test_set) / NFOLDS
    
    del X_train, X_valid, y_train, y_valid
    gc.collect()


# ### 4.3 Logistic Regression <a class="anchor" id="4.3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[28]:


test_df.head()


# In[29]:


get_ipython().run_cell_magic('time', '', '# Standardization for regression models\nscaler = StandardScaler()\ntrain_log = pd.DataFrame(\n    scaler.fit_transform(X),\n    columns=X.columns,\n    index=X.index\n)\ntest_log = pd.DataFrame(\n    scaler.transform(test_df),\n    columns=test_df.columns,\n    index=test_df.index\n)\n')


# In[30]:


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(train_log, y)
coeff_logreg = pd.DataFrame(train_log.columns.delete(0))
coeff_logreg.columns = ['feature']
coeff_logreg["score_logreg"] = pd.Series(logreg.coef_[0])
coeff_logreg.sort_values(by='score_logreg', ascending=False)


# In[31]:


# Eli5 visualization
eli5.show_weights(logreg)


# In[32]:


y_logreg_train = logreg.predict(train_log)
y_logreg_pred = logreg.predict(test_log)


# ## 5. Showing Confusion Matrices <a class="anchor" id="5"></a>
# 
# [Back to Table of Contents](#0.1)

# In[33]:


# Showing Confusion Matrix
# Thanks to https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud
def plot_cm(y_true, y_pred, title, figsize=(7,6)):
    y_pred = y_pred.round().astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)


# In[34]:


# Showing Confusion Matrix for LGB model
plot_cm(y, y_train_lgb, 'Confusion matrix for LGB model')


# In[35]:


# Showing Confusion Matrix for XGB model
plot_cm(y, y_train_xgb, 'Confusion matrix for XGB model')


# In[36]:


# Showing Confusion Matrix for Logistic Regression
plot_cm(y, y_logreg_train, 'Confusion matrix for Logistic Regression')


# ## 6. Comparison and merging solutions <a class="anchor" id="6"></a>
# 
# [Back to Table of Contents](#0.1)

# ### Merging solutions

# In[37]:


y_preds = w_lgb*y_preds_lgb + w_xgb*y_preds_xgb + w_logreg*y_logreg_pred


# ### Confusion Matrix

# In[38]:


# Showing Confusion Matrix for Merging solution
y_train_preds = w_lgb*y_train_lgb + w_xgb*y_train_xgb + w_logreg*y_logreg_train
plot_cm(y, y_train_preds, 'Confusion matrix for Merging solution')


# ## 7. Submission <a class="anchor" id="7"></a>
# 
# [Back to Table of Contents](#0.1)

# In[39]:


sub['Pred'] = y_preds
sub.head()


# In[40]:


sub.info()


# In[41]:


sub['Pred'].hist()


# In[42]:


sub.to_csv('submission.csv', index=False)


# I hope you find this kernel useful and enjoyable.

# Your comments and feedback are most welcome.

# [Go to Top](#0)
