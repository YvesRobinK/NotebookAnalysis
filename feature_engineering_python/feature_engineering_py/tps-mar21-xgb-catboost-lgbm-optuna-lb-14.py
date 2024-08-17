#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Hey, thanks for viewing my Kernel!
# 
# If you like my work, please, leave an upvote: it will be really appreciated and it will motivate me in offering more content to the Kaggle community ! 😊
# 
# Data preparation was done in this [notebook](https://www.kaggle.com/code/hasanbasriakcay/tps-mar21-eda-feature-engineering#Preparation-Data-For-Training).

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings


warnings.simplefilter("ignore")


# In[2]:


X = pd.read_csv('../input/tps-mar21-eda-feature-engineering/x_train.csv')
X_test = pd.read_csv('../input/tps-mar21-eda-feature-engineering/x_test.csv')
y = pd.read_csv('../input/tps-mar21-eda-feature-engineering/y_train.csv')

print('X shape: ', X.shape)
print('X_test shape: ', X_test.shape)
print('y shape: ', y.shape)
display(X.head())

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=42)


# In[3]:


X.describe()


# # Model Creating and Evaluating

# In[4]:


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

def scorer(y, y_pred):
    return roc_auc_score(y, y_pred)


# In[5]:


# XGBClassifier
xgbc_model = XGBClassifier(min_child_weight=0.1, reg_lambda=100, booster='gbtree', objective='binary:logitraw', random_state=42)
xgbc_score = cross_val_score(xgbc_model, train_X, train_y, scoring='roc_auc', cv=5)
print('xgbc_score: ', xgbc_score.mean())

# LGBMClassifier
ligthgbmc_model = LGBMClassifier(boosting_type='gbdt', objective='binary', random_state=42)
ligthgbmc_score = cross_val_score(ligthgbmc_model, train_X, train_y, scoring='roc_auc', cv=5)
print('ligthgbmc_score: ', ligthgbmc_score.mean())

# CatBoostClassifier
cbc_model = CatBoostClassifier(loss_function='Logloss', random_state=42, verbose=False)
cbc_score = cross_val_score(cbc_model, train_X, train_y, scoring='roc_auc', cv=5)
print('cbc_score: ', cbc_score.mean())


# # XGB Optuna

# In[6]:


def objective(trial, data=X, target=y):
    X_train, X_val, y_train, y_val = train_test_split(data, target, test_size=0.2, random_state=42)

    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 32),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.005, 0.02, 0.05, 0.08, 0.1]),
        'n_estimators': trial.suggest_int('n_estimators', 2000, 8000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        'gamma': trial.suggest_float('gamma', 0.0001, 1.0, log = True),
        'alpha': trial.suggest_float('alpha', 0.0001, 10.0, log = True),
        'lambda': trial.suggest_float('lambda', 0.0001, 10.0, log = True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.8),
        'subsample': trial.suggest_float('subsample', 0.1, 0.8),
        'tree_method': 'gpu_hist',
        'booster': 'gbtree',
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'auc'

    }
    
    model = XGBClassifier(**params)  
    model.fit(X_train, y_train, eval_set = [(X_val,y_val)], early_stopping_rounds = 333, verbose = False)
    y_pred = model.predict_proba(X_val)[:,1]
    roc_auc = roc_auc_score(y_val, y_pred)

    return roc_auc


# In[7]:


#study = optuna.create_study(direction='maximize')
#study.optimize(objective, n_trials=50)
#print('Number of finished trials: ', len(study.trials))
#print('Best trial: ', study.best_trial.params)
#print('Best value: ', study.best_value)


# Number of finished trials: 1 Best trial: {'max_depth': 4, 'learning_rate': 0.1, 'n_estimators': 2616, 'min_child_weight': 36, 'gamma': 0.0001231342905079067, 'alpha': 5.138826788428377, 'lambda': 0.006952601632723477, 'colsample_bytree': 0.3019243613187322, 'subsample': 0.7474126793277557} Best value: 0.8941200673933261
# 
# Number of finished trials: 50 Best trial: {'max_depth': 6, 'learning_rate': 0.02, 'n_estimators': 2941, 'min_child_weight': 10, 'gamma': 0.027689264382343946, 'alpha': 2.239319562015662, 'lambda': 0.005116156806904708, 'colsample_bytree': 0.2018103901998171, 'subsample': 0.7452030806282816} Best value: 0.8951492161710065

# # CatBoost Optuna

# In[8]:


def objective(trial, data=X, target=y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 64),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.005, 0.02, 0.05, 0.08, 0.1]),
        'n_estimators': trial.suggest_int('n_estimators', 2000, 8000),
        'max_bin': trial.suggest_int('max_bin', 200, 400),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 300),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.0001, 1.0, log = True),
        'subsample': trial.suggest_float('subsample', 0.1, 0.8),
        'random_seed': 42,
        'task_type': 'GPU',
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'bootstrap_type': 'Poisson'
    }
    
    model = CatBoostClassifier(**params)  
    model.fit(X_train, y_train, eval_set = [(X_val,y_val)], early_stopping_rounds = 222, verbose = False)
    y_pred = model.predict_proba(X_val)[:,1]
    roc_auc = roc_auc_score(y_val, y_pred)

    return roc_auc


# In[9]:


#study = optuna.create_study(direction = 'maximize')
#study.optimize(objective, n_trials = 50)
#print('Number of finished trials:', len(study.trials))
#print('Best trial:', study.best_trial.params)
#print('Best value:', study.best_value)


# Number of finished trials: 50 Best trial: {'max_depth': 4, 'learning_rate': 0.1, 'n_estimators': 2877, 'max_bin': 200, 'min_data_in_leaf': 10, 'l2_leaf_reg': 0.09385107162927438, 'subsample': 0.7990428819543426} Best value: 0.8925910141177894

# # LGBM Optuna

# In[10]:


def objective(trial,data=X,target=y):   
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.15,random_state=42)
    params = {
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 11, 333),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 64),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.02, 0.05, 0.005, 0.1]),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.5),
        'n_estimators': trial.suggest_int('n_estimators', 2000, 8000),
        'cat_smooth' : trial.suggest_int('cat_smooth', 10, 100),
        'cat_l2': trial.suggest_int('cat_l2', 1, 20),
        'min_data_per_group': trial.suggest_int('min_data_per_group', 50, 200),
        'cat_feature' : [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 
                         32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 
                         53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67],
        'n_jobs' : -1, 
        'random_state': 42,
        'boosting_type': 'gbdt',
        'metric': 'AUC',
        'device': 'gpu'
    }
    model = LGBMClassifier(**params)  
    
    model.fit(train_x,train_y,eval_set=[(test_x,test_y)],eval_metric='auc', early_stopping_rounds=300, verbose=False)
    
    preds = model.predict_proba(test_x)[:,1]
    
    auc = roc_auc_score(test_y, preds)
    
    return auc


# In[11]:


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print('Number of finished trials: ', len(study.trials))
print('Best trial: ', study.best_trial.params)
print('Best value: ', study.best_value)


# Trial 32 finished with value: 0.8960118885713237 and parameters: {'reg_alpha': 5.028382776465415, 'reg_lambda': 7.969115943661513, 'num_leaves': 196, 'min_child_samples': 39, 'max_depth': 20, 'learning_rate': 0.01, 'colsample_bytree': 0.22772406492167746, 'n_estimators': 7028, 'cat_smooth': 38, 'cat_l2': 20, 'min_data_per_group': 199}. Best is trial 32 with value: 0.8960118885713237.
# 
# Number of finished trials: 50 Best trial: {'reg_alpha': 5.028382776465415, 'reg_lambda': 7.969115943661513, 'num_leaves': 196, 'min_child_samples': 39, 'max_depth': 20, 'learning_rate': 0.01, 'colsample_bytree': 0.22772406492167746, 'n_estimators': 7028, 'cat_smooth': 38, 'cat_l2': 20, 'min_data_per_group': 199}. Best is trial 32 with value: 0.8960118885713237.

# In[12]:


# Historic
plot_optimization_history(study)


# In[13]:


# Importance
optuna.visualization.plot_param_importances(study)


# In[14]:


lgb_params =  {'reg_alpha': 5.028382776465415, 
               'reg_lambda': 7.969115943661513, 
               'num_leaves': 196, 
               'min_child_samples': 39, 
               'max_depth': 20, 
               'learning_rate': 0.01, 
               'colsample_bytree': 0.22772406492167746, 
               'n_estimators': 7028, 
               'cat_smooth': 38, 
               'cat_l2': 20, 
               'min_data_per_group': 199,
               'cat_feature' : [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 
                                 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 
                                 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67],
               'n_jobs' : -1, 
               'random_state': 42,
               'boosting_type': 'gbdt',
               'metric': 'AUC',
               'device': 'gpu'
}


# In[15]:


lgb_params = study.best_trial.params
lgb_params['device'] = "gpu"
lgb_params['random_state'] = 42
lgb_params['cat_feature'] = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 
                             32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 
                             53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
lgb_params['n_jobs'] = -1 
lgb_params['boosting_type'] =  'gbdt'
lgb_params['metric'] =  'AUC'


# In[16]:


NFOLDS = 20
folds = StratifiedKFold(n_splits=NFOLDS, random_state=42, shuffle=True)
predictions = np.zeros(len(X_test))
for fold, (train_index, test_index) in enumerate(folds.split(X, y)):
    print("--> Fold {}".format(fold + 1))
    
    X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    
    lgb_model = LGBMClassifier(**lgb_params).fit(X_train, y_train, 
                                                  eval_set=[(X_valid, y_valid)], 
                                                  eval_metric='auc', 
                                                  early_stopping_rounds=300, verbose=0)
    
    y_preds = lgb_model.predict_proba(X_valid)[:,1]
    predictions += lgb_model.predict_proba(X_test)[:,1] / folds.n_splits 
    
    print(": LGB - ROC AUC Score = {}".format(roc_auc_score(y_valid, y_preds, average="micro")))


# In[17]:


sub = pd.read_csv('../input/tabular-playground-series-mar-2021/sample_submission.csv')
sub['target'] = predictions
regr_table = sub.to_csv("Sub_lgb_v2.csv", index = False)

