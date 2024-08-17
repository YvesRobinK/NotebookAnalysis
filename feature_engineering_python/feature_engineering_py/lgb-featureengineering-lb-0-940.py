#!/usr/bin/env python
# coding: utf-8

# #### Whats new in this version
# * Added code to train model for 5 folds and then ensemble.
# * added [data-without-drift](https://www.kaggle.com/cdeotte/data-without-drift)
# * removed old features as they were taking too much time
# * added new features mostly from [here](https://www.kaggle.com/nxrprime/wavenet-with-shifted-rfc-proba-and-cbr)
# 
# #### What's next
# * Try Ensemble using GroupKFold
# * Add few top features from old version 
# 

# In[1]:


import numpy as np 
import pandas as pd
from sklearn import *
from sklearn.metrics import f1_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import time
import xgboost as xgb
from catboost import Pool,CatBoostRegressor
import datetime
import gc

sns.set_style("whitegrid")

from sklearn.model_selection import KFold


# In[2]:


#Constants
MODEL_TYPE = 'lgb'  # available types 'lgb', 'xgb', 'cat'
TRAINING = True
ENSEMBLE = False
GROUP_BATCH_SIZE = 4000
WINDOWS = [10, 50]


BASE_PATH = '/kaggle/input/liverpool-ion-switching'
DATA_PATH = '/kaggle/input/data-without-drift'
RFC_DATA_PATH = '/kaggle/input/ion-shifted-rfc-proba'
MODELS_PATH = '/kaggle/input/ensemble-models'


# # 1. Create folds

# In[3]:


# create folds

import pandas as pd
from sklearn import model_selection


df = pd.read_csv(f"{DATA_PATH}/train_clean.csv")
df = df.dropna().reset_index(drop=True)

df["kfold"] = -1

# df = df.sample(frac=1).reset_index(drop=True)

kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (trn_, val_) in enumerate(kf.split(X=df, y=df.open_channels.values)):
    print(len(trn_), len(val_))
    df.loc[val_, 'kfold'] = fold

df.to_csv("train_folds.csv", index=False)


# ## 2. Create Features

# In[4]:


get_ipython().run_cell_magic('time', '', '\ndef create_rolling_features(df):\n    for window in WINDOWS:\n        df["rolling_mean_" + str(window)] = df[\'signal\'].rolling(window=window).mean()\n        df["rolling_std_" + str(window)] = df[\'signal\'].rolling(window=window).std()\n        df["rolling_var_" + str(window)] = df[\'signal\'].rolling(window=window).var()\n        df["rolling_min_" + str(window)] = df[\'signal\'].rolling(window=window).min()\n        df["rolling_max_" + str(window)] = df[\'signal\'].rolling(window=window).max()\n        df["rolling_min_max_ratio_" + str(window)] = df["rolling_min_" + str(window)] / df["rolling_max_" + str(window)]\n        df["rolling_min_max_diff_" + str(window)] = df["rolling_max_" + str(window)] - df["rolling_min_" + str(window)]\n\n    df = df.replace([np.inf, -np.inf], np.nan)    \n    df.fillna(0, inplace=True)\n    return df\n\n\ndef create_features(df, batch_size):\n    \n    df[\'group\'] = df.groupby(df.index//batch_size, sort=False)[\'signal\'].agg([\'ngroup\']).values\n    df[\'group\'] = df[\'group\'].astype(np.uint16)\n    for window in WINDOWS:    \n        df[\'signal_shift_pos_\' + str(window)] = df.groupby(\'group\')[\'signal\'].shift(window).fillna(0)\n        df[\'signal_shift_neg_\' + str(window)] = df.groupby(\'group\')[\'signal\'].shift(-1 * window).fillna(0)\n        \n    df[\'signal_2\'] = df[\'signal\'] ** 2\n    return df   \n')


# In[5]:


## reading data
train = pd.read_csv(f'train_folds.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
test  = pd.read_csv(f'{DATA_PATH}/test_clean.csv', dtype={'time': np.float32, 'signal': np.float32})
sub  = pd.read_csv(f'{BASE_PATH}/sample_submission.csv', dtype={'time': np.float32})

# loading and adding shifted-rfc-proba features
y_train_proba = np.load(f"{RFC_DATA_PATH}/Y_train_proba.npy")
y_test_proba = np.load(f"{RFC_DATA_PATH}/Y_test_proba.npy")

for i in range(11):
    train[f"proba_{i}"] = y_train_proba[:, i]
    test[f"proba_{i}"] = y_test_proba[:, i]

    
train = create_rolling_features(train)
test = create_rolling_features(test)   
    
## normalizing features
train_mean = train.signal.mean()
train_std = train.signal.std()
train['signal'] = (train.signal - train_mean) / train_std
test['signal'] = (test.signal - train_mean) / train_std


print('Shape of train is ',train.shape)
print('Shape of test is ',test.shape)


# In[6]:


## create features

batch_size = GROUP_BATCH_SIZE

train = create_features(train, batch_size)
test = create_features(test, batch_size)

cols_to_remove = ['time','signal','batch','batch_index','batch_slices','batch_slices2', 'group']
cols = [c for c in train.columns if c not in cols_to_remove]
cols_test = [c for c in test.columns if c not in cols_to_remove]

X = train[cols]
y = train['open_channels']
X_test = test[cols_test]


# In[7]:


del train
del test
gc.collect()


# <a id="id5"></a> <br> 
# # **3. Model**
# 

# In[8]:


def f1_score_calc(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")

def lgb_Metric(preds, dtrain):
    labels = dtrain.get_label()
    preds = np.round(np.clip(preds, 0, 10)).astype(int)
    score = f1_score(labels, preds, average="macro")
    return ('KaggleMetric', score, True)


# In[9]:


def train_model(X, X_test, params, model_type='lgb', eval_metric='f1score',
                               columns=None, plot_feature_importance=False, model=None,
                               verbose=50, early_stopping_rounds=200, n_estimators=2000):
    
    # to set up scoring parameters
    metrics_dict = {'f1score': {'lgb_metric_name': lgb_Metric,}}
    results = []    
    oof = np.zeros(len(X) )    
    prediction = np.zeros((len(X_test)))    
    feature_importance = pd.DataFrame()
            
    if True:        
        for fold in range(0,5):
            result_dict = {}
            X_train = X[X.kfold != fold].reset_index(drop=True)
            X_valid = X[X.kfold == fold].reset_index(drop=True)
            y_train = X_train.open_channels
            y_valid = X_valid.open_channels
            
            X_train = X_train.drop(['kfold', 'open_channels'], axis=1)
            X_valid = X_valid.drop(['kfold', 'open_channels'], axis=1)
            
#             X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, y, test_size=0.3, random_state=7)    

            if model_type == 'lgb':
                model = lgb.train(params, lgb.Dataset(X_train, y_train),
                                  n_estimators,  lgb.Dataset(X_valid, y_valid),
                                  verbose_eval=verbose, early_stopping_rounds=early_stopping_rounds, feval=lgb_Metric)

                preds = model.predict(X_valid, num_iteration=model.best_iteration) #model.predict(X_valid) 
                y_pred_valid = np.round(np.clip(preds, 0, 10)).astype(int)
                y_pred = model.predict(X_test, num_iteration=model.best_iteration)

            if model_type == 'xgb':
                train_set = xgb.DMatrix(X_train, y_train)
                val_set = xgb.DMatrix(X_valid, y_valid)
                model = xgb.train(params, train_set, num_boost_round=2222, evals=[(train_set, 'train'), (val_set, 'val')], 
                                         verbose_eval=verbose, early_stopping_rounds=early_stopping_rounds)

                preds = model.predict(xgb.DMatrix(X_valid)) #model.predict(X_valid) 
                y_pred_valid = np.round(np.clip(preds, 0, 10)).astype(int)
                y_pred = model.predict(xgb.DMatrix(X_test))


            print(f'FINAL score fold {fold}: {f1_score_calc(y_valid, y_pred_valid)}')
            print('*'*100)

            result_dict['pred_valid'] = preds
            result_dict['pred_test'] = y_pred
            result_dict['model'] = model
            results.append(result_dict)
    return results


# # 4. Training Model

# In[10]:


## training model

if TRAINING and MODEL_TYPE == 'lgb':
    params = {'learning_rate': 0.1, 'max_depth': 7, 'num_leaves':2**7+1, 'metric': 'rmse', 'random_state': 7, 'n_jobs':-1}

    result_dict_lgb = train_model(X=X, X_test=X_test, params=params, model_type=MODEL_TYPE, eval_metric='f1score', plot_feature_importance=False,
                                                          verbose=50, early_stopping_rounds=150, n_estimators=3000)


# In[11]:


if TRAINING and MODEL_TYPE == 'xgb':
    params_xgb = {'colsample_bytree': 0.375,'learning_rate': 0.1,'max_depth': 10, 'subsample': 1, 'objective':'reg:squarederror',
              'eval_metric':'rmse'}

    result_dict_xgb = train_model(X=X, X_test=X_test, params=params_xgb, model_type=MODEL_TYPE, eval_metric='f1score', plot_feature_importance=False,
                                                          verbose=50, early_stopping_rounds=250)


# <a id="id5"></a><br> 
# ## **5. Plotting feature importance top 20 Features**

# The model can score 0.940 on LB without further optimization. Maybe with some GroupKField Cross Validation, Ensamble or more features engineering, it's possible to get > 0.940

# In[12]:


if TRAINING and MODEL_TYPE == 'lgb':
    for fold in range(0,1):
        booster = result_dict_lgb[fold]['model']
        fi = pd.DataFrame()
        fi['importance'] = booster.feature_importance(importance_type='gain')
        fi['feature'] = booster.feature_name()
        best_features = fi.sort_values(by='importance', ascending=False)[:20]
        plt.figure(figsize=(16, 12));
        sns.barplot(x="importance", y="feature", data=best_features);
        plt.title('LGB Features fold {fold} (avg over folds)');


# ## 6. Saving models

# In[13]:


import joblib

if TRAINING:
    if MODEL_TYPE == 'lgb':
        for fold in range(0,5):
            filename = f'lgb_v{fold}.sav'
            joblib.dump(result_dict_lgb[fold]['model'], filename)
    elif MODEL_TYPE == 'xgb':
        for fold in range(0,5):
            filename = f'xgb_{fold}.sav'
            joblib.dump(result_dict_xgb[fold]['model'], filename)


# ## 7. Importing saved models and create submission

# In[14]:


def get_prediction(test, model, model_type):
    if model_type == 'xgb':
        y_pred = model.predict(xgb.DMatrix(test))
    else:
        y_pred = model.predict(test, num_iteration=model.best_iteration)
    y_pred = np.round(np.clip(y_pred, 0, 10)).astype(int)
    return y_pred


# In[15]:


if TRAINING:
    sub  = pd.read_csv(f'{BASE_PATH}/sample_submission.csv', dtype={'time': np.float32})    
    if MODEL_TYPE == 'lgb':
        preds = []
        for fold in range(0,5):
            preds.append(result_dict_lgb[fold][f'pred_test'])
        preds = np.average(preds, axis=0)
        y_pred = np.round(np.clip(preds, 0, 10)).astype(int)

        sub['open_channels'] =  np.array(np.round(y_pred,0), np.int)
    elif MODEL_TYPE == 'xgb':
        preds = []
        for fold in range(0, 5):
            preds.append(result_dict_xgb[fold][f'pred_test'])
        preds = np.average(preds, axis=0)
        y_pred = np.round(np.clip(preds, 0, 10)).astype(int)
        sub['open_channels'] =  np.array(np.round(y_pred,0), np.int)

    sub.to_csv('submission.csv', index=False, float_format='%.4f')
    print(sub.head(10))
else:
    if ENSEMBLE:
        model_lgb = joblib.load(MODELS_PATH + 'lgb_0.sav')
        model_xgb = joblib.load(MODELS_PATH + 'xgb_0.sav')
        y_pred_lgb = get_prediction(X_test, model_lgb, 'lgb')
        y_pred_xgb = get_prediction(X_test, model_xgb, 'xgb')
        y_pred = 0.50 * y_pred_lgb + 0.50 * y_pred_xgb
    else:
        if MODEL_TYPE == 'lgb':
            model_lgb = joblib.load(MODELS_PATH + 'lgb_0.sav')
            y_pred = get_prediction(X_test, model_lgb, MODEL_TYPE)
        elif MODEL_TYPE == 'xgb':
            model_xgb = joblib.load(MODELS_PATH + 'xgb_1.sav')
            y_pred = get_prediction(X_test, model_xgb, MODEL_TYPE)
            
    sub  = pd.read_csv(f'{BASE_PATH}/sample_submission.csv', dtype={'time': np.float32})

    sub['open_channels'] =  np.array(np.round(y_pred,0), np.int)
    sub.to_csv('submission.csv', index=False, float_format='%.4f')
    print(sub.head(10))


# <a id="ref"></a>
# # **8. References** 
# 
# [1] Deep-Channel uses deep neural networks to detect
# single-molecule events from patch-clamp data https://www.nature.com/articles/s42003-019-0729-3.pdf
# 
# [2] The Patch Clamp Method: https://www.youtube.com/watch?v=mVbkSD5FHOw
# 
# [3] Electophysiology: Patch clamp method https://www.youtube.com/watch?v=CvfXjGVNbmw
# 
# [4] The Action Potential https://www.youtube.com/watch?v=HYLyhXRp298
# 
# [5] https://www.kaggle.com/pestipeti/eda-ion-switching
# 
# [6] https://www.kaggle.com/kmat2019/u-net-1d-cnn-with-keras
# 
# [7] https://www.kaggle.com/cdeotte/one-feature-model-0-930
# 
# [8] https://www.kaggle.com/nxrprime/wavenet-with-shifted-rfc-proba-and-cbr

# * > **Note**: If you like my work, please, upvote ☺ <br>
