#!/usr/bin/env python
# coding: utf-8

# <h1 style='background:#31317C; border:0; color: #FFFFFF'><center>💳 AMEX Default Prediction Top10% Solution 🎉 🥉</center></h1>

# # 💳 AMEX - Default Prediction Top10% Solution 🎉 🥉
# 
# ### **It's My First Medal in Competition Thanks for Kagglers!**
# 
# **I'll study hard in the furture and share very useful information!!**
# 
# **A lot of people shared Kernel, So I think, I got Bronze Medal.**
# 
# **Thank you everyone. This medal is mine and yours!!**
# 
# ![banner](https://storage.googleapis.com/kaggle-competitions/kaggle/35332/logos/header.png?t=2022-03-23-01-05-50)
# 
# 
# <h1 style='background:#31317C; border:0; color:#FFFFFF'><center>TABLE OF CONTENTS</center></h1>
# 
# [1. Import Libraries and Load Dataset](#1)
#     
# [2. Get Difference](#2)    
# 
# [3. Processing Data](#3)        
# 
# [4. AMEX - METRIC](#4)  
# 
# [5. Configuration](#5)    
# 
# [6. KFOLD - Training & Evalutate](#6)  
# 
# [7. Ensemble - Rank Ensemble](#7)
# 
# [8. Reference](#8)
# 
# <h1 style='background:#31317C; border:0; color:#FFFFFF'><center>START</center></h1>

# <a id="1"></a>
# # **<span style="color:#4686C8;">Import Libraries and Load Dataset</span>**
# 
# ### Whats RAPIDS?
# 
# **The RAPIDS suite of open source software libraries and APIs gives you the ability to execute end-to-end data science and analytics pipelines entirely on GPUs**
# 
# **for Detail : <a href = "https://rapids.ai/index.html">LINK</a>**

# In[1]:


import gc
import os
import warnings
import joblib
import glob

import cudf
import cupy
import pandas as pd
import numpy as np

from tqdm.auto import tqdm
import itertools

import scipy as sp
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from itertools import combinations

import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


# <a id="2"></a>
# # **<span style="color:#4686C8;">Get Difference</span>**
# **This Function add new features and improved**
# 
# **I don't know detail of why this function improved performance, But it's useful**
# 
# **Therefore I'm using it!**

# In[2]:


def get_difference(data, num_features):
    df1 = []
    customer_ids = []
    
    for customer_id, df in tqdm(data.groupby(['customer_ID'])):
        diff_df1 = df[num_features].diff(1).iloc[[-1]].values.astype(np.float32)
        df1.append(diff_df1)
        customer_ids.append(customer_id)
        
    df1 = np.concatenate(df1, axis = 0)
    df1 = cudf.DataFrame(df1, columns = [col + '_diff1' for col in df[num_features].columns])
    # Add customer id
    df1['customer_ID'] = customer_ids
    return df1


# <a id="3"></a>
# # **<span style="color:#4686C8;">Processing Data</span>**
# 
# **Since the amount of data is large and takes up quite a lot of memory,**
# 
# **you should always use gc.collect or Del**

# In[3]:


def process_parquet_data(df, istrain = True):
    cat_features = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126", "D_63", "D_64", "D_66", "D_68"]
    features = df.drop(['customer_ID', 'S_2'], axis = 1).columns.to_list()
    num_features = [col for col in features if col not in cat_features]
    
    df_num_agg = df.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last'])
    df_num_agg.columns = ['_'.join(x) for x in train_num_agg.columns]
    df_num_agg.rest_index(inplace = True)
    
    df_cat_agg = df.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
    df_cat_agg.columns = ['_'.join(x) for x in df_cat_agg.columns]
    df_cat_agg.reset_index(inplace = True)
    
    cols = list(df_num_agg.dtypes[df_num_agg.dtypes == 'float64'].index)
    for col in tqdm(cols):
        df_num_agg[col] = df_num_agg[col].astype(np.float32)
        
    cols = list(df_cat_agg.dtypes[df_cat_agg.dtypes == 'int64'].index)  
    for col in tqdm(cols):
        df_cat_agg[col] = df_cat_agg[col].astype(np.int32)
        
    df_diff = get_difference(df, num_features)
    
    if istrain:
        train_labels = pd.read_csv('../input/amex-default-prediction/train_labels.csv')
        df = df_num_agg.merge(df_cat_agg, how = 'inner', on = 'customer_ID').merge(df_diff, how = 'inner', on = 'customer_ID').merge(train_labels, how = 'inner', on = 'customer_ID')
        del df_num_agg, df_cat_agg, train_labels, df_diff
    else:
        df = df_num_agg.merge(df_cat_agg, how = 'inner', on = 'customer_ID').merge(df_dfif, how = 'inner', on = 'customer_ID')
        del df_num_agg, df_cat_agg, df_diff
    
    gc.collect()
    return df


# In[4]:


def read_process_parquet_data():
    print("Staring train Feature Engineering...")
    train = cudf.read_parquet('../input/amex-data-integer-dtypes-parquet-format/train.parquet')
    train = process_parquet_data(train, istrain = True)
    
    print("Staring test Feature Engineering...")
    test = cudf.read_parquet('../input/amex-data/integer-dtypes-parquet-format/test.parquet')
    test = process_parquet_data(test, istrain = False)
    
    print("Saving Train & Test to Parquet...")
    train.to_parquet('data/train_fe.parquet')
    test.to_parquet('data/test_fe.parquet')
    
    del train, test
    gc.collect()

# read_process_parquet_data()


# <a id="4"></a>
# # **<span style="color:#4686C8;">AMEX - METRIC</span>**
# 
# **You can see AMEX - METRIC in Competition main page.**

# In[5]:


def amex_metric(y_true, y_pred):
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:,0]==0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])
    gini = [0,0]
    for i in [1,0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:,0]==0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] *  weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)
    return 0.5 * (gini[1]/gini[0] + top_four)


# In[6]:


def lgb_amex_metric(y_pred, y_true):
    y_true = y_true.get_label()
    return 'amex_metric', amex_metric(y_true, y_pred), True


# <a id="5"></a>
# # **<span style="color:#4686C8;">Configuration</span>**
# 
# **We have limited Memory. Therefore, We need to keep saving the datasets we worked on.**
# 
# **MARTIN KOVACEVIC BUVINIC's kernel say seed blend(42, 52, 62) make LB boost nicely!!**

# In[7]:


class CFG:
    input_dir = 'data/'
    seed = 42  # 52, 62
    n_folds = 5
    target = 'target'
    boosting_type = 'dart'
    metric = 'binary_logloss'

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def read_data():
    train.to_parquet(input_dir + 'train_fe.parquet')
    test.to_parquet(input_dir + 'test_fe.parquet')
    return train, test


# <a id="6"></a>
# # **<span style="color:#4686C8;">KFOLD - Training & Evalutate</span>**
# 
# **If you using Optuna. Don't using Regulation params  (reg_alpha, lambda_l1, lambda_l2, reg_lambda, min_split)**
# 
# **Since we know most Models don't overfit for dataset, Optimizing Regularization featues is unlikely to important**
# 
# **Reducing the parameter search makes a lot of sence to get result in a reasonable time period.!!**
# 
# **I recommend to apply Regularization after optimizing remaining Parameters.**

# In[8]:


def train_and_evaluate(train, test):
    cat_features = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126", "D_63", "D_64", "D_66", "D_68"]
    cat_features = [f"{cf}_last" for cf in cat_features]
    
    for cat_col in cat_features:
        encoder = LabelEncoder()
        train[cat_col] = encoder.fit_transfrom(train[cat_col])
        test[cat_col] = encoder.transform(test[cat_col])
    
    num_cols = list(train.dtypes[(train.dtypes == 'float32') | (train.dtypes == 'float64')].index)
    for col in num_cols:
        train[col + '_round2'] = train[col].round(2)
        test[col + '_round2'] = test[col].round(2)
    
    num_cols = [col for col in train.columns if 'last' in col]
    num_cols = [col[:-5] for col in num_cols if 'round' not in col]
    
    for col in num_cols:
        try:
            train[f'{col}_last_mean_diff'] = train[f'{col}_last'] - train[f'{col}_mean']
            test[f'{col}_last_mean_diff'] = test[f'{col}_last'] - test[f'{col}_mean']
        except:
            pass
        
    num_cols = list(train.dtypes[(train.dtypes == 'float32') | (train.dtypes == 'float64')].index)
    
    for col in tqdm(num_cols):
        train[col] = train[col].astype(np.float16)
        test[col] = test[col].astype(np.float16)
        
    features = [col for col in train.columns if col not in ['customer_ID', CFG.target]]
    
    params = {
        'objective': 'binary',
        'metric': CFG.metric,
        'boosting': CFG.boosting_type,
        'seed': CFG.seed,
        'num_leaves': 100,
        'learning_rate': 0.01,
        'feature_fraction': 0.20,
        'bagging_freq': 10,
        'bagging_fraction': 0.50,
        'n_jobs': -1,
        'lambda_l2': 2,
        'min_data_in_leaf': 40,
        }
    
    test_predictions = np.zeros(len(test))
    oof_predictions = np.zeros(len(train))
    kfold = StratifiedKFold(n_splits = CFG.n_folds, shuffle = True, random_state = CFG.seed)
    
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(train, train[CFG.target])):
        
        print(' ')
        print('-'*50)
        print(f'Training fold {fold} with {len(features)} features...')
        
        x_train, x_val = train[features].iloc[trn_ind], train[features].iloc[val_ind]
        y_train, y_val = train[CFG.target].iloc[trn_ind], train[CFG.target].iloc[val_ind]
        
        lgb_train = lgb.Dataset(x_train, y_train, categorical_feature = cat_features)
        lgb_valid = lgb.Dataset(x_val, y_val, categorical_feature = cat_features)
        
        model = lgb.train(
            params = params,
            train_set = lgb_train,
            num_boost_round = 10500,
            valid_sets = [lgb_train, lgb_valid],
            early_stopping_rounds = 1500,
            verbose_eval = 500,
            feval = lgb_amex_metric
            )
        
        joblib.dump(model,  f'Models/lgbm_{CFG.boosting_type}_fold{fold}_seed{CFG.seed}.pkl')
        val_pred = model.predict(x_val)
        oof_predictions[val_ind] = val_pred
        
        test_pred = model.predict(test[features])
        test_predictions += test_pred / CFG.n_folds
        score = amex_metric(y_val, val_pred)
        
        print(f'Our fold {fold} CV score is {score}')
        del x_train, x_val, y_train, y_val, lgb_train, lgb_valid
        gc.collect()
    
    score = amex_metric(train[CFG.target], oof_predictions)
    print(f'Our out of folds CV score is {score}')
    oof_df = pd.DataFrame({'customer_ID': train['customer_ID'], 'target': train[CFG.target], 
                           'prediction': oof_predictions})
    
    oof_df.to_csv(f'OOF/oof_lgbm_{CFG.boosting_type}_baseline_{CFG.n_folds}fold_seed{CFG.seed}.csv', index = False)
    test_df = pd.DataFrame({'customer_ID': test['customer_ID'], 
                            'prediction': test_predictions})
    test_df.to_csv(f'Predictions/test_lgbm_{CFG.boosting_type}_baseline_{CFG.n_folds}fold_seed{CFG.seed}.csv', index = False)

# seed_everything(CFG.seed)
# train, test = read_data()
# train_and_evaluate(train, test)    


# <a id="7"></a>
# # **<span style="color:#4686C8;">Ensemble - Rank Ensemble</span>**
# 
# **If you know about detail Rank Ensemble, <a href="https://www.analyticsvidhya.com/blog/2021/03/basic-ensemble-technique-in-machine-learning/">Click LINK</a>**
# 
# **Rank Eensemble is great performance than Average Ensemble.**
# 
# **But, I think when I using more Model, improve Private score**

# In[9]:


def AMEX_Rank_Ensemble():
    paths = [x for x in glob.glob('../input/*/*.csv') if 'amex-default-prediction' not in x if 'xgboost' not in x]
    
    df_all = [pd.read_csv(x) for x in paths]
    df_sort_all = [x.sort_values(by='customer_ID') for x in df_all]
    weights = [0.5, 0.9, 0.9, 0.5, 1, 0.8]
    
    for df in df_sort_all:
        df['prediction'] = np.clip(df['prediction'], 0, 1)
        
    submit = pd.read_csv('../input/amex-default-prediction/sample_submission.csv')
    submit['prediction'] = 0
    
    for df, weight in zip(df_sort_all, weights):
        submit['prediction'] += (rankdata(df['prediction'])/df.shape[0]) * weight
        
    submit['prediction'] /= 5
    submit.to_csv('submission_5.csv', index=None)

AMEX_Rank_Ensemble()


# <a id="8"></a>
# # **<span style="color:#4686C8;">Reference</span>**
# 
# - <a href = "https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format">AMEX data - integer dtypes - parquet format</a>
# - <a href = "https://www.kaggle.com/code/thedevastator/amex-features-the-best-of-both-worlds">Amex Features: The best of both worlds</a>
# - <a href = "https://www.kaggle.com/code/thedevastator/the-fine-art-of-hyperparameter-tuning/notebook">The Fine Art of Hyperparameter Tuning</a>
# - <a href = "https://www.kaggle.com/code/ragnar123/amex-lgbm-dart-cv-0-7977">Amex LGBM Dart CV 0.7977</a>
# - <a href = "https://www.kaggle.com/code/slowlearnermack/amex-lgbm-dart-cv-0-7963-improved">Amex LGBM Dart CV 0.7963|Improved</a>
# - <a href = "https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format">AMEX data - integer dtypes - parquet format</a>
# - <a href = "https://www.kaggle.com/code/songseungwon/xgboost-tutorial">🇰🇷한국어🇰🇷 XGBoost Tutorial</a>
