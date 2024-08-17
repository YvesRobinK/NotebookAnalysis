#!/usr/bin/env python
# coding: utf-8

# ### In this notebook, we use [RAPIDS cudf](https://github.com/rapidsai/cudf) to create a bunch of useful features and train XGB models. The entire pipeline is lightning-fast thanks to GPU end-to-end acceleration. Train time is 20 mins and test time is 5 mins. The CV is score is `0.795` and LB score is `0.795`

# ### What you might find useful from this notebook:
# ### - Super fast pipeline. LB 0.795 in 25 mins!
# ### - "After-pay" features. It makes intuitive semse that subtracting the payments from balance/spend etc provides new information about the users' behavior.
# ### - Feature selection and hyperparameter tuning. Hundreds of GPU hours are burned to get these numbers. :P
# ### - Scalable streaming prediction. Each time only a chunk of test data is read, processed and predicted. If more features are added, you could simply make `chunks` bigger and never worry about GPU out of memory 

# In[1]:


'''
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''


# In[2]:


import cudf
import cupy
import xgboost as xgb
import numpy as np
from tqdm import tqdm
cudf.__version__


# ### Feature Engineering

# 

# In[3]:


def get_not_used():
    # cid is the label encode of customer_ID
    # row_id indicates the order of rows
    return ['row_id', 'customer_ID', 'target', 'cid', 'S_2']
    
def preprocess(df):
    df['row_id'] = cupy.arange(df.shape[0])
    not_used = get_not_used()
    cat_cols = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120',
                'D_126', 'D_63', 'D_64', 'D_66', 'D_68']

    for col in df.columns:
        if col not in not_used+cat_cols:
            df[col] = df[col].round(2)

    # compute "after pay" features
    for bcol in [f'B_{i}' for i in [11,14,17]]+['D_39','D_131']+[f'S_{i}' for i in [16,23]]:
        for pcol in ['P_2','P_3']:
            if bcol in df.columns:
                df[f'{bcol}-{pcol}'] = df[bcol] - df[pcol]

    df['S_2'] = cudf.to_datetime(df['S_2'])
    df['cid'], _ = df.customer_ID.factorize()
        
    num_cols = [col for col in df.columns if col not in cat_cols+not_used]
    
    dgs = add_stats_step(df, num_cols)
        
    # cudf merge changes row orders
    # restore the original row order by sorting row_id
    df = df.sort_values('row_id')
    df = df.drop(['row_id'],axis=1)
    return df, dgs

def add_stats_step(df, cols):
    n = 50
    dgs = []
    for i in range(0,len(cols),n):
        s = i
        e = min(s+n, len(cols))
        dg = add_stats_one_shot(df, cols[s:e])
        dgs.append(dg)
    return dgs

def add_stats_one_shot(df, cols):
    stats = ['mean','std']
    dg = df.groupby('customer_ID').agg({col:stats for col in cols})
    out_cols = []
    for col in cols:
        out_cols.extend([f'{col}_{s}' for s in stats])
    dg.columns = out_cols
    dg = dg.reset_index()
    return dg

def load_test_iter(path, chunks=4):
    
    test_rows = 11363762
    chunk_rows = test_rows // chunks
    
    test = cudf.read_parquet(f'{path}/test.parquet',
                             columns=['customer_ID','S_2'],
                             num_rows=test_rows)
    test = get_segment(test)
    start = 0
    while start < test.shape[0]:
        if start+chunk_rows < test.shape[0]:
            end = test['cus_count'].values[start+chunk_rows]
        else:
            end = test['cus_count'].values[-1]
        end = int(end)
        df = cudf.read_parquet(f'{path}/test.parquet',
                               num_rows = end-start, skiprows=start)
        start = end
        yield process_data(df)
    

def load_train(path):
    train = cudf.read_parquet(f'{path}/train.parquet')
    
    train = process_data(train)
    trainl = cudf.read_csv(f'../input/amex-default-prediction/train_labels.csv')
    train = train.merge(trainl, on='customer_ID', how='left')
    return train

def process_data(df):
    df,dgs = preprocess(df)
    df = df.drop_duplicates('customer_ID',keep='last')
    for dg in dgs:
        df = df.merge(dg, on='customer_ID', how='left')
    diff_cols = [col for col in df.columns if col.endswith('_diff')]
    df = df.drop(diff_cols,axis=1)
    return df

def get_segment(test):
    dg = test.groupby('customer_ID').agg({'S_2':'count'})
    dg.columns = ['cus_count']
    dg = dg.reset_index()
    dg['cid'],_ = dg['customer_ID'].factorize()
    dg = dg.sort_values('cid')
    dg['cus_count'] = dg['cus_count'].cumsum()
    
    test = test.merge(dg, on='customer_ID', how='left')
    test = test.sort_values(['cid','S_2'])
    assert test['cus_count'].values[-1] == test.shape[0]
    return test


# ### XGB Params and utility functions

# #### Metrics

# In[4]:


'''
def xgb_train(x, y, xt, yt):
    print("# of features:", x.shape[1])
    assert x.shape[1] == xt.shape[1]
    dtrain = xgb.DMatrix(data=x, label=y)
    dvalid = xgb.DMatrix(data=xt, label=yt)
    params = {
            'objective': 'binary:logistic', 
            'tree_method': 'gpu_hist', 
            'max_depth': 7,
            'subsample':0.88,
            'colsample_bytree': 0.5,
            'gamma':1.5,
            'min_child_weight':8,
            'lambda':70,
            'eta':0.03,
    }
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    bst = xgb.train(params, dtrain=dtrain,
                num_boost_round=2600,evals=watchlist,
                early_stopping_rounds=500, feval=xgb_amex, maximize=True,
                verbose_eval=100)
    print('best ntree_limit:', bst.best_ntree_limit)
    print('best score:', bst.best_score)
    return bst.predict(dvalid, iteration_range=(0,bst.best_ntree_limit)), bst

def lgb_train(x, y, xt, yt):
    print("# of features:", x.shape[1])
    assert x.shape[1] == xt.shape[1]
    lgb_train = lgb.Dataset(x, y, categorical_feature = cat_features)
    lgb_valid = lgb.Dataset(xt, yt, categorical_feature = cat_features)
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
    model = lgb.train(
        params = params,
        train_set = lgb_train,
        num_boost_round = 10500,
        valid_sets = [lgb_train, lgb_valid],
        early_stopping_rounds = 1500,
        verbose_eval = 500,
        feval = lgb_amex_metric
        )

    params = {
            'objective': 'binary:logistic', 
            'tree_method': 'gpu_hist', 
            'max_depth': 7,
            'subsample':0.88,
            'colsample_bytree': 0.5,
            'gamma':1.5,
            'min_child_weight':8,
            'lambda':70,
            'eta':0.03,
    }
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    bst = xgb.train(params, dtrain=dtrain,
                num_boost_round=2600,evals=watchlist,
                early_stopping_rounds=500, feval=xgb_amex, maximize=True,
                verbose_eval=100)
    print('best ntree_limit:', bst.best_ntree_limit)
    print('best score:', bst.best_score)
    return bst.predict(dvalid, iteration_range=(0,bst.best_ntree_limit)), bst
'''


# In[5]:


'''
def xgb_amex(y_pred, y_true):
    return 'amex', amex_metric_np(y_pred,y_true.get_label())

# Created by https://www.kaggle.com/yunchonggan
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/328020
def amex_metric_np(preds: np.ndarray, target: np.ndarray) -> float:
    indices = np.argsort(preds)[::-1]
    preds, target = preds[indices], target[indices]

    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_mask = cum_norm_weight <= 0.04
    d = np.sum(target[four_pct_mask]) / np.sum(target)

    weighted_target = target * weight
    lorentz = (weighted_target / weighted_target.sum()).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    n_pos = np.sum(target)
    n_neg = target.shape[0] - n_pos
    gini_max = 10 * n_neg * (n_pos + 20 * n_neg - 19) / (n_pos + 20 * n_neg)

    g = gini / gini_max
    return 0.5 * (g + d)

# we still need the official metric since the faster version above is slightly off
import pandas as pd
def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)
'''


# ### Load data and add feature

# In[6]:


'''
%%time

path = '../input/amex-data-integer-dtypes-parquet-format'
train = load_train(path)
'''


# ### Train XGB in K-folds

# In[7]:


'''
%%time

not_used = get_not_used()
not_used = [i for i in not_used if i in train.columns]
msgs = {}
folds = 4
score = 0

for i in range(folds):
    mask = train['cid']%folds == i
    tr,va = train[~mask], train[mask]
    
    x, y = tr.drop(not_used, axis=1), tr['target']
    xt, yt = va.drop(not_used, axis=1), va['target']
    yp, bst = xgb_train(x, y, xt, yt)
    bst.save_model(f'xgb_{i}.json')
    amex_score = amex_metric(pd.DataFrame({'target':yt.values.get()}), 
                                    pd.DataFrame({'prediction':yp}))
    msg = f"Fold {i} amex {amex_score:.4f}"
    print(msg)
    score += amex_score
score /= folds
print(f"Average amex score: {score:.4f}")
'''


# In[8]:


# del train
# del tr,va


# In[9]:


path = '../input/amex-data-integer-dtypes-parquet-format'
train = load_train(path)
train.to_parquet('train.parquet')


# In[10]:


get_ipython().run_cell_magic('time', '', "cids = []\nyps = []\nchunks = 4\ni = 0\nfor df in tqdm(load_test_iter(path,chunks),total=chunks):\n    cids.append(df['customer_ID'])\n#     not_used = [i for i in not_used if i in df.columns]\n\n    \n    if i == 0:\n        test_0 = df\n        test_0.to_parquet('test_0.parquet')\n    if i == 1:\n        test_1 = df\n        test_1.to_parquet('test_1.parquet')\n    if i == 2:\n        test_2 = df\n        test_2.to_parquet('test_2.parquet')\n    if i == 3:\n        test_3 = df\n        test_3.to_parquet('test_3.parquet')\n        \n    i += 1\n    \n#     yp = 0\n#     for i in range(folds):\n#         bst = xgb.Booster()\n#         bst.load_model(f'xgb_{i}.json')\n#         dx = xgb.DMatrix(df.drop(not_used, axis=1))\n#         print('best ntree_limit:', bst.best_ntree_limit)\n#         yp += bst.predict(dx, iteration_range=(0,bst.best_ntree_limit))\n#     yps.append(yp/folds)\n\n# test_all.append(test)\n\n    \n# df = cudf.DataFrame()\n# df['customer_ID'] = cudf.concat(cids)\n# df['prediction'] = np.concatenate(yps)\n# df.head()\n")


# In[11]:


df.to_csv('sub.csv',index=False)

