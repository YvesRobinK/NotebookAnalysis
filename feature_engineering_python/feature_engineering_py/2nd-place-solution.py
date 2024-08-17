#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random, os, gc
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(16,12)})

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def gc_clear():
    for i in range(5):
        gc.collect()


# In[2]:


data_path = "../input/digital-turbine-auction-bid-price-prediction/"
train = pd.read_csv(data_path + "train_data.csv").sort_values("eventTimestamp")
test = pd.read_csv(data_path + "test_data.csv")
ss = pd.read_csv(data_path + "sample_submission.csv")

train['bidrank'] = train.groupby('deviceId')['eventTimestamp'].rank(
    method='first',ascending=False
).astype(int)


# In[3]:


def cat_encode(train, test, colname, topN=-1):
    '''
    Encodes topN categories from train_set 
    of "colname" column
    and merge all other categories into one (if topN!=-1).
    '''
    top_df = pd.DataFrame(
        train[train.bidrank==1][colname].value_counts()
    ).reset_index().reset_index()
    if topN!=-1:
        top_df = top_df[:topN]
    else:
        topN = len(top_df)
    col_dict = dict(zip(top_df['index'], top_df['level_0']))
    train[colname] = train[colname].map(col_dict).fillna(topN).astype(int)
    test[colname] = test[colname].map(col_dict).fillna(topN).astype(int)


# In[4]:


ENCODING_LIST = [
    ('unitDisplayType',-1),
    ('brandName',-1),
    ('bundleId',-1),
    ('countryCode',6),
    ('osAndVersion',-1),
    ('connectionType',-1),
    ('c1',-1),
    ('c3',-1),
    ('size',-1),
]

for col, topN in ENCODING_LIST:
    print(f"Encoding {col} column ...")
    cat_encode(train, test, col, topN)
print("Done.")


# In[5]:


# Dictionary with grouping levels and aggregations for feature engineering
GROUP_DICTS = [
    {
        'group_level': ['deviceId'],
        'prefix': "device_",
        'aggregations': {
            'bidFloorPrice':['median'],
            'sentPrice':['mean','median','last'],
            'winBid':['std','var','min','max','mean','median','last','count', lambda x: x.quantile(0.75), lambda x: x.quantile(0.25)],
            'eventTimestamp':['last'],
            'timediff':['mean','median'],
        }
    },
    {
        'group_level': ['deviceId','unitDisplayType'],
        'prefix': 'device_display_',
        'aggregations': {
            'sentPrice':['min','max','last','mean','median'],
            'winBid':['std','var','min','max','mean','median','last','count', lambda x: x.quantile(0.75), lambda x: x.quantile(0.25)],
            'eventTimestamp':['last'],
            'timediffU':['mean','median'],
        }
    },
    {
        'group_level': ['countryCode','bundleId'],
        'prefix': 'country_bundle_',
        'aggregations': {
            'bidFloorPrice':['mean'],
            'winBid':['std','min','max','count'],        
        }
    },
    {
        'group_level': ['countryCode'],
        'prefix': 'country_',
        'aggregations': {
            'sentPrice':['mean'],
            'winBid':['std','var','max'],        
        }
    },
    {
        'group_level': ['countryCode','connectionType'],
        'prefix': 'country_connection_',
        'aggregations': {
            'sentPrice':['max'],
            'winBid':['max'],        
        }
    }
]


# In[6]:


# main function for feature engineering
def feat_eng(df_history, df_current):
    fe = df_current[[
        'eventTimestamp','deviceId','unitDisplayType','countryCode','connectionType',
        'bidFloorPrice','sentPrice','c1','c3','bundleId',
        'brandName','osAndVersion','size'
    ]]

    df_history['timediff'] = df_history.groupby(['deviceId'])['eventTimestamp'].diff()
    df_history['timediffU'] = df_history.groupby(['deviceId','unitDisplayType'])['eventTimestamp'].diff()
    for group_dict in GROUP_DICTS:
        gr_df = df_history.groupby(
            group_dict['group_level']
        ).agg(
            group_dict['aggregations']
        )
        gr_df.columns = ['_'.join(x) for x in gr_df.columns]
        gr_df.columns = [group_dict['prefix']+x for x in gr_df.columns]
        gr_df.reset_index(inplace=True)

        fe = fe.merge(gr_df, on=group_dict['group_level'], how='left')
        
    df_history['display_bidrank'] = df_history.groupby(
        ['deviceId','unitDisplayType']
    )['eventTimestamp'].rank(method='first',ascending=False).astype(int)
    dfs = df_history[
        df_history['display_bidrank']==2
    ][['deviceId','unitDisplayType','winBid']]
    dfs.columns = ['deviceId','unitDisplayType','device_display_winBid_last2']
    fe = fe.merge(dfs, on=['deviceId','unitDisplayType'], how='left')
    del df_history['display_bidrank']
    
    fe['estimation_1'] = (fe['device_winBid_last']/fe['device_sentPrice_last']) * fe['sentPrice']
    fe['estimation_2'] = (fe['device_winBid_mean']/fe['device_sentPrice_mean']) * fe['sentPrice']
    fe['estimation_3'] = (fe['device_winBid_median']/fe['device_sentPrice_median']) * fe['sentPrice']
    fe['estimation_4'] = (fe['device_winBid_last']-fe['device_sentPrice_last']) + fe['sentPrice']
    fe['estimation_5'] = (fe['device_winBid_mean']-fe['device_sentPrice_mean']) + fe['sentPrice']
    fe['estimation_6'] = (fe['device_winBid_median']-fe['device_sentPrice_median']) + fe['sentPrice']
    fe['estimation_4b'] = (fe['device_display_winBid_last']-fe['device_display_sentPrice_last']) + fe['sentPrice']
    fe['estimation_5b'] = (fe['device_display_winBid_mean']-fe['device_display_sentPrice_mean']) + fe['sentPrice']
    fe['estimation_6b'] = (fe['device_display_winBid_median']-fe['device_display_sentPrice_median']) + fe['sentPrice']
    
    fe['estimation_min'] = fe[['estimation_1','estimation_2','estimation_3']].min(axis=1)
    fe['estimation_max'] = fe[['estimation_1','estimation_2','estimation_3']].max(axis=1)
    fe['estimation_mean'] = fe[['estimation_1','estimation_2','estimation_3']].mean(axis=1)
    
    fe['device_eventTimestamp_vs_last'] = fe['eventTimestamp'] - fe['device_eventTimestamp_last']
    fe['device_display_eventTimestamp_vs_last'] = fe['eventTimestamp'] - fe['device_display_eventTimestamp_last']
    
    # drop some useless features according to Permutation Feature Importance
    fe.drop(
        columns = [
            'device_sentPrice_last','device_sentPrice_mean','device_sentPrice_median',
            'device_display_sentPrice_mean','device_display_sentPrice_median',
            'eventTimestamp','device_eventTimestamp_last','device_display_eventTimestamp_last',
        ],
        inplace=True
    )
        
    return fe


# In[7]:


fes = []
for i in range(1,11):
    trainy = train[train.bidrank>i].copy()
    val = train[train.bidrank==i].copy()
    val = trainy[['deviceId']].drop_duplicates().reset_index(drop=True).merge(val, on=['deviceId'], how='left')
    gc_clear()
    
    fe = feat_eng(trainy, val)
    fe = fe.merge(
        val[['deviceId','winBid']], 
        on=['deviceId'], 
        how='left'
    )
    fes.append(fe)
    del fe
    gc_clear()
    print(f"FE for i={i}: done")

fe = pd.concat(fes,ignore_index=True)

test_fe = feat_eng(train, test)


# In[8]:


feats = [col for col in fe.columns if col not in [
    'deviceId','winBid','bundleId','c3','device_display_winBid_last2','connectionType'
]]
catfeats = ['unitDisplayType','countryCode','c1','brandName','osAndVersion','size']


# In[9]:


def run_lgbm(train, test, conf):
    seed_everything(conf.params['seed'])
    
    features = conf.features
    cat_features = conf.cat_features
    num_features = len(features)

    if conf.predict_test:
        test_predictions = np.zeros(len(test))
    oof_predictions = np.zeros(len(train))
    feature_importance_df = pd.DataFrame()
    
    kfold = KFold(n_splits = conf.n_folds, shuffle = True, random_state = conf.params['seed'])

    for fold, (trn_ind, val_ind) in enumerate(kfold.split(train, train[conf.target])):
        print('\n'+'-'*50 + f'\nTraining fold {fold} with {num_features} features...\n')
        x_train, x_val = train[features].iloc[trn_ind], train[features].iloc[val_ind]
        y_train, y_val = train[conf.target].iloc[trn_ind], train[conf.target].iloc[val_ind]
        lgb_train = lgb.Dataset(x_train, y_train, categorical_feature = cat_features)
        lgb_valid = lgb.Dataset(x_val, y_val, categorical_feature = cat_features)
        model = lgb.train(
            params = conf.params,
            train_set = lgb_train,
            num_boost_round = conf.trees,
            valid_sets = [lgb_train, lgb_valid],
            categorical_feature = cat_features,
            callbacks=[
                log_evaluation(conf.verbose),
            ] if conf.params["boosting"]=="dart" else [
                log_evaluation(conf.verbose),
                early_stopping(conf.early,verbose=True)                
            ]
        )
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = model.feature_name()
        fold_importance_df["importance"] = model.feature_importance(importance_type='split')
        fold_importance_df["importance2"] = model.feature_importance(importance_type='gain')
        fold_importance_df["fold"] = fold
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        val_pred = model.predict(x_val)
        oof_predictions[val_ind] = val_pred

        if conf.predict_test:
            test_predictions += model.predict(test[features]) / conf.n_folds
        
        score = mean_squared_error(y_val, val_pred, squared=False)
        print(f'\nFold {fold} CV score is {score:.2f}')
        
        del x_train, x_val, y_train, y_val, lgb_train, lgb_valid
        gc_clear()
        
    oof = pd.DataFrame({'deviceId': train['deviceId'], 'winBid': train['winBid'], 'sentPrice': train['sentPrice'], 'pred': oof_predictions})
    oof['pred'] = oof[['pred', 'sentPrice']].max(axis=1)
    
    score = mean_squared_error(train[conf.target], oof_predictions, squared=False)
    score2 = mean_squared_error(oof['winBid'], oof['pred'], squared=False)
    print(f'\nOut of folds CV score is {score:.2f}; corrected score is  {score2:.2f}')
    
    if conf.predict_test:
        sub = pd.DataFrame({'deviceId': test['deviceId'], 'winBid': test_predictions})
        #sub.to_csv(f"submission_{score:.2f}_{conf.params['boosting']}_{conf.n_folds}folds_seed{conf.params['seed']}.csv", index = False)
    display_importances(feature_importance_df)
    return oof, sub
    

def display_importances(feature_importance_df_):
    tops = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:80].index
    tops2 = feature_importance_df_[["feature", "importance2"]].groupby("feature").mean().sort_values(by="importance2", ascending=False)[:80].index
    
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(tops)]
    best_features2 = feature_importance_df_.loc[feature_importance_df_.feature.isin(tops2)]
    
    plt.figure(figsize=(8, 15))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features Split (avg over folds)')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8, 15))
    sns.barplot(x="importance2", y="feature", data=best_features2.sort_values(by="importance2", ascending=False))
    plt.title('LightGBM Features Gain (avg over folds)')
    plt.tight_layout()
    plt.show()


# In[10]:


class CFG:
    n_folds = 3
    trees = 4500
    early = 5000
    verbose = 100
    params = {
        'objective': 'rmse',
        'metric': 'rmse',
        'boosting': 'dart',
        'seed': 42,
        'num_leaves': 16,
        'learning_rate': 0.01,
        'feature_fraction': 0.40,
        'n_jobs': -1,
        'lambda_l2': 2,
        'min_data_in_leaf': 40,
    }
    target = 'winBid'
    features = feats
    cat_features = catfeats
    predict_test = True


# In[11]:


C1 = CFG()

oofs_mult, subs_mult = [], []

for unitDisplayType in [0,1,2]:
    print(f"Train model for unitDisplayType={unitDisplayType}")
    
    fe_ = fe[fe.unitDisplayType==unitDisplayType].copy()
    test_fe_ = test_fe[test_fe.unitDisplayType==unitDisplayType].copy()

    oofs, subs = [], []
    for seed in [42,4,8]:
        C1.params['seed'] = seed
        oof, sub = run_lgbm(fe_, test_fe_, C1)
        oofs.append(oof)
        subs.append(sub)
        
    oofs_mult.append(oofs)
    subs_mult.append(subs)


# In[12]:


oof_lgb = []
sub_lgb = []

for i in range(len(oofs_mult)):
    print(f"i={i}")
    
    oofs = oofs_mult[i]
    subs = subs_mult[i]
    
    oof = oofs[0].copy()
    oof['pred'] = 0
    for o in oofs:
        oof['pred'] += o['pred'] / len(oofs)

    sub = subs[0].copy()
    sub['winBid'] = 0
    for s in subs:
        sub['winBid'] += s['winBid'] / len(subs)

    cv = mean_squared_error(oof['winBid'],oof['pred'],squared=False)
    print(cv)
    
    oof_lgb.append(oof)
    sub_lgb.append(sub)
    
oof = pd.concat(oof_lgb,ignore_index=True)
sub = pd.concat(sub_lgb,ignore_index=True)

cv = mean_squared_error(oof['winBid'],oof['pred'],squared=False)
print(cv)


# In[13]:


sub_lgbm = ss[['deviceId']].merge(sub[['deviceId','winBid']],on=['deviceId'],how='left')
sub_lgbm = sub_lgbm.merge(test[['deviceId','sentPrice']], on=['deviceId'], how='left')
sub_lgbm['winBid'] = sub_lgbm[['winBid','sentPrice']].max(axis=1)
sub_lgbm[['deviceId','winBid']].to_csv(f'submission_lgbm_{cv:.4f}.csv',index=False)

