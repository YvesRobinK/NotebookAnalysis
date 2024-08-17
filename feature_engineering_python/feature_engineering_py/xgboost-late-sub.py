#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler

def balance(df, tgt):
    df0 = df.loc[tgt==0].copy()
    
    df1 = df.loc[tgt==1].copy()
    df1 = df1.sample(n = df0.shape[0], replace=True, random_state=2023)
    df = pd.concat([df0, df1])
    tgt = np.concatenate((
        np.zeros(len(df0)),
        np.ones(len(df1)),
    ))
    return df, tgt


def balance_probs(p):
    class_0_est_instances = p[:,0].sum()
    others_est_instances = p[:,1].sum()
    rebalance = np.array([1/class_0_est_instances, 1/others_est_instances])
    new_p = p * rebalance
    new_p =  new_p / np.sum(new_p, axis=1, keepdims=1)        
    return new_p


train = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/train.csv')
test = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/test.csv')
sample = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/sample_submission.csv')
greeks = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/greeks.csv')
greeks['epsilon_date'] = pd.to_datetime(greeks['Epsilon'].replace('Unknown', None), format='%m/%d/%Y')

# mega feature engineering )
train.EJ = train.EJ.eq('B').astype('int')
test.EJ = test.EJ.eq('B').astype('int')

# just use all features
features = [i for i in train.columns if i not in ['Id', 'Class']]

# drop null and old data
train = train.merge(greeks, on='Id')
train = train[train['epsilon_date'].dt.year >= 2017]
train['year'] = train['epsilon_date'].dt.year
test['year'] = train['year'].max() + 1

train.shape


# In[2]:


FEATURES = ['year', 'AB', 'AF', 'AH', 'AM', 'AR', 'AX', 'AY', 'AZ', 'BC', 'BD ', 'BN', 'BP', 'BQ', 'BR', 'BZ', 'CB', 'CC', 'CD ', 'CF', 'CH', 'CL', 'CR', 'CS', 'CU', 'CW ', 'DA', 'DE', 'DF', 'DH', 'DI', 'DL', 'DN', 'DU', 'DV', 'DY', 'EB', 'EE', 'EG', 'EH', 'EJ', 'EL', 'EP', 'EU', 'FC', 'FD ', 'FE', 'FI', 'FL', 'FR', 'FS', 'GB', 'GE', 'GF', 'GH', 'GI', 'GL']

SCALER = StandardScaler()
SCALER.fit(pd.concat([train[FEATURES], test[FEATURES]]))
train[FEATURES] = SCALER.transform(train[FEATURES])
test[FEATURES] = SCALER.transform(test[FEATURES])

train = train.fillna(-9999.)
test = test.fillna(-9999.)

features = [f for f in FEATURES if f not in ['CW ', 'GE', 'AY', 'DF', 'CS', 'sum_zero', 'EL', 'DA', 'CF', 'DE']]      
for col in ['FD ', 'EU', 'EE', 'DI', 'CH', 'BN', 'BR', 'GH', 'DH', 'CC', 'CL', 'CS',]: 
    dt = pd.concat([train[[col]],  test[[col]]]).groupby(col)[col].count().to_dict()
    train[col + '_count'] = train[col].map(dt)
    test[col + '_count'] = test[col].map(dt)
    features.append(col + '_count')

# Important to balance trainset
tra_balance, tgt_balance = balance(train[features], train['Class'].values)

params = {
    'subsample': 0.6,
    'colsample_bytree': 0.95,
    'learning_rate': 0.1,
    'max_depth': 8,
    'max_delta_step': 0.200,
    'min_child_weight': 1.000,
    'max_samples': 0.75,
    'max_features': 0.90,
} 
rd = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=880,
    subsample=params['subsample'],
    colsample_bytree=params['colsample_bytree'],
    learning_rate=params['learning_rate'],
    max_depth=int(np.round(params['max_depth'])),
    max_delta_step=params['max_delta_step'],
    min_child_weight=params['min_child_weight'],
    tree_method='hist',
    nthread= 1,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    seed=2023,
    gamma=0,
    base_score=0.5,
)        
model = BaggingClassifier(base_estimator=rd, n_estimators=100, max_samples=params['max_samples'], max_features=params['max_features'], n_jobs= -1)
model.fit(tra_balance, tgt_balance)


# In[3]:


ytest = balance_probs(model.predict_proba(test[features]))

submission = pd.DataFrame(test["Id"], columns=["Id"])
submission["class_1"] = ytest[:, 1]
submission["class_0"] = 1 - submission["class_1"]
submission.to_csv('submission.csv', index=False)


# In[ ]:




