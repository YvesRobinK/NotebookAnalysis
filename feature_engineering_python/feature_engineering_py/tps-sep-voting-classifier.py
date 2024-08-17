#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[1]:


import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import skew
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier


# ## Load source datasets

# In[2]:


train_df = pd.read_csv("../input/tabular-playground-series-sep-2021/train.csv")
train_df.set_index('id', inplace=True)
print(f"train_df: {train_df.shape}")
train_df.head()


# In[3]:


test_df = pd.read_csv("../input/tabular-playground-series-sep-2021/test.csv")
test_df.set_index('id', inplace=True)
print(f"test_df: {test_df.shape}")
test_df.head()


# ## Feature Engineering

# In[4]:


features = test_df.columns.tolist()

train_df['num_missing'] = train_df[features].isna().sum(axis=1)
train_df['num_missing_std'] = train_df[features].isna().std(axis=1).astype('float')
train_df['median'] = train_df[features].median(axis=1)
train_df['std'] = train_df[features].std(axis=1)
train_df['min'] = train_df[features].abs().min(axis=1)
train_df['max'] = train_df[features].abs().max(axis=1)
train_df['sem'] = train_df[features].sem(axis=1)

test_df['num_missing'] = test_df[features].isna().sum(axis=1)
test_df['num_missing_std'] = test_df[features].isna().std(axis=1).astype('float')
test_df['median'] = test_df[features].median(axis=1)
test_df['std'] = test_df[features].std(axis=1)
test_df['min'] = test_df[features].abs().min(axis=1)
test_df['max'] = test_df[features].abs().max(axis=1)
test_df['sem'] = test_df[features].sem(axis=1)

print(f"train_df: {train_df.shape} \ntest_df: {test_df.shape}")
train_df.head()


# In[5]:


dataframe = pd.DataFrame(train_df.groupby(['num_missing'])['claim'].mean())
dataframe['non-claim'] = 1 - dataframe['claim']
dataframe['ratio'] = np.log(dataframe['claim'] / dataframe['non-claim'])
ratio_mapping = dataframe['ratio'].to_dict()

train_df['woe'] = train_df['num_missing'].map(ratio_mapping)
test_df['woe'] = test_df['num_missing'].map(ratio_mapping)
test_df.fillna(-1, inplace=True)
print(f"train_df: {train_df.shape} \ntest_df: {test_df.shape}")

del dataframe
gc.collect()


# In[6]:


skewed_feat = train_df[features].skew()
skewed_feat = [*skewed_feat[abs(skewed_feat.values) > 1].index]

for feat in tqdm(skewed_feat):
    median = train_df[feat].median()
    train_df[feat] = train_df[feat].fillna(median)
    test_df[feat] = test_df[feat].fillna(median)


# In[7]:


rest_cols = [col for col in features if col not in skewed_feat]

for feat in tqdm(rest_cols):
    mean = train_df[feat].mean()
    train_df[feat] = train_df[feat].fillna(mean)
    test_df[feat] = test_df[feat].fillna(mean)


# In[8]:


features = [col for col in train_df.columns if col not in ['num_missing','num_missing_std','claim']]

for col in tqdm(features):
    transformer = QuantileTransformer(n_quantiles=5000, 
                                      random_state=42, 
                                      output_distribution="normal")
    
    vec_len = len(train_df[col].values)
    vec_len_test = len(test_df[col].values)

    raw_vec = train_df[col].values.reshape(vec_len, 1)
    test_vec = test_df[col].values.reshape(vec_len_test, 1)
    transformer.fit(raw_vec)
    
    train_df[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
    test_df[col] = transformer.transform(test_vec).reshape(1, vec_len_test)[0]

print(f"train_df: {train_df.shape} \ntest_df: {test_df.shape}")


# In[9]:


def kmeans_fet(train, test, features, n_clusters):
    
    train_ = train[features].copy()
    test_ = test[features].copy()
    data = pd.concat([train_, test_], axis=0)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
    
    train[f'clusters_k'] = kmeans.labels_[:train.shape[0]]
    test[f'clusters_k'] = kmeans.labels_[train.shape[0]:]
    return train, test


# In[10]:


train_df, test_df = kmeans_fet(train_df, test_df, features, n_clusters=4)

Xtrain = train_df.loc[:, train_df.columns != 'claim'].copy()
Ytrain = train_df['claim'].copy()
Xtest = test_df.copy()

print(f"Xtrain: {Xtrain.shape} \nYtrain: {Ytrain.shape} \nXtest: {Xtest.shape}")

del train_df
del test_df
gc.collect()


# ## Model Hyperparameters

# In[11]:


lgb_params1 = {
    'metric' : 'auc',
    'objective' : 'binary',
    'n_estimators': 10000, 
    'learning_rate': 0.0223, 
    'importance_type': 'gain',
    'min_child_weight': 256,
    'min_child_samples': 20, 
    'reg_alpha': 10, 
    'reg_lambda': 0.1, 
    'subsample': 0.6, 
    'subsample_freq': 1, 
    'colsample_bytree': 0.4
}

lgb_params2 = {
    'metric' : 'auc',
    'objective' : 'binary',
    'n_estimators' : 5000,
    'learning_rate' : 0.095,
    'importance_type': 'gain',
    'max_depth' : 3,
    'num_leaves' : 7,
    'reg_alpha' : 18,
    'reg_lambda' : 17,
    'colsample_bytree' : 0.3,
    'subsample' : 0.5
}

lgb_params3 = {
    'metric' : 'auc',
    'objective' : 'binary',
    'n_estimators': 10000, 
    'learning_rate': 0.12230165751633416, 
    'importance_type': 'gain',
    'num_leaves': 1400, 
    'max_depth': 8, 
    'min_child_samples': 3100, 
    'reg_alpha': 10, 
    'reg_lambda': 65, 
    'min_split_gain': 5.157818977461183, 
    'subsample': 0.5, 
    'subsample_freq': 1, 
    'colsample_bytree': 0.2
}

xgb_params1 = {
    'eval_metric': 'auc', 
    'objective': 'binary:logistic', 
    'tree_method': 'hist', 
    'use_label_encoder': False,
    'n_estimators': 10000, 
    'learning_rate': 0.01063045229441343, 
    'gamma': 0.24652519525750877, 
    'max_depth': 4, 
    'min_child_weight': 366, 
    'subsample': 0.6423040816299684, 
    'colsample_bytree': 0.7751264493218339, 
    'colsample_bylevel': 0.8675692743597421, 
    'lambda': 0, 
    'alpha': 10
}

xgb_params2 = {
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'use_label_encoder': False,
    'n_estimators': 10000,
    'learning_rate': 0.01187431306013263,
    'max_depth': 3,
    'subsample': 0.5,
    'colsample_bytree': 0.5,
    'n_jobs': -1
}


# ## Voting Classifier

# In[12]:


estimators=[
    ('lgb11', LGBMClassifier(**lgb_params1, random_state=17)), 
    ('lgb12', LGBMClassifier(**lgb_params1, random_state=23)), 
    ('lgb21', LGBMClassifier(**lgb_params2, random_state=31)), 
    ('lgb22', LGBMClassifier(**lgb_params2, random_state=37)),
    ('lgb31', LGBMClassifier(**lgb_params3, random_state=7)),
    ('lgb32', LGBMClassifier(**lgb_params3, random_state=11)),
    ('xgb11', XGBClassifier(**xgb_params1, random_state=41)),
    ('xgb12', XGBClassifier(**xgb_params1, random_state=47)),
    ('xgb13', XGBClassifier(**xgb_params1, random_state=61)),
    ('xgb21', XGBClassifier(**xgb_params2, random_state=53)),
    ('xgb22', XGBClassifier(**xgb_params2, random_state=59)),
    ('xgb23', XGBClassifier(**xgb_params2, random_state=67))
]


# In[13]:


model = VotingClassifier(estimators=estimators, 
                         voting='soft', 
                         verbose=True)
model.fit(Xtrain, Ytrain)


# In[14]:


y_pred = model.predict_proba(Xtrain)[:,-1]
roc_auc_score(Ytrain, y_pred)


# In[15]:


y_pred_final = model.predict_proba(Xtest)[:,-1]


# In[16]:


np.savez_compressed('./VC_Meta_Features.npz',
                    y_pred_meta_vc=y_pred, 
                    y_pred_final_vc=y_pred_final)


# ## Create submission files

# In[17]:


submit_df = pd.read_csv("../input/tabular-playground-series-sep-2021/sample_solution.csv")
submit_df['claim'] = y_pred_final
submit_df.to_csv("Voting_Submission.csv", index=False)
submit_df.head(10)


# In[ ]:




