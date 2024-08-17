#!/usr/bin/env python
# coding: utf-8

# **Disclaimer: I am new to Python, use this Kernel at your own risk. GL! **
# 
# v.23 added KmeansSmote oversampler, messed with xgboost params, reduced %undersamples. I don't expect much improvement. Apparently, ~0.94 is where this approach hits the limit in my hands. Need a good CV strategy for tuning. Need to work on feature engineering.
# 
# v.20 **the highest LB so far: 0.9398** with `xgboost_under_over_blend.csv` submission
# 
# v.16-21 trying to add variety by using different params (mostly for xgboost) and by adding another random sampler. 
# 
# v.15 same as v.14 but removed `TransactionDT` - aka "the better way"
# 
# v.14 Current setup: Oversample using SMOTE, Random Oversample, Undersample with 17x runs. Equal blending of three aproaches.
# 
# v.13 `NearMiss version=1` does not perform well yet.
# 
# v.12 Removing SmoteBorderline oversampling part since the kernel keeps crashing. In the future will create oversampled dataset in separate kernels.
# 
# v.8 and v.9 keep crashing. RAM limit? will remove a few more columns in v.10
# 
# v.8 **Warnings:** added back `TransactionDT` as a possible leak/overfit illustration. The feature is used in some top kernels. This is likely the wrong way of using it. 
# 
# v.7 update: added NearMiss for undersampling and BorderlineSmote for oversampling
# 
# v.5 update: trying SMOTE for oversampling '1' class, then blending preds with undersampled '0' class preds.
# 
# For simple random undersampling the main theme is to:
# * randomly undersample 0 class, train on train_new, predict test
# * rinse and repeat until cows come home
# * average test predictions.
# 
# I've borrowed code from several Kernels. Let me know if I forgot to acknowledge your work.
# 
# [Undersampling](https://www.kaggle.com/artkulak/use-only-5-of-0-labels-get-negligible-lb-drop)
# 
# [Remove putative low information content columns](https://www.kaggle.com/artgor/eda-and-models)
# 
# [0.9383](https://www.kaggle.com/artkulak/ieee-fraud-simple-baseline-0-9383-lb)
# 
# [GPU Optimization](https://www.kaggle.com/xhlulu/ieee-fraud-xgboost-with-gpu-fit-in-40s)

# In[1]:


import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
import gc


# ## Pre-processing 

# In[2]:


get_ipython().run_cell_magic('time', '', "train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')\ntest_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')\n\ntrain_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')\ntest_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')\n\nsample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')\n\ntrain = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)\ntest = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)\n\nprint(train.shape)\nprint(test.shape)\n\ny_train = train['isFraud'].copy()\ndel train_transaction, train_identity, test_transaction, test_identity\n\n# Drop target, fill in NaNs\nX_train = train.drop('isFraud', axis=1)\nX_test = test.copy()\n\ndel train, test\n\nmany_null_cols = [col for col in X_train.columns if X_train[col].isnull().sum() / X_train.shape[0] > 0.96]\nmany_null_cols_X_test = [col for col in X_test.columns if X_test[col].isnull().sum() / X_test.shape[0] > 0.96]\nbig_top_value_cols = [col for col in X_train.columns if X_train[col].value_counts(dropna=False, normalize=True).values[0] > 0.96]\nbig_top_value_cols_X_test = [col for col in X_test.columns if X_test[col].value_counts(dropna=False, normalize=True).values[0] > 0.96]\ncols_to_drop = list(set(many_null_cols + many_null_cols_X_test + big_top_value_cols + big_top_value_cols_X_test ))\nlen(cols_to_drop)\nprint(cols_to_drop)\n\n\nX_train = X_train.drop(cols_to_drop, axis=1)\nX_test = X_test.drop(cols_to_drop, axis=1)\n\n\nX_train.drop('TransactionDT', axis=1, inplace=True)\nX_test.drop('TransactionDT', axis=1, inplace=True)\n\nprint(X_train.shape)\nprint(X_test.shape)\n\n# Label Encoding\nfor f in X_train.columns:\n    if X_train[f].dtype=='object' or X_test[f].dtype=='object': \n        lbl = preprocessing.LabelEncoder()\n        lbl.fit(list(X_train[f].values) + list(X_test[f].values))\n        X_train[f] = lbl.transform(list(X_train[f].values))\n        X_test[f] = lbl.transform(list(X_test[f].values)) \n        \nX_train = X_train.fillna(-999)\nX_test = X_test.fillna(-999)\n")


# ## Reducing RAM usage

# In[3]:


get_ipython().run_cell_magic('time', '', '# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage\n# WARNING! THIS CAN DAMAGE THE DATA \ndef reduce_mem_usage(df):\n    """ iterate through all the columns of a dataframe and modify the data type\n        to reduce memory usage.        \n    """\n    start_mem = df.memory_usage().sum() / 1024**2\n    print(\'Memory usage of dataframe is {:.2f} MB\'.format(start_mem))\n    \n    for col in df.columns:\n        col_type = df[col].dtype\n        \n        if col_type != object:\n            c_min = df[col].min()\n            c_max = df[col].max()\n            if str(col_type)[:3] == \'int\':\n                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:\n                    df[col] = df[col].astype(np.int8)\n                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:\n                    df[col] = df[col].astype(np.int16)\n                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:\n                    df[col] = df[col].astype(np.int32)\n                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:\n                    df[col] = df[col].astype(np.int64)  \n            else:\n                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:\n                    df[col] = df[col].astype(np.float16)\n                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:\n                    df[col] = df[col].astype(np.float32)\n                else:\n                    df[col] = df[col].astype(np.float64)\n        else:\n            df[col] = df[col].astype(\'category\')\n\n    end_mem = df.memory_usage().sum() / 1024**2\n    print(\'Memory usage after optimization is: {:.2f} MB\'.format(end_mem))\n    print(\'Decreased by {:.1f}%\'.format(100 * (start_mem - end_mem) / start_mem))\n    \n    return df\nX_train = reduce_mem_usage(X_train)\nX_test = reduce_mem_usage(X_test)\n')


# ## Oversampling using KmeansSMOTE

# In[4]:


from imblearn.over_sampling import KMeansSMOTE

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = KMeansSMOTE(random_state=99, sampling_strategy = 0.15,  k_neighbors = 10,cluster_balance_threshold = 0.02, n_jobs=4)
X_train_new, y_train_new = sm.fit_sample(X_train, y_train.ravel())

X_train_new = pd.DataFrame(X_train_new)
X_train_new.columns = X_train.columns
y_train_new = pd.DataFrame(y_train_new)

print('After OverSampling, the shape of X_train_new: {}'.format(X_train_new.shape))
print('After OverSampling, the shape of y_train_new: {} \n'.format(y_train_new.shape))


# In[5]:


get_ipython().run_cell_magic('time', '', "from sklearn.model_selection import StratifiedKFold\nfrom sklearn.metrics import roc_auc_score\n\nEPOCHS = 4\n\ny_preds = np.zeros(sample_submission.shape[0])\n\nkf = StratifiedKFold(n_splits=EPOCHS, random_state= 99, shuffle=True)\ny_oof_new = np.zeros(X_train_new.shape[0])\ngc.collect()\n\nfor tr_idx, val_idx in kf.split(X_train_new, y_train_new):\n    clf = xgb.XGBClassifier(\n            n_estimators=500,\n            max_depth=17,\n            learning_rate=0.03,\n            subsample=0.9,\n            colsample_bytree=0.9,\n            tree_method='gpu_hist',\n            missing=-999\n        )\n    \n    X_tr, X_vl = X_train_new.iloc[tr_idx, :], X_train_new.iloc[val_idx, :]\n    y_tr, y_vl = y_train_new.iloc[tr_idx], y_train_new.iloc[val_idx]\n    clf.fit(X_tr, y_tr)\n    y_pred_train = clf.predict_proba(X_vl)[:,1]\n    #y_oof[val_idx] = y_pred_train\n    print('ROC AUC {}'.format(roc_auc_score(y_vl, y_pred_train)))\n    y_oof_new[val_idx] = y_pred_train    \n    y_preds+= clf.predict_proba(X_test)[:,1] / EPOCHS\n    del clf\n    gc.collect()\nprint('ROC AUC oof_new {}'.format(roc_auc_score(y_train_new, y_oof_new))) \ndel X_train_new\ngc.collect()\n\nsample_submission1a = sample_submission.copy()\nsample_submission1a['isFraud'] = y_preds\nsample_submission1a.to_csv('xgboost_oversample.csv')\nsample_submission1a['isFraud'].describe()\n")


# ## Oversampling using SMOTE

# In[6]:


from imblearn.over_sampling import SMOTE

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = SMOTE(random_state=99, sampling_strategy = 0.15)
X_train_new, y_train_new = sm.fit_sample(X_train, y_train.ravel())

X_train_new = pd.DataFrame(X_train_new)
X_train_new.columns = X_train.columns
y_train_new = pd.DataFrame(y_train_new)

print('After OverSampling, the shape of X_train_new: {}'.format(X_train_new.shape))
print('After OverSampling, the shape of y_train_new: {} \n'.format(y_train_new.shape))


# In[7]:


get_ipython().run_cell_magic('time', '', "#training on Smote dataset\n#from sklearn.model_selection import KFold\nfrom sklearn.model_selection import StratifiedKFold\nfrom sklearn.metrics import roc_auc_score\n\nEPOCHS = 4\n\n#kf = KFold(n_splits = EPOCHS, shuffle = True)\ny_preds = np.zeros(sample_submission.shape[0])\n\n\nkf = StratifiedKFold(n_splits=EPOCHS, random_state= 99, shuffle=True)\n#y_oof = np.zeros(X_train.shape[0])\n#y_train_new = y_train_new.reset_index().drop(columns = 'TransactionID')\n#X_train_new = X_train_new.reset_index().drop(columns = 'TransactionID')\ny_oof_new = np.zeros(X_train_new.shape[0])\ngc.collect()\n\nfor tr_idx, val_idx in kf.split(X_train_new, y_train_new):\n    clf = xgb.XGBClassifier(\n            n_estimators=500,\n            max_depth=17,\n            learning_rate=0.02,\n            subsample=0.9,\n            colsample_bytree=0.9,\n            tree_method='gpu_hist',\n            missing=-999\n        )\n    \n    X_tr, X_vl = X_train_new.iloc[tr_idx, :], X_train_new.iloc[val_idx, :]\n    y_tr, y_vl = y_train_new.iloc[tr_idx], y_train_new.iloc[val_idx]\n    clf.fit(X_tr, y_tr)\n    y_pred_train = clf.predict_proba(X_vl)[:,1]\n    #y_oof[val_idx] = y_pred_train\n    print('ROC AUC {}'.format(roc_auc_score(y_vl, y_pred_train)))\n    y_oof_new[val_idx] = y_pred_train    \n    y_preds+= clf.predict_proba(X_test)[:,1] / EPOCHS\n    del clf\n    gc.collect()\nprint('ROC AUC oof_new {}'.format(roc_auc_score(y_train_new, y_oof_new))) \ndel X_train_new\ngc.collect()\n\nsample_submission1 = sample_submission.copy()\nsample_submission1['isFraud'] = y_preds\nsample_submission1.to_csv('xgboost_oversample.csv')\nsample_submission1['isFraud'].describe()\n")


# ## Oversampling using RandomOverSampler

# In[8]:


from imblearn.over_sampling import RandomOverSampler

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = RandomOverSampler(random_state=99, sampling_strategy = 0.12)
X_train_new, y_train_new = sm.fit_sample(X_train, y_train.ravel())

X_train_new = pd.DataFrame(X_train_new)
X_train_new.columns = X_train.columns
y_train_new = pd.DataFrame(y_train_new)

print('After OverSampling, the shape of X_train_new: {}'.format(X_train_new.shape))
print('After OverSampling, the shape of y_train_new: {} \n'.format(y_train_new.shape))


# In[9]:


get_ipython().run_cell_magic('time', '', "from sklearn.model_selection import StratifiedKFold\nfrom sklearn.metrics import roc_auc_score\n\nEPOCHS = 4\n\ny_preds = np.zeros(sample_submission.shape[0])\n\nkf = StratifiedKFold(n_splits=EPOCHS, random_state= 99, shuffle=True)\ny_oof_new = np.zeros(X_train_new.shape[0])\ngc.collect()\n\nfor tr_idx, val_idx in kf.split(X_train_new, y_train_new):\n    clf = xgb.XGBClassifier(\n            n_estimators=500,\n            max_depth=11,\n            learning_rate=0.05,\n            subsample=0.9,\n            colsample_bytree=0.9,\n            tree_method='gpu_hist',\n            missing=-999,\n            min_child_weight=1\n        )\n    \n    X_tr, X_vl = X_train_new.iloc[tr_idx, :], X_train_new.iloc[val_idx, :]\n    y_tr, y_vl = y_train_new.iloc[tr_idx], y_train_new.iloc[val_idx]\n    clf.fit(X_tr, y_tr)\n    y_pred_train = clf.predict_proba(X_vl)[:,1]\n    #y_oof[val_idx] = y_pred_train\n    print('ROC AUC {}'.format(roc_auc_score(y_vl, y_pred_train)))\n    y_oof_new[val_idx] = y_pred_train    \n    y_preds+= clf.predict_proba(X_test)[:,1] / EPOCHS\n    del clf\n    gc.collect()\nprint('ROC AUC oof_new {}'.format(roc_auc_score(y_train_new, y_oof_new))) \ndel X_train_new\ngc.collect()\n\nsample_submission2 = sample_submission.copy()\nsample_submission2['isFraud'] = y_preds\nsample_submission2.to_csv('xgboost_oversample_random.csv')\nsample_submission2['isFraud'].describe()\n")


# ## Oversampling using RandomOverSampler (slighly diff params)

# In[10]:


from imblearn.over_sampling import RandomOverSampler

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = RandomOverSampler(random_state=999, sampling_strategy = 0.13)
X_train_new, y_train_new = sm.fit_sample(X_train, y_train.ravel())

X_train_new = pd.DataFrame(X_train_new)
X_train_new.columns = X_train.columns
y_train_new = pd.DataFrame(y_train_new)

print('After OverSampling, the shape of X_train_new: {}'.format(X_train_new.shape))
print('After OverSampling, the shape of y_train_new: {} \n'.format(y_train_new.shape))


# In[11]:


get_ipython().run_cell_magic('time', '', "from sklearn.model_selection import StratifiedKFold\nfrom sklearn.metrics import roc_auc_score\n\nEPOCHS = 4\n\ny_preds = np.zeros(sample_submission.shape[0])\n\n\nkf = StratifiedKFold(n_splits=EPOCHS, random_state= 99, shuffle=True)\ny_oof_new = np.zeros(X_train_new.shape[0])\ngc.collect()\n\nfor tr_idx, val_idx in kf.split(X_train_new, y_train_new):\n    clf = xgb.XGBClassifier(\n            n_estimators=500,\n            max_depth=17,\n            learning_rate=0.035,\n            subsample=0.9,\n            colsample_bytree=0.9,\n            tree_method='gpu_hist',\n            missing=-999\n        )\n    \n    X_tr, X_vl = X_train_new.iloc[tr_idx, :], X_train_new.iloc[val_idx, :]\n    y_tr, y_vl = y_train_new.iloc[tr_idx], y_train_new.iloc[val_idx]\n    clf.fit(X_tr, y_tr)\n    y_pred_train = clf.predict_proba(X_vl)[:,1]\n    #y_oof[val_idx] = y_pred_train\n    print('ROC AUC {}'.format(roc_auc_score(y_vl, y_pred_train)))\n    y_oof_new[val_idx] = y_pred_train    \n    y_preds+= clf.predict_proba(X_test)[:,1] / EPOCHS\n    del clf\n    gc.collect()\nprint('ROC AUC oof_new {}'.format(roc_auc_score(y_train_new, y_oof_new))) \ndel X_train_new\ngc.collect()\n\nsample_submission3 = sample_submission.copy()\nsample_submission3['isFraud'] = y_preds\nsample_submission3.to_csv('xgboost_oversample_random.csv')\nsample_submission3['isFraud'].describe()\n")


# ## Random undersampling of '0' class, multiple runs

# In[12]:


get_ipython().run_cell_magic('time', '', "from sklearn.model_selection import StratifiedKFold\nfrom sklearn.metrics import roc_auc_score\n\nEPOCHS = 4\nTIMES = 24\nfrac_inc = np.arange(0,TIMES,1)/400 +0.1\nprint(frac_inc)\n\ny_preds = np.zeros(sample_submission.shape[0])\n\nfor _ in range (TIMES):\n    kf = StratifiedKFold(n_splits=EPOCHS, random_state= _, shuffle=True)\n    #y_oof = np.zeros(X_train.shape[0])\n    #X_train_new = X_train[y_train == 0].sample(frac = 0.09).append(X_train[y_train == 1])\n    X_train_new = X_train[y_train == 0].sample(frac = frac_inc[_]).append(X_train[y_train == 1])\n    y_train_new = y_train[X_train_new.index].reset_index().drop(columns = 'TransactionID')\n    X_train_new = X_train_new.reset_index().drop(columns = 'TransactionID')\n    #X_train_new\n    #y_train_new\n    y_oof_new = np.zeros(X_train_new.shape[0])\n    gc.collect()\n\n    for tr_idx, val_idx in kf.split(X_train_new, y_train_new):\n        clf = xgb.XGBClassifier(\n            n_estimators=500,\n            max_depth=10,\n            learning_rate=0.04,\n            subsample=0.8,\n            colsample_bytree=0.9,\n            tree_method='gpu_hist',\n            missing=-999,\n            min_child_weight=2\n        )\n    \n        X_tr, X_vl = X_train_new.iloc[tr_idx, :], X_train_new.iloc[val_idx, :]\n        y_tr, y_vl = y_train_new.iloc[tr_idx], y_train_new.iloc[val_idx]\n        clf.fit(X_tr, y_tr)\n        y_pred_train = clf.predict_proba(X_vl)[:,1]\n        #y_oof[val_idx] = y_pred_train\n        #print('ROC AUC {}'.format(roc_auc_score(y_vl, y_pred_train)))\n        y_oof_new[val_idx] = y_pred_train    \n        y_preds+= clf.predict_proba(X_test)[:,1] / EPOCHS   \n    print('ROC AUC oof_new {}'.format(roc_auc_score(y_train_new, y_oof_new))) \n    \nsample_submission4 = sample_submission.copy()\nsample_submission4['isFraud'] = y_preds/TIMES\nsample_submission4.to_csv('xgboost_undersample.csv')\nsample_submission4['isFraud'].describe()\n")


# ## Blending over- and under-sampled results

# In[13]:


sample_submission_blend = sample_submission.copy()
sample_submission_blend['isFraud'] = (sample_submission1a['isFraud']+sample_submission1['isFraud'] + sample_submission2['isFraud']*0.5+ sample_submission3['isFraud']*0.5+ sample_submission4['isFraud'])/4
sample_submission_blend.to_csv('xgboost_under_over_blend.csv')
sample_submission_blend2 = sample_submission.copy()
sample_submission_blend2['isFraud'] = (sample_submission1a['isFraud']*0.5+sample_submission1['isFraud']*0.5 + sample_submission2['isFraud']*0.25+ sample_submission3['isFraud']*0.25+ sample_submission4['isFraud'])/2.5
sample_submission_blend2.to_csv('xgboost_under_over_blend2.csv')
sample_submission_blend_equal = sample_submission.copy()
sample_submission_blend_equal['isFraud'] = (sample_submission1a['isFraud']+sample_submission1['isFraud'] + sample_submission2['isFraud']+ sample_submission3['isFraud']+ sample_submission4['isFraud'])/5
sample_submission_blend_equal.to_csv('xgboost_under_over_blend_equal.csv')

