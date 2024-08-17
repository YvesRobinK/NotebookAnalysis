#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd 
import os
import dask.dataframe as dd

from tsfresh.feature_extraction import feature_calculators
import librosa
import pywt

from joblib import Parallel, delayed
from tqdm import tqdm_notebook
import scipy as sp
import itertools
import gc

from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
import gc


# In[ ]:


train = pd.read_csv('/kaggle/input/predict-volcanic-eruptions-ingv-oe/train.csv')
train['segment_id'] = train['segment_id'].astype(str)
train = train.sort_values('segment_id')
y = train['time_to_eruption']
del(train)


# In[ ]:


train_paths = sorted(glob('/kaggle/input/predict-volcanic-eruptions-ingv-oe/train/*'))


# In[ ]:


sub = pd.read_csv('/kaggle/input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')


# In[ ]:


np.random.seed(1337)
noise = np.random.normal(-10, 200, 60_001)


def denoise_signal_simple(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    #univeral threshold
    uthresh = 10
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec(coeff, wavelet, mode='per')


def feature_gen(path):
    X = (pd.read_csv(path).fillna(200).values.T + noise).T
    z = X - np.median(X,axis=0)
    features = []
    for i in range(X.shape[1]):
        sig = z[:,i]
        den_sample_simple = denoise_signal_simple(sig)
        mfcc = librosa.feature.mfcc(sig)
        mfcc_mean = mfcc.mean(axis=1)
        percentile_roll50_std_20 = np.percentile(pd.Series(sig).rolling(50).std().dropna().values, 20)
        features.extend([feature_calculators.number_peaks(den_sample_simple, 2),percentile_roll50_std_20,mfcc_mean[18],mfcc_mean[4]])
    return features


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_samples = [Parallel(n_jobs=4)(delayed(feature_gen)(train_path) for train_path in tqdm_notebook(train_paths))]\n')


# In[ ]:


train_samples = np.array(train_samples).reshape(-1,40)
# del(samples)
del(train_paths)
gc.collect()


# In[ ]:


test_paths = sorted(glob('/kaggle/input/predict-volcanic-eruptions-ingv-oe/test/*'))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'test_samples = [Parallel(n_jobs=4)(delayed(feature_gen)(test_path) for test_path in tqdm_notebook(test_paths))]\n')


# In[ ]:


test_samples = np.array(test_samples).reshape(-1,40)
del(test_paths)
gc.collect()


# In[ ]:


train_features = pd.DataFrame(train_samples)
train_targets = pd.DataFrame({'target': y})
test_features = pd.DataFrame(test_samples)
train_features.head()


# In[ ]:


train_data = pd.concat([train_features, train_targets], axis = 1)
test_features['target'] = None
train_data.head()


# In[ ]:


all_data = pd.concat([train_data, test_features])
all_data.head()


# In[ ]:


all_data.tail()


# In[ ]:


all_data['id'] = all_data.index
all_data.head()


# In[ ]:


all_data['target'] = all_data['target'].astype('float64')


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
for col in all_data.drop(['target', 'id'],axis=1).columns:
    all_data[col] = scaler.fit_transform(np.array(all_data[col]).reshape(-1, 1))
all_data.head()


# In[ ]:


train_data = all_data[:train_features.shape[0]]
test_data = all_data[train_features.shape[0]:]


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale


# In[ ]:


train_features = train_data.drop(['target'],axis=1)
train_targets = pd.DataFrame(train_data['target'])
test_features = test_data.drop(['target'],axis=1)
train_targets.head()


# In[ ]:


hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['mae'],
    'learning_rate': 0.001,
    'feature_fraction': 0.9,
    'subsample': 0.85,
    'subsample_freq': 2,
    'verbose': 0,
    "max_depth": 40,
    "num_leaves": 250,  
    "max_bin": 128,
    "num_iterations": 10000,
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0
}


# In[ ]:


from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import math  
from sklearn.model_selection import KFold, StratifiedKFold

score = []

skf = KFold(n_splits = 10, shuffle=True, random_state=123)
skf.get_n_splits(train_features, train_targets)
oof_lgbm_df = pd.DataFrame()
predictions = pd.DataFrame(test_features['id'])
x_test = test_features.drop(['id'], axis = 1)


for fold, (trn_idx, val_idx) in enumerate(skf.split(train_features, train_targets)):
    x_train, y_train = train_features.iloc[trn_idx], train_targets.iloc[trn_idx]['target']
    x_valid, y_valid = train_features.iloc[val_idx], train_targets.iloc[val_idx]['target']
    index = x_valid['id']
    x_train = x_train.drop(['id'], axis = 1)
    x_valid = x_valid.drop(['id'], axis = 1)
    p_valid = 0
    yp = 0
    gbm = lgb.LGBMRegressor(**hyper_params)
    gbm.fit(x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric='mae',
        verbose = 500,
        early_stopping_rounds=100)
    score.append(mean_absolute_error(gbm.predict(x_valid), y_valid))
    yp += gbm.predict(x_test)
    fold_pred = pd.DataFrame({'ID': index,
                              'label':gbm.predict(x_valid)})
    oof_rfr_df = pd.concat([oof_lgbm_df, fold_pred], axis=0)
    predictions['fold{}'.format(fold+1)] = yp


# In[ ]:


score = pd.DataFrame(score)
print(score[0].mean())
print(score[0].std())

