#!/usr/bin/env python
# coding: utf-8

# * > https://www.kaggle.com/theoviel/load-the-totality-of-the-data
# * > https://www.kaggle.com/cttsai/forked-lightgbm-msft-malware-predictor
# * > https://www.kaggle.com/fabiendaniel/detecting-malwares-with-lgbm

# In[ ]:


import os
import gc
from functools import partial, wraps
from datetime import datetime as dt
import warnings
warnings.simplefilter('ignore', FutureWarning)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb


# In[ ]:


TARGET = 'HasDetections'
TARGET_INDEX = 'MachineIdentifier'


def modeling_cross_validation(params, X, y, nr_folds=5):
    clfs = list()
    oof_preds = np.zeros(X.shape[0])
    # Split data with kfold
    kfolds = StratifiedKFold(n_splits=nr_folds, shuffle=False, random_state=42)
    for n_fold, (trn_idx, val_idx) in enumerate(kfolds.split(X, y)):
        X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
        X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

        print("Fold {}".format(n_fold+1))
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=200, eval_metric='auc',
            early_stopping_rounds=150
        )

        clfs.append(model)
        oof_preds[val_idx] = model.predict(X_valid, num_iteration=model.best_iteration_)
        
    score = roc_auc_score(y, oof_preds)
    print(score)
    return clfs, score


# In[ ]:


def predict_cross_validation(test, clfs):
    sub_preds = np.zeros(test.shape[0])
    for i, model in enumerate(clfs, 1):    
        test_preds = model.predict_proba(test, num_iteration=model.best_iteration_)
        sub_preds += test_preds[:,1]

    sub_preds = sub_preds / len(clfs)
    ret = pd.Series(sub_preds, index=test.index)
    ret.index.name = test.index.name
    return ret


# In[ ]:


def predict_test_chunk(features, clfs, dtypes, filename='tmp.csv', chunks=100000):
    
    for i_c, df in enumerate(pd.read_csv('../input/test.csv', 
                                         chunksize=chunks, 
                                         dtype=dtypes, 
                                         iterator=True)):
        
        df.set_index(TARGET_INDEX, inplace=True)
        preds_df = predict_cross_validation(df[features], clfs)
        preds_df = preds_df.to_frame(TARGET)

        if i_c == 0:
            preds_df.to_csv(filename, header=True, mode='a', index=True)
        else:
            preds_df.to_csv(filename, header=False, mode='a', index=True)
    
        del preds_df
        gc.collect()


# In[ ]:


# cf Theo Viel https://www.kaggle.com/theoviel/load-the-totality-of-the-data
dtypes = {
       #'MachineIdentifier':                                    'category',
       'ProductName':                                          'category',
       'EngineVersion':                                        'category',
       'AppVersion':                                           'category',
       'AvSigVersion':                                         'category',
       'IsBeta':                                               'int8',
       'RtpStateBitfield':                                     'float16',
       'IsSxsPassiveMode':                                     'int8',
       'DefaultBrowsersIdentifier':                            'float16',
       'AVProductStatesIdentifier':                            'float32',
       'AVProductsInstalled':                                  'float16',
       'AVProductsEnabled':                                    'float16',
       'HasTpm':                                               'int8',
       'CountryIdentifier':                                    'int32',
       'CityIdentifier':                                       'float32',
       'OrganizationIdentifier':                               'float16',
       'GeoNameIdentifier':                                    'float32',
       'LocaleEnglishNameIdentifier':                          'int32',
       'Platform':                                             'category',
       'Processor':                                            'category',
       'OsVer':                                                'category',
       'OsBuild':                                              'int16',
       'OsSuite':                                              'int16',
       'OsPlatformSubRelease':                                 'category',
       'OsBuildLab':                                           'category',
       'SkuEdition':                                           'category',
       'IsProtected':                                          'float16',
       'AutoSampleOptIn':                                      'int8',
       'PuaMode':                                              'category',
       'SMode':                                                'float16',
       'IeVerIdentifier':                                      'float32',
       'SmartScreen':                                          'category',
       'Firewall':                                             'float16',
       'UacLuaenable':                                         'float64',
       'Census_MDC2FormFactor':                                'category',
       'Census_DeviceFamily':                                  'category',
       'Census_OEMNameIdentifier':                             'float32',
       'Census_OEMModelIdentifier':                            'float32',
       'Census_ProcessorCoreCount':                            'float16',
       'Census_ProcessorManufacturerIdentifier':               'float16',
       'Census_ProcessorModelIdentifier':                      'float32',
       'Census_ProcessorClass':                                'category',
       'Census_PrimaryDiskTotalCapacity':                      'float32',
       'Census_PrimaryDiskTypeName':                           'category',
       'Census_SystemVolumeTotalCapacity':                     'float32',
       'Census_HasOpticalDiskDrive':                           'int8',
       'Census_TotalPhysicalRAM':                              'float32',
       'Census_ChassisTypeName':                               'category',
       'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float32',
       'Census_InternalPrimaryDisplayResolutionHorizontal':    'float32',
       'Census_InternalPrimaryDisplayResolutionVertical':      'float32',
       'Census_PowerPlatformRoleName':                         'category',
       'Census_InternalBatteryType':                           'category',
       'Census_InternalBatteryNumberOfCharges':                'float32',
       'Census_OSVersion':                                     'category',
       'Census_OSArchitecture':                                'category',
       'Census_OSBranch':                                      'category',
       'Census_OSBuildNumber':                                 'int32',
       'Census_OSBuildRevision':                               'int32',
       'Census_OSEdition':                                     'category',
       'Census_OSSkuName':                                     'category',
       'Census_OSInstallTypeName':                             'category',
       'Census_OSInstallLanguageIdentifier':                   'float16',
       'Census_OSUILocaleIdentifier':                          'int32',
       'Census_OSWUAutoUpdateOptionsName':                     'category',
       'Census_IsPortableOperatingSystem':                     'int8',
       'Census_GenuineStateName':                              'category',
       'Census_ActivationChannel':                             'category',
       'Census_IsFlightingInternal':                           'float16',
       'Census_IsFlightsDisabled':                             'float16',
       'Census_FlightRing':                                    'category',
       'Census_ThresholdOptIn':                                'float16',
       'Census_FirmwareManufacturerIdentifier':                'float32',
       'Census_FirmwareVersionIdentifier':                     'float32',
       'Census_IsSecureBootEnabled':                           'int8',
       'Census_IsWIMBootEnabled':                              'float16',
       'Census_IsVirtualDevice':                               'float16',
       'Census_IsTouchEnabled':                                'int8',
       'Census_IsPenCapable':                                  'int8',
       'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
       'Wdft_IsGamer':                                         'float16',
       'Wdft_RegionIdentifier':                                'float32',
       'HasDetections':                                        'int8'
   }
       
# Some of these might be redundant, tried to reduce overfitting (num_leaves < 2^max_depth)
model_params = {
           'device': 'cpu', 
       "objective": "binary",
       "boosting_type": "gbdt", 
       "learning_rate": 0.05,
       "max_depth": 8,
       "num_leaves": 120,
       "n_estimators": 1000,
       "bagging_fraction": 0.7,
       "feature_fraction": 0.7,
       "bagging_freq": 5,
       "bagging_seed": 2018,
       'min_child_samples': 80, 
       'min_child_weight': 100.0, 
       'min_split_gain': 0.1, 
       'reg_alpha': 0.005, 
       'reg_lambda': 0.1, 
       'subsample_for_bin': 25000, 
       'min_data_per_group': 100, 
       'max_cat_to_onehot': 4, 
       'cat_l2': 25.0, 
       'cat_smooth': 2.0, 
       'max_cat_threshold': 32, 
       "random_state": 1,
       "silent": True,
       "metric": "auc",
   }


# In[ ]:


train_features = list()
    
train = pd.read_csv('../input/train.csv', nrows=2000000, dtype=dtypes).set_index(TARGET_INDEX)
train_features = [f for f in train.columns if f != TARGET]
    
# modeling
clfs, score = modeling_cross_validation(model_params, train[train_features], train[TARGET])
filename = 'subm_{:.6f}_{}_{}.csv'.format(score, 'LGBM', dt.now().strftime('%Y-%m-%d-%H-%M'))
predict_test_chunk(train_features, clfs, dtypes, filename=filename, chunks=100000)


# In[ ]:




