#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from tqdm.notebook import tqdm
import random
import joblib
import warnings
import gc
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 100)


# In[2]:


# Model Seed 
MODEL_SEED = 42
# Learning rate
LR = 0.05
# Folds
FOLDS = 5


# In[3]:


# Function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
# Calculate pearson correlation coefficient
def pearson_coef(data):
    return data.corr()['target']['prediction']

# Calculate mean pearson correlation coefficient
def comp_metric(valid_df):
    return np.mean(valid_df.groupby(['time_id']).apply(pearson_coef))

# Function to train and evaluate
def train_and_evaluate():
    # Seed everything
    seed_everything(MODEL_SEED)
    # Read data
    train = pd.read_pickle('../input/ubiquant-market-prediction-half-precision-pickle/train.pkl')
    # Feature list
    features = [col for col in train.columns if col not in ['row_id', 'time_id', 'investment_id', 'target']]
    # Some feature engineering
    # Get the correlations with the target to encode time_id
    corr1 = train[features[0:100] + ['target']].corr()['target'].reset_index()
    corr2 = train[features[100:200] + ['target']].corr()['target'].reset_index()
    corr3 = train[features[200:] + ['target']].corr()['target'].reset_index()
    corr = pd.concat([corr1, corr2, corr3], axis = 0, ignore_index = True)
    corr['target'] = abs(corr['target'])
    corr.sort_values('target', ascending = False, inplace = True)
    best_corr = corr.iloc[3:103, 0].to_list()
    del corr1, corr2, corr3, corr
    # Add time id related features (market general features to relate time_ids)
    time_id_features = []
    for col in tqdm(best_corr):
        mapper = train.groupby(['time_id'])[col].mean().to_dict()
        train[f'time_id_{col}'] = train['time_id'].map(mapper)
        train[f'time_id_{col}'] = train[f'time_id_{col}'].astype(np.float16)
        time_id_features.append(f'time_id_{col}')
    print(f'We added {len(time_id_features)} features related to time_id')
    # Update feature list
    features += time_id_features
    np.save('features.npy', np.array(features))
    np.save('best_corr.npy', np.array(best_corr))
    # Store out of folds predictions
    oof_predictions = np.zeros(len(train))
    # Initiate GroupKFold
    kfold = GroupKFold(n_splits = FOLDS)
    # Create groups based on time_id
    train.loc[(train['time_id'] >= 0) & (train['time_id'] < 280), 'group'] = 0
    train.loc[(train['time_id'] >= 280) & (train['time_id'] < 585), 'group'] = 1
    train.loc[(train['time_id'] >= 585) & (train['time_id'] < 825), 'group'] = 2
    train.loc[(train['time_id'] >= 825) & (train['time_id'] < 1030), 'group'] = 3
    train.loc[(train['time_id'] >= 1030) & (train['time_id'] < 1400), 'group'] = 4
    train['group'] = train['group'].astype(np.int16)
    #Lightgbm hyperparammeters
    params = {
        'boosting_type': 'gbdt',
        'metric': 'mse',
        'objective': 'regression',
        'n_jobs': -1,
        'seed': MODEL_SEED,
        'num_leaves': 150,
        'learning_rate': LR,
        'feature_fraction': 0.4,
        'bagging_freq': 7,
        'bagging_fraction': 0.80,
        'lambda_l1': 1,
        'lambda_l2': 3,
    }
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(train, groups = train['group'])):
        print(f'Training fold {fold + 1}')
        x_train, x_val = train[features].loc[trn_ind], train[features].loc[val_ind]
        y_train, y_val = train['target'].loc[trn_ind], train['target'].loc[val_ind]
        n_training_rows = x_train.shape[0]
        n_validation_rows = x_val.shape[0]
        # Build lgbm dataset
        train_set, val_set = lgb.Dataset(x_train, y_train), lgb.Dataset(x_val, y_val)
        print(f'Training with {n_training_rows} rows')
        print(f'Validating with {n_validation_rows} rows')
        print(f'Training light gradient boosting model with {len(features)} features...')
        # Train and evaluate
        model = lgb.train(
            params, 
            train_set, 
            num_boost_round = 10000, 
            early_stopping_rounds = 100, 
            valid_sets = [train_set, val_set], 
            verbose_eval = 100
        )
        # Predict validation set
        val_pred = model.predict(x_val)
        # Add validation prediction to out of folds array
        oof_predictions[val_ind] = val_pred
        # Save model to disk for inference
        joblib.dump(model, f'lgbm_{fold + 1}.pkl')
        del x_train, x_val, y_train, y_val, train_set, val_set
        gc.collect()
    # Compute out of folds Pearson Correlation Coefficient (for each time_id)
    oof_df = pd.DataFrame({'time_id': train['time_id'], 'target': train['target'], 'prediction': oof_predictions})
    # Save out of folds csv for blending
    oof_df.to_csv('simple_lgbm.csv', index = False)
    score = comp_metric(oof_df)
    print(f'Our out of folds mean pearson correlation coefficient is {score}')    
    
train_and_evaluate()


# In[ ]:




