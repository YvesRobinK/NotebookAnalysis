#!/usr/bin/env python
# coding: utf-8

# # Pog Competition Baseline
# - LightGBM

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


train = pd.read_parquet('../input/kaggle-pog-series-s01e01/train.parquet')
test = pd.read_parquet('../input/kaggle-pog-series-s01e01/test.parquet')
ss = pd.read_csv('../input/kaggle-pog-series-s01e01/sample_submission.csv')


# # Setup KFold

# In[3]:


cfg = {
    'TARGET' : 'target',
    'N_FOLDS' : 5,
    'RANDOM_STATE': 529,
    'N_ESTIMATORS' : 50_000,
    'LEARNING_RATE': 0.1
}

train_vids = train['video_id'].unique()


# # Create Folds
# - This is how we will later split when validating our models

# In[4]:


kf = KFold(n_splits=cfg['N_FOLDS'],
           shuffle=True,
           random_state=cfg['RANDOM_STATE'])

# Create Folds
fold = 1
for tr_idx, val_idx in kf.split(train_vids):
    fold_vids = train_vids[val_idx]
    train.loc[train['video_id'].isin(fold_vids), 'fold'] = fold
    fold += 1
train['fold'] = train['fold'].astype('int')


# # Feature Engineering

# In[5]:


def create_features(df, train=True):
    """
    Adds features to training or test set.
    """
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    df['trending_date'] = pd.to_datetime(df['trending_date'], utc=True)
    
    # Feature 1 - Age of video
    df['video_age_seconds'] = (df['trending_date'] - df['publishedAt']) \
        .dt.total_seconds().astype('int')
    
    # Trending day of week As a category
    df['trending_dow'] = df['trending_date'].dt.day_name()
    df['trending_dow']= df['trending_dow'].astype('category')
    
    df['published_dow'] = df['publishedAt'].dt.day_name()
    df['published_dow']= df['published_dow'].astype('category')
    
    df['categoryId'] = df['categoryId'].astype('category')
    
    df['channel_occurance'] = df['channelId'].map(
        df['channelId'].value_counts().to_dict())

    df['channel_unique_video_count'] = df['channelId'].map(
        df.groupby('channelId')['video_id'].nunique().to_dict())
    
    df['video_occurance_count'] = df.groupby('video_id')['trending_date'] \
        .rank().astype('int')
    
    return df


# In[6]:


train['isTrain'] = True
test['isTrain'] = False
tt = pd.concat([train, test]).reset_index(drop=True).copy()
tt = create_features(tt)
train_feats = tt.query('isTrain').reset_index(drop=True).copy()
test_feats = tt.query('isTrain == False').reset_index(drop=True).copy()


# # Set Target and Features

# In[7]:


FEATURES = ['video_age_seconds',
            'trending_dow',
            'published_dow',
            'duration_seconds',
            'categoryId',
            'comments_disabled',
            'ratings_disabled',
            'channel_occurance',
            'channel_unique_video_count',
            'video_occurance_count'
]

TARGET = ['target']


# # Train LGBM Model

# In[8]:


X_test = test_feats[FEATURES]
oof = train_feats[['id','target','fold']].reset_index(drop=True).copy()
submission_df = test[['id']].copy()


# In[9]:


regs = []
fis = []
# Example Fold 1
for fold in range(1, 6):
    print(f'===== Running for fold {fold} =====')
    # Split train / val
    X_tr = train_feats.query('fold != @fold')[FEATURES]
    y_tr = train_feats.query('fold != @fold')[TARGET]
    X_val = train_feats.query('fold == @fold')[FEATURES]
    y_val = train_feats.query('fold == @fold')[TARGET]
    print(X_tr.shape, y_tr.shape, X_val.shape, y_val.shape)

    # Create our model
    reg = lgb.LGBMRegressor(n_estimators=cfg['N_ESTIMATORS'],
                            learning_rate=cfg['LEARNING_RATE'],
                            objective='mae',
                            metric=['mae'],
                            importance_type='gain'
                           )
    # Fit our model
    reg.fit(X_tr, y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=500,
            verbose=200,
           )

    # Predicting on validation set
    fold_preds = reg.predict(X_val,
                             num_iteration=reg.best_iteration_)
    oof.loc[oof['fold'] == fold, 'preds'] = fold_preds
    # Score validation set
    fold_score = mean_absolute_error(
        oof.query('fold == 1')['target'],
            oof.query('fold == 1')['preds']
    )

    # Creating a feature importance dataframe
    fi = pd.DataFrame(index=reg.feature_name_,
                 data=reg.feature_importances_,
                 columns=[f'{fold}_importance'])

    # Predicting on test
    fold_test_pred = reg.predict(X_test,
                num_iteration=reg.best_iteration_)
    submission_df[f'pred_{fold}'] = fold_test_pred
    print(f'Score of this fold is {fold_score:0.6f}')
    regs.append(reg)
    fis.append(fi)


# # Evaluation out of all out of fold predictions

# In[10]:


oof_score = mean_absolute_error(oof['target'], oof['preds'])
print(f'Out of fold score {oof_score:0.6f}')


# # Look at Fold Feature Importances

# In[11]:


fis_df = pd.concat(fis, axis=1)
fis_df.sort_values('1_importance').plot(kind='barh', figsize=(12, 8),
                                       title='Feature Importance Across Folds')
plt.show()


# # Create Submission

# In[12]:


pred_cols = [c for c in submission_df.columns if c.startswith('pred_')]

submission_df['target'] = submission_df[pred_cols].mean(axis=1)
# Visually check correlation between fold predictions
sns.heatmap(submission_df[pred_cols].corr(), annot=True)


# In[13]:


submission_df[['id','target']] \
    .to_csv('submission.csv', index=False)

