#!/usr/bin/env python
# coding: utf-8

# # Pog Competition Baseline
# - Inspiration from Giba's solution: https://www.kaggle.com/titericz/imagenet-embeddings-rapids-svr-finetuned-models
# - Use image embeddings and SVC to predict video likes!

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from cuml.svm import SVR, SVC
from sklearn.svm import SVC as skSVC

import lightgbm as lgb

plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


train = pd.read_parquet('../input/kaggle-pog-series-s01e01/train.parquet')
test = pd.read_parquet('../input/kaggle-pog-series-s01e01/test.parquet')
ss = pd.read_csv('../input/kaggle-pog-series-s01e01/sample_submission.csv')
train_tn = pd.read_parquet('../input/pog-youtube-like-thumbnail-feature-embeddings/train_thumbnail_feats.parquet')
test_tn = pd.read_parquet('../input/pog-youtube-like-thumbnail-feature-embeddings/test_thumbnail_feats.parquet')


# In[3]:


train.shape, train_tn.shape, test.shape, test_tn.shape


# # Setup KFold

# In[4]:


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
# - I should have used GroupKFold

# In[5]:


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


# In[6]:


train.groupby('fold')['target'].agg(['std','mean','count'])


# # Merge Features with Thumbnail Embeddings

# In[7]:


train = train.merge(train_tn[['id'] + [f'f{n}' for n in range(1000)]],
            how='left',
            on=['id'],
           )

test = test.merge(test_tn[['id'] + [f'f{n}' for n in range(1000)]],
            how='left',
            on=['id'],
           )


# In[8]:


tn_feat_cols = [f'f{n}' for n in range(1000)]


# In[9]:


train[tn_feat_cols] = train[tn_feat_cols].fillna(0)
test[tn_feat_cols] = test[tn_feat_cols].fillna(0)


# # Feature Engineering

# In[10]:


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
    
    
    # One Hot Encode Features
    df = pd.concat([df,
           pd.get_dummies(df['trending_dow'], prefix='trending')
          ], axis=1
         )

    df = pd.concat([df,
               pd.get_dummies(df['published_dow'], prefix='pub')
              ], axis=1
             )

    df = pd.concat([df,
               pd.get_dummies(df['categoryId'], prefix='catid')
              ], axis=1
             )
    return df


# In[11]:


train['isTrain'] = True
test['isTrain'] = False
tt = pd.concat([train, test]).reset_index(drop=True).copy()
tt = create_features(tt)
train_feats = tt.query('isTrain').reset_index(drop=True).copy()
test_feats = tt.query('isTrain == False').reset_index(drop=True).copy()


# # Set Target and Features

# In[12]:


# FEATURES = ['video_age_seconds',
#             'trending_dow',
#             'published_dow',
#             'duration_seconds',
#             'categoryId',
#             'comments_disabled',
#             'ratings_disabled',
#             'channel_occurance',
#             'channel_unique_video_count',
#             'video_occurance_count',
#             'pca0', 'pca1', 'pca2', 'pca3', 'pca4'
# ]
FEATURES = tn_feat_cols
TARGET = ['target']


# In[13]:


# FEATURES


# # Train LGBM Model

# In[14]:


X_test = test_feats[FEATURES]
oof = train_feats[['id','target','fold']].reset_index(drop=True).copy()
submission_df = test[['id']].copy()


# In[15]:


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
    model = SVR(C=16.0, kernel='rbf', degree=3, max_iter=40_000, output_type='numpy')
    # Fit our model
    model.fit(X_tr, y_tr,)

    # Predicting on validation set
    fold_preds = model.predict(X_val)
    oof.loc[oof['fold'] == fold, 'preds'] = fold_preds
    # Score validation set
    fold_score = mean_absolute_error(
        oof.query('fold == @fold')['target'],
        oof.query('fold == @fold')['preds']
    )

    # Predicting on test
    fold_test_pred = model.predict(X_test)
    submission_df[f'pred_{fold}'] = fold_test_pred
    print(f'Score of this fold is {fold_score:0.6f}')


# In[16]:


oof_score = mean_absolute_error(oof['target'], oof['preds'])
print(f'Out of fold score {oof_score:0.6f}')


# # Score with just Embedding Features -> 0.05489
# ## Score (from before with lgbm) -> 0.018522

# # SVR with just old features

# In[17]:


FEATURES = ['video_age_seconds',
#             'trending_dow', # Cat
#             'published_dow', # Cat
            'duration_seconds',
#             'categoryId', # Cat
            'comments_disabled',
            'ratings_disabled',
            'channel_occurance',
            'channel_unique_video_count',
            'video_occurance_count',
#             'pca0', 'pca1', 'pca2', 'pca3', 'pca4'
             'trending_Friday',
             'trending_Monday',
             'trending_Saturday',
             'trending_Sunday',
             'trending_Thursday',
             'trending_Tuesday',
             'trending_Wednesday',
             'pub_Friday',
             'pub_Monday',
             'pub_Saturday',
             'pub_Sunday',
             'pub_Thursday',
             'pub_Tuesday',
             'pub_Wednesday',
             'catid_1',
             'catid_2',
             'catid_10',
             'catid_15',
             'catid_17',
             'catid_19',
             'catid_20',
             'catid_22',
             'catid_23',
             'catid_24',
             'catid_25',
             'catid_26',
             'catid_27',
             'catid_28',
             'catid_29'
            
]
# FEATURES = tn_feat_cols
TARGET = ['target']


# In[18]:


X_test = test_feats[FEATURES]
X_test = X_test.astype('float')
X_test = X_test.fillna(0)

oof = train_feats[['id','target','fold']].reset_index(drop=True).copy()
submission_df = test[['id']].copy()

for fold in range(1, 6):
    print(f'===== Running for fold {fold} =====')
    # Split train / val
    X_tr = train_feats.query('fold != @fold')[FEATURES]
    y_tr = train_feats.query('fold != @fold')[TARGET]
    X_val = train_feats.query('fold == @fold')[FEATURES]
    y_val = train_feats.query('fold == @fold')[TARGET]
    
    # Force as floats
    X_tr= X_tr.astype('float')
    X_val= X_val.astype('float')
    
    X_tr = X_tr.fillna(0)
    X_val = X_val.fillna(0)

    print(X_tr.shape, y_tr.shape, X_val.shape, y_val.shape)

    # Create our model
    model = SVR(C=1, kernel='rbf', degree=3, max_iter=5_000, output_type='numpy')
    # Fit our model
    model.fit(X_tr, y_tr,)

    # Predicting on validation set
    fold_preds = model.predict(X_val)
    oof.loc[oof['fold'] == fold, 'preds'] = fold_preds
    # Score validation set
    fold_score = mean_absolute_error(
        oof.query('fold == @fold')['target'],
        oof.query('fold == @fold')['preds']
    )

    # Predicting on test
    fold_test_pred = model.predict(X_test)
    submission_df[f'pred_{fold}'] = fold_test_pred
    print(f'Score of this fold is {fold_score:0.6f}')


# # SVR Model with both Embedding Features + Old Features

# In[19]:


FEATURES = ['video_age_seconds',
#             'trending_dow', # Cat
#             'published_dow', # Cat
            'duration_seconds',
#             'categoryId', # Cat
            'comments_disabled',
            'ratings_disabled',
            'channel_occurance',
            'channel_unique_video_count',
            'video_occurance_count',
#             'pca0', 'pca1', 'pca2', 'pca3', 'pca4'
             'trending_Friday',
             'trending_Monday',
             'trending_Saturday',
             'trending_Sunday',
             'trending_Thursday',
             'trending_Tuesday',
             'trending_Wednesday',
             'pub_Friday',
             'pub_Monday',
             'pub_Saturday',
             'pub_Sunday',
             'pub_Thursday',
             'pub_Tuesday',
             'pub_Wednesday',
             'catid_1',
             'catid_2',
             'catid_10',
             'catid_15',
             'catid_17',
             'catid_19',
             'catid_20',
             'catid_22',
             'catid_23',
             'catid_24',
             'catid_25',
             'catid_26',
             'catid_27',
             'catid_28',
             'catid_29'
] + tn_feat_cols
# FEATURES = tn_feat_cols
TARGET = ['target']


# In[20]:


X_test = test_feats[FEATURES]
X_test = X_test.astype('float')
X_test = X_test.fillna(0)

oof = train_feats[['id','target','fold']].reset_index(drop=True).copy()
submission_df = test[['id']].copy()

for fold in range(1, 6):
    print(f'===== Running for fold {fold} =====')
    # Split train / val
    X_tr = train_feats.query('fold != @fold')[FEATURES]
    y_tr = train_feats.query('fold != @fold')[TARGET]
    X_val = train_feats.query('fold == @fold')[FEATURES]
    y_val = train_feats.query('fold == @fold')[TARGET]
    
    # Force as floats
    X_tr= X_tr.astype('float')
    X_val= X_val.astype('float')
    
    X_tr = X_tr.fillna(0)
    X_val = X_val.fillna(0)

    print(X_tr.shape, y_tr.shape, X_val.shape, y_val.shape)

    # Create our model
    model = SVR(C=1, kernel='linear', degree=3, max_iter=5_000, output_type='numpy')
    # Fit our model
    model.fit(X_tr, y_tr,)

    # Predicting on validation set
    fold_preds = model.predict(X_val)
    oof.loc[oof['fold'] == fold, 'preds'] = fold_preds
    # Score validation set
    fold_score = mean_absolute_error(
        oof.query('fold == @fold')['target'],
        oof.query('fold == @fold')['preds']
    )

    # Predicting on test
    fold_test_pred = model.predict(X_test)
    submission_df[f'pred_{fold}'] = fold_test_pred
    print(f'Score of this fold is {fold_score:0.6f}')


# # Create Submission

# In[21]:


pred_cols = [c for c in submission_df.columns if c.startswith('pred_')]

submission_df['target'] = submission_df[pred_cols].mean(axis=1)
# Visually check correlation between fold predictions
sns.heatmap(submission_df[pred_cols].corr(), annot=True)


# In[22]:


submission_df[['id','target']] \
    .to_csv('submission.csv', index=False)

