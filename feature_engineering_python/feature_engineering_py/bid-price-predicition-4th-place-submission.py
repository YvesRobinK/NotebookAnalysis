#!/usr/bin/env python
# coding: utf-8

# # Loading packages

# In[1]:


import pandas as pd
import numpy as np
import time
import gc
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
import lightgbm as lgb

from sklearn.model_selection import train_test_split


# # Loading raw data: train, test, sample submission

# In[2]:


train = pd.read_csv('../input/digital-turbine-auction-bid-price-prediction/train_data.csv')
test = pd.read_csv('../input/digital-turbine-auction-bid-price-prediction/test_data.csv')
sample = pd.read_csv('../input/digital-turbine-auction-bid-price-prediction/sample_submission.csv')

print(f'Number of rows in train set: {train.shape[0]}')
print(f'Number of rows in test set: {test.shape[0]}')

train.head()


# # Preapre function to create features based on raw data (input train or test set)

# In[3]:


brand_train_popularity = train['brandName'].value_counts().rename('popularity_of_brand')
app_train_popularity = train['bundleId'].value_counts().rename('popularity_of_app')
device_train_popularity = train['deviceId'].value_counts().rename('popularity_of_device')

train['eventDate'] = pd.to_datetime(train['eventTimestamp'], unit='ms')
train['hour'] = train['eventDate'].dt.hour
mean_hour_device = train.groupby('deviceId')['hour'].mean()

mean_win_bid_brand = train.groupby('brandName')['winBid'].mean()
mean_win_bid_app = train.groupby('bundleId')['winBid'].mean()

mean_win_bid_device = train.groupby('deviceId')['winBid'].mean()
median_win_bid_device = train.groupby('deviceId')['winBid'].median()
std_win_bid_device = train.groupby('deviceId')['winBid'].std()
min_win_bid_device = train.groupby('deviceId')['winBid'].min()
max_win_bid_device = train.groupby('deviceId')['winBid'].max()

mean_sent_price_device = train.groupby('deviceId')['sentPrice'].mean()
median_sent_price_device = train.groupby('deviceId')['sentPrice'].median()
std_sent_price_device = train.groupby('deviceId')['sentPrice'].std()
min_sent_price_device = train.groupby('deviceId')['sentPrice'].min()
max_sent_price_device = train.groupby('deviceId')['sentPrice'].max()

mean_win_bid_c1 = train.groupby('c1')['winBid'].mean()
mean_win_bid_c3 = train.groupby('c3')['winBid'].mean()

def create_features(df):
    
    df['eventDate'] = pd.to_datetime(df['eventTimestamp'], unit='ms')
    df['hour'] = df['eventDate'].dt.hour
    df['day_of_week'] = df['eventDate'].dt.dayofweek - 27
    df['is_weekend'] = np.where(df['eventDate'].dt.dayofweek>4, 1, 0)
    df['week_of_year'] = df['eventDate'].dt.week
    df['mean_hour_device'] = df['deviceId'].map(mean_hour_device)
    
    df['is_banner'] = np.where(df['unitDisplayType'] == 'banner', 1, 0)
    df['is_interstitial'] = np.where(df['unitDisplayType'] == 'interstitial', 1, 0)
    df['is_rewarded'] = np.where(df['unitDisplayType'] == 'rewarded', 1, 0)
    
    df['popularity_of_brand'] = df['brandName'].map(brand_train_popularity)
    df['popularity_of_app'] = df['bundleId'].map(app_train_popularity)
    df['popularity_of_device'] = df['deviceId'].map(device_train_popularity)
    
    df['mean_win_bid_brand'] = df['deviceId'].map(mean_win_bid_brand)
    df['mean_win_bid_app'] = df['deviceId'].map(mean_win_bid_app)
    
    df['is_apple'] = np.where(df['brandName'] == 'Apple', 1, 0)
    df['is_android'] = np.where(df['osAndVersion'].str.split('-').str[0] == "Android", 1, 0)
    df['is_ios'] = np.where(df['osAndVersion'].str.split('-').str[0] == "iOS", 1, 0)
    
    df['is_US'] = np.where(df['countryCode'] == 'US', 1, 0)
    df['is_JP'] = np.where(df['countryCode'] == 'JP', 1, 0)
    df['is_CH'] = np.where(df['countryCode'] == 'CH', 1, 0)
    df['is_BR'] = np.where(df['countryCode'] == 'BR', 1, 0)
    
    df['mean_win_bid_device'] = df['deviceId'].map(mean_win_bid_device)
    df['median_win_bid_device'] = df['deviceId'].map(median_win_bid_device)
    df['std_win_bid_device'] = df['deviceId'].map(std_win_bid_device)
    df['min_win_bid_device'] = df['deviceId'].map(min_win_bid_device)
    df['max_win_bid_device'] = df['deviceId'].map(max_win_bid_device)
    
    df['mean_sent_price_device'] = df['deviceId'].map(mean_sent_price_device)
    df['median_sent_price_device'] = df['deviceId'].map(median_sent_price_device)
    df['std_sent_price_device'] = df['deviceId'].map(std_sent_price_device)
    df['min_sent_price_device'] = df['deviceId'].map(min_sent_price_device)
    df['max_sent_price_device'] = df['deviceId'].map(max_sent_price_device)
    
    df['is_WIFI'] = np.where(df['connectionType'] == 'WIFI', 1, 0)
    df['is_3G'] = np.where(df['connectionType'] == '3G', 1, 0)
    
    df['mean_win_bid_c1'] = df['c1'].map(mean_win_bid_c1)
    df['mean_win_bid_c3'] = df['c3'].map(mean_win_bid_c3)
    
    df['c2_multiply_c4'] = df['c2'] * df['c4']
    
    df['size_width'] = df['size'].str.split('x').str[0].astype(int)
    df['size_height'] = df['size'].str.split('x').str[1].astype(int)
    df['size_pixels'] = df['size_width'] * df['size_height']
    
    df['mediation_minor'] = df['mediationProviderVersion'].str.split('.').str[1].astype(int)
    
    return df


# # Apply function and remove raw set

# In[4]:


train_feats = create_features(train)
test_feats = create_features(test)

del train, test
gc.collect()


# # Create list of final features which will take part in creating model

# In[5]:


FEATURES = ['hour', 'day_of_week', 'week_of_year', 'mean_hour_device', 'is_banner', 'is_interstitial', 'is_rewarded', 'popularity_of_brand', 'popularity_of_device', 
            'is_apple', 'is_US', 'mean_win_bid_device', 'median_win_bid_device', 'min_win_bid_device', 'max_win_bid_device', 
            'mean_sent_price_device', 'median_sent_price_device', 'min_sent_price_device', 'max_sent_price_device', 'is_WIFI',
            'is_3G', 'mean_win_bid_c1', 'mean_win_bid_c3', 'size_width', 'size_height', 'mediation_minor', 'sentPrice']


# # Create list empty lists for folds and divide to n folds

# In[6]:


models = []
importances = []
predictions = []
oofs = []

folds = KFold(n_splits = 7, shuffle = True, random_state = 42)


# # Learn using params in folds loop and save results into lists

# In[7]:


params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.1,
    #'colsample_bytree': 0.2,
    'bagging_fraction': 0.6,
    #'bagging_freq': 10,
    'verbose': -1,
    "max_depth": 12,
    "n_estimators": 3200
}

for fold, (train_idx, valid_idx) in enumerate(folds.split(train_feats, train_feats['winBid'])):

    start = time.time()
    print(f'--------- Fold: {fold} ----------')
    
    train_feats['winBid']
    X_train, y_train = train_feats[FEATURES].iloc[train_idx], train_feats['winBid'].iloc[train_idx]
    X_valid, y_valid = train_feats[FEATURES].iloc[valid_idx],  train_feats['winBid'].iloc[valid_idx]

    model = lgb.LGBMRegressor(**params)

    model.fit(X_train[FEATURES], y_train, eval_set = (X_valid[FEATURES], y_valid),
          callbacks = [lgb.early_stopping(40), lgb.log_evaluation(100)])
    
    oof = model.predict(X_valid[FEATURES], num_iteration = model.best_iteration_)
    fold_score = sqrt(mean_squared_error(oof, y_valid))
    print(f'RMSE OFF of this fold is: {fold_score:0.4f}')
    
    importance = pd.DataFrame(index = model.feature_name_, data = model.feature_importances_, columns = [f'{fold}_importance'])

    fold_test_pred = model.predict(test_feats[FEATURES], num_iteration = model.best_iteration_)
    
    predictions.append(fold_test_pred)
    models.append(model)
    oofs.append(fold_score)
    importances.append(importance)
    
    end = time.time()
    full_time = (end - start)
    
    print(f'Time to train, predict and save results: {full_time:0.2f} seconds')


# # Print general OOF (mean oof of all folds)

# In[8]:


print(f'General CV (OOF): {np.mean(oofs):0.4f}')


# In[9]:


pd.set_option('display.float_format', lambda x: '%.3f' % x)
importances_df = pd.concat(importances, axis = 1)
importances_df['Feature'] = importance.index
importances_df['Value'] = importances_df['0_importance'] + importances_df['1_importance'] + importances_df['2_importance'] + importances_df['3_importance'] + \
                          importances_df['4_importance'] + importances_df['5_importance'] + importances_df['6_importance'] 
importances_df = importances_df.sort_values('Value', ascending = False)
importances_df['Value'] = importances_df['Value'] / importances_df['Value'].sum()

plt.rcParams['figure.dpi'] = 600
fig = plt.figure(figsize = (5, 3), facecolor = 'white')
gs = fig.add_gridspec(1, 2)
gs.update(wspace = 0.3, hspace = 0)
ax0 = fig.add_subplot(gs[0, 0])
ax0.set_facecolor("white")
for s in ["right", "top"]:
    ax0.spines[s].set_visible(False)

ax0.text(0, -1.8, 'Feature importance ', color = 'black', fontsize = 7.3, ha = 'left', va = 'bottom', weight = 'bold')
ax0.text(0, -1.8, 'final lightGBM model', color = 'black', fontsize = 6, ha = 'left', va = 'top')
ax0_sns = sns.barplot(ax = ax0, y = importances_df['Feature'], x = importances_df['Value'], edgecolor = "black", linewidth = 0.3, color = "#225478", orient = "h")
ax0_sns.set_ylabel("Feature name", fontsize = 4.5)
ax0_sns.set_xlabel("Feature importance (7 folds, ratio of total)", fontsize = 4.5)
ax0_sns.tick_params(labelsize = 3.5)
plt.show()


# # Predict results (mean win bid of each k-model)

# In[10]:


submission = pd.DataFrame({'deviceId':test_feats['deviceId'], 'floor':test_feats['bidFloorPrice'], 'winBid':0})

submission['winBid'] = np.mean(np.column_stack(predictions), axis = 1)


# # Clip results from 0.01 and create submission file

# In[11]:


submission['winBid'] = np.clip(submission['winBid'], submission['floor'], 3800)

submission = submission[['deviceId', 'winBid']]
submission.to_csv("submission.csv", index = None)
submission

