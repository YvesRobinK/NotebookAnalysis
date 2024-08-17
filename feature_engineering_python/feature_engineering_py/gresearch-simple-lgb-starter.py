#!/usr/bin/env python
# coding: utf-8

# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/30894/logos/header.png?t=2021-09-14-17-32-48)
# 
# 
# ## About this competition
# In this competition, you'll use your machine learning expertise to **forecast short term returns in 14 popular cryptocurrencies**. We have amassed a dataset of millions of rows of high-frequency market data dating back to 2018 which you can use to build your model. Once the submission deadline has passed, **your final score will be calculated over the following 3 months using live crypto data** as it is collected.
# 
# ## Evaluation Metrics
# [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
# 
# ## Data
# This dataset contains information on historic trades for several cryptoassets, such as Bitcoin and Ethereum. Your challenge is to **predict their future returns**.
# 
# As historic cryptocurrency prices are not confidential this will be a forecasting competition using **the time series API**. Furthermore the public leaderboard targets are publicly available and are provided as part of the competition dataset. Expect to see many people submitting perfect submissions for fun. Accordingly, **THE PUBLIC LEADERBOARD FOR THIS COMPETITION IS NOT MEANINGFUL** and is only provided as a convenience for anyone who wants to test their code. The final private leaderboard will be determined using real market data gathered after the submission period closes.
# 
# ## Code Requirements
# This is a code competition! In order for the "Submit" button to be active after a commit, the following conditions must be met:
# 
# - CPU Notebook <= 9 hours run-time
# - GPU Notebook <= 9 hours run-time
# - **Internet access disabled**
# - Freely & publicly available external data is allowed, including pre-trained models
# - Submission file must be named submission.csv
# 
# So let's get the ball rolling!

# # Libraries

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc
import pathlib
from tqdm.auto import tqdm
import joblib
import pathlib
import json
import glob
import time
import datetime
from scipy import stats
from multiprocessing import Pool, cpu_count

# models
from xgboost import XGBRegressor
import lightgbm as lgb

# visualize
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib_venn import venn2, venn3
import seaborn as sns
from matplotlib import pyplot
from matplotlib.ticker import ScalarFormatter
sns.set_context("talk")
style.use('seaborn-colorblind')

import warnings
warnings.simplefilter('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Config

# In[2]:


class CFG:
    INPUT_DIR = '/kaggle/input/g-research-crypto-forecasting/'
    OUTPUT_DIR = './'
    SEED = 20211103


# # Utils
# The data is huge! We might want to reduce memory usage somehow.

# In[3]:


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
#         else:
#             df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# # Load data

# ## train.csv - The training set
# 
# - timestamp - A timestamp for the minute covered by the row.
# - Asset_ID - An ID code for the cryptoasset.
# - Count - The number of trades that took place this minute.
# - Open - The USD price at the beginning of the minute.
# - High - The highest USD price during the minute.
# - Low - The lowest USD price during the minute.
# - Close - The USD price at the end of the minute.
# - Volume - The number of cryptoasset units traded during the minute.
# - VWAP - The volume weighted average price for the minute.
# - Target - 15 minute residualized returns. See the ['Prediction and Evaluation' section of this notebook](https://www.kaggle.com/cstein06/tutorial-to-the-g-research-crypto-competition) for details of how the target is calculated.

# In[4]:


get_ipython().run_cell_magic('time', '', "\ntrain = pd.read_csv(os.path.join(CFG.INPUT_DIR, 'train.csv')).pipe(reduce_mem_usage)\nprint(train.shape)\ntrain.head()\n")


# ## asset_details.csv 
# 
# Provides the real name and of the cryptoasset for each Asset_ID and **the weight each cryptoasset receives in the metric**.
# 

# In[5]:


asset_details = pd.read_csv(os.path.join(CFG.INPUT_DIR, 'asset_details.csv'))
asset_details['Asset_ID'] = asset_details['Asset_ID'].astype(np.int8)
print(asset_details.shape)
asset_details


# ## example_sample_submission.csv
# 

# In[6]:


example_sample_submission = pd.read_csv(os.path.join(CFG.INPUT_DIR, 'example_sample_submission.csv'))
print(example_sample_submission.shape)
example_sample_submission.head()


# ## example_test.csv
# 
# An example of the data that will be delivered by the time series API. The data is just copied from train.csv.

# In[7]:


get_ipython().run_cell_magic('time', '', "\ntest_df = pd.read_csv(os.path.join(CFG.INPUT_DIR, 'example_test.csv')).pipe(reduce_mem_usage)\nprint(test_df.shape)\ntest_df.head()\n")


# ## Other files...
# 
# - gresearch_crypto - An unoptimized version of the time series API files for offline work. You may need Python 3.7 and a Linux environment to run it without errors.
# 
# - supplemental_train.csv - After the submission period is over this file's data will be replaced with cryptoasset prices from the submission period. The current copy, which is just filled approximately the right amount of data from train.csv is provided as a placeholder.

# # EDA
# 
# Some exploratory data analysis are performed here. 
# 
# You might want to check [this official tutorial: Tutorial to the G-Research Crypto Competition
# ](https://www.kaggle.com/cstein06/tutorial-to-the-g-research-crypto-competition) as well.

# In[8]:


# dataframe info
train.info()


# In[9]:


# missing values?
train.isna().sum()


# In[10]:


example_sample_submission.info()


# In[11]:


fig, ax = plt.subplots(3, 5, figsize=(20, 12), sharex=True)
ax = ax.flatten()
for i, asset in enumerate(train['Asset_ID'].unique()):
    train.query('Asset_ID == @asset')['Target'].hist(bins=30, color='k', alpha=0.7, ax=ax[i])
    asset_name = asset_details.query('Asset_ID == @asset')['Asset_Name'].values[0]
    weight = asset_details.query('Asset_ID == @asset')['Weight'].values[0]
    ax[i].set_title(f'{asset_name}\n(weight={weight})')
    
ax[-1].axis('off')
plt.tight_layout()


# # Feature Engineering
# 
# Yeah finally machine learning part!
# 
# Here we generate sets of stock price features. There are some caveats to be aware of:
# 
# - No Leak: we cannot use a feature which uses the future information (this is a forecasting task!)
# - Stationaly features: Our features have to work whenever (scales must be stationaly over the periods of time)
# 
# Also, I already add 'train' or 'validation' flag in a time-series split manner.

# In[12]:


# select train and validation period

# auxiliary function, from datetime to timestamp
totimestamp = lambda s: np.int32(time.mktime(datetime.datetime.strptime(s, "%d/%m/%Y").timetuple()))

# train_window = [totimestamp("01/05/2021"), totimestamp("30/05/2021")]
# test_window = [totimestamp("01/06/2021"), totimestamp("30/06/2021")]
train_window = [totimestamp("01/01/2018"), totimestamp("21/09/2020")]
valid_window = [totimestamp("22/09/2020"), totimestamp("21/09/2021")]

train = train.set_index("timestamp")
beg_ = train.index[0].astype('datetime64[s]')
end_ = train.index[-1].astype('datetime64[s]')
print('>> data goes from ', beg_, 'to ', end_, 'shape=', train.shape)

# drop rows without target
train.dropna(subset=['Target'], inplace=True)

# add train flag
train['train_flg'] = 1
train.loc[valid_window[0]:valid_window[1], 'train_flg'] = 0


# In[13]:


def add_asset_details(train, asset_details):
    """Add asset details to train df
    """
    return train.merge(
        asset_details
        , how='left'
        , on='Asset_ID'
    )

# merge asset_details
train = add_asset_details(train, asset_details)


# In[14]:


def get_row_feats(df):
    """Feature engineering by row
    """
    df['upper_shadow'] = df['High'] / df[['Close', 'Open']].max(axis=1)
    df['lower_shadow'] = df[['Close', 'Open']].min(axis=1) / df['Low']
    df['open2close'] = df['Close'] / df['Open']
    df['high2low'] = df['High'] / df['Low']
    mean_price = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    median_price = df[['Open', 'High', 'Low', 'Close']].median(axis=1)
    df['high2mean'] = df['High'] / mean_price
    df['low2mean'] = df['Low'] / mean_price
    df['high2median'] = df['High'] / median_price
    df['low2median'] = df['Low'] / median_price
    df['volume2count'] = df['Volume'] / (df['Count'] + 1)
    return df   
    


# In[15]:


get_ipython().run_cell_magic('time', '', '\n# feature engineering\nfeature_df = get_row_feats(train)\n\nprint(feature_df.shape)\nfeature_df.tail()\n')


# # Modeling
# 
# As a simple starter, let's use LightGBM. 
# 
# We use time-series split as the validation strategy for this forecasting task.
# 
# There are two ways to try out: full model (using all the crypto data at once) and individual model (model for each asset).

# In[16]:


target = 'Target'
drops = ['timestamp', 'Asset_Name', 'Weight', 'train_flg', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']
features = [f for f in train.columns if f not in drops + [target]]
categoricals = ['Asset_ID']

print('{:,} features: {}'.format(len(features), features))


# ## Full Model

# In[17]:


# parameters
params = {
        'n_estimators': 10000,
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'max_depth': -1,
        'learning_rate': 0.01,
        'subsample': 0.72,
        'subsample_freq': 4,
        'feature_fraction': 0.4,
        'lambda_l1': 1,
        'lambda_l2': 1,
        'seed': 46,
        }

# train (full model)
model = lgb.LGBMRegressor(**params)
model.fit(
    feature_df.query('train_flg == 1')[features],
    feature_df.query('train_flg == 1')[target].values, 
    eval_set=[(feature_df.query('train_flg == 0')[features]
               , feature_df.query('train_flg == 0')[target].values)],
    verbose=-1, 
    early_stopping_rounds=100,
    categorical_feature=categoricals,
)

# save model
joblib.dump(model, os.path.join(CFG.OUTPUT_DIR, 'lgb_model_val.pkl'))
print('lgb model saved!')

# feature importance
fi_df = pd.DataFrame()
fi_df['features'] = features
fi_df['importance'] = model.booster_.feature_importance(importance_type="gain")


# In[18]:


# plot feature importance
fig, ax = plt.subplots(1, 1, figsize=(7, 15))
sns.barplot(
    x='importance'
    , y='features'
    , data=fi_df.sort_values(by=['importance'], ascending=False)
    , ax=ax
)


# # Individual Model

# In[19]:


# train (full model)
for asset in feature_df['Asset_ID'].unique():
    model = lgb.LGBMRegressor(**params)
    model.fit(
        feature_df.query('train_flg == 1 and Asset_ID == @asset')[features],
        feature_df.query('train_flg == 1 and Asset_ID == @asset')[target].values, 
        eval_set=[(feature_df.query('train_flg == 0 and Asset_ID == @asset')[features]
                   , feature_df.query('train_flg == 0 and Asset_ID == @asset')[target].values)],
        verbose=-1, 
        early_stopping_rounds=100,
    )
    
    # save model
    asset_name = feature_df.query('Asset_ID == @asset')['Asset_Name'].values[0]
    joblib.dump(model, os.path.join(CFG.OUTPUT_DIR, 'lgb_model_{}_val.pkl'.format(asset_name)))
    print(f'lgb model for {asset_name} saved!')


# # Validation score
# Let's see how good our model is in terms of the competition metric: weighted correlation.
# 

# In[20]:


# https://stackoverflow.com/questions/38641691/weighted-correlation-coefficient-with-pandas
def m(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)

def cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)

def corr(x, y, w):
    """Weighted Correlation"""
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))

# Compute the correlation
print('FULL MODEL *******************************************')
model = joblib.load(os.path.join(CFG.OUTPUT_DIR, 'lgb_model_val.pkl'))
val_df = train.query('train_flg == 0').copy()
val_df['Prediction'] = model.predict(val_df[features])
for asset in val_df['Asset_ID'].unique():
    tmp = val_df.query('Asset_ID == @asset')
    coin = tmp['Asset_Name'].values[0]
    r = corr(tmp['Prediction'], tmp['Target'], tmp['Weight'])
    print('')
    print('- {}: Validation Score (weighted correlation) = {:.4f}'.format(coin, r))

r = corr(val_df['Prediction'], val_df['Target'], val_df['Weight'])
print('=> Overall Validation Score (weighted correlation) = {:.4f}'.format(r))


# In[21]:


print('INDIVIDUAL MODEL *******************************************')
val_df['Prediction'] = 0
for asset in val_df['Asset_ID'].unique():
    # load model
    model = joblib.load(os.path.join(CFG.OUTPUT_DIR, 'lgb_model_{}_val.pkl'.format(asset_name)))
    
    # inference
    val_df.loc[val_df['Asset_ID'] == asset, 'Prediction'] = model.predict(val_df.loc[val_df['Asset_ID'] == asset, features])
    tmp = val_df.query('Asset_ID == @asset')
    asset_name = tmp['Asset_Name'].values[0]
    r = corr(tmp['Prediction'], tmp['Target'], tmp['Weight'])
    print('')
    print('- {}: Validation Score (weighted correlation) = {:.4f}'.format(asset_name, r))
    
r = corr(val_df['Prediction'], val_df['Target'], val_df['Weight'])
print('=> Overall Validation Score (weighted correlation) = {:.4f}'.format(r))


# # Fit with all the data (no validation)
# 
# Looks like individual model suffers. So let's just stick to the full model.

# In[22]:


# train (full model)
params['n_estimators'] = 101
model = lgb.LGBMRegressor(**params)
model.fit(
    feature_df[features],
    feature_df[target].values, 
    verbose=-1, 
    categorical_feature=categoricals,
)

# save model
joblib.dump(model, 'lgb_model.pkl')
print('lgb model saved!')


# # Submission
# This competition uses Time-Series API. For details see:
# 
# https://www.kaggle.com/c/g-research-crypto-forecasting/overview/evaluation

# In[23]:


import gresearch_crypto
env = gresearch_crypto.make_env()   # initialize the environment
iter_test = env.iter_test()    # an iterator which loops over the test set and sample submission
for (test_df, sample_prediction_df) in iter_test:
    # feature engineering
    test_df = get_row_feats(test_df)
    
    # inference
    sample_prediction_df['Target'] = model.predict(test_df[features])  # make your predictions here
    
    # register your predictions
    env.predict(sample_prediction_df)   


# ALL DONE!
