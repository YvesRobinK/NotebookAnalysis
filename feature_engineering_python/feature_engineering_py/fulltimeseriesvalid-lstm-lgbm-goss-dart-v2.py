#!/usr/bin/env python
# coding: utf-8

# # Created by Corey Levinson

# Since around September 2018 I have been able to code my own LSTM neural networks in Keras. LSTM makes perfect sense for this competition: we have data we receive at a daily granularity, and we're just predicting 1 or 0: did the stock go up, or down (of course, we must transform this by 2*pred - 1)

# A large difficulty of this problem comes from the fact that the data frame is not static. So we can't just do standard LSTM solution. We need to write the model to work as it receives a new row each day. That's a challenge I attempt to solve in this kernel.

# In[ ]:


import pandas as pd # python dataframes
import numpy as np # python numerics
import matplotlib.pyplot as plt # python plotting
import seaborn as sns

# Keras imports
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU
from keras.layers.embeddings import Embedding

# LightGBM imports
import lightgbm as lgb

from kaggle.competitions import twosigmanews # Needed to obtain training/test data

from tqdm import tqdm
import gc


# In[ ]:


# Change DEBUG to False when you're ready, Corey.
DEBUG = False

# Change YEARMIN to change the cutoff point for your data
# All data must be greater than YEARMIN
YEARMIN = 2011


# In[ ]:


#random seeds for stochastic parts of neural network 
np.random.seed(100)
from tensorflow import set_random_seed
set_random_seed(150)


# In[ ]:


# Load in market and news data
env = twosigmanews.make_env()
(market_train, news_train) = env.get_training_data()


# # Light preprocessing:

# In[ ]:


# Require all data to be more recent than YEARMIN
market_train = market_train.loc[market_train['time'].dt.year > YEARMIN]


# In[ ]:


market_train.columns


# In[ ]:


# # Require all TARGETS be in range (-1, 1)
# market_train['returnsOpenNextMktres10'] = market_train['returnsOpenNextMktres10'].clip(-1,1)


# In[ ]:


# Are there any columns that have NA's?
# Recall Neural Networks requires all values imputed
print('MARKET TRAIN:')
for col in market_train.columns:
    print(col+' has '+str(market_train[col].isna().sum())+' NAs')
    
print('\n\nNEWS TRAIN:')

# Are there any columns that have NA's?
# Recall Neural Networks requires all values imputed
for col in news_train.columns:
    print(col+' has '+str(news_train[col].isna().sum())+' NAs')


# In[ ]:


# Not using news_train, yet
del news_train
gc.collect()


# # Four columns have NA's. Let's impute them with the median of the group

# In[ ]:


# If DEBUG, then don't read in all of the data.
if DEBUG:
    market_train = market_train.sample(50000, random_state=4)


# In[ ]:


# Attempt to impute by group by's median
market_train['returnsClosePrevMktres1'] = market_train.groupby(['assetCode'])['returnsClosePrevMktres1'].transform(lambda x: x.fillna(x.median()))
market_train['returnsOpenPrevMktres1'] = market_train.groupby(['assetCode'])['returnsOpenPrevMktres1'].transform(lambda x: x.fillna(x.median()))
market_train['returnsClosePrevMktres10'] = market_train.groupby(['assetCode'])['returnsClosePrevMktres10'].transform(lambda x: x.fillna(x.median()))
market_train['returnsOpenPrevMktres10'] = market_train.groupby(['assetCode'])['returnsOpenPrevMktres10'].transform(lambda x: x.fillna(x.median()))

# If the assetCode has no non-null values, then impute with column median
market_train = market_train.fillna(market_train.median())


# In[ ]:


market_train = market_train.sort_values(['assetCode','time']) # Sort it by time for use in LSTM later
market_train.head()


# In[ ]:


market_train.columns


# # Define the function to shape into the LSTM input

# ## For some reason, shaping the series into LSTM format using parallel processes gives no improvement in speed than just doing it sequentially on one process. I'm just leaving the parallel process here to remind myself of how to code it, and who knows, I might come back to it.

# In[ ]:


# from multiprocessing import Process, Manager

# def dothing(L, assetCode):  # the managed list `L` passed explicitly.
#     L.append(series_to_supervised(market_train.loc[market_train['assetCode']==assetCode, 'returnsOpenNextMktres10'].values.reshape(-1,1).flatten().tolist(),
#                          market_train.loc[market_train['assetCode']==assetCode, ['assetCode','time','TARGET']],
#                          n_in=7,
#                          n_out=1,
#                          dropnan=True,
#                          pad=True))

# with Manager() as manager:
#     L = manager.list()  # <-- can be shared between processes.
#     processes = []
#     for assetCode in tqdm(market_train['assetCode'].unique()):
#         p = Process(target=dothing, args=(L,assetCode))  # Passing the list
#         p.start()
#         processes.append(p)
#     for p in processes:
#         p.join()


# In[ ]:


# Define parameters for LSTM input creation
n_in = 5
n_out = 1


# In[ ]:


# Adapted from: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

def series_to_supervised(data, extraCols, n_in=1, n_out=1, dropnan=True, pad=False):
    if pad: # If you do not have enough data to construct the n_in sequence...
        data = np.asarray([data[0].tolist()]*n_in + data.tolist()) # Pad with earliest piece of data n_in amount of times
        
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    # Add extra columns
    agg = pd.concat([agg.reset_index(drop=True), extraCols.reset_index(drop=True)], axis=1)
    return agg


# # Hypothesis: I think i dont have enough RAM to construct the list. So I am reducing amount of information being fed.

# In[ ]:


market_train['time'] = pd.to_datetime(market_train['time'].dt.date) # Change from datetime to date for less memory and easier merge with news


# In[ ]:


# Feature Engineering
market_train['close_minus_open'] = market_train['close'] - market_train['open']
market_train['margin1'] = market_train['open'] / market_train['close']


# In[ ]:


# Keep last 30 of each asset
total_market_obs_df = [market_train.loc[(market_train['time'].dt.year >= 2016) & (market_train['time'].dt.month >= 9)].groupby('assetCode').tail(30).drop(['universe','returnsOpenNextMktres10'], axis=1)]


# In[ ]:


ewma = pd.Series.ewm


# In[ ]:


# Copied from https://www.kaggle.com/qqgeogor/eda-script-67
from multiprocessing import Pool

def create_lag(df_code,n_lag=[5,],shift_size=1):
    code = df_code['assetCode'].unique()
    
    for col in return_features:
        for window in n_lag:
            #rolled = df_code[col].shift(shift_size).rolling(window=window)
            rolled = df_code[col].rolling(window=window)
            lag_mean = rolled.mean()
            #lag_max = rolled.max()
            #lag_min = rolled.min()
            #lag_std = rolled.std()
            df_code['%s_lag_%s_mean'%(col,window)] = lag_mean
            #df_code['%s_lag_%s_max'%(col,window)] = lag_max
            #df_code['%s_lag_%s_min'%(col,window)] = lag_min
            #df_code['%s_lag_%s_std'%(col,window)] = lag_std

    return df_code#.fillna(-1)

def generate_lag_features(df,n_lag = [5]):
#     features = ['time', 'assetCode', 'assetName', 'volume', 'close', 'open',
#        'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
#        'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
#        'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
#        'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
#        'returnsOpenNextMktres10', 'universe']
    
    assetCodes = df['assetCode'].unique()
    #print(assetCodes)
    all_df = []
    df_codes = df.groupby('assetCode')
    df_codes = [df_code[1][['time','assetCode']+return_features] for df_code in df_codes]
    #print('total %s df'%len(df_codes))
    
    pool = Pool(4)
    all_df = pool.map(create_lag, df_codes)
    
    new_df = pd.concat(all_df)  
    new_df.drop(return_features,axis=1,inplace=True)
    pool.close()
    
    return new_df


# In[ ]:


# Creates MACD

def create_lag2(df_code,n_lag=[5,],shift_size=1):
    code = df_code['assetCode'].unique()
    
    for col in return_features:
        df_code['%s_macd'%(col)] = ewma(df_code[col], span=12).mean() - ewma(df_code[col], span=26).mean()

    return df_code#.fillna(-1)

def generate_lag_features2(df,n_lag = [5]):
    
    assetCodes = df['assetCode'].unique()
    #print(assetCodes)
    all_df = []
    df_codes = df.groupby('assetCode')
    df_codes = [df_code[1][['time','assetCode']+return_features] for df_code in df_codes]
    #print('total %s df'%len(df_codes))
    
    pool = Pool(4)
    all_df = pool.map(create_lag2, df_codes)
    
    new_df = pd.concat(all_df)  
    new_df.drop(return_features,axis=1,inplace=True)
    pool.close()
    
    return new_df


# In[ ]:


# Creates Bollinger Bands

def create_lag3(df_code,n_lag=[5,],shift_size=1):
    code = df_code['assetCode'].unique()
    
    for col in return_features:
        df_code['%s_bollingerband_high'%(col)] = df_code[col].rolling(window=7).mean() + 2 * df_code[col].rolling(window=7).std()
        df_code['%s_bollingerband_low'%(col)] = df_code[col].rolling(window=7).mean() - 2 * df_code[col].rolling(window=7).std()

    return df_code#.fillna(-1)

def generate_lag_features3(df,n_lag = [5]):
    
    assetCodes = df['assetCode'].unique()
    #print(assetCodes)
    all_df = []
    df_codes = df.groupby('assetCode')
    df_codes = [df_code[1][['time','assetCode']+return_features] for df_code in df_codes]
    #print('total %s df'%len(df_codes))
    
    pool = Pool(4)
    all_df = pool.map(create_lag3, df_codes)
    
    new_df = pd.concat(all_df)  
    new_df.drop(return_features,axis=1,inplace=True)
    pool.close()
    
    return new_df


# In[ ]:


# Create EMA

def create_lag4(df_code,n_lag=[5,],shift_size=1):
    code = df_code['assetCode'].unique()
    
    for col in return_features:
        df_code['%s_ewma'%(col)] = ewma(df_code[col], span=9).mean()

    return df_code#.fillna(-1)

def generate_lag_features4(df,n_lag = [5]):
    
    assetCodes = df['assetCode'].unique()
    #print(assetCodes)
    all_df = []
    df_codes = df.groupby('assetCode')
    df_codes = [df_code[1][['time','assetCode']+return_features] for df_code in df_codes]
    #print('total %s df'%len(df_codes))
    
    pool = Pool(4)
    all_df = pool.map(create_lag4, df_codes)
    
    new_df = pd.concat(all_df)  
    new_df.drop(return_features,axis=1,inplace=True)
    pool.close()
    
    return new_df


# In[ ]:


market_train.columns


# In[ ]:


gc.collect()


# In[ ]:


# Get Mean Features

return_features = ['volume','returnsClosePrevRaw1',
       'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
       'returnsOpenPrevMktres1', 'returnsClosePrevRaw10',
       'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
       'returnsOpenPrevMktres10', 
        'open', 'close', 
        'margin1', 'close_minus_open']
n_lag = [5]
new_df = generate_lag_features(market_train,n_lag=n_lag)
market_train = pd.merge(market_train,new_df,how='left',on=['time','assetCode'])

del new_df
gc.collect()


# In[ ]:


# Get MACD Features

return_features = ['open', 'close']
n_lag = [5]
new_df = generate_lag_features2(market_train,n_lag=n_lag)
market_train = pd.merge(market_train,new_df,how='left',on=['time','assetCode'])

del new_df
gc.collect()


# In[ ]:


# Get Bollinger Bands Features

return_features = ['open', 'close']
n_lag = [5]
new_df = generate_lag_features3(market_train,n_lag=n_lag)
market_train = pd.merge(market_train,new_df,how='left',on=['time','assetCode'])

del new_df
gc.collect()


# In[ ]:


# Get EMA Features

return_features = ['open', 'close']
n_lag = [5]
new_df = generate_lag_features4(market_train,n_lag=n_lag)
market_train = pd.merge(market_train,new_df,how='left',on=['time','assetCode'])

del new_df
gc.collect()


# In[ ]:


market_train.head(3)


# In[ ]:


market_train.tail(3)


# In[ ]:


market_train.columns


# In[ ]:


market_train['volume_diff'] = market_train['volume'] - market_train['volume_lag_5_mean']
market_train['returnsClosePrevRaw1_diff'] = market_train['returnsClosePrevRaw1'] - market_train['returnsClosePrevRaw1_lag_5_mean']
market_train['returnsOpenPrevRaw1_diff'] = market_train['returnsOpenPrevRaw1'] - market_train['returnsOpenPrevRaw1_lag_5_mean']
market_train['returnsClosePrevMktres1_diff'] = market_train['returnsClosePrevMktres1'] - market_train['returnsClosePrevMktres1_lag_5_mean']
market_train['returnsOpenPrevMktres1_diff'] = market_train['returnsOpenPrevMktres1'] - market_train['returnsOpenPrevMktres1_lag_5_mean']
market_train['returnsClosePrevRaw10_diff'] = market_train['returnsClosePrevRaw10'] - market_train['returnsClosePrevRaw10_lag_5_mean']
market_train['returnsOpenPrevRaw10_diff'] = market_train['returnsOpenPrevRaw10'] - market_train['returnsOpenPrevRaw10_lag_5_mean']
market_train['returnsClosePrevMktres10_diff'] = market_train['returnsClosePrevMktres10'] - market_train['returnsClosePrevMktres10_lag_5_mean']
market_train['returnsOpenPrevMktres10_diff'] = market_train['returnsOpenPrevMktres10'] - market_train['returnsOpenPrevMktres10_lag_5_mean']
market_train['open_diff'] = market_train['open'] - market_train['open_lag_5_mean']
market_train['close_diff'] = market_train['close'] - market_train['close_lag_5_mean']
market_train['margin1_diff'] = market_train['margin1'] - market_train['margin1_lag_5_mean']
market_train['close_minus_open_diff'] = market_train['close_minus_open'] - market_train['close_minus_open_lag_5_mean']

market_train['open_macd_diff'] = market_train['open_ewma'] - market_train['open_macd']
market_train['close_macd_diff'] = market_train['close_ewma'] - market_train['close_macd']

market_train['open_bb_high_diff'] = market_train['open'] - market_train['open_bollingerband_high']
market_train['open_bb_low_diff'] = market_train['open'] - market_train['open_bollingerband_low']

market_train['close_bb_high_diff'] = market_train['close'] - market_train['close_bollingerband_high']
market_train['close_bb_low_diff'] = market_train['close'] - market_train['close_bollingerband_low']

market_train['open_ewma_diff'] = market_train['open'] - market_train['open_ewma']
market_train['close_ewma_diff'] = market_train['close'] - market_train['close_ewma']


# In[ ]:


date = market_train.time
num_target = market_train.returnsOpenNextMktres10.astype('float32')
bin_target = (market_train.returnsOpenNextMktres10 >= 0).astype('int8')
universe = market_train.universe.astype('int8')

# Drop columns that are not features
#market_train.drop(['returnsOpenNextMktres10', 'time', 'universe', 'assetCode', 'time', 'assetName'], 
#        axis=1, inplace=True)
#gc.collect()


# In[ ]:


market_train.tail()


# In[ ]:


market_train.columns


# # Train LGBM model

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


#x_train, x_test, y_train, y_test = train_test_split(df, target,test_size=0.1)  # Split df
train_index = market_train.index[market_train['time'].dt.year < 2016].tolist()
test_index = market_train.index[(market_train['time'].dt.year >= 2016) & (market_train['universe']==1)].tolist() 


# In[ ]:


lgbm_cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1',
       'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
       'returnsOpenPrevMktres1', 'returnsClosePrevRaw10',
       'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
       'returnsOpenPrevMktres10', 'close_minus_open', 'margin1',
       'volume_lag_5_mean', 'returnsClosePrevRaw1_lag_5_mean',
       'returnsOpenPrevRaw1_lag_5_mean', 'returnsClosePrevMktres1_lag_5_mean',
       'returnsOpenPrevMktres1_lag_5_mean', 'returnsClosePrevRaw10_lag_5_mean',
       'returnsOpenPrevRaw10_lag_5_mean',
       'returnsClosePrevMktres10_lag_5_mean',
       'returnsOpenPrevMktres10_lag_5_mean', 'open_lag_5_mean',
       'close_lag_5_mean', 'margin1_lag_5_mean', 'close_minus_open_lag_5_mean',
       'open_macd', 'close_macd', 'open_bollingerband_high',
       'open_bollingerband_low', 'close_bollingerband_high',
       'close_bollingerband_low', 'open_ewma', 'close_ewma', 'volume_diff',
       'returnsClosePrevRaw1_diff', 'returnsOpenPrevRaw1_diff',
       'returnsClosePrevMktres1_diff', 'returnsOpenPrevMktres1_diff',
       'returnsClosePrevRaw10_diff', 'returnsOpenPrevRaw10_diff',
       'returnsClosePrevMktres10_diff', 'returnsOpenPrevMktres10_diff',
       'open_diff', 'close_diff', 'margin1_diff', 'close_minus_open_diff',
       'open_macd_diff', 'close_macd_diff', 'open_bb_high_diff',
       'open_bb_low_diff', 'close_bb_high_diff', 'close_bb_low_diff',
       'open_ewma_diff', 'close_ewma_diff']


# In[ ]:


d_train = lgb.Dataset(market_train.iloc[train_index].drop(['returnsOpenNextMktres10', 'time', 'universe', 'assetCode', 'time', 'assetName'], axis=1),
                      label = bin_target.iloc[train_index])

d_valid = lgb.Dataset(market_train.iloc[test_index].drop(['returnsOpenNextMktres10', 'time', 'universe', 'assetCode', 'time', 'assetName'], axis=1),
                      label = bin_target.iloc[test_index])


# In[ ]:


t_valid = date.iloc[test_index]
y_valid = num_target.iloc[test_index]
u_valid = universe.iloc[test_index]

# We will 'inject' an extra parameter in order to have access to df_valid['time'] inside sigma_score without globals
d_valid.params = {
    'extra_time': t_valid.factorize()[0],
    'mktres': y_valid.values,
    'universe': u_valid.values
}


# In[ ]:


d_valid.params


# In[ ]:


def sigma_score(preds, valid_data):
    df_time = valid_data.params['extra_time'] # array
    df_mktres = valid_data.params['mktres'] # series
    df_universe = valid_data.params['universe']

    x_t = ((preds * 2) - 1) * pd.Series(df_mktres) * pd.Series(df_universe)
    
    # Here we take advantage of the fact that `df_mktres` (used to calculate `x_t`)
    # is a pd.Series and call `group_by`
    x_t_sum = x_t.groupby(df_time).sum()
    score = x_t_sum.mean() / x_t_sum.std()

    return 'sigma_score', score, True


# In[ ]:


params = {}
#params['max_bin'] = 220
params['learning_rate'] = 0.3
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'None'         
params['feature_fraction'] = 0.80      # feature_fraction 
#params['num_leaves'] = 2583
#params['min_data'] = 213         # min_data_in_leaf
#params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
params['verbose'] = 0
params['max_depth'] = 7
params['lambda_l1'] = 3
params['lambda_l2'] = 3


# In[ ]:


corey_lgbm_model = lgb.train(params=params, 
                             train_set=d_train, 
                             num_boost_round=1000, 
                             valid_sets=d_valid,  
                             early_stopping_rounds=50, 
                             verbose_eval=10,
                             feval=sigma_score
                            )


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


feat_importance = pd.DataFrame()
feat_importance["feature"] = market_train.drop(['returnsOpenNextMktres10', 'time', 'universe', 'assetCode', 'time', 'assetName'], axis=1).columns
feat_importance["gain"] = corey_lgbm_model.feature_importance(importance_type='gain')
feat_importance.sort_values(by='gain', ascending=False, inplace=True)
plt.figure(figsize=(15,20))
ax = sns.barplot(y="feature", x="gain", data=feat_importance)


# In[ ]:


# Make validation predictions
yhat_lgbm = corey_lgbm_model.predict(market_train.iloc[test_index].drop(['returnsOpenNextMktres10', 'time', 'universe', 'assetCode', 'time', 'assetName'], axis=1))
yhat_lgbm = (yhat_lgbm * 2) - 1


# # Train a GOSS Model

# In[ ]:


params = {}
params['learning_rate'] = 0.3
params['boosting_type'] = 'goss'
params['objective'] = 'binary'
params['metric'] = 'None'          # or 'mse'
params['feature_fraction'] = 0.60     # feature_fraction 
#params['min_data'] = 200         # min_data_in_leaf
params['verbose'] = 0
params['max_depth'] = 4
#params['lambda_l1'] = 1
params['lambda_l2'] = 1


# In[ ]:


corey_goss_model = lgb.train(params=params, 
                           train_set=d_train, 
                           num_boost_round=3000, 
                           valid_sets=d_valid,  
                           early_stopping_rounds=25, 
                           verbose_eval=10,
                           feval=sigma_score
                          )


# In[ ]:


feat_importance = pd.DataFrame()
feat_importance["feature"] = market_train.drop(['returnsOpenNextMktres10', 'time', 'universe', 'assetCode', 'time', 'assetName'], axis=1).columns
feat_importance["gain"] = corey_goss_model.feature_importance(importance_type='gain')
feat_importance.sort_values(by='gain', ascending=False, inplace=True)
plt.figure(figsize=(15,20))
ax = sns.barplot(y="feature", x="gain", data=feat_importance)


# In[ ]:


# Make validation predictions
yhat_goss = corey_goss_model.predict(market_train.iloc[test_index].drop(['returnsOpenNextMktres10', 'time', 'universe', 'assetCode', 'time', 'assetName'], axis=1))
yhat_goss = (yhat_goss * 2) - 1


# # Train a DART model

# In[ ]:


params = {}
params['learning_rate'] = 0.3
params['boosting_type'] = 'dart'
params['objective'] = 'binary'
params['metric'] = 'None'          # or 'mse'
params['feature_fraction'] = 0.60     # feature_fraction 
#params['min_data'] = 200         # min_data_in_leaf
params['verbose'] = 0
params['max_depth'] = 10
#params['lambda_l1'] = 1
params['lambda_l2'] = 0.4
params['drop_rate'] = 0.2
params['drop_seed'] = 3230


# In[ ]:


corey_dart_model = lgb.train(params=params, 
                           train_set=d_train, 
                           num_boost_round=3000, 
                           valid_sets=d_valid,  
                           early_stopping_rounds=50, 
                           verbose_eval=10,
                           feval=sigma_score
                          )


# In[ ]:


feat_importance = pd.DataFrame()
feat_importance["feature"] = market_train.drop(['returnsOpenNextMktres10', 'time', 'universe', 'assetCode', 'time', 'assetName'], axis=1).columns
feat_importance["gain"] = corey_dart_model.feature_importance(importance_type='gain')
feat_importance.sort_values(by='gain', ascending=False, inplace=True)
plt.figure(figsize=(15,20))
ax = sns.barplot(y="feature", x="gain", data=feat_importance)


# In[ ]:


# Make validation predictions
yhat_dart = corey_dart_model.predict(market_train.iloc[test_index].drop(['returnsOpenNextMktres10', 'time', 'universe', 'assetCode', 'time', 'assetName'], axis=1))
yhat_dart = (yhat_dart * 2) - 1


# In[ ]:


del d_train, d_valid, params
gc.collect()


# # Begin to train LSTM model

# In[ ]:


market_train.columns


# In[ ]:


LSTM_COLUMNS_TO_USE = ['time', # Time variable is necessary
                       'assetCode', # AssetCode is necessary to perform merges/historical analysis
                       'volume','returnsOpenPrevMktres10','returnsClosePrevMktres10','returnsOpenPrevRaw10','returnsClosePrevRaw10',
                       #'open_macd_diff', # 9 day EMA minus the MACD
                       #'open_ewma_diff', # Open minus 9 day EMA
                       #'open_bb_low_diff', 'open_bb_high_diff', # Bollinger band stuff
                       #'returnsOpenPrevMktres10_diff',
                      'returnsOpenNextMktres10', # Target variable
                      'universe', # binary variable indicating if entry will be used in metric
                     ]


# In[ ]:


market_train.head()


# In[ ]:


# Drop columns not in use
market_train = market_train[LSTM_COLUMNS_TO_USE]


# In[ ]:


market_train.head()


# # Impute NA's

# In[ ]:


print('MARKET TRAIN:')
for col in market_train.columns:
    print(col+' has '+str(market_train[col].isna().sum())+' NAs')


# In[ ]:


# Attempt to impute by group by's median
#market_train['open_bb_low_diff'] = market_train.groupby(['assetCode'])['open_bb_low_diff'].transform(lambda x: x.fillna(x.median()))
#market_train['open_bb_high_diff'] = market_train.groupby(['assetCode'])['open_bb_high_diff'].transform(lambda x: x.fillna(x.median()))
#market_train['returnsOpenPrevMktres10_diff'] = market_train.groupby(['assetCode'])['returnsOpenPrevMktres10_diff'].transform(lambda x: x.fillna(x.median()))

# If the assetCode has no non-null values, then impute with column median
market_train = market_train.fillna(market_train.median())


# # Normalize inputs for NN

# In[ ]:


INFORMATION_COLS = ['assetCode','time','universe','returnsOpenNextMktres10'] # Needs to be in this order
INPUT_COLS = [f for f in market_train.columns if f not in INFORMATION_COLS]


# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
market_train[INPUT_COLS] = scaler.fit_transform(market_train[INPUT_COLS])


# In[ ]:


market_train.head()


# In[ ]:


# Create LSTM input for each assetCode individually and store in a huge list
# This takes 45-60 minutes to run, so be patient, Corey
lstm_df_list = []
for assetCode in tqdm(market_train['assetCode'].unique()):
    lstm_df_list.append(series_to_supervised(market_train.loc[market_train['assetCode']==assetCode, INPUT_COLS].values, # Input columns
                                             market_train.loc[market_train['assetCode']==assetCode, INFORMATION_COLS],  # Information columns (e.g. TARGET, assetCode, ...)
                                             n_in=n_in,
                                             n_out=n_out,
                                             dropnan=True,
                                             pad=True))


# In[ ]:


# Free up memory
del market_train
gc.collect()


# In[ ]:


# Rowbind your list of dataframes
full_lstm_df = pd.concat(lstm_df_list, axis=0).reset_index(drop=True)


# In[ ]:


# Free up memory
del lstm_df_list
gc.collect()


# In[ ]:


full_lstm_df.head()


# In[ ]:


# Convert to a TARGET 1 or 0. Did your stock increase, or not?
full_lstm_df['TARGET'] = (np.sign(full_lstm_df['returnsOpenNextMktres10']) + 1) / 2

# Need to look into: If I don't convert and instead have a tanh activation at the end, does this improve score?


# In[ ]:


full_lstm_df.head()


# # Split into train and test. Yes, even though for time series your validation should be future data, it is extremely necessary to train on the most recent data to make future predictions. Therefore, I'm opting for just a regular old train_test_split.

# In[ ]:


train = full_lstm_df.loc[full_lstm_df['time'].dt.year < 2016]
valid = full_lstm_df.loc[full_lstm_df['time'].dt.year >= 2016]

# Only select those where you have universe
valid = valid.loc[valid['universe']==1]

#train_index = full_lstm_df.index[(full_lstm_df['time'].dt.year < 2016)].tolist()
#valid_index = full_lstm_df.index[(full_lstm_df['time'].dt.year >= 2016) & (full_lstm_df['universe']==1)].tolist()



#from sklearn.model_selection import train_test_split

#train, valid = train_test_split(full_lstm_df,test_size=0.20, random_state=21)


# In[ ]:


# Save the history for recent/active stocks
active_history = full_lstm_df.loc[(full_lstm_df['time'].dt.year >= 2016) & (full_lstm_df['time'].dt.month >= 11)].groupby('assetCode').tail(1) # change to tail(4) eventually?

# Desperately need more memory
del full_lstm_df
gc.collect()


# In[ ]:


train.head()


# In[ ]:


valid_X, valid_y = valid.values[:, :-5], valid['TARGET'].values
valid_returns = valid.values[:, -2]
valid_universe = valid.values[:, -3]
valid_time = valid.values[:, -4]


# In[ ]:


del valid
gc.collect()


# In[ ]:


get_ipython().run_line_magic('whos', '')


# In[ ]:


train_X, train_y = train.values[:, :-5], train['TARGET'].values


# In[ ]:


del train
gc.collect()


# In[ ]:


# # split into input and outputs
# # Assumption: The last column is your target column
# # And the final 4 columns should not be used in training (here assetCode, time, returnsOpenNextMktres10, and TARGET)
# train_X, train_y = train.values[:, :-5], train['TARGET'].values
# valid_X, valid_y = valid.values[:, :-5], valid['TARGET'].values

USE_SIGMOID_ACTIVATION_LAYER = True

# if not USE_SIGMOID_ACTIVATION_LAYER:
#     train_y = train['returnsOpenNextMktres10'].values
#     valid_y = valid['returnsOpenNextMktres10'].values
    
# del train, valid
# gc.collect()


# In[ ]:


# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
valid_X = valid_X.reshape((valid_X.shape[0], 1, valid_X.shape[1]))
print(train_X.shape, train_y.shape, valid_X.shape, valid_y.shape)


# # Okay I'm not using LSTM I'm using GRU instead. but all my variable names are based on the word 'lstm' so deal with it =)

# In[ ]:


from keras import callbacks


# In[ ]:


# https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2

class Metrics(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_predict = (pd.DataFrame(model.predict(X_val)) * 2) - 1 # Need to convert it back to [-1, 1] instead of [0, 1]
        _sigmascore = sigma_scorelstm(y_val, y_predict)
        print(" — sigmascore: %f" % (_sigmascore))

        self._data.append({
            'val_sigmascore': _sigmascore
        })
        return

    def get_data(self):
        return self._data

metrics = Metrics()


# In[ ]:


def sigma_scorelstm(y_true, y_pred):
        x_t_i = y_pred * pd.DataFrame(valid_returns) * pd.DataFrame(valid_universe) # Multiply my confidence by return multiplied by universe
        data = pd.concat([pd.DataFrame(valid_time), x_t_i], axis=1)
        data.columns = ['day','x_t_i']
        x_t = data.groupby('day').sum().values.flatten()
        mean = np.mean(x_t)
        std = np.std(x_t)
        score_valid = mean / std
        return score_valid


# In[ ]:


# model = Sequential()
# model.add(LSTM(50, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(Dropout(0.25))
# model.add(LSTM(50, return_sequences=True))
# model.add(Dropout(0.25))
# model.add(LSTM(50))

# https://arxiv.org/ftp/arxiv/papers/1801/1801.01777.pdf
# This paper reports more success using deeper NN architectures than shallow NN
model = Sequential()
model.add(GRU(50, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.50))
#model.add(GRU(50, return_sequences=True))
#model.add(Dropout(0.50))
model.add(GRU(25, return_sequences=True))
model.add(Dropout(0.50))
#model.add(GRU(25, return_sequences=True))
#model.add(Dropout(0.50))
#model.add(GRU(10, return_sequences=True))
#model.add(Dropout(0.50))
model.add(GRU(10))
model.add(Dropout(0.50))

if USE_SIGMOID_ACTIVATION_LAYER:
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop') # RMS prop is supposed to be better for recurrent neural networks.
else:
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
# fit network
# 15 epochs yielded 0.49 sigma score
history = model.fit(train_X, train_y, epochs=15, batch_size=1028, validation_data=(valid_X, valid_y), verbose=2, shuffle=True, callbacks=[metrics])


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper right')
plt.show()


# In[ ]:


yhat_lstm = (pd.DataFrame(model.predict(valid_X)) * 2) - 1 # Need to convert it back to [-1, 1] instead of [0, 1]
# yhat.to_csv('predictions.csv', index=False)


# In[ ]:


x_t_i = yhat_lstm * pd.DataFrame(valid_returns) * pd.DataFrame(valid_universe) # Multiply my confidence by return multiplied by universe
data = pd.concat([pd.DataFrame(valid_time), x_t_i], axis=1)
data.columns = ['day','x_t_i']
x_t = data.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_valid = mean / std
print(score_valid)


# In[ ]:


# Desperately need memory. Let's delete more stuff.
del history, train_X, train_y, valid_X, valid_y
gc.collect()


# # LGBM, GOSS, DART Blend

# In[ ]:


x_t_i = pd.DataFrame((yhat_lgbm + yhat_goss + yhat_dart)/3) * pd.DataFrame(valid_returns) * pd.DataFrame(valid_universe) # Multiply my confidence by return multiplied by universe
data = pd.concat([pd.DataFrame(valid_time), x_t_i], axis=1)
data.columns = ['day','x_t_i']
x_t = data.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_valid = mean / std
print(score_valid)


# # LGBM, GOSS, DART, LSTM Blend

# In[ ]:


x_t_i = (yhat_lstm + pd.DataFrame(yhat_lgbm + yhat_goss + yhat_dart))/4 * pd.DataFrame(valid_returns) * pd.DataFrame(valid_universe) # Multiply my confidence by return multiplied by universe
data = pd.concat([pd.DataFrame(valid_time), x_t_i], axis=1)
data.columns = ['day','x_t_i']
x_t = data.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_valid = mean / std
print(score_valid)


# # Can i use Logistic Regression to improve my score? Let's try this instead of just averaging.

# In[ ]:


# Convert back to [0,1] for Logistic Regression
# Also convert to DataFrame
yhat_lgbm = pd.DataFrame((yhat_lgbm + 1) / 2)
yhat_goss = pd.DataFrame((yhat_goss + 1) / 2)
yhat_dart = pd.DataFrame((yhat_dart + 1) / 2)
yhat_lstm = (yhat_lstm + 1) / 2


# In[ ]:


truth = (np.sign(valid_returns) + 1) / 2


# In[ ]:


ensemble = pd.concat([yhat_lgbm, yhat_goss, yhat_dart, yhat_lstm], axis=1)
ensemble.columns = ['lgbm','goss','dart', 'lstm']

truth = pd.DataFrame(truth).astype('int')
truth.columns = ['y']


# In[ ]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(ensemble, truth)
logreg.score(ensemble, truth) # R squared


# # Ensemble score:

# In[ ]:


yhat_ens = (pd.DataFrame(logreg.predict_proba(ensemble)[:,1]) * 2) - 1 # Need to convert it back to [-1, 1] instead of [0, 1]

x_t_i = yhat_ens * pd.DataFrame(valid_returns) * pd.DataFrame(valid_universe) # Multiply my confidence by return multiplied by universe
data = pd.concat([pd.DataFrame(valid_time), x_t_i], axis=1)
data.columns = ['day','x_t_i']
x_t = data.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_valid = mean / std
print(score_valid)


# # Predictions:

# In[ ]:


# You can only iterate through a result from `get_prediction_days()` once
# so be careful not to lose it once you start iterating.
days = env.get_prediction_days()


# In[ ]:


# Delete universe, returnsOpenNextMktres10, TARGET. We wont have access to these in test.
del active_history['universe']
del active_history['returnsOpenNextMktres10']
del active_history['TARGET']


# In[ ]:


active_history.columns


# In[ ]:


# Correct LSTM columns to use
LSTM_COLUMNS_TO_USE = [col for col in LSTM_COLUMNS_TO_USE if (col!='universe' and col!='returnsOpenNextMktres10')]


# In[ ]:


for (market_obs_df, _, predictions_template_df) in days:
    # Delete the earliest piece of data in anticipation of the new
    
    active_history.drop([col for col in active_history.columns if '(t-'+str(n_in)+')' in col], axis=1, inplace=True)

    del active_history['time']
    
    
    #######################
    # LGBM and RF modeling:
    
    market_obs_df['time'] = pd.to_datetime(market_obs_df['time'].dt.date)
        
    # Feature Engineering
    market_obs_df['close_minus_open'] = market_obs_df['close'] - market_obs_df['open']
    market_obs_df['margin1'] = market_obs_df['open'] / market_obs_df['close']
    
    # Save to history df
    total_market_obs_df.append(market_obs_df)
    history_df = pd.concat(total_market_obs_df[-(np.max(30)+1):]) # Store last 30 for assetCodes
    
    # Get Mean Features

    return_features = ['volume','returnsClosePrevRaw1',
           'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
           'returnsOpenPrevMktres1', 'returnsClosePrevRaw10',
           'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
           'returnsOpenPrevMktres10', 
            'open', 'close', 
            'margin1', 'close_minus_open']
    n_lag = [5]
    new_df = generate_lag_features(history_df,n_lag=n_lag)
    market_obs_df = pd.merge(market_obs_df,new_df,how='left',on=['time','assetCode'])

    del new_df
    gc.collect()

    # Get MACD Features

    return_features = ['open', 'close']
    n_lag = [5]
    new_df = generate_lag_features2(history_df,n_lag=n_lag)
    market_obs_df = pd.merge(market_obs_df,new_df,how='left',on=['time','assetCode'])

    del new_df
    gc.collect()

    # Get Bollinger Bands Features

    return_features = ['open', 'close']
    n_lag = [5]
    new_df = generate_lag_features3(history_df,n_lag=n_lag)
    market_obs_df = pd.merge(market_obs_df,new_df,how='left',on=['time','assetCode'])

    del new_df
    gc.collect()

    # Get EMA Features

    return_features = ['open', 'close']
    n_lag = [5]
    new_df = generate_lag_features4(history_df,n_lag=n_lag)
    market_obs_df = pd.merge(market_obs_df,new_df,how='left',on=['time','assetCode'])

    del new_df
    gc.collect()

    market_obs_df['volume_diff'] = market_obs_df['volume'] - market_obs_df['volume_lag_5_mean']
    market_obs_df['returnsClosePrevRaw1_diff'] = market_obs_df['returnsClosePrevRaw1'] - market_obs_df['returnsClosePrevRaw1_lag_5_mean']
    market_obs_df['returnsOpenPrevRaw1_diff'] = market_obs_df['returnsOpenPrevRaw1'] - market_obs_df['returnsOpenPrevRaw1_lag_5_mean']
    market_obs_df['returnsClosePrevMktres1_diff'] = market_obs_df['returnsClosePrevMktres1'] - market_obs_df['returnsClosePrevMktres1_lag_5_mean']
    market_obs_df['returnsOpenPrevMktres1_diff'] = market_obs_df['returnsOpenPrevMktres1'] - market_obs_df['returnsOpenPrevMktres1_lag_5_mean']
    market_obs_df['returnsClosePrevRaw10_diff'] = market_obs_df['returnsClosePrevRaw10'] - market_obs_df['returnsClosePrevRaw10_lag_5_mean']
    market_obs_df['returnsOpenPrevRaw10_diff'] = market_obs_df['returnsOpenPrevRaw10'] - market_obs_df['returnsOpenPrevRaw10_lag_5_mean']
    market_obs_df['returnsClosePrevMktres10_diff'] = market_obs_df['returnsClosePrevMktres10'] - market_obs_df['returnsClosePrevMktres10_lag_5_mean']
    market_obs_df['returnsOpenPrevMktres10_diff'] = market_obs_df['returnsOpenPrevMktres10'] - market_obs_df['returnsOpenPrevMktres10_lag_5_mean']
    market_obs_df['open_diff'] = market_obs_df['open'] - market_obs_df['open_lag_5_mean']
    market_obs_df['close_diff'] = market_obs_df['close'] - market_obs_df['close_lag_5_mean']
    market_obs_df['margin1_diff'] = market_obs_df['margin1'] - market_obs_df['margin1_lag_5_mean']
    market_obs_df['close_minus_open_diff'] = market_obs_df['close_minus_open'] - market_obs_df['close_minus_open_lag_5_mean']

    market_obs_df['open_macd_diff'] = market_obs_df['open_ewma'] - market_obs_df['open_macd']
    market_obs_df['close_macd_diff'] = market_obs_df['close_ewma'] - market_obs_df['close_macd']

    market_obs_df['open_bb_high_diff'] = market_obs_df['open'] - market_obs_df['open_bollingerband_high']
    market_obs_df['open_bb_low_diff'] = market_obs_df['open'] - market_obs_df['open_bollingerband_low']

    market_obs_df['close_bb_high_diff'] = market_obs_df['close'] - market_obs_df['close_bollingerband_high']
    market_obs_df['close_bb_low_diff'] = market_obs_df['close'] - market_obs_df['close_bollingerband_low']

    market_obs_df['open_ewma_diff'] = market_obs_df['open'] - market_obs_df['open_ewma']
    market_obs_df['close_ewma_diff'] = market_obs_df['close'] - market_obs_df['close_ewma']
    
    # Make predictions with LGBM, GOSS, and DART.
    
    yhat_lgbm = pd.DataFrame( corey_lgbm_model.predict( market_obs_df[lgbm_cols]) )
    
    yhat_goss = pd.DataFrame( corey_goss_model.predict(market_obs_df[lgbm_cols]) )
    
    yhat_dart = pd.DataFrame( corey_dart_model.predict(market_obs_df[lgbm_cols]) )
    
    ##############################
    #LSTM MODEL PART:
    
    # Reverse column order to prepare for column renaming later
    active_history_cols = ['assetCode']
    for step in range(0, n_in):
        for varnum in range(1, len(INPUT_COLS)+1):
            if step==0:
                stepnum = ''
            else:
                stepnum = '-'+str(step)
            active_history_cols.append('var'+str(varnum)+'(t'+stepnum+')')
            
    active_history = active_history[active_history_cols]
    
    # Drop columns not in use
    market_obs_df = market_obs_df[LSTM_COLUMNS_TO_USE]
    
    # If there are any missing values for the new data, then impute with the median of that day.
    market_obs_df = market_obs_df.fillna(market_obs_df.median())
    
    # StandardScale it
    market_obs_df[INPUT_COLS] = scaler.transform(market_obs_df[INPUT_COLS])
    
    # If there was a column with all NA's, then fill it with 0's
    market_obs_df = market_obs_df.fillna(0)
    
    # Update your active_history to include new data
    active_history = market_obs_df.merge(right=active_history, how='outer', on='assetCode')
    
    # Rename your active_history columns as to what the NN model is expecting
    active_history_cols = ['time','assetCode']
    for step in range(0, n_in+1):
        for varnum in range(1, len(INPUT_COLS)+1):
            if step==0:
                stepnum = ''
            else:
                stepnum = '-'+str(step)
            active_history_cols.append('var'+str(varnum)+'(t'+stepnum+')')
    active_history.columns = active_history_cols
    
    # Shift values to the left if you didn't receive an update from market_obs_df
    # From https://stackoverflow.com/questions/37400246/pandas-update-multiple-columns-at-once (you need .values at the end to dismiss column name indexing)
    active_history_cols = []
    for step in range(0, n_in):
        for varnum in range(1, len(INPUT_COLS)+1):
            if step==0:
                stepnum = ''
            else:
                stepnum = '-'+str(step)
            active_history_cols.append('var'+str(varnum)+'(t'+stepnum+')')
            
    active_history_cols2 = []
    for step in range(1, n_in+1):
        for varnum in range(1, len(INPUT_COLS)+1):
            if step==0:
                stepnum = ''
            else:
                stepnum = '-'+str(step)
            active_history_cols2.append('var'+str(varnum)+'(t'+stepnum+')')

    # var(t), var(t-1), ..., var(t - (n_in) + 1) = var(t-1), var(t-2), ..., var(t-(n_in))
    active_history.loc[active_history['time'].isnull(), active_history_cols] = active_history.loc[active_history['time'].isnull(), active_history_cols2].values
    
    # Impute values with the same value over and over
    LAST_VAL_TO_BE_IMPUTED = 'var1(t-'+str(n_in)+')'
    for step in range(1, n_in+1):
        active_history.loc[active_history[LAST_VAL_TO_BE_IMPUTED].isnull(), [col for col in active_history.columns if '(t-'+str(step)+')' in col]] = active_history.loc[active_history[LAST_VAL_TO_BE_IMPUTED].isnull(), [col for col in active_history.columns if '(t)' in col]].values
    
    # Predict on this
    to_predict = active_history.loc[active_history['time'].notnull(), [col for col in active_history.columns if 'var' in col]]

    to_predict = to_predict.values
    to_predict = to_predict.reshape((to_predict.shape[0], 1, to_predict.shape[1]))
    
    # Predicting with the NN model
    yhat_lstm = pd.DataFrame(model.predict(to_predict))
    
    # Predict on Ensemble now
    ensemble = pd.concat([yhat_lgbm, yhat_goss, yhat_dart, yhat_lstm], axis=1)
    ensemble.columns = ['lgbm','goss','dart', 'lstm']

    preds = logreg.predict_proba(ensemble)[:,1]
    preds = (preds * 2) - 1 # Convert from [0,1] to [-1,1]
    
    predictions_template_df['confidenceValue'] = preds
    env.predict(predictions_template_df)


# In[ ]:


env.write_submission_file()


# In[ ]:




