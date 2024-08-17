#!/usr/bin/env python
# coding: utf-8

# # GPU accelerated solution using NVIDIA RAPIDS cudf and cuml
# # Data loading, preprocessing and feature engineering takes less than 3min in GPU.

# In[1]:


import pandas
import numpy as np
import cudf as pd
import cupy as cp

import glob
import os
import gc
import time

from joblib import Parallel, delayed

from sklearn import preprocessing, model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize


import matplotlib.pyplot as plt 
import seaborn as sns
import numpy.matlib
from catboost import Pool, CatBoostRegressor
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import cuml
from cuml.neighbors import KNeighborsRegressor
from cuml import LinearRegression
from cuml import Ridge
from cuml.ensemble import RandomForestRegressor


path_submissions = '/'

target_name = 'target'
scores_folds = {}

def convert_to_32bit(df):
    for f in df.columns:
        if df[f].dtype == 'int64':
            df[f] = df[f].astype('int32')
        if df[f].dtype == 'float64':
            df[f] = df[f].astype('float32')
    return df


# # Loading train and test sets

# In[2]:


# data directory
data_dir = '../input/optiver-realized-volatility-prediction/'

train = pd.read_csv('../input/optiver-realized-volatility-prediction/train.csv')
test = pd.read_csv('../input/optiver-realized-volatility-prediction/test.csv')

train['row_id'] = train['stock_id'].astype(str) + '-' + train['time_id'].astype(str)
test['row_id'] = test['stock_id'].astype(str) + '-' + test['time_id'].astype(str)

train['is_train'] = 1
test['is_train'] = 0

train = convert_to_32bit(train)
test = convert_to_32bit(test)

print( train.shape )
print( test.shape )


# In[3]:


print(train.head(20))


# In[4]:


print(test.head(20))


# # Checking how many stock_id there are in train and test

# In[5]:


train_stock_ids = train['stock_id'].to_pandas().unique()
test_stock_ids = test['stock_id'].to_pandas().unique()
print( 'Sizes:', len(train_stock_ids), len(test_stock_ids) )
print( 'Train stocks:', train_stock_ids )
print( 'Test stocks:', test_stock_ids )


# In[6]:


# Function to preprocess book data (for each stock id)
def book_preprocessor(file_path):
    df = pd.read_parquet(file_path)
    df = convert_to_32bit(df)
    
    # Calculate Wap
    df['wap1'] = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
    df['wap2'] = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])
    df['wap3'] = (df['bid_price1'] * df['bid_size1'] + df['ask_price1'] * df['ask_size1']) / (df['bid_size1'] + df['ask_size1'])
    df['wap4'] = (df['bid_price2'] * df['bid_size2'] + df['ask_price2'] * df['ask_size2']) / (df['bid_size2'] + df['ask_size2'])
    
    # Calculate log returns
    df['log_return1'] = df['wap1'].log()
    df['log_return1'] = df['log_return1'] - df.groupby(['time_id'])['log_return1'].shift(1).reset_index(drop=True)

    df['log_return2'] = df['wap2'].log()
    df['log_return2'] = df['log_return2'] - df.groupby(['time_id'])['log_return2'].shift(1).reset_index(drop=True)

    df['log_return3'] = df['wap3'].log()
    df['log_return3'] = df['log_return3'] - df.groupby(['time_id'])['log_return3'].shift(1).reset_index(drop=True)

    df['log_return4'] = df['wap4'].log()
    df['log_return4'] = df['log_return4'] - df.groupby(['time_id'])['log_return4'].shift(1).reset_index(drop=True)
    
    # Calculate wap balance
    df['wap_balance'] = abs(df['wap1'] - df['wap2'])
    
    # Calculate spread
    df['price_spread'] = (df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1']) / 2)
    df['price_spread2'] = (df['ask_price2'] - df['bid_price2']) / ((df['ask_price2'] + df['bid_price2']) / 2)
    df['bid_spread'] = df['bid_price1'] - df['bid_price2']
    df['ask_spread'] = df['ask_price1'] - df['ask_price2']
    df["bid_ask_spread"] = abs(df['bid_spread'] - df['ask_spread'])
    df['total_volume'] = (df['ask_size1'] + df['ask_size2']) + (df['bid_size1'] + df['bid_size2'])
    df['volume_imbalance'] = abs((df['ask_size1'] + df['ask_size2']) - (df['bid_size1'] + df['bid_size2']))
    
    df['log_return1_sqr'] = df['log_return1'] ** 2
    df['log_return2_sqr'] = df['log_return2'] ** 2
    df['log_return3_sqr'] = df['log_return3'] ** 2
    df['log_return4_sqr'] = df['log_return4'] ** 2
    # Dict for aggregations
    create_feature_dict = {
        'wap1': ['sum', 'std', 'min','max'],
        'wap2': ['sum', 'std', 'min','max'],
        'wap3': ['sum', 'std', 'min','max'],
        'wap4': ['sum', 'std', 'min','max'],
        'log_return1_sqr': ['sum', 'std', 'min','max'],
        'log_return2_sqr': ['sum', 'std', 'min','max'],
        'log_return3_sqr': ['sum', 'std', 'min','max'],
        'log_return4_sqr': ['sum', 'std', 'min','max'],
        'wap_balance': ['sum', 'mean', 'min','max'],
        'price_spread':['sum', 'mean', 'min','max'],
        'price_spread2':['sum', 'mean', 'min','max'],
        'bid_spread':['sum', 'mean', 'min','max'],
        'ask_spread':['sum', 'mean', 'min','max'],
        'total_volume':['sum', 'mean', 'min','max'],
        'volume_imbalance':['sum', 'mean', 'min','max'],
        "bid_ask_spread":['sum',  'mean', 'min','max'],
    }
    create_feature_dict_time = {
        'log_return1_sqr': ['sum', 'std', 'min','max'],
        'log_return2_sqr': ['sum', 'std', 'min','max'],
        'log_return3_sqr': ['sum', 'std', 'min','max'],
        'log_return4_sqr': ['sum', 'std', 'min','max'],
    }
    
    def get_stats_window(fe_dict,seconds_in_bucket, add_suffix = False):
        df_feature = df[df['seconds_in_bucket'] >= seconds_in_bucket].groupby(['time_id']).agg(fe_dict).reset_index()
        df_feature.columns = ['_'.join(col) for col in df_feature.columns]
        if add_suffix:
            df_feature.columns = [col + '_' + str(seconds_in_bucket) for col in df_feature.columns]
        return df_feature
    
    # Get the stats for different windows
    df_feature = get_stats_window(create_feature_dict,seconds_in_bucket = 0, add_suffix = False)
    df_feature_500 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 500, add_suffix = True)
    df_feature_400 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 400, add_suffix = True)
    df_feature_300 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 300, add_suffix = True)
    df_feature_200 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 200, add_suffix = True)
    df_feature_100 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 100, add_suffix = True)

    # Merge all
    df_feature = df_feature.merge(df_feature_500, how = 'left', left_on = 'time_id_', right_on = 'time_id__500')
    df_feature = df_feature.merge(df_feature_400, how = 'left', left_on = 'time_id_', right_on = 'time_id__400')
    df_feature = df_feature.merge(df_feature_300, how = 'left', left_on = 'time_id_', right_on = 'time_id__300')
    df_feature = df_feature.merge(df_feature_200, how = 'left', left_on = 'time_id_', right_on = 'time_id__200')
    df_feature = df_feature.merge(df_feature_100, how = 'left', left_on = 'time_id_', right_on = 'time_id__100')
    
    # Drop tmp columns
    df_feature.drop(['time_id__500','time_id__400', 'time_id__300', 'time_id__200','time_id__100'], axis = 1, inplace = True)
    
    # Create row_id so we can merge
    stock_id = file_path.split('=')[1]
    df_feature['stock_id'] = str(stock_id) + '-'

    df_feature['row_id'] = df_feature['stock_id'] + df_feature['time_id_'].astype(str)
    
    return df_feature


# In[7]:


get_ipython().run_cell_magic('time', '', "\ndef transform(df, groupby='time_id', feat='price', agg='mean' ):\n    return df.merge( \n        df.groupby(groupby)[feat].agg(agg).reset_index().rename({feat:feat+'_'+agg}, axis=1),\n        on=groupby,\n        how='left' \n    )\n\n# Function to preprocess trade data (for each stock id)\ndef trade_preprocessor(file_path):\n    df = pd.read_parquet(file_path)\n    df = convert_to_32bit(df)\n    \n    df['log_return'] = df['price'].log()\n    df['log_return'] = df['log_return'] - df.groupby(['time_id'])['log_return'].shift(1).reset_index(drop=True)\n    df['log_return_sqr'] = df['log_return'] ** 2\n    \n    df['amount']=df['price']*df['size']\n    \n    # Dict for aggregations\n    create_feature_dict = {\n        'log_return_sqr': ['sum', 'std','max', 'min'],\n        'seconds_in_bucket':['nunique','std', 'mean','max', 'min'],\n        'size':['sum', 'nunique','std','max', 'min'],\n        'order_count':['sum','nunique','max','min','std'],\n        'amount':['sum','std','max','min'],\n    }\n    create_feature_dict_time = {\n        'log_return_sqr': ['sum', 'std','max','min'],\n        'seconds_in_bucket':['nunique'],\n        'size':['sum','mean','std','min','max'],\n        'order_count':['sum','mean','std','min','max'],\n    }\n    # Function to get group stats for different windows (seconds in bucket)\n    def get_stats_window(fe_dict,seconds_in_bucket, add_suffix = False):\n        # Group by the window\n        df_feature = df[df['seconds_in_bucket'] >= seconds_in_bucket].groupby(['time_id']).agg(fe_dict).reset_index()\n        # Rename columns joining suffix\n        df_feature.columns = ['_'.join(col) for col in df_feature.columns]\n        # Add a suffix to differentiate windows\n        if add_suffix:\n            df_feature.columns = [col + '_' + str(seconds_in_bucket) for col in df_feature.columns]\n        return df_feature\n\n    # Get the stats for different windows\n    df_feature = get_stats_window(create_feature_dict,seconds_in_bucket = 0, add_suffix = False)\n    df_feature_500 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 500, add_suffix = True)\n    df_feature_400 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 400, add_suffix = True)\n    df_feature_300 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 300, add_suffix = True)\n    df_feature_200 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 200, add_suffix = True)\n    df_feature_100 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 100, add_suffix = True)\n    df = df.sort_values(['time_id','seconds_in_bucket']).reset_index(drop=True)\n    \n    df = transform(df, groupby='time_id', feat='price', agg='mean' )\n    df = transform(df, groupby='time_id', feat='price', agg='sum' )\n    df = transform(df, groupby='time_id', feat='size', agg='mean' )\n    df['price_dif'] = ((df['price'] - df.groupby(['time_id'])['price'].shift(1).reset_index(drop=True)) / df['price']).fillna(0.)\n    df['tendencyV'] = df['size'] * df['price_dif']\n    df['f_max'] = 1 * (df['price'] >= df['price_mean'])\n    df['f_min'] = 1 * (df['price'] < df['price_mean'])\n    df['df_max'] = 1 * (df['price_dif'] >= 0)\n    df['df_min'] = 1 * (df['price_dif'] < 0)\n    df['abs_dif'] = (df['price'] - df['price_mean']).abs()\n    df['price_sqr'] = df['price']**2\n    df['size_dif'] = (df['size'] - df['size_mean']).abs()\n    df['size_sqr'] = df['size']**2\n    df['iqr_p25'] = df.groupby(['time_id'])['price'].quantile(0.15).reset_index(drop=True)\n    df['iqr_p75'] = df.groupby(['time_id'])['price'].quantile(0.85).reset_index(drop=True)\n    df['iqr_p_v25'] = df.groupby(['time_id'])['size'].quantile(0.15).reset_index(drop=True)\n    df['iqr_p_v75'] = df.groupby(['time_id'])['size'].quantile(0.85).reset_index(drop=True)\n\n    dt = df.groupby('time_id')[['tendencyV','price','price_dif','f_max','f_min','df_max','df_min','abs_dif','price_sqr','size_dif','size_sqr','iqr_p25','iqr_p75','iqr_p_v25','iqr_p_v75']].agg(\n        {\n            'tendencyV':['sum','std','max', 'min'],\n            'price':['mean','std','max', 'min'],\n            'price_dif':['mean','std','max', 'min'],\n            'f_max':['mean','std','max', 'min'],\n            'f_min':['mean','std','max', 'min'],\n            'df_max':['mean','std','max', 'min'],\n            'df_min':['mean','std','max', 'min'],\n            'abs_dif':['median','std','max', 'min'],\n            'price_sqr':['sum','std','max', 'min'],\n            'size_dif':['median','std','max', 'min'],\n            'size_sqr':['sum','std','max', 'min'],\n            'iqr_p25':['mean','std','max', 'min'],\n            'iqr_p75':['mean','std','max', 'min'],\n            'iqr_p_v25':['mean','std','max', 'min'],\n            'iqr_p_v75':['mean','std','max', 'min'],\n        }\n    )\n    dt.columns = [i+'_'+j for i, j in dt.columns] \n    df_feature = df_feature.merge(dt, left_on='time_id_', right_index=True, how='left')\n    \n    # Merge all\n    df_feature = df_feature.merge(df_feature_500, how = 'left', left_on = 'time_id_', right_on = 'time_id__500')\n    df_feature = df_feature.merge(df_feature_400, how = 'left', left_on = 'time_id_', right_on = 'time_id__400')\n    df_feature = df_feature.merge(df_feature_300, how = 'left', left_on = 'time_id_', right_on = 'time_id__300')\n    df_feature = df_feature.merge(df_feature_200, how = 'left', left_on = 'time_id_', right_on = 'time_id__200')\n    df_feature = df_feature.merge(df_feature_100, how = 'left', left_on = 'time_id_', right_on = 'time_id__100')\n    \n    # Drop tmp columns\n    df_feature = df_feature.sort_values(['time_id_' ]).reset_index(drop=True)\n    \n    stock_id = file_path.split('=')[1]\n    df_feature['stock_id'] = str(stock_id) + '-'\n    df_feature['row_id'] = df_feature['stock_id'] + df_feature['time_id_'].astype(str)\n    df_feature.drop(['time_id__500','time_id__400', 'time_id__300', 'time_id__200', 'time_id_','time_id__100','stock_id'], axis = 1, inplace = True)\n\n    fnames = ['trade_' + f for f in df_feature.columns]\n    fnames[-1] = 'row_id'\n    df_feature.columns = fnames\n\n    return df_feature\n")


# # Process all train .parquet files. Create features using cudf (GPU)
# # Note cudf speed to load and apply all feature engineering in all train set stocks.

# In[8]:


get_ipython().run_cell_magic('time', '', 'DF_TRAIN = []\nfor stock_id in tqdm(train_stock_ids):\n    df_tmp = pd.merge( \n        book_preprocessor(data_dir + "book_train.parquet/stock_id=" + str(stock_id)),\n        trade_preprocessor(data_dir + "trade_train.parquet/stock_id=" + str(stock_id)),\n        on = \'row_id\',\n        how = \'left\'\n    )\n    df_tmp[\'stock_id\'] = stock_id\n    df_tmp = convert_to_32bit(df_tmp) # to save memory\n    #df_tmp.to_parquet( \'train_parquet/\'+str(stock_id)+\'.parquet\' )\n    DF_TRAIN.append(df_tmp)\n\n# Concatenate all stock_id in the same dataframe\nDF_TRAIN = pd.concat(DF_TRAIN, ignore_index=True )\n_ = gc.collect()\n\n# Flag to filter train/test rows\nDF_TRAIN[\'is_test\'] = 0\nDF_TRAIN.shape\n')


# # Process all test .parquet files. Create features using cudf (GPU)

# In[9]:


get_ipython().run_cell_magic('time', '', 'DF_TEST = []\nfor stock_id in tqdm(test_stock_ids):\n    df_tmp = pd.merge( \n        book_preprocessor(data_dir + "book_test.parquet/stock_id=" + str(stock_id)),\n        trade_preprocessor(data_dir + "trade_test.parquet/stock_id=" + str(stock_id)),\n        on = \'row_id\',\n        how = \'left\'\n    )\n    df_tmp[\'stock_id\'] = stock_id\n    df_tmp = convert_to_32bit(df_tmp) # to save memory\n    #df_tmp.to_parquet( \'test_parquet/\'+str(stock_id)+\'.parquet\' )\n    DF_TEST.append(df_tmp)\n    \n# Concatenate all stock_id in the same dataframe\nDF_TEST = pd.concat(DF_TEST, ignore_index=True )\n_ = gc.collect()\n\n# Flag to filter train/test rows\nDF_TEST[\'is_test\'] = 1\nDF_TEST.shape\n')


# In[10]:


TRAIN = pd.concat( [DF_TRAIN, DF_TEST] ).sort_values(['stock_id','time_id_']).reset_index(drop=True)

del DF_TRAIN, DF_TEST
_ = gc.collect()
TRAIN.shape


# In[11]:


get_ipython().run_cell_magic('time', '', "\ndef get_time_stock(df_):\n    vol_cols = ['log_return1_sqr_sum_500', 'log_return2_sqr_sum_500', 'log_return3_sqr_sum_500', 'log_return4_sqr_sum_500', 'trade_log_return_sqr_sum', 'trade_log_return_sqr_std', 'trade_seconds_in_bucket_nunique' ]\n\n    df = df_.copy()\n    df_stock_id = df.groupby(['stock_id'])[vol_cols].agg(['mean', 'std', 'max', 'min', ]).reset_index()\n    df_stock_id.columns = ['_'.join(col) + '_stock' for col in df_stock_id.columns]\n\n    df_time_id = df.groupby(['time_id_'])[vol_cols].agg(['mean', 'std', 'max', 'min', ]).reset_index()\n    df_time_id.columns = ['_'.join(col)+ '_time' for col in df_time_id.columns]\n    \n    df = df.merge(df_stock_id, how = 'left', left_on = ['stock_id'], right_on = ['stock_id__stock'])\n    df = df.merge(df_time_id, how = 'left', left_on = ['time_id_'], right_on = ['time_id___time'])\n    df.drop(['stock_id__stock', 'time_id___time'], axis = 1, inplace = True)\n    return df\n\nTRAIN_ = get_time_stock(TRAIN)\nTRAIN_.drop(['stock_id','time_id_'], axis = 1, inplace = True)\nprint(TRAIN_.shape)\nprint(TRAIN_.head())\n")


# In[12]:


train = train.merge(TRAIN_, on='row_id', how='left' )
test  = test.merge(TRAIN_, on='row_id', how='left' )

del TRAIN_, TRAIN
_ = gc.collect()

train.shape, test.shape


# In[13]:


train.head()


# # Now time to calculate correlation between all stock. The best way is using a correlation matrix, so first pivot all target variables by stock_id, then just calculate the correlation matrix.
# # To Find correlated stocks use Kmeans algorithm on the correlation matrix. This procedure is a bit leak because it not being processed using crossvalidation, but it won't leak much since only 6 clusters are being calculated.

# In[14]:


get_ipython().run_cell_magic('time', '', "train_p = pd.read_csv('../input/optiver-realized-volatility-prediction/train.csv')\ntrain_p = train_p.pivot(index='time_id', columns=['stock_id'], values=['target']).fillna(0.)\ncorr = train_p.corr()\n\nkm = cuml.KMeans(n_clusters=6, max_iter=2000, n_init=5).fit(corr)\ndf = pd.DataFrame( {'stock_id': [ f[1] for f in corr.columns ], 'cluster': km.labels_} )\ndf = convert_to_32bit(df)\n\ntrain = train.merge(df, on='stock_id', how='left')\ntest = test.merge(df, on='stock_id', how='left')\n\n\ndel train_p, df, corr, km\n_ = gc.collect()\n\n# Clusters found\ntrain.groupby('cluster')['time_id'].agg('count')\n")


# In[15]:


matTrain = []
matTest = []

# 6 clusters
for ind in range(train.cluster.max()+1):
    print(ind)
    newDf = train.loc[train['cluster']==ind].copy()
    newDf = newDf.groupby(['time_id']).agg('mean')
    newDf.loc[:,'stock_id'] = 127+ind
    matTrain.append ( newDf )
    
    newDf = test.loc[test['cluster']==ind].copy()
    newDf = newDf.groupby(['time_id']).agg('mean')
    newDf.loc[:,'stock_id'] = 127+ind
    matTest.append ( newDf )
    
matTrain = pd.concat(matTrain).reset_index()
matTrain.drop(columns=['target'],inplace=True)

matTest = pd.concat(matTest).reset_index()

matTrain.shape, matTest.shape


# In[16]:


matTest = pd.concat([matTest , matTrain.loc[matTrain.time_id==5]])
matTrain = matTrain.pivot(index='time_id', columns='stock_id')
matTrain.columns = [x[0]+'_stock'+str(int(x[1])) for x in matTrain.columns]
matTrain.reset_index(inplace=True)

matTest = matTest.pivot(index='time_id', columns='stock_id')
matTest.columns = [x[0]+'_stock'+str(int(x[1])) for x in matTest.columns]
matTest.reset_index(inplace=True)

matTrain.shape, matTest.shape


# In[17]:


kfeatures = [
    'time_id',
        
    'wap1_sum_stock127',
    'wap1_sum_stock128',     
    'wap1_sum_stock129',
    'wap1_sum_stock130',     
    'wap1_sum_stock131',
    'wap1_sum_stock132',
        
    'wap2_sum_stock127',
    'wap2_sum_stock128',     
    'wap2_sum_stock129',
    'wap2_sum_stock130',     
    'wap2_sum_stock131',
    'wap2_sum_stock132',
        
    'wap3_sum_stock127',
    'wap3_sum_stock128',     
    'wap3_sum_stock129',
    'wap3_sum_stock130',     
    'wap3_sum_stock131',
    'wap3_sum_stock132',
        
    'wap4_sum_stock127',
    'wap4_sum_stock128',     
    'wap4_sum_stock129',
    'wap4_sum_stock130',     
    'wap4_sum_stock131',
    'wap4_sum_stock132',
    
    'log_return1_sqr_sum_stock127',
    'log_return1_sqr_sum_stock128',     
    'log_return1_sqr_sum_stock129',
    'log_return1_sqr_sum_stock130',     
    'log_return1_sqr_sum_stock131',
    'log_return1_sqr_sum_stock132',

    'log_return2_sqr_sum_stock127',
    'log_return2_sqr_sum_stock128',     
    'log_return2_sqr_sum_stock129',
    'log_return2_sqr_sum_stock130',     
    'log_return2_sqr_sum_stock131',
    'log_return2_sqr_sum_stock132',

    'log_return3_sqr_sum_stock127',
    'log_return3_sqr_sum_stock128',     
    'log_return3_sqr_sum_stock129',
    'log_return3_sqr_sum_stock130',     
    'log_return3_sqr_sum_stock131',
    'log_return3_sqr_sum_stock132',

    'log_return4_sqr_sum_stock127',
    'log_return4_sqr_sum_stock128',     
    'log_return4_sqr_sum_stock129',
    'log_return4_sqr_sum_stock130',     
    'log_return4_sqr_sum_stock131',
    'log_return4_sqr_sum_stock132',
    
    'total_volume_sum_stock127',
    'total_volume_sum_stock128', 
    'total_volume_sum_stock129',
    'total_volume_sum_stock130', 
    'total_volume_sum_stock131',
    'total_volume_sum_stock132',
    
    'trade_size_sum_stock127',
    'trade_size_sum_stock128', 
    'trade_size_sum_stock129',
    'trade_size_sum_stock130', 
    'trade_size_sum_stock131',
    'trade_size_sum_stock132',
    
    'trade_order_count_sum_stock127',
    'trade_order_count_sum_stock128',
    'trade_order_count_sum_stock129',
    'trade_order_count_sum_stock130',
    'trade_order_count_sum_stock131',      
    'trade_order_count_sum_stock132',
    
    'price_spread_sum_stock127',
    'price_spread_sum_stock128',
    'price_spread_sum_stock129',
    'price_spread_sum_stock130',
    'price_spread_sum_stock131',   
    'price_spread_sum_stock132',
    
    'bid_spread_sum_stock127',
    'bid_spread_sum_stock128',
    'bid_spread_sum_stock129',
    'bid_spread_sum_stock130',
    'bid_spread_sum_stock131',
    'bid_spread_sum_stock132',
    
    'ask_spread_sum_stock127',
    'ask_spread_sum_stock128',
    'ask_spread_sum_stock129',
    'ask_spread_sum_stock130',
    'ask_spread_sum_stock131',   
    'ask_spread_sum_stock132',
    
    'volume_imbalance_sum_stock127',
    'volume_imbalance_sum_stock128',
    'volume_imbalance_sum_stock129',
    'volume_imbalance_sum_stock130',
    'volume_imbalance_sum_stock131',       
    'volume_imbalance_sum_stock132',
    
    'bid_ask_spread_sum_stock127',
    'bid_ask_spread_sum_stock128',
    'bid_ask_spread_sum_stock129',
    'bid_ask_spread_sum_stock130',
    'bid_ask_spread_sum_stock131',
    'bid_ask_spread_sum_stock132',
]
matTrain = convert_to_32bit(matTrain)
matTest = convert_to_32bit(matTest)

train = pd.merge(train,matTrain[kfeatures],how='left',on='time_id')
test = pd.merge(test,matTest[kfeatures],how='left',on='time_id')
_ = gc.collect()

print( train.shape, test.shape )


# In[18]:


# train=train[~(train["stock_id"]==31)].reset_index(drop=True)
# _= gc.collect()

train = convert_to_32bit(train)
test  = convert_to_32bit(test)
_= gc.collect()

train.shape, test.shape


# In[19]:


y_target = train.target.to_pandas() #need to be numpy or pandas for sklearn 
time_id = train.time_id.to_pandas()
NFOLD = 5

def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

# Target min and max values
np.min(y_target), np.max(y_target)


# # XGBoost GPU

# In[20]:


xgbtime = time.time()

# Define the custom metric to optimize
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    err = rmspe(labels, preds)
    return 'rmspe', err

def train_and_evaluate_xgb(train, test, params, colNames):
    # Sample weight
    train['target_sqr'] = 1. / (train['target'] ** 1.55 + 9e-7)    

    dtest = xgb.DMatrix(test[colNames])

    y_train = np.zeros(len(train))
    y_test = np.zeros(len(test))

    kf = GroupKFold(n_splits=NFOLD)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train, y_target, time_id)):
        print('Fold:', fold)
        dtrain = xgb.DMatrix(train.loc[train_idx, colNames], train.loc[train_idx, 'target'], weight=train.loc[train_idx, 'target_sqr'])
        dvalid = xgb.DMatrix(train.loc[valid_idx, colNames], train.loc[valid_idx, 'target'])
        model = xgb.train(
            params,
            dtrain,
            3000,
            #[(dtrain, "train"), (dvalid, "valid")],
            [(dvalid, "valid")],
            verbose_eval=250,
            early_stopping_rounds=50,
            feval=evalerror,
        )
        y_train[valid_idx] = np.clip(model.predict(dvalid), 2e-4, 0.072)
        y_test += np.clip((model.predict(dtest)), 2e-4, 0.072)
        print( 'Rmspe Fold:', rmspe(y_target[valid_idx], y_train[valid_idx]) )
    y_test /= NFOLD
    
    print( 'XGBoost Rmspe CV:', rmspe(y_target, y_train) )
    print( pandas.DataFrame.from_dict( model.get_score(), orient='index').sort_values(0, ascending=False).head(20) )
    print()
    
    del model, dtest, dtrain, dvalid
    _ = gc.collect()
    
    return y_train, y_test


colNames = [col for col in list(train.columns) if col not in {'is_train', 'time_id', 'target', 'row_id', 'target_sqr', 'is_train'}]
colNames = [col for col in colNames if col.find('min')<0 ]
params = {
        "subsample": 0.60,
        "colsample_bytree": 0.40,
        "max_depth": 6,
        "learning_rate": 0.02,
        "objective": "reg:squarederror",
        'disable_default_eval_metric': 1, # <- necessary for XGBoost to earlystop by Rmspe and not the default rmse
        "nthread": -1,
        "tree_method": "gpu_hist",
        "gpu_id": 0,
        "max_bin": 128, 
        'min_child_weight': 2,
        'reg_lambda': 0.001,
        'reg_alpha': 0.01, 
        'seed' : 2021,
    }
y_train1a, y_test1a = train_and_evaluate_xgb(train, test, params, colNames)


colNames = [col for col in list(train.columns) if col not in {'is_train', 'time_id', 'target', 'row_id', 'target_sqr', 'is_train'}]
colNames = [col for col in colNames if col.find('max')<0 ]
params = {
        "subsample": 0.85,
        "colsample_bytree": 0.25,
        "max_depth": 7,
        "learning_rate": 0.02,
        "objective": "reg:squarederror",
        'disable_default_eval_metric': 1, # <- necessary for XGBoost to earlystop by Rmspe and not the default rmse
        "nthread": -1,
        "tree_method": "gpu_hist",
        "gpu_id": 0,
        "max_bin": 128, 
        'min_child_weight': 2,
        'reg_lambda': 0.001,
        'reg_alpha': 0.01, 
        'seed' : 2022,
    }
y_train1b, y_test1b = train_and_evaluate_xgb(train, test, params, colNames)


y_train1 = 0.75*y_train1a + 0.25*y_train1b
y_test1  = 0.75*y_test1a  + 0.25*y_test1b


xgbtime = time.time() - xgbtime

print( 'XGBoost Rmspe CV:', rmspe(y_target, y_train1), 'time: ', int(xgbtime), 's', y_test1[:3] )


# In[21]:


catbtime = time.time()

def train_and_evaluate_catb(train, test, params):

    # Sample weight
    train['target_sqr'] = 1. / (train['target'] ** 1.75 + 1e-6)

    colNames = [col for col in list(train.columns) if col not in {'is_train', 'time_id', 'target', 'row_id', 'target_sqr', 'is_train'}]

    y_train = np.zeros(len(train))
    y_test = np.zeros(len(test))

    kf = GroupKFold(n_splits=NFOLD)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train, y_target, time_id)):
        print('Fold:', fold)

        model = CatBoostRegressor(
            iterations=3000,
            learning_rate=0.05,
            depth=7,
            loss_function='RMSE',
            #l2_leaf_reg = 0.001,
            #random_strength = 0.5,
            #bagging_temperature = 1.0,
            task_type="GPU",
            random_seed = 2021,
        )        
        model.fit(
            X=train.loc[train_idx, colNames].to_pandas(), y=train.loc[train_idx, 'target'].to_pandas(),
            sample_weight = train.loc[train_idx, 'target_sqr'].to_pandas(),
            eval_set = (train.loc[valid_idx, colNames].to_pandas(), train.loc[valid_idx, 'target'].to_pandas(),),
            early_stopping_rounds = 20,
            cat_features = [0],
            verbose=False)

        y_train[valid_idx] = np.clip(model.predict(train.loc[valid_idx, colNames].to_pandas()), 2e-4, 0.072)
        y_test += np.clip((model.predict(test[colNames].to_pandas())), 2e-4, 0.072)
        print( 'Catboost Rmspe Fold:', rmspe(y_target[valid_idx], y_train[valid_idx]) )        
        print()
    y_test /= NFOLD
    return y_train, y_test


y_train2, y_test2 = train_and_evaluate_catb(train, test, params)
_= gc.collect()
catbtime = time.time() - catbtime
     
print( 'Catboost Rmspe CV:', rmspe(y_target, y_train2), 'time: ', int(catbtime), 's', y_test2[:3]  )


# # LightGBM GPU

# In[22]:


lgbtime = time.time()

# Define the custom metric to optimize
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

def feval_rmspe(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'RMSPE', rmspe(y_true, y_pred), False

def train_and_evaluate_lgb(train, test, params):
    
    features = [col for col in train.columns if col not in {"time_id", "target", "target_sqr", "row_id", 'is_train'}]
    y = train['target']
    
    y_train = np.zeros(train.shape[0])
    y_test = np.zeros(test.shape[0])
    
    kf = GroupKFold(n_splits=NFOLD)
    for fold, (trn_ind, val_ind) in enumerate(kf.split(train, y_target, time_id)):
        print('Fold:', fold)
        x_train, x_val = train.iloc[trn_ind], train.iloc[val_ind]
        y_tra, y_val = y.iloc[trn_ind], y.iloc[val_ind]
        
        train_dataset = lgb.Dataset(x_train[features], y_tra, weight = (1. / (np.square(y_tra) + 1e-6)) )
        valid_dataset = lgb.Dataset(x_val[features], y_val)
        model = lgb.train(params = params,
                          num_boost_round=3000,
                          train_set = train_dataset, 
                          valid_sets = [train_dataset, valid_dataset], 
                          verbose_eval = 100,
                          early_stopping_rounds=20,
                          feval = feval_rmspe)
        
        y_train[val_ind] = np.clip(model.predict(x_val[features]), 2e-4, 0.072)
        y_test += np.clip((model.predict(test[features])), 2e-4, 0.072)        
    y_test/=NFOLD
    
    print('LightGBM Rmspe Fold:', rmspe(y_target, y_train))
    lgb.plot_importance(model,max_num_features=20)
    
    return y_train, y_test


params = {
    'objective': 'rmse',
    'boosting_type': 'gbdt',
    'max_depth': -1,
    'max_bin':255,
    'min_data_in_leaf':750,
    'learning_rate': 0.05,
    'subsample': 0.72,
    'subsample_freq': 3,
    'feature_fraction': 0.5,
    'lambda_l1': 0.5,
    'lambda_l2': 1.0,
    'categorical_column':[0],
    'seed':2021,
    'n_jobs':-1,
    'verbose': -1,
    'device': 'gpu',
    'num_gpu': 1,
    'gpu_platform_id':-1,
    'gpu_device_id':-1,
    'gpu_use_dp': False,
}

y_train3, y_test3 = train_and_evaluate_lgb(train.to_pandas(), test.to_pandas(), params)
_= gc.collect()

print( 'LightGBM Rmspe CV:', rmspe(y_target, y_train3), 'time: ', int(time.time() - lgbtime), 's', y_test3[:3]   )


# # Ensembling Time

# In[23]:


print( 'LightGBM Rmspe:', rmspe(y_target, y_train3) )
print( 'XGBoost Rmspe:', rmspe(y_target, y_train1) )
print( 'CatBoost Rmspe:', rmspe(y_target, y_train2) )


# In[24]:


def minimize_arit(W):
    ypred = W[0] * y_train1 + W[1] * y_train2 + W[2] * y_train3
    return rmspe(y_target, ypred )

W0 = minimize(minimize_arit, [1./3]*3, options={'gtol': 1e-6, 'disp': True}).x
print('Weights arit:',W0)


# In[25]:


def signed_power(var, p=2):
    return np.sign(var) * np.abs(var)**p

def minimize_geom(W):
    ypred = signed_power(y_train1, W[0]) * signed_power(y_train2, W[1]) * signed_power(y_train3, W[2])
    return rmspe(y_target, ypred)

W1 = minimize(minimize_geom, [1./3]*3, options={'gtol': 1e-6, 'disp': True}).x

print('weights geom:',W1)


# In[26]:


ypred0 = W0[0] * y_train1 + W0[1] * y_train2 + W0[2] * y_train3
print( np.min(ypred0), np.max(ypred0))

ypred1 = signed_power(y_train1, W1[0]) * signed_power(y_train2, W1[1]) * signed_power(y_train3, W1[2])
print( np.min(ypred1) , np.max(ypred1) )

print( 'Ensemble:', rmspe(y_target, np.clip((ypred0+ypred1)/2 ,0.0002, 0.071) ) )


# In[27]:


print( np.min(ypred0),np.mean(ypred0),np.max(ypred0),np.std(ypred0) )
print( np.min(ypred1),np.mean(ypred1),np.max(ypred1),np.std(ypred1) )


# In[28]:


plt.hist(ypred0, bins=100)
plt.hist(ypred1, bins=100, alpha=0.5)


# In[29]:


train['ypred'] = np.clip((ypred0+ypred1)/2 ,0.0002, 0.071)
train['error'] = (train['target'] - train['ypred']) / train['target']
train['error'] = train['error']**2

dt = train.groupby('stock_id')['error'].agg('mean').reset_index()
dt['error'] = np.sqrt(dt['error'])
dt = dt.sort_values('error', ascending=False)
dt.to_csv('error-contribution.csv', index=False)
del train['ypred'], train['error']
dt.head(10)


# In[30]:


dt.tail(10)


# In[31]:


ypred0 = W0[0] * y_test1 + W0[1] * y_test2 + W0[2] * y_test3
ypred1 = signed_power(y_test1, W1[0]) * signed_power(y_test2, W1[1]) * signed_power(y_test3, W1[2])

ypredtest = np.clip((ypred0+ypred1)/2,0.0002, 0.071)
print( ypred0[:3],  ypred1[:3], ypredtest[:3] )

test['target'] = ypredtest
test[['row_id', 'target']].to_csv('submission.csv',index = False)
test[['row_id', 'target']].head(3)


# In[ ]:





# In[ ]:




