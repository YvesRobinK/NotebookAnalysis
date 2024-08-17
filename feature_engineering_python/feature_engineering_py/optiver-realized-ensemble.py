#!/usr/bin/env python
# coding: utf-8

# ## Model 1: TabNet
# Inspired from: https://www.kaggle.com/chumajin/optiver-realized-tabnet-baseline

# In[1]:


import os
import glob
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import scipy as sc
from sklearn.model_selection import KFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
pd.set_option('max_columns', 300)
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

import random


# In[3]:


SEED = 2021

def random_seed(SEED):
    
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

random_seed(SEED)


# In[4]:


get_ipython().system('pip install ../input/pytorchtabnet/pytorch_tabnet-3.1.1-py3-none-any.whl')


# In[5]:


from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.metrics import Metric
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
np.random.seed(0)

import os
from pathlib import Path


# In[6]:


# data directory
data_dir = '../input/optiver-realized-volatility-prediction/'

# Function to calculate first WAP
def calc_wap1(df):
    wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap

# Function to calculate second WAP
def calc_wap2(df):
    wap = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])
    return wap

# Function to calculate the log of the return
# Remember that logb(x / y) = logb(x) - logb(y)
def log_return(series):
    return np.log(series).diff()

# Calculate the realized volatility
def realized_volatility(series):
    return np.sqrt(np.sum(series**2))

# Function to count unique elements of a series
def count_unique(series):
    return len(np.unique(series))

# Function to read our base train and test set
def read_train_test():
    train = pd.read_csv('../input/optiver-realized-volatility-prediction/train.csv')
    test = pd.read_csv('../input/optiver-realized-volatility-prediction/test.csv')
    # Create a key to merge with book and trade data
    train['row_id'] = train['stock_id'].astype(str) + '-' + train['time_id'].astype(str)
    test['row_id'] = test['stock_id'].astype(str) + '-' + test['time_id'].astype(str)
    print(f'Our training set has {train.shape[0]} rows')
    return train, test

# Function to preprocess book data (for each stock id)
def book_preprocessor(file_path):
    df = pd.read_parquet(file_path)
    # Calculate Wap
    df['wap1'] = calc_wap1(df)
    df['wap2'] = calc_wap2(df)
    # Calculate log returns
    df['log_return1'] = df.groupby(['time_id'])['wap1'].apply(log_return)
    df['log_return2'] = df.groupby(['time_id'])['wap2'].apply(log_return)
    # Calculate wap balance
    df['wap_balance'] = abs(df['wap1'] - df['wap2'])
    # Calculate spread
    df['price_spread'] = (df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1']) / 2)
    df['bid_spread'] = df['bid_price1'] - df['bid_price2']
    df['ask_spread'] = df['ask_price1'] - df['ask_price2']
    df['total_volume'] = (df['ask_size1'] + df['ask_size2']) + (df['bid_size1'] + df['bid_size2'])
    df['volume_imbalance'] = abs((df['ask_size1'] + df['ask_size2']) - (df['bid_size1'] + df['bid_size2']))
    
    # Dict for aggregations
    create_feature_dict = {
        'wap1': [np.sum, np.mean, np.std],
        'wap2': [np.sum, np.mean, np.std],
        'log_return1': [np.sum, realized_volatility, np.mean, np.std],
        'log_return2': [np.sum, realized_volatility, np.mean, np.std],
        'wap_balance': [np.sum, np.mean, np.std],
        'price_spread':[np.sum, np.mean, np.std],
        'bid_spread':[np.sum, np.mean, np.std],
        'ask_spread':[np.sum, np.mean, np.std],
        'total_volume':[np.sum, np.mean, np.std],
        'volume_imbalance':[np.sum, np.mean, np.std]
    }
    
    # Function to get group stats for different windows (seconds in bucket)
    def get_stats_window(seconds_in_bucket, add_suffix = False):
        # Group by the window
        df_feature = df[df['seconds_in_bucket'] >= seconds_in_bucket].groupby(['time_id']).agg(create_feature_dict).reset_index()
        # Rename columns joining suffix
        df_feature.columns = ['_'.join(col) for col in df_feature.columns]
        # Add a suffix to differentiate windows
        if add_suffix:
            df_feature = df_feature.add_suffix('_' + str(seconds_in_bucket))
        return df_feature
    
    # Get the stats for different windows
    df_feature = get_stats_window(seconds_in_bucket = 0, add_suffix = False)
    df_feature_450 = get_stats_window(seconds_in_bucket = 450, add_suffix = True)
    df_feature_300 = get_stats_window(seconds_in_bucket = 300, add_suffix = True)
    df_feature_150 = get_stats_window(seconds_in_bucket = 150, add_suffix = True)
    
    # Merge all
    df_feature = df_feature.merge(df_feature_450, how = 'left', left_on = 'time_id_', right_on = 'time_id__450')
    df_feature = df_feature.merge(df_feature_300, how = 'left', left_on = 'time_id_', right_on = 'time_id__300')
    df_feature = df_feature.merge(df_feature_150, how = 'left', left_on = 'time_id_', right_on = 'time_id__150')
    # Drop unnecesary time_ids
    df_feature.drop(['time_id__450', 'time_id__300', 'time_id__150'], axis = 1, inplace = True)
    
    # Create row_id so we can merge
    stock_id = file_path.split('=')[1]
    df_feature['row_id'] = df_feature['time_id_'].apply(lambda x: f'{stock_id}-{x}')
    df_feature.drop(['time_id_'], axis = 1, inplace = True)
    return df_feature

# Function to preprocess trade data (for each stock id)
def trade_preprocessor(file_path):
    df = pd.read_parquet(file_path)
    df['log_return'] = df.groupby('time_id')['price'].apply(log_return)
    
    # Dict for aggregations
    create_feature_dict = {
        'log_return':[realized_volatility],
        'seconds_in_bucket':[count_unique],
        'size':[np.sum],
        'order_count':[np.mean],
    }
    
    # Function to get group stats for different windows (seconds in bucket)
    def get_stats_window(seconds_in_bucket, add_suffix = False):
        # Group by the window
        df_feature = df[df['seconds_in_bucket'] >= seconds_in_bucket].groupby(['time_id']).agg(create_feature_dict).reset_index()
        # Rename columns joining suffix
        df_feature.columns = ['_'.join(col) for col in df_feature.columns]
        # Add a suffix to differentiate windows
        if add_suffix:
            df_feature = df_feature.add_suffix('_' + str(seconds_in_bucket))
        return df_feature
    
    # Get the stats for different windows
    df_feature = get_stats_window(seconds_in_bucket = 0, add_suffix = False)
    df_feature_450 = get_stats_window(seconds_in_bucket = 450, add_suffix = True)
    df_feature_300 = get_stats_window(seconds_in_bucket = 300, add_suffix = True)
    df_feature_150 = get_stats_window(seconds_in_bucket = 150, add_suffix = True)

    # Merge all
    df_feature = df_feature.merge(df_feature_450, how = 'left', left_on = 'time_id_', right_on = 'time_id__450')
    df_feature = df_feature.merge(df_feature_300, how = 'left', left_on = 'time_id_', right_on = 'time_id__300')
    df_feature = df_feature.merge(df_feature_150, how = 'left', left_on = 'time_id_', right_on = 'time_id__150')
    # Drop unnecesary time_ids
    df_feature.drop(['time_id__450', 'time_id__300', 'time_id__150'], axis = 1, inplace = True)
    
    df_feature = df_feature.add_prefix('trade_')
    stock_id = file_path.split('=')[1]
    df_feature['row_id'] = df_feature['trade_time_id_'].apply(lambda x:f'{stock_id}-{x}')
    df_feature.drop(['trade_time_id_'], axis = 1, inplace = True)
    return df_feature

# Function to get group stats for the stock_id and time_id
def get_time_stock(df):
    # Get realized volatility columns
    vol_cols = ['log_return1_realized_volatility', 'log_return2_realized_volatility', 'log_return1_realized_volatility_450', 'log_return2_realized_volatility_450', 
                'log_return1_realized_volatility_300', 'log_return2_realized_volatility_300', 'log_return1_realized_volatility_150', 'log_return2_realized_volatility_150', 
                'trade_log_return_realized_volatility', 'trade_log_return_realized_volatility_450', 'trade_log_return_realized_volatility_300', 'trade_log_return_realized_volatility_150']

    # Group by the stock id
    df_stock_id = df.groupby(['stock_id'])[vol_cols].agg(['mean', 'std', 'max', 'min', ]).reset_index()
    # Rename columns joining suffix
    df_stock_id.columns = ['_'.join(col) for col in df_stock_id.columns]
    df_stock_id = df_stock_id.add_suffix('_' + 'stock')

    # Group by the stock id
    df_time_id = df.groupby(['time_id'])[vol_cols].agg(['mean', 'std', 'max', 'min', ]).reset_index()
    # Rename columns joining suffix
    df_time_id.columns = ['_'.join(col) for col in df_time_id.columns]
    df_time_id = df_time_id.add_suffix('_' + 'time')
    
    # Merge with original dataframe
    df = df.merge(df_stock_id, how = 'left', left_on = ['stock_id'], right_on = ['stock_id__stock'])
    df = df.merge(df_time_id, how = 'left', left_on = ['time_id'], right_on = ['time_id__time'])
    df.drop(['stock_id__stock', 'time_id__time'], axis = 1, inplace = True)
    return df
    
# Funtion to make preprocessing function in parallel (for each stock id)
def preprocessor(list_stock_ids, is_train = True):
    
    # Parrallel for loop
    def for_joblib(stock_id):
        # Train
        if is_train:
            file_path_book = data_dir + "book_train.parquet/stock_id=" + str(stock_id)
            file_path_trade = data_dir + "trade_train.parquet/stock_id=" + str(stock_id)
        # Test
        else:
            file_path_book = data_dir + "book_test.parquet/stock_id=" + str(stock_id)
            file_path_trade = data_dir + "trade_test.parquet/stock_id=" + str(stock_id)
    
        # Preprocess book and trade data and merge them
        df_tmp = pd.merge(book_preprocessor(file_path_book), trade_preprocessor(file_path_trade), on = 'row_id', how = 'left')
        
        # Return the merge dataframe
        return df_tmp
    
    # Use parallel api to call paralle for loop
    df = Parallel(n_jobs = -1, verbose = 1)(delayed(for_joblib)(stock_id) for stock_id in list_stock_ids)
    # Concatenate all the dataframes that return from Parallel
    df = pd.concat(df, ignore_index = True)
    return df


# In[7]:


for fold in range(5):
    get_ipython().system('cp -r ../input/optiver-tabnet-50/tabnet_model_test_{str(fold)}/* .')
    get_ipython().system('zip tabnet_model_test_{str(fold)}.zip model_params.json network.pt')
    
modelpath = [os.path.join("./",s) for s in os.listdir("./") if ("zip" in s)]    


# In[8]:


train = pd.read_pickle("../input/optiver-lgbm-model/lgbm_train.pkl")  

for col in train.columns.to_list()[4:]:
    train[col] = train[col].fillna(train[col].mean())
    
scales = train.drop(['row_id', 'target', 'time_id',"stock_id"], axis = 1).columns.to_list()    

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train[scales])

le=LabelEncoder()
le.fit(train["stock_id"])


# In[9]:


# Read train and test
_, test = read_train_test()

# Get unique stock ids 
test_stock_ids = test['stock_id'].unique()

# Preprocess them using Parallel and our single stock id functions
test_ = preprocessor(test_stock_ids, is_train = False)
test = test.merge(test_, on = ['row_id'], how = 'left')

# Get group stats of time_id and stock_id
test = get_time_stock(test)

## fillna for test data ##
for col in train.columns.to_list()[4:]:
    test[col] = test[col].fillna(train[col].mean())

x_test = test.drop(['row_id', 'time_id',"stock_id"], axis = 1).values

# Transform stock id to a numeric value
x_test = scaler.transform(x_test)
X_testdf = pd.DataFrame(x_test)

X_testdf["stock_id"]=test["stock_id"]

# Label encoding
X_testdf["stock_id"] = le.transform(X_testdf["stock_id"])

x_test = X_testdf.values


# In[10]:


tabnet_params = dict(
    n_d = 32,
    n_a = 32,
    n_steps = 3,
    gamma = 1.3,
    lambda_sparse = 0,
    optimizer_fn = optim.Adam,
    optimizer_params = dict(lr = 1e-2, weight_decay = 1e-5),
    mask_type = "entmax",
    scheduler_params = dict(
        mode = "min", patience = 5, min_lr = 1e-5, factor = 0.9),
    scheduler_fn = ReduceLROnPlateau,
    seed = 42,
    #verbose = 5,
    cat_dims=[len(le.classes_)], cat_emb_dim=[10], cat_idxs=[-1] # define categorical features
)

clf = TabNetRegressor(**tabnet_params)


# In[11]:


preds=[]
for path in modelpath:
    
    clf.load_model(path)
    preds.append(clf.predict(x_test).squeeze(-1))
    
model1_predictions = np.mean(preds,axis=0)


# ## Model 2: LGBM
# Inspired from: https://www.kaggle.com/felipefonte99/optiver-lgb-with-optimized-params    

# In[12]:


import gc
import joblib


# In[13]:


MODEL_DIR = '../input/optiver-lgbm-model/'

train = pd.read_pickle(MODEL_DIR + 'lgbm_train.pkl')

train.shape


# In[14]:


# Function to read our base train and test set
def read_test():
    test = pd.read_csv('../input/optiver-realized-volatility-prediction/test.csv')
    
    # Create a key to merge with book and trade data
    test['row_id'] = test['stock_id'].astype(str) + '-' + test['time_id'].astype(str)
    
    return test

# Read test
test = read_test()

# Get unique stock ids 
test_stock_ids = test['stock_id'].unique()

# Preprocess them using Parallel and our single stock id functions
test_ = preprocessor(test_stock_ids, is_train = False)
test = test.merge(test_, on = ['row_id'], how = 'left')
test = get_time_stock(test)


# In[15]:


# Function to calculate the root mean squared percentage error
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

def evaluate(train, test):
    
    # Split features and target
    x = train.drop(['row_id', 'target', 'time_id'], axis = 1)
    y = train['target']
    x_test = test.drop(['row_id', 'time_id'], axis = 1)
    
    # Transform stock id to a numeric value
    x['stock_id'] = x['stock_id'].astype(int)
    x_test['stock_id'] = x_test['stock_id'].astype(int)
    
    # Create out of folds array
    oof_predictions = np.zeros(x.shape[0])
    
    # Create test array to store predictions
    test_predictions = np.zeros(x_test.shape[0])
    
    SEEDS = [42, 66]
    
    TEST_PREDICTIONS = np.zeros(x_test.shape[0])
    
    for seed in SEEDS:
        
        # Create out of folds array
        oof_predictions = np.zeros(x.shape[0])

        # Create test array to store predictions
        test_predictions = np.zeros(x_test.shape[0])  
        
        num_folds = 5
        
        # Create a KFold object
        kfold = KFold(n_splits = num_folds, random_state = 66, shuffle = True)
    
        # Iterate through each fold
        for fold, (trn_ind, val_ind) in enumerate(kfold.split(x)):
            
            print(f'Evaluating_seed_{seed}_fold_{fold + 1}')
            
            x_train, x_val = x.iloc[trn_ind], x.iloc[val_ind]
            y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]

            model = joblib.load(MODEL_DIR + f'lgbm_seed_{seed}_model_{fold+1}.pkl')

            # Add predictions to the out of folds array
            oof_predictions[val_ind] = model.predict(x_val)

            # Predict the test set
            test_predictions += model.predict(x_test) / num_folds

            del model
            gc.collect()

        rmspe_score = rmspe(y, oof_predictions)
        print(f'\nOur out of folds RMSPE for seed {seed} is {rmspe_score}\n')
        
        TEST_PREDICTIONS += test_predictions / len(SEEDS)
        
    # Return test predictions
    return TEST_PREDICTIONS


# In[16]:


# Traing and evaluate
model2_predictions = evaluate(train, test)


# ## Model 3: Catboost
#     
# Inspired from: https://www.kaggle.com/ramikhreas/catboost-with-optimized-params?scriptVersionId=71165561

# In[17]:


import pandas as pd
from catboost import Pool, CatBoostRegressor


# In[18]:


DATA_DIR = '../input/optiver-lgbm-model/'
MODEL_DIR = '../input/optiver-cb/'

train = pd.read_pickle(DATA_DIR + 'lgbm_train.pkl')

# Function to read our base train and test set
def read_test():
    test = pd.read_csv('../input/optiver-realized-volatility-prediction/test.csv')
    
    # Create a key to merge with book and trade data
    test['row_id'] = test['stock_id'].astype(str) + '-' + test['time_id'].astype(str)
    
    return test

# Read test
test = read_test()

# Get unique stock ids 
test_stock_ids = test['stock_id'].unique()

# Preprocess them using Parallel and our single stock id functions
test_ = preprocessor(test_stock_ids, is_train = False)
test = test.merge(test_, on = ['row_id'], how = 'left')
test = get_time_stock(test)


# In[19]:


# Function to calculate the root mean squared percentage error
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

def evaluate(train, test):
    
    # Split features and target
    x = train.drop(['row_id', 'target', 'time_id'], axis = 1)
    y = train['target']
    x_test = test.drop(['row_id', 'time_id'], axis = 1)
    
    # Transform stock id to a numeric value
    x['stock_id'] = x['stock_id'].astype(int)
    x_test['stock_id'] = x_test['stock_id'].astype(int)
    
    # Create out of folds array
    oof_predictions = np.zeros(x.shape[0])
    
    # Create test array to store predictions
    test_predictions = np.zeros(x_test.shape[0])
    
    SEEDS = [42]
    
    TEST_PREDICTIONS = np.zeros(x_test.shape[0])
    
    for seed in SEEDS:
        
        # Create out of folds array
        oof_predictions = np.zeros(x.shape[0])

        # Create test array to store predictions
        test_predictions = np.zeros(x_test.shape[0])  
        
        num_folds = 5
        
        # Create a KFold object
        kfold = KFold(n_splits = num_folds, random_state = 66, shuffle = True)
    
        # Iterate through each fold
        for fold, (trn_ind, val_ind) in enumerate(kfold.split(x)):
            
            print(f'Evaluating_seed_{seed}_fold_{fold + 1}')
            
            x_train, x_val = x.iloc[trn_ind], x.iloc[val_ind]
            y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]
            
            # Root mean squared percentage error weights
            train_pool = Pool(x_train, y_train)
            val_pool = Pool(x_val, y_val)
            test_pool = Pool(x_test) 

            model = joblib.load(MODEL_DIR + f'cb_model_{fold+1}.pkl')

            # Add predictions to the out of folds array
            oof_predictions[val_ind] = model.predict(val_pool)

            # Predict the test set
            test_predictions += model.predict(test_pool) / num_folds

            del model
            gc.collect()

        rmspe_score = rmspe(y, oof_predictions)
        print(f'\nOur out of folds RMSPE for seed {seed} is {rmspe_score}\n')
        
        TEST_PREDICTIONS += test_predictions / len(SEEDS)
        
    # Return test predictions
    return TEST_PREDICTIONS


# In[20]:


# Traing and evaluate
model3_predictions = evaluate(train, test)


# In[21]:


pd.DataFrame(np.vstack((model1_predictions, model2_predictions, model3_predictions)).transpose(), columns=['model1','model2','model3'])


# # Another LGBM

# In[22]:


from IPython.core.display import display, HTML

import pandas as pd
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import os
import gc

from joblib import Parallel, delayed

from sklearn import preprocessing, model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt 
import seaborn as sns
import numpy.matlib
from numpy.random import seed
seed(2021)
import tensorflow as tf
tf.random.set_seed(2021)
from tensorflow import keras
import numpy as np
from keras import backend as K

path_submissions = '/'

target_name = 'target'
scores_folds = {}
data_dir = '../input/optiver-realized-volatility-prediction/'

# Function to calculate first WAP
def calc_wap1(df):
    wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap

# Function to calculate second WAP
def calc_wap2(df):
    wap = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])
    return wap

# Function to calculate the log of the return
# Remember that logb(x / y) = logb(x) - logb(y)
def log_return(series):
    return np.log(series).diff()

# Calculate the realized volatility
def realized_volatility(series):
    return np.sqrt(np.sum(series**2))

# Function to count unique elements of a series
def count_unique(series):
    return len(np.unique(series))

# Function to read our base train and test set
def read_train_test():
    train = pd.read_csv('../input/optiver-realized-volatility-prediction/train.csv')
    test = pd.read_csv('../input/optiver-realized-volatility-prediction/test.csv')
    # Create a key to merge with book and trade data
    train['row_id'] = train['stock_id'].astype(str) + '-' + train['time_id'].astype(str)
    test['row_id'] = test['stock_id'].astype(str) + '-' + test['time_id'].astype(str)
    print(f'Our training set has {train.shape[0]} rows')
    return train, test

# Function to preprocess book data (for each stock id)
def book_preprocessor(file_path):
    df = pd.read_parquet(file_path)
    # Calculate Wap
    df['wap1'] = calc_wap1(df)
    df['wap2'] = calc_wap2(df)
    # Calculate log returns
    df['log_return1'] = df.groupby(['time_id'])['wap1'].apply(log_return)
    df['log_return2'] = df.groupby(['time_id'])['wap2'].apply(log_return)
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
    
    # Dict for aggregations
    create_feature_dict = {
        'wap1': [np.sum, np.mean, np.std],
        'wap2': [np.sum, np.mean, np.std],
        'log_return1': [np.sum, realized_volatility, np.mean, np.std],
        'log_return2': [np.sum, realized_volatility, np.mean, np.std],
        'wap_balance': [np.sum, np.mean, np.std],
        'price_spread':[np.sum, np.mean, np.std],
        'price_spread2':[np.sum, np.mean, np.std],
        'bid_spread':[np.sum, np.mean, np.std],
        'ask_spread':[np.sum, np.mean, np.std],
        'total_volume':[np.sum, np.mean, np.std],
        'volume_imbalance':[np.sum, np.mean, np.std],
        "bid_ask_spread":[np.sum, np.mean, np.std],
    }
    
    # Function to get group stats for different windows (seconds in bucket)
    def get_stats_window(seconds_in_bucket, add_suffix = False):
        # Group by the window
        df_feature = df[df['seconds_in_bucket'] >= seconds_in_bucket].groupby(['time_id']).agg(create_feature_dict).reset_index()
        # Rename columns joining suffix
        df_feature.columns = ['_'.join(col) for col in df_feature.columns]
        # Add a suffix to differentiate windows
        if add_suffix:
            df_feature = df_feature.add_suffix('_' + str(seconds_in_bucket))
        return df_feature
    
    # Get the stats for different windows
    df_feature = get_stats_window(seconds_in_bucket = 0, add_suffix = False)
    df_feature_450 = get_stats_window(seconds_in_bucket = 450, add_suffix = True)
#     df_feature_500 = get_stats_window(seconds_in_bucket = 500, add_suffix = True)
#     df_feature_400 = get_stats_window(seconds_in_bucket = 400, add_suffix = True)
    df_feature_300 = get_stats_window(seconds_in_bucket = 300, add_suffix = True)
#     df_feature_200 = get_stats_window(seconds_in_bucket = 200, add_suffix = True)
    df_feature_150 = get_stats_window(seconds_in_bucket = 150, add_suffix = True)

    # Merge all
    df_feature = df_feature.merge(df_feature_450, how = 'left', left_on = 'time_id_', right_on = 'time_id__450')
    df_feature = df_feature.merge(df_feature_300, how = 'left', left_on = 'time_id_', right_on = 'time_id__300')
#     df_feature = df_feature.merge(df_feature_300, how = 'left', left_on = 'time_id_', right_on = 'time_id__300')
    df_feature = df_feature.merge(df_feature_150, how = 'left', left_on = 'time_id_', right_on = 'time_id__150')
#     df_feature = df_feature.merge(df_feature_100, how = 'left', left_on = 'time_id_', right_on = 'time_id__100')
    # Drop unnecesary time_ids
    df_feature.drop(['time_id__450', 'time_id__300', 'time_id__150'], axis = 1, inplace = True)
    
    
    # Create row_id so we can merge
    stock_id = file_path.split('=')[1]
    df_feature['row_id'] = df_feature['time_id_'].apply(lambda x: f'{stock_id}-{x}')
    df_feature.drop(['time_id_'], axis = 1, inplace = True)
    return df_feature

# Function to preprocess trade data (for each stock id)
def trade_preprocessor(file_path):
    df = pd.read_parquet(file_path)
    df['log_return'] = df.groupby('time_id')['price'].apply(log_return)
    
    # Dict for aggregations
    create_feature_dict = {
        'log_return':[realized_volatility],
        'seconds_in_bucket':[count_unique],
        'size':[np.sum],
        'order_count':[np.mean],
    }
    
    # Function to get group stats for different windows (seconds in bucket)
    def get_stats_window(seconds_in_bucket, add_suffix = False):
        # Group by the window
        df_feature = df[df['seconds_in_bucket'] >= seconds_in_bucket].groupby(['time_id']).agg(create_feature_dict).reset_index()
        # Rename columns joining suffix
        df_feature.columns = ['_'.join(col) for col in df_feature.columns]
        # Add a suffix to differentiate windows
        if add_suffix:
            df_feature = df_feature.add_suffix('_' + str(seconds_in_bucket))
        return df_feature
    
    # Get the stats for different windows
    df_feature = get_stats_window(seconds_in_bucket = 0, add_suffix = False)
    df_feature_450 = get_stats_window(seconds_in_bucket = 450, add_suffix = True)
#     df_feature_500 = get_stats_window(seconds_in_bucket = 500, add_suffix = True)
#     df_feature_400 = get_stats_window(seconds_in_bucket = 400, add_suffix = True)
    df_feature_300 = get_stats_window(seconds_in_bucket = 300, add_suffix = True)
#     df_feature_200 = get_stats_window(seconds_in_bucket = 200, add_suffix = True)
    df_feature_150 = get_stats_window(seconds_in_bucket = 150, add_suffix = True)

    # Merge all
    df_feature = df_feature.merge(df_feature_450, how = 'left', left_on = 'time_id_', right_on = 'time_id__450')
    df_feature = df_feature.merge(df_feature_300, how = 'left', left_on = 'time_id_', right_on = 'time_id__300')
#     df_feature = df_feature.merge(df_feature_300, how = 'left', left_on = 'time_id_', right_on = 'time_id__300')
    df_feature = df_feature.merge(df_feature_150, how = 'left', left_on = 'time_id_', right_on = 'time_id__150')
#     df_feature = df_feature.merge(df_feature_100, how = 'left', left_on = 'time_id_', right_on = 'time_id__100')
    # Drop unnecesary time_ids
    df_feature.drop(['time_id__450', 'time_id__300', 'time_id__150'], axis = 1, inplace = True)
    
    
    
    df_feature = df_feature.add_prefix('trade_')
    stock_id = file_path.split('=')[1]
    df_feature['row_id'] = df_feature['trade_time_id_'].apply(lambda x:f'{stock_id}-{x}')
    df_feature.drop(['trade_time_id_'], axis = 1, inplace = True)
    return df_feature

# Function to get group stats for the stock_id and time_id
def get_time_stock(df):
    # Get realized volatility columns
    vol_cols = ['log_return1_realized_volatility', 'log_return2_realized_volatility', 'log_return1_realized_volatility_450', 'log_return2_realized_volatility_450', 
                'log_return1_realized_volatility_300', 'log_return2_realized_volatility_300', 'log_return1_realized_volatility_150', 'log_return2_realized_volatility_150', 
                'trade_log_return_realized_volatility', 'trade_log_return_realized_volatility_450', 'trade_log_return_realized_volatility_300', 'trade_log_return_realized_volatility_150']
#     vol_cols = ['log_return1_realized_volatility', 'log_return2_realized_volatility',
#                 'log_return1_realized_volatility_600', 'log_return2_realized_volatility_600', 
#                 'log_return1_realized_volatility_400', 'log_return2_realized_volatility_400',
# #                 'log_return1_realized_volatility_300', 'log_return2_realized_volatility_300', 
#                 'log_return1_realized_volatility_200', 'log_return2_realized_volatility_200',
# #                 'log_return1_realized_volatility_100', 'log_return2_realized_volatility_100', 
#                 'trade_log_return_realized_volatility',
#                 'trade_log_return_realized_volatility_600', 
#                 'trade_log_return_realized_volatility_400',
# #                 'trade_log_return_realized_volatility_300',
# #                 'trade_log_return_realized_volatility_100',
#                 'trade_log_return_realized_volatility_200']

    # Group by the stock id
    df_stock_id = df.groupby(['stock_id'])[vol_cols].agg(['mean', 'std', 'max', 'min', ]).reset_index()
    # Rename columns joining suffix
    df_stock_id.columns = ['_'.join(col) for col in df_stock_id.columns]
    df_stock_id = df_stock_id.add_suffix('_' + 'stock')

    # Group by the stock id
    df_time_id = df.groupby(['time_id'])[vol_cols].agg(['mean', 'std', 'max', 'min', ]).reset_index()
    # Rename columns joining suffix
    df_time_id.columns = ['_'.join(col) for col in df_time_id.columns]
    df_time_id = df_time_id.add_suffix('_' + 'time')
    
    # Merge with original dataframe
    df = df.merge(df_stock_id, how = 'left', left_on = ['stock_id'], right_on = ['stock_id__stock'])
    df = df.merge(df_time_id, how = 'left', left_on = ['time_id'], right_on = ['time_id__time'])
    df.drop(['stock_id__stock', 'time_id__time'], axis = 1, inplace = True)
    return df
    
# Funtion to make preprocessing function in parallel (for each stock id)
def preprocessor(list_stock_ids, is_train = True):
    
    # Parrallel for loop
    def for_joblib(stock_id):
        # Train
        if is_train:
            file_path_book = data_dir + "book_train.parquet/stock_id=" + str(stock_id)
            file_path_trade = data_dir + "trade_train.parquet/stock_id=" + str(stock_id)
        # Test
        else:
            file_path_book = data_dir + "book_test.parquet/stock_id=" + str(stock_id)
            file_path_trade = data_dir + "trade_test.parquet/stock_id=" + str(stock_id)
    
        # Preprocess book and trade data and merge them
        df_tmp = pd.merge(book_preprocessor(file_path_book), trade_preprocessor(file_path_trade), on = 'row_id', how = 'left')
        
        # Return the merge dataframe
        return df_tmp
    
    # Use parallel api to call paralle for loop
    df = Parallel(n_jobs = -1, verbose = 1)(delayed(for_joblib)(stock_id) for stock_id in list_stock_ids)
    # Concatenate all the dataframes that return from Parallel
    df = pd.concat(df, ignore_index = True)
    return df

# Function to calculate the root mean squared percentage error
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

# Function to early stop with root mean squared percentage error
def feval_rmspe(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'RMSPE', rmspe(y_true, y_pred), False
# Read train and test
train, test = read_train_test()

# Get unique stock ids 
train_stock_ids = train['stock_id'].unique()
# Preprocess them using Parallel and our single stock id functions
train_ = preprocessor(train_stock_ids, is_train = True)
train = train.merge(train_, on = ['row_id'], how = 'left')

# Get unique stock ids 
test_stock_ids = test['stock_id'].unique()
# Preprocess them using Parallel and our single stock id functions
test_ = preprocessor(test_stock_ids, is_train = False)
test = test.merge(test_, on = ['row_id'], how = 'left')

# Get group stats of time_id and stock_id
train = get_time_stock(train)
test = get_time_stock(test)
out_train = pd.read_csv('../input/optiver-realized-volatility-prediction/train.csv')
out_train = out_train.pivot(index='time_id', columns='stock_id', values='target')

#out_train[out_train.isna().any(axis=1)]
out_train = out_train.fillna(out_train.mean())
out_train.head()

# code to add the just the read data after first execution

# data separation based on knn ++
nfolds = 5 # number of folds
index = []
totDist = []
values = []
# generates a matriz with the values of 
mat = out_train.values

scaler = MinMaxScaler(feature_range=(-1, 1))
mat = scaler.fit_transform(mat)

nind = int(mat.shape[0]/nfolds) # number of individuals

# adds index in the last column
mat = np.c_[mat,np.arange(mat.shape[0])]


lineNumber = np.random.choice(np.array(mat.shape[0]), size=nfolds, replace=False)

lineNumber = np.sort(lineNumber)[::-1]

for n in range(nfolds):
    totDist.append(np.zeros(mat.shape[0]-nfolds))

# saves index
for n in range(nfolds):
    
    values.append([lineNumber[n]])    


s=[]
for n in range(nfolds):
    s.append(mat[lineNumber[n],:])
    
    mat = np.delete(mat, obj=lineNumber[n], axis=0)

for n in range(nind-1):    

    luck = np.random.uniform(0,1,nfolds)
    
    for cycle in range(nfolds):
         # saves the values of index           

        s[cycle] = np.matlib.repmat(s[cycle], mat.shape[0], 1)

        sumDist = np.sum( (mat[:,:-1] - s[cycle][:,:-1])**2 , axis=1)   
        totDist[cycle] += sumDist        
                
        # probabilities
        f = totDist[cycle]/np.sum(totDist[cycle]) # normalizing the totdist
        j = 0
        kn = 0
        for val in f:
            j += val        
            if (j > luck[cycle]): # the column was selected
                break
            kn +=1
        lineNumber[cycle] = kn
        
        # delete line of the value added    
        for n_iter in range(nfolds):
            
            totDist[n_iter] = np.delete(totDist[n_iter],obj=lineNumber[cycle], axis=0)
            j= 0
        
        s[cycle] = mat[lineNumber[cycle],:]
        values[cycle].append(int(mat[lineNumber[cycle],-1]))
        mat = np.delete(mat, obj=lineNumber[cycle], axis=0)


for n_mod in range(nfolds):
    values[n_mod] = out_train.index[values[n_mod]]
def root_mean_squared_per_error(y_true, y_pred):
         return K.sqrt(K.mean(K.square( (y_true - y_pred)/ y_true )))
    
es = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=20, verbose=0,
    mode='min',restore_best_weights=True)

plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=7, verbose=0,
    mode='min')
colNames = list(train)

colNames.remove('time_id')
colNames.remove('target')
colNames.remove('row_id')
colNames.remove('stock_id')


train.replace([np.inf, -np.inf], np.nan,inplace=True)
test.replace([np.inf, -np.inf], np.nan,inplace=True)
qt_train = []

for col in colNames:
    #print(col)
    qt = QuantileTransformer(random_state=21,n_quantiles=200, output_distribution='normal')
    train[col] = qt.fit_transform(train[[col]])
    test[col] = qt.transform(test[[col]])    
    qt_train.append(qt)
from keras.backend import sigmoid
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
get_custom_objects().update({'swish': Activation(swish)})
hidden_units = (128,64,32)
stock_embedding_size = 24

cat_data = train['stock_id']

def base_model():
    
    # Each instance will consist of two inputs: a single user id, and a single movie id
    stock_id_input = keras.Input(shape=(1,), name='stock_id')
    num_input = keras.Input(shape=(264,), name='num_data')


    #embedding, flatenning and concatenating
    stock_embedded = keras.layers.Embedding(max(cat_data)+1, stock_embedding_size, 
                                           input_length=1, name='stock_embedding')(stock_id_input)
    stock_flattened = keras.layers.Flatten()(stock_embedded)
    out = keras.layers.Concatenate()([stock_flattened, num_input])
    
    # Add one or more hidden layers
    for n_hidden in hidden_units:

        out = keras.layers.Dense(n_hidden, activation='swish')(out)
        

    #out = keras.layers.Concatenate()([out, num_input])

    # A single output: our predicted rating
    out = keras.layers.Dense(1, activation='linear', name='prediction')(out)
    
    model = keras.Model(
    inputs = [stock_id_input, num_input],
    outputs = out,
    )
    
    return model
model_name = 'NN'
pred_name = 'pred_{}'.format(model_name)

n_folds = 5
kf = model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=2020)
scores_folds[model_name] = []
counter = 1

features_to_consider = list(train)

features_to_consider.remove('time_id')
features_to_consider.remove('target')
features_to_consider.remove('row_id')
#features_to_consider.remove('pred_NN')


train[features_to_consider] = train[features_to_consider].fillna(train[features_to_consider].mean())
test[features_to_consider] = test[features_to_consider].fillna(train[features_to_consider].mean())

train[pred_name] = 0
test['target'] = 0


for n_count in range(n_folds):
    print('CV {}/{}'.format(counter, n_folds))
    
    indexes = np.arange(nfolds).astype(int)    
    indexes = np.delete(indexes,obj=n_count, axis=0) 
    
    indexes = np.r_[values[indexes[0]],values[indexes[1]],values[indexes[2]],values[indexes[3]]]
    
    X_train = train.loc[train.time_id.isin(indexes), features_to_consider]
    y_train = train.loc[train.time_id.isin(indexes), target_name]
    X_test = train.loc[train.time_id.isin(values[n_count]), features_to_consider]
    y_test = train.loc[train.time_id.isin(values[n_count]), target_name]
    
    #############################################################################################
    # NN
    #############################################################################################
    
    model = base_model()
    
    model.compile(
        keras.optimizers.Adam(learning_rate=0.005),
        loss=root_mean_squared_per_error
    )
    
    try:
        features_to_consider.remove('stock_id')
    except:
        pass
    
    num_data = X_train[features_to_consider]
    
    scaler = MinMaxScaler(feature_range=(-1, 1))         
    num_data = scaler.fit_transform(num_data.values)    
    
    cat_data = X_train['stock_id']    
    target =  y_train
    
    num_data_test = X_test[features_to_consider]
    num_data_test = scaler.transform(num_data_test.values)
    cat_data_test = X_test['stock_id']

    model.fit([cat_data, num_data], 
              target,               
              batch_size=1024,
              epochs=1000,
              validation_data=([cat_data_test, num_data_test], y_test),
              callbacks=[es, plateau],
              validation_batch_size=len(y_test),
              shuffle=True,
             verbose = 1)

    preds = model.predict([cat_data_test, num_data_test]).reshape(1,-1)[0]
    
    score = round(rmspe(y_true = y_test, y_pred = preds),5)
    print('Fold {} {}: {}'.format(counter, model_name, score))
    scores_folds[model_name].append(score)
    
    tt =scaler.transform(test[features_to_consider].values)
    test[target_name] += model.predict([test['stock_id'], tt]).reshape(1,-1)[0].clip(0,1e10)
    #test[target_name] += model.predict([test['stock_id'], test[features_to_consider]]).reshape(1,-1)[0].clip(0,1e10)
       
    counter += 1
    features_to_consider.append('stock_id')
test[target_name] = test[target_name]/n_folds
nn_pre = test[target_name]
score = round(rmspe(y_true = train[target_name].values, y_pred = train[pred_name].values),5)
print('RMSPE {}: {} - Folds: {}'.format(model_name, score, scores_folds[model_name]))

from sklearn.model_selection import KFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
pd.set_option('max_columns', 300)

def feval_rmspe(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'RMSPE', rmspe(y_true, y_pred), False
def train_and_evaluate(train, test):
    # Hyperparammeters (optimized)
    seed = 2021
    params = {
        'learning_rate': 0.1,        
        'lambda_l1': 2,
        'lambda_l2': 7,
        'num_leaves': 800,
        'min_sum_hessian_in_leaf': 20,
        'feature_fraction': 0.8,
        'feature_fraction_bynode': 0.8,
        'bagging_fraction': 0.9,
        'bagging_freq': 42,
        'min_data_in_leaf': 700,
        'max_depth': 4,
        'seed': seed,
        'feature_fraction_seed': seed,
        'bagging_seed': seed,
        'drop_seed': seed,
        'data_random_seed': seed,
        'objective': 'rmse',
        'boosting': 'gbdt',
        'verbosity': -1,
        'n_jobs': -1,
    }   
    
    # Split features and target
    x = train.drop(['row_id', 'target', 'time_id'], axis = 1)
    y = train['target']
    x_test = test.drop(['row_id', 'time_id'], axis = 1)
    # Transform stock id to a numeric value
    x['stock_id'] = x['stock_id'].astype(int)
    x_test['stock_id'] = x_test['stock_id'].astype(int)
    
    # Create out of folds array
    oof_predictions = np.zeros(x.shape[0])
    # Create test array to store predictions
    test_predictions = np.zeros(x_test.shape[0])
    # Create a KFold object
    kfold = KFold(n_splits = 10, random_state = 1111, shuffle = True)
    # Iterate through each fold
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(x)):
        print(f'Training fold {fold + 1}')
        x_train, x_val = x.iloc[trn_ind], x.iloc[val_ind]
        y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]
        # Root mean squared percentage error weights
        train_weights = 1 / np.square(y_train)
        val_weights = 1 / np.square(y_val)
        train_dataset = lgb.Dataset(x_train, y_train, weight = train_weights, categorical_feature = ['stock_id'])
        val_dataset = lgb.Dataset(x_val, y_val, weight = val_weights, categorical_feature = ['stock_id'])
        model = joblib.load('../input/stock-embedding-ffnn-lgbm-training/'+f'model_fold{fold}.pkl')
        plt.figure(figsize=(12,6))
        lgb.plot_importance(model, max_num_features=10)
        plt.title("Feature importance")
        plt.show()
        # Add predictions to the out of folds array
        oof_predictions[val_ind] = model.predict(x_val)
        # Predict the test set
        test_predictions += model.predict(x_test) / 10
        
    rmspe_score = rmspe(y, oof_predictions)
    print(f'Our out of folds RMSPE is {rmspe_score}')
    # Return test predictions
    return test_predictions
features_to_consider.append('row_id' )
features_to_consider.append('time_id')

test_predictions = train_and_evaluate(train.loc[:, train.columns != 'pred_NN'], test.loc[:, test.columns != 'target'])
# Save test predictions
test['target2'] = test_predictions


# In[23]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import pathlib
from tqdm.auto import tqdm
import json
from multiprocessing import Pool, cpu_count
import time
import requests as re
from datetime import datetime
from dateutil.relativedelta import relativedelta, FR

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import glob
import os
from sklearn import model_selection
import joblib
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

pd.get_option("display.max_columns")
DEBUG = False
# MODE = 'TRAIN'
MODE = 'INFERENCE'
MODEL_DIR = '../input/optiver-lgb-and-te-baseline'
class CFG:
    INPUT_DIR = '../input/optiver-realized-volatility-prediction'
    OUTPUT_DIR = './'
def init_logger(log_file='train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

logger = init_logger(log_file=f'{CFG.OUTPUT_DIR}/baseline.log')
logger.info(f'Start Logging...')
train = pd.read_csv(os.path.join(CFG.INPUT_DIR, 'train.csv'))

logger.info('Train data: {}'.format(train.shape))
test = pd.read_csv(os.path.join(CFG.INPUT_DIR, 'test.csv'))

logger.info('Test data: {}'.format(test.shape))
ss = pd.read_csv(os.path.join(CFG.INPUT_DIR, 'sample_submission.csv'))

logger.info('Sample submission: {}'.format(ss.shape))
train_book_stocks = os.listdir(os.path.join(CFG.INPUT_DIR, 'book_train.parquet'))

if DEBUG:
    logger.info('Debug mode: using 3 stocks only')
    train_book_stocks = train_book_stocks[:3]

logger.info('{:,} train book stocks: {}'.format(len(train_book_stocks), train_book_stocks))
# load stock_id=0
def load_book(stock_id=0, data_type='train'):
    """
    load parquest book data for given stock_id
    """
    book_df = pd.read_parquet(os.path.join(CFG.INPUT_DIR, f'book_{data_type}.parquet/stock_id={stock_id}'))
    book_df['stock_id'] = stock_id
    book_df['stock_id'] = book_df['stock_id'].astype(np.int8)
    
    return book_df

def load_trade(stock_id=0, data_type='train'):
    """
    load parquest trade data for given stock_id
    """
    trade_df = pd.read_parquet(os.path.join(CFG.INPUT_DIR, f'trade_{data_type}.parquet/stock_id={stock_id}'))
    trade_df['stock_id'] = stock_id
    trade_df['stock_id'] = trade_df['stock_id'].astype(np.int8)
    
    return trade_df

book0 = load_book(0)
logger.info('Book data of stock id = 1: {}'.format(book0.shape))
trade0 = load_trade(0)
logger.info('Book data of stock id = 1: {}'.format(trade0.shape))
book_df = book0.merge(
    trade0
    , how='outer'
    , on=['time_id', 'stock_id', 'seconds_in_bucket']
)

def fix_jsonerr(df):
    """
    fix json column error for lightgbm
    """
    df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
    return df

def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff() 

def realized_volatility(series_log_return):
    series_log_return = log_return(series_log_return)
    return np.sqrt(np.sum(series_log_return ** 2))

def fe_row(book):
    """
    Feature engineering (just volatility for now) for each row
    """
    
    # volatility
    for i in [1, 2, ]:  
        # wap
        book[f'book_wap{i}'] = (book[f'bid_price{i}'] * book[f'ask_size{i}'] +
                        book[f'ask_price{i}'] * book[f'bid_size{i}']) / (
                               book[f'bid_size{i}']+ book[f'ask_size{i}'])
        
    # mean wap
    book['book_wap_mean'] = (book['book_wap1'] + book['book_wap2']) / 2
    
    # wap diff
    book['book_wap_diff'] = book['book_wap1'] - book['book_wap2']
    
    # other orderbook features
    book['book_price_spread'] = (book['ask_price1'] - book['bid_price1']) / (book['ask_price1'] + book['bid_price1'])
    book['book_bid_spread'] = book['bid_price1'] - book['bid_price2']
    book['book_ask_spread'] = book['ask_price1'] - book['ask_price2']
    book['book_total_volume'] = book['ask_size1'] + book['ask_size2'] + book['bid_size1'] + book['bid_size2']
    book['book_volume_imbalance'] = (book['ask_size1'] + book['ask_size2']) - (book['bid_size1'] + book['bid_size2'])
    
    return book    

def fe_agg(book_df):
    """
    feature engineering (aggregation by stock_id x time_id)   
    """ 
            
    # features
    book_feats = book_df.columns[book_df.columns.str.startswith('book_')].values.tolist()
    trade_feats = ['price', 'size', 'order_count', 'seconds_in_bucket']
        
    # agg trade features
    trade_df = book_df.groupby(['time_id', 'stock_id'])[trade_feats].agg([
        'sum', 'mean', 'std', 'max', 'min'
    ]).reset_index()
    
    # agg volatility features
    fe_df = book_df.groupby(['time_id', 'stock_id'])[book_feats].agg([
        realized_volatility
    ]).reset_index()
    fe_df.columns = [" ".join(col).strip() for col in fe_df.columns.values]
    
    # merge
    fe_df = fe_df.merge(
        trade_df
        , how='left'
        , on=['time_id', 'stock_id']
    )
    
    return fe_df
    
def fe_all(book_df):
    """
    perform feature engineerings
    """
      
    # row-wise feature engineering
    book_df = fe_row(book_df)
    
    # feature engineering agg by stock_id x time_id 
    fe_df = fe_agg(book_df)
    
    return fe_df
    
def book_fe_by_stock(stock_id=0):
    """
    load orderbook and trade data for the given stock_id and merge
    
    """
    # load data
    book_df = load_book(stock_id, 'train')
    trade_df = load_trade(stock_id, 'train')
    book_feats = book_df.columns.values.tolist()
    
    # merge
    book_df = book_df.merge(
        trade_df
        , how='outer'
        , on=['time_id', 'seconds_in_bucket', 'stock_id']
    )
    
    # sort by time
    book_df = book_df.sort_values(by=['time_id', 'seconds_in_bucket'])
    
    # fillna for book_df
    book_df[book_feats] = book_df[book_feats].fillna(method='ffill')
    
    # feature engineering
    fe_df = fe_all(book_df)
    return fe_df

def book_fe_by_stock_test(stock_id=0):
    """
    same function but for the test
    
    """
    # load data
    book_df = load_book(stock_id, 'test')
    trade_df = load_trade(stock_id, 'test')
    book_feats = book_df.columns.values.tolist()
    
    # merge
    book_df = book_df.merge(
        trade_df
        , how='outer'
        , on=['time_id', 'seconds_in_bucket', 'stock_id']
    )
    
    # sort by time
    book_df = book_df.sort_values(by=['time_id', 'seconds_in_bucket'])
    
    # fillna for book_df
    book_df[book_feats] = book_df[book_feats].fillna(method='ffill')
    
    # feature engineering
    fe_df = fe_all(book_df)
    return fe_df
    
def book_fe_all(stock_ids, data_type='train'): 
    """
    Feature engineering with multithread processing
    """
    # feature engineering agg by stock_id x time_id
    with Pool(cpu_count()) as p:
        if data_type == 'train':
            feature_dfs = list(tqdm(p.imap(book_fe_by_stock, stock_ids), total=len(stock_ids)))
        elif data_type == 'test':
            feature_dfs = list(tqdm(p.imap(book_fe_by_stock_test, stock_ids), total=len(stock_ids)))      
        
    fe_df = pd.concat(feature_dfs)
    
    # feature engineering agg by stock_id
    vol_feats = [f for f in fe_df.columns if ('realized' in f) & ('wap' in f)]
    if data_type == 'train':
        # agg
        stock_df = fe_df.groupby('stock_id')[vol_feats].agg(['mean', 'std', 'max', 'min', ]).reset_index()
        
        # fix column names
        stock_df.columns = ['stock_id'] + [f'{f}_stock' for f in stock_df.columns.values.tolist()[1:]]        
        stock_df = fix_jsonerr(stock_df)
    
    # feature engineering agg by time_id
    time_df = fe_df.groupby('time_id')[vol_feats].agg(['mean', 'std', 'max', 'min', ]).reset_index()
    time_df.columns = ['time_id'] + [f'{f}_time' for f in time_df.columns.values.tolist()[1:]]
    
    # merge
    fe_df = fe_df.merge(
        time_df
        , how='left'
        , on='time_id'
    )
    
    # make sure to fix json error for lighgbm
    fe_df = fix_jsonerr(fe_df)
    
    # out
    if data_type == 'train':
        return fe_df, stock_df
    elif data_type == 'test':
        return fe_df
if MODE == 'TRAIN':
    # all book data feature engineering
    stock_ids = [int(i.split('=')[-1]) for i in train_book_stocks]
    book_df, stock_df = book_fe_all(stock_ids, data_type='train')

    assert book_df['stock_id'].nunique() > 2
    assert book_df['time_id'].nunique() > 2
    
    # save stock_df for the test
    stock_df.to_pickle('train_stock_df.pkl')
    logger.info('train stock df saved!')
    
    # merge
    book_df = book_df.merge(
        stock_df
        , how='left'
        , on='stock_id'
    ).merge(
        train
        , how='left'
        , on=['stock_id', 'time_id']
    ).replace([np.inf, -np.inf], np.nan).fillna(method='ffill')

    # make row_id
    book_df['row_id'] = book_df['stock_id'].astype(str) + '-' + book_df['time_id'].astype(str)

from tqdm import tqdm
# test
test_book_stocks = os.listdir(os.path.join(CFG.INPUT_DIR, 'book_test.parquet'))

logger.info('{:,} test book stocks: {}'.format(len(test_book_stocks), test_book_stocks))

# all book data feature engineering
test_stock_ids = [int(i.split('=')[-1]) for i in test_book_stocks]
test_book_df = book_fe_all(test_stock_ids, data_type='test')

# load stock_df, if inference
if MODE == 'INFERENCE':
    stock_df = pd.read_pickle(f'{MODEL_DIR}/train_stock_df.pkl')
    
# merge
test_book_df = test.merge(
    stock_df
    , how='left'
    , on='stock_id'
).merge(
    test_book_df
    , how='left'
    , on=['stock_id', 'time_id']
).replace([np.inf, -np.inf], np.nan).fillna(method='ffill')

# make row_id
test_book_df['row_id'] = test_book_df['stock_id'].astype(str) + '-' + test_book_df['time_id'].astype(str)
target = 'target'
drops = [target, 'row_id', 'time_id']
features = [f for f in test_book_df.columns.values.tolist() if f not in drops]
cats = ['stock_id', ]

logger.info('{:,} features ({:,} categorical): {}'.format(len(features), len(cats), features))
# evaluation metric
def RMSPEMetric(XGBoost=False):

    def RMSPE(yhat, dtrain, XGBoost=XGBoost):

        y = dtrain.get_label()
        elements = ((y - yhat) / y) ** 2
        if XGBoost:
            return 'RMSPE', float(np.sqrt(np.sum(elements) / len(y)))
        else:
            return 'RMSPE', float(np.sqrt(np.sum(elements) / len(y))), False

    return RMSPE
# LightGBM parameters
params = {
    'n_estimators': 20000,
    'objective': 'rmse',
    'boosting_type': 'gbdt',
    'max_depth': -1,
    'learning_rate': 0.01,
    'subsample': 0.75,
    'subsample_freq': 4,
    'feature_fraction': 0.8,
    'seed':2021,
    'early_stopping_rounds': 500,
    'verbose': -1
} 
def fit_model(params, X_train, y_train, X_test, features=features, cats=[], era='stock_id', fold_type='kfold', n_fold=5, seed=2021):
    """
    fit model with cross validation
    """
    
    models = []
    oof_df = X_train[['time_id', 'stock_id', target]].copy()
    oof_df['pred'] = np.nan
    y_preds = np.zeros((len(X_test),))
    
    if fold_type == 'stratifiedshuffle':
        cv = model_selection.StratifiedShuffleSplit(n_splits=n_fold, random_state=seed)
        kf = cv.split(X_train, X_train[era])
    elif fold_type == 'kfold':
        cv = model_selection.KFold(n_splits=n_fold, shuffle=True, random_state=seed)
        kf = cv.split(X_train, y_train)      
    
    fi_df = pd.DataFrame()
    fi_df['features'] = features
    fi_df['importance'] = 0
        
    for fold_id, (train_index, valid_index) in tqdm(enumerate(kf)):
        # split
        X_tr = X_train.loc[train_index, features]
        X_val = X_train.loc[valid_index, features]
        y_tr = y_train.loc[train_index]
        y_val = y_train.loc[valid_index]
        
        # model (note inverse weighting)
        train_set = lgb.Dataset(X_tr, y_tr, categorical_feature=cats, weight=1/np.power(y_tr, 2))
        val_set = lgb.Dataset(X_val, y_val, categorical_feature=cats, weight=1/np.power(y_val, 2))
        model = lgb.train(
            params
            , train_set
            , valid_sets=[train_set, val_set]
            , feval=RMSPEMetric()
            , verbose_eval=250
        )
        
        # feature importance
        fi_df[f'importance_fold{fold_id}'] = model.feature_importance(importance_type="gain")
        fi_df['importance'] += fi_df[f'importance_fold{fold_id}'].values
        
        # save model
        joblib.dump(model, f'model_fold{fold_id}.pkl')
        logger.debug('model saved!')

        # predict
        oof_df['pred'].iloc[valid_index] = model.predict(X_val)
        y_pred = model.predict(X_test[features])
        y_preds += y_pred / n_fold
        models.append(model)
        
    return oof_df, y_preds, models, fi_df

if MODE == 'TRAIN':
    oof_df, y_preds, models, fi_df = fit_model(params, 
                                          book_df, 
                                          book_df[target], 
                                          test_book_df, 
                                          features=features, 
                                          cats=cats,
                                          era=None,
                                          fold_type='kfold', 
                                          n_fold=5, 
                                          seed=2021
                                              )
from sklearn.metrics import r2_score
def rmspe(y_true, y_pred):
    return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))

if MODE == 'TRAIN':
    oof_df.dropna(inplace=True)
    y_true = oof_df[target].values
    y_pred = oof_df['pred'].values
    
    oof_df[target].hist(bins=100)
    oof_df['pred'].hist(bins=100)
    
    R2 = round(r2_score(y_true, y_pred), 3)
    RMSPE = round(rmspe(y_true, y_pred), 3)
    logger.info(f'Performance of the naive prediction: R2 score: {R2}, RMSPE: {RMSPE}')
# performance by stock_id
if MODE == 'TRAIN':
    for stock_id in oof_df['stock_id'].unique():
        y_true = oof_df.query('stock_id == @stock_id')[target].values
        y_pred = oof_df.query('stock_id == @stock_id')['pred'].values
        
        R2 = round(r2_score(y_true, y_pred), 3)
        RMSPE = round(rmspe(y_true, y_pred), 3)
        logger.info(f'Performance by stock_id={stock_id}: R2 score: {R2}, RMSPE: {RMSPE}')
if MODE == 'INFERENCE':
    """
    used for inference kernel only
    """
    y_preds = np.zeros(len(test_book_df))
    files = glob.glob(f'{MODEL_DIR}/*model*.pkl')
    assert len(files) > 0
    for i, f in enumerate(files):
        model = joblib.load(f)
        y_preds += model.predict(test_book_df[features])
    y_preds /= (i+1)
    
test_book_df[target] = y_preds


# In[24]:


final_predictions = (model1_predictions * 0.23 + model2_predictions * 0.27 + model3_predictions * 0.27+y_preds*0.23)*0.4+0.6*((nn_pre + test_predictions)/2)
submit = pd.read_csv('../input/optiver-realized-volatility-prediction/sample_submission.csv')

submit.target = final_predictions
submit


# In[25]:


submit.to_csv('submission.csv',index = False)

