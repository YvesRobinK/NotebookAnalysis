#!/usr/bin/env python
# coding: utf-8

# # Introduction
# The purpose of this notebook is using [VotingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html) for ensemble learning with XGBoost, CatBoost, LightGBM models.
# 
# **Your upvoting 👍 is important to me and can greatly encourage me!**

# I compared LB on different setups: VotingRegressor for XGboost, Catboost, and LightGBM
# 
# 1. No Feature engineering + No hyperparameter setup: **5.4145**
# 2. No Feature engineering + Hyperparameter setup: **5.4078**
# 3. Feature engineering (130feats) + Hyperparameter setup(500 iterations): **5.3471**
# 4. Feature engineering (130feats) + Hyperparameter setup(1000 iterations): **5.3439**

# In[1]:


import pandas as pd
import numpy as np

from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import gc  
import os  
import time  
import warnings 
from itertools import combinations  
from warnings import simplefilter 
warnings.filterwarnings("ignore")
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


# In[2]:


df = pd.read_csv("/kaggle/input/optiver-trading-at-the-close/train.csv")
df = df.dropna(subset=["target"])
df.reset_index(drop=True, inplace=True)
df.loc[df['imbalance_size'].isna(), 'imbalance_buy_sell_flag'] = 0
df['imbalance_size'].fillna(0, inplace=True)
df[df['imbalance_size'].isna()]
df_shape = df.shape
df_shape


# In[3]:


def reduce_mem_usage(df, verbose=0):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == "int":
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
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)
    if verbose:
        logger.info(f"Memory usage of dataframe is {start_mem:.2f} MB")
        end_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"Memory usage after optimization is: {end_mem:.2f} MB")
        decrease = 100 * (start_mem - end_mem) / start_mem
        logger.info(f"Decreased by {decrease:.2f}%")
    return df


# In[4]:


from numba import njit, prange

@njit(parallel=True)
def compute_triplet_imbalance(df_values, comb_indices):
    num_rows = df_values.shape[0]
    num_combinations = len(comb_indices)
    imbalance_features = np.empty((num_rows, num_combinations))
    for i in prange(num_combinations):
        a, b, c = comb_indices[i]
        for j in range(num_rows):
            max_val = max(df_values[j, a], df_values[j, b], df_values[j, c])
            min_val = min(df_values[j, a], df_values[j, b], df_values[j, c])
            mid_val = df_values[j, a] + df_values[j, b] + df_values[j, c] - min_val - max_val
            
            if mid_val == min_val:
                imbalance_features[j, i] = np.nan
            else:
                imbalance_features[j, i] = (max_val - mid_val) / (mid_val - min_val)

    return imbalance_features

def calculate_triplet_imbalance_numba(price, df):
    df_values = df[price].values
    comb_indices = [(price.index(a), price.index(b), price.index(c)) for a, b, c in combinations(price, 3)]
    features_array = compute_triplet_imbalance(df_values, comb_indices)
    columns = [f"{a}_{b}_{c}_imb2" for a, b, c in combinations(price, 3)]
    features = pd.DataFrame(features_array, columns=columns)
    return features


# In[5]:


df_train = df

global_stock_id_feats = {
        "median_size": df_train.groupby("stock_id")["bid_size"].median() + df_train.groupby("stock_id")["ask_size"].median(),
        "std_size": df_train.groupby("stock_id")["bid_size"].std() + df_train.groupby("stock_id")["ask_size"].std(),
        "ptp_size": df_train.groupby("stock_id")["bid_size"].max() - df_train.groupby("stock_id")["bid_size"].min(),
        "median_price": df_train.groupby("stock_id")["bid_price"].median() + df_train.groupby("stock_id")["ask_price"].median(),
        "std_price": df_train.groupby("stock_id")["bid_price"].std() + df_train.groupby("stock_id")["ask_price"].std(),
        "ptp_price": df_train.groupby("stock_id")["bid_price"].max() - df_train.groupby("stock_id")["ask_price"].min(),
    }


# In[6]:


def imbalance_features(df):
    # Define lists of price and size-related column names
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]
    df["volume"] = df.eval("ask_size + bid_size")
    df["mid_price"] = df.eval("(ask_price + bid_price) / 2")
    df["liquidity_imbalance"] = df.eval("(bid_size-ask_size)/(bid_size+ask_size)")
    df["matched_imbalance"] = df.eval("(imbalance_size-matched_size)/(matched_size+imbalance_size)")
    df["size_imbalance"] = df.eval("bid_size / ask_size")
    df["price_spread"] = df.eval("ask_price-bid_price")
    df['price_pressure'] = df.eval("imbalance_size*(ask_price-bid_price)")
    df['market_urgency'] = df.eval("price_spread*liquidity_imbalance")
    df['depth_pressure'] = df.eval("(ask_size-bid_size)/(far_price-near_price)")
    
    
    for c in combinations(prices, 2):
        df[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")

    for c in [['ask_price', 'bid_price', 'wap', 'reference_price', 'near_price'], sizes]:
        triplet_feature = calculate_triplet_imbalance_numba(c, df)
        df[triplet_feature.columns] = triplet_feature.values
   
    df["imbalance_momentum"] = df.groupby(['stock_id'])['imbalance_size'].diff(periods=1) / df['matched_size']
    df["spread_intensity"] = df.groupby(['stock_id'])['price_spread'].diff()
    
    # Calculate various statistical aggregation features
    for func in ["mean", "std", "skew", "kurt"]:
        df[f"all_prices_{func}"] = df[prices].agg(func, axis=1)
        df[f"all_sizes_{func}"] = df[sizes].agg(func, axis=1)
        

    for col in ['matched_size', 'imbalance_size', 'reference_price', 'imbalance_buy_sell_flag']:
        for window in [1, 2, 3, 10]:
            df[f"{col}_shift_{window}"] = df.groupby('stock_id')[col].shift(window)
            df[f"{col}_ret_{window}"] = df.groupby('stock_id')[col].pct_change(window)
    
    # Calculate diff features for specific columns
    for col in ['ask_price', 'bid_price', 'ask_size', 'bid_size', 'market_urgency', 'imbalance_momentum', 'size_imbalance']:
        for window in [1, 2, 3, 10]:
            df[f"{col}_diff_{window}"] = df.groupby("stock_id")[col].diff(window)
            
    return df.replace([np.inf, -np.inf], 0)

def other_features(df):
    df["dow"] = df["date_id"] % 5  # Day of the week
    df["seconds"] = df["seconds_in_bucket"] % 60  
    df["minute"] = df["seconds_in_bucket"] // 60  
    for key, value in global_stock_id_feats.items():
        df[f"global_{key}"] = df["stock_id"].map(value.to_dict())

    return df

def generate_all_features(df):
    # Select relevant columns for feature generation
    cols = [c for c in df.columns if c not in ["row_id", "time_id", "target"]]
    df = df[cols]
    
    # Generate imbalance features
    df = imbalance_features(df)
    df = other_features(df)
    gc.collect()  
    
    feature_name = [i for i in df.columns if i not in ["row_id", "target", "time_id", "date_id"]]
    
    return df[feature_name]


# In[7]:


df_train_feats = generate_all_features(df_train)
df_train_feats = reduce_mem_usage(df_train_feats)


# In[8]:


# Define base regressors using boosting algorithms
xgb_regressor = XGBRegressor(**{'tree_method'        : 'hist',
                      'device'             : 'cuda',
                      'objective'          : 'reg:absoluteerror',
                      'random_state'       : 42,
                      'colsample_bytree'   : 0.7,
                      'learning_rate'      : 0.07,
                      'max_depth'          : 6,
                      'n_estimators'       : 3500,                         
                      'reg_alpha'          : 0.025,
                      'reg_lambda'         : 1.75,
                      'min_child_weight'   : 1000
    
})
catboost_regressor = CatBoostRegressor(**{
                   'task_type'           : "CPU",
                   'objective'           : "MAE",
                   'eval_metric'         : "MAE",
                   'bagging_temperature' : 0.5,
                   'colsample_bylevel'   : 0.7,
                   'iterations'          : 3500,
                   'learning_rate'       : 0.065,
                   'od_wait'             : 25,
                   'max_depth'           : 7,
                   'l2_leaf_reg'         : 1.5,
                   'min_data_in_leaf'    : 1000,
                   'random_strength'     : 0.65, 
                   'verbose'             : 0
                   
})
lgbm_regressor = LGBMRegressor(**{
                "objective": "mae",
                "n_estimators": 3500,
                "num_leaves": 256,
                "subsample": 0.6,
                "colsample_bytree": 0.8,
                "learning_rate": 0.00871,
                'max_depth': 11,
                "n_jobs": 4,
                "device": "gpu",
                "verbosity": -1,
                "importance_type": "gain",
})

# Create a Voting Regressor
voting_regressor = VotingRegressor(estimators=[
    ('xgb', xgb_regressor),
    ('catboost', catboost_regressor),
    ('lgbm', lgbm_regressor)
], n_jobs=-1, verbose=True)

# Train the Voting Regressor on the training data
voting_regressor.fit(df_train_feats, df_train['target'])


# In[9]:


import optiver2023
env = optiver2023.make_env()
iter_test = env.iter_test()

counter = 0
y_min, y_max = -64, 64
qps, predictions = [], []
cache = pd.DataFrame()

for (test, revealed_targets, sample_prediction) in iter_test:
    test.drop('currently_scored', axis=1, inplace=True)
    now_time = time.time()
    cache = pd.concat([cache, test], ignore_index=True, axis=0)
    if counter > 0:
        cache = cache.groupby(['stock_id']).tail(21).sort_values(by=['date_id', 'seconds_in_bucket', 'stock_id']).reset_index(drop=True)
    feat = generate_all_features(cache)[-len(test):]

    predictions = voting_regressor.predict(feat)
    predictions = predictions - predictions.mean()
    clipped_predictions = np.clip(predictions, y_min, y_max)

    sample_prediction['target'] = clipped_predictions
    env.predict(sample_prediction)
    counter += 1
    qps.append(time.time() - now_time)
    if counter % 10 == 0:
        print(counter, 'qps:', np.mean(qps))

time_cost = 1.146 * np.mean(qps)
print(f"The code will take approximately {np.round(time_cost, 4)} hours to reason about")

