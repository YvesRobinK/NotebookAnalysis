#!/usr/bin/env python
# coding: utf-8

# # Update 
# * Rebase seconds_in_bucket because of the filler data 
# * Forward fill of the `book_data` for train and test data

# # Acknowledgment
# * [we-need-to-go-deeper-and-validate](https://www.kaggle.com/konradb/we-need-to-go-deeper-and-validate)
# * [accelerating-trading-on-gpu-via-rapids](https://www.kaggle.com/aerdem4/accelerating-trading-on-gpu-via-rapids)
# * [Forward filling book data](https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/251277)
# * [Deep Learning approach with a CNN- inference](https://www.kaggle.com/slawekbiel/deep-learning-approach-with-a-cnn-inference)

# # Library

# In[1]:


import sys
sys.path.append('../input/rapids-kaggle-utils')

import cupy as cp
import cudf
import cuml
import glob
from tqdm import tqdm
import cu_utils.transform as cutran

import gc
from joblib import Parallel, delayed


# In[2]:


PATH = "../input/optiver-realized-volatility-prediction"
order_book_training = glob.glob(f'{PATH}/book_train.parquet/*/*')
len(order_book_training)


# # Utils

# In[3]:


def fix_offsets(data_df):
    
    offsets = data_df.groupby(['time_id']).agg({'seconds_in_bucket':'min'})
    offsets.columns = ['offset']
    data_df = data_df.join(offsets, on='time_id')
    data_df.seconds_in_bucket = data_df.seconds_in_bucket - data_df.offset
    
    return data_df

def ffill(data_df):
    # MultiIndex.from_product uses pandas in the background
    # That's why we need to transform the data into pd dataframe
    # Then the MultiIndex.from_product will return cudf dataframe 
    data_df=data_df.set_index(['time_id', 'seconds_in_bucket']).to_pandas()
    data_df = data_df.reindex(pd.MultiIndex.from_product([data_df.index.levels[0], np.arange(0,600)], names = ['time_id', 'seconds_in_bucket']), method='ffill')
    return data_df.reset_index()


def rel_vol_fe(df, null_val=-9999):
    
    # compute wap
    for n in range(1, 3):
        p1 = df[f"bid_price{n}"]
        p2 = df[f"ask_price{n}"]
        s1 = df[f"bid_size{n}"]
        s2 = df[f"ask_size{n}"]
        df["WAP"] = (p1*s2 + p2*s1) / (s1 + s2)


        df["log_wap"] = df["WAP"].log()
        df["log_wap_shifted"] = (df[["time_id", "log_wap"]].groupby("time_id", method='cudf')
                                 .apply_grouped(cutran.get_cu_shift_transform(shift_by=1, null_val=null_val),
                                                incols={"log_wap": 'x'},
                                                outcols=dict(y_out=cp.float32),
                                                tpb=32)["y_out"])
        df = df[df["log_wap_shifted"] != null_val]

        df["diff_log_wap"] = df["log_wap"] - df["log_wap_shifted"]
        df[f"diff_log_wap{n}"] = df["diff_log_wap"]**2
    

    
    # Summary statistics for different 'diff_log_wap'
    sum_df = df.groupby("time_id").agg({"diff_log_wap1": {"sum", "mean", "std", "median", "max", "min"}, 
                                        "diff_log_wap2": {"sum", "mean", "std", "median", "max", "min"}}
                                      ).reset_index()
    
    # Create wanted features for training
    def f(x):
        if x[1] == "":
            return x[0]
        return x[0] + "_" + x[1]
    
    sum_df.columns = [f(x) for x in sum_df.columns]
    sum_df["volatility1"] = (sum_df["diff_log_wap1_sum"])**0.5
    sum_df["volatility2"] = (sum_df["diff_log_wap2_sum"])**0.5
    sum_df["vol1_mean"] = sum_df["diff_log_wap1_mean"].fillna(0).values
    sum_df["vol2_mean"] = sum_df["diff_log_wap2_mean"].fillna(0).values
    sum_df["vol1_std"] = sum_df["diff_log_wap1_std"].fillna(0).values
    sum_df["vol2_std"] = sum_df["diff_log_wap2_std"].fillna(0).values
    sum_df["vol1_median"] = sum_df["diff_log_wap1_median"].fillna(0).values
    sum_df["vol2_median"] = sum_df["diff_log_wap2_median"].fillna(0).values
    sum_df["vol1_max"] = sum_df["diff_log_wap1_max"].fillna(0).values
    sum_df["vol2_max"] = sum_df["diff_log_wap2_max"].fillna(0).values
    sum_df["vol1_min"] = sum_df["diff_log_wap1_min"].fillna(0).values
    sum_df["vol2_min"] = sum_df["diff_log_wap2_min"].fillna(0).values
    sum_df["volatility_rate"] = (sum_df["volatility1"] / sum_df["volatility2"]).fillna(0)
    sum_df["mean_volatility_rate"] = (sum_df["vol1_mean"] / sum_df["vol2_mean"]).fillna(0)
    sum_df["std_volatility_rate"] = (sum_df["vol1_std"] / sum_df["vol2_std"]).fillna(0)
    sum_df["median_volatility_rate"] = (sum_df["vol1_median"] / sum_df["vol2_median"]).fillna(0)
    sum_df["max_volatility_rate"] = (sum_df["vol1_max"] / sum_df["vol2_max"]).fillna(0)
    sum_df["min_volatility_rate"] = (sum_df["vol1_min"] / sum_df["vol2_min"]).fillna(0)
    
    return sum_df[["time_id", "volatility1", "volatility2", 
                   "volatility_rate", "vol1_std", "vol2_std",
                   "vol1_mean", "vol2_mean", "vol1_median", "vol2_median",
                   "vol1_max", "vol2_max", "vol1_min", "vol2_min",
                   "mean_volatility_rate", "std_volatility_rate",
                   "median_volatility_rate", "max_volatility_rate",
                   "min_volatility_rate"]]

def spread_fe(df):
    
    # Bid ask spread
    df['bas'] = (df[['ask_price1', 'ask_price2']].min(axis = 1)
                                / df[['bid_price1', 'bid_price2']].max(axis = 1) - 1)                               

    # different spreads
    df['h_spread_l1'] = df['ask_price1'] - df['bid_price1']
    df['h_spread_l2'] = df['ask_price2'] - df['bid_price2']
    df['v_spread_b'] = df['bid_price1'] - df['bid_price2']
    df['v_spread_a'] = df['ask_price1'] - df['ask_price2']
    
    # Summary statistics for different spread
    sum_df = df.groupby("time_id").agg({"h_spread_l1": { "mean", "std", "median", "max", "min"}, 
                                        "h_spread_l2": { "mean", "std", "median", "max", "min"},
                                        "v_spread_b": {"mean", "std", "median", "max", "min"},
                                        "v_spread_a": {"mean", "std", "median", "max", "min"},
                                        "bas": {"mean"}}
                                      ).reset_index()
    
    
    # Create wanted features for training
    def f(x):
        if x[1] == "":
            return x[0]
        return x[0] + "_" + x[1]
    
    sum_df.columns = [f(x) for x in sum_df.columns]

    return sum_df
    
    

def get_stat(path):
    
    book = cudf.read_parquet(path)
    stock_id = int(path.split("=")[1].split("/")[0])
    book = fix_offsets(book)
    #book = cudf.DataFrame(ffill(book))
    rel_vol_data = rel_vol_fe(book)
    spread_data = spread_fe(book)
    transbook = cudf.merge(rel_vol_data,
                           spread_data,
                           on = ['time_id'], how = 'left')
    transbook['stock_id'] = stock_id
    
    return transbook


def process_data(order_book_paths):
    
    df = Parallel(n_jobs=-1, verbose=1)(delayed(get_stat)(path)
                             for path in tqdm(order_book_paths))
    
    stock_dfs = cudf.concat(df, ignore_index=True)
    return stock_dfs


# # Feature generation

# In[4]:


get_ipython().run_cell_magic('time', '', 'train_past_volatility = process_data(order_book_training)\nprint(train_past_volatility.shape)\ntrain_past_volatility.to_csv("./train_past_volatility.csv")\ntrain_past_volatility.columns\n')

