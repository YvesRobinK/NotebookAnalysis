#!/usr/bin/env python
# coding: utf-8

# ### Try to generate as much features as possible and leverage the best ones.

# #### Helpers

# In[1]:


import os
import glob
from functools import reduce
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm


# In[2]:


N_CPUS = 4


# In[3]:


def get_wap(bid_price, ask_price, bid_size, ask_size):
    return (bid_price * ask_size + ask_price * bid_size) / (
        bid_size + ask_size
    )


def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff()


def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return ** 2))


def get_realized_vol(book: pd.DataFrame):
    """Mutable method to get past realized volatility"""

    book["wap1"] = get_wap(
        book.bid_price1, book.ask_price1, book.bid_size1, book.ask_size1
    )

    book["wap2"] = get_wap(
        book.bid_price2, book.ask_price2, book.bid_size2, book.ask_size2
    )

    a1 = (
        book['bid_price1'] * book['ask_size1'] +
        book['ask_price1'] * book['bid_size1']
    )
    a2 = (
        book['bid_price2'] * book['ask_size2'] +
        book['ask_price2'] * book['bid_size2']
    )
    b = (
        book['bid_size1'] + book['ask_size1'] +
        book['bid_size2']+ book['ask_size2']
    )
    book['wap3'] = (a1 + a2)/ b

    book['wap4'] = (book['wap1'] + book['wap2']) / 2

    book.loc[:, "log_return1"] = log_return(book["wap1"])
    book.loc[:, "log_return2"] = log_return(book["wap2"])
    book.loc[:, "log_return3"] = log_return(book["wap3"])
    book.loc[:, "log_return4"] = log_return(book["wap4"])

    # book = book[~book['log_return'].isnull()]
    # to not remove rows both on train and test set
    book = book.fillna(0)

    book = book.merge(
        book[["time_id", "log_return1"]]
        .rename({"log_return1": "vol1"}, axis=1)
        .groupby("time_id")
        .agg(realized_volatility),
        how="inner",
        on="time_id",
    ).merge(
        book[["time_id", "log_return2"]]
        .rename({"log_return2": "vol2"}, axis=1)
        .groupby("time_id")
        .agg(realized_volatility),
        how="inner",
        on="time_id",
    ).merge(
        book[["time_id", "log_return3"]]
        .rename({"log_return3": "vol3"}, axis=1)
        .groupby("time_id")
        .agg(realized_volatility),
        how="inner",
        on="time_id",
    ).merge(
        book[["time_id", "log_return4"]]
        .rename({"log_return4": "vol4"}, axis=1)
        .groupby("time_id")
        .agg(realized_volatility),
        how="inner",
        on="time_id",
    )

    book['wap_diff12'] = book['wap2'] - book['wap1']
    book['wap_diff13'] = book['wap3'] - book['wap1']
    book['wap_diff14'] = book['wap4'] - book['wap1']

    book['vol_rate'] = (book['vol1'] / book['vol2']).fillna(0)
    book['vol_diff'] = (book['vol1'] - book['vol2'])

    return book


# Price features
def get_stats(df, ind:str, column:str):
    """Get aggregated features from the column provided"""
    stats = pd.merge(
        df[[ind, column]].groupby(ind).mean()
        .rename({column: f'{column}_mean'}, axis=1),
        df[[ind, column]].groupby(ind).median()
        .rename({column: f'{column}_median'}, axis=1),
        how='inner', left_index=True, right_index=True
    ).merge(
        df[[ind, column]].groupby(ind).count()
        .rename({column: f'{column}_count'}, axis=1),
        how='inner', left_index=True, right_index=True
    ).merge(
        df[[ind, column]].groupby(ind).min()
        .rename({column: f'{column}_min'}, axis=1),
        how='inner', left_index=True, right_index=True
    ).merge(
        df[[ind, column]].groupby(ind).max()
        .rename({column: f'{column}_max'}, axis=1),
        how='inner', left_index=True, right_index=True
    ).merge(
        df[[ind, column]].groupby(ind).std()
        .rename({column: f'{column}_std'}, axis=1),
        how='inner', left_index=True, right_index=True
    ).merge(
        df[[ind, column]].groupby(ind).quantile(0.25)
        .rename({column: f'{column}_q25'}, axis=1),
        how='inner', left_index=True, right_index=True
    ).merge(
        df[[ind, column]].groupby(ind).quantile(0.75)
        .rename({column: f'{column}_q75'}, axis=1),
        how='inner', left_index=True, right_index=True
    ).merge(
        df[[ind, column]].groupby(ind).nunique()
        .rename({column: f'{column}_unique'}, axis=1),
        how='inner', left_index=True, right_index=True
    ).merge(
        df[[ind, column]].groupby(ind).mad()
        .rename({column: f'{column}_mad'}, axis=1),
        how='inner', left_index=True, right_index=True
    ).merge(
        df[[ind, column]].groupby(ind).first()
        .rename({column: f'{column}_first'}, axis=1),
        how='inner', left_index=True, right_index=True
    ).merge(
        df[[ind, column]].groupby(ind).last()
        .rename({column: f'{column}_last'}, axis=1),
        how='inner', left_index=True, right_index=True
    )

    stats[f'{column}_delta'] = (
        stats[f'{column}_last'] - stats[f'{column}_first']
    )
    stats[f'{column}_delta_abs'] = (
        stats[f'{column}_last'] - stats[f'{column}_first']
    ).abs()
    stats[f'{column}_unique_pct'] = (
        stats[f'{column}_unique'] / stats[f'{column}_count']
    )

    del stats[f'{column}_count']
    del stats[f'{column}_unique']
    del stats[f'{column}_first']
    del stats[f'{column}_last']

    return stats


def get_book_features(files):
    pieces = []

    for f in tqdm(files):
        book = pd.read_parquet(f)

        book = get_realized_vol(book)

        # price
        book['spread1'] = book['ask_price1'] - book['bid_price1']
        book['spread2'] = book['ask_price2'] - book['bid_price2']
        book['ask_spread'] = book['ask_price2'] - book['ask_price1']
        book['bid_spread'] = book['bid_price1'] - book['bid_price2']
        book['cross_spread1'] = book['ask_price1'] - book['bid_price1']
        book['cross_spread2'] = book['ask_price2'] - book['bid_price2']
        book['bas'] = (
            book[['ask_price1', 'ask_price2']].min(axis = 1) /
            book[['bid_price1', 'bid_price2']].max(axis = 1) -
            1
        )

        # size
        book['skew1'] = book['ask_size1'] - book['bid_size1']
        book['skew2'] = book['ask_size2'] - book['bid_size2']
        book['cross_skew1'] = book['ask_size1'] - book['bid_size2']
        book['cross_skew2'] = book['ask_size2'] - book['bid_size1']
        book['ask_sum'] = book['ask_size1'] - book['ask_size2']
        book['bid_sum'] = book['bid_size1'] - book['bid_size2']
        book['skew_whole'] = book['ask_sum'] - book['bid_sum']

        # price - size combinations
        book['bid_volume1'] = book['bid_price1'] * book['bid_size1']
        book['bid_volume2'] = book['bid_price2'] * book['bid_size2']
        book['ask_volume1'] = book['ask_price1'] * book['ask_size1']
        book['ask_volume2'] = book['ask_price2'] * book['ask_size2']

        # sum of volumes
        book_sums = (
            book[['time_id',
                'bid_volume1', 'bid_volume2',
                'ask_volume1','ask_volume2']]
            .groupby('time_id').sum()
            .rename({
                'bid_volume1': 'bid_volume1_sum',
                'bid_volume2': 'bid_volume2_sum',
                'ask_volume1': 'ask_volume1_sum',
                'ask_volume2': 'ask_volume2_sum'
            }, axis=1)
        )

        features_to_get_stats = [
            'seconds_in_bucket',
            'wap1', 'wap2', 'wap3', 'wap4',
            'log_return1', 'log_return2', 'log_return3', 'log_return4',
            'vol1', 'vol2', 'vol3', 'vol4', 'vol_rate', 'vol_diff',
            'wap_diff12', 'wap_diff13', 'wap_diff14',
            'bid_price1', 'bid_price2', 'ask_price1', 'ask_price2',
            'spread1', 'spread2', 'ask_spread', 'bid_spread', 'cross_spread1', 'cross_spread2', 'bas',
            'bid_size1', 'bid_size2', 'ask_size1', 'ask_size2',
            'skew1', 'skew2', 'cross_skew1', 'cross_skew2', 'ask_sum', 'bid_sum', 'skew_whole',
            'bid_volume1', 'bid_volume2', 'ask_volume1', 'ask_volume2',
        ]

        pool = Pool(N_CPUS)
        stats_df = pool.starmap(
            get_stats,
            zip(
                [book] * len(features_to_get_stats),
                ['time_id'] * len(features_to_get_stats),
                features_to_get_stats
            )
        )
        pool.close() 
        pool.join()

        # do not merge with not aggregated book
        dfs = [
            book_sums,
            *stats_df
        ]

        df_stats = reduce(
            lambda left, right: pd.merge(
                left, right,
                how='inner', left_index=True, right_index=True
            ),
            dfs
        )

        df_stats["stock_id"] = int(f.split("=")[-1])

        pieces.append(df_stats)

    dataset_new = pd.concat(pieces).reset_index()
    
    features = list(dataset_new.keys())

    dataset_new["row_id"] = [
        f"{stock_id}-{time_id}"
        for stock_id, time_id in zip(
            dataset_new["stock_id"], dataset_new["time_id"]
        )
    ]

    
    return dataset_new, features


def get_trade_features(files):
    """Getting features from trading history"""

    pieces = []

    for f in tqdm(files):
        trades = pd.read_parquet(f)

        trades['trade_volume'] = trades['price'] * trades['size']
        trades['trade_size_per_order'] = (
            trades['size'] / trades['order_count']
        )
        trades['trade_volume_per_order_mean'] = (
            trades['trade_volume'] / trades['trade_size_per_order']
        )

        trades = trades.rename(
            {
                'price': 'trade_price',
                'order_count': 'trade_order_count',
                'seconds_in_bucket': 'trade_seconds_in_bucket'
            },
            axis=1
        )
        
        # sum of volumes, orders and sizes
        trades_sums = (
            trades[['time_id',
                'size', 'trade_order_count', 'trade_volume'
            ]]
            .groupby('time_id').sum()
            .rename({
                'size': 'size_sum',
                'trade_order_count': 'trade_order_count_sum',
                'trade_volume': 'trade_volume_sum'
            }, axis=1)
        )

        # volatility of trades
        trades.loc[:, "trade_log_return"] = log_return(trades["trade_price"])

        trades_vol = (
            trades[["time_id", "trade_log_return"]]
            .rename({"trade_log_return": "trade_vol"}, axis=1)
            .groupby("time_id")
            .agg(realized_volatility)
        )

        features_to_get_stats = [
            'trade_seconds_in_bucket',  # where operations are located in the bucket
            'trade_price', 'size', 'trade_order_count',
            'trade_volume', 'trade_size_per_order', 'trade_volume_per_order_mean',
            'trade_log_return'
        ]

        pool = Pool(N_CPUS)
        stats_df = pool.starmap(
            get_stats,
            zip([trades] * len(features_to_get_stats), ['time_id'] * len(features_to_get_stats), features_to_get_stats)
        )
        pool.close() 
        pool.join()

        # do not merge with not aggregated trades
        dfs = [
            trades_sums,
            trades_vol,
            *stats_df
        ]

        dfs.append(
            trades[['time_id', 'trade_seconds_in_bucket']]
            .groupby('time_id').count()
            .rename({'trade_seconds_in_bucket': 'n_trades'}, axis=1)
        )

        df_stats = reduce(
            lambda left, right: pd.merge(
                left, right, how='inner', left_index=True, right_index=True
            ),
            dfs
        )


        df_stats["stock_id"] = int(f.split("=")[-1])
        pieces.append(df_stats)

    dataset_new = pd.concat(pieces).reset_index()
    
    features = list(dataset_new.keys())

    return dataset_new, features


def mean_encoding(
    dataset: pd.DataFrame,
    means: dict = None,
    stds: dict = None,
    medians: dict = None
):
    """Dataset with stock_id and target columns"""
    if means is None:
        means = (
            dataset[["stock_id", "target"]]
            .groupby("stock_id").mean()
        )
        means = means['target'].to_dict()

        stds = (
            dataset[["stock_id", "target"]]
            .groupby("stock_id").std()
        )
        stds = stds['target'].to_dict()

        medians = (
            dataset[["stock_id", "target"]]
            .groupby("stock_id").median()
        )
        medians = medians['target'].to_dict()

    dataset["stock_id_mean"] = dataset["stock_id"].apply(lambda x: means[x])
    dataset["stock_id_std"] = dataset["stock_id"].apply(lambda x: stds[x])
    dataset["stock_id_median"] = dataset["stock_id"].apply(lambda x: medians[x])
    del dataset['stock_id']

    return dataset, (means, stds, medians)


# #### Dataset building

# In[4]:


# target
dataset = pd.read_csv("../input/optiver-realized-volatility-prediction/train.csv")


# In[5]:


# book data
files = glob.glob(
    "../input/optiver-realized-volatility-prediction/book_train.parquet/*"
)
books, features_book = get_book_features(files)

dataset_new = pd.merge(
    books,
    dataset[["time_id", "stock_id", "target"]],
    how="inner",
    on=["time_id", "stock_id"],
)


# In[6]:


# trade data
files_trade = glob.glob(
    "../input/optiver-realized-volatility-prediction/trade_train.parquet/*"
)
trade_stats, features_trade = get_trade_features(files_trade)

# merging dataset
dataset_new = pd.merge(
    dataset_new,
    trade_stats,
    how='inner', on=['time_id', 'stock_id'],
    left_index=False, right_index=False
)


# In[7]:


print(f"Number of features genereated: {dataset_new.shape[1] - 2}")


# In[8]:


dataset_new.to_csv('dataset_new.csv', index=False, header=True)

