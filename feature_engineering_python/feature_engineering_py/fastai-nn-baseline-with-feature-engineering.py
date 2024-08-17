#!/usr/bin/env python
# coding: utf-8

# ## Fast AI TabularPandas and tabular learner

# In[1]:


import pandas as pd
import numpy as np
from itertools import combinations
import gc


# In[2]:


from fastai.tabular.all import *

pd.options.display.float_format = '{:,.2f}'.format
set_seed(32)


# In[3]:


df = pd.read_csv('../input/optiver-trading-at-the-close/train.csv')


# In[4]:


df.dropna(subset=['target'], inplace=True)
df.fillna(0, inplace=True)


# In[5]:


df = df.sort_values(by=['stock_id', 'date_id', 'seconds_in_bucket']).reset_index(drop=True)


# ## Feature Engineering from lgb-baseline-train

# In[6]:


median_sizes = df.groupby('stock_id')['bid_size'].median() + df.groupby('stock_id')['ask_size'].median()
std_sizes = df.groupby('stock_id')['bid_size'].std() + df.groupby('stock_id')['ask_size'].std()
max_sizes = df.groupby('stock_id')['bid_size'].max() + df.groupby('stock_id')['ask_size'].max()
min_sizes = df.groupby('stock_id')['bid_size'].min() + df.groupby('stock_id')['ask_size'].min()
mean_sizes = df.groupby('stock_id')['bid_size'].mean() + df.groupby('stock_id')['ask_size'].mean()
first_sizes = df.groupby('stock_id')['bid_size'].first() + df.groupby('stock_id')['ask_size'].first()
last_sizes = df.groupby('stock_id')['bid_size'].last() + df.groupby('stock_id')['ask_size'].last()


# In[7]:


def feature_eng(df):
    cols = [c for c in df.columns if c not in ['row_id']]
    df = df[cols].copy()
    
    df.loc[:, 'imbalance_ratio'] = df['imbalance_size'] / df['matched_size']
    df['bid_ask_volume_diff'] = df['ask_size'] - df['bid_size']
    df['bid_plus_ask_sizes'] = df['bid_size'] + df['ask_size']
    df['mid_price'] = (df['ask_price'] + df['bid_price']) / 2
    df['median_size'] = df['stock_id'].map(median_sizes.to_dict())
    df['std_size'] = df['stock_id'].map(std_sizes.to_dict())
    df['max_size'] = df['stock_id'].map(max_sizes.to_dict())
    df['min_size'] = df['stock_id'].map(min_sizes.to_dict())
    df['mean_size'] = df['stock_id'].map(mean_sizes.to_dict())
    df['first_size'] = df['stock_id'].map(first_sizes.to_dict())    
    df['last_size'] = df['stock_id'].map(last_sizes.to_dict())       
    
    df['high_volume'] = np.where(df['bid_plus_ask_sizes'] > df['median_size'], 1, 0)
    
    prices = ['reference_price', 'far_price', 'near_price', 'ask_price', 'bid_price', 'wap']
    
    for c in combinations(prices, 2):
        df[f'{c[0]}_minus_{c[1]}'] = (df[f'{c[0]}'] - df[f'{c[1]}']).astype(np.float32)
        df[f'{c[0]}_{c[1]}_imb'] = df.eval(f'({c[0]} - {c[1]})/({c[0]} + {c[1]})')
        
    for c in combinations(prices, 3):
        max_ = df[list(c)].max(axis=1)
        min_ = df[list(c)].min(axis=1)
        mid_ = df[list(c)].sum(axis=1) - min_ - max_
        
        df[f'{c[0]}_{c[1]}_{c[2]}_imb2'] = (max_-mid_)/(mid_-min_)
    
    gc.collect()
    
    return df


# After performing the feature engineering on dataframe clean up any +/- inf and NaNs

# In[8]:


df = feature_eng(df)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)
all_columns = df.columns


# Lets split our training data so stocks are split on date_id

# In[9]:


# create empty lists to store train and valid indices
train_rows = []
valid_rows = []

# batch_size is the number of date_ids to use for building training and validation sets
batch_size = 20
training_size = 15

# loop through each stock_id
for stock_id in df['stock_id'].unique():
    # get rows for current stock_id
    stock_df = df[df['stock_id'] == stock_id]
    
    # get max date_id for current stock_id
    max_date_id = stock_df['date_id'].max()

    # create a list of date_ids for training and validation
    date_ids = stock_df['date_id'].unique()

    for date_id in range(0, max_date_id-batch_size, batch_size):
        train_rows += list(stock_df[stock_df['date_id'].between(date_id, date_id+training_size)].index)
        valid_rows += list(stock_df[stock_df['date_id'].between(date_id+training_size, date_id+batch_size)].index)
    
    if max_date_id % batch_size != 0:
        train_rows += list(stock_df[stock_df['date_id'].between(max_date_id - (max_date_id % batch_size), max_date_id)].index)


# Update splits to use the train_rows and valid_rows we created above.

# In[10]:


dls = TabularPandas(
    df, splits=(train_rows, valid_rows),
    procs = [Categorify, FillMissing, Normalize],
    cat_names = ['stock_id', 'imbalance_buy_sell_flag', 'seconds_in_bucket'],
    cont_names = [c for c in all_columns if c not in ['stock_id', 'imbalance_buy_sell_flag', 'seconds_in_bucket', 'target', 'row_id', 'time_id']],
    y_names='target', y_block=RegressionBlock(),
).dataloaders(path='.', bs=4096)


# Examine if changing the number and size of each layer makes a difference.

# In[11]:


learn = tabular_learner(dls, layers=[1024, 512], loss_func=mae, metrics=mse)


# lr_find is useful to find a good learning rate which is usually between slide and valley.

# In[12]:


learn.lr_find(suggest_funcs=(slide, valley))


# Second parameter to fit is our learning rate. If valid_loss goes haywire during training we need to choose a smaller rate.

# In[13]:


learn.fit(16, 1e-3)


# ## Submit Predictions

# In[14]:


import optiver2023
env = optiver2023.make_env()
iter_test = env.iter_test()


# In[15]:


counter = 0
for (test, revealed_targets, sample_prediction) in iter_test:
    # add the feature engineering to the test set
    test = feature_eng(test)
    test.replace([np.inf, -np.inf], np.nan, inplace=True)
    test.fillna(0, inplace=True)
    dl = learn.dls.test_dl(test)
    sample_prediction['target'], _ = learn.get_preds(dl=dl)
    env.predict(sample_prediction)
    counter += 1


# In[ ]:




