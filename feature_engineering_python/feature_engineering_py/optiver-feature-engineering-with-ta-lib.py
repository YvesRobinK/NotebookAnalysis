#!/usr/bin/env python
# coding: utf-8
I've noticed there are several technical analysis libraries in the stock market field.  
Why don't we utilize one for our feature engineering?
# ## TL;DR
# You can create various features with [TA-Lib](https://github.com/ta-lib/ta-lib-python#supported-indicators-and-functions).
# 
# ```python
# def pre_process(df):
#     # other features...
# 
#     # features using TA-Lib
#     grouped = temp_df.groupby(['stock_id', 'date_id'])
#     for col in ['volume', 'reference_price', 'imbalance_size', 'far_price', 'near_price']:
#         for window in [3, 5, 10]:
#             temp_df[f"{col}_MA{window}"] = grouped[col].transform(lambda x: talib.SMA(x, timeperiod=window))
# 
#     for col in ['reference_price', 'far_price', 'near_price']:
#         temp_df[f'{col}_RSI'] = grouped[col].transform(lambda x: talib.RSI(x))
# 
#     return temp_df.replace([np.inf, -np.inf], 0)
# ```

# ## 1. Install TA-Lib
# Add the dataset below as an input to your notebook and install it.
# - Dataset: https://www.kaggle.com/datasets/sunghoshim/ta-lib-0-4-28/
# 

# In[1]:


# Install TA-Lib without internet
get_ipython().system('dpkg -i /kaggle/input/ta-lib-0-4-28/libta.deb /kaggle/input/ta-lib-0-4-28/ta.deb')
get_ipython().system('pip install /kaggle/input/ta-lib-0-4-28/TA_Lib-0.4.28-cp310-cp310-linux_x86_64.whl')


# In[2]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from xgboost import plot_importance

import talib # TA-Lib


# In[3]:


df = pd.read_csv('/kaggle/input/optiver-trading-at-the-close/train.csv')
print(df.shape)
df.head()


# ## 2. Try out TA-Lib
# **Reference**
# - [{GitHub} ta-lib-python > Function API](https://github.com/ta-lib/ta-lib-python#function-api)
# - [(Korean) {위키독스} 시스템 트레이딩을 위한 데이터 사이언스 > TA-Lib 활용하기](https://wikidocs.net/186885)

# In[4]:


# SMA: Simple Moving Average
result = talib.SMA(df['bid_size'], timeperiod=3)  # 2 NaN values are expected
print('null count:', result.isna().sum())
result


# In[5]:


result = talib.SMA(df['reference_price'], timeperiod=3)  # 2 NaN values are expected
print('null count:', result.isna().sum(), "something wrong!!!")
result


# You can easly create features for technical analysis. However, our dataframe has multiple `stock_id`s.  
# **👉 Use `groupby()` with `stock_id`, `date_id`.**  
# (I assume that the last data from the previous date is not stronly correlated with today's data.)
# 
# Moreover, it seems that missing values are impacting the result.  
# **👉 Handle missing values.**

# ## 3. Handle missing values
# **Reference**
# - [{Notebook} Explain the Data📝 | LightGBM Baseline🚀](https://www.kaggle.com/code/a27182818/explain-the-data-lightgbm-baseline#1.-Explore-&-Explain-the-Data)
# - [{Discussion} Handling Missing Values: An Exploration and Strategy Discussion](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/445414)
# - [{Notebook} Handling Missing Values: An Exploration](https://www.kaggle.com/code/jmascacibar/handling-missing-values-an-exploration)
# - [{Discussion} Missing Values in Various Columns](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/444054)
# - [{Notebook} EDA.Part1-Missing values](https://www.kaggle.com/code/nolansmith/eda-part1-missing-values)

# In[6]:


# Referecne : https://www.kaggle.com/code/a27182818/explain-the-data-lightgbm-baseline#1.-Explore-&-Explain-the-Data
def inspect_columns(df):
    result = pd.DataFrame({
        'unique': df.nunique() == len(df),
        'nunique': df.nunique(),
        'null_count': df.isna().sum(),
        'null_pct': round((df.isnull().sum() / len(df)) * 100, 4),
        '1st_row': df.iloc[0],
        'random_row': df.iloc[np.random.randint(low=0, high=len(df))],
        'last_row': df.iloc[-1],
        'dtype': df.dtypes
    })
    return result


# In[7]:


inspect_columns(df)


# - (1) As you can see in the `null_count` column, `imbalance_size`, `reference_price`, `matched_size`, `bid_price`, `ask_price`, and `wap` each have 220 rows of NaN values.
# - (2) `far_price` and `near_price` have a similar number of NaN values.
# - (3) `target` also has some NaN values.

# ## . 3.1. `imbalance_size` and others (200 NaNs)
# 
# **👉 Drop rows where `imbalance_size` is NaN.**

# In[8]:


df_missing1 = df[df['imbalance_size'].isna()]
print(df_missing1.shape)
df_missing1.head()


# In[9]:


df_missing1['stock_id'].value_counts()


# In[10]:


df_missing1['date_id'].value_counts()


# It seems that all NaNs are on specific dates for specific stocks.  
# Since it's just a small portion, let's just drop the rows.

# In[11]:


# 👉 Drop rows
df = df.dropna(subset=['imbalance_size'])
df.reset_index(drop=True, inplace=True)
print(df.shape)


# In[12]:


inspect_columns(df)


# Cool! NaNs in `target` have also disappeared.

# In[13]:


result = talib.SMA(df['reference_price'], timeperiod=3)  # 2 NaN values are expected
print('null count:', result.isna().sum(), "cool!!!")
result


# ## . 3.2. `far_price`, `near_price`
# 
# 👉 No action is required.

# In[14]:


df_missing2 = df[(df['far_price'].isna()) & (df['seconds_in_bucket'] >= 300)]
print(df_missing2.shape)
df_missing2.head()


# In[15]:


df_missing2['near_price'].isna().sum()  # near_price is alright. (no NaN after seconds_in_bucket 300)


# In[16]:


df_missing2['stock_id'].value_counts()  # 199 / 200


# In[17]:


df_missing2['stock_id'].value_counts()[-10:]


# In[18]:


df_missing2['date_id'].value_counts()  # 479 / 481


# In[19]:


df_missing2['seconds_in_bucket'].value_counts()  # looks like `far_price` could start not at seconds_in_bucket 300.


# In[20]:


df_missing2.query('stock_id==79')


# It seems like `far_price` for a specific stock is not available exactly from `seconds_in_bucket` 300.  
# However, once it starts being recorded, there are no NaN values for that day.
# 
# <br/>
# 
# Let's validate the above assumption.

# In[21]:


df_temp = df_missing2.groupby(['stock_id', 'date_id'])['seconds_in_bucket'].agg(['min', 'max', 'count']).reset_index()
df_temp


# In[22]:


df_temp['expected_count'] = (df_temp['max']/10 - df_temp['min']/10 + 1).astype(int)
df_temp


# In[23]:


(df_temp['count'] != df_temp['expected_count']).sum()


# That's a relief!  
# 
# Since NaN values are present in the earlier part of a specific stock on a specific date, we can also apply TA-Lib for `far_price`.

# ## 4. Try TA-Lib with GroupBy

# In[24]:


# Use only a small subset of dates for testing
df_play = df[(df['date_id'] >= 240) & (df['date_id'] < 250)].copy()
print(df_play.shape)
df_play.head()


# In[25]:


# Check the result of groupby > apply
df_play.groupby(['stock_id', 'date_id'])['reference_price'].apply(lambda x: talib.MA(x, timeperiod = 3))


# In[26]:


grouped_play = df_play.groupby(['stock_id', 'date_id'])


# In[27]:


# Check a single group (stock_id, date_id)
talib.MA(grouped_play.get_group((79, 249))['reference_price'], timeperiod = 3)


# In[28]:


# Check a single group (stock_id, date_id)
talib.MA(grouped_play.get_group((79, 249))['far_price'], timeperiod = 3)


# In[29]:


# groupby > transform
df_play['reference_price_MA3']  = grouped_play['reference_price'].transform(lambda x: talib.MA(x,timeperiod = 3))
df_play['far_price_MA3']  = grouped_play['far_price'].transform(lambda x: talib.MA(x,timeperiod = 3))
df_play


# In[30]:


# Check derived values with original values
pd.DataFrame({
    'reference_price' : grouped_play.get_group((70, 249))['reference_price'].values,
    'reference_price_MA3': grouped_play.get_group((70, 249))['reference_price'].transform(lambda x: talib.SMA(x,timeperiod = 3)).values,
    'reference_price_MA5': grouped_play.get_group((70, 249))['reference_price'].transform(lambda x: talib.SMA(x,timeperiod = 5)).values,
})


# ## 5. Prepare preprocess function

# In[31]:


def pre_process(df):
    temp_df = df.copy()
    temp_df["liquidity_imbalance"] = temp_df.eval("(bid_size-ask_size)/(bid_size+ask_size)")
    temp_df["volume"] = temp_df.eval("ask_size + bid_size")
    
    temp_df["price_spread"] = temp_df["ask_price"] - temp_df["bid_price"]
    temp_df['market_urgency'] = temp_df['price_spread'] * temp_df['liquidity_imbalance']
    
    temp_df["bid_size_over_ask_size"] = temp_df["bid_size"].div(temp_df["ask_size"])
    temp_df['price_diff'] = temp_df['reference_price'] - temp_df['wap']
    temp_df["bid_price_over_ask_price"] = temp_df["bid_price"].div(temp_df["ask_price"])
    
    temp_df['reference_price_wap_imb'] = temp_df.eval("(reference_price-wap) / (reference_price+wap)")
    temp_df['reference_price_bid_price_imb'] = temp_df.eval("(reference_price-bid_price) / (reference_price+bid_price)")   
    
    temp_df['far_price_near_price_imb'] = temp_df.eval("(far_price - near_price) / (far_price + near_price)")   
    temp_df['far_price_over_near_price'] = temp_df["far_price"].div(temp_df["near_price"])
    temp_df['far_price_minus_near_price'] = temp_df["far_price"] - temp_df["near_price"]
    
    # features using TA-Lib
    grouped = temp_df.groupby(['stock_id', 'date_id'])
    for col in ['volume', 'reference_price', 'imbalance_size', 'far_price', 'near_price']:
        for window in [3, 5, 10]:
            temp_df[f"{col}_MA{window}"] = grouped[col].transform(lambda x: talib.MA(x, timeperiod=window))

    for col in ['reference_price', 'far_price', 'near_price']:
        temp_df[f'{col}_RSI'] = grouped[col].transform(lambda x: talib.RSI(x))

    # Replace infinite values with 0 ()
    return temp_df.replace([np.inf, -np.inf], 0)


# In[32]:


print(df.shape)
df.head()


# In[33]:


get_ipython().run_cell_magic('time', '', '\ndf_train = pre_process(df)\nprint(df_train.shape)\ndf_train.tail()\n')


# In[34]:


X = df_train.drop(['row_id', 'time_id', 'date_id', 'target'], axis=1)
print(X.shape)


# ## 6. Train XGBoost model
# 
# **Reference**
# - [{Notebook} Optiver | Cross Validation Strategies](https://www.kaggle.com/code/sunghoshim/optiver-cross-validation-strategies#Holdout)
# - [{Notebook} Optiver | Domain Knowledge + XGB baseline (Korean)](https://www.kaggle.com/code/sunghoshim/optiver-domain-knowledge-xgb-baseline-korean)

# In[35]:


date_ids = np.arange(481)

split_day = 435  # 435 / 481 = 90%
train_index = date_ids <= split_day
valid_index = date_ids > split_day

df_fold_train = X[train_index[df_train['date_id']]]
df_fold_train_target = df_train.loc[train_index[df_train['date_id']], 'target']
df_fold_valid = X[valid_index[df_train['date_id']]]
df_fold_valid_target = df_train.loc[valid_index[df_train['date_id']], 'target']
print(len(df_fold_train), len(df_fold_valid))


# In[36]:


CONFIG = dict()
CONFIG['n_estimators'] = 3500
CONFIG['max_depth'] = 6
CONFIG['learning_rate'] = 0.01
CONFIG['early_stopping_rounds'] = 100


# In[37]:


get_ipython().run_cell_magic('time', '', "\nxgb_model = XGBRegressor(\n    **CONFIG,\n    objective = 'reg:squarederror',\n    eval_metric = 'mae',\n    tree_method = 'gpu_hist',\n    random_state = 123,\n)\n\nxgb_model.fit(\n    df_fold_train,\n    df_fold_train_target,\n    eval_set=[(df_fold_valid, df_fold_valid_target)],\n    verbose = 100,\n)\n")


# In[38]:


xgb_model.best_iteration, xgb_model.best_score


# In[39]:


fig, ax = plt.subplots(figsize=(10,10))
plot_importance(xgb_model, importance_type="gain", ax=ax, max_num_features=30, height=0.6)


# In[40]:


fig, ax = plt.subplots(figsize=(10,10))
plot_importance(xgb_model, importance_type="weight", ax=ax, max_num_features=30, height=0.6)


# In[41]:


fig, ax = plt.subplots(figsize=(10,10))
plot_importance(xgb_model, importance_type="cover", ax=ax, max_num_features=30, height=0.6)

