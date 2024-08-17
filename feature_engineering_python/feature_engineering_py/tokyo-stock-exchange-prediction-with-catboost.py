#!/usr/bin/env python
# coding: utf-8

# # Tokyo Stock Exchange Prediction with CatBoost
# In this notebook, I will build a Tokyo Stock Exchange Prediction Model using CatBoost. To make it easy to get start with, I only use stock prices data for training.

# In[1]:


import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# In[2]:


class Config:
    dataset_path = "../input/jpx-tokyo-stock-exchange-prediction/"


# ### Loading data

# In[3]:


stock_list = pd.read_csv(f"{Config.dataset_path}stock_list.csv")
stock_list.head()


# In[4]:


trades = pd.read_csv(f"{Config.dataset_path}train_files/trades.csv")
trades.tail()


# In[5]:


stock_prices = pd.read_csv(f"{Config.dataset_path}train_files/stock_prices.csv")
stock_prices.head()


# In[6]:


financials = pd.read_csv(f"{Config.dataset_path}train_files/financials.csv")
financials.head()


# In[7]:


options = pd.read_csv(f"{Config.dataset_path}train_files/options.csv")
options.head()


# In[8]:


secondary_stock_prices = pd.read_csv(f"{Config.dataset_path}train_files/secondary_stock_prices.csv")
secondary_stock_prices.head()


# ## Feature Engineering

# In[9]:


def feature_engineering(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df["year"] = df.Date.dt.year
    df["month"] = df.Date.dt.month
    df["day"] = df.Date.dt.day
    df['dayofweek'] = df.Date.dt.dayofweek
    df['hour'] = df.Date.dt.hour
    df.pop("Date")
    df.pop("RowId")
    return df


# In[10]:


stock_prices = feature_engineering(stock_prices)
stock_prices.head()


# In[11]:


target = stock_prices.pop("Target")
target.fillna(0, inplace=True)


# ## Train Validation Split
# I will keep last 10% data as hold-out set.

# In[12]:


validation_split = 0.1
split_index = int(len(secondary_stock_prices) * (1 - validation_split))
X_train = stock_prices.iloc[0:split_index]
X_val = stock_prices.iloc[split_index:]
y_train = target.iloc[0:split_index]
y_val = target.iloc[split_index:]


# ## Modeling

# In[13]:


params = {
    'task_type' : 'GPU',
    'verbose' : 1000,
    "cat_features": ["SecuritiesCode"]
}
model = CatBoostRegressor(**params)
model.fit(X_train, y_train, eval_set=(X_val, y_val))


# ## Submission

# In[14]:


import jpx_tokyo_market_prediction
env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()
counter = 0
# The API will deliver six dataframes in this specific order:
for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    if counter == 0:
        print(prices.head())
        print(options.head())
        print(financials.head())
        print(trades.head())
        print(secondary_prices.head())
        print(sample_prediction.head())
    codes = list(sample_prediction["SecuritiesCode"])
    second_codes = secondary_prices["SecuritiesCode"].unique()
    prices = feature_engineering(prices)
    secondary_prices = feature_engineering(secondary_prices)
    y_pred = model.predict(prices).reshape(-1)
    prediction_dict = dict([(str(code), target) for code, target in zip(codes, list(y_pred))])
    ranks = np.argsort(-1 * np.array(list(prediction_dict.values())), axis=0)
    sample_prediction['Rank'] = ranks
    env.predict(sample_prediction)
    counter += 1

