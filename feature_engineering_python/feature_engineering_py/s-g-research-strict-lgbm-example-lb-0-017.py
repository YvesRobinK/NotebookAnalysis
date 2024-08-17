#!/usr/bin/env python
# coding: utf-8

# # 🪙 G-Research Crypto - Starter LGBM Pipeline
# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/30894/logos/header.png)
# 
# 
# 
# 
# ### This is just a copy of the original [🪙💲 G-Research- Starter LGBM Pipeline](https://www.kaggle.com/julian3833/g-research-starter-lgbm-pipeline), but without the LB score contamination with the "leaky" data. Therefore, this is a valid "`[S]`" (or "`[Strict]`") notebook, as it follows the convention.
# 
# ---
# 
# # [[S] 🪙💲 Proposal for a meaningful LB](https://www.kaggle.com/julian3833/s-proposal-for-submission-common-ground)
# # 👆👆👆 A proposal for an agreement to get comparable, non-leaky models
# 
# 
# ---
# 
# # Not sure what I am talking about? Check these links:
# * __[Watch out!: test LB period is contained in the train csv](https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/285505) (topic)__
# * __[🪙💲 G-Research- Using the overlap fully [LB=0.99]](https://www.kaggle.com/julian3833/g-research-using-the-overlap-fully-lb-0-99) (notebook)__
# * __[Meaningful submission scores / sharing the lower boundary of public test data](https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/285289) (topic)__
# 
# 
# ---
# 

# # Import and load dfs
# 
# References: [Tutorial to the G-Research Crypto Competition](https://www.kaggle.com/cstein06/tutorial-to-the-g-research-crypto-competition)

# In[1]:


import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
import gresearch_crypto


TRAIN_CSV = '/kaggle/input/g-research-crypto-forecasting/train.csv'
ASSET_DETAILS_CSV = '/kaggle/input/g-research-crypto-forecasting/asset_details.csv'

def read_csv_strict(file_name='/kaggle/input/g-research-crypto-forecasting/train.csv'):
    df = pd.read_csv(file_name)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df[df['datetime'] < '2021-06-13 00:00:00']
    return df


# In[2]:


df_train = read_csv_strict()


# In[3]:


df_asset_details = pd.read_csv(ASSET_DETAILS_CSV).sort_values("Asset_ID")
df_asset_details


# # Training

# ## Utility functions to train a model for one asset
# 
# ### Features from [G-Research - Starter [0.361 LB]](https://www.kaggle.com/danofer/g-research-starter-0-361-lb)
# ### And [[GResearch] Simple LGB Starter](https://www.kaggle.com/code1110/gresearch-simple-lgb-starter#Feature-Engineering)

# In[4]:


# Two new features from the competition tutorial
def upper_shadow(df):
    return df['High'] - np.maximum(df['Close'], df['Open'])

def lower_shadow(df):
    return np.minimum(df['Close'], df['Open']) - df['Low']

# A utility function to build features from the original df
# It works for rows to, so we can reutilize it.
def get_features(df, row=False):
    df_feat = df[['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']].copy()
    df_feat['Upper_Shadow'] = upper_shadow(df_feat)
    df_feat['Lower_Shadow'] = lower_shadow(df_feat)
    
    
    df_feat["Close/Open"] = df_feat["Close"] / df_feat["Open"] 
    df_feat["Close-Open"] = df_feat["Close"] - df_feat["Open"] 
    df_feat["High-Low"] = df_feat["High"] - df_feat["Low"] 
    df_feat["High/Low"] = df_feat["High"] / df_feat["Low"]
    if row:
        df_feat['Mean'] = df_feat[['Open', 'High', 'Low', 'Close']].mean()
    else:
        df_feat['Mean'] = df_feat[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    
    df_feat['High/Mean'] = df_feat['High'] / df_feat['Mean']
    df_feat['Low/Mean'] = df_feat['Low'] / df_feat['Mean']
    df_feat['Volume/Count'] = df_feat['Volume'] / (df_feat['Count'] + 1)

    ## possible seasonality, datetime  features (unlikely to me meaningful, given very short time-frames)
    ### to do: add cyclical features for seasonality
    times = pd.to_datetime(df["timestamp"],unit="s",infer_datetime_format=True)
    if row:
        df_feat["hour"] = times.hour  # .dt
        df_feat["dayofweek"] = times.dayofweek 
        df_feat["day"] = times.day 
    else:
        df_feat["hour"] = times.dt.hour  # .dt
        df_feat["dayofweek"] = times.dt.dayofweek 
        df_feat["day"] = times.dt.day 
    #df_feat.drop(columns=["time"],errors="ignore",inplace=True)  # keep original epoch time, drop string

    return df_feat


def get_Xy_and_model_for_asset(df_train, asset_id):
    df = df_train[df_train["Asset_ID"] == asset_id]
    
    # TODO: Try different features here!
    df_proc = get_features(df)
    df_proc['y'] = df['Target']
    df_proc = df_proc.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    
    X = df_proc.drop("y", axis=1)
    y = df_proc["y"]

    # TODO: Try different models here!
    model = LGBMRegressor(n_estimators=10)
    model.fit(X, y)
    return X, y, model


# ## Loop over all assets

# In[5]:


Xs = {}
ys = {}
models = {}

for asset_id, asset_name in zip(df_asset_details['Asset_ID'], df_asset_details['Asset_Name']):
    print(f"Training model for {asset_name:<16} (ID={asset_id:<2})")
    X, y, model = get_Xy_and_model_for_asset(df_train, asset_id)    
    Xs[asset_id], ys[asset_id], models[asset_id] = X, y, model


# In[6]:


# Check the model interface
x = get_features(df_train.iloc[1], row=True)
y_pred = models[0].predict([x])
y_pred[0]


# # Predict & submit
# 
# References: [Detailed API Introduction](https://www.kaggle.com/sohier/detailed-api-introduction)
# 
# Something that helped me understand this iterator was adding a pdb checkpoint inside of the for loop:
# 
# ```python
# import pdb; pdb.set_trace()
# ```
# 
# See [Python Debugging With Pdb](https://realpython.com/python-debugging-pdb/) if you want to use it and you don't know how to.
# 

# In[7]:


env = gresearch_crypto.make_env()
iter_test = env.iter_test()

for i, (df_test, df_pred) in enumerate(iter_test):
    for j , row in df_test.iterrows():
        
        model = models[row['Asset_ID']]
        x_test = get_features(row, row=True)
        y_pred = model.predict([x_test])[0]
        
        df_pred.loc[df_pred['row_id'] == row['row_id'], 'Target'] = y_pred
        
        
        # Print just one sample row to get a feeling of what it looks like
        if i == 0 and j == 0:
            display(x_test)

    # Display the first prediction dataframe
    if i == 0:
        display(df_pred)

    # Send submissions
    env.predict(df_pred)


# In[ ]:




