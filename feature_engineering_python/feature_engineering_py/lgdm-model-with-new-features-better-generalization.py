#!/usr/bin/env python
# coding: utf-8

# **Parent kernal [link ](https://www.kaggle.com/julian3833/g-research-starter-lgbm-pipeline-lb)**
# 
# **changes n_estimators=1000,num_leaves=500,max_depth=10**

# In[1]:


import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import gresearch_crypto


TRAIN_CSV = '/kaggle/input/g-research-crypto-forecasting/train.csv'
ASSET_DETAILS_CSV = '/kaggle/input/g-research-crypto-forecasting/asset_details.csv'


# In[2]:


df_train = pd.read_csv(TRAIN_CSV)
df_train.head()


# In[3]:


df_asset_details = pd.read_csv(ASSET_DETAILS_CSV).sort_values("Asset_ID")
df_asset_details


# # Training

# ## Utility functions to train a model for one asset

# In[4]:


from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
# Two new features from the competition tutorial
def upper_shadow(df):
    return df['High'] - np.maximum(df['Close'], df['Open'])

def lower_shadow(df):
    return np.minimum(df['Close'], df['Open']) - df['Low']

# A utility function to build features from the original df
# It works for rows to, so we can reutilize it.
def get_features(df):
    df_feat = df[['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']].copy()
    df_feat['upper_Shadow'] = upper_shadow(df_feat)
    df_feat['lower_Shadow'] = lower_shadow(df_feat)
    df_feat["high_div_low"] = df_feat["High"] / df_feat["Low"]
    #df_feat["open_sub_close"] = df_feat["Open"] - df_feat["Close"]
    df_feat['trade']=df_feat['Close']-df_feat['Open']
    df_feat['gtrade']=df_feat['trade']/df_feat['Count']
    df_feat['shadow1']=df_feat['trade']/df_feat['Volume']
    #df_feat['shadow2']=df_feat['upper_Shadow']/df['Low']
    df_feat['shadow3']=df_feat['upper_Shadow']/df['Volume']
    #df_feat['shadow4']=df_feat['lower_Shadow']/df['High']
    df_feat['shadow5']=df_feat['lower_Shadow']/df['Volume']
    
    df_feat['upper_Shadow_log']=np.log(df_feat['upper_Shadow'])
    df_feat['lower_Shadow_log']=np.log(df_feat['lower_Shadow'])
    return df_feat
def log(model,X_train, X_valid, y_train, y_valid,train_split=1.0):
    if train_split > 0:
        X_train=X_train[:int(train_split*X_train.shape[0])]
        y_train=y_train[:int(train_split*y_train.shape[0])]
    
        pred=model.predict(X_train)
        print('Training :- ')
        print(f'MSE : {np.mean((y_train-pred)**2)}')
        print(f'CV : {pearsonr(pred,y_train)[0]}')
    pred=model.predict(X_valid)
    print('Validation :- ')
    print(f'MSE : {np.mean((y_valid-pred)**2)}')
    print(f'CV : {pearsonr(pred,y_valid)[0]}')

def get_Xy_and_model_for_asset(df_train, asset_id):
    df = df_train[df_train["Asset_ID"] == asset_id]
   
    # TODO: Try different features here!
    df_proc = get_features(df)
    df_proc['y'] = df['Target']
    df_proc = df_proc.dropna(how="any")
    
    X = df_proc.drop("y", axis=1)
    y = df_proc["y"]
    X_train=X[:int(0.7*X.shape[0])]
    y_train=y[:int(0.7*y.shape[0])]#
    X_test=X[int(X.shape[0]*0.7):]
    y_test=y[int(y.shape[0]*0.7):]
    # TODO: Try different models here!
    model = LGBMRegressor(n_estimators=200,num_leaves=300,learning_rate=0.09)
    model.fit(X_train, y_train)
    print('[Finished Training] evaluating')
    log(model,X_train, X_test, y_train, y_test,0.3)
    
    
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
x = get_features(df_train.iloc[1])
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
        x_test = get_features(row)
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





# 
