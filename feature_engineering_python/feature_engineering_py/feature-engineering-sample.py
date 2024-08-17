#!/usr/bin/env python
# coding: utf-8

# **Indicators used by stock investors to make trading decisions are calculated as features and evaluated by SHAP.**
# 
# **I'm a biginner, so please don't take my word for it.**

# # Import(Unnecessary libraries will be deleted later)

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

from tqdm import tqdm
import jpx_tokyo_market_prediction
from sklearn.model_selection import train_test_split
import warnings; warnings.filterwarnings("ignore")

import gc

import datetime
import time
# import locale


# # Data Loading

# In[2]:


prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")
sprices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/secondary_stock_prices.csv")
supplemental_prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv")
supplemental_sprices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/secondary_stock_prices.csv")


# # Preprocessing 

# In[3]:


prices=prices.append(sprices,ignore_index=True)
prices=prices.append(supplemental_prices,ignore_index=True)
prices=prices.append(supplemental_sprices,ignore_index=True)
prices=prices.drop(['RowId','ExpectedDividend'],axis=1)
prices=prices.dropna()


# Create data for training and validation. The time period chosen here is not meaningful, but it is open for consideration.In this case, it is up to feature generation, so it is hardly used at all.

# In[4]:


prices['DateValue']=prices['Date'].str.replace('-','')
xprices=prices[prices['DateValue']<'20220401']
xprices=xprices.drop(['DateValue'],axis=1)

yprices=prices[prices['DateValue']>='20220401']
yprices=yprices.drop(['DateValue'],axis=1)


# In[5]:


y_train=xprices.pop('Target')
X_train=xprices

y_test=yprices.pop('Target')
X_test=yprices


# # Feature Engineering

# In[6]:


def featuring(train):
    dfa=pd.DataFrame()
    for code in train['SecuritiesCode'].unique():
        df=train[train['SecuritiesCode']==code]

        df=df.sort_values(by=['Date'], ascending=True)
        
        # Moving averages - different periods
        df['MA200'] = df['Close'].rolling(window=200, min_periods=1).mean() 
        df['MA100'] = df['Close'].rolling(window=100, min_periods=1).mean() 
        df['MA50'] = df['Close'].rolling(window=50, min_periods=1).mean() 
        df['MA26'] = df['Close'].rolling(window=26, min_periods=1).mean() 
        df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean() 
        df['MA5'] = df['Close'].rolling(window=5, min_periods=1).mean() 
        df['MA14_low'] = df['Low'].rolling(window=14, min_periods=1).min()
        df['MA14_high'] = df['High'].rolling(window=14, min_periods=1).max()
    
        df['MA20dSTD'] = df['Close'].rolling(window=20, min_periods=1).std() 
        df['MA20dSTD'] = df['MA20dSTD'].fillna(method ='bfill')#missing-value complement
        
        #SMA Differences - different periods
        df['DIFF-MA200-MA50'] = df['MA200'] - df['MA50']
        df['DIFF-MA200-MA100'] = df['MA200'] - df['MA100']
        df['DIFF-MA200-CLOSE'] = df['MA200'] - df['Close']
        df['DIFF-MA100-CLOSE'] = df['MA100'] - df['Close']
        df['DIFF-MA50-CLOSE'] = df['MA50'] - df['Close']
        
        # Exponential Moving Averages (EMAS) - different periods
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # Relative Strength Index (RSI)　70~80:overbought, 20~30:overselling
        df['K-ratio'] = 100*((df['Close'] - df['MA14_low']) / (df['MA14_high'] - df['MA14_low']) )
        df['RSI'] = df['K-ratio'].rolling(window=3, min_periods=1).mean() 
        
        # Bollinger Bands ±2σ
        df['Bollinger_Upper'] = df['MA20'] + (df['MA20dSTD'] * 2)
        df['Bollinger_Lower'] = df['MA20'] - (df['MA20dSTD'] * 2)
        
        # Breakout Indicator(ORIGINAL)
        df['Close_diff_Upper'] = df['Close'] - df['Bollinger_Upper'] 
        df['Close_diff_Lower'] = df['Bollinger_Lower'] - df['Close']
        # 0 if the closing price is between the upper and lower bands, as it is important that the closing price is outside the Bollinger Bands.
        df.loc[df['Close_diff_Upper'] < 0, 'Close_diff_Upper'] = 0
        df.loc[df['Close_diff_Lower'] < 0, 'Close_diff_Lower'] = 0
        
        # Moving Average Convergence/Divergence (MACD)　Golden and dead crosses appear earlier in MACD than seen in Bollinger Bands
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].rolling(window=9, min_periods=1).mean()
        df['Hist'] = df['MACD'] - df['Signal']
        
        dfa=dfa.append(df)

    dfa['year']=pd.to_numeric(dfa['Date'].str[0:4]).astype(float)
    dfa['month']=pd.to_numeric(dfa['Date'].str[5:7]).astype(float)
    dfa['day']=pd.to_numeric(dfa['Date'].str[8:10]).astype(float)
    dfa['delta']=pd.to_numeric(dfa['High']-dfa['Low']).astype(float)
    dfa['change']=pd.to_numeric(dfa['Close']-dfa['Open']).astype(float)

    #Day of Week Information　(In general, Mondays are volatile.)
    dfa['DATE'] = pd.to_datetime(dfa['Date'])
    dfa['weekday'] = dfa['DATE'].dt.weekday+1
    dfa['Monday'] = np.where(dfa['weekday']==1,1,0)
    dfa['Tuesday'] = np.where(dfa['weekday']==2,1,0)
    dfa['Wednesday'] = np.where(dfa['weekday']==3,1,0)
    dfa['Thursday'] = np.where(dfa['weekday']==4,1,0)
    dfa['Friday'] = np.where(dfa['weekday']==5,1,0)

    train=train.merge(dfa,how='left',on=['Date','SecuritiesCode'],suffixes=('', 'b')).set_axis(train.index)
    train=train.drop(['Date'],axis=1)
    
    return train


# In[7]:


X_train=featuring(X_train[X_train['SecuritiesCode'] == 1301])
X_test=featuring(X_test[X_test['SecuritiesCode'] == 1301])


# # Visualization sample

# In[8]:


#Bollinger Band
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=[10,4.2])
plt.plot(list(X_train['DATE']), X_train['Close'], label='close')
plt.plot(list(X_train['DATE']), X_train['MA200'], label='MA200')
plt.plot(list(X_train['DATE']), X_train['MA100'], label='MA100')
plt.plot(list(X_train['DATE']), X_train['MA50'], label='MA50')
plt.plot(list(X_train['DATE']), X_train['MA20'], label='MA20')
plt.plot(list(X_train['DATE']), X_train['MA5'], label='MA5')
plt.legend()


# In[9]:


#Bollinger Band
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=[10,4.2])
plt.plot(list(X_train['DATE']), X_train['Close'], label='close')
plt.plot(list(X_train['DATE']), X_train['Bollinger_Upper'], label='bolinger_Upper')
plt.plot(list(X_train['DATE']), X_train['Bollinger_Lower'], label='bolinger_Lower')
plt.legend()


# In[10]:


#Breakout Indicator
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=[10,4.2])
plt.plot(list(X_train['DATE']), X_train['Close_diff_Upper'], label='Close_diff_Upper')
plt.plot(list(X_train['DATE']), X_train['Close_diff_Lower'], label='Close_diff_Lower')
plt.legend()


# In[11]:


#MACD
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=[10,4.2])
plt.plot(list(X_train['DATE']), X_train['MACD'], label='MACD')
plt.plot(list(X_train['DATE']), X_train['Signal'], label='Signal')
plt.bar(list(X_train['DATE']),X_train['Hist'], label='Hist')
plt.legend()


# In[12]:


#RSI
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=[10,4.2])
plt.plot(list(X_train['DATE']), X_train['RSI'], label='RSI')
plt.legend()


# # under construction
