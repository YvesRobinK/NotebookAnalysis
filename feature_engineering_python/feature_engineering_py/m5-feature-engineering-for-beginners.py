#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import pandas as pd 
from  datetime import datetime, timedelta
import lightgbm as lgb
import gc
import os
import matplotlib.pyplot as plt

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## 1. Introduction 
# 
# In this notebook, i will show how to do feature engineering for timeseries forecasting problems. I will try to make it make simple and begineer friendly and at the same time will try cover as much depth as possible.
# 
# Video explaining what is timeseries and how to do timeseries analysis</br>

# In[77]:


from IPython.display import IFrame, YouTubeVideo
YouTubeVideo('e8Yw4alG16Q',width=600, height=400)


# In[2]:


BASE_PATH = '../input/m5-forecasting-accuracy'
MAX_LAGS = 70
TR_LAST = 1913
START_DATE = datetime(2016,4, 25)  # we are using data points where date > START_DATE to avoid memory errors
START_DATE



calendar_df = pd.read_csv(f'{BASE_PATH}/calendar.csv')
stv_df = pd.read_csv(f'{BASE_PATH}/sales_train_validation.csv')
sp_df = pd.read_csv(f'{BASE_PATH}/sell_prices.csv')
sample_df = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')


# #### Util function to reduce memory

# In[3]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics: 
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
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
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# ## 2. Exploring Dataset
# 
# For better EDA and data exploration please visit [this](https://www.kaggle.com/headsortails/back-to-predict-the-future-interactive-m5-eda) great kernel.

# In[4]:


stv_df.head()


# In[5]:


stv_df.shape


# 
# melting sales_train_validation dataframe to create demand column
# 
# To know more about what pd.melt does please visit [this](https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.melt.html) link.[](http://)

# In[6]:


# melting sales_train_validation dataframe to create demand column
id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
stv_df = pd.melt(stv_df, id_vars = id_vars, var_name = 'd', value_name = 'demand')


# In[7]:


print(f'DF shape after melting is {stv_df.shape}')

# since we have 30490 rows prior melting, we melted 1913 different columns ['d_1', 'd_2', 'd_3', ..., 'd_1911', 'd_1912', 'd_1913']
# 30490 * 1913 = 58327370


# In[8]:


stv_df.head()


# In[10]:


stv_df.store_id.unique()


# In[11]:


stv_df.dept_id.unique()


# > for illustration purpose and to make computation faster as well as avoiding memory errors i am taking data for only 1 store (CA_1)

# In[13]:


stv_df = stv_df.loc[stv_df.store_id == 'CA_1']
stv_df = stv_df.loc[stv_df.dept_id == 'HOBBIES_1']

# Now we have nearly 6 years of sales data for sales of dept_id 'HOBBIES_1' in store_id 'CA_1' in state 'CA'
stv_df.shape


# ## 3. Create Features

# ### 3.1 Date features
# 
# 
# ![Imgur](https://i.imgur.com/0UORwwz.png)
# 
# > Source: https://pandas.pydata.org/pandas-docs/version/0.23/api.html#datetimelike-properties
# 

# In[ ]:


cal_imp_cols = ['d', 'date', 'day', 'wday','week', 'month','quarter', 'year','is_weekend', 'event_name_1', 'event_type_1', ]

def add_date_features(df):
    # date time features
    df['date'] = pd.to_datetime(df['date'])
    attrs = ["year", "quarter", "month", "week", "day", "dayofweek", "is_year_end", "is_year_start", "is_quarter_end", \
        "is_quarter_start", "is_month_end","is_month_start",
    ]

    for attr in attrs:
        dtype = np.int16 if attr == "year" else np.int8
        df[attr] = getattr(df['date'].dt, attr).astype(dtype)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(np.int8)
    return df


calendar_df = add_date_features(calendar_df)
calendar_df = calendar_df[cal_imp_cols]
calendar_df = reduce_mem_usage(calendar_df)


# In[ ]:


calendar_df.head()


# ### 2.2 Creating Lag Features
# 
# Please read [this](https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/) and [this](https://www.analyticsvidhya.com/blog/2019/12/6-powerful-feature-engineering-techniques-time-series/) blog to know more about shift and rolling shift features.
# 
# 
# Consider this – you are predicting the demand for an item. So, the previous day’s demand is important to make a prediction, right? In other words, the value at time t is greatly affected by the value at time t-1. The past values are known as lags, so `t-1 is lag 1`, `t-7 is lag 7`,`t-28 is lag 28`  and so on.
# 
# ![](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/11/3hotmk.gif)

# ### How to find which lags to choose.
# 
# > The lag value we choose will depend on the correlation of individual values with its past values.
# 
# * There is more than one way of determining the lag at which the correlation is significant. For instance, we can use the `ACF (Autocorrelation Function)` and `PACF (Partial Autocorrelation Function)` plots.
# 
# * `ACF`: The ACF plot is a measure of the correlation between the time series and the lagged version of itself
# * `PACF`: The PACF plot is a measure of the correlation between the time series with a lagged version of itself but after eliminating the variations already explained by the intervening comparisons
# * For our particular example, here are the ACF and PACF plots:

# In[64]:


import statsmodels.api as sm


for i in range(0, 5):
    sm.graphics.tsa.plot_acf(list(stv_df.groupby(['id'])['demand'])[i][1], lags=40)
    sm.graphics.tsa.plot_pacf(list(stv_df.groupby(['id'])['demand'])[i][1], lags=40)
    
# below graph shows that window 1, 7, 14, 21, 28 are better choices.


# **We can calculate all sort of dataframe  statistics on rolling features**
#  
#  *  [sum, cumsum, min, max, mean, median, var, skew, kurt]

# In[71]:


# So, we choose window 1, 7 and 28 to calculate our lag features.


def create_lag_features(df):
    
    # shift and rolling demand features
    shifts = [1, 7, 28]
    for shift_window in shifts:
        df[f"shift_t{shift_window}"] = df.groupby(["id"])["demand"].transform(lambda x: x.shift(shift_window))
            
    # rolling mean
    windows = [7, 28]
    for val in windows:
        for col in [f'shift_t{win}' for win in windows]:
            df[f'roll_mean_{col}_{val}'] = df[['id', col]].groupby(['id'])[col].transform(lambda x: x.rolling(val).mean())
    
    # rolling standard deviation    
    for val in windows:
        for col in [f'shift_t{win}' for win in windows]:
            df[f'roll_std_{col}_{val}'] = df[['id', col]].groupby(['id'])[col].transform(lambda x: x.rolling(val).std())
    
    return df
    


# In[72]:


stv_df = create_lag_features(stv_df)


# In[75]:


stv_df.groupby(['id'])['shift_t1'].head(20)


# ## 3. Saving Features

# In[ ]:


stv_df.to_csv('features.csv')


# What about Version 3: 
# 
# In version 3 of this notebook, i will explore following features:
# 
# * Calculate item price related features
# * %age change in price over rolling window of [1, 7, 28] days.
# * More timeseries related features.
# * How to select top k important features
# 
# #### > See you soon.

# * Help taken from these kernels
# * https://www.kaggle.com/rohitsingh9990/m5-lgbm-fe
# 
# > Note: If you like my work, please, upvote ☺

# In[ ]:




