#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gc

import warnings
warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


# # Load and transform functions

# The function "feat_eng" is designed for feature engineering on a DataFrame. It takes a DataFrame "df" as input and performs several operations to transform and create new features. Here is a description of what this function does:
# 
# **Data Type Conversion**:
#     Converts the 'series_id' column to a categorical data type.
#     Converts the 'timestamp' column to a datetime data type, and removes the time zone information.
# 
# **Time-Related Feature Creation**:
#     Creates a new 'hour' column by extracting the hour from the 'timestamp' column.
#     Sorts the DataFrame based on the 'timestamp' and sets 'timestamp' as the index.
# 
# **Outlier Limiting**:
#     Limits outliers in the 'enmo' column by clipping values at an upper limit of 4.
# 
# **lids**:
#     Calculates a modified 'lids' feature by taking the maximum of 0 and the difference between 'enmo' and 0.02.
#     Performs a rolling sum of 'lids' over a 5-minute window and scales the result to a percentage.
#     Computes the mean of the rolling 'lids' over a 15-minute window. This Feature I got from the great work on [LightGBM + Feature Implementation from Paper](https://www.kaggle.com/code/rimbax/lightgbm-feature-implementation-from-paper).
#     
# **anglezdiffabs**:
# Is a term formed from "anglez", that it represents the absolute differences between consecutive values of an angle, likely in degrees or radians. I got the anglez diff abs from the insightul analysis on [Feature Engineering and Random Forest Prediction](https://www.kaggle.com/code/lccburk/feature-engineering-and-random-forest-prediction)
# 
# **Data Type Conversion**:
#     Converts 'enmo' and 'anglez' columns to integer types.
#     Calculates the absolute differences between consecutive values in the 'anglez' column and converts them to a float data type.
# 
# **Feature Engineering with Rolling Statistics**:
#     For columns 'enmo', 'anglez', and 'anglezdiffabs', the function calculates various rolling statistics (median, mean, max, min, and variance) over different time periods (60, 360, and 720 seconds).
#     Additionally, it calculates the Median Absolute Deviation (MAD) for the last time period (720 seconds) for 'enmo'.
#     Computes the amplitude (difference between max and min values) and the minimum amplitude over rolling windows.
#     Calculates the maximum absolute differences between consecutive values over rolling windows for both 'anglez' and 'anglezdiffabs'.
# 
# **Data Cleaning**:
#     Resets the index of the DataFrame and removes rows with missing values.
# 
# **Output**:
#     Returns the modified DataFrame with the newly engineered features.

# In[2]:


def load_serie_by_id(idx):
    
    df = pd.read_parquet(path+'train_series.parquet', filters=[('series_id','=',idx)])
    df['series_id'] = df['series_id'].astype('category')
    df['timestamp'] = pd.to_datetime(df['timestamp']).apply(lambda t: t.tz_localize(None))
    
    return df


def get_df_by_freq(df, freq, col):
    
    df = df.copy()
    df.set_index('timestamp', inplace=True)
    df_ohlc = df[col].resample(freq).ohlc()
    df_ohlc.columns = [f'{c}_{col}' for c in df_ohlc.columns]
    
    df_ohlc = pd.merge(df_ohlc, df[['event', 'series_id']], left_index=True, right_index=True)
    
    return df_ohlc


# # Plot functions

# In[3]:


def plot_by_freq(df, freq, col):
    
    serie_id = df['series_id'].unique()[0]
    
    fig = go.Figure()

    # set up trace for wakeup
    df_wakeup = df[df['event']=='wakeup']
    fig.add_traces(go.Ohlc(x=df_wakeup.index,
                    open=df_wakeup[f'open_{col}'], high=df_wakeup[f'high_{col}'],
                    low=df_wakeup[f'low_{col}'], close=df_wakeup[f'close_{col}'],
                   increasing_line_color= 'orange', decreasing_line_color= 'orange', name='awake'
                ))

    # set up traces for onset
    df_onset = df[df['event']=='onset']
    fig.add_traces(go.Ohlc(x=df_onset.index,
                    open=df_onset[f'open_{col}'], high=df_onset[f'high_{col}'],
                    low=df_onset[f'low_{col}'], close=df_onset[f'close_{col}'],
                   increasing_line_color= 'blue', decreasing_line_color= 'blue', name='sleeping'
                ))

    fig.update_layout(title=f'Serie_id: {serie_id} - {freq} OLHC {col}', xaxis_rangeslider_visible=True)
    fig.show()
    

def plot_columns_by_freq(df, freq, columns):
    
    for col in columns:
        df_ohlc = get_df_by_freq(df, freq, col)
        plot_by_freq(df_ohlc, freq, col)


# In[4]:


SIGMA = 720 # 12 * 60
def gauss(n=SIGMA,sigma=SIGMA*0.15):
    # guassian distribution function
    r = range(-int(n/2),int(n/2)+1)
    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]



def get_macd(s, hfast=4, hslow=48, macd=15, center=False):
    
    sma_fast = s.rolling(hfast*12*60, min_periods=1, center=center).agg('mean')
    sma_slow = s.rolling(hslow*12*60, min_periods=1, center=center).agg('mean')
    macd = (sma_fast - sma_slow).rolling(12*macd, min_periods=1, center=center).mean().astype(np.float32)
    
    return macd


def macd_feat(df, col='anglezdiffabs', dif=120):
    
    s = df[f'{col}']
    df[f'{col}_macd'] = get_macd(s)

    s = df[f'{col}'].sort_index(ascending=False)
    df[f'{col}_macd_rev'] = get_macd(s).sort_index()
    
    df[f'{col}_spred'] = df[f'{col}_macd']-df[f'{col}_macd_rev']
    df[f'{col}_spread_diff']=df[f'{col}_spred'].diff(dif)
    
#     df.loc[df[f'{col}_spred']<3, 'wakeup_zone'] = 1
#     df['wakeup_zone'].fillna(0, inplace=True)
#     df.loc[df[f'{col}_spred']>-3, 'onset_zone'] = 1
#     df['onset_zone'].fillna(0, inplace=True)
    
    return df


def feat_eng(df):
    
    df['series_id'] = df['series_id'].astype('category')
    df['timestamp'] = pd.to_datetime(df['timestamp']).apply(lambda t: t.tz_localize(None))
    df['hour'] = df["timestamp"].dt.hour
    df['month'] = df["timestamp"].dt.month
    
    df.sort_values(['timestamp'], inplace=True)
    df.set_index('timestamp', inplace=True)
    
    # limit outiliers in enmo
    df['enmo'] = df['enmo'].clip(upper=4)
    
    df['lids'] = np.maximum(0., df['enmo'] - 0.02)
    df['lids'] = df['lids'].rolling(f'{120*5}s', center=True, min_periods=1).agg('sum')
    df['lids'] = 100 / (df['lids'] + 1)
    df['lids'] = df['lids'].rolling(f'{360*5}s', center=True, min_periods=1).agg('mean').astype(np.float32)
    
    df['lids_sma_fast'] = df['lids'].rolling('2h', center=True, min_periods=1).agg('mean').astype(np.float32)
    df['lids_sma_macd'] = (df['lids'].rolling('18h', center=True, min_periods=1).agg('mean') - df['lids_sma_fast']
                                 ).rolling(f'1h', center=True, min_periods=1).agg('mean').astype(np.float32)
    
    df["enmo"] = (df["enmo"]*1000).astype(np.int16)
    df["anglez"] = df["anglez"].astype(np.int16)
    df["anglezdiffabs"] = df["anglez"].diff().abs().astype(np.float32)
    
#     df = macd_feat(df, col='anglezdiffabs')
#     df = macd_feat(df, col='lids')

#     for col in ['enmo', 'anglez', 'anglezdiffabs']:
    for col in ['enmo', 'anglez']:
#         df[f'{col}_sma_fast'] = df[f'{col}'].rolling(f'4h', center=True, min_periods=1).agg('mean').astype(np.float32)
#         df[f'{col}_sf_macd'] = (df[f'{col}_sma_fast'] - df[f'{col}'].rolling(f'48h', center=True, min_periods=1).agg('mean')
#                                    ).rolling(f'1h', center=True, min_periods=1).agg('mean').astype(np.float32)
        
        # periods in seconds        
        periods = [720] 
        
        for n in periods:
            
            rol_args = {'window':f'{n+5}s', 'min_periods':10, 'center':True}
            
#             for agg in ['median', 'mean', 'max', 'min', 'std']:
            for agg in ['std']:
                df[f'{col}_{agg}_{n}'] = df[col].rolling(**rol_args).agg(agg).astype(np.float32).values
                gc.collect()
                
            rol_args = {'window':f'8h', 'min_periods':15, 'center':True}
            df[f'{col}_std_max_8h'] = df[f'{col}_std_{n}'].rolling(**rol_args).max().astype(np.float32).values
            
#             if n == max(periods):
#                 df[f'{col}_mad_{n}'] = (df[col] - df[f'{col}_median_{n}']).abs().rolling(**rol_args).median().astype(np.float32)
            
#             df[f'{col}_amplit_{n}'] = df[f'{col}_max_{n}']-df[f'{col}_min_{n}']
#             df[f'{col}_amplit_{n}_min'] = df[f'{col}_amplit_{n}'].rolling(**rol_args).min().astype(np.float32).values
            
#             df[f'{col}_diff_{n}_max'] = df[f'{col}_max_{n}'].diff().abs().rolling(**rol_args).max().astype(np.float32)
#             df[f'{col}_diff_{n}_mean'] = df[f'{col}_max_{n}'].diff().abs().rolling(**rol_args).mean().astype(np.float32)

    
    df.reset_index(inplace=True)
#     df.dropna(inplace=True)

    return df






# df = feat_eng_by_id(series_id[0])


# # Loading data

# In[5]:


path = '/kaggle/input/child-mind-institute-detect-sleep-states/'

train_events = pd.read_csv(path + 'train_events.csv')
train_events['timestamp'] = pd.to_datetime(train_events['timestamp']).apply(lambda t: t.tz_localize(None))
count_na = train_events.groupby('series_id')['timestamp'].apply(lambda x: x.isna().sum())


# # Plot Anglez standard deviation in 8h

# ## Plot series with null events

# In[6]:


for idx in count_na[count_na>=0].index[:3]:
# for idx in ['6ca4f4fca6a2']:
    print(f'ID: {idx}', "="*120, sep='\n')
    train_serie = load_serie_by_id(idx)
    df = train_serie.merge(train_events[train_events['series_id']==idx], on=['series_id', 'timestamp'], how='left').ffill()
    df = feat_eng(df)
    plot_columns_by_freq(df, '5min', ['anglez', 'anglez_std_720', 'anglez_std_max_8h'])


# ## Plot Series without null events

# In[7]:


for idx in count_na[count_na==0].index[:2]:
# for idx in ['6ca4f4fca6a2']:
    print(f'ID: {idx}', "="*120, sep='\n')
    train_serie = load_serie_by_id(idx)
    df = train_serie.merge(train_events[train_events['series_id']==idx], on=['series_id', 'timestamp'], how='left').ffill()
    df = feat_eng(df)
    plot_columns_by_freq(df, '5min', ['anglez', 'anglez_std_max_8h'])


# # Plot serie with weird patern

# In[8]:


for idx in ['6ca4f4fca6a2']:
    print(f'ID: {idx}', "="*120, sep='\n')
    train_serie = load_serie_by_id(idx)
    df = train_serie.merge(train_events[train_events['series_id']==idx], on=['series_id', 'timestamp'], how='left').ffill()
    df = feat_eng(df)
    plot_columns_by_freq(df, '5min', ['anglez', 'anglez_std_max_8h'])


# # Plot all features grouping by time

# In[9]:


for idx in count_na[count_na==0].index[:1]:
    print(f'ID: {idx}', "="*120, sep='\n')
    train_serie = load_serie_by_id(idx)
    df = train_serie.merge(train_events[train_events['series_id']==idx], on=['series_id', 'timestamp'], how='left').ffill()
    df = feat_eng(df)
    plot_columns_by_freq(df, '30min', sorted(df.select_dtypes(include=['float32']).columns[:5]))


# In[ ]:




