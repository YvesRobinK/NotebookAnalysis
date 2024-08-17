#!/usr/bin/env python
# coding: utf-8

# # [Tourist Arrivals] 時系列データにおける特徴量エンジニアリング

# ## ライブラリのインポート

# In[1]:


# ライブラリのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt


# ## データ読み込み

# In[2]:


# データ読み込み
train_df = pd.read_csv('/kaggle/input/prediction-of-tourist-arrivals/train_df.csv')

# datetime型へ変換
train_df['date'] = pd.to_datetime(train_df['date'])
print(f'train_df.shape = {train_df.shape}')
train_df.info()


# # 三角関数特徴量

# 時系列のコンペでは、特徴量として時間や日付を組み込むことがしばしばあります。  
# しかし、時間（hour）を例に考えてみると、23時の次は0時ですがこの間には数値的なつながりがありません。  
# そこで循環性を考慮できるように三角関数を各特徴量に適用させます。

# In[3]:


# 年、月、日、曜日を作成
train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day
train_df['dayofweek'] = train_df['date'].dt.dayofweek


# In[4]:


train_df


# これで時系列の情報を抽出できました。

# In[5]:


train_df[train_df['month'] == 8]['dayofweek'].plot(grid=True);


# これで8月の曜日の情報を見てみると、日曜日（dayofweek == 6）の次が月曜日（dayofweek == 0）になっており、  
# 隣り合った日であっても数値的には隣り合っていません。  
# そこで、循環性を持つように三角関数を使用して特徴量を変換します。

# In[6]:


def encode(df, col):
    # この方法だと場合によって最大値が変化するデータでは正確な値は出ない
    # 例：月の日数が30日や31日の場合がある
    df[col + '_cos'] = np.cos(2 * np.pi * df[col] / df[col].max())
    df[col + '_sin'] = np.sin(2 * np.pi * df[col] / df[col].max())
    return df


# In[7]:


train_df = encode(train_df, 'month')
train_df = encode(train_df, 'day')
train_df = encode(train_df, 'dayofweek')


# In[8]:


train_df


# In[9]:


# 日の循環性を表示
train_df.plot.scatter('day_sin', 'day_cos').set_aspect('equal')


# # 曜日特徴量

# In[10]:


train_df['Mon']=train_df['dayofweek']==0
train_df['Tue']=train_df['dayofweek']==1
train_df['Wed']=train_df['dayofweek']==2
train_df['Thu']=train_df['dayofweek']==3
train_df['Fri']=train_df['dayofweek']==4
train_df['Sat']=train_df['dayofweek']==5
train_df['Sun']=train_df['dayofweek']==6


# In[11]:


train_df


# # まとめ（関数化）

# In[12]:


def encode(df):
    df['day_of_week']=(df['date'].dt.dayofweek.values).astype(np.int64)
    df['weekday']=(df['day_of_week']<5)
    df['Mon']=(df['day_of_week']==0)
    df['Tue']=(df['day_of_week']==1)
    df['Wed']=(df['day_of_week']==2)
    df['Thu']=(df['day_of_week']==3)
    df['Fri']=(df['day_of_week']==4)
    df['Sat']=(df['day_of_week']==5)
    df['Sun']=(df['day_of_week']==6)
    df['sin_day_of_week']=np.sin(2*np.pi*df['day_of_week']/7)
    df['cos_day_of_week']=np.cos(2*np.pi*df['day_of_week']/7) 
                          
    df['month']=(df['date'].dt.month.values).astype(np.int64)
    
    df['spring']=(df['month']>=3)&(df['month']<=5)
    df['summer']=(df['month']>=6)&(df['month']<=8)
    df['fall']=(df['month']>=9)&(df['month']<=11)
    df['winter']=(df['month']==12)&(df['month']<=2)
    
    df['week_of_year']=(df['date'].dt.isocalendar().week.values).astype(np.int64)
    df['sin_week_of_year']=np.sin(2*np.pi*df['day_of_week']/52)
    df['cos_week_of_year']=np.cos(2*np.pi*df['day_of_week']/52) 
    
    df['day_of_year'] = (df['date'].dt.dayofyear.values).astype(np.int64)
    df['sin_day_of_year']=np.sin(2*np.pi*df['day_of_year']/365)
    df['cos_day_of_year']=np.cos(2*np.pi*df['day_of_year']/365)
    
    df['day_of_month']=(df['date'].dt.day.values).astype(np.int64)
    df['sin_day_of_month']=np.sin(2*np.pi*df['day_of_month']/30)
    df['cos_day_of_month']=np.cos(2*np.pi*df['day_of_month']/30)
        
    df['quarter'] = (df['date'].dt.quarter.values).astype(np.int64)
    df['sin_quarter']=np.sin(2*np.pi*df['quarter']/4)
    df['cos_quarter']=np.cos(2*np.pi*df['quarter']/4)

    return df


# In[13]:


train_df = encode(train_df)
train_df


# これらの特徴量を追加すると、スコアが上がります。  
# 最後まで見ていただき、ありがとうございました。

# In[ ]:




