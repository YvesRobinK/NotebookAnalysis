#!/usr/bin/env python
# coding: utf-8

# # 過去のNotebook
# 
# [日本語][EDA] JP Tokyo Stock データ概要を確認 その1 https://www.kaggle.com/code/tatsuyafujii/eda-jp-tokyo-stock-1 <br>
# [日本語][EDA] JP Tokyo Stock データ概要を確認 その2[移動平均線等] https://www.kaggle.com/code/tatsuyafujii/eda-jp-tokyo-stock-2 <br>
# [日本語][EDA] JP Tokyo Stock データ概要を確認 その3[ボリンジャーバンド等] https://www.kaggle.com/code/tatsuyafujii/eda-jp-tokyo-stock-3/notebook <br>
# [日本語][EDA] JP Tokyo Stock データ概要を確認 その4[サイコロジカルライン等] https://www.kaggle.com/code/tatsuyafujii/eda-jp-tokyo-stock-4/notebook <br>
# 

# 上記のノートブックの特徴を作る関数を定義

# In[1]:


# !pip install japanize_matplotlib

import numpy as np
import pandas as pd
import jpx_tokyo_market_prediction
from lightgbm import LGBMRegressor
import optuna.integration.lightgbm as lgb
import seaborn as sns
# import japanize_matplotlib
import matplotlib.pyplot as plt
import datetime
import jpx_tokyo_market_prediction
import warnings
warnings.filterwarnings("ignore")

def MA(series, window=25):
    return series.rolling(window, min_periods=1).mean()

def DMA(series, window=25):
    return series/MA(series, window) - 1

def divergence(series, window=25):
    std = series.rolling(window,min_periods=1).std()
    mean = series.rolling(window,min_periods=1).mean()
    return (series-mean) / std    

def rsi(series, n=14):
    return (series - series.shift(1)).rolling(n).apply(lambda s:s[s>0].sum()/abs(s).sum())

def stochastic(series, k=14, n=3, m=3):
    _min = series.rolling(k).min()
    _max = series.rolling(k).max()
    _k = (series - _min)/(_max - _min)
    _d1 = _k.rolling(n).mean()
    _d2 = _d1.rolling(m).mean()
    return pd.DataFrame({
                    "%K":_k,
                    "FAST-%D":_d1,
                    "SLOW-%D":_d2,
                    },index=series.index)
    # return _k, _d1, _d2

def psy(series, n=14):
    return (series - series.shift(1)).rolling(n).apply(lambda s:(s>=0).mean())

def ICH(series):
    conv = series.rolling(9).apply(lambda s:(s.max()+s.min())/2)
    base = series.rolling(26).apply(lambda s:(s.max()+s.min())/2)
    pre1 = ((conv + base)/2).shift(25)
    pre2 = d.Close_adj.rolling(52).apply(lambda s:(s.max()+s.min())/2).shift(25)
    lagg = d.Close_adj.shift(25)
    return conv, base, pre1, pre2, lagg

def roc(series, window=14):
    return series/series.shift(window) - 1


# ### データ読込

# In[2]:


prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv", parse_dates=["Date"])
# print(prices.shape)
# prices.head()
df = prices.copy()
df["Close_adj"] = df.groupby("SecuritiesCode").apply(lambda d:d["Close"]/d["AdjustmentFactor"].cumprod().shift().fillna(1)).reset_index("SecuritiesCode",drop=True)
df["翌日終値"] = df.groupby("SecuritiesCode")["Close_adj"].shift(-1)
df["翌々日終値"] = df.groupby("SecuritiesCode")["Close_adj"].shift(-2)
df["R"] = (df["翌々日終値"] - df["翌日終値"]) / df["翌日終値"]
(df["R"]-df["Target"]).describe()


# ## 特徴量追加

# In[3]:


def holiday(d):
    return pd.DataFrame({
        "before_holiday":(d["Date"] != d["Date"].shift(-1) - datetime.timedelta(days=1)) | (d["weekday"]==4),
        "after_holiday":(d["Date"] != d["Date"].shift(1) + datetime.timedelta(days=1)) | (d["weekday"]==0)
    }, index=d.index)
# df["weekday"] = df["Date"].dt.weekday
# df = df.join(df.groupby("SecuritiesCode").apply(holiday))

# ## 休み前のTarget期待値が髙い。＝　休み明け初日終値で買って、次の日売る戦略が悪くない？ 期待値的にはごくごくわずか（0.2％未満だが）
# ## ただし、このコンペでは同じ日に売る銘柄と買う銘柄決めないといけないので銘柄間での特徴量の際がわからないと意味がない
# stat = df.groupby(["before_holiday","after_holiday"])["Target"].describe()
# stat["serr"] = stat["std"]/np.sqrt(stat["count"]-1)
# display(stat)
# stat["mean"]/stat["serr"]


# 【To Do】特徴量作成クラス

# In[4]:


class FeatureBase():
    def create_feature(self, d):
        assert False, "NotImplemented"
        
class MAFeature(FeatureBase):
    def create_feature(self, d):
        return self._create_feature(d["Close_adj"])

    def _create_feature(self, series, window1=5, window2=25):
        ma1 = MA(series, window1).rename("MA1")
        ma2 = MA(series, window2).rename("MA2")
        diff = ma1 - ma2
        cross = pd.Series(
                        np.where((diff>0) & (diff<0).shift().fillna(False), 1,
                            np.where((diff<0) & (diff>0).shift().fillna(False), -1, 0
                                )
                        ),
                        index = series.index, name="MA_Cross"
                )
        return pd.concat([ma1, ma2, cross], axis=1)

# class FeatureFactory():
#     def __init__(self):
#         self.feature_bases = []
    
#     def add(self, fb):
#         assert isinstance(fb, FeatureFactory)
#         self.feature_bases.append(fb)
    
#     def make(self, df):
#         return pd.concat([
#             fb.create_feature(df) for fb in self.feature_bases
#         ],axis=1)


# In[5]:


def make_features(df):
    df = df[[
        "Date","SecuritiesCode","Open","Close","AdjustmentFactor",
        "Volume"
    ]].copy()
    df["weekday"] = df["Date"].dt.weekday
    df = df.join(df.groupby("SecuritiesCode").apply(holiday))
    df["Volume_ratio"] = df["Volume"]/df.groupby("SecuritiesCode")["Volume"].rolling(window=15, min_periods=1).mean().reset_index("SecuritiesCode",drop=True)
    df["Close_adj"] = df.groupby("SecuritiesCode").apply(lambda d:d["Close"]/d["AdjustmentFactor"].cumprod().shift().fillna(1)).reset_index("SecuritiesCode",drop=True)
    df[["MA1", "MA2", "MA_Cross"]] = df.groupby("SecuritiesCode").apply(lambda d: MAFeature()._create_feature(d.Close_adj))# .join(df["Target"].shift(-1)).groupby("MA_Cross").describe()
    df["Diff"] = (df["Close"] - df["Open"]) / df[["Close","Open"]].mean(axis=1)
    df["Diff_MA1"] = df["Close_adj"] - df["MA1"]
    df["Diff_MA2"] = df["Close_adj"] - df["MA2"]
    for i in range(1, 3):
        df["MA_Cross_lag_{:}".format(i)] = df.groupby("SecuritiesCode")["MA_Cross"].shift(i)

    df["DivMA"] = df.groupby("SecuritiesCode")["Close_adj"].apply(DMA)
    df["Div"] = df.groupby("SecuritiesCode")["Close_adj"].apply(divergence)
    df["Rsi"] = df.groupby("SecuritiesCode")["Close_adj"].apply(rsi)
    df = df.join(df.groupby("SecuritiesCode")["Close_adj"].apply(stochastic))
        
    return df


# In[6]:


get_ipython().run_cell_magic('time', '', 'df = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv", parse_dates=["Date"])\ndf = make_features(df).join(df.Target)\n')


# In[7]:


def train_model(X, y):
    model=LGBMRegressor(# boosting_type="dart",
                        num_leaves=31,max_depth=12,
                        learning_rate=0.1, n_estimators=1000,
                        # random_state=42
    )
    model.fit(X,y)
    # model.score(X,y)
    return model

columns = [
    "Diff", "Close_adj","Volume_ratio",
    "before_holiday", "after_holiday",
    "Diff_MA1", "Diff_MA2",
    "MA_Cross",'MA_Cross_lag_1', 'MA_Cross_lag_2',
    "DivMA", "Div", "Rsi", "%K", "FAST-%D","SLOW-%D",
]


# In[8]:


get_ipython().run_cell_magic('time', '', 'models = {}\nfor code, d in df.groupby("SecuritiesCode"):\n    d = d[~d.Target.isnull()]\n    X = d[columns]\n    y = d.Target\n    model = train_model(X, y)\n    models[code] = model\n    #print(code, model.score(X,y))\n\n\n')


# In[9]:


# import joblib
# joblib.dump(models, "lgbm_model.bin")


# In[10]:


# models = joblib.load("lgbm_model.bin")


# In[11]:


data = df.copy()


# In[12]:


env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()


# In[13]:


for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    prices["Date"] = pd.to_datetime(prices["Date"])
    data = data.append(prices).drop_duplicates(["SecuritiesCode", "Date"], keep="last").sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)
    data = make_features(data)
    
    
    # sample_prediction["Avg"] = sample_prediction["SecuritiesCode"].apply(get_avg)
    sample_prediction["Date"] = pd.to_datetime(sample_prediction["Date"])
    d = sample_prediction[["Date","SecuritiesCode"]].merge(data, on=["Date","SecuritiesCode"])
    for code, _d in d.groupby("SecuritiesCode"):
        d.loc[_d.index, "Pred"] = models[code].predict(_d[columns])
    sample_prediction = d.sort_values(by="Pred", ascending=False)
    sample_prediction["Rank"] = np.arange(0,2000)
    sample_prediction = sample_prediction.sort_values(by = "SecuritiesCode", ascending=True)
    # sample_prediction.drop(["Prediction"],axis=1)
    submission = sample_prediction[["Date","SecuritiesCode","Rank"]]
    env.predict(submission)
    # break
    


# In[ ]:




