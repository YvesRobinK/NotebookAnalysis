#!/usr/bin/env python
# coding: utf-8

# **[Attention]</br>
# this notebook is simple-baseline for Playground Series3 Episode14.</br>
# you can refer and copy this notebook freely, but this will need a lot of improvement(e.g., feature-engineering, HyperParameter tuning, etc.).</br>
# if you referred or copied this, please vote for this notebook.</br>
# Have fun!</br>**
# 
# 【注意】
# このノートブックはシンプルなベースラインです。</br>
# 参照や複製は自由ですが、多くの改善を必要とするでしょう（特徴量エンジニアリングやハイパーパラメータチューニングなど）。</br>
# もし参照や複製をされた場合は、このノートブックにvoteをお願いします。</br>
# 楽しんでいきましょう！

# In[1]:


# import libraries
# ライブラリのインポート
import gc
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

import warnings
warnings.simplefilter("ignore")

print("imported.")


# In[2]:


# read train-data
# 訓練データの読込
df_train = pd.read_csv("/kaggle/input/playground-series-s3e14/train.csv")
print(df_train.shape)
df_train.head()


# In[3]:


# check information of train-data
# 訓練データの基本情報の確認
df_train.info()


# **[NOTE]**
# - Target is "yield"
# - All features are not null / almost float-type features (id is only int-type feature)
# - number of records are 15289, it is light-weight
# - Some features are likely to be correlated with each other
# 
# - 目的変数は"yield"（これを予測する）
# - 全特徴量で欠損値なし、"id"のみintで他は全てfloat
# - レコード数は15289で、データとしては軽量
# - いくつかの特徴量同士は相関関係がありそう

# In[4]:


# statistics of train-data
# 訓練データの統計量の確認
df_train.describe()


# **[NOTE]</br>
# it is better to check distributions of each features except for "id".</br>
# "id"以外の特徴量の分布を確認した方がよさそう**

# In[5]:


# simple histgram at train-data
# ヒストグラム表示
import math
bins = int(math.log2(len(df_train)) + 1)
fig, axs = plt.subplots(3, 6, figsize=(18, 10))
cnt = 0
for row in range(3):
    for col in range(6):
        axs[row, col].hist(df_train.iloc[:, cnt], bins=bins)
        axs[row, col].set_title(df_train.columns[cnt])
        cnt += 1

plt.show()


# In[6]:


# simple histgram of "yield" at train-data (this feature is target)
# "yield"のヒストグラム表示
#import math
#bins = int(math.log2(len(df_train)) + 1)
#print(bins)
#df_train["yield"].hist(bins=bins)


# **[NOTE]</br>
# Close to Normal Distribution (mean=6025.1940, std=1337.0568).</br>
# "yiled"は正規分布に近い形**

# In[7]:


# correlation between features in train-data
# 特徴量間の相関関係の図示
plt.figure(figsize=(14,12))
colormap = plt.cm.RdBu
sns.heatmap(df_train.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor="white", annot=True)


# **[NOTE]</br>
# We need deeper feature engineering.
# Here, we aim to complete the baseline first.</br>
# 特徴的な傾向があるので、もっと特徴量エンジニアリングが必要</br>
# ここではシンプルなベースラインの構築を目指す**

# In[8]:


# set train-dataset (no categorical features)
# 訓練用データセットの整形
x_train = df_train.drop(columns=["id", "yield"])
y_train = df_train[["yield"]]
id_train = df_train[["id"]]
print(x_train.shape, y_train.shape, id_train.shape)
x_train


# In[9]:


##### additional
# read original-data
df_original = pd.read_csv("/kaggle/input/wild-blueberry-yield-prediction-dataset/WildBlueberryPollinationSimulationData.csv")
df_original


# In[10]:


##### additional
# prepare train-dataset
train = pd.concat([df_train, df_original]).reset_index(drop=True)
x_train = train.drop(columns=["id", "Row#", "yield"])
y_train = train[["yield"]]
id_train = train[["id"]]
print(x_train.shape, y_train.shape, id_train.shape)
x_train


# In[11]:


##### additional
drop_list = [
    "MinOfUpperTRange",
    "AverageOfUpperTRange",
    "MaxOfLowerTRange",
    "MinOfLowerTRange",
    "AverageOfLowerTRange",
    "AverageRainingDays",
]

x_train["set_seeds"] = x_train["fruitset"] * x_train["seeds"]
x_train = x_train.drop(drop_list, axis=1)
x_train


# In[12]:


##### additional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

num_cols = [_ for _ in x_train.columns]
mms = MinMaxScaler()
x_train[num_cols] = mms.fit_transform(x_train[num_cols])
x_train


# In[13]:


# read test-data
# テストデータの読込
df_test = pd.read_csv("/kaggle/input/playground-series-s3e14/test.csv")

# set test-dataset
# テスト用データセットの整形
x_test = df_test.drop("id", axis=1)
id_test = df_test[["id"]]

##### additional
x_test["set_seeds"] = x_test["fruitset"] * x_test["seeds"]
x_test = x_test.drop(drop_list, axis=1)
x_test[num_cols] = mms.transform(x_test[num_cols])

print(x_test.shape, id_test.shape)
x_test


# In[14]:


# fitting by LightGBM and Prediction with K-Fold cross-validation
# LightGBMによる訓練と予測

# parameter
params = {
    "boosting_type": "gbdt",
    "objective": "regression_l1",
    "metric": "mean_absolute_error",
    "learning_rate": 0.045,
    "num_leaves": 48,
    #"subsample": 0.7,
    #"feature_fraction": 0.8,
    #"min_data_in_leaf": 50,
    #"min_sum_hessian_in_leaf": 50,
    "n_estimators": 10000,
    "random_state": 45,
    "importance_type": "gain",
}

n_splits = 5
cv = list(KFold(n_splits=n_splits, shuffle=True, random_state=45).split(x_train, y_train))
metrics = []
imp = pd.DataFrame()

for nfold in np.arange(n_splits):
    print("-"*30, "fold:", nfold, "-"*30)
    idx_tr, idx_va = cv[nfold][0], cv[nfold][1]
    x_tr, y_tr = x_train.loc[idx_tr, :], y_train.loc[idx_tr, :]
    x_va, y_va = x_train.loc[idx_va, :], y_train.loc[idx_va, :]
    print("x/y train-data shapes:", x_tr.shape, y_tr.shape)
    print("x/y valid-data shapes:", x_va.shape, y_va.shape)
    
    # fitting
    model = lgb.LGBMRegressor(**params)
    model.fit(x_tr, y_tr,
            eval_set=[(x_tr, y_tr), (x_va, y_va)],
            early_stopping_rounds=200,
            verbose=100,
            )
    
    # prediction
    y_tr_pred = model.predict(x_tr)
    y_va_pred = model.predict(x_va)
    
    # set metrics(MAE)
    metric_tr = mean_absolute_error(y_tr, y_tr_pred)
    metric_va = mean_absolute_error(y_va, y_va_pred)
    metrics.append([nfold, metric_tr, metric_va])
    
    # importance of features
    _imp = pd.DataFrame({"features": x_train.columns, "importance": model.feature_importances_, "nfold": nfold})
    imp = pd.concat([imp, _imp], axis=0, ignore_index=True)

print("-"*30, "result (Mean Absolute Error)", "-"*30)
metrics = np.array(metrics)
print(metrics)

imp = imp.groupby("features")["importance"].agg(["mean", "std"])
imp.columns = ["importance", "importance_std"]
imp["importance_cov"] = imp["importance_std"] / imp["importance"]
imp = imp.reset_index(drop=False)

print("train-mean-MAE:", "{:.3f}".format(np.mean(metrics[:, 1])), "train-std-MAE:", "{:.3f}".format(np.std(metrics[:, 1])), "valid-mean-MAE:", "{:.3f}".format(np.mean(metrics[:, 2])), "valid-std-MAE:", "{:.3f}".format(np.std(metrics[:, 2])))
print("MAE:", "{:.3f}".format(np.mean(metrics[:, 2]) - np.std(metrics[:, 2])), "-", "{:.3f}".format(np.mean(metrics[:, 2]) + np.std(metrics[:, 2])))


# In[15]:


# prediction with test-data
# テスト用データでの予測
y_test_pred = model.predict(x_test)
sns.histplot(y_test_pred)


# In[16]:


# submission
# 提出用データの整形・CSV出力
df_submit = pd.DataFrame({"id": id_test["id"], "yield": y_test_pred})
df_submit.to_csv("submission.csv", index=None)
print("completed.")
df_submit

