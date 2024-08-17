#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# 学習データの読み込み ###############################
train_df = pd.read_csv('/kaggle/input/prediction-of-tourist-arrivals/train_df.csv')
train_df['date'] = pd.to_datetime(train_df['date'])   # 日付型をdatetime型へ

# 評価データの読み込み ###############################
test_df = pd.read_csv('/kaggle/input/prediction-of-tourist-arrivals/test_df.csv')
test_df['date'] = pd.to_datetime(test_df['date'])   # 日付型をdatetime型へ

# Submissionファイルの読み込み ###############################
submission = pd.read_csv('/kaggle/input/prediction-of-tourist-arrivals/submission.csv')
submission_cv = submission.drop('tourist_arrivals', axis=1)   # 交差検証で予測したデータを書き込む用


# In[4]:


# dateから、year, month, day、dayofweekの情報を生成 ###############################
# 学習データ
train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day
train_df['dayofweek'] = train_df['date'].dt.dayofweek  # 日曜日が0で土曜日が6
# 評価データ
test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df['dayofweek'] = test_df['date'].dt.dayofweek


# **新たな特徴量の作成**
# 
# 追加した特徴量は以下の通りです。2の三角関数特徴量については、[Kei.Shibata](https://www.kaggle.com/code/keishibata/tourist-arrivals-feature-engineering)さんのNotebookを参考にしています。
# 4と5の目的変数に関する特徴量については、交差検証を行うときに学習データに対して算出しその値を検証データと評価データにマッピングするため、ここでは作成していません。
# 
# 
# 1. 季節
# 2. 三角関数特徴量
# 3. dayofweekごとのtourism_indexの平均値
# 4. spot_facilityごとの目的変数(tourist_arrivals)の平均値
# 5. categoryごとの目的変数(tourist_arrivals)の平均値
# 

# In[5]:


# 新たな特徴量の作成 ###############################

# monthから季節を抽出
def get_season(month):
    if month in (1, 2, 3):
        return 'spring'
    elif month in (4, 5, 6):
        return 'summer'
    elif month in (7, 8, 9):
        return 'fall'
    else:
        return 'winter'
train_df['season'] = train_df.apply(lambda x: get_season(x['month']), axis=1)
test_df['season'] = test_df.apply(lambda x: get_season(x['month']), axis=1)

# 三角関数特徴量
def encode(df, col):
    df[col + '_cos'] = np.cos(2 * np.pi * df[col] / df[col].max())
    df[col + '_sin'] = np.sin(2 * np.pi * df[col] / df[col].max())
    return df
train_df = encode(train_df, 'month')
train_df = encode(train_df, 'day')
train_df = encode(train_df, 'dayofweek')
test_df = encode(test_df, 'month')
test_df = encode(test_df, 'day')
test_df = encode(test_df, 'dayofweek')

# dayofweekごとのtourism_indexの平均値
train_weekday_index_mean = train_df.groupby('dayofweek')['tourism_index'].mean()
train_df['dayofweek_index_mean'] = train_df['dayofweek'].map(train_weekday_index_mean)
test_weekday_index_mean = test_df.groupby('dayofweek')['tourism_index'].mean()
test_df['dayofweek_index_mean'] = test_df['dayofweek'].map(test_weekday_index_mean)

train_df.columns
test_df.columns


# 次に、カテゴリ変数に対してGBDTモデルに入力できるcategory型に変換します。[KentaK0928](https://www.kaggle.com/code/kentak0928/host-baseline-lightgbm-prediction-tourist-arrival)さんのNotebookを参考にしています。

# In[6]:


# カテゴリ変数をcategory型に変換 ###############################
category_list = train_df.columns[train_df.dtypes=='object'].values.tolist()
for cat in category_list:
    # 欠損値を補完
    # 学習データ
    train_df[cat] = train_df[cat].fillna('unknown')
    # 評価データ
    test_df[cat] = test_df[cat].fillna('unknown')
    
    # Category型に変換
    # 学習用データ
    train_df[cat] = train_df[cat].astype('category')
    # 評価用データ
    test_df[cat] = test_df[cat].astype('category')


# **交差検証**
# 
# 評価データと検証データの条件を同じにするために各月ごとを検証データとした交差検証を行います。
# 
# また、学習データと検証データを分割した後に目的変数に関する特徴量を追加します。

# In[7]:


# 目的変数と説明変数の設定 ###############################
# 学習データの目的変数を設定
y_train = train_df[['tourist_arrivals']]
# 学習データの説明変数の設計（不必要な情報はdrop）
x_train = train_df.drop(['id', 'date'], axis=1)
# 学習済みモデルに投入する評価データの説明変数を設計（不必要な情報はdrop）
x_test = test_df.drop(['id', 'date'], axis=1)
# カテゴリ変数のリスト
category_list = x_train.columns[x_train.dtypes=='category'].values.tolist()


# ーーーー 交差検証(CV) ーーーー
# 学習データ：2018年8月1日 〜 2019年5月31日、検証データ：それぞれの月（評価データと同じ条件）
# 評価データ : 2019年7月1日 〜 2019年7月31日

# 学習用と検証用のRMSEを追加するリスト
train_list = []
val_list = []

month_index = x_train['month'].unique()   # 各月の数字を取得
for i, m in enumerate(month_index, 1):
    # 検証データの設定
    val_index = x_train[x_train["month"] == m].index   # インデックスを取得
    val_x_df = x_train.iloc[val_index]
    val_y_df = y_train.iloc[val_index]
    # 学習データの設定
    train_x_df = x_train.drop(val_index)   # 検証データのインデックスを落とす
    train_y_df = y_train.drop(val_index)

    # 目的変数に関する特徴量の追加(リークを避けるためにここで追加) 
    # spot_facilityごとの目的変数の平均値
    train_obj = train_x_df.groupby("spot_facility")['tourist_arrivals'].mean()   # 学習データの目的変数に関して求める
    train_x_df['arrivals_spot_mean'] = train_x_df["spot_facility"].map(train_obj)
    val_x_df['arrivals_spot_mean'] = val_x_df["spot_facility"].map(train_obj)
    x_test['arrivals_spot_mean'] = x_test["spot_facility"].map(train_obj)
    # categoryごとの目的変数の平均値
    train_obj = train_x_df.groupby("category")['tourist_arrivals'].mean()   # 学習データの目的変数に関して求める
    train_x_df['arrivals_category_mean'] = train_x_df["category"].map(train_obj)
    val_x_df['arrivals_category_mean'] = val_x_df["category"].map(train_obj)
    x_test['arrivals_category_mean'] = x_test["category"].map(train_obj)

    # 目的変数を落とす
    train_x_df = train_x_df.drop(['tourist_arrivals'], axis=1)
    val_x_df = val_x_df.drop(['tourist_arrivals'], axis=1)
    
    # パラメータ
    lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',

    'learning_rate': 0.05,
    'n_estimators': 100000,
    'importance_type': 'gain',

    'num_leaves': 32,

    'min_data_in_leaf': 20,
    'min_sum_hessian_in_leaf': 20,
    'lambda_l1': 0.0,
    'lambda_l2': 0.0,

    'bagging_fraction': 0.9,
    'bagging_freq': 1,
    'feature_fraction': 0.9,

    'random_seed': 42  # 乱数設定
    }
    
    # LightGBMREgressor()インスタンスの生成
    model = lgb.LGBMRegressor(**lgbm_params)
    
    # モデルのfitting
    model.fit(train_x_df, train_y_df,
              eval_set = [(train_x_df, train_y_df), (val_x_df, val_y_df)],
              early_stopping_rounds = 50,
              categorical_feature = category_list,
              verbose = -1
              )
    
    # 学習用データによるpredict
    y_pred_train_hold = model.predict(train_x_df, num_iteration=model.best_iteration_)
    # 検証用データによるpredict
    y_pred_valid_hold = model.predict(val_x_df, num_iteration=model.best_iteration_)
    # RMSEを算出
    # 学習用データ
    temp_rmse_train = np.sqrt(mean_squared_error(train_y_df, y_pred_train_hold))
    # 検証用データ
    temp_rmse_valid = np.sqrt(mean_squared_error(val_y_df, y_pred_valid_hold))
    # RMSEの表示
    print(f'RMSE(train_data) = {temp_rmse_train:.4f}')
    print(f'RMSE(valid_data) = {temp_rmse_valid:.4f}\n')
    # 学習用データと検証用データのリスト
    train_list.append(temp_rmse_train)
    val_list.append(temp_rmse_valid)
    
    # 評価データで、実際に予測
    # predict
    preds_test = model.predict(x_test, num_iteration=model.best_iteration_)
    # preds_testリストをNumPy配列に変換
    preds_test_np = np.array(preds_test)

    # 予測値を出力
    submission_cv['tourist_arrivals' + '_model' + str(i)] = preds_test_np
    
    
# 学習用データと検証用データの予測精度を出力
for i,m in enumerate(month_index):
    print(f'検証データ{m}月: RMSE(train_data) = {train_list[i]:.4f}, RMSE(valid_data) = {val_list[i]:.4f}')
print(f'train_fold_mean: {np.mean(train_list):.4f}, valid_fold_mean: {np.mean(val_list):.4f}')   # 検証用データの平均値
    


# 最後に、交差検証で得られた評価データに対する予測値の平均値を計算します。

# In[8]:


# 交差検証で得た各モデルの評価データに対する予測値の平均値を書き込む ###############################
index = len(submission_cv['id'])
mean = []
for i in range(index):
    temp = 0
    for num in range(1, len(month_index) + 1):
        temp = temp + submission_cv['tourist_arrivals_model' + str(num)][i]
    temp = temp / len(month_index)   # モデルの数で割る
    mean.append(temp)
mean_np = np.array(mean)
submission['tourist_arrivals'] = mean_np

# 予測値の書き込み
submission.to_csv('submission_pre.csv', index=False)
submission

