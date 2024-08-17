#!/usr/bin/env python
# coding: utf-8

# # ライブラリのインポート

# In[ ]:


import time
import warnings 
warnings.filterwarnings('ignore')

#Analysis 
import pandas as pd
pd.options.display.float_format = '{:.3f}'.format
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

#Visulization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

# data manipulation
import json
from pandas.io.json import json_normalize

# model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # メモリを効率的に使用するためのメソッド

# In[ ]:


# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
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

    #end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# # 1. EDA

# In[ ]:


# データの読み込み
df_train = reduce_mem_usage(pd.read_csv('../input/train_V2.csv'))
df_test = reduce_mem_usage(pd.read_csv('../input/test_V2.csv'))


# In[ ]:


# データの形状確認
print('train : {}'.format(df_train.shape))
print('test : {}'.format(df_test.shape))


# In[ ]:


df_train.head()


# In[ ]:


# 基本統計量の表示
df_train.describe()


# ## 1.1. 欠損値の確認
# 
# 訓練データの欠損値を確認してみる

# In[ ]:


total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head()


# winPlacePercに1つだけ欠損値があることが分かる
# 
# テストデータに関しても表示してみる

# In[ ]:


total = df_test.isnull().sum().sort_values(ascending=False)
percent = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head()


# テストデータには欠損値はなさそう
# 
# 訓練データの欠損値は１件だったため、今回は対象データは削除する

# In[ ]:


df_train.dropna(axis=0, inplace=True)
df_train.isnull().sum()


# ## 1.2. 目的変数（winPlacePerc）に対する相関の確認¶

# In[ ]:


# ヒートマップを表示してみる
k = 10 # 表示する特徴量の数
corrmat = df_train.corr()
cols = corrmat.nlargest(k, 'winPlacePerc').index # リストの最大値から順にk個の要素の添字(index)を取得
# df_train[cols].head()
cm = np.corrcoef(df_train[cols].values.T) # 相関関数行列を求める ※転置が必要
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(16, 12))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# 上記より、相関が最も強いものは**徒歩での移動距離**ということが分かる
# 
# たくさん歩いているプレイヤーほど、高いランクになっている傾向がありそう

# ## 1.3. 相関の強いデータについて確認

# In[ ]:


# 徒歩での移動距離と目的変数
df_train.plot(x='winPlacePerc', y='walkDistance', kind='scatter', figsize=(8,6))


# In[ ]:


# boostアイテムと目的変数
f, ax = plt.subplots(figsize=(14,6))
fig = sns.boxplot(x='boosts', y='winPlacePerc', data=df_train)
fig.axis(ymin=0, ymax=1)


# boostアイテムを8個以上使用しているプレイヤーは高い順位にいることが分かる。
# 
# ここで注目したいのが、boostアイテムを24個もとっているプレイヤーがいることだ。
# このプレイヤーがどのくらいいるのか確認してみる。

# In[ ]:


df_train[df_train['boosts'] >= 24].head()


# データを見てみると、一人のプレイヤーのwinPlacePercの影響で、箱ひげ図が上記のようになっているようだ。（Id=d16b329d5ea64f）
# 
# ここで注目してほしいのが、Id=5b7d6f8755297bのデータである。  
# killsの値が0なので、誰も倒していないのにも関わらず、winPlacePercが0.915039となっている。怪しい。
# 
# これはチートを使っているか、もしくはデータに欠陥がありそうだ。  
# この辺りのは後半の特徴量エンジニアリングで扱ってみるため、今は放置。

# In[ ]:


# 取得した武器の数と目的変数
df_train.plot(x='winPlacePerc',y='weaponsAcquired', kind='scatter', figsize = (8,6))


# 拾った武器の数に関しては、順位の低いプレイヤーもある程度高いことが分かる。  
# たくさん拾うが、全然勝てないプレイヤーが一定数いるということだ。  
# なんとも悲しい。  

# In[ ]:


# 与えたダメージ量と目的変数
df_train.plot(x='winPlacePerc', y='damageDealt', kind='scatter', figsize=(8,6))


# これは納得の結果だ。  

# In[ ]:


# 回復アイテムと目的変数
df_train.plot(x='winPlacePerc',y='heals', kind='scatter', figsize = (8,6))


# こちらも、回復アイテムを多く使っているプレイヤーの方が順位が高い傾向にありそう。

# In[ ]:


# 最長射殺距離と目的変数
df_train.plot(x='winPlacePerc',y='longestKill', kind='scatter', figsize = (8,6))


# こちらもある程度相関が見られる。

# In[ ]:


# 倒した数と目的変数
df_train.plot(x='winPlacePerc',y='kills', kind='scatter', figsize = (8,6))


# こちらも相関は見られるものの、たくさん倒したからといって順位があがるということでもなさそう。

# # 2. 特徴量エンジニアリング
# 
# 相関関係が強いと思われる特徴量を追加していく。

# ## 2.1. ヘッドショットの確率
# この手のゲームは頭を攻撃できれば基本的に1発K.Oできるが、  
# 頭という部位は面積が少ないため、高度なプレイヤースキルが必要とされることは間違いない。  
# というわけで、ヘッドショットの確率と目的変数との相関を確認してみる。  

# In[ ]:


headshot = df_train[['kills', 'winPlacePerc', 'headshotKills']]
headshot['headshotrate'] = headshot['headshotKills'] / headshot['kills']
headshot.corr() # 相関を確認


# 特徴量として追加

# In[ ]:


df_train['headshotrate'] = df_train['headshotKills']/df_train['kills']
df_test['headshotrate'] = df_test['headshotKills']/df_test['kills']
del headshot # メモリ解放


# ## 2.2. 連続Killの確率
# こちらも上記同様の理由で、特徴量として追加する価値があるか確認してみる。

# In[ ]:


killStreak = df_train[['kills','winPlacePerc','killStreaks']]
killStreak['killStreakrate'] = killStreak['killStreaks']/killStreak['kills']
killStreak.corr() # 相関を確認


# 負の相関ではあるが、killsteaksよりも若干強い相関が見られる。  
# こちらも特徴量として追加する。

# In[ ]:


df_train['killStreakrate'] = -(df_train['killStreaks'] / df_train['kills'])
df_test['killStreakrate'] = -(df_test['killStreaks'] / df_test['kills'])
del killStreak # メモリ解放


# ## 2.3. チートを使用していると思われるプレイヤーを探す

# ゲームにチートはつきものだ。  
# しかしデータの正確性が一番大切な機械学習に、チーターの存在とうものは唐揚げについているパセリより不要なものである。  
# （全国のパセリ好きの皆さま、パセリ農家の皆さま、ごめんなさい）
# 
# ここでは、チートを行なっているであろうプレイヤーにはハッカーポイントという特徴量を与えていくことにする。

# In[ ]:


# ハッカーポイントの特徴量を作成
df_train['hacker_pt'] = 0
df_test['hacker_pt'] = 0


# In[ ]:


# pandasの最大表示列数を広げておく（ここでは50列を指定）
pd.set_option('display.max_columns', 50)


# ### 2.3.1. チーターを探すための前準備
# 
# 総移動距離（ウォーク、スイム、ドライブ）を算出し、移動距離が短いにもかかわらず、上位ランクを取得しているプレイヤーを表示してみる

# In[ ]:


df_train['total_Distance'] = df_train['rideDistance'] +df_train['walkDistance'] + df_train['swimDistance']
df_test['total_Distance'] = df_test['rideDistance'] + df_test['walkDistance'] + df_test['swimDistance']
# 総移動距離が100m未満のプレイヤーを表示
df_train[(df_train['winPlacePerc'] == 1) & (df_train['total_Distance'] < 100)].head()


# 移動距離が0のプレイヤーもいることが分かる。
# 
# ここで、先ほど作成した特徴量に欠損値が入っていることがわかったので、0で埋めておく。 （headshotrateとkillStreakrate）

# In[ ]:


df_train['headshotrate'] = df_train['headshotrate'].fillna(0)
df_train['killStreakrate'] = df_train['killStreakrate'].fillna(0)
df_test['headshotrate'] = df_test['headshotrate'].fillna(0)
df_test['killStreakrate'] = df_test['killStreakrate'].fillna(0)
# df_train.isnull().sum()


# ### 2.3.2. チーター検索その1（回復&ブーストアイテムの使用が極端に少ない）

# In[ ]:


# 上位プレイヤーかつ移動距離が少ないプレイヤーの基本統計量を表示
df_train[(df_train['winPlacePerc'] == 1) & (df_train['total_Distance'] < 100)].describe()


# 基本統計量からも分かるように、ほどんど移動していないプレイヤーはブーストも回復アイテムもほとんど使用していないことが分かる。   
# また、好戦的にバトルしていれば上級者でも被弾は避けられないであろう。  
# よって、回復アイテムを少なくとも1回は使用するはずだ。  
# したがって、下記条件を全て満たすプレイヤーはチーターの可能性があるとして、ハッカーポイントを付与する。  
# 
# - 回復アイテムとboostアイテムの使用が0
# - 総移動距離が100未満
# - 倒した敵の数が20より多い

# In[ ]:


df_train['hacker_pt'][(df_train['heals'] + df_train['boosts'] < 1) & (df_train['total_Distance'] < 100) & (df_train['kills'] > 20)] = 1
df_test['hacker_pt'][(df_test['heals'] + df_test['boosts'] < 1) & (df_test['total_Distance'] < 100) & (df_test['kills'] > 20)] = 1


# ### 2.3.3. チーター検索その2（移動していないのに武器を拾えている）
# 移動していないのに武器を一定数取得しており、かつ敵も倒せているプレイヤーの基本統計量を表示してみる

# In[ ]:


df_train[(df_train['kills'] > 10) &(df_train['weaponsAcquired'] >= 10) & (df_train['total_Distance'] == 0)].describe()


# 203プレイヤーもいる・・・・。  
# 全プレイヤーがチーターという訳ではないと思われるが、下記条件を全て満たすプレイヤーにもハッカーポイントを付与する。
# 
# - 総移動距離が0
# - 拾った武器の数が10以上
# - 倒した敵の数が10より多い

# In[ ]:


df_train['hacker_pt'][(df_train['kills'] > 10) &(df_train['weaponsAcquired'] >= 10) & (df_train['total_Distance'] == 0)] += 1
df_test['hacker_pt'][(df_test['kills'] > 10) &(df_test['weaponsAcquired'] >= 10) & (df_test['total_Distance'] == 0)] += 1


# ## 2.3.4. チーター検索その3（1000m以上離れている敵をkillしている）
# PUBGは1000メートル以上離れている敵を殺すことは不可能のようだ。  
# まさかそんなプレイヤーいないと思うが、確認してみよう。  
# （そういえば、最長射殺距離と目的変数のヒストグラムを表示したとき、ちらほらいたような気がする・・・）  

# In[ ]:


df_train[df_train['longestKill'] >= 1000].describe()


# 24人もいるではないか・・・・  
# ゲームの不具合の可能性もあるが、一応このプレイヤー達にもハッカーポイントを付与しておく。  
# 
# - 最長射殺距離が1000以上

# In[ ]:


df_train['hacker_pt'][df_train['longestKill'] >= 1000] += 1
df_test['hacker_pt'][df_train['longestKill'] >= 1000] += 1


# ハッカーポイントの付与はこの辺で終了しておく

# ## 2.4. kill数とアシスト数
# 
# kill数とアシスト数を合わせた特徴量はどうだろう

# In[ ]:


kills = df_train[['assists','winPlacePerc','kills']]
kills['kills_assists'] = (kills['kills'] + kills['assists'])
kills.corr()


# 上記より、assistsやkillsより相関係数が高い特徴量が作成できたため、新しく追加することにする。

# In[ ]:


df_train['kills_assists'] = df_train['kills'] + df_train['assists']
df_test['kills_assists'] = df_test['kills'] + df_test['assists']
del kills


# ## メモリ管理
# 
# この辺でメモリの開放を行う

# In[ ]:


import gc
df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)
gc.collect()


# In[ ]:


# メモリ食ってる変数を確認してみる
import sys
print("{}{: >25}{}{: >10}{}".format('|','Variable Name','|','Memory','|'))
print(" ------------------------------------ ")
for var_name in dir():
    if not var_name.startswith("_") and sys.getsizeof(eval(var_name)) > 1000: #ここだけアレンジ
        print("{}{: >25}{}{: >10}{}".format('|',var_name,'|',sys.getsizeof(eval(var_name)),'|'))


# In[ ]:


# 不要な変数は削除する
del missing_data
del percent
del total
gc.collect()


# ## 2.5. その他の特徴量の追加
# チーム戦では、同じグループでは同じ値をとる特徴量がある。  
# したがって、それぞれのグループの特徴量を計算して追加することで、精度が上がる可能性がある。

# In[ ]:


# マッチID, グループIDごとのサイズを算出
df_train_size = df_train.groupby(['matchId','groupId']).size().reset_index(name='group_size')
df_test_size = df_test.groupby(['matchId','groupId']).size().reset_index(name='group_size')


# In[ ]:


# マッチID, グループIDごとの平均値を算出
df_train_mean = df_train.groupby(['matchId','groupId']).mean().reset_index()
df_test_mean = df_test.groupby(['matchId','groupId']).mean().reset_index()


# また、上級プレイヤーだとしても、同じ試合（ゲーム）にさらに上級プレイヤーがいた場合、相対的にスコアが下がってしまうと思われる。  
# それを加味し、試合（マッチID）毎の特徴量も追加する。

# In[ ]:


# マッチIDごとの平均値を算出
df_train_match_mean = df_train.groupby(['matchId']).mean().reset_index()
df_test_match_mean = df_test.groupby(['matchId']).mean().reset_index()


# In[ ]:


# データのマージ
df_train = pd.merge(df_train, df_train_size, how='left', on=['matchId', 'groupId'])
df_test = pd.merge(df_test, df_test_size, how='left', on=['matchId', 'groupId'])
del df_train_size
del df_test_size

df_train = pd.merge(df_train, df_train_mean, suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
df_test = pd.merge(df_test, df_test_mean, suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
del df_train_mean
del df_test_mean

df_train = pd.merge(df_train, df_train_match_mean, suffixes=["", "_match_mean"], how='left', on=['matchId'])
df_test = pd.merge(df_test, df_test_match_mean, suffixes=["", "_match_mean"], how='left', on=['matchId'])
del df_train_match_mean
del df_test_match_mean


# In[ ]:


# メモリ管理
df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)
gc.collect()


# # 3. LightGBM

# ## 3.1. 前処理

# In[ ]:


# 学習に使用するカラムの選択
train_columns = list(df_test.columns)

train_idx = df_train.Id
test_idx = df_test.Id

# 学習に使用しないカラムの削除
train_columns.remove("Id")
train_columns.remove("matchId")
train_columns.remove("groupId")


# In[ ]:


# 説明変数と目的変数に分割
x_train = df_train[train_columns]
x_test = df_test[train_columns]
y_train = df_train["winPlacePerc"].astype('float')


# In[ ]:


# 確認
x_train.head()


# `matchType`がまだカテゴリデータのままなので、これを処理する

# In[ ]:


encoded_train = pd.get_dummies(x_train.matchType, prefix=x_train.matchType.name ,prefix_sep="_")
encoded_test = pd.get_dummies(x_test.matchType, prefix=x_test.matchType.name ,prefix_sep="_")
encoded_train.head()


# 上記で生成したencodedをマージし、元々あったmatchTypeを削除する。

# In[ ]:


# マージ
x_train = x_train.merge(encoded_train, right_index=True, left_index=True)
x_test = x_test.merge(encoded_test, right_index=True, left_index=True)


# In[ ]:


# 確認
x_train.head()


# In[ ]:


# 削除
del x_train['matchType']
del x_test['matchType']


# In[ ]:


# メモリ管理
del df_train
del df_test
gc.collect()


# ## 3.2. 学習

# In[ ]:


# LightGBM
folds = KFold(n_splits=3,random_state=6)
oof_preds = np.zeros(x_train.shape[0])
sub_preds = np.zeros(x_test.shape[0])

start = time.time()
valid_score = 0
importances = pd.DataFrame()

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(x_train, y_train)):
    trn_x, trn_y = x_train.iloc[trn_idx], y_train[trn_idx]
    val_x, val_y = x_train.iloc[val_idx], y_train[val_idx]    
    
    train_data = lgb.Dataset(data=trn_x, label=trn_y)
    valid_data = lgb.Dataset(data=val_x, label=val_y)
    
    params = {"objective" : "regression", 
              "metric" : "mae", 
              'n_estimators':10000, 
              'early_stopping_rounds':100,
              "num_leaves" : 30, 
              "learning_rate" : 0.3, 
              "bagging_fraction" : 0.9,
              "bagging_seed" : 0}
    
    lgb_model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], verbose_eval=1000) 

    #imp_df = pd.DataFrame()
    #imp_df['feature'] = train_columns
    #imp_df['gain'] = lgb_model.booster_.feature_importance(importance_type='gain')
    
    #imp_df['fold'] = fold_ + 1
    #importances = pd.concat([importances, imp_df], axis=0, sort=False)    
    
    oof_preds[val_idx] = lgb_model.predict(val_x, num_iteration=lgb_model.best_iteration)
    oof_preds[oof_preds>1] = 1
    oof_preds[oof_preds<0] = 0
    sub_pred = lgb_model.predict(x_test, num_iteration=lgb_model.best_iteration) / folds.n_splits
    sub_pred[sub_pred>1] = 1 # should be greater or equal to 1
    sub_pred[sub_pred<0] = 0 
    sub_preds += sub_pred
    print('Fold %2d RMSE : %.6f' % (n_fold + 1, mean_absolute_error(val_y, oof_preds[val_idx])))
    valid_score += mean_absolute_error(val_y, oof_preds[val_idx])


# In[ ]:


print('Done')


# In[ ]:


test_pred = pd.DataFrame({"Id":test_idx})
test_pred["winPlacePerc"] = sub_preds
test_pred.columns = ["Id", "winPlacePerc"]
test_pred.to_csv("lgb_model_181204.csv", index=False) # submission


# In[ ]:





# In[ ]:





# In[ ]:




