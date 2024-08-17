#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from pyclustering.cluster.xmeans import xmeans, kmeans_plusplus_initializer
from pyclustering.utils import draw_clusters


# In[2]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv("../input/test.csv")


# # 1.データ処理

# In[3]:


#データ数、型の確認
train.info()


# In[4]:


# 欠損値NaNの確認
train.isnull().sum()


# In[5]:


test.isnull().sum()


# In[6]:


train.head()


# In[7]:


test.head()


# In[8]:


# Cabinの中身を確認
train["Cabin"].value_counts()


# In[9]:


# Embarkedの中身を確認
train["Embarked"].value_counts()


# In[10]:


# Ticketの中身を確認
train["Ticket"].value_counts()


# ## 欠損値の処理

# In[11]:


# Ageの穴埋め(中央値)
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())
# Cabinは欠損値が多いため削除
train.drop("Cabin", axis=1, inplace=True)
test.drop("Cabin", axis=1, inplace=True)
# Embarkedは最頻値であるSを代入
train["Embarked"] = train["Embarked"].fillna("S")
# Fareの穴埋め(中央値)
test["Fare"] = test["Fare"].fillna(test["Fare"].median())


# ## カテゴリカル変数の処理

# In[12]:


train.head()


# In[13]:


print(f'[train]Nmaeのユニークな要素: {train["Name"].nunique()}')
print(f'[test]Nmaeのユニークな要素: {test["Name"].nunique()}')
print(f'[train]Sexのユニークな要素: {train["Sex"].nunique()}')
print(f'[test]Sexのユニークな要素: {test["Sex"].nunique()}')
print(f'[train]Ticketのユニークな要素: {train["Ticket"].nunique()}')
print(f'[test]Ticketのユニークな要素: {test["Ticket"].nunique()}')
print(f'[train]Embarkedのユニークな要素: {train["Embarked"].nunique()}')
print(f'[test]Embarkedのユニークな要素: {test["Embarked"].nunique()}')


# In[14]:


# Nameは今回扱わない(苗字から家族であることを判別するようなカーネルもある)
train.drop("Name", axis=1, inplace=True)
test.drop("Name", axis=1, inplace=True)

# ticketはユニークな要素が多いため(大変だから)削除
train.drop("Ticket", axis=1, inplace=True)
test.drop("Ticket", axis=1, inplace=True)

# Name以外は全てOne-Hot Encodingで処理する
train = train.join(pd.get_dummies(train["Sex"],prefix="sex"))
test = test.join(pd.get_dummies(test["Sex"],prefix="sex"))

train = train.join(pd.get_dummies(train["Embarked"],prefix="emberk"))
test = test.join(pd.get_dummies(test["Embarked"],prefix="emberk"))

# 使用後のカテゴリカル変数の削除
train.drop(["Sex", "Embarked"], axis=1, inplace=True)
test.drop(["Sex", "Embarked"], axis=1, inplace=True)


# In[15]:


train["tmp"] = "train"
test["tmp"] = "test"


# In[16]:


#　変数を生成するために一時的にtrainとtestを結合
conb = pd.concat([train.drop(["Survived","PassengerId"],axis=1),test.drop("PassengerId",axis=1)])


# In[17]:


# データ処理のためにpandasからnp.arrayに変換
conb_array = np.array(conb[conb.columns[1]].tolist())
for i in range(len(conb.columns)):
    if i <= 1:
        continue
    if conb.columns[i]=="tmp":
        continue
    conb_array = np.vstack((conb_array,conb[conb.columns[i]]))
conb_array = conb_array.T


# In[18]:


# conb_array


# In[19]:


# # クラスタ数の計算
# initial_centers = kmeans_plusplus_initializer(conb_array, 2).initialize()
# instances = xmeans(conb_array, initial_centers, ccore=True)
# instances.process()
# clusters = instances.get_clusters()


# In[20]:


# # 各クラスタの中心の数を求める
# centers = []
# conb_array = np.array(conb_array)
# for cluster in clusters:
#     c = conb_array[cluster, :]
#     centers.append(np.mean(c, axis=0))
    
# cluster_point = len(centers)


# In[21]:


cluster_point = 10


# In[22]:


pred = KMeans(n_clusters=cluster_point).fit_predict(conb_array)


# In[23]:


conb["cluster"] = pred


# In[24]:


train["cluster"] = conb[conb["tmp"]=="train"]["cluster"]
test["cluster"] = conb[conb["tmp"]=="test"]["cluster"]
train.drop("tmp",axis=1,inplace=True)
test.drop("tmp",axis=1,inplace=True)


# In[25]:


train.head()


# # 2.学習部

# In[26]:


# sklearn.ensembleの中からClassifierをインポート
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier


# In[27]:


# 目的変数と説明変数を分解する
X_train = train.drop("Survived", axis=1)
y_train = train["Survived"].values


# ## AdaBoostClassifier

# In[28]:


# AdaBoostClassifierでの学習
ada_clf = AdaBoostClassifier()
ada_clf = ada_clf.fit(X_train, y_train)


# In[29]:


# AdaBoostClassifierでの推論
ada_pred = ada_clf.predict(test)


# ## BaggingClassifier

# In[30]:


bag_clf = BaggingClassifier()
bag_clf = bag_clf.fit(X_train, y_train)


# In[31]:


bag_pred = bag_clf.predict(test)


# ## ExtraTreesClassifier

# In[32]:


et_clf = ExtraTreesClassifier()
et_clf = et_clf.fit(X_train, y_train)


# In[33]:


et_pred = et_clf.predict(test)


# ## GradientBoostingClassifier

# In[34]:


gb_clf = GradientBoostingClassifier()
ba_clf = gb_clf.fit(X_train, y_train)


# In[35]:


gb_pred = gb_clf.predict(test)


# ## RandomForestClassifier

# In[36]:


rf_clf = RandomForestClassifier()
rf_clf = rf_clf.fit(X_train, y_train)


# In[37]:


rf_pred = rf_clf.predict(test)


# ## VotingClassifier

# In[38]:


vote_clf = VotingClassifier(estimators=[('rf', RandomForestClassifier()),
                                        ('gb',GradientBoostingClassifier()),
                                        ('et',ExtraTreesClassifier()),
                                        ('bag',BaggingClassifier()),
                                        ('ada',AdaBoostClassifier())
                                        ])
vote_clf = vote_clf.fit(X_train, y_train)


# In[39]:


vote_pred = vote_clf.predict(test)


# In[40]:


vote_submit = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": vote_pred
    })
vote_submit.to_csv("vote.csv", index=False)


# In[41]:





# In[41]:




