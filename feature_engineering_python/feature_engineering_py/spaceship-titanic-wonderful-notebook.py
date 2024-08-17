#!/usr/bin/env python
# coding: utf-8

# <div style="padding:20px;
#             color:white;
#             margin:10;
#             font-size:200%;
#             text-align:center;
#             display:fill;
#             border-radius:5px;
#             background-color:#294B8E;
#             overflow:hidden;
#             font-weight:700">Spaceship Titanic</div>

# <a id="toc"></a>
# - [1. Introduction](#1)
#     - [1.1 Import Libraries](#1.1)
#     - [1.2 Download Data](#1.2)
#     
# - [2. Visualition](#2)
# - [3. Data Cleaning](#3)
# - [4. Correlation](#4)
# - [5. Data Preprocessing](#5)
# - [6. Feature Engineering](#6)
#     - [6.1 Pipeline](#6.1)
#     - [6.2 train test split](#6.2)
# - [7. Model building](#7)
# - [8. Submitting](#8)
#   

# <a id="1"></a>
# <div style="padding:20px;
#             color:white;
#             margin:10;
#             font-size:170%;
#             text-align:left;
#             display:fill;
#             border-radius:5px;
#             background-color:#294B8E;
#             overflow:hidden;
#             font-weight:700"><span style='color:#CDA63A'>|</span> Introduction</div>

# **Welcome to the year 2912, where your data science skills are needed to solve a cosmic mystery. We've received a transmission from four lightyears away and things aren't looking good.**
# 
# **The Spaceship Titanic was an interstellar passenger liner launched a month ago. With almost 13,000 passengers on board, the vessel set out on its maiden voyage transporting emigrants from our solar system to three newly habitable exoplanets orbiting nearby stars.**
# 
# ![](https://storage.googleapis.com/kaggle-media/competitions/Spaceship%20Titanic/joel-filipe-QwoNAhbmLLo-unsplash.jpg)
# 
# **While rounding Alpha Centauri en route to its first destination—the torrid 55 Cancri E—the unwary Spaceship Titanic collided with a spacetime anomaly hidden within a dust cloud. Sadly, it met a similar fate as its namesake from 1000 years before. Though the ship stayed intact, almost half of the passengers were transported to an alternate dimension!**

# <div style="padding:20px;
#             color:white;
#             margin:10;
#             font-size:170%;
#             text-align:left;
#             display:fill;
#             border-radius:5px;
#             background-color:#294B8E;
#             overflow:hidden;
#             font-weight:700"><span style='color:#CDA63A'>|</span> Dataset Description</div>
# 
# **In this competition your task is to predict whether a passenger was transported to an alternate dimension during the Spaceship Titanic's collision with the spacetime anomaly. To help you make these predictions, you're given a set of personal records recovered from the ship's damaged computer system.**
# 
# # File and Data Field Descriptions
# 
# * **train.csv** - Personal records for about two-thirds (~8700) of the passengers, to be used as training data.
# 
# * **PassengerId** - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
# * **HomePlanet** - The planet the passenger departed from, typically their planet of permanent residence.
# * **CryoSleep** - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
# * **Cabin** - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
# * **Destination** - The planet the passenger will be debarking to.
# * **Age** - The age of the passenger.
# * **VIP** - Whether the passenger has paid for special VIP service during the voyage.
# * **RoomService, FoodCourt, ShoppingMall, Spa, VRDeck** - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
# * **Name** - The first and last names of the passenger.
# * **Transported** - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.
# 
# ***test.csv*** - Personal records for the remaining one-third (~4300) of the passengers, to be used as test data. Your task is to predict the value of Transported for the passengers in this set.
# 
# **sample_submission.csv** - A submission file in the correct format.
# **PassengerId** - Id for each passenger in the test set.
# **Transported** - The target. For each passenger, predict either True or False.

# <a id="1.1"></a>
# ## <b>1.1 <span style='color:#E1B12D'>Import Libraries</span></b> 

# In[1]:


import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import sklearn.metrics as metrics

from sklearn import linear_model
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBClassifier
from sklearn.svm import SVC
# manual nested cross-validation for random forest on a classification dataset
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier, Pool


# <a id="1.2"></a>
# ## <b>1.2 <span style='color:#E1B12A'>Download Data</span></b> 

# In[2]:


train=pd.read_csv("/kaggle/input/spaceship-titanic/train.csv")
test=pd.read_csv("/kaggle/input/spaceship-titanic/test.csv")
sample=pd.read_csv("/kaggle/input/spaceship-titanic/sample_submission.csv")


# In[3]:


train.head()


# In[4]:


test.head()


# 
# <a id="2"></a>
# ## <b>2 <span style='color:#E1B12A'>Visualition</span></b> 

# In[5]:


plt.figure(figsize=(8,4))
sns.barplot(x= 'HomePlanet', y= 'Transported', data= train)
plt.show()


# In[6]:


plt.figure(figsize=(8,4))
sns.barplot(x= 'Destination', y= 'Transported', data= train)
plt.show()


# <a id="3"></a>
# ## <b>3 <span style='color:#E1B12A'>Data Cleaninig</span></b> 

# In[7]:


train['Destination'].unique()


# In[8]:


train.replace({'Europa':7, 'Earth':4, 'Mars':5},inplace=True)
train.replace({'TRAPPIST-1e':4, 'PSO J318.5-22':5, '55 Cancri e':6},inplace=True)
train.replace({False:0,True:1},inplace=True)

test.replace({'Europa':7, 'Earth':4, 'Mars':5},inplace=True)
test.replace({'TRAPPIST-1e':4, 'PSO J318.5-22':5, '55 Cancri e':6},inplace=True)
test.replace({False:0,True:1},inplace=True)


# In[9]:


train.drop(['PassengerId','Name'],axis=1,inplace=True)
test.drop(['PassengerId','Name'],axis=1,inplace=True)


# In[10]:


print(train.shape)
print(test.shape)


# In[11]:


train.Transported.value_counts()


# In[12]:


train.isnull().sum()


# In[13]:


train=train.fillna(train.median())
test=test.fillna(test.median())
train=train.fillna(method='bfill')
test=test.fillna(method='bfill')


# In[14]:


train.groupby('Transported').mean().T


# In[15]:


train.info()


# In[16]:


train.nunique()


# <a id="4"></a>
# ## <b>4 <span style='color:#E1B12A'>Correlation</span></b> 

# In[17]:


train.corrwith(train['Transported']).abs().sort_values(ascending=False)


# In[18]:


train.corr()


# In[19]:


sns.heatmap(train.corr().abs(),cmap='Blues_r')


# <a id="5"></a>
# ## <b>5 <span style='color:#E1B12A'>Data preprocessing</span></b> 

# In[20]:


train['Cab0']=train['Cabin'].str[0]
test['Cab0']=test['Cabin'].str[0]


# In[21]:


train['Cab-1']=train['Cabin'].str[-1]
test['Cab-1']=test['Cabin'].str[-1]


# In[22]:


train.head()


# In[23]:


train['Cab0'].unique()


# In[24]:


X = train.copy()
y = train.Transported
X = X.drop('Transported',axis=1)


# In[25]:


X.info()


# In[26]:


X.columns


# <a id="6"></a>
# ## <b>6 <span style='color:#E1B12A'>Feature Engineering</span></b> 

# <a id="6.1"></a>
# ## <b>6.1 <span style='color:#E1B12A'>Pipeline</span></b> 
# 
# **numerical columns for StandardScaler**
# 
# **catigorical columns for OrdinalEncoder**

# In[27]:


cat_attr=[ 'Cabin','Cab0','Cab-1']
num_attr=['HomePlanet', 'CryoSleep', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 
          'Spa', 'VRDeck', 'Destination','VIP']

full_pip=ColumnTransformer([
    ('num',StandardScaler(),num_attr),
    ('cat',OrdinalEncoder(),cat_attr)
])


# In[28]:


x=full_pip.fit_transform(X)


# In[29]:


x_test=full_pip.fit_transform(test)


# <a id="6.2"></a>
# ## <b>6.2 <span style='color:#E1B12A'>train_test_split</span></b> 

# In[30]:


x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.2,random_state=41)


# <a id="7"></a>
# ## <b>7 <span style='color:#E1B12A'>Model building</span></b>

# In[31]:


javob = []
for N in range(5):
    model = RandomForestClassifier(random_state=N, n_jobs=6, n_estimators=126)
    model.fit(x_train, y_train)
    preds_valid = model.predict(x_val)
    test_preds = model.predict(x_test)
    javob.append(test_preds)
    print(accuracy_score(y_val, preds_valid))


# # LogisticRegression

# In[32]:


LR_model = LogisticRegression()
LR_model.fit(x_train, y_train)


y_pred = LR_model.predict(x_val)
print(metrics.classification_report(y_val, y_pred))
print("Model aniqligi:", metrics.accuracy_score(y_val,y_pred))

## confusion matrix
conf_mat = metrics.confusion_matrix(y_val, y_pred)
sns.heatmap(conf_mat, annot=True,fmt="g")
plt.show()

## ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_val, y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ROC curve')
display.plot()
plt.show()


# # random forest

# In[33]:


# Create model
RF_model = RandomForestClassifier(n_estimators=9)
RF_model.fit(x_train, y_train)

y_pred = RF_model.predict(x_val)
print(classification_report(y_val, y_pred))
print("Model aniqligi:", accuracy_score(y_val,y_pred))

## confusion matrix
conf_mat = metrics.confusion_matrix(y_val, y_pred)
sns.heatmap(conf_mat, annot=True,fmt="g")
plt.show()

## ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_val, y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ROC curve')
display.plot()


# # XGBClassifier

# In[34]:


xgb_model = XGBClassifier()
xgb_model.fit(x_train, y_train)


y_pred = xgb_model.predict(x_val)
print(classification_report(y_val, y_pred))
print("Model aniqligi:", accuracy_score(y_val,y_pred))

# confusion matrix
conf_mat = metrics.confusion_matrix(y_val, y_pred)
sns.heatmap(conf_mat, annot=True,fmt="g")
plt.show()

# ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_val, y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ROC curve')
display.plot()
plt.show()


# # DecisionTree

# In[35]:


tree_model=DecisionTreeClassifier()
tree_model.fit(x_train,y_train)
y_tree=tree_model.predict(x_val)


# In[36]:


accuracy_score(y_val,y_tree)


# # SVC

# In[37]:


svm_model = SVC()
svm_model.fit(x_train, y_train)

# Modelni baholaymiz

y_pred = svm_model.predict(x_val)
print(classification_report(y_val, y_pred))
print("Model aniqligi:", accuracy_score(y_val,y_pred))

## confusion matrix
conf_mat = metrics.confusion_matrix(y_val, y_pred)
sns.heatmap(conf_mat, annot=True,fmt="g")
plt.show()

## ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_val, y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ROC curve')
display.plot()
plt.show()


# <a id="8"></a>
# ## <b>8 <span style='color:#E1B12A'>Submitting</span></b>

# In[38]:


natija = xgb_model.predict(x_test)
natija


# In[39]:


sample['Transported'] = natija
sample


# In[40]:


sample.replace({0:False,1:True},inplace=True)


# In[41]:


sample.to_csv('spaceship.csv',index=False)


# In[ ]:




