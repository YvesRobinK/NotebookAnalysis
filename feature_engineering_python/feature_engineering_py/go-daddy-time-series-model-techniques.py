#!/usr/bin/env python
# coding: utf-8

# # Executive Summary
# 
# This notebook did justice to handling time series problem as a regression problem.
# 
# * This notebook shows Lightgbm seems to be the best model when compared with Catboost and Lightgbm.
# 
# * Extensive Feature Engineering (Adding Time series, Cyclic and Census data features) improve the model from 5.6 to 3.13.

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


# Libraries to help with reading and manipulating data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
default_color = sns.color_palette()[0]
# libaries to help with data visualization
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn import metrics
from sklearn.model_selection import KFold, TimeSeriesSplit


# In[4]:


## The 3 machine learning model
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


# # Evaluation metrics function

# In[5]:


def smape(y_true, y_pred):
    nume = np.abs(y_true - y_pred)
    deno = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    return 100 * np.mean(nume / deno)


# In[6]:


train = pd.read_csv('/kaggle/input/godaddy-microbusiness-density-forecasting/train.csv', parse_dates=['first_day_of_month'])
test = pd.read_csv('/kaggle/input/godaddy-microbusiness-density-forecasting/test.csv', parse_dates=['first_day_of_month'])
sub = pd.read_csv('/kaggle/input/godaddy-microbusiness-density-forecasting/sample_submission.csv')
cs_data = pd.read_csv('/kaggle/input/godaddy-microbusiness-density-forecasting/census_starter.csv')


# In[7]:


print(train.shape)
print(test.shape)
print(sub.shape)
print(cs_data.shape)


# In[8]:


print('The total shape of the train data is {}'.format(train.shape))
print('The total shape of the test data is {}'.format(test.shape))
print('The total shape of the census data is {}'.format(cs_data.shape))


# In[9]:


train.head()


# In[10]:


test.head()


# In[11]:


cs_data.head()


# **Investigating the data**

# In[12]:


train.info()


# In[13]:


test.info()


# In[14]:


cs_data.info()


# **Observation**
# 
# * There is no date features in the census data.
# 
# * The Primary key is cfips.

# ## Exploring the dataset
# 
# In this section we are going to be exploring the train and test data

# **Checking the Target Distribution**

# In[15]:


## The distribution of the target variable
plt.title('The Target variable distribution')
train['microbusiness_density'].hist(bins=20);


# In[16]:


plt.figure(figsize = [20, 10]) 
plt.subplot(1, 2, 1) 
train['microbusiness_density'].hist(bins=20);

plt.subplot(1, 2, 2) # 1 row, 2 cols, subplot 2
(np.log1p(train['microbusiness_density']).hist(bins=20));


# **Observation**
# 
# We observe that the distribution of the original target feature is highly right skewed but after log transformation it becomes to assume a normal distribution which might be a good Distribution when fitting the model.

# In[17]:


train['microbusiness_density'].describe()


# In[18]:


tar_feat = train[['row_id', 'microbusiness_density']]


# **The Time Feature**

# In[19]:


train['first_day_of_month'].describe(datetime_is_numeric=True)


# In[20]:


test['first_day_of_month'].describe(datetime_is_numeric=True)


# **Observation**
# 
# * The training data start from 2019-08-01 and end at 2022-10-01.
# * The test data start from 2022-11-01 and end at 2023-06-01.
# 
# In a nutshell, we have future time series data that is not on Kaggle Test set.

# In[21]:


train.plot( 'first_day_of_month' , 'microbusiness_density' );


# **There seems to be no meaningful trend in the date time data**

# **The Unique identifer column (cfips)**

# In[22]:


print('The total number of cfips train data is {}'.format(train['cfips'].nunique()))
print('The total number of cfips test data is {}'.format(test['cfips'].nunique()))
print('The total number of cfips census data is {}'.format(cs_data['cfips'].nunique()))


# In[23]:


### check whether we have the same cfips in the train and test data
test['cfips'].isin(train['cfips']).value_counts()


# In[24]:


### check whether we have the same cfips in the train and census data
cs_data['cfips'].isin(train['cfips']).value_counts()


# **Observation**
# 
# * Merging the Census data to enrich our dataset is visible.

# # Modelling
# 
# In this section we are going to be modelling the problem as a regression problem, then we are going to compare 3 machine learning models (Catboost, Lightgbm and Xgboost) since there is no free lunch model.

# In[25]:


train.head()


# # Little Feature Engineering

# In[26]:


def make_feature(df):
    feature = pd.DataFrame()
    feature["contry_code"] = df["cfips"] // 100
    feature["state_code"] = df["cfips"] % 100
    feature["year"] = df["first_day_of_month"].dt.year
    feature["month"] = df["first_day_of_month"].dt.month
    feature["week"] = df["first_day_of_month"].dt.dayofweek
    
    return feature


# In[27]:


train_fe = make_feature(train)
test_fe = make_feature(test)


# In[28]:


print(train_fe.shape)
print(test_fe.shape)


# In[29]:


X = train_fe
y = np.log1p(train['microbusiness_density'])


# ### Catboost Model
# 
# CatBoost is a high-performance open source library for gradient boosting on decision trees.
# 
# Read more - https://catboost.ai/

# In[30]:


errcb=[]
y_pred_totcb=[]
fold= KFold(n_splits=5, shuffle=True, random_state=1)
i=1
for train_index, test_index in fold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    m = CatBoostRegressor(eval_metric='SMAPE')
    m.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_test, y_test)],verbose=100, early_stopping_rounds=100)
    preds=m.predict(X_test)
    print("err: ",smape(np.expm1(y_test), np.expm1(preds)))
    errcb.append(smape(np.expm1(y_test), np.expm1(preds)))
    p = m.predict(test_fe)
    y_pred_totcb.append(p)


# In[31]:


np.mean((errcb))


# ### LightGbm Model

# In[32]:


params = {
        "metric" : "mse",
        "learning_rate" : 0.2,
         "sub_feature" : 1.0,
        "bagging_freq" : 1,
        "lambda_l2" : 0.6,
        'verbosity': 1,
       'num_iterations' : 3000,        
}


# In[33]:


errlgb=[]
y_pred_totlgb=[]
fold= KFold(n_splits=5, shuffle=True, random_state=101)
i=1
for train_index, test_index in fold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    m = LGBMRegressor(**params)
    m.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_test, y_test)],verbose=100, early_stopping_rounds=100)
    preds=m.predict(X_test)
    print("err: ",smape(np.expm1(y_test), np.expm1(preds)))
    errlgb.append(smape(np.expm1(y_test), np.expm1(preds)))
    p = m.predict(test_fe)
    y_pred_totlgb.append(p)


# In[34]:


np.mean((errlgb))


# In[35]:


fea_imp = pd.DataFrame({'imp':m.feature_importances_, 'col': X.columns})
fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=False).iloc[-30:]
_ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))
plt.savefig('catboost_feature_importance.png')  


# ### XGBRegressor

# In[36]:


param = {
    'gamma': 10,
    'learning_rate': 0.2,
    'max_depth': 6,
    'n_estimators': 3000,
    'reg_alpha': 10,
    'subsample': 1.0
}


# In[37]:


errxgb=[]
y_pred_totxgb=[]
fold= KFold(n_splits=5, shuffle=True, random_state=1)
i=1
for train_index, test_index in fold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    m = XGBRegressor(**param)
    m.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_test, y_test)],verbose=100, early_stopping_rounds=100)
    preds=m.predict(X_test)
    print("err: ",smape(np.expm1(y_test), np.expm1(preds)))
    errxgb.append(smape(np.expm1(y_test), np.expm1(preds)))
    p = m.predict(test_fe)
    y_pred_totxgb.append(p)


# In[38]:


np.mean((errxgb))


# **Conclusion**
# 
# We observe that Lightgbm is the best Time series model for this problem, with base line feature engineering.

# # Extensive Feature Engineering
# 
# In this section we are going to be using various time series feature engineering techniques
# 
# 1. Time series features
# 
# 2. Cyclic Time Series features (sin and cos)
# 
# 3. Adding the Census data

# In[39]:


def make_feature(df):
    feature = pd.DataFrame()
    feature["contry_code"] = df["cfips"] // 100
    feature["state_code"] = df["cfips"] % 100
    feature["year"] = df["first_day_of_month"].dt.year
    feature["month"] = df["first_day_of_month"].dt.month
    feature["week"] = df["first_day_of_month"].dt.dayofweek
    feature['dayofyear'] = df['first_day_of_month'].dt.dayofyear
    feature['dayofmonth'] = df['first_day_of_month'].dt.day
    feature['weekofyear'] = df['first_day_of_month'].dt.weekofyear
    
    return feature


def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data


# # Adding Time series and Cyclic data

# In[40]:


train_f1 = make_feature(train)
test_f1 = make_feature(test)


# In[41]:


train_f1 = encode(train_f1, 'year', 4)
train_f1 = encode(train_f1, 'month', 12)
train_f1 = encode(train_f1, 'week', 7)


# In[42]:


test_f1 = encode(test_f1, 'year', 2)
test_f1 = encode(test_f1, 'month', 12)
test_f1 = encode(test_f1, 'week', 7)


# # Adding census data

# In[43]:


train_f1['cfips'] = train['cfips']
test_f1['cfips'] = test['cfips']


# In[44]:


train_f1['row_id'] = train['row_id']
test_f1['row_id'] = test['row_id']


# In[45]:


train_f1 = train_f1.merge(cs_data, on=['cfips'])
test_f1 = test_f1.merge(cs_data, on=['cfips'])


# In[46]:


test_id = test_f1['row_id']


# In[47]:


## Adding the target feature
train_f1 = train_f1.merge(tar_feat, on=['row_id'])


# In[48]:


train_f1.drop('row_id',axis=1,inplace=True)
test_f1.drop('row_id',axis=1,inplace=True)


# In[49]:


print(train_f1.shape)
print(test_f1.shape)


# ### Modelling with Time series features, Cyclic and Census data features

# In[50]:


X = train_f1.drop('microbusiness_density', axis=1)
y = np.log1p(train_f1['microbusiness_density'])


# In[51]:


params = {
        "metric" : "mse", 'num_leaves': 64,
         "sub_feature" : 1.0,
        "bagging_freq" : 1,
        "lambda_l2" : 0.9,
        'verbosity': 1,
       'num_iterations' : 3000,
    
}


# In[52]:


errlgb=[]
y_pred_totlgb=[]
fold= KFold(n_splits=5, shuffle=True, random_state=101)
i=1
for train_index, test_index in fold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    m = LGBMRegressor(**params)
    m.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_test, y_test)],verbose=100, early_stopping_rounds=100)
    preds=m.predict(X_test)
    print("err: ",smape(np.expm1(y_test), np.expm1(preds)))
    errlgb.append(smape(np.expm1(y_test), np.expm1(preds)))
    p = m.predict(test_f1)
    y_pred_totlgb.append(p)


# In[53]:


np.mean((errlgb))


# In[54]:


fea_imp = pd.DataFrame({'imp':m.feature_importances_, 'col': X.columns})
fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=False).iloc[-30:]
_ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))
plt.savefig('catboost_feature_importance.png')  


# # Submit Baseline Model

# In[55]:


d = {"row_id": test_id, 'microbusiness_density': abs(np.expm1(np.mean(y_pred_totlgb, 0)))}
test_predictions = pd.DataFrame(data=d)
test_predictions = test_predictions[["row_id", 'microbusiness_density']]


# In[56]:


test_predictions.to_csv("submission.csv", index=False)


# In[57]:


test_predictions.describe()


# ## Final Note
# 
# * Lag features can be added
# 
# * Holiday features can be added
# 
# * Weather Condition Features can be added.

# ## Thank You😊 

# In[ ]:




