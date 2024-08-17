#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:82c8de5d-31c0-499e-b872-5ca3d3487500.png)

# This kernel is going to solve <font color="red"><b> Predict Future Sales with Advanced Regression Analysis</b></font>, a popular machine learning dataset for <b>Kaggler</b>.<br> 
# I am going to share how I work with a dataset step by step  <b>from data preparation and data analysis to statistical tests and implementing machine learning models.</b> <br>
# I will also describe the model results along with other tips.<br>
# Let's get started.</div>
# <br>
# If you like this notebook or find this notebook helpful, Please feel free to <font color="red"><b>UPVOTE</b></font> and/or leave a comment.
#  
# <div> <b>This notebook is always a work in progress. So, please stay tuned for more to come.</b></div>
# 
# 
# This notebook referenced the following.<br>
# https://www.kaggle.com/deinforcement/top-1-predict-future-sales-features-lightgbm<br>
# https://www.kaggle.com/werooring/top-3-5-lightgbm-with-feature-engineering<br>
# https://www.kaggle.com/dlarionov/feature-engineering-xgboost

# # 1: Importing Necessary Libraries and datasets

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')

import seaborn as sns
import sys
import itertools
import gc

from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

import csv
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import datetime


# In[2]:


items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")
shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")
cats = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")
test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")


# # 2. Data Preprocessing

# In[3]:


# Change data type for date calculation
train["date"] = pd.to_datetime(train["date"], format="%d.%m.%Y")
# Merge some duplicate shops
train["shop_id"] = train["shop_id"].replace({0: 57, 1: 58, 11: 10, 40: 39})
# Keep only shops that are in the test set
train = train.loc[train.shop_id.isin(test["shop_id"].unique()), :]
# Drop training items with extreme or negative prices or sales counts
train = train[(train["item_price"] > 0) & (train["item_price"] < 50000)]
train = train[(train["item_cnt_day"] > 0) & (train["item_cnt_day"] < 1000)]


# In[4]:


shops['city'] = shops['shop_name'].apply(lambda x: x.split()[0])
shops.loc[shops['city'] =='!Якутск', 'city'] = 'Якутск'

# Create Label Encoder
label_encoder = LabelEncoder()
# City Feature Label Encoding 
shops['city'] = label_encoder.fit_transform(shops['city'])


# In[5]:


# Create the date the product was first sold as a feature
items['first_sale_date'] = train.groupby('item_id').agg({'date_block_num': 'min'})['date_block_num']
# Refine null values in items table
items = items.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)


# In[6]:


cats['split'] = cats['item_category_name'].str.split('-')
cats['type'] = cats['split'].map(lambda x: x[0].strip())
cats['type_code'] = LabelEncoder().fit_transform(cats['type'])
# if subtype is nan then type
cats['subtype'] = cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype'])
cats = cats[['item_category_id','type_code', 'subtype_code']]


# In[7]:


# Merge data
train = train.merge(shops, on='shop_id', how='left')
train = train.merge(items, on='item_id', how='left')
train = train.merge(cats, on='item_category_id', how='left')
test = test.merge(shops, on='shop_id', how='left')
test = test.merge(items, on='item_id', how='left')
test = test.merge(cats, on='item_category_id', how='left')


# In[8]:


test["date_block_num"] = 34
test["date_block_num"] = test["date_block_num"].astype(np.int8)
test["shop_id"] = test.shop_id.astype(np.int8)
test["item_id"] = test.item_id.astype(np.int16)
test = test.drop(columns="ID")


# In[9]:


df = pd.concat([train, test])


# In[10]:


group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
group.columns = ['item_cnt_month']
group.reset_index(inplace=True)

df = df.merge(group, on=['date_block_num','shop_id','item_id'], how='left')
df['item_cnt_month'] = (df['item_cnt_month']
                                .fillna(0)
                                .clip(0,20) # NB clip target here
                                .astype(np.float16))


# In[11]:


df = df.drop(['date','shop_name','item_name','item_cnt_day'], axis=1)


# In[12]:


def add_mean(df, idx_features):
    # Check base features
    assert (idx_features[0] == 'date_block_num') and \
           len(idx_features) in [2, 3]
    
    # Set derived feature name 
    if len(idx_features) == 2:
        feature_name = idx_features[1] + '_mean_sales'
    else:
        feature_name = idx_features[1] + '_' + idx_features[2] + '_mean_sales'
    
    # Get average monthly sales by grouping based on base features
    group = df.groupby(idx_features).agg({'item_cnt_month': 'mean'})
    group = group.reset_index()
    group = group.rename(columns={'item_cnt_month': feature_name})
    
    # Merge df with group based on idx_features
    df = df.merge(group, on=idx_features, how='left')
    
    return df


# In[13]:


# Create monthly average sales derived features grouped by ['date_block_num', 'item_id']
df = add_mean(df=df, idx_features=['date_block_num', 'item_id'])

# Create monthly average sales derived features grouped by ['date_block_num', 'shop_id']
df = add_mean(df=df, idx_features=['date_block_num', 'shop_id'])

# Create monthly average sales derived features grouped by ['date_block_num', 'item_category_id']
df = add_mean(df=df, idx_features=['date_block_num', 'item_category_id'])

# Create monthly average sales derived features grouped by ['date_block_num', 'item_id', 'city']
df = add_mean(df=df, idx_features=['date_block_num', 'item_id', 'city'])

# Create monthly average sales derived features grouped by ['date_block_num', 'shop_id', 'item_category_id']
df = add_mean(df=df, idx_features=['date_block_num', 'shop_id', 'item_category_id'])

# Create monthly average sales derived features grouped by ['date_block_num', 'shop_id', 'subtype_code']
df = add_mean(df=df, idx_features=['date_block_num', 'shop_id', 'subtype_code'])


# In[14]:


df['duration_after_first_sale'] = df['date_block_num'] - df['first_sale_date']
df['month'] = df['date_block_num']%12


# # 3. Modeling

# In[15]:


keep_from_month = 2  # The first couple of months are dropped because of distortions to their features (e.g. wrong item age)
valid_month = 33

valid = df.loc[df.date_block_num == valid_month, :]
train = df.loc[df.date_block_num < valid_month, :]
train = train[train.date_block_num >= keep_from_month]
X_train = train.drop(columns="item_cnt_month")
y_train = train.item_cnt_month
X_valid = valid.drop(columns="item_cnt_month")
y_valid = valid.item_cnt_month
test = df.drop(columns="item_cnt_month").loc[df.date_block_num == 34, :]


# In[16]:


model_lgb = LGBMRegressor(colsample_bytree=0.8,learning_rate=0.01, max_depth=8,
              min_child_weight=1, min_split_gain=0.0222415, n_estimators=35000,
              num_leaves=966, reg_alpha=0.04, reg_lambda=0.073,
              subsample=0.6)


# In[17]:


start = datetime.datetime.now()
lgbm = model_lgb.fit(X_train, 
                     y_train,
                     eval_set = [(X_train, y_train), (X_valid, y_valid)],
                     eval_metric ='rmse',
                     early_stopping_rounds = 400,
                     verbose =True)
end = datetime.datetime.now()
end-start


# In[18]:


feature_imp= pd.DataFrame(sorted(zip(lgbm.feature_importances_, X_train.columns), reverse = True), columns = ['Value', 'Feature'])
# feature_imp.to_excel("feature_imp.xlsx")

plt.figure(figsize=(7,5))
sns.barplot(x='Value', y='Feature', data=feature_imp[:10].sort_values(by='Value', ascending=False))
plt.tight_layout()
plt.show()
# plt.savefig('lightGBM_ Importances.png')


# In[19]:


sales_prediction = model_lgb.predict(test)
test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
test_result= pd.DataFrame(sales_prediction)
test_result.columns = ['item_cnt_month']
predict = test_result['item_cnt_month']
sales_predict_submission = pd.DataFrame({'ID':test['ID'],'item_cnt_month':predict})
sales_predict_submission.to_csv('submission.csv', index=False)

