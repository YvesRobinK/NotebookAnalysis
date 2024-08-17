#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering and Initial GBT Model
# 
# This notebook picks up from my first notebook in this competition: [EDA and Previous Value Benchmark](https://www.kaggle.com/noahfinberg/eda-previous-value-benchmark), which achieves a benchmark score of 1.16777. That notebook does basic Exploratory Data Analysis and establishes a previous value benchmark (basically using October 2015 target results as predictions for November 2015). Check out that notebook first if you don't already have a baseline.
# 
# This notebook does some basic feature engineering and trains an initial gradient boosted tree model.
# 
# - Creates lag-based features
# - Creates mean-encoded features
# - Train GBT model
# 
# I adapt https://www.kaggle.com/dlarionov/feature-engineering-xgboost to fit these three objectives. His notebook does a more comprehensive job with feature engineering, but I wanted to build a simpler initial model.
# 
# There is certainly more feature engineering to do, but I'm building out the model iteratively, applying what I learn from the Coursera course after watching each week's lectures. This notebook corresponds to knowledge gained up through Week 3 of the course. The previous notebook reflects Week 2.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# using same dependencies as https://www.kaggle.com/dlarionov/feature-engineering-xgboost

import numpy as np

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)

from itertools import product
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from xgboost import XGBRegressor
from xgboost import plot_importance

def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

import time
import sys
import gc
import pickle
sys.version_info

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Final project advice #3
# 
# ---
# 
# You can get a rather good score after creating some lag-based features like in advice from previous week and feeding them into gradient boosted trees model.
# 
# Apart from item/shop pair lags you can try adding lagged values of total shop or total item sales (which are essentially mean-encodings). All of that is going to add some new information.

# > ## Load Data

# In[2]:


test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")
# items: item_name, item_id, item_category_id
items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv") 
# categories: item_category_name, item_category_id
categories = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")
# train: date, date_block_num, shop_id, item_id, item_price, item_cnt_day**
train = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")
# shops: shop_name, shop_id
shops = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")


# ## Clean the Data
# 
# * **Data Types**:
# The first thing we should do is convert the date column in the training data from an "object" to a "datetime" dtype.
# 
# * **Constant features**:
# There doesn't appear to be any constant features (features with no variation across all rows)
# 
# * **Duplicated features and rows**:
# As we discovered from the Pandas Profiler, there is no missing data and there are no duplicates
# 
# * **Handle Outliers**: Not covered in previous notebook.

# ### Outliers

# Again, see https://www.kaggle.com/dlarionov/feature-engineering-xgboost. I've used his walkthrough as a basis for this one.

# In[3]:


plt.figure(figsize=(10,4))
plt.xlim(-100, 3000)
sns.boxplot(x=train.item_cnt_day) # check item_cnt_day

plt.figure(figsize=(10,4))
plt.xlim(train.item_price.min(), train.item_price.max()*1.1) #set x-axis between min sales price and max sales price
sns.boxplot(x=train.item_price)


# Remove outliers. Price > 100,000 and item_cnt_day > 1500. There is also an item with a price below zero.

# In[4]:


train = train[train.item_price<100000]
train = train[train.item_cnt_day<1500]


# I'm going to impute negative prices with the global median price.

# In[5]:


median = train['item_price'].median()
train.loc[train.item_price<0, 'item_price'] = median


# ### Preprocess Shop and Item Categories

# https://www.kaggle.com/dlarionov/feature-engineering-xgboost makes a really interesting observation. Each shop name and each category name actual contains multiple useful pieces of information
# 
# - Shop names begin with the city they are located in.
# - Category names have both type and subtype of store.
# 
# I don't speak Russian so this wasn't immediately obvious to me :)
# 

# In[6]:


shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0]) # extract city from first part of shop name
shops['city_code'] = LabelEncoder().fit_transform(shops['city']) # give each city a unique encoding
shops = shops[['shop_id','city_code']] # no more string features -- all label encoded

categories['split'] = categories['item_category_name'].str.split('-') # splits on dash
categories['type'] = categories['split'].map(lambda x: x[0].strip()) # type is the first element in the split on category name
categories['type_code'] = LabelEncoder().fit_transform(categories['type']) # give each item category types a unique label

# try to extract a subtype (some category names don't have them)
categories['subtype'] = categories['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip()) # use type is subtype doesn't exist
categories['subtype_code'] = LabelEncoder().fit_transform(categories['subtype'])

categories = categories[['item_category_id','type_code', 'subtype_code']] # no more string features -- all label encoded


# Create a dataframe - matrix - that represents every possible row in the dataset. In other words, we want every combination of shop_id and item_id pairs for each month.

# In[7]:


ts = time.time()
matrix = []
cols = ['date_block_num','shop_id','item_id']
for i in range(34):
    
    sales = train[train.date_block_num==i] # all sales in given date block
    
    # matrix is basically an array of every possible tuple (date_block, shop_id, item_id) combination
    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))
    
    
matrix = pd.DataFrame(np.vstack(matrix), columns=cols) # turns matrix back into df    
# downcast bit representations to save memory
matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)

#sort matrix by date_block_num, then by shop_id, and finally by item_id
matrix.sort_values(cols,inplace=True)

time.time() - ts


# In[8]:


print(train.head())
print(matrix.head())


# Create a new feature "revenue" in the training set. Revenue is equal to the number of items sold per day times the price of those items.

# ### Preprocess Training Data

# In[9]:


train['revenue'] = train['item_price'] *  train['item_cnt_day']


# The target variable is the monthly count so we need to group the data by month and then sum across the item_cnt_day to add up each day's sales in the given month.

# In[10]:


ts = time.time()

group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']}) # aggregates monthly sales per shop and item
group.columns = ['item_cnt_month'] # sets column name
group.reset_index(inplace=True) # resets index to get date_block_num, shop_id, and item_id as columns in the df again

# match up grouped training data with our master matrix of all possible combos (most will have target of zero)
matrix = pd.merge(matrix, group, on=cols, how='left')

# fill missing values, clip target between 0,20
matrix['item_cnt_month'] = (matrix['item_cnt_month']
                                .fillna(0)
                                .clip(0,20) # NB clip target here
                                .astype(np.float16))
time.time() - ts


# ### Append Test Set to Matrix
# 
# We need to append the test set (month 34) to the matrix in order to match things up correctly and use "time tricks."

# In[11]:


ts = time.time()
test['date_block_num'] = 34 # represents Nov. 2015

# downcasts to save memory
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)

matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace=True) # 34 month
time.time() - ts


# ### Append Additional Features to Matrix
# 
# In this case, we want to add the features we extracted earlier from the shops, items, and categories dataframes.

# In[12]:


ts = time.time()
# join on shops
matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')
time.time() - ts


# In[13]:


ts = time.time()

# join on items
matrix = pd.merge(matrix, items, on=['item_id'], how='left')

time.time() - ts


# In[14]:


# ts = time.time()
matrix = pd.merge(matrix, categories, on=['item_category_id'], how='left')
time.time() - ts


# In[15]:


ts = time.time()
# downcast to save memory
matrix['city_code'] = matrix['city_code'].astype(np.int8)
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
matrix['type_code'] = matrix['type_code'].astype(np.int8)
matrix['subtype_code'] = matrix['subtype_code'].astype(np.int8)
time.time() - ts


# ## Generating Feature Lags
# 
# https://www.kaggle.com/dlarionov/feature-engineering-xgboost wrote a nice function to lag features. I replicate it here and walk through how it works.
# 

# In[16]:


# Takes a dataframe, an array of integers representing the magnitudes of all the different lags you want, and the column you want to lag on
def lag_feature(df, lags, col):
    tmp = df[['date_block_num','shop_id','item_id',col]] # when generating the lag we only need these 4 features
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)] # rename columns (this will generate len(lags) new columns)
        
        # change date block number by amount of lag shift
        shifted['date_block_num'] += i
        
        # when we merge back into the original df, it will put enter the lag column at the shifted date_block_num
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df


# In[17]:


ts = time.time()
matrix = lag_feature(matrix, [1,2,3,6,12], 'item_cnt_month')
time.time() - ts


# ## Mean Encodings
# 
# The below mean encodings are copied from https://www.kaggle.com/dlarionov/feature-engineering-xgboost.
# 
# Mean encodings provide a nice way to give your features in the model a baseline. Mean encoded features take the average target value for any given category. 
# 
# So for example, if shop 1 tends to sell 5000 items per month on average and shop 2 only sells 2000 per month, our feature can now account for that. Some intuition: if we were given a row with shop 1, it'd be better for us to guess the target value at 5000 than at 2000. On average we'd be right most often.

# In[18]:


# average items sold per month for each data block
# this feature captures baseline for a given month estimate. 
# Of course we don't have mean encodings for month 34, but encodings for previous months may help, esp. if there is some seasonality in the data
# E.g. maybe number of items sold in November 2014 is somewhat predictive of number of items sold in November 2015.
ts = time.time()

group = matrix.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num'], how='left')
matrix['date_avg_item_cnt'] = matrix['date_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_avg_item_cnt') # creates a lag for every date block
matrix.drop(['date_avg_item_cnt'], axis=1, inplace=True) # no longer need the original encoding
time.time() - ts


# In[19]:


ts = time.time()
group = matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_item_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')
matrix['date_item_avg_item_cnt'] = matrix['date_item_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1,2,3,6,12], 'date_item_avg_item_cnt')
matrix.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts


# In[20]:


ts = time.time()
group = matrix.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_shop_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')
matrix['date_shop_avg_item_cnt'] = matrix['date_shop_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1,2,3,6,12], 'date_shop_avg_item_cnt')
matrix.drop(['date_shop_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts


# In[21]:


ts = time.time()
group = matrix.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_cat_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','item_category_id'], how='left')
matrix['date_cat_avg_item_cnt'] = matrix['date_cat_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_cat_avg_item_cnt')
matrix.drop(['date_cat_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts


# In[22]:


ts = time.time()
group = matrix.groupby(['date_block_num', 'shop_id', 'item_category_id']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_cat_avg_item_cnt']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
matrix['date_shop_cat_avg_item_cnt'] = matrix['date_shop_cat_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_shop_cat_avg_item_cnt')
matrix.drop(['date_shop_cat_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts


# In[23]:


ts = time.time()
group = matrix.groupby(['date_block_num', 'shop_id', 'type_code']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_type_avg_item_cnt']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'type_code'], how='left')
matrix['date_shop_type_avg_item_cnt'] = matrix['date_shop_type_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_shop_type_avg_item_cnt')
matrix.drop(['date_shop_type_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts


# In[24]:


ts = time.time()
group = matrix.groupby(['date_block_num', 'shop_id', 'subtype_code']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_subtype_avg_item_cnt']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'subtype_code'], how='left')
matrix['date_shop_subtype_avg_item_cnt'] = matrix['date_shop_subtype_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_shop_subtype_avg_item_cnt')
matrix.drop(['date_shop_subtype_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts


# In[25]:


ts = time.time()
group = matrix.groupby(['date_block_num', 'city_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_city_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'city_code'], how='left')
matrix['date_city_avg_item_cnt'] = matrix['date_city_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_city_avg_item_cnt')
matrix.drop(['date_city_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts


# In[26]:


ts = time.time()
group = matrix.groupby(['date_block_num', 'item_id', 'city_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_item_city_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id', 'city_code'], how='left')
matrix['date_item_city_avg_item_cnt'] = matrix['date_item_city_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_item_city_avg_item_cnt')
matrix.drop(['date_item_city_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts


# In[27]:


ts = time.time()
group = matrix.groupby(['date_block_num', 'type_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_type_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'type_code'], how='left')
matrix['date_type_avg_item_cnt'] = matrix['date_type_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_type_avg_item_cnt')
matrix.drop(['date_type_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts


# In[28]:


ts = time.time()
group = matrix.groupby(['date_block_num', 'subtype_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_subtype_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'subtype_code'], how='left')
matrix['date_subtype_avg_item_cnt'] = matrix['date_subtype_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_subtype_avg_item_cnt')
matrix.drop(['date_subtype_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts


# In[29]:


# check out all the new lag and mean encoding features!
matrix.head()


# https://www.kaggle.com/dlarionov/feature-engineering-xgboost covers trend features and other special features. That is out of the scope of this notebook. I'll return to these kinds of futures in a later notebook.

# In[30]:


# save the matrix so it's easier to reload later and in future notebooks
matrix.to_pickle('data.pkl')


# ## XGBoost Model
# 
# Since we're using a yearlong lag (lag of 12), we only need to really train our model on all months of data after the first 11. In all the previous months, a lag of 12 would mean all NaNs. We could still train our model on lags under 12, but we're guessing that they won't add more predictive power.
# 
# In general, lags create a ton of NaN values so we can fill all remaining NaNs with zeros.

# In[31]:


ts = time.time()
matrix = matrix[matrix.date_block_num > 11]
time.time() - ts


# In[32]:


ts = time.time()
def fill_na(df):
    for col in df.columns:
        if ('_lag_' in col) & (df[col].isnull().any()):
            if ('item_cnt' in col):
                df[col].fillna(0, inplace=True)         
    return df

matrix = fill_na(matrix)
time.time() - ts


# In[33]:


# check out all the features we currently have
matrix.columns


# In[34]:


# quick sanity check on what's in matrix
matrix.info()


# In[35]:


# save the matrix so it's easier to reload later
matrix.to_pickle('data.pkl')
# del matrix
# del group
# del items
# del shops
# del cats
# del train
# # leave test for submission
# gc.collect();


# ### Reload the preprocessed data

# In[36]:


data = pd.read_pickle('data.pkl')


# In[37]:


# pick which features we want.
data = data[
    [
    'date_block_num',
    'shop_id',
    'item_id',
    'item_cnt_month',
#     'ID',
    'city_code',
#     'item_name',
    'item_category_id',
    'type_code',
    'subtype_code',
    'item_cnt_month_lag_1', 
    'item_cnt_month_lag_2',
    'item_cnt_month_lag_3',
    'item_cnt_month_lag_6',
    'item_cnt_month_lag_12',
    'date_avg_item_cnt_lag_1',
    'date_item_avg_item_cnt_lag_1',
    'date_item_avg_item_cnt_lag_2',
    'date_item_avg_item_cnt_lag_3',
    'date_item_avg_item_cnt_lag_6',
    'date_item_avg_item_cnt_lag_12',
    'date_shop_avg_item_cnt_lag_1',
    'date_shop_avg_item_cnt_lag_2',
    'date_shop_avg_item_cnt_lag_3',
    'date_shop_avg_item_cnt_lag_6',
    'date_shop_avg_item_cnt_lag_12',
    'date_cat_avg_item_cnt_lag_1',
    'date_shop_cat_avg_item_cnt_lag_1',
    'date_shop_type_avg_item_cnt_lag_1',
    'date_shop_subtype_avg_item_cnt_lag_1',
    'date_city_avg_item_cnt_lag_1',
    'date_item_city_avg_item_cnt_lag_1',
    'date_type_avg_item_cnt_lag_1',
    'date_subtype_avg_item_cnt_lag_1']
]


# ### Validation Approach
# 
# We will us month 34 as the test set. Month 33 can serve as the validation set for our model. We'll train the model on all other months. Before we submit, we should make sure to train the model on month 33 as well.

# In[38]:


X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)


# In[39]:


del data
gc.collect();


# In[40]:


ts = time.time()

model = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=42)

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 10)

time.time() - ts


# In[41]:


Y_pred = model.predict(X_valid).clip(0, 20)
Y_test = model.predict(X_test).clip(0, 20)

submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test
})
submission.to_csv('xgb_submission.csv', index=False)

# save predictions for an ensemble
pickle.dump(Y_pred, open('xgb_train.pickle', 'wb'))
pickle.dump(Y_test, open('xgb_test.pickle', 'wb'))


# In[42]:


plot_features(model, (10,14))


# In[ ]:




