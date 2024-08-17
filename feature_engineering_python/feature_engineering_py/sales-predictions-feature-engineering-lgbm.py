#!/usr/bin/env python
# coding: utf-8

# # Loading Initial Data

# In[1]:


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from calendar import monthrange
from itertools import product

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
catgs = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
sales = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

testd = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
sampl = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')


# # EDA: Search for Outliers
# 
# Search for NaN values

# In[3]:


print(sales.isna().sum(), '\n')
print(testd.isna().sum())


# No NaN values found, look for data distribution

# In[4]:


plt.figure(figsize=(10,4))
plt.xlim(-100, 3000)
sns.boxplot(x=sales.item_cnt_day)

print('Item count day - Min: {}, Max: {}'.format(sales.item_cnt_day.min(), sales.item_cnt_day.max()))

plt.figure(figsize=(10,4))
plt.xlim(sales.item_price.min(), sales.item_price.max()*1.1)
sns.boxplot(x=sales.item_price)

print('Item price - Min: {}, Max: {}'.format(sales.item_price.min(), sales.item_price.max()))


# As can be seen on the graphs, there are some high outliers on prices and item count. 
# 
# Also there's some negative values on prices and count. Nagative values are expected on count values (devolution cases), but not expected on prices.
# 
# Let's remove the highest outliers and change the strange price values for a common value.

# In[5]:


# Remove outliers
sales = sales[sales.item_price <= 100000]
sales = sales[sales.item_cnt_day <= 1000]

# Adjusting negatice prices (change it for median values)
median = sales[(sales.shop_id == 32) & (sales.item_id == 2973) & (sales.date_block_num == 4) & (sales.item_price > 0)].item_price.median()
sales.loc[sales.item_price < 0, 'item_price'] = median


# # Shops dataset preprocessing
# 
# Since I speak no Russian, I took advantage of other people work to help extract these features. Great part of this code was extracted from [this notebook](https://www.kaggle.com/karell/xgb-baseline-advanced-feature-engineering).
# 
# Several shops are duplicates of each other (according to its name). Fix sales and testd set.

# In[6]:


# Якутск Орджоникидзе, 56
sales.loc[sales.shop_id == 0, 'shop_id'] = 57
testd.loc[testd.shop_id == 0, 'shop_id'] = 57
# Якутск ТЦ "Центральный"
sales.loc[sales.shop_id == 1, 'shop_id'] = 58
testd.loc[testd.shop_id == 1, 'shop_id'] = 58
# Жуковский ул. Чкалова 39м²
sales.loc[sales.shop_id == 10, 'shop_id'] = 11
testd.loc[testd.shop_id == 10, 'shop_id'] = 11
# РостовНаДону ТРК "Мегацентр Горизонт"
sales.loc[sales.shop_id == 39, 'shop_id'] = 40
testd.loc[testd.shop_id == 39, 'shop_id'] = 40


# In[7]:


shops.shop_name.unique()


# Let's categorize shops in ['Орджоникидзе,' 'ТЦ' 'ТРК' 'ТРЦ', 'ул.' 'Магазин' 'ТК' 'склад' ]
# Then transform other values to 'etc'

# In[8]:


shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
shops['shop_category'] = shops['shop_name'].str.split(' ').map(lambda x:x[1]).astype(str)
categories = ['Орджоникидзе,', 'ТЦ', 'ТРК', 'ТРЦ','ул.', 'Магазин', 'ТК', 'склад']
shops.shop_category = shops.shop_category.apply(lambda x: x if (x in categories) else 'etc')
shops.shop_category.unique()


# In[9]:


shops.groupby(['shop_category']).sum()


# However, some categories have small values. So we reduce categories 9 to 5.
# ['Орджоникидзе,', 'ТЦ', 'ТРК', 'ТРЦ','ул.', 'Магазин', 'ТК', 'склад', 'etc'] => ['ТЦ', 'ТРК', 'ТРЦ', 'ТК', 'etc']**

# In[10]:


category = ['ТЦ', 'ТРК', 'ТРЦ', 'ТК']
shops.shop_category = shops.shop_category.apply(lambda x: x if (x in category) else 'etc')
print('Category Distribution', shops.groupby(['shop_category']).sum())

shops['shop_category_code'] = LabelEncoder().fit_transform(shops['shop_category'])


# Extract City name information from the Shop name

# In[11]:


shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
shops['city_code'] = LabelEncoder().fit_transform(shops['city'])
shops = shops[['shop_id','city_code', 'shop_category_code']]

shops.head()


# # Categories dataset preprocessing

# In[12]:


print(len(catgs.item_category_name.unique()))
catgs.item_category_name.unique()


# We think that category 'Игровые консоли' and 'Аксессуары' are same as 'Игры'.
# So, we transform the two features to 'Игры'
# Also, PC - Гарнитуры/Наушники and change to Музыка - Гарнитуры/Наушники

# In[13]:


catgs['type'] = catgs.item_category_name.apply(lambda x: x.split(' ')[0]).astype(str)
catgs.loc[(catgs.type == 'Игровые') | (catgs.type == 'Аксессуары'), 'category'] = 'Игры'
catgs.loc[catgs.type == 'PC', 'category'] = 'Музыка'
category = ['Игры', 'Карты', 'Кино', 'Книги','Музыка', 'Подарки', 'Программы', 'Служебные', 'Чистые', 'Аксессуары']
catgs['type'] = catgs.type.apply(lambda x: x if (x in category) else 'etc')
print(catgs.groupby(['type']).sum())
catgs['type_code'] = LabelEncoder().fit_transform(catgs['type'])

# if subtype is nan then type
catgs['split'] = catgs.item_category_name.apply(lambda x: x.split('-'))
catgs['subtype'] = catgs['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
catgs['subtype_code'] = LabelEncoder().fit_transform(catgs['subtype'])
catgs = catgs[['item_category_id','type_code', 'subtype_code']]

catgs.head()


# # Append train and test data
# 
# Concatenate train (sales) and test (testd) data. Also add manually some missing information on the test data like: date_block_num, year, month, item_cnt_day, item_price.
# 
# `item_price` is a missing information and `item_cnt_day` is part of the information we're trying to predict (in fact we're looking for `item_cnt_month`. `item_cnt_month` is the sum of `item_cnt_day` of a given shop and a given item on a month). For now we're gonna fill these values with 0.
# 

# In[14]:


sales['date'] = pd.to_datetime(sales['date'], format='%d.%m.%Y')
sales['month'] = sales['date'].dt.month
sales['year'] = sales['date'].dt.year
sales = sales.drop(columns=['date'])

# sales.head()
to_append = testd[['shop_id', 'item_id']].copy()

to_append['date_block_num'] = sales['date_block_num'].max() + 1
to_append['year'] = 2015
to_append['month'] = 11
to_append['item_cnt_day'] = 0
to_append['item_price'] = 0

sales = pd.concat([sales, to_append], ignore_index=True, sort=False)
sales.head()


# # Date dataset preprocessing
# 
# Let's remove all date data (except `date_block_num`) from sales and store it on `period`.

# In[15]:


period = sales[['date_block_num', 'year', 'month']].drop_duplicates().reset_index(drop=True)
period['days'] = period.apply(lambda r: monthrange(r.year, r.month)[1], axis=1)

sales = sales.drop(columns=['month', 'year'])

period.head()


# Now let's expand the sales dataset. The new dataset (grid) will contain all combinations of shops and items for every single month.
# 
# This dataset will not contain measures (`item_cnt_day`, `item_price`) data. It will contain base dimensional data (`date_block_num`, `shop_id` and `item_id`).

# In[16]:


index_cols = ['date_block_num', 'shop_id', 'item_id']
grid = [] 
for block_num in sales['date_block_num'].unique():
    cur_shops = sales.loc[sales['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = sales.loc[sales['date_block_num'] == block_num, 'item_id'].unique()
    grid.append(np.array(list(product(*[[block_num], cur_shops, cur_items])), dtype='int16'))

# Turn the grid into a dataframe
grid = pd.DataFrame(np.vstack(grid), columns = index_cols, dtype = np.int16)
grid.head()


# Now, let's add more dimensional data (`date_block_num`, `year`, `month`, `days`, `city_code`, `shop_category_code`, `shop_id`, `item_category_id`, `type_code`, `subtype_code`, `item_id`) to the above dataset. To achieve this goal let's join it with the other pre precessed datasets.
# 
# Also, let's downcast this dataset.

# In[17]:


# Join dimensional data
data = pd.merge(grid, shops, on='shop_id')
data = pd.merge(data, items, on='item_id')
data = pd.merge(data, catgs, on='item_category_id')
data = pd.merge(data, period, on='date_block_num')

# Adjusting columns order
data = data[['date_block_num', 'year', 'month', 'days', 'city_code', 'shop_category_code', 'shop_id', 'item_category_id', 'type_code', 'subtype_code', 'item_id']] # 'item_price', 'item_cnt_day'

# Downcasting values
for c in ['date_block_num', 'month', 'days', 'city_code', 'shop_category_code', 'shop_id', 'item_category_id', 'type_code', 'subtype_code']:
    data[c] = data[c].astype(np.int8)
data['item_id'] = data['item_id'].astype(np.int16)
data['year'] = data['year'].astype(np.int16)

# Remove unused and temporary datasets
del grid, shops, items, catgs, to_append

data.head()


# Now, lets aggregate the measures (`item_cnt_day`, `item_price`) by month. For `item_cnt_day` we're going to sum them to generate `item_cnt_month` values. For `item_price` we're going to average it's values to calculate the `item_price_month`.
# 
# With the measures per month calculated on the step before, let's join this data with the dimensional dataset (calculated on the cell above).
# 
# After this step, we'll generate the month_summary dataset. This dataset contains all the basic information (train + test data) summarized per month.

# In[18]:


aux = sales\
.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False)\
.agg({'item_cnt_day' : 'sum', 'item_price' : 'mean'})\
.rename(columns= {'item_cnt_day' : 'item_cnt_month', 'item_price' : 'item_price_month'})

aux['item_cnt_month'] = aux['item_cnt_month'].astype(np.float16)
aux['item_price_month'] = aux['item_price_month'].astype(np.float16)

month_summary = pd.merge(data, aux, how='left', on=['date_block_num', 'shop_id', 'item_id'])\
    .fillna(0.0).sort_values(by=['shop_id', 'item_id', 'date_block_num'])

del data, aux

month_summary.head()


# In[19]:


print('Min: {} and Max: {} item_cnt_month values'.format(month_summary['item_cnt_month'].min(), month_summary['item_cnt_month'].max()))


# As stated on the problem [evaluation section](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/overview/evaluation):
# 
#     Submissions are evaluated by root mean squared error (RMSE). True target values are clipped into [0,20] range.
#     
# then, let's **clip(0,20)** target (`item_cnt_month`) values. This way train target will be similar to the test predictions.

# In[20]:


month_summary['item_cnt_month'] = month_summary['item_cnt_month'].clip(0,20)


# # Mean Encoded Features
# 
# This section is focused on the generation of new features (measures) based on the existing ones. For instance, we can create a generalization feature that calculate the mean of `item_cnt_month` for every shop on a specific month and add this as a new feature `date_shop_avg_item_cnt`. This technique can be used with other dimensions (`item_id`, `item_category_id`, `city_code`) or a combination of features (`shop_id` + `item_category_id`).
# 
# This is a powerful technique to help generalize the prediction capabilities of a model. However, our test data does not contain any `item_price` or `item_cnt_month` (in fact, we're trying to predict this one) data. That been said, we can't count on any existing or generated actual feature, **but we can** still count on existing or generated features of **past data**. This means that, for instance, we can use the last 12 months prices of an `item_id` or last 3 months of any feature combination (`shop_id` + `item_category_id`).
# 
# To achieve this goal let's define the `agg_by_and_lag` function, it will generate mean encoded features based on an informed `group_cols` list of columns and "lagging" the data N months informed on the `

# In[21]:


def agg_by(month_summary, group_cols, new_col, target_col = 'item_cnt_month', agg_func = 'mean'):
    aux = month_summary\
        .groupby(group_cols, as_index=False)\
        .agg({target_col : agg_func})\
        .rename(columns= {target_col : new_col})
    aux[new_col] = aux[new_col].astype(np.float16)

    return pd.merge(month_summary, aux, how='left', on=group_cols)

def lag_feature(df, col, lags=[1,2,3,6,12]):
    tmp = df[['date_block_num','shop_id','item_id', col]]
    for i in lags:
        shifted = tmp.copy()
        cols = ['date_block_num','shop_id','item_id', '{}_lag_{}'.format(col, i)]
        shifted.columns = cols
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left').fillna(value={(cols[-1]) : 0.0})
    return df

def agg_by_and_lag(month_summary, group_cols, new_col, lags=[1,2,3,6,12], target_col = 'item_cnt_month', agg_func = 'mean'):
    tmp = agg_by(month_summary, group_cols, new_col, target_col, agg_func)
    tmp = lag_feature(tmp, new_col, lags)
    return tmp.drop(columns=[new_col])


# Mean encode and lag `item_cnt_month` data.

# In[22]:


# date_avg_item_cnt
month_summary = agg_by_and_lag(month_summary, ['date_block_num'], 'date_avg_item_cnt', [1])

# date_avg_item_cnt
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'item_id'], 'date_item_avg_item_cnt', [1,2,3,6,12])

# date_city_avg_item_cnt
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'city_code'], 'date_city_avg_item_cnt', [1])

# date_shop_avg_item_cnt
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_id'], 'date_shop_avg_item_cnt', [1,2,3,6,12])

# date_cat_avg_item_cnt
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'item_category_id'], 'date_cat_avg_item_cnt', [1])

# date_type_avg_item_cnt
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'type_code'], 'date_type_avg_item_cnt', [1])

# date_subtype_avg_item_cnt
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'subtype_code'], 'date_subtype_avg_item_cnt', [1])

# date_shop_category_avg_item_cnt
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_category_code'], 'date_shop_category_avg_item_cnt', [1])

# date_shop_cat_avg_item_cnt
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_id', 'item_category_id'], 'date_shop_cat_avg_item_cnt', [1])

# date_shop_type_avg_item_cnt
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_id', 'type_code'], 'date_shop_type_avg_item_cnt', [1])

# date_shop_subtype_avg_item_cnt
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_id', 'subtype_code'], 'date_shop_subtype_avg_item_cnt', [1])

# date_shop_category_subtype_avg_item_cnt
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_category_code', 'subtype_code'], 'date_shop_category_subtype_avg_item_cnt', [1])

# date_item_city_avg_item_cnt
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'city_code', 'item_id'], 'date_item_city_avg_item_cnt', [1])


# Mean encode and lag `item_price_month` data.

# In[23]:


# date_avg_item_price
month_summary = agg_by_and_lag(month_summary, ['date_block_num'], 'date_avg_item_price', [1], 'item_price_month')

# date_item_avg_item_price
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'item_id'], 'date_item_avg_item_price', [1,2,3,6,12], 'item_price_month')

# date_city_avg_item_price
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'city_code'], 'date_city_avg_item_price', [1], 'item_price_month')

# date_shop_avg_item_price
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_id'], 'date_shop_avg_item_price', [1,2,3,6,12], 'item_price_month')

# date_cat_avg_item_price
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'item_category_id'], 'date_cat_avg_item_price', [1], 'item_price_month')

# date_type_avg_item_price
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'type_code'], 'date_type_avg_item_price', [1], 'item_price_month')

# date_subtype_avg_item_price
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'subtype_code'], 'date_subtype_avg_item_price', [1], 'item_price_month')

# date_shop_category_avg_item_price
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_category_code'], 'date_shop_category_avg_item_price', [1], 'item_price_month')

# date_shop_cat_avg_item_price
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_id', 'item_category_id'], 'date_shop_cat_avg_item_price', [1], 'item_price_month')

# date_shop_type_avg_item_price
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_id', 'type_code'], 'date_shop_type_avg_item_price', [1], 'item_price_month')

# date_shop_subtype_avg_item_price
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_id', 'subtype_code'], 'date_shop_subtype_avg_item_price', [1], 'item_price_month')

# date_shop_category_subtype_avg_item_price
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_category_code', 'subtype_code'], 'date_shop_category_subtype_avg_item_price', [1], 'item_price_month')

# date_item_city_avg_item_price
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'city_code', 'item_id'], 'date_item_city_avg_item_price', [1], 'item_price_month')


# # Extra features
# 
# Let's extract some extra features, like the difference in months between and actual sell and the first time it happens.

# In[24]:


month_summary['item_shop_first_sale'] = month_summary['date_block_num'] - month_summary.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
month_summary['item_first_sale'] = month_summary['date_block_num'] - month_summary.groupby('item_id')['date_block_num'].transform('min')


# Our dataset is now ready, let's summarize it.

# In[25]:


month_summary.to_pickle('month_summary.pkl')
month_summary.info()


# # Split Data
# 
# Let's split the generated data into train, validation and test data.
# 
# For test data we will take the last month (34), this is the month we must predict the `item_cnt_month`.
# 
# For the validation data we will use the last month in the original training set (33).
# 
# And for the training data we will use all data between month 12 (since we have lagged some features in 12 months) and 32.

# In[26]:


month_summary = pd.read_pickle('month_summary.pkl')


# In[27]:


def generate_subsample(month_summary, target='item_cnt_month'):
    X_test = month_summary[month_summary['date_block_num'] == 34]
    X_test = X_test.drop(columns=[target])

    X_val = month_summary[month_summary['date_block_num'] == 33]
    y_val = X_val[target]
    X_val = X_val.drop(columns=[target])

    X_train = month_summary[(month_summary['date_block_num'] >= 12) & (month_summary['date_block_num'] < 33)]
    y_train = X_train[target]
    X_train = X_train.drop(columns=[target])

    return X_train, y_train, X_val, y_val, X_test


# In[28]:


X_train, y_train, X_val, y_val, X_test = generate_subsample(month_summary.drop(columns=['item_price_month']), 'item_cnt_month')

del month_summary


# # Train Model
# 
# Lets use the train and validation data to train a simples lightgbm model.

# In[29]:


def train_gbmodel(X_train, y_train, X_val, y_val):

    RAND_SEED = 42

    lgb_params = {'num_leaves': 2**8, 'max_depth': 19, 'max_bin': 107, #'n_estimators': 3747,
              'bagging_freq': 1, 'bagging_fraction': 0.7135681370918421, 
              'feature_fraction': 0.49446461478601994, 'min_data_in_leaf': 2**8, # 88
              'learning_rate': 0.015980721586917768, 'num_threads': 2, 
              'min_sum_hessian_in_leaf': 6,
              'random_state' : RAND_SEED,
              'bagging_seed' : RAND_SEED,
              'boost_from_average' : 'true',
              'boost' : 'gbdt',
              'metric' : 'rmse',
              'verbose' : 1}

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val)

    return lgb.train(lgb_params, lgb_train, 
                      num_boost_round=300,
                      valid_sets=[lgb_train, lgb_val],
                      early_stopping_rounds=20)


# In[30]:


# model_old_item = train_gbmodel(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]).clip(0, 20), X_val, y_val.clip(0, 20))
gbm_model = train_gbmodel(X_train, y_train, X_val, y_val)

y_hat = gbm_model.predict(X_val).clip(0, 20)
print(np.sqrt(mean_squared_error(y_val.clip(0, 20), y_hat)))

with open('./gbm_model.pickle', 'wb') as handle:
    pickle.dump(gbm_model, handle)


# With the trained model, let's finally use it to predict the `item_cnt_month` of the test dataset.

# In[31]:


y_pred = gbm_model.predict(X_test).clip(0, 20)

result = pd.merge(testd, X_test.assign(item_cnt_month=y_pred), how='left', on=['shop_id', 'item_id'])[['ID', 'item_cnt_month']]
result.to_csv('submission.csv', index=False)

