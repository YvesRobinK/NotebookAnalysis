#!/usr/bin/env python
# coding: utf-8

# **Import các thư viện cần thiết**

# In[2]:


import numpy as np # tính toán vector, ma trận
import pandas as pd # thao tác với data frame
import gc
import itertools
from itertools import product
from collections import Counter # tính toán count features
import seaborn as sns # for plot, visualization EDA
import matplotlib.pyplot as plt # for plot, visualization EDA
from sklearn.cluster import KMeans # tính toán cluster features
from sklearn.preprocessing import LabelEncoder # thư viện sklearn giúp cho việc feature engineering như data transformation,..

import category_encoders as ce # library for encoding categorical features
import warnings

pd.set_option('display.max_rows', 400)
pd.set_option('display.max_columns', 160)
pd.set_option('display.max_colwidth', 40)
warnings.filterwarnings("ignore")


# **Kiểm tra xem có bao nhiêu dữ liệu (bảng) có thể dùng**
# 
# 1. Các bạn cũng nên tìm hiểu kỹ về mô tả các trường dữ liệu của các bảng tại đây, ngoài ra cần EDA thêm để hiểu rõ hơn: https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data
# 
# **File descriptions**
# 1. sales_train.csv - the training set. Daily historical data from January 2013 to October 2015.
# 2. test.csv - the test set. You need to forecast the sales for these shops and products for November 2015.
# 3. sample_submission.csv - a sample submission file in the correct format.
# 4. items.csv - supplemental information about the items/products.
# 5. item_categories.csv  - supplemental information about the items categories.
# 6. shops.csv- supplemental information about the shops.
# 
# **Data fields**
# 
# 1. ID - an Id that represents a (Shop, Item) tuple within the test set
# 2. shop_id - unique identifier of a shop
# 3. item_id - unique identifier of a product
# 4. item_category_id - unique identifier of item category
# 5. item_cnt_day - number of products sold. You are predicting a monthly amount of this measure
# 6. item_price - current price of an item
# 7. date - date in format dd/mm/yyyy
# 8. date_block_num - a consecutive month number, used for convenience. January 2013 is 0, February 2013 is 1,..., October 2015 is 33
# 9. item_name - name of item
# 10. shop_name - name of shop
# 11. item_category_name - name of item category

# In[6]:


# list all data
DATA_DIR = "/kaggle/input/competitive-data-science-predict-future-sales/"
import os
print(os.listdir(DATA_DIR))


# **Sử dụng hàm Reduce memory function để Reduce Data Size**

# In[5]:


def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# **Load các dữ liệu cần thiết sử dụng Pandas phục vụ cho việc EDA, Feature Engineering**
# 
# **Note:**
# File data gốc của cuộc thi là tiếng Nga, nên bạn có thể sử dụng bộ data đã được dịch các bảng về categories, items, shops qua tiếng Anh trông dễ nhìn hơn với các trường dạng text: /kaggle/input/predict-future-sales-eng-translation** 

# In[8]:


DATA_DIR_EN = "/kaggle/input/predict-future-sales-eng-translation/"
categories = pd.read_csv(DATA_DIR_EN + 'categories.csv') 
items = pd.read_csv(DATA_DIR_EN + 'items.csv')
sales = pd.read_csv(DATA_DIR + 'sales_train.csv')
shops = pd.read_csv(DATA_DIR_EN + 'shops.csv')


# **Kiểm tra lượng số lượng data các bảng**

# In[9]:


print('categories shape: ', categories.shape)
print('items shape: ', items.shape)
print('sales shape: ', sales.shape)
print('shopes shape: ', shops.shape)


# Dữ liệu ở bảng sales_train là nhiều nhất, thử sử dụng hàm reduce memory xem có thể giảm bớt được data size không

# In[10]:


sales.head()


# In[11]:


sales.info()


# In[12]:


# sử dụng reduce memory function

sales = reduce_mem_usage(sales)


# Giảm bớt được 62.5%

# **Bài tập Thực hiện EDA: Thống kê, Plot các đồ thị ứng, các phương pháp univariate analysis, bi-variate analysis,.. với các biến categorical, numerical, time-series**
# 
# Tham khảo tại đây: https://www.kaggle.com/code/duykhanh99/eda-week1-fds01

# In[ ]:


## các biến categorical

## TO DO


# In[ ]:


## các biến numerical

## TO DO


# In[ ]:


## sử dụng cột date - yếu tố time-series

# TO DO


# **Cleaning Data**

# In[13]:


items.head()


# Cột item_name có dạng text và có thể chưa được clean (chứa những ký tự như !, *,..) --> clean data

# In[ ]:


## example

#clean item_name
items['item_name'] = items['item_name'].str.lower()
items['item_name'] = items['item_name'].str.replace('.', '')

for i in [r'[^\w\d\s\.]', r'\bthe\b', r'\bin\b', r'\bis\b',
          r'\bfor\b', r'\bof\b', r'\bon\b', r'\band\b',  
          r'\bto\b', r'\bwith\b' , r'\byo\b']:
    items['item_name'] = items['item_name'].str.replace(i, ' ')
    
items['item_name'] = items['item_name'].str.replace(r'\b.\b', ' ')


# **Bài tập về Feature Engineering**

# **Tạo thêm các feature text**

# In[ ]:


## TO DO với bảng items, cột item_name

# example couts words, unique words, first n characters of name then counts,..

## các dạng features khác có thể tạo: tf-idf, tf-df + svd, LDA, clustering, word embedding like:word2vec


# **Xem xét bảng Categories**

# In[14]:


categories.head()


# **Note:** Cột category_name có thể breaking-down thành 2 features mới

# In[ ]:


#create broader category groupings
categories['group_name'] = categories['category_name'].str.extract(r'(^[\w\s]*)')
categories['group_name'] = categories['group_name'].str.strip()
# sử dụng label encode cho group names sử dụng thư viện sklearn hoặc pd.factorize của pandas

## TO DO
le = LabelEncoder()
categories['group_id'] = ...

## view
categories.sample(5)


# **Xem xét bảng Sales**
# 
# 1. Có cột date --> xem xét tạo các features về time-series
# 2. Có thể sử dụng thư viện tsfresh đê tạo automatic features

# In[ ]:


sales['date'] = pd.to_datetime(sales.date,format='%d.%m.%Y')
sales['weekday'] = sales.date.dt.dayofweek

## TO DO: tạo thêm các feature như day of month, weekend, holiday,..


## example create some new features

#first day the item was sold, day 0 is the first day of the training set period
sales['first_sale_day'] = sales.date.dt.dayofyear 
sales['first_sale_day'] += 365 * (sales.date.dt.year-2013)

## TO DO: sử dụng hàm groupby first_sale_day theo item_id, statistic transform sử dụng là min
sales['first_sale_day'] = ..

## các feature về mathematical ví dụ như: 
#revenue is needed to accurately calculate prices after grouping
sales['revenue'] = sales['item_cnt_day']*sales['item_price']


# **Give idea:** We calculate the proportion of weekly sales that occurred on each weekday at each shop. Using this information we can assign a measure of weeks of sales power to each month. February always has 4 exactly weeks worth of days since there are no leap years in our time range and all other months have a value >4 since they have extra days of varying sales power.
# 
# Month, year and first day of the month features are also created.

# In[ ]:


## TO DO: sử dụng hàm groupby item_cnt_day theo ['shop_id', 'week_day'], agg: 'sum'
temp = ...
temp = pd.merge(temp, sales.groupby(['shop_id']).agg({'item_cnt_day':'sum'}).reset_index(), on='shop_id', how='left')
temp.columns = ['shop_id','weekday', 'shop_day_sales', 'shop_total_sales']

## tạo thêm features về tỷ lệ giữa shop_day_sales và shop_total_sales
temp['day_quality'] = ...

temp = temp[['shop_id','weekday','day_quality']]

## tạo thêm các features về date-times

dates = pd.DataFrame(data={'date':pd.date_range(start='2013-01-01',end='2015-11-30')})

## TO DO: tính toán các features như dayofweek, month
dates['weekday'] = ...
dates['month'] = ...

dates['year'] = dates.date.dt.year - 2013

dates['date_block_num'] = dates['year']*12 + dates['month'] - 1

dates['first_day_of_month'] = dates.date.dt.dayofyear
dates['first_day_of_month'] += 365 * dates['year']
dates = dates.join(temp.set_index('weekday'), on='weekday')

## sử dụng hàm groupby theo ['date_block_num','shop_id','month','year'] , agg: 'day_quality':'sum','first_day_of_month':'min'

## TO DO ...
dates = dates.groupby(...)


# We now group the sales data by month, shop_id a

# In[ ]:


## TO DO: sử dụng groupby theo ['date_block_num', 'shop_id', 'item_id']
# agg: 'item_cnt_day':'sum', 'revenue':'sum', 'first_sale_day':'first'

# rename tên cột 'item_cnt_day' thành 'item_cnt'

## TO DO
sales = sales.groupby(...)


# In[16]:


## TO DO: clustering shops

#clustering shops theo shop id (có thể thêm tiêu chí khác nữa để cluster hiệu quả hơn) sử dụng KMeans

shops_ = shops.copy()
kmeans = KMeans(n_clusters=7, random_state=0)

## TO DO fit 
....

shops_['shop_cluster'] = kmeans.labels_.astype('int8')
#adding these clusters to the shops dataframe
shops = shops.join(shops_['shop_cluster'], on='shop_id')


# In[ ]:




