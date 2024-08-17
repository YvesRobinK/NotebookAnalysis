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
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.2f}'.format


# # Reading the data from all the data sources and keeping it for future use

# In[2]:


train_df = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/train.csv')
test_df = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/test.csv')
store_df = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/stores.csv')
tr_df = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/transactions.csv')
oil_df = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/oil.csv')
hol_df = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/holidays_events.csv')
sample_df = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/sample_submission.csv')


# # Analysis to understand the training data

# In[3]:


#initial look at training data form and count of rows, columns, type of data etc
train_df[train_df.sales>0]


# In[4]:


# total number of unique stores 
np.sort(train_df.store_nbr.unique())


# In[5]:


# unique family of items
train_df.family.unique()


# In[6]:


train_df.info()


# In[7]:


train_df.describe()


# In[8]:


# to know how the each store have done the business
train_df.groupby(by='store_nbr')['sales'].sum()


# In[9]:


#visual of of the store sales plot, we can clearly see once there are some outliers (huge sales on few days)
import plotly.express as px
fig = px.scatter(train_df[train_df.store_nbr==1], x="date", y="sales")
fig.show()


# In[10]:


#deving date form to different forms of month, year, day for future analysis

train_df['month'] = pd.to_datetime(train_df['date']).dt.month
train_df['day'] = pd.to_datetime(train_df['date']).dt.day
train_df['day_name'] = pd.to_datetime(train_df['date']).dt.day_name()
train_df['year'] = pd.to_datetime(train_df['date']).dt.year
train_df.head(1)


# In[11]:


# check the sales across the months to understand the pattern. In this case we clearly 
# don't see gradual increase of sales for any stores over time. more or less all stores 
# are doing same business  from previous months

table = pd.pivot_table(train_df, values ='sales', index =['store_nbr'],
                         columns =['month'], aggfunc = np.sum)
fig, ax = plt.subplots(figsize=(15,12))         
sns.heatmap(table, annot=False, linewidths=.5, ax=ax, cmap="YlGnBu")
plt.show()


# In[12]:


# check how the different family is impacting the sales, clearly here we have a pattern here, we will use 
# this information later for feature engineering. 

table1 = pd.pivot_table(train_df, values ='sales', index =['family'], aggfunc = np.sum)
fig, ax = plt.subplots(figsize=(10,10))         
sns.heatmap(table1, annot=True, linewidths=.5, ax=ax, cmap="YlGnBu")
plt.show()


# In[13]:


# percentage of sales contributed by each family 

total_sum = table1.sales.sum()
table1/total_sum


# In[14]:


# taking two stores [1 and 2] for checking the impact of promotion on sales.
# clearly there is a week bound between sales and promotion. 
import plotly.express as px
fig = px.scatter(train_df[train_df.store_nbr==1], x="onpromotion", y="sales")
fig.show()


# In[15]:


import plotly.express as px
fig = px.scatter(train_df[train_df.store_nbr==2], x="onpromotion", y="sales")
fig.show()


# In[16]:


# total sales with and without promotion, because of huge data 
# withoug promotion, this graph isn't helping for analysis
import plotly.express as px
df = px.data.tips()
fig = px.histogram(train_df[train_df.onpromotion<200], x="onpromotion", nbins=20)
fig.show()


# In[17]:


# checking if any day of the week have impact on sales, clearly we can see weekend sales is always high
# will use this information for feature engineering.

table3 = pd.pivot_table(train_df, values ='sales', index =['day_name'], aggfunc = np.sum)
fig, ax = plt.subplots(figsize=(10,10))         
sns.heatmap(table3, annot=True, linewidths=.5, ax=ax, cmap="YlGnBu")
plt.show()


# In[18]:


# checking if the sales year have any impact, there is a weak co-relation with year and sales.  
table_year = pd.pivot_table(train_df, values ='sales', index =['store_nbr'],
                         columns =['year'], aggfunc = np.sum)

fig, ax = plt.subplots(figsize=(15,12))         
sns.heatmap(table_year, annot=False, linewidths=.5, ax=ax, cmap="YlGnBu")
plt.show()


# # manipulating the training data for model building/ feature engineering

# In[19]:


# from analysis it was seen that few members sales was high, depending upon sales proportion 
# making the groups for family

family_map       = {'AUTOMOTIVE': 'rest',
                   'BABY CARE': 'rest',
                   'BEAUTY': 'rest',
                   'BOOKS': 'rest',
                   'CELEBRATION': 'rest',
                   'GROCERY II': 'rest',
                   'HARDWARE': 'rest',
                   'HOME AND KITCHEN I': 'rest',
                   'HOME AND KITCHEN II': 'rest',
                   'HOME APPLIANCES': 'rest',
                   'LADIESWEAR': 'rest',
                   'LAWN AND GARDEN': 'rest',
                   'LINGERIE': 'rest',
                   'MAGAZINES': 'rest',
                   'PET SUPPLIES': 'rest',
                   'PLAYERS AND ELECTRONICS': 'rest',
                   'SCHOOL AND OFFICE SUPPLIES': 'rest',
                   'SEAFOOD': 'rest',
                   'DELI': 'first_sec',
                    'EGGS': 'first_sec',
                    'FROZEN FOODS': 'first_sec',
                    'HOME CARE': 'first_sec',
                    'LIQUOR,WINE,BEER': 'first_sec',
                    'PREPARED FOODS': 'first_sec',
                    'PERSONAL CARE': 'first_sec',
                    'BREAD/BAKERY': 'third',
                    'MEATS': 'third',
                    'POULTRY': 'third',
                    'CLEANING':'fourth',
                    'DAIRY':'fourth',
                    'PRODUCE':'seventh',
                    'BEVERAGES':'fifth',
                    'GROCERY I': 'sixth'
                   }

train_df['new_family'] = train_df['family'].map(family_map)
train_df.head(2)


# In[20]:


# Handling the ouliers.


# In[21]:


# graph for before handling the oulier 
import plotly.express as px
fig = px.scatter(train_df[train_df.store_nbr==4], x="date", y="sales")
fig.show()


# In[22]:


# handling the ouliers for each store

for i in range(1,len(train_df.store_nbr.unique())+1):
    val = train_df[train_df.store_nbr == i].sales.quantile(0.99)
    train_df = train_df.drop(train_df[(train_df.store_nbr==i) & (train_df.sales > val)].index)


# In[23]:


#after handling the ouliers
# after removing the outlier columns we will loose nealry 1% of the total rows for all the stores.
fig = px.scatter(train_df[train_df.store_nbr==4], x="date", y="sales")
fig.show()


# # analysis with store metadata and holiday data

# In[24]:


store_df.shape


# In[25]:


# merging the store data with training data
train_df = pd.merge(train_df, store_df, on='store_nbr', how='left') 
train_df.head(3)


# In[26]:


#holiday data
hol_df.head(2)


# In[27]:


# types of holidays
hol_df.locale.unique()


# In[28]:


# unique local names (this info combining with store meta data will help in merging with trining data)
hol_df.locale_name.unique()


# In[29]:


# unique city data
store_df.city.unique()


# In[30]:


store_df.state.unique()


# In[31]:


hol_df.type.unique()


# In[32]:


# renaming a column, since tranining data also has a column 'type' helps for clear merge.
hol_df.rename(columns={'type': 'day_nature'},
          inplace=True, errors='raise')


# In[33]:


# only 1 national name Ecuador
hol_df[hol_df.locale=='National'].head(3)


# In[34]:


# creating different data frames with different holidays to merge
holiday_loc = hol_df[hol_df.locale == 'Local']
holiday_reg = hol_df[hol_df.locale == 'Regional']
holiday_nat = hol_df[hol_df.locale == 'National']


# In[35]:


holiday_loc.rename(columns={'locale_name': 'city'},
          inplace=True, errors='raise')
holiday_reg.rename(columns={'locale_name': 'state'},
          inplace=True, errors='raise')


# In[36]:


holiday_loc


# In[37]:


train_df = pd.merge(train_df, holiday_loc, on=['date', 'city'], how='left') 
train_df = train_df[~((train_df.day_nature == 'Holiday') & (train_df.transferred == False))]
train_df.drop(['day_nature', 'locale', 'description','transferred'], axis=1, inplace=True)
train_df = pd.merge(train_df, holiday_reg, on=['date', 'state'], how='left') 
train_df = train_df[~((train_df.day_nature == 'Holiday') & (train_df.transferred == False))]
train_df.drop(['day_nature', 'locale', 'description','transferred'], axis=1, inplace=True)
train_df = pd.merge(train_df, holiday_nat, on=['date'], how='left') 
train_df = train_df[~((train_df.day_nature == 'Holiday') & (train_df.transferred == False))]
train_df.drop(['day_nature', 'locale', 'description','transferred'], axis=1, inplace=True)


# In[38]:


train_df.drop(['id', 'date', 'family', 'month', 'day','city','state','type', 'cluster', 'locale_name', 'year'],axis=1, inplace=True)


# In[39]:


train_df = pd.get_dummies(train_df, columns = ['day_name','new_family'])
train_df.reset_index(inplace=True)
train_df.drop(['index'],axis=1, inplace=True)
train_df.head(2)


# In[40]:


from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn import preprocessing


for i in range(1,len(train_df.store_nbr.unique())+1):  
    temp_df = train_df[train_df.store_nbr == i]
    sale_out = temp_df[['sales']]
    globals()['max_%s' % i] = temp_df['onpromotion'].max()
    temp_df['onpromotion'] = (temp_df['onpromotion']/globals()['max_%s' % i])
    temp_df.onpromotion = np.where(temp_df.onpromotion<0, 0, temp_df.onpromotion)
    temp_df.drop(['sales','store_nbr'],axis=1, inplace=True)
    globals()['model_%s' % i] = XGBRegressor(verbosity=0)
    globals()['model_%s' % i].fit(temp_df, sale_out)


# In[41]:


test_df['day_name'] = pd.to_datetime(test_df['date']).dt.day_name()
test_df['new_family'] = test_df['family'].map(family_map)
test_df.drop(['date','family'],axis=1, inplace=True)
test_df = pd.get_dummies(test_df, columns = ['day_name','new_family'])
test_df.head(2)


# In[42]:


backup_df_1 = pd.DataFrame()

for i in range(1,len(train_df.store_nbr.unique())+1):
    temp_df = test_df[test_df.store_nbr == i]
    temp_df['onpromotion'] = (temp_df['onpromotion']/globals()['max_%s' % i])
    temp_df.onpromotion = np.where(temp_df.onpromotion<0, 0, temp_df.onpromotion)
    save_id = temp_df[['id']].reset_index()
    temp_df.drop(['id','store_nbr'],axis=1, inplace=True)
    submit = globals()['model_%s' % i].predict(temp_df)
    save_id['sales'] = submit
    df11 = pd.DataFrame(submit, columns = ['sales'])
    backup_df = pd.concat([save_id[['id']], df11], axis = 1, ignore_index = True)
    backup_df_1 = backup_df_1.append(backup_df, ignore_index=True)

backup_df_1.rename(columns={0 : "id", 1 : "sales"}, inplace=True, errors='raise')
test_df = pd.merge(test_df, backup_df_1, on='id', how='left') 


# In[43]:


backup_df_1.head(4)


# In[44]:


sample_df = test_df[['id', 'sales']]
sample_df.sales = np.where(sample_df.sales<0, 0, sample_df.sales)
sample_df.head(3)


# In[45]:


sample_df.to_csv('submission.csv' , index = False)


# In[ ]:




