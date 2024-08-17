#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


# importing the libraries

df_train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv', parse_dates = ['date'], infer_datetime_format = True, dayfirst = True)
df_test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
df_shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
df_items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
df_item_categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')


# In[4]:


# checking the shapes of these datasets
print("Shape of train:", df_train.shape)
print("Shape of test:", df_test.shape)
print("Shape of shops:", df_shops.shape)
print("Shape of items:", df_items.shape)
print("Shape of item_categories:", df_item_categories.shape)


# In[5]:


df_train


# In[6]:


df_test


# In[7]:


df_shops


# In[8]:


df_items


# In[9]:


df_item_categories


# In[10]:


df_train.describe().T


# In[11]:


# checking if there is any Null data inside the given data

print("No. of Null values in the train set :", df_train.isnull().sum().sum())
print("No. of Null values in the test set :", df_test.isnull().sum().sum())
print("No. of Null values in the item set :", df_items.isnull().sum().sum())
print("No. of Null values in the shops set :", df_shops.isnull().sum().sum())
print("No. of Null values in the item_categories set :", df_item_categories.isnull().sum().sum())


# In[12]:


# looking at the number of different categories
plt.rcParams['figure.figsize'] = (22, 12)
sns.barplot(df_items['item_category_id'], df_items['item_id'], palette = 'colorblind')
plt.title('Count for Different Items Categories', fontsize = 30)
plt.xlabel('Item Categories', fontsize = 15)
plt.ylabel('Items in each Categories', fontsize = 15)
plt.show()


# In[13]:


# having a look at the distribution of item sold per day

plt.rcParams['figure.figsize'] = (20, 10)
sns.countplot(df_train['date_block_num'])
plt.title('Date blocks according to months', fontsize = 30)
plt.xlabel('Different blocks of months', fontsize = 15)
plt.ylabel('No. of Purchases', fontsize = 15)
plt.show()


# In[14]:


# having a look at the distribution of item price

plt.rcParams['figure.figsize'] = (15, 10)
sns.distplot(df_train['item_price'], color = 'red')
plt.title('Distribution of the price of Items', fontsize = 30)
plt.xlabel('Range of price of items', fontsize = 15)
plt.ylabel('Distrbution of prices over items', fontsize = 15)
plt.show()


# In[15]:


# having a look at the distribution of item sold per day

plt.rcParams['figure.figsize'] = (16, 8)
sns.distplot(df_train['item_cnt_day'], color = 'purple')
plt.title('Distribution of the no. of Items Sold per Day', fontsize = 30)
plt.xlabel('Range of items sold per day', fontsize = 15)
plt.ylabel('Distrbutions per day', fontsize = 15)
plt.show()


# In[16]:


# checking the no. of unique item present in the stores

x = df_train['item_id'].nunique()
print("The No. of Unique Items Present in the stores available: ", x)


# In[17]:


# checking the no. of unique item present in the stores

x = df_item_categories['item_category_id'].nunique()
print("The No. of Unique categories for Items Present in the stores available: ", x)


# In[18]:


# checking the no. of unique shops given in the dataset

x = df_train['shop_id'].nunique()
print("No. of Unique Shops are :", x)


# In[19]:


# making a word cloud for item categories name

from wordcloud import WordCloud
from wordcloud import STOPWORDS

plt.rcParams['figure.figsize'] = (18, 12)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'lightblue',
                      max_words = 200, 
                      stopwords = stopwords,
                     width = 1200,
                     height = 800,
                     random_state = 42).generate(str(df_item_categories['item_category_name']))


plt.title('Wordcloud for Item Category Names', fontsize = 30)
plt.axis('off')
plt.imshow(wordcloud, interpolation = 'bilinear')


# In[20]:


# making a word cloud for item name

from wordcloud import WordCloud
from wordcloud import STOPWORDS

plt.rcParams['figure.figsize'] = (18, 12)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'pink',
                      max_words = 200, 
                      stopwords = stopwords,
                     width = 1200,
                     height = 800,
                     random_state = 42).generate(str(df_items['item_name']))


plt.title('Wordcloud for Item Names', fontsize = 30)
plt.axis('off')
plt.imshow(wordcloud, interpolation = 'bilinear')


# In[21]:


# making a word cloud for shop name

from wordcloud import WordCloud
from wordcloud import STOPWORDS

plt.rcParams['figure.figsize'] = (18, 12)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'gray',
                      max_words = 200, 
                      stopwords = stopwords,
                     width = 1200,
                     height = 800,
                     random_state = 42).generate(str(df_shops['shop_name']))


plt.title('Wordcloud for Shop Names', fontsize = 30)
plt.axis('off')
plt.imshow(wordcloud, interpolation = 'bilinear')


# In[22]:


# making a new column day
df_train['day'] = df_train['date'].dt.day

# making a new column month
df_train['month'] = df_train['date'].dt.month

# making a new column year
df_train['year'] = df_train['date'].dt.year

# making a new column week
df_train['week'] = df_train['date'].dt.week

# checking the new columns
df_train.columns


# In[23]:


# checking which days are most busisiest for the shops

plt.rcParams['figure.figsize'] = (18, 8)
sns.countplot(df_train['day'])
plt.title('The most busiest days for the shops', fontsize = 30)
plt.xlabel('Days', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)

plt.show()


# In[24]:


# checking which months are most busisiest for the shops

plt.rcParams['figure.figsize'] = (18, 8)
sns.countplot(df_train['month'], palette = 'dark')
plt.title('The most busiest months for the shops', fontsize = 30)
plt.xlabel('Months', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)

plt.show()


# In[25]:


# checking which years are most busisiest for the shops

plt.rcParams['figure.figsize'] = (18, 8)
sns.countplot(df_train['year'], palette = 'colorblind')
plt.title('The most busiest years for the shops', fontsize = 30)
plt.xlabel('Year', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)

plt.show()


# In[26]:


# checking the columns of the train data

df_train.columns


# In[27]:


# feature engineering

df_train['revenue'] = df_train['item_price'] * df_train['item_cnt_day']

sns.distplot(df_train['revenue'], color = 'blue')
plt.title('Distribution of Revenue', fontsize = 30)
plt.xlabel('Range of Revenue', fontsize = 15)
plt.ylabel('Revenue')
plt.show()


# In[28]:


df_train.dtypes


# In[29]:


# plotting a box plot for itemprice and item-cnt-day

plt.rcParams['figure.figsize'] = (18, 9)
sns.violinplot(x = df_train['day'], y = df_train['revenue'])
plt.title('Box Plot for Days v/s Revenue', fontsize = 30)
plt.xlabel('Days', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.show()


# In[30]:


# plotting a box plot for itemprice and item-cnt-day

plt.rcParams['figure.figsize'] = (18, 9)
sns.boxplot(x = df_train['month'], y = df_train['revenue'])
plt.title('Box Plot for Days v/s Revenue', fontsize = 30)
plt.xlabel('Months', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.show()


# In[31]:


# plotting a box plot for itemprice and item-cnt-day

plt.rcParams['figure.figsize'] = (18, 9)
sns.boxplot(x = df_train['year'], y = df_train['revenue'])
plt.title('Box Plot for Days v/s Revenue', fontsize = 30)
plt.xlabel('Years', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.show()


# In[32]:


# converting the data into monthly sales data

# making a dataset with only monthly sales data
data = df_train.groupby([df_train['date'].apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).sum().reset_index()

# specifying the important attributes which we want to add to the data
data = data[['date','item_id','shop_id','item_cnt_day']]

# at last we can select the specific attributes from the dataset which are important 
data = data.pivot_table(index=['item_id','shop_id'], columns = 'date', values = 'item_cnt_day', fill_value = 0).reset_index()

# looking at the newly prepared datset
data.shape


# In[33]:


# let's merge the monthly sales data prepared to the test data set

test = pd.merge(df_test, data, on = ['item_id', 'shop_id'], how = 'left')

# filling the empty values found in the dataset
test.fillna(0, inplace = True)

# checking the dataset
test.head()


# In[34]:


# now let's create the actual training data

x_train = test.drop(['2015-10', 'item_id', 'shop_id'], axis = 1)
y_train = test['2015-10']

# deleting the first column so that it can predict the future sales data
x_test = test.drop(['2013-01', 'item_id', 'shop_id'], axis = 1)

# checking the shapes of the datasets
print("Shape of x_train :", x_train.shape)
print("Shape of x_test :", x_test.shape)
print("Shape of y_test :", y_train.shape)


# In[35]:


# let's check the x_train dataset

x_train.head()


# In[36]:


# let's check the x_test data

x_test.head()


# In[37]:


# splitting the data into train and valid dataset

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.25, random_state = 3)

# checking the shapes
print("Shape of x_train :", x_train.shape)
print("Shape of x_test :", x_test.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of y_test :", y_test.shape)


# In[38]:


from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import r2_score


# # Linear Regression

# In[39]:


lin_reg=LinearRegression()


# In[40]:


lin_reg.fit(x_train, y_train)
y_pred = lin_reg.predict(x_test)
r2_score(y_test,y_pred)


# # Lasso Regression

# In[41]:


las_reg=Lasso(alpha=.01)
las_reg.fit(x_train, y_train)
y_pred = las_reg.predict(x_test)
r2_score(y_test,y_pred)


# # Ridge Regression

# In[42]:


ridge_reg=Ridge(alpha=3)
ridge_reg.fit(x_train, y_train)
y_pred = ridge_reg.predict(x_test)
r2_score(y_test,y_pred)


# # LGBM Regressor

# In[43]:


from lightgbm import LGBMRegressor

model_lgb = LGBMRegressor( n_estimators=200,
                           learning_rate=0.03,
                           num_leaves=32,
                           colsample_bytree=0.9497036,
                           subsample=0.8715623,
                           max_depth=8,
                           reg_alpha=0.04,
                           reg_lambda=0.073,
                           min_split_gain=0.0222415,
                           min_child_weight=40)
model_lgb.fit(x_train, y_train)

y_pred_lgb = model_lgb.predict(x_test)


# In[44]:


r2_score(y_test,y_pred_lgb)


# In[45]:


# Get the test set predictions and clip values to the specified range
y_pred_lgb = model_lgb.predict(x_test).clip(0., 20.)
r2_score(y_test,y_pred_lgb)


# In[46]:


# Create the submission file and submit
preds = pd.DataFrame(y_pred_lgb, columns=['item_cnt_month'])
preds.to_csv('submission.csv',index_label='ID')


# # Thanks You <(^_^)>

# In[ ]:




