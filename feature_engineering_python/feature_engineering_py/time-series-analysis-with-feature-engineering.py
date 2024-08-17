#!/usr/bin/env python
# coding: utf-8

# ### Please help by upvoting this kernel if you feel useful and share your suggestions for any improvement. It will be very motivating for me. 
# 

# 
# **Problem Statement : **
# 
# You are provided with daily historical sales data. The task is to forecast the total amount of products sold in every shop for the test set. Note that the list of shops and products slightly changes every month. Creating a robust model that can handle such situations is part of the challenge.

# **Step 1:** Define the problem and expected output. Break down the problem into simple steps.
# End Goal ->  Forcast total amount of products sold in every shop for the next month

# **Step 2:** Load the data. 
# Here I'm using library pandas.

# In[1]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

path = "/kaggle/input/competitive-data-science-predict-future-sales/"

items = pd.read_csv(path+'/items.csv')
item_cats = pd.read_csv(path+'/item_categories.csv')
shops = pd.read_csv(path+'/shops.csv')
sales = pd.read_csv(path+'/sales_train.csv')
test = pd.read_csv(path+'/test.csv')
submission = pd.read_csv(path+'/sample_submission.csv')

print("Data set loaded successfully.")


# ****Step 2:** Visualize data. First try to visualize some random samples extracted from the data. I'm using different methods which we can use to visualize data in tabular way.

# In[2]:


print("Items")
print(items.head(2))
print("\nItem Catagerios")
print(item_cats.tail(2))
print("\nShops")
print(shops.sample(n=2))
print("\nTraining Data Set")
print(sales.sample(n=3,random_state=1))
print("\nTest Data Set")
print(test.sample(n=3,random_state=1))


# After seeing this dataset we can catagerioze this data to meta data and effective data.
# So, shop_names and item names, we don't care much. We can have a shop and item combined id and the
#  sales data for the analyze further.
# 
# Final goal to predict sales, so we can ignore names of the products. we are interested in item count in a date time series. And price also can be a factor for the sales.
# 
# So, Try to plot some data which is relavant. 
# Before plotting anything it is better to get an idea about the boundaries of the data set.
# ]

# As we can see, simple way to address this is to use sales data and try to group and summarize those.
# For the conveniet purpose we will split date column to year and month

# In[3]:


from datetime import datetime
sales['year'] = pd.to_datetime(sales['date']).dt.strftime('%Y')
sales['month'] = sales.date.apply(lambda x: datetime.strptime(x,'%d.%m.%Y').strftime('%m')) #another way for same thing

sales.head(2)


# Let's try to plot sales for every year, to understand about seosaonal data

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#will make your plot outputs appear and be stored within the notebook.
get_ipython().run_line_magic('matplotlib', 'inline')

grouped = pd.DataFrame(sales.groupby(['year','month'])['item_cnt_day'].sum().reset_index())
sns.pointplot(x='month', y='item_cnt_day', hue='year', data=grouped)


# In[5]:


#Price
grouped_price = pd.DataFrame(sales.groupby(['year','month'])['item_price'].mean().reset_index())
sns.pointplot(x='month', y='item_price', hue='year', data=grouped_price)


# By seeing this graph we can see that
# 1. last two months of the year having more sales.
# 2. 2015, we are expecting more sales.
# 
# Let's try to draw Total sales along with the linear month period time.

# In[6]:


ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.astype('float')
plt.figure(figsize=(16,8))
plt.title('Total Sales of the whole time period')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts);


# Check the distribution, for detectiting outliers

# In[7]:


sns.jointplot(x="item_cnt_day", y="item_price", data=sales, height=8)
plt.show()


# In[8]:


sales.item_cnt_day.hist(bins=100)
sales.item_cnt_day.describe()


# As we can see, "item_cnt_day" > 125 and < 0, "item_price" >= 75000  we can treat as outliers,
# In data cleaing stage we will remove those items.

# **Step 3: Data Cleaning**
# 
# Filter incorrect data. Eg:  
# 1. Item price is equal to 0
# 2. data not in the test set given
# 3. Remove outliers

# In[9]:


print('Data set size before remove item price 0 cleaning:', sales.shape)
sales = sales.query('item_price > 0')
print('Data set size after remove item price 0 cleaning:', sales.shape)


# In[10]:


print('Data set size before filter valid:', sales.shape)
# Only shops that exist in test set.
sales = sales[sales['shop_id'].isin(test['shop_id'].unique())]
# Only items that exist in test set.
sales = sales[sales['item_id'].isin(test['item_id'].unique())]
print('Data set size after filter valid:', sales.shape)


# In[11]:


print('Data set size before remove outliers:', sales.shape)
sales = sales.query('item_cnt_day >= 0 and item_cnt_day <= 125 and item_price < 75000')
print('Data set size after remove outliers:', sales.shape)


# In[12]:


#After cleaning plot
sns.jointplot(x="item_cnt_day", y="item_price", data=sales, height=8)
plt.show()

cleaned = pd.DataFrame(sales.groupby(['year','month'])['item_cnt_day'].sum().reset_index())
sns.pointplot(x='month', y='item_cnt_day', hue='year', data=cleaned)


# **Step 4:**  Data preprocessing. Identify features. This means, selecting only needed features and create the proper dataset for the processing.
# We need to find out what are the features that will affect the sales
# 1. Price
# 2. Month
# 3. Year
# 4. Item catagory
# 
# Based on above features, sales can be vary. So, we will keep only the interested columns and drop others.

# In[13]:


# Aggregate to monthly level the sales
monthly_sales=sales.groupby(["date_block_num","shop_id","item_id"])[
    "date_block_num","date","item_price","item_cnt_day"].agg({"date_block_num":'mean',"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})

monthly_sales.head(5)


# ### Train the dataset
# 
# We will use LSTM(Long Short Term Memory) algorithum to model this time series data.
# LSTM model will learn a function that maps a sequence of past observations as input to an output observation.
# 
# For this approach, we need to prapare our data set with input and output sequence.
# 
# Eg:  Let say we have monthly avarage sales as,  
# 
# [10, 20, 30, 40, 50, 60, 70, 80, 90]
# 
# We can divide the sequence into multiple input/output patterns called samples, where three time steps are used as input and one time step is used as output for the one-step prediction that is being learned.
# 
#             X		 y
#     10, 20, 30		40
#     20, 30, 40		50
#     30, 40, 50		60

# Our 'date_block_num' column will be the sequnce index, sales will be the value. 

# In[14]:


sales_data_flat = monthly_sales.item_cnt_day.apply(list).reset_index()
#Keep only the test data of valid
sales_data_flat = pd.merge(test,sales_data_flat,on = ['item_id','shop_id'],how = 'left')
#fill na with 0
sales_data_flat.fillna(0,inplace = True)
sales_data_flat.drop(['shop_id','item_id'],inplace = True, axis = 1)
sales_data_flat.head(20)


# In[15]:


#We will create pivot table.
# Rows = each shop+item code
# Columns will be out time sequence
pivoted_sales = sales_data_flat.pivot_table(index='ID', columns='date_block_num',fill_value = 0,aggfunc='sum' )
pivoted_sales.head(20)


# **Step 5** : Spilit training,validation and test dataset.

# In[16]:


# X we will keep all columns execpt the last one 
X_train = np.expand_dims(pivoted_sales.values[:,:-1],axis = 2)
# the last column is our prediction
y_train = pivoted_sales.values[:,-1:]

# for test we keep all the columns execpt the first one
X_test = np.expand_dims(pivoted_sales.values[:,1:],axis = 2)

# lets have a look on the shape 
print(X_train.shape,y_train.shape,X_test.shape)


# In[17]:


from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from keras.models import load_model, Model

# our defining sales model 
sales_model = Sequential()
sales_model.add(LSTM(units = 64,input_shape = (33,1)))
#sales_model.add(LSTM(units = 64,activation='relu'))
sales_model.add(Dropout(0.5))
sales_model.add(Dense(1))

sales_model.compile(loss = 'mse',optimizer = 'adam', metrics = ['mean_squared_error'])
sales_model.summary()


# In[18]:


sales_model.fit(X_train,y_train,batch_size = 4096,epochs = 10)


# In[19]:


submission_output = sales_model.predict(X_test)
# creating dataframe with required columns 
submission = pd.DataFrame({'ID':test['ID'],'item_cnt_month':submission_output.ravel()})
# creating csv file from dataframe
#submission.to_csv('submission.csv',index = False)
submission.to_csv('submission_stacked.csv',index = False)
submission.head()


# 
# Expecting your feedback.
# 
# Thank you.
