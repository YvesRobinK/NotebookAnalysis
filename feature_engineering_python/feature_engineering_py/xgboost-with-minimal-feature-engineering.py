#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import gc


# In[2]:


#data path
PATH = '/kaggle/input/instacart-market-basket-analysis/'


# In[3]:


#Reading the datasets.
aisles = pd.read_csv(PATH + 'aisles.csv')
products = pd.read_csv(PATH + 'products.csv')
order_products_train = pd.read_csv(PATH + 'order_products__train.csv')
order_products_prior = pd.read_csv(PATH + 'order_products__prior.csv')
departments = pd.read_csv(PATH + 'departments.csv')
orders = pd.read_csv(PATH + 'orders.csv')


# In[4]:


#creating a datframe that will contain only prior information
op = pd.merge(orders, order_products_prior, on='order_id', how='inner')
op.head()


# # Creating features related to the users. i.e using user_id

# In[5]:


#Total number of orders placed by each users
users = op.groupby(by='user_id')['order_number'].aggregate('max').to_frame('u_num_of_orders').reset_index()
users.head()


# In[6]:


#average number of products bought by the user in each purchase.

#1. First getting the total number of products in each order.
total_prd_per_order = op.groupby(by=['user_id', 'order_id'])['product_id'].aggregate('count').to_frame('total_products_per_order').reset_index()
total_prd_per_order.head(10)


# In[7]:


#2. Getting the average products purchased by each user
avg_products = total_prd_per_order.groupby(by=['user_id'])['total_products_per_order'].mean().to_frame('u_avg_prd').reset_index()
avg_products.head()


# In[8]:


#deleting the total_prd_per_order dataframe
del total_prd_per_order


# In[9]:


#dow of most orders placed by each user
from scipy import stats
dow = op.groupby(by='user_id')['order_dow'].agg(lambda x: stats.mode(x)[0]).to_frame('dow_most_orders_u').reset_index()
dow.head()


# In[10]:


#hour of day when most orders placed by each user
from scipy import stats
hod = op.groupby(by='user_id')['order_hour_of_day'].agg(lambda x: stats.mode(x)[0]).to_frame('hod_most_orders_u').reset_index()
hod.head()


# In[11]:


#merging the user created features.

#1. merging avg_products with users
users = users.merge(avg_products, on='user_id', how='left')
#deleting avg_products
del avg_products
users.head()


# In[12]:


#2. merging dow with users.
users = users.merge(dow, on='user_id', how='left')
#deleting dow
del dow
users.head()


# In[13]:


#3. merging hod with users
#2. merging dow with users.
users = users.merge(hod, on='user_id', how='left')
#deleting dow
del hod
users.head()


# # Creating features related to the products using product_id.

# In[14]:


#number of time a product was purchased.
prd = op.groupby(by='product_id')['order_id'].agg('count').to_frame('prd_count_p').reset_index()
prd.head()


# In[15]:


#products reorder ratio.
reorder_p = op.groupby(by='product_id')['reordered'].agg('mean').to_frame('p_reordered_ratio').reset_index()
reorder_p.head()


# In[16]:


#merging the reorder_p with prd
prd = prd.merge(reorder_p, on='product_id', how='left')
#deleting reorder_p
del reorder_p
prd.head()


# # Creating user-product features.

# In[17]:


#how many times a user bought the same product.
uxp = op.groupby(by=['user_id', 'product_id'])['order_id'].agg('count').to_frame('uxp_times_bought').reset_index()
uxp.head()


# In[18]:


#reorder ratio of the user for each product.
reorder_uxp = op.groupby(by=['user_id', 'product_id'])['reordered'].agg('mean').to_frame('uxp_reordered_ratio').reset_index()
reorder_uxp.head()


# In[19]:


#merging the two dataframes into one
uxp = uxp.merge(reorder_uxp, on=['user_id', 'product_id'], how='left')
#deleting reorder_uxp
del reorder_uxp
uxp.head()


# # Merging all the features into data DF.

# In[20]:


#merging users df into uxp
data = uxp.merge(users, on='user_id', how='left')
data.head()


# In[21]:


#merging products df into data
data = data.merge(prd, on='product_id', how='left')
data.head()


# In[22]:


#deleting unwanted dfs
del [users, prd, uxp]


# # Creating train and test dataset.

# In[23]:


#keeping only the train and test eval set from the orders dataframe.
order_future = orders.loc[((orders.eval_set == 'train') | (orders.eval_set == 'test')), ['user_id', 'eval_set', 'order_id']]
order_future.head()


# In[24]:


#merging the order_future with the data.
data = data.merge(order_future, on='user_id', how='left')
data.head()


# In[25]:


#preparing the train df.
data_train = data[data.eval_set == 'train']
data_train.head()


# In[26]:


#merging the information from the order_proucts_train df into the data_train.
data_train = data_train.merge(order_products_train[['product_id', 'order_id', 'reordered']], on=['product_id', 'order_id'], how='left')
data_train.head()


# In[27]:


#filling the NAN values
data_train.reordered.fillna(0, inplace=True)


# In[28]:


#setting user_id and product_id as index.
data_train = data_train.set_index(['user_id', 'product_id'])

#deleting eval_set, order_id as they are not needed for training.
data_train.drop(['eval_set', 'order_id'], axis=1, inplace=True)


# In[29]:


data_train.head()


# In[30]:


#preparing the test dataset.
data_test = data[data.eval_set == 'test']
data_test.head()


# In[31]:


#deleting unwanted columns
data_test.drop(['eval_set', 'order_id'], axis=1, inplace=True)


# In[32]:


#setting user_id and product_id as index.
data_test = data_test.set_index(['user_id', 'product_id'])


# In[33]:


data_test.head()


# In[34]:


#deleting unwanted df
del [aisles, departments, order_products_prior, order_products_train, orders, order_future, data] #order_future, data


# In[35]:


#resetting index
data_train.reset_index(inplace=True)
data_test.reset_index(inplace=True)


# In[36]:


data_train.head()


# In[37]:


#merging the aisles and department ids to with the train and test data
data_train = data_train.merge(products[['product_id', 'aisle_id']], on='product_id', how='left')
data_test = data_test.merge(products[['product_id', 'aisle_id']], on='product_id', how='left')


# In[38]:


#department
data_train = data_train.merge(products[['product_id', 'department_id']], on='product_id', how='left')
data_test = data_test.merge(products[['product_id', 'department_id']], on='product_id', how='left')


# In[39]:


#setting user_id and product_id as index.
data_test = data_test.set_index(['user_id', 'product_id'])
#setting user_id and product_id as index.
data_train = data_train.set_index(['user_id', 'product_id'])


# # Creating Predictive model

# In[40]:


#Building a XGBoost model.

#importing the package.
import xgboost as xgb

#splitting the train data into training and testing set.
X_train, y_train = data_train.drop('reordered', axis=1), data_train.reordered

#setting boosters parameters
parameters = {
    'eavl_metric' : 'logloss',
    'max_depth' : 5,
    'colsample_bytree' : 0.4,
    'subsample' : 0.8
}

#instantiating the model
xgb_clf = xgb.XGBClassifier(objective='binary:logistic', parameters=parameters, num_boost_round=10)

# TRAIN MODEL
model = xgb_clf.fit(X_train, y_train)

#FEATURE IMPORTANCE - GRAPHICAL
xgb.plot_importance(model)


# In[41]:


#predicting on the testing data
y_pred = xgb_clf.predict(data_test).astype('int')

#setting a threshold.
y_pred = (xgb_clf.predict_proba(data_test)[:, 1] >= 0.21).astype('int')
y_pred[0:10]


# In[42]:


#saving the prediction as a new column in data_test
data_test['prediction'] = y_pred
data_test.head()


# In[43]:


# Reset the index
final = data_test.reset_index()
# Keep only the required columns to create our submission file (for chapter 6)
final = final[['product_id', 'user_id', 'prediction']]

gc.collect()
final.head()


# In[44]:


#Creating a submission file
orders = pd.read_csv(PATH + 'orders.csv')
orders_test = orders.loc[orders.eval_set == 'test', ['user_id', 'order_id']]
orders_test.head()


# In[45]:


#merging our prediction with orders_test
final = final.merge(orders_test, on='user_id', how='left')
final.head()


# In[46]:


#remove user_id column
final = final.drop('user_id', axis=1)


# In[47]:


#convert product_id as integer
final['product_id'] = final.product_id.astype(int)

## Remove all unnecessary objects
del orders
del orders_test
gc.collect()

final.head()


# In[48]:


d = dict()
for row in final.itertuples():
    if row.prediction== 1:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)

for order in final.order_id:
    if order not in d:
        d[order] = 'None'
        
gc.collect()

#We now check how the dictionary were populated (open hidden output)
d


# In[49]:


#Convert the dictionary into a DataFrame
sub = pd.DataFrame.from_dict(d, orient='index')

#Reset index
sub.reset_index(inplace=True)
#Set column names
sub.columns = ['order_id', 'products']

sub.head()


# In[50]:


sub.to_csv('sub.csv', index=False)


# In[ ]:





# In[ ]:





# In[51]:


gc.collect()

