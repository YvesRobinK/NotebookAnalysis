#!/usr/bin/env python
# coding: utf-8

# ![Dream-House-PNG-Free-Download-1.png](attachment:c42d845d-27e3-4c58-8ca0-c695095d81cb.png)

# # Introduction

# The project aims to build a model for housing price prediction using provided Dataset. The model should learn data and be able to predict appropriate house prices. And we use regression techniques and Features engineering using multiple libraries like (Sklearn, Pandas, NumPy, Matplotlib, and Seaborn).

# # Why House Price Prediction is Important
# People are expected to help those who plan to buy a house so they can know the price range in the future, then they can plan their finance well. And also, house price predictions are beneficial for property investors to know the trend of housing prices in a certain location.

# ![predict.jpg](attachment:e9033c31-8533-4188-88a1-cd70f53cad1e.jpg)

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


# **Importing Modules**

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plot # graphically representation
import seaborn as sns # graphically representation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# Creating data files

# In[3]:


test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
sub_file = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')


# # Exploring The Data

# In[4]:


#Data Shape
train_data.shape


# In[5]:


# Data Types
train_data.dtypes


# In[6]:


# Data Info
train_data.info


# In[7]:


# Columns
train_data.columns


# In[8]:


# fisrt 10 Records
train_data.head(10)


# In[9]:


# last 5 Records
train_data.tail()


# # Data (EDA and Preprocessing)

# In[10]:


fig = plot.figure(figsize=(12,10))
plot.subplot(321)
sns.scatterplot(data=train_data, x='GarageArea', y="SalePrice")
plot.subplot(322)
sns.scatterplot(data=train_data, x='YearBuilt', y="SalePrice")
plot.subplot(323)
sns.scatterplot(data=train_data, x='WoodDeckSF', y="SalePrice")
plot.subplot(324)
sns.scatterplot(data=train_data, x='OverallQual', y="SalePrice")
plot.subplot(325)
sns.scatterplot(data=train_data, x='BsmtUnfSF', y="SalePrice")
plot.subplot(326)
sns.scatterplot(data=train_data, x='TotalBsmtSF', y="SalePrice")


# In[11]:


y = train_data.SalePrice


# In[12]:


plot.figure(figsize=(7,7))
sns.boxplot(data=y)


# In[13]:


y.describe()


# In[14]:


data = train_data
y = y.drop_duplicates()


# In[15]:


data.dtypes.value_counts().plot.pie(autopct='%0.2f%%')


# In[16]:


display(plot.figure(figsize=(15,6)))
display(sns.countplot(data.isnull().sum()))


# In[17]:


display(plot.figure(figsize=(15,6)))
display(sns.heatmap(data.isnull()))


# In[18]:


# data.loc[[data.columns.dtypes == type(float)]]
display(data.select_dtypes(include=(float)))


# # Feature engineering

# In[19]:


obj_col =  train_data.select_dtypes(include=['object']).columns
int_col =  train_data.select_dtypes(include=['int']).columns

print("Categorical Data Columns", obj_col, "\n", "Integar Data Columns", int_col)


# In[20]:


train_data['LotFrontage'] = int(train_data['LotFrontage'].mean())
train_data['MasVnrArea'] = int(train_data['MasVnrArea'].mean())
train_data['BsmtFinSF1'] = int(train_data['BsmtFinSF1'].mean())
train_data['BsmtFinSF2'] = int(train_data['BsmtFinSF2'].mean())
train_data['BsmtUnfSF'] = int(train_data['BsmtUnfSF'].mean())
train_data['TotalBsmtSF'] = int(train_data['TotalBsmtSF'].mean())
train_data['BsmtHalfBath'] = int(train_data['BsmtHalfBath'].mean())
train_data['GarageYrBlt'] = int(train_data['GarageYrBlt'].mean())
train_data['GarageCars'] = int(train_data['GarageCars'].mean())
train_data['GarageArea'] = int(train_data['GarageArea'].mean())


# In[21]:


train_data.drop(['Alley'], inplace=True, axis=1)
train_data.drop(['MiscFeature'], inplace=True, axis=1)
train_data.drop(['PoolQC'], inplace=True, axis=1)
train_data.drop(['Fence'], inplace=True, axis=1)


# In[22]:


test_data.drop(['Alley'], inplace=True, axis=1)
test_data.drop(['MiscFeature'], inplace=True, axis=1)
test_data.drop(['PoolQC'], inplace=True, axis=1)
test_data.drop(['Fence'], inplace=True, axis=1)


# In[23]:


train_data.hist(figsize=(15, 13))
plot.show()


# In[24]:


sns.jointplot(data=train_data, x='LotArea', y='SalePrice')
sns.jointplot(data=train_data, x='YearBuilt', y='SalePrice')
sns.jointplot(data=train_data, x='YrSold', y='SalePrice')
sns.jointplot(data=train_data, x='PoolArea', y='SalePrice')


# In[25]:


sns.pairplot(train_data)


# In[26]:


null_p  = train_data.isnull().sum()/data.shape[0]*100
null_p  = test_data.isnull().sum()/data.shape[0]*100


# In[27]:


cread_t_col = null_p[null_p > 30].keys()
test_data = test_data.drop(cread_t_col, "columns")


# In[28]:


create_d_col = null_p[null_p > 30].keys()
train_data = train_data.drop(create_d_col, "columns")


# In[29]:


train_data['MSZoning'] = train_data['MSZoning'].mode()[0]
train_data['MSSubClass'] = train_data['MSSubClass'].mode()[0]
train_data['BsmtCond'] = train_data['BsmtCond'].mode()[0]
train_data['BsmtQual'] = train_data['BsmtQual'].mode()[0]
train_data['GarageType'] = train_data['GarageType'].mode()[0]
train_data['BsmtCond'] = train_data['BsmtCond'].mode()[0]
train_data['BsmtExposure'] = train_data['BsmtExposure'].mode()[0]
train_data['GarageArea'] = train_data['GarageArea'].mode()[0]
train_data['BsmtFinType2'] = train_data['BsmtFinType2'].mode()[0]
train_data['GarageYrBlt'] = train_data['GarageYrBlt'].mode()[0]
train_data['GarageCond'] = train_data['GarageYrBlt'].mode()[0]
train_data['GarageFinish'] = train_data['GarageYrBlt'].mode()[0]
train_data['Exterior2nd'] = train_data['GarageYrBlt'].mode()[0]


# In[30]:


test_data.dropna(axis=0, how="any")


# In[31]:


plot.figure(figsize=(15, 5))
display(sns.heatmap(train_data.isnull()))


# In[32]:


#a = data.columns.tolist()


# In[33]:


train_data.dtypes.value_counts().plot.pie(autopct='%0.2f%%')


# In[34]:


plot.figure(figsize=(25, 28))
sns.heatmap(train_data.corr(), cmap = "coolwarm",annot=True, linewidth=2)


# In[35]:


obj_c = train_data.select_dtypes('object')
obj_c


# In[36]:


'''plot.figure(figsize=(10,8))
plot.plot(y)'''



# In[37]:


train_data.drop("SalePrice", axis=1)
y = train_data["SalePrice"]


# In[38]:


x = train_data.drop('SalePrice', axis=1)


# In[39]:


print(train_data.shape, y.shape)


# In[40]:


train_data


# # Label Encoding

# In[41]:


train_data


# In[42]:


d_train_data = train_data.apply(LabelEncoder().fit_transform)


# In[43]:


d_test_data = test_data.apply(LabelEncoder().fit_transform)


# In[44]:


d_train_data


# In[45]:


# train_data.drop("SalePrice", axis = 1, inplace = True)


# # Train Data

# In[46]:


x_train, x_test, y_train, y_test = train_test_split(d_train_data, y, test_size=0.2)


# # RandomForestRegressor

# In[47]:


rmr = RandomForestRegressor()


# In[48]:


rmr.fit(x_train, y_train)


# In[ ]:





# In[49]:


predy = rmr.predict(x_test)


# In[50]:


predy


# In[51]:


rmr.score(x_test, y_test)


# In[52]:


rmr.score(x_test, y_test)


# In[53]:


x.columns


# In[54]:


x.columns


# In[55]:


train_data.isnull().sum()


# In[56]:


test_data.isnull().sum()


# In[57]:


predy.shape


# In[58]:


# import os

# os.remove('submission.csv')


# In[59]:


# output = pd.DataFrame({
# "Id": x["Id"][:438],
# "SalePrice": predy
# })

# output.to_csv('submission.csv', index=False)
# print("Your submission was successfully saved!")

