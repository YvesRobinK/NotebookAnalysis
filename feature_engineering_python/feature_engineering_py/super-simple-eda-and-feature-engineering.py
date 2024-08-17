#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#SIMPLE EDA, FEATURE ENGINEERING AND XGBOOST EXAMPLE


# In[1]:


import pandas as pd
import numpy as np


# In[13]:


train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")


# In[3]:


train.head()
# LET'S TAKE A LOOK AT THESE FIRST 5 COLUMNS


# In[4]:


train.describe()
# A QUICK SUMMARY OF THE NUMERICAL VALUES IN THE DATASET


# In[5]:


train.shape
# WE HAVE 1460 ROWS AND 81 COLS


# In[6]:


train.columns
# OUR COLUMNS


# In[7]:


train.info()
#INFORMATION ABOUT WHAT IS THE DATA SITUATION, WHAT IS MISSING WHAT IS NOT


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(train['SalePrice'])
plt.show()

# A QUICK LOOK AT THE SALEPRICE DATA


# In[9]:


corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()
# A BIG CORRELATION MATRIX OF OUR LABELS.


# In[10]:


# let's see the missing values
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))
# We have a bunch of missing values here too.


# In[14]:


#drop the columns with more than 15% missing values
train = train.drop((missing_data[missing_data['Percent'] > 0.03]).index,1)
train.isnull().sum().max() # just checking that there's no missing data missing...
train.columns


# In[15]:


train.select_dtypes(include=['object']).columns
# see what values are categorical


# In[16]:


#convert categorical variables to numerical variables
from sklearn.preprocessing import LabelEncoder

categorical_values = train.select_dtypes(include=['object']).columns
for i in categorical_values:
    lbl = LabelEncoder() 
    lbl.fit(list(train[i].values)) 
    train[i] = lbl.transform(list(train[i].values))


# In[18]:


# Finding higly correlated values to the label saleprice since we are aiming to find it
corrmat = train.corr()
k = 12 #number of variables for heatmap
high_corr_values = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
high_corr_values = high_corr_values.drop('SalePrice')


# In[19]:


high_corr_values


# In[21]:


# Fitting an xgboost model to predict SalePrice using the highly correlated variables
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


# In[22]:


X = train[high_corr_values]
y = train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

my_model = XGBRegressor()
my_model.fit(X_train, y_train)
predictions = my_model.predict(X_test)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, y_test)))


# In[24]:


import gc


# In[25]:


real_test = test
del test
gc.collect()
# not needed, 


# In[27]:


real_predictions = my_model.predict(real_test[high_corr_values])
print(len(real_predictions))
print(len(real_test.Id))
# Real predictions on the test data


# In[29]:


output = pd.DataFrame({'Id': real_test.Id, 'SalePrice': real_predictions})
# save the data frame to a csv file
output.to_csv('submission.csv', index=False)

