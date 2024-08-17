#!/usr/bin/env python
# coding: utf-8

# # Introduction 
# 
# In this notebook, I have tried to illustrate the best way (in this case) to feature engineering and handle missing values. I have shown some handy tricks one can use to better their score in Kaggle competitions.
# 
# 
# 

# # Imports 

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('max_rows',100)
pd.set_option('max_columns',100)
import seaborn as sns
import matplotlib.pyplot as  plt
import scipy
from sklearn.preprocessing import StandardScaler

# from pycaret.classification import setup, compare_models

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# 

# # combining train and test 
# 
# We have been given train and test data by the competition, the standard way to do this is taking the train data and spliting it for test or using KFold. However in this setting of competition we can use the extra bit of information provided to us by the test set to impute and scale the data. This will boost the accuracy

# In[2]:


df_train = pd.read_csv('../input/spaceship-titanic/train.csv')
df_test = pd.read_csv('../input/spaceship-titanic/test.csv')
sample_sub = pd.read_csv('../input/spaceship-titanic/sample_submission.csv')
target = df_train['Transported']

df_train.drop(['Transported'], axis= 1, inplace = True)
cdata = pd.concat([df_train, df_test],axis = 0,ignore_index=True)
cdata.drop(['Name'], axis= 1, inplace = True)



# # Split to Create more Features
# We can split the seeming unimportant string type feature to give our model a better chance to understand what is going on. However, we need to filter these features and check their impact.

# In[3]:


cdata[['pass_grp','pass_no']]= cdata['PassengerId'].str.split('_', n = -1, expand = True)
cdata.drop('PassengerId', axis = 1, inplace = True)
cdata[['deck','num','side']]= cdata['Cabin'].str.split('/', n = -1, expand = True)
cdata.drop('Cabin', axis = 1, inplace = True)


# 
# # Handling missing values and Adding new feature
# 
# with trial and error i found that replacing 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck' missing values with zero works best. I guess is these missing values could be there by design. num and pass grp dont add much value to the overall performance and thus can be dropped.
# 

# In[4]:


cdata['Total_cost'] = pd.Series(np.zeros(cdata.shape[0]))
for feat in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
    cdata[feat].fillna(cdata[feat].mean(), inplace = True)
    cdata['Total_cost'] += cdata[feat]
#     cdata3.drop(feat, axis = 1,inplace = True )
cdata.drop(['num', 'pass_grp'],axis = 1, inplace = True)




# # Simple imputing 

# In[5]:


categorical_features = cdata.select_dtypes(object)
num_features = cdata.select_dtypes(np.number)
for feat in categorical_features:
    cdata[feat].fillna(cdata[feat].mode()[0], inplace= True)

cdata['Age'].fillna(int(cdata['Age'].mean()), inplace = True)

