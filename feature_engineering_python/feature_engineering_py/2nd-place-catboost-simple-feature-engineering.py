#!/usr/bin/env python
# coding: utf-8

# In[1]:


import optuna
import pandas as pd, numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# https://www.kaggle.com/code/alvinleenh/gridworld-sales-eda-ctb-prediction
train = pd.read_csv('/kaggle/input/predicting-sales-quantity-in-our-dynamic-gridworld/train.csv')
sup = pd.read_csv('/kaggle/input/predicting-sales-quantity-in-our-dynamic-gridworld/supplemental_cities.csv')
test = pd.read_csv('/kaggle/input/predicting-sales-quantity-in-our-dynamic-gridworld/test.csv')
submission = pd.read_csv('/kaggle/input/predicting-sales-quantity-in-our-dynamic-gridworld/sample_submission.csv')
train = pd.merge(train, sup, on='city_id')
test = pd.merge(test, sup, on='city_id')


# In[3]:


# Code block from @matasferrn

def extract_coord(df):
    df['x'] = df['city_id'].apply(lambda x: int(x.split('/')[0]))
    df['y'] = df['city_id'].apply(lambda x: int(x.split('/')[1]))
    df['city_nr'] = df['city_id'].apply(lambda x: int(x.split('/')[2]))

from sklearn.impute import KNNImputer

extract_coord(sup)

# Set the features
X = sup.drop('city_id', axis=1)

# Create model
imputer = KNNImputer(n_neighbors=2)

# Fit and transform
sup_cities_imputed = pd.DataFrame(imputer.fit_transform(X),columns = X.columns, index=X.index)
sup = sup_cities_imputed.merge(sup['city_id'], left_index=True, right_index=True)

sup['cities_in_coord'] = sup.groupby(['x', 'y'])['city_nr'].transform('count')

train = pd.read_csv('/kaggle/input/predicting-sales-quantity-in-our-dynamic-gridworld/train.csv')
test = pd.read_csv('/kaggle/input/predicting-sales-quantity-in-our-dynamic-gridworld/test.csv')
train = pd.merge(train, sup, on='city_id')
test = pd.merge(test, sup, on='city_id')

# Code block from @matasferrn - end


# In[4]:


train['type']=0
test['type']=1
all_data = pd.concat([train,test],axis=0)


# In[5]:


all_data.city_id, _ = all_data['city_id'].factorize()


# # FE

# # city

# In[6]:


features = ['price', 'ad_level', 'population', 'education_level', 'median_income']


# In[7]:


all_data[f'sum_population_city'] = all_data.groupby(['city_id'])['population'].transform('sum')


# In[8]:


for feature in features:
    all_data[f'mean_{feature}_c'] = all_data.groupby('city_id')[feature].transform('mean')
    all_data[f'min_{feature}_c'] = all_data.groupby('city_id')[feature].transform('min')
    all_data[f'max_{feature}_c'] = all_data.groupby('city_id')[feature].transform('max')


# In[9]:


all_data['num_stores_c'] = all_data.groupby('city_id')['store_id'].transform('nunique')


# # loc

# In[10]:


features = ['price', 'ad_level', 'population', 'education_level', 'median_income']


# In[11]:


all_data[f'sum_population_loc'] = all_data.groupby(['x','y'])['population'].transform('sum')


# In[12]:


for feature in features:
    all_data[f'mean_{feature}_loc'] = all_data.groupby(['x','y'])[feature].transform('mean')
    all_data[f'min_{feature}_loc'] = all_data.groupby(['x','y'])[feature].transform('min')
    all_data[f'max_{feature}_loc'] = all_data.groupby(['x','y'])[feature].transform('max')


# In[13]:


all_data['store_id'] = range(36059)


# In[14]:


all_data['num_cities_loc'] = all_data.groupby(['x', 'y'])['city_id'].transform('nunique')
all_data['num_stores_loc'] = all_data.groupby(['x', 'y'])['store_id'].transform('nunique')


# In[15]:


all_data.drop(['id','city_id','store_id', 'city_nr'], axis=1, inplace=True)
train = all_data[all_data.type==0]
train = train.drop(columns=['type'])
train.reset_index(inplace=True,drop=True)
test = all_data[all_data.type==1]
test = test.drop(columns=['type'])
test.reset_index(inplace=True,drop=True)
test.drop('quantity', axis=1, inplace=True)


# In[16]:


X = train.copy()
y = X.pop('quantity')


# # sub

# In[17]:


catboost_model = CatBoostRegressor(verbose=0)


# In[18]:


cross_val_score(catboost_model, X, y, cv=5, scoring="neg_root_mean_squared_error")


# In[19]:


catboost_model.fit(X,y)


# In[20]:


pred = catboost_model.predict(test)


# In[21]:


submission['quantity'] = pred
submission.to_csv('submission.csv',index=False)
print("Your submission was successfully saved!")
submission.head()

