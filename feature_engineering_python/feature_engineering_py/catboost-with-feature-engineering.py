#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries
# 

# In[1]:


get_ipython().system('pip install catboost')


# In[2]:


get_ipython().system('pip install holidays')


# In[3]:


import catboost
from catboost import CatBoostRegressor
import holidays
import numpy as np
import pandas as pd
import seaborn as sns
import re
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error , r2_score , make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor ,GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
import requests
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score ,KFold
from sklearn.metrics import make_scorer, mean_squared_error
from math import sqrt
from sklearn.svm import SVR
import sklearn.metrics
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.io as pio
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer


# # Load & Discover The Data

# In[4]:


data = pd.read_csv('//kaggle/input/prediction-of-tourist-arrivals/train_df.csv')
test = pd.read_csv('/kaggle/input/prediction-of-tourist-arrivals/test_df.csv')


# In[5]:


data.head()


# In[6]:


data.info()


# In[7]:


data.shape


# In[8]:


data.isnull().sum()


# In[9]:


test.isnull().sum()


# # EDA

# In[10]:


data['tourism_index'].hist()


# In[11]:


data['weather_index'].hist()


# In[12]:


sns.countplot(data=data , x='spot_facility')


# In[13]:


sns.countplot(data=data , x='area')


# In[14]:


sns.countplot(data=data , x='type')


# In[15]:


sns.countplot(data=data , x='event')


# In[16]:


sns.countplot(data=data , x='info')


# # Preprocessing

# In[17]:


def handling_event(df):
  replacement_value = 'Other'
  df.loc[df['event'] != 'A', 'event'] = replacement_value
  return df


# In[18]:


def handling_info(df):
  replacement_value = 'Other'
  df.loc[df['info'] != 'A', 'info'] = replacement_value
  return df


# In[19]:


def is_japan_holiday(date,year):
    jp_holidays = holidays.Japan(years=year)
    return date in jp_holidays


# In[20]:


def handling_date(df):
  df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d', errors='coerce')
  df['date'] = pd.to_datetime(df['date'])
    
  df['month'] = df['date'].dt.month

  
  df['day'] = df['date'].dt.day

  
  df['weekday'] = df['date'].dt.weekday

  df['year'] = df['date'].dt.year

    
  weekday_to_day_name = {
      0: "Monday",
      1: "Tuesday",
      2: "Wednesday",
      3: "Thursday",
      4: "Friday",
      5: "Saturday",
      6: "Sunday"
  }

  
  df['day_name'] = df['weekday'].map(weekday_to_day_name)
  df['is_weekend'] = (df['weekday'] >= 5)

  month_to_season = {
      3: 'Spring',
      4: 'Spring',
      5: 'Spring',
      6: 'Summer',
      7: 'Summer',
      8: 'Summer',
      9: 'Autumn',
      10: 'Autumn',
      11: 'Autumn',
      12: 'Winter',
      1: 'Winter',
      2: 'Winter'
  }


  df['season'] = df['month'].map(month_to_season)

  return df


# In[21]:


def preprocessing(df):
  df = handling_event(df)
  df = handling_info(df)
  df = handling_date(df)
  df['is_holiday'] = df.apply(lambda row: is_japan_holiday(row['date'],row['year']), axis=1)
  df['info'] = df['info'].astype(str)
  df['event'] = df['event'].astype(str) 
  df = df.drop(['id','date'],axis=1)
  return df


# In[22]:


data = preprocessing(data)


# In[23]:


data.info()


# # Data Pipeline

# In[24]:


X = data.drop(['tourist_arrivals'],axis=1)
y = data['tourist_arrivals']


# In[25]:


numerical_columns = X.select_dtypes(include=['number']).columns.tolist()
categorical_columns = X.select_dtypes(exclude=['number']).columns.tolist()


# In[26]:


numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor)
])


# In[27]:


X = pipeline.fit_transform(X)


# In[28]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)


# # Model Training

# In[29]:


catboost_pool = catboost.Pool(X_train, y_train)
cat_boost = CatBoostRegressor(task_type="GPU", devices='0',verbose=False)


# In[30]:


cat_boost.fit(catboost_pool)


# In[31]:


y_pred = cat_boost.predict(X_valid)
rmse = sqrt(mean_squared_error(y_valid, y_pred))
print("RMSE:", rmse)


# # Testing Data

# In[32]:


Id= test['id']
test = preprocessing(test)
test = pipeline.transform(test)
pred = cat_boost.predict(test)
pred = np.squeeze(pred)
final={'id': Id, 'tourist_arrivals':pred }


# In[33]:


df1 = pd.DataFrame(data = final)
df1.to_csv('new.csv',index=False)


# In[ ]:




