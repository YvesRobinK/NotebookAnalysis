#!/usr/bin/env python
# coding: utf-8

# ## **Table of Contents**
# > 1. [Notebook Imports](#1)
# > 2. [Importing Data](#2)
# > 3. [Data Preprocessing](#3)
# > 4. [Exploratory Data Analysis](#4)
# > 5. [Feature Engineering](#5)
# > 6. [Encoding](#6)
# > 7. [Scaling](#7)
# > 8. [Modelling](#8)
# > 9. [Hyperparameter Optimization](#9)
# >10. [Final Predictions](#10)

# <div style='color: #216969;
#            background-color: #EAF6F6;
#            font-size: 200%;
#            border-radius:15px;
#            text-align:center;
#            font-weight:600;
#            border-style: solid;
#            border-color: dark green;
#            font-family: "Verdana";'>
# Notebook Imports
# <a class="anchor" id="1"></a> 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
sns.set()

from scipy.stats import probplot, boxcox
from scipy.special import inv_boxcox
import pylab

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

import optuna


# <div style='color: #216969;
#            background-color: #EAF6F6;
#            font-size: 200%;
#            border-radius:15px;
#            text-align:center;
#            font-weight:600;
#            border-style: solid;
#            border-color: dark green;
#            font-family: "Verdana";'>
# Importing Data
# <a class="anchor" id="2"></a> 

# In[2]:


train = pd.read_csv('../input/tabular-playground-series-sep-2022/train.csv')
test = pd.read_csv('../input/tabular-playground-series-sep-2022/test.csv')
train.head()


# In[3]:


print(f'Number of rows in training set:', train.shape[0])
print(f'Number of columns in training set:', train.shape[1])


# <div style='color: #216969;
#            background-color: #EAF6F6;
#            font-size: 200%;
#            border-radius:15px;
#            text-align:center;
#            font-weight:600;
#            border-style: solid;
#            border-color: dark green;
#            font-family: "Verdana";'>
# Data preprocessing
# <a class="anchor" id="3"></a> 

# In[4]:


train.nunique()


# #### Row ID won't be any useful in our analysis, therefore I will just drop it from the training set and the test set

# In[5]:


test_ids = test['row_id']
train.drop('row_id', axis=1, inplace=True)
test.drop('row_id', axis=1, inplace=True)


# In[6]:


train.info()


# #### Date column has an object datatype, I will convert it into a datetime object

# In[7]:


train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])


# In[8]:


train['Year'] = train['date'].dt.year
train['Month'] = train['date'].dt.month
train['Day'] = train['date'].dt.day

test['Year'] = test['date'].dt.year
test['Month'] = test['date'].dt.month
test['Day'] = test['date'].dt.day


# In[9]:


# Checking for missing values
train.isna().sum()


# #### There are no missing values in our dataset

# In[10]:


num_sold_country_wise = train.groupby(['country', 'product'])['num_sold'].sum().to_frame().unstack().rename(columns={'num_sold': 'Number of Sales'})
num_sold_country_wise['Total Sales'] = num_sold_country_wise.sum(axis=1)
num_sold_country_wise = num_sold_country_wise.sort_values('Total Sales', ascending=False)
num_sold_country_wise


# #### Belgium had the most number of sales followed by France and Germany. Kaggle for Kids sold the most while Kaggle Recipe book wasn't that popular

# <div style='color: #216969;
#            background-color: #EAF6F6;
#            font-size: 200%;
#            border-radius:15px;
#            text-align:center;
#            font-weight:600;
#            border-style: solid;
#            border-color: dark green;
#            font-family: "Verdana";'>
# Exploratory Data Analysis
# <a class="anchor" id="4"></a> 

# In[11]:


train.groupby('Year').sum()['num_sold']


# In[12]:


fig = px.line(train.groupby('Year').sum()['num_sold'], title='Number of Sales by year', markers=True)
fig.update_xaxes(type='date')
fig.show()


# #### Here we can observe that sales in increased quite a bit in 2020 as compared to 2019

# In[13]:


fig = px.line(train.groupby('Month').sum()['num_sold'], title='Number of Sales by Month', markers=True)
fig.show()


# #### Sales were usually higher during the start and end of the year and were lowest around september

# In[14]:


fig = px.line(train.groupby('Day').sum()['num_sold'], title='Number of Sales by Day', markers=True)
fig.show()


# In[15]:


train_year_month = train.groupby(['Year','Month']).sum()['num_sold']
train_year_month = train_year_month.reset_index().pivot('Year', 'Month', 'num_sold').transpose()
train_year_month


# In[16]:


fig = px.line(np.cumsum(train_year_month), title='Number of Sales Cumulative',markers=True)
fig.show()


# #### 2020 line seems to move away from rest of the lines after the month of June

# In[17]:


fig,ax = plt.subplots(1,2,figsize=(20, 6))
plt.suptitle("Distribution of Number of Sales", fontsize=20)

ax0 = sns.histplot(data = train,x ='num_sold', kde=True, ax=ax[0], bins = 30, color= '#554994')
ax0.set_xlabel('Number of sales', fontsize = 15)

ax1 = sns.violinplot(x = train['num_sold'], ax=ax[1], color='#554994')
ax1.set_xlabel('Number of sales', fontsize = 15)
plt.tight_layout()
plt.show()


# In[18]:


sns.set_context('notebook', font_scale=1.3)
plt.figure(figsize= (10, 6))
probplot(train.num_sold, plot=pylab)
pylab.show()


# #### It is clear from the histogram and the QQ Plot that the distribution of number of Sales is clearly skewed, there are a lot of outliers and we will have to apply a log transformation before modelling

# In[19]:


num_sold_country_wise.head()


# In[20]:


sns.set_context('notebook', font_scale = 1.3)
plt.figure(figsize= (15, 7))
sns.barplot(x=num_sold_country_wise['Number of Sales'].index, y=num_sold_country_wise['Total Sales'], palette= 'winter')
plt.title('Number of Sales by country')
plt.show()


# #### Germany had the most number of Sales while Poland had the least

# In[21]:


train.head()


# In[22]:


train_product = train.groupby('product')['num_sold'].sum().to_frame().reset_index().sort_values('num_sold', ascending=False)

plt.figure(figsize=(15, 7))
sns.barplot(x=train_product['product'], y=train_product['num_sold'], palette='summer')
plt.ylabel('Number of Sales');


# #### Kaggle for kids sold the most while Kaggle Recipe Book sold the least

# In[23]:


train_store = train.groupby('store')['num_sold'].sum().to_frame().reset_index().sort_values('num_sold', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=train_store['store'], y=train_store['num_sold'], palette='viridis')
plt.ylabel('Number of Sales')
plt.show()


# In[24]:


train_categorical_sales = train.groupby(['product', 'store', 'country'])['num_sold'].sum().to_frame().reset_index().sort_values('num_sold', ascending=False)
train_categorical_sales.head()


# In[25]:


sns.set_context('notebook', font_scale = 1.3)
plt.figure(figsize= (15, 7))
sns.boxplot(x=train_categorical_sales['country'], y=train_categorical_sales['num_sold'], palette= 'summer')
plt.title('Number of Sales by country')
plt.show()


# In[26]:


sns.set_context('notebook', font_scale = 1.3)
plt.figure(figsize=(15, 7))
sns.boxplot(x=train_categorical_sales['country'], 
            y=train_categorical_sales['num_sold'], 
            hue= train_categorical_sales['product'], 
            palette='viridis')
plt.ylabel('Number of Sales');


# In[27]:


plt.figure(figsize=(15, 7))
sns.barplot(x=train_categorical_sales['country'], 
            y=train_categorical_sales['num_sold'], 
            hue= train_categorical_sales['product'], 
            palette='summer')
plt.ylabel('Number of Sales');


# #### Kaggle for kids in Germany in a Kaggle mart sold the most

# In[28]:


plt.figure(figsize=(15, 7))
sns.barplot(x=train_categorical_sales['country'], y=train_categorical_sales['num_sold'], hue= train_categorical_sales['store'], palette='winter')
plt.ylabel('Number of Sales');


# In[29]:


train.groupby(['Year','country']).sum()['num_sold'].reset_index().pivot('Year', 'country', 'num_sold')


# In[30]:


train_year_country_cummulative = train.groupby(['Year','country']).sum()['num_sold'].reset_index().pivot('Year', 'country', 'num_sold')
train_year_country_cummulative = np.cumsum(train_year_country_cummulative)


fig = px.line(train_year_country_cummulative, title='Number of Sales Cumulative Countrywise vs Year', markers=True)
fig.show()


# #### Belgium and Germany Sales are almost coinciding, there is hardly any differnece in them

# In[31]:


train_year_product_cummulative = train.groupby(['Year','product']).sum()['num_sold'].reset_index().pivot('Year', 'product', 'num_sold')
train_year_product_cummulative = np.cumsum(train_year_product_cummulative)


fig = px.line(train_year_product_cummulative, title='Number of Sales Cumulative Productwise vs Year', markers=True)
fig.show()


# In[32]:


fig = px.line(train.groupby('date').sum()['num_sold'], title='Number of Sales over time')
fig.show()


# #### Sales clearly increased drastically during the start and end of each year

# <div style='color: #216969;
#            background-color: #EAF6F6;
#            font-size: 200%;
#            border-radius:15px;
#            text-align:center;
#            font-weight:600;
#            border-style: solid;
#            border-color: dark green;
#            font-family: "Verdana";'>
# Feature Engineering
# <a class="anchor" id="5"></a> 

# This part is inspired from<a href='https://www.kaggle.com/code/kotrying/tps0922'> this </a> notebook

# In[33]:


train['weekday'] = train['date'].dt.weekday
train['amount_time'] = train['Month']*100 + train['Day']
train['special_days'] = train['amount_time'].isin([101,1228,1229,1230,1231]).astype(int)
train['weekend']= train['weekday'].isin([5,6])

test['weekday'] = test['date'].dt.weekday
test['amount_time'] = test['Month']*100 + test['Day']
test['special_days'] = test['amount_time'].isin([101,1228,1229,1230,1231]).astype(int)
test['weekend']= test['weekday'].isin([5,6])

train['Month'] = np.cos(0.5236 * train['Month'])
test["Month"] = np.cos(0.5236 * test['Month'])


# <div style='color: #216969;
#            background-color: #EAF6F6;
#            font-size: 200%;
#            border-radius:15px;
#            text-align:center;
#            font-weight:600;
#            border-style: solid;
#            border-color: dark green;
#            font-family: "Verdana";'>
# Encoding
# <a class="anchor" id="6"></a> 

# In[34]:


train = pd.get_dummies(train, drop_first = True)
test = pd.get_dummies(test, drop_first = True)
train.head()


# <div style='color: #216969;
#            background-color: #EAF6F6;
#            font-size: 200%;
#            border-radius:15px;
#            text-align:center;
#            font-weight:600;
#            border-style: solid;
#            border-color: dark green;
#            font-family: "Verdana";'>
# Scaling
# <a class="anchor" id="7"></a> 

# #### As we saw in the EDA, our target variable was clearly skewed, therefore it will be a good idea to apply boxcox transformation to it

# In[35]:


train.index = train.date
train.drop('date', axis=1, inplace=True)

test.index = test.date
test.drop('date', axis=1, inplace=True)


# In[36]:


X = train.drop('num_sold', axis=1).values
y = train['num_sold']
X_test = test.values


# In[37]:


# Transforming target
bc_result = boxcox(y)
boxcox_y = bc_result[0]
lam = bc_result[1]


# In[38]:


# Transforming features
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)


# In[39]:


X.shape, X_test.shape


# <div style='color: #216969;
#            background-color: #EAF6F6;
#            font-size: 200%;
#            border-radius:15px;
#            text-align:center;
#            font-weight:600;
#            border-style: solid;
#            border-color: dark green;
#            font-family: "Verdana";'>
# Modelling
# <a class="anchor" id="8"></a> 

# In[40]:


models = {
    'ridge' : Ridge(),
    'xgboost' : XGBRegressor(),
    'catboost' : CatBoostRegressor(verbose=0),
    'lightgbm' : LGBMRegressor(),
    'gradient boosing' : GradientBoostingRegressor(),
    'lasso' : Lasso(),
    'random forest' : RandomForestRegressor(),
    'hist gb' : HistGradientBoostingRegressor(),
    'bayesian ridge' : BayesianRidge(),
    'support vector': SVR(),
    'knn' : KNeighborsRegressor(n_neighbors = 4)
}


# In[41]:


X_train = X[: int(len(X)*0.8)]
y_train_boxcox = boxcox_y[: int(len(boxcox_y)*0.8)]
X_val = X[int(len(X)*0.8): ]
y_val_boxcox = boxcox_y[int(len(boxcox_y)*0.8): ]


# In[42]:


for name, model in models.items():
    model.fit(X_train, y_train_boxcox)
    print(f'{name} trained')


# In[43]:


results = {}
for name, model in models.items():
    mse = mean_squared_error(y_val_boxcox, model.predict(X_val))
    results[name] = mse


# In[44]:


for name, result in results.items():
    print("----------------")
    print(f'{name} : {result}')


# In[45]:


results_df = pd.DataFrame(results, index=range(0,1)).T.rename(columns={0: 'MSE'}).sort_values('MSE', ascending=False)
results_df


# In[46]:


plt.figure(figsize = (15, 6))
sns.barplot(x= results_df.index, y = results_df['MSE'], palette = 'summer')
plt.xlabel('Model')
plt.ylabel('MSE')
plt.title('MSE of different models');


# <div style='color: #216969;
#            background-color: #EAF6F6;
#            font-size: 200%;
#            border-radius:15px;
#            text-align:center;
#            font-weight:600;
#            border-style: solid;
#            border-color: dark green;
#            font-family: "Verdana";'>
# Hyperparameter Optimization
# <a class="anchor" id="9"></a> 

# In[47]:


# def catboost_objective(trial):
#     learning_rate = trial.suggest_float('learning_rate', 0, 0.5)
#     depth = trial.suggest_int('depth', 3, 10)
#     n_estimators = trial.suggest_int('n_estimators', 50, 600)
    
#     model = CatBoostRegressor(
#         learning_rate= learning_rate,
#         depth= depth,
#         n_estimators= n_estimators,
#         verbose= 0
#     )

#     model.fit(X_train, y_train_boxcox)
#     mse = mean_squared_error(y_val_boxcox, model.predict(X_val))

#     return mse

# study_1 = optuna.create_study(direction= 'minimize')
# study_1.optimize(catboost_objective, n_trials= 100)


# In[48]:


# def lightgbm_objective(trial):
#     learning_rate = trial.suggest_float('learning_rate', 0, 0.5)
#     n_estimators = trial.suggest_int('n_estimators', 50, 600)
#     max_depth = trial.suggest_int('max_depth', 5, 30),
#     num_leaves = trial.suggest_int('num_leaves', 15, 60)

#     model = LGBMRegressor(
#         learning_rate= learning_rate,
#         n_estimators= n_estimators,
#         max_depth= max_depth,
#         num_leaves= num_leaves
#     )

#     model.fit(X_train, y_train_boxcox)
#     mse = mean_squared_error(y_val_boxcox, model.predict(X_val))

#     return mse

# study_2 = optuna.create_study(direction= 'minimize')
# study_2.optimize(lightgbm_objective, n_trials= 100)


# In[49]:


# def histgbm_objective(trial):
#     learning_rate = trial.suggest_float('learning_rate', 0, 0.5)
#     max_iter = trial.suggest_int('max_iter', 50, 600)
#     max_depth = trial.suggest_int('max_depth', 5, 50)
#     max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 15, 60)
#     tol = trial.suggest_loguniform('tol', 1e-7, 0.1)

#     model = HistGradientBoostingRegressor(
#         learning_rate= learning_rate,
#         max_iter= max_iter,
#         max_depth= max_depth,
#         tol=tol,
#         max_leaf_nodes= max_leaf_nodes
#     )

#     model.fit(X_train, y_train_boxcox)
#     mse = mean_squared_error(y_val_boxcox, model.predict(X_val))

#     return mse

# study_3 = optuna.create_study(direction= 'minimize')
# study_3.optimize(histgbm_objective, n_trials= 100)


# In[50]:


# def xgboost_objective(trial):
#     eta = trial.suggest_float('eta', 0, 1)
#     max_depth = trial.suggest_int('max_depth', 5, 50)
#     n_estimators = trial.suggest_int('n_estimators', 50, 600)

#     model = XGBRegressor(
#         eta= eta,
#         n_estimators= n_estimators,
#         max_depth= max_depth,
#     )

#     model.fit(X_train, y_train_boxcox)
#     mse = mean_squared_error(y_val_boxcox, model.predict(X_val))

#     return mse

# study_4 = optuna.create_study(direction= 'minimize')
# study_4.optimize(xgboost_objective, n_trials= 100) 


# In[51]:


catboost_params = {
    'learning_rate': 0.3200776816243584, 
    'depth': 6, 
    'n_estimators': 419,
    'verbose':0
    # 0.044 MSE
}

lightgbm_params = {
    'learning_rate': 0.21488865369419763,
    'n_estimators': 530,
    'max_depth': 5,
    'num_leaves': 37
    # 0.052 MSE
}

histgbm_params = {
    'learning_rate': 0.21327894027228303,
    'max_iter': 296,
    'max_depth': 24,
    'max_leaf_nodes': 18,
    'tol': 1.0366124369259605e-06
    # 0.049 MSE
}

xgboost_params = {
    'eta': 0.32867047514344655, 
    'max_depth': 5, 
    'n_estimators': 226
    # 0.051 MSE
}


# <div style='color: #216969;
#            background-color: #EAF6F6;
#            font-size: 200%;
#            border-radius:15px;
#            text-align:center;
#            font-weight:600;
#            border-style: solid;
#            border-color: dark green;
#            font-family: "Verdana";'>
# Final Predictions
# <a class="anchor" id="10"></a> 

# In[52]:


models = {
    'xgboost' : XGBRegressor(**xgboost_params),
    'catboost' : CatBoostRegressor(**catboost_params),
    'lightgbm' : LGBMRegressor(**lightgbm_params),
    'hist gb' : HistGradientBoostingRegressor(**histgbm_params)
}


# In[53]:


for name, model in models.items():
    model.fit(X, boxcox_y)


# In[54]:


final_predictions = (
    0.25 * inv_boxcox(models['catboost'].predict(X_test), lam) +
    0.25 * inv_boxcox(models['xgboost'].predict(X_test), lam) +
    0.25 * inv_boxcox(models['lightgbm'].predict(X_test), lam) + 
    0.25 * inv_boxcox(models['hist gb'].predict(X_test), lam)
)


# In[55]:


combined_test = pd.DataFrame({'row_id': test_ids, 'num_sold': final_predictions})
combined_test.head(5)


# In[56]:


combined_test.to_csv('submission.csv', index=None)

