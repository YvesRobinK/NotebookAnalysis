#!/usr/bin/env python
# coding: utf-8

# <a id="1"></a> <br>
# ## 1-Imports

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


from xgboost import XGBRegressor as xgbr # for modelling
from sklearn.pipeline import Pipeline # for making pipleine 
from sklearn.impute import SimpleImputer # for handling missing variables either categorical or numerical
from sklearn.preprocessing import OneHotEncoder # for one hot encoding categorical variables
from sklearn.metrics import mean_absolute_error # for Mean absolute error
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.compose import ColumnTransformer



# <a id=2></a> <br>
# ## 2- Getting Data
# 

# In[3]:


# Read the data
X_full = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv', index_col='Id')
X_test_full = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and 
                    X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if 
                X_train_full[cname].dtype in ['int64', 'float64']]


# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()


# <a id="3"></a> <br>
# ## 3-Strategy to Handle Categorical and Missing Values

# In[4]:


# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='median')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# handle unknown is set to ignore because sometimes in test set we have variables that were not present in Training set and hence were not encoded while training 
# but if we use these variables while testing we will get error hence to ignore these errors we use this argument

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# <a id=4></a> <br>
# ## 4-Final Model

# In[5]:


# Define the model
my_model_2 = xgbr(random_state=42,n_estimators=2000,learning_rate=0.055) # Your code here

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model2', my_model_2)
                     ])

# Preprocessing of training data, fit model 
clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds))


# <a id=5></a> <br>
# ## 5-Submission

# In[6]:


preds_test = clf.predict(X_test) # Your code here

# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission1.csv', index=False)


# In[7]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV


# In[8]:


X = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv', index_col='Id')
from sklearn.impute import SimpleImputer
# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice              
X.drop(['SalePrice'], axis=1, inplace=True)
cols= [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

X=X[cols]
X.head()
my_imputer=SimpleImputer(strategy='median')

X_train=pd.DataFrame(my_imputer.fit_transform(X))

X_train.columns=X.columns
y_train=y
X_train.head()
def rfr_model(X, y):
# Perform Grid-Search
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(2,10),
            'n_estimators': (10, 50, 100,200,150,300,500,600,700,800,900, 1000),
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    
    grid_result = gsc.fit(X_train,y_train)
    best_params = grid_result.best_params_
    
    rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"], random_state=False, verbose=False)
    scores = cross_val_score(rfr,X_train,y_train, cv=10, scoring='neg_mean_absolute_error')
#     print(best_params)
    return best_params


# In[9]:


rfr_model(X_train,y_train)

