#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# <h3 style="text-align: center;">Loading the data</h3>

# In[3]:


train_df = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv', index_col='PassengerId')
test_df = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv', index_col='PassengerId')


# <h3 style="text-align: center;">Checking basic info</h3>

# In[4]:


train_df.head()


# In[5]:


train_df.info()


# In[6]:


from pandas_profiling import ProfileReport

profile = ProfileReport(train_df, title="Profiling Report")
profile


# <h3 style="text-align: center;">Some feature engineering</h3>

# In[7]:


train_df.drop('Name', axis=1, inplace=True)
test_df.drop('Name', axis=1, inplace=True)


# In[8]:


train_df['Transported'].replace(False, 0, inplace=True)
train_df['Transported'].replace(True, 1, inplace=True)


# <h3 style="text-align: center;">Let's separate the cabin columns in three new features</h3>

# In[9]:


train_df[['deck','num', 'side']] = train_df['Cabin'].str.split('/', expand=True)
test_df[['deck','num', 'side']] = test_df['Cabin'].str.split('/', expand=True)

train_df.drop('Cabin', axis=1, inplace=True)
test_df.drop('Cabin', axis=1, inplace=True)


# In[10]:


object_cols = [col for col in train_df.columns if train_df[col].dtype == 'object' or train_df[col].dtype == 'category']
numeric_cols = [col for col in train_df.columns if train_df[col].dtype == 'float64']

print(f'Object cols -- {object_cols}')
print(f'Numeric cols -- {numeric_cols}')


# <h3 style="text-align: center;">Sum of spent value by passenger, creating a new feature</h3>

# In[11]:


col_to_sum = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

train_df['SumSpends'] = train_df[col_to_sum].sum(axis=1)
test_df['SumSpends'] = test_df[col_to_sum].sum(axis=1)


# <h3 style="text-align: center;">Checking null and object columns</h3>

# In[12]:


null_cols = train_df.isnull().sum().sort_values(ascending=False)
null_cols = list(null_cols[null_cols>1].index)
null_cols


# In[13]:


train_df[object_cols] = train_df[object_cols].astype('category')
test_df[object_cols] = test_df[object_cols].astype('category')


# In[14]:


print(f'Train DF shape: {train_df.shape}')
print(f'Test DF shape: {test_df.shape}')


# <h3 style="text-align: center;">Encoding the categorical variables</h3>

# In[15]:


from sklearn.preprocessing import OrdinalEncoder

oc = OrdinalEncoder()

df_for_encode = pd.concat([train_df, test_df])

df_for_encode[object_cols] = df_for_encode[object_cols].astype('category')

df_for_encode[object_cols] = oc.fit_transform(df_for_encode[object_cols])

del train_df, test_df

train_df = df_for_encode.iloc[:8693, :]
test_df = df_for_encode.iloc[8693: , :]

del df_for_encode

test_df.drop('Transported', inplace=True, axis=1)


# In[16]:


print(f'Train DF shape: {train_df.shape}')
print(f'Test DF shape: {test_df.shape}')


# In[17]:


from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


ct = ColumnTransformer([("imp", SimpleImputer(strategy='mean'), null_cols)])
    
train_df[null_cols] = ct.fit_transform(train_df[null_cols])
test_df[null_cols] = ct.fit_transform(test_df[null_cols])


# <h3 style="text-align: center;">Prearing the dataset for modeling</h3>

# In[18]:


X = train_df.copy()
y = X.pop('Transported')

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=23)


# <h3 style="text-align: center;">Testing 4 models without hyperparameter tunning</h3>

# In[51]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

def predict_and_acc(model, verbose=None):
    if verbose == None:
        model = model()
        model.fit(X_train, y_train)
        predict = model.predict(X_test)
        cvs = cross_val_score(model, X, y, cv=4)
        print(f'The accuracy score of {str(model)} is {float(accuracy_score(y_test, predict))}')
        print(f'The cross validation of {str(model)} is:{cvs} with mean of {cvs.mean()}')
    else:
        model = model(verbose=verbose)
        model.fit(X_train, y_train)
        predict = model.predict(X_test)
        cvs = cross_val_score(model, X, y, cv=4)
        print(f'The accuracy score of {str(model)} is {float(accuracy_score(y_test, predict))}')
        print(f'The cross validation of {str(model)} is:{cvs} with mean of {cvs.mean()}')


# In[47]:


predict_and_acc(RandomForestClassifier, None)


# In[48]:


predict_and_acc(AdaBoostClassifier)


# In[53]:


predict_and_acc(LGBMClassifier)


# In[52]:


predict_and_acc(CatBoostClassifier, verbose=False)


# <h3 style="text-align: center;">Backward Feature Selection for the best model</h3>

# In[55]:


from sklearn.feature_selection import SequentialFeatureSelector

model_fs = CatBoostClassifier(verbose=False)
sf = SequentialFeatureSelector(model_fs, scoring='accuracy', direction = 'backward')
sf.fit(X,y)


# In[56]:


best_features = list(sf.get_feature_names_out())
best_features


# In[57]:


model = CatBoostClassifier(verbose=False, eval_metric='Accuracy')
model.fit(X[best_features], y)
prediction = model.predict(test_df[best_features])


# In[58]:


#Prediction
final = pd.DataFrame()
final.index = test_df.index
final['Transported'] = prediction
final['Transported'].replace(0, False, inplace=True)
final['Transported'].replace(1, True, inplace=True)
final.to_csv('submission.csv')


# <h3 style="text-align: center;">Final score so far: 0.81271 -- in progress</h3>
