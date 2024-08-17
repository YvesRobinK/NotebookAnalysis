#!/usr/bin/env python
# coding: utf-8

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


# ## If you found this notebook useful, please give it an upvote!
# ## Currently trying to become a Kaggle Notebook Expert :)

# # 1. Imports

# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import skew, norm
import scipy.stats as stats
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

import xgboost as xgb
from catboost import Pool
from catboost import CatBoostRegressor

import missingno as msno

get_ipython().run_line_magic('matplotlib', 'inline')


# # 2. Data

# In[3]:


# train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
train = pd.read_csv('/kaggle/input/ames-housing-dataset/AmesHousing.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[4]:


train.columns = train.columns.str.replace(' ', '')
train = train.drop(['Order','PID'], axis = 1)
train


# In[5]:


train.describe()


# In[6]:


test.drop('Id', axis=1, inplace=True)
test


# In[7]:


train_test = pd.concat([train, test], axis=0, ignore_index=True)
train_test


# In[8]:


msno.matrix(train_test)


# In[9]:


numerical_features = train_test.select_dtypes(include=[int, float]).columns.tolist()
categorical_features = train_test.select_dtypes(include='object').columns.tolist()


# # 3. EDA

# #### Sale Price

# In[10]:


sns.distplot(train['SalePrice'], kde=True)


# #### Correlation

# In[11]:


corr = train_test.corr()
sns.heatmap(corr, cmap='viridis')


# #### Distribution of Numerical Features

# In[12]:


train_test[numerical_features].hist(bins=25, figsize=(15, 15))


# ##### NOTE: We can see that some of the features are highly right skewed

# #### Pairplots

# In[13]:


selected_features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(data=train_test[selected_features], size=2)


# #### GrLivArea

# In[14]:


sns.regplot(data=train_test, x='GrLivArea', y='SalePrice')


# #### TotalBsmtSF

# In[15]:


sns.regplot(data=train_test, x='TotalBsmtSF', y='SalePrice')


# #### OverallQual

# In[16]:


sns.boxplot(data=train_test, y='SalePrice', x='OverallQual')


# #### YearBuilt

# In[17]:


sns.boxplot(data=train_test, x='YearBuilt', y='SalePrice')


# # 4. Preprocessing & Feature Engineering

# In[18]:


# Converting non-numeric predictors stored as numbers into string
train_test['MSSubClass'] = train_test['MSSubClass'].apply(str)
train_test['YrSold'] = train_test['YrSold'].apply(str)
train_test['MoSold'] = train_test['MoSold'].apply(str)


# #### Fill NA based on data description file

# In[19]:


train_test['Functional'] = train_test['Functional'].fillna('Typ')
train_test['Electrical'] = train_test['Electrical'].fillna("SBrkr")
train_test['KitchenQual'] = train_test['KitchenQual'].fillna("TA")
train_test['Exterior1st'] = train_test['Exterior1st'].fillna(train_test['Exterior1st'].mode()[0])
train_test['Exterior2nd'] = train_test['Exterior2nd'].fillna(train_test['Exterior2nd'].mode()[0])
train_test['SaleType'] = train_test['SaleType'].fillna(train_test['SaleType'].mode()[0])
train_test["PoolQC"] = train_test["PoolQC"].fillna("None")
train_test["Alley"] = train_test["Alley"].fillna("None")
train_test['FireplaceQu'] = train_test['FireplaceQu'].fillna("None")
train_test['Fence'] = train_test['Fence'].fillna("None")
train_test['MiscFeature'] = train_test['MiscFeature'].fillna("None")

for col in ('GarageArea', 'GarageCars'):
    train_test[col] = train_test[col].fillna(0)
        
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    train_test[col] = train_test[col].fillna('None')
    
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    train_test[col] = train_test[col].fillna('None')


# #### Removing the unneeded features

# In[20]:


useless = ['GarageYrBlt', 'YearRemodAdd'] 
train_test.drop(useless, axis=1, inplace=True)


# #### Check columns with missing values

# In[21]:


missing_values = train_test.isnull().sum()
columns_with_missing = missing_values[missing_values > 0].index.tolist()
columns_with_missing = [col for col in columns_with_missing if train_test[col].isnull().any()]

print("Columns with missing values:", columns_with_missing)


# #### Imputation

# In[22]:


def impute_knn(df):
    ttn = train_test.select_dtypes(include=[np.number])
    ttc = train_test.select_dtypes(exclude=[np.number])

    cols_nan = ttn.columns[ttn.isna().any()].tolist()         # columns w/ nan 
    cols_no_nan = ttn.columns.difference(cols_nan).values     # columns w/n nan

    for col in cols_nan:
        imp_test = ttn[ttn[col].isna()]   # indicies which have missing data will become our test set
        imp_train = ttn.dropna()          # all indicies which which have no missing data 
        model = KNeighborsRegressor(n_neighbors=5)  # KNR Unsupervised Approach
        knr = model.fit(imp_train[cols_no_nan], imp_train[col])
        ttn.loc[ttn[col].isna(), col] = knr.predict(imp_test[cols_no_nan])
    
    return pd.concat([ttn, ttc],axis=1)

train_test = impute_knn(train_test)


# In[23]:


objects = []
for i in train_test.columns:
    if train_test[i].dtype == object:
        objects.append(i)
train_test.update(train_test[objects].fillna('None'))
         
train_test[columns_with_missing].isna().sum()


# #### Feature Engineering

# In[24]:


train_test["SqFtPerRoom"] = train_test["GrLivArea"] / (train_test["TotRmsAbvGrd"] +
                                                       train_test["FullBath"] +
                                                       train_test["HalfBath"] +
                                                       train_test["KitchenAbvGr"])

train_test['Total_Home_Quality'] = train_test['OverallQual'] + train_test['OverallCond']

train_test['Total_Bathrooms'] = (train_test['FullBath'] + (0.5 * train_test['HalfBath']) +
                               train_test['BsmtFullBath'] + (0.5 * train_test['BsmtHalfBath']))

train_test["HighQualSF"] = train_test["GrLivArea"] + train_test["1stFlrSF"] + train_test["2ndFlrSF"] + 0.5 * train_test["GarageArea"] + 0.5 * train_test["TotalBsmtSF"] + 1 * train_test["MasVnrArea"]

train_test["Age"] = pd.to_numeric(train_test["YrSold"]) - pd.to_numeric(train_test["YearBuilt"])

train_test["Renovate"] = pd.to_numeric(train_test["YearRemod/Add"]) - pd.to_numeric(train_test["YearBuilt"])


# #### One Hot Encoding

# In[25]:


train_test_dummy = pd.get_dummies(train_test)
train_test_dummy


# #### Remove skew

# In[26]:


numeric_features = train_test_dummy.dtypes[train_test_dummy.dtypes != object].index
skewed_features = train_test_dummy[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skewed_features[skewed_features > 0.5]
skew_index = high_skew.index
    
for i in skew_index:
    train_test_dummy[i] = np.log1p(train_test_dummy[i])


# #### Log Transform Target (SalePrice)

# In[27]:


target = train['SalePrice']
target_log = np.log1p(target)

fig, ax = plt.subplots(1, 2, figsize= (15, 5))
fig.suptitle("qq-plot & distribution SalePrice ", fontsize=15)

sm.qqplot(target_log, stats.t, distargs=(4,),fit=True, line="45", ax=ax[0])
sns.distplot(target_log, kde=True, hist=True, fit=norm, ax=ax[1])
plt.show()


# #### Log Transform HighQualSF

# In[28]:


HighQualSF_log = np.log1p(train_test["HighQualSF"])

fig, ax = plt.subplots(1, 2, figsize= (15, 5))
fig.suptitle("qq-plot & distribution SalePrice ", fontsize=15)

sm.qqplot(HighQualSF_log, stats.t, distargs=(4,), fit=True, line="45", ax=ax[0])
sns.distplot(HighQualSF_log, kde=True, hist=True, fit=norm, ax=ax[1])
plt.show()

train_test["HighQualSF"] = HighQualSF_log


# #### Log Tranform GrLivArea

# In[29]:


GrLivArea_log = np.log1p(train_test["GrLivArea"])

fig, ax = plt.subplots(1, 2, figsize= (15,5))
fig.suptitle("qq-plot & distribution SalePrice ", fontsize=15)

sm.qqplot(GrLivArea_log, stats.t, distargs=(4,), fit=True, line="45", ax=ax[0])
sns.distplot(GrLivArea_log, kde=True, hist=True, fit = norm, ax=ax[1])
plt.show()

train_test["GrLivArea"]= GrLivArea_log


# #### Split back to Train and test

# In[30]:


train = train_test_dummy[0:2930]
test = train_test_dummy[2930:]
test.drop('SalePrice', axis=1, inplace=True)


# In[31]:


ytrain = target_log
xtrain = train.drop('SalePrice', axis=1)

X_train, X_val, y_train, y_val = train_test_split(xtrain, ytrain, test_size=0.5, random_state=42)
X_train, y_train = xtrain, ytrain
X_train


# In[32]:


X_val


# # 5. Modelling

# In[33]:


# best_params = {'max_leaves': 8,
#           'depth': 3,
#           'od_wait': 200,
#           'l2_leaf_reg': 3,
#           'iterations': 200000,
#           'model_size_reg': 0.7,
#           'learning_rate': 0.05,
#           'random_seed': 42 }
# final_model = CatBoostRegressor(**best_params)

final_model = CatBoostRegressor(random_seed=42)


# In[34]:


final_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)


# In[35]:


final_model.get_all_params()


# In[36]:


def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# In[37]:


final_pred = final_model.predict(X_val)


# In[38]:


final_score = rmse(y_val, final_pred)
final_score


# # 6. Submission

# In[39]:


submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
test_pred = np.expm1(final_model.predict(test))


# In[40]:


submission['SalePrice'] = test_pred
submission


# In[41]:


submission.to_csv("submission.csv", index=False, header=True)


# # 7. References
# - https://www.kaggle.com/code/venkatapadavala/house-prices-advanced-regression-practice/notebook
# - https://www.kaggle.com/code/pmarcelino/comprehensive-data-exploration-with-python
