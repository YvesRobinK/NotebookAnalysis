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


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, skew
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.special import boxcox1p
import warnings

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns', None)


# In[3]:


df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# # Take a look at the data

# In[4]:


df_train.head()


# In[5]:


df_train.shape


# In[6]:


df_test.head()


# In[7]:


df_test.shape


# In[8]:


df_train.columns


# In[9]:


df_test.columns


# # Analyze the data

# In[10]:


print(df_train['SalePrice'].describe())


# In[11]:


sns.distplot(df_train['SalePrice'])


# #### It seems like there is a positive skewness in the target feature (Sale Price)

# In[12]:


print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())


# #### Handle the skewness of the target feature

# In[13]:


df_train['SalePrice'] = np.log1p(df_train['SalePrice'])


# ## Exploring some realtionships between the target and some features

# In[14]:


var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
plt.scatter(x=data[var], y=data['SalePrice'])
print(data[:5])
plt.show()


# In[15]:


var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
plt.scatter(x=data[var], y=data['SalePrice'])
print(data.head())
plt.show()


# ### This looks like a there is a linear realtionship between (Sale Price) and (Total Basement, GrLiveArea)

# In[16]:


var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
sns.boxplot(x=data[var], y=data['SalePrice'])


# In[17]:


var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
sns.boxplot(x=data[var], y=data['SalePrice'])


# ### There is no Strong relation between the (year built) and the (sale price)

# # Exploring the correlations between the target and the rest of the features

# In[18]:


corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat)


# ### Focus on the top 10 features and make some analysis

# In[19]:


k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
f, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(cm, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
            yticklabels=cols.values, xticklabels=cols.values)


# In[20]:


cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols])


# # Features engineering
# ### Conatenate all the features toegther to make the engineering more easy and drop the Id and Sale Price features

# In[21]:


y_train = df_train['SalePrice']
ntrain = df_train.shape[0]
ntest = df_test.shape[0]
test_id = df_test['Id']
all_data = pd.concat([df_train, df_test], axis=0, sort=False)
all_data = all_data.drop(['Id', 'SalePrice'], axis=1)


# # Missing Values

# In[22]:


total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum() / all_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(40)


# ### Drop any columns that contains more than 5 null values and keep the rest

# In[23]:


all_data.drop((missing_data[missing_data['Total'] > 5]).index, axis=1, inplace=True)
print(all_data.isnull().sum().max())
print(all_data.info())


# In[24]:


total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum() / all_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[25]:


print(ntrain)
print(ntest)
print(all_data.shape)
print(all_data.columns)


# # Handling the rest of the null values

# In[26]:


for col in ('GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)


# In[27]:


for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)


# In[28]:


all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])


# In[29]:


all_data.drop(['Utilities'], axis=1, inplace=True)


# In[30]:


all_data["Functional"] = all_data["Functional"].fillna("Typ")


# In[31]:


all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])


# In[32]:


all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])


# In[33]:


all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])


# In[34]:


all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])


# In[35]:


total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum() / all_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(5)


# In[36]:


print(ntrain)
print(ntest)
print(all_data.shape)
print(all_data.columns)


# ## Create a new feature

# In[37]:


all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
print(all_data.shape)


# 

# In[ ]:





# # Handle The Skewness in the data

# In[38]:


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
print(len(numeric_feats))
print(numeric_feats)


# In[39]:


skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# In[40]:


skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))


# In[41]:


skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)


# In[42]:


all_data.head()


# # Encode Categorical data

# In[43]:


all_data = pd.get_dummies(all_data)
all_data.shape


# In[44]:


train = all_data[:ntrain]
test = all_data[ntrain:]
print(train.shape)
print(test.shape)


# # Modelling

# In[45]:


from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
scorer = make_scorer(mean_squared_error,greater_is_better = False)
def rmse_CV_train(model):
    kf = KFold(5,shuffle=True,random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train, y_train,scoring ="neg_mean_squared_error",cv=kf))
    return (rmse)
def rmse_CV_test(model):
    kf = KFold(5,shuffle=True,random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, test, y_test,scoring ="neg_mean_squared_error",cv=kf))
    return (rmse)


# In[46]:


# XGBoost
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


# In[47]:


model_xgb.fit(train, y_train)


# In[48]:


y_pred = np.floor(np.expm1(model_xgb.predict(test)))


# In[49]:


y_pred


# In[50]:


sub = pd.DataFrame()
sub['Id'] = test_id
sub['SalePrice'] = y_pred
sub.to_csv('finalsubmission.csv',index=False)


# In[ ]:




