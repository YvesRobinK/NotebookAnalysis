#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


import warnings
warnings.filterwarnings('ignore')

import scipy
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# In[ ]:


train_df = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
test_df = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
combine = [train_df, test_df]


# In[ ]:


train_df.shape, test_df.shape


# In[ ]:


train_df.info()


# In[ ]:


train_df.describe()


# # **Data visualization**

# In[ ]:


train_df.dtypes.value_counts()


# In[ ]:


train_df.duplicated().value_counts()


# In[ ]:


categorical = train_df.select_dtypes(include='object')
numerical = train_df.select_dtypes(exclude='object')


# In[ ]:


categorical.shape, numerical.shape


# In[ ]:


categorical.columns


# In[ ]:


train_df['Neighborhood'].value_counts()


# In[ ]:


fig = plt.figure(figsize=(20, 15))

for index, column in enumerate(categorical.columns[0:23]):
    sns.countplot(categorical[column])
    plt.subplot(6, 4, index+1)
    
plt.tight_layout()    
plt.show();   


# In[ ]:


fig = plt.figure(figsize=(20, 15))
for index, column in enumerate(categorical.columns[22:43]):
    sns.countplot(categorical[column])
    plt.subplot(6, 4, index+1)
    
plt.tight_layout()    
plt.show();   


# In[ ]:


#features that in most samples  have only one value

features_one_value_cat = ['Street', 'LandContour', 'Utilities', 'LandSlope', 'Condition2', 'RoofMatl', 
                      'BsmtCond', 'BsmtFinType2', 'Heating', 'Functional', 'GarageQual', 'GarageCond',
                     'PavedDrive', 'MiscFeature']

#ordinal features

ord_features = ['ExterQual', 'HeatingQC', 'ExterCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'KitchenQual', 'FireplaceQu', 'PoolQC']


# In[ ]:


fig = plt.figure(figsize=(20, 15))

for index, column in enumerate(ord_features):
    sns.boxplot(data=train_df, x=column, y="SalePrice")
    plt.subplot(6, 4, index+1)
    
plt.tight_layout()    
plt.show();   


# In[ ]:


fig = plt.figure(figsize=(20, 15))

for index, column in enumerate(numerical):
    sns.scatterplot(data=train_df, x=column, y="SalePrice")
    plt.subplot(7, 6, index+1)
    
plt.tight_layout()    
plt.show();   


# In[ ]:


numerical = numerical.drop(['Id'], axis=1)


# In[ ]:


fig = plt.figure(figsize=(20, 15))

for index, column in enumerate(numerical):
    train_df[column].hist(legend=True)
    plt.subplot(7, 6, index+1)
    
plt.tight_layout()    
plt.show();   


# In[ ]:


#features that in most samples  have only one value

features_one_value_num = ['BsmtFinSF2', 'LowQualFinSF', 'KitchenAbvGr',  'PoolArea', 'MiscVal']


# In[ ]:


numerical = numerical[numerical.columns.difference(features_one_value_num)]


# In[ ]:


fig = plt.figure(figsize=(20, 15))

for index, column in enumerate(numerical):
    plt.subplot(7, 5, index+1)
    sns.boxplot(y=column, data=numerical.dropna())
    
    
plt.tight_layout()    
plt.show();  


# In[ ]:


numerical.corr()


# In[ ]:


plt.figure(figsize=(10, 10))
sns.heatmap(numerical.corr());


# Highly correlated features:
# 
# * TotalBsmtSF, 1stFlrSF;
# * GarageYrBlt, YearBuilt;
# * GrLivArea, TotRmsAbvGrd;
# * GarageArea, GarageCars

# In[ ]:


highly_corr_features = ['1stFlrSF', 'GarageYrBlt', 'TotRmsAbvGrd', 'GarageCars']


# In[ ]:


plt.figure(figsize=(10, 10))
sns.heatmap(numerical.corr()[numerical.corr() > 0.8], annot=True, fmt='.1g');


# # **Removing of redundant features**

# In[ ]:


missed_val_features = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence']
useless_features = ['MoSold', 'YrSold']


# In[ ]:


features_for_removing = highly_corr_features + missed_val_features + useless_features + features_one_value_cat + features_one_value_num
features_for_removing


# In[ ]:


for dataset in combine:
    dataset.drop(columns=features_for_removing, inplace=True)


# In[ ]:


train_df.head()


# # **Dealing with outliars**

# In[ ]:


outliars_features = ['BsmtFinSF1', 'LotArea', 'LotFrontage', 'TotalBsmtSF', 'GrLivArea', 'EnclosedPorch']


# In[ ]:


fig = plt.figure(figsize=(10, 5))

for index, column in enumerate(outliars_features):
    plt.subplot(2, 3, index+1)
    sns.boxplot(y=column, data=numerical.dropna())
   
    
plt.tight_layout()    
plt.show();


# In[ ]:


train_df = train_df.drop(train_df[train_df['BsmtFinSF1'] > 4000].index)
train_df = train_df.drop(train_df[train_df['LotArea'] > 100000].index)
train_df = train_df.drop(train_df[train_df['LotFrontage'] > 200].index)
train_df = train_df.drop(train_df[train_df['TotalBsmtSF'] > 5000].index)
train_df = train_df.drop(train_df[train_df['GrLivArea'] > 4000].index)
train_df = train_df.drop(train_df[train_df['EnclosedPorch'] > 400].index)


# In[ ]:


fig = plt.figure(figsize=(10, 5))

for index, column in enumerate(outliars_features):
    plt.subplot(2, 3, index+1)
    sns.boxplot(y=column, data=train_df)
   
    
plt.tight_layout()    
plt.show();


# # **Filling missing values**

# In[ ]:


train_df.isnull().sum().sort_values(ascending=False).head(10)


# In[ ]:


test_df.isnull().sum().sort_values(ascending=False).head(20)


# In[ ]:


filling_NA = ['BsmtExposure', 'BsmtQual', 'BsmtFinType1', 'GarageFinish', 'GarageType', 'KitchenQual']
train_df.loc[:, filling_NA] = train_df[filling_NA].fillna('NA')     
test_df.loc[:, filling_NA] = test_df[filling_NA].fillna('NA')  


# In[ ]:


train_df['MasVnrType'].fillna('None', inplace=True)
test_df['MasVnrType'].fillna('None', inplace=True)


# In[ ]:


train_df['Electrical'] = train_df['Electrical'].fillna(train_df['Electrical'].mode()[0])

filling_mode = ['MSZoning', 'Exterior1st', 'Exterior2nd']
test_df.loc[:, filling_mode] = test_df[filling_mode].apply(lambda x: x.fillna(x.mode()[0]))


# In[ ]:


filling_zero = ['BsmtFullBath', 'GarageArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF']
test_df.loc[:, filling_zero] = test_df[filling_zero].apply(lambda x: x.fillna(x.mode()[0]))


# In[ ]:


test_df['SaleType'] = test_df['SaleType'].fillna('Oth')


# In[ ]:


train_df['MasVnrArea'] = train_df.groupby('MasVnrType')['MasVnrArea'].transform(lambda x: x.fillna(x.mean()))
test_df['MasVnrArea'] = test_df.groupby('MasVnrType')['MasVnrArea'].transform(lambda x: x.fillna(x.mean()))

train_df['LotFrontage'] = train_df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
test_df['LotFrontage'] = test_df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))


# # **Categorical Features Encoding**

# Ordinal variables are replaced by numbers.

# In[ ]:


dict_1 = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
dict_2 = {'Gd': 3, 'Av': 2, 'Mn': 1, 'No': 0, 'NA': 0}
dict_3 = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0}
dict_4 = {'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0}
dict_5 = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0}
dict_6 = {'Y': 1, 'N': 0}


# In[ ]:


train_df['ExterQual'] = train_df['ExterQual'].map(dict_1)
train_df['HeatingQC'] = train_df['HeatingQC'].map(dict_1)
train_df['ExterCond'] = train_df['ExterCond'].map(dict_1)
train_df['BsmtQual'] = train_df['BsmtQual'].map(dict_1)
train_df['KitchenQual'] = train_df['KitchenQual'].map(dict_1)
train_df['BsmtExposure'] = train_df['BsmtExposure'].map(dict_2)
train_df['BsmtFinType1'] = train_df['BsmtFinType1'].map(dict_3)
train_df['LotShape'] = train_df['LotShape'].map(dict_4)
train_df['GarageFinish'] = train_df['GarageFinish'].map(dict_5)
train_df['CentralAir'] = train_df['CentralAir'].map(dict_6)


# In[ ]:


test_df['ExterQual'] = test_df['ExterQual'].map(dict_1)
test_df['HeatingQC'] = test_df['HeatingQC'].map(dict_1)
test_df['ExterCond'] = test_df['ExterCond'].map(dict_1)
test_df['BsmtQual'] = test_df['BsmtQual'].map(dict_1)
test_df['KitchenQual'] = test_df['KitchenQual'].map(dict_1)
test_df['BsmtExposure'] = test_df['BsmtExposure'].map(dict_2)
test_df['BsmtFinType1'] = test_df['BsmtFinType1'].map(dict_3)
test_df['LotShape'] = test_df['LotShape'].map(dict_4)
test_df['GarageFinish'] = test_df['GarageFinish'].map(dict_5)
test_df['CentralAir'] = test_df['CentralAir'].map(dict_6)


# In[ ]:


categorical_train = train_df.select_dtypes(include='object').columns
categorical_test = test_df.select_dtypes(include='object').columns


# Other categorical variables are replaced by the mean value of the SalePrice for a separate category.

# In[ ]:


def code_mean(dataset, cat_feature):
    return (dataset[cat_feature].map(train_df.groupby(cat_feature)['SalePrice'].mean()))


# In[ ]:


for col in  categorical_test:
    test_df[col] = code_mean(test_df, col)


# In[ ]:


for col in  categorical_train:
    train_df[col] = code_mean(train_df, col)


# # **Feature engineering**

# In[ ]:


print(train_df.columns)


# In[ ]:


fig = plt.figure(figsize=(20, 15))

for index, column in enumerate(train_df):
    sns.scatterplot(data=train_df, x=column, y="SalePrice")
    plt.subplot(8, 7, index+1)
    
plt.tight_layout()    
plt.show();  


# In[ ]:


fig = plt.figure(figsize=(20, 15))

for index, column in enumerate(train_df):
    train_df[column].hist(legend=True)
    plt.subplot(8, 7, index+1)
    
plt.tight_layout()    
plt.show();   


# Let combine all features with porch and creat new one 'PorchArea'.

# In[ ]:


train_df['PorchArea'] = train_df['OpenPorchSF'] + train_df['EnclosedPorch'] + train_df['3SsnPorch'] + train_df['ScreenPorch']


# In[ ]:


test_df['PorchArea'] = test_df['OpenPorchSF'] + test_df['EnclosedPorch'] + test_df['3SsnPorch'] + test_df['ScreenPorch']


# In[ ]:


train_df.drop(columns=['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'], inplace=True)
test_df.drop(columns=['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'], inplace=True)


# I create new feature '**NumBath**' (number of bathrooms) based on all features describe bathrooms( '**BsmtFullBath**', '**BsmtHalfBath**', '**FullBath**', '**HalfBath**').
# 
# Also I multiply '**BsmtHalfBath**' and 'HalfBath' on 0.5 because they are not full bathrooms.

# In[ ]:


train_df['NumBath'] = train_df['BsmtFullBath'] + 0.5*train_df['BsmtHalfBath'] + train_df['FullBath'] + 0.5*train_df['HalfBath']
test_df['NumBath'] = test_df['BsmtFullBath'] + 0.5*test_df['BsmtHalfBath'] + test_df['FullBath'] + 0.5*test_df['HalfBath']


# If '**YearRemodAdd**' equal '**YearBuilt**' remodeling or additions was not performed,in this case we assign 0 this column.
# 
# If If '**YearRemodAdd**' not equal '**YearBuilt**', we asign 1 for '**YearRemodAdd**'(emodeling or additions was performed)

# In[ ]:


train_df['YearRemodAdd'].loc[train_df[train_df['YearRemodAdd'] != train_df['YearBuilt']]['YearRemodAdd'].index.values.tolist()] = 1
train_df['YearRemodAdd'].loc[train_df[train_df['YearRemodAdd'] == train_df['YearBuilt']]['YearRemodAdd'].index.values.tolist()] = 0


# In[ ]:


test_df['YearRemodAdd'].loc[test_df[test_df['YearRemodAdd'] != test_df['YearBuilt']]['YearRemodAdd'].index.values.tolist()] = 1
test_df['YearRemodAdd'].loc[test_df[test_df['YearRemodAdd'] == test_df['YearBuilt']]['YearRemodAdd'].index.values.tolist()] = 0


# In[ ]:


train_df['2ndFlrSF'].hist();


# Since most values of '**2ndFlrSF**' are 0, then we create new feature '**Is2ndFlr**'(it shows whether 2nd floor exists).

# In[ ]:


train_df['2ndFlrSF'].loc[train_df[train_df['2ndFlrSF'] != 0]['2ndFlrSF'].index.values.tolist()] = 1


# In[ ]:


train_df.rename(columns={'2ndFlrSF': 'Is2ndFlr'}, inplace=True)


# In[ ]:


test_df['2ndFlrSF'].loc[test_df[test_df['2ndFlrSF'] != 0]['2ndFlrSF'].index.values.tolist()] = 1
test_df.rename(columns={'2ndFlrSF': 'Is2ndFlr'}, inplace=True)


# In[ ]:


train_df.columns


# In[ ]:


sns.scatterplot(data=train_df, x='Fireplaces', y="SalePrice");


# Also we change all values of '**Fireplaces**' that are more than 0 on 1

# In[ ]:


train_df['Fireplaces'].loc[train_df[train_df['Fireplaces'] != 0]['Fireplaces'].index.values.tolist()] = 1
test_df['Fireplaces'].loc[test_df[test_df['Fireplaces'] != 0]['Fireplaces'].index.values.tolist()] = 1


# In[ ]:


test_df['NumBath'] = test_df['NumBath'].fillna(0)
test_df['BsmtHalfBath'] = test_df['BsmtHalfBath'].fillna(0)


# In[ ]:


test_df.info()


# # **Model**

# Linear regression

# In[ ]:


X = train_df.drop(columns=['Id', 'SalePrice'])
y = train_df['SalePrice']
X_test = test_df.drop(columns=['Id'])


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)


# In[ ]:


lin_reg = LinearRegression(n_jobs=-1)
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_val)

mean_squared_error(y_val, y_pred, squared=False)


# In[ ]:


lin_reg.score(X_val, y_val)


# In[ ]:


y.describe()


# In[ ]:


pd.DataFrame(lin_reg.coef_, X.columns, columns=[ 'Coef']).sort_values(by=['Coef'])


# Random Forest.
# 
# Parameters for Random Forest Regressor were tuning by using GridSearchCV.

# In[ ]:


rf = RandomForestRegressor(max_depth=20, max_features=12, min_samples_leaf=2,
                      n_estimators=644, n_jobs=-1, random_state=1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_val)

mean_squared_error(y_val, y_pred, squared=False)


# In[ ]:


model = RandomForestRegressor(max_depth=20, max_features=12, min_samples_leaf=2,
                      n_estimators=644, n_jobs=-1, random_state=1)
model.fit(X, y)
predict = model.predict(X_test)


# In[ ]:


output = pd.DataFrame({'Id': test_df.Id, 'SalePrice': predict})
output.to_csv('house_price_baseline.csv', index=False)

