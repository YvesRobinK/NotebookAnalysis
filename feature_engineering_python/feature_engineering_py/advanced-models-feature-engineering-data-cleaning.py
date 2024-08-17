#!/usr/bin/env python
# coding: utf-8

# In[1063]:


#loading need libraries
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[1064]:


#Load data for train and test
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[1065]:


train


# In[1066]:


#shape of train data
train.shape


# In[1067]:


test.shape


# In[1068]:


train.info()


# In[1069]:


test.info()


# ### Target variable 
# Some analysis on target variable

# In[1070]:


plt.subplots(figsize=(12,9))
sns.distplot(train['SalePrice'], fit=stats.norm)

# Get the fitted parameters used by the function

(mu, sigma) = stats.norm.fit(train['SalePrice'])

# plot with the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')

#Probablity plot

fig = plt.figure()
stats.probplot(train['SalePrice'], plot=plt)
plt.show()


# This target varibale is right skewed. Now, we need to tranform this variable and make it normal distribution.
# 

# #### Here we use log for target variable to make more normal distribution

# In[1071]:


#we use log function which is in numpy
train['SalePrice'] = np.log1p(train['SalePrice'])

#Check again for more normal distribution

plt.subplots(figsize=(12,9))
sns.distplot(train['SalePrice'], fit=stats.norm)

# Get the fitted parameters used by the function

(mu, sigma) = stats.norm.fit(train['SalePrice'])

# plot with the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')

#Probablity plot

fig = plt.figure()
stats.probplot(train['SalePrice'], plot=plt)
plt.show()


# ### Checking the missing values

# In[1072]:


def missingdata(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    ms=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    ms= ms[ms["Percent"] > 0]
    f,ax =plt.subplots(figsize=(8,6))
    plt.xticks(rotation='90')
    fig=sns.barplot(ms.index, ms["Percent"],color="green",alpha=0.8)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    return ms


# In[1073]:


missingdata(train)


# In[1074]:


missingdata(test)


# In[1075]:


#missing value counts in each of these columns
# Isnull = train.isnull().sum()/len(train)*100
# Isnull = Isnull[Isnull>0]
# Isnull.sort_values(inplace=True, ascending=False)
# Isnull


# ### Corralation between train attributes

# In[1076]:


#Separate variable into new dataframe from original dataframe which has only numerical values
#there is 38 numerical attribute from 81 attributes
train_corr = train.select_dtypes(include=[np.number])


# In[1077]:


train_corr.shape


# In[1078]:


#Delete Id because that is not need for corralation plot
del train_corr['Id']


# In[1079]:


#Coralation plot
corr = train_corr.corr()
plt.subplots(figsize=(20,9))
sns.heatmap(corr, annot=True)


# #### Top 50% Corralation  train attributes  with sale-price 

# In[1080]:


top_feature = corr.index[abs(corr['SalePrice']>0.5)]
plt.subplots(figsize=(12, 8))
top_corr = train[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()


# Here OverallQual is highly correlated with target feature of saleprice by 82%

# In[1081]:


#unique value of OverallQual
train.OverallQual.unique()


# In[1082]:


sns.barplot(train.OverallQual, train.SalePrice)


# In[1083]:


#boxplot
plt.figure(figsize=(18, 8))
sns.boxplot(x=train.OverallQual, y=train.SalePrice)


# In[1084]:


col = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
sns.set(style='ticks')
sns.pairplot(train[col], size=3, kind='reg')


# In[1085]:


print("Find most important features relative to target")
corr = train.corr()
corr.sort_values(['SalePrice'], ascending=False, inplace=True)
corr.SalePrice


# ### Imputting missing values

# In[1086]:


# PoolQC has missing value ratio is 99%+. So, there is fill by None
train['PoolQC'] = train['PoolQC'].fillna('None')
test['PoolQC'] = test['PoolQC'].fillna('None')


# In[1087]:


#Arround 50% missing values attributes have been fill by None
test['MiscFeature'] = test['MiscFeature'].fillna('None')
test['Alley'] = test['Alley'].fillna('None')
test['Fence'] = test['Fence'].fillna('None')
test['FireplaceQu'] = test['FireplaceQu'].fillna('None')

train['MiscFeature'] = train['MiscFeature'].fillna('None')
train['Alley'] = train['Alley'].fillna('None')
train['Fence'] = train['Fence'].fillna('None')
train['FireplaceQu'] = train['FireplaceQu'].fillna('None')


# In[1088]:


#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
train['LotFrontage'] = train.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
test['LotFrontage'] = test.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))


# In[1089]:


#GarageType, GarageFinish, GarageQual and GarageCond these are replacing with None
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    train[col] = train[col].fillna('None')
    
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    test[col] = test[col].fillna('None')
    


# In[1090]:


#GarageYrBlt, GarageArea and GarageCars these are replacing with zero
for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:
    train[col] = train[col].fillna(int(0))
    
for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:
    test[col] = train[col].fillna(int(0))


# In[1091]:


#BsmtFinType2, BsmtExposure, BsmtFinType1, BsmtCond, BsmtQual these are replacing with None
for col in ('BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual'):
    train[col] = train[col].fillna('None')
    
for col in ('BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual'):
    test[col] = test[col].fillna('None')


# In[1092]:


#MasVnrArea : replace with zero
train['MasVnrArea'] = train['MasVnrArea'].fillna(int(0))

test['MasVnrArea'] = test['MasVnrArea'].fillna(int(0))


# In[1093]:


#MasVnrType : replace with None
train['MasVnrType'] = train['MasVnrType'].fillna('None')

test['MasVnrType'] = test['MasVnrType'].fillna('None')


# In[1094]:


#There is put mode value 
train['Electrical'] = train['Electrical'].fillna(train['Electrical']).mode()[0]

test['Electrical'] = test['Electrical'].fillna(test['Electrical']).mode()[0]


# In[1095]:


#There is no need of Utilities
train = train.drop(['Utilities'], axis=1)

test = test.drop(['Utilities'], axis=1)


# In[1096]:


#Checking there is any null value or not
plt.figure(figsize=(10, 5))
sns.heatmap(train.isnull())

 Now, there is no any missing values
# In[1097]:


train.isnull().sum()


# In[1098]:


test.isnull().sum()


# #### Encoding str to int

# In[1099]:


cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood',
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'GarageType', 'MiscFeature', 
        'SaleType', 'SaleCondition', 'Electrical', 'Heating')


# In[1100]:


from sklearn.preprocessing import LabelEncoder
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train[c].values)) 
    train[c] = lbl.transform(list(train[c].values))


# #### Prepraring data for prediction

# In[1101]:


#Take targate variable into y
y = train['SalePrice']


# In[1102]:


#Delete the saleprice
del train['SalePrice']


# In[1103]:


#Take their values in X and y
X = train.values
y = y.values


# In[1104]:


# Split data into train and test formate
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# ### Linear Regression

# In[1105]:


#Train the model
from sklearn import linear_model
model = linear_model.LinearRegression()


# In[1106]:


#Fit the model
model.fit(X_train, y_train)


# In[1107]:


#Prediction
print("Predict value " + str(model.predict([X_test[142]])))
print("Real value " + str(y_test[142]))


# In[1108]:


#Score/Accuracy
print("Accuracy --> ", model.score(X_test, y_test)*100)


# ### RandomForestRegression

# In[1109]:


#Train the model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=1000)


# In[1110]:


#Fit
randomforest=model.fit(X_train, y_train)


# In[1111]:


Y_pred_rf=randomforest.predict(X_test)


# In[1112]:


#Score/Accuracy

print("Accuracy --> ", model.score(X_test, y_test)*100)
print("Accuracy --> ", model.score(X_test, Y_pred_rf)*100)


# ### GradientBoostingRegressor

# In[1113]:


#Train the model
from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor(n_estimators=100, max_depth=4)


# In[1114]:


#Fit
gbR=GBR.fit(X_train, y_train)


# In[1115]:


Y_pred_gbr = gbR.predict(X_test)


# In[1116]:


print("Accuracy --> ", GBR.score(X_test, y_test)*100)
print("Accuracy --> ", GBR.score(X_test, Y_pred_gbr)*100)


# In[1117]:


# predictions = model.predict(test)
# submission = pd.DataFrame({
    
#         "Id": test["Id"],
#         "SalePrice": predictions})

# submission.to_csv('my_submission.csv', index=False)
# print("Your submission was successfully saved!")


# ## If someone who is going through my notebook and like my solving approach then please upvote my notebook.
# ## Thank You
