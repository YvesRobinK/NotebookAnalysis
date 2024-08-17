#!/usr/bin/env python
# coding: utf-8

# ![download.jpg](attachment:c8101384-5ad3-416b-9354-65c0b99552aa.jpg)

# ![mortgage-kredisi-nedir-turleri-ve-isleyisi-hakkinda-bilgiler.jpg](attachment:c5f04ab0-adad-4e5c-b1c7-7a707b5277ac.jpg)

# ## THE GOAL
# 
# * **This notebook shares a study and explaines advanced regression techniques used to predict house prices.**
# 
# * **Each feature of the dataset represents a different attribute of each house**
#  
# * **The goal is to predict the values for SalePrice feature by applying advanced regression models**

# ## THE METHOD
# 
# * **Cross Validation: Using 5-fold cross-validation**
# 
# * **4 different advanced regression models were used to predict house prices. Models were XGBoost, LightGBM, Gradient Boosting, Random Forest**
# 
# * **Root Mean Squared Error (RMSE) was used as the metric to evaluate the models success**

# ## IMPORTING PACKAGES AND DATASETS

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")
sns.color_palette("husl", 10)
import missingno as msno

from scipy import stats
from scipy.stats import skew
from scipy.special import boxcox1p

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import DataConversionWarning
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

import warnings
from warnings import filterwarnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


# In[2]:


train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[3]:


train_ID = train['Id']
test_ID = test['Id']

train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)


# ## EXPLORATORY DATA ANALYSIS

# In[4]:


train.head()


# In[5]:


train.describe()


# In[6]:


sns.regplot(x = 'TotalBsmtSF', y = 'SalePrice', color = 'navy', data = train)
plt.show()


# #### **A linear relationship seems to be between TotalBsmtSF and SalePrice features**

# In[7]:


sns.regplot(x = 'OverallQual', y = 'SalePrice', color = 'gold', data = train)
plt.show()


# In[8]:


sns.regplot(x = 'GrLivArea', y = 'SalePrice', color = 'red', data = train)
plt.show()


# #### **A linear relationship seems to be between GrLivArea and SalePrice features**

# In[9]:


train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)


# In[10]:


plt.subplots(figsize=(20,12))
sns.heatmap(train.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1, fmt='.1f',square = True)


# In[11]:


f, ax = plt.subplots(figsize=(18,9))
sns.distplot(train['SalePrice'], color="red")
ax.set(xlabel="SalePrice")
ax.set(title="Distribution of SalePrice Feature")
plt.show()


# #### **The distribution of the SalePrice feature seems not like Normal Distribution. To convert the distribution of the SalePrice Feature to Normal Distribution, np.log1p transformation function in numpy was applied**

# In[12]:


train['SalePrice'] = np.log1p(train['SalePrice'])

f, ax = plt.subplots(figsize=(18,9))
sns.distplot(train['SalePrice'], color="navy")
ax.set(ylabel="Price Frequencies")
ax.set(xlabel="SalePrice")
ax.set(title="Distribution of SalePrice Feature")
plt.show()


# #### **Now the distribution of the SalePrice feature is much more fitted for Normal Distribution**

# In[13]:


f, ax = plt.subplots(figsize=(18,9))
sns.distplot(train['LotFrontage'], color="red")
ax.set(xlabel="LotFrontage")
ax.set(title="Distribution of Lot Frontage Feature")
plt.show()


# #### **The distribution of the LotFrontage feature seems not like Normal Distribution. To convert the distribution of the LotFrontage Feature to Normal Distribution, np.log1p transformation function in numpy was applied**

# In[14]:


train['LotFrontage'] = np.log1p(train['LotFrontage'])

f, ax = plt.subplots(figsize=(18,9))
sns.distplot(train['LotFrontage'], color="navy")
ax.set(xlabel="LotFrontage")
ax.set(title="Distribution of LotFrontage Feature")
plt.show()


# #### **Now the distribution of the LotFrontage feature is much more fitted for Normal Distribution**

# In[15]:


f, ax = plt.subplots(figsize=(18,9))
sns.distplot(train['LotArea'], color="red")
ax.set(xlabel="LotArea")
ax.set(title="Distribution of Lot Area Feature")
plt.show()


# #### **The distribution of the LotArea feature seems not like Normal Distribution. To convert the distribution of the LotArea Feature to Normal Distribution, np.log1p transformation function in numpy was applied**

# In[16]:


train['LotArea'] = np.log1p(train['LotArea'])

f, ax = plt.subplots(figsize=(18,9))
sns.distplot(train['LotArea'], color="navy")
ax.set(xlabel="LotArea")
ax.set(title="Distribution of LotArea Feature")
plt.show()


# #### **Now the distribution of the LotArea feature is much more fitted for Normal Distribution**

# In[17]:


f, ax = plt.subplots(figsize=(18,9))
sns.distplot(train['1stFlrSF'], color="red")
ax.set(xlabel="1stFlrSF")
ax.set(title="1stFlrSF")
plt.show()


# #### **The distribution of the 1stFlrSF feature seems not like Normal Distribution. To convert the distribution of the 1stFlrSF Feature to Normal Distribution, np.log1p transformation function in numpy was applied**

# In[18]:


train['1stFlrSF'] = np.log1p(train['1stFlrSF'])

f, ax = plt.subplots(figsize=(18,9))
sns.distplot(train['1stFlrSF'], color="navy")
ax.set(xlabel="1stFlrSF")
ax.set(title="Distribution of 1stFlrSF Feature")
plt.show()


# #### **Now the distribution of the 1stFlrSF feature is much more fitted for Normal Distribution**

# In[19]:


f, ax = plt.subplots(figsize=(18,9))
sns.distplot(train['GrLivArea'], color="red")
ax.set(xlabel="GrLivArea")
ax.set(title="GrLivArea")
plt.show()


# #### **The distribution of the GrLivArea feature seems not like Normal Distribution. To convert the distribution of the GrLivArea Feature to Normal Distribution, np.log1p transformation function in numpy was applied**

# In[20]:


train['GrLivArea'] = np.log1p(train['GrLivArea'])

f, ax = plt.subplots(figsize=(18,9))
sns.distplot(train['GrLivArea'], color="navy")
ax.set(xlabel="GrLivArea")
ax.set(title="Distribution of GrLivArea Feature")
plt.show()


# #### **Now the distribution of the GrLivArea feature is much more fitted for Normal Distribution**

# In[21]:


ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
df_new = pd.concat((train, test)).reset_index(drop=True)
df_new.drop(['SalePrice'], axis=1, inplace=True)


# ## HANDLING MISSING VALUES

# In[22]:


df_new.isna().sum()


# In[23]:


msno.matrix(df_new)


# In[24]:


df_new["PoolQC"] = df_new["PoolQC"].fillna("Not Available")
df_new["MiscFeature"] = df_new["MiscFeature"].fillna("Not Available")
df_new["Alley"] = df_new["Alley"].fillna("Not Available")
df_new["FireplaceQu"] = df_new["FireplaceQu"].fillna("Not Available")
df_new["LotFrontage"] = df_new.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
df_new["MasVnrType"] = df_new["MasVnrType"].fillna("Not Available")
df_new["Fence"] = df_new["Fence"].fillna("Not Available")
df_new['SaleType'] = df_new['SaleType'].fillna(df_new['SaleType'].mode()[0])
df_new['MSSubClass'] = df_new['MSSubClass'].fillna("Not Available")
df_new = df_new.drop(['Utilities'], axis=1)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    df_new[col] = df_new[col].fillna('Not Available')
    
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    df_new[col] = df_new[col].fillna('Not Available')


# In[25]:


for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    df_new[col] = df_new[col].fillna(0)


# In[26]:


for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    df_new[col] = df_new[col].fillna(0)


# In[27]:


df_new["MasVnrArea"] = df_new["MasVnrArea"].fillna(0)


# In[28]:


df_new['MSZoning'] = df_new['MSZoning'].fillna(df_new['MSZoning'].mode()[0])
df_new["Functional"] = df_new["Functional"].fillna("Typ")
df_new['Electrical'] = df_new['Electrical'].fillna(df_new['Electrical'].mode()[0])
df_new['KitchenQual'] = df_new['KitchenQual'].fillna(df_new['KitchenQual'].mode()[0])
df_new['Exterior1st'] = df_new['Exterior1st'].fillna(df_new['Exterior1st'].mode()[0])
df_new['Exterior2nd'] = df_new['Exterior2nd'].fillna(df_new['Exterior2nd'].mode()[0])


# ## FEATURE ENGINEERING

# In[29]:


df_new['MSSubClass'] = df_new['MSSubClass'].apply(str)
df_new['OverallCond'] = df_new['OverallCond'].astype(str)
df_new['YrSold'] = df_new['YrSold'].astype(str)
df_new['MoSold'] = df_new['MoSold'].astype(str)


# In[30]:


cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

le = LabelEncoder()

for i in cols:
    df_new[i] = le.fit_transform(df_new[[i]])


# In[31]:


numeric_feats = df_new.dtypes[df_new.dtypes != "object"].index

skewed_feats = df_new[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for i in skewed_features:
    df_new[i] = boxcox1p(df_new[i], lam)


# In[32]:


df_new = pd.get_dummies(df_new)
print(df_new.shape)
train = df_new[:ntrain]
test = df_new[ntrain:]
print(train.shape)
print(test.shape)


# ## REGRESSION MODELS

# In[33]:


gbr = GradientBoostingRegressor(n_estimators=6000,
                                learning_rate=0.01,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=42) 


# In[34]:


xgbr = XGBRegressor(learning_rate=0.01,
                       n_estimators=8000,
                       max_depth=4,
                       min_child_weight=0,
                       gamma=0.6,
                       subsample=0.7,
                       colsample_bytree=0.7,
                       nthread=-1,
                       scale_pos_weight=1,
                       seed=27,
                       reg_alpha=0.00006,
                       random_state=42)


# In[35]:


lgbm = LGBMRegressor(objective='regression', 
                       num_leaves=6,
                       learning_rate=0.01, 
                       n_estimators=6000,
                       max_bin=200, 
                       bagging_fraction=0.8,
                       bagging_freq=4, 
                       bagging_seed=8,
                       feature_fraction=0.2,
                       feature_fraction_seed=8,
                       min_sum_hessian_in_leaf = 11,
                       verbose=-1,
                       random_state=42)


# In[36]:


rfr = RandomForestRegressor(n_estimators=7000,
                          max_depth=15,
                          min_samples_split=5,
                          min_samples_leaf=5,
                          max_features=None,
                          oob_score=True,
                          random_state=42)


# In[37]:


kf = KFold(n_splits = 5, shuffle = True, random_state = 123).get_n_splits(train.values)


# In[38]:


gbr_mse_cv_scores = - cross_val_score(gbr, train.values,y_train,cv = kf, scoring = 'neg_mean_squared_error',n_jobs = -1)
gbr_rmse_score = np.sqrt(gbr_mse_cv_scores)

xgbr_mse_cv_scores = - cross_val_score(xgbr, train.values,y_train,cv = kf, scoring = 'neg_mean_squared_error',n_jobs = -1)
xgbr_rmse_score = np.sqrt(xgbr_mse_cv_scores)

lgbm_mse_cv_scores = - cross_val_score(lgbm, train.values,y_train,cv = kf, scoring = 'neg_mean_squared_error',n_jobs = -1)
lgbm_rmse_score = np.sqrt(lgbm_mse_cv_scores)

rfr_mse_cv_scores = - cross_val_score(rfr, train.values,y_train,cv = kf, scoring = 'neg_mean_squared_error',n_jobs = -1)
rfr_rmse_score = np.sqrt(rfr_mse_cv_scores)

print('Gradient Boosting Regressor CV RMSE Score :', gbr_rmse_score.mean())
print('XGBoost Regressor CV RMSE Score :', xgbr_rmse_score.mean())
print('LightGBM Regressor CV RMSE Score :', lgbm_rmse_score.mean())
print('Random Forest Regressor CV RMSE Score :', rfr_rmse_score.mean())


# In[39]:


gbr.fit(train, y_train)
gbr_train_pred = gbr.predict(train)
gbr_pred = np.expm1(gbr.predict(test.values))
rmse_gbr = np.sqrt(mean_squared_error(y_train,gbr_train_pred))
print(rmse_gbr)


# In[40]:


xgbr.fit(train,y_train)
xgbr_train_pred = xgbr.predict(train)
xgbr_pred = np.expm1(xgbr.predict(test.values))
rmse_xgbr = np.sqrt(mean_squared_error(y_train,xgbr_train_pred))
print(rmse_xgbr)


# In[41]:


lgbm.fit(train, y_train)
lgbm_train_pred = lgbm.predict(train)
lgbm_pred = np.expm1(lgbm.predict(test.values))
rmse_lgbm = np.sqrt(mean_squared_error(y_train,lgbm_train_pred))
print(rmse_lgbm)


# In[42]:


rfr.fit(train, y_train)
rfr_train_pred = rfr.predict(train)
rfr_pred = np.expm1(rfr.predict(test.values))
rmse_rfr = np.sqrt(mean_squared_error(y_train,rfr_train_pred))
print(rmse_rfr)


# In[43]:


models = ['Gradient Boosting', 'XGBoost', 'LightGBM', 'Random Forest']

rmse_scores = [rmse_gbr, rmse_xgbr, rmse_lgbm, rmse_rfr]

model_scores = pd.DataFrame({
    
    'Regression Models' : models,
    'RMSE Scores' : rmse_scores
    
})

results = model_scores.sort_values('RMSE Scores', ascending = True, ignore_index= True)
results


# In[44]:


sns.catplot(x = 'Regression Models', y = 'RMSE Scores', order = ['LightGBM', 'Gradient Boosting', 'Random Forest', 'XGBoost'],
           data = results, kind='point', color = 'Navy')
plt.title('RMSE Scores of Advanced Regression Models')
plt.xticks(rotation = 90)
plt.show()


# In[45]:


results.to_csv('final_submission.csv', index = 'None')


# ## CONCLUSION

# **Based on the modelling with advanced regression algorithms, LightGBM seems to have the minimum RMSE score which makes it the best for predictions in this dataset.**
