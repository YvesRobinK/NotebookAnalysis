#!/usr/bin/env python
# coding: utf-8

# <h1 class="list-group-item list-group-item-action active" data-toggle="list" style='color:white; background:#0096FF' role="tab" aria-controls="home"><br><center>House Price prediction</center></h1>

# <a id="top"></a>
# 
# <div class="list-group" id="list-tab" role="tablist">
# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='color:white; background:#0096FF' role="tab" aria-controls="home"><br><center>Quick Navigation</center></h3>
# 
# * [House Price Prediction](#top)
# * [Feature Engineering](#2)
#     * [Handling Outliers](#2.1)
# * [Fix Missing value](#3)
# * [Exploratory Data Analysis](#4)
# * [One Hot Encoding](#5)
# * [Modelling](#6)
#     * [Linear Regression with L1 Regularization](#6.1)
#     * [Linear Regression with L2 Regularization](#6.2)
#     * [XGBoost](#6.3)
#     * [LightGBM](#6.4)
# * [👉Ensemble- Weighted Regression Models🤔](#7)

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

import warnings
warnings.filterwarnings('ignore')

sns.set(style = 'darkgrid', font_scale = 1.6)


# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='color:white; background:#0096FF' role="tab" aria-controls="home"><br><center>Reading Data</center></h3>

# In[2]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print('train shape  {}'.format(train.shape))
print('test shape  {}'.format(test.shape))
display(train.head())
display(test.head())


# In[3]:


test_copy = test.copy()


# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='color:white; background:#0096FF' role="tab" aria-controls="home"><br><center>Handling Outliers</center></h3><a id=2.1></a>

# <h3 class="list-group-item list-group-item-action active" style='color:#0096FF ; background:white' role="tab" aria-controls="home">SalePrice Distribution</h3>

# In[4]:


plt.figure()
train['SalePrice'].hist(bins=20)
plt.title('SalePrice' + ' before transformation')
plt.show()


# The above plot is right skewed ,so we apply log transform to make the target variable less skew and follow Gaussian distibution , which will help to detect outliers and remove it.

# In[5]:


from scipy import stats


# In[6]:


train['SalePrice'] = np.log(train['SalePrice'])

train['z_score_target'] = np.abs(stats.zscore(train['SalePrice']))
train = train.loc[train['z_score_target'] < 3].reset_index(drop=True)
del train['z_score_target']

plt.figure()
train['SalePrice'].hist(bins=20)
plt.title('SalePrice' + ' after transformation')
plt.show()


# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='color:white; background:#0096FF' role="tab" aria-controls="home"><br><center>Feature Engineering</center></h3><a id=2></a>

# we categorize features into three groups
# - Categorical
# - Nominal
# - Numerical

# <h3 class="list-group-item list-group-item-action active" style='color:#0096FF ; background:white' role="tab" aria-controls="home">Categorical feature</h3>
# 
# Consider Object Dtype columns as Categorical features 

# In[7]:


categorical_features = ["Alley", 'MSSubClass', 'MoSold', 'MSZoning', 'LandContour',
                        'LotConfig', 'Neighborhood', 'Condition1', 'Condition2',
                        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
                        'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',
                        'Heating', 'CentralAir', 'Electrical', 'GarageType',
                        'GarageFinish', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

print(len(categorical_features))


# In[8]:


nominal_features = ["BedroomAbvGr", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
                    "BsmtFullBath", "BsmtHalfBath", "BsmtQual", "ExterCond", "ExterQual",
                    "Fireplaces", "FireplaceQu", "Functional", "FullBath", "GarageCars",
                    "GarageCond", "GarageQual", "HalfBath", "HeatingQC", "KitchenAbvGr",
                    "KitchenQual", "LandSlope", "LotShape", "PavedDrive", "PoolQC",
                    "Street", "Utilities", "OverallCond", "OverallQual", "TotRmsAbvGrd"]

ordinal_features = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
                      'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF','1stFlrSF',
                      '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF',
                      'OpenPorchSF','EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
                      'MiscVal', 'YrSold'] 
print('Nominal features length:\t',len(nominal_features),'\nOrdinal Features length:\t',len(ordinal_features))


# In[9]:


numerical_features = nominal_features + ordinal_features
all_features = nominal_features+ ordinal_features+categorical_features


# In[10]:


train = train[all_features + ['SalePrice']].copy()
test = test[all_features].copy()
display(train.head())


# In[11]:


train.info()


# In[12]:


train.describe().T


# In[13]:


print(f'Null values: {train.isnull().sum()}')


# In[14]:


nulls = pd.DataFrame(train.isna().sum().sort_values(ascending=False),columns=['null count'])
nulls


# <h3 class="list-group-item list-group-item-action active" style='color:#0096FF ; background:white' role="tab" aria-controls="home">Find the Duplicates</h3>
# 
# Check for Repeating values in the dataframe

# In[15]:


# Duplicated rows
print(f"{train.duplicated().sum()} duplicates")


# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='color:white; background:#0096FF' role="tab" aria-controls="home"><br><center>Fix Missing values</center></h3><a id=3></a>

# <h3 class="list-group-item list-group-item-action active" style='color:#0096FF ; background:white' role="tab" aria-controls="home">Missing value</h3>

# In[16]:


train.isna().sum()


# In[17]:


test.isna().sum()


# <h3 class="list-group-item list-group-item-action active" style='color:#0096FF ; background:white' role="tab" aria-controls="home">Imputing features</h3>

# In[18]:


# calculate number of rows that contain null values
nulls = train.shape[0]-train.dropna(axis = 0).shape[0]
nulls


# In[19]:


nums = ['BedroomAbvGr','BsmtFullBath','BsmtHalfBath',"BsmtUnfSF","TotalBsmtSF", 
        "BsmtFinSF1","BsmtFinSF2",'EnclosedPorch',"Fireplaces",'GarageArea',
        'GarageCars','HalfBath','KitchenAbvGr','LotFrontage','MasVnrArea','MiscVal',
       'OpenPorchSF','PoolArea','ScreenPorch','TotRmsAbvGrd','WoodDeckSF']


# <h3 class="list-group-item list-group-item-action active" style='color:#0096FF ; background:white' role="tab" aria-controls="home">Data Imputation</h3>

# In[20]:


for feature in train.columns:
    if feature in nums:
        train[feature].fillna(0,inplace=True)
    else:
        if feature in ["Alley","MasVnrType"]:
            train.loc[:,feature] = train.loc[:,feature].fillna("None")
        elif feature in ['BsmtQual',"MiscFeature","PoolQC",'BsmtCond',"BsmtExposure","BsmtFinType1","BsmtFinType2","Fence","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond"]:
            train.loc[:, feature] = train.loc[:, feature].fillna("No")
        elif feature in ["CentralAir","PavedDrive"]:
            train.loc[:, feature] = train.loc[:, feature].fillna("N")
        elif feature in ['Condition1','Condition2']:
            train.loc[:, feature] = train.loc[:, feature].fillna("Norm")
        elif feature in ["ExterCond","ExterQual","HeatingQC","KitchenQual"]:
            train.loc[:, feature] = train.loc[:, feature].fillna("TA")
        elif feature in ['LotShape']:
            train.loc[:, feature] = train.loc[:, feature].fillna("Reg")
        elif feature =="SaleCondition":
            train.loc[:, feature] = train.loc[:, feature].fillna("Normal")
        elif feature == "Utilities":
            train.loc[:, feature] = train.loc[:, feature].fillna("AllPub")
        


# In[21]:


for feature in test.columns:
    if feature in nums:
        test[feature].fillna(0,inplace=True)
    else:
        if feature in ["Alley","MasVnrType"]:
            test.loc[:,feature] = test.loc[:,feature].fillna("None")
        elif feature in ['BsmtQual',"MiscFeature","PoolQC",'BsmtCond',"BsmtExposure","BsmtFinType1","BsmtFinType2","Fence","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond"]:
            test.loc[:, feature] = test.loc[:, feature].fillna("No")
        elif feature in ["CentralAir","PavedDrive"]:
            test.loc[:, feature] = test.loc[:, feature].fillna("N")
        elif feature in ['Condition1','Condition2']:
            test.loc[:, feature] = test.loc[:, feature].fillna("Norm")
        elif feature in ["ExterCond","ExterQual","HeatingQC","KitchenQual"]:
            test.loc[:, feature] = test.loc[:, feature].fillna("TA")
        elif feature in ['LotShape']:
            test.loc[:, feature] = test.loc[:, feature].fillna("Reg")
        elif feature =="SaleCondition":
            test.loc[:, feature] = test.loc[:, feature].fillna("Normal")
        elif feature == "Utilities":
            test.loc[:, feature] = test.loc[:, feature].fillna("AllPub")
        elif feature == "SaleType":
            test.loc[:, feature] = test.loc[:, feature].fillna("WD")


# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='color:white; background:#0096FF' role="tab" aria-controls="home"><br><center>Exploratory Data Analysis (EDA)</center></h3><a id=4></a>

# 
# * Trying to find highly Correlated predictor videos with target value using Heatmap

# In[22]:


cor =  train.corr()
train.corr().SalePrice.sort_values(ascending=False)


# **Check the last column in the heatmap, to find how the predictor variables are correlated to the SalePrice target variable.**

# In[23]:


fig, axs = plt.subplots(1,1, figsize=(75,55))
sns.heatmap(cor,annot=True,fmt='.2f', cmap = 'coolwarm')


# **Therefore OverallQuall is the most highly correlated feature with the target variable SalePrice.**

# In[24]:


plt.figure(figsize = (10, 5))
sns.lineplot(data = train, x = 'OverallQual', y = 'SalePrice')
plt.show()


# <h3 class="list-group-item list-group-item-action active" style='color:#0096FF ; background:white' role="tab" aria-controls="home">Scatter plot of Ordinal features</h3>

# In[25]:


plt.figure(figsize = (25, 25))
for i, feature in enumerate(ordinal_features):
    plt.subplot(10, 3, i+1)
    sns.scatterplot(data = train, x = feature, y = 'SalePrice', color = 'blue')
plt.tight_layout()
plt.show()


# <h3 class="list-group-item list-group-item-action active" style='color:#0096FF ; background:white' role="tab" aria-controls="home">Remarks</h3>
# 
# * SalePrice vs. 1stFirSF and SalePrice vs. GrLivArea seem to follow a trend, which can be explained by saying that "As the prices increased, so does the area".
# 
# * SalePrice shows an unequal level of variance across most predictor(independent) variables - Heteroscedasticity. This is an issue multiple linear regression model. 

# <h3 class="list-group-item list-group-item-action active" style='color:#0096FF ; background:white' role="tab" aria-controls="home">Numerical features</h3>

# In[26]:


sns.scatterplot(data=train,x='GrLivArea', y='SalePrice')


# Samples with GrLivArea greater than 4000 are obviously outliers, so delete them from our training dataset

# In[27]:


train = train[train['GrLivArea'] <4000].reset_index(drop=True)


# **Remove skewed parameters**

# In[28]:


for feature in ordinal_features:
    if feature in ["YearBuilt", "YearRemodAdd", "YrSold"]:
        continue
    
    # if we had zero or negative value in the data, we add another column called has_zero_{feature} to the data and log non-zero values
    if (train[feature]<=0).sum()>0:
        # skip applying if we had not significant skewness
        if train.loc[train[feature]>0, feature].skew() < 0.5:
            continue
        train.loc[train[feature] > 0, feature] = np.log(train.loc[train[feature]>0, feature])

        test.loc[test[feature] > 0, feature] = np.log(test.loc[test[feature]>0, feature])

    # else we just apply log transformation
    else:
        # skip applying if we had not significant skewness
        if train[feature].skew() < 0.5:
            continue
        train[feature] = np.log(train[feature])
        test[feature] = np.log(test[feature])


# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='color:white; background:#0096FF' role="tab" aria-controls="home"><br><center>Assumptions of Regression</center></h3>
# 
# - Linearity <br>
# - Homoscedasticity (absence of Heteroscedasticity).<br>
# - Independence of Errors <br>
# - Multivariate Normality <br>
# - Low or No Multicollinearity. <br>

# The best way to solve multicollinearity from the above heatmap is to use regularization methods like Ridge or Lasso or ElasticNet. It is done further below in the notebook.

# <h3 class="list-group-item list-group-item-action active" style='color:#0096FF ; background:white' role="tab" aria-controls="home">Scaling Values</h3>

# In[29]:


from sklearn.preprocessing import RobustScaler
rob = RobustScaler()

train[ordinal_features] = rob.fit_transform(train[ordinal_features])
test[ordinal_features] =  rob.fit_transform(test[ordinal_features])


# In[30]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor as lgb


# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='color:white; background:#0096FF' role="tab" aria-controls="home"><br><center>One Hot Encoding</center></h3><a id=5></a>

# <h3 class="list-group-item list-group-item-action active" style='color:#0096FF ; background:white' role="tab" aria-controls="home">Encoding categorical variable</h3>

# In[31]:


data_train = train.copy()
data_test = test.copy()
ohe = pd.get_dummies(data_train, columns = categorical_features)
ohe_test = pd.get_dummies(data_test, columns = categorical_features)


# <h3 class="list-group-item list-group-item-action active" style='color:#0096FF ; background:white' role="tab" aria-controls="home">Ordinal Encoding</h3>

# In[32]:


from sklearn.preprocessing import OrdinalEncoder


# In[33]:


oe = OrdinalEncoder()
ohe[ordinal_features] = oe.fit_transform(ohe[ordinal_features])
ohe[nominal_features] = oe.fit_transform(ohe[nominal_features])

ohe_test[ordinal_features] = oe.fit_transform(ohe_test[ordinal_features])
ohe_test[nominal_features] = oe.fit_transform(ohe_test[nominal_features])


# **Concatenating Train and test dataframes**

# In[34]:


total_df = pd.concat([ohe,ohe_test],ignore_index=True)
display(total_df.tail())


# In[35]:


total_df.drop(columns=['MSSubClass_150'],inplace=True)


# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='color:white; background:#0096FF' role="tab" aria-controls="home"><br><center>Split Data </center></h3>

# In[36]:


# Return data to train and test
train_df = total_df[:len(ohe)]
display(train_df.head())
test_df = total_df[len(ohe):]
display(test_df.head())


# In[37]:


train_df['SalePrice'] = train['SalePrice']
train_df.shape,test_df.shape


# In[38]:


train_df.SalePrice.head()


# In[39]:


sns.histplot(train_df['SalePrice'], kde =True)


# In[40]:


x = train_df.drop(columns=['SalePrice'],axis=1)
y = train_df['SalePrice']


# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='color:white; background:#0096FF' role="tab" aria-controls="home"><br><center>Modelling </center></h3><a id=6></a>

# In[41]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV


# In[42]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1,random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1,random_state=42)


# In[43]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[44]:


test_f = test_df.drop(columns=['SalePrice'])
test_f = test_f.fillna(0)


# In[45]:


from sklearn.linear_model import Lasso,LinearRegression,Ridge,ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score


# <h3 class="list-group-item list-group-item-action active" style='color:#0096FF ; background:white' role="tab" aria-controls="home">Performance metrics</h3>

# In[46]:


def regression_metrics(y_test,y_pred):
    print('explained_variance: ', round(explained_variance_score(y_test,y_pred),4))  
    print('r2: ', round(r2_score(y_test,y_pred),4))
    print("MAE:\t",round(mean_absolute_error(y_test,y_pred),4))
    print("MSE:\t",round(mean_squared_error(y_test,y_pred),4))
    print("RMSE:\t",round(np.sqrt(mean_squared_error(y_test,y_pred)),4))


# <h3 class="list-group-item list-group-item-action active" style='color:#0096FF ; background:white' role="tab" aria-controls="home">Cross Validation - GridSearchCV</h3>

# In[47]:


def model_evaluate(model, param_grid, x_train, y_train, x_test,y_test, model_name, k_folds=5, scoring='neg_mean_squared_error', fit_parameters={}):

    model_cv = GridSearchCV(model, param_grid, cv=k_folds, verbose=False, scoring= scoring, refit=True)
    model_cv.fit(x_train, y_train, **fit_parameters)
    y_train_pred = model_cv.predict(x_train)
    y_test_pred = model_cv.predict(x_test)

    print('Mean Squared Error = ', np.abs(model_cv.score(x_train, y_train)))
    print("Training metrics:")
    regression_metrics(y_train ,y_train_pred)
    print("\nTesting metrics:")
    regression_metrics(y_test ,y_test_pred)
    
    return model_cv


# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='color:white; background:#0096FF' role="tab" aria-controls="home"><br><center>Linear Regression with L1 regularization - Lasso</center></h3><a id=6.1></a>

# In[48]:


lasso = Lasso(alpha =0.0005, random_state=20)
#param_grid = [{'alpha':[0.001, 0.005, 0.01, 0.05, 0.03, 0.1, 0.5, 1]}]
param_grid = [{'alpha':[0.0005]}]
lasso_model = model_evaluate(lasso, param_grid, x_train, y_train, x_test, y_test, 'Lasso',k_folds=5)
lasso_pred = np.exp(lasso_model.predict(test_f))


# In[49]:


submission_lasso = pd.DataFrame({'Id': test_copy.Id, 'SalePrice': lasso_pred})
submission_lasso.to_csv(path_or_buf = 'submission_lasso.csv', index = False)
pd.read_csv('submission_lasso.csv')


# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='color:white; background:#0096FF' role="tab" aria-controls="home"><br><center>Linear Regression with L2 regularization - Ridge</center></h3><a id=6.2></a>

# In[50]:


ridge = Ridge(random_state=20, max_iter=10000)
learning_rate = [0.0005,0.001, 0.005, 0.01, 0.05, 0.03, 0.1, 0.5,0.6, 1]
param_grid = [{'alpha': learning_rate}]

ridge_model = model_evaluate(ridge, param_grid, x_train, y_train, x_test, y_test, 'Ridge',k_folds=5)


# In[51]:


ridge_pred = np.exp(ridge_model.predict(test_f))
submission_ridge = pd.DataFrame({'Id': test_copy.Id, 'SalePrice': ridge_pred})
submission_ridge.to_csv(path_or_buf = 'submission_ridge.csv', index = False)
pd.read_csv('submission_ridge.csv')


# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='color:white; background:#0096FF' role="tab" aria-controls="home"><br><center>XGBoost</center></h3><a id=6.3></a>

# In[52]:


from xgboost import XGBRegressor as xgr
import xgboost as xgb


# In[53]:


xgr_model = xgr()
param_grid = {'learning_rate':[0.01],'max_depth':[6],'n_estimators':[1000],'min_child_weight':[0.5],'colsample_bytree':[0.5],'subsample':[0.5], 'eta':[0.1],'seed':[42]} 

model_xgr = model_evaluate(xgr_model, param_grid, x_train, y_train, x_test, y_test, 'XGBM',fit_parameters={'eval_set':[(x_val, y_val)], 'eval_metric':'rmse'})


# In[54]:


xgr_pred= np.exp(model_xgr.predict(test_f))


# XGBoost Output

# In[55]:


submission_xgb = pd.DataFrame({'Id': test_copy.Id, 'SalePrice': xgr_pred})
submission_xgb.to_csv(path_or_buf = 'submission_xgr.csv', index = False)
pd.read_csv('submission_xgr.csv')


# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='color:white; background:#0096FF' role="tab" aria-controls="home"><br><center>LightGBM</center></h3><a id=6.4></a>

# In[56]:


lgb_model = lgb()


# In[57]:


param_grid = {'learning_rate':[0.01], 'num_iterations': [10000], 'n_estimators': [25], 'num_leaves': [40],'colsample_bytree':[0.4], 'subsample': [0.4], 'max_depth': [6]} 

model_lgb = model_evaluate(lgb_model, param_grid, x_train, y_train, x_test, y_test, 'LGBM', fit_parameters={'eval_set':[(x_val, y_val)], 'eval_metric':'rmse'})


# In[58]:


lgb_pred= np.exp(model_lgb.predict(test_f))


# LightGBM Output

# In[59]:


prediction_lgb = model_lgb.predict(test_f)
submission_lgb = pd.DataFrame({'Id': test_copy.Id, 'SalePrice': np.exp(prediction_lgb)})
submission_lgb.to_csv(path_or_buf = 'submission_lgb.csv', index = False)
pd.read_csv('submission_lgb.csv')


# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='color:white; background:#0096FF' role="tab" aria-controls="home"><br><center>Ensembling - Weighted Regression Models🤔</center></h3><a id=7></a>

# **Calculating weighted output of Lasso , Ridge, XGBoost and LightGBM**
# 
# * The best model with good hyperparameter tuning could have been better. In this case Lasso(L1 regulariation with GridSearchCV) was the best model for me in terms of the score. Initially Lasso alone gave me the best score.
# 
# * Since I also performed model Analysis with other different models and had their outputs; as an experiment, I played around by adding different weights to those regression output. Thought process was to add small value(bias kind) to lasso's output and try to reduce the error. It did improve the score.
# 
# * Manually tried various combinations of weights to these outputs. 

# In[60]:


w1,w2,w3,w4 = 0.7,0.032,0.032,0.2


# In[61]:


ensemble = w1*lasso_pred + w2*ridge_pred + w3*xgr_pred + w4*lgb_pred


# **Submission**

# In[62]:


submission = pd.DataFrame()
submission['Id'] = test_copy.Id
submission['SalePrice'] = ensemble
submission.to_csv('submission.csv',index=False)
pd.read_csv('submission.csv')


# Please do upvote , if you liked it. Thanks in advance.  -`@tejasurya`
