#!/usr/bin/env python
# coding: utf-8

# # IMPACT OF FEATURE ENGINEERING
# Author: [Murat Cihan Sorkun](https://www.linkedin.com/in/murat-cihan-sorkun/) 

# This notebook contains feature engineering in addition to the workflow I shared before (see the link below) for predicting house prices. 
# 
# [Simple Workflow: (LB: 0.12426)](https://www.kaggle.com/sorkun/house-prices-simple-workflow-lb-0-12426) 
# 
# 
# The workflow consists of following 5 steps:
# 
# 1. <b>Feature Engineering:</b> 
# * **Analyis of each feature:** I checked every feature with the function below (*analyse_feature()*). 
# * **Filling null values:** I filled the null values based on the analysis.
# * **Fixing outliers:** I softened the extreme values based on the analysis.
# * **Feature extraction:** I extracted Age form of features from Year based features. Also created some total value features.
# * **Feature removal:** I dropped some features which has no significanse based on the analysis.
# 2. <b>Transforming Categorical Variables:</b> I converted all categorical calumns into one hot encodings.
# 3. <b>Log Level Transformation:</b>. I converted all columns and target column into log level (To reduce skewness)
# 4. <b>Training Different Models:</b>. I trained 8 models by mostly-used algorithms
# 5. <b>Blending Models:</b> I simply averaged 4 models (2 tree-based AND 2 linear Models) 
# 
# Adding feature engineering improved the final score around 0.06, which leads to top 1% in the final board.

# In[1]:


import numpy as np 
import pandas as pd 

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from scipy.stats import skew

from sklearn.linear_model import Ridge, ElasticNet,  Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xg

import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

data_b = data.drop(['Id'], axis=1) 
test_b = test.drop(['Id'], axis=1) 
data_b.head()


# # 1.Feature Engineering:

# In[4]:


# Analyse missing Values
# data.info()
data_b.isnull().sum().sort_values(ascending=False).head(20)
# test_b.isnull().sum().sort_values(ascending=False).head(34)


# In[5]:


# I analysed all features one-by-one using the following function
def analyse_feature(feature_name):
#     data_c[feature_name].hist()
#     plt.show()
    sns.jointplot(data=data_b, y="SalePrice", x=feature_name)
    plt.show()
    print("\n****INFO****")
    print(data_b[feature_name].describe())
    print("\n****VALUE COUNTS****")
    print(data_b[feature_name].value_counts())
    print("\n****VALUE AVG SALE PRICE****")
    print(data_b.groupby(feature_name)['SalePrice'].mean())
    if data_b[feature_name].dtype!="O":
        print("\nSkewness:",str(skew(data_b[feature_name])))

#     test_c[feature_name].hist()

    print("\n****TEST INFO****")
    print(test_b[feature_name].describe())
    print("\n****VALUE COUNTS****")
    print(test_b[feature_name].value_counts())
    
    print("\nOnly in Train: "+str(list(set(data_b[feature_name].value_counts().index.values) - set(test_b[feature_name].value_counts().index.values))))
    print("Only in Test: "+ str(list(set(test_b[feature_name].value_counts().index.values) - set(data_b[feature_name].value_counts().index.values))))
    
analyse_feature("MSZoning")


# In[6]:


#AFTER ANALYSING EACH COLUMN -> FILL MISSING VALUES
test_b['MSZoning']=test_b['MSZoning'].fillna("C (all)") 
test_b['GarageCars']=test_b['GarageCars'].fillna(0) 
test_b['GarageArea']=test_b['GarageArea'].fillna(0) 
test_b['Functional']=test_b['Functional'].fillna("Typ") 
test_b['SaleType']=test_b['SaleType'].fillna("WD") 
test_b['SaleCondition']=test_b['SaleCondition'].fillna("Normal")
data_b['Fence']=data_b['Fence'].fillna("None") 
test_b['Fence']=test_b['Fence'].fillna("None") 
data_b['Electrical'] = data_b['Electrical'].fillna("SBrkr")
data_b['FireplaceQu'] = data_b['FireplaceQu'].fillna("None")
test_b['FireplaceQu'] = test_b['FireplaceQu'].fillna("None")
data_b['GarageType'] = data_b['GarageType'].fillna("None")
test_b['GarageType'] = test_b['GarageType'].fillna("None")
data_b['GarageQual'] = data_b['GarageQual'].fillna("None")
test_b['GarageQual'] = test_b['GarageQual'].fillna("None")
data_b['GarageCond'] = data_b['GarageCond'].fillna("None")
test_b['GarageCond'] = test_b['GarageCond'].fillna("None")
data_b['GarageFinish'] = data_b['GarageFinish'].fillna("None")
test_b['GarageFinish'] = test_b['GarageFinish'].fillna("None")
test_b['Exterior1st'] = test_b['Exterior1st'].fillna("VinylSd")
test_b['Exterior2nd']= test_b['Exterior2nd'].fillna("VinylSd")
data_b['MasVnrType'] = data_b['MasVnrType'].fillna("None")
test_b['MasVnrType'] = test_b['MasVnrType'].fillna("None")
data_b['MasVnrArea'] = data_b['MasVnrArea'].fillna(0)
test_b['MasVnrArea'] = test_b['MasVnrArea'].fillna(0)
test_b['BsmtHalfBath']=test_b['BsmtHalfBath'].fillna(0) 
test_b['BsmtFullBath']=test_b['BsmtFullBath'].fillna(0) 
test_b['KitchenQual']=test_b['KitchenQual'].fillna("Gd") 
test_b['TotalBsmtSF']=test_b['TotalBsmtSF'].fillna(0) 
test_b['BsmtUnfSF']=test_b['BsmtUnfSF'].fillna(0)
test_b['BsmtFinSF1']=test_b['BsmtFinSF1'].fillna(0) 
test_b['BsmtFinSF2']= test_b['BsmtFinSF2'].fillna(0) 
test_b['BsmtQual']=test_b['BsmtQual'].fillna("None") 
data_b['BsmtQual']=data_b['BsmtQual'].fillna("None")
test_b['BsmtCond']= test_b['BsmtCond'].fillna("None")
data_b['BsmtCond']= data_b['BsmtCond'].fillna("None")
test_b['BsmtExposure']= test_b['BsmtExposure'].fillna("None") 
data_b['BsmtExposure']= data_b['BsmtExposure'].fillna("None") 
test_b['Utilities']= test_b['Utilities'].fillna("AllPub") 
data_b['GarageYrBlt'] = data_b['GarageYrBlt'].fillna(1895)
test_b['GarageYrBlt'] = test_b['GarageYrBlt'].fillna(1895)


# In[7]:


#AFTER ANALYSING EACH COLUMN -> Fixing Outliers
test_b.loc[test.GarageYrBlt==2207,'GarageYrBlt'] = 2007 
test_b.loc[test_b.LotArea==1533,'LotFrontage'] = 21 
data_b.loc[data_b.LotArea>40000,'LotArea'] = 40000
test_b.loc[test_b.LotArea>40000,'LotArea'] = 40000
data_b.loc[data_b.LotFrontage>150,'LotFrontage'] = 150
test_b.loc[test_b.LotFrontage>150,'LotFrontage'] = 150
data_b.loc[data_b["1stFlrSF"]>3000,'1stFlrSF'] = 3000
test_b.loc[test_b["1stFlrSF"]>3000,'1stFlrSF'] = 3000
data_b.loc[data_b["GrLivArea"]>4000,'GrLivArea'] = 4000
test_b.loc[test_b["GrLivArea"]>4000,'GrLivArea'] = 4000
data_b.loc[data_b["TotRmsAbvGrd"]>12,'TotRmsAbvGrd'] = 12
test_b.loc[test_b["TotRmsAbvGrd"]>12,'TotRmsAbvGrd'] = 12
data_b.loc[data_b["BsmtFinSF1"]>2200,'BsmtFinSF1'] = 2200
test_b.loc[test_b["BsmtFinSF1"]>2200,'BsmtFinSF1'] = 2200
data_b.loc[data_b["TotalBsmtSF"]>2500,'TotalBsmtSF'] = 2500
test_b.loc[test_b["TotalBsmtSF"]>2500,'TotalBsmtSF'] = 2500
data_b.loc[data_b["GarageCars"]>3,'GarageCars'] = 3
test_b.loc[test_b["GarageCars"]>3,'GarageCars'] = 3
data_b.loc[data_b["GarageArea"]>1000,'GarageArea'] = 1000
test_b.loc[test_b["GarageArea"]>1000,'GarageArea'] = 1000

#Simple Linear Reg to fill missin LotFrontage values
test_b.loc[test_b["LotFrontage"].isnull(),["LotFrontage"]]=(test_b.loc[test_b["LotFrontage"].isnull(),["LotArea"]]*0.00885-15.17).values
data_b.loc[data_b["LotFrontage"].isnull(),["LotFrontage"]]=(data_b.loc[data_b["LotFrontage"].isnull(),["LotArea"]]*0.00885-15.17).values


# In[8]:


#Feature Extraction (Converting Years to Ages)
data_b["Age"] = 2011 - data_b["YearBuilt"]
test_b["Age"] = 2011 - test_b["YearBuilt"]

data_b["RemodAfter"] = data_b["YearRemodAdd"] - data_b["YearBuilt"]
data_b["AgeRemodAdd"] = 2011 - data_b["YearRemodAdd"]

test_b["RemodAfter"] = test_b["YearRemodAdd"] - test_b["YearBuilt"]
test_b.loc[test_b.RemodAfter<0,'RemodAfter'] = 0
test_b["AgeRemodAdd"] = 2011 - test_b["YearRemodAdd"]

data_b["Age_Sold"] = data_b["YrSold"] - data_b["YearBuilt"] 
test_b["Age_Sold"] = test_b["YrSold"] - test_b["YearBuilt"]
test_b.loc[test_b.Age_Sold<0,'Age_Sold'] = 0

data_b["Sold_before"] = 2011 - data_b["YrSold"]
test_b["Sold_before"] = 2011 - test_b["YrSold"] 

data_b['Age_Garage'] = 2011 - data_b['GarageYrBlt']
test_b['Age_Garage'] = 2011 - test_b['GarageYrBlt']
data_b['Age_Garage_Sold'] = data_b["YrSold"] - data_b['GarageYrBlt']
test_b['Age_Garage_Sold'] = test_b["YrSold"] - test_b['GarageYrBlt']

data_b['TotalPorch']=data_b['EnclosedPorch'] + data_b['OpenPorchSF'] + data_b['ScreenPorch'] + data_b['3SsnPorch'] 
test_b['TotalPorch']=test_b['EnclosedPorch'] + test_b['OpenPorchSF'] + test_b['ScreenPorch'] + test_b['3SsnPorch'] 

data_b['TotalBath'] = data_b['FullBath']+ data_b['HalfBath']+ data_b['BsmtFullBath']+ data_b['BsmtHalfBath']
test_b['TotalBath'] = test_b['FullBath']+ test_b['HalfBath']+ test_b['BsmtFullBath']+ test_b['BsmtHalfBath']


# In[9]:


#Update some values that is not exist one of two datasets
test_b['MSSubClass'] = test_b['MSSubClass'].replace(["150"], ["160"])
data_b['HouseStyle'] = data_b['HouseStyle'].replace(["2.5Fin"], ["2Story"])
data_b['Exterior1st'] = data_b['Exterior1st'].replace(['Stone', 'ImStucc'],['CemntBd', 'Stucco'])
data_b['Electrical'] = data_b['Electrical'].replace(["Mix","FuseF","FuseP"], ["SBrkr","FuseA","FuseA"])
test_b['Electrical'] = test_b['Electrical'].replace(["Mix","FuseF","FuseP"], ["SBrkr","FuseA","FuseA"])
test_b['BsmtCond'] = test_b['BsmtCond'].replace(["Ex"], ["Gd"])
# data_b['BsmtCond'] = data_b['BsmtCond'].replace(["Po"], ["None"])
data_b['GarageQual'] = data_b['GarageQual'].replace(["Ex"], ["Gd"])
data_b['GarageCond'] = data_b['GarageCond'].replace(["Ex"], ["Gd"])
test_b['GarageCond'] = test_b['GarageCond'].replace(["Ex"], ["Gd"])


# In[10]:


y = data_b.pop("SalePrice")


# In[11]:


# Check Missing values from training set
data_b.isnull().sum().sort_values(ascending=False).head(10)


# In[12]:


# Check Missing values from test set
test_b.isnull().sum().sort_values(ascending=False).head(10)


# In[13]:


#Drop Missing Columns
data_b = data_b.dropna(axis=1) 
test_b = test_b.dropna(axis=1)

#AFTER ANALYSING EACH COLUMN ->  Drop insignificant columns
data_b = data_b.drop(["YearBuilt","YearRemodAdd","Street","Condition2","LowQualFinSF","RoofMatl","Exterior2nd","OpenPorchSF","EnclosedPorch",
                    "3SsnPorch","ScreenPorch","PoolArea","MiscVal","YrSold","Utilities","Heating","KitchenAbvGr","GarageYrBlt"], axis=1) 
test_b = test_b.drop(["YearBuilt","YearRemodAdd","Street","Condition2","LowQualFinSF","RoofMatl","Exterior2nd","OpenPorchSF","EnclosedPorch",
                    "3SsnPorch","ScreenPorch","PoolArea","MiscVal","YrSold","Utilities","Heating","KitchenAbvGr","GarageYrBlt"], axis=1) 


# # 2.Transforming Categorical Variables:

# In[14]:


#Split Categorical / Numeric
cols = data_b.columns
numeric_columns, categorical_columns = [], []
for i in range(len(cols)):
    if data_b[cols[i]].dtypes == 'O':
        categorical_columns.append(cols[i])
    else:
        numeric_columns.append(cols[i])

#Show Unique items in categorical variables
category_analysis = pd.DataFrame(categorical_columns, columns = ["Feature"])
unique_values = []
unique_counts = []
for col in categorical_columns:
    unique_values.append(data_b[col].unique())
    unique_counts.append(len(data_b[col].unique()))
    
category_analysis["Categories"] = unique_values
category_analysis["Number"] = unique_counts
category_analysis


# In[15]:


# Convert Categorical variables into ONE HOT ENCODINGs
data_b = pd.get_dummies(data_b)
test_b = pd.get_dummies(test_b)


# # 3. Log Level Transformation:

# In[16]:


# Skewing all features (Log level transform)
# skewed_feats = data_b[numeric_columns].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewed_feats = data_b.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)

#New data 
data_b_skewed = data_b.copy()
test_b_skewed = test_b.copy()

skewness = skewness[abs(skewness) > 1]
print("There are {} skewed numerical features to Log1p transform".format(skewness.shape[0]))

skewed_features = skewness.index
for feat in skewed_features:
    data_b_skewed[feat] = np.log1p(data_b_skewed[feat])
    test_b_skewed[feat] = np.log1p(test_b_skewed[feat])


# In[17]:


print("\nOnly in Train: "+ str(list(set(data_b_skewed.columns) - set(test_b_skewed.columns))))
print("Only in Test: "+ str(list(set(test_b_skewed.columns) - set(data_b_skewed.columns))))


# In[18]:


# Convert Target also Log level
y_log = np.log1p(y)


# In[19]:


#Convert -INF to Zero
test_b_skewed=test_b_skewed.replace(-np.Inf, 0)


# # 4. Training Different Models:

# In[20]:


#Run Cross validation for each Model 
def run_cvs(X,y):
    
    baseline = ElasticNet(random_state=0,max_iter=10e7,alpha=0.0003)
    baseline_score = cross_val_score(baseline, X, y, cv=10)
    print("ENet avg:",np.mean(baseline_score))
    
    baseline = Ridge(alpha = 1, random_state=0)
    baseline_score = cross_val_score(baseline, X, y, cv=10)
    print("Ridge avg:",np.mean(baseline_score))   
    
    baseline = Lasso(alpha = 0.0001,random_state=0)
    baseline_score = cross_val_score(baseline, X, y, cv=10)
    print("Lasso avg:",np.mean(baseline_score))
    
    baseline = KernelRidge(alpha=0.1)
    baseline_score = cross_val_score(baseline, X, y, cv=10)
    print("KRR avg:",np.mean(baseline_score))

    baseline = lgb.LGBMRegressor(learning_rate=0.01,num_leaves=4,n_estimators=2000, random_state=0)
    baseline_score = cross_val_score(baseline, X, y, cv=10)
    print("LGBM avg:",np.mean(baseline_score))

    baseline = xg.XGBRegressor(learning_rate=0.01,n_estimators=2000, subsample=0.7,colsample_bytree=0.7,random_state=0)
    baseline_score = cross_val_score(baseline, X, y, cv=10)
    print("XGB avg:",np.mean(baseline_score))
    
    baseline = CatBoostRegressor(random_state=0,verbose=0)
    baseline_score = cross_val_score(baseline, X, y, cv=10)
    print("CatB avg:",np.mean(baseline_score))

    baseline = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.02,max_depth=4, max_features='sqrt',
                min_samples_leaf=15, min_samples_split=50,loss='huber', random_state = 0)
    baseline_score = cross_val_score(baseline, X, y, cv=10)
    print("GBR avg:",np.mean(baseline_score))
    
run_cvs(data_b_skewed,y_log) 


# # 5. Blending Models:

# In[21]:


#Combine 4 models (2 tree-based AND 2 linear Models) 
def make_submission(X_train, y_train, X_test):    
    sub_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv', index_col = "Id")
    
    ridge = Ridge(alpha = 1, random_state=0).fit(X_train,y_train)
    ridge_preds_log=ridge.predict(X_test)
    
    lasso = Lasso(alpha = 0.0001,random_state=0).fit(X_train,y_train)
    lasso_preds_log=lasso.predict(X_test)

    catB = CatBoostRegressor(random_state=0,verbose=0).fit(X_train,y_train)
    catB_preds_log=catB.predict(X_test)

    xgb = xg.XGBRegressor(learning_rate=0.01,n_estimators=2000, subsample=0.7,colsample_bytree=0.7,random_state=0).fit(X_train,y_train)
    xgb_preds_log=xgb.predict(X_test)
    
    catb_xbr_lasso_ridge_mean_preds_log=(catB_preds_log+ridge_preds_log+lasso_preds_log+xgb_preds_log)/4
    sub_df['SalePrice'] = np.exp(catb_xbr_lasso_ridge_mean_preds_log)-1
    sub_df.to_csv("submission.csv")
    
make_submission(data_b_skewed,y_log,test_b_skewed)    


# **Hey!** If you find this notebook useful, an upvote is appreciated!
