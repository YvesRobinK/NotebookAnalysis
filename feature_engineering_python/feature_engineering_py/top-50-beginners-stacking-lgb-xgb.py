#!/usr/bin/env python
# coding: utf-8

# 
# ![main](https://www.dataquest.io/blog/content/images/2017/12/kaggle-amassachi-1.png)

# ### With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.
# 
# ## Practice Skills
# * Creative feature engineering 
# * Advanced regression techniques like random forest and gradient boosting

# ### Recommend : This tutorial doesn't cover EDA . If you are looking at this dataset for the first time , Visit the first part of this kernel , which takes care of the Explanatory Data Analysis.
# https://www.kaggle.com/niteshx2/kernel-housing-prices-using-dummy-vars

# ## Step 1 : Import Statements , you can use this code as it is.
# 
# (Output Shows files already present in the Environment (By Kaggle)

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from datetime import datetime
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Now , we load the required files 

# In[2]:


#features['MSSubClass'].dtype


# In[3]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# ## The training data has 1460 rows and 81 columns , testing data has 80 columns 
# * ### (*Target -> **SalePrice** column excluded , obviously , That's what you have to predict !!)*

# In[4]:


train.shape, test.shape


# ## A quick look at the train and test sets , trying to get some intution from the data :
# * ### Getting Any intution is REALLY hard , as there are more than 80 columns ! , that is what we need data scientist for. 

# In[5]:


train.head()


# In[6]:


test.head()


# ## Step 2 : Data processing 
# * ### (These are CRUCIAL , non optional steps . Real world data is even more noisy)
# * ### Not all data is 'Clean' enough to be used directly for mathematical models. We need to 'Clean' it first !
# 1. Remove Outliers 
# 2. Fill up NULL and missing data . Most models can't handle them.

# ...... Leaving out the ID column , as it doesn't add any information for our model. 

# In[7]:


train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)


# In[8]:


train['SalePrice'].hist(bins = 40)


# ### SalePrice is not uniformly distributed and is skewed towards the left . Therefore , we use log1p to remove the skewness .

# ### We take log as logs are used to respond to skewness towards large values; i.e., cases in which one or a few points are much larger than the bulk of the data. 

# In[9]:


train = train[train.GrLivArea < 4500]
train.reset_index(drop=True, inplace=True)
train["SalePrice"] = np.log1p(train["SalePrice"])
y = train['SalePrice'].reset_index(drop=True)


# ### As you can see below , the data is now more balanced , suitable for training and prediction purposes

# In[10]:


train['SalePrice'].hist(bins = 40)


# ### Storing ***SalePrice*** column seperately , as it is the 'Y' label / target that our model will learn to predict. Not to be stored in 'X' or features.

# In[11]:


train_features = train.drop(['SalePrice'], axis=1)
test_features = test
features = pd.concat([train_features, test_features]).reset_index(drop=True)


# ### Now , we are merging train and test datasets , so that we can handle *** 'noise'*** and ***missing data*** in the dataset . 

# In[12]:


features.shape


# In[13]:


#features['MSSubClass'].apply(str)
#features['MSSubClass']


# In[14]:


# Since these column are actually a category , using a numerical number will lead the model to assume
# that it is numerical , so we convert to string .
features['MSSubClass'] = features['MSSubClass'].apply(str)
features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)



## Filling these columns With most suitable value for these columns 
features['Functional'] = features['Functional'].fillna('Typ') 
features['Electrical'] = features['Electrical'].fillna("SBrkr") 
features['KitchenQual'] = features['KitchenQual'].fillna("TA") 
features["PoolQC"] = features["PoolQC"].fillna("None")



## Filling these with MODE , i.e. , the most frequent value in these columns .
features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0]) 
features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])


# In[15]:


### Missing data in GarageYrBit most probably means missing Garage , so replace NaN with zero . 

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    features[col] = features[col].fillna(0)

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    features[col] = features[col].fillna('None')

    
### Same with basement

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    features[col] = features[col].fillna('None')


# GroupBy explained here : https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
# 
# ### This code will filll the missing values with the mode (The frequently category appearing) By each MSsubclass:
# * Idea is that similar MSSubClasses will have similar MSZoning  

# In[16]:


features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))


#  ### Fill the remaining columns as **None**

# In[17]:


objects = []
for i in features.columns:
    if features[i].dtype == object:
        objects.append(i)
features.update(features[objects].fillna('None'))
print(objects)


# ## For missing values in numerical cols , we fillNa with 0.

# In[18]:


# We are still filling up missing values 
features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes:
        numerics.append(i)
features.update(features[numerics].fillna(0))
numerics[1:10]


# ![](https://blog.minitab.com/hubfs/Imported_Blog_Media/distribution_plot-1.jpg) 
# ### We are going to transform skewed columns
# ![.](https://blog.minitab.com/hubfs/Imported_Blog_Media/transformation.jpg)After transforming
# 
# Know more about  boxcox transform here : http://blog.minitab.com/blog/applying-statistics-in-quality-projects/how-could-you-benefit-from-a-box-cox-transformation

# In[19]:


numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics2 = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes:
        numerics2.append(i)
skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

for i in skew_index:
    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))


# ![](https://elitedatascience.com/wp-content/uploads/2018/05/Feature-Engineering-Banner-940px.jpg) 
# 
# 
# 
# ## STEP 3 : Feature Engineering 
# 
# 
# 
# 
# 
# 

# In[20]:


# Removing features that are not very useful . This can be understood only by doing proper EDA on data

features = features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)


# Adding new features . Make sure that you understand this. 

features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']
features['TotalSF']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']

features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +
                                 features['1stFlrSF'] + features['2ndFlrSF'])

features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +
                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))

features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +
                              features['EnclosedPorch'] + features['ScreenPorch'] +
                              features['WoodDeckSF'])


# In[21]:


## For ex, if PoolArea = 0 , Then HasPool = 0 too

features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# ## Now we have 86 columns , having added around 7 more to our data 

# In[22]:


features.shape


# ## Get_dummies converts Categorical data to numerical , as models don't work with Text data .
# * each category gets its different columns , mostly binary . 
# * Know more : https://riptutorial.com/pandas/example/20990/one-hot-encoding-with--get-dummies---

# In[23]:


final_features = pd.get_dummies(features).reset_index(drop=True)
final_features.shape


# Understandably so , no. of columns is increased .

# ## Here , Again train and test are spilt back seperately , as now all data processing is done . 
# * Y is taget and its length is used to split 

# In[24]:


X = final_features.iloc[:len(y), :]
X_sub = final_features.iloc[len(y):, :]
X.shape, y.shape, X_sub.shape


# ## Removing outliers . Read other Kernels to understand how they were found out.
# * A few were already stated by the dataset provider , others can be seen by plotting them in a graph

# In[25]:


outliers = [30, 88, 462, 631, 1322]
X = X.drop(X.index[outliers])
y = y.drop(y.index[outliers])

overfit = []
for i in X.columns:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 99.94:
        overfit.append(i)

overfit = list(overfit)
X = X.drop(overfit, axis=1)
X_sub = X_sub.drop(overfit, axis=1)
overfit


# ### Turns out we have dropped just one column (Compare the shape)

# In[26]:


X.shape, y.shape, X_sub.shape


# # Final Step :
# # Now , we are getting started with the process of modelling

# * K-Folds cross-validator
# 
# * Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default).
# 
# * Each fold is then used once as a validation while the k - 1 remaining folds form the training set.

# In[27]:


# defining error functions for handy use. 


kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))
    return (rmse)


# In[28]:


alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]


# In[29]:


ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))                                
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))


# In[30]:


gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)                             


# In[31]:


lightgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       learning_rate=0.01, 
                                       n_estimators=5000,
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=7,
                                       verbose=-1,
                                       )


# In[32]:


xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)


# ## Stacking 
# Stacking is an ensemble learning technique to combine multiple regression models via a meta-regressor. The StackingCVRegressor extends the standard stacking algorithm (implemented as StackingRegressor) using out-of-fold predictions to prepare the input data for the level-2 regressor.
# 
# In the standard stacking procedure, the first-level regressors are fit to the same training set that is used prepare the inputs for the second-level regressor, which may lead to overfitting. The StackingCVRegressor, however, uses the concept of out-of-fold predictions: the dataset is split into k folds, and in k successive rounds, k-1 folds are used to fit the first level regressor. In each round, the first-level regressors are then applied to the remaining 1 subset that was not used for model fitting in each iteration. The resulting predictions are then stacked and provided -- as input data -- to the second-level regressor. After the training of the StackingCVRegressor, the first-level regressors are fit to the entire dataset for optimal predicitons.
# 
# 
# More : http://rasbt.github.io/mlxtend/user_guide/regressor/StackingCVRegressor/
# I encourage that you read more about stacking.
# 
# ![](http://rasbt.github.io/mlxtend/user_guide/regressor/StackingCVRegressor_files/stacking_cv_regressor_overview.png)
# 
# 
# ### In simple words , Stacking helps avoid fitting on the same data twice , and is effective in reducing overfitting.
# 

# In[33]:


stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)


# ## Here , we compare the various models that we just created..

# In[34]:


# Using various prediction models that we just created 


score = cv_rmse(ridge , X)
score = cv_rmse(lasso , X)
print("LASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(elasticnet)
print("elastic net: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(svr)
print("SVR: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(lightgbm)
print("lightgbm: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(gbr)
print("gbr: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(xgboost)
print("xgboost: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )


# In[35]:


print('START Fit')

print('stack_gen')
stack_gen_model = stack_gen.fit(np.array(X), np.array(y))

print('elasticnet')
elastic_model_full_data = elasticnet.fit(X, y)

print('Lasso')
lasso_model_full_data = lasso.fit(X, y)

print('Ridge')
ridge_model_full_data = ridge.fit(X, y)

print('Svr')
svr_model_full_data = svr.fit(X, y)

print('GradientBoosting')
gbr_model_full_data = gbr.fit(X, y)

print('xgboost')
xgb_model_full_data = xgboost.fit(X, y)

print('lightgbm')
lgb_model_full_data = lightgbm.fit(X, y)


# # Blending Models / 'Ensambling'
# * Notice that we are using a few percent from different models to get our final answer , all decimals add up to 1

# In[36]:


def blend_models_predict(X):
    return ((0.1 * elastic_model_full_data.predict(X)) + \
            (0.05 * lasso_model_full_data.predict(X)) + \
            (0.1 * ridge_model_full_data.predict(X)) + \
            (0.1 * svr_model_full_data.predict(X)) + \
            (0.1 * gbr_model_full_data.predict(X)) + \
            (0.15 * xgb_model_full_data.predict(X)) + \
            (0.1 * lgb_model_full_data.predict(X)) + \
            (0.3 * stack_gen_model.predict(np.array(X))))


# In[37]:


print('RMSLE score on train data:')
print(rmsle(y, blend_models_predict(X)))


# In[38]:


print('Predict submission')
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission.iloc[:,1] = (np.expm1(blend_models_predict(X_sub)))


# ### Here , we are also blending(Read: Ensambling) with the data availiable from other 'top' public Kernels .
# * This helps our model generalize on unseen data pretty well

# In[39]:


'''print('Blend with Top Kernels submissions\n')
sub_1 = pd.read_csv('../input/top-10-0-10943-stacking-mice-and-brutal-force/House_Prices_submit.csv')
sub_2 = pd.read_csv('../input/hybrid-svm-benchmark-approach-0-11180-lb-top-2/hybrid_solution.csv')
sub_3 = pd.read_csv('../input/lasso-model-for-regression-problem/lasso_sol22_Median.csv')
submission.iloc[:,1] = np.floor((0.25 * np.floor(np.expm1(blend_models_predict(X_sub)))) + 
                                (0.25 * sub_1.iloc[:,1]) + 
                                (0.25 * sub_2.iloc[:,1]) + 
                                (0.25 * sub_3.iloc[:,1]))'''


# # Submission

# In[40]:


q1 = submission['SalePrice'].quantile(0.0042)
q2 = submission['SalePrice'].quantile(0.99)
# Quantiles helping us get some extreme values for extremely low or high values 
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
submission.to_csv("submission.csv", index=False)


# In[41]:


submission.head()

