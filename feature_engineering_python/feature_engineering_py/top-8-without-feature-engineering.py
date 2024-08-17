#!/usr/bin/env python
# coding: utf-8

# **This is Step 1.1** - simple notebook on "**House Prices: Advanced Regression Techniques**"
# 
# If you like this work - upvotes and feedback are highly appreciated :)
# 
# **This is Step 1.1** - same as corresponding Step 1 but instead of simple ElasticNet model we used StackingCVRegressor (ridge, lasso, elasticnet, gbr, xgboost, lightgbm) with XGBoost as meta_regressor, for prediction and submission. And it gave us ±0.1146 (Top 8%)
# 
# 
# **Step 1** you can find in our public kernels
# https://www.kaggle.com/eiosifov/house-adv-step-1-data-cleaning-elastic
# 
# **As we have stated at the beginning:**
# Main purpose is learning and evaluating how different data cleaning, feature encoding, feature creation, skew, normalization, external feature engineering, affects different modeling and stacking techniques and how different modeling techniques affect performance of predictions.
# 
# **We will explore this competition in a few steps. **
# 
# Each next step we decided just to use CSV data from the previous step:
# 
# **Step 1 - Data cleaning** - simple data cleaning + Scew + Normalization + ElasticNet. Without any feature engineering, external feature creation, and any advanced modeling techniques. One important thing we have done at this step is using ML to impute missing values rather then just use our logical thinking.
# 
# **Step 2 - Basic Feature Engineering** - basic feature engineering + Scew + Normalization + ElasticNet. At this step, we will test different encoding techniques for our data (ordinal, one-hot, label, binary, ...). We will not create any new features. And we will measure heavily effect of each change on different models. 
# 
# **Step 3 - Advanced Feature Engineering** - at this step, we will create and encode new features, use feature importance to drop not important features, etc. And we will measure heavily effect of each change on different models.
# 
# **Step 1.2, 2.2, 3.2** - same as corresponding 1, 2, 3 but instead of simple ElasticNet model we will use StackingCVRegressor (ridge, lasso, elasticnet, gbr, xgboost, lightgbm) with XGBoost as meta_regressor, for prediction and submission.

# As ML algorithms for prediction we will use:
# 
# Step 1 - simple ElasticNet
# 
# Step 1.1 - same data, StackingCVRegressor (ridge, lasso, elasticnet, gbr, xgboost, lightgbm) with XGBoost as meta_regressor
# 
# Step 2 - basic feature engineering, ML Elastic Net
# 
# Step 2.2 - basic feature engineering, ML StackingCVRegressor (ridge, lasso, elasticnet, gbr, xgboost, lightgbm) with XGBoost as meta_regressor
# 
# Step 3 - advanced feature engineering, ML Elastic Net
# 
# Step 3.2 - advanced feature engineering, ML StackingCVRegressor (ridge, lasso, elasticnet, gbr, xgboost, lightgbm) with XGBoost as meta_regressor

# **So let us start with Step 1.1. This simple Step 1.1, which is used same data cleaned at Step 1 which we have loaded to this kernel and se StackingCVRegressor (ridge, lasso, elasticnet, gbr, xgboost, lightgbm) with XGBoost as meta_regressor, for prediction and submission. This approach gave us score of Top 8% = 0.11469**
# 

# # Initial Setup and Data Load

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import os

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from scipy.stats import norm, skew

import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

#from sklearn.ensemble import RandomForestClassifier

import category_encoders as ce

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, RobustScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC,RidgeCV, LassoCV, ElasticNetCV
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import roc_curve, auc


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_pipeline


from scipy.special import boxcox1p
from scipy.stats import boxcox

from datetime import datetime

import string
import warnings
warnings.filterwarnings('ignore')


# In[3]:


get_ipython().system('ls ../input')


# In[4]:


PATH = "../input/cleaned-data/"


# In[5]:


# Loading data cleaned at Step 1 - links can be find on top of notebook
df_train=pd.read_csv(f'{PATH}train_clean.csv')
df_test=pd.read_csv(f'{PATH}test_clean.csv')


# ### Y (target value) to Log, as stated at Kaggle Evaluation page

# In[6]:


print('Number of Training Examples = {}'.format(df_train.shape[0]))
print('Number of Test Examples = {}\n'.format(df_test.shape[0]))
print('Training X Shape = {}'.format(df_train.shape))
print('Training y Shape = {}\n'.format(df_train['SalePrice'].shape[0]))
print('Test X Shape = {}'.format(df_test.shape))
print('Test y Shape = {}\n'.format(df_test.shape[0]))


# ## DataFrame concatination and Y separation

# In[7]:


#remember where to divide train and test
ntrain = df_train.shape[0]
ntest = df_test.shape[0]


# In[8]:


def concat_df(train_data, test_data):
    # Returns a concatenated df of training and test set on axis 0
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

df_all = concat_df(df_train, df_test)

df_train.name = 'Training Set'
df_test.name = 'Test Set'
df_all.name = 'All Set' 

dfs = [df_train, df_test]

df_all.shape


# In[9]:


#Dividing Target column (Y)
y_train_full = df_train.SalePrice.values
df_all.drop(['SalePrice'], axis=1, inplace=True)
#df_all.drop('Id',axis=1,inplace=True)


# In[10]:


y_train_full


# In[11]:


df_all.head()


# In[12]:


df_all.shape


# # Preparing clean data for ML

# ## Dealining with Scewed data

# In[13]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
skewness = df_all.select_dtypes(include=numerics).apply(lambda x: skew(x))
skew_index = skewness[abs(skewness) >= 0.85].index
skewness[skew_index].sort_values(ascending=False)


# In[14]:


'''BoxCox Transform'''
lam = 0.15

for column in skew_index:
    df_all[column] = boxcox1p(df_all[column], lam)


# In[15]:


# Evaluation after working with skewed data
#evaluate(df_all)


# After this step we drastically improved scores, especially ElasticNet

# ## Dummies

# In[16]:


df_all=pd.get_dummies(df_all)


# In[17]:


df_all.shape


# In[18]:


"""Dividing working DataFrame back to Train and Test"""
# split Validational/Test set from Training set after Categorical Value Engeneering
#def original_train_test(df_all):
X_test=df_all.iloc[ntrain:] # Test set
X_train_full=df_all.iloc[:ntrain] # Train set
X_train, X_valid, y_train, y_valid = train_test_split(pd.get_dummies(X_train_full), y_train_full)


# ## Dropping low variance features
""""
Features that have too low variance can negatively impact the model, so we need to remove them by the number of repetitive equal values. In this case, we used a threshold of 99.2% (not 0 or 1 values). Therefore, if any feature has more than ˜99% reps of 1 or 0 it will be excluded. When doing this,
"""
# In[19]:


# Saving all features for future comparison.
all_features = df_all.keys()
# Removing features.
df_all = df_all.drop(df_all.loc[:,(df_all==0).sum()>=(df_all.shape[0]*0.984)],axis=1)
df_all = df_all.drop(df_all.loc[:,(df_all==1).sum()>=(df_all.shape[0]*0.984)],axis=1) 
# Getting and printing the remaining features.
remain_features = df_all.keys()
remov_features = [st for st in all_features if st not in remain_features]
print(len(remov_features), 'features were removed:', remov_features)


# In[20]:


# Evaluation after dropping not important features
#evaluate(df_all)


# ## Normalization
#Normalization, the Sigmoid, Log, Cube Root and the Hyperbolic Tangent. 
#It all depends on what one is trying to accomplish.
# In[21]:


scaler = RobustScaler()
df_all = pd.DataFrame(scaler.fit_transform(df_all))


# In[22]:


# Evaluation after Normalization
#evaluate(df_all)


# Great! We are ready to Train real model

# ## Pred ML Evaluation

# In[23]:


"""Dividing working DataFrame back to Train and Test"""
# split Validational/Test set from Training set after Categorical Value Engeneering
#def original_train_test(df_all):
X_test=df_all.iloc[ntrain:] # Test set
X_train_full=df_all.iloc[:ntrain] # Train set
X_train, X_valid, y_train, y_valid = train_test_split(pd.get_dummies(X_train_full), y_train_full)


# In[24]:


# Defining evaluation functions
def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m,X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[25]:


# ElasticNet Lasso
print('ElasticNet Lasso')
def lasso_score(X,y):
    lasso = ElasticNet(random_state=1)
    param = {'l1_ratio' : [0],
             'alpha' : [0.017]}
    lasso = GridSearchCV(lasso, param, cv=5, scoring='neg_mean_squared_error')
    lasso.fit(X,y)
    print('Lasso:', np.sqrt(lasso.best_score_*-1))
    return lasso
lasso_score(X_train, y_train)

# XGBoost
print('XGBoost')
m_xgb = XGBRegressor(n_estimators=160, learning_rate=0.05)
# using early_stop to find out where validation scores don't improve
#m_xgb.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)], verbose=False)
m_xgb.fit(X_train, y_train)
print_score(m_xgb,X_train, X_valid, y_train, y_valid)


# Random Forest
print('Random Forest')
m_rf = RandomForestRegressor(n_estimators=160, min_samples_leaf=1, max_features=0.5, n_jobs=-1, oob_score=True)
m_rf.fit(X_train, y_train)
print_score(m_rf,X_train, X_valid, y_train, y_valid)


# # Machine Learning

# In[26]:


kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

# rmsle
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# build our model scoring function
def cv_rmse(model, X_train_full=X_train_full):
    rmse = np.sqrt(-cross_val_score(model, X_train_full, y_train_full,
                                    scoring="neg_mean_squared_error",
                                    cv=kfolds))
    return (rmse)


# setup models    
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

ridge = make_pipeline(RobustScaler(),
                      RidgeCV(alphas=alphas_alt, cv=kfolds))

lasso = make_pipeline(RobustScaler(),
                      LassoCV(max_iter=1e7, alphas=alphas2,
                              random_state=42, cv=kfolds))

elasticnet = make_pipeline(RobustScaler(),
                           ElasticNetCV(max_iter=1e7, alphas=e_alphas,
                                        cv=kfolds, l1_ratio=e_l1ratio))
                                        
svr = make_pipeline(RobustScaler(),
                      SVR(C= 20, epsilon= 0.008, gamma=0.0003,))


gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =42)
                                   

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
                                       

xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)

# stack
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet,
                                            gbr, xgboost, lightgbm),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)
                                

print('TEST score on CV')

score = cv_rmse(ridge)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(lasso)
print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(elasticnet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(svr)
print("SVR score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(lightgbm)
print("Lightgbm score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(gbr)
print("GradientBoosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(xgboost)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[27]:


# Fit models
print('StackingCVRegressor')
stack_gen_model = stack_gen.fit(np.array(X_train_full), np.array(y_train_full))
elastic_model_full_data = elasticnet.fit(X_train_full, y_train_full)
lasso_model_full_data = lasso.fit(X_train_full, y_train_full)
ridge_model_full_data = ridge.fit(X_train_full, y_train_full)
svr_model_full_data = svr.fit(X_train_full, y_train_full)
gbr_model_full_data = gbr.fit(X_train_full, y_train_full)
xgb_model_full_data = xgboost.fit(X_train_full, y_train_full)
lgb_model_full_data = lightgbm.fit(X_train_full, y_train_full)


# In[28]:


def blend_models_predict(X_train_full):
    return ((0.1 * elastic_model_full_data.predict(X_train_full)) + \
            (0.05 * lasso_model_full_data.predict(X_train_full)) + \
            (0.1 * ridge_model_full_data.predict(X_train_full)) + \
            (0.1 * svr_model_full_data.predict(X_train_full)) + \
            (0.1 * gbr_model_full_data.predict(X_train_full)) + \
            (0.15 * xgb_model_full_data.predict(X_train_full)) + \
            (0.1 * lgb_model_full_data.predict(X_train_full)) + \
            (0.3 * stack_gen_model.predict(np.array(X_train_full))))


# In[29]:


print('RMSLE score on train data:')
print(rmsle(y_train_full, blend_models_predict(X_train_full)))


# ## Predictions for submission

# In[30]:


submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission.iloc[:,1] = np.floor(np.expm1(blend_models_predict(X_test)))


# In[31]:


submission.head()


# # Submission

# In[32]:


submission.to_csv("submission.csv", index=False)
print('Save submission', datetime.now(),)


# # Conclusins before Step 2

# **Wow!!!
# As we can see with more complex modeling and Stacking techniques and even without any Feature Engineering we  can achieve score 0.11469 that is about Top 8%**
# 
# From our experience we can say, that some techniques of Feature Engineering and Feature Creation can negatively affect accuracy of model and hence the score. We will see this at Step 2 and Step 2.1

# As next Step (2) we will work on basic Feature Engineering (heavily on data encoding techniques).
