#!/usr/bin/env python
# coding: utf-8

# # Advanced Housing Regression

# <font size="3">
# Ever wonder how houses are priced? Market forces work to reduce divergences between housing prices. However, sometimes, the process gets inefficient due to lack of speed in information dissemination and inaccuracies in estimations. Using regression techniques to price houses aim to address this issue. In this notebook, I aim to provide an example of how we can apply regression techniques to such a problem, end-to-end, from eda and feature engineering to building our model
# </font>

# ![image.png](attachment:image.png)

# <a id="toc"></a>
# 
# # Table of Contents
# <font size="3">    
# * [Introduction](#section-zero)
# * [Executive Summary](#section-one)
# * [Resources Checklist](#section-onea)
# * [Exploratory Data Analysis](#section-two)
#     - [Ensuring Approximate Normality of Dependent Variable](#subsection-one)
#     - [Dealing with Outliers](#subsection-two)
#     - [Exploring Correlations](#subsection-three)
# * [Imputing Missing Values](#section-three)
# * [Data Wrangling](#section-four)
# * [Model Fitting](#section-five)
# * [Stacking & Blending](#section-six)  

# <a id="section-zero"></a>
# Introduction
# ==
# 
# <font size="3">
# This notebook aims to provide an in-depth exploratory data analysis and feature engineering to regression modelling  
#       <br/><br/>
# </font>
#     
# <font size="3">
# If this kernel has helped you in any way, please upvote, it provides me a lot of motivation to continue providing more in-depth analyses and to share what I have learnt about modelling. Any feedback is also greatly welcomed, please comment it down below
#     <br/><br/>
# </font>
# 
# <font size="3">
# Note that this notebook is still a work in progress, I am currently working to improve accuracy further :)
# </font>

# <a id="section-one"></a>
# Executive Summary
# ==
# 
# <font size="3">
# 
# In this notebook, I worked on feature engineering and imputing missing values. There are over 80 features in this dataset, which makes it a mammoth task for beginners. Along the way, I try to add explanations of features to make it as beginner-friendly as possible.
# 
# The models I used are:
# * Lasso regression model 
# * XGBoost model
# * LGBM model

# <a id="section-onea"></a>
# Resources Checklist
# ==
# 
# <font size="3">
#     
# If you are not a beginner in pandas and modelling, please skip this section
#     
# <ins> Beginners' Resources </ins> 
# 
# * New to Python? Please go through: https://www.kaggle.com/learn/python
# * New to Pandas? Please go through: https://www.kaggle.com/learn/pandas
# * New to Machine Learning? Please go through: 
#     * https://www.kaggle.com/learn/intro-to-machine-learning
#     * https://www.kaggle.com/learn/intermediate-machine-learning
# * Notebook Formatting: https://www.kaggle.com/chrisbow/formatting-notebooks-with-markdown-tutorial
#     
#     
#  
# <ins> Intermediate Resources </ins> 
#     
# I found these articles to be helpful in understanding regression: 
#     
# * Linear Regression: https://towardsdatascience.com/linear-regression-understanding-the-theory-7e53ac2831b5
# * Ridge, LASSO and ElasticNet: https://www.datacamp.com/community/tutorials/tutorial-ridge-lasso-elastic-net
#     
# <ins> Documentations </ins>
# * Scikit-learn's documentation homepage: https://scikit-learn.org/stable/
# * Scikit-learn's supervised learning documentation: https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
# * XGBoost documentation: https://xgboost.readthedocs.io/en/latest/
# * LGBM Documentation: https://lightgbm.readthedocs.io/en/latest/
#     
#     
# Know of any other great resources? Please comment down below so I can add them to this section
#     
#     
# [Back to Table of Content](#toc)

# In[1]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_rows', 85)

import os
print(os.listdir("../input"))

from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMRegressor


# In[2]:


df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
full_data = [df_train, df_test]


# In[3]:


df_train.describe()


# In[4]:


df_train.head()


# <font size="3">
# There are a staggering 81 features present in this dataset!
# </font>

# <a id="section-two"></a>
# **Exploratory Data Analysis:**
# ==
# 
# <font size="3">
# 1. Ensuring Approximate Normality of dependent variable <br/>
# 2. Dealing with Outliers  <br/>
# 3. Exploring Correlations  <br/>
#     
# <br/>
#     
# [Back to Table of Content](#toc)

# <a id="subsection-one"></a>
# ## Ensuring Approximate Normality of dependent variable
# ---
# 
# [Back to Table of Content](#toc)

# In[5]:


sns.distplot(df_train['SalePrice'], bins = 20, hist_kws=dict(edgecolor="k", linewidth=2))


# <font size="3">
# It seems that our dependent variable has a skewed distribution and not an approximately normal distribution. We will need to transform our dependent variable to increase accuracy of prediction.
#     </font>

# In[6]:


sns.distplot(np.log(df_train['SalePrice']), bins = 20, hist_kws=dict(edgecolor="k", linewidth=2))


# <font size="3">
# log(SalePrice) is a more regular distribution with less skewness. We can use log(SalePrice) as our dependent variable)
#     </font>

# In[7]:


df_train['LSalePrice'] = np.log(df_train['SalePrice'])


# In[8]:


import statsmodels.api as sm


# In[9]:


fig = sm.qqplot(df_train['LSalePrice'],fit=True, line='45')


# <font size="3">
# QQline represents a normal distribution. The scatterpoints align closely with the QQline, except for the tails. We can conclude that log(SalePrice) is approximately normally distributed
# </font>

# <a id="subsection-two"></a>
# ## Dealing with Outliers
# ---
# 
# [Back to Table of Content](#toc)

# In[10]:


from scipy import stats
m = stats.trim_mean(df_train['LSalePrice'], 0.1)
print("With 10% clipped on both sides, trimmed mean: {}".format(m))
print("Sample mean: {}".format(np.mean(df_train['LSalePrice'])))


# <font size="3">
# Trimmed mean is close to sample mean, this suggests there are no extreme outliers
#     </font>

# **<ins>
# <font size="3">
# Finding out the data type of each column:
#     </font>
#  </ins>**

# In[11]:


df_train.dtypes


# <font size="3">
# Information is not useful for finding out if column represents continous/categorical/ordinal variable. In this case, the only ways are to manually explore each column and read data description.
#     </font>

# <font size="3">
# From description of data (data_description.txt),<br/><br/>
#     </font>
# 
# <font size="3">
# <ins>categorical variables:</ins> <br />
#     
#     
# ```python    
# [ 'MSSubClass' , 'MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2',
#   'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 
#   'Heating', 'CentralAir', 'Electrical', 'GarageType', 'PavedDrive', 'MiscFeature', 'SaleType', 'SaleCondition']
# ```   
#     
#   
# 
# <ins>ordinal variables:</ins>
# ```python  
# [ 'LotShape', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtCond', 'BsmtExposure', 
#   'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'Bedroom', 'Kitchen', 
#   'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces' , 'FireplaceQu',  'BsmtQual', 'GarageFinish', 'GarageQual', 
#   'GarageCond', 'PoolQC', 'Fence']
# ```
# 
#   
# 
#   <ins>continous variables are:</ins>
#     
# ```python 
# [ 'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',  '1stFlrSF',  '2ndFlrSF',  
#   'LowQualFinSF',  'GrLivArea', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 
#   'ScreenPorch', 'PoolArea', 'MiscVal' ]
# ```  
# 
#   <ins>other variables (Date, etc..) :</ins> 
# ```python 
# [' YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold']
# ```

# <font size="3">
# We need to understand our data better. To do so, we can observe patterns between our independent variables and the dependent variable. However, there are many independent variables. We need a smart way to analyse our explanatory variables instead of analysing them one by one.
#     </font>

# <a id="subsection-three"></a>
# ## Exploring Correlations
# ---
# 
# [Back to Table of Content](#toc)

# <font size="3">
# A really useful tool to inspect correlations between variables is a heatmap. This allows us to find out which independent variable is highly correlated with the dependent variable.
#     </font>

# In[12]:


a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)

corr = df_train.corr()
ax = sns.heatmap(corr, vmin=0.5, cmap='coolwarm', ax=ax)


# <font size="3">
# Looking at our heatmap, Important factors include OverallQual, LowQualFinSF, GarageCars
# </font>
# 
# <font size="3">
# This makes a lot of sense. Overall Quality of material of a house will obviously affect SalePrice significantly along with amount of floor that are low quality in sq feet and size of garage.
# </font>

# In[13]:


sns.pairplot(df_train[["SalePrice","OverallQual","OverallCond","YearBuilt","TotalBsmtSF","BsmtUnfSF","GrLivArea","FullBath","GarageCars","GarageArea"]])


# <font size="3">
# The scatterplots exhibit some interesting patterns. <br /><br />
# 
# 1) There seems to be strong linear correlation between Above grade (ground) living area, TotalBsmtSF and Sale Price, except for a couple of outliers.
# 
# 2) For ordinal variables, OverallQual factors in significantly. Number of Full Bathrooms above grade shows a strong, significant pattern with SalePrice, along with GarageCars, although for GarageCars, the largest size tends to have the lowest SalePrice which is interesting. It could be that the largest garage would be difficult to maintain

# <a id="section-three"></a>
# **Imputing Missing Values:**
# ==
# 
# [Back to Table of Content](#toc)

# In[14]:


df_train.isnull().sum()


# In[15]:


df_test.isnull().sum()


# <font size="3">
# <ins>Columns with missing values:</ins> <br /><br />
#     <ins> Training set: </ins><br /><br />
#     
# ```python
# [ 'LotFrontage', 'Alley', 'BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
#   'BsmtFinType2','Electrical','FireplaceQu','GarageType','GarageYrBlt',
#   'GarageFinish', 'GarageQual','GarageCond','PoolQC','Fence','MiscFeature']
# ```
# 
# <ins> Test set: </ins><br />
# ```python
# ['MSZoning', 'LotFrontage','Alley','Utilities','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea', 
# 'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2',
# 'BsmtUnfSF','TotalBsmtSF','BsmtFullBath', 'BsmtHalfBath','KitchenQual','Functional','FireplaceQu', 
# 'GarageType', 'GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond']
# ```

# In[16]:


len([ 'LotFrontage', 'Alley', 'BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','Electrical','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature'])


# In[17]:


len(['MSZoning', 'LotFrontage','Alley','Utilities','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea', 'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath', 'BsmtHalfBath','KitchenQual','Functional','FireplaceQu', 'GarageType', 'GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond'])


# <font size=3>
# There are 17 columns with missing values for training data and 29 columns with missing values for test data!

# In[18]:


df_train['MSSubClass'][df_train['LotFrontage'].isnull()].unique()


# In[19]:


a4_dims = (11.7, 6.27)
fig, ax = plt.subplots(figsize=a4_dims)
fig2, ax2 = plt.subplots(figsize=a4_dims)
sns.boxplot(x="MSSubClass",y="LotFrontage",data=df_train,ax=ax).set_title("Training set")
sns.boxplot(x="MSSubClass",y="LotFrontage",data=df_test,ax=ax2).set_title("Test set")


# <font size="3">
# Intuition: LotFrontage depends on the type of dwelling. Some dwellings will naturally have less access to the street (for more secluded kinds of dwellings).
# 
# Examining the boxplot, we can see that, indeed there are variations in LotFrontage among different kinds of dwellings.
# 
# Thus, missing values of LotFrontage can be imputed

# In[20]:


for dataset in full_data:
    dataset['LotFrontage'] = dataset.groupby('MSSubClass')['LotFrontage'].transform(lambda x: x.fillna(x.median()))


# <font size="3">
# There is 1 missing value for LotFrontage left and that is for MSSubClass 150. We have little information to impute this value, however, we know it is a PUD.  A reasonable value to impute will be to take average of the medians of the 1-story and 2-story PUDs.
#     </font>

# In[21]:


df_test.groupby('MSSubClass')['LotFrontage'].median()


# In[22]:


df_test['LotFrontage'][df_test['MSSubClass']==150] = 38


# <font size="3">
# Lot Frontage ✓
#     </font>
# 

# <font size="3">
# Columns that are categorical in training set and test set that have missing values:
#     
# ```python
# ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
#  'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 
#  'MiscFeature','MasVnrType']
# ```
# 
# The missing values could represent an important piece of information hence it would be unwise to simply drop these columns
# 
# On account of loss of information should we choose to drop these columns, and to make our life easier, let's create a separate category in each of these columns for the values that are missing

# In[23]:


missin_cats = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature','MasVnrType']
df_train.update(df_train[missin_cats].fillna("NA"))
df_test.update(df_test[missin_cats].fillna("NA"))


# <font size="3">
# 
# For our training set, we are left with 
# ```
# Electrical and GarageYrBlt
# ``` 
# 
# <br /><br />
# For the test set, we are left with: 
# 
# ```
# MSZoning, LotFrontage, Utilities, Exterior1st, Exterior2nd, MasVnrType, MasVnrArea, BsmtFinSF1,
# BsmtUnfSF,TotalBsmtSF, BsmtFullBath, BsmtHalfBath, KitchenQual, Functional, GarageYrBlt, 
# GarageCars, GarageArea, SaleType
# ```

# <font size="3">
# The reason why some categorical columns aren't imputed above is because they are exclusive on either data sets. Therefore, we need to be careful as imputing on either dataset will create a unforeseen category, for example, if we create a new category in the test dataset, the model trained on the training set may be unusable on the test set.
#     </font>

# <font size="3">
# Impute remaining categorical variables in test set:
#     </font>

# In[24]:


df_test[df_test['MSZoning'].isnull()]


# <font size="3">
# We can check if there is a relationship between MSZoning and MSSubClass
#     </font>

# In[25]:


def aggregate(rows,columns,df):
    column_keys = df[columns].unique()
    row_keys = df[rows].unique()

    agg = { key : [ len(df[(df[rows]==value) & (df[columns]==key)]) for value in row_keys]
               for key in column_keys }

    aggdf = pd.DataFrame(agg,index = row_keys)
    aggdf.index.rename(rows,inplace=True)

    return aggdf

a4_dims = (11.7, 6.27)
fig, ax = plt.subplots(figsize=a4_dims)
aggregate('MSSubClass','MSZoning',df_test).plot(kind='bar',stacked=True, ax=ax)


# <font size="3">
# We can impute 20:1-STORY 1946 & NEWER ALL STYLES with RL, since it is the mode in the category. However, Subclasses 30 and 70 would be trickier.
#     </font>

# In[26]:


aggregate('MSSubClass','MSZoning',df_test)


# <font size="3">
# In view of time, we impute house with SubClass 30 & 70 as "RM" since it is the mode.
#     </font>

# In[27]:


df_test['MSZoning'][(df_test['MSZoning'].isnull()) & (df_test['MSSubClass']==20) ] = "RL"


# In[28]:


df_test['MSZoning'][(df_test['MSZoning'].isnull()) & ( (df_test['MSSubClass']==30) | (df_test['MSSubClass']==70) ) ] = "RM"


# <font size="3">
# For the remaining columns that are categorical in test set, since they are at most missing a couple of values, we can fill them in by the mode of the data.
#     </font>

# <font size="3">
#     
# ```python
# ['Utilities', 'Exterior1st', 'Exterior2nd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF' ,'TotalBsmtSF', 
#  'KitchenQual', 'Functional','GarageCars', 'GarageArea', 'SaleType']
# ```

# In[29]:


for col in ['Utilities', 'Exterior1st', 'Exterior2nd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF' ,'TotalBsmtSF', 'KitchenQual', 'Functional','GarageCars', 'GarageArea', 'SaleType','BsmtFullBath','BsmtHalfBath']:
    df_test[col].fillna(df_test[col].mode()[0],inplace=True)


# In[30]:


df_test.isnull().sum()


# In[31]:


df_train.isnull().sum()


# In[32]:


df_train['Electrical'].fillna(df_train['Electrical'].mode()[0],inplace=True)


# <font size="3">
# Intuitively, Year that garage is built shouldn't affect SalePrice significantly though we do not know for sure. However, to make life easier, we will drop this column in this analysis.
#     </font>

# In[33]:


for data_set in full_data:
    data_set.drop(['GarageYrBlt'], axis=1,inplace=True)


# <font size="3">
# We are left with MasVnrArea to impute for both training and test data.
#     </font>

# In[34]:


a4_dims = (11.7, 6.27)
fig, ax = plt.subplots(figsize=a4_dims)
fig2, ax2 = plt.subplots(figsize=a4_dims)
sns.boxplot(x="MasVnrType",y="MasVnrArea",data=df_train,ax=ax).set_title("Training set")
sns.boxplot(x="MasVnrType",y="MasVnrArea",data=df_test,ax=ax2).set_title("Test set")


# <font size="3">
# Masonry Veneer Area does not seem seperable by Masonry Veneer Type. Nevertheless, logically, it should depend on the Masonry Veneer Type and there are nuances between different Masonry Types. Given that this is the only logical link that seems present at the moment, we impute by Masonry Veneer Type.
#     </font>

# In[35]:


df_train[df_train['MasVnrArea'].isnull()]


# <font size="3">
# MasonryType is not reported, we cannot impute for training set. There is no other available option except to impute as 0.
#     </font>

# In[36]:


for dataset in full_data:
    dataset['MasVnrArea'] = dataset.groupby('MasVnrType')['MasVnrArea'].transform(lambda x: x.fillna(x.median()))


# In[37]:


df_train['MasVnrArea'][df_train['MasVnrArea'].isnull()] = 0


# In[38]:


sum(df_train.isnull().sum())


# In[39]:


sum(df_test.isnull().sum())


# <font size="3">
# That concludes Imputing of Missing Values. Next, we need to do some Data Wrangling, especially for categorical variables, using techniques such as One-Hot Encoding before we can train our model.
#     </font>

# <a id="section-four"></a>
# Data Wrangling
# ==
# 
# [Back to Table of Content](#toc)

# <font size="3">
# There are 3 types of variables we have to deal with: categorical, ordinal and date variables.
# 
# For categorical, we need to perform one-hot encoding to implement regression.
# 
# For ordinal, we need to assign values to each level.
# 
# For the purpose of this analysis, to be conservative, we encode ordinal variables (in other words, we will just treat ordinal variables like categorical variables).
# 
# For date, we need to either treat it as a categorical variable or drop the column

# <font size="3">
# As a reminder:<br /><br />
#   
# __categorical variables__: <br />
# ```python
# [ 'MSSubClass' , 'MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 
#   'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 
#   'CentralAir', 'Electrical', 'GarageType', 'PavedDrive', 'MiscFeature', 'SaleType', 'SaleCondition']
# ```
#   
# __ordinal variables__: <br />
# ```python
# [ 'LotShape', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtCond', 'BsmtExposure', 
#   'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 
#   'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces' , 'FireplaceQu',  'basement', 'GarageFinish', 
#   'GarageQual', 'GarageCond', 'PoolQC', 'Fence']
# ```
#   
# __other variables__ (Date, etc..) : <br />
# ```python
# [' YearBuilt', 'YearRemodAdd',  'MoSold', 'YrSold']
# ```
#   
# MoSold and YrSold can be treated as categorical variables. We will deal with the rest of the date variables later

# In[40]:


df_train['MSSubClass'].unique()


# In[41]:


df_test['MSSubClass'].unique()


# <font size="3">
# As you can see from the example aboove, the training and test sets may have different categories within each categorical variable. What we have to do is to merge the 2 data sets temporarily to process the categorical columns and then seperate them out again
#     </font>

# In[42]:


df_train['train'] = 1
df_test['train'] = 0


# In[43]:


combined_dataset = pd.concat([df_train,df_test])


# In[44]:


categories={} # contains all the levels in those feature columns
categorical_feature_names =   [ 'MSSubClass' , 'MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig', 
                               'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 
                               'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 
                               'CentralAir', 'Electrical', 'GarageType', 'PavedDrive', 'MiscFeature', 'SaleType', 
                               'SaleCondition', 'LotShape', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond',
                               'ExterQual', 'ExterCond', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                               'BsmtFinType2', 'HeatingQC', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 
                               'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces' , 
                               'FireplaceQu',  'BsmtQual', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 
                               'Fence',]

for f in categorical_feature_names:
    # to make sure the type is indeed category
    combined_dataset[f] = combined_dataset[f].astype('category')
    categories[f] = combined_dataset[f].cat.categories

new_combined_dataset = pd.get_dummies(combined_dataset,columns=categorical_feature_names,drop_first=True)


# In[45]:


df_newtrain = new_combined_dataset[new_combined_dataset['train'] == 1]
df_newtest = new_combined_dataset[new_combined_dataset['train'] == 0]
df_newtrain.drop(["train"],axis=1,inplace=True)
df_newtest.drop(["train","LSalePrice","SalePrice"],axis=1,inplace=True)


# In[46]:


df_newtrain.shape


# In[47]:


df_newtest.shape


# **<font size="3">
# Dealing with dates:
# </font>**

# <font size="3">
# We can take current year and subtract it by year built. this will give us the age of the house. 
# For YearRemodAdd, it will tell us when was the last time the house was remodelled which would indicate information
# about the condition of the house.
# </font>
# 

# In[48]:


new_fulldata = [df_newtrain, df_newtest]
for dataset in new_fulldata:
    for col in ['YearBuilt','YearRemodAdd']:
        dataset['HouseAge'] = 2018 - dataset['YearBuilt']
        dataset['RemodAge'] = 2018 - dataset['YearRemodAdd']
        dataset.drop(['YearBuilt','YearRemodAdd'],axis=1)


# <font size="3">
# Now, we can finally train models on our data!
# </font>

# In[49]:


lambgrid = np.linspace(0.01, 100, num=1000,endpoint=True)


# In[50]:


y_train = df_train['LSalePrice']


# In[51]:


train_id = df_newtrain['Id']
test_id = df_newtest['Id']


# In[52]:


X_train = df_newtrain.drop(['SalePrice','LSalePrice','Id'],axis=1)


# In[53]:


df_newtest.drop(['Id'],axis=1,inplace=True)


# <a id="section-five"></a>
# Model Fitting
# ==
# <font size="3">
# 
# For our model fitting, we will define Pipelines for our sklearn models. 
# 
# <ins>What are sklearn Pipelines?</ins>
# * sklearn Pipelines sequentially apply a list of transforms and a final estimator. You can save pipelines as pkl file for ease of access. It simpifies your training code and improves readability.  
# * See: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html for more details.
# 
# In each Pipeline, we define a RobustScaler.
# 
# <ins>What is RobustScaler?</ins>  
# * RobustScaler removes the median and scales the data according to the quantile range (defaults to IQR: Interquartile Range). This is to make our data robust to outliers. 
# * See: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html for more details
# 
# [Back to Table of Content](#toc)

# In[54]:


#Ridge Regression:
from sklearn.linear_model import RidgeCV
# clf = RidgeCV(alphas=lambgrid, cv=5)
clf = make_pipeline(RobustScaler(),RidgeCV(alphas=lambgrid, cv=5))


# In[55]:


from sklearn.linear_model import Ridge
coefs = []
for a in lambgrid:
    ridge = Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)


# <font size="3">
# As an illustration, below is a graph of weights against alphas.
# The larger alpha is, the bigger the regularization effect and hence, the smaller the coefficients.
#     </font>

# In[56]:


a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
ax.plot(lambgrid, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()


# In[57]:


clf.fit(X_train,y_train)


# In[58]:


preds_ridge = clf.predict(df_newtest)


# In[59]:


from sklearn.model_selection import cross_val_score
print('Ridge Regression Cross Validation Score: %s' % (
                      cross_val_score(clf, X_train, y_train,scoring='neg_mean_squared_error').mean()))


# <font size="3">
# Let's see if we can get a lower CV score with LASSO or  ElasticNet.
#     </font>

# In[60]:


#LASSO:
from sklearn.linear_model import LassoCV
# lasso_clf = LassoCV(alphas=lambgrid, cv=5)
lasso_clf = make_pipeline(RobustScaler(), LassoCV(alphas=lambgrid, cv=5))


# In[61]:


lasso_clf.fit(X_train,y_train)


# In[62]:


preds_lasso = lasso_clf.predict(df_newtest)


# In[63]:


# lasso_alpha = lasso_clf.alpha_
# from sklearn.linear_model import Lasso

print('Lasso Cross Validation Score: %s' % (
                      cross_val_score(lasso_clf, X_train, y_train,scoring='neg_mean_squared_error').mean()))


# <font size="3">
# Lasso has a larger mean squared error as compared to Ridge. Looks like Ridge Regression is leading!
#     </font>

# <font size="3">
# Introducing ElasticNet:
# </font>

# In[64]:


#ElasticNet:
from sklearn.linear_model import ElasticNetCV
# elastic_clf = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],alphas=lambgrid, cv=5)
elastic_clf = make_pipeline(RobustScaler(), ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],alphas=lambgrid, cv=5))


# In[65]:


elastic_clf.fit(X_train,y_train)


# In[66]:


preds_elastic = elastic_clf.predict(df_newtest)


# In[67]:


# elastic_alpha = elastic_clf.alpha_
# from sklearn.linear_model import ElasticNet

print('ElasticNet Cross Validation Score: %s' % (
                      cross_val_score(elastic_clf, X_train, y_train,scoring='neg_mean_squared_error').mean()))


# <font size="3">
# Looks like Ridge Regression wins, followed by ElasticNet then Lasso.
# 
# Can we do even better? How about Gradient Boosted Regression?

# In[68]:


import xgboost as xgb
xgtrain = xgb.DMatrix(X_train,label=y_train)
xgb_clf = xgb.XGBRegressor(n_estimators=1000, learning_rate=.03, max_depth=3, max_features=.04, min_samples_split=4,
                           min_samples_leaf=3, loss='huber', subsample=1.0, random_state=0)
xgb_param = xgb_clf.get_xgb_params()


# In[69]:


print('XGB Regression Cross Validation Score: %s' % (
                      cross_val_score(xgb_clf, X_train, y_train,scoring='neg_mean_squared_error').mean()))


# <font size="3">
# Extreme Gradient Boosting has dethroned Ridge Regression!
#     </font>

# In[70]:


xgb_clf.fit(X_train,y_train)


# In[71]:


preds_xgb = xgb_clf.predict(df_newtest)


# In[72]:


# Fit Light Gradient Boosting:
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
                         verbose=-1)

lightgbm.fit(X_train,y_train)


# In[73]:


preds_lgb = lightgbm.predict(df_newtest)


# <a id="section-six"></a>
# Stacking and Blending
# ==
# 
# <font size="3">
# 
# <ins> Stacking </ins>    
#     
# Stacking involves building a metamodel, that weighs our models' predictions together, to give our final predictions    
#     
# Let's try stacking all our models together using StackingCVRegressor. Stacked generalization consists in stacking the output of individual estimator and use a regressor to compute the final prediction. Read more about it at: 
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html
#     
#   
# Here, we will use StackingCVRegressor for greater stability of results. 
# From the documentation: 
# "In the standard stacking procedure, the first-level regressors are fit to the same training set that is used prepare the inputs for the second-level regressor, which may lead to overfitting. The StackingCVRegressor, however, uses the concept of out-of-fold predictions: the dataset is split into k folds, and in k successive rounds, k-1 folds are used to fit the first level regressor. In each round, the first-level regressors are then applied to the remaining 1 subset that was not used for model fitting in each iteration. The resulting predictions are then stacked and provided -- as input data -- to the second-level regressor. After the training of the StackingCVRegressor, the first-level regressors are fit to the entire dataset for optimal predictions."
#     
# ![image.png](attachment:image.png)
#     
#     
# In a nutshell, rather than fitting our metaregressor solely on seen data, we fit it on both seen and unseen data using KFolds Cross-Validation which provides a better indication of generalization against test data
# 
#     
# <ins> Blending </ins>   
#     
# Blending involves taking simple/weighted averages of our predictions to give our final predictions
#     
# [Back to Table of Content](#toc)

# In[74]:


## Stacking
from mlxtend.regressor import StackingCVRegressor


ridge = make_pipeline(RobustScaler(),RidgeCV(alphas=lambgrid, cv=5))
lasso = make_pipeline(RobustScaler(), LassoCV(alphas=lambgrid, cv=5))
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],alphas=lambgrid, cv=5))
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
xgboost = xgb.XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)


stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, xgboost, lightgbm),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)
stack_gen.fit(np.array(X_train), np.array(y_train))
stacked_preds = stack_gen.predict(np.array(df_newtest))


# In[75]:


## Blending:

def blend_models_predict(preds_ridge, preds_xgb, preds_lgb):
    return (np.exp(preds_ridge) + np.exp(preds_xgb) + np.exp(preds_lgb))/3


# In[76]:


submission = pd.DataFrame({'Id': test_id, 'SalePrice': np.exp(preds_xgb)})


# In[77]:


# Stack Ridge Regression and XGBoost Regression Predictions
blend_preds = blend_models_predict(preds_ridge, preds_xgb, preds_lgb)
blend_submission = pd.DataFrame({'Id': test_id, 'SalePrice': blend_preds})
stacked_submission = pd.DataFrame({'Id': test_id, 'SalePrice': np.exp(stacked_preds)})


# In[78]:


# Save Stacked and XGBoost Regression Predictions
submission.to_csv('Submission.csv',index=False)
stacked_submission.to_csv('Stacked_Submission.csv', index=False)
blend_submission.to_csv('Blend_Submission.csv', index=False)

