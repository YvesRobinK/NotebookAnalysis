#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering - House Prices
# 
# 
# ![](https://camo.githubusercontent.com/9de3e2baa949b4ecc71147533765e07ecfa33a58333aed57f57db765328ade4f/68747470733a2f2f73746f726167652e676f6f676c65617069732e636f6d2f6b6167676c652d636f6d7065746974696f6e732f6b6167676c652f353430372f6d656469612f686f7573657362616e6e65722e706e67)
# 

# # Table of Contents
# 
# * [Introduction](#introduction)
# * [House Keeping](#house)
# * [Data Cleaning](#clean)
# * [Feature Engineering](#feature)
# * [Conclusion](#conc)

# # Introduction <a id="introduction"></a>
# 
# *Feature engineering is the process of transforming raw data into features that better represent the underlying problem to the predictive model. This can include techniques such as scaling, normalization, and creating interaction or polynomial features. The goal of feature engineering is to increase the predictive power of the model by creating relevant features from the raw data, and it is often a crucial step in the development of a successful machine learning model.*
# 
# *Feature engineering should be used when the raw data is not suitable for a predictive model as is. It is an important step in the machine learning pipeline and it can be useful in a variety of situations, such as:*
# 
# * *When the data is not in a format that can be easily used by a model, for example, text data needs to be converted to numerical values before it can be used by a model.*
# * *When the data is not complete, and additional features need to be created to represent the missing information.*
# * *When the data is not in the correct scale, and needs to be scaled or normalized to prevent certain features from dominating the model.*
# * *When there is a domain knowledge, you can use that knowledge to create new features that may be more informative than the raw data.*
# * *When you have correlated variables, you can create new features that are linear combinations of the original ones.*
# 
# *It is important to note that feature engineering is an iterative process and it is crucial to evaluate the performance of the model after each feature engineering step and make adjustments as necessary.*
# 
# In this notebook I will dive into the [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) dataset to explore and learn along the way.
# 
# **Hope you enjoy, let me know how I can improve, and if you liked it, an upvote would help me out alot!**
# 
# **Want to dig deeper into data exploration of this dataset? Check out my notebook on [Explorative Data Analysis](https://www.kaggle.com/code/ulrikthygepedersen/exploratory-data-analysis-house-prices)**
# 
# **Want to learn more about how to further reduce features? Check out my notebook on [Principal Component Analysis](https://www.kaggle.com/code/ulrikthygepedersen/reducing-features-principal-component-analysis/notebook)**
# 
# **Want to learn more about Random Forest Modelling to predict Sale Price? Check out my notebook on [Random Forest Regressor Model](https://www.kaggle.com/code/ulrikthygepedersen/random-forest-regressor-model-house-prices)**
# 
# **Want to learn more about XGBoost Regressor Modelling to predict Sale Price? Check out my notebook on [XGBoost Regressor Model](https://www.kaggle.com/code/ulrikthygepedersen/xgboost-regressor-model-house-prices/notebook)**

# # House Keeping <a id="house"></a>
# 
# ## Import Libraries, load dataset and do a short summary

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# load datasets
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df_sample_submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

# mark train and test sets for future split
df_train['train_test'] = 'Train'
df_test['train_test'] = 'Test'

#combine to a single dataframe with all data for feature engineering
df_all = pd.concat([df_train, df_test])

# print dataset shape and columns
trow, tcol = df_train.shape
erow, ecol = df_test.shape
srow, scol = df_sample_submission.shape

print(f'''
Train Dataset:
Loaded train dataset with shape {df_train.shape} ({trow} rows and {tcol} columns)

Test Dataset:
Loaded test dataset with shape {df_test.shape} ({erow} rows and {ecol} columns)

Sample Submission Dataset:
Loaded sample submission dataset with shape {df_sample_submission.shape} ({srow} rows and {scol} columns)
''')


# # Data Cleaning <a id="clean"></a>
# 
# Based on my [previous notebook on Exploratory Data Analysis](https://www.kaggle.com/code/ulrikthygepedersen/exploratory-data-analysis-house-prices), I will drop features with little information to increase model training time and accuracy:

# In[2]:


# drop the Id and PoolQC columns
df_all = df_all.drop(['Id', 
                      'PoolQC', 
                      'PoolArea'], 
                      axis=1)

# drop features with little information based on visualizations
df_all = (df_all.drop(['BsmtFinSF2',
                       'LowQualFinSF',
                       'BsmtHalfBath',
                       'KitchenAbvGr',
                       'EnclosedPorch',
                       '3SsnPorch',
                       'MiscVal',
                       'Street', 
                       'Utilities', 
                       'Condition2', 
                       'RoofMatl', 
                       'Heating',
                       'MiscFeature'], 
                       axis=1))

# drop features with little information based on heatmap
df_all = (df_all.drop(['MSSubClass',
                       'OverallCond',
                       'ScreenPorch',
                       'MoSold',
                       'YrSold'], 
                       axis=1))


# ## Missing Values
# 
# Missing values can be bad for machine learning for several reasons:
# 
# * Missing values can cause issues with model training and evaluation. Many machine learning algorithms cannot handle missing data and will either throw an error or produce inaccurate results.
# * Missing values can lead to a decrease in the sample size. When a large number of observations have missing values, it can lead to a smaller sample size which can decrease the statistical power of the model.
# * Missing values can introduce bias. If the missing values are not missing at random, then the model may be trained on a biased sample, which can lead to inaccurate predictions on new data.
# * Missing values can affect the correlations and relationships between variables. If a variable with missing values is correlated with other variables, then the absence of that variable can affect the relationships between the other variables and the target variable.
# * Missing values can make it difficult to interpret the model. If there are missing values in the input data, it is hard to understand the effect of individual variables on the outcome, and it can be hard to make sense of the model's predictions.
# 
# For these reasons, it is important to handle missing values appropriately before using the data for machine learning. There are several techniques to deal with missing values such as:
# 
# * Removing observations with missing values
# * Imputing missing values
# * Using algorithms that can handle missing data
# * Using techniques to infer missing values based on the other variables
# 
# **It's important to note that the best way to handle missing values depends on the specific problem and dataset you are working with.**

# In[3]:


df_info = pd.DataFrame(data={
    'Number of Missing Values': df_all.isna().sum(),
    'Number of Unique Values': df_all.nunique(),
    'Unique Values': [df_all[col].unique().tolist() for col in df_all.columns],
    'Column type': df_all.dtypes
})

df_info


# The following columns has null values and our job is now to reduce this to zero: 
# 
# 'MSZoning', 'LotFrontage', 'Alley', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtUnfSF', 'TotalBsmtSF', 'Electrical', 'BsmtFullBath', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'Fence', 'SaleType', 'SalePrice'
# 
# First lets start with the **numerical** features and replace any missing values with the mean:

# In[4]:


# replace numerical features with the mean of the column
for col in df_all.columns:
    if((df_all[col].dtype == 'float64') or (df_all[col].dtype == 'int64')):
        df_all[col].fillna(df_all[col].mean(), inplace=True)


# **Nice!** This reduced our features with missing values alot to the following list: 
# 
# 'MSZoning', 'Alley', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'Fence', 'SaleType'
# 
# Next to the categorical features. We are going to replace the categorical features with the most common value using the .mode()[0] attribute of pandas:

# In[5]:


# replace categorical features with the most common value of the column
for col in df_all.columns:
    if df_all[col].dtype == 'object':
        df_all[col].fillna(df_all[col].mode()[0], inplace=True)


# ...and the list we have left is:
# 
# []
# 
# Nothing! Our data is now free on missing values and ready for modelling!

# # Feature Engineering <a id="feature"></a>
# 
# Feature engineering is the step in the machine learning pipeline that involves preparing the data for use in a model. This includes a variety of tasks such as transforming and formatting the data to make it suitable for use with a specific algorithm or model.
# 
# Some common data pre-processing tasks include:
# 
# * **Data transformation**: Changing the data into a format that can be used by the model. This includes feature scaling, one-hot encoding, and any other necessary transformations.
# * **Data normalization**: Scaling the data to a specific range or to have a mean of 0 and standard deviation of 1.
# * **Data reduction**: Reducing the size of the dataset by removing features or instances that are not relevant or useful for the model.
# 
# By performing these pre-processing steps, the data is made more suitable for use with machine learning models, which can improve the performance and accuracy of the model.
# 
# Before starting any feature engineering our dataframe looks like this:

# In[6]:


df_all.head()


# ## Ordinal Encoding
# 
# **Ordinal encoding** is a method used to convert categorical variables (variables that take on a limited number of values) into numerical variables that can be used in machine learning models. It is a type of encoding where the categorical values are assigned a unique integer value, such as 1, 2, 3, etc.
# 
# The key point of ordinal encoding is that the assigned integers have an explicit order, meaning that the categories have a natural rank or order, for example, "small", "medium", "large" or "low", "medium", "high".
# 
# This is different from one-hot encoding, in which each category is represented by a binary variable, and there is no inherent order among the categories.
# 
# Ordinal encoding can be useful when the categorical variable has a natural ordinal relationship between the categories, as it allows the model to capture the ordinal relationship. However, care should be taken to ensure that the ordinal relationship is not misinterpreted as a numeric relationship.

# In[7]:


# encode ordinal features
for col in ['BsmtQual', 'BsmtCond']:
    OE = OrdinalEncoder(categories=[['No', 'Po', 'Fa', 'TA', 'Gd', 'Ex']])
    df_all[col] = OE.fit_transform(df_all[[col]])

    
for col in ['ExterQual', 'ExterCond', 'KitchenQual']:
    OE = OrdinalEncoder(categories=[['Po', 'Fa', 'TA', 'Gd', 'Ex']])
    df_all[col] = OE.fit_transform(df_all[[col]])
    

OE = OrdinalEncoder(categories=[['N', 'P', 'Y']])
df_all['PavedDrive'] = OE.fit_transform(df_all[['PavedDrive']])


OE = OrdinalEncoder(categories=[['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr']])
df_all['Electrical'] = OE.fit_transform(df_all[['Electrical']])


for col in ['BsmtFinType1', 'BsmtFinType2']:
    OE = OrdinalEncoder(categories=[['No', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']])
    df_all[col] = OE.fit_transform(df_all[[col]])


OE = OrdinalEncoder(categories=[['C (all)', 'RH', 'RM', 'RL', 'FV']])
df_all['MSZoning'] = OE.fit_transform(df_all[['MSZoning']])


OE = OrdinalEncoder(categories=[['Slab', 'BrkTil', 'Stone', 'CBlock', 'Wood', 'PConc']])
df_all['Foundation'] = OE.fit_transform(df_all[['Foundation']])


OE = OrdinalEncoder(categories=[['MeadowV', 'IDOTRR', 'BrDale', 'Edwards', 'BrkSide', 'OldTown', 'NAmes', 'Sawyer', 'Mitchel', 'NPkVill', 'SWISU', 'Blueste', 'SawyerW', 'NWAmes', 'Gilbert', 'Blmngtn', 'ClearCr', 'Crawfor', 'CollgCr', 'Veenker', 'Timber', 'Somerst', 'NoRidge', 'StoneBr', 'NridgHt']])
df_all['Neighborhood'] = OE.fit_transform(df_all[['Neighborhood']])


OE = OrdinalEncoder(categories=[['None', 'BrkCmn', 'BrkFace', 'Stone']])
df_all['MasVnrType'] = OE.fit_transform(df_all[['MasVnrType']])


OE = OrdinalEncoder(categories=[['AdjLand', 'Abnorml','Alloca', 'Family', 'Normal', 'Partial']])
df_all['SaleCondition'] = OE.fit_transform(df_all[['SaleCondition']])


OE = OrdinalEncoder(categories=[['Gambrel', 'Gable','Hip', 'Mansard', 'Flat', 'Shed']])
df_all['RoofStyle'] = OE.fit_transform(df_all[['RoofStyle']])


# ## Feature Scaling
# 
# **Feature scaling** is a technique used to standardize the range of independent variables or features of data. In machine learning, it is a step of data pre-processing that is applied to improve the accuracy and convergence rate of models. Some models (such as SVM, KNN, and Neural Network) are based on the distance between two data points and feature scaling is must for these models to work correctly.
# 
# There are two common ways to perform feature scaling:
# 
# * **Min-Max scaling** (also called normalization): It scales the values of the feature to a given range (usually [0,1]). The formula to scale the feature value x is (x - min(x))/(max(x)-min(x))
# * **Standardization**: It scales the values of the feature to have zero mean and unit variance. The formula to standardize the feature value x is (x - mean(x))/ standard deviation(x)
# 
# It is important to note that feature scaling should be applied only to the independent variables, not the dependent variable. Additionally, it is also important to note that feature scaling should be done after splitting the data into training and test sets, and should be applied to the test set using the parameters of the scaling learned on the training set.

# In[8]:


# scale all numerical features
numerical_features = df_all.select_dtypes(exclude="object").columns

scaler = StandardScaler()

df_all[numerical_features] = scaler.fit_transform(df_all[numerical_features])


# ## Train-Test Split
# 
# We train/test split in machine learning to evaluate the performance of a model on unseen data. The training set is used to train the model, while the test set is used to evaluate the model's performance. 
# 
# This allows us to estimate how well the model will perform on new, unseen data, and identify any overfitting or underfitting issues with the model.

# In[9]:


# resplit into train and test sets
X_train = df_all[df_all['train_test'] == 'Train'].drop(['train_test'], axis =1)
X_test = df_all[df_all['train_test'] == 'Test'].drop(['train_test'], axis =1)
y_train = df_all[df_all['train_test'] == 'Train']['SalePrice']
y_test = df_all[df_all['train_test'] == 'Test']['SalePrice']

print(f'Before training models our train set has {X_train.shape} rows and columns and our test set has {X_test.shape} rows and columns.')


# # Conclusion <a id="conc"></a>
# 
# Our data is now ready for machine learning!
# 
# In this notebook we have taken our data from a rough, uncut gem, to a shinning diamond, ready for modelling!
# 
# * First we removed any remaining null values, to make sure our dataset is clean and machine readable
# 
# * Next we encoded all our categorical features to numerical 1's and 0's
# 
# * And last but not least, we scaled our numerical values to make them equal in the eyes of our model
# 
# **Now our data is ready to do the fun part - Modelling! This will be the next step in our journey, stay tuned and take care!**
