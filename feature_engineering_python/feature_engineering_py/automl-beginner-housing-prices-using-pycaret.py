#!/usr/bin/env python
# coding: utf-8

# # References and Acknowledgements
# https://www.kaggle.com/kritidoneria/automl-titanic-using-pycaret
# 
# https://pycaret.org/ (Official Documentation)
# 

# Hi,
# After seeing the response to [this notebook](https://www.kaggle.com/kritidoneria/titanic-using-pycaret-100-lines-of-code/comments) of mine wherein I made an entry for Titanic competition,I am doing a Regression analysis using PyCaret.
# In this, I'll focus on very basic steps.
# 
# So you have a Kaggle account,what next? What if I tell you you can create your very first submission in less than 100 lines of code?
# 
# No,I'm not talking the usual Linear regression. I'm talking advanced Kaggle concepts like Feature engineering,Blending,Stackimg and Ensembling?
# 
# Welcome Pycaret,a low Code library developed by Moez Ali,which helps professional data scientists develop prototypes quickly with very few lines of code.
# 
# It provides a great starting point to rule out what works for your data and what doesn't,so I highly recommend this. In this code, We will read the data and create models and final predictions. I do recommend reading the official documentation while following along,and typing your own code by reading this notebook.
# 

# In[1]:


#Pycaret needs to be installed
get_ipython().system('pip install pycaret --user')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# # Data fields
# 
# *Here's a brief version of what you'll find in the data description file.*
# 
# * SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
# * MSSubClass: The building class
# * MSZoning: The general zoning classification
# * LotFrontage: Linear feet of street connected to property
# * LotArea: Lot size in square feet
# * Street: Type of road access
# * Alley: Type of alley access
# * LotShape: General shape of property
# * LandContour: Flatness of the property
# * Utilities: Type of utilities available
# * LotConfig: Lot configuration
# * LandSlope: Slope of property
# * Neighborhood: Physical locations within Ames city limits
# * Condition1: Proximity to main road or railroad
# * Condition2: Proximity to main road or railroad (if a second is present)
# * BldgType: Type of dwelling
# * HouseStyle: Style of dwelling
# * OverallQual: Overall material and finish quality
# * OverallCond: Overall condition rating
# * YearBuilt: Original construction date
# * YearRemodAdd: Remodel date
# * RoofStyle: Type of roof
# * RoofMatl: Roof material
# * Exterior1st: Exterior covering on house
# * Exterior2nd: Exterior covering on house (if more than one material)
# * MasVnrType: Masonry veneer type
# * MasVnrArea: Masonry veneer area in square feet
# * ExterQual: Exterior material quality
# * ExterCond: Present condition of the material on the exterior
# * Foundation: Type of foundation
# * BsmtQual: Height of the basement
# * BsmtCond: General condition of the basement
# * BsmtExposure: Walkout or garden level basement walls
# * BsmtFinType1: Quality of basement finished area
# * BsmtFinSF1: Type 1 finished square feet
# * BsmtFinType2: Quality of second finished area (if present)
# * BsmtFinSF2: Type 2 finished square feet
# * BsmtUnfSF: Unfinished square feet of basement area
# * TotalBsmtSF: Total square feet of basement area
# * Heating: Type of heating
# * HeatingQC: Heating quality and condition
# * CentralAir: Central air conditioning
# * Electrical: Electrical system
# * 1stFlrSF: First Floor square feet
# * 2ndFlrSF: Second floor square feet
# * LowQualFinSF: Low quality finished square feet (all floors)
# * GrLivArea: Above grade (ground) living area square feet
# * BsmtFullBath: Basement full bathrooms
# * BsmtHalfBath: Basement half bathrooms
# * FullBath: Full bathrooms above grade
# * HalfBath: Half baths above grade
# * Bedroom: Number of bedrooms above basement level
# * Kitchen: Number of kitchens
# * KitchenQual: Kitchen quality
# * TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# * Functional: Home functionality rating
# * Fireplaces: Number of fireplaces
# * FireplaceQu: Fireplace quality
# * GarageType: Garage location
# * GarageYrBlt: Year garage was built
# * GarageFinish: Interior finish of the garage
# * GarageCars: Size of garage in car capacity
# * GarageArea: Size of garage in square feet
# * GarageQual: Garage quality
# * GarageCond: Garage condition
# * PavedDrive: Paved driveway
# * WoodDeckSF: Wood deck area in square feet
# * OpenPorchSF: Open porch area in square feet
# * EnclosedPorch: Enclosed porch area in square feet
# * 3SsnPorch: Three season porch area in square feet
# * ScreenPorch: Screen porch area in square feet
# * PoolArea: Pool area in square feet
# * PoolQC: Pool quality
# * Fence: Fence quality
# * MiscFeature: Miscellaneous feature not covered in other categories
# * MiscVal: $Value of miscellaneous feature
# * MoSold: Month Sold
# * YrSold: Year Sold
# * SaleType: Type of sale
# * SaleCondition: Condition of sale

# This is a very detailed features list,however I'd like you to think of a few more factors that are not present in this data.
# This helps with critical thinking in Data science
# 
# (2008 housing bubble burst for example)

# In[2]:


train.head()


# In[3]:


test.head()


# In[ ]:





# In[4]:


#Importing regression model from PyCaret. For classification,see the notebook mentioned above.
from pycaret.regression import *


# **Pycaret Setup**
# 
# This is where magic happens.One line does all of these things:
# 
# * I will tell the model to ignore certain features with high cardinality,the target column,and give my session an id.
# * I will also pass categorical features here.
# * I will normalize the data
# * I will pass multicollinearity handling and outlier handling as true so that it takes care of it implicitly
# * I will also experiment with using PCA to reduce dimensionality here and set feature selection to true.
# 
# I highly encourage you to look up each of these terms.

# # Setting up Pycaret

# In[5]:


reg = setup(data = train, 
             target = 'SalePrice',
             numeric_imputation = 'mean',
             categorical_features = ['MSZoning','Exterior1st','Exterior2nd','KitchenQual','Functional','SaleType',
                                     'Street','LotShape','LandContour','LotConfig','LandSlope','Neighborhood',   
                                     'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',    
                                     'MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond',   
                                     'BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir',   
                                     'Electrical','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive',
                                     'SaleCondition']  , 
             ignore_features = ['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','Utilities'],
             feature_selection = True,
             pca=True,
             remove_multicollinearity=True,
             remove_outliers = True,
             normalize = True,
             silent = True)


# # Viewing model results

# In[6]:


#Now lets visualize the results for various models
compare_models(sort='RMSE') # because this is the evaluation metric of the competition


# # Selecting and tuning the best model

# In[7]:


#selecting the best model
best = automl(optimize='rmse')

best


# Time for some hyperparameter tuning. You're right, Pycaret does that for us as well!

# In[8]:


cb = create_model('huber')
tuned_cb = tune_model(cb)


# # Writing the submissions
# Pycaret also allows for model blending like stacking,ensembling etc.For simplicity I shall omit it here. If you do fork this notebook, I highly encourage you to try it.Let's just write the best submission here.

# In[9]:


sample=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')


# In[10]:


#Let's create the final submission!!
predictions = predict_model(tuned_cb, data = test)
sample['SalePrice'] = predictions['Label']
sample.to_csv('submission.csv',index=False)
sample.head()


# In[11]:


plot_model(tuned_cb,'feature')

