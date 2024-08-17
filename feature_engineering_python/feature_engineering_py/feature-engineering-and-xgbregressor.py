#!/usr/bin/env python
# coding: utf-8

# Welcome to my kernel, I hope you'll find it useful as a brief tutorial. It has two steps: data preprocessing, where I modify the datasets to achieve the best performance of the model, and building the model itself. 

# ![Photo from Pixabay](https://cdn.pixabay.com/photo/2016/01/19/17/08/vintage-1149558_1280.jpg)

# # 1-Data preprocessing 🛠

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

routeTrain='../input/house-prices-advanced-regression-techniques/train.csv'
routeTest='../input/house-prices-advanced-regression-techniques/test.csv'


datasetTrain=pd.read_csv(routeTrain)
datasetTest=pd.read_csv(routeTest)  



# In[2]:


datasetTrain.info()


# In[3]:


datasetTest.info()


# Drop the columns with a lot of NaN values **in both datasets**. These colums are MiscFeature, PoolQC, Fence, FireplaceQu,Alley.

# In[4]:


datasetTrain=datasetTrain.drop(['MiscFeature','PoolQC','Fence','FireplaceQu','Alley'],axis=1)
datasetTest=datasetTest.drop(['MiscFeature','PoolQC','Fence','FireplaceQu','Alley'],axis=1)


# Build a heatmap to check which variables are more correlated to SalePrice, the variable that you want to predict. The heatmap will show only the numerical variables. To study the correlation between the target variable and a categorical variable, you can do the ANOVA test. 

# In[5]:


corr=datasetTrain.corr()
plt.figure(figsize = (40,40))
sns.heatmap(corr,annot=True)


# In[6]:


cor_target = abs(corr["SalePrice"])#Selecting highly correlated features
important_numerical_features = cor_target[cor_target>0.5]
important_numerical_features


# The numerical variables that are correlated to SalePrice (correlation >=0.5) are:
# 
# * OverallQual: positive correlation, which means that SalePrice is bigger when the OverallQual is bigger.
# * YearBuilt: positive correlation.
# * YearRemodAdd: positive correlation.
# * TotalBsmtSF: positive correlation.
# * 1stFlrSF: positive correlation.
# * GrLivArea: positive correlation.
# * FullBath: positive correlation.
# * TotRmsAbvGrd: positive correlation.
# * GarageCars: positive correlation.
# * GarageArea: positive correlation.
# 

# In[7]:


#if you want to check the correlation between categorical variables and target variable
#Example-MSZoning
from scipy import stats
F, p = stats.f_oneway(datasetTrain[datasetTrain.MSZoning=='RL'].SalePrice,datasetTrain[datasetTrain.MSZoning=='RM'].SalePrice,datasetTrain[datasetTrain.MSZoning=='C (all)'].SalePrice,
                     datasetTrain[datasetTrain.MSZoning=='FV'].SalePrice,datasetTrain[datasetTrain.MSZoning=='RH'].SalePrice)
print(F)


# The next step is to check if any of the numerical variables are right or left skewed. You can do this with the function skew(). In a normal distribution, the value of skewness is zero. When a distribution is asymmetrical the tail of the distribution is skewed to one side-to the right (value of the skewness is positive) or to the left (skewness is negative). To fix skewed variables, use log transformation.

# In[8]:


numerical_columns=['OverallQual','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd','GarageCars','GarageArea','SalePrice']


# In[9]:


datasetTrain[numerical_columns].skew()


# In[10]:


fig = px.histogram(datasetTrain, x="GrLivArea")
fig.show()


# You can clearly see that GrLivArea is right skewed. 

# In[11]:


fig = px.histogram(datasetTrain, x="TotalBsmtSF")
fig.show()


# In[12]:


fig = px.histogram(datasetTrain, x="GarageArea")
fig.show()


# However, with the variables TotalBsmtSF and GarageArea there is a problem: when the house has not basement or garage, the value for these column in that specific row is 0. So if you apply log transformation to these two, you'll get a weird result. 
# 
# The solution I came up with was to throw away these rows.

# In[13]:


housesWithNoBasement=datasetTrain[datasetTrain.TotalBsmtSF==0]
housesWithNoBasement.shape #rows,columns


# In[14]:


housesWithNoGarage=datasetTrain[datasetTrain.GarageArea==0]
housesWithNoGarage.shape


# There were 1460 not null values in TotalBsmtSF and GarageArea columns so it's not big deal deleting 81 and 37 rows.

# In[15]:


datasetTrain=datasetTrain[datasetTrain.TotalBsmtSF>0]
datasetTrain=datasetTrain[datasetTrain.GarageArea>0]


# Let's select now the columns that had a significant correlation with the target variable, and SalePrice itself, and let's apply log transformation to them. 

# In[16]:


finalDatasetTrain=datasetTrain[numerical_columns]
finalDatasetTrain=np.log1p(finalDatasetTrain)


# In[17]:


finalDatasetTrain.skew()


# In[18]:


finalDatasetTrain.info()


# Let's prepare the test dataset.

# In[19]:


nun_columns=['OverallQual','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd','GarageCars','GarageArea']
finalTestDataset=datasetTest[nun_columns]


# In[20]:


finalTestDataset.info()


# In[21]:


finalTestDataset.GarageArea.unique()


# In[22]:


finalTestDataset.GarageCars.unique()


# In[23]:


finalTestDataset.update(finalTestDataset['GarageCars'].fillna(value=finalTestDataset['GarageCars'].mean(), inplace=True))


# In[24]:


finalTestDataset.update(finalTestDataset['GarageArea'].fillna(value=finalTestDataset['GarageArea'].mean(), inplace=True))


# In[25]:


finalTestDataset.info()


# In[26]:


finalTestDataset=np.log1p(finalTestDataset)


# # 2-Building the model 🏛

# In[27]:


#build a basic model for the tutorial
import xgboost as xgb
model=xgb.XGBRegressor(max_depth=3,eta=0.05,min_child_weight=4)


# In[28]:


features=finalDatasetTrain.drop('SalePrice',axis=1)
y=finalDatasetTrain['SalePrice']


# In[29]:


model.fit(features,y)
predictions=model.predict(finalTestDataset)


# In[30]:


finalPred=np.expm1(predictions)#you need to reverse log transformation



# In[31]:


sample_submission=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
submission = pd.DataFrame({'Id':sample_submission['Id'],'SalePrice':finalPred})
submission


# In[32]:


filename = 'Submission.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)

