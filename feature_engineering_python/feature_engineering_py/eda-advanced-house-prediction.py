#!/usr/bin/env python
# coding: utf-8

# **INTRODUCTION**
# 
# EDA(Exploratory Data Analysis) is one of the very first steps and an important step in any Data Science project. It is a process where data scientists try to study the data and draw non-obvious insights and infernces from the data which helps us in data pre-processing and later on in model building.
# 
# In this kernel, we will perform a detailed and an easy to understand EDA on one of the most popular dataset of Advanced House Prediction.
# 
# Let us first go through the basic steps that are followed in any Data Science project:
# 
# * Exploratory Data Analysis
# * Feature Engineering
# * Feature Selection
# * Feature Scaling
# * Hyperparameter tuning
# * Model building and deployment
# 
# So let us begin this exciting journey of going through the EDA of Advanced House Prediction!!
# 
# 

# 

# ****Importing the required libraries****

# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# Shape of Dataset

# In[5]:


df.shape


# In[6]:


df.columns


# **We now have our data in the form of a Data Frame. The first step in EDA is to find the missing values if there are any and analyze it's relationship with the target variable/feature. That analysis always helps to answer the question on how to replace the missing values**

# In[7]:


features_with_nan=[features for features in df.columns if df[features].isnull().sum()>1]

for feature in features_with_nan:
    print(feature, np.round(df[feature].isnull().mean(),4), '%missing values')


# **Since missing values are present in the dataset, we have to analyze the relationship of all independent features with the target variable of Sale price. To analyze, we first replace the NAN values and Non NAN with 1 and 0 respectively.**

# In[8]:


for feature in features_with_nan:
    data=df.copy()
    data[feature]=np.where(data[feature].isnull(),1,0)
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()


# Observation: There are some features which are related to NAN values. So during Feature Engineering, at the time of replacing 
# missing values, we should chose a way where it does not affect the relationship b\w NAN and sale price.

# Next step is to understand the different type of variables in dataset.
# 
# * Numerical Variable (Discrete and Continous variable)
# * Categorical Variable
# * Textual Data
# * Temporal (Date time variable)
# 
# Let's analyze each type of variable with the target variable (Sale price)

# 2) Numerical Variables: These variables are divided into Discrete variables(limited set of values) and Continous(range).
# 

# In[9]:


numerical_feat=[features for features in df.columns if df[features].dtypes!='O']
print('The number of numerical features are', len(numerical_feat))


# 

# In[10]:


df[numerical_feat].head()


# 2.1) Discrete Variables and it's relationship with target variable

# 

# In[11]:


discrete_feat=[feature for feature in numerical_feat if len(df[feature].unique())<25]
print(len(discrete_feat))


# In[12]:


discrete_feat


# In[13]:


for feature in discrete_feat:
    data=df.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('Sale Price')
    plt.show()


# In[ ]:





# 2.2) Temporal/Date time variable

# In[14]:


year_feature=[feature for feature in numerical_feat if 'Yr' in feature or 'Year' in feature]


# In[15]:


year_feature


# In[16]:


#Lets analyze the temporal datetime variable

df.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year sold')
plt.ylabel('sellin price')
plt.show()


# 2.3) Continous variable and it's relation with target variable

# In[17]:


continous_feat=[feature for feature in numerical_feat if feature not in discrete_feat+year_feature]
continous_feat 


# In[18]:


for feature in continous_feat:              
    data=df.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel('Sale price')
    plt.show()


# >  Observation: From these histograms, we can observe that many features are skewed distributions. Our job is to convert those into Gaussian Distribution as one of the major assumptions of regression algo is that the data should be normally distributed. So we use Log Normal Transoformation to convert it into Gaussian Distribution.

# In[19]:


for feature in continous_feat:
    data=df.copy()
    data['SalePrice']=np.log(data['SalePrice']+1)
    data[feature]=np.log(data[feature]+1)
    plt.scatter(data[feature], data['SalePrice'])
    plt.xlabel(feature)
    plt.ylabel('Sale Price')
    plt.show()
    


# 3) Outlier Analysis using boxplot

# In[20]:


for feature in continous_feat:
    data=df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()


# Observation: We observe from above plots that there are many outliers present in data. So while doing feature engineering i.e. replacing NAN values etc , we need to select the right techniques while doing so.

# 4) Categorical Variables and it's relation with target variable

# In[21]:


categorical_feat=[feature for feature in df.columns if df[feature].dtypes=='O']
print(len(categorical_feat))


# In[22]:


df[categorical_feat].head()


# 4.1) Finding cardinality of each category

# In[23]:


for feature in categorical_feat:
    print("The feature is",feature,"and number of categories are",len(df[feature].unique()))


# In[24]:


for feature in categorical_feat:
    data=df.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('Sale Price')
    plt.title(feature)
    plt.show()


# In[ ]:




