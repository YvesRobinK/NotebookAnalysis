#!/usr/bin/env python
# coding: utf-8

# # FEATURE ENGINEERING
# <font size="4">Feature engineering is one of the important step that needs to be done in Data science. By feature engineering we mean, manipulate the data or handle the dataset so that we can improve he training of our machine learning model. The effectiveness of feature engineering depends on knowledge of data sources and mainly the business problem. Let's look at the different techniques that are used to do Feature Engineering.
# If you like my notebook please upvote. :)) </font>

# ## Table of Contents
# 1. Handling Missing Values
# 2. Handling Imbalanced data
# 3. Handling Outliers
# 4. Encoding
# 5. Feature Scaling

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Image


# **Reading Files**

# In[ ]:


train=pd.read_csv('../input/spaceship-titanic/train.csv')
test=pd.read_csv('../input/spaceship-titanic/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# #### Note: This notebook is ONLY for the feature engineering techniques, there is no modelling part or any EDA part, if you want to see Checkout my spaceship titanic notebook in which I have done all. 

# #### Dropping columns that we don't use

# In[ ]:


train=train.drop(['Name','Cabin','PassengerId','Destination'],axis=1)
train.head(2)


# ## 1. Handling Missing Values
# Missing values can be handled by many ways. Let's do this.
# 

# > looking for null values in training data

# In[ ]:


train.isna().sum()


# #### > We can see except transported column every column has null values.

# <font size="4">For numeric columns either we can fill null values with mean or with median, we use median incase of outliers, but mostly median is used and is good to use, let's try both</font>

# In[ ]:


train.Age=train.Age.fillna(train.Age.median())
train.Age.isna().sum()


# In[ ]:


train.RoomService=train.RoomService.fillna(train.RoomService.mean())
train.FoodCourt=train.FoodCourt.fillna(train.FoodCourt.mean())
train.ShoppingMall=train.ShoppingMall.fillna(train.ShoppingMall.mean())
train.Spa=train.Spa.fillna(train.Spa.mean())
train.VRDeck=train.VRDeck.fillna(train.VRDeck.mean())
train[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].isna().sum()


# <font size="4">So far we have handled numeric columns. We have seen both ways filling with mean and filling with median (you can fill foodcourt,roomservice and others null values with median I used mean just to give you idea). For Categorical columns we will use mode to fill null values.</font>

# In[ ]:


train['HomePlanet']=train['HomePlanet'].fillna(train['HomePlanet'].mode()[0])
train['CryoSleep']=train['CryoSleep'].fillna(train['CryoSleep'].mode()[0])
train['VIP']=train['VIP'].fillna(train['VIP'].mode()[0])
train[['HomePlanet','CryoSleep','VIP']].isna().sum()


# <font size="4">There are different techniques also here it is:
# * You can drop null value row by doing this **train.dropna(how='all')**
# * You can drop features incase there are more than 50% null values to do this **train.drop(['colname'],axis=1).**
# * Also you can replace it with zero **train.fillna(0)**.
# >     Try to do replace method if you do dropping method there is chance that most of the importatnt information will lose. So it is better to choose replace method.
#     <font>

# ## 2. Handling Imbalanced Data
# <font size="3.5">We need to handle imbalanced data inorder to reduce overfitting and underfitting problem. Imbalanced data somehting like if we have 2 output values say 1 and 0, if 1 occurs 95% and 0 only 5% then it is said to be imbalanced we have to deal with it.</font>

# In[ ]:


print(train.Transported.value_counts())


# we can see both are almost equal so don't need to handle it. Don't worry I will make on how to handle imbalanced data separately in which this problem occurs. :)

# ## 3. Handling Outliers
# 
# <font size="4">Plot boxplot to every numeric(continuous) features and check if any values are out of bounds or not. If there, they are called outliers.</font>

# In[ ]:


col_to_plot=['RoomService','ShoppingMall','FoodCourt','Spa','VRDeck']
for i in col_to_plot:
    sns.boxplot(train[i])
    plt.show()


# <font size="3.7">As we can see there outliers in these columns let's handle them. </font>

# <font size="4">Following are the steps to follow inorder handle outliers.
#     
# i. Calculate the quantile values 25%, 75%
#     
# ii. Calculate interquartile range (IQR)
#     
# iii. Calculate lower and upper bound.
# <font>
# 
# <font size="4">There are different methods we will on how to deal with outliers here are some.
#     
# i. Replace with mean.
#     
# ii. Replace with median.
#     
# iii. Replace with quantile values.
#     
# iv. Drop outliers
#  <font>

# <font size="3.8">**Replacing with mean.** </font>

# In[ ]:


Q1=train['RoomService'].quantile(0.25) #25%
Q3=train['RoomService'].quantile(0.75) #75%


# In[ ]:


IQR=Q3-Q1 #Interquartilerange


# In[ ]:


lower_bound=Q3-(1.5*IQR)
upper_bound=Q3+(1.5*IQR)


# In[ ]:


out1=[(train['RoomService']<lower_bound)]
out2=[(train['RoomService']>upper_bound)]
train['RoomService'].replace(out1,train['RoomService'].mean(),inplace=True)
train['RoomService'].replace(out2,train['RoomService'].mean(),inplace=True)


# <font size="3.8">**Replace with quantile values.**</font>

# In[ ]:


Q1=train['FoodCourt'].quantile(0.25) #25%
Q3=train['FoodCourt'].quantile(0.75) #75%


# In[ ]:


IQR=Q3-Q1 #Interquartilerange


# In[ ]:


lower_bound=Q3-(1.5*IQR)
upper_bound=Q3+(1.5*IQR)


# In[ ]:


outf1=[(train['FoodCourt']<lower_bound)]
outf2=[(train['FoodCourt']>upper_bound)]
train['FoodCourt'].replace(outf1,lower_bound,inplace=True)
train['FoodCourt'].replace(outf2,upper_bound,inplace=True)


# <font size="4">To drop the outliers try this 
#     
#    **out=[(train['Spa']<lower_bound)|(train['Spa']>upper_bound)].index**
#     
#    **train.drop(out,inplace=True)**
#     
#    </font>

# ## 4. Encoding 
# We do encoding as dataset contains object type data so we have to convert it into numeric as model understand well numeric values.
# Two techniques mainly used:
# 
# **i. Label encoding**: This is mostly used when we have data that we can compare that one is smaller than other like if we have 3 values low, medium and high so we know that low is less than medium and medium is less than high so by applying label encoding it will convert low=0, medium=1, and high=2.
# 
# **ii. OneHotEncoding**: This is mostly used when we have data that we are not able to compare that which one is lesser or greater, like if we have column in which we have values like mango, oreange, apple we cannot compare them so onehotencoding make new columns for each and replace 1 if they occur else zero.
# 
# **Note: I would suggest not to use onehotencoding when there is situation where you already have 100 columns and the column you are going to encode also have 100 unique values it will create 100 more columns so better go for labelencoding in that case, or create threshold like if you have 20 unique vlues you can do onehotencoding else labelencoding.**

# ### LabelEncoding 

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
train['Transported']=le.fit_transform(train['Transported'])
train['VIP']=le.transform(train['VIP'])


# In[ ]:


train['CryoSleep']=le.transform(train['CryoSleep'])


# In[ ]:


train.head()


# <font size="4">  **You can see the results.**  </font>

# ### OneHotEncoding 

# In[ ]:


dummy=pd.get_dummies(train['HomePlanet'])
train=pd.concat([train,dummy],axis=1)
train.head()


# In[ ]:


train.drop('HomePlanet',axis=1,inplace=True)
train.head()


# **You can see the results after doing onehotencoding**

# ## 5. Feature Scaling 
# <font size="3.6"> To reduce the variance effect we use feature scaling. Various methods are used to do this.</font>

# ### 5.1 Standardization
# <font size="4">We use this when all values are high not 0 and 1.</font>

# In[ ]:


#just commenting out you can use this as well I have scaled age feature using normaliztion you can see below 

# from sklearn.preprocessing import StandardScaler
# ss=StandardScaler()
# train['Age']=ss.fit_transform(train[['Age']])
# train.head(2)


# ### 5.2 Normalization
# <font size="4"> It is a method to rescales the feature in rage of 0,1 by subtracting the minimum value of the feature then dividing by the range. </font>

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()
train['Age']=mms.fit_transform(train[['Age']])
train['RoomService']=mms.transform(train[['RoomService']])
train['ShoppingMall']=mms.transform(train[['ShoppingMall']])
train['FoodCourt']=mms.transform(train[['FoodCourt']])
train['Spa']=mms.transform(train[['Spa']])
train['VRDeck']=mms.transform(train[['VRDeck']])


# In[ ]:


train.head()


# <font size="4">**You can see the results**</font>
# 
# <font size="4">You can also try Mean normalization which rescales features in range of -1 to 1 with mean=0.</font>

# ## 6. Feature Selection 
# <font size="4">In this we select import independent features which are more related with dependent features. This will help to build a good model for our problem.</font>

# ### 6.1 Correlation with Heatmap
# <font size="3">Heatmap is a graphical representation of 2D (two-dimensional) data. Each data value represents in a matrix. Check the relation between the independent and dependent feature and choose features to build a model.</font>

# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(train.corr(),annot=True)
plt.show()


# ### 6.2 UniVariate Seelction
# <font size="4">In this we will use statistical method which will help us in selecting feature and these features have the strongest relationship with the dependent feature.
# We will use SelectKBest statistical method which will help us in selecting specific number of features. </font>

# In[ ]:


x=train.drop('Transported',axis=1)
y=train['Transported']


# In[ ]:


from sklearn.feature_selection import SelectKBest,chi2
model=SelectKBest(score_func=chi2,k=5)
result=model.fit(x,y)
score=pd.DataFrame(result.scores_)
columns=pd.DataFrame(x.columns)
df=pd.concat([columns,score],axis=1)
df.columns=['Col','Score']
df


# <font size="4">Choose those features who has the highest score as it indicates that those features are more related with the dependent feature. </font>
