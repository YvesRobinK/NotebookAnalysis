#!/usr/bin/env python
# coding: utf-8

# This notebook is an extension of my previous work[here](http://https://www.kaggle.com/code/qiaoningchen/house-price-prediction). Previously RandomForestRegression model performed the best among the several regression models employed. Now our goal is to how to improve that model. Two things have been done. First, we transform one feature variable and add that scaled variable to the features. Second, we reduce the number of feature variables based on the model importance analysis and EDA.

# ## Environment Setup

# In[1]:


import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import sklearn
from warnings import simplefilter


simplefilter('ignore')
import math

from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold


# ## Data Loading and Preview

# In[2]:


train_data = pd.read_csv("/kaggle/input/playground-series-s3e6/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s3e6/test.csv")


# In[3]:


train_data.head()


# In[4]:


print("data  information from train.csv:\n")
train_data.info()


# In[5]:


print("data  information from test.csv:\n")
test_data.info()


# > ### Observations: 
# 1. There are the same feature variables in both train data and test data;
# 2. There aren't any missing values from the datasets;
# 3. Some feature variables don't have the right data types.We will convert them. 

# In[6]:


sns.histplot(data=train_data["price"],kde=True)


# > ### Price Distribution
# The price distribution is rather uniform.

# ## EDA and Feature Engineering

# > ### Correlation Heatmap

# In[7]:


plt.figure(figsize=(12, 6))
sns.heatmap(train_data.corr(),
            cmap = 'BrBG',
            fmt = '.2f',
            linewidths = 2,
            annot = True)


# > ### Observations:
# The heatmap shows that the target variable("price") has a very strong correlation with "squareMeters" and has a weak correlation with "numberOfRooms","floors","cityCode","made","hasStormProctector". This observation is the base for our following feature engineering. 

# > ### Data Transformation(Data Type)

# In[8]:


# using dictionary to convert specific columns, prepare for modeling

convert_dict = {'hasYard': object,
                'hasPool': object,
                'isNewBuilt':object,
                'hasStormProtector':object,
                'hasStorageRoom':object,
                'cityPartRange':object,
                'hasGuestRoom':object,          
                }
 
train_data = train_data.astype(convert_dict)
test_data = test_data.astype(convert_dict)
train_data.dtypes


# In[9]:


test_data.dtypes


# In[10]:


target = "price"
numerical= [col for col in train_data.select_dtypes(["int64", "float64","int32", "float32"]).columns if col not in [target,"id"]]
categorical = [col for col in train_data.select_dtypes("object").columns if col!=target]
print("Numerical Columns:\n",numerical)
print("\n")
print("Categorical Columns:\n",categorical)


# In[11]:


unique_values = []
for col in categorical:
  unique_values.append(train_data[col].unique().size)
plt.figure(figsize=(10,6))
plt.title('No. Unique values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=categorical,y=unique_values)


# In[12]:


plt.figure(figsize=(18, 36))
plt.title('Categorical Features: Distribution')
plt.xticks(rotation=90)
index = 1
 
for col in categorical:
    y = train_data[col].value_counts()
    plt.subplot(11, 4, index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index), y=y)
    index += 1


# In[13]:


# train_data:
# Distributions of the numerical features: 
rows, cols = math.ceil(len(numerical) / 4), 4
fig, axes = plt.subplots(rows, cols, figsize=(20, 18))

for col, ax in zip(numerical, axes.flatten()):
    sns.histplot(data=train_data,
                 x=col,
                 ax=ax,
                 bins=60)
    ax.set_ylabel('')
    
for ax in axes.flatten():
    if not ax.get_xlabel():
        ax.set_visible(False)

plt.tight_layout()


# > ### Feature Engineering
# From the heatmap, 'squareMeters'is strongly correlated with the target variable, so we decide to explore its distribution and see if we can do some transformation.The purpose of feature engineering is to make the data "better" for machine learning models.

# In[14]:


# train data
# 'squareMeters' distribution: is highly skewed.
# Transform it to logarithmic scale, add another feature to the dataset

train_data['squareMeters_log'] = np.log10(train_data['squareMeters'].clip(lower=1.0))

rows, cols = 1, 2
fig, axes = plt.subplots(rows, cols, figsize=(16, 5))

for col, ax in zip(['squareMeters', 'squareMeters_log'], axes):
    sns.histplot(data=train_data,
                 x=col,
                 ax=ax,
                 bins=60)

plt.tight_layout()


# In[15]:


# test_data
# 'squareMeters' distribution: is highly skewed.
# Transform it to logarithmic scale, add another feature to the dataset

test_data['squareMeters_log'] = np.log10(test_data['squareMeters'].clip(lower=1.0))

rows, cols = 1, 2
fig, axes = plt.subplots(rows, cols, figsize=(16, 5))

for col, ax in zip(['squareMeters', 'squareMeters_log'], axes):
    sns.histplot(data=test_data,
                 x=col,
                 ax=ax,
                 bins=60)

plt.tight_layout()


# > ### Observations:
# * The train data shows the houses are much bigger than those from the test dataset. 
# * From the train data, "squareMeters" has a much skewed distribution than that from test data. 
# * After scaling "'squareMeters'into a logarithmic scale, the distributions look much better and alike.

# In[16]:


train_data.info()


# In[17]:


## Make sure that we have included the new feature varible 'squareMeters_log'
target = "price"
numerical= [col for col in train_data.select_dtypes(["int64", "float64","int32", "float32"]).columns if col not in [target,"id"]]
categorical = [col for col in train_data.select_dtypes("object").columns if col!=target]
print("Numerical Columns:\n",numerical)
print("\n")
print("Categorical Columns:\n",categorical)


# ### Prepare for Machine Learing
# We use get_dummies function for imputation and encoding because both train data and test data have the same feature variables. This greatly simplifies the coding. 

# In[18]:


features = numerical + categorical
X          = pd.get_dummies(train_data[features])
test_full  = pd.get_dummies(test_data[features])
y          = train_data[target]


# In[19]:


y.describe()


# In[20]:


X.head()


# In[21]:


y.head()


# ## Machine Learning with RandomForestRegressor

# > ### Train the model

# In[22]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=8)
rd_model = RandomForestRegressor(n_estimators = 500,random_state = 0)
rd_model.fit(X_train,y_train)


# > ### Performance Metrics
# 
# In the first 5 playground competitions we are dealing with classification problems where the predicted target belongs to one or another class:
# a binary classification such as Employee Attrition, Credit Card Fraud and  Stroke Prediction, or multi-classification problem such as wine quality prediction. For the classification problems, accuracy, confusion matrix, log-loss, and AUC-ROC are some of the most popular metrics. 
# 
# However in this competition episode, the target varibale "price" is a continous variable, therefore it is not a classification problem. This is a regression problem. For regression models, we use MAE (mean abolute error), Mean Squared Error(MSE),Root Mean Squared Error(RMSE) and others to measure model performance.

# In[23]:


score_RandReg=mean_absolute_error(y_valid,rd_model.predict(X_valid))
score_RandReg
mse = mean_squared_error(y_valid,rd_model.predict(X_valid))
rmse = mse**.5
rmse


# In[24]:


# Before scaling 'squreMeters'
#Mean Absolute Error(MAE)	8846.287615 
#Root Mean Squared Error	135925.396876
#Average House Price	4634456.896876



d = {'Mean Absolute Error(MAE)': [score_RandReg],
    'Root Mean Squared Error': [rmse],
     'Average House Price':y.mean(),
}
df = pd.DataFrame(data=d).T
df = df.rename(index={0:"metric",1:"value"})
df.style.set_caption("Performance Metrics for a Regression Model")


# 

# In[25]:


importances = rd_model.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(X_train.shape[1]), X_train.columns[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()


# In[26]:


rd_pred = rd_model.predict(test_full)
# output predictions data
output = pd.DataFrame({'id': test_data.id, 'price': rd_pred})
#output.to_csv('submission_randomre.csv', index=False)
output.to_csv('submission.csv', index=False)


# ### Feature Engineering

# In[27]:


#reduced_features  = ['squareMeters','squareMeters_log','numberOfRooms','floors','made','cityCode','hasStormProtector']
## MAE = 7378(before scaling),7432(after scaling)
#reduced_features = ['squareMeters','squareMeters_log','numberOfRooms','floors','made','cityCode'] ##MAE = 7101(B),7070(A)
#reduced_features  = ['squareMeters','squareMeters_log','numberOfRooms','floors'] ## MAE = 6687(B),6586(A)
reduced_features  = ['squareMeters','squareMeters_log','numberOfRooms']  ##MAE = 6122(Before),6101(After)
#reduced_features  = ['squareMeters','squareMeters_log'] ## MAE=7076(before),7076(after)
#reduced_features  = ['squareMeters_log'] ## MAE=7079
X2         = pd.get_dummies(train_data[reduced_features])
test_full_2 = pd.get_dummies(test_data[reduced_features])
y         = train_data[target]


# In[28]:


X2.head()


# In[29]:


X_train, X_valid, y_train, y_valid = train_test_split(X2, y, test_size=0.2, random_state=8)
rd_model = RandomForestRegressor(n_estimators = 500,random_state = 0)
rd_model.fit(X_train,y_train)


# In[30]:


score_RandReg2=mean_absolute_error(y_valid,rd_model.predict(X_valid))
mse = mean_squared_error(y_valid,rd_model.predict(X_valid))
rmse2 = mse**.5


# In[31]:


d = {'Mean Absolute Error(MAE)': [score_RandReg2],
    'Root Mean Squared Error': [rmse2],
     'Average House Price':y.mean(),
}
df = pd.DataFrame(data=d).T
df = df.rename(index={0:"metric",1:"value"})
df.style.set_caption("Performance Metrics for a Regression Model")


# In[32]:


rd_pred = rd_model.predict(test_full_2)
# output predictions data
output = pd.DataFrame({'id': test_data.id, 'price': rd_pred})
#output.to_csv('submission_randomReduced.csv', index=False)


# ## Summary of Performance

# From the table below, we can see that scaling and reduced features improve the performance of the model. 

# In[33]:


d = {'Mean Absolute Error(MAE)': [9012.50756356791,score_RandReg,score_RandReg2],
    'Root Mean Squared Error': [136229.32416921048,rmse,rmse2],
     'Average House Price':[y.mean(),y.mean(),y.mean()]
}
df = pd.DataFrame(data=d)
df = df.rename(index={0:"full features(no scaling)",1:"full features(with scaling)",2:"reduced features(with scaling)"})
df.style.set_caption("Performance Metrics for a Regression Model")

