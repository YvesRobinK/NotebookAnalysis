#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Analysis - Playground Series Season 3, Episode 4
# 
# In this notebook, the main objective was to analyze the data for the playground series, related to Credit Card Fraud Analysis and Synthetically-Generated Datasets. This project is presented on the following topics:
# 
# 1. Initial information cleaning
# 2. Comparison between the [actual data](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and the [competition data](https://www.kaggle.com/competitions/playground-series-s3e4/data)
# 3. P-value and correlation analysis
# 4. Training and test data - feature engineering
# 5. Prediction Models
# 6. Final Ensemble and Submission
# 
# In this case, the modeling didn't consider the hyperparameter tuning, but in future implementations, the idea is to build CrossValidation and GridSearch tools for improving the modeling performance.

# # Initial information cleaning
# 
# At first, the main libraries were loaded on the notebook, as to be easier to replicate this analysis and to avoid problems by running certain cells

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing
import seaborn as sns # plotting tool
import matplotlib.pyplot as plt # plotting tool
from scipy import stats # statistical review
from sklearn.model_selection import train_test_split # train - test split
from sklearn.pipeline import Pipeline # data transformation
from sklearn.preprocessing import StandardScaler # data transformation
from sklearn.metrics import roc_auc_score # Competition metric
from xgboost import XGBClassifier # Xgboost
from sklearn.linear_model import LogisticRegression #Sklearn LR
from catboost import CatBoostClassifier #CatBoost


# After, the two databases were loaded, corresponding to:
# 1. **train:** Competition dataset
# 2. **original:** Initial dataset and baseline for possible data augmentation

# In[2]:


train=pd.read_csv("/kaggle/input/playground-series-s3e4/train.csv")
train.head()


# In[3]:


original=pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
original.head()


# Both dataframes have the same columns, with the only difference corresponding to the `ìd` column, but it is not relevant for this exercise. With this in consideration I used the `.describe()` attribute for showing the main trends of each of the datasets.

# In[4]:


train.describe()


# In[5]:


original.describe()


# At a first glance, it is difficult to identify specific trends in each dataset but the magnitude of the maximum values compared with the mean is something that must be considered in the modeling. 
# 
# Then, the `.info()` property helped to identify that there are no null values on both datasets, showing that the data treatment for the prediction modeling will be of low complexity. 

# In[6]:


train.info()


# In[7]:


original.info()


# # Comparison between the actual data and the competition data
# 
# As there are more than 30 columns, the data was split into groups of 5 variables to be able to compare the information:

# In[8]:


#Plotting columns
list_cols=list(train.columns)
list_cols.remove("id")
print(list_cols)


# Firstly, with the column `Time`, the plot shows that there are more values on the original dataset considering the timestamps only.
# 
# The other four variables (V1-V4) present a similar distribution, but with the effects of the previously mentioned outliers the plot is "moved" and it represents that a possible transformation would be used for reducing the effect of these values.

# In[9]:


for i in list_cols[:5]:
    plt.figure(figsize=(15,5))
    plt.grid()
    sns.histplot(train[i])
    sns.histplot(original[i],color='red')
    plt.legend(['train','original'])


# Later on, the same trend happens in the columns (V5-V9) with the presence of outliers. On plots of V5 and V9, there is a difference in the right values of the distribution, being higher for the original dataset.

# In[10]:


for i in list_cols[5:10]:
    plt.figure(figsize=(15,5))
    plt.grid()
    sns.histplot(train[i])
    sns.histplot(original[i],color='red')
    plt.legend(['train','original'])


# This phenomenon repeats with the columns from V10 to V25, where the original dataset has more values than the generated one.

# In[11]:


for i in list_cols[10:15]:
    plt.figure(figsize=(15,5))
    plt.grid()
    sns.histplot(train[i])
    sns.histplot(original[i],color='red')
    plt.legend(['train','original'])


# In[12]:


for i in list_cols[15:20]:
    plt.figure(figsize=(15,5))
    plt.grid()
    sns.histplot(train[i])
    sns.histplot(original[i],color='red')
    plt.legend(['train','original'])


# In[13]:


for i in list_cols[20:25]:
    plt.figure(figsize=(15,5))
    plt.grid()
    sns.histplot(train[i])
    sns.histplot(original[i],color='red')
    plt.legend(['train','original'])


# Then, with the final plots, this trend repeats while, in the Amount column, the effect of the outliers is even higher.

# In[14]:


for i in list_cols[25:]:
    plt.figure(figsize=(15,5))
    plt.grid()
    sns.histplot(train[i])
    sns.histplot(original[i],color='red')
    plt.legend(['train','original'])


# # P-value and correlation analysis
# 
# Then, a [Kolmogorov-Smirnov](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html) test was made between the column to see if they have the exact data distribution, but as referred on the following cell, these values are lower than a significance of 0.05, implying that they don't have the same distribution.
# 
# This could happen for different circumstances as the effect of the data generation, but overall on the plots the data seems to follow a similar trend.

# In[15]:


for i in list_cols:
    result=stats.ks_2samp(original[i], train[i])
    print(f"The p value of {i} column is {result[1]}")


# After, the total rows per category were counted, finding that the relation between the Fraud and Non-Fraud are the same on the two datasets:

# In[16]:


train.groupby(by="Class").count()["id"]


# In[17]:


original.groupby(by="Class").count()["Time"]


# Finally, the heatmaps of both datasets were plotted, showing that there is no correlation between the variables and with a different trend from the artificially generated data:

# In[18]:


plt.figure(figsize=(20,10))
sns.heatmap(train[list_cols].corr())


# In[19]:


plt.figure(figsize=(20,10))
sns.heatmap(original[list_cols].corr())


# # Training and test data - feature engineering
# 
# With these analyses, two additional columns were added, corresponding to the total hours and days calculated from the following equations and being inspired from https://www.kaggle.com/competitions/playground-series-s3e4/discussion/380771.

# In[20]:


#Feature engineering on both datasets
train['hour'] = train['Time'] % (24 * 3600) // 3600
train['day'] = (train['Time'] // (24 * 3600)) % 7
original['hour'] = original['Time'] % (24 * 3600) // 3600
original['day'] = (original['Time'] // (24 * 3600)) % 7


# In[21]:


try:
    train.drop(columns="id",inplace=True)
except:
    pass
train.head()


# Afterward, both dataframes were joined to be used as a piece of extended information for the prediction models:

# In[22]:


final=pd.concat([original,train])
final.head()


# In[23]:


#Train - Test splitting
X_train, X_test, y_train, y_test = train_test_split(final.drop(columns="Class"), final["Class"] , test_size=0.1, random_state=1)


# # Prediction Models
# For this comparison, the following three models were trained
# 
# 1. XGBoost
# 2. Logistic Regression
# 3. Catboost
# 
# In all of them, a simple pipeline was used, using only a `StandardScaler()`. For all of them the XGBoost is the slowest and at the same time gets the best results related to the AUC calculated with the sklearn library. 

# In[24]:


get_ipython().run_cell_magic('time', '', "pipe_1=Pipeline([('scaler', StandardScaler()), ('lrg', XGBClassifier(learning_rate=0.02, n_estimators=1000, objective='binary:logistic',\n                    nthread=-1, colsample_bytree=0.6,\n                    gamma=1, max_depth=5,min_child_weight=10,subsample=0.8))])\nmodel1 = pipe_1.fit(X_train, y_train)\ny_pred1=model1.predict_proba(X_test)[:,1]\nroc_auc_score(y_test, y_pred1)\n")


# In[25]:


get_ipython().run_cell_magic('time', '', "pipe_2=Pipeline([('scaler', StandardScaler()), ('lrg', LogisticRegression(C=0.1,max_iter=200,solver='liblinear'))])\nmodel2 = pipe_2.fit(X_train, y_train)\ny_pred2=model2.predict_proba(X_test)[:,1]\nroc_auc_score(y_test, y_pred2)\n")


# In[26]:


get_ipython().run_cell_magic('time', '', "pipe_3=Pipeline([('scaler', StandardScaler()), ('lrg', CatBoostClassifier())])\nmodel3 = pipe_3.fit(X_train, y_train)\ny_pred3=model3.predict_proba(X_test)[:,1]\nroc_auc_score(y_test, y_pred3)\n")


# # Final Ensemble and Submission
# The test dataset was loaded for prediction with the three models to build a simple ensemble.

# In[27]:


test=pd.read_csv("/kaggle/input/playground-series-s3e4/test.csv")
test.drop(columns="id",inplace=True)
test.head()


# In[28]:


#Feature engineering
test['hour'] = test['Time'] % (24 * 3600) // 3600
test['day'] = (test['Time'] // (24 * 3600)) % 7
#Class predictions
mod1=model1.predict_proba(test)[:,1]
mod2=model2.predict_proba(test)[:,1]
mod3=model3.predict_proba(test)[:,1]


# Finally, the ensemble was built and saved on the submission file.

# In[29]:


example=pd.read_csv("/kaggle/input/playground-series-s3e4/sample_submission.csv")
example.head()


# In[30]:


example["Class"]=0.8*mod1+0.1*mod2+0.1*mod3
print(example.head())
example.to_csv("submission.csv",index=False)


# In this notebook, I combined both data analysis and model prediction, and the main objective is to improve the overall performance.
# 
# If you liked it I would be grateful if you could leave your comments and feedback.
# 
# **Thanks!**
