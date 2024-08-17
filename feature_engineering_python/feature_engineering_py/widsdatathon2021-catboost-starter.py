#!/usr/bin/env python
# coding: utf-8

# ![](https://drive.google.com/uc?id=1KUPBkdldYARjDAdfF79ezcBpq4Nwg0IL)

# **Credit:** All images are taken from internet . The notebook is adapted from from Dan Ofer's Kernel 
# [here](https://www.kaggle.com/danofer/wids-2020-starter-catboost-0-9045-lb). 
# 
# **Note:** If you like this kernel and/or choose to fork it, please appreciate the hard work by up-voting the kernel with the ^ button above.
# 
# Follow me on Twitter @Urengaraju
# 

# ## Objective
# 
# <div class="alert alert-block alert-info">
# The challenge is to create a model that uses data from the first 24 hours of intensive care to to determine whether a patient has been diagnosed with a particular type of diabetes, Diabetes Mellitus.
# 
# </div>
# 

# ## Data Description
# 
# <div class="alert alert-block alert-info">
# MIT’s GOSSIS community initiative, with privacy certification from the Harvard Privacy Lab, has provided a dataset of more than 130,000 hospital Intensive Care Unit (ICU) visits from patients, spanning a one-year timeframe. This data is part of a growing global effort and consortium spanning Argentina, Australia, New Zealand, Sri Lanka, Brazil, and more than 200 hospitals in the United States.
# </div>
# 
# The data includes:
# 
# 📌**TrainingWiDS2021.csv** - the training data. You should see 130,157 encounters represented here. Please view the Data Dictionary file for more information about the columns.
# 
# 📌**UnlabeledWiDS2021.csv** - the unlabeled data (data without diabetes_mellitus provided). You are being asked to predict the diabetes_mellitus variable for these encounters.
# 
# 📌**SampleSubmissionWiDS2021.csv** - a sample submission file in the correct format.
# 
# 📌**SolutionTemplateWiDS2021.csv** - a list of all the rows (and encounters) that should be in your submissions.
# 
# 📌**DataDictionaryWiDS2021.csv** - supplemental information about the data.

# ### Metric AUC — ROC Curve :
# 
# <div class="alert alert-block alert-info">
# AUC — ROC curve is a performance measurement for classification problem at various thresholds settings. ROC is a probability curve and AUC represents degree or measure of separability. It tells how much model is capable of distinguishing between classes. Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s. Higher the AUC, better the model is at distinguishing between patients with disease and no disease. The ROC curve is plotted with TPR against the FPR where TPR is on y-axis and FPR is on the x-axis.
# </div>
# 
# Learn more about AUC [here](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
# 
# ![](https://drive.google.com/uc?id=1blwyxTGjR13darmbW7-MrUIUnix-lmnE)

# 🎯 **Video Tutorials for the notebook and also walkthrough of Kaggle platform can be found in this discussion thread**
# 
# https://www.kaggle.com/c/widsdatathon2021/discussion/209141
# 
# **Also check out the following discussion threads if you are new to Kaggle or Machine Learning**
# 
# 🎯 **Looking for a Team Megathread**
# https://www.kaggle.com/c/widsdatathon2021/discussion/209054
# 
# 🎯 **New to Kaggle or Machine Learning? Come Say Hi!**
# https://www.kaggle.com/c/widsdatathon2021/discussion/209055
# 
# 🎯 **Questions about competition setup, rules, submissions, etc**
# https://www.kaggle.com/c/widsdatathon2021/discussion/209058
# 

# In[1]:


get_ipython().system('pip install dabl')


# ### Import the Libraries

# In[2]:


import numpy as np 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dabl
from pandas_profiling import ProfileReport
from catboost import Pool, cv, CatBoostClassifier, CatBoostRegressor

from sklearn.metrics import mean_squared_error, classification_report
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

       
import gc
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
pd.set_option('max_rows', 300)
import re


pd.set_option('display.max_columns', 300)
np.random.seed(566)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:20,.2f}'.format)
pd.set_option('display.max_colwidth', -1)


# ### Load the Data

# In[3]:


TARGET_COL = "diabetes_mellitus"
df = pd.read_csv("/kaggle/input/widsdatathon2021/TrainingWiDS2021.csv")
print(df.shape)
test = pd.read_csv("/kaggle/input/widsdatathon2021/UnlabeledWiDS2021.csv")
print(test.shape)


# ### Exploratory Data Analysis
# 
# Check out the following discussion threads to understand the data 
# 
# 🎯 **Understanding all Features in the dataset**
# https://www.kaggle.com/c/widsdatathon2021/discussion/210219
# 
# 🎯 **Apache 2 Calculation**
# https://www.kaggle.com/c/widsdatathon2021/discussion/210221
# 
# 🎯 **ANZACS : APACHE III ICU Diagnosis codes Dataset**
# https://www.kaggle.com/c/widsdatathon2021/discussion/210218
# 
# 🎯 **Questions about the dataset and data structure**
# https://www.kaggle.com/c/widsdatathon2021/discussion/209059
# 
# 
# Reference Notebook : 
# 
# https://www.kaggle.com/thedatabeast/wids-2021-tutorial
# 
# https://www.kaggle.com/iamleonie/wids-datathon-2021-diabetes-detection
# 
# https://www.kaggle.com/yubiabia98/visualization-exploratory-data-analysis-light

# ### dabl, the Data Analysis Baseline Library
# 
# **dabl has several tools that make it easy to clean and inspect your data, and create strong baseline models.**
# 
# Official Documentation : https://amueller.github.io/dabl/dev/quick_start.html
# 
# ### Data Cleaning

# In[4]:


dabl.clean(df)


# ### Exploratory Data analysis
# 

# In[5]:


dabl.plot(df, "diabetes_mellitus")


# ### EDA using Pandas Profiling
# 
# pandas_profiling extends the pandas DataFrame with df.profile_report() for quick data analysis.
# 
# Github Repo : https://github.com/pandas-profiling/pandas-profiling
# 
# 
# ![](https://drive.google.com/uc?id=1QEEcCjfj5cnA_9vRfj2nZLRK0QlQGuBV)
# 

# In[6]:


trainprofile = ProfileReport(df,'EDA')


# In[7]:


trainprofile


# In[8]:


## Print the categorical columns
print([c for c in df.columns if (1<df[c].nunique()) & (df[c].dtype != np.number)& (df[c].dtype != int) ])


# In[9]:


categorical_cols =  ['hospital_id',
 'ethnicity', 'gender', 'hospital_admit_source', 'icu_admit_source', 'icu_stay_type', 'icu_type']



# In[10]:


## Handle na values
df[categorical_cols] = df[categorical_cols].fillna("")
test[categorical_cols] = test[categorical_cols].fillna("")

df[categorical_cols].isna().sum()


# In[11]:


## Train Test split and remove Target values
X_train = df.drop([TARGET_COL],axis=1)
y_train = df[TARGET_COL]


# ### Catboost
# 
# <div class="alert alert-block alert-info">
# CatBoost is an algorithm for gradient boosting on decision trees. It is developed by Yandex researchers and engineers, and is used for search, recommendation systems, personal assistant, self-driving cars, weather prediction and many other tasks at Yandex and in other companies, including CERN, Cloudflare, Careem taxi. It is an open-source library
# </div>
# 
# Official Documentation and Video Tutorials https://catboost.ai/
# 
# ![](https://drive.google.com/uc?id=172aE-E6J3-QAxZ5RI9kN0XOxX7Shq-MB)

# In[12]:


## catBoost Pool object
train_pool = Pool(data=X_train,label = y_train,cat_features=categorical_cols)


# In[13]:


model_basic = CatBoostClassifier(verbose=False,iterations=50)#,learning_rate=0.1, task_type="GPU",)
model_basic.fit(train_pool, plot=True,silent=True)
print(model_basic.get_best_score())


# ### HyperParameter Tuning
# 
# Code Commented because it takes more than 1 hour to run

# In[14]:


### hyperparameter tuning example grid for catboost : 
#grid = {'learning_rate': [0.04, 0.1],
#        'depth': [7, 11],
#         'l2_leaf_reg': [1, 3,9],
#        "iterations": [500],
#       "custom_metric":['Logloss', 'AUC']}

#model = CatBoostClassifier()

## can also do randomized search - more efficient typically, especially for large search space - `randomized_search`
#grid_search_result = model.grid_search(grid, 
#                                      train_pool,
#                                      plot=True,
#                                      refit = True, #  refit best model on all data
#                                      partition_random_seed=42)

#print(model.get_best_score())


# ### Feature Engineering
# 
# 🎯**For Feature engineering approaches , you can refer to WiDS Datathon 2020 Solution Thread**
# 
# https://www.kaggle.com/c/widsdatathon2021/discussion/209053
# 
# **Check out the following threads for experimentation**
# 
# 🎯**Awesome Gradient Boosting Research Papers.**
# https://www.kaggle.com/discussion/207264
# 
# 🎯**Research Papers related to Diabetes Prediction**
# https://www.kaggle.com/c/widsdatathon2021/discussion/209064
# 
# 
# **3 rd Place Solution :** https://www.kaggle.com/jayjay75/3rd-place-nn-wids2020
# 

# ### Submission File

# In[15]:


test[TARGET_COL] = model_basic.predict(test,prediction_type='Probability')[:,1]


# In[16]:


test[["encounter_id","diabetes_mellitus"]].to_csv("submission.csv",index=False)

