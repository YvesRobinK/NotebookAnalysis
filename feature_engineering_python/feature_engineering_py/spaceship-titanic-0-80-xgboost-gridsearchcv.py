#!/usr/bin/env python
# coding: utf-8

# <img src="https://edtimes.in/wp-content/uploads/2017/11/eBXIxn-1.jpg">
# <h1><center>🚀Spaceship Titanic🚀</center></h1>
# 
# -----
# **Table of Contents**
# - [1. Introduction, Libraries and Data Import](#intro)
# - [2. Data Preparation](#dataprep)
#    - [2.1 Feature Engineering](#featureeng)
#    - [2.2 Filling Missing Values](#missingvalues)
#    - [2.3 Encoding](#encoding)
# - [3. Modelling](#modelling)
#     - [3.1 Trialing Models](#trials)
#     - [3.2 Hyperparameter Tuning](#tuning)
# - [4. Test Predictions](#testpreds)
# - [Bonus: Model Selection using lazypredict](#lazypredict)
# -----
# 
# <a id='intro'></a>
# # 1. Introduction, Libraries and Data Import
# 
# >  **Goal**: Predict whether or not a passenger travelling on the Spaceship Titanic was transported by an anomaly (binary classification).
# 
# This is a notebook focused on the modelling side of Data Science. As such there will be:
# 
#  - Minimal EDA 
#  - Basic Preprocessing
#  - Comparison of 3 popular classification algorithms
#  - Hyperparameter tuning using GridSearchCV
#  - Bonus: Model Selection using lazypredict
#  
#  >  **Score**: Managed to achieve **0.804** in this competition, placing in the **top 17% of submissions**
#  
# **Be sure to upvote if this helps you!**
#  
# ### Libraries 📚⬇

# In[1]:


# data analysis and wrangling
import numpy as np 
import pandas as pd 

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# modelling
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")


# In[2]:


train_df = pd.read_csv('../input/spaceship-titanic/train.csv')
test_df = pd.read_csv('../input/spaceship-titanic/test.csv')
combine = [train_df, test_df] # to perform same operation on both datasets


# In[3]:


train_df.head()


# In[4]:


# save test_ids for submission
test_ids = test_df['PassengerId']


# <a id='dataprep'></a>
# # 2. Preparing Data
# 
# <a id='featureeng'></a>
# ### 2.1 Feature Engineering

# In[5]:


# Name and PassengerId will not be included in model
for df in combine:
    df.drop(['Name','PassengerId'], axis=1, inplace=True)


# In[6]:


# split Cabin into CabinDeck, CabinNumber and CabinSide and remove original 'Cabin' feature
for df in combine:
    df[['CabinDeck', 'CabinNo', 'CabinSide']] = df['Cabin'].str.split("/", expand=True)
    df.drop('Cabin', axis=1, inplace=True)


# <a id='missingvalues'></a>
# ### 2.1 Filling Missing Values

# In[7]:


# helper function to check for missing values
def check_missed_values(df):
    df_na = ((df.isna().sum())/len(df)) * 100
    df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing %': df_na})
    return missing_data


# In[8]:


check_missed_values(train_df)


# In[9]:


for df in combine:
    
    # fill categorical
    df['HomePlanet'].fillna('Earth', inplace=True)
    df['CryoSleep'].fillna(False, inplace=True)
    df['Destination'].fillna('TRAPPIST-1e', inplace=True)
    df['VIP'].fillna(False, inplace=True)
    df['CabinSide'].fillna('P', inplace=True)
    
    # fill numerical
    for feature in ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "CabinNo"]:
        df[feature].fillna(df[feature].median(), inplace=True)


# In[10]:


# drop rows with missing CabinDeck from train data but fill for test data
train_df = train_df.dropna(subset=['CabinDeck'], axis=0)
test_df['CabinDeck'].fillna('F', inplace = True)


# In[11]:


check_missed_values(test_df)


# <a id='encoding'></a>
# ### 2.3 Encoding

# In[12]:


train_df['CabinNo'] = train_df['CabinNo'].astype('int')
test_df['CabinNo'] = test_df['CabinNo'].astype('int')


# In[13]:


train_df = pd.get_dummies(train_df, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP',
                                     'CabinDeck', 'CabinSide'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP',
                                     'CabinDeck', 'CabinSide'], drop_first=True)


# In[14]:


pd.set_option('display.max_columns', None)
train_df


# <a id='modelling'></a>
# # 3. Modelling

# In[15]:


X_train_full = train_df.drop('Transported', axis=1)
y_train_full = train_df['Transported']
X_test = test_df.copy()


# In[16]:


X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2,
                                                      random_state=6, stratify=y_train_full)


# In[17]:


# helper function to evaluate models
def score_model(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    acc_score = round(accuracy_score(y_valid, y_pred)*100, 2)
    return acc_score


# <a id='trials'></a>
# ## 3.1 Trialling Models
# 
# Let's train a few of the most popular classfication algorithms on the training data and see how they perform on the validation data.
# 
# #### Logistic Regression

# In[18]:


log_reg = LogisticRegression(random_state=6)
acc_log_reg = score_model(log_reg)
acc_log_reg


# #### XGBoost Classifier

# In[19]:


xgb_model = XGBClassifier(random_state=6)
acc_xgb = score_model(xgb_model)
acc_xgb


# #### Random Forest Classifier

# In[20]:


rf_model = RandomForestClassifier(random_state=6)
acc_rf = score_model(rf_model)
acc_rf


# <a id='tuning'></a>
# ## 3.2 Hyperparameter Tuning
# 
# #### RandomForestClassifier performs the best out of the models tested - Let's optimise it.

# In[21]:


# start = time.time()

# parameters = { 
#     'n_estimators': [125,150,175,200],
#     'max_depth' : [18,19,20],
#     'max_features': ['auto','sqrt', 'log2', 'None'],
#     'criterion' :['entropy', 'gini', 'logloss']
# }

# cv_rf_model = GridSearchCV(RandomForestClassifier(random_state=6), parameters,
#                            n_jobs=-1, scoring='precision', cv=5).fit(X_train, y_train)

# running_time = time.time() - start
# print(running_time)

# rf_best_parameters = cv_rf_model.best_params_
# print(rf_best_parameters)


# In[22]:


# from above cell
rf_best_parameters = {'criterion': 'entropy',
                      'max_features': 'auto',
                      'n_estimators': 125,
                      'max_depth': 19}


# In[23]:


opt_rf_model = RandomForestClassifier(**rf_best_parameters, random_state=6)
acc_opt_rf = score_model(opt_rf_model)
acc_opt_rf


# #### Let's also tune an XGBoost Classifier

# In[24]:


# start = time.time()

# parameters = { 
#     'n_estimators': [300,400,500],
#     'max_depth' : [3,4,5],
#     'learning_rate': [0.01, 0.1, 0.5, 1.0],
#     'subsample' :[0.2, 0.5, 1.0]
# }


# cv_xgb_model = GridSearchCV(XGBClassifier(random_state=6), parameters,
#                            n_jobs=-1, scoring='precision', cv=5).fit(X_train, y_train)

# running_time = time.time() - start
# print(running_time)

# xgb_best_parameters = cv_xgb_model.best_params_
# print(xgb_best_parameters)


# In[25]:


# from above cell
xgb_best_parameters = {'n_estimators': 500, 'max_depth': 5 , 'subsample': 1, 'learning_rate': 0.01}


# In[26]:


opt_xgb_model = XGBClassifier(**xgb_best_parameters, random_state=6)
acc_opt_xgb = score_model(opt_xgb_model)
acc_opt_xgb


# #### Model Evaluation

# In[27]:


acc_labels = ['LogReg', 'RandomForest', 'XGBoost', 'RandomForestTuned', 'XGBTuned']
acc = [acc_log_reg, acc_rf, acc_xgb, acc_opt_rf,acc_opt_xgb]
sns.lineplot(x=acc_labels, y=acc)


# <a id='testpreds'></a>
# # 4. Test Predictions
# 
# #### Let's train our 2 best performing models on the full set of training data and make some test predictions

# #### RandomForestClassifier

# In[28]:


# train on full data
test_rf_model = RandomForestClassifier(**rf_best_parameters, random_state=6)
test_rf_model.fit(X_train_full, y_train_full)
rf_test_preds = test_rf_model.predict(X_test)


# #### XGBClassifier

# In[29]:


test_xgb_model = XGBClassifier(**xgb_best_parameters, random_state=6)
test_xgb_model.fit(X_train_full, y_train_full)
xgb_test_preds = map(bool, test_xgb_model.predict(X_test)) # convert 1,0 returned from XGB to Boolean


# #### Test Predictions
# 
# Choose the XGB predictions here.

# In[30]:


submission = pd.DataFrame({'PassengerId': test_ids,
                          'Transported': xgb_test_preds})
submission.to_csv('xgb_submission_500', index=False)


# In[31]:


submission


# <a id='lazypredict'></a>
# # Bonus: using lazypredict for model selection

# Inspiration for <code>lazypredict</code> module taken from https://medium.com/@fareedkhandev/apply-40-machine-learning-models-in-two-lines-of-code-c01dad24ad99

# In[32]:


get_ipython().system('pip install lazypredict')


# In[33]:


import lazypredict

from lazypredict.Supervised import LazyClassifier


# In[34]:


multiple_ML_model = LazyClassifier(verbose=0, ignore_warnings=True, predictions=True)


# In[35]:


models, predictions = multiple_ML_model.fit(X_train, X_valid, y_train, y_valid)


# #### We can see that the RandomForestClassifier and XGBClassifier are both good options to use here according to lazypredict.

# In[36]:


models


# #### Thanks so much for reading through, if this notebook was at all helpful please upvote :)
