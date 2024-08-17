#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# - 1. [Introduction](#section_intro) 
#     - 1.1. [Overview](#)
#     - 1.2. [Pre-request](#)
# - 2. [Understanding Data](#)
#     - 2.1. [Simple Statistics with training data](#)
#     - 2.2. [Simple Statistics with testing data](#)
# - 3. [Data Visualization](#) 
# - 4. [Data Preprocessing](#) 
#     - 4.1. [Handling Missing Values](#section_missingvalues) 
#     - 4.2. [Handling Categorical data](#section_categoricalData) 
#     - 4.3. [Handle imbalance dataset](#section_imbalance) 
#     - 4.4. [Feature Reduction (next submission)](#) 
#     - 4.5. [Scaling](#section_scaling)
# - 5. [Feature Engineering](#section_feature)
# - 6. [Model](#section_model) 
#     - 6.1. [Base model without hyperparameter tuning ](#section_wo_hyper) 
#         - 6.1.1. [XGBoost](#section_xgboost) 
#         - 6.1.2. [Logistic Regression](#section_log_reg) 
#     - 6.2. [Model with hyperparameter tuning using optuna](#section_hyper) 
#         - 6.2.1. [XGBoost](#section_xgb_hyper) 
# - 7. [Submision](#section_sub)

# <a id="section_intro"></a>
# # 1. Introduction

# <a id="section_intro"></a>
# ## 1.1. Overview

# Hi all.....I am a begineer in ML and DL needs lots of fun and resource to learn more in this field. I searched for a community which teach me ML with fun and motivation, I found the answer "Kaggle". This is the great community with lots of compatitions, rewards and lots of learning resources which specially made for ML, DL and AI. So, I started learning with Kaggle. I participated in the "30 day ML" and found that one was very helpful for me to keep consistent learning. This is my second compatition before going into real world ones:)
# <br/>
# ### Compatition overview:
# The goal of these competitions is to provide a fun, and approachable for anyone, tabular dataset. These competitions will be great for people looking for something in between the Titanic Getting Started competition and a Featured competition. If you're an established competitions master or grandmaster, these probably won't be much of a challenge for you. We encourage you to avoid saturating the leaderboard.<br/>
# The dataset is used for this competition is synthetic, but based on a real dataset and generated using a CTGAN. The original dataset deals with predicting whether a claim will be made on an insurance policy. Although the features are anonymized, they have properties relating to real-world features.
# 
# Good luck and have fun!
# ### Prediction:
# For this competition, you will predict whether a customer made a claim upon an insurance policy. The ground truth claim is binary valued, but a prediction may be any number from 0.0 to 1.0, representing the probability of a claim. The features in this dataset have been anonymized and may contain missing values.
# 
# Files
# train.csv - the training data with the target claim column<br>
# test.csv - the test set; you will be predicting the claim for each row in this file<br>
# sample_submission.csv - a sample submission file in the correct format

# <a id="section_intro"></a>
# ## 1.2. Pre-request

# In[1]:


# import the libraries
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# hyperparameter tuning
import optuna

import warnings
warnings.simplefilter("ignore")


# <a id="section_intro"></a>
# # 2. Understanding Data 

# The 'index_col' will make the 'id' feature as a row index because they are no longer needed for model training and it simply represents the row number:)

# In[2]:


# read the datas
train = pd.read_csv("../input/tabular-playground-series-sep-2021/train.csv", index_col = 0)
test = pd.read_csv("../input/tabular-playground-series-sep-2021/test.csv", index_col = 0)


# In[3]:


train.head()


# In[4]:


test.head()


# In[5]:


# get the number of rows.
train_row,train_col = train.shape
test_row, test_col = test.shape
print(f"Number of rows in training dataset------------->{train_row}\nNumber of columns in training dataset---------->{train_col}\n")
print(f"Number of rows in testing dataset-------------->{test_row}\nNumber of columns in testing dataset----------->{test_col}")


# In[6]:


print(train.info())
print("="*50)
test.info()


# <a id="section_intro"></a>
# ## 2.1. Simple Statistics with training data

# From the "train.info()" we can see all the data here are float values and no categorical values available in training data.
# 
# Below is the basic statistics for each variables which contain information on count, mean, standard deviation, minimum, 1st quartile, median, 3rd quartile and maximum.

# In[7]:


train.describe()


# In[8]:


train.corr()


# In[9]:


train.corrwith(train['claim'])


# <a id="section_intro"></a>
# ## 2.2. Simple Statistics with testing data

# From the "test.info()" we can see all the data here are float values and no categorical values available in testing data.
# 
# Below is the basic statistics for each variables which contain information on count, mean, standard deviation, minimum, 1st quartile, median, 3rd quartile and maximum.

# In[10]:


test.describe()


# <a id="section_intro"></a>
# # 3. Data Visualization

# In[11]:


plot , ax = plt.subplots(figsize=(10,8))
sns.heatmap(train.corr())


# ### Number of groups
# From the below plot, we can see there equal number of traget values present in the training data.so, our dataset is a balansed dataset.

# In[12]:


plot , ax = plt.subplots(figsize=(10,8))

sns.countplot(train['claim'])


# In[13]:


hist = train.hist(bins = 25, figsize=(70,45))


# <a id="section_intro"></a>
# # 4. Data Preprocessing
# 4.1. Handling Missing Values
# 4.2. Handling Categorical data
# 4.3. Handle imbalance dataset
# 4.4. Feature Reduction (next submission)
# 4.5. Standardization (or) Normalization

# In[14]:


features = train.columns.tolist()[0:-1]
target = ['claim']


# <a id="section_missingvalues"></a>
# ## 4.1. Handling Missing Values

# In[15]:


train_row,train_col = train.shape
test_row, test_col = test.shape
#find the missing values w.r.t. column
train_colum_missing = train.isnull().sum()
train_total_missing = sum(train_colum_missing)
# find the missing values w.r.t. row(number of missing values in the particular row)
train_row_missing = train[features].isnull().sum(axis=1)

# add the missing values to row to the dataframe as a new value
train['no_of_missing_data'] = train_row_missing



#find the missing values w.r.t. column
test_colum_missing = test.isnull().sum()
test_total_missing=sum(test_colum_missing)
# find the missing values w.r.t. row(number of missing values in the particular row)
test_row_missing = test[features].isnull().sum(axis=1)

# add the missing values to row to the dataframe as a new value
test['no_of_missing_data'] = test_row_missing


# In[16]:


print(f"Total number of missing values in training dataset---->{train_total_missing}")
print(f"Total number of missing values in testing dataset----->{test_total_missing}")
# compare this to the whole data
train_no_of_missing_rows = (train['no_of_missing_data'] != 0).sum()
print("\n{0:{fill}{align}80}\n".format("Training Data" , fill = "=", align = "^"))
print(f"Total rows -----------------------> {train_row}\nNumber of rows has missing data---> {train_no_of_missing_rows}\n{'-'*50}\nNumber of rows has full data--------> {train_row-train_no_of_missing_rows}")

test_no_of_missing_rows = (test['no_of_missing_data'] != 0).sum()
print("\n{0:{fill}{align}80}\n".format("Testing Data" , fill = "=", align = "^"))
print(f"Total rows -----------------------> {test_row}\nNumber of rows has missing data---> {test_no_of_missing_rows}\n{'-'*50}\nNumber of rows has full data--------> {test_row-test_no_of_missing_rows}")


# From this we can see , 2/3 of the dataset containing missing values both in training and testing data. we must need to handle this before going to model traning.<br/>
# There are some methods to deal with missing values, we can see some methods in this separate notebook----->https://www.kaggle.com/ninjaac/methods-to-handle-missing-value

# In[17]:


# here i am going to use the media for now.
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
train.iloc[:,0:-1]  = pd.DataFrame(imputer.fit_transform(train.iloc[:,0:-1]), columns = train.columns.tolist()[0:-1], index = train.index)
test = pd.DataFrame(imputer.transform(test),columns = test.columns, index = test.index)


# <a id="section_categoricalData"></a>
# ## 4.2. Handling Categorical data 

# This dataset has no categorical data, so we don't need to care about this for now. if want we can learn about the categorical values here -->https://www.kaggle.com/alexisbcook/categorical-variables

# <a id="section_imbalance"></a>
# ## 4.3. Handle imbalance dataset

# We can see the model having squal number of training and testing dataset, so here we don't need to balance the dataset. all fine....good to go next:)

# In[18]:


plot , ax = plt.subplots(figsize=(6,3))

sns.countplot(train['claim'])


# <a id="section_scaling"></a>
# ## 4.5. Scaling

# In[ ]:


# scaler = StandardScaler()
# train.iloc[:,0:-1] = pd.DataFrame(imputer.fit_transform(train.iloc[:,0:-1]), columns = train.columns.tolist()[0:-1], index = train.index)
# test = pd.DataFrame(imputer.transform(test), columns = test.columns, index = test.index)


# <a id="section_feature"></a>
# # 5. Feature Engineering

# Adding some feature or reducing some feature by selcting the most important feature(Feature Selection) is called feature engineering.<br/>
# Here we are going to add some features like min, max, mean, median, mode, standard diviation number of missing values in the row. i got this idea from my random learning from some code books of this compatition.

# In[19]:


def feature_adding(data):
    data["min"] = data.min(axis = 1) # 1 or ‘columns’ : get mode of each row.
    data['max'] = data.max(axis = 1)
    data['std'] = data.std(axis = 1)
    data['median'] = data.median(axis = 1)
    data['mean'] = data.mean(axis = 1)
    #data['mode'] = data.mode(axis = 1)
    data['mad'] = data.mad(axis = 1) # mean absolute deviation
    data['skew'] = data.skew(axis=1)
     # no of missing data already added in the handling missing value section.
    # scale the data
    features = data.columns.tolist()
    scaler = StandardScaler()
    data[features] = pd.DataFrame(imputer.fit_transform(data[features]), columns = data.columns.tolist(), index = data.index)
    
    
    return data
    


# In[20]:


train_y = train['claim'].copy()
train_X = feature_adding(train.drop('claim', axis = 1))
test = feature_adding(test)


# In[21]:


del(train)


# <a id="section_wo_hyper"></a>
# ## 6.1. Base model without hyperparameter tuning

# <a id="section_xgboost"></a>
# ### 6.1.1. XGBoost

# In[ ]:


xgb_params = {
    'max_depth': 2,
    'booster': 'gbtree', 
    'n_estimators': 10000,
    'random_state':42,
    'tree_method':'gpu_hist',
    'gpu_id':0,
    'predictor':"gpu_predictor",
}


# In[ ]:


#Setting the kfold parameters
n_fold = 15
kf = KFold(n_splits = n_fold, shuffle = True, random_state = 42)
pred = 0
results = np.zeros((train_X.shape[0],))
mean_acc = 0

xgb_model = XGBClassifier(**xgb_params)


for fold, (train_id, valid_id) in enumerate(kf.split(train_X)):
    X_train, X_val = train_X.loc[train_id],train_X.loc[valid_id]
    y_train, y_val = train_y.iloc[train_id], train_y.iloc[valid_id]
    
    
    xgb_model.fit(X_train, y_train,
             verbose = False,
             eval_set = [(X_train, y_train), (X_val, y_val)],
             eval_metric = "auc",
             early_stopping_rounds = 100)
    
    
    #Out of Fold predictions
    results=  xgb_model.predict_proba(X_val) 
    
    pred += xgb_model.predict_proba(test)[:,1] / n_fold
    
    fold_acc = roc_auc_score(y_val ,results[:,1])
    
    print(f"Fold {fold} | Fold accuracy: {fold_acc}")
    
    mean_acc += fold_acc / n_fold
    
print(f"\nOverall Accuracy: {mean_acc}")


# #### after feature engineeering 
# score with default parameter and Standarscalar and median imputer =  0.8152311464421033 <br/>
# score with default parameter and Normalization and constant zero imputer =  0.8153624191860824 <br/>
# score with default parameter and Standarscalar and mean imputer =  0.8154707510791169 <br/>
# mean is higher so talking that one.

# <a id="section_log_reg"></a>
# ## 6.1.2. Logistic Regression

# In[ ]:


#Setting the kfold parameters
n_fold = 5
kf = KFold(n_splits = n_fold, shuffle = True, random_state = 42)
pred = 0
results = np.zeros((train_X.shape[0],))
mean_acc = 0

log_reg = LogisticRegression(n_jobs = -1,C = 0.01, penalty = 'l2', random_state = 42)


for fold, (train_id, valid_id) in enumerate(kf.split(train_X)):
    X_train, X_val = train_X.loc[train_id], train_X.loc[valid_id]
    y_train, y_val = train_y.iloc[train_id], train_y.iloc[valid_id]
    
    
    log_reg.fit(X_train, y_train,
             )
    
    
    #Out of Fold predictions
    results=  log_reg.predict_proba(X_val) 
    
    pred += log_reg.predict_proba(test)[:,1] / n_fold
    
    fold_acc = roc_auc_score(y_val ,results[:,1])
    
    print(f"Fold {fold} | Fold accuracy: {fold_acc}")
    
    mean_acc += fold_acc / n_fold
    
print(f"\nOverall Accuracy: {mean_acc}")


# # just trying path
# score with default parameter and Standarscalar and median imputer = 0.5075950898769981 (so bad)<br/>
# score with default parameter and Standarscalar and mean imputer = 0.5069218256005761 <br/>
# score with default parameter and Standarscalar and constant zero imputer = 0.507425029718713 (so bad)<br/> 
# 
# #### Not a big difference
# score with default parameter and Normalization and median imputer = 0.5056612627208464 (so bad)<br/>
# score with default parameter and Normalization and mean imputer = 0.5056612627208464 <br/>
# score with default parameter and Normalization and constant zero imputer = 0.5055436207845755(so bad)<br/> 
# 
# #### after feature engineeering ( i can't find any improvement after feature engneering)
# score with default parameter and Standarscalar and median imputer = 0.5090028286920467 <br/>
# score with default parameter and Normalization and constant zero imputer = 0.5125507184919263

# <a id="section_hyper"></a>
# # 6.2. Model with hyperparameter tuning using optuna

# In[22]:


X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, random_state=0, test_size = 0.1)



# <a id="section_xgb_hyper"></a>
# ## 6.2.1. XGBoost

# In[23]:


def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 5000, 50000, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 100.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 100.0),
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 100.0),
        
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.5, log=True),
    }
    
    #XGBoost model fitting.
    xgb = XGBClassifier(**param, 
                         random_state=24,
                         tree_method='gpu_hist' , 
                         gpu_id = 0, 
                         predictor="gpu_predictor",)
    
    
    xgb.fit(X_train, y_train,
             verbose = False,
             eval_set = [(X_train, y_train), (X_val, y_val)],
             eval_metric = "auc",
             early_stopping_rounds = 100)
    
    # precdition .
    xgb_pred = xgb.predict_proba(X_val)
    
    acc = roc_auc_score(y_val, xgb_pred[:,1])
    return acc


# In[24]:


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40) 

print("lenth of the finished trials: ", len(study.trials))


print("Best value for the rmse:", study.best_trial.value)
print("Best parameters:", study.best_params)


# In[25]:


best_param = study.best_params
best_param

"""{'n_estimators': 29000,
 'max_depth': 3,
 'min_child_weight': 66,
 'subsample': 0.7842215931394566,
 'colsample_bytree': 0.5179071234426729,
 'gamma': 0.4700525265587939,
 'alpha': 0.000257843578177255,
 'lambda': 2.4671217994166645,
 'learning_rate': 0.012418427180920295}"""


# In[31]:


#Setting the kfold parameters
n_fold = 5
kf = KFold(n_splits = n_fold, shuffle = True, random_state = 42)
pred = 0
results = np.zeros((train_X.shape[0],))
mean_acc = 0

xgb_model = XGBClassifier(**best_param,               
      booster= 'gbtree', 
    random_state = 42,
    tree_method = 'gpu_hist',
    gpu_id= 0,
    predictor="gpu_predictor",)


for fold, (train_id, valid_id) in enumerate(kf.split(train_X)):
    X_train, X_val = train_X.loc[train_id],train_X.loc[valid_id]
    y_train, y_val = train_y.iloc[train_id], train_y.iloc[valid_id]
    
    
    xgb_model.fit(X_train, y_train,
             verbose = False,
             eval_set = [(X_train, y_train), (X_val, y_val)],
             eval_metric = "auc",
             early_stopping_rounds = 100)
    
    
    #Out of Fold predictions
    results=  xgb_model.predict_proba(X_val) 
    
    pred += xgb_model.predict_proba(test)[:,1] / n_fold
    
    fold_acc = roc_auc_score(y_val ,results[:,1])
    
    print(f"Fold {fold} | Fold accuracy: {fold_acc}")
    
    mean_acc += fold_acc / n_fold
    
print(f"\nOverall Accuracy: {mean_acc}")


# <a id="section_sub"></a>
# ## 6. Submision

# In[32]:


sub = pd.read_csv("../input/tabular-playground-series-sep-2021/sample_solution.csv")


# In[33]:


sub['claim'] = pred


# In[34]:


sub.to_csv('submission3.csv',index=False)


# In[ ]:




