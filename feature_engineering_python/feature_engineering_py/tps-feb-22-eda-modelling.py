#!/usr/bin/env python
# coding: utf-8

# **Created by Sanskar Hasija**
# 
# **[TPS-FEB-22] 📊EDA + Modelling📈**
# 
# **01 February 2022**
# 

# # <center> [TPS-FEB-22] 📊EDA + Modelling📈 </center>
# ## <center>If you find this notebook useful, support with an upvote👍</center>

# # Table of Contents
# <a id="toc"></a>
# - [1. Introduction](#1)
# - [2. Imports](#2)
# - [3. Data Loading and Preperation](#3)
#     - [3.1 Exploring Train Data](#3.1)
#     - [3.2 Exploring Test Data](#3.2)
#     - [3.3 Submission File](#3.3)
# - [4. EDA](#4)
#     - [4.1 Overview of Data](#4.1)
#     - [4.2 Null Value Distribution](#4.7)
#     - [4.3 Continuos and Categorical Data Distribution](#4.2)
#     - [4.4 Feature Distribution of Continous Features](#4.3)
#     - [4.5 Feature Distribution of Categorical Features](#4.4)
#     - [4.6 Target Distribution ](#4.5)
# - [5. Feature Engineering](#5)   
# - [6. Modelling](#6)
#     - [6.1 LGBM Classifier](#6.1)
#     - [6.2 Catboost Classifier](#6.2)
#     - [6.3 XGBoost Classifier](#6.3)
# - [7. Submission](#7)   

# <a id="1"></a>
# # Introduction

# **The task of this compeition is to classify 10 different bacteria species using data from a genomic analysis technique that has some data compression and data loss. The dataset used for this compeition is derived from this [paper](https://www.frontiersin.org/articles/10.3389/fmicb.2020.00257/full).**
# 
# **Submissions are evaluated based on their categorization accuracy..**

# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>

# <a id="2"></a>
# # Imports

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px



from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.stats import mode


from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


from matplotlib import ticker
import time
import warnings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('float_format', '{:f}'.format)
warnings.filterwarnings('ignore')


RANDOM_STATE = 12 
FOLDS = 5


# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>

# <a id="3"></a>
# # Data Loading and Preperation

# In[2]:


train = pd.read_csv("../input/tabular-playground-series-feb-2022/train.csv")
test = pd.read_csv("../input/tabular-playground-series-feb-2022/test.csv")
submission = pd.read_csv("../input/tabular-playground-series-feb-2022/sample_submission.csv")


# <a id="3.1"></a>
# ## Exploring Train Data

# <div class="alert alert-block alert-info" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
#     📌 &nbsp;<b><u>Observations in Train Data:</u></b><br>
# 
# * <i>There are total of <b><u>288</u></b> columns : <b><u>278</u></b> continous , <b><u>8</u></b> categorical <b><u>1</u></b> row_id and <b><u>1</u></b> target column</i><br>
# * <i> There are total of <b><u>200000</u></b> rows in train dataset.</i><br>
# * <i> <b><u>target</u></b> is the target variable which is only available in the <b><u>train</u></b> dataset..</i><br>
# * <i> Train dataset contain <b><u>57600000</u></b> observation with <b><u>0</u></b>  missing / null values.</i><br>
# * <i> No <b><u>NULL</u></b> Values 🙂 </i><br>
#     
# </div>

# ### Quick view of Train Data

# Below is the first 5 rows of train dataset:

# In[3]:


train.head()


# In[4]:


print(f'\033[92mNumber of rows in train data: {train.shape[0]}')
print(f'\033[94mNumber of columns in train data: {train.shape[1]}')
print(f'\033[91mNumber of values in train data: {train.count().sum()}')
print(f'\033[91mNumber missing values in train data: {sum(train.isna().sum())}')


# ### Basic statistics of training data

# Below is the basic statistics for each variables which contain information on `count`, `mean`, `standard deviation`, `minimum`, `1st quartile`, `median`, `3rd quartile` and `maximum`.

# In[5]:


train.describe()


# <a id="3.2"></a>
# ## Exploring Test Data

# <div class="alert alert-block alert-info" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
#     📌 &nbsp;<b><u>Observations in Test Data:</u></b><br>
# 
# * <i> There are total of <b><u>287</u></b> columns : <b><u>278</u></b> continous , <b><u>8</u></b> categorical <b><u>1</u></b> row_id in <b><u>test</u></b> dataset.</i><br>
# * <i> There are total of <b><u>100000</u></b> rows in test dataset.</i><br>
# * <i> Test dataset contain <b><u>28700000</u></b> observation with <b><u>0</u></b>  missing values.</i><br>
# * <i> No <b><u>NULL</u></b> Values again. 🙂</i><br>
#     
# </div>

# ### Quick view of Test Data

# In[6]:


test.head()


# In[7]:


print(f'\033[92mNumber of rows in test data: {test.shape[0]}')
print(f'\033[94mNumber of columns in test data: {test.shape[1]}')
print(f'\033[91mNumber of values in train data: {test.count().sum()}')
print(f'\033[91mNo of rows with missing values  in test data: {sum(test.isna().sum())}')


# ### Basic statistics of test data

# Below is the basic statistics for each variables which contain information on `count`, `mean`, `standard deviation`, `minimum`, `1st quartile`, `median`, `3rd quartile` and `maximum`.

# In[8]:


test.describe()


# <a id="3.3"></a>
# ## Submission File

# ### Quick view of Submission File

# In[9]:


submission.head()


# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>

# <a id="4"></a>
# # EDA

# <a id="4.1"></a>
# ## Overview of Data

# In[10]:


train.drop(["row_id"] , axis = 1 , inplace = True)
test.drop(["row_id"] , axis = 1 , inplace = True)
TARGET = 'target'
FEATURES = [col for col in train.columns if col not in ['row_id', TARGET]]
RANDOM_STATE = 12 


# In[11]:


train.iloc[:, :-1].describe().T.sort_values(by='std' , ascending = False)\
                     .style.background_gradient(cmap='GnBu')\
                     .bar(subset=["max"], color='#F8766D')\
                     .bar(subset=["mean",], color='#00BFC4')


# <a id="4.7"></a>
# ## Null Value Distribution 

# <div class="alert alert-block alert-info" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
#     📌 &nbsp;<b><u>Observations in Null Value Distribution :</u></b><br>
# 
# * <i> No Null values. </i><br>
# </div>

# <a id="4.2"></a>
# ## Continuos and Categorical Data Distribution

# <div class="alert alert-block alert-info" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
#     📌 &nbsp;<b><u>Observations in Data Distribution :</u></b><br>
#     
# * <i>Out of 286 features <b><u>278</u></b> features are continous </i><br>
# * <i>The reamining <b><u>8</u></b> features are categorical. <b><u>(can be considered as categorical,since they have less than 25 unique values)</u></b></i><br>
#     
# </div>
# 

# In[12]:


df = pd.concat([train[FEATURES], test[FEATURES]], axis=0)

cat_features = [col for col in FEATURES if df[col].nunique() < 25]
cont_features = [col for col in FEATURES if df[col].nunique() >= 25]

del df
print(f'Total number of features: {len(FEATURES)}')
print(f'\033[92mNumber of categorical (<25 Unique Values) features: {len(cat_features)}')
print(f'\033[96mNumber of continuos features: {len(cont_features)}')


plt.pie([len(cat_features), len(cont_features)], 
        labels=['Categorical(<25 Unique Values)', 'Continuos'],
        colors=['#F8766D', '#00BFC4'],
        textprops={'fontsize': 13},
        autopct='%1.1f%%')
plt.show()


# <a id="4.3"></a>
# ## Feature Distribution of Continous Features

# #### <i><u>(NOTE : THE ABOVE DISCUSSED CATEGORICAL FEATURES ARE INCLUDED IN THE FOLLOWING CONTINUOS FEATURE DISTRIBUTION PLOTS)</u></i>
# ### Feature Distribution of first 100 Features

# In[13]:


ncols = 5
nrows = 20
n_features = cont_features[:100]
fig, axes = plt.subplots(nrows, ncols, figsize=(25, 15*4))

for r in range(nrows):
    for c in range(ncols):
        col = n_features[r*ncols+c]
        sns.kdeplot(x=train[col], ax=axes[r, c], color='#F8766D', label='Train data' , fill =True )
        sns.kdeplot(x=test[col], ax=axes[r, c], color='#00BFC4', label='Test data', fill =True)
        axes[r,c].legend()
        axes[r, c].set_ylabel('')
        axes[r, c].set_xlabel(col, fontsize=8)
        axes[r, c].tick_params(labelsize=5, width=0.5)
        axes[r, c].xaxis.offsetText.set_fontsize(6)
        axes[r, c].yaxis.offsetText.set_fontsize(4)
plt.show()


# ### Feature Distribution of 101-200 Features

# In[14]:


ncols = 5
nrows = 20
n_features = cont_features[100:200]
fig, axes = plt.subplots(nrows, ncols, figsize=(25, 60))

for r in range(nrows):
    for c in range(ncols):
        col = n_features[r*ncols+c]
        sns.kdeplot(x=train[col], ax=axes[r, c], color='#F8766D', label='Train data' , fill =True )
        sns.kdeplot(x=test[col], ax=axes[r, c], color='#00BFC4', label='Test data', fill =True)
        axes[r,c].legend()
        axes[r, c].set_ylabel('')
        axes[r, c].set_xlabel(col, fontsize=8)
        axes[r, c].tick_params(labelsize=5, width=0.5)
        axes[r, c].xaxis.offsetText.set_fontsize(6)
        axes[r, c].yaxis.offsetText.set_fontsize(4)
plt.show()


# ### Feature Distribution of 201-275 Features

# In[15]:


ncols = 5
nrows = 15
n_features = cont_features[200:]
fig, axes = plt.subplots(nrows, ncols, figsize=(25, 45))

for r in range(nrows):
    for c in range(ncols):
        col = n_features[r*ncols+c]
        sns.kdeplot(x=train[col], ax=axes[r, c], color='#F8766D', label='Train data' , fill =True )
        sns.kdeplot(x=test[col], ax=axes[r, c], color='#00BFC4', label='Test data', fill =True)
        axes[r,c].legend()
        axes[r, c].set_ylabel('')
        axes[r, c].set_xlabel(col, fontsize=8)
        axes[r, c].tick_params(labelsize=5, width=0.5)
        axes[r, c].xaxis.offsetText.set_fontsize(6)
        axes[r, c].yaxis.offsetText.set_fontsize(4)
plt.show()


# <a id="4.4"></a>
# ## Feature Distribution of Categorical Features

# In[16]:


print(f'\033[92mNo Categorical features.')
print(f'\033[92mAll feature distribution with less than 25 unique values plotted above with continous feature distributions')
print(f'\033[94mContinous Features with their unique value count:')
for cat in cat_features:
    print(str(cat) + " -   " + str(train[cat].nunique()))


# <a id="4.5"></a>
# ## Target Distribution

# <div class="alert alert-block alert-info" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
#     📌 &nbsp;<b><u>Observations in Target Distribution :</u></b><br>
# 
# * <i>There are <b><u>10</u></b> different target values</i><br>
# * <i>All target values are equally distributed approx - <b><u>10%</u></b> of total observations for each target.</i><br>
#     
# </div>

# In[17]:


target_df = pd.DataFrame(train[TARGET].value_counts()).reset_index()
target_df.columns = [TARGET, 'count']
fig = px.bar(data_frame =target_df, 
             x = TARGET,
             y = 'count' , 
             color = "count",
             color_continuous_scale="Emrld") 
fig.update_layout(template = "plotly_white")
for idx,target in enumerate(target_df["target"]):
    print("\033[94mPercentage of " + str(target) + " category  : {:.2f} %".format(target_df["count"][idx] *100 / train.shape[0]))
fig.show()


# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>

# <a id="5"></a>
# #  Feature Engineering

# ###  Basic Feature Engineering

# In[18]:


train["mean"] = train[FEATURES].mean(axis=1)
train["std"] = train[FEATURES].std(axis=1)
train["min"] = train[FEATURES].min(axis=1)
train["max"] = train[FEATURES].max(axis=1)

test["mean"] = test[FEATURES].mean(axis=1)
test["std"] = test[FEATURES].std(axis=1)
test["min"] = test[FEATURES].min(axis=1)
test["max"] = test[FEATURES].max(axis=1)

FEATURES.extend(['mean', 'std', 'min', 'max'])


# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>

# <a id="6"></a>
# #  Modelling

# <div class="alert alert-block alert-info" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
#     📌 &nbsp;<b><u>Observations in Target Modelling :</u></b><br>
#     
# * <i> <u><b>LGBMClassifier</u></b> , <u><b>CatBoostClassifier</u></b> and <u><b>XGBClassifier</u></b> used in modelling on 5-fold validation.</i><br>
# * <i> Further Hyperparameter tuning can imporve the results.</i><br>
#     
# </div>

# In[19]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
train[TARGET] = encoder.fit_transform(train[TARGET])


# <a id="6.1"></a>
# ## LGBM Classifier

# In[20]:


lgb_params = {
    'objective' : 'multiclass',
    'metric' : 'multi_logloss',
    'device' : 'gpu',
}


lgb_predictions = []
lgb_scores = []
lgb_fimp = []

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train[FEATURES], train[TARGET])):
    
    print(10*"=", f"Fold={fold+1}", 10*"=")
    start_time = time.time()
    
    X_train, X_valid = train.iloc[train_idx][FEATURES], train.iloc[valid_idx][FEATURES]
    y_train , y_valid = train[TARGET].iloc[train_idx] , train[TARGET].iloc[valid_idx]
    
    model = LGBMClassifier(**lgb_params)
    model.fit(X_train, y_train,verbose=0)
    
    preds_valid = model.predict(X_valid)
    acc = accuracy_score(y_valid,  preds_valid)
    lgb_scores.append(acc)
    run_time = time.time() - start_time
    
    print(f"Fold={fold+1}, Accuracy: {acc:.2f}, Run Time: {run_time:.2f}s")
    fim = pd.DataFrame(index=FEATURES,
                 data=model.feature_importances_,
                 columns=[f'{fold}_importance'])
    lgb_fimp.append(fim)
    test_preds = model.predict(test[FEATURES])
    lgb_predictions.append(test_preds)
    
print("Mean Accuracy :", np.mean(lgb_scores))


# ### Feature Importance for LGBM Classifier (Top 15 Features)

# In[21]:


lgbm_fis_df = pd.concat(lgb_fimp, axis=1).head(15)
lgbm_fis_df.sort_values('1_importance').plot(kind='barh', figsize=(15, 10),
                                       title='Feature Importance Across Folds')
plt.show()


# <a id="6.2"></a>
# ## Catboost Classifier

# In[22]:


catb_params = {
    "objective": "MultiClass",
    "task_type": "GPU",
}

catb_predictions = []
catb_scores = []
catb_fimp = []

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train[FEATURES], train[TARGET])):
    
    print(10*"=", f"Fold={fold+1}", 10*"=")
    start_time = time.time()
    
    X_train, X_valid = train.iloc[train_idx][FEATURES], train.iloc[valid_idx][FEATURES]
    y_train , y_valid = train[TARGET].iloc[train_idx] , train[TARGET].iloc[valid_idx]
    
    model = CatBoostClassifier(**catb_params)
    model.fit(X_train, y_train,verbose=0)
    
    preds_valid = model.predict(X_valid)
    acc = accuracy_score(y_valid,  preds_valid)
    catb_scores.append(acc)
    run_time = time.time() - start_time
    
    print(f"Fold={fold+1}, Accuracy: {acc:.2f}, Run Time: {run_time:.2f}s")
    fim = pd.DataFrame(index=FEATURES,
                 data=model.feature_importances_,
                 columns=[f'{fold}_importance'])
    catb_fimp.append(fim)
    test_preds = model.predict(test[FEATURES])
    catb_predictions.append(test_preds)
    
print("Mean Accuracy :", np.mean(catb_scores))


# ### Feature Importance for Catboost Classifier (Top 15 Features)

# In[23]:


catb_fis_df = pd.concat(catb_fimp, axis=1).head(15)
catb_fis_df.sort_values('1_importance').plot(kind='barh', figsize=(15, 10),
                                       title='Feature Importance Across Folds')
plt.show()


# <a id="6.3"></a>
# ## XGBoost Classifier

# In[24]:


xgb_params = {
    'objective': 'multi:softmax',
    'eval_metric': 'mlogloss',
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    }


xgb_predictions = []
xgb_scores = []
xgb_fimp = []

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train[FEATURES], train[TARGET])):
    
    print(10*"=", f"Fold={fold+1}", 10*"=")
    start_time = time.time()
    
    X_train, X_valid = train.iloc[train_idx][FEATURES], train.iloc[valid_idx][FEATURES]
    y_train , y_valid = train[TARGET].iloc[train_idx] , train[TARGET].iloc[valid_idx]
    
    model = XGBClassifier(**xgb_params)
    model.fit(X_train, y_train,verbose=0)
    
    preds_valid = model.predict(X_valid)
    acc = accuracy_score(y_valid,  preds_valid)
    xgb_scores.append(acc)
    run_time = time.time() - start_time
    
    print(f"Fold={fold+1}, Accuracy: {acc:.2f}, Run Time: {run_time:.2f}s")
    test_preds = model.predict(test[FEATURES])
    fim = pd.DataFrame(index=FEATURES,
                 data=model.feature_importances_,
                 columns=[f'{fold}_importance'])
    xgb_fimp.append(fim)
    xgb_predictions.append(test_preds)
    
print("Mean Accuracy :", np.mean(xgb_scores))


# ### Feature Importance for XGBoost Classifier (Top 15 Features)

# In[25]:


xgb_fis_df = pd.concat(xgb_fimp, axis=1).head(15)
xgb_fis_df.sort_values('1_importance').plot(kind='barh', figsize=(15, 10),
                                       title='Feature Importance Across Folds')
plt.show()


# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>

# <a id="6"></a>
# #  Submission

# ### LGBM Classifier Submission

# In[26]:


lgb_submission = submission.copy()
lgb_submission["target"] = encoder.inverse_transform(np.squeeze(mode(np.column_stack(lgb_predictions),axis = 1)[0]).astype('int'))
lgb_submission.to_csv("lgb-subs.csv",index=False)
lgb_submission.head()


# ### Catboost Classifier Submission

# In[27]:


catb_submission = submission.copy()
catb_submission["target"] = encoder.inverse_transform(np.squeeze(mode(np.column_stack(catb_predictions),axis = 1)[0]).astype('int'))
catb_submission.to_csv("catb-subs.csv",index=False)
catb_submission.head()


# ### XGBoost Classifier Submission

# In[28]:


xgb_submission = submission.copy()
xgb_submission["target"] = encoder.inverse_transform(np.squeeze(mode(np.column_stack(xgb_predictions),axis = 1)[0]).astype('int'))
xgb_submission.to_csv("xgb-subs.csv",index=False)
xgb_submission.head()


# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>

# <div class="alert alert-block alert-info" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
#     
#     
# ### <center>Thank you for reading🙂</center>
# ### <center>If you have any feedback or find anything wrong, please let me know!</center>
# 
