#!/usr/bin/env python
# coding: utf-8

# # Work in progress, Upvote and Spread Love..!!
# # Introduction
# Kaggle competitions are incredibly fun and rewarding, but they can also be intimidating for people who are relatively new in their data science journey. In the past, Kaggle have launched many Playground competitions that are more approachable than Featured competition, and thus more beginner-friendly.
# 
# The goal of these competitions is to provide a fun, but less challenging, tabular dataset. These competitions will be great for people looking for something in between the Titanic Getting Started competition and a Featured competition.
# 
# The dataset is used for this competition is synthetic, but based on a real dataset and generated using a CTGAN. The original dataset deals with predicting whether a claim will be made on an insurance policy. Although the features are anonymized, they have properties relating to real-world features.
# 
# This competition will asked to predict whether a customer made a claim upon an insurance policy. The ground truth claim is binary valued, but a prediction may be any number from 0.0 to 1.0, representing the probability of a claim. The features in this dataset have been anonymized and may contain missing values.
# 
# Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

# # Dataset Preparation Details: 
# Preparing packages and data that will be used in the analysis process. Packages that will be loaded are mainly for data manipulation, data visualization and modeling. There are 2 datasets that are used in the analysis, they are train and test dataset. The main use of train dataset is to train models and use it to predict test dataset. While sample submission file is used to informed participants on the expected submission for the competition. (to see the details, please expand)

# # Importing Librabies and Loading datasets

# In[1]:


import os
import joblib
import numpy as np
import pandas as pd
import warnings

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

# setting up options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('float_format', '{:f}'.format)
warnings.filterwarnings('ignore')

# import datasets
train_df = pd.read_csv('../input/tabular-playground-series-nov-2021/train.csv')
test_df = pd.read_csv('../input/tabular-playground-series-nov-2021/test.csv')
submission = pd.read_csv('../input/tabular-playground-series-nov-2021/sample_submission.csv')


# # Dataset Overview
# The intend of the overview is to get a feel of the data and its structure in train, test and submission file. An overview on train and test datasets will include a quick analysis on missing values and basic statistics, while sample submission will be loaded to see the expected submission.

# # Train dataset
# As stated before, train dataset is mainly used to train predictive model as there is an available target variable in this set. This dataset is also used to explore more on the data itself including find a relation between each predictors and the target variable.
# 
# # Observations:
# 
# * Column named target is the target variable which is only available in the train dataset.
# * There are 102 columns: 100 features, 1 target variable claim and 1 column of id.
# * train dataset contain 600,000 observation with 0 missing values which need to be treated carefully.

# In[2]:


print("Shape: ",train_df.shape)
print("NULL values: ",sum(train_df.isna().sum()))
train_df.head()


# # Basic statistics for train
# Below is the basic statistics for each variables which contain information on count, mean, standard deviation, minimum, 1st quartile, median, 3rd quartile and maximum for train dataset.

# In[3]:


train_df.describe()


# # Test dataset
# Test dataset is used to make a prediction based on the model that has previously trained. Exploration in this dataset is also needed to see how the data is structured and especially on it’s similiarity with the train dataset.
# 
# # Observations:
# 
# There are 101 columns: 100 features and 1 column of id.
# train dataset contain 540,000 observation with 0 missing values which need to be treated carefully.

# In[4]:


print("Shape: ",test_df.shape)
print("NULL values: ",sum(test_df.isna().sum()))
test_df.head()


# # Basic statistics for test
# Below is the basic statistics for each variables which contain information on count, mean, standard deviation, minimum, 1st quartile, median, 3rd quartile and maximum for test dataset.

# In[5]:


test_df.describe()


# In[6]:


submission.head()


# # Features
# Number of features available to be used to create a prediction model are 100.
# #  Missing values
# Counting number of missing value and it's relative with their respective observations between train & test dataset.
# #  Preparation
# Prepare train and test dataset for data analysis and visualization.

# In[7]:


missing_train_df = pd.DataFrame(train_df.isna().sum())
missing_train_df = missing_train_df.drop(['id', 'target']).reset_index()
missing_train_df.columns = ['feature', 'count']

missing_train_percent_df = missing_train_df.copy()
missing_train_percent_df['count'] = missing_train_df['count']/train_df.shape[0]

missing_test_df = pd.DataFrame(test_df.isna().sum())
missing_test_df = missing_test_df.drop(['id']).reset_index()
missing_test_df.columns = ['feature', 'count']

missing_test_percent_df = missing_test_df.copy()
missing_test_percent_df['count'] = missing_test_df['count']/test_df.shape[0]

features = [feature for feature in train_df.columns if feature not in ['id', 'target']]
missing_train_row = train_df[features].isna().sum(axis=1)
missing_train_row = pd.DataFrame(missing_train_row.value_counts()/train_df.shape[0]).reset_index()
missing_train_row.columns = ['no', 'count']

missing_test_row = test_df[features].isna().sum(axis=1)
missing_test_row = pd.DataFrame(missing_test_row.value_counts()/test_df.shape[0]).reset_index()
missing_test_row.columns = ['no', 'count']


# # Distribution
# Showing distribution on each feature that are available in train and test dataset. As there are 118 features, it will be broken down into 25 features for each sections. Yellow represents train dataset while pink will represent test dataset
# 
# # Observations:
# 
# All features distribution on train and test dataset are almost similar.
# # **Features f1 - f25**

# In[8]:


background_color = "#f6f5f5"
plt.rcParams['figure.dpi'] = 600
fig = plt.figure(figsize=(10, 10), facecolor='#f6f5f5')
gs = fig.add_gridspec(5, 5)
gs.update(wspace=0.3, hspace=0.3)

run_no = 0
for row in range(0, 5):
    for col in range(0, 5):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        for s in ["top","right"]:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1  

features = list(train_df.columns[0:25])

background_color = "#f6f5f5"
run_no = 0
for col in features:
    sns.kdeplot(ax=locals()["ax"+str(run_no)], x=train_df[col], zorder=2, alpha=1, linewidth=1, color='#ffd514')
    locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
    locals()["ax"+str(run_no)].grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
    locals()["ax"+str(run_no)].set_ylabel('')
    locals()["ax"+str(run_no)].set_xlabel(col, fontsize=4, fontweight='bold')
    locals()["ax"+str(run_no)].tick_params(labelsize=4, width=0.5)
    locals()["ax"+str(run_no)].xaxis.offsetText.set_fontsize(4)
    locals()["ax"+str(run_no)].yaxis.offsetText.set_fontsize(4)
    run_no += 1

run_no = 0
for col in features:
    sns.kdeplot(ax=locals()["ax"+str(run_no)], x=test_df[col], zorder=2, alpha=1, linewidth=1, color='#ff355d')
    locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
    locals()["ax"+str(run_no)].grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
    locals()["ax"+str(run_no)].set_ylabel('')
    locals()["ax"+str(run_no)].set_xlabel(col, fontsize=4, fontweight='bold')
    locals()["ax"+str(run_no)].tick_params(labelsize=4, width=0.5)
    locals()["ax"+str(run_no)].xaxis.offsetText.set_fontsize(4)
    locals()["ax"+str(run_no)].yaxis.offsetText.set_fontsize(4)
    run_no += 1

plt.show()


# # **Features F26-F51**

# In[9]:


background_color = "#f6f5f5"
plt.rcParams['figure.dpi'] = 600
fig = plt.figure(figsize=(10, 10), facecolor='#f6f5f5')
gs = fig.add_gridspec(5, 5)
gs.update(wspace=0.3, hspace=0.3)

run_no = 0
for row in range(0, 5):
    for col in range(0, 5):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        for s in ["top","right"]:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1  

features = list(train_df.columns[26:51])

background_color = "#f6f5f5"
run_no = 0
for col in features:
    sns.kdeplot(ax=locals()["ax"+str(run_no)], x=train_df[col], zorder=2, alpha=1, linewidth=1, color='#ffd514')
    locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
    locals()["ax"+str(run_no)].grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
    locals()["ax"+str(run_no)].set_ylabel('')
    locals()["ax"+str(run_no)].set_xlabel(col, fontsize=4, fontweight='bold')
    locals()["ax"+str(run_no)].tick_params(labelsize=4, width=0.5)
    locals()["ax"+str(run_no)].xaxis.offsetText.set_fontsize(4)
    locals()["ax"+str(run_no)].yaxis.offsetText.set_fontsize(4)
    run_no += 1

run_no = 0
for col in features:
    sns.kdeplot(ax=locals()["ax"+str(run_no)], x=test_df[col], zorder=2, alpha=1, linewidth=1, color='#ff355d')
    locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
    locals()["ax"+str(run_no)].grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
    locals()["ax"+str(run_no)].set_ylabel('')
    locals()["ax"+str(run_no)].set_xlabel(col, fontsize=4, fontweight='bold')
    locals()["ax"+str(run_no)].tick_params(labelsize=4, width=0.5)
    locals()["ax"+str(run_no)].xaxis.offsetText.set_fontsize(4)
    locals()["ax"+str(run_no)].yaxis.offsetText.set_fontsize(4)
    run_no += 1

plt.show()


# # **Features F52-F76**

# In[10]:


background_color = "#f6f5f5"
plt.rcParams['figure.dpi'] = 600
fig = plt.figure(figsize=(10, 10), facecolor='#f6f5f5')
gs = fig.add_gridspec(5, 5)
gs.update(wspace=0.3, hspace=0.3)

run_no = 0
for row in range(0, 5):
    for col in range(0, 5):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        for s in ["top","right"]:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1  

features = list(train_df.columns[52:76])

background_color = "#f6f5f5"
run_no = 0
for col in features:
    sns.kdeplot(ax=locals()["ax"+str(run_no)], x=train_df[col], zorder=2, alpha=1, linewidth=1, color='#ffd514')
    locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
    locals()["ax"+str(run_no)].grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
    locals()["ax"+str(run_no)].set_ylabel('')
    locals()["ax"+str(run_no)].set_xlabel(col, fontsize=4, fontweight='bold')
    locals()["ax"+str(run_no)].tick_params(labelsize=4, width=0.5)
    locals()["ax"+str(run_no)].xaxis.offsetText.set_fontsize(4)
    locals()["ax"+str(run_no)].yaxis.offsetText.set_fontsize(4)
    run_no += 1

run_no = 0
for col in features:
    sns.kdeplot(ax=locals()["ax"+str(run_no)], x=test_df[col], zorder=2, alpha=1, linewidth=1, color='#ff355d')
    locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
    locals()["ax"+str(run_no)].grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
    locals()["ax"+str(run_no)].set_ylabel('')
    locals()["ax"+str(run_no)].set_xlabel(col, fontsize=4, fontweight='bold')
    locals()["ax"+str(run_no)].tick_params(labelsize=4, width=0.5)
    locals()["ax"+str(run_no)].xaxis.offsetText.set_fontsize(4)
    locals()["ax"+str(run_no)].yaxis.offsetText.set_fontsize(4)
    run_no += 1

plt.show()


# # **Features F77-F99**

# In[11]:


background_color = "#f6f5f5"
plt.rcParams['figure.dpi'] = 600
fig = plt.figure(figsize=(10, 10), facecolor='#f6f5f5')
gs = fig.add_gridspec(5, 5)
gs.update(wspace=0.3, hspace=0.3)

run_no = 0
for row in range(0, 5):
    for col in range(0, 5):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        for s in ["top","right"]:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1  

features = list(train_df.columns[77:99])

background_color = "#f6f5f5"
run_no = 0
for col in features:
    sns.kdeplot(ax=locals()["ax"+str(run_no)], x=train_df[col], zorder=2, alpha=1, linewidth=1, color='#ffd514')
    locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
    locals()["ax"+str(run_no)].grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
    locals()["ax"+str(run_no)].set_ylabel('')
    locals()["ax"+str(run_no)].set_xlabel(col, fontsize=4, fontweight='bold')
    locals()["ax"+str(run_no)].tick_params(labelsize=4, width=0.5)
    locals()["ax"+str(run_no)].xaxis.offsetText.set_fontsize(4)
    locals()["ax"+str(run_no)].yaxis.offsetText.set_fontsize(4)
    run_no += 1

run_no = 0
for col in features:
    sns.kdeplot(ax=locals()["ax"+str(run_no)], x=test_df[col], zorder=2, alpha=1, linewidth=1, color='#ff355d')
    locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
    locals()["ax"+str(run_no)].grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
    locals()["ax"+str(run_no)].set_ylabel('')
    locals()["ax"+str(run_no)].set_xlabel(col, fontsize=4, fontweight='bold')
    locals()["ax"+str(run_no)].tick_params(labelsize=4, width=0.5)
    locals()["ax"+str(run_no)].xaxis.offsetText.set_fontsize(4)
    locals()["ax"+str(run_no)].yaxis.offsetText.set_fontsize(4)
    run_no += 1

plt.show()


# # Target
# 
# # Distribution
# Target variable has a value of 0 to 1 which indicate people that not claim and claim the insurance. Let's see how the distribution of the claim variable.
# 
# # Observations:
# 
# The number of people that not target and target (0 and 1) are almost the same of 296,394 and 303,606 respectively.
# In term of percentage both of people that claim and not claim are around 50%.

# In[12]:


claim_df = pd.DataFrame(train_df['target'].value_counts()).reset_index()
claim_df.columns = ['target', 'count']

claim_percent_df = pd.DataFrame(train_df['target'].value_counts()/train_df.shape[0]).reset_index()
claim_percent_df.columns = ['target', 'count']

plt.rcParams['figure.dpi'] = 600
fig = plt.figure(figsize=(5, 1), facecolor='#f6f5f5')
gs = fig.add_gridspec(1, 2)
gs.update(wspace=0.3, hspace=0.05)

background_color = "#f6f5f5"
sns.set_palette(['#ffd514']*120)

ax0 = fig.add_subplot(gs[0, 0])
for s in ["right", "top"]:
    ax0.spines[s].set_visible(False)
ax0.set_facecolor(background_color)
ax0_sns = sns.barplot(ax=ax0, y=claim_df['target'], x=claim_df['count'], 
                      zorder=2, linewidth=0, orient='h', saturation=1, alpha=1)
ax0_sns.set_xlabel("count",fontsize=3, weight='bold')
ax0_sns.set_ylabel("",fontsize=3, weight='bold')
ax0_sns.tick_params(labelsize=3, width=0.5, length=1.5)
ax0_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
ax0_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
ax0.text(0, -0.8, 'Claim', fontsize=4, ha='left', va='top', weight='bold')
ax0.text(0, -0.65, 'Both of 0 and 1 has almost the same numbers', fontsize=2.5, ha='left', va='top')
ax0.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
# data label
for p in ax0.patches:
    value = f'{p.get_width():,.0f}'
    x = p.get_x() + p.get_width() + 10000
    y = p.get_y() + p.get_height() / 2 
    ax0.text(x, y, value, ha='left', va='center', fontsize=2, 
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.2))
    
ax1 = fig.add_subplot(gs[0, 1])
for s in ["right", "top"]:
    ax1.spines[s].set_visible(False)
ax1.set_facecolor(background_color)
ax1_sns = sns.barplot(ax=ax1, y=claim_percent_df['target'], x=claim_percent_df['count'], 
                      zorder=2, linewidth=0, orient='h', saturation=1, alpha=1)
ax1_sns.set_xlabel("percentage",fontsize=3, weight='bold')
ax1_sns.set_ylabel("",fontsize=3, weight='bold')
ax1_sns.tick_params(labelsize=3, width=0.5, length=1.5)
ax1_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
ax1_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
ax1.text(0, -0.8, 'Target in %', fontsize=4, ha='left', va='top', weight='bold')
ax1.text(0, -0.65, 'Both of 0 and 1 distributrion are alomost the same of 50%', fontsize=2.5, ha='left', va='top')
# data label
for p in ax1.patches:
    value = f'{p.get_width():.2f}'
    x = p.get_x() + p.get_width() + 0.01
    y = p.get_y() + p.get_height() / 2 
    ax1.text(x, y, value, ha='left', va='center', fontsize=2, 
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.2))


# # Model
# Evaluate the performance of base model. Models will be evaluated using five cross validation without any hyperparameters tuning. (to see the packages used, please expand)

# In[13]:


from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from scipy.stats import boxcox
from xgboost import XGBClassifier 
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# import datasets
train_df = pd.read_csv('../input/tabular-playground-series-nov-2021/train.csv')

folds = 5
features = list(train_df.columns[1:101])


# # Base model
# Models that will be evaluated are XGBoost Classifier, LGBM Classifier and Catboost Classifier.
# 
# # Observations:
# 
# All 3 models have quite a same AUC result at around 0.8. The differences are very small among the models.
# * Catboost Classifier has AUC of 0.803.
# * XGBoost Classifier has AUC 0.799.
# * LGBM Classifier has AUC of 0.801

# # XGBoost Classifier

# In[14]:


train_oof = np.zeros((600000,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df[features], train_df['target'])):
    X_train, X_valid = train_df.iloc[train_idx], train_df.iloc[valid_idx]
    y_train = X_train['target']
    y_valid = X_valid['target']
    X_train = X_train.drop('target', axis=1)
    X_valid = X_valid.drop('target', axis=1)

    model = XGBClassifier(random_state=42, verbosity=0, tree_method='gpu_hist')

    model =  model.fit(X_train, y_train, verbose=0)
    temp_oof = model.predict_proba(X_valid)[:, 1]
    train_oof[valid_idx] = temp_oof
    print(f'Fold {fold} AUC: ', roc_auc_score(y_valid, temp_oof))
    
print(f'OOF AUC: ', roc_auc_score(train_df['target'], train_oof))


# # LGBM Classifier

# In[15]:


train_oof = np.zeros((600000,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df[features], train_df['target'])):
    X_train, X_valid = train_df.iloc[train_idx], train_df.iloc[valid_idx]
    y_train = X_train['target']
    y_valid = X_valid['target']
    X_train = X_train.drop('target', axis=1)
    X_valid = X_valid.drop('target', axis=1)

    model = LGBMClassifier(random_state=42)

    model =  model.fit(X_train, y_train, verbose=0)
    temp_oof = model.predict_proba(X_valid)[:, 1]
    train_oof[valid_idx] = temp_oof
    print(f'Fold {fold} AUC: ', roc_auc_score(y_valid, temp_oof))
print(f'OOF AUC: ', roc_auc_score(train_df['target'], train_oof))


# # CatBoost Classifier

# In[16]:


train_oof = np.zeros((600000,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df[features], train_df['target'])):
    X_train, X_valid = train_df.iloc[train_idx], train_df.iloc[valid_idx]
    y_train = X_train['target']
    y_valid = X_valid['target']
    X_train = X_train.drop('target', axis=1)
    X_valid = X_valid.drop('target', axis=1)

    model = CatBoostClassifier(random_state=42)

    model =  model.fit(X_train, y_train, verbose=0)
    temp_oof = model.predict_proba(X_valid)[:, 1]
    train_oof[valid_idx] = temp_oof
    print(f'Fold {fold} AUC: ', roc_auc_score(y_valid, temp_oof))
print(f'OOF AUC: ', roc_auc_score(train_df['target'], train_oof))


# # Base model & feature engineering
# This section will blindly try feature engineering, to see if there are any new features that are useful. This section will use LGBM Classifier as the base model.
# 
# # Observations:
# With the following feature engineering attempts:
# * Log
# * Minimum 
# * Maximum 
# * Sum 
# * Multiplication 
# * Prorate
# * Exponential
# 
# It is observed that AUC in all the attempts are quite similar.

# # Log:

# In[17]:


train_df = pd.read_csv('../input/tabular-playground-series-nov-2021/train.csv')
train_oof = np.zeros((600000,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df[features], train_df['target'])):
    X_train, X_valid = train_df.iloc[train_idx], train_df.iloc[valid_idx]
    y_train = X_train['target']
    y_valid = X_valid['target']
    X_train = X_train.drop('target', axis=1)
    X_valid = X_valid.drop('target', axis=1)
    
    X_train = np.log(X_train)
    X_valid = np.log(X_valid)
    
    model = LGBMClassifier(random_state=42)
    
    model =  model.fit(X_train, y_train, verbose=0)
    temp_oof = model.predict_proba(X_valid)[:, 1]
    train_oof[valid_idx] = temp_oof
    print(f'Fold {fold} AUC: ', roc_auc_score(y_valid, temp_oof))
    
print(f'OOF AUC: ', roc_auc_score(train_df['target'], train_oof))


# # Minimum:

# In[18]:


train_df = pd.read_csv('../input/tabular-playground-series-nov-2021/train.csv')
train_oof = np.zeros((600000,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df[features], train_df['target'])):
    X_train, X_valid = train_df.iloc[train_idx], train_df.iloc[valid_idx]
    y_train = X_train['target']
    y_valid = X_valid['target']
    X_train = X_train.drop('target', axis=1)
    X_valid = X_valid.drop('target', axis=1)
    
    X_train['min'] = X_train.min(axis=1)
    X_valid['min'] = X_valid.min(axis=1)
    
    model = LGBMClassifier(random_state=42)
    
    model =  model.fit(X_train, y_train, verbose=0)
    temp_oof = model.predict_proba(X_valid)[:, 1]
    train_oof[valid_idx] = temp_oof
    print(f'Fold {fold} AUC: ', roc_auc_score(y_valid, temp_oof))
    
print(f'OOF AUC: ', roc_auc_score(train_df['target'], train_oof))


# # Maximum:

# In[19]:


train_df = pd.read_csv('../input/tabular-playground-series-nov-2021/train.csv')
train_oof = np.zeros((600000,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df[features], train_df['target'])):
    X_train, X_valid = train_df.iloc[train_idx], train_df.iloc[valid_idx]
    y_train = X_train['target']
    y_valid = X_valid['target']
    X_train = X_train.drop('target', axis=1)
    X_valid = X_valid.drop('target', axis=1)
    
    X_train['max'] = X_train.max(axis=1)
    X_valid['max'] = X_valid.max(axis=1)
    
    model = LGBMClassifier(random_state=42)
    
    model =  model.fit(X_train, y_train, verbose=0)
    temp_oof = model.predict_proba(X_valid)[:, 1]
    train_oof[valid_idx] = temp_oof
    print(f'Fold {fold} AUC: ', roc_auc_score(y_valid, temp_oof))
    
print(f'OOF AUC: ', roc_auc_score(train_df['target'], train_oof))


# # Sum:

# In[20]:


train_df = pd.read_csv('../input/tabular-playground-series-nov-2021/train.csv')
train_oof = np.zeros((600000,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df[features], train_df['target'])):
    X_train, X_valid = train_df.iloc[train_idx], train_df.iloc[valid_idx]
    y_train = X_train['target']
    y_valid = X_valid['target']
    X_train = X_train.drop('target', axis=1)
    X_valid = X_valid.drop('target', axis=1)
    
    X_train['sum'] = X_train.sum(axis=1)
    X_valid['sum'] = X_valid.sum(axis=1)
    
    model = LGBMClassifier(random_state=42)
    
    model =  model.fit(X_train, y_train, verbose=0)
    temp_oof = model.predict_proba(X_valid)[:, 1]
    train_oof[valid_idx] = temp_oof
    print(f'Fold {fold} AUC: ', roc_auc_score(y_valid, temp_oof))
    
print(f'OOF AUC: ', roc_auc_score(train_df['target'], train_oof))


# # Multiply:

# In[21]:


train_df = pd.read_csv('../input/tabular-playground-series-nov-2021/train.csv')
train_oof = np.zeros((600000,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df[features], train_df['target'])):
    X_train, X_valid = train_df.iloc[train_idx], train_df.iloc[valid_idx]
    y_train = X_train['target']
    y_valid = X_valid['target']
    X_train = X_train.drop('target', axis=1)
    X_valid = X_valid.drop('target', axis=1)
    
    X_train['multiply'] = 1
    X_valid['multiply'] = 1
    for feature in features:
        X_train['multiply'] = X_train[feature] * X_train['multiply']
        X_valid['multiply'] = X_valid[feature] * X_valid['multiply']
    
    model = LGBMClassifier(random_state=42)
    
    model =  model.fit(X_train, y_train, verbose=0)
    temp_oof = model.predict_proba(X_valid)[:, 1]
    train_oof[valid_idx] = temp_oof
    print(f'Fold {fold} AUC: ', roc_auc_score(y_valid, temp_oof))
    
print(f'OOF AUC: ', roc_auc_score(train_df['target'], train_oof))


# # Prorate:

# In[22]:


train_df = pd.read_csv('../input/tabular-playground-series-nov-2021/train.csv')
train_oof = np.zeros((600000,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df[features], train_df['target'])):
    X_train, X_valid = train_df.iloc[train_idx], train_df.iloc[valid_idx]
    y_train = X_train['target']
    y_valid = X_valid['target']
    X_train = X_train.drop('target', axis=1)
    X_valid = X_valid.drop('target', axis=1)
    
    X_train['sum'] = X_train[features].sum(axis=1)
    X_valid['sum'] = X_valid[features].sum(axis=1)
    for feature in features:
        X_train[feature+'_prorate'] = X_train[feature] / X_train['sum']
        X_valid[feature+'_prorate'] = X_valid[feature] / X_valid['sum']
    X_train = X_train.drop('sum', axis=1)
    X_valid = X_valid.drop('sum', axis=1)
    model = LGBMClassifier(random_state=42)
    
    model =  model.fit(X_train, y_train, verbose=0)
    temp_oof = model.predict_proba(X_valid)[:, 1]
    train_oof[valid_idx] = temp_oof
    print(f'Fold {fold} AUC: ', roc_auc_score(y_valid, temp_oof))
    
print(f'OOF AUC: ', roc_auc_score(train_df['target'], train_oof))


# # Exponential:

# In[23]:


train_df = pd.read_csv('../input/tabular-playground-series-nov-2021/train.csv')
train_oof = np.zeros((600000,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df[features], train_df['target'])):
    X_train, X_valid = train_df.iloc[train_idx], train_df.iloc[valid_idx]
    y_train = X_train['target']
    y_valid = X_valid['target']
    X_train = X_train.drop('target', axis=1)
    X_valid = X_valid.drop('target', axis=1)
    
    X_train = np.exp(X_train)
    X_valid = np.exp(X_valid)
    model = LGBMClassifier(random_state=42)
    
    model =  model.fit(X_train, y_train, verbose=0)
    temp_oof = model.predict_proba(X_valid)[:, 1]
    train_oof[valid_idx] = temp_oof
    print(f'Fold {fold} AUC: ', roc_auc_score(y_valid, temp_oof))
    
print(f'OOF AUC: ', roc_auc_score(train_df['target'], train_oof))

