#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
# <a id="table-of-contents"></a>
# - [1 Introduction](#1)
# - [2 Preparations](#2)
# - [3 Datasets Overview](#3)
#     - [3.1 Train dataset](#3.1)
#     - [3.2 Test dataset](#3.2)
#     - [3.3 Submission](#3.3)
# - [4 Features](#4)
#     - [4.1 Missing values](#4.1)
#        - [4.1.1 Preparation](#4.1.1)
#        - [4.1.2 Individual features](#4.1.2)
#        - [4.1.3 Individual rows](#4.1.3)
#        - [4.1.3 Dealing with missing values (reference)](#4.1.4)
#     - [4.2 Distribution](#4.2)
# - [5 Target](#5)
#     - [5.1 Distribution](#5.1)
#     - [5.2 Target & missing value](#5.2)
# - [6 Model](#6)
#     - [6.1 Base model](#6.1)
#         - [6.1.1 XGBoost Classifier](#6.1.1)
#         - [6.1.2 LGBM Classifier](#6.1.2)
#         - [6.1.3 Catboost Classifier](#6.1.3)
#     - [6.2 Base model & feature engineering](#6.2)
#         - [6.2.1 Log](#6.2.1)
#         - [6.2.2 Minimum of features](#6.2.2)
#         - [6.2.3 Maximum of features](#6.2.3)
#         - [6.2.4 Sum of features](#6.2.4) 
#         - [6.2.5 Multiplication of feature](#6.2.5) 
#         - [6.2.6 Prorate of features](#6.2.6) 
#         - [6.2.7 Exponential of features](#6.2.7) 
#     

# [back to top](#table-of-contents)
# <a id="1"></a>
# # 1 Introduction
# 
# Kaggle competitions are incredibly fun and rewarding, but they can also be intimidating for people who are relatively new in their data science journey. In the past, Kaggle have launched many Playground competitions that are more approachable than Featured competition, and thus more beginner-friendly.
# 
# The goal of these competitions is to provide a fun, but less challenging, tabular dataset. These competitions will be great for people looking for something in between the Titanic Getting Started competition and a Featured competition.
# 
# The dataset is used for this competition is synthetic, but based on a real dataset and generated using a CTGAN. The original dataset deals with predicting whether a claim will be made on an insurance policy. Although the features are anonymized, they have properties relating to real-world features.
# 
# This competition will asked to predict whether a customer made a claim upon an insurance policy. The ground truth claim is binary valued, but a prediction may be any number from 0.0 to 1.0, representing the probability of a claim. The features in this dataset have been anonymized and may contain missing values.
# 
# Submissions are evaluated on **area under the ROC curve** between the predicted probability and the observed target.

# [back to top](#table-of-contents)
# <a id="2"></a>
# # 2 Preparations
# Preparing packages and data that will be used in the analysis process. Packages that will be loaded are mainly for data manipulation, data visualization and modeling. There are 2 datasets that are used in the analysis, they are train and test dataset. The main use of train dataset is to train models and use it to predict test dataset. While sample submission file is used to informed participants on the expected submission for the competition. *(to see the details, please expand)*

# In[1]:


# import packages
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
train_df = pd.read_csv('../input/tabular-playground-series-sep-2021/train.csv')
test_df = pd.read_csv('../input/tabular-playground-series-sep-2021/test.csv')
submission = pd.read_csv('../input/tabular-playground-series-sep-2021/sample_solution.csv')


# [back to top](#table-of-contents)
# <a id="3"></a>
# # 3 Dataset Overview
# The intend of the overview is to get a feel of the data and its structure in train, test and submission file. An overview on train and test datasets will include a quick analysis on missing values and basic statistics, while sample submission will be loaded to see the expected submission.
# 
# <a id="3.1"></a>
# ## 3.1 Train dataset
# As stated before, train dataset is mainly used to train predictive model as there is an available target variable in this set. This dataset is also used to explore more on the data itself including find a relation between each predictors and the target variable.
# 
# **Observations:**
# - `claim` column is the target variable which is only available in the `train` dataset.
# - There are `120` columns: `118` features, `1` target variable `claim` and `1` column of `id`.
# - `train` dataset contain `957,919` observation with `1,820,782` missing values which need to be treated carefully.
# 
# 
# ### 3.1.1 Quick view
# Below is the first 5 rows of train dataset:

# In[2]:


train_df.head()


# In[3]:


print(f'Number of rows: {train_df.shape[0]};  Number of columns: {train_df.shape[1]}; No of missing values: {sum(train_df.isna().sum())}')


# ### 3.1.2 Basic statistics
# Below is the basic statistics for each variables which contain information on `count`, `mean`, `standard deviation`, `minimum`, `1st quartile`, `median`, `3rd quartile` and `maximum`.

# In[4]:


train_df.describe()


# [back to top](#table-of-contents)
# <a id="3.2"></a>
# ## 3.2 Test dataset
# Test dataset is used to make a prediction based on the model that has previously trained. Exploration in this dataset is also needed to see how the data is structured and especially on it’s similiarity with the train dataset.
# 
# **Observations:**
# - There are `119` columns: `118` features and `1` column of `id`.
# - `train` dataset contain `493,474` observation with `936,218` missing values which need to be treated carefully.
# 
# ### 3.2.1 Quick view
# Below is the first 5 rows of test dataset:

# In[5]:


test_df.head()


# In[6]:


print(f'Number of rows: {test_df.shape[0]};  Number of columns: {test_df.shape[1]}; No of missing values: {sum(test_df.isna().sum())}')


# [back to top](#table-of-contents)
# <a id="3.3"></a>
# ## 3.3 Submission
# The submission file is expected to have an `id` and `claim` columns.
# 
# Below is the first 5 rows of submission file:

# In[7]:


submission.head()


# [back to top](#table-of-contents)
# <a id="4"></a>
# # 4 Features
# Number of features available to be used to create a prediction model are `118`.
# 
# <a id="4.1"></a>
# ## 4.1 Missing values
# Counting number of missing value and it's relative with their respective observations between train & test dataset.
# 
# <a id="4.1.1"></a>
# ### 4.1.1 Preparation
# Prepare train and test dataset for data analysis and visualization. *(to see the details, please expand)*

# In[8]:


missing_train_df = pd.DataFrame(train_df.isna().sum())
missing_train_df = missing_train_df.drop(['id', 'claim']).reset_index()
missing_train_df.columns = ['feature', 'count']

missing_train_percent_df = missing_train_df.copy()
missing_train_percent_df['count'] = missing_train_df['count']/train_df.shape[0]

missing_test_df = pd.DataFrame(test_df.isna().sum())
missing_test_df = missing_test_df.drop(['id']).reset_index()
missing_test_df.columns = ['feature', 'count']

missing_test_percent_df = missing_test_df.copy()
missing_test_percent_df['count'] = missing_test_df['count']/test_df.shape[0]

features = [feature for feature in train_df.columns if feature not in ['id', 'claim']]
missing_train_row = train_df[features].isna().sum(axis=1)
missing_train_row = pd.DataFrame(missing_train_row.value_counts()/train_df.shape[0]).reset_index()
missing_train_row.columns = ['no', 'count']

missing_test_row = test_df[features].isna().sum(axis=1)
missing_test_row = pd.DataFrame(missing_test_row.value_counts()/test_df.shape[0]).reset_index()
missing_test_row.columns = ['no', 'count']


# <a id="4.1.2"></a>
# ### 4.1.2 Individual features
# Count how many missing values in each features on `train` and `test` dataset to see if there any similiarity between them.
# 
# **Observations:**
# - Every features in `train` and `test` dataset has a missing value of around `1.6%`.
# - `train` dataset has a missing value of around `15,000` for each feature.
# - There are around `7,000 - 8,000` missing values for each feature in `test` dataset.

# In[9]:


plt.rcParams['figure.dpi'] = 600
fig = plt.figure(figsize=(2, 15), facecolor='#f6f5f5')
gs = fig.add_gridspec(1, 2)
gs.update(wspace=1.5, hspace=0.05)

background_color = "#f6f5f5"
sns.set_palette(['#ffd514']*120)

ax0 = fig.add_subplot(gs[0, 0])
for s in ["right", "top"]:
    ax0.spines[s].set_visible(False)
ax0.set_facecolor(background_color)
ax0_sns = sns.barplot(ax=ax0, y=missing_train_df['feature'], x=missing_train_df['count'], 
                      zorder=2, linewidth=0, orient='h', saturation=1, alpha=1)
ax0_sns.set_xlabel("missing values",fontsize=3, weight='bold')
ax0_sns.set_ylabel("features",fontsize=3, weight='bold')
ax0_sns.tick_params(labelsize=3, width=0.5, length=1.5)
ax0_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
ax0_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
ax0.text(0, -1.8, 'Train Dataset', fontsize=4, ha='left', va='top', weight='bold')
ax0.text(0, -1.105, 'Number of missing value are around 15,000 or 1.6%', fontsize=2.5, ha='left', va='top')
ax0.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
# data label
for p in ax0.patches:
    value = f'{p.get_width():,.0f} | {(p.get_width()/train_df.shape[0]):,.1%}'
    x = p.get_x() + p.get_width() + 1000
    y = p.get_y() + p.get_height() / 2 
    ax0.text(x, y, value, ha='left', va='center', fontsize=2, 
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.2))
    
background_color = "#f6f5f5"
sns.set_palette(['#ff355d']*120)
    
ax3 = fig.add_subplot(gs[0, 1])
for s in ["right", "top"]:
    ax3.spines[s].set_visible(False)
ax3.set_facecolor(background_color)
ax3_sns = sns.barplot(ax=ax3, y=missing_test_df['feature'], x=missing_test_df['count'], 
                      zorder=2, linewidth=0, orient='h', saturation=1, alpha=1)
ax3_sns.set_xlabel("missing values",fontsize=3, weight='bold')
ax3_sns.set_ylabel("features",fontsize=3, weight='bold')
ax3_sns.tick_params(labelsize=3, width=0.5, length=1.5)
ax3_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
ax3_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
ax3.text(0, -1.8, 'Test Dataset', fontsize=4, ha='left', va='top', weight='bold')
ax3.text(0, -1.105, 'Number of missing value are around 7,000 - 8,000 or 1.6%', fontsize=2.5, ha='left', va='top')
ax3.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
# data label
for p in ax3.patches:
    value = f'{p.get_width():,.0f} | {(p.get_width()/test_df.shape[0]):,.1%}'
    x = p.get_x() + p.get_width() + 500
    y = p.get_y() + p.get_height() / 2 
    ax3.text(x, y, value, ha='left', va='center', fontsize=2, 
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.2))


# <a id="4.1.3"></a>
# ### 4.1.3 Individual rows
# Count how many missing values in each rows on `train` and `test` dataset to see if there any similiarity between them.
# 
# **Observations:**
# - The maximum of missing value in an observation is `14` and the lowest is `no missing value`.
# - Interestingly, the missing value distribution (row basis) is quite the same between `train` and `test` dataset.
# - Though there is around 2% of missing value in each features, there are around `38%` of the observations (row basis) that has no missing values.
# - In reverse, there are `62%` observations that has missing value.
# - `1 to 3` missing values in the a observations constitute around `41%` of total observations. 

# In[10]:


background_color = "#f6f5f5"

plt.rcParams['figure.dpi'] = 600
fig = plt.figure(figsize=(6, 2), facecolor='#f6f5f5')
gs = fig.add_gridspec(1, 2)
gs.update(wspace=0.3, hspace=0.3)

run_no = 0
for row in range(0, 1):
    for col in range(0, 2):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        for s in ["top","right"]:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1  

sns.barplot(ax=ax0, x=missing_train_row['no'], y=missing_train_row['count'], saturation=1, zorder=2, color='#ffd514')
ax0.grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
ax0.grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
ax0.set_ylabel('')
ax0.set_xlabel('train dataset', fontsize=4, fontweight='bold')
ax0.tick_params(labelsize=4, width=0.5)
ax0.xaxis.offsetText.set_fontsize(4)
ax0.yaxis.offsetText.set_fontsize(4)

ax0.text(-0.5, 0.44, 'Missing Values - Row Basis', fontsize=6, ha='left', va='top', weight='bold')
ax0.text(-0.5, 0.415, 'Train and test dataset have quite a same distribution', fontsize=4, ha='left', va='top')

# data label
for p in ax0.patches:
    value = f'{p.get_height():.2f}'
    x = p.get_x() + p.get_width() - 0.4
    y = p.get_y() + p.get_height() + 0.01 
    ax0.text(x, y, value, ha='center', va='center', fontsize=2.5, 
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.2))

sns.barplot(ax=ax1, x=missing_test_row['no'], y=missing_test_row['count'], saturation=1, zorder=2, color='#ff355d')
ax1.grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
ax1.grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
ax1.set_ylabel('')
ax1.set_xlabel('test dataset', fontsize=4, fontweight='bold')
ax1.tick_params(labelsize=4, width=0.5)
ax1.xaxis.offsetText.set_fontsize(4)
ax1.yaxis.offsetText.set_fontsize(4)

# data label
for p in ax1.patches:
    value = f'{p.get_height():.2f}'
    x = p.get_x() + p.get_width() - 0.4
    y = p.get_y() + p.get_height() + 0.01 
    ax1.text(x, y, value, ha='center', va='center', fontsize=2.5, 
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.2))

plt.show()


# <a id="4.1.4"></a>
# ### 4.1.4 Dealing with missing value (reference)
# Some references on how to deal with missing value:
# - [Missing Values](https://www.kaggle.com/alexisbcook/missing-values) by [Alexis Cook](https://www.kaggle.com/alexisbcook)
# - [Data Cleaning Challenge: Handling missing values](https://www.kaggle.com/rtatman/data-cleaning-challenge-handling-missing-values) by [Rachael Tatman](https://www.kaggle.com/rtatman)
# - [A Guide to Handling Missing values in Python ](https://www.kaggle.com/parulpandey/a-guide-to-handling-missing-values-in-python) by [Parul Pandey](https://www.kaggle.com/parulpandey)
# 
# Some models that have capability to handle missing value by default are:
# - XGBoost: https://xgboost.readthedocs.io/en/latest/faq.html
# - LightGBM: https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html
# - Catboost: https://catboost.ai/docs/concepts/algorithm-missing-values-processing.html

# [back to top](#table-of-contents)
# <a id="4.2"></a>
# ## 4.2 Distribution
# Showing distribution on each feature that are available in train and test dataset. As there are 118 features, it will be broken down into 25 features for each sections. `Yellow` represents train dataset while `pink` will represent test dataset
# 
# **Observations:**
# - All features distribution on train and test dataset are almost similar.
# 
# ### 4.2.1 Features f1 - f25

# In[11]:


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

features = list(train_df.columns[1:26])

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


# ### 4.2.2 Features f26 - f50

# In[12]:


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


# ### 4.2.3 Features f51 - f75

# In[13]:


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

features = list(train_df.columns[51:76])

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


# ### 4.2.4 Features f76 - f100

# In[14]:


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

features = list(train_df.columns[76:101])

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


# ### 4.2.5 Features f101 - f118

# In[15]:


plt.rcParams['figure.dpi'] = 600
fig = plt.figure(figsize=(10, 10), facecolor='#f6f5f5')
gs = fig.add_gridspec(4, 5)
gs.update(wspace=0.3, hspace=0.3)

run_no = 0
for row in range(0, 4):
    for col in range(0, 5):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        for s in ["top","right"]:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

features = list(train_df.columns[101:119])

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

ax18.remove()
ax19.remove()
    
plt.show()


# [back to top](#table-of-contents)
# <a id="5"></a>
# # 5 Target
# 
# <a id="5.1"></a>
# ## 5.1 Distribution
# Target variable has a value of `0` to `1` which indicate people that not claim and claim the insurance. Let's see how the distribution of the `claim` variable.
# 
# **Observations:**
# - The number of people that not claim and claim (`0` and `1`) are almost the same of `480,404` and `477,515`, respectively.
# - In term of percentage both of people that claim and not claim are around 50%.

# In[16]:


claim_df = pd.DataFrame(train_df['claim'].value_counts()).reset_index()
claim_df.columns = ['claim', 'count']

claim_percent_df = pd.DataFrame(train_df['claim'].value_counts()/train_df.shape[0]).reset_index()
claim_percent_df.columns = ['claim', 'count']

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
ax0_sns = sns.barplot(ax=ax0, y=claim_df['claim'], x=claim_df['count'], 
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
ax1_sns = sns.barplot(ax=ax1, y=claim_percent_df['claim'], x=claim_percent_df['count'], 
                      zorder=2, linewidth=0, orient='h', saturation=1, alpha=1)
ax1_sns.set_xlabel("percentage",fontsize=3, weight='bold')
ax1_sns.set_ylabel("",fontsize=3, weight='bold')
ax1_sns.tick_params(labelsize=3, width=0.5, length=1.5)
ax1_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
ax1_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
ax1.text(0, -0.8, 'Claim in %', fontsize=4, ha='left', va='top', weight='bold')
ax1.text(0, -0.65, 'Both of 0 and 1 distributrion are alomost the same of 50%', fontsize=2.5, ha='left', va='top')
# data label
for p in ax1.patches:
    value = f'{p.get_width():.2f}'
    x = p.get_x() + p.get_width() + 0.01
    y = p.get_y() + p.get_height() / 2 
    ax1.text(x, y, value, ha='left', va='center', fontsize=2, 
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.2))


# <a id="5.2"></a>
# ## 5.2 Target & missing value
# As missing value constitute of `62%` of the observations, it's a good idea to explore if higher numbers of missing values in a observation relates to higher probability of claim. An assumption that customers that disclose every information required may be less likely to cheat and they may be more honest.
# 
# **Observations:**
# - An observation that has no missing value has the lowest probability to claim with only `14%`.
# - Observation that has a missing value (`1`) increased the probability to claim to `58%`.
# - A missing value between `2 to 13` has probability to claim above `70%`, while it drop for a missing value of `14` to `50%`.
# - This may be used for `target encoding` in `feature engineering`

# In[17]:


features = [feature for feature in train_df.columns if feature not in ['id', 'claim']]
train_df['no_missing'] = train_df[features].isna().sum(axis=1)
test_df['no_missing'] = test_df[features].isna().sum(axis=1)

missing_target = pd.DataFrame(train_df.groupby('no_missing')['claim'].agg('mean')).reset_index()
missing_target.columns = ['no', 'mean']

background_color = "#f6f5f5"

plt.rcParams['figure.dpi'] = 600
fig = plt.figure(figsize=(6, 2), facecolor='#f6f5f5')
gs = fig.add_gridspec(1, 1)
gs.update(wspace=0.3, hspace=0.3)

ax0 = fig.add_subplot(gs[0, 0])
ax0.set_facecolor(background_color)
for s in ["top","right"]:
    ax0.spines[s].set_visible(False)

sns.barplot(ax=ax0, x=missing_target['no'], y=missing_target['mean'], saturation=1, zorder=2, color='#ffd514')
ax0.grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
ax0.grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
ax0.set_ylabel('')
ax0.set_xlabel('train dataset', fontsize=4, fontweight='bold')
ax0.tick_params(labelsize=4, width=0.5)
ax0.xaxis.offsetText.set_fontsize(4)
ax0.yaxis.offsetText.set_fontsize(4)

ax0.text(-0.5, 0.95, 'Target & missing value', fontsize=6, ha='left', va='top', weight='bold')
ax0.text(-0.5, 0.9, 'Observations that has no missing value has the lowest probability to claim', fontsize=4, ha='left', va='top')

# data label
for p in ax0.patches:
    value = f'{p.get_height():.2f}'
    x = p.get_x() + p.get_width() - 0.4
    y = p.get_y() + p.get_height() + 0.02
    ax0.text(x, y, value, ha='center', va='center', fontsize=2.5, 
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.2))

plt.show()


# [back to top](#table-of-contents)
# <a id="6"></a>
# # 6 Model
# Evaluate the performance of base model. Models will be evaluated using five cross validation without any hyperparameters tuning. *(to see the packages used, please expand)*

# In[18]:


# import packages
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from scipy.stats import boxcox
from xgboost import XGBClassifier 
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# import datasets
train_df = pd.read_csv('../input/tabular-playground-series-sep-2021/train.csv')

folds = 5
features = list(train_df.columns[1:119])


# [back to top](#table-of-contents)
# <a id="6.1"></a>
# ## 6.1 Base model
# Models that will be evaluated are `XGBoost Classifier`, `LGBM Classifier` and `Catboost Classifier`.
# 
# **Observations:**
# - All 3 models have quite a same AUC result at around `0.8`. The differences are very small among the models.
# - `Catboost Classifier` has the best result with `0.803`.
# - `XGBoost Classifier` is the worst performing model with `0.799`.
# - The second place is hold by `LGBM Classifier` with `0.801`.

# <a id="6.1.1"></a>
# ### 6.1.1 XGBoost Classifier

# In[19]:


train_oof = np.zeros((957919,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df[features], train_df['claim'])):
    X_train, X_valid = train_df.iloc[train_idx], train_df.iloc[valid_idx]
    y_train = X_train['claim']
    y_valid = X_valid['claim']
    X_train = X_train.drop('claim', axis=1)
    X_valid = X_valid.drop('claim', axis=1)

    model = XGBClassifier(random_state=42, verbosity=0, tree_method='gpu_hist')

    model =  model.fit(X_train, y_train, verbose=0)
    temp_oof = model.predict_proba(X_valid)[:, 1]
    train_oof[valid_idx] = temp_oof
    print(f'Fold {fold} AUC: ', roc_auc_score(y_valid, temp_oof))
    
print(f'OOF AUC: ', roc_auc_score(train_df['claim'], train_oof))


# <a id="6.1.2"></a>
# ### 6.1.2 LGBM Classifier

# In[20]:


train_oof = np.zeros((957919,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df[features], train_df['claim'])):
    X_train, X_valid = train_df.iloc[train_idx], train_df.iloc[valid_idx]
    y_train = X_train['claim']
    y_valid = X_valid['claim']
    X_train = X_train.drop('claim', axis=1)
    X_valid = X_valid.drop('claim', axis=1)

    model = LGBMClassifier(random_state=42)

    model =  model.fit(X_train, y_train, verbose=0)
    temp_oof = model.predict_proba(X_valid)[:, 1]
    train_oof[valid_idx] = temp_oof
    print(f'Fold {fold} AUC: ', roc_auc_score(y_valid, temp_oof))
    
print(f'OOF AUC: ', roc_auc_score(train_df['claim'], train_oof))


# <a id="6.1.3"></a>
# ### 6.1.3 Catboost Classifier

# In[21]:


train_oof = np.zeros((957919,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df[features], train_df['claim'])):
    X_train, X_valid = train_df.iloc[train_idx], train_df.iloc[valid_idx]
    y_train = X_train['claim']
    y_valid = X_valid['claim']
    X_train = X_train.drop('claim', axis=1)
    X_valid = X_valid.drop('claim', axis=1)

    model = CatBoostClassifier(random_state=42)

    model =  model.fit(X_train, y_train, verbose=0)
    temp_oof = model.predict_proba(X_valid)[:, 1]
    train_oof[valid_idx] = temp_oof
    print(f'Fold {fold} AUC: ', roc_auc_score(y_valid, temp_oof))
    
print(f'OOF AUC: ', roc_auc_score(train_df['claim'], train_oof))


# [back to top](#table-of-contents)
# <a id="6.2"></a>
# ## 6.2 Base model & feature engineering
# This section will `blindly` try feature engineering using previous created notebook [TPS Feb 2021 Base Model & Features Engineering](https://www.kaggle.com/dwin183287/tps-feb-2021-base-model-features-engineering), to see if there are any new features that are useful. This section will use `LGBM Classifier` as the base model.
# 
# **Observations:**
# - Adding up `multiply` feature increased the model performance which can be seen in `6.2.5 Multiplication of features`.
# - Calculate minimum of all features in a row and put it in a new column, slightly increased the model performance, which can be seen in `6.2.2 Minimum of features`.
# - Others feature engineering attempts decrease the model performance.
# 
# <a id="6.2.1"></a>
# ### 6.2.1 Log
# It seems converting all the features into a log decrease the OOF AUC substantialy from `0.801` to `0.701`.

# In[22]:


train_df = pd.read_csv('../input/tabular-playground-series-sep-2021/train.csv')
train_oof = np.zeros((957919,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df[features], train_df['claim'])):
    X_train, X_valid = train_df.iloc[train_idx], train_df.iloc[valid_idx]
    y_train = X_train['claim']
    y_valid = X_valid['claim']
    X_train = X_train.drop('claim', axis=1)
    X_valid = X_valid.drop('claim', axis=1)
    
    X_train = np.log(X_train)
    X_valid = np.log(X_valid)
    
    model = LGBMClassifier(random_state=42)
    
    model =  model.fit(X_train, y_train, verbose=0)
    temp_oof = model.predict_proba(X_valid)[:, 1]
    train_oof[valid_idx] = temp_oof
    print(f'Fold {fold} AUC: ', roc_auc_score(y_valid, temp_oof))
    
print(f'OOF AUC: ', roc_auc_score(train_df['claim'], train_oof))


# <a id="6.2.2"></a>
# ### 6.2.2 Minimum of features
# Create a new feature `min` that calculate the minimum value in a row. There is a very small improvement in the model from `0.801517` to `0.801538`.

# In[23]:


train_df = pd.read_csv('../input/tabular-playground-series-sep-2021/train.csv')
train_oof = np.zeros((957919,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df[features], train_df['claim'])):
    X_train, X_valid = train_df.iloc[train_idx], train_df.iloc[valid_idx]
    y_train = X_train['claim']
    y_valid = X_valid['claim']
    X_train = X_train.drop('claim', axis=1)
    X_valid = X_valid.drop('claim', axis=1)
    
    X_train['min'] = X_train.min(axis=1)
    X_valid['min'] = X_valid.min(axis=1)
    
    model = LGBMClassifier(random_state=42)
    
    model =  model.fit(X_train, y_train, verbose=0)
    temp_oof = model.predict_proba(X_valid)[:, 1]
    train_oof[valid_idx] = temp_oof
    print(f'Fold {fold} AUC: ', roc_auc_score(y_valid, temp_oof))
    
print(f'OOF AUC: ', roc_auc_score(train_df['claim'], train_oof))


# <a id="6.2.3"></a>
# ### 6.2.3 Maximum of features
# Create a new feature `max` that calculate the maximum value in a row. Adding `max` to the model doesn't change the model performance.

# In[24]:


train_df = pd.read_csv('../input/tabular-playground-series-sep-2021/train.csv')
train_oof = np.zeros((957919,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df[features], train_df['claim'])):
    X_train, X_valid = train_df.iloc[train_idx], train_df.iloc[valid_idx]
    y_train = X_train['claim']
    y_valid = X_valid['claim']
    X_train = X_train.drop('claim', axis=1)
    X_valid = X_valid.drop('claim', axis=1)
    
    X_train['max'] = X_train.max(axis=1)
    X_valid['max'] = X_valid.max(axis=1)
    
    model = LGBMClassifier(random_state=42)
    
    model =  model.fit(X_train, y_train, verbose=0)
    temp_oof = model.predict_proba(X_valid)[:, 1]
    train_oof[valid_idx] = temp_oof
    print(f'Fold {fold} AUC: ', roc_auc_score(y_valid, temp_oof))
    
print(f'OOF AUC: ', roc_auc_score(train_df['claim'], train_oof))


# <a id="6.2.4"></a>
# ### 6.2.4 Sum of features
# Create a new feature `sum` that summing up all features values in a row. It seems adding up the `sum` of all features slightly decrease the model performance from `0.80151796` to `0.80151795`.

# In[25]:


train_df = pd.read_csv('../input/tabular-playground-series-sep-2021/train.csv')
train_oof = np.zeros((957919,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df[features], train_df['claim'])):
    X_train, X_valid = train_df.iloc[train_idx], train_df.iloc[valid_idx]
    y_train = X_train['claim']
    y_valid = X_valid['claim']
    X_train = X_train.drop('claim', axis=1)
    X_valid = X_valid.drop('claim', axis=1)
    
    X_train['sum'] = X_train.sum(axis=1)
    X_valid['sum'] = X_valid.sum(axis=1)
    
    model = LGBMClassifier(random_state=42)
    
    model =  model.fit(X_train, y_train, verbose=0)
    temp_oof = model.predict_proba(X_valid)[:, 1]
    train_oof[valid_idx] = temp_oof
    print(f'Fold {fold} AUC: ', roc_auc_score(y_valid, temp_oof))
    
print(f'OOF AUC: ', roc_auc_score(train_df['claim'], train_oof))


# <a id="6.2.5"></a>
# ### 6.2.5 Multiplication of features
# Create a new feature `multiply` that multiply all features values in a row. Adding up `multiply` column increase the model to `0.8019` from `0.8015`.

# In[26]:


train_df = pd.read_csv('../input/tabular-playground-series-sep-2021/train.csv')
train_oof = np.zeros((957919,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df[features], train_df['claim'])):
    X_train, X_valid = train_df.iloc[train_idx], train_df.iloc[valid_idx]
    y_train = X_train['claim']
    y_valid = X_valid['claim']
    X_train = X_train.drop('claim', axis=1)
    X_valid = X_valid.drop('claim', axis=1)
    
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
    
print(f'OOF AUC: ', roc_auc_score(train_df['claim'], train_oof))


# <a id="6.2.6"></a>
# ### 6.2.6 Prorate of features
# Create new `prorate` of each feature that calculate every feature contribution to the sum of all features. Adding `prorate` for each features doesn't improve the model performance. It is decreasing the model from `0.8015` to `0.8014`.

# In[27]:


train_df = pd.read_csv('../input/tabular-playground-series-sep-2021/train.csv')
train_oof = np.zeros((957919,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df[features], train_df['claim'])):
    X_train, X_valid = train_df.iloc[train_idx], train_df.iloc[valid_idx]
    y_train = X_train['claim']
    y_valid = X_valid['claim']
    X_train = X_train.drop('claim', axis=1)
    X_valid = X_valid.drop('claim', axis=1)
    
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
    
print(f'OOF AUC: ', roc_auc_score(train_df['claim'], train_oof))


# <a id="6.2.7"></a>
# ### 6.2.7 Exponential of features
# It seems converting all the features into a exponential decrease the OOF AUC from `0.8015` to `0.8011`.

# In[28]:


train_df = pd.read_csv('../input/tabular-playground-series-sep-2021/train.csv')
train_oof = np.zeros((957919,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df[features], train_df['claim'])):
    X_train, X_valid = train_df.iloc[train_idx], train_df.iloc[valid_idx]
    y_train = X_train['claim']
    y_valid = X_valid['claim']
    X_train = X_train.drop('claim', axis=1)
    X_valid = X_valid.drop('claim', axis=1)
    
    X_train = np.exp(X_train)
    X_valid = np.exp(X_valid)
    
    model = LGBMClassifier(random_state=42)
    
    model =  model.fit(X_train, y_train, verbose=0)
    temp_oof = model.predict_proba(X_valid)[:, 1]
    train_oof[valid_idx] = temp_oof
    print(f'Fold {fold} AUC: ', roc_auc_score(y_valid, temp_oof))
    
print(f'OOF AUC: ', roc_auc_score(train_df['claim'], train_oof))


# Thank you for reading. If you have any critics or find anything wrong, please let me know. I hope you enjoy it.
