#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
# 
# <a id="table-of-contents"></a>
# 1. [Introduction](#introduction)
# 2. [Preparation](#preparation)
# 3. [EDA](#eda)
#     * 3.1 [General](#general)
#     * 3.2 [Features Distribution](#features_distribution)
#     * 3.3 [Features Correlation](#features_correlation)
#     * 3.4 [Features vs Target](#features_target)
#     * 3.5 [Features and Target by Time](#features_target_time)
#     * 3.6 [Features and Target by Division](#features_target_division)
#     * 3.7 [Features and Target by Substraction](#features_target_substraction)
# 4. [Simple Models](#simple_models)
#     * 4.1 [Linear Regression](#linear_regression)
#     * 4.2 [Decision Tree Regressor](#decision_tree_regressor)
#     * 4.3 [Random Forest](#random_forest)
#     * 4.4 [LGBM](#lgbm)
#     * 4.5 [XGBoost](#xgboost)
#     * 4.6 [CatBoost](#catboost)
#     * 4.7 [AdaBoost](#adaboost)
#     * 4.8 [Deep Neural Network](#deep_neural_network)
# 5. [Simple Models Result](#simple_models_result)    
#     * 5.1 [Linear Regression](#linear_regression_result)
#     * 5.2 [Decision Tree Regressor](#decision_tree_regressor_result)
#     * 5.3 [Random Forest](#random_forest_result)
#     * 5.4 [LGBM](#lgbm_result)
#     * 5.5 [XGBoost](#xgboost_result)
#     * 5.6 [CatBoost](#catbosst_result)
#     * 5.7 [AdaBoost](#adaboost_result)
# 6. [Optuna Hyperparameters Tuning](#hyperparameters)
#     * 6.1 [LGBM](#lgbm_hpt)
#     * 6.2 [XGBoost](#xgboost_hpt)
# 7. [Target Splitting](#target_splitting)
#     * 7.1 [Below 8 Feature](#below_feature)
#     * 7.2 [Below 8 Result](#below_result)
# 8. [Winners Solutions](#winners_solutions)

# [back to top](#table-of-contents)
# <a id="introduction"></a>
# # 1. Introduction
# 
# Kaggle competitions are incredibly fun and rewarding, but they can also be intimidating for people who are relatively new in their data science journey. In the past, Kaggle have launched many Playground competitions that are more approachable than Featured competition, and thus more beginner-friendly.
# 
# The goal of these competitions is to provide a fun, but less challenging, tabular dataset. These competitions will be great for people looking for something in between the Titanic Getting Started competition and a Featured competition. 

# [back to top](#table-of-contents)
# <a id="preparation"></a>
# # 2. Preparation

# In[1]:


import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict,RandomizedSearchCV, KFold

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

import optuna


# In[2]:


train_df = pd.read_csv('/kaggle/input/tabular-playground-series-jan-2021/train.csv')
test_df = pd.read_csv('/kaggle/input/tabular-playground-series-jan-2021/test.csv')
submission = pd.read_csv('/kaggle/input/tabular-playground-series-jan-2021/sample_submission.csv')


# In[3]:


features = [feature for feature in train_df.columns if feature not in ['id', 'target']]
X_train = train_df[features]
y_train = train_df['target']
X_test = test_df[features]


# [back to top](#table-of-contents)
# <a id="eda"></a>
# # 3. EDA
# 
# <a id="general"></a>
# ## 3.1 General
# **Observations:**
# * There are 300,000 rows with 16 columns in the train dataset and 200,000 rows with 15 columns in test dataset. 
# * There are 14 features (columns with prefix 'cont'), 1 id and 1 target column. Target column is not included in the test set.
# * There is no missing values in the train and test dataset.
# * cont1 to cont14 features roughly have a range from 0 to 1. 
# * target variable roughly has a range from 0 to 10.

# In[4]:


print('Rows and Columns in train dataset:', train_df.shape)
print('Rows and Columns in test dataset:', test_df.shape)


# In[5]:


print('First 5 data in the train dataset:')
train_df.head()


# In[6]:


print('First 5 data in the test dataset:')
test_df.head()


# In[7]:


print('Missing value in train dataset:', sum(train_df.isnull().sum()))
print('Missing value in test dataset:', sum(test_df.isnull().sum()))


# In[8]:


print('Statistics on Train dataset')
train_df.describe()


# In[9]:


print('Statistics on Test dataset')
test_df.describe()


# [back to top](#table-of-contents)
# <a id="features_distribution"></a>
# ## 3.2 Features and Target Distribution
# **Observations:**
# * Train and test dataset features have bimodal or multimodal distributions.
# * Train and test datset features distribution more or less are the same, there is no significant gap between each features in the test and train dataset.

# In[10]:


fig = plt.figure(figsize=(12, 12), facecolor='#f6f6f6')
gs = fig.add_gridspec(4, 4)
gs.update(wspace=0.1, hspace=0.4)

background_color = "#f6f6f6"

run_no = 0
for row in range(0, 4):
    for col in range(0, 4):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        locals()["ax"+str(run_no)].tick_params(axis='y', left=False)
        locals()["ax"+str(run_no)].get_yaxis().set_visible(False)
        for s in ["top","right","left"]:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

run_no = 0
for feature in features:
        sns.kdeplot(train_df[feature] ,ax=locals()["ax"+str(run_no)], color='#ffd514', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)
        locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
        locals()["ax"+str(run_no)].set_xlabel(feature)
        run_no += 1
        
ax0.text(-0.2, 5, 'Features Distribution on Train Dataset', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.2, 4.5, 'All features have bimodal or multimodal distribution', fontsize=13, fontweight='light', fontfamily='serif')

for s in ["top", "bottom", "right","left"]:
    ax14.spines[s].set_visible(False)
    ax15.spines[s].set_visible(False)

ax14.tick_params(axis='x', bottom=False)
ax14.get_xaxis().set_visible(False)

ax15.tick_params(axis='x', bottom=False)
ax15.get_xaxis().set_visible(False)


# In[11]:


fig = plt.figure(figsize=(7.5, 4), facecolor='#f6f6f6')
gs = fig.add_gridspec(1, 1)
ax0 = fig.add_subplot(gs[0, 0])
sns.kdeplot(train_df['target'], ax=ax0, color='#ffd514', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)
ax0.grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
ax0.set_xlabel('target')

background_color = "#f6f6f6"

ax0.text(-0.9, 0.74, 'Target Distribution on Train Dataset', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.9, 0.68, 'Target feature has bimodal distribution', fontsize=13, fontweight='light', fontfamily='serif')
ax0.tick_params(axis='y', left=False)
ax0.get_yaxis().set_visible(False)
ax0.set_facecolor(background_color)

for s in ["top", "right", "left"]:
    ax0.spines[s].set_visible(False)


# In[12]:


fig = plt.figure(figsize=(12, 12), facecolor='#f6f6f6')
gs = fig.add_gridspec(4, 4)
gs.update(wspace=0.1, hspace=0.4)

background_color = "#f6f6f6"

run_no = 0
for row in range(0, 4):
    for col in range(0, 4):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        locals()["ax"+str(run_no)].tick_params(axis='y', left=False)
        locals()["ax"+str(run_no)].get_yaxis().set_visible(False)
        for s in ["top","right","left"]:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

run_no = 0
for feature in features:
        sns.kdeplot(test_df[feature] ,ax=locals()["ax"+str(run_no)], color='#ffd514', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)
        locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
        locals()["ax"+str(run_no)].set_xlabel(feature)
        run_no += 1
        
ax0.text(-0.2, 5.6, 'Features Distribution on Test Dataset', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.2, 5, 'Test features distribution resemble train features distribution', fontsize=13, fontweight='light', fontfamily='serif')

for s in ["top", "bottom", "right","left"]:
    ax14.spines[s].set_visible(False)
    ax15.spines[s].set_visible(False)

ax14.tick_params(axis='x', bottom=False)
ax14.get_xaxis().set_visible(False)

ax15.tick_params(axis='x', bottom=False)
ax15.get_xaxis().set_visible(False)


# In[13]:


X_all = pd.concat([X_train, X_test], axis=0)

fig = plt.figure(figsize=(12, 12), facecolor='#f6f6f6')
gs = fig.add_gridspec(4, 4)
gs.update(wspace=0.1, hspace=0.4)

background_color = "#f6f6f6"

run_no = 0
for row in range(0, 4):
    for col in range(0, 4):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        locals()["ax"+str(run_no)].tick_params(axis='y', left=False)
        locals()["ax"+str(run_no)].get_yaxis().set_visible(False)
        for s in ["top","right","left"]:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

run_no = 0
for feature in features:
        sns.kdeplot(X_all[feature] ,ax=locals()["ax"+str(run_no)], color='#ffd514', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)
        locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
        locals()["ax"+str(run_no)].set_xlabel(feature)
        run_no += 1
        
ax0.text(-0.2, 5.2, 'Combined Distribution of Train & Test Dataset', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.2, 4.6, 'Combined features between train & test dataset resemble individual distributions', fontsize=13, fontweight='light', fontfamily='serif')

for s in ["top", "bottom", "right","left"]:
    ax14.spines[s].set_visible(False)
    ax15.spines[s].set_visible(False)

ax14.tick_params(axis='x', bottom=False)
ax14.get_xaxis().set_visible(False)

ax15.tick_params(axis='x', bottom=False)
ax15.get_xaxis().set_visible(False)


# [back to top](#table-of-contents)
# <a id="features_correlation"></a>
# ## 3.3 Features Correlation
# 
# **Observations:**
# 
# * Correlation above 0.7 or below -0.7 are considered as high correlation. 
# * Features `cont1`, `cont6` and `cont9` to `cont13` have a high correlation with each others.
# 
# **Ideas:**
# * Consider to remove these features and retain 1 feature with the highest correlation with the target.

# In[14]:


fig = plt.figure(figsize=(18, 8), facecolor='#f6f6f6')
gs = fig.add_gridspec(1, 2)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
colors = ["#ffd514", "#f6f6f6","#ffd514"]
colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

ax0.set_facecolor(background_color)
ax0.text(0, -1, 'Features Correlation on Train Dataset', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(0, -0.4, 'Some features have a high correlation', fontsize=13, fontweight='light', fontfamily='serif')

ax1.set_facecolor(background_color)
ax1.text(-0.1, -1, 'Features Correlation on Test Dataset', fontsize=20, fontweight='bold', fontfamily='serif')
ax1.text(-0.1, -0.4, 'Features in test dataset resemble features in train dataset ', 
         fontsize=13, fontweight='light', fontfamily='serif')

sns.heatmap(X_train.corr()[X_train.corr() >= 0.7], ax=ax0, linewidths=.1, vmin=-1, vmax=1, annot=True, square=True, 
            cbar_kws={"orientation": "horizontal"}, cbar=False, cmap=colormap, fmt='.1g')

sns.heatmap(X_test.corr()[X_test.corr() >= 0.7], ax=ax1, linewidths=.1, vmin=-1, vmax=1, annot=True, square=True, 
            cbar_kws={"orientation": "horizontal"}, cbar=False, cmap=colormap, fmt='.1g')

plt.show()


# [back to top](#table-of-contents)
# <a id="features_target"></a>
# ## 3.4 Features and Target
# **Observations:**
# * The relation between features and the target is following the distribution of the features. There are 2 features that are interesting, they are:
#     * Nine distinct separations can be seen in the `cont2` feature.
#     * There are 2 distinct separations in the `cont14` relative to the target.
# * By multiplying the features by 10 will approximate a range that resemble target variable which can be used to compare the distribution between the target and the features. 
# 
# **Ideas:**
# * It is possible to transformed `cont2` and `cont14` into categorical variables to be used for target encoding.

# In[15]:


fig = plt.figure(figsize=(12, 12), facecolor='#f6f6f6')
gs = fig.add_gridspec(4, 4)
gs.update(wspace=0.5, hspace=0.5)

background_color = "#f6f6f6"

run_no = 0
for row in range(0, 4):
    for col in range(0, 4):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        for s in ["top","right","left"]:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

run_no = 0
for feature in features:
        sns.scatterplot(train_df[feature], train_df['target'] ,ax=locals()["ax"+str(run_no)], color='#ff819a', linewidth=0.3, edgecolor='#5a0012', zorder=3)
        locals()["ax"+str(run_no)].grid(which='major', zorder=0, color='gray', linestyle=':', dashes=(1,5))
        run_no += 1
        
ax0.text(-0.4, 13.8, 'Features and Target Relation', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.4, 12, 'cont2 and cont14 have a distinct separation', fontsize=13, fontweight='light', fontfamily='serif')

for s in ["top", "bottom", "right","left"]:
    ax14.spines[s].set_visible(False)
    ax15.spines[s].set_visible(False)

ax14.tick_params(axis='x', bottom=False)
ax14.tick_params(axis='y', left=False)
ax14.get_xaxis().set_visible(False)
ax14.get_yaxis().set_visible(False)

ax15.tick_params(axis='x', bottom=False)
ax15.tick_params(axis='y', left=False)
ax15.get_xaxis().set_visible(False)
ax15.get_yaxis().set_visible(False)


# In[16]:


fig = plt.figure(figsize=(12, 12), facecolor='#f6f6f6')
gs = fig.add_gridspec(4, 4)
gs.update(wspace=0.1, hspace=0.4)

background_color = "#f6f6f6"

run_no = 0
for row in range(0, 4):
    for col in range(0, 4):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        locals()["ax"+str(run_no)].tick_params(axis='y', left=False)
        locals()["ax"+str(run_no)].get_yaxis().set_visible(False)
        for s in ["top","right","left"]:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

run_no = 0
for feature in features:
        sns.kdeplot(train_df['target'], ax=locals()["ax"+str(run_no)], color='#ff819a', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)
        sns.kdeplot(train_df[feature] * 10 ,ax=locals()["ax"+str(run_no)], color='#ffd514', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)
        locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
        locals()["ax"+str(run_no)].set_xlabel(feature)
        run_no += 1
        
ax0.text(-0.2, 0.9, 'Features and Target Distribution Comparison', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.2, 0.8, 'By multiplying features with 10, make the comparison between features and target distribution possible', 
         fontsize=13, fontweight='light', fontfamily='serif')

for s in ["top", "bottom", "right","left"]:
    ax14.spines[s].set_visible(False)
    ax15.spines[s].set_visible(False)

ax14.tick_params(axis='x', bottom=False)
ax14.get_xaxis().set_visible(False)

ax15.tick_params(axis='x', bottom=False)
ax15.get_xaxis().set_visible(False)


# [back to top](#table-of-contents)
# <a id="features_target_time"></a>
# ## 3.5 Features and Target by Time Series
# Assuming id feature is a time feature
# 
# **Observations:**
# * Features are consistently distributed around 0 - 1 across time (id).
# * `cont10` has lower items on low and high value compared to other features. It gives a sense that the variance is high.
# * `cont7` and `cont9` have lower items on lower value with cont9 has lower items than `cont7`.
# 
# **Ideas:**
# * Removing records that has more variance in the `cont7`, `cont9` and `cont10`.

# In[17]:


fig = plt.figure(figsize=(15, 30), facecolor='#f6f6f6')
gs = fig.add_gridspec(15, 1)
gs.update(wspace=0.5, hspace=0.5)

background_color = "#f6f6f6"

run_no = 0
for row in range(0, 15):
    locals()["ax"+str(row)] = fig.add_subplot(gs[row, 0])
    locals()["ax"+str(row)].set_facecolor(background_color)
    for s in ["top","right","left"]:
        locals()["ax"+str(row)].spines[s].set_visible(False)
    run_no += 1

run_no = 0
for feature in features:
    sns.scatterplot(train_df['id'], train_df[feature],ax=locals()["ax"+str(run_no)],  color='#ffd514', linewidth=0, zorder=3)
    locals()["ax"+str(run_no)].grid(which='major', zorder=0, color='gray', linestyle=':', dashes=(1,5))
    run_no += 1
        
sns.scatterplot(train_df['id'], train_df['target'],ax=ax14, color='#ff819a', linewidth=0, zorder=3)
ax14.grid(which='major', zorder=0, color='gray', linestyle=':', dashes=(1,5))

ax0.text(-100, 1.6, 'Features and Target by Time Series on Train Dataset', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-100, 1.3, 'cont10 has lower items on low and high value compared to other features', 
         fontsize=13, fontweight='light', fontfamily='serif')

for s in ["top", "bottom", "right","left"]:
    ax15.spines[s].set_visible(False)

ax15.tick_params(axis='x', bottom=False)
ax15.tick_params(axis='y', left=False)
ax15.get_xaxis().set_visible(False)
ax15.get_yaxis().set_visible(False)


# [back to top](#table-of-contents)
# <a id="features_target_division"></a>
# ## 3.6 Features and Target by Division
# Divide the target with the individual feature. Individual feature has been multiply to 10 to more resemble with the target.
# 
# **Observations:**
# * Some features has a clear upward diagonal cut which can clearly be seen on `cont4`, `cont5`, `cont8`, `cont11`, `cont12` and `cont14`.
# * Due to small division between target and the feature there are some features that looks like exclamation mark as can be seen on `cont1`, `cont2`, `cont7`, `cont9` and `cont10`.
# 
# **Ideas:**
# * Triangle shape between the target and the division can be further explored for feature engineering.
# * `cont5` division can be futher explored for feature engineering as there is a quite a clear cut in the distribution 

# In[18]:


fig = plt.figure(figsize=(12, 12), facecolor='#f6f6f6')
gs = fig.add_gridspec(4, 4)
gs.update(wspace=0.5, hspace=0.5)

background_color = "#f6f6f6"

run_no = 0
for row in range(0, 4):
    for col in range(0, 4):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        for s in ["top","right","left"]:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

run_no = 0
for feature in features:
        sns.scatterplot(train_df['target']/(train_df[feature] * 10), train_df['target'] ,ax=locals()["ax"+str(run_no)], color='#ff819a', linewidth=0.3, edgecolor='#5a0012', zorder=3)
        locals()["ax"+str(run_no)].grid(which='major', zorder=0, color='gray', linestyle=':', dashes=(1,5))
        locals()["ax"+str(run_no)].set_xlabel(feature)
        run_no += 1
        
ax0.text(-15000, 13.8, 'Division and Target Relation on Train Dataset', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-15000, 12, 'There are feature division that has an upward diagonal cut ', fontsize=13, fontweight='light', fontfamily='serif')

for s in ["top", "bottom", "right","left"]:
    ax14.spines[s].set_visible(False)
    ax15.spines[s].set_visible(False)

ax14.tick_params(axis='x', bottom=False)
ax14.tick_params(axis='y', left=False)
ax14.get_xaxis().set_visible(False)
ax14.get_yaxis().set_visible(False)

ax15.tick_params(axis='x', bottom=False)
ax15.tick_params(axis='y', left=False)
ax15.get_xaxis().set_visible(False)
ax15.get_yaxis().set_visible(False)


# In[19]:


fig = plt.figure(figsize=(12, 12), facecolor='#f6f6f6')
gs = fig.add_gridspec(4, 4)
gs.update(wspace=0.1, hspace=0.4)

background_color = "#f6f6f6"

run_no = 0
for row in range(0, 4):
    for col in range(0, 4):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        locals()["ax"+str(run_no)].tick_params(axis='y', left=False)
        locals()["ax"+str(run_no)].get_yaxis().set_visible(False)
        for s in ["top","right","left"]:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

run_no = 0
for feature in features:
        sns.kdeplot(train_df['target']/(train_df[feature] * 10) ,ax=locals()["ax"+str(run_no)], color='#ffd514', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)
        locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
        locals()["ax"+str(run_no)].set_xlabel(feature)
        run_no += 1
        
ax0.text(-5000, 0.003, 'Division Distribution on Train Dataset', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-5000, 0.0027, 'cont5 shows a clear separation at the amount around 2', fontsize=13, fontweight='light', fontfamily='serif')

for s in ["top", "bottom", "right","left"]:
    ax14.spines[s].set_visible(False)
    ax15.spines[s].set_visible(False)

ax14.tick_params(axis='x', bottom=False)
ax14.get_xaxis().set_visible(False)

ax15.tick_params(axis='x', bottom=False)
ax15.get_xaxis().set_visible(False)


# [back to top](#table-of-contents)
# <a id="features_target_substraction"></a>
# ## 3.7 Features and Target by Substraction
# Substract the target with the individual feature. Individual feature has been multiply to 10 to more resemble with the target.
# 
# **Observations:**
# * Parallelogram is spotted on most of the substraction especially on `cont5`, `cont8` and `cont14`.
# * Substraction on `cont7` showing a one modal distribution.

# In[20]:


fig = plt.figure(figsize=(12, 12), facecolor='#f6f6f6')
gs = fig.add_gridspec(4, 4)
gs.update(wspace=0.5, hspace=0.5)

background_color = "#f6f6f6"

run_no = 0
for row in range(0, 4):
    for col in range(0, 4):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        for s in ["top","right","left"]:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

run_no = 0
for feature in features:
        sns.scatterplot(train_df['target'] - (train_df[feature] * 10), train_df['target'] ,ax=locals()["ax"+str(run_no)], color='#ff819a', linewidth=0.3, edgecolor='#5a0012', zorder=3)
        locals()["ax"+str(run_no)].grid(which='major', zorder=0, color='gray', linestyle=':', dashes=(1,5))
        locals()["ax"+str(run_no)].set_xlabel(feature)
        run_no += 1
        
ax0.text(-10, 13.5, 'Substraction and Target Relation on Train Dataset', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-10, 12, 'Most of the substraction showing a parallelogram shape to the target', fontsize=13, fontweight='light', fontfamily='serif')

for s in ["top", "bottom", "right","left"]:
    ax14.spines[s].set_visible(False)
    ax15.spines[s].set_visible(False)

ax14.tick_params(axis='x', bottom=False)
ax14.tick_params(axis='y', left=False)
ax14.get_xaxis().set_visible(False)
ax14.get_yaxis().set_visible(False)

ax15.tick_params(axis='x', bottom=False)
ax15.tick_params(axis='y', left=False)
ax15.get_xaxis().set_visible(False)
ax15.get_yaxis().set_visible(False)


# In[21]:


fig = plt.figure(figsize=(12, 12), facecolor='#f6f6f6')
gs = fig.add_gridspec(4, 4)
gs.update(wspace=0.1, hspace=0.4)

background_color = "#f6f6f6"

run_no = 0
for row in range(0, 4):
    for col in range(0, 4):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        locals()["ax"+str(run_no)].tick_params(axis='y', left=False)
        locals()["ax"+str(run_no)].get_yaxis().set_visible(False)
        for s in ["top","right","left"]:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

run_no = 0
for feature in features:
        sns.kdeplot(train_df['target'] - (train_df[feature] * 10) ,ax=locals()["ax"+str(run_no)], color='#ffd514', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)
        locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
        locals()["ax"+str(run_no)].set_xlabel(feature)
        run_no += 1
        
ax0.text(-6, 0.28, 'Substraction Distribution on Train Dataset', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-6, 0.25, 'cont7 shows a one modal distribution', fontsize=13, fontweight='light', fontfamily='serif')

for s in ["top", "bottom", "right","left"]:
    ax14.spines[s].set_visible(False)
    ax15.spines[s].set_visible(False)

ax14.tick_params(axis='x', bottom=False)
ax14.get_xaxis().set_visible(False)

ax15.tick_params(axis='x', bottom=False)
ax15.get_xaxis().set_visible(False)


# [back to top](#table-of-contents)
# <a id="simple_models"></a>
# # 4. Simple Models
# Results of models using default hyperparameters and without any feature engineering using 5 cross validations:
# * The best performance is **CatBoost**, the model will be used for the submission.
# * **Decision Tree Regressor** has the worst performance compared to others model.
# * The fastest model is **Linear Regression** and the performance is higher than **Decision Tree Regressor**. 
# * The most time consuming model is **Random Forest** with wall time of around 20 minutes. The prediction result is also medicore compared to other models. 
# * **AdaBoost** and **Linear Regression** performance are quite the same.
# * **LGBM** and **XGBoost** have quite the same performance with LGBM has the fastest time to process.
# * **Deep Neural Network** use 1 normalization layer, 2 hidden layers (64 and 64) and 1 output layer. It still can not beat plain vannila CatBoost.

# Setting up the 5 cross validation using KFold to get a consistent validation dataset:

# In[22]:


cv = KFold(n_splits=5, shuffle=True, random_state=42)


# [back to top](#table-of-contents)
# <a id="linear_regression"></a>
# ## 4.1 Linear Regression

# In[23]:


get_ipython().run_cell_magic('time', '', "lin_reg = LinearRegression()\nscores = cross_val_score(lin_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)\nlin_rmse_scores = np.sqrt(-scores)\nprint('Linear Regression performance:', lin_rmse_scores)\n")


# [back to top](#table-of-contents)
# <a id="decision_tree_regressor"></a>
# ## 4.2 Decision Tree Regressor 

# In[24]:


get_ipython().run_cell_magic('time', '', "tree_reg = DecisionTreeRegressor(random_state=42)\nscores = cross_val_score(tree_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)\ntree_rmse_scores = np.sqrt(-scores)\nprint('Decision Tree Regressor performance:', tree_rmse_scores)\n")


# [back to top](#table-of-contents)
# <a id="random_forest"></a>
# ## 4.3 Random Forest

# In[25]:


get_ipython().run_cell_magic('time', '', "forest_reg = RandomForestRegressor(random_state=42, n_jobs=-1)\nscores = cross_val_score(forest_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)\nforest_rmse_scores = np.sqrt(-scores)\nprint('Random Forest performance:', forest_rmse_scores)\n")


# [back to top](#table-of-contents)
# <a id="lgbm"></a>
# ## 4.4 LGBM

# In[26]:


get_ipython().run_cell_magic('time', '', "lgbm_reg = LGBMRegressor(random_state=42)\nscores = cross_val_score(lgbm_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)\nlgbm_rmse_scores = np.sqrt(-scores)\nprint('LGBM performance:', lgbm_rmse_scores)\n")


# [back to top](#table-of-contents)
# <a id="xgboost"></a>
# ## 4.5 XGBoost

# In[27]:


get_ipython().run_cell_magic('time', '', "xgb_reg = XGBRegressor(random_state=42)\nscores = cross_val_score(xgb_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)\nxgb_rmse_scores = np.sqrt(-scores)\nprint('XGBoost performance:', xgb_rmse_scores)\n")


# [back to top](#table-of-contents)
# <a id="catboost"></a>
# ## 4.6 CatBoost

# In[28]:


get_ipython().run_cell_magic('time', '', "cb_reg = CatBoostRegressor(random_state=42, verbose=False)\nscores = cross_val_score(cb_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)\ncb_rmse_scores = np.sqrt(-scores)\nprint('CatBoost performance:', cb_rmse_scores)\n")


# [back to top](#table-of-contents)
# <a id="adaboost"></a>
# ## 4.7 AdaBoost

# In[29]:


get_ipython().run_cell_magic('time', '', "ab_reg = AdaBoostRegressor(random_state=42)\nscores = cross_val_score(ab_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)\nab_rmse_scores = np.sqrt(-scores)\nprint('AdaBoost performance:', ab_rmse_scores)\n")


# [back to top](#table-of-contents)
# <a id="deep_neural_network"></a>
# ## 4.8 Deep Neural Network

# In[30]:


def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)])
    
    model.compile(loss='mean_squared_error',
                 optimizer=tf.keras.optimizers.Adam(0.001))
    return model


# In[31]:


get_ipython().run_cell_magic('time', '', 'normalizer = preprocessing.Normalization()\nnormalizer.adapt(np.array(X_train))\ndnn_model = build_and_compile_model(normalizer)\nhistory = dnn_model.fit(X_train, y_train, validation_split=0.2,\n                       verbose=0, epochs=100)\n')


# Below are the RMSE distribution on validation dataset from Deep Neural Network

# In[32]:


fig = plt.figure(figsize=(7.5, 4), facecolor='#f6f6f6')
gs = fig.add_gridspec(2, 1)
ax0 = fig.add_subplot(gs[0, 0])
sns.kdeplot(np.sqrt(history.history['val_loss']) ,ax=ax0, color='#ffd514', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)
ax0.grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))

background_color = "#f6f6f6"

ax0.tick_params(axis='y', left=False)
ax0.get_yaxis().set_visible(False)
ax0.set_facecolor(background_color)

for s in ["top", "right", "left"]:
    ax0.spines[s].set_visible(False)


# [back to top](#table-of-contents)
# <a id="simple_models_result"></a>
# # 5. Simple Models Results
# This section shows the result of the simple model prediction with comparison with target:
# * Scatter plot shows the prediction relative to the target **<font color="#ffd514">(yellow)</font>** and perfect prediction **<font color="#ff819a">(pink)</font>** will look like.
# * Distribution plot shows the prediction distribution **<font color="#ffd514">(yellow)</font>** and target distribution **<font color="#ff819a">(pink)</font>**.
# 
# **Observations:**
# * All of the model concentrate their predictions on 6.5 to 10 except for Decision Tree Regressor that have a more diverse result but not accurate
# * All of the model failed to follow the target bimodal distribution except for Decision Tree Regressor. It is follow the target distribution but the prediction is not accurate. 
# * All of the model result (except for Distribution Tree Regressor) are a unimodal distribution and they look like an exclamation mark if it compared with a perfect prediction. This is due to a short range prediction compared to target.
# 
# [back to top](#table-of-contents)
# <a id="linear_regression_result"></a>
# ## 5.1 Linear Regression

# In[33]:


get_ipython().run_cell_magic('time', '', 'lin_reg = LinearRegression()\ny_predict = cross_val_predict(lin_reg, X_train, y_train, cv=cv, n_jobs=-1)\n')


# In[34]:


fig = plt.figure(figsize=(10, 4), facecolor='#f6f6f6')
gs = fig.add_gridspec(1, 2)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
gs.update(wspace=0.2, hspace=0)
background_color = "#f6f6f6"

sns.scatterplot(y_train, y_train, ax=ax0, color='#ff819a', linewidth=0.3, edgecolor='#5a0012', zorder=3)
sns.scatterplot(y_predict, y_train, ax=ax0, color='#ffd514', linewidth=0.3, edgecolor='#4f4100', zorder=3)

sns.kdeplot(y_train, ax=ax1, color='#ff819a', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)
sns.kdeplot(y_predict, ax=ax1, color='#ffd514', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)

ax0.text(-1.5, 12, 'Linear Regression Results', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-1.5, 11, 'Most of the prediction concentrate in a range of 7 to 9', fontsize=13, fontweight='light', fontfamily='serif')

ax0.set_ylabel('target')
ax0.set_xlabel('prediction')
ax0.set_facecolor(background_color)
ax1.set_facecolor(background_color)
ax0.grid(which='major', zorder=0, color='gray', linestyle=':', dashes=(1,5))
ax1.grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))

for s in ["top", "right", "left"]:
    ax0.spines[s].set_visible(False)
    ax1.spines[s].set_visible(False)


# [back to top](#table-of-contents)
# <a id="decision_tree_regressor_result"></a>
# ## 5.2 Decision Tree Regressor 

# In[35]:


get_ipython().run_cell_magic('time', '', 'tree_reg = DecisionTreeRegressor(random_state=42)\ny_predict = cross_val_predict(tree_reg, X_train, y_train, cv=cv, n_jobs=-1)\n')


# In[36]:


fig = plt.figure(figsize=(10, 4), facecolor='#f6f6f6')
gs = fig.add_gridspec(1, 2)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
gs.update(wspace=0.2, hspace=0)
background_color = "#f6f6f6"

sns.scatterplot(y_train, y_train, ax=ax0, color='#ff819a', linewidth=0.3, edgecolor='#5a0012', zorder=3)
sns.scatterplot(y_predict, y_train, ax=ax0, color='#ffd514', linewidth=0.3, edgecolor='#4f4100', zorder=3)

sns.kdeplot(y_train, ax=ax1, color='#ff819a', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)
sns.kdeplot(y_predict, ax=ax1, color='#ffd514', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)

ax0.text(-1.5, 12, 'Decision Tree Regressor Results', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-1.5, 11, 'Prediction distribution resemble the target distribution but the accuracy are way off ', 
         fontsize=13, fontweight='light', fontfamily='serif')

ax0.set_ylabel('target')
ax0.set_xlabel('prediction')
ax0.set_facecolor(background_color)
ax1.set_facecolor(background_color)
ax0.grid(which='major', zorder=0, color='gray', linestyle=':', dashes=(1,5))
ax1.grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))

for s in ["top", "right", "left"]:
    ax0.spines[s].set_visible(False)
    ax1.spines[s].set_visible(False)


# [back to top](#table-of-contents)
# <a id="random_forest_result"></a>
# ## 5.3 Random Forest

# In[37]:


get_ipython().run_cell_magic('time', '', 'forest_reg = RandomForestRegressor(random_state=42, n_jobs=-1)\ny_predict = cross_val_predict(forest_reg, X_train, y_train, cv=cv, n_jobs=-1)\n')


# In[38]:


fig = plt.figure(figsize=(10, 4), facecolor='#f6f6f6')
gs = fig.add_gridspec(1, 2)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
gs.update(wspace=0.2, hspace=0)
background_color = "#f6f6f6"

sns.scatterplot(y_train, y_train, ax=ax0, color='#ff819a', linewidth=0.3, edgecolor='#5a0012', zorder=3)
sns.scatterplot(y_predict, y_train, ax=ax0, color='#ffd514', linewidth=0.3, edgecolor='#4f4100', zorder=3)

sns.kdeplot(y_train, ax=ax1, color='#ff819a', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)
sns.kdeplot(y_predict, ax=ax1, color='#ffd514', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)

ax0.text(-1.5, 12, 'Random Forest Regressor Results', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-1.5, 11, 'Most of the prediction concentrate in a range of 6.7 to 9.5', 
         fontsize=13, fontweight='light', fontfamily='serif')

ax0.set_ylabel('target')
ax0.set_xlabel('prediction')
ax0.set_facecolor(background_color)
ax1.set_facecolor(background_color)
ax0.grid(which='major', zorder=0, color='gray', linestyle=':', dashes=(1,5))
ax1.grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))

for s in ["top", "right", "left"]:
    ax0.spines[s].set_visible(False)
    ax1.spines[s].set_visible(False)


# [back to top](#table-of-contents)
# <a id="lgbm_result"></a>
# ## 5.4 LGBM

# In[39]:


get_ipython().run_cell_magic('time', '', 'lgbm_reg = LGBMRegressor(random_state=42)\ny_predict = cross_val_predict(lgbm_reg, X_train, y_train, cv=cv, n_jobs=-1)\n')


# In[40]:


fig = plt.figure(figsize=(10, 4), facecolor='#f6f6f6')
gs = fig.add_gridspec(1, 2)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
gs.update(wspace=0.2, hspace=0)
background_color = "#f6f6f6"

sns.scatterplot(y_train, y_train, ax=ax0, color='#ff819a', linewidth=0.3, edgecolor='#5a0012', zorder=3)
sns.scatterplot(y_predict, y_train, ax=ax0, color='#ffd514', linewidth=0.3, edgecolor='#4f4100', zorder=3)

sns.kdeplot(y_train, ax=ax1, color='#ff819a', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)
sns.kdeplot(y_predict, ax=ax1, color='#ffd514', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)

ax0.text(-1.5, 12, 'LGBM Results', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-1.5, 11, 'Most of the prediction concentrate in a range of 7 to 9.5', 
         fontsize=13, fontweight='light', fontfamily='serif')

ax0.set_ylabel('target')
ax0.set_xlabel('prediction')
ax0.set_facecolor(background_color)
ax1.set_facecolor(background_color)
ax0.grid(which='major', zorder=0, color='gray', linestyle=':', dashes=(1,5))
ax1.grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))

for s in ["top", "right", "left"]:
    ax0.spines[s].set_visible(False)
    ax1.spines[s].set_visible(False)


# [back to top](#table-of-contents)
# <a id="xgboost_result"></a>
# ## 5.5 XGBoost

# In[41]:


get_ipython().run_cell_magic('time', '', 'xgb_reg = XGBRegressor(random_state=42)\ny_predict = cross_val_predict(xgb_reg, X_train, y_train, cv=cv, n_jobs=-1)\n')


# In[42]:


fig = plt.figure(figsize=(10, 4), facecolor='#f6f6f6')
gs = fig.add_gridspec(1, 2)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
gs.update(wspace=0.2, hspace=0)
background_color = "#f6f6f6"

sns.scatterplot(y_train, y_train, ax=ax0, color='#ff819a', linewidth=0.3, edgecolor='#5a0012', zorder=3)
sns.scatterplot(y_predict, y_train, ax=ax0, color='#ffd514', linewidth=0.3, edgecolor='#4f4100', zorder=3)

sns.kdeplot(y_train, ax=ax1, color='#ff819a', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)
sns.kdeplot(y_predict, ax=ax1, color='#ffd514', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)

ax0.text(-1.5, 12.5, 'XGBoost Results', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-1.5, 11.5, 'Most of the prediction concentrate in a range of 6.5 to 10', 
         fontsize=13, fontweight='light', fontfamily='serif')

ax0.set_ylabel('target')
ax0.set_xlabel('prediction')
ax0.set_facecolor(background_color)
ax1.set_facecolor(background_color)
ax0.grid(which='major', zorder=0, color='gray', linestyle=':', dashes=(1,5))
ax1.grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))

for s in ["top", "right", "left"]:
    ax0.spines[s].set_visible(False)
    ax1.spines[s].set_visible(False)


# [back to top](#table-of-contents)
# <a id="catboost_result"></a>
# ## 5.6 CatBoost

# In[43]:


get_ipython().run_cell_magic('time', '', 'cb_reg = CatBoostRegressor(random_state=42, verbose=False)\ny_predict = cross_val_predict(cb_reg, X_train, y_train, cv=cv, n_jobs=-1)\n')


# In[44]:


fig = plt.figure(figsize=(10, 4), facecolor='#f6f6f6')
gs = fig.add_gridspec(1, 2)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
gs.update(wspace=0.2, hspace=0)
background_color = "#f6f6f6"

sns.scatterplot(y_train, y_train, ax=ax0, color='#ff819a', linewidth=0.3, edgecolor='#5a0012', zorder=3)
sns.scatterplot(y_predict, y_train, ax=ax0, color='#ffd514', linewidth=0.3, edgecolor='#4f4100', zorder=3)

sns.kdeplot(y_train, ax=ax1, color='#ff819a', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)
sns.kdeplot(y_predict, ax=ax1, color='#ffd514', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)

ax0.text(-1.5, 12.5, 'CatBoost Results', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-1.5, 11.5, 'Most of the prediction concentrate in range of 7 to 9.5 resemble the result of LGBM', 
         fontsize=13, fontweight='light', fontfamily='serif')

ax0.set_ylabel('target')
ax0.set_xlabel('prediction')
ax0.set_facecolor(background_color)
ax1.set_facecolor(background_color)
ax0.grid(which='major', zorder=0, color='gray', linestyle=':', dashes=(1,5))
ax1.grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))

for s in ["top", "right", "left"]:
    ax0.spines[s].set_visible(False)
    ax1.spines[s].set_visible(False)


# [back to top](#table-of-contents)
# <a id="adaboost_result"></a>
# ## 5.7 AdaBoost

# In[45]:


get_ipython().run_cell_magic('time', '', 'ab_reg = AdaBoostRegressor(random_state=42)\ny_predict = cross_val_predict(ab_reg, X_train, y_train, cv=cv, n_jobs=-1)\n')


# In[46]:


fig = plt.figure(figsize=(10, 4), facecolor='#f6f6f6')
gs = fig.add_gridspec(1, 2)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
gs.update(wspace=0.2, hspace=0)
background_color = "#f6f6f6"

sns.scatterplot(y_train, y_train, ax=ax0, color='#ff819a', linewidth=0.3, edgecolor='#5a0012', zorder=3)
sns.scatterplot(y_predict, y_train, ax=ax0, color='#ffd514', linewidth=0.3, edgecolor='#4f4100', zorder=3)

sns.kdeplot(y_train, ax=ax1, color='#ff819a', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)
sns.kdeplot(y_predict, ax=ax1, color='#ffd514', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)

ax0.text(-1.5, 12.5, 'AdaBoost Results', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-1.5, 11.5, 'Most of the prediction concentrate in a range of 7.3 to 8.7', 
         fontsize=13, fontweight='light', fontfamily='serif')

ax0.set_ylabel('target')
ax0.set_xlabel('prediction')
ax0.set_facecolor(background_color)
ax1.set_facecolor(background_color)
ax0.grid(which='major', zorder=0, color='gray', linestyle=':', dashes=(1,5))
ax1.grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))

for s in ["top", "right", "left"]:
    ax0.spines[s].set_visible(False)
    ax1.spines[s].set_visible(False)


# [back to top](#table-of-contents)
# <a id="hyperparameters"></a>
# # 6. Optuna Hyperparameters Tuning
# This section purpose is to demonstrate the hyperparameters tuning using Optuna on LGBM, XGBoost and CatBoost. The objective is to optimized average RMSE from 5 CVs and to speed up the process the number of trials is set to 1 which is not ideal. 

# [back to top](#table-of-contents)
# <a id="lgbm_hpt"></a>
# ## 6.1 LGBM

# In[47]:


def objective(trial):    
    params = {
            'random_state': 42,
            'max_depth': trial.suggest_int('max_depth', 1, 14),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0)
            }
    
    lgbm_reg = LGBMRegressor()
    lgbm_reg.set_params(**params)
    scores = cross_val_score(lgbm_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    rmse = np.sqrt(-scores)
    return np.mean(rmse)


# In[48]:


study = optuna.create_study(direction = 'minimize')
study.optimize(objective, n_trials = 1)
best_params = study.best_trial.params


# In[49]:


best_params


# In[50]:


get_ipython().run_cell_magic('time', '', "lgbm_reg = LGBMRegressor()\nlgbm_reg.set_params(**best_params)\nscores = cross_val_score(lgbm_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)\nlgbm_rmse_scores = np.sqrt(-scores)\nprint('LGBM performance:', lgbm_rmse_scores)\n")


# [back to top](#table-of-contents)
# <a id="xgboost_hpt"></a>
# ## 6.2 XGBoost

# In[51]:


def objective(trial):    
    params = {
            'random_state': 42,
            'max_depth': trial.suggest_int('max_depth', 1, 14),
            'eta': trial.suggest_float('eta', 0.01, 1.0),
            }
    xgb_reg = XGBRegressor()
    xgb_reg.set_params(**params)
    scores = cross_val_score(xgb_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    rmse = np.sqrt(-scores)
    return np.mean(rmse)


# In[52]:


study = optuna.create_study(direction = 'minimize')
study.optimize(objective, n_trials = 1)
best_params = study.best_trial.params


# In[53]:


best_params


# In[54]:


get_ipython().run_cell_magic('time', '', "xgb_reg = XGBRegressor()\nxgb_reg.set_params(**best_params)\nscores = cross_val_score(xgb_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)\nxgb_rmse_scores = np.sqrt(-scores)\nprint('XGBoost performance:', xgb_rmse_scores)\n")


# [back to top](#table-of-contents)
# <a id="target_splitting"></a>
# # 7. Target Splitting
# Based on the distribution of the target, splitting the target into two distribution can be done. The distribution can be divided at 8, and a new categorical feature can be created in this case **below 8** feature. The performance of the feature is great, but reproducing the feature on the test will be a challenge. One way is to create a classification problem that predict the **below 8** feature.
# 
# <a id="below_feature"></a>
# ## 7.1 Below 8 Feature
# Producing the feature using the Train set is easy but producing it with the test set will be a challenge.

# In[55]:


X_train['below8'] = np.where(y_train < 8, 1, 0)


# [back to top](#table-of-contents)
# <a id="below_result"></a>
# ## 7.2 Below 8 Result
# 
# Using the catboost without any hyperparameter tuning, RMSE can reach to around 0.37 to 0.38.

# In[56]:


get_ipython().run_cell_magic('time', '', "cb_reg = CatBoostRegressor(random_state=42, verbose=False)\nscores = cross_val_score(cb_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=5)\ncb_rmse_scores = np.sqrt(-scores)\nprint('CatBoost performance:', cb_rmse_scores)\n")


# [back to top](#table-of-contents)
# <a id="winners_solutions"></a>
# # 8. Winners Solutions
# Congratulations for all the winners and thank you for sharing your solution. Below are the winners and their solutions:
# * 1st place position: [danzel](https://www.kaggle.com/springmanndaniel) - [1st place solution](https://www.kaggle.com/c/tabular-playground-series-jan-2021/discussion/216037)
# * 2nd place position: [Ren](https://www.kaggle.com/ryanzhang) - [2nd solution write up.](https://www.kaggle.com/c/tabular-playground-series-jan-2021/discussion/216070)
# * 3rd place position: [Fatih](https://www.kaggle.com/fatihozturk) - [3rd place solution](https://www.kaggle.com/c/tabular-playground-series-jan-2021/discussion/216087)
