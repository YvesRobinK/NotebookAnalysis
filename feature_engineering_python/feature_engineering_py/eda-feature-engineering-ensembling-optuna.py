#!/usr/bin/env python
# coding: utf-8

# # Playground S3E9: Predicting Concrete Strength 🏗️👷🏼🔨
# <hr style="width:80%; float:left;background-color:#d8576b; height:2px">

# ### Table of contents
# 1. [Introduction](#1.-Introduction)
# 2. [Load libraries](#2.-Load-libraries)
# 3. [Data set](#3.-Data-set)
#    - [Load data set](#Load-data-set)
#    - [Data set description](#Data-set-description)
#    - [Data statistics](#Data-statistics)
#    - [Define target variable](#Define-target-variable)
#    - [Check for missing data](#Check-for-missing-data)
# 4. [Exploratory Data Analysis](#4.-Exploratory-Data-Analysis)
#    - [Concrete strength distribution](#Concrete-strength-distribution)
#    - [Continuous feature distribution](#Continuous-feature-distribution)
#    - [Feature correlation](#Feature-correlation)
#    - [Feature engineering](#Feature-engineering)
# 5. [Model Training](#5.-Model-Training)
#    - [Ignoring duplicate rows in training data](#5.1-Ignoring-duplicate-rows-in-training-data)
#        - [XGBoost Regression](#5.1.1-XGBoost-Regression)
#        - [Random Forest Regression](#5.1.2-Random-Forest-Regression)
#    - [Replacing targets in duplicate rows with mean target value](#5.2-Replacing-targets-in-duplicate-rows-with-mean-target-value)
#        - [XGBoost Regression](#5.2.1-XGBoost-Regression)
#        - [Random Forest Regression](#5.2.2-Random-Forest-Regression)
#    - [Ensembling](#Ensembling)
# 6. [Submission](#6.-Submission)

# # 1. Introduction

# <div style='font-size:15px'>
# The objective of this competition is to predict the strength of concrete. In this notebook, we cover the following:
# <ul>
#     <li>Exploratory data analysis</li>
#     <li>Feature engineering</li>
#     <li>Model tuning with Optuna</li>
#     </ul>
# </div>

# In[1]:


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   END = '\033[0m'


# # 2. Load libraries

# In[2]:


import warnings
warnings.simplefilter('ignore')

import os
import datetime

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)

import plotly
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, OneHotEncoder, MinMaxScaler, OrdinalEncoder

from xgboost import XGBRegressor

import optuna

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
plotly.offline.init_notebook_mode()


# In[3]:


pio.templates.default = 'plotly_white'


# In[4]:


palette = px.colors.sequential.Plasma
print(f"{color.BOLD}Color palette:{color.END}\n")
sb.color_palette(palette)


# # 3. Data set

# ## Load data set

# In[5]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[6]:


train_df = pd.read_csv('/kaggle/input/playground-series-s3e9/train.csv', index_col=0)
test_df = pd.read_csv('/kaggle/input/playground-series-s3e9/test.csv', index_col=0)
original_df = pd.read_csv('/kaggle/input/predict-concrete-strength/ConcreteStrengthData.csv')


# In[7]:


train_df.shape, test_df.shape, original_df.shape


# ## Data set description

# To make concrete, a mixture of portland cement (10-15%) and water (15-20%) make a paste. This paste is then mixed with aggregates (65-75%) such as sand and gravel, or crushed stone. As the cement and water mix, they harden and bind the aggregates into an impenetrable rock-like mass.

# <figure>
# <center><img src='https://www.cement.org/images/default-source/default-album/pca_vol_of_ingredients_final_outlined.png' width=900px/>
#     <figcaption style="font-size:20px">Source: <a href="https://www.cement.org/images/default-source/default-album/pca_vol_of_ingredients_final_outlined.png">https://www.cement.org/</a></figcaption>
#     </center>
#  </figure>

# There are various factors that affect the strength of concrete, such as materials used, age, etc. Our goal is to predict the strength of the concrete based on the components of mixture and other factors as predictors. The data set we use for this is the [Concrete Strength Prediction](https://www.kaggle.com/datasets/mchilamwar/predict-concrete-strength) data set.

# Here is a brief description of each of the features in our data set, and how each of them relates to strength of the concrete:
# <ul>
#     <li><b>CementComponent: </b>Cement, as it is commonly known, is a mixture of compounds made by burning limestone and clay together at very high temperatures. Arguably the most important concrete property is compressive strength since compressive strength affects so many properties of concrete. Compressive strength often increases as cement content increases. While this may seem positive, past research has shown links between increased concrete strength and increased drying shrinkage and cracking density.</li> 
#     <li><b>WaterComponent: </b>Water is the key ingredient, which when mixed with cement, forms a paste that binds the aggregate together. The water causes the hardening of concrete through a process called hydration. Too much water reduces concrete strength, while too little will make the concrete unworkable. Thus, a careful balance of the cement-to-water ratio is required when making concrete. <b>The concrete strength is inversely proportional to the w/c (water-cement ratio). </b></li>
#     <li><b>BlastFurnaceComponent: </b>Blast furnace slag cement is the mixture of ordinary Portland cement and fine granulated blast furnace slag obtained as a by product in the manufacture of steel with percent under 70% to that of cement. The initial strength achieved is lesser than that of conventional concrete, but the higher ultimate strength gained is equal and sometimes higher than conventional concrete.</li>
#     <li><b>FlyAshComponent: </b>Fly ash use in concrete improves the workability of plastic concrete, and the strength and durability of hardened concrete. Fly ash use is also cost effective. When fly ash is added to concrete, the amount of portland cement may be reduced.</li>
#     <li><b>SuperplasticizerComponent: </b>Plasticizers reduce the amount of water required to ensure the complete hydration of cement and proper workability of concrete. Hence, strength increases.</li>
#     <li><b>CoarseAggregateComponent and FineAggregateComponent: </b>Aggregate is the solid particles that are bound together by the cement paste to create the synthetic rock known as concrete. Aggregates can be fine, such as sand, or coarse, such as gravel. The relative amounts of each type and the sizes of each type of aggregate determine the physical properties of the concrete.</li>
#     <li><b>AgeInDays: </b>Concrete is usually dry enough after 24 to 48 hours to walk on. For concrete to dry and reach its full strength, it typically takes about 28 days per inch of slab thickness.</li>
#     <li><b>Strength: </b>The final strength of concrete.</li>
# </ul>
# 

# ## Data statistics

# In[8]:


train_df.describe()


# In[9]:


test_df.describe()


# In[10]:


original_df.describe()


# ## Define target variable

# In[11]:


target = 'Strength'


# ## Check for missing data

# In[12]:


print(f"{color.BOLD}No null values! :){color.BOLD}\n")
pd.DataFrame(data={'Train': train_df.isna().sum(), 'Test': test_df.isna().sum(), 'Original': original_df.isna().sum()}).sort_index()


# In[13]:


train_df.dtypes


# In[14]:


original_df.dtypes


# In[15]:


original_df.head()


# ## Combine generated data and original data

# In[16]:


train_df['is_generated'] = 1
test_df['is_generated'] = 1
original_df['is_generated'] = 0


# In[17]:


original_df.columns


# In[18]:


original_df = original_df.rename(columns={'CementComponent ': 'CementComponent'})


# In[19]:


train_df = pd.concat([train_df, original_df], ignore_index=True)


# In[20]:


train_df.head()


# # 4. Exploratory Data Analysis

# ## Concrete strength distribution

# In[21]:


fig = px.histogram(train_df, x=target, title='Distribution of concrete strength in train data set', color_discrete_sequence=[palette[3]], nbins=100)
fig.update_layout(title=dict(x=0.5, xanchor='center', font_size=20))
fig.show()


# In[22]:


original_features = train_df.drop(columns=[target, 'is_generated']).columns.tolist()
print(original_features)


# In[23]:


continuous_features = original_features.copy()
categorical_features = ['is_generated']


# In[24]:


df = pd.concat([train_df, test_df])
df['is_train'] = False
df.iloc[:len(train_df)]['is_train'] = True


# ## Continuous feature distribution

# In[25]:


for c in continuous_features:
    fig = px.box(df, x='is_train', y=c, color='is_train', color_discrete_sequence=[palette[0], palette[6]])
    fig.update_layout(showlegend=False)
    fig.show()


# In[26]:


fig = px.scatter_matrix(train_df, dimensions=continuous_features, color=target, color_discrete_sequence=palette)
fig.update_layout(height=1600)
fig.show()


# ## Feature correlation

# In[27]:


corr_matrix = train_df[continuous_features+[target]].corr()


# In[28]:


fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale=palette, height=500)
fig.update_traces(hovertemplate="x: %{x} <br> y: %{y} <br> Correlation: %{z:.3f}", name="", showlegend=False, texttemplate="%{z:.3f}")
fig.show()


# ## Feature engineering

# In[29]:


df['WaterToCementRatio'] = df['WaterComponent'] / (df['CementComponent']+1e-6)

# Based on discussion post: https://www.kaggle.com/competitions/playground-series-s3e9/discussion/390973
df['CementToFineAggregateRatio'] = df['CementComponent'] / (df['FineAggregateComponent']+1e-6)
df['FineToCoarseAggregateRatio'] = df['FineAggregateComponent'] / (df['CoarseAggregateComponent']+1e-6)
df['CementToFineToCoarseRatio'] = df['CementToFineAggregateRatio'] / (df['FineToCoarseAggregateRatio']+1e-6)

df['CementToTotalRatio'] = df['CementComponent'] / (df['CementComponent']+df['FineAggregateComponent']+df['CoarseAggregateComponent']+df['FlyAshComponent']+df['BlastFurnaceSlag']+df['WaterComponent']+df['SuperplasticizerComponent']+1e-6)
df['SubstituteRatio'] = (df['FlyAshComponent']+df['BlastFurnaceSlag'])/(df['CementComponent']+1e-6)
df['SuperplasticizerToWaterRatio'] = df['SuperplasticizerComponent']/ (df['WaterComponent']+1e-6)


# In[30]:


new_features = ['WaterToCementRatio', 'CementToFineAggregateRatio', 'FineToCoarseAggregateRatio', 'CementToFineToCoarseRatio', 'CementToTotalRatio', 'SubstituteRatio', 'SuperplasticizerToWaterRatio']


# In[31]:


train_df = df.loc[df['is_train']].drop(columns='is_train')
test_df = df.loc[~df['is_train']].drop(columns='is_train')


# In[32]:


corr_matrix_new = train_df[new_features+[target]].corr()


# In[33]:


fig = px.imshow(corr_matrix_new, text_auto=True, color_continuous_scale=palette, height=500)
fig.update_traces(hovertemplate="x: %{x} <br> y: %{y} <br> Correlation: %{z:.3f}", name="", showlegend=False, texttemplate="%{z:.3f}")
fig.show()


# In[34]:


feature_evaluator = XGBRegressor(tree_method='gpu_hist')
feature_evaluator.fit(train_df.drop(columns=target), train_df[target])
print(pd.Series(feature_evaluator.feature_importances_, index=feature_evaluator.feature_names_in_).sort_values(ascending=False))


# # 5. Model Training

# ## 5.1 Ignoring duplicate rows in training data

# In[35]:


features_to_drop = ['is_generated']


# In[36]:


X_train, Y_train = train_df.drop(columns=[target]+features_to_drop), train_df[target]
X_test = test_df.copy().drop(columns=[target]+features_to_drop)


# ### Continuous feature scaling

# In[37]:


standard_scaler = StandardScaler()

X_train.loc[:] = standard_scaler.fit_transform(X_train)
X_test.loc[:] = standard_scaler.transform(X_test)


# In[38]:


print(f"{color.BOLD}Final data shapes: {color.END}")
print(X_train.shape, Y_train.shape, X_test.shape)


# ### 5.1.1 XGBoost Regression

# ### Define Optuna hyper-parameter tuner and hyper-parameter ranges

# In[39]:


def objective_xgb(trial):
    params = {
        'objective': 'reg:squarederror',
        'tree_method': trial.suggest_categorical(
            'tree_method', ['gpu_hist', 'exact']
        ),
        'reg_lambda': trial.suggest_float(
            'reg_lambda', 1e-3, 1e2, log=True
        ),
        'colsample_bytree': trial.suggest_float(
            'colsample_bytree', 0.5, 1.0, step=0.1
        ),
        'colsample_bylevel': trial.suggest_float(
            'colsample_bylevel', 0.5, 1.0, step=0.1
        ),
        'subsample': trial.suggest_float(
            'subsample', 0.5, 1.0, step=0.1
        ),
        'learning_rate': trial.suggest_float(
            'learning_rate', 1e-2, 1e0, log=True
        ),
        'n_estimators': trial.suggest_int(
            'n_estimators', 50, 300, step=10
        ),
        'max_depth': trial.suggest_int(
            'max_depth', 4, 20, step=2
        ),
        'grow_policy': trial.suggest_categorical(
            'grow_policy', ['depthwise', 'lossguide']
        )
    }
    kf = KFold(n_splits=5,random_state=42,shuffle=True)
    val_split_rmse = []
    for train_idx, val_idx in kf.split(X_train):
        X_train_split, X_val_split = X_train.iloc[train_idx], X_train.iloc[val_idx]
        Y_train_split, Y_val_split = Y_train.iloc[train_idx], Y_train.iloc[val_idx]
        estimator = XGBRegressor(**params)
        estimator.fit(X_train_split, Y_train_split, eval_set=[(X_val_split, Y_val_split)], early_stopping_rounds=3, verbose=0)
        Y_pred_val = pd.Series(estimator.predict(X_val_split), index=X_val_split.index)
        rmse = np.sqrt(mean_squared_error(Y_val_split, Y_pred_val))
        val_split_rmse.append(rmse)
        
    val_rmse = np.mean(val_split_rmse)
    return val_rmse


# ### Run tuning

# In[40]:


study = optuna.create_study(direction='minimize')
study.optimize(objective_xgb, n_trials=100, show_progress_bar=True)
print('Number of finished trials:', len(study.trials))


# ### Tuning results

# In[41]:


optuna.importance.get_param_importances(study)


# In[42]:


study.trials_dataframe().sort_values(by='value').head()


# ### Best hyper-parameters

# In[43]:


best_params = study.best_trial.params
best_params


# ### Train final model using best hyper-parameters

# In[44]:


Y_pred_test_xgb = []
val_split_rmse = []
feature_importances = []

n_repeats = 5
n_splits = 5


# In[45]:


kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_split, X_val_split = X_train.iloc[train_idx], X_train.iloc[val_idx]
    Y_train_split, Y_val_split = Y_train.iloc[train_idx], Y_train.iloc[val_idx]
    estimator = XGBRegressor(eval_metric='rmse', **best_params)
    estimator.fit(X_train_split, Y_train_split, eval_set=[(X_val_split, Y_val_split)], early_stopping_rounds=3, verbose=0)
    Y_pred_val = pd.Series(estimator.predict(X_val_split), index=X_val_split.index)
    rmse = np.sqrt(mean_squared_error(Y_val_split, Y_pred_val))
    val_split_rmse.append(rmse)
    Y_pred_test_xgb.append(estimator.predict(X_test))
    feature_importances.append(estimator.feature_importances_)
    print(f"Validation set RMSE for fold {i}: {rmse:.4f}")
    
feature_importances = np.mean(feature_importances, axis=0)
val_rmse = np.mean(val_split_rmse)
Y_pred_test_xgb_1 = np.mean(Y_pred_test_xgb, axis=0)


# ### Evaluation on Validation set

# In[46]:


print(f"{color.BOLD}RMSE for validation set for final XGBoost model: {val_rmse:.3f}{color.BOLD}")


# ### Feature importance

# In[47]:


feature_importances = pd.Series(data=feature_importances, index=X_train.columns.tolist()).sort_values(ascending=False)

fig = px.bar(feature_importances.reset_index(), y='index', x=0, height=800, color=0, text_auto=True)
fig.update_layout(yaxis_automargin=True, xaxis_title='Feature importance', yaxis_title='Feature', coloraxis_showscale=False)
fig.update_traces(textposition='outside', texttemplate="%{x:.3f}")
fig.show()


# ### 5.1.2 Random Forest Regression

# ### Define Optuna hyper-parameter tuner and hyper-parameter ranges

# In[48]:


def objective_rf(trial):
    params = {
        'verbose': 0,
        'max_samples': trial.suggest_float(
            'max_samples', 0.5, 1.0, step=0.1
        ),
        'max_features': trial.suggest_float(
            'max_features', 0.5, 1.0, step=0.1
        ),
        'n_estimators': trial.suggest_int(
            'n_estimators', 50, 200, step=10
        ),
        'max_depth': trial.suggest_int(
            'max_depth', 4, 20, step=2
        ),
        'min_samples_split': trial.suggest_int(
            'min_samples_split', 2, 20, step=2
        ),
        'min_samples_leaf': trial.suggest_int(
            'min_samples_leaf', 2, 20, step=2
        ),
    }
    kf = KFold(n_splits=5,random_state=42,shuffle=True)
    val_split_rmse = []
    for train_idx, val_idx in kf.split(X_train):
        X_train_split, X_val_split = X_train.iloc[train_idx], X_train.iloc[val_idx]
        Y_train_split, Y_val_split = Y_train.iloc[train_idx], Y_train.iloc[val_idx]
        estimator = RandomForestRegressor(**params)
        estimator.fit(X_train_split, Y_train_split, )
        Y_pred_val = pd.Series(estimator.predict(X_val_split), index=X_val_split.index)
        rmse = np.sqrt(mean_squared_error(Y_val_split, Y_pred_val))
        val_split_rmse.append(rmse)
        
    val_rmse = np.mean(val_split_rmse)
    return val_rmse


# ### Run tuning

# In[49]:


study = optuna.create_study(direction='minimize')
study.optimize(objective_rf, n_trials=100, show_progress_bar=True)
print('Number of finished trials:', len(study.trials))


# ### Tuning results

# In[50]:


optuna.importance.get_param_importances(study)


# In[51]:


study.trials_dataframe().sort_values(by='value').head()


# ### Best hyper-parameters

# In[52]:


best_params = study.best_trial.params
best_params


# ### Train final model using best hyper-parameters

# In[53]:


Y_pred_test_rf = []
val_split_rmse = []
feature_importances = []

n_repeats = 5
n_splits = 5


# In[54]:


kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_split, X_val_split = X_train.iloc[train_idx], X_train.iloc[val_idx]
    Y_train_split, Y_val_split = Y_train.iloc[train_idx], Y_train.iloc[val_idx]
    estimator = RandomForestRegressor(**best_params)
    estimator.fit(X_train_split, Y_train_split)
    Y_pred_val = pd.Series(estimator.predict(X_val_split), index=X_val_split.index)
    rmse = np.sqrt(mean_squared_error(Y_val_split, Y_pred_val))
    val_split_rmse.append(rmse)
    Y_pred_test_rf.append(estimator.predict(X_test))
    feature_importances.append(estimator.feature_importances_)
    print(f"Validation set RMSE for fold {i}: {rmse:.4f}")
    
feature_importances = np.mean(feature_importances, axis=0)
val_rmse = np.mean(val_split_rmse)
Y_pred_test_rf_1 = np.mean(Y_pred_test_rf, axis=0)


# ### Evaluation on Validation set

# In[55]:


print(f"{color.BOLD}RMSE for validation set for final RF model: {val_rmse:.3f}{color.BOLD}")


# ### Feature importance

# In[56]:


feature_importances = pd.Series(data=feature_importances, index=X_train.columns.tolist()).sort_values(ascending=False)

fig = px.bar(feature_importances.reset_index(), y='index', x=0, height=800, color=0, text_auto=True)
fig.update_layout(yaxis_automargin=True, xaxis_title='Feature importance', yaxis_title='Feature', coloraxis_showscale=False)
fig.update_traces(textposition='outside', texttemplate="%{x:.3f}")
fig.show()


# ## 5.2 Replacing targets in duplicate rows with mean target value

# In[57]:


features_to_drop = ['is_generated']


# In[58]:


train_df_dup = train_df.copy()
train_df_dup[target] = train_df_dup.groupby(original_features)[target].transform(lambda x: x.median())


# In[59]:


train_df_dup = train_df_dup.drop_duplicates(subset=original_features)  # After our transform above, if original features are equal for two rows, target variable is also equal


# In[60]:


X_train, Y_train = train_df_dup.drop(columns=[target]+features_to_drop), train_df_dup[target]
X_test = test_df.copy().drop(columns=[target]+features_to_drop)


# ### Continuous feature scaling

# In[61]:


standard_scaler = StandardScaler()

X_train.loc[:] = standard_scaler.fit_transform(X_train)
X_test.loc[:] = standard_scaler.transform(X_test)


# In[62]:


print(f"{color.BOLD}Final data shapes: {color.END}")
print(X_train.shape, Y_train.shape, X_test.shape)


# ### 5.2.1 XGBoost Regression

# ### Define Optuna hyper-parameter tuner and hyper-parameter ranges

# In[63]:


def objective_xgb(trial):
    params = {
        'objective': 'reg:squarederror',
        'tree_method':trial.suggest_categorical(
            'tree_method', ['gpu_hist', 'exact']
        ),
        'reg_lambda': trial.suggest_float(
            'reg_lambda', 1e-3, 1e2, log=True
        ),
        'colsample_bytree': trial.suggest_float(
            'colsample_bytree', 0.5, 1.0, step=0.1
        ),
        'colsample_bylevel': trial.suggest_float(
            'colsample_bylevel', 0.5, 1.0, step=0.1
        ),
        'subsample': trial.suggest_float(
            'subsample', 0.5, 1.0, step=0.1
        ),
        'learning_rate': trial.suggest_float(
            'learning_rate', 1e-2, 1e0, log=True
        ),
        'n_estimators': trial.suggest_int(
            'n_estimators', 50, 300, step=10
        ),
        'max_depth': trial.suggest_int(
            'max_depth', 4, 20, step=2
        ),
        'grow_policy': trial.suggest_categorical(
            'grow_policy', ['depthwise', 'lossguide']
        )
    }
    kf = KFold(n_splits=5,random_state=42,shuffle=True)
    val_split_rmse = []
    for train_idx, val_idx in kf.split(X_train):
        X_train_split, X_val_split = X_train.iloc[train_idx], X_train.iloc[val_idx]
        Y_train_split, Y_val_split = Y_train.iloc[train_idx], Y_train.iloc[val_idx]
        estimator = XGBRegressor(**params)
        estimator.fit(X_train_split, Y_train_split, eval_set=[(X_val_split, Y_val_split)], early_stopping_rounds=3, verbose=0)
        Y_pred_val = pd.Series(estimator.predict(X_val_split), index=X_val_split.index)
        rmse = np.sqrt(mean_squared_error(Y_val_split, Y_pred_val))
        val_split_rmse.append(rmse)
        
    val_rmse = np.mean(val_split_rmse)
    return val_rmse


# ### Run tuning

# In[64]:


study = optuna.create_study(direction='minimize')
study.optimize(objective_xgb, n_trials=100, show_progress_bar=True)
print('Number of finished trials:', len(study.trials))


# ### Tuning results

# In[65]:


optuna.importance.get_param_importances(study)


# In[66]:


study.trials_dataframe().sort_values(by='value').head()


# ### Best hyper-parameters

# In[67]:


best_params = study.best_trial.params
best_params


# ### Train final model using best hyper-parameters

# In[68]:


Y_pred_test_xgb = []
val_split_rmse = []
feature_importances = []

n_repeats = 5
n_splits = 5


# In[69]:


kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_split, X_val_split = X_train.iloc[train_idx], X_train.iloc[val_idx]
    Y_train_split, Y_val_split = Y_train.iloc[train_idx], Y_train.iloc[val_idx]
    estimator = XGBRegressor(eval_metric='rmse', **best_params)
    estimator.fit(X_train_split, Y_train_split, eval_set=[(X_val_split, Y_val_split)], early_stopping_rounds=3, verbose=0)
    Y_pred_val = pd.Series(estimator.predict(X_val_split), index=X_val_split.index)
    rmse = np.sqrt(mean_squared_error(Y_val_split, Y_pred_val))
    val_split_rmse.append(rmse)
    Y_pred_test_xgb.append(estimator.predict(X_test))
    feature_importances.append(estimator.feature_importances_)
    print(f"Validation set RMSE for fold {i}: {rmse:.4f}")
    
feature_importances = np.mean(feature_importances, axis=0)
val_rmse = np.mean(val_split_rmse)
Y_pred_test_xgb_2 = np.mean(Y_pred_test_xgb, axis=0)


# ### Evaluation on Validation set

# In[70]:


print(f"{color.BOLD}RMSE for validation set for final XGBoost model: {val_rmse:.3f}{color.BOLD}")


# ### Feature importance

# In[71]:


feature_importances = pd.Series(data=feature_importances, index=X_train.columns.tolist()).sort_values(ascending=False)

fig = px.bar(feature_importances.reset_index(), y='index', x=0, height=800, color=0, text_auto=True)
fig.update_layout(yaxis_automargin=True, xaxis_title='Feature importance', yaxis_title='Feature', coloraxis_showscale=False)
fig.update_traces(textposition='outside', texttemplate="%{x:.3f}")
fig.show()


# ### 5.2.2 Random Forest Regression

# ### Define Optuna hyper-parameter tuner and hyper-parameter ranges

# In[72]:


def objective_rf(trial):
    params = {
        'verbose': 0,
        'max_samples': trial.suggest_float(
            'max_samples', 0.5, 1.0, step=0.1
        ),
        'max_features': trial.suggest_float(
            'max_features', 0.5, 1.0, step=0.1
        ),
        'n_estimators': trial.suggest_int(
            'n_estimators', 50, 200, step=10
        ),
        'max_depth': trial.suggest_int(
            'max_depth', 4, 20, step=2
        ),
        'min_samples_split': trial.suggest_int(
            'min_samples_split', 2, 20, step=2
        ),
        'min_samples_leaf': trial.suggest_int(
            'min_samples_leaf', 2, 20, step=2
        ),
    }
    kf = KFold(n_splits=5,random_state=42,shuffle=True)
    val_split_rmse = []
    for train_idx, val_idx in kf.split(X_train):
        X_train_split, X_val_split = X_train.iloc[train_idx], X_train.iloc[val_idx]
        Y_train_split, Y_val_split = Y_train.iloc[train_idx], Y_train.iloc[val_idx]
        estimator = RandomForestRegressor(**params)
        estimator.fit(X_train_split, Y_train_split, )
        Y_pred_val = pd.Series(estimator.predict(X_val_split), index=X_val_split.index)
        rmse = np.sqrt(mean_squared_error(Y_val_split, Y_pred_val))
        val_split_rmse.append(rmse)
        
    val_rmse = np.mean(val_split_rmse)
    return val_rmse


# ### Run tuning

# In[73]:


study = optuna.create_study(direction='minimize')
study.optimize(objective_rf, n_trials=100, show_progress_bar=True)
print('Number of finished trials:', len(study.trials))


# ### Tuning results

# In[74]:


optuna.importance.get_param_importances(study)


# In[75]:


study.trials_dataframe().sort_values(by='value').head()


# ### Best hyper-parameters

# In[76]:


best_params = study.best_trial.params
best_params


# ### Train final model using best hyper-parameters

# In[77]:


Y_pred_test_rf = []
val_split_rmse = []
feature_importances = []

n_repeats = 5
n_splits = 5


# In[78]:


kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_split, X_val_split = X_train.iloc[train_idx], X_train.iloc[val_idx]
    Y_train_split, Y_val_split = Y_train.iloc[train_idx], Y_train.iloc[val_idx]
    estimator = RandomForestRegressor(**best_params)
    estimator.fit(X_train_split, Y_train_split)
    Y_pred_val = pd.Series(estimator.predict(X_val_split), index=X_val_split.index)
    rmse = np.sqrt(mean_squared_error(Y_val_split, Y_pred_val))
    val_split_rmse.append(rmse)
    Y_pred_test_rf.append(estimator.predict(X_test))
    feature_importances.append(estimator.feature_importances_)
    print(f"Validation set RMSE for fold {i}: {rmse:.4f}")
    
feature_importances = np.mean(feature_importances, axis=0)
val_rmse = np.mean(val_split_rmse)
Y_pred_test_rf_2 = np.mean(Y_pred_test_rf, axis=0)


# ### Evaluation on Validation set

# In[79]:


print(f"{color.BOLD}RMSE for validation set for final RF model: {val_rmse:.3f}{color.BOLD}")


# ### Feature importance

# In[80]:


feature_importances = pd.Series(data=feature_importances, index=X_train.columns.tolist()).sort_values(ascending=False)

fig = px.bar(feature_importances.reset_index(), y='index', x=0, height=800, color=0, text_auto=True)
fig.update_layout(yaxis_automargin=True, xaxis_title='Feature importance', yaxis_title='Feature', coloraxis_showscale=False)
fig.update_traces(textposition='outside', texttemplate="%{x:.3f}")
fig.show()


# # 6. Submission

# In[81]:


Y_pred_test = pd.DataFrame(data={'XGBoost_1': Y_pred_test_xgb_1, 'RF_1': Y_pred_test_rf_1, 'XGBoost_2': Y_pred_test_xgb_2, 'RF_2': Y_pred_test_rf_2}, index=X_test.index)
Y_pred_test[target] = Y_pred_test.mean(axis=1)
Y_pred_test.index.name = 'id'


# In[82]:


Y_pred_test.head()


# In[83]:


submission_df = Y_pred_test[target].reset_index()
submission_df.head()


# In[84]:


fig = px.histogram(submission_df, x=target, color_discrete_sequence=[palette[4]], nbins=100)
fig.show()


# In[85]:


submission_df.to_csv('submission.csv', index=False)

