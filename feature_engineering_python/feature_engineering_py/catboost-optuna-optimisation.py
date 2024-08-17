#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# - This is a starter notebook using catboost and hyperparameter tuning using optuna.
# - For this model I create datetime, holiday and lag features.
# - It is also enriched by covid data.
# - We model the first order differenced and Box-Cox transformed data and revert it back at the end.
# - The model does multi-step forecasting using recursion from the forecasted values.
# - We evaluate the model using a validation and evaulation set for the hyperparameter tuning and early stopping.
# - There is still some further work to be done such as rolling mean etc.
# - I welcome any suggestions!

# ## Packages, Functions and Logging

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from catboost import CatBoostRegressor, Pool, CatBoostError, cv
from catboost.utils import eval_metric

import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

import shap

from scipy.stats import boxcox
from scipy.special import inv_boxcox
from statsmodels.tsa.stattools import adfuller

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from typing import List

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# SMAPE for cross val
def smape(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)


# In[3]:


# Get categorical features
def get_categoricals(df: pd.DataFrame) -> List[str]:
    objs = list(df.head(10).select_dtypes(object).columns)
    cats = list(df.head(10).select_dtypes('category').columns)
    return objs + cats


# ## Data

# In[4]:


# Load in the data
df_train = pd.read_csv('/kaggle/input/godaddy-microbusiness-density-forecasting/train.csv')
df_test = pd.read_csv('/kaggle/input/godaddy-microbusiness-density-forecasting/test.csv')
df_submission = pd.read_csv('/kaggle/input/godaddy-microbusiness-density-forecasting/sample_submission.csv')
df_new_train = pd.read_csv('/kaggle/input/godaddy-microbusiness-density-forecasting/revealed_test.csv')


# In[5]:


# Add new test data
df_train = pd.concat([df_train, df_new_train]).sort_values(by=['cfips','first_day_of_month']).reset_index().drop(columns=['index'])


# In[6]:


# Remove the index from test set
drop_index = (df_test['first_day_of_month'] == '2022-11-01') | (df_test['first_day_of_month'] == '2022-12-01')
df_test = df_test.loc[~drop_index,:]


# ## EDA

# In[7]:


# View data
df_train.head()


# In[8]:


# Column types
df_train.info()


# In[9]:


# Plot the time series for some of the counties 
num_plots = 5
fig = make_subplots(rows=num_plots, cols=1,
                   subplot_titles=(df_train.groupby('cfips').head(1)['cfips'].iloc[:num_plots].to_list()))

for idx, cfip in enumerate(df_train['cfips'].unique()[:num_plots]):
    
    fig.append_trace(go.Scatter(
    x=df_train['first_day_of_month'].loc[df_train['cfips'] == cfip],
    y=df_train['microbusiness_density'].loc[df_train['cfips'] == cfip],
    name=str(df_train['county'].loc[df_train['cfips'] == cfip].tail(1).values[0]) +\
        ', ' + str(df_train['state'].loc[df_train['cfips'] == cfip].tail(1).values[0])    
    ), row=idx+1, col=1)

fig.update_layout(template="simple_white", font=dict(size=18), width=1000, height=1400)
fig.show()


# ## Feature Engineering

# ### Stationarity

# In[10]:


# Make the data stationary through boxcox transform and differencing
df_train['microbusiness_density'].replace({0:0.000000000001}, inplace=True)
df_train['target_boxcox'], lam = boxcox(df_train['microbusiness_density'])
df_train['target_diff'] = df_train.groupby('cfips')['target_boxcox'].diff()
df_train.dropna(subset=['target_diff'], inplace=True)
df_train[df_train['cfips'] == 1001].head()


# In[11]:


# ADF test for stationarity
def adf_test(series):
    """Using an ADF test to determine if the series is stationary"""
    test_results = adfuller(series)
    print('ADF Statistic: ', test_results[0])
    print('P-Value: ', test_results[1])
    print('Critical Values:')
    for thres, adf_stat in test_results[4].items():
        print('\t%s: %.2f' % (thres, adf_stat))
    
for cfip in df_train['cfips'].unique()[0:5]:
    adf_test(df_train['target_diff'].loc[df_train['cfips'] == cfip])


# Definitely stationary!

# In[12]:


# Define target 
TARGET = 'target_diff'


# ### Combine Data

# In[13]:


# Combine dataset to carry out efficient feature engineering
df_train['dataset'] = 'train'
df_test['dataset'] = 'test'
data = pd.concat([df_train, df_test]).sort_values('row_id').reset_index(drop=True)
data['first_day_of_month'] = pd.to_datetime(data['first_day_of_month'])
data['county'] = data.groupby('cfips')['county'].ffill()
data['state'] = data.groupby('cfips')['state'].ffill()
data.tail(10)


# ### Generate Features

# In[14]:


def get_datetime_feature(data):
    data['quarter'] = data['first_day_of_month'].dt.quarter
    data['year'] = data['first_day_of_month'].dt.year
    data['month'] = data['first_day_of_month'].dt.month
    return data


# In[15]:


def get_lags(data, num_lags, target):    
    for lag in range(1, num_lags+1):
        data[f'lag{lag}'] = data.groupby('cfips')[target].shift(lag)
    return data


# In[16]:


# Apply transformations to training set
data = get_datetime_feature(data)
num_lags = 8
data = get_lags(data, num_lags, TARGET)
data.tail(12)


# ### Data Enrichment

# In[17]:


df_enrich = pd.read_csv('/kaggle/input/stats-for-covid19-by-cfips/covidStats.csv')
data = data.merge(df_enrich.drop(columns=['cfips']), how="left", on="row_id")
data


# In[18]:


co_est = pd.read_csv("/kaggle/input/us-indicator/co-est2021-alldata.csv", encoding='latin-1')
co_est["cfips"] = co_est.STATE*1000 + co_est.COUNTY
co_columns = [
    'SUMLEV',
    'DIVISION',
    'ESTIMATESBASE2020',
    'POPESTIMATE2020',
    'POPESTIMATE2021',
    'NPOPCHG2020',
    'NPOPCHG2021',
    'BIRTHS2020',
    'BIRTHS2021',
    'DEATHS2020',
    'DEATHS2021',
    'NATURALCHG2020',
    'NATURALCHG2021',
    'INTERNATIONALMIG2020',
    'INTERNATIONALMIG2021',
    'DOMESTICMIG2020',
    'DOMESTICMIG2021',
    'NETMIG2020',
    'NETMIG2021',
    'RESIDUAL2020',
    'RESIDUAL2021',
    'GQESTIMATESBASE2020',
    'GQESTIMATES2020',
    'GQESTIMATES2021',
    'RBIRTH2021',
    'RDEATH2021',
    'RNATURALCHG2021',
    'RINTERNATIONALMIG2021',
    'RDOMESTICMIG2021',
    'RNETMIG2021'
]
data = data.merge(co_est, on="cfips", how="left")


# ### Outliers

# In[19]:


for cfip in tqdm(data['cfips'].unique()): 
    indices = (data['cfips'] == cfip) 
    tmp = data.loc[indices].copy().reset_index(drop=True)
    var = tmp.microbusiness_density.values.copy()
    for i in range(37, 2, -1):
        thr = 0.10 * np.mean(var[:i]) 
        difa = var[i] - var[i - 1] 
        if (difa >= thr) or (difa <= -thr):              
            if difa > 0:
                var[:i] += difa - 0.003 
            else:
                var[:i] += difa + 0.003  
    var[0] = var[1] * 0.99
    data.loc[indices, 'microbusiness_density'] = var


# In[20]:


data['lastactive'] = data.groupby('cfips')['active'].transform('last')
data


# ## Modelling

# ### Build Model

# In[21]:


# Set features
FEATURES = list(data.drop(columns=['row_id', 'cfips', 'county', 'state', 'active', TARGET, 'first_day_of_month',
                                   'dataset', 'microbusiness_density', 'target_boxcox', 'target_diff']))
data[FEATURES]


# In[22]:


list(data[FEATURES])


# In[23]:


# Initialise where we will store forecasts
df_test[TARGET] = 0


# In[24]:


# Loop to compute catboost for each county
for cfip in tqdm(data['cfips'].unique()):
    
    # Get time series for the county
    train_cfip = data.loc[(data['cfips'] == cfip) & (data['dataset'] == 'train')]

    # Modelling and validation sets
    df_model = train_cfip[0:37]
    df_eval = train_cfip[37:39]
    df_valid = train_cfip.tail(1)
    
    # Function for hyperparameter tuning
    def objective(trial):
        model_pool = Pool(data=df_model[FEATURES], label=df_model[TARGET], cat_features=get_categoricals(df_model[FEATURES]))
        eval_pool = Pool(data=df_eval[FEATURES], label=df_eval[TARGET], cat_features=get_categoricals(df_eval[FEATURES]))
        
        # Define the parameters we will search over
        params = {
        "learning_rate" : trial.suggest_float("learning_rate", 0.01,0.9),
        "colsample_bylevel" : trial.suggest_float("colsample_bylevel", 0.1,1),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "iterations": trial.suggest_int("iterations", 500, 1000),
        "l2_leaf_reg": trial.suggest_discrete_uniform("l2_leaf_reg", 1.0, 5.0, 0.5),
        "min_child_samples": trial.suggest_categorical("min_child_samples", [1, 4, 8, 16, 32]),
        "border_count": trial.suggest_int("border_count", 30, 250),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        }


        # Fit the model and produce errors 
        model = CatBoostRegressor(**params, objective='MAE', eval_metric='SMAPE', grow_policy='SymmetricTree')
        model.fit(model_pool, eval_set=eval_pool, early_stopping_rounds=15, verbose=0)
        error = smape(df_valid[TARGET], model.predict(df_valid[FEATURES]))
        
        return error
    
    # Initiate hyperparameter tuning
    study = optuna.create_study(direction='minimize', sampler=TPESampler())
    study.optimize(objective,n_trials=10)
    hyperparams = study.best_params
    
    # Train final model
    model = CatBoostRegressor(**hyperparams, objective='MAPE', eval_metric='SMAPE')
    train_pool = Pool(data=train_cfip[FEATURES], label=train_cfip[TARGET], feature_names=FEATURES, cat_features=get_categoricals(train_cfip[FEATURES]))
    model.fit(train_pool, verbose=0)
    
    # Get the test set
    test_cfip = data.loc[(data['cfips'] == cfip) & (data['dataset'] == 'test')].reset_index()
    
    # Use recursive forecasting to compute multi-steps
    forecasts_list = []
    for leadtime in range(6):
        forecast = model.predict(test_cfip[FEATURES].loc[leadtime])
        forecasts_list.append(forecast)
        
        for lag in range(1, num_lags+1):
            test_cfip[f'lag{lag}'].loc[leadtime+lag] = forecast
    
    # Save differenced forecasts
    df_test[TARGET].loc[df_test['cfips'] == cfip] = forecasts_list


# ### Convert To Actual Forecasts

# In[25]:


# Initialise where we will store undifferenced forecasts
df_test['target_boxcox'] = 0


# In[26]:


# Add the previous observation to initiating removing differencing
for cfip in tqdm(df_test['cfips'].unique()):
    boxcox_list = []
    test_cfip = df_test.loc[(df_test['cfips'] == cfip)].reset_index()
    
    for leadtime in range(6):
        if leadtime == 0:
            target_boxcox = data['target_boxcox'].loc[(data['cfips'] == cfip) & (data['dataset'] == 'train')].iloc[-1]
        else:
            target_boxcox = boxcox_list[leadtime-1]
            
        undiff_value = test_cfip['target_diff'].loc[leadtime] + target_boxcox
        boxcox_list.append(undiff_value)
        
    # Save un-differenced predictions
    df_test['target_boxcox'].loc[df_test['cfips'] == cfip] = boxcox_list   


# In[27]:


# Convert to actual forecasts with inverse boxcox
df_test['microbusiness_density'] = inv_boxcox(df_test['target_boxcox'], lam) 


# ## Analysis

# In[28]:


# View predictions
df_test


# In[29]:


# Plot the time series for some of the counties 
num_plots = 5
fig = make_subplots(rows=num_plots, cols=1,
                   subplot_titles=(df_train.groupby('cfips').head(1)['cfips'].iloc[:num_plots].to_list()))

for idx, cfip in enumerate(df_train['cfips'].unique()[:num_plots]):
    
    fig.append_trace(go.Scatter(
    x=df_train['first_day_of_month'].loc[df_train['cfips'] == cfip],
    y=df_train['microbusiness_density'].loc[df_train['cfips'] == cfip],    
    name='Train',
    line=dict(color="blue", width=2)), row=idx+1, col=1)
    
    fig.append_trace(go.Scatter(
    x=df_test['first_day_of_month'].loc[df_test['cfips'] == cfip],
    y=df_test['microbusiness_density'].loc[df_test['cfips'] == cfip],    
    name='Forecast',
    line=dict(color="red", width=2)), row=idx+1, col=1)
    
# Removing repeating of names in the legend    
names = set()
fig.for_each_trace(
    lambda trace:
    trace.update(showlegend=False)
    if (trace.name in names) else names.add(trace.name))

fig.update_layout(template="simple_white", font=dict(size=18), width=900, height=1500)
fig.show()


# In[30]:


# Explain model predictions using shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df_model[FEATURES])
shap.summary_plot(shap_values, df_model[FEATURES], plot_type='bar')


# ## Submission

# In[31]:


df_submission = df_submission.drop(columns=['microbusiness_density']).merge(df_test[['row_id', 'microbusiness_density']], on='row_id', how='left')
df_submission['microbusiness_density'] = df_submission['microbusiness_density'].fillna(0)
df_submission.to_csv("submission.csv", index=False)
df_submission

