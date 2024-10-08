#!/usr/bin/env python
# coding: utf-8

# ## Official LightAutoML github repository is [here](https://github.com/sberbank-ai-lab/LightAutoML)
# 
# ## Upvote is the best motivator 👍
# 
# # Step 0.0. LightAutoML installation

# This step can be used if you are working inside Google Colab/Kaggle kernels or want to install LightAutoML on your machine:
# ## Note: here we use our developer version with new logging and model structure description

# In[1]:


get_ipython().system('pip install -U https://github.com/sberbank-ai-lab/LightAutoML/raw/fix/logging/LightAutoML-0.2.16.2-py3-none-any.whl')
get_ipython().system('pip install openpyxl')


# # Step 0.1. Import libraries
# 
# Here we will import the libraries we use in this kernel:
# - Standard python libraries for timing, working with OS etc.
# - Essential python DS libraries like numpy, pandas, scikit-learn and torch (the last we will use in the next cell)
# - LightAutoML modules: presets for AutoML, task and report generation module

# In[2]:


# Standard python libraries
import os
import time

# Essential DS libraries
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error
import matplotlib.pyplot as plt
import torch

# LightAutoML presets, task and report generation
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
from lightautoml.dataset.roles import DatetimeRole
from lightautoml.report.report_deco import ReportDeco


# # Step 0.2. Constants
# 
# Here we setup the constants to use in the kernel:
# - `N_THREADS` - number of vCPUs for LightAutoML model creation
# - `N_FOLDS` - number of folds in LightAutoML inner CV
# - `RANDOM_STATE` - random seed for better reproducibility
# - `TIMEOUT` - limit in seconds for model to train
# - `TARGET_NAME` - target column name in dataset

# In[3]:


N_THREADS = 4
N_FOLDS = 5
RANDOM_STATE = 42
TIMEOUT = 2 * 3600
TARGET_NAME = 'target'


# # Step 0.3. Imported models setup
# 
# For better reproducibility we fix numpy random seed with max number of threads for Torch (which usually try to use all the threads on server):

# In[4]:


np.random.seed(RANDOM_STATE)
torch.set_num_threads(N_THREADS)


# # Step 0.4. Data loading
# Let's check the data we have:

# In[5]:


get_ipython().run_cell_magic('time', '', "\ntrain_data = pd.read_csv('../input/tabular-playground-series-jul-2021/train.csv')\ntrain_data.head()\n")


# In[6]:


train_data.shape


# In[7]:


get_ipython().run_cell_magic('time', '', "\ntest_data = pd.read_csv('../input/tabular-playground-series-jul-2021/test.csv')\ntest_data.head()\n")


# In[8]:


test_data.shape


# In[9]:


get_ipython().run_cell_magic('time', '', "\nsample_sub = pd.read_csv('../input/tabular-playground-series-jul-2021/sample_submission.csv')\nsample_sub.head()\n")


# In[10]:


sample_sub.shape


# ## Don't know what to do with -200? Use pseudolabelling 🧐

# In[11]:


# Pseudolabels from true dataset 
pseudolabels_true = pd.read_excel('/kaggle/input/air-quality-time-series-data-uci/AirQualityUCI.xlsx')
pseudolabels_true = pseudolabels_true.iloc[7110:].reset_index(drop = True)
pseudolabels_true.rename({'CO(GT)': 'target_carbon_monoxide', 'C6H6(GT)': 'target_benzene', 'NOx(GT)': 'target_nitrogen_oxides'}, axis = 1, inplace = True)
pseudolabels_true


# In[12]:


pseudolabels_preds = pd.read_csv('../input/tps-lightautoml-baseline-with-pseudolabels/lightautoml_with_pseudolabelling_kernel_version_14.csv')
pseudolabels_preds


# In[13]:


test_data['target_carbon_monoxide'] = np.where(pseudolabels_true['target_carbon_monoxide'].values >= 0, 
                                               pseudolabels_true['target_carbon_monoxide'].values, 
                                               pseudolabels_preds['target_carbon_monoxide'].values)
test_data['target_benzene'] = np.where(pseudolabels_true['target_benzene'].values >= 0, 
                                       pseudolabels_true['target_benzene'].values, 
                                       pseudolabels_preds['target_benzene'].values)
test_data['target_nitrogen_oxides'] = np.where(pseudolabels_true['target_nitrogen_oxides'].values >= 0, 
                                       pseudolabels_true['target_nitrogen_oxides'].values, 
                                       pseudolabels_preds['target_nitrogen_oxides'].values)
    
test_data


# In[14]:


test_data['target_carbon_monoxide'].value_counts()


# In[15]:


test_data['target_benzene'].value_counts()


# In[16]:


test_data['target_nitrogen_oxides'].value_counts()


# In[17]:


ALL_DF = pd.concat([train_data, test_data]).reset_index(drop = True)
print(ALL_DF.shape)


# In[18]:


# Feature engineering func from Remek Kinas kernel with MLJAR (https://www.kaggle.com/remekkinas/mljar-code-minimal) - do not forget to upvote his kernel
    
import math

def pb_add(X):
    X['day'] = X.date_time.dt.weekday
    is_odd = (X['sensor_4'] < 646) & (X['absolute_humidity'] < 0.238)
    X['is_odd'] = is_odd
    diff = X['date_time'] - min(X['date_time'])
    trend = diff.dt.days
    X['f1s'] = np.sin(trend * 2 * math.pi / (365 * 1)) 
    X['f1c'] = np.cos(trend * 2 * math.pi / (365 * 1))
    X['f2s'] = np.sin(2 * math.pi * trend / (365 * 2)) 
    X['f2c'] = np.cos(2 * math.pi * trend / (365 * 2)) 
    X['f3s'] = np.sin(2 * math.pi * trend / (365 * 3)) 
    X['f3c'] = np.cos(2 * math.pi * trend / (365 * 3)) 
    X['f4s'] = np.sin(2 * math.pi * trend / (365 * 4)) 
    X['f4c'] = np.cos(2 * math.pi * trend / (365 * 4)) 
    X['fh1s'] = np.sin(diff.dt.seconds * 2 * math.pi / ( 3600 * 24 * 1))
    X['fh1c'] = np.cos(diff.dt.seconds * 2 * math.pi / ( 3600 * 24 * 1))
    X['fh2s'] = np.sin(diff.dt.seconds * 2 * math.pi / ( 3600 * 24 * 2))
    X['fh2c'] = np.cos(diff.dt.seconds * 2 * math.pi / ( 3600 * 24 * 2))
    X['fh3s'] = np.sin(diff.dt.seconds * 2 * math.pi / ( 3600 * 24 * 3))
    X['fh3c'] = np.cos(diff.dt.seconds * 2 * math.pi / ( 3600 * 24 * 3))
    
    sensor_features = [
        'deg_C', 
        'relative_humidity', 'absolute_humidity', 
        'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5' ]
    
    lags = [-1, -4, -24, -7 * 24]  
    for sensor_feature in sensor_features:
        this = X[sensor_feature]

        for lag in lags:
            feature = f'{sensor_feature}_{abs(lag)}b'
            this_f = X[sensor_feature].shift(lag)
            X[feature] = (this_f - this).fillna(0)
        # look forwards
        for lag in lags:
            feature = f'{sensor_feature}_{abs(-lag)}f'
            this_f = X[sensor_feature].shift(-lag)
            X[feature] = (this_f - this).fillna(0)
            
    return X

ALL_DF['date_time'] = pd.to_datetime(ALL_DF['date_time'])
ALL_DF["hour"] = ALL_DF["date_time"].dt.hour
ALL_DF["working_hours"] =  ALL_DF["hour"].isin(np.arange(8, 21, 1)).astype("int")
ALL_DF["is_weekend"] = (ALL_DF["date_time"].dt.dayofweek >= 5).astype("int")
ALL_DF['hr'] = ALL_DF.date_time.dt.hour * 60 + ALL_DF.date_time.dt.minute
ALL_DF['satday'] = (ALL_DF.date_time.dt.weekday==5).astype("int")
ALL_DF["SMC"] = (ALL_DF["absolute_humidity"] * 100) / ALL_DF["relative_humidity"]
ALL_DF.drop(columns = 'hour', inplace = True)

pb_add(ALL_DF)

ALL_DF['date_time'] = ALL_DF['date_time'].astype(str)


# ## Important - cell below works only because of the data leak. In real life you can't create features using the future data. Be careful!

# In[19]:


def create_target_feats(df):
    for lag in [1, 4, 24, 7 * 24]:
        for t in ['target_carbon_monoxide', 'target_benzene', 'target_nitrogen_oxides']:
            df['{}_lag_{}'.format(t, lag)] = df[t].shift(lag)
            df['{}_lag_m{}'.format(t, lag)] = df[t].shift(-lag)
            df['diff_{}_{}'.format(t, lag)] = df['{}_lag_m{}'.format(t, lag)] - df['{}_lag_{}'.format(t, lag)]
            df['div_{}_{}'.format(t, lag)] = df['{}_lag_m{}'.format(t, lag)] / df['{}_lag_{}'.format(t, lag)]
create_target_feats(ALL_DF)


# In[20]:


ALL_DF


# In[21]:


train_data, test_data = ALL_DF.iloc[:(len(ALL_DF) - len(test_data)), :], ALL_DF.iloc[(len(ALL_DF) - len(test_data)):, :]
print(train_data.shape, test_data.shape)


# In[22]:


train_data.head()


# In[23]:


test_data.head()


# # =============== LightAutoML model building ===============
# 
# 
# # Step 1. Task setup
# 
# On the cell below we create Task object - the class to setup what task LightAutoML model should solve with specific loss and metric if necessary (more info can be found [here](https://lightautoml.readthedocs.io/en/latest/generated/lightautoml.tasks.base.Task.html#lightautoml.tasks.base.Task) in our documentation):

# In[24]:


get_ipython().run_cell_magic('time', '', "\ndef rmsle_metric(y_true, y_pred, sample_weight, **kwargs):\n    mask = (sample_weight > 1)\n    return mean_squared_log_error(y_true[mask], np.clip(y_pred[mask], 0, None), **kwargs) ** 0.5\n\ntask = Task('reg', loss = 'rmsle', metric = rmsle_metric, greater_is_better=False)\n")


# # Step 2. Feature roles setup

# To solve the task, we need to setup columns roles. The **only role you must setup is target role**, everything else (drop, numeric, categorical, group, weights etc.) is up to user - LightAutoML models have automatic columns typization inside:

# In[25]:


get_ipython().run_line_magic('pinfo', 'DatetimeRole')


# ### Checking BIZEN idea from comments - no drop for any target, another targets using as features

# In[26]:


get_ipython().run_cell_magic('time', '', "\ntargets_and_drop = {\n    'target_carbon_monoxide': [],\n    'target_benzene': [],\n    'target_nitrogen_oxides': []\n}\n\nroles = {\n    # delete day of month from features\n    DatetimeRole(base_date=False, base_feats=True, seasonality=('d', 'wd', 'hour')): 'date_time'\n}\n")


# # Step 3. LightAutoML model creation - TabularAutoML preset

# In next the cell we are going to create LightAutoML model with `TabularAutoML` class - preset with default model structure like in the image below:
# 
# <img src="https://github.com/sberbank-ai-lab/lightautoml-datafest-workshop/raw/master/imgs/tutorial_blackbox_pipeline.png" alt="TabularAutoML preset pipeline" style="width:70%;"/>
# 
# in just several lines. Let's discuss the params we can setup:
# - `task` - the type of the ML task (the only **must have** parameter)
# - `timeout` - time limit in seconds for model to train
# - `cpu_limit` - vCPU count for model to use
# - `reader_params` - parameter change for Reader object inside preset, which works on the first step of data preparation: automatic feature typization, preliminary almost-constant features, correct CV setup etc. For example, we setup `n_jobs` threads for typization algo, `cv` folds and `random_state` as inside CV seed.
# - `general_params` - we use `use_algos` key to setup the model structure to work with (two LGBM models and two CatBoost models on the first level and their weighted composition creation on the second). This setup is only to speedup the kernel, you can remove this `general_params` setup if you want the whole LightAutoML model to run.
# 
# **Important note**: `reader_params` key is one of the YAML config keys, which is used inside `TabularAutoML` preset. [More details](https://github.com/sberbank-ai-lab/LightAutoML/blob/master/lightautoml/automl/presets/tabular_config.yml) on its structure with explanation comments can be found on the link attached. Each key from this config can be modified with user settings during preset object initialization. To get more info about different parameters setting (for example, ML algos which can be used in `general_params->use_algos`) please take a look at our [article on TowardsDataScience](https://towardsdatascience.com/lightautoml-preset-usage-tutorial-2cce7da6f936). 

# In[27]:


get_ipython().run_cell_magic('time', '', "importances = {}\ndt = pd.to_datetime(ALL_DF['date_time'])\nfor targ in targets_and_drop:\n    print('='*50, '='*50, sep = '\\n')\n    automl = TabularAutoML(task = task, \n                           timeout = TIMEOUT,\n                           cpu_limit = N_THREADS,\n                           reader_params = {'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE},\n                           general_params = {'use_algos': [['lgb', 'lgb_tuned', 'cb', 'cb_tuned']]},\n                           verbose = 3 # 0 for no output, 1 - only main steps, 2 - more detailed, 3 - show everything including model scores, optuna iterations etc.\n                          )\n    \n    ALL_DF['weight'] = [1.001] * len(train_data) + list(np.where(pseudolabels_true[targ].values >= 0, 1.001, 0.999))\n    roles['weights'] = 'weight'\n\n    roles['target'] = targ\n    roles['drop'] = targets_and_drop[targ]\n\n    if targ == 'target_nitrogen_oxides':\n        oof_pred = automl.fit_predict(ALL_DF[dt >= np.datetime64('2010-09-01')], roles = roles)\n    else:\n        oof_pred = automl.fit_predict(ALL_DF, roles = roles)\n    print('oof_pred:\\n{}\\nShape = {}'.format(oof_pred, oof_pred.shape))\n    \n    # MODEL STRUCTURE - NEW FEATURE\n    print('\\nFitted model structure:\\n{}\\n'.format(automl.create_model_str_desc()))\n    \n    # Fast feature importances calculation\n    fast_fi = automl.get_feature_scores('fast')\n    importances[targ] = fast_fi\n    \n    test_pred = automl.predict(test_data)\n    print('Prediction for te_data:\\n{}\\nShape = {}'.format(test_pred, test_pred.shape))\n    \n    sample_sub[targ] = np.clip(test_pred.data[:, 0], 0, None)\n")


# In[29]:


for targ in targets_and_drop:
    plt.figure(figsize = (30, 10))
    importances[targ].set_index('Feature')['Importance'].plot.bar()
    plt.title('Feature importances for {} model'.format(targ))
    plt.grid(True)
    plt.show()


# # Step 4. Create submission file

# In[30]:


sample_sub


# In[31]:


pseudolabels_true[['target_carbon_monoxide','target_benzene','target_nitrogen_oxides']]


# In[32]:


for targ in targets_and_drop:
    preds = sample_sub[targ].values
    real_values = pseudolabels_true[targ].values
    final_preds = np.where(real_values >= 0, real_values, preds)
    print(final_preds)
    sample_sub[targ] = final_preds


# In[33]:


sample_sub.to_csv('lightautoml_with_pseudolabelling_kernel_version_15.csv', index = False)


# # Additional materials

# - [Official LightAutoML github repo](https://github.com/sberbank-ai-lab/LightAutoML)
# - [LightAutoML documentation](https://lightautoml.readthedocs.io/en/latest)
# - [Pseudolabelling technique description post](https://www.kaggle.com/c/tabular-playground-series-apr-2021/discussion/231738#1268903)
# - [Baseline LightAutoML kernel without pseudolabelling](https://www.kaggle.com/alexryzhkov/tps-july-21-lightautoml-baseline)

# ## Do not forget to upvote if you like the kernel 👍
