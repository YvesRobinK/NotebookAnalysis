#!/usr/bin/env python
# coding: utf-8

# #### Please upvote if you find the notebook useful
# 
# # Step 0.0. Install LightAutoML

# In[1]:


pip install -U lightautoml


# # Step 0.1. Import necessary libraries 

# In[2]:


# Standard python libraries
import os
import time
import re

# Installed libraries
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# Imports from our package
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task


# # Step 0.2. Parameters 

# In[3]:


N_THREADS = 4 # threads cnt for lgbm and linear models
N_FOLDS = 5 # folds cnt for AutoML
RANDOM_STATE = 42 # fixed random state for various reasons
TEST_SIZE = 0.2 # Test size for metric check
TIMEOUT = 3 * 3600 # Time in seconds for automl run
TARGET_NAME = 'target'


# # Step 0.3. Data load 

# In[4]:


get_ipython().run_cell_magic('time', '', "\ntrain_data = pd.read_csv('../input/tabular-playground-series-may-2021/train.csv')\ntrain_data[TARGET_NAME] = train_data[TARGET_NAME].str.slice(start=6).astype(int) - 1\ntrain_data.head()\n")


# In[5]:


test_data = pd.read_csv('../input/tabular-playground-series-may-2021/test.csv')
test_data.head()


# In[6]:


submission = pd.read_csv('../input/tabular-playground-series-may-2021/sample_submission.csv')
submission.head()


# # Step 0.5. Add new features

# In[7]:


def create_gr_feats(data):
    pass
    

all_df = pd.concat([train_data, test_data]).reset_index(drop = True)
create_gr_feats(all_df)
train_data, test_data = all_df[:len(train_data)], all_df[len(train_data):]
print(train_data.shape, test_data.shape)


# In[8]:


train_data.head()


# # ========= AutoML preset usage =========
# 
# 
# ## Step 1. Create Task

# In[9]:


get_ipython().run_cell_magic('time', '', "\ntask = Task('multiclass',)\n")


# ## Step 2. Setup columns roles

# In[10]:


get_ipython().run_cell_magic('time', '', "\nroles = {\n    'target': TARGET_NAME,\n    'drop': ['id'],\n}\n")


# ## Step 3. Train on full data 

# In[11]:


get_ipython().run_cell_magic('time', '', "\nautoml = TabularUtilizedAutoML(task = task, \n                               timeout = TIMEOUT,\n                               cpu_limit = N_THREADS,\n                               reader_params = {'n_jobs': N_THREADS},\n                               configs_list=[\n                                   '../input/lightautoml-configs/conf_0_sel_type_0.yml',\n                                   '../input/lightautoml-configs/conf_1_sel_type_1.yml'\n                               ])\noof_pred = automl.fit_predict(train_data, roles = roles)\nprint('oof_pred:\\n{}\\nShape = {}'.format(oof_pred[:10], oof_pred.shape))\n")


# In[12]:


get_ipython().run_cell_magic('time', '', "\n# Fast feature importances calculation\nfast_fi = automl.get_feature_scores('fast', silent = False)\nfast_fi.set_index('Feature')['Importance'].plot.bar(figsize = (20, 10), grid = True)\n")


# ## Step 4. Predict for test data and check OOF score

# In[13]:


get_ipython().run_cell_magic('time', '', "\ntest_pred = automl.predict(test_data)\nprint('Prediction for test data:\\n{}\\nShape = {}'.format(test_pred[:10], test_pred.shape))\n\nprint('Check scores...')\nprint('OOF score: {}'.format(log_loss(train_data[TARGET_NAME].values, oof_pred.data)))\n")


# ## Step 5. Prepare submission

# In[14]:


submission.iloc[:, 1:] = test_pred.data
submission.to_csv('lightautoml_2lvl_3hours_2configs.csv', index = False)


# In[15]:


submission


# In[ ]:




