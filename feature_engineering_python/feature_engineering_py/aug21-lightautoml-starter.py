#!/usr/bin/env python
# coding: utf-8

# <img src="https://github.com/sberbank-ai-lab/LightAutoML/raw/master/imgs/LightAutoML_logo_big.png" alt="LightAutoML logo" style="width:50%;"/>
# 
# ## Official LightAutoML github repository is [here](https://github.com/sberbank-ai-lab/LightAutoML)
# 
# ## Upvote is the best motivator 👍
# 
# # Step 0.0. LightAutoML installation

# This step can be used if you are working inside Google Colab/Kaggle kernels or want to install LightAutoML on your machine:

# In[1]:


# Developers version with better logging and final model description
get_ipython().system('pip install -U https://github.com/sberbank-ai-lab/LightAutoML/raw/fix/logging/LightAutoML-0.2.16.2-py3-none-any.whl')


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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import torch

# LightAutoML presets, task and report generation
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
from lightautoml.report.report_deco import ReportDeco


# # Step 0.2. Constants
# 
# Here we setup the constants to use in the kernel:
# - `N_THREADS` - number of vCPUs for LightAutoML model creation
# - `N_FOLDS` - number of folds in LightAutoML inner CV
# - `RANDOM_STATE` - random seed for better reproducibility
# - `TEST_SIZE` - houldout data part size 
# - `TIMEOUT` - limit in seconds for model to train
# - `TARGET_NAME` - target column name in dataset

# In[3]:


N_THREADS = 4
N_FOLDS = 10
RANDOM_STATE = 42
TEST_SIZE = 0.2
TIMEOUT = 3*3600
TARGET_NAME = 'loss'


# # Step 0.3. Imported models setup
# 
# For better reproducibility we fix numpy random seed with max number of threads for Torch (which usually try to use all the threads on server):

# In[4]:


np.random.seed(RANDOM_STATE)
torch.set_num_threads(N_THREADS)


# # Step 0.4. Data loading
# Let's check the data we have:

# In[5]:


get_ipython().run_cell_magic('time', '', "\ntrain_data = pd.read_csv('../input/tabular-playground-series-aug-2021/train.csv')\ntrain_data.head()\n")


# In[6]:


test_data = pd.read_csv('../input/tabular-playground-series-aug-2021/test.csv')
test_data.head()


# In[7]:


samp_sub = pd.read_csv('../input/tabular-playground-series-aug-2021/sample_submission.csv')
samp_sub.head()


# # Step 0.5. Insert OOF and Test predictions from NN model 😎
# 
# ### Thanks for these predictions goes to @oxzplvifi and [his notebook](https://www.kaggle.com/oxzplvifi/tabular-denoising-residual-network) - do not forget to upvote it 👍

# In[8]:


oof_preds = pd.read_csv('../input/tabular-denoising-residual-network/yoof.csv')
oof_preds.head()


# In[9]:


test_preds = pd.read_csv('../input/tabular-denoising-residual-network/submission.csv')
test_preds.head()


# In[10]:


train_data['NN_preds'] = oof_preds[TARGET_NAME].values
test_data['NN_preds'] = test_preds[TARGET_NAME].values
test_data.head()


# # Step 0.6. Data splitting for train-holdout
# As we have only one file with target values, we can split it into 80%-20% for holdout usage:

# In[11]:


get_ipython().run_cell_magic('time', '', "\ntr_data, te_data = train_test_split(train_data, \n                                    test_size=TEST_SIZE, \n                                    stratify=train_data[TARGET_NAME], \n                                    random_state=RANDOM_STATE)\nprint('Data splitted. Parts sizes: tr_data = {}, te_data = {}'.format(tr_data.shape, te_data.shape))\n")


# In[12]:


tr_data.head()


# # =========== LightAutoML model building ===========
# 
# 
# # Step 1. Task setup
# 
# On the cell below we create Task object - the class to setup what task LightAutoML model should solve with specific loss and metric if necessary (more info can be found [here](https://lightautoml.readthedocs.io/en/latest/generated/lightautoml.tasks.base.Task.html#lightautoml.tasks.base.Task) in our documentation):

# In[13]:


get_ipython().run_cell_magic('time', '', "\ndef rmse(y_true, y_pred, **kwargs):\n    return mean_squared_error(y_true, y_pred, squared = False, **kwargs)\n\ntask = Task('reg', )\n")


# # Step 2. Feature roles setup

# To solve the task, we need to setup columns roles. The **only role you must setup is target role**, everything else (drop, numeric, categorical, group, weights etc.) is up to user - LightAutoML models have automatic columns typization inside:

# In[14]:


get_ipython().run_cell_magic('time', '', "\nroles = {'target': TARGET_NAME,\n         'drop': ['id']\n         }\n")


# # Step 3. LightAutoML model creation - TabularAutoML preset

# In next the cell we are going to create LightAutoML model with `TabularAutoML` class - preset with default model structure like in the image below:
# 
# <img src="https://github.com/sberbank-ai-lab/LightAutoML/raw/master/imgs/tutorial_blackbox_pipeline.png" alt="TabularAutoML preset pipeline" style="width:85%;"/>
# 
# in just several lines. Let's discuss the params we can setup:
# - `task` - the type of the ML task (the only **must have** parameter)
# - `timeout` - time limit in seconds for model to train
# - `cpu_limit` - vCPU count for model to use
# - `reader_params` - parameter change for Reader object inside preset, which works on the first step of data preparation: automatic feature typization, preliminary almost-constant features, correct CV setup etc. For example, we setup `n_jobs` threads for typization algo, `cv` folds and `random_state` as inside CV seed.
# - `general_params` - we use `use_algos` key to setup the model structure to work with (Linear and LGBM model on the first level and their weighted composition creation on the second). This setup is only to speedup the kernel, you can remove this `general_params` setup if you want the whole LightAutoML model to run.
# 
# **Important note**: `reader_params` key is one of the YAML config keys, which is used inside `TabularAutoML` preset. [More details](https://github.com/sberbank-ai-lab/LightAutoML/blob/master/lightautoml/automl/presets/tabular_config.yml) on its structure with explanation comments can be found on the link attached. Each key from this config can be modified with user settings during preset object initialization. To get more info about different parameters setting (for example, ML algos which can be used in `general_params->use_algos`) please take a look at our [article on TowardsDataScience](https://towardsdatascience.com/lightautoml-preset-usage-tutorial-2cce7da6f936).
# 
# Moreover, to receive the automatic report for our model we will use `ReportDeco` decorator and work with the decorated version in the same way as we do with usual one. 

# ### If you want to use some specific LGBM or CatBoost params, this is the example for you below:

# In[15]:


lgb_params = {
    'metric': 'RMSE',
    'lambda_l1': 1e-07, 
    'lambda_l2': 2e-07, 
    'num_leaves': 42, 
    'feature_fraction': 0.55, 
    'bagging_fraction': 0.9, 
    'bagging_freq': 3, 
    'min_child_samples': 19,
    'num_threads': 4
}

cb_params = {
    'num_trees': 7000, 
    'od_wait': 1200, 
    'learning_rate': 0.02, 
    'l2_leaf_reg': 64, 
    'subsample': 0.83, 
    'random_strength': 17.17, 
    'max_depth': 6, 
    'min_data_in_leaf': 10, 
    'leaf_estimation_iterations': 3,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'bootstrap_type': 'Bernoulli',
    'leaf_estimation_method': 'Newton',
    'random_seed': 42,
    "thread_count": 4
}


# In[16]:


get_ipython().run_cell_magic('time', '', "\nautoml = TabularAutoML(task = task, \n                       timeout = TIMEOUT,\n                       cpu_limit = N_THREADS,\n                       reader_params = {'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE},\n                       general_params = {'use_algos': [['lgb', 'cb']]}, # LGBM and CatBoost algos only\n                       lgb_params = {'default_params': lgb_params, 'freeze_defaults': True}, # LGBM params\n                       cb_params = {'default_params': cb_params, 'freeze_defaults': True}, # CatBoost params\n                       verbose = 2 # Available values: 0,1,2,3 (from less detailed to more)\n                      )\n\nRD = ReportDeco(output_path = 'tabularAutoML_model_report')\nautoml_rd = RD(automl)\n\noof_pred = automl_rd.fit_predict(tr_data, roles = roles)\n")


# ### Received model looks like 👇👇👇

# In[17]:


print(automl_rd.model.create_model_str_desc())


# # Step 4. Feature importances calculation 
# 
# For feature importances calculation we have 2 different methods in LightAutoML:
# - Fast (`fast`) - this method uses feature importances from feature selector LGBM model inside LightAutoML. It works extremely fast and almost always (almost because of situations, when feature selection is turned off or selector was removed from the final models with all GBM models). no need to use new labelled data.
# - Accurate (`accurate`) - this method calculate *features permutation importances* for the whole LightAutoML model based on the **new labelled data**. It always works but can take a lot of time to finish (depending on the model structure, new labelled dataset size etc.).
# 
# In the cell below we will use `automl_rd.model` instead `automl_rd` because we want to take the importances from the model, not from the report. But **be carefull** - everything, which is calculated using `automl_rd.model` will not go to the report.

# In[18]:


get_ipython().run_cell_magic('time', '', "\n# Fast feature importances calculation\nfast_fi = automl_rd.model.get_feature_scores('fast')\nfast_fi.set_index('Feature')['Importance'].plot.bar(figsize = (30, 10), grid = True)\n")


# In[19]:


get_ipython().run_cell_magic('time', '', "\n# Accurate feature importances calculation (Permutation importances) -  can take long time to calculate\naccurate_fi = automl_rd.model.get_feature_scores('accurate', te_data, silent = False)\n")


# In[20]:


accurate_fi.set_index('Feature')['Importance'].plot.bar(figsize = (30, 10), grid = True)


# # Step 5. Prediction on holdout and metric calculation

# In[21]:


get_ipython().run_cell_magic('time', '', "\nte_pred = automl_rd.predict(te_data)\nprint('Prediction for te_data:\\n{}\\nShape = {}'\n              .format(te_pred, te_pred.shape))\n")


# In[22]:


print('Check scores...')
print('OOF score: {}'.format(rmse(tr_data[TARGET_NAME].values, oof_pred.data[:, 0])))
print('HOLDOUT score: {}'.format(rmse(te_data[TARGET_NAME].values, te_pred.data[:, 0])))


# # Bonus. Where is the automatic report?
# 
# As we used `automl_rd` in our training and prediction cells, it is already ready in the folder we specified - you can check the output kaggle folder and find the `tabularAutoML_model_report` folder with `lama_interactive_report.html` report inside (or just [click this link](tabularAutoML_model_report/lama_interactive_report.html) for short). It's interactive so you can click the black triangles on the left of the texts to go deeper in selected part.

# # Step 6. Spending more from TIMEOUT - `TabularUtilizedAutoML` usage
# 
# Using `TabularAutoML` we spent only 26 minutes to build the model. To spend (almost) all the `TIMEOUT` we can use `TabularUtilizedAutoML` preset instead of `TabularAutoML`, which has the same API:

# In[27]:


# %%time 

# automl = TabularUtilizedAutoML(task = task, 
#                                timeout = TIMEOUT,
#                                cpu_limit = N_THREADS,
#                                reader_params = {'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE},
#                                verbose = 1
#                               )

# oof_pred = automl.fit_predict(tr_data, roles = roles)
# print('oof_pred:\n{}\nShape = {}'.format(oof_pred, oof_pred.shape))


# ### Received model looks like 👇👇👇

# In[28]:


# print(automl.create_model_str_desc())


# # Step 7. Feature importances calculation for `TabularUtilizedAutoML`

# In[29]:


# %%time

# # Fast feature importances calculation
# fast_fi = automl.get_feature_scores('fast')
# fast_fi.set_index('Feature')['Importance'].plot.bar(figsize = (30, 10), grid = True)


# # Step 8. Prediction on holdout and metric calculation

# In[30]:


# %%time

# te_pred = automl.predict(te_data)
# print('Prediction for te_data:\n{}\nShape = {}'
#               .format(te_pred, te_pred.shape))


# In[31]:


# print('Check scores...')
# print('OOF score: {}'.format(roc_auc_score(tr_data[TARGET_NAME].values, oof_pred.data[:, 0])))
# print('HOLDOUT score: {}'.format(roc_auc_score(te_data[TARGET_NAME].values, te_pred.data[:, 0])))


# # Step 9. Retrain on the full dataset
# 
# Here we train LightAutoML model with `verbose = 1` to check the difference between logging levels:

# In[23]:


get_ipython().run_cell_magic('time', '', "\nautoml = TabularAutoML(task = task, \n                       timeout = TIMEOUT,\n                       cpu_limit = N_THREADS,\n                       reader_params = {'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE},\n                       general_params = {'use_algos': [['lgb', 'cb']]}, # LGBM and CatBoost algos only\n                       lgb_params = {'default_params': lgb_params, 'freeze_defaults': True}, # LGBM params\n                       cb_params = {'default_params': cb_params, 'freeze_defaults': True}, # CatBoost params\n                       verbose = 1 # Available values: 0,1,2,3 (from less detailed to more)\n                      )\n\noof_pred = automl.fit_predict(train_data, roles = roles)\n")


# ### Received model looks like 👇👇👇

# In[24]:


print(automl.create_model_str_desc())


# In[25]:


print('Check scores...')
print('OOF score: {}'.format(rmse(train_data[TARGET_NAME].values, oof_pred.data[:, 0])))


# In[26]:


test_pred = automl.predict(test_data)
print('Prediction for test_data:\n{}\nShape = {}'.format(test_pred, test_pred.shape))


# # Step 10. Create submission file

# In[27]:


samp_sub[TARGET_NAME] = test_pred.data[:, 0]
samp_sub.to_csv('In_LightAutoML_we_trust.csv', index = False)


# In[28]:


samp_sub


# ## Upvote if you like the kernel or find it useful 👍

# # Additional materials
# 
# - [Official LightAutoML github repo](https://github.com/sberbank-ai-lab/LightAutoML)
# - [LightAutoML documentation](https://lightautoml.readthedocs.io/en/latest)
