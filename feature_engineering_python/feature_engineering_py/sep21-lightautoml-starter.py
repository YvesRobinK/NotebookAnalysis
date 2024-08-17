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


get_ipython().system('pip install -U lightautoml')


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
from sklearn.metrics import roc_auc_score
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
N_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2
TIMEOUT = 5*3600
TARGET_NAME = 'claim'


# # Step 0.3. Imported models setup
# 
# For better reproducibility we fix numpy random seed with max number of threads for Torch (which usually try to use all the threads on server):

# In[4]:


np.random.seed(RANDOM_STATE)
torch.set_num_threads(N_THREADS)


# # Step 0.4. Data loading
# Let's check the data we have:

# In[5]:


get_ipython().run_cell_magic('time', '', "\ntrain_data = pd.read_csv('../input/tabular-playground-series-sep-2021/train.csv')\ntrain_data.head()\n")


# In[6]:


test_data = pd.read_csv('../input/tabular-playground-series-sep-2021/test.csv')
test_data.head()


# In[7]:


samp_sub = pd.read_csv('../input/tabular-playground-series-sep-2021/sample_solution.csv')
samp_sub.head()


# # Step 0.5. Data splitting for train-holdout
# As we have only one file with target values, we can split it into 80%-20% for holdout usage:

# In[8]:


get_ipython().run_cell_magic('time', '', "\ntr_data, te_data = train_test_split(train_data, \n                                    test_size=TEST_SIZE, \n                                    stratify=train_data[TARGET_NAME], \n                                    random_state=RANDOM_STATE)\nprint('Data splitted. Parts sizes: tr_data = {}, te_data = {}'.format(tr_data.shape, te_data.shape))\n")


# In[9]:


tr_data.head()


# # Step 0.6. Add OOFs and Test predictions from AutoWoE models

# In[10]:


from autowoe import AutoWoE
from sklearn.model_selection import StratifiedKFold

def get_oof_and_test_pred(tr, te, real_te):
    skf = StratifiedKFold(n_splits=N_FOLDS, random_state = RANDOM_STATE)

    oof_preds_woe = np.zeros(len(tr))
    test_preds_woe = np.zeros(len(te))
    real_test_preds_woe = np.zeros(len(real_te))

    y = tr['claim'].values

    for fold, (train_idx, val_idx) in enumerate(skf.split(y, y)):

        X_tr, X_val = tr.iloc[train_idx, :], tr.iloc[val_idx, :]

        auto_woe = AutoWoE(monotonic=False,
                         vif_th=20.,
                         imp_th=0,
                         th_const=32,
                         force_single_split=True,
                         min_bin_size = 0.005,
                         oof_woe=True,
                         n_folds=5,
                         n_jobs=N_THREADS,
                         regularized_refit=True,
                         verbose=0)

        auto_woe.fit(X_tr.sample(100000, random_state = 13).drop('id', axis = 1), 
                     target_name="claim")

        val_pred = auto_woe.predict_proba(X_val)
        print("FOLD {}, AUC_SCORE = {:.5f}".format(fold, roc_auc_score(X_val['claim'], val_pred)))

        oof_preds_woe[val_idx] = val_pred
        test_preds_woe += auto_woe.predict_proba(te) / N_FOLDS
        real_test_preds_woe += auto_woe.predict_proba(real_te) / N_FOLDS

    print("AUC_SCORE TRAIN = {:.5f}".format(roc_auc_score(tr_data['claim'], oof_preds_woe)))
    print("AUC_SCORE TEST = {:.5f}".format(roc_auc_score(te_data['claim'], test_preds_woe)))
    
    return oof_preds_woe, test_preds_woe, real_test_preds_woe


# In[11]:


oof_preds_woe, test_preds_woe, real_test_preds_woe = get_oof_and_test_pred(tr_data, te_data, test_data)

# This idea was in my mind but as it was already announced in @hiro5299834 
tr_data['missed_cnt'] = tr_data.isna().sum(axis=1)
te_data['missed_cnt'] = te_data.isna().sum(axis=1)
test_data['missed_cnt'] = test_data.isna().sum(axis=1)
oof_preds_woe2, test_preds_woe2, real_test_preds_woe2 = get_oof_and_test_pred(tr_data, te_data, test_data)


# In[12]:


print("AUC_SCORE TEST = {:.5f}".format(roc_auc_score(te_data['claim'], test_preds_woe)))
print("AUC_SCORE TEST = {:.5f}".format(roc_auc_score(te_data['claim'], test_preds_woe2)))
print("AUC_SCORE TEST = {:.5f}".format(roc_auc_score(te_data['claim'], 0.5 * test_preds_woe +
                                                                         0.5 * test_preds_woe2)))


# In[13]:


from scipy.stats import rankdata
print("AUC_SCORE TEST = {:.5f}".format(roc_auc_score(te_data['claim'], 0.5 * rankdata(test_preds_woe) +
                                                                         0.5 * rankdata(test_preds_woe2))))


# In[14]:


tr_data['oof_woe_1'] = oof_preds_woe
te_data['oof_woe_1'] = test_preds_woe
test_data['oof_woe_1'] = real_test_preds_woe

tr_data['oof_woe_2'] = oof_preds_woe2
te_data['oof_woe_2'] = test_preds_woe2
test_data['oof_woe_2'] = real_test_preds_woe2

tr_data['oof_woe_12'] = 0.5 * oof_preds_woe + 0.5 * oof_preds_woe2
te_data['oof_woe_12'] = 0.5 * test_preds_woe + 0.5 * test_preds_woe2
test_data['oof_woe_12'] = 0.5 * real_test_preds_woe + 0.5 * real_test_preds_woe2

tr_data['rank_oof_woe_1'] = rankdata(oof_preds_woe)
te_data['rank_oof_woe_1'] = rankdata(test_preds_woe)
test_data['rank_oof_woe_1'] = rankdata(real_test_preds_woe)

tr_data['rank_oof_woe_2'] = rankdata(oof_preds_woe2)
te_data['rank_oof_woe_2'] = rankdata(test_preds_woe2)
test_data['rank_oof_woe_2'] = rankdata(real_test_preds_woe2)

tr_data['rank_oof_woe_12'] = 0.5 * rankdata(oof_preds_woe) + 0.5 * rankdata(oof_preds_woe2)
te_data['rank_oof_woe_12'] = 0.5 * rankdata(test_preds_woe) + 0.5 * rankdata(test_preds_woe2)
test_data['rank_oof_woe_12'] = 0.5 * rankdata(real_test_preds_woe) + 0.5 * rankdata(real_test_preds_woe2)


# # =========== LightAutoML model building ===========
# 
# 
# # Step 1. Task setup
# 
# On the cell below we create Task object - the class to setup what task LightAutoML model should solve with specific loss and metric if necessary (more info can be found [here](https://lightautoml.readthedocs.io/en/latest/generated/lightautoml.tasks.base.Task.html#lightautoml.tasks.base.Task) in our documentation):

# In[15]:


get_ipython().run_cell_magic('time', '', "\ntask = Task('binary', )\n")


# # Step 2. Feature roles setup

# To solve the task, we need to setup columns roles. The **only role you must setup is target role**, everything else (drop, numeric, categorical, group, weights etc.) is up to user - LightAutoML models have automatic columns typization inside:

# In[16]:


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
# - `general_params` - we use `use_algos` key to setup the model structure to work with (Linear and LGBM models on the first level and their weighted composition creation on the second). This setup is only to speedup the kernel, you can remove this `general_params` setup if you want the whole LightAutoML model to run.
# 
# **Important note**: `reader_params` key is one of the YAML config keys, which is used inside `TabularAutoML` preset. [More details](https://github.com/sberbank-ai-lab/LightAutoML/blob/master/lightautoml/automl/presets/tabular_config.yml) on its structure with explanation comments can be found on the link attached. Each key from this config can be modified with user settings during preset object initialization. To get more info about different parameters setting (for example, ML algos which can be used in `general_params->use_algos`) please take a look at our [article on TowardsDataScience](https://towardsdatascience.com/lightautoml-preset-usage-tutorial-2cce7da6f936).
# 
# Moreover, to receive the automatic report for our model we will use `ReportDeco` decorator and work with the decorated version in the same way as we do with usual one. 

# In[17]:


# %%time 

# automl = TabularAutoML(task = task, 
#                        timeout = TIMEOUT,
#                        cpu_limit = N_THREADS,
#                        reader_params = {'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE},
#                        general_params = {'use_algos': [['lgb', 'lgb_tuned']]},
#                        selection_params = {'mode': 0}
#                       )

# RD = ReportDeco(output_path = 'tabularAutoML_model_report')
# automl_rd = RD(automl)

# oof_pred = automl_rd.fit_predict(tr_data, roles = roles)


# # Step 4. Feature importances calculation 
# 
# For feature importances calculation we have 2 different methods in LightAutoML:
# - Fast (`fast`) - this method uses feature importances from feature selector LGBM model inside LightAutoML. It works extremely fast and almost always (almost because of situations, when feature selection is turned off or selector was removed from the final models with all GBM models). no need to use new labelled data.
# - Accurate (`accurate`) - this method calculate *features permutation importances* for the whole LightAutoML model based on the **new labelled data**. It always works but can take a lot of time to finish (depending on the model structure, new labelled dataset size etc.).
# 
# In the cell below we will use `automl_rd.model` instead `automl_rd` because we want to take the importances from the model, not from the report. But **be carefull** - everything, which is calculated using `automl_rd.model` will not go to the report.

# In[18]:


# %%time

# # Fast feature importances calculation
# fast_fi = automl_rd.model.get_feature_scores('fast')
# fast_fi.set_index('Feature')['Importance'].plot.bar(figsize = (30, 10), grid = True)


# In[19]:


# %%time

# # Accurate feature importances calculation (Permutation importances) -  can take long time to calculate, so we sample the data to make it faster
# accurate_fi = automl_rd.model.get_feature_scores('accurate', te_data.sample(25000, random_state = RANDOM_STATE), silent = False)


# In[20]:


# accurate_fi.set_index('Feature')['Importance'].plot.bar(figsize = (30, 10), grid = True)


# # Step 5. Prediction on holdout and metric calculation

# In[21]:


# %%time

# te_pred = automl_rd.predict(te_data)
# print('Prediction for te_data:\n{}\nShape = {}'
#               .format(te_pred, te_pred.shape))


# In[22]:


# print('Check scores...')
# print('OOF score: {}'.format(roc_auc_score(tr_data[TARGET_NAME].values, oof_pred.data[:, 0])))
# print('HOLDOUT score: {}'.format(roc_auc_score(te_data[TARGET_NAME].values, te_pred.data[:, 0])))


# # Bonus. Where is the automatic report?
# 
# As we used `automl_rd` in our training and prediction cells, it is already ready in the folder we specified - you can check the output kaggle folder and find the `tabularAutoML_model_report` folder with `lama_interactive_report.html` report inside (or just [click this link](tabularAutoML_model_report/lama_interactive_report.html) for short). It's interactive so you can click the black triangles on the left of the texts to go deeper in selected part.

# # Step 6. Retrain on the full dataset

# In[23]:


train_data = pd.concat([tr_data, te_data]).reset_index(drop = True)
print(train_data.shape)
train_data.head()


# In[24]:


get_ipython().run_cell_magic('time', '', "\nautoml = TabularAutoML(task = task, \n                       timeout = TIMEOUT,\n                       cpu_limit = N_THREADS,\n                       reader_params = {'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE},\n                       tuning_params = {'max_tuning_time': 900}, # more time for params tuning\n                       general_params = {'use_algos': [['lgb', 'lgb_tuned']]},\n                       selection_params = {'mode': 0} # no feature selection - everything is necessary :)\n                      )\n\noof_pred = automl.fit_predict(train_data, roles = roles)\n")


# In[25]:


print('Check scores...')
print('OOF score: {}'.format(roc_auc_score(train_data[TARGET_NAME].values, oof_pred.data[:, 0])))


# In[26]:


test_pred = automl.predict(test_data)
print('Prediction for test_data:\n{}\nShape = {}'.format(test_pred, test_pred.shape))


# # Step 7. Create submission file

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
