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
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import rankdata
import torch

# LightAutoML presets, task and report generation
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
from lightautoml.report.report_deco import ReportDeco

# Everything for graphs
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


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
TIMEOUT = 6*3600
TARGET_NAME = 'loss'

CUTOFFS = [0, 3, 5, 7, 10, 13, 15, 20]


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


# ### We need to append NN predictions as well to make better generalization 
# #### Thanks for these predictions goes to @pourchot and [his notebook](https://www.kaggle.com/pourchot/in-python-tabular-denoising-residual-network) - do not forget to upvote it 👍

# In[8]:


train_data['NN_preds'] = pd.read_csv('../input/in-python-tabular-denoising-residual-network/oof.csv').iloc[:, 0].values
test_data['NN_preds'] = pd.read_csv('../input/in-python-tabular-denoising-residual-network/submission48.csv')[TARGET_NAME].values


# ### We also calculate and append XGB predictions

# In[9]:


get_ipython().run_cell_magic('time', '', "\nimport xgboost as xgb\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import KFold\n\ndef rmse(y_true, y_pred, **kwargs):\n    return mean_squared_error(y_true, y_pred, squared = False, **kwargs)\n\nX = train_data.drop(['id', 'loss', 'NN_preds'], axis=1).values\ny = train_data['loss'].values\nX_test = test_data.drop(['id', 'NN_preds'], axis=1).values\n\nscaler = StandardScaler()\nX = scaler.fit_transform(X)\nX_test = scaler.transform(X_test)\n\n# Fetch the best trial parameters and set some settings for the KFold predictions.\nxgb_params = {\n     'max_depth': 7, \n     'eta': 0.008373136177752354, \n     'subsample': 0.55, \n     'colsample_bytree': 0.65, \n     'min_child_weight': 56, \n     'reg_lambda': 49, \n     'reg_alpha': 43,\n     'tree_method': 'hist',\n     'n_estimators': 3700,\n     'n_jobs': N_THREADS\n}\n\noof_preds = np.array([0.0] * len(train_data))\ntest_preds = np.array([0.0] * len(test_data))\n\nkf = KFold(n_splits=10, shuffle=True, random_state = RANDOM_STATE)\n\nfor fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):\n    # Fetch the train-validation indices.\n    X_train, X_valid = X[train_idx], X[valid_idx]\n    y_train, y_valid = y[train_idx], y[valid_idx]\n    \n    # Create and fit a new model using the best parameters.\n    model = xgb.XGBRegressor(**xgb_params)\n    model.fit(X_train, y_train,\n            eval_set=[(X_valid, y_valid)],\n            eval_metric='rmse', verbose=False)\n    \n    # Validation predictions.\n    valid_pred = model.predict(X_valid)\n    oof_preds[valid_idx] = valid_pred\n    print('Fold {} RMSE: {}'.format(fold, rmse(y_valid, valid_pred)))\n    \n    test_preds += model.predict(X_test) / 10\n\nprint('Check scores...')\nprint('OOF score: {}'.format(rmse(y, oof_preds)))\n\ntrain_data['XGB_preds'] = oof_preds\ntest_data['XGB_preds'] = test_preds\n")


# In[10]:


plt.figure(figsize = (10, 10))
plt.scatter(train_data['XGB_preds'], train_data['NN_preds'])
plt.plot([0, 18], [0, 18], '--r')
plt.grid(True)
plt.title('Train data: XGB vs NN preds', fontsize = 17)
plt.xlabel('XGB_preds', fontsize = 17)
plt.ylabel('NN_preds', fontsize = 17)
plt.show()


# In[11]:


plt.figure(figsize = (10, 10))
plt.scatter(test_data['XGB_preds'], test_data['NN_preds'])
plt.plot([0, 18], [0, 18], '--r')
plt.grid(True)
plt.title('Test data: XGB vs NN preds', fontsize = 17)
plt.xlabel('XGB_preds', fontsize = 17)
plt.ylabel('NN_preds', fontsize = 17)
plt.show()


# In[12]:


for data in [train_data, test_data]:
    data['NN_minus_XGB'] = data['NN_preds'] - data['XGB_preds']
    data['NN_mul_XGB'] = data['NN_preds'] * data['XGB_preds']
    data['NN_div_XGB'] = data['NN_preds'] / data['XGB_preds']


# # Step 0.5. Data splitting for train-holdout
# As we have only one file with target values, we can split it into 80%-20% for holdout usage:

# In[13]:


get_ipython().run_cell_magic('time', '', "\ntr_data, te_data = train_test_split(train_data, \n                                    test_size=TEST_SIZE, \n                                    stratify=train_data[TARGET_NAME], \n                                    random_state=RANDOM_STATE)\nprint('Data splitted. Parts sizes: tr_data = {}, te_data = {}'.format(tr_data.shape, te_data.shape))\n")


# In[14]:


tr_data.head()


# In[15]:


tr_data[TARGET_NAME].value_counts().shape


# # =========== LightAutoML model building ===========

# # LightAutoML model creation - TabularAutoML preset

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
# ## In the cell below we are going to calculate LightAutoML classifier models for binary targets based on real target and cutoff like "bigger than X"

# In[16]:


get_ipython().run_cell_magic('time', '', "\nIMP_arr = []\nOOF_preds = []\nTEST_preds = []\nOOF_preds_parts = []\nTEST_preds_parts = []\nfor i in CUTOFFS:\n    print('Start {}'.format(i))\n    tr_data[TARGET_NAME + '_cl'] = (tr_data[TARGET_NAME] > i).astype(int)\n    te_data[TARGET_NAME + '_cl'] = (te_data[TARGET_NAME] > i).astype(int)\n    \n    # =============================================================\n    \n    task = Task('binary', )\n\n    roles = {'target': TARGET_NAME + '_cl',\n             'drop': ['id', TARGET_NAME]\n             }\n\n    automl = TabularAutoML(task = task, \n                           timeout = TIMEOUT,\n                           cpu_limit = N_THREADS,\n                           reader_params = {'n_jobs': N_THREADS, 'cv': 3, 'random_state': RANDOM_STATE},\n                           general_params = {'use_algos': [['lgb', 'cb']], \n                                             'return_all_predictions': True, # return all predictions from the layer before blender\n                                             'weighted_blender_max_nonzero_coef': 0.0}, # no drop for algos during blending phase\n                           verbose = 1 # Available values: 0,1,2,3 (from less detailed to more)\n                          )\n\n    oof_pred = automl.fit_predict(tr_data, roles = roles)\n    IMP_arr.append(automl.get_feature_scores('fast').set_index('Feature')['Importance'].to_dict())\n    te_pred = automl.predict(te_data)\n    \n    # =============================================================\n    \n    OOF_preds_parts.append(oof_pred.data)\n    TEST_preds_parts.append(te_pred.data)\n\n    oof_pred_weighted = np.dot(oof_pred.data, automl.blender.wts) # Create weighted OOF preds based on single algos and blender weights\n    te_pred_weighted = np.dot(te_pred.data, automl.blender.wts) # Create weighted Holdout preds based on single algos and blender weights\n    OOF_preds.append(oof_pred_weighted)\n    TEST_preds.append(te_pred_weighted)\n    \n    # =============================================================\n    \n    print('Check scores {}...'.format(i))\n    print('OOF score: {}'.format(roc_auc_score(tr_data[TARGET_NAME + '_cl'].values, oof_pred_weighted)))\n    print('HOLDOUT score: {}'.format(roc_auc_score(te_data[TARGET_NAME + '_cl'].values, te_pred_weighted)))\n")


# In[17]:


# Drop unnecessary columns created in the cell above
tr_data.drop(columns = [TARGET_NAME + '_cl'], inplace = True)
te_data.drop(columns = [TARGET_NAME + '_cl'], inplace = True)


# # It's time to check the feature importances for different target cutoffs

# In[18]:


def feat_imp_plot(df, title):
    plt.figure(figsize=(4,15))
    ax = sns.heatmap(df.set_index('Feature'), 
                     annot=False, 
                     cmap="RdBu", 
                     annot_kws={"weight": "bold", "fontsize":13})
    ax.set_title(title, fontsize=17)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor", weight="normal")
    plt.setp(ax.get_yticklabels(), weight="normal",
             rotation_mode="anchor", rotation=0, ha="right")
    plt.show();

feats_imp_df = pd.DataFrame()
feats_imp_df['Feature'] = ['f'+str(i) for i in range(100)] + ['NN_preds', 'XGB_preds']
for cutoff, mapper in zip(CUTOFFS, IMP_arr):
    feats_imp_df['Imps_'+str(cutoff)] = rankdata(feats_imp_df['Feature'].map(mapper))

feat_imp_plot(feats_imp_df, "Feature importances ranks for different target cutoffs\n100 - best, 0 - worst")


# In[19]:


# Making a plot
Feats_imp_df = pd.DataFrame()
Feats_imp_df['Feature'] = ['f'+str(i) for i in range(100)] + ['NN_preds', 'XGB_preds']
for cutoff, mapper in zip(CUTOFFS, IMP_arr):
    Feats_imp_df['Imps_'+str(cutoff)] = Feats_imp_df['Feature'].map(mapper)
    mx, mn = Feats_imp_df['Imps_'+str(cutoff)].max(), Feats_imp_df['Imps_'+str(cutoff)].min()
    Feats_imp_df['Imps_'+str(cutoff)] = (Feats_imp_df['Imps_'+str(cutoff)] - mn) / (mx - mn)

feat_imp_plot(Feats_imp_df, "Feature importances (min-max transform) for different target cutoffs\n1 - best, 0 - worst")


# ### In both variants we see the known features in the top - `f81`, `f52` and `f25`.  

# # Create new dataframes with classifiers predicts

# In[20]:


def combine_preds_array_to_df(cutoffs_arr, preds_parts):
    preds_df = pd.DataFrame()
    prev_cutoff = None
    for cutoff, pred in zip(cutoffs_arr, preds_parts):
        preds_df['LGBM_'+str(cutoff)] = pred[:, 0]
        preds_df['CB_'+str(cutoff)] = pred[:, 1]
        
        if prev_cutoff is not None:
            preds_df['diff_LGBM_'+str(cutoff)] = preds_df['LGBM_'+str(prev_cutoff)] - preds_df['LGBM_'+str(cutoff)]
            preds_df['diff_CB_'+str(cutoff)] = preds_df['CB_'+str(prev_cutoff)] - preds_df['CB_'+str(cutoff)]
        prev_cutoff = cutoff
    
    return preds_df

tr_preds_df = combine_preds_array_to_df(CUTOFFS, OOF_preds_parts)
te_preds_df = combine_preds_array_to_df(CUTOFFS, TEST_preds_parts)
print(tr_preds_df.shape, te_preds_df.shape)


# In[21]:


tr_preds_df.head()


# In[22]:


for col in tr_preds_df.columns:
    tr_data[col] = tr_preds_df[col].values
    te_data[col] = te_preds_df[col].values


# In[23]:


tr_data.head()


# # Comparing usual model with the model on extended dataset 

# In[24]:


task = Task('reg', )

roles = {'target': TARGET_NAME,
         'drop': ['id'] + list(tr_preds_df.columns)
         }

automl = TabularAutoML(task = task, 
                       timeout = TIMEOUT,
                       cpu_limit = N_THREADS,
                       reader_params = {'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE},
                       general_params = {'use_algos': [['lgb', 'cb']]},
                       verbose = 1 # Available values: 0,1,2,3 (from less detailed to more)
                      )

oof_pred = automl.fit_predict(tr_data, roles = roles)
fast_fi = automl.get_feature_scores('fast')
fast_fi.set_index('Feature')['Importance'].plot.bar(figsize = (30, 10), grid = True)
te_pred = automl.predict(te_data)

print('Check scores...')
print('OOF score: {}'.format(rmse(tr_data[TARGET_NAME].values, oof_pred.data[:, 0])))
print('HOLDOUT score: {}'.format(rmse(te_data[TARGET_NAME].values, te_pred.data[:, 0])))


# In[25]:


task = Task('reg', )

roles = {'target': TARGET_NAME,
         'drop': ['id']
         }

automl = TabularAutoML(task = task, 
                       timeout = TIMEOUT,
                       cpu_limit = N_THREADS,
                       reader_params = {'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE},
                       general_params = {'use_algos': [['lgb', 'cb']]},
                       verbose = 1 # Available values: 0,1,2,3 (from less detailed to more)
                      )

oof_pred = automl.fit_predict(tr_data, roles = roles)
fast_fi = automl.get_feature_scores('fast')
fast_fi.set_index('Feature')['Importance'].plot.bar(figsize = (30, 10), grid = True)
te_pred = automl.predict(te_data)

print('Check scores...')
print('OOF score: {}'.format(rmse(tr_data[TARGET_NAME].values, oof_pred.data[:, 0])))
print('HOLDOUT score: {}'.format(rmse(te_data[TARGET_NAME].values, te_pred.data[:, 0])))


# ### Great - as we can see above the classifier predictions usage idea works fine and we can reproduce it on the full dataset
# 
# # Retrain on the full dataset

# In[26]:


get_ipython().run_cell_magic('time', '', "\nOOF_preds = []\nTEST_preds = []\nOOF_preds_parts = []\nTEST_preds_parts = []\nfor i in CUTOFFS:\n    print('Start {}'.format(i))\n    train_data[TARGET_NAME + '_cl'] = (train_data[TARGET_NAME] > i).astype(int)\n    \n    # =============================================================\n    \n    task = Task('binary', )\n\n    roles = {'target': TARGET_NAME + '_cl',\n             'drop': ['id', TARGET_NAME]\n             }\n\n    automl = TabularAutoML(task = task, \n                           timeout = TIMEOUT,\n                           cpu_limit = N_THREADS,\n                           reader_params = {'n_jobs': N_THREADS, 'cv': 10, 'random_state': RANDOM_STATE},\n                           general_params = {'use_algos': [['lgb', 'cb']], \n                                             'return_all_predictions': True, \n                                             'weighted_blender_max_nonzero_coef': 0.0},\n                           verbose = 1 # Available values: 0,1,2,3 (from less detailed to more)\n                          )\n\n    oof_pred = automl.fit_predict(train_data, roles = roles)\n    test_pred = automl.predict(test_data)\n    \n    # =============================================================\n    \n    OOF_preds_parts.append(oof_pred.data)\n    TEST_preds_parts.append(test_pred.data)\n\n    oof_pred_weighted = np.dot(oof_pred.data, automl.blender.wts)\n    test_pred_weighted = np.dot(test_pred.data, automl.blender.wts)\n    OOF_preds.append(oof_pred_weighted)\n    TEST_preds.append(test_pred_weighted)\n    \n    # =============================================================\n    \n    print('Check scores {}...'.format(i))\n    print('OOF score: {}'.format(roc_auc_score(train_data[TARGET_NAME + '_cl'].values, oof_pred_weighted)))\n")


# In[27]:


# Drop unnecessary columns created in the cell above
train_data.drop(columns = [TARGET_NAME + '_cl'], inplace = True)


# In[28]:


def combine_preds_array_to_df(cutoffs_arr, preds_parts):
    preds_df = pd.DataFrame()
    prev_cutoff = None
    for cutoff, pred in zip(cutoffs_arr, preds_parts):
        preds_df['LGBM_'+str(cutoff)] = pred[:, 0]
        preds_df['CB_'+str(cutoff)] = pred[:, 1]
        preds_df['LGBM_CB_diff_'+str(cutoff)] = pred[:, 0] - pred[:, 1]
        if prev_cutoff is not None:
            preds_df['diff_LGBM_'+str(cutoff)] = preds_df['LGBM_'+str(prev_cutoff)] - preds_df['LGBM_'+str(cutoff)]
            preds_df['diff_CB_'+str(cutoff)] = preds_df['CB_'+str(prev_cutoff)] - preds_df['CB_'+str(cutoff)]
        prev_cutoff = cutoff
    
    return preds_df

train_preds_df = combine_preds_array_to_df(CUTOFFS, OOF_preds_parts)
test_preds_df = combine_preds_array_to_df(CUTOFFS, TEST_preds_parts)
print(train_preds_df.shape, test_preds_df.shape)


# In[29]:


for col in train_preds_df.columns:
    train_data[col] = train_preds_df[col].values
    test_data[col] = test_preds_df[col].values


# # Now we are ready for training the model

# In[30]:


lgb_params = {
    'metric': 'RMSE',
    'feature_pre_filter': False,
    'lambda_l1': 0.45,
    'lambda_l2': 4.8,
    'learning_rate': 0.005,
    'num_trees': 80000,
    'early_stopping_rounds': 200,
    'num_leaves': 10, 
    'feature_fraction': 0.4, 
    'bagging_fraction': 1.0, 
    'bagging_freq': 0, 
    'min_child_samples': 100,
    'num_threads': 4
}

cb_params = {
    'num_trees': 7000, 
    'od_wait': 600, 
    'learning_rate': 0.015, 
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


# In[31]:


get_ipython().run_cell_magic('time', '', "\nCONF_PATH = '../input/lightautoml-configs/'\n\ntask = Task('reg', )\n\nroles = {\n    'target': TARGET_NAME,\n    'drop': ['id']\n}\n\nautoml = TabularUtilizedAutoML(task = task, \n                               timeout = TIMEOUT,\n                               cpu_limit = N_THREADS,\n                               reader_params = {'n_jobs': N_THREADS, 'cv': 10, 'random_state': RANDOM_STATE},\n                               general_params = {'use_algos': [['lgb', 'cb']]}, # LGBM and CatBoost algos only\n                               lgb_params = {'default_params': lgb_params, 'freeze_defaults': True}, # LGBM params\n                               cb_params = {'default_params': cb_params, 'freeze_defaults': True}, # CatBoost params\n                               verbose = 2, # Available values: 0,1,2,3 (from less detailed to more)\n                               configs_list = [CONF_PATH + 'conf_0_sel_type_0.yml',\n                                               CONF_PATH + 'conf_2_select_mode_1_no_typ.yml',\n                                               CONF_PATH + 'conf_4_sel_type_0_no_int.yml',\n                                               CONF_PATH + 'conf_6_sel_type_1_tuning_full_no_int_lgbm.yml'],\n                               max_runs_per_config=2\n                              )\n\noof_pred = automl.fit_predict(train_data, roles = roles)\n")


# In[32]:


fast_fi = automl.get_feature_scores('fast')
fast_fi.set_index('Feature')['Importance'].plot.bar(figsize = (30, 10), grid = True)


# ### Received model looks like 👇👇👇

# In[33]:


print(automl.create_model_str_desc())


# # Predict for test data 

# In[34]:


test_pred = automl.predict(test_data)
print('Prediction for test_data:\n{}\nShape = {}'.format(test_pred, test_pred.shape))


# # Create submission file

# In[35]:


samp_sub[TARGET_NAME] = test_pred.data[:, 0]
samp_sub.to_csv('LightAutoML_utilized_submission.csv', index = False)


# In[36]:


samp_sub


# ## Upvote if you like the kernel or find it useful 👍

# # Additional materials
# 
# - [Official LightAutoML github repo](https://github.com/sberbank-ai-lab/LightAutoML)
# - [LightAutoML documentation](https://lightautoml.readthedocs.io/en/latest)
# - [LightAutoML starter for TPS August 2021](https://www.kaggle.com/alexryzhkov/aug21-lightautoml-starter)

# In[ ]:




