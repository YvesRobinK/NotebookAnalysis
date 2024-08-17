#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.display import display, HTML, Javascript

# ----- Notebook Theme -----
color_map = ['#16a085', '#e8f6f3', '#d0ece7', '#a2d9ce', '#73c6b6', '#45b39d', 
                        '#16a085', '#138d75', '#117a65', '#0e6655', '#0b5345']

prompt = color_map[-1]
main_color = color_map[0]
strong_main_color = color_map[1]
custom_colors = [strong_main_color, main_color]

css_file = ''' 

div #notebook {
background-color: white;
line-height: 20px;
}

#notebook-container {
%s
margin-top: 2em;
padding-top: 2em;
border-top: 4px solid %s; /* light orange */
-webkit-box-shadow: 0px 0px 8px 2px rgba(224, 212, 226, 0.5); /* pink */
    box-shadow: 0px 0px 8px 2px rgba(224, 212, 226, 0.5); /* pink */
}

div .input {
margin-bottom: 1em;
}

.rendered_html h1, .rendered_html h2, .rendered_html h3, .rendered_html h4, .rendered_html h5, .rendered_html h6 {
color: %s; /* light orange */
font-weight: 600;
}

div.input_area {
border: none;
    background-color: %s; /* rgba(229, 143, 101, 0.1); light orange [exactly #E58F65] */
    border-top: 2px solid %s; /* light orange */
}

div.input_prompt {
color: %s; /* light blue */
}

div.output_prompt {
color: %s; /* strong orange */
}

div.cell.selected:before, div.cell.selected.jupyter-soft-selected:before {
background: %s; /* light orange */
}

div.cell.selected, div.cell.selected.jupyter-soft-selected {
    border-color: %s; /* light orange */
}

.edit_mode div.cell.selected:before {
background: %s; /* light orange */
}

.edit_mode div.cell.selected {
border-color: %s; /* light orange */

}
'''
def to_rgb(h): 
    return tuple(int(h[i:i+2], 16) for i in [0, 2, 4])

main_color_rgba = 'rgba(%s, %s, %s, 0.1)' % (to_rgb(main_color[1:]))
open('notebook.css', 'w').write(css_file % ('width: 95%;', main_color, main_color, main_color_rgba, main_color,  main_color, prompt, main_color, main_color, main_color, main_color))

def nb(): 
    return HTML("<style>" + open("notebook.css", "r").read() + "</style>")
nb()


# <img src="https://github.com/AILab-MLTools/LightAutoML/raw/master/imgs/LightAutoML_logo_big.png" alt="LightAutoML logo" style="width:70%;"/>

# # LightAutoML baseline
# 
# Official LightAutoML github repository is [here](https://github.com/AILab-MLTools/LightAutoML). 
# 
# ### Do not forget to put upvote for the notebook and the ⭐️ for github repo if you like it - one click for you, great pleasure for us ☺️ 

# In[2]:


s = '<iframe src="https://ghbtns.com/github-btn.html?user=sb-ai-lab&repo=LightAutoML&type=star&count=true&size=large" frameborder="0" scrolling="0" width="170" height="30" title="LightAutoML GitHub"></iframe>'
HTML(s)


# ## This notebook is the updated copy of our [Tutorial_1 from the GIT repository](https://github.com/AILab-MLTools/LightAutoML/blob/master/examples/tutorials/Tutorial_1_basics.ipynb). Please check our [tutorials folder](https://github.com/AILab-MLTools/LightAutoML/blob/master/examples/tutorials) if you are interested in other examples of LightAutoML functionality.

# ## 0. Prerequisites

# ### 0.0. install LightAutoML

# In[3]:


get_ipython().run_cell_magic('capture', '', '!pip3 install -U lightautoml\n\n# QUICK WORKAROUND FOR PROBLEM WITH PANDAS\n!pip3 install -U pandas\n')


# ### 0.1. Import libraries
# 
# Here we will import the libraries we use in this kernel:
# - Standard python libraries for timing, working with OS etc.
# - Essential python DS libraries like numpy, pandas, scikit-learn and torch (the last we will use in the next cell)
# - LightAutoML modules: `TabularAutoML` preset for AutoML model creation and Task class to setup what kind of ML problem we solve (binary/multiclass classification or regression)

# In[4]:


# Standard python libraries
import os
import time

# Essential DS libraries
import numpy as np
import pandas as pd
import torch

# LightAutoML presets, task and report generation
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task


# ### 0.2. Constants
# 
# Here we setup the constants to use in the kernel:
# - `N_THREADS` - number of vCPUs for LightAutoML model creation
# - `N_FOLDS` - number of folds in LightAutoML inner CV
# - `RANDOM_STATE` - random seed for better reproducibility
# - `TEST_SIZE` - houldout data part size 
# - `TIMEOUT` - limit in seconds for model to train
# - `TARGET_NAME` - target column name in dataset

# In[5]:


N_THREADS = 4
N_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2
TIMEOUT = 8 * 3600 # equal to 8 hours
TARGET_NAME = 'target'


# ### 0.3. Imported models setup
# 
# For better reproducibility fix numpy random seed with max number of threads for Torch (which usually try to use all the threads on server):

# In[6]:


np.random.seed(RANDOM_STATE)
torch.set_num_threads(N_THREADS)


# ### 0.4. Data loading
# 
# For now it's time to load the data:

# In[7]:


train_data = pd.read_pickle('../input/amexaggdatapicklef32/train_agg_f32.pkl', compression="gzip")
print(train_data.shape)
train_data.head()


# ### In the cell below we used the trick with data denoising proposed by [@RADDAR](https://www.kaggle.com/code/raddar/the-data-has-random-uniform-noise-added/notebook) and [Chris](https://www.kaggle.com/competitions/amex-default-prediction/discussion/327651) - upvote their notebook and discussion topic for the great insight 👍

# In[8]:


for col in train_data.columns:
    if train_data[col].dtype=='float16':
        train_data[col] = train_data[col].astype('float32').round(decimals=2).astype('float16')


# ### 0.5. OOF and test predictions from XGB kernel
# 
# #### In cell below we upload predictions for train and test datasets from [XGB Starter kernel](https://www.kaggle.com/code/cdeotte/xgboost-starter-0-793) made by [@Chris](https://www.kaggle.com/cdeotte) - if you still didn't upvote it, that's a great chance 👍:

# In[9]:


oof_mapper = pd.read_csv('../input/xgboost-starter-0-793/oof_xgb_v1.csv').set_index('customer_ID')
test_mapper = pd.read_csv('../input/xgboost-starter-0-793/submission_xgb_v1.csv').set_index('customer_ID')


# In[10]:


chris_xgb_oof = train_data['customer_ID'].map(oof_mapper['oof_pred']).values
chris_xgb_oof


# # 1. Task definition

# ### 1.1. Task type
# 
# On the cell below we create Task object - the class to setup what task LightAutoML model should solve with specific loss and metric if necessary (more info can be found [here](https://lightautoml.readthedocs.io/en/latest/pages/modules/generated/lightautoml.tasks.base.Task.html#lightautoml.tasks.base.Task) in our documentation):

# In[11]:


# COMPETITION METRIC FROM Konstantin Yakovlev
# https://www.kaggle.com/kyakovlev
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/327534
def amex_metric_mod(y_true, y_pred):

    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)


# In[12]:


task = Task('binary', )


# ### 1.2. Feature roles setup

# To solve the task, we need to setup columns roles. The **only role you must setup is target role**, everything else (drop, numeric, categorical, group, weights etc.) is up to user - LightAutoML models have automatic columns typization inside:

# In[13]:


roles = {
    'target': TARGET_NAME,
    'drop': ['customer_ID']
}


# ### 1.3. LightAutoML model creation - TabularAutoML preset

# In next the cell we are going to create LightAutoML model with `TabularAutoML` class - preset with default model structure like in the image below:
# 
# <img src="https://github.com/AILab-MLTools/LightAutoML/raw/master/imgs/tutorial_blackbox_pipeline.png" alt="TabularAutoML preset pipeline" style="width:85%;"/>
# 
# in just several lines. Let's discuss the params we can setup:
# - `task` - the type of the ML task (the only **must have** parameter)
# - `timeout` - time limit in seconds for model to train
# - `cpu_limit` - vCPU count for model to use
# - `reader_params` - parameter change for Reader object inside preset, which works on the first step of data preparation: automatic feature typization, preliminary almost-constant features, correct CV setup etc. For example, we setup `n_jobs` threads for typization algo, `cv` folds and `random_state` as inside CV seed.
# 
# **Important note**: `reader_params` key is one of the YAML config keys, which is used inside `TabularAutoML` preset. [More details](https://github.com/AILab-MLTools/LightAutoML/blob/master/lightautoml/automl/presets/tabular_config.yml) on its structure with explanation comments can be found on the link attached. Each key from this config can be modified with user settings during preset object initialization. To get more info about different parameters setting (for example, ML algos which can be used in `general_params->use_algos`) please take a look at our [article on TowardsDataScience](https://towardsdatascience.com/lightautoml-preset-usage-tutorial-2cce7da6f936).
# 
# Moreover, to receive the automatic report for our model we can use `ReportDeco` decorator and work with the decorated version in the same way as we do with usual one (more details in [this tutorial](https://github.com/AILab-MLTools/LightAutoML/blob/master/examples/tutorials/Tutorial_1_basics.ipynb))

# In[14]:


automl = TabularAutoML(
    task = task, 
    timeout = TIMEOUT,
    cpu_limit = N_THREADS,
    general_params = {'use_algos': [['linear_l2', 'lgb', 'cb']]},
    reader_params = {'n_jobs': 1, 'cv': N_FOLDS, 'random_state': RANDOM_STATE},
    selection_params = {'mode': 0}
)


# # 2. AutoML training

# To run autoML training use fit_predict method:
# - `train_data` - Dataset to train.
# - `roles` - Roles dict.
# - `verbose` - Controls the verbosity: the higher, the more messages.
#         <1  : messages are not displayed;
#         >=1 : the computation process for layers is displayed;
#         >=2 : the information about folds processing is also displayed;
#         >=3 : the hyperparameters optimization process is also displayed;
#         >=4 : the training process for every algorithm is displayed;
# 
# Note: out-of-fold prediction is calculated during training and returned from the fit_predict method

# In[15]:


get_ipython().run_cell_magic('time', '', 'oof_pred = automl.fit_predict(train_data, roles = roles, verbose = 3)\n')


# In[16]:


print(automl.create_model_str_desc())


# In[17]:


print(f'OOF score: {amex_metric_mod(train_data[TARGET_NAME].values, oof_pred.data[:, 0])}')


# In[18]:


best_w = None
best_sc = -1
for w in np.arange(0, 1.01, 0.01):
    sc = amex_metric_mod(train_data[TARGET_NAME].values, w * oof_pred.data[:, 0] + (1-w)*chris_xgb_oof)
    if sc > best_sc:
        best_sc = sc
        best_w = w
        print('{:.7f} {:.2f}'.format(best_sc, best_w))
        
print('Finally selected: Score = {:.7f}, Best_w = {:.2f}'.format(best_sc, best_w))


# In[19]:


print(f'Final OOF score: {amex_metric_mod(train_data[TARGET_NAME].values, best_w * oof_pred.data[:, 0] + (1-best_w)*chris_xgb_oof)}')


# # 3. Feature importances calculation 
# 
# For feature importances calculation we have 2 different methods in LightAutoML:
# - Fast (`fast`) - this method uses feature importances from feature selector LGBM model inside LightAutoML. It works extremely fast and almost always (almost because of situations, when feature selection is turned off or selector was removed from the final models with all GBM models). no need to use new labelled data.
# - Accurate (`accurate`) - this method calculate *features permutation importances* for the whole LightAutoML model based on the **new labelled data**. It always works but can take a lot of time to finish (depending on the model structure, new labelled dataset size etc.).

# In[20]:


# %%time

# # Fast feature importances calculation
# fast_fi = automl.get_feature_scores('fast').head(75)
# top_3_features = fast_fi['Feature'].values[:3]
# fast_fi.set_index('Feature')['Importance'].plot.bar(figsize = (30, 10), grid = True)


# In[21]:


# fast_fi.head()


# ### Plot PDP graphs for LightAutoML model

# In[22]:


data = pd.read_pickle('../input/amexaggdatapicklef32/test_agg_f32_part_0.pkl', compression="gzip")
for col in data.columns:
    if data[col].dtype=='float16':
        data[col] = data[col].astype('float32').round(decimals=2).astype('float16')


# In[23]:


automl.plot_pdp(data.sample(20000), feature_name='P_2_last')


# In[24]:


automl.plot_pdp(data.sample(20000), feature_name='D_39_last')


# # 4. Predict for test dataset
# 
# We are also ready to predict for our test competition dataset and submission file creation:

# In[25]:


import gc
del train_data
gc.collect()


# In[26]:


test_predictions = []
for i in range(10):
    data = pd.read_pickle('../input/amexaggdatapicklef32/test_agg_f32_part_{}.pkl'.format(i), compression="gzip")
    chris_xgb_test = data['customer_ID'].map(test_mapper['prediction']).values
    for col in data.columns:
        if data[col].dtype=='float16':
            data[col] = data[col].astype('float32').round(decimals=2).astype('float16')
    print(i, data.shape)
    test_pred = automl.predict(data)
    test_predictions += list(best_w * test_pred.data[:, 0] + (1-best_w)*chris_xgb_test)


# In[27]:


submission = pd.read_csv('../input/amex-default-prediction/sample_submission.csv')
print(submission.shape)
submission.head()


# In[28]:


submission['prediction'] = test_predictions
submission.to_csv('lightautoml_tabularautoml.csv', index = False)
submission


# # Additional materials

# - [Official LightAutoML github repo](https://github.com/AILab-MLTools/LightAutoML)
# - [LightAutoML documentation](https://lightautoml.readthedocs.io/en/latest)
# - [LightAutoML tutorials](https://github.com/AILab-MLTools/LightAutoML/tree/master/examples/tutorials)
# - LightAutoML course:
#     - [Part 1 - general overview](https://ods.ai/tracks/automl-course-part1) 
#     - [Part 2 - LightAutoML specific applications](https://ods.ai/tracks/automl-course-part2)
#     - [Part 3 - LightAutoML customization](https://ods.ai/tracks/automl-course-part3)
# - [OpenDataScience AutoML benchmark leaderboard](https://ods.ai/competitions/automl-benchmark/leaderboard)

# ### If you still like the notebook, do not forget to put upvote for the notebook and the ⭐️ for github repo if you like it using the button below - one click for you, great pleasure for us ☺️

# In[29]:


s = '<iframe src="https://ghbtns.com/github-btn.html?user=sb-ai-lab&repo=LightAutoML&type=star&count=true&size=large" frameborder="0" scrolling="0" width="170" height="30" title="LightAutoML GitHub"></iframe>'
HTML(s)

