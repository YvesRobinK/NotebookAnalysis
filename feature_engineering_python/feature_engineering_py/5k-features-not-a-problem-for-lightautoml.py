#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# In[2]:


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


# ![](https://raw.githubusercontent.com/sb-ai-lab/LightAutoML/39cb56feae6766464d39dd2349480b97099d2535/imgs/LightAutoML_logo_big.png)

# # LightAutoML feature selection baseline
# 
# ### In this kernel we are going to select only necessary features using LightAutoML feature selection in one line 😎
# 
# Official LightAutoML github repository is [here](https://github.com/sb-ai-lab/LightAutoML). 
# 
# ### Do not forget to put upvote for the notebook, follow me on Kaggle and the ⭐️ for github repo if you like it - one click for you, great pleasure for us ☺️ 

# In[3]:


s = '<iframe src="https://ghbtns.com/github-btn.html?user=sb-ai-lab&repo=LightAutoML&type=star&count=true&size=large" frameborder="0" scrolling="0" width="170" height="30" title="LightAutoML GitHub"></iframe>'
HTML(s)


# ## This notebook is the updated copy of our [Tutorial_1 from the GIT repository](https://github.com/sb-ai-lab/LightAutoML/blob/master/examples/tutorials/Tutorial_1_basics.ipynb). Please check our [tutorials folder](https://github.com/sb-ai-lab/LightAutoML/blob/master/examples/tutorials) if you are interested in other examples of LightAutoML functionality.

# ## 0. Prerequisites

# ### 0.0. install LightAutoML

# In[4]:


get_ipython().system('pip install -U --ignore-installed lightautoml')


# ### 0.1. Import libraries
# 
# Here we will import the libraries we use in this kernel:
# - Standard python libraries for timing, working with OS and HTTP requests etc.
# - Essential python DS libraries like numpy, pandas, scikit-learn and torch (the last we will use in the next cell)
# - LightAutoML modules: presets for AutoML, task and report generation module

# In[5]:


# Standard python libraries
import os
import time
import requests

# Essential DS libraries
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import torch

# LightAutoML presets, task and report generation
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
from lightautoml.report.report_deco import ReportDeco


# ### 0.2. Constants
# 
# Here we setup some parameters to use in the kernel:
# - `N_THREADS` - number of vCPUs for LightAutoML model creation
# - `N_FOLDS` - number of folds in LightAutoML inner CV
# - `RANDOM_STATE` - random seed for better reproducibility
# - `TEST_SIZE` - houldout data part size 
# - `TIMEOUT` - limit in seconds for model to train
# - `TARGET_NAME` - target column name in dataset

# In[6]:


N_THREADS = 4
N_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2
TIMEOUT = 10*3600
TARGET_NAME = 'label'


# ### 0.3. Imported models setup
# 
# For better reproducibility fix numpy random seed with max number of threads for Torch (which usually try to use all the threads on server):

# In[7]:


np.random.seed(RANDOM_STATE)
torch.set_num_threads(N_THREADS)


# ### 0.4. Data loading
# Let's check the data we have:

# In[8]:


INPUT_DIR = '../input/tabular-playground-series-nov-2022/'
SUBS_DIR = INPUT_DIR + 'submission_files/'


# In[9]:


labels = pd.read_csv(INPUT_DIR + "train_labels.csv")
print(labels.shape)
labels.head()


# In[10]:


submission = pd.read_csv(INPUT_DIR + "sample_submission.csv")
print(submission.shape)
submission.head()


# In[11]:


full_data = pd.DataFrame(list(range(40000)), columns=['id'])

for it, fname in enumerate(sorted(os.listdir(SUBS_DIR))):
    if it % 500 == 0:
        print(it)
    full_data['sub_'+str(it)] = np.clip(pd.read_csv(SUBS_DIR + fname)['pred'].values, 1e-6, 1 - 1e-6)

train_data = full_data.iloc[:labels.shape[0], :]
train_data[TARGET_NAME] = labels[TARGET_NAME]
test_data = full_data.iloc[labels.shape[0]:, :].reset_index(drop=True)
print(train_data.shape, test_data.shape)


# In[12]:


train_data.head()


# In[13]:


test_data.head()


# ### 0.5. Simple feature engineering

# In[14]:


tmp = train_data.drop(columns = ['id', TARGET_NAME])
train_data['mean_pred'] = tmp.mean(axis = 1)
train_data['std_pred'] = tmp.std(axis = 1)

tmp = test_data.drop(columns = ['id'])
test_data['mean_pred'] = tmp.mean(axis = 1)
test_data['std_pred'] = tmp.std(axis = 1)


# In[15]:


useful_subs = ['sub_65', 'sub_56', 'sub_62', 'sub_72', 'sub_121']
for i in range(len(useful_subs) - 1):
    for j in range(i+1, len(useful_subs)):
        train_data[useful_subs[i] + '_minus_' + useful_subs[j]] = train_data[useful_subs[i]] - train_data[useful_subs[j]]
        test_data[useful_subs[i] + '_minus_' + useful_subs[j]] = test_data[useful_subs[i]] - test_data[useful_subs[j]]
        
        train_data[useful_subs[i] + '_div_' + useful_subs[j]] = train_data[useful_subs[i]] / train_data[useful_subs[j]]
        test_data[useful_subs[i] + '_div_' + useful_subs[j]] = test_data[useful_subs[i]] / test_data[useful_subs[j]]


# In[16]:


train_data.shape, test_data.shape


# In[17]:


train_data.head()


# It is important to note that missing values (NaN and other) in the data should be left as is, unless the reason for their presence or their specific meaning are known. Otherwise, AutoML model will perceive the filled NaNs as a true pattern between the data and the target variable, without knowledge and assumptions about missing values, which can negatively affect the model quality. LighAutoML can deal with missing values and outliers automatically.

# ## 1. Task definition

# ### 1.1. Task type

# First we need to create ```Task``` object - the class to setup what task LightAutoML model should solve with specific loss and metric if necessary (more info can be found [here](https://lightautoml.readthedocs.io/en/latest/pages/modules/generated/lightautoml.tasks.base.Task.html#lightautoml.tasks.base.Task) in our documentation).
# 
# The following task types are available:
# 
# - ```'binary'``` - for binary classification.
# 
# - ```'reg’``` - for regression.
# 
# - ```‘multiclass’``` - for multiclass classification.
# 
# - ```'multi:reg``` - for multiple regression.
# 
# - ```'multilabel'``` - for multi-label classification.
# 
# In this example we will consider a binary classification:

# In[18]:


task = Task('binary', metric = 'logloss')


# Note that only logloss loss is available for binary task and it is the default loss. Default metric for binary classification is ROC-AUC. See more info about available and default losses and metrics [here](https://lightautoml.readthedocs.io/en/latest/pages/modules/generated/lightautoml.tasks.base.Task.html#lightautoml.tasks.base.Task). 
# 
# **Depending on the task, you can and shold choose exactly those metrics and losses that you want and need to optimize.**

# ### 1.2. Feature roles setup

# To solve the task, we need to setup columns roles. LightAutoML can automatically define types and roles of data columns, but it is possible to specify it directly through the dictionary parameter ```roles``` when training AutoML model (see next section "AutoML training"). Specific roles can be specified using a string with the name (any role can be set like this).  So the key in dictionary must be the name of the role, the value must be a list of the names of the corresponding columns in dataset. The **only role you must setup is** ```'target'``` **role** (that is column with target variable obviously), everything else (```'drop', 'numeric', 'categorical', 'group', 'weights'``` etc) is up to user:

# In[19]:


roles = {
    'target': TARGET_NAME,
    'drop': ['id']
}


# You can also optionally specify the following roles:
# 
# - ```'numeric'``` - numerical feature
# 
# - ```'category'``` - categorical feature
# 
# - ```'text'``` - text data
# 
# - ```'datetime'``` - features with date and time 
# 
# - ```'date'``` - features with date only
# 
# - ```'group'``` - features by which the data can be divided into groups and which can be taken into account for group k-fold validation (so the same group is not represented in both testing and training sets)
# 
# - ```'drop'``` - features to drop, they will not be used in model building
# 
# - ```'weights'``` - object weights for the loss and metric
# 
# - ```'path'``` - image file paths (for CV tasks)
# 
# - ```'treatment'``` - object group in uplift modelling tasks: treatment or control
# 
# Note that role name can be written in any case. Also it is possible to pass individual objects of role classes with specific arguments instead of strings with role names for specific tasks and more optimal pipeline construction ([more details](https://github.com/sb-ai-lab/LightAutoML/blob/master/lightautoml/dataset/roles.py)).
# 
# For example, to set the date role, you can use the ```DatetimeRole``` class. 

# In[20]:


# from lightautoml.dataset.roles import DatetimeRole


# Different seasonality can be extracted from the data through the ```seasonality``` parameter: years (```'y'```), months (```'m'```), days (```'d'```), weekdays (```'wd'```), hours (```'hour'```), minutes (```'min'```), seconds (```'sec'```), milliseconds (```'ms'```), nanoseconds (```'ns'```). This features will be considered as categorical. Another important parameter is ```base_date```. It allows to specify the base date and convert the feature to the distances to this date (set to ```False``` by default). Also for all roles classes there is a ```force_input``` parameter, and if it is ```True```, then the corresponding features will pass all further feature selections and won't be excluded (equals ```False``` by default). Also it is always possible to specify data type for all roles using ```dtype``` argument.
# 
# Here is an example of such a role assignment through a class object for date feature (but there is no such feature in the considered dataset):

# In[21]:


# roles = {
#     DatetimeRole(base_date=False, seasonality=('d', 'wd', 'hour')): 'date_time'
# }


# Any role can be set through a class object. Information about specific parameters of specific roles and other datailed information can be found [here](https://github.com/sb-ai-lab/LightAutoML/blob/master/lightautoml/dataset/roles.py).

# ### 1.3. LightAutoML model creation - TabularAutoML preset

# Next we are going to create LightAutoML model with `TabularAutoML` class - preset with default model structure in just several lines. 
# 
# In general, the whole AutoML model consists of multiple levels, which can contain several pipelines with their own set of data processing methods and ML models. The outputs of one level are the inputs of the next, and on the last level predictions of previous level models are combined with blending procedure. All this can be combined into a model using the ```AutoML``` class and its various descendants (like ```TabularAutoML```).
# 
# Let's look at how the LightAutoML model is arranged and what it consists in general.
# 
# ![](https://raw.githubusercontent.com/sb-ai-lab/LightAutoML/master/imgs/tutorial_1_laml_big.png)
# 
# #### 1.3.1 Reader object
# 
# First the task and data go into ```Reader``` object. It analyzes the data and extracts various valuable information from them. Also it can detect and remove useless features, conduct feature selection, determine types and roles etc. Let's look at this steps in more detail.
# 
# **Role and types guessing**
# 
# Roles can be specified as a string or a specific class object, or defined automatically. For ```TabularAutoML``` preset ```'numeric'```, ```'datetime'``` and ```'category'``` roles can be automatically defined. There are two ways of role defining. **First** is very simple: check if the value can be converted to a date (```'datetime'```), otherwise check if it can be converted to a number (```'numeric'```), otherwise declare it a category (```'categorical'```). But this method may not work well on large data or when encoding categories with integers. The **second** method is based on statistics: the distributions of numerical features are considered, and how similar they are to the distributions of real or categorical value. Also different ways of feature encoding (as a number or as a category) are compared and based on normalized Gini index it is decided which encoding is better. For this case a set of specific rules is created, and if at least one of them is fullfilled, then the feature will be assigned to numerical, otherwise to categorical. This check can be enabled or disabled using the ```advanced_roles``` parameter. 
# 
# If roles are explicitly specified, automatic definition won't be applied to the specified dataset columns. In the case of specifying a role as an object of a certain class, through its arguments, it is possible to set the processing parameters in more detail.
#  
# **Feature selection**
# 
# In general, the AutoML pipeline uses pre-selection, generation and post-selection of features. ```TabularAutoML``` has no post-selection stage. There are three feature selection methods: its absence, using features importances and more strict selection (forward selection). The GBM model is used to evaluate features importances. Importances can be calculated in 2 ways: based on splits (how many times a split was made for each feature in the entire ensemble) or using permutation feature importances (mixing feature values during validation and assessing quality change in this case). Second method is harder but it requires holdout data. Then features with importance above a certain threshold are selected. Faster and more strict feature selection method is forward selection. Features are sorted in descending order of importance, then in blocks (size of 1 by default) a model is built based on selected features, and its quality is measured. Then the next block of features is added, and they are saved if the quality has improved with them, and so on.  
# 
# Also LightAutoML can merge some columns if it is adequate and leads to an improvement in the model quality (for example, an intersection between categorical variables). Different columns join options are considered, the best one is chosen by the normalized Gini index. 
# 
# #### 1.3.2 Machine learning pipelines architecture and training
# 
# As a result, after analyzing and processing the data, the ```Reader``` object forms and returns a ```LAMA Dataset```. It contains the original data and markup with metainformation. In this dataset it is possible to see the roles defined by the ```Reader``` object, selected features etc. Then ML pipelines are trained on this data. 
# 
# ![](https://raw.githubusercontent.com/sb-ai-lab/LightAutoML/master/imgs/tutorial_1_ml_pipeline.png)
# 
# Each such pipeline is one or more machine learning algorithms that share one post-processing block and one validation scheme. Several such pipelines can be trained in parallel on one dataset, and they form a level. Number of levels can be unlimited as possible. List of all levels of AutoML pipeline is available via ```.levels``` attribute of ```AutoML``` instance. Level predictions can be inputs to other models or ML pipelines (i. e. stacking scheme). As inputs for subsequent levels, it is possible to use the original data by setting ```skip_conn``` argument in ```True``` when initializing preset instance. At the last level, if there are several pipelines, blending is used to build a prediction. 
# 
# Different types of features are processed depending on the models. Numerical features are processed for linear model preprocessing: standardization, replacing missing values with median, discretization, log odds (if feature is probability - output of previous level). Categories are processed using label encoding (by default), one hot encoding, ordinal encoding, frequency encoding, out of fold target encoding. 
# 
# The following algorithms are available in the LightAutoML: linear regression with L2 regularization, LightGBM, CatBoost, random forest. 
# 
# By default KFold cross-validation is used during training at all levels (for hyperparameter optimization and building out-of-fold prediction during training), and for each algorithm a separate model is built for each validation fold, and their predictions are averaged. So the predictions at each level and the resulting prediction during training are out-of-fold predictions. But it is also possible to just pass a holdout data for validation or use custom cross-validation schemes, setting ```cv_iter``` iterator returning the indices of the objects for validation. LightAutoML has ready-made iterators, for example, ```TimeSeriesIterator``` for time series split. To further reduce the effect of overfitting, it is possible to use nested cross-validation (```nested_cv``` parameter), but it is not used by default. 
# 
# Prediction on new data is the averaging of models over all folds from validation and blending. 
# 
# Hyperparameter tuning of machine learning algorithms can be performed during training (early stopping by the number of trees in gradient boosting or the number of training epochs of neural networks etc), based on expert rules (according to data characteristics and empirical recommendations, so-called expert parameters), by the sequential model-based optimization (SMBO, bayesian optimization: Optuna with TPESampler) or by grid search. LightGBM and CatBoost can be used with parameter tuning or with expert parameters, with no tuning. For linear regression parameters are always tuned using warm start model training technique. 
# 
# At the last level blending is used to build a prediction. There are three available blending methods: choosing the best model based on a given metric (other models are just discarded), simple averaging of all models, or weighted averaging (weights are selected using coordinate descent algorithm with optimization of a given metric). ```TabularAutoML ``` uses the latter strategy by default. It is worth noting that, unlike stacking, blending can exclude models from composition. 
# 
# #### 1.3.3 Timing
# 
# When creating AutoML object, a certain time limit is set, and it schedules a list of tasks that it can complete during this time, and it will initially allocate approximately equal time for each task. In the process of solving objectives, it understands how to adjust the time allocated to different subtasks. If AutoML finished working earlier than set timeout, it means that it completed the entire list of tasks. If AutoML worked to the limit and turned off, then most likely it sacrificed something, for example, reduced the number of algorithms for training, realized that it would not have time to train the next one, or it might not calculate the full cross-validation cycle for one of the models (then on folds, where the model has not trained, the predictiuons will be NaN, and the model related to this fold will not participate in the final averaging). The resulting quality is evaluated at the blending stage, and if necessary and possible, the composition will be corrected. 
# 
# If you do not set the time for AutoML during initialization, then by default it will be equal to a very large number, that is, sooner or later AutoML will complete all tasks. 
# 
# #### 1.3.4 LightAutoML model creation
# 
# So the entire AutoML pipeline can be composed from various parts by user (see [custom pipeline tutorial](https://github.com/sb-ai-lab/LightAutoML/blob/master/examples/tutorials/Tutorial_6_custom_pipeline.ipynb)), but it is possible to use presets - in a certain sense, fixed strategies for dynamic pipeline building. 
# 
# Here is a default AutoML pipeline for binary classification and regression tasks (```TabularAutoML``` preset):
# 
# ![](https://raw.githubusercontent.com/sb-ai-lab/LightAutoML/ac3c1b38873437eb74354fb44e68a449a0200aa6/imgs/tutorial_blackbox_pipeline.png)
# 
# Another example:
# 
# ![](https://raw.githubusercontent.com/sb-ai-lab/LightAutoML/ac3c1b38873437eb74354fb44e68a449a0200aa6/imgs/tutorial_1_pipeline.png)
# 
# Let's discuss some of the params we can setup:
# - `task` - the type of the ML task (the only **must have** parameter)
# - `timeout` - time limit in seconds for model to train
# - `cpu_limit` - vCPU count for model to use
# - `reader_params` - parameter change for ```Reader``` object inside preset, which works on the first step of data preparation: automatic feature typization, preliminary almost-constant features, correct CV setup etc. For example, we setup `n_jobs` threads for typization algo, `cv` folds and `random_state` as inside CV seed.
# - `general_params` - general parameters dictionary, in which it is possible to specify a list of algorithms used (```'use_algos'```), nested CV using (```'nested_cv'```) etc.
# 
# **Important note**: `reader_params` key is one of the YAML config keys, which is used inside `TabularAutoML` preset. [More details](https://github.com/sb-ai-lab/LightAutoML/blob/master/lightautoml/automl/presets/tabular_config.yml) on its structure with explanation comments can be found on the link attached. Each key from this config can be modified with user settings during preset object initialization. To get more info about different parameters setting (for example, ML algos which can be used in `general_params->use_algos`) please take a look at our [article on TowardsDataScience](https://towardsdatascience.com/lightautoml-preset-usage-tutorial-2cce7da6f936).
# 
# Moreover, to receive the automatic report for our model we will use `ReportDeco` decorator and work with the decorated version in the same way as we do with usual one. 

# ## In the cell below we setup LightAutoML to use iterative selector with testing 50 features at once before all the algorithms training:

# In[22]:


automl = TabularAutoML(
    task = task, 
    timeout = TIMEOUT,
    cpu_limit = N_THREADS,
    selection_params = {'mode': 2, 'feature_group_size': 50, 'select_algos': ['linear_l2', 'gbm']}, # HERE IS THE NEW LINE :)
    reader_params = {'n_jobs': N_THREADS}
)


# ## 2. AutoML training

# To run autoML training use ```fit_predict``` method. 
# 
# Main arguments:
# 
# - `train_data` - dataset to train.
# - `roles` - column roles dict.
# - `verbose` - controls the verbosity: the higher, the more messages:
#         <1  : messages are not displayed;
#         >=1 : the computation process for layers is displayed;
#         >=2 : the information about folds processing is also displayed;
#         >=3 : the hyperparameters optimization process is also displayed;
#         >=4 : the training process for every algorithm is displayed;
# 
# Note: out-of-fold prediction is calculated during training and returned from the fit_predict method

# In[23]:


get_ipython().run_cell_magic('time', '', 'oof_pred = automl.fit_predict(train_data, roles = roles, verbose = 1)\n')


# ## Now in the cell below we can see what are the selected feats:

# In[24]:


selected_feats = sorted(automl.collect_used_feats())
print(selected_feats)
print('LightAutoML has selected {} feats!'.format(len(selected_feats)))


# ## Wow, that's amazing - 300 features out of 5k features (6% of features) ❗️❗️❗️

# ## 3. Prediction for test data

# In[25]:


get_ipython().run_cell_magic('time', '', "\ntest_pred = automl.predict(test_data)\nprint(f'Prediction for te_data:\\n{test_pred}\\nShape = {test_pred.shape}')\n")


# In[26]:


print(f'OOF score: {log_loss(train_data[TARGET_NAME].values, oof_pred.data[:, 0])}')


# ## And the score for 300 features is even better than with the full 5k features ❗️❗️❗️

# ## 4. Model analysis

# ### 4.1. Model structure

# You can obtain the description of the resulting pipeline:

# In[27]:


print(automl.create_model_str_desc())


# ### 4.2 Feature importances calculation 

# 
# For feature importances calculation we have 2 different methods in LightAutoML:
# - Fast (`fast`) - this method uses feature importances from feature selector LGBM model inside LightAutoML. It works extremely fast and almost always (almost because of situations, when feature selection is turned off or selector was removed from the final models with all GBM models). There is no need to use new labelled data.
# - Accurate (`accurate`) - this method calculate *features permutation importances* for the whole LightAutoML model based on the **new labelled data**. It always works but can take a lot of time to finish (depending on the model structure, new labelled dataset size etc.).

# In[28]:


get_ipython().run_cell_magic('time', '', "\n# Fast feature importances calculation\nfast_fi = automl.get_feature_scores('fast').head(100)\nfast_fi.set_index('Feature')['Importance'].plot.bar(figsize = (30, 10), grid = True)\n")


# ## 5. Create submission

# In[29]:


submission['pred'] = test_pred.data[:, 0]


# In[30]:


submission


# In[31]:


submission.to_csv('LightAutoML_submission.csv', index = False)


# ### If you still like the notebook (and the selected feats), do not forget to put upvote for the notebook and the ⭐️ for github repo if you like it using the button below - one click for you, great pleasure for us ☺️

# In[32]:


s = '<iframe src="https://ghbtns.com/github-btn.html?user=sb-ai-lab&repo=LightAutoML&type=star&count=true&size=large" frameborder="0" scrolling="0" width="170" height="30" title="LightAutoML GitHub"></iframe>'
HTML(s)


# ## Additional materials

# - [Official LightAutoML github repo](https://github.com/AILab-MLTools/LightAutoML)
# - [LightAutoML documentation](https://lightautoml.readthedocs.io/en/latest)
# - [LightAutoML tutorials](https://github.com/AILab-MLTools/LightAutoML/tree/master/examples/tutorials)
# - LightAutoML course:
#     - [Part 1 - general overview](https://ods.ai/tracks/automl-course-part1) 
#     - [Part 2 - LightAutoML specific applications](https://ods.ai/tracks/automl-course-part2)
#     - [Part 3 - LightAutoML customization](https://ods.ai/tracks/automl-course-part3)
# - [OpenDataScience AutoML benchmark leaderboard](https://ods.ai/competitions/automl-benchmark/leaderboard)
