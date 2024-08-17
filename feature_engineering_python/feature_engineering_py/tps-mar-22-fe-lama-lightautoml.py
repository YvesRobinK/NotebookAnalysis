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


# <img src="https://github.com/sberbank-ai-lab/LightAutoML/raw/master/imgs/LightAutoML_logo_big.png" alt="LightAutoML logo" style="width:70%;"/>

# # LightAutoML baseline
# 
# Official LightAutoML github repository is [here](https://github.com/sberbank-ai-lab/LightAutoML). 
# 
# ### Do not forget to put upvote for the notebook and the ⭐️ for github repo if you like it using the button below - one click for you, great pleasure for us ☺️ 

# In[2]:


s = '<iframe src="https://ghbtns.com/github-btn.html?user=sberbank-ai-lab&repo=LightAutoML&type=star&count=true&size=large" frameborder="0" scrolling="0" width="170" height="30" title="LightAutoML GitHub"></iframe>'
HTML(s)


# ## 0. Prerequisites

# ### 0.0. install LightAutoML

# In[3]:


get_ipython().run_cell_magic('capture', '', '!pip install -U lightautoml\n')


# ### 0.1. Import libraries
# 
# Here we will import the libraries we use in this kernel:
# - Standard python libraries for timing, working with OS etc.
# - Essential python DS libraries like numpy, pandas, scikit-learn and torch (the last we will use in the next cell)
# - LightAutoML modules: presets for AutoML, task and report generation module

# In[4]:


# Standard python libraries
import os
import time

# Essential DS libraries
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split
import torch

# LightAutoML presets, task and report generation
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
from lightautoml.report.report_deco import ReportDeco

#Libraries for working with date
import holidays
import datetime


# ### 0.2. Constants
# 
# Here we setup the constants to use in the kernel:
# - `N_THREADS` - number of vCPUs for LightAutoML model creation
# - `RANDOM_STATE` - random seed for better reproducibility
# - `TEST_SIZE` - houldout data part size 
# - `TIMEOUT` - limit in seconds for model to train
# - `TARGET_NAME` - target column name in dataset

# In[5]:


N_THREADS = 4 
RANDOM_STATE = 21
TIMEOUT = 8 * 3600
TARGET_NAME = 'congestion'


# ### 0.3. Imported models setup
# 
# For better reproducibility fix numpy random seed with max number of threads for Torch (which usually try to use all the threads on server):

# In[6]:


np.random.seed(RANDOM_STATE)
torch.set_num_threads(N_THREADS)


# ### 0.4. Data loading
# Let's check the data we have:

# In[7]:


INPUT_DIR = '../input/tabular-playground-series-mar-2022/'


# In[8]:


train_data = pd.read_csv(INPUT_DIR + 'train.csv', dtype={'time': str})
print(train_data.shape)
train_data.head()


# In[9]:


train_data.info()


# In[10]:


test_data = pd.read_csv(INPUT_DIR + 'test.csv', dtype={'time': str})
print(test_data.shape)
test_data.head()


# In[11]:


test_data.info()


# In[12]:


submission = pd.read_csv(INPUT_DIR + 'sample_submission.csv')
print(submission.shape)
submission.head()


# ### 0.5. Feature engineering
# Let's make some new features:

# In[13]:


dir_mapper = {'EB': [1,0], 
              'NB': [0,1], 
              'SB': [0,-1], 
              'WB': [-1,0], 
              'NE': [1,1], 
              'SW': [-1,-1], 
              'NW': [-1,1], 
              'SE': [1,-1]}

def is_holiday(date):
    if date in holidays.US():
        return 1
    return 0


def is_day_before_holiday(date):
    if date + datetime.timedelta(days=1) in holidays.US():
        return 1
    return 0


def is_day_after_holiday(date):
    if date - datetime.timedelta(days=1) in holidays.US():
        return 1
    return 0


def feature_engineering(data):
    data['time'] = pd.to_datetime(data['time'])
    data['month'] = data['time'].dt.month
    data['weekday'] = data['time'].dt.weekday
    data['hour'] = data['time'].dt.hour
    data['minute'] = data['time'].dt.minute
#     data['converted_direction_coord_0'] = data['direction'].map(lambda x: dir_mapper[x][0])
#     data['converted_direction_coord_1'] = data['direction'].map(lambda x: dir_mapper[x][1])
    data['is_month_start'] = data['time'].dt.is_month_start.astype('int')
    data['is_month_end'] = data['time'].dt.is_month_end.astype('int')
    data['hour+minute'] = data['time'].dt.hour * 60 + data['time'].dt.minute
    data['is_weekend'] = (data['time'].dt.dayofweek > 4).astype('int')
    data['is_afternoon'] = (data['time'].dt.hour > 12).astype('int')
    data['x+y'] = data['x'].astype('str') + data['y'].astype('str')
    data['x+y+direction'] = data['x'].astype('str') + data['y'].astype('str') + data['direction'].astype('str')
#     data['x+y+direction0'] = data['x'].astype('str') + data['y'].astype('str') + data['converted_direction_coord_0'].astype('str')
#     data['x+y+direction1'] = data['x'].astype('str') + data['y'].astype('str') + data['converted_direction_coord_1'].astype('str')
    data['hour+direction'] = data['hour'].astype('str') + data['direction'].astype('str')
    data['hour+x+y'] = data['hour'].astype('str') + data['x'].astype('str') + data['y'].astype('str')
    data['hour+direction+x'] = data['hour'].astype('str') + data['direction'].astype('str') + data['x'].astype('str')
    data['hour+direction+y'] = data['hour'].astype('str') + data['direction'].astype('str') + data['y'].astype('str')
    data['hour+direction+x+y'] = data['hour'].astype('str') + data['direction'].astype('str') + data['x'].astype('str') + data['y'].astype('str')
    data['hour+x'] = data['hour'].astype('str') + data['x'].astype('str')
    data['hour+y'] = data['hour'].astype('str') + data['y'].astype('str')
    data['is_holiday'] = data['time'].dt.date.apply(is_holiday)
#     data['time_id'] = ( ( (data['time'].dt.dayofyear-1)*24*60 + data['time'].dt.hour*60 + data['time'].dt.minute ) / 20 ).astype(int)
#     data['is_day_before_holiday'] = train_data['time'].dt.date.apply(is_day_before_holiday)
#     data['is_day_after_holiday'] = train_data['time'].dt.date.apply(is_day_after_holiday)
    data = data.drop(['time'], axis=1)
    return data


# In[14]:


get_ipython().run_cell_magic('time', '', '\nfor data in [train_data, test_data]:\n    data = feature_engineering(data)\n')


# # 1. Task definition

# ### 1.1. Task type
# 
# On the cell below we create Task object - the class to setup what task LightAutoML model should solve with specific loss and metric if necessary (more info can be found [here](https://lightautoml.readthedocs.io/en/latest/generated/lightautoml.tasks.base.Task.html#lightautoml.tasks.base.Task) in our documentation):

# In[15]:


task = Task('reg', metric='mae', loss='mae')


# ### 1.2. Feature roles setup
# To solve the task, we need to setup columns roles. The **only role you must setup is target role**, everything else (drop, numeric, categorical, group, weights etc.) is up to user - LightAutoML models have automatic columns typization inside:

# In[16]:


roles = {'target': TARGET_NAME,
         'drop': ['row_id']
         }


# ### 1.3. LightAutoML model creation - TabularAutoML preset

# In next the cell we are going to create LightAutoML model with `TabularAutoML` class - preset with default model structure like in the image below:
# 
# <img src="https://github.com/sberbank-ai-lab/LightAutoML/raw/master/imgs/tutorial_blackbox_pipeline.png" alt="TabularAutoML preset pipeline" style="width:85%;"/>
# 
# in just several lines. Let's discuss the params we can setup:
# - `task` - the type of the ML task (the only **must have** parameter)
# - `timeout` - time limit in seconds for model to train
# - `cpu_limit` - vCPU count for model to use
# - `reader_params` - parameter change for Reader object inside preset, which works on the first step of data preparation: automatic feature typization, preliminary almost-constant features, correct CV setup etc. For example, we setup `n_jobs` threads for typization algo, `cv` folds and `random_state` as inside CV seed.
# 
# **Important note**: `reader_params` key is one of the YAML config keys, which is used inside `TabularAutoML` preset. [More details](https://github.com/sberbank-ai-lab/LightAutoML/blob/master/lightautoml/automl/presets/tabular_config.yml) on its structure with explanation comments can be found on the link attached. Each key from this config can be modified with user settings during preset object initialization. To get more info about different parameters setting (for example, ML algos which can be used in `general_params->use_algos`) please take a look at our [article on TowardsDataScience](https://towardsdatascience.com/lightautoml-preset-usage-tutorial-2cce7da6f936).
# 
# Moreover, to receive the automatic report for our model we will use `ReportDeco` decorator and work with the decorated version in the same way as we do with usual one. 

# In[17]:


get_ipython().run_cell_magic('time', '', "\nautoml = TabularUtilizedAutoML(task = task,\n                               timeout = TIMEOUT,\n                               cpu_limit = N_THREADS,\n                               reader_params = {'n_jobs': N_THREADS, 'random_state': RANDOM_STATE},\n                               tuning_params = {'max_tuning_iter': 20, 'max_tuning_time': 30},\n                               general_params = {'use_algos': [['linear_l2', 'lgb', 'lgb_tuned']]}\n                              )\n")


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

# In[18]:


oof_pred = automl.fit_predict(train_data, roles = roles, verbose=3)
print(f'oof_pred:\n{oof_pred}\nShape = {oof_pred.shape}')


# In[19]:


get_ipython().run_cell_magic('time', '', "\nfast_fi = automl.get_feature_scores('fast')\nfast_fi.set_index('Feature')['Importance'].plot.bar(figsize=(20, 10), grid=True)\n")


# # 4. Predict for test dataset

# In[20]:


get_ipython().run_cell_magic('time', '', "\ntest_pred = automl.predict(test_data)\nprint(f'Prediction for test data:\\n{test_pred}\\nShape = {test_pred.shape}')\n")


# In[21]:


submission[TARGET_NAME] = test_pred.data[:, 0]
submission


# And don't forget to round your predictions like in @AmbrosM's notebook: [TPSMAR22](https://www.kaggle.com/ambrosm/tpsmar22-don-t-forget-to-round)

# In[22]:


submission[TARGET_NAME] = submission[TARGET_NAME].round().astype(int)
assert (submission[TARGET_NAME] >= 0).all()
assert (submission[TARGET_NAME] <= 100).all()


# Save submissions to .csv

# In[23]:


submission.to_csv('lightautoml.csv', index=False)


# In[24]:


submission.describe()


# # Additional materials

# - [Official LightAutoML github repo](https://github.com/sberbank-ai-lab/LightAutoML)
# - [LightAutoML documentation](https://lightautoml.readthedocs.io/en/latest)
# - [LightAutoML tutorials](https://github.com/sberbank-ai-lab/LightAutoML/tree/master/examples/tutorials)
# - LightAutoML course:
#     - [Part 1 - general overview](https://ods.ai/tracks/automl-course-part1) 
#     - [Part 2 - LightAutoML specific applications](https://ods.ai/tracks/automl-course-part2)
#     - [Part 3 - LightAutoML customization](https://ods.ai/tracks/automl-course-part3)
# - [OpenDataScience AutoML benchmark leaderboard](https://ods.ai/competitions/automl-benchmark/leaderboard)

# ### If you still like the notebook, do not forget to put upvote for the notebook and the ⭐️ for github repo if you like it using the button below - one click for you, great pleasure for us ☺️

# In[25]:


s = '<iframe src="https://ghbtns.com/github-btn.html?user=sberbank-ai-lab&repo=LightAutoML&type=star&count=true&size=large" frameborder="0" scrolling="0" width="170" height="30" title="LightAutoML GitHub"></iframe>'
HTML(s)


# In[26]:


# Plot the distribution of the test predictions
# compared to the other Monday afternoons
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
plt.figure(figsize=(16,3))
plt.hist(train_data.congestion[((train_data.time.dt.weekday == 0) &
                           (train_data.time.dt.hour >= 12)).values],
         bins=np.linspace(-0.5, 100.5, 102),
         density=True, label='Train',
         color='#ffd700')
plt.hist(submission['congestion'], np.linspace(-0.5, 100.5, 102),
         density=True, rwidth=0.5, label='Test predictions',
         color='r')
plt.xlabel('Congestion')
plt.ylabel('Frequency')
plt.title('Congestion on Monday afternoons')
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
plt.legend()
plt.show()


# In[ ]:




