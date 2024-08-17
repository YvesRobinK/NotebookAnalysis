#!/usr/bin/env python
# coding: utf-8

# 

# **Hi Guys!**
# 
# For those, who want to play with **automated feature engineering** - please find below small example of how to do it with [featuretools](https://docs.featuretools.com/)
# 
# Featuretools is a framework to perform automated feature engineering. 
# <br>It (at least it's stated that) excels at transforming transactional and relational datasets into feature matrices for machine learning.

# In[132]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[133]:


# import library for automated feature engineering
import featuretools as ft
import gc
from os.path import join as pjoin
from os import cpu_count
import warnings
warnings.simplefilter('ignore')


# In[202]:


# define filepaths
data_dir = '../input'

filepaths = {
    'data_desc': pjoin(data_dir, 'HomeCredit_columns_description.csv'),
    'app_train': pjoin(data_dir, 'application_train.csv'),
    'app_test': pjoin(data_dir, 'application_test.csv'),
    'bureau': pjoin(data_dir, 'bureau.csv'),
    'bureau_bl': pjoin(data_dir, 'bureau_balance.csv'),
    'credit_bl': pjoin(data_dir, 'credit_card_balance.csv'),
    'install_pays': pjoin(data_dir, 'installments_payments.csv'),
    'pc_balance': pjoin(data_dir, 'POS_CASH_balance.csv'),
    'app_prev': pjoin(data_dir, 'previous_application.csv'),
    
}

filepaths


# ### Load Main table (Applications)

# In[203]:


# first X rows are taken for faster calculations, substitute this by whole dataset
nrows = 30000

# load main datasets
df_train = pd.read_csv(
    filepaths['app_train'], 
    low_memory=False, engine='c',
    nrows=nrows,
)
df_test = pd.read_csv(
    filepaths['app_test'], 
    low_memory=False, 
    engine='c',
)

# concat dataframes together, check shapes
print(df_train.shape, df_test.shape)
df_joint = pd.concat([df_train, df_test])
print(df_joint.shape)

del df_train, df_test
gc.collect()

print('memory usage: {:.2f} MB'.format(df_joint.memory_usage().sum() / 2**20))

int_cols = df_joint.select_dtypes(include=[np.int64]).columns
float_cols = df_joint.select_dtypes(include=[np.float64]).columns 

df_joint[int_cols] = df_joint[int_cols].astype(np.int32)
df_joint[float_cols] = df_joint[float_cols].astype(np.float32)

print('memory usage: {:.2f} MB'.format(df_joint.memory_usage().sum() / 2**20))

print(df_joint.dtypes.value_counts())

# df_joint.set_index('SK_ID_CURR', inplace=True, drop=True)
target_col = 'TARGET'

# check sample
df_joint.head()


# ### Load previous applications table

# In[204]:


df_app_prev = pd.read_csv(
    filepaths['app_prev'], 
    engine='c', 
    low_memory=False,
    # first X*3 rows are taken for faster calculations, substitute this by whole dataset
    nrows=nrows*3,
)
print(df_app_prev.shape)

# optimize memory usage
print('memory usage: {:.2f} MB'.format(df_app_prev.memory_usage().sum() / 2**20))

int_cols = df_app_prev.select_dtypes(include=[np.int64]).columns
float_cols = df_app_prev.select_dtypes(include=[np.float64]).columns 

df_app_prev[int_cols] = df_app_prev[int_cols].astype(np.int32)
df_app_prev[float_cols] = df_app_prev[float_cols].astype(np.float32)

print('memory usage: {:.2f} MB'.format(df_app_prev.memory_usage().sum() / 2**20))
print(df_app_prev.dtypes.value_counts())

# check sample
df_app_prev.head()


# In[205]:


# substitute encoded NaNs by "real" np.nan
df_app_prev[
    [c for c in df_app_prev.columns if c.startswith('DAYS_')]
] = df_app_prev[
    [c for c in df_app_prev.columns if c.startswith('DAYS_')]
].replace(365243, np.nan)


# ## TABLE JOINING (FEATURE TOOLS)

# An `EntitySet` is a collection of entities and the relationships between them. 
# 
# They are useful for preparing raw, structured datasets for feature engineering. 
# <br>While many functions in Featuretools take `entities` and `relationships` as separate arguments,
# <br>it is recommended to create an `EntitySet`, so you can more easily manipulate your data as needed.

# In[206]:


# initialize entityset
es = ft.EntitySet('application_data')

# add entities (application table itself)
es.entity_from_dataframe(
    entity_id='apps', # define entity id
    dataframe=df_joint.drop('TARGET', axis=1), # select underlying data
    index='SK_ID_CURR', # define unique index column
    # specify some datatypes manually (if needed)
    variable_types={
        f: ft.variable_types.Categorical 
        for f in df_joint.columns if f.startswith('FLAG_')
    }
)


# In[207]:


# trick! substitute relative days by absolute date shift
# to be used as "true" time_index
# however, prohibit datetime features (like month or year) as they're irrelevant
today = pd.to_datetime('2018-06-11')
df_app_prev['DAYS_DECISION'] = today + pd.to_timedelta(df_app_prev['DAYS_DECISION'], unit='d')


# In[208]:


# add entities (previous applications table)
es = es.entity_from_dataframe(
    entity_id = 'prev_apps', 
    dataframe = df_app_prev,
    index = 'SK_ID_PREV',
    time_index = 'DAYS_DECISION',
    variable_types={
        f: ft.variable_types.Categorical 
        for f in df_app_prev.columns if f.startswith('NFLAG_')
    }
)


# In the call to `entity_from_dataframe`, we specified three important parameters
# 
# - The `index` parameter specifies the column that uniquely identifies rows in the dataframe
# - The `time_index` parameter tells Featuretools when the data was "created".
# - The `variable_types` parameter indicates that some columns should be interpreted as a Categorical variable, 
# even though it just an integer in the underlying data.
# 

# **Adding a Relationship**
# 
# With two entities in our entity set, we can add a relationship between them.
# 
# We want to relate these two entities by the columns called “SK_ID_CURR” in each entity. 
# <br>Each application has multiple previous applications associated with it, 
# <br>so it is called it the parent entity, while the previous applications  entity is known as the child entity. 
# <br>When specifying relationships we list the variable in the parent entity first. Note that each ft.Relationship must denote a **one-to-many relationship** rather than a relationship which is one-to-one or many-to-many.

# In[209]:


# add relationships
r_app_cur_to_app_prev = ft.Relationship(
    es['apps']['SK_ID_CURR'],
    es['prev_apps']['SK_ID_CURR']
)

# Add the relationship to the entity set
es = es.add_relationship(r_app_cur_to_app_prev)

# check constructed entity set
es


# In[210]:


# check created entities
es['apps']


# In[211]:


# check created entities
es['prev_apps']


# **Feature primitives**
# 
# Feature primitives are the building blocks of Featuretools. They define individual computations that can be applied to raw datasets to create new features. Because a primitive only constrains the input and output data types, they can be applied across datasets and can stack to create new calculations.
# 
# **Why primitives?**
# 
# The space of potential functions that humans use to create a feature is expansive. By breaking common feature engineering calculations down into primitive components, we are able to capture the underlying structure of the features humans create today.
# 
# See [documentation](https://docs.featuretools.com/automated_feature_engineering/primitives.html) for further details

# In[212]:


# inspect list of all built-in primitives for feature construction
ft.list_primitives()


# **Handling time**
# 
# When performing feature engineering to learn a model to predict the future, 
# <br>the value to predict will be associated with a time. 
# In this case, it is paramount to only incorporate data prior to this `“cutoff time”` when calculating the feature values.
# 
# Featuretools is designed to take time into consideration when required. 
# <br>By specifying a cutoff time, we can control what portions of the data are used when calculating features.
# 
# We can specify the time for each instance of the `target_entity` to calculate features. 
# <br>The timestamp represents the last time data can be used for calculating features. This is specified using a dataframe of cutoff times. 
# 
# Read more [here](https://docs.featuretools.com/automated_feature_engineering/handling_time.html)

# In[213]:


get_ipython().run_cell_magic('time', '', '# define cut-off times\n# cut-off times are the "right" time values to be used for feature calculation without future leaks\n# none in our case\n\ncutoff_times = pd.DataFrame(df_joint.SK_ID_CURR)\ncutoff_times[\'time\'] = today\n\n# add last_time_index\nes.add_last_time_indexes()\n')


# **Running DFS with training windows**
# 
# Training windows are an extension of cutoff times: starting from the cutoff time and moving backwards through time, only data within that window of time will be used to calculate features. We will use events **within 2 month time window**

# In[214]:


# see feature set definitions (no actual computations yet)
# used for faster prototyping
feature_defs = ft.dfs(
    entityset=es, 
    target_entity="apps", 
    features_only=True,
    agg_primitives=[
        "avg_time_between",
        "time_since_last", 
        "num_unique", 
        "mean", 
        "sum", 
    ],
    trans_primitives=[
        "time_since_previous",
        #"add",
    ],
    max_depth=1,
    cutoff_time=cutoff_times,
    training_window=ft.Timedelta(60, "d"), # use only last X days in computations
    max_features=1000,
    chunk_size=10000,
    verbose=True,
)

# check what's been created so far
feature_defs


# In[215]:


# calculate actual features
fm, feature_defs = ft.dfs(
    entityset=es, 
    target_entity="apps", 
    #features_only=True,
    agg_primitives=[
        "avg_time_between",
        "time_since_last", 
        "num_unique", 
        "mean", 
        "sum", 
    ],
    trans_primitives=[
        "time_since_previous",
        #"add",
    ],
    max_depth=1,
    cutoff_time=cutoff_times,
    training_window=ft.Timedelta(60, "d"),
    max_features=1000,
    chunk_size=4000,
    verbose=True,
)


# In[216]:


# check sample of extracted features
fm = fm.drop_duplicates()
print(fm.shape)
fm[50:100]


# In[217]:


# define validation strategy and run a model atop of generated features
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

skf = StratifiedKFold(5, random_state=42)

print(fm.dtypes.value_counts())
# label-encode categorical variables
for c in fm.select_dtypes(include=['object']).columns:
    fm[c], _ = pd.factorize(fm[c])

print(fm.dtypes.value_counts())


# In[218]:


# define train/test datasets
idx_train = df_joint[~df_joint.TARGET.isnull()].SK_ID_CURR.tolist()
idx_test = df_joint[df_joint.TARGET.isnull()].SK_ID_CURR.tolist()

fm_train = fm[fm.index.isin(idx_train)]
fm_test = fm[fm.index.isin(idx_test)]
fm_train.shape, fm_test.shape


# In[219]:


import lightgbm as lgb

# define lightgbm params
params_lgb = {
    'application': 'binary',
    'boosting': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'max_depth': 7,
    'early_stopping_round': 10,
    
    'num_iteration': 2500, 
    'colsample_bytree':.95, 
    'subsample':.87, 

    'reg_alpha': 0.04, 
    'reg_lambda': 0.07, 
    'min_split_gain': 0.022, 
    'min_child_weight': 5,
}

# make dataset
data_tr = lgb.Dataset(
    data=fm_train,
    label=df_joint[:nrows][target_col],
)

# run cross-validation
cv_results = lgb.cv(
    params_lgb, 
    data_tr, 
    metrics=['auc'], 
    folds=skf.split(fm_train, df_joint[:nrows][target_col]), 
    verbose_eval=25,
)


# In[220]:


get_ipython().run_cell_magic('time', '', "# train model\nparams_lgb['num_iteration'] = int(len(cv_results['auc-mean']) * 5/4)\n\nmodel = lgb.train(\n    params_lgb, \n    data_tr,  \n)\n")


# In[223]:


# predict for test
df_joint.loc[df_joint.SK_ID_CURR.isin(idx_test), target_col] = model.predict(fm_test)

# sample submission
df_joint.loc[df_joint.SK_ID_CURR.isin(idx_test), ['SK_ID_CURR', target_col]].to_csv(
    'featuretools_example_subm.csv',
    index=False
)


# **Hope this small example inspired you to try this approach yourself! 
# <br>Have fun - add custom features, add more tables and relationships, gather hands on experience**
# 
# **Likes and comments are welcome :)**

# In[ ]:




