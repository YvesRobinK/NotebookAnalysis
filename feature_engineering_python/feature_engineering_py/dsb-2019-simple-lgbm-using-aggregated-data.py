#!/usr/bin/env python
# coding: utf-8

# # About this notebook
# 
# You might have noticed that the train dataset is composed of over 11M data points, but there are only 17k training labels, and 1000k test labels you are predicting. The reason for that is there are many thousand different entries for each `installation_id`, each representing an `event`. This notebook simply gathers all the events into 17k groups, each group corresponds to an `installation_id`. Then, it takes the aggregation (using sums, counts, mean, std, etc.) of those groups, thus resulting in a dataset of summary statistics of each `installation_id`. After that, it simply fits a model on that dataset.
# 
# ## Updates
# 
# V20:
# * Updated variable names for clarity.
# 
# V17:
# * Removed statistics on event codes, since that created a lot of columns and LGBM seems to overfit on that information.
# 
# V16:
# * Added mode of title `accuracy_group` (retrieved from training set) as a feature
# 
# V10:
# * Fixed labelling problem. Before that, I was blindly predicting the target without even the title I was trying to assess 🤦. I added that now by using the "title" column from `train_labels.csv`, and using the last row of each installation_id from `test.csv` to construct a `test_labels` dataframe.
# 
# V8: 
# * Added `cv_train`, a function that trains k-models on each of k-fold CV splits. Then, you can use function `cv_predict` to use the list of models to predict an output (and blend the results).
# * Added more summary statistics for `event_code` and `game_time`, including skewness of the distribution.
# 
# ## References
# 
# * CV idea inspired from [this kernel](https://www.kaggle.com/tanreinama/ds-bowl-2019-simple-lgbm-aggregated-data-with-cv). Thank you!
# * Adding mode as a feature: https://www.kaggle.com/mhviraf/a-baseline-for-dsb-2019

# In[1]:


import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lightgbm as lgb
import scipy as sp
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from tqdm import tqdm


# In[2]:


tqdm.pandas()


# # Load Data

# In[3]:


get_ipython().run_cell_magic('time', '', "# Only load those columns in order to save space\nkeep_cols = ['event_id', 'game_session', 'installation_id', 'event_count', 'event_code', 'title', 'game_time', 'type', 'world']\n\ntrain = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv', usecols=keep_cols)\ntest = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv', usecols=keep_cols)\ntrain_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')\nsubmission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')\n")


# In[4]:


test_assess = test[test.type == 'Assessment'].copy()
test_labels = submission.copy()
test_labels['title'] = test_labels.installation_id.progress_apply(
    lambda install_id: test_assess[test_assess.installation_id == install_id].iloc[-1].title
)


# # Group and Reduce

# In[5]:


def compute_game_time_stats(group, col):
    return group[
        ['installation_id', col, 'event_count', 'game_time']
    ].groupby(['installation_id', col]).agg(
        [np.mean, np.sum, np.std]
    ).reset_index().pivot(
        columns=col,
        index='installation_id'
    )


# In[6]:


def group_and_reduce(df, df_labels):
    """
    Author: https://www.kaggle.com/xhlulu/
    Source: https://www.kaggle.com/xhlulu/ds-bowl-2019-simple-lgbm-using-aggregated-data
    """
    
    # First only filter the useful part of the df
    df = df[df.installation_id.isin(df_labels.installation_id.unique())]
    
    # group1 is am intermediary "game session" group,
    # which are reduced to one record by game session. group_game_time takes
    # the max value of game_time (final game time in a session) and 
    # of event_count (total number of events happened in the session).
    group_game_time = df.drop(columns=['event_id', 'event_code']).groupby(
        ['game_session', 'installation_id', 'title', 'type', 'world']
    ).max().reset_index()

    # group3, group4 are grouped by installation_id 
    # and reduced using summation and other summary stats
    title_group = (
        pd.get_dummies(
            group_game_time.drop(columns=['game_session', 'event_count', 'game_time']),
            columns=['title', 'type', 'world'])
        .groupby(['installation_id'])
        .sum()
    )

    event_game_time_group = (
        group_game_time[['installation_id', 'event_count', 'game_time']]
        .groupby(['installation_id'])
        .agg([np.sum, np.mean, np.std, np.min, np.max])
    )
    
    # Additional stats on group1
    world_time_stats = compute_game_time_stats(group_game_time, 'world')
    type_time_stats = compute_game_time_stats(group_game_time, 'type')
    
    return (
        title_group.join(event_game_time_group)
        .join(world_time_stats)
        .join(type_time_stats)
        .fillna(0)
    )


# In[7]:


get_ipython().run_cell_magic('time', '', 'train_small = group_and_reduce(train, train_labels)\ntest_small = group_and_reduce(test, test_labels)\n\nprint(train_small.shape)\ntrain_small.head()\n')


# ## Adding mode as feature

# In[8]:


def create_title_mode(train_labels):
    titles = train_labels.title.unique()
    title2mode = {}

    for title in titles:
        mode = (
            train_labels[train_labels.title == title]
            .accuracy_group
            .value_counts()
            .index[0]
        )
        title2mode[title] = mode
    return title2mode

def add_title_mode(labels, title2mode):
    labels['title_mode'] = labels.title.apply(lambda title: title2mode[title])
    return labels


# In[9]:


title2mode = create_title_mode(train_labels)
train_labels = add_title_mode(train_labels, title2mode)
test_labels = add_title_mode(test_labels, title2mode)


# ## Combine train/test labels with summary stats

# In[10]:


def preprocess_train(train_labels, last_records_only=True):
    """
    last_records_only (bool): Use only the last record of each user.
    """
    final_train = pd.get_dummies(
        (
            train_labels.set_index('installation_id')
            .drop(columns=['num_correct', 'num_incorrect', 'accuracy', 'game_session'])
            .join(train_small)
        ), 
        columns=['title']
    )
    
    if last_records_only:
        final_train = (
            final_train
            .reset_index()
            .groupby('installation_id')
            .apply(lambda x: x.iloc[-1])
            .drop(columns='installation_id')
        )
    
    return final_train

def preprocess_test(test_labels, test_small):
    return pd.get_dummies(
        test_labels.set_index('installation_id').join(test_small), columns=['title']
    )


# In[11]:


final_train = preprocess_train(train_labels)
print(final_train.shape)
final_train.head()


# In[12]:


final_test = preprocess_test(test_labels, test_small)
print(final_test.shape)
final_test.head()


# # Training model

# In[13]:


def cv_train(X, y, cv, **kwargs):
    """
    Author: https://www.kaggle.com/xhlulu/
    Source: https://www.kaggle.com/xhlulu/ds-bowl-2019-simple-lgbm-using-aggregated-data
    """
    models = []
    
    kf = KFold(n_splits=cv, random_state=2019)
    
    for train, test in kf.split(X):
        x_train, x_val, y_train, y_val = X[train], X[test], y[train], y[test]
        
        train_set = lgb.Dataset(x_train, y_train)
        val_set = lgb.Dataset(x_val, y_val)
        
        model = lgb.train(train_set=train_set, valid_sets=[train_set, val_set], **kwargs)
        models.append(model)
        
        if kwargs.get("verbose_eval"):
            print("\n" + "="*50 + "\n")
    
    return models

def cv_predict(models, X):
    return np.mean([model.predict(X) for model in models], axis=0)


# In[14]:


X = final_train.drop(columns='accuracy_group').values
y = final_train['accuracy_group'].values

params = {
    'learning_rate': 0.01,
    'bagging_fraction': 0.9,
    'feature_fraction': 0.2,
    'max_height': 3,
    'lambda_l1': 10,
    'lambda_l2': 10,
    'metric': 'multiclass',
    'objective': 'multiclass',
    'num_classes': 4,
    'random_state': 2019
}

models = cv_train(X, y, cv=20, params=params, num_boost_round=1000,
                  early_stopping_rounds=100, verbose_eval=500)


# # Submission

# In[15]:


X_test = final_test.drop(columns=['accuracy_group'])
test_pred = cv_predict(models=models, X=X_test).argmax(axis=1)

final_test['accuracy_group'] = test_pred
final_test[['accuracy_group']].to_csv('submission.csv')


# # Visualize Model

# In[16]:


for model in models:
    lgb.plot_importance(model, max_num_features=15, height=0.3)


# In[17]:


plt.hist(test_pred)

