#!/usr/bin/env python
# coding: utf-8

# ## General information
# 
# In this competition, we will create algorithms for "Knowledge Tracing," the modeling of student knowledge over time. The goal is to accurately predict how students will perform on future interactions.
# If successful, it’s possible that any student with an Internet connection can enjoy the benefits of a personalized learning experience, regardless of where they live. We can build a better and more equitable model for education in a post-COVID-19 world.
# 
# Our challenge in this competition is to predict whether students are able to answer their next questions correctly.
# 
# This competition is similar to Two Sigma competition, where we got test data using special API.
# 
# ![](https://i.imgur.com/Ko5MxFQ.png)

# ### importing libraries

# In[1]:


# libraries
import riiideducation
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os
from typing import List, Dict, Optional
import numpy as np
from sklearn.model_selection import RepeatedKFold
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import time
import random
import lightgbm as lgb
import gc
import os
from sklearn.preprocessing import LabelEncoder
from numba import jit
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn import metrics


# ### helper functions

# In[2]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
    

@jit
def fast_auc(y_true, y_prob):
    """
    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def eval_auc(y_true, y_pred):
    """
    Fast auc eval function for lgb.
    """
    return 'auc', fast_auc(y_true, y_pred), True


# ## Data overview

# In[3]:


get_ipython().system('wc -l /kaggle/input/riiid-test-answer-prediction/train.csv')


# More than 100m rows in train dataset!

# In[4]:


path = '/kaggle/input'

train = pd.read_csv(f'{path}/riiid-test-answer-prediction/train.csv',
                    usecols=['timestamp', 'user_id', 'content_id', 'content_type_id', 'user_answer', 'answered_correctly',
                             'prior_question_elapsed_time', 'prior_question_had_explanation'],
                       dtype={'timestamp': 'int64',
                              'user_id': 'int32',
                              'content_id': 'int16',
                              'content_type_id': 'int8',
                              'user_answer': 'int8',
                              'answered_correctly': 'int8',
                              'prior_question_elapsed_time': 'float32', 
                              'prior_question_had_explanation': 'boolean',
                             }
                      )
train = train.sort_values(['timestamp'], ascending=True)
questions = pd.read_csv(f'{path}/riiid-test-answer-prediction/questions.csv')
lectures = pd.read_csv(f'{path}/riiid-test-answer-prediction/lectures.csv')
print('Train shapes: ', train.shape)


# In[5]:


train.head()


# In[6]:


train['answered_correctly'].value_counts()


# `answered_correctly` is our target! `-1` is a special value, we'll talk about it later.

# In[7]:


questions.head()


# In[8]:


lectures.head()


# ## Exploring the features

# ### timestamp
# 
# It is imprtant to remember that this is the time between this **user** interaction and the first event from that **user**. So starting time could be different for each user

# In[9]:


plt.hist(train['timestamp'], bins=40);


# In[10]:


train.groupby(['user_id'])['timestamp'].max().sort_values(ascending=False).head()


# Some users have really huge activity time!

# ### user_id
# 
# Obviously this is an unique user id.

# In[11]:


train['user_id'].value_counts()


# There are 1749 unique user ids in out data. Some have little data, some have a lot of data. But please notice that I loaded only a small part of the data.

# ### content_type_id
# 
# 0 if the event was a question being posed to the user, 1 if the event was the user watching a lecture.
# 
# So we will be mainly using `content_type_id` equal to `1` at first, but generating some features based on the lectures should be also useful.

# In[12]:


train['content_type_id'].value_counts()


# In[13]:


train.loc[train['content_type_id'] == 1, 'user_id'].nunique()


# Hm, interesting. It seems that not all people watched lectures.

# ### content_id
# 
# Id of the content - question or lecture

# In[14]:


train['content_id'].value_counts()


# Does low numbers mean that only one person answered this question? I wonder why there are such question.

# Let's have a look at the most popular question

# In[15]:


train.loc[train['content_id'] == 6116]


# In[16]:


train.loc[train['content_id'] == 6116, 'user_answer'].value_counts()


# In[17]:


questions.loc[questions['question_id'] == 6116]


# We can see that a lot of people made mistakes answering this question.

# I'll continue EDA later.

# ## Feature engineering
# 
# Let's generate more features.
# 
# some code is taken from https://www.kaggle.com/ilialar/simple-eda-and-baseline https://www.kaggle.com/lgreig/simple-lgbm-baseline

# In[18]:


# filter out lectures
train = train.loc[train['answered_correctly'] != -1].reset_index(drop=True)
train = train.drop(['timestamp','content_type_id'], axis=1)
train['prior_question_had_explanation'] = train['prior_question_had_explanation'].fillna(value = False).astype(bool)


# In[19]:


user_answers_df = train.groupby('user_id').agg({'answered_correctly': ['mean', 'count']}).copy()
user_answers_df.columns = ['mean_user_accuracy', 'questions_answered']

content_answers_df = train.groupby('content_id').agg({'answered_correctly': ['mean', 'count']}).copy()
content_answers_df.columns = ['mean_accuracy', 'question_asked']

# user_content_answers_df = train.groupby(['user_id', 'content_id']).agg({'answered_correctly': ['mean', 'count']}).copy()
# user_content_answers_df.columns = ['mean_user_content_accuracy', 'content_questions_answered']


# Now we will use only a part of data for training, to avoid leaks and memory error

# In[20]:


train = train.iloc[90000000:,:]


# In[21]:


train = train.merge(user_answers_df, how = 'left', on = 'user_id')
train = train.merge(content_answers_df, how = 'left', on = 'content_id')
# train = train.merge(user_content_answers_df, how = 'left', on = ['user_id', 'content_id'])


# In[22]:


train.fillna(value = 0.5, inplace = True)


# In[23]:


# train['mean_diff1'] = train['mean_user_accuracy'] - train['mean_user_content_accuracy']
# train['mean_diff2'] = train['mean_accuracy'] - train['mean_user_content_accuracy']


# In[24]:


train.head()


# In[25]:


le = LabelEncoder()
train["prior_question_had_explanation"] = le.fit_transform(train["prior_question_had_explanation"])


# In[26]:


train = train.sort_values(['user_id'])


# In[27]:


y = train['answered_correctly']

columns = ['mean_user_accuracy', 'questions_answered', 'mean_accuracy', 'question_asked',
           'prior_question_had_explanation',# 'mean_diff1', 'mean_diff2', 'mean_user_content_accuracy'
          ]
X = train[columns]


# In[28]:


del train


# In[29]:


scores = []
feature_importance = pd.DataFrame()
models = []


# In[30]:


params = {'num_leaves': 32,
          'max_bin': 300,
          'objective': 'binary',
          'max_depth': 13,
          'learning_rate': 0.03,
          "boosting_type": "gbdt",
          "metric": 'auc',
         }


# In[31]:


columns = ['mean_user_accuracy', 'questions_answered', 'mean_accuracy', 'question_asked',
#            'prior_question_had_explanation', 'mean_diff1', 'mean_diff2'
          ]


# In[32]:


folds = StratifiedKFold(n_splits=5, shuffle=False)
for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
    print(f'Fold {fold_n} started at {time.ctime()}')
    X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    model = lgb.LGBMClassifier(**params, n_estimators=700, n_jobs = 1)
    model.fit(X_train, y_train, 
            eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=eval_auc,
            verbose=1000, early_stopping_rounds=10)
    score = max(model.evals_result_['valid_1']['auc'])
    
    models.append(model)
    scores.append(score)

    fold_importance = pd.DataFrame()
    fold_importance["feature"] = columns
    fold_importance["importance"] = model.feature_importances_
    fold_importance["fold"] = fold_n + 1
    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
    break


# In[33]:


print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))


# In[34]:


feature_importance["importance"] /= 1
cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
    by="importance", ascending=False)[:50].index

best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

plt.figure(figsize=(16, 12));
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
plt.title('LGB Features (avg over folds)');


# In[35]:


del X, y


# ## Making predictions.
# 
# Code is taken from https://www.kaggle.com/sishihara/riiid-lgbm-5cv-benchmark

# In[36]:


env = riiideducation.make_env()


# In[37]:


iter_test = env.iter_test()


# In[38]:


for (test_df, sample_prediction_df) in iter_test:
    y_preds = []
    test_df = test_df.merge(user_answers_df, how = 'left', on = 'user_id')
    test_df = test_df.merge(content_answers_df, how = 'left', on = 'content_id')
#     test_df = test_df.merge(user_content_answers_df, how = 'left', on = ['user_id', 'content_id'])
#     test_df['mean_diff1'] = test_df['mean_user_accuracy'] - test_df['mean_user_content_accuracy']
#     test_df['mean_diff2'] = test_df['mean_accuracy'] - test_df['mean_user_content_accuracy']
    test_df['prior_question_had_explanation'] = test_df['prior_question_had_explanation'].fillna(value = False).astype(bool)
    test_df = test_df.loc[test_df['content_type_id'] == 0].reset_index(drop=True)
    test_df.fillna(value = 0.5, inplace = True)
    test_df["prior_question_had_explanation_enc"] = le.fit_transform(test_df["prior_question_had_explanation"])

    for model in models:
        y_pred = model.predict_proba(test_df[columns], num_iteration=model.best_iteration_)[:, 1]
        y_preds.append(y_pred)

    y_preds = sum(y_preds) / len(y_preds)
    test_df['answered_correctly'] = y_preds
    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])

