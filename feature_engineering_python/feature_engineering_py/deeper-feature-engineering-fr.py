#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import cohen_kappa_score
import gc
pd.set_option('display.max_columns', 1000)


# # Objective
# * In the last notebook we make a feature selection function which increase our cohen kappa score
# * In this notebook we will create more features and check if they are usefull
# * In this same notebook i will try more and more features to increase the score, stay tunned.
# 
# Link for the past notebook is here: https://www.kaggle.com/ragnar123/lgbm-feature-selection
# 
# Link for the baseline is here: https://www.kaggle.com/ragnar123/simple-exploratory-data-analysis-and-model

# In[2]:


def read_data():
    print('Reading train.csv file....')
    train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    print('Reading test.csv file....')
    test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    print('Reading train_labels.csv file....')
    train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))

    print('Reading specs.csv file....')
    specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))

    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))
    return train, test, train_labels, specs, sample_submission


# In[3]:


# feature engineering functions
def get_time(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    return df

def get_object_columns(df, columns):
    df = df.groupby(['installation_id', columns])['event_id'].count().reset_index()
    df = df.pivot_table(index = 'installation_id', columns = [columns], values = 'event_id')
    df.columns = list(df.columns)
    df.fillna(0, inplace = True)
    return df

def get_numeric_columns(df, column):
    df = df.groupby('installation_id').agg({f'{column}': ['mean', 'sum', 'std']})
    df.fillna(0, inplace = True)
    df.columns = [f'{column}_mean', f'{column}_sum', f'{column}_std']
    return df

def get_numeric_columns_2(df, agg_column, column):
    df = df.groupby(['installation_id', agg_column]).agg({f'{column}': ['mean', 'sum', 'std']}).reset_index()
    df = df.pivot_table(index = 'installation_id', columns = [agg_column], values = [col for col in df.columns if col not in ['installation_id', 'type']])
    df.fillna(0, inplace = True)
    df.columns = list(df.columns)
    return df

def get_correct_incorrect(df):
    df = df.groupby(['title'])['num_correct', 'num_incorrect'].agg({'num_correct': ['mean', 'std'], 'num_incorrect': ['mean', 'std']}).reset_index()
    df.columns = ['title', 'num_correct_mean', 'num_correct_std', 'num_incorrect_mean', 'num_incorrect_std']
    return df


# In[4]:


def preprocess(train, test, train_labels):
    # columns for feature engineering
    numerical_columns = ['game_time', 'event_count']
    categorical_columns = ['type', 'world']
    numerical_columns_single = ['hour', 'dayofweek', 'month', 'event_id_count', 'event_code_count']

    reduce_train = pd.DataFrame({'installation_id': train['installation_id'].unique()})
    reduce_train.set_index('installation_id', inplace = True)
    reduce_test = pd.DataFrame({'installation_id': test['installation_id'].unique()})
    reduce_test.set_index('installation_id', inplace = True)
    
    # get time features
    train = get_time(train)
    test = get_time(test)
    
    def count_segments(train, test, cols):
        for col in cols:
            for df in [train, test]:
                df[f'{col}_count'] = df.groupby([col])['timestamp'].transform('count')
        return train, test
    
    count_segments(train, test, ['event_id', 'event_code'])
    
    
    
    for i in numerical_columns:
        reduce_train = reduce_train.merge(get_numeric_columns(train, i), left_index = True, right_index = True)
        reduce_test = reduce_test.merge(get_numeric_columns(test, i), left_index = True, right_index = True)
        
    for i in categorical_columns:
        reduce_train = reduce_train.merge(get_object_columns(train, i), left_index = True, right_index = True)
        reduce_test = reduce_test.merge(get_object_columns(test, i), left_index = True, right_index = True)
            
    for i in categorical_columns:
        for j in numerical_columns:
            reduce_train = reduce_train.merge(get_numeric_columns_2(train, i, j), left_index = True, right_index = True)
            reduce_test = reduce_test.merge(get_numeric_columns_2(test, i, j), left_index = True, right_index = True)
            
    for i in numerical_columns_single:
        reduce_train = reduce_train.merge(get_numeric_columns(train, i), left_index = True, right_index = True)
        reduce_test = reduce_test.merge(get_numeric_columns(test, i), left_index = True, right_index = True)
            
    reduce_train.reset_index(inplace = True)
    reduce_test.reset_index(inplace = True)
    
    print('Our training set have {} rows and {} columns'.format(reduce_train.shape[0], reduce_train.shape[1]))
    
    # get the mode of the title
    labels_map = dict(train_labels.groupby('title')['accuracy_group'].agg(lambda x:x.value_counts().index[0]))
    # merge target
    labels = train_labels[['installation_id', 'title', 'accuracy_group']]
    # merge with correct incorrect
    corr_inc = get_correct_incorrect(train_labels)
    labels = labels.merge(corr_inc, how = 'left', on = 'title')
    # replace title with the mode
    labels['title'] = labels['title'].map(labels_map)
    # get title from the test set
    reduce_test['title'] = test.groupby('installation_id').last()['title'].reset_index(drop = True)
    # merge with correct incorrect
    reduce_test = reduce_test.merge(corr_inc, how = 'left', on = 'title')
    # map title
    reduce_test['title'] = reduce_test['title'].map(labels_map)
    # join train with labels
    reduce_train = labels.merge(reduce_train, on = 'installation_id', how = 'left')
    print('We have {} training rows'.format(reduce_train.shape[0]))
    # align datasets
    categoricals = ['title']
    reduce_train = reduce_train[[col for col in reduce_test.columns] + ['accuracy_group']]
    return reduce_train, reduce_test, categoricals


# In[5]:


# best features extracted from run_feature_selection function
usefull_features = ['num_correct_mean', 'num_correct_std', 'num_incorrect_mean', 'num_incorrect_std', 'game_time_mean', 'game_time_sum', 'Activity', 'Clip', 'Game', 'CRYSTALCAVES', 'NONE', 'TREETOPCITY', ('game_time', 'mean', 'Clip'), 
                    ('game_time', 'mean', 'Game'), ('game_time', 'std', 'Assessment'), ('game_time', 'std', 'Clip'), ('game_time', 'std', 'Game'), ('game_time', 'sum', 'Activity'), ('game_time', 'sum', 'Clip'), ('game_time', 'sum', 'Game'), 
                    ('game_time', 'mean', 'NONE'), ('game_time', 'mean', 'TREETOPCITY'), ('game_time', 'std', 'CRYSTALCAVES'), ('game_time', 'std', 'MAGMAPEAK'), ('game_time', 'std', 'NONE'), ('game_time', 'std', 'TREETOPCITY'), 
                    ('game_time', 'sum', 'CRYSTALCAVES'), ('game_time', 'sum', 'MAGMAPEAK'), ('game_time', 'sum', 'NONE'), 'title']
new_features = ['event_count_mean','event_count_sum', 'event_count_std', ('event_count', 'mean', 'Activity'), ('event_count', 'mean', 'Assessment'), ('event_count', 'mean', 'Clip'), ('event_count', 'mean', 'Game'), ('event_count', 'std', 'Activity'), 
                ('event_count', 'std', 'Assessment'), ('event_count', 'std', 'Clip'), ('event_count', 'std', 'Game'), ('event_count', 'sum', 'Activity'), ('event_count', 'sum', 'Assessment'), ('event_count', 'sum', 'Clip'),
                ('event_count', 'sum', 'Game'),  ('event_count', 'mean', 'CRYSTALCAVES'), ('event_count', 'mean', 'MAGMAPEAK'), ('event_count', 'mean', 'NONE'), ('event_count', 'mean', 'TREETOPCITY'), ('event_count', 'std', 'CRYSTALCAVES'),
                ('event_count', 'std', 'MAGMAPEAK'), ('event_count', 'std', 'NONE'), ('event_count', 'std', 'TREETOPCITY'), ('event_count', 'sum', 'CRYSTALCAVES'), ('event_count', 'sum', 'MAGMAPEAK'),
                ('event_count', 'sum', 'NONE'), ('event_count', 'sum', 'TREETOPCITY'), 'hour_mean', 'hour_sum', 'hour_std', 'dayofweek_mean', 'dayofweek_sum', 'dayofweek_std', 'event_id_count_mean', 'event_id_count_sum', 'event_id_count_std', 
                'event_code_count_mean', 'event_code_count_sum']


# In[6]:


# feature selection function that detects if a feature increase cohen kappa score, if it does we add it to the features pool of our model
def run_feature_selection(reduce_train, reduce_test, categoricals, usefull_features, new_features):
    kf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 42)
    target = 'accuracy_group'
    oof_pred = np.zeros((len(reduce_train), 4))
    for fold, (tr_ind, val_ind) in enumerate(kf.split(reduce_train, reduce_train[target])):
        print('Fold {}'.format(fold + 1))
        x_train, x_val = reduce_train[usefull_features].iloc[tr_ind], reduce_train[usefull_features].iloc[val_ind]
        y_train, y_val = reduce_train[target][tr_ind], reduce_train[target][val_ind]
        train_set = lgb.Dataset(x_train, y_train, categorical_feature = categoricals)
        val_set = lgb.Dataset(x_val, y_val, categorical_feature = categoricals)

        params = {
                'learning_rate': 0.01,
                'metric': 'multi_error',
                'objective': 'multiclass',
                'num_classes': 4,
                'feature_fraction': 0.75,
                'subsample': 0.75,
                'n_jobs': -1,
                'seed': 50,
                'max_depth': 10
            }

        model = lgb.train(params, train_set, num_boost_round = 100000, early_stopping_rounds = 100, 
                          valid_sets=[train_set, val_set], verbose_eval = 100)
        oof_pred[val_ind] = model.predict(x_val)
    # using cohen_kappa because it's the evaluation metric of the competition
    loss_score = cohen_kappa_score(reduce_train[target], np.argmax(oof_pred, axis = 1), weights = 'quadratic')
    score = loss_score
    usefull_new_features = []
    for i in new_features:
        print('Our best cohen kappa score is : ', score)
        oof_pred = np.zeros((len(reduce_train), 4))
        features = usefull_features + usefull_new_features + [i]
        print('Evaluating {} column'.format(i))
        for fold, (tr_ind, val_ind) in enumerate(kf.split(reduce_train, reduce_train[target])):
            print('Fold {}'.format(fold + 1))
            x_train, x_val = reduce_train[features].iloc[tr_ind], reduce_train[features].iloc[val_ind]
            y_train, y_val = reduce_train[target][tr_ind], reduce_train[target][val_ind]
            train_set = lgb.Dataset(x_train, y_train, categorical_feature = categoricals)
            val_set = lgb.Dataset(x_val, y_val, categorical_feature = categoricals)

            model = lgb.train(params, train_set, num_boost_round = 100000, early_stopping_rounds = 100,
                              valid_sets=[train_set, val_set], verbose_eval = 500)
            oof_pred[val_ind] = model.predict(x_val)
        loss_score = cohen_kappa_score(reduce_train[target], np.argmax(oof_pred, axis = 1), weights = 'quadratic')
        print('Our new kohen cappa score is : ', loss_score)
        if loss_score > score:
            print('Feature {} is usefull'.format(i))
            usefull_new_features.append(i)
            score = loss_score
        else:
            print('Feature {} is useless'.format(i))
        gc.collect()
    print('The best features are: ', usefull_features + usefull_new_features)

    return usefull_features + usefull_new_features


# In[7]:


# load data
train, test, train_labels, specs, sample_submission = read_data()
# preprocess 
reduce_train, reduce_test, categoricals = preprocess(train, test, train_labels)


# In[8]:


# check if new features are usefull, in that case add them to our feature pool
usefull_features = run_feature_selection(reduce_train, reduce_test, categoricals, usefull_features, new_features)


# In[9]:


usefull_features


# In[10]:


def run_lgb(reduce_train, reduce_test, usefull_features):
    kf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 42)
    target = 'accuracy_group'
    oof_pred = np.zeros((len(reduce_train), 4))
    y_pred = np.zeros((len(reduce_test), 4))
    for fold, (tr_ind, val_ind) in enumerate(kf.split(reduce_train, reduce_train[target])):
        print('Fold {}'.format(fold + 1))
        x_train, x_val = reduce_train[usefull_features].iloc[tr_ind], reduce_train[usefull_features].iloc[val_ind]
        y_train, y_val = reduce_train[target][tr_ind], reduce_train[target][val_ind]
        train_set = lgb.Dataset(x_train, y_train, categorical_feature=categoricals)
        val_set = lgb.Dataset(x_val, y_val, categorical_feature=categoricals)

        params = {
            'learning_rate': 0.01,
            'metric': 'multi_error',
            'objective': 'multiclass',
            'num_classes': 4,
            'feature_fraction': 0.75,
            'subsample': 0.75,
            'n_jobs': -1,
            'seed': 50,
            'max_depth': 10
        }

        model = lgb.train(params, train_set, num_boost_round = 1000000, early_stopping_rounds = 100, 
                          valid_sets=[train_set, val_set], verbose_eval = 100)
        oof_pred[val_ind] = model.predict(x_val)
        y_pred += model.predict(reduce_test[usefull_features]) / 10
    loss_score = cohen_kappa_score(reduce_train[target], np.argmax(oof_pred, axis = 1), weights = 'quadratic')
    print('Our oof cohen kappa score is: ', loss_score)
    return y_pred

def predict(reduce_test, sample_submission, y_pred):
    reduce_test = reduce_test.reset_index()
    reduce_test = reduce_test[['installation_id']]
    reduce_test['accuracy_group'] = y_pred.argmax(axis = 1)
    sample_submission.drop('accuracy_group', inplace = True, axis = 1)
    sample_submission = sample_submission.merge(reduce_test, on = 'installation_id')
    sample_submission.to_csv('submission.csv', index = False)
    print(sample_submission['accuracy_group'].value_counts(normalize = True))
y_pred = run_lgb(reduce_train, reduce_test, usefull_features)
predict(reduce_test, sample_submission, y_pred)

