#!/usr/bin/env python
# coding: utf-8

# <h1>Riid AIEd Challenge 2020 - Part III - Feature Engineering</h1>
# 
# Due to memory/time restrictions in this competition, work is divided into several parts (kernels):
# 
# <lu>
#     <li>Part I - Memory optimization</li>
#     <li>Part II - Splitting data</li>
#     <li>Part III - Feature engineering</li>
#     <li>Part IV - Training and validation</li>
#     <li>Part V - Prediction and submission</li>
# </lu>
# 
# 
# This is Part III. In this part I'll
# 
# Create user and content features from past data (saved in previous kernel) and a History class responsible for expanding a data frame with the format of the training or test sets (columns user_id, content_id, etc.) with user and content features, and of updating history with the new interactions information.

# In[1]:


# Imports

import os
import pandas as pd
import numpy as np
import pickle
import gc
import warnings
import lightgbm as lgb


# In[2]:


warnings.filterwarnings(action='ignore')


# <h2>Load data</h2>

# In[3]:


# Define directories used

DATA_DIR = '/kaggle/input/riiid-test-answer-prediction'
PART_II_OUTPUT_DIR = '/kaggle/input/riiid-aied-part-ii-splitting/'
WORKING_DIR = '/kaggle/working'


# In[4]:


get_ipython().run_cell_magic('time', '', "\n# Load the competition data \npast_data = pd.read_pickle(os.path.join(PART_II_OUTPUT_DIR, 'past_data.pkl'))\npast_data.head()\n")


# In[5]:


past_data.memory_usage()


# In[6]:


# As we need enough memory for feature engineering, let's get rid of unneccesary columns
drop_columns = ['timestamp', 'user_answer', 
                'prior_question_elapsed_time', 'prior_question_had_explanation', 'virtual_timestamp']
past_data.drop(columns=drop_columns, inplace=True)

_ = gc.collect()

past_data.info()


# <h2>Build history dataframes</h2>

# <h3>Question performance features</h3>

# In[7]:


# Read questions meta data

question_types = {
    'question_id': np.int16,
    'bundle_id':np.int16,
    'correct_answer':np.int8,
    'part':np.int8,
    'tags':'object'
}

questions = pd.read_csv(os.path.join(DATA_DIR, 'questions.csv'), dtype=question_types)
questions.set_index('question_id', drop=True, inplace=True)
questions.head()


# In[8]:


get_ipython().run_cell_magic('time', '', "\nquestion_performance_columns = [\n    'q_mean', 'part'\n]\n\nquestion_performance = past_data[past_data.content_type_id == False].groupby('content_id')['answered_correctly'].agg(['mean']).astype(np.float32)\nquestion_performance = pd.merge(question_performance, questions['part'], left_index=True, right_index=True)\nquestion_performance.columns = question_performance_columns\n\n_ = gc.collect()\n\nquestion_performance.head()\n")


# In[9]:


question_performance.info()


# <h2>Task container features</h2>

# In[10]:


task_container_performance_columns = [
    'tc_mean'
]

task_container_performance = past_data.groupby('task_container_id')[['answered_correctly']].agg('mean').astype(np.float32)
task_container_performance.columns = task_container_performance_columns

_ = gc.collect()

task_container_performance.head()


# <h3>User performance features</h3>

# In[11]:


user_performance_columns = [
    'u_count', 'u_correct', 'u_mean'
]

user_performance = past_data.groupby('user_id')['answered_correctly'].agg(['count', 'sum']).astype(np.int32)
user_performance['u_mean'] = (user_performance['sum'] / user_performance['count']).astype(np.float32)
user_performance.columns = user_performance_columns

_ = gc.collect()

user_performance.head()


# In[12]:


user_performance.info()


# In[13]:


get_ipython().run_cell_magic('time', '', "past_data = pd.concat([past_data.reset_index(drop=True), questions['part'].reindex(past_data.content_id.values).reset_index(drop=True)], axis=1)\n_ = gc.collect()\n")


# <h2>Build history class</h2>

# In[14]:


class History:
    def __init__(self, user_performance, question_performance, task_container_performance):
        self.user_performance = user_performance
        self.question_performance = question_performance
        self.task_container_performance = task_container_performance
        
    def expand_features(self, df):
        '''
        Expand dataframe df with features from history. 
        '''
        
        expanded_df = pd.concat([df.reset_index(drop=True), 
                            self.user_performance.reindex(df.user_id.values).reset_index(drop=True),
                            self.question_performance.reindex(df.content_id.values).reset_index(drop=True),
                            self.task_container_performance.reindex(df.task_container_id.values).reset_index(drop=True)], axis=1)
         
        expanded_df.fillna(0.5, inplace=True)
        
        expanded_df['uq_hmean'] = 2 * expanded_df['u_mean'] * expanded_df['q_mean'] / (expanded_df['u_mean'] + expanded_df['q_mean'])
        expanded_df['utc_hmean'] = 2 * expanded_df['u_mean'] * expanded_df['tc_mean'] / (expanded_df['u_mean'] + expanded_df['tc_mean'])
        
        return expanded_df
        
    def update_features_df(self, df):
        new_users_ids = set(df.user_id).difference(self.user_performance.index.values)
        if new_users_ids:
            new_users_df = pd.DataFrame(0, index=new_users_ids, columns=self.user_performance.columns)
            self.user_performance = pd.concat([self.user_performance, new_users_df], axis='rows')
    
        user_update = df.groupby('user_id', sort=False)['answered_correctly'].agg(['count', 'sum'])  
        self.user_performance.loc[user_update.index, ['u_count', 'u_correct']] += user_update.values
        self.user_performance['u_mean'] = self.user_performance.uq_correct / self.user_performance.uq_count
        


# This class is tested and profiled using the competition example test set in another <a href='https://www.kaggle.com/jcesquiveld/riiid-aied-optimize-history-class'>kernel</a>

# <h2>Save everything for next Part</h2>

# In[15]:


# Save history object 

history = History(user_performance, question_performance, task_container_performance)
filehandler = open(os.path.join(WORKING_DIR, 'past_history.pkl'), 'wb') 
pickle.dump(history, filehandler)
filehandler.close()


# <h2>Proof of concept</h2>
# 
# Test features with lightgbm model with default parameters

# In[16]:


df = pd.DataFrame(data=[[115, 0, 0, 1], [2746, 25, 1, 3], [5382, 4, 2, 3]], columns=['user_id', 'content_id', 'task_container_id', 'part'])
df.head()


# In[17]:


get_ipython().run_cell_magic('time', '', 'history.expand_features(df)\n')


# In[18]:


proof_of_concept = True

gc.collect()

# Set the hyper parameters for the booster
params = {
    'objective': 'binary',
    'seed': 42,
    'metric': 'auc',
}

# Set the features used
FEATURES = [
    'prior_question_had_explanation', 'prior_question_elapsed_time',
    'u_count', 
    'u_mean',
    'uq_hmean',
    'q_mean',
    'tc_mean',
    'utc_hmean',
    'part'
]

# Mean for prior_question_elapsed_time (from memory optimization notebook)
prior_question_elapsed_time_mean = 25423.810042960275

if proof_of_concept:
    # Read data for one of the folds
    train = pd.read_pickle(os.path.join(PART_II_OUTPUT_DIR, 'train_0.pkl'))
    val = pd.read_pickle(os.path.join(PART_II_OUTPUT_DIR, 'val_0.pkl'))
    
    # Preprocessing
    train = train.loc[train.content_type_id == False]
    train['prior_question_had_explanation'] = train.prior_question_had_explanation.fillna(False).astype(np.int8)
    train['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, inplace=True)
    train = history.expand_features(train)
    val = val.loc[val.content_type_id == False]
    val['prior_question_had_explanation'] = val.prior_question_had_explanation.fillna(False).astype(np.int8)
    val['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, inplace=True)
    val = history.expand_features(val)
    
    # Datasets
    lgb_train = lgb.Dataset(train[FEATURES], train['answered_correctly'])
    lgb_val = lgb.Dataset(val[FEATURES], val['answered_correctly'])
    
    # Train
    model = lgb.train(
            params,
            lgb_train,
            valid_sets = [lgb_train, lgb_val],
            verbose_eval = 100,
            num_boost_round = 10000,
            early_stopping_rounds = 50
        )


# In[19]:


lgb.plot_importance(model, importance_type='split', figsize=(6,10))


# In[20]:


lgb.plot_importance(model, importance_type='gain', figsize=(6,10))


# That's all folks!!!
