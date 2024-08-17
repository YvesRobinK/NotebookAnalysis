#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import seaborn as sns
import matplotlib.pyplot as plt


# This notebook contains an elementary analysis of 'train.csv' and rule based baseline. I will further develop it during the competition.
# Here I analyze the distributions of different features, generate aggregeted features based on 90% of historical data and then train the model on the rest 10%.

# My other materials on this competition:
# - Double validation (against target leakage): https://www.kaggle.com/ilialar/riiid-5-folds-double-validation
# - Dataset with pretrained models and feature generators: https://www.kaggle.com/ilialar/riiid-models

# # Train data

# The `train.csv` file is too large for kaggle kernel. You will get a memory error if you try to load it all without specifying types. We will ignore some columns for now to save RAM and load only 10M rows.

# In[2]:


data_types_dict = {
    'row_id': 'int64',
    'timestamp': 'int64',
    'user_id': 'int32',
    'content_id': 'int16',
    'content_type_id': 'int8',
#     'task_container_id': 'int16',
#     'user_answer': 'int8',
    'answered_correctly': 'int8',
    'prior_question_elapsed_time': 'float16',
    'prior_question_had_explanation': 'boolean'
}


# In[3]:


train_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', 
                       nrows=10**7,
                       usecols = data_types_dict.keys(),
                       dtype=data_types_dict, 
                       index_col = 0)


# Let's look at the data and main columns properties:

# In[4]:


train_df.head(10)


# In[5]:


train_df.info()


# In[6]:


train_df.describe()


# Let's look at some columns more precisely.

# ## timestamp

# In[7]:


train_df['timestamp'].hist(bins = 100)


# `timestamp` represents the time from the first user interaction to the current one. It is expected that the distribution looks like this.

# In[8]:


grouped_by_user_df = train_df.groupby('user_id')


# In[9]:


grouped_by_user_df.agg({'timestamp': 'max'}).hist(bins = 100)

The distribution of the max timestamp for each user looks similar. It seems most users leave the platform quite soon (at least based on partial data we analyze).
# ## Answered correctly

# In[10]:


(train_df['answered_correctly']==-1).mean()


# ~2% of activities are lectures, we should exclude them for answers analysis.

# In[11]:


train_questions_only_df = train_df[train_df['answered_correctly']!=-1]
train_questions_only_df['answered_correctly'].mean()


# On average users answer ~66% questions correctly. Let's look how it is different from user to user.

# ### Answers by users

# In[12]:


grouped_by_user_df = train_questions_only_df.groupby('user_id')


# In[13]:


user_answers_df = grouped_by_user_df.agg({'answered_correctly': ['mean', 'count'] })

user_answers_df[('answered_correctly','mean')].hist(bins = 100)


# Look's noisy, let's clear it a little bit

# In[14]:


user_answers_df[('answered_correctly','count')].hist(bins = 100)


# In[15]:


(user_answers_df[('answered_correctly','count')]< 50).mean()


# 54% of users answered less than 50 questions. Let's divide all users into novices and active users.

# In[16]:


user_answers_df[user_answers_df[('answered_correctly','count')]< 50][('answered_correctly','mean')].mean()


# In[17]:


user_answers_df[user_answers_df[('answered_correctly','count')]< 50][('answered_correctly','mean')].hist(bins = 100)


# In[18]:


user_answers_df[user_answers_df[('answered_correctly','count')] >= 50][('answered_correctly','mean')].hist(bins = 100)


# In[19]:


user_answers_df[user_answers_df[('answered_correctly','count')] >= 50][('answered_correctly','mean')].mean()


# We can see that active users do much better than novices. But anyway average user score is lower than the overall % of correct answers. It means heavy users have even better scores. Let's look at them.

# In[20]:


user_answers_df[user_answers_df[('answered_correctly','count')] >= 500][('answered_correctly','mean')].hist(bins = 100)


# In[21]:


plt.scatter(x = user_answers_df[('answered_correctly','count')], y=user_answers_df[ ('answered_correctly','mean')])


# Timestamp, the average score for the active user, and the number of questions answered can be useful for baseline.

# ### Answers by content

# In[22]:


grouped_by_content_df = train_questions_only_df.groupby('content_id')


# In[23]:


content_answers_df = grouped_by_content_df.agg({'answered_correctly': ['mean', 'count'] })


# In[24]:


content_answers_df[('answered_correctly','count')].hist(bins = 100)


# In[25]:


content_answers_df[('answered_correctly','mean')].hist(bins = 100)


# Different questions have different popularity and complexity, and it can also be used in the baseline.

# In[26]:


content_answers_df[content_answers_df[('answered_correctly','count')]>50][('answered_correctly','mean')].hist(bins = 100)


# # Questions.csv

# Let's look into the `questions.csv` file

# In[27]:


questions_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/questions.csv')


# In[28]:


questions_df


# In[29]:


print(f"There are {len(questions_df['part'].unique())} different parts")


# In[30]:


questions_df['tags'].values[-1]


# In[31]:


unique_tags = set().union(*[y.split() for y in questions_df['tags'].astype(str).values])
print(f"There are {len(unique_tags)} different tags")


# In[32]:


(questions_df['question_id'] != questions_df['bundle_id']).mean()


# We can create aggregated features using the data from this file as well.

# # Baseline

# Let's try to use discovered features and use them in model to predict the right answer probability.

# In[33]:


train_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',
                       usecols = data_types_dict.keys(),
                       dtype=data_types_dict, 
                       index_col = 0)


# In[34]:


features_part_df = train_df.iloc[:int(9 /10 * len(train_df))]
train_part_df = train_df.iloc[int(9 /10 * len(train_df)):]


# In[35]:


train_questions_only_df = features_part_df[features_part_df['answered_correctly']!=-1]
grouped_by_user_df = train_questions_only_df.groupby('user_id')
user_answers_df = grouped_by_user_df.agg({'answered_correctly': ['mean', 'count']}).copy()
user_answers_df.columns = ['mean_user_accuracy', 'questions_answered']


# In[36]:


grouped_by_content_df = train_questions_only_df.groupby('content_id')
content_answers_df = grouped_by_content_df.agg({'answered_correctly': ['mean', 'count'] }).copy()
content_answers_df.columns = ['mean_accuracy', 'question_asked']


# Let's create additional features using `questions_df`

# In[37]:


questions_df = questions_df.merge(content_answers_df, left_on = 'question_id', right_on = 'content_id', how = 'left')


# In[38]:


bundle_dict = questions_df['bundle_id'].value_counts().to_dict()


# In[39]:


questions_df['right_answers'] = questions_df['mean_accuracy'] * questions_df['question_asked']
questions_df['bundle_size'] =questions_df['bundle_id'].apply(lambda x: bundle_dict[x])


# In[40]:


questions_df


# In[41]:


grouped_by_bundle_df = questions_df.groupby('bundle_id')
bundle_answers_df = grouped_by_bundle_df.agg({'right_answers': 'sum', 'question_asked': 'sum'}).copy()
bundle_answers_df.columns = ['bundle_rignt_answers', 'bundle_questions_asked']
bundle_answers_df['bundle_accuracy'] = bundle_answers_df['bundle_rignt_answers'] / bundle_answers_df['bundle_questions_asked']
bundle_answers_df


# In[42]:


grouped_by_part_df = questions_df.groupby('part')
part_answers_df = grouped_by_part_df.agg({'right_answers': 'sum', 'question_asked': 'sum'}).copy()
part_answers_df.columns = ['part_rignt_answers', 'part_questions_asked']
part_answers_df['part_accuracy'] = part_answers_df['part_rignt_answers'] / part_answers_df['part_questions_asked']
part_answers_df


# In[43]:


del train_df
del features_part_df
del grouped_by_user_df
del grouped_by_content_df


# In[44]:


import gc
gc.collect()


# In[45]:


features = ['timestamp','mean_user_accuracy', 'questions_answered','mean_accuracy', 'question_asked',
            'prior_question_elapsed_time', 'prior_question_had_explanation',
           'bundle_size', 'bundle_accuracy','part_accuracy', 'right_answers']
target = 'answered_correctly'


# In[46]:


train_part_df = train_part_df[train_part_df[target] != -1]


# In[47]:


train_part_df = train_part_df.merge(user_answers_df, how = 'left', on = 'user_id')
train_part_df = train_part_df.merge(questions_df, how = 'left', left_on = 'content_id', right_on = 'question_id')
train_part_df = train_part_df.merge(bundle_answers_df, how = 'left', on = 'bundle_id')
train_part_df = train_part_df.merge(part_answers_df, how = 'left', on = 'part')


# In[48]:


train_part_df['prior_question_had_explanation'] = train_part_df['prior_question_had_explanation'].fillna(value = False).astype(bool)
train_part_df.fillna(value = -1, inplace = True)


# In[49]:


train_part_df.columns


# In[50]:


train_part_df = train_part_df[features + [target]]


# In[51]:


train_part_df


# In[52]:


from sklearn.metrics import roc_auc_score


# In[53]:


from lightgbm import LGBMClassifier


# In[54]:


lgbm = LGBMClassifier(
    num_leaves=31, 
    max_depth= 2, 
    n_estimators = 25, 
    min_child_samples = 1000, 
    subsample=0.7, 
    subsample_freq=5,
    n_jobs= -1,
    is_higher_better = True,
    first_metric_only = True
)


# In[55]:


lgbm.fit(train_part_df[features], train_part_df[target])


# In[56]:


roc_auc_score(train_part_df[target].values, lgbm.predict_proba(train_part_df[features])[:,1])


# In[57]:


import riiideducation

env = riiideducation.make_env()


# In[58]:


iter_test = env.iter_test()


# In[59]:


for (test_df, sample_prediction_df) in iter_test:
    test_df = test_df.merge(user_answers_df, how = 'left', on = 'user_id')
    test_df = test_df.merge(questions_df, how = 'left', left_on = 'content_id', right_on = 'question_id')
    test_df = test_df.merge(bundle_answers_df, how = 'left', on = 'bundle_id')
    test_df = test_df.merge(part_answers_df, how = 'left', on = 'part')
    
    test_df['prior_question_had_explanation'] = test_df['prior_question_had_explanation'].fillna(value = False).astype(bool)
    test_df.fillna(value = -1, inplace = True)

    test_df['answered_correctly'] = lgbm.predict_proba(test_df[features])[:,1]
    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])

