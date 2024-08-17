#!/usr/bin/env python
# coding: utf-8

# # This is a notebook for preprocessing.
# - **inference notebook**: https://www.kaggle.com/tkyiws/single-lgb-model-with-about-23-features
# 
# Thank you very much for your big help.
# - https://www.kaggle.com/ldevyataykina/riiid-exploratory-data-analysis-baseline?scriptVersionId=48691010  
# - https://www.kaggle.com/shoheiazuma/riiid-lgbm-starter  
# - https://www.kaggle.com/markwijkhuizen/riiid-training-and-prediction-using-a-state  
# - https://www.kaggle.com/its7171/time-series-api-iter-test-emulator

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from tqdm import tqdm
import gc
import pickle
import joblib

pd.set_option('display.max_columns', 50)


# In[2]:


data_types_dict = {
    'row_id': 'int64',
    'timestamp': 'int64',
    'user_id': 'int32',
    'content_id': 'int16',
    'content_type_id': 'int8',
    'task_container_id': 'int16',
    'answered_correctly':'int8',
    'prior_question_elapsed_time': 'float32',
    'prior_question_had_explanation': 'boolean'
}
                   
target = 'answered_correctly'

features_dtypes = {
    'content_id': 'int16',
    'content_mean': 'float32',
    'prior_question_elapsed_time': 'float64',
    'prior_question_had_explanation': 'bool',
    'user_correctness': 'float32',
    'content_count': 'int32',
    'part': 'int8',
    'cumcount_u': 'uint16',
    'cumcount_p': 'uint16',
    'attempt': 'uint16',
    'part_avg': 'float32',
    'timestamp_diff1': 'float64',
    'timestamp_diff2': 'float64',
    'cluster_id': 'int8',
    'cluster_avg': 'float32',
    'cumcount_cl': 'uint16',
    'target_lag': 'int8',
    'cluster0_avg': 'float32',
    'cluster1_avg': 'float32',
    'cluster2_avg': 'float32',
    'prior_tag': 'int16',
    'task_num': 'int8',
    'user_rating': 'float32',
    'time_mean_diff': 'float32',
}


# # Data Loading

# In[3]:


train_df = pd.read_csv('../input/riiid-test-answer-prediction/train.csv',
                       usecols=[0, 1, 2, 3, 4, 5, 7, 8, 9],
                       dtype=data_types_dict,
                       nrows=10_000_000, # Some data will be used due to RAM constraints.
                      )
questions_df = pd.read_csv(
    '../input/riiid-test-answer-prediction/questions.csv', 
    usecols=[0, 1, 3],
    dtype={'question_id': 'int16', 'bundle_id': 'int16', 'part': 'int8'}
)

lectures_df = pd.read_csv('../input/riiid-test-answer-prediction/lectures.csv')


# # prior_tag (1/2)
# Tags from a last-minute lecture.
# It will be reset if you take a series of lectures or answer a question.(Tag number or -1)

# In[4]:


lectures_df['content_type_id'] = 1
lectures_df.columns = ['content_id', 'lecture_tag', 'lecture_part', 'type_of', 'content_type_id']
lectures_df = lectures_df[['content_id', 'lecture_tag', 'content_type_id']].astype({'content_id': 'int16', 'lecture_tag': 'int16', 'content_type_id': 'int8'})
# lectures_df.to_pickle('riiid_pre_data/lectures_df.pickle')
# lectures_df = pd.read_pickle('riiid_pre_data/lectures_df.pickle')

train_df = pd.merge(train_df, lectures_df, on=['content_id', 'content_type_id'], how='left')
train_df['lecture_tag'].fillna(-1, inplace=True)
train_df['prior_tag'] = train_df.groupby('user_id')['lecture_tag'].shift()
train_df['prior_tag'].fillna(-1, inplace=True)

last_lecture_dict = train_df.groupby('user_id').tail(1)[['user_id', 'lecture_tag']].set_index('user_id')['lecture_tag'].astype('int16').to_dict()
# joblib.dump(last_lecture_dict, "dict_data/last_lecture_dict.pkl.zip")
# last_lecture_dict = joblib.load("dict_data/last_lecture_dict.pkl.zip")


# In[5]:


train_df.loc[89:91, ['user_id', 'content_type_id', 'lecture_tag', 'prior_tag']]


# In[6]:


print('After the lecture:', train_df[(train_df[target]!=-1)&(train_df['prior_tag']!=-1)][target].mean())
print('-1               :', train_df[(train_df[target]!=-1)&(train_df['prior_tag']==-1)][target].mean())


# The questions about the last lecture I took are simple.

# In[7]:


plt.figure(figsize=(10,5))
plt.subplot(121)
plt.title('Top 10 correct answers by prior_tag')
train_df.groupby('prior_tag')[target].mean().sort_values().iloc[-10:].plot.barh()
plt.subplot(122)
plt.title('Worst 10 correct answers by prior_tag')
train_df.groupby('prior_tag')[target].mean().sort_values(ascending=False).iloc[-10:].plot.barh()
plt.show()
plt.figure(figsize=(10,5))
plt.title('number of occurances')
train_df[train_df['prior_tag']!=-1]['prior_tag'].value_counts().iloc[:30].plot.bar();


# For the full data, the least number of occurrences is 366.

# #  preprocessing (1/2)

# In[8]:


train_df = train_df[train_df[target] != -1].reset_index(drop=True)
train_df['prior_question_had_explanation'].fillna(False, inplace=True)
train_df = train_df.astype(data_types_dict)

# Delete test users.
# Intentionally? Deletes users who have answered the same question incorrectly in succession.
train_df = train_df.drop(index=train_df[train_df['user_id']==1509564249].index).reset_index(drop=True)

# task_num: Number of content_ids that share the task_container_id.
questions_df['task_num'] = questions_df['bundle_id'].map(questions_df.groupby('bundle_id')['question_id'].nunique())
questions_df.drop(columns=['bundle_id'], inplace=True)


# # cluster_id
# Based on the percentage of correct answers, median, standard deviation, and skewness of content_id, we clustered "content_id" into three classes.

# In[9]:


cluster_data = pd.read_pickle('../input/sc-cluster-data/sc_cluster_data.pickle')
questions_df['cluster_id'] = questions_df['question_id'].map(cluster_data)
del cluster_data

# questions_df.to_pickle('riiid_pre_data/questions_df.pickle')
# questions_df = pd.read_pickle('riiid_pre_data/questions_df.pickle')

train_df = pd.merge(train_df, questions_df, left_on='content_id', right_on='question_id', how='left')
train_df.drop(columns=['question_id'], inplace=True)


# In[10]:


train_df.groupby('cluster_id')[target].agg(['mean', 'count'])


# # timestamp_diff
# The time between the completion of the last event and the completion of the current event.

# In[11]:


timestamp_df= train_df.groupby(['user_id', 'task_container_id']).head(1)[['user_id', 'task_container_id', 'timestamp']]

timestamp_df['timestamp_diff1'] = timestamp_df.groupby('user_id')['timestamp'].diff()
timestamp_df['timestamp_diff2'] = timestamp_df.groupby('user_id')['timestamp'].diff(2)

# time_dict1 = timestamp_df.groupby('user_id')['timestamp'].max().to_dict()
# timestamp_df['timestamp'] = timestamp_df.groupby('user_id')['timestamp'].shift()
# time_dict2 = timestamp_df.groupby('user_id')['timestamp'].max().to_dict()
# timestamp_df['timestamp'] = timestamp_df.groupby('user_id')['timestamp'].shift()
# time_dict3 = timestamp_df.groupby('user_id')['timestamp'].max().to_dict()

timestamp_df.drop(columns=['timestamp'], inplace=True)

train_df = pd.merge(train_df, timestamp_df, on=['user_id', 'task_container_id'], how='left')

del timestamp_df
gc.collect()


# In[12]:


# joblib.dump(time_dict1, "riiid_pre_data/time_dict1.pkl.zip")
# joblib.dump(time_dict2, "riiid_pre_data/time_dict2.pkl.zip")
# joblib.dump(time_dict3, "riiid_pre_data/time_dict3.pkl.zip")

# time_dict1 = joblib.load("riiid_pre_data/time_dict1.pkl.zip")
# time_dict2 = joblib.load("riiid_pre_data/time_dict2.pkl.zip")
# time_dict3 = joblib.load("riiid_pre_data/time_dict3.pkl.zip")


# In[13]:


# Divide by task_num.
train_df['timestamp_diff1'] = train_df['timestamp_diff1'] / train_df['task_num']
train_df['timestamp_diff2'] = train_df['timestamp_diff2'] / train_df['task_num']


# In[14]:


train_df[train_df['user_id']==124][['timestamp', 'user_id', 'task_container_id', 'timestamp_diff1', 'timestamp_diff2']].head(7)


# # part_avg
# Cumulative average per "part".
# # cluster_avg
# Cumulative average per "cluster_id".

# In[15]:


train_df['lag'] = train_df.groupby('user_id')[target].shift()
cum = train_df.groupby('user_id')['lag'].agg(['cumsum', 'cumcount'])
train_df['cumcount_u'] = cum['cumcount']
train_df['user_correctness'] = cum['cumsum'] / cum['cumcount']
train_df.drop(columns=['lag'], inplace=True)

train_df['lag'] = train_df.groupby(['user_id', 'part'])[target].shift()
cum = train_df.groupby(['user_id', 'part'])['lag'].agg(['cumsum', 'cumcount'])
train_df['cumcount_p'] = cum['cumcount']
train_df['part_avg'] = cum['cumsum'] / cum['cumcount']
train_df.drop(columns=['lag'], inplace=True)

train_df['lag'] = train_df.groupby(['user_id', 'cluster_id'])[target].shift()
cum = train_df.groupby(['user_id', 'cluster_id'])['lag'].agg(['cumsum', 'cumcount'])
train_df['cumcount_cl'] = cum['cumcount']
train_df['cluster_avg'] = cum['cumsum'] / cum['cumcount']
train_df.drop(columns=['lag'], inplace=True)


# In[16]:


# Share task_container_id.
df_ = train_df.groupby(['user_id', 'task_container_id']).head(1)[['user_id', 'task_container_id', 'user_correctness', 'cumcount_u', 'part_avg', 'cumcount_p']]
train_df.drop(columns=['user_correctness', 'part_avg', 'cumcount_u', 'cumcount_p'], inplace=True)
train_df = pd.merge(train_df, df_, on=['user_id', 'task_container_id'], how='left')

df_ = train_df.groupby(['user_id', 'task_container_id', 'cluster_id']).head(1)[['user_id', 'task_container_id', 'cluster_id', 'cluster_avg', 'cumcount_cl']]
train_df.drop(columns=['cluster_avg', 'cumcount_cl'], inplace=True)
train_df = pd.merge(train_df, df_, on=['user_id', 'task_container_id', 'cluster_id'], how='left')

del cum, df_
gc.collect()


# In[17]:


train_df[train_df['user_id']==124][['user_id', 'task_container_id', target, 'part', 'part_avg', 'cluster_id', 'cluster_avg']].head(7)


# # preprocessing (2/2)

# In[18]:


part_null_data = train_df[train_df['part_avg'].isna()].groupby('part')[target].mean()
cluster_null_data = train_df.groupby('cluster_id')[target].mean()

# part_null_data.to_pickle('riiid_pre_data//part_null_data.pickle')
# cluster_null_data.to_pickle('riiid_pre_data//cluster_null_data.pickle')

# part_null_data = pd.read_pickle('riiid_pre_data//part_null_data.pickle')
# cluster_null_data = pd.read_pickle('riiid_pre_data//cluster_null_data.pickle')

content_agg = train_df.groupby('content_id')[target].agg(['sum', 'count'])

# content_agg.to_pickle('riiid_pre_data/content_agg.pickle')

# content_agg = pd.read_pickle('riiid_pre_data/content_agg.pickle')
train_df['content_count'] = train_df['content_id'].map(content_agg['count']).astype('int32')
train_df['content_mean'] = train_df['content_id'].map(content_agg['sum'] / content_agg['count'])

train_df["attempt"] = train_df.groupby(["user_id","content_id"])[target].cumcount()
train_df['attempt'] = np.where(train_df['attempt']>6, 6, train_df['attempt'])

train_df['prior_question_had_explanation'].fillna(False, inplace=True)
train_df['user_correctness'].fillna(0.68, inplace=True)
train_df['prior_question_had_explanation'] = train_df['prior_question_had_explanation'].astype('bool')
train_df.loc[train_df['part_avg'].isna(), 'part_avg'] = train_df[train_df['part_avg'].isna()]['part'].map(part_null_data)
train_df.loc[train_df['cluster_avg'].isna(), 'cluster_avg'] = train_df[train_df['cluster_avg'].isna()]['cluster_id'].map(cluster_null_data)
train_df['timestamp_diff1'].fillna(25572., inplace=True)
train_df['timestamp_diff2'].fillna(53309., inplace=True)
train_df['prior_question_elapsed_time'].fillna(22000., inplace=True)


# # data for state

# In[19]:


# data = pd.read_csv('riiid-test-answer-prediction/train.csv',
#                    usecols=[2, 3, 7],
#                    dtype=data_types_dict
#                   )
# questions_df_ = pd.read_csv(
#     'riiid-test-answer-prediction/questions.csv', 
#     usecols=[0, 3],
#     dtype={'question_id': 'int16', 'part': 'int8'}
# )
# data = data[data[target] != -1].reset_index(drop=True)
# data = pd.merge(data, questions_df_, left_on='content_id', right_on='question_id', how='left')
# data.drop(columns=['question_id'], inplace=True)

# data.to_pickle('riiid_pre_data/state_data.pickle')

# del data, questions_df_
# gc.collect()


# # make dict

# In[20]:


# from collections import defaultdict

# train_df = pd.read_pickle('my_data/train_df.pickle')

# user_dict_sum = train_df.groupby('user_id')[target].agg('sum').astype('uint16').to_dict(defaultdict(int))
# user_dict_count = train_df.groupby('user_id')[target].agg('count').astype('uint16').to_dict(defaultdict(int))

# part_dict_sum = train_df.groupby(['user_id', 'part'])[target].agg('sum').astype('uint16').to_dict(defaultdict(int))
# part_dict_count = train_df.groupby(['user_id', 'part'])[target].agg('count').astype('uint16').to_dict(defaultdict(int))

# cluster_dict_sum = train_df.groupby(['user_id', 'cluster_id'])[target].agg('sum').astype('uint16').to_dict(defaultdict(int))
# cluster_dict_count = train_df.groupby(['user_id', 'cluster_id'])[target].agg('count').astype('uint16').to_dict(defaultdict(int))


# In[21]:


# joblib.dump(user_dict_sum, "dict_data/user_dict_sum.pkl.zip")
# joblib.dump(user_dict_count, "dict_data/user_dict_count.pkl.zip")
# joblib.dump(part_dict_sum, "dict_data/part_dict_sum.pkl.zip")
# joblib.dump(part_dict_count, "dict_data/part_dict_count.pkl.zip")
# joblib.dump(cluster_dict_sum, "dict_data/cluster_dict_sum.pkl.zip")
# joblib.dump(cluster_dict_count, "dict_data/cluster_dict_count.pkl.zip")


# # cluster0_avg, cluster1_avg, cluster1_avg
# Convert "cluster_avg" to the respective column.

# In[22]:


user_idx = train_df[train_df['cumcount_u']==0].index
for cluster_id in range(0, 3):
    df = train_df[train_df['cluster_id']==cluster_id].groupby('user_id')[target].agg(['cumsum', 'cumcount'])
    df['cumcount'] += 1
    df['mean'] = df['cumsum'] / df['cumcount']
    idx = df.index
    ar = np.empty(len(train_df))
    ar[:] = np.nan
    ar[idx] = df.loc[idx, 'mean']
    train_df[f'cluster{cluster_id}_avg'] = ar
    train_df[f'cluster{cluster_id}_avg'] = train_df.groupby('user_id')[f'cluster{cluster_id}_avg'].shift()
    train_df.loc[user_idx, f'cluster{cluster_id}_avg'] = cluster_null_data[cluster_id]
    train_df[f'cluster{cluster_id}_avg'].fillna(method='ffill', inplace=True)

df = train_df.groupby(['user_id', 'task_container_id']).head(1)[['user_id', 'task_container_id', 'cluster0_avg', 'cluster1_avg', 'cluster2_avg']]
train_df.drop(columns=['cluster0_avg', 'cluster1_avg', 'cluster2_avg'], inplace=True)
train_df = pd.merge(train_df, df, on=['user_id', 'task_container_id'], how='left')

del df
gc.collect()


# In[23]:


train_df[train_df['user_id']==124][['user_id', 'task_container_id', target, 'cluster_id', 'cluster0_avg', 'cluster1_avg', 'cluster2_avg']].head(7)


# In[24]:


display(train_df[[target, 'cluster0_avg', 'cluster1_avg', 'cluster2_avg']].corr())
print('Average of cluster1_avg when cluster_id is 0: ', train_df[train_df['cluster_id']==0]['cluster1_avg'].mean())
print('Average of cluster1_avg when cluster_id is 0 and incorrect answer: ', train_df[(train_df['cluster_id']==0)&(train_df[target]==0)]['cluster1_avg'].mean())
print('Average of cluster1_avg when cluster_id is 0 and correct answer: ', train_df[(train_df['cluster_id']==0)&(train_df[target]==1)]['cluster1_avg'].mean())


# # target_lag

# In[25]:


train_df['target_lag'] = train_df.groupby('user_id')[target].shift()
train_df['target_lag'].fillna(1, inplace=True)

df_ = train_df.groupby(['user_id', 'task_container_id']).head(1)[['user_id', 'task_container_id', 'target_lag']]
train_df.drop(columns=['target_lag'], inplace=True)
train_df = pd.merge(train_df, df_, on=['user_id', 'task_container_id'], how='left')

# lag_dict = train_df.groupby('user_id').tail(1)[['user_id', target]].set_index('user_id')[target].astype('uint8').to_dict()

# joblib.dump(lag_dict, "dict_data/lag_dict.pkl.zip")


# # prior_tag (2/2)

# In[26]:


df_ = train_df.groupby(['user_id', 'task_container_id']).head(1)[['user_id', 'task_container_id', 'prior_tag']]
train_df.drop(columns=['prior_tag'], inplace=True)
train_df = pd.merge(train_df, df_, on=['user_id', 'task_container_id'], how='left')


# # questions_df

# In[27]:


# content_agg['mean'] = content_agg['sum'] / content_agg['count']

# questions_df['content_mean'] = questions_df['question_id'].map(content_agg['mean'])
# questions_df['content_count'] = questions_df['question_id'].map(content_agg['count'])

# questions_df = questions_df.astype({'question_id': 'int16', 'part': 'int8', 'task_num': 'int8', 'cluster_id': 'int8', 'content_mean': 'float32', 'content_count': 'int32'})

# questions_df.to_pickle('riiid_pre_data/questions_df.pickle')


# # user_rating
# Average difference between "answered_correctly" and "content_mean".

# In[28]:


train_df['user_rating'] = train_df[target] - train_df['content_mean']
train_df['user_rating'] = train_df.groupby('user_id')['user_rating'].shift()
train_df['user_rating'] = train_df.groupby('user_id')['user_rating'].cumsum()

df_ = train_df.groupby(['user_id', 'task_container_id']).head(1)[['user_id', 'task_container_id', 'user_rating']]
train_df.drop(columns=['user_rating'], inplace=True)
train_df = pd.merge(train_df, df_, on=['user_id', 'task_container_id'], how='left')

train_df['user_rating'] = train_df['user_rating'] / train_df['cumcount_u']

train_df['user_rating'].fillna(0, inplace=True)

# content_mean_sum_dict = train_df.groupby('user_id')['content_mean'].agg('sum').astype('float32').to_dict(defaultdict(int))

# joblib.dump(content_mean_sum_dict, "dict_data/content_mean_sum_dict.pkl.zip")


# In[29]:


train_df[train_df['user_id']==124][['user_id', 'task_container_id', target, 'content_mean', 'user_rating']].head(7)


# In[30]:


train_df.groupby(target)['user_rating'].mean()


# In[31]:


plt.hist([train_df[train_df[target]==0]['user_rating'].sample(10000),
          train_df[train_df[target]==1]['user_rating'].sample(10000)], label=['0', '1'])
plt.legend();


# Students who answer difficult questions correctly are more versatile.

# # upper limit

# In[32]:


train_df['cumcount_u'] = np.where(train_df['cumcount_u']>7500, 7500, train_df['cumcount_u'])
train_df['cumcount_p'] = np.where(train_df['cumcount_p']>7500, 7500, train_df['cumcount_p'])
train_df['cumcount_cl'] = np.where(train_df['cumcount_cl']>7500, 7500, train_df['cumcount_cl'])


# # time_mean_diff
# The difference between the past "timestamp_diff1" and the current one.The upper limit is set to 100 seconds.

# In[33]:


train_df['time_adm'] = np.where(train_df['timestamp_diff1']>100000, 100000, train_df['timestamp_diff1'])

train_df['time_mean'] = train_df.groupby('user_id')['time_adm'].cumsum() / (train_df.groupby('user_id')[target].cumcount() + 1)
train_df['time_mean'] = train_df.groupby('user_id')['time_mean'].shift()

df_ = train_df.groupby(['user_id', 'task_container_id']).head(1)[['user_id', 'task_container_id', 'time_mean']]
train_df.drop(columns=['time_mean'], inplace=True)
train_df = pd.merge(train_df, df_, on=['user_id', 'task_container_id'], how='left')

train_df['time_mean'].fillna(25572., inplace=True)
train_df['time_mean_diff'] = train_df['time_adm'] - train_df['time_mean']

# time_adm_dict = train_df.groupby('user_id')['time_adm'].agg('sum').to_dict(defaultdict(int))

# joblib.dump(time_adm_dict, "dict_data/time_adm_dict.pkl.zip")


# In[34]:


train_df[train_df['user_id']==124][['user_id', 'task_container_id', target, 'timestamp_diff1', 'time_mean_diff']].head(7)


# In[35]:


plt.hist([train_df[train_df[target]==0]['time_mean_diff'].sample(10000),
          train_df[train_df[target]==1]['time_mean_diff'].sample(10000)], label=['0', '1'])
plt.legend();


# Take more time than usual for problems you are not confident in.

# In[36]:


features = [
    'content_id',
    'prior_question_elapsed_time',
    'prior_question_had_explanation',
    'user_correctness',
    'content_count',
    'part',
    'content_mean',
    'cumcount_u',
    'cumcount_p',
    'attempt',
    'part_avg',
    'timestamp_diff1',
    'timestamp_diff2',
    'cluster_id',
    'cumcount_cl',
    'target_lag',
    'cluster0_avg',
    'cluster1_avg',
    'cluster2_avg',
    'prior_tag',
    'task_num',
    'user_rating',
    'time_mean_diff',
]


# In[37]:


train_df = train_df.astype(features_dtypes)


# In[38]:


train_df[features].head()


# # save

# In[39]:


# train_df.to_pickle('my_data/train_df.pickle')


# **Thank you for a great competition...!!**
