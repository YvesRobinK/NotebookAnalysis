#!/usr/bin/env python
# coding: utf-8

# ## Source Kernel
# This kernel generates and submits predictions using the model and features developed in the kernel titled [RIIID: BigQuery-XGBoost End-to-End](https://www.kaggle.com/calebeverett/riiid-bigquery-xgboost-end-to-end).

# In[1]:


import gc
import json
import pandas as pd
from pathlib import Path
import sqlite3
import riiideducation
import time
import xgboost as xgb


# In[2]:


env = riiideducation.make_env()
iter_test = env.iter_test()


# ## Load Model

# In[3]:


PATH = Path('../input/riiid-submission')


# In[4]:


model = xgb.Booster(model_file=PATH/'model.xgb')
print('model loaded')


# ## Load State

# In[5]:


dtypes = {
    'answered_correctly': 'int8',
    'answered_correctly_content_id_cumsum': 'int16',
    'answered_correctly_content_id_cumsum_pct': 'int16',
    'answered_correctly_cumsum': 'int16',
    'answered_correctly_cumsum_pct': 'int8',
    'answered_correctly_cumsum_upto': 'int8',
    'answered_correctly_rollsum': 'int8',
    'answered_correctly_rollsum_pct': 'int8',
    'answered_incorrectly': 'int8',
    'answered_incorrectly_content_id_cumsum': 'int16',
    'answered_incorrectly_cumsum': 'int16',
    'answered_incorrectly_rollsum': 'int8',
    'bundle_id': 'uint16',
    'content_id': 'int16',
    'content_type_id': 'int8',
    'correct_answer': 'uint8',
    'lecture_id': 'uint16',
    'lectures_cumcount': 'int16',
    'part': 'uint8',
    'part_correct_pct': 'uint8',
    'prior_question_elapsed_time': 'float32',
    'prior_question_elapsed_time_rollavg': 'float32',
    'prior_question_had_explanation': 'bool',
    'question_id': 'uint16',
    'question_id_correct_pct': 'uint8',
    'row_id': 'int64',
    'tag': 'uint8',
    'tag__0': 'uint8',
    'tag__0_correct_pct': 'uint8',
    'tags': 'str',
    'task_container_id': 'int16',
    'task_container_id_orig': 'int16',
    'timestamp': 'int64',
    'type_of': 'str',
    'user_answer': 'int8',
    'user_id': 'int32'
}

batch_cols_all = [
    'user_id',
    'content_id',
    'row_id',
    'task_container_id',
    'timestamp',
    'prior_question_elapsed_time',
    'prior_question_had_explanation'
]

batch_cols_prior = [
    'user_id',
    'content_id',
    'content_type_id'
]

with open(PATH/'columns.json') as cj:
    test_cols = json.load(cj)

batch_cols = ['user_id', 'content_id', 'row_id'] + [c for c in batch_cols_all if c in test_cols]

print('test_cols:')
_ = list(map(print, test_cols))

dtypes_test = {k: v for k,v in dtypes.items() if k in test_cols}
dtypes_test = {**dtypes_test, **{'user_id': 'int32', 'content_id': 'int16'}}


# ### Load Users-Content

# In[6]:


df_users_content = pd.read_pickle(PATH/'df_users_content.pkl')
df_users_content.head()


# ### Create Users Dataframe

# In[7]:


df_users = df_users_content[['user_id', 'answered_correctly', 'answered_incorrectly']].groupby('user_id').sum().reset_index()
df_users = df_users.astype({'user_id': 'int32', 'answered_correctly': 'int16', 'answered_incorrectly': 'int16'})
df_users.head()


# ### Load Questions
# Question related features joined with batches received from competition api prior to making predictions.

# In[8]:


df_questions = pd.read_pickle(PATH/'df_questions.pkl')
df_questions.head()


# ## Create Database

# In[9]:


conn = sqlite3.connect(':memory:')
cursor = conn.cursor()


# ### Create Users-Content Table

# In[10]:


get_ipython().run_cell_magic('time', '', "\nchunk_size = 20000\ntotal = len(df_users_content)\nn_chunks = (total // chunk_size + 1)\n\ni = 0\nwhile i < n_chunks:\n    df_users_content.iloc[i * chunk_size:(i + 1) * chunk_size].to_sql('users_content', conn, method='multi', if_exists='append', index=False)\n    i += 1\n\nconn.execute('CREATE UNIQUE INDEX users_content_index ON users_content (user_id, content_id)')\ndel df_users_content\ngc.collect()\n")


# In[11]:


get_ipython().run_cell_magic('time', '', "pd.read_sql('SELECT * from users_content LIMIT 5', conn)\n")


# ### Create Users Table

# In[12]:


get_ipython().run_cell_magic('time', '', "\nchunk_size = 20000\ntotal = len(df_users)\nn_chunks = (total // chunk_size + 1)\n\ni = 0\nwhile i < n_chunks:\n    df_users.iloc[i * chunk_size:(i + 1) * chunk_size].to_sql('users', conn, method='multi', if_exists='append', index=False)\n    i += 1\n\n_ = conn.execute('CREATE UNIQUE INDEX users_index ON users (user_id)')\ndel df_users\ngc.collect()\n")


# In[13]:


get_ipython().run_cell_magic('time', '', "pd.read_sql('SELECT * from users LIMIT 5', conn)\n")


# ### Create Questions Table

# In[14]:


get_ipython().run_cell_magic('time', '', "\nq_cols = [\n    'question_id',\n    'part',\n    'tag__0',\n    'part_correct_pct',\n    'tag__0_correct_pct',\n    'question_id_correct_pct'\n]\n\ndf_questions[q_cols].to_sql('questions', conn, method='multi', index=False)\n_ = conn.execute('CREATE UNIQUE INDEX question_id_index ON questions (question_id)')\ndel df_questions\ngc.collect()\n")


# In[15]:


get_ipython().run_cell_magic('time', '', "pd.read_sql('SELECT * from questions LIMIT 5', conn)\n")


# In[16]:


db_size = pd.read_sql('SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()', conn)['size'][0]
print(f'Total size of database is: {db_size/1e9:0.3f} GB')


# In[17]:


import sys
if True:
    local_vars = list(locals().items())
    for var, obj in local_vars:
        size = sys.getsizeof(obj)
        if size > 1e7:
            print(f'{var:<18}{size/1e6:>10,.1f} MB')


# ## Predict

# ### Get State

# In[18]:


def select_state(batch_cols, records):
    return f"""
        WITH b ({(', ').join(batch_cols)}) AS (
        VALUES {(', ').join(list(map(str, records)))}
        )
        SELECT
            {(', ').join([f'b.{col}' for col in batch_cols])},
            IFNULL(answered_correctly_cumsum, 0) answered_correctly_cumsum, 
            IFNULL(answered_incorrectly_cumsum, 0) answered_incorrectly_cumsum,
            IIF(
                (answered_correctly_cumsum + answered_incorrectly_cumsum) > 0,
                answered_correctly_cumsum * 100 / (answered_correctly_cumsum + answered_incorrectly_cumsum),
                0
            ) answered_correctly_cumsum_pct,
            IFNULL(answered_correctly_content_id_cumsum, 0) answered_correctly_content_id_cumsum,
            IFNULL(answered_incorrectly_content_id_cumsum, 0) answered_incorrectly_content_id_cumsum,
            {(', ').join(q_cols)}
        FROM b
        LEFT JOIN (
            SELECT user_id, answered_correctly answered_correctly_cumsum,
                answered_incorrectly answered_incorrectly_cumsum
            FROM users
            WHERE {(' OR ').join([f'user_id = {r[0]}' for r in records])}
        ) u ON (u.user_id = b.user_id)
        LEFT JOIN (
            SELECT user_id, content_id, answered_correctly answered_correctly_content_id_cumsum, 
            answered_incorrectly answered_incorrectly_content_id_cumsum
            FROM users_content uc
            WHERE {(' OR ').join([f'(user_id = {r[0]} AND content_id = {r[1]})' for r in records])}
        ) uc ON (uc.user_id = b.user_id AND uc.content_id = b.content_id)
        LEFT JOIN (
            SELECT {(', ').join(q_cols)}
            FROM questions
        ) q ON (q.question_id = b.content_id)
    """


# ### Update State

# In[19]:


def update_state(df):
    
    def get_select_params(r):
        values_uc = f'({r.user_id}, {r.content_id}, {r.answered_correctly}, {1-r.answered_correctly})'
        values_u = f'({r.user_id}, {r.answered_correctly}, {1-r.answered_correctly})'
        return values_uc, values_u
    
    values = df.apply(get_select_params, axis=1, result_type='expand')
    
    return f"""
        INSERT INTO users_content(user_id, content_id, answered_correctly, answered_incorrectly)
        VALUES {(',').join(values[0])}
        ON CONFLICT(user_id, content_id) DO UPDATE SET
            answered_correctly = answered_correctly + excluded.answered_correctly,
            answered_incorrectly = answered_incorrectly + excluded.answered_incorrectly;
             
        INSERT INTO users(user_id, answered_correctly, answered_incorrectly)
        VALUES {(',').join(values[1])}
        ON CONFLICT(user_id) DO UPDATE SET
            answered_correctly = answered_correctly + excluded.answered_correctly,
            answered_incorrectly = answered_incorrectly + excluded.answered_incorrectly;
    """


# In[20]:


get_ipython().run_cell_magic('time', '', "df_batch_prior = None\ncounter = 0\n\nfor test_batch in iter_test:\n    counter += 1\n\n    # update state\n    if df_batch_prior is not None:\n        answers = eval(test_batch[0]['prior_group_answers_correct'].iloc[0])\n        df_batch_prior['answered_correctly'] = answers\n        cursor.executescript(update_state(df_batch_prior[df_batch_prior.content_type_id == 0]))\n\n        if not counter % 100:\n            conn.commit()\n\n    # save prior batch for state update\n    df_batch_prior = test_batch[0][batch_cols_prior].astype({k: dtypes[k] for k in batch_cols_prior})\n\n    # get state\n    df_batch = test_batch[0][test_batch[0].content_type_id == 0]\n    records = df_batch[batch_cols].fillna(0).to_records(index=False)\n    df_batch = pd.read_sql(select_state(batch_cols, records), conn)\n\n    # predict\n    predictions = model.predict(xgb.DMatrix(df_batch[test_cols]))\n    df_batch['answered_correctly'] = predictions\n\n    #submit\n    env.predict(df_batch[['row_id', 'answered_correctly']])\n")

