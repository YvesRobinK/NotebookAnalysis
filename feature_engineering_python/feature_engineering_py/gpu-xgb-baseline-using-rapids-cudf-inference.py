#!/usr/bin/env python
# coding: utf-8

# # Using GPU - Part 2
# ## Making a Submission using CPU
# 
# This is Part 2 of the implementation of Chris's idea to use GPU Kaggle notebook for feature engineering during training and then CPU for inference from [here][1]. 
# 
# Part 1 Notebook is used to for feature engineering and training XGBoost using GPU. Check it out [here][2].
# 
# **Version Updates:**
# 
# **Version 2 -**
# 
# In the previous version I was loading the models in the Kaggle API's Inference Loop, which was slow as the notebook was reading the models 198,000 times (11k users and for 18 questions) from the disk. Thanks @cdeotte for pointing this out. 
# 
# I've now updated the notebook. Now the models are loaded outside before Kaggle's API, so the notebook reads the models 18 times from the disk ONCE.
# 
# **Version 6 -**
# Changed the structure of the notebook and added features with the help of [this][3] amazing notebook by @takanashihumbert
# 
# **Version7 -**
# Made changes in the infer loop according to the changes made in the Kaggle API.
# 
# [1]: https://www.kaggle.com/competitions/predict-student-performance-from-game-play/discussion/386218
# [2]: https://www.kaggle.com/code/shashwatraman/gpu-xgb-baseline-using-rapids-cudf-train
# [3]: https://www.kaggle.com/code/takanashihumbert/magic-bingo-train-part-lb-0-687

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
import pickle
from sklearn.model_selection import KFold, GroupKFold
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from tqdm.notebook import tqdm
from collections import defaultdict
import warnings
from itertools import combinations

warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)


# In[2]:


CATS = ['event_name','name','fqid','room_fqid','text_fqid']

NUMS = ['page', 'room_coor_x','room_coor_y','screen_coor_x','screen_coor_y','hover_duration','time_diff']

EVENTS = ['cutscene_click', 'person_click', 'navigate_click',
       'observation_click', 'notification_click', 'object_click',
       'object_hover', 'map_hover', 'map_click', 'checkpoint',
       'notebook_click']

NAMES = ['basic', 'undefined', 'close', 'open', 'prev', 'next']


# In[3]:


def feature_engineer(x, grp):
    
    x['time_diff'] = x['elapsed_time'] - x.groupby('session_id')['elapsed_time'].shift(1)
    mask = x['time_diff'] < 0
    x.loc[mask,'time_diff'] = 0
    
    x['elapsed_time'] = x['elapsed_time'] / 1000
    x['time_diff'] = x['time_diff'] / 1000
    
    #session duration
    df_final = x.groupby('session_id')['index'].agg('count')
    df_final.name = 'num_events'
    df_final = df_final.reset_index()
    df_final = df_final.set_index('session_id')
    
    #Bingo Features
    if grp == '5-12':
        
        df_final['logbingo-logbook'] = x[(x['fqid']=='logbook.page.bingo')&(x['event_name']=='object_click')].groupby('session_id')['index'].agg('first') - x[x['fqid']=='logbook'].groupby('session_id')['index'].agg('first')
        df_final['readerbingo-reader'] = x[(x['fqid']=='reader.paper2.bingo')&(x['event_name']=='object_click')].groupby('session_id')['index'].agg('first') - x[x['fqid']=='reader'].groupby('session_id')['index'].agg('first')
        df_final['jourbingo-journalspic'] = x[(x['fqid']=='journals.pic_2.bingo')&(x['event_name']=='object_click')].groupby('session_id')['index'].agg('first') - x[x['fqid']=='journals.pic_0.next'].groupby('session_id')['index'].agg('first')
        
        df_final['logbingo-logbook_time'] = x[(x['fqid']=='logbook.page.bingo')&(x['event_name']=='object_click')].groupby('session_id')['elapsed_time'].agg('first') - x[x['fqid']=='logbook'].groupby('session_id')['elapsed_time'].agg('first')
        df_final['readerbingo-reader_time'] = x[(x['fqid']=='reader.paper2.bingo')&(x['event_name']=='object_click')].groupby('session_id')['elapsed_time'].agg('first') - x[x['fqid']=='reader'].groupby('session_id')['elapsed_time'].agg('first')
        df_final['jourbingo-journalspic_time'] = x[(x['fqid']=='journals.pic_2.bingo')&(x['event_name']=='object_click')].groupby('session_id')['elapsed_time'].agg('first') - x[x['fqid']=='journals.pic_0.next'].groupby('session_id')['elapsed_time'].agg('first')
        
    if grp=='13-22':
        
        df_final['readerbingo-reader_flag'] = x[(x['fqid']=='reader_flag.paper2.bingo')&(x['event_name']=='object_click')].groupby('session_id')['index'].agg('first') - x[x['fqid']=='reader_flag'].groupby('session_id')['index'].agg('first')
        df_final['journalbingo-journals_flag'] = x[(x['fqid']=='journals_flag.pic_0.bingo')&(x['event_name']=='object_click')].groupby('session_id')['index'].agg('first') - x[x['fqid']=='journals_flag'].groupby('session_id')['index'].agg('first')
        
        df_final['readerbingo-reader_flag_time'] = x[(x['fqid']=='reader_flag.paper2.bingo')&(x['event_name']=='object_click')].groupby('session_id')['elapsed_time'].agg('first') - x[x['fqid']=='reader_flag'].groupby('session_id')['elapsed_time'].agg('first')
        df_final['journalbingo-journals_flag_time'] = x[(x['fqid']=='journals_flag.pic_0.bingo')&(x['event_name']=='object_click')].groupby('session_id')['elapsed_time'].agg('first') - x[x['fqid']=='journals_flag'].groupby('session_id')['elapsed_time'].agg('first')
        
    df_final['first_elapsed_time'] = x.groupby('session_id')['elapsed_time'].agg('first')
    df_final['elapsed_time'] = x.groupby('session_id')['elapsed_time'].agg('last') - df_final['first_elapsed_time']
    
    for c in CATS:
        df_final[f'{c}_nuniques'] = x.groupby('session_id')[c].agg('nunique')
    
    for c in NUMS:
        df_final[f'{c}_mean'] = x.groupby('session_id')[c].agg('mean')
        df_final[f'{c}_min'] = x.groupby('session_id')[c].agg('min')
        df_final[f'{c}_max'] = x.groupby('session_id')[c].agg('max')
        
    for c in EVENTS:
        x[c] = (x.event_name == c).astype('int8')
    for c in EVENTS:
        df_final[f'{c}_sum'] = x.groupby('session_id')[c].agg('sum')
    x.drop(EVENTS, axis=1, inplace=True)
    
    for c in EVENTS:
        df_final[f'{c}_time_mean'] = x[x['event_name']==c].groupby('session_id')['time_diff'].mean()
        df_final[f'{c}_time_min'] = x[x['event_name']==c].groupby('session_id')['time_diff'].min()
        df_final[f'{c}_time_max'] = x[x['event_name']==c].groupby('session_id')['time_diff'].max()
    
    for c in NAMES:
        x[c] = (x.name == c).astype('int8')
    for c in NAMES:
        df_final[f'{c}_sum'] = x.groupby('session_id')[c].agg('sum')
    x.drop(NAMES, axis=1, inplace=True)
    
    for c in NAMES:
        df_final[f'{c}_time_mean'] = x[x['name']==c].groupby('session_id')['time_diff'].mean()
        df_final[f'{c}_time_min'] = x[x['name']==c].groupby('session_id')['time_diff'].min()
        df_final[f'{c}_time_max'] = x[x['name']==c].groupby('session_id')['time_diff'].max()

    return df_final


# In[4]:


f_read = open('/kaggle/input/gpu-xgb-baseline-using-rapids-cudf-train/importance_dict.pkl', 'rb')
importance_dict = pickle.load(f_read)
f_read.close()


# In[5]:


#Loading the models
QUESTION_MODELS = []
for t in range(1,19):
    clf = XGBClassifier()
    clf.load_model(f'../input/gpu-xgb-baseline-using-rapids-cudf-train/XGB_question{t}.xgb')
    QUESTION_MODELS.append( clf )


# ## Infer Test Data

# In[6]:


# IMPORT KAGGLE API
import jo_wilder
env = jo_wilder.make_env()
iter_test = env.iter_test()


# In[7]:


limits = {'0-4':(1,4), '5-12':(4,14),'13-22':(14,19)}
best_threshold = 0.625

historical_meta = defaultdict(list)

for (test, sample_submission) in iter_test:
    
    grp = test.level_group.values[0]
    session_id = test.session_id.values[0]
    
    df = feature_engineer(test, grp)
    
    a,b = limits[grp]
    for t in range(a,b):
        FEATURES = importance_dict[str(t)]
        
        clf = QUESTION_MODELS[t-1]
        p = clf.predict_proba(df[FEATURES].astype('float32'))[0,1]
        mask = sample_submission.session_id.str.contains(f'q{t}')
        sample_submission.loc[mask,'correct'] = int(p.item()>best_threshold)
    
    env.predict(sample_submission)


# ## EDA submission.csv

# In[8]:


df = pd.read_csv('/kaggle/working/submission.csv')
print(df.shape)
df.head()


# In[9]:


print(df.correct.mean())

