#!/usr/bin/env python
# coding: utf-8

# # Saving Predictions from Previous Levels - 2 Ways!
# 
# One of the challenges of this competition is how to deal with kaggle's time series API when making predictions. Unlike other competitions, we aren't given one file of test data to predict on, but rather presented the data sequentially, where we are provided with their data for one level group at a time, in order, one student at a time.
# 
# This unfortunately limits the models that we can make, because we don't have the full set of information when making predictions, but this is a really valuable skill to learn in data science because this is much more similar to real life - you can never know what happens in the future! 
# 
# That being said, we do know what happens in the past, and using this information in the right way can give our models more information to work with to make more accurate predictions, and boost your model scores on the leaderboard. In this notebook, we will cover two different methods to use information from previous levels to improve our predictions at inference time. The steps taken to train your models with this extra information are very similar!
# 
# # How does the Time Series API work?
# 
# First, it's very important to understand how the API for this competition works. At inference time, we will be presented with the data for one student (or one `session_id`) at a time, in the order of the level groups.
# 
# For example, if we just print out the first few records the API presents us with, we will find:
# 
# | Session ID | Level Group | 
# | ----- | ----- |
# | 20090109393214576 | 0-4 | 
# | 20090312143683264 | 0-4 | 
# | 20090312331414616 | 0-4 | 
# | 20090109393214576 | 5-12 |
# | 20090312143683264 | 5-12 | 
# | 20090312331414616 | 5-12 | 
# | 20090109393214576 | 13-22 |
# | 20090312143683264 | 13-22 | 
# | 20090312331414616 | 13-22 | 
# 
# So we're looping over our session IDs for one level group (20090109393214576 -> 20090312143683264 -> 20090312331414616), and then the same session IDs for the next level groups (0-4 -> 5-12 -> 13-22).  
# 
# 
# Let's print that out to prove it!

# In[1]:


import pandas as pd
import numpy as np
from IPython.display import display


# In[2]:


import jo_wilder
env = jo_wilder.make_env()
iter_test = env.iter_test()


# In[3]:


counter = 0
for (sample_submission, test) in iter_test:
    session_id = test['session_id'].unique()
    level_group = test['level_group'].unique()
    
    print()
    print(f'Session ID: {session_id}')
    print(f'Level Group: {level_group}')
    print()
    print('='*30)
    
    ## Make a dummy submission so we can move on 
    sample_submission['correct'] = 0
    env.predict(sample_submission)
    counter += 1


# Now that we better understand how the API works, let's explore how we can use this ordering of the data to feed some extra information into our models!

# # Method 1: Saving predictions from previous level groups
# One of the easiest ways to save past information is to leverage the fact that the data is presented in the order of level groups. We can make predictions for the first level group for a session_id, store this, and then add these predictions as a feature when we want to make predictions for the next level group! That way we know how well the student has performed (or how well our models think they have performed) on previous quesions, and our model can use this as extra information. 
# 
# Let's see how we can do this!      
# (We won't use an actual model to make predictions for now, but the code to do so has been left commented out. The focus of this notebook is on how to save the predictions at inference time.)

# In[4]:


# First let's reset the API
jo_wilder.make_env.__called__ = False
env.__called__ = False
type(env)._state = type(type(env)._state).__dict__['INIT']

# And reinitialise it
env = jo_wilder.make_env()
iter_test = env.iter_test()


# In[5]:


# Define some basic feature engineering, courtesy of Chris Deotte
CATS = ['event_name', 'fqid', 'room_fqid', 'text']
NUMS = ['elapsed_time','level','page','room_coor_x', 'room_coor_y', 
        'screen_coor_x', 'screen_coor_y', 'hover_duration']

# https://www.kaggle.com/code/kimtaehun/lightgbm-baseline-with-aggregated-log-data
EVENTS = ['navigate_click','person_click','cutscene_click','object_click',
          'map_hover','notification_click','map_click','observation_click',
          'checkpoint']

# https://www.kaggle.com/code/cdeotte/xgboost-baseline-0-676
def feature_engineer(train):
    
    dfs = []
    for c in CATS:
        tmp = train.groupby(['session_id','level_group'])[c].agg('nunique')
        tmp.name = tmp.name + '_nunique'
        dfs.append(tmp)
    for c in NUMS:
        tmp = train.groupby(['session_id','level_group'])[c].agg('mean')
        tmp.name = tmp.name + '_mean'
        dfs.append(tmp)
    for c in NUMS:
        tmp = train.groupby(['session_id','level_group'])[c].agg('std')
        tmp.name = tmp.name + '_std'
        dfs.append(tmp)
    for c in EVENTS: 
        train[c] = (train.event_name == c).astype('int8')
    for c in EVENTS + ['elapsed_time']:
        tmp = train.groupby(['session_id','level_group'])[c].agg('sum')
        tmp.name = tmp.name + '_sum'
        dfs.append(tmp)
    train = train.drop(EVENTS,axis=1)
        
    df = pd.concat(dfs,axis=1)
    df = df.fillna(-1)
    df = df.reset_index()
    df = df.set_index('session_id')
    return df


# In[6]:


# Saving predictions from previous level groups
preds = {}
limits = {'0-4':(1,4), '5-12':(4,14), '13-22':(14,19)}
best_threshold = 0.63

# LOAD MODELS FOR EACH QUESTION (excluded here for simplicity)
# models = {}
# for t in range(1, 19):
#     clf = XGBClassifier()
#     clf.load_model(f'/kaggle/input/.../question_{t}.xgb')
#     models[t] = clf

# The kaggle API will present data in order of level_groups
for (sample_submission, test) in iter_test:
    
    # Figure out our level group and the limits of that level group
    grp = test.level_group.values[0]
    a,b = limits[grp]

    # FEATURE ENGINEERING
    df = feature_engineer(test)

    # Initialise preds for this session_id
    session_id = sample_submission.iloc[0, :]['session_id'].split('_')[0]
    if session_id not in preds.keys():
        preds[session_id] = {}

    # ADD PREDS FROM PREVIOUS LEVEL GROUPS TO PREDICT!
    if grp == '5-12':
        for i in range(1,4):
            df[f'preds_{i}'] = preds[session_id][i]
    elif grp == '13-22':
        for i in range(1,14):
            df[f'preds_{i}'] = preds[session_id][i]
            
    # Go in order of questions for each level group
    for t in range(a,b):
            
        # Make a prediction (excluded here for simplicity)
        # p = clf.predict_proba(df)[:,1].item()
        p = np.random.rand() # (Replace with the above when making actual predictions)

        # SAVE PREDICTION TO DICT
        preds[session_id][t] = p

        # Make a submission 
        mask = sample_submission.session_id.str.contains(f'q{t}')
        sample_submission.loc[mask, 'correct'] = int(p>best_threshold)
        
    env.predict(sample_submission)
    
    # Print out the predictions we save after each level group!
    display(pd.DataFrame.from_dict(preds).T)


# # Method 2: Saving predictions from previous levels
# Another way in which we can perform this knowledge distillation from the past, is by not only saving the predictions from previous level groups, but from *all* previous levels. e.g. Using predictions from level 1 when making predictions for level 2, and from levels 1-17 when predictions for level 18. This should give our model even more information to work with, and our model may be able to pick up patters where if a student performs well on one question then they might do well on another!
# 
# (Again, we won't use an actual model to make predictions for now, but the code to do so has been left commented out. The focus of this notebook is on how to save the predictions at inference time.)

# In[7]:


# Let's reset the API again
jo_wilder.make_env.__called__ = False
env.__called__ = False
type(env)._state = type(type(env)._state).__dict__['INIT']

# And reinitialise it
env = jo_wilder.make_env()
iter_test = env.iter_test()


# In[8]:


# This time we will save predictions from previous levels
preds = {}
limits = {'0-4':(1,4), '5-12':(4,14), '13-22':(14,19)}
best_threshold = 0.63

# LOAD MODELS FOR EACH QUESTION (excluded here for simplicity)
# models = {}
# for t in range(1, 19):
#     clf = XGBClassifier()
#     clf.load_model(f'/kaggle/input/.../question_{t}.xgb')
#     models[t] = clf

# The kaggle API will present data in order of level_groups
for (sample_submission, test) in iter_test:
    
    # Figure out our level group and the limits of that level group
    grp = test.level_group.values[0]
    a,b = limits[grp]

    # FEATURE ENGINEERING
    df = feature_engineer(test)

    # Initialise preds for this session_id
    session_id = sample_submission.iloc[0, :]['session_id'].split('_')[0]
    if session_id not in preds.keys():
        preds[session_id] = {}
            
    # Go in order of questions for each level group
    for t in range(a,b):
        
        # ADD PREDS FROM PREVIOUS LEVELS TO PREDICT!
        if t > 1:
            for i in range(1, t):
                df[i-1] = preds[session_id][i]
            
        # Make a prediction (excluded here for simplicity)
        # p = clf.predict_proba(df)[:,1].item()
        p = np.random.rand() # (Replace with the above when making actual predictions)

        # SAVE PREDICTION TO DICT
        preds[session_id][t] = p

        # Make a submission 
        mask = sample_submission.session_id.str.contains(f'q{t}')
        sample_submission.loc[mask, 'correct'] = int(p>best_threshold)
        
        # Print out the predictions we save after each level!
        display(pd.DataFrame.from_dict(preds).T)
        
    env.predict(sample_submission)


# And now this time you can see how we add predictions one question at a time for each student!
# 
# If you found this notebook useful, please don't forget to upvote it. Hopefully this helps you to squeeze even more out of this competition and climb the leaderboard. Happy Kaggling!
