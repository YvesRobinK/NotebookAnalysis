#!/usr/bin/env python
# coding: utf-8

# # Automated feature engineering using Featuretools
# 
# ## What's Featuretools:
# 
# * a python library to perform automated feature engineering.
# * based on "Deep Feature Synthesis" paper/ research
# * Documentation: https://docs.featuretools.com/
# * Source code: https://github.com/Featuretools/featuretools
# * Other examples: https://www.featuretools.com/demos
# 
# ## Deep Feature Synthesis
# * Paper: http://www.jmaxkanter.com/static/papers/DSAA_DSM_2015.pdf
# * Article: https://www.featurelabs.com/blog/deep-feature-synthesis/
# * DFS works with the structured transactional and relational datasets
# * Across datasets features are derived by using primitive mathematical operations
# * New features are composed from using derived features (hence "Deep")
# 

# In[4]:


import pandas as pd
import featuretools as ft

from datetime import datetime

from featuretools.primitives import *


# # 1. Load data

# In[5]:


dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8'
}
to_read = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
to_parse = ['click_time']


# In[6]:


df = pd.read_csv('../input/train_sample.csv', usecols=to_read, dtype=dtypes, parse_dates=to_parse)
df['id'] = df.index


# # 2. Prepare data

# In[7]:


# Create an entity set, a collection of entities (tables) and their relationships
es = ft.EntitySet(id='clicks')

# Create an entity "clicks" based on pandas dataframe and add it to the entity set
es = es.entity_from_dataframe(
    entity_id='clicks',
    dataframe=df,
    index='id',
    time_index='click_time',
    variable_types={
        # We need to set proper types so that Featuretools won't treat them as numericals
        'ip': ft.variable_types.Categorical,
        'app': ft.variable_types.Categorical,
        'device': ft.variable_types.Categorical,
        'os': ft.variable_types.Categorical,
        'channel': ft.variable_types.Categorical,
        'is_attributed': ft.variable_types.Boolean,
    }
)

# We can create new enities based on information we have, e.g. for ips or apps. We “normalize” the entity and extract a new one, this automatically adds a relationship between them
es = es.normalize_entity(base_entity_id='clicks', new_entity_id='ip', index='ip')
es = es.normalize_entity(base_entity_id='clicks', new_entity_id='app', index='app')
es = es.normalize_entity(base_entity_id='clicks', new_entity_id='device', index='device')
es = es.normalize_entity(base_entity_id='clicks', new_entity_id='channel', index='channel')
es = es.normalize_entity(base_entity_id='clicks', new_entity_id='os', index='os')

# How our entityset looks like:
es


# # 3. Create features

# In[9]:


# Run Deep Feature Synthesis for app as a target entity (features will be create for each app)
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_entity='app'
)

# List of created features:
feature_defs


# In[11]:


# The features values
feature_matrix.head()


# # 4. Feature primitives
# 
# * The units/ building blocks of Featuretools
# * Computations applied to raw datasets to create new features
# * Constrains the input and output data types
# * Two types of primitives: aggregation and transform
# 
# ### Aggregation vs Transform Primitive:
# 
# 1. Aggregation primitives: These primitives take related instances as an input and output a single value. They are applied across a parent-child relationship in an entity set. E.g: Count, Sum, AvgTimeBetween.
# 
# 2. Transform primitives: These primitives take one or more variables from an entity as an input and output a new variable for that entity. They are applied to a single entity. E.g: Hour, TimeSincePrevious, Absolute.
# 
# 3. Custom primitives: You can define your own aggregation and transform primitives

# In[18]:


# Create feature with your own primitives
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_entity='app',
    trans_primitives=[Hour],
    agg_primitives=[PercentTrue, Mode]
)

# List of created features:
feature_defs


# # 5. Handling time
# 
# * Featuretools designed to take time into consideration
# * Entities have a column (time index) that specifies the point in time when data in that row became available
# * Cutoff Time specifies the time to calculate features. Only data prior to this time will be used.
# * Training window specifies the time to calculate features. Only data after this time will be used.

# In[14]:


# Tell Featuretools to add time when entity was last seen 
es.add_last_time_indexes()
    
train_cutoff_time = datetime.datetime(2017, 11, 8, 17, 0)
train_training_window = ft.Timedelta("1 day")

feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_entity='app',
    cutoff_time=train_cutoff_time,
    training_window=train_training_window,
)

feature_matrix.head()

