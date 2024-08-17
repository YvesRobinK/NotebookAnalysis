#!/usr/bin/env python
# coding: utf-8

# # Days overdue feature in the dataset
# 
# This notebook is dedicated to show evidence that anonymized `D_39` feature is in fact `days_overdue` feature - which is equivalent to the `target` in this competition!

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_parquet('../input/amex-data-integer-dtypes-parquet-format/train.parquet')


# ## Days?
# 
# First, let's describe visualize the distribution of this feature.

# In[2]:


# count of 0
np.sum(train['D_39']==0)/train.shape[0]


# As expected, most of the rows are zeroes (meaning no overdue payment days).
# 
# Let's plot the distribution (y-axis in log scale):

# In[3]:


plt.figure(figsize=(16, 10))
sns.kdeplot(train['D_39'], log_scale=[False,True])


# Now this is where it gets interesting. The maximum value of `D_39` is 183 (exactly 6months or ~180days). There are a few spikes at 31,61,91,121,151 - which itself signals about different `D_39` calculation logic based on different products (i.e. instalment vs non-instalment payments). However, most importantly, these values do confirm that this is feature related to days!

# So... all the `D_39` values are less than 6 months (~180 days). Does this mean that the dataset only contains 'good' customers? meaning only such customers who have never previously defaulted?

# ## Already defaulted in train?
# 
# First identify some customers with large `D_39` values:

# In[4]:


pd.options.display.max_colwidth = 100
cols = ['customer_ID','S_2','P_2']+[x for x in train.columns if 'D_' in x]
train.loc[train.D_39>170, cols].head(10)


# Let's explore several customer's timelines:

# In[5]:


train.loc[train.customer_ID=='026ef3a81feea5de51a09d5796b996a1e3ec306ccd7327dd96d55d8d440203a4', cols]


# In[6]:


train.loc[train.customer_ID=='07683296b5cdbcbb9fb41884a545ed7490dbf17816358af23fec5d8c4a03ccf6', cols]


# In[7]:


train.loc[train.customer_ID=='176ad229cbd819198ffc212077a54ea3c3b4a1edbd7b5a10fd461a062de27f77', cols]


# Did you notice what they have in common? THERE ARE MONTHS MISSING JUST AFTER LARGE `D_39` VALUES! This may indicate that the value increased in these missing periods!
# 
# What does it mean? The most likely answer is that organizers removed months which had `D_39` showing that customer already has defaulted (`D_39`>180 or so)!
# 
# Now this is interesting! You can actually reverse engineer the removed month logic - and this is not too hard to do so.
# 
# What can be done with this information? I can think of 2 key things:
# 
# 1. Feature engineering related to large D_39 and missing months (i.e. does the customer had large D_39 and missing following months?)
# 2. Expanding training dataset, by splitting customer timeline into segments (defaulted in train & defaulted as per competition)
# 
# Ultimately this can lead to using all rows (not only last) for training when defaulted in train has been handled. Currently this is impossible or does not produce good results as `target` is noisy for these customers in retrospective months.
# 

# ## Finalle
# 
# Deleted observations related to `target` is not the first time that has happened in kaggle. In one of the previous competition this has caused huge information leak. Thank god that this is time related competition.
# 
# Anyway, I hope you enjoyed and are ready to dive in into this problem further!
