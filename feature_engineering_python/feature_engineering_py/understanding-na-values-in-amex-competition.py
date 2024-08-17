#!/usr/bin/env python
# coding: utf-8

# # An attempt to understand missing values
# 
# People keep on asking what to do with missing values. It depends on the situation, but I am certain 9/10 times you do not want to do anything and you would actually treat `NA` as a number. Well this itself is a hard problem - where do I put `NA` in a number space? There are many things to do, but what really matters is to understand how `NA` are generated in the first place. Having that knowledge you can find unexpected solutions!
# 
# Let's dive in into AMEX competition dataset. We are going to use the curated [dataset](https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format) which contains discovered integer column types.

# In[1]:


import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 200)

train = pd.read_parquet('../input/amex-data-integer-dtypes-parquet-format/train.parquet')
train[train==-1] = np.nan # revert -1 to na encoding as in the original dataset


# Let's calculate `NA` counts by feature. The idea is to identify feature clusters having the same `NA` values for each row - that is what I will be focusing in this notebook onwards. Let's explore clusters having more than 10 features.

# In[2]:


cols = sorted(train.columns[2:].tolist())
nas = train[cols].isna().sum(axis=0).reset_index(name='NA_count')
nas['group_count'] = nas.loc[nas.NA_count>0].groupby('NA_count').transform('count')
clusters = nas.loc[nas.group_count>10].sort_values(['NA_count','index']).groupby('NA_count')['index'].apply(list).values
clusters


# There are 3 large clusters having many overlapping `NA` rows! Let's analyse each cluster!

# In[3]:


cluster0_customers = set(train.loc[pd.isnull(train[clusters[0][0]]),'customer_ID'])
cluster1_customers = set(train.loc[pd.isnull(train[clusters[1][0]]),'customer_ID'])
cluster2_customers = set(train.loc[pd.isnull(train[clusters[2][0]]),'customer_ID'])


# ## cluster ZERO
# 
# Let's just simply look at how the customer profile looks for this feature cluster:

# In[4]:


train.loc[train.customer_ID.isin(cluster0_customers),['customer_ID','S_2']+clusters[0]].head(100)


# Were you able to notice the pattern? What if I told you that `NA` appears only on the first observation for each customer! So for this feature cluster `NA` represents fresh credit card accounts with probably zero balance!
# 
# Let's double check that `NA` only appears on the first row:

# In[5]:


# first row
print('first row of the customer')
print(train.loc[train.customer_ID.isin(cluster0_customers)].groupby('customer_ID')[clusters[0]].head(1).isna().sum(axis=0))

# any row
print('any row of the customer')
print(train[clusters[0]].isna().sum(axis=0))


# Assumption correct ^

# ### How could I use this information?
# 
# We know that the dataset has varying number of available observations for each customer (N=1,...,13). It is so far obvious that `last` type features are the strongest ones to build a model.
# 
# For N=1 it gets a bit interesting, because this `NA` propagates to `last` type features. This may cause headache for non-GBDT models. 
# 
# Let me show you what I mean - the distribution of `B_33_last` as an example:

# In[6]:


train.groupby('customer_ID').tail(1).groupby('B_33', dropna=False)['customer_ID'].count()


# There are 31 `NA` values, each coming from customers having a single entry in the data. So if you had the feature like `number_of_observations`, you could add dummy value for the counter meaning `1 observation and B_X is NA` - I will encode this as 0.5 in the example below:

# In[7]:


train['number_of_observations'] = train.groupby('customer_ID')['customer_ID'].transform('count')
print(train.groupby('customer_ID').tail(1).groupby('number_of_observations')['customer_ID'].count().reset_index())
train.loc[pd.isnull(train.B_33) & (train.number_of_observations==1),'number_of_observations'] = 0.5
print(train.groupby('customer_ID').tail(1).groupby('number_of_observations')['customer_ID'].count().reset_index())


# And now you are safe to set all `last` type features' `NA` to zero without losing any information. This means less categories for tree methods, and less hassle for building your neural networks.

# In[8]:


train.loc[train.number_of_observations==0.5,['customer_ID','S_2','number_of_observations']+clusters[0]].fillna(0)


# Another observation could be made that for aggregate type features (min/max/mean,...), it is safe to not include `NA` in any way (the information is already stored in `number_of_observations=0.5`). So using functions like `np.nanmin`, `np.nanmax`, etc. are prefered

# ## cluster ONE
# 
# Again, simply look at what the data has to show:

# In[9]:


train.loc[train.customer_ID.isin(cluster1_customers),['customer_ID','S_2']+clusters[1]].head(100)


# This appears similar to cluster ZERO. However, not all `NA` are in the first row!

# In[10]:


# first row
print('first row of the customer')
print(train.loc[train.customer_ID.isin(cluster1_customers)].groupby('customer_ID')[clusters[1]].head(1).isna().sum(axis=0))

# any row
print('any row of the customer')
print(train[clusters[1]].isna().sum(axis=0))


# In fact, half of the `NA` are in first row and other are in random positions in the timeline. I have not yet found any pattern for the latter.
# 
# So, what can be done!?
# 
# For first row `NA` we can apply the same logic as in cluster ZERO - will leave this as an exercise for you :)
# 
# As for randomly appearing `NA` I think it is worth doing some kind of imputation. Can't stress this enough, but THE MODEL MUST VERIFY THAT IMPUTATION IS A VALID APPROACH! It may be obvious for you that imputation is a legitimate strategy, but the model you will build has to confirm it, meaning that the model with imputed values should perform no worse than with originally missing values. This is how you test the hypothesis if the `NA` is missing at random or not.
# 
# 
# What would I do here? I would probably take average values of `t-1` and `t+1` values for `NA` at timestep `t`. It is pretty straightforward to do so I won't cover this in this notebook. Let's better finish up with the last cluster!

# ## cluster TWO
# 
# Once again, simply look at what the data has to show:

# In[11]:


train.loc[train.customer_ID.isin(cluster2_customers),['customer_ID','S_2']+clusters[2]].head(100)


# If you looked closely enough you would identify that we have rolling `NA` values and the rolling stops when the first numeric information appears, and `NA` no longer reappears. Let's confirm this with a code:

# In[12]:


train['cluster2_NA'] = 0
train.loc[pd.isnull(train[clusters[2][0]]),'cluster2_NA'] = 1
train['rank'] = train.groupby('customer_ID')['S_2'].rank()

not_na = train.loc[train.cluster2_NA==0].groupby('customer_ID')['rank'].agg(('min','max'))
yes_na = train.loc[train.cluster2_NA==1].groupby('customer_ID')['rank'].agg(('min','max'))
c2 = pd.merge(yes_na,not_na,on='customer_ID')


# The idea is that max rank of `NA` should be lower than min rank of non-`NA`. Let's see if it holds:

# In[13]:


c2.loc[c2.max_x>c2.min_y].shape[0], c2.loc[c2.max_x<c2.min_y].shape[0]


# In[14]:


#example customer timeframe showing that `NA` can start from the first observation
train.loc[train.customer_ID == 'fffe2bc02423407e33a607660caeed076d713d8a5ad32321530e92704835da88',['customer_ID','S_2']+clusters[2]]


# In[15]:


#example customer timeframe showing that `NA` does not always start from the first observation
train.loc[train.customer_ID == '002259d195fcb87b15b98503ac5d2c4d29bc8383053ca3ad8951c598fbb6812a',['customer_ID','S_2']+clusters[2]]


# So my initial hypothesis was incorrect! Although 60922 customers do have rolling `NA` from first observation, however there are 6879 `NA` in the mix which may or may not be missing at random. So as in cluster ONE, we have two types of `NA` here! And things gets really complicated from here. One thing is obvious - `NA` count features for rolling first `NA` could be very useful for the models. However it is not clear if other `NA` should be included in those calculations or not! I am not convinced that they should. But the models could help you answer that!

# # Summary
# 
# This dataset is rich of `NA` information. We have both systemic `NA` and very likely missing at random `NA`. This means feature-wise `NA` should not be treated equally when doing any kind of feature engineering.
# 
# On top of that, I have showed you that it is possible to identify many interesting `NA` properties just by looking at 100 rows in the dataset.
# 
# Hope you enjoyed this one. I sure did while I was making this!
