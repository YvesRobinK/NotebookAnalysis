#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering: Transform Functions
# 
# Another powerfull tool offered by pandas is transformation of feature. The goal of this notebook is to show what transform approaches can offer on top of agregation functions.
# 
# **This Notebook is part of a serie built for the AMEX competition:**
# - [Base Feature engineering](https://www.kaggle.com/code/lucasmorin/amex-feature-engineering-base)
# - [Baseline lgbm](https://www.kaggle.com/code/lucasmorin/amex-lgbm-features-eng)
# - [Feature Engineering 2: aggregation function](https://www.kaggle.com/code/lucasmorin/amex-feature-engineering-2-aggreg-functions)
# - [Feature Engineering 3: transformation function](https://www.kaggle.com/code/lucasmorin/amex-feature-engineering-3-transform-functions)
# 
# **With associated Data Sets:**
# - [Base Feature engineering](https://www.kaggle.com/datasets/lucasmorin/amex-base-fe)
# - [Feature Engineering 2 - aggregation function](https://www.kaggle.com/datasets/lucasmorin/amex-fe2)
# - [Feature Engineering 3 - transform function](https://www.kaggle.com/datasets/lucasmorin/amex-fe3)
# 
# 
# **Please make sure to upvote everything you use / find interesting / usefull**

# In[1]:


import numpy as np
import pandas as pd
import warnings


# # Pandas transform
# 
# Transform allows for direct transformation of values (mutation). It is generally usefull to create new features, such as ranks and perform imputation work, such as missing value / target imputation. The general idea is to use the pandas groupby on a categorical features and perform the transformation. This allows to build features that depends on multiple rows. 

# # Pandas base function
# 
# For a base exemple we can compare to aggregating function:

# In[2]:


df = pd.DataFrame({'group':[1,1,2,2],'values':[4,1,1,2],'values2':[0,1,1,2]})
df


# In[3]:


df.groupby('group').mean()


# In[4]:


df.groupby('group').transform('mean')


# We get the same value, the main difference is that we get as much lines as the initial df. The aggregation could be performed with a groupby/add then mapped, but transform is more practical.
# This practicality helps realise different stuff easily.

# # Getting data

# Public and private are split to avoid leakage.
# (Note: This is a best practice for ML, you might want a bit of leakage to climb the LB).

# In[5]:


get_ipython().run_cell_magic('time', '', "# separating public and private\n\ntest = pd.read_parquet('../input/amex-data-integer-dtypes-parquet-format/test.parquet')\n\ndate_max = pd.to_datetime(test[['customer_ID','S_2']].drop_duplicates(subset=['customer_ID'], keep='last').set_index('customer_ID').S_2)\ndate_max.hist();\nID_Public = date_max[date_max<'2019-07-01'].index.to_list()\nID_Private = date_max[~(date_max<'2019-07-01')].index.to_list()\n\ndel test\n")


# In[6]:


get_ipython().run_cell_magic('time', '', "\ntrain = pd.read_pickle('../input/amex-feature-engineering-base/train_data_agg.pkl')\ntest = pd.read_pickle('../input/amex-feature-engineering-base/test_data_agg.pkl')\n\npublic = test[test.index.isin(ID_Public)]\nprivate = test[test.index.isin(ID_Private)]\n\ndel test\n")


# # Main Features

# In[7]:


main_features = [f'B_{i}' for i in [11,14,17]]+['D_39','D_131']+[f'S_{i}' for i in [16,23]]+['P_2','P_3']
cat_features = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68'] + ['B_31', 'D_87']

main_features_last = [f+'_last' for f in main_features]
cat_features_last = [f+'_last' for f in cat_features]


# In[8]:


train = train[main_features_last+cat_features_last]
public = public[main_features_last+cat_features_last]
private = private[main_features_last+cat_features_last]


# In[9]:


import gc
gc.collect()


# # Mean by categories

# In[10]:


def agg_mean_by_cat(df):
    df_list = []
    for c in cat_features_last:
        df_agg = df[main_features_last].groupby(df[c]).transform('mean')
        df_agg.columns = [f+'_mean_by_'+c for f in df_agg.columns]
        df_list.append(df_agg.astype('float16'))
    return pd.concat(df_list,axis=1).astype('float16')


# # Global (pct) rank 

# In[11]:


def agg_global_rank(df):
    df_rank = df[main_features_last].transform('rank')
    df_rank.columns = [s+'_global_rank' for s in df_rank.columns]
    return (df_rank/len(df)).astype('float16')


# # (Pct) Rank by categories

# In[12]:


def agg_pct_rank_by_cat(df):
    df_list = []
    for c in cat_features_last:
        df_agg = df[main_features_last].groupby(df[c]).transform('rank')/df[main_features_last].groupby(df[c]).transform('count')
        df_agg.columns = [f+'_pct_rank_by_'+c for f in df_agg.columns]
        df_list.append(df_agg.astype('float16'))

    return pd.concat(df_list,axis=1).astype('float16')


# # Standardize by categories
# 
# We can use lambda functions for agregation. This allow to perform somewhat complex operations easily.

# In[13]:


def agg_standardize_by_cat(df):
    df_list = []
    for c in cat_features_last:
        df_agg = df[main_features_last].groupby(df[c]).transform(lambda x: (x - np.nanmean(x)) / np.nanstd(x))
        df_agg.columns = [f+'_standardized_by_'+c for f in df_agg.columns]
        df_list.append(df_agg.astype('float16'))

    return pd.concat(df_list,axis=1).astype('float16')


# # Target by categories
# 
# This allows for target encoding. However the application to tests require saving the encoded target. So it is in fact better to use a groupy+agg then map the result. 

# In[14]:


train['target'] = pd.read_csv('../input/amex-default-prediction/train_labels.csv').target.values


# In[15]:


df_list_train = []
df_list_public = []
df_list_private = []

for c in cat_features_last:
    to_impute = train['target'].groupby(train[c]).agg('mean')
    df_list_train.append(train[c].map(to_impute).astype('float16'))
    df_list_public.append(public[c].map(to_impute).astype('float16'))
    df_list_private.append(private[c].map(to_impute).astype('float16'))

def save_target(df_list,name):
    df_agg = pd.concat(df_list,axis=1).astype('float16')
    df_agg.columns = ['target_mean_by_'+c for f in df_agg.columns]
    df_agg.to_pickle(name+'_target_impute.p')

save_target(df_list_train,'train')
save_target(df_list_public,'public')
save_target(df_list_private,'private')


# # Prepare all data

# In[16]:


def prepare_df(df, name):
    df1 = agg_mean_by_cat(df)
    df1.to_pickle(name+'_mean_by_cat.p')
    if name=='train':
        print('mean_by_cat')
        display(df1)
        display(df1.describe())
    del df1
    df2 = agg_global_rank(df)
    df2.to_pickle(name+'_rank.p')
    if name=='train':
        print('global rank')
        display(df2)
        display(df2.describe())
    del df2
    df3 = agg_pct_rank_by_cat(df)
    df3.to_pickle(name+'_rank_by_cat.p')
    if name=='train':
        print('rank by cat')
        display(df3)
        display(df3.describe())
    del df3
    df4 = agg_standardize_by_cat(df)
    df4.to_pickle(name+'_standardize_by_cat.p')
    if name=='train':
        print('standardize by cat')
        display(df4)
        display(df4.describe())
    del df4


# In[17]:


prepare_df(train, 'train')
del train
prepare_df(public, 'public')
del public
prepare_df(private, 'private')
del private


# In[18]:


for c in ['rank_by_cat','standardize_by_cat','mean_by_cat','rank','target_impute']:
    
    public_name = 'public_'+c+'.p'
    private_name = 'private_'+c+'.p'
    
    test = pd.concat([pd.read_pickle(public_name),pd.read_pickle(private_name)])
    test.to_pickle('test_'+c+'.p')

