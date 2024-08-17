#!/usr/bin/env python
# coding: utf-8

# Notebook to prepare features for the AMEX competition.
# 
# Note that this is a starter notebook, mainly aimed at showing how you can more or less easily handle a big dataset. 
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
# **Please make sure to upvote everything you use / find interesting / usefull**

# In[1]:


import numpy as np
import pandas as pd
import gc

DEBUG = False


# # Quick look at the data

# In[2]:


train_labels = pd.read_csv('../input/amex-default-prediction/train_labels.csv')
test_read = pd.read_csv('../input/amex-default-prediction/train_data.csv',nrows=10)

test_read.head()


# # prepare and agregate data

# In[3]:


cat_features = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
bin_features = ['B_31', 'D_87']

remove = ['customer_ID','S_2_max']

f_names = ['mean','std','min','max','last','nunique','first']
f_names_cat = ['last','nunique','first']


# https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def prepare_df_num(df):
    #replace -0.5 with na for aggregation ?
    
    # prepare date - TO DO: change max based on data set (train / private / public)
    df['S_2'] = pd.to_datetime(df['S_2'])
    df['S_2_max'] = df[['S_2','customer_ID']].groupby('customer_ID').S_2.transform('max')
    df['S_2_diff'] = df[['S_2','customer_ID']].groupby('customer_ID').S_2.transform('diff').dt.days
    df['S_2'] = (df['S_2_max']-df['S_2']).dt.days

    # compute "after pay" features - see: https://www.kaggle.com/code/jiweiliu/rapids-cudf-feature-engineering-xgb
    for bcol in [f'B_{i}' for i in [11,14,17]]+['D_39','D_131']+[f'S_{i}' for i in [16,23]]:
        for pcol in ['P_2','P_3']:
            if bcol in df.columns:
                df[f'{bcol}-{pcol}'] = df[bcol] - df[pcol]
                            
    cols_num = [c for c in df.columns if c not in cat_features+bin_features+remove]
    df.loc[:,cols_num] = df.loc[:,cols_num].astype('float16')
    
    return df

def prepare_df_cat(df):
    # identifiy cat columns
    df.loc[:,cat_features+bin_features] = df.loc[:,cat_features+bin_features].astype(str)
    return df

def agg_df_num(df):
    df_agg = df.groupby('customer_ID').agg(f_names)
    df_agg.columns = [str(c[0])+'_'+str(c[1]) for c in df_agg.columns]
    return df_agg

def agg_df_cat(df):
    df_agg = df.groupby('customer_ID').agg(f_names_cat)
    df_agg.columns = [str(c[0])+'_'+str(c[1]) for c in df_agg.columns]
    return df_agg


def prepare_dataset(train_test = 'train'):
    
    data = pd.read_parquet('../input/amex-data-integer-dtypes-parquet-format/'+train_test+'.parquet')
    
    if DEBUG:
        data = data.iloc[:int((len(data)/60))]
    
    split_ids = split(data.customer_ID.unique(),10)

    df_list = []

    for (i,ids) in enumerate(split_ids):
        print(i)
        data_ids = data[data.customer_ID.isin(ids)]

        data_ids = prepare_df_num(data_ids)
        data_ids = prepare_df_cat(data_ids)

        cols_num = [c for c in data_ids.columns if c not in cat_features+bin_features]

        num_agg =  agg_df_num(data_ids[cols_num])
        cat_agg =  agg_df_cat(data_ids[['customer_ID']+cat_features+bin_features])

        df_list.append(pd.concat([num_agg, cat_agg],axis=1).astype('float16'))
        gc.collect()

    pd.concat(df_list,axis=0).astype('float16').to_pickle(train_test+'_data_agg.pkl')


# In[4]:


get_ipython().run_cell_magic('time', '', "\nprepare_dataset(train_test = 'train')\n")


# In[5]:


get_ipython().run_cell_magic('time', '', "\nprepare_dataset(train_test = 'test')\n")

