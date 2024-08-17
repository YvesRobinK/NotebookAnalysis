#!/usr/bin/env python
# coding: utf-8

# **<h3>This notebook demonstrates some Feature Preprocesing & Engineering Techniques for time, categorical and numerical Features. Some of those techniques are inspired from the notebook and discussion mentioned below. </h3>**
# 
# <h4><font color='red'>If you like this notebook then please upvote.</h4>

# **ACKNOWLEDGEMENTS**
# 
# 
# * The Techniques I used to reduce the size of the data is inspired from this great [discussion](https://www.kaggle.com/competitions/amex-default-prediction/discussion/328054) by [@cdeotte](https://www.kaggle.com/cdeotte)
# 
# * Many of the techniques related to Numerical Features are inspired from this awesome [notebook](https://www.kaggle.com/code/lucasmorin/amex-feature-engineering) by [@lucasmorin](https://www.kaggle.com/lucasmorin)
# 
# <h3>Plese checkout these links too. </h3>
# 
# 
# 
# 
# 

# **IMPORTS**

# In[1]:


import os
import gc
import glob
import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('max_rows', 100)
pd.set_option('max_columns', 300)


# **LOADING DATA**
# 
# Since the data given by the competition hosts are too large to fit in the memory, I am using 
# the [dataset](https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format)
# of Feather & Parquet files by [@RADDAR](https://www.kaggle.com/raddar) 

# In[2]:


train_data = pd.read_parquet('../input/amex-data-integer-dtypes-parquet-format/train.parquet')
train_labels = pd.read_csv('../input/amex-default-prediction/train_labels.csv')
test_data = pd.read_parquet('../input/amex-data-integer-dtypes-parquet-format/test.parquet')
submission = pd.read_csv('../input/amex-default-prediction/sample_submission.csv')

print(train_data.shape, train_labels.shape)
print(test_data.shape, submission.shape)

bin_cols = ['B_31', 'D_87']
cat_cols = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
num_cols = list(set(train_data.columns)-set(cat_cols+['S_2', 'customer_ID']))
int8_num_cols = list(set(train_data.dtypes[train_data.dtypes==np.int8].axes[0]) - set(cat_cols))
int16_num_cols = list(set(train_data.dtypes[train_data.dtypes==np.int16].axes[0]) - set(cat_cols))
float32_num_cols = list(set(train_data.dtypes[train_data.dtypes==np.float32].axes[0]) - set(cat_cols))

def last_2(series):
    return series.values[-2] if len(series.values)>=2 else -127

def last_3(series):
    return series.values[-3] if len(series.values)>=3 else -127


print("We have {} Categorical features and {} Numerical features".format(len(cat_cols), len(num_cols)))


# **ENCODING CUSTOMER ID**

# In[3]:


### Encode customer ids(https://www.kaggle.com/competitions/amex-default-prediction/discussion/328054)

train_data['customer_ID'] = train_data['customer_ID'].apply(lambda x: int(x[-16:], 16)).astype(np.int64)
train_labels['customer_ID'] = train_labels['customer_ID'].apply(lambda x: int(x[-16:], 16)).astype(np.int64)
test_data['customer_ID'] = test_data['customer_ID'].apply(lambda x: int(x[-16:], 16)).astype(np.int64)
submission['customer_ID'] = submission['customer_ID'].apply(lambda x: int(x[-16:], 16)).astype(np.int64)


# **DATE RELATED FEATURES**(S_2)

# In[4]:


def take_first_col(series):
    return series.values[0]

def prepare_date_features(df):
    ### Drop all other columns except the S_2 and customer_ID(cat_cols, num_cols)
    df = df.drop(cat_cols+num_cols, axis=1)
    
    ### Converting S_2 column to datetime column
    df['S_2'] = pd.to_datetime(df['S_2'])

    ### How many rows of records does each customer has?
    df['rec_len'] = df[['customer_ID', 'S_2']].groupby(by=['customer_ID'])['S_2'].transform('count')

    ### Encode the 1st statement and the last statement time
    df['S_2_first'] = df[['customer_ID', 'S_2']].groupby(by=['customer_ID'])['S_2'].transform('min')
    df['S_2_last'] = df[['customer_ID', 'S_2']].groupby(by=['customer_ID'])['S_2'].transform('max')

    ### For how long(days) the customer is receiving the statements
    df['S_2_period'] = (df[['customer_ID', 'S_2']].groupby(by=['customer_ID'])['S_2'].transform('max') - df[['customer_ID', 'S_2']].groupby(by=['customer_ID'])['S_2'].transform('min')).dt.days

    ### Days Between 2 statements 
    df['days_between_statements'] = df[['customer_ID', 'S_2']].sort_values(by=['customer_ID', 'S_2']).groupby(by=['customer_ID'])['S_2'].transform('diff').dt.days
    df['days_between_statements'] = df['days_between_statements'].fillna(0)
    df['days_between_statements_mean'] = df[['customer_ID', 'days_between_statements']].sort_values(by=['customer_ID', 'days_between_statements']).groupby(by=['customer_ID']).transform('mean')
    df['days_between_statements_std'] = df[['customer_ID', 'days_between_statements']].sort_values(by=['customer_ID', 'days_between_statements']).groupby(by=['customer_ID']).transform('std')
    df['days_between_statements_max'] = df[['customer_ID', 'days_between_statements']].sort_values(by=['customer_ID', 'days_between_statements']).groupby(by=['customer_ID']).transform('max')
    df['days_between_statements_min'] = df[['customer_ID', 'days_between_statements']].sort_values(by=['customer_ID', 'days_between_statements']).groupby(by=['customer_ID']).transform('min')

    ### https://www.kaggle.com/code/lucasmorin/amex-lgbm-features-eng/notebook
    df['S_2'] = (df['S_2_last']-df['S_2']).dt.days

    ### Difference between S_2_last(max) and S_2_last 
    df['S_2_last_diff'] = (df['S_2_last'].max()-df['S_2_last']).dt.days

    ### Difference between S_2_first(min) and S_2_first 
    df['S_2_first_diff'] = (df['S_2_first'].min()-df['S_2_first']).dt.days

    ### Get the (day,month,year) and drop the S_2_first because we can't directly use them
    df['S_2_first_dd'] = df['S_2_first'].dt.day
    df['S_2_first_mm'] = df['S_2_first'].dt.month
    df['S_2_first_yy'] = df['S_2_first'].dt.year
    
    df['S_2_last_dd'] = df['S_2_last'].dt.day
    df['S_2_last_mm'] = df['S_2_last'].dt.month
    df['S_2_last_yy'] = df['S_2_last'].dt.year
    
    agg_df = df.groupby(by=['customer_ID']).agg({'S_2':['last', last_2, last_3],
                                                 'days_between_statements':['last', last_2, last_3]})
    agg_df.columns = [i+'_'+j for i in ['S_2', 'days_between_statements'] for j in ['last', 'last_2', 'last_3']]
    df = df.groupby(by=['customer_ID']).agg(take_first_col)
    df = df.merge(agg_df, how='inner', left_index=True, right_index=True)
    df = df.drop(['S_2', 'days_between_statements', 'S_2_first', 'S_2_last_x'], axis=1)

    return df 


# **NUMERICAL FEATURES**(num_cols)

# In[5]:


def prepare_numerical_features(df):
    for num_c in list(num_cols):
        col_dtype = df[num_c].dtype
        df[num_c] = df[num_c].fillna(df[num_c].mean())
        df[num_c] = df[num_c].astype(col_dtype)
    
    df['S_2'] = pd.to_datetime(df['S_2'])
    df = df.sort_values(by=['customer_ID', 'S_2'])
    ### Drop cat columns and S_2 so that you only have num features and customer_ID
    df = df.drop(cat_cols+['S_2'], axis=1)
    num_feature_list = ['min', 'max', 'mean', 'std', 'last', last_2, last_3]
    
    df_float32_agg = df[['customer_ID']+float32_num_cols].groupby(by=['customer_ID']).agg(num_feature_list).astype(np.float32)
    df_float32_agg.columns = [str(c[0])+'_'+str(c[1]) for c in df_float32_agg.columns]
    
    df_int_agg = df[['customer_ID']+int8_num_cols+int16_num_cols].groupby(by=['customer_ID']).agg(num_feature_list).astype(np.float16)
    df_int_agg.columns = [str(c[0])+'_'+str(c[1]) for c in df_int_agg.columns]
    
    #df_agg = df.groupby(by=['customer_ID']).agg(num_feature_list).astype(np.float32)
    #df_agg.columns = [str(c[0])+'_'+str(c[1]) for c in df_agg.columns]
    df_agg = df_float32_agg.merge(df_int_agg, left_index=True, right_index=True)
    df_agg[[ii+'_last' for ii in int8_num_cols]] = df_agg[[ii+'_last' for ii in int8_num_cols]].astype(np.int8)
    df_agg[[ii+'_last_2' for ii in int8_num_cols]] = df_agg[[ii+'_last_2' for ii in int8_num_cols]].astype(np.int8)
    df_agg[[ii+'_last_3' for ii in int8_num_cols]] = df_agg[[ii+'_last_3' for ii in int8_num_cols]].astype(np.int8)
    
    del df, df_float32_agg, df_int_agg
    gc.collect()
    return df_agg


# **CATEGORICAL FEATURES**(cat_cols+bin_cols)

# In[6]:


### https://www.kaggle.com/code/lucasmorin/amex-feature-engineering
def prepare_cat_features(df):
    remove = ['customer_ID']

    agg_dict_num = {}
    agg_dict_cat = {}

    mean_diff = lambda x: np.nanmean(np.diff(x.values))
    mean_diff.__name__ = 'mean_diff'

    for c in df.columns:
        if c not in remove:
            if c not in cat_cols+bin_cols:
                agg_dict_num[c] = ['mean','std','min','max','last', last_2, last_3]
            else:
                agg_dict_cat[c] = ['nunique', ] 
    
    df.loc[:,cat_cols+bin_cols] = df.loc[:,cat_cols+bin_cols].astype(str)
    df_agg = df.groupby('customer_ID').agg(agg_dict_cat)
    df_agg.columns = [str(c[0])+'_'+str(c[1]) for c in df_agg.columns]
    df_list = []
    for c in cat_cols+bin_cols:
        df_cat = df.groupby(['customer_ID',c])[c].count()
        df_cat = df_cat.unstack()
        df_cat.columns = [df_cat.columns.name + '_' + c for c in df_cat.columns]
        df_cat = df_cat.fillna(0)
        df_list.append(df_cat)
    df_out = pd.concat([df_agg]+df_list, axis=1)
    df_out = df_out.fillna(np.nanmean(df_out))
    
    del df
    gc.collect()
    return df_out


# **SCALING**

# In[7]:


### Currently I am only training tree based models and because Normalization or Standardization
### don't affect them that much, I haven't created that pipeline till now.

from sklearn.preprocessing import StandardScaler, MinMaxScaler
def standardize(train_df, test_df):
    scaler = StandardScaler()
    scaler.fit(train_df)
    train_df = scaler.transform(train_df)
    test_df = scaler.transform(test_df)
    return train_df, test_df

def minmax(train_df, test_df):
    scaler = MinMaxScaler()
    scaler.fit(train_df)
    train_df = scaler.transform(train_df)
    test_df = scaler.transform(test_df)
    return train_df, test_df


# **After we have all the features defined it's time to split the data(both train and test) and apply the transformations and then save them to pickle files**

# **Applying on Train Data**

# In[8]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import StratifiedKFold\n\nkf = StratifiedKFold(n_splits=5)\ntrain_5_fold_splits = []\nfor train_index, val_index in kf.split(train_labels[\'customer_ID\'], train_labels[\'target\']):\n    print(train_labels.iloc[val_index][\'target\'].value_counts())\n    train_5_fold_splits.append(train_labels.iloc[val_index][\'customer_ID\'])\n\ntrain_labels = train_labels.set_axis(train_labels[\'customer_ID\'])\ntrain_labels = train_labels.drop([\'customer_ID\'], axis=1)\n\nfor (i,ids) in enumerate(train_5_fold_splits):\n    print(i, len(ids))\n    train_data_part = train_data[train_data.customer_ID.isin(ids)].sort_values(by=[\'customer_ID\'])\n    y = train_labels.loc[ids][\'target\']\n    np.save("train_y_{}.npy".format(i), y)\n    train_data_time = prepare_date_features(train_data_part).sort_values(by=[\'customer_ID\'])\n    train_data_num = prepare_numerical_features(train_data_part).sort_values(by=[\'customer_ID\'])\n    train_data_cat = prepare_cat_features(train_data_part).sort_values(by=[\'customer_ID\'])\n    assert list(train_data_time.axes[0])==list(train_data_num.axes[0])==list(train_data_cat.axes[0])\n    ### Save to Pickle\n    train_data_time.merge(train_data_cat, left_index=True, right_index=True).merge(train_data_num, left_index=True, right_index=True).to_pickle(\'train_data_{}.pkl\'.format(i))\n\n    del train_data_time, train_data_num, train_data_cat, train_data_part, y\n    gc.collect()\n')


# **Applying on Test Data**

# In[9]:


get_ipython().run_cell_magic('time', '', "\n# https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length\ndef split(a, n):\n    k, m = divmod(len(a), n)\n    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))\n\ntest_split_ids = split(test_data.customer_ID.unique(),25)\n\nfor (i,ids) in enumerate(test_split_ids):\n    print(i, len(ids))\n    test_data_part = test_data[test_data.customer_ID.isin(ids)].sort_values(by=['customer_ID'])\n    \n    test_data_time = prepare_date_features(test_data_part).sort_values(by=['customer_ID'])\n    test_data_num = prepare_numerical_features(test_data_part).sort_values(by=['customer_ID'])\n    test_data_cat = prepare_cat_features(test_data_part).sort_values(by=['customer_ID'])\n    assert list(test_data_part.axes[0])==list(test_data_part.axes[0])==list(test_data_part.axes[0])\n    ### Save to Pickle\n    test_data_time.merge(test_data_cat, left_index=True, right_index=True).merge(test_data_num, left_index=True, right_index=True).to_pickle('test_data_{}.pkl'.format(i))\n\n    del test_data_time, test_data_num, test_data_cat, test_data_part\n    gc.collect()\n")


# **<h3>FEATURE IMPORTANCE FINDINGS (XGBOOST)</h3>**
# 
# <h3>
# I trained 100 XGBOOST models (with different parameters) in order to validate the importance of Features generated using the notebook (All the files are in the directory ../input/amexxgboostfeature-importances) </h3>

# In[10]:


feature_importance_list = []

for file_path in tqdm.tqdm(glob.glob('../input/amexxgboostfeature-importances/XGB_model_*')):
    file = pd.read_pickle(file_path)
    feature_importance_list.append(file)
feature_importance_list = pd.concat([fe.T for fe in feature_importance_list], axis=0).fillna(0)


# In[11]:


def plot_importance(importance_df, PLOT_TOP_N = 20, figsize=(10, 10)):
    sorted_indices = importance_df.median(axis=0).sort_values(ascending=False).index
    sorted_importance_df = importance_df.loc[:, sorted_indices]
    plot_cols = sorted_importance_df.columns[:PLOT_TOP_N]
    _, ax = plt.subplots(figsize=figsize)
    ax.grid()
    #ax.set_xscale('log')
    ax.set_ylabel('Feature')
    ax.set_xlabel('Importance')
    sns.boxplot(data=sorted_importance_df[plot_cols],
                orient='h',
                ax=ax)
    plt.show()
    
plot_importance(feature_importance_list, PLOT_TOP_N = 100, figsize=(10, 20))


# <h3>As we can see that the most important feature is **P_2_last** and the most useful features are last recorded values of themselves.<br><br>
# 
#     
#     
# So a very common question will be : How about the 2nd last and 3rd last values, do they help?
# <br>The answer is not that much :( <br> Let's plot the graph first.
# </h3>

# In[12]:


def plot_importance_groups(importance_df, PLOT_TOP_N = 1500, figsize=(10, 20)):
    sorted_indices = importance_df.median(axis=0).sort_values(ascending=False).index
    sorted_importance_df = importance_df.loc[:, sorted_indices]
    plot_cols = sorted_importance_df.columns[:PLOT_TOP_N]

    t = sorted_importance_df.transpose()
    t['groups'] = [s.split('_')[-1] for s in sorted_importance_df.columns]
    t = t.groupby('groups').sum().transpose()
    t = t.loc[:,t.columns.isin(['last3','last2','last','max','mean','min','std'])]

    _, ax = plt.subplots(figsize=figsize)
    ax.grid()
    #ax.set_xscale('log')
    ax.set_ylabel('Feature')
    ax.set_xlabel('Importance')
    sns.boxplot(data=t,
                orient='h',
                ax=ax)
    plt.show()
    
    
last_2_cols = dict([(c, c[:-7]+'_last2') for c in feature_importance_list.columns if c.endswith('_last_2')])
last_3_cols = dict([(c, c[:-7]+'_last3') for c in feature_importance_list.columns if c.endswith('_last_3')])
last_2_cols.update(last_3_cols)
last_2_3_cols = last_2_cols
plot_importance_groups(feature_importance_list.rename(columns=last_2_3_cols))


# <h3>As you can see, the **last2** and **last3** are the least important groups here.
# <br>Maybe those features are not that useful or it may be totally due to XGBOOST's mechanism, one way or the other, it doesn't mean that those features are trash. Those are not useful for the XGBOOST but can still be very useful for NNs or other models.</h3> 

# In[ ]:





# **CONCLUSION**
# 
# * There are some features which don't increase the performance at all, I will remove them in future versions after testing on some models.
# 
# * There are some techniques for numerical features like *binning* haven't used here, since the data is not readable and it's very hard to make bins for these types of data. 

# In[ ]:





# In[ ]:




