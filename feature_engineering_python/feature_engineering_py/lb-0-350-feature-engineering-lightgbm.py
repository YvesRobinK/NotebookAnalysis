#!/usr/bin/env python
# coding: utf-8

# # Vote up me, thanks so much 

# In[1]:


import numpy as np
import pandas as pd
from itertools import groupby
from sklearn.model_selection import train_test_split
from pandas.api.types import is_datetime64_ns_dtype
from imblearn.under_sampling import RandomUnderSampler
import lightgbm as lgb
import matplotlib.pyplot as plt
import plotly.express as px
from joblib import Parallel, delayed
import gc

import warnings
warnings.filterwarnings("ignore")


# In[2]:


def reduce_mem_usage(df):
    
    """ 
    Iterate through all numeric columns of a dataframe and modify the data type
    to reduce memory usage.        
    """
    
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and not is_datetime64_ns_dtype(df[col]) and not 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int32)  
            else:

                df[col] = df[col].astype(np.float16)
        
    return df


# In[3]:


# def conv1d(s):
#     if len(s)>2:
#         return mean(np.multiply(s, [1,0,1]))
#     else:
#         return np.nan

# [conv1d(s) for s in df['enmo'].rolling(3, center=True)]


# In[4]:


signal_awake = dict(zip(range(1440), np.sin(np.linspace(0, np.pi, 1440) + 0.208 * np.pi) ** 24))
signal_onset = dict(zip(range(1440), np.sin(np.linspace(0, np.pi, 1440) + 0.555 * np.pi) ** 24))

def feat_eng(df):
    
    df['series_id'] = df['series_id'].astype('category')
    df['timestamp'] = pd.to_datetime(df['timestamp']).apply(lambda t: t.tz_localize(None))
    
    df.sort_values(['timestamp'], inplace=True)
    
    df['signal_onset'] = (df.timestamp.dt.hour * 60 + df.timestamp.dt.minute).map(signal_onset).astype(np.float32)
    df['signal_awake'] = (df.timestamp.dt.hour * 60 + df.timestamp.dt.minute).map(signal_awake).astype(np.float32)
    df["anglez_diff"] = df["anglez"].diff().astype(np.float32)
    df["anglez_diffabs"] = abs(df["anglez_diff"]).astype(np.float32)
    df["anglezabs"] = abs(df["anglez"]).astype(np.float32)
    df['anglez_x_enmo'] = (df['anglez'] * df['enmo']).astype(np.float32)
    
    df.set_index('timestamp', inplace=True)
    
    df['lids'] = np.maximum(0., df['enmo'] - 0.02)
    df['lids'] = df['lids'].rolling(f'{120*5}s', center=True, min_periods=1).agg('sum')
    df['lids'] = 100 / (df['lids'] + 1)
    df['lids'] = df['lids'].rolling(f'{360*5}s', center=True, min_periods=1).agg('mean').astype(np.float32)
    
    
    for col in ['enmo', 'anglez', 'anglez_x_enmo', 'anglezabs', 'anglez_diff', "anglez_diffabs"]:
        
        for n in [21, 61]:
            df[f'{col}_diff_{n}'] = df[col].diff(periods=n).astype(np.float32)
        
            rol_args = {'window':f'{n*5}s', 'min_periods':1, 'center':True}
            
            for agg in ['median', 'mean', 'max', 'min', 'std']:
                df[f'{col}_{agg}_{n}'] = df[col].rolling(**rol_args).agg(agg).astype(np.float32).values
                gc.collect()
            
            df[f'{col}_mad_{n}'] = (df[col] - df[f'{col}_median_{n}']).abs().rolling(**rol_args).median().astype(np.float32)

            df[f'{col}_amplit_{n}'] = df[f'{col}_max_{n}']-df[f'{col}_min_{n}']
            df[f'{col}_diff_{n}_max'] = df[f'{col}_max_{n}'].rolling(**rol_args).max().astype(np.float32)
            df[f'{col}_medianxstd_{n}'] = df[f'{col}_median_{n}'] * df[f'{col}_std_{n}']
    
            gc.collect()
        
#         df[f'conv1d_{col}']

    df.drop(columns=['anglez_x_enmo', 'anglez_diffabs', 'anglez_diff', 'anglez',], inplace = True)
    
    df.drop(columns=[
        'anglez_x_enmo_diff_21', 'anglez_std_21', 'anglezabs_std_21', 'anglezabs_mad_21', 
        'anglezabs_amplit_21', 'anglez_diff_21', 'anglez_x_enmo_std_21', 'anglez_diff_median_21', 
        'anglez_diff_mean_21', 'anglez_diff_diff_21', 'anglez_x_enmo_mean_21', 'anglez_diff_amplit_21',
        'anglez_diff_medianxstd_21', 'enmo_diff_21', 'anglez_diffabs_diff_21',
        'anglez_diff_max_21', 'anglez_diffabs_max_21', 'anglez_diffabs_amplit_21'
    ], inplace = True)
    
    
    
    df.reset_index(inplace=True)
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    df = reduce_mem_usage(df)

    return df


# In[5]:


file = '/kaggle/input/zzzs-lightweight-training-dataset-target/Zzzs_train_multi.parquet'

def feat_eng_by_id(idx):
    
    df  = pd.read_parquet(file, filters=[('series_id','=',idx)])
    df['awake'] = df['awake'].astype(np.int8)
    df = feat_eng(df)
    
    return df


# In[6]:


series_id  = pd.read_parquet(file, columns=['series_id'])
series_id = series_id.series_id.unique()


# In[7]:


get_ipython().run_cell_magic('time', '', '\ntrain = Parallel(n_jobs=6)(delayed(feat_eng_by_id)(i) for i in series_id)\ntrain = pd.concat(train)\n')


# In[8]:


train.shape


# In[9]:


drop_cols = ['series_id', 'step', 'timestamp']

# X = train.drop(columns=drop_cols+['awake'])
# y = train['awake']


# In[10]:


train['awake'].value_counts()


# In[11]:


sampling_strategy = {0: 4500020 , 1: 5000000, 2:3120000  }
rus = RandomUnderSampler(random_state=42, sampling_strategy=sampling_strategy)
X, y = rus.fit_resample(train.drop(columns=drop_cols+['awake']), train['awake'])
gc.collect()


# In[12]:


del train
gc.collect()


# In[13]:


lgb_params = {    
    'boosting_type':'gbdt',
    'num_leaves':35,
    'max_depth':6,
    'learning_rate':0.0125,
    'n_estimators':200,
    'subsample_for_bin':200000,
    'min_child_weight':0.001,
    'min_child_samples':20,
    'subsample':0.75,
#     'colsample_bytree':0.7,
    'reg_alpha':0.05,
    'reg_lambda':0.05,
    'random_state':666
             }


m = lgb.LGBMClassifier(**lgb_params)
m.fit(X,y)


# In[14]:


X.shape


# In[15]:


feat_imp = pd.Series(m.feature_importances_, index=X.columns).sort_values()
fig = px.bar(x=feat_imp, y=feat_imp.index, orientation='h')
fig.show()


# In[16]:


feat_imp[feat_imp==0].index


# In[17]:


del X, y
gc.collect()


# In[18]:


def predict_test(idx):
    
    test  = pd.read_parquet('/kaggle/input/child-mind-institute-detect-sleep-states/test_series.parquet',
                            filters=[('series_id','=',idx)])
    test = feat_eng(test)

    X_test = test.drop(columns=drop_cols)
    p = m.predict_proba(X_test)
    
    test["not_awake"] = p[:,0]
    test["awake"]     = p[:,1]
    
    smoothing_length = 2*230

    test["score"]  = test["awake"].rolling(smoothing_length, center=True).mean().fillna(method="bfill").fillna(method="ffill")
    test["smooth"] = test["not_awake"].rolling(smoothing_length, center=True).mean().fillna(method="bfill").fillna(method="ffill")
    # re-binarize
    test["smooth"] = test["smooth"].round()

    # https://stackoverflow.com/questions/73777727/how-to-mark-start-end-of-a-series-of-non-null-and-non-0-values-in-a-column-of-a
    def get_event(df):
        lstCV = zip(df.series_id, df.smooth)
        lstPOI = []
        for (c, v), g in groupby(lstCV, lambda cv: 
                                (cv[0], cv[1]!=0 and not pd.isnull(cv[1]))):
            llg = sum(1 for item in g)
            if v is False: 
                lstPOI.extend([0]*llg)
            else: 
                lstPOI.extend(['onset']+(llg-2)*[0]+['wakeup'] if llg > 1 else [0])
        return lstPOI

    test['event'] = get_event(test)
    
    return test.loc[test['event'] != 0][['series_id','step','event','score']]


# In[19]:


series_id  = pd.read_parquet('/kaggle/input/child-mind-institute-detect-sleep-states/test_series.parquet', columns=['series_id'])
series_id = series_id.series_id.unique()
tests = []

for idx in series_id:
    tests.append(predict_test(idx))


# In[20]:


test = pd.concat(tests)


# In[21]:


test


# In[22]:


sample_submission = test.copy().reset_index(drop=True).reset_index(names='row_id')
sample_submission.to_csv('submission.csv', index=False)


# In[23]:


sample_submission


# In[ ]:




