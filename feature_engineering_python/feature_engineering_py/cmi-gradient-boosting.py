#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from itertools import groupby
import gc


# Inspired from the [work](http://www.kaggle.com/code/carlmcbrideellis/zzzs-random-forest-model-starter) by CARL MCBRIDE ELLIS

# In[2]:


train = pd.read_parquet("/kaggle/input/zzzs-lightweight-training-dataset-target/Zzzs_train_multi.parquet")


# #### Feature engineering

# In[3]:


def make_features(df):
    # parse the timestamp and create an "hour" feature
    df['timestamp'] = pd.to_datetime(df['timestamp']).apply(lambda t: t.tz_localize(None))
    df["hour"] = df["timestamp"].dt.hour
    
    periods = 20
    df["anglez"] = abs(df["anglez"])
    df["anglez_diff"] = df.groupby('series_id')['anglez'].diff(periods=periods).fillna(method="bfill").astype('float16')
    df["enmo_diff"] = df.groupby('series_id')['enmo'].diff(periods=periods).fillna(method="bfill").astype('float16')
    df["anglez_rolling_mean"] = df["anglez"].rolling(periods,center=True).mean().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["enmo_rolling_mean"] = df["enmo"].rolling(periods,center=True).mean().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["anglez_rolling_max"] = df["anglez"].rolling(periods,center=True).max().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["enmo_rolling_max"] = df["enmo"].rolling(periods,center=True).max().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["anglez_rolling_std"] = df["anglez"].rolling(periods,center=True).std().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["enmo_rolling_std"] = df["enmo"].rolling(periods,center=True).std().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["anglez_diff_rolling_mean"] = df["anglez_diff"].rolling(periods,center=True).mean().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["enmo_diff_rolling_mean"] = df["enmo_diff"].rolling(periods,center=True).mean().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["anglez_diff_rolling_max"] = df["anglez_diff"].rolling(periods,center=True).max().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["enmo_diff_rolling_max"] = df["enmo_diff"].rolling(periods,center=True).max().fillna(method="bfill").fillna(method="ffill").astype('float16')
    
    return df

features = ["hour","anglez","anglez_rolling_mean","anglez_rolling_max","anglez_rolling_std","anglez_diff",
            "anglez_diff_rolling_mean","anglez_diff_rolling_max","enmo","enmo_rolling_mean",
            "enmo_rolling_max","enmo_rolling_std","enmo_diff","enmo_diff_rolling_mean","enmo_diff_rolling_max",]


# #### Training

# In[4]:


train   = make_features(train)

X_train = train[features]
y_train = train["awake"]

# save some memory
del train
gc.collect();


# In[5]:


get_ipython().run_cell_magic('time', '', '\nfrom sklearn.ensemble import GradientBoostingClassifier\nclassifier = GradientBoostingClassifier(min_samples_leaf=100,n_estimators=50)\n\nclassifier.fit(X_train, y_train)\n\n\ndel X_train, y_train\ngc.collect();\n')


# #### Predictions

# In[6]:


test  = pd.read_parquet("/kaggle/input/child-mind-institute-detect-sleep-states/test_series.parquet")

test  = make_features(test)

X_test = test[features]

test["not_awake"] = classifier.predict_proba(X_test)[:,0]
test["awake"]     = classifier.predict_proba(X_test)[:,1]


# #### Processing

# In[7]:


# smoothing the predictions
smoothing_length = 2*251
test["score"]  = test["awake"].rolling(smoothing_length,center=True).mean().fillna(method="bfill").fillna(method="ffill")
test["smooth"] = test["not_awake"].rolling(smoothing_length,center=True).mean().fillna(method="bfill").fillna(method="ffill")
# re-binarize
test["smooth"] = test["smooth"].round()


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

test["event"] = get_event(test)


# #### Write in submission file

# In[8]:


sample_submission = test.loc[test["event"] != 0][["series_id","step","event","score"]].copy().reset_index(drop=True).reset_index(names="row_id")
sample_submission.to_csv('submission.csv', index=False)

