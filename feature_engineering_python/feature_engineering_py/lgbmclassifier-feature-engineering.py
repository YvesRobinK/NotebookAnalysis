#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import polars as pl
import datetime 
from tqdm import tqdm

import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import gc
import lightgbm as lgb
from sklearn.ensemble import BaggingClassifier, StackingClassifier
import lightgbm as lgbm


# In[2]:


# Importing data 

# Column transformations

dt_transforms = [
    pl.col('timestamp').str.to_datetime(), 
    (pl.col('timestamp').str.to_datetime().dt.year()-2000).cast(pl.UInt8).alias('year'), 
    pl.col('timestamp').str.to_datetime().dt.month().cast(pl.UInt8).alias('month'),
    pl.col('timestamp').str.to_datetime().dt.day().cast(pl.UInt8).alias('day'), 
    pl.col('timestamp').str.to_datetime().dt.hour().cast(pl.UInt8).alias('hour'),
    pl.col('timestamp').str.to_datetime().dt.minute().cast(pl.UInt8).alias('minute') # Add minute
]

data_transforms = [
    pl.col('anglez').cast(pl.Float32), # Casting anglez to Float32
    (pl.col('enmo')).cast(pl.Float32), # Convert enmo to Float32
]

train_series = pl.scan_parquet('/kaggle/input/child-mind-institute-detect-sleep-states/train_series.parquet').with_columns(
    dt_transforms + data_transforms
    ) # -> LazyFrame

train_events = pl.read_csv('/kaggle/input/child-mind-institute-detect-sleep-states/train_events.csv').with_columns(
    dt_transforms
    ) # -> DataFramm

test_series = pl.scan_parquet('/kaggle/input/child-mind-institute-detect-sleep-states/test_series.parquet').with_columns(
    dt_transforms + data_transforms
    ) # -> LazyFrame


# In[3]:


train_events = train_events.filter(~((train_events['series_id'] == '0ce74d6d2106') & (train_events['night'] == 20)) |
                              ((train_events['series_id'] == '154fe824ed87') & (train_events['night'] == 30)) |
                              ((train_events['series_id'] == '44a41bba1ee7') & (train_events['night'] == 10)) |
                              ((train_events['series_id'] == 'efbfc4526d58') & (train_events['night'] == 7)) |
                              ((train_events['series_id'] == 'f8a8da8bdd00') & (train_events['night'] == 17)))
train_events.shape


# In[4]:


# Getting series ids as a list for convenience
series_ids = train_events['series_id'].unique(maintain_order=True).to_list()

# Removing series with mismatched counts: 
onset_counts = train_events.filter(pl.col('event')=='onset').group_by('series_id').count().sort('series_id')['count']
wakeup_counts = train_events.filter(pl.col('event')=='wakeup').group_by('series_id').count().sort('series_id')['count']

counts = pl.DataFrame({'series_id':sorted(series_ids), 'onset_counts':onset_counts, 'wakeup_counts':wakeup_counts})
count_mismatches = counts.filter(counts['onset_counts'] != counts['wakeup_counts'])
print(count_mismatches.shape)
train_series = train_series.filter(~pl.col('series_id').is_in(count_mismatches['series_id']))
train_events = train_events.filter(~pl.col('series_id').is_in(count_mismatches['series_id']))


# In[5]:


train_events = train_events.with_columns(pl.when(pl.col("hour") <= 12)
    .then(pl.col("hour"))
    .otherwise(24-pl.col("hour"))
    .alias("hour_scaled")
)                                       


# In[6]:


all_night = train_events.group_by(['series_id', 'night']).sum().sort(['series_id', 'night'])
all_night = all_night[['series_id', 'night', 'hour_scaled']]
print(all_night.shape)
invalid_night = all_night.filter(pl.col('hour_scaled') > 24).sort(['series_id', 'night'])
print(invalid_night.shape)


# In[7]:


valid_night = all_night.filter(pl.col('hour_scaled') <= 24).sort(['series_id', 'night'])
print(valid_night.shape)
print(valid_night.shape[0]*2)


# In[8]:


train_events = train_events.drop("hour_scaled")


# In[9]:


train_events = train_events \
    .join(valid_night, on=['series_id', 'night'], how='inner') \
    .select(train_events.columns)
train_events.shape


# In[10]:


valid_id = train_events['series_id'].unique()
train_series = train_series.filter(pl.col('series_id').is_in(valid_id))


# In[11]:


#train_series = train_series.sort("timestamp")
#test_series = train_series.sort("timestamp")
series_ids = train_events.drop_nulls()['series_id'].unique(maintain_order=True).to_list()


# ## Features Engineering

# In[12]:


features, feature_cols = [pl.col('hour')], ['hour']

for mins in [5, 30, 60*2, 60*8] :
    features += [
        (pl.col('enmo')*1000).cast(pl.UInt16).rolling_mean(12 * mins, center=True, min_periods=1).abs().cast(pl.UInt16).alias(f'enmo_{mins}m_mean'),
        (pl.col('enmo')*1000).cast(pl.UInt16).rolling_max(12 * mins, center=True, min_periods=1).abs().cast(pl.UInt16).alias(f'enmo_{mins}m_max')
    ]

    feature_cols += [ 
        f'enmo_{mins}m_mean', f'enmo_{mins}m_max'
    ]

    # Getting first variations
    features += [
        ((pl.col('enmo')*1000).cast(pl.UInt16).diff().abs().rolling_mean(12 * mins, center=True, min_periods=1)*10).abs().cast(pl.UInt32).alias(f'enmo_1v_{mins}m_mean'),
        ((pl.col('enmo')*1000).cast(pl.UInt16).diff().abs().rolling_max(12 * mins, center=True, min_periods=1)*10).abs().cast(pl.UInt32).alias(f'enmo_1v_{mins}m_max')
    ]

    feature_cols += [ 
        f'enmo_1v_{mins}m_mean', f'enmo_1v_{mins}m_max'
    ]
    
    features += [
        (pl.col('anglez').diff().abs().rolling_mean(12 * mins, center=True, min_periods=1)*10).abs().cast(pl.UInt32).alias(f'anglez_1v_{mins}m_mean'),
        (pl.col('anglez').diff().abs().rolling_max(12 * mins, center=True, min_periods=1)*10).abs().cast(pl.UInt32).alias(f'anglez_1v_{mins}m_max')
    ]

    feature_cols += [ 
        f'anglez_1v_{mins}m_mean', f'anglez_1v_{mins}m_max'
    ]
        
# Add 'signal' and 'lids'
signal_onset = dict(zip(range(1440), np.sin(np.linspace(0, np.pi, 1440) + 0.208 * np.pi) ** 24))
signal_awake = dict(zip(range(1440), np.sin(np.linspace(0, np.pi, 1440) + 0.555 * np.pi) ** 24)) 

features += [
    # Add 'signal'
    ((pl.col('hour') * 60 + pl.col('minute')).cast(pl.UInt32)).map_dict(signal_onset).cast(pl.Float32).alias('signal_onset'),
    ((pl.col('hour') * 60 + pl.col('minute')).cast(pl.UInt32)).map_dict(signal_awake).cast(pl.Float32).alias('signal_awake'),
    
]

feature_cols += [ 
        'signal_onset', 'signal_awake'
    ]

id_cols = ['series_id', 'step', 'timestamp']

train_series = train_series.with_columns(
    features
).select(id_cols + feature_cols)

test_series = test_series.with_columns(
    features
).select(id_cols + feature_cols)


# In[13]:


def make_train_dataset(train_data, train_events, drop_nulls=False) :
    
    series_ids = train_data['series_id'].unique(maintain_order=True).to_list()
    X, y = pl.DataFrame(), pl.DataFrame()
    for idx in tqdm(series_ids) : 
        
        # Normalizing sample features
        sample = train_data.filter(pl.col('series_id')==idx).with_columns(
            [(pl.col(col) / pl.col(col).std()).cast(pl.Float32) \
             for col in feature_cols if col not in ['hour','signal_onset', 'signal_awake', 'lids']]
        )
        
        events = train_events.filter(pl.col('series_id')==idx)
        
        if drop_nulls : 
            # Removing datapoints on dates where no data was recorded
            sample = sample.filter(
                pl.col('timestamp').dt.date().is_in(events['timestamp'].dt.date())
            )
        
        X = X.vstack(sample[id_cols + feature_cols])

        onsets = events.filter((pl.col('event') == 'onset') & (pl.col('step') != None))['step'].to_list()
        wakeups = events.filter((pl.col('event') == 'wakeup') & (pl.col('step') != None))['step'].to_list()

        # NOTE: This will break if there are event series without any recorded onsets or wakeups
        y = y.vstack(sample.with_columns(
            sum([(onset <= pl.col('step')) & (pl.col('step') <= wakeup) for onset, wakeup in zip(onsets, wakeups)]).cast(pl.Boolean).alias('asleep')
            ).select('asleep')
            )
    
    y = y.to_numpy().ravel()
    
    return X, y


# In[14]:


def get_events(series, classifier) :
    '''
    Takes a time series and a classifier and returns a formatted submission dataframe.
    '''
    
    series_ids = series['series_id'].unique(maintain_order=True).to_list()
    events = pl.DataFrame(schema={'series_id':str, 'step':int, 'event':str, 'score':float})

    for idx in tqdm(series_ids) : 

        # Collecting sample and normalizing features
        scale_cols = [col for col in feature_cols \
                      if (col not in ['hour','signal_onset', 'signal_awake', 'lids']) & (series[col].std() !=0)]
        X = series.filter(pl.col('series_id') == idx).select(id_cols + feature_cols).with_columns(
            [(pl.col(col) / series[col].std()).cast(pl.Float32) for col in scale_cols]
        )

        # Applying classifier to get predictions and scores
        preds, probs = classifier.predict(X[feature_cols]), classifier.predict_proba(X[feature_cols])[:, 1]

        #NOTE: Considered using rolling max to get sleep periods excluding <30 min interruptions, but ended up decreasing performance
        X = X.with_columns(
            pl.lit(preds).cast(pl.Int8).alias('prediction'), 
            pl.lit(probs).alias('probability')
                        )
        
        # Getting predicted onset and wakeup time steps
        pred_onsets = X.filter(X['prediction'].diff() > 0)['step'].to_list()
        pred_wakeups = X.filter(X['prediction'].diff() < 0)['step'].to_list()
        
        if len(pred_onsets) > 0 : 
            
            # Ensuring all predicted sleep periods begin and end
            if min(pred_wakeups) < min(pred_onsets) : 
                pred_wakeups = pred_wakeups[1:]

            if max(pred_onsets) > max(pred_wakeups) :
                pred_onsets = pred_onsets[:-1]

            # Keeping sleep periods longer than 30 minutes
            sleep_periods = [(onset, wakeup) for onset, wakeup in zip(pred_onsets, pred_wakeups) if wakeup - onset >= 12 * 30]

            for onset, wakeup in sleep_periods :
                # Scoring using mean probability over period
                score = X.filter((pl.col('step') >= onset) & (pl.col('step') <= wakeup))['probability'].mean()

                # Adding sleep event to dataframe
                events = events.vstack(pl.DataFrame().with_columns(
                    pl.Series([idx, idx]).alias('series_id'), 
                    pl.Series([onset, wakeup]).alias('step'),
                    pl.Series(['onset', 'wakeup']).alias('event'),
                    pl.Series([score, score]).alias('score')
                ))

    # Adding row id column
    events = events.to_pandas().reset_index().rename(columns={'index':'row_id'})

    return events


# ## Training Models

# In[15]:


train_data = train_series.filter(pl.col('series_id').is_in(series_ids)).take_every(12 * 5).collect()


# In[16]:


train_data.shape


# In[17]:


# Creating train dataset
X_train, y_train = make_train_dataset(train_data, train_events)


# In[18]:


X_train[feature_cols]


# ## Train

# ## 

# In[19]:


lgb_opt = {
    'n_estimators':500,
    'min_samples_leaf':300,
    'boosting_type':'gbdt',
    'random_state':42,
    'n_jobs':-1
}

classifier = lgbm.LGBMClassifier(**lgb_opt)

classifier.fit(X_train[feature_cols], y_train)


# ## Applying to test data

# In[20]:


del train_data
gc.collect()


# In[21]:


# Getting event predictions for the test set using the trained classifier
submission = get_events(test_series.collect(), classifier)

# Saving the submission dataframe to a CSV file
submission.to_csv('submission.csv', index=False)


# In[22]:


submission


# 
