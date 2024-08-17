#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering and Random Forest Prediction to Detect Sleep States
# 
# I took Lucas's public notebook and add Optuna to find better hyperparameters. 
# 
# Thanks to Lucas Burke!
# Please upvote his great notebook: https://www.kaggle.com/code/lccburk/feature-engineering-and-random-forest-prediction
# 

# In[2]:


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

from metric import score # Import event detection ap score function

# These are variables to be used by the score function
column_names = {
    'series_id_column_name': 'series_id',
    'time_column_name': 'step',
    'event_column_name': 'event',
    'score_column_name': 'score',
}

tolerances = {
    'onset': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360], 
    'wakeup': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360]
}

#"Dla obu klas używamy progów tolerancji błędu wynoszących 1, 3, 5, 7,5, 10, 12,5, 15, 20, 25, 30 minut 
# lub 12, 36, 60, 90, 120, 150, 180, 240, 300, 360 step"


# In[5]:


get_ipython().system('pip install -q /kaggle/input/xgboost-2-0-0-whl/xgboost-2.0.0-py3-none-manylinux2014_x86_64.whl')

import xgboost as xgb

xgb.__version__


# ## Importing data

# In[6]:


# Importing data 

# Column transformations

# przekształcenia daty:
dt_transforms = [
    pl.col('timestamp').str.to_datetime(), 
    (pl.col('timestamp').str.to_datetime().dt.year()-2000).cast(pl.UInt8).alias('year'), 
    pl.col('timestamp').str.to_datetime().dt.month().cast(pl.UInt8).alias('month'),
    pl.col('timestamp').str.to_datetime().dt.day().cast(pl.UInt8).alias('day'), 
    pl.col('timestamp').str.to_datetime().dt.hour().cast(pl.UInt8).alias('hour')
]

# przekształcenia anglez i enmo
data_transforms = [
    pl.col('anglez').cast(pl.Int16), # Casting anglez to 16 bit integer
    (pl.col('enmo')*1000).cast(pl.UInt16), # Convert enmo to 16 bit uint
]


#wczytanie train i test
train_series = pl.scan_parquet('/kaggle/input/child-mind-institute-detect-sleep-states/train_series.parquet').with_columns(
    dt_transforms + data_transforms
    )

train_events = pl.read_csv('/kaggle/input/child-mind-institute-detect-sleep-states/train_events.csv').with_columns(
    dt_transforms
    )

test_series = pl.scan_parquet('/kaggle/input/child-mind-institute-detect-sleep-states/test_series.parquet').with_columns(
    dt_transforms + data_transforms
    )


# In[7]:


train_events.head(2)


# * Tworzy listę unikalnych identyfikatorów serii (series_id) z danych zawartych w kolumnie 'series_id' w train_events. Te identyfikatory serii są przechowywane w zmiennej series_ids.
# 
# * Usuwa serie, które mają niezgodne liczby wystąpień zdarzeń 'onset' i 'wakeup'. Aby to zrobić, najpierw grupuje dane w train_events według 'series_id' i zlicza wystąpienia 'onset' i 'wakeup' w każdej serii. Następnie tworzy ramkę danych counts, która zawiera informacje o identyfikatorze serii oraz liczbach wystąpień 'onset' i 'wakeup' w każdej serii. count_mismatches to ramka danych zawierająca serie, dla których liczby wystąpień 'onset' i 'wakeup' są różne.
# 
# * Filtruje train_series i train_events, usuwając serie z identyfikatorami, które występują w count_mismatches. W ten sposób usuwa serie, które mają niezgodne liczby wystąpień 'onset' i 'wakeup'.
# 
# * Aktualizuje listę identyfikatorów serii series_ids, nie zawierając serii bez wartości nie-null (usuwa serie, które nie mają wartości 'series_id' w train_events)
# 
# Po co? w danych train_events są przypadki, że człowiek poszedł spać a się nie obudził lub na odwrót ;)

# In[8]:


# Getting series ids as a list for convenience
series_ids = train_events['series_id'].unique(maintain_order=True).to_list()

# Removing series with mismatched counts: 
onset_counts = train_events.filter(pl.col('event')=='onset').group_by('series_id').count().sort('series_id')['count']
wakeup_counts = train_events.filter(pl.col('event')=='wakeup').group_by('series_id').count().sort('series_id')['count']

counts = pl.DataFrame({'series_id':sorted(series_ids), 'onset_counts':onset_counts, 'wakeup_counts':wakeup_counts})
count_mismatches = counts.filter(counts['onset_counts'] != counts['wakeup_counts'])

train_series = train_series.filter(~pl.col('series_id').is_in(count_mismatches['series_id']))
train_events = train_events.filter(~pl.col('series_id').is_in(count_mismatches['series_id']))

# Updating list of series ids, not including series with no non-null values.
series_ids = train_events.drop_nulls()['series_id'].unique(maintain_order=True).to_list()


# In[9]:


count_mismatches


# Trzeba sprawdzić, czy ta funkcja jest potrzebna i/lub ją zmodyfikować. 
# Link do dyskusji o aktualizacji danych: https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/440697

# ## Feature Engineering

# In[10]:


features, feature_cols = [pl.col('hour')], ['hour']

for mins in [5, 30, 60*2, 60*8] :
    
    for var in ['enmo', 'anglez'] :
        
        features += [
            pl.col(var).rolling_mean(12 * mins, center=True, min_periods=1).abs().cast(pl.UInt16).alias(f'{var}_{mins}m_mean'),
            pl.col(var).rolling_max(12 * mins, center=True, min_periods=1).abs().cast(pl.UInt16).alias(f'{var}_{mins}m_max'),
            pl.col(var).rolling_std(12 * mins, center=True, min_periods=1).abs().cast(pl.UInt16).alias(f'{var}_{mins}m_std')
        ]

        feature_cols += [ 
            f'{var}_{mins}m_mean', f'{var}_{mins}m_max', f'{var}_{mins}m_std'
        ]

        # Getting first variations
        features += [
            (pl.col(var).diff().abs().rolling_mean(12 * mins, center=True, min_periods=1)*10).abs().cast(pl.UInt32).alias(f'{var}_1v_{mins}m_mean'),
            (pl.col(var).diff().abs().rolling_max(12 * mins, center=True, min_periods=1)*10).abs().cast(pl.UInt32).alias(f'{var}_1v_{mins}m_max'),
            (pl.col(var).diff().abs().rolling_std(12 * mins, center=True, min_periods=1)*10).abs().cast(pl.UInt32).alias(f'{var}_1v_{mins}m_std')
        ]

        feature_cols += [ 
            f'{var}_1v_{mins}m_mean', f'{var}_1v_{mins}m_max', f'{var}_1v_{mins}m_std'
        ]

id_cols = ['series_id', 'step', 'timestamp']

train_series = train_series.with_columns(
    features
).select(id_cols + feature_cols)

test_series = test_series.with_columns(
    features
).select(id_cols + feature_cols)


# In[11]:


def make_train_dataset(train_data, train_events, drop_nulls=False) :
    
    series_ids = train_data['series_id'].unique(maintain_order=True).to_list()
    X, y = pl.DataFrame(), pl.DataFrame()
    for idx in tqdm(series_ids) : 
        
        # Normalizing sample features
        sample = train_data.filter(pl.col('series_id')==idx).with_columns(
            [(pl.col(col) / pl.col(col).std()).cast(pl.Float32) for col in feature_cols if col != 'hour']
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


# In[12]:


def get_events(series, classifier) :
    '''
    Takes a time series and a classifier and returns a formatted submission dataframe.
    '''
    
    series_ids = series['series_id'].unique(maintain_order=True).to_list()
    events = pl.DataFrame(schema={'series_id':str, 'step':int, 'event':str, 'score':float})

    for idx in tqdm(series_ids) : 

        # Collecting sample and normalizing features
        scale_cols = [col for col in feature_cols if (col != 'hour') & (series[col].std() !=0)]
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


# In[ ]:


# def reduce_mem_usage(df):
#     """
#     Iterate through all numeric columns of a Polars DataFrame and modify the data type
#     to reduce memory usage.
#     """

#     col_names = df.schema.get_field_names()
#     df_dict = {}

#     for col_name in col_names:
#         col = df[col_name]
#         col_type = col.data_type()

#         if isinstance(col_type, (Int8, Int16, Int32, Int64)):
#             c_min = col.min().unwrap()
#             c_max = col.max().unwrap()

#             if isinstance(col_type, Int8):
#                 if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
#                     col = col.cast(pl.Int8)
#                 elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
#                     col = col.cast(pl.Int16)
#                 elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
#                     col = col.cast(pl.Int32)
#                 elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
#                     col = col.cast(pl.Int64)
#             elif isinstance(col_type, Float32):
#                 if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
#                     col = col.cast(pl.Float32)
#                 else:
#                     col = col.cast(pl.Float64)

#         df_dict[col_name] = col

#     df_optimized = pl.DataFrame(df_dict)
#     df_optimized = df_optimized.with_column('series_id', df['series_id'].cast(Categorical))

#     return df_optimized


# ## Training Models

# In[ ]:


# reduce_mem_usage(train_series)


# In[13]:


train_ids, val_ids = train_test_split(series_ids, train_size=0.7, random_state=42, shuffle = False)

# We will collect datapoints at 5 minute intervals for training for validating
# train_data = train_series.filter(pl.col('series_id').is_in(train_ids)).take_every(12 * 5).collect()

# train_data = train_series.filter(pl.col('series_id').is_in(train_ids)).collect()

train_data = train_series.filter(pl.col('series_id').is_in(series_ids)).collect().sample(int(1e6))


val_data = train_series.filter(pl.col('series_id').is_in(val_ids)).collect()
val_solution = train_events.filter(pl.col('series_id').is_in(val_ids)).select(['series_id', 'event', 'step']).to_pandas()


# In[14]:


# Creating train dataset
X_train, y_train = make_train_dataset(train_data, train_events)


# ### Training and validating random forest

# In[15]:


X_train.head(2)


# ## Optuna

# In[ ]:


# def objective(trial):
#     # Define the hyperparameters to search over
#     n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=100)
#     min_samples_leaf = trial.suggest_int('min_samples_leaf', 100, 500)
#     max_depth = trial.suggest_int('max_depth', 5, 20)  # Add max_depth as a hyperparameter

#     # Create and train the random forest classifier
#     rf_classifier = RandomForestClassifier(
#         n_estimators=n_estimators,
#         min_samples_leaf=min_samples_leaf,
#         max_depth=max_depth,  # Use the suggested max_depth value
#         random_state=42,
#         n_jobs=-1
#     )

#     rf_classifier.fit(X_train[feature_cols], y_train)

#     # Make predictions on the validation set
#     y_pred = rf_classifier.predict(X_val[feature_cols])

#     # Calculate the F1 score as the objective to minimize
#     f1 = f1_score(y_val, y_pred)

#     return 1 - f1  # Optuna minimizes, so we return 1 - f1 to minimize F1 score

# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# study = optuna.create_study(direction='minimize')  # We want to minimize the F1 score
# study.optimize(objective, n_trials=100)  # You can adjust the number of trials as needed

# best_params = study.best_params
# best_score = 1 - study.best_value  # Convert back to F1 score

# print("Best Hyperparameters:", best_params)
# print("Best F1 Score:", best_score)

# [I 2023-09-25 11:12:55,468] A new study created in memory with name: no-name-9526f79b-21e9-4e9d-ad84-c314ece354ab
# [I 2023-09-25 11:52:54,599] Trial 0 finished with value: 0.06946539490413939 and parameters: {'n_estimators': 700, 'min_samples_leaf': 146, 'max_depth': 13}. Best is trial 0 with value: 0.06946539490413939.
# [I 2023-09-25 13:03:29,482] Trial 1 finished with value: 0.07323151639082381 and parameters: {'n_estimators': 1000, 'min_samples_leaf': 377, 'max_depth': 19}. Best is trial 0 with value: 0.06946539490413939.
# [I 2023-09-25 13:37:44,074] Trial 2 finished with value: 0.07325825955419085 and parameters: {'n_estimators': 700, 'min_samples_leaf': 187, 'max_depth': 11}. Best is trial 0 with value: 0.06946539490413939.
# [I 2023-09-25 13:51:40,476] Trial 3 finished with value: 0.07031010973254115 and parameters: {'n_estimators': 200, 'min_samples_leaf': 243, 'max_depth': 18}. Best is trial 0 with value: 0.06946539490413939.
# [I 2023-09-25 14:16:29,696] Trial 4 finished with value: 0.07151886891478065 and parameters: {'n_estimators': 400, 'min_samples_leaf': 275, 'max_depth': 15}. Best is trial 0 with value: 0.06946539490413939.
# [I 2023-09-25 14:38:45,216] Trial 5 finished with value: 0.07515495515949588 and parameters: {'n_estimators': 500, 'min_samples_leaf': 269, 'max_depth': 10}. Best is trial 0 with value: 0.06946539490413939.
# [I 2023-09-25 14:47:46,083] Trial 6 finished with value: 0.07710140184366987 and parameters: {'n_estimators': 200, 'min_samples_leaf': 467, 'max_depth': 10}. Best is trial 0 with value: 0.06946539490413939.
# [I 2023-09-25 15:14:30,617] Trial 7 finished with value: 0.07310587260317014 and parameters: {'n_estimators': 500, 'min_samples_leaf': 288, 'max_depth': 12}. Best is trial 0 with value: 0.06946539490413939.
# [I 2023-09-25 15:23:29,831] Trial 8 finished with value: 0.07534936936446102 and parameters: {'n_estimators': 200, 'min_samples_leaf': 283, 'max_depth': 10}. Best is trial 0 with value: 0.06946539490413939.
# [I 2023-09-25 16:02:09,101] Trial 9 finished with value: 0.07403436401137597 and parameters: {'n_estimators': 800, 'min_samples_leaf': 222, 'max_depth': 11}. Best is trial 0 with value: 0.06946539490413939.
# [I 2023-09-25 16:26:45,116] Trial 10 finished with value: 0.08593876559209446 and parameters: {'n_estimators': 1000, 'min_samples_leaf': 103, 'max_depth': 5}. Best is trial 0 with value: 0.06946539490413939.
# [I 2023-09-25 16:34:26,003] Trial 11 finished with value: 0.06653632867100934 and parameters: {'n_estimators': 100, 'min_samples_leaf': 122, 'max_depth': 19}. Best is trial 11 with value: 0.06653632867100934.
# [I 2023-09-25 17:26:56,238] Trial 12 finished with value: 0.06678471353857662 and parameters: {'n_estimators': 800, 'min_samples_leaf': 112, 'max_depth': 16}. Best is trial 11 with value: 0.06653632867100934.


# In[ ]:


# # Plotting feature importances
# px.bar(x=feature_cols, 
#        y=rf_classifier.feature_importances_,
#        title='Random forest feature importances'
#       )


# In[ ]:


#for 1mln and 10 minutes:
# rf_classifier = RandomForestClassifier(n_estimators=700,
#                                     min_samples_leaf=101,
#                                     random_state=42,
#                                     n_jobs=-1)

#RF with hyperparameters from Optuna

# Random Forest with hyperparameters from Optuna

# rf_classifier = RandomForestClassifier(n_estimators=100,
#                                     min_samples_leaf=122,
#                                     max_depth = 19,
#                                     random_state=42,
#                                     n_jobs=-1)

# rf_classifier.fit(X_train[feature_cols], y_train)


# In[ ]:


# # Checking performance on validation set
# rf_submission = get_events(val_data, rf_classifier)

# print(f"Random forest score: {score(val_solution, rf_submission, tolerances, **column_names)}")


# In[ ]:


# Saving classifier 
# import pickle
# with open('rf_classifier_5m_8h.pkl', 'wb') as f:
#     pickle.dump(rf_classifier, f)

#with open('rf_classifier.pkl', 'rb') as f:
#    rf_classifier = pickle.load(f)


# # XGBoost 2.0

# In[16]:


# import gc
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=520, objective="binary:logistic", learning_rate=0.02, max_depth=7, random_state=42)


# In[17]:


model.fit(X_train[feature_cols], y_train)

# Checking performance on validation set
val_submission = get_events(val_data, model)

print(f"Model score: {score(val_solution, val_submission, tolerances, **column_names)}")


# ## Applying to test data

# In[ ]:


# Recovering memory
del train_data 


# In[ ]:


# Getting event predictions for test set and saving submission
submission = get_events(test_series.collect(), model)
submission.to_csv('submission.csv', index=False)


# In[ ]:


submission


# * Why submission is empty? https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/444673
# 
# * Playing with the competition metric: https://www.kaggle.com/code/carlmcbrideellis/zzzs-playing-with-the-competition-metric
