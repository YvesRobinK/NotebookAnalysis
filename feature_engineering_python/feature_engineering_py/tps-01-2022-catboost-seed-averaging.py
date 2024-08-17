#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OrdinalEncoder
from catboost import CatBoostRegressor
import optuna


# # Loading the data
# 
# When reading in our files, we can pass the `'date'` column in a list in the `parse_dates` argument. This transforms the column's datatype from str to datetime64 and allows us to extract more information from the dates later.

# In[2]:


df_train = pd.read_csv('../input/tabular-playground-series-jan-2022/train.csv', parse_dates=['date'])
df_test = pd.read_csv('../input/tabular-playground-series-jan-2022/test.csv', parse_dates=['date'])
sample_submission = pd.read_csv('../input/tabular-playground-series-jan-2022/sample_submission.csv')
df_gdp = pd.read_csv('../input/gdp-20152019-finland-norway-and-sweden/GDP_data_2015_to_2019_Finland_Norway_Sweden.csv', index_col='year')
df_festivities = pd.read_csv('../input/festivities-in-finland-norway-sweden-tsp-0122/nordic_holidays.csv', parse_dates=['date'], usecols=['date', 'country', 'holiday'])


# # Preprocessing and feature engineering
# 
# Instead of using a single `'date'` feature, we will split it up into three separate features of `'year'`, `'month'` and `'day'`. We can derive some additional useful information from the dates, like whether it falls on a weekend, what quarter it's in, and so on. We will also add a time-step feature that counts the days that have passed since the first date in the dataset. Finally we will add GDP and holidays data. Thank you to Carl McBride Ellis and Luca Massaron for their research and work. Please give the original notebooks and datasets an upvote.
# 
# * More on how the scores improve with these additions [here](https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/298300) and [here](https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/298831).
# * More on time-steps in the first lesson of the Time Series course [here](https://www.kaggle.com/ryanholbrook/linear-regression-with-time-series).
# * Original GDP [notebook](https://www.kaggle.com/carlmcbrideellis/gdp-of-finland-norway-and-sweden-2015-2019) and [dataset](https://www.kaggle.com/carlmcbrideellis/gdp-20152019-finland-norway-and-sweden)
# * Original holidays [notebook](https://www.kaggle.com/lucamassaron/festivities-in-finland-norway-sweden/notebook) and [dataset](https://www.kaggle.com/lucamassaron/festivities-in-finland-norway-sweden-tsp-0122)

# In[3]:


df_train['year'] = df_train['date'].dt.year
df_train['quarter'] = df_train['date'].dt.quarter
df_train['month'] = df_train['date'].dt.month
df_train['week'] = df_train['date'].dt.isocalendar().week.astype(int)
df_train['day'] = df_train['date'].dt.day
df_train['dayofyear'] = df_train['date'].dt.dayofyear
df_train['daysinmonth'] = df_train['date'].dt.days_in_month
df_train['dayofweek'] = df_train['date'].dt.dayofweek
df_train['weekend'] = ((df_train['date'].dt.dayofweek) // 5 == 1).astype(int)

df_test['year'] = df_test['date'].dt.year
df_test['quarter'] = df_test['date'].dt.quarter
df_test['month'] = df_test['date'].dt.month
df_test['week'] = df_test['date'].dt.isocalendar().week.astype(int)
df_test['day'] = df_test['date'].dt.day
df_test['dayofyear'] = df_test['date'].dt.dayofyear
df_test['daysinmonth'] = df_test['date'].dt.days_in_month
df_test['dayofweek'] = df_test['date'].dt.dayofweek
df_test['weekend'] = ((df_test['date'].dt.dayofweek) // 5 == 1).astype(int)

t0 = np.datetime64('2015-01-01')
df_train['time_step'] = (df_train.date-t0).astype('timedelta64[D]').astype(np.int)
df_test['time_step'] = (df_test.date-t0).astype('timedelta64[D]').astype(np.int)

df_gdp.columns = ['Finland', 'Norway', 'Sweden']
gdp_dictionary = df_gdp.unstack().to_dict()
df_train['gdp'] = df_train.set_index(['country','year']).index.map(gdp_dictionary.get)
df_test['gdp'] = df_test.set_index(['country','year']).index.map(gdp_dictionary.get)

df_festivities.holiday = 1
df_train = df_train.merge(df_festivities, on=['date', 'country'], how='left')
df_test = df_test.merge(df_festivities, on=['date', 'country'], how='left')
df_train['holiday'] = df_train['holiday'].fillna(0)
df_test['holiday'] = df_test['holiday'].fillna(0)

features = [c for c in df_test.columns if c not in ('row_id', 'date')]
cat_features = ['country', 'store', 'product']

ordinal_encoder = OrdinalEncoder()
df_train[cat_features] = ordinal_encoder.fit_transform(df_train[cat_features])
df_test[cat_features] = ordinal_encoder.fit_transform(df_test[cat_features])


# # Hyperparameter optimization
# 
# We will be using **Optuna** to find optimal hyperparameter values and add regularization to combat overfitting.

# In[4]:


# tss = TimeSeriesSplit(n_splits=4)
# s = 0

# def objective(trial):
#     fold_valid_preds = {}
#     fold_scores = []
#     seed_valid_ids = []

#     for fold, (i_train, i_test) in enumerate(tss.split(df_train)):
#         X_train = df_train.iloc[i_train]
#         y_train = df_train['num_sold'].iloc[i_train]

#         X_valid = df_train.iloc[i_test]
#         y_valid = df_train['num_sold'].iloc[i_test]

#         fold_valid_ids = X_valid.row_id.values.tolist()
#         seed_valid_ids += fold_valid_ids

#         X_train = X_train[features]
#         X_valid = X_valid[features]

#         params = {
#             'depth': trial.suggest_int('depth', 4, 8),
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.03),
#             'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0, 10),
# #             'rsm': trial.suggest_float('rsm', 0.9, 1.0),
#             'random_strength': trial.suggest_float('random_strength', 1, 10),
#             'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 10)
#         }

#         model = CatBoostRegressor(**params,
#                                   iterations=10000,
#                                   bootstrap_type='Bayesian',
#                                   boosting_type='Plain',
#                                   loss_function='MAE',
#                                   eval_metric='SMAPE',
#                                   random_seed=s)

#         model.fit(X_train,
#                   y_train,
#                   early_stopping_rounds=200,
#                   eval_set=[(X_valid, y_valid)],
#                   verbose=0)

#         fold_valid_pred = model.predict(X_valid)
#         fold_valid_preds.update(dict(zip(fold_valid_ids, fold_valid_pred)))

#         fold_score = np.mean(np.abs(fold_valid_pred - y_valid) / ((np.abs(y_valid) + np.abs(fold_valid_pred)) / 2)) * 100
#         fold_scores.append(fold_score)
        
#     seed_score = np.mean(fold_scores)
#     return seed_score

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=1000, timeout=72000)


# # Model training
# 
# As we are dealing with time series, we can't use regular Kfold cross-validation. If we did, we would train our model on future data and predict past data, resulting in data leakage. Instead, we need to make sure predictions are made only on folds that come after the folds used for training. Luckily, scikit-learn offers a time series cross-validator, called **TimeSeriesSplit**.
# 
# Here's the [documentation's](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) description:
# 
# > *In the kth split, it returns first k folds as train set and the (k+1)th fold as test set. Note that unlike standard cross-validation methods, successive training sets are supersets of those that come before them.*
# 
# The pictures below illustrate the difference well.
# 
# ![](https://scikit-learn.org/stable/_images/sphx_glr_plot_cv_indices_006.png)
# 
# ![](https://scikit-learn.org/stable/_images/sphx_glr_plot_cv_indices_013.png)
# 
# In this competition, our submissions will be evaluated on **SMAPE** (Symmetric mean absolute percentage error) between forecasts and actual values. We can calculate this score using the following formula:
# 
# ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/9d7003eba8a7ffe2379cd5c232adf78daa3d1edf)
# 
# We are going to be using CatBoost, which offers SMAPE as an evaluation metric, but it's nice to calculate it anyway.
# 
# Finally, because the training set is so small, we can afford to train our model on more than one random state/seed. I go into a bit more detail as to why this is useful and how much it improves results in [this](https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/298623) post.

# In[5]:


tss = TimeSeriesSplit(n_splits=4)

m = 1 # change this if you want to train different models
seeds = 5 # set the number of seeds you want to average

seed_valid_preds = []
seed_test_preds = []
seed_scores = []

for s in range(seeds):
    fold_valid_preds = {}
    fold_test_preds = []
    fold_scores = []
    seed_valid_ids = []

    for fold, (i_train, i_test) in enumerate(tss.split(df_train)):
        X_train = df_train.iloc[i_train]
        y_train = df_train['num_sold'].iloc[i_train]

        X_test = df_test.copy()

        X_valid = df_train.iloc[i_test]
        y_valid = df_train['num_sold'].iloc[i_test]

        fold_valid_ids = X_valid.row_id.values.tolist()
        seed_valid_ids += fold_valid_ids

        X_train = X_train[features]
        X_valid = X_valid[features]
        
        params = {'depth': 5,
                  'learning_rate': 0.01,
                  'l2_leaf_reg': 5.0,
                  'random_strength': 2.0,
                  'min_data_in_leaf': 2}
                  
        model = CatBoostRegressor(**params,
                                  iterations=20000,
                                  bootstrap_type='Bayesian',
                                  boosting_type='Plain',
                                  loss_function='MAE',
                                  eval_metric='SMAPE',
                                  random_seed=s)

        model.fit(X_train,
                  y_train,
                  early_stopping_rounds=200,
                  eval_set=[(X_valid, y_valid)],
                  verbose=0)

        fold_valid_pred = model.predict(X_valid)
        fold_test_pred = model.predict(X_test)

        fold_valid_preds.update(dict(zip(fold_valid_ids, fold_valid_pred)))
        fold_test_preds.append(fold_test_pred)

        fold_score = np.mean(np.abs(fold_valid_pred - y_valid) / ((np.abs(y_valid) + np.abs(fold_valid_pred)) / 2)) * 100
        fold_scores.append(fold_score)
        print(f'Seed {s} fold {fold} SMAPE: {fold_score}')

    print(f'Seed {s} SMAPE {np.mean(fold_scores)}, std {np.std(fold_scores)}')
    
    seed_valid_pred = np.array(list(fold_valid_preds.values()))
    seed_test_pred = np.mean(np.column_stack(fold_test_preds), axis=1)
    
    seed_valid_preds.append(seed_valid_pred)
    seed_test_preds.append(seed_test_pred)
    
    seed_score = np.mean(fold_scores)
    seed_scores.append(seed_score)
    
print(f'SMAPE of {s+1} seeds: {np.mean(seed_scores)}, std {np.std(seed_scores)}')

# Out-of-fold predictions for later use
valid_preds = pd.DataFrame(list(zip(seed_valid_ids, np.mean(np.column_stack(seed_valid_preds), axis=1))))
valid_preds.columns = ['row_id', f'CB{m}_pred']
valid_preds.to_csv(f'CB{m}_valid_pred.csv', index=False)

# Test predictions for later use
sample_submission.num_sold = np.mean(np.column_stack(seed_test_preds), axis=1)
sample_submission.columns = ['row_id', f'CB{m}_pred']
sample_submission.to_csv(f'CB{m}_test_pred.csv', index=False)

# Submission
sample_submission.num_sold = np.mean(np.column_stack(seed_test_preds), axis=1)
sample_submission.columns = ['row_id', 'num_sold']
sample_submission.to_csv('submission.csv', index=False)

