#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
# <a id="table-of-contents"></a>
# - [1 Introduction](#Introduction)
# - [2 Import](#import)

# <a id="introduction"></a>
# # Introduction

# More feature engineering and model impplementdation from [EDA, preprocessing pipeline and submission](https://www.kaggle.com/batprem/eda-preprocessing-pipeline-and-submission).

# <a id="import"></a>
# # Import
# <a id="modules"></a>
# ## modules

# In[1]:


# From eda-preprocessing-pipeline-and-submission
# OS
import os

# Data format
import datetime

# Tying
from copy import copy

# Data processing
import pandas as pd

# Data virtualisation
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px

# Widgets
import ipywidgets as widgets

# Exporter
from inspect import getsource

# Math and model
import numpy as np
import scipy
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

# Normaliser
from scipy.special import (
    boxcox,
    inv_boxcox
)

# Modules in this notebook
from catboost import Pool, CatBoostRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMRegressor
import tensorflow as tf

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import StackingRegressor
from sklearn.neighbors import KNeighborsRegressor

import xgboost

## Optuna tuner
import optuna
from sklearn.metrics import mean_squared_error

# Submission plot
import plotly.figure_factory as ff


# ## Data

# In[2]:


Train_data = pd.read_csv('../input/tabular-playground-series-jul-2021/train.csv')
Test_data = pd.read_csv('../input/tabular-playground-series-jul-2021/test.csv')


# ## Pipeline
# 
# In this section, the preprocessing pipelines made in [EDA, preprocessing pipeline and submission](https://www.kaggle.com/batprem/eda-preprocessing-pipeline-and-submission?scriptVersionId=67990508) are imported

# In[3]:


pipeline_path = '../input/eda-preprocessing-pipeline-and-submission/pipeline/'
for node in os.listdir(pipeline_path):
    with open(pipeline_path + node, 'r') as f:
        exec(f.read())


# In[4]:


Train_data = preprocess_test(Train_data)
# Train_data['month'] = list(pd.Series(Train_data.index).apply(lambda date: date.month))
# Train_data['week_day*hour'] = Train_data['week_day']*Train_data['hour']
Train_data = sort_columns(Train_data)
# Train_data.drop('hour', axis=1, inplace=True)
Train_data.head(5)


# In[5]:


Test_data = preprocess_test(Test_data)
# Test_data['month'] = list(pd.Series(Test_data.index).apply(lambda date: date.month))
# Test_data['week_day*hour'] = Test_data['week_day']*Test_data['hour']
# Test_data.drop(['hour'], axis=1, inplace=True)
Test_data.head(5)


# ## Visualise Preprocessing
# ### Benzene

# In[6]:


fig = px.parallel_coordinates(
    Train_data,
    color='target_benzene',
    dimensions=[
        'hour',
        'relative_humidity',
        'sensor_3',
        'absolute_humidity',
        'deg_C',
        # 'month'
        # 'week_day',
        'sensor_1',
        'sensor_2',
        'sensor_4',
        'sensor_5',
        # 'week_day*hour',
        'target_benzene',
    ],
    labels={
            'target_benzene': 'Benzene',
            'target_nitrogen_oxides': 'Nitrogen oxides',
            "target_carbon_monoxide": "Carbon monoxide"
    },
    color_continuous_scale=px.colors.diverging.Tealrose,
    color_continuous_midpoint=2)
fig.show()


# ### Nitrogen oxides

# In[7]:


fig = px.parallel_coordinates(
    Train_data,
    color='target_nitrogen_oxides',
    dimensions=[
        'hour',
        'relative_humidity',
        'sensor_3',
        'absolute_humidity',
        'deg_C',
        'sensor_1',
        'sensor_2',
        # 'week_day',
        # 'week_day*hour',
        'target_benzene',
        "target_carbon_monoxide",
        'sensor_4',
        'sensor_5',
        'target_nitrogen_oxides',
    ],
    labels={
            'target_benzene': 'Benzene',
            'target_nitrogen_oxides': 'Nitrogen oxides',
            "target_carbon_monoxide": "Carbon monoxide"
    },
    color_continuous_scale=px.colors.diverging.Tealrose,
    color_continuous_midpoint=2,
    range_color=[
        min(Train_data.target_nitrogen_oxides),
        max(Train_data.target_nitrogen_oxides)
    ]
)
fig.show()


# ### Carbon monoxide

# In[8]:


fig = px.parallel_coordinates(
    Train_data,
    color='target_carbon_monoxide',
    dimensions=[
        'hour',
        'relative_humidity',
        'sensor_3',
        'absolute_humidity',
        'deg_C',
        'sensor_1',
        'sensor_2',
        'sensor_4',
        'sensor_5',
        # 'week_day',
        'target_carbon_monoxide',
    ],
    labels={
            'target_benzene': 'Benzene',
            'target_nitrogen_oxides': 'Nitrogen oxides',
            "target_carbon_monoxide": "Carbon monoxide"
    },
    color_continuous_scale=px.colors.diverging.Tealrose,
    color_continuous_midpoint=2,
    range_color=[
        min(Train_data.target_carbon_monoxide),
        max(Train_data.target_carbon_monoxide)
    ]
)
fig.show()


# # Modeling

# In[9]:


# Classic K fold
num_fold = 5
kf = KFold(n_splits=num_fold, shuffle=True, random_state=1234)
kf.get_n_splits(Train_data)

print(kf)

K_FOLD = []
for train_index, test_index in kf.split(Train_data):
    print("TRAIN:", train_index, "TEST:", test_index)
    K_FOLD.append((train_index, test_index))


# In[10]:


# # Back testing
# num_fold = 5
# kf = KFold(n_splits=num_fold, shuffle=False)
# tscv = TimeSeriesSplit()
# kf.get_n_splits(Train_data)

# print(kf)

# K_FOLD = []
# for train_index, test_index in tscv.split(Train_data):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     K_FOLD.append((train_index, test_index))


# In[11]:


import pickle

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


# In[12]:


def fit_svr(X_Train, y_Train, X_Test, y_Test, verbose=True, **params):
    model = make_pipeline(
        StandardScaler(),
        SVR(**params)
        # SVR(C=1.0, epsilon=0.3)
    )
    model.fit(X_Train, y_Train)
    prediction = model.predict(X_Test)
    rmsle = RMSLE(prediction, y_Test)
    rmsle = RMSLE(model.predict(X_Train), y_Train)
    if verbose: print(f"rmsle Train: {rmsle}")
    rmsle = RMSLE(prediction, y_Test)
    if verbose: print(f"rmsle validation: {rmsle}")
    return model, prediction, y_Test


def objective_decorate(X_Train, y_Train, X_Test, y_Test):
    def objective(trial):
        params = {
            # 'tree_method':'gpu_hist',  # this parameter means using the GPU when training our model to speedup the training process
            'C': trial.suggest_loguniform('C', 1e-3, 10.0),
            'epsilon': trial.suggest_loguniform('epsilon', 1e-3, 10.0),
        }

        train_predictions = np.array([])
        y_trains = np.array([])
        predictions = np.array([])
        y_validations = np.array([])


        # print(f'Columns: {y_col}')
        model, prediction, y_validation = fit_svr(
            X_Train,
            y_Train,
            X_Test,
            y_Test,
            verbose=False,
            **params
        )
        train_prediction = model.predict(X_Train).flatten()
        train_predictions = np.concatenate([train_predictions, train_prediction])
        y_trains = np.concatenate([y_trains, y_Train.values])
        predictions = np.concatenate([predictions, prediction])
        y_validations = np.concatenate([y_validations, y_validation])

        rmsle_train = RMSLE(train_predictions, y_trains)
        rmsle_test = RMSLE(predictions, y_validations)
        overfit_protect = rmsle_test * (rmsle_test/rmsle_train)**(2)
#         print(f"Overall rmsle train: {rmsle_train}")
#         print(f"Overall rmsle test: {rmsle_test}")
#         print(f"Overfit protect: {overfit_protect}")
        return overfit_protect
    
    return objective

# study = optuna.create_study(direction='minimize')
# optuna.logging.disable_default_handler()
# study.optimize(
#     objective_decorate(
#         X_Train=X_Train,
#         y_Train=y_Train[y_col],
#         X_Test=X_Test,
#         y_Test=y_Test[y_col],
#     ),
#     n_trials=50,
# )

# print('Best trial:', study.best_trial.params)



# In[13]:


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error as rmsle
from sklearn.ensemble import RandomForestRegressor


# Metric
def RMSLE(pred, act):
    pred = inv_boxcox(pred, 0.0001)
    act = inv_boxcox(act, 0.0001)
    return (np.mean(
        (np.log(pred + 1) - np.log(act + 1))**2
    )) ** 0.5

# Fit
def fit_catboosts(X_Train, y_Train, X_Test, y_Test):
    param = {'iterations':5}
    model = CatBoostRegressor(
        iterations=50, 
        depth=5, 
        learning_rate=0.1, 
        l2_leaf_reg=0.15, #0.3,
        loss_function='RMSE'
    )
    
    train_pool = Pool(
        X_Train,
        y_Train, 
        cat_features=None
    )
    
    test_pool = Pool(
        X_Test, 
        cat_features=None
    )
    model.fit(X_Train, y_Train, verbose=0)
    prediction = model.predict(test_pool)
    
    rmsle = RMSLE(model.predict(X_Train), y_Train)
    print(f"rmsle Train: {rmsle}")
    rmsle = RMSLE(prediction, y_Test)
    print(f"rmsle validation: {rmsle}")
    return model, prediction, y_Test


def fit_lgbm(X_Train, y_Train, X_Test, y_Test):
    model = LGBMRegressor(
        reg_lambda=2,
        reg_alpha=6,
        random_state=1234
    )
    model.fit(X_Train, y_Train)
    prediction = model.predict(X_Test)
    rmsle = RMSLE(prediction, y_Test)
    rmsle = RMSLE(model.predict(X_Train), y_Train)
    print(f"rmsle Train: {rmsle}")
    rmsle = RMSLE(prediction, y_Test)
    print(f"rmsle validation: {rmsle}")
    return model, prediction, y_Test


def fit_svr(X_Train, y_Train, X_Test, y_Test, verbose=True, **params):
    if not params:
        params = {'C': 1.5395074336816164, 'epsilon': 0.19287719821166877}
    model = make_pipeline(StandardScaler(), SVR(
        # C=1.0, epsilon=0.3
        **params
    ))
    model.fit(X_Train, y_Train)
    prediction = model.predict(X_Test)
    rmsle = RMSLE(prediction, y_Test)
    rmsle = RMSLE(model.predict(X_Train), y_Train)
    if verbose: print(f"rmsle Train: {rmsle}")
    rmsle = RMSLE(prediction, y_Test)
    if verbose: print(f"rmsle validation: {rmsle}")
    return model, prediction, y_Test


def fit_linear(X_Train, y_Train, X_Test, y_Test):
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    
    model.fit(X_Train, y_Train)
    prediction = model.predict(X_Test)
    
    feature_importance = dict(
        zip(X_Train.columns, model[-1].coef_)
    )
    feature_importance = {
        k: v
        for k, v
        in sorted(
            feature_importance.items(),
            key=lambda item: item[1], reverse=True
        )
    }
    print("Feature important")
    print(feature_importance)
    rmsle = RMSLE(prediction, y_Test)
    rmsle = RMSLE(model.predict(X_Train), y_Train)
    print(f"rmsle Train: {rmsle}")
    rmsle = RMSLE(prediction, y_Test)
    print(f"rmsle validation: {rmsle}")
    return model, prediction, y_Test



def fit_randomForest(X_Train, y_Train, X_Test, y_Test):
    model = RandomForestRegressor(max_features=3)
    model.fit(X_Train, y_Train)
    prediction = model.predict(X_Test)
    
    feature_importance = dict(
        zip(X_Train.columns, model.feature_importances_)
    )
    feature_importance = {
        k: v
        for k, v
        in sorted(
            feature_importance.items(),
            key=lambda item: item[1], reverse=True
        )
    }
    print("Feature important")
    print(feature_importance)
    rmsle = RMSLE(prediction, y_Test)
    rmsle = RMSLE(model.predict(X_Train), y_Train)
    print(f"rmsle Train: {rmsle}")
    rmsle = RMSLE(prediction, y_Test)
    print(f"rmsle validation: {rmsle}")
    return model, prediction, y_Test


# KNeighborsRegressor(10, weights='uniform')
def fit_knn(X_Train, y_Train, X_Test, y_Test):
    # model = make_pipeline(StandardScaler(), LinearRegression())
    model = make_pipeline(StandardScaler(), KNeighborsRegressor(10, weights='uniform'))
    
    model.fit(X_Train, y_Train)
    prediction = model.predict(X_Test)
    rmsle = RMSLE(prediction, y_Test)
    rmsle = RMSLE(model.predict(X_Train), y_Train)
    print(f"rmsle Train: {rmsle}")
    rmsle = RMSLE(prediction, y_Test)
    print(f"rmsle validation: {rmsle}")
    return model, prediction, y_Test


# def fit_svrWithOptunaTune(X_Train, y_Train, X_Test, y_Test):
#     study = optuna.create_study(direction='minimize')
#     optuna.logging.disable_default_handler()
#     study.optimize(
#         objective_decorate(
#             X_Train=X_Train,
#             y_Train=y_Train,
#             X_Test=X_Test,
#             y_Test=y_Test,
#         ),
#         n_trials=50,
#     )
    
    
#     model = make_pipeline(
#         StandardScaler(),
#         SVR(
#             **study.best_params
#         )
#     )
#     print(model)
    
#     model.fit(X_Train, y_Train)
#     prediction = model.predict(X_Test)
#     rmsle = RMSLE(prediction, y_Test)
#     rmsle = RMSLE(model.predict(X_Train), y_Train)
#     print(f"rmsle Train: {rmsle}")
#     rmsle = RMSLE(prediction, y_Test)
#     print(f"rmsle validation: {rmsle}")
#     return model, prediction, y_Test



def fit_xgboost(X_Train, y_Train, X_Test, y_Test, **params):
    if not params:
        params = {'reg_lambda': 8.806514467534535, 'reg_alpha': 5.10815789088487}
    model = xgboost.XGBRegressor(
        **params
#         reg_lambda=2,
#         reg_alpha=5
    )
    model.fit(X_Train, y_Train)
    prediction = model.predict(X_Test)
    rmsle = RMSLE(prediction, y_Test)
    rmsle = RMSLE(model.predict(X_Train), y_Train)
    print(f"rmsle Train: {rmsle}")
    rmsle = RMSLE(prediction, y_Test)
    print(f"rmsle validation: {rmsle}")
    return model, prediction, y_Test


# def fit_stacking(X_Train, y_Train, X_Test, y_Test):
#     estimators = [
#         ('svr_rbf', make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))),
#         (
#             'lgbm', LGBMRegressor(
#                 reg_lambda=2,
#                 reg_alpha=6,
#                 random_state=1234
#             )
#         )
#     ]
#     model = StackingRegressor(
#         estimators=estimators,
#         final_estimator= make_pipeline(StandardScaler(), Ridge(alpha=0.1))
#     )
#     model.fit(X_Train, y_Train)
#     prediction = model.predict(X_Test)
#     rmsle = RMSLE(prediction, y_Test)
#     rmsle = RMSLE(model.predict(X_Train), y_Train)
#     print(f"rmsle Train: {rmsle}")
#     rmsle = RMSLE(prediction, y_Test)
#     print(f"rmsle validation: {rmsle}")
#     return model, prediction, y_Test



def get_fit_function(key: str):
    return key.startswith('fit_')

models = {}
cv_report = pd.DataFrame()

for k, (Train, Test) in enumerate(K_FOLD):
    print(f"K: {k}")
    
    # Ge
    y_columns = list(
        Train_data.columns[
            Train_data.columns.str.startswith('target')
        ]
    )
    X_Train = Train_data.iloc[Train].drop(y_columns, axis=1)
    y_Train = Train_data.iloc[Train][y_columns]

    X_Test = Train_data.iloc[Test].drop(y_columns, axis=1)
    y_Test = Train_data.iloc[Test][y_columns]
    
    
    
    for fit in filter(get_fit_function,dict(globals())):
        _, model_name = fit.split('_')
        train_predictions = np.array([])
        y_trains = np.array([])
        predictions = np.array([])
        y_validations = np.array([])
        model = {}
        for y_col in y_columns:
            print(y_col)
            print(f'Model {model_name}')
            model[y_col], prediction, y_validation = globals()[fit](
                X_Train, y_Train[y_col],
                X_Test, y_Test[y_col]
            )
            train_prediction = model[y_col].predict(X_Train)
            train_predictions = np.concatenate([train_predictions, train_prediction])
            y_trains = np.concatenate([y_trains, y_Train[y_col].values])
            predictions = np.concatenate([predictions, prediction])
            y_validations = np.concatenate([y_validations, y_validation])
            
            save_model(model[y_col], f"Fold_{k}_Model_{model_name}_Col_{y_col}.pickle")
        
        rmsle_train = RMSLE(train_predictions, y_trains)
        rmsle_test = RMSLE(predictions, y_validations)
        
        cv_report.loc[k, f'{model_name}_train'] = rmsle_train
        cv_report.loc[k, f'{model_name}_test'] = rmsle_test
        if model_name in models:
            models[model_name].append(model)
        else:
            models[model_name] = [model]
        print('-' * 36)
        
    
#     catboost_models.append(catboost)
#     svr_models.append(svr)
#     linear_models.append(linear)
#     lgbm_models.append(lgbm)


# In[14]:


feature_importance = dict(zip(X_Train.columns, models['linear'][0]['target_nitrogen_oxides'][-1].coef_))


# In[ ]:





# In[15]:


from pprint import pprint

for name, model in models.items():
    print(f"{name}:")
    pprint(model[0])


# In[16]:


cv_report


# In[17]:


cv_report.mean()


# # Make submission

# In[18]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go


fig = go.Figure()


mean_submission = pd.read_csv(
        '../input/tabular-playground-series-jul-2021/sample_submission.csv'
    )
mean_submission = mean_submission.set_index('date_time')
mean_submission['target_carbon_monoxide'] = 0
mean_submission['target_benzene'] = 0
mean_submission['target_nitrogen_oxides'] = 0

for sub_count, fit in enumerate(filter(get_fit_function,dict(globals()))):
    _, model_name = fit.split('_')

    submission = pd.read_csv(
        '../input/tabular-playground-series-jul-2021/sample_submission.csv'
    )
    submission = submission.set_index('date_time')
    submission['target_carbon_monoxide'] = 0
    submission['target_benzene'] = 0
    submission['target_nitrogen_oxides'] = 0
    
    for k in range(num_fold):
        for y_col in y_columns:
            submission[y_col] += models[model_name][k][y_col].predict(
                Test_data
            )/num_fold


    submission = inv_boxcox(submission, 0.0001)
    submission.to_csv(f'{model_name}_submission.csv')
    for y_col in y_columns:
        fig.add_trace(
            go.Scatter(
                x=submission.index,
                y=submission[y_col],
                mode='lines',
                name=f'{model_name}_{y_col}')
        ) 
    mean_submission += submission

    
mean_submission = mean_submission / (sub_count + 1)
mean_submission.to_csv('mean_submission.csv')
for y_col in y_columns:
    fig.add_trace(
        go.Scatter(
            x=mean_submission.index,
            y=mean_submission[y_col],
            mode='lines',
            name=f'mean_submission_{y_col}')
    )
fig.show()


# In[19]:


mean_submission

