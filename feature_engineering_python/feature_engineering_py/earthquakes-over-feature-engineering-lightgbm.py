#!/usr/bin/env python
# coding: utf-8

# Greatly inspired by: [Earthquakes FE. More features and samples](https://www.kaggle.com/artgor/earthquakes-fe-more-features-and-samples).

# In[1]:


import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 15

import eli5
from eli5.sklearn import PermutationImportance

import lightgbm as lgb
import xgboost as xgb
import time
import datetime
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy import stats
from sklearn.kernel_ridge import KernelRidge


# In test set, the data comes in segments and not like a big chunk in training dataset. I want to know how long the segments are in test segments, we are going to chunk the training data into chunk sizes which are equal to the testing set chunk sizes.

# In[2]:


test_seg = pd.read_csv('../input/test/seg_004cd2.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})


# In[3]:


len(test_seg)


# I chunked the training data with 150,000 rows per chunk. I also tried what would it be like when data is supersample, that is, taking overlapping rows, where each starting point differs by 10,000 rows. It turns out that the model overfits quite a bit for that kind of scheme. Maybe, I can reduce the overlap and also tune the parameters for thraining method to reduce overfitting.

# But right now, just a simple chunking with 150,000 rows per chunk.

# In[4]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})\n")


# In[5]:


chunk_row = 150_000
rows = 150_000
segments = int(np.floor(train.shape[0] / rows))


# In[6]:


segments


# 4,194 segments are present so there will be 4,194 rows in our training dataset. The main concern I have is that the model will overfit with such a small dataset.

# Now, let's create some placeholder dataframes for features (X_tr) and labels (y_tr)

# In[7]:


X_tr = pd.DataFrame(index=range(segments), dtype=np.float64)

y_tr = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])


# I increased the number of features during feature engineering process. I added Fast Fourier Transform to extract some features from the data. I think it makes sense since the acoustic emission might probably be broken down into a bunch of sine waves.

# In[8]:


def create_features(X, segment_num, chunk):
    def add_trend_feature(arr, abs_values=False):
        idx = np.array(range(len(arr)))
        if abs_values:
            arr = np.abs(arr)
        lr = LinearRegression()
        lr.fit(idx.reshape(-1, 1), arr)
        return lr.coef_[0]
    
    percentiles = [10, 50, 75, 90, 99]
    moving_avg_window = [1500, 3000, 6000, 9000, 12000]
    first_segments = [10000, 50000]
    last_segments = [10000, 50000]
    rolling_percentiles = [1, 5, 50, 95, 99]
    
    x = pd.Series(chunk['acoustic_data'].values)

    X.loc[segment_num, 'mean'] = x.mean()
    X.loc[segment_num, 'std'] = x.std()
    X.loc[segment_num, 'var'] = np.var(x)
    X.loc[segment_num, 'max'] = x.max()
    X.loc[segment_num, 'min'] = x.min()
    
    z = np.fft.fft(x)
    realFFT = np.real(z)
    imagFFT = np.imag(z)
    X.loc[segment_num, 'Real_mean'] = realFFT.mean()
    X.loc[segment_num, 'Real_std'] = realFFT.std()
    X.loc[segment_num, 'Imag_mean'] = imagFFT.mean()
    X.loc[segment_num, 'Imag_std'] = imagFFT.std()
    
    for f_seg in first_segments:
        X.loc[segment_num, 'std_first_{}'.format(f_seg)] = x[:f_seg].std()
        X.loc[segment_num, 'mean_first_{}'.format(f_seg)] = x[:f_seg].mean()
        X.loc[segment_num, 'Rstd_last_{}'.format(f_seg)] = realFFT[:f_seg].std()
        X.loc[segment_num, 'Rmean_last_{}'.format(f_seg)] = realFFT[:f_seg].mean()

    for l_seg in last_segments:
        X.loc[segment_num, 'std_last_{}'.format(l_seg)] = x[-l_seg:].std()
        X.loc[segment_num, 'mean_last_{}'.format(l_seg)] = x[-l_seg:].mean()
        X.loc[segment_num, 'Rstd_last_{}'.format(l_seg)] = realFFT[-l_seg:].std()
        X.loc[segment_num, 'Rmean_last_{}'.format(l_seg)] = realFFT[-l_seg:].mean()
        
    for percent in percentiles:
        X.loc[segment_num, 'percentile_{}'.format(percent)] = np.percentile(x, percent)
    
    X.loc[segment_num, 'mad'] = x.mad()
    X.loc[segment_num, 'kurt'] = x.kurtosis()
    X.loc[segment_num, 'skew'] = x.skew()
    X.loc[segment_num, 'med'] = x.median()
    
    X.loc[segment_num, 'Hilbert_mean'] = np.abs(hilbert(x)).mean()
    X.loc[segment_num, 'Hann_window_mean'] = (convolve(x, hann(150), mode='same') / sum(hann(150))).mean()
    
    X.loc[segment_num, 'trend'] = add_trend_feature(x)
    X.loc[segment_num, 'abs_trend'] = add_trend_feature(x, abs_values=True)
    
    for mv_avg_window in moving_avg_window:
        X.loc[segment_num, 'Moving_average_{}_mean'.format(mv_avg_window)] = x.rolling(window=mv_avg_window).mean().mean(skipna=True)
    
    for windows in [10, 100, 1000]:
        x_roll_std = x.rolling(windows).std().dropna().values
        x_roll_mean = x.rolling(windows).mean().dropna().values

        X.loc[segment_num, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
        X.loc[segment_num, 'std_roll_std_' + str(windows)] = x_roll_std.std()
        X.loc[segment_num, 'max_roll_std_' + str(windows)] = x_roll_std.max()
        X.loc[segment_num, 'min_roll_std_' + str(windows)] = x_roll_std.min()
        
        X.loc[segment_num, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
        X.loc[segment_num, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
        X.loc[segment_num, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
        X.loc[segment_num, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()

        for roll_perc in rolling_percentiles:
            X.loc[segment_num, 'p_{}_roll_std_{}'.format(roll_perc, windows)] = np.percentile(x_roll_std, roll_perc)
            X.loc[segment_num, 'p_{}_roll_mean_{}'.format(roll_perc, windows)] = np.percentile(x_roll_mean, roll_perc)

        X.loc[segment_num, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
        X.loc[segment_num, 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
        X.loc[segment_num, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()
        X.loc[segment_num, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
        X.loc[segment_num, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
        X.loc[segment_num, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()


# In[9]:


for segment in tqdm_notebook(range(segments)):
    seg = train.iloc[segment*rows:segment*rows+chunk_row]
    y = seg['time_to_failure'].values[-1]
    y_tr.loc[segment, 'time_to_failure'] = y
    create_features(X_tr, segment, seg)


# In[10]:


print(f'{X_tr.shape[0]} samples in new train data and {X_tr.shape[1]} columns.')


# Checking how well the added features correlate to labels

# In[11]:


np.abs(X_tr.corrwith(y_tr['time_to_failure'])).sort_values(ascending=False).head(10)


# For all the missing values, create a dictionary with mean values which will be used both for replacing the missing values in training dataset as well as testing dataset.

# In[12]:


means_dict = {}
for col in X_tr.columns:
    if X_tr[col].isnull().any():
        mean_value = X_tr.loc[X_tr[col] != -np.inf, col].mean()
        X_tr.loc[X_tr[col] == -np.inf, col] = mean_value
        X_tr[col] = X_tr[col].fillna(mean_value)
        means_dict[col] = mean_value


# Standardize features by removing the mean and scaling to unit variance

# In[13]:


scaler = StandardScaler()
scaler.fit(X_tr)
X_train_scaled = pd.DataFrame(scaler.transform(X_tr), columns=X_tr.columns)


# ### Reading test data

# Perform the same kind of feature engineering to the test data

# In[14]:


submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame(columns=X_tr.columns, dtype=np.float64, index=submission.index)

for i, seg_id in enumerate(tqdm_notebook(X_test.index)):
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    create_features(X_test, seg_id, seg)
    
for col in X_test.columns:
    if X_test[col].isnull().any():
        X_test.loc[X_test[col] == -np.inf, col] = means_dict[col]
        X_test[col] = X_test[col].fillna(means_dict[col])
        
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# ## Building models

# In[15]:


n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)


# In[16]:


def train_model(X=X_train_scaled, X_test=X_test_scaled, y=y_tr, params=None, folds=folds, model_type='lgb', plot_feature_importance=False, model=None):

    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators = 50000, n_jobs = -1)
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
                    verbose=10000, early_stopping_rounds=200)
            
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
            
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
        
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            
            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = mean_absolute_error(y_valid, y_pred_valid)
            print(f'Fold {fold_n}. MAE: {score:.4f}.')
            print('')
            
            y_pred = model.predict(X_test).reshape(-1,)
        
        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000,  eval_metric='MAE', task_type='GPU', **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)
        
        oof[valid_index] = y_pred_valid.reshape(-1,)
        scores.append(mean_absolute_error(y_valid, y_pred_valid))

        prediction += y_pred    
        
        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_fold
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
        
            return oof, prediction, feature_importance, model
        return oof, prediction, model
    
    else:
        return oof, prediction, model


# I just want to train with one machine learning method (light gradient boosting) rather than using xgboost, catboot and SVN. I didn't see the results any getting better with multiple models.

# In[17]:


params = {'num_leaves': 128,
          'min_data_in_leaf': 79,
          'objective': 'huber',
          'max_depth': -1,
          'learning_rate': 0.01,
          "boosting": "gbdt",
          "bagging_freq": 5,
          "bagging_fraction": 0.8126672064208567,
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1,
          'reg_alpha': 0.1302650970728192,
          'reg_lambda': 0.3603427518866501
         }
oof_lgb, prediction_lgb, feature_importance, model = train_model(params=params,
                                                                 model_type='lgb',
                                                                 plot_feature_importance=True)


# In[18]:





# This is just for the sake of testing. I actually am not planning to use XGB outputs for submission.

# xgb_params = {'eta': 0.03,
#               'max_depth': 10,
#               'subsample': 0.9,
#               'objective': 'reg:linear',
#               'eval_metric': 'mae',
#               'silent': True,
#               'nthread': 4}
# oof_xgb, prediction_xgb = train_model(X=X_train_scaled, X_test=X_test_scaled, params=xgb_params, model_type='xgb')

# xgb_preds = oof_xgb
# lgb_preds = oof_lgb
# 
# fig, ax1 = plt.subplots(figsize=(16, 8))
# plt.title("LGB Vs XGB prediction values on test data")
# plt.plot(xgb_preds, color='red', alpha=0.9)
# ax1.set_ylabel('time to failure', color='red')
# plt.legend(['xgb'])
# ax2 = ax1.twinx()
# plt.plot(lgb_preds, color='blue', alpha=0.5)
# ax2.set_ylabel('time to failure', color='blue')
# plt.legend(['lgb'], loc=(0.875, 0.9))
# plt.grid(False)

# In[18]:


prediction = oof_lgb
y_values = y_tr

fig, ax1 = plt.subplots(figsize=(16, 8))
plt.title("Predictions Vs Actual time to failure (Training data)")
plt.plot(prediction, color='aqua')
ax1.set_ylabel('time to failure', color='b')
plt.legend(['predictions'])
ax2 = ax1.twinx()
plt.plot(y_values, color='g')
ax2.set_ylabel('actual value', color='g')
plt.legend(['actual value'], loc=(0.875, 0.9))
plt.grid(False)


# The predictions are too cyclical and thus missing out to capture the different time_to_failure after the quakes have occured. The predicions pretty much think that the time will be rest the same way (thus, near perfect cycle). I think this is the next step I have to work on without overfitting.

# In[19]:


feature_importance_values = np.abs(X_test_scaled.corrwith(pd.DataFrame(prediction_lgb)[0]))


# In[20]:


feature_importance_values.sort_values(ascending=False).head(10)


# Rolling windows are the most important features! I am figuring whether I should reduce training my model with just the important features thresholded by certain value. For, example, only the features where the feature importance is above 0.6

# In[21]:


feature_importance_values[feature_importance_values > 0.6].sort_values(ascending=False)


# # Investigating Feature Importance and parameter fine tuning

# What would happen if we only train the features with feature importance greater than 0.5?

# selected_cols = feature_importance_values[feature_importance_values > 0].index
# reduced_X_tr = X_tr[selected_cols]
# reduced_X_test = X_test[selected_cols]
# 
# scaler = StandardScaler()
# scaler.fit(reduced_X_tr)
# reduced_X_train_scaled = pd.DataFrame(scaler.transform(reduced_X_tr), columns=reduced_X_tr.columns)
# reduced_X_test_scaled = pd.DataFrame(scaler.transform(reduced_X_test), columns=reduced_X_test.columns)

# I have heavily fine-tuned num_leaves in tandem with max_depth. Reduced learning rate. Increased lambda_l1 (L1 regularization).

# n_fold = 5
# folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
# 
# params = {'num_leaves': 100,
#           'min_data_in_leaf': 80,
#           'objective': 'huber',
#           'max_depth': 14,
#           'learning_rate': 0.008,
#           "boosting": "dart",
#           "bagging_freq": 2,
#           "bagging_fraction": 0.6,
#           "feature_fraction": 0.75,
#           "bagging_seed": 11,
#           "metric": 'mae',
#           "verbosity": -1,
#           'lambda_l1': 0.18,
#           'lambda_l2': 0.25
#          }
# reduced_oof_lgb, reduced_prediction_lgb, reduced_feature_importance = train_model(X=reduced_X_train_scaled,
#                                                                                   X_test=reduced_X_test_scaled,
#                                                                                   y=y_tr,
#                                                                                   params=params,
#                                                                                   folds=folds,
#                                                                                   model_type='lgb',
#                                                                                   plot_feature_importance=True)

# Plots of time_to_failure values (training set) with rest to important features

# plt.figure(figsize=(44, 24))
# cols = list(feature_importance_values.sort_values(ascending=False).head(20).index)
# for i, col in enumerate(cols):
#     plt.subplot(5, 4, i + 1)
#     plt.plot(X_tr[col], color='blue')
#     plt.title("{} - {:.3f}".format(col, feature_importance_values[col]))
#     ax1.set_ylabel(col, color='b')
# 
#     ax2 = ax1.twinx()
#     plt.plot(y_tr, color='g')
#     ax2.set_ylabel('time_to_failure', color='g')
#     plt.legend([col, 'time_to_failure'], loc=(0.875, 0.9))
#     plt.grid(False)

# Right now I am just going to take the LGB's output predictions. I will think about what to do with XGB outputs. I don't want to average the two results either.

# In[22]:


submission['time_to_failure'] = prediction_lgb


# In[23]:


submission.to_csv('submission_rolling_v6.csv')


# Sometimes, I just play around with stuff and don't want to commit the whole thing just to download the submission file. Here is a great solution to get around that. Feature engineering and hyper parameter tuning takes a long time so I would like to have a download link of submission file when I have played around.

# from IPython.display import HTML
# 
# def create_download_link(title = "Download CSV file", filename = "data.csv"):  
#     html = '<a href={filename}>{title}</a>'
#     html = html.format(title=title,filename=filename)
#     return HTML(html)
# 
# # create a link to download the dataframe
# create_download_link(filename='submission_rolling_v5.csv')
# 
