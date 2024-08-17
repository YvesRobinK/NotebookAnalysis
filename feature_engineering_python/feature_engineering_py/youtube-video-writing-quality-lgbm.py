#!/usr/bin/env python
# coding: utf-8

# ## 🔴 Youtube Video 🔴
# 
# ## Video Link : https://youtu.be/DqvXJRb0q9c?si=dJeEg5cP7q9qcQDi

# ## Imports

# In[1]:


import gc
import os
import itertools
import pickle
import re
import time

import warnings
warnings.filterwarnings('ignore')

from random import choice, choices
from functools import reduce
from tqdm import tqdm
from itertools import cycle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
get_ipython().run_line_magic('matplotlib', 'inline')

from collections import Counter
from functools import reduce
from tqdm import tqdm
from itertools import cycle
from scipy import stats
from scipy.stats import skew, kurtosis
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import ensemble
from sklearn import decomposition
from sklearn import tree

import lightgbm as lgb
import xgboost as xgb

import optuna

pd.set_option("display.max_columns", None)

plt.style.use("ggplot")
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])


# In[2]:


tqdm.pandas()

sns.set_style("whitegrid")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
warnings.simplefilter('ignore')

import random
random.seed(42)


# ## Load Data

# In[3]:


INPUT_DIR = '/kaggle/input/linking-writing-processes-to-writing-quality'

train_logs = pd.read_csv(f'{INPUT_DIR}/train_logs.csv')
train_scores = pd.read_csv(f'{INPUT_DIR}/train_scores.csv')
test_logs = pd.read_csv(f'{INPUT_DIR}/test_logs.csv')

ss_df = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')


# In[4]:


train_logs.shape, train_scores.shape, test_logs.shape, ss_df.shape


# In[5]:


train_logs.head()


# In[6]:


train_scores.head()


# In[7]:


train_scores.describe()


# ## EDA

# In[8]:


plt.figure(figsize=(15, 5))
train_scores['score'].hist()
plt.show()


# In[9]:


train_scores['score'].value_counts()


# In[10]:


event_stats = train_logs.groupby("id")['event_id'].count()

fig, ax = plt.subplots(1,2, figsize=(12,5))
ax[0].set_title('Distribution of events')
ax[0].set_xlabel('Number of events in an essay')
sns.histplot(event_stats, bins=100, ax=ax[0])
ax[1].set_title('Boxplot of events')
sns.boxplot(event_stats, ax=ax[1])
plt.show()


# In[11]:


event_stats.describe()


# In[12]:


stats = train_logs.groupby("id")["event_id"].max().reset_index()
stats_score = stats.merge(train_scores, on='id')

catplot = sns.catplot(data=stats_score, x="score", y="event_id", kind="bar", height=5, aspect=8/5)
plt.xlabel('Score', fontsize=12)
plt.ylabel('Number of events', fontsize=12)
plt.show()


# In[13]:


tmp = train_logs["up_time"] - train_logs["down_time"]
results = (tmp == train_logs["action_time"])
results.value_counts()


# In[14]:


train_logs.head()


# In[15]:


train_logs_scores_df = train_logs.merge(train_scores, on='id', how='left')
train_logs_scores_df.head()


# In[16]:


train_logs_scores_df.columns


# In[17]:


train_logs_agg_df = train_logs_scores_df.groupby("id")[['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count', 'score']].mean().reset_index()


# In[18]:


train_logs_agg_df


# In[19]:


def plot_dist_box(data, target):
    
    color = choice(color_pal)
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))

    # Plot the distribution plot on the first subplot
    sns.histplot(data, ax=axes[0], color=color)
    axes[0].set_title('Distribution Plot')

    # Plot the box plot on the second subplot
    sns.boxplot(data, ax=axes[1], color=color)
    axes[1].set_title('Box Plot')
    
    # Plot the box plot on the second subplot
    sns.ecdfplot(data, ax=axes[2], color=color)
    axes[2].set_title('CDF Plot')
    
    # Plot the box plot on the second subplot
    sns.scatterplot(x=data, y=target, ax=axes[3], color=color)
    axes[3].set_title('Scatter Plot')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()


# In[20]:


# num_cols = ['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']
# for col in num_cols:
#     plot_dist_box(train_logs_scores_df[col], train_logs_scores_df['score'])


# In[21]:


num_cols = ['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']
for col in num_cols:
    plot_dist_box(train_logs_agg_df[col], train_logs_agg_df['score'])


# In[22]:


# def plot_cat_feature_distributions(data, feature_names, target):
#     import math
#     num_features = len(feature_names)
#     num_columns = 2
#     num_rows = math.ceil(num_features / num_columns)
    
#     fig, axes = plt.subplots(num_rows, num_columns, figsize=(20, 5*num_rows))
    
#     for i, feature in enumerate(feature_names):
#         row_idx = i // num_columns
#         col_idx = i % num_columns
#         ax = axes[row_idx, col_idx]
        
#         # sns.countplot(data=data, x=feature, ax=ax)
#         sns.boxplot(data=data, x=feature, y=target, ax=ax)
        
#         ax.set_title(f'Distribution of {feature}')
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    
#     # Remove empty subplots if there are fewer features than expected
#     for i in range(num_features, num_rows * num_columns):
#         fig.delaxes(axes.flatten()[i])
    
#     plt.tight_layout()
#     plt.show()
    
# cat_cols = ['activity', 'down_event', 'up_event', 'text_change']
# plot_cat_feature_distributions(data=train_logs_scores_df, feature_names=cat_cols, target='score')


# In[23]:


train_logs['activity'].value_counts()


# In[24]:


train_logs['down_event'].value_counts()


# In[25]:


train_logs['up_event'].value_counts()


# In[26]:


train_logs['text_change'].value_counts()


# ## Feature Engineering

# In[27]:


class Preprocessor:
    
    def __init__(self, seed):
        self.seed = seed
        
        self.activities = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste']
        self.events = ['q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft', '.', ',', 
              'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', "'", 'Delete', 'Unidentified']
        self.text_changes = ['q', ' ', 'NoChange', '.', ',', '\n', "'", '"', '-', '?', ';', '=', '/', '\\', ':']
        self.punctuations = ['"', '.', ',', "'", '-', ';', ':', '?', '!', '<', '>', '/',
                        '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+']
        self.gaps = [1, 2, 3, 5, 10, 20, 50, 100]
#         self.gaps = [1, 2]
    
    def activity_counts(self, df):
        tmp_df = df.groupby('id').agg({'activity': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['activity'].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.activities:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'activity_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols
        return ret


    def event_counts(self, df, colname):
        tmp_df = df.groupby('id').agg({colname: list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df[colname].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.events:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'{colname}_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols
        return ret


    def text_change_counts(self, df):
        tmp_df = df.groupby('id').agg({'text_change': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['text_change'].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.text_changes:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'text_change_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols
        return ret

    def match_punctuations(self, df):
        tmp_df = df.groupby('id').agg({'down_event': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['down_event'].values):
            cnt = 0
            items = list(Counter(li).items())
            for item in items:
                k, v = item[0], item[1]
                if k in self.punctuations:
                    cnt += v
            ret.append(cnt)
        ret = pd.DataFrame({'punct_cnt': ret})
        return ret


    def get_input_words(self, df):
        tmp_df = df[(~df['text_change'].str.contains('=>'))&(df['text_change'] != 'NoChange')].reset_index(drop=True)
        tmp_df = tmp_df.groupby('id').agg({'text_change': list}).reset_index()
        tmp_df['text_change'] = tmp_df['text_change'].apply(lambda x: ''.join(x))
        tmp_df['text_change'] = tmp_df['text_change'].apply(lambda x: re.findall(r'q+', x))
        tmp_df['input_word_count'] = tmp_df['text_change'].apply(len)
        tmp_df['input_word_length_mean'] = tmp_df['text_change'].apply(lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_max'] = tmp_df['text_change'].apply(lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_std'] = tmp_df['text_change'].apply(lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df.drop(['text_change'], axis=1, inplace=True)
        return tmp_df
    
    def make_feats(self, df):
        
        print("Starting to engineer features")
        
        # initialize features dataframe
        feats = pd.DataFrame({'id': df['id'].unique().tolist()})
        
        # get shifted features
        # time shift
        print("Engineering time data")
        for gap in self.gaps:
            print(f"> for gap {gap}")
            df[f'up_time_shift{gap}'] = df.groupby('id')['up_time'].shift(gap)
            df[f'action_time_gap{gap}'] = df['down_time'] - df[f'up_time_shift{gap}']
        df.drop(columns=[f'up_time_shift{gap}' for gap in self.gaps], inplace=True)

        # cursor position shift
        print("Engineering cursor position data")
        for gap in self.gaps:
            print(f"> for gap {gap}")
            df[f'cursor_position_shift{gap}'] = df.groupby('id')['cursor_position'].shift(gap)
            df[f'cursor_position_change{gap}'] = df['cursor_position'] - df[f'cursor_position_shift{gap}']
            df[f'cursor_position_abs_change{gap}'] = np.abs(df[f'cursor_position_change{gap}'])
        df.drop(columns=[f'cursor_position_shift{gap}' for gap in self.gaps], inplace=True)

        # word count shift
        print("Engineering word count data")
        for gap in self.gaps:
            print(f"> for gap {gap}")
            df[f'word_count_shift{gap}'] = df.groupby('id')['word_count'].shift(gap)
            df[f'word_count_change{gap}'] = df['word_count'] - df[f'word_count_shift{gap}']
            df[f'word_count_abs_change{gap}'] = np.abs(df[f'word_count_change{gap}'])
        df.drop(columns=[f'word_count_shift{gap}' for gap in self.gaps], inplace=True)
        
        # get aggregate statistical features
        print("Engineering statistical summaries for features")
        # [(feature name, [ stat summaries to add ])]
        feats_stat = [
            ('event_id', ['max']),
            ('up_time', ['max']),
            ('action_time', ['sum', 'max', 'mean', 'std']),
            ('activity', ['nunique']),
            ('down_event', ['nunique']),
            ('up_event', ['nunique']),
            ('text_change', ['nunique']),
            ('cursor_position', ['nunique', 'max', 'mean']),
            ('word_count', ['nunique', 'max', 'mean'])]
        for gap in self.gaps:
            feats_stat.extend([
                (f'action_time_gap{gap}', ['max', 'min', 'mean', 'std', 'sum', skew, kurtosis]),
                (f'cursor_position_change{gap}', ['max', 'mean', 'std', 'sum', skew, kurtosis]),
                (f'word_count_change{gap}', ['max', 'mean', 'std', 'sum', skew, kurtosis])
            ])
        
        pbar = tqdm(feats_stat)
        for item in pbar:
            colname, methods = item[0], item[1]
            for method in methods:
                pbar.set_postfix()
                if isinstance(method, str):
                    method_name = method
                else:
                    method_name = method.__name__
                    
                pbar.set_postfix(column=colname, method=method_name)
                tmp_df = df.groupby(['id']).agg({colname: method}).reset_index().rename(columns={colname: f'{colname}_{method_name}'})
                feats = feats.merge(tmp_df, on='id', how='left')

        # counts
        print("Engineering activity counts data")
        tmp_df = self.activity_counts(df)
        feats = pd.concat([feats, tmp_df], axis=1)
        
        print("Engineering event counts data")
        tmp_df = self.event_counts(df, 'down_event')
        feats = pd.concat([feats, tmp_df], axis=1)
        tmp_df = self.event_counts(df, 'up_event')
        feats = pd.concat([feats, tmp_df], axis=1)
        
        print("Engineering text change counts data")
        tmp_df = self.text_change_counts(df)
        feats = pd.concat([feats, tmp_df], axis=1)
        
        print("Engineering punctuation counts data")
        tmp_df = self.match_punctuations(df)
        feats = pd.concat([feats, tmp_df], axis=1)

        # input words
        print("Engineering input words data")
        tmp_df = self.get_input_words(df)
        feats = pd.merge(feats, tmp_df, on='id', how='left')

        # compare feats
        print("Engineering ratios data")
        feats['word_time_ratio'] = feats['word_count_max'] / feats['up_time_max']
        feats['word_event_ratio'] = feats['word_count_max'] / feats['event_id_max']
        feats['event_time_ratio'] = feats['event_id_max']  / feats['up_time_max']
        feats['idle_time_ratio'] = feats['action_time_gap1_sum'] / feats['up_time_max']
        
        print("Done!")
        return feats


# In[28]:


preprocessor = Preprocessor(seed=42)

print("Engineering features for training data")

train_feats = preprocessor.make_feats(train_logs)

print()
print("-"*25)
print("Engineering features for test data")
print("-"*25)
test_feats = preprocessor.make_feats(test_logs)


# In[29]:


train_feats.shape


# In[30]:


train_feats


# In[31]:


num_cols


# In[32]:


train_agg_fe_df = train_logs.groupby("id")[['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']].agg(['mean', 'std', 'min', 'max', 'last', 'first', 'sem', 'median', 'sum'])
train_agg_fe_df.columns = ['_'.join(x) for x in train_agg_fe_df.columns]
train_agg_fe_df = train_agg_fe_df.add_prefix("tmp_")
train_agg_fe_df.reset_index(inplace=True)


# In[33]:


test_agg_fe_df = test_logs.groupby("id")[['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']].agg(['mean', 'std', 'min', 'max', 'last', 'first', 'sem', 'median', 'sum'])
test_agg_fe_df.columns = ['_'.join(x) for x in test_agg_fe_df.columns]
test_agg_fe_df = test_agg_fe_df.add_prefix("tmp_")
test_agg_fe_df.reset_index(inplace=True)


# In[34]:


train_agg_fe_df.head()


# In[35]:


test_agg_fe_df.head()


# In[36]:


train_feats = train_feats.merge(train_agg_fe_df, on='id', how='left')
test_feats = test_feats.merge(test_agg_fe_df, on='id', how='left')


# In[37]:


train_feats.shape, test_feats.shape


# In[38]:


train_feats = train_feats.merge(train_scores, on='id', how='left')


# In[39]:


train_feats.head()


# ## Base Model

# In[40]:


target_col = ['score']

drop_cols = ['id']

train_cols = [col for col in train_feats.columns if col not in target_col + drop_cols]

train_cols.__len__(), target_col.__len__()


# In[41]:


kf = model_selection.KFold(n_splits=5, random_state=42, shuffle=True)

oof_valid_preds = np.zeros(train_feats.shape[0], )

X_test = test_feats[train_cols]
test_predict_list = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(train_feats)):
    
    print("==-"* 50)
    print("Fold : ", fold)
    
    X_train, y_train = train_feats.iloc[train_idx][train_cols], train_feats.iloc[train_idx][target_col]
    X_valid, y_valid = train_feats.iloc[valid_idx][train_cols], train_feats.iloc[valid_idx][target_col]
    
    print("Trian :", X_train.shape, y_train.shape)
    print("Valid :", X_valid.shape, y_valid.shape)
    
    params = {
            "objective": "regression",
            "metric": "rmse",
            "n_estimators" : 10000,
            "boosting_type": "gbdt",                
            "seed": 42
    }
    
    model = lgb.LGBMRegressor(**params)
        
    early_stopping_callback = lgb.early_stopping(200, first_metric_only=True, verbose=False)
    verbose_callback = lgb.log_evaluation(100)
        
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],  
              callbacks=[early_stopping_callback, verbose_callback],
    )
        
    valid_predict = model.predict(X_valid)
    oof_valid_preds[valid_idx] = valid_predict
    
    test_predict = model.predict(X_test)
    test_predict_list.append(test_predict)
    
    score = metrics.mean_squared_error(y_valid, valid_predict, squared=False)
    print("Fold RMSE Score : ", score)

    
oof_score = metrics.mean_squared_error(train_feats[target_col], oof_valid_preds, squared=False)
print("OOF RMSE Score : ", oof_score)


# In[42]:


test_predict_list


# In[43]:


np.mean(test_predict_list, axis=0)


# In[44]:


test_feats.head()


# In[45]:


test_feats['score'] = np.mean(test_predict_list, axis=0)


# In[46]:


test_feats.head()


# In[47]:


ss_df


# In[48]:


test_feats[['id', 'score']].to_csv("submission.csv", index=False)


# ## Hyperparameter Tunning

# In[49]:


def objective(trial):
    
    params = {
        "objective": "regression",
        "metric": "rmse",
        'random_state': 48,
        "n_estimators" : 10000,
        "verbosity": -1,
        
        'max_depth': trial.suggest_categorical('max_depth', [5,10,20,40,100, -1]),
        'num_leaves' : trial.suggest_int('num_leaves', 2, 256),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 1.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        'reg_sqrt': trial.suggest_categorical('reg_sqrt', ['true', 'false']),
        
        'early_stopping_round' : 50,
        'n_jobs': -1,
    }
    
    kf = model_selection.KFold(n_splits=5, random_state=42, shuffle=True)

    oof_valid_preds = np.zeros(train_feats.shape[0], )

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_feats)):

        X_train, y_train = train_feats.iloc[train_idx][train_cols], train_feats.iloc[train_idx][target_col]
        X_valid, y_valid = train_feats.iloc[valid_idx][train_cols], train_feats.iloc[valid_idx][target_col]

        model = lgb.LGBMRegressor(**params)

        early_stopping_callback = lgb.early_stopping(200, first_metric_only=True, verbose=False)

        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],  
                  callbacks=[early_stopping_callback],
        )

        valid_predict = model.predict(X_valid)
        oof_valid_preds[valid_idx] = valid_predict
    

    oof_score = metrics.mean_squared_error(train_feats[target_col], oof_valid_preds, squared=False)
    
    return oof_score


# In[50]:


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)


# In[51]:


kf = model_selection.KFold(n_splits=10, random_state=42, shuffle=True)

oof_valid_preds = np.zeros(train_feats.shape[0], )

X_test = test_feats[train_cols]
test_predict_list = []

models_dict = {}

for fold, (train_idx, valid_idx) in enumerate(kf.split(train_feats)):
    
    print("==-"* 50)
    print("Fold : ", fold)
    
    X_train, y_train = train_feats.iloc[train_idx][train_cols], train_feats.iloc[train_idx][target_col]
    X_valid, y_valid = train_feats.iloc[valid_idx][train_cols], train_feats.iloc[valid_idx][target_col]
    
    print("Trian :", X_train.shape, y_train.shape)
    print("Valid :", X_valid.shape, y_valid.shape)
    
    params = {
            "objective": "regression",
            "metric": "rmse",
            "n_estimators" : 10000,
            "boosting_type": "gbdt",                
            "seed": 42,
        
            'reg_alpha': 0.002333698391220378, 
            'reg_lambda': 0.19122210526689265, 
            'colsample_bytree': 0.5381228886377286, 
            'subsample': 0.8921958229333354, 
            'learning_rate': 0.01836065797320474, 
            'num_leaves': 14, 
            'min_child_samples': 66
    }
    
    model = lgb.LGBMRegressor(**params)
        
    early_stopping_callback = lgb.early_stopping(200, first_metric_only=True, verbose=False)
    verbose_callback = lgb.log_evaluation(100)
        
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],  
              callbacks=[early_stopping_callback, verbose_callback],
    )
        
    valid_predict = model.predict(X_valid)
    oof_valid_preds[valid_idx] = valid_predict
    
    test_predict = model.predict(X_test)
    test_predict_list.append(test_predict)
    
    score = metrics.mean_squared_error(y_valid, valid_predict, squared=False)
    print("Fold RMSE Score : ", score)
    
    models_dict[fold] = model

    
oof_score = metrics.mean_squared_error(train_feats[target_col], oof_valid_preds, squared=False)
print("OOF RMSE Score : ", oof_score)


# In[52]:


feature_importances_values = np.asarray([model.feature_importances_ for model in models_dict.values()]).mean(axis=0)
feature_importance_df = pd.DataFrame({'name': train_cols, 'importance': feature_importances_values})

feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)


# In[53]:


plt.figure(figsize=(15, 6))

ax = sns.barplot(data=feature_importance_df.head(30), x='name', y='importance')
ax.set_title(f"Mean feature importances")
ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=90)

plt.show()


# In[54]:


test_feats['score'] = np.mean(test_predict_list, axis=0)


# In[55]:


test_feats[['id', 'score']].to_csv("submission.csv", index=False)


# In[ ]:




