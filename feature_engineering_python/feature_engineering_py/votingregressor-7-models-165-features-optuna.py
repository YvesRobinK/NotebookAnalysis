#!/usr/bin/env python
# coding: utf-8

# This notebook is forked from @AWQATAK [Silver Bullet | Single Model | 165 Features](https://www.kaggle.com/code/awqatak/silver-bullet-single-model-165-features)
# This notebook extracts 165 features for predicting writing score and trains a single, lightly hypertuned LightGBM model. This provides a simple solution that can be easily extended with additional features and experimentation with other models.
# 
# - Employ an ensemble of models for enhanced prediction accuracy
# - Include optuna code to find the optimal parameters of the model
# - Refactor the code using object-oriented principles to enhance memory management. Separate the training and evaluation functions.
# 
# 
# **Change logs**
# - [Version 84] (score=0.58) Integrates five models (`ridge` `CatBoost` `LightGBM` `XGBRegressor` `SVR`) models
# - [Version 79] (score=0.58) Integrates five models (`ridge` `CatBoost` `LightGBM` `XGBRegressor` `SVR`) models and enable optuna to find the optimal parameters for LightGBM model (early stop = 100)
# 
# - [Version 73] (score=0.58) Integrates five models (`ridge` `CatBoost` `LightGBM` `XGBRegressor` `SVR`) models 
# 
# - [Version 74] (score=0.585) Integrates four models (`CatBoost` `LightGBM` `SVR`) models and utilize the best-performing parameters from [LGBM (X2) + NN + Fusion](https://www.kaggle.com/code/kononenko/lgbm-x2-nn-fusion#Writing-Quality(fusion_notebook)) 
# - [Version 63] (score=0.58) Integrates four models (`CatBoost` `LightGBM` `XGBRegressor` `SVR`) models and utilize the best-performing parameters discovered through Optuna's hyperparameter optimization
# - [Version 58] (score=0.582) Integrates seven models (`CatBoost` `LightGBM` `XGBRegressor` `SVR` `RandomForestRegressor`, `Lasso` and `Ridge`)
# - [Version 55] (score=0.582) Integrates five models (`CatBoost` `LightGBM` `XGBRegressor` `SVR` `RandomForestRegressor`) models
# - [Version 53] (score=0.58) Integrates four models (`CatBoost` `LightGBM` `XGBRegressor` `SVR`) models with `VotingRegressor`. To addresses SVR's inability to process NaN values, fill nan values with the most frequent (mode) value in the column using `SimpleImputer`
# - [Version 49] (score=0.582) Integrates three models (`CatBoost` `LightGBM` `XGBRegressor`) models with `VotingRegressor` and use RMSE of each model as weights.
# - [Version 47] (score=0.582) Integrates three models (`CatBoost` `LightGBM` `XGBRegressor`) models with `VotingRegressor`.
# - [Version 44] (score=0.589) Integrates `CatBoost` and two`LightGBM` models with `VotingRegressor` and Optuna hyperparameter tuning for `lgbm1` model.
# - [Version 40] (score=0.583) Leverages a model ensemble comprising `CatBoost` and two `LightGBM` models, integrated using `VotingRegressor` for enhanced prediction accuracy.
# - [Version 39] (score=0.588) Included `Cat` and `LGBM` models and average the predictions of both models. 
# - [Version 10] (score=0.582) Added `FeatureExtractor` class for extracting 115 writing features from logs. Added `EssayConstructor` class for extracting 50 essay-related features (word, sentence, paragraph, keypress, and product-to-key). 
# 
# 
# 
# **References:**
# [📕 Writing Processes - Baseline V2 (LGBM + NN)](https://www.kaggle.com/code/mcpenguin/writing-processes-baseline-v2-lgbm-nn) || 
# [LGBM (X2) + NN + Fusion](https://www.kaggle.com/code/kononenko/lgbm-x2-nn-fusion)
# 
# [.581 | 3 Models | Optimized Weights | 165 Features](https://www.kaggle.com/code/snnclsr/581-3-models-optimized-weights-165-features#Solution) || 
# [lgbm + lgbm & xgb ensemble](https://www.kaggle.com/code/myeongwang/lgbm-lgbm-xgb-ensemble) ||
# [Link Writing Simple LGBM baseline](https://www.kaggle.com/code/hengzheng/link-writing-simple-lgbm-baseline) || [📒 Writing Processes to Quality - Baseline](https://www.kaggle.com/code/mcpenguin/writing-processes-to-quality-baseline/notebook) || [{ENTER}ing the TimeSeries {SPACE} Sec 3 + New Aggs](https://www.kaggle.com/code/abdullahmeda/enter-ing-the-timeseries-space-sec-3-new-aggs) || [Feature Engineering: Sentence & paragraph features](https://www.kaggle.com/code/hiarsl/feature-engineering-sentence-paragraph-features) || [Essay Contructor](https://www.kaggle.com/code/kawaiicoderuwu/essay-contructor) || [fast essay constructor](https://www.kaggle.com/code/yuriao/fast-essay-constructor)

# # Libraries

# In[1]:


import re, random, torch, pickle, gc, os, sklearn
import optuna
import lightgbm as lgb
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
## Sklearn package
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error 
from scipy.stats import skew, kurtosis
import ctypes
libc = ctypes.CDLL("libc.so.6")  # clear the memory
import warnings
warnings.filterwarnings("ignore")
pd.options.display.max_rows = 999
pd.options.display.max_colwidth = 99


# In[2]:


DEBUG = True
SEED = 42
N_FOLD = 5


# In[3]:


# Seed the same seed to all 
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
seed_everything(SEED)


# # Load train logs using `polars` dataframe

# In[4]:


data_path = '/kaggle/input/linking-writing-processes-to-writing-quality/'
train_logs = pl.scan_csv(data_path + 'train_logs.csv') # Create a datafrom using polars for fast execution
display(train_logs.collect().head()) # collect() produce the actual df
# Train_logs dataset has 11 columns


# # Extract statistical features from train logs 

# In[5]:


class FeatureExtractor():
    def __init__(self, logs):
        self.logs = logs # Training logs
        
    def count_by_values(self, colname, used_cols):
        fts = self.logs.select(pl.col('id').unique(maintain_order=True))
        for i, col in enumerate(used_cols):
            tmp_logs = self.logs.group_by('id').agg(
                            pl.col(colname).is_in([col]).sum().alias(f'{colname}_{i}_cnt')
                                    )
            fts  = fts.join(tmp_logs, on='id', how='left') 
        return fts
    
    # Create the features from statistics of activities, text changes, events
    def create_count_by_values_feats(self):
        activities = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste']
        events = ['q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft', '.',
                       ',', 'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', "'", 'Delete', 'Unidentified']
        text_changes = ['q', ' ', '.', ',', '\n', "'", '"', '-', '?', ';', '=', '/', '\\', ':']        
        #=== Create the feature columns using count by values ===
        df = self.count_by_values('activity', activities) # Create 'activity' column
        df = df.join(self.count_by_values('text_change', text_changes), on='id', how='left') 
        df = df.join(self.count_by_values('down_event', events), on='id', how='left') 
        df = df.join(self.count_by_values('up_event', events), on='id', how='left')
        # print(df.collect().head())
        return df
    
    # Create the features 
    def create_input_words_feats(self):
        # Filter no changes 
        df = self.logs.filter((~pl.col('text_change').str.contains('=>')) & (pl.col('text_change') != 'NoChange'))
        # Aggregate the text changes by id
        df = df.group_by('id').agg(pl.col('text_change').str.concat('').str.extract_all(r'q+'))
        # creates only two columns ('id' and 'text_change') 
        df = df.with_columns(input_word_count=pl.col('text_change').list.lengths(),
                             input_word_length_mean=pl.col('text_change').apply(lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0)),
                             input_word_length_max=pl.col('text_change').apply(lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0)),
                             input_word_length_std=pl.col('text_change').apply(lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0)),
                             input_word_length_median=pl.col('text_change').apply(lambda x: np.median([len(i) for i in x] if len(x) > 0 else 0)),
                             input_word_length_skew=pl.col('text_change').apply(lambda x: skew([len(i) for i in x] if len(x) > 0 else 0)))
        df = df.drop('text_change') # Remove 'text_change' to avoid including duplicated `text_change` column
        return df
    
    # Create the statistical numeric features (e.g. sum, median, mean min, 0.5_quantile)
    def create_numeric_feats(self):
        num_cols = ['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']
        df = self.logs.group_by("id").agg(pl.sum('action_time').suffix('_sum'),
                                                pl.mean(num_cols).suffix('_mean'),
                                                pl.std(num_cols).suffix('_std'),
                                                pl.median(num_cols).suffix('_median'), pl.min(num_cols).suffix('_min'), pl.max(num_cols).suffix('_max'),
                                                pl.quantile(num_cols, 0.5).suffix('_quantile'))
        return df
    
    def create_categorical_feats(self):
        df  = self.logs.group_by("id").agg(
                pl.n_unique(['activity', 'down_event', 'up_event', 'text_change']))
        return df
    
    # Create the idle time features 
    def create_idle_time_feats(self):
        df = self.logs.with_columns(pl.col('up_time').shift().over('id').alias('up_time_lagged'))
        df = df.with_columns((abs(pl.col('down_time') - pl.col('up_time_lagged')) / 1000).fill_null(0).alias('time_diff'))
        df = df.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
        df = df.group_by("id").agg(inter_key_largest_lantency = pl.max('time_diff'),
                                   inter_key_median_lantency = pl.median('time_diff'),
                                   mean_pause_time = pl.mean('time_diff'),
                                   std_pause_time = pl.std('time_diff'),
                                   total_pause_time = pl.sum('time_diff'),
                                   pauses_half_sec = pl.col('time_diff').filter((pl.col('time_diff') > 0.5) & (pl.col('time_diff') < 1)).count(),
                                   pauses_1_sec = pl.col('time_diff').filter((pl.col('time_diff') > 1) & (pl.col('time_diff') < 1.5)).count(),
                                   pauses_1_half_sec = pl.col('time_diff').filter((pl.col('time_diff') > 1.5) & (pl.col('time_diff') < 2)).count(),
                                   pauses_2_sec = pl.col('time_diff').filter((pl.col('time_diff') > 2) & (pl.col('time_diff') < 3)).count(),
                                   pauses_3_sec = pl.col('time_diff').filter(pl.col('time_diff') > 3).count(),)
        return df
    
    # Create p-bursts features using up and down time and activity
    def create_p_bursts_feats(self):
        df = self.logs.with_columns(pl.col('up_time').shift().over('id').alias('up_time_lagged'))
        df = df.with_columns((abs(pl.col('down_time') - pl.col('up_time_lagged')) / 1000).fill_null(0).alias('time_diff'))
        df = df.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
        df = df.with_columns(pl.col('time_diff')<2)
        df = df.with_columns(pl.when(pl.col("time_diff") & pl.col("time_diff").is_last()).then(pl.count()).over(pl.col("time_diff").rle_id()).alias('P-bursts'))
        df = df.drop_nulls()
        df = df.group_by("id").agg(pl.mean('P-bursts').suffix('_mean'),
                                   pl.std('P-bursts').suffix('_std'),
                                   pl.count('P-bursts').suffix('_count'),
                                   pl.median('P-bursts').suffix('_median'),
                                   pl.max('P-bursts').suffix('_max'),
                                   pl.first('P-bursts').suffix('_first'),
                                   pl.last('P-bursts').suffix('_last'))
        return df
    
    # Create r-burst features using activity 
    def create_r_bursts_feats(self):
        df = self.logs.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
        df = df.with_columns(pl.col('activity').is_in(['Remove/Cut']))
        df = df.with_columns(pl.when(pl.col("activity") & pl.col("activity").is_last()).then(pl.count()).over(pl.col("activity").rle_id()).alias('R-bursts'))
        df = df.drop_nulls()
        df = df.group_by("id").agg(pl.mean('R-bursts').suffix('_mean'),
                                   pl.std('R-bursts').suffix('_std'), 
                                   pl.median('R-bursts').suffix('_median'),
                                   pl.max('R-bursts').suffix('_max'),
                                   pl.first('R-bursts').suffix('_first'),
                                   pl.last('R-bursts').suffix('_last'))
        return df

    # Main function creates all 122 features
    def create_feats(self):
        feats = self.create_count_by_values_feats()  # 52 columns in total
#         print(f"< Count by values features > {len(feats.columns)}")        
        feats = feats.join(self.create_input_words_feats(), on='id', how='left')  # 58 columns
#         print(f"< Input words stats features > {len(feats.columns)}")
        feats = feats.join(self.create_numeric_feats(), on='id', how='left') # 89 columns
#         print(f"< Numerical features > {len(feats.columns)}")
        feats = feats.join(self.create_categorical_feats(), on='id', how='left') # 93 columns      
#         print(f"< Categorical features > {len(feats.columns)}")
        feats = feats.join(self.create_idle_time_feats(), on='id', how='left') # 103 columns
#         print(f"< Idle time features > {len(feats.columns)}")
        feats = feats.join(self.create_p_bursts_feats(), on='id', how='left') # 110 columns
#         print(f"< P-bursts features > {len(feats.columns)}")
        feats = feats.join(self.create_r_bursts_feats() , on='id', how='left') # 116 columns
#         print(f"< R-bursts features > {len(feats.columns)}")        
        return feats # 116 features


# In[6]:


fe = FeatureExtractor(train_logs)
train_feats = fe.create_feats() # Extract features from trainning logs (polars df)
train_feats = train_feats.collect().to_pandas() # Convert polars df to pandas df
train_feats.to_csv("train_feats_0.csv")
# print(train_feats.describe())
train_logs = train_logs.collect().to_pandas()  # Convert polars df to pandas df
del fe


# # Extract essay features 
# Essay features include word-level, sentence-level, paragraph-level, keypress and product-to-key features

# In[7]:


def q1(x):
    return x.quantile(0.25)
def q3(x):
    return x.quantile(0.75)

class EssayConstructor():
    def __init__(self, logs):
        self.logs = logs
        self.train_essays = self.get_train_essays(self.logs)
        self.AGGREGATIONS = ['count', 'mean', 'min', 'max', 'first', 'last', q1, 'median', q3, 'sum']
    
    # Get the essay from train logs 
    def get_train_essays(self, logs):
        def reconstruct_essay(currTextInput):
            essayText = ""
            for Input in currTextInput.values:
                if Input[0] == 'Replace':
                    replaceTxt = Input[2].split(' => ')
                    essayText = essayText[:Input[1] - len(replaceTxt[1])] + replaceTxt[1] + essayText[Input[1] - len(replaceTxt[1]) + len(replaceTxt[0]):]
                    continue
                if Input[0] == 'Paste':
                    essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
                    continue
                if Input[0] == 'Remove/Cut':
                    essayText = essayText[:Input[1]] + essayText[Input[1] + len(Input[2]):]
                    continue
                if "M" in Input[0]:
                    croppedTxt = Input[0][10:]
                    splitTxt = croppedTxt.split(' To ')
                    valueArr = [item.split(', ') for item in splitTxt]
                    moveData = (int(valueArr[0][0][1:]), int(valueArr[0][1][:-1]), int(valueArr[1][0][1:]), int(valueArr[1][1][:-1]))
                    if moveData[0] != moveData[2]:
                        if moveData[0] < moveData[2]:
                            essayText = essayText[:moveData[0]] + essayText[moveData[1]:moveData[3]] + essayText[moveData[0]:moveData[1]] + essayText[moveData[3]:]
                        else:
                            essayText = essayText[:moveData[2]] + essayText[moveData[0]:moveData[1]] + essayText[moveData[2]:moveData[0]] + essayText[moveData[1]:]
                    continue
                essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
            return essayText
        
        # Filter logs 
        df = logs[logs.activity != 'Nonproduction']
        group_df = df.groupby('id').apply(lambda x: reconstruct_essay(x[['activity', 'cursor_position', 'text_change']]))
        essay_df = pd.DataFrame({'id': df['id'].unique().tolist()})
        essay_df = essay_df.merge(group_df.rename('essay'), on='id')
        return essay_df

    # Create word level features from train essay
    def create_word_feats(self):
        df = self.train_essays.copy()
        df['word'] = df['essay'].apply(lambda x: re.split(' |\\n|\\.|\\?|\\!',x))
        df = df.explode('word')
        df['word_len'] = df['word'].apply(lambda x: len(x))
        df = df[df['word_len'] != 0] # Remove all the no-word record
        # Aggregate word level features
        word_agg_df = df[['id','word_len']].groupby(['id']).agg(self.AGGREGATIONS)
        word_agg_df.columns = ['_'.join(x) for x in word_agg_df.columns]
        word_agg_df['id'] = word_agg_df.index
        word_agg_df = word_agg_df.reset_index(drop=True)
        return word_agg_df
    # Create sentence level features
    def create_sentence_feats(self):
        df = self.train_essays.copy()
        df['sent'] = df['essay'].apply(lambda x: re.split('\\.|\\?|\\!',x))
        df = df.explode('sent')
        df['sent'] = df['sent'].apply(lambda x: x.replace('\n','').strip())
        # Number of characters in sentences
        df['sent_len'] = df['sent'].apply(lambda x: len(x))
        # Number of words in sentences
        df['sent_word_count'] = df['sent'].apply(lambda x: len(x.split(' ')))
        df = df[df.sent_len!=0].reset_index(drop=True)
        # Aggregate sentence level features
        sent_agg_df = pd.concat([df[['id','sent_len']].groupby(['id']).agg(self.AGGREGATIONS), 
                                 df[['id','sent_word_count']].groupby(['id']).agg(self.AGGREGATIONS)], axis=1)
        sent_agg_df.columns = ['_'.join(x) for x in sent_agg_df.columns]
        sent_agg_df['id'] = sent_agg_df.index
        sent_agg_df = sent_agg_df.reset_index(drop=True)
        sent_agg_df.drop(columns=["sent_word_count_count"], inplace=True)
        sent_agg_df = sent_agg_df.rename(columns={"sent_len_count":"sent_count"})
        return sent_agg_df
    # Create paragraph level features
    def create_paragraph_feats(self):
        df = self.train_essays.copy()
        df['paragraph'] = df['essay'].apply(lambda x: x.split('\n'))
        df = df.explode('paragraph')
        # Number of characters in paragraphs
        df['paragraph_len'] = df['paragraph'].apply(lambda x: len(x)) 
        # Number of words in paragraphs
        df['paragraph_word_count'] = df['paragraph'].apply(lambda x: len(x.split(' ')))
        df = df[df.paragraph_len!=0].reset_index(drop=True)
        # Aggregate paragraph level features
        paragraph_agg_df = pd.concat([df[['id','paragraph_len']].groupby(['id']).agg(self.AGGREGATIONS), 
                                      df[['id','paragraph_word_count']].groupby(['id']).agg(self.AGGREGATIONS)], axis=1) 
        paragraph_agg_df.columns = ['_'.join(x) for x in paragraph_agg_df.columns]
        paragraph_agg_df['id'] = paragraph_agg_df.index
        paragraph_agg_df = paragraph_agg_df.reset_index(drop=True)
        paragraph_agg_df.drop(columns=["paragraph_word_count_count"], inplace=True)
        paragraph_agg_df = paragraph_agg_df.rename(columns={"paragraph_len_count":"paragraph_count"})
        return paragraph_agg_df

    
    # Create product to keys features
    def create_product_to_keys_feats(self):
        essays = self.train_essays.copy()
        logs = self.logs.copy()
        essays['product_len'] = essays['essay'].str.len()
        tmp_df = logs[logs['activity'].isin(['Input', 'Remove/Cut'])].groupby(['id']).agg({'activity': 'count'}).reset_index().rename(columns={'activity': 'keys_pressed'})
        essays = essays.merge(tmp_df, on='id', how='left')
        essays['product_to_keys'] = essays['product_len'] / essays['keys_pressed']
        return essays[['id', 'product_to_keys']]

    # Create key pressed features
    def create_keys_pressed_feats(self):
        logs = self.logs.copy()
        temp_df = logs[logs['activity'].isin(['Input', 'Remove/Cut'])].groupby(['id']).agg(keys_pressed=('event_id', 'count')).reset_index()
        temp_df_2 = logs.groupby(['id']).agg(min_down_time=('down_time', 'min'), max_up_time=('up_time', 'max')).reset_index()
        temp_df = temp_df.merge(temp_df_2, on='id', how='left')
        temp_df['keys_per_second'] = temp_df['keys_pressed'] / ((temp_df['max_up_time'] - temp_df['min_down_time']) / 1000)
        return temp_df[['id', 'keys_per_second']]
    
    def create_feats(self, feats):
        feats = feats.merge(self.create_word_feats(), on='id', how='left') # 126 columns in total
#         print(f"{len(feats.columns)}")
        feats = feats.merge(self.create_sentence_feats(), on='id', how='left') # 145 columns
#         print(f"{len(feats.columns)}")
        feats = feats.merge(self.create_paragraph_feats(), on='id', how='left') # 164 columns
#         print(f"{len(feats.columns)}")
        feats = feats.merge(self.create_keys_pressed_feats(), on='id', how='left') # 166 columns
#         print(f"{len(feats.columns)}")
        feats = feats.merge(self.create_product_to_keys_feats(), on='id', how='left') # 165 columns
#         print(f"{len(feats.columns)}")
        return feats


# In[8]:


print('< Essay Reconstruction >')
ec = EssayConstructor(train_logs)
train_feats = ec.create_feats(train_feats)
# Writing to csg
train_feats.to_csv("train_feats.csv")
del ec


# # Create train data and test data
# Train data include all the extracted features, essays and scores (target)

# In[9]:


train_scores  = pd.read_csv(data_path + 'train_scores.csv')
# Merge train features and train scores as training data
train_data = train_feats.merge(train_scores, on='id', how='left')


# In[10]:


print('< Testing Data >')  # Load test data
test_logs = pl.scan_csv(data_path + 'test_logs.csv')
# Extract statistical features
fe = FeatureExtractor(test_logs)
test_feats = fe.create_feats() 
test_feats = test_feats.collect().to_pandas()
test_logs = test_logs.collect().to_pandas()
# Extract essay features
ec = EssayConstructor(test_logs)
test_feats = ec.create_feats(test_feats)
print("Nan columns of test data", test_feats.columns[test_feats.isna().any()].tolist()) # Check if any na in the test data

test_ids = test_feats['id'].values
test_x = test_feats.drop(['id'], axis=1)


# In[11]:


# Split train data into X and Y
data_X = train_data.drop(['id', 'score'], axis=1)
data_Y = train_data['score'].values


# # Train the model with 5 fold Cross validation 

# In[12]:


class ModelTrainer():
    def __init__(self, model_name, **params):
        # Model
        self.model_name = model_name
        self.params = params
        self.create_model()
        
        self.X = data_X
        self.Y = data_Y        
        print(f'Number of features: {len(self.X.columns)}')
        
    
    def make_pipeline(self, model):
        return Pipeline([
            ('remove_infs', FunctionTransformer(lambda x: np.nan_to_num(x, nan=np.nan, posinf=0, neginf=0))),
            ('imputer', SimpleImputer(strategy='mean')),
            ('normalizer', FunctionTransformer(lambda x: np.log1p(np.abs(x)))),
            ('scaler', RobustScaler()),
            ('model', model)
        ])
    
    # Create the model
    def create_model(self):
        match model_name:
            case "lgbm":
                self.model = LGBMRegressor(**self.params)
            case "xgb":
                self.model = XGBRegressor(**self.params)
            case "catboost":
                self.model = CatBoostRegressor(**self.params)
            case 'rfr':
                self.model = self.make_pipeline(RandomForestRegressor(**self.params))
            case "svr":
                self.model = self.make_pipeline(SVR(**self.params))
            case 'lasso':
                self.model = self.make_pipeline(Lasso(**self.params))
            case 'ridge':
                self.model = self.make_pipeline(Ridge(**self.params))
            case other:
                print("Not implemented")
                sys.exit(-1)
    
    # Get the trained model        
    def get_model(self):
        return self.model
        
    # Train the model with 5-fold CV
    def train_model(self):        
        early_stopping_callback = lgb.early_stopping(200, first_metric_only=True, verbose=False)
        verbose_callback = lgb.log_evaluation(100)        
        # Split the training data into 5 fold
        skf = StratifiedKFold(n_splits=N_FOLD, random_state=SEED, shuffle=True)
        fold_rmses = []
        for fold, (train_index, valid_index) in enumerate(skf.split(self.X, self.Y.astype(str))):
            train_x = self.X.iloc[train_index]
            train_y = self.Y[train_index]
            valid_x = self.X.iloc[valid_index]
            valid_y = self.Y[valid_index]
            if model_name == 'lgbm':
                # Train the model with early stop of 100 
                self.model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)],
                          callbacks=[
                                lgb.callback.early_stopping(stopping_rounds=100),
                                lgb.callback.log_evaluation(period=100),
                          ])  
            else:
                # Fit the model with train x and train y
                self.model.fit(train_x, train_y)            
            predictions = self.model.predict(valid_x)
            rmse = mean_squared_error(y_true=valid_y, y_pred=predictions, squared=False) # Return RMSE
            fold_rmses.append(rmse)
        avg_rmse = np.mean(fold_rmses)
        print(f"Average rmse: {avg_rmse}") 
        return avg_rmse
    
    # Evaluate the model with entire X data
    def evaluation(self):
        preds = self.predict(self.X)
        rmse = mean_squared_error(y_true=self.Y, y_pred=preds, squared=False)
        return rmse
        
    # Predict the test data. 
    def predict(self, test_x):
        # Prediction loop
        tests_y = np.zeros((len(test_x), N_FOLD))
        for fold in range(N_FOLD):
            preds = self.model.predict(test_x)
            tests_y[:, fold] = preds
            #print(f"Fold = {fold} Prediction = {preds[:5]}")
        test_y = np.mean(tests_y, axis=1)
        return test_y# Average the prediction of each fold model
    
    # Clear the memory
    def clear_memory(self):
        del self.model
        libc.malloc_trim(0)
        torch.cuda.empty_cache()
        gc.collect()


# # Find the best model parameters 

# In[13]:


params_dict ={}
# CatBoostRegressor
params_dict['catboost'] =  {
    "iterations": 5000,
    "early_stopping_rounds": 50,
    "depth": 6,
    "loss_function": "RMSE",
    "random_seed": SEED,
    "silent": True
}

## Best parameters of LGBM
params_dict['lgbm'] = {
    'n_estimators': 1024,
    'learning_rate': 0.005,
    'metric': 'rmse',
    'random_state': SEED,
    'force_col_wise': True,
    'verbosity': 0,
}

# XGBRegressor
params_dict['xgb'] = {
    "max_depth": 4,
    "learning_rate": 0.1,
    "objective": "reg:squarederror",
    "num_estimators": 1000,
    "num_boost_round": 1000,
    "eval_metric": "rmse",
    "seed": SEED
}
# svr
params_dict['svr'] = {
    'kernel':'rbf',
    'C':1.0,
    'epsilon': 0.1
}
# rfr 
params_dict['rfr'] = {
    'max_depth': 6,
    'max_features': 'sqrt',
    'min_impurity_decrease': 0.0016295128631816343,
    'n_estimators': 200,
    'random_state': SEED,
    }
# Ridge
params_dict['ridge'] = {
    'alpha': 1,
    'random_state': SEED,
    'solver': 'auto'
    }
# Lasso
params_dict['lasso'] = {
    'alpha': 0.04198227921905038, 
    'max_iter': 2000, 
    'random_state': SEED,
    }


# In[14]:


best_score = 1.0
# Find the optimal learning rate
def objective(trial, model_name):
    global params_dict
    # Parameters
    params = params_dict[model_name] # Load the default parameters
    # set the trial for tunable parameters
    if model_name == 'xgb':
        # Parameters for 'xgb' model
        params['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-4, 0.5)
        params['max_depth'] = trial.suggest_int('max_depth', 2, 64)
    elif model_name == 'catboost':
        params['depth'] = trial.suggest_int('depth', 2, 30)
    elif model_name == 'svr':
        params['epsilon'] = trial.suggest_float('epsilon', 0.01, 1)
    elif model_name == 'ridge':
        params['alpha'] = trial.suggest_loguniform('alpha', 1e-3, 10.0)
    elif model_name == 'lgbm':
        params['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-4, 0.5)
        params['reg_alpha'] = trial.suggest_loguniform('reg_alpha', 1e-3, 10.0)
        params['reg_lambda'] = trial.suggest_loguniform('reg_lambda', 1e-3, 10.0)
        params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1)
        params['subsample'] = trial.suggest_float('subsample', 0.5, 1)
        params['num_leaves'] = trial.suggest_int('num_leaves', 8, 64)
        params['min_child_samples'] = trial.suggest_int('min_child_samples', 1, 100)
    # Experiment the parameters
    trainer = ModelTrainer(model_name, **params)
    avg_score = trainer.train_model()
    # Save the model is the avg score > current best score
    global best_score
    if avg_score < best_score:
        best_score = avg_score
    # Clean up
    trainer.clear_memory()
    del trainer    
    print(f"Average result {avg_score} and the best score {best_score}")
    return avg_score

def run_optuna(model_name):
    study_name = f"{model_name}_study"
    study_file_path = f"/kaggle/working/{study_name}.db"
    if os.path.exists(study_file_path):
        os.remove(study_file_path)
    # # Create a study to find the optimal hyper-parameters    
    study = optuna.create_study(direction="minimize", study_name=study_name,
                                storage="sqlite:///" + f"{study_file_path}", # Storage path of the database keeping the study results
                                load_if_exists=False)  # Resume the existing study
    # Set up the timeout to avoid runing out of quote
    # n_jobs =-1 is CPU bounded
    study.optimize(lambda trial: objective(trial, model_name), 
                   n_jobs=4, n_trials=1000,
                   show_progress_bar=True, gc_after_trial=True)
    ## Print the best parameters    
    best_trial = study.best_trial
    best_params = study.best_params
    # Print out the experiment results
    print(f"Best parameters: {best_params}\n\n"
          f"Number of finished trials: {len(study.trials)}\n\n"
          f"Best trial:{best_trial}")    
    return study


# # Ensembling the model using `VoteRegressor`
# 
# Best model parameters and best weights achieve the lowest RSME 

# In[15]:


# Train the model and make the predictions
def train_model(model_name, is_loaded=True):
    best_params = params_dict[model_name]
    # If is_loaded is True, load the best parameters.
    # Otherwise, initiate an Optuna study to optimize parameters.
    if is_loaded:  # Loaded the best parameters that are found from previous experiments
        study_name = f"{model_name}_study"
        study_file_path = f"/kaggle/input/writing-quality-dataset/{study_name}.db"
        if os.path.isfile(study_file_path):
            loaded_study = optuna.load_study(study_name=study_name,
                                         storage="sqlite:///" + f"{study_file_path}")
            best_params.update(loaded_study.best_params)
            print(f"Best parameters: {best_params}\n\n")
    else:
        study = run_optuna(model_name)
        best_params.update(study.best_params)
        # Print out the experiment results
        print(f"Best parameters: {best_params}\n\n")
    ## Parameters for LGBMRegressor model
    trainer = ModelTrainer(model_name, **best_params)
    trainer.train_model()
    rmse = trainer.evaluation()
    model = trainer.get_model()
    print(f"Complete training {model_name} RMSE = {rmse}")
    return model

# Collect all the models
models = []
model_names = ['ridge', 'svr', 'catboost', 'lgbm', 'xgb'] # 5 models 
# model_names = ['lasso', 'ridge', 'rfr', 'svr', 'catboost', 'lgbm', 'xgb'] # 7 models
preds_y = []
tests_y = []
for model_name in model_names:
    is_loaded = True
#     if 'lgbm' == model_name: # Enable optuna
#         is_loaded = False
    model = train_model(model_name, is_loaded)
    models.append((model_name, model))
print(models)


# ## Use VotingRegressor class
# Use VotingRegressor to combine the predictions of three models. Model weights are determined by their  RMSE score on the full training dataset, in order to ensure optimal contribution to the ensemble's prediction.

# In[16]:


def evaluate_models(models, data_X, data_Y):
    # split the full train data (data_X and data_Y) into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(data_X, data_Y,
                                                      test_size=0.2, random_state=SEED)
    # fit and evaluate the models
    weights = list()
    for name, model in models:
        # fit the model
        model.fit(X_train, y_train)
        # evaluate the model
        y_preds = model.predict(X_val)
        # Calculate the 
        rmse = mean_squared_error(y_true=y_val, y_pred=y_preds, squared=False)
        # store the performance
        weights.append(rmse)
    # report model performance
    print(f"Weight = {weights}")
    return weights


# In[17]:


try:
    weights = evaluate_models(models, data_X, data_Y)
    # Use the weights (scores) as a weighting for the ensemble
    ensemble = VotingRegressor(estimators=models, weights=weights)
    ensemble.fit(data_X, data_Y)
    test_y = ensemble.predict(test_x)
    print(test_y)
except Exception as e: 
    print(e)


# # Submission

# In[18]:


submission = pd.DataFrame({'id': test_ids, 'score': test_y})
submission.to_csv('submission.csv', index=False)
display(submission)

