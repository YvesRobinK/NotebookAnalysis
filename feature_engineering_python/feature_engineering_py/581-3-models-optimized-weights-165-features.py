#!/usr/bin/env python
# coding: utf-8

# # New Contribution
# 
# In this notebook, instead of using a single model, we utilise three models Catboost, XGBoost, and LGBM respectively. We also show how to optimize the weights of individual models using [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html). Please upvote if you find this useful!
# 
# The original notebook: https://www.kaggle.com/code/awqatak/silver-bullet-single-model-165-features

# In this notebook, we use 165 features and a single LGBM model (not overly hypertuned) to provide a simple solution to this problem. You can increase the performance by adding more features, ensembling, hypertuning. It is noteworthy that there are less than 2500 samples in the dataset, so don't add alot features to avoid the dimensionality problem. Most of the features are based on the references below. Our addition is summed as follows:
# * Bursts features.
# * Ratio of final product length to keys pressed.
# * Keys per second.
# * A simple filter added to previously public features.
# 
# **Note that this notebook is currently at the 11th for efficiency track.**
# 
# References:
# * https://www.kaggle.com/code/hengzheng/link-writing-simple-lgbm-baseline
# * https://www.kaggle.com/code/mcpenguin/writing-processes-to-quality-baseline/notebook
# * https://www.kaggle.com/code/abdullahmeda/enter-ing-the-timeseries-space-sec-3-new-aggs
# * https://www.kaggle.com/code/hiarsl/feature-engineering-sentence-paragraph-features
# * https://www.kaggle.com/code/kawaiicoderuwu/essay-contructor
# * https://www.kaggle.com/code/yuriao/fast-essay-constructor

# # Libraries

# In[1]:


import polars as pl
import pandas as pd
import numpy as np
import re
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings("ignore")


# # Polars FE & Helper Functions

# In[2]:


num_cols = ['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']
activities = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste']
events = ['q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft', '.', ',', 'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', "'", 'Delete', 'Unidentified']
text_changes = ['q', ' ', '.', ',', '\n', "'", '"', '-', '?', ';', '=', '/', '\\', ':']


def count_by_values(df, colname, values):
    fts = df.select(pl.col('id').unique(maintain_order=True))
    for i, value in enumerate(values):
        tmp_df = df.group_by('id').agg(pl.col(colname).is_in([value]).sum().alias(f'{colname}_{i}_cnt'))
        fts  = fts.join(tmp_df, on='id', how='left') 
    return fts


def dev_feats(df):
    
    print("< Count by values features >")
    
    feats = count_by_values(df, 'activity', activities)
    feats = feats.join(count_by_values(df, 'text_change', text_changes), on='id', how='left') 
    feats = feats.join(count_by_values(df, 'down_event', events), on='id', how='left') 
    feats = feats.join(count_by_values(df, 'up_event', events), on='id', how='left') 

    print("< Input words stats features >")

    temp = df.filter((~pl.col('text_change').str.contains('=>')) & (pl.col('text_change') != 'NoChange'))
    temp = temp.group_by('id').agg(pl.col('text_change').str.concat('').str.extract_all(r'q+'))
    temp = temp.with_columns(input_word_count = pl.col('text_change').list.lengths(),
                             input_word_length_mean = pl.col('text_change').apply(lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0)),
                             input_word_length_max = pl.col('text_change').apply(lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0)),
                             input_word_length_std = pl.col('text_change').apply(lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0)),
                             input_word_length_median = pl.col('text_change').apply(lambda x: np.median([len(i) for i in x] if len(x) > 0 else 0)),
                             input_word_length_skew = pl.col('text_change').apply(lambda x: skew([len(i) for i in x] if len(x) > 0 else 0)))
    temp = temp.drop('text_change')
    feats = feats.join(temp, on='id', how='left') 


    
    print("< Numerical columns features >")

    temp = df.group_by("id").agg(pl.sum('action_time').suffix('_sum'), pl.mean(num_cols).suffix('_mean'), pl.std(num_cols).suffix('_std'),
                                 pl.median(num_cols).suffix('_median'), pl.min(num_cols).suffix('_min'), pl.max(num_cols).suffix('_max'),
                                 pl.quantile(num_cols, 0.5).suffix('_quantile'))
    feats = feats.join(temp, on='id', how='left') 


    print("< Categorical columns features >")
    
    temp  = df.group_by("id").agg(pl.n_unique(['activity', 'down_event', 'up_event', 'text_change']))
    feats = feats.join(temp, on='id', how='left') 


    
    print("< Idle time features >")

    temp = df.with_columns(pl.col('up_time').shift().over('id').alias('up_time_lagged'))
    temp = temp.with_columns((abs(pl.col('down_time') - pl.col('up_time_lagged')) / 1000).fill_null(0).alias('time_diff'))
    temp = temp.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
    temp = temp.group_by("id").agg(inter_key_largest_lantency = pl.max('time_diff'),
                                   inter_key_median_lantency = pl.median('time_diff'),
                                   mean_pause_time = pl.mean('time_diff'),
                                   std_pause_time = pl.std('time_diff'),
                                   total_pause_time = pl.sum('time_diff'),
                                   pauses_half_sec = pl.col('time_diff').filter((pl.col('time_diff') > 0.5) & (pl.col('time_diff') < 1)).count(),
                                   pauses_1_sec = pl.col('time_diff').filter((pl.col('time_diff') > 1) & (pl.col('time_diff') < 1.5)).count(),
                                   pauses_1_half_sec = pl.col('time_diff').filter((pl.col('time_diff') > 1.5) & (pl.col('time_diff') < 2)).count(),
                                   pauses_2_sec = pl.col('time_diff').filter((pl.col('time_diff') > 2) & (pl.col('time_diff') < 3)).count(),
                                   pauses_3_sec = pl.col('time_diff').filter(pl.col('time_diff') > 3).count(),)
    feats = feats.join(temp, on='id', how='left') 
    
    print("< P-bursts features >")

    temp = df.with_columns(pl.col('up_time').shift().over('id').alias('up_time_lagged'))
    temp = temp.with_columns((abs(pl.col('down_time') - pl.col('up_time_lagged')) / 1000).fill_null(0).alias('time_diff'))
    temp = temp.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
    temp = temp.with_columns(pl.col('time_diff')<2)
    temp = temp.with_columns(pl.when(pl.col("time_diff") & pl.col("time_diff").is_last()).then(pl.count()).over(pl.col("time_diff").rle_id()).alias('P-bursts'))
    temp = temp.drop_nulls()
    temp = temp.group_by("id").agg(pl.mean('P-bursts').suffix('_mean'), pl.std('P-bursts').suffix('_std'), pl.count('P-bursts').suffix('_count'),
                                   pl.median('P-bursts').suffix('_median'), pl.max('P-bursts').suffix('_max'),
                                   pl.first('P-bursts').suffix('_first'), pl.last('P-bursts').suffix('_last'))
    feats = feats.join(temp, on='id', how='left') 


    print("< R-bursts features >")

    temp = df.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
    temp = temp.with_columns(pl.col('activity').is_in(['Remove/Cut']))
    temp = temp.with_columns(pl.when(pl.col("activity") & pl.col("activity").is_last()).then(pl.count()).over(pl.col("activity").rle_id()).alias('R-bursts'))
    temp = temp.drop_nulls()
    temp = temp.group_by("id").agg(pl.mean('R-bursts').suffix('_mean'), pl.std('R-bursts').suffix('_std'), 
                                   pl.median('R-bursts').suffix('_median'), pl.max('R-bursts').suffix('_max'),
                                   pl.first('R-bursts').suffix('_first'), pl.last('R-bursts').suffix('_last'))
    feats = feats.join(temp, on='id', how='left')
    
    return feats


# # Pandas FE & Helper Functions

# In[3]:


def q1(x):
    return x.quantile(0.25)
def q3(x):
    return x.quantile(0.75)

AGGREGATIONS = ['count', 'mean', 'min', 'max', 'first', 'last', q1, 'median', q3, 'sum']

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


def get_essay_df(df):
    df       = df[df.activity != 'Nonproduction']
    temp     = df.groupby('id').apply(lambda x: reconstruct_essay(x[['activity', 'cursor_position', 'text_change']]))
    essay_df = pd.DataFrame({'id': df['id'].unique().tolist()})
    essay_df = essay_df.merge(temp.rename('essay'), on='id')
    return essay_df


def word_feats(df):
    essay_df = df
    df['word'] = df['essay'].apply(lambda x: re.split(' |\\n|\\.|\\?|\\!',x))
    df = df.explode('word')
    df['word_len'] = df['word'].apply(lambda x: len(x))
    df = df[df['word_len'] != 0]

    word_agg_df = df[['id','word_len']].groupby(['id']).agg(AGGREGATIONS)
    word_agg_df.columns = ['_'.join(x) for x in word_agg_df.columns]
    word_agg_df['id'] = word_agg_df.index
    word_agg_df = word_agg_df.reset_index(drop=True)
    return word_agg_df


def sent_feats(df):
    df['sent'] = df['essay'].apply(lambda x: re.split('\\.|\\?|\\!',x))
    df = df.explode('sent')
    df['sent'] = df['sent'].apply(lambda x: x.replace('\n','').strip())
    # Number of characters in sentences
    df['sent_len'] = df['sent'].apply(lambda x: len(x))
    # Number of words in sentences
    df['sent_word_count'] = df['sent'].apply(lambda x: len(x.split(' ')))
    df = df[df.sent_len!=0].reset_index(drop=True)

    sent_agg_df = pd.concat([df[['id','sent_len']].groupby(['id']).agg(AGGREGATIONS), 
                             df[['id','sent_word_count']].groupby(['id']).agg(AGGREGATIONS)], axis=1)
    sent_agg_df.columns = ['_'.join(x) for x in sent_agg_df.columns]
    sent_agg_df['id'] = sent_agg_df.index
    sent_agg_df = sent_agg_df.reset_index(drop=True)
    sent_agg_df.drop(columns=["sent_word_count_count"], inplace=True)
    sent_agg_df = sent_agg_df.rename(columns={"sent_len_count":"sent_count"})
    return sent_agg_df


def parag_feats(df):
    df['paragraph'] = df['essay'].apply(lambda x: x.split('\n'))
    df = df.explode('paragraph')
    # Number of characters in paragraphs
    df['paragraph_len'] = df['paragraph'].apply(lambda x: len(x)) 
    # Number of words in paragraphs
    df['paragraph_word_count'] = df['paragraph'].apply(lambda x: len(x.split(' ')))
    df = df[df.paragraph_len!=0].reset_index(drop=True)
    
    paragraph_agg_df = pd.concat([df[['id','paragraph_len']].groupby(['id']).agg(AGGREGATIONS), 
                                  df[['id','paragraph_word_count']].groupby(['id']).agg(AGGREGATIONS)], axis=1) 
    paragraph_agg_df.columns = ['_'.join(x) for x in paragraph_agg_df.columns]
    paragraph_agg_df['id'] = paragraph_agg_df.index
    paragraph_agg_df = paragraph_agg_df.reset_index(drop=True)
    paragraph_agg_df.drop(columns=["paragraph_word_count_count"], inplace=True)
    paragraph_agg_df = paragraph_agg_df.rename(columns={"paragraph_len_count":"paragraph_count"})
    return paragraph_agg_df

def product_to_keys(logs, essays):
    essays['product_len'] = essays.essay.str.len()
    tmp_df = logs[logs.activity.isin(['Input', 'Remove/Cut'])].groupby(['id']).agg({'activity': 'count'}).reset_index().rename(columns={'activity': 'keys_pressed'})
    essays = essays.merge(tmp_df, on='id', how='left')
    essays['product_to_keys'] = essays['product_len'] / essays['keys_pressed']
    return essays[['id', 'product_to_keys']]

def get_keys_pressed_per_second(logs):
    temp_df = logs[logs['activity'].isin(['Input', 'Remove/Cut'])].groupby(['id']).agg(keys_pressed=('event_id', 'count')).reset_index()
    temp_df_2 = logs.groupby(['id']).agg(min_down_time=('down_time', 'min'), max_up_time=('up_time', 'max')).reset_index()
    temp_df = temp_df.merge(temp_df_2, on='id', how='left')
    temp_df['keys_per_second'] = temp_df['keys_pressed'] / ((temp_df['max_up_time'] - temp_df['min_down_time']) / 1000)
    return temp_df[['id', 'keys_per_second']]


# # Solution

# In[4]:


data_path     = '/kaggle/input/linking-writing-processes-to-writing-quality/'
train_logs    = pl.scan_csv(data_path + 'train_logs.csv')
train_feats   = dev_feats(train_logs)
train_feats   = train_feats.collect().to_pandas()

print('< Essay Reconstruction >')
train_logs             = train_logs.collect().to_pandas()
train_essays           = get_essay_df(train_logs)
train_feats            = train_feats.merge(word_feats(train_essays), on='id', how='left')
train_feats            = train_feats.merge(sent_feats(train_essays), on='id', how='left')
train_feats            = train_feats.merge(parag_feats(train_essays), on='id', how='left')
train_feats            = train_feats.merge(get_keys_pressed_per_second(train_logs), on='id', how='left')
train_feats            = train_feats.merge(product_to_keys(train_logs, train_essays), on='id', how='left')


print('< Mapping >')
train_scores   = pd.read_csv(data_path + 'train_scores.csv')
data           = train_feats.merge(train_scores, on='id', how='left')
x              = data.drop(['id', 'score'], axis=1)
y              = data['score'].values
print(f'Number of features: {len(x.columns)}')


print('< Testing Data >')
test_logs   = pl.scan_csv(data_path + 'test_logs.csv')
test_feats  = dev_feats(test_logs)
test_feats  = test_feats.collect().to_pandas()

test_logs             = test_logs.collect().to_pandas()
test_essays           = get_essay_df(test_logs)
test_feats            = test_feats.merge(word_feats(test_essays), on='id', how='left')
test_feats            = test_feats.merge(sent_feats(test_essays), on='id', how='left')
test_feats            = test_feats.merge(parag_feats(test_essays), on='id', how='left')
test_feats            = test_feats.merge(get_keys_pressed_per_second(test_logs), on='id', how='left')
test_feats            = test_feats.merge(product_to_keys(test_logs, test_essays), on='id', how='left')

test_ids = test_feats['id'].values
testin_x = test_feats.drop(['id'], axis=1)

# print('< Learning and Evaluation >')
# param = {'n_estimators': 1024,
#          'learning_rate': 0.005,
#          'metric': 'rmse',
#          'random_state': 42,
#          'force_col_wise': True,
#          'verbosity': 0,}
# solution = LGBMRegressor(**param)
# y_pred   = evaluate(x.copy(), y.copy(), solution, test_x=testin_x.copy()) 

# sub = pd.DataFrame({'id': test_ids, 'score': y_pred})
# sub.to_csv('submission.csv', index=False)


# In[5]:


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
data["fold"] = -1
for idx, (train_idx, val_idx) in enumerate(skf.split(data, data["score"].astype(str))):
    data.loc[val_idx, "fold"] = idx


# In[6]:


from sklearn.metrics import mean_squared_error

rmse = lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False)


# In[7]:


from collections import defaultdict

drop_cols = ["id", "score", "fold"]
oof_df = pd.DataFrame()
models = defaultdict(list)
models_to_ensemble = ["catboost", "xgboost", "lgbm"]
model_params = {
    "catboost": {
        "iterations": 5000,
        "early_stopping_rounds": 50,
        "depth": 6,
        "loss_function": "RMSE",
        "random_seed": 42,
        "silent": True
    },
    "lgbm": {
        'n_estimators': 1024,
        'learning_rate': 0.005,
        'metric': 'rmse',
        'random_state': 42,
        'force_col_wise': True,
        'verbosity': 0
    },
    "xgboost": {
        "max_depth": 4,
        "learning_rate": 0.1,
        "objective": "reg:squarederror",
        "num_estimators": 1000,
        "num_boost_round": 1000,
        "eval_metric": "rmse",
        "seed": 42
    },
}
for idx, model_name in enumerate(models_to_ensemble):
    params = model_params[model_name]
    oof_folds = pd.DataFrame()
    print(f"Started the {model_name} model...")
    for fold in range(n_splits):
        if model_name == "lgbm":
            model = LGBMRegressor(**params)
        elif model_name == "xgboost":
            model = xgb.XGBRegressor(**params)
        elif model_name == "catboost":
            model = CatBoostRegressor(**params)
        else:
            raise ValueError("Unknown base model name.")

        x_train = data[data["fold"] != fold].reset_index(drop=True)
        x_valid = data[data["fold"] == fold].reset_index(drop=True)

        y_train = x_train["score"]
        y_valid = x_valid["score"]
        ids = x_valid["id"]

        x_train = x_train.drop(drop_cols, axis="columns")
        x_valid = x_valid.drop(drop_cols, axis="columns")
        model.fit(x_train, y_train)
        val_preds = model.predict(x_valid)
        oof_fold = pd.concat(
            [ids, y_valid, pd.Series(val_preds)], 
            axis=1).rename({0: f"{model_name}_preds"}, axis="columns")
        oof_folds = pd.concat([oof_folds, oof_fold])
        models[model_name].append(model)
        print(f"Fold: {fold} - Score: {rmse(oof_fold['score'], oof_fold[f'{model_name}_preds']):.5f}")
    
    if idx == 0:
        oof_df = pd.concat([oof_df, oof_folds])
    else:
        oof_df[f"{model_name}_preds"] = oof_folds[f"{model_name}_preds"]
    cv_score = rmse(oof_df["score"], oof_df[f"{model_name}_preds"])
    print(f"{model_name} cv_score: ", round(cv_score, 5))


# In[8]:


import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

pred_cols = [f"{model_name}_preds" for model_name in models_to_ensemble]
true_targets = oof_df["score"]

def objective_function(weights):
    ensemble_preds = (oof_df[pred_cols] * weights).sum(axis=1)
    score = rmse(oof_df["score"], ensemble_preds)
    return score

def find_weights(oof_df):
    len_models = len(models_to_ensemble)
    initial_weights = np.ones(len_models) / len_models
    bounds = [(0, 1)] * len_models
    result = minimize(objective_function, initial_weights, bounds=bounds, method='SLSQP') # L-BFGS-B
    optimized_weights = result.x
    optimized_weights /= np.sum(optimized_weights)
    return optimized_weights

optimized_weights = find_weights(oof_df)
print("Optimized Weights:", optimized_weights)


# In[9]:


oof_df["ensemble_optimized_preds"] = (oof_df[pred_cols] * optimized_weights).sum(axis=1)
cv_optimized = rmse(oof_df["score"], oof_df["ensemble_optimized_preds"])
print("cv_score with optimized weights: ", round(cv_optimized, 5))


# In[10]:


oof_df.to_csv("oof_df.csv")


# In[11]:


preds = None
for weights, model_name in zip(optimized_weights, models_to_ensemble):
    models_list = models[model_name]
    current_preds = weights * np.mean([model.predict(testin_x) for model in models_list], axis=0)
    if preds is None:
        preds = current_preds
    else:
        preds += current_preds


# In[12]:


y_pred = preds
sub = pd.DataFrame({'id': test_ids, 'score': y_pred})
sub.to_csv('submission.csv', index=False)


# In[13]:


sub


# In[ ]:




