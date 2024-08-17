#!/usr/bin/env python
# coding: utf-8

# In this notebook, referring to [Getting Started](https://www.kaggle.com/code/robikscube/nfl-player-contact-detection-getting-started), I created a training model using xgboost with some features other than distance, such as speed and acceleration.

# In[1]:


import os
import torch

class Config:
    AUTHOR = "colum2131"

    NAME = "NFLC-" + "Exp001-simple-xgb-baseline"

    COMPETITION = "nfl-player-contact-detection"

    seed = 42
    num_fold = 5
    
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate':0.03,
        'tree_method':'hist' if not torch.cuda.is_available() else 'gpu_hist'
    }


# In[2]:


import os
import gc
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import Video, display

from scipy.optimize import minimize
import cv2
from glob import glob
from tqdm import tqdm

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score,
    matthews_corrcoef,
)

import xgboost as xgb

import torch

if torch.cuda.is_available():
    import cupy 
    import cudf
    from cuml import ForestInference


# In[3]:


def setup(cfg):
    cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # set dirs
    cfg.INPUT = f'../input/{cfg.COMPETITION}'
    cfg.EXP = cfg.NAME
    cfg.OUTPUT_EXP = cfg.NAME
    cfg.SUBMISSION = './'
    cfg.DATASET = '../input/'

    cfg.EXP_MODEL = os.path.join(cfg.EXP, 'model')
    cfg.EXP_FIG = os.path.join(cfg.EXP, 'fig')
    cfg.EXP_PREDS = os.path.join(cfg.EXP, 'preds')

    # make dirs
    for d in [cfg.EXP_MODEL, cfg.EXP_FIG, cfg.EXP_PREDS]:
        os.makedirs(d, exist_ok=True)
        
    return cfg


# In[4]:


# ==============================
# function
# ==============================
# ref: https://www.kaggle.com/code/robikscube/nfl-player-contact-detection-getting-started
def add_contact_id(df):
    # Create contact ids
    df["contact_id"] = (
        df["game_play"]
        + "_"
        + df["step"].astype("str")
        + "_"
        + df["nfl_player_id_1"].astype("str")
        + "_"
        + df["nfl_player_id_2"].astype("str")
    )
    return df

def expand_contact_id(df):
    """
    Splits out contact_id into seperate columns.
    """
    df["game_play"] = df["contact_id"].str[:12]
    df["step"] = df["contact_id"].str.split("_").str[-3].astype("int")
    df["nfl_player_id_1"] = df["contact_id"].str.split("_").str[-2]
    df["nfl_player_id_2"] = df["contact_id"].str.split("_").str[-1]
    return df

# cross validation
def get_groupkfold(train, target_col, group_col, n_splits):
    kf = GroupKFold(n_splits=n_splits)
    generator = kf.split(train, train[target_col], train[group_col])
    fold_series = []
    for fold, (idx_train, idx_valid) in enumerate(generator):
        fold_series.append(pd.Series(fold, index=idx_valid))
    fold_series = pd.concat(fold_series).sort_index()
    return fold_series

# xgboost code
def fit_xgboost(cfg, X, y, params, add_suffix=''):
    """
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate':0.01,
        'tree_method':'gpu_hist'
    }
    """
    oof_pred = np.zeros(len(y), dtype=np.float32)
    for fold in sorted(cfg.folds.unique()):
        if fold == -1: continue
        idx_train = (cfg.folds!=fold)
        idx_valid = (cfg.folds==fold)
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]
        display(pd.Series(y_valid).value_counts())

        xgb_train = xgb.DMatrix(x_train, label=y_train)
        xgb_valid = xgb.DMatrix(x_valid, label=y_valid)
        evals = [(xgb_train,'train'),(xgb_valid,'eval')]

        model = xgb.train(
            params,
            xgb_train,
            num_boost_round=10_000,
            early_stopping_rounds=100,
            evals=evals,
            verbose_eval=100,
        )

        model_path = os.path.join(cfg.EXP_MODEL, f'xgb_fold{fold}{add_suffix}.model')
        model.save_model(model_path)
        if not torch.cuda.is_available():
            model = xgb.Booster().load_model(model_path)
        else:
            model = ForestInference.load(model_path, output_class=True, model_type='xgboost')
        pred_i = model.predict_proba(x_valid)[:, 1]
        oof_pred[x_valid.index] = pred_i
        score = round(roc_auc_score(y_valid, pred_i), 5)
        print(f'Performance of the prediction: {score}\n')
        del model; gc.collect()

    np.save(os.path.join(cfg.EXP_PREDS, f'oof_pred{add_suffix}'), oof_pred)
    score = round(roc_auc_score(y, oof_pred), 5)
    print(f'All Performance of the prediction: {score}')
    return oof_pred

def pred_xgboost(X, data_dir, add_suffix=''):
    models = glob(os.path.join(data_dir, f'xgb_fold*{add_suffix}.model'))
    if not torch.cuda.is_available():
         models = [xgb.Booster().load_model(model_path) for model in models]
    else:
        models = [ForestInference.load(model, output_class=True, model_type='xgboost') for model in models]
    preds = np.array([model.predict_proba(X)[:, 1] for model in models])
    preds = np.mean(preds, axis=0)
    return preds


# In[5]:


# ==============================
# read data
# ==============================
cfg = setup(Config)

if not torch.cuda.is_available():
    tr_tracking = pd.read_csv(os.path.join(cfg.INPUT, 'train_player_tracking.csv'), parse_dates=["datetime"])
    te_tracking = pd.read_csv(os.path.join(cfg.INPUT, 'test_player_tracking.csv'), parse_dates=["datetime"])
    # tr_helmets = pd.read_csv(os.path.join(cfg.INPUT, 'train_baseline_helmets.csv'))
    # te_helmets = pd.read_csv(os.path.join(cfg.INPUT, 'test_baseline_helmets.csv'))
    # tr_video_metadata = pd.read_csv(os.path.join(cfg.INPUT, 'train_video_metadata.csv'))
    # te_video_metadata = pd.read_csv(os.path.join(cfg.INPUT, 'test_video_metadata.csv'))
    sub = pd.read_csv(os.path.join(cfg.INPUT, 'sample_submission.csv'))

    train = pd.read_csv(os.path.join(cfg.INPUT, 'train_labels.csv'), parse_dates=["datetime"])
    test = expand_contact_id(sub)
    
else:
    tr_tracking = cudf.read_csv(os.path.join(cfg.INPUT, 'train_player_tracking.csv'), parse_dates=["datetime"])
    te_tracking = cudf.read_csv(os.path.join(cfg.INPUT, 'test_player_tracking.csv'), parse_dates=["datetime"])
    # tr_helmets = cudf.read_csv(os.path.join(cfg.INPUT, 'train_baseline_helmets.csv'))
    # te_helmets = cudf.read_csv(os.path.join(cfg.INPUT, 'test_baseline_helmets.csv'))
    # tr_video_metadata = cudf.read_csv(os.path.join(cfg.INPUT, 'train_video_metadata.csv'))
    # te_video_metadata = cudf.read_csv(os.path.join(cfg.INPUT, 'test_video_metadata.csv'))
    sub = pd.read_csv(os.path.join(cfg.INPUT, 'sample_submission.csv'))

    train = cudf.read_csv(os.path.join(cfg.INPUT, 'train_labels.csv'), parse_dates=["datetime"])
    test = cudf.DataFrame(expand_contact_id(sub))


# The following code is used to create the features.  
# Basically, the numerical features contained in player_tracking.csv are merged into player_id_1 and player_id_2 respectively.

# In[6]:


# ==============================
# feature engineering
# ==============================
def create_features(df, tr_tracking, merge_col="step", use_cols=["x_position", "y_position"]):
    output_cols = []
    df_combo = (
        df.astype({"nfl_player_id_1": "str"})
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id",] + use_cols
            ],
            left_on=["game_play", merge_col, "nfl_player_id_1"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .rename(columns={c: c+"_1" for c in use_cols})
        .drop("nfl_player_id", axis=1)
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id"] + use_cols
            ],
            left_on=["game_play", merge_col, "nfl_player_id_2"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .drop("nfl_player_id", axis=1)
        .rename(columns={c: c+"_2" for c in use_cols})
        .sort_values(["game_play", merge_col, "nfl_player_id_1", "nfl_player_id_2"])
        .reset_index(drop=True)
    )
    output_cols += [c+"_1" for c in use_cols]
    output_cols += [c+"_2" for c in use_cols]
    
    if ("x_position" in use_cols) & ("y_position" in use_cols):
        index = df_combo['x_position_2'].notnull()
        if torch.cuda.is_available():
            index = index.to_array()
        distance_arr = np.full(len(index), np.nan)
        tmp_distance_arr = np.sqrt(
            np.square(df_combo.loc[index, "x_position_1"] - df_combo.loc[index, "x_position_2"])
            + np.square(df_combo.loc[index, "y_position_1"]- df_combo.loc[index, "y_position_2"])
        )
        if torch.cuda.is_available():
            tmp_distance_arr = tmp_distance_arr.to_array()
        distance_arr[index] = tmp_distance_arr
        df_combo['distance'] = distance_arr
        output_cols += ["distance"]
        
    df_combo['G_flug'] = (df_combo['nfl_player_id_2']=="G")
    output_cols += ["G_flug"]
    return df_combo, output_cols


use_cols = [
    'x_position', 'y_position', 'speed', 'distance',
    'direction', 'orientation', 'acceleration', 'sa'
]
train, feature_cols = create_features(train, tr_tracking, use_cols=use_cols)
test, feature_cols = create_features(test, te_tracking, use_cols=use_cols)
if torch.cuda.is_available():
    train = train.to_pandas()
    test = test.to_pandas()

display(train)


# In[7]:


# ==============================
# training & inference
# ==============================
train_X = train[feature_cols]
test_X = test[feature_cols]
train_y = train['contact']
cfg.folds = get_groupkfold(train, 'contact', 'game_play', cfg.num_fold)
cfg.folds.to_csv(os.path.join(cfg.EXP_PREDS, 'folds.csv'), index=False)

oof_pred = fit_xgboost(cfg, train_X, train_y, cfg.xgb_params, add_suffix="_xgb_1st")
sub_pred = pred_xgboost(test_X, cfg.EXP_MODEL, add_suffix="_xgb_1st")


# In[8]:


# ==============================
# optimize
# ==============================
def func(x_list):
    score = matthews_corrcoef(train['contact'], oof_pred>x_list[0])
    return -score

x0 = [0.5]
result = minimize(func, x0,  method="nelder-mead")
cfg.threshold = result.x[0]
print("score:", round(matthews_corrcoef(train['contact'], oof_pred>cfg.threshold), 5))
print("threshold", round(cfg.threshold, 5))

test = add_contact_id(test)
test['contact'] = (sub_pred > cfg.threshold).astype(int)
test[['contact_id', 'contact']].to_csv('submission.csv', index=False)
display(test[['contact_id', 'contact']].head())

