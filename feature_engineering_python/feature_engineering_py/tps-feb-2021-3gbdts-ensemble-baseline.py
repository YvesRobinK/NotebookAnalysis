#!/usr/bin/env python
# coding: utf-8

# # About
# 
# I reused my notebook in Tabular Playground Series - Jan 2021:  
# https://www.kaggle.com/ttahara/tps-jan-2021-gbdts-baseline
# 
# <br>
# 
# * GBDT Models baseline
#     * LightGBM, XGBoost, CatBoost
#     * each model is trained by 5 folds cross validation
# 
# 
# * feature engineering
#     * **label-encoding** for category features
#     * no feature engineering for continuous features
# 
# 
# * inference test by **weighted** averaging 3 GBDT Models(5 folds averaging)
# 
# <br>
# 
# There is a lot of room for improvement such as feature engineering, parameter tuning, other models, and so on. enjoy ;) 
# 

# # Prepare

# ## import libraries

# In[1]:


import os
import sys
import time
import random
import logging
import typing as tp
from pathlib import Path
from contextlib import contextmanager

from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

import category_encoders as ce
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error, mean_squared_error

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoost, Pool

get_ipython().run_line_magic('matplotlib', 'inline')


# ## read data

# In[2]:


ROOT = Path.cwd().parent
INPUT = ROOT / "input"
DATA = INPUT / "tabular-playground-series-feb-2021"
WORK = ROOT / "working"

for path in DATA.iterdir():
    print(path.name)


# In[3]:


train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
smpl_sub = pd.read_csv(DATA / "sample_submission.csv")
print("train: {}, test: {}, sample sub: {}".format(
    train.shape, test.shape, smpl_sub.shape
))


# In[4]:


train.head().T


# ## Definition

# In[5]:


@contextmanager
def timer(logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None):
    if prefix: format_str = str(prefix) + format_str
    if suffix: format_str = format_str + str(suffix)
    start = time.time()
    yield
    d = time.time() - start
    out_str = format_str.format(d)
    if logger:
        logger.info(out_str)
    else:
        print(out_str)


# In[6]:


def rmse(y_true, y_pred):
    """"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# In[7]:


class TreeModel:
    """Wrapper for LightGBM/XGBoost/CATBoost"""
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.trn_data = None
        self.val_data = None
        self.model = None

    def train(self,
              params: dict,
              X_train: pd.DataFrame, y_train: np.ndarray,
              X_val: pd.DataFrame, y_val: np.ndarray,
              train_weight: tp.Optional[np.ndarray] = None,
              val_weight: tp.Optional[np.ndarray] = None,
              train_params: dict = None,
              cat_cols: list = None,
            ):
        if self.model_type == "lgb":
            self.trn_data = lgb.Dataset(X_train, label=y_train, weight=train_weight)
            self.val_data = lgb.Dataset(X_val, label=y_val, weight=val_weight)
            self.model = lgb.train(params=params,
                                   train_set=self.trn_data,
                                   valid_sets=[self.trn_data, self.val_data],
                                   **train_params)
        elif self.model_type == "xgb":
            self.trn_data = xgb.DMatrix(X_train, y_train, weight=train_weight)
            self.val_data = xgb.DMatrix(X_val, y_val, weight=val_weight)
            self.model = xgb.train(params=params,
                                   dtrain=self.trn_data,
                                   evals=[(self.trn_data, "train"), (self.val_data, "val")],
                                   **train_params)
        elif self.model_type == "cat":
            self.trn_data = Pool(
                X_train, label=y_train, cat_features=cat_cols)  #, group_id=[0] * len(X_train))
            self.val_data = Pool(
                X_val, label=y_val, cat_features=cat_cols)  #, group_id=[0] * len(X_val))
            self.model = CatBoost(params)
            self.model.fit(
                self.trn_data, eval_set=[self.val_data], use_best_model=True, **train_params)
        else:
            raise NotImplementedError

    def predict(self, X: pd.DataFrame):
        if self.model_type == "lgb":
            return self.model.predict(
                X, num_iteration=self.model.best_iteration)  # type: ignore
        elif self.model_type == "xgb":
            X_DM = xgb.DMatrix(X)
            return self.model.predict(
                X_DM, ntree_limit=self.model.best_ntree_limit)  # type: ignore
        elif self.model_type == "cat":
            return self.model.predict(X)
        else:
            raise NotImplementedError

    @property
    def feature_names_(self):
        if self.model_type == "lgb":
            return self.model.feature_name()
        elif self.model_type == "xgb":
            return list(self.model.get_score(importance_type="gain").keys())
        elif self.model_type == "cat":
             return self.model.feature_names_
        else:
            raise NotImplementedError

    @property
    def feature_importances_(self):
        if self.model_type == "lgb":
            return self.model.feature_importance(importance_type="gain")
        elif self.model_type == "xgb":
            return list(self.model.get_score(importance_type="gain").values())
        elif self.model_type == "cat":
            return self.model.feature_importances_
        else:
            raise NotImplementedError


# # Training & Inference

# ## Config 

# In[8]:


ID_COL = "id"
CAT_COLS= [f"cat{i}" for i in range(10)]
CONT_COLS = [f"cont{i}" for i in range(14)]
TGT_COL = "target"

N_SPLITS = 10
RANDOM_SEED_LIST = [
    42,
#   2021, 0, 1086, 39
]


# ## Feature Engineering

# In[9]:


use_feat_cols = []
train_feat = train[[ID_COL]].copy()
test_feat = test[[ID_COL]].copy()


# ### for categorical features
# 
# apply label encoding using [`category_encoders.OrdinalEncoder`](https://contrib.scikit-learn.org/category_encoders/ordinal.html)

# In[10]:


ord_enc = ce.OrdinalEncoder(cols=CAT_COLS)
train_cat_feat = ord_enc.fit_transform(train[CAT_COLS])
test_cat_feat = ord_enc.transform(test[CAT_COLS])


# In[11]:


train_feat = pd.concat([
    train_feat, train_cat_feat], axis=1)
test_feat = pd.concat([
    test_feat, test_cat_feat], axis=1)
use_feat_cols.extend(train_cat_feat.columns)


# ### for continuous features
# 
# Use them as they are

# In[12]:


train_cont_feat = train[CONT_COLS]
test_cont_feat = test[CONT_COLS]


# In[13]:


train_feat = pd.concat([
    train_feat, train_cont_feat], axis=1)
test_feat = pd.concat([
    test_feat, test_cont_feat], axis=1)
use_feat_cols.extend(CONT_COLS)


# In[14]:


train_feat.head().T


# In[15]:


test_feat.head().T


# ## Training

# In[16]:


def run_train_and_inference(
    X, X_test, y, use_model, model_params, train_params, seed_list, n_splits, cat_cols=None
):
    
    oof_pred_arr = np.zeros(len(X))
    test_pred_arr = np.zeros(len(X_test))
    feature_importances = pd.DataFrame()
    score_list = []
    
    for seed in seed_list:
        if use_model == "cat":
            model_params['random_state'] = seed
        else:
            model_params["seed"] = seed
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        tmp_oof_pred = np.zeros(len(X))
        tmp_test_pred = np.zeros(len(X_test))

        for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
            print("*" * 100)
            print(f"Seed: {seed} - Fold: {fold}")
            X_trn = X.loc[trn_idx].reset_index(drop=True)
            X_val = X.loc[val_idx].reset_index(drop=True)
            y_trn = y[trn_idx]
            y_val = y[val_idx]

            model = TreeModel(model_type=use_model)
            with timer(prefix="Model training"):
                model.train(
                    params=model_params, X_train=X_trn, y_train=y_trn,
                    X_val=X_val, y_val=y_val, train_params=train_params, cat_cols=cat_cols
                )
            with timer(prefix="Get Feature Importance"):
                fi_tmp = pd.DataFrame()
                fi_tmp["feature"] = model.feature_names_
                fi_tmp["importance"] = model.feature_importances_
                fi_tmp["fold"] = fold
                fi_tmp["seed"] = seed
                feature_importances = feature_importances.append(fi_tmp)

            with timer(prefix="Predict Valid"):
                val_pred = model.predict(X_val)
                score = mean_squared_error(y_val, val_pred, squared=False)
                # score = rmse(y_val, val_pred)
                print(f"score: {score:.5f}")
                score_list.append([seed, fold, score])
                tmp_oof_pred[val_idx] = val_pred
                tmp_test_pred += model.predict(X_test)
            
        oof_score = mean_squared_error(y, tmp_oof_pred, squared=False)
        # oof_score = rmse(y, tmp_oof_pred)
        print(f"oof score: {oof_score: 5f}")
        score_list.append([seed, "oof", oof_score])

        oof_pred_arr += tmp_oof_pred
        test_pred_arr += tmp_test_pred / n_splits

    oof_pred_arr /= len(seed_list)
    test_pred_arr /= len(seed_list)
    
    oof_score = mean_squared_error(y, oof_pred_arr, squared=False)
    # oof_score = rmse(y, oof_pred_arr)
    score_list.append(["avg", "oof", oof_score])
    score_df = pd.DataFrame(
        score_list, columns=["seed", "fold", "rmse score"])
    
    return oof_pred_arr, test_pred_arr, score_df, feature_importances


# In[17]:


X = train_feat[use_feat_cols]
X_test = test_feat[use_feat_cols]

y = train[TGT_COL].values

print(f"train_feat: {X.shape}, test_feat: {X_test.shape}")


# In[18]:


X_cat = X.copy()
X_cat[CAT_COLS] = train[CAT_COLS]
X_test_cat = X_test.copy()
X_test_cat = test[CAT_COLS]


# In[19]:


MODEL_PARAMS = {
    "lgb": {
        "objective": "root_mean_squared_error",
        "boosting": "gbdt",
        "max_depth": 8,
        "learning_rate": 0.005,
        "colsample_bytree": 0.2,
        "subsample": 0.8,
        "subsample_freq": 6,
        "reg_alpha": 20,
        "min_data_in_leaf": 200,
        "n_jobs": 2,
        "seed": RANDOM_SEED_LIST[0],
        # "device": "gpu",
        # "gpu_device_id": 0
    },
    "xgb": {
        "objective": "reg:squarederror",
        "max_depth": 8,
        "learning_rate": 0.003,
        "colsample_bytree": 0.2,
        "subsample": 0.8,
        "reg_alpha" : 6,
        "min_child_weight": 200,
        "n_jobs": 2,
        "seed": RANDOM_SEED_LIST[0],
        'tree_method': "gpu_hist",
        "gpu_id": 0,
    },
    "cat": {
        'loss_function': 'RMSE',
        "max_depth": 4,
        'learning_rate': 0.03,
        "bootstrap_type": 'Poisson',
        "subsample": 0.8,
        "border_count": 512,
        "l2_leaf_reg": 200,
        'random_state': RANDOM_SEED_LIST[0],
        "thread_count": 2,
        "task_type": "GPU",
        "devices" : "0",
        'num_boost_round': 50000,
    }
}
TRAIN_PARAMS = {
    "lgb": {
        "num_boost_round": 50000,
        "early_stopping_rounds": 200,
        "verbose_eval": 200,
    },
    "xgb": {
        "num_boost_round": 50000,
        "early_stopping_rounds": 200,
        "verbose_eval":  200,
    },
    "cat": {
        'early_stopping_rounds': 200,
        'verbose_eval': 200,
    }
}


# In[ ]:





# ### LightGBM

# In[20]:


oof_pred_lgb, test_pred_lgb, score_lgb, feat_imps_lgb = run_train_and_inference(
    X, X_test, y, "lgb", MODEL_PARAMS["lgb"], TRAIN_PARAMS["lgb"], RANDOM_SEED_LIST, N_SPLITS)


# In[21]:


score_lgb


# In[22]:


score_lgb.loc[score_lgb.fold == "oof"]


# In[23]:


order = list(feat_imps_lgb.groupby("feature").mean().sort_values("importance", ascending=False).index)
plt.figure(figsize=(10, 10))
sns.barplot(x="importance", y="feature", data=feat_imps_lgb, order=order)
plt.title("{} importance".format("lgb"))
plt.tight_layout()


# ### XGBoost

# In[24]:


oof_pred_xgb, test_pred_xgb, score_xgb, feat_imps_xgb = run_train_and_inference(
    X, X_test, y, "xgb", MODEL_PARAMS["xgb"], TRAIN_PARAMS["xgb"], RANDOM_SEED_LIST, N_SPLITS)


# In[25]:


score_xgb


# In[26]:


score_xgb.loc[score_xgb.fold == "oof"]


# In[27]:


order = list(feat_imps_xgb.groupby("feature").mean().sort_values("importance", ascending=False).index)
plt.figure(figsize=(10, 10))
sns.barplot(x="importance", y="feature", data=feat_imps_xgb, order=order)
plt.title("{} importance".format("xgb"))
plt.tight_layout()


# ### CatBoost

# In[28]:


oof_pred_cat, test_pred_cat, score_cat, feat_imps_cat = run_train_and_inference(
    X, X_test, y, "cat", MODEL_PARAMS["cat"], TRAIN_PARAMS["cat"],
    RANDOM_SEED_LIST, N_SPLITS,)  #cat_cols=list(range(10)))


# In[29]:


score_cat


# In[30]:


score_cat.loc[score_cat.fold == "oof"]


# In[31]:


order = list(feat_imps_cat.groupby("feature").mean().sort_values("importance", ascending=False).index)
plt.figure(figsize=(10, 10))
sns.barplot(x="importance", y="feature", data=feat_imps_cat, order=order)
plt.title("{} importance".format("cat"))
plt.tight_layout()


# ### Ensemble LGB, XGB, Cat

# ### check correlation

# In[32]:


model_names = ["lgb", "xgb", "cat"]


# In[33]:


# # prediction for oof
pd.DataFrame(
    np.corrcoef([
        oof_pred_lgb,
        oof_pred_xgb,
        oof_pred_cat
    ]),
    columns=model_names, index=model_names)


# In[34]:


# # prediction for test
pd.DataFrame(
    np.corrcoef([
        test_pred_lgb,
        test_pred_xgb,
        test_pred_cat
    ]),
    columns=model_names, index=model_names)


# ### simple averaging

# In[35]:


oof_pred_avg = (oof_pred_lgb + oof_pred_xgb + oof_pred_cat) / 3
oof_score_avg = mean_squared_error(y, oof_pred_avg, squared=False)

print(f"oof score avg: {oof_score_avg:.5f}")

test_pred_avg = (test_pred_lgb + test_pred_xgb + test_pred_cat) / 3


# ### weighted averaging

# In[36]:


weights = [0.5, 0.4, 0.1]

oof_pred_wavg = weights[0] * oof_pred_lgb + weights[1] * oof_pred_xgb + weights[2] * oof_pred_cat
oof_score_wavg = mean_squared_error(y, oof_pred_wavg, squared=False)

print(f"oof score weighted avg: {oof_score_wavg:.6f}")

test_pred_wavg = weights[0] * test_pred_lgb + weights[1] * test_pred_xgb + weights[2] * test_pred_cat


# ## Make submission

# In[37]:


sub = smpl_sub.copy()
# sub[TGT_COL] = test_pred_avg
sub[TGT_COL] = test_pred_wavg

sub.to_csv("submission.csv", index=False)

sub.head()


# In[ ]:




