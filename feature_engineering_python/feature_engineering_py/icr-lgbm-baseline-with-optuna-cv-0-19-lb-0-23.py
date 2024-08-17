#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# This is a LightGBM baseline notebook with Optuna parameter optimization. 
# I keep it in mind that the code is readable.
# 
# * LightGBM
# * Optuna Optimization
# * Stratified the dataset by Alpha
# * 5 fold CV
# 
# Thanks!
# 
# ### Update
# - Ver. 13
#     - Add reference to validatin dataset for LightGBM
#     - Add Class weight to dataset and remove oversampling
# - Ver. 12
#     - Add comments
#     - Split train data by Gamma
# - Ver. 11
#     - Oversample by Class to avoid overfitting
#     - Increase number of iterations
# - Ver. 10
#     - Remove some features
# - Ver. 9
#     - Fix metric
#     - Strip spaces in data columns
# - Ver. 8
#     - Fix oversampling issue
# - Ver. 7
#     - Random over sampling based on "Class"
#     - Remove unused library
# - Ver. 6
#     - Take Random over sampling into train dataset
#     - Make train dataset oversampled by Alpha
# - Ver. 5
#     - Some refactoring
#     - Add Debug flag
# - Ver. 4 (CV 0.24 -> 0.22)
#     - Add feature: Epsilon + 1
#     - Add data handling: oversampling
#     - Add postprocess: set class 0 if BQ is None
#     - Some refactoring
# - Ver. 3 (CV 0.25 -> 0.24)
#     - Add feature: the number of NaN per record
#     - Add submission csv file saving method
# - Ver. 2
#     - Add score in CV and LB to notebook title
#     - Change LGBM parameter name
# - Ver. 1 Published

# # Libraries

# In[1]:


import os
import json
from pathlib import Path
from pprint import pprint
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

import optuna.integration.lightgbm as lgb


# # Configuration

# In[2]:


@dataclass(frozen=True)
class Config:
    n_folds: int = 8
    random_seed: int = 42


# In[3]:


DEBUG = False

# Inputs
BASE_PATH = Path("/kaggle/input/icr-identify-age-related-conditions")

TRAIN = BASE_PATH / "train.csv"
GREEKS = BASE_PATH / "greeks.csv"

TEST = BASE_PATH / "test.csv"

# Outputs
OUTPUT_PATH = Path("/kaggle/working")


# # Model Training

# ## Train Data

# In[4]:


class TrainDataSet:
    
    def __init__(self, df: pd.DataFrame):
        from imblearn.over_sampling import RandomOverSampler
        
        self.non_feat_cols = [
            'Id', # ID
            'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', # Greeks
            'fold', 'oof', # Artifacts
            'DV', 'BZ', 'CL', 'CH', 'EL', 'EP', 'EG' # Unimportant
        ]
        self.target_col = ["Class"]
        self.df = df
        self.df.columns = self.df.columns.str.strip()
        self.ros = RandomOverSampler(random_state=Config.random_seed)
        
        self._add_fold_col()
        self.df["EJ"] = self.df.EJ.map({"A":0, "B":1}).astype('int')
        
    def get_feature(self, fold:int, feature_name: str, phase: str='train') -> pd.Series:
        return self.df.loc[~self._is_val_target(fold), feature_name] if phase == 'train' else self.df.loc[self._is_val_target(fold), feature_name] 
    
    def _add_fold_col(self) -> None:
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=Config.n_folds, random_state=Config.random_seed, shuffle=True)

        self.df["fold"] = np.nan
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.df, self.df["Gamma"])):
            self.df.loc[val_idx, "fold"] = fold
            
    def _is_val_target(self, fold: int) -> pd.Series:
        return self.df.fold == fold
    
    def add_oof_preds(self, fold: int, oof_preds: np.array) -> None:
        self.df.loc[self._is_val_target(fold), "oof"] = oof_preds
    
    def targets(self, fold: int) -> (pd.DataFrame, pd.DataFrame):
        return self.df.loc[~self._is_val_target(fold), self.target_col], self.df.loc[self._is_val_target(fold), self.target_col]
    
    def inputs(self, fold: int) -> (pd.DataFrame, pd.DataFrame):
        input_cols = [f for f in self.df.columns if not f in self.target_col + self.non_feat_cols]
        return self.df.loc[~self._is_val_target(fold), input_cols], self.df.loc[self._is_val_target(fold), input_cols]

    def train_dataset_oversampled(self, fold: int, feature_name: str='Class') -> (pd.DataFrame, pd.DataFrame):
        oversampled_data, _ = self.ros.fit_resample(self.df.loc[~self._is_val_target(fold)], self.df.loc[~self._is_val_target(fold), feature_name])
        oversampled_df = pd.DataFrame(oversampled_data)
        input_cols = [f for f in oversampled_df.columns if not f in self.target_col + self.non_feat_cols]
        return oversampled_df.loc[:, input_cols], oversampled_df.loc[:, self.target_col]


# ## Metric

# In[5]:


from sklearn.metrics import log_loss

def balance_logloss(y_true, y_pred):
    nc = np.bincount(y_true)
    return log_loss(y_true, y_pred, sample_weight = 1/nc[y_true], eps=1e-15)


# ## Model Parameter

# In[6]:


@dataclass
class ParamsLGBM:
    objective: str = "binary"
    metric: str = "binary_logloss"
    verbosity: int = -1
    is_unbalance: bool = True
    boosting_type: str = "gbdt"
    learning_rate: float = 0.01
    max_depth: int = 3
    subsample: float = 0.8
    colsample_bytree: float = 0.63
    lambda_l2: float = 6.6
    
    def dict(self) -> dict:
        return asdict(self)


# ## Feature Engineering Functions

# In[7]:


def add_n_NaN(df: pd.DataFrame) -> pd.DataFrame:
    df["n_NaN"] = df.isnull().sum(axis=1)
    return df


# In[8]:


def generate_epsilon_as_feature() -> pd.DataFrame:
    """
    generate ordinal days (Epsilon ordinal days)
    """
    import datetime
    
    train_df = pd.read_csv(TRAIN)
    greeks_df = pd.read_csv(GREEKS)
    train_df = train_df.merge(greeks_df, on='Id')
    
    test_df = pd.read_csv(TEST)
    
    unknown = 'Unknown'
    epsilon = 'Epsilon'
    fe_name = 'ordinal'
    
    # train dataset
    is_unknown = train_df.Epsilon == unknown
    train_df.loc[~is_unknown, fe_name] = train_df.loc[~is_unknown, epsilon].map(lambda x: datetime.datetime.strptime(x,'%m/%d/%Y').toordinal())
    train_df.loc[is_unknown, fe_name] = np.nan
    train_df[fe_name] = train_df[fe_name].astype('float')
    
    # test dataset
    max_epsilon = train_df[fe_name].max()
    test_df[fe_name] = max_epsilon + 1
    test_df[fe_name] = test_df[fe_name].astype('float')
    
    df = pd.concat([train_df.loc[:, ['Id', fe_name]], test_df.loc[:, ['Id', fe_name]]], axis=0)
    return df


# ## Postprocess

# In[9]:


def set_class_0_if_BQ_is_None(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df.BQ.isnull(), "Class"] = 1e-15
    return df


# ## Training Pipeline

# In[10]:


train_df = pd.read_csv(TRAIN)
greeks_df = pd.read_csv(GREEKS)

train_df = train_df.merge(greeks_df, on=["Id"])


# In[11]:


# feature engineering
train_df = add_n_NaN(train_df)
train_df = train_df.merge(generate_epsilon_as_feature(), on='Id')


# In[12]:


train_ds = TrainDataSet(train_df)


# In[13]:


for fold in range(Config.n_folds):
    train_input, val_input = train_ds.inputs(fold)
    train_target, val_target = train_ds.targets(fold)
    
#     train_input, train_target = train_ds.train_dataset_oversampled(fold, 'Class')
    
    N0, N1 = np.bincount(train_df.Class)
    train_weight = train_target['Class'].map({0: 1/N0, 1: 1/N1})
    val_weight = val_target['Class'].map({0: 1/N0, 1: 1/N1})
    
    train_ds_lgb = lgb.Dataset(train_input, train_target, weight=train_weight)
    val_ds_lgb = lgb.Dataset(val_input, val_target, weight=val_weight, reference=train_ds_lgb)

    model = lgb.train(
        ParamsLGBM().dict(),
        train_ds_lgb, 
        num_boost_round=100000 if not DEBUG else 1,
        valid_sets=[train_ds_lgb, val_ds_lgb], 
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(400),
        ]
    )
    model.save_model(f'lgbm_fold_{fold}.txt', num_iteration=model.best_iteration)
    preds = model.predict(val_input)
    
    train_ds.add_oof_preds(fold, preds)
    
    print(f'Best Params in fold {fold}:')
    pprint(model.params)
    
    with open(f'params_lgbm_fold_{fold}.json', 'w') as f:
        json.dump(model.params, f, ensure_ascii=False, indent=4)


# In[14]:


oof_score = balance_logloss(train_df["Class"], train_df.loc[:, "oof"])
print(f'Out of fold score: {oof_score}')


# In[15]:


# Postprocess: BQ is None -> Class is 0
df = train_df.copy()
df.loc[df.BQ.isnull(), 'oof'] = 0

oof_score = balance_logloss(df["Class"], df.loc[:, "oof"])
print(f'Out of fold score: {oof_score}')


# # Inference

# ## Test Data

# In[16]:


class TestDataSet:
    
    def __init__(self, df: pd.DataFrame):
        self.non_feat_cols = [
            'Id',  # ID
            'DV', 'BZ', 'CL', 'CH', 'EL', 'EP', 'EG' # Unimportant
        ]
        self.target_col = ["Class"]
        self.df = df
        self.df.columns = self.df.columns.str.strip()
        
        self.df["Class"] = 0
        self.df["EJ"] = self.df.EJ.map({"A":0, "B":1}).astype('int')
    
    def inputs(self) -> pd.DataFrame:
        input_cols = [f for f in self.df.columns if not f in self.target_col + self.non_feat_cols]
        return self.df.loc[:, input_cols]
    
    def add_preds(self, preds: np.ndarray) -> None:
        self.df.loc[:, self.target_col] = preds
        
    def get_submission_df(self) -> pd.DataFrame:
        # Postprocess
        self.df = set_class_0_if_BQ_is_None(self.df)
        
        sub_df = self.df.loc[:, ["Id", "Class"]]
        sub_df["class_1"] = sub_df["Class"]
        sub_df["class_0"] = 1 - sub_df["Class"]
        return sub_df.loc[:, ["Id", "class_0", "class_1"]]
    
    def save_submission_csv(self) -> None:
        sub_df = self.get_submission_df()
        sub_df.to_csv('submission.csv', index=False)


# ## Inference Pipeline

# In[17]:


test_df = pd.read_csv(TEST)

# feature engineering
test_df = add_n_NaN(test_df)
test_df = test_df.merge(generate_epsilon_as_feature(), on='Id')

test_ds = TestDataSet(test_df)

preds = np.zeros(len(test_df))
for fold in range(Config.n_folds):
    model_path = OUTPUT_PATH / f"lgbm_fold_{fold}.txt"
    
    model = lgb.Booster(model_file=model_path)
    preds += model.predict(test_ds.inputs())
    
test_ds.add_preds(preds / Config.n_folds)


# In[18]:


test_ds.df.head()


# ## Submission

# In[19]:


sub_df = test_ds.get_submission_df()
sub_df.head()


# In[20]:


test_ds.save_submission_csv()


# In[ ]:




