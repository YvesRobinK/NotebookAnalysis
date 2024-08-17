#!/usr/bin/env python
# coding: utf-8

# _____
# **Credits:**
# - This is an optimized version of the amazing [notebook](https://www.kaggle.com/code/ragnar123/amex-lgbm-dart-cv-0-7977) by [ragnar123](https://www.kaggle.com/ragnar123)
# _____
# 
# 
# # The Fine Art of Hyperparameter Tuning
# 
# In this notebook, I go through the process of hyperparameter tuning in a structured way, using some best practices I have picked up along the way.
# 
# ### The objective function
# When tuning the parameters of a model, we basically want to find the combination of parameters that minimize a predefined loss function. For simplicity, we are going to maximize the competition metric.
# 
# **We should define the optimization metric as early as possible since the whole process revolves around it.**
# 
# It is essential to keep the scoring metric consistent throughout the whole process, otherwise, the results may be misleading. This means that the metric used for cross-validation should be the same on the leaderboard. 
# 
# ### Compute Time
# Since this is a competition and we want to be on the leaderboard as soon as possible, our optimization function should not take too long to run.
# 
# ### Getting a basic model first
# In order to tune the hyperparameters, we need to set up a basic model that has all the hyperparameters we want to tune. 
# 
# This is to ensure we will not miss out on any critical parameters.
# 

# ### Feature Engineering
# **Credit:** This amazing [notebook](https://www.kaggle.com/code/ragnar123/amex-lgbm-dart-cv-0-7977) by [Martin Kovacevic Buvinic](https://www.kaggle.com/ragnar123)

# In[1]:


import gc
import itertools
import scipy as sp
import numpy as np
import pandas as pd
from tqdm import tqdm
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
import warnings; warnings.filterwarnings('ignore')

def get_difference(data, num_features):
    df1 = []
    customer_ids = []
    for customer_id, df in tqdm(data.groupby(['customer_ID'])):
        diff_df1 = df[num_features].diff(1).iloc[[-1]].values.astype(np.float32)
        df1.append(diff_df1)
        customer_ids.append(customer_id)
    df1 = np.concatenate(df1, axis = 0)
    df1 = pd.DataFrame(df1, columns = [col + '_diff1' for col in df[num_features].columns])
    df1['customer_ID'] = customer_ids
    return df1

def read_preprocess_data():
    train = pd.read_parquet('/content/data/train.parquet')
    features = train.drop(['customer_ID', 'S_2'], axis = 1).columns.to_list()
    cat_features = [
        "B_30",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_66",
        "D_68",
    ]
    num_features = [col for col in features if col not in cat_features]
    print('Starting training feature engineer...')
    train_num_agg = train.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last'])
    train_num_agg.columns = ['_'.join(x) for x in train_num_agg.columns]
    train_num_agg.reset_index(inplace = True)
    train_cat_agg = train.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
    train_cat_agg.columns = ['_'.join(x) for x in train_cat_agg.columns]
    train_cat_agg.reset_index(inplace = True)
    train_labels = pd.read_csv('/content/data/train_labels.csv')
    cols = list(train_num_agg.dtypes[train_num_agg.dtypes == 'float64'].index)
    for col in tqdm(cols):
        train_num_agg[col] = train_num_agg[col].astype(np.float32)
    cols = list(train_cat_agg.dtypes[train_cat_agg.dtypes == 'int64'].index)
    for col in tqdm(cols):
        train_cat_agg[col] = train_cat_agg[col].astype(np.int32)
    train_diff = get_difference(train, num_features)
    train = train_num_agg.merge(train_cat_agg, how = 'inner', on = 'customer_ID').merge(train_diff, how = 'inner', on = 'customer_ID').merge(train_labels, how = 'inner', on = 'customer_ID')
    del train_num_agg, train_cat_agg, train_diff
    gc.collect()
    test = pd.read_parquet('/content/data/test.parquet')
    print('Starting test feature engineer...')
    test_num_agg = test.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last'])
    test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]
    test_num_agg.reset_index(inplace = True)
    test_cat_agg = test.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
    test_cat_agg.columns = ['_'.join(x) for x in test_cat_agg.columns]
    test_cat_agg.reset_index(inplace = True)
    cols = list(test_num_agg.dtypes[test_num_agg.dtypes == 'float64'].index)
    for col in tqdm(cols):
        test_num_agg[col] = test_num_agg[col].astype(np.float32)
    cols = list(test_cat_agg.dtypes[test_cat_agg.dtypes == 'int64'].index)
    for col in tqdm(cols):
        test_cat_agg[col] = test_cat_agg[col].astype(np.int32)
    test_diff = get_difference(test, num_features)
    test = test_num_agg.merge(test_cat_agg, how = 'inner', on = 'customer_ID').merge(test_diff, how = 'inner', on = 'customer_ID')
    del test_num_agg, test_cat_agg, test_diff
    gc.collect()
    train.to_parquet('/content/drive/MyDrive/Amex/train_fe.parquet')
    test.to_parquet('/content/drive/MyDrive/Amex/test_fe.parquet')

# Read & Preprocess Data
# read_preprocess_data()


# ### Configurations & Setup

# In[2]:


import os
import gc
import random
import joblib
import itertools
import scipy as sp
import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
from itertools import combinations
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
from sklearn.preprocessing import LabelEncoder
import warnings; warnings.filterwarnings('ignore')
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import StratifiedKFold, train_test_split

class CFG:
    seed = 42
    n_folds = 5
    target = 'target'
    boosting_type = 'dart'
    metric = 'binary_logloss'
    input_dir = '/content/data/'
    
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def read_data():
    train = pd.read_parquet(CFG.input_dir + 'train_fe.parquet')
    test = pd.read_parquet(CFG.input_dir + 'test_fe.parquet')
    return train, test


# ### The Competition's Metric
# 
# The evaluation metric, *M* for this competition is the mean of two measures of rank ordering: Normalized Gini Coefficient, *G*, and default rate captured at 4%, *D*.
# 
# $$M = 0.5 \cdot ( G + D )$$
# 
# The default rate captured at 4% is the percentage of the positive labels (defaults) captured within the highest-ranked 4% of the predictions, and represents a Sensitivity/Recall statistic.
# 
# For both of the sub-metrics *G* and *D*, the negative labels are given a weight of 20 to adjust for downsampling.
# 
# This metric has a maximum value of 1.0.

# In[3]:


def amex_metric(y_true, y_pred):
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:,0]==0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])
    gini = [0,0]
    for i in [1,0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:,0]==0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] *  weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)
    return 0.5 * (gini[1]/gini[0] + top_four)


# ### Model Specific Metrics
# 
# Now that we have a stable numpy implementation of the metric, we need to create a version for each of the models that we are going to use. 
# We do this in order for us to be able to early stop each of our models (to avoid overfitting).

# #### LGBM Amex Metric

# In[4]:


def lgb_amex_metric(y_pred, y_true):
    y_true = y_true.get_label()
    return 'amex_metric', amex_metric(y_true, y_pred), True


# #### Catboost Amex Metric

# In[5]:


class AmexCatboostMetric(object):
   def get_final_error(self, error, weight): return error
   def is_max_optimal(self): return True
   def evaluate(self, approxes, target, weight): return amex_metric(np.array(target), approxes[0]), 1.0


# #### XGBoost Amex Metric

# In[6]:


def xgb_amex(y_pred, y_true):
    return 'amex', amex_metric_np(y_pred,y_true.get_label())


# ### The Actual Optimization
# ____
# 
# Now all that is left for us to do is optimize each of our models as much as we can then ensemble and submit them to the leaderboard.

# ### LightGBM
# 
# The following are the hyperparameters we are going to optimize.
# 
# #### Hyperparameters of interest
#                 
# - **`subsample`:**
#     -  Randomly select part of data (without resampling)
# - **`reg_alpha`:**
#     - L1 regularization term on weights
# - **`lambda_l1`:**
#     - L1 regularization term on weights
# - **`lambda_l2`:**
#     - L2 regularization term on weights
# - **`max_depth`:**
#     - Maximum tree depth for base learners
# - **`reg_lambda`:**
#     - L2 regularization term on weights
# - **`num_leaves`:**
#     - Maximum tree leaves for base learners
# - **`min_split_gain`:**
#     - Minimum loss reduction required to make a further partition on a leaf node of the tree
# - **`learning_rate`:**
#     - Boosting learning rate
# - **`min_child_weight`:**
#     - Minimum sum of instance weight (hessian) needed in a child (leaf)
# - **`feature_fraction`:**
#     - Randomly select part of features
# - **`bagging_fraction`:**
#     - Randomly bag or subsample training data
# - **`colsample_bytree`:**
#     - Subsample ratio of columns when constructing each tree
# - **`boosting_type`:**
#     - gbdt (traditional Gradient Boosting Decision Tree)
#     - dart (Dropouts meet Multiple Additive Regression Trees)
# - **`min_data_in_leaf`:**
#     - Minimum number of data needed in a leaf
# 

# #### Hyperparameters Optimization

# In[7]:


lgb_space = {
                'objective': 'binary',
                'subsample': hp.uniform('subsample', 0.5, 1.0),
                'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
                'lambda_l1': hp.uniform('lambda_l1', 0.0, 10.0),
                'lambda_l2': hp.uniform('lambda_l2', 0.0, 10.0),
                'max_depth': hp.quniform("max_depth", 2, 16, 1),
                'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
                'num_leaves': hp.quniform('num_leaves', 30, 200, 1),
                'min_split_gain': hp.uniform('min_split_gain', 0.0, 1.0),
                'learning_rate': hp.loguniform('learning_rate', -5.0, -2.3),
                'min_child_weight': hp.uniform('min_child_weight', 0.5, 10),
                'feature_fraction': hp.uniform('feature_fraction', 0.4, 1.0),
                'bagging_fraction': hp.uniform('bagging_fraction', 0.4, 1.0),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
                'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
                'min_data_in_leaf': hp.quniform('min_data_in_leaf', 20, 500, 1)
            }

def get_lgb_hyper_search(x, y, xt, yt):
    def obj(space):
        print("# of features:", x.shape[1])
        dtrain = lgb.Dataset(x, label=y)
        dvalid = lgb.Dataset(xt, label=yt)
        params = {
                    'n_jobs': -1,
                    'lambda_l2': 2,
                    'seed': CFG.seed,
                    'num_leaves': 100,
                    'bagging_freq': 10,
                    'metric': CFG.metric,
                    'objective': 'binary',
                    'learning_rate': 0.01,        
                    'min_data_in_leaf': 40,
                    'bagging_fraction': 0.50,
                    'feature_fraction': 0.20,
                    'boosting': CFG.boosting_type,
                    
                 }
        params_to_update = dict(
                                    num_leaves = int(space['num_leaves']),
                                    learning_rate = space['learning_rate'],
                                    feature_fraction = space['feature_fraction'],
                                    bagging_fraction = space['bagging_fraction'],
                                    max_depth = int(space['max_depth']),
                                    reg_alpha = space['reg_alpha'],
                                    reg_lambda = space['reg_lambda'],
                                    min_split_gain = space['min_split_gain'],
                                    min_child_weight = space['min_child_weight'],
                                    colsample_bytree = space['colsample_bytree'],
                                    subsample = space['subsample'],
                                    lambda_l1 = space['lambda_l1'],
                                    lambda_l2 = space['lambda_l2'],
                                    min_data_in_leaf = int(space['min_data_in_leaf'])
                               )
        params.update(params_to_update)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        bst = lgb.train(params, dtrain=dtrain,
                    num_boost_round=20500,evals=watchlist,
                    early_stopping_rounds=1500, feval=lgb_amex, maximize=True,
                    verbose_eval=500)
        print('best ntree_limit:', bst.best_iteration)
        print('best score:', bst.best_score)
        return {'loss': -bst.best_score['eval']['auc'], 'status': STATUS_OK}
    return obj


# ### Training

# In[8]:


def train_and_evaluate(train, test):
    cat_features = [
        "B_30",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_66",
        "D_68"
    ]
    cat_features = [f"{cf}_last" for cf in cat_features]
    for cat_col in cat_features:
        encoder = LabelEncoder()
        train[cat_col] = encoder.fit_transform(train[cat_col])
        test[cat_col] = encoder.transform(test[cat_col])
    num_cols = list(train.dtypes[(train.dtypes == 'float32') | (train.dtypes == 'float64')].index)
    num_cols = [col for col in num_cols if 'last' in col]
    for col in num_cols:
        train[col + '_round2'] = train[col].round(2)
        test[col + '_round2'] = test[col].round(2)
    num_cols = [col for col in train.columns if 'last' in col]
    num_cols = [col[:-5] for col in num_cols if 'round' not in col]
    for col in num_cols:
        try:
            train[f'{col}_last_mean_diff'] = train[f'{col}_last'] - train[f'{col}_mean']
            test[f'{col}_last_mean_diff'] = test[f'{col}_last'] - test[f'{col}_mean']
        except: pass
    num_cols = list(train.dtypes[(train.dtypes == 'float32') | (train.dtypes == 'float64')].index)
    for col in tqdm(num_cols):
        train[col] = train[col].astype(np.float16)
        test[col] = test[col].astype(np.float16)
    features = [col for col in train.columns if col not in ['customer_ID', CFG.target]]
    params = {
                    'n_jobs': -1,
                    'lambda_l2': 2,
                    'seed': CFG.seed,
                    'num_leaves': 100,
                    'bagging_freq': 10,
                    'metric': CFG.metric,
                    'objective': 'binary',
                    'learning_rate': 0.01,        
                    'min_data_in_leaf': 40,
                    'bagging_fraction': 0.50,
                    'feature_fraction': 0.20,
                    'boosting': CFG.boosting_type,
             }
    test_predictions = np.zeros(len(test))
    oof_predictions = np.zeros(len(train))
    kfold = StratifiedKFold(n_splits = CFG.n_folds, shuffle = True, random_state = CFG.seed)
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(train, train[CFG.target])):
        print(' ')
        print('-'*50)
        print(f'Training fold {fold} with {len(features)} features...')
        x_train, x_val = train[features].iloc[trn_ind], train[features].iloc[val_ind]
        y_train, y_val = train[CFG.target].iloc[trn_ind], train[CFG.target].iloc[val_ind]
        lgb_train = lgb.Dataset(x_train, y_train, categorical_feature = cat_features)
        lgb_valid = lgb.Dataset(x_val, y_val, categorical_feature = cat_features)
        
        # hyperparameter tuning
        obj = get_lgb_hyper_search(x_train, y_train, x_val, y_val)
        trials = Trials()
        lgb_best_params = fmin(fn = obj, space = space, algo = tpe.suggest, max_evals = 1000, trials = trials)
        
        params.update(lgb_best_params)
        model = lgb.train(
            params = params,
            train_set = lgb_train,
            num_boost_round = 20500,
            valid_sets = [lgb_train, lgb_valid],
            early_stopping_rounds = 1500,
            verbose_eval = 500,
            feval = lgb_amex_metric
            )
        joblib.dump(model, f'/content/drive/MyDrive/Amex/Models/lgbm_{CFG.boosting_type}_fold{fold}_seed{CFG.seed}.pkl')
        val_pred = model.predict(x_val)
        oof_predictions[val_ind] = val_pred
        test_pred = model.predict(test[features])
        test_predictions += test_pred / CFG.n_folds
        score = amex_metric(y_val, val_pred)
        print(f'Our fold {fold} CV score is {score}')
        del x_train, x_val, y_train, y_val, lgb_train, lgb_valid
        gc.collect()
    score = amex_metric(train[CFG.target], oof_predictions)
    print(f'Our out of folds CV score is {score}')
    oof_df = pd.DataFrame({'customer_ID': train['customer_ID'], 'target': train[CFG.target], 'prediction': oof_predictions})
    oof_df.to_csv(f'/content/drive/MyDrive/Amex/OOF/oof_lgbm_{CFG.boosting_type}_baseline_{CFG.n_folds}fold_seed{CFG.seed}.csv', index = False)
    test_df = pd.DataFrame({'customer_ID': test['customer_ID'], 'prediction': test_predictions})
    test_df.to_csv(f'/content/drive/MyDrive/Amex/Predictions/test_lgbm_{CFG.boosting_type}_baseline_{CFG.n_folds}fold_seed{CFG.seed}.csv', index = False)
    
# seed_everything(CFG.seed)
# train, test = read_data()
# train_and_evaluate(train, test)


# ### XGBoost
# 
# The following are the hyperparameters we are going to optimize.
# 
# #### Hyperparameters of interest
#                 
# - **`gamma`:**
#     - Minimum loss reduction required to make a further partition on a leaf node of the tree
# - **`eta`:**
#     - Step size shrinkage used in update to prevents overfitting
# - **`lambda`:**
#     - L2 regularization term on weights
# - **`reg_lambda`:**
#     - L2 regularization term on weights
# - **`subsample`:**
#     - Subsample ratio of the training instance
# - **`rate_drop`:**
#     - Dropout rate (a fraction of previous trees to drop during the dropout)
# - **`skip_drop`:**
#     - Probability of skipping the dropout procedure during a boosting iteration
# - **`reg_alpha`:**
#     - L1 regularization term on weights
# - **`max_depth`:**
#     - Maximum depth of a tree
# - **`colsample_bytree`:**
#     - Subsample ratio of columns when constructing each tree
# - **`min_child_weight`:**
#     - Minimum sum of instance weight (hessian) needed in a child (leaf)
# - **`sample_type`:**
#     - Type of sampling algorithm
# - **`normalize_type`:**
#     - Type of normalization algorithm
# - **`grow_policy`:**
#     - Split policy for all trees
# 

# #### Hyperparameters Optimization

# In[9]:


space = {            
            'seed': 0,
            'gamma': hp.uniform ('gamma', 1,9),
            'eta': hp.uniform('eta', 0.01, 0.1),
            'lambda': hp.uniform('lambda', 0.0, 100),            
            'reg_lambda': hp.uniform('reg_lambda', 0,1),
            'subsample': hp.uniform('subsample', 0.6, 1),
            'rate_drop': hp.uniform('rate_drop', 0.0, 1.0),                
            'skip_drop': hp.uniform('skip_drop', 0.0, 1.0),        
            'reg_alpha': hp.quniform('reg_alpha', 40,180,1),
            'max_depth': hp.quniform("max_depth", 3, 18, 1),    
            'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
            'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),            
            'sample_type': hp.choice('sample_type', ['uniform', 'weighted']),        
            'normalize_type': hp.choice('normalize_type', ['tree', 'forest']),        
            'grow_policy': hp.choice('grow_policy', ['depthwise', 'lossguide']),            
        }

def get_xgb_hyper_search(x, y, xt, yt):
    def obj(space):
        print("# of features:", x.shape[1])
        dtrain = xgb.DMatrix(data=x, label=y)
        dvalid = xgb.DMatrix(data=xt, label=yt)

        params_to_update = dict(
                                    eta = space['eta'],
                                    gamma = space['gamma'],
                                    subsample = space['subsample'],                     
                                    max_depth = int(space['max_depth']),                    
                                    reg_alpha = int(space['reg_alpha']),
                                    min_child_weight = int(space['min_child_weight']),
                                    colsample_bytree = int(space['colsample_bytree'])
                               )
        
        params_to_update['lambda'] = space['lambda']
        params_to_update['reg_lambda'] = space['reg_lambda']

        params = {
                    'eta': 0.03,
                    'gamma': 1.5,
                    'lambda': 70,
                    'max_depth': 7,
                    'subsample': 0.88,
                    'min_child_weight': 8,
                    'colsample_bytree': 0.5,
                    'tree_method': 'gpu_hist',
                    'objective': 'binary:logistic',
                 }
        
        params.update(params_to_update)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        bst = xgb.train(params, dtrain=dtrain,
                    num_boost_round=20500,evals=watchlist,
                    early_stopping_rounds=1500, feval=xgb_amex, maximize=True,
                    verbose_eval=500)
        print('best ntree_limit:', bst.best_ntree_limit)
        print('best score:', bst.best_score)
        return {'loss': -bst.best_score, 'status': STATUS_OK}
    return obj


# ### Training

# In[10]:


def train_and_evaluate(train, test):
    cat_features = [
        "B_30",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_66",
        "D_68"
    ]
    cat_features = [f"{cf}_last" for cf in cat_features]
    for cat_col in cat_features:
        encoder = LabelEncoder()
        train[cat_col] = encoder.fit_transform(train[cat_col])
        test[cat_col] = encoder.transform(test[cat_col])
    num_cols = list(train.dtypes[(train.dtypes == 'float32') | (train.dtypes == 'float64')].index)
    num_cols = [col for col in num_cols if 'last' in col]
    for col in num_cols:
        train[col + '_round2'] = train[col].round(2)
        test[col + '_round2'] = test[col].round(2)
    num_cols = [col for col in train.columns if 'last' in col]
    num_cols = [col[:-5] for col in num_cols if 'round' not in col]
    for col in num_cols:
        try:
            train[f'{col}_last_mean_diff'] = train[f'{col}_last'] - train[f'{col}_mean']
            test[f'{col}_last_mean_diff'] = test[f'{col}_last'] - test[f'{col}_mean']
        except: pass
    num_cols = list(train.dtypes[(train.dtypes == 'float32') | (train.dtypes == 'float64')].index)
    for col in tqdm(num_cols):
        train[col] = train[col].astype(np.float16)
        test[col] = test[col].astype(np.float16)
    features = [col for col in train.columns if col not in ['customer_ID', CFG.target]]
    
    params = {
                'eta': 0.03,
                'gamma': 1.5,
                'lambda': 70,
                'max_depth': 7,
                'subsample': 0.88,
                'min_child_weight': 8,
                'colsample_bytree': 0.5,
                'tree_method': 'gpu_hist',
                'objective': 'binary:logistic',
             }

    test_predictions = np.zeros(len(test))
    oof_predictions = np.zeros(len(train))
    kfold = StratifiedKFold(n_splits = CFG.n_folds, shuffle = True, random_state = CFG.seed)
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(train, train[CFG.target])):
        print(' ')
        print('-'*50)
        print(f'Training fold {fold} with {len(features)} features...')
        x_train, x_val = train[features].iloc[trn_ind], train[features].iloc[val_ind]
        y_train, y_val = train[CFG.target].iloc[trn_ind], train[CFG.target].iloc[val_ind]
        dtrain = xgb.DMatrix(data = x_train, label = y_train)
        dvalid = xgb.DMatrix(data = x_val, label = y_val)
        
        # hyperparameter tuning
        obj = get_xgb_hyper_search(x_train, y_train, x_val, y_val)
        trials = Trials()
        xgb_best_params = fmin(fn = obj, space = space, algo = tpe.suggest, max_evals = 1000, trials = trials)
        params.update(xgb_best_params)
        
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        model = xgb.train(params, dtrain = dtrain, num_boost_round = 20500, evals = watchlist, early_stopping_rounds = 1500, feval = xgb_amex, maximize = True, verbose_eval = 500)
        print('best ntree_limit:', model.best_ntree_limit)
        print('best score:', model.best_score)        
        # Save best model
        model.save_model(f'/content/drive/MyDrive/Amex/Models/xgb_{CFG.boosting_type}_fold{fold}_seed{CFG.seed}.pkl')
        # Predict validation        
        val_pred = model.predict(xgb.DMatrix(x_val), iteration_range=(0, model.best_ntree_limit))
        # Add to out of folds array
        oof_predictions[val_ind] = val_pred
        # Predict the test set        
        test_pred = model.predict(xgb.DMatrix(test[features]), iteration_range = (0, model.best_ntree_limit))
        test_predictions += test_pred / CFG.n_folds
        # Compute fold metric
        score = amex_metric(y_val, val_pred)
        print(f'Our fold {fold} CV score is {score}')
        del x_train, x_val, y_train, y_val
        gc.collect()
    # Compute out of folds metric
    score = amex_metric(train[CFG.target], oof_predictions)
    print(f'Our out of folds CV score is {score}')
    oof_df = pd.DataFrame({'customer_ID': train['customer_ID'], 'target': train[CFG.target], 'prediction': oof_predictions})
    oof_df.to_csv(f'/content/drive/MyDrive/Amex/OOF/oof_xgb_{CFG.boosting_type}_baseline_{CFG.n_folds}fold_seed{CFG.seed}.csv', index = False)
    test_df = pd.DataFrame({'customer_ID': test['customer_ID'], 'prediction': test_predictions})
    test_df.to_csv(f'/content/drive/MyDrive/Amex/Predictions/test_xgb_{CFG.boosting_type}_baseline_{CFG.n_folds}fold_seed{CFG.seed}.csv', index = False)
    
# seed_everything(CFG.seed)
# train, test = read_data()
# train_and_evaluate(train, test)


# ### CatBoost
# 
# The following are the hyperparameters we are going to optimize.
# 
# #### Hyperparameters of interest
# 
# - **`max_depth`:**
#     - Maximum depth of a tree
# - **`l2_leaf_reg`:**
#     - L2 regularization term on weights
# - **`random_strength`:**
#     - Random strength
# - **`colsample_bytree`:**
#     - Subsample ratio of columns when constructing each tree
# - **`min_child_weight`:**
#     - Minimum sum of instance weight (hessian) needed in a child (leaf)
# - **`bagging_temperature`:**
#     - Controls intensity of Bayesian bagging
# 

# #### Hyperparameters Optimization

# In[11]:


space = {
            'max_depth': hp.quniform("max_depth", 3, 16, 1),
            'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0,100),
            'random_strength': hp.uniform ('random_strength', 0, 1),
            'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
            'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
            'bagging_temperature': hp.uniform('bagging_temperature', 0,100),
        }

def get_catboost_hyper_search(x, y, xt, yt):
    def obj(space):
        print("# of features:", x.shape[1])
        dtrain = Pool(data=x, label=y)
        dvalid = Pool(data=xt, label=yt)
        
        params = {
                    'max_depth': 7,
                    'od_type': 'Iter',
                    'l2_leaf_reg': 70,
                    'random_seed': 42,
                    'iterations': 20500,
                    'learning_rate': 0.03,
                    'loss_function': 'Logloss',
                    'early_stopping_rounds': 1500
                 }
        
        params_to_update = dict(max_depth = int(space['max_depth']), bagging_temperature = space['bagging_temperature'],  random_strength = space['random_strength'], l2_leaf_reg = space['l2_leaf_reg'])
        params.update(params_to_update)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        bst = CatBoostClassifier(**params, eval_metric = AmexCatboostMetric())
        bst.fit(dtrain, eval_set=dvalid, use_best_model=True, verbose=500)
        print('best score:', bst.best_score_)
        return {'loss': -bst.best_score_['validation']['AUC'], 'status': STATUS_OK}
    return obj


# ### Training

# In[12]:


def train_and_evaluate(train, test):
    cat_features = [
        "B_30",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_66",
        "D_68"
    ]
    cat_features = [f"{cf}_last" for cf in cat_features]
    for cat_col in cat_features:
        encoder = LabelEncoder()
        train[cat_col] = encoder.fit_transform(train[cat_col])
        test[cat_col] = encoder.transform(test[cat_col])
    num_cols = list(train.dtypes[(train.dtypes == 'float32') | (train.dtypes == 'float64')].index)
    num_cols = [col for col in num_cols if 'last' in col]
    for col in num_cols:
        train[col + '_round2'] = train[col].round(2)
        test[col + '_round2'] = test[col].round(2)
    num_cols = [col for col in train.columns if 'last' in col]
    num_cols = [col[:-5] for col in num_cols if 'round' not in col]
    for col in num_cols:
        try:
            train[f'{col}_last_mean_diff'] = train[f'{col}_last'] - train[f'{col}_mean']
            test[f'{col}_last_mean_diff'] = test[f'{col}_last'] - test[f'{col}_mean']
        except: pass
    num_cols = list(train.dtypes[(train.dtypes == 'float32') | (train.dtypes == 'float64')].index)
    for col in tqdm(num_cols):
        train[col] = train[col].astype(np.float16)
        test[col] = test[col].astype(np.float16)
    features = [col for col in train.columns if col not in ['customer_ID', CFG.target]]
    
    params = {
                'max_depth': 7,    
                'od_type': 'Iter',        
                'l2_leaf_reg': 70,    
                'random_seed': 42,                    
                'iterations': 20500,    
                'learning_rate': 0.03,
                'loss_function': 'Logloss',                                                
                'early_stopping_rounds': 1500
             }

    test_predictions = np.zeros(len(test))
    oof_predictions = np.zeros(len(train))
    kfold = StratifiedKFold(n_splits = CFG.n_folds, shuffle = True, random_state = CFG.seed)
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(train, train[CFG.target])):
        print(' ')
        print('-'*50)
        print(f'Training fold {fold} with {len(features)} features...')
        x_train, x_val = train[features].iloc[trn_ind], train[features].iloc[val_ind]
        y_train, y_val = train[CFG.target].iloc[trn_ind], train[CFG.target].iloc[val_ind]
        
        # hyperparameter tuning
        obj = get_catboost_hyper_search(x_train, y_train, x_val, y_val)
        trials = Trials()
        best_catboost_hyperparams = fmin(fn = obj, space = space, algo = tpe.suggest, max_evals = 100, trials = trials)        
        
        model = CatBoostClassifier(iterations = 20500, random_state = 22, nan_mode = 'Min', early_stopping_rounds = 1500, eval_metric=AmexCatboostMetric())
        model.fit(x_train, y_train, eval_set = [(x_val, y_val)], cat_features = cat_features, verbose = 500)
        
        # Save best model
        joblib.dump(model, f'/content/drive/MyDrive/Amex/Models/catboost_{CFG.boosting_type}_fold{fold}_seed{CFG.seed}.pkl')
        
        # Predict validation        
        val_pred = model.predict_proba(x_val)[:, 1]
        # Add to out of folds array
        oof_predictions[val_ind] = val_pred
        # Predict the test set        
        test_pred = model.predict_proba(test[features])[:, 1]
        test_predictions += test_pred / CFG.n_folds
        # Compute fold metric
        score = amex_metric(y_val, val_pred)
        print(f'Our fold {fold} CV score is {score}')
        del x_train, x_val, y_train, y_val
        gc.collect()
    # Compute out of folds metric
    score = amex_metric(train[CFG.target], oof_predictions)
    print(f'Our out of folds CV score is {score}')
    oof_df = pd.DataFrame({'customer_ID': train['customer_ID'], 'target': train[CFG.target], 'prediction': oof_predictions})
    oof_df.to_csv(f'/content/drive/MyDrive/Amex/OOF/oof_catboost_{CFG.boosting_type}_baseline_{CFG.n_folds}fold_seed{CFG.seed}.csv', index = False)
    test_df = pd.DataFrame({'customer_ID': test['customer_ID'], 'prediction': test_predictions})
    test_df.to_csv(f'/content/drive/MyDrive/Amex/Predictions/test_catboost_{CFG.boosting_type}_baseline_{CFG.n_folds}fold_seed{CFG.seed}.csv', index = False)
    
# seed_everything(CFG.seed)
# train, test = read_data()
# train_and_evaluate(train, test)


# ### Submission
# 
# This whole process is still running on my local machine.
# 
# > Needless to say: we canno't run it on Kaggle..
# 
# I keep uploading the final results as a dataset as they become better over time..
# 
# Enjoy!

# In[13]:


sub = pd.read_csv('../input/dart-v2-features-ensemble-sub/blend_xgb_lgb_catb.csv')
sub.to_csv('submission.csv', index = False)

