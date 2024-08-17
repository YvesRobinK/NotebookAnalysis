#!/usr/bin/env python
# coding: utf-8

# # Optiver LGBM feature engineering with Optuna
# 
# I combined three notebooks with some change in the codes in order to use both feature engineering and optuna for hyperparameter optimization. I used number of trial as 1 for illustration purpose only but you can increase the number. The original notebooks that I referred are :
# 
# https://www.kaggle.com/code/moritzm00/optiver-lgbm-tuning-w-optuna?scriptVersionId=144585564
# 
# https://www.kaggle.com/code/ravi20076/optiver-memoryreduction
# 
# https://www.kaggle.com/code/renatoreggiani/optv-lgbm-starter

# In[1]:


#!pip install -qqq loguru
LOGGING_LEVELS = ["DEBUG", "WARNING", "INFO", "SUCCESS", "WARNING"]


# In[2]:


import sys
import numpy as np
import pickle
#from loguru import logger # for nice colored logging
from pprint import pprint, pformat
import pandas as pd
import json
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from timeit import default_timer as timer
from IPython.display import clear_output
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)

sns.set_style("ticks")
from itertools import combinations
import gc
import plotly.express as px


# In[3]:


import joblib


# # Data Loading
# 
# No feature engineering for now. I'll add this later.

# In[4]:


#train = (
#    pd
#    .read_csv("/kaggle/input/optiver-trading-at-the-close/train.csv")
#    .drop(columns=["time_id", "row_id", "date_id", "near_price", "far_price"], axis=1)
#    .dropna()
#)
#train.info()


# In[5]:


train = pd.read_csv('/kaggle/input/optiver-trading-at-the-close/train.csv')
revealed_targets = pd.read_csv('/kaggle/input/optiver-trading-at-the-close/example_test_files/revealed_targets.csv')
test = pd.read_csv('/kaggle/input/optiver-trading-at-the-close/example_test_files/test.csv')
sample_submission = pd.read_csv('/kaggle/input/optiver-trading-at-the-close/example_test_files/sample_submission.csv')


# In[6]:


train


# In[7]:


median_vol = train.groupby('stock_id')['bid_size'].median() + train.groupby('stock_id')['ask_size'].median()


# In[8]:


def reduce_mem_usage(df):
    
    """ 
    Iterate through all numeric columns of a dataframe and modify the data type
    to reduce memory usage.        
    """
    
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage of dataframe is {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage after optimization is: {end_mem:.2f} MB')
    decrease = 100 * (start_mem - end_mem) / start_mem
    print(f'Decreased by {decrease:.2f}%')
    
    return df


# In[9]:


def feat_eng(df):
    
    cols = [c for c in df.columns if c not in ['row_id', 'time_id']]
    df = df[cols]
    
    df['imbalance_buy_flag'] = np.where(df['imbalance_buy_sell_flag']==1, 1, 0) 
    df['imbalance_sell_flag'] = np.where(df['imbalance_buy_sell_flag']==-1, 1, 0) 
    df['bid_plus_ask_sizes'] = df['bid_size'] + train['ask_size']
    df['median_vol'] = df['stock_id'].map(median_vol.to_dict())
    df['high_volume'] = np.where(df['bid_plus_ask_sizes'] > df['median_vol'], 1, 0) 
    df['imbalance_ratio'] = df['imbalance_size'] / df['matched_size']
    
    df['imb_s1'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)')
    df['imb_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)')

    df['ask_x_size'] = df.eval('ask_size*ask_price')
    df['bid_x_size'] = df.eval('bid_size*bid_price')
        
    df['ask_minus_bid'] = df['ask_x_size'] - df['bid_x_size'] 
    
    df["bid_size_over_ask_size"] = df["bid_size"].div(df["ask_size"])
    df["bid_price_over_ask_price"] = df["bid_price"].div(df["ask_price"])
    
    prices = ['reference_price','far_price', 'near_price', 'ask_price', 'bid_price', 'wap']
    
    for c in combinations(prices, 2):
        
        df[f'{c[0]}_minus_{c[1]}'] = (df[f'{c[0]}'] - df[f'{c[1]}']).astype(np.float32)
        df[f'{c[0]}_times_{c[1]}'] = (df[f'{c[0]}'] * df[f'{c[1]}']).astype(np.float32)
        df[f'{c[0]}_{c[1]}_imb'] = df.eval(f'({c[0]}-{c[1]})/({c[0]}+{c[1]})')

    for c in combinations(prices, 3):
        max_ = df[list(c)].max(axis=1)
        min_ = df[list(c)].min(axis=1)
        mid_ = df[list(c)].sum(axis=1)-min_-max_

        df[f'{c[0]}_{c[1]}_{c[2]}_imb2'] = (max_-mid_)/(mid_-min_)
    df.drop(columns=['date_id'], inplace=True)
    df=reduce_mem_usage(df)
    gc.collect()
    
    return df


# In[10]:


train.dropna(subset=['target'], inplace=True)


# In[11]:


y = train['target']
X = feat_eng(train.drop(columns='target'))


# In[12]:


#X = train.drop(columns="target")
#y = train["target"]


# # Hyperparameter Optimization

# ### Evaluation via Cross Validation with TimeSeriesSplit

# In[13]:


def cross_validate(model, X, y, cv):
    scores = np.zeros(cv.n_splits)
    
    #logger.info(f"Starting evaluation...")
    #logger.info("="*80)
    for i, (train_index, test_index) in enumerate(cv.split(X)):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, shuffle=False, test_size=0.1)
        
        start = timer()
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        end = timer()
        
        y_pred = model.predict(X_test)
        scores[i] = mean_absolute_error(y_pred, y_test)
        #logger.info(f"Fold {i+1}: {scores[i]:.4f} (took {end - start:.2f}s)")
        
    #logger.info("-"*80)
    #logger.success(f"Average MAE = {scores.mean():.4f} +- {scores.std():.2f}")
    #logger.info("="*80)
    
    return scores


# ### Simple evaluation with train-test split

# In[14]:


def evaluate_simple(model, X, y, cv):
    #logger.info(f"Starting evaluation...")
    #logger.info("="*40)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    
    # eval sets for early stopping
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, shuffle=False, test_size=0.1)

    start = timer()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    end = timer()

    y_pred = model.predict(X_test)
    score = mean_absolute_error(y_pred, y_test)
    #logger.success(f"MAE = {score:.4f} (took {end - start:.2f}s)")
    #logger.info("="*40)
    
    return score


# In[15]:


# convenience function
def run_optimization(objective, n_trials=100, n_jobs=1):
    """Run the given objective with Optuna and return the study results."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    #logger.info(f"Number of finished trials: {len(study.trials)}")
    #logger.success(f"Best MAE: {study.best_value:.4f}")

    #logger.info("Params")
    #logger.info("=" * 10)
    #logger.success(pformat(study.best_params, indent=4))
    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f)
    return study


# In[16]:


def get_objective_function(evaluation="simple", cv=None, logging_level="info"):
    """Returns the objective function for optuna."""
    
    
    if evaluation == "simple":
        eval_function = evaluate_simple
    else:
        eval_function = cross_validate
        
    def optimize_lgbm(trial):
        """Optimizes a LGBMRegressor with cross-validation."""
        # num_leaves should be smaller than 2^{max_depth}
        max_depth = trial.suggest_int("max_depth", 6, 9)
        num_leaves = trial.suggest_int("num_leaves", 32, int((2**max_depth) * 0.90))

        param_space = {
            "boosting":'gbdt',
            "objective": trial.suggest_categorical("objective", ["mae"]),
            "random_state": trial.suggest_categorical("random_state", [42]),
            "n_estimators": trial.suggest_categorical("n_estimators", [600]), # automatically via early stopping
            #"min_child_samples": trial.suggest_int("min_child_samples", 500, 2000),
            #"subsample": trial.suggest_float("subsample", 0.2, 1.0),
            #"subsample_freq": trial.suggest_categorical("subsample_freq", [1, 2, 3]),
            #"colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            #"min_split_gain": trial.suggest_float("min_split_gain", 0, 5),
        }
        model = LGBMRegressor(**param_space)    
        scores = eval_function(model, X, y, cv=cv)
        return scores.mean()
    return optimize_lgbm


# In[17]:


m = lgb.LGBMRegressor(objective='mae', n_estimators=600, random_state=51)
m.fit(X, y)


# In[18]:


feat_imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
print('Columns with poor contribution', feat_imp[feat_imp<10].index)
fig = px.bar(x=feat_imp, y=feat_imp.index, orientation='h')
fig.show()


# In[19]:


feat_imp


# In[20]:


test=feat_eng(test)


# In[21]:


test.shape


# In[22]:


y.isna().any()


# In[23]:


y


# ### Start the optimization
# 
# We can set these variables here:
# 1. `run_lgbm_optimization`: Whether to run the optimization or use already computed ones.
# 2. `n_trials`: How many trials we want to sample.
# 3. `logging_level`: Configures the logging level inside evaluation functions (use either `'info'` or `'success'`)
# 4. `evaluation`: Use either 'simple' for simple train-test split or 'cross_validate' for cross validation using TimeSeriesSplit.
# 5. `cv`: the Split object
# 
# *Warning*: `evaluation='cross_validate'` takes very long!

# In[24]:


gc.collect()


# In[25]:


run_lgbm_optimization = True
n_trials = 30
logging_level = "success"
evaluation = "simple"
cv = TimeSeriesSplit(n_splits=3)

if run_lgbm_optimization:
    clear_output(wait=True) # clears output before rerunning optimization
    objective = get_objective_function(evaluation=evaluation, cv=cv)
    study = run_optimization(objective, n_trials=n_trials, n_jobs=1)
    lgbm_best_params = study.best_params
    
    plot_optimization_history(study).show()
    if n_trials > 1:
        plot_param_importances(study).show()
        plot_parallel_coordinate(study).show()
else:
    lgbm_best_params = {}


# ## Retrain and save for inference

# In[26]:


model = LGBMRegressor(**lgbm_best_params)


# In[27]:


model.fit(X,y)


# In[28]:


model.predict(test)


# In[29]:


#joblib.dump(model, 'model_filename.pkl')


# In[30]:


#loaded_model = joblib.load('model_filename.pkl')


# In[31]:


#loaded_model.predict(feat)


# In[32]:


import optiver2023
env = optiver2023.make_env()
iter_test = env.iter_test()


# In[33]:


test


# In[34]:


def zero_sum(prices, volumes):
    
#    I got this idea from https://github.com/gotoConversion/goto_conversion/
    
    std_error = np.sqrt(volumes)
    step = np.sum(prices)/np.sum(std_error)
    out = prices-std_error*step
    
    return out


# In[35]:


counter = 0
for (test, revealed_targets, sample_prediction) in iter_test:
    
    feat = feat_eng(test)
    sample_prediction['target'] = model.predict(feat)
    
    sample_prediction['target'] = zero_sum(sample_prediction['target'], test.loc[:,'bid_size'] + test.loc[:,'ask_size'])
    
    env.predict(sample_prediction)
    
    counter += 1


# In[36]:


sample_prediction['target']


# In[ ]:




