#!/usr/bin/env python
# coding: utf-8

# # Feature engineering pipeline with optimized LightGBM
# 
# This notebook is an adaptation of [this notebook](https://www.kaggle.com/tommy1028/lightgbm-starter-with-feature-engineering-idea) with improvements in organization, performance, code generalization and readability. I also changed the modelling and feature importance parts to use the scikit-learn API and focus on normalized gain. A lot more could be done in the feature engineering but I'm going to leave it this way, since for me now it has a good organization to keep progressing. Hyperparameter optimization with Optuna was implemented as well.
# 
# **If you think this is relevant or helped you, please give it an upvote. Thanks!**

# ## Preparation

# In[1]:


# Libs to deal with tabular data
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)

# Statistics
from scipy.stats import chi2_contingency
from scipy.stats.contingency import expected_freq

# Plotting packages
import seaborn as sns
sns.axes_style("darkgrid")
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# Machine Learning
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
from boruta import BorutaPy

from lightgbm import LGBMRegressor

# Optimization
import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_contour, plot_optimization_history
from optuna.visualization import plot_param_importances, plot_slice

# To display stuff in notebook
from IPython.display import display, Markdown

# Misc 
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
import time
import os
import glob


# In[2]:


# data directory
DATA_DIR = '../input/optiver-realized-volatility-prediction/'


# ## Simple feature creation functions

# In[3]:


def compute_wap(df, index):
    numerator = df[f'bid_price{index}'] * df[f'ask_size{index}'] + df[f'ask_price{index}'] * df[f'bid_size{index}'] 
    wap = numerator / (df[f'bid_size{index}'] + df[f'ask_size{index}'])
    return wap

def compute_realized_volatility(returns):
    return np.sqrt(np.sum(returns**2))


# ## Main function for preprocessing book data

# In[4]:


def create_book_features(df, windows = [600]):
    # compute prices and returns
    for idx in [1,2]:
        df[f'wap{idx}'] = compute_wap(df, idx)
        df[f'log_wap{idx}'] = np.log(df[f'wap{idx}'])
        df[f'log_return{idx}'] = df.groupby('time_id')[f'log_wap{idx}'].diff()
    
    # compute general book features
    df['wap_balance'] = abs(df['wap1'] - df['wap2'])
    df['price_spread'] = (df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1'])/2)
    df['bid_spread'] = df['bid_price1'] - df['bid_price2']
    df['ask_spread'] = df['ask_price1'] - df['ask_price2']
    df['total_volume'] = (df['ask_size1'] + df['ask_size2']) + (df['bid_size1'] + df['bid_size2'])
    df['volume_imbalance'] = abs((df['ask_size1'] + df['ask_size2']) - (df['bid_size1'] + df['bid_size2']))
    
    # compute aggregations over the features created above for different time windows
    feature_creation_dict = {
        'log_return1':[compute_realized_volatility],
        'log_return2':[compute_realized_volatility],
        'wap_balance':[np.mean],
        'price_spread':[np.mean],
        'bid_spread':[np.mean],
        'ask_spread':[np.mean],
        'volume_imbalance':[np.mean],
        'total_volume':[np.mean],
        'wap1':[np.mean]
    }
    
    window_features = []
    for seconds in windows:
        df_window = df.loc[df['seconds_in_bucket'] >= (600 - seconds), :] if seconds != 600 else df
        df_features = df_window.groupby(['time_id']).agg(feature_creation_dict)
        df_features.columns = ['_'.join(col) + f'_l{seconds}' for col in df_features.columns] # join multi-index column names
        window_features.append(df_features)
        
    df_features = pd.concat(window_features, axis=1, copy=False)     
    return df_features


# In[5]:


get_ipython().run_cell_magic('time', '', 'stock_id = 0\nfile_path = DATA_DIR + f"book_train.parquet/stock_id={stock_id}"\ndf = pd.read_parquet(file_path)\nbook_features = create_book_features(df, [600, 300])\n')


# In[6]:


book_features.head()


# ## Main function for preprocessing trade data

# In[7]:


def create_trade_features(df, windows = [600]):
    # compute return
    df['log_price'] = np.log(df['price'])
    df['log_return'] = df.groupby('time_id')['log_price'].diff()
    
    # compute aggregations for different time windows
    feature_creation_dict = {
        'log_return':[compute_realized_volatility],
        'seconds_in_bucket':'nunique',
        'size':[np.sum],
        'order_count':[np.mean],
    }
    
    window_features = []
    for seconds in windows:
        df_window = df.loc[df['seconds_in_bucket'] >= (600 - seconds), :] if seconds != 600 else df
        df_features = df_window.groupby(['time_id']).agg(feature_creation_dict)
        df_features.columns = ['_'.join(col) + f'_l{seconds}' for col in df_features.columns] # join multi-index column names
        window_features.append(df_features)
    
    df_features = pd.concat(window_features, axis=1, copy=False)   
    return df_features


# In[8]:


get_ipython().run_cell_magic('time', '', 'stock_id = 0\nfile_path = DATA_DIR + f"trade_train.parquet/stock_id={stock_id}"\ndf = pd.read_parquet(file_path)\ntrade_features = create_trade_features(df, [600, 300])\n')


# In[9]:


trade_features.head()


# ## Combined preprocessor function

# In[10]:


def preprocessor(stock_id_list, mode = 'train', windows = [600, 300]):
    # the function above will be parallelized
    def create_stock_features(stock_id):
        book = pd.read_parquet(f"{DATA_DIR}book_{mode}.parquet/stock_id={stock_id}")
        trade = pd.read_parquet(f"{DATA_DIR}trade_{mode}.parquet/stock_id={stock_id}")

        features = create_book_features(book, windows).join(
            create_trade_features(trade, windows), 
            how='outer'
        )
        
        # create row_id
        features['row_id'] = features.index.map(lambda x: f'{stock_id}-{x}')
        features = features.reset_index(drop=True)

        return features
    
    features_list = Parallel(n_jobs=-1, verbose=1)(
        delayed(create_stock_features)(stock_id) for stock_id in stock_id_list
    )
    features = pd.concat(features_list, ignore_index = True)
    return features


# In[11]:


list_stock_ids = [0,1]
features = preprocessor(list_stock_ids, 'train')
features.head()


# ## Training set

# In[12]:


# Reading train file, which maps stock_id and time_it to the target
train = pd.read_csv(DATA_DIR + 'train.csv')
train_stock_ids = train.stock_id.unique()
train['row_id'] = train['stock_id'].astype(str) + '-' + train['time_id'].astype(str)
train = train[['row_id','target']]

# Creating train dataset
df_train = preprocessor(train_stock_ids, 'train')
df_train = train.merge(df_train, on=['row_id'], how='left')

df_train['stock_id'] = df_train['row_id'].apply(lambda x: x.split('-')[0]).astype(int)


# In[13]:


df_train.shape


# ## LightGBM

# In[14]:


X = df_train.drop(['row_id','target'],axis=1)
y = df_train['target']


# In[15]:


def rmspe(y_true, y_pred):
    metric_val = (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))
    return  'rmspe', metric_val, False


# ### Hyperparameter optimization

# In[16]:


class Light_GBM_CV:
    def __init__(self, X, y, folds=5, random_state=42):
        self.X = X
        self.y = y
        self.folds = folds
        self.random_state = random_state

    def __call__(self, trial):
        cv = KFold(
            self.folds, 
            random_state = self.random_state, 
            shuffle=True
        )
        
        clf = LGBMRegressor(
            boosting_type = 'gbdt',
            objective = 'rmse',
            random_state = self.random_state,
            first_metric_only = True,
            num_leaves = trial.suggest_int('num_leaves', 16, 256),
            max_depth = trial.suggest_int('max_depth', 4, 8),
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1.0),
            min_child_samples = trial.suggest_int('min_child_samples', 5, 100),
            n_estimators = trial.suggest_int('n_estimators', 10, 1000),
            lambda_l1 = trial.suggest_loguniform('lambda_l1', 1e-5, 1.0),
            lambda_l2 = trial.suggest_loguniform('lambda_l2', 1e-5, 1.0),
            max_bin = trial.suggest_int('max_bin', 10, 256),
            feature_fraction = trial.suggest_float('feature_fraction', 0.1, 1),
            bagging_fraction = trial.suggest_float('bagging_fraction', 0.1, 1),
            categorical_feature = ['stock_id']
        )
        
        cv_scores = []

        for array_idxs in cv.split(self.X):
            train_index, val_index = array_idxs[0], array_idxs[1]
            X_train, X_val = self.X.loc[train_index], self.X.loc[val_index]
            y_train, y_val = self.y.loc[train_index], self.y.loc[val_index]
            
            clf.fit(
                X_train, y_train,
                sample_weight = 1 / np.square(y_train),
                eval_set = [(X_val, y_val), (X_train, y_train)],
                eval_metric = rmspe,
                early_stopping_rounds = 10,
                verbose = False,
                categorical_feature = ['stock_id']
            )
            cv_scores.append(clf.best_score_['valid_0']['rmspe'])

        return sum(cv_scores) / len(cv_scores)


# In[17]:


lgbm_cv = Light_GBM_CV(X, y)
study = optuna.create_study(sampler=TPESampler(seed = 42), direction='minimize')
study.optimize(lgbm_cv, n_trials=100)


# In[18]:


print('Best model')
print('Mean validation RMSPE: ', study.best_value, '\n')


# In[19]:


study.best_params


# ### Cross validation

# In[20]:


get_ipython().run_cell_magic('time', '', '\nmodels_list = []\ncv_scores = []\n\nkf = KFold(n_splits=5, random_state=19901028, shuffle=True)\nfor fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):\n    print("Fold :", fold+1)\n    \n    # Dataset creation\n    X_train, y_train = X.loc[trn_idx], y[trn_idx]\n    X_valid, y_valid = X.loc[val_idx], y[val_idx]\n    \n    # Modelling\n    model = LGBMRegressor(\n        objective = "rmse",\n        boosting_type = "gbdt",\n        importance_type = \'gain\',\n        first_metric_only = True,\n        random_state = 42,\n        categorical_feature = [\'stock_id\'],\n        **study.best_params\n    )\n    \n    model.fit(\n        X_train, y_train,\n        sample_weight = 1 / np.square(y_train),\n        eval_set = [(X_valid, y_valid), (X_train, y_train)],\n        eval_metric = rmspe,\n        early_stopping_rounds = 30,\n        verbose = 100,\n        categorical_feature = [\'stock_id\']\n    )\n    \n    # validation\n    rmspe_val = model.best_score_[\'valid_0\'][\'rmspe\']\n    print(f\'Performance fold #{fold+1}: {rmspe_val}\')\n\n    #keep scores and models\n    cv_scores.append(rmspe_val)\n    models_list.append(model)\n    print("*" * 100)\n')


# In[21]:


print(f'CV score:', pd.Series(cv_scores).mean())
cv_scores


# ## Feature importance

# In[22]:


raw_imp_vetors = [model.feature_importances_.reshape(1, -1) for model in models_list]
raw_imp_matrix = np.concatenate(raw_imp_vetors, axis=0)
norm_imp = raw_imp_matrix / raw_imp_matrix.sum(1).reshape(-1, 1)
mean_imp = norm_imp.mean(0)
imp_series = pd.Series(mean_imp, index=X.columns).sort_values(ascending=False)


# In[23]:


imp_series


# In[24]:


plt.figure(figsize=(10, 8))
sns.barplot(x = imp_series.values, y = imp_series.index, color='lightblue')
plt.title('Normalized CV feature importance (gain)', fontsize=16)
plt.show()


# # Test set

# In[25]:


# Reading test file
test = pd.read_csv(DATA_DIR + 'test.csv')
test_stock_ids = test.stock_id.unique()
test['row_id'] = test['stock_id'].astype(str) + '-' + test['time_id'].astype(str)
test = test[['row_id']]

# Creating train dataset
df_test = preprocessor(test_stock_ids, 'test')
df_test = test.merge(df_test, on=['row_id'], how='left')

df_test['stock_id'] = df_test['row_id'].apply(lambda x: x.split('-')[0]).astype(int)


# In[26]:


submission = df_test[['row_id']]
X_test = df_test.drop(columns=['row_id'])

# Scoring ensemble
target = np.zeros(len(X_test))
for model in models_list:
    pred = model.predict(X_test, num_iteration=model.best_iteration_)
    target += pred / len(models_list)

submission = submission.assign(target = target)    


# In[27]:


submission.head()


# In[28]:


submission.to_csv('submission.csv', index = False)

