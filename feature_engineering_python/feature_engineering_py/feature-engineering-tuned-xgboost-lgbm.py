#!/usr/bin/env python
# coding: utf-8

# Wouldnt be possible without these amazing notebooks
# 
# * https://www.kaggle.com/abhishek1aa/feature-engineering-xgboost-lgbm-baseline/notebook
# * https://www.kaggle.com/yus002/realized-volatility-prediction-lgbm-train/data
# * https://www.kaggle.com/konradb/we-need-to-go-deeper

# ### Imports

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import gc

from sklearn.model_selection import train_test_split, KFold

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


# ### Config

# In[2]:


class Config:
    data_dir = '../input/optiver-realized-volatility-prediction/'
    seed = 42


# In[3]:


train = pd.read_csv(Config.data_dir + 'train.csv')
train.head()


# In[4]:


train.stock_id.unique()


# In[5]:


test = pd.read_csv(Config.data_dir + 'test.csv')
test.head()


# In[6]:


display(train.groupby('stock_id').size())

print("\nUnique size values")
display(train.groupby('stock_id').size().unique())


# ### Helper Functions

# #### File reading

# In[7]:


def get_trade_and_book_by_stock_and_time_id(stock_id, time_id=None, dataType = 'train'):
    book_example = pd.read_parquet(f'{Config.data_dir}book_{dataType}.parquet/stock_id={stock_id}')
    trade_example =  pd.read_parquet(f'{Config.data_dir}trade_{dataType}.parquet/stock_id={stock_id}')
    if time_id:
        book_example = book_example[book_example['time_id']==time_id]
        trade_example = trade_example[trade_example['time_id']==time_id]
    book_example.loc[:,'stock_id'] = stock_id
    trade_example.loc[:,'stock_id'] = stock_id
    return book_example, trade_example


# #### Feature engineering

# In[8]:


def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff() 

def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return**2))


def rmspe(y_true, y_pred):
    return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))

def calculate_wap1(df):
    a1 = df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']
    b1 = df['bid_size1'] + df['ask_size1']
    a2 = df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']
    b2 = df['bid_size2'] + df['ask_size2']
    
    x = (a1/b1 + a2/b2)/ 2
    
    return x


def calculate_wap2(df):
        
    a1 = df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']
    a2 = df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']
    b = df['bid_size1'] + df['ask_size1'] + df['bid_size2']+ df['ask_size2']
    
    x = (a1 + a2)/ b
    return x

def realized_volatility_per_time_id(file_path, prediction_column_name):

    stock_id = file_path.split('=')[1]

    df_book = pd.read_parquet(file_path)
    df_book['wap1'] = calculate_wap1(df_book)
    df_book['wap2'] = calculate_wap2(df_book)

    df_book['log_return1'] = df_book.groupby(['time_id'])['wap1'].apply(log_return)
    df_book['log_return2'] = df_book.groupby(['time_id'])['wap2'].apply(log_return)
    df_book = df_book[~df_book['log_return1'].isnull()]

    df_rvps =  pd.DataFrame(df_book.groupby(['time_id'])[['log_return1', 'log_return2']].agg(realized_volatility)).reset_index()
    df_rvps[prediction_column_name] = 0.6 * df_rvps['log_return1'] + 0.4 * df_rvps['log_return2']

    df_rvps['row_id'] = df_rvps['time_id'].apply(lambda x:f'{stock_id}-{x}')
    
    return df_rvps[['row_id',prediction_column_name]]


# In[9]:


def get_agg_info(df):
    agg_df = df.groupby(['stock_id', 'time_id']).agg(mean_sec_in_bucket = ('seconds_in_bucket', 'mean'), 
                                                     mean_price = ('price', 'mean'),
                                                     mean_size = ('size', 'mean'),
                                                     mean_order = ('order_count', 'mean'),
                                                     max_sec_in_bucket = ('seconds_in_bucket', 'max'), 
                                                     max_price = ('price', 'max'),
                                                     max_size = ('size', 'max'),
                                                     max_order = ('order_count', 'max'),
                                                     min_sec_in_bucket = ('seconds_in_bucket', 'min'), 
                                                     min_price = ('price', 'min'),
                                                     #min_size = ('size', 'min'),
                                                     #min_order = ('order_count', 'min'),
                                                     median_sec_in_bucket = ('seconds_in_bucket', 'median'), 
                                                     median_price = ('price', 'median'),
                                                     median_size = ('size', 'median'),
                                                     median_order = ('order_count', 'median')
                                                    ).reset_index()
    
    return agg_df


# #### Most of the feature engineering code

# In[10]:


def get_stock_stat(stock_id : int, dataType = 'train'):
    
    book_subset, trade_subset = get_trade_and_book_by_stock_and_time_id(stock_id, dataType=dataType)
    book_subset.sort_values(by=['time_id', 'seconds_in_bucket'])

    ## book data processing
    
    book_subset['bas'] = (book_subset[['ask_price1', 'ask_price2']].min(axis = 1)
                                / book_subset[['bid_price1', 'bid_price2']].max(axis = 1)
                                - 1)                               

    
    book_subset['wap1'] = calculate_wap1(book_subset)
    book_subset['wap2'] = calculate_wap2(book_subset)
    
    book_subset['log_return_bid_price1'] = np.log(book_subset['bid_price1'].pct_change() + 1)
    book_subset['log_return_ask_price1'] = np.log(book_subset['ask_price1'].pct_change() + 1)
    # book_subset['log_return_bid_price2'] = np.log(book_subset['bid_price2'].pct_change() + 1)
    # book_subset['log_return_ask_price2'] = np.log(book_subset['ask_price2'].pct_change() + 1)
    book_subset['log_return_bid_size1'] = np.log(book_subset['bid_size1'].pct_change() + 1)
    book_subset['log_return_ask_size1'] = np.log(book_subset['ask_size1'].pct_change() + 1)
    # book_subset['log_return_bid_size2'] = np.log(book_subset['bid_size2'].pct_change() + 1)
    # book_subset['log_return_ask_size2'] = np.log(book_subset['ask_size2'].pct_change() + 1)
    book_subset['log_ask_1_div_bid_1'] = np.log(book_subset['ask_price1'] / book_subset['bid_price1'])
    book_subset['log_ask_1_div_bid_1_size'] = np.log(book_subset['ask_size1'] / book_subset['bid_size1'])
    

    book_subset['log_return1'] = (book_subset.groupby(by = ['time_id'])['wap1'].
                                  apply(log_return).
                                  reset_index(drop = True).
                                  fillna(0)
                                 )
    book_subset['log_return2'] = (book_subset.groupby(by = ['time_id'])['wap2'].
                                  apply(log_return).
                                  reset_index(drop = True).
                                  fillna(0)
                                 )
    
    stock_stat = pd.merge(
        book_subset.groupby(by = ['time_id'])['log_return1'].agg(realized_volatility).reset_index(),
        book_subset.groupby(by = ['time_id'], as_index = False)['bas'].mean(),
        on = ['time_id'],
        how = 'left'
    )
    
    stock_stat = pd.merge(
        stock_stat,
        book_subset.groupby(by = ['time_id'])['log_return2'].agg(realized_volatility).reset_index(),
        on = ['time_id'],
        how = 'left'
    )
    
    stock_stat = pd.merge(
        stock_stat,
        book_subset.groupby(by = ['time_id'])['log_return_bid_price1'].agg(realized_volatility).reset_index(),
        on = ['time_id'],
        how = 'left'
    )
    stock_stat = pd.merge(
        stock_stat,
        book_subset.groupby(by = ['time_id'])['log_return_ask_price1'].agg(realized_volatility).reset_index(),
        on = ['time_id'],
        how = 'left'
    )
    stock_stat = pd.merge(
        stock_stat,
        book_subset.groupby(by = ['time_id'])['log_return_bid_size1'].agg(realized_volatility).reset_index(),
        on = ['time_id'],
        how = 'left'
    )
    stock_stat = pd.merge(
        stock_stat,
        book_subset.groupby(by = ['time_id'])['log_return_ask_size1'].agg(realized_volatility).reset_index(),
        on = ['time_id'],
        how = 'left'
    )
    
    stock_stat = pd.merge(
        stock_stat,
        book_subset.groupby(by = ['time_id'])['log_ask_1_div_bid_1'].agg(realized_volatility).reset_index(),
        on = ['time_id'],
        how = 'left'
    )
    stock_stat = pd.merge(
        stock_stat,
        book_subset.groupby(by = ['time_id'])['log_ask_1_div_bid_1_size'].agg(realized_volatility).reset_index(),
        on = ['time_id'],
        how = 'left'
    )
    
    
    stock_stat['stock_id'] = stock_id
    
    # Additional features that can be added. Referenced from https://www.kaggle.com/yus002/realized-volatility-prediction-lgbm-train/data
    
    # trade_subset_agg = get_agg_info(trade_subset)
    
    #     stock_stat = pd.merge(
    #         stock_stat,
    #         trade_subset_agg,
    #         on = ['stock_id', 'time_id'],
    #         how = 'left'
    #     )
    
    ## trade data processing 
    
    return stock_stat

def get_data_set(stock_ids : list, dataType = 'train'):

    stock_stat = Parallel(n_jobs=-1)(
        delayed(get_stock_stat)(stock_id, dataType) 
        for stock_id in stock_ids
    )
    
    stock_stat_df = pd.concat(stock_stat, ignore_index = True)

    return stock_stat_df


# #### Metric

# In[11]:


def rmspe(y_true, y_pred):
    return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))


# #### Plotting

# In[12]:


def plot_feature_importance(df, model):
    feature_importances_df = pd.DataFrame({
        'feature': df.columns,
        'importance_score': model.feature_importances_
    })
    plt.rcParams["figure.figsize"] = [10, 5]
    ax = sns.barplot(x = "feature", y = "importance_score", data = feature_importances_df)
    ax.set(xlabel="Features", ylabel = "Importance Score")
    plt.xticks(rotation=45)
    plt.show()
    return feature_importances_df


# ### Example of book and trade data

# In[13]:


book_stock_1, trade_stock_1 = get_trade_and_book_by_stock_and_time_id(1, 5)
display(book_stock_1.shape)
display(trade_stock_1.shape)


# In[14]:


book_stock_1.head()


# In[15]:


trade_stock_1.head()


# ### Preparing Train and Test set for training and prediction with the desired features
# The following cell takes around 25 mins for execution. You can also use the pickled data from the notebook output and build on that

# In[16]:


get_ipython().run_cell_magic('time', '', "train_stock_stat_df = get_data_set(train.stock_id.unique(), dataType = 'train')\ntrain_stock_stat_df.head()\n")


# In[17]:


train_data_set = pd.merge(train, train_stock_stat_df, on = ['stock_id', 'time_id'], how = 'left')
train_data_set.head()


# In[18]:


train_data_set.info()


# In[19]:


get_ipython().run_cell_magic('time', '', "test_stock_stat_df = get_data_set(test['stock_id'].unique(), dataType = 'test')\ntest_stock_stat_df\n")


# In[20]:


test_data_set = pd.merge(test, test_stock_stat_df, on = ['stock_id', 'time_id'], how = 'left')
test_data_set.fillna(-999, inplace=True)
test_data_set


# #### Storing for later usages. Processing time for features took 25 mins
# You can directly use this from the notebook output and build on that

# In[21]:


train_data_set.to_pickle('train_features_df.pickle')
test_data_set.to_pickle('test_features_df.pickle')


# In[22]:


x = gc.collect()


# In[23]:


X_display = train_data_set.drop(['stock_id', 'time_id', 'target'], axis = 1)
X = X_display.values
y = train_data_set['target'].values

X.shape, y.shape


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=Config.seed, shuffle=False)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# # Optuna Tuned XGBoost

# In[25]:


rs = Config.seed


# In[26]:


import optuna
from optuna.samplers import TPESampler

def objective(trial, data=X, target=y):
    
    def rmspe(y_true, y_pred):
        return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=rs, shuffle=False)
    
    param = {
        'tree_method':'gpu_hist', 
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.008,0.009,0.01,0.012,0.014,0.016,0.018, 0.02]),
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'max_depth': trial.suggest_categorical('max_depth', [5,7,9,11,13,15,17,20]),
        'random_state': trial.suggest_categorical('random_state', [24, 48,2020]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300)}
    
    model = XGBRegressor(**param)
    
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-rmse")
    model.fit(X_train ,y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
    
    preds = model.predict(X_test)
    
    rmspe = rmspe(y_test, preds)
    
    return rmspe


# In[27]:


study = optuna.create_study(sampler=TPESampler(), direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
study.optimize(objective, n_trials=1000, gc_after_trial=True)


# In[28]:


print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)


# In[29]:


optuna.visualization.plot_optimization_history(study)


# In[30]:


optuna.visualization.plot_param_importances(study)


# In[31]:


best_xgbparams = study.best_params
best_xgbparams


# In[32]:


xgb = XGBRegressor(**best_xgbparams, tree_method='gpu_hist')


# In[33]:


get_ipython().run_cell_magic('time', '', "xgb.fit(X_train ,y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)\n\npreds = xgb.predict(X_test)\nR2 = round(r2_score(y_true = y_test, y_pred = preds), 5)\nRMSPE = round(rmspe(y_true = y_test, y_pred = preds), 5)\nprint(f'Performance of the Tuned XGB prediction: R2 score: {R2}, RMSPE: {RMSPE}')\n")


# # Optuna Tuned LGBM

# In[34]:


def objective(trial):
    
    def rmspe(y_true, y_pred):
        return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rs, shuffle=False)
    valid = [(X_test, y_test)]
    
    param = {
        "device": "gpu",
        "metric": "rmse",
        "verbosity": -1,
        'learning_rate':trial.suggest_loguniform('learning_rate', 0.005, 0.5),
        "max_depth": trial.suggest_int("max_depth", 2, 500),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "n_estimators": trial.suggest_int("n_estimators", 100, 4000),
#         "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100000, 700000),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100)}

    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "rmse")
    model = LGBMRegressor(**param)
    
    model.fit(X_train, y_train, eval_set=valid, verbose=False, callbacks=[pruning_callback], early_stopping_rounds=100)

    preds = model.predict(X_test)
    
    rmspe = rmspe(y_test, preds)
    return rmspe


# In[35]:


study = optuna.create_study(sampler=TPESampler(), direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
study.optimize(objective, n_trials=1000, gc_after_trial=True)


# In[36]:


print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)


# In[37]:


optuna.visualization.plot_optimization_history(study)


# In[38]:


optuna.visualization.plot_param_importances(study)


# In[39]:


best_lgbmparams = study.best_params
best_lgbmparams


# In[40]:


lgbm = LGBMRegressor(**best_lgbmparams, device='gpu')


# In[41]:


get_ipython().run_cell_magic('time', '', "lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False, early_stopping_rounds=100)\n\npreds = xgb.predict(X_test)\nR2 = round(r2_score(y_true = y_test, y_pred = preds), 6)\nRMSPE = round(rmspe(y_true = y_test, y_pred = preds), 6)\nprint(f'Performance of the Tuned LIGHTGBM prediction: R2 score: {R2}, RMSPE: {RMSPE}')\n")


# # Stacking Regressor

# In[42]:


def_xgb = XGBRegressor(tree_method='gpu_hist', random_state = rs, n_jobs= - 1)

def_lgbm = LGBMRegressor(device='gpu', random_state=rs)


# In[43]:


from sklearn.ensemble import StackingRegressor


estimators = [('def_xgb', def_xgb),
              ('def_lgbm', def_lgbm),
              ('tuned_xgb', xgb)]

clf = StackingRegressor(estimators=estimators, final_estimator=lgbm, verbose=1)


# In[44]:


get_ipython().run_cell_magic('time', '', 'clf.fit(X_train, y_train)\n')


# In[45]:


preds = clf.predict(X_test)
R2 = round(r2_score(y_true = y_test, y_pred = preds),6)
RMSPE = round(rmspe(y_true = y_test, y_pred = preds), 6)
print(f'Performance of the STACK prediction: R2 score: {R2}, RMSPE: {RMSPE}')


# # Submission

# In[46]:


test_data_set_final = test_data_set.drop(['stock_id', 'time_id'], axis = 1)

y_pred = test_data_set_final[['row_id']]
X_test = test_data_set_final.drop(['row_id'], axis = 1)


# In[47]:


X_test


# In[48]:


y_pred = y_pred.assign(target = clf.predict(X_test))
y_pred.to_csv('submission.csv',index = False)


# In[ ]:




