#!/usr/bin/env python
# coding: utf-8

# # On-line FE + XGB ranker framework 

# In[1]:


import jpx_tokyo_market_prediction
env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()


# In[2]:


import numpy as np
import pandas as pd
import random
import os

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from datetime import datetime
import pickle

pd.set_option('max_columns', 350)

def timestamp_to_date(timestamp):
    return(datetime.fromtimestamp(timestamp))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

import xgboost as xgb

DEBUG = False
BUILD_DATA = True
GPU = True

import warnings
if ~DEBUG:
    warnings.filterwarnings('ignore')


# # Build / Read Data

# In[3]:


if BUILD_DATA:
    # Macro_data
    macro_data = pd.read_csv('../input/jpx-macro-data-from-public-apis/Macro_data.csv',index_col=0).ffill()

    print(f'Macro Data Shape: {macro_data.shape}')

    macro_data_diff = macro_data.diff()
    macro_data_diff.columns = [c+'_diff' for c in macro_data.columns]
    macro_data = pd.concat([macro_data,macro_data_diff],axis=1)
    macro_data.columns = ['macro_'+c for c in macro_data.columns]

    macro_data.tail()

    # Options_data
    option_data = pd.read_parquet('../input/jpx-eda-feature-engineering-options/options_train_FE.parquet').ffill().fillna(0)

    option_data_diff = option_data.diff()
    option_data_diff.columns = [c+'_diff' for c in option_data.columns]
    option_data = pd.concat([option_data,option_data_diff],axis=1)
    option_data.columns = ['options_'+c for c in option_data.columns]

    print(f'Options Data Shape: {option_data.shape}')
    
    # Price data
    price_features = ['SupervisionFlag','Side','ret_HL','ret','ret_Div','log_Dollars','GK_sqrt_vol','RS_sqrt_vol']
    market_features = ['Side_M_mean','ret_HL_M_mean','ret_M_mean','ret_Div_M_mean','log_Dollars_M_mean','GK_sqrt_vol_M_mean','RS_sqrt_vol_M_mean',
    'Side_M_std','ret_HL_M_std','ret_M_std','ret_Div_M_std','log_Dollars_M_std','GK_sqrt_vol_M_std','RS_sqrt_vol_M_std','Side_M_skew',
                       'ret_HL_M_skew','ret_M_skew','ret_Div_M_skew','log_Dollars_M_skew','GK_sqrt_vol_M_skew','RS_sqrt_vol_M_skew']
    weekly_features  = ['Side_W','ret_HL_W','ret_W','ret_Div_W','log_Dollars_W','GK_sqrt_vol_W','RS_sqrt_vol_W',
    'Side_M_mean_W','ret_HL_M_mean_W','ret_M_mean_W','ret_Div_M_mean_W','log_Dollars_M_mean_W','GK_sqrt_vol_M_mean_W','RS_sqrt_vol_M_mean_W','Side_M_std_W',
    'ret_HL_M_std_W','ret_M_std_W','ret_Div_M_std_W','log_Dollars_M_std_W','GK_sqrt_vol_M_std_W','RS_sqrt_vol_M_std_W','Side_M_skew_W','ret_HL_M_skew_W',
    'ret_M_skew_W','ret_Div_M_skew_W','log_Dollars_M_skew_W','GK_sqrt_vol_M_skew_W','RS_sqrt_vol_M_skew_W']

    time_features = ['sin_month','cos_month','sin_week','cos_week','sin_day','cos_day']
    beta = ['Beta_M', 'Beta_Q','Beta_Y']
    Ids = ['Date','SecuritiesCode','Target']

    price_data = pd.read_parquet('../input/jpx-online-feature-engineering-prices/train_FE.parquet').ffill()

    price_data =  price_data[Ids + price_features+market_features+weekly_features+time_features+beta]
    price_data.columns = ['price_'+c for c in price_data.columns]

    # financial data 
    financial_data = pd.read_parquet('../input/jpx-eda-feature-engineering-fundamental-data/train_financials.parquet').ffill()

    ratios_features = financial_data.columns[financial_data.columns.str.startswith('r')].to_list()
    yoy_growth_features = financial_data.columns[financial_data.columns.str.endswith('YoY_growth')].to_list()
    financial_ids_features = ['Date','DisclosureNumber','DateCode','SecuritiesCode','DaysSinceDisclosure']

    financial_data = financial_data[financial_ids_features + ratios_features + yoy_growth_features]
    financial_data.columns = ['financial_'+c for c in financial_data.columns]

    # merge all  the stuff
    exo_data = pd.concat([option_data,macro_data],axis=1).astype('float32')
    exo_data.head()

    price_data = price_data.merge(financial_data, left_on=['price_Date', 'price_SecuritiesCode'], right_on=['financial_Date', 'financial_SecuritiesCode'], how='left')
    del financial_data

    price_data = price_data.merge(exo_data,how='left', left_on='price_Date', right_index=True)
    price_data.to_parquet('price_data.parquet')
    
else:
    price_data = pd.read_parquet('price_data.parquet')
    price_data.to_parquet('price_data.parquet')


# In[4]:


price_data['price_Date'] = pd.to_datetime(price_data['price_Date'])
date_min = price_data['price_Date'].min()
date_max = price_data['price_Date'].max()

#date_end_train = date_min + (date_max-date_min)*0.8
date_start_train = pd.to_datetime('2020-01-01')
date_end_train = pd.to_datetime('2020-12-31')

train = price_data[(price_data['price_Date']>=date_start_train)&(price_data['price_Date']<date_end_train)]
test = price_data[price_data['price_Date']>date_end_train]

del price_data


# # clean data types

# In[5]:


Num_Features = train.columns[(train.dtypes=='float32')|(train.dtypes=='float64')]
Cat_Features = ['price_SecuritiesCode', 'price_SupervisionFlag']
Target_Features = ['price_Target']
Groups_Features = ['price_Date']

Num_Features = [f for f in Num_Features if f not in Cat_Features+Target_Features+Groups_Features]
All_Features = Num_Features + Cat_Features


# In[6]:


def prepare_data(df):
    y_df = df[Target_Features]
    groups_df = df.groupby(Groups_Features).count()['price_SecuritiesCode'].values
    time_id_df = df[Groups_Features]
    return (df[All_Features], y_df, groups_df, time_id_df)


# In[7]:


X_train, y_train, groups_train, time_id_train = prepare_data(train)
del train

X_test, y_test, groups_test, time_id_test = prepare_data(test)
del test


# In[8]:


X_train.replace([-np.inf], -100, inplace=True)
X_test.replace([-np.inf], -100, inplace=True)

X_train.replace([np.inf], 10000, inplace=True)
X_test.replace([np.inf], 10000, inplace=True)


# In[9]:


X_train.price_SecuritiesCode = X_train.price_SecuritiesCode.astype('category')
X_test.price_SecuritiesCode = X_test.price_SecuritiesCode.astype('category')


# In[10]:


def calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
    weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
    purchase = (df.sort_values(by='Rank')['Target'][:portfolio_size] * weights).sum() / weights.mean()
    short = (df.sort_values(by='Rank', ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
    return purchase - short

def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2):
    buf = df.groupby('Date').apply(calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio

def pred_to_rank(df_to_test, remove_top_n = 0):
    df_to_test['Rank'] = (df_to_test.groupby('Date')['preds'].transform('rank')-1).astype('int')
    return df_to_test
    
def eval_preds(df_to_test):
    spread_return_per_day = df_to_test.groupby('Date').apply(calc_spread_return_per_day, 200, 2)
    corr_by_day = df_to_test.groupby('Date')['Target','preds'].corr(method='spearman').iloc[0::2,-1].values
    plt.plot(spread_return_per_day.reset_index(drop=True))
    plt.hlines(np.mean(spread_return_per_day),xmin=0,xmax=len(spread_return_per_day),color='k')
    plt.show()
    plt.plot(corr_by_day)
    plt.hlines(np.nanmean(corr_by_day),xmin=0,xmax=len(corr_by_day),color='k')
    plt.show()
    sharpe = calc_spread_return_sharpe(df_to_test, 200, 2)
    return (corr_by_day, sharpe)


# In[11]:


get_ipython().run_cell_magic('time', '', "\nn_seed = 2 if DEBUG else 5\nn_estimators = 100\n\ndict_model = {}\nscores = []\nimportances = []\nensemble_preds = []\n\nfor seed in range(n_seed):\n    print(f'calibrating - seed:{seed}')\n    evals_result = {}\n\n    model = xgb.XGBRanker(booster='gbtree',\n                          tree_method = 'gpu_hist',\n                          enable_categorical=True,\n                          #use_label_encoder=False,\n                          #evals_result=evals_result,\n                          objective='rank:pairwise',\n                          #reg_lambda=1,\n                          #feval=metric,\n                          learning_rate=0.01,\n                          colsample_bytree=0.3, \n                          #eta=0.05,\n                          max_depth=6,\n                          n_estimators=n_estimators,\n                          subsample=0.3,\n                          random_state = seed,\n                          )\n\n    model.fit(X_train, y_train, #sample_weight = weights_train, \n              group=groups_train, eval_set= [(X_train, y_train), (X_test, y_test)], eval_group = [groups_train, groups_test])\n\n    preds = model.predict(X_test)\n    ensemble_preds.append(preds)\n    \n    df_to_test = pd.DataFrame(data={'Date':time_id_test.values.flatten(),'Target':y_test.values.flatten(),'preds':preds})\n    \n    df_to_test = pred_to_rank(df_to_test)\n    corr_by_day, sharpe = eval_preds(df_to_test)\n    \n    print(f'evaluating - seed:{seed}, average spearman:{np.nanmean(corr_by_day):.2%}, sharpe:{sharpe:.2%}')\n\n    dict_model[seed] = model\n    scores.append((seed, sharpe))\n    importances.append(model.feature_importances_)\n")


# In[12]:


df_to_test = pd.DataFrame(data={'Date':time_id_test.values.flatten(),'Target':y_test.values.flatten(),'preds':np.array(ensemble_preds).mean(axis=0)})
df_to_test = pred_to_rank(df_to_test) 

corr_by_day_ens, sharpe_ens = eval_preds(df_to_test)
print(f'evaluating - seed:{seed}, average spearman:{np.nanmean(corr_by_day_ens):.2%}, sharpe:{sharpe_ens:.2%}')


# In[13]:


df_results = pd.DataFrame(scores,columns=['seed','sharpe']).set_index('seed')

df_results.loc['seed_mean']= df_results.mean(numeric_only=True, axis=0)

cm = sns.light_palette("green", as_cmap=True)
df_results.style.background_gradient(axis=0, cmap=cm,vmin=0,vmax=0.1)


# In[14]:


for seed in range(n_seed):
    model = dict_model[seed] 
    plt.plot(model.evals_result()['validation_0']['map'], label= f'Training - seed {seed}')
    plt.plot(model.evals_result()['validation_1']['map'], label= f'Testing - seed {seed}')

plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.show()


# In[15]:


id_features = ['index','row_id','time_id','target']
cat_features = ['pattern_id','investment_id'] + [c for c in X_train.columns if 'c' in c]
num_features = [c for c in X_train.columns if c not in id_features+cat_features]


# In[16]:


def plot_importance(importances, features_names = X_train.columns, PLOT_TOP_N = 20, figsize=(10, 20)):
    importance_df = pd.DataFrame(data=importances, columns=features_names)
    sorted_indices = importance_df.median(axis=0).sort_values(ascending=False).index
    sorted_importance_df = importance_df.loc[:, sorted_indices]
    plot_cols = sorted_importance_df.columns[:PLOT_TOP_N]
    _, ax = plt.subplots(figsize=figsize)
    ax.grid()
    ax.set_xscale('log')
    ax.set_ylabel('Feature')
    ax.set_xlabel('Importance')
    sns.boxplot(data=sorted_importance_df[plot_cols],
                orient='h',
                ax=ax)
    plt.show()

def plot_importance_groups(importances, features_names = X_train.columns, PLOT_TOP_N = 20, figsize=(4, 8)):
    importance_df = pd.DataFrame(data=importances, columns=features_names)
    sorted_indices = importance_df.median(axis=0).sort_values(ascending=False).index
    sorted_importance_df = importance_df.loc[:, sorted_indices]
    plot_cols = sorted_importance_df.columns[:PLOT_TOP_N]
    
    t = sorted_importance_df.transpose()
    t['groups'] = [s.split('_')[0] for s in sorted_importance_df.columns]
    
    t = t.groupby('groups').sum().transpose()

    
    _, ax = plt.subplots(figsize=figsize)
    ax.grid()
    #ax.set_xscale('log')
    ax.set_xlim(left=0, right=1)
    ax.set_ylabel('Feature')
    ax.set_xlabel('Importance')
    sns.boxplot(data=t,
                orient='h',
                ax=ax)
    plt.show()
    
plot_importance(np.array(importances))
plot_importance_groups(np.array(importances))


# In[17]:


pickle.dump(dict_model, open('lgbm_models.pkl', 'wb'))
pickle.dump(scores, open('scores.pkl', 'wb'))
pickle.dump(importances, open('importances.pkl', 'wb'))
pickle.dump(X_train.columns, open('features.pkl', 'wb'))


# # predictions

# In[18]:


# for (test_df, sample_prediction_df) in iter_test:
#     preds = []
#     row_id = test_df['row_id']
#     test_df['time_id'] = test_df.row_id.str.split('_',expand=True)[0]
#     test_df = prepare_X(test_df)
#     for diff in range(n_diff):
#         for fold in range(n_fold):
#             for seed in range(n_seed):
#                 model = dict_model[(diff,fold,seed)] 
#                 preds.append(model.predict(test_df))
#     test_df['target'] = np.mean(np.array(preds),axis=0)
#     test_df['row_id'] = row_id
#     env.predict(test_df[['row_id','target']])

