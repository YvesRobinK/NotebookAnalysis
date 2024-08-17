#!/usr/bin/env python
# coding: utf-8

# ## Elo Merchant Category Recommendation
# ---
# > ***Help understand customer loyalty***
# 
# ![](https://www.cartaoelo.com.br/images/home-promo-new.jpg)
# 
# ---
# 
# > ## Objective
# > <p style="text-align:justify">Elo has built machine learning models to understand the most important aspects and preferences in their customers’ lifecycle, from food to shopping. But so far none of them is specifically tailored for an individual or profile. This is where you come in.</p>
# > ## Solution thought by me
# > In this kernel, I build a LGBM model that aggregates the `new_merchant_transactions.csv` and `historical_transactions.csv` tables to the main train table. New features are built by successive grouping on`card_id` and `month_lag`, in order to recover some information from the time serie.
# </div></div></div>
# 
# 
# > ## Notebook  Content
# > 1. [***Loading the data***](#1)
# > 1. [***Feature engineering***](#2)
# > 1. [***Training the model***](#3)
# > 1. [***Feature importance***](#4)
# > 1. [***Submission***](#5)
# > 1. [***Stacking***](#6)
# ---

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold
import warnings
import time
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from tqdm import tqdm_notebook as tqdm, tqdm_pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge, BayesianRidge
warnings.simplefilter(action='ignore', category=FutureWarning)
import gc
import time
import datetime


# ## 1. Loading the data <a id="1"></a> <br>
# 
# First, we load the `new_merchant_transactions.csv` and `historical_transactions.csv`. In practice, these two files contain the same variables and the difference between the two tables only concern the position with respect to a reference date.  Also, booleans features are made numeric:

# In[ ]:


get_ipython().run_cell_magic('time', '', "def reduce_mem_usage(df, verbose=True):\n    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']\n    start_mem = df.memory_usage().sum() / 1024**2    \n    for col in df.columns:\n        col_type = df[col].dtypes\n        if col_type in numerics:\n            c_min = df[col].min()\n            c_max = df[col].max()\n            if str(col_type)[:3] == 'int':\n                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:\n                    df[col] = df[col].astype(np.int8)\n                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:\n                    df[col] = df[col].astype(np.int16)\n                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:\n                    df[col] = df[col].astype(np.int32)\n                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:\n                    df[col] = df[col].astype(np.int64)  \n            else:\n                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:\n                    df[col] = df[col].astype(np.float16)\n                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:\n                    df[col] = df[col].astype(np.float32)\n                else:\n                    df[col] = df[col].astype(np.float64)    \n    end_mem = df.memory_usage().sum() / 1024**2\n    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))\n    return df\n\ngc.collect()\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "new_transactions = pd.read_csv('../input/new_merchant_transactions.csv', parse_dates=['purchase_date'])\nhistorical_transactions = pd.read_csv('../input/historical_transactions.csv', parse_dates=['purchase_date'])\ngc.collect()\n")


# We then load the main files, formatting the dates and extracting the target:

# In[ ]:


get_ipython().run_cell_magic('time', '', "def read_data(input_file):\n    df = pd.read_csv(input_file)\n    df['first_active_month'] = pd.to_datetime(df['first_active_month'])\n    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days\n    return df\n\n# Read data train and test file\ntrain = read_data('../input/train.csv')\ntest = read_data('../input/test.csv')\n\ntarget = train['target']\ndel train['target']\ngc.collect()\n")


# ## 2.Feature engineering <a id="2"></a> <br>
# * First, following [Robin Denz](https://www.kaggle.com/denzo123/a-closer-look-at-date-variables) and [konradb](https://www.kaggle.com/konradb/lgb-fe-lb-3-707) analysis, I define a few dates features.
# * Binarize the categorical variables where it makes sense

# ### Historical Transactions

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nhistorical_transactions['authorized_flag'] = historical_transactions['authorized_flag'].map({'Y':1, 'N':0})\nhistorical_transactions['category_1'] = historical_transactions['category_1'].map({'Y':1, 'N':0})\n\nhistorical_transactions['category_2x1'] = (historical_transactions['category_2'] == 1) + 0\nhistorical_transactions['category_2x2'] = (historical_transactions['category_2'] == 2) + 0\nhistorical_transactions['category_2x3'] = (historical_transactions['category_2'] == 3) + 0\nhistorical_transactions['category_2x4'] = (historical_transactions['category_2'] == 4) + 0\nhistorical_transactions['category_2x5'] = (historical_transactions['category_2'] == 5) + 0\n\nhistorical_transactions['category_3A'] = (historical_transactions['category_3'].astype(str) == 'A') + 0\nhistorical_transactions['category_3B'] = (historical_transactions['category_3'].astype(str) == 'B') + 0\nhistorical_transactions['category_3C'] = (historical_transactions['category_3'].astype(str) == 'C') + 0\n\nhistorical_transactions = reduce_mem_usage(historical_transactions)\nnew_transactions = reduce_mem_usage(new_transactions)\ngc.collect()\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "def aggregate_historical_transactions(history):\n    \n    history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']).\\\n                                      astype(np.int64) * 1e-9\n    \n    agg_func = {\n        'authorized_flag': ['sum', 'mean'],\n        'category_1': ['sum', 'mean'],\n        'category_2': ['nunique'],\n        'category_3A': ['sum'],\n        'category_3B': ['sum'],\n        'category_3C': ['sum'],\n        'category_2x1': ['sum','mean'],\n        'category_2x2': ['sum','mean'],\n        'category_2x3': ['sum','mean'],\n        'category_2x4': ['sum','mean'],\n        'category_2x5': ['sum','mean'],        \n        'city_id': ['nunique'],\n        'installments': ['sum', 'median', 'max', 'min', 'std'],\n        'merchant_category_id': ['nunique'],\n        'merchant_id': ['nunique'],\n        'month_lag': ['min', 'max'],\n        'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],\n        'purchase_date': [np.ptp, 'max', 'min'],\n        'state_id': ['nunique'],\n        'subsector_id': ['nunique'],\n\n        }\n    agg_history = history.groupby(['card_id']).agg(agg_func)\n    agg_history.columns = ['hist_' + '_'.join(col).strip() \n                           for col in agg_history.columns.values]\n    agg_history.reset_index(inplace=True)\n    \n    df = (history.groupby('card_id')\n          .size()\n          .reset_index(name='hist_transactions_count'))\n    \n    agg_history = pd.merge(df, agg_history, on='card_id', how='left')\n    \n    return agg_history\n\nhistory = aggregate_historical_transactions(historical_transactions)\nhistory.columns = ['hist_' + c if c != 'card_id' else c for c in history.columns]\ndisplay(history[:5])\n\ndel historical_transactions\ngc.collect()\n")


# ### New Transaction

# Then I define two functions that aggregate the info contained in these two tables. The first function aggregates the function by grouping on `card_id`:

# In[ ]:


get_ipython().run_cell_magic('time', '', "new_transactions['authorized_flag'] = new_transactions['authorized_flag'].map({'Y':1, 'N':0})\n\nnew_transactions['category_1'] = new_transactions['category_1'].map({'Y':1, 'N':0})\nnew_transactions['category_3A'] = (new_transactions['category_3'].astype(str) == 'A') + 0\nnew_transactions['category_3B'] = (new_transactions['category_3'].astype(str) == 'B') + 0\nnew_transactions['category_3C'] = (new_transactions['category_3'].astype(str) == 'C') + 0\n\nnew_transactions['category_2x1'] = (new_transactions['category_2'] == 1) + 0\nnew_transactions['category_2x2'] = (new_transactions['category_2'] == 2) + 0\nnew_transactions['category_2x3'] = (new_transactions['category_2'] == 3) + 0\nnew_transactions['category_2x4'] = (new_transactions['category_2'] == 4) + 0\nnew_transactions['category_2x5'] = (new_transactions['category_2'] == 5) + 0\n\ngc.collect()\n")


# In[ ]:


def aggregate_new_transactions(new_trans):    
    
    new_transactions['purchase_date'] = pd.DatetimeIndex(new_transactions['purchase_date']).astype(np.int64) * 1e-9
    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'category_1': ['sum', 'mean'],
        'category_2': ['nunique'],
        'category_3A': ['sum'],
        'category_3B': ['sum'],
        'category_3C': ['sum'],     
        'category_2x1': ['sum','mean'],
        'category_2x2': ['sum','mean'],
        'category_2x3': ['sum','mean'],
        'category_2x4': ['sum','mean'],
        'category_2x5': ['sum','mean'],        

        'city_id': ['nunique'],
        'installments': ['sum', 'median', 'max', 'min', 'std'],
        'merchant_category_id': ['nunique'],
        'merchant_id': ['nunique'],
        'month_lag': ['min', 'max'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
        'purchase_date': [np.ptp, 'max', 'min'],
        'state_id': ['nunique'],
        'subsector_id': ['nunique']        
        }
    agg_new_trans = new_trans.groupby(['card_id']).agg(agg_func)
    agg_new_trans.columns = ['new_' + '_'.join(col).strip() 
                           for col in agg_new_trans.columns.values]
    agg_new_trans.reset_index(inplace=True)
    
    df = (new_trans.groupby('card_id')
          .size()
          .reset_index(name='new_transactions_count'))
    
    agg_new_trans = pd.merge(df, agg_new_trans, on='card_id', how='left')
    
    return agg_new_trans

new = aggregate_new_transactions(new_transactions)
new.columns = ['new_' + c if c != 'card_id' else c for c in new.columns]
display(new[:5])

del new_transactions
gc.collect()


# The second function first aggregates on the two variables `card_id` and `month_lag`. Then a second grouping is performed to aggregate over time:

# ## 3. Training the model<a id="3"></a> 
# We now train the model with the features we previously defined. A first step consists in merging all the dataframes:

# In[ ]:


get_ipython().run_cell_magic('time', '', "print(train.shape)\nprint(test.shape)\n\ntrain = pd.merge(train, history, on='card_id', how='left')\ntest = pd.merge(test, history, on='card_id', how='left')\n\nprint(train.shape)\nprint(test.shape)\n\ntrain = pd.merge(train, new, on='card_id', how='left')\ntest = pd.merge(test, new, on='card_id', how='left')\n\nprint(train.shape)\nprint(test.shape)\n\n# train = pd.merge(train, final_group, on='card_id')\n# test = pd.merge(test, final_group, on='card_id')\n# print(train.shape)\n# print(test.shape)\ndel history\ndel new\ngc.collect()\n")


# and to define the features we want to keep to train the model:

# In[ ]:


features = [c for c in train.columns if c not in ['card_id', 'first_active_month']]
categorical_feats = [c for c in features if 'feature_' in c]

# categorical_feats = ['feature_1', 'feature_2', 'feature_3']

for col in categorical_feats:
    print(col)
    lbl = LabelEncoder()
    lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
    train[col] = lbl.transform(list(train[col].values.astype('str')))
    test[col] = lbl.transform(list(test[col].values.astype('str')))


# ## Alpha Value in Bayesian Ridge Regression
# 
# * So, if the **alpha** value is **0,** it means that it is just an **Ordinary Least Squares Regression model.** So, the **larger is the alpha**, the **higher** is the **smoothness constraint.** So, the **smaller** the value of **alpha,** the **higher would be the magnitude of the coefficients.**

# In[ ]:


from sklearn.model_selection import RepeatedKFold
folds = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4520)
oof_ridge = np.zeros(train.shape[0])
predictions_ridge = np.zeros(test.shape[0])

tst_data = test.copy()
tst_data.fillna((tst_data.mean()), inplace=True)

tst_data = tst_data[features].values

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, target)):
    print("fold n°{}".format(fold_+1))
    trn_data, trn_y = train.iloc[trn_idx][features], target.iloc[trn_idx].values
    val_data, val_y = train.iloc[val_idx][features], target.iloc[val_idx].values
    
    trn_data.fillna((trn_data.mean()), inplace=True)
    val_data.fillna((val_data.mean()), inplace=True)
    
    trn_data = trn_data.values
    val_data = val_data.values

    clf = BayesianRidge()
    clf.fit(trn_data, trn_y)
    
    oof_ridge[val_idx] = clf.predict(val_data)
    predictions_ridge += clf.predict(tst_data) / 10

np.save('oof_ridge', oof_ridge)
np.save('predictions_ridge', predictions_ridge)
np.sqrt(mean_squared_error(target.values, oof_ridge))


# In[ ]:


param = {'num_leaves': 31,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.01,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "nthread": 4,
         "random_state": 4590}


# # Light GBM
# We now train the model. Here, we use a standard KFold split of the dataset in order to validate the results and to stop the training. Interstingly, during the writing of this kernel, the model was enriched adding new features, which improved the CV score. The variations observed on the CV were found to be quite similar to the variations on the LB: it seems that the current competition won't give us headaches to define the correct validation scheme:

# In[ ]:


from sklearn.model_selection import RepeatedKFold
folds = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4520)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
start = time.time()
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx], categorical_feature=categorical_feats)
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx], categorical_feature=categorical_feats)

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=500, early_stopping_rounds = 150)
    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / 10

print("CV score: {:<8.5f}".format(mean_squared_error(oof, target)**0.5))


# # CATBOOST

# In[ ]:


# %%time
# from catboost import CatBoostRegressor
# folds = KFold(n_splits=5, shuffle=True, random_state=15)
# oof_cat = np.zeros(len(train))
# predictions_cat = np.zeros(len(test))

# for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
#     print("fold n°{}".format(fold_ + 1))
#     trn_data, trn_y = train.iloc[trn_idx][features], target.iloc[trn_idx].values
#     val_data, val_y = train.iloc[val_idx][features], target.iloc[val_idx].values
#     print("-" * 10 + "Catboost " + str(fold_) + "-" * 10)
#     cb_model = CatBoostRegressor(iterations=3000, learning_rate=0.1, depth=8, l2_leaf_reg=20, bootstrap_type='Bernoulli',  eval_metric='RMSE', metric_period=50, od_type='Iter', od_wait=45, random_seed=17, allow_writing_files=False)
#     cb_model.fit(trn_data, trn_y, eval_set=(val_data, val_y), cat_features=[], use_best_model=True, verbose=True)
    
#     oof_cat[val_idx] = cb_model.predict(val_data)
#     predictions_cat += cb_model.predict(test[features]) / folds.n_splits
    
# np.save('oof_cat', oof_cat)
# np.save('predictions_cat', predictions_cat)
# np.sqrt(mean_squared_error(target.values, oof_cat))
# gc.collect()


# # XGBOOST

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nimport xgboost as xgb\n\nxgb_params = {\'eta\': 0.005, \'max_depth\': 3, \'subsample\': 0.8, \'colsample_bytree\': 0.8, \'alpha\':0.1,\n          \'objective\': \'reg:linear\', \'eval_metric\': \'rmse\', \'silent\': True, \'random_state\':folds}\n\n\nfolds = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4520)\noof_xgb = np.zeros(len(train))\npredictions_xgb = np.zeros(len(test))\n\nfor fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):\n    print("fold n°{}".format(fold_ + 1))\n    trn_data = xgb.DMatrix(data=train.iloc[trn_idx][features], label=target.iloc[trn_idx])\n    val_data = xgb.DMatrix(data=train.iloc[val_idx][features], label=target.iloc[val_idx])\n    watchlist = [(trn_data, \'train\'), (val_data, \'valid\')]\n    print("xgb " + str(fold_) + "-" * 50)\n    num_round = 11000\n    xgb_model = xgb.train(xgb_params, trn_data, num_round, watchlist, early_stopping_rounds=50, verbose_eval=1000)\n    oof_xgb[val_idx] = xgb_model.predict(xgb.DMatrix(train.iloc[val_idx][features]), ntree_limit=xgb_model.best_ntree_limit+50)\n\n    predictions_xgb += xgb_model.predict(xgb.DMatrix(test[features]), ntree_limit=xgb_model.best_ntree_limit+50) / 10\n    \nnp.save(\'oof_xgb\', oof_xgb)\nnp.save(\'predictions_xgb\', predictions_xgb)\nprint("RMSE : ",np.sqrt(mean_squared_error(target.values, oof_xgb)))\ngc.collect()\n')


# ## 4. Feature importance <a id="4"></a> <br>
# Finally, we can have a look at the features that were used by the model:

# In[ ]:


cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,16))
sns.barplot(x="importance",
            y="feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')


# ## 5. Submission<a id="5"></a> <br>
# Now, we just need to prepare the submission file:

# ## Lightgbm

# In[ ]:


sub_df = pd.DataFrame({"card_id":test["card_id"].values})
sub_df["target"] = predictions
sub_df.to_csv("submit_lgb.csv", index=False)


# In[ ]:


sub_df = pd.DataFrame({"card_id":test["card_id"].values})
sub_df["target"] = predictions_xgb
sub_df.to_csv("submit_xgb.csv", index=False)


# ## 6. Stacking Using LightGBM<a id="6"></a> <br>

# In[ ]:


train_stack = np.vstack([oof_ridge, oof, oof_xgb]).transpose()
test_stack = np.vstack([predictions_ridge, predictions,predictions_xgb]).transpose()

folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof_stack = np.zeros(train_stack.shape[0])
predictions_stack = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_stack, target)):
    print("fold n°{}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values

    print("-" * 10 + "Ridge Regression" + str(fold_) + "-" * 10)
#     cb_model = CatBoostRegressor(iterations=3000, learning_rate=0.1, depth=8, l2_leaf_reg=20, bootstrap_type='Bernoulli',  eval_metric='RMSE', metric_period=50, od_type='Iter', od_wait=45, random_seed=17, allow_writing_files=False)
#     cb_model.fit(trn_data, trn_y, eval_set=(val_data, val_y), cat_features=[], use_best_model=True, verbose=True)
    clf = Ridge(alpha=100)
    clf.fit(trn_data, trn_y)
    
    oof_stack[val_idx] = clf.predict(val_data)
    predictions_stack += clf.predict(test_stack) / 5


np.sqrt(mean_squared_error(target.values, oof))


# In[ ]:


sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission['target'] = predictions_stack
sample_submission.to_csv('RLS.csv', index=False)

