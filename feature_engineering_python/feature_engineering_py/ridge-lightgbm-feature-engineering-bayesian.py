#!/usr/bin/env python
# coding: utf-8

# ## Overview
# 
# The purpose of this kernel is to take a look at the data, come up with some insights, and attempt to create a predictive model or two. This notebook is still **very** raw. I will work on it as my very limited time permits, and hope to expend it in the upcoming days and weeks.
# 
# NB: Most of the feature engineering and some of the modeling is based on [Peter Hurford's excellent kernel.](https://www.kaggle.com/peterhurford/you-re-going-to-want-more-categories-lb-3-737/notebook) 
# 
# Inspired From : https://www.kaggle.com/tunguz/eloda-with-feature-engineering-and-stacking
# 
# ## Packages
# 
# First, let's load a few useful Python packages. This section will keep growing in subsequent versions of this EDA.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.linear_model import Ridge

from sklearn import preprocessing
import warnings
import datetime
warnings.filterwarnings("ignore")
import gc

from scipy.stats import describe
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
# Any results you write to the current directory are saved as output.


# Let's see what files we have in the input directory:

# In[ ]:


import os
print(os.listdir("../input"))


# We see that in addition to the usual,`train`, `test` and `sample_submission` files, we also have `merchants`, `historical_transactions`, `new_merchant_transactions`, and even one (**HORROR!!!**) excel file - `Data_Ditionary`. The names of the files are pretty self-explanatory, but we'll take a look at them and explore them. First, let's look at the `train` and `test` files.

# In[ ]:


get_ipython().run_cell_magic('time', '', '#Loading Train and Test Data\ntrain = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])\ntest = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])\nprint("{} observations and {} features in train set.".format(train.shape[0],train.shape[1]))\nprint("{} observations and {} features in test set.".format(test.shape[0],test.shape[1]))\ngc.collect()\n')


# In[ ]:


get_ipython().run_line_magic('time', 'train.head()')


# In[ ]:


get_ipython().run_line_magic('time', 'test.head()')


# Seems farily straightforward - just ID, first active months, three anonimous features, and target firld for train set.
# 
# Let's take a look at the target variable:
# 

# In[ ]:


get_ipython().run_line_magic('time', 'train.target.describe()')


# Seems like a very wide range of values, relatively spaking. Let's take a look at the graph of the distribution:

# In[ ]:


get_ipython().run_cell_magic('time', '', "plt.figure(figsize=(12, 5))\nplt.hist(train.target.values, bins=200)\nplt.title('Histogram target counts')\nplt.xlabel('Count')\nplt.ylabel('Target')\nplt.show()\ngc.collect()\n")


# Seems like a pretty nice normal-looking distribution, except for the few anomalous elements at teh far left. They will have to be dealt with separately.
# 
# Let's look at the "violin" version of the same plot. 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'sns.set_style("whitegrid")\nax = sns.violinplot(x=train.target.values)\nplt.show()\ngc.collect()\n')


# Yup, there is that little bump on the far left again.

# Let's now look at the distributions of various "features"

# In[ ]:


get_ipython().run_cell_magic('time', '', "plt.figure(figsize=(12, 5))\nplt.hist(train.feature_1.values, bins=50, color = 'orange')\nplt.title('Histogram feature_1 counts')\nplt.xlabel('Count')\nplt.ylabel('Target')\nplt.show()\ngc.collect()\n")


# In[ ]:


plt.figure(figsize=(12, 5))
plt.hist(train.feature_2.values, bins=50, color="green")
plt.title('Histogram feature_2 counts')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()


# In[ ]:


plt.figure(figsize=(12, 5))
plt.hist(train.feature_3.values, bins=50, color = "red")
plt.title('Histogram feature_3 counts')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()


# For now I am not including plots for the test set, as they at first approsimation look very similar.
# 
# A couple of things that stand out are:
# 
# 1. There are only a handful of values for each of the three features.
# 2. They are discrete
# 3. They are relatively eavenly distributed.
# 
# All of this suggests that these features are categorical and have been label-encoded. 

# Here is a gratuitous embedding of YouTube video of 'The Girl From Ipanema'. For no good reason.

# # Feature Engineering

# In[ ]:


get_ipython().run_cell_magic('time', '', 'train["month"] = train["first_active_month"].dt.month\ntest["month"] = test["first_active_month"].dt.month\ntrain["year"] = train["first_active_month"].dt.year\ntest["year"] = test["first_active_month"].dt.year\ntrain[\'elapsed_time\'] = (datetime.date(2018, 2, 1) - train[\'first_active_month\'].dt.date).dt.days\ntest[\'elapsed_time\'] = (datetime.date(2018, 2, 1) - test[\'first_active_month\'].dt.date).dt.days\ntrain.head()\ngc.collect()\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', "train = pd.get_dummies(train, columns=['feature_1', 'feature_2'])\ntest = pd.get_dummies(test, columns=['feature_1', 'feature_2'])\ntrain.head()\ngc.collect()\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'hist_trans = pd.read_csv("../input/historical_transactions.csv")\nhist_trans.head()\ngc.collect()\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', "hist_trans = pd.get_dummies(hist_trans, columns=['category_2', 'category_3'])\nhist_trans['authorized_flag'] = hist_trans['authorized_flag'].map({'Y': 1, 'N': 0})\nhist_trans['category_1'] = hist_trans['category_1'].map({'Y': 1, 'N': 0})\nhist_trans.head()\ngc.collect()\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "def aggregate_transactions(trans, prefix):  \n    trans.loc[:, 'purchase_date'] = pd.DatetimeIndex(trans['purchase_date']).\\\n                                      astype(np.int64) * 1e-9\n    \n    agg_func = {\n        'authorized_flag': ['sum', 'mean'],\n        'category_1': ['mean'],\n        'category_2_1.0': ['mean'],\n        'category_2_2.0': ['mean'],\n        'category_2_3.0': ['mean'],\n        'category_2_4.0': ['mean'],\n        'category_2_5.0': ['mean'],\n        'category_3_A': ['mean'],\n        'category_3_B': ['mean'],\n        'category_3_C': ['mean'],\n        'merchant_id': ['nunique'],\n        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],\n        'installments': ['sum', 'mean', 'max', 'min', 'std'],\n        'purchase_date': [np.ptp],\n        'month_lag': ['min', 'max']\n    }\n    agg_trans = trans.groupby(['card_id']).agg(agg_func)\n    agg_trans.columns = [prefix + '_'.join(col).strip() \n                           for col in agg_trans.columns.values]\n    agg_trans.reset_index(inplace=True)\n    \n    df = (trans.groupby('card_id')\n          .size()\n          .reset_index(name='{}transactions_count'.format(prefix)))\n    \n    agg_trans = pd.merge(df, agg_trans, on='card_id', how='left')\n    \n    return agg_trans\ngc.collect()\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "import gc\nmerch_hist = aggregate_transactions(hist_trans, prefix='hist_')\ndel hist_trans\ngc.collect()\ntrain = pd.merge(train, merch_hist, on='card_id',how='left')\ntest = pd.merge(test, merch_hist, on='card_id',how='left')\ndel merch_hist\ngc.collect()\ntrain.head()\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'new_trans = pd.read_csv("../input/new_merchant_transactions.csv")\nnew_trans.head()\ngc.collect()\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', "new_trans = pd.get_dummies(new_trans, columns=['category_2', 'category_3'])\nnew_trans['authorized_flag'] = new_trans['authorized_flag'].map({'Y': 1, 'N': 0})\nnew_trans['category_1'] = new_trans['category_1'].map({'Y': 1, 'N': 0})\nnew_trans.head()\ngc.collect()\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "merch_new = aggregate_transactions(new_trans, prefix='new_')\ndel new_trans\ngc.collect()\ntrain = pd.merge(train, merch_new, on='card_id',how='left')\ntest = pd.merge(test, merch_new, on='card_id',how='left')\ndel merch_new\ngc.collect()\ntrain.head()\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "target = train['target']\ndrops = ['card_id', 'first_active_month', 'target']\nuse_cols = [c for c in train.columns if c not in drops]\nfeatures = list(train[use_cols].columns)\ntrain[features].head()\ngc.collect()\n")


# In[ ]:


print(train[features].shape)
print(test[features].shape)
gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', "train[features+['target']].to_csv('new_train.csv', index=False)\ntest[features].to_csv('new_test.csv', index=False)\ngc.collect()\n")


# # Modeling
# 
# Now let's do some of what everyone is here for - modeling. We'll start with a simple Ridge regression model. 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'folds = KFold(n_splits=7, shuffle=True, random_state=15)\noof_ridge = np.zeros(train.shape[0])\npredictions_ridge = np.zeros(test.shape[0])\n\ntst_data = test.copy()\ntst_data.fillna((tst_data.mean()), inplace=True)\n\ntst_data = tst_data[features].values\n\nfor fold_, (trn_idx, val_idx) in enumerate(folds.split(train, target)):\n    print("fold n°{}".format(fold_+1))\n    trn_data, trn_y = train.iloc[trn_idx][features], target.iloc[trn_idx].values\n    val_data, val_y = train.iloc[val_idx][features], target.iloc[val_idx].values\n    \n    trn_data.fillna((trn_data.mean()), inplace=True)\n    val_data.fillna((val_data.mean()), inplace=True)\n    \n    trn_data = trn_data.values\n    val_data = val_data.values\n\n    clf = Ridge(alpha=100)\n    clf.fit(trn_data, trn_y)\n    \n    oof_ridge[val_idx] = clf.predict(val_data)\n    predictions_ridge += clf.predict(tst_data) / folds.n_splits\n\nnp.save(\'oof_ridge\', oof_ridge)\nnp.save(\'predictions_ridge\', predictions_ridge)\nnp.sqrt(mean_squared_error(target.values, oof_ridge))\ngc.collect()\n')


# 3.83 CV is not bad, but it's far from what the best models can do in this competition. Let's take a look at a few non-linear models. We'll start with LightGBM.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'param = {\'num_leaves\': 50,\n         \'min_data_in_leaf\': 30, \n         \'objective\':\'regression\',\n         \'max_depth\': -1,\n         \'learning_rate\': 0.005,\n         "boosting": "gbdt",\n         "feature_fraction": 0.9,\n         "bagging_freq": 1,\n         "bagging_fraction": 0.9,\n         "bagging_seed": 11,\n         "metric": \'rmse\',\n         "lambda_l1": 0.1,\n         "verbosity": -1}\n\nfolds = KFold(n_splits=5, shuffle=True, random_state=15)\noof_lgb = np.zeros(len(train))\npredictions_lgb = np.zeros(len(test))\n\nfor fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):\n    print(\'-\')\n    print("Fold {}".format(fold_ + 1))\n    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx])\n    val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx])\n\n    num_round = 10000\n    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds=100)\n    oof_lgb[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)\n    predictions_lgb += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits\n    \nnp.save(\'oof_lgb\', oof_lgb)\nnp.save(\'predictions_lgb\', predictions_lgb)\nnp.sqrt(mean_squared_error(target.values, oof_lgb))\ngc.collect()\n')


# Now soem XGBoost:

# In[ ]:


get_ipython().run_cell_magic('time', '', 'xgb_params = {\'eta\': 0.005, \'max_depth\': 10, \'subsample\': 0.8, \'colsample_bytree\': 0.8, \n          \'objective\': \'reg:linear\', \'eval_metric\': \'rmse\', \'silent\': True}\n\nfolds = KFold(n_splits=5, shuffle=True, random_state=15)\noof_xgb = np.zeros(len(train))\npredictions_xgb = np.zeros(len(test))\n\nfor fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):\n    print(\'-\')\n    print("Fold {}".format(fold_ + 1))\n    trn_data = xgb.DMatrix(data=train.iloc[trn_idx][features], label=target.iloc[trn_idx])\n    val_data = xgb.DMatrix(data=train.iloc[val_idx][features], label=target.iloc[val_idx])\n    watchlist = [(trn_data, \'train\'), (val_data, \'valid\')]\n    print("xgb " + str(fold_) + "-" * 50)\n    num_round = 10000\n    xgb_model = xgb.train(xgb_params, trn_data, num_round, watchlist, early_stopping_rounds=50, verbose_eval=1000)\n    oof_xgb[val_idx] = xgb_model.predict(xgb.DMatrix(train.iloc[val_idx][features]), ntree_limit=xgb_model.best_ntree_limit+50)\n\n    predictions_xgb += xgb_model.predict(xgb.DMatrix(test[features]), ntree_limit=xgb_model.best_ntree_limit+50) / folds.n_splits\n    \nnp.save(\'oof_xgb\', oof_xgb)\nnp.save(\'predictions_xgb\', predictions_xgb)\nnp.sqrt(mean_squared_error(target.values, oof_xgb))\ngc.collect()\n')


# # Second Set of Features and Models

# A great thing about stackign is that you can not only use different set of models, but also create the same models with a different set of features. Here we'll use features from the wonderful [Elo world kernel](https://www.kaggle.com/fabiendaniel/elo-world) by Fabien Daniel:

# In[ ]:


get_ipython().run_cell_magic('time', '', 'del train, test\ngc.collect()\n')


# In[ ]:


new_transactions = pd.read_csv('../input/new_merchant_transactions.csv', parse_dates=['purchase_date'])
historical_transactions = pd.read_csv('../input/historical_transactions.csv', parse_dates=['purchase_date'])

def binarize(df):
    for col in ['authorized_flag', 'category_1']:
        df[col] = df[col].map({'Y':1, 'N':0})
    return df

historical_transactions = binarize(historical_transactions)
new_transactions = binarize(new_transactions)


# In[ ]:


def read_data(input_file):
    df = pd.read_csv(input_file)
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days
    return df
#_________________________________________
train = read_data('../input/train.csv')
test = read_data('../input/test.csv')

target = train['target']
del train['target']


# In[ ]:


historical_transactions = pd.get_dummies(historical_transactions, columns=['category_2', 'category_3'])
new_transactions = pd.get_dummies(new_transactions, columns=['category_2', 'category_3'])

historical_transactions = reduce_mem_usage(historical_transactions)
new_transactions = reduce_mem_usage(new_transactions)


# In[ ]:


historical_transactions['purchase_month'] = historical_transactions['purchase_date'].dt.month
new_transactions['purchase_month'] = new_transactions['purchase_date'].dt.month

def aggregate_transactions(history):
    
    history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']).\
                                      astype(np.int64) * 1e-9
    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'category_1': ['sum', 'mean'],
        'category_2_1.0': ['mean'],
        'category_2_2.0': ['mean'],
        'category_2_3.0': ['mean'],
        'category_2_4.0': ['mean'],
        'category_2_5.0': ['mean'],
        'category_3_A': ['mean'],
        'category_3_B': ['mean'],
        'category_3_C': ['mean'],
        'merchant_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'state_id': ['nunique'],
        'city_id': ['nunique'],
        'subsector_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        'purchase_month': ['mean', 'max', 'min', 'std'],
        'purchase_date': [np.ptp],
        'month_lag': ['min', 'max']
        }
    
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    
    df = (history.groupby('card_id')
          .size()
          .reset_index(name='transactions_count'))
    
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    
    return agg_history


# In[ ]:


history = aggregate_transactions(historical_transactions)
history.columns = ['hist_' + c if c != 'card_id' else c for c in history.columns]
history[:5]


# In[ ]:


new = aggregate_transactions(new_transactions)
new.columns = ['new_' + c if c != 'card_id' else c for c in new.columns]
new[:5]


# In[ ]:


def aggregate_per_month(history):
    grouped = history.groupby(['card_id', 'month_lag'])

    agg_func = {
            'purchase_amount': ['count', 'sum', 'mean', 'min', 'max', 'std'],
            'installments': ['count', 'sum', 'mean', 'min', 'max', 'std'],
            }

    intermediate_group = grouped.agg(agg_func)
    intermediate_group.columns = ['_'.join(col).strip() for col in intermediate_group.columns.values]
    intermediate_group.reset_index(inplace=True)

    final_group = intermediate_group.groupby('card_id').agg(['mean', 'std'])
    final_group.columns = ['_'.join(col).strip() for col in final_group.columns.values]
    final_group.reset_index(inplace=True)
    
    return final_group
#___________________________________________________________
final_group =  aggregate_per_month(historical_transactions) 
final_group[:10]


# In[ ]:


train = pd.merge(train, history, on='card_id', how='left')
test = pd.merge(test, history, on='card_id', how='left')

train = pd.merge(train, new, on='card_id', how='left')
test = pd.merge(test, new, on='card_id', how='left')

train = pd.merge(train, final_group, on='card_id')
test = pd.merge(test, final_group, on='card_id')

features = [c for c in train.columns if c not in ['card_id', 'first_active_month']]
categorical_feats = [c for c in features if 'feature_' in c]


# In[ ]:


train[features+['target']].to_csv('new_train_2.csv', index=False)
test[features].to_csv('new_test_2.csv', index=False)


# In[ ]:


param = {'num_leaves': 50,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.005,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1}

folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof_lgb_2 = np.zeros(len(train))
predictions_lgb_2 = np.zeros(len(test))
start = time.time()
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx], categorical_feature=categorical_feats)
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx], categorical_feature=categorical_feats)

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)
    oof_lgb_2[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions_lgb_2 += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

np.save('oof_lgb_2', oof_lgb_2)
np.save('predictions_lgb_2', predictions_lgb_2)
print("CV score: {:<8.5f}".format(mean_squared_error(oof_lgb_2, target)**0.5))


# In[ ]:


'''xgb_params = {'eta': 0.005, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8, 
          'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True}

folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof_xgb_2 = np.zeros(len(train))
predictions_xgb_2 = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print('-')
    print("Fold {}".format(fold_ + 1))
    trn_data = xgb.DMatrix(data=train.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = xgb.DMatrix(data=train.iloc[val_idx][features], label=target.iloc[val_idx])
    watchlist = [(trn_data, 'train'), (val_data, 'valid')]
    print("xgb " + str(fold_) + "-" * 50)
    num_round = 10000
    xgb_model = xgb.train(xgb_params, trn_data, num_round, watchlist, early_stopping_rounds=50, verbose_eval=1000)
    oof_xgb_2[val_idx] = xgb_model.predict(xgb.DMatrix(train.iloc[val_idx][features]), ntree_limit=xgb_model.best_ntree_limit+50)

    predictions_xgb_2 += xgb_model.predict(xgb.DMatrix(test[features]), ntree_limit=xgb_model.best_ntree_limit+50) / folds.n_splits
    
np.save('oof_xgb_2', oof_xgb_2)
np.save('predictions_xgb_2', predictions_xgb_2)
np.sqrt(mean_squared_error(target.values, oof_xgb_2))'''


# Finally, we'll stack them all together:

# In[ ]:


train_stack = np.vstack([oof_ridge, oof_lgb, oof_xgb, oof_lgb_2]).transpose()
test_stack = np.vstack([predictions_ridge, predictions_lgb, predictions_xgb, 
                        predictions_lgb_2]).transpose()

folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_stack, target)):
    print("fold n°{}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values

    clf = Ridge(alpha=1)
    clf.fit(trn_data, trn_y)
    
    oof[val_idx] = clf.predict(val_data)
    predictions += clf.predict(test_stack) / folds.n_splits


np.sqrt(mean_squared_error(target.values, oof))


# In[ ]:


sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission['target'] = predictions
sample_submission.to_csv('stacker_2.csv', index=False)

