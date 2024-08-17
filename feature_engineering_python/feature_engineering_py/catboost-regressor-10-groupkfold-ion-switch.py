#!/usr/bin/env python
# coding: utf-8

# # Necessary imports

# In[1]:


get_ipython().system(' pip install catboost')


# In[2]:


import gc
import numpy as np
import pandas as pd
from math import sqrt
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error


# # Loading the data and making bins

# In[3]:


test_clean = pd.read_csv("../input/data-without-drift/test_clean.csv")
test_clean['group'] = -1
x = [[(0,100000),(300000,400000),(800000,900000),(1000000,2000000)],[(400000,500000)], 
     [(100000,200000),(900000,1000000)],[(200000,300000),(600000,700000)],[(500000,600000),(700000,800000)]]
for k in range(5):
    for j in range(len(x[k])): test_clean.iloc[x[k][j][0]:x[k][j][1],2] = k

train_clean = pd.read_csv("../input/data-without-drift/train_clean.csv")
train_clean['group'] = -1
x = [(0,500000),(1000000,1500000),(1500000,2000000),(2500000,3000000),(2000000,2500000)]
for k in range(5): train_clean.iloc[x[k][0]:x[k][1],3] = k


# # Memory Reduction
# Else the notebook will crash due to overhead of memory.

# In[4]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# # Feature Engineering

# In[5]:


window_sizes = [10, 25, 50, 100, 500, 1000, 5000, 10000, 25000]

for window in window_sizes:
    train_clean["rolling_mean_" + str(window)] = train_clean['signal'].rolling(window=window).mean()
    #train_clean["rolling_std_" + str(window)]  = train_clean['signal'].rolling(window=window).std()
    train_clean["rolling_var_" + str(window)]  = train_clean['signal'].rolling(window=window).var()
    train_clean["rolling_min_" + str(window)]  = train_clean['signal'].rolling(window=window).min()
    train_clean["rolling_max_" + str(window)]  = train_clean['signal'].rolling(window=window).max()
    
    train_clean["rolling_min_max_ratio_" + str(window)] = train_clean["rolling_min_" + str(window)] / train_clean["rolling_max_" + str(window)]
    train_clean["rolling_min_max_diff_" + str(window)]  = train_clean["rolling_max_"  + str(window)] - train_clean["rolling_min_" + str(window)]
    
    a = (train_clean['signal'] - train_clean['rolling_min_' + str(window)]) / (train_clean['rolling_max_' + str(window)] - train_clean['rolling_min_' + str(window)])
    train_clean["norm_" + str(window)] = a * (np.floor(train_clean['rolling_max_' + str(window)]) - np.ceil(train_clean['rolling_min_' + str(window)]))
    
train_clean = train_clean.replace([np.inf, -np.inf], np.nan)
train_clean.fillna(0, inplace=True)


# In[6]:


for window in window_sizes:
    
    test_clean["rolling_mean_" + str(window)] = test_clean['signal'].rolling(window=window).mean()
    #test_clean["rolling_std_" + str(window)]  = test_clean['signal'].rolling(window=window).std()
    test_clean["rolling_var_" + str(window)]  = test_clean['signal'].rolling(window=window).var()
    test_clean["rolling_min_" + str(window)]  = test_clean['signal'].rolling(window=window).min()
    test_clean["rolling_max_" + str(window)]  = test_clean['signal'].rolling(window=window).max()
    
    test_clean["rolling_min_max_ratio_" + str(window)]  = test_clean["rolling_min_" + str(window)] /  test_clean["rolling_max_" + str(window)]
    test_clean["rolling_min_max_diff_"  + str(window)]  = test_clean["rolling_max_"  + str(window)] - test_clean["rolling_min_" + str(window)]
    
    a = (test_clean['signal'] - test_clean['rolling_min_' + str(window)]) / (test_clean['rolling_max_' + str(window)] - test_clean['rolling_min_' + str(window)])
    test_clean["norm_" + str(window)] = a * (np.floor(test_clean['rolling_max_' + str(window)]) - np.ceil(test_clean['rolling_min_' + str(window)]))
    
test_clean = test_clean.replace([np.inf, -np.inf], np.nan)
test_clean.fillna(0, inplace=True)


# In[7]:


#train_clean['signal_median'] = train_clean.groupby('group')['signal'].median()
#train_clean['signal_mean']   = train_clean.groupby('group')['signal'].mean()
#train_clean['signal_min']    = train_clean.groupby('group')['signal'].min()
#train_clean['signal_max']    = train_clean.groupby('group')['signal'].max()

train_clean['cum_sum_signal'] = train_clean['signal'].cumsum()
train_clean['cum_perc_signal']= 100*train_clean['cum_sum_signal']/train_clean['signal'].sum()


# In[8]:


#test_clean['signal_median'] = test_clean.groupby('group')['signal'].median()
#test_clean['signal_mean']   = test_clean.groupby('group')['signal'].mean()
#test_clean['signal_min']    = test_clean.groupby('group')['signal'].min()
#test_clean['signal_max']    = test_clean.groupby('group')['signal'].max()

test_clean['cum_sum_signal'] = test_clean['signal'].cumsum()
test_clean['cum_perc_signal']= 100*test_clean['cum_sum_signal']/test_clean['signal'].sum()


# In[9]:


train_clean = reduce_mem_usage(train_clean)
test_clean  = reduce_mem_usage(test_clean)


# In[10]:


train_clean.head()


# In[11]:


test_clean.head()


# In[12]:


y     = train_clean['open_channels']
train = train_clean.drop(['open_channels'],axis=1)
test  = test_clean
train.head()


# In[13]:


test.head()


# In[14]:


del train_clean   # Delete the copy of train data.
del test_clean    # Delete the copy of test data.
gc.collect()      # Collect the garbage.


# # Group KFold Technique as a CV strategy.

# In[15]:


id_train = train['time']
id_test  = test['time']

train = train.drop('time', axis = 1)
test  = test.drop( 'time', axis = 1)

nfolds = 10
groups = np.array(train.signal.values)
folds = GroupKFold(n_splits = 10)


# In[16]:


'''param = {'num_leaves': 129,
         'min_data_in_leaf': 148, 
         'objective':'regression',
         'max_depth': 9,
         'learning_rate': 0.005,
         "min_child_samples": 24,
         "boosting": "gbdt",
         "feature_fraction": 0.7202,
         "bagging_freq": 1,
         "bagging_fraction": 0.8125 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.3468,
         "verbosity": -1}'''


# # Train the model

# In[17]:


get_ipython().run_cell_magic('time', '', 'feature_importance_df = np.zeros((train.shape[1], nfolds))\nmvalid = np.zeros(len(train))\nmfull  = np.zeros(len(test))\n\nfor fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, train.values, groups)):\n    print(\'----\')\n    print("fold n°{}".format(fold_))\n    \n    x0,y0 = train.iloc[trn_idx], y[trn_idx]\n    x1,y1 = train.iloc[val_idx], y[val_idx]\n    \n    print(y0.size, y1.size)\n    \n    pd.DataFrame(y1).to_csv(\'y_test_fold\' + str(fold_) + \'.csv\', index=False)\n    \n    model = CatBoostRegressor(iterations=1000,\n                              grow_policy=\'Lossguide\',\n                              use_best_model=True,\n                              min_data_in_leaf=100, \n                              max_depth=9,\n                              learning_rate=0.01,\n                              boosting_type="Plain",\n                              subsample=0.8125,\n                              random_seed=777,\n                              loss_function=\'RMSE\',\n                              l2_leaf_reg=0.3468,\n                              verbose=-1,\n                              early_stopping_rounds=100,\n                              task_type=\'GPU\',\n                              bootstrap_type=\'Poisson\')\n    \n    model.fit(x0, y0, eval_set=(x1, y1), verbose_eval=200)\n    \n    mvalid[val_idx] = model.predict(x1)\n    pd.DataFrame(mvalid).to_csv(\'catboost_val_preds_fold\' + str(fold_) + \'.csv\', index=False)\n    \n    feature_importance_df[:, fold_] = model.feature_importances_\n    \n    mfull += model.predict(test) / folds.n_splits\n    pd.DataFrame(mfull).to_csv(\'catboost_preds_fold\' + str(fold_) + \'.csv\', index=False)\n    \nprint("RMSE: ", np.sqrt(mean_squared_error(mvalid, y)))\nprint("MAE: ", mean_absolute_error(mvalid, y))\n')


# # Plotting the feature importance

# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns

ximp = pd.DataFrame()
ximp['feature'] = train.columns
ximp['importance'] = feature_importance_df.mean(axis = 1)

plt.figure(figsize=(14,14))
sns.barplot(x="importance",
            y="feature",
            data=ximp.sort_values(by="importance",
                                           ascending=False))
plt.title('CatBoost Features (avg over folds)')
plt.tight_layout()


# In[19]:


sub = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv")
#sub.head()


# # Prepare file for the submission
# We have to conver the 'open_channels' column to make it look like a classifier problem, we used a regressor to predict the values. For this I did:
# 1. The rounding of the values predicted.
# 2. Converting the datatype of 'open_channels' from float to int.

# In[20]:


submission = pd.DataFrame()
submission['time']  = sub['time'] #id_test
submission['open_channels'] = mfull
submission['open_channels'] = submission['open_channels'].round(decimals=0)   # Round the 'open_channels' values to the nearest decimal as we implemented a regressor.
submission['open_channels'] = submission['open_channels'].astype(int)         # Convert the datatype of 'open_channels' from float to integer to match the requirements of submission.
submission.to_csv('submission.csv', index = False,float_format='%.4f')


# In[21]:


submission.tail()

