#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.metrics import mean_squared_log_error as msle
from sklearn.model_selection import KFold

sns.set_theme(style = 'white', palette = 'viridis')
pal = sns.color_palette('viridis')


# In[2]:


train = pd.read_csv(r'../input/playground-series-s3e11/train.csv')
test_1 = pd.read_csv(r'../input/playground-series-s3e11/test.csv')
orig_train = pd.read_csv(r'../input/media-campaign-cost-prediction/train_dataset.csv')
orig_test = pd.read_csv(r'../input/media-campaign-cost-prediction/test_dataset.csv')

train.drop('id', axis = 1, inplace = True)
test = test_1.drop('id', axis = 1)


# # Null Values and Duplicates

# In[3]:


print(f'There are {train.isna().sum().sum()} null values in train dataset')
print(f'There are {test.isna().sum().sum()} null values in test dataset')
print(f'There are {orig_train.isna().sum().sum()} null values in original train dataset')
print(f'There are {orig_test.isna().sum().sum()} null values in original test dataset\n')

print(f'There are {train.duplicated(subset = list(train)[0:-1]).value_counts()[0]} non-duplicate values out of {train.count()[0]} rows in train dataset')
print(f'There are {test.duplicated().value_counts()[0]} non-duplicate values out of {test.count()[0]} rows in test dataset')
print(f'There are {orig_train.duplicated(subset = list(train)[0:-1]).value_counts()[0]} non-duplicate values out of {orig_train.count()[0]} rows in original train dataset')
print(f'There are {orig_test.duplicated().value_counts()[0]} non-duplicate values out of {orig_test.count()[0]} rows in original test dataset')


# In[4]:


orig_train.drop_duplicates(subset = list(train)[0:-1], inplace = True)


# # Correlation

# In[5]:


def heatmap(dataset, label = None):
    corr = dataset.corr()
    plt.figure(figsize = (14, 10), dpi = 300)
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask = mask, annot = True, annot_kws = {'size' : 7}, cmap = 'viridis')
    plt.yticks(fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.title(f'{label} Dataset Correlation Matrix\n', fontsize = 25, weight = 'bold')
    plt.show()


# In[6]:


heatmap(train, 'Train')
heatmap(test, 'Test')


# **Key points:** 
# 1. There is perfect correlation between `salad_bar` and `prepared_food`. This means that all prepared foods come from salad bar. We must drop one of them in any case.
# 2. For the purpose of engineering, I will create two new features: facilities and children ratio. Facilities is based on correlation between categorical features, while children ratio is based on correlation between `num_children_at_home` and `total_children`.

# In[7]:


train = pd.concat([train, orig_train], axis = 0)

train.drop('salad_bar', axis = 1, inplace = True)
test.drop('salad_bar', axis = 1, inplace = True)


# # Model I: Base Model

# In[8]:


X = train.copy()
y = X.pop('cost')

seed = 42
splits = 10
repeats = 1 

np.random.seed(seed)


# In[9]:


xgb_params = {
    'seed': seed,
    'objective': 'reg:squaredlogerror',
    'eval_metric': 'rmsle',
    'tree_method' : 'gpu_hist',
    'n_jobs' : -1,
}

predictions = np.zeros(len(test))
train_scores, val_scores = [], []
k = KFold(n_splits=splits, random_state=seed, shuffle=True)
for fold, (train_idx, val_idx) in enumerate(k.split(X, y)):
    
    dtrain = xgb.DMatrix(
        data=X.iloc[train_idx], 
        label=y.iloc[train_idx]
    )
    
    dvalid = xgb.DMatrix(
        data=X.iloc[val_idx], 
        label=y.iloc[val_idx]
    )

    xgb_model = xgb.train(
        params=xgb_params, 
        dtrain=dtrain,
        verbose_eval = False,
        num_boost_round = 1000,
        evals=[(dtrain, 'train'), 
               (dvalid, 'eval')], 
        callbacks=[xgb.callback.EarlyStopping(rounds=100,
                                              data_name='eval',
                                              maximize=False,
                                              save_best=True)]
    )
    
    best_iter = xgb_model.best_iteration
    
    train_preds = xgb_model.predict(dtrain)
    val_preds = xgb_model.predict(dvalid)
    
    train_score = msle(y.iloc[train_idx], train_preds, squared = False)
    val_score = msle(y.iloc[val_idx], val_preds, squared = False)
    
    train_scores.append(train_score)
    val_scores.append(val_score)
    
    predictions += xgb_model.predict(xgb.DMatrix(test)) / splits
    print(f'Fold {fold}: val RMSLE = {val_score:.5f} | train RMSLE = {train_score:.5f} | best_iter = {best_iter}')

print(f'Average val RMSLE = {np.mean(val_scores):.5f} | train RMSLE = {np.mean(train_scores):.5f}')
xgb_preds = predictions.copy()


# # Model II: Feature Selection Only

# In[10]:


feat_info = pd.DataFrame.from_dict(xgb_model.get_score(importance_type = 'gain'), orient = 'index')

plt.figure(figsize = (10,6), dpi = 300)
sns.barplot(feat_info.sort_values(ascending = False, by = 0).T, orient = 'h', palette = 'viridis')
plt.title('Feature Importance')
plt.show()


# **Key points:** Unit sales features and the rest below has significantly worse importance than the rest. We should try to drop them

# In[11]:


drop_features = ['low_fat', 'gross_weight', 'recyclable_package', 'store_sales(in millions)', 'units_per_case', 'unit_sales(in millions)']

X.drop(drop_features, axis = 1, inplace = True)
test.drop(drop_features, axis = 1, inplace = True)


# In[12]:


xgb_params = {
    'seed': seed,
    'objective': 'reg:squaredlogerror',
    'eval_metric': 'rmsle',
    'tree_method' : 'gpu_hist',
    'n_jobs' : -1,
}

predictions = np.zeros(len(test))
train_scores, val_scores = [], []
k = KFold(n_splits=splits, random_state=seed, shuffle=True)
for fold, (train_idx, val_idx) in enumerate(k.split(X, y)):
    
    dtrain = xgb.DMatrix(
        data=X.iloc[train_idx], 
        label=y.iloc[train_idx]
    )
    
    dvalid = xgb.DMatrix(
        data=X.iloc[val_idx], 
        label=y.iloc[val_idx]
    )

    xgb_model = xgb.train(
        params=xgb_params, 
        dtrain=dtrain,
        verbose_eval = False,
        num_boost_round = 1000,
        evals=[(dtrain, 'train'), 
               (dvalid, 'eval')], 
        callbacks=[xgb.callback.EarlyStopping(rounds=100,
                                              data_name='eval',
                                              maximize=False,
                                              save_best=True)]
    )
    
    best_iter = xgb_model.best_iteration
    
    train_preds = xgb_model.predict(dtrain)
    val_preds = xgb_model.predict(dvalid)
    
    train_score = msle(y.iloc[train_idx], train_preds, squared = False)
    val_score = msle(y.iloc[val_idx], val_preds, squared = False)
    
    train_scores.append(train_score)
    val_scores.append(val_score)
    
    predictions += xgb_model.predict(xgb.DMatrix(test)) / splits
    print(f'Fold {fold}: val RMSLE = {val_score:.5f} | train RMSLE = {train_score:.5f} | best_iter = {best_iter}')

print(f'Average val RMSLE = {np.mean(val_scores):.5f} | train RMSLE = {np.mean(train_scores):.5f}')
xgb_preds = predictions.copy()


# # Model III: Feature Engineering Only

# In[13]:


X = train.copy()
y = X.pop('cost')

test = test_1.drop(['id', 'salad_bar'], axis = 1)

X['child_ratio'] = X.eval('total_children / num_children_at_home')
X.replace([np.inf, -np.inf], 10, inplace = True)
X.fillna(0, inplace = True)

test['child_ratio'] = test.eval('total_children / num_children_at_home')
test.replace([np.inf, -np.inf], 10, inplace = True)
test.fillna(0, inplace = True)

X['facilities'] = X.eval('coffee_bar + video_store + prepared_food + florist')
test['facilities'] = test.eval('coffee_bar + video_store + prepared_food + florist')


# In[14]:


xgb_params = {
    'seed': seed,
    'objective': 'reg:squaredlogerror',
    'eval_metric': 'rmsle',
    'tree_method' : 'gpu_hist',
    'n_jobs' : -1,
}

predictions = np.zeros(len(test))
train_scores, val_scores = [], []
k = KFold(n_splits=splits, random_state=seed, shuffle=True)
for fold, (train_idx, val_idx) in enumerate(k.split(X, y)):
    
    dtrain = xgb.DMatrix(
        data=X.iloc[train_idx], 
        label=y.iloc[train_idx]
    )
    
    dvalid = xgb.DMatrix(
        data=X.iloc[val_idx], 
        label=y.iloc[val_idx]
    )

    xgb_model = xgb.train(
        params=xgb_params, 
        dtrain=dtrain,
        verbose_eval = False,
        num_boost_round = 1000,
        evals=[(dtrain, 'train'), 
               (dvalid, 'eval')], 
        callbacks=[xgb.callback.EarlyStopping(rounds=100,
                                              data_name='eval',
                                              maximize=False,
                                              save_best=True)]
    )
    
    best_iter = xgb_model.best_iteration
    
    train_preds = xgb_model.predict(dtrain)
    val_preds = xgb_model.predict(dvalid)
    
    train_score = msle(y.iloc[train_idx], train_preds, squared = False)
    val_score = msle(y.iloc[val_idx], val_preds, squared = False)
    
    train_scores.append(train_score)
    val_scores.append(val_score)
    
    predictions += xgb_model.predict(xgb.DMatrix(test)) / splits
    print(f'Fold {fold}: val RMSLE = {val_score:.5f} | train RMSLE = {train_score:.5f} | best_iter = {best_iter}')

print(f'Average val RMSLE = {np.mean(val_scores):.5f} | train RMSLE = {np.mean(train_scores):.5f}')
xgb_preds = predictions.copy()


# # Model IV: Feature Selection + Feature Engineering

# In[15]:


feat_info = pd.DataFrame.from_dict(xgb_model.get_score(importance_type = 'gain'), orient = 'index')

plt.figure(figsize = (10,6), dpi = 300)
sns.barplot(feat_info.sort_values(ascending = False, by = 0).T, orient = 'h', palette = 'viridis')
plt.title('Feature Importance')
plt.show()


# **Key points:** It's very similar to the previous plot so we'll just remove the same features.

# In[16]:


drop_features = ['low_fat', 'gross_weight', 'recyclable_package', 'store_sales(in millions)', 'units_per_case', 'unit_sales(in millions)']

X.drop(drop_features, axis = 1, inplace = True)
test.drop(drop_features, axis = 1, inplace = True)


# In[17]:


xgb_params = {
    'seed': seed,
    'objective': 'reg:squaredlogerror',
    'eval_metric': 'rmsle',
    'tree_method' : 'gpu_hist',
    'n_jobs' : -1,
}

predictions = np.zeros(len(test))
train_scores, val_scores = [], []
k = KFold(n_splits=splits, random_state=seed, shuffle=True)
for fold, (train_idx, val_idx) in enumerate(k.split(X, y)):
    
    dtrain = xgb.DMatrix(
        data=X.iloc[train_idx], 
        label=y.iloc[train_idx]
    )
    
    dvalid = xgb.DMatrix(
        data=X.iloc[val_idx], 
        label=y.iloc[val_idx]
    )

    xgb_model = xgb.train(
        params=xgb_params, 
        dtrain=dtrain,
        verbose_eval = False,
        num_boost_round = 1000,
        evals=[(dtrain, 'train'), 
               (dvalid, 'eval')], 
        callbacks=[xgb.callback.EarlyStopping(rounds=100,
                                              data_name='eval',
                                              maximize=False,
                                              save_best=True)]
    )
    
    best_iter = xgb_model.best_iteration
    
    train_preds = xgb_model.predict(dtrain)
    val_preds = xgb_model.predict(dvalid)
    
    train_score = msle(y.iloc[train_idx], train_preds, squared = False)
    val_score = msle(y.iloc[val_idx], val_preds, squared = False)
    
    train_scores.append(train_score)
    val_scores.append(val_score)
    
    predictions += xgb_model.predict(xgb.DMatrix(test)) / splits
    print(f'Fold {fold}: val RMSLE = {val_score:.5f} | train RMSLE = {train_score:.5f} | best_iter = {best_iter}')

print(f'Average val RMSLE = {np.mean(val_scores):.5f} | train RMSLE = {np.mean(train_scores):.5f}')
xgb_preds = predictions.copy()


# # Submission

# In[18]:


test_1.drop(list(test_1.drop('id', axis = 1)), axis = 1, inplace = True)

test_1['cost'] = xgb_preds
test_1.to_csv('submission.csv', index = False)


# # Summary
# 
# Both feature selection and feature engineering can help our CV when used together.
# 
# **Additional note:** I actually tried to create new features that is average weight but apparently, it doesn't help the CV because it's importance is still low. I assume that engineering features with low importance will not help our model.

# Thank you for reading!
