#!/usr/bin/env python
# coding: utf-8

# ## General information
# 
# In this we work with the data from ventilators connected to a sedated patient's lung. Our goal is to similate the ventilator and correctly predict the airway pressure in the respiratory circuit during the breath.
# 
# It is important to understand that this isn't a usual time-series competition: in time-series tasks we need to predict the future values of the series based on the previous values. Here we use the values of different series to predict the values of other series. So this is a regression task.
# 
# 
#  ![](https://www.nhlbi.nih.gov/sites/default/files/inline-images/19-1096-NHLBI-OY1-Q41-ES-Ventilator-Support_900px_dev1.jpg)

# In[1]:


# libraries
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os
import numpy as np
import time
import lightgbm as lgb

from sklearn.model_selection import GroupKFold
from sklearn import metrics


# ## Data loading and overview

# In[2]:


path = '../input/ventilator-pressure-prediction'
train = pd.read_csv(os.path.join(path, 'train.csv'))
test = pd.read_csv(os.path.join(path, 'test.csv'))
sub = pd.read_csv(os.path.join(path, 'sample_submission.csv'))


# In[3]:


train.head()


# In[4]:


train.describe()


# The dataset is quite large - more than 6 mln rows. There are 2 categorical features (R and C), the id of a row, the id of a breath, timestamp and continuous variables.

# In[5]:


print(set(test['breath_id'].unique()).intersection(set(train['breath_id'].unique())))
print(set(train['breath_id'].unique()).intersection(set(test['breath_id'].unique())))


# Here is a very **important** point:
# The breath ids in train and test don't overlap! This means we should use GroupKFold validation using this variable.

# In[6]:


fig, ax = plt.subplots(figsize = (12, 8))
plt.subplot(2, 2, 1)
sns.countplot(x='R', data=train)
plt.title('Counts of R in train');
plt.subplot(2, 2, 2)
sns.countplot(x='R', data=test)
plt.title('Counts of R in test');
plt.subplot(2, 2, 3)
sns.countplot(x='C', data=train)
plt.title('Counts of C in train');
plt.subplot(2, 2, 4)
sns.countplot(x='C', data=test)
plt.title('Counts of C in test');


# `R` and `C` categorical variables have a similar distribution in train and test data.

# Now, let's have a look at one of the series in the data.

# In[7]:


fig, ax1 = plt.subplots(figsize = (12, 8))

breath_1 = train.loc[train['breath_id'] == 1]
ax2 = ax1.twinx()

ax1.plot(breath_1['time_step'], breath_1['pressure'], 'r-', label='pressure')
ax1.plot(breath_1['time_step'], breath_1['u_in'], 'g-', label='u_in')
ax2.plot(breath_1['time_step'], breath_1['u_out'], 'b-', label='u_out')

ax1.set_xlabel('Timestep')

ax1.legend(loc=(1.1, 0.8))
ax2.legend(loc=(1.1, 0.7))
plt.show()


# This is quite interesting: we can see that at first the `pressure` (our target) is rising and then, after the `u_out` becomes equal to 1, it has an abrupt drop. I think it would be useful to create new features based on the behavior of these features.

# ## Feature engineering

# In[8]:


# rewritten calculation of lag features from this notebook: https://www.kaggle.com/patrick0302/add-lag-u-in-as-new-feat
# some of ideas from this notebook: https://www.kaggle.com/mst8823/google-brain-lightgbm-baseline
train['last_value_u_in'] = train.groupby('breath_id')['u_in'].transform('last')
train['u_in_lag1'] = train.groupby('breath_id')['u_in'].shift(1)
train['u_out_lag1'] = train.groupby('breath_id')['u_out'].shift(1)
train['u_in_lag_back1'] = train.groupby('breath_id')['u_in'].shift(-1)
train['u_out_lag_back1'] = train.groupby('breath_id')['u_out'].shift(-1)
train['u_in_lag2'] = train.groupby('breath_id')['u_in'].shift(2)
train['u_out_lag2'] = train.groupby('breath_id')['u_out'].shift(2)
train['u_in_lag_back2'] = train.groupby('breath_id')['u_in'].shift(-2)
train['u_out_lag_back2'] = train.groupby('breath_id')['u_out'].shift(-2)
train['u_in_lag3'] = train.groupby('breath_id')['u_in'].shift(3)
train['u_out_lag3'] = train.groupby('breath_id')['u_out'].shift(3)
train['u_in_lag_back3'] = train.groupby('breath_id')['u_in'].shift(-3)
train['u_out_lag_back3'] = train.groupby('breath_id')['u_out'].shift(-3)
train = train.fillna(0)


train['R__C'] = train["R"].astype(str) + '__' + train["C"].astype(str)

# max value of u_in and u_out for each breath
train['breath_id__u_in__max'] = train.groupby(['breath_id'])['u_in'].transform('max')
train['breath_id__u_out__max'] = train.groupby(['breath_id'])['u_out'].transform('max')

# difference between consequitive values
train['u_in_diff1'] = train['u_in'] - train['u_in_lag1']
train['u_out_diff1'] = train['u_out'] - train['u_out_lag1']
train['u_in_diff2'] = train['u_in'] - train['u_in_lag2']
train['u_out_diff2'] = train['u_out'] - train['u_out_lag2']
# from here: https://www.kaggle.com/yasufuminakama/ventilator-pressure-lstm-starter
train.loc[train['time_step'] == 0, 'u_in_diff'] = 0
train.loc[train['time_step'] == 0, 'u_out_diff'] = 0

# difference between the current value of u_in and the max value within the breath
train['breath_id__u_in__diffmax'] = train.groupby(['breath_id'])['u_in'].transform('max') - train['u_in']
train['breath_id__u_in__diffmean'] = train.groupby(['breath_id'])['u_in'].transform('mean') - train['u_in']

# OHE
train = train.merge(pd.get_dummies(train['R'], prefix='R'), left_index=True, right_index=True).drop(['R'], axis=1)
train = train.merge(pd.get_dummies(train['C'], prefix='C'), left_index=True, right_index=True).drop(['C'], axis=1)
train = train.merge(pd.get_dummies(train['R__C'], prefix='R__C'), left_index=True, right_index=True).drop(['R__C'], axis=1)

# https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/273974
train['u_in_cumsum'] = train.groupby(['breath_id'])['u_in'].cumsum()
train['time_step_cumsum'] = train.groupby(['breath_id'])['time_step'].cumsum()


# In[9]:


# all the same for the test data
test['last_value_u_in'] = test.groupby('breath_id')['u_in'].transform('last')
test['u_in_lag1'] = test.groupby('breath_id')['u_in'].shift(1)
test['u_out_lag1'] = test.groupby('breath_id')['u_out'].shift(1)
test['u_in_lag_back1'] = test.groupby('breath_id')['u_in'].shift(-1)
test['u_out_lag_back1'] = test.groupby('breath_id')['u_out'].shift(-1)
test['u_in_lag2'] = test.groupby('breath_id')['u_in'].shift(2)
test['u_out_lag2'] = test.groupby('breath_id')['u_out'].shift(2)
test['u_in_lag_back2'] = test.groupby('breath_id')['u_in'].shift(-2)
test['u_out_lag_back2'] = test.groupby('breath_id')['u_out'].shift(-2)
test['u_in_lag3'] = test.groupby('breath_id')['u_in'].shift(3)
test['u_out_lag3'] = test.groupby('breath_id')['u_out'].shift(3)
test['u_in_lag_back3'] = test.groupby('breath_id')['u_in'].shift(-3)
test['u_out_lag_back3'] = test.groupby('breath_id')['u_out'].shift(-3)
test = test.fillna(0)
test['R__C'] = test["R"].astype(str) + '__' + test["C"].astype(str)

test['breath_id__u_in__max'] = test.groupby(['breath_id'])['u_in'].transform('max')
test['breath_id__u_out__max'] = test.groupby(['breath_id'])['u_out'].transform('max')

test['u_in_diff1'] = test['u_in'] - test['u_in_lag1']
test['u_out_diff1'] = test['u_out'] - test['u_out_lag1']
test['u_in_diff2'] = test['u_in'] - test['u_in_lag2']
test['u_out_diff2'] = test['u_out'] - test['u_out_lag2']
test.loc[test['time_step'] == 0, 'u_in_diff'] = 0
test.loc[test['time_step'] == 0, 'u_out_diff'] = 0

test['breath_id__u_in__diffmax'] = test.groupby(['breath_id'])['u_in'].transform('max') - test['u_in']
test['breath_id__u_in__diffmean'] = test.groupby(['breath_id'])['u_in'].transform('mean') - test['u_in']

test = test.merge(pd.get_dummies(test['R'], prefix='R'), left_index=True, right_index=True).drop(['R'], axis=1)
test = test.merge(pd.get_dummies(test['C'], prefix='C'), left_index=True, right_index=True).drop(['C'], axis=1)
test = test.merge(pd.get_dummies(test['R__C'], prefix='R__C'), left_index=True, right_index=True).drop(['R__C'], axis=1)

test['u_in_cumsum'] = test.groupby(['breath_id'])['u_in'].cumsum()
test['time_step_cumsum'] = test.groupby(['breath_id'])['time_step'].cumsum()


# ## Model training

# In[10]:


scores = []
feature_importance = pd.DataFrame()
models = []
columns = [col for col in train.columns if col not in ['id', 'breath_id', 'pressure']]
X = train[columns]
y = train['pressure']


# In[11]:


params = {'objective': 'regression',
          'learning_rate': 0.3,
          "boosting_type": "gbdt",
          "metric": 'mae',
          'n_jobs': -1,
          'min_data_in_leaf':32,
          'num_leaves':1024,
         }


# In[12]:


folds = GroupKFold(n_splits=5)
for fold_n, (train_index, valid_index) in enumerate(folds.split(train, y, groups=train['breath_id'])):
    print(f'Fold {fold_n} started at {time.ctime()}')
    X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    model = lgb.LGBMRegressor(**params, n_estimators=10000)
    model.fit(X_train, y_train, 
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=1000, early_stopping_rounds=15)
    score = metrics.mean_absolute_error(y_valid, model.predict(X_valid))
    
    models.append(model)
    scores.append(score)

    fold_importance = pd.DataFrame()
    fold_importance["feature"] = columns
    fold_importance["importance"] = model.feature_importances_
    fold_importance["fold"] = fold_n + 1
    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)


# In[13]:


print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))


# In[14]:


feature_importance["importance"] /= 5
cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
    by="importance", ascending=False)[:50].index

best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

plt.figure(figsize=(16, 12));
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
plt.title('LGB Features (avg over folds)');


# ## Making predictions

# In[15]:


for model in models:
    sub['pressure'] += model.predict(test[columns])
sub['pressure'] /= 5


# In[16]:


sub.to_csv('first_sub.csv', index=False)

