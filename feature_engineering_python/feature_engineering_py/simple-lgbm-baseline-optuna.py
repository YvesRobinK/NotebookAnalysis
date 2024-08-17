#!/usr/bin/env python
# coding: utf-8

# ## Overview
# 
# Here is a starter for this competition. The main steps in this notebook are:
# 
# * Importing and combining the competition- and original data.
# * Optuna Hyperparam Search
# * Engineering of Time Features
# * Stratified KFold: CV and ensemble of test-predictions
# 
# The cv-ensemble trained with early-stopping is pretty solid in my experience. (does also well on private-leaderboard)
# You can improve the score by more hyper-paramater tuning. Create additional features. Ensemble with other models (XGB, Catboost..)

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgbm
from sklearn.metrics import roc_auc_score
import optuna
from optuna.samplers import TPESampler

pd.options.display.max_columns = 50


# ## Data Loading 

# In[2]:


path = '/kaggle/input/playground-series-s3e4/'
train = pd.read_csv(path +'train.csv')
#combine with original training set
orig_train = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
train = pd.concat([train,orig_train]).reset_index(drop=True)
test = pd.read_csv(path +'test.csv')
sub = pd.read_csv(path +'sample_submission.csv')
target = 'Class'


# In[3]:


# anonymized features
train.head(3)


# In[4]:


# no missing values
# no categorical features
train.info()


# In[5]:


# very inbalanced classes
train.Class.value_counts()


# In[6]:


# I observed overfitting with the 'Time' feature
features = [c for c in train.columns if c not in ['id','Time', target]]


# ## Optuna 

# In[7]:


def objective(trial):
    
    params_optuna = {
        
        'scale_pos_weight':trial.suggest_int('scale_pos_weight', 1, 3),
        #'lambda_l1': trial.suggest_float('lambda_l1', 1e-12, 2, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 5, 25.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 35, 50),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.65, 0.85),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.65),
        'bagging_freq': trial.suggest_int('bagging_freq', 4, 9),
         'min_child_samples': trial.suggest_int('min_child_samples', 40, 90),
         'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 90, 150),
        "max_depth": trial.suggest_int("max_depth", 6, 12),

        'num_iterations':10000,
        'learning_rate':0.1
    }
    n=3
    cv = StratifiedKFold(n,shuffle=True, random_state=42)
    all_scores = []
    for i,(train_idx,val_idx) in enumerate(cv.split(train[features],train[target])):
        X_train, y_train = train.loc[train_idx, features],train.loc[train_idx, target]
        X_val, y_val = train.loc[val_idx, features],train.loc[val_idx, target]

        model = lgbm.LGBMClassifier(**params_optuna)
        model.fit(X_train,
                  y_train,
                  eval_set = [(X_val,y_val)],
                  early_stopping_rounds=50,
                  verbose=500)

        y_pred = model.predict_proba(X_val)[:,1]
        score = roc_auc_score(y_val,y_pred)
        all_scores.append(score)

    return np.mean(all_scores)


# Here I use the TPESampler for the optuna search. TPE means Tree-structured Parzen Estimator, which is a Bayesian optimization method.
# 
# The TPE algorithm uses a probabilistic model to estimate the distribution of the objective function, and then samples new trial points based on this estimated distribution.

# In[8]:


#study = optuna.create_study(direction='maximize', sampler = TPESampler())
#study.optimize(func=objective, n_trials=100)
#study.best_params


# ## Time Feature Engineering
# 
# inspired by: https://www.kaggle.com/competitions/playground-series-s3e4/discussion/380771

# In[9]:


import seaborn as sns
from matplotlib import pyplot as plt
fig, axs = plt.subplots(1,2, figsize= (12,4))

df_time = pd.concat([train[['Time']],test[['Time']]]).reset_index(drop=True)
df_time.loc[:len(train),'split'] = 'Train'
df_time.loc[len(train):,'split'] = 'Test'
df_time.loc[:len(orig_train),'source'] = 'Synthetic'
df_time.loc[len(orig_train):len(train),'source'] = 'Original'
df_time.loc[len(train):,'source'] = 'Synthetic'

sns.histplot(df_time, x='Time',hue='split', ax= axs[0])
sns.histplot(df_time[df_time['source']=='Synthetic'], x='Time',hue='split', ax= axs[1])
axs[0].set_title('Distribution of Time with original Data')
axs[1].set_title('Distribution of Time without original Data')
plt.show()


# Especially on the second plot it becomes obvious that we can not use the 'Time' feature as is. The (synthetic) competition data has no overlap between train and test. The GBDT model would shurely use the feature and find splits that minimizes the loss but these splits only work on the training set --> overfitting
# 
# Lets create time features that work both with the train and test data.

# In[10]:


train['hour'] = train['Time'] % (24 * 3600) // 3600
train['day'] = (train['Time'] // (24 * 3600)) % 7

test['hour'] = test['Time'] % (24 * 3600) // 3600
test['day'] = (test['Time'] // (24 * 3600)) % 7

features = [c for c in train.columns if c not in ['id','Time', target]]


# In[11]:


# the data is only from two days
print(f"train days: {train['day'].nunique()} test days: {test['day'].nunique()} ") 


# ## GBDT CV  

# In[12]:


n=10
cv = StratifiedKFold(n,shuffle=True, random_state=42)
test_preds = []

all_scores = []
for i,(train_idx,val_idx) in enumerate(cv.split(train[features],train[target])):
    X_train, y_train = train.loc[train_idx, features],train.loc[train_idx, target]
    X_val, y_val = train.loc[val_idx, features],train.loc[val_idx, target]
    
    params={'objective': 'binary',
             'metric': 'auc',
             'lambda_l1': 1.0050418664783436e-08, 
             'lambda_l2': 9.938606206413121,
             'scale_pos_weight': 1, #param could be ignored
             'num_leaves': 44,
             'feature_fraction': 0.8247273276668773,
             'bagging_fraction': 0.5842711778104962,
             'bagging_freq': 6,
             'min_data_in_leaf': 134,
             'min_child_samples': 70,
             'max_depth': 8,
             'num_iterations': 300,
             'learning_rate':0.05}
    
    model = lgbm.LGBMClassifier(**params)
    model.fit(X_train,
              y_train,
              eval_set = [(X_val,y_val)],
              early_stopping_rounds=50,
              verbose=500)
    
    y_pred = model.predict_proba(X_val)[:,1]
    score = roc_auc_score(y_val,y_pred)
    all_scores.append(score)
    
    test_pred = model.predict_proba(test[features])[:,1]
    test_preds.append(test_pred)
    print(f'=== Fold {i} ROC AUC Score {score} ===')

print(f'=== Average ROC AUC Score {np.mean(all_scores)} ===')


# In[13]:


#feature importance
fe =sorted(list(zip(model.feature_name_,model.feature_importances_)),key= lambda x: x[1], reverse=True)
for feat, importance in fe:
    print(f'{feat:25} {importance}')


# ## Submission

# In[14]:


sub['Class'] = np.array(test_preds).mean(axis=0)
sub.to_csv('submission.csv', index=False)


# In[15]:


sub.head()

