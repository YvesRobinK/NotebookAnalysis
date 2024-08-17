#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Hey, thanks for viewing my Kernel!
# 
# If you like my work, please, leave an upvote: it will be really appreciated and it will motivate me in offering more content to the Kaggle community ! :)
# 
# EDA was done in this [notebook](https://www.kaggle.com/code/hasanbasriakcay/tpsapr22-eda-fe-baseline)<br />
# Pseudo Labeling was done in this [notebook](https://www.kaggle.com/code/hasanbasriakcay/tpsapr22-fe-pseudo-labels-baseline)

# In[1]:


import pandas as pd
import numpy as np
import warnings 

warnings.simplefilter("ignore")
train_ = pd.read_csv("../input/tabular-playground-series-apr-2022/train.csv")
test = pd.read_csv("../input/tabular-playground-series-apr-2022/test.csv")
test_pseudo = pd.read_csv("../input/tpsapr22-pseudo-labels/pseudo_labeled_test.csv")
train_labels = pd.read_csv("../input/tabular-playground-series-apr-2022/train_labels.csv")
sub = pd.read_csv("../input/tabular-playground-series-apr-2022/sample_submission.csv")

display(train_.head())
display(test.head())
display(train_labels.head())
display(sub.head())


# In[2]:


train = train_.merge(train_labels, on='sequence', how='left')


# # Feature Engineering

# In[3]:


def create_new_features(df, aggregation_cols=['sequence'], prefix=''):
    df['sensor_02_num'] = df['sensor_02'] > -15
    df['sensor_02_num'] = df['sensor_02_num'].astype(int)
    df['sensor_sum1'] = (df['sensor_00'] + df['sensor_09'] + df['sensor_06'] + df['sensor_01'])
    df['sensor_sum2'] = (df['sensor_01'] + df['sensor_11'] + df['sensor_09'] + df['sensor_06'] + df['sensor_00'])
    df['sensor_sum3'] = (df['sensor_03'] + df['sensor_11'] + df['sensor_07'])
    df['sensor_sum4'] = (df['sensor_04'] + df['sensor_10'])
    
    agg_strategy = {
                    'sensor_00': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_01': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_02': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_03': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_04': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_05': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_06': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_07': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_08': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_09': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_10': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_11': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_12': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_02_num': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_sum1': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_sum2': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_sum3': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_sum4': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                   }
    
    group = df.groupby(aggregation_cols).aggregate(agg_strategy)
    group.columns = ['_'.join(col).strip() for col in group.columns]
    group.columns = [str(prefix) + str(col) for col in group.columns]
    group.reset_index(inplace = True)
    
    temp = (df.groupby(aggregation_cols).size().reset_index(name = str(prefix) + 'size'))
    group = pd.merge(temp, group, how = 'left', on = aggregation_cols,)
    return group


# In[4]:


train_fe = create_new_features(train, aggregation_cols=['sequence', 'subject'])
test_fe = create_new_features(test, aggregation_cols=['sequence', 'subject'])


# In[5]:


train_fe_subjects = create_new_features(train, aggregation_cols = ['subject'], prefix = 'subject_')
test_fe_subjects = create_new_features(test, aggregation_cols = ['subject'], prefix = 'subject_')


# In[6]:


train_fe = train_fe.merge(train_fe_subjects, on='subject', how='left')
train_fe = train_fe.merge(train_labels, on='sequence', how='left')
test_fe = test_fe.merge(test_fe_subjects, on='subject', how='left')


# In[7]:


print(train_fe.shape, test_fe.shape)


# In[8]:


def select_pseudo_labeled_test(df_train, df, th=0.99):
    temp_df = df.loc[((df['state_proba']>=th) | (df['state_proba']<=(1 - th))), :]
    temp_df['state_proba'] = temp_df['state_proba'].round()
    temp_df = temp_df.rename(columns={'state_proba':'state'})
    new_df = pd.concat([df_train, temp_df])
    return new_df


# In[9]:


train_fe = select_pseudo_labeled_test(train_fe, test_pseudo, th=0.95)


# In[10]:


print(train_fe.shape, test_fe.shape)


# In[11]:


train_fe.reset_index(inplace=True, drop=True)


# # Optuna

# In[12]:


from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import optuna

features = list(test_fe.columns)
features.remove("sequence")
features.remove("subject")
X_train = train_fe[features]
X_test = test_fe[features]
y_train = train_fe[['state']]
nfold=10


# In[13]:


def objective(trial):
    params = {
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 11, 333),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 32),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.5),
        'n_estimators': trial.suggest_categorical('n_estimators', [2000]),
        'min_data_per_group': trial.suggest_int('min_data_per_group', 50, 200),
        'n_jobs' : -1, 
        'random_state': 42,
        'boosting_type': 'gbdt',
        'metric': 'AUC',
        #'device': 'gpu',
        'force_col_wise':True,
        'verbose':-1
    }
    oof = np.zeros(len(X_train))
    idx1 = X_train.index
    skf = StratifiedKFold(n_splits=nfold, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(X_train, y_train):
        model = LGBMClassifier(**params)  
        model.fit(X_train.loc[train_index,:], y_train.loc[train_index, 'state'], 
                  eval_set = [(X_train.loc[test_index,:], y_train.loc[test_index, 'state'])], 
                  early_stopping_rounds=300, verbose=False)
        oof[idx1[test_index]] = model.predict_proba(X_train.loc[test_index,:])[:,1]
    
    auc = roc_auc_score(y_train, oof)
    
    return auc


# In[14]:


#study = optuna.create_study(direction='maximize')
#study.optimize(objective, n_trials=100)
#print('Number of finished trials: ', len(study.trials))
#print('Best trial: ', study.best_trial.params)
#print('Best value: ', study.best_value)


# In[15]:


params = {
          'reg_alpha': 0.015957101403194344, 
          'reg_lambda': 0.7496196641999897, 
          'num_leaves': 69, 
          'min_child_samples': 15, 
          'max_depth': 16, 
          'learning_rate': 0.1, 
          'colsample_bytree': 0.45484527384984225, 
          'n_estimators': 2000, 
          'min_data_per_group': 190
}


# # Blending LGBM

# In[16]:


def lgbm_blending(X_train, y_train, X_test, nfold=10):
    preds = np.zeros(len(X_test))
    idx2 = X_test.index
    skf = StratifiedKFold(n_splits=nfold, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(X_train, y_train):
        model = LGBMClassifier(**params)  
        model.fit(X_train.loc[train_index,:], y_train.loc[train_index, 'state'], 
                  eval_set = [(X_train.loc[test_index,:], y_train.loc[test_index, 'state'])], 
                  early_stopping_rounds=300, verbose=False)
        preds[idx2] += model.predict_proba(X_test)[:,1] / skf.n_splits
    return preds


# In[17]:


X_train = train_fe[features]
X_test = test_fe[features]
y_train = train_fe[['state']]
nfold=10
preds = lgbm_blending(X_train, y_train, X_test, nfold=15)


# In[18]:


sub['state'] = preds
sub.to_csv('submission.csv', index=False)

