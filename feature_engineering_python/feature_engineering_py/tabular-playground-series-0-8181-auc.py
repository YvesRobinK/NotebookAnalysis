#!/usr/bin/env python
# coding: utf-8

# ## Importing necessary libraries

# In[1]:


import tensorflow.compat.v1 as tf
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.io import loadmat
import os
from pywt import wavedec
from functools import reduce
from scipy import signal
from scipy.stats import entropy
from scipy.fft import fft, ifft
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from tensorflow import keras as K
import matplotlib.pyplot as plt
import scipy
import tqdm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold,cross_validate
from tensorflow.keras.layers import Dense, Activation, Flatten,Embedding, concatenate, Input, Dropout, LSTM, Bidirectional,BatchNormalization,PReLU,ReLU,Reshape
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential, Model, load_model
import matplotlib.pyplot as plt;
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.decomposition import PCA
from tensorflow import keras
from tensorflow.keras.layers import Conv1D,Conv2D,Add
from tensorflow.keras.layers import MaxPool1D, MaxPooling2D
import seaborn as sns
import sklearn


# ## Reading the train and test datasets.

# In[2]:


train_data = pd.read_csv('../input/tabular-playground-series-sep-2021/train.csv')
test_data = pd.read_csv('../input/tabular-playground-series-sep-2021/test.csv')


# In[3]:


train_data


# In[4]:


test_data


# In[5]:


train_data.describe()


# In[6]:


test_data.describe()


# ## Deleting unnecessary columns.

# From the count rows in the tables above, we see that columns like "Id" are unique, hence wont contribute for our predictions. Hence we remove them.

# In[7]:


train_data.pop('id')
test_data.pop('id')
y = train_data.pop('claim')


# ## Dealing with missing values.

# In[8]:


train_data.isna().sum() , test_data.isna().sum()


# We see that each column has multiple missing values, hence we use Imputation to fill the null values instead of dropping those rows which would lead to loss of information.
# 

# ## Feature engineering, Median imputation and Z-score normalization.

# 1. Feature Engineering: We construct new features with summary statistics like Mean, Variance, Standard Deviation (SD) and additional features like n_missing denoting the number of missing values along each row.
# 
# 2. Imputation: We use median imputation for filling in the null values. We create a dictionary to decide how to impute each column.
# 
#     Mean: normal distribution
# 
#     Median: unimodal and skewed 
# 
#     Mode: all other cases
# 
# 3. Normalization/ Standardization: We observe that the values in each column have different scales, hence we perform RobustScaling to avoid effect of outliers. 

# In[9]:


features = [x for x in train_data.columns.values if x[0]=="f"]

train_data['n_missing'] = train_data[features].isna().sum(axis = 1)
test_data['n_missing'] = test_data[features].isna().sum(axis = 1)

train_data['std'] = train_data[features].std(axis = 1)
test_data['std'] = test_data[features].std(axis = 1)

train_data['var'] = train_data[features].var(axis = 1)
test_data['var'] = test_data[features].var(axis = 1)

train_data['mean'] = train_data[features].mean(axis = 1)
test_data['mean'] = test_data[features].mean(axis = 1)

train_data['median'] = train_data[features].median(axis = 1)
test_data['median'] = test_data[features].median(axis = 1)

train_data['rms'] = (train_data.iloc[:,1:]**2).sum(1).pow(1/2)
test_data['rms'] = (test_data.iloc[:,1:]**2).sum(1).pow(1/2)

train_data['abs_sum'] = train_data[features].abs().sum(axis = 1)
test_data['abs_sum'] = test_data[features].abs().sum(axis = 1)

train_data['max'] = train_data[features].max(axis = 1)
test_data['max'] = test_data[features].max(axis = 1)

train_data['min'] = train_data[features].min(axis = 1)
test_data['min'] = test_data[features].min(axis = 1)

features += ['n_missing', 'std', 'var','mean','median','rms','abs_sum','max','min']



# In[10]:


sc = RobustScaler()
train_data[features] = sc.fit_transform(train_data[features])
test_data[features] = sc.transform(test_data[features])


# In[11]:


fill_value_dict = {
    'f1': 'Mean', 
    'f2': 'Median', 
    'f3': 'Median', 
    'f4': 'Median', 
    'f5': 'Mode', 
    'f6': 'Mean', 
    'f7': 'Median', 
    'f8': 'Median', 
    'f9': 'Median', 
    'f10': 'Median', 
    'f11': 'Mean', 
    'f12': 'Median', 
    'f13': 'Mean', 
    'f14': 'Median', 
    'f15': 'Mean', 
    'f16': 'Median', 
    'f17': 'Median', 
    'f18': 'Median', 
    'f19': 'Median', 
    'f20': 'Median', 
    'f21': 'Median', 
    'f22': 'Mean', 
    'f23': 'Mode', 
    'f24': 'Median', 
    'f25': 'Median', 
    'f26': 'Median', 
    'f27': 'Median', 
    'f28': 'Median', 
    'f29': 'Mode', 
    'f30': 'Median', 
    'f31': 'Median', 
    'f32': 'Median', 
    'f33': 'Median', 
    'f34': 'Mean', 
    'f35': 'Median', 
    'f36': 'Mean', 
    'f37': 'Median', 
    'f38': 'Median', 
    'f39': 'Median', 
    'f40': 'Mode', 
    'f41': 'Median', 
    'f42': 'Mode', 
    'f43': 'Mean', 
    'f44': 'Median', 
    'f45': 'Median', 
    'f46': 'Mean', 
    'f47': 'Mode', 
    'f48': 'Mean', 
    'f49': 'Mode', 
    'f50': 'Mode', 
    'f51': 'Median', 
    'f52': 'Median', 
    'f53': 'Median', 
    'f54': 'Mean', 
    'f55': 'Mean', 
    'f56': 'Mode', 
    'f57': 'Mean', 
    'f58': 'Median', 
    'f59': 'Median', 
    'f60': 'Median', 
    'f61': 'Median', 
    'f62': 'Median', 
    'f63': 'Median', 
    'f64': 'Median', 
    'f65': 'Mode', 
    'f66': 'Median', 
    'f67': 'Median', 
    'f68': 'Median', 
    'f69': 'Mean', 
    'f70': 'Mode', 
    'f71': 'Median', 
    'f72': 'Median', 
    'f73': 'Median', 
    'f74': 'Mode', 
    'f75': 'Mode', 
    'f76': 'Mean', 
    'f77': 'Mode', 
    'f78': 'Median', 
    'f79': 'Mean', 
    'f80': 'Median', 
    'f81': 'Mode', 
    'f82': 'Median', 
    'f83': 'Mode', 
    'f84': 'Median', 
    'f85': 'Median', 
    'f86': 'Median', 
    'f87': 'Median', 
    'f88': 'Median', 
    'f89': 'Median', 
    'f90': 'Mean', 
    'f91': 'Mode', 
    'f92': 'Median', 
    'f93': 'Median', 
    'f94': 'Median', 
    'f95': 'Median', 
    'f96': 'Median', 
    'f97': 'Mean', 
    'f98': 'Median', 
    'f99': 'Median', 
    'f100': 'Mode', 
    'f101': 'Median', 
    'f102': 'Median', 
    'f103': 'Median', 
    'f104': 'Median', 
    'f105': 'Median', 
    'f106': 'Median', 
    'f107': 'Median', 
    'f108': 'Median', 
    'f109': 'Mode', 
    'f110': 'Median', 
    'f111': 'Median', 
    'f112': 'Median', 
    'f113': 'Mean', 
    'f114': 'Median', 
    'f115': 'Median', 
    'f116': 'Mode', 
    'f117': 'Median', 
    'f118': 'Mean'
}


for col in (features):
    if fill_value_dict.get(col)=='Mean':
        fill_value = train_data[col].mean()
    elif fill_value_dict.get(col)=='Median':
        fill_value = train_data[col].median()
    elif fill_value_dict.get(col)=='Mode':
        fill_value = train_data[col].mode().iloc[0]
    
    train_data[col].fillna(fill_value, inplace=True)
    test_data[col].fillna(fill_value, inplace=True)


# ## Removing outliers using Z-score for data points.

# In[12]:


#from scipy import stats
#train_data = train_data[(np.abs(stats.zscore(train_data)) < 3).all(axis=1)]


# In[13]:


train_data


# ## Dividing the dependent and independent variables.

# In[14]:


x = pd.DataFrame(train_data)
y = pd.Series(y)


# ## Hyperparameter tuning of LGBM Classifier.

# In[15]:


"""
import optuna
from lightgbm import LGBMClassifier
def objective(trial, data = x, target = y):

    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 40000),
        'max_depth': trial.suggest_int('max_depth', 2, 3),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 50, 500),
        'min_data_per_group': trial.suggest_int('min_data_per_group', 50, 200),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.8),
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'random_state': 228,
        'metric': 'auc',
        'device_type': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0
    }
    
    model = LGBMClassifier(**params)
    scores = []
    k = StratifiedKFold(n_splits = 2, random_state = 228, shuffle = True)
    for i, (trn_idx, val_idx) in enumerate(k.split(x, y)):
        
        X_train, X_val = x.iloc[trn_idx], x.iloc[val_idx]
        y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]

        model.fit(X_train, y_train, eval_set = [(X_val, y_val)], early_stopping_rounds = 300, verbose = False)
        
        tr_preds = model.predict_proba(X_train)[:,1]
        tr_score = sklearn.metrics.roc_auc_score(y_train, tr_preds)
        
        val_preds = model.predict_proba(X_val)[:,1]
        val_score = sklearn.metrics.roc_auc_score(y_val, val_preds)

        scores.append((tr_score, val_score))
        
        print(f"Fold {i+1} | AUC: {val_score}")
        
    scores = pd.DataFrame(scores, columns = ['train score', 'validation score'])
    
    return scores['validation score'].mean()

study = optuna.create_study(direction = 'maximize')
study.optimize(objective, n_trials = 10)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
print('Best value:', study.best_value)

"""


# ## Performing 10 fold cross validation using LGBMClassifier.

# In[16]:


from lightgbm import LGBMClassifier

SEED = 228
paramsLGBM = {'objective': 'binary',
               'boosting_type': 'gbdt',
               'num_leaves': 6,
               'max_depth': 2,
               'n_estimators': 40000,
               'reg_alpha': 25.0,
               'reg_lambda': 76.7,
               'random_state': SEED,
               'bagging_seed': SEED, 
               'feature_fraction_seed': SEED,
               'n_jobs': -1,
               'subsample': 0.98,
               'subsample_freq': 1,
               'colsample_bytree': 0.69,
               'min_child_samples': 54,
               'min_child_weight': 256,
               'learning_rate': 0.2,
               'metric': 'AUC',
               'verbosity': -1,
              }

folds = StratifiedKFold(n_splits = 5, random_state = SEED, shuffle = True)
y_pred = np.zeros(len(test_data))
for fold, (trn_idx, val_idx) in enumerate(folds.split(x, y)):
    
    X_train, X_val = x.iloc[trn_idx], x.iloc[val_idx]
    y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]

    model = LGBMClassifier(**paramsLGBM)
   
    model.fit(X_train, y_train, eval_set = [(X_val, y_val)], verbose = False, early_stopping_rounds = 300)
    
    y_pred += model.predict_proba(test_data)[:,1] / folds.n_splits 


# ## Submitting predictions.

# In[17]:


submission = pd.read_csv('../input/tabular-playground-series-sep-2021/sample_solution.csv')
submission['claim'] = y_pred
submission.to_csv('submission.csv',index = False)


# #### With this, we end up with a score of 0.81793 on the leaderboard. There is definitely room for improvement.

# #### Please upvote if you liked it :)

# In[ ]:




