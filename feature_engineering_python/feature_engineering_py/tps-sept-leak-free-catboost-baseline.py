#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from scipy import stats

from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split,TimeSeriesSplit
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

from sklearn.pipeline import Pipeline
import os, glob, math, cv2, gc, logging, warnings, random

import lightgbm as lgb
import catboost
from catboost import CatBoostRegressor,Pool
from sklearn.metrics import *
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, LGBMRegressor

import shap
warnings.filterwarnings("ignore")


# # Load Data

# In[2]:


train = pd.read_csv('../input/tabular-playground-series-sep-2022/train.csv')
test = pd.read_csv('../input/tabular-playground-series-sep-2022/test.csv')

sample_submission = pd.read_csv("../input/tabular-playground-series-sep-2022/sample_submission.csv")


# In[3]:


train.head(3)


# In[4]:


feature_cols = test.columns.tolist()
cat_cols = ["row_id"]
features = [col for col in feature_cols if col not in cat_cols]


# In[5]:


categoricals = ['country','store','product']


# # Preprocessing

# In[6]:


plt.figure(figsize=(25, 10))
for i, col in enumerate(categoricals):
    plt.subplot(3, 5, i+1)
    sns.violinplot(data=train[categoricals+['num_sold']], x='num_sold', y=col)
    plt.title(col)
    plt.xlabel("")
    plt.ylabel("")
plt.show()


# In[ ]:





# In[7]:


le = LabelEncoder()
data_concat = pd.concat([train,test],axis=0).reset_index(drop=True)

for i in categoricals:
    print(f'encoding column {i}')
    data_concat[i] = le.fit_transform(data_concat[i])


# In[8]:


len(train)


# In[9]:


train = data_concat.iloc[:len(train),:]
test = data_concat.iloc[len(train):,:]


# In[10]:


plt.figure(figsize=(20, 10))
sns.heatmap(train.corr(), annot=True)
plt.show()


# # Some Feature Engineering 

# In[11]:


def add_date_features(df):
    df['date'] = pd.to_datetime(df.date)    
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month   
    df['day'] = df['date'].dt.day   
    df['week'] = df['date'].dt.week   
    
    
    df['day_of_week'] = df['date'].dt.day_of_week
    df['day_of_year'] = df['date'].dt.day_of_year
    
    df['is_weekend'] = np.where(df['day_of_week'].isin([5,6]), 1,0)

    df = df.drop(['date'],axis=1,inplace=False)
    
    return df



# In[12]:


train = add_date_features(train)
test = add_date_features(test)


# In[13]:


train.head(2)


# In[14]:


feature_cols = categoricals
new_features = ["year", "month", "day", "week","day_of_week","day_of_year","is_weekend"]
feature_cols += new_features


# In[ ]:





# In[ ]:





# # Leak Free Time Series Split

# In[15]:


from sklearn.model_selection import TimeSeriesSplit


# In[ ]:





# In[16]:


scores = []
folds = 4
train["kfold"] = -1

val_idx = list(train[(train['year']==2020)& (train['month']>4)& (train['month']<=6)].index)
train.loc[val_idx,"kfold"] = 1

val_idx = list(train[(train['year']==2020)& (train['month']>6)& (train['month']<=8)].index)
train.loc[val_idx,"kfold"] = 2

val_idx = list(train[(train['year']==2020)& (train['month']>8)& (train['month']<=10)].index)
train.loc[val_idx,"kfold"] = 3

val_idx = list(train[(train['year']==2020)& (train['month']>10)].index)
train.loc[val_idx,"kfold"] = 4


# In[17]:


train.kfold.value_counts()


# In[ ]:





# In[18]:


test['preds'] = 0


# In[19]:


def smape(preds, target):
    '''
    Function to calculate SMAPE
    '''
    n = len(preds)
    masked_arr = ~((preds==0)&(target==0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds-target)
    denom = np.abs(preds)+np.abs(target)
#     print('hi')
    smape_val = (200*np.sum(num/denom))/n
#     smape_val = (np.sum(num/denom))/n
#     smape_val = 1
    return smape_val


def lgbm_smape(preds, train_data):
    '''
    Custom Evaluation Function for LGBM
    '''
    labels = train_data.get_label()
    preds = np.array(np.expm1(preds), dtype=np.float)
    labels = np.array(np.expm1(labels), dtype=np.float)
    
#     labels = np.expm1(labels)
    smape_val = smape(preds,labels)
    return 'SMAPE', smape_val, False


# In[20]:


cat_features = list(test[feature_cols].select_dtypes(int).columns)


# In[ ]:





# In[21]:


params = {'depth': 6,
          'learning_rate': 0.02,
          'l2_leaf_reg': 20.0,
          'random_strength': 2.0,
          'min_data_in_leaf': 2}


# In[ ]:





# In[22]:


get_ipython().run_cell_magic('time', '', 'feature_importances = pd.DataFrame()\nfeature_importances[\'feature\'] = feature_cols\nSEEDS = [42,2022,24,100]\nNUM_SEEDS = len(SEEDS)\nfor seed in SEEDS:\n\n    print(f\'*********Training on Seed {seed} ***********\')\n    for fold in range(1,folds+1):\n        x_train = train[train.kfold < fold].copy()\n        x_valid = train[train.kfold == fold].copy()\n        x_test  = test[feature_cols].copy()\n\n        y_train = x_train[\'num_sold\']\n        y_valid = x_valid[\'num_sold\']\n\n        x_train = x_train[feature_cols]\n        x_valid = x_valid[feature_cols]\n\n        train_data = Pool(\n        data=x_train, # ensure your target values are removed\n        label=y_train,# insert your target values\n    )\n\n        valid_data = Pool(\n        data=x_valid, # ensure your target values are removed\n        label=y_valid, # insert your target values\n    )\n\n\n        clf = CatBoostRegressor(**params,\n                                      iterations=5000,\n                                      bootstrap_type=\'Bayesian\',\n    #                                   boosting_type=\'Plain\',\n                                      loss_function=\'MAE\',\n                                      eval_metric=\'SMAPE\',\n                                      random_seed=seed)\n\n        clf.fit(X = train_data,\n                      early_stopping_rounds=200,\n                      eval_set=valid_data,\n                      verbose=100)\n\n        feature_importances[f\'fold_{fold + 1}\'] = clf.get_feature_importance(train_data)\n\n\n        preds_train = clf.predict(x_train)\n        preds_valid = clf.predict(x_valid)\n        smape_train = mean_absolute_error(y_train,preds_train)\n        smape = mean_absolute_error(y_valid,preds_valid)\n        print(f"| Fold {fold+1} | train MAE: {smape_train:.5f} | valid MAE: {smape:.5f} |")\n        print("|--------|----------------|----------------|")\n        scores.append(smape)\n\n        preds_test = clf.predict(x_test)\n        test["preds"] += preds_test \n\ntest["preds"] /= folds*NUM_SEEDS\nprint("\\nAVG MAE:",np.mean(scores))\n')


# In[ ]:





# In[23]:


feature_importances['average'] = feature_importances[[f'fold_{fold+1}' for fold in range(1,folds)]].mean(axis=1)
feature_importances.to_csv('feature_importances.csv')

plt.figure(figsize=(16, 12))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(20), x='average', y='feature');
plt.title('20 TOP feature importance over {} folds average'.format(folds));


# # Submission

# In[24]:


sample_submission


# In[25]:


sample_submission['num_sold'] = test['preds'].values
sample_submission.to_csv("submission.csv", index=False)
sample_submission.head()


# In[26]:


plt.figure(figsize=(10,5))
sns.histplot(sample_submission["num_sold"], kde=True, color="blue")
plt.title("Predictions")
plt.show()


# In[27]:


plt.figure(figsize=(10,5))
sns.histplot(train["num_sold"], kde=True, color="blue")
plt.title("Train labels")
plt.show()


# In[ ]:





# In[ ]:




