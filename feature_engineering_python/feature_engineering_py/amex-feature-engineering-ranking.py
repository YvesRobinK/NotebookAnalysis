#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc
import shap
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from pandas.api.types import is_numeric_dtype

SEED = 42
random.seed(SEED)


# In[2]:


get_ipython().run_cell_magic('time', '', "train = pd.read_feather('../input/amexfeather/train_data.ftr')\n")


# ### Generating features

# In[3]:


agg_train = pd.DataFrame()
train['S_2'] = pd.to_datetime(train['S_2'])
agg_train['S_2_min'] = train[['S_2','customer_ID']].groupby('customer_ID')['S_2'].first()
agg_train['S_2_max'] = train[['S_2','customer_ID']].groupby('customer_ID')['S_2'].last()
agg_train['S_2_period'] = (agg_train['S_2_max'] - agg_train['S_2_min']).dt.days
agg_train['S_2_start_month'] = agg_train['S_2_min'].apply(lambda x: x.strftime('%m'))


# In[4]:


def agg_feat_eng(df):
    num_features = [col for col in df.columns if is_numeric_dtype(df[col]) and col != 'target']
    agg_feature_names = [f'{feat}_mean' for feat in num_features] + [f'{feat}_std' for feat in num_features]
    num_feats_agg = df.groupby('customer_ID')[num_features].agg(['mean','std'])
    num_feats_agg.columns = agg_feature_names
    return num_feats_agg


# In[5]:


train_agg = agg_feat_eng(train)
train_df = train.groupby('customer_ID').tail(1).set_index('customer_ID')
train_df = pd.concat([train_df,train_agg], axis=1)

del train
gc.collect()


# In[6]:


train_df = pd.concat([train_df, agg_train[['S_2_period','S_2_start_month']]], axis=1)


# ### Pre-processing

# In[7]:


object_cols = [col for col in train_df.columns if str(train_df[col].dtype) == 'object']
train_df[object_cols] = train_df[object_cols].astype('category')

train_cols = [col for col in train_df.columns if col not in ['customer_ID','target','S_2']]
X_train, X_val, y_train, y_val = train_test_split(train_df[train_cols], train_df['target'], test_size=0.2, random_state=SEED)

del train_df
gc.collect()


# ### Training LightGBM Model

# In[8]:


dtrain = lgb.Dataset(X_train, y_train)
deval = lgb.Dataset(X_val, y_val)

params = {'objective':'binary',
          'learning_rate':0.05,
          'metric':['auc','binary_logloss'],
          'max_depth':7,
          'num_leaves':70,
          'verbose':-1
         }

del X_train
gc.collect()


# In[9]:


get_ipython().run_cell_magic('time', '', 'model = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dtrain, deval], callbacks=[early_stopping(50), log_evaluation(500)])\n')


# ### Feature Ranking based on lgb importances

# In[10]:


feature_importance_df = pd.DataFrame({'feature_name': model.feature_name(),'tree_split': model.feature_importance(importance_type='split')})
feature_importance_df = feature_importance_df.sort_values(by='tree_split', ascending=False)
feature_importance_df['lgb_rank'] = feature_importance_df['tree_split'].rank(method='first',ascending=False).astype(int)
lgb.plot_importance(model, max_num_features = 20,figsize=(10, 9))


# ### Feature Ranking based on shap importances

# In[11]:


get_ipython().run_cell_magic('time', '', 'explainer = shap.TreeExplainer(model)\nshap_values = explainer.shap_values(X_val)\nshap.summary_plot(shap_values, features=X_val, feature_names=X_val.columns)\n')


# In[12]:


means = [np.abs(shap_values[class_]).mean(axis=0) for class_ in range(len(shap_values))]
shap_means = np.sum(np.column_stack(means), 1)
importance_df = pd.DataFrame({'feature_name': X_val.columns, 'mean_shap_value': shap_means}).sort_values(by='mean_shap_value', ascending=False).reset_index(drop=True)
importance_df['shap_rank'] = importance_df['mean_shap_value'].rank(method='first',ascending=False).astype(int)


# In[13]:


feature_importance_df = feature_importance_df.merge(importance_df,on='feature_name')


# In[14]:


feature_importance_df


# In[ ]:




