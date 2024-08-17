#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('bmh')


# ## Please refer to this  notebook I created for getting the feature rankings: 
# **https://www.kaggle.com/code/ishaan45/amex-feature-engineering-ranking**
# 
# **Any suggestions are most welcome!**

# ### Selecting feaures

# In[2]:


feats_rank = pd.read_csv('../input/amexfeatureranking/all_feat_ranking.csv')
feats_rank['mean_shap_value'] = feats_rank['mean_shap_value'].round(4)


# In[3]:


# LGB Based Feature importance scores distribution
sns.histplot(feats_rank['tree_split'], kde=True)


# In[4]:


# shap values distribution for all features
sns.histplot(feats_rank['mean_shap_value'], bins=50)


# ### Getting relevant features from raw data

# In[5]:


def get_relevant_features(imp_df,top_n=0,stratergy='shap',val_thresh=0):
    if top_n > 0:
        top_feats = imp_df.sort_values(by='shap_rank' if stratergy =='shap' else 'lgb_rank', ascending=True).head(top_n)
    else:
        top_feats = imp_df[imp_df['tree_split' if stratergy =='lgb' else 'mean_shap_value'] > val_thresh]
    
    top_feat_names = top_feats['feature_name'].values.tolist()
    return top_feat_names


# In[6]:


feats_list = get_relevant_features(feats_rank, top_n=204, stratergy='shap')
raw_feats = [feat for feat in feats_list if not ('_mean' in feat or '_std' in feat)]


# In[7]:


mean_feats = [feat for feat in feats_list if '_mean' in feat]
std_feats = [feat for feat in feats_list if '_std' in feat]


# In[8]:


mean_cols = list(map(lambda x:x[:-5], mean_feats))
std_cols = list(map(lambda x:x[:-4], std_feats))
cols_to_load = list(set(raw_feats + mean_cols + std_cols))


# In[9]:


print(f"Total number of columns to be loaded: {len(cols_to_load)}")


# ### Reading specific columns

# In[10]:


get_ipython().run_cell_magic('time', '', "train_df = pd.read_feather('../input/amexfeather/train_data.ftr', columns=cols_to_load + ['customer_ID','target'])\n")


# In[11]:


print(f"Initial shape of train data: {train_df.shape}")


# In[12]:


# %%time
# missing_vals = train_df.isnull().sum().reset_index()
# missing_vals.columns = ['Column','Count']
# missing_vals['Missing %'] = np.round(missing_vals['Count'] / train_df.shape[0] * 100,2)
# missing_vals.sort_values(by='Missing %', ascending=False)

# high_miss_cols = missing_vals[missing_vals['Missing %'] > 95]['Column'].values.tolist()
# train_df = train_df.drop(high_miss_cols, axis=1)


# ### Generating aggregated features

# In[13]:


def agg_feat_eng(df, cols, stat='mean'):
    feature_names = [f'{feat}_{stat}' for feat in cols]
    stat_feats_agg = df.groupby('customer_ID')[cols].agg(stat)
    stat_feats_agg.columns = feature_names
    return stat_feats_agg


# In[14]:


mean_df = agg_feat_eng(train_df,mean_cols, stat='mean')
std_df = agg_feat_eng(train_df, std_cols, stat='std')
train_data = train_df.groupby('customer_ID').tail(1).set_index('customer_ID')

del train_df
gc.collect()

train_data = pd.concat([train_data,mean_df,std_df], axis=1)
train_data = train_data[feats_list+['target']]
print(f"Shape of data after adding aggregated features: {train_data.shape}")

del mean_df, std_df
gc.collect()


# In[15]:


y = train_data['target'] 
X = train_data.drop('target', axis=1)

del train_data
gc.collect()


# ## Loading and transforming test

# In[16]:


get_ipython().run_cell_magic('time', '', "test = pd.read_feather('../input/amexfeather/test_data.ftr',columns=cols_to_load + ['customer_ID'])\n")


# In[17]:


mean_df = agg_feat_eng(test, mean_cols, stat='mean')
std_df = agg_feat_eng(test, std_cols, stat='std')
test_df = test.groupby('customer_ID').tail(1).set_index('customer_ID', drop=True).sort_index()

del test
gc.collect()


# In[18]:


test_df = pd.concat([test_df,mean_df,std_df], axis=1)
test_df = test_df[feats_list]
print(f"Shape of data after adding aggregated features: {test_df.shape}")

del mean_df, std_df
gc.collect()


# In[19]:


def amex_metric(y_true: np.array, y_pred: np.array) -> float:

    # count of positives and negatives
    n_pos = y_true.sum()
    n_neg = y_true.shape[0] - n_pos

    # sorting by descring prediction values
    indices = np.argsort(y_pred)[::-1]
    preds, target = y_pred[indices], y_true[indices]

    # filter the top 4% by cumulative row weights
    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_filter = cum_norm_weight <= 0.04

    # default rate captured at 4%
    d = target[four_pct_filter].sum() / n_pos

    # weighted gini coefficient
    lorentz = (target / n_pos).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    # max weighted gini coefficient
    gini_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))

    # normalized weighted gini coefficient
    g = gini / gini_max

    return 0.5 * (g + d)


# ### Stratified K-Fold Model Training (with conservative weights added to positive samples)

# In[20]:


get_ipython().run_cell_magic('time', '', 'gbm_test_preds =[]\nsk_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)\n\nfor fold, (train_idx, val_idx) in enumerate(sk_fold.split(X, y)):\n    \n    print("\\nFold {}".format(fold+1))\n    X_train, y_train = X.iloc[train_idx,:], y[train_idx]\n    X_val, y_val = X.iloc[val_idx,:], y[val_idx]\n    print("Train shape: {}, {}, Valid shape: {}, {}\\n".format(\n        X_train.shape, y_train.shape, X_val.shape, y_val.shape))\n    \n    target_value_counts = y_train.value_counts()\n    scale_pos_weight= np.power(target_value_counts[0] / target_value_counts[1], 1/2)\n    \n    params = {\'boosting_type\': \'gbdt\',\n              \'n_estimators\': 10000,\n              \'num_leaves\': 80,\n              \'learning_rate\': 0.05,\n              \'feature_fraction\': 0.7,\n              \'bagging_fraction\': 0.85,\n              \'scale_pos_weight\':scale_pos_weight,\n              \'max_depth\':7,\n              \'objective\': \'binary\',\n              \'random_state\': 42}\n    \n    gbm = LGBMClassifier(**params).fit(X_train, y_train,\n                                       eval_set=[(X_train, y_train), (X_val, y_val)],\n                                       callbacks=[early_stopping(50), log_evaluation(500)],\n                                       eval_metric=[\'auc\',\'binary_logloss\'])\n    gbm_test_preds.append(gbm.predict_proba(test_df)[:,1])\n    \n    print(f"Fold Score:{np.round(amex_metric(y_val, gbm.predict_proba(X_val)[:,1]),4)}")\n    \n    del X_train, y_train, X_val, y_val\n    _ = gc.collect()\n    \ndel X,y\ngc.collect()\n')


# In[21]:


sub = pd.read_csv("../input/amex-default-prediction/sample_submission.csv")
sub['prediction']=np.mean(gbm_test_preds, axis=0)
sub.to_csv('submission.csv', index=False)


# In[22]:


sub


# In[ ]:




