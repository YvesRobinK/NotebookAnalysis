#!/usr/bin/env python
# coding: utf-8

# ## <div style='background:#2b6684;color:white;padding:0.5em;border-radius:0.2em'>Introduction</div>

# **Hi**,<br>
# this is my current solution - just a simple catboost classifier.<br>
# The most important part here is feature engineering, where I'm calculating the sum of missing values per row.<br><br>
# **If you like this notebook or copy some parts of it, please leave an upvote.**<br>
# 
# Best Regards.

# ## <div style='background:#2b6684;color:white;padding:0.5em;border-radius:0.2em'>Import Data</div>

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.gridspec as gridspec
import seaborn as sns

from warnings import filterwarnings
filterwarnings('ignore')

plt.rcParams['font.family'] = 'serif'

cmap = sns.color_palette("ch:start=.2,rot=-.3")
sns.set_palette(cmap)


# In[2]:


get_ipython().run_cell_magic('time', '', "# read dataframe\ndf_train = pd.read_csv('../input/tabular-playground-series-sep-2021/train.csv')\ndf_test = pd.read_csv('../input/tabular-playground-series-sep-2021/test.csv')\n\nsample_submission = pd.read_csv('../input/tabular-playground-series-sep-2021/sample_solution.csv')\n")


# ## <div style='background:#2b6684;color:white;padding:0.5em;border-radius:0.2em'>Preprocessing</div>

# In[3]:


# prepare dataframe for modeling
X = df_train.drop(columns=['id','claim']).copy()
y = df_train['claim'].copy()

test_data = df_test.drop(columns=['id']).copy()


# In[4]:


# feature Engineering
def get_stats_per_row(data):
    data['mv_row'] = data.isna().sum(axis=1)
    data['min_row'] = data.min(axis=1)
    data['std_row'] = data.std(axis=1)
    return data

X = get_stats_per_row(X)
test_data = get_stats_per_row(test_data)


# In[5]:


# create preprocessing pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', StandardScaler())
])

X = pd.DataFrame(columns=X.columns, data=pipeline.fit_transform(X))
test_data = pd.DataFrame(columns=test_data.columns, data=pipeline.transform(test_data))


# ## <div style='background:#2b6684;color:white;padding:0.5em;border-radius:0.2em'>Modeling</div>

# In[6]:


# params from optuna study, i've done earlier
best_params = {
    'iterations': 15585, 
    'objective': 'CrossEntropy', 
    'bootstrap_type': 'Bernoulli', 
    'od_wait': 1144, 
    'learning_rate': 0.023575206684596582, 
    'reg_lambda': 36.30433203563295, 
    'random_strength': 43.75597655616195, 
    'depth': 7, 
    'min_data_in_leaf': 11, 
    'leaf_estimation_iterations': 1, 
    'subsample': 0.8227911142845009,
    'task_type' : 'GPU',
    'devices' : '0',
    'verbose' : 0
}


# In[7]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import KFold\nfrom sklearn.metrics import roc_curve, auc\nfrom catboost import CatBoostClassifier\n\nkf = KFold(n_splits=5, shuffle=True, random_state=1)\n\npred_tmp = []\nscores = []\n\nfor fold, (idx_train, idx_valid) in enumerate(kf.split(X)):\n    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]\n    X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]\n\n    model = CatBoostClassifier(**best_params)\n    model.fit(X_train, y_train)\n\n    # validation prediction\n    pred_valid = model.predict_proba(X_valid)[:,1]\n    fpr, tpr, _ = roc_curve(y_valid, pred_valid)\n    score = auc(fpr, tpr)\n    scores.append(score)\n    \n    print(f"Fold: {fold + 1} Score: {score}")\n    print(\'::\'*20)\n    \n    # test prediction\n    y_hat = model.predict_proba(test_data)[:,1]\n    pred_tmp.append(y_hat)\n    \nprint(f"Overall Validation Score: {np.mean(scores)}")\n')


# In[8]:


# average predictions over all folds
predictions = np.mean(np.column_stack(pred_tmp),axis=1)

# create submission file
sample_submission['claim'] = predictions
sample_submission.to_csv('./catb_baseline.csv', index=False)

