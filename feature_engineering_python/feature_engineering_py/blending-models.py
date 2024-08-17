#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

from scipy.stats import spearmanr, loguniform, randint, uniform
from scipy.special import logit

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set_style('whitegrid')

get_ipython().system('pip install -q flaml')
from flaml import AutoML


# <span style='font-size:large; color:green'>Collection of my work on TPS Nov 2022 Challenge:</span>
# <ul style='color:green'>
#     <li><a href='https://www.kaggle.com/competitions/tabular-playground-series-nov-2022/discussion/363683'>Collection of helpful ideas</a></li>
#     <li><a href='https://www.kaggle.com/code/phongnguyen1/baselines-feature-selection'>Baselines, feature selection, model tunning</a></li>
#     <li><a href='https://www.kaggle.com/code/phongnguyen1/analyze-visualize-submissions'>EDA of submissions with different perspectives</a></li>
# </ul>

# ## Notebook overview
# - Explain the competition, look at the data
# - Build some baselines
# - Apply feature selection to improve
# - Tuning models
# - Stacking models 

# ## Competition overview
# It's to practice creating ensembles from model predictions to produce better results than individual models. Data include:
# - `submission_files` folder contains submissions of predictions for a binary classification task. There are 5,000 submissions, including:
#  - a `pred` column as probability of the binary classification
#  - 40,000 rows as data points for the classification task
# - `train_labels.csv` file has 20,000 ground truths (1/0) for the first 20,000 rows in the submission
# - `sample_submission.csv` file is a sample of predictions of the last 20,000 rows to be submitted

# ## Loading data

# In[2]:


REBUILD_DATA = False
path = Path('/kaggle/input/tabular-playground-series-nov-2022/')
submission = pd.read_csv(path / 'sample_submission.csv', index_col='id')
    
if REBUILD_DATA:
    labels = pd.read_csv(path / 'train_labels.csv', index_col='id')

    # the ids of the submission rows (useful later)
    sub_ids = submission.index

    # the ids of the labeled rows (useful later)
    gt_ids = labels.index 

    # list of files in the submission folder
    subs = sorted(os.listdir(path / 'submission_files'))
    
    dfs = [pd.read_csv(path / 'submission_files' / sub)['pred'] for sub in subs]
    df = pd.concat(dfs, axis=1)
    df.columns = [f'sub_{i+1:04}' for i in range(len(df.columns))]
    df['label'] = labels
#     df.to_feather('/kaggle/working/tps.ftr', index=None)
else:
    df = pd.read_feather('../input/tps-nov-2022/tps.ftr')
    
df.head()


# ## Data check

# In[3]:


df['label'].value_counts()


# > The provided training data is perfectly balanced.

# In[4]:


df_train = df[df['label'].notna()]
df_test = df[df['label'].isna()]
y_train = df_train['label']
y_test = df_test['label']
df_train = df_train.drop(columns=['label'])
df_test = df_test.drop(columns=['label'])
X_train = df_train.values
X_test = df_test.values
# loglosses = np.array([log_loss(y_train, df_train[c]) for c in df_train.columns])


# ## Data preprocessing
# AmbroseM [suggested](https://www.kaggle.com/competitions/tabular-playground-series-nov-2022/discussion/364013) to calibrate the predicted probability and use the results as raw data. It's clear that after calibration, the performance of each individual submission improves quite significantly.

# In[5]:


X_train_logit = np.array([logit(X_train[:,i].clip(1e-6, 1-1e-6)) for i in range(df_train.shape[1])]).transpose()
X_test_logit = np.array([logit(X_test[:,i].clip(1e-6, 1-1e-6)) for i in range(df_train.shape[1])]).transpose()


# Look at a single submission to compare before and after calibration.

# In[6]:


# loglosses[0]


# In[7]:


# X0 = X_train_logit[:,:1]
# y_logistic = LogisticRegression().fit(X0, y_train).predict_proba(X0)[:,1]
# log_loss(y_train, y_logistic)


# ## Baseline 1
# Averaging predictions of top 100 models could be a decent baseline.

# In[8]:


# topn = 100
# y_pred = X_train[:,:topn].mean(axis=1)
# log_loss(y_train, y_pred)


# > Better than the best individual model (0.622) but worse than a single calibrated one! Now, just switch to the calibrated ones completely.

# In[9]:


X_train = X_train_logit
X_test = X_test_logit


# ## Baseline 2
# A simple logistic regression using top 100 features.

# In[10]:


def test_logreg(feature_indices):
    "Build a simple default logistic regression with 5-fold cross validation."
    logreg = LogisticRegression(max_iter=1000, random_state=0)
    neg_loglosses = cross_val_score(logreg, X_train[:,feature_indices], y_train, scoring='neg_log_loss', cv=5)
    return logreg.fit(X_train[:,feature_indices], y_train), -neg_loglosses.mean()


# In[11]:


# logreg, loss = test_logreg(range(topn))
# loss


# > Wow, much much better than the simple average Baseline 1. This is the way to go.

# ## Baseline 3
# Gradient boosting with lightgbm with default parameters

# In[12]:


def test_lgbm(feature_indices):
    "Build a simple default logistic regression and return logloss with 5-fold cross validation."
    lgbm = LGBMClassifier(max_depth=3, random_state=0)
    neg_loglosses = cross_val_score(lgbm, X_train[:,feature_indices], y_train, scoring='neg_log_loss', cv=5)
    return lgbm.fit(X_train[:,feature_indices], y_train), -neg_loglosses.mean()


# In[13]:


# lgbm, loss = test_lgbm(range(topn))
# y_pred = lgbm.predict_proba(X_test[:,:topn])[:,1]
# pd.DataFrame({'id':range(20000,40000), 'pred':y_pred}).to_csv('submission.csv', index=None)


# ## Feature selection
# Selecting top 100 submissions is just a quick way to build a baseline. Let's consider other alternatives.

# ### Just different number of top features
# A simper approach, taking N top features with differnt values of N.

# In[14]:


# stats = []
# for c in [50, 100, 200, 300, 400, 500]:
#     stats.append((c, test_lgbm(range(c))[1]))
# pd.DataFrame(stats, columns=['#features', 'logloss'])


# In[15]:


topn = 300


# In[16]:


# lgbm = test_lgbm(range(topn))[0]
# y_pred = lgbm.predict_proba(X_test[:,:topn])[:,1]
# pd.DataFrame({'id':range(20000,40000), 'pred':y_pred}).to_csv('submission.csv', index=None)


# ## Tune models
# So far, I focus on the data and feature engineering. There are two more things we can do to improve our work. Firs, tune the models then stack them up.

# In[17]:


def get_df_results(cv):
    stats = pd.DataFrame(cv.cv_results_['params'])
    stats['loss'] = -cv.cv_results_['mean_test_score']
    return stats.sort_values('loss')

X_train = X_train[:,:topn]
X_test = X_test[:,:topn]


# ### Logistic regression
# The only parameter to tune is `C` to control regularization; simply use `GridSearchCV`.

# In[18]:


logreg = LogisticRegression(max_iter=1000, random_state=0)
params = {'C': np.logspace(-3, -1, 10)}
logreg_cv = GridSearchCV(logreg, param_grid=params, cv=5, scoring='neg_log_loss')
logreg_cv.fit(X_train, y_train)
get_df_results(logreg_cv).head()


# > This is an improvement compared to previous logistic regression with default `C=1`.

# In[19]:


joblib.dump(logreg_cv.best_estimator_, 'best_logreg.bin')


# ### kNN

# In[20]:


knn = KNeighborsClassifier()
params = {'n_neighbors': np.arange(5, 500, 20)}
knn_cv = GridSearchCV(knn, param_grid=params, cv=5, scoring='neg_log_loss')
knn_cv.fit(X_train, y_train)
get_df_results(knn_cv).head()


# In[21]:


joblib.dump(knn_cv.best_estimator_, 'best_knn.bin')


# ### MLP

# In[22]:


# %%time
# params = {
#     'hidden_layer_sizes': [(128,), (128, 32), (128, 64, 32)],
#     'activation': ['tanh', 'relu'],
#     'solver': ['sgd', 'adam'],
#     'alpha': [1e-3, 1e-2],
#     'learning_rate': ['constant','adaptive'],
#     'learning_rate_init': [1e-3, 3e-3, 1e-2]
# }

# mlp = MLPClassifier(max_iter=1000, random_state=0)
# mlp_cv = RandomizedSearchCV(mlp, param_distributions=params, n_iter=10, cv=5, scoring='neg_log_loss')
# mlp_cv.fit(X_train, y_train)
# get_df_results(mlp_cv).head()


# In[23]:


# joblib.dump(mlp_cv.best_estimator_, 'best_mlp.bin')


# ### LGBMClassifier

# #### Auto tuning with FLAML
# It has so many parameters... The documentation links to an auto-ml library [FLAML](https://github.com/microsoft/FLAML). Let's use it first then to tune manually to compare with. Just need to `pip install flaml` to install the library.

# In[24]:


automl = AutoML()
settings = {
    'time_budget': 60*10, # time in seconds
    'metric': 'log_loss',
    'estimator_list': ['lgbm'],
    'task': 'classification', 
    'seed': 0,
    'verbose': 0
}
automl.fit(X_train=X_train, y_train=y_train, **settings)
lgbm = LGBMClassifier(**automl.best_config, device='gpu', random_state=0)
-cross_val_score(lgbm, X_train, y_train, scoring='neg_log_loss', cv=5).mean()


# In[25]:


lgbm.fit(X_train, y_train)
y_pred = lgbm.predict_proba(X_test)[:,1]
pd.DataFrame({'id':range(20000,40000), 'pred':y_pred}).to_csv('submission.csv', index=None)


# In[26]:


joblib.dump(lgbm, 'best_lgbm.bin')


# > Yay, an improvement in LB.

# ## Stacking models
# When we build the first model, we already did stacking because data input are predictions from 5,000 models. Now, we just do another level up.

# In[27]:


best_logreg = joblib.load('best_logreg.bin')
best_knn = joblib.load('best_knn.bin')
# best_mlp = joblib.load('best_mlp.bin')
best_lgbm = joblib.load('best_lgbm.bin')
kf = StratifiedKFold(shuffle=True, random_state=0)


# In[28]:


def build_oof(model):
    oof = np.zeros(len(X_train))
    for fold, (idx_tr, idx_va) in enumerate(kf.split(X_train, y_train)):
        X_tr = X_train[idx_tr]
        X_va = X_train[idx_va]
        y_tr = y_train[idx_tr]
        y_va = y_train[idx_va]
        model.fit(X_tr, y_tr)
        oof[idx_va] = model.predict_proba(X_va)[:,1]
    oof_test = np.zeros(len(X_test))
    
    return oof

X_oof = np.zeros((len(X_train), 3))
X_oof[:,0] = build_oof(LogisticRegression().set_params(**best_logreg.get_params()))
X_oof[:,1] = build_oof(KNeighborsClassifier().set_params(**best_knn.get_params()))
X_oof[:,2] = build_oof(LGBMClassifier().set_params(**best_lgbm.get_params()))
# X_oof[:,3] = build_oof(MLPClassifier().set_params(**best_mlp.get_params()))


# ### Preparing test data with level 0 models

# In[29]:


X_test_meta = np.zeros((len(X_test), 3))
X_test_meta[:,0] = best_logreg.predict_proba(X_test)[:,1]
X_test_meta[:,1] = best_knn.predict_proba(X_test)[:,1]
X_test_meta[:,2] = best_lgbm.predict_proba(X_test)[:,1]
# X_test_meta[:,3] = best_mlp.predict_proba(X_test)[:,1]


# In[30]:


def test_ridge(X, y):
    "Build a Ridge with 5-fold cross validation."
    model = Ridge()
    losses = []
    for fold, (idx_tr, idx_va) in enumerate(kf.split(X, y)):
        X_tr = X[idx_tr]
        X_va = X[idx_va]
        y_tr = y[idx_tr]
        y_va = y[idx_va]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_va)
        loss = log_loss(y_va, y_pred)
        losses.append(loss)
    return np.mean(losses)


# In[31]:


blend_model = Ridge(alpha=1e-1, fit_intercept=False)
blend_model.fit(X_oof, y_train)
y_pred = blend_model.predict(X_test_meta).clip(1e-6, 1-1e-6)


# In[32]:


pd.DataFrame({'id':range(20000,40000), 'pred':y_pred}).to_csv('submission.csv', index=None)


# ---

# ## Ideas that don't work quite well

# ### Relevant features
# From my [notebook](https://www.kaggle.com/code/phongnguyen1/analyze-visualize-submissions), we know that there are invalid and underperformed submissions. How's about remove those and keep the rest?

# In[33]:


# highest_logloss = 0.693
# relevant_feature_indices = [i for (i,c) in enumerate(df.columns[:-1]) if 
#                             (df[c].min() >= 0) and (df[c].max() <= 1) and (loglosses[i] < highest_logloss)]
# len(relevant_feature_indices)


# > well, still too many. Go for a smaller loss.

# In[34]:


# highest_logloss = 0.68
# relevant_feature_indices = [i for (i,c) in enumerate(df.columns[:-1]) if 
#                             (df[c].min() >= 0) and (df[c].max() <= 1) and (loglosses[i] < highest_logloss)]
# len(relevant_feature_indices)


# In[35]:


# test_logreg(relevant_feature_indices)[1]


# In[36]:


# test_lgbm(relevant_feature_indices)[1]


# > Higher loss with logistic regression but slightly lower loss with gradient boosting. Not significant.

# ### Correlation
# I will try the idea keep only the best features and the ones that are not too much correlated with them ([source](https://www.kaggle.com/code/takanashihumbert/drop-4000-features-lb-0-51530?scriptVersionId=109855117)). But first let's do some analysis on the correleation.

# In[37]:


# df_corr = df_train.iloc[:,relevant_feature_indices].corr(method='spearman')
# corrs_flat = df_corr.values.flatten()
# print(f'{(corrs_flat>0.8).mean():.1%} pairs have correlation above 0.8')


# In[38]:


# plt.figure(figsize=(10,4))
# plt.xlim(0.8,1)
# plt.hist(corrs_flat, bins=16, range=[0.8,1])
# plt.show()


# In[39]:


# def select_features_corr(max_corr=0.9, top_features=5, expand_corr_check=False):
#     fixed_top_features = list(range(top_features)) # Note that features are sorted with increasing loss
#     added_feature_indices = list(range(top_features))
    
#     for idx in relevant_feature_indices[top_features:]:
#         # If the feature is highly correlated with one of the added set, ignore
#         features_to_check = added_feature_indices if expand_corr_check else fixed_top_features
#         for old_idx in features_to_check:
#             if df_corr.loc[df.columns[old_idx], df.columns[idx]] > max_corr:
#                 break
#         else:
#             added_feature_indices.append(idx)
            
#     return added_feature_indices


# In[40]:


# stats = []
# for c in [0.8, 0.85, 0.9, 0.95, 0.98, 0.99]:
#     feature_indices = select_features_corr(max_corr=c, top_features=1, expand_corr_check=True)
#     loss = test_lgbm(feature_indices)[1]
#     stats.append((c, len(feature_indices), loss))
# pd.DataFrame(stats, columns=['corr-threshold', '#features', 'logloss'])


# > I tried different correlation thresholds but none of them produced better performance!

# In[41]:


# A note for the correlation notebook by takanashihumbert
# It turns out that the correlation threshold 0.98 is too high. The feature with the highest correlation with the best feature is 0.952, thus none of the removed features is the result of that. They're just have higher loss than the chosen loss threshold (0.68).
# m = max(spearmanr(df_train.iloc[:,0], df_train.loc[:,c]).correlation for c in df_train.columns[1:])
# print('max correlation with the first (best) feature is', m)


# ### SVM

# In[42]:


# %%time
# svm = SVC(probability=True)
# params = {'C': [1]}
# svm_cv = GridSearchCV(svm, param_grid=params, cv=5, scoring='neg_log_loss')
# svm_cv.fit(X_train, y_train)
# get_df_results(svm_cv).head(20)


# Took 26 minutes to train SVM and got 0.565. Too bad to continue?

# In[43]:


# joblib.dump(svm_cv.best_estimator_, 'best_svm.bin')


# #### Manual tunning

# In[44]:


# %%time
# params = {
#     'max_depth': [2,3,4,5,6],
#     'num_leaves': randint(50, 150),
#     'learning_rate': loguniform(0.01, 0.1),
#     'n_estimators': randint(100, 1000),
#     'reg_lambda': randint(10, 50),
#     'subsample': uniform(0.7, 0.3),
#     'colsample_bytree': uniform(0.7, 0.3),
#     'reg_lambda': uniform(0, 0.3),
# }
# lgbm = LGBMClassifier(device='gpu', random_state=0)
# lgbm_cv = RandomizedSearchCV(lgbm, param_distributions=params, n_iter=100, cv=5, scoring='neg_log_loss', random_state=0)
# lgbm_cv.fit(X_train, y_train)
# get_df_results(lgbm_cv).head(20)

