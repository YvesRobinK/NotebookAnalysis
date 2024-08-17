#!/usr/bin/env python
# coding: utf-8

# ## <div style='background:#2b6684;color:white;padding:0.5em;border-radius:0.2em'>Introduction</div>

# **Hi**,<br>
# this is my current solution - a one stacked meta-learner or ensemble based on xgb, catboost and lgbm.<br><br>
# 
# The approach is relatively simple. <br>
# I just train different base models and save the oof_predictions to build a meta-set in the end.<br>
# As my final (meta) learner I simple utilize a Logistic-Regression-Model to predict the probability.<br><br>
# 
# **Thanks for checkin' out my notebook, if you like it or even copy some parts of it, be sure to leave an upvote.**<br>
# 
# Best Regards.
# 
# Check out my other notebooks as well:
# - [[TPS-09] Optuna Study-CatBoostClassifier](https://www.kaggle.com/mlanhenke/tps-09-optuna-study-catboostclassifier)
# - [[TPS-09] Single CatBoostClassifier ](https://www.kaggle.com/mlanhenke/tps-09-single-catboostclassifier)
# - [[TPS-09] Spot-Check (XGB,LGBM, CATB GPU)](https://www.kaggle.com/mlanhenke/tps-09-spot-check-xgb-lgbm-catb-gpu)

# ## <div style='background:#2b6684;color:white;padding:0.5em;border-radius:0.2em'>Import Data</div>

# In[1]:


import numpy as np
import pandas as pd

from warnings import filterwarnings
filterwarnings('ignore')


# In[2]:


get_ipython().run_cell_magic('time', '', "# read dataframe\ndf_train = pd.read_csv('../input/tabular-playground-series-sep-2021/train.csv')\ndf_test = pd.read_csv('../input/tabular-playground-series-sep-2021/test.csv')\n\nsample_submission = pd.read_csv('../input/tabular-playground-series-sep-2021/sample_solution.csv')\n")


# ## <div style='background:#2b6684;color:white;padding:0.5em;border-radius:0.2em'>Preprocessing</div>

# In[3]:


# prepare dataframe for modeling
X = df_train.drop(columns=['id','claim']).copy()
y = df_train['claim'].copy()

test_data = df_test.drop(columns=['id']).copy()


# In[4]:


# feature-engineering
def get_stats_per_row(data):
    data['mv_row'] = data.isna().sum(axis=1)
    data['min_row'] = data.min(axis=1)
    data['std_row'] = data.std(axis=1)
    return data

X = get_stats_per_row(X)
test_data = get_stats_per_row(test_data)


# In[5]:


# get skewed features to impute median instead of mean
from scipy.stats import skew

def impute_skewed_features(data):
    skewed_feat = data.skew()
    skewed_feat = [*skewed_feat[abs(skewed_feat.values) > 1].index]

    for feat in skewed_feat:
        median = data[feat].median()
        data[feat] = data[feat].fillna(median)
        
    return data

X = impute_skewed_features(X)
test_data = impute_skewed_features(test_data)


# In[6]:


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

# In[7]:


# helper functions
def get_auc(y_true, y_hat):
    fpr, tpr, _ = roc_curve(y_true, y_hat)
    score = auc(fpr, tpr)
    return score


# params taken from:
# - [catb1: my own optuna study](https://www.kaggle.com/mlanhenke/tps-09-optuna-study-catboostclassifier)
# - [Stacking Ensemble for Beginner](https://www.kaggle.com/junhyeok99/stacking-ensemble-for-beginner)

# In[8]:


# best params
lgbm1_params = {
    'metric' : 'auc',
    'max_depth' : 3,
    'num_leaves' : 7,
    'n_estimators' : 5000,
    'colsample_bytree' : 0.3,
    'subsample' : 0.5,
    'random_state' : 42,
    'reg_alpha' : 18,
    'reg_lambda' : 17,
    'learning_rate' : 0.095,
    'device' : 'gpu',
    'objective' : 'binary'
}

lgbm2_params = {
    'metric' : 'auc',
    'objective': 'binary',
    'n_estimators': 10000,
    'random_state': 42,
    'learning_rate': 0.095,
    'subsample': 0.6,
    'subsample_freq': 1,
    'colsample_bytree': 0.4,
    'reg_alpha': 10.0,
    'reg_lambda': 1e-1,
    'min_child_weight': 256,
    'min_child_samples': 20,
    'device' : 'gpu',
    'max_depth' : 3,
    'num_leaves' : 7
}

lgbm3_params = {
    'metric' : 'auc',
    'objective' : 'binary',
    'device_type': 'gpu', 
    'n_estimators': 10000, 
    'learning_rate': 0.12230165751633416, 
    'num_leaves': 1400, 
    'max_depth': 8, 
    'min_child_samples': 3100, 
    'reg_alpha': 10, 
    'reg_lambda': 65, 
    'min_split_gain': 5.157818977461183, 
    'subsample': 0.5, 
    'subsample_freq': 1, 
    'colsample_bytree': 0.2
}

catb1_params = {
    'eval_metric' : 'AUC',
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

catb2_params = {
    'eval_metric' : 'AUC',
    'depth' : 5,
    'grow_policy' : 'SymmetricTree',
    'l2_leaf_reg' : 3.0,
    'random_strength' : 1.0,
    'learning_rate' : 0.1,
    'iterations' : 10000,
    'loss_function' : 'CrossEntropy',
    'task_type' : 'GPU',
    'devices' : '0',
    'verbose' : 0
}

xgb1_params = {
    'eval_metric' : 'auc',
    'lambda': 0.004562711234493688, 
    'alpha': 7.268146704546314, 
    'colsample_bytree': 0.6468987558386358, 
    'colsample_bynode': 0.29113878257290376, 
    'colsample_bylevel': 0.8915913499148167, 
    'subsample': 0.37130229826185135, 
    'learning_rate': 0.021671163563123198, 
    'grow_policy': 'lossguide', 
    'max_depth': 18, 
    'min_child_weight': 215, 
    'max_bin': 272,
    'n_estimators': 10000,
    'random_state': 0,
    'use_label_encoder': False,
    'objective': 'binary:logistic',
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'predictor': 'gpu_predictor'
}

xgb2_params = dict(
    eval_metric='auc',
    max_depth=3,
    subsample=0.5,
    colsample_bytree=0.5,
    learning_rate=0.01187431306013263,
    n_estimators=10000,
    n_jobs=-1,
    use_label_encoder=False,
    objective='binary:logistic',
    tree_method='gpu_hist',
    gpu_id=0,
    predictor='gpu_predictor'
)

xgb3_params = {
    'eval_metric': 'auc', 
    'objective': 'binary:logistic', 
    'tree_method': 'gpu_hist', 
    'gpu_id': 0, 
    'predictor': 'gpu_predictor', 
    'n_estimators': 10000, 
    'learning_rate': 0.01063045229441343, 
    'gamma': 0.24652519525750877, 
    'max_depth': 4, 
    'min_child_weight': 366, 
    'subsample': 0.6423040816299684, 
    'colsample_bytree': 0.7751264493218339, 
    'colsample_bylevel': 0.8675692743597421, 
    'lambda': 0, 
    'alpha': 10
}


# ### <div style='background:#3b606f;color:white;padding:0.5em;border-radius:0.2em'>Train Base Models</div>

# In[9]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import StratifiedKFold\nfrom sklearn.metrics import roc_curve, auc\nfrom lightgbm import LGBMClassifier\nfrom xgboost import XGBClassifier\nfrom catboost import CatBoostClassifier\n\n# create list[tuples] of base_models\nmodels = [\n    (\'lgbm1\', LGBMClassifier(**lgbm1_params)),\n#     (\'lgbm2\', LGBMClassifier(**lgbm2_params)),\n    (\'lgbm3\', LGBMClassifier(**lgbm3_params)),\n    (\'catb1\', CatBoostClassifier(**catb1_params)),\n    (\'catb2\', CatBoostClassifier(**catb2_params)),\n    (\'xgb1\', XGBClassifier(**xgb1_params)),\n    (\'xgb2\', XGBClassifier(**xgb2_params)),\n    (\'xgb3\', XGBClassifier(**xgb3_params))\n]\n\n# create dictionaries to store predictions\noof_pred_tmp = dict()\ntest_pred_tmp = dict()\nscores_tmp = dict()\n\n# create cv\nkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)\n\nfor fold, (idx_train, idx_valid) in enumerate(kf.split(X, y)):\n    # create train, validation sets\n    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]\n    X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]\n    \n    # fit & predict all models on the same fold\n    for name, model in models:\n        if name not in scores_tmp:\n            oof_pred_tmp[name] = list()\n            oof_pred_tmp[\'y_valid\'] = list()\n            test_pred_tmp[name] = list()\n            scores_tmp[name] = list()\n     \n        model.fit(\n            X_train, y_train,\n            eval_set=[(X_valid,y_valid)],\n#             early_stopping_rounds=500,\n            verbose=0\n        )\n        \n        # validation prediction\n        pred_valid = model.predict_proba(X_valid)[:,1]\n        score = get_auc(y_valid, pred_valid)\n        \n        scores_tmp[name].append(score)\n        oof_pred_tmp[name].extend(pred_valid)\n        \n        print(f"Fold: {fold + 1} Model: {name} Score: {score}")\n        print(\'--\'*20)\n        \n        # test prediction\n        y_hat = model.predict_proba(test_data)[:,1]\n        test_pred_tmp[name].append(y_hat)\n    \n    # store y_validation for later use\n    oof_pred_tmp[\'y_valid\'].extend(y_valid)\n        \n# print overall validation scores\nfor name, model in models:\n    print(f"Overall Validation Score | {name}: {np.mean(scores_tmp[name])}")\n    print(\'::\'*20)\n')


# ### <div style='background:#3b606f;color:white;padding:0.5em;border-radius:0.2em'>Simple Blending of Base Models</div>

# In[10]:


# create df with base predictions on test_data
base_test_predictions = pd.DataFrame(
    {name: np.mean(np.column_stack(test_pred_tmp[name]), axis=1) 
    for name in test_pred_tmp.keys()}
)

# save csv checkpoint
base_test_predictions.to_csv('./base_test_predictions.csv', index=False)

# create simple average blend 
base_test_predictions['simple_avg'] = base_test_predictions.mean(axis=1)

# create submission file with simple blend average
simple_blend_submission = sample_submission.copy()
simple_blend_submission['claim'] = base_test_predictions['simple_avg']
simple_blend_submission.to_csv('./simple_blend_submission.csv', index=False)


# In[11]:


# create training set for meta learner based on the oof_predictions of the base models
oof_predictions = pd.DataFrame(
    {name:oof_pred_tmp[name] for name in oof_pred_tmp.keys()}
)

# save csv checkpoint
oof_predictions.to_csv('./oof_predictions.csv', index=False)

# get simple blend validation score
y_valid = oof_predictions['y_valid'].copy()
y_hat_blend = oof_predictions.drop(columns=['y_valid']).mean(axis=1)
score = get_auc(y_valid, y_hat_blend)

print(f"Overall Validation Score | Simple Blend: {score}")
print('::'*20)


# ### <div style='background:#3b606f;color:white;padding:0.5em;border-radius:0.2em'>Train Meta Learner</div>

# In[12]:


get_ipython().run_cell_magic('time', '', 'from sklearn.linear_model import LogisticRegression\n\n# prepare meta_training set\nX_meta = oof_predictions.drop(columns=[\'y_valid\']).copy()\ny_meta = oof_predictions[\'y_valid\'].copy()\ntest_meta = base_test_predictions.drop(columns=[\'simple_avg\']).copy()\n\nmeta_pred_tmp = []\nscores_tmp = []\n\n# create cv\nkf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)\n\nfor fold, (idx_train, idx_valid) in enumerate(kf.split(X_meta, y_meta)):\n    # create train, validation sets\n    X_train, y_train = X_meta.iloc[idx_train], y_meta.iloc[idx_train]\n    X_valid, y_valid = X_meta.iloc[idx_valid], y_meta.iloc[idx_valid]\n\n    model = LogisticRegression()\n    model.fit(X_train, y_train)\n    \n    # validation prediction\n    pred_valid = model.predict_proba(X_valid)[:,1]\n    score = get_auc(y_valid, pred_valid)\n    scores_tmp.append(score)\n    \n    print(f"Fold: {fold + 1} Score: {score}")\n    print(\'--\'*20)\n    \n    # test prediction based on oof_set\n    y_hat = model.predict_proba(test_meta)[:,1]\n    meta_pred_tmp.append(y_hat)\n    \n# print overall validation scores\nprint(f"Overall Validation Score | Meta: {np.mean(scores_tmp)}")\nprint(\'::\'*20)\n')


# In[13]:


# average meta predictions over each fold
meta_predictions = np.mean(np.column_stack(meta_pred_tmp), axis=1)

# create submission file
stacked_submission = sample_submission.copy()
stacked_submission['claim'] = meta_predictions
stacked_submission.to_csv('./stacked_submission.csv', index=False)

