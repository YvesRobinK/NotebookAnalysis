#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
from functools import partial
import json

import pandas as pd
import numpy as np
import scipy as sp

from sklearn.model_selection import StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold, GridSearchCV, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import roc_auc_score, cohen_kappa_score, confusion_matrix, classification_report
qwk = partial(cohen_kappa_score, weights='quadratic')

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, TransformerMixin

from category_encoders import TargetEncoder, LeaveOneOutEncoder, WOEEncoder

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor, log_evaluation, early_stopping
from catboost import CatBoostClassifier, CatBoostRegressor

import torch
DEVICE = 'gpu' if torch.cuda.is_available() else 'cpu'
DEVICE_XGB = 'gpu_hist' if torch.cuda.is_available() else 'auto'

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('notebook', font_scale=1.1)

# Uncomment to use AutoML
get_ipython().system('pip install -q flaml')
import flaml

# !pip install -q autogluon.tabular[all]
# from autogluon.tabular import TabularPredictor


# ## Data at a glance
# An ordinal classification problem predicting wine quality (score between 0 and 10). Original dataset is [Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset).

# In[2]:


df_train = pd.read_csv('/kaggle/input/playground-series-s3e5/train.csv', index_col='Id')
df_test = pd.read_csv('/kaggle/input/playground-series-s3e5/test.csv', index_col='Id')
df_original = pd.read_csv('/kaggle/input/wine-quality-dataset/WineQT.csv', index_col='Id')

print(df_train.info())
display(df_train.head())

pd.DataFrame(
    [len(df_train), len(df_test), len(df_original)],
    index=['train', 'test', 'original'],
    columns=['count']
)


# <div class='alert alert-block alert-success'><b>Notes:</b>
#     
# - All features are numerical.
# - The datasets are pretty small.

# In[3]:


TARGET = 'quality'
pd.concat([
    pd.DataFrame(df_train.drop(columns=[TARGET]).isnull().sum(), columns=['missing train']),
    pd.DataFrame(df_test.isnull().sum(), columns=['missing test']),
    pd.DataFrame(df_original.drop(columns=[TARGET]).isnull().sum(), columns=['missing original'])
], axis=1)


# <div class='alert alert-block alert-success'><b>Insights:</b>
#     
# - There are no missing values in any sets.

# ## Target distribution

# In[4]:


df = pd.concat([
    pd.DataFrame(df_train[TARGET].value_counts()),
    pd.DataFrame(df_original[TARGET].value_counts()),
    pd.DataFrame(df_train[TARGET].value_counts(normalize=True)*100).round(1),
    pd.DataFrame(df_original[TARGET].value_counts(normalize=True)*100).round(1)
], axis=1)
df.columns=['train', 'test', 'train (%)', 'test (%)']
df.index.name = TARGET
df


# <div class='alert alert-block alert-success'><b>Insights:</b>
#     
# - Even though the quality column range is 0-10, the actual data is just between 3 and 8.
# - The majority are in the middle range 5-6 whereas the lowest quality 3 is really small. It is needed to use a stratified CV here with just 5 folds.

# ## Distribution of numerical features by set

# In[5]:


NUM_FEATURES = [c for c in df_train.columns if c not in [TARGET]]
ncols = 3
nrows = np.ceil(len(NUM_FEATURES)/ncols).astype(int)
fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(15,nrows*4))
for c, ax in zip(NUM_FEATURES, axs.flatten()):
    sns.kdeplot(data=df_train, x=c, ax=ax, label='train')
    sns.kdeplot(data=df_test, x=c, ax=ax, label='test')
    sns.kdeplot(data=df_original, x=c, ax=ax, label='original')
    ax.legend(loc='upper right', prop={'size': 10})

fig.suptitle('Distribution of numerical features')
plt.tight_layout(rect=[0, 0, 1, 0.98])


# <div class='alert alert-block alert-success'><b>Insights:</b>
#     
# - Distribution in all 3 sets look very similar.
# - There is some difference for `volatile acidity` and `free sulfur dioxide`.

# ## Distribution of numerical features in the train set by target variable

# In[6]:


def plot_features_by_target(df, num_features):
    """Display all columns in df except TARGET group by TARGET.
    """
    saved_type = df[TARGET].dtype
    df[TARGET] = df[TARGET].astype('category')
    columns = [c for c in df.columns if c != TARGET]
    ncols = 3
    nrows = np.ceil(len(columns)/ncols).astype(int)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(15,nrows*4))
    for c, ax in zip(columns, axs.flatten()):
        if c in num_features:
            sns.boxplot(data=df, x=c, y=TARGET, ax=ax)
        else:
            sns.countplot(data=df, x=c, hue=TARGET, ax=ax)
    fig.suptitle('Distribution of variables grouped by the target variable', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    df[TARGET] = df[TARGET].astype(saved_type)
    
plot_features_by_target(df_train, NUM_FEATURES)


# <div class='alert alert-block alert-success'><b>Insights:</b>
#     
# - `sulphates` and `alcohol`: the higher the value, the higher the quality

# ## Correlation in the train set
# The above observation could be checked more easily by looking at correlation directly.

# In[7]:


plt.figure(figsize=(8,8))
corr = df_train[NUM_FEATURES + [TARGET]].corr()
annot_labels = np.where(corr.abs() > 0.4, corr.round(1).astype(str), '')
upper_triangle = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=upper_triangle,
    vmin=-1, vmax=1, center=0, square=True, annot=annot_labels,
    cmap='coolwarm', linewidths=.5, fmt=''
)


# <div class='alert alert-block alert-success'><b>Insights:</b>
#     
# - There are some high correlation (>0.5) `alcohol` has the highest correlation with the target variable.
# - Plotting 2 more similar heatmaps for test and original sets making it really difficult to compare. We can just directly look at the pairs with highest difference in correlation among the 3 sets.

# Build a dataset with all absolute difference of correlation pairs among train, test and original sets.

# In[8]:


# Check if any pair that is larger than a threshold
corr_train = df_train[NUM_FEATURES].corr()
corr_test = df_test[NUM_FEATURES].corr()
corr_original = df_original[NUM_FEATURES].corr()
diff_corr = []
for i, col_i in enumerate(corr_train.columns):
    for j, col_j in enumerate(corr_train.columns):
        if i < j:
            diff_corr.append((col_i, 'train', col_j, 'test', abs(corr_train.values[i][j] - corr_test.values[i][j])))
            diff_corr.append((col_i, 'train', col_j, 'original', abs(corr_train.values[i][j] - corr_original.values[i][j])))
            diff_corr.append((col_i, 'original', col_j, 'test', abs(corr_original.values[i][j] - corr_test.values[i][j])))
df = pd.DataFrame(diff_corr, columns=['feature1', 'set1', 'feature2', 'set2', 'abs_diff'])
display(df.sort_values('abs_diff', ascending=False).head(5))


# Filter to show only the differences between train and test sets.

# In[9]:


display(df.query('set1 == "train" & set2 == "test"').sort_values('abs_diff', ascending=False).head(5))


# <div class='alert alert-block alert-success'><b>Insights:</b>
#     
# - Correlation between `chlorides` and `sulphates` in the original set is quite different from that in the train/test sets.
# - Correlation between features are consistent between the train and test sets.

# ## Preparation for model building
# - Add original data indicator feature
# - Apply LabelEncoder to convert the range starting with 0 (for xgboost)
# - Use StratifiedKFold with 5 splits

# In[10]:


encoder = LabelEncoder()
y_train = encoder.fit_transform(df_train[TARGET])
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


# In[11]:


def build_pipeline(model_fn, num_features):
    num_proc = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
    processing = ColumnTransformer([
        ('num', num_proc, num_features)
    ])
    
    return Pipeline([ 
        ('proc', processing),
        ('model', model_fn())
    ])

def run(df_train, y_train, df_org=None, y_org=None, df_test=None, cv=CV, model_fn=None, num_features=NUM_FEATURES, early_stopping_rounds=500, return_models=False, save_file=None, verbose=False):
    oof_preds = np.zeros(len(df_train))
    pipelines = []
    scores = []
    rounders = []

    for fold, (idx_tr, idx_vl) in enumerate(cv.split(df_train, y_train)):
        # Fold train: add the entire original data
        df_tr, y_tr = df_train.iloc[idx_tr], y_train[idx_tr]
        if df_org is not None:
            df_tr = pd.concat([df_tr, df_org])
            y_tr = np.hstack([y_tr, y_org])
            
        # Fold validation: just synthetic data
        df_vl, y_vl = df_train.iloc[idx_vl], y_train[idx_vl]
        
        # eval_set for early stopping
        pipeline = build_pipeline(model_fn, num_features)
        X_vl = pipeline['proc'].fit_transform(df_vl, y_vl)
        eval_set = [(X_vl, y_vl)]
        
        if type(pipeline['model']) in [CatBoostClassifier, CatBoostRegressor]:
            pipeline.fit(df_tr, y_tr, model__eval_set=eval_set, model__early_stopping_rounds=early_stopping_rounds, model__verbose=verbose)
        elif type(pipeline['model']) in [XGBClassifier, XGBRegressor]:
            pipeline['model'].early_stopping_rounds = early_stopping_rounds
            pipeline.fit(df_tr, y_tr, model__eval_set=eval_set, model__verbose=verbose)
        elif type(pipeline['model']) in [LGBMClassifier, LGBMRegressor]:
            callbacks = [early_stopping(early_stopping_rounds), log_evaluation(-1)]
            pipeline.fit(df_tr, y_tr, model__eval_set=eval_set, model__callbacks=callbacks)
        else:
            pipeline.fit(df_tr, y_tr)
            
        oof_preds[idx_vl] = pipeline.predict(df_vl).squeeze()
        
        # Round optimizer 
        if type(pipeline['model']) in [CatBoostRegressor, XGBRegressor, LGBMRegressor]:
            optimized_rounder = OptimizedRounder()
            optimized_rounder.fit(oof_preds[idx_vl], y_vl)
            oof_preds[idx_vl] = optimized_rounder.predict(oof_preds[idx_vl], optimized_rounder.coefficients())
            rounders.append(optimized_rounder)
        
        score = qwk(y_vl, oof_preds[idx_vl])
        scores.append(score)
        pipelines.append(pipeline)
        
        if verbose:
            print(f'Fold {fold} score={score:.4}')

    print(f'   OOF score={np.mean(scores):.4}')
    
    if save_file is not None:
        df = pd.DataFrame(data={'Id': df_train.index, TARGET: encoder.inverse_transform(oof_preds.astype(int))})
        df.to_csv(f'{save_file}_oof_preds.csv', index=None)
        
        y_pred = []
        if type(pipeline['model']) in [CatBoostRegressor, XGBRegressor, LGBMRegressor]:
            for p, rounder in zip(pipelines, rounders):
                pred = p.predict(df_test).squeeze()
                pred = rounder.predict(pred, rounder.coefficients())
                y_pred.append(pred)
        else:
            y_pred = [p.predict(df_test).squeeze() for p in pipelines]
            
        y_pred = pd.DataFrame(y_pred).T.mode(axis=1)[0]
        df = pd.DataFrame(data={'Id': df_test.index, TARGET: encoder.inverse_transform(y_pred.astype(int))})
        df.to_csv(f'{save_file}_test_preds.csv', index=None)
    
    if return_models:
        return pipelines


# ## First baseline

# In[12]:


run(df_train, y_train, model_fn=partial(LogisticRegression, max_iter=1000, random_state=0))


# In[13]:


run(df_train, y_train, model_fn=partial(LogisticRegression, max_iter=1000, class_weight='balanced', random_state=0))


# In[14]:


run(df_train, y_train, model_fn=partial(LGBMClassifier, random_state=0))


# In[15]:


run(df_train, y_train, model_fn=partial(CatBoostClassifier, random_state=0))


# In[16]:


run(df_train, y_train, model_fn=partial(XGBClassifier, random_state=0))


# <div class='alert alert-block alert-success'><b>Insights:</b>
#     
# - The default logistic regression seems too bad. Setting `class_weight` makes it worse.
# - Default LGBM is better than default CatBoost and xgboost.

# ## Making first submission

# In[17]:


lgbm_pipelines = run(df_train, y_train, df_test=df_test, model_fn=partial(LGBMClassifier, random_state=0), save_file='01_untuned_lgbm', return_models=True)


# ## Where are the model mistakes?

# In[18]:


df = pd.read_csv('01_untuned_lgbm_oof_preds.csv')
y_train_corrected = encoder.inverse_transform(y_train)
labels = sorted(set(y_train_corrected))
matrix = confusion_matrix(y_train_corrected, df['quality'], labels=labels)
df_matrix = pd.DataFrame(matrix, index=labels, columns=labels)

plt.figure(figsize=(8,8))
ax = sns.heatmap(df_matrix, cmap='rocket_r', annot=True, fmt='', square=True)
ax.set_xlabel('Prediction')
ax.set_ylabel('Truth')
plt.show()
print(classification_report(y_train_corrected, df['quality']))


# <div class='alert alert-block alert-success'><b>Insights:</b>
#     
# - The model doesn't predict any wine with class 3 or 4.
# - Class 8 is also very poor, close to disaster.
# - Class 5 and 6 are ok.

# ## Which features does the model think as important?

# In[19]:


df = pd.DataFrame({'feature': NUM_FEATURES})
df['importance'] = np.array([p['model'].feature_importances_ for p in lgbm_pipelines]).mean(axis=0)
plt.figure(figsize=(8,8))
sns.barplot(data=df.sort_values('importance'), x='importance', y='feature')


# <div class='alert alert-block alert-success'><b>Insights:</b>
#     
# - `sulphates` and `alcohol` influence the model the most. This aligns with the correlation discovered before.

# ## Feature engineering
# Adding both features suggested by [ChatGPT](https://www.kaggle.com/competitions/playground-series-s3e5/discussion/383685) and manually crafted features by [Jose Caliz](https://www.kaggle.com/competitions/playground-series-s3e5/discussion/382698).

# In[20]:


def add_features(df):
    # From https://www.kaggle.com/competitions/playground-series-s3e5/discussion/383685
    df['acidity_ratio'] = df['fixed acidity'] / df['volatile acidity']
    df['free_sulfur/total_sulfur'] = df['free sulfur dioxide'] / df['total sulfur dioxide']
    df['sugar/alcohol'] = df['residual sugar'] / df['alcohol']
    df['alcohol/density'] = df['alcohol'] / df['density']
    df['total_acid'] = df['fixed acidity'] + df['volatile acidity'] + df['citric acid']
    df['sulphates/chlorides'] = df['sulphates'] / df['chlorides']
    df['bound_sulfur'] = df['total sulfur dioxide'] - df['free sulfur dioxide']
    df['alcohol/pH'] = df['alcohol'] / df['pH']
    df['alcohol/acidity'] = df['alcohol'] / df['total_acid']
    df['alkalinity'] = df['pH'] + df['alcohol']
    df['mineral'] = df['chlorides'] + df['sulphates'] + df['residual sugar']
    df['density/pH'] = df['density'] / df['pH']
    df['total_alcohol'] = df['alcohol'] + df['residual sugar']
    
    # From https://www.kaggle.com/competitions/playground-series-s3e5/discussion/382698
    df['acid/density'] = df['total_acid']  / df['density']
    df['sulphate/density'] = df['sulphates']  / df['density']
    df['sulphates/acid'] = df['sulphates'] / df['volatile acidity']
    df['sulphates*alcohol'] = df['sulphates'] * df['alcohol']
    
    return df

df_fe = add_features(df_train.copy())
new_num_features = NUM_FEATURES + df_fe.columns.difference(df_train.columns).tolist()
lgbm_pipelines = run(df_fe, y_train, model_fn=partial(LGBMClassifier, random_state=0), num_features=new_num_features, return_models=True)
df = pd.DataFrame({'feature': new_num_features})
df['importance'] = np.array([p['model'].feature_importances_ for p in lgbm_pipelines]).mean(axis=0)
plt.figure(figsize=(8,8))
sns.barplot(data=df.sort_values('importance'), x='importance', y='feature')


# <div class='alert alert-block alert-warning'><b>Insights:</b>
#     
# - The new features don't help!

# ## Adding original data

# In[21]:


y_original = encoder.transform(df_original[TARGET])
df_train['original'] = 0
df_original['original'] = 1
df_test['original'] = 0
org_num_features = NUM_FEATURES + ['original']

run(df_train, y_train, df_org=df_original, y_org=y_original, df_test=df_test, model_fn=partial(LGBMClassifier, random_state=0), num_features=org_num_features, save_file='02_untuned_lgbm_extra_data')


# <div class='alert alert-block alert-warning'><b>Insights:</b>
#     
# - Adding original data makes the model perform worse. Did I make any mistakes?
# - The 1st model (0.504-CV) has 0.51574-LB but the 2nd model (0.4801-CV) has 0.52633-LB. Our CV isn't reliable?

# ## AutoML

# In[22]:


TIME_BUDGET = 60 * 60 * 4
EARLY_STOPPING_ROUNDS = 500


# In[23]:


# auto_flaml = flaml.AutoML()
# auto_flaml.fit(df_train, y_train, task='classification', estimator_list=['catboost'], time_budget=TIME_BUDGET, early_stop=EARLY_STOPPING_ROUNDS, verbose=0)
# print(auto_flaml.best_config)
# with open(f'tuned_{TIME_BUDGET}_catboost.json', 'w') as f:
#     f.write(json.dumps(auto_flaml.best_config))
    
# run(df_train, y_train, model_fn=partial(CatBoostClassifier, **auto_flaml.best_config, random_state=0), early_stopping_rounds=EARLY_STOPPING_ROUNDS)


# In[24]:


lgbm_params = {'n_estimators': 394, 'num_leaves': 4, 'min_child_samples': 4, 'learning_rate': 0.15961062177409088, 'colsample_bytree': 0.9999857982053186, 'reg_alpha': 0.0009765625, 'reg_lambda': 0.0009765625}
run(df_train, y_train, df_test=df_test, model_fn=partial(LGBMClassifier, random_state=0, **lgbm_params), save_file='03_tuned_lgbm')


# In[25]:


# This is better than my tuned parameters: https://www.kaggle.com/code/alexandershumilin/ps-s3-e5-ensemble-model
# xgb_params = {'n_estimators': 27, 'max_leaves': 71, 'min_child_weight': 0.001, 'learning_rate': 0.9660323602219278, 'subsample': 0.9990858742619992, 'colsample_bylevel': 1.0, 'colsample_bytree': 1.0, 'reg_alpha': 0.0009765625, 'reg_lambda': 0.0009765625}
xgb_params = {
  'n_estimators'    : 2000,
  'subsample'       : 0.1,
  'reg_lambda'      : 50,
  'min_child_weight': 1,
  'max_depth'       : 6,
  'learning_rate'   : 0.05,
  'colsample_bytree': 0.4
}
run(df_train, y_train, df_test=df_test, model_fn=partial(XGBClassifier, random_state=0, **xgb_params), save_file='04_tuned_xgb')


# In[26]:


# This is better than my tuned parameters: https://www.kaggle.com/code/alexandershumilin/ps-s3-e5-ensemble-model
# cb_params = {'learning_rate': 0.17489380497226825, 'n_estimators': 8192}
cb_params = {
  'n_estimators'    : 3000,
  'learning_rate'   : 0.01,
  'depth'           : 3,
  'min_data_in_leaf': 25,
  'l2_leaf_reg'     : 70
 }
run(df_train, y_train, df_test=df_test, model_fn=partial(CatBoostClassifier, random_state=0, **cb_params), save_file='05_tuned_cb')


# ## Using regression
# with `OptimizedRounder` class from https://www.kaggle.com/competitions/petfinder-adoption-prediction/discussion/76107.

# In[27]:


from functools import partial
import numpy as np
import scipy as sp

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            else:
                X_p[i] = 5

        ll = qwk(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            else:
                X_p[i] = 5
        return X_p

    def coefficients(self):
        return self.coef_['x']


# In[28]:


run(df_train, y_train, df_test=df_test, model_fn=partial(LGBMRegressor, random_state=0, **lgbm_params), save_file='06_tuned_lgbm_regressor')


# In[29]:


run(df_train, y_train, df_test=df_test, model_fn=partial(XGBRegressor, random_state=0, **xgb_params), save_file='07_tuned_xgb_regressor')


# In[30]:


run(df_train, y_train, df_test=df_test, model_fn=partial(CatBoostRegressor, random_state=0, **cb_params), save_file='08_tuned_cb_regressor')


# <div class='alert alert-block alert-success'><b>Insights:</b>
#     
# - WOW. A huge improvement!

# ## Regression with original data

# In[31]:


run(df_train, y_train, df_org=df_original, y_org=y_original, df_test=df_test, model_fn=partial(LGBMRegressor, random_state=0, **lgbm_params), num_features=org_num_features, save_file='09_tuned_lgbm_regressor_extra_data')


# In[32]:


run(df_train, y_train, df_org=df_original, y_org=y_original, df_test=df_test, model_fn=partial(XGBRegressor, random_state=0, **xgb_params), num_features=org_num_features, save_file='10_tuned_xgb_regressor_extra_data')


# In[33]:


run(df_train, y_train, df_org=df_original, y_org=y_original, df_test=df_test, model_fn=partial(CatBoostRegressor, random_state=0, **cb_params), num_features=org_num_features, save_file='11_tuned_cb_regressor_extra_data')


# ## Regression with feature engineering

# ### Only synthetic data

# In[34]:


df_fe = add_features(df_train.copy())
df_test_fe = add_features(df_test.copy())
new_num_features = NUM_FEATURES + df_fe.columns.difference(df_train.columns).tolist()
print('lgbm')
run(df_fe, y_train, df_test=df_test_fe, model_fn=partial(LGBMRegressor, random_state=0, **lgbm_params), num_features=new_num_features, save_file='12_tuned_lgbm_regressor_fe')
print('xgboost')
run(df_fe, y_train, model_fn=partial(XGBRegressor, random_state=0, **xgb_params), num_features=new_num_features)
print('catboost')
run(df_fe, y_train, df_test=df_test_fe, model_fn=partial(CatBoostRegressor, random_state=0, **cb_params), num_features=new_num_features, save_file='13_tuned_cb_regressor_fe')


# ### With original data

# In[35]:


df_original_fe = add_features(df_original.copy())
print('lgbm')
run(df_fe, y_train, df_org=df_original_fe, y_org=y_original, df_test=df_test_fe, model_fn=partial(LGBMRegressor, random_state=0, **lgbm_params), num_features=new_num_features)
print('xgboost')
run(df_fe, y_train, df_org=df_original_fe, y_org=y_original, model_fn=partial(XGBRegressor, random_state=0, **xgb_params), num_features=new_num_features)
print('catboost')
run(df_fe, y_train, df_org=df_original_fe, y_org=y_original, df_test=df_test_fe,model_fn=partial(CatBoostRegressor, random_state=0, **cb_params), num_features=new_num_features)


# ## Summary of submissions

# In[36]:


df_sub = pd.DataFrame([
    ('01_untuned_lgbm', 0.504, 0.51574),
    ('02_untuned_lgbm_extra_data', 0.4801, 0.52633),
    ('03_tuned_lgbm', 0.5124, 0.51148),
    ('04_tuned_xgb', 0.4927, 0.53412),
    ('05_tuned_cb', 0.5153, 0.54142),
    ('06_tuned_lgbm_regressor', 0.5797, 0.57465),
    ('07_tuned_xgb_regressor', 0.5697, 0.55185),
    ('08_tuned_cb_regressor', 0.5827, 0.58144),
    ('09_tuned_lgbm_regressor_extra_data', 0.5746, 0.57461),
    ('10_tuned_xgb_regressor_extra_data', 0.5771, 0.59257),
    ('11_tuned_cb_regressor_extra_data', 0.5735, 0.58171),
    ('12_tuned_lgbm_regressor_fe', 0.5814, 0.57716),
    ('13_tuned_cb_regressor_fe', 0.5825, 0.59687)
], columns=['name', 'CV', 'LB'])
display(df_sub)
df_sub.plot.scatter(x='CV', y='LB')


# ## Ensemble of 7 best models

# In[37]:


dfs = []
for n in df_sub.sort_values('LB').iloc[-7:]['name']:
    filename = n + '_test_preds.csv'
    dfs.append(pd.read_csv(filename)[['quality']])
df_ensemble = pd.read_csv('01_untuned_lgbm_test_preds.csv')
df_ensemble['quality'] = pd.concat(dfs, axis=1).mode(axis=1)
df_ensemble.to_csv('14_ensemble_top7_test_preds.csv', index=False)


# <div class='alert alert-block alert-warning'><b>Insights:</b>
#     
# - Oops, the LB score comes down quite a lot?!
