#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('wget https://raw.githubusercontent.com/JoseCaliz/dotfiles/main/css/custom_css.css 2>/dev/null')

from IPython.core.display import HTML
with open('./custom_css.css', 'r') as file:
    custom_css = file.read()
    
HTML(custom_css)


# # Tabular Playground Series August 22 - XGB Starter
# 
# <div style=' height:150px; background: url("https://i.imgur.com/yNSrKpb.png");'></div>
# 
# # Table Of Content <span id='toc'/>
# 
# 1. [Table Of Content ](#Table-Of-Content-)
# 1. [Library Import ](#Library-Import-)
# 1. [Data Read ](#Data-Read-)
# 1. [Feature Engineering ](#Feature-Engineering-)
#   1. [Create Colum Transformer ](#Create-Colum-Transformer-)
# 1. [Modeling ](#Modeling-)
#   1. [Define Baseline Pipeline ](#Define-Baseline-Pipeline-)
#   1. [Group Cross Validation ](#Group-Cross-Validation-)
#   1. [Feature selection ](#Feature-selection-)
#   1. [Hyperparameter Search](#Hyperparameter-Search)
# 1. [Train Final Model](#Train-Final-Model)
# 1. [Generate Predictions ](#Generate-Predictions-)

# # Library Import <span id='library_import'/>
# 
# [Go back to TOC ⬆️](#toc)

# In[2]:


import pandas as pd
import numpy as np
import xgboost as xgb
import humanize
import category_encoders as ce
import warnings

from rich import print
from rich.progress import track
from skimage import io
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from category_encoders.target_encoder import TargetEncoder
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import RFECV
from sklearn.model_selection import ParameterGrid
from collections import defaultdict
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline


warnings.simplefilter(action='ignore', category=FutureWarning)
f = "https://i.imgur.com/58jc3OY.png"


# # Data Read <span id='data_read'/>
# 
# [Go back to TOC ⬆️](#toc)

# In[3]:


train_df = pd.read_csv('../input/tabular-playground-series-aug-2022/train.csv')
test_df = pd.read_csv('../input/tabular-playground-series-aug-2022/test.csv')

train_df.set_index('id', inplace=True)
test_df.set_index('id', inplace=True)


# # Feature Engineering <span id='feature_engineering'/>
# 
# ## Create Colum Transformer <span id='create_transformer'/>
# [Go back to TOC ⬆️](#toc)

# In[4]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def add_features(df):
    df['area'] = df['attribute_2'] * df['attribute_3']
    df['avg'] = df.filter(regex='measurement').apply(np.nanmean, axis=1)
    df['std'] = df.filter(regex='measurement').apply(np.nanstd, axis=1)

features_to_drop = ['failure', 'product_code']
feature_preprocessing = ColumnTransformer([
    ('attribute_0', OneHotEncoder(handle_unknown='ignore'), ['attribute_0']),
    ('attribute_1', OneHotEncoder(handle_unknown='ignore'), ['attribute_1']),
    ('attribute_2', OneHotEncoder(handle_unknown='ignore'), ['attribute_2']),
    ('attribute_3', OneHotEncoder(handle_unknown='ignore'), ['attribute_3']),
    ('drop_features', 'drop', features_to_drop),
], remainder='passthrough')

add_features(train_df)
add_features(test_df)


# # Modeling <span id='modeling'/>
# 
# ## Define Baseline Pipeline <span id='define_pipeline'/>
# 
# [Go back to TOC ⬆️](#toc)

# In[5]:


cv = GroupKFold(train_df.product_code.nunique())
pipeline = Pipeline([
    ('feature_preprocessing', feature_preprocessing),
    ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
    ('XGB', xgb.XGBClassifier(random_state=0, tree_method='gpu_hist'))
])


# ## Group Cross Validation <span id='group_cross_validation'/>
# [Go back to TOC ⬆️](#toc)

# In[6]:


cv_results = cross_validate(
    pipeline,
    train_df, train_df.failure,
    groups=train_df.product_code,
    scoring=['roc_auc'],
    return_train_score=True,
    return_estimator=True,
)

print('train', cv_results['train_roc_auc'], 'train_mean:', np.mean(cv_results['train_roc_auc']))
print('test', cv_results['test_roc_auc'], 'test_mean:', np.mean(cv_results['test_roc_auc']))


# In[7]:


fig, ax = plt.subplots(figsize=[7, 7])
a = io.imread(f)
plt.imshow(a)
plt.axis('off')
plt.show()


# ## Feature selection <span id='feature_selection'/>
# [Go back to TOC ⬆️](#toc)

# In[8]:


feature_processing_pipeline = Pipeline([
    ('feature_preprocessing', feature_preprocessing),
    ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
])

X_train = feature_processing_pipeline.fit_transform(train_df, train_df.failure)
X_test = feature_processing_pipeline.transform(test_df)
y_train = train_df.failure

feature_names = np.array(feature_preprocessing.get_feature_names())
selector = RFECV(
    xgb.XGBClassifier(
        random_state=0,
        max_depth=2,
        min_child_weight=14,
        gamma=13,
        n_estimators=100,
        tree_method='gpu_hist'
    ),
    min_features_to_select=1,
    cv=cv,
    step=1,
    scoring='roc_auc'
)

selector.fit(X_train, train_df.failure, groups=train_df.product_code);
support = selector.support_

print('features_used', list(feature_names[support]))


# In[9]:


support = feature_names[selector.ranking_ <= 6]
print(support)


# In[10]:


feature_names


# ## Hyperparameter Search
# [Go back to TOC ⬆️](#toc)

# In[11]:


param_grid = ParameterGrid({
    'max_depth': [3],
    'reg_alpha': [5],
    'max_leaves': [3, 4, 5, 6, 7],
#     'min_child_weight':np.arange(1, 10, 1),
#     'gamma':[14],
    'n_estimators': [5000, 5500],
    'ranking': [3],
    'learning_rate': [1e-5, 5e-5, 1e-4]
})

gs_results = defaultdict(dict)
for params in track(param_grid):
    ranking = params.pop('ranking')
    
    support = selector.ranking_ <= ranking
    support[0] = True
    
    feature_filter = FunctionTransformer(lambda X: X[:, support])
    classification_pipeline = make_pipeline(
        feature_filter,
         xgb.XGBClassifier(random_state=0,tree_method='gpu_hist', **params),
    )
    
    key = str(params)
    cv_scores = cross_validate(
        classification_pipeline,
        X_train, y_train,
        cv=cv,
        groups=train_df.product_code,
        scoring=['roc_auc'],
        return_train_score=True,
        return_estimator=True,
    )
    
    gs_results[key] = params
    gs_results[key]['ranking'] = ranking
    gs_results[key]['train_mean'] = np.array(cv_scores['train_roc_auc']).mean()
    gs_results[key]['test_mean'] = np.array(cv_scores['test_roc_auc']).mean()
    
gs_results_df = pd.DataFrame.from_dict(gs_results, orient='index')
gs_results_df['diff'] = gs_results_df.eval('(train_mean - test_mean)')

display(
    gs_results_df
    .reset_index(drop=True)
    .sort_values('diff')
    .drop_duplicates('diff', keep='first')
    .head(5)
)


# # Train Final Model
# [Go back to TOC ⬆️](#toc)

# In[12]:


params = dict(
    random_state=0,
    max_depth=3,
    reg_alpha=0.25,
    min_child_weight=10,
    gamma=14,
    n_estimators=100,
    tree_method='gpu_hist',
)

feature_filter = FunctionTransformer(lambda X: X[:, selector.support_])
classification_pipeline = make_pipeline(
    feature_filter,
     xgb.XGBClassifier(**params),
)

cv_scores = cross_validate(
    classification_pipeline,
    X_train, y_train,
    cv=cv,
    groups=train_df.product_code,
    scoring=['roc_auc'],
    return_train_score=True,
    return_estimator=True,
)

print('train', cv_scores['train_roc_auc'], np.mean(cv_scores['train_roc_auc']))
print('test', cv_scores['test_roc_auc'], np.mean(cv_scores['test_roc_auc']))


# # Generate Predictions <span id='generate_predictions'/>
# 
# 
# [Go back to TOC ⬆️](#toc)

# In[13]:


scores = np.zeros(test_df.shape[0])
for estimator in cv_scores['estimator']:
    scores += estimator.predict_proba(X_test)[:, 1]
    
scores /= len(cv_scores['estimator'])
submission = pd.DataFrame(dict(failure=scores), index=test_df.index)
submission.to_csv('submission.csv')

