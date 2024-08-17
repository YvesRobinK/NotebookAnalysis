#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this competition, our task is to predict machine failure based on 13 features:
# 1. Product ID
# 2. Type
# 3. Air temperature
# 4. Process temperature
# 5. Torque
# 6. Rotational speed
# 7. Tool wear
# 8. TWF
# 9. HDF
# 10. PWF
# 11. OSF
# 12. RNF

# # Loading Libraries

# In[1]:


get_ipython().system('pip install iterative-stratification')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import optuna

from category_encoders import OneHotEncoder, MEstimateEncoder, GLMMEncoder, OrdinalEncoder, CatBoostEncoder
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, KFold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_error, roc_auc_score, roc_curve
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

sns.set_theme(style = 'white', palette = 'viridis')
pal = sns.color_palette('viridis')

pd.set_option('display.max_rows', 100)


# In[3]:


train = pd.read_csv(r'../input/playground-series-s3e17/train.csv')
test_1 = pd.read_csv(r'../input/playground-series-s3e17/test.csv')
orig_train = pd.read_csv(r'../input/machine-failure-predictions/machine failure.csv')

train.drop('id', axis = 1, inplace = True)
test = test_1.drop('id', axis = 1)
orig_train.drop('UDI', axis = 1, inplace = True)


# # Descriptive Statistics

# In[4]:


train.head(10)


# In[5]:


desc = pd.DataFrame(index = list(train))
desc['count'] = train.count()
desc['nunique'] = train.nunique()
desc['%unique'] = desc['nunique'] / len(train) * 100
desc['null'] = train.isnull().sum()
desc['type'] = train.dtypes
desc = pd.concat([desc, train.describe().T.drop('count', axis = 1)], axis = 1)
desc


# In[6]:


desc = pd.DataFrame(index = list(test))
desc['count'] = test.count()
desc['nunique'] = test.nunique()
desc['%unique'] = desc['nunique'] / len(test) * 100
desc['null'] = test.isnull().sum()
desc['type'] = test.dtypes
desc = pd.concat([desc, test.describe().T.drop('count', axis = 1)], axis = 1)
desc


# In[7]:


desc = pd.DataFrame(index = list(orig_train))
desc['count'] = orig_train.count()
desc['nunique'] = orig_train.nunique()
desc['%unique'] = desc['nunique'] / len(orig_train) * 100
desc['null'] = orig_train.isnull().sum()
desc['type'] = orig_train.dtypes
desc = pd.concat([desc, orig_train.describe().T.drop('count', axis = 1)], axis = 1)
desc


# **Key point**: There are 6 categorical features and 6 numerical features if we remove all alphabet from `Product ID`

# In[8]:


categorical_features = ['Product ID', 'Type', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
numerical_features = test.drop(categorical_features, axis = 1).columns


# # Feature Name Preprocessing

# In[9]:


#https://stackoverflow.com/questions/48645846/pythons-xgoost-valueerrorfeature-names-may-not-contain-or
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
train.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in train.columns.values]
orig_train.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in orig_train.columns.values]
test.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in test.columns.values]


# # Preparation

# In[10]:


combo_train = pd.concat([train, orig_train])


# In[11]:


X = combo_train.copy()
y = X.pop('Machine failure')

seed = 42
splits = 10
k = MultilabelStratifiedKFold(n_splits = splits, random_state = seed, shuffle = True)
skf = StratifiedKFold(n_splits = 5, random_state = seed, shuffle = True)

np.random.seed(seed)


# # Baseline Models

# In[12]:


def cross_val_score(model, cv = k, label = '', include_original = False):
    
    X = train.copy()
    y = X.pop('Machine failure')
    
    #initiate prediction arrays and score lists
    val_predictions = np.zeros((len(train)))
    test_predictions = np.zeros((len(test)))
    #train_predictions = np.zeros((len(train)))
    train_scores, val_scores = [], []
    
    #training model, predicting prognosis probability, and evaluating log loss
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, train[['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']])):
        
        #define train set
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        
        #concat train set with original dataset if include_original is True
        if include_original == True:
            X_train = pd.concat([X_train, orig_train.drop('Machine failure', axis = 1)])
            y_train = pd.concat([y_train, orig_train['Machine failure']])
        
        #define validation set
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        #train model
        model.fit(X_train, y_train)
        
        #make predictions
        train_preds = model.predict_proba(X_train)[:,1]
        val_preds = model.predict_proba(X_val)[:, 1]
                  
        val_predictions[val_idx] += val_preds
        
        #evaluate model for a fold
        train_score = roc_auc_score(y_train, train_preds)
        val_score = roc_auc_score(y_val, val_preds)
        
        #append model score for a fold to list
        train_scores.append(train_score)
        val_scores.append(val_score)
        
        test_predictions += model.predict_proba(test)[:,1] / splits
    
    print(f'Val MAE: {np.mean(val_scores):.5f} ± {np.std(val_scores):.5f} | Train MAE: {np.mean(train_scores):.5f} ± {np.std(train_scores):.5f} | {label}')
    
    return val_scores, val_predictions, test_predictions


# In[13]:


score_list, oof_list = pd.DataFrame(), pd.DataFrame()
models = []


# In[14]:


Encoder = CatBoostEncoder(cols = ['Product ID', 'Type'])


# # Feature Engineering

# In[15]:


class FailureTypeFE(BaseEstimator, TransformerMixin):
    def fit(self, x, y = None): return self
    def transform(self, x, y = None):
        x_copy = x.copy()
        x_copy['IsTrouble'] = x_copy.TWF | x_copy.HDF | x_copy.OSF | x_copy.PWF
        return x_copy


# # Naive Bayes

# In[16]:


GNB = CalibratedClassifierCV(make_pipeline(Encoder, FailureTypeFE(), StandardScaler(), PowerTransformer(), GaussianNB()), cv = skf)

score_list['GNB'], oof_list['GNB'], _ = cross_val_score(GNB, include_original = True)


# # LightGBM

# In[17]:


def lgb_objective(trial):
    params = {
        'learning_rate' : trial.suggest_float('learning_rate', .001, .1, log = True),
        'max_depth' : trial.suggest_int('max_depth', 2, 20),
        'subsample' : trial.suggest_float('subsample', .5, 1),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', .1, 1),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 15),
        'reg_lambda' : trial.suggest_float('reg_lambda', 0, 1),
        'reg_alpha' : trial.suggest_float('reg_alpha', 0, 1),
        'n_estimators' : trial.suggest_int('n_estimators', 100, 1000),
        'random_state' : seed,
        'subsample_freq' : 1,
        'objective' : 'binary'
    }
    
    optuna_model = make_pipeline(Encoder, LGBMClassifier(**params))
    
    optuna_score, optuna_val, _ = cross_val_score(optuna_model, include_original = True)
    
    return np.mean(optuna_score)

lgb_study = optuna.create_study(direction = 'maximize')


# In[18]:


LGB = LGBMClassifier(
    random_state = seed,
    subsample_freq = 1,
    learning_rate = .01900716552442574,
    max_depth = 18,
    subsample = .9914530850784232,
    colsample_bytree = .25953740492885596,
    min_child_weight = 1,
    reg_lambda = .13711420234705485,
    reg_alpha = .5350957634949961,
    n_estimators = 464
)

LGB = make_pipeline(Encoder, LGB)

score_list['LGB'], oof_list['LGB'], _ = cross_val_score(LGB, include_original = True)


# # Random Forest

# # XGBoost

# In[19]:


def xgb_objective(trial):
    params = {
        'eta' : trial.suggest_float('eta', .001, .1, log = True),
        'max_depth' : trial.suggest_int('max_depth', 2, 20),
        'subsample' : trial.suggest_float('subsample', .5, 1),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', .1, 1),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 15),
        'reg_lambda' : trial.suggest_float('reg_lambda', 0, 1),
        'reg_alpha' : trial.suggest_float('reg_alpha', 0, 1),
        'n_estimators' : trial.suggest_int('n_estimators', 100, 1000),
        'random_state' : seed,
        'tree_method' : 'hist',
    }
    
    optuna_model = make_pipeline(Encoder, XGBClassifier(**params))
    
    optuna_score, optuna_val, _ = cross_val_score(optuna_model, include_original = True)
    
    return np.mean(optuna_score)

xgb_study = optuna.create_study(direction = 'maximize')


# In[20]:


XGB = XGBClassifier(
    random_state = seed,
    tree_method = 'hist',
    eta = .005881767577411666,
    max_depth = 21,
    subsample = .6480153695329861,
    colsample_bytree = .3053943349038963,
    min_child_weight = 5,
    reg_lambda = .4416157637771767,
    reg_alpha = .06263026301442894,
    n_estimators = 910
)

XGB = make_pipeline(Encoder, XGB)

score_list['XGB'], oof_list['XGB'], _ = cross_val_score(XGB, include_original = True)


# # CatBoost

# In[21]:


def cb_objective(trial):
    params = {
        'learning_rate' : trial.suggest_float('learning_rate', .001, .1, log = True),
        'max_depth' : trial.suggest_int('max_depth', 2, 10),
        'subsample' : trial.suggest_float('subsample', .5, 1),
        'colsample_bylevel' : trial.suggest_float('colsample_bylevel', .1, 1),
        'min_child_samples' : trial.suggest_int('min_child_samples', 1, 15),
        'reg_lambda' : trial.suggest_float('reg_lambda', 0, 1),
        'n_estimators' : trial.suggest_int('n_estimators', 100, 1000),
        'random_state' : seed,
        'bootstrap_type' : 'Bernoulli',
        'verbose' : 0,
        'cat_features' : ['Product ID', 'Type']
    }
    
    optuna_model = CatBoostClassifier(**params)
    
    optuna_score, optuna_val, _ = cross_val_score(optuna_model, include_original = True)
    
    return np.mean(optuna_score)

cb_study = optuna.create_study(direction = 'maximize')


# In[22]:


CB = CatBoostClassifier(
    random_state = seed,
    bootstrap_type = 'Bernoulli',
    verbose = 0,
    cat_features = ['Product ID', 'Type'],
    learning_rate = .018480276482909272,
    max_depth = 9,
    subsample = .8258690838674613,
    colsample_bylevel = .7172593798586308,
    min_child_samples = 6,
    reg_lambda = .6476166291805024,
    n_estimators = 574,
)

score_list['CB'], oof_list['CB'], _ = cross_val_score(CB, include_original = True)


# # Ensemble
# 
# Now let's try to build an ensemble. For simplicity, we will use RidgeClassifier to find the optimal weight.

# In[23]:


weights = LogisticRegression(random_state = seed).fit(oof_list, train['Machine failure']).coef_[0]

pd.DataFrame(weights, index = oof_list.columns, columns = ['weight per model'])


# In[24]:


models = [
#    ('Log', Log),
#    ('BNB', BNB),
    ('GNB', GNB),
    ('LGB', LGB),
    ('XGB', XGB),
    ('CB', CB)
]

voter = VotingClassifier(models, weights = weights, voting = 'soft')

_ = cross_val_score(voter, label = 'Voting Ensemble', include_original = True)


# # Retraining
# 
# Now that we've done our cross-validation, let's train our model on the whole dataset.

# In[25]:


prediction = voter.fit(X, y).predict_proba(test)[:,1]


# # Submission

# In[26]:


test_1.drop(list(test_1.drop('id', axis = 1)), axis = 1, inplace = True)


# In[27]:


test_1['Machine failure'] = prediction
test_1.to_csv('submission.csv', index = False)


# Thank you for reading!
