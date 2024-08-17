#!/usr/bin/env python
# coding: utf-8

# September 30 - linked to GitHub

# # <h1 style = "font-family: Georgia;font-weight: bold; font-size: 30px; color: #1192AA; text-align:left">Import</h1>

# In[1]:


import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
from copy import deepcopy
from functools import partial
from itertools import combinations
import random
import gc

# Import sklearn classes for model selection, cross validation, and performance evaluation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from category_encoders import OneHotEncoder, OrdinalEncoder, CountEncoder, CatBoostEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, NMF
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error
from collections import defaultdict
from sklearn.model_selection import cross_validate
from sklearn.ensemble import StackingRegressor
from typing import List

# Import libraries for Hypertuning
import optuna

# Import libraries for gradient boosting
import xgboost as xgb
import lightgbm as lgb
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, LassoCV
from sklearn.linear_model import PassiveAggressiveRegressor, ARDRegression, RidgeCV, ElasticNetCV
from sklearn.linear_model import TheilSenRegressor, RANSACRegressor, HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression
from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier
from catboost import Pool

# Useful line of code to set the display option so we could see all the columns in pd dataframe
pd.set_option('display.max_columns', None)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


#  # <h1 style = "font-family: Georgia;font-weight: bold; font-size: 30px; color: #1192AA; text-align:left">Check Dataset</h1>

# In[2]:


PATH_ORIGIN = '/kaggle/input/wild-blueberry-yield-prediction-dataset/WildBlueberryPollinationSimulationData.csv'
PATH_TRAIN = '/kaggle/input/playground-series-s3e14/train.csv'
PATH_TEST = '/kaggle/input/playground-series-s3e14/test.csv'
PATH_SUB = '/kaggle/input/playground-series-s3e14/sample_submission.csv'

df_train =  pd.read_csv(PATH_TRAIN).drop(columns='id')
df_test =   pd.read_csv(PATH_TEST).drop(columns='id')
original = pd.read_csv(PATH_ORIGIN).drop(columns='Row#')

target_col = 'yield'


# In[3]:


print(f'[INFO] Shapes:'
      f'\n original: {original.shape}'
      f'\n train: {df_train.shape}'
      f'\n test: {df_test.shape}\n')

print(f'[INFO] Any missing values:'
      f'\n original: {original.isna().any().any()}'
      f'\n train: {df_train.isna().any().any()}'
      f'\n test: {df_test.isna().any().any()}')


# In[4]:


full_train = pd.concat([df_train, original], axis=0).reset_index(drop=True)


# In[5]:


full_train.head()


# # <h1 style = "font-family: Georgia;font-weight: bold; font-size: 30px; color: #1192AA; text-align:left">EDA</h1>

# In[6]:


# Create figure
fig = px.histogram(x = df_train[target_col],
                   template='simple_white',
                   color_discrete_sequence = ['#1192AA'])



# Set Title and x/y axis labels
fig.update_layout(
    xaxis_title="Yield Value",
    yaxis_title="Frequency",
    showlegend = False,
    font = dict(
            size = 14
            ),    
    title={
        'text': "Yield Distribution in `df_train`",
        'y':0.95,
        'x':0.5
        }
    )

# Display
fig.show()


# In[7]:


# Create figure
fig = px.histogram(x = original[target_col],
                   template='simple_white',
                   color_discrete_sequence = ['#1192AA'])



# Set Title and x/y axis labels
fig.update_layout(
    xaxis_title="Yield Value",
    yaxis_title="Frequency",
    showlegend = False,
    font = dict(
            size = 14
            ),    
    title={
        'text': "Yield Distribution in `original`",
        'y':0.95,
        'x':0.5
        }
    )

# Display
fig.show() # for Kaggle version


# In[8]:


# Create figure
fig = px.imshow(df_train.corr(), template='simple_white')

# Set Title and x/y axis labels
fig.update_layout(
    showlegend = False,
    font = dict(
            size = 14
            ),    
    title={
        'text': "Train Dataset Correlation",
        'y':0.98,
        'x':0.49
        }
    )

# Display
fig.show() 


# In[9]:


# Create figure
fig = px.imshow(original.corr(), template='simple_white')

# Set Title and x/y axis labels
fig.update_layout(
    showlegend = False,
    font = dict(
            size = 14
            ),    
    title={
        'text': "Original Dataset Correlation",
        'y':0.98,
        'x':0.49
        }
    )

# Display
fig.show() 


# In[10]:


# Plot function
def plot_column_distribution(df, column_name):
    """plot a distribution of certain column with [column_name] from [df] dataframe"""

    # Create figure
    fig = px.histogram(df[column_name],
                       template = 'simple_white',
                       color_discrete_sequence = ['#1192AA'])

    # Set Title and x/y axis labels
    fig.update_layout(
        xaxis_title="Value",
        yaxis_title="Frequency",
        showlegend = False,
        font = dict(
                size = 14
                ),    
        title={
            'text': column_name,
            'y':0.95,
            'x':0.5
            }
        )

    # Display
    fig.show()

for column in df_train.columns[0:-1]:
    plot_column_distribution(df_train, column)


# # <h1 style = "font-family: Georgia;font-weight: bold; font-size: 30px; color: #1192AA; text-align:left">Feature Engineering</h1>

# In[11]:


def add_features(df_in):
    df = df_in.copy(deep = True)
    
    df["fruit_seed"] = df["fruitset"] * df["seeds"]
    return df

df_train = add_features(df_train)
df_test = add_features(df_test)
original = add_features(original)


# # <h1 style = "font-family: Georgia;font-weight: bold; font-size: 30px; color: #1192AA; text-align:left">Preprocess</h1>

# In[12]:


# Concatenate train and original dataframes, and prepare train and test sets
df_train = pd.concat([df_train, original])
X_train = df_train.drop([f'{target_col}'],axis=1).reset_index(drop=True)
y_train = df_train[f'{target_col}'].reset_index(drop=True)
X_test = df_test.reset_index(drop=True)

# StandardScaler
categorical_columns = ['is_generated']
numeric_columns = [_ for _ in X_train.columns if _ not in categorical_columns]
sc = MinMaxScaler()
X_train[numeric_columns] = sc.fit_transform(X_train[numeric_columns])
X_test[numeric_columns] = sc.transform(X_test[numeric_columns])

# # Randomly sample 80% of the data
# X_train = X_train.sample(frac=0.8, random_state=42)

# pca = PCA(n_components=3)
# pca_train = pca.fit_transform(X_train)
# pca_test = pca.fit_transform(X_test)

# X_train= X_train.join(pd.DataFrame(pca_train))
# X_test= X_test.join(pd.DataFrame(pca_train))

# X_train.columns = X_train.columns.astype(str)
# X_test.columns = X_test.columns.astype(str)

print(f"X_train shape :{X_train.shape} , y_train shape :{y_train.shape}")
print(f"X_test shape :{X_test.shape}")

# Delete the train and test dataframes to free up memory
del df_train, df_test, original

X_train.head(5)


# # <h1 style = "font-family: Georgia;font-weight: bold; font-size: 30px; color: #1192AA; text-align:left">Models</h1>

# Thanks to https://www.kaggle.com/tetsutani

# In[13]:


class Splitter:
    def __init__(self, kfold=True, n_splits=5):
        self.n_splits = n_splits
        self.kfold = kfold

    def split_data(self, X, y, random_state_list):
        if self.kfold:
            for random_state in random_state_list:
                kf = KFold(n_splits=self.n_splits, random_state=random_state, shuffle=True)
                for train_index, val_index in kf.split(X, y):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                    yield X_train, X_val, y_train, y_val
        else:
            raise ValueError(f"Invalid kfold: Must be True")


# In[14]:


X = X_train
y = y_train

def objective(trial):
    params = {
        'n_estimators': 250,
        'num_leaves': trial.suggest_int('num_leaves', 8, 128),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-9, 10.0),
        'objective': 'regression_l1',
        'metric': 'mean_absolute_error',
        'boosting_type': 'gbdt',
        'device': 'cpu',
        'random_state': 42
    }
    
    lgbm = lgb.LGBMRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_list = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        lgbm.fit(X_train, y_train)
        y_pred = lgbm.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mae_list.append(mae)
    return np.mean(mae_list)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1)


# In[15]:


X = X_train
y = y_train

def objective(trial):
    params = {
        'n_estimators': 250,
        'depth': trial.suggest_int('depth', 3, 12),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-9, 10.0),
        'random_strength': trial.suggest_uniform('random_strength', 0.01, 1.0),
        'max_bin': trial.suggest_int('max_bin', 50, 500),
        'od_wait': trial.suggest_int('od_wait', 10, 100),
        'grow_policy': 'Lossguide',
        'bootstrap_type': 'Bayesian',
        'od_type': 'Iter',
        'eval_metric': 'MAE',
        'loss_function': 'MAE',
        'random_state': 42
    }
    cb = CatBoostRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_list = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        cb.fit(X_train, y_train, verbose=False)
        y_pred = cb.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mae_list.append(mae)
    return np.mean(mae_list)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1)


# In[16]:


X = X_train
y = y_train

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'bootstrap': True,
        'random_state': 42
    }
    rf = RandomForestRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_list = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mae_list.append(mae)
    return np.mean(mae_list)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1)


# In[17]:


class Regressor:
    def __init__(self, n_estimators=100, device="cpu", random_state=0):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.models = self._define_model()
        self.models_name = list(self._define_model().keys())
        self.len_models = len(self.models)
        
    def _define_model(self):
        
        xgb_params = {
            'n_estimators': self.n_estimators,
            'max_depth': 7,
            'learning_rate': 0.0116,
            'colsample_bytree': 1,
            'subsample': 0.6085,
            'min_child_weight': 9,
            'reg_lambda': 4.879e-07,
            'max_bin': 431,
            'n_jobs': -1,
            'eval_metric': 'mae',
            'objective': "reg:squarederror",
            'verbosity': 0,
            'random_state': self.random_state,
        }
        if self.device == 'gpu':
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['predictor'] = 'gpu_predictor'
        xgb_exact_params = xgb_params.copy()
        xgb_exact_params['tree_method'] = 'exact'
        xgb_approx_params = xgb_params.copy()
        xgb_approx_params['tree_method'] = 'approx'
        
        lgb_params = {
            'n_estimators': self.n_estimators,
            'max_depth': 7,
            "num_leaves": 16,
            'learning_rate': 0.05,
            'subsample': 0.60,
            'colsample_bytree': 1,
            'reg_alpha': 0.25,
            'reg_lambda': 5e-07,
            'objective': 'regression_l1',
            'metric': 'mean_absolute_error',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state
        }
        lgb2_params = {
            'n_estimators': self.n_estimators,
            'num_leaves': 93, 
            'min_child_samples': 20, 
            'learning_rate': 0.05533790147941807, 
            'colsample_bytree': 0.8809128870084636, 
            'reg_alpha': 0.0009765625, 
            'reg_lambda': 0.015589408048174165,
            'objective': 'regression_l1',
            'metric': 'mean_absolute_error',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state
        }
        lgb3_params = {
            'n_estimators': self.n_estimators,
            'num_leaves': 45,
            'max_depth': 13,
            'learning_rate': 0.0684383311038932,
            'subsample': 0.5758412171285148,
            'colsample_bytree': 0.8599714680300794,
            'reg_lambda': 1.597717830931487e-08,
            'objective': 'regression_l1',
            'metric': 'mean_absolute_error',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state,
            'force_col_wise': True
        }
        lgb_goss_params = lgb_params.copy()
        lgb_goss_params['boosting_type'] = 'goss'
        lgb_dart_params = lgb_params.copy()
        lgb_dart_params['boosting_type'] = 'dart'
        lgb_dart_params['n_estimators'] = 500
                
        cb_params = {
            'iterations': self.n_estimators,
            'depth': 8,
            'learning_rate': 0.01,
            'l2_leaf_reg': 0.7,
            'random_strength': 0.2,
            'max_bin': 200,
            'od_wait': 65,
            'one_hot_max_size': 70,
            'grow_policy': 'Depthwise',
            'bootstrap_type': 'Bayesian',
            'od_type': 'Iter',
            'eval_metric': 'MAE',
            'loss_function': 'MAE',
            'task_type': self.device.upper(),
            'random_state': self.random_state
        }
        cb2_params = {
            'iterations': self.n_estimators,
            'depth': 9, 
            'learning_rate': 0.456,
            'l2_leaf_reg': 8.41,
            'random_strength': 0.18,
            'max_bin': 225, 
            'od_wait': 58, 
            'grow_policy': 'Lossguide',
            'bootstrap_type': 'Bayesian',
            'od_type': 'Iter',
            'eval_metric': 'MAE',
            'loss_function': 'MAE',
            'task_type': self.device.upper(),
            'random_state': self.random_state
        }
        cb3_params = {
            'n_estimators': self.n_estimators,
            'depth': 11,
            'learning_rate': 0.08827842054729117,
            'l2_leaf_reg': 4.8351074756668864e-05,
            'random_strength': 0.21306687539993183,
            'max_bin': 483,
            'od_wait': 97,
            'grow_policy': 'Lossguide',
            'bootstrap_type': 'Bayesian',
            'od_type': 'Iter',
            'eval_metric': 'MAE',
            'loss_function': 'MAE',
            'task_type': self.device.upper(),
            'random_state': self.random_state,
            'silent': True
        }
        cb_sym_params = cb_params.copy()
        cb_sym_params['grow_policy'] = 'SymmetricTree'
        cb_loss_params = cb_params.copy()
        cb_loss_params['grow_policy'] = 'Lossguide'
        
        models = {
            #"xgb": xgb.XGBRegressor(**xgb_params),
            #"xgb_exact": xgb.XGBRegressor(**xgb_exact_params),
            #"xgb_approx": xgb.XGBRegressor(**xgb_approx_params),
            "lgb": lgb.LGBMRegressor(**lgb_params),
            "lgb2": lgb.LGBMRegressor(**lgb2_params),
            "lgb3": lgb.LGBMRegressor(**lgb3_params),
            "cat": CatBoostRegressor(**cb_params),
            "cat2": CatBoostRegressor(**cb2_params),
            "cat3": CatBoostRegressor(**cb3_params),
            #"cat_sym": CatBoostRegressor(**cb_sym_params),
            "cat_loss": CatBoostRegressor(**cb_loss_params),
            #"Ridge": RidgeCV(),
            #"Lasso": LassoCV(),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=self.random_state, n_jobs=-1),
            #"PLSRegression": PLSRegression(n_components=10, max_iter=2000),
            "PassiveAggressiveRegressor": PassiveAggressiveRegressor(max_iter=3000, tol=1e-3, n_iter_no_change=30, random_state=self.random_state),
            #"GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=2000, learning_rate=0.05, loss="absolute_error", random_state=self.random_state),
            "HistGradientBoostingRegressor": HistGradientBoostingRegressor(max_iter=self.n_estimators, learning_rate=0.01, loss="absolute_error", n_iter_no_change=300,random_state=self.random_state),
            #"ARDRegression": ARDRegression(n_iter=1000),
            "HuberRegressor": HuberRegressor(max_iter=3000),
            "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
        }
        
        return models


# In[18]:


class OptunaWeights:
    def __init__(self, random_state: int = 42, n_trials: int = 2000):
        self.study = None
        self.weights = None
        self.random_state = random_state
        self.n_trials = n_trials

    def _objective(self, trial: optuna.trial.Trial, y_true: np.ndarray, y_preds: List[np.ndarray]) -> float:
        # Define the weights for the predictions from each model
        weights = np.array([trial.suggest_float(f"weight{n}", 1e-14, 1) for n in range(len(y_preds))])

        # Calculate the weighted prediction
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=weights)

        # Calculate the score for the weighted prediction
        score = mean_absolute_error(y_true, weighted_pred)
        return score

    def fit(self, y_true: np.ndarray, y_preds: List[np.ndarray]) -> None:
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        pruner = optuna.pruners.HyperbandPruner()
        self.study = optuna.create_study(sampler=sampler, pruner=pruner, study_name="OptunaWeights", direction='minimize')
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        self.study.optimize(objective_partial, n_trials=self.n_trials)
        self.weights = np.array([self.study.best_params[f"weight{n}"] for n in range(len(y_preds))])

    def predict(self, y_preds: List[np.ndarray]) -> np.ndarray:
        assert self.weights is not None, 'OptunaWeights error, must be fitted before predict'
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=self.weights)
        return weighted_pred

    def fit_predict(self, y_true: np.ndarray, y_preds: List[np.ndarray]) -> np.ndarray:
        self.fit(y_true, y_preds)
        return self.predict(y_preds)


# In[19]:


kfold = True
n_splits = 1 if not kfold else 5
random_state = 42
random_state_list = [42]
n_estimators = 9999
early_stopping_rounds = 333
verbose = False
device = 'cpu'
unique_targets = np.unique(y_train)

splitter = Splitter(kfold=kfold, n_splits=n_splits)

# Initialize an array for storing test predictions
regressor = Regressor(n_estimators, device, random_state)
test_predss = np.zeros((X_test.shape[0]))
oof_predss = np.zeros((X_train.shape[0]))
ensemble_score = []
weights = []
trained_models = {'lgb_test':[], 'cat_test':[], "rf_test":[]}
score_dict = dict(zip(regressor.models_name, [[] for _ in range(regressor.len_models)]))

    
for i, (X_train_, X_val, y_train_, y_val) in enumerate(splitter.split_data(X_train, y_train, random_state_list=random_state_list)):
    n = i % n_splits
    m = i // n_splits
            
    # Get a set of Regressor models
    regressor = Regressor(n_estimators, device, random_state)
    models = regressor.models
    
    # Initialize lists to store oof and test predictions for each base model
    oof_preds = []
    test_preds = []
    
    # Loop over each base model and fit it to the training data, evaluate on validation data, and store predictions
    for name, model in models.items():
        if ('xgb' in name) or ('lgb' in name) or ('cat' in name):
            model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds, verbose=verbose)
        else:
            model.fit(X_train_, y_train_)
            
        if name in trained_models.keys():
            trained_models[f'{name}'].append(deepcopy(model))
        
        test_pred = model.predict(X_test).reshape(-1)
        y_val_pred = model.predict(X_val).reshape(-1)
        
        y_val_pred = [min(unique_targets, key = lambda x: abs(x - pred)) for pred in y_val_pred]
        test_pred = [min(unique_targets, key = lambda x: abs(x - pred)) for pred in test_pred]
        
        score = mean_absolute_error(y_val, y_val_pred)
        score_dict[name].append(score)
        print(f'{name} [FOLD-{n} SEED-{random_state_list[m]}] MAE score: {score:.5f}')
        
        oof_preds.append(y_val_pred)
        test_preds.append(test_pred)
    
    # Use Optuna to find the best ensemble weights
    optweights = OptunaWeights(random_state=random_state)
    y_val_pred = optweights.fit_predict(y_val.values, oof_preds)
    
    score = mean_absolute_error(y_val, y_val_pred)
    print(f'Ensemble [FOLD-{n} SEED-{random_state_list[m]}] MAE score {score:.5f}')
    ensemble_score.append(score)
    weights.append(optweights.weights)
    
    # Predict to X_test by the best ensemble weights
    test_predss += optweights.predict(test_preds) / (n_splits * len(random_state_list))
    oof_predss[X_val.index] = optweights.predict(oof_preds)
    
    gc.collect()


# In[20]:


# Calculate the mean LogLoss score of the ensemble
mean_score = np.mean(ensemble_score)
std_score = np.std(ensemble_score)
print(f'Ensemble MAE score {mean_score:.5f} ± {std_score:.5f}')

print('')
# Print the mean and standard deviation of the ensemble weights for each model
print('--- Model Weights ---')
mean_weights = np.mean(weights, axis=0)
std_weights = np.std(weights, axis=0)
for name, mean_weight, std_weight in zip(models.keys(), mean_weights, std_weights):
    print(f'{name}: {mean_weight:.5f} ± {std_weight:.5f}')


# # <h1 style = "font-family: Georgia;font-weight: bold; font-size: 30px; color: #1192AA; text-align:left">Make Submission</h1>

# In[21]:


sub = pd.read_csv(PATH_SUB)
sub[f'{target_col}'] = test_predss

sub.to_csv('submission.csv', index=False)
sub

