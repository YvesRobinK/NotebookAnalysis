#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
from functools import partial
import json
from time import time

import pandas as pd
import numpy as np

from scipy import stats
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_squared_log_error
rmse = partial(mean_squared_error, squared=False)
rmsle = partial(mean_squared_log_error, squared=False)

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

from category_encoders import TargetEncoder, LeaveOneOutEncoder, WOEEncoder

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, log_evaluation, early_stopping
from catboost import CatBoostRegressor

import torch
DEVICE = 'gpu' if torch.cuda.is_available() else 'cpu'
DEVICE_XGB = 'gpu_hist' if torch.cuda.is_available() else 'auto'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


# <div class='alert alert-block alert-info'>
#     
# This notebook contains a framework for regression analysis that I built over the number of playground episodes.

# ## The Code

# In[2]:


METRICS = {
    'rmse': rmse,
    'rmsle': rmsle
}

class RegressionPlayer():
    """The main class to simplify EDA and modelling.
    """
    def __init__(self, dataset_name, original_filename, target, metric='rmse'):
        self.df_train = pd.read_csv(f'/kaggle/input/{dataset_name}/train.csv', index_col=0)
        self.df_test = pd.read_csv(f'/kaggle/input/{dataset_name}/test.csv', index_col=0)
        self.df_original = pd.read_csv(original_filename)

        self.target = target
        self.y_train = self.df_train[target]
        self.y_original = self.df_original[target]
        self.cv = KFold(n_splits=5, shuffle=True, random_state=0)
        
        self.metric_fn = METRICS[metric]

        self.leaderboard = {}
        self.models = {}
        self.oof_preds = {}
        self.test_preds = {}
        
        self._view_data()

    def perform_eda(self, num_features):
        """Perform basic EDA.
        """
        self.num_features = num_features
        self._plot_target()
        self._plot_feature_distribution()
        self._plot_correlation()
        
    def train_model(self, model_fn=None, num_features=None, feature_fn=None, use_original_data=False, model_name=None, 
                    early_stopping_rounds=500, return_models=False, verbose=False):
        """Train `model_fn` with self.cv, optinally with `feature_fn` to create addtional features and `use_original_data`.
        Can save test predictions for submission.
        """
        self.num_features = num_features
        df_train = self.df_train.copy()
        df_original = self.df_original.copy()
        df_test = self.df_test.copy()
        if feature_fn is not None:
            feature_fn(df_train)
            feature_fn(df_test)
            if use_original_data:
                feature_fn(df_original)
        
        oof_preds = np.zeros(len(df_train))
        pipelines = []

        for fold, (idx_tr, idx_vl) in enumerate(self.cv.split(df_train, self.y_train)):
            # Fold train: add the entire original data
            df_tr, y_tr = df_train.iloc[idx_tr], self.y_train[idx_tr]
            if use_original_data:
                df_tr = pd.concat([df_tr, df_original])
                y_tr = np.hstack([y_tr, self.y_original])

            # Fold validation: just synthetic data
            df_vl, y_vl = df_train.iloc[idx_vl], self.y_train[idx_vl]

            # eval_set for early stopping
            pipeline = self._build_pipeline(model_fn)
            pipeline['proc'].fit(df_tr, y_tr)
            X_vl = pipeline['proc'].transform(df_vl)
            eval_set = [(X_vl, y_vl)]

            if type(pipeline['model']) == CatBoostRegressor:
                pipeline.fit(df_tr, y_tr, model__eval_set=eval_set, model__early_stopping_rounds=early_stopping_rounds, model__verbose=verbose)
            elif type(pipeline['model']) == XGBRegressor:
                pipeline['model'].early_stopping_rounds = early_stopping_rounds
                pipeline.fit(df_tr, y_tr, model__eval_set=eval_set, model__verbose=verbose)
            elif type(pipeline['model']) == LGBMRegressor:
                callbacks = [early_stopping(early_stopping_rounds), log_evaluation(-1)]
                pipeline.fit(df_tr, y_tr, model__eval_set=eval_set, model__callbacks=callbacks)
            else:
                pipeline.fit(df_tr, y_tr)

            oof_preds[idx_vl] = pipeline.predict(df_vl).squeeze()
            score = self.metric_fn(y_vl, oof_preds[idx_vl])
            pipelines.append(pipeline)

            if verbose:
                print(f'Fold {fold} score = {score:.4f}')

        score = self.metric_fn(self.y_train, oof_preds)
        print(f'OOF score = {score:.4f}')
        
        if model_name is not None:
            df = pd.DataFrame(data={'id': df_train.index, self.target: oof_preds})
            df.to_csv(f'{model_name}_oof_preds.csv', index=None)
            y_pred = [p.predict(df_test) for p in pipelines]
            y_pred = pd.DataFrame(y_pred).T.mean(axis=1)
            df = pd.DataFrame(data={'id': df_test.index, self.target: y_pred})
            df.to_csv(f'{model_name}_test_preds.csv', index=None)
            self.leaderboard[model_name] = score        
            self.models[model_name] = pipelines
            self.oof_preds[model_name] = oof_preds
            self.test_preds[model_name] = y_pred

        if return_models:
            return pipelines
        
    def show_leaderboard(self):
        return pd.DataFrame(self.leaderboard.values(), index=self.leaderboard.keys(), columns=['CV score']).sort_values('CV score')
    
    def build_mean_ensemble(self, model_names, ensemble_name):
        """Create an ensemble of provided model names by taking average of predictions. 
        Save oof_preds and test_preds.
        """
        preds = np.mean([self.oof_preds[m] for m in model_names], axis=0)
        df = pd.DataFrame(data={'id': self.df_train.index, self.target: preds})
        df.to_csv(f'{ensemble_name}_oof_preds.csv', index=None)
        score = self.metric_fn(self.y_train, preds)
        print(f'Ensemble score={score:.4f}')
        
        preds = np.mean([self.test_preds[m] for m in model_names], axis=0)
        df = pd.DataFrame(data={'id': self.df_test.index, self.target: preds})
        df.to_csv(f'{ensemble_name}_test_preds.csv', index=None)
        
    def _view_data(self):
        """Glance at the data.
        """
        df = pd.DataFrame([len(self.df_train), len(self.df_test), len(self.df_original)], index=['train', 'test', 'original'], columns=['count'])
        display(df)
        
        display(pd.DataFrame([
            self.df_train.dtypes,
            self.df_train.nunique(),
            self.df_test.nunique(),
            self.df_original.nunique(),
            self.df_train.isnull().sum(),
            self.df_test.isnull().sum(),
            self.df_original.isnull().sum()
        ], index=['datatype', 'unique values train', 'unique values test', 'unique values original', 'missing train', 'missing test', 'missing original']).T)

        display(self.df_train.head())
        
    def _plot_target(self):
        """Plot distribution of the target feature synthetic train vs. original dataset.
        """
        plt.figure(figsize=(8,6))
        sns.kdeplot(data=self.df_train, x=self.target, label='synthetic')
        sns.kdeplot(data=self.df_original, x=self.target, label='original')
        plt.title('Target distribution: synthetic vs. original')
        plt.legend()
        plt.show()

    def _plot_feature_distribution(self):
        """Plot feature distribution grouped by the 3 sets.
        """
        features = self.df_test.columns
        df_train = self.df_train.copy()
        df_test = self.df_test.copy()
        df_original = self.df_original.copy()
        
        df_train['set'] = 'train'
        df_original['set'] = 'original'
        df_test['set'] = 'test'
        df_combined = pd.concat([df_train, df_test, df_original])
        ncols = 2
        nrows = np.ceil(len(features)/ncols).astype(int)
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(15,nrows*4))
        for c, ax in zip(features, axs.flatten()):
            if c in self.num_features:
                sns.boxplot(data=df_combined, x=c, ax=ax, y='set')
            else:
                sns.countplot(data=df_combined, x='set', ax=ax, hue=c)

        fig.suptitle('Distribution of features by set')
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()
        
    def _plot_correlation(self):
        """Plot correlation between numerical features and the target feature.
        """
        plt.figure(figsize=(8,8))
        features = self.num_features + [self.target]
        corr = self.df_train[features].corr()
        annot_labels = np.where(corr.abs() > 0.5, corr.round(1).astype(str), '')
        upper_triangle = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=upper_triangle,
            vmin=-1, vmax=1, center=0, square=True, annot=annot_labels,
            cmap='coolwarm', linewidths=.5, fmt=''
        )
        plt.title('Correlation between numerical features and the target feature')
        plt.show()
        
    def _build_pipeline(self, model_fn):
        num_proc = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
        processing = ColumnTransformer([
            ('num', num_proc, self.num_features)
        ])

        return Pipeline([ 
            ('proc', processing),
            ('model', model_fn())
        ])


# ## Episode 9: Concrete Strength

# ### Initialize the player

# In[3]:


player = RegressionPlayer(
    dataset_name='playground-series-s3e9', 
    original_filename='/kaggle/input/predict-concrete-strength/ConcreteStrengthData.csv',
    target='Strength')


# <div class='alert alert-block alert-success'><b>Notes:</b>
#     
# - All features are numerical.
# - The dataset sizes are tiny.

# ### Basic EDA
# <div class='alert alert-block alert-info'>
# I will check
# 
# - missing values
# - distribution of the target feature
# - distribution of features grouped by different datasets
# - correlation of numerical features and the target feature

# In[4]:


# Post processing due to an extra space after CementComponent in the original dataset
player.df_original.columns = [c.strip() for c in player.df_original.columns]

# Use all features as numerical features
num_features = player.df_test.columns.tolist()

player.perform_eda(num_features)


# <div class='alert alert-block alert-success'><b>Notes:</b>
#     
# - Target has the same distribution between train and original.
# - Other features have similar distribution.

# ### Baseline models
# 
# <div class='alert alert-block alert-info'>
# For a regression task, I will train one model with default ridge regression and 3 GBDT models.

# In[5]:


models = [
    ('ridge', partial(Ridge, random_state=0)),
    ('lgbm', partial(LGBMRegressor, random_state=0)),
    ('xgb', partial(XGBRegressor, random_state=0)),
    ('cb', partial(CatBoostRegressor, random_state=0))
]

for model_name, model_fn in models:
    print(model_name)
    player.train_model(model_fn=model_fn, num_features=num_features, model_name=model_name)
    print()
    
player.show_leaderboard()


# ### Which features are the most important?
# using CatBoost model

# In[6]:


df = pd.DataFrame({'feature': num_features})
df['importance'] = np.array([p['model'].feature_importances_ for p in player.models['cb']]).mean(axis=0)
plt.figure(figsize=(8,8))
sns.barplot(data=df.sort_values('importance'), x='importance', y='feature')


# <div class='alert alert-block alert-success'><b>Notes:</b>
#     
# `AgeInDays` is dominant.

# ### Feature engineering

# In[7]:


def add_features(df):
    # Calculate TotalComponentWeight
    df['TotalComponentWeight'] = df['CementComponent'] + df['BlastFurnaceSlag'] + df['FlyAshComponent'] + df['WaterComponent'] + df['SuperplasticizerComponent'] + df['CoarseAggregateComponent'] + df['FineAggregateComponent']

    # Calculate Water-Cement-Ratio (WCR)
    df['WCR'] = df['WaterComponent'] / df['CementComponent']

    # Calculate Aggregate-Ratio (AR)
    df['AR'] = (df['CoarseAggregateComponent'] + df['FineAggregateComponent']) / df['CementComponent']

    # Calculate Water-Cement-Plus-Pozzolan-Ratio (WCPR)
    df['WCPR'] = df['WaterComponent'] / (df['CementComponent'] + df['BlastFurnaceSlag'] + df['FlyAshComponent'])

    # Calculate Cement-Age
    df['Cement-Age'] = df['CementComponent'] * df['AgeInDays']


# In[8]:


prefix = 'extra_features_'
extra_features = num_features + ['TotalComponentWeight', 'WCR', 'AR', 'WCPR', 'Cement-Age']
for model_name, model_fn in models:
    print(model_name)
    player.train_model(model_fn=model_fn, num_features=extra_features, feature_fn=add_features, model_name=prefix+model_name)
    print()
    
player.show_leaderboard()


# In[9]:


df = pd.DataFrame({'feature': extra_features})
df['importance'] = np.array([p['model'].feature_importances_ for p in player.models['extra_features_cb']]).mean(axis=0)
plt.figure(figsize=(8,8))
sns.barplot(data=df.sort_values('importance'), x='importance', y='feature')


# <div class='alert alert-block alert-success'><b>Notes:</b>
#     
# New features seem to be very good:
# - `Cement-Age` has the highest importance
# - `WCR` and `WPR` are also good

# ### Adding original data

# In[10]:


prefix = 'extra_data_'
for model_name, model_fn in models:
    print(model_name)
    player.train_model(model_fn=model_fn, num_features=num_features, use_original_data=True, model_name=prefix+model_name)
    print()
    
player.show_leaderboard()


# In[11]:


prefix = 'extra_features_extra_data_'
extra_features = num_features + ['TotalComponentWeight', 'WCR', 'AR', 'WCPR', 'Cement-Age']
for model_name, model_fn in models:
    print(model_name)
    player.train_model(model_fn=model_fn, num_features=extra_features, feature_fn=add_features, use_original_data=True, model_name=prefix+model_name)
    print()
    
player.show_leaderboard()


# ### Hypeparameters tuning with FLAML

# In[12]:


X_train = player.df_train.drop(columns=[player.target]).values
flaml_tuned = True

TIME_BUDGET = 60 * 60
EARLY_STOPPING_ROUNDS = 500


# In[13]:


if not flaml_tuned:
    for model_name in ['lgbm', 'xgboost', 'catboost']:
        auto_flaml = flaml.AutoML()
        auto_flaml.fit(X_train, player.y_train, task='regression', estimator_list=[model_name], time_budget=TIME_BUDGET, early_stop=EARLY_STOPPING_ROUNDS, verbose=0)
        print(model_name)
        print(auto_flaml.best_config)
        
lgbm_params = {'n_estimators': 129, 'num_leaves': 15, 'min_child_samples': 42, 'learning_rate': 0.07779512138199543, 'log_max_bin': 10, 'colsample_bytree': 0.8675396181891428, 'reg_alpha': 0.0009765625, 'reg_lambda': 454.03639725321733}
xgb_params = {'n_estimators': 35, 'max_leaves': 12, 'min_child_weight': 128.0, 'learning_rate': 0.1463919918170542, 'subsample': 0.7325065386963903, 'colsample_bylevel': 0.6667932104813801, 'colsample_bytree': 0.9460503049970427, 'reg_alpha': 0.001760399960739169, 'reg_lambda': 0.028974203631898784}
cb_params = {'early_stopping_rounds': 37, 'learning_rate': 0.0355150326550479, 'n_estimators': 113}


# In[14]:


models = [
    ('lgbm', partial(LGBMRegressor, random_state=0, **lgbm_params)),
    ('xgb', partial(XGBRegressor, random_state=0, **xgb_params)),
    ('cb', partial(CatBoostRegressor, random_state=0, **cb_params))
]

prefix = 'extra_features_tuned_'
for model_name, model_fn in models:
    print(model_name)
    player.train_model(model_fn=model_fn, num_features=extra_features, feature_fn=add_features, model_name=prefix+model_name)
    print()
    
player.show_leaderboard()


# ### Ensembling

# In[15]:


selected_models = ['extra_features_tuned_lgbm', 'extra_features_tuned_xgb', 'extra_features_tuned_cb']
selected_pred_files = [m + '_oof_preds.csv' for m in selected_models]
dfs = [pd.read_csv(f, index_col=0) for f in selected_pred_files]


# In[16]:


mean_pred = pd.concat(dfs, axis=1).mean(axis=1)
rmse(player.y_train, mean_pred)


# <div class='alert alert-block alert-warning'><b>Notes:</b>
#     
# - Simple average of 3 models gives worse result!

# ---

# ## Episode 8

# <div class='alert alert-block alert-success'><b>Notes:</b>
#     
# - All features are numerical except `cut`, `color`, `clarity`.
# - However, those 3 features are ordinal and should be encoded in such way.
# - The dataset sizes are pretty good.

# <div class='alert alert-block alert-success'><b>Insights:</b>
#     
# - There are 697 data points without `depth` in the original dataset.

# <div class='alert alert-block alert-success'><b>Insights:</b>
#    
# - The three datasets are pretty similar
# - Some numerical features have a few 'outliers' 

# In[17]:


# def encode(feature_name, feature_values):
#     encoder = OrdinalEncoder(categories=feature_values)
#     df_train[feature_name] = encoder.fit_transform(df_train[[feature_name]])
#     df_test[feature_name] = encoder.transform(df_test[[feature_name]])
#     df_original[feature_name] = encoder.transform(df_original[[feature_name]])


# In[18]:


# cut_values = [['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']]
# clarity_values = [['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']]
# color_values = [['D', 'E', 'F', 'G', 'H', 'I', 'J']]
# encode('cut', cut_values)
# encode('clarity', clarity_values)
# encode('color', color_values)


# In[19]:


# NUM_FEATURES = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']


# <div class='alert alert-block alert-success'><b>Insights:</b>
#    
# - `price` has strong correlation with carat, x, y, z, which indicate the weight and volume of the diamond.

# In[20]:


# def replace_0(df, val=1):
#     df['x'] = df['x'].replace(0, val)
#     df['y'] = df['z'].replace(0, val)
#     df['z'] = df['z'].replace(0, val)
    
# replace_0(df_train)
# replace_0(df_test)
# replace_0(df_original)


# In[21]:


# run(df_train, y_train, model_fn=partial(Ridge, random_state=0), save_file='01_default_ridge')


# In[22]:


# lgbm_pipelines = run(df_train, y_train, model_fn=partial(LGBMRegressor, random_state=0), save_file='02_default_lgbm', return_models=True)


# In[23]:


# df = pd.DataFrame({'feature': NUM_FEATURES})
# df['importance'] = np.array([p['model'].feature_importances_ for p in lgbm_pipelines]).mean(axis=0)
# plt.figure(figsize=(8,8))
# sns.barplot(data=df.sort_values('importance'), x='importance', y='feature')


# In[24]:


# def add_features(df):
#     df['volume'] = df['x'] * df['y'] * df['z']
#     df['density'] = df['carat'] / df['volume']
#     df['table_ratio'] = (df['table'] / ((df['x'] + df['y']) / 2))
#     df['depth_ratio'] = (df['depth'] / ((df['x'] + df['y']) / 2))
#     df['symmetry'] = (abs(df['x'] - df['z']) + abs(df['y'] - df['z'])) / (df['x'] + df['y'] + df['z'])
#     df['surface_area'] = 2 * ((df['x'] * df['y']) + (df['x'] * df['z']) + (df['y'] * df['z']))
#     df['depth_to_table_ratio'] = df['depth'] / df['table']
#     return df

# df_fe = add_features(df_train.copy())
# new_num_features = NUM_FEATURES + df_fe.columns.difference(df_train.columns).tolist()
# lgbm_pipelines = run(df_fe, y_train, model_fn=partial(LGBMRegressor, random_state=0), num_features=new_num_features, return_models=True)
# df = pd.DataFrame({'feature': new_num_features})
# df['importance'] = np.array([p['model'].feature_importances_ for p in lgbm_pipelines]).mean(axis=0)
# plt.figure(figsize=(8,8))
# sns.barplot(data=df.sort_values('importance'), x='importance', y='feature')


# <div class='alert alert-block alert-success'><b>Insights:</b>
#    
# - Slighly worse!

# In[25]:


# run(df_train, y_train, model_fn=partial(XGBRegressor, random_state=0), save_file='03_default_xgb')


# In[26]:


# run(df_train, y_train, model_fn=partial(CatBoostRegressor, random_state=0), save_file='04_default_cb')


# In[27]:


# TIME_BUDGET = 60 * 60 * 6
# EARLY_STOPPING_ROUNDS = 500


# In[28]:


# auto_flaml = flaml.AutoML()
# auto_flaml.fit(df_train, y_train, task='regression', estimator_list=['catboost'], metric='rmse', time_budget=TIME_BUDGET, early_stop=EARLY_STOPPING_ROUNDS)
# print(auto_flaml.best_config)
# with open(f'tuned_{TIME_BUDGET}_cb.json', 'w') as f:
#     f.write(json.dumps(auto_flaml.best_config))


# In[29]:


# lgbm_params = {'n_estimators': 11504, 'num_leaves': 4, 'min_child_samples': 13, 'learning_rate': 0.09166794698941537, 'colsample_bytree': 0.97938601127897, 'reg_alpha': 0.0014560424484998061, 'reg_lambda': 0.030870581342053376}
# run(df_train, y_train, model_fn=partial(LGBMRegressor, **lgbm_params, random_state=0), early_stopping_rounds=EARLY_STOPPING_ROUNDS, save_file='05_tuned_lgbm')


# In[30]:


# xgb_params = {'n_estimators': 520, 'max_leaves': 5018, 'min_child_weight': 76.34621712044539, 'learning_rate': 0.03274206517107865, 'subsample': 0.8175573064803033, 'colsample_bylevel': 1.0, 'colsample_bytree': 1.0, 'reg_alpha': 0.8859233894025167, 'reg_lambda': 896.6263563095711}
# run(df_train, y_train, model_fn=partial(XGBRegressor, **xgb_params, random_state=0), early_stopping_rounds=EARLY_STOPPING_ROUNDS, save_file='06_tuned_xgb')


# In[31]:


# cb_params = {'early_stopping_rounds': 10, 'learning_rate': 0.08908399489851125, 'n_estimators': 757}
# run(df_train, y_train, model_fn=partial(CatBoostRegressor, **cb_params, random_state=0), early_stopping_rounds=EARLY_STOPPING_ROUNDS, save_file='07_tuned_cb')


# <div class='alert alert-block alert-warning'><b>Warning:</b>
#    
# - FLAML tuning gives poorer performance than the the default one!

# In[32]:


# from torch import nn, optim
# import torch.nn.functional as F
# from torch.utils.data import Dataset, TensorDataset, DataLoader


# In[33]:


# num_proc = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
# X_train = num_proc.fit_transform(df_train[NUM_FEATURES])
# X_train.shape


# In[34]:


# X_tr, X_vl, y_tr, y_vl = train_test_split(X_train, y_train, test_size=0.1, random_state=0)
# # Convert to tensor
# X_tr_tensor = torch.from_numpy(X_tr.astype(np.float32))
# X_vl_tensor = torch.from_numpy(X_vl.astype(np.float32))

# y_tr_tensor = torch.from_numpy(y_tr.values.astype(np.float32)).view(-1,1)
# y_vl_tensor = torch.from_numpy(y_vl.values.astype(np.float32)).view(-1,1)


# In[35]:


# batch_size = 128
# ds_train = TensorDataset(X_tr_tensor, y_tr_tensor)
# dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
# ds_valid = TensorDataset(X_vl_tensor, y_vl_tensor)
# dl_valid = DataLoader(ds_valid, batch_size=batch_size)
# X, y = iter(dl_train).next()
# X.shape, y.shape


# In[36]:


# class NN(nn.Module):
#     def __init__(self, input_size):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_size, 200),
#             nn.ReLU(),
#             nn.Linear(200, 100),
#             nn.ReLU(),
#             nn.Linear(100, 1)
#         )
        
#     def forward(self, x):
#         x = self.fc(x)
#         return x
    
# # Hyperparameters
# input_size = X_train.shape[1]
# learning_rate = 1e-3
# n_epochs = 100

# model = NN(input_size)
# model.to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20, verbose=True)

# # Training loop
# best_loss = 1000

# for e in range(n_epochs):
#     t0 = time()
#     train_loss = 0
#     total_loss = 0
#     count_y = 0
    
#     model.train()
#     for X, y in dl_train:
#         X = X.to(device)
#         y = y.to(device)
        
#         y_pred = model(X)
#         loss = criterion(y_pred, y)
#         loss = torch.sqrt(loss)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
        
#         total_loss = train_loss * count_y + loss.item() * len(y)
#         count_y += len(y)
#         train_loss = total_loss / count_y
    
#     val_loss = 0
#     total_loss = 0
#     count_y = 0
#     model.eval()
    
#     for X, y in dl_valid:
#         X = X.to(device)
#         y = y.to(device)
#         with torch.no_grad():
#             y_pred = model(X)
#             loss = criterion(y_pred, y)
#             loss = torch.sqrt(loss)
#             total_loss = val_loss * count_y + loss.item() * len(y)
#             count_y += len(y)
#             val_loss = total_loss / count_y
            
#     scheduler.step(val_loss)
    
#     saved_indicator = ''
#     if val_loss < best_loss:
#         torch.save(model, 'best_model.pt')
#         best_loss = val_loss
#         saved_indicator = '(*)'
        
#     t1 = time()
#     if (e+1) % 1 == 0:
#         print(f'epoch: {e+1}, train_loss={train_loss:.1f}, val_loss={val_loss:.1f}, time={(t1-t0):.1f} {saved_indicator}')


# In[37]:


# y_original = df_original[TARGET]
# df_train['original'] = 0
# df_original['original'] = 1
# df_test['original'] = 0
# org_num_features = NUM_FEATURES + ['original']

# run(df_train, y_train, df_org=df_original, y_org=y_original, model_fn=partial(LGBMRegressor, random_state=0), num_features=org_num_features, save_file='04_default_lgbm_extra_data')


# In[38]:


# num_proc = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
# df_tr, df_vl = train_test_split(df_train, test_size=0.1, random_state=0)
# X_tr = num_proc.fit_transform(df_tr[NUM_FEATURES])
# X_vl = num_proc.transform(df_vl[NUM_FEATURES])
# y_tr = df_tr[TARGET]
# y_vl = df_vl[TARGET]


# In[39]:


# %%time
# mlp_model = MLPClassifier(random_state=0)
# mlp_model.fit(X_tr, y_tr)
# y_pred = mlp_model.predict(X_vl)
# rmse(y_vl, y_pred)


# <div class='alert alert-block alert-warning'><b>Warning:</b>
#    
# - Way too slow!

# In[40]:


# Uncomment to tune
# df_combined = pd.concat([df_train, df_original])
# predictor = TabularPredictor(label=TARGET, problem_type='regression', eval_metric='root_mean_squared_error', path=f'AutoGluon_{TIME_BUDGET}')
# predictor.fit(df_combined, time_limit=TIME_BUDGET, presets='best_quality', verbosity=0)
# y_pred = predictor.get_oof_pred(train_data=df_train).values
# df = pd.DataFrame(data={'id': df_train.index, TARGET: y_pred})
# df = df.iloc[:len(df_train)]
# df.to_csv(f'05_autogluon_oof_preds.csv', index=None)
# print(rmse(y_train, y_pred))

# display(pd.DataFrame(predictor.fit_summary(verbosity=0)))

# y_pred = predictor.predict(df_test).values
# df = pd.DataFrame(data={'id': df_test.index, TARGET: y_pred})
# df.to_csv(f'05_autogluon_test_preds.csv', index=None)


# <div class='alert alert-block alert-success'><b>Insights:</b>
#    
# - Tiny improvement

# In[41]:


# %%time
# reg = ak.StructuredDataRegressor(overwrite=True, max_trials=1, seed=0)
# history = reg.fit(X_tr, y_tr, validation_data=(X_vl,y_vl), epochs=3, verbose=0)
# y_pred = reg.predict(X_vl)
# rmse(y_vl, y_pred)


# In[42]:


# plt.figure(figsize=(8,5))
# xticks = np.arange(len(history.history['val_loss']))
# plt.plot(xticks, np.sqrt(history.history['val_loss']))
# plt.xlabel('epoch')
# plt.ylabel('root mean squared error')
# plt.xticks(xticks)
# plt.show()


# <div class='alert alert-block alert-warning'><b>Insights:</b>
#    
# - It's very slow and the results are very poor. Need to use GPU and make more trials.

# In[43]:


# X_test = num_proc.transform(df_test[NUM_FEATURES])
# y_pred = reg.predict(X_test)
# df = pd.DataFrame(data={'id': df_test.index, TARGET: y_pred.flatten()})
# df.to_csv(f'06_autokeras_test_preds.csv', index=None)


# In[44]:


# models = ['02_default_lgbm', '03_default_xgb', '04_default_cb']
# models = [m + '_test_preds.csv' for m in models]
# dfs = [pd.read_csv(m, index_col=0) for m in models]
# df = pd.concat(dfs, axis=1)
# mean_target = df.mean(axis=1)
# df = df.drop(columns=[TARGET])
# df[TARGET] = mean_target
# df.to_csv('10_default_ensemble.csv')


# In[45]:


# models = ['05_tuned_lgbm', '06_tuned_xgb', '07_tuned_cb']
# models = [m + '_test_preds.csv' for m in models]
# dfs = [pd.read_csv(m, index_col=0) for m in models]
# df = pd.concat(dfs, axis=1)
# mean_target = df.mean(axis=1)
# df = df.drop(columns=[TARGET])
# df[TARGET] = mean_target
# df.to_csv('11_tuned_ensemble.csv')


# In[46]:


# pd.read_csv('11_tuned_ensemble.csv', index_col='id').to_csv('submission.csv')


# ---

# ## Episode 6 - Paris Housing Price

# #### Data at a glance
# An regression task predicting house price. Original dataset is [Paris Housing Price Prediction](https://www.kaggle.com/datasets/mssmartypants/paris-housing-price-prediction). All attributes are numeric variables and they are listed bellow:
# 
# - squareMeters
# - numberOfRooms
# - hasYard
# - hasPool
# - floors - number of floors
# - cityCode - zip code
# - cityPartRange - the higher the range, the more exclusive the neighbourhood is
# - numPrevOwners - number of prevoious owners
# - made - year
# - isNewBuilt
# - hasStormProtector
# - basement - basement square meters
# - attic - attic square meteres
# - garage - garage size
# - hasStorageRoom
# - hasGuestRoom - number of guest rooms
# - price - predicted value

# In[47]:


# df_train = pd.read_csv('/kaggle/input/playground-series-s3e6/train.csv', index_col='id')
# df_test = pd.read_csv('/kaggle/input/playground-series-s3e6/test.csv', index_col='id')
# df_original = pd.read_csv('/kaggle/input/paris-housing-price-prediction/ParisHousing.csv')

# print(df_train.info())
# display(df_train.head())

# pd.DataFrame(
#     [len(df_train), len(df_test), len(df_original)],
#     index=['train', 'test', 'original'],
#     columns=['count']
# )


# <div class='alert alert-block alert-success'><b>Notes:</b>
#     
# - All features are numerical.
# - Binary features: hasYard, hasPool, isNewBuilt, hasStormProtector, hasStorageRoom. Note: **hasGuestRoom** is not binary.
# - The datasets are not too small.

# In[48]:


# TARGET = 'price'
# pd.concat([
#     pd.DataFrame(df_train.drop(columns=[TARGET]).isnull().sum(), columns=['missing train']),
#     pd.DataFrame(df_test.isnull().sum(), columns=['missing test']),
#     pd.DataFrame(df_original.drop(columns=[TARGET]).isnull().sum(), columns=['missing original'])
# ], axis=1)


# <div class='alert alert-block alert-success'><b>Insights:</b>
#     
# - There are no missing values in any sets.

# #### Outlier detection

# In[49]:


# def check_outliers(df, z=3):
#     df_z_score = stats.zscore(df)
#     columns = df.columns.tolist()
#     if TARGET in columns:
#         columns.remove(TARGET)
#     for c in columns:
#         df_outlier = df_z_score.query(f'{c} > 3 | {c} < -3')
#         if len(df_outlier):
#             display(df.loc[df_outlier.index][[c]])


# In[50]:


# check_outliers(df_train)
# check_outliers(df_test)
# check_outliers(df_original)


# <div class='alert alert-block alert-success'><b>Insights:</b>
#    
# Some notable outliers in the train set:
# - `squareMeters`: id=15334, value=6,071,330
# - `floor`: id=5659, value=6,000
# - `made`: 5 houses with value of 10,000
# - `cityCode`, `basement`, `attic`, `garage` also have outliers
# 
# In the test set, there are only outliers in `cityCode`, `basement` and `attic`. We won't don't drop those features in our train set.
# 
# Interesting, there are no outliers at all in the original dataset.

# In[51]:


# df_z_score = stats.zscore(df_train)
# columns_to_remove = ['squareMeters', 'floors', 'made', 'garage']
# df_outlier = df_z_score[((df_z_score[columns_to_remove] > 3) | (df_z_score[columns_to_remove] < -3)).max(axis=1)]
# df_train = df_train.loc[df_train.index.difference(df_outlier.index)]
# df_train.set_index(df_train.reset_index().index, inplace=True)


# #### Target distribution

# In[52]:


# plt.figure(figsize=(8,6))
# sns.kdeplot(data=df_train, x=TARGET, label='train')
# sns.kdeplot(data=df_original, x=TARGET, label='original')
# plt.legend()


# #### Distribution of features by set

# In[53]:


# df_train['set'] = 'train'
# df_original['set'] = 'original'
# df_test['set'] = 'test'
# df_combined = pd.concat([df_train, df_test, df_original])
# NUM_FEATURES = [c for c in df_train.columns if c not in [TARGET, 'set']]
# ncols = 2
# nrows = np.ceil(len(NUM_FEATURES)/ncols).astype(int)
# fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(15,nrows*4))
# for c, ax in zip(NUM_FEATURES, axs.flatten()):
#     if df_combined[c].nunique() > 20:
#         sns.boxplot(data=df_combined, x=c, ax=ax, y='set')
#     else:
#         sns.countplot(data=df_combined, x='set', ax=ax, hue=c)
#         ax.legend(bbox_to_anchor=(1,1), loc='upper left', title=c)
#         ax.set_xlabel('')

# fig.suptitle('Distribution of features by set')
# plt.tight_layout(rect=[0, 0, 1, 0.98])
# for df in [df_train, df_test, df_original]:
#     df.drop(columns=['set'], errors='ignore')


# <div class='alert alert-block alert-success'><b>Insights:</b>
#    
# The original dataset is quite different from the synthetic dataset.
# - For float features, the synthetic dataset adds outliers
# - Count features in the original dataset are distributed equally, but they are varied much more in both train and test set. The good news is that the distributions in the train and test set look similar.

# #### Correlation in the train set

# In[54]:


# def show_correlation(df, features):
#     plt.figure(figsize=(8,8))
#     corr = df[features].corr()
#     annot_labels = np.where(corr.abs() > 0.4, corr.round(1).astype(str), '')
#     upper_triangle = np.triu(np.ones_like(corr, dtype=bool))
#     sns.heatmap(
#         corr, mask=upper_triangle,
#         vmin=-1, vmax=1, center=0, square=True, annot=annot_labels,
#         cmap='coolwarm', linewidths=.5, fmt=''
#     )
    
# show_correlation(df_train, NUM_FEATURES + [TARGET])


# #### Correlation in the original set

# In[55]:


# show_correlation(df_original, NUM_FEATURES + [TARGET])


# <div class='alert alert-block alert-success'><b>Insights:</b>
#    
# - So interesting, there is almost perfect correlation between `price` and `squareMeters` in both sets.

# In[56]:


# plt.figure(figsize=(8,8))
# sns.scatterplot(data=df_train, x='squareMeters', y='price')


# #### Preparation for modelling

# In[57]:


# y_train = df_train[TARGET]
# CV = KFold(n_splits=5, shuffle=True, random_state=0)


# #### First baseline

# In[58]:


# run(df_train, y_train, model_fn=partial(Ridge, random_state=0), save_file='01_default_ridge')


# ##### A bit tunning
# also address this topic https://www.kaggle.com/competitions/playground-series-s3e6/discussion/384596

# In[59]:


# pipeline = build_pipeline(model_fn=partial(Ridge, random_state=0), num_features=NUM_FEATURES)
# parameters = {'model__alpha': np.logspace(-4, 0, 100)}
# search = GridSearchCV(pipeline, parameters, cv=CV, scoring='neg_mean_squared_error')
# search.fit(df_train, y_train)
# plt.figure(figsize=(8,6))
# plt.plot(search.cv_results_['param_model__alpha'].data, np.sqrt(-search.cv_results_['mean_test_score']))
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('alpha (log scale)')
# plt.ylabel('RMSE (log scale)')
# plt.tight_layout()
# plt.show()
# # plt.savefig('ridge_alpha.jpg', dpi=300)


# In[60]:


# optimal_alpha = search.best_params_['model__alpha']
# run(df_train, y_train, model_fn=partial(Ridge, random_state=0, alpha=optimal_alpha), save_file='02_tuned_ridge')


# <div class='alert alert-block alert-success'><b>Insights:</b>
#    
# - Tuning has almost no effect here because `squareMeters` has too high weight.

# #### GBDT

# In[61]:


# run(df_train, y_train, model_fn=partial(LGBMRegressor, random_state=0), save_file='03_default_lgbm')


# In[62]:


# run(df_train, y_train, model_fn=partial(XGBRegressor, random_state=0), save_file='04_default_xgb')


# In[63]:


# run(df_train, y_train, model_fn=partial(CatBoostRegressor, random_state=0), save_file='05_default_cb')


# #### Which features does the model think as important?
# Look at XGB model.

# In[64]:


# xgb_pipelines = run(df_train, y_train, model_fn=partial(XGBRegressor, random_state=0), return_models=True)
# df = pd.DataFrame({'feature': NUM_FEATURES})
# df['importance'] = np.array([p['model'].feature_importances_ for p in xgb_pipelines]).mean(axis=0)
# plt.figure(figsize=(8,8))
# sns.barplot(data=df.sort_values('importance'), x='importance', y='feature')


# <div class='alert alert-block alert-success'><b>Insights:</b>
#    
# - This is pretty much expected with the high correlation discovered earlier.

# #### Feature engineering
# See my ideas in this post https://www.kaggle.com/competitions/playground-series-s3e6/discussion/384679.

# In[65]:


# def add_features(df):
#     df['allSpace'] = df['squareMeters'] + df['basement'] + df['attic']
#     df['roomsPerFloor'] = df['numberOfRooms'] / df['floors']
#     df['age'] = 2022 - df['made']
#     df['prevOwnersPerYear'] = df['numPrevOwners'] / df['age']
#     city_code_count = df['cityCode'].value_counts()    
#     df['numHousesInCityCode'] = df['cityCode'].map(city_code_count)
#     city_code_space = df.groupby('cityCode')['squareMeters'].sum()
#     df['sumAllSpaceInCityCode'] = df['cityCode'].map(city_code_space)
#     return df

# df_fe = add_features(df_train.copy())
# new_num_features = NUM_FEATURES + df_fe.columns.difference(df_train.columns).tolist()
# xgb_pipelines = run(df_fe, y_train, model_fn=partial(XGBRegressor, random_state=0), num_features=new_num_features, return_models=True)
# df = pd.DataFrame({'feature': new_num_features})
# df['importance'] = np.array([p['model'].feature_importances_ for p in xgb_pipelines]).mean(axis=0)
# plt.figure(figsize=(8,8))
# sns.barplot(data=df.sort_values('importance'), x='importance', y='feature')


# <div class='alert alert-block alert-warning'><b>Insights:</b>
#    
# - Doesn't quite work yet.
