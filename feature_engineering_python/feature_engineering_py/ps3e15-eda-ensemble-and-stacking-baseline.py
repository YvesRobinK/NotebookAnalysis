#!/usr/bin/env python
# coding: utf-8

# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Libraries</p>

# In[1]:


import numpy as np
import pandas as pd
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
from category_encoders import OrdinalEncoder, CountEncoder, CatBoostEncoder
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, LabelEncoder
from sklearn.compose import ColumnTransformer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.decomposition import PCA

# Import libraries for Hypertuning
import optuna

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
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.cross_decomposition import PLSRegression
from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier
from catboost import Pool

get_ipython().system('git clone https://github.com/analokmaus/kuma_utils.git')
import sys
sys.path.append("kuma_utils/")
from kuma_utils.preprocessing.imputer import LGBMImputer

get_ipython().system('pip install sklego')
from sklego.linear_model import LADRegression # Least Absolute Deviation Regression

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from colorama import Style, Fore
blk = Style.BRIGHT + Fore.BLACK
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
res = Style.RESET_ALL


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Data</p>

# In[2]:


filepath = '/kaggle/input/playground-series-s3e15'

data = pd.read_csv(os.path.join(filepath, 'data.csv'), index_col=[0])
original = pd.read_csv('/kaggle/input/predicting-heat-flux/Data_CHF_Zhao_2020_ATE.csv', index_col=[0])

data['is_generated'] = 1
original['is_generated'] = 0

original = original.reset_index()
original['id'] = original['id'] + data.index[-1]
original.set_index('id', inplace=True)

print(f"data shape :{data.shape}")
print(f"original shape :{original.shape}")

target_col = 'x_e_out [-]'
cat_cols = ['author', 'geometry']
num_cols = ['pressure [MPa]', 'mass_flux [kg/m2-s]', 'x_e_out [-]', 'D_e [mm]', 'D_h [mm]', 'length [mm]', 'chf_exp [MW/m2]']

df_train = data[~data[target_col].isnull()].copy()
df_test = data[data[target_col].isnull()].copy()


# In[3]:


def set_frame_style(df, caption=""):
    """Helper function to set dataframe presentation style.
    """
    return df.style.background_gradient(cmap='Blues').set_caption(caption).set_table_styles([{
    'selector': 'caption',
    'props': [
        ('color', 'Blue'),
        ('font-size', '18px'),
        ('font-weight','bold')
    ]}])

cols = data.columns.to_list()

display(set_frame_style(data[cols].head(),'First 5 Rows Of Data'))

display(set_frame_style(data[cols].describe(),'Summary Statistics'))

display(set_frame_style(data[cols].nunique().to_frame().rename({0:'Unique Value Count'}, axis=1).transpose(), 'Unique Value Counts In Each Column'))

display(set_frame_style(data[cols].isna().sum().to_frame().transpose(), 'Columns With Nan'))


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">EDA</p>
# 1. Train and Test histograms
# 2. Correlation of Features
# 3. Hierarchical Clustering
# 4. Pie chart and Bar plot
# 5. Scatter Plot with x_e_out [-] Column by author and geometry
# 6. Distribution Plot by author and geometry
# 7. Boxplot by author and geometry
# 8. Violinplot by author and geometry

# In[4]:


def plot_histograms(df_train, df_test, original, target_col, n_cols=3):
    n_rows = (len(df_train.columns) - 1) // n_cols + 1

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 4*n_rows))
    axes = axes.flatten()

    for i, var_name in enumerate(df_train.columns.tolist()):
        if var_name != 'is_generated':
            ax = axes[i]
            sns.distplot(df_train[var_name], kde=True, ax=ax, label='Train')
            if var_name != target_col:
                sns.distplot(df_test[var_name], kde=True, ax=ax, label='Test')
            sns.distplot(original[var_name], kde=True, ax=ax, label='Original')
            ax.set_title(f'{var_name} Distribution (Train vs Test)')
            ax.legend()

    plt.tight_layout()
    plt.show()
        
plot_histograms(df_train[num_cols], df_test[num_cols], original[num_cols], target_col, n_cols=2)


# In[5]:


def plot_heatmap(df, title):
    # Create a mask for the diagonal elements
    mask = np.zeros_like(df.astype(float).corr())
    mask[np.triu_indices_from(mask)] = True

    # Set the colormap and figure size
    colormap = plt.cm.RdBu_r
    plt.figure(figsize=(16, 16))

    # Set the title and font properties
    plt.title(f'{title} Correlation of Features', fontweight='bold', y=1.02, size=20)

    # Plot the heatmap with the masked diagonal elements
    sns.heatmap(df.astype(float).corr(), linewidths=0.1, vmax=1.0, vmin=-1.0, 
                square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 14, "weight": "bold"},
                mask=mask)

plot_heatmap(data[num_cols], title='Data')
plot_heatmap(original[num_cols], title='original')


# In[6]:


from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

def hierarchical_clustering(data):
    fig, ax = plt.subplots(1, 1, figsize=(14, 6), dpi=120)
    correlations = data.corr()
    converted_corr = 1 - np.abs(correlations)
    Z = linkage(squareform(converted_corr), 'complete')
    
    dn = dendrogram(Z, labels=data.columns, ax=ax, above_threshold_color='#ff0000', orientation='right')
    hierarchy.set_link_color_palette(None)
    plt.grid(axis='x')
    plt.title('Hierarchical clustering, Dendrogram', fontsize=18, fontweight='bold')
    plt.show()

hierarchical_clustering(data[num_cols])


# In[7]:


def plot_target_feature(df_train, target_col, figsize=(16,5), palette='colorblind', name='Train'):
    
    df_train = df_train.fillna('Nan')
    
    fig, ax = plt.subplots(1, 2, figsize = figsize)
    ax = ax.flatten()

    # Pie chart
    ax[0].pie(
        df_train[target_col].value_counts(), 
        shadow=True, 
        explode=[0.05] * len(df_train[target_col].unique()),
        autopct='%1.f%%',
        textprops={'size': 10, 'color': 'white'},
        colors=sns.color_palette(palette, len(df_train[target_col].unique()))
    )

    # Bar plot
    sns.countplot(
        data=df_train, 
        y=target_col, 
        ax=ax[1], 
        palette=palette
    )
    ax[1].yaxis.label.set_size(18)
    plt.yticks(fontsize=12)
    ax[1].set_xlabel('Count', fontsize=20)
    plt.xticks(fontsize=12)

    fig.suptitle(f'{target_col} in {name} Dataset', fontsize=20, fontweight='bold')
    plt.tight_layout()

    # Show the plot
    plt.show()
    
plot_target_feature(data, 'author', figsize=(16,5), palette='colorblind', name='Data')
plot_target_feature(original, 'author', figsize=(16,5), palette='colorblind', name='Original')
plot_target_feature(data, 'geometry', figsize=(16,5), palette='colorblind', name='Data')
plot_target_feature(original, 'geometry', figsize=(16,5), palette='colorblind', name='Original')


# In[8]:


def plot_scatter_with_fixed_col(df, fixed_col, hue=False, drop_cols=[], size=10, title=''):
    sns.set_style('whitegrid')
    
    if hue:
        cols = df.columns.drop([hue, fixed_col] + drop_cols)
    else:
        cols = df.columns.drop([fixed_col] + drop_cols)
    n_cols = 2
    n_rows = (len(cols) - 1) // n_cols + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(size, size/n_cols*n_rows), sharex=False, sharey=False)
    fig.suptitle(f'{title} Set Scatter Plot with Target Column by {hue}', fontsize=24, fontweight='bold', y=1.01)

    for i, col in enumerate(cols):
        n_row = i // n_cols
        n_col = i % n_cols
        ax = axes[n_row, n_col]

        ax.set_xlabel(f'{col}', fontsize=14)
        ax.set_ylabel(f'{fixed_col}', fontsize=14)

        # Plot the scatterplot
        if hue:
            sns.scatterplot(data=df, x=col, y=fixed_col, hue=hue, ax=ax,
                            s=80, edgecolor='gray', alpha=0.35, palette='bright')
            ax.legend(title=hue, title_fontsize=12, fontsize=12) # loc='upper right'
        else:
            sns.scatterplot(data=df, x=col, y=fixed_col, ax=ax,
                            s=80, edgecolor='gray', alpha=0.35)

        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_title(f'{col}', fontsize=18)
    
    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.show()
    
plot_scatter_with_fixed_col(data, fixed_col=target_col, hue='author', drop_cols=['is_generated', 'geometry'], size=24, title='Data')
plot_scatter_with_fixed_col(data, fixed_col=target_col, hue='geometry', drop_cols=['is_generated', 'author'], size=24, title='Data')
# plot_scatter_with_fixed_col(original, fixed_col=target_col, hue='author', drop_cols=['is_generated', 'geometry'], size=24, title='Original')


# In[9]:


def plot_distribution(df, hue, title='', drop_cols=[]):
    sns.set_style('whitegrid')

    cols = df.columns.drop([hue] + drop_cols)
    n_cols = 2
    n_rows = (len(cols) - 1) // n_cols + 1

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(14, 4*n_rows))

    for i, var_name in enumerate(cols):
        row = i // n_cols
        col = i % n_cols

        ax = axes[row, col]
        sns.histplot(data=df, x=var_name, kde=True, ax=ax, hue=hue) # sns.distplot(df_train[var_name], kde=True, ax=ax, label='Train')
        ax.set_title(f'{var_name} Distribution')

    fig.suptitle(f'{title} Distribution Plot by {hue}', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.show()

plot_distribution(data, hue='geometry', title='Data Set', drop_cols=['is_generated', 'author'])
plot_distribution(data, hue='author', title='Data Set', drop_cols=['is_generated', 'geometry'])


# In[10]:


def plot_boxplot(df, hue, title='', drop_cols=[], n_cols=3):
    sns.set_style('whitegrid')

    cols = df.columns.drop([hue] + drop_cols)
    n_rows = (len(cols) - 1) // n_cols + 1

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 5*n_rows))

    for i, var_name in enumerate(cols):
        row = i // n_cols
        col = i % n_cols

        ax = axes[row, col]
        sns.boxplot(data=df, x=hue, y=var_name, ax=ax, showmeans=True, 
                    meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue", "markersize":"5"})
        ax.set_title(f'{var_name} by {hue}')
        ax.set_xlabel('')

    fig.suptitle(f'{title} Boxplot by {hue}', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.show()

plot_boxplot(data, hue='author', title='Data Set', drop_cols=['is_generated', 'geometry'], n_cols=2)
plot_boxplot(data, hue='geometry', title='Data Set', drop_cols=['is_generated', 'author'], n_cols=3)

plot_boxplot(original, hue='author', title='Original Set', drop_cols=['is_generated', 'geometry'], n_cols=2)
plot_boxplot(original, hue='geometry', title='Original Set', drop_cols=['is_generated', 'author'], n_cols=3)


# In[11]:


def plot_violinplot(df, hue, title='', drop_cols=[], n_cols=2):
    sns.set_style('whitegrid')

    cols = df.columns.drop([hue] + drop_cols)
    n_rows = (len(cols) - 1) // n_cols + 1

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 4*n_rows))

    for i, var_name in enumerate(cols):
        row = i // n_cols
        col = i % n_cols

        ax = axes[row, col]
        sns.violinplot(data=df, x=hue, y=var_name, ax=ax, inner='quartile')
        ax.set_title(f'{var_name} Distribution')

    fig.suptitle(f'{title} Violin Plot by {hue}', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.show()

plot_violinplot(data, hue='geometry', title='Data Set', drop_cols=['is_generated', 'author'], n_cols=3)
plot_violinplot(original, hue='geometry', title='Original Set', drop_cols=['is_generated', 'author'], n_cols=3)


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Feature Engineering</p>
# - `cat_encoder()` : Perform label encoding and one-hot encoding
# - `create_features()` : Create new features. Pre-processing is based on this discussion.
#   - https://www.kaggle.com/competitions/playground-series-s3e15/discussion/411353
# - `add_pca_features()` : Add PCA results as features. Select features for dimensionality reduction as needed.
# - `replace_null()` : Returns the label encoder result to Null. Here we use the LightGBM imputer.
#   - https://github.com/analokmaus/kuma_utils

# In[12]:


def cat_encoder(X_train, X_test, cat_cols, encode='label'):
    
    if encode == 'label':
        ## Label Encoder
        encoder = OrdinalEncoder(cols=cat_cols, handle_missing='ignore')
        train_encoder = encoder.fit_transform(X_train[cat_cols]).astype(int)
        test_encoder = encoder.transform(X_test[cat_cols]).astype(int)
        X_train[cat_cols] = train_encoder[cat_cols]
        X_test[cat_cols] = test_encoder[cat_cols]
        encoder_cols = cat_cols
    
    else:
        ## OneHot Encoder
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        train_encoder = encoder.fit_transform(X_train[cat_cols]).astype(int)
        test_encoder = encoder.transform(X_test[cat_cols]).astype(int)
        X_train = pd.concat([X_train, train_encoder], axis=1)
        X_test = pd.concat([X_test, test_encoder], axis=1)
        X_train.drop(cat_cols, axis=1, inplace=True)
        X_test.drop(cat_cols, axis=1, inplace=True)
        encoder_cols = list(train_encoder.columns)
        
    return X_train, X_test, encoder_cols

def create_features(df):
    """
    pressure: Pressure
    mass_flux: Mass flux
    D_e: Inner diameter
    D_h: Horizontal diameter
    length: Length
    chf_exp: Critical heat flux in experiments
    """ 
    # Adiabatic surface area
    df['adiabatic_surface_area'] = df['D_e'] * df['length']
    
    # Surface area to horizontal diameter ratio
    df['surface_diameter_ratio'] = df['D_e'] / df['D_h']
    
    # Velocity
    # df['velocity'] = df['mass_flux'] / (3.14 * (df['D_e']**2) / 4)

    # Heat flux density
    # df['heat_flux_density'] = df['chf_exp'] / (df['adiabatic_surface_area'] + np.finfo(float).eps)
    
    # df['chf_exp'] = np.log(df['chf_exp'])
    # df['D_e'] = np.log(df['D_e'])
    # df['D_h'] = np.log(df['D_h'])
    # df['length'] = np.log(df['length'])
    
    return df

def add_pca_features(X_train, X_test):    
    
    # Select the columns for PCA
    # pca_features = X_train.select_dtypes(include=['float64']).columns.tolist()
    pca_features = ["pressure", "mass_flux", "chf_exp"]

    # Create the pipeline
    pipeline = make_pipeline(StandardScaler(), PCA(n_components=len(pca_features)))
    
    # Perform PCA
    pipeline.fit(X_train[pca_features])

    # Create column names for PCA features
    pca_columns = [f'PCA_{i}' for i in range(len(pca_features))]

    # Add PCA features to the dataframe
    X_train[pca_columns] = pipeline.transform(X_train[pca_features])
    X_test[pca_columns] = pipeline.transform(X_test[pca_features])

    return X_train, X_test

def replace_null(df):
    df["author"] = df["author"].replace({3:np.nan})
    df["geometry"] = df["geometry"].replace({2:np.nan})
    
    return df

def rename(df):
    df.columns = df.columns.str.replace('\[.*?\]', '', regex=True)
    df.columns = df.columns.str.strip()
    
    # df.loc[df['D_e'].isna(), 'D_e'] = df.loc[df['D_e'].isna(), 'D_h']
    # df.loc[df['D_h'].isna(), 'D_h'] = df.loc[df['D_h'].isna(), 'D_e']
    
    return df


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Prepare train and test sets</p>

# In[13]:


# Concatenate train and original dataframes, and prepare train and test sets
df_train = pd.concat([df_train, original])
X_train = df_train.drop([f'{target_col}'],axis=1).reset_index(drop=True)
y_train = df_train[f'{target_col}'].reset_index(drop=True)
X_test = df_test.drop([f'{target_col}'],axis=1).reset_index(drop=True)

num_cols = X_train.select_dtypes(include=['float64']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# Drop_col
drop_cols = []
X_train.drop(drop_cols, axis=1, inplace=True)
X_test.drop(drop_cols, axis=1, inplace=True)

# category_encoders
X_train, X_test, cat_cols = cat_encoder(X_train, X_test, cat_cols, encode='label')
X_train, X_test = replace_null(X_train), replace_null(X_test)

# Rename
X_train = rename(X_train)
X_test = rename(X_test)

# Imputer
imputer = LGBMImputer(n_iter=1000, cat_features=cat_cols)
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Create Features
X_train = create_features(X_train)
X_test = create_features(X_test)
# X_train, X_test = add_pca_features(X_train, X_test)

for cat_col in cat_cols:
    X_train[cat_col] = X_train[cat_col].astype("int").astype("category")
    X_test[cat_col] = X_test[cat_col].astype("int").astype("category")

print(f"X_train shape :{X_train.shape} , y_train shape :{y_train.shape}")
print(f"X_test shape :{X_test.shape}")

del data, df_train, df_test, original

X_train.head(5)


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Data Splitting</p>

# In[14]:


class Splitter:
    def __init__(self, kfold=True, n_splits=5, cat_df=pd.DataFrame()):
        self.n_splits = n_splits
        self.kfold = kfold
        self.cat_df = cat_df

    def split_data(self, X, y, random_state_list):
        if self.kfold == 'skf':
            for random_state in random_state_list:
                kf = StratifiedKFold(n_splits=self.n_splits, random_state=random_state, shuffle=True)
                for train_index, val_index in kf.split(X, self.cat_df):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                    yield X_train, X_val, y_train, y_val
        elif self.kfold:
            for random_state in random_state_list:
                kf = KFold(n_splits=self.n_splits, random_state=random_state, shuffle=True)
                for train_index, val_index in kf.split(X, y):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                    yield X_train, X_val, y_train, y_val
        else:
            raise ValueError(f"Invalid kfold: Must be True")


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Define Model</p>
# LightGBM, CatBoost Xgboost and HistGradientBoosting hyper parameters are determined by optuna.

# In[15]:


class Regressor:
    def __init__(self, n_estimators=1000, device="cpu", random_state=0):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.models = self._define_model()
        self.models_name = list(self._define_model().keys())
        self.len_models = len(self.models)
        
    def _define_model(self):
        
        xgb_params = {
            'n_estimators': self.n_estimators,
            'max_depth': 6,
            'learning_rate': 0.0116,
            'colsample_bytree': 1,
            'subsample': 0.6085,
            'min_child_weight': 9,
            'reg_lambda': 4.879e-07,
            'max_bin': 431,
            'n_jobs': -1,
            'eval_metric': 'rmse',
            'objective': "reg:squarederror",
            'verbosity': 0,
            'random_state': self.random_state,
        }
        if self.device == 'gpu':
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['predictor'] = 'gpu_predictor'
        
        lgb1_params = {
            'n_estimators': self.n_estimators,
            'learning_rate': 0.00187132612825764,
            "reg_alpha": 0.218098671693009,
            "reg_lambda": 0.843310784903128,
            "num_leaves": 35,
            "colsample_bytree": 0.46532069763536,
            'subsample': 0.900877679402001,
            'subsample_freq': 1,
            'min_child_samples': 61,
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state
        }
        lgb2_params = {
            'n_estimators': self.n_estimators,
            'learning_rate': 0.0272191862297552,
            "reg_alpha": 0.000369546074207732,
            "reg_lambda": 0.000372594824801102,
            "num_leaves": 129,
            "colsample_bytree": 0.465991955308929,
            'subsample': 0.992343325538791,
            'subsample_freq': 4,
            'min_child_samples': 53,
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state
        }
        lgb3_params = {
            'n_estimators': self.n_estimators,
            'max_depth': 8,
            "num_leaves": 16,
            'learning_rate': 0.05,
            'subsample': 0.7,
            'colsample_bytree': 0.8,
            'reg_lambda': 5e-07,
            'objective': 'regression_l2',
            'metric': 'mean_squared_error',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state
        }
        lgb4_params = {
            'n_estimators': self.n_estimators,
            'max_depth': 10,
            "num_leaves": 16,
            'learning_rate': 0.02,
            'subsample': 0.3,
            'colsample_bytree': 0.5,
            'reg_alpha': 0.280490490266035,
            'reg_lambda':0.8609215326333549,
            'objective': 'regression_l2',
            'metric': 'mean_squared_error',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state
        }
        cat1_params = {
            'iterations': self.n_estimators,
            'colsample_bylevel': 0.0981364003850992,
            'depth': 7,
            'learning_rate': 0.0405542421804142,
            'l2_leaf_reg': 3.96440812572686,
            'random_strength': 0.244309262101739,
            'od_type': 'IncToDec',
            'od_wait': 54,
            'bootstrap_type': 'Bayesian',
            'grow_policy': 'Depthwise',
            'bagging_temperature': 5.56557834528889,
            'eval_metric': 'RMSE',
            'loss_function': 'RMSE',
            'task_type': self.device.upper(),
            'random_state': self.random_state
        }
        cat2_params = {
            'iterations': self.n_estimators,
            'colsample_bylevel': 0.0826155582319386,
            'depth': 7,
            'learning_rate': 0.0550490364072692,
            'l2_leaf_reg': 0.140958369944226,
            'random_strength': 0.198890547890241,
            'od_type': 'Iter',
            'od_wait': 55,
            'bootstrap_type': 'Bayesian',
            'grow_policy': 'Lossguide',
            'bagging_temperature': 3.83350857230709,
            'eval_metric': 'RMSE',
            'loss_function': 'RMSE',
            'task_type': self.device.upper(),
            'random_state': self.random_state
        }
        cat3_params = {
            'iterations': self.n_estimators,
            'depth': 6,
            'learning_rate': 0.02,
            'l2_leaf_reg': 0.5,
            'random_strength': 0.2,
            'max_bin': 150,
            'od_wait': 80,
            'one_hot_max_size': 70,
            'grow_policy': 'SymmetricTree',
            'bootstrap_type': 'Bayesian',
            'od_type': 'IncToDec',
            'eval_metric': 'RMSE',
            'loss_function': 'RMSE',
            'task_type': self.device.upper(),
            'random_state': self.random_state
        }
        hist_params = {
            'l2_regularization': 0.171778774949396,
            'early_stopping': True,
            'learning_rate': 0.005,
            'max_iter': self.n_estimators,
            'n_iter_no_change': 100,
            'max_depth': 29,
            'max_bins': 255,
            'min_samples_leaf': 33,
            'max_leaf_nodes':41,
            'random_state': self.random_state,
            'categorical_features': ['author', 'geometry']
        }
        
        one_hot_encoder = make_column_transformer(
            (
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                make_column_selector(dtype_include="category"),
            ),
            remainder="passthrough",
        )
        scaler = make_column_transformer(
            (
                StandardScaler(),
                make_column_selector(dtype_include="float"),
            ),
            remainder="passthrough",
        ).set_output(transform="pandas")
        
        
        models = { 
            "xgb": xgb.XGBRegressor(**xgb_params),
            "lgb1": lgb.LGBMRegressor(**lgb1_params),
            "lgb2": lgb.LGBMRegressor(**lgb2_params),
            "lgb3": lgb.LGBMRegressor(**lgb3_params),
            #"lgb4": lgb.LGBMRegressor(**lgb4_params),
            "cat1": CatBoostRegressor(**cat1_params),
            "cat2": CatBoostRegressor(**cat2_params),
            "cat3": CatBoostRegressor(**cat3_params),
            'hgb': HistGradientBoostingRegressor(**hist_params),
#             "SVR_rbf": make_pipeline(scaler, one_hot_encoder, SVR(kernel="rbf", gamma="auto")),
            #"Ridge": make_pipeline(scaler, one_hot_encoder, RidgeCV()),
            #"Lasso": make_pipeline(scaler, one_hot_encoder, LassoCV()),
#             "KNeighborsRegressor": make_pipeline(scaler, one_hot_encoder, KNeighborsRegressor(n_neighbors=5, n_jobs=-1)),            
#             "RandomForestRegressor": make_pipeline(one_hot_encoder, RandomForestRegressor(n_estimators=500, random_state=self.random_state, n_jobs=-1)),
            #"SGDRegressor": make_pipeline(one_hot_encoder, SGDRegressor(max_iter=2000, early_stopping=True, n_iter_no_change=100, random_state=self.random_state)),
#             "MLPRegressor": make_pipeline(scaler, one_hot_encoder, MLPRegressor(max_iter=500, early_stopping=True, n_iter_no_change=10, random_state=self.random_state)),
#             "ExtraTreesRegressor": make_pipeline(one_hot_encoder, ExtraTreesRegressor(n_estimators=500, n_jobs=-1, random_state=self.random_state)), 
        }
        
        return models


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Optimizer (--> Optimize RMSE)</p>

# In[16]:


class OptunaWeights:
    def __init__(self, random_state, n_trials=100):
        self.study = None
        self.weights = None
        self.random_state = random_state
        self.n_trials = n_trials

    def _objective(self, trial, y_true, y_preds):
        # Define the weights for the predictions from each model
        weights = [trial.suggest_float(f"weight{n}", 1e-15, 1) for n in range(len(y_preds))]

        # Calculate the weighted prediction
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=weights)

        # Calculate the score for the weighted prediction
        score = np.sqrt(mean_squared_error(y_true, weighted_pred))
        return score

    def fit(self, y_true, y_preds):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        pruner = optuna.pruners.HyperbandPruner()
        self.study = optuna.create_study(sampler=sampler, pruner=pruner, study_name="OptunaWeights", direction='minimize')
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        self.study.optimize(objective_partial, n_trials=self.n_trials)
        self.weights = [self.study.best_params[f"weight{n}"] for n in range(len(y_preds))]

    def predict(self, y_preds):
        assert self.weights is not None, 'OptunaWeights error, must be fitted before predict'
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=self.weights)
        return weighted_pred

    def fit_predict(self, y_true, y_preds):
        self.fit(y_true, y_preds)
        return self.predict(y_preds)
    
    def weights(self):
        return self.weights


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Weighted Ensemble by Optuna on Training</p>
# A weighted average is performed during training;  
# The weights were determined for each model using the predictions for the train data created in the out of fold with Optuna's CMAsampler. (Here it is defined by `OptunaWeights`)  
# This is an extension of the averaging method. All models are assigned different weights defining the importance of each model for prediction.
# 
# ![](https://www.analyticsvidhya.com/wp-content/uploads/2015/08/Screen-Shot-2015-08-22-at-6.40.37-pm.png)
# 

# In[17]:


get_ipython().run_cell_magic('time', '', '\nkfold = True\nn_splits = 1 if not kfold else 15\nn_reapts = 1\nrandom_state = 42\nn_estimators = 99999\nn_trials = 3000\nearly_stopping_rounds = 2000\nverbose = False\ndevice = \'cpu\'\n\n# Fix seed\nrandom.seed(random_state)\nrandom_state_list = random.sample(range(9999), n_reapts)\n\nsplitter = Splitter(kfold=kfold, n_splits=n_splits)\n\n# Initialize an array for storing test predictions\nregressor = Regressor(n_estimators, device, random_state)\ntest_predss = np.zeros((X_test.shape[0]))\noof_predss = np.zeros((X_train.shape[0]))\noof_each_predss = []\noof_each_preds = np.zeros((X_train.shape[0], regressor.len_models))\ntest_each_predss = []\ntest_each_preds = np.zeros((X_test.shape[0], regressor.len_models))\nensemble_score = []\nfold_scores = []\nweights = []\ntrained_models = dict(zip([_ for _ in regressor.models_name if (\'lgb\' in _) or (\'cat\' in _)], [[] for _ in range(regressor.len_models)]))\nscore_dict = dict(zip(regressor.models_name, [[] for _ in range(regressor.len_models)]))\n\n    \nfor i, (X_train_, X_val, y_train_, y_val) in enumerate(splitter.split_data(X_train, y_train, random_state_list=random_state_list)):\n    n = i % n_splits\n    m = i // n_splits\n            \n    # Get a set of Regressor models\n    regressor = Regressor(n_estimators, device, random_state)\n    models = regressor.models\n    \n    # Initialize lists to store oof and test predictions for each base model\n    oof_preds = []\n    test_preds = []\n    \n    # Loop over each base model and fit it to the training data, evaluate on validation data, and store predictions\n    for name, model in models.items():\n        X_test_, X_val_ = X_test.copy(), X_val.copy()\n        if (\'xgb\' in name) or (\'lgb\' in name) or (\'cat\' in name):\n            if \'lgb\' in name:\n                model.fit(\n                    X_train_, y_train_, eval_set=[(X_val, y_val)], categorical_feature=cat_cols,\n                    early_stopping_rounds=early_stopping_rounds, verbose=verbose)\n            elif \'cat\' in name :\n                model.fit(\n                    Pool(X_train_, y_train_, cat_features=cat_cols), eval_set=Pool(X_val, y_val, cat_features=cat_cols),\n                    early_stopping_rounds=early_stopping_rounds, verbose=verbose)\n            else:\n                one_hot_encoder = make_column_transformer((\n                            OneHotEncoder(sparse_output=False, handle_unknown="ignore"),\n                            make_column_selector(dtype_include="category"),),remainder="passthrough").set_output(transform="pandas")\n                X_train__ = one_hot_encoder.fit_transform(X_train_)\n                X_val_ = one_hot_encoder.transform(X_val)\n                X_test_ = one_hot_encoder.transform(X_test_)\n                model.fit(X_train__, y_train_, eval_set=[(X_val_, y_val)], early_stopping_rounds=early_stopping_rounds, verbose=verbose)\n        else:\n            model.fit(X_train_, y_train_)\n            \n        if name in trained_models.keys():\n            trained_models[f\'{name}\'].append(deepcopy(model))\n        \n        test_pred = model.predict(X_test_).reshape(-1)\n        y_val_pred = model.predict(X_val_).reshape(-1)\n        score = np.sqrt(mean_squared_error(y_val, y_val_pred))\n        score_dict[name].append(score)\n        print(f\'{blu}{name}{res} [FOLD-{n} SEED-{random_state_list[m]}] RMSE {blu}{score:.5f}\')\n        \n        oof_preds.append(y_val_pred)\n        test_preds.append(test_pred)\n    \n    # Use Optuna to find the best ensemble weights\n    optweights = OptunaWeights(random_state=random_state, n_trials=n_trials)\n    y_val_pred = optweights.fit_predict(y_val.values, oof_preds)\n    \n    score = np.sqrt(mean_squared_error(y_val, y_val_pred))\n    print(f\'{red}>>> Ensemble{res} [FOLD-{n} SEED-{random_state_list[m]}] RMSE {red}{score:.5f}\')\n    print(f\'{"-" * 50}\')\n    ensemble_score.append(score)\n    fold_scores.append(score)\n    weights.append(optweights.weights)\n    \n    # Predict to X_test by the best ensemble weights\n    test_predss += optweights.predict(test_preds) / (n_splits * len(random_state_list))\n    oof_predss[X_val.index] = optweights.predict(oof_preds)\n    oof_each_preds[X_val.index] = np.stack(oof_preds).T\n    test_each_preds += np.array(test_preds).T / n_splits\n    \n    if n == (n_splits - 1):\n        oof_each_predss.append(oof_each_preds)\n        oof_each_preds = np.zeros((X_train.shape[0], regressor.len_models))\n        test_each_predss.append(test_each_preds)\n        oof_each_preds = np.zeros((X_test.shape[0], regressor.len_models))\n        print(f\'{red}Mean Ensemble{res} [SEED-{random_state_list[m]}] RMSE {red}{np.mean(fold_scores):.5f}\')\n        print(f\'{"=" * 50}{res}\\n\')\n        fold_scores = []\n        \n    gc.collect()\n    \noof_each_predss = np.mean(np.array(oof_each_predss), axis=0)\ntest_each_predss = np.mean(np.array(test_each_predss), axis=0)\n')


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Stacking by LAD regression</p>
# 
# ![](https://www.researchgate.net/publication/324552457/figure/fig3/AS:616245728645121@1523935839872/An-example-scheme-of-stacking-ensemble-learning.png)
# 
# Level1 here uses LAD regression;  
# LAD (Least Absolute Deviations) regression is a statistical method that minimizes the sum of absolute differences between predicted and actual values. 
# It is robust to outliers and less sensitive to extreme values in the dataset compared to traditional regression methods like ordinary least squares (OLS) regression.  
# LAD regression is particularly useful when dealing with outliers or when the median of the dependent variable is of interest.  
# - kernel for reference https://www.kaggle.com/code/adaubas/ps-s3e14-stacking-leastabsolutedeviation-reg

# In[18]:


LADRegression_blend = LADRegression(positive=True)
LADRegression_blend.fit(oof_each_predss, y_train)
lad_score = np.sqrt(mean_squared_error(y_train, LADRegression_blend.predict(oof_each_predss)))


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Mean Scores for each model</p>

# In[19]:


def plot_score_from_dict(score_dict, title='RMSE', ascending=True):
    score_df = pd.melt(pd.DataFrame(score_dict))
    score_df = score_df.sort_values('value', ascending=ascending)
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x='value', y='variable', data=score_df, palette='Blues_r', errorbar='sd')
    plt.xlabel(f'{title}', fontsize=14)
    plt.ylabel('')
    #plt.title(f'{title}', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, axis='x')
    plt.show()

print('--- Mean RMSE Scores---')    
for name, score in score_dict.items():
    mean_score = np.mean(score)
    std_score = np.std(score)
    print(f'{name}: {red}{mean_score:.5f} ± {std_score:.5f}{res}')
plot_score_from_dict(score_dict, title=f'RMSE (n_splits:{n_splits})')


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Weight of the Optuna Ensemble and LAD regression</p>

# In[20]:


# Calculate the mean LogLoss score of the ensemble
mean_score = np.mean(ensemble_score)
std_score = np.std(ensemble_score)
print(f'{red}Mean{res} Optuna Ensemble RMSE {red}{mean_score:.5f} ± {std_score:.5f}{res}')
print(f'{red}Mean{res} LAD regression Ensemble RMSE {red}{lad_score:.5f}{res}')

print('')
# Print the mean and standard deviation of the ensemble weights for each model
print('--- Optuna Weights---')
mean_weights = np.mean(weights, axis=0)
std_weights = np.std(weights, axis=0)
for name, mean_weight, std_weight in zip(models.keys(), mean_weights, std_weights):
    print(f'{name}: {blu}{mean_weight:.5f} ± {std_weight:.5f}{res}')

print('')
print('--- LAD regression Weights---')
for i, name in enumerate(models.keys()):
    print(f'{name}: {blu}{LADRegression_blend.coef_[i]:.5f}{res}')
    
# weight_dict = dict(zip(list(score_dict.keys()), np.array(weights).T.tolist()))
# plot_score_from_dict(weight_dict, title='Model Weights', ascending=False)
normalize = [((weight - np.min(weight)) / (np.max(weight) - np.min(weight))).tolist() for weight in weights]
weight_dict = dict(zip(list(score_dict.keys()), np.array(normalize).T.tolist()))
plot_score_from_dict(weight_dict, title='Optuna Weights (Normalize 0 to 1)', ascending=False)

# normalize = [((weight - np.min(weight)) / (np.max(weight) - np.min(weight))).tolist() for weight in LADRegression_blend.coef_]
weight_dict = dict(zip(list(score_dict.keys()), [[_] for _ in LADRegression_blend.coef_]))
plot_score_from_dict(weight_dict, title='LAD regression Weights', ascending=False)


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Feature importance Visualization</p>

# In[21]:


def visualize_importance(models, feature_cols, title, top=9):
    importances = []
    feature_importance = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df["importance"] = model.feature_importances_
        _df["feature"] = pd.Series(feature_cols)
        _df["fold"] = i
        _df = _df.sort_values('importance', ascending=False)
        _df = _df.head(top)
        feature_importance = pd.concat([feature_importance, _df], axis=0, ignore_index=True)
        
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    # display(feature_importance.groupby(["feature"]).mean().reset_index().drop('fold', axis=1))
    plt.figure(figsize=(12, 4))
    sns.barplot(x='importance', y='feature', data=feature_importance, color='skyblue', errorbar='sd')
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.title(f'{title} Feature Importance', fontsize=18)
    plt.grid(True, axis='x')
    plt.show()
    
for name, models in trained_models.items():
    if name in list(trained_models.keys()):
        visualize_importance(models, list(X_train.columns), name)


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Make Submission</p>
# `mattop_post_process` takes a list of predictions and returns an array of target values. It calculates the absolute difference between each prediction and all target values, and then selects the target value with the smallest difference for each prediction.  
# Scores were up just a bit in this competition as well.
# - https://www.kaggle.com/competitions/playground-series-s3e14/discussion/407327

# In[22]:


plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 1)
sns.histplot(oof_predss, kde=True, alpha=0.5, label='oof_preds')
sns.histplot(y_train.values, kde=True, alpha=0.5, label='y_train')
plt.title('Histogram of OOF Predictions and Train Values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_train.values, y=oof_predss, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('OOF Predicted Values')
plt.title('Actual vs. OOF Predicted Values')

plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', alpha=0.5)
plt.show()


# In[23]:


unique_targets = np.unique(y_train)
def mattop_post_process(preds):
     return np.array([min(unique_targets, key = lambda x: abs(x - pred)) for pred in preds])

sub = pd.read_csv(os.path.join(filepath, 'sample_submission.csv'))
average_test_predss = np.average(np.vstack([test_predss, LADRegression_blend.predict(test_each_predss)]), axis=0, weights=[1, 1])
sub[f'{target_col}'] = mattop_post_process(average_test_predss)
sub.to_csv('submission.csv', index=False)

sub_ = pd.read_csv(os.path.join(filepath, 'sample_submission.csv'))
sub_[f'{target_col}'] = mattop_post_process(test_predss)
sub_.to_csv('submission_ensemble.csv', index=False)

sub_ = pd.read_csv(os.path.join(filepath, 'sample_submission.csv'))
sub_[f'{target_col}'] = mattop_post_process(LADRegression_blend.predict(test_each_predss))
sub_.to_csv('submission_lad.csv', index=False)

for i, name in enumerate(regressor.models_name):
    sub_ = pd.read_csv(os.path.join(filepath, 'sample_submission.csv'))
    sub_[f'{target_col}'] = test_each_predss[:, i]
    sub_.to_csv(f'submission_{name}.csv', index=False)


# In[24]:


data = pd.read_csv(os.path.join(filepath, 'data.csv'), index_col=[0])
original = pd.read_csv('/kaggle/input/predicting-heat-flux/Data_CHF_Zhao_2020_ATE.csv', index_col=[0])

plt.figure(figsize=(18, 5))
sns.set_theme(style="whitegrid")

sns.kdeplot(data=sub, x=target_col, fill=True, alpha=0.5, common_norm=False, label="Predictive")
sns.kdeplot(data=data, x=target_col, fill=True, alpha=0.5, common_norm=False, label="Data")
sns.kdeplot(data=original, x=target_col, fill=True, alpha=0.5, common_norm=False, label="Original")

plt.title('Predictive vs Training Distribution')
plt.legend()
plt.subplots_adjust(top=0.9)
plt.show()


# In[25]:


sub

