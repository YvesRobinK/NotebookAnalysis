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
import time
import math

# Import sklearn classes for model selection, cross validation, and performance evaluation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
from category_encoders import OrdinalEncoder, CountEncoder, CatBoostEncoder, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer, LabelEncoder # OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.decomposition import PCA, NMF
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
import shap

# Import libraries for Hypertuning
import optuna

# Import libraries for gradient boosting
import lightgbm as lgb
import xgboost as xgb
from xgboost.callback import EarlyStopping
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.svm import NuSVC, SVC
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier
from catboost import Pool

from sklearn.feature_selection import RFE, RFECV
from sklearn.inspection import permutation_importance

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from colorama import Style, Fore
blk = Style.BRIGHT + Fore.BLACK
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
res = Style.RESET_ALL


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Data</p>

# In[2]:


filepath = '/kaggle/input/playground-series-s3e17'

df_train = pd.read_csv(os.path.join(filepath, 'train.csv'), index_col=[0])
df_test = pd.read_csv(os.path.join(filepath, 'test.csv'), index_col=[0])
original = pd.read_csv('/kaggle/input/machine-failure-predictions/machine failure.csv', index_col=[0])

target_col = 'Machine failure'
# num_cols = df_test.select_dtypes(include=['int64']).columns.tolist()
num_cols = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]
binary_cols = [
    'TWF',
    'HDF',
    'PWF',
    'OSF',
    'RNF'
]
cat_cols = df_test.select_dtypes(include=['object']).columns.tolist()

df_train['is_generated'] = 1
df_test['is_generated'] = 1
original['is_generated'] = 0

print(f"train shape :{df_train.shape}, ", f"test shape :{df_test.shape}")
print(f"original shape :{original.shape}")


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

def check_data(data, title):
    cols = data.columns.to_list()
    display(set_frame_style(data[cols].head(),f'{title}: First 5 Rows Of Data'))
    display(set_frame_style(data[cols].describe(),f'{title}: Summary Statistics'))
    display(set_frame_style(data[cols].nunique().to_frame().rename({0:'Unique Value Count'}, axis=1).transpose(), f'{title}: Unique Value Counts In Each Column'))
    display(set_frame_style(data[cols].isna().sum().to_frame().transpose(), f'{title}:Columns With Nan'))
    
check_data(df_train, 'Train data')
print('-'*100)
check_data(df_test, 'Test data')
print('-'*100)
check_data(original, 'Original data')


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">EDA</p>
# **Contents:**
# 1. Train, Test and Original data histograms
# 2. Correlation of Features
# 3. Scatter plots of features by Machine failure (random undersampling)
# 3. Hierarchical Clustering
# 4. Pie and bar charts for categorical column features
# 6. Distribution Plot by Type
# 7. Boxplot by Machine failure
# 8. Violinplot by Machine failure
# 9. Scatter plots after dimensionality reduction with PCA by Machine failure

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
        
plot_histograms(df_train[num_cols], df_test[num_cols], original[num_cols], target_col, n_cols=3)


# In[5]:


def plot_heatmap(df, title):
    # Create a mask for the diagonal elements
    mask = np.zeros_like(df.astype(float).corr())
    mask[np.triu_indices_from(mask)] = True

    # Set the colormap and figure size
    colormap = plt.cm.RdBu_r
    plt.figure(figsize=(15, 15))

    # Set the title and font properties
    plt.title(f'{title} Correlation of Features', fontweight='bold', y=1.02, size=20)

    # Plot the heatmap with the masked diagonal elements
    sns.heatmap(df.astype(float).corr(), linewidths=0.1, vmax=1.0, vmin=-1.0, 
                square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 14, "weight": "bold"},
                mask=mask)

plot_heatmap(df_train[num_cols+binary_cols+[target_col]], title='Train data')
plot_heatmap(df_test[num_cols+binary_cols], title='Test data')
plot_heatmap(original[num_cols+binary_cols+[target_col]], title='original')


# In[6]:


def plot_scatter_matrix(df, target_col, drop_cols=[], size=26):
    # sns.pairplot()
    
    sns.set_style('whitegrid')
    cols = df.columns.drop([target_col] + drop_cols)
    fig, axes = plt.subplots(len(cols), len(cols), figsize=(size, size), sharex=False, sharey=False)

    for i, col in enumerate(cols):
        for j, col_ in enumerate(cols):
            axes[i,j].set_xlabel(f'{col}', fontsize=14)
            axes[i,j].set_ylabel(f'{col_}', fontsize=14)

            # Plot the scatterplot
            sns.scatterplot(data=df, x=col, y=col_, hue=target_col, ax=axes[i,j],
                            s=80, edgecolor='gray', alpha=0.2, palette='bright')

            axes[i,j].tick_params(axis='both', which='major', labelsize=12)

            if i == 0:
                axes[i,j].set_title(f'{col_}', fontsize=18)
            if j == 0:
                axes[i,j].set_ylabel(f'{col}', fontsize=18)

    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.legend(loc='upper right', ncol=5, fontsize=18)
    plt.show()
    
sampling_strategy = 0.5
rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
X_train_res, y_train_res = rus.fit_resample(df_train.drop(target_col, axis=1), df_train[target_col])
plot_scatter_matrix(pd.concat([X_train_res[num_cols], y_train_res], axis=1), target_col)

# del X_train_res, y_train_res, rus


# In[7]:


from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

def hierarchical_clustering(data, title):
    fig, ax = plt.subplots(1, 1, figsize=(14, 8), dpi=120)
    correlations = data.corr()
    converted_corr = 1 - np.abs(correlations)
    Z = linkage(squareform(converted_corr), 'complete')
    
    dn = dendrogram(Z, labels=data.columns, ax=ax, above_threshold_color='#ff0000', orientation='right')
    hierarchy.set_link_color_palette(None)
    plt.grid(axis='x')
    plt.title(f'{title} Hierarchical clustering, Dendrogram', fontsize=18, fontweight='bold')
    plt.show()

hierarchical_clustering(df_train[num_cols+binary_cols], title='Train data')
hierarchical_clustering(df_test[num_cols+binary_cols], title='Test data')


# In[8]:


def plot_target_feature(df_train, target_col, figsize=(16,5), palette='colorblind', name='Train'):
    df_train = df_train.fillna('Nan')

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax = ax.flatten()

    # Pie chart
    pie_colors = sns.color_palette(palette, len(df_train[target_col].unique()))
    ax[0].pie(
        df_train[target_col].value_counts(),
        shadow=True,
        explode=[0.05] * len(df_train[target_col].unique()),
        autopct='%1.f%%',
        textprops={'size': 15, 'color': 'white'},
        colors=pie_colors
    )
    ax[0].set_aspect('equal')  # Fix the aspect ratio to make the pie chart circular

    # Bar plot
    bar_colors = sns.color_palette(palette)
    sns.countplot(
        data=df_train,
        y=target_col,
        ax=ax[1],
        palette=bar_colors
    )
    ax[1].set_xlabel('Count', fontsize=14)
    ax[1].set_ylabel('')
    ax[1].tick_params(labelsize=12)
    ax[1].yaxis.set_tick_params(width=0)  # Remove tick lines for y-axis

    fig.suptitle(f'{target_col} in {name} Dataset', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Show the plot
    plt.show()

plot_target_feature(df_train, 'Type', figsize=(16,5), palette='colorblind', name='Train data')
plot_target_feature(df_test, 'Type', figsize=(16,5), palette='colorblind', name='Test data')
plot_target_feature(df_train, target_col, figsize=(16,5), palette='colorblind', name='Train data')
plot_target_feature(original, target_col, figsize=(16,5), palette='colorblind', name='Original data')


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

plot_distribution(df_train[num_cols+['Type']], hue='Type', title='Train data')
plot_distribution(df_test[num_cols+['Type']], hue='Type', title='Test data')
# plot_distribution(df_train[num_cols+[target_col]], hue=target_col, title='Train data')
# plot_distribution(original[num_cols+[target_col]], hue=target_col, title='Original data')


# In[10]:


def plot_boxplot(df, hue, drop_cols=[], n_cols=3, title=''):
    sns.set_style('whitegrid')

    cols = df.columns.drop([hue] + drop_cols)
    n_rows = (len(cols) - 1) // n_cols + 1

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(14, 4*n_rows))

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
    
plot_boxplot(df_train[num_cols+[target_col]], hue=target_col, n_cols=3, title='Train data')
plot_boxplot(original[num_cols+[target_col]], hue=target_col, n_cols=3, title='Original data')


# In[11]:


def plot_violinplot(df, hue, drop_cols=[], n_cols=2, title=''):
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
    
plot_violinplot(df_train[num_cols+[target_col]], hue=target_col, n_cols=3, title='Train data')
plot_violinplot(original[num_cols+[target_col]], hue=target_col, n_cols=3, title='Original data')


# In[12]:


class Decomp:
    def __init__(self, n_components, method="pca", scaler_method='standard'):
        self.n_components = n_components
        self.method = method
        self.scaler_method = scaler_method
        
    def dimension_reduction(self, df):
            
        X_reduced = self.dimension_method(df)
        df_comp = pd.DataFrame(X_reduced, columns=[f'{self.method.upper()}_{_}' for _ in range(self.n_components)], index=df.index)
        return df_comp
    
    def dimension_method(self, df):
        
        X = self.scaler(df)
        if self.method == "pca":
            pca = PCA(n_components=self.n_components, random_state=0)
            X_reduced = pca.fit_transform(X)
            self.comp = pca
        elif self.method == "nmf":
            nmf = NMF(n_components=self.n_components, random_state=0)
            X_reduced = nmf.fit_transform(X)
        else:
            raise ValueError(f"Invalid method name: {method}")
        
        return X_reduced
    
    def scaler(self, df):
        
        _df = df.copy()
            
        if self.scaler_method == "standard":
            return StandardScaler().fit_transform(_df)
        elif self.scaler_method == "minmax":
            return MinMaxScaler().fit_transform(_df)
        elif self.scaler_method == None:
            return _df.values
        else:
            raise ValueError(f"Invalid scaler_method name")
        
    def get_columns(self):
        return [f'{self.method.upper()}_{_}' for _ in range(self.n_components)]
    
    def get_explained_variance_ratio(self):
        return np.sum(self.comp.explained_variance_ratio_)
    
    def transform(self, df):
        X = self.scaler(df)
        X_reduced = self.comp.transform(X)
        df_comp = pd.DataFrame(X_reduced, columns=[f'{self.method.upper()}_{_}' for _ in range(self.n_components)], index=df.index)
        
        return df_comp
    
    def decomp_plot(self, tmp, label, hue='genre'):
        plt.figure(figsize = (16, 9))
        sns.scatterplot(x = f"{label}_0", y = f"{label}_1", data=tmp, hue=hue, alpha=0.7, s=100, palette='muted');

        plt.title(f'{label} on {hue}', fontsize = 20)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 10);
        plt.xlabel(f"{label} Component 1", fontsize = 15)
        plt.ylabel(f"{label} Component 2", fontsize = 15)
    
    
data = X_train_res[num_cols].copy()
for method in ['pca', 'nmf']:
    decomp = Decomp(n_components=2, method=method, scaler_method='minmax')
    decomp_feature = decomp.dimension_reduction(data)
    decomp_feature = pd.concat([y_train_res, decomp_feature], axis=1)
    decomp.decomp_plot(decomp_feature, method.upper(), target_col)
    
del X_train_res, y_train_res, rus, data


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Feature Engineering</p>
# - `replace_Type()` : Perform label encoding for the `Type` column.
# - `create_features()` : Create new features. 
# - `add_pca_features()` : Add PCA results as features.
# - `cat_encoder` : Label Encoder
# 
# **Note: Not all of the above feature engineering is adapted in this kernel. Please take it as an idea.**

# In[13]:


def create_features(df):
    
    # Create a new feature by subtracting 'Air temperature' from 'Process temperature'
    # df['Temperature difference [K]'] = df['Process temperature [K]'] - df['Air temperature [K]']
    
    # Create a new feature by divided 'Air temperature' from 'Process temperature'
    df["Temperature ratio"] = df['Process temperature [K]'] / df['Air temperature [K]']
    
    # Create a new feature by multiplying 'Torque' and 'Rotational speed'
    df['Torque * Rotational speed'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']

    # Create a new feature by multiplying 'Torque' by 'Tool wear'
    df['Torque * Tool wear'] = df['Torque [Nm]'] * df['Tool wear [min]']

    # Create a new feature by adding 'Air temperature' and 'Process temperature'
    # df['Temperature sum [K]'] = df['Air temperature [K]'] + df['Process temperature [K]']
    
    # Create a new feature by multiplying 'Torque' by 'Rotational speed'
    df['Torque * Rotational speed'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']
        
    new_cols = [
        #'Temperature difference [K]', 
        'Temperature ratio', 
        'Torque * Rotational speed',
        'Torque * Tool wear', 
        # 'Temperature sum [K]', 
        'Torque * Rotational speed'
    ]
    
    return df, new_cols

def add_pca_features(X_train, X_test):    
    
    # Select the columns for PCA
    pca_features = X_train.select_dtypes(include=['float64']).columns.tolist()
    n_components = 2 # len(pca_features)

    # Create the pipeline
    pipeline = make_pipeline(StandardScaler(), PCA(n_components=n_components))
    
    # Perform PCA
    pipeline.fit(X_train[pca_features])

    # Create column names for PCA features
    pca_columns = [f'PCA_{i}' for i in range(n_components)]

    # Add PCA features to the dataframe
    X_train[pca_columns] = pipeline.transform(X_train[pca_features])
    X_test[pca_columns] = pipeline.transform(X_test[pca_features])

    return X_train, X_test

def replace_Type(df):
    
    df["Type"] = df["Type"].replace({'L':0})
    df["Type"] = df["Type"].replace({'M':1})
    df["Type"] = df["Type"].replace({'H':2})
    df["Type"] = df["Type"].astype(int)
    
    return df

def cat_encoder(X_train, X_test, cat_cols, encode='label'):
    
    if encode == 'label':
        ## Label Encoder
        encoder = OrdinalEncoder(cols=cat_cols)
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

def rename(df):
    df.columns = df.columns.str.replace('\[.*?\]', '', regex=True)
    df.columns = df.columns.str.strip()
    # df.columns = df.columns.str.replace(' ', '')
    
    return df


# In[14]:


# Concatenate train and original dataframes, and prepare train and test sets
train = pd.concat([df_train, original])
test = df_test.copy()

X_train = train.drop([f'{target_col}'],axis=1).reset_index(drop=True)
y_train = train[f'{target_col}'].reset_index(drop=True)
X_test = test.reset_index(drop=True)

# Category Encoders
X_train = replace_Type(X_train)
X_test = replace_Type(X_test)
X_train, X_test, _ = cat_encoder(X_train, X_test, ['Product ID'], encode='label')
cat_cols = ['Type', 'Product ID']

# Create Features
new_cols = []
X_train, _ = create_features(X_train)
X_test, new_cols = create_features(X_test)
# X_train, X_test = add_pca_features(X_train, X_test)

# StandardScaler
sc = StandardScaler() # MinMaxScaler or StandardScaler
X_train[num_cols+new_cols] = sc.fit_transform(X_train[num_cols+new_cols])
X_test[num_cols+new_cols] = sc.transform(X_test[num_cols+new_cols])

# Drop_col
drop_cols = ['is_generated', 'RNF'] # binary_cols
X_train.drop(drop_cols, axis=1, inplace=True)
X_test.drop(drop_cols, axis=1, inplace=True)

# Rename
X_train = rename(X_train)
X_test = rename(X_test)

print(f"X_train shape :{X_train.shape} , y_train shape :{y_train.shape}")
print(f"X_test shape :{X_test.shape}")

del train, test, df_train, df_test

X_train.head(5)


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Data Splitting</p>

# In[15]:


class Splitter:
    def __init__(self, kfold=True, n_splits=5, cat_df=pd.DataFrame(), test_size=0.5):
        self.n_splits = n_splits
        self.kfold = kfold
        self.cat_df = cat_df
        self.test_size = test_size

    def split_data(self, X, y, random_state_list):
        if self.kfold == 'skf':
            for random_state in random_state_list:
                kf = StratifiedKFold(n_splits=self.n_splits, random_state=random_state, shuffle=True)
                for train_index, val_index in kf.split(X, self.cat_df):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                    yield X_train, X_val, y_train, y_val, val_index
        elif self.kfold:
            for random_state in random_state_list:
                kf = KFold(n_splits=self.n_splits, random_state=random_state, shuffle=True)
                for train_index, val_index in kf.split(X, y):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                    yield X_train, X_val, y_train, y_val, val_index
        else:
            for random_state in random_state_list:
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.test_size, random_state=random_state)
                yield X_train, X_val, y_train, y_val


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Define Model</p>
# LightGBM, CatBoost Xgboost and HistGradientBoosting hyper parameters are determined by optuna.

# In[16]:


class Classifier:
    def __init__(self, n_estimators=100, device="cpu", random_state=0):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.models = self._define_model()
        self.models_name = list(self._define_model().keys())
        self.len_models = len(self.models)
        
    def _define_model(self):
                
        xgb1_params = {
            'n_estimators': self.n_estimators,
            'learning_rate': 0.0503196477566407,
            'booster': 'gbtree',
            'lambda': 0.00379319640405843,
            'alpha': 0.106754104302093,
            'subsample': 0.938028434508189,
            'colsample_bytree': 0.212545425027345,
            'max_depth': 9,
            'min_child_weight': 2,
            'eta': 1.03662446190642E-07,
            'gamma': 0.000063826049787043,
            'grow_policy': 'lossguide',
            'n_jobs': -1,
            'objective': 'binary:logistic',
            #'eval_metric': 'auc',
            'verbosity': 0,
            'random_state': self.random_state,
        }
        xgb2_params = {
            'n_estimators': self.n_estimators,
            'learning_rate': 0.00282353606391198,
            'booster': 'gbtree',
            'lambda': 0.399776698351379,
            'alpha': 1.01836149061356E-07,
            'subsample': 0.957123754766769,
            'colsample_bytree': 0.229857555596548,
            'max_depth': 9,
            'min_child_weight': 4,
            'eta': 2.10637756839133E-07,
            'gamma': 0.00314857715085414,
            'grow_policy': 'depthwise',
            'n_jobs': -1,
            'objective': 'binary:logistic',
            #'eval_metric': 'auc',
            'verbosity': 0,
            'random_state': self.random_state,
        }
        xgb3_params = {
            'n_estimators': self.n_estimators,
            'learning_rate': 0.00349356650247156,
            'booster': 'gbtree',
            'lambda': 0.0002963239871324443,
            'alpha': 0.0000162103492458353,
            'subsample': 0.822994064549709,
            'colsample_bytree': 0.244618079894501,
            'max_depth': 9,
            'min_child_weight': 2,
            'eta': 8.03406601824666E-06,
            'gamma': 3.91180893163099E-07,
            'grow_policy': 'depthwise',
            'n_jobs': -1,
            'objective': 'binary:logistic',
            #'eval_metric': 'auc',
            'verbosity': 0,
            'random_state': self.random_state,
        }
        if self.device == 'gpu':
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['predictor'] = 'gpu_predictor'
        
        lgb1_params = {
            'n_estimators': self.n_estimators,
            'learning_rate': 0.0124415817896377,
            'reg_alpha': 0.00139174509988134,
            'reg_lambda': 0.000178964551019674,
            'num_leaves': 249,
            'colsample_bytree': 0.675264038614975,
            'subsample': 0.421482143660471,
            'subsample_freq': 4,
            'min_child_samples': 8,
            'objective': 'binary',
            'metric': 'binary_error',
            'boosting_type': 'gbdt',
            'is_unbalance':True,
            # 'n_jobs': -1,
            #'force_row_wise': True,
            'device': self.device,
            'random_state': self.random_state
        }
        lgb2_params = {
            'n_estimators': self.n_estimators,
            'learning_rate': 0.0247403801218241,
            'reg_alpha': 6.84813726047269E-06,
            'reg_lambda': 3.40443691552308E-08,
            'num_leaves': 223,
            'colsample_bytree': 0.597332047776164,
            'subsample': 0.466442641250326,
            'subsample_freq': 2,
            'min_child_samples': 5,
            'objective': 'binary',
            'metric': 'binary_error',
            'boosting_type': 'gbdt',
            'is_unbalance':True,
            # 'n_jobs': -1,
            #'force_row_wise': True,
            'device': self.device,
            'random_state': self.random_state
        }
        lgb3_params = {
            'n_estimators': self.n_estimators,
            'learning_rate': 0.0109757020463629,
            'reg_alpha': 0.174927073496136,
            'reg_lambda': 2.45325882544558E-07,
            'num_leaves': 235,
            'colsample_bytree': 0.756605772162953,
            'subsample': 0.703911560320816,
            'subsample_freq': 5,
            'min_child_samples': 21,
            'objective': 'binary',
            'metric': 'binary_error',
            'boosting_type': 'gbdt',
            'is_unbalance':True,
            # 'n_jobs': -1,
            #'force_row_wise': True,
            'device': self.device,
            'random_state': self.random_state
        }
        cat1_params = {
            'iterations': self.n_estimators,
            'depth': 3,
            'learning_rate': 0.020258010893459,
            'l2_leaf_reg': 0.583685138705941,
            'random_strength': 0.177768021213223,
            'od_type': "Iter", 
            'od_wait': 116,
            'bootstrap_type': "Bayesian",
            'grow_policy': 'Depthwise',
            'bagging_temperature': 0.478048798393903,
            'eval_metric': 'Logloss', # AUC
            'loss_function': 'Logloss',
            'auto_class_weights': 'Balanced',
            'task_type': self.device.upper(),
            'verbose': False,
            'allow_writing_files': False,
            'random_state': self.random_state
        }
        cat2_params = {
            'iterations': self.n_estimators,
            'depth': 5,
            'learning_rate': 0.00666304601039438,
            'l2_leaf_reg': 0.0567881687170355,
            'random_strength': 0.00564702921370138,
            'od_type': "Iter", 
            'od_wait': 93,
            'bootstrap_type': "Bayesian",
            'grow_policy': 'Depthwise',
            'bagging_temperature': 2.48298505165348,
            'eval_metric': 'Logloss', # AUC
            'loss_function': 'Logloss',
            'auto_class_weights': 'Balanced',
            'task_type': self.device.upper(),
            'verbose': False,
            'allow_writing_files': False,
            'random_state': self.random_state
        }
        cat3_params = {
            'iterations': self.n_estimators,
            'depth': 5,
            'learning_rate': 0.0135730417743519,
            'l2_leaf_reg': 0.0597353604503262,
            'random_strength': 0.0675876600077264,
            'od_type': "Iter", 
            'od_wait': 122,
            'bootstrap_type': "Bayesian",
            'grow_policy': 'Depthwise',
            'bagging_temperature': 1.85898154006468,
            'eval_metric': 'Logloss', # AUC
            'loss_function': 'Logloss',
            'auto_class_weights': 'Balanced',
            'task_type': self.device.upper(),
            'verbose': False,
            'allow_writing_files': False,
            'random_state': self.random_state
        }
        hist_params = {
            'l2_regularization': 0.880153002159043,
            'learning_rate': 0.00251637614495506,
            'max_iter': self.n_estimators,
            'max_depth': 18,
            'max_bins': 255,
            'min_samples_leaf': 67,
            'max_leaf_nodes':66,
            'early_stopping': True,
            'n_iter_no_change': 50,
            'categorical_features': ['Type'],
            'class_weight':'balanced',
            'random_state': self.random_state
        }
        
        models = {
            "xgb": xgb.XGBClassifier(**xgb1_params),
            #"xgb2": xgb.XGBClassifier(**xgb2_params),
            #"xgb3": xgb.XGBClassifier(**xgb3_params),
            "lgb": lgb.LGBMClassifier(**lgb1_params),
            "lgb2": lgb.LGBMClassifier(**lgb2_params),
            "lgb3": lgb.LGBMClassifier(**lgb3_params),
            "cat": CatBoostClassifier(**cat1_params),
            "cat2": CatBoostClassifier(**cat2_params),
            "cat3": CatBoostClassifier(**cat3_params),
            'hgb': HistGradientBoostingClassifier(**hist_params),
            'rf': RandomForestClassifier(n_estimators=500, n_jobs=-1, class_weight="balanced", random_state=self.random_state),
            #'brf': BalancedRandomForestClassifier(n_estimators=1000, random_state=self.random_state), #  n_jobs=-1, 
            'lr': LogisticRegressionCV(max_iter=2000, random_state=self.random_state),
            #'svc': SVC(max_iter=300, kernel="rbf", gamma="auto", probability=True, class_weight="balanced",random_state=self.random_state),
        }
        
        return models


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Feature Selection (RFE-CV)</p>
# RFECV is a technique for automated feature selection that combines recursive feature elimination and cross-validation to identify the optimal subset of features for a given machine learning task.
# 
# **Note: RFE-CV takes a lot of time. Here n_estimators are reduced to save time. When originally used, it is recommended to run with the actual hyperparameters.**
# 
# ![](https://www.researchgate.net/publication/362548828/figure/fig4/AS:11431281087244666@1664503792971/Recursive-Feature-Elimination-with-Cross-ValidationRFECV-flowchart-model-diagram.png)
# 
# Reference: Feature fusion based machine learning pipeline to improve breast cancer prediction, 81, p.37627–37655 (2022) https://doi.org/10.1007/s11042-022-13498-4

# In[17]:


splitter = Splitter(kfold=False, test_size=0.6)
for X_train_, X_val, y_train_, y_val in splitter.split_data(X_train, y_train, random_state_list=[42]):
    print('Data set by train_test_split')


# In[18]:


get_ipython().run_cell_magic('time', '', '\nn_estimators = 200\nscoring = \'roc_auc\'\nmin_features_to_select = 10\n\nclassifier = Classifier(n_estimators, device=\'cpu\', random_state=0)\nmodels = classifier.models\n\nmodels_name = [_ for _ in classifier.models_name if (\'xgb\' in _) or (\'lgb\' in _) or (\'cat\' in _)]\ntrained_models = dict(zip(models_name, [\'\' for _ in range(classifier.len_models)]))\nunnecessary_features = dict(zip(models_name, [[] for _ in range(classifier.len_models)]))\nfor name, model in models.items():\n    if (\'xgb\' in name) or (\'lgb\' in name) or (\'cat\' in name):\n        elimination = RFECV(\n            model, \n            step=1,\n            min_features_to_select=min_features_to_select,\n            cv=2,\n            scoring=scoring, \n            n_jobs=-1)\n        elimination.fit(X_train_, y_train_)\n        unnecessary_feature = list(X_train.columns[~elimination.get_support()])\n        idx = np.argmax(elimination.cv_results_[\'mean_test_score\'])\n        mean_score = elimination.cv_results_[\'mean_test_score\'][idx]\n        std_score = elimination.cv_results_[\'std_test_score\'][idx]\n        print(f\'{blu}{name}{res} {red} Best Mean{res} {scoring} {red}{mean_score:.5f} ± {std_score:.5f}{res} | N_STEP {idx}\')\n        print(f"Best unnecessary_feature: {unnecessary_feature}")\n        removed_features = [f for i, f in enumerate(X_train.columns) if elimination.support_[i] == False]\n        ranked_features = sorted(zip(X_train.columns, elimination.ranking_), key=lambda x: x[1])\n        removed_features_by_ranking = [f[0] for f in ranked_features if f[0] in removed_features][::-1]\n        print("Removed features:", removed_features_by_ranking)\n        print(f\'{"-" * 60}\')\n        \n        trained_models[f\'{name}\'] = deepcopy(elimination)\n        unnecessary_features[f\'{name}\'].extend(unnecessary_feature)\n        \nunnecessary_features = np.concatenate([_ for _ in unnecessary_features.values()])\nfeatures = np.unique(unnecessary_features, return_counts=True)[0]\ncounts = np.unique(unnecessary_features, return_counts=True)[1]\ndrop_features = list(features[counts >= 2])\nprint("Features recommended to be removed:", drop_features)\n')


# In[19]:


def plot_recursive_feature_elimination(elimination, scoring, min_features_to_select, name):
    n_scores = len(elimination.cv_results_["mean_test_score"])
    plt.figure(figsize=(10, 4))
    plt.xlabel("Number of features selected")
    plt.ylabel(f"{scoring}")

    # Plot the mean test scores with error bars
    plt.errorbar(
        range(min_features_to_select, n_scores + min_features_to_select),
        elimination.cv_results_["mean_test_score"],
        yerr=elimination.cv_results_["std_test_score"],
        fmt='o-',
        capsize=3,
        markersize=4,
    )

    plt.title(f"{name} Recursive Feature Elimination with correlated features", fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
for name, elimination in trained_models.items():
    plot_recursive_feature_elimination(elimination, scoring, min_features_to_select, name)


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Configuration</p>

# In[20]:


# Settings
kfold = 'skf'
n_splits = 10 # 10
n_reapts = 1 # 1
random_state = 42
n_estimators = 9999 # 99999
early_stopping_rounds = 200
n_trials = 2000 # 2000
verbose = False
device = 'cpu'

# Under Sampling
n_under_sampling = False # or False

# Pseudo Labeling
true_th = 0.99
false_th = 0.9999

# Fix seed
random.seed(random_state)
random_state_list = random.sample(range(9999), n_reapts)

# metrics
def auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)
metric = auc
metric_name = metric.__name__.upper()

# To calculate runtime
def sec_to_minsec(t):
    min_ = int(t / 60)
    sec = int(t - min_*60)
    return min_, sec

# Process


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">One Model Xgboost</p>
# The xgboost model architecture is based on the following: 
# [[PS S3E14, 2023] EDA and Submission](https://www.kaggle.com/code/sergiosaharovskiy/ps-s3e14-2023-eda-and-submission)

# In[21]:


feature_importances_ = pd.DataFrame(index=X_train.columns)
eval_results_ = {}
models_ = []
oof = np.zeros((X_train.shape[0]))
test_predss = np.zeros((X_test.shape[0]))

splitter = Splitter(kfold=kfold, n_splits=n_splits, cat_df=y_train)
for i, (X_train_, X_val, y_train_, y_val, val_index) in enumerate(splitter.split_data(X_train, y_train, random_state_list=[0])):
    fold = i % n_splits
    m = i // n_splits
    
    # XGB .train() requires xgboost.DMatrix.
    # https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.DMatrix
    fit_set = xgb.DMatrix(X_train_, y_train_)
    val_set = xgb.DMatrix(X_val, y_val)
    watchlist = [(fit_set, 'fit'), (val_set, 'val')]

    # Training.
    # https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.training
    classifier = Classifier(3000, device)
    xgb_params = classifier.models['xgb'].get_params()
    # xgb_params = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.01).get_params()
    
    eval_results_[fold] = {}
    model = xgb.train(
        num_boost_round=xgb_params['n_estimators'],
        params=xgb_params,
        dtrain=fit_set,
        evals=watchlist,
        evals_result=eval_results_[fold],
        verbose_eval=False,
        callbacks=[EarlyStopping(early_stopping_rounds, data_name='val', save_best=True)])
        
    val_preds = model.predict(val_set)
    test_predss += model.predict(xgb.DMatrix(X_test)) / n_splits
    
    oof[val_index] = val_preds

    val_score = metric(y_val, val_preds)
    best_iter = model.best_iteration
    print(f'Fold: {blu}{fold:>3}{res}| {metric_name}: {blu}{val_score:.5f}{res}' f' | Best iteration: {blu}{best_iter:>4}{res}')

    # Stores the feature importances
    feature_importances_[f'gain_{fold}'] = feature_importances_.index.map(model.get_score(importance_type='gain'))
    feature_importances_[f'split_{fold}'] = feature_importances_.index.map(model.get_score(importance_type='weight'))

    # Stores the model
    models_.append(model)

# Submission
sub = pd.read_csv(os.path.join(filepath, 'sample_submission.csv'))
sub[f'{target_col}'] = test_predss
sub.to_csv(f'xgb_submission.csv', index=False)    

mean_cv_score_full = metric(y_train, oof)
print(f'{"*" * 50}\n{red}Mean{res} {metric_name} : {red}{mean_cv_score_full:.5f}')


# In[22]:


metric_score_folds = pd.DataFrame.from_dict(eval_results_).T
fit_rmsle = metric_score_folds.fit.apply(lambda x: x['logloss'])
val_rmsle = metric_score_folds.val.apply(lambda x: x['logloss'])

n_splits = len(metric_score_folds)
n_rows = math.ceil(n_splits / 3)

fig, axes = plt.subplots(n_rows, 3, figsize=(20, n_rows * 4), dpi=150)
ax = axes.flatten()

for i, (f, v, m) in enumerate(zip(fit_rmsle, val_rmsle, models_)): 
    sns.lineplot(f, color='#B90000', ax=ax[i], label='fit')
    sns.lineplot(v, color='#048BA8', ax=ax[i], label='val')
    ax[i].legend()
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].set_title(f'Fold {i}', fontdict={'fontweight': 'bold'})
    
    color = ['#048BA8', '#90A6B1']
    best_iter = m.best_iteration
    span_range = [[0, best_iter], [best_iter + 10, best_iter + early_stopping_rounds]]
    
    for idx, sub_title in enumerate([f'Best\nIteration: {best_iter}', f'Early\n Stopping: {early_stopping_rounds}']):
        ax[i].annotate(sub_title,
                       xy=(sum(span_range[idx]) / 2, 4000),
                       xytext=(0, 0),
                       textcoords='offset points',
                       va="center",
                       ha="center",
                       color="w",
                       fontsize=12,
                       fontweight='bold',
                       bbox=dict(boxstyle='round4', pad=0.4, color=color[idx], alpha=0.6))
        ax[i].axvspan(span_range[idx][0] - 0.4, span_range[idx][1] + 0.4, color=color[idx], alpha=0.07)
        
    ax[i].set_xlim(0, best_iter + 20 + early_stopping_rounds)
    ax[i].set_xlabel('Boosting Round', fontsize=12)
    ax[i].set_ylabel('MAE', fontsize=12)
    ax[i].legend(loc='upper right', title=metric_name)

for j in range(i+1, n_rows * 3):
    ax[j].axis('off')

plt.tight_layout()
plt.show()


# In[23]:


fi = feature_importances_
fi_gain = fi[[col for col in fi.columns if col.startswith('gain')]].mean(axis=1)
fi_splt = fi[[col for col in fi.columns if col.startswith('split')]].mean(axis=1)

fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

# Split fi.
data_splt = fi_splt.sort_values(ascending=False)
sns.barplot(x=data_splt.values, y=data_splt.index, 
            color='#1E90FF', linewidth=0.5, edgecolor="black", ax=ax[0])
ax[0].set_title(f'Feature Importance "Split"', fontdict={'fontweight': 'bold'})
ax[0].set_xlabel("Importance", fontsize=12)
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)

# Gain fi.    
data_gain = fi_splt.sort_values(ascending=False)
sns.barplot(x=data_gain.values, y=data_gain.index,
            color='#4169E1', linewidth=0.5, edgecolor="black", ax=ax[1])
ax[1].set_title(f'Feature Importance "Gain"', fontdict={'fontweight': 'bold'})
ax[1].set_xlabel("Importance", fontsize=12)
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)

plt.tight_layout()
plt.show()

del eval_results_, models_, oof


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Pseudo Labeling</p>
# 
# In the following description, it is SVM, but any algorithm can be used. Here the pseudo labels are generated based on the Xgboost results. To avoid labeling uncertain data, only data with a high degree of confidence should be labeled. Here, class 0 is set to 0.999 and class 1 to 0.99.  
# **Note: Pseudo-labels were tried on this data set, but did not lead to an increase in LB scores, so they are commented out.**
# 
# ![](https://journals.sagepub.com/cms/10.1177/00368504221124004/asset/images/large/10.1177_00368504221124004-fig2.jpeg)
# Reference. https://journals.sagepub.com/doi/10.1177/00368504221124004?icid=int.sj-abstract.similar-articles.4

# In[24]:


# df = pd.DataFrame(np.stack([1 - test_predss, test_predss], axis=1), columns=[f'{n}' for n in range(2)])
# true_idx, false_idx = df[df['1'] > true_th].index, df[df['0'] > false_th].index
# print(f'False label: {len(true_idx)}, ', f'True label: {len(false_idx)}')

# false_X_test = X_test.loc[false_idx].copy()
# false_X_test[target_col] = int(0)
# true_X_test = X_test.loc[true_idx].copy()
# true_X_test[target_col] = int(1)

# X_train = pd.concat([X_train_ori, true_X_test.drop(target_col, axis=1), false_X_test.drop(target_col, axis=1)], axis=0)
# y_train = pd.concat([y_train_ori, true_X_test[target_col], false_X_test[target_col]], axis=0)

# X_train.reset_index(drop=True, inplace=True)
# y_train.reset_index(drop=True, inplace=True)

# print(f"X_train shape :{X_train.shape} , y_train shape :{y_train.shape}")
# print(f"X_test shape :{X_test.shape}")

# del df, true_idx, false_idx, true_X_test, false_X_test


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Weighted Ensemble Model by Optuna on Training</p>
# A weighted average is performed during training;  
# The weights were determined for each model using the predictions for the train data created in the out of fold with Optuna's CMAsampler. (Here it is defined by `OptunaWeights`)  
# This is an extension of the averaging method. All models are assigned different weights defining the importance of each model for prediction.
# 
# ![](https://www.analyticsvidhya.com/wp-content/uploads/2015/08/Screen-Shot-2015-08-22-at-6.40.37-pm.png)

# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:85%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Optimizer (--> Optimize AUC)</p>

# In[25]:


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
        score = metric(y_true, weighted_pred)
        return score

    def fit(self, y_true, y_preds):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        pruner = optuna.pruners.HyperbandPruner()
        self.study = optuna.create_study(sampler=sampler, pruner=pruner, study_name="OptunaWeights", direction='maximize')
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


# In[26]:


get_ipython().run_cell_magic('time', '', '\n# Initialize an array for storing test predictions\nclassifier = Classifier(n_estimators, device, random_state)\ntest_predss = np.zeros((X_test.shape[0]))\noof_predss = np.zeros((X_train.shape[0], n_reapts))\nensemble_score, ensemble_score_ = [], []\nweights = []\ntrained_models = dict(zip([_ for _ in classifier.models_name if (\'xgb\' in _) or (\'lgb\' in _) or (\'cat\' in _)], [[] for _ in range(classifier.len_models)]))\nscore_dict = dict(zip(classifier.models_name, [[] for _ in range(classifier.len_models)]))\n\nsplitter = Splitter(kfold=kfold, n_splits=n_splits, cat_df=y_train)\nfor i, (X_train_, X_val, y_train_, y_val, val_index) in enumerate(splitter.split_data(X_train, y_train, random_state_list=random_state_list)):\n    n = i % n_splits\n    m = i // n_splits\n            \n    # Get a set of classifier models\n    classifier = Classifier(n_estimators, device, random_state_list[m])\n    models = classifier.models\n    \n    # Initialize lists to store oof and test predictions for each base model\n    oof_preds = []\n    test_preds = []\n    \n    # Loop over each base model and fit it to the training data, evaluate on validation data, and store predictions\n    for name, model in models.items():\n        best_iteration = None\n        start_time = time.time()\n        \n        if (\'xgb\' in name) or (\'lgb\' in name) or (\'cat\' in name):\n            early_stopping_rounds_ = int(early_stopping_rounds*2) if (\'xgb\' not in name) else early_stopping_rounds\n            \n            if \'lgb\' in name:\n                model.fit(\n                    X_train_, y_train_, eval_set=[(X_val, y_val)], categorical_feature=cat_cols,\n                    early_stopping_rounds=early_stopping_rounds_, verbose=verbose)\n            elif \'cat\' in name :\n                model.fit(\n                    Pool(X_train_, y_train_, cat_features=cat_cols), eval_set=Pool(X_val, y_val, cat_features=cat_cols),\n                    early_stopping_rounds=early_stopping_rounds_, verbose=verbose)\n            else:\n                model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds_, verbose=verbose)\n                \n            best_iteration = model.best_iteration if (\'xgb\' in name) else model.best_iteration_\n        else:\n            model.fit(X_train_, y_train_)\n                \n        end_time = time.time()\n        min_, sec = sec_to_minsec(end_time - start_time)\n        \n        if name in trained_models.keys():\n            trained_models[f\'{name}\'].append(deepcopy(model))\n        \n        y_val_pred = model.predict_proba(X_val)[:, 1].reshape(-1)\n        test_pred = model.predict_proba(X_test)[:, 1].reshape(-1)\n        \n        score = metric(y_val, y_val_pred)\n        score_dict[name].append(score)\n        print(f\'{blu}{name}{res} [FOLD-{n} SEED-{random_state_list[m]}] {metric_name} {blu}{score:.5f}{res} | Best iteration {blu}{best_iteration}{res} | Runtime {min_}min {sec}s\')\n        \n        oof_preds.append(y_val_pred)\n        test_preds.append(test_pred)\n    \n    # Use Optuna to find the best ensemble weights\n    optweights = OptunaWeights(random_state=random_state_list[m], n_trials=n_trials)\n    y_val_pred = optweights.fit_predict(y_val.values, oof_preds)\n    \n    score = metric(y_val, y_val_pred)\n    print(f\'{red}>>> Ensemble{res} [FOLD-{n} SEED-{random_state_list[m]}] {metric_name} {red}{score:.5f}{res}\')\n    print(f\'{"-" * 60}\')\n    ensemble_score.append(score)\n    weights.append(optweights.weights)\n    \n    # Predict to X_test by the best ensemble weights\n    test_predss += optweights.predict(test_preds) / (n_splits * len(random_state_list))\n    oof_predss[X_val.index, m] += optweights.predict(oof_preds)\n    \n    gc.collect()\n')


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Mean Scores for each model</p>

# In[27]:


def plot_score_from_dict(score_dict, title='', ascending=True):
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

print(f'--- Mean {metric_name} Scores---')    
for name, score in score_dict.items():
    mean_score = np.mean(score)
    std_score = np.std(score)
    print(f'{name}: {red}{mean_score:.5f} ± {std_score:.5f}{res}')
plot_score_from_dict(score_dict, title=f'{metric_name} (n_splits:{n_splits})')


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Weight of the Optuna Ensemble</p>

# In[28]:


# Calculate the mean LogLoss score of the ensemble
mean_score = np.mean(ensemble_score)
std_score = np.std(ensemble_score)
print(f'{red}Mean{res} Optuna Ensemble {metric_name} {red}{mean_score:.5f} ± {std_score:.5f}{res}')

print('')
# Print the mean and standard deviation of the ensemble weights for each model
print('--- Optuna Weights---')
mean_weights = np.mean(weights, axis=0)
std_weights = np.std(weights, axis=0)
for name, mean_weight, std_weight in zip(models.keys(), mean_weights, std_weights):
    print(f'{name}: {blu}{mean_weight:.5f} ± {std_weight:.5f}{res}')

# weight_dict = dict(zip(list(score_dict.keys()), np.array(weights).T.tolist()))
# plot_score_from_dict(weight_dict, title='Model Weights', ascending=False)
normalize = [((weight - np.min(weight)) / (np.max(weight) - np.min(weight))).tolist() for weight in weights]
weight_dict = dict(zip(list(score_dict.keys()), np.array(normalize).T.tolist()))
plot_score_from_dict(weight_dict, title='Optuna Weights (Normalize 0 to 1)', ascending=False)


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Feature importance Visualization</p>

# In[29]:


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


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">SHAP Analysis</p>
# SHAP stands for SHapley Additive exPlanations, a method for determining the contribution of each variable (feature) to the model's predicted outcome. Since SHAP cannot be adapted for ensemble models, let's use SHAP to understand `Xgboost` and `Catboost`.
# 
# **Consideration of Results:**  
# The `HDF` appears to be the most effective, as it comes the highest in both Catboost and Xgboost. This feature is binary, though. Next in importance is `Rotational speed`. This is understandably tied to `Machine failure`.
# 
# ![](https://data-analysis-stats.jp/wp-content/uploads/2020/01/SHAP02.png)
# 
# Reference1. https://meichenlu.com/2018-11-10-SHAP-explainable-machine-learning/  
# Reference2. https://christophm.github.io/interpretable-ml-book/shap.html

# ### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:85%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Xgboost</p>

# In[30]:


shap.initjs()
explainer = shap.TreeExplainer(model=trained_models['xgb'][-1])
shap_values = explainer.shap_values(X=X_val)


# In[31]:


# Bar plot
plt.figure(figsize=(20, 14))
shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
plt.title("Feature Importance - Bar", fontsize=16)

# Dot plot
plt.figure(figsize=(20, 14))
shap.summary_plot(shap_values, X_val, plot_type="dot", show=False)
plt.title("Feature Importance - Dot", fontsize=16)

# Adjust layout and display the plots side by side
plt.tight_layout()
plt.show()


# In[32]:


num_cols = min(3, len(X_val.columns))
num_rows = (len(X_val.columns) + num_cols - 1) // num_cols

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(6*num_cols, 4*num_rows))

for i, ind in enumerate(X_val.columns):
    row = i // num_cols
    col = i % num_cols

    shap.dependence_plot(ind=ind, shap_values=shap_values, features=X_val, feature_names=X_val.columns, ax=axes[row, col], show=False)

plt.tight_layout() 
plt.show()


# ### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:85%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Catboost</p>

# In[33]:


explainer = shap.TreeExplainer(model=trained_models['cat'][-1])
shap_values = explainer.shap_values(X=X_val)


# In[34]:


# Bar plot
plt.figure(figsize=(20, 14))
shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
plt.title("Feature Importance - Bar", fontsize=16)

# Dot plot
plt.figure(figsize=(20, 14))
shap.summary_plot(shap_values, X_val, plot_type="dot", show=False)
plt.title("Feature Importance - Dot", fontsize=16)

# Adjust layout and display the plots side by side
plt.tight_layout()
plt.show()


# In[35]:


num_cols = min(3, len(X_val.columns))
num_rows = (len(X_val.columns) + num_cols - 1) // num_cols

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(6*num_cols, 4*num_rows))

for i, ind in enumerate(X_val.columns):
    row = i // num_cols
    col = i % num_cols

    shap.dependence_plot(ind=ind, shap_values=shap_values, features=X_val, feature_names=X_val.columns, ax=axes[row, col], show=False)

plt.tight_layout() 
plt.show()


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Make Submission</p>

# In[36]:


from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

def show_confusion_roc(oof, title='Model Evaluation Results'):
    f, ax = plt.subplots(1, 2, figsize=(13.3, 4))
    df = pd.DataFrame(np.stack([oof[0], oof[1]]), index=['preds', 'target']).T
    cm = confusion_matrix(df.target, df.preds.ge(0.5).astype(int))
    cm_display = ConfusionMatrixDisplay(cm).plot(cmap='GnBu_r', ax=ax[0])
    ax[0].grid(False)
    RocCurveDisplay.from_predictions(df.target, df.preds, ax=ax[1])
    ax[1].grid(True)
    plt.suptitle(f'{title}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    #plt.grid()
    
show_confusion_roc(oof=[np.mean(oof_predss, axis=1), y_train], title=f'OOF Evaluation Results')


# In[37]:


def make_submission(test_predss, prefix=''):
    sub = pd.read_csv(os.path.join(filepath, 'sample_submission.csv'))
    sub[f'{target_col}'] = test_predss
    sub.to_csv(f'{prefix}submission.csv', index=False)
    return  sub

sub = make_submission(test_predss, prefix='')
sub


# In[38]:


# df_train = pd.read_csv(os.path.join(filepath, 'train.csv'), index_col=[0])
# df_test = pd.read_csv(os.path.join(filepath, 'test.csv'), index_col=[0])
# original = pd.read_csv('/kaggle/input/machine-failure-predictions/machine failure.csv', index_col=[0])

plt.figure(figsize=(16, 6))
sns.set_theme(style="whitegrid")

sns.kdeplot(data=sub, x=target_col, fill=True, alpha=0.5, common_norm=False, label="Predict")
# sns.kdeplot(data=df_train, x=target_col, fill=True, alpha=0.5, common_norm=False, label="Data")
# sns.kdeplot(data=original, x=target_col, fill=True, alpha=0.5, common_norm=False, label="Original")

plt.title('Predictive vs Training Distribution')
plt.legend()
plt.subplots_adjust(top=0.9)
plt.show()

