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

# Import sklearn classes for model selection, cross validation, and performance evaluation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
from category_encoders import OrdinalEncoder, CountEncoder, CatBoostEncoder, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer, LabelEncoder # OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.decomposition import PCA, NMF

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

get_ipython().system('pip install sklego')
from sklego.linear_model import LADRegression # Least Absolute Deviation Regression

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


filepath = '/kaggle/input/playground-series-s3e16'

df_train = pd.read_csv(os.path.join(filepath, 'train.csv'), index_col=[0])
df_test = pd.read_csv(os.path.join(filepath, 'test.csv'), index_col=[0])
original = pd.read_csv('/kaggle/input/crab-age-prediction/CrabAgePrediction.csv')

df_train['is_generated'] = 1
df_test['is_generated'] = 1
original['is_generated'] = 0

target_col = 'Age'
num_cols = df_test.select_dtypes(include=['float64']).columns.tolist()
cat_cols = df_test.select_dtypes(include=['object']).columns.tolist()

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
# 1. Train, Test and Original data histograms
# 2. Correlation of Features
# 3. Hierarchical Clustering
# 4. Pie and bar charts for categorical column features
# 5. Scatter Plot with Age Column by Sex
# 6. Distribution Plot by Sex
# 7. Boxplot by Sex
# 8. Violinplot by Sex
# 9. Scatter plots after dimensionality reduction with PCA by Sex

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
        
plot_histograms(df_train[num_cols], df_test[num_cols], original[num_cols], target_col, n_cols=4)


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

plot_heatmap(df_train[num_cols+[target_col]], title='Train data')
plot_heatmap(df_test[num_cols], title='Test data')
plot_heatmap(original[num_cols+[target_col]], title='original')


# In[6]:


from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

def hierarchical_clustering(data, title):
    fig, ax = plt.subplots(1, 1, figsize=(14, 6), dpi=120)
    correlations = data.corr()
    converted_corr = 1 - np.abs(correlations)
    Z = linkage(squareform(converted_corr), 'complete')
    
    dn = dendrogram(Z, labels=data.columns, ax=ax, above_threshold_color='#ff0000', orientation='right')
    hierarchy.set_link_color_palette(None)
    plt.grid(axis='x')
    plt.title(f'{title} Hierarchical clustering, Dendrogram', fontsize=18, fontweight='bold')
    plt.show()

hierarchical_clustering(df_train[num_cols], title='Train data')
hierarchical_clustering(df_test[num_cols], title='Test data')


# In[7]:


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
    
plot_target_feature(df_train, 'Sex', figsize=(16,5), palette='colorblind', name='Train data')
plot_target_feature(df_test, 'Sex', figsize=(16,5), palette='colorblind', name='Test data')


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
    
plot_scatter_with_fixed_col(df_train, fixed_col=target_col, hue='Sex', drop_cols=['is_generated'], size=16, title='Train data')
plot_scatter_with_fixed_col(original, fixed_col=target_col, hue='Sex', drop_cols=['is_generated'], size=16, title='Original data')


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
    
plot_distribution(df_train, hue='Sex', title='Train data', drop_cols=['is_generated'])
plot_distribution(df_test, hue='Sex', title='Test data', drop_cols=['is_generated'])
plot_distribution(original, hue='Sex', title='Original data', drop_cols=['is_generated'])


# In[10]:


def plot_boxplot(df, hue, title='', drop_cols=[], n_cols=3):
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
    
plot_boxplot(df_train, hue='Sex', title='Train data', drop_cols=['is_generated'], n_cols=2)
plot_boxplot(original, hue='Sex', title='Original data', drop_cols=['is_generated'], n_cols=2)
plot_boxplot(df_test, hue='Sex', title='Test data', drop_cols=['is_generated'], n_cols=2)


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
    
plot_violinplot(df_train, hue='Sex', title='Train data', drop_cols=['is_generated'], n_cols=2)


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
    
    
data = df_train.drop(['Sex', 'Age'], axis=1)
for method in ['pca', 'nmf']:
    decomp = Decomp(n_components=2, method=method, scaler_method='minmax')
    decomp_feature = decomp.dimension_reduction(data)
    decomp_feature = pd.concat([df_train[['Sex']], decomp_feature], axis=1)
    decomp.decomp_plot(decomp_feature, method.upper(), 'Sex')


# In[13]:


def plot_scatter_with_fixed_col(df, fixed_col, hue=False, drop_cols=[], size=10, title=''):
    sns.set_style('whitegrid')
    
    if hue:
        cols = df.columns.drop([hue, fixed_col] + drop_cols)
    else:
        cols = df.columns.drop([fixed_col] + drop_cols)
    n_cols = 2
    n_rows = 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(size, size/n_cols*n_rows), sharex=False, sharey=False)
    fig.suptitle(f'{title} Set Scatter Plot with Target Column by {hue}', fontsize=24, fontweight='bold', y=1.01)

    for i, col in enumerate(cols):
        n_row = i // n_cols
        n_col = i % n_cols
        ax = axes[n_col]

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
    

decomp = Decomp(n_components=2, method='pca', scaler_method='standard')
decomp_feature = decomp.dimension_reduction(data)
decomp_feature = pd.concat([df_train[['Sex', 'Age']], decomp_feature], axis=1)    
plot_scatter_with_fixed_col(decomp_feature, fixed_col=target_col, hue='Sex', drop_cols=[], size=16, title='PCA data')


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Feature Engineering</p>
# - `cat_encoder()` : Perform label encoding and one-hot encoding. OneHotEnconder was done with the Sex feature this time.
# - `create_features()` : Create new features. 
# - `add_pca_features()` : Add PCA results as features. Since this data set is highly correlated, it may be a good idea to perform a PCA.
# 
# **Note: Not all of the above feature engineering is adapted in this kernel. Please take it as an idea.**

# In[14]:


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
        encoder = OneHotEncoder(cols=cat_cols)
        train_encoder = encoder.fit_transform(X_train[cat_cols]).astype(int)
        test_encoder = encoder.transform(X_test[cat_cols]).astype(int)
        X_train = pd.concat([X_train, train_encoder], axis=1)
        X_test = pd.concat([X_test, test_encoder], axis=1)
        X_train.drop(cat_cols, axis=1, inplace=True)
        X_test.drop(cat_cols, axis=1, inplace=True)
        encoder_cols = list(train_encoder.columns)
        
    return X_train, X_test, encoder_cols

def create_features(df):
    
    # Calculate the Length-to-Diameter Ratio
    df["Length_to_Diameter_Ratio"] = df["Length"] / df["Diameter"]
    
    # Calculate the Length-Minus-Height
    df["Length_Minus_Height"] = df["Length"] - df["Height"]
    
    # Calculate the Weight-to-Shell Weight Ratio
    # df["Weight_to_Shell_Weight_Ratio"] = df["Weight"] / (df["Shell Weight"] + 1e-15)
        
    return df

def add_pca_features(X_train, X_test):    
    
    # Select the columns for PCA
    pca_features = X_train.select_dtypes(include=['float64']).columns.tolist()
    n_components = 4 # len(pca_features)

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


# In[15]:


# Concatenate train and original dataframes, and prepare train and test sets
train = pd.concat([df_train, original])
test = df_test.copy()

X_train = train.drop([f'{target_col}'],axis=1).reset_index(drop=True)
y_train = train[f'{target_col}'].reset_index(drop=True)
X_test = test.reset_index(drop=True)

num_cols = X_train.select_dtypes(include=['float64']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# Category Encoders
X_train, X_test, cat_cols = cat_encoder(X_train, X_test, cat_cols, encode='ohe')

# Create Features
# X_train = create_features(X_train)
# X_test = create_features(X_test)
# X_train, X_test = add_pca_features(X_train, X_test)

# StandardScaler
sc = StandardScaler() # MinMaxScaler or StandardScaler
X_train[num_cols] = sc.fit_transform(X_train[num_cols])
X_test[num_cols] = sc.transform(X_test[num_cols])

# Drop_col
drop_cols = ['is_generated']
X_train.drop(drop_cols, axis=1, inplace=True)
X_test.drop(drop_cols, axis=1, inplace=True)

print(f"X_train shape :{X_train.shape} , y_train shape :{y_train.shape}")
print(f"X_test shape :{X_test.shape}")

# del train, test, df_train, df_test

X_train.head(5)


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Data Splitting</p>

# In[16]:


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

# In[17]:


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
            'learning_rate': 0.00482382842096919,
            'booster': 'gbtree',
            'lambda': 0.000235366507474591,
            'alpha': 0.0000115977765684837,
            'subsample': 0.35955930593108,
            'colsample_bytree': 0.898528184386095,
            'max_depth': 9,
            'min_child_weight': 8,
            'eta': 0.0000784943239744148,
            'gamma': 1.6661346939401E-07,
            'grow_policy': 'lossguide',
            'n_jobs': -1,
            'objective': 'reg:squarederror', # reg:pseudohubererror
            'eval_metric': 'mae',
            'verbosity': 0,
            'random_state': self.random_state,
        }
        if self.device == 'gpu':
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['predictor'] = 'gpu_predictor'
        
        lgb1_params = {
            'n_estimators': self.n_estimators,
            'learning_rate': 0.00659605502010782,
            "reg_alpha": 0.0134568843414818,
            "reg_lambda": 2.38367559632979E-06,
            "num_leaves": 117,
            "colsample_bytree": 0.850706320762174,
            'subsample': 0.691827302225948,
            'subsample_freq': 4,
            'min_child_samples': 33,
            'objective': 'regression_l2',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state
        }
        cat1_params = {
            'iterations': self.n_estimators,
            'depth': 7,
            'learning_rate': 0.00454306521731278,
            'l2_leaf_reg': 0.113774158297261,
            'random_strength': 0.0179641854849499,
            'od_type': 'IncToDec',
            'od_wait': 50,
            'bootstrap_type': 'Bayesian',
            'grow_policy': 'Lossguide',
            'bagging_temperature': 1.39240858193441,
            'eval_metric': 'MAE',
            'loss_function': 'MAE',
            'task_type': self.device.upper(),
            'verbose': False,
            'allow_writing_files': False,
            'random_state': self.random_state
        }
        hist_params = {
            'loss': 'absolute_error',
            'l2_regularization': 0.0104104133357932,
            'early_stopping': True,
            'learning_rate': 0.00627298859709192,
            'max_iter': self.n_estimators,
            'n_iter_no_change': 200,
            'max_depth': 16,
            'max_bins': 255,
            'min_samples_leaf': 54,
            'max_leaf_nodes':57,
            'random_state': self.random_state,
            #'categorical_features': []
        }
        gbd_params = {
            'loss': 'absolute_error',
            'n_estimators': 800,
            'max_depth': 10,
            'learning_rate': 0.01,
            'min_samples_split': 10,
            'min_samples_leaf': 20,
            'random_state': self.random_state,
        }
        
        models = {
#             "lad": LADRegression(),
            "xgb": xgb.XGBRegressor(**xgb_params),
            "lgb": lgb.LGBMRegressor(**lgb1_params),
#             "cat": CatBoostRegressor(**cat1_params),
            'hgb': HistGradientBoostingRegressor(**hist_params),
            "SVR_rbf": SVR(kernel="rbf", gamma="auto"),
#             "SVR_linear": SVR(kernel="linear", gamma="auto"),
#             "Ridge": RidgeCV(),
#             "Lasso": LassoCV(),
#             "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),            
            "RandomForestRegressor": RandomForestRegressor(n_estimators=500, random_state=self.random_state, n_jobs=-1),
#             "SGDRegressor": SGDRegressor(max_iter=2000, early_stopping=True, n_iter_no_change=100, random_state=self.random_state),
            "MLPRegressor": MLPRegressor(max_iter=500, early_stopping=True, n_iter_no_change=10, random_state=self.random_state),
#             "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=500, n_jobs=-1, random_state=self.random_state),
#             "PLSRegression": PLSRegression(n_components=10, max_iter=1000),
            #"PassiveAggressiveRegressor": PassiveAggressiveRegressor(max_iter=1000, tol=1e-3, random_state=self.random_state),
#             "TheilSenRegressor": TheilSenRegressor(max_iter=1000, random_state=self.random_state, n_jobs=-1),
            "GradientBoostingRegressor": GradientBoostingRegressor(**gbd_params),
#             "ARDRegression": ARDRegression(n_iter=1000),
#             "HuberRegressor": HuberRegressor(max_iter=2000)
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

# In[18]:


splitter = Splitter(kfold=False)
for X_train_, X_val, y_train_, y_val in splitter.split_data(X_train, y_train, random_state_list=[42]):
    print('Data set by train_test_split')


# In[19]:


get_ipython().run_cell_magic('time', '', '\nn_estimators = 500\nscoring = \'neg_mean_absolute_error\'\nmin_features_to_select = 6\n\nregressor = Regressor(n_estimators, device=\'cpu\', random_state=0)\nmodels = regressor.models\n\nmodels_name = [_ for _ in regressor.models_name if (\'xgb\' in _) or (\'lgb\' in _) or (\'cat\' in _)]\ntrained_models = dict(zip(models_name, [\'\' for _ in range(regressor.len_models)]))\nunnecessary_features = dict(zip(models_name, [[] for _ in range(regressor.len_models)]))\nfor name, model in models.items():\n    if (\'xgb\' in name) or (\'lgb\' in name) or (\'cat\' in name):\n        elimination = RFECV(\n            model, \n            step=1,\n            min_features_to_select=min_features_to_select,\n            cv=2,\n            scoring=scoring, \n            n_jobs=-1)\n        elimination.fit(X_train_, y_train_)\n        unnecessary_feature = list(X_train.columns[~elimination.get_support()])\n        idx = np.argmax(elimination.cv_results_[\'mean_test_score\'])\n        mean_score = elimination.cv_results_[\'mean_test_score\'][idx]\n        std_score = elimination.cv_results_[\'std_test_score\'][idx]\n        print(f\'{blu}{name}{res} {red} Best Mean{res} MAE {red}{mean_score:.5f} ± {std_score:.5f}{res} | N_STEP {idx}\')\n        print(f"unnecessary_feature: {unnecessary_feature}")\n        print(f\'{"-" * 60}\')\n        \n        trained_models[f\'{name}\'] = deepcopy(elimination)\n        unnecessary_features[f\'{name}\'].extend(unnecessary_feature)\n        \nunnecessary_features = np.concatenate([_ for _ in unnecessary_features.values()])\nfeatures = np.unique(unnecessary_features, return_counts=True)[0]\ncounts = np.unique(unnecessary_features, return_counts=True)[1]\ndrop_features = list(features[counts >= 2])\nprint("Features recommended to be removed:", drop_features)\n')


# In[20]:


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


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Weighted Ensemble by Optuna on Training</p>
# A weighted average is performed during training;  
# The weights were determined for each model using the predictions for the train data created in the out of fold with Optuna's CMAsampler. (Here it is defined by `OptunaWeights`)  
# This is an extension of the averaging method. All models are assigned different weights defining the importance of each model for prediction.
# 
# ![](https://www.analyticsvidhya.com/wp-content/uploads/2015/08/Screen-Shot-2015-08-22-at-6.40.37-pm.png)

# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:85%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Optimizer (--> Optimize MAE)</p>

# In[21]:


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
        score = mean_absolute_error(y_true, weighted_pred)
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


# In[22]:


# Settings
kfold = 'skf'
n_splits = 5
n_reapts = 1 # 1
random_state = 42
n_estimators = 99999 # 99999
early_stopping_rounds = 200
n_trials = 3000 # 3000
verbose = False
device = 'cpu'

# Fix seed
random.seed(random_state)
random_state_list = random.sample(range(9999), n_reapts)

# To calculate runtime
def sec_to_minsec(t):
    min_ = int(t / 60)
    sec = int(t - min_*60)
    return min_, sec

# Process
def mattop_post_process(preds, y_true):
    unique_targets = np.unique(y_true)
    return np.array([min(unique_targets, key = lambda x: abs(x - pred)) for pred in preds])


# In[23]:


get_ipython().run_cell_magic('time', '', '\n# Initialize an array for storing test predictions\nregressor = Regressor(n_estimators, device, random_state)\ntest_predss = np.zeros((X_test.shape[0]))\npost_test_predss = np.zeros((X_test.shape[0]))\noof_predss = np.zeros((X_train.shape[0], n_reapts))\nensemble_score, ensemble_score_ = [], []\nweights = []\ntrained_models = dict(zip([_ for _ in regressor.models_name if (\'xgb\' in _) or (\'lgb\' in _) or (\'cat\' in _)], [[] for _ in range(regressor.len_models)]))\nscore_dict = dict(zip(regressor.models_name, [[] for _ in range(regressor.len_models)]))\n\nsplitter = Splitter(kfold=kfold, n_splits=n_splits, cat_df=y_train)\nfor i, (X_train_, X_val, y_train_, y_val, val_index) in enumerate(splitter.split_data(X_train, y_train, random_state_list=random_state_list)):\n    n = i % n_splits\n    m = i // n_splits\n            \n    # Get a set of regressor models\n    regressor = Regressor(n_estimators, device, random_state_list[m])\n    models = regressor.models\n    \n    # Initialize lists to store oof and test predictions for each base model\n    oof_preds = []\n    test_preds = []\n    post_test_preds = []\n    \n    # Loop over each base model and fit it to the training data, evaluate on validation data, and store predictions\n    for name, model in models.items():\n        best_iteration = None\n        start_time = time.time()\n        if (\'xgb\' in name) or (\'lgb\' in name) or (\'cat\' in name):\n            early_stopping_rounds_ = int(early_stopping_rounds*2) if (\'cat\' not in name) else early_stopping_rounds\n            model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds_, verbose=verbose)\n            best_iteration = model.best_iteration if (\'xgb\' in name) else model.best_iteration_\n        else:\n            model.fit(X_train_, y_train_)\n        end_time = time.time()\n        min_, sec = sec_to_minsec(end_time - start_time)\n            \n        if name in trained_models.keys():\n            trained_models[f\'{name}\'].append(deepcopy(model))\n        \n        y_val_pred = mattop_post_process(model.predict(X_val).reshape(-1), y_val)\n        test_pred = model.predict(X_test).reshape(-1)\n        post_test_pred = mattop_post_process(model.predict(X_test).reshape(-1), y_train)\n        \n        score = mean_absolute_error(y_val, y_val_pred)\n        score_dict[name].append(score)\n        print(f\'{blu}{name}{res} [FOLD-{n} SEED-{random_state_list[m]}] MAE {blu}{score:.5f}{res} | Best iteration {blu}{best_iteration}{res} | {min_}min {sec}s\')\n        \n        oof_preds.append(y_val_pred)\n        test_preds.append(test_pred)\n        post_test_preds.append(post_test_pred)\n    \n    # Use Optuna to find the best ensemble weights\n    optweights = OptunaWeights(random_state=random_state_list[m], n_trials=n_trials)\n    y_val_pred = optweights.fit_predict(y_val.values, oof_preds)\n    \n    score = mean_absolute_error(y_val, y_val_pred)\n    print(f\'{red}>>> Ensemble{res} [FOLD-{n} SEED-{random_state_list[m]}] MAE {red}{score:.5f}{res}\')\n    print(f\'{"-" * 60}\')\n    ensemble_score.append(score)\n    weights.append(optweights.weights)\n    \n    # Predict to X_test by the best ensemble weights\n    test_predss += optweights.predict(test_preds) / (n_splits * len(random_state_list))\n    post_test_predss += optweights.predict(post_test_preds) / (n_splits * len(random_state_list))\n    oof_predss[X_val.index, m] += optweights.predict(oof_preds)\n    \n    gc.collect()\n')


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Mean Scores for each model</p>

# In[24]:


def plot_score_from_dict(score_dict, title='MAE', ascending=True):
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

print('--- Mean MAE Scores---')    
for name, score in score_dict.items():
    mean_score = np.mean(score)
    std_score = np.std(score)
    print(f'{name}: {red}{mean_score:.5f} ± {std_score:.5f}{res}')
plot_score_from_dict(score_dict, title=f'MAE (n_splits:{n_splits})')


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Weight of the Optuna Ensemble</p>

# In[25]:


# Calculate the mean LogLoss score of the ensemble
mean_score = np.mean(ensemble_score)
std_score = np.std(ensemble_score)
print(f'{red}Mean{res} Optuna Ensemble MAE {red}{mean_score:.5f} ± {std_score:.5f}{res}')

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

# In[26]:


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
# Reference: https://www.kaggle.com/competitions/playground-series-s3e14/discussion/407327

# In[27]:


unique_targets = np.unique(y_train)
def mattop_post_process(preds):
     return np.array([min(unique_targets, key = lambda x: abs(x - pred)) for pred in preds])


# In[28]:


def oof_result(oof_preds, y_train, title):
    plt.figure(figsize=(20, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(oof_preds, kde=True, alpha=0.5, label='oof_preds')
    sns.histplot(y_train.values, kde=True, alpha=0.5, label='y_train')
    plt.title('Histogram of OOF Predictions and Train Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=y_train.values, y=oof_preds, alpha=0.5)
    plt.xlabel('Actual Values')
    plt.ylabel('OOF Predicted Values')
    plt.title('Actual vs. OOF Predicted Values')
    plt.suptitle(f'{title}', fontweight='bold', fontsize=16)

    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', alpha=0.5)
    plt.show()
    
oof_result(np.mean(oof_predss, axis=1), y_train, title='')
oof_result(mattop_post_process(np.mean(oof_predss, axis=1)), y_train, title='After Mattop post process')


# In[29]:


def make_submission(test_predss, prefix=''):
    # No post-processing during training
    subs = pd.DataFrame()

    sub = pd.read_csv(os.path.join(filepath, 'sample_submission.csv'))
    sub[f'{target_col}'] = test_predss
    sub.to_csv(f'{prefix}submission.csv', index=False)

    subs = pd.concat([subs, sub], axis=1)

    sub = pd.read_csv(os.path.join(filepath, 'sample_submission.csv'))
    sub[f'{target_col}'] = mattop_post_process(test_predss)
    sub.to_csv(f'{prefix}submission_mattop.csv', index=False)

    display(pd.merge(subs, sub, on='id', suffixes=("", "_mattop")))
    
make_submission(test_predss, prefix='') # No post-processing during training
make_submission(post_test_predss, prefix='post_') # With post-processing during training

