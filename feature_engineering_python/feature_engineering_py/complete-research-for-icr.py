#!/usr/bin/env python
# coding: utf-8

# ![cover_1](https://raw.githubusercontent.com/AniMilina/ICR---Identifying-Age-Related-Conditions/main/cover_1.jpg)

# # Task: Identifying Health Characteristics Associated with Three Age-Related Conditions
# 
# ### Input Data
# 
# - The competition data includes over fifty anonymous health characteristics associated with three age-related conditions.
# 
# ### Goal
# 
# - The goal of this research is to predict whether a subject has been diagnosed with one of these conditions - making it a binary classification task.
# 
# ### Evaluation Metric
# 
# - The evaluation in this competition is based on balanced log loss, which takes into account the importance of both classes and their predicted probabilities.
# 
# ### Tasks
# 
# 1. Exploratory Data Analysis:
#    - Study the data structure and health characteristics' features.
#    - Analyze the distribution of features and the target variable.
#    - Identify outliers and missing values.
# 
# 2. Data Preprocessing:
#    - Handle outliers and abnormal values.
#    - Fill in missing values or remove corresponding records.
# 
# 3. Feature Engineering:
#    - Extract information from existing features.
#    - Encode categorical features.
#    - Generate combined features.
# 
# 4. Modeling:
#    - Choose appropriate machine learning algorithms for classification.
#    - Tune hyperparameters of models.
#    - Evaluate the impact of original and new features on model performance.
# 
# 5. Model Evaluation:
#    - Measure model training and prediction times.
#    - Calculate additional model metrics.
# 
# 6. Selecting the Best Model:
#    - Compare models based on quality metrics.
#    - Study the results to decide on the best model.
# 
# 7. Further Research:
#    - Conduct an in-depth analysis of relationships between characteristics and health conditions.
# 
# # Our Data: Dataset Description
# 
# The competition data includes over fifty anonymous health characteristics associated with three age-related conditions. Our goal is to predict whether a subject has been diagnosed with one of these conditions - making it a binary classification problem.
# 
# At the initial stage, the actual test set is hidden. The full test set contains about 400 rows, which will be provided by the client after evaluating the research.
# 
# ### Files and Field Descriptions
# 
# - train.csv: Training dataset.
#   - Id: Unique identifier for each observation.
#   - AB-GL: Fifty-six anonymous health characteristics. All are numerical, except EJ, which is categorical.
#   - Class: Binary target. Value 1 indicates that the subject has been diagnosed with one of the three conditions, and value 0 indicates the absence of a diagnosed condition.
# 
# - test.csv: Test dataset. Your goal is to predict the probability of the subject in this dataset belonging to each of the two classes.
# 
# - greeks.csv: Additional metadata, available only for the training set.
#   - Alpha: Defines the type of age-related condition, if present.
#   - A: Absence of age-related changes. Corresponds to class 0.
#   - B, D, G: Three age-related conditions. Correspond to class 1.
#   - Beta, Gamma, Delta: Three experimental features.
#   - Epsilon: Data collection date for this question. Note that all data in the test set was collected after the training set.
# 
# - sample_submission.csv: Sample submission file in the correct format. Refer to the Evaluation page for more detailed information.

# # Data Preprocessing

#  ###  Importing  Libraries

# In[1]:


import pandas as pd
import numpy as np

import os
import json
import warnings
import joblib
from datetime import date, time, datetime
from time import time
from tqdm.notebook import tqdm
from itertools import combinations

import tensorflow_decision_forests as tfdf
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.io as pio

import phik

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import SequentialFeatureSelector, SelectKBest, f_regression

from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, QuantileTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# from sklearn_extensions.preprocessing import SplineTransformer

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor,DummyClassifier
from xgboost import XGBRegressor,XGBClassifier
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from lightgbm import LGBMRegressor,LGBMClassifier
from catboost import CatBoostRegressor,CatBoostClassifier

from sklearn.metrics import mean_squared_error,log_loss
from sklearn.inspection import permutation_importance

import optuna
# from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution


# # Constants

# In[2]:


PATH_LOCAL = 'datasets/'                               # local path to data
PATH_REMOTE = '/datasets/'                             # remote path to data

CR = '\n'                                              # carriage return

RANDOM_STATE = RANDOM_SEED = RS = 88                   # random state
TARGET = 'Class'                                       # target feature
SCORING = 'neg_log_loss'                               # scoring metric
VALID_FRAC = 0.2                                       # fraction of validation set
N_CV = 5                                               # number of cross-validation splits

N_TRIALS = 30                                          # maximum number of trials for Optuna optimization
TIMEOUT = 1000                                         # maximum execution time for Optuna optimization


# In[3]:


ESTIMATOR_LIST = [
    'DummyClassifier',
    'XGBClassifier',
    # 'LinearSVC',
    'RandomForestClassifier',
    'LGBMClassifier',
    # 'CatBoostClassifier',
]


# # Functions

# In[4]:


# Function to Get Data Information

def explore_dataframe(df):
#     # Shape
#     shape_info = pd.DataFrame({"Shape of dataframe": [f"Total: {df.shape[0]} rows, {df.shape[1]} columns"]})
#     shape_info = shape_info.replace(np.nan, "-")
    
    # Data Types
    data_types_info = df.dtypes.to_frame().reset_index().rename(columns={"index": "Data Type", 0: ""})
    
    # Missing Values
    missing_values_info = df.isnull().sum().to_frame().reset_index().rename(columns={"index": "Missing Values", 0: ""})
    missing_values_info["Missing Values"] = missing_values_info["Missing Values"].fillna("-")
    
    # Duplicate Rows
    duplicate_rows_info = pd.DataFrame({"Duplicate rows in dataframe": [f"Total: {df.duplicated().sum()}"]})
    duplicate_rows_info = duplicate_rows_info.replace(np.nan, "-")
    
     # Unique Values
    unique_values_info = df.nunique().to_frame().reset_index().rename(columns={"index": "Column", 0: "Unique Values"})
    
    # Describe
    describe_info = df.describe().transpose().reset_index().rename(columns={"index": "Column"})

    # Concatenate tables
    info_table = pd.concat([data_types_info, missing_values_info, unique_values_info], axis=1) #shape_info, на время удалила
    
    
    # Display tables
    display(df.head())
    display(df.describe())  
    display(info_table)


# In[5]:


def mutual_info(df, target_name, task=None, min_neighbors=1, max_neighbors=7):
    '''
    Calculates feature importance using mutual_info
    df: dataframe with features and target variable
    target_name: name of the target variable
    task: choose the task - classification or regression
    min_neighbors, max_neighbors: range of k for k-neighbors (the final result is averaged)
    '''

    if max_neighbors < min_neighbors:
        print("Parameter 'max_neighbors' can't be less than parameter 'min_neighbors'.")
        return

    X = df.copy()
    Y = X.pop(target_name)

    df_mutual_info = pd.DataFrame(index=X.columns)

    # Label encoding for categoricals
    for column in X.select_dtypes(exclude='number'):
        X[column], _ = X[column].factorize()

    # All discrete features should have integer dtypes
    for k in range(min_neighbors, max_neighbors+1, 2):
        if task == 'classification':
            df_mutual_info[f'k_{k}'] = mutual_info_classif(X, Y, n_neighbors=k, random_state=RS)
        elif task == 'regression':
            df_mutual_info[f'k_{k}'] = mutual_info_regression(X, Y, n_neighbors=k, random_state=RS)
        else:
            print('Wrong parameter "task". Available task="classification" or task="regression".')
            return

    df_mutual_info['average'] = df_mutual_info.mean(axis=1)
    df_mutual_info = df_mutual_info.sort_values('average', ascending=False)

    display(df_mutual_info)

    fig, ax = plt.subplots(figsize=(15, df_mutual_info.shape[0]/3), dpi=PLOT_DPI)
    sns.barplot(x=df_mutual_info.average, y=df_mutual_info.index, palette=['#808080', 'hotpink'], data=df_mutual_info)
    ax.set_xlabel(f'mutual_info (average across from 1 to {max_neighbors} neighbors)')
    plt.show()
    
    df_mutual_info['average'] = df_mutual_info.mean(axis=1)
    df_mutual_info = df_mutual_info.sort_values('average', ascending=False)
    
    display(df_mutual_info)
    
    fig, ax = plt.subplots(figsize=(15, df_mutual_info.shape[0]/3), dpi=PLOT_DPI)
    sns.barplot(x=df_mutual_info.average, y=df_mutual_info.index, color='steelblue')
    ax.set_xlabel(f'mutual_info (average across from 1 to {max_neighbors} neighbours)')
    plt.show()


# In[6]:


def target_correlation_significance(df, target_name, interval_features):
    '''
    df: DataFrame containing features and target variable
    target_name: name of the target variable
    interval_features: list of interval features (required for more accurate Phik computation)

    Calculates:
    - Correlation of features with respect to the target variable
    - Normalized statistical significance of features
    - Product of correlation and statistical significance
    - Harmonic mean of correlation and statistical significance

    Sorts the features based on the harmonic mean.
    '''
    
    # correlation to target
    df_corr = df.phik_matrix(interval_cols=interval_features)[target_name].to_frame().drop(target_name, axis=0)
    df_corr.columns = ['correlation']
    
    # significance of the correlations
    df_significance = df.significance_matrix(interval_cols=interval_features, nsim=50)[target_name].to_frame().drop(target_name, axis=0)
    df_significance = df_significance.assign(significance=lambda x: x[target_name] / x[target_name].max()).drop(target_name, axis=1)
    
    # joined
    df_joined = df_corr.join(df_significance, how='outer')
    df_joined['product'] = df_joined['correlation'] * df_joined['significance']
    df_joined['harmonic_mean'] = (2 * df_joined['correlation'] * df_joined['significance']) / (
                df_joined['correlation'] + df_joined['significance'])
    df_joined = df_joined.sort_values('harmonic_mean', ascending=False)
    
    # display table
    display(df_joined)
    
    # plot
    fig, ax = plt.subplots(figsize=(15, df_joined.shape[0]/3), dpi=100)
    sns.barplot(x='harmonic_mean', y=df_joined.index, palette=['#808080', 'hotpink'], data=df_joined)
    ax.set_xlabel('Harmonic Mean of Target Correlation and Significance')
    plt.show()


# In[7]:


def plot_Optuna(study, plot_kind='plot_slice', model_name=''):
    '''
    Additional customization of original Optuna plots.
    For example, for the `plot_slice` plot, the color of points initially depended on the iteration number.
    Now, all points have the same color and are semi-transparent, making clusters of points more visible.

    study: trained object of OptunaSearchCV class
    plot_kind: type of Optuna plot
    model_name: name of the model
    '''
    
    if plot_kind == 'plot_slice':
        fig = optuna.visualization.plot_slice(study)
        fig.update_traces(
            marker_color='lightgrey',
            marker_size=3,
            marker_opacity=0.5,
            marker_line_width=0.5,
            marker_line_color='black',
        )
    
    elif plot_kind == 'plot_param_importances':
        fig = optuna.visualization.plot_param_importances(study)
        
    elif plot_kind == 'plot_optimization_history':
        fig = optuna.visualization.plot_optimization_history(study)
        fig.update_traces(
            marker_size=5,
            marker_opacity=0.3,
            marker_line_width=1,
            marker_line_color='black',
        )

    fig.update_layout(
        title_text=model_name,
        title_x=0,
        font_size=10,
    )    
    fig.show()


# In[8]:


def plot_feature_importances(chart_title, feature_names, feature_importances):
    """
    Plots the feature importance chart used by the model.

   chart_title: title of the chart
   feature_names: names of the features
   feature_importances: importance of the features
    """

    df = pd.DataFrame({'features': feature_names,
                       'importances': feature_importances.importances_mean,
                       'std_err': feature_importances.importances_std,
                      }).sort_values('importances', ascending=False)
    
    fig, ax = plt.subplots(figsize=(15, df.shape[0]/3), dpi=PLOT_DPI)
    
    sns.barplot(
                x=df.importances,
                y=df.features,
                xerr=df.std_err,
                color='steelblue',
               )
    
    ax.set_title(f'{chart_title}')
    ax.set_xlim(-0.02,)


# In[9]:


def add_model_metrics(models, X_train, Y_train, X_valid, Y_valid, cv=N_CV, scoring_list=['f1', 'neg_log_loss']):
    '''
    Accepts:
    - dataframe with a list of models and their characteristics;
    - two datasets (features and target) - training and validation sets;
    - cv parameter for cross_val_score;
    - a list of metrics.
    
    For each model in the dataframe, it adds the specified metrics for both datasets.
    '''

    def cv_score(model, X, Y, scoring, cv):
        invert_koeff = -1 if scoring.split('_')[0] == 'neg' else 1   # invert metrics prefixed with "neg_"
        if scoring == 'neg_log_loss':
            return -1 * cross_val_score(model, X, Y, scoring=scoring, cv=cv, n_jobs=-1, method='predict_proba').mean()
        else:
            return invert_koeff * cross_val_score(model, X, Y, scoring=scoring, cv=cv, n_jobs=-1).mean()
    
    for scoring in scoring_list:
    
        # model results on the training set (cross-validation averaging)
        models[scoring + '_train'] = models.model.apply(cv_score, args=(X_train, Y_train, scoring, cv))

        # results of models on the test set (averaging over cross-validation)
        models[scoring + '_valid'] = models.model.apply(cv_score , args=(X_valid, Y_valid, scoring, cv))
    
    # optimal hyperparameters
    models['best_params'] = models.study.apply(lambda model: model.best_params)
    
    return models


# In[10]:


def extract_final_features(pipeline_model):
    '''
    Accepts pipeline.
    Returns a list of features on which the final estimator of the pipeline is trained.
    '''
    feature_list = []
    
    for feature in pipeline_model.steps[-2][1].get_feature_names_out():
        feature_list.append(feature.split('__')[1])

    return feature_list


# In[11]:


def plot_feature_importances(chart_title, feature_names, feature_importances):
    '''
    Displays a graph of the importance of the features used by the model.

    :chart_title: chart title
    :feature_names: feature names
    :feature_importances: feature importance
    '''

    fig, ax = plt.subplots(figsize=(15, 5), dpi=PLOT_DPI)

    df = pd.DataFrame({'features': feature_names,
                       'importances': feature_importances.importances_mean,
                       'std_err': feature_importances.importances_std
                       }).sort_values('importances', ascending=False)

    sns.barplot(x='importances', y='features', data=df, xerr=df.std_err, color='hotpink')

    ax.set_title(chart_title)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Features')
    ax.set_xlim(0, None)
    ax.grid(True, axis='x')

    plt.tight_layout()
    plt.show()


# In[12]:


def clean_dataset(df):
    """
    Clears the dataset from extra spaces and other values,
    which can create non-obvious duplicates in the data.

    :param dataset: source dataset (pandas DataFrame)
    :return: cleared dataset (pandas DataFrame)
    """
    cleaned_data = df.copy()  # create a copy of the original dataset for changes

   # Clear values from extra spaces
    cleaned_data = cleaned_data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

   # Remove duplicates
    cleaned_data = cleaned_data.drop_duplicates()

    return cleaned_data


# # Settings

# In[13]:


# TextStyle

class f:    
    BOLD = "\033[1m"     # Bold text
    ITALIC = "\033[3m"   # Italic text
    END = "\033[0m"      # Reset style


# In[14]:


# Matplotlib, Seaborn

PLOT_DPI = 150 # dpi for drawing charts
sns.set_style('whitegrid', {'axes.facecolor': '0.98', 'grid.color': '0.9', 'axes.edgecolor': '1.0'})
plt.rc('axes', labelweight='bold', titlesize=16, titlepad=10)

# Plotly Graph_Objects
pio.templates['my_theme'] = go.layout.Template(
    layout_autosize=True,
    layout_height=200,
    layout_legend_orientation="h",
    layout_margin=dict(t=40, b=40),
    layout_template='seaborn'
)
pio.templates.default = 'my_theme'

# colors, color schemes
CMAP_SYMMETRIC = LinearSegmentedColormap.from_list('', ['gray', 'steelblue', 'hotpink'])


# In[15]:


# Pandas defaults

pd.options.display.max_colwidth = 100
pd.options.display.max_rows = 500
pd.options.display.max_columns = 100
pd.options.display.float_format = '{:.3f}'.format
pd.options.display.colheader_justify = 'left'


# In[16]:


# Optuna design

optuna.logging.set_verbosity(optuna.logging.WARNING) # disable logging when optuna is running


# In[17]:


# Disable warnings

warnings.filterwarnings('ignore')


# # Read and validate data

# In[18]:


# Path to data 

path_greeks = '/kaggle/input/icr-identify-age-related-conditions/greeks.csv'
path_train = '/kaggle/input/icr-identify-age-related-conditions/train.csv'
path_test = '/kaggle/input/icr-identify-age-related-conditions/test.csv'


# In[19]:


# Reading data 

greeks_data = pd.read_csv(path_greeks)
train_data = pd.read_csv(path_train)
test_data = pd.read_csv(path_test)


# In[20]:


# Merging data

merged_data = pd.merge(train_data, greeks_data, on='Id', how='left')


# In[21]:


explore_dataframe(train_data)


# After examining the received data on the training set (`train_data`), we can draw the following conclusions:
# 
# **Data type:** Most of the columns are of type `float64`, except for the `"Id"`, `"EJ"` and `"Class"` columns. The `"Class"` column is of type `int64` and the `"EJ"` column is of type `object`
# 
# **Missing values:** Some columns contain missing values. The columns `"BQ"`, `"DU"`, `"DV"`, `"EL"`, `"FC"`, `"FL"`, and `"FS"` have a small number of missing values
# 
# **Unique Values:** Each column has a different number of unique values. Some columns have a large number of unique values, for example, the `"BD"` column has 617 unique values, and the `"BQ"` column has 515 unique values. This is due to the fact that one hundred specified columns contain individual health indicators.
# 
# **Columns:** Column names are presented in the `"Column"` column. They include `AB, AF, AH, AM, AR, AX, AY, AZ` and so on
# 
# **Target variable:** The `"Class"` column is the target variable. It contains two unique values: 0 and 1, which indicate the absence or presence of the disease, respectively.
# 
# **Overall output:** The training dataset (`train_data`) contains many features represented by numerical values. Some columns contain missing values that may need to be processed. The "Class" column is the target variable to be predicted. This data will be useful for training the model and predicting the presence of a disease based on input features.

# In[22]:


explore_dataframe(merged_data)


# Having studied the obtained data on the merged data set `(merged_data)`, we can draw the following conclusions:
# 
# **Data type:** Most of the columns are of type float64, except for the columns `"Id"`, `"EJ"`, `"Alpha"`, `"Beta"`, `"Gamma"`, `"Delta" ` and `"Class"`
# The `"Class"` column is of type int64, and the `"Id"`, `"EJ"`, `"Alpha"`, `"Beta"`, `"Gamma"` and `"Delta"` columns are of type object
# 
# **Missing values:** Some columns contain missing values. The columns `"BQ"`, `"DU"`, `"FL"` and `"FC"` have a small number of missing values, and the column `"EL"` has 60 missing values
# 
# **Unique Values:** Each column has a different number of unique values. Some columns have a large number of unique values. For example, the `"BD"` column has 617 unique values, and the `"Epsilon"` column has 198 unique values. This is due to the fact that one hundred specified columns contain individual health indicators.
# 
# **Columns:** Column names are presented in the `"Column"` column. They include `AB, AF, AH, AM, AR, AX, AY, AZ` and so on
# 
# **Target variable:** The `"Class"` column is the target variable. It contains two unique values: 0 and 1, which indicate the absence or presence of the disease, respectively.
# 
# **Additional metadata:** The merged dataset also contains additional columns such as `"Alpha"`, `"Beta"`, `"Gamma"`, `"Delta"` and `"Epsilon"`. They are of type object and contain information about the type of age state and experimental characteristics.
# 
# **Overall conclusion:** The merged dataset (`merged_data`) contains extended information about health traits, including data from the training set and additional metadata. Most features are represented by numeric values, but there are also categorical columns. Some columns contain missing values that may need to be processed. The "Class" column is the target variable, and additional columns may contain important information for analysis and prediction

# In[23]:


explore_dataframe(test_data)


# # Data correction

# In[24]:


train_data = clean_dataset(train_data)
merged_data = clean_dataset(merged_data)


# In[25]:


# The folder train_data contains a small number of gaps in the columns of individual health indicators, we will replace them with the median

median_BQ = train_data['BQ'].median()
train_data['BQ'].fillna(median_BQ, inplace=True)

median_EL = train_data['EL'].median()
train_data['EL'].fillna(median_EL, inplace=True)

median_DU = train_data['DU'].median()
train_data['DU'].fillna(median_DU, inplace=True)

median_FC = train_data['FC'].median()
train_data['FC'].fillna(median_FC, inplace=True)

median_FL = train_data['FL'].median()
train_data['FL'].fillna(median_FL, inplace=True)


# In[26]:


# The merged_data folder contains a small number of gaps in the columns of individual health indicators, we will replace them with the median

median_BQ = merged_data['BQ'].median()
merged_data['BQ'].fillna(median_BQ, inplace=True)

median_EL = merged_data['EL'].median()
merged_data['EL'].fillna(median_EL, inplace=True)

median_DU = merged_data['DU'].median()
merged_data['DU'].fillna(median_DU, inplace=True)

median_FC = merged_data['FC'].median()
merged_data['FC'].fillna(median_FC, inplace=True)

median_FL = merged_data['FL'].median()
merged_data['FL'].fillna(median_FL, inplace=True)


# In[27]:


# Check the categorical value of the category column of individual health indicators

print(train_data['EJ'].unique())
print(merged_data['EJ'].unique())


# In[28]:


# Fix the EJ column type to a binary value

# train_data['EJ'] = train_data['EJ'].replace({'A': 0, 'B': 1})
# test_data['EJ'] = test_data['EJ'].replace({'A': 0, 'B': 1})
merged_data['EJ'] = merged_data['EJ'].replace({'A': 0, 'B': 1})


# In[29]:


# # # Check the result

# print(merged_data.info())


# ### Correction of duplicates

# In[30]:


duplicates_train = train_data.duplicated()
print("Number of duplicates in train_data:", duplicates_train.sum())


# In[31]:


duplicates_merged = merged_data.duplicated()
print("Number of duplicates in merged_data:", duplicates_merged.sum())


# No duplicates - excellent

# # Investigate features for errors and outliers
# 

# ### Graphs of scatter of numerical features

# > Let's study the training set, since merge_data is identical in numerical terms, then consider one dataset:

# In[32]:


print(f'\nScatter plots of numerical features\n')

num_features = train_data.select_dtypes(include=np.number).columns.to_list()

for feature in num_features:
    fig, ax = plt.subplots(figsize=(15, 0.5), dpi=100)
    sns.boxplot(data=train_data, x=feature, color='lightgray', flierprops={'marker': '.', 'markeredgecolor': '#FF00FF', 'markersize': 1})
    ax.set_xticklabels(ax.get_xticks(), fontsize=8)  # Decreasing the font size of labels along the x-axis
    plt.show()


# We see fairly confident indicators in the data, there are outliers, but since we have individual medical data sets in numerical values, it is obvious that in this case we define an outlier as a variant of the indicator

# ### Check the minimum and maximum values of some features

# In[33]:


# value_counts = merged_data['Epsilon'].value_counts()
# print(value_counts)


# In[34]:


merged_data['Epsilon'] = pd.to_datetime(merged_data['Epsilon'], errors='coerce')

# Checking the minimum and maximum date

min_date = merged_data['Epsilon'].min()
max_date = merged_data['Epsilon'].max()

print('Minimum date:', min_date)
print('Maximum date:', max_date)


# >Data for the study were collected over a period of >8 years. Also, I will not remove dates with a small number of results.
# >
# >The fact that all the data in the test set was collected after the training set was collected may affect the results of the model and the interpretation of those results
# >
# >Here are a few factors that might be important:
# 
# >`Temporal features:` If the data in the test set represents newer information, then it may reflect changes over time, such as new trends or events. In such a case, a model trained on old data may not be accurate enough to predict new data.
# 
# >`Possibility of overfitting:` If the test set was collected after the training set, there is a risk of overfitting the model on the training data and then accurately predicting the same data in the test set. This can lead to an overestimation of the accuracy of the model.
# 
# >`Different data collection conditions:` If the data in the test set were collected under different conditions or using different methods, this may lead to differences in data distribution. In this case, a model trained on some data may not be able to predict well on other data.
# 
# **NB!Given these points, be careful when interpreting model results, especially if the test dataset contains newer information**

# ### Values of categorical features

# > Let's study the training set, only merge_data contains categorical features, then we will consider it:

# In[35]:


# Count the number of unique values

def count_unique_values(data):
    for column in data.select_dtypes(include='object'):
        if column != 'Id':  
            unique_values = data[column].value_counts()
            print(f"Unique values for a column {column}:")
            print(unique_values)
            print()


# In[36]:


count_unique_values(merged_data)


# In[37]:


def plot_bar_charts(data):
    categorical_columns = data.select_dtypes(include='object').columns
    for column in categorical_columns:
        if column != 'Id':
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=data, x=column, palette=['gray', 'hotpink'])
            sns.despine()
            ax.set_xlabel(column)
            ax.set_ylabel('Count')
            ax.set_title(f"Distribution of {column}")
            plt.show()
plot_bar_charts(merged_data)


# We see:
# 
# > `Alpha` - Specifies the type of age state, if present:
# >
# >**A**- No age-related changes `(0)`
# >
# >**B** \
# >
# >**D** - Three age states of change `(1)` I'll assume it's a straight line with Beta, Gamma, Delta
# >
# >**G** /
# >
# > `Beta, Gamma, Delta` - Three experimental characteristics, their detailed characteristics are hidden under the symbols `A - M `
# 
# Respectively:
# 
# **Alpha** has an indicator A - predominantly; B | G | D - descending respectively
# 
# **Beta** has C index - mainly; B | A - descending respectively
# 
# **Gamma** has an indicator M - predominantly; N | H | B | A | F | G | E - descending respectively
# 
# **Delta** has an indicator B - predominantly; A | C | D - descending respectively

# ### Check the distribution of values of the target variable Class in the training set

# In[38]:


plt.figure(figsize=(10, 6))
plt.hist(train_data['Class'], bins=50, color='grey')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Distribution of Class')
plt.show()


# # Research part

# For the research part and creating new features, you do not need to create a copy of the training dataset, since we will be experimenting on merged_data. Then it will be possible to experiment with creating new features and adding them to the model training pipeline

# ### New features
# 
# Lists for easy tracking of new features

# In[39]:


new_num_features_list = []
new_cat_features_list = []


# In[40]:


# Selection of features for counting the number of values
feature_columns = merged_data.columns[1:57]

# Threshold value for determining "Popular Feature"
popularity_threshold = 300

# Create a new feature "Popular Feature"
merged_data['Popular Feature'] = merged_data[feature_columns].nunique(axis=1) > popularity_threshold

# Creation of a new feature "Type of age state"
age_state_columns = ['Alpha', 'Beta', 'Gamma', 'Delta']
merged_data['Age State'] = merged_data[age_state_columns].mode(axis=1)[0]

# Output updated dataframe
merged_data.head()


# As a result, new features have been created:
# 
# `"Popular Feature"` - contains the values False or True, which tells us if this feature meets the popularity criteria
# `"Age State"` - which will contain the number of unique types of age states that occur more often in this Id (relevant if 'Alpha', 'Beta', 'Gamma', 'Delta' use the same designations)

# In[41]:


if merged_data['Popular Feature'].dtype != 'object':
    new_num_features_list.append('Popular Feature')


# In[42]:


if merged_data['Age State'].dtype == 'object':
    new_cat_features_list.append('Age State')


# #### Target Encoding of categorical features

# In[43]:


def target_encode(df, feature_list, target, agg_func_list=['mean'], fill_na=0):
    '''
     Takes a dataframe and a feature_list and makes a new feature for it,
     using target encoding and a given aggregation function agg_func.
     Additionally stores the name of the new feature in new_num_features_list
     (this is only needed in this section: checking correlations, mutual info, etc.)
    '''
    
    for agg_func in agg_func_list:
        
        new_feature = '_'.join(feature_list) + '_TRG_' + agg_func
        df[new_feature] = df.groupby(feature_list)[target].transform(agg_func) #.fillna(fill_na)
        
        new_num_features_list.append(new_feature)
    
    return df


# In[44]:


def pair_cat_feature_target_mean(df):
    '''
    iterates over all categorical features in pairs;
    for each pair creates a new feature using mean target encoding
    '''
    cat_features = df.select_dtypes(exclude=np.number).columns.to_list()   # list of categorical features
    n_cat_features = len(cat_features)                                     # number of categorical features
    
    for i in range(n_cat_features):
        for j in range(i+1, n_cat_features):
            df = target_encode(
                               df,
                               feature_list=[cat_features[i], cat_features[j]],
                               target=TARGET,
                               agg_func_list=['mean'],
                              )
    return df


# In[45]:


merged_data = pair_cat_feature_target_mean(merged_data)


# In[46]:


merged_data.info()


# #### New feature for combination Class + Popular Feature + Age State
# 
# This will allow the model to take into account possible relationships and the influence of a combination of these factors on the presence of age-related changes that change the state

# In[47]:


def named_cat_feature_target_mean(df, feature_list):
    '''
    For the given list of categorical features, creates a new feature using mean target encoding
    '''
    df = target_encode(
                       df,
                       feature_list=feature_list,
                       target=TARGET,
                       agg_func_list=['mean']
                      )
    return df


# In[48]:


merged_data = pair_cat_feature_target_mean(merged_data)


# In[49]:


merged_data.info()


# In[50]:


merged_data.dropna(inplace=True)
merged_data.reset_index(drop=True, inplace=True)

merged_data.info()


# In[51]:


merged_data.to_csv('/kaggle/working/merged_data.csv', index=False)


# ### Perhaps the use of this file (merged_data) will be useful for an alternative research with additional features available in it that can expand the boundaries of the research

# In[52]:


# List of interval features

interval_features = ['AB', 'AF', 'AH', 'AM', 'AR', 'AX', 'AY', 'AZ', 'BC', 'BD', 'BN', 'BP', 'BQ', 'BR',
                     'BZ', 'CB', 'CC', 'CD', 'CF', 'CH', 'CL', 'CR', 'CS', 'CU', 'CW', 'DA', 'DE', 'DF', 
                     'DH', 'DI', 'DL', 'DN', 'DU', 'DV', 'DY','EB', 'EE', 'EG', 'EH', 'EL', 'EP', 'EU', 
                     'FC', 'FD', 'FE', 'FI', 'FL','FR', 'FS', 'GB', 'GE', 'GF', 'GH', 'GI', 'GL'] + new_num_features_list


# In[53]:


df = merged_data.phik_matrix(interval_cols=interval_features)

fig, ax = plt.subplots(figsize=(15, 0.4*df.shape[1]), dpi=100)
sns.heatmap(df[(0.3 < df) & (df < 1.0)], annot=False, cbar=False, linewidths=0.2, cmap='Blues')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()


# In[54]:


get_ipython().run_cell_magic('time', '', 'target_correlation_significance(merged_data, TARGET, interval_features)\n')


# Analyzing the correlation matrix, we can draw the following conclusions:
# 
# Some features have a positive correlation close to `1.0`, which indicates a strong relationship between them. For example, `Alpha_Gamma_TRG_mean`, `Id_Popular Feature_TRG_mean` and `Id_Age State_TRG_mean` have a positive correlation close to 1.0
# Some features have a negative correlation close to -1.0. For example, `FS` is negatively correlated, which may indicate an inverse relationship with other features
# Some features have a low correlation close to 0, which means there is no linear relationship between them. 

# In[55]:


merged_data = "/kaggle/working/merged_data.csv"


# The dataset **merged_data** might be useful for your future research, as it contains additional features, including combinations of those present in the standard dataset

# # Model

# In[56]:


del df


# # Data preparation
# 
# Feature extraction and target variable

# In[57]:


train_data.head(3)


# In[58]:


X = train_data.drop('Class', axis=1)
Y = train_data['Class']


# In[59]:


X.shape, Y.shape


# In[60]:


VALID_FRAC = 0.2

if VALID_FRAC > 0:
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=VALID_FRAC, random_state=RS)
else:
    X_train, X_valid, Y_train, Y_valid = X, X, Y, Y


# In[61]:


X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape


# In[62]:


# X_valid.head(2)


# In[63]:


class column_dropper_transformer(BaseEstimator, TransformerMixin):

    def __init__(self, drop_columns):
        self.drop_columns = drop_columns

    def fit(self, X, Y=None):
        return self
        
    def transform(self, X, Y=None):
        return X.drop(self.drop_columns, axis=1)


# In[64]:


# list of features to remove
drop_columns = ['Id']

column_dropper = column_dropper_transformer(drop_columns)


# # Preprocessing

# #### Categorical features

# In[65]:


cat_features = list(set(X_train.select_dtypes(exclude='number').columns.to_list()))
cat_features


# ### Selectors for numeric and categorical features

# In[66]:


num_selector = make_column_selector(dtype_include=np.number)
cat_selector = make_column_selector(dtype_exclude=np.number)


# ### Preprocessing of numeric and categorical features

# In[67]:


# Preprocessing of numerical features

num_preprocessor = make_pipeline(
                                 IterativeImputer(initial_strategy='mean', random_state=RS), 
                                 StandardScaler(),
                                )


# In[68]:


# for linear models
cat_preprocessor_linr = OneHotEncoder(drop='first', handle_unknown='error')

# for tree models
cat_preprocessor_tree = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=9999)


# #### Combining preprocessing of numeric and categorical features

# In[69]:


# for linear models
preprocessing_linr = make_column_transformer(
                                             (num_preprocessor, num_selector),
                                             (cat_preprocessor_linr, cat_selector),
                                             remainder='passthrough'
                                            )
# for tree models
preprocessing_tree = make_column_transformer(
                                             (num_preprocessor, num_selector),
                                             (cat_preprocessor_tree, cat_selector),
                                             remainder='passthrough'
                                            )
# # for CatBoost
preprocessing_catb = make_column_transformer(
                                             (num_preprocessor, num_selector),
                                             (cat_preprocessor_tree, cat_selector),
                                             remainder='passthrough'
                                            )


# preprocessing_tree = make_column_transformer(
#     (num_preprocessor, num_selector),
#     (cat_preprocessor_tree, cat_selector),
#     (new_features_transformer, ['Id']),  # Используем новый преобразователь
#     remainder='passthrough'
# )

# In[70]:


N_FEATURE_SELECT = 10 # was 10 changed to 50 and 25 and 15  - bad result


# In[71]:


feature_selector = SelectKBest(f_regression, k=N_FEATURE_SELECT)


# * The line `num_preprocessor = make_pipeline(IterativeImputer(initial_strategy='mean', random_state=RS), StandardScaler(),)` creates an instance of the Pipeline class that contains two transformers: `IterativeImpute`r and `StandardScaler`. These transformers are used to iteratively fill in missing numeric feature values using the 'mean' strategy and data standardization
# 
# 
# * Additionally, the code defines the transformers `cat_preprocessor_linr` and `cat_preprocessor_tree` for preprocessing categorical features depending on the model type. `cat_preprocessor_linr` uses the OHE (OneHotEncoder) centering method for linear models, and `cat_preprocessor_tree` uses the OrdinalEncoder method for tree-based models
# 
# 
# * As a result, the pipeline will combine all the transformers in the right order and allow you to process all the features at the same time to achieve the best result in the machine learning task

# # Pipeline table

# In[72]:


pipelines = [
    Pipeline([
        ('column_dropper', column_dropper),
        ('preproc_tree', preprocessing_tree),
        ('feature_selector', feature_selector),
        ('DC', DummyClassifier())
    ]),

     Pipeline([
        ('column_dropper', column_dropper),
        ('preproc_tree', preprocessing_tree),
        ('feature_selector', feature_selector),
        ('XGBC', XGBClassifier(random_state=RS))
    ]),

    Pipeline([
        ('column_dropper', column_dropper),
        ('preproc_tree', preprocessing_tree),
        ('feature_selector', feature_selector),
        ('RFC', RandomForestClassifier(random_state=RS))
    ]),

    Pipeline([
        ('column_dropper', column_dropper),
        ('preproc_tree', preprocessing_tree),
        ('feature_selector', feature_selector),
        ('LGBC', LGBMClassifier(random_state=RS))
    ])
]


names = [
    'DummyClassifier',
    'XGBClassifier',
    'RandomForestClassifier',
    'LGBMClassifier',
]

short_names = ['DC', 'XGBC', 'RFC', 'LGBC']

models = pd.DataFrame(
    data={'name': names,
          'short_name': short_names,
          'model': pipelines
    }
)
models


# In the model table, we leave only those algorithms that are specified in the ESTIMATOR_LIST list 

# In[73]:


for item in range(models.shape[0]):
    if models.loc[item,'name'] not in ESTIMATOR_LIST:
        models = models.drop(item, axis=0)
        
models = models.reset_index(drop=True)

models


# # Model selection

# In[74]:


X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape


# In[75]:


# Perform training and select the best model
best_log_loss = float('inf')
best_model = None

for item in range(models.shape[0]):
    model = models.loc[item, 'model']
    model.fit(X_train, Y_train)  # Train on scaled data

    Y_pred = model.predict(X_valid)
    logloss = log_loss(Y_valid, Y_pred)

    if logloss < best_log_loss:
        best_log_loss = logloss
        best_model = model

print("The smallest Log Loss of the algorithm:", best_model.steps[-1][-1])
print("Lowest Log Loss value:", best_log_loss)


# * The best model is RandomForestClassifier
# 
# * Best Log Loss : 2.906

# In[76]:


best_model


# In[77]:


train_predictions = best_model.predict(X_train)


# In[78]:


train_predictions


# # SUBMISSION

# My version 1

# In[79]:


# train_data.info()


# In[80]:


test_data.info()


# In[81]:


X_train.info()


# In[82]:


predictions_df = pd.DataFrame()


# In[83]:


# for _, model_row in models.iterrows():
#     model = model_row['model']
#     X_test_preprocessed = model.named_steps['preproc_tree'].transform(test_data)
    
#     predictions = model.named_steps[model_row['short_name']].predict(X_test_preprocessed)
#     predictions_df[model_row['short_name']] = predictions


# In[84]:


# final_predictions = predictions_df.mean(axis=1)


# In[85]:


# submission_df = pd.DataFrame({
#     'Id': test_data['Id'],  
#     'class_0': 1 - final_predictions,  
#     'class_1': final_predictions  
# })


# In[86]:


# submission_df.to_csv('/kaggle/working/submission.csv', index=False)

