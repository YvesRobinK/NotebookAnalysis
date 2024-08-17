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
import datetime 
import holidays
import dateutil.easter as easter
from scipy.signal import savgol_filter

# Import sklearn classes for model selection, cross validation, and performance evaluation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GroupKFold
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
import shap

# Import libraries for Hypertuning
import optuna

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
import xgboost as xgb
from xgboost.callback import EarlyStopping
import lightgbm as lgb
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


# Load Data
filepath = '/kaggle/input/playground-series-s3e19'

df_train = pd.read_csv(os.path.join(filepath, 'train.csv'), index_col=[0])
df_test = pd.read_csv(os.path.join(filepath, 'test.csv'), index_col=[0])

# Set columns
target_col = 'num_sold'
cat_cols = ['country', 'store', 'product']

# To Datetime
df_train['date'] = pd.to_datetime(df_train['date'])
df_test['date'] = pd.to_datetime(df_test['date'])

# Repalece "Using LLMs to"
df_train['product'] = df_train['product'].str.replace('Using LLMs to ', '')
df_test['product'] = df_test['product'].str.replace('Using LLMs to ', '')

print(f"train shape :{df_train.shape}, ", f"test shape :{df_test.shape}")


# In[3]:


df_train.head(5)


# In[4]:


df_train.info()


# In[5]:


df_test.info()


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">EDA</p>
# **Contents:**
# 1. Pie and bar charts for Target column features
# 2. Monthly, weekly, and daily trend
# 3. Sold Ratio by 'country', 'store', 'product'

# In[6]:


def plot_target_feature(df_train, target_col, figsize=(16,5), palette='colorblind', name='Train'):
    df_train = df_train.fillna('Nan')

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax = ax.flatten()

    # Get unique categories and their counts
    unique_categories, category_counts = np.unique(df_train[target_col], return_counts=True)

    # Pie chart
    pie_colors = sns.color_palette(palette, len(unique_categories))
    pie_explode = [0.05] * len(unique_categories)
    min_category_index = np.argmin(category_counts)
    pie_explode[min_category_index] = 0.2

    ax[0].pie(
        category_counts,
        shadow=True,
        explode=pie_explode,
        autopct='%1.f%%',
        textprops={'size': 15, 'color': 'white'},
        colors=pie_colors
    )
    ax[0].set_aspect('equal')
    ax[0].set_title('Distribution', fontsize=14)

    # Bar plot
    bar_colors = sns.color_palette(palette, len(unique_categories))
    bar_indices = np.argsort(category_counts)
    bar_data = category_counts[bar_indices]
    bar_labels = unique_categories[bar_indices]

    ax[1].barh(
        range(len(bar_labels)),  # Use range of indices as y-values
        bar_data,
        color=[bar_colors[i] for i in bar_indices]
    )
    ax[1].set_yticks(range(len(bar_labels)))  # Set y-ticks at the center of each bar
    ax[1].set_yticklabels(bar_labels)  # Set the correct category labels
    ax[1].set_xlabel('Count', fontsize=14)
    ax[1].set_ylabel('')
    ax[1].tick_params(labelsize=12)
    ax[1].yaxis.set_tick_params(width=0)
    ax[1].set_title('Count', fontsize=14)

    fig.suptitle(f'{target_col} in {name} Dataset', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

for cat_col in cat_cols:
    plot_target_feature(df_train[cat_cols], cat_col, figsize=(16,5), palette='colorblind', name='Train data')
for cat_col in cat_cols:
    plot_target_feature(df_test[cat_cols], cat_col, figsize=(16,5), palette='colorblind', name='Test data')


# In[7]:


def plot_daily_sales(df):
    plt.figure(figsize=(15, 6))
    df_grouped = df.groupby('date')['num_sold'].sum().reset_index()

    sns.set_style("whitegrid")
    sns.lineplot(data=df_grouped, x='date', y='num_sold', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Number of Items Sold')
    plt.title('Daily Sales')
    # plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()

plot_daily_sales(df_train)


# In[8]:


def plot_daily_sales_by_country(df_train, cat_col, name='Daily'):
    plt.figure(figsize=(15, 6))
    
    for cat in df_train[cat_col].unique():
        filt_train = df_train[df_train[cat_col] == cat]
        df_grouped = filt_train.groupby('date')['num_sold'].sum().reset_index()
        sns.lineplot(data=df_grouped, x='date', y='num_sold', label=cat, linewidth=2)
    
    plt.legend(loc='upper left')
    plt.xlabel('Date')
    plt.ylabel('Number of Items Sold')
    plt.title(f'{name} Sales by {cat_col}')
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_daily_sales_by_country(df_train, 'country')
plot_daily_sales_by_country(df_train, 'store')
plot_daily_sales_by_country(df_train, 'product')


# In[9]:


weekly_df = df_train.groupby(cat_cols+[pd.Grouper(key="date", freq="W")])["num_sold"].sum().rename("num_sold").reset_index()
plot_daily_sales_by_country(weekly_df, 'country', name='Weekly')
plot_daily_sales_by_country(weekly_df, 'store', name='Weekly')
plot_daily_sales_by_country(weekly_df, 'product', name='Weekly')
# del weekly_df


# In[10]:


monthly_df = df_train.groupby(cat_cols+[pd.Grouper(key="date", freq="MS")])["num_sold"].sum().rename("num_sold").reset_index()
plot_daily_sales_by_country(monthly_df, 'country', name='Monthly')
plot_daily_sales_by_country(monthly_df, 'store', name='Monthly')
plot_daily_sales_by_country(monthly_df, 'product', name='Monthly')
# del monthly_df


# In[11]:


store_weights = df_train.groupby("store")["num_sold"].sum()/df_train["num_sold"].sum()


# In[12]:


def create_barplot(data, x_col, y_col, hue_col):
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    
    if hue_col == None:
        sns.barplot(x=x_col, y=y_col, data=data, errorbar=None)
    else:
        colors = sns.color_palette("Set2", len(data[hue_col].unique()))
        sns.barplot(x=x_col, y=y_col, hue=hue_col, data=data, palette=colors, errorbar=None)

    plt.xlabel(x_col)
    plt.ylabel(y_col+' weight')
    plt.title('Number of Products Sold Weight')

    plt.xticks(rotation=45)
    plt.legend(title=hue_col, bbox_to_anchor=(1, 1))

    plt.tight_layout()
    sns.despine()

    plt.show()
    
product_store_weights = monthly_df.groupby(["product","store"])["num_sold"].sum() / monthly_df.groupby(["product"])["num_sold"].sum()
create_barplot(product_store_weights.reset_index(), 'product', "num_sold", 'store')
# product_store_weights.reset_index()


# In[13]:


store_weights = df_train.groupby("store")["num_sold"].sum()/df_train["num_sold"].sum()


# In[14]:


store_weights = df_train.groupby("store")["num_sold"].sum()/df_train["num_sold"].sum()
create_barplot(store_weights.reset_index(), 'store', "num_sold", None)
# store_weights.reset_index()


# In[15]:


product_country_weights = monthly_df.groupby(["product","country"])["num_sold"].sum() / monthly_df.groupby(["product"])["num_sold"].sum()


# In[16]:


# new_monthly_df = monthly_df.loc[monthly_df["date"] < "2020-01-01"]
product_country_weights = monthly_df.groupby(["product","country"])["num_sold"].sum() / monthly_df.groupby(["product"])["num_sold"].sum()
create_barplot(product_country_weights.reset_index(), 'product', "num_sold", 'country')
# product_country_weights.reset_index()


# In[17]:


product_df = df_train.groupby(["date","product"])["num_sold"].sum().reset_index()
product_ratio_df = product_df.pivot(index="date", columns="product", values="num_sold")
product_ratio_df = product_ratio_df.apply(lambda x: x/x.sum(),axis=1)
product_ratio_df = product_ratio_df.stack().rename("ratios").reset_index()


# In[18]:


def plot_daily_sales_by_country(df_train, y_col, cat_col, name='Daily'):
    plt.figure(figsize=(15, 6))
    
    for cat in df_train[cat_col].unique():
        filt_train = df_train[df_train[cat_col] == cat]
        df_grouped = filt_train.groupby('date')[y_col].sum().reset_index()
        sns.lineplot(data=df_grouped, x='date', y=y_col, label=cat, linewidth=2)
    
    plt.legend(loc='upper left')
    plt.xlabel('Date')
    plt.ylabel('Number of Items Sold Ratio')
    plt.title(f'{name} Sales Ratio by {cat_col}')
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
# plot_daily_sales_by_country(product_df, 'num_sold', 'product', name='Daily')
plot_daily_sales_by_country(product_ratio_df, 'ratios', 'product', name='Daily')


# In[19]:


df = df_train.copy()
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['year_month'] = df['month'].astype(str).str.zfill(2) + '_' + df['year'].astype(str) 
_df = pd.DataFrame()
for col in np.unique(df['year']):
    df[df['year'] == col]['num_sold']
    _df[f'{col}year_num_sold(Log)'] = np.log(df[df['year'] == col]['num_sold'].reset_index(drop=True))
    _df[f'country'] = df[df['year'] == col]['country'].reset_index(drop=True)
    _df[f'store'] = df[df['year'] == col]['store'].reset_index(drop=True)
    _df[f'product'] = df[df['year'] == col]['product'].reset_index(drop=True)


# In[20]:


def plot_distribution(df, hue, title='', drop_cols=[], n_cols=2):
    
    sns.set_style('whitegrid')

    cols = df.columns.drop([hue] + drop_cols)
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

plot_distribution(_df, hue='product', title='num_sold', drop_cols=['store', 'country'])
plot_distribution(_df, hue='country', title='num_sold', drop_cols=['store', 'product'])
plot_distribution(_df, hue='store', title='num_sold', drop_cols=['country', 'product'])


# In[21]:


_df = pd.DataFrame()
for col in np.unique(df['month']):
    df[df['month'] == col]['num_sold']
    _df[f'{col}month_num_sold(Log)'] = np.log(df[df['month'] == col]['num_sold'].reset_index(drop=True))
    _df[f'country'] = df[df['month'] == col]['country'].reset_index(drop=True)
    _df[f'store'] = df[df['month'] == col]['store'].reset_index(drop=True)
    _df[f'product'] = df[df['month'] == col]['product'].reset_index(drop=True)
    
# plot_distribution(_df, hue='product', title='Train data', drop_cols=['store', 'country'], n_cols=3)
plot_distribution(_df, hue='country', title='num_sold', drop_cols=['store', 'product'], n_cols=3)
plot_distribution(_df, hue='store', title='num_sold', drop_cols=['country', 'product'], n_cols=3)


# In[22]:


_df = pd.DataFrame()
for col in np.unique(df['year_month']):
    df[df['year_month'] == col]['num_sold']
    _df[f'{col}'] = np.log(df[df['year_month'] == col]['num_sold'].reset_index(drop=True))
    _df[f'country'] = df[df['year_month'] == col]['country'].reset_index(drop=True)
    _df[f'store'] = df[df['year_month'] == col]['store'].reset_index(drop=True)
    _df[f'product'] = df[df['year_month'] == col]['product'].reset_index(drop=True)
_df = _df[np.sort(_df.columns)]
    
def plot_boxplot(df, hue, drop_cols=[], n_cols=3, title=''):
    sns.set_style('whitegrid')

    cols = df.columns.drop([hue] + drop_cols)
    n_rows = (len(cols) - 1) // n_cols + 1

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 4*n_rows))

    for i, var_name in enumerate(cols):
        row = i // n_cols
        col = i % n_cols

        ax = axes[row, col]
        sns.boxplot(data=df, x=hue, y=var_name, ax=ax, showmeans=True, 
                    meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue", "markersize":"5"})
        ax.set_title(f'{var_name} by {hue}')
        ax.set_xlabel('')
        ax.set_ylim(0, 7) #

    fig.suptitle(f'{title} Boxplot by {hue}', fontweight='bold', fontsize=14, y=1.005)
    plt.tight_layout()
    plt.show()
    
plot_boxplot(_df, hue='country', drop_cols=['store', 'product'], n_cols=5, title='{Month}_{Year} num_sold')
# plot_boxplot(_df, hue='store', drop_cols=['country', 'product'], n_cols=4, title='{Year}_{month} num_sold')


# In[23]:


del _df , df


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Creating Data Sets</p>
# - The year 2021 was excluded due to the impact of Civid19.

# In[24]:


train = df_train.groupby(["date"])["num_sold"].sum().reset_index()
weekly_df = df_train.groupby([pd.Grouper(key="date", freq="W")])["num_sold"].sum().rename("num_sold").reset_index()
monthly_df = df_train.groupby([pd.Grouper(key="date", freq="MS")])["num_sold"].sum().rename("num_sold").reset_index()
# train_nocovid_df = train.loc[~((train["date"] >= "2020-03-01") & (train["date"] < "2020-06-01"))]
train_nocovid_df = train.loc[~((train["date"] >= "2020-01-01") & (train["date"] < "2021-01-01"))] # train.copy()
train_nocovid_df['date'].dt.year.unique()


# In[25]:


plt.figure(figsize=(15, 6))
sns.lineplot(data=train_nocovid_df, x="date", y="num_sold", label="No COVID", alpha=0.7, color='blue')
sns.lineplot(data=train, x="date", y="num_sold", label="Overall", alpha=0.4, color='red')

plt.xlabel('Date')
plt.ylabel('Number of Products Sold')
plt.title('Number of Products Sold Over Time')

plt.legend()
plt.tight_layout()
plt.show()


# In[26]:


train = train_nocovid_df.copy()

# get the dates to forecast for
test_all = df_test.sort_index().groupby(["date"]).first().reset_index()

# keep dates for Test
test = test_all[["date"]].copy()

# keep dates for Train
train_date = train[["date"]].copy()


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Feature Engineering</p>
# 
# - `feature_engineer()` : Create new timeseries features.
# - `get_holidays()` : Add holidays. Using the holidays library.
# -  Set `transform_log=True` if log-transforming the Target.
# 
# I am referring to the notebook here. 
# - https://www.kaggle.com/code/mikailduzenli/tps-2022-notebook#Forecasting
# - https://www.kaggle.com/code/juhjoo/4-41-tpg-sep-ridge-lasso-linear-elastic-ensemble?scriptVersionId=106534597
# 
# **Note: Feature engineering is still in progress. Not all of the above feature engineering is adapted in this kernel. Please take it as an idea.**

# In[27]:


def get_holidays(df):
    years_list = [2017, 2018, 2019, 2020, 2021, 2022]

    holiday_AR = holidays.CountryHoliday('AR', years = years_list)
    holiday_CA = holidays.CountryHoliday('CA', years = years_list)
    holiday_EE = holidays.CountryHoliday('EE', years = years_list)
    holiday_JP = holidays.CountryHoliday('JP', years = years_list)
    holiday_ES = holidays.CountryHoliday('ES', years = years_list)

    holiday_dict = holiday_AR.copy()
    holiday_dict.update(holiday_CA)
    holiday_dict.update(holiday_EE)
    holiday_dict.update(holiday_JP)
    holiday_dict.update(holiday_ES)

    df['holiday_name'] = df['date'].map(holiday_dict)
    df['is_holiday'] = np.where(df['holiday_name'].notnull(), 1, 0)
    df['holiday_name'] = df['holiday_name'].fillna('Not Holiday')
    df = df.drop(columns=["holiday_name"])
    
    return df

def periodic_spline_transformer(period, n_splines=None, degree=3):
    """
    Kaynak: https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html
    """
    from sklearn.preprocessing import SplineTransformer
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1  # periodic and include_bias is True
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True)

def seasonality_spline_features(hours=np.arange(1,32)):
    hour_df = pd.DataFrame(np.linspace(1, 32, 32).reshape(-1, 1),columns=["day"])
    splines = periodic_spline_transformer(32, n_splines=4).fit_transform(hour_df)
    splines_df = pd.DataFrame(splines,columns=[f"spline_{i}" for i in range(splines.shape[1])])
    splines_df =pd.concat([pd.Series(hours, name='day'), splines_df], axis="columns")
    
    return splines_df

def feature_engineer(df):
    new_df = df.copy()
    new_df["month"] = df["date"].dt.month
    new_df["month_sin"] = np.sin(new_df['month'] * np.pi / 24)
    new_df["month_cos"] = np.cos(new_df['month'] * np.pi / 24)
    # new_df['is_month_start'] = new_df.date.dt.is_month_start.astype(np.int8)
    # new_df['is_month_end'] = new_df.date.dt.is_month_end.astype(np.int8)
    
    new_df["day"] = df["date"].dt.day
    new_df['day_sin'] = np.sin(2*np.pi*new_df["day"] /24)
    new_df['day_cos'] = np.cos(2*np.pi*new_df["day"] /24)
    
    new_df["day_of_week"] = new_df["date"].dt.dayofweek
    new_df["day_of_week"] = new_df["day_of_week"].apply(lambda x: 0 if x<=3 else(1 if x==4 else (2 if x==5 else (3))))
    
    new_df['quarter'] = new_df['date'].dt.quarter
    new_df['day_of_month'] = new_df['date'].dt.day
    new_df['week_of_year'] = new_df['date'].dt.isocalendar().week.astype(int)
    
    new_df['friday'] = new_df.date.dt.weekday.eq(4).astype(np.uint8)
    new_df['saturday'] = new_df.date.dt.weekday.eq(5).astype(np.uint8)
    new_df['sunday'] = new_df.date.dt.weekday.eq(6).astype(np.uint8)
    
    new_df["day_of_year"] = df["date"].dt.dayofyear
    new_df["day_of_year"] = new_df.apply(lambda x: x["day_of_year"]-1 if (x["date"] > pd.Timestamp("2020-02-29") and x["date"] < pd.Timestamp("2021-01-01"))  else x["day_of_year"], axis=1)

    new_df["year"] = df["date"].dt.year - 2016
    new_df['is_year_end'] = new_df['date'].dt.is_year_end.astype(np.int8)
    new_df['is_year_start'] = new_df['date'].dt.is_year_start.astype(np.int8)
    
    important_dates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,16,17, 124, 125, 126, 127, 140, 141,142, 167, 168, 169, 170, 171, 173, 174, 175, 176, 177, 178, 179,
                 180, 181, 203, 230, 231, 232, 233, 234, 282, 289, 290, 307, 308, 309, 310, 311, 312, 313, 317, 318, 319, 320, 360, 361, 362, 363, 364, 365]
    new_df["important_dates"] = new_df["day_of_year"].apply(lambda x: x if x in important_dates else 0)
    
    easter_date = new_df.date.apply(lambda date: pd.Timestamp(easter.easter(date.year)))
    for day in list(range(-5, 5)) + list(range(40, 48)):
        new_df[f'easter_{day}'] = ((new_df.date - easter_date).dt.days.eq(day)).astype(int)
    easter_cols = [col for col in new_df.columns if 'easter_' in col]
    new_df['easter'] = new_df[easter_cols].sum(axis=1).astype(int)
    
    # for col in new_df.columns :
    #     if 'easter' in col :
    #         new_df = pd.get_dummies(new_df, columns = [col], drop_first=True)
            
    for day in range(24, 32):
        new_df[f'Dec_{day}'] = (new_df.date.dt.day.eq(day) & new_df.date.dt.month.eq(12)).astype(int)
    dec_cols = [col for col in new_df.columns if 'Dec_' in col]
    new_df['Dec'] = new_df[dec_cols].sum(axis=1).astype(int)
    
    new_df = get_holidays(new_df)
    splines_df = seasonality_spline_features()
    new_df = pd.merge(new_df, splines_df.dropna(axis=0), on='day', how='left')
    new_df = new_df.drop(columns=["date", "month", "day"]+easter_cols+dec_cols)
    # new_df = pd.get_dummies(new_df, columns = ["day_of_week"], drop_first=True)
    
    return new_df


# In[28]:


# Logarithmically transform
transform_log = False

# Applay Feature Engineering
train = feature_engineer(train)
test = feature_engineer(test)

if transform_log:
    train['num_sold'] = np.log(train['num_sold'])

X_train = train.drop(columns=["num_sold"]).reset_index(drop=True)
y_train = train["num_sold"].reset_index(drop=True)
X_test = test.reset_index(drop=True)

print(f"X_train shape :{X_train.shape} , y_train shape :{y_train.shape}")
print(f"X_test shape :{X_test.shape}")
print(f"X_train ->  isnull :{X_train.isnull().values.sum()}", f", isinf :{np.isinf(X_train).values.sum()}")
print(f"X_test -> isnull :{X_test.isnull().values.sum()}", f", isinf :{np.isinf(X_train).values.sum()}")
X_train.head(3)


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Data Splitting</p>
# This time, do a group Kfold for the year column and Time Series Split

# In[29]:


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
        elif self.kfold == 'group':
            for random_state in random_state_list:
                kf = GroupKFold(n_splits=self.n_splits)
                for train_index, val_index in kf.split(X, y, X['year']):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                    yield X_train, X_val, y_train, y_val, val_index
        elif self.kfold == 'tscv':
            for random_state in random_state_list:
                kf = TimeSeriesSplit(n_splits=self.n_splits)
                for train_index, val_index in kf.split(X, y, X['year']):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                    yield X_train, X_val, y_train, y_val, val_index
        elif self.kfold == 'tscv_2':
            for random_state in random_state_list:
                kf = TimeSeriesSplit(n_splits=self.n_splits, test_size=27375, max_train_size=82200)
                for train_index, val_index in kf.split(X, y):
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
# ~~LightGBM, CatBoost Xgboost and HistGradientBoosting hyper parameters are determined by optuna.~~  
# **Ensemble model setup and hyperparameters are in progress.**

# In[30]:


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
            'learning_rate': 0.171655957085321,
            'booster': 'gbtree',
            'lambda': 0.0856890877950814,
            'alpha': 0.000223555755009136,
            'subsample': 0.817828998442447,
            'colsample_bytree': 0.247566473896556,
            'max_depth': 9,
            'min_child_weight': 6,
            'eta': 0.0000146950106167549,
            'gamma': 0.0315067984172879,
            'grow_policy': 'depthwise',
            'n_jobs': -1,
            'objective': 'reg:squarederror', # reg:pseudohubererror
            'eval_metric': 'mape',
            'verbosity': 0,
            'random_state': self.random_state,
        }
        if self.device == 'gpu':
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['predictor'] = 'gpu_predictor'
        
        lgb_params = {
            'n_estimators': self.n_estimators,
            'learning_rate': 0.919954028302801,
            "reg_alpha": 0.00424724917570821,
            "reg_lambda": 1.02450316198987E-06,
            "num_leaves": 103,
            "colsample_bytree": 0.661659900143368,
            'subsample': 0.704065754123,
            'subsample_freq': 6,
            'min_child_samples': 100,
            #'objective': 'regression_l2',
            #'metric': 'mape',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state
        }
        cat_params = {
            'iterations': self.n_estimators,
            'depth': 2,
            'learning_rate': 0.0197025173696438,
            'l2_leaf_reg': 0.209291767991176,
            'random_strength': 0.0920413820919862,
            'od_type': 'IncToDec',
            'od_wait': 22,
            'bootstrap_type': 'Bayesian',
            'grow_policy': 'SymmetricTree',
            'bagging_temperature': 4.30830953881447,
            #'eval_metric': 'MAE',
            #'loss_function': 'MAE',
            'task_type': self.device.upper(),
            'verbose': False,
            'allow_writing_files': False,
            'random_state': self.random_state
        }
        xgb_params1 = {
            'n_estimators': self.n_estimators,
            'learning_rate': 0.01,
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
            'eval_metric': 'mape',
            'verbosity': 0,
            'random_state': self.random_state,
        }
        lgb_params1 = {
            'n_estimators': self.n_estimators,
            'learning_rate': 0.00659605502010782,
            "reg_alpha": 0.0134568843414818,
            "reg_lambda": 2.38367559632979E-06,
            "num_leaves": 31,
            "colsample_bytree": 0.850706320762174,
            'subsample': 0.691827302225948,
            'subsample_freq': 4,
            'min_child_samples': 33,
            'objective': 'regression_l2',
            'metric': 'mape',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state
        }
        cat_params1 = {
            'iterations': self.n_estimators,
            'depth': 6,
            'learning_rate': 0.00454306521731278,
            'l2_leaf_reg': 0.113774158297261,
            'random_strength': 0.0179641854849499,
            'od_type': 'IncToDec',
            'od_wait': 50,
            'bootstrap_type': 'Bayesian',
            'grow_policy': 'Lossguide',
            'bagging_temperature': 1.39240858193441,
            #'eval_metric': 'MAE',
            #'loss_function': 'MAE',
            'task_type': self.device.upper(),
            'verbose': False,
            'allow_writing_files': False,
            'random_state': self.random_state
        }
        hist_params = {
            'loss': 'absolute_error',
            'l2_regularization': 0.0122289470885951,
            'early_stopping': True,
            'learning_rate': 0.00100661954916865,
            'max_iter': 1000, # self.n_estimators
            'n_iter_no_change': 200,
            'max_depth': 27,
            'max_bins': 255,
            'min_samples_leaf': 98,
            'max_leaf_nodes':62,
            'random_state': self.random_state,
            #'categorical_features': []
        }
        gbd_params = {
            'loss': 'absolute_error',
            'n_estimators': 200,
            'max_depth': 10,
            'learning_rate': 0.01,
            'min_samples_split': 10,
            'min_samples_leaf': 20,
            'random_state': self.random_state,
        }
        
        models = {
            "lad": LADRegression(),
            "xgb": xgb.XGBRegressor(**xgb_params),
            "lgb": lgb.LGBMRegressor(**lgb_params),
            "cat": CatBoostRegressor(**cat_params),
#             "xgb1": xgb.XGBRegressor(**xgb_params1),
#             "lgb1": lgb.LGBMRegressor(**lgb_params1),
#             "cat1": CatBoostRegressor(**cat_params1),
            'hgb': HistGradientBoostingRegressor(**hist_params),
            "svr": SVR(kernel="rbf", gamma="auto"),
            #"SVR_linear": SVR(kernel="linear", gamma="auto"),
            "rid": RidgeCV(),
            "las": LassoCV(),
            "knn": KNeighborsRegressor(n_neighbors=15, n_jobs=-1),            
            "rfr": RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1),
            #"SGDRegressor": SGDRegressor(max_iter=2000, early_stopping=True, n_iter_no_change=100, random_state=self.random_state),
            "mlp": MLPRegressor(max_iter=1000, early_stopping=True, n_iter_no_change=50, random_state=self.random_state),
            "etr": ExtraTreesRegressor(n_estimators=500, n_jobs=-1, random_state=self.random_state),
            #"PLSRegression": PLSRegression(n_components=10, max_iter=1000),
            #"PassiveAggressiveRegressor": PassiveAggressiveRegressor(max_iter=1000, tol=1e-3, random_state=self.random_state),
            #"TheilSenRegressor": TheilSenRegressor(max_iter=1000, random_state=self.random_state, n_jobs=-1),
            #"gbd": GradientBoostingRegressor(**gbd_params),
            #"ARDRegression": ARDRegression(n_iter=1000),
            #"HuberRegressor": HuberRegressor(max_iter=2000)
        }
        
        return models


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Configuration</p>

# In[31]:


# Settings
kfold = 'tscv'
n_splits = 4 # Fix for group by Year
n_reapts = 1 # Fix for group by Year
random_state = 42
n_estimators = 3000 # 9999
early_stopping_rounds = 200
n_trials = 100 # 2000
verbose = False
device = 'cpu'

# Fix seed
random.seed(random_state)
random_state_list = random.sample(range(9999), n_reapts)

# metrics
def smape(y_true, y_pred):
    return 1 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)) * 100)
metric = smape
metric_name = metric.__name__.upper()

# To calculate runtime
def sec_to_minsec(t):
    min_ = int(t / 60)
    sec = int(t - min_*60)
    return min_, sec

# Process


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">One Model Xgboost</p>
# 
# The xgboost model architecture is based on the following: 
# [[PS S3E14, 2023] EDA and Submission](https://www.kaggle.com/code/sergiosaharovskiy/ps-s3e14-2023-eda-and-submission)

# In[32]:


def plot_training_process(lossfunc_key, eval_results_, best_iters_, early_stopping_rounds):

    metric_score_folds = pd.DataFrame.from_dict(eval_results_).T
    fit_rmsle = metric_score_folds.fit.apply(lambda x: x[lossfunc_key])
    val_rmsle = metric_score_folds.val.apply(lambda x: x[lossfunc_key])

    n_splits = len(metric_score_folds)
    n_rows = math.ceil(n_splits / 3)

    fig, axes = plt.subplots(n_rows, 3, figsize=(20, n_rows * 4), dpi=150)
    ax = axes.flatten()

    for i, (f, v, best_iter) in enumerate(zip(fit_rmsle, val_rmsle, best_iters_)): 
        sns.lineplot(f, color='#B90000', ax=ax[i], label='fit')
        sns.lineplot(v, color='#048BA8', ax=ax[i], label='val')
        ax[i].legend()
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].set_title(f'Fold {i}', fontdict={'fontweight': 'bold'})

        color = ['#048BA8', '#90A6B1']
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
        ax[i].set_ylabel(f'{lossfunc_key}', fontsize=12)
        ax[i].legend(loc='upper right', title=lossfunc_key)

    for j in range(i+1, n_rows * 3):
        ax[j].axis('off')

    plt.tight_layout()
    plt.show()
    

def plot_feature_importance(fi):
    fi_gain = fi[[col for col in fi.columns if col.startswith('gain')]].mean(axis=1)
    fi_splt = fi[[col for col in fi.columns if col.startswith('split')]].mean(axis=1)
    
    fig, ax = plt.subplots(1, 2, figsize=(18, 6), dpi=150)

    # Split fi.
    data_splt = fi_splt.sort_values(ascending=False)
    data_splt = data_splt.head(20)
    sns.barplot(x=data_splt.values, y=data_splt.index,
                color='#1E90FF', linewidth=0.5, edgecolor="black", ax=ax[0])
    ax[0].set_title(f'Feature Importance "Split"', fontdict={'fontweight': 'bold'})
    ax[0].set_xlabel("Importance", fontsize=12)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)

    # Gain fi.
    data_gain = fi_gain.sort_values(ascending=False)
    data_gain = data_gain.head(20)
    sns.barplot(x=data_gain.values, y=data_gain.index,
                color='#4169E1', linewidth=0.5, edgecolor="black", ax=ax[1])
    ax[1].set_title(f'Feature Importance "Gain"', fontdict={'fontweight': 'bold'})
    ax[1].set_xlabel("Importance", fontsize=12)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)

    plt.tight_layout()
    plt.show()
    
def plot_true_oof(df, title=''):
    plt.figure(figsize=(15, 6))
    sns.lineplot(data=df, x="date", y="true", label="True", alpha=0.7, color='blue')
    sns.lineplot(data=df, x="date", y="oof", label="Out of Sample", alpha=0.4, color='red')

    plt.xlabel('Date')
    plt.ylabel('Number of Products Sold')
    plt.title(f'Number of Products Sold: True vs Out-of-fold {title}', fontweight='bold', fontsize=10)

    plt.legend(loc='upper right')
    plt.tight_layout()
    sns.despine()  # グラフの枠線を削除

    plt.show()


# In[33]:


xgb_params = {
    'n_estimators': 1000,
#     'learning_rate': 0.7,
    'booster': 'gbtree',
    'lambda': 0.0856890877950814,
    'alpha': 0.000223555755009136,
    'subsample': 0.817828998442447,
    'colsample_bytree': 0.247566473896556,
    'max_depth': 9,
    'min_child_weight': 6,
    'eta': 0.0222221,
    'gamma': 0.0315067984172879,
    'grow_policy': 'depthwise',
    'n_jobs': -1,
    'objective': 'reg:squarederror',
    'eval_metric': 'mape',
    'verbosity': 0,
    'random_state': 42,
}

feature_importances_ = pd.DataFrame(index=X_train.columns)
eval_results_ = {}
best_iters_ = []
oof = np.zeros((X_train.shape[0]))
test_preds = np.zeros((X_test.shape[0]))
val_scores = []

splitter = Splitter(kfold=kfold, n_splits=n_splits)
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
    # clf = Regressor(n_estimators_, device)
    # xgb_params = clf.models['xgb'].get_params()
    # xgb_params = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.01).get_params()

    eval_results_[fold] = {}
    model = xgb.train(
        num_boost_round=xgb_params['n_estimators'],
        params=xgb_params,
        dtrain=fit_set,
        evals=watchlist,
        evals_result=eval_results_[fold],
        verbose_eval=False,
        callbacks=[xgb.callback.EarlyStopping(early_stopping_rounds, data_name='val', save_best=True)])

    val_preds = model.predict(val_set)
    test_preds += model.predict(xgb.DMatrix(X_test)) / n_splits

    oof[val_index] = val_preds

    val_score = metric(y_val, val_preds)
    best_iter = model.best_iteration
    best_iters_.append(best_iter)
    val_scores.append(val_score)
    print(f'Fold: {blu}{fold:>3}{res}| {metric_name}: {blu}{val_score:.5f}{res}' f' | Best iteration: {blu}{best_iter:>4}{res}')

    # Stores the feature importances
    feature_importances_[f'gain_{fold}'] = feature_importances_.index.map(model.get_score(importance_type='gain'))
    feature_importances_[f'split_{fold}'] = feature_importances_.index.map(model.get_score(importance_type='weight'))

mean_cv_score_full = metric(y_train, oof)
# print(f'{"*" * 50}\n{red}Mean full{res} {metric_name} : {red}{mean_cv_score_full:.5f}{res}')
print(f'{red}Mean val{res} {metric_name}  : {red}{np.mean(val_scores):.5f}{res}')

plot_training_process('mape', eval_results_, best_iters_, early_stopping_rounds)
plot_feature_importance(feature_importances_)

df = pd.DataFrame([y_train.values, oof], index=['true', 'oof']).T
df = df[df['oof'] > 0].reset_index(drop=True)
df['date'] = train_date[len(train_date) - len(df):].reset_index(drop=True)['date']
test_ = test_all[['date']].copy()
test_['oof'] = test_preds
df = pd.concat([df, test_]).reset_index(drop=True)
if transform_log:
    df['true'], df['oof'] = np.exp(df['true']), np.exp(df['oof'])
df = df[df['date'] > datetime.datetime(2018, 1, 1)]
plot_true_oof(df)

# _df = df[df['date'] > datetime.datetime(2021, 1, 1)]
# incremental_weight = (_df['true'] / _df['oof']).mean()
# print('incremental_weight(2021>=):', incremental_weight)
# df['oof'] = df['oof'] * (incremental_weight)
# plot_true_oof(df, title='(incremental_weight)')


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Neural Network by skorch</p>
# **A scikit-learn compatible neural network library that wraps PyTorch.**
# 
# ![](https://cpp-learning.com/wp-content/uploads/2020/03/skorch_logo.jpg)
# 
# https://skorch.readthedocs.io/en/stable/#

# In[34]:


get_ipython().system('pip install skorch')


# In[35]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import skorch
from skorch import NeuralNetClassifier, NeuralNetRegressor, NeuralNet
from skorch.callbacks import Callback, Checkpoint, EarlyStopping
from skorch.callbacks import LRScheduler
from skorch.dataset import ValidSplit
from skorch.helper import predefined_split


# In[36]:


class TimeseriesDataset(Dataset):
    """Onchain dataset."""

    def __init__(self, data, seq_length, features, target):
        self.data = data
        self.target = target
        self.features = features
        self.seq_length = seq_length
        self.data_length = len(data)

        self.metrics = self.create_xy_pairs()

    def create_xy_pairs(self):
        pairs = []
        for idx in range(self.data_length - self.seq_length):
            x = self.data[idx:idx + self.seq_length][self.features].values
            y = self.data[idx + self.seq_length:idx + self.seq_length + 1][self.target].values
            
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
            
            pairs.append((x, y))
        return pairs

    def __len__(self):
        return len(self.metrics)

    def __getitem__(self, idx):
        return self.metrics[idx]

patience = 5
max_epochs = 200
seq_length = 2

# set the random seed 
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Define concatdata for Dataset
if transform_log == False:
    data = pd.concat([X_train, np.log(y_train)], axis=1).copy()
else:
    data = pd.concat([X_train, y_train], axis=1).copy()
feat = list(X_train.columns)
# dammy = pd.DataFrame(data=np.zeros(X_test.shape[0]), columns=[target_col])
# dammy_test = pd.concat([X_test, dammy], axis=1)
# test_dataset = TimeseriesDataset(dammy_test, seq_length, feat, target_col)

scaler = StandardScaler() # StandardScaler or MinMaxScaler
data[feat] = scaler.fit_transform(data[feat])
dammy = pd.DataFrame(data=np.zeros(X_test.shape[0]), columns=[target_col])
dammy_test = pd.concat([X_test, dammy], axis=1)
dammy_test[feat] = scaler.transform(dammy_test[feat])
test_dataset = TimeseriesDataset(dammy_test, seq_length, feat, target_col)

# Transfer to accelerator
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


# ### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">LSTM (Long Short-Term Memory)</p>

# In[37]:


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob, device, directions=1):
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.directions = directions

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(hidden_size, output_size)

    def init_hidden_states(self, batch_size):
        # Initialize hidden and cell states with dimension:
        # (num_layers * num_directions, batch, hidden_size)
        state_dim = (self.num_layers * self.directions, batch_size, self.hidden_size)
        return (torch.zeros(state_dim).to(self.device), torch.zeros(state_dim).to(self.device))

    def forward(self, x):
        states = self.init_hidden_states(x.size(0))
        x, (h, c) = self.lstm(x, states)
        out = self.linear(x)
        out = out[:, -1, :]
        return out


# In[38]:


eval_results_ = {}
best_iters_ = []
oof = np.zeros((X_train.shape[0]))
test_preds = np.zeros((X_test.shape[0] - seq_length)) # X_test.shape[0]
val_scores = []

splitter = Splitter(kfold=kfold, n_splits=n_splits)
for i, (X_train_, X_val, y_train_, y_val, val_index) in enumerate(splitter.split_data(X_train, y_train, random_state_list=[0])):
    fold = i % n_splits
    m = i // n_splits
    
    train_dataset = TimeseriesDataset(data.iloc[X_train_.index], seq_length, feat, target_col)
    X_val = TimeseriesDataset(data.iloc[val_index], seq_length, feat, target_col)
    y_val = np.array([y.numpy() for x, y in iter(X_val)]).reshape(-1)
    
    early_stopping = skorch.callbacks.EarlyStopping(
        monitor='valid_loss',
        patience=patience,
        threshold=0.001,
        threshold_mode='rel',
        lower_is_better=True,
        load_best=True
    )
    model = NeuralNet(
        module=LSTM,
        batch_size=32,
        max_epochs=max_epochs,
        module__input_size=len(feat),
        module__hidden_size=30,
        module__num_layers=1,
        module__output_size=1,
        module__dropout_prob=0.50,
        module__device=device,
        criterion=nn.L1Loss, # nn.MSELoss() nn.L1Loss
        optimizer=optim.AdamW,
        lr=1e-3,
        callbacks=[
            ('lr_scheduler',
             LRScheduler(policy=ReduceLROnPlateau,
                         mode='min',  # or 'max' depending on your task
                         factor=0.1,
                         patience=10,
                         verbose=True)),
            early_stopping
        ],
        device=device,
        train_split=predefined_split(X_val),
        verbose=0,
    )
    
    eval_results_[i] = {}
    model.fit(train_dataset)
    
    eval_results_[i]['fit'] = {'loss': model.history[:, 'train_loss']}
    eval_results_[i]['val'] = {'loss': model.history[:, 'valid_loss']}
    
    val_preds = model.predict(X_val).reshape(-1)
    test_preds = model.predict(test_dataset).reshape(-1) #/ n_splits

    oof[val_index[seq_length:]] = val_preds

    val_score = metric(y_val, val_preds)
    best_iter = early_stopping.best_epoch_
    best_iters_.append(best_iter)
    val_scores.append(val_score)
    
    print(f'Fold: {blu}{i:>3}{res} | {metric_name}: {blu}{val_score:.5f}{res} | Best iteration: {blu}{best_iter:>4}{res}')
    
lstm_test_preds = test_preds.copy()

# mean_cv_score_full = metric(y_train, oof)
# print(f'{"*" * 50}\n{red}Mean{res} {metric_name} : {red}{mean_cv_score_full:.5f}')
print(f'{red}Mean val{res} {metric_name}  : {red}{np.mean(val_scores):.5f}{res}')

plot_training_process('loss', eval_results_, best_iters_, patience)
if transform_log == False:
    df = pd.DataFrame([np.log(y_train.values), np.clip(oof, 8.5, 11)], index=['true', 'oof']).T
else:
    df = pd.DataFrame([y_train.values, np.clip(oof, 8.5, 11)], index=['true', 'oof']).T
df = df[df['oof'] > 0].reset_index(drop=True)
df['date'] = train_date[len(train_date) - len(df):].reset_index(drop=True)['date']
test_ = test_all[['date']].iloc[seq_length:].copy()
test_['oof'] = test_preds
df = pd.concat([df, test_]).reset_index(drop=True)
if transform_log:
    df['true'], df['oof'] = np.exp(df['true']), np.exp(df['oof'])
df = df[df['date'] > datetime.datetime(2018, 1, 1)]
plot_true_oof(df)

# _df = df[df['date'] > datetime.datetime(2021, 1, 1)]
# incremental_weight = (_df['true'] / _df['oof']).mean()
# print('incremental_weight(2021>=):', incremental_weight)
# df['oof'] = df['oof'] * (incremental_weight)
# plot_true_oof(df, title='(incremental_weight)')


# ### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">GRU（Gated Recurrent Unit）</p>

# In[39]:


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob, device, directions=1):
        super(GRU, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.directions = directions

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(hidden_size, output_size)

    def init_hidden_states(self, batch_size):
        # Initialize hidden and cell states with dimension:
        # (num_layers * num_directions, batch, hidden_size)
        state_dim = (self.num_layers * self.directions, batch_size, self.hidden_size)
        return torch.zeros(state_dim).to(self.device)

    def forward(self, x):
        h = self.init_hidden_states(x.size(0))
        x, h = self.gru(x, h)
        out = self.linear(x)
        out = out[:, -1, :]
        return out


# In[40]:


eval_results_ = {}
best_iters_ = []
oof = np.zeros((X_train.shape[0]))
test_preds = np.zeros((X_test.shape[0] - seq_length)) # X_test.shape[0]
val_scores = []

splitter = Splitter(kfold=kfold, n_splits=n_splits)
for i, (X_train_, X_val, y_train_, y_val, val_index) in enumerate(splitter.split_data(X_train, y_train, random_state_list=[0])):
    fold = i % n_splits
    m = i // n_splits
    
    train_dataset = TimeseriesDataset(data.iloc[X_train_.index], seq_length, feat, target_col)
    X_val = TimeseriesDataset(data.iloc[val_index], seq_length, feat, target_col)
    y_val = np.array([y.numpy() for x, y in iter(X_val)]).reshape(-1)
    
    early_stopping = skorch.callbacks.EarlyStopping(
        monitor='valid_loss',
        patience=patience,
        threshold=0.001,
        threshold_mode='rel',
        lower_is_better=True,
        load_best=True
    )
    model = NeuralNet(
        module=GRU,
        batch_size=32,
        max_epochs=max_epochs,
        module__input_size=len(feat),
        module__hidden_size=20,
        module__num_layers=1,
        module__output_size=1,
        module__dropout_prob=0.50,
        module__device=device,
        criterion=nn.L1Loss, # nn.MSELoss() nn.L1Loss
        optimizer=optim.AdamW,
        lr=1e-3,
        callbacks=[
            ('lr_scheduler',
             LRScheduler(policy=ReduceLROnPlateau,
                         mode='min',  # or 'max' depending on your task
                         factor=0.1,
                         patience=10,
                         verbose=True)),
            early_stopping
        ],
        device=device,
        train_split=predefined_split(X_val),
        verbose=0,
    )
    
    eval_results_[i] = {}
    model.fit(train_dataset)
    
    eval_results_[i]['fit'] = {'loss': model.history[:, 'train_loss']}
    eval_results_[i]['val'] = {'loss': model.history[:, 'valid_loss']}
    
    val_preds = model.predict(X_val).reshape(-1)
    test_preds = model.predict(test_dataset).reshape(-1) #/ n_splits

    oof[val_index[seq_length:]] = val_preds

    val_score = metric(y_val, val_preds)
    best_iter = early_stopping.best_epoch_
    best_iters_.append(best_iter)
    val_scores.append(val_score)
    
    print(f'Fold: {blu}{i:>3}{res} | {metric_name}: {blu}{val_score:.5f}{res} | Best iteration: {blu}{best_iter:>4}{res}')
    
gru_test_preds = test_preds.copy()

# mean_cv_score_full = metric(y_train, oof)
# print(f'{"*" * 50}\n{red}Mean{res} {metric_name} : {red}{mean_cv_score_full:.5f}')
print(f'{red}Mean val{res} {metric_name}  : {red}{np.mean(val_scores):.5f}{res}')

plot_training_process('loss', eval_results_, best_iters_, patience)
if transform_log == False:
    df = pd.DataFrame([np.log(y_train.values), np.clip(oof, 8.5, 11)], index=['true', 'oof']).T
else:
    df = pd.DataFrame([y_train.values, np.clip(oof, 8.5, 11)], index=['true', 'oof']).T
df = df[df['oof'] > 0].reset_index(drop=True)
df['date'] = train_date[len(train_date) - len(df):].reset_index(drop=True)['date']
test_ = test_all[['date']].iloc[seq_length:].copy()
test_['oof'] = test_preds
df = pd.concat([df, test_]).reset_index(drop=True)
if transform_log:
    df['true'], df['oof'] = np.exp(df['true']), np.exp(df['oof'])
df = df[df['date'] > datetime.datetime(2018, 1, 1)]
plot_true_oof(df)

# _df = df[df['date'] > datetime.datetime(2021, 1, 1)]
# incremental_weight = (_df['true'] / _df['oof']).mean()
# print('incremental_weight(2021>=):', incremental_weight)
# df['oof'] = df['oof'] * (incremental_weight)
# plot_true_oof(df, title='(incremental_weight)')


# ### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">extra: CNN</p>
# FYI CNN, you are not predicting well. Maybe bad tuning.

# In[41]:


class CNN(nn.Module):
    def __init__(self, in_channels=6):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=1),

            nn.Conv1d(16, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=1),

            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=1),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=1),

            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=1),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256*2, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1) # (batch_size, feature_dim, seq_len)
        x = self.conv_layers(x)
        x = x.view(-1, 256*2)
        x = self.fc_layers(x)
        return x


# In[42]:


patience = 5
max_epochs = 200
seq_length = 10

# Define concatdata for Dataset
data = pd.concat([X_train, y_train], axis=1).copy()
feat = list(X_train.columns)
# dammy = pd.DataFrame(data=np.zeros(X_test.shape[0]), columns=[target_col])
# dammy_test = pd.concat([X_test, dammy], axis=1)
# test_dataset = TimeseriesDataset(dammy_test, seq_length, feat, target_col)

scaler = StandardScaler() # StandardScaler or MinMaxScaler
data[feat] = scaler.fit_transform(data[feat])
dammy = pd.DataFrame(data=np.zeros(X_test.shape[0]), columns=[target_col])
dammy_test = pd.concat([X_test, dammy], axis=1)
dammy_test[feat] = scaler.transform(dammy_test[feat])
test_dataset = TimeseriesDataset(dammy_test, seq_length, feat, target_col)


eval_results_ = {}
best_iters_ = []
oof = np.zeros((X_train.shape[0]))
test_preds = np.zeros((X_test.shape[0] - seq_length)) # X_test.shape[0]
val_scores = []

splitter = Splitter(kfold=kfold, n_splits=n_splits)
for i, (X_train_, X_val, y_train_, y_val, val_index) in enumerate(splitter.split_data(X_train, y_train, random_state_list=[0])):
    fold = i % n_splits
    m = i // n_splits
    
    train_dataset = TimeseriesDataset(data.iloc[X_train_.index], seq_length, feat, target_col)
    X_val = TimeseriesDataset(data.iloc[val_index], seq_length, feat, target_col)
    y_val = np.array([y.numpy() for x, y in iter(X_val)]).reshape(-1)
    
    early_stopping = skorch.callbacks.EarlyStopping(
        monitor='valid_loss',
        patience=patience,
        threshold=0.001,
        threshold_mode='rel',
        lower_is_better=True,
        load_best=True
    )
    model = NeuralNet(
        module=CNN,
        batch_size=64,
        max_epochs=max_epochs,
        module__in_channels=len(feat),
        criterion=nn.L1Loss, # nn.MSELoss() nn.L1Loss
        optimizer=optim.AdamW,
        lr=1e-2,
        callbacks=[
            ('lr_scheduler',
             LRScheduler(policy=ReduceLROnPlateau,
                         mode='min',  # or 'max' depending on your task
                         factor=0.1,
                         patience=10,
                         verbose=True)),
            early_stopping
        ],
        device=device,
        train_split=predefined_split(X_val),
        verbose=0,
    )
    
    eval_results_[i] = {}
    model.fit(train_dataset)
    
    eval_results_[i]['fit'] = {'loss': model.history[:, 'train_loss']}
    eval_results_[i]['val'] = {'loss': model.history[:, 'valid_loss']}
    
    val_preds = model.predict(X_val).reshape(-1)
    test_preds = model.predict(test_dataset).reshape(-1) #/ n_splits

    oof[val_index[seq_length:]] = val_preds

    val_score = metric(y_val, val_preds)
    best_iter = early_stopping.best_epoch_
    best_iters_.append(best_iter)
    val_scores.append(val_score)
    
    print(f'Fold: {blu}{i:>3}{res} | {metric_name}: {blu}{val_score:.5f}{res} | Best iteration: {blu}{best_iter:>4}{res}')
    
# mean_cv_score_full = metric(y_train, oof)
# print(f'{"*" * 50}\n{red}Mean{res} {metric_name} : {red}{mean_cv_score_full:.5f}')
print(f'{red}Mean val{res} {metric_name}  : {red}{np.mean(val_scores):.5f}{res}')


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Weighted Ensemble Model by Optuna on Training</p>
# A weighted average is performed during training;  
# The weights were determined for each model using the predictions for the train data created in the out of fold with Optuna's CMAsampler. (Here it is defined by `OptunaWeights`)  
# This is an extension of the averaging method. All models are assigned different weights defining the importance of each model for prediction.
# 
# ![](https://www.analyticsvidhya.com/wp-content/uploads/2015/08/Screen-Shot-2015-08-22-at-6.40.37-pm.png)

# In[43]:


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


# In[44]:


get_ipython().run_cell_magic('time', '', '\ndevice = "gpu" if use_cuda else "cpu"\n\n# Initialize an array for storing test predictions\nclf = Regressor(n_estimators, device, random_state)\ntest_predss = np.zeros((X_test.shape[0]))\noof_predss = np.zeros((X_train.shape[0], n_reapts))\nensemble_score, ensemble_score_ = [], []\nweights = []\ntrained_models = dict(zip([_ for _ in clf.models_name if (\'xgb\' in _) or (\'lgb\' in _) or (\'cat\' in _)], [[] for _ in range(clf.len_models)]))\nscore_dict = dict(zip(clf.models_name, [[] for _ in range(clf.len_models)]))\n\nsplitter = Splitter(kfold=kfold, n_splits=n_splits, cat_df=y_train)\nfor i, (X_train_, X_val, y_train_, y_val, val_index) in enumerate(splitter.split_data(X_train, y_train, random_state_list=random_state_list)):\n    n = i % n_splits\n    m = i // n_splits\n\n    # Get a set of Regressor models\n    clf = Regressor(n_estimators, device, random_state_list[m])\n    models = clf.models\n\n    # Initialize lists to store oof and test predictions for each base model\n    oof_preds = []\n    test_preds = []\n\n    # Loop over each base model and fit it to the training data, evaluate on validation data, and store predictions\n    for name, model in models.items():\n        best_iteration = None\n        start_time = time.time()\n\n        if (\'xgb\' in name) or (\'lgb\' in name) or (\'cat\' in name):\n            early_stopping_rounds_ = int(early_stopping_rounds*2) if (\'lgb\' in name) else early_stopping_rounds\n\n            #if \'lgb\' in name:\n            #    model.fit(\n            #        X_train_, y_train_, eval_set=[(X_val, y_val)], categorical_feature=cat_cols,\n            #        early_stopping_rounds=early_stopping_rounds_, verbose=verbose)\n            #elif \'cat\' in name :\n            #    model.fit(\n            #        Pool(X_train_, y_train_, cat_features=cat_cols), eval_set=Pool(X_val, y_val, cat_features=cat_cols),\n            #        early_stopping_rounds=early_stopping_rounds_, verbose=verbose)\n            #else:\n            \n            model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds_, verbose=verbose)\n\n            best_iteration = model.best_iteration if (\'xgb\' in name) else model.best_iteration_\n        else:\n            model.fit(X_train_, y_train_)\n\n        end_time = time.time()\n        min_, sec = sec_to_minsec(end_time - start_time)\n\n        if name in trained_models.keys():\n            trained_models[f\'{name}\'].append(deepcopy(model))\n\n        y_val_pred = model.predict(X_val).reshape(-1)\n        test_pred = model.predict(X_test).reshape(-1)\n\n        score = metric(y_val, y_val_pred)\n        score_dict[name].append(score)\n        print(f\'{blu}{name}{res} [FOLD-{n} SEED-{random_state_list[m]}] {metric_name} {blu}{score:.5f}{res} | Best iteration {blu}{best_iteration}{res} | Runtime {min_}min {sec}s\')\n\n        oof_preds.append(y_val_pred)\n        test_preds.append(test_pred)\n\n    # Use Optuna to find the best ensemble weights\n    optweights = OptunaWeights(random_state=random_state_list[m], n_trials=n_trials)\n    y_val_pred = optweights.fit_predict(y_val.values, oof_preds)\n\n    score = metric(y_val, y_val_pred)\n    print(f\'{red}>>> Ensemble{res} [FOLD-{n} SEED-{random_state_list[m]}] {metric_name} {red}{score:.5f}{res}\')\n    print(f\'{"-" * 60}\')\n    ensemble_score.append(score)\n    weights.append(optweights.weights)\n\n    # Predict to X_test by the best ensemble weights\n    # test_predss += optweights.predict(test_preds) / (n_splits * len(random_state_list))\n    test_predss = optweights.predict(test_preds) #/ (n_splits * len(random_state_list))\n    oof_predss[X_val.index, m] += optweights.predict(oof_preds)\n\n    gc.collect()\n')


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Mean Scores for each model</p>

# In[45]:


def plot_score_from_dict(score_dict, title='', ascending=True):
    score_df = pd.melt(pd.DataFrame(score_dict))
    score_df = score_df.sort_values('value', ascending=ascending)
    
    plt.figure(figsize=(12, 6))
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

# In[46]:


from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

def plot_heatmap_with_dendrogram(df, title, figsize=(18, 10), fontsize=10):
    mask = np.zeros_like(df.astype(float).corr())
    mask[np.triu_indices_from(mask)] = True

    colormap = plt.cm.RdBu_r
    fig, ax = plt.subplots(2, 1, figsize=figsize)

    # Plot heatmap
    ax[0].set_title(f'{title} Correlation of Features', fontweight='bold', y=1.02, size=15)
    sns.heatmap(df.astype(float).corr(), linewidths=0.1, vmax=1.0, vmin=-1.0,
                square=True, cmap=colormap, linecolor='white', annot=True,
                annot_kws={"size": fontsize, "weight": "bold"}, mask=mask, ax=ax[0], cbar=False)

    # Plot dendrogram
    correlations = df.corr()
    converted_corr = 1 - np.abs(correlations)
    Z = linkage(squareform(converted_corr), 'complete')
    
    dn = dendrogram(Z, labels=df.columns, above_threshold_color='#ff0000', ax=ax[1])
    ax[1].set_title(f'{title} Hierarchical Clustering Dendrogram', fontsize=15, fontweight='bold')
    ax[1].grid(axis='x')
    ax[1].tick_params(axis='x', rotation=90)
    ax[1].tick_params(axis='y', labelsize=fontsize)

    plt.tight_layout()
    plt.show()


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

# Plot Optuna Weights
normalize = [((weight - np.min(weight)) / (np.max(weight) - np.min(weight))).tolist() for weight in weights]
weight_dict = dict(zip(list(score_dict.keys()), np.array(normalize).T.tolist()))
plot_score_from_dict(weight_dict, title='Optuna Weights (Normalize 0 to 1)', ascending=False)

# Plot oof_predict analyis for each model
plot_heatmap_with_dendrogram(pd.DataFrame(oof_preds, index=list(score_dict.keys())).T, title='OOF Predict', figsize=(10, 10), fontsize=8)


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">OOF analysis of the Optuna Ensemble</p>

# In[47]:


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
    
df = pd.DataFrame([y_train.values, np.mean(oof_predss, axis=1)], index=['true', 'oof']).T
df = df[df['oof'] > 0].reset_index(drop=True)
df['date'] = train_date[len(train_date) - len(df):].reset_index(drop=True)['date']
test_ = test_all[['date']].copy()
test_['oof'] = test_predss
df = pd.concat([df, test_]).reset_index(drop=True)
if transform_log:
    df['true'], df['oof'] = np.exp(df['true']), np.exp(df['oof'])
df = df[df['date'] > datetime.datetime(2018, 1, 1)]
plot_true_oof(df)

# _df = df[df['date'] >= datetime.datetime(2021, 1, 1)].copy()
# incremental_weight = (_df['true'] / _df['oof']).mean()
# print('incremental_weight(2021>=):', incremental_weight)
# df['oof'] = df['oof'] * (incremental_weight)
# plot_true_oof(df, title='(incremental_weight)')


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Test prediction tuning by smoothing</p>

# In[48]:


incremental_weight_dict = {}
for window_length in [11, 21, 31, 51, 71, 101, 201, 301]: # [11, 101, 301, 501, 601, 701, 801, 901]
    print(f'incremental_weight: window_length={window_length}')
    
    df = pd.DataFrame([y_train.values, np.mean(oof_predss, axis=1)], index=['true', 'oof']).T
    df = df[df['oof'] > 0].reset_index(drop=True)
    df['date'] = train_date[len(train_date) - len(df):].reset_index(drop=True)['date']
    test_ = test_all[['date']].copy()
    test_['oof'] = test_predss
    df = pd.concat([df, test_]).reset_index(drop=True)
    if transform_log:
        df['true'], df['oof'] = np.exp(df['true']), np.exp(df['oof'])
    df = df[df['date'] > datetime.datetime(2018, 1, 1)]
    _df = df[df['date'] >= datetime.datetime(2021, 1, 1)].copy()
    
    __df = _df[_df['date'] < datetime.datetime(2022, 1, 1)].copy()
    incremental_weight = savgol_filter((__df['true'] / __df['oof']).values, window_length, 3, mode='nearest')
    df.loc[_df[_df['date'] >= datetime.datetime(2022, 1, 1)].index, ['incremental_weight']] = incremental_weight
    df['incremental_weight'] = df['incremental_weight'].fillna(1)
    df['oof'] = df['oof'] * df['incremental_weight']
    df = df[df['date'] >= datetime.datetime(2019, 1, 1)].copy()
    plot_true_oof(df, title=f'(incremental_weight {window_length})')
    incremental_weight_dict[f'window_length_{window_length}'] = incremental_weight
    
incremental_weight_dict[f'window_length_0'] = np.ones(365)


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">SHAP Analysis</p>
# SHAP stands for SHapley Additive exPlanations, a method for determining the contribution of each variable (feature) to the model's predicted outcome. Since SHAP cannot be adapted for ensemble models, let's use SHAP to understand `Xgboost` and `Catboost`.
# 
# **Consideration of Results:**  
# 
# Reference1. https://meichenlu.com/2018-11-10-SHAP-explainable-machine-learning/  
# Reference2. https://christophm.github.io/interpretable-ml-book/shap.html

# ### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:85%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Xgboost</p>

# In[49]:


shap.initjs()
explainer = shap.TreeExplainer(model=trained_models['xgb'][-1])
shap_values = explainer.shap_values(X=X_val)

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


# ### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:85%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Catboost</p>

# In[50]:


shap.initjs()
explainer = shap.TreeExplainer(model=trained_models['cat'][-1])
shap_values = explainer.shap_values(X=X_val)

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


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Inference with [country store product] Ratio </p>
# Convert predicted values by multiplying store_weights and country_weights,product_ratio.  
# The strategy is based on the notebook here.
# https://www.kaggle.com/code/mikailduzenli/tps-2022-notebook#Forecasting
# 
# **I don't know why, but country_weights are all the same.**

# In[51]:


test_all_df_dates = test_all[["date"]].copy()
test_all_df_dates["num_sold"] = test_predss * incremental_weight_dict['window_length_101']
if transform_log:
    test_all_df_dates["num_sold"] = np.exp(test_all_df_dates["num_sold"])


# In[52]:


def preprocess_product_ratio(product_ratio_df, year):
    product_ratio_year = product_ratio_df.loc[product_ratio_df['date'].dt.year == year].copy()
    product_ratio_year['mm-dd'] = product_ratio_year['date'].dt.strftime('%m-%d')
    product_ratio_year = product_ratio_year.drop(columns='date')
    product_ratio_year = product_ratio_year.reset_index(drop=True)
    return product_ratio_year

product_ratio_ = []
for year in [2018, 2019]:
    product_ratio_.append(preprocess_product_ratio(product_ratio_df, year))
product_ratio = preprocess_product_ratio(product_ratio_df, 2017)
product_ratio['mean_ratios'] = pd.concat(product_ratio_, axis=1)['ratios'].mean(axis=1)


# In[53]:


test_product_ratio_df = df_test.copy()
test_product_ratio_df['mm-dd'] = test_product_ratio_df['date'].dt.strftime('%m-%d')
test_product_ratio_df = pd.merge(test_product_ratio_df, product_ratio, how="left", on = ["mm-dd","product"])


# In[54]:


temp_df = pd.concat([product_ratio_df, test_product_ratio_df]).reset_index(drop=True)
f,ax = plt.subplots(figsize=(15, 6))
sns.lineplot(data=temp_df, x="date", y="ratios", hue="product");


# In[55]:


test_sub_df = pd.merge(df_test, test_all_df_dates, how="left")
test_sub_df["ratios"] = test_product_ratio_df["ratios"]
test_sub_df["mean_ratios"] = test_product_ratio_df["mean_ratios"]


# In[56]:


df_copy = test_sub_df.copy()
for book in df_copy['product'].unique():
    sf = savgol_filter(df_copy.loc[df_copy['product'] == book, 'mean_ratios'], 1001, 3, mode='nearest')
    df_copy.loc[df_copy['product'] == book, 'smoothed_ratios'] = sf
test_sub_df["smoothed_ratios"] = df_copy["smoothed_ratios"]
test_sub_df.head(5)


# In[57]:


f, ax = plt.subplots(figsize=(15, 6))
# sns.lineplot(data=df_copy.tail(20000), x="date", y="mean_ratios", hue="product", ax=ax)
sns.lineplot(data=df_copy.tail(20000), x="date", y="smoothed_ratios", hue="product", linestyle='--', ax=ax)


# In[58]:


def ratio_multiplied_inference(df):
    new_df = df.copy()
    store_weights = df_train.groupby('store')['num_sold'].sum()/df_train['num_sold'].sum()
    
    # country_weights = df_train.groupby('country')['num_sold'].sum()/df_train['num_sold'].sum()
    country_weights = pd.Series(index = test_sub_df["country"].unique(), data=1/5)
    # country_weights_dict = {'Argentina': 0.187, 'Canada': 0.180, 'Estonia': 0.177, 'Japan': 0.167, 'Spain': 0.177}
    # country_weights = pd.Series(country_weights_dict)
    
    for country in country_weights.index:
        new_df.loc[(new_df["country"] == country), "num_sold"] = new_df.loc[(new_df["country"] == country), "num_sold"] *  country_weights[country]
    for store in store_weights.index:
        new_df.loc[new_df["store"] == store, "num_sold"] = new_df.loc[new_df["store"] == store, "num_sold"] * store_weights[store]
        
    #new_df["num_sold"] = new_df["num_sold"] * new_df["ratios"]
    # new_df["num_sold"] = new_df["num_sold"] * new_df["mean_ratios"] * 1.5
    new_df["num_sold"] = new_df["num_sold"] * new_df["smoothed_ratios"] * 1.5
    new_df["num_sold"] = new_df["num_sold"].round()
    new_df = new_df.drop(columns=["ratios"])    
    
    return new_df

# data = ratio_multiplied_inference(test_sub_df)
# f,ax = plt.subplots(figsize=(15, 6))
# sns.lineplot(data=data[(data['store'] == 'Kagglazon') & (data['product'] == 'Improve Your Coding')], x="date", y="num_sold", hue="country");


# In[59]:


sub = pd.read_csv(os.path.join(filepath, 'sample_submission.csv'))
sub["num_sold"] = ratio_multiplied_inference(test_sub_df)["num_sold"]
sub = sub.rename(columns={'Unnamed: 0': 'id'})
sub.to_csv('submission_ratio.csv', index=False)
sub


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Xgboost model without [country store product] Ratios</p>

# In[60]:


train = df_train.reset_index(drop=True)
test = df_test.reset_index(drop=True)
# train = train.loc[~((train["date"] >= "2020-01-01") & (train["date"] < "2021-01-01"))]
# train['date'].dt.year.unique()


# In[61]:


# Applay Feature Engineering
train = feature_engineer(train)
test = feature_engineer(test)

# Logarithmically transform
if transform_log:
    train['num_sold'] = np.log(train['num_sold'])

X_train = train.drop(columns=["num_sold"]).reset_index(drop=True)
y_train = train["num_sold"].reset_index(drop=True)
X_test = test.reset_index(drop=True)

oe = OrdinalEncoder(cols=['country', 'store', 'product'])
X_train = oe.fit_transform(X_train)
X_test = oe.transform(X_test)

print(f"X_train shape :{X_train.shape} , y_train shape :{y_train.shape}")
print(f"X_test shape :{X_test.shape}")
print(f"X_train ->  isnull :{X_train.isnull().values.sum()}")
print(f"X_test -> isnull :{X_test.isnull().values.sum()}")
X_train.head(3)


# In[62]:


xgb_params = {
    'n_estimators': 2000,
    'learning_rate': 0.171655957085321,
    'booster': 'gbtree',
    'lambda': 0.0856890877950814,
    'alpha': 0.000223555755009136,
    'subsample': 0.817828998442447,
    'colsample_bytree': 0.247566473896556,
    'max_depth': 9,
    'min_child_weight': 6,
    'eta': 0.0000146950106167549,
    'gamma': 0.0315067984172879,
    'grow_policy': 'depthwise',
    'n_jobs': -1,
    'objective': 'reg:squarederror',
    'eval_metric': 'mape',
    'verbosity': 0,
    'random_state': 42,
}

eval_results_ = {}
best_iters_ = []
oof = np.zeros((X_train.shape[0]))
test_preds = np.zeros((X_test.shape[0]))
val_scores = []

splitter = Splitter(kfold='tscv_2', n_splits=5)
for i, (X_train_, X_val, y_train_, y_val, val_index) in enumerate(splitter.split_data(X_train, y_train, random_state_list=[0])):
    fold = i % n_splits
    m = i // n_splits
    
    fit_set = xgb.DMatrix(X_train_, y_train_)
    val_set = xgb.DMatrix(X_val, y_val)
    watchlist = [(fit_set, 'fit'), (val_set, 'val')]

    eval_results_[fold] = {}
    model = xgb.train(
        num_boost_round=xgb_params['n_estimators'],
        params=xgb_params,
        dtrain=fit_set,
        evals=watchlist,
        evals_result=eval_results_[fold],
        verbose_eval=False,
        callbacks=[xgb.callback.EarlyStopping(early_stopping_rounds, data_name='val', save_best=True)])

    val_preds = model.predict(val_set)
    test_preds = model.predict(xgb.DMatrix(X_test)) #/ n_splits

    oof[val_index] = val_preds

    val_score = metric(y_val, val_preds)
    best_iter = model.best_iteration
    best_iters_.append(best_iter)
    val_scores.append(val_score)
    print(f'Fold: {blu}{fold:>3}{res}| {metric_name}: {blu}{val_score:.5f}{res}' f' | Best iteration: {blu}{best_iter:>4}{res}')

xgb_test_preds = test_preds
    
mean_cv_score_full = metric(y_train, oof)
# print(f'{"*" * 50}\n{red}Mean full{res} {metric_name} : {red}{mean_cv_score_full:.5f}{res}')
print(f'{red}Mean val{res} {metric_name}  : {red}{np.mean(val_scores):.5f}{res}')


# ## <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#1871c9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #000966">Catboost model without [country store product] Ratios</p>

# In[63]:


params={
    'n_estimators': 2000,
    'learning_rate': 0.07725732658711602, # 0.07725732658711602
    'depth': 5,
    'l2_leaf_reg': 8.601133541582584,
    'subsample': 0.4279526734063217,
    'colsample_bylevel': 0.6767696482697301,
    'task_type': 'CPU',
    'verbose': False,
    'allow_writing_files': False,
    'random_state': 42
}
cat_features = ['country', 'store', 'product']

feature_importances_ = pd.DataFrame(index=X_train.columns)
best_iters_ = []
oof = np.zeros((X_train.shape[0]))
test_preds = np.zeros((X_test.shape[0]))
val_scores = []

splitter = Splitter(kfold='tscv_2', n_splits=5)
for i, (X_train_, X_val, y_train_, y_val, val_index) in enumerate(splitter.split_data(X_train, y_train, random_state_list=[0])):
    fold = i % n_splits
    m = i // n_splits
    
    train_pool = Pool(X_train_, y_train_, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=[train_pool, val_pool], 
              use_best_model=True, verbose=False, early_stopping_rounds=10, plot=False)
    
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test) #/ n_splits

    oof[val_index] = val_preds

    val_score = metric(y_val, val_preds)
    best_iter = model.best_iteration_
    best_iters_.append(best_iter)
    val_scores.append(val_score)
    
    print(f'Fold: {blu}{i:>3}{res} | {metric_name}: {blu}{val_score:.5f}{res} | Best iteration: {blu}{best_iter:>4}{res}')

cat_test_preds = test_preds
    
mean_cv_score_full = metric(y_train, oof)
# print(f'{"*" * 50}\n{red}Mean full{res} {metric_name} : {red}{mean_cv_score_full:.5f}{res}')
print(f'{red}Mean val{res} {metric_name}  : {red}{np.mean(val_scores):.5f}{res}')


# In[64]:


for name, test_preds in zip(['xgb', 'cat'], [xgb_test_preds, cat_test_preds]):
    sub = pd.read_csv(os.path.join(filepath, 'sample_submission.csv'))
    sub["num_sold"] = test_preds
    if transform_log:
        sub["num_sold"] = np.exp(test_preds)
    sub["num_sold"] = sub["num_sold"] * 1.4
    sub["num_sold"] = sub["num_sold"].round()
    sub = sub.rename(columns={'Unnamed: 0': 'id'})
    sub.to_csv(f'submission_{name}_withoutRatios.csv', index=False)


# In[65]:


def plot_scatter_with_regression(file_path_x, file_path_y, target_col, graph_title=None, figsize=(8, 6)):
    # Read the data
    x = pd.read_csv(file_path_x)[target_col]
    y = pd.read_csv(file_path_y)[target_col]

    # Calculate the correlation coefficient
    correlation_coefficient = x.corr(y)

    # Set the size of the figure
    plt.figure(figsize=figsize)

    # Create the scatter plot with transparency (alpha) for better visualization
    sns.scatterplot(x=x, y=y, s=50, alpha=0.7, color='skyblue')

    # Create the regression line
    sns.regplot(x=x, y=y, scatter=False, line_kws={'color': 'red'})

    # Set the title of the graph and add the correlation coefficient to it if provided
    if graph_title:
        plt.title(f"{graph_title} [Correlation Coefficient: {correlation_coefficient:.2f}]")
    else:
        plt.title(f"Correlation Coefficient: {correlation_coefficient:.2f}")

    # Set the labels for x and y axes
    plt.xlabel(f'withoutRatio {target_col} pred')
    plt.ylabel(f'{target_col} pred')
    plt.show()

# Example usage:
file_path_y = '/kaggle/working/submission_ratio.csv'
graph_title = 'Scatter Plot with Test Prediction'

file_path_x = '/kaggle/working/submission_xgb_withoutRatios.csv'
plot_scatter_with_regression(file_path_x, file_path_y, target_col, graph_title+' Xgboost', figsize=(8, 6))
file_path_x = '/kaggle/working/submission_cat_withoutRatios.csv'
plot_scatter_with_regression(file_path_x, file_path_y, target_col, graph_title+' Catoost', figsize=(8, 6))


# In[66]:


file_path = [
    '/kaggle/working/submission_ratio.csv',
    '/kaggle/working/submission_xgb_withoutRatios.csv',
    '/kaggle/working/submission_cat_withoutRatios.csv',
]

test_predss = pd.concat([pd.read_csv(_) for _ in file_path], axis=1)['num_sold'].values
sub = pd.read_csv(os.path.join(filepath, 'sample_submission.csv'))
sub["num_sold"] = np.average(test_predss, weights=[30, 2, 2], axis=1)
sub["num_sold"] = sub["num_sold"].round()
sub.to_csv('submission.csv', index=False)
sub

