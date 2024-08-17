#!/usr/bin/env python
# coding: utf-8

# # 1. INTRODUCTION
# <center>
# <img src="https://cdn09.allafrica.com/download/pic/main/main/csiid/00520144:4b641081094b9e454d56ebd33737ca3c:arc614x376:w1200.jpg" width=1300 height=800 />
# </center>

# **<font size="4">Porblem Description</font>**
# 
# **Predicting CO2 Emissions**
# 
# <font size="3">The ability to accurately monitor carbon emissions is a critical step in the fight against climate change. Precise carbon readings allow researchers and governments to understand the sources and patterns of carbon mass output. While Europe and North America have extensive systems in place to monitor carbon emissions on the ground, there are few available in Africa.</font>
# 
# <font size="3">The objective of this challenge is to create a machine learning models using open-source CO2 emissions data from Sentinel-5P satellite observations to predict future carbon emissions.</font>
# 
# <font size="3">These solutions may help enable governments, and other actors to estimate carbon emission levels across Africa, even in places where on-the-ground monitoring is not possible.</font>
# 
# 
# 

# # 2. IMPORT LIBRARIES & DATA

# In[1]:


import sys
assert sys.version_info >= (3, 5)

import sklearn
assert sklearn.__version__ >= "0.20"
import numpy as np
import os

import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
tqdm_notebook.get_lock().locks = []
from prettytable import PrettyTable
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.4)
from copy import deepcopy
from functools import partial
from itertools import combinations

from sklearn.cluster import KMeans
get_ipython().system('pip install yellowbrick')
from yellowbrick.cluster import KElbowVisualizer
import folium
from haversine import haversine
import random
from random import uniform
import gc
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_squared_log_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler,PowerTransformer, FunctionTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import stats
import statsmodels.api as sm
import math
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import BaseEstimator, TransformerMixin
get_ipython().system('pip install optuna')
import optuna
import xgboost as xgb
get_ipython().system('pip install catboost')
get_ipython().system('pip install lightgbm --install-option=--gpu --install-option="--boost-root=C:/local/boost_1_69_0" --install-option="--boost-librarydir=C:/local/boost_1_69_0/lib64-msvc-14.1"')
import lightgbm as lgb
get_ipython().system('pip install category_encoders')
from category_encoders import OneHotEncoder, OrdinalEncoder, CountEncoder, CatBoostEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import PassiveAggressiveRegressor, ARDRegression, RidgeCV, ElasticNetCV
from sklearn.linear_model import TheilSenRegressor, RANSACRegressor, HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor,GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoost, CatBoostRegressor,CatBoostClassifier
from catboost import Pool
from sklearn.neighbors import KNeighborsRegressor
# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
pd.pandas.set_option('display.max_columns',None)


# In[2]:


train=pd.read_csv('/kaggle/input/playground-series-s3e20/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s3e20/test.csv')
train=train.drop(columns=["ID_LAT_LON_YEAR_WEEK"])
test=test.drop(columns=["ID_LAT_LON_YEAR_WEEK"])

train_copy=train.copy()
test_copy=test.copy()

train.head()


# ## 2.1 Missing Values

# In[3]:


table = PrettyTable()

table.field_names = ['Column Name', 'Data Type', 'Train Missing %', 'Test Missing %']
for column in train.columns:
    data_type = str(train[column].dtype)
    non_null_count_train= np.round(100-train[column].count()/train.shape[0]*100,1)
    if column!='emission':
        non_null_count_test = np.round(100-test[column].count()/test.shape[0]*100,1)
    else:
        non_null_count_test="NA"
    table.add_row([column, data_type, non_null_count_train,non_null_count_test])
print(table)


# **INFERENCES**
# 
# <font size="3">There are a lot of missing values and it seems like data extraction process limitation because datapoints from the main factors like SO2 has same missing values across the sub features</font> 

# # 3. Exploratory Data Analysis

# ## 3.1 Target Analysis

# In[4]:


emission = train['emission']

mean_emission = np.mean(emission)
median_emission = np.median(emission)

fig, ax = plt.subplots(figsize=(12, 5))

ax.hist(emission, bins=20, density=True, alpha=0.5, label='Emission Histogram')

x_values = np.linspace(emission.min(), emission.max(), len(emission))
density_values = (1 / (np.sqrt(2 * np.pi) * np.std(emission))) * np.exp(-0.5 * ((x_values - mean_emission) / np.std(emission))**2)
ax.plot(x_values, density_values, color='red', label='Emission Density')

ax.axvline(mean_emission, color='blue', linestyle='dashed', linewidth=2, label='Mean Emission')
ax.axvline(median_emission, color='green', linestyle='dashed', linewidth=2, label='Median Emission')

ax.set_xlabel('Emission')
ax.set_ylabel('Frequency / Density')
ax.set_title('Emission Distribution and Density Plot')

x_min = emission.min()
x_max = emission.max()
ax.set_xlim([x_min, x_max])

ax.legend(bbox_to_anchor=(1, 1), fancybox=False, shadow=False, loc='upper left')

plt.tight_layout()
plt.show()


# <font size="3">There is a big tail towards the right, could be due to emissions from certain location. Maybe, a log transformation on the traget would be a good option to be considered</font>
# 
# <font size="3"> Let us also look at the log transformed target</font>

# In[5]:


emission = np.log1p(train['emission'])

mean_emission = np.mean(emission)
median_emission = np.median(emission)

fig, ax = plt.subplots(figsize=(12, 5))

ax.hist(emission, bins=20, density=True, alpha=0.5, label='Emission Histogram')

x_values = np.linspace(emission.min(), emission.max(), len(emission))
density_values = (1 / (np.sqrt(2 * np.pi) * np.std(emission))) * np.exp(-0.5 * ((x_values - mean_emission) / np.std(emission))**2)
ax.plot(x_values, density_values, color='red', label='Emission Density')

ax.axvline(mean_emission, color='blue', linestyle='dashed', linewidth=2, label='Mean Emission')
ax.axvline(median_emission, color='green', linestyle='dashed', linewidth=2, label='Median Emission')

ax.set_xlabel('Emission')
ax.set_ylabel('Frequency / Density')
ax.set_title('Emission Distribution and Density Plot')

x_min = emission.min()
x_max = emission.max()
ax.set_xlim([x_min, x_max])

ax.legend(bbox_to_anchor=(1, 1), fancybox=False, shadow=False, loc='upper left')

plt.tight_layout()
plt.show()


# <font size="3">This looks better, all the high values now became close to the rest of the distribution</font>

# ## 3.2 Numerical Features Analysis

# ### 3.2.1 Train & Test Data Distributions

# In[6]:


cont_cols=[f for f in train.columns if train[f].dtype in [float,int] and train[f].nunique()>2 and f not in ['emission']]

# Calculate the number of rows needed for the subplots
num_rows = (len(cont_cols) + 2) // 3

# Create subplots for each continuous column
fig, axs = plt.subplots(num_rows, 3, figsize=(15, num_rows*4))

# Loop through each continuous column and plot the histograms
for i, col in enumerate(cont_cols):
    # Determine the range of values to plot
    max_val = max(train[col].max(), test[col].max())
    min_val = min(train[col].min(), test[col].min())
    range_val = max_val - min_val
    
    # Determine the bin size and number of bins
    bin_size = range_val / 20
    num_bins_train = round(range_val / bin_size)
    num_bins_test = round(range_val / bin_size)
    
    # Calculate the subplot position
    row = i // 3
    col_pos = i % 3
    
    # Plot the histograms
    sns.histplot(train[col], ax=axs[row][col_pos], color='green', kde=True, label='Train', bins=num_bins_train)
    sns.histplot(test[col], ax=axs[row][col_pos], color='blue', kde=True, label='Test', bins=num_bins_test)
    axs[row][col_pos].set_title(col)
    axs[row][col_pos].set_xlabel('Value')
    axs[row][col_pos].set_ylabel('Frequency')
    axs[row][col_pos].legend()

# Remove any empty subplots
if len(cont_cols) % 3 != 0:
    for col_pos in range(len(cont_cols) % 3, 3):
        axs[-1][col_pos].remove()

plt.tight_layout()
plt.show()


# **INFERENCES**
# 
# <font size="3"> Every feature has less distribution(less data) in the testset except the UV Aerosal Layer based features. Even the missing values % is up for all the featires except the mentioned ones</font>

# ## 3.3 Yearly emissions distributions

# In[7]:


fig, ax = plt.subplots(figsize=(14, 4))
data=train.copy()
data['emission']=np.log1p(data['emission'])
sns.boxplot(x='year', y='emission', data=data, ax=ax)
ax.set_title(f'Boxplot of emissions Across year')
plt.show()


# <font size="3"> There is a slight decrease in the year 2020 and I think it is because of COVID when we had less operations across sectors. We can actually try doing a covid adjustment in the target by increasing the esmissions by X% </font>

# In[8]:


data=train.copy()

data['emission']=np.where(data['year']!=2019,data['emission']*1.10,data['emission'])
fig, ax = plt.subplots(figsize=(14, 4))
data=train.copy()
data['emission']=np.log1p(data['emission'])
sns.boxplot(x='year', y='emission', data=data, ax=ax)
ax.set_title(f'Boxplot of emissions Across year')
plt.show()


# # 4. Feature Engineering

# ## 4.1 Iterative Missing Value Imputation

# <font size='4'>The idea is to implement missing values using an Iterative updation of missing features using Decision Trees or any tree based algorithms. There is also a similar package that uses MICE alogorithm however, we will develop a raw code than relying on the package</font>
# 
# **STEPS TO FILL MISSING VALUES**
# 1. <font size="4">Store the instances where there are missing values in each feature assuming we have N features with missing values</font>
# 2. <font size="4">Initially fill all the missing values with median/mean</font>
# 3. <font size="4">Take each feature(i) and use the rest of the features to predict the missing values in that feature. This way we can update all the N features</font>
# 4. <font size="4">Iterate this until the change in update values gets saturated or for n interations, this is evaluated using the error change between each iteration</font>
# 5. <font size="4">Target feature will not be used to impute to avoid data leakages</font>

# In[9]:


# drop columns with missing %>90
high_missing_cols=[f for f in train.columns if train[f].isna().sum()/train[f].nunique()>0.9 and f not in ['emission']]

# train=train.drop(columns=high_missing_cols)
# test=test.drop(columns=high_missing_cols)
cont_missing=[f for f in train.columns if train[f].dtype!='O' and f not in ['emission']]


# In[10]:


cb_params = {
            'iterations': 200,
            'depth': 6,
            'learning_rate': 0.008,
            'l2_leaf_reg': 0.5,
            'random_strength': 0.2,
            'max_bin': 150,
            'od_wait': 80,
            'one_hot_max_size': 70,
            'grow_policy': 'Depthwise',
            'bootstrap_type': 'Bayesian',
            'od_type': 'IncToDec',
            'eval_metric': 'RMSE',
            'loss_function': 'RMSE',
            'random_state': 42,
        }
lgb_params = {
            'n_estimators': 100,
            'max_depth': 6,
            "num_leaves": 16,
            'learning_rate': 0.05,
            'subsample': 0.7,
            'colsample_bytree': 0.8,
            #'reg_alpha': 0.25,
            'reg_lambda': 5e-07,
            'objective': 'regression_l2',
            'metric': 'mean_squared_error',
            'boosting_type': 'gbdt',
            'random_state': 42
        }
def rmse(y1,y2):
    return(np.sqrt(mean_squared_error(y1,y2)))

def store_missing_rows(df, features):
    missing_rows = {}
    
    for feature in features:
        missing_rows[feature] = df[df[feature].isnull()]
    
    return missing_rows
def fill_missing_numerical(train,test,target, features, max_iterations=10):
    
    df=pd.concat([train.drop(columns=[target]),test],axis="rows")
    df=df.reset_index(drop=True)
    
    features=[f for f in features if df[f].isna().sum()>0]
    # Step 1: Store the instances with missing values in each feature
    missing_rows = store_missing_rows(df, features)
    
    # Step 2: Initially fill all missing values with "Median"
    for f in features:
        df[f]=df[f].fillna(df[f].median())
    
    cat_features=[f for f in df.columns if df[f].dtype=="O"]
    dictionary = {feature: [] for feature in features}
    
    for iteration in tqdm(range(max_iterations), desc="Iterations"):
        for feature in features:
            # Skip features with no missing values
            rows_miss = missing_rows[feature].index
            
            missing_temp = df.loc[rows_miss].copy()
            non_missing_temp = df.drop(index=rows_miss).copy()
            y_pred_prev=missing_temp[feature]
            missing_temp = missing_temp.drop(columns=[feature])
            
            
            # Step 3: Use the remaining features to predict missing values using Random Forests
            X_train = non_missing_temp.drop(columns=[feature])
            y_train = non_missing_temp[[feature]]
            if iteration/max_iterations<0.5:
                model = LinearRegression()
                model.fit(X_train, y_train)
            else:
#                 model = CatBoostRegressor(**cb_params)
                model= lgb.LGBMRegressor(**lgb_params)
#                 model.fit(X_train, y_train,cat_features=cat_features, verbose=False)
                model.fit(X_train, y_train, verbose=False)


            
            
            # Step 4: Predict missing values for the feature and update all N features
            y_pred = model.predict(missing_temp)
            df.loc[rows_miss, feature] = y_pred
            error_minimize=rmse(y_pred,y_pred_prev)
            dictionary[feature].append(error_minimize)  # Append the error_minimize value

    for feature, values in dictionary.items():
        iterations = range(1, len(values) + 1)  # x-axis values (iterations)
        plt.plot(iterations, values, label=feature)  # plot the values
        plt.xlabel('Iterations')
        plt.ylabel('RMSE')
        plt.title('Minimization of RMSE with iterations')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#         if len(dictionary) % 3 == 0 or feature == list(dictionary.keys())[-1]:
    plt.show()
    train[features] = np.array(df.iloc[:train.shape[0]][features])
    test[features] = np.array(df.iloc[train.shape[0]:][features])

    return train,test


train,test = fill_missing_numerical(train,test,"emission",cont_missing,8)


# ## 4.2 Pre-Processing Features

# <font size="3"> Align the values of 2019 & 2020 with 2021 emission since 2022 is closer to 2021. This is based on the findings by [@patrick0302](https://www.kaggle.com/code/patrick0302/align-2019-2020-emission-values-with-2021), a small modification has been made which is to use log of emissions and transform it back to normal emissions</font>

# In[11]:


data=train.copy()
data['emission']=np.log1p(data['emission'])

data.insert(2, "lat_lon", list(zip(data["latitude"], data["longitude"])))

# Calculate statistics for each year and lat_lon combination
agg_stats = data.groupby(['year', 'lat_lon'])['emission'].agg(['std', 'mean']).reset_index()

# Merge the calculated statistics with the data DataFrame
data = pd.merge(data, agg_stats, on=['year', 'lat_lon'], suffixes=('', '_stat'))

# Normalize emissions using the calculated mean and std
data['emission_norm'] = (data['emission'] - data['mean']) / data['std']

# Adjust 'std' and 'mean' values for 2019 and 2020 to match 2021
adjust_years = [2019, 2020]
for year in adjust_years:
    mask = data['year'] == year
    data.loc[mask, 'std'] = data.loc[data['year'] == 2021, 'std'].values
    data.loc[mask, 'mean'] = data.loc[data['year'] == 2021, 'mean'].values

# Denormalize to get adjusted emission values
data['emission_new'] = data['emission_norm'] * data['std'] + data['mean']

# Replace NaN values with 0 in the 'emission_new' column
data['emission_new'].fillna(0, inplace=True)

data['emission_new']=np.expm1(data['emission_new'])
train['emission'] = data['emission_new']

del(data)


# <font size="3">Pre-Processing has been taken from the work of [dmitryuarov](https://www.kaggle.com/code/dmitryuarov/ps3e20-rwanda-emission-advanced-fe-29-7), please support this work too</font>

# In[12]:


import datetime as dt
seed = 228
def get_id(row):
    return int(''.join(filter(str.isdigit, str(row['latitude']))) + ''.join(filter(str.isdigit, str(row['longitude']))))

train['id'] = train[['latitude', 'longitude']].apply(lambda row: get_id(row), axis=1)
test['id'] = test[['latitude', 'longitude']].apply(lambda row: get_id(row), axis=1)
new_ids = {id_: new_id for new_id, id_ in enumerate(train['id'].unique())}
train['id'] = train['id'].map(new_ids)
test['id'] = test['id'].map(new_ids)

def get_month(row):
    date = dt.datetime.strptime(f'{row["year"]}-{row["week_no"]+1}-1', "%Y-%W-%w")
    return date.month


rwanda_center = (-1.9607, 29.9707)
park_biega = (-1.8866, 28.4518) 
kirumba = (-0.5658, 29.1714) 
massif = (-2.9677, 28.6469)
lake = (-1.9277, 31.4346)
mbarara = (-0.692, 30.602)
muy = (-2.8374, 30.3346)


def cluster_features(df, cluster_centers):
    for i, cc in enumerate(cluster_centers.values()):
        df[f'cluster_{i}'] = df.apply(lambda x: haversine((x['latitude'], x['longitude']), cc, unit='ft'), axis=1)
    return df


def coor_rotation(df):
    df['rot_15_x'] = (np.cos(np.radians(15)) * df['longitude']) + \
                     (np.sin(np.radians(15)) * df['latitude'])
    
    df['rot_15_y'] = (np.cos(np.radians(15)) * df['latitude']) + \
                     (np.sin(np.radians(15)) * df['longitude'])

    df['rot_30_x'] = (np.cos(np.radians(30)) * df['longitude']) + \
                     (np.sin(np.radians(30)) * df['latitude'])

    df['rot_30_y'] = (np.cos(np.radians(30)) * df['latitude']) + \
                     (np.sin(np.radians(30)) * df['longitude'])
    return df

y = train['emission']

def preprocessing(df):
    
#     cols_save = ['id', 'latitude', 'longitude', 'year', 'week_no', 'Ozone_solar_azimuth_angle']
#     df = df[cols_save]
    
    good_col = 'Ozone_solar_azimuth_angle'
    df[good_col] = df.groupby(['id', 'year'])[good_col].ffill().bfill()
    df[f'{good_col}_lag_1'] = df.groupby(['id', 'year'])[good_col].shift(1).fillna(0)
    
    df = coor_rotation(df)
    
    for col, coors in zip(
        ['dist_rwanda', 'dist_park', 'dist_kirumba', 'dist_massif', 'dist_lake', 'dist_mbarara', 'dist_muy'], 
        [rwanda_center, park_biega, kirumba, massif, lake, mbarara, muy]
    ):
        df[col] = df.apply(lambda x: haversine((x['latitude'], x['longitude']), coors, unit='ft'), axis=1)
    
    df['month'] = df[['year', 'week_no']].apply(lambda row: get_month(row), axis=1)
    df['is_covid'] = (df['year'] == 2020) & (df['month'] > 2) | (df['year'] == 2021) & (df['month'] == 1)
    df["week"] = (df["year"] - 2019) * 53 + df["week_no"]
    df['is_lockdown'] = (df['year'] == 2020) & ((df['month'].isin([3,4])))
    df['is_covid_peak'] = (df['year'] == 2020) & ((df['month'].isin([4,5,6])))
    df['is_covid_dis_peak'] = (df['year'] == 2021) & ((df['month'].isin([7,8,9])))
    df['public_holidays'] = (df['week_no'].isin([0, 51, 12, 30]))
            
    return df
    
train = preprocessing(train)
test = preprocessing(test)

df = pd.concat([train, test], axis=0, ignore_index=True)
coordinates = df[['latitude', 'longitude']].values
clustering = KMeans(n_clusters=12, max_iter=1000, random_state=seed).fit(coordinates)
cluster_centers = {i: tuple(centroid) for i, centroid in enumerate(clustering.cluster_centers_)}
df = cluster_features(df, cluster_centers)

train = df.iloc[:-len(test),:]
test = df.iloc[-len(test):,:]
del df

train = train.drop('id', axis=1)
test = test.drop(['id','emission'], axis=1)

# In case anything is missing, I'm calling the missing value filling algorithm 
cont_missing=[f for f in train.columns if train[f].dtype!='O' and f not in ['emission']]
train,test = fill_missing_numerical(train,test,"emission",cont_missing,10)


# ## 4.2 Numeric Transformations

# <font size="3">We're going to see what transformation works better for each feature and select them, the idea is to compress the data. There could be situations where you will have to stretch the data. These are the methods applied:</font>
# 
# 1. <font size="3">**Log Transformation**</font>: <font size="3">This transformation involves taking the logarithm of each data point. It is useful when the data is highly skewed and the variance increases with the mean.</font>
#                 y = log(x)
# 
# 2. <font size="3">**Square Root Transformation**</font>: <font size="3">This transformation involves taking the square root of each data point. It is useful when the data is highly skewed and the variance increases with the mean.</font>
#                 y = sqrt(x)
# 
# 3. <font size="3">**Box-Cox Transformation**</font>: <font size="3">This transformation is a family of power transformations that includes the log and square root transformations as special cases. It is useful when the data is highly skewed and the variance increases with the mean.</font>
#                 y = [(x^lambda) - 1] / lambda if lambda != 0
#                 y = log(x) if lambda = 0
# 
# 4. <font size="3">**Yeo-Johnson Transformation**</font>: <font size="3">This transformation is similar to the Box-Cox transformation, but it can be applied to both positive and negative values. It is useful when the data is highly skewed and the variance increases with the mean.</font>
#                 y = [(|x|^lambda) - 1] / lambda if x >= 0, lambda != 0
#                 y = log(|x|) if x >= 0, lambda = 0
#                 y = -[(|x|^lambda) - 1] / lambda if x < 0, lambda != 2
#                 y = -log(|x|) if x < 0, lambda = 2
# 
# 5. <font size="3">**Power Transformation**</font>: <font size="3">This transformation involves raising each data point to a power. It is useful when the data is highly skewed and the variance increases with the mean. The power can be any value, and is often determined using statistical methods such as the Box-Cox or Yeo-Johnson transformations.</font>
#                 y = [(x^lambda) - 1] / lambda if method = "box-cox" and lambda != 0
#                 y = log(x) if method = "box-cox" and lambda = 0
#                 y = [(x + 1)^lambda - 1] / lambda if method = "yeo-johnson" and x >= 0, lambda != 0
#                 y = log(x + 1) if method = "yeo-johnson" and x >= 0, lambda = 0
#                 y = [-(|x| + 1)^lambda - 1] / lambda if method = "yeo-johnson" and x < 0, lambda != 2
#                 y = -log(|x| + 1) if method = "yeo-johnson" and x < 0, lambda = 2

# In[13]:


cont_cols=[f for f in train.columns if train[f].dtype!="O" and f not in ['emission'] and train[f].nunique()/train[f].shape[0]*100>25]
len(cont_cols)


# In[14]:


sc=MinMaxScaler()
table = PrettyTable()
unimportant_features=[]
overall_best_score=200
overall_best_col='none'
table.field_names = ['Feature', 'Original RMSE', 'Transformation', 'Tranformed RMSE']

def min_max_scaler(train, test, column):
    
    sc=MinMaxScaler()
    
    max_val=max(train[column].max(),test[column].max())
    min_val=min(train[column].min(),test[column].min())

    train[column]=(train[column]-min_val)/(max_val-min_val)
    test[column]=(test[column]-min_val)/(max_val-min_val)
    
    return train,test
for col in cont_cols:
     train, test=min_max_scaler(train, test, col)
for col in cont_cols:
    
    # Log Transformation after MinMax Scaling(keeps data between 0 and 1)
    train["log_"+col]=np.log1p(train[[col]])
    test["log_"+col]=np.log1p(test[[col]])
    
    # Square Root Transformation
    train["sqrt_"+col]=np.sqrt(train[[col]])
    test["sqrt_"+col]=np.sqrt(test[[col]])
    
    # Box-Cox transformation
    combined_data = pd.concat([train[[col]], test[[col]]], axis=0)
    transformer = PowerTransformer(method='box-cox')
    # Apply scaling and transformation on the combined data
    scaled_data = sc.fit_transform(combined_data)+1
    transformed_data = transformer.fit_transform(scaled_data)

    # Assign the transformed values back to train and test data
    train["bx_cx_" + col] = transformed_data[:train.shape[0]]
    test["bx_cx_" + col] = transformed_data[train.shape[0]:]
    
    # Yeo-Johnson transformation
    transformer = PowerTransformer(method='yeo-johnson')
    train["y_J_"+col] = transformer.fit_transform(train[[col]])
    test["y_J_"+col] = transformer.transform(test[[col]])
    
    # Power transformation, 0.25
#     power_transform = lambda x: np.power(x + 1 - np.min(x), 0.25)
#     transformer = FunctionTransformer(power_transform)
#     train["pow_"+col] = transformer.fit_transform(train[[col]])
#     test["pow_"+col] = transformer.transform(test[[col]])
    
    # Power transformation, 0.1
#     power_transform = lambda x: np.power(x + 1 - np.min(x), 0.1)
#     transformer = FunctionTransformer(power_transform)
#     train["pow2_"+col] = transformer.fit_transform(train[[col]])
#     test["pow2_"+col] = transformer.transform(test[[col]])
    
    # log to power transformation
    train["log_sqrt"+col]=np.log1p(train["sqrt_"+col])
    test["log_sqrt"+col]=np.log1p(test["sqrt_"+col])
    
    temp_cols=[col,"log_"+col,"sqrt_"+col, "bx_cx_"+col,"y_J_"+col ,"log_sqrt"+col ]#"pow_"+col,"pow2_"+col,
    
    # Fill na becaue, it would be Nan if the vaues are negative and a transformation applied on it
    train[temp_cols]=train[temp_cols].fillna(0)
    test[temp_cols]=test[temp_cols].fillna(0)

    #Apply PCA on  the features and compute an additional column
    pca=TruncatedSVD(n_components=1)
    x_pca_train=pca.fit_transform(train[temp_cols])
    x_pca_test=pca.transform(test[temp_cols])
    x_pca_train=pd.DataFrame(x_pca_train, columns=[col+"_pca_comb"])
    x_pca_test=pd.DataFrame(x_pca_test, columns=[col+"_pca_comb"])
    temp_cols.append(col+"_pca_comb")
    #print(temp_cols)
    test=test.reset_index(drop=True) # to combine with pca feature
    
    train=pd.concat([train,x_pca_train],axis='columns')
    test=pd.concat([test,x_pca_test],axis='columns')
    
    # See which transformation along with the original is giving you the best univariate fit with target
    kf=KFold(n_splits=5, shuffle=True, random_state=42)
    
    MAE=[]
    
    for f in temp_cols:
        X=train[[f]].values
        y=train["emission"].values
        
        mae=[]
        for train_idx, val_idx in kf.split(X,y):
            X_train,y_train=X[train_idx],y[train_idx]
            x_val,y_val=X[val_idx],y[val_idx]
            
            model=LinearRegression()
            model.fit(X_train,y_train)
            y_pred=model.predict(x_val)
            mae.append(rmse(y_val,y_pred))
        MAE.append((f,np.mean(mae)))
        if overall_best_score>np.mean(mae):
            overall_best_score=np.mean(mae)
            overall_best_col=f
        if f==col:
            orig_mae=np.mean(mae)
    best_col, best_acc=sorted(MAE, key=lambda x:x[1], reverse=False)[0]
    
    cols_to_drop = [f for f in temp_cols if  f!= best_col and f not in col]
#     print(cols_to_drop)
    final_selection=[f for f in temp_cols if f not in cols_to_drop]
    if cols_to_drop:
        unimportant_features=unimportant_features+cols_to_drop
#         train=train.drop(columns=cols_to_drop)
#         test=test.drop(columns=cols_to_drop)

    table.add_row([col,orig_mae,best_col ,best_acc])
print(table)  
print("overall best CV RMSE score: ",overall_best_score)


# ## 4.3 Numerical Clustering

# <font size="3"> All the unimportant features that are not the best transformation technique are selected and applied a K-Means Clustering technique

# In[15]:


table = PrettyTable()
table.field_names = ['Cluster WOE Feature', 'MAE(CV-TRAIN)']
for col in cont_cols:
    sub_set=[f for f in unimportant_features if col in f]
#     print(sub_set)
    temp_train=train[sub_set]
    temp_test=test[sub_set]
    sc=StandardScaler()
    temp_train=sc.fit_transform(temp_train)
    temp_test=sc.transform(temp_test)
    model = KMeans()

    # print(ideal_clusters)
    kmeans = KMeans(n_clusters=12)
    kmeans.fit(np.array(temp_train))
    labels_train = kmeans.labels_

    train[col+"_unimp_cluster_WOE"] = labels_train
    test[col+"_unimp_cluster_WOE"] = kmeans.predict(np.array(temp_test))
    
#     cat_labels=cat_labels=train.groupby([col+"_unimp_cluster_WOE"])['emission'].mean()
#     cat_labels2=cat_labels.to_dict()
#     train[col+"_unimp_cluster_WOE"]=train[col+"_unimp_cluster_WOE"].map(cat_labels2)
#     test[col+"_unimp_cluster_WOE"]=test[col+"_unimp_cluster_WOE"].map(cat_labels2)
    
    kf=KFold(n_splits=5, shuffle=True, random_state=42)
    
    X=train[[col+"_unimp_cluster_WOE"]].values
    y=train["emission"].values

    best_rmse=[]
    for train_idx, val_idx in kf.split(X,y):
        X_train,y_train=X[train_idx],y[train_idx]
        x_val,y_val=X[val_idx],y[val_idx]
        model=LinearRegression()
        model.fit(X_train,y_train)
        y_pred=model.predict(x_val)
        best_rmse.append(rmse(y_val,y_pred))
        
    table.add_row([col+"_unimp_cluster_WOE",np.mean(best_rmse)])
    if overall_best_score<np.mean(best_rmse):
            overall_best_score=np.mean(best_rmse)
            overall_best_col=col+"_unimp_cluster_WOE"
    
print(table)


# ## 4.3 Arithmetic Better Features

# In[16]:


def better_features(train, test, target, cols, best_score):
    new_cols = []
    skf = KFold(n_splits=5, shuffle=True, random_state=42)  # Stratified k-fold object
    best_list=[]
    for i in tqdm(range(len(cols)), desc='Generating Columns'):
        col1 = cols[i]
        temp_df = pd.DataFrame()  # Temporary dataframe to store the generated columns
        temp_df_test = pd.DataFrame()  # Temporary dataframe for test data

        for j in range(i+1, len(cols)):
            col2 = cols[j]
            # Multiply
            temp_df[col1 + '*' + col2] = train[col1] * train[col2]
            temp_df_test[col1 + '*' + col2] = test[col1] * test[col2]

            # Divide (col1 / col2)
            temp_df[col1 + '/' + col2] = train[col1] / (train[col2] + 1e-5)
            temp_df_test[col1 + '/' + col2] = test[col1] / (test[col2] + 1e-5)

            # Divide (col2 / col1)
            temp_df[col2 + '/' + col1] = train[col2] / (train[col1] + 1e-5)
            temp_df_test[col2 + '/' + col1] = test[col2] / (test[col1] + 1e-5)

            # Subtract
            temp_df[col1 + '-' + col2] = train[col1] - train[col2]
            temp_df_test[col1 + '-' + col2] = test[col1] - test[col2]

            # Add
            temp_df[col1 + '+' + col2] = train[col1] + train[col2]
            temp_df_test[col1 + '+' + col2] = test[col1] + test[col2]

        SCORES = []
        for column in temp_df.columns:
            scores = []
            for train_index, val_index in skf.split(train, train[target]):
                X_train, X_val = temp_df[column].iloc[train_index].values.reshape(-1, 1), temp_df[column].iloc[val_index].values.reshape(-1, 1)
                y_train, y_val = train[target].iloc[train_index], train[target].iloc[val_index]
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = rmse(y_val, y_pred)
                scores.append(score)
            mean_score = np.mean(scores)
            SCORES.append((column, mean_score))

        if SCORES:
            best_col, best_acc = sorted(SCORES, key=lambda x: x[1])[0]
            corr_with_other_cols = train.drop([target] + new_cols, axis=1).corrwith(temp_df[best_col])
#             print(corr_with_other_cols.abs().max())
            if (corr_with_other_cols.abs().max() < 0.9 or best_acc < best_score) and corr_with_other_cols.abs().max() !=1 :
                train[best_col] = temp_df[best_col]
                test[best_col] = temp_df_test[best_col]
                new_cols.append(best_col)
                print(f"Added column '{best_col}' with mean RMSE: {best_acc:.4f} & Correlation {corr_with_other_cols.abs().max():.4f}")

    return train, test, new_cols


# <font size="3"> There are a lot of features already, it will be time consuming to explore all combinations. Hence I will be selecting top 50 features from Single features based Linear Regression</font>

# In[17]:


exist_cols = [f for f in train.columns if f not in ['emission', 'ID_LAT_LON_YEAR_WEEK']]
top_features = {}

for f in exist_cols:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X = train[[f]].values
    y = np.log1p(train["emission"].values)

    best_rmse = []
    for train_idx, val_idx in kf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        best_rmse.append(rmse(y_val, y_pred))

    avg_rmse = np.mean(best_rmse)
    top_features[f] = avg_rmse

# Sort the features based on log loss in ascending order
sorted_top_features = sorted(top_features.items(), key=lambda x: x[1])

# Get the top 100 features with the least log loss
top_50_features = [feature for feature, _ in sorted_top_features[:50]]
print("Top 50 features with the least RMSE:")
print(top_50_features)


# In[18]:


train, test,new_cols=better_features(train, test, 'emission', top_50_features, overall_best_score)


# <font size="3"> All the features created using arithmetic combinations are stored in a list so that the next time we do not have to identify the best combinations again and directly go ahead with the computations</font>

# In[19]:


new_cols=['cluster_5*dist_rwanda',
 'dist_rwanda/rot_30_y',
 'cluster_4+cluster_9',
 'cluster_3+cluster_10',
 'dist_lake*cluster_9',
 'cluster_8+cluster_6',
 'cluster_9/longitude',
 'cluster_1+cluster_9',
 'cluster_9/rot_15_x',
 'cluster_7/cluster_9',
 'rot_30_x/cluster_11',
 'dist_park-cluster_11',
 'cluster_10/cluster_11',
 'rot_30_y-rot_15_y',
 'cluster_0+cluster_10',
 'cluster_6-cluster_10',
 'dist_massif-cluster_10',
 'CarbonMonoxide_H2O_column_number_density_unimp_cluster_WOE/cluster_7',
 'Formaldehyde_tropospheric_HCHO_column_number_density_amf_unimp_cluster_WOE/cluster_7',
 'bx_cx_Formaldehyde_tropospheric_HCHO_column_number_density_amf+Cloud_surface_albedo_unimp_cluster_WOE',
 'y_J_Formaldehyde_tropospheric_HCHO_column_number_density_amf+Cloud_surface_albedo_unimp_cluster_WOE',
 'Formaldehyde_tropospheric_HCHO_column_number_density_amf_pca_comb-Cloud_surface_albedo_unimp_cluster_WOE',
 'log_sqrtFormaldehyde_tropospheric_HCHO_column_number_density_amf/cluster_7',
 'sqrt_Formaldehyde_tropospheric_HCHO_column_number_density_amf/cluster_7',
 'log_Formaldehyde_tropospheric_HCHO_column_number_density_amf/cluster_7',
 'Formaldehyde_tropospheric_HCHO_column_number_density_unimp_cluster_WOE/cluster_7',
 'Formaldehyde_tropospheric_HCHO_column_number_density_amf/cluster_7',
 'cluster_10/cluster_7',
 'cluster_10/rot_15_y',
 'log_CarbonMonoxide_H2O_column_number_density-log_sqrtCloud_surface_albedo',
 'CarbonMonoxide_H2O_column_number_density-log_sqrtCloud_surface_albedo',
 'Cloud_surface_albedo_unimp_cluster_WOE*rot_15_y',
 'sqrt_CarbonMonoxide_H2O_column_number_density-sqrt_Cloud_surface_albedo',
 'log_sqrtCarbonMonoxide_H2O_column_number_density/rot_15_y',
 'log_sqrtCarbonMonoxide_H2O_column_number_density/sqrt_Cloud_surface_albedo',
 'log_sqrtCloud_surface_albedo/sqrt_Cloud_surface_albedo',
 'y_J_CarbonMonoxide_H2O_column_number_density-sqrt_Cloud_surface_albedo',
 'CarbonMonoxide_H2O_column_number_density_pca_comb+sqrt_Cloud_surface_albedo',
 'bx_cx_CarbonMonoxide_H2O_column_number_density-sqrt_Cloud_surface_albedo']


# In[20]:


def apply_arithmetic_operations(train_df, test_df, expressions_list):
    for expression in expressions_list:
        # Split the expression based on operators (+, -, *, /)
        parts = expression.split('+') if '+' in expression else \
                expression.split('-') if '-' in expression else \
                expression.split('*') if '*' in expression else \
                expression.split('/')
        
        # Get the DataFrame column names involved in the operation
        cols = [col for col in parts]
        
        # Perform the corresponding arithmetic operation based on the operator in the expression
        if cols[0] in train.columns and cols[1] in train.columns:
            if '+' in expression:
                train_df[expression] = train_df[cols[0]] + train_df[cols[1]]
                test_df[expression] = test_df[cols[0]] + test_df[cols[1]]
            elif '-' in expression:
                train_df[expression] = train_df[cols[0]] - train_df[cols[1]]
                test_df[expression] = test_df[cols[0]] - test_df[cols[1]]
            elif '*' in expression:
                train_df[expression] = train_df[cols[0]] * train_df[cols[1]]
                test_df[expression] = test_df[cols[0]] * test_df[cols[1]]
            elif '/' in expression:
                train_df[expression] = train_df[cols[0]] / (train_df[cols[1]]+1e-5)
                test_df[expression] = test_df[cols[0]] /( test_df[cols[1]]+1e-5)
    
    return train_df, test_df

# train, test = apply_arithmetic_operations(train, test, new_cols)


# ## 4.4 Feature Selection

# **Steps to Eliminate Correlated Features**:<font size="3"> A lot of features have been created from the parent features using transformations and Clustering techniques which would be correlated to an extent. We will have to identify the best features among them and eliminate the rest</font>
# 1. <font size="3">Group features based on their parent feature. For example, all features derived from SO2 come under one set</font>
# 2. <font size="3">Apply PCA on the set to create a single PC1 component and Cluster-Target Encoding on the set</font>
# 3. <font size="3">See the performance of each feature in the set along with the new featires from PCA & Clustering with a cross-validated single feature-target model</font>
# 4. <font size="3">Select the feature with highest CV-RMSE</font>

# In[21]:


# Drop all the unimportant features
train=train.drop(columns=unimportant_features)
test=test.drop(columns=unimportant_features)


# In[22]:


final_drop_list=[]

table = PrettyTable()
table.field_names = ['Original', 'Final Transformation', 'RMSE CV']
threshold=0.95
# It is possible that multiple parent features share same child features, so store selected features to avoid selecting the same feature again
best_cols=[]

for col in cont_cols:
    sub_set=[f for f in train.columns if col in f and train[f].nunique()>2]
#     print(sub_set)
    if len(sub_set)>2:
        correlated_features = []

        for i, feature in enumerate(sub_set):
            # Check correlation with all remaining features
            for j in range(i+1, len(sub_set)):
                correlation = np.abs(train[feature].corr(train[sub_set[j]]))
                # If correlation is greater than threshold, add to list of highly correlated features
                if correlation > threshold:
                    correlated_features.append(sub_set[j])

        # Remove duplicate features from the list
        correlated_features = list(set(correlated_features))
#         print(correlated_features)
        if len(correlated_features)>=2:

            temp_train=train[correlated_features]
            temp_test=test[correlated_features]
            #Scale before applying PCA
            sc=StandardScaler()
            temp_train=sc.fit_transform(temp_train)
            temp_test=sc.transform(temp_test)

            # Initiate PCA
            pca=TruncatedSVD(n_components=1)
            x_pca_train=pca.fit_transform(temp_train)
            x_pca_test=pca.transform(temp_test)
            x_pca_train=pd.DataFrame(x_pca_train, columns=[col+"_pca_comb_final"])
            x_pca_test=pd.DataFrame(x_pca_test, columns=[col+"_pca_comb_final"])
            train=pd.concat([train,x_pca_train],axis='columns')
            test=pd.concat([test,x_pca_test],axis='columns')

            # Clustering
            model = KMeans()
            kmeans = KMeans(n_clusters=12)
            kmeans.fit(np.array(temp_train))
            labels_train = kmeans.labels_

            train[col+'_final_cluster'] = labels_train
            test[col+'_final_cluster'] = kmeans.predict(np.array(temp_test))

#             cat_labels=cat_labels=train.groupby([col+"_final_cluster"])['emission'].mean()
#             cat_labels2=cat_labels.to_dict()
#             train[col+"_final_cluster"]=train[col+"_final_cluster"].map(cat_labels2)
#             test[col+"_final_cluster"]=test[col+"_final_cluster"].map(cat_labels2)

            correlated_features=correlated_features+[col+"_pca_comb_final",col+"_final_cluster"]

            # See which transformation along with the original is giving you the best univariate fit with target
            kf=KFold(n_splits=5, shuffle=True, random_state=42)

            scores=[]

            for f in correlated_features:
                X=train[[f]].values
                y=train["emission"].values

                mae=[]
                for train_idx, val_idx in kf.split(X,y):
                    X_train,y_train=X[train_idx],y[train_idx]
                    x_val,y_val=X[val_idx],y[val_idx]

                    model=LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(x_val)
                    score = rmse(y_val, y_pred)
                    mae.append(score)
                if f not in best_cols:
                    scores.append((f,np.mean(mae)))
            best_col, best_acc=sorted(scores, key=lambda x:x[1], reverse=False)[0]
            best_cols.append(best_col)

            cols_to_drop = [f for f in correlated_features if  f not in best_cols]
            if cols_to_drop:
                final_drop_list=final_drop_list+cols_to_drop
            table.add_row([col,best_col ,best_acc])

print(table)      


# # 5. Feature Selection

# In[23]:


feature_scale=[feature for feature in train.columns if feature not in ['emission']]

# scaler=StandardScaler()

# train[feature_scale]=scaler.fit_transform(train[feature_scale])
# test[feature_scale]=scaler.transform(test[feature_scale])


# In[24]:


X_train = train.drop(['emission'], axis=1)
y_train = train['emission']

X_test = test.copy()

print(X_train.shape, X_test.shape)


# <font size="3">There are a lot of features yet remains, I will use feature importance to select them. A union of n important features from catBoost, XGBoost, & LightGBM are considered below</font>

# In[25]:


def get_most_important_features(X_train, y_train, n,model_input):
    # Initialize XGBoost Regressor with specified parameters
    
    lgb_params = {
            'n_estimators': 100,
            'max_depth': 6,
            "num_leaves": 16,
            'learning_rate': 0.05,
            'subsample': 0.7,
            'colsample_bytree': 0.8,
            #'reg_alpha': 0.25,
            'reg_lambda': 5e-07,
            'objective': 'regression_l2',
            'metric': 'mean_absolute_error',
            'boosting_type': 'gbdt',
            'random_state': 42
        }
    cb_params = {
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.02,
            'l2_leaf_reg': 0.5,
            'random_strength': 0.2,
            'max_bin': 150,
            'od_wait': 80,
            'one_hot_max_size': 70,
            'grow_policy': 'Depthwise',
            'bootstrap_type': 'Bayesian',
            'od_type': 'IncToDec',
            'eval_metric': 'RMSE',
            'loss_function': 'RMSE',
            'random_state': 42
        }
    if 'xgb' in model_input:
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=50,
            learning_rate=0.02,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
            )
    elif 'cat' in model_input:
        model=CatBoostRegressor(**cb_params)
    else:
        model=lgb.LGBMRegressor(**lgb_params)
    

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses = []

    for train_idx, val_idx in kfold.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_train_fold, y_train_fold,verbose=False)

        y_pred = model.predict(X_val_fold)

        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
#         rmse = np.sqrt(mean_squared_error(np.expm1(y_val_fold), np.expm1(y_pred)))

        rmses.append(rmse)

    avg_rmse = np.mean(rmses)

    feature_importances = model.feature_importances_

    feature_importance_list = [(X_train.columns[i], importance) for i, importance in enumerate(feature_importances)]

    sorted_features = sorted(feature_importance_list, key=lambda x: x[1], reverse=True)

    top_n_features = [feature[0] for feature in sorted_features[:n]]
    print(avg_rmse)
    return top_n_features


# In[26]:


n_imp_features_cat=get_most_important_features(X_train, y_train,10, 'cat')
n_imp_features_xgb=get_most_important_features(X_train, y_train,30, 'xgb')
n_imp_features_lgbm=get_most_important_features(X_train, y_train,30, 'lgbm')


# In[27]:


n_imp_features=[*set(n_imp_features_xgb+n_imp_features_lgbm+n_imp_features_cat)]
print(f"{len(n_imp_features)} features have been selected from three algorithms for the final model")


# In[28]:


X_train=train[n_imp_features]
X_test=test[n_imp_features]


# # 6. Modeling

# <font size="3">Kudos to [tetsutani](http://www.kaggle.com/code/tetsutani/ps3e11-eda-xgb-lgbm-cat-ensemble-lb-0-29267) for a great modeling framework from where the below parts are adopted, please support the page if you like my work</font>

# ## 6.1 Model Selection

# In[29]:


class Splitter:
    def __init__(self, test_size=0.2, kfold=True, n_splits=5):
        self.test_size = test_size
        self.kfold = kfold
        self.n_splits = n_splits

    def split_data(self, X, y, random_state_list):
        if self.kfold:
            for random_state in random_state_list:
                kf = KFold(n_splits=self.n_splits, random_state=random_state, shuffle=True)
                for train_index, val_index in kf.split(X, y):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                    yield X_train, X_val, y_train, y_val
        else:
            for random_state in random_state_list:
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.test_size, random_state=random_state)
                yield X_train, X_val, y_train, y_val

class Regressor:
    def __init__(self, n_estimators=100, device="gpu", random_state=0):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.reg_models = self._define_reg_model()
        self.len_models = len(self.reg_models)
        
    def _define_reg_model(self):
        
        xgb_params = {
            'n_estimators': self.n_estimators,
            'max_depth': 6,
            'learning_rate': 0.0116,
            'colsample_bytree': 1,
            'subsample': 0.6085,
            'min_child_weight': 9,
            'reg_lambda': 4.879e-07,
            'max_bin': 431,
            #'booster': 'dart',
            'n_jobs': -1,
            'eval_metric': 'rmse',
            'objective': "reg:squarederror",
            #'tree_method': 'hist',
            'verbosity': 0,
            'random_state': self.random_state,
        }
        if self.device == 'gpu':
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['predictor'] = 'gpu_predictor'
        
        xgb_params1=xgb_params.copy()
        xgb_params1['subsample']=0.7
        xgb_params1['max_depth']=6
#         xgb_params1['learning_rate']=0.01
        xgb_params1['colsample_bytree']=0.6

        xgb_params2=xgb_params.copy()
        xgb_params2['subsample']=0.5
        xgb_params2['max_depth']=8
        xgb_params2['learning_rate']=0.047
        xgb_params2['colsample_bytree']=0.9
        xgb_params2['tree_method']='approx'

        lgb_params = {
            'n_estimators': self.n_estimators,
            'max_depth': 6,
            "num_leaves": 16,
            'learning_rate': 0.05,
            'subsample': 0.7,
            'colsample_bytree': 0.8,
            #'reg_alpha': 0.25,
            'reg_lambda': 5e-07,
            'objective': 'regression_l2',
            'metric': 'mean_squared_error',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state
        }
        lgb_params1=lgb_params.copy()
        lgb_params1['subsample']=0.9
        lgb_params1['reg_lambda']=0.8994221730208598
        lgb_params1['reg_alpha']=0.6236579699090548
        lgb_params1['max_depth']=6
        lgb_params1['learning_rate']=0.01
        lgb_params1['colsample_bytree']=0.5

        lgb_params2=lgb_params.copy()
        lgb_params2['subsample']=0.1
        lgb_params2['reg_lambda']=0.5940716788024517
        lgb_params2['reg_alpha']=0.4300477974434703
        lgb_params2['max_depth']=8
        lgb_params2['learning_rate']=0.019000000000000003
        lgb_params2['colsample_bytree']=0.8
        lgb_params3 = {
            'n_estimators': self.n_estimators,
            'num_leaves': 45,
            'max_depth': 5,
            'learning_rate': 0.0684383311038932,
            'subsample': 0.5758412171285148,
            'colsample_bytree': 0.8599714680300794,
            'reg_lambda': 1.597717830931487e-08,
            'objective': 'regression_l2',
            'metric': 'mean_squared_error',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state,
            'verbosity': 0,
            'force_col_wise': True
        }
        lgb_params4=lgb_params.copy()
        lgb_params4['subsample']=0.3
        lgb_params4['reg_lambda']=0.5488355125638069
        lgb_params4['reg_alpha']=0.23414681424407247
        lgb_params4['max_depth']=7
        lgb_params4['learning_rate']=0.019000000000000003
        lgb_params4['colsample_bytree']=0.5

        cb_params = {
            'iterations': self.n_estimators,
            'depth': 6,
            'learning_rate': 0.02,
            'l2_leaf_reg': 0.5,
            'random_strength': 0.2,
            'max_bin': 150,
            'od_wait': 80,
            'one_hot_max_size': 70,
            'grow_policy': 'Depthwise',
            'bootstrap_type': 'Bayesian',
            'od_type': 'IncToDec',
            'eval_metric': 'RMSE',
            'loss_function': 'RMSE',
            'task_type': self.device.upper(),
            'random_state': self.random_state
        }
        cb_sym_params = cb_params.copy()
        cb_sym_params['grow_policy'] = 'SymmetricTree'
        cb_loss_params = cb_params.copy()
        cb_loss_params['grow_policy'] = 'Lossguide'
    
        cb_params1 = {
            'iterations': self.n_estimators,
            'depth': 8,
            'learning_rate': 0.01,
            'l2_leaf_reg': 0.1,
            'random_strength': 0.2,
            'max_bin': 150,
            'od_wait': 50,
            'one_hot_max_size': 70,
            'grow_policy': 'Depthwise',
            'bootstrap_type': 'Bernoulli',
            'od_type': 'Iter',
            'eval_metric': 'RMSE',
            'loss_function': 'RMSE',
            'task_type': self.device.upper(),
            'random_state': self.random_state
        }
        cb_params2= {
            'n_estimators': self.n_estimators,
            'depth': 10,
            'learning_rate': 0.08827842054729117,
            'l2_leaf_reg': 4.8351074756668864e-05,
            'random_strength': 0.21306687539993183,
            'max_bin': 483,
            'od_wait': 97,
            'grow_policy': 'Lossguide',
            'bootstrap_type': 'Bayesian',
            'od_type': 'Iter',
            'eval_metric': 'RMSE',
            'loss_function': 'RMSE',
            'task_type': self.device.upper(),
            'random_state': self.random_state,
            'silent': True
        }
        dt_params= {'min_samples_split': 8, 'min_samples_leaf': 4, 'max_depth': 16, 'criterion': 'squared_error'}
        knn_params= {'weights': 'uniform', 'p': 1, 'n_neighbors': 12, 'leaf_size': 20, 'algorithm': 'kd_tree'}

        reg_models = {
            'xgb_reg': xgb.XGBRegressor(**xgb_params),
#             'xgb_reg1': xgb.XGBRegressor(**xgb_params1),
            'xgb_reg2': xgb.XGBRegressor(**xgb_params2),
            'lgb_reg': lgb.LGBMRegressor(**lgb_params),
#             'lgb2_reg': lgb.LGBMRegressor(**lgb_params1),
            'lgb3_reg': lgb.LGBMRegressor(**lgb_params2),
#             'lgb4_reg': lgb.LGBMRegressor(**lgb_params3),
            'lgb5_reg': lgb.LGBMRegressor(**lgb_params4),
            "hgbm": HistGradientBoostingRegressor(max_iter=self.n_estimators, learning_rate=0.01, loss="squared_error", 
                                                  n_iter_no_change=300,random_state=self.random_state),
            'cat_reg': CatBoostRegressor(**cb_params),
#             'cat_reg2': CatBoostRegressor(**cb_params1),
#             'cat_reg3': CatBoostRegressor(**cb_params2),
            "cat_sym": CatBoostRegressor(**cb_sym_params),
#             "cat_loss": CatBoostRegressor(**cb_loss_params),
            'etr': ExtraTreesRegressor(min_samples_split=12, min_samples_leaf= 6, max_depth=16,
                                       n_estimators=500,random_state=self.random_state),
#             'ann':ann,
            "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,loss="squared_error", random_state=self.random_state),
#             "RandomForestRegressor": RandomForestRegressor(max_depth= 6,max_features= 'auto',min_samples_split= 4,
#                                                            min_samples_leaf= 4,  n_estimators=500, random_state=self.random_state, n_jobs=-1),
            'dt': DecisionTreeRegressor(**dt_params),
            
#             "lr":LinearRegression(),
#             "knn":KNeighborsRegressor(**knn_params),
#             "PassiveAggressiveRegressor": PassiveAggressiveRegressor(max_iter=3000, tol=1e-3, n_iter_no_change=30, random_state=self.random_state),
#             "HuberRegressor": HuberRegressor(max_iter=3000),

            
            
            
        }


        return reg_models


# ## 6.2 Weighted Esembling 

# In[30]:


class OptunaWeights:
    def __init__(self, random_state):
        self.study = None
        self.weights = None
        self.random_state = random_state

    def _objective(self, trial, y_true, y_preds):
        # Define the weights for the predictions from each model
        weights = [trial.suggest_float(f"weight{n}", 0, 1) for n in range(len(y_preds))]

        # Calculate the weighted prediction
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=weights)

        # Calculate the RMSE score for the weighted prediction
        score = rmse(y_true, weighted_pred)
        return score

    def fit(self, y_true, y_preds, n_trials=1000):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        self.study = optuna.create_study(sampler=sampler, study_name="OptunaWeights", direction='minimize')
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        self.study.optimize(objective_partial, n_trials=n_trials)
        self.weights = [self.study.best_params[f"weight{n}"] for n in range(len(y_preds))]

    def predict(self, y_preds):
        assert self.weights is not None, 'OptunaWeights error, must be fitted before predict'
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=self.weights)
        return weighted_pred

    def fit_predict(self, y_true, y_preds, n_trials=1000):
        self.fit(y_true, y_preds, n_trials=n_trials)
        return self.predict(y_preds)
    
    def weights(self):
        return self.weights


# ## 6.3 Model Fit

# In[31]:


kfold = True
n_splits = 1 if not kfold else 5
random_state = 42
random_state_list = [42] 
n_estimators = 9999 
early_stopping_rounds = 600
verbose = False
device = 'cpu'

splitter = Splitter(kfold=kfold, n_splits=n_splits)


# Initialize an array for storing test predictions
test_predss = np.zeros(X_test.shape[0])
ensemble_score = []
weights = []
trained_models = dict(zip(Regressor().reg_models.keys(), [[] for _ in range(Regressor().len_models)]))
trained_models = {'lgb_reg':[]}

# Evaluate on validation data and store predictions on test data
for i, (X_train_, X_val, y_train_, y_val) in enumerate(splitter.split_data(X_train, y_train, random_state_list=random_state_list)):
    n = i % n_splits
    m = i // n_splits

    # Get a set of Regressor models
    reg = Regressor(n_estimators, device, random_state)
    models = reg.reg_models

    # Initialize lists to store oof and test predictions for each base model
    oof_preds = []
    test_preds = []

    # Loop over each base model and fit it to the training data, evaluate on validation data, and store predictions
    for name, model in models.items():
        if name in ['cat_reg','lgb_reg','cat_sym']:
            model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds, verbose=verbose)
        elif name=='hgbm':
            model.fit(X_train_, y_train_)
        elif name=='ann':
            model.fit(X_train_, y_train_, validation_data=(X_val, y_val),batch_size=2, epochs=60,verbose=verbose)
        else:
            model.fit(X_train_, y_train_)
        if name=="ann":
            y_val_pred = model.predict(X_val)[:,0]
            test_pred = model.predict(X_test)[:,0]
        else:
            y_val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)

        

#         # Convert predicted values back to their original scale by applying the expm1 function
#         y_val_pred = np.expm1(y_val_pred)
#         test_pred = np.expm1(test_pred)

        score = rmse(y_val, y_val_pred)
#         score = rmse(np.expm1(y_val), y_val_pred)

        print(f'{name} [FOLD-{n} SEED-{random_state_list[m]}] RMSE score: {score:.5f}')

        oof_preds.append(y_val_pred)
        test_preds.append(test_pred)
        if name in trained_models.keys():
            trained_models[f'{name}'].append(deepcopy(model))

    # Use Optuna to find the best ensemble weights
    optweights = OptunaWeights(random_state=random_state)
    y_val_pred = optweights.fit_predict(y_val.values, oof_preds)
#     y_val_pred = optweights.fit_predict(np.expm1(y_val.values), oof_preds)

    score = rmse(y_val, y_val_pred)
#     score = rmse(np.expm1(y_val), y_val_pred)

    print(f'Ensemble [FOLD-{n} SEED-{random_state_list[m]}] RMSE score -------> {score:.5f}')
    ensemble_score.append(score)
    weights.append(optweights.weights)
    test_predss += optweights.predict(test_preds) / (n_splits * len(random_state_list))

    gc.collect()


# In[32]:


mean_score = np.mean(ensemble_score)
std_score = np.std(ensemble_score)
print(f'Ensemble RMSE score {mean_score:.5f} ± {std_score:.5f}')

# Print the mean and standard deviation of the ensemble weights for each model
print('--- Model Weights ---')
mean_weights = np.mean(weights, axis=0)
std_weights = np.std(weights, axis=0)
for name, mean_weight, std_weight in zip(models.keys(), mean_weights, std_weights):
    print(f'{name} {mean_weight:.5f} ± {std_weight:.5f}')


# ## 6.4 Feature Importance

# In[33]:


def visualize_importance(models, feature_cols, title, top=25):
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
    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=feature_importance, color='grey', errorbar='sd')
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.title(f'{title} Feature Importance [Top {top}]', fontsize=15)
    plt.grid(True, axis='x')
    plt.show()
    
for name, models in trained_models.items():
    visualize_importance(models, list(X_train.columns), name)


# # 7. Results- Post Processing

# <font size="3">Amplifying the results using the findings by [patrick0302](https://www.kaggle.com/code/patrick0302/find-and-fix-the-error-bug/notebook)</font>

# In[34]:


sub = pd.read_csv('/kaggle/input/playground-series-s3e20/sample_submission.csv')
sub['emission'] = test_predss
sub["emission"] = sub["emission"].apply(lambda x: max(x, 0)) #ensure non negative predictions
sub['latitude'] = test['latitude']
sub['longitude'] = test['longitude']
sub['emission_avg'] = sub.groupby(['latitude','longitude']).transform('mean')

# Scale up those locations with high emissions (top 50%) by 1.05
multiplier = 1.05
sub.loc[sub['emission_avg']>sub['emission_avg'].median(), 'emission'] = sub.loc[sub['emission_avg']>sub['emission_avg'].median(), 'emission']*multiplier

sub.loc[test_copy['longitude']==29.321, 'emission'] = train_copy.loc[(train_copy['year']==2021)&(train_copy['week_no']<=48)&(train_copy['longitude']==29.321),'emission'].values


# <font size="3">Also get the max emissions from years 2019, 2020, & 2021. This is based on approach by [danbraswell](https://www.kaggle.com/code/danbraswell/no-ml-public-lb-23-02231) . If the predictions are lesser than the maximum emissions occured in the previous three years, this could be replaced with the maximum emission</font>

# In[35]:


if "lat_lon" not in train_copy.columns:
    train_copy.insert(2,"lat_lon", list(zip(train["latitude"],train["longitude"])))

# Location of each station
locations = train_copy["lat_lon"].unique()

# Function to get emissions for specified year and location. Only use first 49 weeks.
def get_emissions_loc_year( loc, year ):
    df = train_copy[(train_copy["lat_lon"]==loc) & (train_copy["year"]==year) & (train_copy["week_no"]<49)].copy()
    return df["emission"].values

def get_emissions_max( loc ):
    emiss2019 = get_emissions_loc_year(loc,2019)
    emiss2020 = get_emissions_loc_year(loc,2020)
    emiss2021 = get_emissions_loc_year(loc,2021)
    return np.max([emiss2019,emiss2020,emiss2021],axis=0)

predictions_acc = []
for loc in locations:
    emission = get_emissions_max( loc )
    predictions_acc.append( emission )
#
predictions_2022 = np.hstack(predictions_acc)

sub["emission"]=np.maximum(predictions_2022,np.array(sub["emission"]))

# For longitude 28.467 and latitude 1.833, there were no emissions across all datapoints
sub.loc[(test_copy['longitude']==29.321) & (test_copy['longitude']==1.833), 'emission'] =0
sub = sub[['ID_LAT_LON_YEAR_WEEK','emission']]


# In[36]:


sub.to_csv('submission.csv',index=False)
sub.head()

