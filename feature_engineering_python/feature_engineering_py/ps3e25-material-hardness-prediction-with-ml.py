#!/usr/bin/env python
# coding: utf-8

# # 1. INTRODUCTION
# <center>
# <img src="https://mcstaging-blog.astteria.com/wp-content/uploads/2022/08/image2-3.jpg" width=1300 height=800 />
# </center>

# **PROBLEM STATEMENT: PREDICTION OF MOHS HARDNESS WITH MACHINE LEARNING**
# 
# <font size="3">The problem addressed in this study revolves around predicting the Mohs hardness of minerals using a machine learning approach. Hardness, a critical property in materials design, is non-destructively assessed through hardness testing. The study proposes a model that integrates atomic and electronic features derived from composition across various mineral compositions and crystal systems. The dataset, sourced from experimental Mohs hardness data, crystal classes, and chemical compositions of minerals, consists of 369 uniquely named minerals. Compositional permutations were performed to handle multiple composition combinations for minerals with the same name, resulting in a database of 622 minerals.</font>
# 
# **METRIC OF EVALUATION** MEDIAN ABSOLUTE ERROR
# 
#         MedAE=median(|y_actual(i)-y_predicted(i)|,......, |y_actual(n)-y_predicted(n)|)
# 

# # 2. IMPORTS

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
from sklearn.metrics import mean_squared_error,median_absolute_error, mean_absolute_error
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
get_ipython().system('pip install cmaes')
import cmaes
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


# ## 2.1 Data

# In[2]:


global device
device = 'cpu'


train=pd.read_csv('/kaggle/input/playground-series-s3e25/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s3e25/test.csv')
original=pd.read_csv("/kaggle/input/prediction-of-mohs-hardness-with-machine-learning/jm79zfps6b-1/Artificial_Crystals_Dataset.csv")

train.drop(columns=["id"],inplace=True)
test.drop(columns=["id"],inplace=True)

crystal_structure=original.copy()
original.drop(columns=['Unnamed: 0','Formula','Crystal structure'],inplace=True)

original=original.rename(columns={'Hardness (Mohs)':"Hardness"})
train_copy=train.copy()
test_copy=test.copy()
original_copy=original.copy()

original["original"]=1

train["original"]=0
test["original"]=0

train=pd.concat([train,original],axis=0)
train.reset_index(inplace=True,drop=True)
train.head()


# **NOTE:** <font size='3'>The original dataset has crystal structures and the formula of the compound. This makes easier to measure hardness, cubis crystal structure is the hardest, diamond is the best example for that. </font>

# ## 2.2 Missing Values

# In[3]:


table = PrettyTable()

table.field_names = ['Feature', 'Data Type', 'Train Missing %', 'Test Missing %',"Original Missing%"]
for column in train_copy.columns:
    data_type = str(train_copy[column].dtype)
    non_null_count_train= np.round(100-train_copy[column].count()/train_copy.shape[0]*100,1)
    if column!='Hardness':
        non_null_count_test = np.round(100-test_copy[column].count()/test_copy.shape[0]*100,1)
    else:
        non_null_count_test="NA"
    non_null_count_orig= np.round(100-original_copy[column].count()/original_copy.shape[0]*100,1)
    table.add_row([column, data_type, non_null_count_train,non_null_count_test,non_null_count_orig])
print(table)


# <font size='3'>No missing values in the dataset</font>

# # 3. EXPLORATORY DATA ANALYSIS

# ## 3.1 Target Analysis

# In[4]:


# train=train[(train['Hardness']>2)& (train['Hardness']<6)]
# train=train.reset_index(drop=True)


# In[5]:


sns.set(style="whitegrid")
emission = train['Hardness']

mean_emission = np.mean(emission)
median_emission = np.median(emission)

fig, ax = plt.subplots(figsize=(12, 5))

ax.hist(emission, bins=20, density=True, alpha=0.5, label='Hardness Histogram', color=sns.color_palette("Blues")[4])

x_values = np.linspace(emission.min(), emission.max(), len(emission))
density_values = (1 / (np.sqrt(2 * np.pi) * np.std(emission))) * np.exp(-0.5 * ((x_values - mean_emission) / np.std(emission))**2)
ax.plot(x_values, density_values, color=sns.color_palette("Reds")[4], label='Hardness Density')

ax.axvline(mean_emission, color=sns.color_palette("Greens")[4], linestyle='dashed', linewidth=2, label='Mean Hardness')
ax.axvline(median_emission, color=sns.color_palette("Oranges")[4], linestyle='dashed', linewidth=2, label='Median Hardness')

ax.set_xlabel('Hardness')
ax.set_ylabel('Frequency / Density')
ax.set_title('Hardness Distribution and Density Plot')

x_min = emission.min()
x_max = emission.max()
ax.set_xlim([x_min, x_max])
ax.legend(bbox_to_anchor=(1, 1), fancybox=False, shadow=False, loc='upper left')
plt.tight_layout()
plt.show()


# **INFERENCES:** <font size="3">Now you must have understood the reason why the error metric is MedAE and not MAE. Also, the maximum hardness value is 10 and that belongs to diamond and 1 is the smallest which likely is Talc. A lot of frst principles based analysis can be incorporated in this problem statement</font>

# ## 3.2 Numerical Feature Distributions

# In[6]:


sns.set_palette("Set2")

cont_cols = [f for f in train.columns if train[f].dtype in [float, int] and train[f].nunique() > 2 and f not in ['Hardness']]

num_rows = (len(cont_cols) + 2) // 3

fig, axs = plt.subplots(num_rows, 3, figsize=(15, num_rows * 4))

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
    sns.histplot(train[col], ax=axs[row][col_pos], kde=True, label='Train', bins=num_bins_train)
    sns.histplot(test[col], ax=axs[row][col_pos], kde=True, label='Test', bins=num_bins_test)
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


# # 4. FEATURE ENGINEERING

# ### Data Preprocessing

# In[7]:


def data_process(df):
    df['n_elements']=df['allelectrons_Total']/(df['allelectrons_Average']+1e-5)
    
    df['total_weight']=df['n_elements']*df['atomicweight_Average']
    
    return df

train=data_process(train)
test=data_process(test)
crystal_structure=data_process(crystal_structure)   


# ### Basic Functions

# In[8]:


def min_max_scaler(train, test, column):
    '''
    Min Max just based on train might have an issue if test has extreme values, hence changing the denominator uding overall min and max
    '''
    sc=MinMaxScaler()
    
    max_val=max(train[column].max(),test[column].max())
    min_val=min(train[column].min(),test[column].min())

    train[column]=(train[column]-min_val)/(max_val-min_val)
    test[column]=(test[column]-min_val)/(max_val-min_val)
    
    return train,test  

def OHE(train_df,test_df,cols,target):
    '''
    Function for one hot encoding, it first combines the data so that no category is missed and
    the category with least frequency can be dropped because of redundancy
    '''
    combined = pd.concat([train_df, test_df], axis=0)
    for col in cols:
        one_hot = pd.get_dummies(combined[col])
        counts = combined[col].value_counts()
        min_count_category = counts.idxmin()
        one_hot = one_hot.drop(min_count_category, axis=1)
        one_hot.columns=[str(f)+col+"_OHE" for f in one_hot.columns]
        combined = pd.concat([combined, one_hot], axis="columns")
        combined = combined.loc[:, ~combined.columns.duplicated()]
    
    # split back to train and test dataframes
    train_ohe = combined[:len(train_df)]
    test_ohe = combined[len(train_df):]
    test_ohe.reset_index(inplace=True,drop=True)
    test_ohe.drop(columns=[target],inplace=True)
    return train_ohe, test_ohe

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
            'random_state': 42,
            'device': device,
        }
def rmse(y1,y2):
    ''' Median Absolute Error Evaluator'''
    return(np.sqrt(mean_squared_error(np.array(y1),np.array(y2))))

def med_abs_error(y1,y2):
    ''' Median Absolute Error Evaluator'''
    return median_absolute_error(np.array(y1),np.array(y2))

def store_missing_rows(df, features):
    '''Function stores where missing values are located for given set of features'''
    missing_rows = {}
    
    for feature in features:
        missing_rows[feature] = df[df[feature].isnull()]
    
    return missing_rows

def fill_missing_numerical(train,test,target, max_iterations=10):
    '''Iterative Missing Imputer: Updates filled missing values iteratively using CatBoost Algorithm'''
    train_temp=train.copy()
    if target in train_temp.columns:
        train_temp=train_temp.drop(columns=target)
        
    
    df=pd.concat([train_temp,test],axis="rows")
    df=df.reset_index(drop=True)
    features=[ f for f in df.columns if df[f].isna().sum()>0]
    if len(features)>0:
        # Step 1: Store the instances with missing values in each feature
        missing_rows = store_missing_rows(df, features)

        # Step 2: Initially fill all missing values with "Missing"
        for f in features:
            df[f]=df[f].fillna(df[f].mean())

        cat_features=[f for f in df.columns if not pd.api.types.is_numeric_dtype(df[f])]
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

                model= lgb.LGBMRegressor(**lgb_params)
                model.fit(X_train, y_train, verbose=False)

                # Step 4: Predict missing values for the feature and update all N features
                y_pred = model.predict(missing_temp)
                df.loc[rows_miss, feature] = y_pred
                error_minimize=med_abs_error(y_pred,y_pred_prev)
                dictionary[feature].append(error_minimize)  # Append the error_minimize value

#         for feature, values in dictionary.items():
#             iterations = range(1, len(values) + 1)  # x-axis values (iterations)
#             plt.plot(iterations, values, label=feature)  # plot the values
#             plt.xlabel('Iterations')
#             plt.ylabel('RMSE')
#             plt.title('Minimization of RMSE with iterations')
#             plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.show()
        train[features] = np.array(df.iloc[:train.shape[0]][features])
        test[features] = np.array(df.iloc[train.shape[0]:][features])

    return train,test


# # 4.1 Numerical Transformations

# <font size="3">We're going to see what transformation works better for each feature and select them, the idea is to compress the data. There could be situations where you will have to stretch the data. These are the methods applied:</font>
# 
# 1. **Log Transformation**: <font size="3">This transformation involves taking the logarithm of each data point. It is useful when the data is highly skewed and the variance increases with the mean.</font>
#                 y = log(x)
# 
# 2. **Square Root Transformation**: <font size="3">This transformation involves taking the square root of each data point. It is useful when the data is highly skewed and the variance increases with the mean.</font>
#                 y = sqrt(x)
# 
# 3. **Box-Cox Transformation**: <font size="3">This transformation is a family of power transformations that includes the log and square root transformations as special cases. It is useful when the data is highly skewed and the variance increases with the mean.</font>
#                 y = [(x^lambda) - 1] / lambda if lambda != 0
#                 y = log(x) if lambda = 0
# 
# 4. **Yeo-Johnson Transformation**: <font size="3">This transformation is similar to the Box-Cox transformation, but it can be applied to both positive and negative values. It is useful when the data is highly skewed and the variance increases with the mean.</font>
#                 y = [(|x|^lambda) - 1] / lambda if x >= 0, lambda != 0
#                 y = log(|x|) if x >= 0, lambda = 0
#                 y = -[(|x|^lambda) - 1] / lambda if x < 0, lambda != 2
#                 y = -log(|x|) if x < 0, lambda = 2
# 
# 5. **Power Transformation**: <font size="3">This transformation involves raising each data point to a power. It is useful when the data is highly skewed and the variance increases with the mean. The power can be any value, and is often determined using statistical methods such as the Box-Cox or Yeo-Johnson transformations.</font>
#                 y = [(x^lambda) - 1] / lambda if method = "box-cox" and lambda != 0
#                 y = log(x) if method = "box-cox" and lambda = 0
#                 y = [(x + 1)^lambda - 1] / lambda if method = "yeo-johnson" and x >= 0, lambda != 0
#                 y = log(x + 1) if method = "yeo-johnson" and x >= 0, lambda = 0
#                 y = [-(|x| + 1)^lambda - 1] / lambda if method = "yeo-johnson" and x < 0, lambda != 2
#                 y = -log(|x| + 1) if method = "yeo-johnson" and x < 0, lambda = 2

# In[9]:


cont_cols = [f for f in train.columns if pd.api.types.is_numeric_dtype(train[f]) and train[f].nunique() >2 and f not in ['Hardness']]

sc=MinMaxScaler()

global unimportant_features
global overall_best_score
global overall_best_col
unimportant_features=[]
overall_best_score=100
overall_best_col='none'
dt_params={'min_samples_split': 8, 'min_samples_leaf': 4, 'max_depth': 8, 'criterion': 'absolute_error'}


def transformer(train, test,cont_cols, target):
    '''
    Algorithm applies multiples transformations on selected columns and finds the best transformation using a single variable model performance
    '''
    global unimportant_features
    global overall_best_score
    global overall_best_col
    train_copy = train.copy()
    test_copy = test.copy()
    table = PrettyTable()
    table.field_names = ['Feature', 'Initial MedAE', 'Transformation', 'Tranformed MedAE']

    for col in cont_cols:
        train, test=min_max_scaler(train, test, col)
        for c in ["log_"+col, "sqrt_"+col, "bx_cx_"+col, "y_J_"+col, "log_sqrt"+col, "pow_"+col, "pow2_"+col]:
            if c in train_copy.columns:
                train_copy = train_copy.drop(columns=[c])
        
        # Log Transformation after MinMax Scaling (keeps data between 0 and 1)
        train_copy["log_"+col] = np.log1p(train_copy[col])
        test_copy["log_"+col] = np.log1p(test_copy[col])
        
        # Square Root Transformation
        train_copy["sqrt_"+col] = np.sqrt(train_copy[col])
        test_copy["sqrt_"+col] = np.sqrt(test_copy[col])
        
        # Box-Cox transformation
        combined_data = pd.concat([train_copy[[col]], test_copy[[col]]], axis=0)
        epsilon = 1e-5
        transformer = PowerTransformer(method='box-cox')
        scaled_data = transformer.fit_transform(combined_data + epsilon)

        train_copy["bx_cx_" + col] = scaled_data[:train_copy.shape[0]]
        test_copy["bx_cx_" + col] = scaled_data[train_copy.shape[0]:]
        # Yeo-Johnson transformation
        transformer = PowerTransformer(method='yeo-johnson')
        train_copy["y_J_"+col] = transformer.fit_transform(train_copy[[col]])
        test_copy["y_J_"+col] = transformer.transform(test_copy[[col]])
        
        # Power transformation, 0.25
        power_transform = lambda x: np.power(x + 1 - np.min(x), 0.25)
        transformer = FunctionTransformer(power_transform)
        train_copy["pow_"+col] = transformer.fit_transform(train_copy[[col]])
        test_copy["pow_"+col] = transformer.transform(test_copy[[col]])
        
        # Power transformation, 2
        power_transform = lambda x: np.power(x + 1 - np.min(x), 2)
        transformer = FunctionTransformer(power_transform)
        train_copy["pow2_"+col] = transformer.fit_transform(train_copy[[col]])
        test_copy["pow2_"+col] = transformer.transform(test_copy[[col]])
        
        # Log to power transformation
        train_copy["log_sqrt"+col] = np.log1p(train_copy["sqrt_"+col])
        test_copy["log_sqrt"+col] = np.log1p(test_copy["sqrt_"+col])
        
        temp_cols = [col, "log_"+col, "sqrt_"+col, "bx_cx_"+col, "y_J_"+col,  "pow_"+col , "pow2_"+col,"log_sqrt"+col]
        
        train_copy,test_copy = fill_missing_numerical(train_copy,test_copy,target,5)
#         train_copy[temp_cols] = train_copy[temp_cols].fillna(0)
#         test_copy[temp_cols] = test_copy[temp_cols].fillna(0)
        
        pca = TruncatedSVD(n_components=1)
        x_pca_train = pca.fit_transform(train_copy[temp_cols])
        x_pca_test = pca.transform(test_copy[temp_cols])
        x_pca_train = pd.DataFrame(x_pca_train, columns=[col+"_pca_comb"])
        x_pca_test = pd.DataFrame(x_pca_test, columns=[col+"_pca_comb"])
        temp_cols.append(col+"_pca_comb")
        
        test_copy = test_copy.reset_index(drop=True)
        
        train_copy = pd.concat([train_copy, x_pca_train], axis='columns')
        test_copy = pd.concat([test_copy, x_pca_test], axis='columns')
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        mae_scores = []
        
        for f in temp_cols:
            X = train_copy[[f]].values
            y = train_copy[target].values
            
            mae = []
            for train_idx, val_idx in kf.split(X, y):
                X_train, y_train = X[train_idx], y[train_idx]
                x_val, y_val = X[val_idx], y[val_idx]
                model=LinearRegression()
#                 model=DecisionTreeRegressor(**dt_params)
#                 model=HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.01, loss="absolute_error", 
#                                                   n_iter_no_change=300,random_state=42)
                model.fit(X_train,y_train)
                y_pred=model.predict(x_val)
                mae.append(med_abs_error(y_val,y_pred))
            mae_scores.append((f, np.mean(mae)))
            
            if overall_best_score > np.mean(mae):
                overall_best_score = np.mean(mae)
                overall_best_col = f

            if f == col:
                orig_mae = np.mean(mae)
                
        best_col, best_mae=sorted(mae_scores, key=lambda x:x[1], reverse=False)[0]
    
        cols_to_drop = [f for f in temp_cols if  f!= best_col and f not in col]
        final_selection=[f for f in temp_cols if f not in cols_to_drop]
        
        if cols_to_drop:
            unimportant_features = unimportant_features+cols_to_drop
        table.add_row([col,orig_mae,best_col ,best_mae])
    print(table)   
    print("overall best CV Median Absolute Error: ",overall_best_score)
    return train_copy, test_copy

train, test= transformer(train, test,cont_cols, "Hardness")


# ## 4.2 Numerical Clustering

# <font size="3"> All the unimportant features that are not the best transformation technique are selected and applied a K-Means Clustering technique</font>

# In[10]:


table = PrettyTable()
table.field_names = ['Cluster WOE Feature', 'MedAE(CV-TRAIN)']
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
    y=train["Hardness"].values

    best_mae=[]
    for train_idx, val_idx in kf.split(X,y):
        X_train,y_train=X[train_idx],y[train_idx]
        x_val,y_val=X[val_idx],y[val_idx]
        model=LinearRegression()
#         model=HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.01, loss="absolute_error", 
#                                                   n_iter_no_change=300,random_state=42)
        model.fit(X_train,y_train)
        y_pred=model.predict(x_val)
        best_mae.append(med_abs_error(y_val,y_pred))
        
    table.add_row([col+"_unimp_cluster_WOE",np.mean(best_mae)])
    if overall_best_score<np.mean(best_mae):
            overall_best_score=np.mean(best_mae)
            overall_best_col=col+"_unimp_cluster_WOE"
    
print(table)


# ## 4.3 Arithmetic Better Features

# In[11]:


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
#                 model = HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.01, loss="absolute_error", 
#                                                   n_iter_no_change=300,random_state=42)
                model=LinearRegression()
#                 model=DecisionTreeRegressor(**dt_params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = med_abs_error(y_val, y_pred)
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
                print(f"Added column '{best_col}' with mean MedAE: {best_acc:.4f} & Correlation {corr_with_other_cols.abs().max():.4f}")

    return train, test, new_cols


# In[12]:


# train, test,new_cols=better_features(train, test, 'Hardness', cont_cols, overall_best_score)


# <font size="3"> All the features created using arithmetic combinations are stored in a list so that the next time we do not have to identify the best combinations again and directly go ahead with the computations</font>

# In[13]:


new_cols=['allelectrons_Total+atomicweight_Average',
 'density_Total-atomicweight_Average',
 'allelectrons_Average-el_neg_chi_Average',
 'allelectrons_Average+zaratio_Average',
 'val_e_Average-atomicweight_Average',
 'atomicweight_Average-el_neg_chi_Average',
 'ionenergy_Average-density_Average',
 'el_neg_chi_Average-density_Average',
 'R_vdw_element_Average-density_Average',
 'R_cov_element_Average*density_Average',
 'zaratio_Average+density_Average']


# In[14]:


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

train, test = apply_arithmetic_operations(train, test, new_cols)


# ## 4.4 Feature Selection

# **Steps to Eliminate Correlated Features**:<font size="3"> A lot of features have been created from the parent features using transformations and Clustering techniques which would be correlated to an extent. We will have to identify the best features among them and eliminate the rest</font>
# 1. <font size="3">Group features based on their parent feature</font>
# 2. <font size="3">Apply PCA on the set to create a single PC1 component and Cluster-Target Encoding on the set</font>
# 3. <font size="3">See the performance of each feature in the set along with the new featires from PCA & Clustering with a cross-validated single feature-target model</font>
# 4. <font size="3">Select the feature with highest CV-MedAE</font>

# In[15]:


# train=train.drop(columns=unimportant_features)
# test=test.drop(columns=unimportant_features)


# In[16]:


final_drop_list=[]

table = PrettyTable()
table.field_names = ['Original', 'Final Transformation', 'MedAE CV']
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
                y=train["Hardness"].values

                mae=[]
                for train_idx, val_idx in kf.split(X,y):
                    X_train,y_train=X[train_idx],y[train_idx]
                    x_val,y_val=X[val_idx],y[val_idx]

                    model=LinearRegression()
#                     model=HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.01, loss="absolute_error", 
#                                                   n_iter_no_change=300,random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(x_val)
                    score = med_abs_error(y_val, y_pred)
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

# In[17]:


final_features=[feature for feature in train.columns if feature not in ['Hardness']]
final_features=[*set(final_features)]

scaler=StandardScaler()

train_scaled=train.copy()
test_scaled=test.copy()
train_scaled[final_features]=sc.fit_transform(train[final_features])
test_scaled[final_features]=sc.transform(test[final_features])


# In[18]:


def post_processor(train, test):
    '''
    After Scaling, some of the features may be the same and can be eliminated
    '''
    cols=[f for f in train.columns if "Hardness" not in f and "OHE" not in f]
    train_cop=train.copy()
    test_cop=test.copy()
    drop_cols=[]
    for i, feature in enumerate(cols):
        for j in range(i+1, len(cols)):
            if sum(abs(train_cop[feature]-train_cop[cols[j]]))==0:
                if cols[j] not in drop_cols:
                    drop_cols.append(cols[j])
    print(drop_cols)
    train_cop.drop(columns=drop_cols,inplace=True)
    test_cop.drop(columns=drop_cols,inplace=True)
    
    return train_cop, test_cop

                    
train_cop, test_cop=   post_processor(train_scaled, test_scaled)        

train_cop.to_csv('train_processed.csv',index=False)
test_cop.to_csv('test_processed.csv',index=False)


# In[19]:


X_train = train.drop(['Hardness'], axis=1)
y_train = train['Hardness']

X_test = test.copy()

print(X_train.shape, X_test.shape)


# In[20]:


def get_most_important_features(X_train, y_train, n,model_input):
    xgb_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.0116,
            'colsample_bytree': 1,
            'subsample': 0.6085,
            'min_child_weight': 9,
            'reg_lambda': 4.879e-07,
            'max_bin': 431,
            #'booster': 'dart',
            'n_jobs': -1,
            'eval_metric': 'mae',
            'objective': "reg:absoluteerror",
            #'tree_method': 'hist',
            'verbosity': 0,
            'random_state': 42,
        }
    if device == 'gpu':
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['predictor'] = 'gpu_predictor'
    lgb_params = {
            'n_estimators': 200,
            'max_depth': 6,
            "num_leaves": 16,
            'learning_rate': 0.05,
            'subsample': 0.7,
            'colsample_bytree': 0.8,
            #'reg_alpha': 0.25,
            'reg_lambda': 5e-07,
            'objective': 'mae',
            'metric': 'mean_absolute_error',
            'boosting_type': 'gbdt',
            'random_state': 42
        }
    cb_params = {
            'iterations': 300,
            'depth': 8,
            'learning_rate': 0.01,
            'random_strength': 0.2,
            'max_bin': 150,
            'od_wait': 50,
            'one_hot_max_size': 70,
            'grow_policy': 'Depthwise',
            'bootstrap_type': 'Bernoulli',
            'od_type': 'Iter',
            'eval_metric': 'MAE',
            'loss_function': 'MAE',
            'random_state': 42
        }
    
    if 'xgb' in model_input:
        model = xgb.XGBRegressor(**xgb_params)
    elif 'cat' in model_input:
        model=CatBoostRegressor(**cb_params)
    else:
        model=lgb.LGBMRegressor(**lgb_params)
        
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    maes = []
    
    feature_importances_list = []
    
    for train_idx, val_idx in kfold.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_train_fold, y_train_fold,verbose=False)

        y_pred = model.predict(X_val_fold)

        mae = med_abs_error(y_val_fold, y_pred)
        maes.append(mae)
        feature_importances = model.feature_importances_
        feature_importances_list.append(feature_importances)

    avg_mae= np.mean(maes)
    avg_feature_importances = np.mean(feature_importances_list, axis=0)

    feature_importance_list = [(X_train.columns[i], importance) for i, importance in enumerate(avg_feature_importances)]
    sorted_features = sorted(feature_importance_list, key=lambda x: x[1], reverse=True)
    top_n_features = [feature[0] for feature in sorted_features[:n]]

    display_features=top_n_features[:10]
    
    sns.set_palette("Set2")
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(display_features)), [avg_feature_importances[X_train.columns.get_loc(feature)] for feature in display_features])
    plt.yticks(range(len(display_features)), display_features, fontsize=12)
    plt.xlabel('Average Feature Importance', fontsize=14)
    plt.ylabel('Features', fontsize=10)
    plt.title(f'Top {10} of {n} Feature Importances with MedAE score {avg_mae}', fontsize=16)
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    # Add data labels on the bars
    for index, value in enumerate([avg_feature_importances[X_train.columns.get_loc(feature)] for feature in display_features]):
        plt.text(value + 0.005, index, f'{value:.3f}', fontsize=12, va='center')

    plt.tight_layout()
    plt.show()

    return top_n_features


# In[21]:


n_imp_features_cat=get_most_important_features(X_train.reset_index(drop=True), y_train,50, 'cat')
n_imp_features_xgb=get_most_important_features(X_train.reset_index(drop=True), y_train,50, 'xgb')
n_imp_features_lgbm=get_most_important_features(X_train.reset_index(drop=True), y_train,50, 'lgbm')


# In[22]:


n_imp_features=[*set(n_imp_features_xgb+n_imp_features_lgbm+n_imp_features_cat)]
print(f"{len(n_imp_features)} features have been selected from three algorithms for the final model")


# In[23]:


X_train=train[n_imp_features]
X_test=test[n_imp_features]

# X_train=train_copy.drop(columns=['Hardness'])
# y_train=train_copy['Hardness']
# X_test=test_copy.copy()
# n_imp_features=X_train.columns


# # 6. Modeling

# In[24]:


import tensorflow as tf
from tensorflow_probability import stats as tfp_stats
from tensorflow.keras import layers, callbacks

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LeakyReLU, PReLU, ELU
from keras.layers import Dropout
from keras import backend as K


# In[25]:


sgd=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.5, nesterov=True)
rms = tf.keras.optimizers.RMSprop()
nadam=tf.keras.optimizers.Nadam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam"
)
lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)


early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=100,
    restore_best_weights=True,
)


def med_abs_loss(y_true, y_pred):
    median = tfp_stats.percentile(tf.abs(y_pred - y_true), q=50.0)
    return median

def metric_fn(y_true, y_pred):
    return tfp_stats.percentile(tf.abs(y_true - y_pred), q=100) - tfp_stats.percentile(tf.abs(y_true - y_pred), q=0)


# In[26]:


ann = Sequential()
ann.add(Dense(64, input_dim=X_train.shape[1], kernel_initializer='he_uniform', activation=lrelu))
ann.add(Dropout(0.2))
ann.add(Dense(32,  kernel_initializer='he_uniform', activation=lrelu))
# ann.add(Dropout(0.1))
ann.add(Dense(4,  kernel_initializer='he_uniform', activation='relu'))
# ann.add(Dropout(0.1))
ann.add(Dense(2,  kernel_initializer='he_uniform', activation=lrelu))

ann.add(Dense(1,  kernel_initializer='he_uniform'))
ann.compile(loss=med_abs_loss, optimizer='adam')

ann2 = Sequential()
ann2.add(Dense(16, input_dim=X_train.shape[1], kernel_initializer='he_uniform', activation=lrelu))
ann2.add(Dropout(0.2))
ann2.add(Dense(4,  kernel_initializer='he_uniform', activation=lrelu))
# ann.add(Dropout(0.1))
# ann.add(Dense(4,  kernel_initializer='he_uniform', activation='relu'))
# ann.add(Dropout(0.1))
ann2.add(Dense(2,  kernel_initializer='he_uniform', activation=lrelu))

ann2.add(Dense(1,  kernel_initializer='he_uniform'))
ann2.compile(loss=med_abs_loss, optimizer='adam')


# ## 6.1 Model Selection

# <font size="3">Kudos to [tetsutani](http://www.kaggle.com/code/tetsutani/ps3e11-eda-xgb-lgbm-cat-ensemble-lb-0-29267) for a great modeling framework from where the below parts are adopted, please support the page if you like my work</font>

# In[27]:


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
            'eval_metric':'mae',
            'objective': "reg:absoluteerror",
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
        xgb_params1['eval_metric']= lambda y_true, y_pred: med_abs_error(y_true, y_pred)
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
            'objective': 'mae',
            'metric': 'mean_absolute_error',
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
            'objective': 'mae',
            'metric': 'mean_absolute_error',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state,
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
            'eval_metric': 'MAE',
            'loss_function': 'MAE',
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
            'eval_metric': 'MAE',
            'loss_function': 'MAE',
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
            'eval_metric': 'MAE',
            'loss_function': 'MAE',
            'task_type': self.device.upper(),
            'random_state': self.random_state,
            'silent': True
        }
        dt_params= {'min_samples_split': 8, 'min_samples_leaf': 4, 'max_depth': 16, 'criterion': 'absolute_error'}
        knn_params= {'weights': 'uniform', 'p': 1, 'n_neighbors': 12, 'leaf_size': 20, 'algorithm': 'kd_tree'}

        reg_models = {
            'xgb_reg': xgb.XGBRegressor(**xgb_params),
            'xgb_reg1': xgb.XGBRegressor(**xgb_params1),
            'xgb_reg2': xgb.XGBRegressor(**xgb_params2),
            'lgb_reg': lgb.LGBMRegressor(**lgb_params),
            'lgb2_reg': lgb.LGBMRegressor(**lgb_params1),
            'lgb3_reg': lgb.LGBMRegressor(**lgb_params2),
            'lgb4_reg': lgb.LGBMRegressor(**lgb_params3),
            'lgb5_reg': lgb.LGBMRegressor(**lgb_params4),
            'lgb6_reg': lgb.LGBMRegressor(),
#             "hgbm": HistGradientBoostingRegressor(max_iter=self.n_estimators, learning_rate=0.01, loss="absolute_error", 
#                                                   n_iter_no_change=300,random_state=self.random_state),
            'cat_reg': CatBoostRegressor(**cb_params),
            'cat_reg2': CatBoostRegressor(**cb_params1),
            'cat_reg3': CatBoostRegressor(**cb_params2),
            "cat_sym": CatBoostRegressor(**cb_sym_params),
            "cat_loss": CatBoostRegressor(**cb_loss_params),
#             'etr': ExtraTreesRegressor(min_samples_split=12, min_samples_leaf= 6, max_depth=16,
#                                        n_estimators=500,random_state=self.random_state),
#             'ann':ann,
#             'ann2':ann2,
#             "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=1000, learning_rate=0.02, max_depth=6,loss="absolute_error", random_state=self.random_state),
#             "RandomForestRegressor": RandomForestRegressor(max_depth= 6,max_features= 'auto',min_samples_split= 4,
#                                                            min_samples_leaf= 4,  n_estimators=500, random_state=self.random_state, n_jobs=-1),
#             'dt': DecisionTreeRegressor(**dt_params),
            
#             "lr":LinearRegression(),
#             "knn":KNeighborsRegressor(**knn_params),
#             "PassiveAggressiveRegressor": PassiveAggressiveRegressor(max_iter=3000, tol=1e-3, n_iter_no_change=30, random_state=self.random_state),
#             "HuberRegressor": HuberRegressor(max_iter=3000),

            
            
            
        }


        return reg_models


# ## 6.2 Weighted Esembling 

# In[28]:


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

        # Calculate the MedAE score for the weighted prediction
        score = med_abs_error(y_true, weighted_pred)
        return score

    def fit(self, y_true, y_preds, n_trials=10000):
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

    def fit_predict(self, y_true, y_preds, n_trials=10000):
        self.fit(y_true, y_preds, n_trials=n_trials)
        return self.predict(y_preds)
    
    def weights(self):
        return self.weights
    
def create_model(n_cols):
    
    input_layer = tf.keras.Input(shape=(n_cols, ))
    x = tf.keras.layers.BatchNormalization(epsilon=0.00001)(input_layer)
    x = tf.keras.layers.Dense(16, activation=lrelu)(x)
    #x = tf.keras.layers.BatchNormalization(epsilon=0.00001)(x)
    x = tf.keras.layers.Dense(32, activation=lrelu)(x)
    #x = tf.keras.layers.BatchNormalization(epsilon=0.00001)(x)
    output_layer = tf.keras.layers.Dense(1)(x)    
    
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(0.013, beta_1=0.5),
                  loss=med_abs_loss,
                  metrics=metric_fn)
    
    return model
callbacks_list = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=2, mode='min',restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, min_lr=0.00001),
    tf.keras.callbacks.TerminateOnNaN()
] 


# ## 6.3 Model Fit

# In[29]:


# !pip install --upgrade xgboost


# In[30]:


kfold = True
n_splits = 1 if not kfold else 4
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
        if ('cat' in name) or ("lgb" in name) or ("xgb" in name):
            model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)]\
                      , early_stopping_rounds=early_stopping_rounds, verbose=verbose)
            X_new=X_train_.copy()
            X_new['Hardness_pred'] = model.predict(X_new)
                
            model_nn = create_model(X_new.shape[1])
            if 'lgb' in name:
                history = model_nn.fit(X_new.astype('float32'), y_train_.astype('float32'),
                                    epochs=100,
                                    class_weight=model.class_weight,
                                    callbacks=callbacks_list,
                                    validation_split=0.1,verbose=verbose )
            else:
                history = model_nn.fit(X_new.astype('float32'), y_train_.astype('float32'),
                                    epochs=100,
                                    callbacks=callbacks_list,
                                    validation_split=0.1,verbose=verbose )
            
        elif 'ann' in name:
            model.fit(X_train_, y_train_, validation_data=(X_val, y_val), epochs=1000,\
                      callbacks=[early_stopping],verbose=verbose)
        else:
            model.fit(X_train_, y_train_)
            
        if 'ann' in name:
            y_val_pred = np.array(model.predict(X_val)[:,0])
            test_pred = np.array(model.predict(X_test)[:,0])
        elif ('cat' in name) or ("lgb" in name) or ("xgb" in name):
            X_val_temp=X_val.copy()
            X_val_temp['Hardness_pred']=model.predict(X_val)
            y_val_pred= model_nn.predict(X_val_temp.astype('float32'))[:,0]
            
            test_temp=X_test.copy()
            test_temp['Hardness_pred']=model.predict(test_temp)
            test_pred = model_nn.predict(test_temp.astype('float32'))[:,0]
            
            
        else:
            y_val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
        

#         # Convert predicted values back to their original scale by applying the expm1 function
#         y_val_pred = np.expm1(y_val_pred)
#         test_pred = np.expm1(test_pred)

        score = med_abs_error(y_val, y_val_pred)
#         score = med_abs_error(np.expm1(y_val), y_val_pred)

        print(f'{name} [FOLD-{n} SEED-{random_state_list[m]}] MedAE score: {score:.5f}')

        oof_preds.append(y_val_pred)
        test_preds.append(test_pred)
        if name in trained_models.keys():
            trained_models[f'{name}'].append(deepcopy(model))

    # Use Optuna to find the best ensemble weights
    optweights = OptunaWeights(random_state=random_state)
    y_val_pred = optweights.fit_predict(y_val.values, oof_preds)
#     y_val_pred = optweights.fit_predict(np.expm1(y_val.values), oof_preds)

    score = med_abs_error(y_val, y_val_pred)
#     score = med_abs_error(np.expm1(y_val), y_val_pred)

    print(f'Ensemble [FOLD-{n} SEED-{random_state_list[m]}] MedAE score ------------------------> {score:.5f}')
    ensemble_score.append(score)
    weights.append(optweights.weights)
    test_predss += optweights.predict(test_preds) / (n_splits * len(random_state_list))

    gc.collect()


# In[31]:


mean_score = np.mean(ensemble_score)
std_score = np.std(ensemble_score)
print(f'Ensemble MedAE score {mean_score:.5f} ± {std_score:.5f}')

# Print the mean and standard deviation of the ensemble weights for each model
print('--- Model Weights ---')
mean_weights = np.mean(weights, axis=0)
std_weights = np.std(weights, axis=0)
for name, mean_weight, std_weight in zip(models.keys(), mean_weights, std_weights):
    print(f'{name} {mean_weight:.5f} ± {std_weight:.5f}')


# ## 6.4 Feature Importance

# In[32]:


def visualize_importance(models, feature_cols, title, top=10):
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
    sns.set_palette("Set2")
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance, errorbar='sd')
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.title(f'{title} Feature Importance [Top {top}]', fontsize=15)
    plt.grid(True, axis='x')
    plt.show()
    
for name, models in trained_models.items():
    visualize_importance(models, list(X_train.columns), name)


# # 7. Results

# In[33]:


sub = pd.read_csv('/kaggle/input/playground-series-s3e25/sample_submission.csv')
sub['Hardness'] = test_predss
sub.to_csv('submission_pure.csv',index=False)
sub.head()


# # 8. Generalization Ensemble

# In[34]:


X = train_copy.drop(columns='Hardness')
y = train_copy.Hardness

model_pre = lgb.LGBMRegressor()
model_pre.fit(X, y)
X_new = X.copy()
X_new['Hardness_pred'] = model_pre.predict(X)

model = create_model(X_new.shape[1])
history = model.fit(X_new.astype('float32'), y.astype('float32'),
                    epochs=100,
                    class_weight=model_pre.class_weight,
                    callbacks=callbacks_list,
                    validation_split=0.1)

test_copy['Hardness_pred'] = model_pre.predict(test_copy.astype('float32'))
sub_external=sub.copy()
sub_external["Hardness"] = model.predict(test_copy.astype('float32'))


# <font size="3">Let us also use publicly available results which are derived from different feature engineering techniques. No matter how good the modeling is, poor feature engineering cannot give good results and hence combining results from different feature engineering generalizes results</font>

# In[35]:


sub_high=pd.read_csv("/kaggle/input/notebook5fc3ba6c81/submission.csv")
sub1=pd.read_csv("/kaggle/input/k/tonyyunyang99/regression-with-a-mohs-hardness-dataset/submission.csv")
# sub2=pd.read_csv("/kaggle/input/mohs-hardness-model/sub.csv")
# sub3=pd.read_csv("/kaggle/input/k/tonyyunyang99/regression-with-a-mohs-hardness-dataset/submission.csv")


# ### 8.1 Weights
# <font size="3">The weights can be assigned based on the ranking the scores, any Means(Arithmetic/Geometric/Harmonic) or using any series of numbers. For example, (4, 2, 1, 1/2, 1/4)</font>

# In[36]:


sub_comb=sub1.copy()

sub_comb['Hardness']=(100*sub_high["Hardness"]+0.1*sub1['Hardness']+0.1*sub['Hardness']+0.1*sub_external['Hardness'])/(100+0.1+0.1+0.1)

sub_comb.to_csv('submission.csv',index=False)
sub_comb.head()

