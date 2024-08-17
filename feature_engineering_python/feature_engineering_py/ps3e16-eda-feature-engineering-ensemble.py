#!/usr/bin/env python
# coding: utf-8

# # 1. INTRODUCTION
# <center>
# <img src="https://www.dga.org/-/media/Images/DGAQ-Article-Images/1602-Spring-2016/DGAQSpring2016ShotToRemember09PiratesCaribbean.ashx?la=en&hash=F5D0141156FDD3195A19E94E863F926D84F8CA46" width=1500 height=800 />
# </center>
# 
# 

# **PROJECT DESCRIPTION**
# 
# <font size="3">This is the simplest dataset to understand among all the Playground Series, all we need to do is to estimate the Age of a crab based on the below given physical attributes</font>
# 
# **PHYSICAL ATTRIBUTES**
# 1. **SEX:** <font size="3"> Male/Female/Immatured</font>
# 2. **LENGTH:** <font size="3"> Length of the crab</font>
# 3. **DIAMETER:** <font size="3"> Diameter of the crab</font>
# 4. **HEIGHT:** <font size="3"> Height of the crab</font>
# 5. **WEIGHT:** <font size="3"> Weight of the crab</font>
# 6. **SHUCKED WEIGHT:** <font size="3"> Weight of the crab without the shell</font>
# 7. **VISCERA WEIGHT:** <font size="3"> Interal Oragns</font>
# 8. **SHELL WEIGHT:** <font size="3"> Shell Weight of the Crab</font>
# 9. **AGE:** <font size="3"> Age of the crab</font>
# 
# **EVALUATION METRIC:** <font size="3"> Mean Absolute Error</font>
# 
# **ADDITIONAL DATA:** Extended synthetic data has been taken from [MYKHAILO](https://www.kaggle.com/datasets/shalfey/extended-crab-age-prediction?datasetId=3343435&sortBy=dateRun&tab=profile) . Thanks for adding this to the competition.
# 
# **NOTE:** The modeling section is taken and modified from [TETSUTANI](https://www.kaggle.com/code/tetsutani/ps3e11-eda-xgb-lgbm-cat-ensemble-lb-0-29267/notebook) . Please support this page, if you like my work

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


train=pd.read_csv('/kaggle/input/playground-series-s3e16/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s3e16/test.csv')
train_exteded=pd.read_csv('/kaggle/input/extended-crab-age-prediction/train_extended.csv')

original=pd.read_csv("/kaggle/input/crab-age-prediction/CrabAgePrediction.csv")
original["original"]=1
train["original"]=0
test["original"]=0
train_exteded['original']=0
train.drop(columns=["id"],inplace=True)
test.drop(columns=["id"],inplace=True)
train_exteded.drop(columns=["id"],inplace=True)
train_exteded=train_exteded[train_exteded['Sex']!="0.025"]
train=pd.concat([train,original],axis=0)
train.reset_index(inplace=True,drop=True)


# In[3]:


train.head()


# ## 2.1 Check Missing Values

# In[4]:


table = PrettyTable()

table.field_names = ['Column Name', 'Data Type', 'Train Missing %', 'Test Missing %']
for column in train.columns:
    data_type = str(train[column].dtype)
    non_null_count_train= 100-train[column].count()/train.shape[0]*100
    if column!='Age':
        non_null_count_test = 100-test[column].count()/test.shape[0]*100
    else:
        non_null_count_test="NA"
    table.add_row([column, data_type, non_null_count_train,non_null_count_test])
print(table)


# # 3. Exploratory Data Analysis

# ## 3.1 Target Analysis

# In[5]:


import numpy as np
import matplotlib.pyplot as plt

class_0 = train[train['original'] == 0]['Age']
class_1 = train[train['original'] == 1]['Age']

mean_0 = np.mean(class_0)
median_0 = np.median(class_0)
mean_1 = np.mean(class_1)
median_1 = np.median(class_1)

fig, ax = plt.subplots(figsize=(12, 6))

ax.hist(class_0, bins=20, density=True, alpha=0.5, label='Original=0 Histogram')
ax.hist(class_1, bins=20, density=True, alpha=0.5, label='Original=1 Histogram')

x_values_0 = np.linspace(class_0.min(), class_0.max(), len(class_0))
density_values_0 = (1 / (np.sqrt(2 * np.pi) * np.std(class_0))) * np.exp(-0.5 * ((x_values_0 - mean_0) / np.std(class_0))**2)
ax.plot(x_values_0, density_values_0, color='red', label='Original=0 Density')

x_values_1 = np.linspace(class_1.min(), class_1.max(), len(class_1))
density_values_1 = (1 / (np.sqrt(2 * np.pi) * np.std(class_1))) * np.exp(-0.5 * ((x_values_1 - mean_1) / np.std(class_1))**2)
ax.plot(x_values_1, density_values_1, color='green', label='Original=1 Density')

ax.axvline(mean_0, color='blue', linestyle='dashed', linewidth=2, label='Mean (Original=0)')
ax.axvline(median_0, color='green', linestyle='dashed', linewidth=2, label='Median (Original=0)')
ax.axvline(mean_1, color='blue', linestyle='dashed', linewidth=2, label='Mean (Original=1)')
ax.axvline(median_1, color='red', linestyle='dashed', linewidth=2, label='Median (Original=1)')

ax.set_xlabel('Age')
ax.set_ylabel('Frequency / Density')
ax.set_title('Histograms and Density Plots')

# Manually set x-axis limits to ensure full visibility
x_min = min(min(class_0), min(class_1))
x_max = max(max(class_0), max(class_1))
ax.set_xlim([x_min, x_max])

ax.legend(bbox_to_anchor=(1,1),fancybox=False,shadow=False, loc='upper left')

plt.tight_layout()
plt.show()


# **INFERENCES**
# 
# <font size="3">The distribution, the mean/median between original and train datasets are really close and I have never known that crabs have such a big lifetime :)</font>

# ## 3.2 Train & Test Data Distributions

# In[6]:


cont_cols=[f for f in train.columns if train[f].dtype in [float,int] and train[f].nunique()>2 and f not in ['Age']]

# Calculate the number of rows needed for the subplots
num_rows = (len(cont_cols) + 2) // 3

# Create subplots for each continuous column
fig, axs = plt.subplots(num_rows, 3, figsize=(15, num_rows*5))

# Loop through each continuous column and plot the histograms
for i, col in enumerate(cont_cols):
    # Determine the range of values to plot
    max_val = max(train[col].max(), test[col].max(), original[col].max())
    min_val = min(train[col].min(), test[col].min(), original[col].min())
    range_val = max_val - min_val
    
    # Determine the bin size and number of bins
    bin_size = range_val / 20
    num_bins_train = round(range_val / bin_size)
    num_bins_test = round(range_val / bin_size)
    num_bins_original = round(range_val / bin_size)
    
    # Calculate the subplot position
    row = i // 3
    col_pos = i % 3
    
    # Plot the histograms
    sns.histplot(train[col], ax=axs[row][col_pos], color='red', kde=True, label='Train', bins=num_bins_train)
    sns.histplot(test[col], ax=axs[row][col_pos], color='grey', kde=True, label='Test', bins=num_bins_test)
    sns.histplot(original[col], ax=axs[row][col_pos], color='black', kde=True, label='Original', bins=num_bins_original)
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
# 1. <font size="3">Only Height follows a Normal Distribution</font>
# 2. <font size="3">Length Features & Weight Features have similar distribution across them. I suspect strong correlations between these categories. It is natural to have high shell weight & high visceral weight</font>

# ## 3.3 Sex & Numerical features

# In[7]:


sns.pairplot(data=train, vars=cont_cols+['Age'], hue='Sex')
plt.show()


# **INFERENCES**
# 1. <font size="3">We can see the growth of a crab in physical attributes when they mature </font>
# 2. <font size="3">There are few datapoints which are outliers because naturally difficult to have low weights, normal length, & 3 times taller than the population(Possibility of experimental errors). </font> **That is not a crab!**
# 3. <font size="3">I think Sex is an important feature especially the immatured category</font>
# 4. <font size="3">Correlation across weights is observed and also between length-diameter. This is expected, crabs do look like a square from the top :)</font>

# ## 3.4 Age vs Sex

# In[8]:


plt.subplots(figsize=(16, 5))
sns.violinplot(x='Sex', y=col, data=train)
plt.title('Age Distribution by Sex', fontsize=14)
plt.xlabel('Sex', fontsize=12)
plt.ylabel('Age', fontsize=12)
sns.despine()
fig.tight_layout()
plt.show()


# <font size="3">There is a noticeable difference from Immatured and Male/Female</font>

# ## 3.5 Correlation Plot

# In[9]:


features=[f for f in train.columns if train[f].astype!='O']
corr = train[features].corr()
plt.figure(figsize = (10, 8), dpi = 300)
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, mask = mask, cmap = 'magma', annot = True, annot_kws = {'size' : 7})
plt.title('Features Correlation Matrix\n', fontsize = 15, weight = 'bold')
plt.show()


# <font size="3">We have high correlation but they are important features</font>

# # 4. Feature Engineering

# # 4.1 New Features

# 1. **Top Surface Area**:<font size="3"> Length X Diameter</font>
# 2. **Water Loss**: <font size="3">During the experiment of data collection, it is possible to have some water loss after dissecting the crab to measure different weights</font>
# 3. **Crab Density**:<font size="3">Measure of body density</font>
# 4. **BMI:**<font size="3">Body Mass index</font>
# 5. **Measurement ratios:**<font size="3">You can also calculate ratios like length/height, length/diameter. </font>
# 6. **Incorrect Weights:**<font size="3"> I have noticed that there are sub weight columns greater than the total weight of the Crab. Part of the body cannot have more weight that the whole body</font>
# 7. **Incorrect Measurements:**<font size="3"> There are datapoints with 0 height, 0 diaameter, 0 length. We can use original data to replace them</font>

# In[10]:


def feat(df,original):
    # Correct Measurements
    df['Length']=np.where(df['Length']==0,original['Length'].min(),df['Length'])
    df['Diameter']=np.where(df['Diameter']==0,original['Diameter'].min(),df['Diameter'])
    df['Height']=np.where(df['Height']==0,0.025,df['Height']) #0.025 was the nearest non zero value

    # Clean the weights by capping the over weights with total body weights
    df['Shell Weight']=np.where(df['Shell Weight']>df['Weight'],df['Weight'],df['Shell Weight'])
    df['Viscera Weight']=np.where(df['Viscera Weight']>df['Weight'],df['Weight'],df['Viscera Weight'])
    df['Shucked Weight']=np.where(df['Shucked Weight']>df['Weight'],df['Weight'],df['Shucked Weight'])
    
    # Crab Surface area
    df["crab_area"]=df["Length"]*df["Diameter"]
    
    # Crab density approx
    df['approx_density']=df['Weight']/(df['crab_area']*df['Height'])
    
    # Crab BMI
    df['bmi']=df['Weight']/(df['Height']**2)
    
    # Measurement ratios
#     df["length_dia_ratio"]=df['Length']/df['Diameter']
#     df["length_height_ratio"]=df['Length']/df['Height']
#     df['shell_shuck_ratio']=df["Shell Weight"]/df["Shucked Weight"]
#     df['shell_viscera_ratio']=df['Shell Weight']/df['Viscera Weight']
    
    
    # Water Loss during experiment
    df["water_loss"]=df["Weight"]-df["Shucked Weight"]-df['Viscera Weight']-df['Shell Weight']
    df["water_loss"]=np.where(df["water_loss"]<0,min(df["Shucked Weight"].min(),df["Viscera Weight"].min(),df["Shell Weight"].min()),df["water_loss"])
    return df

train=feat(train,original)
test=feat(test,original)
original=feat(original,original)


# # 4.2 Numerical Transformations

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

# In[11]:


cont_cols=[f for f in train.columns if train[f].dtype!="O" and f not in ['Age','Height','original'] and train[f].nunique()>2]


sc=MinMaxScaler()
dt_params={'criterion': 'absolute_error'}
table = PrettyTable()
unimportant_features=[]
overall_best_score=100
overall_best_col='none'
table.field_names = ['Feature', 'Original MAE', 'Transformation', 'Tranformed MAE']
for col in cont_cols:
    
    # Log Transformation after MinMax Scaling(keeps data between 0 and 1)
    train["log_"+col]=np.log1p(sc.fit_transform(train[[col]]))
    test["log_"+col]=np.log1p(sc.transform(test[[col]]))
    
    # Square Root Transformation
    train["sqrt_"+col]=np.sqrt(sc.fit_transform(train[[col]]))
    test["sqrt_"+col]=np.sqrt(sc.transform(test[[col]]))
    
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
    power_transform = lambda x: np.power(x + 1 - np.min(x), 0.25)
    transformer = FunctionTransformer(power_transform)
    train["pow_"+col] = transformer.fit_transform(sc.fit_transform(train[[col]]))
    test["pow_"+col] = transformer.transform(sc.transform(test[[col]]))
    
    # Power transformation, 0.1
    power_transform = lambda x: np.power(x + 1 - np.min(x), 0.1)
    transformer = FunctionTransformer(power_transform)
    train["pow2_"+col] = transformer.fit_transform(sc.fit_transform(train[[col]]))
    test["pow2_"+col] = transformer.transform(sc.transform(test[[col]]))
    
    # log to power transformation
    train["log_sqrt"+col]=np.log1p(train["sqrt_"+col])
    test["log_sqrt"+col]=np.log1p(test["sqrt_"+col])
    
    temp_cols=[col,"log_"+col,"sqrt_"+col, "bx_cx_"+col,"y_J_"+col ,"pow_"+col,"pow2_"+col,"log_sqrt"+col ]
    
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
        y=train["Age"].values
        
        mae=[]
        for train_idx, val_idx in kf.split(X,y):
            X_train,y_train=X[train_idx],y[train_idx]
            x_val,y_val=X[val_idx],y[val_idx]
            
            model=LinearRegression()
#             model=DecisionTreeRegressor(**dt_params)
            model.fit(X_train,y_train)
            y_pred=model.predict(x_val)
            mae.append(mean_absolute_error(y_val,y_pred))
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
print("overall best CV score: ",overall_best_score)


# # 4.3 Categorical Encoding

# <font size="3">For each categorical variable, perform the following encoding techniques:</font>
# 1. <font size="3">**Count/Frequency Encoding**</font>: <font size="3">Count the number of occurrences of each category and replace the category with its log count.</font>
# 2. <font size="3">**Count Labeling**</font>: <font size="3">Assign a label to each category based on its count, with higher counts receiving higher labels.</font>
# 3. <font size="3"> **Target-Guided Mean Encoding**</font>: <font size="3">Rank the categories based on the mean of target column across each category</font>
# 
# <font size="3"> Please note that the features a particular encoding technique is not selected only if it has superior technique and the correlation with that is high</font>

# In[12]:


def cat_feat(df):
    # Round float values to two decimal places
    df['Cat_Length'] = np.round(df['Length'], 2)
    df['Cat_Diameter'] = np.round(df['Diameter'], 2)
    df['Cat_Height'] = np.round(df['Height'], 2)
    
    # Handle instances with less frequency
    feat = ['Cat_Length', 'Cat_Diameter', 'Cat_Height']
    min_frequency=20
    for f in feat:
        value_counts = df[f].value_counts()
        # Filter unique values based on frequency
        filtered_values = value_counts[value_counts >= min_frequency].index
        # Iterate over the value counts
        for value, count in value_counts.items():
            # Check if the count is less than min_frequency
            if count < min_frequency:
                # Find the closest value to replace it with
                if len(filtered_values) > 0:
                    closest_value = filtered_values[np.argmin(np.abs(filtered_values - value))]
                    # Replace the value with the closest value
                    df[f] = df[f].replace(value, closest_value)

    return df

train = cat_feat(train)
test = cat_feat(test)
original = cat_feat(original)

cat_features = ['Sex', 'Cat_Diameter', 'Cat_Length', 'Cat_Height']


# In[13]:


def OHE(train,test,cols,target):
    combined = pd.concat([train, test], axis=0)
    for col in cols:
        one_hot = pd.get_dummies(combined[col])
        counts = combined[col].value_counts()
        min_count_category = counts.idxmin()
        one_hot = one_hot.drop(min_count_category, axis=1)
        combined = pd.concat([combined, one_hot], axis="columns")
        combined = combined.drop(col, axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
    
    # split back to train and test dataframes
    train_ohe = combined[:len(train)]
    test_ohe = combined[len(train):]
    test_ohe.reset_index(inplace=True,drop=True)
    test_ohe.drop(columns=[target],inplace=True)
    
    return train_ohe, test_ohe

table = PrettyTable()
table.field_names = ['Feature', 'Encoded Features', 'MAE']

for feature in cat_features:
    ## Target Guided Mean --Data Leakage Possible
    
    cat_labels=train.groupby([feature])['Age'].mean().sort_values().index
    cat_labels2={k:i for i,k in enumerate(cat_labels,0)}
    train[feature+"_target"]=train[feature].map(cat_labels2)
    test[feature+"_target"]=test[feature].map(cat_labels2)
    
    ## Count Encoding
    
    dic=train[feature].value_counts().to_dict()
    train[feature+"_count"]=np.log1p(train[feature].map(dic))
    test[feature+"_count"]=np.log1p(test[feature].map(dic))

    
    ## Count Labeling
    
    dic2=train[feature].value_counts().to_dict()
    list1=np.arange(len(dic2.values()),0,-1) # Higher rank for high count
    # list1=np.arange(len(dic2.values())) # Higher rank for low count
    dic3=dict(zip(list(dic2.keys()),list1))
    train[feature+"_count_label"]=train[feature].replace(dic3)
    test[feature+"_count_label"]=test[feature].replace(dic3)
    
    
    
    temp_cols=[feature+"_target", feature+"_count", feature+"_count_label"]
    if train[feature].dtype!="O":
        temp_cols.append(feature)
    else:
        if train[feature].nunique()<=10:
            train, test=OHE(train,test,[feature],"Age")
    
    # See which transformation along with the original is giving you the best univariate fit with target
    kf=KFold(n_splits=10, shuffle=True, random_state=42)
    
    MAE=[]
    
    for f in temp_cols:
        X=train[[f]].values
        y=train["Age"].values
        
        mae=[]
        for train_idx, val_idx in kf.split(X,y):
            X_train,y_train=X[train_idx],y[train_idx]
            x_val,y_val=X[val_idx],y[val_idx]
            
            model=LinearRegression()
            model.fit(X_train,y_train)
            y_pred=model.predict(x_val)
            mae.append(mean_absolute_error(y_val,y_pred))
        MAE.append((f,np.mean(mae)))
        if overall_best_score>np.mean(mae):
            overall_best_score=np.mean(mae)
            overall_best_col=f
    best_col, best_acc=sorted(MAE, key=lambda x:x[1], reverse=False)[0]
    
    # check correlation between best_col and other columns and drop if correlation >0.5
    corr = train[temp_cols].corr(method='pearson')
    corr_with_best_col = corr[best_col]
    cols_to_drop = [f for f in temp_cols if corr_with_best_col[f] > 0.5 and f != best_col]
    final_selection=[f for f in temp_cols if f not in cols_to_drop]
    if cols_to_drop:
        train = train.drop(columns=cols_to_drop)
        test = test.drop(columns=cols_to_drop)
    
    table.add_row([feature,final_selection,best_acc])
print(table)
print("overall best CV score: ",overall_best_score)


# # 4.4 Clustering-One Hot Transformation

# <font size="3"> WE can take the less important transformed features and apply clustering with 28 clusters since we have 28 unique age values</font>

# In[14]:


table = PrettyTable()
table.field_names = ['Cluster WOE Feature', 'MAE(CV-TRAIN)']
for col in cont_cols:
    sub_set=[f for f in unimportant_features if col in f]
    print(sub_set)
    temp_train=train[sub_set]
    temp_test=test[sub_set]
    sc=StandardScaler()
    temp_train=sc.fit_transform(temp_train)
    temp_test=sc.transform(temp_test)
    model = KMeans()

    # print(ideal_clusters)
    kmeans = KMeans(n_clusters=28)
    kmeans.fit(np.array(temp_train))
    labels_train = kmeans.labels_

    train[col+"_unimp_cluster_WOE"] = labels_train
    test[col+"_unimp_cluster_WOE"] = kmeans.predict(np.array(temp_test))
    
    cat_labels=cat_labels=train.groupby([col+"_unimp_cluster_WOE"])['Age'].mean()
    cat_labels2=cat_labels.to_dict()
    train[col+"_unimp_cluster_WOE"]=train[col+"_unimp_cluster_WOE"].map(cat_labels2)
    test[col+"_unimp_cluster_WOE"]=test[col+"_unimp_cluster_WOE"].map(cat_labels2)
    
    kf=KFold(n_splits=5, shuffle=True, random_state=42)
    
    X=train[[col+"_unimp_cluster_WOE"]].values
    y=train["Age"].values

    mae=[]
    for train_idx, val_idx in kf.split(X,y):
        X_train,y_train=X[train_idx],y[train_idx]
        x_val,y_val=X[val_idx],y[val_idx]

        model=LinearRegression()
        model.fit(X_train,y_train)
        y_pred=model.predict(x_val)
        mae.append(mean_absolute_error(y_val,y_pred))
    table.add_row([col+"_unimp_cluster_WOE",np.mean(mae)])
    if overall_best_score>np.mean(mae):
            overall_best_score=np.mean(mae)
            overall_best_col=col+"_unimp_cluster_WOE"
    
print(table)


# # 4.5 Arithmetic Better features

# <font size="3">There are many columns that are primarily existed in the dataset and many others that got created from the feature engineering technique. Among all of the columns, we will have one column that has the best CV Score with the target variable. The idea is to create a column that performs better than the existing best column</font>
# 
# <font size="3">Below are the steps followed in the function</font> **better_features:**
# 1. <font size="3">Note the best score and best column in the existing dataset</font>
# 2. <font size="3">Select a list of columns from the dataset and perform various arithmetic operations</font>
# 3. <font size="3">Add the new column to the dataset only if it's CV score is better that the existing best score</font>

# In[15]:


print("The best column and it's CV MAE Score are ",overall_best_col, overall_best_score )


# In[16]:


def better_features(train, test, target, cols, best_score):
    new_cols = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Stratified k-fold object

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
                score = mean_absolute_error(y_val, y_pred)
                scores.append(score)
            mean_score = np.mean(scores)
            SCORES.append((column, mean_score))

        if SCORES:
            best_col, best_acc = sorted(SCORES, key=lambda x: x[1])[0]
            if best_acc < best_score:
                train[best_col] = temp_df[best_col]
                test[best_col] = temp_df_test[best_col]
                new_cols.append(best_col)
                print(f"Added column '{best_col}' with mean MAE: {best_acc:.4f}")

    return train, test, new_cols


# In[17]:


weight_cols=[f for f in train.columns if f not in ['Age']+unimportant_features]
len(weight_cols)
train, test,new_cols=better_features(train, test, 'Age', weight_cols, overall_best_score)


# # 4.6 Feature Selection

# In[18]:


corr = train.corr()
plt.figure(figsize = (30, 30), dpi = 300)
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, mask = mask, cmap = "magma", annot = True, annot_kws = {'size' : 7})
plt.title('Correlation Matrix\n', fontsize = 25, weight = 'bold')
plt.show()


# **Steps to Eliminate Correlated Fruit Features**:
# 1. <font size="3">Group features based on their parent feature. For example, all features derived from weight come under one set</font>
# 2. <font size="3">Apply PCA on the set, Cluster-Target Encoding on the set</font>
# 3. <font size="3">See the performance of each feature on a cross-validated single feature-target model</font>
# 4. <font size="3">Select the feature with highest CV-MAE</font>

# In[19]:


final_drop_list=[]

table = PrettyTable()
table.field_names = ['Original', 'Final Transformation', "MAE(CV)- Regression"]
dt_params={'criterion': 'absolute_error'}
threshold=0.98
# It is possible that multiple parent features share same child features, so store selected features to avoid selecting the same feature again
best_cols=[]

for col in cont_cols:
    sub_set=[f for f in train.columns if col in f and train[f].nunique()>100]
    print(sub_set)
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
        print(correlated_features)
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
            kmeans = KMeans(n_clusters=28)
            kmeans.fit(np.array(temp_train))
            labels_train = kmeans.labels_

            train[col+'_final_cluster'] = labels_train
            test[col+'_final_cluster'] = kmeans.predict(np.array(temp_test))

            cat_labels=cat_labels=train.groupby([col+"_final_cluster"])['Age'].mean()
            cat_labels2=cat_labels.to_dict()
            train[col+"_final_cluster"]=train[col+"_final_cluster"].map(cat_labels2)
            test[col+"_final_cluster"]=test[col+"_final_cluster"].map(cat_labels2)

            correlated_features=correlated_features+[col+"_pca_comb_final",col+"_final_cluster"]

            # See which transformation along with the original is giving you the best univariate fit with target
            kf=KFold(n_splits=5, shuffle=True, random_state=42)

            MAE=[]

            for f in correlated_features:
                X=train[[f]].values
                y=train["Age"].values

                mae=[]
                for train_idx, val_idx in kf.split(X,y):
                    X_train,y_train=X[train_idx],y[train_idx]
                    x_val,y_val=X[val_idx],y[val_idx]

                    model=LinearRegression()
        #             model=DecisionTreeRegressor(**dt_params)
                    model.fit(X_train,y_train)
                    y_pred=model.predict(x_val)
                    mae.append(mean_absolute_error(y_val,y_pred))
                if f not in best_cols:
                    MAE.append((f,np.mean(mae)))
            best_col, best_acc=sorted(MAE, key=lambda x:x[1], reverse=False)[0]
            best_cols.append(best_col)

            cols_to_drop = [f for f in correlated_features if  f not in best_cols]
            if cols_to_drop:
                final_drop_list=final_drop_list+cols_to_drop
            table.add_row([col,best_col ,best_acc])

print(table)      


# In[20]:


final_drop_list=[f for f in final_drop_list if f]
train.drop(columns=[*set(final_drop_list)],inplace=True)
test.drop(columns=[*set(final_drop_list)],inplace=True)


# <font size="3"> There is scope for more feature elimination, you can try some methods to increase the CV</font>

# # 5. Scaling & Data Selection

# In[21]:


feature_scale=[feature for feature in train.columns if feature not in ['Age']]

scaler=StandardScaler()

train[feature_scale]=scaler.fit_transform(train[feature_scale])
test[feature_scale]=scaler.transform(test[feature_scale])


# In[22]:


X_train = train.drop(['Age'], axis=1)
y_train = train['Age']

X_test = test.copy()

print(X_train.shape, X_test.shape)


# # 6. Modeling

# ## 6.1 Model Selection

# In[23]:


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
            'max_depth': 8,
            'learning_rate': 0.005,
            'colsample_bytree': 0.9,
            'subsample': 0.5,
            'min_child_weight': 9,
            'reg_lambda': 4.879e-07,
            'max_bin': 249,
            'booster': 'gbtree',
            'n_jobs': -1,
            'eval_metric': 'mae',
            'objective': "reg:squarederror",
            'grow_policy':'depthwise',
            'max_leaves':'256',
            'tree_method': 'hist',
            'verbosity': 0,
            'random_state': self.random_state,
        }
        if self.device == 'gpu':
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['predictor'] = 'gpu_predictor'

        xgb_params1=xgb_params.copy()
        xgb_params1['subsample']=0.3
        xgb_params1['max_depth']=6
        xgb_params1['learning_rate']=0.002
        xgb_params1['colsample_bytree']=0.9

        xgb_params2=xgb_params.copy()
        xgb_params2['subsample']=0.5
        xgb_params2['max_depth']=8
        xgb_params2['learning_rate']=0.005
        xgb_params2['colsample_bytree']=0.9
        xgb_params2['tree_method']='approx'

        lgb_params = {
            'n_estimators': self.n_estimators,
            'max_depth': 10,
            "num_leaves": 16,
            'learning_rate': 0.019000000000000003,
            'subsample': 0.3,
            'colsample_bytree': 0.5,
            'reg_alpha': 0.203704984640389,
            'reg_lambda': 0.48776091610863503,
            'objective': 'regression_l1',
            'metric': 'mean_absolute_error',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state
        }
        
        lgb_params1=lgb_params.copy()
        lgb_params1['num_leaves']=90
        lgb_params1['reg_lambda']=0.015589408048174165
        lgb_params1['reg_alpha']=0.09765625
        lgb_params1['min_child_samples']=40
        lgb_params1['learning_rate']=0.01533790147941807
        lgb_params1['colsample_bytree']=0.8809128870084636

        lgb_params2=lgb_params.copy()
        lgb_params2['subsample']=0.3
        lgb_params2['reg_lambda']=5e-07
#         lgb_params2['reg_alpha']=0.4300477974434703
        lgb_params2['max_depth']=10
        lgb_params2['learning_rate']=0.005
        lgb_params2['colsample_bytree']=0.8
        
        lgb_params3 = {
            'n_estimators': self.n_estimators,
            'num_leaves': 45,
            'max_depth': 8,
            'learning_rate': 0.01,
            'subsample': 0.5758412171285148,
            'colsample_bytree': 0.8599714680300794,
            'reg_lambda': 1.597717830931487e-08,
            'objective': 'regression_l1',
            'metric': 'mean_absolute_error',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state,
            'verbosity': 0,
            'force_col_wise': True
        }
        lgb_params4=lgb_params.copy()
        lgb_params4['subsample']=0.3
        lgb_params4['reg_lambda']=0.1355199957045592
        lgb_params4['reg_alpha']=0.8414930901189801
        lgb_params4['max_depth']=10
        lgb_params4['learning_rate']=0.019000000000000003
        lgb_params4['colsample_bytree']=0.8

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
#         cb_loss_params = cb_params.copy()
#         cb_loss_params['grow_policy'] = 'Lossguide'
    
        cb_params1 = {
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
        cb_params2= {
            'n_estimators': self.n_estimators,
            'depth': 9,
            'learning_rate': 0.01,
            'l2_leaf_reg': 4.8351074756668864e-05,
            'random_strength': 0.21306687539993183,
            'max_bin': 225,
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
        cb_loss_params={
            'iterations': self.n_estimators,
            'depth': 8,
            'learning_rate': 0.01,
            'l2_leaf_reg': 0.7,
            'random_strength': 0.2,
            'max_bin': 200,
            'od_wait': 65,
            'one_hot_max_size': 70,
            'grow_policy': 'Lossguide',
            'bootstrap_type': 'Bayesian',
            'od_type': 'Iter',
            'eval_metric': 'MAE',
            'loss_function': 'MAE',
            'task_type': self.device.upper(),
            'random_state': self.random_state
        }
        dt_params= {'min_samples_split': 90, 'min_samples_leaf': 40, 'max_depth': 6, 'criterion': 'absolute_error'}
        knn_params= {'weights': 'uniform', 'p': 1, 'n_neighbors': 12, 'leaf_size': 20, 'algorithm': 'kd_tree'}

        reg_models = {
            'xgb_reg': xgb.XGBRegressor(**xgb_params),
#             'xgb_reg1': xgb.XGBRegressor(**xgb_params1),
            'xgb_reg2': xgb.XGBRegressor(**xgb_params2),
            'lgb_reg': lgb.LGBMRegressor(**lgb_params),
#             'lgb2_reg': lgb.LGBMRegressor(**lgb_params1),
            'lgb3_reg': lgb.LGBMRegressor(**lgb_params2),
#             'lgb4_reg': lgb.LGBMRegressor(**lgb_params3),
#             'lgb5_reg': lgb.LGBMRegressor(**lgb_params4),
            "hgbm": HistGradientBoostingRegressor(max_iter=self.n_estimators, learning_rate=0.005, loss="absolute_error", 
                                                  n_iter_no_change=300,random_state=self.random_state),
            'cat_reg': CatBoostRegressor(**cb_params),
#             'cat_reg2': CatBoostRegressor(**cb_params1),
#             'cat_reg3': CatBoostRegressor(**cb_params2),
            "cat_sym": CatBoostRegressor(**cb_sym_params),
#             "cat_loss": CatBoostRegressor(**cb_loss_params),
#             'etr': ExtraTreesRegressor(min_samples_split=40, min_samples_leaf= 10, max_depth=16,
#                                        n_estimators=500,random_state=self.random_state),
#             'ann':ann,
            "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=500, learning_rate=0.02, max_depth=6,loss="absolute_error", random_state=self.random_state),
#             "RandomForestRegressor": RandomForestRegressor(max_depth= 9,max_features= 'auto',min_samples_split= 30,
#                                                            min_samples_leaf= 10,  n_estimators=500, random_state=self.random_state, n_jobs=-1),
#             'dt': DecisionTreeRegressor(**dt_params),
            
            "lr":LinearRegression(),
#             "knn":KNeighborsRegressor(**knn_params),
#             "PassiveAggressiveRegressor": PassiveAggressiveRegressor(max_iter=3000, tol=1e-3, n_iter_no_change=30, random_state=self.random_state),
#             "HuberRegressor": HuberRegressor(max_iter=3000),

            
            
        }


        return reg_models


# ## 6.2 Optuna- Weighted Ensembling

# In[24]:


class OptunaWeights:
    def __init__(self, random_state):
        self.study = None
        self.weights = None
        self.random_state = random_state

    def _objective(self, trial, y_true, y_preds):
        # Define the weights for the predictions from each model
        weights = [trial.suggest_float(f"weight{n}", -1, 1) for n in range(len(y_preds))]

        # Calculate the weighted prediction
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=weights)

        # Calculate the MAE score for the weighted prediction
        score = mean_absolute_error(y_true, weighted_pred)
        return score

    def fit(self, y_true, y_preds, n_trials=3000):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        self.study = optuna.create_study(sampler=sampler, study_name="OptunaWeights", direction='minimize')
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        self.study.optimize(objective_partial, n_trials=n_trials)
        self.weights = [self.study.best_params[f"weight{n}"] for n in range(len(y_preds))]
        weights_sum = sum(self.weights)
        self.weights = [weight / weights_sum for weight in self.weights]

    def predict(self, y_preds):
        assert self.weights is not None, 'OptunaWeights error, must be fitted before predict'
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=self.weights)
        return weighted_pred

    def fit_predict(self, y_true, y_preds, n_trials=3000):
        self.fit(y_true, y_preds, n_trials=n_trials)
        return self.predict(y_preds)
    
    def weights(self):
        return self.weights


# ## 6.3 Model Fit

# In[25]:


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
        if ('cat' in name) or ("lgb" in name) or ("xgb" in name):
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
#         y_val= np.expm1(y_val)
#         y_val_pred = np.round(y_val_pred)

        score = mean_absolute_error(y_val, np.round(y_val_pred))
        print(f'{name} [FOLD-{n} SEED-{random_state_list[m]}] MAE score: {score:.5f}')

        oof_preds.append(y_val_pred)
        test_preds.append(test_pred)
        if name in trained_models.keys():
            trained_models[f'{name}'].append(deepcopy(model))

    # Use Optuna to find the best ensemble weights
    optweights = OptunaWeights(random_state=random_state)
    y_val_pred = np.round(optweights.fit_predict(y_val.values, oof_preds))
    score = mean_absolute_error(y_val, y_val_pred)
    print(f'Ensemble [FOLD-{n} SEED-{random_state_list[m]}] MAE score {score:.5f}')
    ensemble_score.append(score)
    weights.append(optweights.weights)
    test_predss += optweights.predict(test_preds) / (n_splits * len(random_state_list))

    gc.collect()


# In[26]:


mean_score = np.mean(ensemble_score)
std_score = np.std(ensemble_score)
print(f'Ensemble MAE score {mean_score:.5f} ± {std_score:.5f}')

# Print the mean and standard deviation of the ensemble weights for each model
print('--- Model Weights ---')
mean_weights = np.mean(weights, axis=0)
std_weights = np.std(weights, axis=0)
for name, mean_weight, std_weight in zip(models.keys(), mean_weights, std_weights):
    print(f'{name} {mean_weight:.5f} ± {std_weight:.5f}')


# ## 6.4 Feature importance Visualization

# In[27]:


def visualize_importance(models, feature_cols, title, top=45):
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
    plt.figure(figsize=(12, 15))
    sns.barplot(x='importance', y='feature', data=feature_importance, color='crimson', errorbar='sd')
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.title(f'{title} Feature Importance [Top {top}]', fontsize=18)
    plt.grid(True, axis='x')
    plt.show()
    return feature_importance
feat_imp={}    
for name, models in trained_models.items():
    feat_imp[name]=visualize_importance(models, list(X_train.columns), name)


# ## 6.5 Final Submission

# <font size="3">All the float values are converted to the nearest integer</font>

# In[28]:


sub = pd.read_csv('/kaggle/input/playground-series-s3e16/sample_submission.csv')
sub['Age'] = np.round(test_predss)
sub.to_csv('submission.csv',index=False)
sub

