#!/usr/bin/env python
# coding: utf-8

# # 1. INTRODUCTION
# <center>
# <img src="https://hdwallpaperim.com/wp-content/uploads/2017/09/17/63175-WALL%C2%B7E.jpg" width=1200 height=1000 />
# </center>

# <font size="3">Hello Kagglers! We're back with the 17th Episode of the Playground Series. In this episode, we'll be participating and tackling the challenge of identifying whether a group of machines has experienced failures. Let's dive in and explore the data together!"</font>
# 
# **FEATURE DESCRIPTION**
# 1. **Product ID**:<font size="3"> Unique Id, combination of Type variable followed by a number identifier</font>
# 2. **Type**:<font size="3"> Type of product/device (L/M/H)</font>
# 3. **Air Temperature[K]**:<font size="3"> Air temperature in Kelvin</font>
# 4. **Process Temperature[K]**:<font size="3"> Production process temperature in Kelvin</font>
# 5. **Rotational Speed**:<font size="3"> Speed in RPM (Rotations Per Minute) calculated with the power of 2860W</font>
# 6. **Torque**:<font size="3"> Torque in Nm (Newton Meter)</font>
# 7. **Tool Wear**:<font size="3"> Time unit needed to wear down the product/tool</font>
# 8. **Machine Failure**:<font size="3"> Machine Failure binary feature</font>
# 9. **TWF**:<font size="3"> Tool Wear Failure binary feature, indicating industrial tool failure resulting in the need for equipment change and defective products.</font>
# 10. **HDF**:<font size="3"> Heat Dissipation Failure binary feature, indicating failure in heat dissipation during the production process.</font>
# 12. **PWF**:<font size="3"> Power Failure binary featurer, indicating that the power supplied was not fit to the production process need resulting in a failure.</font>
# 13. **OSF**:<font size="3"> Overstain Failure binary feature, indicating failure involves product overstains which may be the result of high load and tension during production.</font>
# 14. **RNF**:<font size="3"> Random Failure binary feature, indicating that a random error causes the failure.</font>
# 
# Thanks to [luisfrentzen](https://www.kaggle.com/competitions/playground-series-s3e17/discussion/416765) for sharing the above details.
# 
# **EVALUATION METRIC:** <font size="3"> Area under the ROC Curve</font>
# 
# NOTE: The modeling section is referenced and modified from [tetsutani](https://www.kaggle.com/code/tetsutani/ps3e12-eda-ensemble-baseline#Visualize-Feature-importance-(XGBoost,-LightGBM,-Catboost)) . Please support this page, if you like my work
# 

# # 2. IMPORTS

# In[1]:


import sklearn
import numpy as np
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
from prettytable import PrettyTable
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.4)
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
tqdm_notebook.get_lock().locks = []
# !pip install sweetviz
# import sweetviz as sv
import concurrent.futures
from copy import deepcopy       
from functools import partial
from itertools import combinations
import random
from random import randint, uniform
import gc
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler,PowerTransformer, FunctionTransformer
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from itertools import combinations
from sklearn.impute import SimpleImputer
import xgboost as xg
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error,mean_squared_log_error, roc_auc_score, accuracy_score, f1_score, precision_recall_curve, log_loss
from sklearn.cluster import KMeans
get_ipython().system('pip install yellowbrick')
from yellowbrick.cluster import KElbowVisualizer
get_ipython().system('pip install gap-stat')
from gap_statistic.optimalK import OptimalK
from scipy import stats
import statsmodels.api as sm
from scipy.stats import ttest_ind
from scipy.stats import boxcox
import math
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
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier,ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
# from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier
from sklearn.svm import NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from catboost import Pool
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
pd.pandas.set_option('display.max_columns',None)


# In[2]:


train=pd.read_csv('/kaggle/input/playground-series-s3e17/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s3e17/test.csv')

original=pd.read_csv("/kaggle/input/machine-failure-predictions/machine failure.csv")

original["original"]=1
train["original"]=0
test["original"]=0
train.drop(columns=["id"],inplace=True)
test.drop(columns=["id"],inplace=True)
original.drop(columns=["UDI"],inplace=True)

train=pd.concat([train,original],axis=0)
train.reset_index(inplace=True,drop=True)

train.head()


# ## 2.1 BASIC ANALYSIS

# In[3]:


table = PrettyTable()
table.field_names = ['Column Name', 'Data Type', 'Train Missing %', 'Test Missing %', 'Unique Values %']

for column in train.columns:
    data_type = str(train[column].dtype)
    non_null_count_train = 100 - train[column].count() / train.shape[0] * 100

    if column != 'Machine failure':
        non_null_count_test = 100 - test[column].count() / test.shape[0] * 100
    else:
        non_null_count_test = "NA"

    unique_values_percentage = f"{train[column].nunique() / train.shape[0] * 100:.2f}%"

    table.add_row([column, data_type, non_null_count_train, non_null_count_test, unique_values_percentage])

print(table)


# **INFERENCES:**
# 1. <font size="3">We don't have any missing values</font>
# 2. <font size="3">**Crucial Observation!!!**, All variables have less than 10 % of  unique values. Even the product ID. My first thought here is to convert them into categorical and test a bunch of encoding techniques</font>

# # 3. EDA

# ## 3.1 Target Analysis

# In[4]:


class_counts = train['Machine failure'].value_counts()
class_proportions = class_counts / train.shape[0]
class_proportions_str = [f'{prop:.2%}' for prop in class_proportions]

labels = ['Working Machines', 'Failed Machines']

colors = ['black', 'red']

plt.figure(figsize=(8, 6))
_, _, autotexts = plt.pie(class_counts, labels=[None, None], autopct='%1.2f%%', colors=colors, startangle=90)
plt.title('Distribution of Target Variable', fontsize=16)
plt.axis('equal')

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)

plt.legend(labels, title='Machine Failure', loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# **HIGH CLASS IMBALACE DETECTED**
# 
# <font size="3"> Here on, I will be using **F1 Score** to take any decisions especially in the feature engineering because I would like to give emphasis on both the classes. Of course, the final modeling would be based on **ROC-AUC** which is our Metric of evaluation</font>

# ## 3.2 Numerical Features Analysis

# ### 3.2.1 Train & Test Data Distributions

# In[5]:


cont_cols=[f for f in train.columns if train[f].dtype in [float,int] and train[f].nunique()>2 and f not in ['Machine failure']]

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
# 1. <font size="3">Torque & rotational Speed follow a normal distribution</font>
# 2. <font size="3">The original dataset doesn't have much abnormalities, looks like an amplification due to synthetic generation</font>

# ### 3.2.2 Train Data Distributions Across Classes

# In[6]:


fig, axs = plt.subplots(nrows=1, ncols=len(cont_cols), figsize=(4 * len(cont_cols), 6))

colors = ['black', 'red']

for i, col in enumerate(cont_cols):
    sns.boxplot(x='Machine failure', y=col, data=train, ax=axs[i], palette=colors)
    axs[i].set_title(f'{col.title()} vs Target', fontsize=16)
    axs[i].set_xlabel('Machine failure', fontsize=14)
    axs[i].set_ylabel(col.title(), fontsize=14)
    axs[i].tick_params(axis='both', labelsize=14)
    sns.despine()

fig.tight_layout()
plt.show()


# <font size="3">Torque & rotational Speed have significant outliers</font>

# ### 3.2.3 Bivariate Analysis

# #### 3.2.3.1 Pair Plots

# In[7]:


sns.set(font_scale=0.8)  
sns.pairplot(data=train, vars=cont_cols, hue='Machine failure', palette=['black', 'red'])
plt.show()


# **INFERENCES**
# 1. **TORQUE:**<font size="3"> Across all the features, high torque and low torque have high machine failures. So, if any feature is created using torque and some other feature, it requires some kind of non linear transformation that can seperate the classes. </font>
# 2. **TOOL WEAR:**<font size="3"> High Tool Waer coupled with high temperatures increases the chance of failure. We can try some product based features here</font>
# 3. **ROTATIONAL SPEED:**<font size="3"> Similar to torque, low speeds and high speeds are the problem. If you look at both Torque and Rotational Speed, this tells a low torque -high rotational speed and high torque-low rotational speed is a good differentiator. I would definitely multiply rotational speed with torque, that would give the lower ends to be probable with machine failuers</font>
# 
# <font size="3">Finally, I will not use any distance based algorithms at all, SVMs would be a good choice however we have a big dataset.</font>

# ## 3.3 Correlation Plot

# In[8]:


import matplotlib.colors as mcolors

features = [f for f in train.columns if train[f].dtype != 'O' and f not in ['Machine failure']]
corr = train[features].corr()

plt.figure(figsize=(10, 8), dpi=300)
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

cmap = mcolors.LinearSegmentedColormap.from_list('RedBlack', ['#000000', '#FF0000'])
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, annot_kws={'size': 7})

plt.title('Features Correlation Matrix\n', fontsize=15, weight='bold')
plt.show()


# <font size="3"> The correlations we see above are natural. High process temperatures in the plant can lead to high air temperatures</font>

# # 4. FEATURE ENGINEERING

# # 4.1 New Features

# 1. **TORQUE x ROTATIONAL SPEED** <font size="3"> Based on EDA, we have identified the importance of this</font>
# 2. **TOOL WEAR x TEMPERATURES**<font size="3"> High Tool Wear coupled with high temperarures increases the chances, so let's multiply them</font>
# 3. **Product Series**<font size="3"> Based on suspicion,I'm extracting the series from Product ID that maybe be able to tell the age of product</font>
# 4.  **Temperature Difference**<font size="3"> The difference in temperature between the process and air temperature. When the machine becomes old, the processs temperature can get higher</font>
# 5. **Categorical Conversion**<font size="3">  Then convert all the numerical into categorical and apply encoding technqies</font>

# In[9]:


def new_features(df):
    
    df['Torque_x_Rot_speed']=df["Torque [Nm]"]*df["Rotational speed [rpm]"]
    
    df["Tool_x_Air_T"]=df["Tool wear [min]"]*df["Air temperature [K]"]
    
    df["Tool_x_Process_T"]=df["Tool wear [min]"]*df["Process temperature [K]"]
    
    df['Torque_x_Tool'] = df['Torque [Nm]'] * df['Tool wear [min]']
    
    df['Series']=df["Product ID"].str[1:].astype(int)
    
    df['Temp diff']=df['Air temperature [K]']-df["Process temperature [K]"]
    
    cat_converters=['Air temperature [K]', 'Process temperature [K]',
       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]','Temp diff']
    for feature in cat_converters:
        if feature in ['Rotational speed [rpm]']:
            df[feature[:feature.index(' ')]+"_cat"]=feature[:feature.index(' ')]+" "+np.round(df[feature]/10).astype(int).astype(str)
        else:
            df[feature[:feature.index(' ')]+"_cat"]=feature[:feature.index(' ')]+" "+np.round(df[feature]).astype(int).astype(str)


    
    return df

train=new_features(train)
test=new_features(test)
original=new_features(original)


# # 4.2 Transformations

# <font size="3">We're going to see what transformation works better for each feature and select them, the idea is to compress the data. There could be situations where you will have to stretch the data. These are the methods applied:</font>
# <font size="3">
# 1. <font size="3"> **Log Transformation**</font>: <font size="3">This transformation involves taking the logarithm of each data point. It is useful when the data is highly skewed and the variance increases with the mean.</font>
#                             y = log(x)
# 
# 2. <font size="3">**Square Root Transformation**</font>: <font size="3">This transformation involves taking the square root of each data point. It is useful when the data is highly skewed and the variance increases with the mean.</font>
#                             y = sqrt(x)
# 
# 3. <font size="3">**Box-Cox Transformation**</font>: <font size="3">This transformation is a family of power transformations that includes the log and square root transformations as special cases. It is useful when the data is highly skewed and the variance increases with the mean.</font>
#                             y = [(x^lambda) - 1] / lambda if lambda != 0
#                             y = log(x) if lambda = 0
# 
# 4. <font size="3">**Yeo-Johnson Transformation**</font>: <font size="3">This transformation is similar to the Box-Cox transformation, but it can be applied to both positive and negative values. It is useful when the data is highly skewed and the variance increases with the mean.</font>
#                             y = [(|x|^lambda) - 1] / lambda if x >= 0, lambda != 0
#                             y = log(|x|) if x >= 0, lambda = 0
#                             y = -[(|x|^lambda) - 1] / lambda if x < 0, lambda != 2
#                             y = -log(|x|) if x < 0, lambda = 2
# 
# 5. <font size="3">**Power Transformation**</font>: <font size="3">This transformation involves raising each data point to a power. It is useful when the data is highly skewed and the variance increases with the mean. The power can be any value, and is often determined using statistical methods such as the Box-Cox or Yeo-Johnson transformations.</font>
#                             y = [(x^lambda) - 1] / lambda if method = "box-cox" and lambda != 0
#                             y = log(x) if method = "box-cox" and lambda = 0
#                             y = [(x + 1)^lambda - 1] / lambda if method = "yeo-johnson" and x >= 0, lambda != 0
#                             y = log(x + 1) if method = "yeo-johnson" and x >= 0, lambda = 0
#                             y = [-(|x| + 1)^lambda - 1] / lambda if method = "yeo-johnson" and x < 0, lambda != 2
#                             y = -log(|x| + 1) if method = "yeo-johnson" and x < 0, lambda = 2

# In[10]:


cont_cols=[f for f in train.columns if train[f].nunique()>80 and train[f].dtype!='O']

sc=MinMaxScaler()
unimportant_features=[]
table = PrettyTable()
overall_best_score=0
overall_best_col='none'

dt_params= {'min_samples_split': 80, 'min_samples_leaf': 30, 'max_depth': 8, 'criterion': 'absolute_error'}

table.field_names = ['Feature', 'Original AUC', 'Transformed Feature', 'Tranformed AUC']
for col in cont_cols:
    
    # Log Transformation after MinMax Scaling(keeps data between 0 and 1)
    train["log_"+col]=np.log1p(sc.fit_transform(train[[col]]))
    test["log_"+col]=np.log1p(sc.transform(test[[col]]))
    original["log_"+col]=np.log1p(sc.transform(original[[col]]))

    
    
    # Square Root Transformation
    train["sqrt_"+col]=np.sqrt(sc.fit_transform(train[[col]]))
    test["sqrt_"+col]=np.sqrt(sc.transform(test[[col]]))
    original["sqrt_"+col]=np.sqrt(sc.transform(original[[col]]))

    
    # Box-Cox transformation
    transformer = PowerTransformer(method='box-cox')
    train["bx_cx_"+col] = transformer.fit_transform(sc.fit_transform(train[[col]])+1) # adjusted to make it +ve
    test["bx_cx_"+col] = transformer.transform(sc.transform(test[[col]])+1)
    original["bx_cx_"+col] = transformer.transform(sc.transform(original[[col]])+1)

    
    # Yeo-Johnson transformation
    transformer = PowerTransformer(method='yeo-johnson')
    train["y_J_"+col] = transformer.fit_transform(train[[col]])
    test["y_J_"+col] = transformer.transform(test[[col]])
    original["y_J_"+col] = transformer.transform(original[[col]])

    
    # Power transformation, 0.25
    power_transform = lambda x: np.power(x, 0.25) 
    transformer = FunctionTransformer(power_transform)
    train["pow_"+col] = transformer.fit_transform(sc.fit_transform(train[[col]]))
    test["pow_"+col] = transformer.transform(sc.transform(test[[col]]))
    original["pow_"+col] = transformer.transform(sc.transform(original[[col]]))

    
    # Power transformation, 0.1
    power_transform = lambda x: np.power(x, 0.1) 
    transformer = FunctionTransformer(power_transform)
    train["pow2_"+col] = transformer.fit_transform(sc.fit_transform(train[[col]]))
    test["pow2_"+col] = transformer.transform(sc.transform(test[[col]]))
    original["pow2_"+col] = transformer.transform(sc.transform(original[[col]]))

    
    # log to power transformation
    train["log_pow2"+col]=np.log1p(train["pow2_"+col])
    test["log_pow2"+col]=np.log1p(test["pow2_"+col])
    original["log_pow2"+col]=np.log1p(original["pow2_"+col])

    
    temp_cols=[col,"log_"+col,"sqrt_"+col, "bx_cx_"+col,"y_J_"+col ,"pow_"+col,"pow2_"+col,"log_pow2"+col ]
    
    # Fill na becaue, it would be Nan if the vaues are negative and a transformation applied on it
    train[temp_cols]=train[temp_cols].fillna(0)
    test[temp_cols]=test[temp_cols].fillna(0)
    original[temp_cols]=original[temp_cols].fillna(0)

    
    #Apply PCA on  the features and compute an additional column
    pca=TruncatedSVD(n_components=1)
    x_pca_train=pca.fit_transform(train[temp_cols])
    x_pca_test=pca.transform(test[temp_cols])
    x_pca_orig=pca.transform(original[temp_cols])
    x_pca_train=pd.DataFrame(x_pca_train, columns=[col+"_pca_comb"])
    x_pca_test=pd.DataFrame(x_pca_test, columns=[col+"_pca_comb"])
    x_pca_orig=pd.DataFrame(x_pca_orig, columns=[col+"_pca_comb"])
    temp_cols.append(col+"_pca_comb")
    #print(temp_cols)
    
    train=pd.concat([train,x_pca_train],axis='columns')
    test=pd.concat([test,x_pca_test],axis='columns')
    original=pd.concat([original,x_pca_orig],axis='columns')

    
    # See which transformation along with the original is giving you the best univariate fit with target
    kf=KFold(n_splits=10, shuffle=True, random_state=42)
    
    ACC=[]
    
    for f in temp_cols:
        X=train[[f]].values
        y=train["Machine failure"].values
        
        acc=[]
        for train_idx, val_idx in kf.split(X,y):
            X_train,y_train=X[train_idx],y[train_idx]
            x_val,y_val=X[val_idx],y[val_idx]
            
            model=LogisticRegression()
#             model=DecisionTreeRegressor(**dt_params)
            model.fit(X_train,y_train)
            y_pred=model.predict_proba(x_val)[:,1]
            acc.append(roc_auc_score(y_val,y_pred))
        ACC.append((f,np.mean(acc)))
        if f==col:
            orig_acc=np.mean(acc)
        if overall_best_score<np.mean(acc):
            overall_best_score=np.mean(acc)
            overall_best_col=f
    best_col, best_acc=sorted(ACC, key=lambda x:x[1], reverse=True)[0]
    
    cols_to_drop = [f for f in temp_cols if  f!= best_col]
#     print(cols_to_drop)
    final_selection=[f for f in temp_cols if f not in cols_to_drop]
    if cols_to_drop:
        unimportant_features=unimportant_features+cols_to_drop
    table.add_row([col,orig_acc,best_col ,best_acc])
print(table)    


# # 4.3 Categorical Encoding

# In[11]:


cat_features=[f for f in train.columns if train[f].dtype=="O"]

for feature in cat_features:
    train_categories = set(train[feature].unique())
    test_categories = set(test[feature].unique())
    
    if train_categories == test_categories:
        print(f"All categories in '{feature}' are present in both train and test datasets.")
    else:
        missing_categories = test_categories-train_categories 
        print(f"The following categories in '{feature}' are new in the test dataset: {missing_categories}")

# Since some categories are missing, I'm replacing it with the closest ones
test['Rotational_cat']=test['Rotational_cat'].replace({'Rotational 249':'Rotational 248'})
test['Temp_cat']=test['Temp_cat'].replace({'Temp -5':'Temp -3','Temp -16':'Temp -15', 'Temp -4':'Temp -3'})


# In[12]:


table = PrettyTable()
table.field_names = ['Feature', 'Encoded Feature', "ROC-AUC (CV)- Logistic"]

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

for feature in cat_features:
    # Target Guided Mean --Data Leakage Possible
    
#     cat_labels=train.groupby([feature])['Machine failure'].mean().sort_values().index
#     cat_labels2={k:i for i,k in enumerate(cat_labels,0)}
#     train[feature+"_target"]=train[feature].map(cat_labels2)
#     test[feature+"_target"]=test[feature].map(cat_labels2)
#     original[feature+"_target"]=original[feature].map(cat_labels2)

    
    ## Count Encoding
    
    dic=train[feature].value_counts().to_dict()
    train[feature+"_count"]=train[feature].map(dic)
    test[feature+"_count"]=test[feature].map(dic)
    original[feature+"_count"]=original[feature].map(dic)

    
    ## Count Labeling
    
    dic2=train[feature].value_counts().to_dict()
#     list1=np.arange(len(dic2.values()),0,-1) # Higher rank for high count
    list1=np.arange(len(dic2.values())) # Higher rank for low count
    dic3=dict(zip(list(dic2.keys()),list1))
    train[feature+"_count_label"]=train[feature].replace(dic3)
    test[feature+"_count_label"]=test[feature].replace(dic3)
    original[feature+"_count_label"]=original[feature].replace(dic3)

    
# #     ## WOE Binning
#     din=train.groupby([feature])['Machine failure'].sum()
#     num=(train.groupby([feature])['Machine failure'].count()-train.groupby([feature])['Machine failure'].sum())
#     cat_labels=np.log1p(num/(din+1e-5))#.sort_values().index
#     cat_labels2=cat_labels.to_dict()
#     train[feature+"_WOE"]=train[feature].map(cat_labels2)
#     test[feature+"_WOE"]=test[feature].map(cat_labels2)
#     original[feature+"_WOE"]=original[feature].map(cat_labels2)
    
    
    temp_cols=[ feature+"_count", feature+"_count_label"]#feature+"_target",,feature+"_WOE"
    
    
    if train[feature].nunique()<100:
        train,test=OHE(train,test,[feature],'Machine failure')
    else:
        train=train.drop(columns=[feature])
        test=test.drop(columns=[feature])
   
    # Also, doing a group clustering on all encoding types and an additional one-hot on the clusters
    
    temp_train=train[temp_cols]
    temp_test=test[temp_cols]
    orig_test=original[temp_cols]
    
    sc=StandardScaler()
    temp_train=sc.fit_transform(temp_train)
    temp_test=sc.transform(temp_test)
    orig_test=sc.transform(orig_test)
    model = KMeans()


    kmeans = KMeans(n_clusters=10)
    kmeans.fit(np.array(temp_train))
    labels_train = kmeans.labels_

    train[feature+'_cat_cluster_WOE'] = labels_train
    test[feature+'_cat_cluster_WOE'] = kmeans.predict(np.array(temp_test))
    original[feature+'_cat_cluster_WOE'] = kmeans.predict(np.array(orig_test))

#     cat_labels=cat_labels=train.groupby([feature+'_cat_cluster_WOE'])['Machine failure'].mean()
#     cat_labels2=cat_labels.to_dict()
#     train[feature+'_cat_cluster_WOE']=train[feature+'_cat_cluster_WOE'].map(cat_labels2)
#     test[feature+'_cat_cluster_WOE']=test[feature+'_cat_cluster_WOE'].map(cat_labels2)
#     original[feature+'_cat_cluster_WOE']=original[feature+'_cat_cluster_WOE'].map(cat_labels2)


#     temp_cols=temp_cols+[feature+'_cat_cluster_WOE']
  
    
    
    # See which transformation along with the original is giving you the best univariate fit with target
    skf=StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    
    accuaries=[]
    
    for f in temp_cols:
        X=train[[f]].values
        y=train["Machine failure"].values
        
        acc=[]
        for train_idx, val_idx in skf.split(X,y):
            X_train,y_train=X[train_idx],y[train_idx]
            x_val,y_val=X[val_idx],y[val_idx]
            
            model=LogisticRegression()
            model.fit(X_train,y_train)
            y_pred=model.predict_proba(x_val)[:,1]
            acc.append(roc_auc_score(y_val,y_pred))
        accuaries.append((f,np.mean(acc)))
    best_col, best_acc=sorted(accuaries, key=lambda x:x[1], reverse=True)[0]
    
    # check correlation between best_col and other columns and drop if correlation >0.75
    corr = train[temp_cols].corr(method='pearson')
    corr_with_best_col = corr[best_col]
    cols_to_drop = [f for f in temp_cols if corr_with_best_col[f] > 0.8 and f != best_col]
    final_selection=[f for f in temp_cols if f not in cols_to_drop]
#     if cols_to_drop:
#         train = train.drop(columns=cols_to_drop)
#         test = test.drop(columns=cols_to_drop)
    table.add_row([feature,best_col ,best_acc])
print(table)


# # 4.4 Cluster Unimportant Features

# <font size="3">WE can take the less important transformed features and apply clustering technique followed by a target-mean encoding</font>

# In[13]:


table = PrettyTable()
table.field_names = ['Cluster WOE Feature', 'ROC-AUC(CV 5 fold)']
for col in cont_cols:
    sub_set=[f for f in unimportant_features if col in f]
#     print(sub_set)
    temp_train=train[sub_set]
    temp_test=test[sub_set]
    temp_orig=original[sub_set]
    sc=StandardScaler()
    temp_train=sc.fit_transform(temp_train)
    temp_test=sc.transform(temp_test)
    temp_orig=sc.transform(temp_orig)

    model = KMeans()

    # print(ideal_clusters)
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(np.array(temp_train))
    labels_train = kmeans.labels_

    train[col+"_unimp_cluster_WOE"] = labels_train
    test[col+"_unimp_cluster_WOE"] = kmeans.predict(np.array(temp_test))
    original[col+"_unimp_cluster_WOE"] = kmeans.predict(np.array(temp_orig))

    
    cat_labels=cat_labels=train.groupby([col+"_unimp_cluster_WOE"])['Machine failure'].mean()
    cat_labels2=cat_labels.to_dict()
    train[col+"_unimp_cluster_WOE"]=train[col+"_unimp_cluster_WOE"].map(cat_labels2)
    test[col+"_unimp_cluster_WOE"]=test[col+"_unimp_cluster_WOE"].map(cat_labels2)
    original[col+"_unimp_cluster_WOE"]=original[col+"_unimp_cluster_WOE"].map(cat_labels2)

    
    kf=KFold(n_splits=5, shuffle=True, random_state=42)
    
    X=train[[col+"_unimp_cluster_WOE"]].values
    y=train["Machine failure"].values

    acc=[]
    for train_idx, val_idx in kf.split(X,y):
        X_train,y_train=X[train_idx],y[train_idx]
        x_val,y_val=X[val_idx],y[val_idx]

        model=LogisticRegression()
        model.fit(X_train,y_train)
        y_pred=model.predict_proba(x_val)[:,1]
        acc.append(roc_auc_score(y_val,y_pred))
    table.add_row([col+"_unimp_cluster_WOE",np.mean(acc)])
    if overall_best_score<np.mean(acc):
            overall_best_score=np.mean(acc)
            overall_best_col=col+"_unimp_cluster_WOE"
    
print(table)


# # 4.5 Arithmetic Features

# <font size="3">There are many columns that are primarily existed in the dataset and many others that got created from the feature engineering technique. Among all of the columns, we will have one column that has the best CV Score with the target variable. The idea is to create a column that performs better than the existing best column</font>
# 
# <font size="3">Below are the steps followed in the function</font> **better_features:**
# 1. <font size="3">Note the best score and best column in the existing dataset</font>
# 2. <font size="3">Select a list of columns from the dataset and perform various arithmetic operations</font>
# 3. <font size="3">Add the new column to the dataset only if it's CV score is better that the existing best score</font>

# In[14]:


print("The best column and it's CV ROC AUC Score are ",overall_best_col, overall_best_score )


# In[15]:


def better_features(train, test,original, target, cols, best_score):
    new_cols = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Stratified k-fold object

    for i in tqdm(range(len(cols)), desc='Generating Columns'):
        col1 = cols[i]
        temp_df = pd.DataFrame()  # Temporary dataframe to store the generated columns
        temp_df_test = pd.DataFrame()  # Temporary dataframe for test data
        temp_df_orig  = pd.DataFrame()
        for j in range(i+1, len(cols)):
            col2 = cols[j]
            # Multiply
            temp_df[col1 + '*' + col2] = train[col1] * train[col2]
            temp_df_test[col1 + '*' + col2] = test[col1] * test[col2]
            temp_df_orig[col1 + '*' + col2] = original[col1] * original[col2]

            # Divide (col1 / col2)
            temp_df[col1 + '/' + col2] = train[col1] / (train[col2] + 1e-5)
            temp_df_test[col1 + '/' + col2] = test[col1] / (test[col2] + 1e-5)
            temp_df_orig[col1 + '/' + col2] = original[col1] / (original[col2] + 1e-5)

            # Divide (col2 / col1)
            temp_df[col2 + '/' + col1] = train[col2] / (train[col1] + 1e-5)
            temp_df_test[col2 + '/' + col1] = test[col2] / (test[col1] + 1e-5)
            temp_df_orig[col2 + '/' + col1] = original[col2] / (original[col1] + 1e-5)

            # Subtract
            temp_df[col1 + '-' + col2] = train[col1] - train[col2]
            temp_df_test[col1 + '-' + col2] = test[col1] - test[col2]
            temp_df_orig[col1 + '-' + col2] = original[col1] - original[col2]

            # Add
            temp_df[col1 + '+' + col2] = train[col1] + train[col2]
            temp_df_test[col1 + '+' + col2] = test[col1] + test[col2]
            temp_df_orig[col1 + '+' + col2] = original[col1] + original[col2]

        SCORES = []
        for column in temp_df.columns:
            scores = []
            for train_index, val_index in skf.split(original, original[target]):
                X_train, X_val = temp_df[column].iloc[train_index].values.reshape(-1, 1), temp_df[column].iloc[val_index].values.reshape(-1, 1)
                y_train, y_val = train[target].iloc[train_index], train[target].iloc[val_index]
                model = LogisticRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict_proba(X_val)[:,1]
                score = roc_auc_score(y_val, y_pred)
                scores.append(score)
            mean_score = np.mean(scores)
            SCORES.append((column, mean_score))

        if SCORES:
            best_col, best_acc = sorted(SCORES, key=lambda x: x[1],reverse=True)[0]
            # Check correlation with other columns
            corr_with_other_cols = train.drop([target] + new_cols, axis=1).corrwith(temp_df[best_col])
#             print(corr_with_other_cols.abs().max())
            if corr_with_other_cols.abs().max() < 0.95 or best_acc > best_score:
                train[best_col] = temp_df[best_col]
                test[best_col] = temp_df_test[best_col]
                new_cols.append(best_col)
                print(f"Added column '{best_col}' with mean ROC AUC: {best_acc:.4f} & Correlation {corr_with_other_cols.abs().max():.4f}")

    return train, test, new_cols


# In[16]:


selected_cols=[f for f in train.columns if f not in ['Machine failure']+unimportant_features and train[f].dtype!='O' and train[f].nunique()>2]
len(selected_cols)
train, test,new_cols=better_features(train, test,original, 'Machine failure', selected_cols, overall_best_score)


# # 4.6 Feature Selection

# **Steps to Eliminate Correlated Fruit Features**:
# 1. <font size="3">Group features based on their parent feature. For example, all features derived from TORQUE come under one set</font>
# 2. <font size="3">Apply PCA on the set, Cluster-Target Encoding on the set</font>
# 3. <font size="3">See the performance of each feature on a cross-validated single feature-target model</font>
# 4. <font size="3">Select the feature with highest ROC-AUC</font>

# In[17]:


final_drop_list=[]

table = PrettyTable()
table.field_names = ['Original', 'Final Transformation', "ROC-AUC(CV)"]
threshold=0.901
# It is possible that multiple parent features share same child features, so store selected features to avoid selecting the same feature again
best_cols=[]

for col in cont_cols:
    sub_set=[f for f in train.columns if col in f and train[f].nunique()>100]
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
            kmeans = KMeans(n_clusters=10)
            kmeans.fit(np.array(temp_train))
            labels_train = kmeans.labels_

            train[col+'_final_cluster'] = labels_train
            test[col+'_final_cluster'] = kmeans.predict(np.array(temp_test))

            cat_labels=cat_labels=train.groupby([col+"_final_cluster"])['Machine failure'].mean()
            cat_labels2=cat_labels.to_dict()
            train[col+"_final_cluster"]=train[col+"_final_cluster"].map(cat_labels2)
            test[col+"_final_cluster"]=test[col+"_final_cluster"].map(cat_labels2)

            correlated_features=correlated_features+[col+"_pca_comb_final",col+"_final_cluster"]

            # See which transformation along with the original is giving you the best univariate fit with target
            kf=KFold(n_splits=5, shuffle=True, random_state=42)

            MAE=[]

            for f in correlated_features:
                X=train[[f]].values
                y=train["Machine failure"].values

                mae=[]
                for train_idx, val_idx in kf.split(X,y):
                    X_train,y_train=X[train_idx],y[train_idx]
                    x_val,y_val=X[val_idx],y[val_idx]

                    model=LogisticRegression()
                    model.fit(X_train,y_train)
                    y_pred=model.predict_proba(x_val)[:,1]
                    mae.append(roc_auc_score(y_val,y_pred))
                if f not in best_cols:
                    MAE.append((f,np.mean(mae)))
            best_col, best_acc=sorted(MAE, key=lambda x:x[1], reverse=True)[0]
            best_cols.append(best_col)

            cols_to_drop = [f for f in correlated_features if  f not in best_cols]
            if cols_to_drop:
                final_drop_list=final_drop_list+cols_to_drop
            table.add_row([col,best_col ,best_acc])

print(table)      


# In[18]:


# final_drop_list=[f for f in final_drop_list if f]
train.drop(columns=[*set(final_drop_list)],inplace=True)
test.drop(columns=[*set(final_drop_list)],inplace=True)


# # 5. DATA SELECTION & SCALING

# In[19]:


# train=train.drop(columns=['Product ID'])
# test=test.drop(columns=['Product ID'])

feature_scale=[feature for feature in train.columns if train[feature].nunique()>2 and 'count' not in f]
scaler=StandardScaler()
# feature_scale
train[feature_scale]=scaler.fit_transform(train[feature_scale])
test[feature_scale]=scaler.transform(test[feature_scale])

# Clean column names
def clean_cols(df):
    clean_cols=[re.sub(r'[^a-z A-Z 0-9]','',f) for f in df.columns]
    df.columns=clean_cols
    return df
train=clean_cols(train)
test=clean_cols(test)


X_train=train.drop(['Machine failure'],axis=1)
y_train=train['Machine failure']

X_test=test.copy()
print(X_train.shape,X_test.shape)

cat_features=[f for f in X_train.columns if (X_train[f].nunique()<=10 or "count" in f) and (X_train[f].dtype==int) and (X_train[f].nunique()>2)]


# # 6. Modeling

# ## 6.1 Model Selection

# In[20]:


class Splitter:
    def __init__(self, kfold=True, n_splits=5):
        self.n_splits = n_splits
        self.kfold = kfold

    def split_data(self, X, y, random_state_list):
        if self.kfold:
            for random_state in random_state_list:
                kf = StratifiedKFold(n_splits=self.n_splits, random_state=random_state, shuffle=True)
                for train_index, val_index in kf.split(X, y):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                    yield X_train, X_val, y_train, y_val
        else:
            X_train, X_val = X.iloc[:int(X_train.shape[0]/10)], X.iloc[int(X_train.shape[0]/10):]
            y_train, y_val = y.iloc[:int(X_train.shape[0]/10)], y.iloc[int(X_train.shape[0]/10):]
            yield X_train, X_val, y_train, y_val

class Classifier:
    def __init__(self, n_estimators=100, device="cpu", random_state=0):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.models = self._define_model()
        self.len_models = len(self.models)
        
    def _define_model(self):
        xgb_params={
            'colsample_bytree': 0.8498791800104656, 
            'learning_rate': 0.020233442882782587, 
            'max_depth': 7, 
            'subsample': 0.746529796772373,
            'n_estimators': self.n_estimators,
            'objective': 'binary:logistic',
            'n_jobs': -1,
            'random_state': self.random_state,
            "scale_pos_weight":50,

        }
        if self.device == 'gpu':
            xgb_params.update({
            'tree_method' :'gpu_hist',
            'predictor': 'gpu_predictor',
          })
        lgb_params={
            'colsample_bytree': 0.7774799983649324, 
            'learning_rate': 0.007653648135411494, 
            'max_depth': 8, 
            'reg_alpha': 0.14326300616140863, 
            'reg_lambda': 0.9310129332502252, 
            'subsample': 0.6189257947519665,
            'n_estimators': self.n_estimators,
            'objective': 'binary',
            'random_state': self.random_state,
            "scale_pos_weight":50,
        }
        cat_params={
            'random_strength': 0.1, 
            'one_hot_max_size': 10, 
            'max_bin': 100, 
            'learning_rate': 0.002, 
            'l2_leaf_reg': 0.5, 
            'grow_policy': 'Lossguide', 
            'depth': 8, 
            'bootstrap_type': 'Bernoulli',
            'n_estimators': self.n_estimators,
            'task_type': self.device.upper(),
            'random_state': self.random_state,
            "class_weights":{0: 1, 1: 50}
        }
        cat_params2={
            'random_strength': 0.3, 
            'one_hot_max_size': 10, 
            'max_bin': 150, 
            'learning_rate': 0.008, 
            'l2_leaf_reg': 0.1, 
            'grow_policy': 'SymmetricTree', 
            'depth': 4, 
            'bootstrap_type': 'Bernoulli',
            'n_estimators': self.n_estimators,
            'task_type': self.device.upper(),
            'random_state': self.random_state,
            "class_weights":{0: 1, 1: 50}
        }
        
        cat_sym_params = cat_params.copy()
        cat_sym_params['grow_policy'] = 'SymmetricTree'
        cat_dep_params = cat_params.copy()
        cat_dep_params['grow_policy'] = 'Depthwise'
        
        dt_params= {'min_samples_split': 80, 'min_samples_leaf': 30, 'max_depth': 8, 'criterion': 'gini'}
        gbm_params={
            'min_samples_split': 2, 
            'min_samples_leaf': 4, 
            'max_features': 'log2', 
            'max_depth': 5, 
            'learning_rate': 0.0004430621457583882
        }
        models = {
            'xgb': xgb.XGBClassifier(**xgb_params),
            'lgb': lgb.LGBMClassifier(**lgb_params),
            'cat': CatBoostClassifier(**cat_params),
            'cat2': CatBoostClassifier(**cat_params2),
            "cat_sym": CatBoostClassifier(**cat_sym_params),
            "cat_dep": CatBoostClassifier(**cat_dep_params),
#             'lr': LogisticRegression(),
#             'rf': RandomForestClassifier(max_depth= 9,max_features= 'auto',min_samples_split= 10,
#                                                            min_samples_leaf= 4,  n_estimators=500,random_state=self.random_state),
            'hgb': HistGradientBoostingClassifier(max_iter=self.n_estimators,learning_rate=0.01, loss="binary_crossentropy", 
                                                  n_iter_no_change=300,random_state=self.random_state),
#             'gbdt': GradientBoostingClassifier(**gbm_params,random_state=self.random_state),
#             'svc': SVC(gamma="auto", probability=True),
#             'knn': KNeighborsClassifier(n_neighbors=5),
#             'mlp': MLPClassifier(random_state=self.random_state, max_iter=1000),
#             'etr':ExtraTreesClassifier(min_samples_split=55, min_samples_leaf= 15, max_depth=10,
#                                        n_estimators=200,random_state=self.random_state),
#             'dt' :DecisionTreeClassifier(**dt_params,random_state=self.random_state),
#             'ada': AdaBoostClassifier(random_state=self.random_state),
#             'GNB': GaussianNB(**nb_params),
#             'ann':ann,
        }
        
        return models


# ## 6.2 Weighted Ensembling

# In[21]:


def f1_cutoff(precisions, recalls, thresholds):
    a=precisions*recalls/(recalls+precisions)
    b=sorted(zip(a,thresholds))
    return b[-1][1]

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

        # Calculate the Recall score for the weighted prediction
#         precisions,recalls, thresholds=precision_recall_curve(y_true,weighted_pred)
#         cutoff=f1_cutoff(precisions,recalls, thresholds)
#         y_weight_pred=np.where(weighted_pred>float(cutoff),1,0)        
        score = metrics.roc_auc_score(y_true, weighted_pred)
        return score

    def fit(self, y_true, y_preds, n_trials=2000):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        self.study = optuna.create_study(sampler=sampler, study_name="OptunaWeights", direction='maximize')
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        self.study.optimize(objective_partial, n_trials=n_trials)
        self.weights = [self.study.best_params[f"weight{n}"] for n in range(len(y_preds))]

    def predict(self, y_preds):
        assert self.weights is not None, 'OptunaWeights error, must be fitted before predict'
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=self.weights)
        return weighted_pred

    def fit_predict(self, y_true, y_preds, n_trials=2000):
        self.fit(y_true, y_preds, n_trials=n_trials)
        return self.predict(y_preds)
    
    def weights(self):
        return self.weights
    
def acc_cutoff_class(y_valid, y_pred_valid):
    y_valid=np.array(y_valid)
    y_pred_valid=np.array(y_pred_valid)
    fpr, tpr, threshold = metrics.roc_curve(y_valid, y_pred_valid)
    pred_valid = pd.DataFrame({'label': y_pred_valid})
    thresholds = np.array(threshold)
    pred_labels = (pred_valid['label'].values > thresholds[:, None]).astype(int)
    acc_scores = (pred_labels == y_valid).mean(axis=1)
    acc_df = pd.DataFrame({'threshold': threshold, 'test_acc': acc_scores})
    acc_df.sort_values(by='test_acc', ascending=False, inplace=True)
    cutoff = acc_df.iloc[0, 0]
    y_pred_valid=np.where(y_pred_valid<float(cutoff),0,1)
    return y_pred_valid


# ## 6.3 Model Training

# In[22]:


kfold = True
n_splits = 1 if not kfold else 5
random_state = 42
random_state_list = [42] 
n_estimators = 9999
early_stopping_rounds = 300
verbose = False
device = 'cpu'

splitter = Splitter(kfold=kfold, n_splits=n_splits)

# Initialize an array for storing test predictions
test_predss = np.zeros(X_test.shape[0])
ensemble_score = []
weights = []
trained_models = {'xgb':[], 'lgb':[], 'cat':[]}

    
for i, (X_train_, X_val, y_train_, y_val) in enumerate(splitter.split_data(X_train, y_train, random_state_list=random_state_list)):
    n = i % n_splits
    m = i // n_splits
            
    # Get a set of Regressor models
    classifier = Classifier(n_estimators, device, random_state)
    models = classifier.models
    
    # Initialize lists to store oof and test predictions for each base model
    oof_preds = []
    test_preds = []
    
    # Loop over each base model and fit it to the training data, evaluate on validation data, and store predictions
    for name, model in models.items():
        if ('cat' in name) or ("lgb" in name) or ("xgb" in name):
            if 'xgb' in name:
                model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)],early_stopping_rounds=early_stopping_rounds, verbose=verbose)
            elif 'lgb' in name:
                model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], 
                          categorical_feature=cat_features, early_stopping_rounds=early_stopping_rounds, verbose=verbose)
            elif 'cat' in name: 
                model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], 
                          cat_features=cat_features, early_stopping_rounds=early_stopping_rounds, verbose=verbose)
        elif name in 'ann':
            model.fit(X_train_, y_train_, validation_data=(X_val, y_val),batch_size=5, epochs=50,verbose=verbose)
        else:
            model.fit(X_train_, y_train_)
        
        if name in 'ann':
            test_pred = np.array(model.predict(X_test))[:, 0]
            y_val_pred = np.array(model.predict(X_val))[:, 0]
        else:
            test_pred = model.predict_proba(X_test)[:, 1]
            y_val_pred = model.predict_proba(X_val)[:, 1]

#         score = roc_auc_score(y_val, y_val_pred)
        score = roc_auc_score(y_val, y_val_pred)

        print(f'{name} [FOLD-{n} SEED-{random_state_list[m]}] ROC AUC score: {score:.5f}')
        
        oof_preds.append(y_val_pred)
        test_preds.append(test_pred)
        
        if name in trained_models.keys():
            trained_models[f'{name}'].append(deepcopy(model))
    # Use Optuna to find the best ensemble weights
    optweights = OptunaWeights(random_state=random_state)
    y_val_pred = optweights.fit_predict(y_val.values, oof_preds)
    
    score = roc_auc_score(y_val, y_val_pred)
    print(f'Ensemble [FOLD-{n} SEED-{random_state_list[m]}] ROC AUC score {score:.5f}')
    ensemble_score.append(score)
    weights.append(optweights.weights)
    
    test_predss += optweights.predict(test_preds) / (n_splits * len(random_state_list))
    
    gc.collect()


# In[23]:


# Calculate the mean Accuracy score of the ensemble
mean_score = np.mean(ensemble_score)
std_score = np.std(ensemble_score)
print(f'Ensemble ROC AUC score {mean_score:.5f} ± {std_score:.5f}')

# Print the mean and standard deviation of the ensemble weights for each model
print('--- Model Weights ---')
mean_weights = np.mean(weights, axis=0)
std_weights = np.std(weights, axis=0)
for name, mean_weight, std_weight in zip(models.keys(), mean_weights, std_weights):
    print(f'{name}: {mean_weight:.5f} ± {std_weight:.5f}')


# ## 6.4 Feature Importances

# In[24]:


def visualize_importance(models, feature_cols, title, top=20):
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
    sns.barplot(x='importance', y='feature', data=feature_importance, color='red', errorbar='sd')
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.title(f'{title} Feature Importance [Top {top}]', fontsize=18)
    plt.grid(True, axis='x')
    plt.show()
    
for name, models in trained_models.items():
    visualize_importance(models, list(X_train.columns), name)


# ## 6.5 Results Submission

# In[25]:


sub = pd.read_csv('/kaggle/input/playground-series-s3e17/sample_submission.csv')
sub['Machine failure'] = test_predss
sub.to_csv('submission.csv',index=False)
sub.head()

