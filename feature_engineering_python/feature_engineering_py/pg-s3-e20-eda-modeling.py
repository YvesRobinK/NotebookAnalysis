#!/usr/bin/env python
# coding: utf-8

# **Created by Yang Zhou**
# 
# **[PLAYGROUND S-3,E-20] 📊EDA + MODELLING📈**
# 
# **1 Aug 2023**

# # <center style="font-family: consolas; font-size: 32px; font-weight: bold;">🐼Predict CO2 Emissions in Rwanda[EN/CN]</center>
# <p><center style="color:#949494; font-family: consolas; font-size: 20px;">Playground Series - Season 3, Episode 20</center></p>
# 
# ***

# # <center style="font-family: consolas; font-size: 32px; font-weight: bold;">Overview</center>
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ The objective of this challenge is to create machine learning models that use open-source emissions data (from Sentinel-5P satellite observations) to predict carbon emissions.</p>
# <p style="font-family: consolas; font-size: 12px;">🔴 本次任务是使用开源排放数据预测碳排放。</p>
# 
# <p style="font-family: consolas; font-size: 16px;"> The current training data includes data from 2019 to 2021 included in the training data and the task is to predict the CO2 emission data from 2022 to November.</p>
# <p style="font-family: consolas; font-size: 12px;">🔴 训练集中包含了2019-2021的排放数据，任务是预测2022年到11月的排放数据。</p>
# 
# <p style="font-family: consolas; font-size: 16px;"> This will be a time series forecasting task and the models available to us are traditional tree models, time series models such as Arima/Sarima, and neural network models such as LSTM/GRU.</p>
# <p style="font-family: consolas; font-size: 12px;">🔴 又是一个时间序列任务，与上次一样，同样可以选择时序模型，NN或树模型。</p>
# 

# # <center style="font-family: consolas; font-size: 32px; font-weight: bold;">Version Detail</center>
# 
# | Version | Description | LB Score |
# |---------|-------------|----------|
# | Version 7 | Add New NO ML Approach | 31.669 ==> 23.03
# | Version 6 | Add RF model | 31.669
# | Version 5 | Add tricks | 31.669 ==> 30.9
# | Version 4 | Create clusters feature | 33.35 ==> 31.669
# | Version 3 | FS with 10 features | 48.78 ==> 33.35
# | Version 2 | Drop features with more then 50% NaN values | 135.36 ==> 48.78
# | Version 1 | simple XGBoost baseline | 135.36 |

# #### <a id="top"></a>
# 
# <div style="background-color: rgba(60, 121, 245, 0.03); padding:30px; font-size:15px; font-family: consolas;">
# 
# * [0. Imports](#0)
# * [1. Load data](#1)
# * [2. EDA](#2)
# * [3. Data preprocecssing](#3)
# * [4. Feature selection](#4)
#     * [4.1 Pearson Correlation](#4.1)
#     * [4.2 Step Forward Selection](#4.2)
#     * [4.3 Step Forward Selection](#4.3)
#     * [4.4 Create Features](#4.4)
# * [5. Baseline Modeling](#5)
# * [6. Optuna](#6)
# * [7. No ML Approach](#7)
# * [8. Tricks and Submission](#)

# <a id="0"></a>
# # <b> 0. Imports </b>

# In[1]:


import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
import geopandas as gpd
from haversine import haversine
import datetime as dt

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Statistical Tests
from scipy.stats import f_oneway

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Model Selection
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, classification_report
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC

# Models
import optuna
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import Lasso, Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
import haversine as hs 

import warnings
warnings.filterwarnings('ignore')

import re

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, KFold, TimeSeriesSplit
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor, StackingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import HuberRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# In[2]:


# Adjusting plot style

rc = {
    "axes.facecolor": "#F8F8F8",
    "figure.facecolor": "#F8F8F8",
    "axes.edgecolor": "#000000",
    "grid.color": "#EBEBE7" + "30",
    "font.family": "serif",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "grid.alpha": 0.4
}

sns.set(rc=rc)
palette = ['#302c36', '#037d97', '#E4591E', '#C09741',
           '#EC5B6D', '#90A6B1', '#6ca957', '#D8E3E2']

from colorama import Style, Fore
blk = Style.BRIGHT + Fore.BLACK
mgt = Style.BRIGHT + Fore.MAGENTA
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
res = Style.RESET_ALL


# <a id="1"></a>
# # <b> 1. Load data </b>

# In[3]:


submission = pd.read_csv('/kaggle/input/playground-series-s3e20/sample_submission.csv')
train = pd.read_csv('/kaggle/input/playground-series-s3e20/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s3e20/test.csv')

train.drop(['ID_LAT_LON_YEAR_WEEK'], axis=1, inplace=True)
test.drop(['ID_LAT_LON_YEAR_WEEK'], axis=1, inplace=True)


# In[4]:


train.head()


# <a id="2"></a>
# # <b> 2. EDA </b>

# In[5]:


train.describe().T\
    .style.bar(subset=['mean'], color=px.colors.qualitative.G10[2])\
    .background_gradient(subset=['std'], cmap='Blues')\
    .background_gradient(subset=['50%'], cmap='BuGn')


# In[6]:


def summary(df):
    sum = pd.DataFrame(df.dtypes, columns=['dtypes'])
    sum['missing#'] = df.isna().sum().values*100
    sum['missing%'] = (df.isna().sum().values*100)/len(df)
    sum['uniques'] = df.nunique().values
    sum['count'] = df.count().values
    #sum['skew'] = df.skew().values
    desc = pd.DataFrame(df.describe().T)
    sum['min'] = desc['min']
    sum['max'] = desc['max']
    sum['mean'] = desc['mean']
    return sum

summary(train).style.background_gradient(cmap='Blues')


# <p style="font-family: consolas; font-size: 16px;">⚪ There are some missing values in the training data, which we will address later. In addition, latitude and longitude can be used for visualization.</p>
# <p style="font-family: consolas; font-size: 12px;">🔴 训练数据中有一些缺失值，我们将在后面解决这个问题。此外，经纬度可以用来可视化。</p>

# In[7]:


# Latitude and longitude

fig = px.scatter_mapbox(train, lat="latitude", lon="longitude", 
                        zoom=7, height=800, width=1000,) 
fig.update_layout(mapbox_style="open-street-map", title='Map')
fig.show()


# In[8]:


# line plot for emision 
sns.lineplot(data = train, x = 'week_no', y = 'emission', errorbar = None)
    
plt.title('Emission Over Week', fontsize = 24, fontweight = 'bold')
plt.show()


# In[9]:


sns.lineplot(data = train, x = 'year', y = 'emission', errorbar = None)
    
plt.title('Emission Over Year', fontsize = 24, fontweight = 'bold')
plt.xticks(rotation = 25)
plt.show()


# In[10]:


num = train.columns.tolist()[1:-1]
df = pd.concat([train[num].assign(Source = 'Train'), 
                test[num].assign(Source = 'Test')], 
               axis=0, ignore_index = True);

fig, axes = plt.subplots(len(num), 3 ,figsize = (16, len(num) * 4.2), 
                         gridspec_kw = {'hspace': 0.35, 'wspace': 0.3, 'width_ratios': [0.80, 0.20, 0.20]});

for i,col in enumerate(num):
    ax = axes[i,0];
    sns.kdeplot(data = df[[col, 'Source']], x = col, hue = 'Source', ax = ax, linewidth = 2.1)
    ax.set_title(f"\n{col}",fontsize = 9, fontweight= 'bold');
    ax.grid(visible=True, which = 'both', linestyle = '--', color='lightgrey', linewidth = 0.75);
    ax.set(xlabel = '', ylabel = '');

    ax = axes[i,1];
    sns.boxplot(data = df.loc[df.Source == 'Train', [col]], y = col, width = 0.25,saturation = 0.90, linewidth = 0.90, fliersize= 2.25, color = '#037d97',
                ax = ax);
    ax.set(xlabel = '', ylabel = '');
    ax.set_title(f"Train",fontsize = 9, fontweight= 'bold');

    ax = axes[i,2];
    sns.boxplot(data = df.loc[df.Source == 'Test', [col]], y = col, width = 0.25, fliersize= 2.25,
                saturation = 0.6, linewidth = 0.90, color = '#E4591E',
                ax = ax); 
    ax.set(xlabel = '', ylabel = '');
    ax.set_title(f"Test",fontsize = 9, fontweight= 'bold');


plt.suptitle(f"\nDistribution analysis- continuous columns\n",fontsize = 12, fontweight= 'bold',
             y = 0.89, x = 0.57);
plt.tight_layout();
plt.show();


# <p style="font-family: consolas; font-size: 16px;">⚪ I noticed that the difference in distribution between the training dataset and the test dataset seems to be a bit large, which could lead to generalization problems.</p>
# <p style="font-family: consolas; font-size: 12px;">🔴 注意到，训练数据集和测试数据集的分布差别似乎有点大，这可能会导致泛化问题。</p>

# In[11]:


# Correlation

corr_matrix = train[num].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='Blues', fmt='.2f', linewidths=1, square=True, annot_kws={"size": 9} )
plt.title('Correlation Matrix', fontsize=15)
plt.show()


# <p style="font-family: consolas; font-size: 16px;">⚪ I have noticed a strong correlation between some of the features. I might consider removing them.</p>
# <p style="font-family: consolas; font-size: 12px;">🔴 有一些特征相关性较强。需要考虑共线性问题。</p>

# <a id="3"></a>
# # <b> 3. Data preprocecssing </b>

# In[12]:


# from @DMITRY UAROV: https://www.kaggle.com/code/dmitryuarov/ps3e20-rwanda-emission-advanced-fe-29-7

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
massif = (-3.42, 28.592)
lake = (-2.0073, 31.6269)

def cluster_features(df, cluster_centers):
    for i, cc in enumerate(cluster_centers.values()):
        df[f'cluster_{i}'] = df.apply(lambda x: haversine((x['latitude'], x['longitude']), cc, unit='ft'), axis=1)
    return df

Y = train['emission']

def preprocessing(df):  
    # drop features with more than 50% nan.
    missing_ratios = df.isnull().mean()
    columns_to_drop = missing_ratios[missing_ratios > 0.5].index
    df = df.drop(columns_to_drop, axis=1)
    df = df.fillna(df.mean())
    
    cols_save = ['id', 'latitude', 'longitude', 'year', 'week_no', 'Ozone_solar_azimuth_angle']
    df = df[cols_save]
    
    # add features
    good_col = 'Ozone_solar_azimuth_angle'
    df[good_col] = df.groupby(['id', 'year'])[good_col].ffill().bfill()
    df[f'{good_col}_lag_1'] = df.groupby(['id', 'year'])[good_col].shift(1).fillna(0)
            
    df['rot_15_x'] = (np.cos(np.radians(15)) * df['longitude']) + \
                     (np.sin(np.radians(15)) * df['latitude'])
    
    df['rot_15_y'] = (np.cos(np.radians(15)) * df['latitude']) + \
                     (np.sin(np.radians(15)) * df['longitude'])

    df['rot_30_x'] = (np.cos(np.radians(30)) * df['longitude']) + \
                     (np.sin(np.radians(30)) * df['latitude'])

    df['rot_30_y'] = (np.cos(np.radians(30)) * df['latitude']) + \
                     (np.sin(np.radians(30)) * df['longitude'])
    
    for col, coors in zip(
        ['dist_rwanda', 'dist_park', 'dist_kirumba', 'dist_massif', 'dist_lake'], 
        [rwanda_center, park_biega, kirumba, massif, lake]
    ):
        df[col] = df.apply(lambda x: haversine((x['latitude'], x['longitude']), coors, unit='ft'), axis=1)
    
    df['month'] = df[['year', 'week_no']].apply(lambda row: get_month(row), axis=1)
    df['is_covid'] = (df['year'] == 2020) & (df['month'] > 2) | (df['year'] == 2021) & (df['month'] == 1)
    df['is_lockdown'] = (df['year'] == 2020) & ((df['month'].isin([3,4])))
    df['week_no_sin'] = np.sin(df['week_no']*(2*np.pi/52))
    df['week_no_sin'] = np.sin(df['week_no']*(2*np.pi/52))
    
    df.fillna(0, inplace=True)

    return df
    
train = preprocessing(train)
test = preprocessing(test)

df = pd.concat([train, test], axis=0, ignore_index=True)
coordinates = df[['latitude', 'longitude']].values
clustering = KMeans(n_clusters=12, max_iter=1000, random_state=42).fit(coordinates)
cluster_centers = {i: tuple(centroid) for i, centroid in enumerate(clustering.cluster_centers_)}
df = cluster_features(df, cluster_centers)

train = df.iloc[:-len(test),:]
test = df.iloc[-len(test):,:]
del df


# In[13]:


def feature_scalar(df):
    # scalar
    sc = StandardScaler()
    for i in df.columns:
        if i not in ['week_no', 'covid_flag','latitude','longitude','emission']:
            df[i] = sc.fit_transform(df[i].values.reshape(-1,1))
    return df

train = feature_scalar(train)
test = feature_scalar(test)


# <a id="4"></a>
# # <b> 4. Feature Engineering </b>
# 
# From @DR.ALVINLEENH and @LUCAS BOESEN
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ I chose 5,10,15 features respectively and the RMSE could only reach a maximum of 44, so I considered using the method in the high score notebook: constructing features using kmeans clustering.</p>
# <p style="font-family: consolas; font-size: 12px;">🔴 我分别选择了5,10,15个特征，RMSE最高只能达到44，因此我考虑使用高分notebook中的方法：使用kmeans聚类构建特征。</p>

# In[14]:


def calculateRMSE(x,y,features,model):
    for i,val in enumerate(x.columns):
        if val not in features:
            x = x.drop(columns=val)
    X_train, X_val, y_train, y_val = train_test_split(x,y,test_size=0.2, random_state=42)
    clf = LGBMRegressor().fit(X_train,y_train)
    prediction = clf.predict(X_val)
    loss = np.sqrt(mean_squared_error(y_val,prediction))
    
    print(f"{model} has RMSE = {loss}")


# <a id="4.1"></a>
# ## <b> 4.1 Pearson Correlation </b>

# In[15]:


X = train.copy()

train_fs = pd.concat([X,Y],axis = 1)


# In[16]:


trainCorr = train_fs.corr()
threshold = 0.05
corr=abs(trainCorr['emission'])
result = corr[corr>threshold]
result.sort_values(ascending=False)


# <p style="font-family: consolas; font-size: 16px;">⚪ I'm trying to find features that correlate with the target variable, but due to some preprocessing and normalization I've done, the current features don't correlate very well with the target variable.</p>
# <p style="font-family: consolas; font-size: 12px;">🔴 我们想找到与目标变量相关的特征，但是由于我们进行了一些预处理和标准化，目前的特征与目标变量相关性并不高。</p>

# In[17]:


# baseline
featurePC = X.columns
calculateRMSE(X,Y,featurePC,'Pearson Correlation')


# In[18]:


featurePC = result.index.drop('emission')
calculateRMSE(X,Y,featurePC,'Pearson Correlation')


# <p style="font-family: consolas; font-size: 16px;">⚪ Key point: The features we constructed do not have a strong linear correlation with emission, but that does not mean they are not useful.</p>
# <p style="font-family: consolas; font-size: 12px;">🔴 我们构建的特征与emission没有很强的线性相关性，但这不代表它们没有用。</p>

# <a id="4.2"></a>
# ## <b> 4.2 Step Forward Selection </b>

# In[19]:


# from mlxtend.feature_selection import SequentialFeatureSelector as sfs

# X_train, X_val, y_train, y_val = train_test_split(X,Y,test_size=0.2, random_state=42)
# clf = LGBMRegressor()
# sfs_1 = sfs(clf,k_features=10,forward=True,verbose=1, cv=5)
# sfs_1 = sfs_1.fit(X_train, y_train)


# In[20]:


# feat_cols = list(sfs_1.k_feature_idx_)
# featureSFS = X.columns[feat_cols].tolist()

featureSFS = ['id',
 'week_no',
 'Ozone_solar_azimuth_angle',
 'dist_rwanda',
 'dist_lake',
 'is_covid',
 'cluster_2',
 'cluster_3',
 'cluster_4',
 'cluster_8']

print("Selected features:", featureSFS)
calculateRMSE(X,Y,featureSFS,'Step Forward Selection')


# <p style="font-family: consolas; font-size: 16px;">⚪ The SFS feature achieves an RMSE of 20.98.</p>
# <p style="font-family: consolas; font-size: 12px;">🔴 SFS特征能达到20.98的RMSE。</p>

# <a id="4.3"></a>
# ## <b> 4.3 Recursive Feature elimination </b>

# In[21]:


# from sklearn.feature_selection import RFE

# clf = LGBMRegressor()
# rfe = RFE(clf)
# X_rfe = rfe.fit_transform(X,Y)
# clf.fit(X_rfe,Y)
# cols = list(X.columns)
# temp = pd.Series(rfe.support_,index = cols)


# In[22]:


# featureRFE = temp[temp==True].index.tolist()
# print("Selected features: ",featureRFE)

featureRFE = ['id', 'longitude', 'week_no', 'rot_15_y', 'dist_rwanda', 'dist_park', 'dist_kirumba', 'dist_lake', 'cluster_0', 'cluster_1', 'cluster_2', 'cluster_3', 'cluster_5', 'cluster_6', 'cluster_8', 'cluster_9']
calculateRMSE(X,Y,featureRFE,'Recursive Feature Elimination')


# <p style="font-family: consolas; font-size: 16px;">⚪ The RFE feature achieves an RMSE of 21.17.</p>
# <p style="font-family: consolas; font-size: 12px;">🔴 RFE特征能达到21.17的RMSE。</p>

# <a id="5"></a>
# # <b> 5. Baseline Modeling </b>

# <p style="font-family: consolas; font-size: 16px;">⚪ I selected some features to go into the model.</p>
# <p style="font-family: consolas; font-size: 12px;">🔴 我选择了一些特征进入模型。</p>

# In[23]:


# selected_feats = train.columns
selected_feats = featureSFS
# selected_feats = featureRFE


X = train[selected_feats]
Y = Y

test = test[selected_feats]


# In[24]:


xgb_cv_scores, xgb_preds = list(), list()
lgbm_cv_scores, lgbm_preds = list(), list()
rf_cv_scores, rf_preds = list(), list()
# ens_cv_scores, ens_preds = list(), list()

kf = KFold(n_splits=3, random_state=42, shuffle=True)



for i, (train_ix, test_ix) in enumerate(kf.split(X)):
    X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
    Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]
    
    print('---------------------------------------------------------------')
    
    ## XGBoost
    xgb_md = XGBRegressor().fit(X_train, Y_train)
    xgb_pred = xgb_md.predict(X_test)   
    xgb_score_fold = np.sqrt(mean_squared_error(Y_test, xgb_pred))
    print('Fold', i+1, '==> XGBoost oof RMSE score is ==>', xgb_score_fold)
    xgb_cv_scores.append(xgb_score_fold)
    
    xgb_pred_test = xgb_md.predict(test)
    xgb_preds.append(xgb_pred_test)
    
    ## LGBM
    lgbm_md = LGBMRegressor(n_estimators = 1000,
                           max_depth = 15,
                           learning_rate = 0.01,
                           num_leaves = 105,
                           reg_alpha = 0.1, #0.3
                           reg_lambda = 0.1,
                           subsample = 0.7,
                           colsample_bytree = 0.8).fit(X_train, Y_train)
    lgbm_pred = lgbm_md.predict(X_test) 
    lgbm_score_fold = np.sqrt(mean_squared_error(Y_test, lgbm_pred))
    print('Fold', i+1, '==> LGBM oof RMSE score is ==>', lgbm_score_fold)
    lgbm_cv_scores.append(lgbm_score_fold)
    
    lgbm_pred_test = lgbm_md.predict(test)
    lgbm_preds.append(lgbm_pred_test) 
    
    ## RF
    rf_md = RandomForestRegressor().fit(X_train, Y_train)
    rf_pred = rf_md.predict(X_test) 
    rf_score_fold = np.sqrt(mean_squared_error(Y_test, rf_pred))
    print('Fold', i+1, '==> RF oof RMSE score is ==>', rf_score_fold)
    rf_cv_scores.append(rf_score_fold)
    
    rf_pred_test = rf_md.predict(test)
    rf_preds.append(rf_pred_test) 
    
print('---------------------------------------------------------------')
print('Average RMSE of XGBoost model is:', np.mean(xgb_cv_scores))
print('Average RMSE of LGBM model is:', np.mean(lgbm_cv_scores))
print('Average RMSE of RF model is:', np.mean(rf_cv_scores))


# ## <center style="font-family: consolas; font-size: 32px; font-weight: bold;">CV Scores</center>

# | Feature | XGB CV Score | LGBM CV Score|
# |---------|--------------|--------------|
# | 6 Features | 19.92 | 19.66 |
# | Feature SFS | 14.62 | 15.72 |
# | Feature RFE | 16.90 | 16.97 |
# 

# <a id="6"></a>
# # <b> 6. Optuna </b>

# In[25]:


# def objective(trial):
    
#    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.15,random_state=42)
#    param = { 
#        'tree_method':'gpu_hist',
#        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
#        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
#        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
#        'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
#        'learning_rate': trial.suggest_categorical('learning_rate', [0.008,0.01,0.012,0.014,0.016,0.018, 0.02]),
#        'n_estimators': 10000,
#        'max_depth': trial.suggest_categorical('max_depth', [5,7,9,11,13,15,17]),
#        'random_state': trial.suggest_categorical('random_state', [2020]),
#        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
#    }
     
#    rmse_values = []
#    model = XGBRegressor(**param)      
#    model.fit(train_x,train_y,eval_set=[(test_x,test_y)],early_stopping_rounds=100,verbose=False)   
#    preds = model.predict(test_x)
    
#    rmse = mean_squared_error(test_y, preds,squared=False)
#    rmse_values.append(rmse)
#    return rmse


# In[26]:


# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=500)
# print('Number of finished trials:', len(study.trials))
# print('Best trial:', study.best_trial.params)

# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(rmse_values) + 1), rmse_values)
# plt.xlabel('Trials')
# plt.ylabel('RMSE')
# plt.title('Learning Curve')
# plt.show()


params = {
    'lambda': 0.09494389716848735,
    'alpha': 0.09394603884245925,
    'colsample_bytree': 0.9,
    'subsample': 0.4,
    'learning_rate': 0.02,
    'max_depth': 17,
    'random_state': 2020,
    'min_child_weight': 110
}


# <a id="7"></a>
# # <b> 7. No ML Approach </b>
# From @RANDOM DRAW
# 
# https://www.kaggle.com/code/danbraswell/no-ml-public-lb-23-02231/notebook

# <p style="font-family: consolas; font-size: 16px;">⚪ This approach looks better than the averages used in another previous notebook as emissions for 2022. I suspect this is because the emissions are higher in 2022. I'm going to try to add some tricks to this (e.g., multiply by 1.07) and let's see if the score goes up.</p>
# <p style="font-family: consolas; font-size: 12px;">🔴 这个方法比之前的NO ML方法效果更好，我猜测这是因为本来排列量就在逐年增加，如果我在这基础上再略微增加一些会怎样？。</p>

# In[27]:


train_noml = pd.read_csv('/kaggle/input/playground-series-s3e20/train.csv')
train_noml.insert(2,"lat_lon", list(zip(train["latitude"],train["longitude"])))

train_noml['id'] = train_noml[['latitude', 'longitude']].apply(lambda row: get_id(row), axis=1)
locations = train_noml["lat_lon"].unique()

def get_emissions_loc_year(loc, year):
    df = train_noml[(train_noml["lat_lon"]==loc) & (train_noml["year"]==year) & (train_noml["week_no"]<49)].copy()
    return df["emission"].values

# Function to get the max emission (over 3 years) at location.
def get_emissions_max(loc):
    emiss2019 = get_emissions_loc_year(loc,2019)
    emiss2020 = get_emissions_loc_year(loc,2020)
    emiss2021 = get_emissions_loc_year(loc,2021)
    return np.max([emiss2019,emiss2020,emiss2021],axis=0)


# In[28]:


train_noml.head(1)


# In[29]:


predictions_acc = []
for loc in locations:
    emission = get_emissions_max(loc)
    predictions_acc.append(emission)
    
# Create submission
submission_noml = submission.copy()
submission_noml["emission"] = np.hstack(predictions_acc)
submission_noml["emission"] = submission_noml["emission"]*0.995
submission_noml.loc[train_noml['id']==293, 'emission'] = np.array(train_noml[(train_noml['id']==293)&(train_noml['year']==2021)&(train_noml['week_no']<49)]['emission'])
    
submission_noml.head()


# <a id="8"></a>
# # <b> 8. Trick and Submission </b>

# In[30]:


def trick(submission,train):
    submission['emission'] = submission['emission']*1.07
    submission.loc[train['id']==293, 'emission'] = np.array(train[(train['id']==293)&(train['year']==2021)&(train['week_no']<49)]['emission'])
    return submission


# In[31]:


submission_xgb = submission.copy()
submission_lgb = submission.copy()
submission_rf = submission.copy()
submission_ens = submission.copy()

# Slightly improved forecasting results, from https://www.kaggle.com/code/johnsmith44/ps3e20-co2-emissions-in-rwanda-compact-trick
submission_xgb['emission'] = xgb_md.predict(test) 
submission_lgb['emission'] = lgbm_md.predict(test) 
submission_rf['emission'] = rf_md.predict(test) 

submission_xgb = trick(submission_xgb,train_noml)
submission_lgb = trick(submission_lgb,train_noml)
submission_rf = trick(submission_rf,train_noml)

submission_ens['emission'] = 0.2*submission_xgb['emission']+0.3*submission_lgb['emission']+0.5*submission_rf['emission']


submission_xgb.to_csv('/kaggle/working/xgb_submission.csv', index=False)
submission_lgb.to_csv('/kaggle/working/lgb_submission.csv', index=False)
submission_rf.to_csv('/kaggle/working/rf_submission.csv', index=False)
submission_ens.to_csv('/kaggle/working/ens_submission.csv', index=False)

submission_noml.to_csv("noml_submission.csv",index=False)


# In[32]:


submission_noml


# <p style="font-family: consolas; font-size: 16px;">⚪ If you like it, please upvote it!</p>
