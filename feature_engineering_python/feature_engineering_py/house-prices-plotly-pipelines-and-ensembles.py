#!/usr/bin/env python
# coding: utf-8

# ![](https://www.mipropertygroup.com.au/wp-content/uploads/2016/10/house-prices-double.jpeg)

# # **House Prices: Plotly, Pipelines and Ensembles**
# This is my second Kaggle kernel for the House Prices competition.  
# For a more basic, beginner level study of this data set check [my first House Prices kernel.](https://www.kaggle.com/dejavu23/house-prices-eda-to-ml-beginner)  
# In this kernel, I am going to explore the following a bit more advanced approaches and techniques:  
# * EDA with **Seaborn** and interactive charts with **Plotly**  
# * possible improvements by **Feature Engineering**  
# * Preprocessing using **sklearn Pipeline**    
# * use **GridSearchCV** with Pipelines     
# * apply linear models like **Ridge, Lasso, ElasticNet**   
# * and Ensemble models like **Boosting, Stacking, Voting**    
# * compare the performance of the Regression models for validation and test data

# ### **Outline of this kernel:**
# 
# [**Part 0: Imports, functions and info on data**](#Part 0: Imports, useful functions)  
# **0.1 data fields**  
# **0.2 data_description.txt**  
# [**Part 1: Exploratory Data Analysis**](#PART-1:-Exploratory-Data-Analysis)  
# **1.1 First look with Pandas**  
# shape, info, head   
# describe for [numerical](#describe-for-numerical-features) and [categorical](#describe-for-categorical-features) columns  
# **1.2 Handling missing values**  
# [Barchart: NaN in test and train](#Barchart:-NaN-in-test-and-train)  
# [Drop columns with lots of missing data](#Drop-columns-with-lots-of-missing-data)   
# **1.3 Visualizations for numerical features**    
# 1.3.0 Distribution of the target  
# [Distplot for SalePrice and SalePrice_Log](#Distplot-for-SalePrice-and-SalePrice_Log)  
# [Skewness and Kurtosis](#Skewness-and-Kurtosis)  
# 1.3.1 Correlation of numerical features to SalePrice  
# [Barchart: Correlation to SalePrice](#Barchart:-Correlation-to-SalePrice)  
# 1.3.2 Area features  
# [Scatterplot: SalePrice vs GrLivArea](#Scatterplot:-SalePrice-vs-GrLivArea)   
# [Scatterplots: SalePrice vs Area features](#Scatterplots:-SalePrice-vs-Area-features)  
# [New feature: all_SF = sum of many area features](#New-feature:-Sum-of-many-area-features)  
# [Scatterplot: SalePrice vs all_SF](#Scatterplot:-SalePrice-vs-all_SF)  
# [Boxplot: SalePrice vs. OverallQual](#Boxplot:-SalePrice-vs.-OverallQual)  
# [Scatterplot categorical colors: SalePrice vs. all_SF and OverallQual](http://)  
# **1.4 Visualizations for categorical features**    
# [Boxplot: SalePrice for Neighborhood](#Boxplot:-SalePrice-for-Neighborhood)    
# [Boxplot: SalePrice for MSZoning](#Boxplot:-SalePrice-for-MSZoning)  
# 
# 

# [**PART 2: Preprocessing and Pipelines**](#PART-2:-Preprocessing-and-Pipelines)  
# **2.0 Define data for regression models**  
# **2.1 Pipeline approach**  
# **2.2 Preproccessing Pipeline**  
# for [numerical](#for-numerical-features) and [categorical](#for-categorical-features) features   
# [ColumnTransformer](#ColumnTransformer)  
# **2.3 Append regressors to pipeline**  
# [2.3.1 Linear Models](#2.3.1-Linear-Models)  
# LinearRegression +++ Ridge +++ Lasso +++ ElaNet  
# [2.3.2 Ensemble Models](#2.3.2-Ensemble-Models)  
# GradientBoostingregressor +++ XGB +++ LGBM +++ ADABoost  
# 
# [**Part 3: Crossvalidation**](#Part-3:-Crossvalidation)  
# **3.1 Linear Models**  
# [Loop over Pipelines: Linear](#Loop-over-Pipelines:-Linear)  
# **3.2 Ensemble Models**  
# [Loop over Pipelines: Ensembles](#Loop-over-Pipelines:-Ensembles)
# 
# [**Part 4: GridSearchCV**](#Part-4:-GridSearchCV)  
# **4.1 Linear Models**  
# Loop over GridSearchCV Pipelines: Linear  
# **4.2 Ensemble Models**  
# Loop over GridSearchCV Pipelines: Ensembles
# 
# [**Part 5: Predictions for test data**](#Part-5:-Predictions-for-test-data)  
# 
# [Stacking](#5.3-Stacking)

# In[ ]:





# ## ToDo:
# 
# 
# 

# In[ ]:





# In[ ]:





# # PART 0: Imports, info

# ### Imports

# In[1]:


import numpy as np
import pandas as pd
pd.set_option('max_columns', 105)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

from scipy import stats
from scipy.stats import skew
from math import sqrt

# plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

# sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, Lasso, ElasticNet, BayesianRidge
from sklearn.kernel_ridge import KernelRidge

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor

from mlxtend.regressor import StackingRegressor

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.filterwarnings("ignore")

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# ### some useful functions

# In[2]:


def get_best_score(grid):
    
    best_score = np.sqrt(-grid.best_score_)
    print(best_score)    
    print(grid.best_params_)
    print(grid.best_estimator_)
    
    return best_score


# In[3]:


def plotly_scatter_x_y(df_plot, val_x, val_y):
    
    value_x = df_plot[val_x] 
    value_y = df_plot[val_y]
    
    trace_1 = go.Scatter( x = value_x, y = value_y, name = val_x, 
                         mode="markers", opacity=0.8 )

    data = [trace_1]
    
    plot_title = val_y + " vs. " + val_x
    
    layout = dict(title = plot_title, 
                  xaxis=dict(title = val_x, ticklen=5, zeroline= False),
                  yaxis=dict(title = val_y, side='left'),                                  
                  legend=dict(orientation="h", x=0.4, y=1.0),
                  autosize=False, width=750, height=500,
                 )

    fig = dict(data = data, layout = layout)
    iplot(fig)


# In[4]:


def plotly_scatter_x_y_color(df_plot, val_x, val_y, val_z):
    
    value_x = df_plot[val_x] 
    value_y = df_plot[val_y]
    value_z = df_plot[val_z]
    
    trace_1 = go.Scatter( 
                         x = value_x, y = value_y, name = val_x, 
                         mode="markers", opacity=0.8, text=value_z,
                         marker=dict(size=6, color = value_z, 
                                     colorscale='Jet', showscale=True),                        
                        )
                            
    data = [trace_1]
    
    plot_title = val_y + " vs. " + val_x
    
    layout = dict(title = plot_title, 
                  xaxis=dict(title = val_x, ticklen=5, zeroline= False),
                  yaxis=dict(title = val_y, side='left'),                                  
                  legend=dict(orientation="h", x=0.4, y=1.0),
                  autosize=False, width=750, height=500,
                 )

    fig = dict(data = data, layout = layout)
    iplot(fig)


# In[5]:


def plotly_scatter_x_y_catg_color(df, val_x, val_y, val_z):
    
    catg_for_colors = sorted(df[val_z].unique().tolist())

    fig = { 'data': [{ 'x': df[df[val_z]==catg][val_x],
                       'y': df[df[val_z]==catg][val_y],    
                       'name': catg, 
                       'text': df[val_z][df[val_z]==catg], 
                       'mode': 'markers',
                       'marker': {'size': 6},
                      
                     } for catg in catg_for_colors       ],
                       
            'layout': { 'xaxis': {'title': val_x},
                        'yaxis': {'title': val_y},                    
                        'colorway' : ['#a9a9a9', '#e6beff', '#911eb4', '#4363d8', '#42d4f4',
                                      '#3cb44b', '#bfef45', '#ffe119', '#f58231', '#e6194B'],
                        'autosize' : False, 
                        'width' : 750, 
                        'height' : 600,
                      }
           }
  
    iplot(fig)


# In[6]:


def plotly_scatter3d(data, feat1, feat2, target) :

    df = data
    x = df[feat1]
    y = df[feat2]
    z = df[target]

    trace1 = go.Scatter3d( x = x, y = y, z = z,
                           mode='markers',
                           marker=dict( size=5, color=y,               
                                        colorscale='Viridis',  
                                        opacity=0.8 )
                          )
    data = [trace1]
    camera = dict( up=dict(x=0, y=0, z=1),
                   center=dict(x=0, y=0, z=0.0),
                   eye=dict(x=2.5, y=0.1, z=0.8) )

    layout = go.Layout( title= target + " as function of " +  
                               feat1 + " and " + feat2 ,
                        autosize=False, width=700, height=600,               
                        margin=dict( l=15, r=25, b=15, t=30 ) ,
                        scene=dict(camera=camera,
                                   xaxis = dict(title=feat1),
                                   yaxis = dict(title=feat2),
                                   zaxis = dict(title=target),                                   
                                  ),
                       )

    fig = go.Figure(data=data, layout=layout)
    iplot(fig)


# ## 0.1 data fields

# from the Kaggle data overview,  
# grouped by House and Land features as explored in this kernel

# **SalePrice** - the property's sale price in dollars.  
# This is the target variable that you're trying to predict.  
# 
# 
# **Areas**  
# 1stFlrSF: First Floor square feet  
# 2ndFlrSF: Second floor square feet  
# GrLivArea: Above grade (ground) living area square feet  
# TotalBsmtSF: Total square feet of basement area  
# MasVnrArea: Masonry veneer area in square feet  
# GarageArea: Size of garage in square feet  
# 
# LowQualFinSF: Low quality finished square feet (all floors)  
# BsmtFinSF1: Type 1 finished square feet  
# BsmtFinSF2: Type 2 finished square feet  
# BsmtUnfSF: Unfinished square feet of basement area  
# 
# WoodDeckSF: Wood deck area in square feet  
# OpenPorchSF: Open porch area in square feet  
# EnclosedPorch: Enclosed porch area in square feet  
# 3SsnPorch: Three season porch area in square feet  
# ScreenPorch: Screen porch area in square feet  
# PoolArea: Pool area in square feet  
#   
#   
# **Class, Condition, Quality**  
# OverallQual: Overall material and finish quality  
# OverallCond: Overall condition rating  
# MSSubClass: The building class  
# MSZoning: The general zoning classification  
# Neighborhood: Physical locations within Ames city limits  
# BldgType: Type of dwelling  
# HouseStyle: Style of dwelling   
# Foundation: Type of foundation  
# Functional: Home functionality rating  
# 
# RoofStyle: Type of roof  
# RoofMatl: Roof material  
# Exterior1st: Exterior covering on house  
# Exterior2nd: Exterior covering on house (if more than one material)  
# MasVnrType: Masonry veneer type  
#   
# KitchenQual: Kitchen quality  
# ExterQual: Exterior material quality  
# ExterCond: Present condition of the material on the exterior  
# FireplaceQu: Fireplace quality  
#   
# PoolQC: Pool quality  
# Fence: Fence quality 
# 
# Utilities: Type of utilities available  
# Heating: Type of heating  
# HeatingQC: Heating quality and condition  
# CentralAir: Central air conditioning  
# Electrical: Electrical system  
# 
# 
# **Rooms, numbers**  
# FullBath: Full bathrooms above grade  
# HalfBath: Half baths above grade  
# Bedroom: Number of bedrooms above basement level  
# Kitchen: Number of kitchens  
# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)  
# 
# Fireplaces: Number of fireplaces  
# 
# 
# **Lot, Street, Alley**  
# LotFrontage: Linear feet of street connected to property  
# LotArea: Lot size in square feet  
# Street: Type of road access  
# Alley: Type of alley access  
# LotShape: General shape of property  
# LandContour: Flatness of the property  
# LotConfig: Lot configuration  
# LandSlope: Slope of property  
# Condition1: Proximity to main road or railroad  
# Condition2: Proximity to main road or railroad (if a second is present)  
# PavedDrive: Paved driveway 
# 
# 
# **BASEMENT**  
# BsmtQual: Height of the basement  
# BsmtCond: General condition of the basement  
# BsmtExposure: Walkout or garden level basement walls  
# BsmtFinType1: Quality of basement finished area  
# BsmtFullBath: Basement full bathrooms  
# BsmtHalfBath: Basement half bathrooms  
# 
# 
# **Garage**  
# GarageType: Garage location  
# GarageYrBlt: Year garage was built  
# GarageFinish: Interior finish of the garage  
# GarageCars: Size of garage in car capacity    
# GarageQual: Garage quality  
# GarageCond: Garage condition  
# 
# **Years**  
# YearBuilt: Original construction date  
# YearRemodAdd: Remodel date  
# MoSold: Month Sold  
# YrSold: Year Sold  
# 
#  
# MiscFeature: Miscellaneous feature not covered in other categories  
# MiscVal: $Value of miscellaneous feature  
# 
# SaleType: Type of sale  
# SaleCondition: Condition of sale 

# ### 0.2 data_description.txt

# **For a detailed description of the 79 features**  
# **including a list of all categorical entries,** 
# **see** [this file](https://www.kaggle.com/c/5407/download/data_description.txt)
# 

# In[ ]:





# # PART 1: Exploratory Data Analysis

# ### 1.1 First look with Pandas

# In[7]:


df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")


# In[8]:


print("df_train.shape : " , df_train.shape)
print("*"*50)
print("df_test.shape  : " , df_test.shape)


# In[9]:


df_train.info()


# In[10]:


df_train.head()


# In[11]:


# dropping the column "Id" since it is not useful for predicting SalePrice
df_train.drop('Id',axis=1,inplace=True )
id_test = df_test['Id']                      # for submissions
df_test.drop('Id',axis=1,inplace=True )


# #### describe for numerical features

# In[12]:


df_train.describe().transpose()


# For the numerical columns, only three have missing values: LotFrontage, MasVnrArea and GarageYrBlt.

# In[ ]:





# #### describe for categorical features

# In[13]:


df_train.describe(include = ['O']).transpose()


# In[ ]:





# ## 1.2 Handling missing values  

# In[14]:


df_train_null = pd.DataFrame()
df_train_null['missing'] = df_train.isnull().sum()[df_train.isnull().sum() > 0].sort_values(ascending=False)

df_test_null = pd.DataFrame(df_test.isnull().sum(), columns = ['missing'])
df_test_null = df_test_null.loc[df_test_null['missing'] > 0]


# ### Barchart: NaN in test and train

# In[15]:


trace1 = go.Bar(x = df_train_null.index, 
                y = df_train_null['missing'],
                name="df_train", 
                text = df_train_null.index)

trace2 = go.Bar(x = df_test_null.index, 
                y = df_test_null['missing'],
                name="df_test", 
                text = df_test_null.index)

data = [trace1, trace2]

layout = dict(title = "NaN in test and train", 
              xaxis=dict(ticklen=10, zeroline= False),
              yaxis=dict(title = "number of rows", side='left', ticklen=10,),                                  
              legend=dict(orientation="v", x=1.05, y=1.0),
              autosize=False, width=750, height=500,
              barmode='stack'
              )

fig = dict(data = data, layout = layout)
iplot(fig)


# In[ ]:





# ### Drop columns with lots of missing data

# In[16]:


df_train.drop(['PoolQC', 'FireplaceQu', 'Fence', 
               'Alley', 'MiscFeature'], axis=1, inplace=True)
df_test.drop(['PoolQC', 'FireplaceQu', 'Fence',
               'Alley', 'MiscFeature'], axis=1, inplace=True)


# **Note:**  
# **Dropping these features improves the performance of the linear regressors**

# **For the remaining missing values we use a preprocessing Pipeline**  
# **with Imputer from sklearn (see: 2.2 Preprocessing pipeline)**

# In[ ]:





# ## 1.3 Visualizations for numerical features

# In[17]:


numerical_columns = df_train.select_dtypes(exclude=['object']).columns.tolist()
print(numerical_columns)


# ## 1.3.0 Distribution of the target

# #### Distplot for SalePrice and SalePrice_Log

# In[18]:


df_train["SalePrice_Log"] = np.log1p(df_train["SalePrice"])


# In[19]:


fig = tools.make_subplots(rows=1, cols=2, print_grid=False, 
                          subplot_titles=["SalePrice", "SalePriceLog"])


trace_1 = go.Histogram(x=df_train["SalePrice"], name="SalePrice")
trace_2 = go.Histogram(x=df_train["SalePrice_Log"], name="SalePriceLog")

fig.append_trace(trace_1, 1, 1)
fig.append_trace(trace_2, 1, 2)

iplot(fig)


# #### Skewness and Kurtosis

# In[20]:


from scipy.stats import skew, kurtosis
print(df_train["SalePrice"].skew(),"   ", df_train["SalePrice"].kurtosis())
print(df_train["SalePrice_Log"].skew(),"  ", df_train["SalePrice_Log"].kurtosis())


# In[ ]:





# ### 1.3.1 Correlation of numerical features to SalePrice

# #### Barchart: Correlation to SalePrice

# In[21]:


df_corr = df_train.corrwith(df_train['SalePrice']).abs().sort_values(ascending=False)[2:]

data = go.Bar(x=df_corr.index, 
              y=df_corr.values )
       
layout = go.Layout(title = 'Correlation to Sale Price', 
                   xaxis = dict(title = ''), 
                   yaxis = dict(title = 'correlation'),
                   autosize=False, width=750, height=500,)

fig = dict(data = [data], layout = layout)
iplot(fig)


# **Note on correlation of numerical features to SalePrice**  
# Keeping other parameters constant, we expect the value of a House to increase with its size and area.    
# Also for this dataset, large correlations to SalePrice are observed for many of the Area features:  
# GrLivArea, GarageArea, TotalBsmtSF, 1stFlrSF, etc.   
# Lets explore these in more detail and see how the results can be used for  
# outlier detection and feature engineering

# ### 1.3.2 Area features

# #### Scatterplot: SalePrice vs GrLivArea

# Of the Area features, 'GrLivArea' has the largest correlation to SalePrice.

# In[22]:


plotly_scatter_x_y(df_train, 'GrLivArea', 'SalePrice')


# **Note on Outlier Detection**  
# We store the index of the two data points to the lower right,  
# with SalePrice < 200 k and GrLivArea > 4000

# In[23]:


# outliers GrLivArea
outliers_GrLivArea = df_train.loc[(df_train['GrLivArea']>4000.0) & (df_train['SalePrice']<300000.0)]
outliers_GrLivArea[['GrLivArea' , 'SalePrice']]


# **Note on Feature Engineering**  
# In the data fields description it says that  
# GrLivArea: Above grade (ground) living area square feet  
# We find that for all entries in train and test data,  
# GrLivArea is equal to the sum of the 1st and 2nd Floor square feet  
# together with the LowQualFinSF: 

# In[24]:


df_train['sum_1SF_2SF_LowQualSF'] =  df_train['1stFlrSF'] + df_train['2ndFlrSF'] + df_train['LowQualFinSF']  
df_test['sum_1SF_2SF_LowQualSF'] =  df_test['1stFlrSF'] + df_test['2ndFlrSF'] + df_test['LowQualFinSF'] 
print(sum(df_train['sum_1SF_2SF_LowQualSF'] != df_train['GrLivArea']))
print(sum(df_test['sum_1SF_2SF_LowQualSF'] != df_test['GrLivArea']))


# '1stFlrSF' has a correlation to SalePrice of 0.605  
# '2ndFlrSF' has a correlation to SalePrice of 0.32  
# 'LowQualFinSF' has a correlation to SalePrice of 0.02  
# By adding these three areas we get a feature that has a correlation to target of 0.709  
# In the following we check if we can derive further useful features by adding or  
# subtracting some of the area features.

# **Dropping the column "sum_1SF_2SF_LowQualSF" again since it already exists as GrLivArea**

# In[25]:


df_train.drop('sum_1SF_2SF_LowQualSF',axis=1,inplace=True )
df_test.drop('sum_1SF_2SF_LowQualSF',axis=1,inplace=True )


# In[26]:


df_train['GrLivArea'].corr(df_train['SalePrice'])


# In[27]:


(df_train['GrLivArea']-df_train['LowQualFinSF']).corr(df_train['SalePrice'])


# 'GrLivArea' minus 'LowQualFinSF' which is equal to the sum of  
# '1stFlrSF' + '2ndFlrSF' has a larger correlation to SalePrice than 'GrLivArea'

# Lets look at the other Area features and see if we can derive a feature  
# that has even larger correlation to SalePrice

# ### Scatterplots: SalePrice vs Area features

# In[28]:


y_col_vals = 'SalePrice'
area_features = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
                 'MasVnrArea', 'GarageArea', 'LotArea',
                 'WoodDeckSF', 'OpenPorchSF', 'BsmtFinSF1']
                # 'ScreenPorch'
x_col_vals = area_features


# In[ ]:





# In[ ]:





# In[29]:


nr_rows=3
nr_cols=3

fig = tools.make_subplots(rows=nr_rows, cols=nr_cols, print_grid=False,
                          subplot_titles=area_features )
                                                                
for row in range(1,nr_rows+1):
    for col in range(1,nr_cols+1): 
        
        i = (row-1) * nr_cols + col-1
                   
        trace = go.Scatter(x = df_train[x_col_vals[i]], 
                           y = df_train[y_col_vals], 
                           name=x_col_vals[i], 
                           mode="markers", 
                           opacity=0.8)

        fig.append_trace(trace, row, col,)
 
                                                                                                  
fig['layout'].update(height=700, width=900, showlegend=False,
                     title='SalePrice' + ' vs. Area features')
iplot(fig)                                                


# In[ ]:





# **new feature : sum of all Living SF areas**  
# **all_Liv_SF = 'TotalBsmtSF' + '1stFlrSF' + '2ndFlrSF'**

# In[30]:


df_train['all_Liv_SF'] = df_train['TotalBsmtSF'] + df_train['1stFlrSF'] + df_train['2ndFlrSF'] 
df_test['all_Liv_SF'] = df_test['TotalBsmtSF'] + df_test['1stFlrSF'] + df_test['2ndFlrSF'] 

print(df_train['all_Liv_SF'].corr(df_train['SalePrice']))
print(df_train['all_Liv_SF'].corr(df_train['SalePrice_Log']))


# By summming up square feet for Basement, 1st and 2nd floor. we derive  
# a feature 'all_Liv_SF' that has a correlation to SalePrice of 0.78

# ### New feature: Sum of many area features

#   
# **all_SF = 'all_Liv_SF' + 'GarageArea' + 'MasVnrArea' + 'WoodDeckSF' + 'OpenPorchSF' + 'ScreenPorch'**

# For 'all_SF' we further add some of the outside area values.    
# This results in a correlation to SalePrice and also SalePriceLog of around 0.82

# In[31]:


df_train['all_SF'] = ( df_train['all_Liv_SF'] + df_train['GarageArea'] + df_train['MasVnrArea'] 
                       + df_train['WoodDeckSF'] + df_train['OpenPorchSF'] + df_train['ScreenPorch'] )
df_test['all_SF'] = ( df_test['all_Liv_SF'] + df_test['GarageArea'] + df_test['MasVnrArea']
                      + df_test['WoodDeckSF'] + df_test['OpenPorchSF'] + df_train['ScreenPorch'] )

print(df_train['all_SF'].corr(df_train['SalePrice']))
print(df_train['all_SF'].corr(df_train['SalePrice_Log']))


# The two new features are highly correlated.  
# This multicorrelation may be a problem for some linear models.  

# In[32]:


df_train['all_SF'].corr(df_train['all_Liv_SF'])


# In[ ]:





# #### Scatterplot: SalePrice vs all_SF

# In[33]:


plotly_scatter_x_y(df_train, 'all_SF', 'SalePrice')


# **Like for GrlivArea, there are two outliers at the lower right also for all_SF**  
# **We are going to drop these now.**

# In[34]:


outliers_allSF = df_train.loc[(df_train['all_SF']>8000.0) & (df_train['SalePrice']<200000.0)]
outliers_allSF[['all_SF' , 'SalePrice']]


# **Indexes for the outliers are the same like for GrLivArea**

# In[35]:


df_train = df_train.drop(outliers_allSF.index)


# In[36]:


df_train.corr().abs()[['SalePrice','SalePrice_Log']].sort_values(by='SalePrice', ascending=False)[2:16]


# **After dropping these two outliers all_SF has a correlation**  
# **to SalePrice (and also SalePriceLog) of 0.86**

# In[ ]:





# In[ ]:





# ### Boxplot: SalePrice vs. OverallQual

# In[37]:


trace = []
for name, group in df_train[["SalePrice", "OverallQual"]].groupby("OverallQual"):
    trace.append( go.Box( y=group["SalePrice"].values, name=name ) )
    
layout = go.Layout(title="OverallQual", 
                   xaxis=dict(title='OverallQual',ticklen=5, zeroline= False),
                   yaxis=dict(title='SalePrice', side='left'),
                   autosize=False, width=750, height=500)

fig = go.Figure(data=trace, layout=layout)
iplot(fig)


# As can be expected from the large correlation coefficient of 0.796 ,  
# there is an almost perfect linear increase of SalePrice with the OverallQual.  
# We notice that this feature is in fact categorical (ordinal),  
# only the discrete values 1,2..10 occur.  
# Also there are a few outliers for some of the OverallQual values.  
# We are dropping those that are very far from the upper fences:

# In[38]:


outliers_OverallQual_4 = df_train.loc[(df_train['OverallQual']==4) & (df_train['SalePrice']>200000.0)]
outliers_OverallQual_8 = df_train.loc[(df_train['OverallQual']==8) & (df_train['SalePrice']>500000.0)]
outliers_OverallQual_9 = df_train.loc[(df_train['OverallQual']==9) & (df_train['SalePrice']>500000.0)]
outliers_OverallQual_10 = df_train.loc[(df_train['OverallQual']==10) & (df_train['SalePrice']>700000.0)]

outliers_OverallQual = pd.concat([outliers_OverallQual_4, outliers_OverallQual_8, 
                                  outliers_OverallQual_9, outliers_OverallQual_10])


# In[39]:


outliers_OverallQual[['OverallQual' , 'SalePrice']]


# In[40]:


df_train = df_train.drop(outliers_OverallQual.index)


# In[41]:


df_train.corr().abs()[['SalePrice','SalePrice_Log']].sort_values(by='SalePrice', ascending=False)[2:16]


# In[ ]:





# ### Scatterplot colors: SalePrice vs. all_SF and OverallQual

# In[42]:


plotly_scatter_x_y_catg_color(df_train, 'all_SF', 'SalePrice', 'OverallQual')


# As seen before in the simple xatter plot, there is a strong tendency for 
# increasing SalePrice with a higher value for OverallQual.
# But this color plot also shows a correlation of all_SF and OverallQual.  
# So, the probability that a house has a large area increases with its Overall Quality.  
# And vice versa: Quality increases with House size.  
# This corrrelation is not necessary, one would expect that there are also small houses with high  
# quality and big houses with low quality.  
# It would be nice to know how the rating for OverallQual is calculated or estimated, 
# but that info is not included in the data description.

# In[ ]:





# Another option to highlight the correlation of SalePrice to all_SF and  
# OverallQual as well as the correlation between all_SF and OverallQual is  
# a 3d scatter plot:

# In[43]:


plotly_scatter3d(df_train, 'all_SF', 'OverallQual', 'SalePrice')


# Rotating the 3d view reveals that:
# 
# * SalePrice increases almost linearly with all_SF and OverallQual  
# * all_SF increases almost linearly with OverallQual and vice versa  
# 
# In fact, the bulk of the data follows the 45 degree line in 3 dim space.  
# This also results in the high correlation coefficient for OverallQual and all_SF:

# In[44]:


print(df_train['OverallQual'].corr(df_train['all_SF']))


# In[ ]:





# **other numerical features**

# In[45]:


print(df_train['OverallCond'].corr(df_train['SalePrice']))
print(df_train['OverallCond'].corr(df_train['SalePrice_Log']))


# In[46]:


print(df_train['MSSubClass'].corr(df_train['SalePrice']))
print(df_train['MSSubClass'].corr(df_train['SalePrice_Log']))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 1.4 Visualizations for Categorical features

# In[47]:


categorical_columns = df_train.select_dtypes(include=['object']).columns.tolist()


# ### Boxplots for categorical features

# In[48]:


def plotly_boxplots_sorted_by_yvals(df, catg_feature, sort_by_target):
    
    df_by_catg   = df.groupby([catg_feature])
    sortedlist_catg_str = df_by_catg[sort_by_target].median().sort_values().keys().tolist()
    
    
    data = []
    for i in sortedlist_catg_str :
        data.append(go.Box(y = df[df[catg_feature]==i][sort_by_target], name = i))

    layout = go.Layout(title = sort_by_target + " vs " + catg_feature, 
                       xaxis = dict(title = catg_feature), 
                       yaxis = dict(title = sort_by_target))

    fig = dict(data = data, layout = layout)
    return fig


# In[ ]:





# ### Boxplot: SalePrice for Neighborhood

# In[49]:


fig = plotly_boxplots_sorted_by_yvals(df_train, 'Neighborhood', 'SalePrice')
iplot(fig)


# ### Boxplot: SalePrice for MSZoning

# In[50]:


fig = plotly_boxplots_sorted_by_yvals(df_train, 'MSZoning', 'SalePrice')
iplot(fig)


# In[ ]:





# In[ ]:





# # PART 2: Preprocessing and Pipelines

# ## 2.0 define data for regression models

# In[51]:


outliers_all = []
df_train = df_train.drop(outliers_all)


# In[52]:


# store target as y and y_log:
y , y_log = df_train["SalePrice"] , df_train["SalePrice_Log"]
# drop target from df_train:
df_train.drop(["SalePrice", "SalePrice_Log"] , axis=1, inplace=True)


# **1: SalePriceLog as target**

# In[53]:


X_1 = df_train
y_1 = y_log


# **2: SalePrice as target**

# In[54]:


X_2 = df_train
y_2 = y


# In[ ]:





# ## 2.1 Pipeline approach

# **sklearn.pipeline**  
# The sklearn.pipeline module implements utilities to build a composite estimator, as a chain of transforms and estimators.

# https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline

# **sklearn.pipeline.Pipeline**  
# Pipeline of transforms with a final estimator.  
# Sequentially apply a list of transforms and a final estimator. Intermediate steps of the pipeline must be ‘transforms’, that is, they must implement fit and transform methods. The final estimator only needs to implement fit. The transformers in the pipeline can be cached using memory argument.  
# The purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters. For this, it enables setting parameters of the various steps using their names and the parameter name separated by a ‘__’, as in the example below. A step’s estimator may be replaced entirely by setting the parameter with its name to another estimator, or a transformer removed by setting to None.

# In[55]:


from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# ## 2.2 Preprocessing pipeline

# In[56]:


numerical_features   = df_train.select_dtypes(exclude=['object']).columns.tolist()
categorical_features = df_train.select_dtypes(include=['object']).columns.tolist()


# ### for numerical features

# In[57]:


numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])


# ### for categorical features

# In[58]:


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


# ### ColumnTransformer

# In[59]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer,   numerical_features),
        ('cat', categorical_transformer, categorical_features)])


# In[ ]:





# ## 2.3 Append regressors to pipeline

# ### 2.3.1 Linear Models

# Pipelines with default model parameters

# **LinearRegression** minimizes the residual sum of squares between the observed targets and the targets predicted by the linear approximation = Ordinary Least Squares fit  
# **Ridge** regression addresses some of the problems of Ordinary Least Squares by imposing a penalty on the size of the coefficients. The ridge coefficients minimize a penalized residual sum of squares.  
# **HuberRegressor** is different to Ridge because it applies a linear loss to samples that are classified as outliers.
# 

# In[60]:


# LinearRegression
pipe_Linear = Pipeline(
    steps   = [('preprocessor', preprocessor),
               ('Linear', LinearRegression()) ])    
# Ridge
pipe_Ridge = Pipeline(
    steps  = [('preprocessor', preprocessor),
              ('Ridge', Ridge(random_state=5)) ])  
# Huber
pipe_Huber = Pipeline(
    steps  = [('preprocessor', preprocessor),
              ('Huber', HuberRegressor()) ])  
# Lasso
pipe_Lasso = Pipeline(
    steps  = [ ('preprocessor', preprocessor),
               ('Lasso', Lasso(random_state=5)) ])
# ElasticNet
pipe_ElaNet = Pipeline(
    steps   = [ ('preprocessor', preprocessor),
                ('ElaNet', ElasticNet(random_state=5)) ])

# BayesianRidge
pipe_BayesRidge = Pipeline(
    steps   = [ ('preprocessor', preprocessor),
                ('BayesRidge', BayesianRidge(n_iter=500, compute_score=True)) ])


# ### 2.3.2 Ensemble Models

# Pipelines with default model parameters

# In[61]:


# GradientBoostingRegressor
pipe_GBR  = Pipeline(
    steps = [ ('preprocessor', preprocessor),
              ('GBR', GradientBoostingRegressor(random_state=5 )) ])

# XGBRegressor
pipe_XGB  = Pipeline(
    steps = [ ('preprocessor', preprocessor),
              ('XGB', XGBRegressor(objective='reg:squarederror', metric='rmse', 
                      random_state=5, nthread = -1)) ])
# LGBM
pipe_LGBM = Pipeline(
    steps= [('preprocessor', preprocessor),
            ('LGBM', LGBMRegressor(objective='regression', metric='rmse',
                                  random_state=5)) ])
# AdaBoostRegressor
pipe_ADA = Pipeline(
    steps= [('preprocessor', preprocessor),
            ('ADA', AdaBoostRegressor(DecisionTreeRegressor(), 
                random_state=5, loss='exponential')) ])


# In[ ]:





# # Part 3: Crossvalidation

# We now run a 5 fold cross validation for each pipeline/model:  
# Linear Models: Linear Regression, Ridge, Lasso, Elastic Net  
# Ensembles: GBR, XGB, LGBM, ADA  
# For this we create loops over two list of pipelines (Linear models and Ensembles) and calculate  
# the mean, std and min score (=error) for every model.  
# By this we get a first estimate for the different regression pipelines (Linear models and Ensembles):   
# We fit the the data (features X and target y) using the default model parameters.

# ## 3.1 Linear Models

# ### Loop over Pipelines: Linear

# In[62]:


list_pipelines = [pipe_Linear, pipe_Ridge, pipe_Huber, pipe_Lasso, pipe_ElaNet]


# In[63]:


print("model", "\t", "mean rmse", "\t", "std", "\t", "\t", "min rmse")
print("-+"*30)
for pipe in list_pipelines :
    
    scores = cross_val_score(pipe, X_1, y_1, scoring='neg_mean_squared_error', cv=5)
    scores = np.sqrt(-scores)
    print(pipe.steps[1][0], "\t", 
          '{:08.6f}'.format(np.mean(scores)), "\t",  
          '{:08.6f}'.format(np.std(scores)),  "\t", 
          '{:08.6f}'.format(np.min(scores)))


# Linear Regression and especially Ridge model give quite good results already with default parameters.  
# For Huber, Lasso and Elastic Net we need to tune hyperparameters (see Part 4: GridSearchCV)

# In[ ]:





# ## 3.2 Ensemble Models   

# ### Loop over Pipelines: Ensembles

# In[64]:


list_pipelines = [pipe_GBR, pipe_XGB, pipe_LGBM, pipe_ADA]


# In[65]:


print("model", "\t", "mean rmse", "\t", "std", "\t", "\t", "min rmse")
print("-+"*30)

for pipe in list_pipelines :
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=FutureWarning)
        scores = cross_val_score(pipe, X_1, y_1, scoring='neg_mean_squared_error', cv=5)
        scores = np.sqrt(-scores)
        print(pipe.steps[1][0], "\t", 
          '{:08.6f}'.format(np.mean(scores)), "\t",  
          '{:08.6f}'.format(np.std(scores)),  "\t", 
          '{:08.6f}'.format(np.min(scores)))


# Except for ADA Boost, the ensemble models with default parameters give  
# already good results for this task.

# In the following we check if we can improve these scores when we  
# optimize the model hyperparameters with GridSearchCV.

# In[ ]:





# # Part 4: GridSearchCV

# Preprocessing: Scalers

# In[66]:


list_scalers = [StandardScaler(), 
                RobustScaler(), 
                QuantileTransformer(output_distribution='normal')]


# For some linear models, QuantileTransformer gives best score for CV.  
# But for test score, performance is best with StandardScaler for all models.  
# Therefore:

# In[67]:


list_scalers = [StandardScaler()]


# ## 4.1 Linear Models

# #### Linear Regression

# **fit_intercept** : boolean, optional, default True  
# **normalize** : boolean, optional, default False  
# **copy_X** : boolean, optional, default True  
# **n_jobs** : int or None, optional (default=None)  
# The number of jobs to use for the computation. This will only provide speedup for n_targets > 1 and sufficient large problems. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.

# In[68]:


parameters_Linear = { 'preprocessor__num__scaler': list_scalers,
                     'Linear__fit_intercept':  [True,False],
                     'Linear__normalize':  [True,False] }

gscv_Linear = GridSearchCV(pipe_Linear, parameters_Linear, n_jobs=-1, 
                          scoring='neg_mean_squared_error', verbose=0, cv=5)
gscv_Linear.fit(X_1, y_1)


# In[69]:


print(np.sqrt(-gscv_Linear.best_score_))  
gscv_Linear.best_params_


# In[ ]:





# #### Ridge

# **alpha** :
# Regularization strength, must be a positive float  
# **fit_intercept** : bool, default True  
# **normalize** : boolean, optional, default False  
# **copy_X** : boolean, optional, default True  
# **max_iter** : int  
# **tol** : float  
# Precision of the solution  
# **solver** : {‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’}  
# Solver to use in the computational routines

# In[70]:


parameters_Ridge = { 'preprocessor__num__scaler': list_scalers,
                     'Ridge__alpha': [7,8,9],
                     'Ridge__fit_intercept':  [True,False],
                     'Ridge__normalize':  [True,False] }

gscv_Ridge = GridSearchCV(pipe_Ridge, parameters_Ridge, n_jobs=-1, 
                          scoring='neg_mean_squared_error', verbose=0, cv=5)
gscv_Ridge.fit(X_1, y_1)


# In[71]:


print(np.sqrt(-gscv_Ridge.best_score_))  
gscv_Ridge.best_params_


# **Huber Regressor**

# **epsilon** : > 1.0, default 1.35  
# controls the number of samples that should be classified as outliers.  
# The smaller the epsilon, the more robust it is to outliers.  
# **max_iter** : int, default 100  
# Maximum number of iterations that scipy.optimize.fmin_l_bfgs_b should run for.  
# **alpha** : float, default 0.0001  
# Regularization parameter.  
# **fit_intercept** : bool, default True

# In[72]:


parameters_Huber = { 'preprocessor__num__scaler': list_scalers,                   
                     'Huber__epsilon': [1.3, 1.35, 1.4],    
                     'Huber__max_iter': [150, 200, 250],                    
                     'Huber__alpha': [0.0005, 0.001, 0.002],
                     'Huber__fit_intercept':  [True], }

gscv_Huber = GridSearchCV(pipe_Huber, parameters_Huber, n_jobs=-1, 
                          scoring='neg_mean_squared_error', verbose=1, cv=5)
gscv_Huber.fit(X_1, y_1)


# In[73]:


print(np.sqrt(-gscv_Huber.best_score_))  
gscv_Huber.best_params_


# #### Lasso

# In[74]:


parameters_Lasso = { 'preprocessor__num__scaler': list_scalers,
                     'Lasso__alpha': [0.0005, 0.001],
                     'Lasso__fit_intercept':  [True],
                     'Lasso__normalize':  [True,False] }

gscv_Lasso = GridSearchCV(pipe_Lasso, parameters_Lasso, n_jobs=-1, 
                          scoring='neg_mean_squared_error', verbose=1, cv=5)
gscv_Lasso.fit(X_1, y_1)


# In[75]:


print(np.sqrt(-gscv_Lasso.best_score_))  
gscv_Lasso.best_params_


# **ElasticNet**

# In[76]:


parameters_ElaNet = { 'ElaNet__alpha': [0.0005, 0.001],
                      'ElaNet__l1_ratio':  [0.85, 0.9],
                      'ElaNet__normalize':  [True,False] }

gscv_ElaNet = GridSearchCV(pipe_ElaNet, parameters_ElaNet, n_jobs=-1, 
                          scoring='neg_mean_squared_error', verbose=1, cv=5)
gscv_ElaNet.fit(X_1, y_1)


# In[77]:


print(np.sqrt(-gscv_ElaNet.best_score_))  
gscv_ElaNet.best_params_


# In[ ]:





# In[ ]:





# ### Loop over GridSearchCV Pipelines: Linear

# In[78]:


list_pipelines_gscv = [gscv_Linear,gscv_Ridge,gscv_Huber,gscv_Lasso,gscv_ElaNet]


# In[79]:


print("model", "\t", "mean rmse", "\t", "std", "\t", "\t", "min rmse")
print("-+"*30)
for gscv in list_pipelines_gscv :
    
    scores = cross_val_score(gscv.best_estimator_, X_1, y_1, 
                             scoring='neg_mean_squared_error', cv=5)
    scores = np.sqrt(-scores)
    print(gscv.estimator.steps[1][0], "\t", 
          '{:08.6f}'.format(np.mean(scores)), "\t",  
          '{:08.6f}'.format(np.std(scores)),  "\t", 
          '{:08.6f}'.format(np.min(scores)))


# After GridSearchCV, results for Lasso and Elastic Net are much better compared to using the default parameters.  
# The Ridge model also improves a bit, score for Linear Regression is the same as with default parameters.  
# Huber regression is not better than Ordinary least Squares for this task.

# In[ ]:





# ## 4.2 Ensemble Models

# **GradientBoostingRegressor**

# **loss** : {‘ls’, ‘lad’, ‘huber’, ‘quantile’}, optional (default=’ls’)  
# **learning_rate** : float, optional (default=0.1)  
# **n_estimators** : int (default=100)  
# **subsample** : float, optional (default=1.0)  
# **criterion** : string, optional (default=”friedman_mse”)  
# **min_samples_split** : int, float, optional (default=2)  
# If int: minimum number. If float: fraction  
# **min_samples_leaf** : int, float, optional (default=1)   
# If int minimum number. If float fraction   
# **min_weight_fraction_leaf** : float, optional (default=0.)  
# **max_depth** : integer, optional (default=3)
# maximum depth of the individual regression estimators.  
# **min_impurity_decrease** : float, optional (default=0.)
# 
# **max_features** : int, float, string or None, optional (default=None)  
# The number of features to consider when looking for the best split:  
# If float: fraction  
# If “auto”, then max_features=n_features.  
# If “sqrt”, then max_features=sqrt(n_features).  
# If “log2”, then max_features=log2(n_features).  
# If None, then max_features=n_features.  
# Choosing max_features < n_features leads to a reduction of variance and an increase in bias.  
# 

# In[80]:


parameters_GBR = { 'GBR__n_estimators':  [400], 
                   'GBR__max_depth':  [3,4],
                   'GBR__min_samples_leaf':  [5,6],                 
                   'GBR__max_features':  ["auto",0.5,0.7],                  
                 }
                   
gscv_GBR = GridSearchCV(pipe_GBR, parameters_GBR, n_jobs=-1, 
                        scoring='neg_mean_squared_error', verbose=1, cv=5)
gscv_GBR.fit(X_1, y_1)


# In[81]:


print(np.sqrt(-gscv_GBR.best_score_))  
gscv_GBR.best_params_


# **XGB**

# https://xgboost.readthedocs.io/en/latest/parameter.html  
# https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html  
# https://xgboost.readthedocs.io/en/latest/python/python_api.html

# General Parameters  
# **booster**: gbtree, gblinear or dart, default= gbtree   
# 
# Parameters for Tree Booster  
# **eta**, alias: learning_rate, 0<eta<1 , default=0.3  
# **gamma**, alias: min_split_loss,  default=0,  
# Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.  
# **max_depth**, default=6  
# **min_child_weight**, default=1
# The larger min_child_weight is, the more conservative the algorithm will be.  
# **max_delta_step** [default=0]  
# **subsample** [default=1],  range: (0,1]  
# Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. Subsampling will occur once in every boosting iteration.  
# **colsample_bytree, colsample_bylevel, colsample_bynode** [default=1]  
# This is a family of parameters for subsampling of columns.  
# All colsample_by* parameters have a range of (0, 1], the default value of 1, and specify the fraction of columns to be subsampled.  
# colsample_by* parameters work cumulatively. For instance, the combination {'colsample_bytree':0.5, 'colsample_bylevel':0.5, 'colsample_bynode':0.5} with 64 features will leave 8 features to choose from at each split.  
# **lambda** [default=1, alias: reg_lambda]  
# L2 regularization term on weights. Increasing this value will make model more conservative.  
# **alpha** [default=0, alias: reg_alpha]  
# L1 regularization term on weights. Increasing this value will make model more conservative.  
# **tree_method** string [default= auto]  
# Choices: auto, exact, approx, hist, gpu_hist  

# In[82]:


parameters_XGB = { 'XGB__learning_rate': [0.021,0.022],
                   'XGB__max_depth':  [2,3],
                   'XGB__n_estimators':  [2000], 
                   'XGB__reg_lambda':  [1.5, 1.6], 
                   'XGB__reg_alpha':  [1,1.5],                   
# colsample_bytree , subsample               
                  }
                   
gscv_XGB = GridSearchCV(pipe_XGB, parameters_XGB, n_jobs=-1, 
                        scoring='neg_mean_squared_error', verbose=1, cv=5)
gscv_XGB.fit(X_1, y_1)


# In[83]:


print(np.sqrt(-gscv_XGB.best_score_))  
gscv_XGB.best_params_


# #### LGBM

# https://testlightgbm.readthedocs.io/en/latest/Parameters.html  
# https://testlightgbm.readthedocs.io/en/latest/Parameters-tuning.html  

# **num_iterations**, default=100, alias=num_iteration,num_tree,num_trees,num_round,num_rounds  
# **learning_rate**, default=0.1, alias=shrinkage_rate  
# **num_leaves**, default=31
# 
# 
# **max_depth**, default=-1, < 0 means no limit  
# **min_data_in_leaf**, default=20, type=int, alias=min_data_per_leaf , min_data  
# **min_sum_hessian_in_leaf**, default=1e-3, alias=min_sum_hessian_per_leaf, min_sum_hessian, min_hessian  
# **feature_fraction**, default=1.0, 0.0 < feature_fraction < 1.0, alias=sub_feature  
# **bagging_fraction**, default=1.0, 0.0 < bagging_fraction < 1.0, alias=sub_row  
# **bagging_freq**, default=0,   
# Frequency for bagging, 0 means disable bagging. k means will perform bagging at every k iteration   
# **early_stopping_round** , default=0, type=int, alias=early_stopping_rounds,early_stopping  
# Will stop training if one metric of one validation data doesn’t improve in last early_stopping_round rounds  
# **lambda_l1** , default=0  
# **lambda_l2** , default=0

# In[84]:


parameters_LGBM = { 'LGBM__learning_rate': [0.01,0.02],
                    'LGBM__n_estimators':  [1000], 
                    'LGBM__num_leaves':  [8,10],
                    'LGBM__bagging_fraction':  [0.7,0.8],
                    'LGBM__bagging_freq':  [1,2],                  
                   }

gscv_LGBM = GridSearchCV(pipe_LGBM, parameters_LGBM, n_jobs=-1, 
                       scoring='neg_mean_squared_error', verbose=1, cv=5)
gscv_LGBM.fit(X_1, y_1)


# In[85]:


print(np.sqrt(-gscv_LGBM.best_score_))  
gscv_LGBM.best_params_


# In[ ]:





# #### AdaBoostRegressor

# In[86]:


parameters_ADA = { 'ADA__learning_rate': [3.5],
                   'ADA__n_estimators':  [500], 
                   'ADA__base_estimator__max_depth':  [8,9,10],                  
                 }

pipe_ADA = Pipeline(
    steps= [('preprocessor', preprocessor),
            ('ADA', AdaBoostRegressor(
                DecisionTreeRegressor(min_samples_leaf=5,
                                      min_samples_split=5), 
                random_state=5,loss='exponential')) ])

gscv_ADA = GridSearchCV(pipe_ADA, parameters_ADA, n_jobs=-1, 
                       scoring='neg_mean_squared_error', verbose=1, cv=5)
gscv_ADA.fit(X_1, y_1)


# In[87]:


print(np.sqrt(-gscv_ADA.best_score_))  
gscv_ADA.best_params_


# In[ ]:





# ### Loop over GridSearchCV Pipelines: Ensembles

# In[88]:


list_pipelines_gscv = [gscv_GBR, gscv_XGB,gscv_LGBM,gscv_ADA]


# In[89]:


print("model", "\t", "mean rmse", "\t", "std", "\t", "\t", "min rmse")
print("-+"*30)
for gscv in list_pipelines_gscv :
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=FutureWarning)    
        scores = cross_val_score(gscv.best_estimator_, X_1, y_1, 
                             scoring='neg_mean_squared_error', cv=5)
        scores = np.sqrt(-scores)
        print(gscv.estimator.steps[1][0], "\t", 
          '{:08.6f}'.format(np.mean(scores)), "\t",  
          '{:08.6f}'.format(np.std(scores)),  "\t", 
          '{:08.6f}'.format(np.min(scores)))


# In[ ]:





# # Part 5: Predictions for test data

# list of models

# In[90]:


linear_models = [gscv_Linear,gscv_Ridge,gscv_Huber,gscv_Lasso,gscv_ElaNet]
boost_models  = [gscv_GBR, gscv_XGB,gscv_LGBM,gscv_ADA]


# **Linear Models**

# In[91]:


pred_Linear = gscv_Linear.predict(df_test)
pred_Ridge  = gscv_Ridge.predict(df_test)
pred_Huber  = gscv_Huber.predict(df_test)
pred_Lasso  = gscv_Lasso.predict(df_test)
pred_ElaNet = gscv_ElaNet.predict(df_test)


# In[92]:


predictions_linear = {'Linear': pred_Linear, 'Ridge': pred_Ridge, 'Huber': pred_Huber,
                      'Lasso':  pred_Lasso, 'ElaNet': pred_ElaNet }


# In[93]:


for model,values in predictions_linear.items():
    str_filename = model + ".csv"
    print("witing submission to : ", str_filename)
    subm = pd.DataFrame()
    subm['Id'] = id_test
    subm['SalePrice'] = np.expm1(values)
    subm.to_csv(str_filename, index=False)


# **blend_1: gscv_Ridge and gscv_Lasso**

# In[94]:


pred_Blend_1 = (pred_Lasso + pred_Ridge) / 2
sub_Blend_1 = pd.DataFrame()
sub_Blend_1['Id'] = id_test
sub_Blend_1['SalePrice'] = np.expm1(pred_Blend_1)
sub_Blend_1.to_csv('Blend_Ridge_Lasso.csv',index=False)
sub_Blend_1.head()


# **blend_2: gscv_Lasso and gscv_ElaNet**

# In[95]:


pred_Blend_2 = (pred_Lasso + pred_ElaNet) / 2
sub_Blend_2 = pd.DataFrame()
sub_Blend_2['Id'] = id_test
sub_Blend_2['SalePrice'] = np.expm1(pred_Blend_2)
sub_Blend_2.to_csv('Blend_2.csv',index=False)
sub_Blend_2.head()


# **blend_3: gscv_Ridge, gscv_Lasso and gscv_ElaNet**

# In[96]:


pred_Blend_3 = (pred_Ridge + pred_Lasso + pred_ElaNet) / 3
sub_Blend_3 = pd.DataFrame()
sub_Blend_3['Id'] = id_test
sub_Blend_3['SalePrice'] = np.expm1(pred_Blend_3)
sub_Blend_3.to_csv('Blend_3.csv',index=False)
sub_Blend_3.head()


# In[ ]:





# **Boost Models**

# In[97]:


boost_models  = [gscv_GBR, gscv_XGB,gscv_LGBM,gscv_ADA]


# In[98]:


pred_GBR  = gscv_GBR.predict(df_test)
pred_XGB  = gscv_XGB.predict(df_test)
pred_LGBM = gscv_LGBM.predict(df_test)
pred_ADA  = gscv_ADA.predict(df_test)


# In[99]:


predictions_boost = {'GBR': pred_GBR, 'XGB': pred_XGB, 'LGBM': pred_LGBM,
                     'ADA': pred_ADA }


# In[100]:


for model,values in predictions_boost.items():
    str_filename = model + ".csv"
    print("witing submission to : ", str_filename)
    subm = pd.DataFrame()
    subm['Id'] = id_test
    subm['SalePrice'] = np.expm1(values)
    subm.to_csv(str_filename, index=False)


# In[ ]:





# ## 5.1 Correlation of predictions

# In[101]:


predictions = {'Ridge': pred_Ridge, 'Lasso': pred_Lasso, 'ElaNet': pred_ElaNet, 
               'GBR': pred_GBR, 'XGB': pred_XGB, 'LGBM': pred_LGBM, 'ADA': pred_ADA}
df_predictions = pd.DataFrame(data=predictions) 
df_predictions.corr()


# **to be continued**

# In[ ]:





# Blend: Ridge + XGB

# In[102]:


pred_Blend_10 = (pred_Ridge + pred_XGB) / 2
sub_Blend_10 = pd.DataFrame()
sub_Blend_10['Id'] = id_test
sub_Blend_10['SalePrice'] = np.expm1(pred_Blend_10)
sub_Blend_10.to_csv('Blend_Ridge_XGB.csv',index=False)
sub_Blend_10.head()


# In[ ]:





# In[ ]:





# ## 5.3 Stacking

# In[103]:


lnr = LinearRegression(n_jobs = -1)

rdg = Ridge(alpha=3.0, copy_X=True, fit_intercept=True, random_state=1)

rft = RandomForestRegressor(n_estimators = 12, max_depth = 3, n_jobs = -1, random_state=1)

gbr = GradientBoostingRegressor(n_estimators = 40, max_depth = 2, random_state=1)

mlp = MLPRegressor(hidden_layer_sizes = (90, 90), alpha = 2.75, random_state=1)


# stack1

# In[104]:


stack1 = StackingRegressor(regressors = [rdg, rft, gbr], 
                           meta_regressor = lnr)


# In[105]:


pipe_STACK_1 = Pipeline(steps=[ ('preprocessor', preprocessor),
                                ('stack1', stack1) ])

pipe_STACK_1.fit(X_1, y_1)


# In[ ]:





# https://stackoverflow.com/questions/50722270/convergence-warningstochastic-optimizer-maximum-iterations-200-reached-and-t?rq=1

# In[106]:


pred_stack1 = pipe_STACK_1.predict(df_test)
sub_stack1 = pd.DataFrame()
sub_stack1['Id'] = id_test
sub_stack1['SalePrice'] = np.expm1(pred_stack1)
sub_stack1.to_csv('pipe_stack1.csv',index=False)


# In[107]:


sub_stack1.head(10)

