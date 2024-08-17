#!/usr/bin/env python
# coding: utf-8

# # <b>1 <span style='color:lightseagreen'>|</span> Introduction</b>
# 
# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>1.1 | House Prices Prediction Competition</b></p>
# </div>
# 
# Ask a home buyer to describe their **<span style='color:lightseagreen'>dream house</span>**, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this **<span style='color:lightseagreen'>playground competition's dataset</span>** proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
# 
# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/5407/media/housesbanner.png)
# 
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home. Practice skills: 
# 
# 1. Creative feature engineering 
# 2. Advanced regression techniques like random forest and gradient boosting
# 
# 
# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>1.2 | Acknowledgements</b></p>
# </div>
# 
# * Kaggle's [Feature Engineering](https://www.kaggle.com/learn/feature-engineering).

# In[1]:


get_ipython().system('pip install pycaret')
import os
import warnings
from pathlib import Path
from IPython.display import clear_output
from IPython.display import display
from pandas.api.types import CategoricalDtype

# Basic libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_profiling as pp
import seaborn as sns

# Clustering
from sklearn.cluster import KMeans

# Principal Component Analysis (PCA)
from sklearn.decomposition import PCA

#Mutual Information
from sklearn.feature_selection import mutual_info_regression

# Cross Validation
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, learning_curve, train_test_split

# Encoders
from category_encoders import MEstimateEncoder
from sklearn.preprocessing import LabelEncoder
from category_encoders import MEstimateEncoder

# Algorithms
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor

# Optuna - Bayesian Optimization 
import optuna
from optuna.samplers import TPESampler

# Plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.offline as offline
import plotly.graph_objs as go

# Metric
from sklearn.metrics import mean_absolute_error as mae

# PyCaret
from pycaret.regression import *

# Permutation Importance
import eli5
from eli5.sklearn import PermutationImportance

warnings.filterwarnings('ignore')

def reduce_mem_usage(df, verbose=True):
    numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
 
    return df

def load_data():
    data_dir = Path("../input/house-prices-advanced-regression-techniques/")
    df_train = pd.read_csv(data_dir / "train.csv", index_col="Id")
    df_test = pd.read_csv(data_dir / "test.csv", index_col="Id")
    # Merge the splits so we can process them together
    df = pd.concat([df_train, df_test])
    return df
df_data = load_data()
clear_output()


# In[2]:


#pp.ProfileReport(df_data)


# # <b>2 <span style='color:lightseagreen'>|</span> Exploratory Data Analysis</b>
# 
# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.1 | Target Analysis</b></p>
# </div>
# 
# ### 2.1.1 | Numerical Features

# In[3]:


# Defining all our palette colours.
primary_blue = "#496595"
primary_blue2 = "#85a1c1"
primary_blue3 = "#3f4d63"
primary_grey = "#c6ccd8"
primary_black = "#202022"
primary_bgcolor = "#f4f0ea"

# "coffee" pallette turqoise-gold.
f1 = "#a2885e"
f2 = "#e9cf87"
f3 = "#f1efd9"
f4 = "#8eb3aa"
f5 = "#235f83"
f6 = "#b4cde3"

def plot_box(fig, feature, r, c):
    fig.add_trace(go.Box(x=df_data[feature].astype(object), y=df_data.SalePrice, marker = dict(color= px.colors.sequential.Viridis_r[5])), row =r, col = c)
    fig.update_xaxes(showgrid = False, showline = True, linecolor = 'gray', linewidth = 2, zeroline = False,row = r, col = c)
    fig.update_yaxes(showgrid = False, gridcolor = 'gray', gridwidth = 0.5, showline = True, linecolor = 'gray', linewidth = 2, row = r, col = c)
    
def plot_scatter(fig, feature, r, c):
    fig.add_trace(go.Scatter(x=df_data[feature], y=df_data.SalePrice, mode='markers', marker = dict(color=np.random.randn(10000), colorscale = px.colors.sequential.Viridis)), row = r, col = c)
    fig.update_xaxes(showgrid = False, showline = True, linecolor = 'gray', linewidth = 2, zeroline = False, row = r, col = c)
    fig.update_yaxes(showgrid = False, gridcolor = 'gray', gridwidth = 0.5, showline = True, linecolor = 'gray', linewidth = 2, row = r, col = c)
    

def plot_hist(fig, feature, r, c):
    fig.add_trace(go.Histogram(x=df_data['SalePrice'], name='Sale Price Distribution', marker = dict(color = px.colors.sequential.Viridis_r[5])), row = r, col = c)
    fig.update_xaxes(showgrid = False, showline = True, linecolor = 'gray', linewidth = 2, row = r, col = c)
    fig.update_yaxes(showgrid = False, gridcolor = 'gray', gridwidth = 0.5, showline = True, linecolor = 'gray', linewidth = 2, row = r, col = c)
    
# chart
fig = make_subplots(rows=6, cols=3, 
                    specs=[[{"type": "scatter"}, {"type":"histogram"}, {'type':'scatter'}], [{'type':'scatter'}, {'type':'scatter'}, 
                    {'type':'scatter'}], [{'type':'box'}, {'type':'scatter'}, {'type':'scatter'}], [{"type": "scatter"}, {"type":"box"}, 
                    {'type':'scatter'}],[{"type": "box"}, {"type":"scatter"}, {'type':'box'}],[{"type": "scatter"}, {"type":"scatter"}, 
                    {'type':'scatter'}],], column_widths=[0.34, 0.33, 0.33], 
                    vertical_spacing=0.05, horizontal_spacing=0.1, subplot_titles=('Target vs LotFrontage',"Target Distribution",
                    'Target vs MasVnrArea','Target vs BsmtFinSF1','Target vs BsmtFinSF2','Target vs TotalBsmtSF','Target vs BsmtFullBath',
                    'Target vs GarageArea','Target vs GarageCars','Target vs LotArea','Target vs MSSubClass','Target vs YearBuilt','Target vs OverallQual',
                    'Target vs YearRemodAdd','Target vs OverallCond','Target vs GrLivArea','Target vs 1stFlrSF','Target vs 2ndFlrSF'))

# Charts
plot_scatter(fig, 'LotFrontage',1,1)
plot_hist(fig, 'SalePrice',1,2)
plot_scatter(fig, 'MasVnrArea',1,3)
plot_scatter(fig, 'BsmtFinSF1',2,1)
plot_scatter(fig, 'BsmtFinSF2',2,2)
plot_scatter(fig, 'TotalBsmtSF',2,3)
plot_box(fig, 'BsmtFullBath', 3, 1)
plot_scatter(fig, 'GarageArea',3,2)
plot_box(fig, 'GarageCars', 3, 3)
plot_scatter(fig, 'LotArea',4,1)
plot_box(fig, 'MSSubClass',4,2)
plot_scatter(fig, 'YearBuilt',4,3)
plot_box(fig, 'OverallQual',5,1)
plot_scatter(fig, 'YearRemodAdd',5,2)
plot_box(fig, 'OverallCond',5,3)
plot_scatter(fig, 'GrLivArea',6,1)
plot_scatter(fig, '1stFlrSF', 6, 2)
plot_scatter(fig, '2ndFlrSF', 6, 3)

# General Styling
fig.update_layout(height=1500, bargap=0.2,
                  margin=dict(b=50,r=30,l=100),
                  title = "<span style='font-size:36px; font-family:Times New Roman'>Sales Prices Analysis</span>",                  
                  plot_bgcolor='rgb(242,242,242)',
                  paper_bgcolor = 'rgb(242,242,242)',
                  font=dict(family="Times New Roman", size= 14),
                  hoverlabel=dict(font_color="floralwhite"),
                  showlegend=False)


# ### 2.1.2 | Categorical Features

# In[4]:


categorical_col = df_data.select_dtypes(['object']).columns[0:21]

def plot_box(fig, feature, r, c):
    fig.add_trace(go.Box(x=df_data[feature].astype(object), y=df_data.SalePrice, marker = dict(color= px.colors.sequential.Viridis_r[5])), row =r, col = c)
    fig.update_xaxes(showgrid = False, showline = True, linecolor = 'gray', linewidth = 2, zeroline = False,row = r, col = c)
    fig.update_yaxes(showgrid = False, gridcolor = 'gray', gridwidth = 0.5, showline = True, linecolor = 'gray', linewidth = 2, row = r, col = c)
    
def plot_scatter(fig, feature, r, c):
    fig.add_trace(go.Scatter(x=df_data[feature], y=df_data.SalePrice, mode='markers', marker = dict(color=np.random.randn(10000), colorscale = px.colors.sequential.Viridis)), row = r, col = c)
    fig.update_xaxes(showgrid = False, showline = True, linecolor = 'gray', linewidth = 2, zeroline = False, row = r, col = c)
    fig.update_yaxes(showgrid = False, gridcolor = 'gray', gridwidth = 0.5, showline = True, linecolor = 'gray', linewidth = 2, row = r, col = c)
    

def plot_hist(fig, feature, r, c):
    fig.add_trace(go.Histogram(x=df_data['SalePrice'], name='Sale Price Distribution', marker = dict(color = px.colors.sequential.Viridis_r[5])), row = r, col = c)
    fig.update_xaxes(showgrid = False, showline = True, linecolor = 'gray', linewidth = 2, row = r, col = c)
    fig.update_yaxes(showgrid = False, gridcolor = 'gray', gridwidth = 0.5, showline = True, linecolor = 'gray', linewidth = 2, row = r, col = c)
    
# chart
fig = make_subplots(rows=7, cols=3, 
                    column_widths=[0.34, 0.33, 0.33], 
                    vertical_spacing=0.05, horizontal_spacing=0.1, subplot_titles=('Target vs MSZoning',"Target vs Street",
                    'Target vs Alley','Target vs LotShape','Target vs LandContour','Target vs Utilities','Target vs LotConfig',
                    'Target vs LandSlope','Target vs Neighborhood','Target vs Condition1','Target vs Condition2','Target vs BldgType','Target vs HouseStyle',
                    'Target vs RoofStyle','Target vs RoofMatl','Target vs Exterior1st','Target vs Exterior2nd','Target vs MasVnrType',
                    'Target vs ExterQual','Target vs ExterCond','Target vs Foundation'))

for i in range(1,22):
        if i%3 != 0:
            plot_box(fig, categorical_col[i-1], int(i/3)+1, int(i%3))
        else:
            plot_box(fig, categorical_col[i-1], int(i/3), 3)

# General Styling
fig.update_layout(height=2000, bargap=0.2,
                  margin=dict(b=50,r=30,l=100),
                  title = "<span style='font-size:36px; font-family:Times New Roman'>Sales Prices Analysis</span>",                  
                  plot_bgcolor='rgb(242,242,242)',
                  paper_bgcolor = 'rgb(242,242,242)',
                  font=dict(family="Times New Roman", size= 14),
                  hoverlabel=dict(font_color="floralwhite"),
                  showlegend=False)


# In[5]:


categorical_col = df_data.select_dtypes(['object']).columns[21:42]

def plot_box(fig, feature, r, c):
    fig.add_trace(go.Box(x=df_data[feature].astype(object), y=df_data.SalePrice, marker = dict(color= px.colors.sequential.Viridis_r[5])), row =r, col = c)
    fig.update_xaxes(showgrid = False, showline = True, linecolor = 'gray', linewidth = 2, zeroline = False,row = r, col = c)
    fig.update_yaxes(showgrid = False, gridcolor = 'gray', gridwidth = 0.5, showline = True, linecolor = 'gray', linewidth = 2, row = r, col = c)
    
def plot_scatter(fig, feature, r, c):
    fig.add_trace(go.Scatter(x=df_data[feature], y=df_data.SalePrice, mode='markers', marker = dict(color=np.random.randn(10000), colorscale = px.colors.sequential.Viridis)), row = r, col = c)
    fig.update_xaxes(showgrid = False, showline = True, linecolor = 'gray', linewidth = 2, zeroline = False, row = r, col = c)
    fig.update_yaxes(showgrid = False, gridcolor = 'gray', gridwidth = 0.5, showline = True, linecolor = 'gray', linewidth = 2, row = r, col = c)
    

def plot_hist(fig, feature, r, c):
    fig.add_trace(go.Histogram(x=df_data['SalePrice'], name='Sale Price Distribution', marker = dict(color = px.colors.sequential.Viridis_r[5])), row = r, col = c)
    fig.update_xaxes(showgrid = False, showline = True, linecolor = 'gray', linewidth = 2, row = r, col = c)
    fig.update_yaxes(showgrid = False, gridcolor = 'gray', gridwidth = 0.5, showline = True, linecolor = 'gray', linewidth = 2, row = r, col = c)
    
# chart
fig = make_subplots(rows=7, cols=3, 
                    column_widths=[0.34, 0.33, 0.33], 
                    vertical_spacing=0.05, horizontal_spacing=0.1, subplot_titles=('Target vs BsmtQual',"Target vs BsmtCond",
                    'Target vs BsmtExposure','Target vs BsmtFinType1','Target vs BsmtFinType2','Target vs Heating','Target vs HeatingQC',
                    'Target vs CentralAir','Target vs Electrical','Target vs KitchenQual','Target vs Functional','Target vs FireplaceQu','Target vs GarageType',
                    'Target vs GarageFinish','Target vs GarageQual','Target vs GarageCond','Target vs PavedDrive','Target vs PoolQC', 
                     'Target vs Fence', 'Target vs MiscFeature','Saletype'))

for i in range(1,22):
        if i%3 != 0:
            plot_box(fig, categorical_col[i-1], int(i/3)+1, int(i%3))
        else:
            plot_box(fig, categorical_col[i-1], int(i/3), 3)

# General Styling
fig.update_layout(height=2000, bargap=0.2,
                  margin=dict(b=50,r=30,l=100),
                  title = "<span style='font-size:36px; font-family:Times New Roman'>Sales Prices Analysis</span>",                  
                  plot_bgcolor='rgb(242,242,242)',
                  paper_bgcolor = 'rgb(242,242,242)',
                  font=dict(family="Times New Roman", size= 14),
                  hoverlabel=dict(font_color="floralwhite"),
                  showlegend=False)


# ### 2.1.3 | Heatmap - Most Correlated Features with Target

# In[6]:


corr = df_data[df_data.select_dtypes(['float','int']).columns].corr()
highest_corr_features = corr.index[abs(corr["SalePrice"])>0.5]

fig = px.imshow(df_data[highest_corr_features].corr(), color_continuous_scale='RdBu_r', origin='lower', text_auto=True, aspect='auto')
fig.update_xaxes(showgrid = False, linecolor='gray', linewidth = 2, zeroline = False)
fig.update_yaxes(showgrid = True, gridcolor='gray',gridwidth=0.5, linecolor='gray',linewidth=2, zeroline = False)

# General Styling
fig.update_layout(height=500, bargap=0.2,
                  margin=dict(b=50,r=30,l=100, t=100),
                  title = "<span style='font-size:36px; font-family:Times New Roman'>Heatmap - Most Correlated Features with Target</span>",                  
                  plot_bgcolor='rgb(242,242,242)',
                  paper_bgcolor = 'rgb(242,242,242)',
                  font=dict(family="Times New Roman", size= 14),
                  hoverlabel=dict(font_color="floralwhite"),
                  showlegend=False)
fig.show()


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.2 | OverallQual Analysis</b></p>
# </div>

# In[7]:


df_graph = df_data[df_data['SalePrice'].isnull() == False]

# chart
fig = make_subplots(rows=1, cols=3, 
                    specs=[[{"type": "bar"}, {"type": "pie"}, {'type':'bar'}]],
                    column_widths=[0.33, 0.34, 0.33], vertical_spacing=0.1, horizontal_spacing=0.1,
                    subplot_titles=("Mean Price per OverallQual", "Sold Houses per OverallQual", "TotalBsmtSF per OverallQual"))

# Left chart
df_qual = df_graph.groupby(['OverallQual']).agg({"SalePrice" : "mean"})
values = list(range(10))
fig.add_trace(go.Bar(x=df_qual.index, y=df_qual['SalePrice'], marker = dict(color=values, colorscale="Blugrn"), name='Mean Price'), row=1, col=1)

fig.update_xaxes(showgrid = False, linecolor='gray', linewidth = 2, zeroline = False, row=1, col=1)
fig.update_yaxes(showgrid = False, linecolor='gray',linewidth=2, zeroline = False, row=1, col=1)

# Middle Chart
df_qual = df_graph.groupby(['OverallQual']).agg({"SalePrice" : "count"})
values = list(range(10))
fig.add_trace(go.Pie(values=df_qual['SalePrice'], labels=df_qual.index, name='Count',
                     marker=dict(colors=px.colors.sequential.Blugrn)), row=1, col=2)

# Right Chart
df_qual = df_graph.groupby(['OverallQual']).agg({"TotalBsmtSF" : "mean"})
values = list(range(10))
fig.add_trace(go.Bar(x=df_qual.index, y=df_qual['TotalBsmtSF'], marker = dict(color=values, colorscale="Blugrn"), name='Count'), row=1, col=3)

fig.update_xaxes(showgrid = False, linecolor='gray', linewidth = 2, zeroline = False, row=1, col=3)
fig.update_yaxes(showgrid = False, linecolor='gray',linewidth=2, zeroline = False, row=1, col=3)

# General Styling
fig.update_layout(height=400, bargap=0.2,
                  margin=dict(b=50,r=30,l=100),
                  title = "<span style='font-size:36px; font-family:Times New Roman'>OverallQual Analysis</span>",                  
                  plot_bgcolor='rgb(242,242,242)',
                  paper_bgcolor = 'rgb(242,242,242)',
                  font=dict(family="Times New Roman", size= 14),
                  hoverlabel=dict(font_color="floralwhite"),
                  showlegend=False)


# # <b>2 <span style='color:lightseagreen'>|</span> Data Cleaning</b>
# 
# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.1 | Garage, Basement, Frontage, Alley, Utilities, Fireplace and Masonry veneer</b></p>
# </div>
# 
# We'll start by loading the data and **<span style='color:lightseagreen'>concatenate</span>** train and test datasets, in order to **<span style='color:lightseagreen'>preprocess</span>** it, and then divide them again. We are going to start with some features that not every home has. We have some missing values for Garage and Basement. It is easily interpretable that they are due to its **<span style='color:lightseagreen'>absence</span>** in those homes, as not all houses have for example a Fire Place. Thus, we are going to replace missing values as follows: 
# 
# * For numerical features, we will replace NaN for 0.0
# * For object features, we will replace NaN for None.

# In[8]:


missing_col = [col for col in df_data.columns if df_data[col].isnull().sum() >0]
print('{}\n'.format(missing_col))


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.2 | MSZoning, Exterior Covering, Kitchen Quality, Functional, SaleType</b></p>
# </div>
# 
# As you can observe, all of these features are categorical. Moreover, they have a **<span style='color:lightseagreen'>small percentage</span>** of missing values as show below. Therefore, we are going to fill these features missing values with most repeated values for each feature.

# In[9]:


columns = ['MSZoning','Exterior1st', 'Exterior2nd','KitchenQual', 'Functional','SaleType']
for col in columns: 
    print('Missing values percentage for {}: {}'.format(col, df_data[col].isnull().sum()/df_data.shape[0]))


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.3 | PoolQC, Fence, MiscFeature</b></p>
# </div>
# 
# Finally, we are going to consider features with a **<span style='color:lightseagreen'>high percentage</span>** of missing values. Although they should be very uninformative features, we are going to keep them **<span style='color:lightseagreen'>until Feature Importance Analysis</span>**, which is next section. There, we'll decide whether to keep these features or not.

# In[10]:


high_perc_columns = ['PoolQC','Fence','MiscFeature']
for col in high_perc_columns: 
    print('Missing values percentage for {}: {}'.format(col, df_data[col].isnull().sum()/df_data.shape[0]))


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.4 | Missing Values Function</b></p>
# </div>

# In[11]:


def fill_missing_values(df):
    df_data = df.copy()
    absent_features = ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
                  'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
                     'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Electrical', 'BsmtFullBath', 'BsmtHalfBath', 
                  'LotFrontage','Alley','Utilities','FireplaceQu','MasVnrType', 'MasVnrArea']

    numerical_features = [col for col in absent_features if df_data[col].dtype == float]
    object_features = df_data.select_dtypes(["object"]).columns

    df_data[numerical_features] = df_data[numerical_features].fillna(0.0)
    df_data[object_features] = df_data[object_features].fillna('None')
    
    columns = ['MSZoning','Exterior1st', 'Exterior2nd','KitchenQual', 'Functional','SaleType']
    for column in columns: 
        moda = df_data[column].value_counts().index[0]
        df_data[column] = df_data[column].fillna(moda)
        
    high_perc_columns = ['PoolQC','Fence','MiscFeature']
    for col in high_perc_columns: 
        if col == 'Fence' or col == 'MiscFeature':
            df_data[col] = df_data[col].fillna('None')
            
    return df_data


# # <b>3 <span style='color:lightseagreen'>|</span> Feature Engineering</b>
# 
# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>3.1 | Heatmap - Correlation</b></p>
# </div>
# 
# We'll start by looking at what features have a **<span style='color:lightseagreen'>huge impact</span>** on sales prices. In our case, we are going to choose those features such that: **<span style='color:lightseagreen'>$|\gamma|>0.5$</span>**, where $\gamma$ is the correlation coefficient of each feature respect to sales price. To do that, we are going to plot a heatmap which is going to show us correlations between features and sale price.

# In[12]:


df_data = fill_missing_values(df_data)
df_train = df_data[df_data['SalePrice'].isnull() == False]
df_test = df_data[df_data['SalePrice'].isnull() == True]

corr = df_train.corr()
highest_corr_features = corr.index[abs(corr["SalePrice"])>0.5]

fig = px.imshow(df_train[highest_corr_features].corr(), color_continuous_scale='RdBu_r', origin='lower', text_auto=True, aspect='auto')
fig.show()


# 📌 **Interpret:** We can see that **<span style='color:lightseagreen'>OverQual</span>** is in the top of highest correlation with 0.79. It is followed by **<span style='color:lightseagreen'>GrLivArea</span>**, **<span style='color:lightseagreen'>GarageCars</span>** and **<span style='color:lightseagreen'>GarageArea</span>**. It is easily to observe that both **<span style='color:lightseagreen'>GarageCars</span>** and **<span style='color:lightseagreen'>GarageArea</span>** are quite correlated, as they have 0.88 correlation coefficient between then.

# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>3.2 | Mutual Information</b></p>
# </div>
# 
# Mutual information describes **<span style='color:lightseagreen'>relationships</span>** in terms of **<span style='color:lightseagreen'>uncertainty</span>**. The mutual information (MI) between two quantities is a measure of the extent to which knowledge of one quantity reduces uncertainty about the other. If you knew the value of a feature, how much more confident would you be about the target? Scikit-learn has two mutual information **<span style='color:lightseagreen'>metrics</span>** in its feature_selection module: one for **<span style='color:lightseagreen'>real-valued targets</span>** (mutual_info_regression) and one for **<span style='color:lightseagreen'>categorical targets</span>** (mutual_info_classif). Our target, price, is real-valued. The next cell computes the MI scores for our features and wraps them up in a nice dataframe. Hereafter, we are going to drop uninformative features as they are useless.

# In[13]:


from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    #discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def uninformative_cols(df, mi_scores):
    return df.loc[:, mi_scores == 0.0].columns

train_copy = df_train.copy()
for col in train_copy.select_dtypes("object"):
    train_copy[col], _ = train_copy[col].factorize()
    
y = train_copy['SalePrice']
x = train_copy.drop('SalePrice',axis=1)

mi_scores = make_mi_scores(x, y)
col = uninformative_cols(x, mi_scores)
x = x.drop(col,axis=1)
print('Uninformative features:\n{}'.format(mi_scores[mi_scores == 0.0]))
mi_scores = pd.DataFrame(mi_scores).reset_index().rename(columns={'index':'Feature'})


# Hereafter, we are gooin to plot results obtained previously in order to regard which features are the **<span style='color:lightseagreen'>most informative</span>**, and which ones requires some more analysis. You can see that we have a number of features that are highly informative and also some that don't seem to be informative at all (at least by themselves). Top scoring features will usually pay-off the **<span style='color:lightseagreen'>most during feature development</span>**, so it could be a good idea to focus your efforts on those. On the other hand, training on uninformative features can lead to overfitting.

# In[14]:


fig = px.bar(mi_scores, x='MI Scores', y='Feature', color="MI Scores",
             color_continuous_scale='darkmint')
fig.update_layout(height = 1500, title_text="Mutual Information Scores",
                  title_font=dict(size=29, family="Lato, sans-serif"), xaxis={'categoryorder':'category ascending'}, margin=dict(t=80))


# ### 3.2.1 | Building Type
# 
# We have seen that **<span style='color:lightseagreen'>BldgType</span>** feature didn't get a very high MI score. A plot confirms that the categories in BldgType don't do a good job of distinguishing values in SalePrice as the distributions look fairly similar.

# In[15]:


px.box(df_train, x="BldgType", y="SalePrice", color="BldgType")


# However, the type of a dwelling seems like it should be important information. Investigating whether BldgType produces a significant interaction we find that the trend lines are significantly different from one category to the next. This indicates an interaction effect, so we are keeping this variable.

# In[16]:


sns.lmplot(x='GrLivArea', y="SalePrice", hue="BldgType", col="BldgType",data=df_train, scatter_kws={"edgecolor": 'w'}, col_wrap=3, height=6, aspect = 1)


# Hereafter, we are going to define a baseline score which is going to help us to know whether some set of features we've assembled has actually led to any **<span style='color:lightseagreen'>improvement</span>** or not.

# In[17]:


def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding is good for XGBoost and RandomForest, but one-hot
    # would be better for models like Lasso or Ridge. The `cat.codes`
    # attribute holds the category levels.
    for colname in X.select_dtypes(["object"]):
        X[colname] = X[colname].factorize()
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    log_y = np.log(y)
    score = cross_val_score(
        model, X, log_y, cv=5, scoring="neg_mean_squared_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score

baseline_score = score_dataset(x, y)
print(f"Baseline score: {baseline_score:.5f} RMSLE")


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>3.3 | Label Encoding</b></p>
# </div>
# 
# Let's begin by defining a **<span style='color:lightseagreen'>transformation</span>** firstly, a label encoding for the categorical features. A **<span style='color:lightseagreen'>label encoding</span>** is okay for any kind of categorical feature when we use a **<span style='color:lightseagreen'>tree-ensemble</span>** like XGBoost, even for unordered categories. If you want to try a linear regression model (also popular in this competition), you would instead want to use a one-hot encoding, especially for the features with unordered categories.

# In[18]:


def label_encode(df):
    X = df.copy()
    for colname in X.select_dtypes(["object"]).columns:
        X[colname] = LabelEncoder().fit_transform(X[colname])
    return X


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>3.4 | Normalization</b></p>
# </div>
# 
# Our algorithm works better with features **<span style='color:lightseagreen'>distributed normally</span>**. Thus, let's check whether our variables are distributed normally or not. To do so, we are going to make some plots hereafter.

# In[19]:


quantitative = [f for f in df_train.columns if df_train.dtypes[f] == 'float']

sns.set_style('darkgrid')
f = pd.melt(df_train, value_vars=quantitative)
g = sns.FacetGrid(f, col="variable", col_wrap=3, sharex=False, sharey=False, height=3, aspect=2)
g = g.map(sns.distplot, "value", color='lightseagreen')


# 📌 **Interpret:** We can see that **<span style='color:lightseagreen'>neither</span>** of them are normally distributed. This is one of the awesome things you can learn in statistical books: in case of positive skewness, log transformations usually works well. Hereafter, we are going to make the logaritmic transformation to **<span style='color:lightseagreen'>SalePrice</span>**. 
# 
# Now we are going to solve normalization issue for the other quantitative variables. In general, they present some **<span style='color:lightseagreen'>skewness</span>**. However, before making the transformation we have to face another issue. They have a significant number of observations with **<span style='color:lightseagreen'>zero value</span>**, houses without basement or garage for example. That's a big issue as zero value **<span style='color:lightseagreen'>does not allow</span>** us to do log transformaion. To apply it here, we'll create a variable that can get the effect of having or not having basement (binary variable). Then, we'll do a log transformation to all the non-zero observations, **<span style='color:lightseagreen'>ignoring</span>** those with value zero. This way we can transform data, without losing the effect of having or not basement or garage.

# In[20]:


def normalize(df):
    df['SalePrice'] = np.log1p(df["SalePrice"])
    df.loc[df['hasbsmt']==1,'TotalBsmtSF'] = np.log1p(df['TotalBsmtSF'])
    df.loc[df['hasgarage']==1,'GarageArea'] = np.log1p(df['GarageArea'])
    df.loc[df['hasfrontage']==1,'LotFrontage'] = np.log1p(df['LotFrontage'])
    df['LivLotRatio'] = np.log1p(df['LivLotRatio'])
    df['Total_SF'] = np.log1p(df['Total_SF'])
    df['Spaciousness'] = np.log1p(df['Spaciousness'])
    return df


# In[21]:


#columns_with_many_zeros = ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','2ndFlrSF','LowQualFinSF',
 #                          'GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea']


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>3.5 | Creating New Features</b></p>
# </div>
# 
# ### 3.5.1 | Mathematical Transformation
# 
# Relationships among numerical features are often expressed through mathematical formulas, which you'll frequently come across as part of your **<span style='color:lightseagreen'>domain research</span>**. In this case, we are going to make a few of then. 
# 1. A living lot ratio, which is going to refer to the **<span style='color:lightseagreen'>percentage of living area in our lot</span>**. 
# 2. A **<span style='color:lightseagreen'>spaciousness</span>** feature, referring to average square feet area per room above the ground. 
# 3. We are going to combine OverallQual with OverallCond by (previously converting them to numerical type) taking a product.
# 4. A **<span style='color:lightseagreen'>total square feet area</span>** of the house. 
# 5. Total number of **<span style='color:lightseagreen'>bathrooms</span>** in the whole house. 
# 6. We are going to add years from building and remodelation, in order to take into account how **<span style='color:lightseagreen'>modern</span>** a house is
# 7. Total square feet area of the **<span style='color:lightseagreen'>porch</span>**.

# In[22]:


def mathematical_transforms(df):
    X = pd.DataFrame()  # dataframe to hold new features
    X["LivLotRatio"] = df.GrLivArea / df.LotArea
    X["Spaciousness"] = (df['1stFlrSF'] + df['2ndFlrSF']) / df.TotRmsAbvGrd
    X['QualCond'] = df.OverallCond * df.OverallQual
    X['Total_SF'] = df['TotalBsmtSF'] +df['1stFlrSF'] + df['2ndFlrSF']
    X['TotalBth'] = (df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))
    X['YrBltAndRemod']=df['YearBuilt']+df['YearRemodAdd']
    X['Total_porch_sf'] = (df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF'])
    return X


# ### 3.5.2 | Counts 
# 
# Features describing the **<span style='color:lightseagreen'>presence</span>** or **<span style='color:lightseagreen'>absence</span>** of something often come in sets, for example the set of risk factors for a disease. We can aggregate such features by creating a count. These features will be binary (1 for Present, 0 for Absent) or boolean (True or False). In Python, booleans can be added up just as if they were integers.

# In[23]:


def counts(df):
    X = pd.DataFrame()
    X["PorchTypes"] = df[[
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
    ]].gt(0.0).sum(axis=1)
    return X


# ### 3.5.3 | Building-Up and Breaking-Down Features
# 
# We also have complex strings that can usefully be broken into **<span style='color:lightseagreen'>simple pieces</span>**. Features like these will often have some kind of structure that you can make use of. The **<span style='color:lightseagreen'>str accessor</span>** lets you apply string methods like split directly to columns.

# In[24]:


def break_down(df):
    X = pd.DataFrame()
    X["MSClass"] = df.MSSubClass.str.split("_", n=1, expand=True)[0]
    return X


# ### 3.5.4 | Group Transforms
# 
# Hereafter we have group transforms, which **<span style='color:lightseagreen'>aggregate</span>** information across multiple rows grouped by some category. Using an aggregation function, a group transform combines two features: a categorical feature that provides the grouping and another feature whose values you wish to aggregate.

# In[25]:


def group_transforms(df):
    X = pd.DataFrame()
    X["MedNhbdArea"] = df.groupby("Neighborhood")["GrLivArea"].transform("median")
    return X


# ### 3.5.5 | Interactions
# 
# Finally, we are going to create features for interactions proved before. BldgType produces a **<span style='color:lightseagreen'>significant interaction</span>** as we found out that the trend lines are significantly different from one category to the next, referring to house prices.

# In[26]:


def interactions(df):
    X = pd.get_dummies(df.BldgType, prefix="Bldg")
    X = X.mul(df.GrLivArea, axis=0)
    return X


# ### 3.5.6 | Presence of House Features
# We are going to create some variables to know whether a house has different features, such as fireplace, pool, basement, second floor ...

# In[27]:


def features_presence(df):
    X = pd.DataFrame()
    X['haspool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    X['has2ndfloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    X['hasgarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    X['hasbsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    X['hasfireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    X['hasfrontage'] = df['LotFrontage'].apply(lambda x: 1 if x > 0 else 0)
    return X


# ### 3.5.7 | Principal Component Analysis (PCA)
# 
# Like clustering is a **<span style='color:lightseagreen'>partitioning of the dataset based on proximity</span>**, you could think of PCA as a partitioning of the variation in the data. PCA is a great tool to help you discover important relationships in the data and can also be used to create more informative features.
# 
# > (Technical note: PCA is typically applied to **<span style='color:lightseagreen'>standardized</span>** data. With standardized data "variation" means "correlation". With unstandardized data "variation" means "covariance". All data in this course will be standardized before applying PCA.)

# In[28]:


def apply_pca(X, standardize=True):
    # Standardize
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Create principal components
    pca = PCA()
    X_pca = pca.fit_transform(X)
    # Convert to dataframe
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    # Create loadings
    loadings = pd.DataFrame(
        pca.components_.T,  # transpose the matrix of loadings
        columns=component_names,  # so the columns are the principal components
        index=X.columns,  # and the rows are the original features
    )
    return pca, X_pca, loadings


def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs

def pca_inspired(df):
    X = pd.DataFrame()
    X["Feature1"] = df.GrLivArea + df.TotalBsmtSF
    X["Feature2"] = df.YearRemodAdd * df.TotalBsmtSF
    return X


def pca_components(df, features):
    X = df.loc[:, features]
    _, X_pca, _ = apply_pca(X)
    return X_pca


pca_features = [
    "GarageArea",
    "YearRemodAdd",
    "TotalBsmtSF",
    "GrLivArea",
]


# ### 3.5.8 | Target Encoding
# There is, however, a way you can use target encoding without having to use held-out encoding data. It's basically the same trick used in cross-validation:
# 
# 1. Split the data into folds, each fold having two splits of the dataset.
# 2. Train the encoder on one split but transform the values of the other.
# 3. Repeat for all the splits.
# 
# This way, training and transformation always take place on independent sets of data, just like when you use a holdout set but without any data going to waste. In the next hidden cell is a wrapper you can use with any target encoder:

# In[29]:


class CrossFoldEncoder:
    def __init__(self, encoder, **kwargs):
        self.encoder_ = encoder
        self.kwargs_ = kwargs  # keyword arguments for the encoder
        self.cv_ = KFold(n_splits=5)

    # Fit an encoder on one split and transform the feature on the
    # other. Iterating over the splits in all folds gives a complete
    # transformation. We also now have one trained encoder on each
    # fold.
    def fit_transform(self, X, y, cols):
        self.fitted_encoders_ = []
        self.cols_ = cols
        X_encoded = []
        for idx_encode, idx_train in self.cv_.split(X):
            fitted_encoder = self.encoder_(cols=cols, **self.kwargs_)
            fitted_encoder.fit(
                X.iloc[idx_encode, :], y.iloc[idx_encode],
            )
            X_encoded.append(fitted_encoder.transform(X.iloc[idx_train, :])[cols])
            self.fitted_encoders_.append(fitted_encoder)
        X_encoded = pd.concat(X_encoded)
        X_encoded.columns = [name + "_encoded" for name in X_encoded.columns]
        return X_encoded

    # To transform the test data, average the encodings learned from
    # each fold.
    def transform(self, X):
        from functools import reduce

        X_encoded_list = []
        for fitted_encoder in self.fitted_encoders_:
            X_encoded = fitted_encoder.transform(X)
            X_encoded_list.append(X_encoded[self.cols_])
        X_encoded = reduce(
            lambda x, y: x.add(y, fill_value=0), X_encoded_list
        ) / len(X_encoded_list)
        X_encoded.columns = [name + "_encoded" for name in X_encoded.columns]
        return X_encoded


# ### 3.5.9 | Final Feature Set
# 
# Now let's combine everything together. Putting the transformations into separate functions makes it **<span style='color:lightseagreen'>easier</span>** to experiment with various combinations. We'll score our dataset in order to see how it has improve with all new features creations.

# In[30]:


def create_features(df, df_test=None):
    X = df.copy()
    y = X.pop("SalePrice")
    mi_scores = make_mi_scores(X, y)

    if df_test is not None:
        X_test = df_test.copy()
        X_test.pop("SalePrice")
        X = pd.concat([X, X_test])

    # Feature Engineering Course Lesson 2 - Mutual Information
    col = uninformative_cols(X, mi_scores)

    # Feature Engineering Course Lesson 3 - Transformations
    X = X.join(mathematical_transforms(X))
    X = X.join(interactions(X))
    X = X.join(counts(X))
    #X = X.join(break_down(X))
    X = X.join(group_transforms(X))
    X = X.join(features_presence(X))
    # Feature Engineering Course Lesson 5 - PCA
    X = X.join(pca_inspired(X))
    # X = X.join(pca_components(X, pca_features))
    # X = X.join(indicate_outliers(X))

    X = label_encode(X)
    
    # Reform splits
    if df_test is not None:
        X_test = X.loc[df_test.index, :]
        X.drop(df_test.index, inplace=True)

    # Feature Engineering Course Lesson 6 - Target Encoder
    encoder = CrossFoldEncoder(MEstimateEncoder, m=1)
    X = X.join(encoder.fit_transform(X, y, cols=["MSSubClass"]))
    if df_test is not None:
        X_test = X_test.join(encoder.transform(X_test))
    
    X = X.drop(col,axis=1)
    X_test = X_test.drop(col,axis=1)
    
    if df_test is not None:
        return X, X_test
    else:
        return X

df_train = df_data[df_data['SalePrice'].isnull() == False]
df_test = df_data[df_data['SalePrice'].isnull() == True]
X_train, X_test = create_features(df_train, df_test)
y_train = df_train.loc[:, "SalePrice"]

df_train = pd.concat([X_train,y_train],axis=1)
df_test = pd.concat([X_test, df_test['SalePrice']],axis=1)

# Logarithmic transformation
df_train = normalize(df_train)
df_test = normalize(df_test)

df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)

X_train = df_train.drop('SalePrice',axis=1)
X_test = df_test.drop('SalePrice',axis=1)
y_train = df_train['SalePrice']

score_dataset(X_train, y_train)


# Let us now see how the logarithmic conversion has had its effect on our variables, which are now **<span style='color:lightseagreen'>normally distributed</span>**, unlike before. We'll start with **<span style='color:lightseagreen'>Normal probability plot</span>**, where data distribution should closely follow the diagonal that represents the normal distribution.

# In[31]:


from scipy import stats
res = stats.probplot(df_train['SalePrice'],  plot = plt)


# In[32]:


#histogram and normal probability plot
sns.set_style('darkgrid')
f = pd.melt(df_train, value_vars=['GarageArea','TotalBsmtSF','LotFrontage','Total_SF','Spaciousness','LivLotRatio'])
g = sns.FacetGrid(f, col="variable", col_wrap=3, sharex=False, sharey=False, aspect = 2)
g = g.map(sns.distplot, "value", color='lightseagreen')


# In[33]:


y = df_train['SalePrice']
x = df_train.drop('SalePrice',axis=1)

mi_scores = make_mi_scores(x, y)
print('Uninformative features:\n{}'.format(mi_scores[mi_scores == 0.0]))
mi_scores = pd.DataFrame(mi_scores).reset_index().rename(columns={'index':'Feature'})

X_train = df_train.drop('SalePrice',axis=1)
X_test = df_test.drop('SalePrice',axis=1)
y_train = df_train['SalePrice']


# In[34]:


fig = px.bar(mi_scores, x='MI Scores', y='Feature', color="MI Scores",
             color_continuous_scale='darkmint')
fig.update_layout(height = 1500, title_text="Mutual Information Scores After Feature Engineering",
                  title_font=dict(size=29, family="Lato, sans-serif"), xaxis={'categoryorder':'category ascending'}, margin=dict(t=80))


# # <b>4 <span style='color:lightseagreen'>|</span> Modeling</b>
# 
# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>4.1 | Comparison with PyCaret</b></p>
# </div>
# 
# [PyCaret Tutorial](https://todobi.com/pycaret-paso-a-paso/)
# 
# PyCaret is an open source, low-code machine learning library in Python that allows you to go from preparing your data to deploying your model within seconds in your choice of notebook environment. PyCaret being a low-code library makes you more productive. With less time spent coding, you and your team can now focus on business problems. PyCaret is simple and easy to use machine learning library that will help you to perform end-to-end ML experiments with less lines of code. PyCaret is a business ready solution. It allows you to do prototyping quickly and efficiently from your choice of notebook environment. You can reach pycaret website and documentation from [PyCaret](https://pycaret.org)
# 
# ![](https://pycaret.org/wp-content/uploads/2020/03/Divi93_43.png)

# In[35]:


reg = setup(data = df_train, target = 'SalePrice', silent = True, use_gpu = True)
clear_output()
compare_models()


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>4.2 | Gradient Boosting</b></p>
# </div>
# 
# ### 4.2.1 | Hyperparameter Tuning - Optuna
# 
# In this case, only for Catboost, we are going to make the **<span style='color:lightseagreen'>tuning with Optuna</span>**. I will add the code for hyperparameter tuning below. However, for not **<span style='color:lightseagreen'>wasting CPU time</span>**, since I have run it once, I will simply create the model with the specific features values. I will control whether making hyperparameter tuning or not with **<span style='color:lightseagreen'>allow_optimize</span>** Finally, just say that code for tuning takes plenty of time. Due to that I enabled GPU technology. 

# In[36]:


def objective(trial):
    params = {
        "random_state":trial.suggest_categorical("random_state", [2022]),           # categorical for concrete values
        'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 1),   # loguniform for continuos values
        "n_estimators": trial.suggest_int('n_estimators',50,2000),                 # int for discrete values. Interval between [100,2000]
        "max_depth" : trial.suggest_int("max_depth", 1, 20),
        "min_samples_split" : trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf" : trial.suggest_int("min_samples_leaf", 2, 20),
        "alpha" : trial.suggest_loguniform('alpha',0.9,1),
        "max_features" : trial.suggest_int("max_features", 10, 50)
    }

    model = GradientBoostingRegressor(**params)
    X_train_tmp, X_valid_tmp, y_train_tmp, y_valid_tmp = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    model.fit(X_train_tmp, y_train_tmp)
        
    y_train_pred = model.predict(X_train_tmp)
    y_valid_pred = model.predict(X_valid_tmp)
    train_mae = mae(y_train_tmp, y_train_pred)
    valid_mae = mae(y_valid_tmp, y_valid_pred)
    
    print(f'MAE of Train: {train_mae}')
    print(f'MAE of Validation: {valid_mae}')
    
    return valid_mae

allow_optimize = 0


# In[37]:


TRIALS = 100
TIMEOUT = 3600

if allow_optimize:
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name = 'gbr_parameter_opt',
        direction = 'minimize',
        sampler = sampler,
    )
    study.optimize(objective, n_trials=TRIALS)
    print("Best Score:",study.best_value)
    print("Best trial",study.best_trial.params)
    
    best_params = study.best_params
    
    X_train_tmp, X_valid_tmp, y_train_tmp, y_valid_tmp = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    model_tmp = GradientBoostingRegressor(**best_params).fit(X_train_tmp, y_train_tmp)


# ### 4.2.2 | Fitting - Feature Importances

# In[38]:


if allow_optimize:
    model = GradientBoostingRegressor(**best_params, n_estimators=model_tmp.get_best_iteration(), verbose=1000).fit(X_train, y_train)
else:
    model = GradientBoostingRegressor(
        verbose=0,
        random_state = 2022, learning_rate = 0.023841354286362568, n_estimators = 1577, max_depth = 3, 
        min_samples_split = 16, min_samples_leaf = 2, alpha = 0.9581363403215747, max_features = 20).fit(X_train, y_train)  
    
permutation_gbr = PermutationImportance(model, random_state=1).fit(X_train, y_train)


# In[39]:


def plot_feature_importance(importance,names,model_type):
    
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    fi_df = fi_df[fi_df.feature_importance > 0]
    fig = px.bar(fi_df, x='feature_names', y='feature_importance', color="feature_importance",
             color_continuous_scale='Blugrn')
    # General Styling
    fig.update_layout(height=750, bargap=0.2,
                  margin=dict(b=50,r=30,l=100,t=100),
                  title = "<span style='font-size:36px; font-family:Times New Roman'>Feature Importance Analysis</span>",                  
                  plot_bgcolor='rgb(242,242,242)',
                  paper_bgcolor = 'rgb(242,242,242)',
                  font=dict(family="Times New Roman", size= 14),
                  hoverlabel=dict(font_color="floralwhite"),
                  showlegend=False)
    fig.show()
    
plot_feature_importance(model.feature_importances_,X_train.columns,'Gradient Boosting')


# ### 4.2.3 | Making Predictions

# In[40]:


predictions_gbr = model.predict(X_test)
predictions_gbr = np.expm1(predictions_gbr)


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>4.3 | Gradient Boosting</b></p>
# </div>
# 
# ### 4.3.1 | Hyperparameter Tuning - Optuna
# 
# In this case, only for Catboost, we are going to make the **<span style='color:lightseagreen'>tuning with Optuna</span>**. I will add the code for hyperparameter tuning below. However, for not **<span style='color:lightseagreen'>wasting CPU time</span>**, since I have run it once, I will simply create the model with the specific features values. I will control whether making hyperparameter tuning or not with **<span style='color:lightseagreen'>allow_optimize</span>** Finally, just say that code for tuning takes plenty of time. Due to that I enabled GPU technology. 

# In[41]:


def objective(trial):
    params = {
        "random_state":trial.suggest_categorical("random_state", [2022]),           
        'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 1),   
        "n_estimators": trial.suggest_int('n_estimators',50,2000),                 
        "max_depth" : trial.suggest_int("max_depth", 1, 20),
        "_min_child_weight" : trial.suggest_float("_min_child_weight", 0.1, 10),
        "reg_lambda" : trial.suggest_float("reg_lambda", 0.01, 10),
        "reg_alpha" : trial.suggest_float('reg_alpha',0.01,10),
        "num_leaves" : trial.suggest_int("num_leaves", 50, 100),
        'subsample' : trial.suggest_float('subsample', 0.01, 1)
    }

    #model = LGBMRegressor(**params, device='GPU')
    model = LGBMRegressor(**params)
    X_train_tmp, X_valid_tmp, y_train_tmp, y_valid_tmp = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    model.fit(X_train_tmp, y_train_tmp)
        
    y_train_pred = model.predict(X_train_tmp)
    y_valid_pred = model.predict(X_valid_tmp)
    train_mae = mae(y_train_tmp, y_train_pred)
    valid_mae = mae(y_valid_tmp, y_valid_pred)
    
    print(f'MAE of Train: {train_mae}')
    print(f'MAE of Validation: {valid_mae}')
    
    return valid_mae

allow_optimize = 0


# In[42]:


TRIALS = 100
TIMEOUT = 3600

if allow_optimize:
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name = 'lgbm_parameter_opt',
        direction = 'minimize',
        sampler = sampler,
    )
    study.optimize(objective, n_trials=TRIALS)
    print("Best Score:",study.best_value)
    print("Best trial",study.best_trial.params)
    
    best_params = study.best_params
    
    X_train_tmp, X_valid_tmp, y_train_tmp, y_valid_tmp = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    #model_tmp = LGBMRegressor(**best_params, device='GPU').fit(X_train_tmp, y_train_tmp)
    model_tmp = LGBMRegressor(**best_params).fit(X_train_tmp, y_train_tmp)    


# ### 4.3.2 | Fitting - Feature Importances

# In[43]:


if allow_optimize:
    model = LGBMRegressor(**best_params, device='GPU', n_estimators=model_tmp.get_best_iteration(), verbose=1000).fit(X_train, y_train)
else:
    model = LGBMRegressor(        
        random_state = 2022, learning_rate = 0.06708645071974023, n_estimator = 1014, max_depth = 2, 
        min_child_weight = 7.3813273033314495, reg_lambda = 2.4677587687488165, reg_alpha = 0.44277253423371826, 
        num_leaves = 98, subsample = 0.8813132421543259).fit(X_train, y_train)  
    
permutation_lgbm = PermutationImportance(model, random_state=1).fit(X_train, y_train)


# In[44]:


importances = pd.DataFrame({'importance':model.feature_importances_, 'feature': X_train.columns})
importances = importances[importances.importance > 0]


# In[45]:


plot_feature_importance(model.feature_importances_,X_train.columns,'LGBM')


# ### 4.3.3 | Predictions - Submitting

# In[46]:


predictions_lgbm = model.predict(X_test)
predictions_lgbm = np.expm1(predictions_lgbm)
submit = pd.DataFrame({'Id': df_test.index, 'SalePrice':predictions_gbr*0.95 + predictions_lgbm * 0.05}).set_index('Id')
submit.to_csv('./submission.csv')


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>4.4 | Permutation Importance</b></p>
# </div>
# 
# One of the most basic questions we might ask of a model is: **<span style='color:lightseagreen'>What features have the biggest impact on predictions?</span>** This concept is called feature importance. There are multiple ways to measure feature importance. Some approaches answer subtly different versions of the question above. Other approaches have documented shortcomings. In this section, we'll focus on permutation importance. Compared to most other approaches, permutation importance is:
# 
# - Fast to calculate,
# - Widely used and understood, and
# - Consistent with properties we would want a feature importance measure to have.

# In[47]:


eli5.show_weights(permutation_gbr, feature_names = X_test.columns.tolist(), top=20)


# In[48]:


eli5.show_weights(permutation_lgbm, feature_names = X_test.columns.tolist(), top=20)


# In[ ]:




