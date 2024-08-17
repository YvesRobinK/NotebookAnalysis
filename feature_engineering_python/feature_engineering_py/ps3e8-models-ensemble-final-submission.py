#!/usr/bin/env python
# coding: utf-8

# # <b> <span style='color:#F1A424'>|</span> TABLE OF CONTENTS</b>
# 
# <a id="toc"></a>
# 
# - [1. Import Libraries](#1)
# - [2. Load Dataset](#2)
# - [3. Read the Problem Statement](#3)
# - [4. Overview of MetaData](#4)
# - [5. Explore Missing Data](#5)
# - [6. Exploratory Data Analysis](#6)
# - [7. Handling Missing Data](#7)
# - [8. Feature Engineering](#8)
# - [9. Dealing with Categorical Columns](#9)
# - [10. Feature Correlations](#10)
# - [11. Handling Highly correlated Features](#11)
# - [12. Model Building and training](#12)
# - [13. Making Predictions](#13)
# - [14. Submission](#14)
# 

# <a id="1"></a>
# # <b> <span style='color:#F1A424'>|</span> Step 1 - Import Libraries</b>

# In[1]:


# pip list
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,KFold

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import optuna
from warnings import filterwarnings
filterwarnings('ignore')

from sklearn.metrics import mean_squared_error


TARGET = 'price'


# <a id="1"></a>
# # <b> <span style='color:#F1A424'>|</span> Step 2 - Load Dataset</b>

# In[2]:


train = pd.read_csv('/kaggle/input/playground-series-s3e8/train.csv',index_col = 'id')
test = pd.read_csv('/kaggle/input/playground-series-s3e8/test.csv',index_col = 'id')
submission= pd.read_csv('/kaggle/input/playground-series-s3e8/sample_submission.csv')
og = pd.read_csv('/kaggle/input/gemstone-price-prediction/cubic_zirconia.csv',index_col=0)
metadata = pd.read_excel('/kaggle/input/gemstone-price-prediction/Data Dictionary.xlsx')

display(train.head())
display(test.head())
display(og.head())
display(submission.head())


# <a id="1"></a>
# # <b> <span style='color:#F1A424'>|</span> Step 3 - Read the Problem Statement</b>

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#5642C5;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#               color:white;">
# ❓ Problem Statement
#     
#               You are hired by a company Gem Stones co ltd, which is a cubic zirconia manufacturer. You are provided with the dataset containing the prices and other attributes of almost 27,000 cubic zirconia (which is an inexpensive diamond alternative with many of the same qualities as a diamond). The company is earning different profits on different prize slots. You have to help the company in predicting the price for the stone on the basis of the details given in the dataset so it can distinguish between higher profitable stones and lower profitable stones so as to have a better profit share. Also, provide them with the best 5 attributes that are most important.
# </p>
# </div>
# 
# 
# 

# # <b> <span style='color:#F1A424'>|</span> Step 4 - Overview of MetaData</b>

# <img src="data:image/png;base64,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" alt="Dimond Terminology" />
# 
# 

# In[3]:


# pd.set_option('display.max_colwidth', None)
# metadata


# # <b> <span style='color:#F1A424'>|</span> Step 5 - Explore Missing Data</b>

# In[4]:


# print(train.isnull().sum().sum(),test.isnull().sum().sum(),og.isnull().sum().sum())


# In[5]:


# train.describe()


# # <b> <span style='color:#F1A424'>|</span> Step 6 - Exploratory Data Analysis</b>

# In[6]:


# plt.figure(figsize=(30,5))

# plt.subplot(1,3,1)
# sns.countplot(data=train, x='cut',order = train['cut'].value_counts().index)
# plt.xticks(rotation=90)
# plt.title('cut',fontsize = 20)

# plt.subplot(1,3,2)
# sns.countplot(data=train, x='color',order = train['color'].value_counts().index)
# plt.xticks(rotation=90)
# plt.title('color',fontsize = 20)

# plt.subplot(1,3,3)
# sns.countplot(data=train, x='clarity',order = train['clarity'].value_counts().index)
# plt.xticks(rotation=90)
# plt.title('clarity',fontsize = 20)


# In[7]:


# def violin(df):
#     features = ['carat','depth','table','x','y','z']
#     plt.figure(figsize=(30,15))
#     rows,columns = 5,5
#     for i in range(len(features)):
#         plt.subplot(rows,columns,i+1)
#         sns.violinplot(data=df, x=features[i])
#         plt.xticks(rotation=90)
#         plt.title(f'{features[i]}',fontsize = 20)
#     plt.show()
    
# violin(train)


# In[8]:


# new_train = train.copy()
# new_test = test.copy()
# new_og = og.copy()
# new_train['set'] = 'train'
# new_og['set'] = 'og'
# new_test['set'] = 'test'

# df_combined = pd.concat([new_train, new_test, new_og])
# NUM_FEATURES = [c for c in train.columns if c not in [TARGET, 'set']]
# ncols = 2
# nrows = np.ceil(len(NUM_FEATURES)/ncols).astype(int)
# fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(15,nrows*4))
# for c, ax in zip(NUM_FEATURES, axs.flatten()):
#     if df_combined[c].nunique() > 20:
#         sns.boxplot(data=df_combined, x=c, ax=ax, y='set')
#     else:
#         sns.countplot(data=df_combined, x='set', ax=ax, hue=c)
#         ax.legend(bbox_to_anchor=(1,1), loc='upper left', title=c)
#         ax.set_xlabel('')

# fig.suptitle('Distribution of features by set')
# plt.tight_layout(rect=[0, 0, 1, 0.98])
# for df in [train, test, og]:
#     df.drop(columns=['set'], errors='ignore')


# <div class="alert alert-block alert-info"> 
# 📌 Insight: 
#     
# 1. From histograms it is sure that all the graphs follow same distribution irrespective of the count
# 2.  sum columns like carat, z has skewed data
# </div>

# [reference](https://www.kaggle.com/code/samuelcortinhas/ps-s3e8-gem-price-prediction): 

# ## Train - Test Overlap

# In[9]:


# cont_col = [i for (i,j) in zip(test.columns,test.dtypes) if j in ['int','float']]
# fig,axes = plt.subplots(3,2,figsize=(30,20))
# for i,ax in enumerate(axes.flat):
#     sns.kdeplot(ax=ax,data=train,x=cont_col[i],color='#F8766D',label='Train',fill=True)
#     sns.kdeplot(ax=ax,data=test,x=cont_col[i],color='#00BFC4',label='Test',fill=True)
#     ax.set_title(f'{cont_col[i]} distribution')
# fig.tight_layout()
# plt.legend()


# ## Train - Original Data Overlap

# In[10]:


# fig,axes = plt.subplots(3,2,figsize=(30,20))
# for i,ax in enumerate(axes.flat):
#     sns.kdeplot(ax=ax,data=train,x=cont_col[i],color='#F8766D',label='Train',fill=True)
#     sns.kdeplot(ax=ax,data=og,x=cont_col[i],color='#00BFC4',label='Original',fill=True)
#     ax.set_title(f'{cont_col[i]} distribution')
# fig.tight_layout()
# plt.legend()


# <div class="alert alert-block alert-info"> 📌 Insight: Training and Testing set follow a close distibution unlike Train and original dataset</div>
# 

# In[11]:


df = pd.concat([train,og],axis=0).drop_duplicates()
# df.shape
# df = train.copy()


# # <b> <span style='color:#F1A424'>|</span> Step 7 - Handling Missing Data</b>

# In[12]:


# df.loc[df['x']<2 , 'x'] = np.nan
# df.loc[df['y']<2 , 'y'] = np.nan
# df.loc[df['z']<2 , 'z'] = np.nan

# df['depth'].fillna(df['depth'].median(),inplace=True)
# df['x'].fillna(df['x'].median(),inplace=True)
# df['y'].fillna(df['y'].median(),inplace=True)
# df['z'].fillna(df['z'].median(),inplace=True)


# # <b> <span style='color:#F1A424'>|</span> Step 8 - Feature Engineering</b>
# [reference](https://www.kaggle.com/code/phongnguyen1/automl-with-autokeras-flaml-autogluon)

# 

# In[13]:


# def xy_order_transform(df):
#     df_copy = df.copy()
#     xy_change_index = df_copy.x < df_copy.y
#     temp = df_copy.loc[xy_change_index, 'x']
#     df_copy.loc[xy_change_index, 'x'] = df_copy.loc[xy_change_index, 'y']
#     df_copy.loc[xy_change_index, 'y'] = temp
    
#     return df_copy

# df = xy_order_transform(df)
# test = xy_order_transform(test)


# In[14]:


# Feature engineering
def add_features(df):
#     df['depth_calculated'] = 100 * df.z / ((df.x + df.y) / 2)
#     df['depth_diff'] = df.depth - df.depth_calculated
#     df['xy_ratio'] = df.x/df.y
#     df['volume'] = df['x'] * df['y'] * df['z']
#     df['density'] = df['carat'] / df['volume']
#     df['table_ratio'] = (df['table'] / ((df['x'] + df['y']) / 2))
#     df['depth_ratio'] = (df['depth'] / ((df['x'] + df['y']) / 2))
#     df['symmetry'] = (abs(df['x'] - df['z']) + abs(df['y'] - df['z'])) / (df['x'] + df['y'] + df['z'])
#     df['surface_area'] = 2 * ((df['x'] * df['y']) + (df['x'] * df['z']) + (df['y'] * df['z']))
#     df['depth_to_table_ratio'] = df['depth'] / df['table']
    return df

# adding this features did not improve model performance


# In[15]:


train_data = add_features(df)
test_data = add_features(test)


# In[16]:


x = train_data.drop('price',axis=1)
y = train_data['price']


# In[17]:


# x.head()


# In[18]:


# y.head()


# # <b> <span style='color:#F1A424'>|</span> Step 9 - Dealing with Categorical Columns</b>

# In[19]:


def preprocess(df):   
    
    replacement = {'cut':{'Fair':0, 'Good':1, 'Very Good':2, 'Premium':3, "Ideal":4},
                  'color':{'J':0,'I':1,'H':2, 'G':3, 'F':4, 'E':5, "D":6},
                  'clarity':{'FL':10, 'IF':9, 'VVS1':8, 'VVS2':7, 'VS1':6, 'VS2':5, 'SI1':4, 'SI2':3, 'I1':2, 'I2':1, 'I3':0}}
    df = df.replace(replacement) 
    return df    


# In[20]:


x = preprocess(x)
test_data = preprocess(test_data)


# In[21]:


# print(x.shape)
# x.head()


# In[22]:


# print(test_data.shape)
# test_data.head()


# In[23]:


# new_train = x.copy()
# new_test = test_data.copy()
# # new_og = og.copy()
# new_train['set'] = 'train'
# # new_og['set'] = 'og'
# new_test['set'] = 'test'

# df_combined = pd.concat([new_train, new_test])
# NUM_FEATURES = [c for c in x.columns if c not in [TARGET, 'set']]
# ncols = 2
# nrows = np.ceil(len(NUM_FEATURES)/ncols).astype(int)
# fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(15,nrows*4))
# for c, ax in zip(NUM_FEATURES, axs.flatten()):
#     if df_combined[c].nunique() > 20:
#         sns.boxplot(data=df_combined, x=c, ax=ax, y='set')
#     else:
#         sns.countplot(data=df_combined, x='set', ax=ax, hue=c)
#         ax.legend(bbox_to_anchor=(1,1), loc='upper left', title=c)
#         ax.set_xlabel('')

# fig.suptitle('Distribution of features by set')
# plt.tight_layout(rect=[0, 0, 1, 0.98])
# for df in [train, test]:
#     df.drop(columns=['set'], errors='ignore')


# In[24]:


# print(x.shape,y.shape,test_data.shape)


# # <b> <span style='color:#F1A424'>|</span> Step 10 - Feature Correlations </b>

# In[25]:


correlation = pd.concat([x,y],axis=1)
corr = correlation.corr()


# In[26]:


def highlighter(cell_value):
    if (cell_value >= 0.5) & (cell_value != 1):
        return "background-color: green"

corr.style.applymap(highlighter)


# <div class="alert alert-block alert-info"> 📌 Insight:
#     
#     Price has a strong relation with carat ,x,y, z, volume and surface Area
# </div>
# 

# # <b> <span style='color:#F1A424'>|</span> Step 11 - Handling Highly correlated Features</b>

# In[27]:


def correlation(dataset, threshold):
    col_corr = set()  
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (abs(corr_matrix.iloc[i, j]) > threshold):
                print(f'i : {corr_matrix.columns[i]},             j : {corr_matrix.columns[j]}       {corr_matrix.iloc[i, j]}')
                colname = corr_matrix.columns[i]
                print(f'colname = {colname}')
#             print(f'Column Name: {colname}')
    col_corr.add(colname)
    print()
    print(f'Remaining Column: {col_corr}')
    return col_corr 
corr_features = correlation(x, 0.85)


# In[28]:


# x.drop(corr_features,axis=1,inplace=True)
# # x_test.drop(corr_features,axis=1,inplace=True)
# test_data.drop(corr_features,axis=1,inplace=True)


# In[29]:


# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[30]:


# print(x.shape,y.shape,test_data.shape)


# In[31]:


# FEATURES = x.columns


# In[32]:


# display(x_train.head())
# display(x.head())
# display(test_data.head())


# # <b> <span style='color:#F1A424'>|</span> Step 12 - Model Building and training</b>

# In[33]:


# FOLDS = 15

# train_list,test_list = list(),list()

# predictions, RMSE = list(),list()

# kf = KFold(n_splits=FOLDS,random_state=42,shuffle=True)

# for train_index,test_index in kf.split(x,y):
#     x_train,x_test = x.iloc[train_index],x.iloc[test_index]
#     y_train,y_test = y.iloc[train_index],y.iloc[test_index]
    
#     train_list.append((x_train,y_train))
#     test_list.append((x_test,y_test))
    
    
# for i in range(FOLDS):
    
#     cat = CatBoostRegressor(n_estimators=15000,random_state=42,learning_rate=0.15)
# #     model =  XGBRegressor(n_estimators=8000,random_state=42,learning_rate=0.15)
    
#     x_train = train_list[i][0]
#     y_train = train_list[i][1]
    
#     x_test = test_list[i][0]
#     y_test = test_list[i][1]
    
#     cat.fit(x_train,y_train,early_stopping_rounds=100,eval_set=[(x_test,y_test)])
           
#     pred = cat.predict(x_test)
    
#     mse = mean_squared_error(pred,y_test)
#     rmse = np.sqrt(mse)
    
#     RMSE.append(rmse)
    
#     print(f'Fold {i+1}')
#     print('RMSE {:.1F}'.format(rmse))
    
#     final_pred = cat.predict(test_data)
#     predictions.append(final_pred)
    
# print(f'AVERAGE RMSE : {np.mean(RMSE)}')
    


# # <b> <span style='color:#F1A424'>|</span> Step 13 - Making Predictions</b>

# In[34]:


# cat_predictions = np.mean(predictions,axis=0)


# In[35]:


# FOLDS = 15

# train_list,test_list = list(),list()

# predictions, RMSE = list(),list()

# kf = KFold(n_splits=FOLDS,random_state=42,shuffle=True)

# for train_index,test_index in kf.split(x,y):
#     x_train,x_test = x.iloc[train_index],x.iloc[test_index]
#     y_train,y_test = y.iloc[train_index],y.iloc[test_index]
    
#     train_list.append((x_train,y_train))
#     test_list.append((x_test,y_test))
    
    
# for i in range(FOLDS):
    
#     xgb = XGBRegressor(max_depth=10,learning_rate=0.24,n_estimators=50,objective='reg:linear',booster='gbtree')
# #     model =  XGBRegressor(n_estimators=8000,random_state=42,learning_rate=0.15)
    
#     x_train = train_list[i][0]
#     y_train = train_list[i][1]
    
#     x_test = test_list[i][0]
#     y_test = test_list[i][1]
    
#     xgb.fit(x_train,y_train,early_stopping_rounds=100,eval_set=[(x_test,y_test)])
           
#     pred = xgb.predict(x_test)
    
#     mse = mean_squared_error(pred,y_test)
#     rmse = np.sqrt(mse)
    
#     RMSE.append(rmse)
    
#     print(f'Fold {i+1}')
#     print('RMSE {:.1F}'.format(rmse))
    
#     final_pred = xgb.predict(test_data)
#     predictions.append(final_pred)
    
# print(f'AVERAGE RMSE : {np.mean(RMSE)}')
    


# In[36]:


# xgb_predictions = np.mean(predictions,axis=0)


# In[37]:


# xgb = XGBRegressor(max_depth=10,learning_rate=0.24,n_estimators=30,objective='reg:linear',booster='gbtree')
# xgb.fit(x_train,y_train)
# sample_predictions = xgb.predict(x_test)
# print(np.sqrt(mean_squared_error(y_test,sample_predictions)))


# In[38]:


# xgb_predictions = xgb.predict(test_data)


# In[39]:


# FOLDS = 15

# train_list,test_list = list(),list()

# predictions, RMSE = list(),list()

# kf = KFold(n_splits=FOLDS,random_state=42,shuffle=True)

# for train_index,test_index in kf.split(x,y):
#     x_train,x_test = x.iloc[train_index],x.iloc[test_index]
#     y_train,y_test = y.iloc[train_index],y.iloc[test_index]
    
#     train_list.append((x_train,y_train))
#     test_list.append((x_test,y_test))
    
    
# for i in range(FOLDS):
    
#     lgbm = LGBMRegressor(n_estimators=15000,random_state=42,learning_rate=0.15)
# #     model =  XGBRegressor(n_estimators=8000,random_state=42,learning_rate=0.15)
    
#     x_train = train_list[i][0]
#     y_train = train_list[i][1]
    
#     x_test = test_list[i][0]
#     y_test = test_list[i][1]
    
#     lgbm.fit(x_train,y_train,early_stopping_rounds=100,eval_set=[(x_test,y_test)])
           
#     pred = lgbm.predict(x_test)
    
#     mse = mean_squared_error(pred,y_test)
#     rmse = np.sqrt(mse)
    
#     RMSE.append(rmse)
    
#     print(f'Fold {i+1}')
#     print('RMSE {:.1F}'.format(rmse))
    
#     final_pred = lgbm.predict(test_data)
#     predictions.append(final_pred)
    
# print(f'AVERAGE RMSE : {np.mean(RMSE)}')
    


# In[40]:


# lgbm_predictions = np.mean(predictions,axis=0)


# In[41]:


# predicted = (0.5*cat_predictions+0.2*xgb_predictions+0.3*lgbm_predictions) ***************
# # predicted = (0.5*cat_predictions+0.5*lgbm_predictions)

# # predicted = (0.7*cat_predictions+0.3*xgb_predictions)
# # predicted = (cat_predictions)


# In[42]:


# # predictions = xgb.predict(test_data)
# model_submission = submission.copy()
# model_submission['price'] = predicted
# model_submission.head()


# # <b> <span style='color:#F1A424'>|</span> Step 14 - Submission</b>

# In[43]:


# model_submission.to_csv('output',index=False)


# In[ ]:




