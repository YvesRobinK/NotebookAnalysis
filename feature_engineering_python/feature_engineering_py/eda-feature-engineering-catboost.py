#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


#installing intel optimizations for scikit-learn
get_ipython().system('pip install -U scikit-learn scikit-learn-intelex >> z_pip.log')
get_ipython().system('pip install delayed')


# In[3]:


from sklearnex import patch_sklearn
patch_sklearn()


# In[4]:


# importing relevant modules and classes
from sklearn.preprocessing import StandardScaler,LabelEncoder, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# In[5]:


# Loading Raw data
#pseudo_df = pd.read_csv('../input/tps12-pseudolabels/tps12-pseudolabels_v2.csv')
#train_df_inter = pd.read_csv('../input/tabular-playground-series-dec-2021/train.csv')
train_df = pd.read_csv('../input/tabular-playground-series-dec-2021/train.csv')
test_df = pd.read_csv('../input/tabular-playground-series-dec-2021/test.csv')


# In[6]:


#train_df = pd.concat([train_df_inter,pseudo_df],ignore_index=True)
if 'pseudo_df' in globals():
    del pseudo_df
if 'train_df_inter' in globals():
    del train_df_inter


# #### Printing first 10 rows 
# Data definitions here https://www.kaggle.com/c/forest-cover-type-prediction/data

# In[7]:


train_df.head(10)


# ##### Summary of columns in the data

# In[8]:


train_df.describe().T


# ##### We can draw the following observations :-
# 1. Soil_TypeXX and Wilderness_AreaX are categorical faetures while others are numerical features.  
# 2. Soil_Type7 and Soil_Type15 are all 0. Hence they must be dropped while creating the model.  
# 3. The categorical variables are sparse.
# 4. Some values exist outside of their supposed range.  
#     1. The distance values must not be -ve, but they are.  
#     2. Slope and Ascent are angles and must be  betwen 0 and 360 but we see outliers.  
#     3. Hill shade indices must be between 0 and 255 but we observe values lying outside. 
# 
# The 'anaomolies' could have been introduced while creation of systhetic data. The values can be rectified with some feature engineering.

# In[9]:


## Creating Relevant features list
soil_features = [x for x in train_df.columns if x.__contains__('Soil')]
wild_features = [x for x in train_df.columns if x.__contains__('Wild')]
distance_features = [x for x in train_df.columns if x.__contains__('Horizontal') or x.__contains__('Vertical')]
hillshade_features = [x for x in train_df.columns if x.__contains__('Hill')]
angle_features = ['Aspect', 'Slope']
numerical_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']


# #### Exploratory Data Analysis

# ##### 1. Distribution of target variable

# In[10]:


train_df['Cover_Type'].value_counts()


# ###### Cover Type 5 has only one record, hence we would be dropping the row.

# #### Visual Examination of relation between variables  
# Let us explore how the numerical values are distrbuted wrt the forest cover types. Since cover type 5 has only 1 observation, we would be dropping this for visual inspection.

# In[11]:


def plot_distribution_by_category(train_df):
    fig, axis = plt.subplots(nrows=5,ncols=2,figsize=(25,25))
    axis = axis.flatten()
    ax_i = 0
    
    for i,X in enumerate(numerical_features):
        plt.xlim(min(train_df[X]),max(train_df[X]))
        sns.kdeplot(x=X,
                    hue='Cover_Type',
                    fill=True,
                    common_norm=False,
                    data=train_df[train_df['Cover_Type'] != 5].sample(frac=0.4),ax=axis[i])


# In[12]:


plot_distribution_by_category(train_df[train_df['Cover_Type'] != 5])


# From the above graph we can clearly see that Elevation is an important feature giving great distinction between forst cover types.

# Let us now examine the distribution of categorical variables by cover type. We are plotting bar graph of count of categories for each forest type. Starting with soild features.

# In[13]:


def plot_soil_count_by_cover_type(train_df):
    fig, axis = plt.subplots(nrows=3,ncols=2,figsize=(25,25))
    axis = axis.flatten()
    ax_i = 0
    
    for soil_id in range(1,8):
        if soil_id  == 5:
            continue
        soil_count_df = pd.DataFrame()
        soil_count_df['soil_id'] = range(1,41)
        soil_count_df['count'] = train_df[train_df['Cover_Type'] == soil_id][soil_features].sum(axis=0).tolist()   
        sns.barplot(data=soil_count_df,x='soil_id',y='count',ax=axis[ax_i])
        axis[ax_i].set_xticks([ 1,  6, 11, 16, 21, 26, 31, 36])
        axis[ax_i].set_title('Cover_Type ' + str(soil_id))
        ax_i = ax_i + 1


# In[14]:


plot_soil_count_by_cover_type(train_df)


# Let us also visually examine the distributin of wilderness across forest type in a similar manner.

# In[15]:


def plot_wild_count_by_cover_type(train_df):
    fig, axis = plt.subplots(nrows=3,ncols=2,figsize=(25,25))
    axis = axis.flatten()
    ax_i = 0
    
    for wild_id in range(1,8):
        if wild_id == 5:
            continue
        wild_count_df = pd.DataFrame()
        wild_count_df['wild_id'] = range(1,5)
        wild_count_df['count'] = train_df[train_df['Cover_Type'] == wild_id][wild_features].sum(axis=0).tolist()   
        sns.barplot(data=wild_count_df,x='wild_id',y='count',ax=axis[ax_i])
        axis[ax_i].set_title('Cover_Type ' + str(wild_id))
        ax_i = ax_i + 1


# In[16]:


plot_wild_count_by_cover_type(train_df)


# #### Feature Engineering
# ###### 1. Clipping of -ve values from distance based metrics.
# ###### 2. Clip the values of angle and hillshade metrics in 0-255 and 0-360
# ###### 3. Introduce count of soil_type present and wild_area present as features
# ###### 4. Add l1 and l2 distance metrics using vertical and horizontal distances
# ###### 5. Add avg hillshade index

# In[17]:


if 'train_df_modified' in globals():
    del train_df_modified
if 'test_df_modified' in globals():
    del test_df_modified

train_df_modified = train_df.copy(deep=True)
test_df_modified = test_df.copy(deep=True)


# In[18]:


train_df_modified[distance_features] = train_df_modified[distance_features].clip(lower=0)
train_df_modified[hillshade_features] = train_df_modified[hillshade_features].clip(lower=0,upper=255)
train_df_modified['Aspect'][train_df_modified['Aspect'] < 0] += 360
train_df_modified['Aspect'][train_df_modified['Aspect'] >= 360] -= 360
train_df_modified['Slope'][train_df_modified['Slope'] < 0] += 360
train_df_modified['Slope'][train_df_modified['Slope'] >= 360] -= 360


test_df_modified[distance_features] = test_df_modified[distance_features].clip(lower=0)
test_df_modified[hillshade_features] = test_df_modified[hillshade_features].clip(lower=0,upper=255)
test_df_modified['Aspect'][test_df_modified['Aspect'] < 0] += 360
test_df_modified['Aspect'][test_df_modified['Aspect'] >= 360] -= 360
test_df_modified['Slope'][test_df_modified['Slope'] < 0] += 360
test_df_modified['Slope'][test_df_modified['Slope'] >= 360] -= 360


train_df_modified['mhtn_hydr_dist'] = np.abs(train_df_modified['Horizontal_Distance_To_Hydrology']) + np.abs(train_df_modified['Vertical_Distance_To_Hydrology'])
test_df_modified['mhtn_hydr_dist'] = np.abs(test_df_modified['Horizontal_Distance_To_Hydrology']) + np.abs(test_df_modified['Vertical_Distance_To_Hydrology'])

train_df_modified['eucd_hydr_dist'] = np.sqrt((train_df_modified['Horizontal_Distance_To_Hydrology'].astype(np.int32))**2 + 
                                        (train_df_modified['Vertical_Distance_To_Hydrology'].astype(np.int32))**2)
test_df_modified['eucd_hydr_dist'] = np.sqrt((test_df_modified['Horizontal_Distance_To_Hydrology'].astype(np.int32))**2 + 
                                        (test_df_modified['Vertical_Distance_To_Hydrology'].astype(np.int32))**2)


train_df_modified['mhtn_hydr_dist'] = np.abs(train_df_modified['Horizontal_Distance_To_Hydrology']) + np.abs(train_df_modified['Vertical_Distance_To_Hydrology'])
test_df_modified['mhtn_hydr_dist'] = np.abs(test_df_modified['Horizontal_Distance_To_Hydrology']) + np.abs(test_df_modified['Vertical_Distance_To_Hydrology'])

train_df_modified['eucd_hydr_dist'] = np.sqrt((train_df_modified['Horizontal_Distance_To_Hydrology'].astype(np.int32))**2 + 
                                        (train_df_modified['Vertical_Distance_To_Hydrology'].astype(np.int32))**2)
test_df_modified['eucd_hydr_dist'] = np.sqrt((test_df_modified['Horizontal_Distance_To_Hydrology'].astype(np.int32))**2 + 
                                        (test_df_modified['Vertical_Distance_To_Hydrology'].astype(np.int32))**2)


train_df_modified['soil_count'] = train_df_modified[soil_features].sum(axis=1)
train_df_modified['wild_count'] = train_df_modified[wild_features].sum(axis=1)

test_df_modified['soil_count'] = test_df_modified[soil_features].sum(axis=1)
test_df_modified['wild_count'] = test_df_modified[wild_features].sum(axis=1)

train_df_modified['avg_hillshade_index'] = train_df_modified[hillshade_features].mean(axis=1)
test_df_modified['avg_hillshade_index'] = test_df_modified[hillshade_features].mean(axis=1)

if 'Soil_Type7' in train_df_modified.columns and 'Soil_Type15' in train_df_modified.columns and 'Id' in train_df_modified.columns:
    train_df_modified = train_df_modified.drop(columns=['Soil_Type7','Soil_Type15','Id'])
if 'Soil_Type7' in test_df_modified.columns and 'Soil_Type15' in test_df_modified.columns and 'Id' in test_df_modified.columns:
    test_df_modified = test_df_modified.drop(columns=['Soil_Type7','Soil_Type15','Id'])


# In[19]:


if 'train_df' in globals():
    del train_df
if 'test_df' in globals():
    del test_df


# #### Memory Usage Reduction

# In[20]:


## The following function has been leveraged from an existing notebook
def reduce_mem_usage(df, verbose=True):
    """
    Reduce memory usage by downcasting features.
    
    Args:
        df (pd.DataFrame): DataFrame with features.
        verbose (bool): Determines verbosity of output.
    Returns:
        df (pd.DataFrame): DataFrame with reduces memory usage, due to smaller datatypes.
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
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
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[21]:


train_df_red = reduce_mem_usage(train_df_modified)
test_df_red = reduce_mem_usage(test_df_modified)


# In[22]:


if 'train_df_modified' in globals():
    del train_df_modified
if 'test_df_modified' in globals():
    del test_df_modified


# In[23]:


scaler = RobustScaler()
encoder = LabelEncoder() #Relabeling from 0 to 5


# In[24]:


X = train_df_red[train_df_red['Cover_Type'] != 5].drop(columns=['Cover_Type'])
y = train_df_red[train_df_red['Cover_Type'] != 5]['Cover_Type']
test_X = test_df_red

X = scaler.fit_transform(X)
test_X = scaler.transform(test_X)
y = encoder.fit_transform(y)


# In[25]:


model = CatBoostClassifier(iterations=5000,
                          task_type="GPU",
                          devices="0:1",
                          verbose=False)
model.fit(X,y)


# In[26]:


xgb_params = {
    'objective': 'multi:softmax',
    'eval_metric': 'mlogloss', 
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    }

model2 = XGBClassifier(**xgb_params)
model2.fit(X,y)

# VC = VotingClassifier(estimators = [('xgb',XGBClassifier(**xgb_params)),
#                                     ('lgbm_gdbt',LGBMClassifier(n_jobs=4)),
#                                     ('lgbm_dart',LGBMClassifier(boosting_type='dart',n_jobs=4)),
#                                     ('lgbm_goss',LGBMClassifier(boosting_type='goss',n_jobs=4))],
#                      voting='soft',flatten_transform=True,verbose=True)


# In[27]:


# VC.fit(X,y)


# In[28]:


pred = np.argmax(model.predict_proba(test_X)*0.5 + model2.predict_proba(test_X)*0.5,axis=1)


# In[29]:


pred_y = encoder.inverse_transform(pred) # reversing back to original labels


# In[30]:


sample_submission = pd.DataFrame(data=range(4000000,5000000),columns=['Id'])
sample_submission['Cover_Type'] = pd.DataFrame(data=pred_y,columns=['1'])['1']
sample_submission


# In[31]:


sample_submission.to_csv(path_or_buf='./submission.csv',index=False)


# 
