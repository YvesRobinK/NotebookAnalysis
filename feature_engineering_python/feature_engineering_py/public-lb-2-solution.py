#!/usr/bin/env python
# coding: utf-8

# Hi Kagglers!
# 
# The notebook include the model and tricks I used in this competition for my final solution.
# 
# The only external data is from my another notebook of clustering: [Clustering might help😎](https://www.kaggle.com/code/patrick0302/clustering-might-help)
# 
# While this competition relies less on ML models and modeling skills, it does require time series analysis and error observations to fine-tune submissions.
# 
# So, I've also released what and how I did in these published notebooks:
# - [Errors and Where to Find Them🤔](https://www.kaggle.com/code/patrick0302/errors-and-where-to-find-them)
# - [Find and fix the error bug🐛](https://www.kaggle.com/code/patrick0302/find-and-fix-the-error-bug)
# - [Fix more error bugs 🐞](https://www.kaggle.com/code/patrick0302/fix-more-error-bugs)
# - [Clustering might help😎](https://www.kaggle.com/code/patrick0302/clustering-might-help)
# 
# No matter you like the competition or not, hope you still learned something from it : )

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import gc
import numpy as np 
import pandas as pd
import datetime as dt

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

import folium
from haversine import haversine


from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_columns', None)

seed = 228


# In[2]:


# Read files
path = '/kaggle/input/playground-series-s3e20/'
path_cluster = '/kaggle/input/clustering-might-help/'
train = pd.read_csv(path_cluster+'train_with_ClusterNo.csv')
test = pd.read_csv(path_cluster+'test_with_ClusterNo.csv')
ss = pd.read_csv(path+'sample_submission.csv')


# ## Modeling and feature engineering

# The model is from @dmitryuarov's notebook: https://www.kaggle.com/code/dmitryuarov/ps3e20-rwanda-emission-advanced-fe-20-88
# 
# Thanks for his great work on feature engineering and modeling!

# In[3]:


def get_id(row):
    return int(''.join(filter(str.isdigit, str(row['latitude']))) + ''.join(filter(str.isdigit, str(row['longitude']))))

train['id'] = train[['latitude', 'longitude']].apply(lambda row: get_id(row), axis=1)
test['id'] = test[['latitude', 'longitude']].apply(lambda row: get_id(row), axis=1)
new_ids = {id_: new_id for new_id, id_ in enumerate(train['id'].unique())}
train['id'] = train['id'].map(new_ids)
test['id'] = test['id'].map(new_ids)

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

def get_month(row):
    date = dt.datetime.strptime(f'{row["year"]}-{row["week_no"]+1}-1', "%Y-%W-%w")
    return date.month

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
    
    cols_save = ['id', 'latitude', 'longitude', 'year', 'week_no', 'Ozone_solar_azimuth_angle', 'ClusterNo']
    df = df[cols_save]
    
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
    df['is_lockdown'] = (df['year'] == 2020) & ((df['month'].isin([3,4])))
    df['is_covid_peak'] = (df['year'] == 2020) & ((df['month'].isin([4,5,6])))
    df['is_covid_dis_peak'] = (df['year'] == 2021) & ((df['month'].isin([7,8,9])))
    df['public_holidays'] = (df['week_no'].isin([0, 51, 12, 30]))
    
#     df['high_em'] = (df['week_no'].between(14, 17)) | (df['week_no'].between(40, 43)) 
            
    df.fillna(0, inplace=True)
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

X = train.drop('id', axis=1)
test = test.drop('id', axis=1)


# My final solution is based on a single random forest model with 3000 estimators (aka trees).

# In[4]:


final_preds = np.zeros(len(test))
train['emission'] = y

rf = RandomForestRegressor(n_estimators=3000, random_state=seed, n_jobs=-1)
rf.fit(X, y)
final_preds = rf.predict(test)

ss['emission'] = final_preds


# ## Postprocessing

# In[5]:


ss['id'] = np.array(train[(train['week_no']<49)&(train['year']==2021)]['id'])
ss['week_no'] = ss.groupby('id').cumcount()


# Multipliers for each cluster:

# In[6]:


coeffs_pred_cluster = [1.10, #ClusterNo 0
                       1.02, #ClusterNo 1
                       1.10, #ClusterNo 2
                       1.07, #ClusterNo 3
                       1.10, #ClusterNo 4
                       1.05, #ClusterNo 5
                       1.07, #ClusterNo 6
                       1.07, #7
                      ]

test = test.reset_index(drop=True)
for ClusterNo in range(8):
    ss.loc[test['ClusterNo']==ClusterNo, 'emission'] = ss.loc[test['ClusterNo']==ClusterNo, 'emission']*coeffs_pred_cluster[ClusterNo]


# Fix error bugs:
# - [Find and fix the error bug🐛](https://www.kaggle.com/code/patrick0302/find-and-fix-the-error-bug)
# - [Fix more error bugs 🐞](https://www.kaggle.com/code/patrick0302/fix-more-error-bugs)

# In[7]:


ss.loc[test['longitude']==29.321, 'emission'] = train.loc[(train['year']==2021)&(train['week_no']<=48)&(train['longitude']==29.321),'emission'].values

coeff_29222 = 1.10
ss.loc[test['longitude']==29.222, 'emission'] = pd.Series(final_preds).loc[test['longitude']==29.222].values * coeff_29222


# Multiplier for ClusterNo 0 and 4's low and high values:
# - [Clustering might help😎](https://www.kaggle.com/code/patrick0302/clustering-might-help)

# In[8]:


coeff_low_values = 0.962

coeff_hi_values = 0.995

ss.loc[(test['ClusterNo'].isin([0,4])), 'emission'] = coeff_hi_values * ss.loc[(test['ClusterNo'].isin([0,4])), 'emission']

ss.loc[(test['ClusterNo'].isin([0,4]))&(test['week_no']<=13), 'emission'] = coeff_low_values * ss.loc[(test['ClusterNo'].isin([0,4]))&(test['week_no']<=13), 'emission']
ss.loc[(test['ClusterNo'].isin([0,4]))&(test['week_no']>=17)&(test['week_no']<=39), 'emission'] = coeff_low_values * ss.loc[(test['ClusterNo'].isin([0,4]))&(test['week_no']>=17)&(test['week_no']<=39), 'emission']
ss.loc[(test['ClusterNo'].isin([0,4]))&(test['week_no']>=44), 'emission'] = coeff_low_values * ss.loc[(test['ClusterNo'].isin([0,4]))&(test['week_no']>=44), 'emission']


# Assume there are shifts in ClusterNo0 and 4 and therefore doing some shift:

# In[9]:


test['id'] = test[['latitude', 'longitude']].apply(lambda row: get_id(row), axis=1)
test['id'] = test['id'].map(new_ids)

ss.loc[(test['ClusterNo'].isin([0,4]))&(test['week_no']==13), 'emission'] = np.nan
ss = ss.fillna(method='bfill')

ss.loc[(test['ClusterNo'].isin([0,4]))&(test['week_no']==17), 'emission'] = np.nan
ss = ss.fillna(method='ffill')

ss.loc[(test['ClusterNo'].isin([0,4]))&(test['week_no']==39), 'emission'] = np.nan
ss = ss.fillna(method='ffill')

ss.loc[(test['ClusterNo'].isin([0,4]))&(test['week_no']==44), 'emission'] = np.nan
ss = ss.fillna(method='bfill')


# In[10]:


ss.drop(['id', 'week_no'], axis=1, inplace=True)
ss.to_csv('submission.csv', index=False)


# In[ ]:




