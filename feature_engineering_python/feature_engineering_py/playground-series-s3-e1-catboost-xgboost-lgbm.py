#!/usr/bin/env python
# coding: utf-8

# **Please upvote my notebook if You like it! :)**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


get_ipython().system('pip install reverse_geocoder')


# **Downloading data**

# In[ ]:


train_df = pd.read_csv('/kaggle/input/playground-series-s3e1/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s3e1/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s3e1/sample_submission.csv')
train_df = train_df.drop('id', axis=1)


# In[ ]:


extra_data = fetch_california_housing()
train_data2 = pd.DataFrame(extra_data['data'])
train_data2['MedHouseVal'] = extra_data['target']
train_data2.columns = train_df.columns
train_df['generated'] = 1
test_df['generated'] = 1
train_data2['generated'] = 0
# train_df = pd.concat([train_df, train_data2],axis=0).drop_duplicates()
train_df = pd.concat([train_df, train_data2],axis=0, ignore_index=True)

train_df.loc[33228,['Latitude','Longitude']] = [32.74, -117]
train_df.loc[34363,['Latitude','Longitude']] = [32.71, -117]
train_df.loc[20991,['Latitude','Longitude']] = [34.2, -119]

print(train_df.shape)
train_df.head()


# In[ ]:


train_df['r'] = np.sqrt(train_df['Latitude']**2 + train_df['Longitude']**2)
train_df['theta'] = np.arctan2(train_df['Latitude'], train_df['Longitude'])

test_df['r'] = np.sqrt(test_df['Latitude']**2 + test_df['Longitude']**2)
test_df['theta'] = np.arctan2(test_df['Latitude'], test_df['Longitude'])


# In[ ]:


df = pd.concat([train_df, test_df], axis=0, ignore_index=True)


# Encoding trick (see here: https://www.kaggle.com/competitions/playground-series-s3e1/discussion/376210)

# In[ ]:


emb_size = 20
precision = 1e6 

latlon = np.expand_dims(df[['Latitude', 'Longitude']].values, axis=-1) 

m = np.exp(np.log(precision) / emb_size) 
angle_freq = m ** np.arange(emb_size) 
angle_freq = angle_freq.reshape(1, 1, emb_size) 

latlon = latlon * angle_freq 
latlon[..., 0::2] = np.cos(latlon[..., 0::2]) 
latlon[..., 1::2] = np.sin(latlon[..., 1::2]) 
latlon = latlon.reshape(-1, 2 * emb_size) 

df['exp_latlon1'] = [lat[0] for lat in latlon]
df['exp_latlon2'] = [lat[1] for lat in latlon]


# Thanks to @dmitryuarov for feature engineering ideas with coordinates! Please, upvote his notebook: https://www.kaggle.com/code/dmitryuarov/ps-s3e1-coordinates-key-to-victory

# In[ ]:


from sklearn.decomposition import PCA

def pca(data):
    '''
    input: dataframe containing Latitude(x) and Longitude(y)
    '''
    coordinates = data[['Latitude','Latitude']].values
    pca_obj = PCA().fit(coordinates)
    pca_x = pca_obj.transform(data[['Latitude', 'Longitude']].values)[:,0]
    pca_y = pca_obj.transform(data[['Latitude', 'Longitude']].values)[:,1]
    return pca_x, pca_y

# train_df['pca_x'], train_df['pca_y'] = pca(train_df)
# test_df['pca_x'], test_df['pca_y'] = pca(test_df)
df['pca_x'], df['pca_y'] = pca(df)


# In[ ]:


from umap import UMAP
coordinates = df[['Latitude', 'Longitude']].values
umap = UMAP(n_components=2, n_neighbors=50, random_state=228).fit(coordinates)
df['umap_lat'] = umap.transform(coordinates)[:,0]
df['umap_lon'] = umap.transform(coordinates)[:,1]


# In[ ]:


def crt_crds(df): 
    df['rot_15_x'] = (np.cos(np.radians(15)) * df['Longitude']) + \
                      (np.sin(np.radians(15)) * df['Latitude'])
    
    df['rot_15_y'] = (np.cos(np.radians(15)) * df['Latitude']) + \
                      (np.sin(np.radians(15)) * df['Longitude'])
    
    df['rot_30_x'] = (np.cos(np.radians(30)) * df['Longitude']) + \
                      (np.sin(np.radians(30)) * df['Latitude'])
    
    df['rot_30_y'] = (np.cos(np.radians(30)) * df['Latitude']) + \
                      (np.sin(np.radians(30)) * df['Longitude'])
    
    df['rot_45_x'] = (np.cos(np.radians(45)) * df['Longitude']) + \
                      (np.sin(np.radians(45)) * df['Latitude'])
    return df

# train_df = crt_crds(train_df)
# test_df = crt_crds(test_df)
df = crt_crds(df)


# In[ ]:


import reverse_geocoder as rg
from sklearn.preprocessing import LabelEncoder

def geocoder(df):
    coordinates = list(zip(df['Latitude'], df['Longitude']))
    results = rg.search(coordinates)
    return results

# results = geocoder(train_df)
# train_df['place'] = [x['admin2'] for x in results]
# results = geocoder(test_df)
# test_df['place'] = [x['admin2'] for x in results]

results = geocoder(df)
df['place'] = [x['admin2'] for x in results]

places = ['Los Angeles County', 'Orange County', 'Kern County',
          'Alameda County', 'San Francisco County', 'Ventura County',
          'Santa Clara County', 'Fresno County', 'Santa Barbara County',
          'Contra Costa County', 'Yolo County', 'Monterey County',
          'Riverside County', 'Napa County']

def replace(x):
    if x in places:
        return x
    else:
        return 'Other'
    
# train_df['place'] = train_df['place'].apply(lambda x: replace(x))
# test_df['place'] = test_df['place'].apply(lambda x: replace(x))

df['place'] = df['place'].apply(lambda x: replace(x))

# le = LabelEncoder()
# train_df['place'] = le.fit_transform(train_df['place'])
# test_df['place'] = le.transform(test_df['place'])

# test_df = pd.get_dummies(test_df)
# train_df = pd.get_dummies(train_df)

df = pd.get_dummies(df)


# Distances to cities and coast points

# In[ ]:


from haversine import haversine

Sac = (38.576931, -121.494949)
SF = (37.780080, -122.420160)
SJ = (37.334789, -121.888138)
LA = (34.052235, -118.243683)
SD = (32.715759, -117.163818)

df['dist_Sac'] = df.apply(lambda x: haversine((x['Latitude'], x['Longitude']), Sac, unit='ft'), axis=1)
df['dist_SF'] = df.apply(lambda x: haversine((x['Latitude'], x['Longitude']), SF, unit='ft'), axis=1)
df['dist_SJ'] = df.apply(lambda x: haversine((x['Latitude'], x['Longitude']), SJ, unit='ft'), axis=1)
df['dist_LA'] = df.apply(lambda x: haversine((x['Latitude'], x['Longitude']), LA, unit='ft'), axis=1)
df['dist_SD'] = df.apply(lambda x: haversine((x['Latitude'], x['Longitude']), SD, unit='ft'), axis=1)
df['dist_nearest_city'] = df[['dist_Sac', 'dist_SF', 'dist_SJ', 
                              'dist_LA', 'dist_SD']].min(axis=1)


# In[ ]:


from shapely.geometry import LineString, Point

coast_points = LineString([(32.6644, -117.1613), (33.2064, -117.3831),
                           (33.7772, -118.2024), (34.4634, -120.0144),
                           (35.4273, -120.8819), (35.9284, -121.4892),
                           (36.9827, -122.0289), (37.6114, -122.4916),
                           (38.3556, -123.0603), (39.7926, -123.8217),
                           (40.7997, -124.1881), (41.7558, -124.1976)])

df['dist_to_coast'] = df.apply(lambda x: Point(x['Latitude'], x['Longitude']).distance(coast_points), axis=1)


# In[ ]:


# combine latitude and longitude
# codes from 
# https://datascience.stackexchange.com/questions/49553/combining-latitude-longitude-position-into-single-feature
from math import radians, cos, sin, asin, sqrt

def single_pt_haversine(lat, lng, degrees=True):
    """
    'Single-point' Haversine: Calculates the great circle distance
    between a point on Earth and the (0, 0) lat-long coordinate
    """
    r = 6371 # Earth's radius (km). Have r = 3956 if you want miles

    # Convert decimal degrees to radians
    if degrees:
        lat, lng = map(radians, [lat, lng])

    # 'Single-point' Haversine formula
    a = sin(lat/2)**2 + cos(lat) * sin(lng/2)**2
    d = 2 * r * asin(sqrt(a)) 

    return d
# add more metric 
# referred to this discussion
# https://www.kaggle.com/competitions/playground-series-s3e1/discussion/376210

def manhattan(lat,lng):
    return np.abs(lat) + np.abs(lng)
def euclidean(lat,lng):
    return (lat**2 + lng**2) **0.5

def add_combine(df):      
    df['haversine'] = [single_pt_haversine(x, y) for x, y in zip(df.Latitude, df.Longitude)]
    df['manhattan'] = [manhattan(x,y) for x,y in zip(df.Latitude, df.Longitude)]
    df['euclidean'] = [euclidean(x,y) for x,y in zip(df.Latitude,df.Longitude)]
    return df

df = add_combine(df)


# In[ ]:


# from sklearn.neighbors import KNeighborsRegressor

# neighbors = 15

# knn = KNeighborsRegressor(n_neighbors = neighbors,
#                             metric = 'haversine',
#                             n_jobs = -1)
# knn.fit(df[['Latitude','Longitude']], df.index)
# dists, nears = knn.kneighbors(df[['Latitude', 'Longitude']], 
#                                 return_distance = True)

# df[['dist1', 'dist2', 'dist3', 'dist4', 'dist5',
#     'dist6', 'dist7', 'dist8', 'dist9', 'dist10',
#     'dist11', 'dist12', 'dist13', 'dist14', 'dist15']] = dists


# In[ ]:


# df['number_houses_per_block'] = df['Population'] / df['AveOccup']
# df['total_income_of_block'] = df['MedInc'] * df['Population']
# df['occupants_to_bedrooms'] = df['AveOccup'] / df['AveBedrms']
# df['total_number_of_rooms'] = df['AveBedrms'] + df['AveRooms']
# df['bedrooms_to_rooms'] = df['AveBedrms'] / df['AveRooms']
# df['occupants_to_rooms'] = df['AveOccup'] / df['AveRooms']


# In[ ]:


df


# In[ ]:


train_df = df.iloc[:-len(test_df),:]
test_df = df.iloc[-len(test_df):,:].drop('MedHouseVal', axis=1).reset_index(drop=True)

X = train_df.drop(['MedHouseVal', 'id'], axis=1)
y = train_df.MedHouseVal
X_test = test_df.drop('id', axis=1)


# **Catboost model**

# In[ ]:


import catboost
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

n_folds = 15

MAX_ITER = 15000
PATIENCE = 1000
DISPLAY_FREQ = 100

eval_predsCB = []
predsCB = []

k_fold = KFold(n_splits=n_folds, random_state=42, shuffle=True)

MODEL_PARAMS = {
                'random_seed': 1234,    
#                 'learning_rate': 0.1,   # 0.15: 0.5678, 0.12: 0.5685, 0.1: 0.56757, 0.05: 0.57, 0.01, 0.57             
                'iterations': MAX_ITER,
                'early_stopping_rounds': PATIENCE,
#                 'metric_period': DISPLAY_FREQ,
                'use_best_model': True,
                'eval_metric': 'RMSE',
                'verbose': 1000,
#                 'task_type': 'GPU'
               }


for train_index, test_index in k_fold.split(X, y):
    X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    
    model = catboost.CatBoostRegressor(**MODEL_PARAMS)
    
    model.fit(X=X_train, y=y_train,
          eval_set=[(X_valid, y_valid)],
          early_stopping_rounds = PATIENCE,
#           metric_period = DISPLAY_FREQ
         )
    predsCB.append(model.predict(X_test))
#     eval_predsCB.append(model.predict(X))
#     print("RMSE valid = {}".format(mean_squared_error(y_valid, model.predict(X_valid))))
#     print("RMSE full = {}".format(mean_squared_error(y, model.predict(X))))


# **XGBoost model**

# In[ ]:


from xgboost import XGBRegressor

# n_folds = 20
k_fold = KFold(n_splits=n_folds, random_state=42, shuffle=True)

eval_predsXB = []
predsXB = []

PATIENCE = 200

MODEL_PARAMS = {       'n_estimators': 1000, #1000, 5000
#                        'learning_rate': 0.05,
                       'max_depth': 4, # 3
                       'colsample_bytree': 0.9, # 0.95
                       'subsample': 1,
                       'reg_lambda': 20,
                       'early_stopping_rounds': PATIENCE,
#                        'tree_method': 'gpu_hist',
                       'seed': 1
}

for train_index, test_index in k_fold.split(X, y):
    X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    
    model = XGBRegressor(**MODEL_PARAMS)
    
    model.fit(X=X_train, y=y_train,
          eval_set=[(X_valid, y_valid)],
#           early_stopping_rounds = PATIENCE,
          verbose = 100
         )
    predsXB.append(model.predict(X_test))
#     eval_predsXB.append(model.predict(X))


# **LGBM model**

# In[ ]:


import lightgbm as lgbm
from lightgbm.sklearn import LGBMRegressor

# n_folds = 20
k_fold = KFold(n_splits=n_folds, random_state=42, shuffle=True)

eval_predsLB = []
predsLB = []

MODEL_PARAMS = {
                       'learning_rate': 0.01,
                       'max_depth': 9,
                       'num_leaves': 90,
                       'colsample_bytree': 0.8,
                       'subsample': 0.9,
                       'subsample_freq': 5,
                       'min_child_samples': 36,
                       'reg_lambda': 28,
                       'n_estimators': 20000,
                       'metric': 'rmse',
                       'random_state': 1
}

callbacks = [lgbm.early_stopping(30, verbose=1), lgbm.log_evaluation(period=0)]

for train_index, test_index in k_fold.split(X, y):
    X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    
    model = lgbm.LGBMRegressor(**MODEL_PARAMS)
    
    model.fit(X=X_train, y=y_train,
          eval_set=[(X_valid, y_valid)],
#           early_stopping_rounds = PATIENCE,
          callbacks=callbacks
         )
    predsLB.append(model.predict(X_test))
#     eval_predsLB.append(model.predict(X))


# In[ ]:


# mean_squared_error(y, np.average(np.array(eval_predsCB),axis=0), squared=False)


# In[ ]:


# mean_squared_error(y, np.average(np.array(eval_predsXB),axis=0), squared=False)


# In[ ]:


# mean_squared_error(y, np.average(np.array(eval_predsLB),axis=0), squared=False)


# In[ ]:


a = 0.4
b = 0.2
c = 0.4


# In[ ]:


# mean_squared_error(y, a* np.average(np.array(eval_predsCB),axis=0) + b* np.average(np.array(eval_predsXB),axis=0) + c* np.average(np.array(eval_predsLB),axis=0), squared=False)


# **Making prediction**

# In[ ]:


predCB = np.average(np.array(predsCB),axis=0)
predXB = np.average(np.array(predsXB),axis=0)
predLB = np.average(np.array(predsLB),axis=0)
pred = predCB * a + predXB * b + predLB * c


# **Making submission**

# In[ ]:


submission['MedHouseVal'] = pred
submission


# In[ ]:


vals = train_df['MedHouseVal'].unique().tolist()
submission['MedHouseVal'] = submission['MedHouseVal'].apply(lambda x: min(vals, key=lambda v: abs(v - x)))
submission.MedHouseVal.clip(0, 5, inplace=True)


# In[ ]:


submission.to_csv('submission.csv', index=False)

