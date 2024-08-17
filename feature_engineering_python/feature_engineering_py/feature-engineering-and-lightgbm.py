#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
import lightgbm as lgb
import os, sys


# In[2]:


# Load data 
train = pd.read_csv('../input/bigquery-geotab-intersection-congestion/train.csv')
test = pd.read_csv('../input/bigquery-geotab-intersection-congestion/test.csv')
submission = pd.read_csv('../input/bigquery-geotab-intersection-congestion/sample_submission.csv')


# ## <span style='color:blue;background:yellow'>Data Cleaning and Preprocessing
# 

# In[3]:


train.nunique()


# In[4]:


print(train["City"].unique())
print(test["City"].unique())


# In[5]:


# test.groupby(["City"]).apply(np.unique)
test.groupby(["City"]).nunique()


# In[6]:


train.isna().sum(axis=0)


# In[7]:


test.isna().sum(axis=0)


# #### <span style='color:blue;background:yellow'>**Fill NAs**

# In[8]:


# %%time
# def fill_na(df):
#     df['ExitStreetName'] = df.apply(lambda x: x.EntryStreetName if type(x.ExitStreetName) != str else x.ExitStreetName, axis =1)
#     df['EntryStreetName'] = df.apply(lambda x: x.ExitStreetName if type(x.EntryStreetName) != str else x.EntryStreetName, axis =1)
#     df.fillna('ffill', inplace=True)
#     return df
# train = fill_na(train)
# test = fill_na(test)


# <span style='color:blue;background:yellow'> Road Encoding

# In[9]:


road_encoding = {
"Street":0,
 "St":0,
 "Avenue":1,
 "Ave":1,
 "Boulevard":2,
 "Road":3,
 "Drive":4,
 "Lane":5,
 "Tunnel":6,
 "Highway":7,
 "Way":8,
 "Parkway":9,
 "Parking":10,
 "Oval":11,
 "Square":12,
 "Place":13,
 "Bridge":14}


# In[10]:


def encode(x):
    if pd.isna(x):
        return 0
    for road in road_encoding.keys():
        if road in x:
            return road_encoding[road]
        
    return 0


# In[11]:


train['EntryType'] = train['EntryStreetName'].apply(encode)
train['ExitType'] = train['ExitStreetName'].apply(encode)
test['EntryType'] = test['EntryStreetName'].apply(encode)
test['ExitType'] = test['ExitStreetName'].apply(encode)


# <span style='color:blue;background:yellow'> add cordinal direction
# ##### turn direction: 
# The cardinal directions can be expressed using the equation: $$ \frac{\theta}{\pi} $$
# 
# Where $\theta$ is the angle between the direction we want to encode and the north compass direction, measured clockwise.
# 
# 

# In[12]:


directions = {
    'N': 0,
    'NE': 1/4,
    'E': 1/2,
    'SE': 3/4,
    'S': 1,
    'SW': 5/4,
    'W': 3/2,
    'NW': 7/4
}


# In[13]:


train['EntryHeading'] = train['EntryHeading'].map(directions)
train['ExitHeading'] = train['ExitHeading'].map(directions)

test['EntryHeading'] = test['EntryHeading'].map(directions)
test['ExitHeading'] = test['ExitHeading'].map(directions)


# In[14]:


train['diffHeading'] = train['EntryHeading']-train['ExitHeading']  
test['diffHeading'] = test['EntryHeading']-test['ExitHeading'] 


# In[15]:


train.head()


#  ### <span style='color:blue;background:yellow'>entering street == exit street?

# In[16]:


train["same_street_exact"] = (train["EntryStreetName"] ==  train["ExitStreetName"]).astype(int)
test["same_street_exact"] = (test["EntryStreetName"] ==  test["ExitStreetName"]).astype(int)


# <span style='color:blue;background:yellow'>Make a new columns--> Intersection ID + City name

# In[17]:


train["Intersection"] = train["IntersectionId"].astype(str) + train["City"]
test["Intersection"] = test["IntersectionId"].astype(str) + test["City"]


# In[18]:


encoder = LabelEncoder()
encoder.fit(pd.concat([train["Intersection"],test["Intersection"]]).drop_duplicates().values)
train["Intersection"] = encoder.transform(train["Intersection"])
test["Intersection"] = encoder.transform(test["Intersection"])


# <span style='color:blue;background:yellow'> Add temperature (°F) of each city by month </span>
#     
# 

# In[19]:


monthly_av = {'Atlanta1': 43, 'Atlanta5': 69, 'Atlanta6': 76, 'Atlanta7': 79, 'Atlanta8': 78, 'Atlanta9': 73,
              'Atlanta10': 62, 'Atlanta11': 53, 'Atlanta12': 45, 'Boston1': 30, 'Boston5': 59, 'Boston6': 68,
              'Boston7': 74, 'Boston8': 73, 'Boston9': 66, 'Boston10': 55,'Boston11': 45, 'Boston12': 35,
              'Chicago1': 27, 'Chicago5': 60, 'Chicago6': 70, 'Chicago7': 76, 'Chicago8': 76, 'Chicago9': 68,
              'Chicago10': 56,  'Chicago11': 45, 'Chicago12': 32, 'Philadelphia1': 35, 'Philadelphia5': 66,
              'Philadelphia6': 76, 'Philadelphia7': 81, 'Philadelphia8': 79, 'Philadelphia9': 72, 'Philadelphia10': 60,
              'Philadelphia11': 49, 'Philadelphia12': 40}
# Concatenating the city and month into one variable
train['city_month'] = train["City"] + train["Month"].astype(str)
test['city_month'] = test["City"] + test["Month"].astype(str)

# Creating a new column by mapping the city_month variable to it's corresponding average monthly temperature
train["average_temp"] = train['city_month'].map(monthly_av)
test["average_temp"] = test['city_month'].map(monthly_av)


# <span style='color:blue;background:yellow'> Add climate data </span>

# In[20]:


monthly_rainfall = {'Atlanta1': 5.02, 'Atlanta5': 3.95, 'Atlanta6': 3.63, 'Atlanta7': 5.12, 'Atlanta8': 3.67, 'Atlanta9': 4.09,
              'Atlanta10': 3.11, 'Atlanta11': 4.10, 'Atlanta12': 3.82, 'Boston1': 3.92, 'Boston5': 3.24, 'Boston6': 3.22,
              'Boston7': 3.06, 'Boston8': 3.37, 'Boston9': 3.47, 'Boston10': 3.79,'Boston11': 3.98, 'Boston12': 3.73,
              'Chicago1': 1.75, 'Chicago5': 3.38, 'Chicago6': 3.63, 'Chicago7': 3.51, 'Chicago8': 4.62, 'Chicago9': 3.27,
              'Chicago10': 2.71,  'Chicago11': 3.01, 'Chicago12': 2.43, 'Philadelphia1': 3.52, 'Philadelphia5': 3.88,
              'Philadelphia6': 3.29, 'Philadelphia7': 4.39, 'Philadelphia8': 3.82, 'Philadelphia9':3.88 , 'Philadelphia10': 2.75,
              'Philadelphia11': 3.16, 'Philadelphia12': 3.31}
# Creating a new column by mapping the city_month variable to it's corresponding average monthly rainfall
train["average_rainfall"] = train['city_month'].map(monthly_rainfall)
test["average_rainfall"] = test['city_month'].map(monthly_rainfall)


# In[21]:


monthly_snowfall = {'Atlanta1': 0.6, 'Atlanta5': 0, 'Atlanta6': 0, 'Atlanta7': 0, 'Atlanta8': 0, 'Atlanta9': 0,
              'Atlanta10': 0, 'Atlanta11': 0, 'Atlanta12': 0.2, 'Boston1': 12.9, 'Boston5': 0, 'Boston6': 0,
              'Boston7': 0, 'Boston8': 0, 'Boston9': 0, 'Boston10': 0,'Boston11': 1.3, 'Boston12': 9.0,
              'Chicago1': 11.5, 'Chicago5': 0, 'Chicago6': 0, 'Chicago7': 0, 'Chicago8': 0, 'Chicago9': 0,
              'Chicago10': 0,  'Chicago11': 1.3, 'Chicago12': 8.7, 'Philadelphia1': 6.5, 'Philadelphia5': 0,
              'Philadelphia6': 0, 'Philadelphia7': 0, 'Philadelphia8': 0, 'Philadelphia9':0 , 'Philadelphia10': 0,
              'Philadelphia11': 0.3, 'Philadelphia12': 3.4}

# Creating a new column by mapping the city_month variable to it's corresponding average monthly snowfall
train["average_snowfall"] = train['city_month'].map(monthly_snowfall)
test["average_snowfall"] = test['city_month'].map(monthly_snowfall)


# In[22]:


monthly_daylight = {'Atlanta1': 10, 'Atlanta5': 14, 'Atlanta6': 14, 'Atlanta7': 14, 'Atlanta8': 13, 'Atlanta9': 12,
              'Atlanta10': 11, 'Atlanta11': 10, 'Atlanta12': 10, 'Boston1': 9, 'Boston5': 15, 'Boston6': 15,
              'Boston7': 15, 'Boston8': 14, 'Boston9': 12, 'Boston10': 11,'Boston11': 10, 'Boston12': 9,
              'Chicago1': 10, 'Chicago5': 15, 'Chicago6': 15, 'Chicago7': 15, 'Chicago8': 14, 'Chicago9': 12,
              'Chicago10': 11,  'Chicago11': 10, 'Chicago12': 9, 'Philadelphia1': 10, 'Philadelphia5': 14,
              'Philadelphia6': 15, 'Philadelphia7': 15, 'Philadelphia8': 14, 'Philadelphia9':12 , 'Philadelphia10': 11,
              'Philadelphia11': 10, 'Philadelphia12': 9}

# Creating a new column by mapping the city_month variable to it's corresponding average monthly daylight
train["average_daylight"] = train['city_month'].map(monthly_daylight)
test["average_daylight"] = test['city_month'].map(monthly_daylight)


# In[23]:


monthly_sunsine = {'Atlanta1': 5.3, 'Atlanta5': 9.3, 'Atlanta6': 9.5, 'Atlanta7': 8.8, 'Atlanta8': 8.3, 'Atlanta9': 7.6,
              'Atlanta10': 7.7, 'Atlanta11': 6.2, 'Atlanta12': 5.3, 'Boston1': 5.3, 'Boston5': 8.6, 'Boston6': 9.6,
              'Boston7': 9.7, 'Boston8': 8.9, 'Boston9': 7.9, 'Boston10': 6.7,'Boston11': 4.8, 'Boston12': 4.6,
              'Chicago1': 4.4, 'Chicago5': 9.1, 'Chicago6': 10.4, 'Chicago7': 10.3, 'Chicago8': 9.1, 'Chicago9': 7.6,
              'Chicago10': 6.2,  'Chicago11': 3.6, 'Chicago12': 3.4, 'Philadelphia1': 5.0, 'Philadelphia5': 7.9,
              'Philadelphia6': 9.0, 'Philadelphia7': 8.9, 'Philadelphia8': 8.4, 'Philadelphia9':7.9 , 'Philadelphia10': 6.6,
              'Philadelphia11': 5.2, 'Philadelphia12': 4.4}

# Creating a new column by mapping the city_month variable to it's corresponding average monthly sunsine
train["average_sunsine"] = train['city_month'].map(monthly_sunsine)
test["average_sunsine"] = test['city_month'].map(monthly_sunsine)


# In[24]:


train.drop('city_month', axis=1, inplace=True)
test.drop('city_month', axis=1, inplace=True)


# <span style='color:blue;background:yellow'> New feature --> is_day

# In[25]:


train['is_day'] = train['Hour'].apply(lambda x: 1 if 5 < x < 20 else 0)
test['is_day'] = test['Hour'].apply(lambda x: 1 if 5 < x < 20 else 0)


# <span style='color:blue;background:yellow'>  Distance from center of city

# In[26]:


def add_distance(df):
    
    df_center = pd.DataFrame({"Atlanta":[33.753746, -84.386330],
                             "Boston":[42.361145, -71.057083],
                             "Chicago":[41.881832, -87.623177],
                             "Philadelphia":[39.952583, -75.165222]})
    
    df["CenterDistance"] = df.apply(lambda row: math.sqrt((df_center[row.City][0] - row.Latitude) ** 2 +
                                                          (df_center[row.City][1] - row.Longitude) ** 2) , axis=1)

add_distance(train)
add_distance(test)


# * <span style='color:blue;background:yellow'>  HotEncoding of cities

# In[27]:


train = pd.concat([train,pd.get_dummies(train["City"],dummy_na=False, drop_first=False)],axis=1).drop(["City"],axis=1)
test = pd.concat([test,pd.get_dummies(test["City"],dummy_na=False, drop_first=False)],axis=1).drop(["City"],axis=1)


# In[28]:


# scale Log and lat columns
scaler = StandardScaler()
for col in ['Latitude','Longitude']:
    scaler.fit(train[col].values.reshape(-1, 1))
    train[col] = scaler.transform(train[col].values.reshape(-1, 1))
    test[col] = scaler.transform(test[col].values.reshape(-1, 1))


# In[29]:


train.head()


# In[30]:


train.shape,test.shape


# In[31]:


train_road_id = train['RowId']
test_road_id = test['RowId']
preds = train.iloc[:,12:27]
train.drop(['RowId', 'Path','EntryStreetName','ExitStreetName'],axis=1, inplace=True)
test.drop(['RowId', 'Path','EntryStreetName','ExitStreetName'],axis=1, inplace=True)


# In[32]:


plt.subplots(figsize=(16,12))
sns.heatmap(train.corr(), color ='BGR4R')


# In[33]:


train.corr()


# In[34]:


train.drop(preds.columns.tolist(), axis=1, inplace =True)


# In[35]:


target1 = preds['TotalTimeStopped_p20']
target2 = preds['TotalTimeStopped_p50']
target3 = preds['TotalTimeStopped_p80']
target4 = preds['DistanceToFirstStop_p20']
target5 = preds['DistanceToFirstStop_p50']
target6 = preds['DistanceToFirstStop_p80']


# In[36]:


train.columns


# In[37]:


cat_feat = ['IntersectionId','Hour', 'Weekend','Month', 'same_street_exact', 'Intersection',
       'Atlanta', 'Boston', 'Chicago', 'Philadelphia', 'EntryType', 'ExitType']


# In[38]:


all_preds ={0:[],1:[],2:[],3:[],4:[],5:[]}
all_target = [target1, target2, target3, target4, target5, target6]


# ### Reference: https://medium.com/analytics-vidhya/hyperparameters-optimization-for-lightgbm-catboost-and-xgboost-regressors-using-bayesian-6e7c495947a9

# In[39]:


dtrain = lgb.Dataset(data=train, label=target3)

# Objective Function
def hyp_lgbm(num_leaves, feature_fraction, bagging_fraction, max_depth, min_split_gain, min_child_weight, lambda_l1, lambda_l2):
      
        params = {'application':'regression','num_iterations': 450,
                  'learning_rate':0.02,
                  'metric':'rmse'} # Default parameters
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        params['lambda_l1'] = lambda_l1
        params['lambda_l2'] = lambda_l2
        
        cv_results = lgb.cv(params, dtrain, nfold=5, seed=17,categorical_feature=cat_feat, stratified=False,
                            verbose_eval =None)
#         print(cv_results)
        return -np.min(cv_results['rmse-mean'])


# In[40]:


# Domain space-- Range of hyperparameters
pds = {'num_leaves': (120, 230),
          'feature_fraction': (0.3, 0.9),
          'bagging_fraction': (0.8, 1),
           'lambda_l1': (0,3),
           'lambda_l2': (0,5),
          'max_depth': (8, 19),
          'min_split_gain': (0.001, 0.1),
          'min_child_weight': (1, 20)
          }


# In[41]:


# Surrogate model
optimizer = BayesianOptimization(hyp_lgbm,pds,random_state=7)
                                  
# Optimize
optimizer.maximize(init_points=5, n_iter=12)


# In[42]:


optimizer.max


# In[43]:


p = optimizer.max['params']


# In[44]:


param = {'num_leaves': int(round(p['num_leaves'])),
         'feature_fraction': p['feature_fraction'],
         'bagging_fraction': p['bagging_fraction'],
         'max_depth': int(round(p['max_depth'])),
         'lambda_l1': p['lambda_l1'],
         'lambda_l2': p['lambda_l2'],
         'min_split_gain': p['min_split_gain'],
         'min_child_weight': p['min_child_weight'],
         'learning_rate':0.05,
         'objective': 'regression',
         'boosting_type': 'gbdt',
         'verbose': 1,
         'metric': 'rmse',
         'seed': 7,
        }


# In[45]:


param


# In[46]:


get_ipython().run_cell_magic('time', '', 'nfold = 5\nkf = KFold(n_splits=nfold, random_state=227, shuffle=True)\nfor i in range(len(all_preds)):\n    print(\'Training and predicting for target {}\'.format(i+1))\n    oof = np.zeros(len(train))\n    all_preds[i] = np.zeros(len(test))\n    n =1\n    for train_index, valid_index in kf.split(all_target[i]):\n        print("fold {}".format(n))\n        xg_train = lgb.Dataset(train.iloc[train_index],\n                               label=all_target[i][train_index]\n                               )\n        xg_valid = lgb.Dataset(train.iloc[valid_index],\n                               label=all_target[i][valid_index]\n                               )   \n\n        clf = lgb.train(param, xg_train, 15000, valid_sets=[xg_valid],categorical_feature=cat_feat\n                        , verbose_eval=200, early_stopping_rounds=500)\n        oof[valid_index] = clf.predict(train.iloc[valid_index], num_iteration=clf.best_iteration) \n\n        all_preds[i] += clf.predict(test, num_iteration=clf.best_iteration) / nfold\n        n = n + 1\n\n    print("\\n\\nCV RMSE: {:<0.4f}".format(np.sqrt(mean_squared_error(all_target[i], oof))))  \n')


# In[47]:


data2 = pd.DataFrame(all_preds).stack()
data2 = pd.DataFrame(data2)
submission['Target'] = data2[0].values


# In[48]:


submission.head(15)


# In[49]:


submission.to_csv('submission.csv', index=False)


# <center><span style='color:blue;background:yellow;font-size:30px'>**Please upvote this notebook if you like my work. **

# **References:**
# 1. https://www.kaggle.com/whatust/fork-of-kernel56e53f4445
# 2. https://www.kaggle.com/brokenfulcrum/geotab-baseline
# 3. https://www.kaggle.com/danofer/baseline-feature-engineering-geotab-69-5-lb
# 4.  https://www.kaggle.com/bgmello/how-one-percentile-affect-the-others
