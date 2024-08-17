#!/usr/bin/env python
# coding: utf-8

# In[10]:


get_ipython().system('pip install fasteda')

from fasteda import fast_eda


# # Importing basic libraries

# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Configures

# In[76]:


class conf:
    index = 'id'
    target = 'booking_status'
    random = 42
    folds = 8

np.random.seed(conf.random)


# # Importing Data

# In[101]:


train_full = pd.read_csv('/kaggle/input/playground-series-s3e7/train.csv',index_col=conf.index)
test_full = pd.read_csv('/kaggle/input/playground-series-s3e7/test.csv',index_col=conf.index)
orginal_df = pd.read_csv('/kaggle/input/reservation-cancellation-prediction/train__dataset.csv')
train = train_full.copy()
test = test_full.copy()


# ## Adding Extra Data

# In[105]:


train = pd.concat([train,orginal_df],axis=0)
train = train.drop_duplicates()
train = train.reset_index(drop=True)


# # Fixing anomalies 

# [Fix your date anomalies like a boss](http://www.kaggle.com/competitions/playground-series-s3e7/discussion/386655)

# In[110]:


# Creates a dummy date to find days_in_month and clips the values by assignment.
train['arrival_year_month'] = pd.to_datetime(train['arrival_year'].astype(str)
                                            +train['arrival_month'].astype(str), format='%Y%m')
test['arrival_year_month'] = pd.to_datetime(test['arrival_year'].astype(str)
                                            +test['arrival_month'].astype(str), format='%Y%m')

train.loc[train.arrival_date > train.arrival_year_month.dt.days_in_month, 'arrival_date'] = train.arrival_year_month.dt.days_in_month
test.loc[test.arrival_date > test.arrival_year_month.dt.days_in_month, 'arrival_date'] = test.arrival_year_month.dt.days_in_month

train.drop(columns='arrival_year_month', inplace=True)
test.drop(columns='arrival_year_month', inplace=True)


# In[111]:


fast_eda(train,target=conf.target)


# As we can see from the previous charts, there is a lot of outliers.

# # Feature Engineering

# In[44]:


# def fe(df):
        # Fix date anomalies (pd.to_datetime throws parsing error for some days, see anomalies section).
#         df['year_month'] = pd.to_datetime(df[['arrival_year', 'arrival_month']].astype(str).sum(1), format='%Y%m')
#         df.loc[df.arrival_date > df.year_month.dt.days_in_month, 'arrival_date'] = df.year_month.dt.days_in_month
#         df.drop(columns='year_month', inplace=True)
        
# #         # Creates date features.
#         df['arrival_full_date'] = (df['arrival_year'].astype(str) 
#                                    + '-' + df['arrival_month'].astype(str)
#                                    + '-' + df['arrival_date'].astype(str))
#         df['arrival_full_date'] = pd.to_datetime(df.arrival_full_date)
#         df['arrival_week'] = df['arrival_full_date'].dt.isocalendar().week.astype(float)
#         df['arrival_dayofweek'] = df['arrival_full_date'].dt.dayofweek
#         df['arrival_quarter'] = df['arrival_full_date'].dt.quarter
#         df['arrival_dayofyear'] = df['arrival_full_date'].dt.dayofyear
        
#         # Creates the season and holiday features. (also you can add holidays).
#         # ['winter', 'spring', 'summer', 'fall']
#         df['season'] = df.arrival_month%12 // 3 + 1
#         cal = USFederalHolidayCalendar()
#         holidays = cal.holidays(start='2017-01-01', end='2018-12-31')
#         df['is_holiday'] = 0
#         df.loc[df.arrival_full_date.isin(holidays), 'is_holiday'] = 1
        
#         # Aggregation by `season` as key and 'avg_price_per_room' as value (you can try quarters, months, etc).
#         aggr_df = df.groupby(by=key, sort=False)['avg_price_per_room'].agg(['mean', 'std', 'min', 'max', 'sum', 'last'])
#         aggr_df = aggr_df.add_prefix('avg_price_per_room_')
#         df = df.merge(aggr_df.reset_index(), on=key, how='left')
        
#         # Interaction between the correlated features and also lead time.
#         df['no_of_adults_div_price'] = df.no_of_adults / (df.avg_price_per_room + 1e-6)
#         df['no_of_children_div_price'] = df.no_of_children / (df.avg_price_per_room + 1e-6)
#         df['lead_time_div_price'] = df.lead_time / (df.avg_price_per_room + 1e-6)
#         df.drop(columns=['arrival_full_date'], inplace=True)
#         return df
# train = fe(train)
# test = fe(test)


# In[45]:


# cat_col = [
#     'required_car_parking_space',
#     'market_segment_type',
#     'room_type_reserved', 
#     'type_of_meal_plan',
#     'arrival_year'
# ]


# ## Handling Outliers 
# ### Using Floor and Capping style

# In[112]:


def capping(df):
    for col in df.columns:
        percentiles = df[col].quantile([0.01, 0.99]).values
        df[col][df[col] <= percentiles[0]] = percentiles[0]
        df[col][df[col] >= percentiles[1]] = percentiles[1]

capping(train)
capping(test)


# # Modeling

# In[82]:


import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, MinMaxScaler, RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_predict, KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBClassifier
from tqdm.notebook import tqdm 

get_ipython().run_line_magic('matplotlib', 'inline')


# #  Prepare Data

# In[113]:


x_full = train.copy()
y_full = x_full.pop(conf.target).to_numpy()

num_col = list(train.drop(conf.target,axis=1).select_dtypes(include=['int64','float64']).columns)
print(num_col)
print('-'*180)

tr = ColumnTransformer([
    ('num',MinMaxScaler(), num_col)
])

x_full = tr.fit_transform(x_full)
x_test = tr.transform(test)
print("train shape = ", x_full.shape)
print("test shape = ", x_test.shape)


# # Train with StratifiedKfold

# In[114]:


models = []
skf = StratifiedKFold(n_splits=conf.folds, random_state=conf.random,shuffle=True)


# In[115]:


# Preparing the grid to estimate the best parameters for the xgboost
param_grid = {'n_estimators': [100,150,200,250,300,600],
   "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ],
   "max_depth": [ 3, 4, 5, 6, 8, 10, 12, 15],
   "min_child_weight": [ 1, 3, 5, 7 ],
   "gamma": [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
   "colsample_bytree": [ 0.3, 0.4, 0.5 , 0.7 ]
}


# In[116]:


from sklearn.model_selection import train_test_split
X = train.drop(conf.target,axis=1)
y = train[conf.target]
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.33,stratify=train['booking_status'],random_state=conf.random)


# ## Random Search

# In[117]:


RS = RandomizedSearchCV(estimator=XGBClassifier(early_stopping_rounds=30,tree_method="gpu_hist",random_state=conf.random), 
                        param_distributions=param_grid, 
                        n_iter=5, cv=5, random_state=conf.random)
RS.fit(X_train, y_train,
      eval_set=[(X_train,y_train),(X_val,y_val)],
       verbose=0)


# In[118]:


RS.best_params_


# ### best parameters 
# {'n_estimators': 100,
#   'min_child_weight': 5,
#   'max_depth': 10,
#   'learning_rate': 0.2,
#   'gamma': 0.0,
#   'colsample_bytree': 0.4}

# ## Grid Search
# ### Which will take more time but its results will be more accurate.

# In[27]:


# GS = GridSearchCV(estimator=XGBClassifier(early_stopping_rounds=30,tree_method="gpu_hist"), 
#                         param_grid=param_grid,n_jobs=-1)
# GS.fit(X_train, y_train,
#       eval_set=[(X_train,y_train),(X_val,y_val)],
#        verbose=0)

# # GS.best_params_
# {'colsample_bytree': 0.3,
#  'gamma': 0.3,
#  'learning_rate': 0.25,
#  'max_depth': 7,
#  'n_estimators': 300}


# In[119]:


scores = []
i=1
for train_index, val_index in tqdm(skf.split(x_full, y_full),total=conf.folds):
    X_train, X_valid = x_full[train_index], x_full[val_index]
    y_train, y_valid = y_full[train_index], y_full[val_index]
    
    m = XGBClassifier(**RS.best_params_,tree_method="gpu_hist",random_state=conf.random)
    m.fit(X_train,y_train)

    models.append(m)
    score = roc_auc_score(y_valid, m.predict_proba(X_valid)[:, 1])
    scores.append(score)
    print('*'*50)
    print(f'the score for the training in {i} fold/s: {score}')
    i+=1
print(f'mean score: {np.mean(scores):.4f}')


# # Feature Importance

# In[120]:


feature_importance =  [models[x].feature_importances_ for x in range(conf.folds)]
feature_importance = np.average(feature_importance,axis=0)
feature_importance


# In[121]:


sorted_idx = np.argsort(feature_importance)
fig = plt.figure(figsize=(20, 15))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(test.columns)[sorted_idx])
plt.title('Feature Importance')


# # Predict

# In[122]:


test_preds = []

for m in models:
    preds = m.predict_proba(x_test)[:, 1]
    test_preds.append(preds)


# In[123]:


test_preds = np.array(test_preds).mean(0)

pd.DataFrame(test_preds).hist(bins=25, figsize=(10,6))


# In[125]:


sub = pd.read_csv("/kaggle/input/playground-series-s3e7/sample_submission.csv", index_col=conf.index)
sub[conf.target] = test_preds
sub.to_csv("submission.csv")
sub.head()


# # In Progress

# In[ ]:




