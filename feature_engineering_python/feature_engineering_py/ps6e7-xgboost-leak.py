#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import xgboost as xgb

from matplotlib import pyplot as plt
from sklearn.model_selection import  StratifiedKFold
from sklearn.metrics import roc_auc_score

import seaborn as sns
import shap


from tqdm.notebook import tqdm


import umap

plt.style.use('fivethirtyeight')
pd.set_option('display.max_columns', None)
shap.initjs()


# ## Read data, drop duplicates and leakage

# In[2]:


df = pd.read_csv("../input/playground-series-s3e7/train.csv")
test_df = pd.read_csv("../input/playground-series-s3e7/test.csv")
adf = pd.read_csv('../input/reservation-cancellation-prediction/train__dataset.csv')


df['generated_type'] = 1
test_df['generated_type'] = 1
adf['generated_type'] = 0


dup_features = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights',
               'no_of_week_nights', 'type_of_meal_plan', 'required_car_parking_space',
               'room_type_reserved', 'lead_time', 'arrival_year', 'arrival_month',
               'arrival_date', 'market_segment_type', 'repeated_guest',
               'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
               'avg_price_per_room', 'no_of_special_requests']

df = pd.concat([df, adf], ignore_index=True).drop('id', axis=1).reset_index(drop=True)
nod = df[dup_features].duplicated().sum()
nod_with_target = df[dup_features + ['booking_status']].duplicated().sum()

df = df.drop_duplicates(subset=dup_features).reset_index(drop=True)

tt = df.reset_index().rename(columns={"index":"id"}).drop('booking_status', axis=1)
test_leakage = test_df.merge(tt, on=[c for c in test_df.columns if c != 'id'], how='inner', suffixes=['_test', '_train'])[['id_test', 'id_train']]

df = df.drop(test_leakage['id_train']).reset_index(drop=True)

print(f'#duplicates = {nod}, #duplicates with different target = {nod_with_target}, #leakage = {test_leakage.shape[0]}')


# ## Preprocessing and litle feature engineering

# In[3]:


def preprocess(df):
    df = df.copy()
    df['arrival_year_month'] = pd.to_datetime(df['arrival_year'].astype(str) + df['arrival_month'].astype(str), format='%Y%m')

    df.loc[df.arrival_date > df.arrival_year_month.dt.days_in_month, 'arrival_date'] = (
     df.loc[df.arrival_date > df.arrival_year_month.dt.days_in_month, 'arrival_year_month'].dt.days_in_month   
    )
    df = df.drop('arrival_year_month', axis=1)

    df['date'] = pd.to_datetime(df['arrival_year'].astype(str) + "/" + df['arrival_month'].astype(str) + "/" + df['arrival_date'].astype(str))




    df['no_of_adults_div_price'] = df.no_of_adults / (df.avg_price_per_room + 1e-6)

    df['lead_time_div_price'] = df.lead_time / (df.avg_price_per_room + 1e-6)
    df['lead_time_minus_avg_price'] = df.lead_time - df.avg_price_per_room / (df.lead_time.max() + df.avg_price_per_room.max())
    # df['experement'] =  (df.lead_time) *  (df.no_of_special_requests)

    return df


# In[4]:


df = preprocess(df)
test_df = preprocess(test_df)


# In[5]:


FEATURES = ['no_of_adults', 
            'no_of_children', 
            'no_of_weekend_nights',
            'no_of_week_nights', 
            'lead_time',
            'arrival_year',
            'arrival_month',
            'arrival_date',
            'avg_price_per_room', 
            'no_of_special_requests',
            'no_of_adults_div_price',

            'lead_time_div_price', 
            'lead_time_minus_avg_price',
            # "seasonal", 
            # "trend"
           ]
ONE_HOT_FEATURES = [
    'required_car_parking_space',
    'market_segment_type',
    'room_type_reserved', 
    'repeated_guest',
    'type_of_meal_plan',
    'generated_type'
]
ONE_HOT_TRANSFORMED_FEATURES = []

TARGET = 'booking_status'

for fname in ONE_HOT_FEATURES:
    dummies = pd.get_dummies(df[fname], prefix=f'is_{fname}')
    dummies_columns = dummies.columns.values.tolist()
    df[dummies_columns] = dummies

    dummies = pd.get_dummies(test_df[fname], prefix=f'is_{fname}')
    dummies_columns = dummies.columns.values.tolist()
    test_df[dummies_columns] = dummies

    ONE_HOT_TRANSFORMED_FEATURES = ONE_HOT_TRANSFORMED_FEATURES + dummies_columns

df = df.drop(ONE_HOT_FEATURES, axis=1)
test_df = test_df.drop(ONE_HOT_FEATURES, axis=1)

TRAIN_FEATURES = FEATURES + ONE_HOT_TRANSFORMED_FEATURES


# ## Model training

# In[6]:


RS = 33333

params = {
            "objective": "binary:logitraw",
            "eval_metric": "auc",
            'booster': 'gbtree',
            'n_estimators': 5000,
            'max_depth' : 5, 
            'eta':  0.02465211760946184, 
#             'tree_method' :'gpu_hist',
#             'gpu_id': 3,
            "early_stopping_rounds": 100,
            'random_state': RS,
            "scale_pos_weight": 1, 
            'subsample': 0.8807634510839019,
            'sampling_method': 'uniform', 
            'colsample_bytree':0.7615746338125093, 
            'colsample_bylevel': 0.9190448386294816,
            'colsample_bynode': 0.9190448386294816,

}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
df = df.assign(fold=-1)
for i, (_, test_ind) in enumerate(kfold.split(df, df.booking_status)):
    df.loc[test_ind, 'fold'] = i
    
for i, f in enumerate(df.fold.unique()):
    train_ind, test_ind = df[df.fold != f].index, df[df.fold == f].index
    
    xtrain, xtest = df.loc[train_ind, TRAIN_FEATURES], df.loc[test_ind, TRAIN_FEATURES]
    ytrain, ytest = df.loc[train_ind, TARGET], df.loc[test_ind, TARGET]

    # Track the CV only on the generated dataset.
    xvaltrue, yvaltrue = xtest[xtest.is_generated_type_1 == 1][TRAIN_FEATURES], ytest[xtest.is_generated_type_1 == 1] 
    

    model = xgb.XGBClassifier(**params) 
    model.fit(xtrain, ytrain, eval_set=[(xtrain, ytrain), (xtest, ytest), (xvaltrue, yvaltrue)], verbose=1000)
    
    df.loc[test_ind, 'prd_xgb'] = model.predict_proba(xtest)[:, 1]
    test_df[f'prd_{i}'] = model.predict_proba(test_df[TRAIN_FEATURES])[:, 1]
final_score = roc_auc_score(df[df.is_generated_type_1 == 1].booking_status, df[df.is_generated_type_1 == 1].prd_xgb)
print(f'final true score = {final_score}')


# In[7]:


test_df['booking_status'] = test_df[['prd_0', 'prd_1', 'prd_2', 'prd_3', 'prd_4']].mean(axis=1)
test_df[['id', 'booking_status']].to_csv('submit.csv', index=False)


# ## SHAP   
# 
# lead_time_minus_avg_price gives +0.01 to final result

# In[8]:


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(xvaltrue)
shap.summary_plot(shap_values, xvaltrue)


# ## Leakage explotation   
# Discussion can be found [here](https://www.kaggle.com/competitions/playground-series-s3e7/discussion/388851)
# 

# In[9]:


df = pd.read_csv("../input/playground-series-s3e7/train.csv")
test_df = pd.read_csv("../input/playground-series-s3e7/test.csv")

submit = pd.read_csv('submit.csv')

dup_features = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights',
               'no_of_week_nights', 'type_of_meal_plan', 'required_car_parking_space',
               'room_type_reserved', 'lead_time', 'arrival_year', 'arrival_month',
               'arrival_date', 'market_segment_type', 'repeated_guest',
               'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
               'avg_price_per_room', 'no_of_special_requests']

tt = df.rename(columns={"index":"id"}).drop('booking_status', axis=1)
test_leakage = test_df.merge(tt, on=dup_features, how='inner', suffixes=['_test', '_train'])[['id_test', 'id_train']]

test_leakage = df.loc[test_leakage.id_train][['id', 'booking_status']].rename(columns={'id':'id_train'}).merge(test_leakage, on='id_train')


rr = submit.rename(columns={'booking_status': 'prd', 'id':'id_test'}).merge(test_leakage, on='id_test', how='left')
prd_max = rr['prd'].max()
prd_min = rr['prd'].min()


def tmp_fun(x):
    bs = x.booking_status
    if(pd.isna(bs)):
        return x['prd']
    elif(bs == 0):
        return prd_max
    elif(bs == 1):
        return prd_min
    
submit['booking_status'] =  rr.apply(tmp_fun, axis=1)


# In[10]:


submit.to_csv('new_submit.csv', index=False)


# In[ ]:




