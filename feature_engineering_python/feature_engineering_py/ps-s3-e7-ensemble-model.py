#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import optuna

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


train_df = pd.read_csv('/kaggle/input/playground-series-s3e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s3e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s3e7/sample_submission.csv')
addition_data = pd.read_csv('/kaggle/input/reservation-cancellation-prediction/train__dataset.csv')

train_df['is_generated'] = 1
test_df['is_generated'] = 1
addition_data['is_generated'] = 0

# train_df['is_test'] = 0
# test_df['is_test'] = 1
# addition_data['is_test'] = 0


# In[3]:


train_df.info()


# In[4]:


train_df


# In[5]:


test_df


# In[6]:


addition_data['id'] = np.arange(70168, 70168+addition_data.shape[0])
addition_data


# In[7]:


addition_data.isna().any()


# In[8]:


train_df.booking_status.hist()


# In[9]:


train_df = pd.concat([train_df, addition_data],axis=0, ignore_index=True)
train_df


# In[10]:


# train_df = train_df.drop_duplicates()


# In[11]:


df = pd.concat([train_df, test_df], axis=0)
df = df.drop('id', axis=1)
df


# In[12]:


# feat1 = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights',
#        'no_of_week_nights', 'type_of_meal_plan', 'required_car_parking_space',
#        'room_type_reserved', 'lead_time', 'arrival_year', 'arrival_month',
#        'arrival_date', 'market_segment_type', 'repeated_guest',
#        'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
#        'avg_price_per_room', 'no_of_special_requests']
# df = df.drop_duplicates(subset=feat1, keep=False)
# df = df[df.is_test == 0]


# In[13]:


# df = pd.concat([df, test_df], axis=0)
# df = df.drop(['id', 'is_test'], axis=1)
# df


# Feature engneering from this amazing notebook:
# https://www.kaggle.com/code/sergiosaharovskiy/ps-s3e7-2023-eda-and-submission#Basic-Feature-Engineering
# Please, upvote it!

# In[14]:


from pandas.tseries.holiday import USFederalHolidayCalendar

def fe(df):
        # Fix date anomalies (pd.to_datetime throws parsing error for some days, see anomalies section).
        df['arrival_year_month'] = pd.to_datetime(df['arrival_year'].astype(str) + df['arrival_month'].astype(str), format='%Y%m')
        df.loc[df.arrival_date > df.arrival_year_month.dt.days_in_month, 'arrival_date'] = df.loc[df.arrival_date > df.arrival_year_month.dt.days_in_month, 'arrival_year_month'].dt.days_in_month
        df.drop(columns='arrival_year_month', inplace=True)
        
        # Creates date features.
        df['arrival_full_date'] = (df['arrival_year'].astype(str) 
                                   + '-' + df['arrival_month'].astype(str)
                                   + '-' + df['arrival_date'].astype(str))
        df['arrival_full_date'] = pd.to_datetime(df.arrival_full_date)
#         df['arrival_week'] = df['arrival_full_date'].dt.isocalendar().week.astype(float)
#         df['arrival_dayofweek'] = df['arrival_full_date'].dt.dayofweek
#         df['arrival_quarter'] = df['arrival_full_date'].dt.quarter
#         df['arrival_dayofyear'] = df['arrival_full_date'].dt.dayofyear
        
        # Creates the season and holiday features. (also you can add holidays).
#         ['winter', 'spring', 'summer', 'fall']
#         df['season'] = df.arrival_month%12 // 3 + 1
#         cal = USFederalHolidayCalendar()
#         holidays = cal.holidays(start='2017-01-01', end='2018-12-31')
#         df['is_holiday'] = 0
#         df.loc[df.arrival_full_date.isin(holidays), 'is_holiday'] = 1
        
        # Aggregation by `season` as key and 'avg_price_per_room' as value (you can try quarters, months, etc).
#         key = 'season'
#         aggr_df = df.groupby(by=key, sort=False)['avg_price_per_room'].agg(['mean', 'std', 'min', 'max', 'sum', 'last'])
#         aggr_df = aggr_df.add_prefix('avg_price_per_room_')
#         df = df.merge(aggr_df.reset_index(), on=key, how='left')
        
        # Interaction between the correlated features and also lead time.
        df['no_of_adults_div_price'] = df.no_of_adults / (df.avg_price_per_room + 1e-6)
        df['no_of_children_div_price'] = df.no_of_children / (df.avg_price_per_room + 1e-6)
        df['lead_time_div_price'] = df.lead_time / (df.avg_price_per_room + 1e-6)
        df.drop(columns=['arrival_full_date'], inplace=True)
        return df


# In[15]:


df = fe(df)
# df = df.drop(['no_of_children', 'no_of_previous_cancellations'], axis=1)
df


# In[16]:


features = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights',
       'no_of_week_nights', 'type_of_meal_plan', 'required_car_parking_space',
       'room_type_reserved', 'lead_time', 'arrival_year', 'arrival_month',
       'arrival_date', 'market_segment_type', 'repeated_guest',
       'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
       'avg_price_per_room', 'no_of_special_requests',
       'is_generated', 'no_of_adults_div_price', 'no_of_children_div_price',
       'lead_time_div_price']


# In[17]:


cat_features = [
    'required_car_parking_space',
    'market_segment_type',
    'room_type_reserved', 
    'repeated_guest',
    'type_of_meal_plan',
    'is_generated'
]


# In[18]:


# for feat in cat_features:
#     dummies = pd.get_dummies(df[feat], prefix=f'is_{feat}')
#     dummies_columns = dummies.columns.values.tolist()
#     df[dummies_columns] = dummies
    
# df = df.drop(cat_features, axis=1)

# df


# In[19]:


# from category_encoders import WOEEncoder

# woe = WOEEncoder(drop_invariant=True, randomized = True)
# for col in cat_features:
#     df[col] = df[col].astype(str)
# woe.fit(df[features][:-len(test_df)], df['booking_status'][:-len(test_df)], cols = cat_features)
# X = woe.transform(df[features])
# X['booking_status'] = df['booking_status']
# df = X

# df


# In[20]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

y = df['booking_status']
df = df.drop(['booking_status'], axis=1)

df[df.columns] = scaler.fit_transform(df[df.columns])


# In[21]:


train_df = df.iloc[:-len(test_df),:]
train_df['booking_status'] = y[:-len(test_df)]
test_df = df.iloc[-len(test_df):,:].reset_index(drop=True)

# oversample = train_df[train_df['Class']==1]
# undersample = train_df[train_df['Class']==0]

X = train_df.drop(['booking_status'], axis=1)
y = train_df.booking_status.astype('int')
X_test = test_df


# In[22]:


from sklearn.model_selection import train_test_split

# X_train, X_test1, y_train, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)

# X, y = X_train, y_train
# X1, y1 = X_test1, y_test1


# In[23]:


from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold

n_folds = 6 #6 -
repeats = 5 #5 - 


# In[24]:


# !pip install scikit-optimize

# import scikit-optimize as skopt


# In[25]:


# from skopt import BayesSearchCV
# import catboost

# gpu_params = {'task_type' : "GPU", 'devices' : '0:1'}
# cbc_params = {'iterations': 5000, #164,
# #               'max_depth': 5, #10,
# #               'learning_rate': 0.05, #0.1, 
#               'verbose': 0,
#               'eval_metric': 'AUC',
#               'loss_function': 'Logloss',
# #               **gpu_params
#              }
# model = catboost.CatBoostClassifier(**cbc_params)

# opt = BayesSearchCV(
#     model,
#     {
#         'learning_rate': (0.001, 0.05),
#         'max_depth': (1, 10),  # integer valued parameter
#     },
#     n_iter=30,
#     cv=2
# )

# opt.fit(X_train, y_train)

# print("val. score: %s" % opt.best_score_)
# print("test score: %s" % opt.score(X_test1, y_test1))
# print("best params: %s" % str(opt.best_params_))


# In[26]:


# import matplotlib.pyplot as plt
# from skopt.plots import plot_objective, plot_histogram, plot_evaluations

# # _ = plot_objective(opt.optimizer_results_[0])

# _ = plot_evaluations(opt.optimizer_results_[0])

# plt.show()


# In[27]:


import catboost
from sklearn.utils.class_weight import compute_sample_weight

MAX_ITER = 15000
PATIENCE = 500 #50 #1000
DISPLAY_FREQ = 100

modelsCBC = []
predsCBC = []

# k_fold = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)
k_fold = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=repeats, random_state=42) 

gpu_params = {'task_type' : "GPU", 'devices' : '0:1'}
cbc_params = {'iterations': 5000, #164,
              'max_depth': 6, #5, #10,
              'learning_rate': 0.052, #0.05, #0.1, 
              'verbose': 100,
              'eval_metric': 'AUC',
              'loss_function': 'Logloss',
#               **gpu_params
             }

for train_index, test_index in k_fold.split(X, y):
    X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    
    model = catboost.CatBoostClassifier(**cbc_params)
#     model = catboost.CatBoostRegressor(**MODEL_PARAMS)
    
    model.fit(X=X_train, y=y_train,
          eval_set=[(X_valid, y_valid)],
          early_stopping_rounds = PATIENCE,
#           metric_period = DISPLAY_FREQ
         )
    modelsCBC.append(model)
    predsCBC.append(model.predict_proba(X_test))
#     predsCBC.append(model.predict(X_test))


# In[28]:


feature_importance =  [modelsCBC[x].feature_importances_ for x in range(n_folds*repeats)]
feature_importance = np.average(feature_importance,axis=0)
feature_importance


# In[29]:


import matplotlib.pyplot as plt

# feature_importance = modelsCBC[0].feature_importances_
sorted_idx = np.argsort(feature_importance)
fig = plt.figure(figsize=(20, 15))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
plt.title('Feature Importance')


# In[30]:


from xgboost import XGBClassifier, XGBRegressor

k_fold = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=repeats, random_state=42) 

modelsXBC = []
predsXBC = []

PATIENCE = 50 #500

xgbc_params = {'n_estimators': 3000, #54, 
               'max_depth': 5, #10, #3, 
               'learning_rate': 0.05, 
               'subsample': 0.568355005569169, 
               'eval_metric'     : 'auc',
               'objective'       : 'binary:logistic',
#                'tree_method': 'gpu_hist'
              }

for train_index, test_index in k_fold.split(X, y):
    X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    
    model = XGBClassifier(**xgbc_params)
#     model = XGBRegressor(**MODEL_PARAMS)
    
    model.fit(X=X_train, y=y_train,
#               sample_weight=weights,
          eval_set=[(X_valid, y_valid)],
          early_stopping_rounds = PATIENCE,
          verbose = 100
         )
    modelsXBC.append(model)
    predsXBC.append(model.predict_proba(X_test))


# In[31]:


import lightgbm as lgbm

k_fold = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=repeats, random_state=42) 

modelsLBC = []
predsLBC = []

PATIENCE = 50 #200

classes = np.unique(y)
weights = compute_sample_weight("balanced", y=y)
class_weights = dict(zip(classes, weights))

gpu_params = {'device' : "gpu"}
gbc_params = {'n_estimators': 5000, 
              'max_depth': 3, #4, #5, #10, <--
              'num_leaves': 8,
              'learning_rate': 0.07, #0.05, #0.1, 
              'subsample': 0.8501198417003352,
              'lambda_l1': 3,
              'lambda_l2': 3,
              'bagging_fraction': 0.8,
              'feature_fraction': 0.8,
              'objective': 'binary',
              'metric': 'auc',
#               **gpu_params
             }

for train_index, test_index in k_fold.split(X, y):
    X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    
    model = lgbm.LGBMClassifier(**gbc_params)
#     model = lgbm.LGBMRegressor(**MODEL_PARAMS)
    
    model.fit(X=X_train, y=y_train,
          eval_set=[(X_valid, y_valid)],
          early_stopping_rounds = PATIENCE,
          verbose = 100
         )
    modelsLBC.append(model)
    predsLBC.append(model.predict_proba(X_test))


# In[32]:


PATIENCE = 100

modelsCB = []
predsCB = []

k_fold = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=repeats, random_state=42) 

gpu_params = {'task_type' : "GPU", 'devices' : '0:1'}
cbr_params = {'iterations': 2000, 
              'max_depth': 7,
              'learning_rate': 0.05, #0.1, # 0.05, 
              'verbose': 100,
#               **gpu_params
             }

for train_index, test_index in k_fold.split(X, y):
    X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    
    model = catboost.CatBoostRegressor(**cbr_params)
    
    model.fit(X=X_train, y=y_train,
          eval_set=[(X_valid, y_valid)],
          early_stopping_rounds = PATIENCE,
         )
    modelsCB.append(model)
    predsCB.append(model.predict(X_test))


# In[33]:


from xgboost import XGBClassifier, XGBRegressor

k_fold = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=repeats, random_state=42) 

modelsXB = []
predsXB = []

PATIENCE = 50

xgbr_params = {'n_estimators': 2000, 
               'max_depth': 5,
               'learning_rate': 0.05, 
               'subsample': 0.8291850469303983,
#                'tree_method': 'gpu_hist'
              }


for train_index, test_index in k_fold.split(X, y):
    X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    
    model = XGBRegressor(**xgbr_params)
    
    model.fit(X=X_train, y=y_train,
          eval_set=[(X_valid, y_valid)],
          early_stopping_rounds = PATIENCE,
          verbose = 100
         )
    modelsXB.append(model)
    predsXB.append(model.predict(X_test))


# In[34]:


import lightgbm as lgbm
k_fold = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=repeats, random_state=42) 

modelsLB = []
predsLB = []

PATIENCE = 50

gpu_params = {'device' : "gpu"}
lgbr_params = {'n_estimators': 10000, 
               'metric': 'rmse',
               'max_depth': 8, #10, #8
               'num_leaves': 8,
               'learning_rate': 0.1, #0.05, # 0.1
               'subsample': 0.944652288803578,
               'lambda_l1': 3,
               'lambda_l2': 3,
               'bagging_fraction': 0.8, 
               'feature_fraction': 0.8,
#                **gpu_params
              }

for train_index, test_index in k_fold.split(X, y):
    X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    
    model = lgbm.LGBMRegressor(**lgbr_params)
    
    model.fit(X=X_train, y=y_train,
          eval_set=[(X_valid, y_valid)],
          early_stopping_rounds = PATIENCE,
          verbose = 100
         )
    modelsLB.append(model)
    predsLB.append(model.predict(X_test))


# In[35]:


from sklearn.linear_model import LassoCV, Lasso

k_fold = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=2*repeats, random_state=42)  

modelsLR = []
predsLR = []

MODEL_PARAMS = {
                       'precompute': "auto",
                       'fit_intercept': True,
                       'normalize': False,
                       'max_iter': 10000,
                       'verbose': False,
                       'eps': 0.0001,
                       'cv': 5,
                       'n_alphas': 1000,
                       'n_jobs': 8,
#                        'tol': 0.0001
}


for train_index, test_index in k_fold.split(X, y):
    X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    
    model = LassoCV(**MODEL_PARAMS)
    
    model.fit(X=X_train, y=y_train,
#           eval_set=[(X_valid, y_valid)],
         )
    
    modelsLR.append(model)
    predsLR.append(model.predict(X_test))


# In[36]:


from sklearn.metrics import roc_auc_score, precision_score, cohen_kappa_score

 
def coef_objective(trial):
    a = trial.suggest_float('a', 0, 1)
    b = trial.suggest_float('b', 0, 1)
    c = trial.suggest_float('c', 0, 1)
    d = trial.suggest_float('d', 0, 1)
    e = trial.suggest_float('e', 0, 1)
    f = trial.suggest_float('f', 0, 1)
    g = trial.suggest_float('g', 0, 1)

#     X = X1
#     y = y1
    
    preds_eval = []
    for model in modelsCBC:
        preds_eval.append(model.predict_proba(X))
    
    resCBC = np.average(np.array(preds_eval),axis=0)[:, 1]
    
    preds_eval = []
    for model in modelsXBC:
        preds_eval.append(model.predict_proba(X))
    
    resXBC = np.average(np.array(preds_eval),axis=0)[:, 1]
    
    preds_eval = []
    for model in modelsLBC:
        preds_eval.append(model.predict_proba(X))
    
    resLBC = np.average(np.array(preds_eval),axis=0)[:, 1]
    
    preds_eval = []
    for model in modelsCB:
        preds_eval.append(model.predict(X))
    
    resCB = np.average(np.array(preds_eval),axis=0)
    
    preds_eval = []
    for model in modelsXB:
        preds_eval.append(model.predict(X))
    
    resXB = np.average(np.array(preds_eval),axis=0)
    
    preds_eval = []
    for model in modelsLB:
        preds_eval.append(model.predict(X))
    
    resLB = np.average(np.array(preds_eval),axis=0)
    
    preds_eval = []
    for model in modelsLR:
        preds_eval.append(model.predict(X))
    
    resLR = np.average(np.array(preds_eval),axis=0)
    
    
    res1 =  (resCBC * a + resXBC * b + resLBC * c + resCB * d + resXB * e + resLB * f + resLR * g)/(a + b + c + d + e + f + g)
    
    res = roc_auc_score(y, res1)

    return res

study = optuna.create_study(direction= 'maximize')
# study.optimize(coef_objective, n_trials= 100)


# In[37]:


# study.best_params


# In[38]:


# a = study.best_params['a']
# b = study.best_params['b']
# c = study.best_params['c']
# d = study.best_params['d']
# e = study.best_params['e']
# f = study.best_params['f']
# g = study.best_params['g']

# sum_coef = a + b + c + d + e + f + g 
# a = a / sum_coef
# b = b / sum_coef
# c = c / sum_coef
# d = d / sum_coef
# e = e / sum_coef
# f = f / sum_coef
# g = g / sum_coef

# a, b, c, d, e, f, g


# In[39]:


# a = 0.5155893321929897
# b = 0.37502083547425785
# c = 0.10938983233275243

# a = 1/7
# b = 1/7
# c = 1/7
# d = 1/7
# e = 1/7
# f = 1/7
# g = 1/7

# no fe
# a = 0.3684464981582445
# b = 0.004585031672493245
# c = 0.2925567160027242
# d = 0.13748170575667054
# e = 0.19039203361124
# f = 0.00046850646601934637
# g = 0.006069508332608441

# fe_full
# a = 0.14124297684031703
# b = 0.1871351260219016
# c = 0.2605818224474295
# d = 0.062108235284312134
# e = 0.23701521522852273
# f = 0.10212910045887301
# g = 0.009787523718643953

# fe_1
# a = 0.20524289112918595
# b = 0.10059359652492328
# c = 0.2841825597505599
# d = 0.07193809633352445
# e = 0.32521870793805235
# f = 0.0010696292862930317
# g = 0.011754519037461091

# #no fe_3 

a = 0.47638751902534116
b = 0.004824891719549555
c = 0.2973933002866936
d = 0.04650187463781135
e = 0.15383267428849448
f = 0.020325112860756125
g = 0.0007346271813538352

# fe_3+

# a = 0.27133374754938955
# b = 0.19787520176081633
# c = 0.24533540791384462
# d = 0.0718966958855005
# e = 0.16159876921816566
# f = 0.03474777307544428
# g = 0.017212404596839016

#no fe_3 - 2

# a = 0.32587792959994877
# b = 0.30558467467938666
# c = 0.007787585042345024
# d = 0.04096400139748795
# e = 0.21534167376391417
# f = 0.09802822377077806
# g = 0.0064159117461392995

#fe_3+ 

# a = 0.2942046227792033
# b = 0.25011027193132657
# c = 0.05899492325372577
# d = 0.19463132682786688
# e = 0.07347798831758359
# f = 0.11867800995692179
# g = 0.009902856933371876

# +oh

# a = 0.24551003345505282
# b = 0.241738373615494
# c = 0.18849595969550098
# d = 0.11041398246339396
# e = 0.21191018641192438
# f = 0.00010140834459551104
# g = 0.001830056014038418

# fe_3 WOE

# a = 0.4837723290969251
# b = 0.14301013534752807
# c = 0.013281487647083633
# d = 0.10810687275445123
# e = 0.20180736799006885
# f = 0.015489025779565952
# g = 0.034532781384377306


# In[40]:


predCBC = np.average(np.array(predsCBC),axis=0)[:, 1]
predXBC = np.average(np.array(predsXBC),axis=0)[:, 1]
predLBC = np.average(np.array(predsLBC),axis=0)[:, 1]
predCB = np.average(np.array(predsCB),axis=0).clip(0, 1)
predXB = np.average(np.array(predsXB),axis=0).clip(0, 1)
predLB = np.average(np.array(predsLB),axis=0).clip(0, 1)
predLR = np.average(np.array(predsLR),axis=0).clip(0, 1)


# In[41]:


predC = predCBC * a + predXBC * b + predLBC * c + predCB * d + predXB * e + predLB * f + predLR * g
# pred = np.round(predC).astype('int')
pred = predC
pred


# In[42]:


# def use_data_leakage(preds):
#     out = preds.copy()
#     feature_cols = train_df.drop(columns='booking_status').columns.to_list()
#     dup_ids_in_train = train_df[train_df.duplicated(subset=feature_cols, keep=False)].index  # shape (1124,)
#     dup_ids_in_test  = test_df[test_df.duplicated(subset=feature_cols, keep=False)].index  # shape (506,)

#     joint = pd.concat([train_df.assign(source='train'), test_df.assign(source='test')])
#     dup_ids_in_joint = joint[joint.duplicated(subset=feature_cols, keep=False)].index  # shape (3062,)

#     exploitable_indices = dup_ids_in_joint.drop(dup_ids_in_train).drop(dup_ids_in_test)
#     dups_across_joint = joint.loc[exploitable_indices].sort_values(by=feature_cols)

#     for i in range(len(dups_across_joint) // 2):
#         if dups_across_joint.iloc[2*i].source == 'train':
#             train_row = dups_across_joint.iloc[2*i]
#             test_row = dups_across_joint.iloc[2*i+1]
#         else:
#             train_row = dups_across_joint.iloc[2*i+1]
#             test_row = dups_across_joint.iloc[2*i]
#         out.loc[test_row.name] = (1 - train_row.booking_status)
#     out[dup_ids_in_test] = 0.5
#     return out


# In[43]:


submission['booking_status'] = pred
submission


# In[44]:


# submission.to_csv('submission.csv', index=False)


# In[45]:


submission.booking_status.hist()


# This trick from this notebook https://www.kaggle.com/code/sergiosaharovskiy/ps-s3e7-2023-eda-and-submission

# In[46]:


train = pd.read_csv('/kaggle/input/playground-series-s3e7/train.csv')
# addition_data = pd.read_csv('/kaggle/input/reservation-cancellation-prediction/train__dataset.csv')
# addition_data['id'] = np.arange(70168, 70168+addition_data.shape[0])
# train = pd.concat([train, addition_data],axis=0)

test = pd.read_csv('/kaggle/input/playground-series-s3e7/test.csv')

# train = train_df

y = 'booking_status'
dup_features = test.drop(columns='id').columns.tolist()
values_to_assign = test.merge(train.drop(columns='id'), on=dup_features, how='inner')[['id', y]]
map_di = {0: submission[y].max(), 1: submission[y].min()}


# In[47]:


# values_to_assign = values_to_assign.drop_duplicates(subset=['id'])
# values_to_assign


# In[48]:


submission.loc[submission.id.isin(values_to_assign.id), y] = values_to_assign[y].map(map_di).values
submission.loc[submission.id.isin(values_to_assign.id), y]

submission.to_csv('submission.csv', index=False)
submission.loc[submission.id.isin(values_to_assign.id)].head(10)


# In[49]:


submission.booking_status.hist()

