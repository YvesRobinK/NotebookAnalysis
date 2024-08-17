#!/usr/bin/env python
# coding: utf-8

# <a id="table"></a>
# <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Table of Contents</h1>
# 
# [1. Notebook Versions](#Notebook-Versions)
# 
# [2. Loading Libraries](#Loading-Libraries)
# 
# [3. Reading Data Files](#Reading-Data-Files)
# 
# [4. Data Description](#Data-Description)
# 
# [5. Data Exploration](#Data-Exploration)
# 
# [6. LightGBM Modeling](#LightGBM-Modeling)
# 
# [7. LightGBM Modeling Optuna](#LightGBM-Modeling-Optuna)
# 
# [8. Feature Engineering](#Feature-Engineering)
# 
# [9. Feature Selection](#Feature-Selection)
# 
# [10. Exploiting Data Leakage: Approach 1](#Exploiting-Data-Leakage:-Approach-1)
# 
# [11. Exploiting Data Leakage: Approach2](#Exploiting-Data-Leakage:-Approach-2)
# 
# [12. Model Comparison](#Model-Comparison)
# 
# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Notebook Versions</h1>
# 
# 1. Version 1 (02/13/2023)
#     * EDA 
#     * LightGBM Modeling
#     
# 2. Version 2 (02/14/2023)
#     * Data description added
#     * LightGBM optuna hyper-parameters
#     
# 3. Version 3 (02/15/2023)
#     * Feature engineering added
#     * Model comparison added
#     
# 4. Version 4 (02/22/2023)
#     * Feature engineering updated
#     * Feature selection added
#     * Exploiting data leakage approach 1 added
#     
# 5. Version 4 (02/27/2023)
#     * Exploiting data leakage approach 2 added
#  
# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Loading Libraries</h1>    

# In[1]:


import pandas as pd; pd.set_option('display.max_columns', 100)
import numpy as np

from tqdm import tqdm

from functools import partial
import scipy as sp

import matplotlib.pyplot as plt; plt.style.use('ggplot')
import seaborn as sns
import plotly.express as px

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay, cohen_kappa_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier

import optuna


# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Reading Data Files</h1> 

# In[2]:


train = pd.read_csv('../input/playground-series-s3e7/train.csv')
test = pd.read_csv('../input/playground-series-s3e7/test.csv')
submission = pd.read_csv('../input/playground-series-s3e7/sample_submission.csv')

print('The dimession of the train dataset is:', train.shape)
print('The dimession of the test dataset is:', test.shape)


# In[3]:


train.info()


# In[4]:


test.info()


# In[5]:


train.head()


# In[6]:


train.describe()


# In[7]:


test.head()


# In[8]:


test.describe()


# > <div class="alert alert-block alert-info">
# <b>💡</b> There are no missing values neither in the train nor test datasets. Also, by a quick eye-ball comparison of the summary statistics of the train and test datasets, they seem to have similar distributions. 
# </div>
# 
# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Data Description</h1>
# 
# This is a synthetic dataset generated from the [Reservation Cancellation Prediction](https://www.kaggle.com/datasets/gauravduttakiit/reservation-cancellation-prediction) dataset. These are the descriptions of the variables in this dataset:
# 
# <ul>
#     <li> id: unique identifier of each booking. </li>
#     <li> no_of_adults: number of adults. </li>
#     <li> no_of_children: number of Children. </li>
#     <li> no_of_weekend_nights: number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel. </li>
#     <li> no_of_week_nights: number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel.  </li>
#     <li> type_of_meal_plan: type of meal plan booked by the customer. </li>
#     <li> required_car_parking_space: does the customer require a car parking space? (0 - No, 1- Yes). </li>
#     <li> room_type_reserved: type of room reserved by the customer. The values are ciphered (encoded) by INN Hotels. </li>
#     <li> lead_time: number of days between the date of booking and the arrival date. </li>
#     <li> arrival_year: year of arrival date. </li>
#     <li> arrival_month: month of arrival date. </li>
#     <li> arrival_date: date of the month. </li>
#     <li> market_segment_type: market segment designation. </li>
#     <li> repeated_guest: Is the customer a repeated guest? (0 - No, 1- Yes). </li>
#     <li> no_of_previous_cancellations: number of previous bookings that were canceled by the customer prior to the current booking. </li>
#     <li> no_of_previous_bookings_not_canceled: number of previous bookings not canceled by the customer prior to the current booking. </li>
#     <li> avg_price_per_room: average price per day of the reservation; prices of the rooms are dynamic. (in euros). </li>
#     <li> no_of_special_requests: total number of special requests made by the customer (e.g. high floor, view from the room, etc). </li>
#     <li> booking_status: flag indicating if the booking was canceled or not. </li>
# </ul>
# 
# 
# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Data Exploration</h1> 

# In[9]:


round(100*train['booking_status'].value_counts() / train.shape[0], 2) 


# In[10]:


sns.countplot(x = 'booking_status', hue = 'booking_status', data = train);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the above chart, we see that the data is not that imbalanced (close to be 50-50).</div>

# In[11]:


fig, axes = plt.subplots(1, 2, figsize = (20, 7))

sns.countplot(ax = axes[0], x = 'no_of_adults', hue = 'booking_status', data = train);
sns.countplot(ax = axes[1], x = 'no_of_children', hue = 'booking_status', data = train);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the above left panel, we see that there a couple of reservations with 0 adults, which is suspicious.</div>

# In[12]:


fig, axes = plt.subplots(1, 2, figsize = (20, 7))

sns.countplot(ax = axes[0], x = 'no_of_weekend_nights', hue = 'booking_status', data = train);
sns.countplot(ax = axes[1], x = 'no_of_week_nights', hue = 'booking_status', data = train);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the above left panel, we see that there are a few reservations with three or more weekend nights, which is not possible. From the above the right panel, we see that there are a few reservations with 6 of more week nights. </div>

# In[13]:


fig, axes = plt.subplots(1, 2, figsize = (20, 7))

sns.countplot(ax = axes[0], x = 'type_of_meal_plan', hue = 'booking_status', data = train);
sns.countplot(ax = axes[1], x = 'required_car_parking_space', hue = 'booking_status', data = train);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the above left panel, we see that the number of cancellations is higher when the type of meal plan is 1. </div>

# In[14]:


fig, axes = plt.subplots(1, 2, figsize = (20, 7))

sns.countplot(ax = axes[0], x = 'room_type_reserved', hue = 'booking_status', data = train);
sns.countplot(ax = axes[1], x = 'market_segment_type', hue = 'booking_status', data = train);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the above right panel, customers from market segment type 1 are more likely to cancel their reservations. </div>

# In[15]:


fig, axes = plt.subplots(1, 2, figsize = (20, 7))

sns.countplot(ax = axes[0], x = 'repeated_guest', hue = 'booking_status', data = train);
sns.countplot(ax = axes[1], x = 'no_of_previous_cancellations', hue = 'booking_status', data = train);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the above left panel, repeated guests didn't cancel their reservations. </div>

# In[16]:


fig, axes = plt.subplots(1, 2, figsize = (20, 7))

sns.countplot(ax = axes[0], x = 'no_of_previous_bookings_not_canceled', hue = 'booking_status', data = train);
sns.countplot(ax = axes[1], x = 'no_of_special_requests', hue = 'booking_status', data = train);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the above left panel, we see that customers, who didn't cancel one or more of their previous booking, didn't cancel theur reservations. </div>

# In[17]:


fig, axes = plt.subplots(1, 2, figsize = (22, 8))

sns.boxplot(ax = axes[0], x = 'booking_status', y = 'lead_time', hue = 'booking_status', data = train)
sns.boxplot(ax = axes[1], x = 'booking_status', y = 'avg_price_per_room', hue = 'booking_status', data = train);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the above left panel, we see that, on average, as the number of days between the date of booking and the arrival date increases, the reservation is more likely to be cancelled. From the above right panel, there is a slighlty difference when comparing the average price room between reservations that were cancelled and reservations that were not cancelled. </div>

# In[18]:


fig, axes = plt.subplots(1, 2, figsize = (22, 8))

sns.countplot(ax = axes[0], x = 'arrival_month', hue = 'booking_status', data = train)
sns.countplot(ax = axes[1], x = 'arrival_year', hue = 'booking_status', data = train);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the above left panel, we see that in August reservations are more likely to be cancelled than any other month. </div>

# In[19]:


plt.figure(figsize = (15, 8))

sns.countplot(x = 'arrival_date', hue = 'booking_status', data = train);


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the above chart, we see that in the midle of the month reservations are more likeliy to be cancelled. </div>
# 
# <div class="alert alert-block alert-info">
# <b>💡 Insights from EDA:</b><br> 
# <ul>
#     <li> The dataset is not that imbalanced (~60%-~40%). </li>
#     <li> There a couple of reservations with 0 adults, which is supicious. </li>
#     <li> There are a few reservations with three or more weekend nights. </li>
#     <li> There are a few reservations with 6 of more week nights. </li>
#     <li> Customers from market segment type 1 are more likely to cancel their reservations.  </li>
#     <li> Repeated guests didn't cancel their reservations. </li>
#     <li> On average, as the number of days between the date of booking and the arrival date increases, the reservation is more likely to be cancelled. </li>
#     <li> August reservations are more likely to be cancelled than any other month. </li>
#     <li> Reservations in the middle of the month are more likely to be cancelled. </li>
# </ul>
# </div>

# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">LightGBM Modeling</h1> 

# In[20]:


train_lgb = train.copy()
test_lgb = test.copy()

X = train_lgb.drop(columns = ['id', 'booking_status'], axis = 1)
Y = train_lgb['booking_status']

test_lgb = test_lgb.drop(columns = ['id'], axis = 1)

cv_scores, roc_auc_scores = list(), list()
preds = list()

## Running 5 times CV
for i in range(5):
    
    skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)
    
    for train_ix, test_ix in skf.split(X, Y):
        
        ## Splitting the data 
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]
    
        ## Building RF model
        lgb_md = LGBMClassifier(n_estimators = 1000,
                                max_depth = 7,
                                learning_rate = 0.01,
                                num_leaves = 20,
                                lambda_l1 = 3,
                                lambda_l2 = 3,
                                bagging_fraction = 0.7,
                                feature_fraction = 0.7).fit(X_train, Y_train)
        
        ## Predicting on X_test and test
        lgb_pred_1 = lgb_md.predict_proba(X_test)[:, 1]
        lgb_pred_2 = lgb_md.predict_proba(test_lgb)[:, 1]
        
        ## Computing roc-auc score
        roc_auc_scores.append(roc_auc_score(Y_test, lgb_pred_1))
        preds.append(lgb_pred_2)
        
    cv_scores.append(np.mean(roc_auc_scores))

lgb_cv_score = np.mean(cv_scores)    
print('The roc-auc score over 5-folds (run 5 times) is:', lgb_cv_score)


# In[21]:


lgb_cv_score = np.mean(cv_scores)    
print('The roc-auc score over 5-folds (run 5 times) is:', lgb_cv_score)


# In[22]:


## Building model in the entire train dataset
lgb_md = LGBMClassifier(n_estimators = 1000,
                        max_depth = 7,
                        learning_rate = 0.01,
                        num_leaves = 20,
                        lambda_l1 = 3,
                        lambda_l2 = 3,
                        bagging_fraction = 0.7,
                        feature_fraction = 0.7).fit(X, Y)

train_preds = pd.DataFrame({'booking_status': Y, 'booking_status_pred': lgb_md.predict_proba(X)[:, 1]})
train_preds.head()


# In[23]:


RocCurveDisplay.from_predictions(train_preds['booking_status'], train_preds['booking_status_pred'])
plt.title('LightGBM ROC-AUC Curve on Train')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate');


# In[24]:


lgb_preds_train = train_preds['booking_status_pred']
lgb_preds_test = pd.DataFrame(preds).apply(np.mean, axis = 0)

data1 = pd.DataFrame({'Predicted Likelihood': lgb_preds_train})
data2 = pd.DataFrame({'Predicted Likelihood': lgb_preds_test})

fig, axes = plt.subplots(1, 2, figsize = (20, 8), dpi = 200)
sns.histplot(data = data1, kde = True, stat = 'density', ax = axes[0]).set(title = 'LightGBM Predicted Likelihood in Train')
sns.histplot(data = data2, kde = True, stat = 'density', ax = axes[1]).set(title = 'LightGBM Predicted Likelihood in Test')
plt.show();


# In[25]:


submission['booking_status'] = lgb_preds_test
submission.head()


# In[26]:


submission.to_csv('Baseline_LightGBM_submission.csv', index = False)


# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">LightGBM Modeling Optuna</h1>
# 
# In this section, we re-run the LightGBM model using optimal hyper-parameters from the optuna framework.

# In[27]:


train_lgb = train.copy()
test_lgb = test.copy()

X = train_lgb.drop(columns = ['id', 'booking_status'], axis = 1)
Y = train_lgb['booking_status']

test_lgb = test_lgb.drop(columns = ['id'], axis = 1)


# In[28]:


cv_scores, roc_auc_scores = list(), list()
preds = list()

## Running 5 times CV
for i in range(5):
    
    skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)
    
    for train_ix, test_ix in skf.split(X, Y):
        
        ## Splitting the data 
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]
    
        ## Building RF model
        lgb_md = LGBMClassifier(n_estimators = 7880,
                                max_depth = 10,
                                learning_rate = 0.009689077948120357,
                                num_leaves = 20,
                                lambda_l1 = 1.2185030034603348,
                                lambda_l2 = 1.8144608820124146,
                                bagging_fraction = 0.6383698341881532,
                                feature_fraction = 0.5452440168291733).fit(X_train, Y_train)
        
        ## Predicting on X_test and test
        lgb_pred_1 = lgb_md.predict_proba(X_test)[:, 1]
        lgb_pred_2 = lgb_md.predict_proba(test_lgb)[:, 1]
        
        ## Computing roc-auc score
        roc_auc_scores.append(roc_auc_score(Y_test, lgb_pred_1))
        preds.append(lgb_pred_2)
        
    cv_scores.append(np.mean(roc_auc_scores))


# In[29]:


lgb_cv_score = np.mean(cv_scores)    
print('The oof roc-auc score over 5-folds (run 5 times) is:', lgb_cv_score)


# In[30]:


## Building model in the entire train dataset
lgb_md = LGBMClassifier(n_estimators = 7880,
                        max_depth = 10,
                        learning_rate = 0.009689077948120357,
                        num_leaves = 20,
                        lambda_l1 = 1.2185030034603348,
                        lambda_l2 = 1.8144608820124146,
                        bagging_fraction = 0.6383698341881532,
                        feature_fraction = 0.5452440168291733).fit(X, Y)

train_preds = pd.DataFrame({'booking_status': Y, 'booking_status_pred': lgb_md.predict_proba(X)[:, 1]})
train_preds.head()


# In[31]:


RocCurveDisplay.from_predictions(train_preds['booking_status'], train_preds['booking_status_pred'])
plt.title('LightGBM ROC-AUC Curve on Train')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate');


# In[32]:


lgb_preds_train = train_preds['booking_status_pred']
lgb_preds_test = pd.DataFrame(preds).apply(np.mean, axis = 0)

data1 = pd.DataFrame({'Predicted Likelihood': lgb_preds_train})
data2 = pd.DataFrame({'Predicted Likelihood': lgb_preds_test})

fig, axes = plt.subplots(1, 2, figsize = (20, 8), dpi = 200)
sns.histplot(data = data1, kde = True, stat = 'density', ax = axes[0]).set(title = 'LightGBM Predicted Likelihood (Optuna) in Train')
sns.histplot(data = data2, kde = True, stat = 'density', ax = axes[1]).set(title = 'LightGBM Predicted Likelihood (Optuna) in Test')
plt.show();


# In[33]:


submission['booking_status'] = lgb_preds_test
submission.head()


# In[34]:


submission.to_csv('Baseline_LightGBM_Optuna_submission.csv', index = False)


# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Feature Engineering</h1>
# 
# In this section, we engineer feature and evalute their effect in the overall model performance. First, we start by taking a close look at `avg_price_per_room`. Let's visualize the distribution of this feature in the train and test datasets.

# In[35]:


fig, axes = plt.subplots(1, 2, figsize = (20, 8), dpi = 200)
sns.histplot(data = train['avg_price_per_room'], kde = True, stat = 'density', ax = axes[0]).set(title = 'Average Price Distribution in Train')
sns.histplot(data = test['avg_price_per_room'], kde = True, stat = 'density', ax = axes[1]).set(title = 'Average Price Distribution in Test')
plt.show();


# > <div class="alert alert-block alert-info">
# <b>💡</b> From the above chart, we see that there are several reservations with `avg_price_per_room` equal to 0. Also, there are reservations with small values for `avg_price_per_room`. </div>
# 
# According to the description in the original dataset, `avg_price_per_room` represents average price per day of the reservation; prices of the rooms are dynamic. (in euros). A potential reason for those small values could be that those reservations are based on promotions. Let's take a close look at the cancellation rate of reservations with low `avg_price_per_room`. I used 30 euros as threshold to identify reservations with low `avg_price_per_room`.

# In[36]:


print('There are ', train[train['avg_price_per_room'] < 30].shape[0], ' reservations with avg_price_per_room less than 30 in the train dataset')
print('There are ', test[test['avg_price_per_room'] < 30].shape[0], ' reservations with avg_price_per_room less than 30 in the test dataset')


# Now, let's take a look at their cancellation rate in the train dataset.

# In[37]:


train[train['avg_price_per_room'] < 30]['booking_status'].value_counts() / train[train['avg_price_per_room'] < 30].shape[0]


# From the above, we see that cancellation rate is less than 6%, which is small when compared to cancellation rates of reservation with `avg_price_per_room` greater than 30 (~39%). We next proceed to engineer a flag feature that identifies reservations with low `avg_price_per_room` as follows:

# In[38]:


train['low_price_flag'] = np.where(train['avg_price_per_room'] < 30, 1, 0)
test['low_price_flag'] = np.where(test['avg_price_per_room'] < 30, 1, 0)


# Next, we re-run the CV routine to see if there is an improvement in the CV-score.

# In[39]:


train_lgb = train.copy()
test_lgb = test.copy()

X = train_lgb.drop(columns = ['id', 'booking_status'], axis = 1)
Y = train_lgb['booking_status']

test_lgb = test_lgb.drop(columns = ['id'], axis = 1)

cv_scores, roc_auc_scores = list(), list()
preds = list() 

## Running 5 times CV
for i in range(5):
    
    skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)
    
    for train_ix, test_ix in skf.split(X, Y):
        
        ## Splitting the data 
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]
    
        ## Building RF model
        lgb_md = LGBMClassifier(n_estimators = 9691,
                                max_depth = 6,
                                learning_rate = 0.0462768823884295,
                                num_leaves = 55,
                                lambda_l1 = 3.4343007400000185,
                                lambda_l2 = 2.4712185408144425,
                                bagging_fraction = 0.6704473114789922,
                                feature_fraction = 0.29190676287540945).fit(X_train, Y_train)
        
        ## Predicting on X_test and test
        lgb_pred_1 = lgb_md.predict_proba(X_test)[:, 1]
        lgb_pred_2 = lgb_md.predict_proba(test_lgb)[:, 1]
        
        ## Computing roc-auc score
        roc_auc_scores.append(roc_auc_score(Y_test, lgb_pred_1))
        preds.append(lgb_pred_2)
        
    cv_scores.append(np.mean(roc_auc_scores))


# In[40]:


lgb_cv_score = np.mean(cv_scores)    
print('The oof roc-auc score over 5-folds (run 5 times) is:', lgb_cv_score)


# From the above result, we see the CV score had a small decrease. We next visualize the importance of each of the features under the LightGBM model.

# In[41]:


lgb_preds_test = pd.DataFrame(preds).apply(np.mean, axis = 0)
submission['booking_status'] = lgb_preds_test
submission.head()


# In[42]:


submission.to_csv('Baseline_LightGBM_Optuna_low_price_flag_submission.csv', index = False)


# We next proceed to engineered more features in the train and test datasets as follows:

# In[43]:


## Fixing dates (https://www.kaggle.com/competitions/playground-series-s3e7/discussion/386655)
train['arrival_year_month'] = pd.to_datetime(train['arrival_year'].astype(str) + train['arrival_month'].astype(str), format = '%Y%m')
test['arrival_year_month'] = pd.to_datetime(test['arrival_year'].astype(str) + test['arrival_month'].astype(str), format = '%Y%m')

train.loc[train.arrival_date > train.arrival_year_month.dt.days_in_month, 'arrival_date'] = train.arrival_year_month.dt.days_in_month
test.loc[test.arrival_date > test.arrival_year_month.dt.days_in_month, 'arrival_date'] = test.arrival_year_month.dt.days_in_month

train.drop(columns = 'arrival_year_month', inplace = True)
test.drop(columns = 'arrival_year_month', inplace = True)

train['segment_0'] = np.where(train['market_segment_type'] == 0, 1, 0)
train['segment_1'] = np.where(train['market_segment_type'] == 1, 1, 0)
train['total_guests'] = train['no_of_adults'] + train['no_of_children']
train['stay_length'] = train['no_of_weekend_nights'] + train['no_of_week_nights']
train['stay_during_weekend'] = np.where(train['no_of_weekend_nights'] > 0, 1, 0)
train['quarter_1'] = np.where(train['arrival_month'] <= 3, 1, 0)
train['quarter_2'] = np.where(((train['arrival_month'] >= 4) & (train['arrival_month'] <= 6)), 1, 0)
train['quarter_3'] = np.where(((train['arrival_month'] >= 7) & (train['arrival_month'] <= 9)), 1, 0)
train['quarter_4'] = np.where(train['arrival_month'] >= 10, 1, 0)
train['segment_0_feature_1'] = np.where(((train['market_segment_type'] == 0) & (train['lead_time'] <= 90)), 1, 0)
train['segment_0_feature_2'] = np.where(((train['market_segment_type'] == 0) & (train['avg_price_per_room'] > 98)), 1, 0)
train['segment_1_feature_1'] = np.where(((train['market_segment_type'] == 1) & (train['no_of_special_requests'] == 0)), 1, 0)
train['segment_1_feature_2'] = np.where(((train['market_segment_type'] == 1) & (train['no_of_special_requests'] > 0) & (train['lead_time'] <= 150)), 1, 0)
train['segment_0_year_flag'] = np.where(((train['market_segment_type'] == 0) & (train['arrival_year'] == 2018)), 1, 0)
train['segment_1_year_flag'] = np.where(((train['market_segment_type'] == 1) & (train['arrival_year'] == 2018)), 1, 0)
train['price_lead_time_flag'] = np.where(((train['avg_price_per_room'] > 100) & (train['lead_time'] > 150)), 1, 0)

test['low_price_flag'] = np.where(test['avg_price_per_room'] < 30, 1, 0)
test['segment_0'] = np.where(test['market_segment_type'] == 0, 1, 0)
test['segment_1'] = np.where(test['market_segment_type'] == 1, 1, 0)
test['total_guests'] = test['no_of_adults'] + test['no_of_children']
test['stay_length'] = test['no_of_weekend_nights'] + test['no_of_week_nights']
test['stay_during_weekend'] = np.where(test['no_of_weekend_nights'] > 0, 1, 0)
test['quarter_1'] = np.where(test['arrival_month'] <= 3, 1, 0)
test['quarter_2'] = np.where(((test['arrival_month'] >= 4) & (test['arrival_month'] <= 6)), 1, 0)
test['quarter_3'] = np.where(((test['arrival_month'] >= 7) & (test['arrival_month'] <= 9)), 1, 0)
test['quarter_4'] = np.where(test['arrival_month'] >= 10, 1, 0)
test['segment_0_feature_1'] = np.where(((test['market_segment_type'] == 0) & (test['lead_time'] <= 90)), 1, 0)
test['segment_0_feature_2'] = np.where(((test['market_segment_type'] == 0) & (test['avg_price_per_room'] > 98)), 1, 0)
test['segment_1_feature_1'] = np.where(((test['market_segment_type'] == 1) & (test['no_of_special_requests'] == 0)), 1, 0)
test['segment_1_feature_2'] = np.where(((test['market_segment_type'] == 1) & (test['no_of_special_requests'] > 0) & (test['lead_time'] <= 150)), 1, 0)
test['segment_0_year_flag'] = np.where(((test['market_segment_type'] == 0) & (test['arrival_year'] == 2018)), 1, 0)
test['segment_1_year_flag'] = np.where(((test['market_segment_type'] == 1) & (test['arrival_year'] == 2018)), 1, 0)
test['price_lead_time_flag'] = np.where(((test['avg_price_per_room'] > 100) & (test['lead_time'] > 150)), 1, 0)


# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Feature Selection</h1>
# 
# In the previous section, we engineered some featueres. In this section, we will indentify the important features under the LightGBM model and the recurvise feature elimination framework (for more info on the RFE see [here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html)).

# In[44]:


X = train.drop(columns = ['id', 'low_price_flag', 'no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 'booking_status'], axis = 1)
Y = train['booking_status']

## Running RFECV multiple times
RFE_results = list()

for i in tqdm(range(0, 10)):
    
    auto_feature_selection = RFECV(estimator = LGBMClassifier(), step = 1, min_features_to_select = 2, cv = 5, scoring = 'roc_auc').fit(X, Y)
    
    ## Extracting and storing features to be selected
    RFE_results.append(auto_feature_selection.support_)

## Changing to data-frame
RFE_results = pd.DataFrame(RFE_results)
RFE_results.columns = X.columns

## Computing the percentage of time features are flagged as important
RFE_results = 100*RFE_results.apply(np.sum, axis = 0) / RFE_results.shape[0]
RFE_results


# In[45]:


## Identifying features with a percentage score > 80%
features_to_select = RFE_results.index[RFE_results > 80].tolist()
features_to_select


# We next proceed to build another LightGBM model with the above features. 

# In[46]:


X = train[features_to_select]
Y = train['booking_status']

test_lgb = test[features_to_select]

class Objective:

    def __init__(self, seed):
        # Hold this implementation specific arguments as the fields of the class.
        self.seed = seed

    def __call__(self, trial):
        
        ## Parameters to be evaluated
        param = dict(objective = 'binary',
                     metric = 'auc',
                     tree_method = 'gbdt', 
                     n_estimators = trial.suggest_int('n_estimators', 300, 10000),
                     learning_rate = trial.suggest_float('learning_rate', 0.001, 1, log = True),
                     max_depth = trial.suggest_int('max_depth', 3, 12),
                     lambda_l1 = trial.suggest_float('lambda_l1', 0.01, 10.0, log = True),
                     lambda_l2 = trial.suggest_float('lambda_l2', 0.01, 10.0, log = True),
                     num_leaves = trial.suggest_int('num_leaves', 2, 100),
                     bagging_fraction = trial.suggest_float('bagging_fraction', 0.2, 0.9),
                     feature_fraction = trial.suggest_float('feature_fraction', 0.2, 0.9),
                     device = 'gpu'
                    )

        scores = []
        
        skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = self.seed)

        for train_idx, valid_idx in skf.split(X, Y):

            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            Y_train , Y_valid = Y.iloc[train_idx] , Y.iloc[valid_idx]

            model = LGBMClassifier(**param).fit(X_train, Y_train)

            preds_valid = model.predict_proba(X_valid)[:, 1]

            score = roc_auc_score(Y_valid, preds_valid)
            scores.append(score)

        return np.mean(scores)
    
## Defining SEED and Trials
SEED = 42
N_TRIALS = 50

# Execute an optimization
study = optuna.create_study(direction = 'maximize')
study.optimize(Objective(SEED), n_trials = N_TRIALS)


# In[47]:


print('The best trial auc score is: ', study.best_trial.values)
print('The best hyper-parameter combination is: ', study.best_trial.params)


# We next re-run the CV procedure with the optimal hyper-parameters from Optuna.

# In[48]:


cv_scores, roc_auc_scores = list(), list()
preds = list() 

## Running 5 times CV
for i in range(5):
    
    skf = StratifiedKFold(n_splits = 5, random_state = SEED, shuffle = True)
    
    for train_ix, test_ix in skf.split(X, Y):
        
        ## Splitting the data 
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]
    
        ## Building RF model
        lgb_md = LGBMClassifier(**study.best_trial.params, 
                                device = 'gpu').fit(X_train, Y_train)
        
        ## Predicting on X_test and test
        lgb_pred_1 = lgb_md.predict_proba(X_test)[:, 1]
        lgb_pred_2 = lgb_md.predict_proba(test_lgb)[:, 1]
        
        ## Computing roc-auc score
        roc_auc_scores.append(roc_auc_score(Y_test, lgb_pred_1))
        preds.append(lgb_pred_2)
        
    cv_scores.append(np.mean(roc_auc_scores))


# In[49]:


lgb_cv_score = np.mean(cv_scores)    
print('The oof roc-auc score over 5-folds (run 5 times) is:', lgb_cv_score)


# In[50]:


lgb_preds_test = pd.DataFrame(preds).apply(np.mean, axis = 0)
submission['booking_status'] = lgb_preds_test
submission.head()


# In[51]:


submission.to_csv('LightGBM_Optuna_FE_FS_submission.csv', index = False)


# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Exploiting Data Leakage: Approach 1</h1>
# 
# In this section, we exploit the data leakage identify by a few kagglers. For more infor see this [post](https://www.kaggle.com/competitions/playground-series-s3e7/discussion/388851). The first approach to identify the duplicated observations from the train and test and change the classes to the opposite in the test dataset. That is, if a duplicated row in the train dataset has a `booking_status = 1` in the train, we assign `booking_status = 0` in the test dataset.

# In[52]:


train_dup = train.drop(columns = 'id', axis = 1)
test_dup = test.copy()

duplicates = pd.merge(train, test, on = train_dup.columns.tolist()[0:17])
duplicates = duplicates[['id_y', 'booking_status']]
duplicates.columns = ['id', 'booking_status']

duplicates['flip_booking_status'] = 1 - duplicates['booking_status']
duplicates.drop(columns = 'booking_status', axis = 1, inplace = True)

submission = pd.merge(submission, duplicates, on = 'id', how = 'left')
submission.head()


# In[53]:


submission['booking_status'] = np.where(np.isnan(submission['flip_booking_status']), submission['booking_status'], submission['flip_booking_status'])
submission.drop(columns = 'flip_booking_status', axis = 1, inplace = True)
submission.head()


# In[54]:


submission.to_csv('LightGBM_Optuna_FE_FS_submission_data_leakage.csv', index = False)


# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Exploiting Data Leakage: Approach 2</h1>
# 
# In this section, we exploit the data leakeage in a slighlty different way from approach 1. In this case, we first remove the duplicate observations from the `train` dataset and then proceed to build and tune the model. After that, we predict on the `test` dataset (without duplicated observations). 

# In[55]:


## Identifying and separating duplicates from non-duplicates
train_dup = train.copy()
test_dup = test.copy()

duplicates = pd.merge(train, test, on = train_dup.columns.tolist()[1:18])
train_dup_ids = duplicates['id_x'].tolist()
test_dup_ids = duplicates['id_y'].tolist()

train_clean = train[~np.isin(train['id'], train_dup_ids)].reset_index(drop = True)
train_dup = train[np.isin(train['id'], train_dup_ids)].reset_index(drop = True)

test_clean = test[~np.isin(test['id'], test_dup_ids)].reset_index(drop = True)
test_dup = test[np.isin(test['id'], test_dup_ids)].reset_index(drop = True)

X = train_clean.drop(columns = ['id', 'low_price_flag', 'no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 'booking_status'], axis = 1)
Y = train_clean['booking_status']

## Running features selection multiple times
RFE_results = list()

for i in tqdm(range(0, 10)):
    
    auto_feature_selection = RFECV(estimator = LGBMClassifier(), step = 1, min_features_to_select = 2, cv = 5, scoring = 'roc_auc').fit(X, Y)
    
    ## Extracting and storing features to be selected
    RFE_results.append(auto_feature_selection.support_)

## Changing to data-frame
RFE_results = pd.DataFrame(RFE_results)
RFE_results.columns = X.columns

## Computing the percentage of time features are flagged as important
RFE_results = 100*RFE_results.apply(np.sum, axis = 0) / RFE_results.shape[0]

## Identifying features with a percentage score > 80%
features_to_select = RFE_results.index[RFE_results > 80].tolist()
print(features_to_select)


# We next proceed to tune the model with the Optuna framework.

# In[56]:


X = train_clean[features_to_select]
Y = train_clean['booking_status']

test_lgb = test_clean[features_to_select]

class Objective:

    def __init__(self, seed):
        # Hold this implementation specific arguments as the fields of the class.
        self.seed = seed

    def __call__(self, trial):
        
        ## Parameters to be evaluated
        param = dict(objective = 'binary',
                     metric = 'auc',
                     boosting_type = 'gbdt', 
                     n_estimators = trial.suggest_int('n_estimators', 300, 10000),
                     learning_rate = trial.suggest_float('learning_rate', 0.001, 1, log = True),
                     max_depth = trial.suggest_int('max_depth', 3, 12),
                     lambda_l1 = trial.suggest_float('lambda_l1', 0.01, 10.0, log = True),
                     lambda_l2 = trial.suggest_float('lambda_l2', 0.01, 10.0, log = True),
                     num_leaves = trial.suggest_int('num_leaves', 2, 100),
                     bagging_fraction = trial.suggest_float('bagging_fraction', 0.2, 0.9),
                     feature_fraction = trial.suggest_float('feature_fraction', 0.2, 0.9),
                     device = 'gpu'
                    )

        scores = []
        
        skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = self.seed)

        for train_idx, valid_idx in skf.split(X, Y):

            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            Y_train , Y_valid = Y.iloc[train_idx] , Y.iloc[valid_idx]

            model = LGBMClassifier(**param).fit(X_train, Y_train)

            preds_valid = model.predict_proba(X_valid)[:, 1]

            score = roc_auc_score(Y_valid, preds_valid)
            scores.append(score)

        return np.mean(scores)
    
## Defining SEED and Trials
SEED = 42
N_TRIALS = 50

# Execute an optimization
study = optuna.create_study(direction = 'maximize')
study.optimize(Objective(SEED), n_trials = N_TRIALS)


# Next, we run the cross-validation procedure.

# In[57]:


cv_scores, roc_auc_scores = list(), list()
preds = list() 

## Running 5 times CV
for i in range(5):
    
    skf = StratifiedKFold(n_splits = 5, random_state = SEED, shuffle = True)
    
    for train_ix, test_ix in skf.split(X, Y):
        
        ## Splitting the data 
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]
    
        ## Building RF model
        lgb_md = LGBMClassifier(**study.best_trial.params, 
                                device = 'gpu').fit(X_train, Y_train)
        
        ## Predicting on X_test and test
        lgb_pred_1 = lgb_md.predict_proba(X_test)[:, 1]
        lgb_pred_2 = lgb_md.predict_proba(test_lgb)[:, 1]
        
        ## Computing roc-auc score
        roc_auc_scores.append(roc_auc_score(Y_test, lgb_pred_1))
        preds.append(lgb_pred_2)
        
    cv_scores.append(np.mean(roc_auc_scores))


# In[58]:


lgb_cv_score = np.mean(cv_scores)    
print('The oof roc-auc score over 5-folds (run 5 times) is:', lgb_cv_score)


# Finally, we put together the predictions on the test (submission file) dataset.

# In[59]:


lgb_preds_test = pd.DataFrame(preds).apply(np.mean, axis = 0)
clean_pred = pd.DataFrame({'id': test_clean['id']})
clean_pred['booking_status_clean'] = lgb_preds_test

## Flipping labels of duplicates
dup_pred = duplicates[['id_y', 'booking_status']]
dup_pred.columns = ['id', 'booking_status_dup']
dup_pred['booking_status_dup'] = 1 - dup_pred['booking_status_dup']

submission = pd.merge(submission.drop(columns = 'booking_status', axis = 1), clean_pred, on = 'id', how = 'left')
submission = pd.merge(submission, dup_pred, on = 'id', how = 'left')
submission['booking_status'] = np.where(np.isnan(submission['booking_status_clean']), submission['booking_status_dup'], submission['booking_status_clean'])
submission.drop(columns = ['booking_status_clean', 'booking_status_dup'], axis = 1, inplace = True)

submission.head()


# In[60]:


submission.to_csv('LightGBM_Optuna_FE_FS_submission_data_leakage_2.csv', index = False)


# <a id="table"></a>
# # <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Model Comparison</h1>
# 
# | Model | Features | CV | LB |
# | --- | --- | --- | --- |
# | LightGBM | all raw features | 0.8923750 | 0.90135 |
# | LightGBM (tuned) | all raw features | 0.9004299 | 0.91049 | 
# | LightGBM (tuned) | all raw features + low price flag | 0.9004093 | 0.91120 |

# # More work coming soon...
# ![work_in_progress.jpg](attachment:54294a5b-bc31-455d-9e18-70cf78cad7cf.jpg)
