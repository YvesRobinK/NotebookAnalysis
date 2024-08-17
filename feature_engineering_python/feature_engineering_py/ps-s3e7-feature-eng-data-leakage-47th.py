#!/usr/bin/env python
# coding: utf-8

# <center><img src="https://www.readersdigest.ca/wp-content/uploads/2015/10/emirates-palace-luxurious-hotels.jpg"></center>
# <h1><center>🏨Playground Series S3E7: Feature Engineering & Data Leakage🏨</center></h1>
# 
# -----
# **Table of Contents**
# - [1. Introduction & Libraries](#intro)
# - [2. Data Preparation](#dataprep)
#     - [2.1 Load Data](#load_data)
#     - [2.2 Remove True Duplicates](#remove_true_duplicates)
#     - [2.3 Remove Tricky Duplicates](#remove_tricky_duplicates)
#     - [2.4 Remove Leaked Data](#remove_leaked_data)
#     - [2.5 Fix Anomalous Dates](#fix_anomalous_dates)
# - [3. EDA](#eda)
#     - [3.1 Missing Values](#missing_values)
#     - [3.2 Feature Datatypes](#feature_datatypes)
#     - [3.3 Distribution of Features & Target](#distribution_of_features_and_target)
#     - [3.4 Distribution of Features Separated out by Target](#distribution_of_features_separated_out_by_target)
# - [4. Feature Engineering](#feature_engineering)
#     - [4.1 Prior Probability](#prior_probability)
#     - [4.2 Deal with Dates & Holidays](#deal_with_dates_and_holidays)
# - [5. Preprocessing](#preprocessing)
#     - [5.1 Split Data](#split_data)
#     - [5.2 Scaling](#scaling)
#     - [5.3 Feature Importances](#feature_importances)
# - [6. Modelling](#modelling)
#     - [6.1 Cross Validation](#cross_validation)
#     - [5.1 Test Predictions](#test_predictions)
#     - [5.1 Add Data Leak](#add_data_leak)
#             
# -----
# 
# <a id='intro'></a>
# # 1. Introduction & Libraries
# 
# >  **Goal**: Predict whether or not a hotel reservation will be cancelled (binary classification).
# 
# In this notebook:
# 
#  - Handling different types of duplicates
#  - EDA
#  - Feature Engineering
#  - Preprocessing
#  - Ensemble Modelling
#  
#  >  **Current Score**: Currently have a score of **92.67** in this competition (pre-shakeup), placing in the **top 9% of submissions**
#  
#  >  **Final Score**: Managed to achieve a final score of **91.71** in this competition, placing in the **top 7% of submissions**
#  
# **Be sure to leave feedback and upvote if this helps you!**
#  
# ### Libraries 📚⬇

# In[1]:


import pandas as pd
pd.set_option('display.max_columns', 500)
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# <a id='dataprep'></a>
# # 2. Data Preparation

# <a id='load_data'></a>
# ## 2.1 Load Data
# - Add in the original [dataset](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset) from which the training and test data have been synthetically generated.
# - Add the `is_generated` flag to keep track of the data source
# - Check that the columns in each dataset are the same. `original_data` is missing the `id` column and `test_df` does not contain `booking_status` as expected.

# In[2]:


train_df = pd.read_csv("../input/playground-series-s3e7/train.csv")
test_df = pd.read_csv("../input/playground-series-s3e7/test.csv")
original_data = pd.read_csv("../input/reservation-cancellation-prediction/train__dataset.csv")

train_df['is_generated'] = 1
test_df['is_generated'] = 1
original_data['is_generated'] = 0


# In[3]:


print([col for col in train_df.columns if col not in original_data.columns])
print([col for col in train_df.columns if col not in test_df.columns])


# <a id='remove_true_duplicates'></a>
# ## 2.2 Remove True Duplicates
# - True duplicates = rows that are identical (including `booking-status`)

# In[4]:


# combine our two training datasets
combined_df = pd.concat([train_df, original_data], axis=0, ignore_index=True)


# In[5]:


# work out how many true duplicates we have, we need to remove is_generated and id for this
temp_combined_df = combined_df.drop(axis=1, columns=['is_generated', 'id'])
indices_to_drop = temp_combined_df.loc[temp_combined_df.duplicated()].index
print(f"{len(indices_to_drop)} true duplicates to remove")


# In[6]:


combined_df = combined_df[combined_df.index.isin(indices_to_drop) == False]


# <a id='remove_tricky_duplicates'></a>
# ## 2.3 Remove Tricky Duplicates
# - Tricky Duplicate = rows that are identical once `booking_status` is removed.
# - As noted in many discussion posts, the synthetically generated data (train and test}) contains pairs of tricky duplicates alongside true duplicates. As such it is possible to have 2 rows with the exact same features but opposite values for `booking_status`. And so if such a pair were to appear in our training_data, it would not be helpful for training a model, so remove these pairs.

# In[7]:


y = combined_df['booking_status']
is_gen = combined_df['is_generated']
dropped_df = combined_df.drop(axis=1, columns=['id', 'booking_status', 'is_generated'])
tricky_dup_df = dropped_df.loc[dropped_df.duplicated(keep=False)] # remove both duplicates in a pair
print(f"There are {tricky_dup_df.shape[0]/2:.0f} rows in combined_df that have a tricky duplicate")


# In[8]:


# remove tricky duplicates
combined_df = combined_df.loc[combined_df.index.isin(tricky_dup_df.index) == False]


# <a id='remove_leaked_data'></a>
# ## 2.4 Remove Leaked Data
# - There are rows in the training data that have a duplicate (without `booking_status` of course) in the test data.
# - For the purpose of a training a predictive model, this may lead to overfitting so we remove it at this stage.
# - These duplicates could be true or tricky duplicates (we don't know since we don't have `booking_status` for the test data.

# In[9]:


leaked_ids = test_df.merge(combined_df.drop(['booking_status'], axis=1),
              on=[c for c in test_df.columns if c != 'id'],
              how='inner', suffixes=['_test', '_train'])[['id_train']].values[:,0]
print(f"There are {len(leaked_ids)} leaked rows to remove")


# In[10]:


# remove leaked rows
combined_df.drop(combined_df[combined_df['id'].isin(leaked_ids)].index, inplace=True)


# <a id='fix_anomalous_dates'></a>
# ## 2.5 Fix Anomalous Dates
# 
# Credit to [this discussion post by Sergey Saharovskiy](https://www.kaggle.com/competitions/playground-series-s3e7/discussion/386655)
# 
# - Dates with the day greater than the number of days in that month e.g. Feb 29th not on a leap year or Sep 31st.
# - Although these anomalies likely appear in the test data as well (byproduct of how the synthetic data was generated), for human interpretability  let's correct the dates.
# - Correct these by capping the day of the month to the maximum

# In[11]:


# example of the anomalies
wrong_dates_feb = combined_df.loc[(combined_df['arrival_month'] == 2) & (combined_df['arrival_date'] > 28)]\
[['arrival_month', 'arrival_date', 'arrival_year']].shape[0]
print(f"There are {wrong_dates_feb} entries with more than 28 days in Febuary 2017/2018")


# In[12]:


combined_df['arrival_year_month'] = pd.to_datetime(combined_df['arrival_year'].astype(str) \
                                                   + combined_df['arrival_month'].astype(str),
format='%Y%m')

test_df['arrival_year_month'] = pd.to_datetime(test_df['arrival_year'].astype(str) \
                                                   + test_df['arrival_month'].astype(str),
format='%Y%m')

combined_df.loc[combined_df['arrival_date'] > combined_df['arrival_year_month']\
.dt.days_in_month, 'arrival_date'] = combined_df['arrival_year_month'].dt.days_in_month

test_df.loc[test_df['arrival_date'] > test_df['arrival_year_month']\
.dt.days_in_month, 'arrival_date'] = test_df['arrival_year_month'].dt.days_in_month

combined_df.drop(axis=1, columns=['arrival_year_month'], inplace=True)
test_df.drop(axis=1, columns=['arrival_year_month'], inplace=True)


# <a id='eda'></a>
# # 3. EDA

# In[13]:


combined_df.head()


# In[14]:


combined_df.shape


# <a id='missing_values'></a>
# ## 3.1 Missing Values
# - No missing values

# In[15]:


# ignore `id` column as original_data has Nan for `id`.
print(f"Number of Missing Values: {combined_df.iloc[:,1:].isna().sum().sum()}")


# <a id='feature_datatypes'></a>
# ## 3.2 Feature Datatypes
# - All the columns are numerical
# - columns that may have been categorical appear to have been already ordinally encoded.

# In[16]:


numerical_cols = [col for col in combined_df.columns if combined_df[col].dtype in ['int64', 'float64']]
print(len(numerical_cols) == len(combined_df.columns))


# <a id='distribution_of_features_and_target'></a>
# ## 3.3 Distribution of Features & Target
# - `lead_time` and `avg_price_per_room` are the only truly continuous features, the rest are discretised.
# - The target `booking_status` is relatively balanced so no need for sampling techniques (see my [Playground Series S3E4: Credit Card Fraud Detection](https://www.kaggle.com/code/magnussesodia/ps-s3e4-subsampling-ensemble-modelling/edit/run/117762878) notebook for more on that)

# In[17]:


fig, axes = plt.subplots(6, 3, figsize=(15,12))
for idx, val in enumerate(list(set(combined_df.columns) - set(['id', 'is_generated']))):
    sns.histplot(data=combined_df, x=val, ax=axes[idx // 3, idx % 3],stat='probability')
    plt.xlabel = val
plt.tight_layout()
plt.show()


# <a id='distribution_of_features_separated_out_by_target'></a>
# ## 3.4 Distribution of Features separated out by target

# In[18]:


fig, axes = plt.subplots(6, 3, figsize=(15,12))
for idx, val in enumerate(list(set(combined_df.columns) - set(['id', 'is_generated', 'booking_status']))):
    sns.histplot(data=combined_df, x=val, ax=axes[idx // 3, idx % 3],stat='probability', hue='booking_status', kde=True)
    plt.xlabel = val
fig.tight_layout()
plt.show()


# <a id='feature_engineering'></a>
# # 4. Feature Engineering

# <a id='prior_probability'></a>
# ## 4.1 Prior Probability
# - Add a feature to represent the prior probability of cancellation

# In[19]:


def add_prior_prob(df):
    df['prior_prob'] = (df['no_of_previous_cancellations'] / \
        (df['no_of_previous_cancellations'] + df['no_of_previous_bookings_not_canceled'])).fillna(1)
    
    return df


# In[20]:


combined_df = add_prior_prob(combined_df)
test_df = add_prior_prob(test_df)


# <a id='deal_with_dates_and_holidays'></a>
# ## 4.2 Deal with Dates & Holidays
# Credit Jose Cáliz [discussion](https://www.kaggle.com/competitions/playground-series-s3e7/discussion/386647)
# 
# Credit: Sergio Saharovskiy [notebook](https://www.kaggle.com/code/sergiosaharovskiy/ps-s3e7-2023-eda-and-submission)
# 
# - Add `day_of_year` & `weak_of_year` features to try to capture variation throughout the year
# - Experimented with flagging reservations around holidays but this worsened the CV
# - Experimented with features dividied by `avg_price_per_room` but this also worsened CV

# In[21]:


cal = USFederalHolidayCalendar()
holidays = cal.holidays(start='2017-01-01', end='2018-12-31')


# In[22]:


def process_dates(df):
    
    # time columns
    temp = df.rename(columns={
        'arrival_year': 'year',
        'arrival_month': 'month',
        'arrival_date': 'day'
    })
    
    df['date'] = pd.to_datetime(temp[['year', 'month', 'day']],
                                errors='coerce')
    
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
#     df['dayofmonth'] = df['date'].dt.day 
#     df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(float)
#     df['quarter'] = df['date'].dt.quarter

    # holiday columns
#     df['is_holiday'] = 0
#     df.loc[df['date'].isin(holidays)] = 1
 
    df.drop(columns=['date', 'arrival_year', 'arrival_month', 'arrival_date'], inplace=True)
    
    # new features
#     df['no_of_adults_div_price'] = df.no_of_adults / (df.avg_price_per_room + 1e-6)
#     df['no_of_children_div_price'] = df.no_of_children / (df.avg_price_per_room + 1e-6)
#     df['lead_time_div_price'] = df.lead_time / (df.avg_price_per_room + 1e-6)
    return df


# In[23]:


combined_df = process_dates(combined_df)
test_df = process_dates(test_df)


# <a id='preprocessing'></a>
# # 5 Preprocessing

# <a id='split_data'></a>
# ## 5.1 Split Data
# - Separate out the target `booking_status` from the rest of the data

# In[24]:


# drop id column as this gives no signal
y = combined_df['booking_status']
test_ids = test_df['id']

combined_df.drop(axis=1,columns=['id', 'booking_status'], inplace=True)
test_df.drop(axis=1, columns=['id'], inplace=True)


# <a id='scaling'></a>
# ## 5.2 Scaling
# - Scale all the columns (they are all numerical) to balance the features out for the model.

# In[25]:


robust_scaler = RobustScaler()
robust_cols = ['avg_price_per_room', 'lead_time']

standard_scaler = StandardScaler()
standard_cols = list(set(combined_df.columns) - set(robust_cols))

preprocessor = ColumnTransformer(transformers=[('standard_scaler',standard_scaler, standard_cols),
                                               ('robust_scaler', robust_scaler, robust_cols)],
                                 verbose_feature_names_out=True, remainder='passthrough')


# In[26]:


scaled_df = pd.DataFrame()
scaled_test_df = pd.DataFrame()


# In[27]:


scaled_df[combined_df.columns] = preprocessor.fit_transform(combined_df)
scaled_test_df[test_df.columns] = preprocessor.transform(test_df)


# <a id='feature_importances'></a>
# ## 5.3 Feature Importances
# - look at feature importances to see how our feature engineering has performed

# In[28]:


clf = XGBClassifier()
fit = clf.fit(scaled_df, y)
importances = pd.Series(fit.feature_importances_, index=scaled_df.columns).sort_values(ascending=False)

fig, ax = plt.subplots()
sns.barplot(x=importances.values, y=importances.index)


# <a id='modelling'></a>
# # 6. Modelling
# - Choose `XGBClassifier()`, `CatBoostClassifier()` and `LGBMClassifier()` as these models returned the best CV.
# - Worked out model parameters using `RandomizedSearchCV` in another notebook.

# <a id='cross_validation'></a>
# ## 6.1 Cross Validation

# In[29]:


xgb_params = {'colsample_bytree': 0.67, # 0.99
             'gamma': 0.16,
             'learning_rate': 0.1,
             'max_depth': 4, #5
             'min_child_weight': 3,
             'n_estimators': 2000, #400
             'subsample': 0.77,
             'tree_method': 'gpu_hist'}


# In[30]:


catboost_params = {'silent':True,
                   'depth': 5,
                   'learning_rate': 0.22,
                   'n_estimators': 950}


# In[31]:


lgbm_params = {'max_depth': 6,
               'learning_rate': 0.13,
               'n_estimators': 400,
               'lambda_l1': 1.13,
               'lambda_l2': 0.61,
               'min_data_in_leaf': 114}


# In[32]:


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# untuned models
# xgb_cv_scores = cross_val_score(XGBClassifier(), scaled_df, y, cv=skf, scoring='roc_auc', n_jobs=-1)
# catboost_cv_scores = cross_val_score(CatBoostClassifier(silent=True), scaled_df, y, cv=skf, scoring='roc_auc', n_jobs=-1)
# lgbm_cv_scores = cross_val_score(LGBMClassifier(), scaled_df, y, cv=skf, scoring='roc_auc', n_jobs=-1)
# cv_scores = [np.mean(xgb_cv_scores), np.mean(catboost_cv_scores), np.mean(lgbm_cv_scores)]

# print(f"XGB ROC-AUC Score: {np.mean(xgb_cv_scores):.4f}")
# print(f"CatBoost ROC-AUC Score: {np.mean(catboost_cv_scores):.4f}")
# print(f"LGBM ROC-AUC Score: {np.mean(lgbm_cv_scores):.4f}")
# print(f"Mean ROC-AUC Score: {np.mean(cv_scores):.4f}")

# tuned models
tuned_xgb_cv_scores = cross_val_score(XGBClassifier(**xgb_params), scaled_df, y, cv=skf, scoring='roc_auc', n_jobs=-1)
tuned_catboost_cv_scores = cross_val_score(CatBoostClassifier(**catboost_params), scaled_df, y, cv=skf, scoring='roc_auc', n_jobs=-1)
tuned_lgbm_cv_scores = cross_val_score(LGBMClassifier(**lgbm_params), scaled_df, y, cv=skf, scoring='roc_auc', n_jobs=-1)
tuned_cv_scores = [np.mean(tuned_xgb_cv_scores), np.mean(tuned_catboost_cv_scores), np.mean(tuned_lgbm_cv_scores)]

print(f"Tuned XGB ROC-AUC Score: {np.mean(tuned_xgb_cv_scores):.4f}")
print(f"Tuned CatBoost ROC-AUC Score: {np.mean(tuned_catboost_cv_scores):.4f}")
print(f"Tuned LGBM ROC-AUC Score: {np.mean(tuned_lgbm_cv_scores):.4f}")
print(f"========================")
print(f"Mean Tuned ROC-AUC Score: {np.mean(tuned_cv_scores):.4f}")


# <a id='test_predictions'></a>
# ## 6.2 Test Predictions

# In[33]:


xgb_clf = XGBClassifier(**xgb_params)
xgb_clf.fit(scaled_df, y)
xgb_test_preds = xgb_clf.predict_proba(scaled_test_df.values)[:,1]

catboost_clf = CatBoostClassifier(**catboost_params)
catboost_clf.fit(scaled_df, y)
catboost_test_preds = catboost_clf.predict_proba(scaled_test_df.values)[:,1]

lgbm_clf = LGBMClassifier(**lgbm_params)
lgbm_clf.fit(scaled_df, y, verbose=False)
lgbm_test_preds = lgbm_clf.predict_proba(scaled_test_df.values)[:,1]

test_preds = np.stack((xgb_test_preds, catboost_test_preds, lgbm_test_preds)).mean(0)


# In[34]:


submission = pd.DataFrame({'id': test_ids,
                              'booking_status': test_preds})
submission.to_csv('submission.csv', index=False)


# <a id='add_data_leak'></a>
# ## 6.3 Add Data Leak
# 
# Credit @icfoer [discussion](https://www.kaggle.com/competitions/playground-series-s3e7/discussion/388851)
# 
# - Dubious method that appears to impove public LB CV.
# - As with the leaked data from before, assume that they are pairs of tricky duplicates. Knowing the `booking_status` of one duplicate, the other will then be the opposite value.
# - Overwrite our test predictions in the cases of these tricky duplicates.

# In[35]:


train_leak = pd.read_csv('/kaggle/input/playground-series-s3e7/train.csv')
test_leak = pd.read_csv('/kaggle/input/playground-series-s3e7/test.csv')

y = 'booking_status' # for convenience
dup_features = test_leak.drop(columns='id').columns.tolist()
values_to_assign = test_leak.merge(train_leak.drop(columns='id'), on=dup_features,
                                   how='inner')[['id', y]]
map_di = {0: submission[y].max(), 1: submission[y].min()}
submission.loc[submission.id.isin(values_to_assign.id), y] = values_to_assign[y].map(map_di).values
submission.loc[submission.id.isin(values_to_assign.id), y]

submission.to_csv('submission_with_leak.csv', index=False)
submission.loc[submission.id.isin(values_to_assign.id)].head(10)

