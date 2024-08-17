#!/usr/bin/env python
# coding: utf-8

# ## Inspirations
# 
# The following individuals and posts were very influential on this work.  Some code is used, some with modifications, but all are attributed here.  thanks to all of you for your inspiration, your coversation, and your code.
# 
# **Please visit these notebooks and upvote them!!**
# 
# StratifiedKFold inspired by @tilii7 who used RepeatedKFold - https://www.kaggle.com/code/tilii7/modeling-stroke-dataset-with-lasso-regression
# 
# One-hot-encoding code from @craigmthomas, with modifications - https://www.kaggle.com/code/craigmthomas/play-s3e2-eda-models#4.7---One-Hot-Encoding
# 
# Average ranking of predictions rather than average probabilities from @jcaliz - https://www.kaggle.com/competitions/playground-series-s3e2/discussion/377732  (Ulimately not used.)
# 
# Risk factor calculation from @craigmthomas - https://www.kaggle.com/code/craigmthomas/play-s3e2-eda-models#4.8---Number-of-Risk-Factors
# 
# Add stroke cases from original - I also had this idea and was pleased to see I wasn't the only one.  Code from @hikmatullahmohammadi https://www.kaggle.com/competitions/playground-series-s3e2/discussion/377875    Also demonstrated bu @dmitryuarov here:  https://www.kaggle.com/code/dmitryuarov/ps-s3e2-catboost-lasso-nn-above-0-891/notebook
# 
# NN imputation of BMI by @dmitryuarov, with modifications from https://www.kaggle.com/code/dmitryuarov/ps-s3e2-catboost-lasso-nn-above-0-891/notebook

# ## Imports

# In[1]:


import numpy as np 
import pandas as pd 

import os
import random
from collections import defaultdict
from statistics import mean
from bisect import bisect_left

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBClassifier


# ## Load data

# In[2]:


DATA_DIR = '/kaggle/input/playground-series-s3e2'
EX_DATA_DIR = '/kaggle/input/stroke-prediction-dataset'

# load datasets for this episode
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

# load the external dataset
ext_data = pd.read_csv(os.path.join(EX_DATA_DIR, 'healthcare-dataset-stroke-data.csv'))
ext_data = ext_data[ext_data['stroke'] == 1]


# ## Feature Engineering

# In[3]:


FEATURES = train.columns.to_list()[1:11]
FEATURES.append('risk_factors')
CAT_FEATURES = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
CON_FEATURES = ['age', 'avg_glucose_level', 'bmi']
TARGET = 'stroke'


# ### One-hot encoding
# 
# Code from https://www.kaggle.com/code/craigmthomas/play-s3e2-eda-models#4.7---One-Hot-Encoding by @craigmthomas.

# In[4]:


# one hot encoding of categorical features
train_ohe = train.copy()
test_ohe = test.copy()
ext_ohe = ext_data.copy()

for feature in CAT_FEATURES:
    ohe = OneHotEncoder()
    train_temp = pd.DataFrame(train_ohe[feature])
    test_temp = pd.DataFrame(test_ohe[feature])
    ext_temp = pd.DataFrame(ext_ohe[feature])
    
    merged_temp = pd.DataFrame(train_ohe[feature])
    merged_temp = merged_temp.append(pd.DataFrame(test_ohe[feature]), ignore_index = True)
    merged_temp = merged_temp.append(pd.DataFrame(ext_ohe[feature]), ignore_index = True)
   
    ohe = OneHotEncoder(sparse = False, drop = 'first')
    ohe.fit(merged_temp)
    
    new_columns = ["{}_{}_ohe".format(feature, val) for val in ohe.categories_[0][1:]]
    
    train_ohe_column = pd.DataFrame(ohe.transform(train_temp), columns = new_columns)
    test_ohe_column = pd.DataFrame(ohe.transform(test_temp), columns = new_columns)
    ext_ohe_column = pd.DataFrame(ohe.transform(ext_temp), columns = new_columns)
    
    for column in new_columns:
        train_ohe[column] = train_ohe_column[column]
        test_ohe[column] = test_ohe_column[column]
        ext_ohe[column] = ext_ohe_column[column]
        FEATURES.append(column)
    
for feature in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]:
    FEATURES.remove(feature)
    
train_ohe.drop(CAT_FEATURES, axis = 1, inplace = True)
train_ohe.drop("id", axis = 1, inplace = True)

test_ohe.drop(CAT_FEATURES, axis = 1, inplace = True)
test_ohe.drop("id", axis = 1, inplace = True)

ext_ohe.drop(CAT_FEATURES, axis = 1, inplace = True)
ext_ohe.drop('id', axis = 1, inplace = True)


# ### Impute BMI
# 
# In adding the stroke samples from the original dataset, some observations have no entry for BMI and needed imputation.
# 
# Code by @dmitryuarov, with modifications from  https://www.kaggle.com/code/dmitryuarov/ps-s3e2-catboost-lasso-nn-above-0-891/notebook

# In[ ]:


# manual evaluation of best estimates of BMI showed that age by 
# itself gave nearly the best performance.  avg_glucose_level 
# added about 1% improvement.  All others did not help and some
# made the estimate worse.

KNNCOLS = [
            'age', 
#             'gender_Male_ohe', 'gender_Other_ohe',
#             'hypertension', 
#             'heart_disease', 
            'avg_glucose_level', 
#             'work_type_Never_worked_ohe', 'work_type_Private_ohe', 'work_type_Self-employed_ohe', 'work_type_children_ohe',
#             'Residence_type_Urban_ohe', 
#             'smoking_status_formerly smoked_ohe', 'smoking_status_never smoked_ohe', 'smoking_status_smokes_ohe'
]

ext_impute = ext_ohe.copy()

knn = KNeighborsRegressor(n_neighbors = 100,
                          metric = 'minkowski',
                          n_jobs = -1)
knn.fit(ext_impute[KNNCOLS], ext_impute[TARGET])
dists, nears = knn.kneighbors(ext_impute[KNNCOLS], return_distance = True)

# calculate error on predictions
result = []
for i, n in enumerate(nears):
    n = list(n)
    
    # depending on the features used, the ith index may not be
    # in the nearest neihbors list
    # for categorical features only, sometimes all distances are zero.
    # using just the median of the whole dataset gives RMSE 6.26
    try:
        n.remove(i)
    except:
        continue

    try:
        avg_bmi = ext_impute.iloc[n]['bmi'].median()
        if( not pd.isna(ext_impute.iloc[i]['bmi']) ):
            result.append( (ext_impute.iloc[i]['bmi'] - avg_bmi)**2 )
    except:
        continue

print(f'RMSE: {round(np.mean(result) ** 0.5, 2)}')
        
result = []
for i in ext_impute.query('bmi!=bmi').index:
    result.append(round(ext_impute.iloc[nears[i]]['bmi'].median(),1))
ext_impute.loc[ext_impute.query('bmi!=bmi').index, 'bmi'] = result
ext_ohe['bmi'] = ext_impute['bmi']


# ### Compute total risk factors
# 
# Interesting idea to compute the total number of risk factors for each individual.  I don't think it actually contributed to the overall model much, however.
# 
# Code from @craigmthomas - https://www.kaggle.com/code/craigmthomas/play-s3e2-eda-models#4.8---Number-of-Risk-Factors

# In[ ]:


def feature_risk_factors(df):
    df["risk_factors"] = df[[
        "avg_glucose_level", "age", "bmi", 
        "hypertension", "heart_disease", 
        "smoking_status"
    ]].apply(
        lambda x: \
        0 + (1 if x.avg_glucose_level > 116.5 else 0) + \
        (1 if x.age > 52.5 else 0) + (1 if x.bmi > 25.5 else 0) + \
        (1 if x.hypertension == 1 else 0) + \
        (1 if x.heart_disease == 1 else 0) + \
        (1 if x.smoking_status in ["formerly smoked", "smokes"] else 0),
        axis=1
    )
    return df

train_ohe['risk_factors'] = feature_risk_factors(train)['risk_factors']
test_ohe['risk_factors'] = feature_risk_factors(test)['risk_factors']
ext_ohe['risk_factors'] = feature_risk_factors(ext_data)['risk_factors']


# ### Center and scale

# In[ ]:


for feature in CON_FEATURES:
    mu = np.mean(train_ohe[feature])
    sigma = np.std(train_ohe[feature])
    train_ohe[feature] = (train_ohe[feature] - mu) / sigma
    test_ohe[feature] = (test_ohe[feature] - mu) / sigma
    ext_ohe[feature] = (ext_ohe[feature] - mu) / sigma


# ## XGBoost Model
# 

# ### Tuning
# 
# The overall premise is quite simple:
# 
# * Do random search across various hyperparameters for XGBoost for 500 iterations.
# * Each iteration is run with 10-fold cross-validation.  Data from the original dataset are added to the training set for each cross-validation fold, but **only the synthetic dataset is used for CV calculations.**  Addition of the extra data gave me a CV and LB boost consistently, and only monitoring CV on the synthetic set helped avoid over-training to the original set.
# * At each CV fold, a prediction is made on the test set.
# * Average CV across 10-folds is monitored and the models are orderd by 10-fold CV average AUC.
# * Predictiosn for the test set are made at each fold.
# * Models with the highest AUCs are used to create an ensemble predictions later.

# In[ ]:


# define tuning parameters
n_estimators_values = [10, 25, 50, 100, 150, 200, 250, 300]
eta_values = [ v / 10 for v in range(10) ]
max_depth_values = [2, 4, 6, 8, 10]
subsample_values = [0.25, 0.50, 0.75, 0.90]
colsample_bytree_values = [0.25, 0.50, 0.75, 0.90]

cv_folds = 10
tuning_steps = 500
include_orig = True
tuning_results = defaultdict(list)

col_names = [f'XGB_Step_{step}_Fold_{fold}' 
                 for step in range(tuning_steps) 
                     for fold in range(cv_folds)]
test_predictions = pd.DataFrame(0, index = test_ohe.index, columns = col_names)
valid_predictions = pd.DataFrame(0, index = train_ohe.index, columns = col_names)

random.seed(2201014)

skf_seed = random.randint(0, 2023)
skf = StratifiedKFold(n_splits = cv_folds, random_state = skf_seed, shuffle = True)

for step in range(tuning_steps):
    n_estimators = random.choice(n_estimators_values)
    eta = random.choice(eta_values)
    max_depth = random.choice(max_depth_values)
    subsample = random.choice(subsample_values)
    colsample_bytree = random.choice(colsample_bytree_values)
    
    aucs = []
    test_probs = []

    for i, (train_index, val_index) in enumerate(skf.split(train_ohe[FEATURES], train_ohe[TARGET])):
        X_train, X_val = train_ohe[FEATURES].iloc[train_index], train_ohe[FEATURES].iloc[val_index]
        y_train, y_val = train_ohe[TARGET].iloc[train_index], train_ohe[TARGET].iloc[val_index]

        if include_orig:
            # add in original stroke samples
            X_train = X_train.append(ext_ohe[FEATURES], ignore_index = True)
            y_train = y_train.append(ext_ohe[TARGET], ignore_index = True)
        
        xgb_seed = random.randint(0, 2023)
        xgb = XGBClassifier(n_estimators = n_estimators,
                            eta = eta,
                            max_depth = max_depth,
                            subsample = subsample,
                            colsample_bytree = colsample_bytree,
                            random_state = xgb_seed).fit(X_train.values, y_train)
        
        val_probs = [probs[1] for probs in xgb.predict_proba(X_val[FEATURES])]
        valid_predictions.loc[val_index, f'XGB_Step_{step}_Fold_{i}'] = val_probs
        
        fpr, tpr, thresholds = metrics.roc_curve(y_val, val_probs, pos_label = 1)
        auc = metrics.auc(fpr, tpr)
        aucs.append(auc)
        
        test_predictions[f'XGB_Step_{step}_Fold_{i}'] = \
            [probs[1] for probs in xgb.predict_proba(test_ohe[FEATURES])]
    
    tuning_results['step'].append(step)
    tuning_results['auc'].append(mean(aucs))
    tuning_results['n_estimators'].append(n_estimators)
    tuning_results['eta'].append(eta)
    tuning_results['max_depth'].append(max_depth)
    tuning_results['subsample'].append(subsample)
    tuning_results['colsample_bytree'].append(colsample_bytree)
    tuning_results['skf_seed'].append(skf_seed)
    tuning_results['xgb_seed'].append(xgb_seed)
    
    print(f'Step: {step}  AUC: {mean(aucs)}')
    
valid_predictions.to_csv('XGBoost_valid_predictions.csv', index = False)
test_predictions.to_csv('XGBoost_test_predictions.csv', index = False)

tuning_results = pd.DataFrame(tuning_results)
tuning_results.sort_values(by = 'auc', axis = 0, inplace = True, ascending = False)
tuning_results.to_csv('XGBoost_tuning_results.csv', index = False)
tuning_results


# ## Create submissions
# 
# ### Ensemble of top 5 model CV ensembles
# 
# The final entry was developed from the average of the predictions across the 10-folds for the best 5 models - 50 predictions for the test set were averaged.

# In[ ]:


sub = sub.copy()

best_cols = [f'XGB_Step_{step}_Fold_{fold}' 
               for step in tuning_results['step'][0:5]
                   for fold in range(cv_folds)]
cv_probs = test_predictions[best_cols].mean(axis = 1).round(decimals = 4)

sub['stroke'] = cv_probs
sub.to_csv('submisson_16.csv', index = False)

