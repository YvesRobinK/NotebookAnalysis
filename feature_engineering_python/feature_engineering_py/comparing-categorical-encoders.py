#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# This notebook was created for the playground series S3E2. Since this competition has several categorical features, I created this notebook to compare different categorical encoding techniques. This notebook accompanies [this discussion describing the many ways to go about categorical encoding](https://www.kaggle.com/competitions/playground-series-s3e2/discussion/377827).
# 
# I test the following techniques:
# - one-hot-encoding
# - ordinal encoding
# - target encoding
# - leave-one-out encoding
# - catboost encoding
# 
# For each technique, I train and evaluate a 10 fold stratified kfold cross validation for a CatBoost model.
# 
# This notebook is for demonstration purposes, and combining these techniques will likely yield a better result than an individual technique demonstrated in this notebook.
# 

# In[1]:


import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, Lasso
import catboost as cb

import warnings 
warnings.filterwarnings('ignore')


# In[2]:


train = pd.read_csv("/kaggle/input/playground-series-s3e2/train.csv")
og = pd.read_csv("/kaggle/input/stroke-prediction-dataset/healthcare-dataset-stroke-data.csv").dropna()
test = pd.read_csv("/kaggle/input/playground-series-s3e2/test.csv")
ss = pd.read_csv("/kaggle/input/playground-series-s3e2/sample_submission.csv")

train['source'] = 1
og['source'] = 0
train_df = pd.concat([train, og]).reset_index(drop=True)


# # Cross Validation and Training Functions

# In[3]:


def setup_cv(df, num_splits=10):
    kf = StratifiedKFold(n_splits=num_splits)
    
    for f, (t_, v_) in enumerate(kf.split(X=df, y=df.stroke)):
        df.loc[v_, 'fold'] = f
        
    return df

train_df = setup_cv(train_df)


# In[4]:


cb_params = {
    'depth': 3,
    'learning_rate': 0.01,
    'rsm': 0.5,
    'subsample': 0.931,
    'l2_leaf_reg': 69,
    'min_data_in_leaf': 20,
    'random_strength': 0.175,
    
    'random_seed': 42,
    'use_best_model': True,
    'task_type': 'CPU',
    'bootstrap_type': 'Bernoulli',
    'grow_policy': 'SymmetricTree',
    'loss_function': 'Logloss',
    'eval_metric': 'AUC'
}

def catboost_cross_validation(feature_function):
    oof_scores = list()
    for f in range(10):
        X_train = train_df[train_df.fold!=f] 
        y_train = train_df[train_df.fold!=f].stroke

        X_valid = train_df[train_df.fold==f]
        y_valid = train_df[train_df.fold==f].stroke

        X_train, X_valid = feature_function(X_train, X_valid)

        cb_train = cb.Pool(data=X_train,
                           label=y_train)
        cb_valid = cb.Pool(data=X_valid,
                           label=y_valid)

        model = cb.train(params=cb_params,
                         dtrain=cb_train,
                         num_boost_round=10000,
                         evals=cb_valid, 
                         early_stopping_rounds=500,
                         verbose=False)

        oof = model.predict(cb_valid)
        oof_scores.append(roc_auc_score(y_valid, oof))

    print('avg_score:', np.mean(oof_scores))


# # One-Hot-Encoding
# 
# One-hot encoding is useful when there is no order to the categorical data. The encoding creates a number of columns (the number of columns created is the number of unique values in the categorical column. Each column contains a binary representation of each possible value.

# ### Feature Engineering

# In[5]:


def get_one_hot_encoded_features(df, test_df):
    data = pd.DataFrame()
    test_data = pd.DataFrame()
    
    # non-categorical or already encoded
    data['age'] = df.age
    data['hypertension'] = df.hypertension
    data['heart_disease'] = df.heart_disease
    data['avg_glucose_level'] = df.avg_glucose_level
    data['bmi'] = df.bmi
    data['source'] = df.source
    
    test_data['age'] = test_df.age
    test_data['hypertension'] = test_df.hypertension
    test_data['heart_disease'] = test_df.heart_disease
    test_data['avg_glucose_level'] = test_df.avg_glucose_level
    test_data['bmi'] = test_df.bmi
    test_data['source'] = test_df.source
    
    # one-hot-encoding
    encoder = ce.OneHotEncoder(cols='gender', return_df=True, use_cat_names=True)
    names = [f'gender{i}' for i in range(3)]
    data[names] = encoder.fit_transform(df['gender'])
    test_data[names] = encoder.transform(test_df['gender'])
    
    encoder = ce.OneHotEncoder(cols='ever_married', return_df=True, use_cat_names=True)
    names = [f'ever_married{i}' for i in range(2)]
    data[names] = encoder.fit_transform(df['ever_married'])
    test_data[names] = encoder.transform(test_df['ever_married'])
    
    encoder = ce.OneHotEncoder(cols='work_type', return_df=True, use_cat_names=True)
    names = [f'work_type{i}' for i in range(5)]
    data[names] = encoder.fit_transform(df['work_type'])
    test_data[names] = encoder.transform(test_df['work_type'])
    
    encoder = ce.OneHotEncoder(cols='Residence_type', return_df=True, use_cat_names=True)
    names = [f'Residence_type{i}' for i in range(2)]
    data[names] = encoder.fit_transform(df['Residence_type'])
    test_data[names] = encoder.transform(test_df['Residence_type'])
    
    encoder = ce.OneHotEncoder(cols='smoking_status', return_df=True, use_cat_names=True)
    names = [f'smoking_status{i}' for i in range(4)]
    data[names] = encoder.fit_transform(df['smoking_status'])
    test_data[names] = encoder.transform(test_df['smoking_status'])
    
    return data, test_data


# ### Modelling

# In[6]:


catboost_cross_validation(get_one_hot_encoded_features)


# # Ordinal Encoding
# 
# Ordinal categorical data has an inherent order that can be converted to a numerical ranking. For example, a column with values “low, medium, high” has a clear order that can be converted into “0, 1, 2.” Ordinal encoding will result in a single column that contains the numerical representation of the categorical column.

# ### Feature Engineering

# In[7]:


def get_ordinal_encoded_features(df, test_df):
    data = pd.DataFrame()
    test_data = pd.DataFrame()
    
    # non-categorical or already encoded
    data['age'] = df.age
    data['hypertension'] = df.hypertension
    data['heart_disease'] = df.heart_disease
    data['avg_glucose_level'] = df.avg_glucose_level
    data['bmi'] = df.bmi
    data['source'] = df.source
    
    test_data['age'] = test_df.age
    test_data['hypertension'] = test_df.hypertension
    test_data['heart_disease'] = test_df.heart_disease
    test_data['avg_glucose_level'] = test_df.avg_glucose_level
    test_data['bmi'] = test_df.bmi
    test_data['source'] = test_df.source
    
    # ordinal encoding
    encoder = ce.OrdinalEncoder(cols=['gender'], return_df=True)
    data['gender'] = encoder.fit_transform(df['gender'])
    test_data['gender'] = encoder.transform(test_df['gender'])
    
    encoder = ce.OrdinalEncoder(cols=['ever_married'], return_df=True)
    data['ever_married'] = encoder.fit_transform(df['ever_married'])
    test_data['ever_married'] = encoder.transform(test_df['ever_married'])
    
    encoder = ce.OrdinalEncoder(cols=['work_type'], return_df=True)
    data['work_type'] = encoder.fit_transform(df['work_type'])
    test_data['work_type'] = encoder.transform(test_df['work_type'])
    
    encoder = ce.OrdinalEncoder(cols=['Residence_type'], return_df=True)
    data['Residence_type'] = encoder.fit_transform(df['Residence_type'])
    test_data['Residence_type'] = encoder.transform(test_df['Residence_type'])
    
    encoder = ce.OrdinalEncoder(cols=['smoking_status'], return_df=True,
                            mapping=[{'col':'smoking_status',
                                      'mapping':{
                                          'Unknown':-1,
                                          'never smoked':0,
                                          'formerly smoked':1,
                                          'smokes':2}}])
    data['smoking_status'] = encoder.fit_transform(df['smoking_status'])
    test_data['smoking_status'] = encoder.transform(test_df['smoking_status'])
    
    return data, test_data


# ### Modelling

# In[8]:


catboost_cross_validation(get_ordinal_encoded_features)


# # Target Encoding
# 
# Target encoding calculates the mean of the target variable for each category and the categories get replaced by the mean. This can be a fairly powerful technique, but you should be careful not to overfit.

# ### Feature Engineering

# In[9]:


def get_target_encoded_features(df, test_df):
    data = pd.DataFrame()
    test_data = pd.DataFrame()
    
    # non-categorical or already encoded
    data['age'] = df.age
    data['hypertension'] = df.hypertension
    data['heart_disease'] = df.heart_disease
    data['avg_glucose_level'] = df.avg_glucose_level
    data['bmi'] = df.bmi
    data['source'] = df.source
    
    test_data['age'] = test_df.age
    test_data['hypertension'] = test_df.hypertension
    test_data['heart_disease'] = test_df.heart_disease
    test_data['avg_glucose_level'] = test_df.avg_glucose_level
    test_data['bmi'] = test_df.bmi
    test_data['source'] = test_df.source
    
    # target encoding
    encoder = ce.TargetEncoder(cols='gender') 
    data['gender'] = encoder.fit_transform(df['gender'], df['stroke'])
    test_data['gender'] = encoder.transform(test_df['gender'])
    
    encoder = ce.TargetEncoder(cols='ever_married') 
    data['ever_married'] = encoder.fit_transform(df['ever_married'], df['stroke'])
    test_data['ever_married'] = encoder.transform(test_df['ever_married'])
    
    encoder = ce.TargetEncoder(cols='work_type') 
    data['work_type'] = encoder.fit_transform(df['work_type'], df['stroke'])
    test_data['work_type'] = encoder.transform(test_df['work_type'])
    
    encoder = ce.TargetEncoder(cols='Residence_type') 
    data['Residence_type'] = encoder.fit_transform(df['Residence_type'], df['stroke'])
    test_data['Residence_type'] = encoder.transform(test_df['Residence_type'])
    
    encoder = ce.TargetEncoder(cols='smoking_status') 
    data['smoking_status'] = encoder.fit_transform(df['smoking_status'], df['stroke'])
    test_data['smoking_status'] = encoder.transform(test_df['smoking_status'])
    
    return data, test_data


# ### Modelling

# In[10]:


catboost_cross_validation(get_target_encoded_features)


# # Leave-One-Out Encoding
# 
# This is basically the same as target encoding, but it excludes the current row’s target when calculating the mean target for a level to reduce the effect of outliers.

# ### Feature Engineering

# In[11]:


def get_leave_one_out_features(df, test_df):
    data = pd.DataFrame()
    test_data = pd.DataFrame()
    
    # non-categorical or already encoded
    data['age'] = df.age
    data['hypertension'] = df.hypertension
    data['heart_disease'] = df.heart_disease
    data['avg_glucose_level'] = df.avg_glucose_level
    data['bmi'] = df.bmi
    data['source'] = df.source
    
    test_data['age'] = test_df.age
    test_data['hypertension'] = test_df.hypertension
    test_data['heart_disease'] = test_df.heart_disease
    test_data['avg_glucose_level'] = test_df.avg_glucose_level
    test_data['bmi'] = test_df.bmi
    test_data['source'] = test_df.source
    
    # leave_one_out encoding
    encoder = ce.LeaveOneOutEncoder(cols='gender') 
    data['gender'] = encoder.fit_transform(df['gender'], df['stroke'])
    test_data['gender'] = encoder.transform(test_df['gender'])
    
    encoder = ce.LeaveOneOutEncoder(cols='ever_married') 
    data['ever_married'] = encoder.fit_transform(df['ever_married'], df['stroke'])
    test_data['ever_married'] = encoder.transform(test_df['ever_married'])
    
    encoder = ce.LeaveOneOutEncoder(cols='work_type') 
    data['work_type'] = encoder.fit_transform(df['work_type'], df['stroke'])
    test_data['work_type'] = encoder.transform(test_df['work_type'])
    
    encoder = ce.LeaveOneOutEncoder(cols='Residence_type') 
    data['Residence_type'] = encoder.fit_transform(df['Residence_type'], df['stroke'])
    test_data['Residence_type'] = encoder.transform(test_df['Residence_type'])
    
    encoder = ce.LeaveOneOutEncoder(cols='smoking_status') 
    data['smoking_status'] = encoder.fit_transform(df['smoking_status'], df['stroke'])
    test_data['smoking_status'] = encoder.transform(test_df['smoking_status'])
    
    return data, test_data


# ### Modelling

# In[12]:


catboost_cross_validation(get_leave_one_out_features)


# # CatBoost Encoding
# 
# Similar to target encoding and leave one out encoding, this is another method of supervised categorical encoding. The main difference is that it calculates the values on-the-fly. As a result, it is not necessary to add random noise because values naturally vary during the training phase.

# ### Feature Engineering

# In[13]:


def get_catboost_encoding_features(df, test_df):
    data = pd.DataFrame()
    test_data = pd.DataFrame()
    
    # non-categorical or already encoded
    data['age'] = df.age
    data['hypertension'] = df.hypertension
    data['heart_disease'] = df.heart_disease
    data['avg_glucose_level'] = df.avg_glucose_level
    data['bmi'] = df.bmi
    data['source'] = df.source
    
    test_data['age'] = test_df.age
    test_data['hypertension'] = test_df.hypertension
    test_data['heart_disease'] = test_df.heart_disease
    test_data['avg_glucose_level'] = test_df.avg_glucose_level
    test_data['bmi'] = test_df.bmi
    test_data['source'] = test_df.source
    
    # catboost encoding
    encoder = ce.CatBoostEncoder(cols='gender') 
    data['gender'] = encoder.fit_transform(df['gender'], df['stroke'])
    test_data['gender'] = encoder.transform(test_df['gender'])
    
    encoder = ce.CatBoostEncoder(cols='ever_married') 
    data['ever_married'] = encoder.fit_transform(df['ever_married'], df['stroke'])
    test_data['ever_married'] = encoder.transform(test_df['ever_married'])
    
    encoder = ce.CatBoostEncoder(cols='work_type') 
    data['work_type'] = encoder.fit_transform(df['work_type'], df['stroke'])
    test_data['work_type'] = encoder.transform(test_df['work_type'])
    
    encoder = ce.CatBoostEncoder(cols='Residence_type') 
    data['Residence_type'] = encoder.fit_transform(df['Residence_type'], df['stroke'])
    test_data['Residence_type'] = encoder.transform(test_df['Residence_type'])
    
    encoder = ce.CatBoostEncoder(cols='smoking_status') 
    data['smoking_status'] = encoder.fit_transform(df['smoking_status'], df['stroke'])
    test_data['smoking_status'] = encoder.transform(test_df['smoking_status'])
    
    return data, test_data


# ### Modelling

# In[14]:


catboost_cross_validation(get_catboost_encoding_features)

