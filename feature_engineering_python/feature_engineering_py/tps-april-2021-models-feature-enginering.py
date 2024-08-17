#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
# <a id="table-of-contents"></a>
# * [1. Introduction](#1)
# * [2. Preparation](#2)
#     * [2.1. Load packages](#2.1)
#     * [2.2. Load dataset](#2.2)
#     * [2.3. Data pre-processing](#2.3)
# * [3. Baseline Model](#3)
#     * [3.1. Catboost](#3.1)
#     * [3.2. XGBoost](#3.2)
#     * [3.3. LGBM](#3.3)
#     * [3.4. Hard Voting](#3.4)
# * [4. Features Engineering](#4)
#     * [4.1. Family](#4.1)
#     * [4.2. Without family](#4.2)
#     * [4.3. First name and last name](#4.3)
#     * [4.4. Ticket](#4.4)
#     * [4.5. Age - binning](#4.5)
#     * [4.6. Fare - binning](#4.6)
# * [5. Baseline Model Post Features Engineering](#5)
#     * [5.1. Catboost](#5.1)
#     * [5.2. XGBoost](#5.2)
#     * [5.3. LGBM](#5.3)
#     * [5.4. Hard Voting](#5.4)
# * [6. Psuedo Labeling](#6)
#     * [6.1 Preparation](#6.1)
#     * [6.2. Features Engineering](#6.2)
#     * [6.3. Train Model & Prediction](#6.3)
#     * [6.4. Submission](#6.4)

# [back to top](#table-of-contents)
# <a id="1"></a>
# # 1. Introduction
# 
# The notebook will try to explore models and feature engineering performance in predicting whether or not a passenger survived the sinking of the Synthanic (a synthetic, much larger dataset based on the actual Titanic dataset). The score is the percentage of passengers that are correctly predicted, known as accuracy.

# [back to top](#table-of-contents)
# <a id="2"></a>
# # 2. Preparation
# 
# Steps that will be performed:
# 
# * Load packages for performing label encoding, cross validation, modeling and accuracy measurement.
# * Combine train and test dataset, the purpose is to tackle missing categories when performing label encoding and to fill missing value on continuous features.
# * Label encode all the categorical features and fill missing value in continous features.
# * Split back preprocessed combine dataset into train and test dataset.
# 
# <a id="2.1"></a>
# ## 2.1. Load packages
# Load packages for performing label encoding, cross validation, modeling and accuracy measurement.

# In[1]:


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# [back to top](#table-of-contents)
# <a id="2.2"></a>
# ## 2.2. Load dataset
# 
# Load `train`, `test` and`submission` dataset and combine `train` and `test` dataset into `combine` dataset for performing data pre-processing.

# In[2]:


train = pd.read_csv('../input/tabular-playground-series-apr-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-apr-2021/test.csv')
submission = pd.read_csv('../input/tabular-playground-series-apr-2021/sample_submission.csv')
combine = pd.concat([train, test], axis=0)


# [back to top](#table-of-contents)
# <a id="2.3"></a>
# ## 2.3. Data pre-processing
# 
# Data pre-processing that are used:
# 
# * **Cabin**
#     * Take the first letter from the string.
#     * Fill missing values with `NA` category. 
#     * Use label encoding to convert them into numbers.
# 
# * **Embarked**
#     * Fill missing values with `NA` category.
#     * Use label encoding to convert them into numbers.
#     
# * **Sex**
#     * Use label encoding to convert them into numbers.
#     
# * **Fare**
#     * Fill missing values with `Fare` mean.
#     
# * **Age**
#     * Fill missing values with `Age` mean.
#     
# * **Name**
#     * Will be treated as categorical feature.
#     
# * **Ticket**
#     * Will be treated as categorical feature.
#     
# * **PassengerId**
#     * Will be taken out from the model.

# In[3]:


combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine = combine.drop(['PassengerId'], axis=1)

cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket']
cont_features = [col for col in combine.columns if col not in cat_features + ['Survived']]
features = cat_features + cont_features

label_encoder = LabelEncoder()
for col in cat_features:
    combine[col] = label_encoder.fit_transform(combine[col])


# [back to top](#table-of-contents)
# <a id="3"></a>
# # 3. Baseline Model
# 
# This section will evaluate the performance of `Catboost`, `XGBoost` and `LGBM` using preprocessed train dataset without any hyperparameters tuning. `Hard Voting` will be used to ensemble the models. `Hard Voting` calculation is based on the most voting from three models, for example: if `Catboost` vote 1, `XGboost` vote 1 and `LGBM` vote 0, then the final result is 1.
# 
# **Observations:** 
# * `Catboost` gives the best performance even higher than the `Hard Voting` ensemble which is expected to be performed better than individual model. It seems the others model drag down `Catboost` performance.
# * `XGBoost` is the second best performance followed by `LGBM`.

# In[4]:


train = combine.iloc[:100000, :]
test = combine.iloc[100000:, :]
test = test.drop('Survived', axis=1)
model_results = pd.DataFrame()
folds = 5


# [back to top](#table-of-contents)
# <a id="3.1"></a>
# ## 3.1. Catboost
# `Catboost` has the highest `accuracy` score of `0.78428` compared to others model even higher than hard voting ensemble.

# In[5]:


train_oof = np.zeros((100000,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train[features], train['Survived'])):
    X_train, X_valid = train.iloc[train_idx], train.iloc[valid_idx]
    y_train = X_train['Survived']
    y_valid = X_valid['Survived']
    X_train = X_train.drop('Survived', axis=1)
    X_valid = X_valid.drop('Survived', axis=1)

    model = CatBoostClassifier(
        verbose=0,
        eval_metric="Accuracy",
        random_state=42,
        cat_features=cat_features
    )

    model =  model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    temp_oof = model.predict(X_valid)
    train_oof[valid_idx] = temp_oof
    print(f'Fold {fold} Accuracy: ', accuracy_score(y_valid, temp_oof))
    
print(f'OOF Accuracy: ', accuracy_score(train['Survived'], train_oof))
model_results['CatBoost'] = train_oof


# [back to top](#table-of-contents)
# <a id="3.2"></a>
# ## 3.2. XGBoost
# `XGBoost` has the second best accuracy with `0.78201`.

# In[6]:


train_oof = np.zeros((100000,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train[features], train['Survived'])):
    X_train, X_valid = train.iloc[train_idx], train.iloc[valid_idx]
    y_train = X_train['Survived']
    y_valid = X_valid['Survived']
    X_train = X_train.drop('Survived', axis=1)
    X_valid = X_valid.drop('Survived', axis=1)

    model = XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    model =  model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=0)
    temp_oof = model.predict(X_valid)
    train_oof[valid_idx] = temp_oof
    print(f'Fold {fold} Accuracy: ', accuracy_score(y_valid, temp_oof))
    
print(f'OOF Accuracy: ', accuracy_score(train['Survived'], train_oof))
model_results['XGBoost'] = train_oof


# [back to top](#table-of-contents)
# <a id="3.3"></a>
# ## 3.3. LGBM
# `LGBM` is in the last position with `0.78061` accuracy.

# In[7]:


cat_features_index = []
for col in cat_features:
    cat_features_index.append(train.columns.get_loc(col))

train_oof = np.zeros((100000,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train[features], train['Survived'])):
    X_train, X_valid = train.iloc[train_idx], train.iloc[valid_idx]
    y_train = X_train['Survived']
    y_valid = X_valid['Survived']
    X_train = X_train.drop('Survived', axis=1)
    X_valid = X_valid.drop('Survived', axis=1)

    model = LGBMClassifier(
        verbose=0,
        metric="Accuracy",
        random_state=42,
        cat_feature=cat_features_index,
        force_row_wise=True
    )

    model =  model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=0)
    temp_oof = model.predict(X_valid)
    train_oof[valid_idx] = temp_oof
    print(f'Fold {fold} Accuracy: ', accuracy_score(y_valid, temp_oof))
    
print(f'OOF Accuracy: ', accuracy_score(train['Survived'], train_oof))
model_results['LGBM'] = train_oof


# [back to top](#table-of-contents)
# <a id="3.4"></a>
# ## 3.4. Hard voting
# `Hard Voting` ensemble accuracy of `0.78154` higher than individual accuracy of `XGBoost` and `LGBM`.

# In[8]:


train_oof = np.zeros((100000,))
train_oof = np.where(model_results.sum(axis=1) > 2, 1, 0)
print(f'OOF Accuracy: ', accuracy_score(train['Survived'], train_oof))


# [back to top](#table-of-contents)
# <a id="4"></a>
# # 4. Features Engineering
# This section will explore if there are new feature that can be generated from the dataset and improved the baseline model accuracy. Baseline model that will be used is `LGBM` as it is light and fast. `LGBM` baseline accuracy is `0.78061`, it's expected by adding/removing feature/s will give a better accuracy than the baseline model.

# In[9]:


def combine_dataset():
    train = pd.read_csv('../input/tabular-playground-series-apr-2021/train.csv')
    test = pd.read_csv('../input/tabular-playground-series-apr-2021/test.csv')
    return pd.concat([train, test], axis=0)

def lgbm_oof(combine, cat_features):
    cont_features = [col for col in combine.columns if col not in cat_features + ['Survived']]
    features = cat_features + cont_features

    label_encoder = LabelEncoder()
    for col in cat_features:
        combine[col] = label_encoder.fit_transform(combine[col])
    train = combine.iloc[:100000, :]
    test = combine.iloc[100000:, :]
    test = test.drop('Survived', axis=1)

    cat_features_index = []
    for col in cat_features:
        cat_features_index.append(train.columns.get_loc(col))    

    train_oof = np.zeros((100000,))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(train[features], train['Survived'])):
        X_train, X_valid = train.iloc[train_idx], train.iloc[valid_idx]
        y_train = X_train['Survived']
        y_valid = X_valid['Survived']
        X_train = X_train.drop('Survived', axis=1)
        X_valid = X_valid.drop('Survived', axis=1)

        model = LGBMClassifier(
            verbose=0,
            metric="Accuracy",
            random_state=42,
            cat_feature=cat_features_index,
            force_row_wise=True
        )

        model =  model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
        temp_oof = model.predict(X_valid)
        train_oof[valid_idx] = temp_oof
        print(f'Fold {fold} Accuracy: ', accuracy_score(y_valid, temp_oof))

    print(f'OOF Accuracy: ', accuracy_score(train['Survived'], train_oof))


# [back to top](#table-of-contents)
# <a id="4.1"></a>
# ## 4.1. Family
# This is one of the most popular feature engineering for `Titanic` dataset. It added `SibSp` and `Parch` together to create a new feature called `Family`.
# 
# **Observations:**
# * It seems that adding `Family` feature doesn't improve the model.
# * Adding `Family` and also keeping `SibSp` and `Parch` has a better accuracy than removing `SibSp` and `Parch`.
# 
# ### 4.1.1. Add Family - Continuous
#  
# * Add `Family` features and also keep `SibSp` and `Parch` to see if `SibSp` and `Parch` features can help imporove the model performance. 
# * Adding `Family` features into the data increased the model accuracy to `0.78055`.

# In[10]:


combine = combine_dataset()

combine['Family'] = combine['Parch'] + combine['SibSp']
combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine = combine.drop(['PassengerId'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket']

lgbm_oof(combine, cat_features)


# ### 4.1.2. Add Family and remove SibSp and Parch - Continuous
# 
# * Add `Family` features and remove `SibSp` and `Parch` features to see if `SibSp` and `Parch` is createing noise to the model. 
# * Deleting `SibSp` and `Parch` decreased the accuracy to `0.78049`.

# In[11]:


combine = combine_dataset()

combine['Family'] = combine['Parch'] + combine['SibSp']
combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine = combine.drop(['PassengerId', 'SibSp', 'Parch'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket']

lgbm_oof(combine, cat_features)


# [back to top](#table-of-contents)
# <a id="4.2"></a>
# ## 4.2. Without family
# Continuing the `Family` feature, a `WithoutFamily` categorical feature can also be derived to indicate if a passenger is traveling with/without family.
# 
# **Observations:**
# * Creating a new `WithoutFamily` categorical feature doesn't improve the model, removing `SibSp` and `Parch` features results to a worse model than keeping them.
# 
# ### 4.2.1. Adding WithoutFamily feature - Categorical
# * Add `WithoutFamily` feature to see if it can improve the prediction. 
# * `WithoutFamily` feature decreased the accuracy score to `0.77986`.

# In[12]:


combine = combine_dataset()

combine['Family'] = combine['Parch'] + combine['SibSp']
combine['WithoutFamily'] = np.where(combine['Family']==0, 1, 0)
combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine = combine.drop(['PassengerId', 'Family'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket', 'WithoutFamily']

lgbm_oof(combine, cat_features)


# ### 4.2.2. Adding WithoutFamily feature and remove SibSp and Parch - Categorical
# Add `WithoutFamily` feature  while remove `SibSp` and `Parch` to see if these 2 features add noise to the model. `WithoutFamily` without`SibSp` and `Parch` features decreased accuracy score to `0.7766`.

# In[13]:


combine = combine_dataset()

combine['Family'] = combine['Parch'] + combine['SibSp']
combine['WithoutFamily'] = np.where(combine['Family']==0, 1, 0)
combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine = combine.drop(['PassengerId', 'Family', 'SibSp', 'Parch'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket', 'WithoutFamily']

lgbm_oof(combine, cat_features)


# [back to top](#table-of-contents)
# <a id="4.3"></a>
# ## 4.3. First name and last name
# There are 2 general methods that will be explored to generate new feature/s from `Name`:
# * Extracting the first letter from the `first name` and `last name`.
# * Extracting the character length of `first name` and `last name`.
# 
# **Observations:**
# * There is only one feature that improved the model accuracy by adding`first name` and `last name` features as continuous **(4.3.6)** but the improvement is considered small compared to the baseline model.
# 
# ### 4.3.1. Last Name - Categorical
# * Extract `last name` first letter from `Name` feature and convert it into categorical feature using label encoding.
# * Converting the first letter from `last name` and convert it into categorical feature resulted to `0.78005` accuracy which doesn't improve the accuracy

# In[14]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine = pd.concat([combine, combine['Name'].str.split(',', expand=True)], axis=1)
combine = combine.rename(columns={0: 'LastName', 1:'FirstName'})
combine['LastName'] = combine["LastName"].str[0]
combine = combine.drop(['Name', 'FirstName', 'PassengerId'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Ticket', 'LastName']

lgbm_oof(combine, cat_features)


# ### 4.3.2. First Name - Categorical
# * Extract `first name` first letter from `Name` feature and convert it into categorical feature using label encoding.
# * Adding a categorical `first name` from the first letter only doesn't improve the model, the accuracy drop to `0.77979`.

# In[15]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine = pd.concat([combine, combine['Name'].str.split(',', expand=True)], axis=1)
combine = combine.rename(columns={0:'LastName', 1: 'FirstName'})
combine['FirstName'] = combine["FirstName"].str[1]
combine = combine.drop(['Name', 'LastName', 'PassengerId'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Ticket', 'FirstName']

lgbm_oof(combine, cat_features)


# ### 4.3.3. First name and last name - Categorical
# * Extract `first name` and `last name` first letter from `Name` feature and convert them into categorical feature using label encoding.
# * Adding both `first name` and `last name` from the first letter only doesn't improve the model, the accuracy drop to `0.77171`.

# In[16]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine = pd.concat([combine, combine['Name'].str.split(',', expand=True)], axis=1)
combine = combine.rename(columns={0:'LastName', 1: 'FirstName'})
combine['FirstName'] = combine["FirstName"].str[1:]
combine['LastName'] = combine["LastName"].str[0:]
combine = combine.drop(['Name', 'PassengerId'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Ticket', 'FirstName', 'LastName']

lgbm_oof(combine, cat_features)


# ### 4.3.4. Last Name - Continuous
# * Extract `last name` and calculate its length from `Name` feature.
# * Taking the length of `last name` doesn't improve the accuracy, it landed at `0.78016`.

# In[17]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine = pd.concat([combine, combine['Name'].str.split(',', expand=True)], axis=1)
combine = combine.rename(columns={0:'LastName', 1: 'FirstName'})
combine['LastName'] = combine["LastName"].str.len()
combine = combine.drop(['Name', 'FirstName', 'PassengerId'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Ticket']

lgbm_oof(combine, cat_features)


# ### 4.3.5. First Name - Continuous
# * Extract `first name` and calculate its length from `Name` feature.
# * Using the `first name` as continuous feature doesn't improve the model, accuracy is at `0.78049`.

# In[18]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine = pd.concat([combine, combine['Name'].str.split(',', expand=True)], axis=1)
combine = combine.rename(columns={0:'LastName', 1: 'FirstName'})
combine['FirstName'] = combine["FirstName"].str[1:]
combine['FirstName'] = combine['FirstName'].str.len()
combine = combine.drop(['Name', 'LastName', 'PassengerId'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Ticket']

lgbm_oof(combine, cat_features)


# ### 4.3.6. First name and last Name - Continuous
# * Extract `first name` and `last name` and calculate its length from `Name` feature.
# * Adding both `first name` and `last name` as continuous features improved the accuracy to `0.78067`.

# In[19]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine = pd.concat([combine, combine['Name'].str.split(',', expand=True)], axis=1)
combine = combine.rename(columns={0:'LastName', 1: 'FirstName'})
combine['LastName'] = combine["LastName"].str.len()
combine['FirstName'] = combine["FirstName"].str[1:]
combine['FirstName'] = combine['FirstName'].str.len()
combine = combine.drop(['Name', 'PassengerId'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Ticket']

lgbm_oof(combine, cat_features)


# ### 4.3.7. Sum Length of first name and last name - Continuous
# * Create a new feature based by summing up the length of `first name` and `last name` and delete `Name`, `FirstName` and `LastName` features.
# * Summing up numbers of letters from `first name` and `last name` does not improve the accuracy of `0.78014`.

# In[20]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine = pd.concat([combine, combine['Name'].str.split(',', expand=True)], axis=1)
combine = combine.rename(columns={0:'LastName', 1: 'FirstName'})
combine['LastName'] = combine["LastName"].str.len()
combine['FirstName'] = combine["FirstName"].str[1:]
combine['FirstName'] = combine['FirstName'].str.len()
combine['Name'] = combine['FirstName'] + combine['LastName']
combine = combine.drop(['FirstName', 'LastName', 'PassengerId'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Ticket']

lgbm_oof(combine, cat_features)


# ### 4.3.8. Combine first name and last Name - Continuous
# * Combine the length of `first name` and `last name` and combine them. In example: `Jack, Wilson` will be converted to `4` and `6` then both of it will be combined into `46` and be treated as continuous feature.
# * Combine the `first name` and `last name` (not summing up) features and convert it back to continuous features resulting to a decreased in accuracy to `0.77947`.

# In[21]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine = pd.concat([combine, combine['Name'].str.split(',', expand=True)], axis=1)
combine = combine.rename(columns={0:'LastName', 1: 'FirstName'})
combine['LastName'] = combine["LastName"].str.len()
combine['FirstName'] = combine["FirstName"].str[1:]
combine['FirstName'] = combine['FirstName'].str.len()
combine['Name'] = combine['LastName'].astype(str) + combine['FirstName'].astype(str)
combine['Name'] = combine['Name'].astype(float)
combine = combine.drop(['FirstName', 'LastName', 'PassengerId'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Ticket']

lgbm_oof(combine, cat_features)


# [back to top](#table-of-contents)
# <a id="4.4"></a>
# ## 4.4. Ticket
# There are 2 methods that will be used for `Ticket`:
# * Extract the digits from the `Ticket` features and use the digits as a new features called `Ticket Number`.
# * Extract alphabet from the `Ticket` features and use the digits as a new features called `Ticket Code`.
# 
# **Observations:**
# * Creating new feature from `Ticket Number` and put its type as continuous resulting the same accuracy as the baseline model.
# * Creating a new feature from `Ticket Code` and put it as categorical feature improve the accuracy.
# * Combining the `Ticket Number` as continuous and `Ticket Code` as categorical features resulting a more improved model accuracy than baseline model. **(4.4.4)**
# * Combining the `Ticket Number` and `Ticket Code` as categorical features improved model accuracy than baseline model but same as combining `Ticket Number` as continuous and `Ticket Code` as categorical features.

# ### 4.4.1. Ticket Numbers - Continuous
# Extracting `Ticket` numbers, put it as continuous features and fill the missing value to `0` has a accuracy of `0.78061` which is the same as the baseline model.

# In[22]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].str.extract('(\d+)')
combine['Ticket'] = combine['Ticket'].astype(float)
combine['Ticket'] = combine['Ticket'].fillna(0)
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine = combine.drop(['PassengerId'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name']

lgbm_oof(combine, cat_features)


# ### 4.4.2. Ticket Numbers - Categorical
# Extracting `Ticket` numbers, put it as categorical features and fill the missing value to `0` improve the accuracy to `0.77891`.

# In[23]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].str.extract('(\d+)')
combine['Ticket'] = combine['Ticket'].astype(float)
combine['Ticket'] = combine['Ticket'].fillna(0)
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine = combine.drop(['PassengerId'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket']

lgbm_oof(combine, cat_features)


# ### 4.4.3. Ticket Code - Categorical
# Extracting `Ticket` non-numerical string, put it as categorical features and fill the missing value to `NA` improve the accuracy to `0.77886`.

# In[24]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].str.replace('[^\w\s]','')
combine['Ticket'] = combine['Ticket'].str.replace(' ','')
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Ticket'] = combine['Ticket'].replace('(\d)', '', regex=True)
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine = combine.drop(['PassengerId'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket']

lgbm_oof(combine, cat_features)


# ### 4.4.4. Ticket Code - Categorical & Ticket Number - Continuous
# Combining `Ticket Code` as categorical and `Ticket Number` as continuous improve the accuracy to `0.78194` which is better than individual accuracy and baseline model.

# In[25]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['TicketCode'] = combine['Ticket'].str.replace('[^\w\s]','')
combine['TicketCode'] = combine['TicketCode'].str.replace(' ','')
combine['TicketCode'] = combine['TicketCode'].fillna('NA')
combine['TicketCode'] = combine['TicketCode'].replace('(\d)', '', regex=True)
combine['TicketNumber'] = combine['Ticket'].str.extract('(\d+)')
combine['TicketNumber'] = combine['TicketNumber'].astype(float)
combine['TicketNumber'] = combine['TicketNumber'].fillna(0)
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine = combine.drop(['Ticket', 'PassengerId'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name', 'TicketCode']

lgbm_oof(combine, cat_features)


# ### 4.4.5. Ticket Code - Categorical & Ticket Number - Categorical
# Combining `Ticket Code` and `Ticket Number` as categorical improve the accuracy to `0.78194` which is same as putting the `Ticket Number` as continuous.

# In[26]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['TicketCode'] = combine['Ticket'].str.replace('[^\w\s]','')
combine['TicketCode'] = combine['TicketCode'].str.replace(' ','')
combine['TicketCode'] = combine['TicketCode'].fillna('NA')
combine['TicketCode'] = combine['TicketCode'].replace('(\d)', '', regex=True)
combine['TicketNumber'] = combine['Ticket'].str.extract('(\d+)')
combine['TicketNumber'] = combine['TicketNumber'].astype(float)
combine['TicketNumber'] = combine['TicketNumber'].fillna(0)
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine = combine.drop(['Ticket', 'PassengerId'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name', 'TicketCode', 'TicketNumber']

lgbm_oof(combine, cat_features)


# [back to top](#table-of-contents)
# <a id="4.5"></a>
# ## 4.5. Age - binning
# This section will explore several methods for binning `Age` feature:
# * [Human Life Cycle](https://med.libretexts.org/Courses/American_Public_University/APUS%3A_An_Introduction_to_Nutrition_(Byerley)/Text/12%3A_Maternal_Infant_Childhood_and_Adolescent_Nutrition/12.02%3A_The_Human_Life_Cycle)
# * Fixed Interval
# 
# **Observations:**
# * Binning `Age` has a signficant impact to the model.
# * Converting `Age` into categorical features by binning it, improve the model performance.
# * Keeping both the continuous and the categorical (binning) `Age` doesn't make the model performed well though it's still better than the baseline model.
# * Converting missing value to new categorical `NA` improve the model than using the mean value.
# * Binning to only 2 categories stil has a significant impact to the accuracy.
# * It seems that `Age` feature has created a noise to the model, removing it improve the baseline model quite significantly.

# ### 4.5.1. Human Life Cycle - Categorical
# Binning the age using human life cycle improve the accuracy into `0.7807`, a litte bit better than the baseline model

# In[27]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine['AgeBin']=pd.cut(combine['Age'],[-np.inf, 2, 4, 9, 14, 19, 31, 51, np.inf], right=False,
                         labels = ['Infancy', 'Toddler', 'Childhood', 'Puberty', 'Older adolescence', 
                                   'Adulthood', 'Middle age', 'Senior years'])
combine = combine.drop(['PassengerId'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket', 'AgeBin']

lgbm_oof(combine, cat_features)


# ### 4.5.2. Human Life Cycle and remove Age - Categorical
# Binning the age using human life cycle and remove the `Age` feature improve the accuracy into `0.78267`, quite a significant improvement compared to the baseline model. I seems the `Age` feature create a noise to the model as it rudandant to `AgeBin`.

# In[28]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine['AgeBin'] = pd.cut(combine['Age'],[-np.inf, 2, 4, 9, 14, 19, 31, 51, np.inf], right=False,
                         labels = ['Infancy', 'Toddler', 'Childhood', 'Puberty', 'Older adolescence', 
                                   'Adulthood', 'Middle age', 'Senior years']).astype(str)
combine = combine.drop(['PassengerId', 'Age'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket', 'AgeBin']

lgbm_oof(combine, cat_features)


# ### 4.5.3. Human Life Cycle, remove Age and missing value - Categorical
# The different between this section compared to **4.5.2** is the way to treat `Age` missing value; in this section, missing value will be tag as `NA` instead of calculating the mean of the `Age`. There is an improvement in the accuracy compared to the baseline and previous section, the accuracy improved to `0.78282`.

# In[29]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['AgeBin'] = pd.cut(combine['Age'],[-np.inf, 2, 4, 9, 14, 19, 31, 51, np.inf], right=False,
                         labels = ['Infancy', 'Toddler', 'Childhood', 'Puberty', 'Older adolescence', 
                                   'Adulthood', 'Middle age', 'Senior years']).astype(str)
combine = combine.drop(['PassengerId', 'Age'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket', 'AgeBin']

lgbm_oof(combine, cat_features)


# ### 4.5.4. Fixed Interval 20 - Categorical
# Using an interval of 20 years, the model accuracy performed a little bit higher compared to the baseline model with accuracy of `0.78065` 

# In[30]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine['AgeBin']=pd.cut(combine['Age'],[-np.inf, 20, 40, 60, 80, np.inf], right=False)
combine = combine.drop(['PassengerId'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket', 'AgeBin']

lgbm_oof(combine, cat_features)


# ### 4.5.5. Fixed Interval 20 and remove Age - Categorical
# Once again, removing continuous `Age` features improve the model significantly into `0.78264`.

# In[31]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine['AgeBin']=pd.cut(combine['Age'],[-np.inf, 20, 40, 60, 80, np.inf], right=False)
combine = combine.drop(['PassengerId', 'Age'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket', 'AgeBin']

lgbm_oof(combine, cat_features)


# ### 4.5.6. Fixed Interval 20, remove Age and missing value - Categorical
# The different between this section compared to **4.5.5** is the way to treat `Age` missing value; in this section, missing value will be tag as `NA` instead of calculating the mean of the `Age`. There is an improvement in the accuracy compared to the baseline and previous section, the accuracy improved to `0.78288`.

# In[32]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['AgeBin']=pd.cut(combine['Age'],[-np.inf, 20, 40, 60, 80, np.inf], right=False)
combine = combine.drop(['PassengerId', 'Age'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket', 'AgeBin']

lgbm_oof(combine, cat_features)


# ### 4.5.7. Fixed Interval 30, remove Age and missing value - Categorical
# Base on 2 previous try on binning `Age`, this section will directly remove the `Age` continuous and treat the missing value as categorical features. The accuracy using a fixed interval of 30 is `0.78253`.

# In[33]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['AgeBin']=pd.cut(combine['Age'],[-np.inf, 30, 60, 90, np.inf], right=False)
combine = combine.drop(['PassengerId', 'Age'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket', 'AgeBin']

lgbm_oof(combine, cat_features)


# ### 4.5.8. Fixed Interval 50, remove Age and missing value - Categorical
# Surprisingly enough, creating only 2 categories by binnin at age 50 resulted a high accuracy of `0.78256`.

# In[34]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['AgeBin']=pd.cut(combine['Age'],[-np.inf, 50, np.inf], right=False)
combine = combine.drop(['PassengerId', 'Age'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket', 'AgeBin']

lgbm_oof(combine, cat_features)


# ### 4.5.9. Remove Age
# It's suspected that `Age` feature has created a noise to the model, splitting the features only into 2 category still create a high accuracy as can be seen in the previos section **(4.5.8)**. After removing `Age` feature without creating any binning, the accuracy is increased to `0.78275`.

# In[35]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine = combine.drop(['PassengerId', 'Age'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket']

lgbm_oof(combine, cat_features)


# [back to top](#table-of-contents)
# <a id="4.6"></a>
# ## 4.6. Fare - binning
# This section will explore binning possibility for `Fare` feature using a fixed interval and subjectivity base on the `Fare` distribution.
# 
# **Observations:**
# * It may be better to keep the `Fare` feature as a continuous features.
# * Using a fixed interval of 200 doesn't improve the accuracy.
# * Splitting `Fare` feature into 2 category still doesn't improve the accuracy.
# * Subjectively splitting the `Fare` feature also doesn't imporve the model.
# * Removing `Fare` has a little impact to the model accuracy on fixed interval 200 without `Fare` features and interval distribution without `Fare` feature.
# 

# ### 4.6.1. Fixed Interval 200 - Categorical
# Binning the `Fare` using a 200 fixed interval, decrease the accuracy to `0.77988`.

# In[36]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['FareBin']=pd.cut(combine['Fare'],[-np.inf, 200, 400, 600, 800, np.inf], right=False)
combine = combine.drop(['PassengerId'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket', 'FareBin']

lgbm_oof(combine, cat_features)


# ### 4.6.2. Fixed Interval 200 and remove Fare - Categorical
# Binning the fare using a 200 fixed interval and remove the Fare feature decrease the accuracy to `0.77989`.

# In[37]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['FareBin']=pd.cut(combine['Fare'],[-np.inf, 200, 400, 600, 800, np.inf], right=False)
combine = combine.drop(['PassengerId', 'Fare'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket', 'FareBin']

lgbm_oof(combine, cat_features)


# ### 4.6.3. Fixed Interval 400 - Categorical
# Binning the fare using a 400 fixed interval which technically split the feature into to 2 category increased the accuracy to `0.78078`.

# In[38]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['FareBin']=pd.cut(combine['Fare'],[-np.inf, 400, np.inf], right=False)
combine = combine.drop(['PassengerId'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket', 'FareBin']

lgbm_oof(combine, cat_features)


# ### 4.6.4. Fixed Interval 400 and remove Fare - Categorical
# Binning the fare using a 400 fixed interval which technically split the feature into to 2 category decrease the accuracy to `0.77997`.

# In[39]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['FareBin']=pd.cut(combine['Fare'],[-np.inf, 400, np.inf], right=False)
combine = combine.drop(['PassengerId', 'Fare'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket', 'FareBin']

lgbm_oof(combine, cat_features)


# ### 4.6.5. Interval Distribution - Categorical
# The split is based on subjectivity at `Fare` distribution which is separated on 50, 100 and 300. The accuracy result is `0.78002` which still below baseline model. 

# In[40]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['FareBin']=pd.cut(combine['Fare'],[-np.inf, 50, 100, 300, np.inf], right=False)
combine = combine.drop(['PassengerId'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket', 'FareBin']

lgbm_oof(combine, cat_features)


# ### 4.6.5. Interval Distribution and remove Fare - Categorical
# The split is based on subjectivity at `Fare` distribution which is separated on 50, 100 and 300 and also removing the `Fare` features. The accuracy result is `0.78017` which below baseline model. 

# In[41]:


combine = combine_dataset()

combine['Cabin'] = combine["Cabin"].str[0]
combine['Ticket'] = combine['Ticket'].fillna('NA')
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Age'] = combine['Age'].fillna(np.mean(combine['Age']))
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))
combine['FareBin']=pd.cut(combine['Fare'],[-np.inf, 50, 100, 300, np.inf], right=False)
combine = combine.drop(['PassengerId', 'Fare'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket', 'FareBin']

lgbm_oof(combine, cat_features)


# [back to top](#table-of-contents)
# <a id="5"></a>
# # 5. Baseline Model Post Features Engineering
# This section will evaluate the performance of `Catboost`, `XGBoost` and `LGBM` using features engineering that has been evaluated before using `LGBM`:
# * Convert `first name` and `last name` into its respective length from `Name` feature and remove `Name` feature.
# * Create new feature `TicketCode` and `TicketNumber` from `Ticket` feature and classify `TicketCode` as categorical and `TicketNumber` as continuous.
# * Create a new feature `AgeBin` which come from 20 interval of `Age` and remove `Age` feature.
# 
# **Observations:** 
# * Using new/modified features improves the accuracy on all models.
# * `Catboost` is still has the highest accuracy of `0.78596` improve from `0.78428`.
# * `XGBoost` performance is getting worse compared to baseline model a decreased from `0.78201` to `0.78109`. This may due to categorical feature treatement in `XGBoost`.
# * `LGBM` accuracy performance improves to `0.78418` from `0.78061`. It has the highest improvement as `LGBM` is used as baseline when doing the feature engineering.
# * Accuracy performance on Hard Voting ensemble using all models is worse than only use 2 models of `Catboost` and `LGBM`.
# * Even after after taking out the `XGBoost` from Hard Voting ensemble, it is still lower than `Catboost` performance but higher than `LGBM`.

# In[42]:


combine = combine_dataset()

combine = pd.concat([combine, combine['Name'].str.split(',', expand=True)], axis=1)
combine = combine.rename(columns={0:'LastName', 1: 'FirstName'})
combine['LastName'] = combine["LastName"].str.len()
combine['FirstName'] = combine["FirstName"].str[1:]
combine['FirstName'] = combine['FirstName'].str.len()
combine['TicketCode'] = combine['Ticket'].str.replace('[^\w\s]','')
combine['TicketCode'] = combine['TicketCode'].str.replace(' ','')
combine['TicketCode'] = combine['TicketCode'].fillna('NA')
combine['TicketCode'] = combine['TicketCode'].replace('(\d)', '', regex=True)
combine['TicketNumber'] = combine['Ticket'].str.extract('(\d+)')
combine['TicketNumber'] = combine['TicketNumber'].astype(float)
combine['TicketNumber'] = combine['TicketNumber'].fillna(0)
combine['AgeBin']=pd.cut(combine['Age'],[-np.inf, 20, 40, 60, 80, np.inf], right=False)
combine['Cabin'] = combine["Cabin"].str[0]
combine['Cabin'] = combine['Cabin'].fillna('NA')
combine['Embarked'] = combine['Embarked'].fillna('NA')
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))

combine = combine.drop(['Name', 'Ticket', 'Age', 'PassengerId'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'TicketCode', 'AgeBin']

cont_features = [col for col in combine.columns if col not in cat_features + ['Survived']]
features = cat_features + cont_features

label_encoder = LabelEncoder()
for col in cat_features:
    combine[col] = label_encoder.fit_transform(combine[col])
train = combine.iloc[:100000, :]
test = combine.iloc[100000:, :]
test = test.drop('Survived', axis=1)

cat_features_index = []
for col in cat_features:
    cat_features_index.append(train.columns.get_loc(col))


# [back to top](#table-of-contents)
# <a id="5.1"></a>
# ## 5.1. Catboost
# `Catboost` has the highest `accuracy` score of `0.78596` compared to others model even higher than hard voting ensemble.

# In[43]:


train_oof = np.zeros((100000,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train[features], train['Survived'])):
    X_train, X_valid = train.iloc[train_idx], train.iloc[valid_idx]
    y_train = X_train['Survived']
    y_valid = X_valid['Survived']
    X_train = X_train.drop('Survived', axis=1)
    X_valid = X_valid.drop('Survived', axis=1)

    model = CatBoostClassifier(
        verbose=0,
        eval_metric="Accuracy",
        random_state=42,
        cat_features=cat_features
    )

    model =  model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    temp_oof = model.predict(X_valid)
    train_oof[valid_idx] = temp_oof
    print(f'Fold {fold} Accuracy: ', accuracy_score(y_valid, temp_oof))
    
print(f'OOF Accuracy: ', accuracy_score(train['Survived'], train_oof))
model_results['CatBoost'] = train_oof


# [back to top](#table-of-contents)
# <a id="5.2"></a>
# ## 5.2. XGBoost
# `XGBoost` has the second best accuracy with `0.78109` worse than baseline model without feature engineering.

# In[44]:


train_oof = np.zeros((100000,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train[features], train['Survived'])):
    X_train, X_valid = train.iloc[train_idx], train.iloc[valid_idx]
    y_train = X_train['Survived']
    y_valid = X_valid['Survived']
    X_train = X_train.drop('Survived', axis=1)
    X_valid = X_valid.drop('Survived', axis=1)

    model = XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    model =  model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=0)
    temp_oof = model.predict(X_valid)
    train_oof[valid_idx] = temp_oof
    print(f'Fold {fold} Accuracy: ', accuracy_score(y_valid, temp_oof))
    
print(f'OOF Accuracy: ', accuracy_score(train['Survived'], train_oof))
model_results['XGBoost'] = train_oof


# [back to top](#table-of-contents)
# <a id="5.3"></a>
# ## 5.3. LGBM
# `LGBM` has an accuracy of `0.78418` higher than `XGBoost` accuracy post feature engineering.

# In[45]:


cat_features_index = []
for col in cat_features:
    cat_features_index.append(train.columns.get_loc(col))

train_oof = np.zeros((100000,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train[features], train['Survived'])):
    X_train, X_valid = train.iloc[train_idx], train.iloc[valid_idx]
    y_train = X_train['Survived']
    y_valid = X_valid['Survived']
    X_train = X_train.drop('Survived', axis=1)
    X_valid = X_valid.drop('Survived', axis=1)

    model = LGBMClassifier(
        verbose=0,
        metric="Accuracy",
        random_state=42,
        cat_feature=cat_features_index,
        force_row_wise=True
    )

    model =  model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=0)
    temp_oof = model.predict(X_valid)
    train_oof[valid_idx] = temp_oof
    print(f'Fold {fold} Accuracy: ', accuracy_score(y_valid, temp_oof))
    
print(f'OOF Accuracy: ', accuracy_score(train['Survived'], train_oof))
model_results['LGBM'] = train_oof


# [back to top](#table-of-contents)
# <a id="5.4"></a>
# ## 5.4. Hard voting
# **Observations:**
# * `Hard Voting` has an accuracy of `0.78326` higher than `XGBoost`.
# * Dropping `XGBoost` from the ensemble improve the model accuracy to `0.78541` but still lower than `Catboost`.
# 
# ### 5.4.1. Hard voting using all models

# In[46]:


train_oof = np.zeros((100000,))
train_oof = np.where(model_results.sum(axis=1) > 2, 1, 0)
print(f'OOF Accuracy: ', accuracy_score(train['Survived'], train_oof))


# ### 5.4.2. Hard voting without XGBoost

# In[47]:


model_results = model_results.drop('XGBoost', axis=1)
train_oof = np.zeros((100000,))
train_oof = np.where(model_results.sum(axis=1) > 1, 1, 0)
print(f'OOF Accuracy: ', accuracy_score(train['Survived'], train_oof))


# [back to top](#table-of-contents)
# <a id="6"></a>
# # 6. Pseudo-Labeling
# 
# **Pseudo-Labeling** is a technique of using unlabeled data (in this case the test dataset) combine with labeled data (in this case train dataset) to create a better model. There are 4 steps on performing pseudo-labeling:
# 1. Train model on train data and make a prediction for test data.
# 2. Use predictions from stage 1 as `pseudo` labels for test data. 
# 3. Combined pseudolabeled dataset with train dataset.
# 4. Fit a new model on this combined dataset.
# 
# **Notes:** Taken from [Pseudolabelling - Tips and tricks](https://www.kaggle.com/c/tabular-playground-series-apr-2021/discussion/231738) by [Alexander Ryzhkov](https://www.kaggle.com/alexryzhkov)
# 
# This section also mainly inspired by these notebooks:
# * Main code is taken from [TPS-Apr2021 Catboost Run Pseudo label](https://www.kaggle.com/gomes555/tps-apr2021-catboost-run-pseudo-label) by [Fellipe Gomes](https://www.kaggle.com/gomes555)
# * First pseudo label result are hard voting from these notebooks:
#     * [TPS Apr 2021 single DecisionTreeModel](https://www.kaggle.com/hiro5299834/tps-apr-2021-single-decisiontreemodel) by [BIZEN](https://www.kaggle.com/hiro5299834)
#     * [TPS-APR-2021-LGBM](https://www.kaggle.com/svyatoslavsokolov/tps-apr-2021-lgbm) by [Svyatoslav Sokolov](https://www.kaggle.com/svyatoslavsokolov)
#     * [Catboost](https://www.kaggle.com/belov38/catboost-lb) by [Ilya Belov](https://www.kaggle.com/belov38)
# 
# Please check out their great notebooks!
# 
# [back to top](#table-of-contents)
# <a id="6.1"></a>
# ## 6.1. Preparation
# This section covered `step 1 to 3`. Prediction for unlabeled data (test dataset) is taken from [TPS Apr 2021 single DecisionTreeModel](https://www.kaggle.com/hiro5299834/tps-apr-2021-single-decisiontreemodel) by [BIZEN](https://www.kaggle.com/hiro5299834)

# In[48]:


train = pd.read_csv('../input/tabular-playground-series-apr-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-apr-2021/test.csv')
submission = pd.read_csv('../input/tabular-playground-series-apr-2021/sample_submission.csv')
decision_tree = pd.read_csv('../input/tps-apr-2021-single-decisiontreemodel/submission.csv', index_col=0)
lgbm = pd.read_csv('../input/tps-apr-2021-lgbm/submission.csv', index_col=0)
catboost = pd.read_csv('../input/catboost-lb/result.csv', index_col=0)
pseudo_label = pd.DataFrame()
pseudo_label = pd.concat([decision_tree, lgbm, catboost], axis=1)
pseudo_label['final'] = np.where(pseudo_label.sum(axis=1) > 1, 1, 0)
FOLDS = 5
test['Survived'] = [x for x in pseudo_label.final]
combine = pd.concat([train, test], axis=0)


# [back to top](#table-of-contents)
# <a id="6.2"></a>
# ## 6.2. Features Engineering
# Features engineering are taken from previous sections.

# In[49]:


# Cabin
combine['Cabin'] = combine["Cabin"].str[0]
combine['Cabin'] = combine['Cabin'].fillna('NA')

#Embarked
combine['Embarked'] = combine['Embarked'].fillna('NA')

# Fare
combine['Fare'] = combine['Fare'].fillna(np.mean(combine['Fare']))

# Ticket
combine['TicketCode'] = combine['Ticket'].str.replace('[^\w\s]','')
combine['TicketCode'] = combine['TicketCode'].str.replace(' ','')
combine['TicketCode'] = combine['TicketCode'].fillna('NA')
combine['TicketCode'] = combine['TicketCode'].replace('(\d)', '', regex=True)
combine['TicketNumber'] = combine['Ticket'].str.extract('(\d+)')
combine['TicketNumber'] = combine['TicketNumber'].astype(float)
combine['TicketNumber'] = combine['TicketNumber'].fillna(0)

# Age 
combine['AgeBin']=pd.cut(combine['Age'],[-np.inf, 20, 40, 60, 80, np.inf], right=False)

# Preprocess
combine = combine.drop(['Name'], axis = 1)
cat_features = ['Pclass', 'Sex', 'AgeBin', 'Cabin', 'Embarked', 'TicketCode']
label_encoder = LabelEncoder()
for col in cat_features:
    combine[col] = label_encoder.fit_transform(combine[col])
target = 'Survived'
features = ['Cabin', 'Embarked', 'Pclass', 'Sex', 'AgeBin', 'Parch', 'SibSp', 
            'Fare', 'TicketCode', 'TicketNumber']

# Splitting into train and test
train = combine.iloc[:100000, :]
test = combine.iloc[100000:, :]
test = test.drop('Survived', axis=1)


# [back to top](#table-of-contents)
# <a id="6.3"></a>
# ## 6.3. Train Model & Prediction
# Most of the code are taken from [TPS-Apr2021 Catboost Run Pseudo label](https://www.kaggle.com/gomes555/tps-apr2021-catboost-run-pseudo-label) by [Fellipe Gomes](https://www.kaggle.com/gomes555). It has also implemented threshold optimization in his code.

# In[50]:


train_oof = np.zeros((200000,))
test_predicts = pd.DataFrame()

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=314)
for fold, (train_idx, valid_idx) in enumerate(skf.split(combine[features], combine['Survived'])):
    X_train, X_valid = combine.iloc[train_idx], combine.iloc[valid_idx]
    y_train = X_train['Survived']
    y_valid = X_valid['Survived']
    X_train = X_train[features]
    X_valid = X_valid[features]
    X_test = test[features]

    params = {'iterations': 10000,
              'use_best_model':True ,
              'eval_metric': 'AUC',
              'loss_function':'Logloss',
              'od_type':'Iter',
              'od_wait':500,
              'depth': 6,
              'l2_leaf_reg': 3,
              'bootstrap_type': 'Bayesian',
              'bagging_temperature': 2,
              'max_bin': 254,
              'grow_policy': 'SymmetricTree',
              'cat_features': cat_features,
              'verbose': 0,
              'random_seed': 314}

    model = CatBoostClassifier(**params)
    model = model.fit(X_train,y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                use_best_model=True,
                plot=False)
    predict = model.predict_proba(X_valid)[:, 1]
    accuracy = accuracy_score(y_valid, np.where(predict>0.5, 1, 0))
    print(f'Fold {fold} Base Accuracy:', accuracy)
    
    # Threshold optimization
    thresholds = np.arange(0.0, 1.0, 0.01)
    accuracies = []
    for threshold in thresholds:
        accuracies.append(accuracy_score(y_valid, np.where(predict>threshold, 1, 0)))
    
    accuracies = np.array(accuracies)
    best_accuracy = accuracies.max()
    best_accuracy_threshold = thresholds[accuracies.argmax()]
    print(f'Fold {fold} Best Accuracy:', best_accuracy, 'with threshold of', f'{best_accuracy_threshold}')
    
    temp_oof = np.where(predict>best_accuracy_threshold, 1, 0)
    train_oof[valid_idx] = temp_oof
    
    test_predict = model.predict_proba(X_test)[:, 1]
    test_predict = np.where(test_predict>best_accuracy_threshold, 1, 0)
    test_predicts['Fold '+str(fold)] = test_predict
    
print(f'OOF Accuracy: ', accuracy_score(combine['Survived'], train_oof))


# [back to top](#table-of-contents)
# <a id="6.4"></a>
# ## 6.4. Submission
# `Hard voting` is used to get the final prediction, `above 2` will be `1` and `below 2` will be `0`.

# In[51]:


test_predicts['voting'] = np.where(test_predicts.sum(axis=1) > (FOLDS/2), 1, 0)
submission['Survived'] = test_predicts['voting']
submission.to_csv('submission.csv', index=False)

