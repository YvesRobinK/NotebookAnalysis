#!/usr/bin/env python
# coding: utf-8

# This is the submission v10 of the classic titanic dataset ML model
# 
# **This notebook include the following techniques:**
# * Data Imputation using MICE
# * label Encoding
# * Feature Scaling
# * Hyperparameter tuning
# * Feature Selection
# * Ensemble RandomForest Model
# 
# What it does not include:
# * Extensive EDA - as most other notebooks on the titanic dataset has extensive graphs and analysis

# In[1]:


import os
import numpy as np 
import pandas as pd 
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
train_data


# In[3]:


train_data.isnull().sum()


# Here, both the train and test sets have missing values. These will be dealt with an imputer model later

# In[4]:


test_data.isnull().sum()


# Name, Ticket and Cabin features are dropped as they are contain unique values, which do not provide any valuable information on the survived/not survived classification

# In[5]:


y_train = train_data['Survived']
x_train = train_data.drop(['Survived','Name' ,'Ticket', 'Cabin'], axis=1)
ids = test_data['PassengerId']
x_test = test_data.drop(['Name' ,'Ticket', 'Cabin'], axis=1)
x_train


# **LABEL ENCODING**
# 
# Label Encoding the categorical features using sklearn LabelEncoder class. Learn more about the class here https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

# In[6]:


labelEncoder1 = LabelEncoder()
x_train['Sex'] = labelEncoder1.fit_transform(x_train['Sex'])
x_test['Sex'] = labelEncoder1.transform(x_test['Sex'])

labelEncoder2 = LabelEncoder()
x_train['Embarked'] = labelEncoder2.fit_transform(x_train['Embarked'])
x_test['Embarked'] = labelEncoder2.transform(x_test['Embarked'])
x_train


# **DATA SCALING**
# 
# **MINMAX SCALER**
# 
# A normalization technique is required to scale the continuous features to a range. For this the minmax scaler has been used. Learn more about min max scaler here https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

# In[7]:


minmax = MinMaxScaler()
x_train[["Age", "Fare"]] = minmax.fit_transform(x_train[["Age", "Fare"]])
x_test[["Age", "Fare"]] = minmax.fit_transform(x_test[["Age", "Fare"]])


# **DATA IMPUTATION**
# 
# **MICE IMPUTER**
# 
# An imputer is used to fill the missing values in a dataset. I favoured the MICE technique as it takes into account the other features in the dataset in order to fill the missing values of a particular feature. It helps in reducing the anomlaies that may occur due to the univariate imputation techiques. Learn more about MICE here https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html

# In[8]:


imputer = IterativeImputer(random_state=42, verbose=1)
train_imputed = pd.DataFrame(imputer.fit_transform(x_train), columns = ['PassengerId','Pclass', 'Sex' ,'Age', 
                                                                        'SibSp', 'Parch', 'Fare','Embarked'])

test_imputed = pd.DataFrame(imputer.transform(x_test), columns = ['PassengerId','Pclass', 'Sex' ,'Age', 'SibSp', 
                                                                  'Parch', 'Fare', 'Embarked'])


# **Hyperparameter Tuning**
# 
# It is used to find the best subset of parameters to find the best model by using all provided combinations of parameters. GridSearch can be used for this purpose

# In[9]:


run_gs = False

if run_gs:
    parameter_grid = {
                 'max_depth' : [2, 4, 6],
                 'n_estimators': [100, 50],
                 'criterion' : ['entropy', 'gini'],
                 'min_samples_split': [2, 4, 6],
                 'min_samples_leaf': [1, 3, 6]
                 }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation,
                               verbose=1
                              )

    grid_search.fit(train_imputed, y_train)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))


# **Feature Engineering**
# 
# Feature Engineering is used to select features based on statistic or wrapper methods to find the subset of features that can improve the model performance

# In[10]:


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)

estimator = RandomForestClassifier(max_depth=6, criterion='gini' , min_samples_leaf=1, min_samples_split=4, 
                             n_estimators=50, bootstrap=True, random_state=42)
selector = RFECV(estimator, step=1, cv=cv, min_features_to_select=1)
selector = selector.fit(train_imputed, y_train)
selector.support_


# In[11]:


train_imputed = train_imputed.drop(['Parch', 'Embarked'], axis=1)
test_imputed = test_imputed.drop(['Parch', 'Embarked'], axis=1)


# In[12]:


clf = RandomForestClassifier(max_depth=6, criterion='gini' , min_samples_leaf=1, min_samples_split=4, 
                             n_estimators=50, bootstrap=True, random_state=42)
clf.fit(train_imputed, y_train)
predictions = clf.predict(test_imputed)


# In[13]:


output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)


# In[ ]:




