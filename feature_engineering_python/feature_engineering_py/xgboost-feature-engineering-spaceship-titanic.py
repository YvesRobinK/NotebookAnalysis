#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename == 'train.csv':
            data = pd.read_csv(os.path.join(dirname, 'train.csv'))
        elif filename == 'test.csv':
            test_data = pd.read_csv(os.path.join(dirname, 'test.csv'))
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


# Create test ids for evaluation
test_ids = test_data['PassengerId']


# In[3]:


from sklearn import preprocessing  
# Helper function to encode all the labels that need encoding in the dataset
def encode_labels(dataset):
    label_encoder = preprocessing.LabelEncoder()
    for column in dataset:
        dtype = dataset[column].dtype
        if dtype not in ['int64', 'float64']:
            dataset[column] = label_encoder.fit_transform(dataset[column])
    return dataset


# In[4]:


data.head()


# PassengerId is useless, same with Name. 

# In[5]:


# Drop useless attributes
train_x = data.drop(['PassengerId', 'Name', 'Cabin', 'Age'], axis=1)


# In[6]:


# Encoding of labels
train_x = encode_labels(train_x)
train_y = train_x['Transported']
train_x = train_x.drop('Transported', axis=1)


# In[7]:


# Check how balanced the classes are
class_counts = train_y.value_counts()
print(class_counts)


# Seems like the dataset is fairly balanced, so there is no need to do any oversampling for the minority class

# In[8]:


# Standardize and normalize the data
from sklearn.preprocessing import StandardScaler
common_cols = [col for col in set(train_x.columns).intersection(test_data.columns)]
train_x = train_x[common_cols]
test_data = test_data[common_cols]
test_data = encode_labels(test_data)
sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_data = encode_labels(test_data)
test_data = sc.transform(test_data)


# In[9]:


train_x.shape


# In[10]:


# Training with XGBoost using stratified K fold and randomized search cross-validation
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
# split data into train and test sets
params = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],  # You can choose more values within this range
    "n_estimators": [50, 100, 200, 300, 500, 1000],  # The list can be adjusted based on computational capacity
    "max_depth": [3, 4, 5, 6, 7, 8, 9],  # Depth of the tree
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # Subsample ratio of the training instances
    "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Subsample ratio of columns when constructing each tree
}

xgb = XGBClassifier(objective='binary:logistic', silent=True, nthread=1)
folds = 5
param_comb = 64

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=8, cv=skf.split(train_x,train_y), verbose=3, random_state=1001)
random_search.fit(train_x, train_y)


# In[11]:


# Perform predictions on test data
model = random_search
submission_pred = model.predict(test_data)
submission_pred = ['True' if value == 1 else 'False' for value in submission_pred]
submission = pd.DataFrame()
submission['PassengerId'] = test_ids
submission['Transported'] = submission_pred


# In[12]:


# Save the submission file
from pathlib import Path  
filepath = Path('submission.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv('submission.csv', sep=',', index=False)

