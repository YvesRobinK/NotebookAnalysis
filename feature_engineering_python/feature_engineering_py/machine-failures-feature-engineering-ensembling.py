#!/usr/bin/env python
# coding: utf-8

# ## Library & Data Import

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import numpy as np


# In[3]:


train = pd.read_csv('/kaggle/input/playground-series-s3e17/train.csv')
train.columns = train.columns.str.replace('[\[\]]', '', regex=True)
train.drop(['Product ID'], axis=1, inplace=True)
train['Tool wear min'] = train['Tool wear min'].astype(float)
train['Rotational speed rpm'] = train['Rotational speed rpm'].astype(float)

train.set_index('id', inplace=True)



# ## Initial EDA

# In[4]:


train.head()


# In[5]:


test = pd.read_csv('/kaggle/input/playground-series-s3e17/test.csv')
test.columns = test.columns.str.replace('[\[\]]', '', regex=True)
test.drop(['Product ID'], axis=1, inplace=True)
test['Tool wear min'] = test['Tool wear min'].astype(float)
test['Rotational speed rpm'] = test['Rotational speed rpm'].astype(float)

test.set_index('id', inplace=True)


test.head()


# ## Feature Engineering

# In[6]:


# Define the feature engineering steps
    
# Create a new feature: 'Power' (multiplication of 'Rotational speed' and 'Torque')
train['Power'] = train['Rotational speed rpm'] * train['Torque Nm']
    
# Create additional features
train['Temp_diff'] = train['Process temperature K'] - train['Air temperature K']  # Temperature difference
train['Speed_to_Torque_ratio'] = train['Rotational speed rpm'] / train['Torque Nm']  # Speed to torque ratio
train['Temp_sum'] = train['Air temperature K'] + train['Process temperature K']  # Temperature sum
train['Wear_rate'] = train['Tool wear min'] / train['Rotational speed rpm'] # Wear rate

train


# In[7]:


# Check for missing values
missing_values = train.isnull().sum()
print(missing_values)


# ## Data Preprocessing

# In[8]:


from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Apply ordinal encoding to the 'Type' column
ordinal_encoder = OrdinalEncoder()
train['Type'] = ordinal_encoder.fit_transform(train[['Type']])

# Select the float columns
float_columns = train.select_dtypes(include='float').columns

# Scale the float columns using StandardScaler
scaler = StandardScaler()
train[float_columns] = scaler.fit_transform(train[float_columns])


# ## Train Test Split

# In[9]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Splitting the transformed data into X and y
X = train.drop(['Machine failure'], axis=1)
y = train['Machine failure']

# Splitting X and y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[10]:


X_train


# In[11]:


y_train


# ## Models

# ### Logistic Regression

# In[12]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train, y_train)


# ### Random Forest Classifier

# In[13]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)


# ### Gradient Boosting Classifier

# In[14]:


from sklearn.ensemble import GradientBoostingClassifier

gbm = GradientBoostingClassifier()
gbm.fit(X_train, y_train)


# ### Support Vector Classifier

# In[15]:


from sklearn.svm import SVC

svc = SVC(kernel='rbf', C=1000, probability=True)
svc.fit(X_train, y_train)


# ### XGBoost Classifier

# In[16]:


from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_train, y_train)


# In[17]:


from lightgbm import LGBMClassifier

lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)


# ### CatBoost Classifier

# In[18]:


from catboost import CatBoostClassifier

catboost = CatBoostClassifier(verbose=False)
catboost.fit(X_train, y_train);


# ## Model Ensembling

# In[19]:


from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score

ensemble = VotingClassifier(estimators=[
    ('logreg', logreg),
    ('rfc', rfc),
    ('gbm', gbm),
    ('svc', svc),
    ('xgb', xgb),
    ('lgbm', lgbm),
    ('catboost', catboost)
], voting='soft')

ensemble.fit(X_train, y_train)
ensemble_preds = ensemble.predict_proba(X_test)[:, 1]
ensemble_roc_auc = roc_auc_score(y_test, ensemble_preds)


# In[20]:


print(ensemble_roc_auc)


# ## Test Preprocessing

# In[21]:


# Define the feature engineering steps
    
# Create a new feature: 'Power' (multiplication of 'Rotational speed' and 'Torque')
test['Power'] = test['Rotational speed rpm'] * test['Torque Nm']
    
# Create additional features
test['Temp_diff'] = test['Process temperature K'] - test['Air temperature K']  # Temperature difference
test['Speed_to_Torque_ratio'] = test['Rotational speed rpm'] / test['Torque Nm']  # Speed to torque ratio
test['Temp_sum'] = test['Air temperature K'] + test['Process temperature K']  # Temperature sum
test['Wear_rate'] = test['Tool wear min'] / test['Rotational speed rpm'] # Wear rate


test


# In[22]:


from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()

test['Type'] = ordinal_encoder.fit_transform(test[['Type']])

# Select the float columns
float_columns = test.select_dtypes(include='float').columns

# Scale the float columns using StandardScaler
scaler = StandardScaler()
test[float_columns] = scaler.fit_transform(test[float_columns])

test


# In[23]:


# Check for missing values
missing_values = test.isnull().sum()
print(missing_values)


# ## Submit

# In[24]:


# Make predictions on the test dataframe
test_preds = ensemble.predict_proba(test)[:, 1]

# Create a submission dataframe with 'id' and 'Machine failure' columns
submission_df = pd.DataFrame({'id': test.index, 'Machine failure': test_preds})

# Save the submission dataframe as a CSV file
submission_df.to_csv('submission.csv', index=False)

