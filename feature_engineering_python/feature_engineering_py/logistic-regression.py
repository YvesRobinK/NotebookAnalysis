#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import FileLink

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[3]:


train_data


# In[4]:


train_data.isnull().sum()


# In[5]:


train_data = train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)


# In[6]:


train_data


# In[7]:


# Imputers for Age and Embarked
age_imputer = SimpleImputer(strategy='median')
embarked_imputer = SimpleImputer(strategy='most_frequent')


# In[8]:


# Applying imputers
train_data['Age'] = age_imputer.fit_transform(train_data[['Age']])
train_data['Embarked'] = embarked_imputer.fit_transform(train_data[['Embarked']])


# In[9]:


train_data['Sex'].unique()


# In[10]:


# Converting categorical data to numeric

train_data['Sex'] = train_data['Sex'].replace('male', 0)
train_data['Sex'] = train_data['Sex'].replace('female', 1)
train_data['Sex'] = train_data['Sex'].astype(int)

train_data['Embarked'] = train_data['Embarked'].replace('S', 0)
train_data['Embarked'] = train_data['Embarked'].replace('C', 1)
train_data['Embarked'] = train_data['Embarked'].replace('Q', 2)
train_data['Embarked'] = train_data['Embarked'].astype(int)


# In[11]:


train_data


# In[12]:


# Splitting the dataset into training and testing sets
y = train_data['Survived']
train_data = train_data.drop(['Survived'], axis=1)
X = train_data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[13]:


# Train the logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predict on the test set
y_pred = log_reg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

accuracy, report


# In[14]:


# Generating a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix of Logistic Regression Model')
plt.show()


# # Testing

# In[15]:


test_data


# In[16]:


test_data.isnull().sum()


# In[17]:


# We need PassengerID before dropping it
test_output = pd.DataFrame()

test_output['PassengerId'] = test_data['PassengerId']


# In[18]:


test_data = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

fare_imputer = SimpleImputer(strategy='median')
test_data['Fare'] = fare_imputer.fit_transform(test_data[['Fare']])

# Applying imputers
test_data['Age'] = age_imputer.fit_transform(test_data[['Age']])
test_data['Embarked'] = embarked_imputer.fit_transform(test_data[['Embarked']])


# Converting categorical data to numeric

test_data['Sex'] = test_data['Sex'].replace('male', 0)
test_data['Sex'] = test_data['Sex'].replace('female', 1)
test_data['Sex'] = test_data['Sex'].astype(int)

test_data['Embarked'] = test_data['Embarked'].replace('S', 0)
test_data['Embarked'] = test_data['Embarked'].replace('C', 1)
test_data['Embarked'] = test_data['Embarked'].replace('Q', 2)
test_data['Embarked'] = test_data['Embarked'].astype(int)


# In[19]:


test_data


# In[20]:


# Now, we can make predictions using the logistic regression model
predicted_survival = log_reg.predict(test_data)


# In[21]:


predicted_survival


# In[22]:


test_output['Survived'] = predicted_survival


# In[23]:


test_output.to_csv("output.csv", index = False)


# In[24]:


FileLink("output.csv")


# In[ ]:




