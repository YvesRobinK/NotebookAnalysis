#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[2]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn import tree


# In[3]:


train_data=pd.read_csv('../input/titanic/train.csv')
test_data=pd.read_csv('../input/titanic/test.csv')


# In[4]:


train_data.head()


# In[5]:


# Cleaning the Data
train_data.isnull().sum()
# Replace the age by the mean then remove the NaN


# In[6]:


train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
train_data['Age'].isnull().sum()


# In[7]:


# Using feature engineering to get a better gist of the Namme column
train_data['Name'].unique().tolist()


# In[8]:


titles=['Mrs','Mr','Master','Miss','Major','Rev','Dr','Ms','Mlle','Col','Capt','Mme','Countess','Dona','Jonkheer']


# In[9]:


# Feature Engineering to exract the title from the Name column
def get_title(name):
    for title in titles:
        if title in name:
            return title
        else:
            return 'None'


# In[10]:


# Created a new column called title and used lambda to assign titles to all the Passennger
train_data['Title']=train_data['Name'].apply(lambda x:get_title(x))
train_data.head()


# In[11]:


# Dropping the Cabin values with 687 Na values
train_data.drop('Cabin',axis=1,inplace=True)


# In[12]:


train_data['Embarked']=train_data['Embarked'].fillna('S')


# In[13]:


# Repalcing with the mode S
train_data.head()


# In[14]:


X_train=train_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title']]
y_train=train_data['Survived']
# Here we are taking the columnns that might actually affect the survival


# In[15]:


# One Hot encoding all the Categorical columns to bring them to numerical form
X_train=pd.get_dummies(X_train)
X_train.head()


# In[16]:


# Creating the model
model=RandomForestClassifier(max_depth=75,random_state=0)
# Fitting the Model
model.fit(X_train,y_train)


# In[17]:


model1=AdaBoostClassifier(n_estimators=100, random_state=0)
# Fitting the Model
model1.fit(X_train,y_train)


# In[18]:


model2=tree.DecisionTreeClassifier()
# Fitting the Mode
model2.fit(X_train,y_train)


# In[19]:


# Preprocessing the Test Data as well in a similar way
test_data.isnull().sum()


# In[20]:


# Created a new column called title and used lambda to assign titles to all the Passennger
test_data['Title']=test_data['Name'].apply(lambda x:get_title(x))
test_data.head()


# In[21]:


test_data.drop('Cabin',axis=1,inplace=True)


# In[22]:


test_data['Embarked']=test_data['Embarked'].fillna('S')


# In[23]:


test_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].isnull().sum()


# In[24]:


# For the fare column
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)


# In[25]:


X_test=test_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title']]


# In[26]:


# One Hot Encoding
X_test=pd.get_dummies(X_test)
X_test.head()


# In[27]:


total_titles=train_data['Title'].unique().tolist()
test_titles=test_data['Title'].unique().tolist()
for title in total_titles:
    if title not in test_titles:
        X_test['Title'+str(title)]=0


# In[28]:


Survived=model.predict(X_test)


# In[29]:


Survived1=model1.predict(X_test)


# In[30]:


Survived2=model2.predict(X_test)


# In[31]:


Survived,Survived1,Survived2


# In[32]:


Final_Survived=(Survived+Survived1+Survived2)/3
Final_Survived.astype(int)


# In[33]:


prediction=pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':Survived1})
prediction.to_csv('submission.csv',index=False)


# In[ ]:




