#!/usr/bin/env python
# coding: utf-8

# ## Introduction

# **The Titanic challenge** on Kaggle is a competition in which the task is to predict the survival or the death of a given passenger based on a set of variables describing him such as his age, his sex, or his passenger class on the boat. 
# 
# I have recently achieved an accuracy score of **0.78708** on the public leaderboard. I have used **Logistic regression** for prediction. As I'm writing this post, I am ranked among the **top 9%** of all Kagglers.

# ## Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Exploratory Data Analysis

# In[2]:


# reading train data
train=pd.read_csv('../input/titanic/train.csv')

# reading test data
test=pd.read_csv('../input/titanic/test.csv') 


# In[3]:


# printing first five rows of the data set
train.head()


# In[4]:


# number of rows and columns
train.shape 


# In[5]:


# column names
train.columns 


# In[6]:


# number of null values in dataset
train.isnull().sum() 


# In[7]:


train['Sex'].value_counts()


# In[8]:


# countplot
sns.countplot(x='Sex', data=train)


# In[9]:


train['Pclass'].value_counts()


# In[10]:


sns.countplot(x='Pclass', data=train)


# In[11]:


train['Embarked'].value_counts()


# In[12]:


sns.countplot(x='Embarked', data=train)


# In[13]:


train['SibSp'].value_counts()


# In[14]:


sns.countplot(x='SibSp', data=train)


# Visualizing survival based on the gender.

# In[15]:


# new feature
train['Died'] = 1 - train['Survived']


# In[16]:


train.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='bar',
                                                           figsize=(10, 5),
                                                           stacked=True)


# Plotting the same graph but with ratio instead.

# In[17]:


train.groupby('Sex').agg('mean')[['Survived', 'Died']].plot(kind='bar',
                                                            figsize=(10, 5),
                                                            stacked=True)


# The Sex variable seems to be a discriminative feature. Women are more likely to survive.

# Now, visualizing survival based on the fare.

# In[18]:


figure = plt.figure(figsize=(16, 7))
plt.hist([train[train['Survived'] == 1]['Fare'], train[train['Survived'] == 0]['Fare']], 
         stacked=True, bins = 50, label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()


# Passengers with cheaper ticket fares are more likely to die. Put differently, passengers with more expensive tickets, and therefore a more important social status, seem to be rescued first.

# ## Feature Engineering

# In[19]:


titles = set()
for name in train['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())
print(titles)


# In[20]:


Title_Dictionary = {"Capt": "Officer","Col": "Officer","Major": "Officer","Jonkheer": "Royalty","Don": "Royalty","Sir" : "Royalty","Dr": "Officer","Rev": "Officer","the Countess":"Royalty","Mme": "Mrs","Mlle": "Miss","Ms": "Mrs","Mr" : "Mr","Mrs" : "Mrs","Miss" : "Miss","Master" : "Master","Lady" : "Royalty"}


# In[21]:


train['Title'] = train['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
train['Title'] = train.Title.map(Title_Dictionary)
train.head()


# ## Cleaning the  train dataset

# In[22]:


# dropping umwanted columns
df1=train.drop(['Name','Ticket','Cabin','PassengerId','Died'], axis=1)
df1.head()


# In[23]:


train.Title.value_counts()


# In[24]:


# Converting categorical feature to numeric
df1.Sex=df1.Sex.map({'female':0, 'male':1})
df1.Embarked=df1.Embarked.map({'S':0, 'C':1, 'Q':2,'nan':'NaN'})
df1.Title=df1.Title.map({'Mr':0, 'Miss':1, 'Mrs':2,'Master':3,'Officer':4,'Royalty':5})
df1.head()


# In[25]:


# median age of each sex
median_age_men=df1[df1['Sex']==1]['Age'].median()
median_age_women=df1[df1['Sex']==0]['Age'].median()


# In[26]:


# filling null values in 'Age' with respective median age
df1.loc[(df1.Age.isnull()) & (df1['Sex']==0),'Age']=median_age_women
df1.loc[(df1.Age.isnull()) & (df1['Sex']==1),'Age']=median_age_men


# In[27]:


# checking for null values
df1.isnull().sum()


# Two null values in Embarked column

# In[28]:


# dropping rows with null value
df1.dropna(inplace=True)


# In[29]:


# Data is cleaned to have no null value
df1.isnull().sum()


# In[30]:


# cleaned dataset
df1.head()


# ## Feature Scaling

# In[31]:


df1.Age = (df1.Age-min(df1.Age))/(max(df1.Age)-min(df1.Age))
df1.Fare = (df1.Fare-min(df1.Fare))/(max(df1.Fare)-min(df1.Fare))


# In[32]:


df1.describe()


# ## Data Modelling

# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(
    df1.drop(['Survived'], axis=1),
    df1.Survived,
    test_size= 0.2,
    random_state=0,
    stratify=df1.Survived
)


# - **Logistic Regression**

# In[35]:


# Logistic regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

Y_pred = clf.predict(X_test)
accuracy_score(y_test, Y_pred)


# ## Confusion Matrix

# In[36]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, Y_pred)
cm


# In[37]:


sns.heatmap(cm,annot=True)


# ## Cleaning test dataset

# In[38]:


# test dataset
test.head()


# Different titles in the train set are:

# In[39]:


titles = set()
for name in test['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())
print(titles)


# In[40]:


test['Title'] = test['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
test['Title'] = test.Title.map(Title_Dictionary)
test.head()


# In[41]:


# dropping unwanted columns
df2=test.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)


# In[42]:


# Converting categorical feature to numeric
df2.Sex=df2.Sex.map({'female':0, 'male':1})
df2.Embarked=df2.Embarked.map({'S':0, 'C':1, 'Q':2,'nan':'nan'})
df2.Title=df2.Title.map({'Mr':0, 'Miss':1, 'Mrs':2,'Master':3,'Officer':4,'Royalty':5})
df2.head()


# In[43]:


# Checking for null values
df2.isnull().sum()


# In[44]:


# median age of each sex
median_age_men2=df2[df2['Sex']==1]['Age'].median()
median_age_women2=df2[df2['Sex']==0]['Age'].median()


# In[45]:


# filling null values with respective median age
df2.loc[(df2.Age.isnull()) & (df2['Sex']==0),'Age']=median_age_women2
df2.loc[(df2.Age.isnull()) & (df2['Sex']==1),'Age']=median_age_men2


# In[46]:


# filling null values with median fare
df2['Fare']=df2['Fare'].fillna(df2['Fare'].median())


# In[47]:


df2.isnull().sum()


# In[48]:


# Null value in the title column
df2[df2.Title.isnull()]


# Since the person is a woman with age 39, filling the title with 2 (mapped value for the title 'Mrs')

# In[49]:


df2=df2.fillna(2)


# In[50]:


# Data is cleaned to have no null value
df2.isnull().sum()


# In[51]:


# cleaned dataset
df2.head()


# In[52]:


# feature scaling
df2.Age = (df2.Age-min(df2.Age))/(max(df2.Age)-min(df2.Age))
df2.Fare = (df2.Fare-min(df2.Fare))/(max(df2.Fare)-min(df2.Fare))


# In[53]:


# test dataset
df2.head()


# ## Prediction

# In[54]:


pred = clf.predict(df2)


# In[55]:


pred


# In[56]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": pred
    })
submission.to_csv('submission.csv', index=False)


# In[57]:


pred_df = pd.read_csv('submission.csv')


# In[58]:


# visualizing predicted values
sns.countplot(x='Survived', data=pred_df)


# #### Thank You
