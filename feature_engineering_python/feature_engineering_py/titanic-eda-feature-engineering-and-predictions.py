#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


#importing data
data_train = pd.read_csv("/kaggle/input/titanic/train.csv")
data_test = pd.read_csv("/kaggle/input/titanic/test.csv")
y = data_train.Survived


# In[3]:


data_train.head()


# # Exploratory Data Analysis
# 

# In[4]:


data_train.info()


# In[5]:


data_test.info()


# * Cabin: most no. of missing values
# * Age: A few missing values can be imputed
# * Embarked has 2 and Fare has 1 missing value

# In[6]:


data_train.describe()


# * Most were 3rd class passengers
# * only 38.38% survived
# * Most people were without family

# ## Effect of various features on Survival
# 

# In[7]:


data_train.groupby('Pclass').Survived.mean()


# 1st class was preffered to be saved

# In[8]:


data_train.groupby('Sex').Survived.mean()


# Females and children are usually oreffered during rescue, thats clearly visible

# In[9]:


data_train.groupby('SibSp').Survived.agg(['mean','count'])


# In[10]:


data_train.groupby('Parch').Survived.agg(['mean','count'])


# In[11]:


data_train.groupby('Embarked').Survived.mean()


# Seems like people from CHEROBOURG had luck in their favour, half of em were saved

# ## Survival by Age

# In[12]:


survived_by_age=[data_train[data_train.Age<i].Survived.mean() for i in range(5,int(data_train.Age.max()),5)]
for i in range(len(survived_by_age)):
    print("Age "+str(i*5)+" to "+str((i+1)*5)+", "+"Survival Rate: "+str(survived_by_age[i]))


# * More children were priorotized to be saved
# * For ages > 20 chances for survival were more or less the sam 40%

# ## A whole report in one line..

# In[13]:


import pandas_profiling
pandas_profiling.ProfileReport(data_train)


# # Featue Engineering

# 1. Let's drop the name and ticket columns, they're irrelevant and we dont need'em much!!

# In[14]:


X_train = data_train.drop(['Name','Ticket','PassengerId'],axis=1)
X_test = data_test.drop(['Name','Ticket','PassengerId'],axis=1)


# In[15]:


X_train.head()


# Let's now check for missing values

# In[16]:


sns.heatmap(X_train.isnull(),cbar=False)


# In[17]:


sns.heatmap(X_test.isnull(),cbar=False)


# cabin is missing more than half of times, its better to drop it instead of imputing it with other values

# In[18]:


X_train = X_train.drop(['Cabin'],axis=1)
X_test = X_test.drop(['Cabin'],axis=1)


# In[19]:


X_train[X_train.Embarked.isnull()]


# In[20]:


X_train.groupby('Embarked').Embarked.count()


# In[21]:


plt.figure(figsize=(16,9))
sns.FacetGrid(X_train,col = 'Embarked',row = 'Sex',height = 4, aspect = .8).map(sns.barplot,'Pclass','Fare',order=[1,2,3])
plt.show()


# Clearly showing that Female with rs 80 fare will be S embarked 

# In[22]:


X_train.Embarked=X_train.Embarked.fillna('S')


# In[23]:


X_train[X_train.Embarked.isnull()]


# In[24]:


plt.figure(figsize=(14,6))
sns.scatterplot(x=X_train.Age,y=X_train.Fare,hue=X_train.Pclass)


# In[25]:


X_train[X_train.Pclass==1].Age.mean()


# In[26]:


X_train[X_train.Pclass==2].Age.mean()


# In[27]:


X_train[X_train.Pclass==3].Age.mean()


# In[28]:


def impute(cols):
    Age = cols[0]
    Pclass = cols[1]
    if(pd.isnull(Age)):
        if Pclass==1:
            return 38
        elif Pclass==2:
            return 30
        else:
            return 25
    return Age


# In[29]:


X_train.Age = X_train[['Age','Pclass']].apply(impute,axis=1)
X_test.Age = X_test[['Age','Pclass']].apply(impute,axis=1)


# In[30]:


X_test[X_test.Fare.isnull()]


# In[31]:


X_train[(X_train.Sex=='male')&(X_train.Pclass==3)].Fare.mean()


# In[32]:


X_test['Fare']=X_test['Fare'].fillna(13)


# In[33]:


sns.heatmap(X_train.isnull(),cbar=False)


# In[34]:


sns.heatmap(X_test.isnull(),cbar=False)


# Full black, means all missing values are handled

# ## Categorical Feature Encoding

# In[35]:


X_train = X_train.drop(['Survived'],axis=1)


# In[36]:


X_test.groupby('Sex').Sex.count()


# In[37]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , LabelEncoder


# In[38]:


le =LabelEncoder()
X_train['Sex']=le.fit_transform(X_train['Sex'])
X_test['Sex'] = le.fit_transform(X_test['Sex'])
X_train.head()


# In[39]:


onh = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_train_trans = pd.DataFrame(onh.fit_transform(X_train[['Embarked']]))
X_test_trans = pd.DataFrame(onh.fit_transform(X_test[['Embarked']]))
X_train_trans.index = X_train.index
X_test_trans.index = X_test.index
X_train_conc = X_train.drop(['Embarked'],axis=1)
X_test_conc = X_test.drop(['Embarked'],axis=1)
X_train_final = pd.concat([X_train_conc,X_train_trans],axis=1)
X_test_final = pd.concat([X_test_conc,X_test_trans],axis=1)


# In[40]:


X_train_final.head()


# In[41]:


X_test_final.head()


# In[42]:


X_train_final.info()


# In[43]:


X_test_final.info()


# ## Feature Scaling

# In[44]:


from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train_final = sc.fit_transform(X_train_final)
X_test_final = sc.transform(X_test_final)


# # Model Prediction

# In[45]:


from sklearn.svm import SVC
clf = SVC(kernel='rbf', degree = 5)
clf.fit(X_train_final,y)


# In[46]:


pred = clf.predict(X_test_final)
output = pd.DataFrame({'PassengerId':data_test.PassengerId,'Survived':pred})
output.to_csv('submission.csv',index=False)


# In[47]:


output


# ## Submitted with top 20% performance can be tuned to perform more better

# # If found informative, upvote and give feedback, Thanks and Keep coding
