#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd
from sklearn.preprocessing import StandardScaler# data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


train=pd.read_csv('../input/titanic/train.csv') # reading train data
test=pd.read_csv('../input/titanic/test.csv')


# In[3]:


train.head()


# In[4]:


train.shape


# In[5]:


train.columns


# In[6]:


train.isnull().sum()


# In[7]:


train["Sex"].value_counts()


# In[8]:


#Feature extraction
titles = set()
for names in train['Name']:
    titles.add(names.split(",")[1].split(".")[0].strip(" "))
print(titles)


# In[9]:


title_dict={'Mrs':'Mrs','Major':'Other','Master':'Master','Lady':'Other','Mlle':'Miss','Dr':'Other','Col':'Other','Capt':'Other','Don':'Other','the Countess':'Other','Mme':'Mrs','Miss':'Miss','Jonkheer':'Other','Rev':'Other','Sir':'Other','Ms':'Miss','Mr':'Mr'}


# In[10]:


train_test = [train,test]
for dataset in train_test:
    dataset["Title"] = dataset["Name"].map(lambda name:name.split(",")[1].split(".")[0].strip())
    dataset["Title"] = dataset['Title'].map(title_dict)
train.head()


# In[11]:


train.tail()


# In[12]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
for dataset in train_test:
    dataset["Title"] = dataset["Title"].map(title_mapping)


# In[13]:


train["Title"].isnull().sum()
test["Title"].isnull().sum()
test["Title"]= test["Title"].fillna(0)


# In[14]:


test["Title"].isnull().sum()


# In[15]:


#Cleaning the dataset
df1 = train.drop(["Name","Ticket","Cabin","PassengerId","Embarked"], axis =1)
df1.head()


# In[16]:


df1.isnull().sum()


# In[17]:


df1.Sex=df1.Sex.map({'female':0, 'male':1})
df1.head()


# In[18]:


df1.isnull().sum()


# In[19]:


df1.head()


# In[20]:


df1.isnull().sum()


# In[21]:


medianage_female = df1[df1["Sex"]==0]["Age"].median()
medianage_male = df1[df1["Sex"]==1]["Age"].median()


# In[22]:


df1.loc[(df1["Sex"]==0) & (df1["Age"].isnull()), "Age"] = medianage_female
df1.loc[(df1["Sex"]==1) & (df1["Age"].isnull()), "Age"] = medianage_male


# In[23]:


df1.isnull().sum()


# In[24]:


df1.head()


# In[25]:


test.head()


# In[26]:


#Feature Scaling
df1.Age=(df1.Age-min(df1.Age))/(max(df1.Age)-min(df1.Age))
df1.Fare=(df1.Fare-min(df1.Fare))/(max(df1.Fare)-min(df1.Fare))



# In[27]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df1.drop(["Survived"], axis=1), df1.Survived, test_size = 0.2 , random_state=0, stratify = df1.Survived) 


# In[28]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 100,max_depth = 5,random_state=1)
clf.fit(df1.drop(["Survived"],axis=1),df1["Survived"])
#clf = LogisticRegression()
#clf.fit(df1.drop(["Survived"],axis=1),df1["Survived"])


# In[29]:


test.head()


# In[30]:


df2 = test.drop(["Name","Ticket","Cabin","PassengerId","Embarked"], axis =1)
df2.head()
df2.Sex=df2.Sex.map({'female':0, 'male':1})

df2.head()


# In[31]:


df2.isnull().sum()


# In[32]:


median_age_men2=df2[df2['Sex']==1]['Age'].median()
median_age_women2=df2[df2['Sex']==0]['Age'].median()
df2.loc[(df2.Age.isnull()) & (df2['Sex']==0),'Age']=median_age_women2
df2.loc[(df2.Age.isnull()) & (df2['Sex']==1),'Age']=median_age_men2
df2['Fare']=df2['Fare'].fillna(df2['Fare'].median())

df2.isnull().sum()



# In[33]:


df2.head()


# In[34]:


df2['Age']=(df2.Age-min(df2.Age))/(max(df2.Age)-min(df2.Age))
df2['Fare']=(df2.Fare-min(df2.Fare))/(max(df2.Fare)-min(df2.Fare))


# In[35]:


pred = clf.predict(df2)


# In[36]:


pred


# In[37]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": pred
    })
submission.to_csv('submission2.csv', index=False)


# In[ ]:




