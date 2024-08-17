#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns,set()


# ## Importing Dataset

# In[2]:


titanic_data=pd.read_csv("../input/titanic/train.csv")
titanic_test=pd.read_csv("../input/titanic/test.csv")


# ## Performing Exploratory Data Analysis

# In[3]:


titanic_data.head()


# In[4]:


titanic_data.shape


# In[5]:


titanic_data.columns


# #### Checking various null entries in the dataset, with the help of heatmap

# In[6]:


titanic_data.isnull().sum()


# #### Visualization of various relationships between variables

# In[7]:


sns.countplot(x='Survived', data=titanic_data)


# In[8]:


sns.countplot(x='Survived',hue='Sex', data=titanic_data)


# In[9]:


sns.countplot(x='Survived',hue='Pclass', data=titanic_data)


# #### Replacing null values in Age column using function
# 

# In[10]:


def add_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return titanic_data[titanic_data['Pclass']==1]['Age'].mean()
        elif Pclass==2:
            return titanic_data[titanic_data['Pclass']==2]['Age'].mean()
        elif Pclass==3:
            return titanic_data[titanic_data['Pclass']==3]['Age'].mean()
    else:
        return Age
        


# In[11]:


df=titanic_data


# In[12]:


df['Age']=df[['Age','Pclass']].apply(add_age,axis=1)


# ## Feature engineering

# In[13]:


df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')


# #### Combining sibsip and parch to alone

# In[14]:


#adding a new column which will define Family size

df['FamilySize'] = df.Parch + df.SibSp+1


# #### Convert sex and embarked, title columns to numerical values

# In[15]:


df.Sex=df.Sex.map({'female':1, 'male':0})
df.Embarked=df.Embarked.map({'S':0, 'C':1, 'Q':2, 'nan':'NaN'})
df.Title=df.Title.map({'Mr':0, 'Miss':1, 'Mrs':2,'Master':3,'Rare':4})


# #### Dropping Null Data

# In[16]:


df.drop('Cabin',axis=1,inplace=True)


# Removing rows with null values

# In[17]:


df.dropna(inplace=True)


# In[18]:


df.drop(['Name', 'PassengerId', 'Ticket','SibSp', 'Parch'], axis = 1, inplace = True)


# ## Feature scaling

# In[19]:


min_age=min(df.Age)
max_age=max(df.Age)
min_fare=min(df.Age)
max_fare=max(df.Age)


# In[20]:


df.Age = (df.Age-min_age)/(max_age-min_age)
df.Fare = (df.Fare-min_fare)/(max_fare-min_fare)


# In[ ]:





# #### Print the finalised data

# In[21]:


df.head()


# #### Split the data set into x and y data

# In[22]:


x_data=df.drop('Survived',axis=1)
y_data=df['Survived']


# #### Split the data set into training data and test data

# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x_data, y_data, test_size = 0.2, random_state=0, stratify=y_data)


# ## Create the model

# In[25]:


from sklearn.linear_model import LogisticRegression


# In[26]:


model = LogisticRegression(max_iter=1000)


# #### Train the model and create predictions

# In[27]:


model.fit(x_training_data, y_training_data)
predictions = model.predict(x_test_data)


# #### Let’s see how accurate is our model for predictions:

# In[28]:


from sklearn.metrics import classification_report


# In[29]:


print(classification_report(y_test_data, predictions))


# In[30]:


from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test_data, predictions))


# #### Let’s see the confusion matrix

# In[31]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test_data, predictions)


# ### confusion matrix using seaborn

# In[32]:


cf_matrix=confusion_matrix(y_test_data, predictions)


# In[33]:


import seaborn as sns
sns.heatmap(cf_matrix, annot=True)


# #### Cleaning test datset

# In[34]:


df1=titanic_test.copy(deep=True)


# In[35]:


df1.head()


# In[36]:


df1.isnull().sum()


# In[37]:


df1['Age']=df1[['Age','Pclass']].apply(add_age,axis=1)


# #### Feature Engineering

# In[38]:


df1['Title'] = df1['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
df1['Title'] = df1['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df1['Title'] = df1['Title'].replace('Mlle', 'Miss')
df1['Title'] = df1['Title'].replace('Ms', 'Miss')
df1['Title'] = df1['Title'].replace('Mme', 'Mrs')


# In[39]:


#adding a new column which will define the familysize

df1['FamilySize'] = df1.Parch + df1.SibSp+1


# In[40]:


df1['Fare']=df1['Fare'].fillna(df1['Fare'].median())


# In[41]:


df1.Sex=df1.Sex.map({'female':1, 'male':0})
df1.Embarked=df1.Embarked.map({'S':0, 'C':1, 'Q':2, 'nan':'NaN'})
df1.Title=df1.Title.map({'Mr':0, 'Miss':1, 'Mrs':2,'Master':3,'Rare':4})


# In[42]:


min_age1=min(df1.Age)
max_age1=max(df1.Age)
min_fare1=min(df1.Age)
max_fare1=max(df1.Age)


# In[43]:


df1.Age = (df1.Age-min_age1)/(max_age1-min_age1)
df1.Fare = (df1.Fare-min_fare1)/(max_fare1-min_fare1)


# In[44]:


df1.drop(['Cabin','PassengerId','Name','Ticket','SibSp', 'Parch'],axis=1,inplace=True)


# In[45]:


df1.head()


# In[46]:


df1.isnull().sum()


# #### Prediction

# In[47]:


prediction=model.predict(df1)


# In[48]:


prediction


# In[49]:


submission = pd.DataFrame({"PassengerId": titanic_test["PassengerId"],"Survived": prediction})
submission.to_csv('submission.csv', index=False)


# In[ ]:




