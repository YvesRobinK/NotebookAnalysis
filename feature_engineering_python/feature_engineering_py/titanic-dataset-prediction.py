#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### Importing Libraries


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns,set()


# #### Importing Dataset

# In[3]:


titanic_data=pd.read_csv("../input/titanic/train.csv")
titanic_test=pd.read_csv("../input/titanic/test.csv")


# ### Performing Exploratory Data Analysis

# In[4]:


titanic_data.head()


# In[5]:


titanic_data.shape


# In[6]:


titanic_data.columns


# #### Checking various null entries in the dataset, with the help of heatmap

# In[7]:


titanic_data.isnull().sum()


# #### Visualization of various relationships between variables

# In[8]:


sns.countplot(x='Survived', data=titanic_data)


# In[9]:


sns.countplot(x='Survived',hue='Sex', data=titanic_data)


# In[10]:


sns.countplot(x='Survived',hue='Pclass', data=titanic_data)


# #### Replacing null values in Age column using function
# 

# In[11]:


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
        


# In[12]:


df=titanic_data


# In[13]:


df['Age']=df[['Age','Pclass']].apply(add_age,axis=1)


# #### Convert sex and embarked columns to numerical values

# In[14]:


df.Sex=df.Sex.map({'female':0, 'male':1})
df.Embarked=df.Embarked.map({'S':0, 'C':1, 'Q':2, 'nan':'NaN'})


# #### Dropping Null Data

# In[15]:


df.drop('Cabin',axis=1,inplace=True)


# Removing rows with null values

# In[16]:


df.dropna(inplace=True)


# In[17]:


df.drop(['Name', 'PassengerId', 'Ticket'], axis = 1, inplace = True)


# #### Feature engineering

# In[18]:


min_age=min(df.Age)
max_age=max(df.Age)
min_fare=min(df.Fare)
max_fare=max(df.Fare)


# In[19]:


df.Age = (df.Age-min_age)/(max_age-min_age)
df.Fare = (df.Fare-min_fare)/(max_fare-min_fare)


# In[ ]:





# #### Print the finalised data

# In[20]:


df.head()


# #### Split the data set into x and y data

# In[21]:


x_data=df.drop('Survived',axis=1)
y_data=df['Survived']


# #### Split the data set into training data and test data

# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x_data, y_data, test_size = 0.2, random_state=0, stratify=y_data)


# #### Create the model

# In[24]:


from sklearn.linear_model import LogisticRegression


# In[25]:


model = LogisticRegression()


# #### Train the model and create predictions

# In[26]:


model.fit(x_training_data, y_training_data)
predictions = model.predict(x_test_data)


# #### Let’s see how accurate is our model for predictions:

# In[27]:


from sklearn.metrics import classification_report


# In[28]:


print(classification_report(y_test_data, predictions))


# In[29]:


from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test_data, predictions))


# #### Let’s see the confusion matrix

# In[30]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test_data, predictions)


# ### confusion matrix using seaborn

# In[31]:


cf_matrix=confusion_matrix(y_test_data, predictions)


# In[32]:


import seaborn as sns

sns.heatmap(cf_matrix, annot=True)


# #### Cleaning test datset

# In[33]:


df1=titanic_test


# In[34]:


df1.head()


# In[35]:


df1.isnull().sum()


# In[36]:


df1['Age']=df1[['Age','Pclass']].apply(add_age,axis=1)


# In[37]:


df1['Fare']=df1['Fare'].fillna(df1['Fare'].median())


# In[38]:


df1.Sex=df1.Sex.map({'female':0, 'male':1})
df1.Embarked=df1.Embarked.map({'S':0, 'C':1, 'Q':2, 'nan':'NaN'})


# In[39]:


min_age1=min(df1.Age)
max_age1=max(df1.Age)
min_fare1=min(df1.Fare)
max_fare1=max(df1.Fare)


# In[40]:


df1.Age = (df1.Age-min_age1)/(max_age1-min_age1)
df1.Fare = (df1.Fare-min_fare1)/(max_fare1-min_fare1)


# In[41]:


df1.drop(['Cabin','PassengerId','Name','Ticket'],axis=1,inplace=True)


# In[42]:


df1.head()


# #### Prediction

# In[43]:


prediction=model.predict(df1)


# In[44]:


prediction


# In[45]:


test=pd.read_csv('../input/titanic/test.csv')


# In[46]:


submission = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": prediction})
submission.to_csv('submission.csv', index=False)


# In[47]:


pred_df = pd.read_csv('submission.csv')


# In[48]:


sns.countplot(x='Survived', data=pred_df)


# In[ ]:





# In[ ]:




