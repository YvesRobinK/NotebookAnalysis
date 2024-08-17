#!/usr/bin/env python
# coding: utf-8

# I wrote this notebook using the following notebook:
# 
# * https://www.kaggle.com/code/faeghehgh/linear-regression-with-few-features
# 
# In this notebook, the accuracy of the model has been improved by adding new features (Feature Engineering).

# ## Import the Libraries and Dataset

# In[1]:


import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')


# In[2]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


Train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
Test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
All_data = pd.concat([Train_data, Test_data], sort=True).reset_index(drop=True)


# #### Data Overview

# In[4]:


print('Training Shape = {}'.format(Train_data.shape))
print('Test Shape = {}'.format(Test_data.shape))
print('Name of columns in Training dataframe = {}'.format(Train_data.columns))
print('Name of columns in Test dataframe = {}'.format(Test_data.columns))


# In[5]:


Train_data.head()


# In[6]:


Train_data.info()


# In[7]:


Test_data.head()


# In[8]:


Test_data.info()


# ## Exploratory Data Analysis
# 
# To better determine the correlation of the features and increase the accuracy of the model, it is necessary to fill the NAN values related to each feature correctly.

# In[9]:


print('missing values of Train ')
print('\n')
for column in Train_data.columns.tolist():          
    print('{} column: {}'.format(column, Train_data[column].isnull().sum()))


# In[10]:


print('missing values of Test ')
print('\n')
for column in Test_data.columns.tolist():          
    print('{} column: {}'.format(column, Test_data[column].isnull().sum()))


# According to the above results, some values of Age, Cabin, Fare, and Embarked features are NaN. I filled these values as follows. Note: For a full description, you can refer to the following notebook.
# 
# * https://www.kaggle.com/code/faeghehgh/linear-regression-with-few-features

# In[11]:


All_data['Age'] = All_data.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
All_data['Embarked'] = All_data['Embarked'].fillna('S')
Class = All_data.groupby(['Pclass'])['Fare'].mean()
All_data['Fare'] = All_data['Fare'].fillna(Class[3])


# In[12]:


check_nan = All_data['Cabin'].isnull()
All_data['newCabin']=np.where(check_nan == False, All_data['Cabin'].astype(str).str[0],0)
All_data['newCabin']


# ## Feature Engineering
# 
# I added 4 new features to this notebook using the available features, which can help to accurately predict the target feature values.
# 
# **1. Title**
# Title is created by extracting the prefix before Name feature. These values are included Mr, Mrs, Miss, Master, Don, Rev, Dr, Mme, Ms, Major, Lady, Sir, Mlle, Col, Capt, Countess, Jonkheer, and Dona. I put these values in the Title column as follows:
# 
#     * 'Ms', 'Mrs', 'Mme', 'Mlle', 'Miss', 'Lady', 'Dona', 'Countess' --> Female
#     * 'Sir', 'Mr', 'Major', 'Jonkheer', 'Don' --> Male
#     * 'Rev', 'Dr', 'Col', 'Capt' --> Other
# **2. Is_Married**
# Is_Married is a binary feature based on the Mrs title.

# In[13]:


re = '([A-Za-z]+)\.'
for data in All_data['Name']:
    All_data['Title'] = All_data.Name.str.extract(re, expand=False)
All_data.Title.unique()


# In[14]:


All_data['Is_Married'] = 0
All_data['Is_Married'].loc[All_data['Title'] == 'Mrs'] = 1


# In[15]:


All_data.groupby(['Title','Sex'])['Title'].count()


# In[16]:


female_titles = ['Ms', 'Mrs', 'Mme', 'Mlle', 'Miss', 'Lady', 'Dona', 'Countess']
male_titles = ['Sir', 'Mr', 'Major', 'Jonkheer', 'Don']
other_titles = ['Rev', 'Dr', 'Col', 'Capt']
for data in All_data['Title']:
    All_data['Title'] = All_data['Title'].replace(female_titles, 'Female')
    All_data['Title'] = All_data['Title'].replace(male_titles, 'Male')
    All_data['Title'] = All_data['Title'].replace(other_titles, 'Other')
All_data


# **3. Age Range**

# In[17]:


c1 = 10
c2 = 20
y1 = 30
y2 = 40
ag1 = 50
ag2 = 60
old= 70

def discretize_Age (Age) :
    if Age < c1 : 
        return "0"
    elif Age < c2 :
        return "1"
    elif Age < y1 :
        return "2"
    elif Age < y2 :
        return "3"
    elif Age < ag1 :
        return "4"
    elif Age < ag2 :
        return "5"
    elif Age < old :
        return "6"
    else :
        return "7"

All_data['AgeRange'] = (All_data['Age'].apply(discretize_Age)).astype('int')
All_data


# **4. Family**

# In[18]:


All_data['Family'] = All_data['SibSp']+All_data['Parch']


# ### correlation

# In[19]:


All_data.drop(['Ticket','Cabin','Name'], axis=1, inplace=True)

All_data['Sex'] = pd.factorize(All_data['Sex'])[0]
All_data['Embarked'] = pd.factorize(All_data['Embarked'])[0]
All_data['newCabin'] = pd.factorize(All_data['newCabin'])[0]
All_data['Title'] = pd.factorize(All_data['Title'])[0]


# In[20]:


Train = All_data.head(891)
Test = All_data.tail(418)
Test.drop(['Survived'], axis=1, inplace=True)


# In[21]:


plt.figure(figsize=(12,10))
cor_Train = Train.corr()
sns.heatmap(cor_Train, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[22]:


plt.figure(figsize=(12,10))
cor_Test = Test.corr()
sns.heatmap(cor_Test, annot=True, cmap=plt.cm.Reds)
plt.show()


# ## Model

# In[23]:


new_features = ['Embarked', 'Sex', 'newCabin', 'Title', 'Pclass', 'Family', 'Is_Married', 'AgeRange']

X = pd.get_dummies(Train[new_features])
X_Test = pd.get_dummies(Test[new_features])
y = Train['Survived']

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X_Test)
for i in range(predictions.size):
    if predictions[i]>=0.55:
        predictions[i]=1
    else:
        predictions[i]=0
        
predictions=predictions.astype('int')

output = pd.DataFrame({'PassengerId': Test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:




