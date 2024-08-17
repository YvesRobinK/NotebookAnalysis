#!/usr/bin/env python
# coding: utf-8

# # Titanic Tutorial for Beginners[Accuracy: 0.789]-
# 
# 
# * This is my first tutorial. Do point out my mistakes in comment section.
# * Do upvote if you find this notebook interesting.
# * I have uploaded my second tutorial on this problem statement with better accuracy(https://www.kaggle.com/rishabhdhyani4/titanic-tutorial2). Do check it.

#  This is default first cell in any kaggle kernel. They import **NumPy** and **Pandas** libraries and it also lists the available Kernel files.** NumPy** is the fundamental package for scientific computing with Python. **Pandas** is the most popular python library that is used for data analysis.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Data Loading
# 
# 
# Our first step is to extract train and test data. We will be extracting data using pandas function read_csv. Specify the location to the dataset and import them.

# In[2]:


# Reading data
df_train=pd.read_csv('/kaggle/input/titanic/train.csv')
df_test=pd.read_csv('/kaggle/input/titanic/test.csv')
df_test_copy=df_test.copy()


# In[3]:


# Have a first look at train data
print(df_train.shape)


# By using df_train.shape we get to know that train data has 891 rows and 12 columns.

# In[4]:


# Now, lets explore first five data from training set.
df_train.head()


# We got 12 features in our training data. From https://www.kaggle.com/c/titanic/data, we have:
# 
# * Survival = Survival
# * Pclass = Ticket class
# * Sex = Sex
# * Age = Age in years
# * Sibsp = # of siblings / spouses aboard the Titanic
# * Parch = # of parents / children aboard the Titanic
# * Ticket = Ticket number
# * Fare = Passenger fare
# * Cabin = Cabin number
# * Embarked = Port of Embarkation
# 
# Qualitative Features (Categorical) : PassengerId , Pclass , Survived , Sex , Ticket , Cabin , Embarked.
# 
# Quantitative Features (Numerical) : SibSp , Parch , Age , Fare.
# 
# It is obvious from the problem statement that we have to predict **Survival** feature.

# In[5]:


# We will use describe function to calculate count,mean,max and other for numerical feature.
df_train.describe().transpose()


# In[6]:


# The feature survived contain binary data which can also be seen from its max(1) and min(0) value.


# **Our next step is to examine NULL values.**

# In[7]:


# Have a look for possible missing values
df_train.info()


# In[8]:


df_train.isnull().sum()


# In[9]:


# We see that Age, Cabin and Embarked feature have NULL values.


# In[10]:


# Have a first look at test data
print(df_test.shape)


# In[11]:


# Have a look at train and test columns
print('Train columns:', df_train.columns.tolist())
print('Test columns:', df_test.columns.tolist())


# It looks OK, the only additional column in train is 'Survived', which is our target variable, i.e. the one we want to actually predict in the test dataset.

# In[12]:


# Let's look at the figures and Understand the Survival Ratio
df_train.Survived.value_counts(normalize=True)


# In[13]:


# We observe that less people survived.


# In[14]:


# To get better understanding of count of people who survived, we will plot it.


# # Load our plotting libraries

# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


sns.countplot(x='Survived',data=df_train)


# So, out of 891 examples only 342 (38%) survived and rest all died.

# # Feature Examining-

# ## ** Pclass**
#  Come, let's examine Survival based on Pclass.

# In[17]:


sns.countplot(x='Pclass',data=df_train,hue='Survived')


# **On examining the chart above, wer can clearly say that people belonging to third class died in large numbers.**

# ##  **Sex**
# 
# Come, let's examine Survival based on gender.

# In[18]:


sns.countplot(x='Sex',data=df_train,hue='Survived')


# In[19]:


sns.catplot(x='Sex' , y='Age' , data=df_train , hue='Survived' , kind='violin' , palette=['r','g'] , split=True)


# **On examining the chart above, we can clearly say that male are more likely to die in comparision to female.**

# ## **Age**
# 
# Come, let's examine Survival based on gender.

# **We have noticed earlier that column Age has some null values. So. first we will complete the Age column and then we will analyze it.**

# In[20]:


sns.kdeplot(df_train.Age , shade=True , color='r')


# **Fill the Age with it's Median, and that is because, for a dataset with great Outliers, it is advisable to fill the Null values with median.**

# In[21]:


# To fill the missing values, we will calculate median of age with respect to Pclass.
df_train.groupby('Pclass').median()


# In[22]:


# Now we will create a function to fill missing age values. This function is used to fill the age according to Pclass.
def fill_age(cols):
    
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        
        elif Pclass == 2:
            return 29
        
        else:
            return 24
        
    else:
        
        return Age


# In[23]:


df_train['Age'] = df_train[['Age','Pclass']].apply(fill_age,axis=1)


# In[24]:


print(df_train.Age.count())  # Null values filled


# In[25]:


sns.factorplot(x='Sex',y='Age' , col='Pclass', data=df_train , hue='Survived' , kind = 'box', palette=['r','g'])


# In[26]:


# Understanding Box Plot :

# The bottom line indicates the min value of Age.
# The upper line indicates the max value.
# The middle line of the box is the median or the 50% percentile.
# The side lines of the box are the 25 and 75 percentiles respectively.


# ## **Fare**
# 
# Come, let's examine Survival based on Fare.

# In[27]:


plt.figure(figsize=(20,30))
sns.factorplot(x='Embarked' , y ='Fare' , kind='bar', data=df_train , hue='Survived' , palette=['r','g'])


# In[28]:


plt.figure(figsize=(20,10))
sns.boxplot(x='Embarked',y='Fare',data=df_train,hue='Survived')


# **We observe that people who paid more are more likely to survive.**

# ## **Embarked**
# 
# Come, let's examine Survival based on Embarked.

# **We have noticed earlier that column Embarked has some null values. So. first we will complete this column and then we will analyze it.**

# In[29]:


# The best way to fill it would be by most occured value
df_train['Embarked'].fillna(df_train['Embarked'].mode()[0] ,inplace=True)


# In[30]:


df_train.Embarked.count() # filled the values with Mode.


# ## **Cabin**
# 
# Come, let's examine Survival based on Embarked.

# In[31]:


#Since Cabin has so many missing value, we will remove that column.


# In[32]:


df_train.drop('Cabin',axis=1,inplace=True)


# In[33]:


sns.violinplot(x='Embarked' , y='Pclass' , data=df_train , hue='Survived' , palette=['r','g'])


# **We can see that those who embarked at C with First Class ticket had a good chance of Survival. Whereas for S, it seems that all classes had nearly equal probability of Survival. And for Q, third Class seems to have Survived and Died with similar probabilities.**

# In[34]:


df_train.isnull().sum()


# In[35]:


# None of the columns are empty.


# ## **SibSp**
# 
# Now lets analyze SibSp column.

# In[36]:


sns.countplot(data=df_train,x='SibSp',hue='Survived')


# In[37]:


df_train[['SibSp','Survived']].groupby('SibSp').mean()


# **It seems that there individuals having 1 or 2 siblings/spouses had the highest Probability of Survival, followed by individuals who were Alone.**

# ## **Parch**
# 
# Now lets analyze Parch column.

# In[38]:


df_train[['Parch','Survived']].groupby('Parch').mean()


# **It seems that individuals with 1,2 or 3 family members had a greater Probability of Survival, followed by individuals who were Alone.**

# **Now let us perform some feature engineering to get informative and valuable attributes.**

# # **Feature Engineering:**

# **Now let us create an attribute 'Alone' so that we could know whether the passenger is travelling alone or not.**

# In[39]:


df_train['Alone'] = 0
df_train.loc[(df_train['SibSp']==0) & (df_train['Parch']==0) , 'Alone'] = 1

df_test['Alone'] = 0
df_test.loc[(df_test['SibSp']==0) & (df_test['Parch']==0) , 'Alone'] = 1


# In[40]:


df_train.head()


# *  Now we are going to drop features which are not contributing much.
# * Names, PassengerId and Ticket Number doesn't help in finding Probability of Survival.
# * We have created Alone feature and therefore I'll be Dropping SibSp and Parch.

# In[41]:


drop_features = ['PassengerId' , 'Name' , 'SibSp' , 'Parch' , 'Ticket' ]

df_train.drop(drop_features , axis=1, inplace = True)


# In[42]:


df_test.info()


# In[43]:


# We have a few Null values in Test (Age , Fare) , let's fill it up.


# In[44]:


df_test['Fare'].fillna(df_test['Fare'].median() , inplace=True)


# In[45]:


df_test.groupby('Pclass').median()


# In[46]:


def fill_ages(cols):
    
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 42
        
        elif Pclass == 2:
            return 26.5
        
        else:
            return 24
        
    else:
        
        return Age


# In[47]:


df_test['Age'] = df_test[['Age','Pclass']].apply(fill_ages,axis=1)


# In[48]:


df_test.info()


# In[49]:


drop_featuress = ['PassengerId' , 'Name' , 'SibSp' , 'Parch' , 'Ticket','Cabin' ]

df_test.drop(drop_featuress , axis=1 , inplace = True)


# In[50]:


df_test.info()


# ### Lets convert categorical feature into numerical value.
# 
# * Sex Attribute has (Male/Female) , which will be mapped to 0/1.
# * Divide Age into 5 categories and Map them with 0/1/2/3/4.
# * Divide Fare into 4 categories and Map them to 0/1/2/3.
# * Embarked Attribute has (S/C/Q) , which will be mapped to 0/1/2.

# In[51]:


def mapping(frame):
    
    frame['Sex'] = frame.Sex.map({'female': 0 ,  'male': 1}).astype(int)
    
    
    frame['Embarked'] = frame.Embarked.map({'S' : 0 , 'C': 1 , 'Q':2}).astype(int)
    
    
    
    frame.loc[frame.Age <= 16 , 'Age'] = 0
    frame.loc[(frame.Age >16) & (frame.Age<=32) , 'Age'] = 1
    frame.loc[(frame.Age >32) & (frame.Age<=48) , 'Age'] = 2
    frame.loc[(frame.Age >48) & (frame.Age<=64) , 'Age'] = 3
    frame.loc[(frame.Age >64) & (frame.Age<=80) , 'Age'] = 4
    
    
    frame.loc[(frame.Fare <= 7.91) , 'Fare'] = 0
    frame.loc[(frame.Fare > 7.91) & (frame.Fare <= 14.454) , 'Fare'] = 1
    frame.loc[(frame.Fare > 14.454) & (frame.Fare <= 31) , 'Fare'] = 2
    frame.loc[(frame.Fare > 31) , 'Fare'] = 3


# In[52]:


mapping(df_train)
df_train.head()


# In[53]:


mapping(df_test)
df_test.head()


# # **Now, it's right time to choose best model.**

# In[54]:


# Importing some algorithms from sklearn.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[55]:


# Splitting data into test and train set.
x_train,x_test,y_train,y_test=train_test_split(df_train.drop('Survived',axis=1),df_train.Survived,test_size=0.20,random_state=66)


# In[56]:


models = [LogisticRegression(),RandomForestClassifier(),
        DecisionTreeClassifier()]

model_names=['LogisticRegression','RandomForestClassifier','DecisionTree']

accuracy = []

for model in range(len(models)):
    clf = models[model]
    clf.fit(x_train,y_train)
    pred = clf.predict(x_test)
    accuracy.append(accuracy_score(pred , y_test))
    
compare = pd.DataFrame({'Algorithm' : model_names , 'Accuracy' : accuracy})
compare


# **We get highest accuracy for DecisionTreeClassifier**

# **Well, DecisionTree did a great job there, with the highest accuracy[82.6%]**
# 

# **Now lets try to tune parameter**
# 

# In[57]:


params_dict={'criterion':['gini','entropy'],'max_depth':[5.21,5.22,5.23,5.24,5.25,5.26,5.27,5.28,5.29,5.3]}
clf_dt=GridSearchCV(estimator=DecisionTreeClassifier(),param_grid=params_dict,scoring='accuracy', cv=5)
clf_dt.fit(x_train,y_train)
pred=clf_dt.predict(x_test)
print(accuracy_score(pred,y_test))
print(clf_dt.best_params_)


# In[58]:


predio = clf_dt.predict(df_test)

d = {'PassengerId' : df_test_copy.PassengerId , 'Survived' : predio}
answer = pd.DataFrame(d)
# Generate CSV file based on DecisionTree Classifier
answer.to_csv('predio.csv' , index=False)


# # Thank you
# 
# Guys,do put your query in comment section and if you like the implementation method, do upvote it. 
