#!/usr/bin/env python
# coding: utf-8

# ## Welcome to my Titanic EDA for Beginners!
# This kernel covers simple Data Exploration & Visualization, Feature Engineering.
# 
# The main goal of this notebooks is not to show you the best score, but to give a clear explanation of the things that you need in [Titanic](https://www.kaggle.com/c/titanic) competition.
# 
# ## <font color="green">If this notebook were useful for you, please <b>UPVOTE</b> it =)</font>

# Let's get started. First, we have to import the libraries and look at the data:

# In[1]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[3]:


get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# Pandas allows you to read any csv file in one line, then we can do whatever we want.

# In[4]:


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')


# ## 1. Data Observation.

# In[5]:


display(train.head(5))
print(train.shape)
print(train.columns.values)


# Well, we have 12 features, some of which are very useful and affect survivability. 
# For instance, **Age** - obviously a child will have priority for a place in the boat. 
# Also, status of the person's cabin, because people in the first class will have more chances to survive.
# 
# We will definitely check the correlation of all features with the target variable, just a little later.
# 
# Note that not all attributes are numbers. We have words, categorical variables etc. Let's take a look at the data types:

# In[6]:


train.info()


# **float64** & **int64** are numbers, whereas **object** is related with words or letters.
# 
# Furthermore, we have another problem: **NaN values**. These are missing values in our dataset, and they can hurt out model a lot.
# 
# Next, The **Survived** feature is our target. It takes the value 0 or 1, 0 for death, 1 for surviving, and we will predict it.

# In[7]:


train.describe(include=['object'])


# By this table we can draw some conclusion, e.g. 
# * Every person's name on the Titanic is unique, 
# * The Sex trait takes two values, male and female. 
# * Staterooms are not unique because many people were in groups and shared one cabin with several of them.
# 
# As you can see, we haven't done anything yet, but despite that fact, we know about our data a lot! :D It's a very important skill in Data Science: trying to draw preliminary conclusions from the data.
# 
# Let's visualize out Target Variable:

# In[8]:


plt.figure(figsize=(10, 5))
train['Survived'].value_counts().plot(kind='barh')
plt.title('Target Variable')
plt.show()


# In[9]:


plt, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, i in zip(axes.flatten(), range(0, 2)):
    sns.distplot(train[train['Survived'] == i]['Age'].dropna(), ax=ax, axlabel='Age of Survived ' + str(i))


# We can see that infants have better survival rate as the people aged ~80 years: they have a 100% chance of survival. 
# Our hypothesis is confirmed.

# In[10]:


sns.barplot(x='Pclass', y='Survived', data=train)
plt.show()


# Being in **first** class gives you a better chance of surviving than in **third** one.

# In[11]:


train.head(5)


# ## 2. Data Cleaning & Feature Engineering

# Let's clean up the data a bit. 
# It's really important to remove attributes that don't provide information for the model at all. 
# I mean **Ticket** and **Cabine** features (don't forget about "inplace=True", for proper removal).

# In[12]:


print('Data size before deletion: {}'.format(train.shape))

train.drop(columns=['Ticket', 'Cabin'], inplace=True)
test.drop(columns=['Ticket', 'Cabin'], inplace=True)

print('Data size after deletion: {}'.format(train.shape))


# The **Name** feature seems useless at first glance. But let's think how we can benefit from it. As you can see below, every **Name** contains titles like 'Mr', 'Mrs' etc.
# 
# Maybe if there's a doctor (Dr.) on the boat, he might have a better chance of surviving?

# In[13]:


train['Name'].head(5)


# In[14]:


train['Name'].str.extract('([A-Za-z]+)\.', expand=False).unique()


# Let's create a separate feature thereof:

# In[15]:


train['Title'] = train['Name'].str.extract('([A-Za-z]+)\.', expand=False)
test['Title'] = test['Name'].str.extract('([A-Za-z]+)\.', expand=False)


# In[16]:


train['Title'].value_counts()


# Number of values is too high. I implement the For Loop below: 
# 
# Those titles that are rare will be replaced by **'Rare'** value, whereas the rest will be called by common names. 

# In[17]:


for dataset in [train, test]:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


# In[18]:


train['Title'].value_counts()


# That's better.

# Let's check the survival rate of each group:

# In[19]:


train[['Title', 'Survived']].groupby(['Title'], as_index=False) \
                            .mean() \
                            .sort_values(by='Survived', ascending=False)


# We need to code 'Title' in numbers, because putting words into a Machine Learning model isn't a good idea.
# 
# "Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5

# In[20]:


for dataset in [train, test]:
    dataset['Title'] = dataset['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}).fillna(0)
    
train.head(5)


# We've done all we can do with the **Name** and **PassengerId** features. Let's drop them:

# In[21]:


print('Data size before deletion: {}'.format(train.shape))

for dataset in [train, test]:
    dataset.drop(columns=['Name'], inplace=True) #, 'PassengerId'

print('Data size after deletion: {}'.format(train.shape))


# In[22]:


train.head(5)


# Let's deal with categorical variable – **Sex**. 1 will be male, 0 – female. 

# In[23]:


# we need to code Sex column in numbers
for dataset in [train, test]:
    dataset['Sex'] = dataset['Sex'].map({'male': 1, 'female': 0})


# In[24]:


train.head()


# There are a lot of null valies in **'Age'**, we must fix it:

# In[25]:


np.sum(train['Age'].isnull())


# Here I used median value (not 0, because it will break the results of the model). 
# 
# Moreover, I would like to point out that there are many ways of dealing with the 0 value, but in this kernel I've chosen the clearest and easiest =)

# In[26]:


for dataset in [train, test]:
    dataset['Age'] = dataset['Age'].fillna(np.median(dataset['Age'].median()))


# In[27]:


train.head(5)


# In[28]:


np.sum(train['Age'].isnull())


# Good!

# Let's create a new feature: **FamilySize** based on **SibSp** and **Parch**:

# In[29]:


for dataset in [train, test]:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1


# In[30]:


train.head(5)


# And a feature for those who have travelled without family – **Alone**

# In[31]:


for dataset in [train, test]:
    dataset['Alone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'Alone'] = 1


# In[32]:


train.head(5)


# After that, we can delete useless columns:

# In[33]:


train = train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test = test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)


# In[34]:


print('Data size after deletion: {}'.format(train.shape))


# In[35]:


train.head()


# Now I want to turn my attention to the **Embarked** feature.

# In[36]:


train['Embarked'].value_counts()


# In[37]:


for dataset in [train, test]:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# The same approach we used for the previous categorical features:

# In[38]:


for dataset in [train, test]:
    dataset['Embarked'] = dataset['Embarked'].map({'S' : 0,
                                                   'C' : 1,
                                                   'Q' : 2}) \
                                             .astype('int')


# In[39]:


train.head()


# Good job! Now let's look at the correlations for all features:

# In[40]:


sns.heatmap(train.corr(), annot=True)
plt.show()


# ## 3. Modelling.

# ##### I decided to use the following algorithms:
# 1. Logistic Regression
# 2. Decision Tree
# 3. Random Forest

# 1. Logistic Regression

# In[41]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[42]:


y = train['Survived']
X = train.drop(columns=['Survived'])
X_test = test


# We have to verify the result of the model somehow, so we do a delayed sampling and make a validation on it.
# 
# This is the simplest way, but you can read about other validation methods here:
# 
# https://towardsdatascience.com/supervised-machine-learning-model-validation-a-step-by-step-approach-771109ae0253
# 
# https://www.geeksforgeeks.org/cross-validation-machine-learning/

# In[43]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


# In[44]:


print('Training sample:', X_train.shape)
print('Validation sample:', X_val.shape)


# In[45]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)

pred = logreg.predict(X_val)
accuracy_score(y_val, pred)


# 2. Decision tree

# In[46]:


from sklearn.tree import DecisionTreeClassifier


# In[47]:


clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

pred = clf.predict(X_val)
accuracy_score(y_val, pred)


# 3. Random Forest

# In[48]:


from sklearn.ensemble import RandomForestClassifier


# In[49]:


rf = RandomForestClassifier()
rf.fit(X_train, y_train)

pred = rf.predict(X_val)
accuracy_score(y_val, pred)


# ## 4. Submission

# In[50]:


test = pd.read_csv('../input/titanic/test.csv')
test['Survived'] = 0
test.loc[test['Sex'] == 'female','Survived'] = 1
data_to_submit = pd.DataFrame({
    'PassengerId':test['PassengerId'],
    'Survived':test['Survived']
})


# In[51]:


data_to_submit.to_csv('csv_to_submit.csv', index = False)
print('Saved file: ' + filename)


# Well, without too much effort, we:
# 1. Processed the data, 
# 2. Created a couple of our own features, 
# 3. Visualized them 
# 4. Trained 3 machine learning models. 
# 
# Sure, it's not the best score you can get, but this notebook give you basic knowledge about processes in Kaggle competition.
# 
# We can improve this result. For example, deal with missing data individually for each feature rather than in a cycle, create new 'features' based on existing ones, or even find parameters that will increase the score of the model in the leaderboard, when we try another models.
# 
# Finding the best solution at Kaggle competitions is a whole art that can be mastered by combining different techniques with non-traditional methods. Good luck!
# 
# ## <font color="green">Stay tuned and don't forget to <b>UPVOTE</b> this kernel =)</font>
