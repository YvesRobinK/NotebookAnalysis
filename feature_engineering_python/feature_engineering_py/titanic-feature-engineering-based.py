#!/usr/bin/env python
# coding: utf-8

# Here is a truth about the 100% accuracy on Leader Board, this is something that everyone should be aware of while starting - https://www.kaggle.com/tarunpaparaju/titanic-competition-how-top-lb-got-their-score

# If you like this work, please <b>UPVOTE.</b>

# This dataset demands a feature centric approach and I haev kept it simple. For each step there is always something new one can do, so one must feel free to experiment. However, keeping it simple is the key.

# ### Imports

# In[1]:


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV


# ### Load Data

# In[2]:


df = pd.read_csv('../input/titanic/train.csv')


# In[3]:


df


# In[4]:


df.info()


# So our task is to help the model get a clean and effective data and here we see 
# 1. PassengerId is useless
# 2. Missing values in - Age, Cabin

# ### Feature Engineering

# #### PassengerId

# In[5]:


df.drop('PassengerId', axis=1, inplace=True)


# #### Pclass

# In[6]:


df.groupby('Pclass').Survived.mean()


# In[7]:


sns.countplot(x=df.Pclass, hue=df.Survived)
plt.show()


# We leave the Pclass as it is as no of categories are low and they seem to have meaning in relation to target feature.

# #### Name

# In the Name feature, however, anything before the second space is like a title that repeats for several people but anything else in both its sides do not seem to repeat so they give no context to the model. In short they are not helpful.

# In[8]:


df.Name.sample(4)


# In[9]:


df.Name = df.Name.apply(lambda x: x.split()[1].replace('.',''))


# In[10]:


df.Name.unique()


# In[11]:


df.Name.value_counts()


# We will keep till 'Rev' and put all others under one label called 'Others'.

# In[12]:


def name_fun(x):
    if x not in ['Mr', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev']:
        return 'Others'
    else:
        return x


# In[13]:


df.Name = df.Name.apply(name_fun)


# In[14]:


df.Name = df.Name.map({'Mr':0, 'Miss':1, 'Mrs':2, 'Master':3, 'Dr':3, 'Rev':3, 'Others':4})


# In[15]:


sns.countplot(x=df.Name)
plt.xticks(rotation=75)
plt.show()


# In[16]:


sns.countplot(x=df.Name, hue=df.Survived)
plt.show()


# #### Sex

# In[17]:


df.Sex = df.Sex.map({'male':1, 'female':0})


# #### SibSp and Parch

# In[18]:


df['Family'] = df.SibSp + df.Parch
df.drop(['SibSp', 'Parch'], axis=1, inplace=True)


# This gives meaning to the Sibling Spouse Parent Child features and defines one feature called Family.

# #### Ticket

# The tickets must correspond to something, cabin area or booking sequence maybe. But they surely form a group, by the initials, be they of the words present in the ticket columns or the first digits of the ticket no. So we break them like that.

# In[19]:


df.Ticket.sample(10)


# In[20]:


def ticket_fun(x):
    if x.isdigit():
        return str(x)[:1]
    else:
        return x[:1]


# In[21]:


df.Ticket = df.Ticket.apply(lambda x: x.split()[0].replace('.','').replace('/','').lower())
df.Ticket = df.Ticket.apply(ticket_fun)


# In[22]:


dict(df.Ticket.value_counts())


# So they do form a group, as we can see.

# In[23]:


sns.countplot(x=df.Ticket, hue=df.Survived)
plt.show()


# And they do have meaning when it comes to survival. But not all categories are relevant so lets reduce the no of buckets.

# In[24]:


def ticket_fun2(x):
    if x not in ['a', 'p', 's', '1', '2', '3', 'c']:
        return 'Others'
    else:
        return str(x)
df.Ticket = df.Ticket.apply(ticket_fun2)


# In[25]:


sns.countplot(x=df.Ticket, hue=df.Survived)
plt.show()


# Now it is cleaner and more precise

# In[26]:


df.Ticket = df.Ticket.map({'a':0, 'p':1, 's':2, '1':3, '2':3, '3':4, 'c':5, 'Others':6})


# #### Fare

# If we look at the Fare, they follow a pareto chart. They do have groups as we will see next.

# In[27]:


sns.histplot(df.Fare, bins=range(1,500, 10))
plt.show()


# If we group them from 0-50, 51-100, 100 to any value, they fall in line well.

# In[28]:


def fare_fun(x):
    if x>0 and x<50:
        return 'gr1'
    elif x>50 and x<100:
        return 'gr2'
    elif x>100:
        return 'gr3'
df.Fare = df.Fare.apply(fare_fun)


# In[29]:


sns.countplot(x=df.Fare, hue=df.Survived)
plt.show()


# In[30]:


df.Fare = df.Fare.fillna('gr1')
df[df.Fare.isnull()]


# I took the liberty give few missing values to group 1 as they are the most in no.

# In[31]:


df.Fare = df.Fare.map({'gr1':0, 'gr2':1, 'gr3':2})


# #### Cabin

# In[32]:


df.Cabin = df.Cabin.apply(lambda x: str(x)[0])


# In[33]:


df.drop(['Cabin'], axis=1, inplace=True)


# It has so many missing values that it is better to drop.

# #### Embarked

# In[34]:


df.Embarked = df.Embarked.fillna('S')
df.Embarked = df.Embarked.map({'S':0, 'Q':1, 'C':2})


# #### Age

# Lets see survival by the age group. Looks like only 0-5 year olds were certain to live others just followed a natural patter. But we will group them too.

# In[35]:


sns.histplot(df[df.Survived==0].Age, color='r', label='Died')
sns.histplot(df[df.Survived==1].Age, color='g', label='Survived')
plt.legend()
plt.show()


# The bin sizes below hold the values well, so they should make good groups.

# In[36]:


sns.histplot(df[df.Survived==0].Age, color='r', label='Died', bins=[0,15,30,45,60,80])
sns.histplot(df[df.Survived==1].Age, color='g', label='Survived', bins=[0,15,30,45,60,80])
plt.legend()
plt.show()


# In[37]:


def age_fun(x):
    if x>0 and x<30:
        return 'gr1'
    elif x>30 and x<45:
        return 'gr2'
    elif x>45 and x<60:
        return 'gr3'
    elif x>60:
        return 'gr4'
df.Age = df.Age.apply(age_fun)


# In[38]:


sns.countplot(x=df.Age, hue=df.Survived)
plt.show()


# This should be good enough.

# In[39]:


sns.countplot(x=df.Age, hue=df.Fare)
plt.show()


# In[40]:


df[df.Age.isnull()]


# There are plenty of Null values to be filled. I could try to infer from other features but lets give this task to a model instead.
# 
# We will predict the age from the data we have.

# In[41]:


train_data = df[df.Age.isnull()==False]
train_surv = train_data.Survived
target = train_data.Age
train_data.drop(['Age'], axis=1, inplace=True)

test_data =  df[df.Age.isnull()==True]
test_surv = test_data.Survived
test_data.drop(['Age'], axis=1, inplace=True)


# In[42]:


train_data.info()


# In[43]:


dtclf = DecisionTreeClassifier()
dtclf.fit(train_data, target)
test_data['Age'] = dtclf.predict(test_data)


# In[44]:


sns.countplot(x=test_data.Age)
plt.show()


# This is the prediction for the null values of Age column.

# In[45]:


train_data['Age'] = target
train_data['Survived'] = train_surv
test_data['Survived'] = test_surv
df = pd.concat([train_data, test_data], axis=0)


# In[46]:


df.Age = df.Age.map({'gr1':0, 'gr2':1, 'gr3':2, 'gr4':3})


# ### Feature Engineering Outcome

# In[47]:


df.info()


# So here is what it looks like after all the tweaking.

# In[48]:


df


# ### Training Model

# Lets split the train data and check once how a basic model performs. 

# In[49]:


X_train, X_val, y_train, y_val = train_test_split(df.drop(['Survived'], axis=1), df.Survived,
                                                    test_size=0.10, random_state=42)

gbclf = GradientBoostingClassifier()
gbclf.fit(X_train, y_train)
y_pred = gbclf.predict(X_val)
acc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc)


# That's good enough.

# ### Predicting and Creating Submission File

# In[50]:


tdf = pd.read_csv('../input/titanic/test.csv')


# In[51]:


tdf.info()


# It turns out, test data has some NULL values too, we need to take care of that also the encodings must be same.

# In[52]:


tdf.drop(['PassengerId','Cabin'], axis=1, inplace=True)


# In[53]:


tdf.Name = tdf.Name.apply(lambda x: str(x).split()[1].replace('.',''))


# In[54]:


# For test data

def name_fun(x):
    if x not in ['Mr', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev']:
        return 'Others'
    else:
        return x
    
tdf.Name = tdf.Name.apply(name_fun)
tdf.Name = tdf.Name.map({'Mr':0, 'Miss':1, 'Mrs':2, 'Master':3, 'Dr':3, 'Rev':3, 'Others':4})

tdf.Sex = tdf.Sex.map({'male':1, 'female':0})

def ticket_fun(x):
    if x.isdigit():
        return str(x)[:1]
    else:
        return x[:1]
    
tdf.Ticket = tdf.Ticket.apply(lambda x: x.split()[0].replace('.','').replace('/','').lower())
tdf.Ticket = tdf.Ticket.apply(ticket_fun)

def ticket_fun2(x):
    if x not in ['a', 'p', 's', '1', '2', '3', 'c']:
        return 'Others'
    else:
        return str(x)
tdf.Ticket = tdf.Ticket.apply(ticket_fun2)
tdf.Ticket = tdf.Ticket.map({'a':0, 'p':1, 's':2, '1':3, '2':3, '3':4, 'c':5, 'Others':6})

def fare_fun(x):
    x = float(x)
    if x>0 and x<50:
        return 'gr1'
    elif x>50 and x<100:
        return 'gr2'
    elif x>100:
        return 'gr3'
tdf.Fare = tdf.Fare.apply(fare_fun)
tdf.Fare = tdf.Fare.fillna('gr1')
tdf.Fare = tdf.Fare.map({'gr1':0, 'gr2':1, 'gr3':2})
tdf.Embarked = tdf.Embarked.map({'S':0, 'Q':1, 'C':2})

def age_fun(x):
    if x>0 and x<30:
        return 'gr1'
    elif x>30 and x<45:
        return 'gr2'
    elif x>45 and x<60:
        return 'gr3'
    elif x>60:
        return 'gr4'
tdf.Age = tdf.Age.apply(age_fun)
tdf.Age = tdf.Age.fillna('gr1')
tdf.Age = tdf.Age.map({'gr1':0, 'gr2':1, 'gr3':2, 'gr4':3})


# In[55]:


tdf['Family'] = tdf.SibSp + tdf.Parch
tdf.drop(['SibSp','Parch'], axis=1, inplace=True)


# Here is the test data after cleaning that we will use for prediction.

# In[56]:


tdf


# In[57]:


# This one gave 73.4 % accuracy on Submission
# gbclf = GradientBoostingClassifier()

# params = {    
#               'loss' : ["deviance"],
#               'n_estimators' : [50,100,200,300,500],
#               'learning_rate': [0.1, 0.05, 0.01, 0.001, 0.0001],
#               'max_depth': [4, 8]
#          }

# gs = GridSearchCV(gbclf,param_grid = params, cv=5, scoring="accuracy", n_jobs= -1, verbose = 1)

# gs.fit(X_train,y_train)


# In[59]:


RFC = RandomForestClassifier()


params = { 
              "max_features": [1, 3, 5, 10],
              "min_samples_split": [2, 3, 10],
              "n_estimators" :[100,300, 500, 700, 1000, 1200, 1500, 1700],
              "criterion": ["gini", "entropy"]
          }


gs = GridSearchCV(RFC, param_grid = params, cv=3, scoring="accuracy", n_jobs= -1, verbose = 1)

gs.fit(X_train,y_train)

print(gs.best_estimator_)
print(gs.best_score_)


# In[60]:


submission = pd.DataFrame({ 'PassengerId' : pd.read_csv('../input/titanic/test.csv')['PassengerId'], 
                           'Survived': gs.predict(tdf)})
submission.to_csv('submission.csv', index=False)

