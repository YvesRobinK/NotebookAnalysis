#!/usr/bin/env python
# coding: utf-8

# # <p style="background-color:#80ccff; font-family:newtimeroman; font-size:150%; text-align:center; border-radius:  80px 5px; padding-top:8px; padding-bottom:8px;">Import</p>

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import iplot
import plotly.graph_objects as go
sns.set(rc={'figure.figsize':(10,6)})
sns.set(font_scale=1.3)

import string 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import ShuffleSplit, GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix

import warnings
warnings.filterwarnings("ignore")


# # <p style="background-color:#80ccff; font-family:newtimeroman; font-size:150%; text-align:center; border-radius:  80px 5px; padding-top:8px; padding-bottom:8px;">Input</p>

# In[2]:


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
sub = pd.read_csv('../input/titanic/gender_submission.csv')


# In[3]:


train.head()


# # <p style="background-color:#80ccff; font-family:newtimeroman; font-size:150%; text-align:center; border-radius:  80px 5px; padding-top:8px; padding-bottom:8px;">Preprocessing</p>

# In[4]:


train.info()


# In[5]:


train.describe()


# In[6]:


(train.isnull().sum()/len(train))*100


# In[7]:


train.drop(columns=['Cabin'], inplace=True)


# # Missing Value

# In[8]:


train['Age'].fillna(train['Age'].median(), inplace=True)
train['Embarked'].fillna(method='ffill', inplace=True)


# In[9]:


(train.isnull().sum()/len(train))*100


# <div style="color:black; background-color:#f5f7b0; border-radius:10px; padding:20px;">
# <b>Observation</b><br/>
# The Age columns has 19.86% of missing values theses values have been replaced by the average.<br/>
# The Cabin columns has 77.10% of missing values. There are many missing values, so I deleted it.
# </div>

# # <p style="background-color:#80ccff; font-family:newtimeroman; font-size:150%; text-align:center; border-radius:  80px 5px; padding-top:8px; padding-bottom:8px;">Plots</p>

# In[10]:


values=train["Survived"].value_counts().values
fig = go.Figure(data=[go.Pie(labels=['Not Survived','Survived'],values=values,textinfo='label+percent')])
fig.update_layout(title={'text': "Titanic Survived",'y':0.9,'x':0.45,'xanchor': 'center','yanchor': 'top'},
                  font=dict(size=18, color='black', family="Courier New, monospace"))


# In[11]:


values=train["Pclass"].value_counts().values
fig = go.Figure(data=[go.Pie(labels=['3º Class','2° Class','1° Class'],values=values,textinfo='label+percent')])
fig.update_layout(title={'text': "Class of Ship",'y':0.9,'x':0.45,'xanchor': 'center','yanchor': 'top'},
                  font=dict(size=18, color='black', family="Courier New, monospace"))
fig.show()


# In[12]:


values=train["Sex"].value_counts().values
fig = go.Figure(data=[go.Pie(labels=['Male','Female'],values=values,textinfo='label+percent')])
fig.update_layout(title={'text': "Sex",'y':0.9,'x':0.47,'xanchor': 'center','yanchor': 'top'},
                  font=dict(size=18, color='black', family="Courier New, monospace"))
fig.show()


# In[13]:


survived = train[train['Survived']==1]
values=survived["Sex"].value_counts().values
fig = go.Figure(data=[go.Pie(labels=['Male','Female'],values=values,textinfo='label+percent')])
fig.update_layout(title={'text': "Survivor by Sex",'y':0.9,'x':0.47,'xanchor': 'center','yanchor': 'top'},
                  font=dict(size=18, color='black', family="Courier New, monospace"))
fig.show()


# In[14]:


g = sns.FacetGrid(train, col='Survived', height=6)
g.map(plt.hist, 'Age', bins=20);


# In[15]:


g = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.8, aspect=1.6)
g.map(plt.hist, 'Age', bins=20);


# In[16]:


g = sns.FacetGrid(train, row='Embarked', size=2.4, aspect=2.6)
g.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
g.add_legend();


# In[17]:


g = sns.FacetGrid(train, row='Embarked', col='Survived', size=2.2, aspect=2.4)
g.map(sns.barplot, 'Sex', 'Fare')
g.add_legend();


# In[18]:


corr = train.corr()
plt.figure(figsize=(16,8))
sns.heatmap(data=corr, annot=True);


# <div style="color:black; background-color:#f5f7b0; border-radius:10px; padding:20px;">
# <b>Observation</b><br/>
# - 61.6% dit not survive<br/>
# - 55.1% was 3rd class<br/>
# - 64.8% were male<br/>
# - The median age of the survivors was 29 years<br/>
# - The majority of the survivors were in the 3rd class (the 3rd class was the majority on the ship<br/>
# </div>

# # <p style="background-color:#80ccff; font-family:newtimeroman; font-size:150%; text-align:center; border-radius:  80px 5px; padding-top:8px; padding-bottom:8px;">Feature Engineering</p>

# In[19]:


def encoder(data):
    le = LabelEncoder()
    for col in data.select_dtypes('object'):
        data[col] = le.fit_transform(data[col])
    return(data)


# In[20]:


train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train['Title'] = train['Title'].replace(['Lady', 'Countess','Capt', 'Col', 
                                             'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')


# In[21]:


train.loc[ train['Age'] <= 16, 'Age'] = 0
train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1
train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2
train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3
train.loc[ train['Age'] > 64, 'Age']


# In[22]:


train.head()


# In[23]:


train.drop(columns=['Name','PassengerId','Ticket'], inplace=True)


# In[24]:


train = encoder(train)
train.head()


# <div style="color:black; background-color:#f5f7b0; border-radius:10px; padding:20px;">
# <b>Observation</b><br/>
# - A class was created with the passengers title<br/>
# - A class was created with age<br/>
# - Some columns have been deleted<br/>
# - The object type of the columns has been transformed<br/>
# </div>

# # <p style="background-color:#80ccff; font-family:newtimeroman; font-size:150%; text-align:center; border-radius:  80px 5px; padding-top:8px; padding-bottom:8px;">Model</p>

# In[25]:


x = train.drop(['Survived'], axis=1)
y = train.iloc[:,0]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)


# In[26]:


colunas = ['Modelo','Acuracy']
resultado = pd.DataFrame(columns=colunas)


# In[27]:


models = []

models.append(('GradientBoostingClassifier', GradientBoostingClassifier()))
models.append(('AdaBoostClassifier', AdaBoostClassifier()))
models.append(('ExtraTreesClassifier', ExtraTreesClassifier()))
models.append(('BaggingClassifier', BaggingClassifier()))
models.append(('RandomForestClassifier', RandomForestClassifier()))
models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
models.append(('ExtraTreeClassifier', ExtraTreeClassifier()))
models.append(("XGBClassifier", XGBClassifier()))

for name, model in models:
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    resultado = resultado.append(pd.DataFrame([[name, acc]], columns=colunas))
    
resultado.sort_values(by=['Acuracy'], ascending=False, inplace=True)
resultado


# In[28]:


test.drop(columns=['Cabin'], inplace=True)

test['Age'].fillna(test['Age'].median(), inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)

test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test['Title'] = test['Title'].replace(['Lady', 'Countess','Capt', 'Col', 
                                             'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test['Title'] = test['Title'].replace('Mlle', 'Miss')
test['Title'] = test['Title'].replace('Ms', 'Miss')
test['Title'] = test['Title'].replace('Mme', 'Mrs')

test.loc[ test['Age'] <= 16, 'Age'] = 0
test.loc[(test['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1
test.loc[(test['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2
test.loc[(test['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3
test.loc[ test['Age'] > 64, 'Age']

test = encoder(test)

test.drop(columns=['Name','PassengerId','Ticket'], inplace=True)


# In[29]:


model = GradientBoostingClassifier()
model.fit(x_train,y_train)
pred = model.predict(x_test)
acc = accuracy_score(y_test, pred)
acc


# In[30]:


sub['Survived'] =  model.predict(test)
sub.to_csv('submission.csv', index=False)

