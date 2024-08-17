#!/usr/bin/env python
# coding: utf-8

# ##  **Titanic - Machine Learning from Disaster**
# 
# The **RMS Titanic** sank in the **early morning hours of 15 April 1912** in the North Atlantic Ocean, four days into her maiden voyage from Southampton to New York City. The largest ocean liner in service at the time, Titanic had an estimated **2,224** people on board when she **struck an iceberg** at around 23:40 (ship's time) on Sunday, 14 April 1912. Her sinking **two hours and forty minutes later** at 02:20 (ship's time; 05:18 GMT) on Monday, 15 April, resulted in the **deaths of more than 1,500** people, making it one of the **deadliest peacetime maritime disasters** in history.
# 
# <img src = "https://media.tenor.com/dRQMwkMSytYAAAAC/titanic.gif" width = 400 height = 450/>

# # <h1 style='background:#80ced6; border:2; border-radius: 10px;padding-top: 2%;; font-size:200%; font-weight: bold; color:#c83349'><center>Table of contents</center></h1> 
# 1. [Content Of Dataset](#1)   
# 2. [Importing libraries](#2)
# 3. [Preprocessing](#3)  
# 4. [EDA](#4)
# 5. [Fearture Selection & Engineering](#5)
# 6. [BaseLine](#6)     
# 7. [HyperParameter Tuning](#7)      
# 9. [Thank You](#8)

# # Content Of Dataset

# <a id="1"></a> 
# 
# - 1. **PassengerId:** This column assigns a unique identifier for each passenger.
# - 2. **Survived:** Specifies whether the given passenger survived or not (1 - survived, 0 - didn't survive)
# - 3. **Pclass:** The passenger's class. (1 = Upper Deck, 2 = Middle Deck, 3 = Lower Deck)
# - 4. **Name:** The name of the passenger.
# - 5. **Sex:** The sex of the passenger (male, female)
# - 6. **Age:** The age of the passenger in years. If the age is estimated, is it in the form of xx.5.
# - 7. **SibSp:** How many siblings or spouses the passenger had on board with them. Sibling = brother, sister, stepbrother,          stepsister and Spouse = husband, wife (mistresses and fiancés were ignored)
# - 8. **Parch:** How many parents or children the passenger had on boad with them. Parent = mother and father, child = daughter,      son, stepdaughter and stepson and some children travelled only with a nanny, therefore parch=0 for them.
# - 9. **Ticket:** The ticket of the passenger.
# - 10. **Fare:** The fare amount paid by the passenger for the trip.
# - 11. **Cabin:** The cabin in which the passenger stayed.
# - 12. **Embarked:** The place from which the passenger embarked (S, C, Q)

# # Importing Libraries
# 
# <a id="2"></a>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly 
import plotly.express as px
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


# # Preprocessing
# 
# <a id ="3"> </a>

# In[2]:


df_train=pd.read_csv("/kaggle/input/titanic/train.csv")
df_test=pd.read_csv("/kaggle/input/titanic/test.csv")
df_submission=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

train = df_train.copy()
test = df_test.copy()


# In[3]:


train.drop(['PassengerId'],axis=1,inplace=True)
test.drop(['PassengerId'],axis=1,inplace=True)


# In[4]:


df_train.head()


# In[5]:


df_train.info()


#  **So we Have both numerical columns & categorical columns .**

# In[6]:


df_train.describe()


# In[7]:


df_train.isnull().sum()


# In[8]:


df_test.insert(1,'Survived', df_submission['Survived'])


# In[9]:


y_train = df_train['Survived']
y_test = df_submission['Survived']


# In[10]:


df_test


# **So Null Values are present in Age , Cabin & Embarked Column . We will see later on to handling these .**

# In[11]:


cat_cols = [cname for cname in train.columns if train[cname].nunique() < 10 and 
                        train[cname].dtype == "object"]


num_cols = [cname for cname in train.columns if train[cname].dtype in ['int64', 'float64']]
print(cat_cols)
print(num_cols)


# # EDA
# 
# ## 1.Univariate Analysis

# **1.SEX**
# <a id="4"></a>

# In[12]:


print (f'{round(train["Sex"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(train, x="Sex", title='Sex', width=400, height=400)
fig.show()


# **577 MALES & 314 FEMALES WERE PRESENT THERE .**

# **2.Pclass**

# In[13]:


print (f'{round(train["Pclass"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(train, x="Pclass", title='Pclass', width=400, height=400)
fig.show()


# - Class 1 / Upper Deck  =  216
# - Class 2 / Middle Deck =  184
# - Class 3 / Lower Deck  =  491
# 
# **So More People were belonged from Lower Deck Class.**

# **3.SibSp**

# In[14]:


print (f'{round(train["SibSp"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(train, x="SibSp", title='SibSp', width=400, height=400)
fig.show()


# **4.Embarked**

# In[15]:


print (f'{round(train["Embarked"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(train, x="Embarked", title='Embarked', width=400, height=400)
fig.show()


# **People Embarked From Southampton Port Are More .**

# **5.Survived**

# In[16]:


print (f'{round(train["Survived"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(train, x="Survived", title='Survived', width=400, height=400)
fig.show()


# In[17]:


sns.countplot('Survived',data=df_train).set(title='Survived')
# set_title('Survived')
plt.show()


# **Most of the People Din't survived , but some brave & lucky ones fought very hard at that night & managed to Survive .**

# ## 2.Bivariate & Multivariate Analysis

# **QN.1 Let's see what was the age of fellows who travelled that night  ?**

# In[18]:


sns.swarmplot(x='Sex',y='Age',hue='Survived',dodge='True',data=df_train)


# ## Observation:
# - **Let's Start with a good one .i.e, Most Of The Boy Children(2-7 yrs) were survived🧒🧒 .**
# - **That Night Also One 80 years Male Passenger was travelled & fortunately he was survived.**
# - **Though Female Passengers were less than males but less of them are died .**
# - **Most Of The Males are between age 13 - 38 yrs were didn't survived .**
# - **May Be At That Night Crew gave Priority To Rescue More Females and childern.**
# - **Sex can also be play a major role for our model .**

# **QN.2 Which Class People Survived More At That Night ?**

# In[19]:


sns.countplot('Pclass',hue='Survived',data=df_train)


# - **Who says Money💸💸 Can't Buy Everything!!**
# - **Pclass 1 were given a very high priority while rescue.**
# - **Though the the number of Passengers in Pclass 3 were more ,but  still the number of survival from them is very low .**
# - **So Pclass would be an important feature for our model .**

# In[20]:


sns.stripplot(x='Embarked',y='Age',data=df_train,hue='Survived')


# In[21]:


sns.stripplot(x='Sex',y='Fare',data=df_train,hue='Survived')


# **Understanding Parch Column**

# In[22]:


sns.factorplot('Parch','Survived',data=df_train)


# - The chances of survival is good for somebody who has **1-3 parents on the ship.**
# - Being alone also proved to be deadly and the chances for **survival decreases** when **somebody was > 4 parents** on that ship .

# **Qn 3. How Fair Was Distributed Over Classes ?**

# In[23]:


f,ax=plt.subplots(1,3,figsize=(12,6))
sns.distplot(df_train[df_train['Pclass']==1].Fare,ax=ax[0])
ax[0].set_title('Fares in Pclass 1')
sns.distplot(df_train[df_train['Pclass']==2].Fare,ax=ax[1])
ax[1].set_title('Fares in Pclass 2')
sns.distplot(df_train[df_train['Pclass']==3].Fare,ax=ax[2])
ax[2].set_title('Fares in Pclass 3')
plt.show()


# **We Can See That Fare In Class - 1 gradually increases & after reaching it's peak gradually decreases . But In Class - 3 , It is gradually increasing but after less time it faces a sharp fall.**

# In[24]:


# Visualizing The Distributions

n_bins = 50
histplot_hyperparams = {
    'kde':True,
    'alpha':0.4,
    'stat':'percent',
    'bins':n_bins,
   
}
# features= df_train.columns
cols=[ 'Age', 'Sex', 'Embarked', 'Survived',
       'Pclass', 'SibSp']
fig, ax = plt.subplots(2,3, figsize=(16, 10))
ax = ax.flatten()

for i, column in enumerate(cols):
    sns.histplot(
        df_train[column], label='Train',
        ax=ax[i], color='red', **histplot_hyperparams
    )


# **We Could See That Passengers having More Fare Were Survived.**

# # Feature Selection & Engineering

# <a id="5"></a> 
# **Here We Will Try To Select & Make Useful Features**

# ### Handling Missing Data

# - **Age Column is having 177 NAN values .**
# - **Cabin Column is having 687 NAN values .**
# - **Embarked Column is having 2 NAN values .**
# 
# #### Age:
# - As we  cab clearly see , the Age feature has 177 null values. To replace these NaN values, we can assign them the mean age of the dataset.
# - But the problem is, there were many people with many different ages. We just cant assign a 4 year kid  or 80 years old man     with the mean age that is 29 years. Is there any way to find out what age-band does the passenger lie??

# In[25]:


train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)

test["Fare"].fillna(test['Fare'].median(), inplace=True)

train["Embarked"].fillna("S", inplace = True) 


# In[26]:


train


# In[27]:


test


# **Creating New feature Family . IF The Fellow was having  any relationships then he will be included else excluded .**

# In[28]:


def Isfam(x):
    if  (x['SibSp'] + x['Parch'])  > 0:
        return 1
    else:
        return 0

train['Family'] = train.apply(Isfam, axis = 1)
test['Family'] = test.apply(Isfam, axis = 1)



# In[29]:


# Deleting SibSp & Parch
train = train.drop(['SibSp','Parch'],axis=1)
test = test.drop(['SibSp','Parch'],axis=1)


# ## Titanic Cutway Diagram 
# 
# <img src = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Olympic_%26_Titanic_cutaway_diagram.png/800px-Olympic_%26_Titanic_cutaway_diagram.png" width = 400 height = 450/>
# 
# **Collected From Wiki**

# In[30]:


# Mapping Cabin Values
train["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in train['Cabin'] ])
test["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in test['Cabin'] ])

train["Cabin"] = train["Cabin"].map({"X":0, "A":1, "B" : 2 , "C":3, "D":4, "E":5, "F":6, "G":7,"T":0})
train["Cabin"] = train["Cabin"].astype(int)
test["Cabin"] = test["Cabin"].map({"X":0, "A":1, "B" : 2 , "C":3, "D":4, "E":5, "F":6, "G":7,"T":0})
test["Cabin"] = test["Cabin"].astype(int)


# In[31]:


# Extracting Title from Name

train_title = [i.split(",")[1].split(".")[0].strip() for i in train["Name"]]
train["Title"] = pd.Series(train_title)

test_title = [i.split(",")[1].split(".")[0].strip() for i in test["Name"]]
test["Title"] = pd.Series(test_title)

train["Title"] = train["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train["Title"] = train["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
train["Title"] = train["Title"].astype(int)

test["Title"] = test["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test["Title"] = test["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
test["Title"] = test["Title"].astype(int)


# # Ticket Ticket🎫🎫 ..

# <img src = "https://miro.medium.com/max/828/1*6CkrEMfZ3CSDe24avFlxhQ.webp" width = 400 height = 450/>
# 

# In[32]:


# Extracting Prefix From Ticket
# Don't Need To dlt entire col
# It Contains info about Passenger
# we had got some unique ones

Ticket1 = []
for i in list(train.Ticket):
    if not i.isdigit() :
        Ticket1.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket1.append("X")
train["Ticket"] = Ticket1

Ticket2 = []
for j in list(test.Ticket):
    if not j.isdigit() :
        Ticket2.append(j.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket2.append("X")
        
test["Ticket"] = Ticket2

np.union1d(train["Ticket"], test["Ticket"])

train= pd.get_dummies(train, columns = ["Ticket"], prefix="T")
test = pd.get_dummies(test, columns = ["Ticket"], prefix="T")

train = train.drop(['T_SP','T_SOP','T_Fa','T_LINE','T_SWPP','T_SCOW','T_PPP','T_AS','T_CASOTON'],axis = 1)
test = test.drop(['T_SCA3','T_STONOQ','T_AQ4','T_A','T_LP','T_AQ3'],axis = 1)


# In[33]:


# Deleting Name Col 
# As Title has been extracted

train.drop(['Survived','Name'],axis=1,inplace=True)
test.drop('Name',axis=1,inplace=True)


# In[34]:


from sklearn.preprocessing import OneHotEncoder

trans= ColumnTransformer(transformers=[
    ('tnf1',OneHotEncoder(sparse=False,drop='first',handle_unknown='ignore'),['Sex','Embarked'])
],remainder='passthrough')

X_train_trans=trans.fit_transform(train)
X_test_trans=trans.transform(test)

X_train_trans=pd.DataFrame(X_train_trans)
X_test_trans=pd.DataFrame(X_test_trans)

print('shape of X_train_trans is: ',X_train_trans.shape)
print('shape of X_test_trans is: ',X_test_trans.shape)

X_train_trans


# In[35]:


# Scaling 

sc = StandardScaler()

train2 = sc.fit_transform(X_train_trans)
test2 = sc.transform(X_test_trans)


# # Baseline

# <a id="6"></a> 
# **Models With Out Tuning**

# In[36]:


from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score

KFold_Score = pd.DataFrame()
classifiers = ['Linear SVM', 'Radial SVM', 'LogisticRegression', 
               'RandomForestClassifier', 'AdaBoostClassifier', 
               'XGBoostClassifier', 'KNeighborsClassifier','GradientBoostingClassifier','LGBM']
models = [svm.SVC(kernel='linear'),
          svm.SVC(kernel='rbf'),
          LogisticRegression(max_iter = 1000),
          RandomForestClassifier(n_estimators=200, random_state=0),
          AdaBoostClassifier(random_state = 0),
          xgb.XGBClassifier(n_estimators=100),
          KNeighborsClassifier(),
          GradientBoostingClassifier(random_state=0),
          LGBMClassifier(random_state=0,n_estimators=50,max_depth=5)
         ]
j = 0
for i in models:
    model = i
    cv = KFold(n_splits=5, random_state=0, shuffle=True)
    KFold_Score[classifiers[j]] = (cross_val_score(model, train2, np.ravel(y_train), scoring = 'accuracy', cv=cv))
    j = j+1


# In[37]:


mean = pd.DataFrame(KFold_Score.mean(), index= classifiers)
KFold_Score = pd.concat([KFold_Score,mean.T])
KFold_Score.index=['Fold 1','Fold 2','Fold 3','Fold 4','Fold 5','Mean']
KFold_Score.T.sort_values(by=['Mean'], ascending = False)


# # HyperParameter  Tuning
# 

# <a id="7"></a>
# **Here We Will use grid Search CV**
# - **GridSearchCV is a technique for finding the optimal parameter values from a given set of parameters in a grid.**
# - **It's essentially a cross-validation technique.The model as well as the parameters must be entered.**
# - **After extracting the best parameter values, predictions are made.**
# <img src = "https://b2633864.smushcdn.com/2633864/wp-content/uploads/2021/03/hyperparameter_tuning_cv_grid_search-e1615719602429.png?lossy=1&strip=1&webp=1" width = 400 height = 450/>
# 

# In[38]:


model1 = RandomForestClassifier(random_state=0)
param_grid = { 
    'n_estimators': [ 100,200,300,400,500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# In[39]:


#Commenting Out As Taking Much Time
#Optuna may be Good Choice

# from sklearn.model_selection import GridSearchCV
# CV_rfc = GridSearchCV(estimator=model1, param_grid=param_grid, cv= 5)
# CV_rfc.fit(train2,y_train )
# CV_rfc.best_params_


# # Prediction

# In[40]:


model1 = RandomForestClassifier(random_state=0, n_estimators= 200, criterion = 'gini',max_features = 'auto',max_depth = 8)


# In[41]:


model1.fit(train2, y_train)


# In[42]:


pred=model1.predict(test2)
print(pred)


# In[43]:


df_submission.Survived=pred
df_submission


# In[44]:


# df_submission.to_csv('s4.csv',index=False)


# <img src="https://media.tenor.com/ZrFooc6A9ysAAAAC/goodgoodgeneral-mental-health.gif" width= 450 height = 400 />

# ### **I Will Update This If Some Experiments Would be Successfull.**
# ### **If You Came So Far & liked It , Could Share Your Feedback & Make An 👍**

# <img src="https://media.tenor.com/OnlexijkJI4AAAAC/thanks-you-tube.gif" width= 450 height = 400 />
# <a id="8"></a>

# In[ ]:




