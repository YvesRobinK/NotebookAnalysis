#!/usr/bin/env python
# coding: utf-8

# ## Hyperparamter Tuning + Sk Learn Pipelines
# The goals of this notebook is to make Pipelines and this competition easy for you.
# ___

# <a id="1"></a> 
# # Description:
# 
# Welcome to the year 2912, where your data science skills are needed to solve a cosmic mystery. We've received a transmission from four lightyears away and things aren't looking good.
# 
# The Spaceship Titanic was an interstellar passenger liner launched a month ago. With almost 13,000 passengers on board, the vessel set out on its maiden voyage transporting emigrants from our solar system to three newly habitable exoplanets orbiting nearby stars.
# 
# While rounding Alpha Centauri en route to its first destination—the torrid 55 Cancri E—the unwary Spaceship Titanic collided with a spacetime anomaly hidden within a dust cloud. Sadly, it met a similar fate as its namesake from 1000 years before. Though the ship stayed intact, almost half of the passengers were transported to an alternate dimension!
# 
# To help rescue crews and retrieve the lost passengers, you are challenged to predict which passengers were transported by the anomaly using records recovered from the spaceship’s damaged computer system.
# 
# Help save them and change history!
# ___

# > **Table of Contents:**
# > * [Description of the competiton](#1)
# > * [Importing Libraries](#2)
# > * [Loading Data](#3)
# > * [EDA - Exploratory Data Analysis](#4)
# > * [Feature Engineering](#5)
# > * [PreProcessing (Transformations)](#6)
# > * [ML X Pipelines](#8)
# > * [HyperParameter Tunning](#9)
# > * [Prediction and Submission](#10)
# > ---

# <a id="2"></a> 
# # 1. Importing Libraries 😀

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style='darkgrid', font_scale=2)
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Models
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn import set_config


# In[2]:


set_config(display='diagram')


# <a id="3"></a> 
# # 2- Loading the Data 📅

# In[3]:


df_train = pd.read_csv('../input/spaceship-titanic/train.csv')
df_test = pd.read_csv('../input/spaceship-titanic/test.csv')

df_train.head()


# **Columns Description**
# * PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
# * HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.
# * CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
# * Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
# * Destination - The planet the passenger will be debarking to.
# * Age - The age of the passenger.
# * VIP - Whether the passenger has paid for special VIP service during the voyage.
# * RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
# * Name - The first and last names of the passenger.
# * Transported - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

# <a id="4"></a> 
# # 3- Let's Explore 👓

# In[4]:


r1,c1 = df_train.shape
print('The training data has {} rows and {} columns'.format(r1,c1))
r2,c2 = df_test.shape
print('The validation data has {} rows and {} columns'.format(r2,c2))


# In[5]:


df_train.info()


# In[6]:


df_train.describe()


# In[7]:


df_test.describe()


# **Let's look into them**

# <a id="5"></a> 
# # 4.Feature Engineering 💎

# In[8]:


# Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
df_train[['Deck','Num','Side']] = df_train.Cabin.str.split('/',expand=True)
df_test[['Deck','Num','Side']] = df_test.Cabin.str.split('/',expand=True)


# In[9]:


df_train['total_spent']= df_train['RoomService']+ df_train['FoodCourt']+ df_train['ShoppingMall']+ df_train['Spa']+ df_train['VRDeck']
df_test['total_spent']=df_test['RoomService']+df_test['FoodCourt']+df_test['ShoppingMall']+df_test['Spa']+df_test['VRDeck']


# In[10]:


df_train['AgeGroup'] = 0
for i in range(6):
    df_train.loc[(df_train.Age >= 10*i) & (df_train.Age < 10*(i + 1)), 'AgeGroup'] = i
# Same for test data
df_test['AgeGroup'] = 0
for i in range(6):
    df_test.loc[(df_test.Age >= 10*i) & (df_test.Age < 10*(i + 1)), 'AgeGroup'] = i


# In[11]:


plt.figure(figsize=(10,6))
sns.countplot(df_train.Deck,hue=df_train.Transported);


# In[12]:


df_train['Num'].nunique()


# In[13]:


plt.figure(figsize=(10,5))
sns.countplot(df_train.Side,hue=df_train.Transported)
plt.legend(loc=4);


# In[14]:


plt.figure(figsize=(10,6))
sns.countplot(y=df_train['AgeGroup'],hue=df_train['Transported']);


# <a id="6"></a> 
# # 5. Preprocessing 🤖 (Transformations)

# In[15]:


df_train.head()


# In[16]:


X=df_train.drop('Transported',axis=1)
y = df_train['Transported']


# In[17]:


X['Num'] = pd.to_numeric(X['Num'])


# In[18]:


X=X.drop(['PassengerId','Name'],axis=1)


# ## 5.B. Seperating Categorical and Numerical Columns

# In[19]:


cat_cols=X.select_dtypes('object').columns.to_list()
cat_cols


# In[20]:


num_cols=X.select_dtypes(exclude='object').columns.to_list()
num_cols


# <a id="7"></a> 
# # 5.C.Making Seperate Preprocessing Pipelines for numeric and categorical columns.

# In[21]:


numeric_preprocessor = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaling',StandardScaler()),
])


# In[22]:


categorical_preprocessor = Pipeline(steps=[
    ('encoder',OneHotEncoder(handle_unknown='ignore')),  
    ('imputer',SimpleImputer(strategy='constant')),
    
])


# ### Using Column Transfer to Transform numerical and categorical columns

# In[23]:


preprocessor = ColumnTransformer([
    ('categorical',categorical_preprocessor,cat_cols),
    ('numeric',numeric_preprocessor,num_cols)

])


# <a id="8"></a> 
# # 6. Pipeline TIME!!!😎

# ## Adding the preprocessor and model in the Pipeline

# In[24]:


Pipe = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',GradientBoostingClassifier())])


# In[25]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=1)


# In[26]:


Pipe.fit(X_train,y_train)


# In[27]:


pred=Pipe.predict(X_val)


# In[28]:


pred=Pipe.predict(X_train)
pred_y=Pipe.predict(X_val)
print('Train accuracy ',accuracy_score(y_train.values,pred))
print('Validation accuracy',accuracy_score(y_val.values,pred_y))


# <a id="9"></a> 
# # 7. Using Hyperparameter Tuning 🌠

# **Add 'model__' to the original hyperparameter name in the parameter grid. We are using 'model__' because in pipelines we refered GBC as 'model'**

# In[29]:


# you can try more parameters, but hell it takes a lot of time.
param_grid={'model__n_estimators':[500,1000],'model__learning_rate':[0.1,0.2],'model__verbose':[1],'model__max_depth':[2,3]}
from sklearn.model_selection import GridSearchCV
gcv=GridSearchCV(Pipe,param_grid=param_grid,cv=5,scoring="roc_auc")


# In[30]:


gcv.fit(X,y)


# In[31]:


params = gcv.best_params_
params


# In[32]:


gcv.best_score_


# In[33]:


Hyper_Pipe = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',GradientBoostingClassifier(n_estimators=500,max_depth=3, random_state=1)),
])
Hyper_Pipe.fit(X_train,y_train)


# In[34]:


pred=Hyper_Pipe.predict(X_train)
pred_y=Hyper_Pipe.predict(X_val)
print('Train accuracy ',accuracy_score(y_train.values,pred))
print('Validation accuracy',accuracy_score(y_val.values,pred_y))


# In[35]:


confusion_matrix(pred_y,y_val.values)


# <a id="10"></a> 
# # 8. Prediction and Submission 😎

# In[36]:


y_pred = Hyper_Pipe.predict(df_test)

sub=pd.DataFrame({'Transported':y_pred.astype(bool)},index=df_test['PassengerId'])

sub.head()


# In[37]:


sub.to_csv('submission')


# In[38]:


pd.read_csv('../input/spaceship-titanic/sample_submission.csv')


# ### <center>Thanks for reading:)</center>
# ### <center>Upvote! and Leave some suggestions</center>
# 

# In[ ]:




