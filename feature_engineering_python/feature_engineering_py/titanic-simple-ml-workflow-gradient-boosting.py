#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip uninstall sklearn --yes


# In[2]:


get_ipython().system('pip install pycaret')


# # Titanic - Machine Learning from Disaster
# 
# # Table of Contents
# 
# *  [1) Introduction](#intro)
# 
# *  [2) Getting started](#getting_started)
#     *  [2.1) Libraries](#lib)
#     *  [2.2) Acquire Data](#acq)
#     *  [2.3) Combine Train and Test Sets](#combine)
#     *  [2.4) Understand the data](#Und)
#  
# *  [3) Exploratory Data analysis](#exp)
# 
# *  [4) Data Cleaning](#dc)
#     *  [4.1) Categorical Missing Values](#cmv)
#     *  [4.1) Numeric  Missing Values](#nmv)
# *  [5) Feature Engineering](#fe)
# *  [6) Data Preprocessing for Model](#dpm)
#     *  [6.1) Standardize data](#sd1)
#     *  [6.2) Split data](#sd2)
# *  [7) Model Selection](#ms)
#     *  [7.1) Hyperparameter Optimization](#ho)
#     *  [7.2) Evaluation](#ev)
# *  [8) Submission](#sub)
# 
# 
# 

# ## References for data engineering and EDA parts
# 
# * [Titanic Data Science Solutions](https://www.kaggle.com/code/startupsci/titanic-data-science-solutions)
# * [Titanic - Advanced Feature Engineering Tutorial](https://www.kaggle.com/code/gunesevitan/titanic-advanced-feature-engineering-tutorial)
# * [A Statistical Analysis & ML workflow of Titanic](https://www.kaggle.com/code/masumrumi/a-statistical-analysis-ml-workflow-of-titanic)

# <a id="intro"></a>
# # 1) Introduction

# This is my first machine learning work in Kaggle, which includes basic steps. I have reviewed many studies for the parts of data engineering, and I have added these notebooks to the reference section.

# <a id="getting_started"></a>
# # 2) Getting Started

# <a id="lib"></a>
# ### 2.1) Libraries

# In[3]:


# Data Analysis and wrangling
import pandas as pd
import numpy as np
import pandas_profiling

pd.set_option('max_columns', None)
pd.set_option('max_rows', 90)
import warnings
warnings.filterwarnings('ignore')


# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':(11,8)})

# Preprocessing for models
from sklearn.preprocessing import StandardScaler

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from pycaret.classification import *
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier


# Model Performance
from sklearn import metrics 
from sklearn.metrics import classification_report, plot_roc_curve, plot_confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score


# <a id="acq"></a>
# ## 2.2) Acquire Data

# In[4]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
train0=pd.read_csv('/kaggle/input/titanic/train.csv')
test0=pd.read_csv('/kaggle/input/titanic/test.csv')


# <a id="combine"></a>
# ## 2.3) Combine Train and Test Sets

# In[5]:


target = train0['Survived']
test_ids = test0['PassengerId']

train1 = train0.drop(['PassengerId','Survived','Ticket','Cabin'], axis=1)
test1 = test0.drop(['PassengerId','Ticket','Cabin'], axis=1)

data1 = pd.concat([train1, test1], axis=0).reset_index(drop=True)
data1.info()
print('-'*40)
data1


# In[6]:


print(train0.isna().sum())
print('-'*40)
print(data1.isna().sum())


# <a id="Und"></a>
# ## 2.4) Understand the Training Data

# In[7]:


train0.head()


# ### Categorical Variables
#    *  Nominal
#        *  Survived
#        *  Embarked
#        *  Sex
#    *  Ordinal
#        
#        *  Pclass 
# 
# 
# ### Numerical Variables
# 
#    *  Continous
#         *  Age
#         *  Fare
#    * Discrete
#         *  SibSp
#         *  ParCh
#     
# ### Undefined variables
# 
#  *  Name
#  *  Ticket
#  *  Cabin

# <a id="exp"></a>
# #  3) Exploratory Data Analysis

# In[8]:


sum_of_survived = train0['Survived'].value_counts()[1]
sum_of_not_survived = train0['Survived'].value_counts()[0]

plt.figure(figsize=(8,6))
sns.set_context('notebook',font_scale=1)
sns.set_style('darkgrid')
sns.countplot(data=train0, x='Survived',palette='OrRd')

plt.title('')
plt.ylabel('Number of passengers')
plt.xlabel('')
plt.xticks((0,1),['Not Survived ({})'.format(sum_of_not_survived),"Survived ({})".format(sum_of_survived)])
plt.show()


# In[9]:


male_not_survived = train0.loc[(train0['Sex'] == 'male'),'Survived'].value_counts()[0]
female_not_survived = train0.loc[(train0['Sex'] == 'female'),'Survived'].value_counts()[0]
Per_not_male = round(male_not_survived *100 / (male_not_survived + female_not_survived),2)
Per_not_female = round(female_not_survived *100 / (male_not_survived + female_not_survived),2)

male_survived = train0.loc[(train0['Sex'] == 'male'),'Survived'].value_counts()[1]
female_survived = train0.loc[(train0['Sex'] == 'female'),'Survived'].value_counts()[1]
Per_male = round(male_survived *100 / (male_survived + female_survived),2)
Per_female = round(female_survived *100 / (male_survived + female_survived),2)

plt.figure(figsize=(8,6))
sns.set_context('notebook',font_scale=0.9)
sns.set_style('darkgrid')
sns.countplot(data=train0, x='Survived',hue='Sex',palette='BuPu_r')

plt.title('')
plt.ylabel('Number of passengers')
plt.xlabel('')
plt.xticks((0,1),['Not Survived (%{} male %{} female)'.format(Per_not_male,Per_not_female),"Survived (%{} male %{} female)".format(Per_male,Per_female)])
plt.show()


# In[10]:


print(train0[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False))
ax = sns.countplot(data=train0, x='Survived',hue='Pclass',palette='BuPu_r')
for container in ax.containers:
    ax.bar_label(container)

plt.show()


# In[11]:


temp = train0.copy()
temp['Family'] = temp['SibSp'] + temp['Parch'] + 1
print(temp[['Family','Survived']].groupby(['Family'],as_index=False).mean().sort_values(by='Survived',ascending=False))
ax = sns.countplot(data=temp, x='Survived',hue='Family',palette='coolwarm')
for container in ax.containers:
    ax.bar_label(container)
ax.legend(loc='upper right')

plt.show()


# In[12]:


sns.histplot(data=train0, x="Age",hue='Pclass',multiple="stack",kde=True)


# In[13]:


train0.corr().loc['Age','Pclass']


# <a id="dc"></a>
# # 4) Data Cleaning

# In[14]:


data2 = data1.copy()
data2.isna().sum()


# <a id="cmv"></a>
# ### 4.1) Categorical Missing Values

# In[15]:


data2['Embarked'] = data2["Embarked"].fillna(data2['Embarked'].mode()[0])


# <a id="nmv"></a>
# ### 4.1) Numeric Missing Values

# In[16]:


def knn_impute(df,na_target):
    df=df.copy()
    
    numeric_df = df.select_dtypes(np.number)
    non_na_columns = numeric_df.loc[:,numeric_df.isna().sum() == 0].columns
    
    y_train = numeric_df.loc[numeric_df[na_target].isna()==False,na_target]
    X_train = numeric_df.loc[numeric_df[na_target].isna()==False,non_na_columns]
    X_test = numeric_df.loc[numeric_df[na_target].isna()==True,non_na_columns]
    
    knn = KNeighborsRegressor()
    knn.fit(X_train,y_train)
    
    y_pred = knn.predict(X_test)
    
    df.loc[df[na_target].isna()==True,na_target] = y_pred
    
    return df


# In[17]:


for column in ['Age','Fare'
]:
    data2 = knn_impute(data2, column)


# In[18]:


data2.isna().sum()


# <a id="fe"></a>
# # 5. Feature Engineering

# In[19]:


data3 = data2.copy()


# In[20]:


#Mapping Sex
data3['Sex'] = data3['Sex'].map({'male':1,'female':0}).astype(int)

#Mapping Embarked
data3['Embarked'] = data3['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[21]:


data3['Title'] = data3['Name'].str.extract('([A-Za-z]+)\.',expand=False)

data3['Title'].replace(['Mlle','Ms','Lady'], 'Miss',inplace=True)
data3['Title'].replace(['Mme'], 'Mrs',inplace=True)
data3['Title'].replace(['Countess','Capt','Col','Don','Dr',
                        'Major','Rev',"Sir","Jonkheer",'Dona'], 'Rare',inplace=True)
data3.drop(['Name'],inplace=True, axis=1)
Tempdate = data3.copy()
Tempdate['Survived'] = target
Tempdate.loc[:train0.index.max(),:][['Title','Survived']].groupby(['Title'],as_index=False).mean().sort_values(by='Survived',ascending=True)

data3['Title'] = data3['Title'].map({'Master':0,'Miss':1,'Mrs':1,'Mr':2,'Rare':3})


# <a id="dpm"></a>
# # 6) Data Preprocessing for Models

# In[22]:


data4 = data3.copy()


# In[23]:


data4.info()


# <a id="sd1"></a>
# ## 6.1) Standardize data

# In[24]:


scaler = StandardScaler()
scaler.fit(data4)
data4 = pd.DataFrame(scaler.transform(data4),index=data4.index,columns=data4.columns)


# <a id="sd2"></a>
# ## 6.2) Split data

# In[25]:


train_final = data4.loc[:train0.index.max(), :].copy()
test_final = data4.loc[train0.index.max() + 1:, :].reset_index(drop=True).copy()


# In[26]:


train_final.info()
print('_'*40)
test_final.info()


# <a id="ms"></a>
# # 7) Model Selection

# In[27]:


_ = setup(data = pd.concat([train_final,target],axis=1),target='Survived')


# In[28]:


compare_models()


# In[62]:


models = { 'gbc': GradientBoostingClassifier(),
          'catboost':CatBoostClassifier(),
          'ridge':RidgeClassifier(),
          'log_reg':LogisticRegression(),
          'random_f':RandomForestClassifier(),
          'svc': SVC()
    }


# In[63]:


for name, model in models.items():
    model.fit(train_final,target)
    print(name+ 'trained')


# In[73]:


results = {}
kf = KFold(n_splits=15)
for name, model in models.items():
    result = cross_val_score(model,train_final,target,scoring='accuracy',cv=kf)
    results[name] = result


# In[74]:


for name, result in results.items():
    print(name + '\n---------')
    print(np.mean(result))
    print(np.std(result))


# In[75]:


df = pd.DataFrame(results)


# In[76]:


fig, axs = plt.subplots(2, 3, figsize=(7, 7))

sns.histplot(data=df, x="gbc", kde=True, color="skyblue", ax=axs[0, 0])
sns.histplot(data=df, x="catboost", kde=True, color="olive", ax=axs[0, 1])
sns.histplot(data=df, x="ridge", kde=True, color="gold", ax=axs[0, 2])
sns.histplot(data=df, x="log_reg", kde=True, color="teal", ax=axs[1, 0])
sns.histplot(data=df, x="random_f", kde=True, color="teal", ax=axs[1, 1])
sns.histplot(data=df, x="svc", kde=True, color="teal", ax=axs[1, 2])

plt.show()


# In[72]:


plt.figure(figsize=(16,10))
sns.displot(result,bins=10,kde=True)


# ## 7.1) Hyperparameter Optimization

# In[80]:


max_features = list(range(1,train_final.shape[1]))


# In[84]:


parameters = {
    'max_features':max_features,
    'n_estimators':[5,10,20,100,250],
    'max_depth':[1,3,5,7,9],
    'learning_rate':[0.01,0.05,0.1]
}


gbc = GridSearchCV(GradientBoostingClassifier(), parameters, cv=10)


gbc.fit(train_final, target)



# In[85]:


print(gbc.score(train_final, target))
print(gbc.best_params_)


# <a id="ev"></a>
# ## 7.2) Evaluation

# In[92]:


gbc = GradientBoostingClassifier(learning_rate= 0.05, loss= 'deviance', max_depth=5, n_estimators= 100,max_features=6)
gbc.fit(train_final,target)
pred = gbc.predict(train_final)
print(classification_report(target,pred))


# In[90]:


final_predictions = gbc.predict(test_final)


# <a id="sub"></a>
# # 8) Submission

# In[91]:


submission = pd.concat([test_ids,pd.Series(final_predictions,name="Survived")],axis=1)
submission.to_csv("./submission.csv",index=False,header=True)


# ## Thank you for checking my notebook, I will be glad to share your thoughts.
