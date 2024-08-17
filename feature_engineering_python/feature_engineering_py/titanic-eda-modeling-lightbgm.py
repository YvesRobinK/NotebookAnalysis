#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

import scipy
from scipy.stats import norm

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # EDA and Visualization

# In[2]:


PATH = '/kaggle/input/titanic/'


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test  = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[3]:


print(train.info(),test.info())


# In[4]:


train.drop(['PassengerId','Cabin','Ticket','Name'], axis=1, inplace=True)



test.drop(['PassengerId','Cabin','Ticket','Name'], axis=1, inplace=True)


# In[5]:


plt.figure(figsize=(6,5))
sns.heatmap(train.corr(),annot=True,cmap='coolwarm',mask=np.triu(np.ones_like(train.corr(),dtype='bool')))
plt.grid(True)
plt.show()


# In[6]:


train['Family'] = train['SibSp'] + train['Parch']



test['Family']  = test['SibSp'] + test['Parch']


# In[7]:


plt.subplots(3,4,figsize=(22,12))
plt.subplot(341)
x = train['Survived'].value_counts()
plt.pie(x.values,startangle=122,labels=[0,1],wedgeprops={"edgecolor":"white", "width":0.5},radius=1.1,colors=["skyblue", "gold"],autopct="%1.1f%%",pctdistance=0.75)
plt.title('Distribution of Survived',y=0.45)
plt.ylabel('')
plt.subplot(342)
sns.countplot(train['Survived'],palette='cool',linewidth=1.0,edgecolor="gold")
plt.ylabel('')
plt.subplot(343)
sns.countplot(train['Pclass'],palette='cool',linewidth=1.0,edgecolor="gold")
plt.ylabel('')
plt.subplot(344)
sns.countplot(train['Sex'],palette='cool',linewidth=1.0,edgecolor="gold")
plt.ylabel('')
plt.subplot(345)
sns.distplot(train['Age'],color='violet',fit=norm,fit_kws={'color':'crimson'})
plt.ylabel('')
plt.subplot(346)
sns.countplot(train['SibSp'],palette='cool_r',linewidth=1.5,edgecolor="gold")
plt.ylabel('')
plt.subplot(347)
sns.countplot(train['Parch'],palette='cool_r',linewidth=1.5,edgecolor="gold")
plt.ylabel('')
plt.subplot(348)
sns.countplot(train['Family'],palette='cool_r',linewidth=1.5,edgecolor="gold")
plt.ylabel('')
plt.subplot(349)
sns.distplot(train['Fare'],color='violet',fit=norm,fit_kws={'color':'crimson'})
plt.ylabel('')
plt.subplot(3,4,10)
x = train['Embarked'].value_counts()
plt.pie(x.values,startangle=122,labels=[0,1,2],wedgeprops={"edgecolor":"white", "width":0.5},radius=1.1,colors=["skyblue","gold","lightgreen"],autopct="%1.1f%%",pctdistance=0.75)
plt.title('Distribution of Embarked',y=0.45)
plt.subplot(3,4,11)
sns.countplot(train['Embarked'],palette='cool',linewidth=1.5,edgecolor="gold")
plt.ylabel('')
plt.subplot(3,4,12)
sns.heatmap(train.corr(),cmap='coolwarm')
plt.show()


# #### Let's see if the number of family members has an impact on survival rates, by gender.

# In[8]:


plt.figure(figsize=(12,4))
graph = pd.pivot_table(train,index='Family',columns='Sex',values='Survived',aggfunc='sum')
plt.plot(list(graph.index),graph['male'],label='Male',color='deepskyblue')
plt.plot(list(graph.index),graph['female'],label='Female',color='hotpink')
plt.legend()
plt.grid(True)
plt.show()


# # Feature Engineering

# In[9]:


train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])


# In[10]:


train['Sex']      = train['Sex'].replace({'male':0, 'female':1})
train['Embarked'] = train['Embarked'].replace({'C':0, 'Q':1, 'S':2})



test['Sex']       = test['Sex'].replace({'male':0, 'female':1})
test['Embarked']  = test['Embarked'].replace({'C':0, 'Q':1, 'S':2})


# #### I completed the missing values based on the information in the other columns.

# In[11]:


# # I checked the correlation coefficients between columns with missing values and other columns.

def check(df, column):
    col = np.abs(df.corr()[column])
    print(col.sort_values(ascending=False).head(13))


# In[12]:


def missing_value(df,column,column2,column3,column4):
    target = df[[column,column2,column3,column4]]
    notnull = target[target[column].notnull()].values
    null = target[target[column].isnull()].values
    X = notnull[:, 1:]
    y = notnull[:, 0]
    rf = RandomForestRegressor(random_state=0,n_estimators=1000,n_jobs=-1)
    rf.fit(X,y)
    predict = rf.predict(null[:, 1::])
    print(predict)
    df.loc[(df[column].isnull(), column)] = predict


# In[13]:


check(train, 'Age')


# In[14]:


missing_value(train,'Age','Pclass','SibSp','Parch')


# In[15]:


missing_value(test,'Age','Pclass','SibSp','Parch')


# In[16]:


print(train.info(),test.info())


# In[17]:


test['Fare'] = test['Fare'].fillna(test['Fare'].median())


# In[18]:


# After adding 1, take the logarithm

train['Fare_log'] = np.log1p(train['Fare'])



test['Fare_log']  = np.log1p(test['Fare'])


# In[19]:


plt.subplots(1,4,figsize=(18,4))
plt.subplot(141)
sns.distplot(train['Fare'],color='blueviolet')
plt.ylabel('')
plt.subplot(142)
sns.distplot(test['Fare'],color='blueviolet')
plt.ylabel('')
plt.subplot(143)
sns.distplot(train['Fare_log'],color='blueviolet',fit=norm,fit_kws={'color':'gold'})
plt.ylabel('')
plt.subplot(144)
sns.distplot(test['Fare_log'],color='blueviolet',fit=norm,fit_kws={'color':'gold'})
plt.ylabel('')
plt.show()


# In[20]:


# Binning

train['Fare_bin'] = pd.cut(train['Fare'], 10)
train['Age_bin']  = pd.cut(train['Age'], 10)



test['Fare_bin']  = pd.cut(test['Fare'], 10)
test['Age_bin']   = pd.cut(test['Age'], 10)


# In[21]:


train['Fare_log^2'] = train['Fare_log'] * train['Fare_log']



test['Fare_log^2']  = test['Fare_log'] * test['Fare_log']


# In[22]:


train.head()


# In[23]:


test.head()


# In[24]:


oe = OrdinalEncoder()

encoded = oe.fit_transform(train[['Fare_bin','Age_bin']].values)
train[['Fare_bin','Age_bin']] = encoded



encoded = oe.fit_transform(test[['Fare_bin','Age_bin']].values)
test[['Fare_bin','Age_bin']] = encoded


# In[25]:


X_train = train.drop('Survived', axis=1)
y_train = train['Survived']
X_test  = test.copy()


# # Modeling 

# In[26]:


X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=0, shuffle=True)


# #### Using lightGBM.

# In[27]:


categorical_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Fare_bin', 'Age_bin', 'Family']


lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)
lgb_eval  = lgb.Dataset(X_valid, y_valid, reference=lgb_train,categorical_feature=categorical_features)


# In[28]:


# grid search

import time
t1 = time.time()

model = lgb.LGBMClassifier()

param_grid = {"max_depth": [4,5,6],
              "learning_rate" : [0.04,0.050,0.055],
              "num_leaves": [60,65,70],
              "n_estimators": [10000],
              "objective": ['binary']
             }

grid_result = GridSearchCV(estimator = model,
                           param_grid = param_grid,
                           scoring = 'accuracy',
                           cv = 2,
                           return_train_score = False,
                           n_jobs = -1)

grid_result.fit(X_train,y_train)

t2 = time.time()

print(grid_result.best_estimator_)
print((grid_result.best_params_))
print((t2-t1)/60)


# In[29]:


lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)
lgb_eval  = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_features)

params={
    'num_leaves':65,
    'objective':'binary',
    'max_depth':5,
    'learning_rate':.005,
    'n_estimators':100000000,
    'early_stopping_rounds':30,
}
num_round=1000


# In[30]:


model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], num_boost_round=num_round, categorical_feature=categorical_features, verbose_eval=False)


# In[31]:


y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)


# In[32]:


sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
sub['Survived'] = list(map(int, y_pred))
sub.to_csv('submission.csv', index=False)


# # Conclusion

# #### After feature engineering and creating a model with lightBGM, the score reached 0.78947.
# 
# #### This is a high score, reaching the top 5%.
# 
# #### Binning, and logarithmic transformation for fares led to a significant improvement in the score.
# 
# #### I also believe that the ingenious supplementation of "Age" was also a factor in the score improvement.
