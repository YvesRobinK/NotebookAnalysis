#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


# In[3]:


def label_encoder(c):
    lc = LabelEncoder()
    return lc.fit_transform(c)


# In[4]:


train = pd.read_csv('../input/tabular-playground-series-apr-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-apr-2021/test.csv')
submission = pd.read_csv('../input/tabular-playground-series-apr-2021/sample_submission.csv')


# In[5]:


train.head()


# ### A Quick Information about the features:
# 
# #### pclass - Passenger Ticket class : Class 1, 2 and 3.
# 
# #### Name - Name of the passenger
# 
# #### sex - Sex of the Passenger
# 
# #### Age - Age in years of the Passenger
# 
# #### sibsp - Number of siblings / spouses aboard the Titanic
# 
# #### Parch - Number of parents / children aboard the Titanic
# 
# #### Ticket - Ticket number
# 
# #### Fare - Passenger fare
# 
# #### Cabin - Cabin number
# 
# #### Embarked - Port of Embarkation shows the port from which the passenger boarded the titanic
# #### where the ports are    C - Cherbourg, Q - Queenstown, & S - Southampton

# In[6]:


train.info()


# In[7]:


test.info()


# In[8]:


print(train.shape)
print(test.shape)


# In[9]:


df_Big=train.append(test,sort=False)
df_Big.head()


# In[10]:


df_Big.shape


# ### Data Preprocessing & EDA!

# In[11]:


df_Big.isnull().sum()


# In[12]:


df_Big=df_Big.drop(['Cabin'],axis=1)
df_Big=df_Big.drop(['PassengerId'],axis=1)


# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='Sex',hue='Survived',data=train)


# In[14]:


plt.hist(x=df_Big.Age, bins=10)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Age')
plt.ylabel('Frequency')


# In[15]:


sns.countplot(x='Parch',hue='Survived',data=train)

#Passengers with 0 ,4,5,6 Parch are less likely to survive than 1,2,3


# In[16]:


df_Big.head()


# In[17]:


df_Big['Age'].fillna((df_Big['Age'].mean()),inplace=True)
df_Big['Fare'].fillna((df_Big['Fare'].mean()),inplace=True)


# In[18]:


df_Big.isnull().sum()


# In[19]:


sns.countplot(x='Embarked',hue='Survived',data=df_Big)


# In[20]:


df_Big['Embarked'].fillna("S",inplace=True)


# In[21]:


df_Big['Ticket'] = df_Big['Ticket'].str.replace('[^a-zA-Z]', '').str[:1]
df_Big['Ticket'] = df_Big['Ticket'].str.strip()


# In[22]:


df_Big['Ticket'] = df_Big['Ticket'].fillna('ZZ')


# In[23]:


df_Big.loc[df_Big['Ticket']=='', 'Ticket']='ZZ'


# In[24]:


df_Big.loc[df_Big['Ticket']=='L', 'Ticket']='ZZ'

df_Big.groupby(by=['Ticket'])['Survived'].mean()


# In[25]:


df_Big['Ticket'].value_counts()


# In[26]:


df_Big.head()


# In[27]:


df_Big.isnull().sum()


# In[28]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()


# In[29]:


df_Big["Sex"]=encoder.fit_transform(df_Big['Sex'])
df_Big["Embarked"]=encoder.fit_transform(df_Big['Embarked'])
df_Big["Ticket"]=encoder.fit_transform(df_Big['Ticket'])


# In[30]:


df_Big.head()


# In[31]:


df_Big['FirstName'] = df_Big['Name'].apply(lambda x:x.split(', ')[0])
df_Big['SecondName'] = df_Big['Name'].str.split(', ', 1, expand=True)[1]


# In[32]:


le = LabelEncoder()
le1 = LabelEncoder()
df_Big['FirstName'] = le.fit_transform(df_Big['FirstName'])
df_Big['SecondName'] = le1.fit_transform(df_Big['SecondName'])


# In[33]:


df_Big.head()


# In[34]:


df_Big=df_Big.drop(['Name'],axis=1)


# In[35]:


# introducing a new feature : the size of families (including the passenger)
df_Big['FamilySize'] = df_Big['Parch'] + df_Big['SibSp'] + 1


# In[36]:


# introducing other features based on the family size
df_Big['Singleton'] = df_Big['FamilySize'].map(lambda s: 1 if s == 1 else 0)
df_Big['SmallFamily'] = df_Big['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
df_Big['LargeFamily'] = df_Big['FamilySize'].map(lambda s: 1 if 5 <= s else 0)


# In[37]:


df_Big.head()


# In[38]:


df_Big=df_Big.drop(['FamilySize'],axis=1)


# In[39]:


df_Big.isnull().sum()


# ### Feature Engineering!

# In[40]:


plt.figure(figsize=(15,10))
sns.heatmap(data=df_Big.corr())


# In[41]:


df_Big=df_Big.drop(['Singleton'],axis=1)


# In[42]:


plt.figure(figsize=(15,10))
sns.heatmap(data=df_Big.corr())


# In[43]:


print(train.shape)
print(test.shape)


# In[44]:


df_train=df_Big[0:100000]
df_test=df_Big[100000:]


# In[45]:


X=df_train.drop(['Survived'],axis=1)
y=df_train.Survived


# In[46]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.18,random_state=33)


# In[47]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression


# In[48]:


# feature selection
f_selector = SelectKBest(score_func=f_regression, k='all')
# learn relationship from training data
f_selector.fit(X_train, y_train)


# In[49]:


# transform train input data
X_train_fs = f_selector.transform(X_train)
# transform test input data
X_test_fs = f_selector.transform(X_test)


# In[50]:


# Plot the scores for the features
plt.bar([i for i in range(len(f_selector.scores_))], f_selector.scores_)
plt.xlabel("feature index")
plt.ylabel("F-value (transformed from the correlation values)")
plt.show()


# In[51]:


f_selector.scores_


# In[52]:


score = f_selector.scores_
Best_Features = pd.DataFrame({'Feature_Score': score})
Best_Features.head(5)


# In[53]:


Names = df_train.columns
Names


# In[54]:


Column_Name = pd.DataFrame({'Column_Name': Names})
Merged = pd.concat([Column_Name, Best_Features], axis=1)
Merged.sort_values(['Feature_Score'], ascending=False).head(20)


# #### Here we can see in future tuning these parameters, whether to keep these least important features or not!

# In[55]:


from sklearn.metrics import classification_report, roc_auc_score, make_scorer, accuracy_score, roc_curve
import optuna
from math import sqrt
import lightgbm as lgb


# In[56]:


lgb_train = lgb.Dataset(X_train, y_train)
lgb_valid = lgb.Dataset(X_test, y_test, reference=lgb_train)


# In[57]:


def objective(trial):    
    params = {
        'reg_alpha' : trial.suggest_loguniform('reg_alpha' , 1e-5 , 12),
        'reg_lambda' : trial.suggest_loguniform('reg_lambda' , 1e-5 , 12),
        'num_leaves' : trial.suggest_int('num_leaves' , 11 , 900),
        'learning_rate' : trial.suggest_uniform('learning_rate' , 0.0000001 , 0.2),
        'max_depth' : trial.suggest_int('max_depth' , 5 , 400),
        'n_estimators' : trial.suggest_int('n_estimators' , 1 , 9999),
        'min_child_samples' : trial.suggest_int('min_child_samples' , 1 , 110),
        'min_child_weight' : trial.suggest_loguniform('min_child_weight' , 1e-5 , 1),
        'subsample' : trial.suggest_uniform('subsample' , 1e-5 , 1.0),
        'colsample_bytree' : trial.suggest_loguniform('colsample_bytree' , 1e-5 , 1),
        'random_state' : trial.suggest_categorical('random_state' , [2,22,222,2222]),
        'metric' : 'accuracy',
        'device_type' : 'cpu',
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train , y_train , eval_set = [(X_test , y_test)] ,eval_metric='logloss', early_stopping_rounds = 1000 , \
             verbose = False)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test , preds)
    return acc


# In[58]:


study = optuna.create_study(direction = 'maximize')
study.optimize(objective, n_trials = 10)


# In[63]:


# print('numbers of the finished trials:' , len(study.trials))
# print('the best params:' , study.best_trial.params)
# print('the best value:' , study.best_value)


# Well, after executing optuna, I have used the best parameters in Model initialized below!

# In[64]:


import lightgbm as lgb
clf = lgb.LGBMClassifier(num_leaves=39, learning_rate=0.05, max_depth=28, n_estimators= 20000,
min_child_weight=0.0023505426039016975, min_child_samples=71, reg_alpha=13.0124692806962, reg_lambda=17.429087848443793)
clf.fit(X_train, y_train)


# In[67]:


# df_test = df_test.drop("Survived")


# In[68]:


pred = clf.predict(X_test)

print(accuracy_score(y_test, pred))


# In[69]:


df_test.head()


# In[70]:


df_test=df_test.drop(['Survived'],axis=1)


# In[71]:


prediction = clf.predict(df_test)
prediction


# In[72]:


test_new = pd.read_csv('../input/tabular-playground-series-apr-2021/test.csv')
test_new.head()


# In[73]:


# output = pd.DataFrame({'PassengerId': test_new['PassengerId'], 'Survived': prediction})
# output.to_csv('TabularSeriesStarterSubmission_Apr1.csv', index=False)


# In[ ]:




