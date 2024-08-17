#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


train_df = pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId')

train_df.head()


# In[3]:


train_df.info()


# In[4]:


plt.figure(figsize=(8,8))
sns.distplot(a=train_df['Age'], kde=True)
plt.show


# In[5]:


plt.figure(figsize=(8,8))
sns.stripplot(x="Survived", y="Age", data=train_df)
plt.show


# In[6]:


plt.figure(figsize=(8,8))
sns.stripplot(x="Survived", y="Fare", data=train_df)
plt.show


# In[7]:


train_df.groupby("Embarked").Survived.count().plot.bar()


# In[8]:


train_df.groupby('Survived').Pclass.count().plot.bar()


# In[9]:


train_df.groupby('SibSp').Survived.count().plot.bar()


# In[10]:


df2 = train_df.drop(['Name','Ticket','Cabin'], axis=1)

df2.head()


# In[11]:


df2.isnull().sum()


# In[12]:


df2.Age.fillna(df2.Age.mean(), inplace=True)
df2.Embarked.fillna("S", inplace=True)
df2.isnull().sum()


# In[13]:


y = df2.Survived
X  = df2.drop('Survived', axis=1)

for col in X.columns:
    if X[col].dtype == 'object':
        X[col],_= X[col].factorize()
        
        
mi_score = mutual_info_classif(X,y, random_state=1)

miscore = pd.Series(mi_score*100 , name="MI_score" , index=X.columns)

print(miscore) 


# # Model training

# In[14]:


# splitting dataset into training and validation
X_train, X_valid, y_train, y_valid = train_test_split(X,y, train_size=0.80, test_size=0.20,random_state=0)

# model object
model = RandomForestClassifier(n_estimators=500, random_state=0)

# fitting model to data
model.fit(X_train,y_train)

# predicting values

preds = model.predict(X_valid)

# accuracy and confusion matrix

acc = accuracy_score(y_valid,preds)

print("accuracy score for classification: ",acc)

conf = confusion_matrix(y_valid,preds)

print(conf)

# precision score

score = precision_score(y_valid,preds)

print(score)

# roc and auc score

rocauc = roc_auc_score(y_valid,preds)

print("ROC and AUC score: ", rocauc)

# roc curve

curve = roc_curve(y_valid,preds)
print(curve)


# In[15]:


# prediction on test data

test_df = pd.read_csv('/kaggle/input/titanic/test.csv', index_col = 'PassengerId')

test_df.info()


# # Prediciton on test data

# In[16]:


testdf2 = test_df.drop(['Name','Ticket','Cabin'], axis=1)

testdf2.isnull().sum()

testdf2.Age.fillna(testdf2.Age.mean(), inplace=True)
testdf2.Fare.fillna(testdf2.Fare.mean(), inplace=True)

# factorize object columns

for col in testdf2.columns:
    if testdf2[col].dtype == 'object':
        testdf2[col],_= testdf2[col].factorize()

# prediction on survival

prediction = model.predict(testdf2)

# submission csv

mysub = pd.DataFrame({'PassengerId':testdf2.index,
                       'Survived': prediction})

mysub.to_csv('mysubmission.csv', index=False)


# # Gradient Boosting classifier model training

# In[17]:


clf = GradientBoostingClassifier(n_estimators=500,
                                 learning_rate=0.01, 
                                 max_leaf_nodes=150,
                                 random_state=1,
                                 max_depth=4)


clf.fit(X_train,y_train)

prediction = clf.predict(X_valid)

# accuracy score

accu = accuracy_score(y_valid, prediction)

print("accuracy score of GBC:", accu)

# confusion metrix

con = confusion_matrix(y_valid,prediction)

print(con)


# # Prediction on testing data

# In[18]:


test_pred = clf.predict(testdf2)

sub = pd.DataFrame({'PassengerId':testdf2.index,
                    'Survived':test_pred})

sub.to_csv('submission.csv', index=False)


# # Kmeans clustring

# In[19]:


# kmeans clustering 

y = df2.Survived
X2  = X.copy()

X2['Age'] = X2['Age'].map(lambda x: (x - X2['Age'].mean())/X2['Age'].std())
X2['Fare'] = X2['Fare'].map(lambda x: (x - X2['Fare'].mean())/X2['Fare'].std())

kmeans = KMeans(n_clusters=2, n_init=10, max_iter=10, random_state=0)

X2['Cluster'] = kmeans.fit_predict(X2)
X2['Cluster'] = X2['Cluster'].astype('category')

X2.head()


# In[20]:


sns.stripplot(x='Cluster', y='Age', data = X2)
plt.show


# In[21]:


df = pd.DataFrame({'age': np.array(X['Age']),
                   'Cluster': np.array(X2['Cluster']),
                    'Survived': y})

sns.stripplot(x='Survived', y='age', data=df)

plt.show


# In[22]:


sns.stripplot(x='Cluster', y='age', data=df)

plt.show


# # Feature engineering and improving accuracy of model

# > From above models we can see there's issue of overfiiting, because of that we are not getting desired accuracy of from test data
# 
# > Below code tries to solve that problem so we can improve accuracy and avoid overfitting

# In[23]:


# load the dataset again

data = pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId')

data.head()


# In[24]:


# Name and Ticket columns are instances/individual itself so we can remove from dataframe

data2 = data.drop(['Name','Ticket'], axis=1)

# EDA on numerical variables and it's releations with our target

num_col = [col for col in data2.columns if data2[col].dtype in ['int64','float64']]

d3 = data2[num_col]

# average of each variable who survived or not

d3.groupby('Survived').mean()


# > We can see that travellers from uppper class are more likely to survive
# 
# > Younger ones are more likely to survive than older age but there are some missing values to consider
# 
# > Subsequently we can see that upper class travellers as they are had to pay more to get onboard are more likely to survive
# 
# > Above all features can explain our target in some way so we will keep all features

# In[25]:


# given the age is availabel or not we have imporance of age feature 

d3.groupby(d3['Age'].isnull()).mean()


# In[26]:


d3['Age'].fillna(d3['Age'].mean(), inplace=True)

d3.info()


# In[27]:


# eda on categorical features

cat_features = ['Survived','Pclass','SibSp','Parch','Sex','Cabin','Embarked']

data2['Age'].fillna(data2['Age'].mean(), inplace=True)

data2[cat_features]

data2.info()
# we are having two variable with missing values


# In[28]:


for idx, col in enumerate(['Pclass','SibSp','Parch']):
    plt.figure(idx, figsize=(6,6))
    sns.catplot(x=col, y='Survived', data=data2, kind='point')


# > from above plots we can say that when siblings/family members increases survival ratio decreases

# In[29]:


data2.pivot_table('Survived', index='Embarked', columns='Pclass')

# since embarkation is not related to survival of passanger as we can see in below table
#  survival rations are depend on class of ticket


# In[30]:


plt.figure(figsize=(6,6))
sns.catplot(x='Embarked', y='Survived', data=data2, kind='point')
plt.show


# In[31]:


# relations between having cabin or not

# data2.groupby(data2['Cabin'].isnull()).mean()

# where cabin is not available, there we have less survival ratio and this validate by average fair 
#  of the ticket as well

data2['Cabin_cnt'] = np.where(data2['Cabin'].isnull(), 0,1)


# In[32]:


for col in data2.columns:
    if data2[col].isnull().sum() == True:
        data2.drop(col, axis=1) 


# In[33]:


sex_arr = np.array(train_df['Sex'].map({'male':0, 'female':1}))


# In[34]:


data2['Sex'] = data2['Sex'].fillna(pd.Series(sex_arr), inplace=True)

data3 = data2.drop('Sex', axis=1)
data3.head()


# In[35]:


data3['Sex_'] = sex_arr


# In[36]:


data3.head()


# In[37]:


data3.drop(['Cabin','Embarked'], axis=1)


# # Traning model via gradient boosting

# In[38]:


X = data3[['Pclass','Age','SibSp','Parch','Fare','Sex_','Cabin_cnt']]
y = train_df['Survived']

X_train, X_valid, y_train, y_valid = train_test_split(X,y, train_size=0.8, test_size=0.2,
                                                       random_state=0)

mymodel = GradientBoostingClassifier(n_estimators=2000,
                                 learning_rate=0.01, 
                                 max_leaf_nodes=500,
                                 random_state=0,
                                 max_depth=2)


# fitting model to data

mymodel.fit(X_train,y_train)

# prediction values

predictions = mymodel.predict(X_valid)

# accuracy score

score = accuracy_score(y_valid,predictions)

print("Accuracy score of the model: ", score)

# confusion matrix

confusion = confusion_matrix(y_valid,predictions)

print(confusion)

