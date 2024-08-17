#!/usr/bin/env python
# coding: utf-8

# A huge thanks to the following notebooks from which I learnt a lot:
# 
# - https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial
# - https://www.kaggle.com/sinakhorami/titanic-best-working-classifier
# - https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling
# - https://www.kaggle.com/wikaiqi/titaniclearningqi
# - https://www.kaggle.com/startupsci/titanic-data-science-solutions
# 
# Go check them out !

# # Import libraries

# In[1]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


# # Import dataset

# In[2]:


train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
full_data = [train,test]


# In[3]:


# we delete "Cabin" and "Ticket" features because not relevant
del train['Cabin']
del train['Ticket']
del test['Cabin']
del test['Ticket']


# # Check for missing values

# In[4]:


train.isnull().sum()


# In[5]:


test.isnull().sum()


# # Feature engineering

# In[6]:


# we add a "Title" feature with "Name" feature
for dataset in full_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[7]:


# let's check that all "Name" features had a "Title"
train.isnull().sum()
test.isnull().sum()


# In[8]:


# we realize some "Title" have few elements, so non relevant
pd.crosstab(train['Title'], train['Sex'])
pd.crosstab(test['Title'], test['Sex'])


# In[9]:


# we regroup some "Title"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Capt','Col','Countess','Don','Dona','Dr','Jonkheer','Lady','Major','Rev','Sir'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Mlle','Ms'], 'Miss')
    dataset['Title'] = dataset['Title'].replace(['Mme'], 'Mrs')


# In[10]:


# we study "Title" impact on "Survived"
print(train[['Title','Survived']].groupby(['Title'], as_index=False).mean())


# In[11]:


# we can now erase "Name"
del train['Name']
del test['Name']


# # "Age" feature (missing + mapping)

# In[12]:


# we deal with "age" missing values with "sex" and "pclass" median
train['Age'] = train.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
test['Age'] = test.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))


# In[13]:


train.isnull().sum()
test.isnull().sum()


# In[14]:


# we regroup "age"
train['AgeBand'] = pd.qcut(train['Age'], 5)    

for dataset in full_data:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4


# In[15]:


# we realize there is an issue on "Master" because they are not all classed as "0" and it is not  
pd.crosstab(train['Title'], train['Age'])


# In[16]:


# so we do it manually
i = 0
while i < len(train):
    if train['Title'][i] == 'Master':
        train['Age'][i] = 0
    i = i + 1
    
i = 0
while i < len(test):
    if test['Title'][i] == 'Master':
        test['Age'][i] = 0
    i = i + 1

pd.crosstab(train['Title'], train['Age'])
pd.crosstab(test['Title'], test['Age'])
del train['AgeBand']


# In[17]:


# We now need to deal with "Embarked" and "Fare" features
train.isnull().sum()
test.isnull().sum()


# 
# # "Embarked" feature (missing + mapping)

# In[18]:


# There is 2 "Embarked" values missing in the train set. We will replace them by the most frequent value

# most of the passengers come from Southampton
mode_embarked = train['Embarked'].mode()[0]

print(mode_embarked)


# In[19]:


# we fill the missing values with "S" for Southampton
train['Embarked'] = train['Embarked'].fillna(mode_embarked)


# In[20]:


# there is no more "Embarked" missing values
train.isnull().sum()


# In[21]:


# Note: by googling the name of missing embarked person, we would have seen that they embarked at Southampton


# # "Title" feature mapping

# In[22]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in full_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# # "Fare" feature (missing + mapping)

# In[23]:


for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())


# In[24]:


# no missing "Fare" values in the test set
train.isnull().sum()
test.isnull().sum()
# il n'y a plus de valeurs manquantes dans le test set


# In[25]:


# we regroup the "Fare" values in 4 groups
train['FareBand'] = pd.qcut(train['Fare'], 4)
print(train[['FareBand','Survived']].groupby(['FareBand'], as_index=False).mean())

for dataset in full_data:
    dataset.loc[ dataset['Fare'] <= 7.91, ['Fare'] ] = 0    
    dataset.loc[ (dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), ['Fare'] ] = 1
    dataset.loc[ (dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31.0), ['Fare'] ] = 2
    dataset.loc[ (dataset['Fare'] > 31.0), ['Fare'] ] = 3

del train['FareBand']


# # "FamilySize" feature creation

# In[26]:


for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
del train['SibSp']
del train['Parch']    
del test['SibSp']
del test['Parch']


# # Matrices creation

# In[27]:


X_train = train.iloc[:, 2:].values
X_test = test.iloc[:, 1:].values
y_train = train.iloc[:, 1].values


# # Features mapping

# In[28]:


label = LabelEncoder()
X_train[:, 1] = label.fit_transform(X_train[:, 1])
X_test[:, 1] = label.fit_transform(X_test[:, 1])


# In[29]:


label_e = LabelEncoder()
X_train[:, 4] = label_e.fit_transform(X_train[:, 4])
X_test[:, 4] = label_e.fit_transform(X_test[:, 4])


# # Classifier creation + prediction

# In[30]:


classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)

y_prediction = classifier.predict(X_test)


# # Apply k-fold cross validation

# In[31]:


accuracies = cross_val_score(estimator = classifier,X=X_train, y=y_train, cv=10)


# # Results

# In[32]:


# mean() gives a good idea of our model accuracy (bias)
accuracies.mean() # 82,4%


# In[33]:


# std() gives us an idea of the standard deviation (variance)
accuracies.std() # 3.3%


# In[34]:


# submission
results = pd.Series(y_prediction,name="Survived")
submission = pd.concat([pd.Series(range(892,1310),name = "PassengerID"),results],axis = 1)
submission.to_csv("submission.csv", index=False)


# In[35]:


submission

