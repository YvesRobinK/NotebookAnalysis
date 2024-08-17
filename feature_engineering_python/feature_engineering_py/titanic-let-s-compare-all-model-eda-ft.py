#!/usr/bin/env python
# coding: utf-8

# # Titanic EDA --Let's compare diffrerent Models
# This kernel serves as a beginner tutorial and basic guideline for approaching the Exploratory data analysis. I decided to write this kernel because Titanic: Machine Learning from Disaster is one of my favorite competitions on Kaggle. This is a beginner level kernel which focuses on Exploratory Data Analysis,Feature Engineering and Modelling. A lot of people start Kaggle with this competition and they get lost in extremely long tutorial kernels. This is a short kernel compared to the other ones. I hope this will be a good guide for starters and inspire them with new feature engineering ideas.
# 
# Exploratory Data Analysis(EDA): 1)Analysis of the features. 2)Finding any relations or trends considering multiple features.
# 
# Feature Engineering-How to make new features
# 
# Modelling-I have used Logistic Regression,Support Vector Machine,K-Nearest Neighbour,Naive Bayes,Decison Tree,Random Forest
# 
# Inthis kernel let's see how all the model behave on same dataset.
# 
# Thanks a lot for having a look at this notebook. If you found this notebook useful, Do Upvote.

# In[1]:


import pandas as pd
import numpy as np
import pandas_profiling
import seaborn as sns


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')


# ## 2. Exploratory data analysis
# Printing first 5 rows of the train dataset.

# In[2]:


train.head(5)


# ### Data Dictionary
# 
# Variable Notes
# pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# 
# sibsp: The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# parch: The dataset defines family relations in this way...
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.

# In[3]:


train.shape


# **Total rows and columns**
# 
# We can see that there are 891 rows and 12 columns in our training dataset.

# In[4]:


test.head()


# In[5]:


train.shape


# In[6]:


test.shape


# In[7]:


train.info()


# In[8]:


test.info()


# We can see that *Age* value is missing for many rows. 
# 
# Out of 891 rows, the *Age* value is present only in 714 rows.
# 
# Similarly, *Cabin* values are also missing in many rows. Only 204 out of 891 rows have *Cabin* values.

# In[9]:


sns.heatmap(train.isnull())


# In[10]:


train.isnull().sum()


# Roughly 20 percent of the Age data is missing. The proportion of Age missing is likely small enough for reasonable replacement with some form of imputation. Looking at the Cabin column, it looks like we are just missing too much of that data to do something useful with at a basic level. We'll probably drop this later, or change it to another feature like "Cabin Known: 1 or 0"

# In[11]:


sns.heatmap(test.isnull())


# In[12]:


test.isnull().sum()


# There are 86 rows with missing *Age*, 327 rows with missing *Cabin* and 2 rows with missing *Embarked* information.

# # Understanding data using single line

# Whenever you want to get started on a problem like regression and classification using machine learning just use ProfileReport it will give us initial insights that will be very useful to understand the data provided.Let's see what we have got..

# In[13]:


pandas_profiling.ProfileReport(train)


# ### import python lib for visualization

# In[14]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set() # setting seaborn default for plots


# ### Bar Chart for Categorical Features
# - Pclass
# - Sex
# - SibSp ( # of siblings and spouse)
# - Parch ( # of parents and children)
# - Embarked
# - Cabin

# In[15]:


def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[16]:


bar_chart('Sex')


# The Chart confirms **Women** more likely survivied than **Men**

# In[17]:


bar_chart('Pclass')


# The Chart confirms **1st class** more likely survivied than **other classes**  
# The Chart confirms **3rd class** more likely dead than **other classes**

# In[18]:


bar_chart('SibSp')


# The Chart confirms **a person aboarded with more than 2 siblings or spouse** more likely survived  
# The Chart confirms ** a person aboarded without siblings or spouse** more likely dead

# In[19]:


bar_chart('Parch')


# The Chart confirms **a person aboarded with more than 2 parents or children** more likely survived  
# The Chart confirms ** a person aboarded alone** more likely dead

# In[20]:


bar_chart('Embarked')


# The Chart confirms **a person aboarded from C** slightly more likely survived  
# The Chart confirms **a person aboarded from Q** more likely dead  
# The Chart confirms **a person aboarded from S** more likely dead

# ## 3. Feature engineering
# 
# Feature engineering is the process of using domain knowledge of the data  
# to create features (**feature vectors**) that make machine learning algorithms work.  
# 
# feature vector is an n-dimensional vector of numerical features that represent some object.  
# Many algorithms in machine learning require a numerical representation of objects,  
# since such representations facilitate processing and statistical analysis.

# In[21]:


train.head()


# ## 4. Feature Engineering
# 
# The features you use influence more than everything else the result. No algorithm alone, to my knowledge, can supplement the information gain given by correct feature engineering.
# 
# — Luca Massaron
# 
# ### 4.1 Name

# In[22]:


train_test_data = [train, test] # combining train and test dataset

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[23]:


train['Title'].value_counts()


# In[24]:


test['Title'].value_counts()


# #### Title map
# Mr : 0  
# Miss : 1  
# Mrs: 2  
# Others: 3
# 

# In[25]:


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[26]:


train.head()


# In[27]:


test.head()


# In[28]:


bar_chart('Title')


# In[29]:


# delete unnecessary feature from dataset
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)


# In[30]:


train.head()


# In[31]:


test.head()


# ### 4.2 Sex
# 
# male: 0
# female: 1

# In[32]:


sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# In[33]:


bar_chart('Sex')


# ## 4.3 Filling Missing Values
# ### 4.3.1 Embarked

# In[34]:


for dataset in train_test_data:
    dataset.Embarked[ dataset.Embarked.isnull() ] = dataset.Embarked.dropna().mode().values


# ### 4.3.2 Fare

# In[35]:


# fill missing Fare with median fare for each Pclass
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
train.head()


# ### 4.3.3 Cabin

# In[36]:


for dataset in train_test_data:
    print(dataset.Cabin.value_counts())


# In[37]:


for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]


# In[38]:


Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[39]:


cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)


# In[40]:


# fill missing Fare with median fare for each Pclass
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


# In[41]:


train.head()


# ## Drop Ticket Column

# In[42]:


for dataset in train_test_data:
    dataset.drop(['Ticket'], axis = 1, inplace = True)


# In[43]:


train.head()


# In[44]:


test.head()


# ### 4.3.4 Age

# In[45]:


from sklearn.ensemble import RandomForestRegressor

### Populate missing ages using RandomForestClassifier
def setMissingAges(df):
    # Grab all the features that can be included in a Random Forest Regressor
    age_df = df[['Age', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Title']]
    # Split into sets with known and unknown Age values
    knownAge = age_df.loc[ (df.Age.notnull()) ]
    unknownAge = age_df.loc[ (df.Age.isnull()) ]

    # All age values are stored in a target array
    y = knownAge.values[:, 0]

    # All the other values are stored in the feature array
    X = knownAge.values[:, 1::]

    # Create and fit a model
    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rtr.fit(X, y)

    # Use the fitted model to predict the missing values
    predictedAges = rtr.predict(unknownAge.values[:, 1::])

    # Assign those predictions to the full data set
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges

    return df


# In[46]:


for dataset in train_test_data:
    setMissingAges(dataset)


# In[47]:


train.isnull().sum()


# In[48]:


test.isnull().sum()


# ## 5. Derived Variable
# ### 5.1 FamilySize

# In[49]:


train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1


# In[50]:


family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)


# ## 6.Scaling Age and Fare

# In[51]:


from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder


# In[52]:


train


# In[53]:


train['Fare'] = pd.qcut(train['Fare'], 6)
test['Fare'] = pd.qcut(test['Fare'], 6)
    
train['Fare'] = LabelEncoder().fit_transform(train['Fare'])
test['Fare'] = LabelEncoder().fit_transform(test['Fare'])


# In[54]:


train['Age'] = pd.cut(train.Age, bins=[0,16,26,62, np.inf])
test['Age'] = pd.cut(test.Age, bins=[0,17, 30,100, np.inf])
    
train['Age'] = LabelEncoder().fit_transform(train['Age'])
test['Age'] = LabelEncoder().fit_transform(test['Age'])


# #### feature vector map:  
# child: 0  
# young: 1  
# adult: 2  
# mid-age: 3  
# senior: 4

# In[55]:


test.head()


# ## 7. Category Embarked and Pclass Title Dummy Variables
# ### 7.1 Pclass

# In[56]:


dummy_df = pd.get_dummies(train.Pclass, drop_first = True)
dummy_df = dummy_df.rename(columns=lambda x: 'Pclass_' + str(x))
train = pd.concat([train, dummy_df], axis=1)


# In[57]:


dummy_df = pd.get_dummies(test.Pclass, drop_first = True)
dummy_df = dummy_df.rename(columns=lambda x: 'Pclass_' + str(x))
test = pd.concat([test, dummy_df], axis=1)


# ### 7.2 Embarked

# In[58]:


dummy_df = pd.get_dummies(train.Embarked, drop_first = True)
dummy_df = dummy_df.rename(columns=lambda x: 'Embarked_' + str(x))
train = pd.concat([train, dummy_df], axis=1)


# In[59]:


dummy_df = pd.get_dummies(test.Embarked, drop_first = True)
dummy_df = dummy_df.rename(columns=lambda x: 'Embarked_' + str(x))
test = pd.concat([test, dummy_df], axis=1)


# ### 7.3 Title

# In[60]:


dummy_df = pd.get_dummies(train.Title, drop_first = True)
dummy_df = dummy_df.rename(columns=lambda x: 'Title_' + str(x))
train = pd.concat([train, dummy_df], axis=1)


# In[61]:


dummy_df = pd.get_dummies(test.Title, drop_first = True)
dummy_df = dummy_df.rename(columns=lambda x: 'Title_' + str(x))
test = pd.concat([test, dummy_df], axis=1)


# In[62]:


train.head()


# In[63]:


test.head()


# ### Removing Extra features

# In[64]:


features_drop = ['SibSp', 'Parch', 'Pclass', 'Embarked', 'Title']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)


# In[65]:


train_data = train.drop('Survived', axis=1)
target = train['Survived']

train_data.shape, target.shape


# In[66]:


test


# In[67]:


train_data.head(10)


# In[68]:


test_data = test.drop('PassengerId', axis=1)
passenger = test['PassengerId']


# In[69]:


test_data


# In[70]:


passenger


# # Modelling

# The outputs of prediction and feature engineering are a set of label times, historical examples of what we want to predict, and features, predictor variables used to train a model to predict the label. The process of modeling means training a machine learning algorithm to predict the labels from the features, tuning it for the challenge need, and validating it on holdout data.
# 

# In[71]:


# Importing Classifier Modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np


# ### Cross Validation (K-fold)

# In[72]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# ### KNN

# In[73]:


clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[74]:


# kNN Score
round(np.mean(score)*100, 2)


# In[75]:


clf.fit(train_data, target)


# In[76]:


y_preds = clf.predict(test_data)
submission = pd.DataFrame({'PassengerId':passenger, 
              'Survived':y_preds})
submission.to_csv('submission.csv', index=False)
pd.read_csv('submission.csv')


# ### Decision Tree

# In[77]:


clf1 = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf1, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[78]:


# decision tree Score
round(np.mean(score)*100, 2)


# In[79]:


clf1.fit(train_data, target)
y_preds1 = clf1.predict(test_data)
submission1 = pd.DataFrame({'PassengerId':passenger, 
              'Survived':y_preds1})
submission1.to_csv('submission1.csv', index=False)
pd.read_csv('submission1.csv')


# ### Ramdom Forest

# In[80]:


clf2 = RandomForestClassifier(n_estimators=12)
scoring = 'accuracy'
score = cross_val_score(clf2, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[81]:


# Random Forest Score
round(np.mean(score)*100, 2)


# In[82]:


clf2.fit(train_data, target)
y_preds2 = clf2.predict(test_data)
submission2 = pd.DataFrame({'PassengerId':passenger, 
              'Survived':y_preds2})
submission2.to_csv('submission2.csv', index=False)
pd.read_csv('submission2.csv')


# ### Naive Bayes

# In[83]:


clf3 = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf3, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[84]:


# Naive Bayes Score
round(np.mean(score)*100, 2)


# In[85]:


clf3.fit(train_data, target)
y_preds3 = clf3.predict(test_data)
submission3 = pd.DataFrame({'PassengerId':passenger, 
              'Survived':y_preds3})
submission3.to_csv('submission3.csv', index=False)
pd.read_csv('submission3.csv')


# ### SVM

# In[86]:


clf4 = SVC()
scoring = 'accuracy'
score = cross_val_score(clf4, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[87]:


round(np.mean(score)*100,2)


# In[88]:


clf4.fit(train_data, target)
y_preds4 = clf4.predict(test_data)
submission4 = pd.DataFrame({'PassengerId':passenger, 
              'Survived':y_preds4})
submission4.to_csv('submission4.csv', index=False)
pd.read_csv('submission4.csv')


# # Logistic Regression

# In[89]:


from sklearn.linear_model import LogisticRegression
clf5 = LogisticRegression(solver = 'newton-cg', random_state = 0)
scoring = 'accuracy'
score = cross_val_score(clf5, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[90]:


round(np.mean(score)*100,2)


# In[91]:


clf5.fit(train_data,target)
y_preds5 = clf5.predict(test_data)
submission5 = pd.DataFrame({'PassengerId':passenger, 
              'Survived':y_preds5})
submission5.to_csv('submission5.csv', index=False)
pd.read_csv('submission5.csv')


# # NN with 3 Layers

# In[92]:


from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(12, input_dim=12, activation='relu'))
model.add(Dense(12, input_dim=12, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[93]:


#compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mean_absolute_error'])


# In[94]:


model.summary()


# In[95]:


#fit the model
model.fit(train_data, target, epochs=500, batch_size=50)


# In[96]:


scores = model.evaluate(train_data,target)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# # OBSERVATIONS
# Age,Cabin have null values that must be treated.
# 
# Out of 891 passengers in training set, only around 350 survived i.e Only 38.4% of the total training set survived the crash.
# 
# Women more likely survivied than Men
# 
# The analysis for outliers show that Fare, Embarked and Parch column have some outliers. Fare and Survive has best correlation.Variable are not very much correlated so we can use them all.
# 
# The chances for survival for Port C is highest around 0.55 while it is lowest for S.
# 
# It is an important feature as it reveals that passengers with family size 2 - 4 had a better survival rate than passengers travelling alone or who had larger families.
# 
# Person aboarded from C slightly more likely survived,Q more likely dead,S more likely dead.
# 
# 1st class more likely survivied than other classes 3rd class more likely dead than other classes

# # Model Analysis
# I just created a deep learning CNN model just for knowledge that how DL model performs on this small dataset
# 
# SVM is giving us the max Score.
# 
# KNN and Linear Regression score is above 81 in train data
# 
# Decison Tree And Random forest also worked good
# 
# NB doesn't perform well
# 

# # Thanks a lot for having a look at this notebook. If you found this notebook useful, Do Upvote.
# If you have forked the kernel and not upvoted yet, then show the support by upvoting :)
# 
# Please leave you constructive criticism and suggestion in comments below!!
