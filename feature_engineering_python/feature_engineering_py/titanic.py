#!/usr/bin/env python
# coding: utf-8

# In[1]:


####....#.......#.....####.....#.......#...
  #........##.....##....#.........##.....##...
#.......#.#...#.#.....#........#.#...#.#...
 #......#..#.#..#......#.......#..#.#..#...
  #.....#...#...#.......#......#...#...#...
   #....#.......#........#.....#.......#...
  ####.....#.......#....####......#.......#...
 

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from imblearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
 for filename in filenames:
     print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Reading the data set into pandas Dataframe

# In[2]:


train_data=pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head(5)


# # Size of Data set

# In[3]:


# look at the number of rows and columns
print(f'Rows: {train_data.shape[0]}, Columns: {train_data.shape[1]}')


# # Overview

# In[4]:


# look at the columns
print(train_data.columns)


# In[5]:


train_data.info()


# In[6]:


train_data.describe()


# # Missing Data Handling 
# ## 1. Delete Row/Column
# 
# 
# As Cabin column has more than 70% missing data, it can not be used into the Model. We have to remove this column from feature set.
# 
# 
# ## 2. Replace with Mean/Median/Mode
# 
# Embarked has only 3 missing values. As this is a categorical variable, we can replace missing data with mode value ie we are going to fill missing values with most common embarkment value
# 

# In[7]:


# look at the columns have null values
train_data.isna().sum()


# In[8]:


#count the number of occurrences of each unique value in the 'Embarked' column 
embarked_plot = train_data['Embarked'].value_counts().plot(kind = 'bar')
plt.title('Embarked Distribution')
plt.xlabel('Embarked')
plt.ylabel('Frequency')


# In[9]:


# fill the null values by mode
train_data['Embarked'] = train_data['Embarked'].fillna('S')


# 
# 
# ### 3. Predict Missing Value
# 
# We can predict the null values using other features which do not have missing values. For eg we can predict missing Age values with help of machine learning algorithm where features could be Pclass, Sex, Fare etc.
# 
# Before predicting the missing value lets explore the data to check what columns will help to predict Age and relationship between features.
# 

# # Data Preparation
# Lets deep dive into the data through Exploratory Data Analysis

# # Data Visualization
# 

# In[10]:


#Survival Distribution
train_data['Survived'].value_counts(normalize=True).plot(kind='bar')
plt.title('Survival Distribution')
plt.xlabel('Survived')
plt.ylabel('Normalized Count')


# In[11]:


# Ticket Class Distribution
train_data['Pclass'].value_counts(normalize=True).plot(kind='bar')
plt.title('Ticket Class Distribution')
plt.xlabel('Pclass')
plt.ylabel('Normalized Count')


# In[12]:


# Gender Distribution
train_data['Sex'].value_counts(normalize=True).plot(kind='bar')
plt.title('Gender Distribution')
plt.xlabel('Sex')
plt.ylabel('Normalized Count')


# In[13]:


# number of Siblings/Spouses Distribution
train_data['SibSp'].value_counts(normalize=True).plot(kind='bar')
plt.title('# of Siblings/Spouses Distribution')
plt.xlabel('Sibling/Spouse Count')
plt.ylabel('Normalized Count')


# In[14]:


# number of parent child count
train_data['Parch'].value_counts(normalize=True).plot(kind='bar')
plt.title('# of Parents/Children Distribution')
plt.xlabel('Parent Child Count')
plt.ylabel('Normalized Count')


# In[15]:


# age distribution
train_data['Age'].plot(kind='hist')
plt.title('Age Distribution')
plt.xlabel('Age Range')
plt.ylabel('Frequency')


# In[16]:


# fare distribution
train_data['Fare'].plot(kind='hist')
plt.title('Fare Distribution')
plt.xlabel('Fare Range')
plt.ylabel('Frequency')


# In[17]:


# fill a missing values in the "Age" column using the median of age has a same sex and pclass
train_data["Age"] = train_data.groupby(['Sex','Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))


# In[18]:


# Survival against Gender
train_data['Sex'][train_data['Survived']==1].value_counts(normalize=True, ascending=True).plot(kind='bar')
plt.title('Survival against Gender')
plt.xlabel('Sex')
plt.ylabel('Survived Frequency (Normalized)')


# In[19]:


# Ticket Class vs Survival
train_data.groupby(['Survived', 'Pclass']).size().unstack().plot(kind='bar', stacked=True)
plt.title('Ticket Class vs Survival')


# In[20]:


# relationship between survival rate and a different variable related to family relationships 

fig = plt.figure(figsize=(12,5))
plt.subplots_adjust(hspace=.5)

ax = plt.subplot2grid((1,2), (0,0))
train_data.groupby(['Parch', 'Survived']).size().unstack().plot(kind='bar', stacked=True, ax=ax)
plt.title('Parent/Child vs Survived')
plt.ylabel('Count')

ax = plt.subplot2grid((1,2), (0,1))
train_data.groupby(['SibSp', 'Survived']).size().unstack().plot(kind='bar', stacked=True, ax=ax)
plt.title('Sibling/Spouse vs Survived')
plt.ylabel('Count')


# 
# # Feature Engineering
# 
# Before we train our model with this data, lets do some Feature Engineering.
# 
# Feature Engineering is a process of feature extraction from raw dataset using domain knowledge.
# 
# In previous graph we have seen, both Sibsp and Parch are having similar influence on Survival. Lets calculate if passenger travelled alone or with family. Instead of 2 separate features, lets combine SibSp and Parch and create a new feature named FamilySize.
# 

# In[21]:


# create a new feature : family size
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1


# In[22]:


train_data.groupby(['FamilySize', 'Survived']).size().unstack().plot(kind='bar', stacked=True)
plt.title('Family vs Survival')


# In[23]:


train_data.info()


# In[24]:


test = pd.read_csv("/kaggle/input/titanic/test.csv")
test.info()


# In[25]:


test["Age"] = test.groupby(['Sex','Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
test["Fare"] = test["Fare"].fillna(test["Fare"].dropna().median())   
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1


# In[26]:


test.info()


# # Feature selection and encoding categorical variables

# In[27]:


# Feature selection
X_train = train_data[['Pclass','Sex', 'Age', 'Fare', 'Embarked', 'FamilySize']]
X_test = test[['Pclass','Sex', 'Age', 'Fare', 'Embarked', 'FamilySize']]
# one-hot encoding
labels_to_encode = ['Pclass', 'Sex', 'Embarked']
for label in labels_to_encode:
    X_train = X_train.join(pd.get_dummies(X_train[label], prefix = label))
    X_train.drop(label, axis=1, inplace=True)
for label in labels_to_encode:
    X_test = X_test.join(pd.get_dummies(X_test[label], prefix = label))
    X_test.drop(label, axis=1, inplace=True)
y = train_data['Survived'].values


# In[28]:


X_train.head(1)


# 
# # Train Model
# 
# 
# 

# In[29]:


from sklearn.linear_model import LogisticRegression            # for Machine Learning algorithms (Logistic Regression)
from sklearn.model_selection import cross_val_score            # for Cross validation score
from sklearn.ensemble import RandomForestClassifier            # for Machine Learning algorithms (Random Forest).
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn import tree


# In[30]:


#logistic regression
lr_model =  LogisticRegression(solver='liblinear')
scores = cross_val_score(lr_model, X_train, y, cv=5)
print(round(np.mean(scores*100)))
lr_model.fit(X_train,y)
predictions_lr = lr_model.predict(X_test)
submission_lr = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions_lr
})
submission_lr.to_csv('titanic_lr.csv', index=False)



# In[31]:


# Random forset
rfclr = RandomForestClassifier(n_estimators=100, max_depth = 7)
scores = cross_val_score(rfclr, X_train, y, cv=5)
print(round(np.mean(scores*100)))
rfclr.fit(X_train, y)
predictions_rf = rfclr.predict(X_test)
submission_rf = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions_rf
})
submission_rf.to_csv('titanic_rd.csv', index=False)


# In[32]:


# Gaussian
GNBclf = GaussianNB()
GNBmodel = GNBclf.fit(X_train, y)
# Model Prediction
GNB_pred = GNBclf.predict(X_test)
submission_rf = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": GNB_pred
})
submission_rf.to_csv('titanic_gn.csv', index=False)
# score = 0.75


# In[33]:


#svm 
SVM_clf = svm.SVC()
SVM_model = SVM_clf.fit(X_train, y)
# Model Prediction
SVM_pred = SVM_clf.predict(X_test)
submission_rf = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": SVM_pred
})
submission_rf.to_csv('titanic_svm.csv', index=False)
# score = 0.66


# In[34]:


#dicision tree
DT_clf = tree.DecisionTreeClassifier()
DT_model = DT_clf.fit(X_train, y)
# Model Prediction
DT_pred = DT_clf.predict(X_test)
submission_rf = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": DT_pred
})
submission_rf.to_csv('titanic_dt.csv', index=False)
# score = 0.73

