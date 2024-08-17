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


# **Voting Classifier using Sklearn**
# 
# 
#   *  A Voting Classifier is a machine learning model that trains on an ensemble of numerous models
#     and predicts an output (class) based on their highest probability of chosen class as the output.*
#  It simply aggregates the findings of each classifier passed into Voting Classifier and predicts the output class based on the highest majority of voting. The idea is instead of creating separate dedicated models and finding the accuracy for each them, we create a single model which trains by these models and predicts output based on their combined majority of voting for each output class.
#  
# **Voting Classifier supports two types of votings
# 1.	Hard Voting: In hard voting, the predicted output class is a class with the highest majority of votes i.e the class which had the highest probability of being predicted by each of the classifiers. Suppose three classifiers predicted the output class(A, A, B), so here the majority predicted A as output. Hence A will be the final prediction.
# 2.	Soft Voting: In soft voting, the output class is the prediction based on the average of probability given to that class. Suppose given some input to three models, the prediction probability for class A = (0.30, 0.47, 0.53) and B = (0.20, 0.32, 0.40). So the average for class A is 0.4333 and B is 0.3067, the winner is clearly class A because it had the highest probability averaged by each classifier.
# 
# 
# **Note: Make sure to include a variety of models to feed a Voting Classifier to be sure that the error made by one might be resolved by the other.**
# 

# 

# **# Import Libraries**

# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier 
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# **Read Data**

# In[3]:


data=pd.read_csv('../input/titanic/train.csv')


# **Apply some feature engineering**

# In[4]:


data['Embarked']=data['Embarked'].fillna(0)
data['Parch']=data['Parch'].fillna(0)
data['SibSp']=data['SibSp'].fillna(0)

data['Age']=data['Age'].fillna(data['Age'].mean())
data['Sex']=data['Sex'].fillna(0)
data['Pclass']=data['Pclass'].fillna(0)
data['Embarked']=data['Embarked'].fillna('0')


lblenc=LabelEncoder()
lblenc.fit(data['Sex'])
data['Sex']=lblenc.transform(data['Sex'])


data['Embarked']=data['Embarked'].replace('S',1)
data['Embarked']=data['Embarked'].replace('C',2)
data['Embarked']=data['Embarked'].replace('Q',3)


# **#Splitting data**

# In[5]:


data=data[['Pclass','Sex','Age','SibSp','Parch','Survived','Embarked']]
X=data[['Pclass','Sex','Age','SibSp','Parch','Embarked']]
y=data['Survived']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)


# **#loading models for Voting Classifier**

# In[6]:


DTModel_=DecisionTreeClassifier(criterion = 'entropy',max_depth=3,random_state = 33)
LDAModel_=LinearDiscriminantAnalysis(n_components=1 ,solver='svd')
SGDModel_=SGDClassifier(loss='log', penalty='l2', max_iter=10000, tol=1e-5)
GBCModule_=GradientBoostingClassifier(n_estimators=1000,max_depth=5,random_state=33)#
MLPModule_=MLPClassifier()
RFCModule_=RandomForestClassifier(n_estimators=100,max_depth=2, random_state=33)
SVCModel_=SVC(kernel= 'rbf', max_iter=1000,C=10,gamma='auto')
LGnModel_=LogisticRegression(penalty='l1',solver='liblinear',C=1.0,random_state=50)


# **#loading Voting Classifier**

# In[7]:


VotingClassifierModel = VotingClassifier(estimators=[('DTModel',DTModel_),('LDAModel',LDAModel_),('SGDModel',SGDModel_),                                                     
('GBCModule',GBCModule_),('MLPModule',MLPModule_),('RFCModule',RFCModule_),('SVCModel',SVCModel_),('LGnModel',LGnModel_)],
voting='hard')
VotingClassifierModel.fit(X_test, y_test)


# 

# **#Calculating Details**

# In[8]:


print('VotingClassifierModel Train Score is : ' , VotingClassifierModel.score(X_train, y_train))
print('VotingClassifierModel Test Score is : ' , VotingClassifierModel.score(X_test, y_test))


# **#Calculating Prediction**

# In[9]:


y_pred = VotingClassifierModel.predict(X_test)
print('Predicted Value for VotingClassifierModel is : ' , y_pred)


# **#Calculating Confusion Matrix**

# In[10]:


CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', CM)


# **# drawing confusion matrix**

# In[11]:


sns.heatmap(CM, center = True)
plt.show()


# 
