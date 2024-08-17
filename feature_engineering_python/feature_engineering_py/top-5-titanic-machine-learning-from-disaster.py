#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# Titanic competition is a very good way to introduce feature engineering and classification models. I'm gonna explore the data and make something with them and also imput missing values. Feature engineering is an important part of machine learning process so I want to spend more time for this part. I'm gonna try I few models and tell you which work the best with train dataset from this competition. Please consider upvoting if this is useful to you :)

# **Import the Libraries**

# In[2]:


import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import math 
import xgboost as xgb
np.random.seed(2019)
from scipy.stats import skew
from scipy import stats

import statsmodels
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
print("done")


# **Import Data**

# I'm adding here 'train' variable in order to check in the easiest way later which observations are from train and test dataset because I'm gonna join train and test datasets.

# In[3]:


def read_and_concat_dataset(training_path, test_path):
    train = pd.read_csv(training_path)
    train['train'] = 1
    test = pd.read_csv(test_path)
    test['train'] = 0
    data = train.append(test, ignore_index=True)
    return train, test, data

train, test, data = read_and_concat_dataset('../input/train.csv', '../input/test.csv')
data = data.set_index('PassengerId')


# ##**Explore the Data**

# In[4]:


data.head(5)


# In[5]:


data.describe()


# **PassengerId** - the unique id of the row, it doesn't have any effect on Survived value.
# 
# **Survived** - binary:
# * 1 -> Survived
# * 0 -> Not survived
# 
# **Pclass** (Passenger Class) - economic status of the passenger, this variable has 3 values;
# * 1 -> Upper Class
# * 2 -> Middle Class
# * 3 -> Lower Class
# 
# **Name**, **Sex** and **Age** - are self-explanatory.
# 
# **SibSp** - the total number of the passengers' siblings and spouse.
# 
# **Parch** - the total number of the passengers' parents and children.
# 
# **Ticket** - the ticket number.
# 
# **Fare** - the passenger fare.
# 
# **Cabin** - the cabin number.
# 
# **Embarked** is port of embarkation, 3 values:
# * C -> Cherbourg
# * Q -> Queenstown
# * S -> Southampton

# Correlation matrix between numerical values:

# In[6]:


g = sns.heatmap(data[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, cmap = "coolwarm")


# Correlations between numerical variables and Survived aren't so high but it doesn't mean that the other features are not useful.

# In[7]:


def comparing(data,variable1, variable2):
    print(data[[variable1, variable2]][data[variable2].isnull()==False].groupby([variable1], as_index=False).mean().sort_values(by=variable2, ascending=False))
    g = sns.FacetGrid(data, col=variable2).map(sns.distplot, variable1)


# In[8]:


def counting_values(data, variable1, variable2):
    return data[[variable1, variable2]][data[variable2].isnull()==False].groupby([variable1], as_index=False).mean().sort_values(by=variable2, ascending=False)


# Parch vs Survived

# In[9]:


comparing(data, 'Parch','Survived')


# SibSp vs Survived

# In[10]:


comparing(data, 'SibSp','Survived')


# Fare vs Survived

# In[11]:


comparing(data, 'Fare','Survived')


# Age vs Survived

# In[12]:


comparing(data, 'Age','Survived')


# Sex vs Survived

# In[13]:


counting_values(data, 'Sex','Survived')


# In[14]:


data['Women'] = np.where(data.Sex=='female',1,0)
comparing(data, 'Women','Survived')


# Pclass vs Survived

# In[15]:


comparing(data, 'Pclass','Survived')


# In[16]:


grid = sns.FacetGrid(data, col='Survived', row='Pclass', size=2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)


# Embarked vs Survived

# In[17]:


grid = sns.FacetGrid(data, row='Embarked', col='Survived', size=2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)


# ##**Missing values**

# In[18]:


data.isnull().sum()


# There are 263 missing ages, 1014 missing cabins. Age is very important variable, so it's worth spending time to imput them. If it comes to imputing cabins - it's too hard to do because dataset has only 1309 observations so 77% cabins are missing.

# Missing values in Embarked and Fare variables are very easy to imput because we can use the most popular value or something like that.
# 
# I'm gonna replace missing value in Fare with 0 and in Embarked with the most popular value ('S').

# In[19]:


data.groupby('Pclass').Fare.mean()


# In[20]:


data.Fare = data.Fare.fillna(0)


# In[21]:


print(data.Embarked.value_counts())
data.Embarked = data.Embarked.fillna('S')


# If it comes to Cabin variable, I'm gonna fill up NaN values with 'Unknown' and get first letter from every Cabin in dataset.

# In[22]:


data.Cabin = data.Cabin.fillna('Unknown_Cabin')
data['Cabin'] = data['Cabin'].str[0]


# Let's check the distribution of the cabins in individual passenger classes.

# In[23]:


data.groupby('Pclass').Cabin.value_counts()


# The Cabin 'Unknown' will be set to C for the first class, D for the second class and G for the third class. One observation with Cabin 'T' and first class I'll fix with C.

# In[24]:


data['Cabin'] = np.where((data.Pclass==1) & (data.Cabin=='U'),'C',
                                            np.where((data.Pclass==2) & (data.Cabin=='U'),'D',
                                                                        np.where((data.Pclass==3) & (data.Cabin=='U'),'G',
                                                                                                    np.where(data.Cabin=='T','C',data.Cabin))))


# Now I'm gonna get title from each Name in dataset. This variable will be very useful and it can help to imput missing value in Age. People's titles can represent their age, earnings and life status and all these three properties can be associated with the possibility of survival on a ship.

# In[25]:


data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(data['Title'], data['Sex'])
data = data.drop('Name',axis=1)


# I need to replace a few titles with 'other' values because these titles are not as popular and have a low frequency of occurrence in this dataset.

# In[26]:


#let's replace a few titles -> "other" and fix a few titles
data['Title'] = np.where((data.Title=='Capt') | (data.Title=='Countess') | (data.Title=='Don') | (data.Title=='Dona')
                        | (data.Title=='Jonkheer') | (data.Title=='Lady') | (data.Title=='Sir') | (data.Title=='Major') | (data.Title=='Rev') | (data.Title=='Col'),'Other',data.Title)

data['Title'] = data['Title'].replace('Ms','Miss')
data['Title'] = data['Title'].replace('Mlle','Miss')
data['Title'] = data['Title'].replace('Mme','Mrs')


# Let's check how the distribution of survival variable  depending on the title.

# In[27]:


data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
facet = sns.FacetGrid(data = data, hue = "Title", legend_out=True, size = 4.5)
facet = facet.map(sns.kdeplot, "Age")
facet.add_legend();


# People with 'Master' have the highest survival rate. Maybe because people with the master are mainly boys under 13 years old.

# Let's see distributions on box plots.

# In[28]:


sns.boxplot(data = data, x = "Title", y = "Age")


# In[29]:


facet = sns.FacetGrid(data, hue="Survived",aspect=3)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, data['Age'].max()))
facet.add_legend()


# Age has a very large impact on the survival rate, but when this variable has missing values - it is useless. I'm gonna impute the missing values using the average age values in particular groups due to the titles.

# In[30]:


data.groupby('Title').Age.mean()


# In[31]:


data['Age'] = np.where((data.Age.isnull()) & (data.Title=='Master'),5,
                        np.where((data.Age.isnull()) & (data.Title=='Miss'),22,
                                 np.where((data.Age.isnull()) & (data.Title=='Mr'),32,
                                          np.where((data.Age.isnull()) & (data.Title=='Mrs'),37,
                                                  np.where((data.Age.isnull()) & (data.Title=='Other'),45,
                                                           np.where((data.Age.isnull()) & (data.Title=='Dr'),44,data.Age))))))                   


# A few new variables:

# ##**Feature engineering**

# * FamilySize - number of family members, people travelling alone will have a value of 1
# * Women - it depends on Sex variable but I'm making it in binary way
# * Mother - women with Mrs title and at least 1 parch, women, children and mothers probably have a survival factor
# * Free - people who don't need to pay fare, these people could win tickets or something like that, they can have a similar survival rate
# * TypeOfTicket - prefixes of ticket, tickets with same prefixes may have a similar class and survival.
# 
# If it comes to TypeOfTicket variable I'm gonna also replace a few values of this variable with 'other' values, relying on the same as in the case of titles.

# In[32]:


data['FamilySize'] = data.SibSp + data.Parch + 1
data['Mother'] = np.where((data.Title=='Mrs') & (data.Parch >0),1,0)
data['Free'] = np.where(data['Fare']==0, 1,0)
data = data.drop(['SibSp','Parch','Sex'],axis=1)


# In[33]:


import string
TypeOfTicket = []
for i in range(len(data.Ticket)):
    ticket = data.Ticket.iloc[i]
    for c in string.punctuation:
                ticket = ticket.replace(c,"")
                splited_ticket = ticket.split(" ")   
    if len(splited_ticket) == 1:
                TypeOfTicket.append('NO')
    else: 
                TypeOfTicket.append(splited_ticket[0])
            
data['TypeOfTicket'] = TypeOfTicket

data.TypeOfTicket.value_counts()
data['TypeOfTicket'] = np.where((data.TypeOfTicket!='NO') & (data.TypeOfTicket!='PC') & (data.TypeOfTicket!='CA') & 
                                (data.TypeOfTicket!='A5') & (data.TypeOfTicket!='SOTONOQ'),'other',data.TypeOfTicket)
data = data.drop('Ticket',axis=1)


# FamilySize vs Survived

# In[34]:


comparing(data, 'FamilySize','Survived')


# Title vs Survived

# In[35]:


counting_values(data, 'Title','Survived')


# TypeOfTicket vs Survived

# In[36]:


counting_values(data, 'TypeOfTicket','Survived')


# Cabin vs Survived

# In[37]:


counting_values(data, 'Cabin','Survived')


# Mother vs Survived

# In[38]:


comparing(data, 'Mother','Survived')


# Free vs Survived

# In[39]:


comparing(data, 'Free','Survived')


# I'm cutting Age variable to 5 equal intervals.

# In[40]:


bins = [0,12,24,45,60,data.Age.max()]
labels = ['Child', 'Young Adult', 'Adult','Older Adult','Senior']
data["Age"] = pd.cut(data["Age"], bins, labels = labels)


# I create dummy variables for all variables with categories using the function get_dummies from pandas.

# In[41]:


data = pd.get_dummies(data)


# ##**Modeling**
# * Decision Tree Classifier
# * Random Forest Classifier
# * KNeighbors Classifier
# * SVM
# * Logistic Regression
# * XGB Classifier

# To check how good each model is I'm gonna split dataset to train (70%) and test (30%) dataset (excluding missing values in Survived variable) and use Accuracy Score from sklearn.metrics. I set random_state to 2019 in order to compare the results between the models.

# In[42]:


from sklearn.model_selection import train_test_split
trainX, testX, trainY, testY = train_test_split(data[data.Survived.isnull()==False].drop('Survived',axis=1),data.Survived[data.Survived.isnull()==False],test_size=0.30, random_state=2019)


# I'm gonna to put result of each model in Data Frame 'Results'

# In[43]:


Results = pd.DataFrame({'Model': [],'Accuracy Score': [], 'Recall':[], 'F1score':[]})


# In[44]:


from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


# **Decision Tree Classifier**

# In[45]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=4)
model.fit(trainX, trainY)
y_pred = model.predict(testX)
res = pd.DataFrame({"Model":['DecisionTreeClassifier'],
                    "Accuracy Score": [accuracy_score(y_pred,testY)],
                   "Recall": [recall_score(testY, y_pred)],
                   "F1score": [f1_score(testY, y_pred)]})
Results = Results.append(res)


# In[46]:


pd.crosstab(testY, y_pred, rownames=['Real data'], colnames=['Predicted'])


# **Random Forest Classifier**

# In[47]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=2500, max_depth=4)
model.fit(trainX, trainY)
y_pred = model.predict(testX)
from sklearn.metrics import accuracy_score
res = pd.DataFrame({"Model":['RandomForestClassifier'],
                    "Accuracy Score": [accuracy_score(y_pred,testY)],
                   "Recall": [recall_score(testY, y_pred)],
                   "F1score": [f1_score(testY, y_pred)]})
Results = Results.append(res)


# In[48]:


pd.crosstab(testY, y_pred, rownames=['Real data'], colnames=['Predicted'])


# **KNeighbors Classifier**

# In[49]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(trainX, trainY)
y_pred = model.predict(testX)
from sklearn.metrics import accuracy_score
res = pd.DataFrame({"Model":['KNeighborsClassifier'],
                    "Accuracy Score": [accuracy_score(y_pred,testY)],
                   "Recall": [recall_score(testY, y_pred)],
                   "F1score": [f1_score(testY, y_pred)]})
Results = Results.append(res)


# In[50]:


pd.crosstab(testY, y_pred, rownames=['Real data'], colnames=['Predicted'])


# **SVM**

# In[51]:


from sklearn.svm import SVC
model = SVC()
model.fit(trainX, trainY)
y_pred = model.predict(testX)
from sklearn.metrics import accuracy_score
res = pd.DataFrame({"Model":['SVC'],
                    "Accuracy Score": [accuracy_score(y_pred,testY)],
                   "Recall": [recall_score(testY, y_pred)],
                   "F1score": [f1_score(testY, y_pred)]})
Results = Results.append(res)


# In[52]:


pd.crosstab(testY, y_pred, rownames=['Real data'], colnames=['Predicted'])


# **Logistic Regression**

# In[53]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(trainX, trainY)
y_pred = model.predict(testX)
from sklearn.metrics import accuracy_score
res = pd.DataFrame({"Model":['LogisticRegression'],
                    "Accuracy Score": [accuracy_score(y_pred,testY)],
                   "Recall": [recall_score(testY, y_pred)],
                   "F1score": [f1_score(testY, y_pred)]})
Results = Results.append(res)


# In[54]:


pd.crosstab(testY, y_pred, rownames=['Real data'], colnames=['Predicted'])


# **XGB Classifier**

# I put here some hyper-parameters tuning with n_estmators, max_depth and learning_rate parameters.

# In[55]:


from xgboost.sklearn import XGBClassifier
model = XGBClassifier(learning_rate=0.001,n_estimators=2500,
                                max_depth=4, min_child_weight=0,
                                gamma=0, subsample=0.7,
                                colsample_bytree=0.7,
                                scale_pos_weight=1, seed=27,
                                reg_alpha=0.00006)
model.fit(trainX, trainY)
y_pred = model.predict(testX)
from sklearn.metrics import accuracy_score
res = pd.DataFrame({"Model":['XGBClassifier'],
                    "Accuracy Score": [accuracy_score(y_pred,testY)],
                   "Recall": [recall_score(testY, y_pred)],
                   "F1score": [f1_score(testY, y_pred)]})
Results = Results.append(res)


# In[56]:


pd.crosstab(testY, y_pred, rownames=['Real data'], colnames=['Predicted'])


# ##**Results**

# In[57]:


Results


# How we see - XGB Classifier gives the best results. This model helps me to get 0.81339 on competition test dataset and it gives me place in 5% best results on Leaderboard.

# In[58]:


from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
trainX = data[data.Survived.isnull()==False].drop(['Survived','train'],axis=1)
trainY = data.Survived[data.Survived.isnull()==False]
testX = data[data.Survived.isnull()==True].drop(['Survived','train'],axis=1)
model = XGBClassifier(learning_rate=0.001,n_estimators=2500,
                                max_depth=4, min_child_weight=0,
                                gamma=0, subsample=0.7,
                                colsample_bytree=0.7,
                                scale_pos_weight=1, seed=27,
                                reg_alpha=0.00006)
model.fit(trainX, trainY)
test = data[data.train==0]
test['Survived'] = model.predict(testX).astype(int)
test = test.reset_index()
test[['PassengerId','Survived']].to_csv("submissionXGB.csv",index=False)
print("done1")

