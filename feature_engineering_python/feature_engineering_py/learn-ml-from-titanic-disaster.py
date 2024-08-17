#!/usr/bin/env python
# coding: utf-8

# # Learn_ML_from_Titanic_Disaster 
# Exploratory Data Analysis (EDA) and Machine Learning to predict the survival of Titanic Passengers
# 
# ![](https://miro.medium.com/max/1024/0*KfHijq1bO1nDV5Dl.jpg)
# 
# 
# 
# 
# 
# 
# 
# You are going to embark on your first Exploratory Data Analysis (EDA) and Machine Learning to predict the survival of Titanic Passengers. This is the genesis challenge for most onboarding data scientists and will set you up for success. I hope this article inspires you. All aboard!!!

# 1. Introduction of ML
# 2. About Titanic Problem
# 3. Project Work Flow
#     - Problem Defintion
#         - Problme Feature
#         - Variables
#         - Objective
#         - Input & Output
#     - Loading Packages or Import Libraries
#     - Gathering Data or Data Collection
#     - Exploratory Data Analysis(EDA)
#         - Data Analysis
#         - Data Pre-processing
#         - Data Wraggling   
#     - Training and Testing the model
#     - Evaluation
#     - Submission

# ## Introduction of ML:
# What is Machine Learning (ML)?
# Machine Learning (ML) is the science of getting machine to act without being explicitly programmed.
# 
# Machine learning is a type of artificial intelligence (AI) that allows software applications to learn from the data and become more accurate in predicting outcomes without human intervention.
# 
# Machine Learning is a subset of artificial intelligence (AI) which focuses mainly on machine learning from their experience and making predictions based on its experience.
# 
# Machine Learning is an application of artificial intelligence (AI) which provides systems the ability to automatically learn and improve from experience without being explicitly programmed.

# ## About Titanic Problem
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

# ## Project WorkFlow

# ### Problem Definition:
# I think one of the important things when you start a new machine learning project is Defining your problem. that means you should understand business problem.( Problem Formalization).

# ### Problem Feature
# The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. That's why the name DieTanic. This is a very unforgetable disaster that no one in the world can forget.
# 
# It took about $7.5 million to build the Titanic and it sunk under the ocean due to collision. The Titanic Dataset is a very good dataset for begineers to start a journey in data science and participate in competitions in Kaggle.

# ### Objective:
# As a data scientist, it's your job to predict if a passenger survived the sinking of the Titanic or not. For each PassengerId in the test set, you must predict a 0 or 1 value for the Survived variable.(binary classification)

# ### Variables:
# 1. Age : Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# 2. Sibsp : The dataset defines family relations in this way...
# 
#     a. Sibling = brother, sister, stepbrother, stepsister
# 
#     b. Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# 3. Parch: The dataset defines family relations in this way...
# 
#     a. Parent = mother, father
# 
#     b. Child = daughter, son, stepdaughter, stepson
# 
#     c. Some children travelled only with a nanny, therefore parch=0 for them.
# 
# 4. Pclass : A proxy for socio-economic status (SES).
#     1st = Upper
#     2nd = Middle
#     3rd = Lower
# 
# 5. Embarked : Nominal datatype
# 
# 6. Name: Nominal datatype . It could be used in feature engineering to derive the gender from title.
# 
# 7. Sex: Nominal datatype
# 
# 8. Ticket: That have no impact on the outcome variable. Thus, they will be excluded from analysis
# 
# 9. Cabin: It'a a nominal datatype that can be used in feature engineering
# 
# 10. Fare: Indicating the fare
# 
# 11. PassengerID: have no impact on the outcome variable. Thus, it will be excluded from analysis
# 
# 12. Survival: dependent variable , 0 or 1
# 

# ## Import all required or necessary libraries

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization
sns.set_style('whitegrid')
import matplotlib.pyplot as plt # data visualization
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Librarires using for Machine Learning Algorithm

# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier


# ## Gathering Data or Data Collection

# ### Know how to import your data?
# 
# Find what you have in your data folder?

# In[3]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### Import or Load all of your data from the data folder

# In[4]:


titanic_train = pd.read_csv("../input/train.csv")
titanic_test = pd.read_csv("../input/test.csv")


# #### Let's see the size or shpae of your data

# In[5]:


print("Train: rows:{} columns:{}".format(titanic_train.shape[0], titanic_train.shape[1]))


# In[6]:


print("Test Data Shape",titanic_test.shape)


# #### Let's preview top 5 and bottom 5 records from training dataset

# In[7]:


titanic_train.head()


# In[8]:


titanic_train.tail()


# #### Let's preview top 5 and bottom 5 records form test dataset

# In[9]:


titanic_test.head()


# In[10]:


titanic_test.tail()


# In[11]:


print("Total Number of passagner on Titanic (from training data):", str(len(titanic_train)))


# ## Exploratory Data Analysis

# ### Data Analysis
# Data pre-processing is one of the most important steps in machine learning. It is the most important step that helps in building machine learning models more accurately. In machine learning, there is an 80/20 rule. Every data scientist should spend 80% time for data pre-processing and 20% time to actually perform the analysis.
# 

# In[12]:


sns.countplot(x="Survived", data=titanic_train)


# In[13]:


sns.countplot(x="Survived", hue = 'Sex', data=titanic_train)


# In[14]:


sns.countplot(x = "Survived", hue = "Pclass", data = titanic_train)


# In[15]:


titanic_train['Age'].plot.hist()


# In[16]:


titanic_train['Fare'].plot.hist(bins = 20, figsize = (10,5))


# In[17]:


titanic_train.info()
print("----------------------------")
titanic_test.info()


# Above mentioned information shows some missing values present in both training and test datasets

# In[18]:


sns.countplot(x= "SibSp", data = titanic_train)


# In[19]:


sns.countplot(x = "Parch", data = titanic_train)


# ### Data Pre-Processing & Data Cleaning

# Let's see, how to check missed data?

# In[20]:


titanic_train.isnull().sum()


# In[21]:


sns.heatmap(titanic_train.isnull(), yticklabels=False, cmap = 'viridis')


# In[22]:


sns.boxplot(x = 'Pclass', y = 'Age', data = titanic_train)


# In[23]:


print("Training dataset columns:",titanic_train.columns)
print("-------------------------------")
print("Training dataset columns:",titanic_test.columns)


# In[24]:


# Drop unnecessary columns, these columns won't be useful in analysis and prediction
titanic_train = titanic_train.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
titanic_test = titanic_test.drop(['Name','Ticket','Cabin'], axis=1)


# In[25]:


print("Training dataset columns:",titanic_train.columns)
print("-------------------------------")
print("Training dataset columns:",titanic_test.columns)


# ### Embarked

# In[26]:


# Only in titanic_train dataset, fill the two missing values with the most occurred value, which is "S".
titanic_train["Embarked"] = titanic_train["Embarked"].fillna("S")

# plot
sns.factorplot('Embarked','Survived', data=titanic_train, size=5, aspect=3)

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

# sns.factorplot('Embarked', data=titanic_df, kind='count', order=['S','C','Q'], ax=axis1)
# sns.factorplot('Survived',hue="Embarked",data=titanic_df,kind='count',order=[1,0],ax=axis2)
sns.countplot(x='Embarked', data=titanic_train, ax=axis1)
sns.countplot(x='Survived', hue="Embarked", data=titanic_train, order=[1,0], ax=axis2)

# group by embarked, and get the mean for survived passengers for each value in Embarked
embark_perc = titanic_train[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)


# In[27]:


'''
Either to consider Embarked column in predictions, and remove "S" dummy variable, 
and leave "C" & "Q", since they seem to have a good rate for Survival.

OR, don't create dummy variables for Embarked column, just drop it, 
because logically, Embarked doesn't seem to be useful in prediction.
'''

embark_dummies_titanic_train  = pd.get_dummies(titanic_train['Embarked'])
embark_dummies_titanic_train.drop(['S'], axis=1, inplace=True)

embark_dummies_titanic_test  = pd.get_dummies(titanic_test['Embarked'])
embark_dummies_titanic_test.drop(['S'], axis=1, inplace=True)


# In[28]:


titanic_train = titanic_train.join(embark_dummies_titanic_train)
titanic_test  = titanic_test.join(embark_dummies_titanic_test)


# In[29]:


titanic_train.drop(['Embarked'], axis=1,inplace=True)
titanic_test.drop(['Embarked'], axis=1,inplace=True)


# In[30]:


titanic_train.head()


# In[31]:


titanic_test.head()


# ## Fare

# In[32]:


# Only for titanic_test, since there is a missing "Fare" values
titanic_test["Fare"].fillna(titanic_test["Fare"].median(), inplace=True)


# In[33]:


# Convert from float to int
titanic_train['Fare'] = titanic_train['Fare'].astype(int)
titanic_test['Fare'] = titanic_test['Fare'].astype(int)


# In[34]:


# Get fare for survived & didn't survive passengers 
fare_not_survived = titanic_train["Fare"][titanic_train["Survived"] == 0]
fare_survived     = titanic_train["Fare"][titanic_train["Survived"] == 1]

# Get average and std for fare of survived/not survived passengers
avg_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])


# In[35]:


# plot
titanic_train['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))

avg_fare.index.names = std_fare.index.names = ["Survived"]
avg_fare.plot(yerr=std_fare,kind='bar',legend=False)


# ## Age

# In[36]:


fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original Age Values - Titanic')
axis2.set_title('New Age Values - Titanic')

# axis3.set_title('Original Age values - Test')
# axis4.set_title('New Age values - Test')

# Get Average, STD, and Number of NaN values in titanic_train
avg_age_titanic_train = titanic_train["Age"].mean()
std_age_titanic_train = titanic_train["Age"].std()
count_nan_age_titanic_train = titanic_train["Age"].isnull().sum()

# Get Average, STD, and Number of NaN values in titanic_test
avg_age_titanic_test = titanic_test["Age"].mean()
std_age_titanic_test = titanic_test["Age"].std()
count_nan_age_titanic_test = titanic_test["Age"].isnull().sum()

# Generate random numbers between (mean - std) & (mean + std)
rand_1 = np.random.randint(avg_age_titanic_train - std_age_titanic_train, avg_age_titanic_train + std_age_titanic_train, size = count_nan_age_titanic_train)
rand_2 = np.random.randint(avg_age_titanic_test - std_age_titanic_test, avg_age_titanic_test + std_age_titanic_test, size = count_nan_age_titanic_test)

# plot original Age values
# NOTE: drop all null values, and convert to int
titanic_train['Age'].dropna().astype(int).hist(bins=70, ax=axis1)
# test_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

# fill NaN values in Age column with random values generated
titanic_train["Age"][np.isnan(titanic_train["Age"])] = rand_1
titanic_test["Age"][np.isnan(titanic_test["Age"])] = rand_2

# Convert from float to int
titanic_train['Age'] = titanic_train['Age'].astype(int)
titanic_test['Age'] = titanic_test['Age'].astype(int)
        
# plot new Age Values
titanic_train['Age'].hist(bins=70, ax=axis2)
#titanic_test['Age'].hist(bins=70, ax=axis4)


# In[37]:


# peaks for survived/not survived passengers by their age
facet = sns.FacetGrid(titanic_train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, titanic_train['Age'].max()))
facet.add_legend()

# average survived passengers by age
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
avg_age = titanic_train[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=avg_age)


# ## Family

# In[38]:


""" 
Instead of having two columns Parch & SibSp, We can have only one column represent 
if the passenger had any family member aboard or not,
Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.
"""
# make changes with training dataset
titanic_train['Family'] =  titanic_train["Parch"] + titanic_train["SibSp"]
titanic_train['Family'].loc[titanic_train['Family'] > 0] = 1
titanic_train['Family'].loc[titanic_train['Family'] == 0] = 0

# make changes with test dataset
titanic_test['Family'] =  titanic_test["Parch"] + titanic_test["SibSp"]
titanic_test['Family'].loc[titanic_test['Family'] > 0] = 1
titanic_test['Family'].loc[titanic_test['Family'] == 0] = 0


# In[39]:


# Now we will drop Parch & SibSp
titanic_train = titanic_train.drop(['SibSp','Parch'], axis=1)
titanic_test = titanic_test.drop(['SibSp','Parch'], axis=1)


# In[40]:


# Plot
fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))

# sns.factorplot('Family',data=titanic_train,kind='count',ax=axis1)
sns.countplot(x='Family', data=titanic_train, order=[1,0], ax=axis1)

# average of survived for those who had/didn't have any family member
family_perc = titanic_train[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()
sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)

axis1.set_xticklabels(["With Family","Alone"], rotation=0)


# ## Sex

# In[41]:


# As we see, children(age < 16) on aboard seem to have a high chances for Survival.
# So, we can classify passengers as males, females, and child

def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex
    
titanic_train['Person'] = titanic_train[['Age','Sex']].apply(get_person,axis=1)
titanic_test['Person'] = titanic_test[['Age','Sex']].apply(get_person,axis=1)

# No need to use Sex column since we created Person column
titanic_train.drop(['Sex'],axis=1,inplace=True)
titanic_test.drop(['Sex'],axis=1,inplace=True)


# In[42]:


# Create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
person_dummies_titanic_train  = pd.get_dummies(titanic_train['Person'])
person_dummies_titanic_train.columns = ['Child','Female','Male']
person_dummies_titanic_train.drop(['Male'], axis=1, inplace=True)

person_dummies_titanic_test  = pd.get_dummies(titanic_test['Person'])
person_dummies_titanic_test.columns = ['Child','Female','Male']
person_dummies_titanic_test.drop(['Male'], axis=1, inplace=True)

titanic_train = titanic_train.join(person_dummies_titanic_train)
titanic_test = titanic_test.join(person_dummies_titanic_test)

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))

# sns.factorplot('Person',data=titanic_train,kind='count',ax=axis1)
sns.countplot(x='Person', data=titanic_train, ax=axis1)

# average of survived for each Person(male, female, or child)
person_perc = titanic_train[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()
sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])


# In[43]:


titanic_train.drop(['Person'],axis=1,inplace=True)
titanic_test.drop(['Person'],axis=1,inplace=True)


# # PClass

# In[44]:


# sns.factorplot('Pclass',data=titanic_train,kind='count',order=[1,2,3])
sns.factorplot('Pclass','Survived',order=[1,2,3], data=titanic_train, size=5)

# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers
pclass_dummies_titanic_train  = pd.get_dummies(titanic_train['Pclass'])
pclass_dummies_titanic_train.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_titanic_train.drop(['Class_3'], axis=1, inplace=True)

pclass_dummies_titanic_test  = pd.get_dummies(titanic_test['Pclass'])
pclass_dummies_titanic_test.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_titanic_test.drop(['Class_3'], axis=1, inplace=True)

titanic_train = titanic_train.join(pclass_dummies_titanic_train)
titanic_test = titanic_test.join(pclass_dummies_titanic_test)


# In[45]:


titanic_train.drop(['Pclass'],axis=1,inplace=True)
titanic_test.drop(['Pclass'],axis=1,inplace=True)


# In[46]:


titanic_train.head()


# In[47]:


titanic_test.head()


# In[48]:


# Descriptive statistics for each column
titanic_train.describe()


# In[49]:


titanic_test.describe()


# ## Define Training and Testing datasets

# In[50]:


X_train = titanic_train.drop("Survived",axis=1)
Y_train = titanic_train["Survived"]
X_test  = titanic_test.drop("PassengerId",axis=1).copy()


# ## Training and Testing the Models
# 1. Logistic Regression
# 2. SVM (Support Vetor Machine)
# 3. Random Forest

# ### 1. Logistic Regression
# 

# In[51]:


logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = logreg.score(X_train, Y_train)

acc_log


# ## 2. Support Vector Machine (SVM)

# In[52]:


# Support Vector Machines

svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = svc.score(X_train, Y_train)

acc_svc


# ## 3. Random Forest

# In[53]:


# Random Forests

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

acc_random_forest = random_forest.score(X_train, Y_train)

acc_random_forest


# ## Evaluation

# In[54]:


# Get Correlation Coefficient for each feature using Logistic Regression
coeff_df = pd.DataFrame(titanic_train.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])

# preview
coeff_df


# In[55]:


models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines',
              'Random Forest'],
    'Score': [acc_log, acc_svc,
              acc_random_forest]})
models.sort_values(by='Score', ascending=False)


# ## Submission

# In[56]:


submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('titanic_submission1.csv', index=False)


# ![](https://miro.medium.com/max/1000/0*w5x4Af4EEQPvD7La)

# I hope this kernal is useful to you to learn exploratory data analysis and classification problem.
# 
# If find this notebook help you to learn, Please Upvote.
# 
# Thank You!!
