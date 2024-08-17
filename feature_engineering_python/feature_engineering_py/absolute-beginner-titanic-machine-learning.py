#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster
# 
# This notebook is detailed walkthrough of all the steps I took to make my first Kaggle submission. The Challenges I faced in the process are not unique to you or someone who is just starting with Kaggle. Keeping in mind I have provided a detailed description of steps along with references and links. Taking reference from various internet blogs and videos, I finally put my code to work, I hope you find it useful and a kickstart for your ML journey. As you explore through more and more projects and get your hands dirty, you will gain more knowlege. The tough part is to get started, and believe me to just take a leap of faith. and its okay if you dont understand the code completely, with repetions the things form a clear picture of concepts.So lets dive in!

# #### Best Model: Random Forest Classifier
# #### Best Score on train data: 80.92
# #### Score on submission: 74.16

# ## Contents:
# 
# 1. Project Skeleton
# 2. Loading dataset (Downloaded from Kaggle)
# 3. Exploratory Data Analysis
#     -   Exploring Missing Values
#     -  Data Interpretation and Visualization
#     - Count Plot for Features
#     -  Feature Relationships
# 4. Data Pre-processing and Feature Engineering
#     -   Check Feature Data Types
#     -  Walk Through Each Feature One by One
#     - Re-Check Datasets
# 5. Modelling
#     -  Try Different Models
#     - Survival Prediction on Test Data
# 6. Conclusion

# ## 1. Project Skeleton
# 
# Before starting out any project, we must first plan our steps and have clarity on what type of problem we are tackling and what tools can be used and what cannot be used and why not?. This "why not" question will help you gain more insights on your ML journey. The following are key points I took into consideration.

# #### Staircase
# 1. What kind of ML problem statement is it? Try to define it
# 2. Understand the type of data?
# 3. Plot the values counts
# 4. Check what data is missing (and also how can it be filled?)
# 5. Relationships between various features
# 6. Try your intuition about the field:
#     i. Are people with family more likely to survive?
#     ii. Are richer people more likely to survive
#     iii. Location of cabin and how the ship actually sank, did the front part sink first?
# 7. Feature Engineering
# 8. Pre-processing and scaling features
# 9. Apply some base models and move to more advanced ones
# 10. Cross validate on different subsets of data
# 11. Chose the best one and predict on Test data
# 

# ## 2. Loading Datasets
# Here I have basically downloaded data into my repository and loaded them into pandas dataframe. Nothing too fancy

# In[ ]:


#importing base libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#magic function to view plot inside jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')

#loading datasets
train=pd.read_csv("../input/titanic/train.csv")
test=pd.read_csv("../input/titanic/test.csv")


# ## 3. Exploratory Data Analysis
# Major portion of time while solving a ML problem will go into EDA and feature engineering. Applying a model is really 2-4 lines of code and easy part. So be patient here and use your own intuition. if you have strong foundation in stats, this can be really easy for you. Here we will first explore misisng data, then visualize the features and generate correlation to have an idea about how features are inter-related to each other.

# ### I. Exploring Missing Values
# 
# Reference for msno: https://towardsdatascience.com/visualize-missing-values-with-missingno-ad4d938b00a1

# In[ ]:


import missingno as msno #for visualizing missing data

#train dataset missing data and preview
print("Train data rows x columns: ",train.shape,"\n")
print(train.isnull().sum())
train.head()


# In[ ]:


#test dataset missing data and preview
print("Test data rows x columns: ",train.shape,"\n")
print(test.isnull().sum())
test.head()


# In[ ]:


#visualize missing data in train as matrix
msno.matrix(train)


# In[ ]:


#visualize missing data in train as matrix
msno.matrix(test)


# In[ ]:


#visualize missing data in train as bar chart
msno.bar(train)


# In[ ]:


#visualize missing data in test as bar chart
msno.bar(test)


# In[ ]:


#check if more than 40% of information is missing in columns for train data
print(train.isnull().sum()>int(0.40*train.shape[0]))


# In[ ]:


#check if more than 40% of information is missing in columns for test data
print(train.isnull().sum()>int(0.40*train.shape[0]))


# ### II. Data Interpretation and Visualization
# Reference for seaborn: https://towardsdatascience.com/data-visualization-using-seaborn-fc24db95a850

# In[ ]:


#Data is mssing in Age columns. Histogram plot for train data.
sns.distplot(train['Age'].dropna(),hist=True, kde=True,rug=True, bins=40)


# In[ ]:


#Histogram plot for test data
sns.distplot(test['Age'].dropna(),hist=True, kde=True, bins=40, rug=True)


# ### III. Count Plot for Features

# In[ ]:


sns.countplot(x="Survived",data=train,palette="deep")


# In[ ]:


sns.countplot(x="Survived",hue="Pclass",data=train,palette="deep")


# ### IV. Feature Relationships
# To get an idea about how features are related to each other, we can generate a correlation matrix and check the pearson correlation coefficient
# 
# Reference: https://www.statisticshowto.com/probability-and-statistics/correlation-coefficient-formula/

# In[ ]:


corr =train.corr()
sns.heatmap(corr,annot=True)


# ## 4. Data Pre-processing and Feature Engineering

# ### I. Check Feature Data Types

# In[ ]:


#Check the data type of each feature
train.info()


# ### II. Walk Through Each Feature One by One
# Here we explore each feature one by one and check if we can directly drop it, use it or extract some information from it.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


# In[ ]:


#intantiate label encoder
le=LabelEncoder()


# In[ ]:


#Combining both datasets
dataset=[train,test]


# ### a. PassengerId
# This is just the serial number identification for passengers and doe not contain much infromation. we can drop it directly

# In[ ]:


#Dropping the PassengerId column only in train as we equire passengerid in our test dataset for submission
train.drop("PassengerId",axis=1,inplace=True)


# ### b. Name
# Name looks like it does not contain much information but, we can still create new feature "Title" and extract titles from each name in test and train data set which can be informative going further. To create this new feature we can use regular expressions

# In[ ]:


train["Name"]


# In[ ]:


for data in dataset:
    data["Title"]=data["Name"].str.extract('([A-Za-z]+)\.',expand=False)


# Check the Titles that were extracted

# In[ ]:


train["Title"].value_counts()


# In[ ]:


test["Title"].value_counts()


# Since most of the values are in Mr, Miss, Mrs, we can include the others in separate category and hence have four categories. We could have used label encoder but lot of small categories can be clubbed together into single category

# In[ ]:


#create mapping
title_mapping={"Mr":0,"Miss":1,"Mrs":2,"Master":3,"Col":3,"Rev":3,"Ms":3,"Dr":3,"Dona":3,"Major":3,
         "Mlle":3,"Countess":3,"Sir":3,"Jonkheer":3,"Don":3,"Mme":3,"Capt":3,"Lady":3}


# In[ ]:


for data in dataset:
    data["Title"]=data["Title"].map(title_mapping)


# In[ ]:


#check the count of survived according ot titles
sns.countplot(x="Survived",hue="Title",data=train,palette="deep")


# In[ ]:


#Dropping the column Names as it is no longer required
for data in dataset:
    data.drop("Name",axis=1,inplace=True)


# ### c. Sex 
# Since sex is a categorical variable we need to encode it using some sort of encoding, maybe manually using a dictionary or by the use of label encoder. Here I have used Label Encoder.

# In[ ]:


for data in dataset:
    data["Sex"]=le.fit_transform(data["Sex"])


# ### d. Age
# Since has lots of missing values lets check how we can fill it. Lets first check the skewness

# In[ ]:


Skewness = 3*(train["Age"].mean()-train["Age"].median())/train["Age"].std()
Skewness


# Skewness is positive so we can use medium to fill the missing values. But in our dataset we can do something better. We have feature "Names" from which we have extract titles like Mr,Mrs, master, miss, and now we can use the median of each group to fill the missing values for better accuracy of our model. 

# In[ ]:


for data in dataset:
    data["Age"].fillna(data.groupby("Title")["Age"].transform("median"),inplace=True)


# ### e. SibSp
# 
# it contains information about the family, So we will combine this feature along with Parch to have a new feature called "Family Size"

# ### f. Parch
# This also contains information about no of parents and children so let us combine it with SibSip to have new feature as "Family Size"

# In[ ]:


for data in dataset:
    data["Family Size"]=data["SibSp"]+data["Parch"]+1


# In[ ]:


#dropping the Parch and SibSp columns
for data in dataset:
    data.drop(["SibSp","Parch"],axis=1,inplace=True)


# ### g. Ticket
# This feature does not contain much data so we can drop it directly

# In[ ]:


#dropping Ticket column
for data in dataset:
    data.drop(["Ticket"],axis=1,inplace=True)


# ### h. Fare
# Since Fare has very few missing values, we can replace fare with its mean value

# In[ ]:


for data in dataset:
    data["Fare"].fillna(data["Fare"].mean(),inplace=True)


# ### i. Cabin
# Since more than 40% of value in cabin data is missing, we can drop it directly if we want. But here I will extract the first letter of each cabin name and apply a numerical mapping on it and hence fill the missing values later. 

# In[ ]:


test["Cabin"].value_counts()


# In[ ]:


for data in dataset:
    data["Cabin"] = data["Cabin"].str[:1]


# In[ ]:


cabin_mapping={"A":0,"B":0.4,"C":0.8,"D":1.2,"E":1.6,"F":2,"G":2.4,"T":2.8} #This is called feature scaling, please explore more on this advanced topic


# In[ ]:


for data in dataset:
    data["Cabin"] = data["Cabin"].map(cabin_mapping)


# Fill the missing value by grouping by Pclass, since cabins are related to class of booking

# In[ ]:


for data in dataset:
    data["Cabin"].fillna(data.groupby("Pclass")["Cabin"].transform("median"),inplace=True)


# ### j. Embarked
# Since Embarked has we missing values and is Categorical, we will just simply fill it by its mode from values S,C,Q.

# In[ ]:


for data in dataset:
    data["Embarked"].fillna(data["Embarked"].mode()[0],inplace=True)


# In[ ]:


#Label encoding Embarked
for data in dataset:
    data["Embarked"]=le.fit_transform(data["Embarked"])


# ### III. Re-Check Datasets 
# it is good practice to re-check datasets before actually applying models. Check if both train and test datasets have no null values. Here we must also separate labels and features from train dataset

# Checking for null values if any:

# In[ ]:


#for train dataset
sns.heatmap(train.isnull(),cmap = 'magma' )


# In[ ]:


#For test dataset
sns.heatmap(test.isnull(),cmap = 'magma' )


# In[ ]:


#separating labels
y_train = train["Survived"]
#separating features
train.drop("Survived", axis=1,inplace=True)
X_train=train


# In[ ]:


#checking train dataset
X_train.head()


# In[ ]:


#checking test dataset after removing passengerid as a copy of test data, as we reuire passengerid in final submission on Kaggle
X_test = test.drop("PassengerId",axis=1).copy()
X_test.head()


# In[ ]:


#Finally check the shapes
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)


# ## 5. Modelling
# We will start with some basic models and then go through more advanced ensemble models. Lets drive straight into it!

# In[ ]:


#import classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Import libraries and functions for cross validation and metrics for accuracy

# In[ ]:


from sklearn.model_selection import KFold, cross_val_score
#set values for K-folds
folds= KFold(n_splits=10,shuffle=True,random_state=0)
metric="accuracy"


# ### I. Try Different Models

# ### a. K-nearest neighbor Classifier

# In[ ]:


knn=KNeighborsClassifier(n_neighbors=10)
score= cross_val_score(knn,X_train,y_train,cv=folds,n_jobs=1,scoring=metric)
print(score)


# In[ ]:


#mean score rounded to 2 decimal points
round(np.mean(score)*100,2)


# ### b. Decision Tree Classifier

# In[ ]:


dtc=DecisionTreeClassifier()
score= cross_val_score(dtc,X_train,y_train,cv=folds,n_jobs=1,scoring=metric)
print(score)


# In[ ]:


#mean score rounded to 2 decimal points
round(np.mean(score)*100,2)


# ### c. Random Forest Classifier

# In[ ]:


rfc=RandomForestClassifier()
score= cross_val_score(rfc,X_train,y_train,cv=folds,n_jobs=1,scoring=metric)
print(score)


# In[ ]:


#mean score rounded to 2 decimal points
round(np.mean(score)*100,2)


# ### d. Gaussian NB Classifier

# In[ ]:


gnb=GaussianNB()
score= cross_val_score(gnb,X_train,y_train,cv=folds,n_jobs=1,scoring=metric)
print(score)


# In[ ]:


#mean score rounded to 2 decimal points
round(np.mean(score)*100,2)


# ### e. Support Vector Machine Classifier

# In[ ]:


svmcl=SVC()
score= cross_val_score(svmcl,X_train,y_train,cv=folds,n_jobs=1,scoring=metric)
print(score)


# In[ ]:


#mean score rounded to 2 decimal points
round(np.mean(score)*100,2)


# ### II. Survival Prediction on Test Data

# In[ ]:


#using the classifier that gave highest average accuracy on train dataset
clf=RandomForestClassifier()
clf.fit(X_train,y_train)
y_test = clf.predict(X_test)


# In[ ]:


#wrapping up into a submission dataframe
submission=pd.DataFrame({"PassengerId": test["PassengerId"],"Survived":y_test})

#converting to submission csv
submission.to_csv("submission.csv",index=False)


# In[ ]:


submission=pd.read_csv("submission.csv")
submission.head()


# ## 6. Conclusion
# 
# The output csv is generated in the right pane in the kernel under folder "output" You can submit it on Kaggle using the following link:
# 
# https://www.kaggle.com/c/titanic
# 
# I am new to ML path, so this may not be the finest implementation but I tried to get my hands on with the knowledge I had. Feel free to comment on techniques I can use to improve the score of the models and dont forget to Upvote!
# 
# 
# 
# 

# In[ ]:




