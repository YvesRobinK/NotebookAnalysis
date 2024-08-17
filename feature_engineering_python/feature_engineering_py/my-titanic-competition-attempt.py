#!/usr/bin/env python
# coding: utf-8

# # Introduction
#     
#     Content:
#     
# 1.  [Loading Data](#1)
# 
# 1.  [Variable Description](#2)
#     * [Univariate Variable Analysis](#3)
#         * [Categorical Variable Analysis](#4)
#         * [Numerical Variable Analysis](#5)
# 
# 1.  [Basic Data Analysis](#6)
# 
# 1.  [Outlier Detection](#7)
# 
# 1.  [Missing Values](#8)
#     * [Finding Missing Values](#9)
#     * [Filling Missing Values In](#10)
# 
# 1.  [Visualization](#11)
#     * [Correlation](#12)
#     * ["SibSp" and "Survived"](#13)
#     * ["ParCh" and "Survived"](#14)
#     * ["PClass" and "Survived"](#15)
#     * ["Age" and "Survived"](#16)
#     * ["PClass", "Age" and "Survived"](#17)
#     * ["Embarked", "PClass" and "Survived"](#18)
#     * ["Embarked", "Sex", "Fare" and "Survived"](#19)
# 1.  [Filling Missing Age Values](#20)
# 1.  [Feature Engineering](#21)
#     *  [Name and Title](#22)
#     *  [Family Size](#23)
#     *  [Embarked](#24)
#     *  [Ticket](#25)
#     *  [Pclass](#26)
#     *  [Sex](#27)
#     * [Dropping Passenger ID & Cabin](#28)
# 1. [Modelling](29)
#     *  [Train Test Split](#30)
#     *  [Simple Logistic Regression](#31)
#     *  [Hyperparameter Tuning, Grid Search & Cross Validation](#32)
#     *  [Ensemble Modelling](#33)
# 1. [Prediction & Submission](#34)

# <a id = "1"></a>
# 
# # 1. Loading Data

# In[1]:


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

import os

import warnings
warnings.filterwarnings("ignore")

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
test_PassengerId = test_df["PassengerId"] # for future purposes


# In[3]:


train_df.head()


# In[4]:


train_df.columns


# In[5]:


train_df.describe()


# <a id = "2"></a>
# 
# ## 2. Variable Descriptions
# 
# 1. PassengerId: unique ID # of each passenger
# 1. Survived: passenger survived (1) or did not (0)
# 1. Pclass: class of each passenger
# 1. Name
# 1. Sex
# 1. Age
# 1. SibSp: # of Siblings or Spouses
# 1. Parch: # of Parents or children
# 1. Ticket: assigned ticket number
# 1. Fare: price paid for a ticket
# 1. Cabin: cabin category
# 1. Embarked: which port passenger embarked from (C: Cherbourg, Q: Queenstown, S: Southampton)

# In[6]:


train_df.info()


# ### Every data type:
# 
# * float64(2): Fare | Age
# * int64(5): Pclass | SibSp | Parch | passengerid | survived
# * object(5): Cabin | Embarked | Ticket | Name | Sex

# <a id = "3"></a>
# # Univariate Variable Analysis
# * Categorical Variable: Survived | Sex | Pclass | Embarked | Cabin | Name | Ticket | SibSp | Parch
# * Numerical Variable: PassengerId | Age | Fare

# <a id = "4"></a>
# ## Categorical Variables

# In[7]:


def bar_plot(variable):
    """
    input: variable, example: "Sex"
    output: bar plot & value count
    
    """
    # getting the feature
    
    var=train_df[variable]
    
    # counting the number of categorical variables (value or sample)
    
    varValue=var.value_counts()
    
    # visualizing
    
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))


# In[8]:


category1 = ["Survived", "Sex", "Pclass", "Embarked", "SibSp", "Parch"]
for c in category1:
    bar_plot(c)


# In[9]:


# moving on to remaining categorical variables
# unlike the ones so far,
# these can be confusing when visualized

category2=["Cabin","Name","Ticket"]
for c in category2:
    print("{} \n".format(train_df[c].value_counts()))


# <a id = "5"></a>
# 
# ## Numerical Variables

# In[10]:


def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(train_df[variable], bins =  80)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()


# In[11]:


numericVar = ["Fare", "Age", "PassengerId"]
for n in numericVar:
    plot_hist(n)


# <a id = "6"></a>
# 
# ## 3. Basic Data Analysis
# 
# In this part, I will analyse the relationship between certain features vs 'Survived' to see what are the chances of survival for passengers with different features in our data.

# In[12]:


# correlation between features

train_df.corr()


# In[13]:


# heatmap of correlation

f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(train_df.corr(), annot=True, linewidths =.5, fmt ='.1f',ax=ax)
plt.show()


# In[14]:


# Pclass vs Survived

train_df[["Pclass","Survived"]]


# In[15]:


# Pclass vs Survived

train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[16]:


# Sex vs Survived

train_df[["Sex","Survived"]].groupby(["Sex"], as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[17]:


# SibSp vs Survived

train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[18]:


# Parch vs Survived

train_df[["Parch","Survived"]].groupby(["Parch"], as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[19]:


# the chances of females who had SibSp amounts of siblings or spouses surviving

train_df[["SibSp", "Sex","Survived"]].groupby(["SibSp", "Sex"], as_index=False).mean().sort_values(by="Survived",ascending=False)


# <a id = "7"></a>
# # 4. Outlier Detection
# 
# In this part, I will demonstrate how to detect and eliminate outliers in our data.

# In[20]:


def detect_outliers(df,features):
    outlier_indices = []
    
    for c in features:
        
        # 1st quartile:
        
        Q1 = np.percentile(df[c],25)
        
        # 3rd quartile:
        
        Q3 = np.percentile(df[c],75)
        
        # IQR:
        
        IQR = Q3 - Q1
        
        # Outlier step:
        
        outlier_step = IQR * 1.5
        
        # detect outlier and their indices:
        
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        
        # store indices:
        
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers


# In[21]:


train_df.loc[detect_outliers(train_df,["Age", "SibSp","Parch","Fare"])]


# In[22]:


# dropping outliers

train_df = train_df.drop(detect_outliers(train_df,["Age", "SibSp","Parch","Fare"]), axis = 0).reset_index(drop=True)


# <a id = "8"></a>
# 
# ## 5.Missing Values

# In[23]:


train_df_len = len(train_df)

train_df = pd.concat([train_df, test_df], axis = 0).reset_index(drop = True)


# <a id = "9"></a>
# ## Finding Missing Values

# In[24]:


train_df.columns[train_df.isnull().any()]


# In[25]:


# these values are normal because of the concatenation

train_df.isnull().sum()


# <a id = "10"></a>
# ## Filling Missing Values In

# In[26]:


train_df["Embarked"].isnull()


# In[27]:


train_df[train_df["Embarked"].isnull()]


# In[28]:


train_df.boxplot(column="Fare",by = "Embarked")


# In[29]:


train_df["Embarked"] = train_df["Embarked"].fillna("C")


# In[30]:


train_df[train_df["Embarked"].isnull()]


# In[31]:


train_df[train_df["Fare"].isnull()]


# In[32]:


train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))


# In[33]:


# how much these passengers have paid

third_class_price = train_df[train_df["Pclass"] ==3]["Fare"]
third_class_price


# In[34]:


train_df[train_df["Fare"].isnull()]


# <a id = "11"></a>
# # 6. Visualization
# 
# Starting with correlation, we will visualize the data.

# <a id = "12"></a>
# # Correlation

# In[35]:


list1 = ["SibSp", "Parch", "Age", "Fare", "Survived"]
sns.heatmap(train_df[list1].corr(), annot=True, fmt = ".2f")
plt.show()


# The highest values in this correlation matrix are 0.41 and 0.26, which indicate that:
# 
# Parent&Children <---> Sibling&Spouse
# 
# Fare <---> Survived
# 
# both have a correlation

# <a id = "13"></a>
# # "SibSp" and "Survived"

# In[36]:


g = sns.factorplot(x = "SibSp", y = "Survived", data = train_df, kind = "bar", size = 6)
g.set_ylabels("Probability of Surviving")
g.set_xlabels("Total # of Siblings and Spouses")
plt.show()


# More siblings and/or spouses mean less chance of surviving.

# <a id = "14"></a>
# # "ParCh" and "Survived"

# In[37]:


g = sns.factorplot(x = "Parch", y = "Survived", kind = "bar", data = train_df, size = 6)
g.set_ylabels("Probability of Surviving")
g.set_xlabels("Total # of Parents and Children")
plt.show()


# <a id = "15"></a>
# # "PClass" and "Survived"

# In[38]:


g = sns.factorplot(x="Pclass", y ="Survived", data = train_df, kind = "bar", size = 5)
g.set_ylabels("Probability of SUrviving")
g.set_xlabels("Class no.")


# <a id = "16"></a>
# # "Age" and "Survived"

# In[39]:


g = sns.FacetGrid(train_df, col = "Survived")
g.map(sns.distplot, "Age", bins = 25)
plt.show()


# <a id = "17"></a>
# # "PClass", "Age" and "Survived"

# In[40]:


g = sns.FacetGrid(train_df, col = "Survived", row = "Pclass")
g.map(plt.hist, "Age", bins = 25)
g.add_legend()
plt.show()


# The reverse relationship between being a higher class passenger and having a chance of survival is apparent

# <a id = "18"></a>
# # "Embarked", "PClass" and "Survived"

# In[41]:


g = sns.FacetGrid(train_df, row = "Embarked", size = 5)
g.map(sns.pointplot, "Pclass", "Survived", "Sex")
plt.show()
g.add_legend()


# <a id = "19"></a>
# # "Embarked", "Sex", "Fare" and "Survived"

# In[42]:


g = sns.FacetGrid(train_df, row = "Embarked", col = "Survived", size = 2.5)
g.map(sns.barplot, "Sex", "Fare")
g.add_legend()
plt.show()


# We can see that higher fare means higher chance of survival.

# <a id = "20"></a>
# # 7.Filling Missing Age Values

# In[43]:


train_df[train_df["Age"].isnull()]


# In[44]:


sns.factorplot(x = "Sex", y = "Age", data = train_df, kind = "box")


# We can deduce that sex is not informative in filling missing age values, since the medians are very close for each sex.

# In[45]:


sns.factorplot(x = "Pclass", y = "Age", data = train_df, kind = "box")


# ages of passengers are higher as classes of passengers go from 3 to 2 to 1.

# In[46]:


sns.factorplot(x = "Parch", y = "Age", data = train_df, kind = "box")


# In[47]:


sns.heatmap(train_df[["Age", "Sex", "SibSp", "Parch", "Pclass"]].corr(), annot = True)
plt.show()


# No "Sex" feature in this heatmap since its categorical, we want to correct that.

# In[48]:


train_df["Sex"] = [1 if i == "male" else 0 for i in train_df["Sex"]]


# In[49]:


sns.heatmap(train_df[["Age", "Sex", "SibSp", "Parch", "Pclass"]].corr(), annot = True)
plt.show()


# In[50]:


index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)
for i in index_nan_age:
    age_pred = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) &(train_df["Parch"] == train_df.iloc[i]["Parch"])& (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))].median()
    age_med = train_df["Age"].median()
    if not np.isnan(age_pred):
        train_df["Age"].iloc[i] = age_pred
    else:
        train_df["Age"].iloc[i] = age_med


# In[51]:


train_df[train_df["Age"].isnull()]


# <a id = "21"></a>
# # 8.Feature Engineering

# <a id = "22"></a>
# ## Name and Title

# In[52]:


train_df["Name"].head(10)


# In[53]:


name = train_df["Name"]
train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]
train_df["Title"].head(10)


# In[54]:


sns.countplot(x="Title", data = train_df)
plt.xticks(rotation = 60)
plt.show()


# In[55]:


train_df["Title"].unique()


# In[56]:


# convert to categorical

train_df["Title"] = train_df["Title"].replace(["Dr",'Don','Rev','Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'the Countess', 'Jonkheer', 'Dona'], "Other")


# In[57]:


sns.countplot(x="Title", data = train_df)
plt.xticks(rotation = 60)
plt.show()


# In[58]:


train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Mrs" or i == "Mme" else 2 if i == "Mr" else 3 for i in train_df["Title"]]


# In[59]:


sns.countplot(x="Title", data = train_df)
plt.xticks(rotation = 60)
plt.show()


# In[60]:


g = sns.factorplot(x = "Title", y = "Survived", data = train_df, kind = "bar")
g.set_xticklabels(["Master", "Mrs", "Mr", "Other"])
g.set_ylabels("Survival Probability")
plt.show()


# In[61]:


train_df.drop(labels = ["Name"], axis = 1, inplace = True)


# In[62]:


train_df.head()


# In[63]:


train_df = pd.get_dummies(train_df, columns=["Title"])
train_df.head()


# <a id = "23"></a>
# ## Family Size

# In[64]:


train_df["Fsize"] = train_df["SibSp"] + train_df["Parch"] + 1
train_df.head(5)


# In[65]:


g = sns.factorplot(x = "Fsize", y = "Survived", data = train_df, kind = "bar")
g.set_ylabels("Survival")
plt.show()


# In[66]:


train_df["family_size"] = [1 if i < 5 else 0 for i in train_df["Fsize"]]


# In[67]:


train_df.head(10)


# In[68]:


sns.countplot(x = "family_size", data = train_df)
plt.show()


# In[69]:


# therefore, big families have less chance of survival


# In[70]:


train_df = pd.get_dummies(train_df, columns = ["family_size"])


# In[71]:


train_df.head(10)


# <a id = "24"></a>
# ## Embarked

# In[72]:


train_df["Embarked"].head()


# In[73]:


sns.countplot(x = "Embarked", data = train_df)


# In[74]:


train_df = pd.get_dummies(train_df, columns = ["Embarked"])


# In[75]:


train_df.head(10)


# <a id = "25"></a>
# ## Ticket

# In[76]:


train_df["Ticket"].head(10)


# In[77]:


tickets = []
for i in list(train_df.Ticket):
    if not i.isdigit():
        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])
    else:
         tickets.append("X")
train_df["Ticket"] = tickets


# In[78]:


# making tickets categorical

train_df = pd.get_dummies(train_df, columns = ["Ticket"], prefix = "T")


# In[79]:


train_df.head(10)


# <a id = "26"></a>
# ## Pclass

# In[80]:


sns.countplot(x = "Pclass", data = train_df)
plt.show()


# In[81]:


train_df["Pclass"] = train_df["Pclass"].astype("category")
train_df = pd.get_dummies(train_df, columns = ["Pclass"])


# In[82]:


train_df.head(10)


# <a id = "27"></a>
# ## Sex

# In[83]:


train_df["Sex"] = train_df["Sex"].astype("category")
train_df = pd.get_dummies(train_df, columns = ["Sex"])


# In[84]:


train_df.head()


# <a id = "28"></a>
# ## Dropping Passenger ID & Cabin 

# In[85]:


train_df.drop(labels = ["PassengerId", "Cabin"], axis = 1, inplace = True)


# <a id = "29"></a>
# # 9.Modelling

# In[86]:


train_df


# In[87]:


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# <a id = "30"></a>
# ## Train Test Split

# In[88]:


train_df_len


# In[89]:


test = train_df[train_df_len:]
test.drop(labels = ["Survived"], axis = 1, inplace = True)


# In[90]:


test.head()


# In[91]:


train = train_df[:train_df_len]
X_train = train.drop(labels = "Survived", axis = 1)
y_train = train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.33, random_state = 0)


# In[92]:


print("X_train", len(X_train))
print("X_test", len(X_test))
print("y_train", len(y_train))
print("y_test", len(y_test))
print("test", len(test))


# <a id = "31"></a>
# ## Simple Logistic Regression

# In[93]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
acc_log_train = round(logreg.score(X_train, y_train)*100,2)
acc_log_test = round(logreg.score(X_test, y_test)*100,2)
print("Training Accuracy: % {}".format(acc_log_train))
print("Testing Accuracy: % {}".format(acc_log_test))


# <a id = "32"></a>
# ## Hyperparameter Tuning, Grid Search & Cross Validation
# 
# We will compare 5 classifier methods & evaluate them
# * Decision Tree
# * SVM
# * Random Forest
# * KNN
# * Logistic Regression

# In[94]:


random_state = 40
classifier = [DecisionTreeClassifier(random_state = random_state),
             SVC(random_state = random_state),
             RandomForestClassifier(random_state = random_state),
             LogisticRegression(random_state = random_state),
             KNeighborsClassifier()]

dt_param_grid = {"min_samples_split" : range(10,500,20),
                "max_depth": range(1,20,2)}

svc_param_grid = {"kernel" : ["rbf"],
                 "gamma": [0.001, 0.01, 0.1, 1],
                 "C": [1,10,50,100,200,300,1000]}

rf_param_grid = {"max_features": [1,3,10],
                "min_samples_split":[2,3,10],
                "min_samples_leaf":[1,3,10],
                "bootstrap":[False],
                "n_estimators":[100,300],
                "criterion":["gini"]}

logreg_param_grid = {"C":np.logspace(-3,3,7),
                    "penalty": ["l1","l2"]}

knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),
                 "weights": ["uniform","distance"],
                 "metric":["euclidean","manhattan"]}
classifier_param = [dt_param_grid,
                   svc_param_grid,
                   rf_param_grid,
                   logreg_param_grid,
                   knn_param_grid]


# In[95]:


# grid search

cv_result = []
best_estimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)
    clf.fit(X_train,y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])


# In[96]:


# create a seaborn to visualize results


# In[97]:


cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",
             "LogisticRegression",
             "KNeighborsClassifier"]})

g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)
g.set_xlabel("Mean Accuracy")
g.set_title("Cross Validation Scores")


# <a id = "33"></a>
# ## Ensemble Modelling

# In[98]:


votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),
                                        ("rfc",best_estimators[2]),
                                        ("lr",best_estimators[3])],
                                        voting = "soft", n_jobs = -1)
votingC = votingC.fit(X_train, y_train)
print(accuracy_score(votingC.predict(X_test),y_test))


# <a id = "34"></a>
# # 10.Prediction & Submission

# In[99]:


test_survived = pd.Series(votingC.predict(test), name = "Survived").astype(int)
results = pd.concat([test_PassengerId, test_survived],axis = 1)
results.to_csv("titanic.csv", index = False)


# ### Citation:
# DATAI Team's tutorial has been helpful in helping me create this. 
