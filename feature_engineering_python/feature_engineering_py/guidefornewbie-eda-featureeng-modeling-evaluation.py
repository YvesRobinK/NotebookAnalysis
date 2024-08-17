#!/usr/bin/env python
# coding: utf-8

# # Objective

# After learning from some exceptional work from incredible Kaggler's I have decided to come up with this Kernel for Exploratory Data Analysis and Feature Engineering along with basic data modeling and model evaluation. 
# 
# This is primarily for newbies in Machine Learning to introduce them with these terms and ways to use them. I have kept the language, code, and explanation as simple as possible for ease of understanding.
# 
# I have used [dataset](https://www.kaggle.com/c/titanic/data) which is provided by <a>Kaggle</a> for [Titanic: Machine Learning from Disaster Competition](https://www.kaggle.com/c/titanic/overview)
# 
# If you like the work please **upvote** and do leave a comment for any feedback.

# # Exploratory Data Analysis

# Let's start with importing libraries and data set

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


#Importing required libraries
#Importing the required libraries and data set 
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

print("Important libraries loaded successfully")


# In[3]:


ds_train=pd.read_csv("/kaggle/input/titanic/train.csv")
ds_test=pd.read_csv("/kaggle/input/titanic/test.csv")
ds_result=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
print("Train and Test data sets are imported successfully")


# ## Data Overview

# After importing the library let's check how many rows are present in Train and Test set.

# In[4]:


print("Test and Training data details are as follows: ")
print('Number of Training Examples = {}'.format(ds_train.shape[0]))
print('Number of Test Examples = {}\n'.format(ds_test.shape[0]))


# ## Features Analysis and Explanation

# Let's try to understand each features in training data set

# In[5]:


ds_train.head()


# * **Survived** is a target variable where survival is predicted in binanry format i.e. **0** for Not Survived and **1** for Survived
# * **PassengerId** and **Ticket** variables can be assumed as Random unique Identifiers of Passengers and they don't have any impact on outcome ,hence we can ignore them
# * **Pclass** is an ordinal datatype for the ticket class,it's a passenger's socio-economic status which played an important role in survival , it may impact target variable so we will keep it in our train data set. It's unique values are **1 = Upper Class** , **2 = Middle Class** and **3 = Lower Class**
# * **Name** It could be used to derive socio-economic status from title (like Doctor or Master)
# * **Sex** Gender played an important role in survival , so we will keep this in our feature list 
# * **SibSp and Parch** These two variables represent total number of the passenger's siblings/spouse and parents/children on board , it could be used to create a new variable 'Family Size'. This is an example of Feature Engineering
# * **Age** , Like Gender Age could also played a role in survival , will keep this is our feature list
# * **Fare** , price of ticket also represnt socio-economic status , let's keep this also 
# * **Cabin** this is Cabin number of the passenger and it can be used in feature engineering to get an approximate position of passenger when accident happened,also from deck level we can deduce socio-economic status. However, on closer look at data it looks like that there are many null values so we can drop this column from our feature list
# * **Embarked** is port of embarkation and it is a categorical feature which has following **3** unique values **C = Cherbourg**,**Q = Queenstown** and **S = Southampton** , this may have impact on target variable we will keep this variable for now.
# 
# Let's drop **Ticket** and **Cabin** columns from training data set

# In[6]:


#Drop columns from training data set
ds_train=ds_train.drop(['Ticket','Cabin'],axis=1)
print("Columns Dropped Successfully")
ds_train.head()


# Now , we will try to see some relation between these features.
# 
# First start with passenger's **Age**

# In[7]:


#Converting Age into series and visualizing the age distribution
age_series=pd.Series(ds_train['Age'].value_counts())
fig=px.scatter(age_series,y=age_series.values,x=age_series.index)
fig.update_layout(
    title="Age Distribution",
    xaxis_title="Age in Years",
    yaxis_title="Count of People",
    font=dict(
        family="Courier New, monospace",
        size=18,
    )
)
fig.show()


# We can see that there are some number of passengers who less than 20 years , let's calculate the count

# In[8]:


print("Number of teenagers and child passengers in ship are {}".format(len(ds_train[ds_train['Age'] < 20 ])))


# 📌 **Take Away Points**
# * Majority of passengers aged more than **20** years and less than **50** years
# * Maximum number of passengers (30 in numbers) are of **24** years old
# * There are **164** passengers who are less than 20 years old  
# 
# Let's break this further and add **Gender** with Age. First let's see how diversified among passenger

# In[9]:


print("Number of Passengers Gender Wise \n{}".format(ds_train['Sex'].value_counts()))
#Gender wise distribution
fig = go.Figure(data=[go.Pie(labels=ds_train['Sex'],hole=.4)])
fig.update_layout(
    title="Sex Distribution",
    font=dict(
        family="Courier New, monospace",
        size=18
    ))
fig.show()


# It's quiet evident that number of male passengers are almost double of female passengers.
# 
# Let's see how many female and male survived.

# In[10]:


#Create categorical variable graph for Age,Sex and Survived variables
sns.catplot(x="Survived", y="Age", hue="Sex", kind="swarm", data=ds_train,height=10,aspect=1.5)
plt.title('Passengers Survival Distribution: Age and Sex',size=25)
plt.show()


# It's pretty evident from above graph that majority of female passengers are survived
# 
# 📌 **Take Away Points**
# 
#  * Majority of Male passengers aged between 20 to 50 years had not survived . It means **most of the young men had not survived this disaster**
#  * Oldest male passenger aged 80 years ,had survived
#  * Age and Sex were major factors in deciding passenger's fate
#  
#  Now , let's see **Pclass** variable relation with survival

# In[11]:


#Visualize relation between Pclass and Survival
fig = go.Figure(data=[go.Pie(labels=ds_train['Pclass'],hole=.4)])
fig.update_layout(
    title="PClass Distribution",
    font=dict(
        family="Courier New, monospace",
        size=18
    ))
fig.show()


# More than half of the passengers were travelling in **Lower Class**. Let's see how survival is linked with Pclass

# In[12]:


#Visualize PClass and Survival
#Create categorical variable graph for Age,Pclass and Survived variables
sns.catplot(x="Survived", y="Age", hue="Pclass", kind="swarm", data=ds_train,height=10,aspect=1.5)
plt.title('Passengers Survival Distribution: Age and Pclass',size=25)
plt.show()


# Well,it looks like that majority of young passengers who are travelling in lower class had not survived
# 
# 📌 **Take Away Points**
# 
#  * Majority of young male passengers aged between 20 to 50 years and travelling in lower class had not survived 
#  * Oldest male passenger who survived the disaster was travelling in upper class
#  * Young men who survived the disaster were travelling in upper class
#  * Passengers Socio Economic Status palyed a vital role in survival
#  
#  > **We can deduce one thing clearly ,if passenger was man aged between 20-50 and not so rich at the time of travel then their chances of survival were very less**
# 
# To support our Socio Economic Status theory let's focus on one more variable **Fare**
# 
# 
# 

# In[13]:


#Visualize Fare and Survival
#Create categorical variable graph for Sex,Fare and Survived variables
sns.catplot(x="Survived", y="Fare", hue="Sex", kind="swarm", data=ds_train,height=8,aspect=1.5)
plt.title('Passengers Survival Distribution: Fare and Sex',size=20)
plt.show()


# It's clear that female passengers with lowest fare also survived the disaster and passenger with highest fare also survived , irrespective of the gender and this proves our theory that **Socio Economic Status played an improtant role in survival**
# 
# At last we will see **Embarked** variable's impact on survival

# In[14]:


#Visualize relation between Embarked and Survival
fig = go.Figure(data=[go.Pie(labels=ds_train['Embarked'],hole=.4)])
fig.update_layout(
    title="Embarked Distribution",
    font=dict(
        family="Courier New, monospace",
        size=18
    ))
fig.show()


# Majority of passengers embarked from **Southampton** , it may be the journey start point

# In[15]:


#Visualize Embarked and Survival
#Create categorical variable graph for Embarked,Age and Survived variables
sns.catplot(x="Survived", y="Age", hue="Embarked", kind="swarm", data=ds_train,height=8,aspect=1.5)
plt.title('Passengers Survival Distribution: Embarked and Age',size=20)
plt.show()


# As there is no direct releation between Embarked and Survived variables we can drop this from our feature list.
# 
# Also, we can drop 'Name' column from our feature list as we have other features/columns for Socio Economic Status relation with survival
# 
# 

# In[16]:


#Drop columns from training data set
ds_train=ds_train.drop(['Embarked','Name'],axis=1)
print("Columns Dropped Successfully")
ds_train.head()


# Let's check correlation cofficent between our features
# 

# In[17]:


# Training set high correlations
ds_train.corr()


# We can see correlation between 'Survived' and 'Fare' but other variables are not directly related with Survived but related to other variable
# 
# 📌 **Take Away Points**
# 
# * Age is correlated to Fare and Fare is correlated to Survived and our previous analysis also show how Age played role in survival
# * SibSp and Parch are realted to each other and also both related to Fare which make sense becuase more number of people means more fare, by virtue of this both can be related to Survived, we can further analysis on this in next section of Feature Engineering

# # Feature Engineering

# Let's start Feature Engineering with creating new variable **Family Size** by adding **SibSp** , **Parch** and **One**(Current Passenger)

# In[18]:


#Add new column 'Family Size' in training model set
ds_train['Family_Size'] = ds_train['SibSp'] + ds_train['Parch'] + 1
print("Family Size column created sucessfully")
ds_train.head()


# Now we will see how Family size will is realted with Survived variable

# In[19]:


#Visualize Family size and Survival
sns.barplot(x="Family_Size", y="Age", hue="Survived", data=ds_train,palette = 'rainbow')
plt.title('Family Size - Age Survival Distribution',size=20)
plt.show()


# In[20]:


sns.catplot(y="Family_Size", x="Survived", hue='Sex',kind="swarm", data=ds_train,height=8,aspect=1.5)
plt.title('Family Size - Gender Survival Distribution',size=20)
plt.show()


# 📌 **Take Away Points**
# 
# * Chances of survival are less for large Family (>5 memebers) 
# * If family size is small then main passenger gender decides on survival , this prove previous deduction that gender played major role in survival
# * Survival data is marked for main passenger and not for whole family, whereas family members name must be there in the list and they may or may not survived . In other words on just looking at survival column we can not dedeuce that fate of all family member were same

# ## Missing Values

# Before start with modeling let's check with missing values in training data set columns.
# 
# 

# In[21]:


print("Information on Train Data Set :")
ds_train.info()


# Only 'Age' is having missing values and we can replace missing values with median age , but putting median age for whole data set is not a good idea becuase passenger belongs to different Age group.
# 
# To overcome this we can calculate Median age based on 'Pclass' and 'Sex'
# 
# > **Note: I took some help from this [Kernel](https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial#1.-Exploratory-Data-Analysis) for this median value of Age**

# In[22]:


age_by_pclass_sex = ds_train.groupby(['Sex', 'Pclass']).median()['Age']

for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
print('Median age of all passengers: {}'.format(ds_train['Age'].median()))

# Filling the missing values in Age with the medians of Sex and Pclass groups
ds_train['Age'] = ds_train.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))


# Let's check the data set information one more time to verify missing values

# In[23]:


print("Information on Train Data Set :")
ds_train.info()


# Well , there is no missing values in our train data set but before we start our modeling let's encode 'Sex' column as this is the only column left as categorical variable.
# 
# As this column consist of only two values let's encode this with **1** for feamle and **0** for male , we can use hot encoder also but for starters let's avoid that as we have very simple column to encode

# In[24]:


#Replacing 'Male' and 'Female' with '0' and '1' respectively
ds_train=ds_train.replace(to_replace='male',value=0)
ds_train=ds_train.replace(to_replace='female',value=1)
ds_train.head()


# Now, every feature is in same scale let's start with Data Modeling and Prediction

# # Modeling and Prediction

# Let's start with moving target and feature variables

# In[25]:


X_train=ds_train.drop(['Survived'],axis=1)
y_train=ds_train['Survived'].values
print('X_train shape: {}'.format(X_train.shape))
print('y_train shape: {}'.format(y_train.shape))


# For this data set I will be using Random Forest Classifier , we can use other classifier models but for the sake of simplicity I will use only one model here

# In[26]:


classifier_rf=RandomForestClassifier(criterion='gini', 
                                           n_estimators=1100,
                                           max_depth=5,
                                           min_samples_split=4,
                                           min_samples_leaf=5,
                                           max_features='auto',
                                           oob_score=True,
                                           random_state=42,
                                           n_jobs=-1,
                                           verbose=1)
classifier_rf.fit(X_train,y_train)


# Let's try XGBoost classifier model also

# In[27]:


classifier_xgb=XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
classifier_xgb.fit(X_train,y_train)


# Before predicting from test set we need to clean test data set to make it equivalent with training data set i.e. need to drop unnecessary columns and encoded Sex column and missing values
# 
# Let's start with missing values

# In[28]:


ds_test.info()


# Here also Age is missing , let's fill in the similar way how we did it for training data

# In[29]:


age_by_pclass_sex = ds_test.groupby(['Sex', 'Pclass']).median()['Age']

for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
print('Median age of all passengers: {}'.format(ds_test['Age'].median()))

# Filling the missing values in Age with the medians of Sex and Pclass groups
ds_test['Age'] = ds_test.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))


# Let's check one more time for missing values

# In[30]:


ds_test.info()


# Now , only one value is missing from 'Fare' column which we can fill by median fare

# In[31]:


#Filling missing fare with median fare
null_index=ds_test['Fare'].isnull().index
medianFare=ds_test['Fare'].median()
ds_test.at[null_index,'Fare'] = medianFare
print("Missing Fare updated as Median Fare :{}".format(medianFare))


# Let's check one more time for missing values

# In[32]:


ds_test.info()


# In[33]:


#Drop columns from test data set
ds_test=ds_test.drop(['Ticket','Cabin','Embarked','Name'],axis=1)
print("Columns Dropped Successfully")

#Creating Family Size columns from test data set
ds_test['Family_Size'] = ds_test['SibSp'] + ds_test['Parch'] + 1
print("Family Size column created sucessfully")

#Encoding Gender column from test data set
ds_test=ds_test.replace(to_replace='male',value=0)
ds_test=ds_test.replace(to_replace='female',value=1)
X_test=ds_test

X_test.head()


# With this we are ready to get predicted values and submission file

# In[34]:


#Prediction test results
y_pred_rf=classifier_rf.predict(X_test)
y_pred_xgb=classifier_xgb.predict(X_test)

#Converting 2 dimensional  y_pred array into single dimension 
y_pred_rf=y_pred_rf.ravel()
y_pred_xgb=y_pred_xgb.ravel()

#Creating submission data frame and subsequent csv file for submission
submission_df_rf = pd.DataFrame(columns=['PassengerId', 'Survived'])
submission_df_rf['PassengerId'] = X_test['PassengerId'].astype(int)
submission_df_rf['Survived'] = y_pred_rf
submission_df_rf.to_csv('submissions_rf.csv', header=True, index=False)

submission_df_xgb = pd.DataFrame(columns=['PassengerId', 'Survived'])
submission_df_xgb['PassengerId'] = X_test['PassengerId'].astype(int)
submission_df_xgb['Survived'] = y_pred_xgb
submission_df_xgb.to_csv('submissions_xgb.csv', header=True, index=False)


# # Model Improvement 

# Above Models are giving good score(**0.779**) but this can be improved , let's try **K-Fold techniques** to check model accuracy

# In[35]:


#Apply K-fold in current model to check model accuracy
from sklearn.model_selection import cross_val_score
accuracies_rf = cross_val_score(estimator = classifier_rf, X = X_train, y = y_train, cv = 10)
accuracies_xgb = cross_val_score(estimator = classifier_xgb, X = X_train, y = y_train, cv = 10)


# In[36]:


#Checking accuracies for 10 fold in Random Forest and XG Boost Models
print("Accuracies for 10 Fold in Random Forest Model is {}".format(accuracies_rf))
print("Accuracies for 10 Fold in XG Boost Model is {}".format(accuracies_xgb))


# In[37]:


#Checking Mean and Standard Deviation between Accuracies
print("Mean Accuracy for Random Forest Model is {}".format(accuracies_rf.mean()))
print("Mean Accuracy for XG Boost Model is {}".format(accuracies_xgb.mean()))
print("Standard Deviation for Random Forest Model is {}".format(accuracies_rf.std()))
print("Standard Deviation for XG Boost Model is {}".format(accuracies_xgb.std()))


# With above data it's evident that Random Forest Accuracy and Standard Deviation is better than XG boost for training data.
# 
# To get best parameters for Random Forest let's do **Grid Search** for Random Forest model

# In[38]:


#Importing required library for Grid Search
from sklearn.model_selection import GridSearchCV
#Create the parameter grid based on the results of random search
param_grid = { 'bootstrap': [True],
              'max_depth': [80, 90, 100, 110],
              'max_features': [2, 3], 
              'min_samples_leaf': [3, 4, 5],
              'min_samples_split': [8, 10, 12],
              'n_estimators': [100, 300, 500, 1000] }
grid_search = GridSearchCV(estimator = classifier_rf, param_grid = param_grid,cv = 3, n_jobs = -1) 
grid_search = grid_search.fit(X_train, y_train)


# In[39]:


#Getting the best params
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy for Random Forest Classifier is {}".format(best_accuracy))
print("Best Parameters for Random Forest Classifier is {}".format(best_parameters))


# With above best parameters let's create one more classifier and predict from test data

# In[40]:


#Creating new classifier and fitting Training set
classifier_rf_new = RandomForestClassifier(n_estimators = 719,
                                           bootstrap=False,
                                           max_depth=464,
                                           max_features=0.3,
                                           min_samples_leaf=1,
                                           min_samples_split=2,
                                           random_state=42)
classifier_rf_new.fit(X_train, y_train)


# In[41]:


print("Predicting Results from new Classifier and Converting into Submission file")
# Predicting the Train set results
y_pred_rf_new=classifier_rf_new.predict(X_test)
#Converting 2 dimensional  y_pred array into single dimension 
y_pred_rf_new=y_pred_rf_new.ravel()
#Creating submission data frame and subsequent csv file for submission
submission_df_rf_new = pd.DataFrame(columns=['PassengerId', 'Survived'])
submission_df_rf_new['PassengerId'] = X_test['PassengerId'].astype(int)
submission_df_rf_new['Survived'] = y_pred_rf_new
submission_df_rf_new.to_csv('submissions_rf_new.csv', header=True, index=False)
print("Created Submission file from new classifier successfully")


# Submission with this is not increasing our score from previous Random Classifier Model , there can be several reason behind this but I found [this](https://towardsdatascience.com/optimizing-hyperparameters-in-random-forest-classification-ec7741f9d3f6) explanation very useful

# > **If you like the work please upvote and do leave a comment for any feedback**
