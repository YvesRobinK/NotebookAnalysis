#!/usr/bin/env python
# coding: utf-8

# # Simple Breakdown of Titanic Dataset- EDA,Comparisons and Predictions
# 
# ![](https://drive.google.com/uc?export=download&id=13no8f5E2ffXcBHqftiFrPls6gbQVnV50)

# <a></a>
# 
# # Introduction
# Welcome aboard my Titanic Kernel. In this Kernel I'll be using the 'OG' [Titanic Dataset](https://www.kaggle.com/c/titanic/data) to perform-
# - Starter Exploratory Data Analysis using Distribution Plots,Box Plots and Count Plots.
# - DataCleaning/Data Analysis and Feature Engineering
# - Trainging our Dataset on:
# > 1. XGBoost
# > 2. Random Forest
# > 3. LightGBM
# > 4. CatBOOST
# > 5. AdaBoost
# > 6. Logistic Regression
# - Evaluating and Comparing the Predictions
# 
# # UPDATE!!!!
# 
# **After some thorough scribbling I was able to improve my submission score from** 0.78468 to 0.78947.**I was able to improve my feature engineering section by better handling of Age and Fare column Features as well as introducing a new Feature column 'TravelAlone'.**

# # Imports
# 
# 
# > 1. Let's get our environment ready with the libraries we'll need and then import the relevant ones beforehand!
# > 2. Pandas is one of the most widely used python libraries in data science. It provides high-performance, easy to use structures and data analysis tools.
# > 3. Matplotlib is a plotting library for the Python programming language
# > 4. Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

# In[129]:


# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

#core imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Fetching the Data
# > Using Pandas to load the dataset into this notebook. Using pandas we can read our datafile train.csv with the line below. Data-set loaded will be assigned to the respective variable.

# In[130]:


#load training dataset and assign it to a variable
train=pd.read_csv("../input/titanic/train.csv")
test=pd.read_csv("../input/titanic/test.csv")


# ## Check out the Data

# In[131]:


#use the 'head' method to show the first five rows of the table as well as their names. 
train.head()


# In[132]:


# shape of the Titanic train dataframe
train.shape


# In[133]:


train.info()


# > Above is a concise summary of our dataframe returning columns' data-type,index data-type and number of non-null values !

# ## Exploratory Data Analysis(EDA)
# > Let's create some simple plots to analyze and identify patterns in our data.

# > > # 1. Continuous Features
# > > > ## a) Age
# 

# In[134]:


print('Avgerage. Age : ',train['Age'].mean())


# In[135]:


plt.figure(figsize=(8,4))
fig = sns.distplot(train['Age'], color="darkorange")
fig.set_xlabel("Age",size=15)
fig.set_ylabel("Density of Passengers",size=15)
plt.title('Passenger Age Distribution',size = 20)
plt.show()


# ***We can observe that the probability for the age to be between 20-30 was high for a passenger onboard the Titanic***

# > > > ## b) Fare

# In[136]:


print('Avgerage. Fare : ',train['Fare'].mean())


# In[137]:


plt.figure(figsize=(8,4))
fig = sns.distplot(train['Fare'], color="darkorange")
fig.set_xlabel("Fare",size=15)
fig.set_ylabel("Density of Passengers",size=15)
plt.title('Titanic Fare Distribution',size = 20)
plt.show()


# ***Clearly, the probability that a passenger was travelling with a cheaper ticket was quite high on Titanic***

# In[138]:


plt.figure(figsize=(8,4))
fig = sns.distplot(train[train['Fare']<=150]['Fare'], color="darkorange")
fig.set_xlabel("Fare",size=15)
fig.set_ylabel("Density of Passengers",size=15)
plt.title('Titanic Fare(Scaled) Distribution',size = 20)
plt.show()


# For Fare price less than $25 the probability of finding a buyer was predominantly ***high***

# > > # 2. Continuous vs Categorical
# 

# In[139]:


#WE PLOT THE PASSENGER AGE DISTRIBUTION VS PASSENGER CLASS ON TITANIC
plt.figure(figsize=(8,4))
fig=sns.boxplot(train['Pclass'],train['Age'],palette='Blues')
fig.set_xlabel("Passenger Class",size=15)
fig.set_ylabel("Age of Passenger",size=15)
plt.title('Age Distribution/Pclass',size = 20)
plt.show()


# PClass 1 was predominantly occupied by ***older citizens***

# In[140]:


#WE PLOT THE PASSENGER AGE DISTRIBUTION VS PASSENGER Gender ON TITANIC
plt.figure(figsize=(8,4))
fig=sns.boxplot(train['Sex'],train['Age'],palette='Blues')
fig.set_xlabel("Gender",size=15)
fig.set_ylabel("Age of Passenger",size=15)
plt.title('Age Distribution/Gender',size = 20)
plt.show()


# The ***Male presence*** OnBoard Titanic was considerably ***older*** than the ***female***.

# In[141]:


#WE PLOT THE PASSENGER AGE DISTRIBUTION VS SURVIVED ON TITANIC
plt.figure(figsize=(8,4))
fig=sns.boxplot(train['Survived'],train['Age'],palette='Blues',labels=["No"])
fig.set_xlabel("Survived",size=15)
fig.set_ylabel("Age of Passenger",size=15)
plt.title('Age Distribution/Survived',size = 20)
fig.set(xticklabels=["0-No","1-Yes"])
plt.show()


# Youth Presence onBoard was largely able to ***survive the disaster***

# In[142]:


#WE PLOT THE FARE DISTRIBUTION VS PASSENGER CLASS ON TITANIC
plt.figure(figsize=(8,4))
fig=sns.boxplot(train['Pclass'],train['Fare'],palette='Reds')
fig.set_xlabel("Passenger Class",size=15)
fig.set_ylabel("Fare",size=15)
plt.title('Fare Distribution/Pclass',size = 20)
plt.show()


# We narrow our observation range for Fare i.e(fare<=300) to get a better visualization of the distribution.

# In[143]:


#WE PLOT THE FARE DISTRIBUTION VS PASSENGER CLASS ON TITANIC
plt.figure(figsize=(8,4))
fig=sns.boxplot(train['Pclass'],train[train['Fare']<=300]['Fare'],palette='Reds')
fig.set_xlabel("Passenger Class",size=15)
fig.set_ylabel("Fare",size=15)
plt.title('Fare(Scaled) Distribution/Pclass',size = 20)
plt.show()


# On Titanic, passenger class '3' was the ***most affordable*** and cheap with class '1' being the ***most expensive***

# In[144]:


#WE PLOT THE FARE DISTRIBUTION VS PASSENGER CLASS ON TITANIC
plt.figure(figsize=(8,4))
fig=sns.boxplot(train['Survived'],train[train['Fare']<=300]['Fare'],palette='Reds')
fig.set_xlabel("Survived",size=15)
fig.set_ylabel("Fare",size=15)
fig.set(xticklabels=["0-No","1-Yes"])
plt.title('Fare(Scaled) Distribution/Survived',size = 20)
plt.show()


# Passengers with ***high-priced tickets***, mostly ended up ***SURVIVING***

# In[145]:


plt.figure(figsize=(8,4))
fig=sns.violinplot(train["Age"],train["Sex"], hue=train["Survived"],split=True,palette='Reds')
fig.set_ylabel("Sex",size=15)
fig.set_xlabel("Age",size=15)
plt.title('Age and Sex vs Survived',size = 20)
plt.show()


# ***Young Passengers(10<age<35) seems to have a good survival rate irrespective of the gender.***

# In[146]:


plt.figure(figsize=(8,4))
fig=sns.violinplot(train["Pclass"],train['Age'], hue=train["Survived"],split=True,palette='Blues')
fig.set_xlabel("Pclass",size=15)
fig.set_ylabel("Age",size=15)
plt.title('Age and Pclass vs Survived',size = 20)
plt.show()


# 1. for Pclass 1 and 2 survival rate is generally high for **aged 15-40***
# 2. for Pclass 3 we see a high survival rate amongst ***children i.e  0<age<10***

# > # 3. Categorical Features

# In[147]:


bg_color = (0.25, 0.25, 0.25)
sns.set(rc={"font.style":"normal",
            "axes.facecolor":bg_color,
            "figure.facecolor":bg_color,"text.color":"white",
            "xtick.color":"white",
            "ytick.color":"white",
            "axes.labelcolor":"white"})
plt.figure(figsize=(8,4))
fig=sns.countplot(train['Survived'],hue=train['Pclass'],palette='Blues',saturation=0.8)
fig.set_xlabel("Survived",size=15)
fig.set_ylabel("#",size=15)
fig.set(xticklabels=["0-No","1-Yes"])
plt.title('# of Survived/PClass',size = 20)
plt.show()


# 1. ***Passengers from class 3(cheapest Fare) suffered the most!***
# 2. ***Pclass 1 reported the highest survival count*** 

# In[148]:


plt.figure(figsize=(8,4))
fig=sns.countplot(train['Survived'],hue=train['Sex'],palette='Oranges',saturation=0.8)
fig.set_xlabel("Survived",size=15)
fig.set_ylabel("#",size=15)
fig.set(xticklabels=["0-No","1-Yes"])
plt.title('# of Survived/Sex',size = 20)
plt.show()


# 1. ***Female presence*** largely survived the disaster.
# 2. ***Highest Casualty*** was reported from the ***Male side***.

# In[149]:


plt.figure(figsize=(8,4))
fig=sns.countplot(train['Survived'],hue=train['SibSp']>0,palette='Blues',saturation=0.8)
fig.set_xlabel("Survived",size=15)
fig.set_ylabel("#",size=15)
fig.set(xticklabels=["0-No","1-Yes"])
plt.title('# of Survived per Siblings/Spouses Onboard',size = 20)
plt.show()


# 1. Passengers without Siblings/Spouse onboard ***mostly survived***
# 2. Interesting observation was the passengers travelling without a siblings/spouse onboard mostly ended up on the casualty side

# In[150]:


plt.figure(figsize=(8,4))
fig=sns.countplot(train['Survived'],hue=train['Parch']>0,palette='Oranges',saturation=0.8)
fig.set_xlabel("Survived",size=15)
fig.set_ylabel("#",size=15)
fig.set(xticklabels=["0-No","1-Yes"])
plt.title('# of Survived per Parents/Children Onboard',size = 20)
plt.show()


# In[151]:


sns.set(rc={"font.style":"normal",
            "axes.facecolor":"white",
            "figure.facecolor":"white","text.color":"black",
            "xtick.color":"black",
            "ytick.color":"black",
            "axes.labelcolor":"black"})
plt.figure(figsize=(8,4))
fig=sns.countplot(y=train['Pclass'],hue=train['SibSp']>0,palette='Blues',saturation=1.0)
fig.set_xlabel("#",size=15)
fig.set_ylabel("Passenger Class",size=15)
plt.title('# of Pclass per Siblings/Spouses Onboard',size = 20)
plt.show()


# ***Passenger Class 1,2 and 3*** mostly had passengers ***accompanied neither by their Spouses nor Siblings***.

# In[152]:


plt.figure(figsize=(8,4))
fig=sns.countplot(y=train['Pclass'],hue=train['Parch']>0,palette='Reds',saturation=0.6)
fig.set_ylabel("Pclass",size=15)
fig.set_xlabel("#",size=15)
plt.title('# of Pclass per Parents/Children Onboard',size = 20)
plt.show()


# ## Introducing 'TravelAlone' Feature
# 
# - 'True': Travelling Alone
# - 'False': Has Company(Either Sibling/Spouse/Parents/Children)

# In[153]:


def func(x):
    x1=x[0]
    x2=x[1]
    if x1>0 or x2>0:
        return False
    else:
        return True
train['TravelAlone']=train[['SibSp','Parch']].apply(func,axis=1)
test['TravelAlone']=test[['SibSp','Parch']].apply(func,axis=1)

plt.figure(figsize=(8,4))
fig=sns.countplot(y=train['Survived'],hue=train['TravelAlone'],palette='RdBu',saturation=2.0)
fig.set_ylabel("Survived",size=15)
fig.set_xlabel("#",size=15)
plt.title('# of Survived/Onboard Alone',size = 20)
plt.show()


# 1. Passengers ***travelling alone*** mostly ***lost*** their lives in the disaster
# 2. Passengers accompanied by their Siblings/Spouses/Parents/Children ***mostly survived***.

# In[154]:


plt.figure(figsize=(8,4))
fig=sns.countplot(y=train['Pclass'],hue=train['TravelAlone'],palette='Reds',saturation=0.8)
fig.set_ylabel("Passenger Class",size=15)
fig.set_xlabel("#",size=15)
plt.title('# of Pclass/Onboard Alone',size = 20)
plt.show()


# ***Passenger Class 1,2 and 3*** on Titanic predominantly had passengers ***travelling alone***.

# # Data Cleaning
# 

# In[155]:


#Checking the status of null values inside our training Dataframe
train.isnull().sum()


# In[156]:


#Checking the status of null values inside our testing Dataframe
test.isnull().sum()


# - **Cabin** feature-column contains **NULL values**
# - We will drop **'PassengerId'** feature since it consists of unique ID having no influence on the model training result.
# - We will drop **'Cabin','Ticket' and 'Name'** Feature Columns as we are not using any NLP to process the information for model training from non numerical columns.

# In[157]:


train.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)
#train.head()

#same processing for test dataset
test.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)


# - 'TravelAlone' Feature column was introduced to combine the information of 'SibSp' and 'Parch' feature columns
# - SibSp and Parch can now be dropped from our test/train datasets

# In[158]:


train.drop(columns=['SibSp','Parch'],inplace=True)
#train.head()

#same processing for test dataset
test.drop(columns=['SibSp','Parch'],inplace=True)


# # Feature Engineering
# - We will use **one-hot encoding technique** for our **Categorical Features-'Sex','Embarked' and 'TravelAlone'** to use them in model training for better predictions(Converts Categorical data to Integer Data(0 or 1)).
# - We will get rid of **null values** inside the dataframe by filling them with an appropriate replacement,also taking care of **outliers**.

# 
# - We get rid of **Null Values** inside the Age Feature Column
# - We replace every Null Value by the median Age value for every Pclass passenger.
# - The median Age is Calculcated per Passenger Class

# In[159]:


#median age for Pclass 1
a=train.groupby('Pclass').median()['Age'].iloc[0]
#median age for Pclass 2
b=train.groupby('Pclass').median()['Age'].iloc[1] 
#median age for Pclass 3
c=train.groupby('Pclass').median()['Age'].iloc[2] 
def fillAge_train(x):
    age=x[0]
    pclass=x[1]
    if pd.isnull(age):
        if pclass==1:
            return a
        elif pclass==2:
            return b
        else:
            return c
    else:
        return age
#median age for Pclass 1
a_test=test.groupby('Pclass').median()['Age'].iloc[0]
#median age for Pclass 2
b_test=test.groupby('Pclass').median()['Age'].iloc[1] 
#median age for Pclass 3
c_test=test.groupby('Pclass').median()['Age'].iloc[2]
def fillAge_test(x):
    age=x[0]
    pclass=x[1]
    if pd.isnull(age):
        if pclass==1:
            return a_test
        elif pclass==2:
            return b_test
        else:
            return c_test
    else:
        return age
    
#replacing null Age values in training dataset
train['Age']=train[['Age','Pclass']].apply(fillAge_train,axis=1)
#replacing null Age values in Test Dataset
test['Age']=test[['Age','Pclass']].apply(fillAge_test,axis=1)
     


# - We now get rid of Null Fare values in our test dataset.
# - We replace null with Median Fare value for every Pclass.

# In[160]:


a=test.groupby('Pclass').median()['Fare'].iloc[0]
b=test.groupby('Pclass').median()['Fare'].iloc[1]
c=test.groupby('Pclass').median()['Fare'].iloc[2]
def fillFare(x):
    fare=x[0]
    pclass=x[1]
    if pd.isnull(fare):
        if pclass==1:
            return a
        elif pclass==2:
            return b
        else:
            return c
    else:
        return fare

#replace Fare value using built in Pandas Functions
test['Fare']=test[['Fare','Pclass']].apply(fillFare,axis=1)


# - We now get rid of **Null Values** inside **Embarked** Feature Column.
# - We replace the Null value with the **Mode value of the column.**
# - We fill Mode since it is a non-numerical Column where median is not applicable.

# In[161]:


#fill null values for feature column- Embarked using Pandas built-in function
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)


# In[162]:


#Convert Categorical Column to Integer Column using One-hot Encoding
train=pd.get_dummies(train,columns=['Sex','Embarked','TravelAlone'],drop_first=True)
#train.head()

#Same conversion for test dataset
test=pd.get_dummies(test,columns=['Sex','Embarked','TravelAlone'],drop_first=True)
#test.head()


# ## Handling Age and Fare Continuous Data Columns
# 
# - We divide data from 'Age Column' in 5 bands using Pandas' built in cut()/qcut() method and map the categories to either of these values- 0/1/2/3/4

# In[163]:


train['AgeCateg'] = pd.cut(train['Age'], 5)
train[['AgeCateg', 'Survived']].groupby(['AgeCateg'], as_index=False).mean().sort_values(by='AgeCateg', ascending=True)

train['FareCateg'] = pd.qcut(train['Fare'], 4)
train[['FareCateg', 'Survived']].groupby(['FareCateg'], as_index=False).mean().sort_values(by='FareCateg', ascending=True)

#func to process values
def encodeAgeFare(train):
    train.loc[train['Age'] <= 16, 'Age'] = 0
    train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1
    train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2
    train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3
    train.loc[ (train['Age'] > 48) & (train['Age'] <= 80), 'Age'] = 4
    
    train.loc[train['Fare'] <= 7.91, 'Fare'] = 0
    train.loc[(train['Fare'] > 7.91) & (train['Fare'] <= 14.454), 'Fare'] = 1
    train.loc[(train['Fare'] > 14.454) & (train['Fare'] <= 31.0), 'Fare'] = 2
    train.loc[(train['Fare'] > 31.0) & (train['Fare'] <= 512.329), 'Fare'] = 3
    
encodeAgeFare(train)
encodeAgeFare(test)

#dropping AgeCateg and FareCateg columns
train.drop(columns=['AgeCateg','FareCateg'],inplace=True)
    


# In[164]:


train.head()


# In[165]:


test.head()


# # Train/Test Data Split

# In[170]:


# We Seperate and assign our Target Variable Column to a new variable
X = train.drop('Survived',axis=1)
# Dropped 'SibSp' and 'Parch' from Input Feature Data because we have Alone_True Column

y = train['Survived']


# In[171]:


# We're splitting up our data set into groups called 'train' and 'test'
from sklearn.model_selection import train_test_split

np.random.seed(0)
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3)


# In[172]:


#Core Imports for Model Training
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV


# # Tuning/Training Models
# 
# > ## 1. XGBoost

# In[173]:


#initialize our object
xgbclassifier=XGBClassifier()

#fit train data
xgbclassifier.fit(X_train,y_train)

#predictions
pred_xgb=xgbclassifier.predict(X_test)

# print(accuracy_score(y_test,pred_xgb))


# > ## 2. RandomForest

# In[174]:


randomfc = RandomForestClassifier(n_estimators=100)

#fit train data
randomfc.fit(X_train,y_train)

#predictions
pred_rf=randomfc.predict(X_test)

# print(accuracy_score(y_test,pred_rf))


# > ## 3. LighGBM

# In[175]:


lightgb=LGBMClassifier()

#fit train data
lightgb.fit(X_train,y_train)

#predictions
pred_lgb=lightgb.predict(X_test)

# print(accuracy_score(y_test,pred_lgb))


# > ## 4. AdaBOOST

# In[176]:


ada=AdaBoostClassifier(n_estimators=50,learning_rate=1)

#fit train data
ada.fit(X_train,y_train)

#predictions
pred_ada=ada.predict(X_test)

# print(accuracy_score(y_test,pred_ada))


# > ## 5. CatBoost

# In[177]:


cbc=CatBoostClassifier(verbose=0, n_estimators=100)

#fit train data
cbc.fit(X_train,y_train)

#predictions
pred_cbc=cbc.predict(X_test)

# print(accuracy_score(y_test,pred_cbc))


# > ## 6. Logistic Regression

# In[178]:


log=LogisticRegression(max_iter=1000)

#fit train data
log.fit(X_train,y_train)

#predictions
pred_log=log.predict(X_test)

# print(accuracy_score(y_test,pred_log))


# # Evaluations/Comparisons

# ## 1. Confidence Score

# In[179]:


print('XGBoost:', round(xgbclassifier.score(X_train, y_train) * 100, 2), '%.\t\t\t RandomForest:', round(randomfc.score(X_train, y_train) * 100, 2), '%.')
print('LightGBM:', round(lightgb.score(X_train, y_train) * 100, 2), '%.\t\t\t AdaBoost:', round(ada.score(X_train, y_train) * 100, 2), '%.')
print('CatBoost:', round(cbc.score(X_train, y_train) * 100, 2), '%.\t\t\t LogisticRegression:', round(log.score(X_train, y_train) * 100, 2), '%.')


# **Observation**
# - ***Classification confidence scores are designed to measure the accuracy of the model when predicting class assignment.***
# - Random Forest,XgBoost and LightGBM nearly top the chart with a very high Confidence Score

# ## 2. Accuracy Score

# In[180]:


print('XGBoost:', round(accuracy_score(y_test,pred_xgb) * 100, 2), '%.\t\t\t RandomForest:', round(accuracy_score(y_test,pred_rf) * 100, 2), '%.')
print('LightGBM:', round(accuracy_score(y_test,pred_lgb) * 100, 2), '%.\t\t\t AdaBoost:', round(accuracy_score(y_test,pred_ada) * 100, 2), '%.')
print('CatBoost:', round(accuracy_score(y_test,pred_cbc) * 100, 2), '%.\t\t\t LogisticRegression:', round(accuracy_score(y_test,pred_log) * 100, 2), '%.')


# ## 3. Confusion Matrix

# In[181]:


fig, axs = plt.subplots(3, 2,figsize=(20,12))

sns.heatmap(confusion_matrix(y_test,pred_xgb), annot=True,ax=axs[0][0])
axs[0, 0].set_title('XGBoost')

sns.heatmap(confusion_matrix(y_test,pred_rf), annot=True,ax=axs[0][1])
axs[0, 1].set_title('RandomForest')

sns.heatmap(confusion_matrix(y_test,pred_lgb), annot=True,ax=axs[1][0])
axs[1, 0].set_title('LightGBM')

sns.heatmap(confusion_matrix(y_test,pred_ada), annot=True,ax=axs[1][1])
axs[1, 1].set_title('XgBoost')

sns.heatmap(confusion_matrix(y_test,pred_cbc), annot=True,ax=axs[2][0])
axs[2, 0].set_title('CatBoost')

sns.heatmap(confusion_matrix(y_test,pred_log), annot=True,ax=axs[2][1])
axs[2, 1].set_title('Logistic Regression')

plt.show()


# ## 4. Hyperparameter Tuning

# - Tuned Random Forest model was used for submission to achieve a score of top14%.
# - We will now use RandomizedSearchCV to tune our **RandomForest model**

# # Random Forest Tuned- Model used for Submission- 0.78947

# In[167]:


#assigning X_train,X_test and y_train vatiables
X_train=train.drop('Survived',axis=1)
X_test=test.copy()
y_train = train['Survived']

#declaring and inititalizing params attribute
from scipy.stats import uniform, truncnorm, randint
model_params = {
    # randomly sample numbers from 4 to 204 estimators
    'n_estimators': randint(4,200),
    # normally distributed max_features, with mean .25 stddev 0.1, bounded between 0 and 1
    'max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1),
    # uniform distribution from 0.01 to 0.2 (0.01 + 0.199)
    'min_samples_split': uniform(0.01, 0.199)
}

rf_model = RandomForestClassifier()

# set up random search meta-estimator
# this will train 100 models over 5 folds of cross validation (500 models total)
clf = RandomizedSearchCV(rf_model, model_params, n_iter=100, cv=5, random_state=1)

# train the random search meta-estimator to find the best model out of 100 candidates
model = clf.fit(X_train, y_train)

# print winning set of hyperparameters
from pprint import pprint
pprint(model.best_estimator_.get_params())


# In[168]:


testData=test.copy()
predictions_rf=model.predict(testData)


# In[169]:


subm=pd.read_csv('../input/titanic/gender_submission.csv')
Survived=pd.Series(predictions_rf)
subm.drop(columns=['Survived'],inplace=True)
subm['Survived']=pd.Series(predictions_rf)
subm.to_csv(r'RandomForestSubmission.csv', index = False)
print("Submitted")


# ## Thank You!
# 
# - Any suggestions for improvements in the score and accuracy are most welcome.
# - Do mention the irregularities found in the comments below.
