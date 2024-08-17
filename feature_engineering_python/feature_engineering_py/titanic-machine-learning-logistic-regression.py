#!/usr/bin/env python
# coding: utf-8

# # **Introduction**

# **In this notebook I've been dealing with taitanic data while learning data analysis.**

# This Dataset contains the information about Titanic ship and included 891Rows and 12 columns.In this notebook I try to find the best ML model to predict which passengers survived the Titanic shipwreck.    Variable Notes:
# 
# pclass: A proxy for socio-economic status (SES)(Ticket class) 1 = 1st, 2 = 2nd, 3 = 3rd
# 
# 1st = Upper
# 
# 2nd = Middle
# 
# 3rd = Lower
# 
# age:Age in years
# 
# Sibsp: The dataset defines family relations in this way...(number of siblings / spouses aboard the Titanic)
# 
# Sibling = brother, sister, stepbrother, stepsister
# 
# Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# Parch: The dataset defines family relations in this way...(number of parents / children aboard the Titanic)
# 
# Parent = mother, father
# 
# Child = daughter, son, stepdaughter, stepson
# 
# Some children travelled only with a nanny, therefore parch=0 for them.
# 
# Survival:0 = No, 1 = Yes
# 
# Sex:Sex
# 
# Ticket:Ticket number
# 
# Fare:Passenger fare
# 
# Cabin:Cabin number
# 
# Embarked:Port of Embarkation(C = Cherbourg, Q = Queenstown, S = Southampton)

# In[1]:


import numpy as np#import all necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train=pd.read_csv("../input/c/titanic/train.csv")#import the dataset
test=pd.read_csv("../input/c/titanic/train.csv")


# In[3]:


#store the passengerid of test data
passenger_id=test["PassengerId"]

train.head()


# In[4]:


#set the index as passengerid
train.set_index(["PassengerId"],inplace=True)
test.set_index(["PassengerId"],inplace=True)
train.head()


# # **Checking the missing data**:
# ### The real-world data often has a lot of missing values. The cause of missing values can be data corruption or failure to record data. The handling of missing data is very important during the preprocessing of the dataset as many machine learning algorithms do not support missing values. Now I want to identify the rows with the most number of missing values and drop or transform them.

# In[5]:


train.isnull().sum()#show the missing datas


# In[6]:


100*(train.isnull().sum()/len(train))
def missing_values_percent(train):#we can use this function in all dataframes.
    nan_percent=100*(train.isnull().sum()/len(train))
    nan_percent=nan_percent[nan_percent>0].sort_values()
    return(nan_percent)

nan_percent=missing_values_percent(train)
nan_percent


# In[7]:


#deciding about age coloumn which have almost 19% missing datas.
#Imputer age coloumn
from sklearn.impute import SimpleImputer
#train data             
Imp=SimpleImputer(strategy='median')
new_train=Imp.fit_transform(train.Age.values.reshape(-1,1))
train['Age2'] = new_train

#test data
new_test=Imp.fit_transform(test.Age.values.reshape(-1,1))
test['Age2'] = new_test


train.drop('Age',axis=1,inplace=True)
test.drop('Age',axis=1,inplace=True)


train.head()


# In[8]:


train.isnull().sum()


# In[9]:


train["Embarked"].value_counts()


# In[10]:


#So we can replace missing datas in Embarked with s
train["Embarked"].fillna("s",inplace=True)


# In[11]:


#cabin has 687 missing datas so we can get rid of it by dropping this feature.
train.drop("Cabin",axis=1,inplace=True)


# In[12]:


train.isnull().sum()


# ### **Now we have no missing data**

# # **Exploratory Data Analysis**

# In[13]:


train["Survived"].value_counts(normalize=True)#How many passengers survived?


# In[14]:


def bar_chart_stacked(dataset,feature,stacked=True):
  survived=train[train["Survived"]==1][feature].value_counts()
  dead=train[train["Survived"]==0][feature].value_counts()   
  df_survived_dead=pd.DataFrame([survived,dead])  
  df_survived_dead.index=["passengers survived","passengers died"]   
  df_survived_dead.plot(kind="bar",stacked=stacked,figsize=(8,5))
                              


# In[15]:


bar_chart_stacked(train,"Survived")


# ### As expected the majority of passengers in the training data died. Only 38% survived the disaster. So the training data suffers from data imbalance.

# In[16]:


train["Sex"].value_counts().to_frame()#passengers count on gender


# In[17]:


bar_chart_stacked(train,"Sex")#compare the survived  and dead passengers counts on gender


# ### We can see that even though the majority of passenger were men, the majority of survivors were women. The key observation here is that the survival rate for female passengers is 4 times higher than the survival rate of male passengers.Maybe the women were rescued earlier or the man helped the women and they didnt have enough time to save themselves. 

# In[18]:


train.groupby(["Pclass"])["Survived"].mean().to_frame()


# In[19]:


bar_chart_stacked(train,"Pclass")


# ### We see that 62% of passengers in class 1 were survived but this amount is reduced to 47% for class 2 and only 24% of passengers in class 3 were survived.On the other words the percentage of survived passengers in class 1 is 2 times bigger than the percentage of died passengers in this class.But in class 2 the percentage of survived people and died people is almost equal and for class 3 the percentage of died people is three times bigger than survived one.

# In[20]:


def bar_chart_compare(dataset,feature1,feature2=None):
    plt.figure(figsize=(8,5))
    plt.title("survived rate by sex and pclass")
    g=sns.barplot(x=feature1,y="Survived",hue=feature2,data=dataset).set 


# In[21]:


bar_chart_compare(train,"Pclass","Sex")


# ### We see that the number of men and women who were survived is decreasing according to class.In addition,men and women in class 1 had a significantly higher chance of survival if they bought class 1 tickets.

# In[22]:


sns.scatterplot(data=train,x="Age2",y="Survived")


# ### As we see the passangers who were 1 to almost 57 years old,were survived but the paasengers who were above 60 years old mostly died.

# In[23]:


plt.figure(figsize=(8,5))
plt.title("survival swarmplot for fare and gender")
sns.swarmplot(y="Fare",x="Sex",hue="Survived",data=train)


# ### Gender of all passengers with a fare above 500 dollar survived.The men who paid between 200 and 300 dollar died but the women paid between 200 and 300 dollar survived.

# In[24]:


sns.pairplot(train)


# In[25]:


corr = train.corr()
plt.subplots(figsize=(12, 8))
sns.heatmap(corr, annot=True)


# # **Feature Engineering**

# ## **Name feature**

# In[26]:


train["Name"]


# In[27]:


train['Title'] =train['Name'].apply(lambda x: x.split(', ')[1].split('. ')[0].strip())
test['Title'] =train['Name'].apply(lambda x: x.split(', ')[1].split('. ')[0].strip())
train["Title"].unique()


# In[28]:


plt.figure(figsize=(18,3))
sns.barplot(data=train,x="Title",y="Survived")


# ### We see the titles which are women title have a high survival rate in comparison with men title.Master and Dr have a high survival rate, too even though both are male titles.but Mr have a low survival rate.

# # **Family feature**

# In[29]:


train.info()


# In[30]:


train["family_size"]=train["SibSp"]+train["Parch"]+1


# In[31]:


def family_group(size):
    a=""
    if(size<=1):
        a="alone"
    elif(size<=4):
        a="small"
    else:
        a="large"
    return a


# In[32]:


train["family_group"]=train["family_size"].map(family_group)
train["fare_per_person"]=train["Fare"]/train["family_size"]
train.head()


# In[33]:


plt.figure(figsize=(8,5))
sns.barplot(data=train,x="family_group",y="Survived")


# ### As we see small family survived more in comparison with alone and large family.on the other words the percentage of survived passenger for small family is three times bigger than large family and two times bigger than alone passenger.

# In[34]:


train.drop(["Ticket"],axis=1,inplace=True)#This feature doesnt give us any useful infomation.


# # **Encoding str to int**

# In[35]:


train["Pclass"].apply(str)


# In[36]:


#Divide dataframe to 2 parts(num and str)
train_num=train.select_dtypes(exclude="object")
train_obj=train.select_dtypes(include="object")


# In[37]:


train_obj=pd.get_dummies(train_obj,drop_first=True)#use one-hot encoding to transform str to int and float
train_obj.shape


# In[38]:


Final_train=pd.concat([train_num,train_obj],axis=1)
Final_train.info()


# # **Logestic regression**

# In[39]:


#Determine the feature and lable
X=Final_train.drop("Survived",axis=1)
y=Final_train["Survived"]


# In[40]:


#Split the dataset to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[41]:


from sklearn.preprocessing import StandardScaler#scaling the features
scaler= StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[42]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)


# In[43]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix


# In[44]:


accuracy_score(y_test, y_pred)


# In[45]:


confusion_matrix(y_test, y_pred)


# In[46]:


print(classification_report(y_test, y_pred))

