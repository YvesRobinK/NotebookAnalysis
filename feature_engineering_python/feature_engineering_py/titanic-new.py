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


# In[ ]:





# My notebook
# -----
# The scope of this work is to involve as lot as possible of people in a space to talk about modeling and predictions with  the kaggle competitions and if possible creating connections and a network of people with the same passion.
# 
# I think that exchange of ideas can help us in our process of learning and growth as data scientist and kaggle is a place in which you have real opportunities to use your skills and make practice.
# So feel free to comment if you find some insights in this notebook.
# 
# In the first part the focus is more oriented to the exploratory data analysis with attention to 
# - Missing values
# - outliers
# - Feature Engineering
# - Data Cleaning
# 
# In this first part after using some tecnics to manipulate our dataset and finding some insights about the most significatn variables i will submit my first prediction using a random forest model that will perform with a 77% score. ThaYou can consider this model as the baseline.
# 
# Thatn our focus will be more about the models and to see the various metrics of evaluation
# 
# Lastly we will  try some tecnic to improve our predictions with the Ensemble models.
# 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Reading the data

# In[3]:


train=pd.read_csv("/kaggle/input/titanic/train.csv")
train


# In[4]:


test=pd.read_csv("/kaggle/input/titanic/test.csv")
test


# EDA

# In[5]:


#let's have information about datasets
train.info()
test.info()


# In[6]:


#Distribution of null values for training data
print(train.isnull().sum())
train.isnull().mean()


# In[7]:


#Distribution of null values for test data
print(test.isnull().sum())
test.isnull().mean()


# In[8]:


train.nunique()#@unique values for training data


# In[9]:


test.nunique()#unique values for testing data


# Analizing Categorical data

# Age

# In[10]:


train.Sex.value_counts()


# In[11]:


train.groupby("Sex")["Survived"].mean()#it seems gender is going to be  very interesting for our predictions


# In[12]:


pd.crosstab(train["Sex"],train["Survived"])


# Pclass

# In[13]:


train.Pclass.value_counts()# the distribution for classes of trip


# In[14]:


train.groupby("Pclass")["Survived"].mean()#people in 1st class had more chances to survive


# In[15]:


pd.crosstab(train["Pclass"],train["Survived"])


# SibSp

# In[16]:


train.SibSp.value_counts()#most of people were alone or withouth siblings or spouses


# In[17]:


train.groupby("SibSp")["Survived"].mean()#People with 1 or 2 siblings had more chances to survive


# In[18]:


pd.crosstab(train["SibSp"],train["Survived"])


# Parch

# In[19]:


train.Parch.value_counts()


# In[20]:


pd.crosstab(train["Parch"],train["Survived"])#Similar to Sibsp. We have the maximum of chance to survive for a low value of Parch


# In[21]:


pd.crosstab(train["Parch"],train["Survived"])


# Embarked

# In[22]:


train.Embarked.value_counts()


# In[23]:


pd.crosstab(train["Embarked"],train["Survived"])#Values for Embarked in Cherbourg seems better. Let's look at the correlation with Pclass


# In[24]:


pd.crosstab(train["Embarked"],train["Pclass"])#As we can see the most of Embarked in Cherbourg were in 1st class. They are not equally distributed for this variable


# In[25]:


train.groupby("Embarked")["Survived"].mean()


# Quantitative Variable

# In[26]:


train.describe()#Some variable are more skewed, like Fare


# In[27]:


test.describe()


# In[28]:


train.corr()# Pclass(negatively) and Fare(positively) seems the more correlated with Age. Sibsp and Parch are correlated as well


# In[29]:


import seaborn as sns
plt.figure(figsize=(8,6))
sns.heatmap(train.corr())# we can see it better with the heatmap


# In[30]:


sns.pairplot(train)#let's take a look at the distributions. Many variables looks like categorical even they are numeric
plt.show()


# Creation of new Variables

# In[31]:


train_b=train.copy()
train_b


# Log Fare
# We prefer to use the logarithm of Fare because of the high skewness

# In[32]:


train["log_Fare"]=np.log1p(train["Fare"])
train.head()


# In[33]:


test["log_Fare"]=np.log1p(test["Fare"])
test.head()


# In[34]:


train.skew()#As we can see the skewness is reduced with the new variable


# Relatives.
# 
# We count the number of people nearest to the passenger ( Sibsp + Parch)

# Not Alone.
# 
# A Variable who distinguuishes who is alone from others

# In[35]:


data = [train, test]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)
train['not_alone'].value_counts()


# In[36]:


test["not_alone"].value_counts()


# Deck
# 
# Distinguishes who has the Deck from others

# In[37]:


train.Cabin=train.Cabin.fillna("Missing")#We operate on Cabin variable to deal with missings. 
train.Cabin


# In[38]:


test.Cabin=test.Cabin.fillna("Missing")
test.Cabin


# In[39]:


#i create a function for finding the first letter of the cabin
def desk (string):
    prima=string[0]
    return prima


# In[40]:


train["deck"]=train["Cabin"].apply(desk)
train


# In[41]:


test["deck"]=test["Cabin"].apply(desk)
test


# In[42]:


pd.crosstab(train["deck"],train["Survived"])#We can clearly see that who doesn't have a deck had less chances to survive. We can modify Deck in a binary variable 


# In[43]:


train.groupby("deck")["Survived"].mean()


# Ponte
# 
# Binary variable who distinguish who was withouth deck

# In[44]:


def ponte (string):
    if string=="M":
        return 0
    else:
        return 1
    
train["Ponte"]=train.deck.apply(ponte)
train.Ponte


# In[45]:


test["Ponte"]=test.deck.apply(ponte)
test.Ponte


# In[46]:


pd.crosstab(train.Ponte,train.Survived)#We can see the influence of this variable


# In[47]:


train.info()#We have now more variable ... we have to cut some of them to make a good choice for the model("Occam's Razor"). Except for Deck they are all numeric


# In[48]:


train.corr#Finally we can see that the new variables are correlated with "Survived" we can take them and delete other original variables


# Before to proceed... we have still to manage some missing values

# In[49]:


train.isnull().mean()# We will impute the median for  Age because is more robust to outliers.. and we will impute the mode for Embarked


# In[50]:


test.isnull().mean()# like below we will impute the median for age and for the few missings for log_Fare( Fare doesn't interest us we will exclude it)


# In[51]:


train["Age"]=train["Age"].fillna(train["Age"].median())#Imputation for Age in the train set
train.isnull().mean()


# In[52]:


train["Embarked"]=train["Embarked"].fillna("S") # Southampton is the mode for the distribution 
train.isnull().mean()


# In[53]:


test["Age"]=test["Age"].fillna(test["Age"].median()) # imputation for the median of Age in test set 
test.isnull().mean()


# In[54]:


test["log_Fare"]=test["log_Fare"].fillna(test["log_Fare"].median()) # Imputation for log_Fare in test set
test.isnull().mean()


# In[55]:


from sklearn.ensemble import RandomForestClassifier

y = train["Survived"]

features = ["Pclass", "Sex","log_Fare", "Age","Embarked","Ponte",'relatives','not_alone']
X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")


# In[56]:


test


# In[57]:


output


# In[ ]:




