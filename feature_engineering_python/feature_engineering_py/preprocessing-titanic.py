#!/usr/bin/env python
# coding: utf-8

# #  Preprocessing  
# 
# **Detect and remove outliers in numerical variables**
# 
# **One month from now it will be complete.**
# 

# # Contents
# 
# 1-Import Necessary Libraries
# 
# 2-Read In and Explore the Data(Numerical variables in our dataset are **SibSp, Parch, Age and Fare**)
# 
# 3-Data Visualization
# 
# 4-Data preprocessing
# 

# # 1) Import Necessary Libraries

# **1-1: Data Analysis Libraries(Data wrangling)**

# In[1]:


import pandas as pd
import numpy as np
import missingno
from collections import Counter


# **1-2: Visualization Libraries**

# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# **1-3: Machine Learning Models**

# They will be used 

# In[3]:


from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier


# **1-4: Model evaluation**

# In[4]:


from sklearn.model_selection import cross_val_score


# **1-5: Hyperparameter tuning**

# In[5]:


from sklearn.model_selection import GridSearchCV


# **1-5: Remove warnings**

# In[6]:


import warnings
warnings.filterwarnings('ignore')


# # 2-Read our training and testing data
# 
# **Importing our CSV files**

# In[7]:


my_train_data = pd.read_csv("../input/titanic/train.csv")
my_test_data = pd.read_csv("../input/titanic/test.csv")
my_submission=pd.read_csv("../input/titanic/gender_submission.csv")


# **Let's have a look at the datasets:**
# 
# Looking training data by describe() and info()

# # Nacessary Information:
# 
# 
# **Survival: Survival (0 = No; 1 = Yes)**
# 
# **Pclass: Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)**
# 
# **Name : Name**
# 
# **Sex : Male or female**
# 
# **Age : Age in years, fractional if less than 1**
# 
# **Sibsp : Number of siblings or spouses aboard the titanic**
# 
# **Parch : Number of parents or children aboard the titanic**
# 
# **Ticket : Passenger ticket number**
# 
# **Fare : Passenger Fare**
# 
# **Cabin : Cabin Number**
# 
# **Embarked : Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)**

# In[8]:


my_train_data.describe(include="all")


# In[9]:


my_train_data.describe(include="all")


# # 3-Data Analysis
# 
# # Exploratory Data Analysis (EDA)
# 
# 
# **whats up in our dataset?**
# 
# Exploratory data analysis is the process of visualising and analysing data to extract insights. In other words, we want to summarise important characteristics and trends in our data in order to gain a better understanding of our dataset.
# 
# get a list of the features within the titanic dataset

# In[10]:


my_train_data.info()
print('-'*40)
my_test_data.info()


# In[11]:


print('my_train_data is :', my_train_data.shape)
print(' '*27)
print('my_test_data is :', my_test_data.shape)


# In[12]:


# Missing data in training set

missingno.matrix(my_train_data)


# In[13]:


# Missing data in test set 

missingno.matrix(my_test_data)


# 
# * Note that the test set has one column less than training set, the Survived column.
# 
# * This is because Survived is our response variable, or sometimes called a target variable. 
# 
# * Our job is to analyse the data in the training set and predict the survival of the passengers in the test set.

# In[14]:


my_submission.head()


# In[15]:


my_submission.shape


# Our final dataframe that is to be submitted should look something like this: **418 rows and 2 columns, one for PassengerId and one for Survived.**

# # Numerical variables
# 
# Numerical variables in our dataset are: **SibSp, Parch, Age and Fare**
# 
# 
# **Detect and remove outliers in numerical variables:**
# 
# * Outliers are data points that have extreme values and they do not conform with the majority of the data.
# * It is important to address this because outliers tend to skew our data towards extremes and can cause inaccurate model predictions.
# * I will use the Tukey method to remove these outliers.

# 
#  **This function will loop through a list of features and detect outliers in each one of those features**
#     
# 1- In each loop, a data point is deemed an outlier if it is less than the first quartile minus the outlier step or exceeds
#     
# 2- third quartile plus the outlier step. The outlier step is defined as 1.5 times the interquartile range. 
# 
# 3- Once the outliers have been determined for one feature, their indices will be stored in a list before proceeding to the next feature and the process repeats until the very last feature is completed. 
#   
# 4- Finally, using the list with outlier indices, we will count the frequencies of the index numbers and return them if their frequency exceeds n times.    
# 

# In[16]:


def detect_outliers(df, n, features):
   
    outlier_indices = [] 
    for col in features: 
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR 
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col) 
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(key for key, value in outlier_indices.items() if value > n) 
    return multiple_outliers

outliers_to_drop = detect_outliers(my_train_data, 2, ['Age', 'SibSp', 'Parch', 'Fare'])
print("We will drop these {} indices: ".format(len(outliers_to_drop)), outliers_to_drop)


# In[17]:


my_train_data.loc[outliers_to_drop, :]


# In[18]:


print("Before: {} rows".format(len(my_train_data)))
my_train_data = my_train_data.drop(outliers_to_drop, axis = 0).reset_index(drop = True)
print("After: {} rows".format(len(my_train_data)))


# # 4-Data Visualization
# **Numerical variables correlation with survival**

# In[19]:


sns.heatmap(my_train_data[['Survived', 'SibSp', 'Parch', 'Age', 'Fare']].corr(), annot = True, fmt = '.2f', cmap = 'coolwarm')


# # Numerical variable: SibSp

# In[20]:


# Value counts of the SibSp column 

my_train_data['SibSp'].value_counts(dropna = False)


# In[21]:


# Mean of survival by SibSp

my_train_data[['SibSp', 'Survived']].groupby('SibSp', as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# In[22]:


sns.barplot(x = 'SibSp', y ='Survived', data = my_train_data)
plt.ylabel('Survival Probability')
plt.title('Survival Probability by SibSp')


# # Numerical variable: Parch

# In[23]:


# Value counts of the Parch column 

my_train_data['Parch'].value_counts(dropna = False)


# In[24]:


# Mean of survival by Parch

my_train_data[['Parch', 'Survived']].groupby('Parch', as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# In[25]:


sns.barplot(x = 'Parch', y ='Survived', data = my_train_data)
plt.ylabel('Survival Probability')
plt.title('Survival Probability by Parch')


# # Numerical variable: Age

# In[26]:


# Null values in Age column 

my_train_data['Age'].isnull().sum()


# In[27]:


# Passenger age distribution

sns.distplot(my_train_data['Age'], label = 'Skewness: %.3f'%(my_train_data['Age'].skew()))
plt.legend(loc = 'best')
plt.title('Passenger Age Distribution')


# In[28]:


# Age distribution by survival

g = sns.FacetGrid(my_train_data, col = 'Survived')
g.map(sns.distplot, 'Age')


# # Numerical variable: Fare

# In[29]:


# Null values of Fare column 

my_train_data['Fare'].isnull().sum()


# In[30]:


# Passenger fare distribution

sns.distplot(my_train_data['Fare'], label = 'Skewness: %.2f'%(my_train_data['Fare'].skew()))
plt.legend(loc = 'best')
plt.ylabel('Passenger Fare Distribution')


# #  4. Data preprocessing
# 
# **Data preprocessing is the process of getting our dataset ready for model training. In this section, we will perform the following preprocessing steps:**
# 
# - **Drop and fill missing values**
# - **Data trasformation (log transformation)**
# - **Feature engineering**
# - **Feature encoding**

# # 4.1 Drop and fill missing values

# In[31]:


# Drop ticket and cabin features from training and test set

my_train_data = my_train_data.drop(['Ticket', 'Cabin'], axis = 1)
my_test_data = my_test_data.drop(['Ticket', 'Cabin'], axis = 1)


# 
# I have decided to drop both ticket and cabin for simplicity of this tutorial but if you have the time, I would recommend going through them and see if they can help improve your model.

# In[32]:


# Missing values in training set 

my_train_data.isnull().sum().sort_values(ascending = False)


# In[33]:


# Compute the most frequent value of Embarked in training set

mode = my_train_data['Embarked'].dropna().mode()[0]
mode


# In[34]:


# Fill missing value in Embarked with mode

my_train_data['Embarked'].fillna(mode, inplace = True)


# In[35]:


# Missing values in test set

my_test_data.isnull().sum().sort_values(ascending = False)


# In[36]:


# Compute median of Fare in test set 

median = my_test_data['Fare'].dropna().median()
median


# In[37]:


# Fill missing value in Fare with median

my_test_data['Fare'].fillna(median, inplace = True)

# Combine training set and test set

combine = pd.concat([my_train_data, my_test_data], axis = 0).reset_index(drop = True)
combine.head()


# In[38]:


# Convert Sex into numerical values where 0 = male and 1 = female

combine['Sex'] = combine['Sex'].map({'male': 0, 'female': 1})

sns.factorplot(y = 'Age', x = 'Sex', hue = 'Pclass', kind = 'box', data = combine)
sns.factorplot(y = 'Age', x = 'Parch', kind = 'box', data = combine)
sns.factorplot(y = 'Age', x = 'SibSp', kind = 'box', data = combine)


# In[39]:


sns.heatmap(combine.drop(['Survived', 'Name', 'PassengerId', 'Fare'], axis = 1).corr(), annot = True, cmap = 'coolwarm')


# In[ ]:




