#!/usr/bin/env python
# coding: utf-8

# # Project: Titanic - Machine Learning from Disaster
# 
# Predict survival on the Titanic
# 
# ##### Kaggle link: https://www.kaggle.com/competitions/titanic/overview
# 
# Courtesy:
# * https://www.kaggle.com/code/ccastleberry/titanic-cabin-features
# * https://www.kaggle.com/code/gunesevitan/titanic-advanced-feature-engineering-tutorial
# * https://www.kaggle.com/code/yassineghouzam/titanic-top-4-with-ensemble-modeling

# ## Problem Statement: Predict the survival from Titanic
# 
# Using the provided train and test data, we have to predict the `Surivived` column where `1` means `Survived` and `0` means `Not Survived`.
# 
# **Train Data** : Train data has features columns and target variable (`Survived`) column.
# 
# **Test Data** : Test only has features columns and do not have the `Survived` column. We will not going to using test data while building our model and also to check accuracy during testing of multiple models. Only use test data for final prediction. (This will prevent overfitting of ML models till some extent.)
# 
# **Submission Data** : We have to submit our prediction in this format. It will have `PassengerId` and `Survived` column.

# ## Task to perform to solve the problem:
# * Exploratory Data Analysis
# * Data Preprocessing & Feature Engineering
# * Data Preparation for Machine Learning
# * Implement Machine Learning Models and compare accuracy
# * Ensembling (Stacking of multiple ML models)
# * Final Prediction using final ML model with Test Data

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


# # Load required packages

# In[2]:


# Import data wrangling packages
import pandas as pd
import numpy as np
import missingno as mn
from collections import Counter
import warnings
from time import process_time

# Import visualization packages
import matplotlib.pyplot as plt
import seaborn as sns


# ML metrics packages
from sklearn.metrics import confusion_matrix

# ML model building packages
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

warnings.filterwarnings("ignore");


# # Import and read the data

# In[3]:


# Importing 3 datasets
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
gender_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')


# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


gender_submission.head()


# In[7]:


print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)
print('Submission data shape: ', gender_submission.shape)


# **Note:** We can see that training data has all the features columns and target data (`Survived`). In testing data, we have all the feature variable but not the target variable (`Survived`) which we have to predict using our machine learning model. Submission data is the format in which we have to submit the predictions in kaggle.

# # Exploratory Data Analysis
# 
# Doing univariate analysis, bivariate analysis, statistical analysis and coorelation of features with target variable.

# ### Data types, summary statistics, missing data

# In[8]:


# Summary info of the data
print('Summary information of train data')
display(train.info())
print('---'*30)
print('Summary information of test data')
display(test.info())


# In[9]:


# Summary statistics
print('Summary statistics for train data')
display(train.describe())
print('---'*30)
print('Summary statistics for test data')
display(test.describe())


# In[10]:


# Missing data
print('Missing values in train data')
display(train.isnull().sum().sort_values(ascending=False))
print('---'*30)
print('Missing values in test data')
display(test.isnull().sum().sort_values(ascending=False))


# In[11]:


# Missing value percentage
print('Percentage of missing values in train data')
display(round((train.isnull().sum() / train.shape[0])*100, 2))
print('---'*30)
print('Percentage of missing values in test data')
display(round((test.isnull().sum() / test.shape[0])*100, 2))


# In[12]:


# Visualize missing data training set
_ = mn.matrix(train, figsize=(14,6))
_ = plt.title('Missing values in training data')
plt.show()


# In[13]:


# Visualize missing data testing set
_ = mn.matrix(test, figsize=(14,6))
_ = plt.title('Missing values in testing data')
plt.show()


# **Note:** In training set `Age, Cabin, Embarked` has missing values and in testing set `Age, Fare, Cabin` has missing values.

# ### Explore Target Variable

# ***Explore `Survived` Column***
# 
# It is Survival data of passengers. Key `0 = No and 1 = Yes`

# In[14]:


# Total passenger
total_passenger = len(train.Survived)
print('Total passenger onboard: ', total_passenger)

# Total Survived
total_survived = sum(train.Survived)
print('Total passenger Survived: ', total_survived)

# Percentage of Survived
print('Percentage of passenger Survived: ', (total_survived/total_passenger))

# Plot Survived Counts
_ = sns.countplot(x='Survived', data=train)
_ = plt.title('Survived/ Unsurvived Counts')
_ = plt.xlabel('0: Unsurvived, 1: Survived')
_ = plt.ylabel('Counts')
plt.show()


# **As per the target variable result, we can say that it is a `Binary Classification` problem.**

# # Joining Train and Test Data
# 
# Here I joined train and test data, to do EDA and feature engineering.

# In[15]:


# Join train and test set
train_len = len(train)
# Prepare Test Data
testID = test['PassengerId']

dataset = pd.concat([train, test], axis=0).reset_index(drop=True)

display(train_len)

display(dataset.head())


# In[16]:


# Summary info of the data
print('Summary information of joined dataset')
display(dataset.info())
print('---'*30)

# Summary statistics
print('Summary statistics for joined dataset')
display(dataset.describe())
print('---'*30)

# Missing data
print('Missing values in joined dataset')
display(dataset.isnull().sum().sort_values(ascending=False))
print('---'*30)

# Missing value percentage
print('Percentage of missing values in joined dataset')
display(round((dataset.isnull().sum() / dataset.shape[0])*100, 2).sort_values(ascending=False))
print('---'*30)

# Visualize missing data joined dataset
_ = mn.matrix(dataset, figsize=(14,6))
_ = plt.title('Missing values in joined dataset')
plt.show()


# **Note:** `Cabin` and `Age` has most missing values. `Survived` missing values due to test dataset.

# ### Explore Feature Variables: Categorical Variable
# 
# Variables are Pclass, Sex, Embarked.

# ***Explore `Pclass` column***
# 
# It is ticket class of the passengers. Key `1 = 1st, 2 = 2nd, 3 = 3rd`
# 
# It is also a proxy for socio-economic status (SES). `1st = Upper, 2nd = Middle, 3rd = Lower`

# In[17]:


_ = plt.figure(figsize=(15,4))

# Plot Pclass counts
_ = plt.subplot(1, 3, 1)
_ = sns.countplot(x='Pclass', data=dataset)
_ = plt.title('Count of Passenger in each class')
_ = plt.xlabel('Passenger Class')
_ = plt.ylabel('Passenger Counts')

# Plot Pclass vs Survived percentage
_ = plt.subplot(1, 3, 2)
_ = sns.barplot(x='Pclass', y='Survived', data=dataset)
_ = plt.title('Survived percentage in each class')
_ = plt.xlabel('Passenger Class')
_ = plt.ylabel('Survival Probability')

# Plot Survival count of Pclass
_ = plt.subplot(1, 3, 3)
_ = sns.countplot(x='Survived', hue='Pclass', data=dataset)
_ = plt.title('Survival Count of Passenger in each class')
_ = plt.xlabel('Survived')
_ = plt.ylabel('Counts')

plt.show()

# Survival rate of each class
print('Survival rate of each class')
display(pd.crosstab(dataset['Pclass'], dataset['Survived']).apply(lambda r : round((r/r.sum())*100,1),axis=1))


# Concluded: Passengers in class 3rd is very high as compared to 1st and 2nd class. Survived percentage for 1st class passengers is higher than 2nd and 3rd class.

# ***Explore `Sex` column***
# 
# It is Gender of the passengers.

# In[18]:


_ = plt.figure(figsize=(15,4))

# Plot Gender distribution
_ = plt.subplot(1,3,1)
_ = sns.countplot(x='Sex', data=dataset)
_ = plt.title('Count of Passenger in each gender')
_ = plt.xlabel('Gender')
_ = plt.ylabel('Passenger Counts')

# Plot Sex vs Survived percentage
_ = plt.subplot(1,3,2)
_ = sns.barplot(x='Sex', y='Survived', data=dataset)
_ = plt.title('Survived percentage in each gender')
_ = plt.xlabel('Gender')
_ = plt.ylabel('Survival Probability')

# Plot Survival count of each Gender
_ = plt.subplot(1, 3, 3)
_ = sns.countplot(x='Survived', hue='Sex', data=dataset)
_ = plt.title('Survival Count of Passenger in each gender')
_ = plt.xlabel('Survived')
_ = plt.ylabel('Counts')

plt.show()

# Survival rate of each gender
print('Survival rate of each gender')
display(pd.crosstab(dataset['Sex'], dataset['Survived']).apply(lambda r : round((r/r.sum())*100,1),axis=1))


# Conclude: Male passengers are more onboard than female passengers but survived female passengers percentage are very high than male passengers percentage.

# ***Explore `Sex` and `Pclass`***

# In[19]:


# Survival of gender and passenger class
_ = sns.catplot(x='Pclass', y='Survived', hue='Sex', data=dataset, kind='bar')
_ = plt.title('Survival Probability by Sex and Passenger Class')
_ = plt.ylabel('Survival Probability')
plt.show()


# ***Explore `Embarked` column***
# 
# Port of Embarkation
# 
# C = Cherbourg, Q = Queenstown, S = Southampton

# In[20]:


_ = plt.figure(figsize=(15,4))

# Plot Pclass counts
_ = plt.subplot(1, 3, 1)
_ = sns.countplot(x='Embarked', data=dataset)
_ = plt.title('Count of Passenger based on Embarkation Port')
_ = plt.xlabel('Embarkation Port')
_ = plt.ylabel('Passenger Counts')

# Plot Pclass vs Survived percentage
_ = plt.subplot(1, 3, 2)
_ = sns.barplot(x='Embarked', y='Survived', data=dataset)
_ = plt.title('Survived percentage based on Embarkation Port')
_ = plt.xlabel('Embarkation Port')
_ = plt.ylabel('Survival Probability')

# Plot Survival count of Pclass
_ = plt.subplot(1, 3, 3)
_ = sns.countplot(x='Survived', hue='Embarked', data=dataset)
_ = plt.title('Survival Count based on Embarkation Port')
_ = plt.xlabel('Survived')
_ = plt.ylabel('Counts')

plt.show()

# Survival rate with each embarkation point
print('Survival rate with each embarkation point')
display(pd.crosstab(dataset['Embarked'], dataset['Survived']).apply(lambda r : round((r/r.sum())*100,1),axis=1))


# **Note:** Survival Probability is higher in `C = Cherbourg` location than `Q = Queenstown` and `S = Southampton`.
# 
# **What could be the reason behind this?**
# 
# From the above observation, we can also see that majority passengers on boarded from `S = Southampton` location. Also, we know that majority of the passengers are on `3rd class` and chance of survival of `3rd class` is very low than `1st class`, so it could be possible that most `1st class` passengers onboarded from `C = Cherbourg` location and most `3rd class` passengers on boarded from `S = Southampton` location.
# 
# Let's explore this.

# In[21]:


# Plot categorical plot
_ = sns.catplot(x='Pclass', col='Embarked', data=dataset, kind='count')
plt.show()


# We can see that most `3rd class` passengers on boarded in `S` location.

# In[22]:


# Survival probability of all categorical variables
grid = sns.FacetGrid(data=dataset, col='Embarked')
grid = grid.map_dataframe(sns.pointplot, x='Pclass', y='Survived', hue='Sex')
grid.add_legend()
plt.show()


# ### Explore Feature Variables: Numerical Variable
# 
# Variables are SibSp, Parch, Age and Fare.

# #### Detect and remove outliers in numerical variables
# 
# Outliers in data can skew our data towards extreme and can cause inaccurate model predictions. Here I will remove those outliers.

# In[23]:


# Function to find outliers
def find_outliers(df, n, features):
    """
    This function will loop through a given list of features and find outliers in each feature.
    A datapoint is an outlier if it is less than (1st quartile - outlier step) or exceeds (3rd quartile + outlier step). 
    Outlier step is 1.5 times the interquartile range.
    After finding the outlier for one feature, their indices will be stored in a list and repeats for all features.
    Finally, count the frequency of index numbers and return if its greater than n times.
    """
    outlier_indices = []
    for col in features:
        q1 = np.percentile(df[col], 25)
        q3 = np.percentile(df[col], 75)
        iqr = q3 - q1
        outlier_step = 1.5 * iqr
        outlier_list_col = df[(df[col] < q1 - outlier_step) | (df[col] > q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = [key for key, val in outlier_indices.items() if val > n]
    return multiple_outliers


# In[24]:


# find outliers
outliers = find_outliers(dataset, 2, ['Age', 'SibSp', 'Parch', 'Fare'])
print('Have to drop this {} indices: '.format(len(outliers)), outliers)


# In[25]:


# outliers in train data
dataset.loc[outliers, :]


# In[26]:


# # drop outliers and reset index
# print('Before rows: ', dataset.shape[0])
# dataset = dataset.drop(outliers, axis=0).reset_index(drop=True)
# print('After rows: ', dataset.shape[0])


# After looking at the outlier data. It is not making sense to delete these records. We can see many correlations like, SibSp = 8 has survival chance zero. Fare high, Pclass 1 survival chance 1. So let's not delete these. We will use scaling methods to normalize the data.

# ***Explore `Age` column***
# 
# Age in years. Age is fractional if less than 1. If the age is estimated, it is in the form of xx.5

# In[27]:


_ = plt.figure(figsize=(16,6))

# Plot Age distribution
_ = plt.subplot(2, 2, 1)
_ = sns.histplot(dataset['Age'], bins=20, kde=True)
_ = plt.title("Distribution of Passenger's Age")
_ = plt.xlabel('Age')
_ = plt.ylabel('Count')

# Plot Age distribution
_ = plt.subplot(2, 2, 2)
_ = sns.histplot(dataset['Age'], stat='density', element='step', kde=True)
_ = plt.title("Distribution of Passenger's Age")
_ = plt.xlabel('Age')
_ = plt.ylabel('Percentage')

_ = plt.figure(figsize=(16,6))

# Plot Age vs Survived Percentage
_ = plt.subplot(2, 2, 3)
_ = sns.lineplot(x='Age', y='Survived', data=dataset)
_ = plt.title('Survived percentage vs Age')
_ = plt.xlabel('Age')
_ = plt.ylabel('Survived Percentage')

# Plot boxplot to see outliers
_ = plt.subplot(2, 2, 4)
_ = sns.boxplot(x='Age', data=dataset)
_ = plt.title('Boxplot of Age distribution')

plt.show()


# In[28]:


# Age distribution by survival

grid = sns.FacetGrid(dataset, col='Survived')
grid.map(sns.kdeplot, 'Age')


# In[29]:


# Plot age distribution by survival
g = sns.kdeplot(dataset['Age'][dataset['Survived'] == 0])
g = sns.kdeplot(dataset['Age'][dataset['Survived'] == 1])
plt.legend(['Not Survived', 'Survived'])
plt.title('Passenger Age Distribution by Survival')
plt.show()


# In[30]:


# Check missing value
missing_values = dataset.Age.isnull().sum()
print('Missing values: ', missing_values)

# percentage of missing value
print('Percentage of missing values: ', round((missing_values/dataset.shape[0])*100, 2),'%')


# Conclude: Age is also a feature which is impacting survival rate. We can see in the plot that `passengers < age: 14 and passenger > age: 50` have higer survival rate as compared to other. Also we have very less data and percentage of missing values are `approx 20%`, so we cannot delete the records. We will work on the missing data in `Feature Engineering` section.

# ***Explore `SibSp` column***
# 
# Number of siblings / spouses aboard the Titanic
# 
# The dataset defines family relations as:
# * Sibling = brother, sister, stepbrother, stepsister
# * Spouse = husband, wife (mistresses and fiances were ignored)

# In[31]:


_ = plt.figure(figsize=(15,4))

# Plot SibSp distribution
_ = plt.subplot(1,3,1)
_ = sns.countplot(x='SibSp', data=dataset)
_ = plt.title('Number of siblings / spouses aboard the Titanic')
_ = plt.xlabel('Siblings / Spouses')
_ = plt.ylabel('Passenger Counts')

# Plot SibSp vs Survived percentage
_ = plt.subplot(1,3,2)
_ = sns.barplot(x='SibSp', y='Survived', data=dataset)
_ = plt.title('Survived probability by SibSp')
_ = plt.xlabel('Siblings / Spouses')
_ = plt.ylabel('Survival Probability')

# Plot Survival count of each SibSp
_ = plt.subplot(1, 3, 3)
_ = sns.countplot(x='Survived', hue='SibSp', data=dataset)
_ = plt.title('Survival Count of SibSp by Survival')
_ = plt.xlabel('Survived')
_ = plt.ylabel('Counts')

plt.show()

# Survival rate of Siblings / Spouses
print('Survival rate of Siblings / Spouses')
display(pd.crosstab(dataset['SibSp'], dataset['Survived']).apply(lambda r : round((r/r.sum())*100,1),axis=1))


# Conclude: 
# * Survival count of single passengers is higher than siblings / spouses, also single passengers much more than siblings / spouses.
# * If we look at the percentage of each type of passengers, we can see that chances of survival for single passengers are less than siblings / spouses (1 & 2) but if SibSp increases then it gets decreased.

# ***Explore `Parch` column***
# 
# Number of parents / children aboard the Titanic
# 
# The dataset defines family relations as:
# * Parent = mother, father
# * Child = daughter, son, stepdaugther, stepson
# 
# Some children travelled only with a nanny, therefore `parch = 0` for them.

# In[32]:


_ = plt.figure(figsize=(15,4))

# Plot Parch distribution
_ = plt.subplot(1,3,1)
_ = sns.countplot(x='Parch', data=dataset)
_ = plt.title('Number of parents / children aboard the Titanic')
_ = plt.xlabel('Parents / Children')
_ = plt.ylabel('Passenger Counts')

# Plot Parch vs Survived percentage
_ = plt.subplot(1,3,2)
_ = sns.barplot(x='Parch', y='Survived', data=dataset)
_ = plt.title('Survival Probability by Parch')
_ = plt.xlabel('Parents / Children')
_ = plt.ylabel('Survival Probability')

# Plot Survival count of each Parch
_ = plt.subplot(1, 3, 3)
_ = sns.countplot(x='Survived', hue='Parch', data=dataset)
_ = plt.title('Passenger Count Parch by Survival')
_ = plt.xlabel('Survived')
_ = plt.ylabel('Counts')

plt.show()

# Survival rate of Parents / Children
print('Survival rate of Parents / Children')
display(pd.crosstab(dataset['Parch'], dataset['Survived']).apply(lambda r : round((r/r.sum())*100,1),axis=1))


# Conclude: 
# * Survival count of single passengers is higher than parents / children, also single passengers much more than parents / children.
# * If we look at the percentage of each type of passengers, we can see that chances of survival for single passengers are less than parents / children.

# ***Explore `Fare` column***
# 
# Price of the ticket.

# In[33]:


_ = plt.figure(figsize=(17,4))

# Plot Fare distribution
_ = plt.subplot(1, 2, 1)
_ = sns.histplot(dataset['Fare'], bins=20, kde=True)
_ = plt.title("Distribution of Passenger's Fare")
_ = plt.xlabel('Fare')
_ = plt.ylabel('Count')

# Plot boxplot to see outliers
_ = plt.subplot(1, 2, 2)
_ = sns.boxplot(x='Fare', data=dataset)
_ = plt.title('Boxplot of Fare distribution')

plt.show()


# In[34]:


# Survived Distribution with Fare
plt.figure(figsize=(15,6))
sns.histplot(dataset[dataset.Survived==0]['Fare'], element='step', kde=True)
sns.histplot(dataset[dataset.Survived==1]['Fare'], element='step', kde=True)

plt.show()


# In[35]:


# percentage of fare less than 100
print('Percentage passengers with fare less than 25 : ',
      dataset[dataset['Fare'] < 25].shape[0] / dataset.shape[0])

print('Percentage passengers with between 26 and 50 : ',
      dataset[(dataset['Fare'] > 25) & (dataset['Fare'] < 50)].shape[0] / dataset.shape[0])

print('Percentage passengers with between 51 and 100 : ',
      dataset[(dataset['Fare'] > 51) & (dataset['Fare'] < 100)].shape[0] / dataset.shape[0])


# In[36]:


# Percentage fare distibution
_ = sns.histplot(dataset['Fare'], stat='density', element='step', kde=True, 
                 label='Skewness: %.2f'%(dataset['Fare'].skew()))
_ = plt.legend(loc='best')
_ = plt.ylabel('Passenger Fare Distribution')
plt.show()


# Conclude: 
# * As we have seen above percentage of 3rd class passengers are much higher, we can see here also passengers is are more than 80% with ticket fare less than 100.
# * Also we have high skewness in data. We will transform this data using log transformation in `Feature Engineering` section.

# ***Correleation of Numerical Variables with Survival***

# In[37]:


# Plot a heat map with correlation matrix
_ = sns.heatmap(dataset[['Survived', 'Age', 'SibSp', 'Parch', 'Fare']].corr(),
               annot=True, fmt='.2f', cmap='YlGnBu')
plt.show()


# In[38]:


_ = sns.pairplot(dataset)
plt.show()


# # Data Preprocessing
# * Feature Engineering
# * Feature Encoding
# * Data Transformation (Scaling of Data)
# * Combine all features
# 
# Note: As we have very low data points, so I am not dropping any data, instead I am filling missing values using various imputation methods.

# ***Working with `Pclass` column***

# #### Feature Encoding: One Hot Encoding
# 
# We will use onehot encoding with `Pclass` column as it is a categorical column. (use get dummies)

# In[39]:


# Encoding data using pandas get_dummies
pclass_onehotCoding = pd.get_dummies(dataset['Pclass'], drop_first=True)


# In[40]:


pclass_onehotCoding.head()


# In[41]:


# Check shape of the data
print('Shape of joined data: ', dataset.shape)
print('Shape of pclass_onehoCoding data: ', pclass_onehotCoding.shape)


# ***Working with `Name` column***
# 
# It is a categorical variable.

# #### Feature Engineering
# 
# Here I will get the title of each person and create a new variable `Title`. Then explore the relationship of Title with survival rate.

# In[42]:


# Get title from `Name`
dataset['Title'] = [name.split(',')[1].split('.')[0].strip() for name in dataset['Name']]

# Get the head data
dataset[['Name', 'Title']].head()


# In[43]:


# Get unique values
print('Count of unique Title: ', dataset['Title'].nunique())
display(dataset['Title'].unique())


# Get the value counts
display(dataset['Title'].value_counts())


# In[44]:


# Putting all the titles with lesser count in 1 bucket
dataset['Title'] = dataset.Title.replace(['Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col', 
                                          'Capt', 'the Countess', 'Jonkheer', 'Dona'], 'Rare')

dataset['Title'] = dataset.Title.replace(['Mlle', 'Ms'], 'Miss')
dataset['Title'] = dataset.Title.replace('Mme', 'Mrs')


# In[45]:


# Plot the Titles
_ = sns.countplot(x='Title', data=dataset)
_ = plt.title('Distribution of Titles')
_ = plt.ylabel('Count of Titles')
_ = plt.xlabel('Title')
plt.show()


# In[46]:


_ = plt.figure(figsize=(15,4))

# Plot Title distribution
_ = plt.subplot(1,3,1)
_ = sns.countplot(x='Title', data=dataset)
_ = plt.title('Count of Passenger with each Title')
_ = plt.xlabel('Titles')
_ = plt.ylabel('Passenger Counts')

# Plot Title vs Survived percentage
_ = plt.subplot(1,3,2)
_ = sns.barplot(x='Title', y='Survived', data=dataset)
_ = plt.title('Survived percentage for each Title')
_ = plt.xlabel('Titles')
_ = plt.ylabel('Survival Probability')

# Plot Survival count of each Title
_ = plt.subplot(1, 3, 3)
_ = sns.countplot(x='Survived', hue='Title', data=dataset)
_ = plt.title('Survival Count of Passenger for each Title')
_ = plt.xlabel('Survived')
_ = plt.ylabel('Counts')

plt.show()

# Survival rate of each Title
print('Survival rate of each Title')
display(pd.crosstab(dataset['Title'], dataset['Survived']).apply(lambda r : round((r/r.sum())*100,1),axis=1))


# Conclude: We can see that chances of survival for `Mrs` and `Miss` is very high as compared to other.

# #### Feature Encoding: One Hot Encoding
# 
# We will use onehot encoding with `Title` column as it is a categorical column. (use get dummies)

# In[47]:


# Encoding data using pandas get_dummies
title_onehotCoding = pd.get_dummies(dataset['Title'], drop_first=True)


# In[48]:


title_onehotCoding.head()


# In[49]:


# Check shape of the data
print('Shape of joined data: ', dataset.shape)
print('Shape of title_onehoCoding data: ', title_onehotCoding.shape)


# ***Working with `Sex` column***

# #### Feature Encoding: One Hot Encoding
# 
# We will use onehot encoding with `Sex` column as it is a categorical column. (use get dummies)

# In[50]:


# Encoding data using pandas get_dummies
sex_onehotCoding = pd.get_dummies(dataset['Sex'], drop_first=True)


# In[51]:


sex_onehotCoding.head()


# In[52]:


# Check shape of the data
print('Shape of joined data: ', dataset.shape)
print('Shape of sex_onehoCoding data: ', sex_onehotCoding.shape)


# ***Working with `Age` column***

# #### Feature Engineering
# * Imputation: If we look at the distribution Passenger's age it is densly populated near `mean age`. So we can replace the missing data with `mean age`.
# * Transform age into ordinal variable. It is like categorical variable with intrinsic ordering within values.
# * Create `Age*Class` feature

# ***Imputing missing data in `Age` with mean imputation***

# In[53]:


# Function to do mean imputation
def impute_nan_mean(df, column, mean):
    df[column + '_mean'] = df[column].fillna(mean)
    return df


# In[54]:


mean_val = dataset.Age.mean()
dataset = impute_nan_mean(dataset, 'Age', mean_val)


# In[55]:


dataset.tail()


# In[56]:


dataset.shape


# In[57]:


dataset.isnull().sum()


# In[58]:


# Group Ages into 6 groups
dataset['AgeGroup'] = pd.cut(dataset['Age_mean'], 6)
dataset['AgeGroup'].value_counts()


# In[59]:


# Plot AgeGroup vs Survived percentage
_ = sns.barplot(x='Survived', y='AgeGroup', data=dataset)
_ = plt.title('Survived percentage for each AgeGroup')
_ = plt.ylabel('AgeGroup')
_ = plt.xlabel('Survival Probability')


# In[60]:


# Assign ordinals to each age band 
dataset.loc[dataset['Age_mean'] <= 13.683, 'Age_ordinal'] = 0
dataset.loc[(dataset['Age_mean'] > 13.683) & (dataset['Age_mean'] <= 26.947), 'Age_ordinal'] = 1
dataset.loc[(dataset['Age_mean'] > 26.947) & (dataset['Age_mean'] <= 40.21), 'Age_ordinal'] = 2
dataset.loc[(dataset['Age_mean'] > 40.21) & (dataset['Age_mean'] <= 54.437), 'Age_ordinal'] = 3
dataset.loc[(dataset['Age_mean'] > 54.437) & (dataset['Age_mean'] <= 66.737), 'Age_ordinal'] = 4
dataset.loc[dataset['Age_mean'] > 66.737, 'Age_ordinal'] = 5


# In[61]:


dataset.head()


# In[62]:


# Age and Pclass data types 
dataset[['Age_ordinal', 'Pclass']].dtypes


# In[63]:


# Convert Age_ordinal into integer
dataset['Age_ordinal'] = dataset['Age_ordinal'].astype('int')
dataset['Age_ordinal'].dtype


# In[64]:


# Create Age*Class feature
dataset['Age*Class'] = dataset['Age_ordinal'] * dataset['Pclass']
dataset[['Age_ordinal', 'Pclass', 'Age*Class']].head()


# #### Feature Encoding: One Hot Encoding
# 
# We will use onehot encoding with `Age_ordinal` column as it is a categorical column. (use get dummies)

# In[65]:


# Encoding data using pandas get_dummies
age_ordinal_onehotCoding = pd.get_dummies(dataset['Age_ordinal'], drop_first=True)


# In[66]:


age_ordinal_onehotCoding.head()


# In[67]:


# Check shape of the data
print('Shape of joined data: ', dataset.shape)
print('Shape of age_ordinal_onehoCoding data: ', age_ordinal_onehotCoding.shape)


# ***Working with `SibSp` and `Parch` column***

# #### Feature Engineering
# 
# Create a new feature `FamilyType`. From above we can see majority of passengers are alone. And it has impact on the survival rate.

# In[68]:


# calculate family size from SibSp and Parch
dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
dataset[['SibSp', 'Parch', 'FamilySize']].head(10)


# In[69]:


dataset['FamilySize'].value_counts()


# In[70]:


_ = plt.figure(figsize=(15,4))

# Plot Family Size distribution
_ = plt.subplot(1,3,1)
_ = sns.countplot(x='FamilySize', data=dataset)
_ = plt.title('Count of Passenger with Family Size')
_ = plt.xlabel('Family Size')
_ = plt.ylabel('Passenger Counts')

# Plot Family Size vs Survived percentage
_ = plt.subplot(1,3,2)
_ = sns.barplot(x='FamilySize', y='Survived', data=dataset)
_ = plt.title('Survived percentage for Family Size')
_ = plt.xlabel('Family Size')
_ = plt.ylabel('Survival Probability')

# Plot Survival count of each Family Size
_ = plt.subplot(1, 3, 3)
_ = sns.countplot(x='Survived', hue='FamilySize', data=dataset)
_ = plt.title('Survival Count of Passenger for Family Size')
_ = plt.xlabel('Family Size')
_ = plt.ylabel('Counts')

plt.show()

# Survival rate of each Family Size
print('Survival rate of Family Size')
display(pd.crosstab(dataset['FamilySize'], dataset['Survived']).apply(lambda r : round((r/r.sum())*100,1),axis=1))


# Encoding Frequency:
# * Family Size with 1 are labeled as **Alone**
# * Family Size with 2, 3, 4 are labeled as **Small**
# * Family Size with 5, 6 are labeled as **Medium**
# * Family Size with 7, 8, 9 are labeled as **Large**

# In[71]:


# Create new feature
family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 
              7: 'Large', 8: 'Large', 11: 'Large'}

dataset['FamilySize_Group'] = dataset['FamilySize'].map(family_map)


# In[72]:


_ = plt.figure(figsize=(15,4))

# Plot Family Size Group distribution
_ = plt.subplot(1,3,1)
_ = sns.countplot(x='FamilySize_Group', data=dataset)
_ = plt.title('Count of Passenger with Family Size')
_ = plt.xlabel('Family Size Group')
_ = plt.ylabel('Passenger Counts')

# Plot Family Size Group vs Survived percentage
_ = plt.subplot(1,3,2)
_ = sns.barplot(x='FamilySize_Group', y='Survived', data=dataset)
_ = plt.title('Survived percentage for Family Size Group')
_ = plt.xlabel('Family Size Group')
_ = plt.ylabel('Survival Probability')

# Plot Survival count of each Family Size Group
_ = plt.subplot(1, 3, 3)
_ = sns.countplot(x='FamilySize_Group', hue='Survived', data=dataset)
_ = plt.title('Survival Count of Passenger for Family Size Group')
_ = plt.xlabel('Family Size Group')
_ = plt.ylabel('Counts')

plt.show()

# Survival rate of each Family Size Group
print('Survival rate of Family Size Group')
display(pd.crosstab(dataset['FamilySize_Group'], dataset['Survived']).apply(lambda r : round((r/r.sum())*100,1),axis=1))


# #### Feature Encoding: One Hot Encoding
# 
# We will use onehot encoding with `FamilySize_Group` column as it is a categorical column. (use get dummies)

# In[73]:


# Encoding data using pandas get_dummies
family_size_onehotCoding = pd.get_dummies(dataset['FamilySize_Group'], drop_first=True)


# In[74]:


family_size_onehotCoding.head()


# In[75]:


# Check shape of the data
print('Shape of joined data: ', dataset.shape)
print('Shape of family_size_onehotCoding data: ', family_size_onehotCoding.shape)


# ***Working with `Ticket` column***

# ##### Feature Engineering
# 
# Grouping of `Ticket` column.

# In[76]:


dataset['Ticket_Frequencey'] = dataset.groupby('Ticket')['Ticket'].transform('count')


# In[77]:


dataset['Ticket_Frequencey'].value_counts()


# In[78]:


# Plot Survival count of each Family Size Group

_ = sns.countplot(x='Ticket_Frequencey', hue='Survived', data=dataset)
_ = plt.title('Survival Count of Passenger per Ticket Frequency')
_ = plt.xlabel('Ticket Frequency')
_ = plt.ylabel('Counts')

plt.show()


# ***Working with `Fare` column***

# ##### Data Transformation (Scaling of Data)
# 
# Applying log transformation.

# In[79]:


# Passenger Fare distribution
_ = sns.distplot(dataset['Fare'], label='Skewness: %.2f'%(dataset['Fare'].skew()))
_ = plt.legend(loc='best')
_ = plt.title('Passenger Fare Distribution')
plt.show()


# In[80]:


# Apply log transformation
dataset['Fare_scaled'] = dataset['Fare'].map(lambda x: np.log(x) if x > 0 else 0)
dataset['Fare_scaled'].head()


# In[81]:


# Passenger Fare_scaled distribution
_ = sns.distplot(dataset['Fare_scaled'], label='Skewness: %.2f'%(dataset['Fare_scaled'].skew()))
_ = plt.legend(loc='best')
_ = plt.title('Passenger Fare Scaled Distribution')
plt.show()


# ***Working with `Cabin` column***
# 
# Cabin Number

# #### Feature Engineering
# 
# Creating 2 new columns - `Deck` and `Room`

# In[82]:


print('Number of rows in joined data: ', dataset.shape[0])
print('Number of missing data in Cabin column: ', dataset['Cabin'].isnull().sum())
print('Percentage of missing data: ', dataset['Cabin'].isnull().sum()/dataset.shape[0])


# In[83]:


dataset['Cabin'].value_counts()


# From above we can see that in `Cabin` column approx 77% missing data present. So, it is thoughtful to delete this column from the dataset. But because we have very less data in this problem, so I thought of keeping this column. 
# 
# Also, most of the cabins consists of single letter at beginning followed by 2 digit or 3 digit numbers. As per our understanding from any cruise ship it is possible that letter represents the deck or section of the ship followed by the room number. We can assume that based which section a passenger is staying can have impact on their chances of survival.
# 
# Let's clean the data based on above understanding.

# In[84]:


# isolating the non null data
cabin_only = dataset[['Cabin']].copy()
cabin_only['Cabin_data'] = cabin_only['Cabin'].isnull().apply(lambda x: not x)


# In[85]:


# take first character and assign it to 'Deck' column
cabin_only['Deck'] = cabin_only['Cabin'].str.slice(0, 1)

# take numerical sequence and assign it to 'Room' column
cabin_only['Room'] = cabin_only['Cabin'].str.slice(1, 5).str.extract('([0-9]+)', expand=False).astype('float')

cabin_only[cabin_only['Cabin_data']]


# In[86]:


cabin_only[cabin_only['Deck']=='F']


# In[87]:


sns.boxplot(x=cabin_only['Room'])
print('Mean of Room Numbers: ', cabin_only['Room'].mean())
print('Median of Room Numbers: ', cabin_only['Room'].median())


# In[88]:


# Decks in the ship
cabin_only['Deck'].unique()


# Note:
# * Some entries contains multiple cabin rooms and rooms are closser. And in most cases this are belongs to the same deck.
# * Some entries contains 2 separate letters where we only get the first letter. But this are very less so ignoring it.

# **Dealing with missing data**
# 
# Impution with 'N' character in `Deck` column
# 
# Imputation using median in `Room` column

# In[89]:


# Drop Cabin and Cabin_data columns
cabin_only.drop(['Cabin', 'Cabin_data'], axis=1, inplace=True)


# In[90]:


cabin_only


# In[91]:


# dealing with missing data

# in Deck column replace the null values with an unused letter
cabin_only['Deck'] = cabin_only['Deck'].fillna('N')

# in Room column replace the null values with median
cabin_only['Room'] = cabin_only['Room'].fillna(cabin_only['Room'].median())


# In[92]:


cabin_only.head()


# In[93]:


cabin_only.info()


# #### Feature Encoding: One Hot Encoding
# 
# We will use onehot encoding with `Deck` column as it is a categorical column. (use get dummies)

# In[94]:


deck_onehotCoding = pd.get_dummies(cabin_only['Deck'], drop_first=True)


# In[95]:


deck_onehotCoding.head()


# In[96]:


# Check shape of the data
print('Shape of joined data: ', dataset.shape)
print('Shape of deck_onehoCoding data: ', deck_onehotCoding.shape)
print('Shape of the Cabin Room data: ', cabin_only['Room'].shape)


# ***Working with `Embarked` column***

# #### Feature Engineering
# 
# Working with missing values.

# In[97]:


# Missing data
dataset.Embarked.isnull().sum()


# ***Imputing missing data in `Embarked` with mode imputation***

# In[98]:


# STEP 1: Find Mode Values
mode_embarked = dataset.Embarked.dropna().mode()[0]
mode_embarked


# In[99]:


# STEP 2: Fill Missing Values With Most Frequent Category
dataset['Embarked_mode'] = dataset['Embarked'].fillna(mode_embarked)

# Check For Results
dataset['Embarked_mode'].isnull().sum()


# #### Feature Encoding: One Hot Encoding
# 
# We will use onehot encoding with `Embarked_mode` column as it is a categorical column. (use get dummies)

# In[100]:


embarked_onehotCoding = pd.get_dummies(dataset['Embarked_mode'], drop_first=True)


# In[101]:


embarked_onehotCoding.head()


# In[102]:


# Check shape of the data
print('Shape of joined data: ', dataset.shape)
print('Shape of embarked_onehoCoding data: ', embarked_onehotCoding.shape)


# ## Combine all the features

# In[103]:


all_cols = dataset[['Survived', 'Ticket_Frequencey', 'Fare_scaled']]
all_cols.head()


# In[104]:


# Renaming columns
pclass_onehotCoding.rename(columns={2:'pclass_1', 3:'pclass_2'}, inplace = True)
title_onehotCoding.rename(columns={'Miss':'title_1', 'Mr':'title_2', 
                                   'Mrs':'title_3', 'Rare':'title_4'}, inplace = True)
sex_onehotCoding.rename(columns={'male': 'sex_1'}, inplace=True)
age_ordinal_onehotCoding.rename(columns={1:'age_1', 2:'age_2', 3:'age_3', 
                                         4:'age_4', 5:'age_5'}, inplace=True)
family_size_onehotCoding.rename(columns={'Large':'family_1', 'Medium':'family_2',
                                        'Small':'family_3'}, inplace=True)
deck_onehotCoding.rename(columns={'B':'deck_1', 'C':'deck_2', 'D':'deck_3',
                                 'E':'deck_4', 'F':'deck_5', 'G':'deck_6',
                                 'N':'deck_7', 'T':'deck_8'}, inplace=True)
embarked_onehotCoding.rename(columns={'Q':'embarked_1', 'S':'embarked_2'}, inplace=True)


# In[105]:


final_dataset = pd.concat([all_cols, pclass_onehotCoding, title_onehotCoding, 
               sex_onehotCoding, age_ordinal_onehotCoding, family_size_onehotCoding,
               deck_onehotCoding, embarked_onehotCoding], axis=1)


# In[106]:


final_dataset.head()


# In[107]:


final_dataset.info()


# In[108]:


# Check shape of the data
display(final_dataset.shape)


# # Machine Learning Modeling

# In[109]:


## Separate train dataset and test dataset

train_data = final_dataset[:train_len]
test_data = final_dataset[train_len:]
test_data = test_data.drop(['Survived'], axis=1)


# In[110]:


print('Number of data points in train data:', train_data.shape)
print('Number of data points in test data:', test_data.shape)


# In[111]:


# convert 'Survived' column to integer
train_data['Survived'] = train_data['Survived'].astype(int)
train_data['Survived'].head()


# In[112]:


# Separate features and target variable from train_data
X_train = train_data.drop(['Survived'], axis=1)
y_train = train_data['Survived']
X_train.head()


# ## Modeling

# ### Cross Validate Models

# Now comparing multiple models and evaluate the mean accuracy using stratified kfold cross validation.
# * K Nearest Neighbors Classifier
# * Logistic Regression
# * Support Vector Classification
# * Decision Tree Classifier
# * Random Forest
# * Extra Tree Classifier
# * Ada Boost
# * Gradient Boosting
# * Linear Discriminant Analysis
# * Multi Layer Perceptron (Neural Network)

# In[113]:


# cross validate model with Stratified Kfold cross validation
kfold = StratifiedKFold(n_splits=12)


# In[114]:


# create list of different algorithms
random_state = 42
classifiers = []

classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state=random_state))
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),
                                     random_state=random_state, learning_rate=0.1))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(LinearDiscriminantAnalysis())
classifiers.append(MLPClassifier(random_state=random_state, max_iter=1000))


# In[115]:


# Start the stopwatch / counter 
t1_start = process_time() 

# get cross validataion results for all algorithms
cv_results = []

for clf in classifiers:
    cv_results.append(cross_val_score(clf, X_train, y_train, scoring='accuracy',
                                     cv=kfold, n_jobs=4))

# Stop the stopwatch / counter
t1_stop = process_time()

print('Time took for performing Cross Validation of all algorithms: ', 
      t1_stop-t1_start, 'seconds')


# In[116]:


# calculate cross validation means and standard deviation for each classifier
cv_means = []
cv_std = []

for res in cv_results:
    cv_means.append(res.mean())
    cv_std.append(res.std())


# In[117]:


# create a dataframe with cv results
cv_df = pd.DataFrame({'cross_val_means': cv_means, 'cross_val_errors': cv_std,
                     'algorithm':['KNeighborsClassifier', 'LogisticRegression',
                                 'SVC', 'DecisionTreeClassifier', 'RandomForestClassifier',
                                 'ExtraTreesClassifier', 'AdaBoostClassifier',
                                 'GradientBoostingClassifier', 'LinearDiscriminantAnalysis',
                                 'MLPClassifier']})

cv_df.sort_values(by='cross_val_means', ascending=False)


# In[118]:


# Plot the cross validation results
_ = sns.barplot(x='cross_val_means', y='algorithm', 
                data=cv_df.sort_values(by='cross_val_means', ascending=False), 
                orient='h', **{'xerr':cv_std})
_ = plt.xlabel('Mean Accuracy')
_ = plt.ylabel('ML Algorithm')
_ = plt.title('Cross Validation Scores')


# From all the algorithms, I chose these algorithms, `GradientBoosting, SVC, MLPClassifier, LogisticRegression, LinearDiscriminantAnalysis, RandomForestClassifier` classifiers for the ensemble modeling.

# ## Hyperparameter tuning for best models
# 
# * Perform grid search optimization for `GradientBoosting, SVC, LogisticRegression and LinearDiscriminantAnalysis` classifiers.
# * Set the 'n_jobs' parameter to 4 (as per 4 CPU).

# In[119]:


# Start the stopwatch / counter 
t1_start = process_time() 

# GradientBoosting tuning
clf1 = GradientBoostingClassifier()
clf1_param = {'loss': ['deviance'],
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.1, 0.05, 0.01],
            'max_depth': [4, 8],
            'min_samples_leaf': [1, 2],
            'max_features': [0.3, 0.1]}

gs_clf1 = GridSearchCV(clf1, param_grid=clf1_param, cv=kfold,
                      scoring='accuracy', n_jobs=4, verbose=True)

gs_clf1.fit(X_train, y_train)

# best estimator
clf1_best = gs_clf1.best_estimator_
display(clf1_best)

# best score
print('Best Score: ', gs_clf1.best_score_)

# Stop the stopwatch / counter
t1_stop = process_time()

print('Time took for performing Gradient Boosting classifier: ', 
      t1_stop-t1_start, 'seconds')


# In[120]:


# Start the stopwatch / counter 
t1_start = process_time() 

# SVC classifier tuning
clf2 = SVC(probability=True)

clf2_param = {'kernel': ['rbf'],
             'gamma': [0.001, 0.01, 0.1, 1],
             'C': [1, 10, 50, 100, 200, 300, 1000]}

gs_clf2 = GridSearchCV(clf2, param_grid=clf2_param, cv=kfold,
                      scoring='accuracy', n_jobs=4, verbose=True)

gs_clf2.fit(X_train, y_train)

# best estimator
clf2_best = gs_clf2.best_estimator_
display(clf2_best)

# best score
print('Best Score: ', gs_clf2.best_score_)

# Stop the stopwatch / counter
t1_stop = process_time()

print('Time took for performing SVC classifier: ', 
      t1_stop-t1_start, 'seconds')


# In[121]:


# Start the stopwatch / counter 
t1_start = process_time() 

# LogisticRegression classifier tuning
clf3 = LogisticRegression()

clf3_param = {'penalty' : ['l2'],
              'C' : [0.01, 0.1, 1, 10, 100],
              'solver' : ['lbfgs','newton-cg','liblinear'],
              'max_iter': [1000, 2000],
              'intercept_scaling': [1, 2, 3, 4],
              'tol': [0.0001, 0.0002, 0.0003]
             }

gs_clf3 = GridSearchCV(clf3, param_grid=clf3_param, cv=kfold,
                      scoring='accuracy', n_jobs=4, verbose=True)

gs_clf3.fit(X_train, y_train)

# best estimator
clf3_best = gs_clf3.best_estimator_
display(clf3_best)

# best score
print('Best Score: ', gs_clf3.best_score_)

# Stop the stopwatch / counter
t1_stop = process_time()

print('Time took for performing Logistic Regression classifier: ', 
      t1_stop-t1_start, 'seconds')


# In[122]:


# Start the stopwatch / counter 
t1_start = process_time() 

# LinearDiscriminantAnalysis classifier tuning
clf4 = LinearDiscriminantAnalysis()

clf4_param = {'solver': ['svd', 'lsqr', 'eigen'],
              'tol': [0.0001, 0.0002, 0.0003]
             }

gs_clf4 = GridSearchCV(clf4, param_grid=clf4_param, cv=kfold,
                      scoring='accuracy', n_jobs=4, verbose=True)

gs_clf4.fit(X_train, y_train)

# best estimator
clf4_best = gs_clf4.best_estimator_
display(clf4_best)

# best score
print('Best Score: ', gs_clf4.best_score_)

# Stop the stopwatch / counter
t1_stop = process_time()

print('Time took for performing Linear Discriminant Analysis classifier: ', 
      t1_stop-t1_start, 'seconds')


# In[123]:


# Start the stopwatch / counter 
t1_start = process_time() 

# RandomForest classifier tuning
clf5 = RandomForestClassifier()

clf5_param = {'max_depth': [None],
             'max_features': [1, 3, 10],
             'min_samples_split': [2, 3, 10],
             'min_samples_leaf': [1, 3, 10],
             'bootstrap': [False],
             'n_estimators': [100, 300],
             'criterion': ['gini']}

gs_clf5 = GridSearchCV(clf5, param_grid=clf5_param, cv=kfold,
                      scoring='accuracy', n_jobs=4, verbose=True)

gs_clf5.fit(X_train, y_train)

# best estimator
clf5_best = gs_clf5.best_estimator_
display(clf5_best)

# best score
print('Best Score: ', gs_clf5.best_score_)

# Stop the stopwatch / counter
t1_stop = process_time()

print('Time took for performing RandomForest classifier: ', 
      t1_stop-t1_start, 'seconds')


# ## Plot Learning Curves
# 
# See overfitting effect on the training set and effect of training size on accuracy using Learning curves.

# In[124]:


def plot_learning_curves(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate simple plot for the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y,
                                                           cv=cv, n_jobs=n_jobs,
                                                           train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
            label='Training Score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
            label='Cross Validation Score')
    plt.legend(loc='best')
    return plt


# In[125]:


_ = plot_learning_curves(clf1_best, 'Gradient Boosting Learning Curves', 
                         X_train, y_train, cv=kfold)

_ = plot_learning_curves(clf2_best, 'SVC Learning Curves', 
                         X_train, y_train, cv=kfold)

_ = plot_learning_curves(clf3_best, 'LogisticRegression Learning Curves', 
                         X_train, y_train, cv=kfold)

_ = plot_learning_curves(clf4_best, 'LinearDiscriminantAnalysis Learning Curves', 
                         X_train, y_train, cv=kfold)

_ = plot_learning_curves(clf5_best, 'RandomForest Learning Curves', 
                         X_train, y_train, cv=kfold)


# ##### Typical Features of the learning curve of an overfit model:
# * Training score and Validation score are far away from each other.
# * Gradually increasing validation score (without flattening) upon adding training examples.
# * training score slightly increasing upon adding training examples.
# 
# ##### Typical Features of the learning curve of an good fit model:
# * Training score and validation score are close to each other with training accuracy slightly greater than validation score
# * Innitially increasing training and validation score and a pretty flat training and validation score after some point till end.

# Note: Gradient Boosting seems to over fit and SVC, LDA, RF seems a good fit model.

# ## Feature Importance of classifiers
# 
# To see most informative features for the prediction of passenger survival, lets plot feature importance

# In[126]:


_ = plt.figure(figsize=(14,6))


# Plot feature importance of GradientBoosting
_ = plt.subplot(1,2,1)
indices = np.argsort(clf1_best.feature_importances_)[::-1][:40]
_ = sns.barplot(y=X_train.columns[indices][:40], x=clf1_best.feature_importances_[indices][:40],
               orient='h')
_ = plt.title('Gradient Boosting Feature Importance')
_ = plt.xlabel('Relative Imporatance')
_ = plt.ylabel('Features')

# Plot feature importance of RandomForest
_ = plt.subplot(1,2,2)
indices = np.argsort(clf5_best.feature_importances_)[::-1][:40]
_ = sns.barplot(y=X_train.columns[indices][:40], x=clf5_best.feature_importances_[indices][:40],
               orient='h')
_ = plt.title('Random Forest Feature Importance')
_ = plt.xlabel('Relative Imporatance')


# Above 2 classifiers has same top 3 features according to relative importance. They share common features for classification.
# 
# Also we can say:
# * pclass_1, pclass_2 and Fare_scaled refer to the general social standing of passengers.
# * family_1, family_2, family_3 refer to the size of the passenger family.

# ### Prediction on all classifiers

# In[127]:


test_data.head()


# In[128]:


y_pred_clf1 = pd.Series(clf1_best.predict(test_data), name='GBC')
y_pred_clf2 = pd.Series(clf2_best.predict(test_data), name='SVC')
y_pred_clf3 = pd.Series(clf3_best.predict(test_data), name='LR')
y_pred_clf4 = pd.Series(clf4_best.predict(test_data), name='LDA')
y_pred_clf5 = pd.Series(clf5_best.predict(test_data), name='RFC')

# Concatenate all classifier results
ensemble_results = pd.concat([y_pred_clf1, y_pred_clf2,
                             y_pred_clf3, y_pred_clf4,
                             y_pred_clf5], axis=1)


g= sns.heatmap(ensemble_results.corr(),annot=True)


# The prediction seems to be quite similar for the 5 classifiers.
# 
# The 5 classifiers give more or less the same prediction but there is some differences. Theses differences between the 5 classifier predictions are sufficient to consider an ensembling vote.

# ## Ensemble Modeling

# ## Combining Models
# 
# Using a voting classifier to combine the predictions from 5 classifiers. Pass 'soft' argument to voting parameter to take into account of probability of each vote.

# In[129]:


# create a voting classifier

voting_clf = VotingClassifier(estimators=[('gbc', clf1_best), ('svc', clf2_best),
                                         ('lr', clf3_best), ('lda', clf4_best),
                                         ('rfc', clf5_best)], voting='soft', n_jobs=4)

voting_clf = voting_clf.fit(X_train, y_train)


# ## Prediction

# ### Predict and Submit Results

# In[130]:


# Prediction using voting classifier
y_pred = pd.Series(voting_clf.predict(test_data), name='Survived')

# Create results dataframe with PassengerID, Survived column
results = pd.concat([testID, y_pred], axis=1)


# In[131]:


# Save the results to csv file
results.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:




