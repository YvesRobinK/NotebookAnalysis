#!/usr/bin/env python
# coding: utf-8

# ## Feature Selection & Data Engineering Tutorial (Step by Step For New Comers):
# 
# Hello kagglers.
# 
# In this notebook we will learn:
#  - 1 ) What is feature selection?    
#  - 2 ) Why is feature selection very important?  
#  - 3 ) Dimensionality reduction and feature selection.  
#  - 4 ) Filter Methods.  
# 
# Then we will apply what we learned on the Titanic Dataset.
# 
# I hope this notebook will be useful to you ..  
# Now let's start, happy learning :)

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # data visualization
import os 
import warnings
warnings.filterwarnings("ignore")

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


def read_data():
    train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
    print("Train data imported successfully!!")
    print("-"*50)
    test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
    print("Test data imported successfully!!")
    return train_data , test_data


# In[3]:


train_data , test_data = read_data()


# ### 1 ) What is feature selection?

# Feature Selection is the process of selecting the **most significant** features from a given dataset. In many cases, Feature Selection can also **enhance** the performance of a machine learning model.
# 
# The problem of having unnecessary features:
# 
# - Unnecessary resource allocation for these features.
# - These features act as a noise for which the machine learning model can perform terribly poorly.
# - The machine model takes more time to get trained.
# 
# Feature selection is also known as **Variable selection** or **Attribute selection**.
# 
# 
# 

# ### 2 ) Why is feature selection very important?

# The importance of feature selection can best be recognized when you are dealing with a dataset that contains a vast number of features. This type of dataset is often referred to as a high dimensional dataset. Now, with this high dimensionality, comes a lot of problems such as - this high dimensionality will significantly increase the training time of your machine learning model, it can make your model very complicated which in turn may lead to **Overfitting**.  
# 
# **The objective of variable selection is three-fold:**  
# 1 - improving the prediction performance of the predictors.  
# 2 - providing faster and more cost-effective predictors.  
# 3 - providing a better understanding of the underlying process that generated the data.
# 
# 

# ### 3 ) Dimensionality reduction Vs Feature selection:

# Sometimes, feature selection is mistaken with dimensionality reduction. But they are different. Feature selection is different from dimensionality reduction. Both methods tend to reduce the number of attributes in the dataset, but:
# - **Dimensionality reduction** method does so by creating new combinations of attributes (sometimes known as feature transformation). whereas  
# - **Feature selection** methods include and exclude attributes present in the data without changing them.

# ### 4 ) Filter methods:

# The following image best describes filter-based feature selection methods:

# <img src="https://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1537552825/Image3_fqsh79.png">
# 

# <center>(Image Source: Analytics Vidhya)<\center>

# The filter method selects subsets of features based on their relationship with the target by:
# - Statistical Methods  
# - Feature Importance Methods   
# 
# It is common to use correlation type statistical measures between input and output variables as the basis for filter feature selection.
# 
# As such, the choice of statistical measures is **highly dependent** upon the variable **data types**.
# 
# Common data types include numerical (such as height) and categorical (such as a label), although each may be further subdivided such as integer and floating point for numerical variables, and boolean, ordinal, or nominal for categorical variables.
# 

# <img src="https://machinelearningmastery.com/wp-content/uploads/2020/06/Overview-of-Data-Variable-Types2.png">
# 
# 

# In this section, we will consider two broad categories of variable types: numerical and categorical; also, the two main groups of variables to consider: input and output.  
# **Numerical Output:** Regression predictive modeling problem.  
# **Categorical Output:** Classification predictive modeling problem. (This is our state)  
# The statistical measures used in filter-based feature selection are generally calculated one input variable at a time with the target variable. As such, they are referred to as **univariate** statistical measures. This may mean that any interaction between input variables is **not considered** in the filtering process.

# 
# <img src="https://machinelearningmastery.com/wp-content/uploads/2019/11/How-to-Choose-Feature-Selection-Methods-For-Machine-Learning.png">
# 
# 

# #### Let's see our dataset: 

# In[4]:


train_data.head(3)


# In[5]:


# ==============================================================================
#  Data Types:
# ==============================================================================

train_data.info()


# **Numerical Data:**
# 
#  - PassengerId (int)
#  - SibSp (int)
#  - Parch (int)
#  - Age (float)
#  - Fare (float)  
# 
# **Categorical Data:**
#  - Pclass (Ordinal)
#  - Name (Nominal)
#  - Ticket (Nominal)
#  - Cabin (Nominal)
#  - Embarked (Nominal)
# 

# In[6]:


# ==============================================================================
#  Missed Values
# ==============================================================================

train_data.isnull().sum()


# In[7]:


# ==============================================================================
# Data Cleaning:
# ==============================================================================

# Dropping Unuseful feature because it has too many missed values:

train_data.drop(columns = ["Cabin"] , inplace = True)
train_data.drop(columns = ["Ticket"] , inplace = True)


# ==============================================================================
#  Fill missed embarked values:  

train_data.Embarked = train_data.Embarked.fillna(train_data.Embarked.dropna().max())

# ==============================================================================
#  Fill missed age values:  

train_data['Sex'] = train_data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
guess_ages = np.zeros((2,3))

for i in range(0, 2):
    for j in range(0, 3):
        guess_df = train_data[(train_data['Sex'] == i) & \
                              (train_data['Pclass'] == j+1)]['Age'].dropna()
        age_guess = guess_df.median()

        # Convert random age float to nearest .5 age
        guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

for i in range(0, 2):
    for j in range(0, 3):
        train_data.loc[ (train_data.Age.isnull()) & (train_data.Sex == i) & (train_data.Pclass == j+1),\
                'Age'] = guess_ages[i,j]

train_data['Age'] = train_data['Age'].astype(int)
train_data.head()


# In[8]:


train_data.isnull().sum()


# **No More Missed Values !!!**

# #### Now Let's see the correlation between the features and our target.

# In[9]:


train_data.corr()["Survived"].sort_values(ascending=False)


# From the correlation above:
# - Sex and Fare has strong positive correlation with the target. 
# - PassengerId  has very weak correlation with the target. 
# - Pclass has strong negative correlation with the target. 

# It's obviouse that **Data Engineering** is very important for this problem.

# **Note : This approach to feature selection will likely fail if there are important interactions between attributes where only one of the attributes is significant**

# Let's see the correlation between all our features:

# In[10]:


sns.set(rc = {'figure.figsize':(10,6)})
sns.heatmap(train_data.corr(), annot = True, fmt='.2g',cmap= 'YlGnBu')


# In[11]:


train_data.head()


# In[12]:


# ==========================================================================================
# Data Engineering:
# ==========================================================================================

# Family Size:

train_data['Family_Size'] = train_data["Parch"] + train_data["SibSp"] + 1

# ==========================================================================================
# Is Alone:

train_data['IsAlone'] = 0
train_data.loc[train_data['Family_Size'] == 1, 'IsAlone'] = 1

# ==========================================================================================
# Age Band:

train_data.loc[ train_data['Age'] <= 16, 'Age'] = 0
train_data.loc[(train_data['Age'] > 16) & (train_data['Age'] <= 32), 'Age'] = 1
train_data.loc[(train_data['Age'] > 32) & (train_data['Age'] <= 48), 'Age'] = 2
train_data.loc[(train_data['Age'] > 48) & (train_data['Age'] <= 64), 'Age'] = 3
train_data.loc[ train_data['Age'] > 64, 'Age']

# ==========================================================================================
# Fare Band:

train_data.loc[ train_data['Fare'] <= 130, 'Fare'] = 0
train_data.loc[(train_data['Fare'] > 130) & (train_data['Fare'] <= 256), 'Fare'] = 1
train_data.loc[(train_data['Fare'] > 256) & (train_data['Fare'] <= 384), 'Fare'] = 2
train_data.loc[ train_data['Fare'] > 384, 'Fare'] = 3
train_data['Fare'] = train_data['Fare'].astype(int)

# ==========================================================================================
# Name Title:

train_data['Title'] = train_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train_data['Title'] = train_data['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train_data['Title'] = train_data['Title'].replace('Mlle', 'Miss')
train_data['Title'] = train_data['Title'].replace('Ms', 'Miss')
train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
train_data['Title'] = train_data['Title'].map(title_mapping)
train_data['Title'] = train_data['Title'].fillna(0)

train_data.drop(columns = ["Name"] , inplace = True)


# ==========================================================================================
# Embarked:
train_data['Embarked'] = train_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# ==========================================================================================
# Passenger Id:
train_data.drop(columns = ["PassengerId"] , inplace = True)

# ==========================================================================================
# ==========================================================================================

print("Data Engineering Finished !!!")


# In[13]:


train_data.head()


# In[14]:


train_data.corr()["Survived"].sort_values(ascending=False)


# In[15]:


sns.set(rc = {'figure.figsize':(10,6)})
sns.heatmap(train_data.corr(), annot = True, fmt='.2g',cmap= 'YlGnBu')


# In[16]:


# ==========================================================================================
# Feature Selection
# We will select best 8 features
# ==========================================================================================


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

fs = SelectKBest(score_func=f_classif, k=8)

print("Data shape before feature selection:")
print(train_data.shape)

# apply feature selection
Selected_train_data = fs.fit_transform(train_data.iloc[:,1:], train_data["Survived"])
print("Data shape After feature selection:")
print(Selected_train_data.shape)


# #### Feature Selection Finished !!

# ### References:
# 
# - [Data Camp Python Feature Selection Tutorial: A Beginner's Guide](https://www.datacamp.com/tutorial/feature-selection-python)
# - [Machine Learning Mastery (How to Choose a Feature Selection Method For Machine Learning Article)](https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/)
# 
# 
