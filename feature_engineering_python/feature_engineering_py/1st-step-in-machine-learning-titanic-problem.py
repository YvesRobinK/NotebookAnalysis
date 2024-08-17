#!/usr/bin/env python
# coding: utf-8

# In the [last lesson](https://www.kaggle.com/prashantkikani/introduction-to-basic-python-libraries-for-ml), we saw basics of Python libraris which we use in Machine Learning.<br>
# Today in this lesson, we are going to solve one of the most famous & interesting problem. **Titanic survival problem** 🛳️
# 
# ![titanic](https://upload.wikimedia.org/wikipedia/commons/6/6e/St%C3%B6wer_Titanic.jpg)
# 
# <br>
# In this problem, we have some data about each passenger that were into that ship.<br>
# **Our problem is to predict or forecast, whether this person will survive the ship sinking or not**.<br><br>
# 
# We will try to solve this problem in a standard way.<br>
# We can solve any machine learning problem this way.<br>
# 
# ## Goal of this lesson is to learn a standard way to solve / approch any machine learning problem.
# 
# We will keep this lesson as simple as possible so that everyone can grasp the idea & learn to solve any basic problem in ML.
# 
# Here are the steps we tentatively follow.<br>
# 1. Open the data files.
# 2. Understand the data. What each column in the table means.
# 3. Preprocess data
#     * Remove the outliers.
#     * Fill `NaN` or `null` values. Sometimes, we also remove all the rows with `NaN` values.
#     * Feature engineering - Create new columns out of existing columns using our understanding.
#     * Converting data into numeric form if it's not.
# 4. Train a machine learning model.
# 5. Validate the trained model i.e. checking it's performance on unseen data.
# 6. If it performs good in validation, use model to predict future real world data.
# 
# <br>
# Above steps are generally followed to solve a ML problem.<br>
# So, let's start..

# In[1]:


# import necessary libraries first

# pandas to open data files & processing it.
import pandas as pd

# numpy for numeric data processing
import numpy as np

# sklearn to do preprocessing & ML models
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Matplotlob & seaborn to plot graphs & visulisation
import matplotlib.pyplot as plt 
import seaborn as sns

# to fix random seeds
import random, os

# ignore warnings
import warnings
warnings.simplefilter(action='ignore')


# # 1. Open the data files.

# In[2]:


titanic_data = pd.read_csv("../input/titanic/train.csv")
titanic_data.shape


# So, we have total of 891 rows & 12 columns

# In[3]:


titanic_data.head()


# # 2. Understand the data. What each column in the table means.
# 
# **Each row in above table contains data of a passenger.<br>**
# Those details include following columns.<br>
# 
# Here is all the columns of above table mean.<br>
# 
# `PassengerId` : Unique ID for each passenger.<br><br>
# `Survived` : Whether that passenger survived or not. (0 = No, 1 = Yes)<br><br>
# `Pclass` : Ticket class of passenger. (1 = Upper, 2 = Middle, 3 = Lower)<br><br>
# `Name` : Name of passenger<br><br>
# `Sex` : Gender of passenger<br><br>
# `Age` : Age of passenger<br><br>
# `SibSp` : # of siblings / spouses aboard the Titanic of passenger<br><br>
# `Parch` : # of parents / children aboard the Titanic of passenger<br><br>
# `Ticket` : Ticket number of passenger<br><br>
# `Fare` : Ticket amount / passenger fare.<br><br>
# `Cabin` : Cabin number of passenger<br><br>
# `Embarked` : Port of Embarkation of passenger. (C = Cherbourg, Q = Queenstown, S = Southampton)<br><br>

# In[4]:


titanic_data.describe()


# ## Let's check unique values for each column

# In[5]:


# Survival
titanic_data['Survived'].value_counts()


# So, total 335 people have survived & 547 people have died in the Titanic.

# In[6]:


# Ticket class
titanic_data['Pclass'].value_counts()


# This tells `3` value occurs 491 times, `1` value occurs 207 times etc.

# In[7]:


# Gender
titanic_data['Sex'].value_counts()


# In[8]:


# Siblings
titanic_data['SibSp'].value_counts()


# In[9]:


# Parent or Childs
titanic_data['Parch'].value_counts()


# In[10]:


# Embarked station
titanic_data['Embarked'].value_counts()


# Most of the passengers have embarked from "Cherbourg" & "Southampton"

# In[11]:


sns.countplot(titanic_data['Sex']);


# In[12]:


sns.barplot(titanic_data['Survived'], titanic_data['Sex']);


# Wow !<br>
# ~75% of females have survived.<br>
# Even if total number of females are less than males.<br><br>
# 
# May be because, females were given more priority in lifeboats than males. May be.

# In[13]:


sns.barplot(titanic_data['Survived'], titanic_data['Fare'], titanic_data['Pclass']);


# People with higher class have higher chances of survival !

# # 3. Preprocess data
# 
# In preprocessing step, we detect outliers & remove them from our data.
# 
# ## 3.1
# 
# ## What is an outlier in data? Why does it occur?
# Outliers are as the name suggests, very different from general / normal trend.<br>
# They occur in data because of some faults in data collection pipeline.<br><br>
# 
# ## Why we generally remove outliers?
# Because one big outlier can mess up whole model's performance.<br>
# Even if all other contributions might be of a low value, one high outlier value already shifts the entire gradient towards higher values as well.<br>
# 
# Most of the time, we remove outliers so that, we can train our model only from general trends.<br>
# Let's see some examples using our titanic data.<br>
# 
# ### One common practice followed to detect outliers is BoxPlot.

# In[14]:


sns.boxplot(x=titanic_data["Fare"])
plt.show()


# We can see, for majority of passengers, `Fare` price is less than 250.<br>
# So, let's only keep the rows with `Fare` < 250.

# In[15]:


# Only take rows which have "Fare" value less than 250.
titanic_data = titanic_data[titanic_data['Fare'] < 250]
titanic_data.shape


# So, we have removed 9 rows.<br>Originally, there were 891 rows.

# In[16]:


sns.boxplot(x=titanic_data["Age"])
plt.show()


# We can see there are some outliers in `Age`, but they are not much far. So, we will keep as of now.

# ## 3.2 Fill NaN or null values in data
# 
# ### Why NaN (not a number) values occur in data?
# Sometimes, while collecting data, if some information is missing for some rows, it's filled as NaN.<br>
# It means nothing is there.<br>
# It's empty.<br>
# 
# ### How NaN values can be handled?
# There are several methods.
# * Fill a specified value like "EMPTY" or -1 for all the NaN values.
#     * This option is good for categorical type columns / features.
# * If column is numeric in nature, fill with mean or median of that specific column.
#     * This option is good for numerical type columns / features.
# * Remove all the raws who have atleast 1 NaN value in any column.
#     * If total number of raws with NaN values is less, we can just remove those rows from our data.
# 
# Let's look if there are any missing values in our data.

# In[17]:


titanic_data.isna().sum()


# There are 177 NaN values in Age & 686 NaN values in Cabin column.<br>
# In Cabin more than 75% values are empty.<br>
# So, we will just remove that column.

# In[18]:


titanic_data.drop("Cabin", axis=1, inplace=True)
titanic_data.shape


# In[19]:


titanic_data.columns


# We can see `Cabin` column is removed from our data.<br><br>
# Now, `Age` is a numeric column.<br>
# So, let's fill NaN values by mean of all the other non-NaN values.

# In[20]:


age_mean = titanic_data['Age'].mean()
print(age_mean)


# We can fill all the NaN values using `fillna` 

# In[21]:


titanic_data['Age'].fillna(age_mean, inplace=True)


# In[22]:


titanic_data.isna().sum()


# There are just 2 NaN values in `Embarked` column.<br>
# We handle NaN values in `Embarked` column by filling most occuring value in that column.

# In[23]:


titanic_data['Embarked'].value_counts()


# In[24]:


titanic_data['Embarked'].fillna("S", inplace=True)


# In[25]:


titanic_data.isna().sum()


# Now, we can see, no NaN values are there in our whole data.

# Next step is **Feature Engineering**
# 
# ## 3.3
# 
# ### What is Feature Engineering?
# > Feature Engineering is creating more meaningful data out of existing data using our domain knowledge & comman sense.<br>
# 
# In other words, we try to create more relevant information for our ML models. <br>
# So, that our model can capture patterns in faster & better ways.
# 
# ### Now, this is a creative step. We need to use brain to create relevant features in the data.
# 
# Let's think.

# In[26]:


titanic_data.head(10)


# Let's once again look at what we have at hand.<br><br><br>
# `PassengerId` : Unique ID for each passenger.<br><br>
# `Survived` : Whether that passenger survived or not. (0 = No, 1 = Yes)<br><br>
# `Pclass` : Ticket class of passenger. (1 = Upper, 2 = Middle, 3 = Lower)<br><br>
# `Name` : Name of passenger<br><br>
# `Sex` : Gender of passenger<br><br>
# `Age` : Age of passenger<br><br>
# `SibSp` : # of siblings / spouses aboard the Titanic of passenger<br><br>
# `Parch` : # of parents / children aboard the Titanic of passenger<br><br>
# `Ticket` : Ticket number of passenger<br><br>
# `Fare` : Ticket amount / passenger fare.<br><br>
# `Embarked` : Port of Embarkation of passenger. (C = Cherbourg, Q = Queenstown, S = Southampton)<br><br>
# 
# ### How can we use these columns to create more relevant information?
# 
# Let's use `SibSp` & `Parch` to create a `total_family_members` feature.

# In[27]:


titanic_data['total_family_members'] = titanic_data['Parch'] + titanic_data['SibSp'] + 1

# if total family size is 1, person is alone.
titanic_data['is_alone'] = titanic_data['total_family_members'].apply(lambda x: 0 if x > 1 else 1)

titanic_data.head(10)


# In[28]:


sns.barplot(titanic_data['total_family_members'], titanic_data['Survived'])


# Interesting.<br>
# People with total_family_members = 4 have more than 70% chances of survival !<br><br>

# In[29]:


sns.barplot(titanic_data['is_alone'], titanic_data['Survived'])


# People with family have 20% higher chance of survival than people travelling alone !!
# 
# `Age` column also can be used to create partitions.<br>
# We can use `apply` function to `Age` column to create new column `age_group`<br>
# Like..

# In[30]:


def age_to_group(age):
    if 0 < age < 12:
        # children
        return 0
    elif 12 <= age < 50:
        # adult
        return 1
    elif age >= 50:
        # elderly people
        return 2
    
titanic_data['age_group'] = titanic_data['Age'].apply(age_to_group)
titanic_data.head(10)


# ### Why this age_group feature is useful ?
# Let's see..

# In[31]:


sns.barplot(titanic_data['age_group'], titanic_data['Survived']);


# `0` i.e. children have higher survival rate compared to adults & elderly people.<br>
# This data may become useful to our model.<br>
# 
# ### Can you think of any way we can use `name` column ?
# We can capture name title like Mr. Ms. Miss. etc.

# In[32]:


titanic_data['name_title'] = titanic_data['Name'].str.extract('([A-Za-z]+)\.', expand=False)
titanic_data.head()


# In[33]:


titanic_data['name_title'].value_counts()


# In[34]:


def clean_name_title(val):
    if val in ['Rev', 'Col', 'Mlle', 'Mme', 'Ms', 'Sir', 'Lady', 'Don', 'Jonkheer', 'Countess', 'Capt']:
        return 'RARE'
    else:
        return val

titanic_data['name_title'] = titanic_data['name_title'].apply(clean_name_title)
titanic_data['name_title'].value_counts()


# In[35]:


sns.barplot(titanic_data['name_title'], titanic_data['Survived']);


# People with `Mrs` & `Miss` titles i.e. females have high chances of survival.<br>
# But in males, with `Master` title, you have higher chances of survival !<br><br>

# In[36]:


titanic_data.head(10)


# Let's drop columns which are not useful to us as of now.<br>

# In[37]:


# save the target column 
target = titanic_data['Survived'].tolist()

titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket'], axis=1, inplace=True)


# In[38]:


titanic_data.head()


# ## 3.4 Convert all the data into numeric form
# 
# We can see, `Sex`, `Embarked` & `name_title` are not in numeric form.<br>
# Let's convert them via LabelEncoder from sci-kit learn.

# In[39]:


le = preprocessing.LabelEncoder()
titanic_data['Sex'] = le.fit_transform(titanic_data['Sex'])
titanic_data['Embarked'] = le.fit_transform(titanic_data['Embarked'])
titanic_data['name_title'] = le.fit_transform(titanic_data['name_title'])
titanic_data.head()


# Now, we have everything in numbers !!

# # 4. Train a machine learning model.
# 
# In this step, we choose a ML model & train it one the data we have.<br>
# For this lesson, we will use basic `LogisticRegression` model.<br>
# 
# But first of all, let's split our data into training & validation part.<br>
# There's `train_test_split` from sci-kit learn.

# In[40]:


train_data, val_data, train_target, val_target = train_test_split(titanic_data, target, test_size=0.2)
train_data.shape, val_data.shape, len(train_target), len(val_target)


# We have our training data & validation data.<br>
# We have randomly choosen 20% of the all the rows on which we will check our model's performance.

# In[41]:


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# We fix all the random seed so that, we can reproduce the results.
seed_everything(2020)


# In[42]:


# Train the LogisticRegression model.

model = LogisticRegression()
model.fit(train_data, train_target)


# Training is done.<br>
# We have trainied our Logistic Regression model.<br><br>
# 
# # 5. Validate the trained model i.e. checking it's performance on unseen data.
# 
# It's called "unseen" because our ML model have never seen this data.<br>
# It's kind of a test for it.<br>
# Where it's performance will be checked on data which it have never seen or train.

# In[43]:


# Predict labels on Validation data which model have never seen before.

val_predictions = model.predict(val_data)
len(val_predictions)


# In[44]:


# first 10 values of validation_predictions
val_predictions[:10]


# In[45]:


# Calculate the accuracy score on validation data.
# We already have correct target information for them.

accuracy = accuracy_score(val_target, val_predictions)
accuracy


# ## Voila !!<br>
# 

# In[46]:


print("We got %.3f percent accuracy on our validation unseen data !!"%(accuracy*100))
print("We are %.3f correct in predicting whether a person will survice in Titanic crash !!"%(accuracy*100))


# ## How cool is that..!!
# 
# There's a lot can be done to improve performance.<br>
# But we will not do that as of now to keep things simple as of now.<br>

# # 6. If it performs good in validation, use model to predict future real world data.
# 
# ### Now, we can use this model to other people & predict if they were on Titanic ship in 1912 !! 

# # Summary
# 
# So, in this lesson, we saw, what a typical pipeline looks like in solving a machine learning(ML) problem.
# 
# 1. Open the data files.
# 2. Understand the data. What each column in the table means.
# 3. Preprocess data
#     * Remove the outliers.
#     * Fill `NaN` or `null` values. Sometimes, we also remove all the rows with `NaN` values.
#     * Feature engineering - Create new columns out of existing columns using our understanding.
#     * Converting data into numeric form if it's not.
# 4. Train a machine learning model.
# 5. Validate the trained model i.e. checking it's performance on unseen data.
# 6. If it performs good in validation, use model to predict future real world data.
# 
# 
# ## Upvote this kernel if you have learned something from it.
# ## Tell me if you have any kind of doubts / questions in comment section below.
# 
# ## In next lesson we will solve this same problem with deep learning.
# ## See you in the [next lesson](https://www.kaggle.com/prashantkikani/solving-the-titanic-problem-deep-learning-way) 👋

# In[ ]:




