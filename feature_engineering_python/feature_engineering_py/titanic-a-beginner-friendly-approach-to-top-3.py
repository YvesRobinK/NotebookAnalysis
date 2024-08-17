#!/usr/bin/env python
# coding: utf-8

# <font size=+3 color="#141774"><center><b>Titanic Competition: A Beginner-friendly Approach to Top 3% with Ensemble Learning 🛳️</b></center></font>
# 
# <img src="https://images.unsplash.com/photo-1542614370-156b709e78f8?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1350&q=80" width = 400>
# <center>Photo by Annie Spratt (Unsplash)</center>
# 
# # Introduction
# 
# Hello readers and welcome to my attempt at the [Titanic ML competition](https://www.kaggle.com/c/titanic) on Kaggle! In this notebook we will:
# 
# - Perform **Exploratory Data Analysis** (EDA) and gain insights on the factors that affected passenger survival,
# - Perform **Feature Engineering** to create better features and improve our models,
# - Built several **Machine Learning models** to predict whether a passenger survived the shipwreck.
# 
# I have included text to explain my reasoning/workflow and make this kernel as <font size=+0 color="#BF570F"><b>beginner friendly</b></font> as possible. I didn't go into much detail about Machine Learning concepts ('What is SVC?', 'what's k in k-Nearest Neighbors?' etc.) but you are welcome to ask me anything in the comments.
# 
# Please consider <font size=+0 color="red"><b>upvoting</b></font> if you found it useful! 🧐
#     
# <br>
# 
# **Table of Contents**
# 
# 1. [Introduction](#Introduction)
# 2. [Libraries](#Libraries)
# 3. [Getting the Data](#Getting-the-Data)
# 4. [A Quick Look at our Data](#A-Quick-Look-at-our-Data)
# 5. [Exploratory Data Analysis](#Exploratory-Data-Analysis)
# 6. [Preparing Data](#Preparing-Data)
# 7. [Building Machine Learning Models](#Building-Machine-Learning-Models)
# 8. [Conclusions](#Conclusions)

# # Libraries
# 
# We start by importing the necessary libraries and setting some parameters for the whole notebook (such as parameters for the plots, etc.). We will mainly use:
# 
# - Pandas for handling and analysing data,
# - Seaborn and Matplotlib for data visualization, and
# - Scikit-learn for building Machine Learning models.

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
pd.set_option('precision', 3)

import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import seaborn as sns
sns.set_style('dark')

mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 15
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler 
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

print ('Libraries Loaded!')


# # Getting the Data
# 
# The data has already been split into a training set ('train.csv') and a test set ('test.csv'). We can use the `read_csv()` method to load them as Pandas dataframes:

# In[2]:


train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')

print ('Dataframes loaded!')
print ('Training set: {} rows and {} columns'.format(train_df.shape[0], train_df.shape[1]))
print ('    Test set: {} rows and {} columns'.format(test_df.shape[0], test_df.shape[1]))


# The training set is labeled, i.e. we know the outcome for each passenger, hence the difference in the number of columns. 
# 
# We are going to merge the two dataframes into one. The new dataframe will have **NaN** in the 'Survived' column for instances of the test set:

# In[3]:


all_data = pd.concat([train_df, test_df])

print ('Combined set: {} rows and {} columns'.format(all_data.shape[0], all_data.shape[1]))
print ('\nSurvived?: ')
all_data['Survived'].value_counts(dropna = False)


# # A Quick Look at our Data
# 
# In this stage, we will temporarily forget the test set and focus on the training set.
# 
# We can take a look at the top five rows of the training set using the `head()` method:

# In[4]:


train_df.head()


# The meaning of each attribute is the following:
# 
# - **PassengerId**: the ID given to each passenger,
# - **Survived**: the target attribute (1 for passengers who survived, 0 for those who didn't),
# - **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd class),
# - **Name**, **Sex**, **Age**: self-explanatory,
# - **SibSp**: Number of siblings & spouses aboard the Titanic,
# - **Parch**: Number of parents & children aboard the Titanic,
# - **Ticket**: Ticket number, 
# - **Fare**: Passenger fare (in pounds),
# - **Cabin**: Passenger's cabin number, and
# - **Embarked**: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).
# 
# 'PassengerId' is unique to each passenger and can be dropped:

# In[5]:


train_df.drop('PassengerId', axis = 1, inplace = True)


# The `info()` method can give us valuable information such  as the type of each attribute and the number of missing values:

# In[6]:


train_df.info()


# The training set has 891 instances and 11 columns (10 attributes + the target attribute). 6 attributes are numerical, while 5 are categorical.
# 
# Let's take a closer look at the missing values:

# In[7]:


missing_counts = train_df.isnull().sum().sort_values(ascending = False)
percent = (train_df.isnull().sum()*100/train_df.shape[0]).sort_values(ascending = False)

missing_df = pd.concat([missing_counts, percent], axis = 1, keys = ['Counts', '%'])
print('Missing values: ')
display(missing_df.head().style.background_gradient(cmap = 'Reds', axis = 0))


# Replacing missing values in the 'Age' and 'Embarked' columns won't be that difficult. We could use the median and the most frequent value as a replacement, respectively. However, we will probably have to discard the 'Cabin' attribute since more than 75% of all values are missing.
# 
# The `describe()` method gives us a statistical summary of the numerical attributes:

# In[8]:


train_df.describe()


# The most important things to note are:
# 
# - Only **38%** of passenger **survived**,
# - The **mean age** is approximately **30** years old, while the **median** is **28** (therefore it won't matter much which one we use for imputation),
# - The median for both 'SibSp' and 'Parch' is 0 (most passengers were **alone**),
# - The mean fare is £32.20, and
# - These attributes have **different scales**, so we need to take care of that before feeding them to a Machine Learning algorithm. 
# 
# We can quickly visualize the difference in scales, by plotting a histogram for each numerical attribute.

# In[9]:


num_atts = ['Age', 'SibSp', 'Parch', 'Fare', 'Pclass']
train_df[num_atts].hist(figsize = (15, 6), color = 'steelblue', edgecolor = 'firebrick', linewidth = 1.5, layout = (2, 3));


# We can see that most of the passengers: 
# 
# - were **young** (age < 40),
# - boarded the ship **alone** (SibSp and Parch equal to 0), 
# - paid a **low fare** and boarded in the **3rd class**.

# # Exploratory Data Analysis
# 
# Let's have a look at (almost) all attributes in greater detail.
# 
# ## 1. Gender

# In[10]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (11, 4))

sns.countplot(x = 'Sex', hue = 'Survived', data = train_df,  palette = 'tab20', ax = ax1) 
ax1.set_title('Count of (non-)Survivors by Gender')
ax1.set_xlabel('Gender')
ax1.set_ylabel('Number of Passenger')
ax1.legend(labels = ['Deceased', 'Survived'])

sns.barplot(x = 'Sex', y = 'Survived', data = train_df,  palette = ['#94BFA7', '#FFC49B'], ci = None, ax = ax2)
ax2.set_title('Survival Rate by Gender')
ax2.set_xlabel('Gender')
ax2.set_ylabel('Survival Rate');


# In[11]:


pd.crosstab(train_df['Sex'], train_df['Survived'], normalize = 'index')


# There were more men than women on board. However, **more women survived** the shipwreck (the survival rate is almost 75% for women compared to only 20% for men!).
# 
# We can read in [wikipedia](https://en.wikipedia.org/wiki/Titanic) that a "women and children first" protocol was implemented for boarding lifeboats. Therefore,  apart from women, younger people had an advantage. With that in mind, let's see the age distribution.
# 
# ## 2. Age

# In[12]:


men = train_df[train_df['Sex']  == 'male']
women = train_df[train_df['Sex']  == 'female']


# In[13]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (13, 4))

sns.distplot(train_df[train_df['Survived'] == 1]['Age'].dropna(), bins = 20, label = 'Survived', ax = ax1, kde = False)
sns.distplot(train_df[train_df['Survived'] == 0]['Age'].dropna(), bins = 20, label = 'Deceased', ax = ax1, kde = False)
ax1.legend()
ax1.set_title('Age Distribution - All Passengers')

sns.distplot(women[women['Survived'] == 1]['Age'].dropna(), bins = 20, label = 'Survived', ax = ax2, kde = False)
sns.distplot(women[women['Survived'] == 0]['Age'].dropna(), bins = 20, label = 'Deceased', ax = ax2, kde = False)
ax2.legend()
ax2.set_title('Age Distribution - Women')

sns.distplot(men[men['Survived'] == 1]['Age'].dropna(), bins = 20, label = 'Survived', ax = ax3, kde = False)
sns.distplot(men[men['Survived'] == 0]['Age'].dropna(), bins = 20, label = 'Deceased', ax = ax3, kde = False)
ax3.legend()
ax3.set_title('Age Distribution - Men')

plt.tight_layout();


# It is evident that **different age groups** had very **different survival rates**. For instance, both genders display a higher probability of survival between the ages of 15 and 45. Also, the spike at young ages (0-4) shows that infants and young children have higher odds of survival.
# 
# Since survival seems to favour certain age groups, it might be useful to **bin** 'Age' before feeding it to an algorithm. We will pick an interval of 15 years.

# In[14]:


# train_df['Age_Bin'] = pd.qcut(train_df['Age'], 4)  # Quantile-based discretization
train_df['Age_Bin'] = (train_df['Age']//15)*15
train_df[['Age_Bin', 'Survived']].groupby(['Age_Bin']).mean()


# ## 3. Port of Embarkation

# In[15]:


sns.countplot(x = 'Embarked', hue = 'Survived', data = train_df,  palette = 'tab20') 
plt.ylabel('Number of Passenger')
plt.title('Count of (non-)Survivors by Port of Embarkation')
plt.legend(['Deceased', 'Survived']);


# Most passengers embarked from Southampton, the port from which the ship started its voyage. It has by far the highest count for both survivors and non-survivors. Cherbourg has the second largest number of passengers and interestingly, more than half of them survived.
# 
# Looking at the data, I wasn't confident that this attribute would be useful. After all, the ship sank at the same point and at the same time for all passengers so it doesn't really matter where they embarked. However, I decided to test it anyway and observed that the performance of my models got worse when I included it, therefore we can **ignore it**.

# ## 4. Pclass

# In[16]:


print ('Number of passengers in each class:')
train_df['Pclass'].value_counts()


# In[17]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))

sns.countplot(x = 'Pclass', hue = 'Survived', data = train_df,  palette = 'tab20', ax = ax1) 
ax1.legend(['Deceased', 'Survived'])
ax1.set_title('Count of (non-)Survivors by Class')
ax1.set_ylabel('Number of Passengers')

sns.barplot(x = 'Pclass', y = 'Survived', data = train_df,  palette = ['#C98BB9', '#F7D4BC', '#B5E2FA'], ci = None, ax = ax2)
ax2.set_title('Survival Rate by Class')
ax2.set_ylabel('Survival Rate');


# More than 50% of passengers boarded in the 3rd class. Nevertheless, **survival** favours the **wealthy** as shown in the right figure (the survival rate increases as we move from 3rd to 1st class).

# ## 5. Fare
# 
# One would assume that fare is closely related to class. Let's plot a boxplot for the distribution of Fare values across classes and a histogram for survival:

# In[18]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))

sns.boxplot(x = 'Pclass', y = 'Fare', data = train_df, palette = 'tab20', ax = ax1)
ax1.set_title('Distribution of Fares by Class')

sns.distplot(train_df[train_df['Survived'] == 1]['Fare'], label = 'Survived', ax = ax2)
sns.distplot(train_df[train_df['Survived'] == 0]['Fare'], label = 'Not Survived', ax = ax2)
ax2.set_title('Distribution of Fares for (non-)Survivors')
ax2.set_xlim([-20, 200])
ax2.legend();


# It's not a surprise that people in class 1 paid more than the other two classes. As we already saw in the comparison of the classes, a **higher fare** leads to a **higher chance of survival**.
# 
# As with 'Age', we can benefit from **bining** the fare value. I prefer quantile-based discretization with 5 quantiles for this attribute.

# In[19]:


train_df['Fare_Bin'] = pd.qcut(train_df['Fare'], 5)
train_df[['Fare_Bin', 'Survived']].groupby(['Fare_Bin']).mean()


# 
# 
# ## 6. SibSp and Parch
# 
# Someone could argue that having relatives could influence a passenger's odds of surviving. Let's test that:

# In[20]:


alone = train_df[(train_df['SibSp'] == 0) & (train_df['Parch'] == 0)]
not_alone = train_df[(train_df['SibSp'] != 0) | (train_df['Parch'] != 0)]


# In[21]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))

sns.countplot(x = 'Survived', data = alone,  palette = 'tab20', ax = ax1) 
ax1.set_title('Count of Alone (non-)Survivors')
ax1.set_xlabel('')
ax1.set_xticklabels(['Deceased', 'Survived'])
ax1.set_ylabel('Number of Passengers')

sns.countplot(x = 'Survived', data = not_alone,  palette = 'tab20', ax = ax2) 
ax2.set_title('Count of (non-)Survivors with Family Onboard')
ax2.set_xlabel('')
ax2.set_xticklabels(['Deceased', 'Survived'])
ax2.set_ylabel('Number of Passengers')

plt.tight_layout();


# Having **relatives** on board **increases your chances of survival**. 
# 
# Is the number of relative relevant? We can create a new attribute for the number of relatives on board and test that:

# In[22]:


train_df['Relatives'] = train_df['SibSp'] + train_df['Parch']
# train_df[['Relatives', 'Survived']].groupby(['Relatives']).mean()

sns.factorplot('Relatives', 'Survived', data = train_df, color = 'firebrick', aspect = 1.5)
plt.title('Survival rate by Number of Relatives Onboard');


# Having 1 to 3 relatives can actually increase you chances of survival.

# ## 7. Name/Title
# 
# Finally, we could see if a person's title (Mr, Miss etc.) plays a role in survival. I used Ken's [code](https://www.kaggle.com/kenjee/titanic-project-example) to extract the title for each instance. I then replaced rare titles with more common ones.

# In[23]:


train_df['Title'] = train_df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

train_df['Title'].replace({'Mlle': 'Miss', 'Mme': 'Mrs', 'Ms': 'Miss'}, inplace = True)
train_df['Title'].replace(['Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col', 'Capt', 'the Countess', 'Jonkheer'],
                           'Rare Title', inplace = True)
train_df['Title'].value_counts()


# In[24]:


cols = ['#067BC2', '#84BCDA', '#ECC30B', '#F37748', '#D56062']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 4))

sns.countplot(x = 'Title', data = train_df,  palette = cols, ax = ax1)
ax1.set_title('Passenger Count by Title')
ax1.set_ylabel('Number of Passengers')

sns.barplot(x = 'Title', y = 'Survived', data = train_df,  palette = cols, ci = None, ax = ax2)
ax2.set_title('Survival Rate by Title')
ax2.set_ylabel('Survival Rate');


# We have already talked about the fact that women (Mrs or Miss) had higher odds of survival. What's really interesting is that **Masters** and people with a **Rare Title** have indeed a **higher chance of survival** compared to 'common' men (Mr).

# ## 8. Others
# 
# ### Cabin
# 
# As we saw earlier, 3 out of 4 instances in the training set have a missing value for 'Cabin'. Additionally, it has a large number of unique values among the existing (non-NaN) values:

# In[25]:


print ('Cabin:\n  Number of existing values: ', train_df['Cabin'].notnull().sum())
print ('    Number of unique values: ', train_df['Cabin'].nunique())


# Consequently, we can safely discard it. You can have a look at this [notebook](https://www.kaggle.com/ccastleberry/titanic-cabin-features) for more information about the 'Cabin' feature.
# 
# ### Ticket/Family Survival
# 
# We will indirectly use the 'Ticket' attribute to engineer a new feauture called 'Family_Survival'. The idea comes from S.Xu's [kernel](https://www.kaggle.com/shunjiangxu/blood-is-thicker-than-water-friendship-forever), in which he groups families and people with the same tickets together and searches for info based on that. A cleaner version of the code is taken from Konstantin's [kernel](https://www.kaggle.com/konstantinmasich/titanic-0-82-0-83/notebook) (see next section).

# ## 9. Summary
# 
# 
# |      Attribute      | Important |            Action           |
# |:-------------------:|:---------:|:---------------------------:|
# |     PassengerId     |     No    |           Discard           |
# |         Sex         |    Yes    |            Encode           |
# |         Age         |    Yes    |        Bin and Encode       |
# | Port of Embarkation |     No    |           Discard           |
# |        Pclass       |    Yes    |              -              |
# |         Fare        |    Yes    |        Bin and Encode       |
# |   SibSp and Parch   |    Yes    |     Engineer 'Relatives'    |
# |         Name        |    Yes    | Engineer 'Title' and Encode |
# |        Cabin        |     No    |           Discard           |
# |        Ticket       |    Yes    |  Engineer 'Family_Survival' |

# # Preparing Data
# 
# In this section, we will prepare the dataframe before we build any machine learning algorithm. We will use the combined dataframe so that both the train and the test set get processed at the same time. Another alternative would be to use pipilines.
# 
# Steps:
# 
# 1) Replace missing values in 'Age' and 'Fare' with the corresponding median of the train set. Note that the test set has one missing value for 'Fare' (which we can easily check by calling test_df.isnull().sum()).

# In[26]:


all_data['Age'] = all_data['Age'].fillna(train_df['Age'].median())
all_data['Fare'] = all_data['Fare'].fillna(train_df['Fare'].median())
print ('Done!')


# 2) Add the new attributes ('Family_Survival', 'Age_Bin', 'Fare_Bin', 'Relatives', 'Title).

# In[27]:


# Again, the code for 'Family_Survival' comes from this kernel:
# https://www.kaggle.com/konstantinmasich/titanic-0-82-0-83/notebook

all_data['Last_Name'] = all_data['Name'].apply(lambda x: str.split(x, ',')[0])
all_data['Fare'].fillna(all_data['Fare'].mean(), inplace = True)

default_sr_value = 0.5
all_data['Family_Survival'] = default_sr_value

for grp, grp_df in all_data[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId', 'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
    
    if (len(grp_df) != 1):  # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            
            if (smax == 1.0):
                all_data.loc[all_data['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                all_data.loc[all_data['PassengerId'] == passID, 'Family_Survival'] = 0

for _, grp_df in all_data.groupby('Ticket'):
    
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                
                if (smax == 1.0):
                    all_data.loc[all_data['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    all_data.loc[all_data['PassengerId'] == passID, 'Family_Survival'] = 0
                    
#####################################################################################
all_data['Age_Bin'] = (all_data['Age']//15)*15
all_data['Fare_Bin'] = pd.qcut(all_data['Fare'], 5)
all_data['Relatives'] = all_data['SibSp'] + all_data['Parch']
#####################################################################################
all_data['Title'] = all_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
all_data['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'}, inplace = True)
all_data['Title'].replace(['Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col', 'Capt', 'the Countess', 'Jonkheer', 'Dona'],
                           'Rare Title', inplace = True)    

print ('Done!')


# 3) Use scikit-learn's `LabelEncoder()` to encode 'Fare_Bin', 'Age_Bin', 'Title' and 'Sex'.

# In[28]:


all_data['Fare_Bin'] = LabelEncoder().fit_transform(all_data['Fare_Bin'])
all_data['Age_Bin'] = LabelEncoder().fit_transform(all_data['Age_Bin'])
all_data['Title_Bin'] = LabelEncoder().fit_transform(all_data['Title'])
all_data['Sex'] = LabelEncoder().fit_transform(all_data['Sex'])

print ('Done!')


# 4) Discard all unnecessary attributes.

# In[29]:


all_data.drop(['PassengerId', 'Age', 'Fare', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Title', 'Last_Name', 'Embarked'], axis = 1, inplace = True)

print ('Done!')
print ('Modified dataset: ')
all_data.head()


# 5) Split the combined dataset into train and test set and scale each feature vector.

# In[30]:


train_df = all_data[:891]

X_train = train_df.drop('Survived', 1)
y_train = train_df['Survived']

#######################################################

test_df = all_data[891:]

X_test = test_df.copy()
X_test.drop('Survived', axis = 1, inplace = True)
print ('Splitting: Done!')


# In[31]:


std_scaler = StandardScaler()

X_train_scaled = std_scaler.fit_transform(X_train)  # fit_transform the X_train
X_test_scaled = std_scaler.transform(X_test)        # only transform the X_test

print ('Scaling: Done!')


# # Building Machine Learning Models
# 
# ## Baseline Models
# 
# The aim of this subsection is to calculate the **baseline performance** of 8 different estimators/classifiers on the training set. This will enable us to later see how tuning improves each of these models. 
# 
# The classifiers are:
# 
# 1) Gaussian Naive Bayes , <br>
# 2) Logistic Regression, <br>
# 3) K-Nearest Neighbor Classifier, <br>
# 4) Support Vector Classifier, <br>
# 5) Decision Tree Classifier, <br>
# 6) Random Forest Classifier, <br>
# 7) Xtreme Gradient Boosting Classifier, and <br>
# 8) AdaBoost classifier.
# 
# I won't go into detail about how these classifiers work. You can read more in this excellent [book](https://www.oreilly.com/library/view/hands-on-machine-learning/9781491962282/).
# 
# For the baseline models, we will use their **default parameters** and evaluate their (mean) accuracy by performing **k-fold cross validation**. 
# 
# The idea behind k-fold cross validation, which is illustrated in the following figure, is simple:
# it splits the (training) set into k subsets/folds, trains the models using k-1 folds and evaluates the model on the remaining one fold. This process is repeated until every fold is tested once. 
# 
# <img src="https://scikit-learn.org/stable/_images/grid_search_cross_validation.png" width = 400>
# <center> Taken from the official documentation on scikit-learn's website </center>
# 
# <br>
# 
# We can implement cross validation by using the `cross_val_score()` method from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html). We will use k = 5 folds.

# In[32]:


random_state = 1

# Step 1: create a list containing all estimators with their default parameters
clf_list = [GaussianNB(), 
            LogisticRegression(random_state = random_state),
            KNeighborsClassifier(), 
            SVC(random_state = random_state, probability = True),
            DecisionTreeClassifier(random_state = random_state), 
            RandomForestClassifier(random_state = random_state),
            XGBClassifier(random_state = random_state), 
            AdaBoostClassifier(base_estimator = DecisionTreeClassifier(random_state = random_state), random_state = random_state)]


# Step 2: calculate the cv mean and standard deviation for each one of them
cv_base_mean, cv_std = [], []
for clf in clf_list:  
    
    cv = cross_val_score(clf, X_train_scaled, y = y_train, scoring = 'accuracy', cv = 5, n_jobs = -1)
    
    cv_base_mean.append(cv.mean())
    cv_std.append(cv.std())

    
# Step 3: create a dataframe and plot the mean with error bars
cv_total = pd.DataFrame({'Algorithm': ['Gaussian Naive Bayes', 'Logistic Regression', 'k-Nearest Neighboors', 'SVC', 'Decision Tree', 'Random Forest', 'XGB Classifier', 'AdaBoost Classifier'],
                         'CV-Means': cv_base_mean, 
                         'CV-Errors': cv_std})

sns.barplot('CV-Means', 'Algorithm', data = cv_total, palette = 'Paired', orient = 'h', **{'xerr': cv_std})
plt.xlabel('Mean Accuracy')
plt.title('Cross Validation Scores')
plt.xlim([0.725, 0.88])
plt.axvline(x = 0.80, color = 'firebrick', linestyle = '--');


# All estimators have a score **above 80%**, with SVC scoring the highest (85%). 
# 
# We can combine the predictions of all these base classifiers and see if we get better predictive performance compared to each constituent individual classifier. This is the main motivation behind **Ensemble Learning**.
# 
# There are two options (see [here](https://www.oreilly.com/library/view/machine-learning-for/9781783980284/47c32d8b-7b01-4696-8043-3f8472e3a447.xhtml) and [here](https://www.oreilly.com/library/view/hands-on-machine-learning/9781491962282/)):<br>
# 1) **Hard Voting**: A hard voting classifier counts the votes of each estimator in the ensemble and picks the class that gets the most votes. In other words, the majority wins. <br>
# 2) **Soft Voting**: Every individual classifier provides a probability value that a specific data point belongs to a particular target class. The predictions are weighted by the classifier's importance and summed up. Then the target label with the greatest sum of weighted probabilities wins the vote.

# In[33]:


estimators = [('gnb', clf_list[0]), ('lr', clf_list[1]),
              ('knn', clf_list[2]), ('svc', clf_list[3]),
              ('dt', clf_list[4]), ('rf', clf_list[5]),
              ('xgb', clf_list[6]), ('ada', clf_list[7])]

base_voting_hard = VotingClassifier(estimators = estimators , voting = 'hard')
base_voting_soft = VotingClassifier(estimators = estimators , voting = 'soft') 

cv_hard = cross_val_score(base_voting_hard, X_train_scaled, y_train, cv = 5)
cv_soft = cross_val_score(base_voting_soft, X_train_scaled, y_train, cv = 5)

print ('Baseline Models - Ensemble\n--------------------------')
print ('Hard Voting: {}%'.format(np.round(cv_hard.mean()*100, 1)))
print ('Soft Voting: {}%'.format(np.round(cv_soft.mean()*100, 1)))


# The ensemble has indeed a higher (cv) score than most individual classifiers. We can also try dropping some classifiers and see if it improves more.

# In[34]:


base_voting_hard.fit(X_train_scaled, y_train)
base_voting_soft.fit(X_train_scaled, y_train)

y_pred_base_hard = base_voting_hard.predict(X_test_scaled)
y_pred_base_soft = base_voting_hard.predict(X_test_scaled)


# ## Model Tuning
# 
# We are ready to tune hyperparameters using grid search and see if performance improves. For more information about hyperparemeters, please visit the corresponding [documentation](https://scikit-learn.org/stable/). 
# 
# We write a simple performance reporting function (taken from [Ken](https://www.kaggle.com/kenjee/titanic-project-example)'s kernel).

# In[35]:


cv_means_tuned = [np.nan] # we can't actually tune the GNB classifier, so we fill its element with NaN

#simple performance reporting function
def clf_performance(classifier, model_name):
    print(model_name)
    print('-------------------------------')
    print('   Best Score: ' + str(classifier.best_score_))
    print('   Best Parameters: ' + str(classifier.best_params_))
    
    cv_means_tuned.append(classifier.best_score_)


# ### Logistic Regression

# In[36]:


lr = LogisticRegression()

param_grid = {'max_iter' : [100],
              'penalty' : ['l1', 'l2'],
              'C' : np.logspace(-2, 2, 20),
              'solver' : ['lbfgs', 'liblinear']}

clf_lr = GridSearchCV(lr, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)

best_clf_lr = clf_lr.fit(X_train_scaled, y_train)
clf_performance(best_clf_lr, 'Logistic Regression')


# ### k-Nearest Neighbors

# In[37]:


# n_neighbors = np.concatenate((np.arange(3, 30, 1), np.arange(22, 32, 2)))

knn = KNeighborsClassifier()
param_grid = {'n_neighbors' : np.arange(3, 30, 2),
              'weights': ['uniform', 'distance'],
              'algorithm': ['auto'],
              'p': [1, 2]}

clf_knn = GridSearchCV(knn, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_knn = clf_knn.fit(X_train_scaled, y_train)
clf_performance(best_clf_knn, 'KNN')


# ### Support Vector Classifier

# In[38]:


svc = SVC(probability = True)
param_grid = tuned_parameters = [{'kernel': ['rbf'], 
                                  'gamma': [0.01, 0.1, 0.5, 1, 2, 5],
                                  'C': [.1, 1, 2, 5]},
                                 {'kernel': ['linear'], 
                                  'C': [.1, 1, 2, 10]},
                                 {'kernel': ['poly'], 
                                  'degree' : [2, 3, 4, 5], 
                                  'C': [.1, 1, 10]}]

clf_svc = GridSearchCV(svc, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_svc = clf_svc.fit(X_train_scaled, y_train)
clf_performance(best_clf_svc, 'SVC')


# ### Decision Tree

# In[39]:


dt = DecisionTreeClassifier(random_state = 1)
param_grid = {'max_depth': [3, 5, 10, 20, 50],
              'criterion': ['entropy', 'gini'],
              'min_samples_split': [5, 10, 15, 30],
              'max_features': [None, 'auto', 'sqrt', 'log2']}
                                  
clf_dt = GridSearchCV(dt, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_dt = clf_dt.fit(X_train_scaled, y_train)
clf_performance(best_clf_dt, 'Decision Tree')


# <br>
# 
# Estimators such as Random Forests, XGBoost and AdaBoost Clasiffiers allow us to see the **importance** of each feature.
# 
# ### Random Forest Classifier

# In[40]:


rf = RandomForestClassifier(random_state = 42)
param_grid = {'n_estimators': [50, 150, 300, 450],
              'criterion': ['entropy'],
              'bootstrap': [True],
              'max_depth': [3, 5, 10],
              'max_features': ['auto','sqrt'],
              'min_samples_leaf': [2, 3],
              'min_samples_split': [2, 3]}
                                  
clf_rf = GridSearchCV(rf, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_rf = clf_rf.fit(X_train_scaled, y_train)
clf_performance(best_clf_rf, 'Random Forest')


# In[41]:


best_rf = best_clf_rf.best_estimator_

importances = pd.DataFrame({'Feature': X_train.columns,
                            'Importance': np.round(best_rf.feature_importances_, 3)})

importances = importances.sort_values('Importance', ascending = True).set_index('Feature')

importances.plot.barh(color = 'steelblue', edgecolor = 'firebrick', legend=False)
plt.title('Random Forest Classifier')
plt.xlabel('Importance');


# ### XGBoost Classifier

# In[42]:


xgb = XGBClassifier(random_state = 42)

param_grid = {'n_estimators': [15, 25, 50, 100],
              'colsample_bytree': [0.65, 0.75, 0.80],
              'max_depth': [None],
              'reg_alpha': [1],
              'reg_lambda': [1, 2, 5],
              'subsample': [0.50, 0.75, 1.00],
              'learning_rate': [0.01, 0.1, 0.5],
              'gamma': [0.5, 1, 2, 5],
              'min_child_weight': [0.01],
              'sampling_method': ['uniform']}

clf_xgb = GridSearchCV(xgb, param_grid = param_grid, cv = 3, verbose = True, n_jobs = -1)
best_clf_xgb = clf_xgb.fit(X_train_scaled, y_train)
clf_performance(best_clf_xgb, 'XGB')


# In[43]:


best_xgb = best_clf_xgb.best_estimator_

importances = pd.DataFrame({'Feature': X_train.columns,
                            'Importance': np.round(best_xgb.feature_importances_, 3)})

importances = importances.sort_values('Importance', ascending = True).set_index('Feature')

importances.plot.barh(color = 'darkgray', edgecolor = 'firebrick', legend = False)
plt.title('XGBoost Classifier')
plt.xlabel('Importance');


# ### AdaBoost

# In[44]:


adaDTC = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(random_state = random_state), random_state=random_state)

param_grid = {'algorithm': ['SAMME', 'SAMME.R'],
              'base_estimator__criterion' : ['gini', 'entropy'],
              'base_estimator__splitter' : ['best', 'random'],
              'n_estimators': [2, 5, 10, 50],
              'learning_rate': [0.01, 0.1, 0.2, 0.3, 1, 2]}

clf_ada = GridSearchCV(adaDTC, param_grid = param_grid, cv = 5, scoring = 'accuracy', n_jobs = -1, verbose = 1)
best_clf_ada = clf_ada.fit(X_train_scaled, y_train)

clf_performance(best_clf_ada, 'AdaBost')


# In[45]:


best_ada = best_clf_ada.best_estimator_
importances = pd.DataFrame({'Feature': X_train.columns,
                            'Importance': np.round(best_ada.feature_importances_, 3)})

importances = importances.sort_values('Importance', ascending = True).set_index('Feature')

importances.plot.barh(color = 'cadetblue', edgecolor = 'firebrick', legend = False)
plt.title('AdaBoost Classifier')
plt.xlabel('Importance');


# The results are:

# In[46]:


cv_total = pd.DataFrame({'Algorithm': ['Gaussian Naive Bayes', 'Logistic Regression', 'k-Nearest Neighboors', 'SVC', 'Decision Tree', 'Random Forest', 'XGB Classifier', 'AdaBoost Classifier'],
                         'Baseline': cv_base_mean, 
                         'Tuned Performance': cv_means_tuned})

cv_total


#  We will now build the **final ensembles** 😌:

# In[47]:


best_lr = best_clf_lr.best_estimator_
best_knn = best_clf_knn.best_estimator_
best_svc = best_clf_svc.best_estimator_
best_dt = best_clf_dt.best_estimator_
best_rf = best_clf_rf.best_estimator_
best_xgb = best_clf_xgb.best_estimator_
# best_ada = best_clf_ada.best_estimator_  # didn't help me in my final ensemble

estimators = [('lr', best_lr), ('knn', best_knn), ('svc', best_svc),
              ('rf', best_rf), ('xgb', best_xgb), ('dt', best_dt)]

tuned_voting_hard = VotingClassifier(estimators = estimators, voting = 'hard', n_jobs = -1)
tuned_voting_soft = VotingClassifier(estimators = estimators, voting = 'soft', n_jobs = -1)

tuned_voting_hard.fit(X_train_scaled, y_train)
tuned_voting_soft.fit(X_train_scaled, y_train)

cv_hard = cross_val_score(tuned_voting_hard, X_train_scaled, y_train, cv = 5)
cv_soft = cross_val_score(tuned_voting_soft, X_train_scaled, y_train, cv = 5)

print ('Tuned Models - Ensemble\n-----------------------')
print ('Hard Voting: {}%'.format(np.round(cv_hard.mean()*100, 2)))
print ('Soft Voting: {}%'.format(np.round(cv_soft.mean()*100, 2)))

y_pred_tuned_hd = tuned_voting_hard.predict(X_test_scaled).astype(int)
y_pred_tuned_sf = tuned_voting_soft.predict(X_test_scaled).astype(int)


# ## Submission

# In[48]:


test_df = pd.DataFrame(pd.read_csv('../input/titanic/test.csv')['PassengerId'])

pd.DataFrame(data = {'PassengerId': test_df.PassengerId, 
                     'Survived': y_pred_base_hard.astype(int)}).to_csv('01-Baseline_Hard_voting.csv', index = False)

pd.DataFrame(data = {'PassengerId': test_df.PassengerId, 
                     'Survived': y_pred_base_soft.astype(int)}).to_csv('02-Baseline_Soft_voting.csv', index = False)

pd.DataFrame(data = {'PassengerId': test_df.PassengerId, 
                     'Survived': y_pred_tuned_hd.astype(int)}).to_csv('03-Tuned_Hard_Voting.csv', index = False)

pd.DataFrame(data = {'PassengerId': test_df.PassengerId, 
                     'Survived': y_pred_tuned_sf.astype(int)}).to_csv('04-Tuned_Soft_Voting.csv', index = False)


# # Conclusions
# 
# This notebook came to an end! We can summarise it by mentioning a few points:
# 
# - **EDA** helped us understand where to **focus**. Factors such as a passenger's gender or/ and title showed that the [initial assumption](https://www.kaggle.com/c/titanic) was actually true and even though 'there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others'.
# - We should not rely completely on given feautures since we could benefit from **engineering new ones**.
# - Building Machine Learning models requires a lot of **tweaking** of the parameter before we get a good/optimal result. **Ensemble learning** can usually help us towards this direction.
# - Lastly, I would like to mention that looking at other people's work can give us new ideas and inspiration for our own project. Just make sure you **give credit** and don't copy a whole kernel.
# 
# Feel free to ask me anything in the comment section.
# 
# Please <font size=+0 color="red"><b>upvote</b></font> if you liked this notebook! Thank you! 😉
