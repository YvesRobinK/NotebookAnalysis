#!/usr/bin/env python
# coding: utf-8

# # Titanic - Machine Learning from Disaster
# ---

# Copyright [2021] [Data Scientist & ML Engineer: [Ahmed](https://www.kaggle.com/dsxavier/)]
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ## An Overview

# The sinking of the Titanic is one of the most infamous shipwrecks in history.
# 
# On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.
# 
# While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
# 
# In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

# ## [Data Dictionary](https://www.kaggle.com/c/titanic/data#:~:text=should%20look%20like.-,data%20dictionary,-Variable)
# 

# <table>
# <thead>
#   <tr>
#     <th>Variable</th>
#     <th>Definition</th>
#     <th>Key</th>
#   </tr>
# </thead>
# <tbody>
#   <tr>
#     <td>survival</td>
#     <td>Survival</td>
#     <td>0 = No, 1 = Yes</td>
#   </tr>
#   <tr>
#     <td>pclass</td>
#     <td>Ticket class</td>
#     <td>1 = 1st, 2 = 2nd, 3 = 3rd</td>
#   </tr>
#   <tr>
#     <td>sex</td>
#     <td>Sex</td>
#     <td></td>
#   </tr>
#   <tr>
#     <td>Age</td>
#     <td>Age in years</td>
#     <td></td>
#   </tr>
#   <tr>
#     <td>sibsp</td>
#     <td># of siblings / spouses aboard the Titanic</td>
#     <td></td>
#   </tr>
#   <tr>
#     <td>parch</td>
#     <td># of parents / children aboard the Titanic</td>
#     <td></td>
#   </tr>
#   <tr>
#     <td>ticket</td>
#     <td>Ticket number</td>
#     <td></td>
#   </tr>
#   <tr>
#     <td>fare</td>
#     <td>Passenger fare</td>
#     <td></td>
#   </tr>
#   <tr>
#     <td>cabin</td>
#     <td>Cabin number</td>
#     <td></td>
#   </tr>
#   <tr>
#     <td>embarked</td>
#     <td>Port of Embarkation</td>
#     <td>C = Cherbourg, Q = Queenstown, S = Southampton</td>
#   </tr>
# </tbody>
# </table>

# ---

# ## Table of Contents

# <table>
# <thead>
#   <tr>
#       <th><a href='#Table-of-Contents'>Table of Contents</a></th>
#     <th></th>
#     <th></th>
#   </tr>
# </thead>
# <tbody>
#   <tr>
#       <td><a href='#Dependencies'>Dependencies</a><br></td>
#     <td></td>
#     <td></td>
#   </tr>
#   <tr>
#     <td></td>
#       <td><a href='#(A)-Import-Libraries'>(A) Import Libraries</a></td>
#     <td></td>
#   </tr>
#   <tr>
#       <td><a href='#Workflow-Pipeline'>Workflow Pipeline</a></td>
#     <td></td>
#     <td></td>
#   </tr>
#   <tr>
#     <td></td>
#       <td><a href='#1.-Data-Preprocessing'>1. Data Preprocessing</a></td>
#     <td></td>
#   </tr>
#   <tr>
#     <td></td>
#     <td></td>
#       <td><a href='#(A)-Data-Wrangling'>(A) Data Wrangling</a></td>
#   </tr>
#   <tr>
#     <td></td>
#     <td></td>
#       <td><a href='#(B)-Exploratory-Data-Analysis'>(B) Exploratory Data Analysis</a></td>
#   </tr>
#   <tr>
#     <td></td>
#       <td><a href='#2.-Train-&-Test-Split'>2. Train & Test Split</a></td>
#     <td></td>
#   </tr>
#   <tr>
#     <td></td>
#       <td><a href='#3.-Algorithm-Setup'>3. Algorithm Setup</a></td>
#     <td></td>
#   </tr>
#   <tr>
#     <td></td>
#       <td><a href='#4.-Model-Fitting'>4. Model Fitting</a></td>
#     <td></td>
#   </tr>
#   <tr>
#     <td></td>
#       <td><a href='#5.-Model-Predictions'>5. Model Predictions</a></td>
#     <td></td>
#   </tr>
#   <tr>
#     <td></td>
#       <td><a href='#6.-Model-Evalutaion'>6. Model Evalutaion</a></td>
#     <td></td>
#   </tr>
#   <tr>
#     <td></td>
#       <td><a href='#7.-Export-Predictions'>7. Export Predictions</a></td>
#     <td></td>
#   </tr>
# </tbody>
# </table>

# ## Dependencies

# ### (A) Import Libraries

# In[1]:


# Using this for auto-compeletion..
get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[2]:


import os
from typing import Text, List, Dict, Set, Tuple, Optional, Union

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = [18, 10]
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')


# ---

# ## Workflow Pipeline

# ### 1. Data Preprocessing

# First, we're going to explore the titanic dataset and start do some data wrangling and EDA 

# #### (A) Data Wrangling

# In[3]:


train_data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')


# In[4]:


train_data.head(10)


# In[5]:


test_data.head()


# In[6]:


train_data.info()


# In[7]:


train_data.describe()


# We noticed that; `Age` has `NaN` values, but if we compared between `Age` and `Cabin`, you can see that `Age` can be handled compared with `Cabin`.
# 
# - <span style='color:green;'><b>Notice:</b></span> We're going to handle `Cabin` at the end, so we can get the benefit from it, too.

# Now, Let's do some **Feature Engineering**

# ##### **Feature Engineering**

# <span style='color:CornflowerBlue;font-size:16px'>Feature 01: Age</span>

# In[8]:


train_data['Age'].isnull().value_counts()


# To fill the `NaN` values in the `Age` column, we can get the highest frequency of `Age` and start to fulfil it randomly instead of the `NaN` values. 
# 
# Note: 
# - We're going to select the highest 10 frequencies of our `Age` class and their frequencies and then, we're going to assign them, randomly.

# In[9]:


train_data['Age'].value_counts().head(10)


# To have a fair probability between all the `NaN` values replacements, we need to normalize the frequency of our `Age` feature so the total sum of the probability is equal 1. .

# Equation for that is;
# 
# # $X_{norm} = \frac{{1}}{\sum_{i}^{N}{x_i}}\times{x_j}$

# In[10]:


age_highest_freq = train_data['Age'].value_counts().head(10).index.values
freq_of_highes_age = train_data['Age'].value_counts().head(10).values
age_probabilities = list(map(lambda value: (1/sum(freq_of_highes_age)) * value,
                             freq_of_highes_age))


# In[11]:


train_data['Age'] = train_data['Age'].apply(lambda value: value if not np.isnan(value) 
                                            else np.random.choice(age_highest_freq,
                                                                  p=age_probabilities))


# In[12]:


train_data.info()


# <span style='color:CornflowerBlue;font-size:16px'>Feature 02: Embarked</span>

# `Embarked` Feature has only 1 missing value, we can fill this value with the mode value of the `Embarked` feature.

# In[13]:


train_data['Embarked'].isnull().value_counts()


# In[14]:


freq = train_data['Embarked'].value_counts().values[0]
train_data['Embarked'].fillna(freq, inplace=True)


# In[15]:


train_data['Embarked'].isnull().value_counts()


# Let's check the values inside the feature

# In[16]:


train_data['Embarked'].unique()


# We can see that `Embarked` has a number inside its values. Let's change this number of the max frequency value inside this feature.

# In[17]:


train_data['Embarked'].replace(to_replace=644,
                               value=train_data['Embarked'].value_counts().index.values[0],
                               inplace=True)


# In[18]:


train_data['Embarked'].unique()


# Before continuing **Feature Engineering** – all the features are stable now. We need to expand our vision more to understand the data that we have before applying any additional FE. What is going to help us in our mission of understanding the dataset is EDA.

#    

# #### (B) Exploratory Data Analysis

# First, we need to check the correlation between the features – this might be the first step to help us identify which feature will be really helpful in predictions.

# In[19]:


train_corr = train_data.corr()
train_corr


# In[20]:


plt.figure(figsize=(18,10))
sns.heatmap(train_corr, annot=True)
plt.show()


# The correlation between the features and the label beside the features within itself needs more work so we can discover the patterns.

# Let's go back to the FE

# <span style='color:CornflowerBlue;font-size:16px'>Feature 03: Name</span>

# The idea behind slicing `Name` helps for observing patterns from the name of the passengers and converting them into a categorical feature.
# 
# - <span style='color:green;'><b>Read This article:</b></span> <a href="https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/">Basic Feature Engineering with the Titanic Data</a>

# In[21]:


TITLES = ('Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
          'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
          'Don', 'Jonkheer')


# In[22]:


def patterns(name: Text, titles: Set) -> Optional[Text]:
    for title in titles:
        if title in name:
            return title
    return "Untitled"


# In[23]:


train_data['Title'] = train_data['Name'].apply(lambda name: patterns(name, TITLES))


# In[24]:


train_data


# Now, That's good but not perfect – we want to squeeze the number of categories in the feature. We can iterate throw all the observations and check the type of title this person has, but we may have a problem! One of the titles is "*Dr*". This title can fit both ["Men", "Women"] – we will have to add one more feature to help us decide whether this person is a "*Male*" or "*Female*", we will use `Sex` for this job.

# In[25]:


def squeeze_title(dataframe: pd.DataFrame) -> Text:
    title = dataframe['Title']
    if title in ('Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col'):
        return 'Mr'
    elif title in ('Countess', 'Mme'):
        return 'Mrs'
    elif title in ('Mlle', 'Ms'):
        return 'Miss'
    elif title == 'Dr':
        if dataframe['Sex'] == 'male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title


# In[26]:


train_data['Title'] = train_data.apply(squeeze_title, axis=1) 
train_data


# <u>Quoted from *TRIANGLEINEQUALITY*</u>
# > *You may be interested to know that ‘Jonkheer’ is a male honorific for Dutch nobility. Also interesting is that I was tempted to just send ‘Dr’ -> ‘Mr’, but decided to check first, and there was indeed a female doctor aboard! It seems 1912 was further ahead of its time than Doctor Who!*
# 
# > *Curious, I looked her up: her name was Dr. Alice Leader, and she and her husband were physicians in New York city*.[$^1$](https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/)
# 
# ![](https://i0.wp.com/www.encyclopedia-titanica.org/images/leader_af.jpg)

# <span style='color:CornflowerBlue;font-size:16px'>Feature 04: Cabin & Pclass</span>

# To work on `Cabin`, we need to understand what these values related to, the letters before each number. So that, we need to understand the infrastructure of the ship. 
# 
# After Reading more about how `Cabin` is related to passengers in Titanic – you can find this article in **dummies** – [Duites and Cabins for Passengers on the Titanic](https://www.dummies.com/article/academics-the-arts/history/20th-century/suites-and-cabins-for-passengers-on-the-titanic-180677). 
# 
# This article explains the correlation between the `Cabin` and the `Passenger Class (Pclass)`: we were not able to observe that from the correlation diagram since `Cabin` has lot of `NaN` values.

# According to **dummies**:
# 
# - Titanic's `Passenger first-class` Cabins:
# 
# > *First-class accommodations were located amidships, <u>where the rocking of the ship was less keenly felt and passengers were less likely to get seasick</u>. They were decorated opulently in different period styles: Queen Anne, Louis XVI, and Georgian.*
# 
# - Titanic's `Passenger second-class` Cabins:
# 
# > *passengers slept in berths built into the walls of the cabins. At two to four berths per cabin, privacy was hard to come by, although a passenger could close the curtain around his or her berth. <u>Each second-class cabin had a washbasin and a chamber pot to be used in case of seasickness</u>.*
# 
# - Titanic's `Passenger third-class` Cabins:
# 
# > *passengers slept on bunk beds in crowded quarters at six to a narrow cabin. Like second-class passengers, they shared bathrooms, but the number of people sharing a bathroom was much higher in third class: Only two bathtubs were available for all 710 third-class passengers, one for the men and one for the women.*

# ![](https://miro.medium.com/max/1400/1*VeHSAW_AxHz-GYwexWfdQg.jpeg)
# 
# Ref: [Titanic Survival Analysis Using R](https://chanida-limt.medium.com/titanic-survival-prediction-c421aac8da32)

# Group the data by `Pclass`

# In[27]:


CaPc = train_data.groupby('Pclass').count()
CaPc


# According to **BBC**: 
# > *The first half of the ship reaches the botton first. Two minutes later, the back half of the Titanic joins it on the floor of the Atlantic.*

# The first half of the ship is the place where most first-class cabins exist. So, using the `Cabin` feature may help us if we extract the letters from the `Cabin` observation will help us as it represents the Deck.

# In[28]:


plt.figure(figsize=(18,10))
sns.jointplot(x='Pclass', y='Cabin', data=CaPc, dropna=True)
plt.show()


# `NaN` Cabins will be replaced with `Unknown`

# In[29]:


cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
train_data['Deck'] = train_data['Cabin'].apply(lambda letter:
                                               patterns(str(letter), cabin_list))
train_data.drop(columns='Cabin', inplace=True)


# In[30]:


train_data.head()


# <span style='color:CornflowerBlue;font-size:16px'>Feature 05: SibSp & Parch</span>

# Let's visualize the `# of siblings / spouses aboard the Titanic (SibSp)` and `# of parents / children aboard the Titanic (Parch)` with `Survived` Class & the Linear Correlation between them

# In[31]:


fig = plt.figure(figsize=(15,21))
SS_SUR = fig.add_subplot(221)
PC_SUR = fig.add_subplot(222)
SS_PC = fig.add_subplot(2,2,(3,4))

sns.barplot(x='Survived', y='SibSp', data=train_data, ax=SS_SUR)
sns.barplot(x='Survived', y='Parch', data=train_data, ax=PC_SUR)

SS_PC.title.set_text("Correlation between `SibSp` & `Parch`")
sns.regplot(x='SibSp', y='Parch', data=train_data, ax=SS_PC)

plt.show()


# We can see, there're **positive Linear Correlation** between `SibSp` and `Parch`. Maybe this shows us – most people who had family relationships in the ship are more likely to risk themselves to rescue their family, especially if they're siblings/spouses.
# 
# Why not combine them!? Both features are likely to fit the same idea. It will be really efficient when we're going to use decision tree classification.

# In[32]:


train_data['Family_size'] = train_data['SibSp'] + train_data['Parch']
train_data.tail()


# <span style='color:CornflowerBlue;font-size:16px'>Feature 06: Age & Pclass & Fare</span>

# First, Let's check if the anyone from *First-class* or the *Second-class* has `Fare =0.0`. If so, we can't involve price with us.

# In[33]:


train_data[["Pclass", "Fare"]].loc[train_data['Fare'] == 0,:]


# These data are likely missed. We can't cross `Fare` with `Pclass`.

# In[34]:


train_data[['Pclass', 'Fare']].corr()


# We can see that – `P-class` and `Fare` have **Negative Linear Correlation**. (Note that we're trying to check the linear correlation although this is a categorical feature, we're doing that for the sake of seeing the effection on the `Fare` feature).

# In[35]:


plt.figure(figsize=(18,10))
sns.regplot(x='Pclass', y='Age', data=train_data)
plt.show()


# **What's the `Age` most of ppl `Died`?**

# In[36]:


tester = train_data.copy()
tester[['Survived', 'Died']] = pd.get_dummies(tester['Survived'])
age_sur = tester[['Age', 'Survived', 'Died']].groupby('Age').sum()
age_sur.reset_index(inplace=True)
age_sur.sort_values(by=['Died', 'Survived'], ascending=False, inplace=True)
age_sur.head(10)


# Young age are the most people died, specially `Age` of **24**, maybe this has been effected by filling the missing value!

# Let's check if there's any correlation between the `Age` and `Fare`

# In[37]:


plt.figure(figsize=(18,10))
sns.regplot(x='Age', y='Fare', data=train_data)
plt.show()


# As we can see, `Age` and `Fare` have **Weak Linear Correlation**, which means; I can't apply feature cross on them even if I want.

# <span style='color:CornflowerBlue;font-size:16px'>Feature 06: Sex & Title</span>

# We need to check the `Title` with the `Sex` to see if our FE works well

# In[38]:


plt.figure(figsize=(18,10))
sns.catplot(x='Sex', y='Title', data=train_data)
plt.show()


# Now, Let's clear our dataset from any useless features and prepare the dataset for our next phase which is **Train & Test Split**.

# In[39]:


train_data.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)


# For formatting purposes – we want to shift our `Survive` Label to the end of the table. Also, we need to add each Feature corresponding to its correlations. For the sake of curiosity, let's check the correlation between the new features and the old ones after deleting unnecessary features

# In[40]:


COLUMNS = ['Title', 'Sex', 'Age', 'SibSp', 'Parch', 'Family_size', 'Pclass',
           'Fare', 'Deck', 'Embarked', 'Survived']
train_data = train_data[COLUMNS]
train_data


# In[41]:


plt.figure(figsize=(18,10))
sns.heatmap(train_data.corr(), cmap='Blues', annot=True)
plt.show()


# The Correlation after **Feature Engineering** is efficient!

# ### 2. Train & Test Split

# Let's split our train dataset into a train, and test datasets [75, 25].

# In[42]:


y_label = train_data['Survived']
X_features = train_data.iloc[:,:-1]
X_features


# In[43]:


folds = 4
X_train, X_test, y_train, y_test = train_test_split(X_features,
                                                    y_label,
                                                    test_size=(1/folds),
                                                    random_state=42,
                                                    stratify=y_label)


# One of the critical things I read that written by *Prof. Andrew Neg* in his book **Machine Learning Yearning** – the proportion between your train, dev, and test data has to be on the same line of the distribution, you can't have a different proportion between your train, and test data and you expect to give you high accuracy.
# 
# `stratify` parameter in `train_test_split` accomplish that goal for you. When it splits the label, it will make sure that the proportion of the training label is similar to the proportion of the testing label.

# But we still can face the problem of overfitting..but how!? We're going to face the problem of overfitting because we split our data, randomly. We need to ensure that even if we sent weak hyperparameter to our probabilistic models, it wouldn't go throw overfitting because of <u>skew distribution</u>.
# 
# Therefore, I always prefer to add a procedure called **[cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics))**

#   

# ### 3. Algorithm Setup

# Since this project is all about classification, we're going to set up the best probabilistic models that can help us to predict the highest accuracy we can obtain.
# 
# But first, we want to make sure that our datasets have transformed & normalized, correctly. So, we're going to build a transformed pipeline to add our features into it.Since this project is all about classification, we're going to set up the best probabilistic models that can help us to predict the highest accuracy we can obtain.
# 
# But first, we want to make sure that our datasets have transformed & normalized, correctly. So, we're going to build a transformed pipeline to add our features into it.

# In[44]:


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import (StandardScaler, OneHotEncoder)
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score


# First, we're going to start with building the transformers pipeline

# In[45]:


NUMERICAL_COLUMNS = ["Age", "SibSp", "Parch", "Family_size", "Fare"]
CATEGORICAL_COLUMNS = ["Title", "Sex", "Pclass", "Deck", "Embarked"]


# In[46]:


transformers = make_column_transformer(
                                       (StandardScaler(), NUMERICAL_COLUMNS),
                                       (OneHotEncoder(handle_unknown='ignore'),
                                                      CATEGORICAL_COLUMNS))


# Then, we're going to the Models' pipeline. If you want to have multiple models in one pipeline, we can do that by building a [custom estimator](https://stackoverflow.com/questions/48507651/multiple-classification-models-in-a-scikit-pipeline-python#:~:text=consider%20checking%20out%20similar%20questions%20here%3A).

# In[47]:


class ClfSwitcher(BaseEstimator):
    
    def __init__(self, 
                 estimator = DecisionTreeClassifier()):
        """
        Custom Estimator is a custom class that helps you to switch
        between classifiers.
        
        Args:
            estimator: sklearn object – classifier
        """
        self.estimator = estimator
    
    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self
    
    def predict(self, X, y=None):
        return self.estimator.predict(X)
    
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
    
    def score(self, X, y):
        return self.estimator.score(X, y)


# In[48]:


pipeline = Pipeline([
    ('transformer', transformers),
    ('clf', ClfSwitcher())
])

parameters = [
    {
        'clf__estimator': [DecisionTreeClassifier()],
        'clf__estimator__criterion': ['gini', 'entropy'],

    },
    {
        'clf__estimator': [ExtraTreesClassifier()],
        'clf__estimator__n_estimators': [100, 250],
        'clf__estimator__criterion': ['gini', 'entropy'],

    },
    {
        'clf__estimator': [RandomForestClassifier()],
        'clf__estimator__n_estimators': [100, 250],
        'clf__estimator__criterion': ['gini', 'entropy'],
    },
    {
        'clf__estimator': [SVC()],
        'clf__estimator__kernel': ['rbf', 'sigmoid'],
        'clf__estimator__C': [1e-3, 1e-2, 1e-1, 1.0, 10., 1e+2, 1e+3],
        'clf__estimator__degree': [3, 4, 5, 6]
    },
    {
        'clf__estimator': [LogisticRegression()],
        'clf__estimator__penalty': ['l1', 'l2'],
        'clf__estimator__tol': [1e-4, 1e-3, 1e-2],
        'clf__estimator__C': [1e-3, 1e-2, 1e-1, 1.0, 10., 1e+2, 1e+3],
        'clf__estimator__solver': ['lbfgs', 'liblinear']   
    }
    
]


# In[49]:


cv = KFold(n_splits=(folds - 1))


# ### 4. Model Fitting

# Now, we're fine-tune the model and see what is the best model, besides what are the best hyperparameters.

# In[50]:


gscv = GridSearchCV(pipeline,
                    parameters,
                    cv=cv,
                    scoring='r2',
                    n_jobs=12,
                    verbose=3)
gscv.fit(X_train, y_train)


# In[51]:


gscv.best_params_


# In[52]:


model = pipeline.set_params(**gscv.best_params_)


# In[53]:


model.fit(X_train, y_train)


# ### 5. Model Predictions

# Now, let's test the model we trained using our test data we split from the main train dataset.

# In[54]:


y_hat = model.predict(X_test)


# ### 6. Model Evalutaion

# Let's evaluate our model using **Single Evaluation Metric** and **Multi-evaluation Metrics**

# 1. Single-number Evaluation Metric

# In[55]:


model.score(X_test, y_test)


# 2. Multi-evaluation Metrics

# In[56]:


cm = confusion_matrix(y_test, y_hat, labels=[1, 0])
cm


# In[57]:


confusion_plot = ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels=['Survived', 'Died'])

fig, ax = plt.subplots(1,1,figsize=(12, 8))
ax.grid(False)
confusion_plot.plot(cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix Between True Label & Predicted Label')
plt.show()


# In[58]:


f1_score(y_test, y_hat)


# According to *Prof. Andrew Neg* in a book of **Machine Learning Yearning** – If you want to relay on accurate accuracy for your classification model, you should relay on <u>Multi-evaluation metric</u> [ Precision, Recall ]. You can combine them into <u>single-number evaluation metric</u> which `F1-score `.

# ### 7. Export Predictions

# Lastly, we're going to create a function that apply all the above steps and provide to us the predictions of the `test.csv`.

# In[59]:


def prediction_fn(data_dir: Text, model) -> pd.DataFrame:
    """
    A Function created for producing a batch prediction
    for titanic model.
    
    Args:
        data_dir [Text]: directory to the test dataset to use it in the prediction.
        model: model pipeline that is going to be used in the prediction.
    
    Returns:
        pd.DataFrame: predicted dataframe.
    """
    # read dataset
    test_data = pd.read_csv(data_dir, index_col=False)
    
    # pre-processing & Feature Engineering
    ## Age
    age_highest_freq = test_data['Age'].value_counts().head(10).index.values
    freq_of_highes_age = test_data['Age'].value_counts().head(10).values
    age_probabilities = list(map(lambda value: (1/sum(freq_of_highes_age)) * value,
                                 freq_of_highes_age))
    test_data['Age'] = test_data['Age'].apply(lambda value: value if not np.isnan(value) 
                                                else np.random.choice(age_highest_freq,
                                                                      p=age_probabilities))
    ## Embarked
    freq = test_data['Embarked'].value_counts().values[0]
    test_data['Embarked'].fillna(freq, inplace=True)
    test_data['Embarked'].replace(to_replace=644,
                                   value=test_data['Embarked'].value_counts().index.values[0],
                                   inplace=True)
    ## Fare
    test_data['Fare'].fillna(method='bfill', inplace=True)
    ## Title
    test_data['Title'] = test_data['Name'].apply(lambda name: patterns(name, TITLES))
    test_data['Title'] = test_data.apply(squeeze_title, axis=1) 
    
    ## Cabin
    test_data['Deck'] = test_data['Cabin'].apply(lambda letter:
                                                   patterns(str(letter), cabin_list))
    test_data.drop(columns='Cabin', inplace=True)
    
    ## Family Size
    test_data['Family_size'] = test_data['SibSp'] + test_data['Parch']
    
    # Drop the useless features (Only keep PassengerId for merging  it with the predictions)
    passId = test_data['PassengerId']
    test_data.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)
    
    # Model predictions
    predictions = model.predict(test_data)
    
    # Building the dataframe of the predction values
    predictions = pd.DataFrame(predictions, columns=['Survived'])
    predictions = pd.merge(passId, predictions, left_index=True, right_index=True)
    
    return predictions


# In[60]:


import warnings
warnings.filterwarnings('ignore')
prediction = prediction_fn('../input/titanic/test.csv', model)
prediction


# In[61]:


prediction.to_csv('gender_submission.csv', index=False, index_label=False)
print("You have sucessfully exported the predictions!")


# <p style='text-align:center;'>Thanks for reaching this level of expermenting
# the idea of</p>
# <p style='text-align:center;'><b>Titanic - How Data Scientist applies Analysis & ML</b></p>
# <p style='text-align:center;'>Data Scientist & ML Engineer: <a href='https://www.linkedin.com/in/drxavier997/'>Ahmed</a></p>
# <p style='text-align:center;'>Created at: 2022-01-13
