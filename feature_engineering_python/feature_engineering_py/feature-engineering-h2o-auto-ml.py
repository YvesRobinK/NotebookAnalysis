#!/usr/bin/env python
# coding: utf-8

# <h1 align="center"> Titanic Survival prediction</h1>
# 
# <img src="https://images.pexels.com/photos/813011/pexels-photo-813011.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940" width="60%">
# <center>
# <a href="https://www.pexels.com/photo/white-cruise-ship-813011/">Photo by Matthew Barra from Pexels</a>
# </center>
# 
# # Plan of Action
# 
# [1. Data pre-processing](#1)
# 
# [2. Plots and Charts](#2)
# 
# [3. Feature engineering](#3)
# 
# [4. Modeling](#h20)
# 
# [5. Evaluation](#4)
# 
# [6. Submission & References](#5)

# ### Importing python libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')


# ### Load the data into memory

# In[2]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')

print(f"train set : {train.shape[1]} columns and {train.shape[0]} rows")
print(f"test set : {test.shape[1]} columns and {test.shape[0]} rows")


# ### Sample data

# In[3]:


train.head()


# ### Combine train and test

# In[4]:


test_id = test.PassengerId.tolist()
test.insert(1, "Survived", np.nan)

dataset = [train, test]


# <a id="1"></a>
# # Data pre-processing
# 
# ### Columns
# <table align="left" style="font-size:15px;">
# <tbody>
# <tr><th><b>Variable</b></th><th><b>Definition</b></th><th><b>Key</b></th></tr>
# <tr>
# <td>PassengerId</td>
# <td>Unique Id of passenger</td>
# <td></td>
# </tr>
# <tr>
# <td>Survived</td>
# <td>Survival</td>
# <td>0 = No, 1 = Yes</td>
# </tr>
# <tr>
# <td>pclass</td>
# <td>Ticket class</td>
# <td>1 = 1st, 2 = 2nd, 3 = 3rd</td>
# </tr>
# <tr>
# <td>Name</td>
# <td>Name of passenger</td>
# <td></td>
# </tr>
# <tr>
# <td>sex</td>
# <td>Sex</td>
# <td></td>
# </tr>
# <tr>
# <td>Age</td>
# <td>Age in years</td>
# <td></td>
# </tr>
# <tr>
# <td>sibsp</td>
# <td># of siblings / spouses aboard the Titanic</td>
# <td></td>
# </tr>
# <tr>
# <td>parch</td>
# <td># of parents / children aboard the Titanic</td>
# <td></td>
# </tr>
# <tr>
# <td>ticket</td>
# <td>Ticket number</td>
# <td></td>
# </tr>
# <tr>
# <td>fare</td>
# <td>Passenger fare</td>
# <td></td>
# </tr>
# <tr>
# <td>cabin</td>
# <td>Cabin number</td>
# <td></td>
# </tr>
# <tr>
# <td>embarked</td>
# <td>Port of Embarkation</td>
# <td>C = Cherbourg, Q = Queenstown, S = Southampton</td>
# </tr>
# </tbody>
# </table>

# ### Variable separation

# In[5]:


features = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
target = "Survived"


# ## Handling Missing values

# In[6]:


def plot_missing(train, test):
    fig, axes = plt.subplots(1,2, figsize=(20,6))
    
    m1 = train[features].isna()
    m2 = test[features].isna()
    
    sns.heatmap(m1, cmap='viridis', cbar=False, yticklabels=[], ax=axes[0])
    sns.heatmap(m2, cmap='viridis', cbar=False, yticklabels=[], ax=axes[1])
    
    axes[0].set_title(f"Missing values in train", fontsize=14)
    axes[1].set_title(f"Missing values in test", fontsize=14)
        
plt.show()


# In[7]:


plot_missing(train, test)


# In[8]:


train[features].isna().mean().to_frame('% of missing values')*100


# ### Age
# #### Filling missing values in age using the median age for each gender within each passenger class as a proxy.

# In[9]:


plt.figure(figsize=(10,5))

df=train.groupby(['Sex',"Pclass"])['Age'].median().to_frame().reset_index()

plot = sns.boxplot(x='Sex', y='Age', hue='Pclass', data=train)
plot.set_xlabel("Sex", fontsize=14, fontweight="bold")
plot.set_ylabel("Age",fontsize=14, fontweight="bold")

plt.show()


# In[10]:


# Calaculate median ages

median_ages = np.zeros((2,3))

for i, sex in enumerate(["male", "female"]):
    for j in range(0,3):
        median_ages[i,j] = train[((train['Sex'] == sex) &
                                 (train['Pclass'] == j+1))]['Age'].median()


# In[11]:


# Put our estimates into NaN rows of new column AgeFill.

train['AgeFill'] = train['Age']
test["AgeFill"] = test["Age"]

for data in dataset:
    for i, sex in enumerate(["male", "female"]):
        for j in range(0, 3):
            data.loc[((data.Age.isnull()) &
                      (data.Sex == sex) &
                      (data.Pclass == j+1)), 'AgeFill'] = median_ages[i,j]


    # Create a feature that records whether the Age was originally missing
    data['AgeIsNull'] = pd.isnull(data['Age']).astype(int)

    # Drop original age column
    data.drop("Age", axis=1, inplace=True)


# ### Fare
# #### Filling missing values of fare with the median fare

# In[12]:


medfare = data['Fare'].dropna().median()

for data in dataset:
    data['Fare'].fillna(medfare, inplace=True)


# ### Cabin
# 
# #### Create a new feature `Has_cabin` and Drop Cabin feature since it has 77% of missing values

# In[13]:


# Feature that tells whether a passenger had a cabin on the Titanic
for data in dataset:
    
    data['Has_cabin'] = data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    #Drop column
    data.drop('Cabin', axis=1, inplace=True)


# ### Embarked
# #### Filling missing values of Embarked column with the most frequent label

# In[14]:


mf = data.Embarked.mode()[0]

for data in dataset:
    data.Embarked.fillna(value=mf, inplace=True)


# <a id="2"></a>
# # Plots and charts

# In[15]:


continuous = ['Fare', 'AgeFill']
discrete = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', "Has_cabin"]


# ### Target distribution

# In[16]:


fig, ax = plt.subplots(figsize=(12, 5))

df = train[target].value_counts(normalize=True)

df.plot.bar(width=0.2, color=('red','green'), ax=ax, fontsize=14)
ax.set_title("Target distribution", fontweight='bold', fontsize=15)
plt.show()


# ### Pie charts

# In[17]:


fig, axes = plt.subplots(3, 2, figsize=(10,12))
axes = [ax for axes_row in axes for ax in axes_row]

for i,c in enumerate(discrete):
    df = train[c].value_counts()
    df.plot(kind='pie', ax=axes[i], title=c, autopct="%.2f", fontsize=14)
    axes[i].set_ylabel('')
    axes[i].set_title(c, fontsize=14, fontweight='bold')
    
plt.show()


# ### Continuous columns distribution

# In[18]:


fig, axes = plt.subplots(ncols=2,figsize=(20, 6))

for i, c in enumerate(continuous):
    
    hist = train[c].plot(kind = 'hist',
                         ax=axes[i], 
                         color='blue',
                         bins=30)
    axes[i].set_title(c, fontsize=14, fontweight='bold')
    
plt.show()


# <a id="3"></a>
# # Feature Engineering

# #### Extracting a new feature `Ticket_type` from `Ticket`

# In[19]:


for data in dataset:
    data['Ticket_type'] = data['Ticket'].apply(lambda x: x[0:3])
    data['Ticket_type'] = data['Ticket_type'].astype('category')
    data['Ticket_type'] = data['Ticket_type'].cat.codes

    # Remove ticket column
    data.drop("Ticket", axis=1, inplace=True)


# #### Extracting new features `Name_length`, `Word_count` and `Title` from `Name`

# In[20]:


# Title from name
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

for data in dataset:
    
    data.loc[:, 'Name_length'] = data.Name.apply(len)
    data.loc[:, 'Word_count'] = data.Name.apply(lambda x: len(x.split()))

    data.loc[:, 'Title'] = data['Name'].apply(get_title)
    rare = ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']

    data['Title'] = data['Title'].replace(rare, 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')

    # Remove Name column
    data.drop('Name', axis=1, inplace=True)


# ### Label encoding the categorical features

# In[21]:


for data in dataset:
    
    # Mapping Sex
    data['Sex'] = data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'] = data['Title'].fillna(0)

    # Mapping Embarked
    data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# ### Binning : Converting continuous features into discrete features

# In[22]:


# Mapping Fare
for data in dataset:
    
    data.loc[ data['Fare'] <= 7.91, 'Fare'] = 0

    data.loc[((data['Fare'] > 7.91) & 
                 (data['Fare'] <= 14.454)), 'Fare'] = 1

    data.loc[((data['Fare'] > 14.454) &
                 (data['Fare'] <= 31)), 'Fare'] = 2

    data.loc[ data['Fare'] > 31, 'Fare'] = 3

    data['Fare'] = data['Fare'].astype(int)


# In[23]:


# Mapping Age
for data in dataset:
    
    data.loc[ data['AgeFill'] <= 16, 'AgeFill'] = 0

    data.loc[((data['AgeFill'] > 16) &
                 (data['AgeFill'] <= 32)), 'AgeFill'] = 1

    data.loc[((data['AgeFill'] > 32) &
              (data['AgeFill'] <= 48)), 'AgeFill'] = 2

    data.loc[((data['AgeFill'] > 48) & 
              (data['AgeFill'] <= 64)), 'AgeFill'] = 3

    data.loc[data['AgeFill'] > 64, 'AgeFill'] = 4


# ### Hand picked features

# In[24]:


for data in dataset:
    
    # Create new feature FamilySize as a combination of SibSp and Parch
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    # Create new feature IsAlone from FamilySize
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1


# In[25]:


train.head()


# In[26]:


final_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare','Embarked',
                  'AgeFill', 'AgeIsNull', 'Has_cabin', 'Ticket_type',
                  'Name_length', 'Word_count', 'Title', 'FamilySize', 'IsAlone']

train = train[final_features+[target]]
test = test[final_features]

X_train, X_test, y_train, y_test = train_test_split(train[final_features],
                                                    train[target].astype(int),
                                                    test_size=0.25,
                                                    random_state=1,
                                                    stratify=train[target]
                                                   )


# Make a copy of the test test
test_copy = test.copy()


# ### Correlation between variables

# In[27]:


plt.figure(figsize=(18,8))

df = train.corr()

mask = np.triu(np.ones_like(df))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(df, annot=True, cbar=False, cmap="Blues",mask=mask)
plt.show()


# <a id="h20"></a>
# # Modeling
# 

# In[28]:


from h2o.automl import H2OAutoML
import h2o
# initiate h2o instance
h2o.init()


# In[52]:


hf_train = h2o.H2OFrame(train)
aml = H2OAutoML(max_models = 10, seed = 10, exclude_algos = ["DeepLearning"], verbosity="info", nfolds=0)
aml.train(x = final_features, y = target, training_frame = hf_train)


# In[53]:


lb = aml.leaderboard
lb.head(rows=lb.nrows)


# In[54]:


preds = aml.predict(h2o.H2OFrame(test))


# In[62]:


predictions = preds.as_data_frame().predict.apply(lambda x : 1 if x>0.5 else 0).values


# ## Submission

# In[63]:


df = pd.DataFrame({ 'PassengerId': test_id, 'Survived': predictions })
df.to_csv("predictions.csv", index=False)


# In[64]:


df.head(5)


# <a id="5"></a>
# # References
# Feature engineering :  https://www.kaggle.com/startupsci/titanic-data-science-solutions <br>
# Modeling :  https://thecleverprogrammer.com/2020/08/01/automate-machine-learning-with-h2o-automl/
