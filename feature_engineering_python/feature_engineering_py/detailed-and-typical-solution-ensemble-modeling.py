#!/usr/bin/env python
# coding: utf-8

# ## Detailed and typical solution (>81% Score)
# 
# Hello kagglers ..   
# 
# This notebook designed to be as **detailed** as possible solution for the Titanic problem, I tried to make it typical, clear, tidy and **beginner-friendly**. 
# 
# If you find this notebook useful press the **upvote** button, This helps me a lot ^-^.  
# I hope you find it helpful.

# <img src="data:image/jpeg;base64,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">

# In[1]:


#=======================================================================================
# Importing the libaries:
#=======================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings("ignore")

#=======================================================================================


# <div style="color:#00ADB5;
#            display:fill;
#            border-radius:5px;
#            background-color:#393E46;
#            font-size:20px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 10px;
#               color:white;">
#             <b>1 ) Importing the data:</b>
#         </p>
# </div>
# 

# In[2]:


#=======================================================================================
# Reading the data:
#=======================================================================================

def read_data():
    train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
    print("Train data imported successfully!!")
    print("-"*50)
    test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
    print("Test data imported successfully!!")
    return train_data , test_data

train_data , test_data = read_data()
combine = [train_data , test_data]

#=======================================================================================


# <div style="color:#00ADB5;
#            display:fill;
#            border-radius:5px;
#            background-color:#393E46;
#            font-size:20px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 10px;
#               color:white;">
#             <b> 2 ) Discovering the data:</b>
#         </p>
# </div>
# 

# In[3]:


train_data.head()


# In[4]:


test_data.head()


# In[5]:


#=======================================================================================
# Discovering the features:
#=======================================================================================

print("Train data features are:\n")
print(train_data.columns.values)
print('\n' ,"="*80 , '\n')
print("Test data features are:\n")
print(test_data.columns.values)


# In[6]:


#=======================================================================================
# Discovering the features types:
#=======================================================================================

train_data.info()
print('_'*40 , '\n')
test_data.info()


# #### Features discovering results:
# 
# - Features of the test data are the same as features of the training data except for the Survived feature (because it's the target).
# - Features types:
#     - Categorical:
#         - Pclass (ordinal)
#         - Name (nominal)
#         - Sex (nominal)
#     - Numerical:
#         - Age (continuous)
#         - Fare (continuous)
#         - SibSp (discrete)
#         - Parch (discrete)
#     - Mixed:
#         - Ticket (numeric and alphanumeric)
#         - Cabin (alphanumeric)
#         
# **Insights:** Features types are very important for EDA step.

# In[7]:


#=======================================================================================
# Discovering the missed values:
#=======================================================================================

print("Train data missed values:\n")
print(train_data.isnull().sum())
print('\n','_'*40 , '\n')
print("Test data missed values:")
print(test_data.isnull().sum())


# #### Missed values discovering results:
# - Train Data:
#     - Age: 177/891 missed values (19.8% are missed).
#     - Cabin: 687/891 missed values (77.1% are missed).
#     - Embarked: 2/891 missed values.
# - Test Data:
#     - Age: 86/418 missed values (20.5% are missed).
#     - Cabin: 327/418 missed values (78.2% are missed).
#     - Fare: 1/418 missed values.
#     
# **Insights:** Now we know what are the missed values, And we are going to discover the correlations to do what is appropriate.

# In[8]:


#=======================================================================================
# Discovering the numerical data distribution :
#=======================================================================================

train_data.describe()


# #### Numerical data distribution discovering results:
#   - The survival rate for this data is 38.3%.
#   - More than 75% of the passengers are below 38 years old.
#   - There are too few old passengers.
#   - Most passengers travel alone.
#   - There are a few outliers in the Fare feature.
# 
# **insights:**  
#   - Age feature has right skewness, So if we are going to fill missing values we will not use the average.
#   - There are outliers in Fare, Age, SibSp and Parch features. This inspire us for EDA Step.

# In[9]:


#=======================================================================================
# Discovering the categorical data distribution :
#=======================================================================================

train_data.describe(include=['O'])


# #### Categorical data distribution discovering results:
#  - There are no duplicated names.
#  - 64.7% of the passengers are males
#  - There are duplicated values in the Ticket feature (23.5% are duplicated).
#  - There are duplicated values in the Cabin feature (27.9% are duplicated).
#  - 72.4% of the passengers used "S" Embarked.
# 
# **insights:**
#  - Names uniqueness gives us unsight for Data Engineering Step.
#  - Duplicate values in Ticket and Cabin features give us insights to make the right decision when filling missed values.
#  - we will fill missed Embarked values with 'S' type.

# <div style="color:#00ADB5;
#            display:fill;
#            border-radius:5px;
#            background-color:#393E46;
#            font-size:20px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 10px;
#               color:white;">
#             <b> 3 ) Exploratory Data Analysis (EDA):</b>
#         </p>
# </div>
# 
# 

# In[10]:


# ===================================================================
# Count of survived
# ===================================================================
f,ax=plt.subplots(1,2,figsize=(8,4))
train_data['Survived'].replace({0:"died",1:"survived"}).value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_ylabel('')
sns.countplot(x = train_data["Survived"].replace({0:"died",1:"survived"}) , ax = ax[1])
ax[1].set_ylabel('')
ax[1].set_xlabel('')
plt.show()


# We saw before that only 338 (38%) of the passengers survived, We need to dig down more to get better insights from the data and see which categories of the passengers did survive and who didn't.
# 
# 

# <div style="color:black;
#            border-radius:0px;
#            background-color:#00ADB5;
#            font-size:14px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 6px;
#               color:white;">
#             <b>Discovering the features correlation with Survived:</b>
#         </p>
# </div>
# 

# In[11]:


# Helper functions:
def survived_bar_plot(feature):
    plt.figure(figsize = (6,4))
    sns.barplot(data = train_data , x = feature , y = "Survived").set_title(f"{feature} Vs Survived")
    plt.show()
def survived_table(feature):
    return train_data[[feature, "Survived"]].groupby([feature], as_index=False).mean().sort_values(by='Survived', ascending=False).style.background_gradient(low=0.75,high=1)
def survived_hist_plot(feature):
    plt.figure(figsize = (6,4))
    sns.histplot(data = train_data , x = feature , hue = "Survived",binwidth=5,palette = sns.color_palette(["yellow" , "green"]) ,multiple = "stack" ).set_title(f"{feature} Vs Survived")
    plt.show()


# #### Sex Vs Survived:

# In[12]:


survived_bar_plot('Sex')


# In[13]:


survived_table("Sex")


# Females have higher Survival rate (74%)

# #### Pclass Vs Survived:

# In[14]:


survived_bar_plot("Pclass")


# In[15]:


survived_table("Pclass")


# First Pclass passengers are more likely to survive then Seconde Pclass then Third.

# #### Embarked:

# In[16]:


survived_bar_plot("Embarked")


# In[17]:


survived_table("Embarked")


# Passengers who used C Embarked are most likely to survive, Then Q, Then S. (This reasong maybe undirect, I thinks that most of 1 Pclass passengers used C Embarked)

# #### Parch:

# In[18]:


survived_bar_plot("Parch")


# In[19]:


survived_table("Parch")


# Parch feature has zero correlation for some values, Maybe we can use it to derive more useful feature.

# #### SibSp:

# In[20]:


survived_table("SibSp")


# In[21]:


survived_bar_plot("SibSp")


# SibSp feature has zero correlation for some values too, Maybe we can use it to derive more useful feature.

# #### Age:

# In[22]:


sns.set_style("dark") # to remove the grid.
survived_hist_plot("Age") # Note: This plot is stack plot.


# - Infants (age<=5) and childrens (between 10 and 15 years old) are most likely to survive.
# - elder passengers (>75) survived.
# - most passengers are between 15 and 40 years old.
# 
# **insights:** It's good to convert the age feature to age band groups of length 5.

# <div style="color:black;
#            border-radius:0px;
#            background-color:#00ADB5;
#            font-size:14px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 6px;
#               color:white;">
#             <b>Discovering the correlation between the features:</b>
#         </p>
# </div>
# 

# In[23]:


#=======================================================================================
# Discovering the correlations:
#=======================================================================================

sns.set(rc = {'figure.figsize':(10,6)})
sns.heatmap(train_data.corr(), annot = True, fmt='.2g',cmap= 'YlGnBu')


# - Passenger Id has no correlation with any feature.
# - PClass has strong negative correlation with age and Fare.
# - Age has negative correlation with parch and sibsp.

# #### Pclass - Age - Survived:

# In[24]:


plot , ax = plt.subplots(1 , 3 , figsize=(14,4))
sns.histplot(data = train_data.loc[train_data["Pclass"]==1] , x = "Age" , hue = "Survived",binwidth=5,ax = ax[0],palette = sns.color_palette(["yellow" , "green"]),multiple = "stack").set_title("1-Pclass")
sns.histplot(data = train_data.loc[train_data["Pclass"]==2] , x = "Age" , hue = "Survived",binwidth=5,ax = ax[1],palette = sns.color_palette(["yellow" , "green"]),multiple = "stack").set_title("2-Pclass")
sns.histplot(data = train_data.loc[train_data["Pclass"]==3] , x = "Age" , hue = "Survived",binwidth=5,ax = ax[2],palette = sns.color_palette(["yellow" , "green"]),multiple = "stack").set_title("3-Pclass")
plt.show()


# - Pclass=3 had most passengers, Most if them did not survive.
# - Infant passengers in Pclass=2 and Pclass=3 mostly survived.
# - Most passengers in Pclass=1 survived.

# #### Sex - Age - Survived:

# In[25]:


plot , ax = plt.subplots(1 , 2 , figsize=(14,3))
sns.histplot(data = train_data.loc[train_data["Sex"]=="male"] , x = "Age" , hue = "Survived",binwidth=5,ax = ax[0],palette = sns.color_palette(["yellow" , "green"]),multiple = "stack").set_title("Males")
sns.histplot(data = train_data.loc[train_data["Sex"]=="female"] , x = "Age" , hue = "Survived",binwidth=5,ax = ax[1],palette = sns.color_palette(["yellow" , "green"]),multiple = "stack").set_title("Females")


# - as we saw before, Females are most likely to survive.
# - Elder passengers (>=70) are all males.
# 

# <div style="color:#00ADB5;
#            display:fill;
#            border-radius:5px;
#            background-color:#393E46;
#            font-size:20px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 10px;
#               color:white;">
#             <b> 4 ) Wrangling The Data:</b>
#         </p>
# </div>
# 
# 

# <div style="color:black;
#            border-radius:0px;
#            background-color:#00ADB5;
#            font-size:14px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 6px;
#               color:white;">
#             <b> Drop unuseful features:</b>
#         </p>
# </div>
# 

# In[26]:


train_data.drop(columns = ["PassengerId"] , inplace = True)

for dataset in combine:
    dataset.drop(columns = ["Ticket" , "Cabin"] , inplace = True)
    
print("Dropping features Done !!")


# <div style="color:black;
#            border-radius:0px;
#            background-color:#00ADB5;
#            font-size:14px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 6px;
#               color:white;">
#             <b>Converting Categorical Features to Numerical and Filling Missed Values:</b>
#         </p>
# </div>
# 

# #### Embarked:

# ![](http://)

# In[27]:


train_data.Embarked.fillna(train_data.Embarked.dropna().max(), inplace=True)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].dropna().map({'S':0,'C':1,'Q':2}).astype(int)


# #### Sex:

# In[28]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)    


# #### Age:

# In[29]:


guess_ages = np.zeros((2,3))

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)


# #### Fare:

# In[30]:


test_data.Fare.fillna(test_data.Fare.dropna().median() , inplace= True)


# In[31]:


print(train_data.isnull().sum())
print("-" * 50)
print(test_data.isnull().sum())


# No more missed values !!

# In[32]:


train_data.head()


# <div style="color:black;
#            border-radius:0px;
#            background-color:#00ADB5;
#            font-size:14px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 6px;
#               color:white;">
#             <b>Creating Bands:</b>
#         </p>
# </div>
# 

# #### Age Band:

# In[33]:


train_data['AgeBand'] = pd.cut(train_data['Age'], 5)
train_data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# In[34]:


for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_data.head()


# In[35]:


train_data.drop(['AgeBand'], axis=1 , inplace = True)


# #### Fare Band:

# In[36]:


train_data['FareBand'] = pd.qcut(train_data['Fare'], 4)
train_data[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=False)


# In[37]:


for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_data.drop(['FareBand'], axis=1 , inplace = True)


# In[38]:


train_data.head()


# <div style="color:black;
#            border-radius:0px;
#            background-color:#00ADB5;
#            font-size:14px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 6px;
#               color:white;">
#             <b>Data Engineering:</b>
#         </p>
# </div>
# 

# #### Family Size:

# In[39]:


for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_data.drop(['Parch', 'SibSp'], axis=1 , inplace = True)
test_data.drop(['Parch', 'SibSp'], axis=1 , inplace = True)    

train_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[40]:


# Create new feature of family size
for dataset in combine:
    dataset['Single'] = dataset['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    dataset['SmallF'] = dataset['FamilySize'].map(lambda s: 1 if  s == 2  else 0)
    dataset['MedF'] = dataset['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
    dataset['LargeF'] = dataset['FamilySize'].map(lambda s: 1 if s >= 5 else 0)
    
train_data.drop(columns = ["FamilySize"] , inplace = True)
test_data.drop(columns = ["FamilySize"] , inplace = True)


# #### Name Title:

# In[41]:


for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_data['Title'], train_data['Sex'])


# In[42]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[43]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)


# In[44]:


train_data.drop(['Name'], axis=1 , inplace = True)
test_data.drop(['Name'], axis=1 , inplace = True)    


# In[45]:


train_data.head()


# In[46]:


test_data.head()


# <div style="color:#00ADB5;
#            display:fill;
#            border-radius:5px;
#            background-color:#393E46;
#            font-size:20px;
#            font-family:sans-serif;
#            letter-spacing:0.5px">
#         <p style="padding: 10px;
#               color:white;">
#             <b> 5 ) Modeling:</b>
#         </p>
# </div>
# 
# 

# In[47]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve


# In[48]:


# ==================================================================================
# Preparing Data For Training:
# ==================================================================================

Y_train = train_data["Survived"]
X_train = train_data.drop(labels = ["Survived"],axis = 1)
Test = test_data.drop(labels = ["PassengerId"],axis = 1)
print(f"X_train shape is = {X_train.shape}" )
print(f"Y_train shape is = {Y_train.shape}" )
print(f"Test shape is = {Test.shape}" )


# In[49]:


# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)


# In[50]:


# Modeling step Test differents algorithms 
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")


# In[51]:


### META MODELING  WITH ADABOOST, RF, EXTRATREES and GRADIENTBOOSTING

# Adaboost
DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsadaDTC.fit(X_train,Y_train)

ada_best = gsadaDTC.best_estimator_

gsadaDTC.best_score_


# In[52]:


#ExtraTrees 
ExtC = ExtraTreesClassifier()


## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsExtC.fit(X_train,Y_train)

ExtC_best = gsExtC.best_estimator_

# Best score
gsExtC.best_score_


# In[53]:


# RFC Parameters tunning 
RFC = RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(X_train,Y_train)

RFC_best = gsRFC.best_estimator_

# Best score
gsRFC.best_score_


# In[54]:


# Gradient boosting tunning

GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsGBC.fit(X_train,Y_train)

GBC_best = gsGBC.best_estimator_

# Best score
gsGBC.best_score_


# In[55]:


### SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsSVMC.fit(X_train,Y_train)

SVMC_best = gsSVMC.best_estimator_

# Best score
gsSVMC.best_score_


# In[56]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

g = plot_learning_curve(gsRFC.best_estimator_,"RF mearning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsSVMC.best_estimator_,"SVC learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsadaDTC.best_estimator_,"AdaBoost learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsGBC.best_estimator_,"GradientBoosting learning curves",X_train,Y_train,cv=kfold)


# In[57]:


test_Survived_RFC = pd.Series(RFC_best.predict(Test), name="RFC")
test_Survived_ExtC = pd.Series(ExtC_best.predict(Test), name="ExtC")
test_Survived_SVMC = pd.Series(SVMC_best.predict(Test), name="SVC")
test_Survived_AdaC = pd.Series(ada_best.predict(Test), name="Ada")
test_Survived_GBC = pd.Series(GBC_best.predict(Test), name="GBC")

# Concatenate all classifier results
ensemble_results = pd.concat([test_Survived_RFC,test_Survived_ExtC,test_Survived_AdaC,test_Survived_GBC, test_Survived_SVMC],axis=1)
g= sns.heatmap(ensemble_results.corr(),annot=True)


# In[58]:


# Concatenate all classifier results
ensemble_results = pd.concat([test_Survived_RFC,test_Survived_ExtC,test_Survived_AdaC,test_Survived_GBC, test_Survived_SVMC],axis=1)


g= sns.heatmap(ensemble_results.corr(),annot=True)


# In[59]:


votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=4)

votingC = votingC.fit(X_train, Y_train)


# In[60]:


test_Survived = pd.Series(votingC.predict(Test), name="Survived")

results = pd.concat([test_data.PassengerId,test_Survived],axis=1)

results.to_csv("ensemble_python_voting.csv",index=False)


# #### My Other Useful Notebooks:
# - [House Prices - Advanced Regression Problem](https://www.kaggle.com/code/odaymourad/detailed-full-solution-step-by-step-top-3).
# - [Spaceship Titanic](https://www.kaggle.com/code/odaymourad/detailed-and-full-solution-step-by-step-80-score).
# - [Feature Selection and Data Engineering](https://www.kaggle.com/code/odaymourad/feature-selection-data-engineering-step-by-step).
# - [Learn Overfitting and Underfitting](https://www.kaggle.com/code/odaymourad/learn-overfitting-and-underfitting-79-4-score).
# - [Random Forest Algorithm](https://www.kaggle.com/code/odaymourad/random-forest-model-clearly-explained).
# - [NLP with Disaster tweets](https://www.kaggle.com/code/odaymourad/detailed-and-full-solution-78-4-score).

# ### References:
# - https://www.kaggle.com/code/startupsci/titanic-data-science-solutions
# - https://www.kaggle.com/code/yassineghouzam/titanic-top-4-with-ensemble-modeling
