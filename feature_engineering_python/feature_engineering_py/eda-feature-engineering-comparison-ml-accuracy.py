#!/usr/bin/env python
# coding: utf-8

# <h1> Welcome to my Titanic Kernel! </h1>

# Here you find basic Data Exploration and Visualization, data handling with some features, and modelling.
# 
# **This Kernel Focus on the effects of some features in the performance of the learning algorithms** 
# 
# 
# I used most the common supervise learning classification algorithms. 
# I compared them in a train/test set and I chose some for submiting the answers

#  Table of Contents:
#  
#  **1. [Introduction & Imports](#Introduction)** <br>
#  **2. [Exploratory Data Analysis](#EDA)** <br>
#  **3. [Feature Engineering](#Feature)** <br>
#  **4. [Preparing the Test dataframe](#test)** <br>
#  **5. [Testing several Supervise learning models](#ML)** <br>
#  **6. [Trainning all data on some Classifiers](#train)** <br>
#  **7. [Results](#results)** <br>
# 

# <a id="Introduction"></a> <br> 
#  **1. Introduction & Imports** 
# 

# Import some libraries for data exploration

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

import os
print(os.listdir("../input"))


# **Import Data & Exploratory Data Analysis**

# In[2]:


train = pd.read_csv("../input/train.csv")
#drop cabin, Name and Ticket data that are not neccesary to train the model
train.head()


# <a id="EDA"></a> <br> 
#  **2. Exploratory Data Analysis**

# The first step is the detect in which columns there are non valid values

# In[3]:


#Check for the missing values in the columns 
fig, ax = plt.subplots(figsize=(9,5))
sns.heatmap(train.isnull(), cbar=False, cmap="YlGnBu_r")
plt.show()


# Supposing that the categorical values such as the Name, the Cabin, the Ticket code and the ID doesnt have any relationship to the fact that the passanger died or survived:

# In[4]:


#I drop those columns
train = train.drop(columns = ['Cabin','Name','Ticket','PassengerId'])


# In[5]:


#filling Non valid values with mean for age, 
train['Age'].fillna((train['Age'].mean()), inplace=True)


# **Survival as function of Pclass and Sex**
# 

# To start the exploration, it is possible to group passanger by Sex and Class, these groups could give insights if higher class have better chance of survive, or woman have better chance than men for example.

# In[6]:


sns.barplot(x='Sex', y='Survived', data=train)
plt.ylabel("Survival Rate")
plt.title("Survival as function of Sex", fontsize=16)

plt.show()
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[7]:


#This information can be displayed in the next plot too:
#sns.catplot(x='Sex', col='Survived', kind='count', data=train);


# It is clear that women have better chance than men. 
# If you create a model saying that only woman survive it would have a score of **0.76555**, so the mission is to create a model at least better
# 
# Next, I explore the change of survive regardign the passanger Class:

# In[8]:


sns.barplot(x='Pclass', y='Survived', data=train)
plt.ylabel("Survival Rate")
plt.title("Survival as function of Pclass", fontsize=16)

plt.show()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


#  I explore both PClass and Sex in the same plot:

# In[9]:


sns.barplot(x='Sex', y='Survived', hue='Pclass', data=train)
plt.ylabel("Survival Rate")
plt.title("Survival as function of Pclass and Sex")
plt.show()


# Next, Explore the Parch and SibSp column:

# In[10]:


train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[11]:


train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# To get a better insight of the relationship of these features and the survival rate, a general pairplot will give some clues:

# In[12]:


sns.pairplot(data=train, hue="Survived")


# <a id="Feature"></a> <br> 
#  **3. Feature Engineering**

# The features can be built to:
# - reduce the number of states of the SibSp and Parch column
# - Create smaller classes for continues columns, such as Age and Fare
# - Create new columns that could improve prediction: such as if the passanger is alone or not
# - Drop columns that doesn't improve predictions

# The first features to work on are SibSp and Parch

# In[13]:


# I Create a swarmplot to detect patterns, where is the highest survival rate? 
sns.swarmplot(x = 'SibSp', y = 'Parch', hue = 'Survived', data = train, split = True, alpha=0.8)
plt.show()


# To explore better the relationship between these variables before featuring, I create a first model:

# In[14]:


from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

yf = train.Survived
base_features = ['Parch',
                 'SibSp','Age', 'Fare','Pclass']

Xf = train[base_features]

train_X, val_X, train_y, val_y = train_test_split(Xf, yf, random_state=1)
first_model = RandomForestRegressor(n_estimators=21, random_state=1).fit(train_X, train_y)


# In[15]:


#Explore the relationship between SipSp and Parch in the predictions for a RF Model
inter  =  pdp.pdp_interact(model=first_model, dataset=val_X, model_features=base_features, features=['SibSp', 'Parch'])

pdp.pdp_interact_plot(pdp_interact_out=inter, feature_names=['SibSp', 'Parch'], plot_type='contour')
plt.show()


# Then, introducing new features as Family size (to join these Parch and SibSp)

# In[16]:


train['FamilySize'] = train['SibSp'] + train['Parch'] 
train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).agg('mean')


# The next option is to cerate IsAlone feature to check wheter a person traveling alolne is more likely to survived or died

# In[17]:


train['IsAlone'] = 0
train.loc[train['FamilySize'] == 0, 'IsAlone'] = 1

train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# To sum up the work, the next set of graphics shows the relationships with and without the new features

# In[18]:


cols = ['Survived', 'Parch', 'SibSp', 'Embarked','IsAlone', 'FamilySize']

nr_rows = 2
nr_cols = 3

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))

for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        
        i = r*nr_cols+c       
        ax = axs[r][c]
        sns.countplot(train[cols[i]], hue=train["Survived"], ax=ax)
        ax.set_title(cols[i], fontsize=14, fontweight='bold')
        ax.legend(title="survived", loc='upper center') 
        
plt.tight_layout()


# **The Fare Column**

# This continus feature could be converted in a continues feature in order to increase prediction of the model

# In[19]:


feat_name = 'Fare'
pdp_dist = pdp.pdp_isolate(model=first_model, dataset=val_X, model_features=base_features, feature=feat_name)
pdp.pdp_plot(pdp_dist, feat_name)
plt.show()


# The fare is distributed in several continues values, and it is not clear how can we discretize these values to improve model's performance.

# To solve this problem, first, it would be likely to think that the chance of survival could depend on the Fare

# In[20]:


train[["Fare", "Survived"]].groupby(['Survived'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Highest fare mean improve the chance of survival
# We know that female have better chance than male, so we group the data in these values:

# In[21]:


train.groupby(['Sex','Survived'])[['Fare']].agg(['min','mean','max'])


# Based on the exploration of the data, I propose to discretize the Fare in four states:

# In[22]:


train.loc[ train['Fare'] <= 7.22, 'Fare'] = 0
train.loc[(train['Fare'] > 7.22) & (train['Fare'] <= 21.96), 'Fare'] = 1
train.loc[(train['Fare'] > 21.96) & (train['Fare'] <= 40.82), 'Fare'] = 2
train.loc[ train['Fare'] > 40.82, 'Fare'] = 3
train['Fare'] = train['Fare'].astype(int)


# In[23]:


g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Fare', bins=20)
plt.show()


# In[24]:


sns.barplot(x='Sex', y='Survived', hue='Fare', data=train)
plt.ylabel("Survival Rate")
plt.title("Survival as function of Fare and Sex")
plt.show()


# This plot show us how the new fare states are relatid to Sex and rate of survival. Higher fare have better chance of survive than lower fare, and female more than male in general.

# **Now the relationship between age and survived**

# Age has continue values too:

# In[25]:


g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
plt.show()


# I check in a first model how can age correlate with the chance of survive, also related to the passanger Class:

# In[26]:


feat_name = 'Age'
pdp_dist = pdp.pdp_isolate(model=first_model, dataset=val_X, model_features=base_features, feature=feat_name)
pdp.pdp_plot(pdp_dist, feat_name)
plt.show()
#Exploring the relationship between Age and Pclass for a given model preductions
inter  =  pdp.pdp_interact(model=first_model, dataset=val_X, model_features=base_features, features=['Age', 'Pclass'])

pdp.pdp_interact_plot(pdp_interact_out=inter, feature_names=['Age', 'Pclass'], plot_type='contour')
plt.show()


# It seems like less age and higher class is a better combination to survive. 

# In[27]:


#bins=np.arange(0, 80, 10)
g = sns.FacetGrid(train, row='Sex', col='Pclass', hue='Survived', margin_titles=True, size=3, aspect=1.1)
g.map(sns.distplot, 'Age', kde=False, bins=4, hist_kws=dict(alpha=0.6))
g.add_legend()  
plt.show()


# Following the graphics below, The age can be groupped into less classes:

# In[28]:


train.loc[ train['Age'] <= 16, 'Age'] = 1
train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 2
train.loc[(train['Age'] > 32) & (train['Age'] <= 64), 'Age'] = 3
train.loc[ train['Age'] > 64, 'Age'] = 4
train['Age'] = train['Age'].astype(int)


# In[29]:


sns.barplot(x='Pclass', y='Survived', hue='Age', data=train)
plt.ylabel("Survival Rate")
plt.title("Survival as function of Age and Sex")
plt.show()


# This plot show us how higher class and lower age have better chance of survive, while lower class (3) and older (age >2) have lower chance of survive.
# This seems logic, the reduction of classes can improve the learning of the model based on the (relative small) data we have  

# finally I explore new features, for example, a measure of 'Age x Class' would give better insight of the survival rate? 

# In[30]:


train['Age*Class'] = train.Age * train.Pclass


# In[31]:


train[["Age*Class", "Survived"]].groupby(['Age*Class'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Less values give a clue of more survival mean, however a crosstab maybe would give more clear information:

# In[32]:


pd.crosstab([train.Survived], [train.Sex,train['Age*Class']], margins=True).style.background_gradient(cmap='autumn_r')


# the new featre show how female in AgexClass between 2 to 6 have better chance of survive, and male from 4 to 6 AgexClass have lower chance of survive 

# The same analysis for the IsAlone feature:

# In[33]:


pd.crosstab([train.Survived], [train.Sex,train['IsAlone']], margins=True).style.background_gradient(cmap='autumn_r')


# men who were alone have lower chance of survive

# In[34]:


pd.crosstab([train.Survived], [train.Fare], margins=True).style.background_gradient(cmap='autumn_r')


# This last crosstab show how the people having Fare group 1 (Fare > 7.22 & Fare <= 21.96) have the lower chance of survive, 
# 
# The presented groupes show the tendency of the data. However is hard to know wheter these groups really optimize the larning. This work only can be done by trial and error

# **Estimation of the Survival rate using the new features defined **

# this is how the new train dataframe looks like:

# In[35]:


train.head()


# Now, I would simulate the training using the new features 

# In[36]:


y2 = train.Survived

base_features2 = ['Parch','SibSp','Age', 'Fare','Pclass','Age*Class','FamilySize','IsAlone']

X2 = train[base_features2]
train_X2, val_X2, train_y2, val_y2 = train_test_split(X2, y2, random_state=1)
second_model = RandomForestRegressor(n_estimators=21, random_state=1).fit(train_X2, train_y2)

inter2  =  pdp.pdp_interact(model=second_model, dataset=val_X2, model_features=base_features2, features=['Age', 'Pclass'])
pdp.pdp_interact_plot(pdp_interact_out=inter2, feature_names=['Age', 'Pclass'], plot_type='contour')
plt.show()


# These new features provide a more clear distribution that the dataframe without features:

# In[37]:


feat_name = 'FamilySize'
pdp_dist = pdp.pdp_isolate(model=second_model, dataset=val_X2, model_features=base_features2, feature=feat_name)
pdp.pdp_plot(pdp_dist, feat_name)
plt.show()


# In[38]:


inter2  =  pdp.pdp_interact(model=second_model, dataset=val_X2, model_features=base_features2, features=['FamilySize', 'Pclass'])
pdp.pdp_interact_plot(pdp_interact_out=inter2, feature_names=['FamilySize', 'Pclass'], plot_type='contour')
plt.show()


# Also for the Passanger Class and its family size. After defining the new groups, it's more clear for the algorithm that lower class and lower family size increase the chance of survive 

# Now I explore the effects of the other featuers independently

# In[39]:


feat_name = 'IsAlone'
pdp_dist = pdp.pdp_isolate(model=second_model, dataset=val_X2, model_features=base_features2, feature=feat_name)
pdp.pdp_plot(pdp_dist, feat_name)
plt.show()


# In[40]:


feat_name = 'Age*Class'
pdp_dist = pdp.pdp_isolate(model=second_model, dataset=val_X2, model_features=base_features2, feature=feat_name)
pdp.pdp_plot(pdp_dist, feat_name)
plt.show()


# For Is alone and AgexClass, the effect in the survival rate is not that clear

# In[41]:


inter2  =  pdp.pdp_interact(model=second_model, dataset=val_X2, model_features=base_features2, features=['Age*Class', 'IsAlone'])
pdp.pdp_interact_plot(pdp_interact_out=inter2, feature_names=['Age*Class', 'IsAlone'], plot_type='contour')
plt.show()


# As Sex and Embarked are not numerical I do the pandas OneHotEncoder:

# In[42]:


# convert Sex values and Embearked values into dummis to use a numerical classifier 
dummies_Sex = pd.get_dummies(train.Sex)
dummies_Embarked = pd.get_dummies(train.Embarked)
#join the dummies to the final dataframe
train_ready = pd.concat([train, dummies_Sex,dummies_Embarked], axis=1)
train_ready.head()


# and drop the respective columns:

# In[43]:


#Drop the columns that are not usefull now
#train_ready = train_ready.drop(columns = ['Sex','Embarked','male','SibSp','Parch','Q'])

train_ready = train_ready.drop(columns = ['Sex','Embarked'])


# the AgexClass can be dropped or not as I experiment to increase the general performance of the model in the next steps:

# In[44]:


#train_ready = train_ready.drop(columns = ['Age*Class'])


# same for the FamiliSize columns

# In[45]:


#train_ready = train_ready.drop(columns = ['FamilySize'])


# In[46]:


#alst check before trainning
train_ready.info()


# In[47]:


train_ready.head(10)


# I explore the entropy to check wheter the values can give a good learning to the algoritmh

# In[48]:


from scipy import stats
for name in train_ready:
    print(name, "column entropy :", round(stats.entropy(train_ready[name].value_counts(normalize=True), base=2),2))


# I train the model, then I came back and drop AgexClass Siwe column (entropy 2,14) then I train again the model droping the FamilySize column (entropy 1,82)

# <a id="test"></a> <br> 
# **4. Preparing the Test dataframe**
Here I complete the same steps for the test set
# In[49]:


#Upload the test file 
test = pd.read_csv("../input/test.csv")

#Drop unecessary columns
test = test.drop(columns = ['Cabin','Name','Ticket','PassengerId'])
#check the test dataframe
test.head()


# In[50]:


#Check for the missing values in the columns 
fig, ax = plt.subplots(figsize=(9,5))
sns.heatmap(test.isnull(), cbar=False, cmap="YlGnBu_r")
plt.show()


# In[51]:


#filling Non valid values with mean for age, 
test['Age'].fillna((test['Age'].mean()), inplace=True)
test['Fare'].fillna((test['Fare'].mean()), inplace=True)


# In[52]:


test.loc[ test['Fare'] <= 7.22, 'Fare'] = 0
test.loc[(test['Fare'] > 7.22) & (test['Fare'] <= 21.96), 'Fare'] = 1
test.loc[(test['Fare'] > 21.96) & (test['Fare'] <= 40.82), 'Fare'] = 2
test.loc[ test['Fare'] > 40.82, 'Fare'] = 3


# In[53]:


test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
test['IsAlone'] = 0
test.loc[test['FamilySize'] == 1, 'IsAlone'] = 1


# In[54]:


test.loc[ test['Age'] <= 16, 'Age'] = 1
test.loc[(test['Age'] > 16) & (test['Age'] <= 32), 'Age'] = 2
test.loc[(test['Age'] > 32) & (test['Age'] <= 64), 'Age'] = 3
test.loc[ test['Age'] > 64, 'Age'] = 4


# In[55]:


test['Age*Class'] = test.Age * test.Pclass


# In[56]:


test.info()


# In[57]:


#as in the train dataset, build dummis in the sex and embarked columns
test_dummies_Sex = pd.get_dummies(test.Sex)
test_dummies_Embarked = pd.get_dummies(test.Embarked)
test_ready = pd.concat([test, test_dummies_Sex,test_dummies_Embarked], axis=1)
test_ready.head()


# In[58]:


#drop these columns, we keep only numerical values
#train_ready = train_ready.drop(columns = ['Sex','Embarked','Survived','SibSp','Parch'])
test_ready = test_ready.drop(columns = ['Sex','Embarked'])


# When dropping the colmuns in the train dataset it would be neccesary to do the same in the test dataset:

# In[59]:


#test_ready = test_ready.drop(columns = ['Age*Class'])


# In[60]:


#test_ready = test_ready.drop(columns = ['FamilySize'])


# In[61]:


#check all is ok 
test_ready.info()


# In[62]:


test_ready.head()


# I explore the entropy to check wheter the values can give a good learning to the algoritmh

# In[63]:


from scipy import stats
for name in test_ready:
    print(name, "column entropy :",round(stats.entropy(test_ready[name].value_counts(normalize=True), base=2),2))


# <a id="ML"></a> <br> 
# **5. Testing several Supervise learning models** 
# 

# First, I would use a train/test division on the test csv, and I would check the performance of several algorithms:

# In[64]:


## import ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[65]:


# Create arrays for the features and the response variable
y = train_ready['Survived'].values
X = train_ready.drop('Survived',axis=1).values


# In[66]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=21, stratify=y)


# In[67]:


#Importing the auxiliar and preprocessing librarys 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import accuracy_score

#Models
import warnings
warnings.filterwarnings("ignore")

import eli5
from eli5.sklearn import PermutationImportance

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier, RandomTreesEmbedding


# In[68]:


clfs = []
seed = 3

clfs.append(("LogReg", 
             Pipeline([("Scaler", StandardScaler()),
                       ("LogReg", LogisticRegression())])))

clfs.append(("XGBClassifier",
             Pipeline([("Scaler", StandardScaler()),
                       ("XGB", XGBClassifier())]))) 
clfs.append(("KNN", 
             Pipeline([("Scaler", StandardScaler()),
                       ("KNN", KNeighborsClassifier(n_neighbors=8))]))) 

clfs.append(("DecisionTreeClassifier", 
             Pipeline([("Scaler", StandardScaler()),
                       ("DecisionTrees", DecisionTreeClassifier())]))) 

clfs.append(("RandomForestClassifier", 
             Pipeline([("Scaler", StandardScaler()),
                       ("RandomForest", RandomForestClassifier())]))) 

clfs.append(("GradientBoostingClassifier", 
             Pipeline([("Scaler", StandardScaler()),
                       ("GradientBoosting", GradientBoostingClassifier(n_estimators=100))]))) 

clfs.append(("RidgeClassifier", 
             Pipeline([("Scaler", StandardScaler()),
                       ("RidgeClassifier", RidgeClassifier())])))

clfs.append(("BaggingRidgeClassifier",
             Pipeline([("Scaler", StandardScaler()),
                       ("BaggingClassifier", BaggingClassifier())])))

clfs.append(("ExtraTreesClassifier",
             Pipeline([("Scaler", StandardScaler()),
                       ("ExtraTrees", ExtraTreesClassifier())])))

#'neg_mean_absolute_error', 'neg_mean_squared_error','r2'
scoring = 'accuracy'
n_folds = 7

results, names  = [], [] 

for name, model  in clfs:
    kfold = KFold(n_splits=n_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, 
                                 cv= 5, scoring=scoring,
                                 n_jobs=-1)    
    names.append(name)
    results.append(cv_results)    
    msg = "%s: %f (+/- %f)" % (name, cv_results.mean(),  cv_results.std())
    print(msg)
    
# boxplot algorithm comparison
fig = plt.figure(figsize=(15,6))
fig.suptitle('Classifier Algorithm Comparison', fontsize=22)
ax = fig.add_subplot(111)
sns.boxplot(x=names, y=results)
ax.set_xticklabels(names)
ax.set_xlabel("Algorithmn", fontsize=20)
ax.set_ylabel("Accuracy of Models", fontsize=18)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
plt.show()


# In[69]:


perm_xgb = PermutationImportance(XGBClassifier().fit(X_train, y_train), random_state=1).fit(X_test,y_test)
eli5.show_weights(perm_xgb, feature_names = train_ready.drop('Survived',axis=1).columns.tolist())


# In[70]:


perm_knn = PermutationImportance(KNeighborsClassifier(n_neighbors=8).fit(X_train, y_train), random_state=1).fit(X_test,y_test)
eli5.show_weights(perm_knn, feature_names = train_ready.drop('Survived',axis=1).columns.tolist())


# In[71]:


perm_gbc = PermutationImportance(GradientBoostingClassifier(n_estimators=100).fit(X_train, y_train), random_state=1).fit(X_test,y_test)
eli5.show_weights(perm_gbc, feature_names = train_ready.drop('Survived',axis=1).columns.tolist())


# In[72]:


perm_gbc = PermutationImportance(RidgeClassifier().fit(X_train, y_train), random_state=1).fit(X_test,y_test)
eli5.show_weights(perm_gbc, feature_names = train_ready.drop('Survived',axis=1).columns.tolist())


# In[73]:


#train_ready.drop('Survived',axis=1).columns


# In[74]:


train_ready.drop('Survived',axis=1).info()


# In[75]:


train_ready.shape


# ***Here I can drop the columns as AgexClass and IsAlone to check wheter the algorithms produce better performance ***

# **Bulding the model for the test set **

# In[76]:


#apply Scla to train in order to standardize data 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X)
scaled_features = scaler.transform(X)
train_sc = pd.DataFrame(scaled_features) # columns=df_train_ml.columns[1::])

#apply Scla to test csv (new file)  in order to standardize data 

X_csv_test = test_ready.values  #X_csv_test the new data that is going to be test 
scaler.fit(X_csv_test)
scaled_features_test = scaler.transform(X_csv_test)
test_sc = pd.DataFrame(scaled_features_test) # , columns=df_test_ml.columns)


# In[77]:


scaled_features_test.shape


# In[78]:


scaled_features.shape


# <a id="train"></a> <br> 
# **6. Trainning all data on several Classifier**

# **First Model: KNN**

# First we run this loop to detect the correct number of Nieghbors in KNN

# In[79]:


# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier 

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 19)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')


# **This plot vary depending the features in the train dataframe **

# I will keep two of these models, with 6 neighbors and with 10 neighbors

# In[80]:


# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier 

# Create a k-NN classifier with 6 neighbors: knn
knn_6 = KNeighborsClassifier(n_neighbors = 6)

# Fit the classifier to the data
knn_6.fit(scaled_features,y)

# Predict the labels for the training data X
y_pred_knn_6 = knn_6.predict(scaled_features_test)


# In[81]:


# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier 

# Create a k-NN classifier with 6 neighbors: knn
knn_10 = KNeighborsClassifier(n_neighbors = 10)

# Fit the classifier to the data
knn_10.fit(scaled_features,y)

# Predict the labels for the training data X
y_pred_knn_10 = knn_10.predict(scaled_features_test)


# In[82]:


#Upload the test file for KNN (scaled)
result_knn_6 = pd.read_csv("../input/gender_submission.csv")
result_knn_6['Survived'] = y_pred_knn_6
result_knn_6.to_csv('Titanic_knn_5.csv', index=False)


# In[83]:


#Upload the test file for KNN (scaled)
result_knn_10 = pd.read_csv("../input/gender_submission.csv")
result_knn_10['Survived'] = y_pred_knn_10
result_knn_10.to_csv('Titanic_knn_7.csv', index=False)


# **Second model: Logistic Regression**

# In[84]:


logreg = LogisticRegression()
logreg.fit(scaled_features,y)
y_pred_logreg = logreg.predict(scaled_features_test)
y_pred_logreg.shape


# In[85]:


#Upload the test file for Random Forest 
result_logreg = pd.read_csv("../input/gender_submission.csv")
result_logreg['Survived'] = y_pred_logreg
result_logreg.to_csv('Titanic_logreg.csv', index=False)


# **Third model : XGB Classifier**

# In[86]:


import xgboost as xgb
from xgboost import XGBClassifier

clf = xgb.XGBClassifier(n_estimators=250, random_state=4,bagging_fraction= 0.791787170136272, colsample_bytree= 0.7150126733821065,feature_fraction= 0.6929758008695552,gamma= 0.6716290491053838,learning_rate= 0.030240003246947006,max_depth= 2,min_child_samples= 5,num_leaves= 15,reg_alpha= 0.05822089056228967,reg_lambda= 0.14016232510869098,subsample= 0.9)

clf.fit(scaled_features, y)

y_pred_xgb= clf.predict(scaled_features_test)


# In[87]:


#Upload the test file for Random Forest 
result_xgb = pd.read_csv("../input/gender_submission.csv")
result_xgb['Survived'] = y_pred_xgb
result_xgb.to_csv('Titanic_xgb.csv', index=False)


# **4th Model RidgeClassifier**

# In[88]:


rcf= RidgeClassifier()
rcf.fit(scaled_features, y)

y_pred_rcf= rcf.predict(scaled_features_test)


# In[89]:


#Upload the test file for  Ridge Classifier
result_rcf = pd.read_csv("../input/gender_submission.csv")
result_rcf['Survived'] = y_pred_rcf
result_rcf.to_csv('Titanic_rcf.csv', index=False)


# **5th model: Gradient Boosting Classifier**

# In[90]:


gbc= GradientBoostingClassifier(n_estimators=100)
gbc.fit(scaled_features, y)
y_pred_gbc= gbc.predict(scaled_features_test)


# In[91]:


#Upload the test file for Bagging Ridge Classifie
result_gbc = pd.read_csv("../input/gender_submission.csv")
result_gbc['Survived'] = y_pred_gbc
result_gbc.to_csv('Titanic_gbc.csv', index=False)


# <a id="results"></a> <br> 
# **7. Results** 
# 

# We tested 4 options, first one training without any change in the features

# ![](https://cdn-images-1.medium.com/max/1000/1*kZ9X3rMW9-Ohxd2UKx-y1w.png)

# the next try included all the features created in the first section:

# ![](https://cdn-images-1.medium.com/max/1000/1*XbtLALV28nrdzBSG5IHm4g.png)

# Next, we include only with those features with entropy < 2, that is droping those columns that maybe add more noise than value

# ![](https://cdn-images-1.medium.com/max/1000/1*Rn1oQJHrDRcDs3_oM6ZxbA.png)

# it seems that the global accuracy of all the models is increasing
# Next, we select only those features with entropy < 1,5 :

# ![](https://cdn-images-1.medium.com/max/1000/1*ofXtRDzL5PtXvhIT0Qdk1A.png)

# It seems this feature combination give the better accuracy for all the algorithms

# Another point, for example, when trainning a KNN with several neighbords, the result depend on the features defined.
# We plot the accuracy of KNN for several neighbors:

# ![](https://cdn-images-1.medium.com/max/1000/1*xZwK315Z-wxV5EmqGI9y9Q.png)
# 

# After submitting the diferent CSV I have obatined this results:
# - Using Random Forest: 0.73684
# - Using KNN with 6 neighboors:  **0.77990**
# - Using KNN with 10 neighboors: 0.77033
# - Using KGBClassifier:** 0.77990**
# - Using Ridge Classifier: 0.77511

#  <font color="red">If this kernel were useful for you, please <b>UPVOTE</b> the kernel ;)</font>
