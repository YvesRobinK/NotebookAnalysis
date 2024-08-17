#!/usr/bin/env python
# coding: utf-8

# # Titanic Disaster
# ## Improve your score to 82.78% (Top 3%) 
# 
# In this work I have used some basic techniques to process of the easy way Titanic dataset. 

# # 1. Preprocessing and EDA

# Here, I reviewed the variables, impute missing values, found patterns and watched relationship between columns.

# ### 1.1. Missing Values

# Reading the dataset and merging Train and Test to get better results.

# In[1]:


# Libraries used

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

from numpy.random import seed

seed(11111)


# In[2]:


# Reading
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")

# Putting on index to each dataset before split it
train = train.set_index("PassengerId")
test = test.set_index("PassengerId")

# dataframe 
df = pd.concat([train, test], axis=0, sort=False)

df


# As you can see Name, Sex, Ticket, Cabin, and Embarked column are objects, before processing each column we should know if there are NAs or missing values.

# In[3]:


df.info()


# There are three columns with missing values (Age, Fare and Cabin) and Survived column has NaNs because the Test dataset doesn't have that information.   

# In[4]:


df.isna().sum()


# To visualize better the columns we will transform the Sex and Embarked columns to numeric. Sex column only has two categories Female and Male, Embarked column has tree labels S, C and Q.

# In[5]:


# Sex
change = {'female':0,'male':1}
df.Sex = df.Sex.map(change)

# Embarked
change = {'S':0,'C':1,'Q':2}
df.Embarked = df.Embarked.map(change)


# The following figure show us numeric columns vs Survived column to know the behavior. In the last fig (3,3) you can see that we are working with unbalanced dataset. 

# In[6]:


columns = ['Pclass', 'Sex','Embarked','SibSp', 'Parch','Survived']

plt.figure(figsize=(16, 14))
sns.set(font_scale= 1.2)
sns.set_style('ticks')

for i, feature in enumerate(columns):
    plt.subplot(3, 3, i+1)
    sns.countplot(data=df, x=feature, hue='Survived', palette='Paired')
    
sns.despine()


# In[7]:


columns = ['Pclass', 'Sex','Embarked','SibSp', 'Parch','Survived']

plt.figure(figsize=(16, 14))
sns.set(font_scale= 1.2)
sns.set_style('ticks')

for i, feature in enumerate(columns):
    plt.subplot(3, 3, i+1)
    sns.countplot(data=df, x=feature, hue='Sex', palette='BrBG')
    
sns.despine()


# ### 1.2. Age column

# The easy way to impute the missing values is with mean or median on base its correlation with other columns. Below you can see the correlation beetwen variables, Pclass has a good correlation with Age, but I also added Sex column to impute missing values.

# In[8]:


corr_df = df.corr()
fig, axs = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_df).set_title("Correlation Map",fontdict= { 'fontsize': 20, 'fontweight':'bold'});


# In[9]:


df.groupby(['Pclass','Sex','Survived'])['Age'].median()


# In[10]:


#Filling the missing values with mean of Pclass and Sex.
df["Age"].fillna(df.groupby(['Pclass','Sex'])['Age'].transform("mean"), inplace=True)


# In[11]:


fig, axs = plt.subplots(figsize=(10, 5))
sns.histplot(data=df, x='Age').set_title("Age distribution",fontdict= { 'fontsize': 20, 'fontweight':'bold'});
sns.despine()


# Let's binning the columns to process it the best way.

# In[12]:


auxage = pd.cut(df['Age'], 4)
fig, axs = plt.subplots(figsize=(15, 5))
sns.countplot(x=auxage, hue='Survived', data=df).set_title("Age Bins",fontdict= { 'fontsize': 20, 'fontweight':'bold'});
sns.despine()


# In[13]:


# converting to categorical
df['Age'] = LabelEncoder().fit_transform(auxage) 


# In[14]:


pd.crosstab(df['Age'], df['Survived'])


# ### 1.3. Fare column

# Fare has only one missing value and I imputed with the median or moda

# In[15]:


df["Fare"].fillna(df.groupby(['Pclass', 'Sex'])['Fare'].transform("median"), inplace=True)


# In[16]:


auxfare = pd.cut(df['Fare'],5)
fig, axs = plt.subplots(figsize=(15, 5))
sns.countplot(x=auxfare, hue='Survived', data=df).set_title("Fare Bins",fontdict= { 'fontsize': 20, 'fontweight':'bold'});
sns.despine()


# In[17]:


df['Fare'] = LabelEncoder().fit_transform(auxfare) 


# In[18]:


pd.crosstab(df['Fare'], df['Survived'])


# ### 1.4. Embarked column

# Has two missing values.

# In[19]:


print("mean of embarked",df.Embarked.median())

df.Embarked.fillna(df.Embarked.median(), inplace = True)


# ### 1.5. Cabin column

# This column has many missing values and thats the reason I dropped it.

# In[20]:


print("Percentage of missing values in the Cabin column :" ,round(df.Cabin.isna().sum()/ len(df.Cabin)*100,2))


# In[21]:


df.drop(['Cabin'], axis = 1, inplace = True)


# # 2. Feature Extraction

# In this part I have used the Name column to extract the Title of each person.

# In[22]:


df['Title'] = df.Name.str.extract('([A-Za-z]+)\.', expand = False)


# In[23]:


df.Title.value_counts()


# The four titles most ocurring are Mr, Miss, Mrs and Master. 

# In[24]:


least_occuring = ['Rev','Dr','Major', 'Col', 'Capt','Jonkheer','Countess']

df.Title = df.Title.replace(['Ms', 'Mlle','Mme','Lady'], 'Miss')
df.Title = df.Title.replace(['Countess','Dona'], 'Mrs')
df.Title = df.Title.replace(['Don','Sir'], 'Mr')

df.Title = df.Title.replace(least_occuring,'Rare')

df.Title.unique()


# In[25]:


pd.crosstab(df['Title'], df['Survived'])


# In[26]:


df['Title'] = LabelEncoder().fit_transform(df['Title']) 


# ## 2.1. SibSp and Parch column

# In[27]:


# I got the total number of each family adding SibSp and Parch. (1) is the same passenger.
df['FamilySize'] = df['SibSp'] + df['Parch']+1
df.drop(['SibSp','Parch'], axis = 1, inplace = True)


# In[28]:


fig, axs = plt.subplots(figsize=(15, 5))
sns.countplot(x='FamilySize', hue='Survived', data=df).set_title("Raw Column",fontdict= { 'fontsize': 20, 'fontweight':'bold'});
sns.despine()


# In[29]:


# Binning FamilySize column
df.loc[ df['FamilySize'] == 1, 'FamilySize'] = 0                            # Alone
df.loc[(df['FamilySize'] > 1) & (df['FamilySize'] <= 4), 'FamilySize'] = 1  # Small Family 
df.loc[(df['FamilySize'] > 4) & (df['FamilySize'] <= 6), 'FamilySize'] = 2  # Medium Family
df.loc[df['FamilySize']  > 6, 'FamilySize'] = 3                             # Large Family 


# In[30]:


fig, axs = plt.subplots(figsize=(15, 5))
sns.countplot(x='FamilySize', hue='Survived', data=df).set_title("Variable Bined",fontdict= { 'fontsize': 20, 'fontweight':'bold'});
sns.despine()


# ### 2.2. Ticket column

# With the following lambda function I got the ticket's number and I changed the LINE ticket to zero.

# In[31]:


df['Ticket'] = df.Ticket.str.split().apply(lambda x : 0 if x[:][-1] == 'LINE' else x[:][-1])


# In[32]:


df.Ticket = df.Ticket.values.astype('int64')


# ### 2.3. Name Column

# To get a better model,I got the Last Name of each passenger.

# In[33]:


df['LastName'] = last= df.Name.str.extract('^(.+?),', expand = False)


# ### 2.4. Woman or Child column

# Here, I created a new column to know if the passenger is woman a child, I selected the Title parameter because most of children less than 16 years have the master title.

# In[34]:


df['WomChi'] = ((df.Title == 0) | (df.Sex == 0))


# ### 2.4 Family Survived Rate column

# In this part I created three new columns FTotalCount, FSurviviedCount and FSurvivalRate, the F is of Family.  FTotalCount uses a lambda function to count of the WomChi column on base of LastName, PClass and Ticked  detect families and then subtract the same passanger with a boolean process the passenger is woman or child. FSurvivedCount also uses a lambda function to sum WomChi column and then with mask function filters if the passenger is woman o child subtract the state of survival, and the last FSurvivalRate only divide FSurvivedCount and FTotalCount.
# 

# In[35]:


family = df.groupby([df.LastName, df.Pclass, df.Ticket]).Survived

df['FTotalCount'] = family.transform(lambda s: s[df.WomChi].fillna(0).count())
df['FTotalCount'] = df.mask(df.WomChi, (df.FTotalCount - 1), axis=0)

df['FSurvivedCount'] = family.transform(lambda s: s[df.WomChi].fillna(0).sum())
df['FSurvivedCount'] = df.mask(df.WomChi, df.FSurvivedCount - df.Survived.fillna(0), axis=0)

df['FSurvivalRate'] = (df.FSurvivedCount / df.FTotalCount.replace(0, np.nan))


# In[36]:


df.isna().sum()


# In[37]:


# filling the missing values
df.FSurvivalRate.fillna(0, inplace = True)
df.FTotalCount.fillna(0, inplace = True)
df.FSurvivedCount.fillna(0, inplace = True)


# In[38]:


# You can review the result Family Survival Rate with these Families Heikkinen, Braund, Rice, Andersson,
# Fortune, Asplund, Spector,Ryerson, Allison, Carter, Vander, Planke

df[df['LastName'] == "Dean"]


# # 3. Modeling

# In[39]:


df['PassengerId'] = df.index


# In[40]:


df = pd.get_dummies(df, columns=['Sex','Fare','Pclass'])


# In[41]:


df.drop(['Name','LastName','WomChi','FTotalCount','FSurvivedCount','Embarked','Title'], axis = 1, inplace = True)


# In[42]:


df.columns


# In[43]:


# I splitted df to train and test
train, test = df.loc[train.index], df.loc[test.index]

X_train = train.drop(['PassengerId','Survived'], axis = 1)
Y_train = train["Survived"]
train_names = X_train.columns

X_test = test.drop(['PassengerId','Survived'], axis = 1)


# In[44]:


corr_train = X_train.corr()
fig, axs = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_train).set_title("Correlation Map",fontdict= { 'fontsize': 20, 'fontweight':'bold'});
plt.show()


# In[45]:


# Scaler
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)


# In[46]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_predDT = decision_tree.predict(X_test)

print("Accuracy of the model: ",round(decision_tree.score(X_train, Y_train) * 100, 2))


# In[47]:


importances = pd.DataFrame(decision_tree.feature_importances_, index = train_names)
importances.sort_values(by = 0, inplace=True, ascending = False)
importances = importances.iloc[0:6,:] 

plt.figure(figsize=(8, 5)) 
sns.barplot(x=0, y=importances.index, data=importances,palette="deep").set_title("Feature Importances",
                                                                                 fontdict= { 'fontsize': 20,
                                                                                            'fontweight':'bold'});
sns.despine()


# In[48]:


submit = pd.DataFrame({"PassengerId":test.PassengerId, 'Survived':Y_predDT.astype(int).ravel()})
submit.to_csv("submissionJavier_Vallejos.csv",index = False)


# # 4. Conclutions

# This report is part of a bootcamp of Data Science, and as you can see I achieved to be on the Top 3%. In the fist part I did an analysis to visualize each column and impute their missing values. After that I applied feature engineering to extract the title, last name of the Name column and Family Size is the adding of SibSp and Parch plus one that means the same passenger. Age and Fare columns have been Binning to get better results. To get Family Survival Rate is base on two rules:
# 
# 1. All males die except boys in families where all females and boys live.
# 2. All females live except those in families where all females and boys die.
# 
# With rules above you can get an score near to 81% but if you add the ticket number and other changes that I did you can improve it to 82.78% on Kaggle leaderboard.
# 
# To the model part I used only Desicion tree because is the easy way to getting this score.
# 
# Finally, if you want to increase your score, then I suggest you read this [work](https://www.kaggle.com/cdeotte/titanic-wcg-xgboost-0-84688). and like Chris Deotte said in his [post](https://www.kaggle.com/c/titanic/discussion/57447) this is the fist step to improve your score. 
# 
# 

# # 5. References

# * [Advanced Feature Engineering Tutorial](https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial)
# * [Top 5% Titanic: Machine Learning from Disaster](https://www.kaggle.com/kpacocha/top-5-titanic-machine-learning-from-disaster)
# * [Titanic - Top score : one line of the prediction](https://www.kaggle.com/vbmokin/titanic-top-score-one-line-of-the-prediction?scriptVersionId=42197143&select=survived.csv)
# * [Titanic survival prediction from Name and Sex](https://www.kaggle.com/mauricef/titanic)
# * [Titanic Dive Through: Feature scaling and outliers](https://www.kaggle.com/allunia/titanic-dive-through-feature-scaling-and-outliers)
# * [Titanic (Top 20%) with ensemble VotingClassifier](https://www.kaggle.com/amiiiney/titanic-top-20-with-ensemble-votingclassifier#5--Machine-Learning)
# * [Titanic Survival Rate](https://www.kaggle.com/prakharrathi25/titanic-survival-rate#Titanic-Survival-Prediction)
# 
