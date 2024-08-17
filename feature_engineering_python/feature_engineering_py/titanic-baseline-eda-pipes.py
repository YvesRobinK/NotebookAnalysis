#!/usr/bin/env python
# coding: utf-8

# <h1> Welcome to my Titanic Kernel! </h1>
# This kernel covers: Data Exploration and Visualization, data handling and modelling, Features preprocessing, ML pipeline, automated hyper parameter with HyperOpt and prediction of a dependent variable ('survived').  
# 
# When I started on Data Science field, my second work on Kaggle was on titanic Dataset and now, I want to improve my work here.
# 
# I will use a easy code that maybe could be useful to many people that are starting on Data Science or PyData libraries.

# ## <font color="red">If this kernel were useful for you, please <b>UPVOTE</b> the kernel =)</font>
# Also, don't forget to give me your feedback, it's many important to me

# If you want many other simple kernels with pythonic code <a href="https://www.kaggle.com/kabure/kernels">CLICK HERE</a> <br>
# 
# 

# <i>*I'm from Brazil, so english is not my first language, sorry about some mistakes</i>

# # Table of Contents:
# 
# **1. [Introduction](#Introduction)** <br>
# **2. [Librarys](#Librarys)** <br>
# **3. [Knowning the data](#Known)** <br>
# **4. [Exploring some Variables](#Explorations)** <br>
# **5. [Preprocessing](#Prepocess)** <br>
# **6. [Modelling](#Model)** <br>
# **7. [Validation](#Validation)** <br>
# 

# <a id="Introduction"></a> <br> 
# # **1. Introduction:** 
# <h3> The data have 891 entries on train dataset and 418 on test dataset</h3>
# - 10 columns in train_csv and 9 columns in train_test
# 

# <h2>Competition Description: </h2>
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

# <h3>Data Dictionary</h3><br>
# Variable	Definition	Key<br>
# <b>survival</b>	Survival	0 = No, 1 = Yes<br>
# <b>pclass</b>	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd<br>
# <b>sex</b>	Sex	<br>
# <b>Age</b>	Age in years	<br>
# <b>sibsp</b>	# of siblings / spouses aboard the Titanic	<br>
# <b>parch</b>	# of parents / children aboard the Titanic	<br>
# <b>ticket</b>	Ticket number	<br>
# <b>fare</b>	Passenger fare	<br>
# <b>cabin</b>	Cabin number	<br>
# <b>embarked	</b>Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton<br>
# <h3>Variable Notes</h3><br>
# <b>pclass: </b>A proxy for socio-economic status (SES)<br>
# 1st = Upper<br>
# 2nd = Middle<br>
# 3rd = Lower<br>
# <b>age: </b>Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5<br>
# <b>sibsp:</b> The dataset defines family relations in this way...<br>
# - <b>Sibling </b>= brother, sister, stepbrother, stepsister<br>
# - <b>Spouse </b>= husband, wife (mistresses and fiancés were ignored)<br>
# 
# <b>parch: </b>The dataset defines family relations in this way...<br>
# - <b>Parent</b> = mother, father<br>
# - <b>Child </b>= daughter, son, stepdaughter, stepson<br>
# 
# Some children travelled only with a nanny, therefore parch=0 for them.<br>

# I am using the beapproachs as possible but if you think I can do anything another best way, please, let me know.

# <a id="Librarys"></a> <br> 
# # **2. Importing Librarys:** 

# In[1]:


#This librarys is to work with matrices
import pandas as pd 
# This librarys is to work with vectors
import numpy as np
# This library is to create some graphics algorithmn
import seaborn as sns
# to render the graphs
import matplotlib.pyplot as plt
# import module to set some ploting parameters
from matplotlib import rcParams
# Library to work with Regular Expressions
import re
import gc

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from xgboost import XGBClassifier
import xgboost as xgb

## Hyperopt modules
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
from functools import partial

from scipy import stats

# This function makes the plot directly on browser
get_ipython().run_line_magic('matplotlib', 'inline')

# Seting a universal figure size 
rcParams['figure.figsize'] = 12,5


# ## Importing Datasets

# In[2]:


# Importing train dataset
df_train = pd.read_csv("../input/titanic/train.csv")

# Importing test dataset
df_test = pd.read_csv("../input/titanic/test.csv")

submission = pd.read_csv("../input/titanic/gender_submission.csv", index_col='PassengerId')


# <a id="Known"></a> <br> 
# # **3. First look at the data:** 
# - I will implement a function to we summary all columns in a meaningful table.

# In[3]:


def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary


# ## Summary of df train

# In[4]:


resumetable(df_train)


# Cool!! We can see very important information about all our data. <br>
# Our target is "Survived" column that informs if the passenger survived or not the disaster

# ## Summary of df test

# In[5]:


resumetable(df_test)


# In df test, we have missing values only on 

# <a id="Known"></a> <br> 
# # **4. Exploring the data:** 
# - Different of the other Kernel, as now I am more experienced in data science, I will start by the target distribution

# In[6]:


df_train['Survived'].replace({0:'No', 1:'Yes'}, inplace=True)


# In[7]:


total = len(df_train)
plt.figure(figsize=(12,7))
#plt.subplot(121)
g = sns.countplot(x='Survived', data=df_train, color='green')
g.set_title(f"Passengers alive or died Distribution \nTotal Passengers: {total}", 
            fontsize=22)
g.set_xlabel("Passenger Survived?", fontsize=18)
g.set_ylabel('Count', fontsize=18)
for p in g.patches:
    height = p.get_height()
    g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=15) 
g.set_ylim(0, total *.70)

plt.show()


# Cool. We can see that only 38.38% of passengers survived. <br>
# Let's explore our other featuers and try to find some patterns

# # Let's start exploring Age Column
# I will start by the "simplest" columns that are columns that don't need some transformations or have only few unique values.
# The first objective is to:
# - Explore the features
# - Imput missing values
# - See the distribution of numerical and categorical features
# - Understand the difference between groups that survived and not

# In[8]:


#First I will look my distribuition without NaN's
#I will create a df to look distribuition 
age_high_zero_died = df_train[(df_train["Age"] > 0) & 
                              (df_train["Survived"] == 'No')]
age_high_zero_surv = df_train[(df_train["Age"] > 0) & 
                              (df_train["Survived"] == 'Yes')]

#figure size
plt.figure(figsize=(16,5))

plt.subplot(121)
plt.suptitle('Age Distributions', fontsize=22)
sns.distplot(df_train[(df_train["Age"] > 0)]["Age"], bins=24)
plt.title("Distribuition of Age",fontsize=20)
plt.xlabel("Age Range",fontsize=15)
plt.ylabel("Probability",fontsize=15)

plt.subplot(122)

sns.distplot(age_high_zero_surv["Age"], bins=24, color='r', label='Survived')
sns.distplot(age_high_zero_died["Age"], bins=24, color='blue', label='Not Survived')
plt.title("Distribution of Age by Target",fontsize=20)
plt.xlabel("Age",fontsize=15)
plt.ylabel("Probability",fontsize=15)
plt.legend()


plt.show()


# Interesting! A big part of all passengers has between 20 to 40 old years. <br>
# When we analyze the distribution by the target we can note that youngest adults has a highest density in not survived passengers.
# 
# I will continue working in Age feature but before, I will try to understand the other columns. Maybe it could work well together 

# # Gender Column
# Understanding Gender distribution and distribution by target

# In[9]:


def plot_categoricals(df, col=None, cont='Age', binary=None, dodge=True):
    tmp = pd.crosstab(df[col], df[binary], normalize='index') * 100
    tmp = tmp.reset_index()

    plt.figure(figsize=(16,12))

    plt.subplot(221)
    g= sns.countplot(x=col, data=df, order=list(tmp[col].values) , color='green')
    g.set_title(f'{col} Distribuition', 
                fontsize=20)
    g.set_xlabel(f'{col} Values',fontsize=17)
    g.set_ylabel('Count Distribution', fontsize=17)
    sizes = []
    for p in g.patches:
        height = p.get_height()
        sizes.append(height)
        g.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.0f}'.format(height),
                ha="center", fontsize=15) 
    g.set_ylim(0,max(sizes)*1.15)

    plt.subplot(222)
    g1= sns.countplot(x=col, data=df, order=list(tmp[col].values),
                     hue=binary,palette="hls")
    g1.set_title(f'{col} Distribuition by {binary} ratio %', 
                fontsize=20)
    gt = g1.twinx()
    gt = sns.pointplot(x=col, y='Yes', data=tmp, order=list(tmp[col].values),
                       color='black', legend=False)
    gt.set_ylim(0,tmp['Yes'].max()*1.1)
    gt.set_ylabel("Survived %Ratio", fontsize=16)
    g1.set_ylabel('Count Distribuition',fontsize=17)
    g1.set_xlabel(f'{col} Values', fontsize=17)
    
    sizes = []
    
    for p in g1.patches:
        height = p.get_height()
        sizes.append(height)
        g1.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center", fontsize=10) 
    g1.set_ylim(0,max(sizes)*1.15)

    plt.subplot(212)
    g2= sns.swarmplot(x=col, y=cont, data=df, dodge=dodge, order=list(tmp[col].values),
                     hue="Survived",palette="hls")
    g2.set_title(f'{cont} Distribution by {col} and {binary}', 
                fontsize=20)
    g2.set_ylabel(f'{cont} Distribuition',fontsize=17)
    g2.set_xlabel(f'{col} Values', fontsize=17)


    plt.suptitle(f'{col} Distributions', fontsize=22)
    plt.subplots_adjust(hspace = 0.4, top = 0.90)
    
    plt.show()


# In[10]:


plot_categoricals(df_train, col='Sex', cont='Age', binary='Survived')


# Cool. Now we can see meaningful informations about the passengers. <br>
# The distribution of Ages and Gender by Survived can show us a interesting pattern in people who survived and who not

# # PClass
# - Other feature that I think that could be important to understand the passenger's survivors
# - Let's understand distributions of Pclass and how it is distributed in considering our target feature

# In[11]:


plot_categoricals(df_train, col='Pclass', cont='Age', binary='Survived')


# We can see that 55% of passengers are in the 3rd Class and also, is the Class where more people died. <br>
# Let's use the powerful <b>pd.crosstab</b> to see the distribution of Pclass by Sex and get the ratio of survivors

# In[12]:


(round(pd.crosstab(df_train['Survived'], [df_train['Pclass'], df_train['Sex']], 
             normalize='columns' ) * 100,2))


# I noted one interesting thing. <br>
# In first and second class the female have 92%+ of survivors, and in 3rd class the ratio is 50% in female survivors<br>
# Another interesting information is that in first class the percent of male survivors are almost 37% and in 2 and 3 class the ratio is 15.7 and 13.5 respectivelly.

# # Embarked Feature
# - Exploring the Distributions of the feature
# - Filling Na's values (we have only 2 missing values in this feature)

# In[13]:


plot_categoricals(df_train, col='Embarked', cont='Age', binary='Survived')


# Another interesting information. We can see that 72.4% of all passengers embarked in "S" (Southampton). <br>
# Also, 47% of all died passengers is from S.

# ### Let's fill na's in Embarked

# In[14]:


#lets input the NA's with the highest frequency
df_train["Embarked"] = df_train["Embarked"].fillna('S')


# ## Crossing Embarked by PClass and Survived

# In[15]:


(round(pd.crosstab(df_train['Survived'], [df_train['Embarked'], df_train['Pclass']], 
             normalize='columns' ) * 100,2))


# 

# ## Crossing Embarked by Sex and Survived

# In[16]:


(round(pd.crosstab(df_train['Survived'], [df_train['Embarked'], df_train['Sex']], 
             normalize='columns' ) * 100,2))


# We can see that we have we have different ratios when considering the Embarked place and Sex. It could be useful to build some features. 

# # Fare Column

# ## Looking quantiles of Fare 

# In[17]:


df_train['Fare'].quantile([.01, .1, .25, .5, .75, .9, .99]).reset_index()


# # Geting the Fare Log 

# In[18]:


df_train['Fare_log'] = np.log(df_train['Fare'] + 1)
df_test['Fare_log'] = np.log(df_test['Fare'] + 1)


# ## Ploting Fare Distribution

# In[19]:


# Seting the figure size
plt.figure(figsize=(16,10))

# Understanding the Fare Distribuition 
plt.subplot(221)
sns.distplot(df_train["Fare"], bins=50 )
plt.title("Fare Distribuition", fontsize=20)
plt.xlabel("Fare", fontsize=15)
plt.ylabel("Density",fontsize=15)

plt.subplot(222)
sns.distplot(df_train["Fare_log"], bins=50 )
plt.title("Fare LOG Distribuition", fontsize=20)
plt.xlabel("Fare (Log)", fontsize=15)
plt.ylabel("Density",fontsize=15)

plt.subplot(212)
g1 = plt.scatter(range(df_train[df_train.Survived == 'No'].shape[0]),
                 np.sort(df_train[df_train.Survived == 'No']['Fare'].values), 
                 label='No Survived', alpha=.5)
g1 = plt.scatter(range(df_train[df_train.Survived == 'Yes'].shape[0]),
                 np.sort(df_train[df_train.Survived == 'Yes']['Fare'].values), 
                 label='Survived', alpha=.5)
g1= plt.title("Fare ECDF Distribution", fontsize=18)
g1 = plt.xlabel("Index")
g1 = plt.ylabel("Fare Amount", fontsize=15)
g1 = plt.legend()

plt.suptitle('Fare Distributions', fontsize=22)
plt.subplots_adjust(hspace = 0.4, top = 0.90)

plt.show()


# Cool. We can note that the big part of passengers paid less than USD 100. We can't see some difference between the survived or not group. <br>
# I will try cross Fare by other features and try to find some interesting patterns

# <br>
# Description of Fare variable<br>
# - Min: 0<br>
# - Median: 14.45<br>
# - Mean: 32.20<br>
# - Max: 512.32<br> 
# - Std: 49.69<br>
# 
# 

# # Categorical features by Fare

# In[20]:


def ploting_cat_group(df, col):
    plt.figure(figsize=(14,6))
    tmp = pd.crosstab(df['Survived'], df[col], 
                      values=df['Fare'], aggfunc='mean').unstack(col).reset_index().rename(columns={0:'FareMean'})
    g = sns.barplot(x=col, y='FareMean', hue='Survived', data=tmp)
    g.set_xlabel(f'{col} values', fontsize=18)
    g.set_ylabel('Fare Mean', fontsize=18)
    g.set_title(f"Fare Distribution by {col} ", fontsize=20)
    
    plt.show()


# ## Fare mean by Pclass

# In[21]:


ploting_cat_group(df_train, 'Pclass')


# The fist class passengers has highest Fare mean, that make many sense. But we can't see a high difference between second and third class Fare mean

# ## Fare mean by Embarked

# In[22]:


ploting_cat_group(df_train, 'Embarked')


# People of C has a highest Fare mean.

# ## Fare mean by Sex

# In[23]:


ploting_cat_group(df_train, 'Sex')


# We can see that We can infer that pooverty people had more probability to die.

# ## Fare by Age

# In[24]:


plt.figure(figsize=(14,6))
g = sns.scatterplot(x='Age', y='Fare_log', data=df_train, hue='Survived')
g.set_title('Fare Distribution by Age', fontsize= 22)
g.set_xlabel('Age Distribution', fontsize=18)
g.set_ylabel("Fare Log Distribution", fontsize=18)

plt.show()


# Cool! We can see see the Fare distribution by Age and confirm and infer some questions. For example, the first class, probably has a highest age mean. Let's confirm that. 

# In[25]:


df_train.groupby(['Survived', 'Pclass'])['Age'].mean().unstack('Survived').reset_index()


# Cool!!! Exactly what I tought. We can see that in 3 class we have a smallest Age mean

# # Names Column

# In[26]:


df_train['Name'].unique()[:10]


# We can note that all names have the titles of passengers. Let's use regex to extract titles of passengers.

# ### Let's see the extracted titles

# In[27]:


# Extracting the prefix of all Passengers
df_train['Title'] = df_train.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
df_test['Title'] = df_test.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))

(df_train['Title'].value_counts(normalize=True) * 100).head(5)


# ## Mapping the titles

# In[28]:


#Now, I will identify the social status of each title

Title_Dictionary = {
        "Capt":       "Officer",
        "Col":        "Officer",
        "Major":      "Officer",
        "Dr":         "Officer",
        "Rev":        "Officer",
        "Jonkheer":   "Royalty",
        "Don":        "Royalty",
        "Sir" :       "Royalty",
        "the Countess":"Royalty",
        "Dona":       "Royalty",
        "Lady" :      "Royalty",
        "Mme":        "Mrs",
        "Ms":         "Mrs",
        "Mrs" :       "Mrs",
        "Mlle":       "Miss",
        "Miss" :      "Miss",
        "Mr" :        "Mr",
        "Master" :    "Master"
}
    
# we map each title to correct category
df_train['Title'] = df_train.Title.map(Title_Dictionary)
df_test['Title'] = df_test.Title.map(Title_Dictionary)


# # Ploting Title Distributions

# In[29]:


plot_categoricals(df_train, col='Title', cont='Age', binary='Survived')


# Very interesting information. The data has 18 Officer's and we can note that Officer's have the highest Age mean. <br>
# Curiously we can note that "Master" has a very low Age distribution. It sounds very strange to me. 

# # Let's use some features to help us fill Age NaN's 

# In[30]:


#Let's group the median age by sex, pclass and title, to have any idea and maybe input in Age NAN's
age_group = df_train.groupby(["Sex","Pclass","Title"])["Age"]

#printing the variabe that we created by median
age_group.median().unstack('Pclass').reset_index()


# In[31]:


#inputing the values on Age Na's 
# using the groupby to transform this variables
df_train.loc[df_train.Age.isnull(), 'Age'] = df_train.groupby(['Sex','Pclass','Title']).Age.transform('median')
df_test.loc[df_train.Age.isnull(), 'Age'] = df_test.groupby(['Sex','Pclass','Title']).Age.transform('median')

# printing the total of nulls in Age Feature
print(df_train["Age"].isnull().sum())


# In[32]:


#df_train.Age = df_train.Age.fillna(-0.5)

#creating the intervals that we need to cut each range of ages
interval = (0, 5, 12, 18, 25, 35, 60, 120) 

#Seting the names that we want use to the categorys
cats = ['babies', 'Children', 'Teen', 'Student', 'Young', 'Adult', 'Senior']

# Applying the pd.cut and using the parameters that we created 
df_train["Age_cat"] = pd.cut(df_train.Age, interval, labels=cats)
df_test["Age_cat"] = pd.cut(df_test.Age, interval, labels=cats)

# Printing the new Category
df_train["Age_cat"].unique()


# # Ploting Age Cat Distributions - Fare

# In[33]:


plot_categoricals(df_train, col='Age_cat', cont='Fare', binary='Survived')


# 

# # Sibsp	feature
# this feature refers to siblings / spouses aboard the Titanic	
# 

# In[34]:


plot_categoricals(df_train, col='SibSp', cont='Age', binary='Survived')


# 

# # Parch	feature
# The feature refers to parents / children aboard the Titanic	<br>

# In[35]:


plot_categoricals(df_train, col='Parch', cont='Age', binary='Survived', dodge=False)


# 

# # Creating the Family Size feature

# In[36]:


#Create a new column and sum the Parch + SibSp + 1 that refers the people self
df_train["FSize"] = df_train["Parch"] + df_train["SibSp"] + 1
df_test["FSize"] = df_test["Parch"] + df_test["SibSp"] + 1

family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 
              5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large',
              11: 'Large'}

df_train['FSize'] = df_train['FSize'].map(family_map)
df_test['FSize'] = df_test['FSize'].map(family_map)


# # Ploting Family Size

# In[37]:


plot_categoricals(df_train, col='FSize', cont='Fare', binary='Survived', dodge=True)


# Cool!!! We can see that 60% of passengers are traveling alone. Curiosly, we can note that these people have a highest ratio of Not survived Passengers. <br>
# The chance to survive is highest to people with small families on the boat. 
# 

# # Keep thinking about Familys
# ### Extracting Sur Names 
# - Taking advantage that we are dealing with family features, lets extract sur name from Name Features

# In[38]:


## I saw this code in another kernel and it is very useful
## Link: https://www.kaggle.com/gunesevitan/advanced-feature-engineering-tutorial-with-titanic
import string

def extract_surname(data):    
    
    families = []
    
    for i in range(len(data)):        
        name = data.iloc[i]

        if '(' in name:
            name_no_bracket = name.split('(')[0] 
        else:
            name_no_bracket = name
            
        family = name_no_bracket.split(',')[0]
        title = name_no_bracket.split(',')[1].strip().split(' ')[0]
        
        for c in string.punctuation:
            family = family.replace(c, '').strip()
            
        families.append(family)
            
    return families


# In[39]:


df_train['Family'] = extract_surname(df_train['Name'])
df_test['Family'] = extract_surname(df_test['Name'])


# ## Ticket feature
# - Understanding and creading new feature

# In[40]:


df_train['Ticket'].value_counts()[:10]


# Ticket is a very sparse data, with many values. So, lets try associate it with Family's and feed our model
# 

# In[41]:


df_train['Ticket_Frequency'] = df_train.groupby('Ticket')['Ticket'].transform('count')
df_test['Ticket_Frequency'] = df_test.groupby('Ticket')['Ticket'].transform('count')


# In[42]:


# Creating a list of families and tickets that are occuring in both training and test set
non_unique_families = [x for x in df_train['Family'].unique() if x in df_test['Family'].unique()]
non_unique_tickets = [x for x in df_train['Ticket'].unique() if x in df_test['Ticket'].unique()]


# ## Let's see the means of Fare 

# In[43]:


df_train.groupby(['Survived', 'FSize'])['Fare'].mean().unstack('FSize').reset_index()


# Now it look's better and clearly

# In[44]:


#Filling the NA's with -0.5
df_train.Fare = df_train.Fare.fillna(-1)
df_test.Fare = df_test.Fare.fillna(-1)
#intervals to categorize
quant = (-1, 0, 12, 30, 80, 100, 200, 600)

#Labels without input values
label_quants = ['NoInf', 'quart_1', 'quart_2', 'quart_3', 'quart_4', 'quart_5', 'quart_6']

#doing the cut in fare and puting in a new column
df_train["Fare_cat"] = pd.cut(df_train.Fare, quant, labels=label_quants)
df_test["Fare_cat"] = pd.cut(df_test.Fare, quant, labels=label_quants)


# In[45]:


plot_categoricals(df_train, col='Fare_cat', cont='Age', binary='Survived', dodge=False)


# In[46]:


# Excellent implementation from: 
# https://www.kaggle.com/franjmartin21/titanic-pipelines-k-fold-validation-hp-tuning

def cabin_extract(df):
    return df['Cabin'].apply(lambda x: str(x)[0] if(pd.notnull(x)) else str('M'))

df_train['Cabin'] = cabin_extract(df_train)
df_test['Cabin'] = cabin_extract(df_test)


# In[47]:


plot_categoricals(df_train, col='Cabin', cont='Age', binary='Survived', dodge=True)


# ### Seting Cabin into Groups

# In[48]:


pd.crosstab(df_train['Cabin'], df_train['Pclass'])


# Based on informations of the boat and the confirmation of te crosstab.
# - ABC cabins are to first class
# - DE cabins are to first and second class
# - FG are majority to third class
# - M are the missing values
# - On the Boat Deck there were **6** rooms labeled as **T, U, W, X, Y, Z**, but only the **T** cabin is present in the dataset.

# In[49]:


df_train['Cabin'] = df_train['Cabin'].replace(['A', 'B', 'C'], 'ABC')
df_train['Cabin'] = df_train['Cabin'].replace(['D', 'E'], 'DE')
df_train['Cabin'] = df_train['Cabin'].replace(['F', 'G'], 'FG')
# Passenger in the T deck is changed to A
df_train.loc[df_train['Cabin'] == 'T', 'Cabin'] = 'A'

df_test['Cabin'] = df_test['Cabin'].replace(['A', 'B', 'C'], 'ABC')
df_test['Cabin'] = df_test['Cabin'].replace(['D', 'E'], 'DE')
df_test['Cabin'] = df_test['Cabin'].replace(['F', 'G'], 'FG')
df_test.loc[df_test['Cabin'] == 'T', 'Cabin'] = 'A'


# # End of EDA

# # Modelling
# - To a better understanding of the modelling part, I will delete df and train and 

# ## Dropping unecessary features

# In[50]:


from pandas.api.types import CategoricalDtype 
family_cats = CategoricalDtype(categories=['Alone', 'Small', 'Medium', 'Large'], ordered=True)


# In[51]:


df_train.FSize = df_train.FSize.astype(family_cats)
df_test.FSize = df_test.FSize.astype(family_cats)


# In[52]:


df_train.Age_cat = df_train.Age_cat.cat.codes
df_train.Fare_cat = df_train.Fare_cat.cat.codes
df_test.Age_cat = df_test.Age_cat.cat.codes
df_test.Fare_cat = df_test.Fare_cat.cat.codes
df_train.FSize = df_train.FSize.cat.codes
df_test.FSize = df_test.FSize.cat.codes


# In[53]:


#Now lets drop the variable Fare, Age and ticket that is irrelevant now
df_train.drop([ 'Ticket', 'Name'], axis=1, inplace=True)
df_test.drop(['Ticket', 'Name', ], axis=1, inplace=True)
#df_train.drop(["Fare", 'Ticket', 'Age', 'Cabin', 'Name', 'SibSp', 'Parch'], axis=1, inplace=True)
#df_test.drop(["Fare", 'Ticket', 'Age', 'Cabin', 'Name', 'SibSp', 'Parch'], axis=1, inplace=True)


# # Preprocessing

# Now we might have information enough to think about the model structure

# In[54]:


df_test['Survived'] = 'test'
df = pd.concat([df_train, df_test], axis=0, sort=False )


# # Encoding and getting Dummies of categorical features

# ### Encoding
# 

# In[55]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Family'] = le.fit_transform(df['Family'].astype(str))


# ### Dummies

# In[56]:


df = pd.get_dummies(df, columns=['Sex', 'Cabin', 'Embarked', 'Title'],\
                          prefix=['Sex', "Cabin", 'Emb', 'Title'], drop_first=True)

df_train, df_test = df[df['Survived'] != 'test'], df[df['Survived'] == 'test'].drop('Survived', axis=1)
del df


# In[57]:


df_train['Survived'].replace({'Yes':1, 'No':0}, inplace=True)


# In[58]:


print(f'Train shape: {df_train.shape}')
print(f'Train shape: {df_test.shape}')


# In[59]:


df_train.drop(['Age', 'Fare','Fare_log','Family', 'SibSp', 'Parch'], axis=1, inplace=True)
df_test.drop(['Age', 'Fare','Fare_log','Family', 'SibSp', 'Parch'], axis=1, inplace=True)


# ## Setting X and Y

# In[60]:


X_train = df_train.drop(["Survived","PassengerId"],axis=1)
y_train = df_train["Survived"]

X_test = df_test.drop(["PassengerId"],axis=1)


# In[61]:


resumetable(X_train)


# <a id="Model"></a> <br> 
# # **6. Modelling Pipeline of models to find the algo that best fit our problem ** 

# In[62]:


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


# In[63]:


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
                       ("KNN", KNeighborsClassifier())]))) 

clfs.append(("DecisionTreeClassifier", 
             Pipeline([("Scaler", StandardScaler()),
                       ("DecisionTrees", DecisionTreeClassifier())]))) 

clfs.append(("RandomForestClassifier", 
             Pipeline([("Scaler", StandardScaler()),
                       ("RandomForest", RandomForestClassifier(n_estimators=100))]))) 

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


# Very cool! Based on the result of our CVLet's try model LogReg and XGBClassifier to predict who will survive or not

# # HyperOpt - Automated Bayeasian Hyperparameter serach. 

# # HyperOpt with Random Forest

# In[64]:


import time

def objective(params):
    time1 = time.time()
    params = {
        'max_depth': params['max_depth'],
        'max_features': params['max_features'],
        'n_estimators': params['n_estimators'],
        'min_samples_split': params['min_samples_split'],
        'criterion': params['criterion']
    }

    print("\n############## New Run ################")
    print(f"params = {params}")
    FOLDS = 10
    count=1

    skf = StratifiedKFold(n_splits=FOLDS, random_state=42, shuffle=True)

    kf = KFold(n_splits=FOLDS, shuffle=False, random_state=42)

    score_mean = 0
    for tr_idx, val_idx in kf.split(X_train, y_train):
        clf = RandomForestClassifier(
            random_state=4, 
            verbose=0,  n_jobs=-1, 
            **params
        )

        X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        
        clf.fit(X_tr, y_tr)
        #y_pred_train = clf.predict_proba(X_vl)[:,1]
        #print(y_pred_train)
        score = make_scorer(accuracy_score)(clf, X_vl, y_vl)
        # plt.show()
        score_mean += score
        print(f'{count} CV - score: {round(score, 4)}')
        count += 1
    time2 = time.time() - time1
    print(f"Total Time Run: {round(time2 / 60,2)}")
    gc.collect()
    print(f'Mean ROC_AUC: {score_mean / FOLDS}')
    del X_tr, X_vl, y_tr, y_vl, clf, score
    return -(score_mean / FOLDS)

rf_space = {
    'max_depth': hp.choice('max_depth', range(2,8)),
    'max_features': hp.choice('max_features', range(1,X_train.shape[1])),
    'n_estimators': hp.choice('n_estimators', range(100,500)),
    'min_samples_split': hp.choice('min_samples_split', range(5,35)),
    'criterion': hp.choice('criterion', ["gini", "entropy"])
}


# ## Running the HyperOpt to get the best params

# In[65]:


best = fmin(fn=objective,
            space=rf_space,
            algo=tpe.suggest,
            max_evals=40, 
            # trials=trials
           )


# ## Best params 

# In[66]:


best_params = space_eval(rf_space, best)
best_params


# ## Predicting the X_test with Random Forest

# In[67]:


clf = RandomForestClassifier(
        **best_params, random_state=4,
        )

clf.fit(X_train, y_train)

y_preds= clf.predict(X_test)

submission['Survived'] = y_preds.astype(int)
submission.to_csv('Titanic_rf_model_pred.csv')


# _______________________________________________
# # Predicting X_test with Logreg and HyperOpt

# ### Calculating the class_weights

# In[68]:


from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(y_train),
                                                  y_train)


# # Objective LogReg

# In[69]:


def objective_logreg(params):
    time1 = time.time()
    params = {
        'tol': params['tol'],
        'C': params['C'],
        'solver': params['solver'],
    }

    print("\n############## New Run ################")
    print(f"params = {params}")
    FOLDS = 10
    count=1

    skf = StratifiedKFold(n_splits=FOLDS, random_state=42, shuffle=True)

    kf = KFold(n_splits=FOLDS, shuffle=False, random_state=42)

    score_mean = 0
    for tr_idx, val_idx in kf.split(X_train, y_train):
        clf = LogisticRegression(
            random_state=4,  
            **params
        )

        X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        
        clf.fit(X_tr, y_tr)
        score = make_scorer(accuracy_score)(clf, X_vl, y_vl)
        score_mean += score
        print(f'{count} CV - score: {round(score, 4)}')
        count += 1
    time2 = time.time() - time1
    print(f"Total Time Run: {round(time2 / 60,2)}")
    gc.collect()
    print(f'Mean ROC_AUC: {score_mean / FOLDS}')
    del X_tr, X_vl, y_tr, y_vl, clf, score
    return -(score_mean / FOLDS)

space_logreg = {
    'tol' : hp.uniform('tol', 0.00001, 0.001),
    'C' : hp.uniform('C', 0.001, 2),
    'solver' : hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
}


# ## Running LogReg HyperOpt

# In[70]:


best = fmin(fn=objective_logreg,
            space=space_logreg,
            algo=tpe.suggest,
            max_evals=45, 
            # trials=trials
           )


# In[71]:


best_params = space_eval(space_logreg, best)
best_params


# In[72]:


clf = LogisticRegression(
        **best_params, random_state=4,
        )

clf.fit(X_train, y_train)

y_preds= clf.predict(X_test)

submission['Survived'] = y_preds.astype(int)
submission.to_csv('Titanic_logreg_model_pred.csv')



# # Stay tuned and don't forget to votesup this kernel =)
