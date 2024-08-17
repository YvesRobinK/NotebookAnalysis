#!/usr/bin/env python
# coding: utf-8

# ## **0. Introduction**
# 
# ###  "81.1%" score on Kaggle Leaderboard! This Notebook is written keeping in mind the basics.
# 
# Hello Fellow Kagglers,I decided to write this kernel because **Titanic: Machine Learning from Disaster** is one of the Well-Known competition on Kaggle. This is a beginner level kernel which tests your **Exploratory Data Analysis** and **Feature Engineering** skills. Most beginners get lost in the field, because they fall into the black box approach, using libraries and algorithms they don't understand. At first this kernel would look Quite large , but if u stick to it to it , to the very end u will truly learn a lot.
# 
# **Titanic: Machine Learning from Disaster** is a great competition to apply domain knowledge for feature engineering, so I made a research and learned a lot about **Exploratory Data Analysis** and **Feature Engineering** from other kernels available here , that can help Improve the accuracy of the model.
# 
# **If you have any idea that might improve this kernel, please be sure to comment, or fork and experiment as you like.**
# 
# I have researched and learned a lot from other kernels to provide better results. I just want to **`Thank`** the kaggle community for being so generous.    

# ## **1. Define The Problem**
# 
# **Project Summary:**
# The RMS Titanic was a British passenger liner that sank in the North Atlantic Ocean in the early morning hours of 15 April 1912, after it collided with an iceberg during its maiden voyage from Southampton to New York City. There were an estimated 2,224 passengers and crew aboard the ship, and more than 1,500 died, making it one of the deadliest commercial peacetime maritime disasters in modern history.
# 
# I have tried my best to explain every complex code written and even given links in between for Reference.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.
# 
# Practice Skills
# * Binary classification
# * Python
# 
# ![alt text](https://vignette.wikia.nocookie.net/titanic/images/f/f9/Titanic_side_plan.png/revision/latest?cb=20180322183733)
# 
# # **2. Gather the Data**
# 
#  Test and train data at [Kaggle's Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)
# 
# 
# 

# # **3. Import Libraries**
# 
# The following code is written in Python 3.x. Libraries provide pre-written functionality to perform necessary tasks.

# In[1]:


# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization(for EDA)
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib import style

#We will use the popular scikit-learn library to develop our machine learning algorithms

# Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc

# Models
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

import string

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


# # **4. Reading the Data**

# In[2]:


# link --->https://www.geeksforgeeks.org/python-read-csv-using-pandas-read_csv/
df_test = pd.read_csv("../input/titanic/test.csv")
df_train = pd.read_csv("../input/titanic/train.csv")

# link---> w3resource.com/pandas/concat.php
def concat_df(train_data, test_data):
    # Returns a concatenated df of training and test set
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

def divide_df(all_data):
    # Use DataFrame.loc attribute to access a particular cell in the given Dataframe using the index and column labels.
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)
    # Returns divided dfs of training and test set 

df_all = concat_df(df_train, df_test)

df_train.name = 'Training Set'
df_test.name = 'Test Set'
df_all.name = 'All Set' 

dfs = [df_train, df_test]  # List consisting of both Train and Test set

# Pls note:- df_all and dfs is not same (df_all is a Dataframe and dfs is a list)


# In[3]:


# Pandas sample() is used to generate a sample random row or column from the function caller data frame.
df_all.sample(10)


# # **5. Exploratory Data Analysis**
# 
# ** Developer Documentation: **
# * [pandas.DataFrame](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)
# * [pandas.DataFrame.info](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.info.html)
# * [pandas.DataFrame.describe](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html)
# * [Indexing and Selecting Data](https://pandas.pydata.org/pandas-docs/stable/indexing.html)
# * [pandas.isnull](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.isnull.html)
# * [pandas.DataFrame.sum](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sum.html)
# * [pandas.DataFrame.mode](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.mode.html)
# * [pandas.DataFrame.copy](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.copy.html)
# * [pandas.DataFrame.fillna](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html)
# * [pandas.DataFrame.drop](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.drop.html)
# * [pandas.Series.value_counts](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.value_counts.html)
# * [pandas.DataFrame.loc](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.loc.html)

# In[4]:


#preview data
print (df_train.info()) # link ---> https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.info.html


# **The training-set has 891 examples and 11 features + the target variable (survived).** 2 of the features are floats, 5 are integers and 5 are objects. Below I have listed the features with a short description:

# * `PassengerId` is the unique id of the row and it doesn't have any effect on target
# * `Survived` is the target variable we are trying to predict (**0** or **1**):
#     - **1 = Survived**
#     - **0 = Not Survived**
# * `Pclass` (Passenger Class) is the socio-economic status of the passenger and it is a categorical ordinal feature which has **3** unique values (**1**,  **2 **or **3**):
#     - **1 = Upper Class**
#     - **2 = Middle Class**
#     - **3 = Lower Class**
# * `Name`, `Sex` and `Age` are self-explanatory
# * `SibSp` is the total number of the passengers' siblings and spouse
# * `Parch` is the total number of the passengers' parents and children
# * `Ticket` is the ticket number of the passenger
# * `Fare` is the passenger fare
# * `Cabin` is the cabin number of the passenger
# * `Embarked` is port of embarkation and it is a categorical feature which has **3** unique values (**C**, **Q** or **S**):
#     - **C = Cherbourg**
#     - **Q = Queenstown**
#     - **S = Southampton**

# In[5]:


#df_train.head() # link --> https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.head.html
#df_train.tail() # link --> https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.tail.html

df_train.sample(10) # link --> https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sample.html

#If u look at the 'cabin' Feature , u can see 'NAN' depicting missing values. 


# In[6]:


df_test.info()
df_test.sample(10) #https://www.geeksforgeeks.org/python-pandas-dataframe-sample/


# In[7]:


df_train.describe() #link --> https://www.geeksforgeeks.org/python-pandas-dataframe-describe-method/


# ### How many Survived??

# In[8]:


# link --> https://www.geeksforgeeks.org/matplotlib-pyplot-subplots-in-python/
# link --> https://www.geeksforgeeks.org/plot-a-pie-chart-in-python-using-matplotlib/
# link --> https://www.geeksforgeeks.org/countplot-using-seaborn-in-python/

f,ax=plt.subplots(1,2,figsize=(18,8)) # 1 row , 2 columns subplots 
df_train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Survived') 
ax[0].set_ylabel('')

sns.countplot('Survived',data=df_train,ax=ax[1])

ax[1].set_title('Survived') # ax[0] & ax[1] are different axis for different plots.

plt.show()


# Above we can see that 38% out of the training-set survived the Titanic. 
# 
# We can also see that the passenger ages range from 0.4 to 80. 
# 
# On top of that we can already detect some features, that contain missing values, like the ‘Age’ and 'Cabin' feature.

# It is evident that not many passengers survived the accident.
# 
# Out of 891 passengers in training set, only around 350 survived i.e Only 38.4% of the total training set survived the crash. We need to dig down more to get better insights from the data and see which categories of the passengers did survive and who didn't.
# 
# We will try to check the survival rate by using the different features of the dataset. Some of the features being Sex, Port Of Embarcation, Age,etc.
# 
# First let us understand the different types of features.

# ## Types Of Features
# 
# ### Categorical Features:
# A categorical variable is one that has two or more categories and each value in that feature can be categorised by them.For example, gender is a categorical variable having two categories (male and female). Now we cannot sort or give any ordering to such variables. They are also known as **Nominal Variables**.
# 
# **Categorical Features in the dataset: Sex,Embarked.**
# 
# ### Ordinal Features:
# An ordinal variable is similar to categorical values, but the difference between them is that we can have relative ordering or sorting between the values. For eg: If we have a feature like **Height** with values **Tall, Medium, Short**, then Height is a ordinal variable. Here we can have a relative sort in the variable.
# 
# **Ordinal Features in the dataset: PClass**
# 
# ### Continous Feature:
# A feature is said to be continous if it can take values between any two points or between the minimum or maximum values in the features column.
# 
# **Continous Features in the dataset: Age**

# ## **5.1 Missing Values**
# **Let’s take a more detailed look at what data is actually missing:**

# In[9]:


# Counting the total missing values in respective features
total_missing_train = df_train.isnull().sum().sort_values(ascending=False)

# Calculating the percent of missing values in respective features
percent_1 = df_train.isnull().sum()/df_train.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False) # Rounding the percent calculated in percent_1 to one decimal.

#w3resource.com/pandas/concat.php
train_missing_data = pd.concat([total_missing_train, percent_2], axis=1, keys=['Total', '%'])

print(total_missing_train)

print('_'*25)

train_missing_data.head(5) # prints/shows top 5 rows of dataframe


# In[10]:


total_missing_test = df_test.isnull().sum().sort_values(ascending=False)

percent_3 = df_test.isnull().sum()/df_test.isnull().count()*100
percent_4 = (round(percent_3, 1)).sort_values(ascending=False) 

test_missing_data = pd.concat([total_missing_test, percent_4], axis=1, keys=['Total', '%']) #w3resource.com/pandas/concat.php

print(total_missing_test)

print('_'*25)

test_missing_data.head(5)


# 
# As seen from below, some columns have missing values. `df_test.isnull().sum()` function shows the count of missing values in every column in both training and test set.
# * Training set have missing values in `Age`, `Cabin` and `Embarked` columns
# * Test set have missing values in `Age`, `Cabin` and `Fare` columns
# 
# It is convenient to work on concatenated training and test set while dealing with missing values, otherwise filled data may overfit to training or test set samples. The count of missing values in `Age`, `Embarked` and `Fare` are smaller compared to total sample, but roughly **78%** of the `Cabin` is missing. Missing values in `Age`, `Embarked` and `Fare` can be filled with descriptive statistical measures but that wouldn't work for `Cabin`.

# ### **Age and Sex:**

# In[11]:


# link --> https://www.geeksforgeeks.org/matplotlib-pyplot-subplots-in-python/
f,ax=plt.subplots(figsize=(18,8))

# link --> https://seaborn.pydata.org/generated/seaborn.violinplot.html
sns.violinplot("Pclass","Age", hue="Survived", data=df_train,split=True,ax=ax)

ax.set_title('Pclass and Age vs Survived')

ax.set_yticks(range(0,110,10)) # set_yticks() function in axes module is used to Set the y ticks with list of ticks.

plt.show()


# **Observations:**
# 
# 1)The number of children increases with Pclass and the survival rate for passenegers below Age 10(i.e children) looks to be good irrespective of the Pclass.
# 
# 2)Survival chances for Passenegers aged 20-50 from Pclass1 is high.
# 

# In[12]:


# link --> https://www.geeksforgeeks.org/python-pandas-dataframe-corr/
df_all_corr = df_all.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()

df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)

df_all_corr[df_all_corr['Feature 1'] == 'Pclass'] 


# Missing values in Age are filled with median age, but using median age of the whole data set is not a good choice. Median age of `Pclass` groups is the best choice because of its **high correlation** `Age` (0.408106) and `Survived` (0.338481). It is also more logical to group ages by passenger classes instead of other features.

# In order to be more accurate, Sex feature is used as the second level of groupby while filling the missing Age values.
# 
# **Let's see why**

# In[13]:


f,ax=plt.subplots(figsize=(18,8))

# link --> http://alanpryorjr.com/visualizations/seaborn/violinplot/violinplot/
sns.violinplot("Sex","Age", hue="Survived", data=df_train,split=True,ax=ax)

ax.set_title('Sex and Age vs Survived') # setting the title of plot

ax.set_yticks(range(0,110,10))

plt.show()


# You can see that men have a high probability of survival when they are between 18 and 30 years old, which is also a little bit true for women but not fully.
# 
# For women the survival chances are higher between 14 and 40.
# 
# For men the probability of survival is very low between the age of 5 and 18, but that isn’t true for women. 
# 
# Another thing to note is that infants also have a little bit higher probability of survival.
# 
# When passenger class increases, the median age for both males and females also increases. However, females tend to have slightly lower median Age than males. The median ages below are used for filling the missing values in Age feature.

# In[14]:


# link ---> https://www.geeksforgeeks.org/python-pandas-dataframe-groupby/
age_by_pclass_sex = df_all.groupby(['Sex', 'Pclass']).median()['Age']

for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Median age of Pclass {} {}s: {} '.format(pclass, sex, age_by_pclass_sex[sex][pclass].astype(int)))

# Filling the missing values in Age with the medians of Sex and Pclass groups
df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
# link --> https://www.w3schools.com/python/python_lambda.asp


# ### **Embarked**

# **Chances for Survival by Port Of Embarkation --**

# In[15]:


# link --> https://www.geeksforgeeks.org/python-seaborn-factorplot-method/
sns.factorplot('Embarked','Survived',data=df_train)
fig=plt.gcf() # pyplot. gcf() is primarily used to get the current figure. 
fig.set_size_inches(5,3)
plt.show()


# `Embarked` is a categorical feature and there are only **2** missing values in whole data set. Both of those passengers are female, upper class and they have the same ticket number. This means that they know each other and embarked from the same port together. The mode `Embarked` value for an upper class female passenger is **C (Cherbourg)**, but this doesn't necessarily mean that they embarked from that port.

# **Two Missing members ---**

# In[16]:


df_all[df_all['Embarked'].isnull()]


# When I googled **Stone, Mrs. George Nelson (Martha Evelyn)**, I found that she embarked from **S (Southampton)** with her maid **Amelie Icard**, in this page [Martha Evelyn Stone: Titanic Survivor](https://www.encyclopedia-titanica.org/titanic-survivor/martha-evelyn-stone.html).

# In[17]:


# Filling the missing values in Embarked with S
df_all['Embarked'] = df_all['Embarked'].fillna('S')
# link --> https://www.geeksforgeeks.org/python-pandas-dataframe-fillna-to-replace-null-values-in-dataframe/


# In[18]:


# link --> https://www.kaggle.com/residentmario/faceting-with-seaborn
FacetGrid = sns.FacetGrid(df_train, row='Embarked', size=4.5, aspect=1.6)

# link --> https://www.geeksforgeeks.org/python-seaborn-pointplot-method/
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex')

FacetGrid.add_legend() # Draw a legend, maybe placing it outside axes and resizing the figure.


# `Embarked` seems to be correlated with `survival`, depending on the `Sex` and `Pclass`.
# Women on port Q and on port S have a higher chance of survival. The inverse is true, if they are at port C. Men have a high survival probability if they are on port C, but a low probability if they are on port Q or S.

# #### **Fare**
# There is only one passenger with missing `Fare` value. We can assume that `Fare` is related to family size (`Parch` and `SibSp`) and `Pclass` features. Median `Fare` value of a male with a third class ticket and no family is a logical choice to fill the missing value.

# In[19]:


df_all[df_all['Fare'].isnull()]


# In[20]:


med_fare = df_all.groupby(['Pclass', 'Parch', 'SibSp'])['Fare'].median()[3][0][0]
# Median of a Fare satisying condition([3][0][0] -- 3=Pclass,0=Parch,SibSp) 

# Filling the missing value in Fare with the median Fare of 3rd class alone passenger
df_all['Fare'] = df_all['Fare'].fillna(med_fare)


# ### **Pclass**

# In[21]:


# link --> https://www.geeksforgeeks.org/seaborn-barplot-method-in-python/
sns.barplot(x='Pclass', y='Survived',hue='Sex',data=df_train)


# Here we see clearly, that `Pclass` is contributing to a persons chance of survival, especially if this person is in class 1. 
# 
# Looking at the BarPlot , we can easily infer that survival for Women from Pclass1 is about 95-96%, as only 3 out of 94 Women from Pclass1 died.
# 
# It is evident that irrespective of Pclass, Women were given first priority while rescue. Even Men from Pclass1 have a very low survival rate.
# 
# Looks like Pclass is also an important feature. 

# In[22]:


grid = sns.FacetGrid(df_train, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();


# The plot above confirms our assumption about pclass 1, but we can also spot a high probability that a person in pclass 3 will not survive.

# ### **SibSp and Parch:**

# `SibSp` and `Parch` would make more sense as a combined feature, that shows the total `Family Size`, a person has on the Titanic. I will create it below and also a feature that sows if someone is not alone.

# In[23]:


data1=df_train.copy() # shallow copy
data1['Family_size'] = data1['SibSp'] + data1['Parch'] +1
# 1 is considered 'Alone'

data1['Family_size'].value_counts().sort_values(ascending=False)


# In[24]:


axes = sns.factorplot('Family_size','Survived', data=data1, aspect = 2.5, )


# Here we can see that you had a high probabilty of survival with 2 to 4 Family Size, but a lower one if you had less than 2 or more than 4 (except for some cases with 7 ).

# #### **Cabin**
# `Cabin` feature is little bit tricky and it needs further exploration. The large portion of the `Cabin` feature is missing and the feature itself cant be ignored completely because some the cabins might have higher survival rates. It turns out to be the first letter of the `Cabin` 
# values are the decks in which the cabins are located. 
# Those decks were mainly separated for one passenger class, but some of them were used by multiple passenger classes.
# ![alt text](https://vignette.wikia.nocookie.net/titanic/images/f/f9/Titanic_side_plan.png/revision/latest?cb=20180322183733)
# * On the Boat Deck there were **6** rooms labeled as **T, U, W, X, Y, Z** but only the **T** cabin is present in the dataset
# * **A**, **B** and **C** decks were only for 1st class passengers
# * **D** and **E** decks were for all classes
# * **F** and **G** decks were for both 2nd and 3rd class passengers

# In[25]:


# Creating Deck column by extracting the first letter of the Cabin(string s) column M stands for Missing
df_all['Deck'] = df_all['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')


df_all_decks = df_all.groupby(['Deck', 'Pclass']).count().drop(columns=['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 
                                                                        'Fare', 'Embarked', 'Cabin', 'PassengerId', 
                                                                        'Ticket']).rename(columns={'Name': 'Count'})

df_all_decks


# **Line 1:** s is Cabin name and **s[0]**  gives the Cabin alphabet like 'C' , if S[0] is missing then it goes to category 'M'
# 
# **Line 2:** Various columns are dropped from df_all , 'deck' is grouped with 'pclass' and the 'Name' column is renamed to 'Count'

# In[26]:


# Transpose is done for accessbility
df_all_decks=df_all_decks.transpose()


# In[27]:


def get_pclass_dist(df):
    
    # Creating a dictionary for every passenger class count in every deck
    deck_counts = {'A': {}, 'B': {}, 'C': {}, 'D': {}, 'E': {}, 'F': {}, 'G': {}, 'M': {}, 'T': {}}
    
    #Deck column is extracted from df_all_decks 
    decks = df.columns.levels[0]    
    
    # Creating a new dataframe just a copy of df_all_decks with 0 in respective Pclass if empty ... See Output below.
    # Start
    for deck in decks:
        for pclass in range(1, 4):
            try:
                count = df[deck][pclass][0]
                deck_counts[deck][pclass] = count 
            except KeyError:
                deck_counts[deck][pclass] = 0
                
    df_decks = pd.DataFrame(deck_counts) 
    # End
    
    deck_percentages = {}
   
    # Creating a dictionary for every passenger class percentage in every deck
    for col in df_decks.columns:
        deck_percentages[col] = [(count / df_decks[col].sum()) * 100 for count in df_decks[col]]
        
    return deck_counts, deck_percentages,df_decks


all_deck_count, all_deck_per,df_decks_return = get_pclass_dist(df_all_decks)

print(df_decks_return)

print("_"*25)

all_deck_per


# In[28]:


def display_pclass_dist(percentages):
    
    #converting dictionary to dataframe and then transpose
    df_percentages = pd.DataFrame(percentages).transpose()
    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M', 'T')
    bar_count = np.arange(len(deck_names))  
    bar_width = 0.85
    
    pclass1 = df_percentages[0]
    pclass2 = df_percentages[1]
    pclass3 = df_percentages[2]
    
    plt.figure(figsize=(20, 10))
    
    # link --> https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm
    plt.bar(bar_count, pclass1,width=bar_width,edgecolor='white',label='Passenger Class 1')
    plt.bar(bar_count, pclass2, bottom=pclass1, color='#f9bc86', edgecolor='white', width=bar_width, label='Passenger Class 2')
    plt.bar(bar_count, pclass3, bottom=pclass1 + pclass2, color='#a3acff', edgecolor='white', width=bar_width, label='Passenger Class 3')

    plt.xlabel('Deck', size=15, labelpad=20)
    plt.ylabel('Passenger Class Percentage', size=15, labelpad=20)
    plt.xticks(bar_count, deck_names)    
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    
    plt.legend(loc='best',bbox_to_anchor=(1, 1),prop={'size': 15})
    plt.title('Passenger Class Distribution in Decks',size=18, y=1.05)   
    
    plt.show()    
    
display_pclass_dist(all_deck_per)    


# * **100%** of **A**, **B** and **C** decks are 1st class passengers
# * Deck **D** has **87%** 1st class and **13%** 2nd class passengers
# * Deck **E** has **83%** 1st class, **10%** 2nd class and **7%** 3rd class passengers
# * Deck **F** has **62%** 2nd class and **38%** 3rd class passengers
# * **100%** of **G** deck are 3rd class passengers
# * There is one person on the boat deck in **T** cabin and he is a 1st class passenger. **T** cabin passenger has the closest resemblance to **A** deck passengers so he is grouped with **A** deck
# * Passengers labeled as **M** are the missing values in `Cabin` feature. I don't think it is possible to find those passengers' real `Deck` so I decided to use **M** like a deck

# In[29]:


# Passenger in the T deck is changed to A
idx = df_all[df_all['Deck'] == 'T'].index
df_all.loc[idx, 'Deck'] = 'A'


# In[30]:


# Same Method is applied as above just this time , deck is grouped with 'Survived' Feature

df_all_decks_survived = df_all.groupby(['Deck', 'Survived']).count().drop(columns=['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                                                                                   'Embarked', 'Pclass', 'Cabin', 'PassengerId', 'Ticket']).rename(columns={'Name':'Count'}).transpose()

def get_survived_dist(df):
    
    # Creating a dictionary for every survival count in every deck
    surv_counts = {'A':{}, 'B':{}, 'C':{}, 'D':{}, 'E':{}, 'F':{}, 'G':{}, 'M':{}}
    decks = df.columns.levels[0]    

    for deck in decks:
        for survive in range(0, 2):
            surv_counts[deck][survive] = df[deck][survive][0]
            
    df_surv = pd.DataFrame(surv_counts)
    surv_percentages = {}

    for col in df_surv.columns:
        surv_percentages[col] = [(count / df_surv[col].sum()) * 100 for count in df_surv[col]]
        
    return surv_counts, surv_percentages

def display_surv_dist(percentages):
    
    df_survived_percentages = pd.DataFrame(percentages).transpose()
    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M')
    bar_count = np.arange(len(deck_names))  
    bar_width = 0.85    

    not_survived = df_survived_percentages[0]
    survived = df_survived_percentages[1]
    
    plt.figure(figsize=(20, 10))
    plt.bar(bar_count, not_survived, color='#b5ffb9', edgecolor='white', width=bar_width, label="Not Survived")
    plt.bar(bar_count, survived, bottom=not_survived, color='#f9bc86', edgecolor='white', width=bar_width, label="Survived")
 
    plt.xlabel('Deck', size=15, labelpad=20)
    plt.ylabel('Survival Percentage', size=15, labelpad=20)
    plt.xticks(bar_count, deck_names)    
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 15})
    plt.title('Survival Percentage in Decks', size=18, y=1.05)
    
    plt.show()

all_surv_count, all_surv_per = get_survived_dist(df_all_decks_survived)
display_surv_dist(all_surv_per)


# As I suspected, every deck has different survival rates and that information can't be discarded. Deck **B**, **C**, **D** and **E** have the highest survival rates. Those decks are mostly occupied by 1st class passengers. **M** has the lowest survival rate which is mostly occupied by 2nd and 3rd class passengers. To conclude, cabins used by 1st class passengers have higher survival rates than cabins used by 2nd and 3rd class passengers. In my opinion **M** (Missing `Cabin` values) has the lowest survival rate because they couldn't retrieve the cabin data of the victims. That's why I believe labeling that group as **M** is a reasonable way to handle the missing data. It is a unique group with shared characteristics. `Deck` feature has high-cardinality right now so some of the values are grouped with each other based on their similarities.
# * **A**, **B** and **C** decks are labeled as **ABC** because all of them have only 1st class passengers
# * **D** and **E** decks are labeled as **DE** because both of them have similar passenger class distribution and same survival rate
# * **F** and **G** decks are labeled as **FG** because of the same reason above
# * **M** deck doesn't need to be grouped with other decks because it is very different from others and has the lowest survival rate.

# In[31]:


df_all['Deck'] = df_all['Deck'].replace(['A', 'B', 'C'], 'ABC')
df_all['Deck'] = df_all['Deck'].replace(['D', 'E'], 'DE')
df_all['Deck'] = df_all['Deck'].replace(['F', 'G'], 'FG')

df_all['Deck'].value_counts()


# After filling the missing values in `Age`, `Embarked`, `Fare` and `Deck` features, there is no missing value left in both training and test set. `Cabin` is dropped because `Deck` feature is used instead of it.

# In[32]:


# Dropping the Cabin feature
df_all.drop(['Cabin'], inplace=True, axis=1)

df_train, df_test = divide_df(df_all)
dfs = [df_train, df_test]

for df in dfs:
    print(df_test.isnull().sum())
    print('-'*25)


# ### **Continuous Features**
# Both of the continuous features (`Age` and `Fare`) have good split points and spikes for a decision tree to learn. One potential problem for both features is, the distribution has more spikes and bumps in training set, but it is smoother in test set. Model may not be able to generalize to test set because of this reason.
# 
# * Distribution of `Age` feature clearly shows that children younger than 15 has a higher survival rate than any of the other age groups
# * In distribution of `Fare` feature, the survival rate is higher on distribution tails. The distribution also has positive skew because of the extremely large outliers

# In[33]:


cont_features = ['Age', 'Fare']
surv = df_train['Survived'] == 1

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 20))
plt.subplots_adjust(right=1.5)

for i, feature in enumerate(cont_features): # link --> https://www.geeksforgeeks.org/enumerate-in-python/   
    # Distribution of survival in feature
    sns.distplot(df_train[~surv][feature], label='Not Survived', hist=True, color='#e74c3c', ax=axs[0][i]) 
    # [-surv] means "Not Survived"
    sns.distplot(df_train[surv][feature], label='Survived', hist=True, color='#2ecc71', ax=axs[0][i])
    
    # Distribution of feature in dataset
    sns.distplot(df_train[feature], label='Training Set', hist=False, color='#e74c3c', ax=axs[1][i])
    sns.distplot(df_test[feature], label='Test Set', hist=False, color='#2ecc71', ax=axs[1][i])
    
    axs[0][i].set_xlabel('')
    axs[1][i].set_xlabel('')
     
    # just providing the ticks for x & y axis in respective plots    
    for j in range(2):        
        axs[i][j].tick_params(axis='x', labelsize=20)
        axs[i][j].tick_params(axis='y', labelsize=20)
    
    axs[0][i].legend(loc='upper right', prop={'size': 20})
    axs[1][i].legend(loc='upper right', prop={'size': 20})
    axs[0][i].set_title('Distribution of Survival in {}'.format(feature), size=20, y=1.05)

axs[1][0].set_title('Distribution of {} Feature'.format('Age'), size=20, y=1.05)
axs[1][1].set_title('Distribution of {} Feature'.format('Fare'), size=20, y=1.05)
        
plt.show()


# #### **Categorical Features**
# Every categorical feature has at least one class with high mortality rate. Those classes are very helpful to predict whether the passenger is a survivor or victim. Best categorical features are `Pclass` and `Sex` because they have the most homogenous distributions.
# 
# * Passengers boarded from **Southampton** has a lower survival rate unlike other ports. More than half of the passengers boarded from **Cherbourg** had survived. This observation could be related to `Pclass` feature
# * `Parch` and `SibSp` features show that passengers with only one family member has a higher survival rate

# In[34]:


cat_features = ['Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp', 'Deck']

fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(20, 20))
plt.subplots_adjust(right=1.5, top=1.25)

for i, feature in enumerate(cat_features, 1):    
    plt.subplot(2, 3, i)
    sns.countplot(x=feature, hue='Survived', data=df_train)
    
    plt.xlabel('{}'.format(feature), size=20, labelpad=15)
    plt.ylabel('Passenger Count', size=20, labelpad=15)    
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    
    plt.legend(['Not Survived', 'Survived'], loc='upper center', prop={'size': 18})
    plt.title('Count of Survival in {} Feature'.format(feature), size=20, y=1.05)

plt.show()


# ### **Conclusion(EDA)**
# Most of the features are correlated with each other. This relationship can be used to create new features with feature transformation and feature interaction. Target encoding could be very useful as well because of the high correlations with `Survived` feature.
# 
# Split points and spikes are visible in continuous features. They can be captured easily with a decision tree model, but linear models may not be able to spot them.
# 
# Categorical features have very distinct distributions with different survival rates. Those features can be one-hot encoded. Some of those features may be combined with each other to make new features.
# 
# Created a new feature called `Deck` and dropped `Cabin` feature at the **Exploratory Data Analysis** part.

# In[35]:


df_all = concat_df(df_train, df_test)
df_all.head()


# # Correlation Between The Features

# In[36]:


# link ---> https://likegeeks.com/seaborn-heatmap-tutorial/
sns.heatmap(df_all.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# ### Interpreting The Heatmap
# 
# The first thing to note is that only the numeric features are compared as it is obvious that we cannot correlate between alphabets or strings. Before understanding the plot, let us see what exactly correlation is.
# 
# **POSITIVE CORRELATION:** If an **increase in feature A leads to increase in feature B, then they are positively correlated**. A value **1 means perfect positive correlation**.
# 
# **NEGATIVE CORRELATION:** If an **increase in feature A leads to decrease in feature B, then they are negatively correlated**. A value **-1 means perfect negative correlation**.
# 
# Now lets say that two features are highly or perfectly correlated, so the increase in one leads to increase in the other. This means that both the features are containing highly similar information and there is very little or no variance in information. This is known as **MultiColinearity** as both of them contains almost the same information.
# 
# So do you think we should use both of them as **one of them is redundant**. While making or training models, we should try to eliminate redundant features as it reduces training time and many such advantages.
# 
# Now from the above heatmap,we can see that the features are not much correlated. The highest correlation is between **SibSp and Parch i.e 0.37**. So we can carry on with all features.

# # **6. Feature Engineering**

#  **Links**
# * [Binning Continuous Features](https://www.geeksforgeeks.org/python-binning-method-for-data-smoothing/)
# * [How to use pandas cut() and qcut() for Binnning?](https://www.geeksforgeeks.org/how-to-use-pandas-cut-and-qcut/)

# #### **Fare**
# `Fare` feature is positively skewed and survival rate is extremely high on the right end. **13** quantile based bins are used for `Fare` feature. Even though the bins are too much, they provide decent amount of information gain. The groups at the left side of the graph has the lowest survival rate and the groups at the right side of the graph has the highest survival rate. This high survival rate was not visible in the distribution graph. There is also an unusual group **(15.742, 23.25]** in the middle with high survival rate that is captured in this process.

# In[37]:


df_all['Fare'] = pd.qcut(df_all['Fare'], 13) # visit the link above


# In[38]:


fig, axs = plt.subplots(figsize=(22, 9))
sns.countplot(x='Fare', hue='Survived', data=df_all)

plt.xlabel('Fare', size=15, labelpad=20)
plt.ylabel('Passenger Count', size=15, labelpad=20)
plt.tick_params(axis='x', labelsize=10)
plt.tick_params(axis='y', labelsize=15)

plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})
plt.title('Count of Survival in {} Feature'.format('Fare'), size=15, y=1.05)

plt.show()


# ### **Age**
# `Age` feature has a normal distribution with some spikes and bumps and **10** quantile based bins are used for `Age`. The first bin has the highest survival rate and 4th bin has the lowest survival rate. Those were the biggest spikes in the distribution. There is also an unusual group **(34.0, 40.0]** with high survival rate that is captured in this process.

# In[39]:


df_all['Age'] = pd.qcut(df_all['Age'], 10)


# In[40]:


fig, axs = plt.subplots(figsize=(22, 9))
sns.countplot(x='Age', hue='Survived', data=df_all)

plt.xlabel('Age', size=15, labelpad=20)
plt.ylabel('Passenger Count', size=15, labelpad=20)
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})
plt.title('Survival Counts in {} Feature'.format('Age'), size=15, y=1.05)

plt.show()


# ### **Frequency Encoding**
# `Family_Size` is created by adding `SibSp`, `Parch` and **1**. `SibSp` is the count of siblings and spouse, and `Parch` is the count of parents and children. Those columns are added in order to find the total size of families. Adding **1** at the end, is the current passenger. Graphs have clearly shown that family size is a predictor of survival because different values have different survival rates.
# * Family Size with **1** are labeled as **Alone**
# * Family Size with **2**, **3** and **4** are labeled as **Small**
# * Family Size with **5** and **6** are labeled as **Medium**
# * Family Size with **7**, **8** and **11** are labeled as **Large**

# In[41]:


df_all['Family_Size'] = df_all['SibSp'] + df_all['Parch'] + 1

fig, axs = plt.subplots(figsize=(20, 20), ncols=2, nrows=2)
plt.subplots_adjust(right=1.5)

sns.barplot(x=df_all['Family_Size'].value_counts().index, y=df_all['Family_Size'].value_counts().values, ax=axs[0][0])
sns.countplot(x='Family_Size', hue='Survived', data=df_all, ax=axs[0][1])

axs[0][0].set_title('Family Size Feature Value Counts', size=20, y=1.05)
axs[0][1].set_title('Survival Counts in Family Size ', size=20, y=1.05)

# Mapping Family Size
family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
df_all['Family_Size_Grouped'] = df_all['Family_Size'].map(family_map)

sns.barplot(x=df_all['Family_Size_Grouped'].value_counts().index, y=df_all['Family_Size_Grouped'].value_counts().values, ax=axs[1][0])
sns.countplot(x='Family_Size_Grouped', hue='Survived', data=df_all, ax=axs[1][1])

axs[1][0].set_title('Family Size Feature Value Counts After Grouping', size=20, y=1.05)
axs[1][1].set_title('Survival Counts in Family Size After Grouping', size=20, y=1.05)


for i in range(2):
    axs[i][1].legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 20})
    for j in range(2):
        axs[i][j].tick_params(axis='x', labelsize=20)
        axs[i][j].tick_params(axis='y', labelsize=20)
        axs[i][j].set_xlabel('')
        axs[i][j].set_ylabel('')

plt.show()


# ### **Ticket**

# There are too many unique `Ticket` values to analyze, so grouping them up by their frequencies makes things easier.
# 
# **How is this feature different than `Family_Size`?** Many passengers travelled along with groups. Those groups consist of friends, nannies, maids and etc. They weren't counted as family, but they used the same ticket.
# 
# **Why not grouping tickets by their prefixes?** If prefixes in `Ticket` feature has any meaning, then they are already captured in `Pclass` or `Embarked` features because that could be the only logical information which can be derived from the `Ticket` feature.
# 
# According to the graph below, groups with **2**,**3** and **4** members had a higher survival rate. Passengers who travel alone has the lowest survival rate. After **4** group members, survival rate decreases drastically. This pattern is very similar to `Family_Size` feature but there are minor differences. `Ticket_Frequency` values are not grouped like `Family_Size` because that would basically create the same feature with perfect correlation. This kind of feature wouldn't provide any additional information gain.

# In[42]:


df_all['Ticket_Frequency'] = df_all.groupby('Ticket')['Ticket'].transform('count')


# In[43]:


fig, axs = plt.subplots(figsize=(12, 9))
sns.countplot(x='Ticket_Frequency', hue='Survived', data=df_all)

plt.xlabel('Ticket Frequency', size=15, labelpad=20)
plt.ylabel('Passenger Count', size=15, labelpad=20)
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})
plt.title('Count of Survival in {} Feature'.format('Ticket Frequency'), size=15, y=1.05)

plt.show()


# ### **Title & IsMarried**
# `Title` is created by extracting the prefix before `Name` feature. According to graph below, there are many titles that are occuring very few times. Some of those titles doesn't seem correct and they need to be replaced. **Miss**, **Mrs**, **Ms**, **Mlle**, **Lady**, **Mme**, **the Countess**, **Dona** titles are replaced with **Miss/Mrs/Ms** because all of them are female. Values like **Mlle**, **Mme** and **Dona** are actually the name of the passengers, but they are classified as titles because `Name` feature is split by comma. **Dr**, **Col**, **Major**, **Jonkheer**, **Capt**, **Sir**, **Don** and **Rev** titles are replaced with **Dr/Military/Noble/Clergy** because those passengers have similar characteristics. **Master** is a unique title. It is given to male passengers below age **26**. They have the highest survival rate among all males.
# 
# `Is_Married` is a binary feature based on the **Mrs** title. **Mrs** title has the highest survival rate among other female titles. This title needs to be a feature because all female titles are grouped with each other.
# 

# In[44]:


df_all['Title'] = df_all['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
# https://www.w3schools.com/python/ref_string_split.asp

df_all['Is_Married'] = 0
df_all['Is_Married'].loc[df_all['Title'] == 'Mrs'] = 1


# In[45]:


fig, axs = plt.subplots(nrows=2, figsize=(20, 20))
sns.barplot(x=df_all['Title'].value_counts().index, y=df_all['Title'].value_counts().values, ax=axs[0])

axs[0].tick_params(axis='x', labelsize=10)
axs[1].tick_params(axis='x', labelsize=15)

for i in range(2):    
    axs[i].tick_params(axis='y', labelsize=15)

axs[0].set_title('Title Feature Value Counts', size=20, y=1.05)

df_all['Title'] = df_all['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
df_all['Title'] = df_all['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')

sns.barplot(x=df_all['Title'].value_counts().index, y=df_all['Title'].value_counts().values, ax=axs[1])
axs[1].set_title('Title Feature Value Counts After Grouping', size=20, y=1.05)

plt.show()


# In[46]:


df_train,df_test= divide_df(df_all)
dfs=[df_train,df_test]


# ### **Target Encoding**
# `extract_surname` function is used for extracting surnames of passengers from the `Name` feature. `Family` feature is created with the extracted surname. This is necessary for grouping passengers in the same family.

# In[47]:


df_all['Name'].sample(10)


# In[48]:


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

df_all['Family'] = extract_surname(df_all['Name'])
df_train = df_all.loc[:890]
df_test = df_all.loc[891:]
dfs = [df_train, df_test]


# `Family_Survival_Rate` is calculated from families in training set since there is no `Survived` feature in test set. A list of family names that are occuring in both training and test set (`non_unique_families`), is created. The survival rate is calculated for families with more than 1 members in that list, and stored in `Family_Survival_Rate` feature.
# 
# An extra binary feature `Family_Survival_Rate_NA` is created for families that are unique to the test set. This feature is also necessary because there is no way to calculate those families' survival rate. This feature implies that family survival rate is not applicable to those passengers because there is no way to retrieve their survival rate.
# 
# `Ticket_Survival_Rate` and `Ticket_Survival_Rate_NA` features are also created with the same method. `Ticket_Survival_Rate` and `Family_Survival_Rate` are averaged and become `Survival_Rate`, and `Ticket_Survival_Rate_NA` and `Family_Survival_Rate_NA` are also averaged and become `Survival_Rate_NA`.

# In[49]:


# Creating a list of families and tickets that are occuring in both training and test set
non_unique_families = [x for x in df_train['Family'].unique() if x in df_test['Family'].unique()]
non_unique_tickets = [x for x in df_train['Ticket'].unique() if x in df_test['Ticket'].unique()]

df_family_survival_rate = df_train.groupby('Family')['Survived', 'Family','Family_Size'].median()
df_ticket_survival_rate = df_train.groupby('Ticket')['Survived', 'Ticket','Ticket_Frequency'].median()

family_rates = {}
ticket_rates = {}

for i in range(len(df_family_survival_rate)):
    # Checking a family exists in both training and test set, and has members more than 1
    if df_family_survival_rate.index[i] in non_unique_families and df_family_survival_rate.iloc[i, 1] > 1:
        family_rates[df_family_survival_rate.index[i]] = df_family_survival_rate.iloc[i, 0]

for i in range(len(df_ticket_survival_rate)):
    # Checking a ticket exists in both training and test set, and has members more than 1
    if df_ticket_survival_rate.index[i] in non_unique_tickets and df_ticket_survival_rate.iloc[i, 1] > 1:
        ticket_rates[df_ticket_survival_rate.index[i]] = df_ticket_survival_rate.iloc[i, 0]


# In[50]:


mean_survival_rate = np.mean(df_train['Survived'])

train_family_survival_rate = []
train_family_survival_rate_NA = []
test_family_survival_rate = []
test_family_survival_rate_NA = []

for i in range(len(df_train)):
    if df_train['Family'][i] in family_rates:
        train_family_survival_rate.append(family_rates[df_train['Family'][i]])
        train_family_survival_rate_NA.append(1)
    else:
        train_family_survival_rate.append(mean_survival_rate)
        train_family_survival_rate_NA.append(0)
        
for i in range(len(df_test)):
    if df_test['Family'].iloc[i] in family_rates:
        test_family_survival_rate.append(family_rates[df_test['Family'].iloc[i]])
        test_family_survival_rate_NA.append(1)
    else:
        test_family_survival_rate.append(mean_survival_rate)
        test_family_survival_rate_NA.append(0)
        
df_train['Family_Survival_Rate'] = train_family_survival_rate
df_train['Family_Survival_Rate_NA'] = train_family_survival_rate_NA
df_test['Family_Survival_Rate'] = test_family_survival_rate
df_test['Family_Survival_Rate_NA'] = test_family_survival_rate_NA

train_ticket_survival_rate = []
train_ticket_survival_rate_NA = []
test_ticket_survival_rate = []
test_ticket_survival_rate_NA = []

for i in range(len(df_train)):
    if df_train['Ticket'][i] in ticket_rates:
        train_ticket_survival_rate.append(ticket_rates[df_train['Ticket'][i]])
        train_ticket_survival_rate_NA.append(1)
    else:
        train_ticket_survival_rate.append(mean_survival_rate)
        train_ticket_survival_rate_NA.append(0)
        
for i in range(len(df_test)):
    if df_test['Ticket'].iloc[i] in ticket_rates:
        test_ticket_survival_rate.append(ticket_rates[df_test['Ticket'].iloc[i]])
        test_ticket_survival_rate_NA.append(1)
    else:
        test_ticket_survival_rate.append(mean_survival_rate)
        test_ticket_survival_rate_NA.append(0)
        
df_train['Ticket_Survival_Rate'] = train_ticket_survival_rate
df_train['Ticket_Survival_Rate_NA'] = train_ticket_survival_rate_NA
df_test['Ticket_Survival_Rate'] = test_ticket_survival_rate
df_test['Ticket_Survival_Rate_NA'] = test_ticket_survival_rate_NA


# In[51]:


for df in [df_train, df_test]:
    df['Survival_Rate'] = (df['Ticket_Survival_Rate'] + df['Family_Survival_Rate']) / 2
    df['Survival_Rate_NA'] = (df['Ticket_Survival_Rate_NA'] + df['Family_Survival_Rate_NA']) / 2    


# ## **Feature Transformation**

# ### Convert Formats
# 
# We will convert categorical data to dummy variables for mathematical analysis. There are multiple ways to encode categorical variables; we will use the sklearn and pandas functions.
# 
# In this step, we will also define our x (independent/features/explanatory/predictor/etc.) and y (dependent/target/outcome/response/etc.) variables for data modeling.
# 
# ** Developer Documentation: **
# * [Categorical Encoding](http://pbpython.com/categorical-encoding.html)
# * [Sklearn LabelEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
# * [Sklearn OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
# * [Label Encoder vs OneHot Encoder](https://www.analyticsvidhya.com/blog/2020/03/one-hot-encoding-vs-label-encoding-using-scikit-learn/)
# * [Pandas Categorical dtype](https://pandas.pydata.org/pandas-docs/stable/categorical.html)
# * [pandas.get_dummies](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html)

# #### **Label Encoding Non-Numerical Features**
# `Embarked`, `Sex`, `Deck` , `Title` and `Family_Size_Grouped` are object type, and `Age` and `Fare` features are category type. They are converted to numerical type with `LabelEncoder`. `LabelEncoder` basically labels the classes from **0** to **n**. This process is necessary for models to learn from those features.

# In[52]:


non_numeric_features = ['Embarked', 'Sex', 'Deck', 'Title', 'Family_Size_Grouped', 'Age', 'Fare']

for df in dfs:
    for feature in non_numeric_features:        
        df[feature] = LabelEncoder().fit_transform(df[feature])


# #### **One-Hot Encoding the Categorical Features**
# The categorical features (`Pclass`, `Sex`, `Deck`, `Embarked`, `Title`) are converted to one-hot encoded features with `OneHotEncoder`. `Age` and `Fare` features are not converted because they are ordinal unlike the previous ones.

# In[53]:


onehot_features = ['Pclass', 'Sex', 'Deck', 'Embarked', 'Title', 'Family_Size_Grouped']
encoded_features = []

for df in dfs:
    for feature in onehot_features:
        encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()
        n = df[feature].nunique()
        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
        encoded_df = pd.DataFrame(encoded_feat, columns=cols)
        encoded_df.index = df.index
        encoded_features.append(encoded_df)

# *encoded_features gives all encoded features of each of Six onehot_features         
df_train = pd.concat([df_train, *encoded_features[:6]], axis=1)
df_test = pd.concat([df_test, *encoded_features[6:]], axis=1)


# ### **Conclusion(F.E.)**
# `Age` and `Fare` features are binned. Binning helped dealing with outliers and it revealed some homogeneous groups in those features. `Family_Size` is created by adding `Parch` and `SibSp` features and **1**. `Ticket_Frequency` is created by counting the occurence of `Ticket` values.
# 
# `Name` feature is very useful. First, `Title` and `Is_Married` features are created from the title prefix in the names. Second, `Family_Survival_Rate` and `Family_Survival_Rate_NA`  features are created by target encoding the surname of the passengers. `Ticket_Survival_Rate` is created by target encoding the `Ticket` feature. `Survival_Rate` feature is created by averaging the `Family_Survival_Rate` and `Ticket_Survival_Rate` features.
# 
# Finally, the non-numeric type features are label encoded and categorical features are one-hot encoded. Created **5** new features (`Family_Size`, `Title`, `Is_Married`, `Survival_Rate` and `Survival_Rate_NA`) and dropped the useless features after encoding.

# In[54]:


df_all = concat_df(df_train, df_test)

# Dropping Un-needed feature
drop_cols = ['Deck', 'Embarked', 'Family', 'Family_Size', 'Family_Size_Grouped', 'Survived',
             'Name', 'Parch', 'PassengerId', 'Pclass', 'Sex', 'SibSp', 'Ticket', 'Title',
            'Ticket_Survival_Rate', 'Family_Survival_Rate', 'Ticket_Survival_Rate_NA', 'Family_Survival_Rate_NA']

df_all.drop(columns=drop_cols, inplace=True)
df_all.head()


# # **7.Building Machine Learning Models**

# Now we will train several Machine Learning models and compare their results. Note that because the dataset does not provide labels for their testing-set, we need to use the predictions on the training set to compare the algorithms with each other. Later on, we will use cross validation.

# In[55]:


X = df_train.drop(columns=drop_cols)


# In[56]:


X_train = StandardScaler().fit_transform(X)
Y_train = df_train['Survived'].values
X_test = StandardScaler().fit_transform(df_test.drop(columns=drop_cols))

print('X_train shape: {}'.format(X_train.shape))
print('Y_train shape: {}'.format(Y_train.shape))
print('X_test shape: {}'.format(X_test.shape))


# ### Stochastic Gradient Descent (SGD):

# In[57]:


sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)

sgd.score(X_train, Y_train)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)


# ### Random Forest:

# In[58]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)


# ### Logistic Regression:

# In[59]:


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)


# ### K Nearest Neighbor:

# In[60]:


# KNN 
knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X_train, Y_train)  
Y_pred = knn.predict(X_test)  
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)


# ### Gaussian Naive Bayes:

# In[61]:


gaussian = GaussianNB() 
gaussian.fit(X_train, Y_train)  
Y_pred = gaussian.predict(X_test)  
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)


# ### Perceptron:

# In[62]:


perceptron = Perceptron(max_iter=5)
perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)


# ### Linear Support Vector Machine:

# In[63]:


linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)


# ### Decision Tree

# In[64]:


decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X_train, Y_train)  
Y_pred = decision_tree.predict(X_test)  
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)


# ### Which is the best Model ?

# In[65]:


results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 
              'Decision Tree'],
    'Score': [acc_linear_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)


# As we can see, the Random Forest classifier goes on the first place. But first, let us check, how random-forest performs, when we use cross validation.

# ### **K-Fold Cross Validation:**
# 
# **[Introduction to K-Fold Cross Validation](https://medium.com/datadriveninvestor/k-fold-cross-validation-6b8518070833)**
# 
# K-Fold Cross Validation randomly splits the training data into **K subsets called folds**. Let’s image we would split our data into 4 folds (K = 4). Our random forest model would be trained and evaluated 4 times, using a different fold for evaluation everytime, while it would be trained on the remaining 3 folds.
# The image below shows the process, using 4 folds (K = 4). Every row represents one training + evaluation process. In the first row, the model get’s trained on the first, second and third subset and evaluated on the fourth. In the second row, the model get’s trained on the second, third and fourth subset and evaluated on the first. K-Fold Cross Validation repeats this process till every fold acted once as an evaluation fold.
# 
# ![alt text](https://miro.medium.com/max/875/1*HzpaubLj_o-zt1klnB81Yg.png)
# 
# The result of our K-Fold Cross Validation example would be an array that contains 4 different scores. We then need to compute the mean and the standard deviation for these scores.
# The code below perform K-Fold Cross Validation on our random forest model, using 10 folds (K = 10). Therefore it outputs an array with 10 different scores.

# In[66]:


# Link ---> ttps://stackoverflow.com/questions/25006369/what-is-sklearn-cross-validation-cross-val-score
from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100,oob_score=True)
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# In[67]:


rf.fit(X_train, Y_train)
Y_prediction = rf.predict(X_test)

rf.score(X_train, Y_train)

acc_random_forest = round(rf.score(X_train, Y_train) * 100, 2)
print(round(acc_random_forest,2,), "%")


# This looks much more realistic than before. Our model has a average accuracy of **84%** with a standard deviation of **4 %**. The standard deviation shows us, how precise the estimates are .
# This means in our case that the accuracy of our model can differ + — 4%.
# I think the accuracy is still really good and since random forest is an easy to use model, we will try to increase it’s performance even further in the following section.

# # **Random Forest**
# 
# 
# 
# * **[What is Random Forest ?](https://builtin.com/data-science/random-forest-algorithm)**
# * **[What is Feature Importance ?](https://towardsdatascience.com/explaining-feature-importance-by-example-of-a-random-forest-d9166011959e)**

# Random forest builds multiple decision trees and merges them together to get a more accurate and stable prediction.
# One big advantage of random forest is, that it can be used for both classification and regression problems, which form the majority of current machine learning systems. With a few exceptions a random-forest classifier has all the hyperparameters of a decision-tree classifier and also all the hyperparameters of a bagging classifier, to control the ensemble itself.

# ### Feature Importance
# 
# Another great quality of random forest is that they make it very easy to measure the relative importance of each feature using random_forest.feature_importances_ function. 

# In[68]:


importances = pd.DataFrame({'feature':X.columns,'importance':np.round(rf.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(26)


# In[69]:


importances.plot.bar()


# ### Training random forest again:

# In[70]:


random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(round(acc_random_forest,2,), "%")


# Our random forest model predicts as good as it did before. A general rule is that, the more features you have, **the more likely your model will suffer from overfitting** and vice versa. But I think our data looks fine for now and hasn't too much features.
# There is also another way to evaluate a random-forest classifier, which is probably much more accurate than the score we used before. What I am talking about is the **out-of-bag samples** to estimate the generalization accuracy. I will not go into details here about how it works. Just note that out-of-bag estimate is as accurate as using a test set of the same size as the training set. Therefore, using the out-of-bag error estimate removes the need for a set aside test set.

# In[71]:


print("oob score:", round(random_forest.oob_score_, 4)*100, "%")


# In[72]:


print("oob score:", round(rf.oob_score_, 4)*100, "%")


# Now we can start tuning the hyperameters of random forest.

# # Hyperparameter Tuning

# * **[What are Hyperparameters ?](https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/)**
# * **[Ml-hyperparameter-tuning](https://www.geeksforgeeks.org/ml-hyperparameter-tuning/)**
# 
# Below you can see the code of the hyperparamter tuning for the parameters criterion, min_samples_leaf, min_samples_split and n_estimators.
# I put this code into a markdown cell and not into a code cell, because it takes a long time to run it.

# param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10,], "min_samples_split" : [2, 4, 10,], "n_estimators": [100,500,11000,1500]}
# 
# 
# gd=GridSearchCV(estimator=RandomForestClassifier(random_state=42),param_grid=param_grid,verbose=True)
# 
# 
# gd.fit(X,Y)
# 
# 
# print(gd.best_score_)
# print(gd.best_estimator_)

# #### Testing new Parameters:

# In[73]:


random_forest = RandomForestClassifier(criterion='gini',
                                           n_estimators=1750,
                                           max_depth=7,
                                           min_samples_split=6,
                                           min_samples_leaf=6,
                                           max_features='auto',
                                           oob_score=True,
                                           random_state=42,
                                           n_jobs=-1,
                                           verbose=1) 
random_forest.fit(X_train, Y_train)
Y_prediction = (random_forest.predict(X_test)).astype(int)

random_forest.score(X_train, Y_train)

print("oob score:", round(random_forest.oob_score_, 4)*100, "%")


# StratifiedKFold is used for stratifying the target variable. The folds are made by preserving the percentage of samples for each class in target variable (Survived).

# In[74]:


from sklearn.model_selection import StratifiedKFold
N = 5
oob = 0
probs = pd.DataFrame(np.zeros((len(X_test), N * 2)), columns=['Fold_{}_Prob_{}'.format(i, j) for i in range(1, N + 1) for j in range(2)])
fprs, tprs, scores = [], [], []

skf = StratifiedKFold(n_splits=N, random_state=N, shuffle=True)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, Y_train), 1):
    print('Fold {}\n'.format(fold))
    
    # Fitting the model
    random_forest.fit(X_train[trn_idx], Y_train[trn_idx])
    
    # Computing Train AUC score
    trn_fpr, trn_tpr, trn_thresholds = roc_curve(Y_train[trn_idx], random_forest.predict_proba(X_train[trn_idx])[:, 1])
    trn_auc_score = auc(trn_fpr, trn_tpr)
    # Computing Validation AUC score
    val_fpr, val_tpr, val_thresholds = roc_curve(Y_train[val_idx],random_forest.predict_proba(X_train[val_idx])[:, 1])
    val_auc_score = auc(val_fpr, val_tpr)  
      
    scores.append((trn_auc_score, val_auc_score))
    fprs.append(val_fpr)
    tprs.append(val_tpr)
    
    # X_test probabilities
    probs.loc[:, 'Fold_{}_Prob_0'.format(fold)] = random_forest.predict_proba(X_test)[:, 0]
    probs.loc[:, 'Fold_{}_Prob_1'.format(fold)] = random_forest.predict_proba(X_test)[:, 1]
        
    oob += random_forest.oob_score_ / N
    print('Fold {} OOB Score: {}\n'.format(fold, random_forest.oob_score_))   
    
print('Average OOB Score: {}'.format(oob))


# Now that we have a proper model, we can start evaluating it’s performace in a more accurate way. Previously we only used accuracy and the oob score, which is just another form of accuracy. The problem is just, that it’s more complicated to evaluate a classification model than a regression model. We will talk about this in the following section.

# In[75]:


from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100,oob_score=True)
scores = cross_val_score(random_forest, X_train, Y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# ### Further Evaluation:
# 
# #### Confusion Matrix:
#  
# **[What is Confusion Matrix?](https://www.geeksforgeeks.org/confusion-matrix-machine-learning/)** 

# In[76]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)
confusion_matrix(Y_train, predictions)


# The first row is about the not-survived-predictions: **500 passengers were correctly classified as not survived** (called true negatives) and **49 where wrongly classified as not survived** (false positives).
# The second row is about the survived-predictions: **88 passengers where wrongly classified as survived** (false negatives) and **254 where correctly classified as survived** (true positives).
# A confusion matrix gives you a lot of information about how well your model does, but theres a way to get even more, like computing the classifiers precision.

# #### Precision and Recall:
# 
# **[What's Precision And Recall ?](https://towardsdatascience.com/precision-vs-recall-386cf9f89488)**

# In[77]:


from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(Y_train, predictions))
print("Recall:",recall_score(Y_train, predictions))


# Our model predicts 84% of the time, a passengers survival correctly (precision). The recall tells us that it predicted the survival of 74 % of the people who actually survived.

# #### F-Score

# In[78]:


from sklearn.metrics import f1_score
f1_score(Y_train, predictions)


# There we have it, a 79 % F-score. The score is not that high, because we have a recall of 74%. But unfortunately the F-score is not perfect, because it favors classifiers that have a similar precision and recall. This is a problem, because you sometimes want a high precision and sometimes a high recall. The thing is that an increasing precision, sometimes results in an decreasing recall and vice versa (depending on the threshold). This is called the precision/recall tradeoff.

# #### ROC AUC Curve
# 
# **[What is an ROC AUC Curve ?](https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/)**

# In[79]:


from sklearn.metrics import roc_curve

# getting the probabilities of our predictions
y_scores = random_forest.predict_proba(X_train)
y_scores = y_scores[:,1]

# compute true positive rate and false positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, y_scores)
# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()


# The red line in the middel represents a purely random classifier (e.g a coin flip) and therefore your classifier should be as far away from it as possible. Our Random Forest model seems to do a good job.
# Of course we also have a tradeoff here, because the classifier produces more false positives, the higher the true positive rate is.

# #### ROC AUC Score:
# 
# The ROC AUC Score is the corresponding score to the ROC AUC Curve. It is simply computed by measuring the area under the curve, which is called AUC.
# A classifiers that is 100% correct, would have a ROC AUC Score of 1 and a completely random classiffier would have a score of 0.5.

# In[80]:


from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(Y_train, y_scores)
print("ROC-AUC-Score:", r_a_score)


# Nice ! I think that score is good enough to submit the predictions for the test-set to the Kaggle leaderboard.

# The Accuracy of this model on kaggle leaderboard = **81.1%**. Quite Reasonable Score for so much HardWork .

# In[81]:


submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": Y_prediction
    })

submission.to_csv('submission.csv', index=False)


# In[82]:


data=pd.read_csv("submission.csv")
data.head(10)


# In[ ]:




