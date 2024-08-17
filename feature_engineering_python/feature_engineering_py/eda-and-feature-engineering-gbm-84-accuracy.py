#!/usr/bin/env python
# coding: utf-8

# # ***🚢Titanic🚢* - Machine Learning from Disaster** 
# 
# 

# ![61ee91a5ed13441560df77e0.jpg](attachment:82249f9b-248c-47fd-b819-352e14d1c8b6.jpg)

# # ***What Is The Problem To Solve❓***
# 
# 

# > The sinking of the Titanic is one of the most infamous shipwrecks in history.
# On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.
# While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others. 
# 
# 

# > **Let's find who were more likely to survive than others and trying to generalize with machine learning algorithms!**

# # **Explaining of The Data**

# #### **Table of the columns**

# ![download.png](attachment:122284cf-3bce-497b-8379-00a131d97de0.png)

# - **Variable Notes**
# 
#     - pclass: A proxy for socio-economic status (SES)
#         
#         1st = Upper
#         
#         2nd = Middle
#         
#         3rd = Lower
# 
#     - age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
#     - sibsp: The dataset defines family relations in this way...
#     
#         Sibling = brother, sister, stepbrother, stepsister
#         
#         Spouse = husband, wife (mistresses and fiancés were ignored)
# 
#     - parch: The dataset defines family relations in this way...
#     
#         Parent = mother, father
#         
#         Child = daughter, son, stepdaughter, stepson
#         
#         Some children travelled only with a nanny, therefore parch=0 for them.

# ***

# <a id = "1"></a> 
# <h1 id="Introduction">
#     <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#000000;
#            font-size:90%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            font-weight: Bold">
# 
# <p style="padding: 10px;
#           color:white;
#           text-align:center;">📜CONTENT📜
# </p>
# </div>
# </h1>

# ## Importing of Libraries
# ## First Look to Data
# ## Exploratory Data Analysis
# ## Feature Engineering
# ## Model
# ## Submission

# <a id = "1"></a> 
# <h1 id="Introduction">
#     <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#ef5353;
#            font-size:90%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            font-weight: Bold">
# 
# <p style="padding: 10px;
#           color:white;
#           text-align:center;">📚Importing of Libraries📚
# </p>
# </div>
# </h1>

# ## İmporting of the core libraries

# In[59]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import missingno
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore') # Deleting of the warnings which we are not be interested in


# ## İmporting of the metrics&encoding&split&preprocessing libraries

# In[60]:


from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import confusion_matrix 


# ## İmporting of the machine learning algorithms

# In[61]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
import xgboost as xgb
import lightgbm as lgbm


# <a id = "2"></a> 
# <h1 id="Introduction">
#     <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#ef5353;
#            font-size:90%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            font-weight:Bold">
# 
# <p style="padding: 10px;
#           color:white;
#           text-align:center;">🧐First Look To Data🧐
# </p>
# </div>
# </h1>

# In[62]:


df_train = pd.read_csv("../input/titanic/train.csv")
df_test = pd.read_csv("../input/titanic/test.csv")

df_train = df_train.copy()
df_test = df_test.copy()

# Let's concatenate test and train datasets, because we'll do a lot feature engineering and we need to get same feature sizes of train and test datasets
df_train_len = len(df_train)
df = {}
df = pd.DataFrame(df)
df = pd.concat([df_train,df_test],axis=0).reset_index(drop=True)


# ### **For Train Data**

# In[63]:


print("*"*30, "HEAD", "*"*30)
display(df.head(5))
print("*"*30, "SHAPE", "*"*30)
print(f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}")
print("*"*30, "INFO", "*"*30)
display(df.info())
print("*"*30, "DESCRIBE", "*"*30)
display(df.describe().T)
print("*"*30, "NULL?", "*"*30)
display(df.isnull().sum())
print("*"*30, "DUPLICATED", "*"*30)
display(df.duplicated().sum())
print("*"*30, "EXPLAINING", "*"*30)


# **With the first look of the train data, we know:**
# - There are 1309 rows and 12 columns
# - There are "5 int64", "5 object", "2 float" columns
# - **5 columns have missing values** , **Age: 263, Cabin: 1014, Embarked: 2, Fare: 1 and Survived(But,Survived missing values correspond to the concatenate test-train datasets (Survived column doesn't exist in test set and has been replace by NaN values when concatenating the train and test set)**
#     - We need to handle with missing values, because **some machine learning algorithms don't know the modelling with the missing values** and missing values can cause the errors.
# - There are no duplicated values
#     - This is good, because **duplicated values cause the "bias"**. If there was a duplicated values, we need to handle with them also.
#     

# ![ocean_se_below-surface_aspot_AdobeStock.jpg](attachment:4e15c60c-320c-4fa5-899b-d59c4fc08c7b.jpg)

# #### **Let's dive deep into!**

# <a id = "3"></a> 
# <h1 id="Introduction">
#     <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#ef5353;
#            font-size:90%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            font-weight: Bold">
# 
# <p style="padding: 10px;
#           color:white;
#           text-align:center;">📈📊Exploratory Data Analysis📈📊
# </p>
# </div>
# </h1>

# # **Let's analyze all features by one by🎬**

# ## **Survived ⁉**

# In[64]:


f,ax=plt.subplots(1,2,figsize=(18,8))
df_train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0], colors = ['red', 'blue'],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=df_train,ax=ax[1],palette="Set1", hue = "Sex")
ax[1].set_title('Survived')
plt.show()


# #### **We can observe from the figures, there are %61,6 died and %38,4 survived people in train dataset.**

# ## Sex👩👨

# In[65]:


pd.crosstab(df.Sex,df.Survived,margins=True).style.background_gradient(cmap='inferno')


# In[66]:


female_survived_percentage = (233/314)*100
print(f"Percentage of survived females: {female_survived_percentage}")
male_survived_percentage = (109/577)*100
print(f"Percentage of survived males: {male_survived_percentage}")


# ![kate-jack-titanic.jpg](attachment:5d35cfce-c8f0-4ebb-ab03-4da23cc48f71.jpg)

# #### From the crosstab, we can say the females had chance to live much more than the males. According to the percantage values, we can easily see that.
# #### But, will we see the importance of the "Sex" feature in machine learning algorithms? Let's see at the end.

# ## PCLASS 💲

# In[67]:


display(df_train.groupby("Pclass")[["Survived"]].sum())

f,ax=plt.subplots(1,2,figsize=(18,8))

sns.countplot(x='Pclass',data=df_train,ax=ax[0], palette = "icefire_r")
ax[0].set_title('Number Of Passengers By Pclass')

sns.countplot(x='Pclass',hue='Survived',data=df_train,ax=ax[1], palette = "inferno")
ax[1].set_title('Pclass:Survived vs Dead')



# So, we did some visualizations but what was the "Pclass"?
# - pclass: A proxy for socio-economic status (SES)
# 
#     - 1st = Upper
# 
#     - 2nd = Middle
# 
#     - 3rd = Lower
#     
# From the first table, it can be seen the exact number of the survived people according to the "Pclass". 
# So, 
# - 1=136
# - 2=87
# - 3=119
# 
# From these numbers and the plot, we can see **the most survived number is in the first class, then third and last one is second class.**
# 
# In addition we can see from the first plot, there are the most people in ship from the third class.
# 
# **Let's examine this feature deep into**

# In[68]:


pd.crosstab([df_train.Sex,df_train.Survived],df_train.Pclass).plot.bar(stacked=True, color = ["pink","red", "blue"], figsize=(10,5))


# When we look at the stacked bar plot:
# 
#   - **The most died people in "Pclass 3" for females and in "Pclass 1" there are only 3 women died.**
#   - **Also, the most died people in "Pclass 3" for males.**
#     
# In titanic, people with "Pclass 3" were much more than others. So, we expect to the most died people in there. But, we need to look at the percentage of each Pclass to explain correctly.

# In[69]:


pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', None) 
pd.set_option('display.max_colwidth', None) # To see whole dataframe

percentage_list1 = []
percentage_list2 = []
percentage_dict = {"Percentages": {"Survived females": percentage_list1, "Survived males": percentage_list2 } }

for i in range(1,4):
    df_female_all = df_train.query(f'Pclass == {i} & Sex == "female"')
    f_all = df_female_all["Survived"].count()
    df_female_survived = df_train.query(f'Pclass == {i} & Survived == 0 & Sex == "female"')
    f_survived = df_female_survived["Survived"].count()
    percentage_f = 100-((f_survived/f_all)*100)
    percentage_list1.append([f"Pclass{i}",percentage_f])
   
for i in range(1,4):
    df_male_all = df_train.query(f'Pclass == {i} & Sex == "male"')
    m_all = df_male_all["Survived"].count()
    df_male_survived = df_train.query(f'Pclass == {i} & Survived == 0 & Sex == "male"')
    m_survived = df_male_survived["Survived"].count()
    percentage_m = 100-((m_survived/m_all)*100)
    percentage_list2.append([f"Pclass{i}",percentage_m])
              
table = pd.DataFrame(percentage_dict)               
display(table)    


# We can see the number of died and survived counts from the crosstab, but we need to fast looking of the percentages, because of that I made a simply code to see survived percentages.
# 
# **According to the percentages which can be seen from above table, "Survived" percentages P1>P2>P3 for both females and males.**
# 
# ***It can be seen that, upper socio economic status had high order in survived.***
# 
# Let's look at the correlation between the Pclass and Survived features.
# 

# In[70]:


df_pclass = df_train[["Pclass", "Survived"]]
display(df_pclass.corr())


# #### **Explaining of the correlation:**
#     - Correlation helps us to compare two variable.
#     - Correleation takes values in range of -1 and 1
#     - If we get -1, we have strong correlation in opposite direction. Well, while one variable is increasing, the other one is decreasing.
#     - If we get 1, we have strong correlation in the same direction. Well, while one variable is increasing, the other one is increasing also.
#     - Lastly, If we get 0, we have no correlation between two variables.
#     
# 

# ![korelasyon.png](attachment:2b108bd6-4baf-43a7-9b66-61c0ba907f0a.png)

# #### According to the correlation coefficient(-0.33) between the Pclass and Survived features. We can say there is a "weak correlation".
# 
# #### **Observation:** After the all analysis, Pclass feature looks important but not too much.

# ![main-qimg-5ab46f31803d2242e89996144a228ab1-lq.jpg](attachment:e458b5cc-bc86-450b-bdbb-3c119ff2d933.jpg)

# ## AGE 👨‍👩‍👧‍👦

# In[71]:


display(df_train[["Age"]].describe().T)
sns.displot(data=df_train, x="Age",kind="kde")


# Remember the explaining for age feature: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# **So, we have 80 max and 0.42 min values for the Age feature. In addition, mean value of age is 29.6 in the ship.**
# 

# In[72]:


g = sns.FacetGrid(df_train, col = "Survived")
g.map(sns.histplot, "Age", bins = 20)
plt.show()


# - Age <= 10 has a high survival rate,
# - Oldest passenger(80) in the ship survived,
# - Large number of around 20 years old did not survive,
# - Most passengers are in 15-35 age range,

# In[73]:


g = sns.FacetGrid(df_train, col = "Pclass")
g.map(sns.histplot, "Age", bins = 20)
plt.show()


# It can be easily seen, 
# - The majority of the elderly(>50) population on ship is in "Pclass1"
# - The majority of the younger(<35) population is in "Pclass3".

# #### **Observation:** Age feature looks like a important feature, and we should remember there was a missing values. I'll handle with the missing values in feature engineering part.

# ## Embarked⛵

# What was the "embarked" feature?
# 
# Port of Embarkation: C = Cherbourg, Q = Queenstown, S = Southampton

# In[74]:


sns.displot(data=df_train, x="Embarked", hue="Sex", palette="inferno")


# - Most people get in ship from Southampton and lowest people from Queenstown
# - From the Queenstown, most people are female.

# In[75]:


sns.factorplot('Embarked','Survived',data=df_train, palette = "inferno")
fig=plt.gcf()
fig.set_size_inches(5,3)


# #### **Survival rate order according to the port of embarkation is, Cherbourg>Queenstown>Southampton**

# In[76]:


sns.factorplot('Pclass','Survived',hue='Sex',col='Embarked',data=df_train, palette = "inferno")
fig=plt.gcf()
fig.set_size_inches(8,3)


# - As we examined before, Pclass 3 survival rate is very low.
# - For port Q, males survival rate is unfortuanetely very low for each Pclass
# - For port S, Pclass 3 survival rate is low

# ## Fare 💰

# Passenger fare➡

# In[77]:


display(df_train[["Fare"]].describe().T)

display(df_train.groupby("Pclass")[["Fare"]].mean())

display(df_train.groupby("Sex")[["Fare"]].mean())

sns.histplot(data=df_train, x="Fare")
fig = plt.gcf()
fig.set_size_inches(10,5)


# Let's explain the values that we get from the describe and groupby function:
# 
# - Mean value of Fare is 32.20, min value is 0 and max value is 512.32
# - We have a high standard deviation value that is 49.69
# - Mean fare value of passengers according to the Pclass;
#     - Pclass1: 84.15
#     - Pclass2: 20.66
#     - Pclass3: 13.67
#         - So, we can clearly see that, socio enomic status effect on fares.
# - If we get into the mean fare value of according to the "Sex", females fares is higher than the males.
# - Lastly, we can get from the plot, majority of people paid lowest fares. We already knew this from the Pclass feature also, because the most people were from Pclass3.

# In[78]:


g = sns.FacetGrid(df_train, col = "Survived")
g.map(sns.histplot, "Fare", bins = 10)

df_fares = df_train[["Fare", "Survived"]]
display(df_fares.corr())


# #### **Money is effect to the survival rate?**
# 
# - We can observe it from the graphs, died people are majority in lowest fares.
# 
# #### **Observation:** Money is unfortuanetely effect the survival rate. But we must notice that, there is a "weak correlation", because correlation coefficient is 0.25
# 

# ![62-original.jpg](attachment:dd431d36-11f1-473b-8e6c-8dc72d45530e.jpg)

# ## SibSp 

# Let's remember what was the SibSp feature.
# 
# - sibsp: The dataset defines family relations in this way...
# 
#     - Sibling = brother, sister, stepbrother, stepsister
# 
#     - Spouse = husband, wife (mistresses and fiancés were ignored)

# In[79]:


display(pd.crosstab([df_train.SibSp],[df_train.Survived, df_train.Pclass], margins=True).style.background_gradient(cmap='inferno'))
sns.catplot(data=df_train, x="SibSp", hue="Survived", kind="count" )


# #### Let's examine the SibSp feature according to the crosstab and countplot ➡
# 
# - Majority of the people have "0" value of SibSp, well 608 people are alone and 209 people just have "1" siblings or spouse. In total 817 people are alone or have 1 SibSp.
# - People have above the 3 siblings are in Pclass3 and almost of these people are death. It can be caused the number of the families, imagine you are in the titanic with your family, you were trying to save them and find them. Maybe, while they were searching or trying to save their family, they died.
# - While siblings value is increasing, survival rate is decreasing.

# In[80]:


df_sibsp = df_train[["SibSp", "Survived"]]
display(df_sibsp.corr())


# #### There is no correlation between the SibSp and Survived features. Because, density of "0-1 SibSp" is high.

# ## Parch

# parch: The dataset defines family relations in this way...
# 
# Parent = mother, father
# 
# Child = daughter, son, stepdaughter, stepson
# 
# Some children travelled only with a nanny, therefore parch=0 for them.

# #### **We have one more family feature "Parch". I'll concatenate Parch and SibSp to get whole Family_Number in "Feature Engineering" part and we get rid off the redundant features like that.**
# 
# - But before these operations, Let's examine "Parch" in the same way like "SibSp"➡.

# In[81]:


display(pd.crosstab([df_train.Parch],[df_train.Survived, df_train.Pclass], margins=True).style.background_gradient(cmap='inferno'))
sns.catplot(data=df_train, x="Parch", hue="Survived", kind="count" )


# - Density of the "Parch" feature is high in 0-1-2 numbers, We noticed in "SibSp" part the most people were alone, also we can observe this situation from the "Parch".
# - While parch number is increasing(>3), we can observe, almost everyone is from Pclass3 and survival rate is so low.

# <a id = "4"></a> 
# <h1 id="Introduction">
#     <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#ef5353;
#            font-size:90%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            font-weight: Bold">
# 
# <p style="padding: 10px;
#           color:white;
#           text-align:center;">🔎💡Feature Engineering🔎💡
# </p>
# </div>
# </h1>

# ### **Steps:**
# 
# - **Handling with missing values**
# - **Handling with outliers**
# - **Feature Transformation**
# - **Encoding**
# - **Feature Selection**
# - **Preparing of Train and Test Values**
# - **Feature Scaling**

# ## **❗HANDLING WITH MISSING VALUES❗**

# In[82]:


missingno.matrix(df,figsize=(10,5), fontsize=12);


# #### **It can be clearly seen from the matrix, like we said age and cabin features have missing values and cabin's values looks like have much more missing values.Also, survived have missing values but because of the test dataset. We will split the dataset in model section, so we will not have the missing value problem of Survived feature.**

# In[83]:


missingno.bar(df, color="black", sort="ascending", figsize=(10,5), fontsize=12);


# #### It can be clearly seen 2 missing values of "embarked" and 1 missing value of the "Fare" in here.

# There are mainly two methods of handling with missing values.
# - **Deleting missing value**s(This is useful when column has a %65+ percent of missing values or we couldn't get any information from the colums causing by the missing values)
# - **Impute missing values with "mean,median,zero" for quantative variables**(Like, Age)
# - **Impute missing values with "mode, "other"" for qualitative variables**(Like, Embarked)
# Let's get rid off the missing values with "Imputation Method"
# 
# **Let's begin ➡**

# In[84]:


df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace = True)
df["Fare"]= df["Fare"].fillna(df["Fare"].mean())
df.isnull().sum()


# We filled the missing values of the "Age" and "Embarked"

# In[85]:


cabin_missing =df["Cabin"].isnull().value_counts()[1] #missing values of cabin
cabin_values = df["Cabin"].count() # values 
missing_percentage_cabin = cabin_missing/(cabin_missing+cabin_values)
print(f"Percentage of the missing values in Cabin attiribute: {missing_percentage_cabin}")


# Cabin have a %77 percent of missing values, So I will delete this attribute, because it's redundant.

# In[86]:


df.drop(columns="Cabin", inplace=True)


# In[87]:


df.isnull().sum()


# **So, we get rid off the all missing values ✅**

# ## Handling with Outliers

# In[88]:


sns.boxplot(data=df_train, x="Fare")


# We can see the outliers in "Fare", but, titanic have a lot of choices for passengers fares and mostly people from the ship are in Pclass3 and mean value of fare is creating density in around 0-100 because of that, we are detecting outliers on boxplot. Shortly, these outliers can be and we can't describe them as an outliers. 
# 
# Also, we can explain the same situation for the "Age" feature, so, I will not examine it in outliers
# 

# ## Feature Transformation

# In this subset, I'll create new features with existing features.
# 
# In the EDA section, I said I will concatanete the "SibSp" and "Parch" to new feature "Family". Because, in totaly these two feature have the family numbers.
# 
# Let's implement it➡

# #### "SibSp" + "Parch" ➡ "Family"

# In[89]:


df["Family"]=0
df["Family"]= df["SibSp"]+df["Parch"]
display(df.head())
display(df.groupby("Family")[["Sex"]].count())


# - We can see new feature on the table, total of the "Parch" and "SibSp".
# - SibSp had 8 bins and Parch had 6 bins, now it can be seen from the bottom table Family feature have 10 bins for the family sizes. It is the clear way of looking the total family size.

# ### Encoding

# - **Any machine learning algorithm**, whether it is a linear-regression or a KNN-utilizing Euclidean distance, requires numerical input features to learn from.
# - Categorical inputs cause the some errors and mostly machine learning algorithms don't know to work with the object inputs.
# 
# There are several methods we can rely on to transform our categorical data into numerical data.
# 
# We will use two of them:
# - Label Encoding
# - One-Hot Encoding
# 
# This methods simply the convert object inputs to numerical input and in binary system.
# 
# - We will use **label encoding for "Sex"** feature and it'll convert the female-male to 0-1.
# - For **Embarked feature we will use one-hot encoding** and we'll get 2 new columns with binary system.
# 
# Note: While we implement encoding methods, We should care the **dummy variable trap**. But, what is the dummy variable trap?
# 
# **The dummy variable trap** is when you have independent variables that are multicollinear, or highly correlated. Simply put, these variables can be predicted from each other. If we create two columns by "Sex" feature, female(0-1) and male(0-1) these two columns can be predicted from each other. So, we'll create just one column for "Sex" feature.
# 

# #### **"Sex"**

# In[90]:


lbe=LabelEncoder()
df["Sex"] = lbe.fit_transform(df['Sex'])
display(df.head())


# We tranformed the "Sex" feature to ingeter values.

# #### **"Embarked"**

# In[91]:


df = pd.get_dummies(df, columns = ["Embarked"], prefix = ["Embarked"], drop_first = True)
# It is created dummy variables of "Embarked" features with one-hot encoding method. 
#For avoid the dummy variable trap, I used "drop_first", with this first column that I have created is deleted.


# In[92]:


df.head(5)


# ## Feature Selection

# ### Redundant Features

# **Name**➡ We don't need name feature as it cannot be converted into any categorical value. It include some titles(Mr,Mrs,Ms) of people but, we already have family feature.
# 
# **Ticket**➡ It is any random string that cannot be categorised.
# 
# **Cabin**➡ We already drop this feature because of the missing values
# 
# **PassengerId**➡ Cannot be categorised.

# In[93]:


df.drop(columns=["Name","Ticket","PassengerId"], inplace = True)


# ### Selection with Correlation

# In[94]:


fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,cmap='inferno',linewidths=.9, ax=ax)


# **The first thing to note is that only the numeric features are compared as it is obvious that we cannot correlate between alphabets or strings.We analyze some correlations in the previous parts, for example, we said correlation between the Survived-Fare is 0.25. It's the same value in heatmap, it is 0.26 just rounded value of that we calculated. Wıth the heatmap we can analyze and get observations of whole data according to the correlations.**
# #### Observations from the heatmap:
#  - **The maximum correleation numbers** in heatmap from the Family with Parch-SibSp because we create new feature as family using Parch and SibSp. Because of that, **there is multicollinearity, so I'll delete Parch and SibSp.**
#  - Pclass and Fare has a good negative relation. Because when Pclass is increasing, passengers fares is decreasing.    
# 

# In[95]:


df.drop(columns=["Parch","SibSp"], inplace = True)


# In[96]:


df.head()


# **Now, we have 6 features to predict "Survived" values and all features are in numeric format.**

# In[97]:


df.corr()[["Survived"]]


#  - **Let's look at the "Survived" feature's correlation numbers**, when we look at the dependent variable's correlation numbers we want to strong relations because we care prediction. But, If we look at the correlations of "Survived", except the Pclass and Sex features almost all features have no correlation. Already we deleted a lot features, because of that I can't do anything with correlation numbers.

# ### We did feature transformation and selection parts, so we can split the df into the train and test datasets again.

# In[98]:


df_train = df[:df_train_len]
df_test = df[df_train_len:]
df_test.drop(labels=["Survived"],axis = 1,inplace=True)


# In[99]:


df_train["Survived"]=df_train["Survived"].astype("int64")


# In[100]:


df_train.head()


# In[101]:


df_test.head()


# ### Preparing of Train and Test Values

# In[102]:


y_train = df_train["Survived"]
X_train = df_train.drop(labels = ["Survived"],axis = 1)
X_test = df_test


# ### Feature Scaling

# Some machine learning models rely on learning methods that are affected greatly by the scale of the data, meaning that if we have a column such as "x" that lives between 24 and 122, and an "y" column between 21 and 81, then our learning algorithms will not learn optimally.
# 
# Normalization techniques are meant to level the playing field of data by ensuring that **all rows and columns are treated equally under the eyes of machine learning**.
# 
# There are mainly three types of scaling:
# 
# - Z-score standardization
# 
# - Min-max scaling
# 
# - Normalization
# 
# I'll use normalization method ➡
# 
# **Note:** No matter the scaling method, feature scaling always divides the feature by a constant (known as the normalization constant). Therefore, it does not change the shape of the single-feature distribution.

# ![normalization.png](attachment:7fa0f777-8d24-4a0c-9b97-3a283bed7323.png)

# In[103]:


preprocessing.normalize(X_train)


# We get array after the normalization, let's convert it to dataframe.

# In[104]:


X_train = pd.DataFrame(X_train)


# ### After the all operations, we prepared our dataset to fit model, Let's Begin and see how many accuracy we get?

# <a id = "5"></a> 
# <h1 id="Introduction">
#     <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#ef5353;
#            font-size:90%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            font-weight: Bold">
# 
# <p style="padding: 10px;
#           color:white;
#           text-align:center;">🤖Model🤖
# </p>
# </div>
# </h1>

# ### In this section, I am going to create base modelling with cross validation and then I'll chose the best model according to the accuracy score. After that, I'll do hyper parameter tuning for the best model and I'll try to improve accuracy score.

# ## Base Model with Cross Validation

# In[105]:


kfold = StratifiedKFold(n_splits=10)
random_state = 6

logistic_model = LogisticRegression(solver='lbfgs', max_iter=400,random_state=random_state).fit(X_train,y_train)
knn_model = KNeighborsClassifier().fit(X_train, y_train)
decision_model = DecisionTreeClassifier(random_state=random_state).fit(X_train,y_train)
mlp_model = MLPClassifier(random_state=random_state).fit(X_train, y_train)
gaussian_model = GaussianNB().fit(X_train, y_train)
linear_svm_model = SVC(kernel='linear').fit(X_train,y_train)
adaboost_model = AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state, learning_rate=0.1).fit(X_train,y_train)
randomforest_model = RandomForestClassifier(random_state=random_state).fit(X_train,y_train)
extra_model = ExtraTreesClassifier(random_state=random_state).fit(X_train,y_train)
gb_model = GradientBoostingClassifier(random_state=random_state).fit(X_train,y_train)
xgb_model = xgb.XGBClassifier().fit(X_train,y_train)
lgbm_model = lgbm.LGBMClassifier().fit(X_train,y_train)

model_names = ["Logistic","Knn","DecisionTree","MLP","GaussianNB","SupportVectorMachine","AdaBoost","RandomForest","ExtraTrees","GradientBoost","Xgboost","Lightgbm"]
model_list = [logistic_model,decision_model,mlp_model,knn_model,gaussian_model,linear_svm_model,adaboost_model,randomforest_model,extra_model,gb_model,xgb_model,lgbm_model]
results = []
for i in model_list:
    result = cross_val_score(i, X_train, y_train, scoring = "accuracy", cv = kfold, n_jobs=4)
    results.append(result.mean())

acc_of_models = {"Model": model_names, "Mean Accuracy": results}    
acc_of_models = pd.DataFrame(acc_of_models)
acc_of_models


# In[106]:


f,ax =plt.subplots(2,2, figsize = (15,10))

pd.Series(gb_model.feature_importances_,X_train.columns).sort_values(ascending=True).plot.barh(width=0.8,color='#800080',ax=ax[0,0])
ax[0,0].set_title('Feature Importance in Gradient Boosting')
pd.Series(xgb_model.feature_importances_,X_train.columns).sort_values(ascending=True).plot.barh(width=0.8,color='#FF00FF',ax=ax[0,1])
ax[0,1].set_title('Feature Importance in XGBoost')
pd.Series(lgbm_model.feature_importances_,X_train.columns).sort_values(ascending=True).plot.barh(width=0.8,color='#FFC0CB',ax=ax[1,0])
ax[1,0].set_title('Feature Importance in LightGBM')
pd.Series(randomforest_model.feature_importances_,X_train.columns).sort_values(ascending=True).plot.barh(width=0.8,color='#FFFF00',ax=ax[1,1])
ax[1,1].set_title('Feature Importance in RandomForests')


# In EDA part we were expecting the high importance of the "Fare","Sex","Pclass" and our some expectations are happened.

# In[107]:


sns.barplot(data=acc_of_models, x="Mean Accuracy", y="Model")


# We fit 10 types of classification and ensemble models with Kfold(n=10) cross validation.
# 
# So, we get the **best mean accuracy** from the **"GradientBoosting" with %83 and it's the good value for the base model**
# 
# #### Let's into the "HyperParemeter Tuning" with "GradientBoostingClassifier".

# ## Hyper Parameter Tunning with GridSearchCV

# In[108]:


#GBC = GradientBoostingClassifier()
#gbc_params = {
              #'n_estimators' : [100,200,500,1000],
              #'learning_rate': [0.1,0.01,0.001],
              #'max_depth': [2,3,7],
              #'min_samples_split':[2,10,50,100],
              #'min_samples_leaf': [1,5,7],
              #'max_features': [0.3, 0.1] 
              #}
#gradient_tuned = GridSearchCV(GBC,gbc_params, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1).fit(X_train,y_train)


# In[109]:


#print(f"Best parameters for the Gradient Boosting Model: {gradient_tuned.best_params_}")


# - Best parameters for the Gradient Boosting Model: {'learning_rate': 0.1, 'max_depth': 7, 'max_features': 0.3, 'min_samples_leaf': 7, 'min_samples_split': 2, 'n_estimators': 100}
# - We get best params, because of that I close the code of the hyper parameter tuning with "#". It takes so much time.

# In[110]:


GBC = GradientBoostingClassifier(random_state=6)
gbc_params = {'learning_rate': [0.1], 'max_depth': [7], 'max_features': [0.3], 'min_samples_leaf': [7], 'min_samples_split': [2], 'n_estimators': [100]}
gradient_tuned = GridSearchCV(GBC,gbc_params , cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1).fit(X_train,y_train)
print(f"Gradient Boosting Best Accuracy Score: {gradient_tuned.best_score_}")
y_pred = gradient_tuned.predict(X_test)


# - Gradient Boosting Best Accuracy Score: 0.8395880149812734
# - We improved our model a little
# 
# ### **After the tuning we get the %84 accuracy and it's the perfect score**

# <a id = "6"></a> 
# <h1 id="Introduction">
#     <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#ef5353;
#            font-size:90%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            font-weight: Bold">
# 
# <p style="padding: 10px;
#           color:white;
#           text-align:center;">✅Submission✅
# </p>
# </div>
# </h1>

# In[111]:


submission = pd.read_csv("../input/titanic/gender_submission.csv")


# In[112]:


submission[["PassengerId"]].head()


# In[113]:


y_pred = pd.DataFrame(y_pred, columns = ["Survived"])
y_pred.head()


# In[114]:


submission_last = pd.concat([submission[["PassengerId"]],y_pred],axis=1)


# In[115]:


submission_last.head()


# In[116]:


submission_last.to_csv('submission_last.csv',index=False)

