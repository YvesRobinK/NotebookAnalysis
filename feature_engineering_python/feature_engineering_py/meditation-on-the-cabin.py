#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Logically, intuitively and with correlation scores that are computed in the notebook mentioned below, we know that the **Cabin column** is one of the most important feature that can help us to predict the target column (Survived).
# 
# https://www.kaggle.com/code/khashayarrahimi94/knn-xgboost-svc-ensemble-with-just-5-feature/notebook
# 
# So understanding the Cabin is very helpful for our EDA and creating models to predict the target.
# 
# In this notebook we will try to extract some information about the cabin from other columns.
# 
# I wrote some notebooks that may help to understand the Cabin better, which I listed them below:
# 
# 1. https://www.kaggle.com/code/khashayarrahimi94/what-not-to-do-in-titanic-feature-engineering/notebook
# *(Focused on the Cabin and extracts some **interesting secrets** about it)*
# 
# 2. https://www.kaggle.com/code/khashayarrahimi94/how-divergence-the-train-test-distributions-are/notebook

# In[1]:


import numpy as np 
import pandas as pd
from Levenshtein import distance as lev
import re
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


train = pd.read_csv(r'../input/titanic/train.csv')
test =  pd.read_csv(r'../input/titanic/test.csv')

All = pd.concat([train, test], sort=True).reset_index(drop=True)
All.head()


# As we see in the followig, The Cabin column, besides its importance, has very missing values.

# In[3]:


columns = test.columns
for i in range(len(columns)):
    print(columns[i],'--->',"train:",train[columns[i]].isnull().sum(),
         "|","test:",test[columns[i]].isnull().sum()) 
    
print("Total number of Cabin's missing values is:",All['Cabin'].isnull().sum())


# Due to using the "Fare" and "Embarked" fo filling missing values of the "Cabin", We should first fill their nans. 

# In[4]:


All[All['Fare'].isnull()]
mean_fare = All.groupby(['Pclass', 'Parch', 'SibSp']).Fare.mean()[3][0][0]
All['Fare'] = All['Fare'].fillna(mean_fare)


# In[5]:


All[All['Embarked'].isnull()]
All['Embarked'] = All['Embarked'].fillna('S')


# The approach we use here for filling mising values of the "Cabin" column is searching for equality and also some sort of similarity between some features of records/passengers that their cabins are known and the ones that are not. 
# 
# For this, we choose 4 features that seems have logical relations with Cabin and they are:
# 
# 1. Pclass
# 2. Fare
# 3. Embarked
# 4. Ticket
# 
# For first try we seperate passengers by their Pclass, and in each class we search for passengers that paid same fares, have same embarked and similar tickets.
# 
# # Ticket simility
# 
# As tickets values are string, for measuring their similarity we must use **string metric**.
# Here we use **Levenshtein Distance** that its mathematical definition is as follows (from wikipedia):
# 
# ${\displaystyle \operatorname {lev} (a,b)={\begin{cases}|a|&{\text{ if }}|b|=0,\\|b|&{\text{ if }}|a|=0,\\\operatorname {lev} {\big (}\operatorname {tail} (a),\operatorname {tail} (b){\big )}&{\text{ if }}a[0]=b[0]\\1+\min {\begin{cases}\operatorname {lev} {\big (}\operatorname {tail} (a),b{\big )}\\\operatorname {lev} {\big (}a,\operatorname {tail} (b){\big )}\\\operatorname {lev} {\big (}\operatorname {tail} (a),\operatorname {tail} (b){\big )}\\\end{cases}}&{\text{ otherwise,}}\end{cases}}}$
# 
# To replacing missing values in the cabin with known ones, we apply the above conditions (same pclass,fare and embarked) and measure the similarity between their tickets.
# 
# At last, for each passenger that has unknown cabin value, we choose the one with known cabin value, that has shortest Levenshtein distance between their tickets among the others, and fill the nan cabin value with known one.

# ## How to read the following code output
# 
# As an example the following means:
# 
# > 1099 {2: 1099, 3: 964, 6: 1295} 30
# 
# The cabin value of *record 30* (PassengerId=31) is missing.
# 
# The records {1099,964,1295} (PassengerId={1100,965,1296} respectively) have known cabin value, and are in the same class with *record 30*, paid same fare and had same embarked.
# 
# The numbers in front of them are their Levenshtein distance between their tickets and *record 30*'s ticket;
# 
# > $lev(record 30,record 1099)=2$
# 
# > $lev(record 30,record 964)=3$
# 
# > $lev(record 30,record 1295)=6$
# 
# And *record 1099* has the the shortest Levenshtein distance and equivalently most similaity to *record 30* in terms of their tickets.

# In[6]:


All1 = All.copy()
pclass = [1,2,3]
for t in pclass:
    for i in range(All.shape[0]):
        if (All1.Pclass[i]==t)&(pd.isnull(All1["Cabin"][i])== True):
            LD_Ticket={}
            for j in range(All.shape[0]):
                if (All1.Pclass[j]==t)&(pd.isnull(All1["Cabin"][j])== False)&(All1.Fare[j]==All1.Fare[i])&(All1.Embarked[j]==All1.Embarked[i]):
                    LD_Ticket[lev(All1.Ticket[i],All1.Ticket[j])] = j
            if LD_Ticket !={}:
                similar_ticket = LD_Ticket[min(list(LD_Ticket))]
                #print(similar_ticket, LD_Ticket,i)                 #Uncomment this line to see the output, I comment it to make the notebook more clear
                All["Cabin"][i] = All["Cabin"][similar_ticket]


# As we see below, 312 missing value of the Cabin filled.

# In[7]:


All['Cabin'].isnull().sum()


# In[8]:


All2 = All.copy()
pclass = [1,2,3]
for t in pclass:
    for i in range(All2.shape[0]):
        if (All2.Pclass[i]==t)&(pd.isnull(All2["Cabin"][i])== True):
            LD_Ticket={}
            for j in range(All.shape[0]):
                if (All2.Pclass[j]==t)&(pd.isnull(All2["Cabin"][j])== False)&(abs((All2.Fare[j])- (All2.Fare[i]))<1):
                    LD_Ticket[lev(All2.Ticket[i],All2.Ticket[j])] = j
            if LD_Ticket !={}:
                similar_ticket = LD_Ticket[min(list(LD_Ticket))]
                #print(similar_ticket, LD_Ticket,i)
                All["Cabin"][i] = All["Cabin"][similar_ticket]


# In[9]:


All['Cabin'].isnull().sum()


# Here we replace equality in fares, with their difference lower than 20.

# In[10]:


All3 = All.copy()
pclass = [1,2,3]
for t in pclass:
    for i in range(All3.shape[0]):
        if (All3.Pclass[i]==t)&(pd.isnull(All3["Cabin"][i])== True):
            LD_Ticket={}
            for j in range(All.shape[0]):
                if (All3.Pclass[j]==t)&(pd.isnull(All3["Cabin"][j])== False)&(abs((All3.Fare[j])-(All3.Fare[i]))<20):
                    LD_Ticket[lev(All3.Ticket[i],All3.Ticket[j])] = j
            if LD_Ticket !={}:
                similar_ticket = LD_Ticket[min(list(LD_Ticket))]
                #print(similar_ticket, LD_Ticket,i)
                All["Cabin"][i] = All["Cabin"][similar_ticket]


# Now we have only 39 missing value:

# In[11]:


All['Cabin'].isnull().sum()


# In the last step, we eliminate the proximity of the fare:

# In[12]:


All4 = All.copy()
pclass = [1,2,3]
for t in pclass:
    for i in range(All.shape[0]):
        if (All4.Pclass[i]==t)&(pd.isnull(All4["Cabin"][i])== True):
            LD_Ticket={}
            for j in range(All.shape[0]):
                if (All4.Pclass[j]==t)&(pd.isnull(All4["Cabin"][j])== False):
                    LD_Ticket[lev(All4.Ticket[i],All4.Ticket[j])] = j
            if LD_Ticket !={}:
                similar_ticket = LD_Ticket[min(list(LD_Ticket))]
                #print(similar_ticket, LD_Ticket,i)
                All["Cabin"][i] = All["Cabin"][similar_ticket]


# Finally all of the missing values of the Cabin column are filled.

# In[13]:


All['Cabin'].isnull().sum()


# In[14]:


All['Cabin']


# In[15]:


All["Deck"]=All["Cabin"]
All["Cabin_Number"]=All["Cabin"]
for i in range(All.shape[0]):
    All["Deck"][i] = All["Cabin"][i][0]
    if not [int(x) for x in re.findall('\d+', All["Cabin"][i])]:
        All["Cabin_Number"][i] = 0
    else:
         All["Cabin_Number"][i] =[int(x) for x in re.findall('\d+', All["Cabin"][i])][0]


All[['Deck','Cabin_Number']]


# In[16]:


All['Deck'].value_counts().plot(kind='bar')


# In[17]:


sns.countplot(x='Deck',data=All.head(891), palette='rainbow',hue='Survived')


# In[18]:


Cabin_Number_list = All['Cabin_Number'].tolist()
print(max(Cabin_Number_list),min(Cabin_Number_list))


# In[19]:


Cabin_Number = sns.countplot(x='Cabin_Number',data=All.head(891))
Cabin_Number.set(xticks=[0,20,40,60,80,100])


# In[20]:


Cabin_Number = sns.countplot(x='Cabin_Number',data=All.head(891), hue='Survived')
Cabin_Number.set(xticks=[0,20,40,60,80,100])


# In[21]:


sns.relplot(x="Cabin_Number", y="Deck", data=All);


# ### I'm very interested to know your feedback on this notebook and if there is anything to add or modify.
# 
# ### Thanks for your attention
