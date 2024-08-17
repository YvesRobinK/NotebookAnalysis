#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this notebook I try to predict the target "Survived" with only 3 features:
# 
# 1. **Pclass**
# 2. **Title** (Extracted from "Name" column)
# 3. **Deck** (Extracted from "Cabin" column)

# In[1]:


import numpy as np 
import pandas as pd
from Levenshtein import distance as lev
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="ticks", color_codes=True)
import warnings
warnings.filterwarnings('ignore')


# In[2]:


train = pd.read_csv(r'../input/titanic/train.csv')
test =  pd.read_csv(r'../input/titanic/test.csv')

All = pd.concat([train, test], sort=True).reset_index(drop=True)
All.head()


# In[3]:


columns = test.columns
for i in range(len(columns)):
    print(columns[i],'--->',"train:",train[columns[i]].isnull().sum(),
         "|","test:",test[columns[i]].isnull().sum()) 
    
print("Total number of Cabin's missing values is:",All['Cabin'].isnull().sum())


# As we see above, we have a mising values in Age, Fare, Embarked and massively in Cabin.
# 
# At first we filling he missing values in "Fare" and "Embarked".

# In[4]:


All[All['Fare'].isnull()]
mean_fare = All.groupby(['Pclass', 'Parch', 'SibSp']).Fare.mean()[3][0][0]
All['Fare'] = All['Fare'].fillna(mean_fare)

All[All['Embarked'].isnull()]
All['Embarked'] = All['Embarked'].fillna('S')



# # Cabin
# 
# I wrote another notebook called "Meditation in the Cabin" in which I tried to find a logical way to fill the missing values in the Cabin column.
# 
# There I have explained in detail my method of doing this.
# 
# Here I am just rewriting the script I used to fill the missing values in the cabin.
# 
# For more information and explanations, you can refer to the notebook mentioned below:
# 
# https://www.kaggle.com/code/khashayarrahimi94/meditation-on-the-cabin

# In[5]:


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

All['Cabin']


# In[6]:


All['Cabin'].isnull().sum()


# # Title
# 
# Here we extract a new feature called "Title" from the "Name" column. 
# (I mentioned the source here:
# 
# https://www.kaggle.com/code/khashayarrahimi94/knn-xgboost-svc-ensemble-with-just-5-feature)
# 
# **NOTE:**
# 
# Further we drop the important column "Sex", because it's information exists in the "Title" column too. 

# In[7]:


for name in All["Name"]:
    All["Title"] = All["Name"].str.extract("([A-Za-z]+)\.",expand=True)

title_replacements = {"Mlle": "Other", "Major": "Other", "Col": "Other", "Sir": "Other", "Don": "Other", "Mme": "Other",
          "Jonkheer": "Other", "Lady": "Other", "Capt": "Other", "Countess": "Other", "Dona": "Other"
                     ,"Dr":"Other","Rev":"Other", "Mrs":"Woman","Ms":"Woman","Miss":"Woman"}

All.replace({"Title": title_replacements}, inplace=True)


# In[8]:


All['Title'].value_counts().plot(kind='bar')


# In[9]:


plt.figure(figsize=(8,5))
sns.countplot(x='Title',data=All.head(891), palette='rainbow',hue='Survived')


# In[10]:


All


# # Age
# 
# Here we fill the missing values in the "Age" column by usuing mutual information to detect the most related feature and then, filling the missing values grouping by the "Sex" and "Pclass" and replacing the median of them with nan values.

# In[11]:


from sklearn.feature_selection import mutual_info_classif as MIC
mi_score = MIC(train.loc[: , ['Age' ,'Pclass','Parch','Fare','SibSp' ]].values.astype('int'),
               train.loc[: , ['Age']].values.astype('int').reshape(-1, 1))
Feature2 = ['Age' ,'Pclass','Parch','Fare','SibSp' ]
Mutual_Information_table = pd.DataFrame(columns=['Feature1', 'Feature2', 'MIC'], index=range(5))
Mutual_Information_table['Feature1'] = 'Age'
for feature in range(5):
    Mutual_Information_table['Feature2'][feature] = Feature2[feature]
for value in range(5):
    Mutual_Information_table['MIC'][value] = mi_score[value]
Mutual_Information_table


# In[12]:


age_by_pclass_sex = round(All.groupby(['Sex', 'Pclass']).median()['Age'])

for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Mean age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
print('Mean age of all passengers: {}'.format(round(All['Age'].mean())))

All['Age'] = All.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(round(x.median())))


# In[13]:


All_org = All.copy()


# In[14]:


All


# In[15]:


All['Title'] = pd.factorize(All['Title'])[0]
All['Cabin'] = pd.factorize(All['Cabin'])[0]
All['Sex'] = pd.factorize(All['Sex'])[0]
All['Embarked'] = pd.factorize(All['Embarked'])[0]
All.drop(['Name', 'Parch', 'PassengerId','SibSp',  'Ticket'], axis=1, inplace=True)
All.head()


# # Correlation Table
# 
# Here we have correlation table and choose the 3 features that are most correlated with the target "Survived":
# 
# * Pclass
# * Title 
# * Deck

# In[16]:


corr = All.head(891).corr()
corr.style.background_gradient(cmap='coolwarm')


# In[17]:


All.drop(['Age', 'Embarked', 'Fare','Sex'], axis=1, inplace=True)
All.head()


# # Deck
# Here we delete the numbers in front of decks to simplify the dataset and its encoding.

# In[18]:


All["Deck"]=All_org["Cabin"]
for i in range(All.shape[0]):
    All["Deck"][i] = All_org["Cabin"][i][0]
All.drop(['Cabin'], axis=1, inplace=True)
All.head()


# In[19]:


All['Deck'].value_counts().plot(kind='bar')


# In[20]:


sns.countplot(x='Deck',data=All.head(891), palette='rainbow',hue='Survived')


# In[21]:


All = pd.get_dummies(All, columns=['Pclass', 'Deck','Title'], prefix=['Pclass', 'Deck','Title'])
All.head()


# In[22]:


All.insert(16, "survived", All['Survived'])
All.drop(['Survived'], axis=1, inplace=True)
All.head()


# # Model
# 
# We tune the Random Forest Classifier and here we just use the result.

# In[23]:


train = All.head(891)

X = train.values[:,:-1]
Y = train.values[:,-1]
label_encoded_y = LabelEncoder().fit_transform(Y)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

RF = RandomForestClassifier(max_depth=5, max_leaf_nodes= 8, n_estimators= 175)

results = cross_val_score(RF, X, label_encoded_y, cv=kfold)
print(results.mean() ,results.std() )


# In[24]:


test = All.tail(418)
test.drop(['survived'], axis=1, inplace=True)
RF.fit(X, label_encoded_y)

Submission = pd.DataFrame({'PassengerId':list(range(892,1310))})
Submission['Survived']=RF.predict(test)
Submission


# In[25]:


Submission.to_csv('submission.csv', index=False)


# Here are my other notebooks, I'll be happy to know your comments and modificaions on the if needed:
# 
# 1. What NOT TO DO in Titanic(Feature Engineering):
# 
# https://www.kaggle.com/code/khashayarrahimi94/what-not-to-do-in-titanic-feature-engineering
# 
# 2. How divergence the train & test distributions are?
# 
# https://www.kaggle.com/code/khashayarrahimi94/how-divergence-the-train-test-distributions-are
