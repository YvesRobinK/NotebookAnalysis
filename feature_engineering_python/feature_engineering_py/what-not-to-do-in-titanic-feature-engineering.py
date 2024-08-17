#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# 
# I have worked on the Titanic competition over time and tried several ways for days which lead me to ideas that don't work.
# In this notebook I want to point out them so that others can read it and save their time by not doing them.
# 
# This notebook is divided into two parts:
# 
# 1. Ideas for data preparation that doesn't help (+Some semi-secrets in Titanic dataset).
# 2. Ideas that don't improve the accuracy or even decrease it.
# 
# **This notebook will be completed over time.**

# In[1]:


import numpy as np 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from matplotlib import pyplot
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# In[2]:


train = pd.read_csv(r'../input/titanic/train.csv')
test =  pd.read_csv(r'../input/titanic/test.csv') 


# In[3]:


All = pd.concat([train, test], sort=True).reset_index(drop=True)
All


# # Data Engineering

# In[4]:


columns = test.columns
for i in range(len(columns)):
    print(columns[i],'--->',"train:",train[columns[i]].isnull().sum(),
         "|","test:",test[columns[i]].isnull().sum()) 


# ## 1. Filling nan values in Cabin by family relations

# As you see above, the Cabin column has 1014 nan values and only 295 are known, But as we see in the following, the positions of the Cabins affects the survival of the passengers.
# 
# 
# ![Cabins1](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/Titanic_side_plan_annotated_English.png/1100px-Titanic_side_plan_annotated_English.png)
# 
# <div style="width:100%;text-align: center;"> <img align=middle src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Olympic_%26_Titanic_cutaway_diagram.png/400px-Olympic_%26_Titanic_cutaway_diagram.png" style="height:600px;margin-top:3rem;"> </div>
# <br>
# To observe better the effects we can see the Titanic sinking simulation:
# <div style="width:100%;text-align: center;"> <img align=middle src="https://upload.wikimedia.org/wikipedia/commons/3/33/Titanic_sinking_gif.gif" style="height:300px;margin-top:3rem;"> </div>
# 

# Here we replace Cabin values by Deck.

# In[5]:


All["Cabin_dumb"]=All["Cabin"]
for i in range(All.shape[0]):
    if pd.isnull(All["Cabin"][i])== False:
        All["Cabin_dumb"][i] = All["Cabin"][i][0]
    else:
        All["Cabin_dumb"][i] =0
        


All["Cabin_dumb"]


# In[6]:


All["Cabin_dumb"].value_counts()


#  One of the natural idea for filling the missing values of the Cabin column is that:
#  <br>
# **All members of a family are in a same cabin and surely Deck.**
# 
# <br>
# But as we see in the following, If the passenger's deck is known, the deck of all his or her family will also be known.
# 
# <br>
# Therefore, WE CAN NOT fill missing values in Cabin/Deck with idea that said above.

# In[7]:


All['Name_new'] = All['Name']
nick = ["Mlle. ", "Major. ", "Col. ", "Sir. ", "Don. ", "Mme. ","Jonkheer. ",'Mr. ',
"Lady. ", "Capt. ", "Countess. ", "Dona. ","Dr. ","Rev. ", "Mrs. ","Ms. ","Miss. ",'Woman. ', 'Master. ']
                                        
for i in range(All.shape[0]):
    for j in nick:
        All['Name_new'][i]=All['Name_new'][i].replace(j,'')
All


# We check for each passenger with known deck their family members with the following conditions.
# 
# ## Conditions:
# 
# **max(match_size)>0:**                  The number of common words in passenger's name, set to at least one word.
# <br>
# **All["Cabin_dumb"][j]==0:**            The deck's value is nan.
# <br>
# **All['Pclass'][i]==All['Pclass'][j]:** Passengers be in the same Pclass.
# <br>
# **All['Family'][i]>0:**                 Passengers has family in Titanic.
# <br>
# **All['Family'][i]==All['Family'][j]:** Family member of nan and known passengers be equal.
# <br>
# **i!=j:**                               Prevent checking a passenger with him/herself. 

# In[8]:


All['Family'] = All['SibSp']+All['Parch']

class Solution:
    def solve(self, s0, s1):
        s0 = s0.lower()
        s1 = s1.lower()
        s0List = s0.split(" ")
        s1List = s1.split(" ")
        return len(list(set(s0List)&set(s1List)))
ob = Solution()

k = 0
for i in range(All.shape[0]):
    if All["Cabin_dumb"][i]!=0:
        match_size =[] 
        for j in range(All.shape[0]):
            match_size =[] 
            match = ob.solve(All['Name_new'][i],All['Name_new'][j])
            match_size.append(match)
            if (max(match_size)>0) & (All["Cabin_dumb"][j]==0) &(All['Pclass'][i]==All['Pclass'][j])&(All['Family'][i]>0)&(All['Family'][i]==All['Family'][j])&(i!=j):
                print(i,All['Name_new'][i],All['Family'][i],'-------' ,j,All['Name_new'][j],All['Family'][j])
                k=k+1
                
print(k)


# At last, By checking these 39 possibilities, we understand that none of them are in the same family.
# 
# # Result1:
# **Filling nan values in Cabin(Deck) column by using name and family values, is not helpful.**

# ## 2. Filling nan value in Cabin by Machine learning 

# You may tempt to consider the Cabin column as a label, train the dataset to learn the relation between other features and the Cabin, and finally predict the nan values of the Cabin by your Machine Learning model.
# 
# I tried the same and the result was terrible: 53% accuracy for sumbisssion.
# 
# This could be because of my workflow (data enginnering and modelling), but I think it will at least prevent you from wasting time doing the same thing.
# 
# In the following, we have tried to prepare data and test different models.
# 

# ### Filling missing values

# In[9]:


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


# In[10]:


age_by_pclass_sex = round(All.groupby(['Sex', 'Pclass']).mean()['Age'])

for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Mean age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
print('Mean age of all passengers: {}'.format(round(All['Age'].mean())))

# Filling the missing values in Age with the medians of Sex and Pclass groups
All['Age'] = All.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(round(x.mean())))


# In[11]:


All[All['Embarked'].isnull()]
All['Embarked'] = All['Embarked'].fillna('S')


# In[12]:


All[All['Fare'].isnull()]
mean_fare = All.groupby(['Pclass', 'Parch', 'SibSp']).Fare.mean()[3][0][0]
# Filling the missing value in Fare with the median Fare of 3rd class alone passenger
All['Fare'] = All['Fare'].fillna(mean_fare)


# In[13]:


All['Ticket_Frequency'] = All.groupby('Ticket')['Ticket'].transform('count')


# In[14]:


freq = All.head(891)['Ticket_Frequency'].value_counts().tolist()
Ticket_freq = All.head(891)['Ticket_Frequency'].unique().tolist()

death = []
for n in Ticket_freq:
    k = 0
    for i in range(891):
        if (All.head(891)['Ticket_Frequency'][i] == n) & (All.head(891)['Survived'][i] == 0):
            k = k+1
    death.append(k)    
     
survive_rate = []
for j,w in zip(death,freq):
    rate = (w-j)/w
    survive_rate.append(rate)

Survive_rate_index = {}
for u,r in zip(Ticket_freq,survive_rate):
    Survive_rate_index[u] = r
Survive_rate_index


# In[15]:


new_ticket_freq = []
for i in range(All.shape[0]):
    new_ticket_freq.append(Survive_rate_index[All['Ticket_Frequency'][i]])
new_ticket_freq = pd.DataFrame(new_ticket_freq)
All['Ticket_Frequency'] = pd.DataFrame(new_ticket_freq)


# In[16]:


All['Ticket_class1'] = All['Ticket']
All['Ticket_class2'] = All['Ticket']
All['Ticket_class3'] = All['Ticket']

for i in range(All.shape[0]):
    if All.Pclass[i]==1:
        All['Ticket_class1'][i]=All['Ticket'][i]
    else:
        All['Ticket_class1'][i]=0
    if All.Pclass[i]==2:
        All['Ticket_class2']=All['Ticket'][i]
    else:
        All['Ticket_class2'][i]=0
    if All.Pclass[i]==3:
        All['Ticket_class3'][i]=All['Ticket'][i]
    else:
        All['Ticket_class3'][i]=0
All['Ticket_class1'] = pd.factorize(All['Ticket_class1'])[0]
All['Ticket_class2'] = pd.factorize(All['Ticket_class2'])[0]
All['Ticket_class3'] = pd.factorize(All['Ticket_class3'])[0]


# In[17]:


for name in All["Name"]:
    All["Title"] = All["Name"].str.extract("([A-Za-z]+)\.",expand=True)

title_replacements = {"Mlle": "Other", "Major": "Other", "Col": "Other", "Sir": "Other", "Don": "Other", "Mme": "Other",
          "Jonkheer": "Other", "Lady": "Other", "Capt": "Other", "Countess": "Other", "Dona": "Other"
                     ,"Dr":"Other","Rev":"Other", "Mrs":"Woman","Ms":"Woman","Miss":"Woman"}

All.replace({"Title": title_replacements}, inplace=True)


# In[18]:


All = pd.get_dummies(All, prefix=['Pclass', 'Sex','Embarked','Title'], columns=['Pclass', 'Sex','Embarked','Title'])


# In[19]:


All[["Age", "Fare"]] = MinMaxScaler().fit_transform(All[["Age", "Fare"]])
All['Cabin_Dumb'] = pd.factorize(All['Cabin_dumb'])[0]
All.drop(['Name','Cabin_dumb','Cabin','Ticket','Name','Name_new','PassengerId'], axis=1, inplace=True)
All_org = All.copy()
All.drop(['Survived'], axis=1, inplace=True)


# In[20]:


df_known_cabin = pd.DataFrame()
for i in range(All.shape[0]):
    if All['Cabin_Dumb'][i]!=0:
        df_known_cabin = df_known_cabin.append(pd.DataFrame(All.values[i,:]).T)

df_unknown_cabin = pd.DataFrame()
for i in range(All.shape[0]):
    if All['Cabin_Dumb'][i]==0:
        df_unknown_cabin = df_unknown_cabin.append(pd.DataFrame(All.values[i,:-1]).T)


# ### Models

# In[21]:


X = df_known_cabin.values[:, :-1]
Y = df_known_cabin.values[:,-1]
Y=Y.astype('int')

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('Extra', ExtraTreesClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10)
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# In[22]:


dtc=DecisionTreeClassifier()
dtc.fit(X,Y)
predict_cabin=dtc.predict(df_unknown_cabin.values)
predict_cabin


# In[23]:


count = Counter(predict_cabin)
cabin_distribution = pd.DataFrame.from_dict(count, orient='index')
cabin_distribution.plot(kind='bar')


# In[24]:


df_known_cabin_final = pd.DataFrame()
for i in range(All.shape[0]):
    if All_org['Cabin_Dumb'][i]!=0:
        df_known_cabin_final = df_known_cabin_final.append(pd.DataFrame(All_org.values[i,:]).T)

df_unknown_cabin_ = pd.DataFrame()
for i in range(All.shape[0]):
    if All_org['Cabin_Dumb'][i]==0:
        df_unknown_cabin_ = df_unknown_cabin_.append(pd.DataFrame(All_org.values[i,:-1]).T)


# In[25]:


predict_cabin = pd.DataFrame(predict_cabin)
df_unknown_cabin_[22] = df_unknown_cabin_[1]
for i in range(df_unknown_cabin_.shape[0]):
    df_unknown_cabin_[22].values[i] = predict_cabin[0][i]


# In[26]:


All_with_cabin = pd.concat([df_unknown_cabin_, df_known_cabin_final],ignore_index=True)

train_cabin = pd.DataFrame()
for i in range(All.shape[0]):
    if pd.isnull(All_with_cabin[4][i])== False:
        train_cabin = train_cabin.append(pd.DataFrame(All_with_cabin.values[i,:]).T)

test_cabin = pd.DataFrame()
for i in range(All.shape[0]):
    if pd.isnull(All_with_cabin[4][i])== True:
        test_cabin = test_cabin.append(pd.DataFrame(All_with_cabin.values[i,:]).T)

All_with_cabin_final = pd.concat([train_cabin, test_cabin],ignore_index=True)
All_with_cabin_final.head()


# In[27]:


All_with_cabin_final.insert(23, "survived", All_with_cabin_final[4])
All_with_cabin_final.drop(4, axis=1, inplace=True)
All_with_cabin_final


# Now if we test some Machine Learning models on the train part of the above dataset (All_with_cabin_final.head(891)), we obtain an overfit model that result in a very poor submission.

# # Result 2:
# 
# **Filling nan values in the Cabin column by predicting them using machine learning model trained with Cabin's known values, is not helpful.**

# In[ ]:




