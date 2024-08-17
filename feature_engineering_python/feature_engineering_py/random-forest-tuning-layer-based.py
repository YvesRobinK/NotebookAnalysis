#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# My focus in this notebook is on the Random Forest Classifier hyperparameter tuning and not on exploratory data analysis.
# 
# However, since we need a prepare and clean dataset for tuning, the first part of the notebook dedicated to data engineering on the Titanic dataset.
# 
# For explanation and details of methods used in the data cleaning part, you can visit these notebooks:
# 
# 1. Meditation on the Cabin
# 
# https://www.kaggle.com/code/khashayarrahimi94/meditation-on-the-cabin
# 
# 2. What NOT TO DO in Titanic(Feature Engineering)
# 
# https://www.kaggle.com/code/khashayarrahimi94/what-not-to-do-in-titanic-feature-engineering
# 
# 3. How divergence the train & test distributions are?
# 
# https://www.kaggle.com/code/khashayarrahimi94/how-divergence-the-train-test-distributions-are

# In[1]:


import numpy as np 
import pandas as pd
from Levenshtein import distance as lev
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import warnings
import re
warnings.filterwarnings('ignore')


# # Feature Engineering

# In[2]:


train = pd.read_csv(r'../input/titanic/train.csv')
test =  pd.read_csv(r'../input/titanic/test.csv')
All = pd.concat([train, test], sort=True).reset_index(drop=True)
All


# In[3]:


All[All['Fare'].isnull()]
mean_fare = All.groupby(['Pclass', 'Parch', 'SibSp']).Fare.mean()[3][0][0]
# Filling the missing value in Fare with the median Fare of 3rd class alone passenger
All['Fare'] = All['Fare'].fillna(mean_fare)


# In[4]:


All[All['Embarked'].isnull()]
All['Embarked'] = All['Embarked'].fillna('S')


# In[5]:


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


# In[6]:


age_by_pclass_sex = round(All.groupby(['Sex', 'Pclass']).median()['Age'])

for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Mean age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
print('Mean age of all passengers: {}'.format(round(All['Age'].mean())))

All['Age'] = All.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(round(x.median())))


# In[7]:


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


# In[8]:


All['Cabin'].isnull().sum()


# In[9]:


for name in All["Name"]:
    All["Title"] = All["Name"].str.extract("([A-Za-z]+)\.",expand=True)

title_replacements = {"Mlle": "Other", "Major": "Other", "Col": "Other", "Sir": "Other", "Don": "Other", "Mme": "Other",
          "Jonkheer": "Other", "Lady": "Other", "Capt": "Other", "Countess": "Other", "Dona": "Other"
                     ,"Dr":"Other","Rev":"Other"}

All.replace({"Title": title_replacements}, inplace=True)


# In[10]:


All[["Age", "Fare"]] = StandardScaler().fit_transform(All[["Age", "Fare"]])


# In[11]:


All["Deck"]=All["Cabin"]
for i in range(All.shape[0]):
    All["Deck"][i] = All["Cabin"][i][0]

All.drop(['Cabin'], axis=1, inplace=True)
All


# In[12]:


All['Title'] = pd.factorize(All['Title'])[0]
All['Deck'] = pd.factorize(All['Deck'])[0]
All['Sex'] = pd.factorize(All['Sex'])[0]
All['Embarked'] = pd.factorize(All['Embarked'])[0]


# In[13]:


All.drop(['Name','PassengerId','Ticket'], axis=1, inplace=True)
All.insert(10, "survived", All['Survived'])
All.drop(['Survived'], axis=1, inplace=True)
All


# In[14]:


train = All.head(891)
test = All.tail(418)
test.drop(['survived'], axis=1, inplace=True)


# # Random Forest tuning (Layer Based)
# 
# The importance of Machine Learning models hyperparameter tuning is clear to all practitioners, and often is not a challengeable part of data science.
# Here I just want to share my approach for ML models tuning and in this case Random Forest tuning.
# 
# ## What is Layer based tuning?
# 
# In fact,this is not a formal concept and it was just named by me. Also, most likely many practitioners do the same.
# 
# Here I mean modifying the hyperparameters after a best result in the pervious layer obtained until we achieve the best values for hyperparameters.
# 
# In other words, first we start with different values for hyperparameters and after running we obtain a best. In next step or layer, we modify the values in the **neighbourhood** of the pervious layer and repeat this process until achieving specific value for each hyperparameters.
# 
# We can combine all of this pocess in one cell or by defining a function. I hope to do it soon and share it here.

# In[15]:


def estimator(param):
    #param={}
    #param = grid_result.best_params_

    best_n_estimators = param.get('n_estimators')
    
    if param_grid[0].get('n_estimators')[1] - param_grid[0].get('n_estimators')[0] == 100:
        n_estimators = [*range(best_n_estimators-90, best_n_estimators+90, 10)]  
        
    if param_grid[0].get('n_estimators')[1] - param_grid[0].get('n_estimators')[0] == 10:
         n_estimators = [*range(best_n_estimators-9, best_n_estimators+9, 1)]
        
    return n_estimators


def depth(param):
    #param={}
    #param = grid_result.best_params_

    best_max_depth = param.get('max_depth')
    
    if param_grid[0].get('max_depth')[1] - param_grid[0].get('max_depth')[0] == 2:
        max_depth = [*range(best_max_depth-2, best_max_depth+2, 1)]
        
    if param_grid[0].get('max_depth')[1] - param_grid[0].get('max_depth')[0] == 1:
        max_depth = [*range(best_max_depth-1, best_max_depth+1, 1)]
        
    return max_depth


def leaf_nodes(param):
    #param = grid_result.best_params_

    best_leaf_nodes = param.get('max_leaf_nodes')
    
    if param_grid[0].get('max_leaf_nodes')[1] - param_grid[0].get('max_leaf_nodes')[0] == 5:
        max_leaf_nodes = [*range(best_leaf_nodes-5, best_leaf_nodes+5, 1)]
    if param_grid[0].get('max_leaf_nodes')[1] - param_grid[0].get('max_leaf_nodes')[0] == 1:
        max_leaf_nodes = [best_leaf_nodes]
        
    return max_leaf_nodes


# In[16]:


X = train.values[:,:-1]
Y = train.values[:,-1]
label_encoded_y = LabelEncoder().fit_transform(Y)



# defining parameter range
param_grid = [
    {'n_estimators': [100,200,300], 
     'max_depth': [2, 4, 6, 8, 10, 12], 
     'max_leaf_nodes': [10, 15, 20, 25]}, 
]

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, scoring="accuracy", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X, label_encoded_y)
#summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

print('n_estimators =',param_grid[0].get('n_estimators'))
print('max_depth =',param_grid[0].get('max_depth'))
print('max_leaf_nodes =',param_grid[0].get('max_leaf_nodes'))

#Uncomment these two line to see the result for all possible combination of hyperparameters

#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']



#for mean, stdev, param in zip(means, stds, params):     
#   print("%f (%f) with: %r" % (mean, stdev, param))


# In[17]:


param = grid_result.best_params_

# defining parameter range
param_grid = [
    {'n_estimators': estimator(param), 
     'max_depth': depth(param), 
     'max_leaf_nodes': leaf_nodes(param)}, 
]

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, scoring="accuracy", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X, label_encoded_y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

print('n_estimators =',param_grid[0].get('n_estimators'))
print('max_depth =',param_grid[0].get('max_depth'))
print('max_leaf_nodes =',param_grid[0].get('max_leaf_nodes'))

#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']


#for mean, stdev, param in zip(means, stds, params):     
#   print("%f (%f) with: %r" % (mean, stdev, param))


# In[18]:


param = grid_result.best_params_
# defining parameter range
param_grid = [
    {'n_estimators': estimator(param), 
     'max_depth': depth(param), 
     'max_leaf_nodes': leaf_nodes(param)}, 
]

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, scoring="accuracy", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X, label_encoded_y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

print('n_estimators =',param_grid[0].get('n_estimators'))
print('max_depth =',param_grid[0].get('max_depth'))
print('max_leaf_nodes =',param_grid[0].get('max_leaf_nodes'))

#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']


#for mean, stdev, param in zip(means, stds, params):     
#   print("%f (%f) with: %r" % (mean, stdev, param))


# In[19]:


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

RFC = RandomForestClassifier(max_depth=grid_result.best_params_.get('max_depth'),
                            max_leaf_nodes= grid_result.best_params_.get('max_leaf_nodes'),
                            n_estimators= grid_result.best_params_.get('n_estimators')
)

results = cross_val_score(RFC, X, label_encoded_y, cv=kfold)
print(results.mean() ,results.std())


# # Hyperparameters for score: 0.79186
# 
# The hyperparameters may changes due to random_state or other things, However when I run this code I obtain the below values for hyperparameters:
# 
# * max_depth = 6
# * max_leaf_nodes = 29
# * n_estimators = 112

# In[20]:


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7) 

RF = RandomForestClassifier(max_depth=6,max_leaf_nodes= 29,n_estimators=112)

results = cross_val_score(RF, X, label_encoded_y, cv=kfold)
print(results.mean() ,results.std())


# In[21]:


RFC.fit(X, label_encoded_y)
predict_RFC = RFC.predict(test)
Submission = pd.DataFrame({'PassengerId':list(range(892,1310))})
Submission['Survived']=predict_RFC
Submission


# In[22]:


Submission.to_csv('submission.csv', index=False)

