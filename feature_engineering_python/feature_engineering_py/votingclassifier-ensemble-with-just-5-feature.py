#!/usr/bin/env python
# coding: utf-8

# In this notebook I reuse some codes from below notebooks:
# 
# * [https://www.kaggle.com/code/gunesevitan/titanic-advanced-feature-engineering-tutorial](http://)
# * [https://www.kaggle.com/code/khashayarrahimi94/what-not-to-do-in-titanic-feature-engineering](http://)

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
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')


# In[2]:


train = pd.read_csv(r'../input/titanic/train.csv')
test =  pd.read_csv(r'../input/titanic/test.csv')


# In[3]:


All = pd.concat([train, test], sort=True).reset_index(drop=True)
All


# # Exploratory Data Analysis

# Get the number of nan values of each feature:

# In[4]:


columns = test.columns
for i in range(len(columns)):
    print(columns[i],'--->',"train:",train[columns[i]].isnull().sum(),
         "|","test:",test[columns[i]].isnull().sum()) 


# Create a new column named "Cabin_dumb" which replace the Cabin ID with their Decks, and nan values with zero:

# In[5]:


All["Cabin_dumb"]=All["Cabin"]
for i in range(All.shape[0]):
    if pd.isnull(All["Cabin"][i])== False:
        All["Cabin_dumb"][i] = All["Cabin"][i][0]
    else:
        All["Cabin_dumb"][i] =0


All["Cabin_dumb"]


# Create a new column named "Family" which summarizes the number of family members of each passegener:

# In[6]:


All['Family'] = All['SibSp']+All['Parch']


# As we saw above, "Age" column has 263 nan values and we need to fill them. Here we use mutual information between Age column and the four other features, to choose the one has the most value.

# In[7]:


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


# Filling the missing values in Age with the medians of Sex and Pclass groups:

# In[8]:


age_by_pclass_sex = round(All.groupby(['Sex', 'Pclass']).median()['Age'])

for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Mean age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
print('Mean age of all passengers: {}'.format(round(All['Age'].mean())))

All['Age'] = All.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(round(x.median())))


# In[9]:


All[All['Embarked'].isnull()]
All['Embarked'] = All['Embarked'].fillna('S')


# In[10]:


All[All['Fare'].isnull()]
mean_fare = All.groupby(['Pclass', 'Parch', 'SibSp']).Fare.mean()[3][0][0]
# Filling the missing value in Fare with the median Fare of 3rd class alone passenger
All['Fare'] = All['Fare'].fillna(mean_fare)


# In[11]:


All['Ticket_Frequency'] = All.groupby('Ticket')['Ticket'].transform('count')
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


# In[12]:


new_ticket_freq = []
for i in range(All.shape[0]):
    new_ticket_freq.append(Survive_rate_index[All['Ticket_Frequency'][i]])
new_ticket_freq = pd.DataFrame(new_ticket_freq)
All['Ticket_Frequency'] = pd.DataFrame(new_ticket_freq)


# In[13]:


for name in All["Name"]:
    All["Title"] = All["Name"].str.extract("([A-Za-z]+)\.",expand=True)

title_replacements = {"Mlle": "Other", "Major": "Other", "Col": "Other", "Sir": "Other", "Don": "Other", "Mme": "Other",
          "Jonkheer": "Other", "Lady": "Other", "Capt": "Other", "Countess": "Other", "Dona": "Other"
                     ,"Dr":"Other","Rev":"Other", "Mrs":"Woman","Ms":"Woman","Miss":"Woman"}

All.replace({"Title": title_replacements}, inplace=True)


# In[14]:


All['Title'] = pd.factorize(All['Title'])[0]
All['Cabin_dumb'] = pd.factorize(All['Cabin_dumb'])[0]
All['Sex'] = pd.factorize(All['Sex'])[0]
All['Embarked'] = pd.factorize(All['Embarked'])[0]
All.head()


# In[15]:


All.drop(['Ticket','Cabin','Name','PassengerId'], axis=1, inplace=True)


# In[16]:


All[["Age", "Fare"]] = MinMaxScaler().fit_transform(All[["Age", "Fare"]]) 


# ## Correlation Table

# In[17]:


All.head(891).corr(method ='pearson')


# In[18]:


corr = All.head(891).corr()
corr.style.background_gradient(cmap='coolwarm')


# In[19]:


All.insert(12, "survived", All['Survived'])
All.drop(['Survived'], axis=1, inplace=True)
All.head()


# ## Choose 5 Feature

# After checking the correlation table, we just consider the feature which have corr > 0.25 and delete other columns.

# In[20]:


All.drop(['SibSp','Age','Parch','SibSp','Ticket_Frequency','Family','Embarked'], axis=1, inplace=True)
All.head()


# In[21]:


All = pd.get_dummies(All, columns=['Pclass', 'Sex','Cabin_dumb','Title'], prefix=['Pclass', 'Sex','Cabin_dumb','Title'])
All.head()


# In[22]:


All.insert(20, "Survived", All['survived'])
All.drop(['survived'], axis=1, inplace=True)
All.head()


# # Models

# In[23]:


train = All.head(891)
X = train.values[:, :-1]
Y = train.values[:,-1]
Y=Y.astype('int')

# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('Extra', ExtraTreesClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
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
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# ## KNN-XGB-SVC Ensemble

# In[24]:


X = train.values[:,:-1]
Y = train.values[:,-1]
label_encoded_y = LabelEncoder().fit_transform(Y)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
# create the sub models
estimators = []
model1 = KNeighborsClassifier(leaf_size=100,algorithm='auto',metric = 'minkowski', n_neighbors = 15,p=12, weights = 'distance')
estimators.append(('KNeighborsClassifier', model1))

model2 = XGBClassifier(learning_rate = 0.39, max_depth = 2, n_estimators = 53)
estimators.append(('XGBClassifier', model2))
model3 = SVC(C = 89, gamma = 1.1, kernel = 'rbf')
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble, X, label_encoded_y, cv=kfold)
print(results.mean() ,results.std())


# In[25]:


test = All.tail(418)
test.drop('Survived', axis=1, inplace=True)
test.head()


# In[26]:


ensemble.fit(X, label_encoded_y)
predict_ensemble = ensemble.predict(test)
Submission = pd.DataFrame({'PassengerId':list(range(892,1310))})
Submission['Survived']=predict_ensemble
Submission


# In[27]:


Submission.to_csv('submission.csv', index=False)

