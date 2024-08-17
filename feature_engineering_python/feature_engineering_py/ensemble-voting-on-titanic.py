#!/usr/bin/env python
# coding: utf-8

# <div style="color:black;
#            display:fill;
#            border-radius:1px;
#            background-color:#87ceeb;
#            align:center;">
# 
# <p style="padding: 1px;
#               color:white;">
# <center>
#     
# # Ensemble Voting Classifier for Titanic
# 

# ## Table of Contents
# 
# * [1. Introduction to Notebook and Data Preparation](#1)
# * [2. EDA](#2)
# * [3. Ensemble Learning: Voting Classifier](#3)

# <div style="color:black;
#            display:fill;
#            border-radius:1px;
#            background-color:#e4f2f8;
#            align:center;">
# 
# <p style="padding: 1px;
#               color:white;">
# <center>
#     
# <a id="1"></a>
# ## 1. Introduction to Notebook and Data Preparation

# <p style="font-size:16px"> In this notebook, we will use ensemble voting method ( Linear regression, KNN, XGBoost and Random Forest) to predict survived column on the titanic dataset</p>

# ### Import relevant libraries

# In[1]:


import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;
import seaborn as sns;
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import numpy as np
scaler = StandardScaler()


# ### Define a function to clean the data
# 
# Here I take care of missing data in the dataset: 
# - Missing Age are replaced by Mean.
# - Missing Fare by minimum Fare.
# - Missing Embarked by mode value.

# In[2]:


def clean_data(df):
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True); # take care of missing Embarked values
    df['Age'] = df['Age'].fillna(df['Age'].mean()); # take care of missing Age values
    df['Fare'] = df['Fare'].fillna(df['Fare'].min()); # take care of missing Fare values
    df = df.drop(columns=['Cabin','Ticket','PassengerId'], axis=1); # drop unwanted columns that do not contribute to survivability
    return df


# ### Define a function to encode non-numerical data

# In[3]:


def encode_data(df):
    # quantize the age column
    df['Age'] = pd.cut(df['Age'],[0, 10, 20, 50, np.inf],labels=[1, 2, 3, 4]).astype(int);
    # encode non-numerical values
    df.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True);
    # scale
    scaler.fit(df[['Fare']]);
    df[['Fare']] = scaler.transform(df[['Fare']]);
    return df


# ### Add new columns by engineering the available features
# - Here I extract useful information from name column, namely passengers social status from their titles.
# - If title represents professional (Dr., Captain.) or nobility, then it is of class 1 else 0.
# - I also add new column representing family size
# - and a new column representing whether passenger was adult or not.

# In[4]:


regex = "([A-Za-z]+)\."

# I will keep Mrs. as 2 because, it signifies married or not.

title_status = {'Master.':0,
 'Mrs.':0,
 'Mr.':0,
 'Ms.':0,
 'Col.':1,
 'Mme.':0,
 'Countess.':1,
 'Mlle.':0,
 'Don.':1,
 'Lady.':0,
 'Miss.':0,
 'Dr.':1,
 'Sir.':0,
 'Capt.':1,
 'Rev.':1,
 'Major.':1,
 'Jonkheer.':0,
  'Dona.':1};

female_married_status = {'Master.':0,
 'Mrs.':1,
 'Mr.':0,
 'Ms.':0,
 'Col.':0,
 'Mme.':0,
 'Countess.':0,
 'Mlle.':0,
 'Don.':0,
 'Lady.':0,
 'Miss.':0,
 'Dr.':0,
 'Sir.':0,
 'Capt.':0,
 'Rev.':0,
 'Major.':0,
 'Jonkheer.':0,
  'Dona.':0};

def get_title(row):
    match = re.search(regex, str(row))
    title = match.group(0);
    return title

def feature_engineer(df):
    df['FamilySize'] = df['SibSp'] + df['Parch']; # add new feature of family size
    df['socialstatus'] = df.Name.apply(lambda x: get_title(x)); # add titles from name column
    df.replace({'socialstatus':title_status}, inplace=True); # replace titles by an assumed social status
    df['married'] = df.Name.apply(lambda x: get_title(x)); # add titles from name column
    df.replace({'married':female_married_status}, inplace=True); # replace titles by female marital status

    df = df.drop(columns=['Name'], axis=1); # once titles are extracted, drop the names
    # add a new column indicating adult or not
    adult = [];
    for i in range(len(df['Age'])):
        X = df['Age'].iloc[i];
        if(X>=2):
            adult.append(0);
        else:
            adult.append(1);
    df['adult'] = adult;
    df['Age_Class']= df['Age']* df['Pclass']; # new feature of Age times Pclass
    return df


# ### Import data and apply the data processing

# In[5]:


# import the data as pandas dataframe
test  = pd.read_csv('../input/titanic/test.csv');
train = pd.read_csv('../input/titanic/train.csv');
PID = test['PassengerId']; # save PID for competition submission

# process the data

train = clean_data(train);
test  = clean_data(test);

train = encode_data(train);
test  = encode_data(test);

train = feature_engineer(train);
test  = feature_engineer(test);


# <div style="color:black;
#            display:fill;
#            border-radius:1px;
#            background-color:#e4f2f8;
#            align:center;">
# 
# <p style="padding: 1px;
#               color:white;">
# <center>
#     
# <a id="2"></a>
# ## 2. EDA

# In[6]:


train.info()


# In[7]:


train.hist(bins=25,figsize=(9,7),grid=False);


# In[8]:


corr=train.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corr,annot=True,square=True,cmap='rocket')
plt.title('Correlation between features');


# ### Split the data into train and test subsets for model accuracy

# In[9]:


X = train.drop(columns = ['Survived'],axis=1);
y = train['Survived'];
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X,y, test_size=0.2, random_state=10);


# <div style="color:black;
#            display:fill;
#            border-radius:1px;
#            background-color:#e4f2f8;
#            align:center;">
# 
# <p style="padding: 1px;
#               color:white;">
# <center>
#     
# <a id="3"></a>
# ## 3. Ensemble Learning: Voting Classifier

# <p style="font-size:16px">
# We could take several classification model to create an ensemble model. Here we will use LinearRegression, KNN and RandomForest model to create an Ensemble Voting classifier.Each different models within Ensemble (here in this case LinearRegression, KNN, XGBoost and RandomForest) predicts different outcome from the training. Ensemble have lower error and lesser overfitting as compared to individual models. Since each individual models have different bias (or personality), the biases also gets averaged out in the ensemble.One of the Ensemble methods is the Voting classifier that combines the prediction of different models. We will use soft voting which averages the probability of predictions from different models.<\p>

# In[10]:


LogisticRegression(class_weight='balanced')
logistic_regression = LogisticRegression(max_iter=500,C=0.4,tol=1e-6,solver='liblinear');
random_forest = RandomForestClassifier(max_depth = 7,min_samples_split=2, n_estimators = 73);
knn           = KNeighborsClassifier(leaf_size=16, n_neighbors=18);
xgb = XGBClassifier(objective= 'binary:logistic', eval_metric="auc");
model = VotingClassifier (estimators=[('lr',logistic_regression), ('rf', random_forest), ('knn',knn) ,
                                      ('xgb',xgb)],voting='soft')
model.fit(X_train_m, y_train_m);
y_pred_m = model.predict(X_test_m);
from sklearn import metrics 
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test_m, y_pred_m));


# ### Submit the prediction to competition forum

# In[11]:


model.fit(X, y); # refit model on entire training set
y_submission = model.predict(test);
output = pd.DataFrame({'PassengerId': PID, 'Survived': y_submission})
output.to_csv('submission.csv', index=False)

