#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# A serious problem arising in the early stages of development of oil and gas fields is the insufficient amount of information about the parameters of these fields. Incorrectly estimated parameters threaten with large errors in assessing the underlying volumes of oil and gas, which brings economic losses to the company that develops the field.
# 
# A particularly important parameter is the location of the oil field.

# ## If you find this notebook helpful, please Upvote :) 
# This motivates to add more quality work.

# # Loading Imp Libraries

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Reading Dataset

# In[2]:


train=pd.read_csv('/kaggle/input/lassification-type-of-oil-field/train_final.csv')
test=pd.read_csv('/kaggle/input/lassification-type-of-oil-field/test_final.csv')


# In[3]:


train.head()


# In[4]:


train['Onshore/Offshore'].value_counts()


# In[5]:


# ONSHORE-OFFSHORE category has only 5 example, adding one more by duplicating to use SMOTE later on.
# This increases score from .81 -> .84
# Please continue to read below to follow !
train=pd.concat([train,train[train['Onshore/Offshore']=='ONSHORE-OFFSHORE'].head(1)])


# In[6]:


train.info()


# In[7]:


train.describe()


# In[8]:


for i in train.columns:
    print(i,train[i].nunique())


# # Feature Engineering

# In[9]:


from nltk import flatten
# Unique Techtonic Regime
unique_tect_reg=train['Tectonic regime'].values
unique_tect_reg=[i.split('/') for i in unique_tect_reg]
unique_tect_reg=list(set(flatten(unique_tect_reg)))

#Unique structural setting
unique_str_set=train['Structural setting'].values
unique_str_set=[i.split('/') for i in unique_str_set]
unique_str_set=list(set(flatten(unique_str_set)))

# Period
#Unique structural setting
unique_per=train['Period'].values
unique_per=np.append(unique_per,test['Period'].values)
unique_per=[i.split('-') for i in unique_per]
unique_per=list(set(flatten(unique_per)))

# Lith
#Unique Lithology setting
unique_lit=train['Lithology'].values
unique_lit=np.append(unique_lit,test['Lithology'].values)
unique_lit=list(set(unique_lit))



# In[10]:


lith_dict = dict(zip(unique_lit, range(len(unique_lit))))
lith_dict


# In[11]:


unique_tect_reg



# In[12]:


unique_str_set


# In[13]:


unique_per


# Adding dummy columns for every level in Tectonic regime,Structural setting, Period

# In[14]:


for i in unique_tect_reg:
    train[i]=train["Tectonic regime"].str.contains(i).astype(int)
    test[i]=test["Tectonic regime"].str.contains(i).astype(int)
for i in unique_str_set:
    train[i]=train["Structural setting"].str.contains(i).astype(int)
    test[i]=test["Structural setting"].str.contains(i).astype(int)
for i in unique_per:
    train[i]=train["Period"].str.contains(i).astype(int)
    test[i]=test["Period"].str.contains(i).astype(int)


# Label encoding other categorical columns

# In[15]:


from sklearn import preprocessing
 
for i in ['Hydrocarbon type','Reservoir status']:
# label_encoder object knows how to understand word labels.
    label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
    train[i]= label_encoder.fit_transform(train[i])
    test[i]= label_encoder.transform(test[i])


# In[16]:


train['Lithology'] = train['Lithology'].map(lith_dict)
test['Lithology'] = test['Lithology'].map(lith_dict)


# In[17]:


y_train=train['Onshore/Offshore']
target2int = {'ONSHORE': 0, 'OFFSHORE': 1, 'ONSHORE-OFFSHORE': 2}
# encode target
y_train = y_train.map(target2int)
train=train.drop(['Tectonic regime','Structural setting','Onshore/Offshore','Period'],axis=1)
test=test.drop(['Tectonic regime','Structural setting','Period'],axis=1)


# # Smote Upsampling

# In[18]:


from imblearn.over_sampling import SMOTE
oversample = SMOTE()
train, y_train = oversample.fit_resample(train,y_train)
y_train.value_counts()


# # Model Selection

# In[19]:


# Model Selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[20]:


from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid=train_test_split(train,y_train,test_size=0.1, random_state=42)


# In[21]:


#Testing
clf = LogisticRegression()
clf.fit(X_train, y_train)
clf.score(X_valid, y_valid)


# In[22]:


#Testing
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
clf.score(X_valid, y_valid)


# In[23]:


#Testing
clf = SVC()
clf.fit(X_train, y_train)
clf.score(X_valid, y_valid)


# In[24]:


clf = RandomForestClassifier()
clf.fit(X_train, y_train)
clf.score(X_valid, y_valid)


# In[25]:


from sklearn.model_selection import cross_val_score
for i in [LogisticRegression(),RandomForestClassifier(),DecisionTreeClassifier(),SVC()]:
    score_lr=cross_val_score(i, X_train, y_train,cv=3)
    print(score_lr)
    print("Avg :",i,np.average(score_lr))


# In[26]:


from lightgbm import LGBMClassifier

## Grid definition for model selection

classifiers = {
    "LogisticRegression" : LogisticRegression(random_state=0),
    "RandomForest" : RandomForestClassifier(random_state=0),
    "DecisionTree" : DecisionTreeClassifier(random_state=0),
   # "SVC" : SVC(random_state=0, verbose=False),
     "LGBM" : LGBMClassifier(random_state=0)
}

# Grids for grid search
LR_grid = {'penalty': ['l1','l2'],
           'C': [0.25, 0.5, 0.75, 1, 1.25, 1.5],
           'max_iter': [50, 100, 150]}



SVC_grid = {'C': [0.25, 0.5, 0.75, 1, 1.25, 1.5],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']}

RF_grid = {'n_estimators': [50, 100, 150, 200, 250, 300],
        'max_depth': [4, 6, 8, 10, 12]}

tree_grid = {'max_depth': [4, 8, 12]}

boosted_grid = {'n_estimators': [50, 100, 150, 200],
        'max_depth': [4, 8, 12],
        'learning_rate': [0.05, 0.1, 0.15]}



# Dictionary of all grids
grid = {
    "LogisticRegression" : LR_grid,
    "RandomForest" : RF_grid,
    "DecisionTree" : tree_grid,
    "SVC" : SVC_grid,
    'LGBM':boosted_grid
}


# In[27]:


from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
# GridSearchCV for model selection in Action !
import time
i=0
clf_best_params=classifiers.copy()
valid_scores=pd.DataFrame({'Classifer':classifiers.keys(), 'Validation accuracy': np.zeros(len(classifiers)), 'Training time': np.zeros(len(classifiers))})
for key, classifier in classifiers.items():
    start = time.time()
    clf = GridSearchCV(estimator=classifier, param_grid=grid[key], n_jobs=-1, cv=None)

    # Train and score
    clf.fit(X_train, y_train)
    valid_scores.iloc[i,1]=clf.score(X_valid, y_valid)

    # Save trained model
    clf_best_params[key]=clf.best_params_
    
    # Print iteration and training time
    stop = time.time()
    valid_scores.iloc[i,2]=np.round((stop - start)/60, 2)
    
    print('Model:', key)
    print('Training time (mins):', valid_scores.iloc[i,2])
    print('')
    i+=1


# In[28]:


valid_scores


# In[29]:


clf_best_params


# In[30]:


# Using LGBM
clf=LGBMClassifier(**clf_best_params['LGBM']).fit(X_train,y_train)


# In[31]:


cv_acc = cross_val_score(clf,
                         X_train,
                         y_train,
                         cv=10,
                         scoring="accuracy")
np.mean(cv_acc)


# In[32]:


sub=clf.predict(test)


# In[33]:


ans=pd.DataFrame(range(len(sub)),columns=['index'])


# In[34]:


ans['Onshore/Offshore']=sub


# In[35]:


ans


# In[36]:


target2int = {0:'ONSHORE',1: 'OFFSHORE', 2:'ONSHORE-OFFSHORE'}
# encode target
ans['Onshore/Offshore'] = ans['Onshore/Offshore'].map(target2int)


# # Submission

# In[37]:


ans.to_csv('submission.csv',index=False)


# In[ ]:





# In[ ]:




