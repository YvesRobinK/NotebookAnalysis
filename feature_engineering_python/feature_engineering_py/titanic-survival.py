#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier


# # Features

# Load the data and select only the features that are not redundant, don't have too many missing values and affect the output significantly.

# In[3]:


train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[4]:


train_data.info()


# PassengerId is just a serial number and Cabin has too many missing values. The name of a person only gives information about gender (through honorifics - Mr, Mrs, Miss, etc) and family size (through surname) however we already know both of these through other features (Sex and SibSp, Parch). Ticket only gives information about the group size of a person by counting the number of people with the same ticket, but in most cases it will be equal to family size.
# 
# The other features might affect the survivability significantly. Let's check:

# In[5]:


# Sex

women = train_data.loc[train_data['Sex'] == 'female']['Survived']
print("Women: ", sum(women)/len(women)*100)
men = train_data.loc[train_data['Sex'] == 'male']['Survived']
print("Men: ", sum(men)/len(men)*100)


# In[6]:


# Embarked

southampton = train_data.loc[train_data['Embarked'] == 'S']['Survived']
print("Southampton: ", sum(southampton)/len(southampton)*100)
cherbourg = train_data.loc[train_data['Embarked'] == 'C']['Survived']
print("Cherbourg: ", sum(cherbourg)/len(cherbourg)*100)
queenstown = train_data.loc[train_data['Embarked'] == 'Q']['Survived']
print("Queenstown: ", sum(queenstown)/len(queenstown)*100)


# In[7]:


# Pclass

class_1 = train_data.loc[train_data['Pclass'] == 1]['Survived']
print("Class 1: ", sum(class_1)/len(class_1)*100)
class_2 = train_data.loc[train_data['Pclass'] == 2]['Survived']
print("Class 2: ", sum(class_2)/len(class_2)*100)
class_3 = train_data.loc[train_data['Pclass'] == 3]['Survived']
print("Class 3: ", sum(class_3)/len(class_3)*100)


# In[8]:


# Age

young = train_data.loc[train_data['Age'] < 15]['Survived']
print("Young: ", sum(young)/len(young)*100)
middle = train_data.loc[(train_data['Age'] <= 60) & (train_data['Age'] >= 15)]['Survived']
print("Middle: ", sum(middle)/len(middle)*100)
old = train_data.loc[train_data['Age'] > 60]['Survived']
print("Old: ", sum(old)/len(old)*100)


# In[9]:


# SibSp

for num in set(train_data['SibSp'].unique()):
    survived = train_data.loc[train_data['SibSp'] == num]['Survived']
    print(f"{num}:", sum(survived)/len(survived)*100)


# In[10]:


# Parch

for num in set(train_data['Parch'].unique()):
    survived = train_data.loc[train_data['Parch'] == num]['Survived']
    print(f"{num}:", sum(survived)/len(survived)*100)


# In[11]:


# Fare

poor = train_data.loc[train_data['Fare'] <= 14]['Survived']
print("Poor: ", sum(poor)/len(poor)*100)
middle = train_data.loc[(train_data['Fare'] <= 31) & (train_data['Fare'] > 14)]['Survived']
print("Middle: ", sum(middle)/len(middle)*100)
rich = train_data.loc[train_data['Fare'] > 31]['Survived']
print("Rich: ", sum(rich)/len(rich)*100)


# As is clear, the features that need to be selected are Sex, Pclass, Age, SibSp, Parch and Fare. Embarked needs to be dropped.
# 
# Sex: 74% of females survived while only 19% males survived. This makes sense - women are given higher priority in such situations.
# 
# Pclass: 63% Class 1 passengers survived and the survivability went down with Class 2 and 3. This makes sense - the affluent are given higher priority in such situations.
# 
# Age: 58% children (under 15) survived with survivability decreasing with increase in age. This makes sense - children are given higher priorit in such situations.
# 
# SibSp & Parch: Survivability decreased with increasing family size. This makes sense - people with large families will prioritise their family members, especially children.
# 
# Fare: The rich (people who bought more expensive tickets) had better survivability. Again, this correlates with Pclass.
# 
# Embarked: The nationality of an individual - English, French or Kiwi (New Zealander) - doesn't seem to affect their survivability significantly.

# And that's all for the features. Don't try to make this too complicated with sophisticated feature engineering - this problem is one of the simplest on this site and is meant to be dealt with in a simple manner. Feature engineering works but a lot of it increases the score only a little and may even decrease the score if it isn't done properly.

# In[12]:


train_data = train_data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
test_data = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]


# # Preprocessing

# Split into x_train and y_train, and intialize the preprocessors. The categorical feature - Sex - is One Hot Encoded and one of them is dropped to prevent linear dependence of features (by looking at whether someone is male or not, you can determine whether he/she is female or not - it doesn't give any extra information). The feature with missing values - Age - is imputed in a principled manner with Iterative Imputation. Feature scaling isn't required since I will be using a tree based method - Random Forest.
# 
# Its always a good idea to do all preprocessing through a pipeline - it reduces the amount of code, makes it easier to understand and also provides the option of tuning hyperparameters of all preprocessors and estimator at once.
# 
# Once again, don't try to make this too complicated. No need to remove outliers, transform features to uniform/normal distribution or handle class rebalancing with oversampling. Such things make the dataset too complicated to make a good estimation.

# In[13]:


x_train = train_data.iloc[:, 1:]
y_train = train_data.iloc[:, 0]
x_test = test_data.iloc[:, :]
encoder = make_column_transformer((OneHotEncoder(drop='first', sparse=False), x_train.select_dtypes(include='object').columns), remainder='passthrough', verbose_feature_names_out=False)
pipe = make_pipeline(encoder, IterativeImputer(random_state=42))


# In[14]:


x_train = pd.DataFrame(pipe.fit_transform(x_train), columns=encoder.get_feature_names_out())
x_test = pd.DataFrame(pipe.transform(x_test), columns=encoder.get_feature_names_out())


# In[15]:


x_train.info()


# # Estimation

# As has been said before, keep it simple. Don't use something like XGBoost or Neural Networks (Multi-layer Perceptron). Something simple like Random Forest or SVC will be more than enough. The tutorial used Random Forest and that's what I will be using too.
# 
# First of all, do some hyperparameter tuning with GridSearchCV and use the best estimator.

# In[16]:


# rf = RandomForestClassifier(random_state=1)
# params = {'n_estimators': [50, 100, 150, 200],
#           'max_depth': [2, 3, 4, 5]}
# search = GridSearchCV(rf, params)
# search.fit(x_train, y_train)
# search.best_params_

# max_depth = 5, n_estimators = 150


# In[17]:


# rf = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=1)
# rf.fit(x_train, y_train)
# y_pred = rf.predict(x_test)

# score = 0.78468


# Unfortunately, the above 'best' estimator didn't give the best results. The best results were actually obtained by the estimator used in the tutorial. This is because of the difference in training and evaluation sets.

# In[18]:


rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

# score = 0.78708


# The score of 0.799 was obtained in a crude manner - during preprocessing, both features for Sex_male and Sex_female were retained and missing values were all imputed with -1. It gave the best score but didn't make statistical sense.

# # Submission

# Load the sample submission and build your submission file in the same format.

# In[19]:


pd.read_csv('/kaggle/input/titanic/gender_submission.csv')


# In[20]:


df = pd.DataFrame()
df['PassengerId'] = range(892, 1310)
df['Survived'] = y_pred
df.to_csv('submission.csv', index=False)

