#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Importing and Studying Data

# In[2]:


train_data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')
combine = [train_data, test_data]


# In[3]:


train_data.info()


# In[4]:


train_data.head(10)


# In[5]:


test_data.head(10)


# In[6]:


print(train_data.shape)
print(test_data.shape)


# In[7]:


train_data.describe()


# In[8]:


women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women survivors:", rate_women)


# In[9]:


men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men survivors:", rate_men)


# # Checking Missing Values

# In[10]:


train_data.isnull().sum()


# # Dropping Cabin(too many missing values) and Ticket(no contribution to survival)

# In[11]:


for dataset in combine:
    dataset.drop(['Cabin','Ticket'],axis = 1,inplace = True)
    print(dataset.shape)


# # Creating a new feature 'Title' by extracting title from names

# In[12]:


for dataset in combine:
    dataset['Title'] = train_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_data['Title'], train_data['Sex'])


# # Replacing Titles with common ones or Rare and coverting them to ordinal values

# In[13]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


# In[14]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_data.head()


# # Dropping Name Column as we have extracted the Titles

# In[15]:


train_data.drop(['Name','PassengerId'],axis = 1,inplace = True)
test_data.drop(['Name'],axis = 1,inplace = True)
combine = [train_data, test_data]
print(train_data.shape)
print(test_data.shape)


# # Visualization and Analysis

# In[16]:


sns.countplot(x="Survived",data=train_data)


# In[17]:


sns.countplot(x="Survived", hue = 'Pclass',data=train_data)


# In[18]:


sns.countplot(x="Survived", hue = 'Sex',data=train_data)


# In[19]:


sns.countplot(x="Survived", hue = 'SibSp',data=train_data)


# In[20]:


sns.countplot(x="Survived", hue = 'Parch',data=train_data)


# In[21]:


sns.violinplot(x="Survived", y="Age", data = train_data, size = 9)


# In[22]:


sns.countplot(x="Survived",hue = "Embarked",data=train_data)


# # Handling missing values for Age and Embarked

# In[23]:


#filling missing values for 'Embarked' with most frequent one
freq_port = train_data.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[24]:


for dataset in combine:
    mean = train_data["Age"].mean()
    std = test_data["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_data["Age"].astype(int)


# In[25]:


train_data.isnull().sum()


# In[26]:


test_data.isnull().sum()


# In[27]:


# Filling the one missing value from Fare in test_data
test_data['Fare'].fillna(test_data['Fare'].dropna().median(), inplace=True)
test_data.isnull().sum()


# # Handling categorical variables

# In[28]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[29]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[30]:


train_data.head()


# # Creating new features FamilySize and IsAlone from Parch ana SibSp

# In[31]:


for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[32]:


for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1


# In[33]:


train_data.head()


# # Creating bands for age and Fare

# In[34]:


for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 36), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 50), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 50) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
train_data.head()


# In[35]:


train_data['FareBand'] = pd.qcut(train_data['Fare'], 4)
test_data['FareBand'] = pd.qcut(test_data['Fare'], 4)
train_data[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
test_data.head()


# In[36]:


for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_data = train_data.drop(['FareBand'], axis=1)
test_data = test_data.drop(['FareBand'], axis=1)
combine = [train_data, test_data]


# In[37]:


train_data.head()


# In[38]:


test_data.head()


# # Creating The Models and Checking individual performances

# In[39]:


X_train = train_data.drop(['Survived'], axis = 1).values
Y_train = train_data['Survived'].values
X_test  = test_data.drop("PassengerId", axis=1).copy()


# In[40]:


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)


# In[41]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_train,Y_train,test_size=0.2)


# ## Logistic Regression

# In[42]:


from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression()
lr_clf.fit(x_train, y_train)


# In[43]:


pred_train = lr_clf.predict(x_train)
pred_test = lr_clf.predict(x_test)


# In[44]:


from sklearn.metrics import accuracy_score
train_accuracy = accuracy_score(y_train,pred_train)
test_accuracy = accuracy_score(y_test,pred_test)
print("Training Accuracy: ", train_accuracy)
print("Testing Accuracy: ", test_accuracy)


# ## Random Forest Classifier

# In[45]:


from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_jobs= -1, n_estimators = 100, warm_start= True, max_depth= 5, min_samples_leaf= 2, max_features = 'sqrt',verbose = 0)
rf_clf.fit(x_train, y_train)


# In[46]:


pred_train = rf_clf.predict(x_train)
pred_test = rf_clf.predict(x_test)
train_accuracy = accuracy_score(y_train,pred_train)
test_accuracy = accuracy_score(y_test,pred_test)
print("Training Accuracy: ", train_accuracy)
print("Testing Accuracy: ", test_accuracy)


# ## Ada Boost Classifier

# In[47]:


from sklearn.ensemble import AdaBoostClassifier
adb_clf = AdaBoostClassifier(n_estimators = 100, learning_rate = 0.5)
adb_clf.fit(x_train, y_train)


# In[48]:


pred_train = adb_clf.predict(x_train)
pred_test = adb_clf.predict(x_test)
train_accuracy = accuracy_score(y_train,pred_train)
test_accuracy = accuracy_score(y_test,pred_test)
print("Training Accuracy: ", train_accuracy)
print("Testing Accuracy: ", test_accuracy)


# ## Gradient Boosting Classifier

# In[49]:


from sklearn.ensemble import GradientBoostingClassifier
gdb_clf = GradientBoostingClassifier(n_estimators = 100, max_depth = 3, min_samples_leaf = 2, verbose = 0)
gdb_clf.fit(x_train, y_train)


# In[50]:


pred_train = gdb_clf.predict(x_train)
pred_test = gdb_clf.predict(x_test)
train_accuracy = accuracy_score(y_train,pred_train)
test_accuracy = accuracy_score(y_test,pred_test)
print("Training Accuracy: ", train_accuracy)
print("Testing Accuracy: ", test_accuracy)


# ## Extra Trees Classifier

# In[51]:


from sklearn.ensemble import ExtraTreesClassifier
et_clf = ExtraTreesClassifier(n_jobs = -1, n_estimators = 100, max_depth = 5, min_samples_leaf = 2, verbose = 0)
et_clf.fit(x_train,y_train)


# In[52]:


pred_train = et_clf.predict(x_train)
pred_test = et_clf.predict(x_test)
train_accuracy = accuracy_score(y_train,pred_train)
test_accuracy = accuracy_score(y_test,pred_test)
print("Training Accuracy: ", train_accuracy)
print("Testing Accuracy: ", test_accuracy)


# ## Support Vector Classifier

# In[53]:


from sklearn.svm import SVC
svc_clf = SVC(kernel = 'linear', C = 0.025)
svc_clf.fit(x_train,y_train)


# In[54]:


pred_train = svc_clf.predict(x_train)
pred_test = svc_clf.predict(x_test)
train_accuracy = accuracy_score(y_train,pred_train)
test_accuracy = accuracy_score(y_test,pred_test)
print("Training Accuracy: ", train_accuracy)
print("Testing Accuracy: ", test_accuracy)


# ## XGBoost Classifier

# In[55]:


from xgboost import XGBClassifier
xgb_clf = XGBClassifier(n_estimators= 50, max_depth= 5, min_samples_leaf= 2)
xgb_clf.fit(x_train, y_train)


# In[56]:


pred_train = xgb_clf.predict(x_train)
pred_test = xgb_clf.predict(x_test)
train_accuracy = accuracy_score(y_train,pred_train)
test_accuracy = accuracy_score(y_test,pred_test)
print("Training Accuracy: ", train_accuracy)
print("Testing Accuracy: ", test_accuracy)


# ## LightGBM Classifier

# In[57]:


import lightgbm as lgb
lgb_clf = lgb.LGBMClassifier(n_estimators=100)
lgb_clf.fit(x_train,y_train)


# In[58]:


pred_train = lgb_clf.predict(x_train)
pred_test = lgb_clf.predict(x_test)
train_accuracy = accuracy_score(y_train,pred_train)
test_accuracy = accuracy_score(y_test,pred_test)
print("Training Accuracy: ", train_accuracy)
print("Testing Accuracy: ", test_accuracy)


# # Final Model (Ensembling Stacking all the models)

# In[59]:


from sklearn.ensemble import VotingClassifier
vt_classifier = VotingClassifier(estimators = [('lr', lr_clf),
                                               ('rf',rf_clf),
                                               ('adb',adb_clf),
                                               ('gdb',gdb_clf),
                                               ('etc',et_clf),
                                               ('svc',svc_clf),
                                               ('xgb',xgb_clf),
                                               ('lgbm',lgb_clf),], voting = 'hard')


# In[60]:


vt_classifier.fit(x_train,y_train)


# In[61]:


pred_train = vt_classifier.predict(x_train)
pred_test = vt_classifier.predict(x_test)
train_accuracy = accuracy_score(y_train,pred_train)
test_accuracy = accuracy_score(y_test,pred_test)
print("Training Accuracy: ", train_accuracy)
print("Testing Accuracy: ", test_accuracy)


# In[62]:


final_pred = vt_classifier.predict(X_train)
train_accuracy = accuracy_score(final_pred,Y_train)
print("Training Accuracy: ", train_accuracy)


# In[63]:


X_test = X_test.values
final_pred = final_model.predict(X_test)


# In[64]:


submission = pd.read_csv('../input/titanic/gender_submission.csv')
submission['Survived'] = final_pred
submission.to_csv('titanic_submission3.csv', index=False)


# In[ ]:




