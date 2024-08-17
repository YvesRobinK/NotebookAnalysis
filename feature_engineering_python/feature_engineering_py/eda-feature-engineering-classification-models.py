#!/usr/bin/env python
# coding: utf-8

# Kindly,provide feedback and help me to grow.
# Upvote if you like my analysis.

# In[1]:


# importing basic libraries and dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_profiling
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

train_data= pd.read_csv('../input/titanic/train.csv')

test_data= pd.read_csv('../input/titanic/test.csv')
test_data['Survived']= np.nan
full_data= pd.concat([train_data,test_data])


# In[2]:


full_data.profile_report()


# # 1. Feature Engineering
# 
# ## 1.1 Dealing with missing values

# In[3]:


# missingno is a python library used to visualiza missing data
import missingno as msno
msno.matrix(full_data);


# Null values in Survived column are of the test dataset. Age and Cabin columns contain many missing values.

# In[4]:


print("Percentages of missing values: ")
full_data.isnull().mean().sort_values(ascending = False)


# Embarked and Fare have less than 1% missing values.So, we will simply fill them with mode and median.
# To fill missing Age values I will find most correlated factor with age.

# In[5]:


from statistics import mode
full_data["Embarked"] = full_data["Embarked"].fillna(mode(full_data["Embarked"]))


# In[6]:


sns.heatmap(full_data.corr(),cmap='viridis');


# So, we will fill Age and Fare column with help of Pclass feature.

# In[7]:


full_data['Fare'] = full_data.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median()))
full_data['Age'] = full_data.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))


# In[8]:


full_data['Cabin'].isna().sum()/len(full_data)


# Almost 3/4th data is missing in Cabin feature.So, we will drop this column.

# In[9]:


full_data.drop('Cabin',axis=1,inplace=True)


# In[10]:


full_data.info()


# ## 1.2 Converting categorical columns.

# Now we will convert categorical columns into numerical using dummy variables.

# In[11]:


embarked = pd.get_dummies(full_data[['Embarked','Sex']],drop_first=True)
full_data = pd.concat([full_data,embarked],axis=1)


# We will drop PassengerId and Ticket column as it doesn't seem important.Name too is not of much significance but salutation can be of importance.

# In[12]:


Name1 = full_data['Name'].apply(lambda x : x.split(',')[1])


# In[13]:


full_data['Title'] = Name1.apply(lambda x : x.split('.')[0])


# In[14]:


full_data['Title'].value_counts(normalize=True)*100


# Except first four titles all form less than 1% of the data.So, we will combine them into one category and then form dummy variables.

# In[15]:


full_data['Title'] = full_data['Title'].replace([ ' Don', ' Rev', ' Dr', ' Mme',' Ms', ' Major', ' Lady', ' Sir', ' Mlle', ' Col', ' Capt',' the Countess', ' Jonkheer', ' Dona'], 'Other')


# In[16]:


full_data['Title'].unique()


# In[17]:


embarked = pd.get_dummies(full_data['Title'],drop_first=True)
full_data = pd.concat([full_data,embarked],axis=1)


# In[18]:


full_data.drop(['PassengerId','Name','Sex','Ticket','Title','Embarked'],axis=1,inplace=True)


# In[19]:


full_data.info()


# Now, let's retrieve our training and test data. And then convert each feature to integer.

# In[20]:


test = full_data[full_data['Survived'].isna()].drop(['Survived'], axis = 1)
train = full_data[full_data['Survived'].notna()]


# In[21]:


train = train.astype(np.int64)
test = test.astype(np.int64)


# In[22]:


train.shape,test.shape


# # 2.Exploratory data analysis

# In[23]:


sns.countplot(x='Survived',data=train_data,hue='Sex');


# Somehow, those who survived had more ratio of females and vice versa.  ;)

# In[24]:


sns.countplot(x='Survived',data=train_data,hue='Pclass');


# In[25]:


sns.distplot(train['Age'],kde=False,color='darkred',bins=30);


# Mostly people on board were aged between 20-40.

# In[26]:


sns.countplot(x='SibSp',data=train);


# Mostly people on board were without their siblings or spouse.

# In[27]:


sns.countplot(x='Parch',data=train);

Mostly people on board were travelling alone.
# In[28]:


train['Fare'].hist(color='green',bins=40,figsize=(12,6))
plt.xlabel('Fare');


# Fare seems to be mostly below 100.

# # 3. Applying Logistic Regression.

# In[29]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived'], axis = 1), 
                                                    train['Survived'], test_size = 0.2, 
                                                    random_state = 2)


# In[30]:


logisticRegression = LogisticRegression(max_iter = 10000)
logisticRegression.fit(X_train, y_train)
predictions = logisticRegression.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# Let's improve our accuracy by using N-fold cross-validation.

# In[31]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score,cross_val_predict


# In[32]:


kf = KFold(n_splits = 5)
score = cross_val_score(logisticRegression, train.drop('Survived', axis = 1),train['Survived'], cv = kf)
print(f"Accuracy after cross validation is {score.mean()*100}")


# It has improved to 81%.
# 

# # 4. Applying deep neural network

# In[33]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.callbacks import EarlyStopping


# In[34]:


model = Sequential()
model.add(Dense(units=12,activation='tanh'))
model.add(Dense(units=100,activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(units=100,activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(units=100,activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

model.fit(x=X_train.values, 
          y=y_train.values, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]
          )


# In[35]:


model_loss = pd.DataFrame(model.history.history)
model_loss.plot();


# In[36]:


dnn_predictions = model.predict_classes(X_test)
print(classification_report(y_test,dnn_predictions))
print(confusion_matrix(y_test,dnn_predictions))


# It provides less accuracy than logistic regression

# # 5. Applying Random Forest.

# In[37]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print(classification_report(y_test,rfc_pred))
print(confusion_matrix(y_test,rfc_pred))


# In[38]:


param_grid = { 
    'criterion' : ['gini', 'entropy'],
    'n_estimators': [100, 300,500],
    'max_features': ['auto', 'log2'],
    'max_depth' : [3,5, 7,9]    
}

from sklearn.model_selection import GridSearchCV
randomForest_CV = GridSearchCV(estimator = rfc, param_grid = param_grid, cv = 5)
randomForest_CV.fit(X_train, y_train)
grid_pred = randomForest_CV.predict(X_test)
print(classification_report(y_test,grid_pred))
print(confusion_matrix(y_test,grid_pred))


# In[39]:


randomForest_CV.best_params_


# # 6. Applying XGBoost

# In[40]:


from xgboost import plot_importance,XGBClassifier


# In[41]:


xgb = XGBClassifier().fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
print(confusion_matrix(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))


# In[42]:


print("Feature Importance")
plot_importance(xgb);


# # 7. Submitting predictions.

# In[43]:


test['Survived'] = logisticRegression.predict(test)
test['PassengerId'] = test_data['PassengerId']
test[['PassengerId', 'Survived']].to_csv('lm_submission.csv', index = False)


# In[44]:


test['Survived'] = model.predict_classes(test.iloc[:,:12])
test[['PassengerId', 'Survived']].to_csv('dnn_submission.csv', index = False)


# In[45]:


test['Survived'] = rfc.predict(test.iloc[:,:12])
test[['PassengerId', 'Survived']].to_csv('rfc_submission.csv', index = False)


# In[46]:


test['Survived'] = xgb.predict(test.iloc[:,:12])
test[['PassengerId', 'Survived']].to_csv('xgb_submission.csv', index = False)

