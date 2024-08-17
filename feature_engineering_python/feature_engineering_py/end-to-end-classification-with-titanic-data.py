#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')


# In[3]:


train.head()


# In[4]:


train.shape, test.shape


# ## Data Preprocessing on Train data
# 
# Removing columns that we don't need

# In[5]:


train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)


# Checking for null values

# In[6]:


train.isna().sum()


# We have null values in two columns. Let's take care of this problem.
# 
# Let's fill the null values of age columns with the mean values

# In[7]:


train['Age'].fillna(train['Age'].mean(), inplace = True)


# Now we have to take care of null values of Embarked column.
# 
# Let's first check which embarkation port we have most in our dataset.

# In[8]:


train.Embarked.value_counts()


# `Southampton` is the top port of embarkation. So, let's fill the null values with `S`

# In[9]:


train['Embarked'].fillna('S', inplace = True)


# Let's check again for null values.

# In[10]:


train.isna().sum()


# **Nice!**
# 
# We don't any null values now

# # Data Exploration on Train set

# Let's first check how many people survived

# In[11]:


train.Survived.value_counts()


# In[12]:


train.Survived.value_counts().plot(kind = 'bar', color = ['lightblue', 'lightgreen']);


# Let's check how many male and female was there

# In[13]:


train.Sex.value_counts()


# In[14]:


train.Sex.value_counts().plot(kind = 'bar', color = ['skyblue', 'plum']);


# let's check out survivors w.r.t sex

# In[15]:


pd.crosstab(train.Sex, train.Survived)


# In[16]:


pd.crosstab(train.Sex, train.Survived).plot(kind = 'bar', color = ['slategray', 'salmon']);


# Survivors w.r.t pclass

# In[17]:


pd.crosstab(train.Pclass, train.Survived)


# In[18]:


pd.crosstab(train.Pclass, train.Survived).plot(kind = 'bar', color = ['slategray', 'lightcoral']);


# Let's check the Port of Embarkation

# In[19]:


train.Embarked.value_counts()


# Let's look at our age column

# In[20]:


sns.countplot(x = 'Embarked', data = train);


# In[21]:


sns.displot(x = 'Age', data = train, color = 'cadetblue', kde = True);


# In[22]:


sns.displot(x = 'Fare', data = train, kind = 'kde');


# Let's now find a relation among age, survived and pclass columns

# In[23]:


sns.lmplot(x = 'Age', y = 'Survived', hue = 'Pclass', data = train);


# In[24]:


correlation_matrix = train.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, 
            annot=True, 
            linewidths=0.5, 
            fmt= ".2f", 
            cmap="YlGnBu");


# # Feature Engineering in train data

# In[25]:


train['family'] = train['SibSp'] + train['Parch']


# In[26]:


train.head(10)


# Removing skewness in `Age` column

# In[27]:


train['Age']=np.log(train['Age']+1)


# In[28]:


train['Age'].plot(kind = 'density', figsize=(10, 6));


# Removing skewness in `Fare` column

# In[29]:


train['Fare']=np.log(train['Fare']+1)


# In[30]:


train['Fare'].plot(kind = 'density', figsize=(10, 6));


# In[31]:


train.head(10)


# Let's create x and y matrix of features

# In[32]:


x = train.drop('Survived',  axis = 1)
y = train['Survived']


# In[33]:


x.shape


# In[34]:


x.head()


# We have two `categorical` columns. Let's take care of them now.

# In[35]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ['Sex', 'Embarked', 'Pclass']
onehotencode = OneHotEncoder()

transformer = ColumnTransformer([('Encoder', onehotencode, categorical_features)], remainder = 'passthrough')

encoded = transformer.fit_transform(x)


# In[36]:


encoded_df = pd.DataFrame(encoded)


# In[37]:


encoded_df.shape


# In[38]:


encoded_df.head()


# **Avoiding Dummy variables**

# In[39]:


encoded_x = encoded_df.drop([0, 2, 5], axis = 1)


# In[40]:


encoded_x.head()


# In[41]:


encoded_x.shape


# In[42]:


y.shape


# # Feature Engineering in test data

# In[43]:


test['family'] = test['SibSp'] + test['Parch']


# In[44]:


test.head()


# Removing skewness in `Age` column

# In[45]:


test['Age']=np.log(test['Age']+1)


# Removing skewness in `Fare` column

# In[46]:


test['Fare']=np.log(test['Fare']+1)


# In[47]:


test['Age'].plot(kind = 'density', figsize=(10, 6));


# In[48]:


test['Fare'].plot(kind = 'density', figsize=(10, 6));


# In[49]:


test.head(10)


# # Preparing test set

# In[50]:


test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)


# In[51]:


test.head(10)


# Checking for null values

# In[52]:


test.isna().sum()


# In[53]:


test['Age'].fillna(test['Age'].mean(), inplace = True)
test['Fare'].fillna(test['Fare'].mean(), inplace = True)


# In[54]:


test.isna().sum()


# We succesfully removed all the null values

# As before we now have to take care of `categorical columns`

# In[55]:


categorical_features = ['Sex', 'Embarked', 'Pclass']
onehotencode = OneHotEncoder()

transformer = ColumnTransformer([('Encoder', onehotencode, categorical_features)], remainder = 'passthrough')

encoded_test = transformer.fit_transform(test)


# In[56]:


encoded_test = pd.DataFrame(encoded_test)


# In[57]:


encoded_test.head()


# Avoiding dummy variable trap

# In[58]:


encoded_test_x = encoded_test.drop([0, 2, 5], axis = 1)


# In[59]:


encoded_test_x.head()


# In[60]:


encoded_test_x.shape


# # Modeling

# Let's split our dataset

# In[61]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(encoded_x,y,random_state = 31)


# In[62]:


len(x_train), len(x_test), len(y_train), len(y_test)


# In[63]:


x_train.shape


# In[64]:


y_train.shape


# # Logistic Regression

# In[65]:


from sklearn.linear_model import LogisticRegression
log_clf = LogisticRegression(max_iter = 1000, random_state = 4)
log_clf.fit(x_train, y_train)
log_score = log_clf.score(x_test, y_test)
log_score


# ### Logistic Regression Hyperparameter Tuning

# In[66]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# In[67]:


log_grid = {'C': np.logspace(-4, 4),
            'solver': ['liblinear'],
            'max_iter': np.arange(100, 2000, 100),
            'penalty':['l1', 'l2']
           }

log_gscv = GridSearchCV(LogisticRegression(max_iter = 1000, random_state = 7),
                          param_grid=log_grid,
                          cv=5,
                          verbose=True)

log_gscv.fit(x_train, y_train)
log_tuned_score = log_gscv.score(x_test, y_test)
log_tuned_score


# In[68]:


log_gscv.best_params_


# ### Evaluating logistic regression model

# In[69]:


from sklearn.metrics import classification_report
y_preds = log_clf.predict(x_test)
print(classification_report(y_test, y_preds))


# In[70]:


from sklearn.metrics import plot_roc_curve
plot_roc_curve(log_clf, x_test, y_test)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve');


# # Linear SVC

# In[71]:


from sklearn import svm
svc_clf = svm.SVC(random_state = 7)
svc_clf.fit(x_train, y_train)
svc_score = svc_clf.score(x_test, y_test)
svc_score


# ### SVC Hyperparameter tuning

# In[72]:


svc_grid = {'C': np.logspace(-5, 5, 20),
            'kernel': ['rbf'],
            'degree': [2,3,4]
           }

svc_gscv = GridSearchCV(svm.SVC(random_state = 31),
                        param_grid=svc_grid,
                        cv=5,
                        verbose=True)

svc_gscv.fit(x_train, y_train)
svc_tuned_score = log_gscv.score(x_test, y_test)
svc_tuned_score


# In[73]:


svc_gscv.best_params_


# ### Evaluating with SVC 

# In[74]:


y_preds = svc_clf.predict(x_test)
print(classification_report(y_test, y_preds))


# In[75]:


from sklearn.metrics import plot_roc_curve
plot_roc_curve(svc_clf, x_test, y_test)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve');


# # KNeighbors Classifier

# In[76]:


from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train, y_train)
knn_score = knn_clf.score(x_test, y_test)
knn_score


# ### KNeighbors Classifier Hyperparameter Tuning

# In[77]:


knn_grid = {'n_neighbors': np.arange(2, 15),
            'leaf_size': [10, 15, 20, 25, 30, 35, 40, 45, 50],
            'p':[1,2,3,4,5], 
            'algorithm': ['auto', 'ball_tree', 'kd_tree']}

knn_gscv = GridSearchCV(KNeighborsClassifier(),
                        param_grid=knn_grid,
                        cv=5,
                        verbose=True)

knn_gscv.fit(x_train, y_train)
knn_tuned_score = knn_gscv.score(x_test, y_test)
knn_tuned_score


# In[78]:


knn_gscv.best_params_


# ### Evaluating KNN model

# In[79]:


y_preds = knn_clf.predict(x_test)
print(classification_report(y_test, y_preds))


# In[80]:


from sklearn.metrics import plot_roc_curve
plot_roc_curve(knn_clf, x_test, y_test)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve');


# # Random Forest

# In[81]:


from sklearn.ensemble import RandomForestClassifier

rand_clf = RandomForestClassifier(n_estimators=1000, random_state = 35)
rand_clf.fit(x_train, y_train)
ranf_score = rand_clf.score(x_test, y_test)
ranf_score


# ### Random Forest hyperparameter tuning

# In[82]:


rfcv_grid = {"n_estimators": np.arange(500, 2000, 100),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}

rfcv_clf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions = rfcv_grid,
                           cv=5,
                           n_iter = 100,
                           verbose=True)

rfcv_clf.fit(x_train, y_train)
ranf_tuned_score = rfcv_clf.score(x_test, y_test)
ranf_tuned_score


# In[83]:


rfcv_clf.best_params_


# ### Evaluating Random Forest model

# In[84]:


y_preds = rfcv_clf.predict(x_test)
print(classification_report(y_test, y_preds))


# In[85]:


from sklearn.metrics import plot_roc_curve
plot_roc_curve(rfcv_clf, x_test, y_test)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve');


# # GradientBoostingClassifier

# In[86]:


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)
gbc_score = gbc.score(x_test, y_test)
gbc_score


# ### GradientBoostingClassifier hyperparameter tuning

# In[87]:


gbc_grid = {'loss': ['deviance', 'exponential'],
            'learning_rate': [0.1,0.01],
            'n_estimators': [100, 200, 500, 1000],
            'min_samples_split': [2, 4, 6, 8, 10],
            'min_samples_leaf': [1, 2, 3, 5],
            'max_depth': [1, 2, 3]}


gbc_clf = GridSearchCV(GradientBoostingClassifier(),
                      param_grid = gbc_grid,
                           cv=5,
                           verbose=True)
gbc_clf.fit(x_train, y_train)
gbc_tuned_score = gbc_clf.score(x_test, y_test)
gbc_tuned_score


# In[88]:


gbc_clf.best_params_


# ### Evaluating gradient boosting model

# In[89]:


y_preds = gbc.predict(x_test)
print(classification_report(y_test, y_preds))


# In[90]:


from sklearn.metrics import plot_roc_curve
plot_roc_curve(gbc, x_test, y_test)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve');


# # CatBoost

# In[91]:


from catboost import CatBoostClassifier
cbc = CatBoostClassifier(random_seed = 31)
cbc.fit(x_train, y_train, verbose=False);
cbc_score = cbc.score(x_test, y_test);
cbc_score


# ### CatBoostClassifier hyperparameter tuning

# In[92]:


cbc_grid = {'iterations':[10, 100, 200, 500, 1000],
            'learning_rate': [0.1, 0.01]}


cbc_clf = GridSearchCV(CatBoostClassifier(random_state = 31),
                      param_grid = cbc_grid,
                           cv=5,
                           verbose=True)

cbc_clf.fit(x_train, y_train, verbose=False)
cbc_tuned_score = cbc_clf.score(x_test, y_test)
cbc_tuned_score


# In[93]:


cbc_clf.best_params_


# ### Evaluating CatBoost model

# In[94]:


y_preds = cbc_clf.predict(x_test)
print(classification_report(y_test, y_preds))


# In[95]:


from sklearn.metrics import plot_roc_curve
plot_roc_curve(cbc_clf, x_test, y_test)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve');


# # LGBM

# In[96]:


from lightgbm import LGBMClassifier
lgbm = LGBMClassifier()
lgbm.fit(x_train, y_train)
lgbm_score = lgbm.score(x_test, y_test)
lgbm_score


# ### LGBM hyperparameter tuning

# In[97]:


lgbm_grid = {'num_leaves': [10, 15, 30, 31, 40, 45],
             'n_estimators':[10, 50, 100, 200],
             'learning_rate': [0.1, 0.01],
             'min_child_samples': [5, 10, 15, 20, 25]}


lgbm_clf = GridSearchCV(LGBMClassifier(random_state = 31),
                           param_grid = lgbm_grid,
                           cv=5,
                           verbose=True)

lgbm_clf.fit(x_train, y_train, verbose=False)
lgbm_tuned_score = lgbm_clf.score(x_test, y_test)
lgbm_tuned_score


# In[98]:


lgbm_clf.best_params_


# ### Evaluating LGBM model

# In[99]:


y_preds = lgbm.predict(x_test)
print(classification_report(y_test, y_preds))


# In[100]:


from sklearn.metrics import plot_roc_curve
plot_roc_curve(lgbm, x_test, y_test)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve');


# Let's make a dictionary of all models and accuracy

# In[101]:


score = [{'Model':'Logistic Regression', 'Score': log_score, 'Tuned_score': log_tuned_score}, 
         {'Model':'SVC', 'Score': svc_score, 'Tuned_score': svc_tuned_score},
         {'Model':'KNN', 'Score': knn_score, 'Tuned_score': knn_tuned_score},
         {'Model':'Random Forest', 'Score': ranf_score, 'Tuned_score': ranf_tuned_score},
         {'Model':'Gradient Boosting', 'Score': gbc_score, 'Tuned_score': gbc_tuned_score},
         {'Model':'CatBoost', 'Score': cbc_score, 'Tuned_score': cbc_tuned_score},
         {'Model':'LGBM', 'Score': lgbm_score, 'Tuned_score': lgbm_tuned_score}]


# Let's view all model score as a dataframe to get a good overview

# In[102]:


pd.DataFrame(score, columns=['Model','Score','Tuned_score'])


# **Looks like SVC classifier is doing best. So, let's predict with this**.

# In[103]:


final_preds = svc_clf.predict(encoded_test_x)


# # Creating file for submission 

# In[104]:


sub_data = pd.read_csv('../input/titanic/gender_submission.csv')
final_data = {'PassengerId': sub_data.PassengerId, 'Survived': final_preds}
final_submission = pd.DataFrame(data=final_data)
final_submission.to_csv('submission_file_titanic.csv',index =False)


# **`If this notebook was useful to you. Don't forget to upvote. Thanks`**
