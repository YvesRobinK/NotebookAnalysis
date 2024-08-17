#!/usr/bin/env python
# coding: utf-8

# <center><img src='https://www.marineinsight.com/wp-content/uploads/2019/09/Titanic-sinking-1.png'></img></center>
# <center><h2 style='font-family:monospace;'>TITANIC SURVIVIAL PREDICTION USING ML</h2></center>
# <center>Dataset Link <br><a 'https://www.kaggle.com/competitions/titanic'>TITANIC - MACHINE LEARNING FROM DISASTER</a></center>
# 
# <b> The Challenge </b><br>
# The sinking of the Titanic is one of the most infamous shipwrecks in history.
# 
# On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.
# 
# While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
# 
# In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).
# 
# 
# <b> What Data Will I Use in This Competition? </b><br>
# In this competition, you’ll gain access to two similar datasets that include passenger information like name, age, gender, socio-economic class, etc. One dataset is titled `train.csv` and the other is titled `test.csv`.
# 
# Train.csv will contain the details of a subset of the passengers on board (891 to be exact) and importantly, will reveal whether they survived or not, also known as the “ground truth”.
# 
# The `test.csv` dataset contains similar information but does not disclose the “ground truth” for each passenger. It’s your job to predict these outcomes.
# 
# Using the patterns you find in the train.csv data, predict whether the other 418 passengers on board (found in test.csv) survived.
# 
# Check out the “Data” tab to explore the datasets even further. Once you feel you’ve created a competitive model, submit it to Kaggle to see where your model stands on our leaderboard against other Kagglers.
# 
# <b>Goal</b><br>
# 
# It is your job to predict if a passenger survived the sinking of the Titanic or not.
# For each in the test set, you must predict a 0 or 1 value for the variable.

# --------STEPS--------------
# 1. Loading Dataset
# 2. EDA
# 3. Feature Engineering
# 4. Model Building and Training
# 5. Model Evaluation
# 6. Hyper parameter Tuning
# 7. Test Set Cleaning
# 8. Prediction on Test Set
# 9. Submitting Result To Kaggle

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Loading Dataset

# In[2]:


df_train = pd.read_csv ('../input/titanic/train.csv')
df_test = pd.read_csv('../input/titanic/test.csv')


# In[3]:


df_train.head()


# In[4]:


df_train.info()


# In[5]:


## Checking for  the number of rows and columns  in the dataset
print(f"Number of rows :{df_train.shape[0]} \nNumber of columns:{df_train.shape[1]}")


# ## EDA

# **Survived**

# In[6]:


ax = sns.countplot(data=df_train,x = 'Survived');
ax.bar_label(ax.containers[0])
plt.title("Number of people Survived vs  Deceased")
plt.xlabel("Survived vs  Deceased")
plt.ylabel("Number of people")
plt.xticks(ticks=[0,1],labels=['Deceased','Survived'])
plt.show();


# **Age**

# In[7]:


df_train.Age.isna().sum()


# In[8]:


### Filling NAN Values With Mean
df_train["Age"]= df_train["Age"].fillna(df_train["Age"].mean())
### Plotting People On Different Age Groups
df_train.Age = df_train.Age.astype(int)


# In[9]:


#### Creating A Copy of The Dataset
temp = df_train.copy()


# In[10]:


temp['Age'] = pd.cut(temp['Age'], bins=[0,12,20,40,120], labels=['Children','Teenage','Adult','Elder'])


# In[11]:


ax = sns.countplot(data=temp,x = 'Age')
ax.bar_label(ax.containers[0]);


# In[12]:


ax = sns.countplot(data=temp,x = 'Age',hue='Survived')
ax.bar_label(ax.containers[0]);
ax.bar_label(ax.containers[1]);
plt.legend(title='Survived or Not', loc='upper right', labels=['No', 'Yes']);
plt.title('Age Vs Survived')
plt.show();


# **Fare**

# In[13]:


temp['Fare'].mean()


# In[14]:


temp['Fare'].hist(bins=20)


# In[15]:


temp['Fare'] = pd.cut(temp['Fare'], bins=[0,8,16,32,110], labels=['Low_fare','median_fare','Average_fare','high_fare'])
ax = sns.countplot(data=temp, x='Fare')
ax.bar_label(ax.containers[0]);


# In[16]:


ax = sns.countplot(data=temp, x='Fare',hue='Survived')
ax.bar_label(ax.containers[0]);
ax.bar_label(ax.containers[1]);
plt.legend(title='Survived or Not', loc='upper right', labels=['No', 'Yes']);
plt.title('Fare Vs Survived')
plt.show();


# **Cabin**

# In[17]:


nullCabin = temp.Cabin.isna().sum()
nullCabin


# In[18]:


str(round(nullCabin/(len(temp))*100,2))+"% Null Values"


# There Are So Many Null Values So Its Better To Drop It.

# **Pclass : Passenger Class**

# In[19]:


temp.Pclass.unique()


# In[20]:


ax = sns.countplot(data=temp,x='Pclass');
ax.bar_label(ax.containers[0]);


# In[21]:


ax = sns.countplot(data=temp,x='Pclass',hue='Survived');
ax.bar_label(ax.containers[0]);
ax.bar_label(ax.containers[1]);
plt.legend(title='Survived or Not', loc='upper left', labels=['No', 'Yes']);


# **Sex**

# In[22]:


temp.Sex.unique()


# In[23]:


ax = sns.countplot(data=temp,x='Sex');
ax.bar_label(ax.containers[0]);


# In[24]:


### Sex vs Survived
ax = sns.countplot(data=temp,x='Sex',hue='Survived');
ax.bar_label(ax.containers[0]);
ax.bar_label(ax.containers[1]);
plt.legend(title='Survived or Not', loc='upper right', labels=['No', 'Yes']);


# **SibSp  : # of siblings / spouses aboard the Titanic**

# In[25]:


temp.SibSp.unique()


# In[26]:


ax = sns.countplot(data=temp,x='SibSp');
ax.bar_label(ax.containers[0]);
ax.set_xlabel('Number of Siblings');


# In[27]:


### Number of Siblings vs Survived
ax = sns.countplot(data=temp,x='SibSp',hue='Survived');
ax.bar_label(ax.containers[0]);
ax.bar_label(ax.containers[1]);
plt.legend(title='Survived or Not', loc='upper right', labels=['No', 'Yes']);
plt.title('Effect of Number of Siblings On Survival')
plt.show();


# **Parch: # of parents / children aboard the titanic**

# In[28]:


temp.Parch.unique()


# In[29]:


ax = sns.countplot(data=temp,x='Parch');
ax.bar_label(ax.containers[0]);
ax.set_xlabel(' # Parents / Children Aboard The Titanic');


# In[30]:


### Parch vs Survived
plt.figure(figsize=(10,6))
ax = sns.countplot(data=temp,x='Parch',hue='Survived');
ax.bar_label(ax.containers[0]);
ax.bar_label(ax.containers[1]);
plt.legend(title='Survived or Not', loc='upper right', labels=['No', 'Yes']);
plt.title('Effect of # Parents / Children Aboard The Titanic On Survival')
plt.show();


# In[31]:


temp['Family Size'] = temp['SibSp']+temp['Parch'] + 1


# In[32]:


ax = sns.countplot(data=temp, x='Family Size')
ax.bar_label(ax.containers[0])
ax.set_title('Family Size');


# In[33]:


### Parch vs Survived
plt.figure(figsize=(10,6))
ax = sns.countplot(data=temp,x='Family Size',hue='Survived');
ax.bar_label(ax.containers[0]);
ax.bar_label(ax.containers[1]);
plt.legend(title='Survived or Not', loc='upper right', labels=['No', 'Yes']);
plt.title('Family Size vs Survival')
plt.show();


# **Embarked**

# In[34]:


temp.Embarked.unique()


# In[35]:


ax = sns.countplot(data=temp,x='Embarked');
ax.bar_label(ax.containers[0]);
ax.set_xlabel(' Port of Embarkation');


# In[36]:


### Port of Embarkation vs Survived
plt.figure(figsize=(10,6))
ax = sns.countplot(data=temp,x='Embarked',hue='Survived');
ax.bar_label(ax.containers[0]);
ax.bar_label(ax.containers[1]);
plt.legend(title='Survived or Not', loc='upper right', labels=['No', 'Yes']);
plt.title('Port of Embarkation Effect On Survival')
plt.show();


# ### EDA Analysis Results
# 
# * Based On The Dataset Only 40% People Were Able To Survived The Disaster.
# * Mostly Childrens Are Being Rescued.
# * People Above Age 20 had a chance of 35% of Being Survived From The Disaster.
# * People Paid High Fare and Class Means VIPs Are Given Priority For Rescued. 
# * Out of Male and Females, Almost 75% Femals Survived The Disaster.
# * People Traveling Alone or With A Smaller Family Size Upto 2 children had high chances of survival.
# * Family Size Under 5 Had Higher Chance of Survival On Titanic Disaster.
# * People Traveling Alone had approx 43% Chances of Survival.
# * Family with Size 5+ Had Lesser Chance of Complete Survival On Titanic Disaster. 

# ## Feature Engineering

# In[37]:


def clean_data(data):
    #### Let's Start By Dropping `PassengerID` `Name` `Ticket` `Cabin` 
    data = data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
    
    #### Converting Age Into Four Different Classes 'Children', 'Teenage','Adult', 'old'
    data['Age'] = pd.cut(data['Age'], bins=[0,12,20,40,120], labels=['Children','Teenage','Adult','Elder'])
    
    #### Converting Fare Into Four Distinct Categories 'Low_fare','median_fare','Average_fare','high_fare'
    data['Fare'] = pd.cut(data['Fare'], bins=[0,7.91,14.45,31,120], labels=['Low_fare','median_fare','Average_fare','high_fare'])
    
    #### Getting OneHotEncoding For Categorical Columns ['Age','Fare','Sex','Embarked']
    data = pd.get_dummies(data, columns = ["Sex","Age","Embarked","Fare"])
    
    return data


# In[38]:


df_train = clean_data(df_train)


# In[39]:


X = df_train.drop('Survived',axis=1)
y = df_train['Survived']


# In[40]:


X.head()


# In[41]:


y.head()


# In[42]:


sns.heatmap(X.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(20,12)
plt.show()


# `POSITIVE CORRELATION` : if increase in A leads to increase in Feature B Then Those Two Features Are Positive correlated. The Value for a perfectly positive correlated features is `1`.
# 
# `NEGATIVE CORRELATION`: if increase in A leads to decrease in Feature B Then Those Two Features Are Negative correlated. Then Value for a perfectly negative correlated feature is `-1`.
# 
# Having Two Highly or Perfectly Correlated Feature In Our Training Data Will Cause  MultiColinearity So It is better to remove them. 
# 
# In The Above Heatmap we can see that there are no highly correlated feature. the highest value for correlation is `0.41` Between features `Parch` and `SibSp`. So There Are No Need To Remove Any Feature.
# 
# We Can Also Do This With Help of Code By Defining A Threshold Value like `0.6`
# 

# In[43]:


threshold=0.60
# find and remove correlated features
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

correlation(X,threshold)


# We Got A Empty `set()` means there are no correlated features.

# ## Model Training
# 
# **Models**
# 1. Naive Bayes
# 2. SVM
# 3. KNN
# 4. Logistic Regression
# 5. Random Forest
# 6. Gradient Boost
# 7. ADA Boost

# In[44]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[45]:


from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict ## Cross Validation


# ### Naive Bayes

# In[46]:


#################################### Naive Bayes ########################
from sklearn.naive_bayes import GaussianNB
model= GaussianNB()
model.fit(X_train,y_train)
prediction_gnb=model.predict(X_test)

print('--------GaussianNB Naive Bayes -------')
print('The accuracy Gaussian Naive Bayes Classifier is',round(accuracy_score(prediction_gnb,y_test)*100,2))

kfold = KFold(n_splits=8,shuffle=True,random_state=42) # split the data into 10 equal parts

result_gnb=cross_val_score(model,X,y,cv=10,scoring='accuracy')

print('The cross validated score for Gaussian Naive Bayes classifier is:',round(result_gnb.mean()*100,2))

y_pred = cross_val_predict(model,X,y,cv=10)
sns.heatmap(confusion_matrix(y,y_pred),annot=True,fmt='3.0f',cmap="Accent_r")
plt.title('Confusion Matrix', y=1, size=15);


# ### SVM

# In[47]:


#################################### SVM ########################
from sklearn.svm import SVC, LinearSVC
model = SVC()
model.fit(X_train,y_train)
prediction_svm=model.predict(X_test)

print('--------SVM -------')
print('The accuracy SVM is',round(accuracy_score(prediction_svm,y_test)*100,2))

kfold = KFold(n_splits=8,shuffle=True,random_state=42) # split the data into 10 equal parts

result_svm=cross_val_score(model,X,y,cv=10,scoring='accuracy')

print('The cross validated score for SVM is:',round(result_svm.mean()*100,2))

y_pred = cross_val_predict(model,X,y,cv=10)
sns.heatmap(confusion_matrix(y,y_pred),annot=True,fmt='3.0f',cmap="Accent_r")
plt.title('Confusion Matrix', y=1, size=15);


# ### KNN

# In[48]:


#################################### KNN ########################
from sklearn.neighbors import KNeighborsClassifier
model =  KNeighborsClassifier()
model.fit(X_train,y_train)
prediction_knn=model.predict(X_test)

print('--------KNN -------')
print('The accuracy KNN Classifier is',round(accuracy_score(prediction_knn,y_test)*100,2))

kfold = KFold(n_splits=8,shuffle=True, random_state=42) # split the data into 10 equal parts

result_knn=cross_val_score(model,X,y,cv=10,scoring='accuracy')

print('The cross validated score for KNN classifier is:',round(result_knn.mean()*100,2))

y_pred = cross_val_predict(model,X,y,cv=10)
sns.heatmap(confusion_matrix(y,y_pred),annot=True,fmt='3.0f',cmap="Accent_r")
plt.title('Confusion Matrix', y=1, size=15);


# ### Logistic Regression

# In[49]:


#################################### Logistic Regression ########################
from sklearn.linear_model import LogisticRegression
model =  LogisticRegression()
model.fit(X_train,y_train)
prediction_lr=model.predict(X_test)

print('--------Logistic Regression -------')
print('The accuracy Logistic Regression is',round(accuracy_score(prediction_lr,y_test)*100,2))

kfold = KFold(n_splits=8,shuffle=True, random_state=42) # split the data into 10 equal parts

result_lr=cross_val_score(model,X,y,cv=10,scoring='accuracy')

print('The cross validated score for Logistic Regression is:',round(result_lr.mean()*100,2))

y_pred = cross_val_predict(model,X,y,cv=10)
sns.heatmap(confusion_matrix(y,y_pred),annot=True,fmt='3.0f',cmap="Accent_r")
plt.title('Confusion Matrix', y=1, size=15);


# ### Random Forest

# In[50]:


# Random Forests
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=800,
                             min_samples_split=12,
                             max_features='auto',oob_score=True,
                             random_state=1,n_jobs=-1)

model.fit(X_train,y_train)
prediction_rf=model.predict(X_test)

print('--------Random Forest Classifier -------')
print('The accuracy Random Forest Classifier  is',round(accuracy_score(prediction_rf,y_test)*100,2))

kfold = KFold(n_splits=8,shuffle=True, random_state=42) # split the data into 10 equal parts

result_rf=cross_val_score(model,X,y,cv=10,scoring='accuracy')

print('The cross validated score for Random Forest Classifier is:',round(result_rf.mean()*100,2))

y_pred = cross_val_predict(model,X,y,cv=10)
sns.heatmap(confusion_matrix(y,y_pred),annot=True,fmt='3.0f',cmap="Accent_r")
plt.title('Confusion Matrix', y=1, size=15);


# ### Gradient Boost

# In[51]:


from sklearn.ensemble import GradientBoostingClassifier
model= GradientBoostingClassifier()

model.fit(X_train,y_train)
prediction_gb=model.predict(X_test)

print('-------- Gradient Boost -------')
print('The accuracy Gradient Boost Classifier is',round(accuracy_score(prediction_gb,y_test)*100,2))

kfold = KFold(n_splits=8, shuffle=True,random_state=42) # split the data into 10 equal parts

result_gb=cross_val_score(model,X,y,cv=10,scoring='accuracy')

print('The cross validated score for Gradient Boost classifier is:',round(result_gb.mean()*100,2))

y_pred = cross_val_predict(model,X,y,cv=10)
sns.heatmap(confusion_matrix(y,y_pred),annot=True,fmt='3.0f',cmap="Accent_r")
plt.title('Confusion Matrix', y=1, size=15);


# ### ADA Boost

# In[52]:


from sklearn.ensemble import AdaBoostClassifier
model= AdaBoostClassifier()

model.fit(X_train,y_train)
prediction_ada=model.predict(X_test)

print('--------ADA Boost -------')
print('The accuracy ADA Boost Classifier is',round(accuracy_score(prediction_ada,y_test)*100,2))

kfold = KFold(n_splits=8,shuffle=True, random_state=42) # split the data into 10 equal parts

result_ada=cross_val_score(model,X,y,cv=10,scoring='accuracy')

print('The cross validated score for ADA Boost classifier is:',round(result_ada.mean()*100,2))

y_pred = cross_val_predict(model,X,y,cv=10)
sns.heatmap(confusion_matrix(y,y_pred),annot=True,fmt='3.0f',cmap="Accent_r")
plt.title('Confusion Matrix', y=1, size=15);


# **Model Evaluation**

# In[53]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'AdaBoostClassifier', 
              'Gradient Boosting'],
    'Score': [result_svm.mean(), result_knn.mean(), result_lr.mean(), 
              result_rf.mean(), result_gnb.mean(), result_ada.mean(), 
              result_gb.mean()]})
models.sort_values(by='Score',ascending=False)


# ## HyperParameters Tuning

# In[54]:


from sklearn.model_selection import GridSearchCV

n_estim=range(100,1000,100)
max_depth = range(5,15,2)

param_grid = {"n_estimators" :n_estim,"max_depth":max_depth}
model = RandomForestClassifier()
model_rf = GridSearchCV(model,param_grid = param_grid, cv=5, scoring="accuracy", n_jobs= 4, verbose = 1)

model_rf.fit(X_train,y_train)

print(model_rf.best_score_)

#best estimator
model_rf.best_estimator_


# In[55]:


### Applying Param Got From GridSearchCV
rfmodel = RandomForestClassifier(max_depth=11, min_samples_split=12, n_jobs=-1,
                       oob_score=True, random_state=11)
rfmodel.fit(X_train,y_train)


# ## Test Data

# In[56]:


test = clean_data(df_test)


# In[57]:


y_pred = rfmodel.predict(test)


# In[58]:


submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": y_pred})


# In[59]:


submission.to_csv('Submission.csv', index=False)


# ### VOTE
# 
# * Give a Upvote 🙌 if You Liked The Notebook.

# ### CONNECT WITH ME
# 
# [LinkedIN](https://www.linkedin.com/in/abhayparashar31/) | [Medium](https://medium.com/@abhayparashar31) | [Twitter](https://twitter.com/abhayparashar31) | [Github](https://github.com/Abhayparashar31)

# #### HOPE TO SEE YOU IN MY NEXT KAGGLE NOTEBOOK 😀
