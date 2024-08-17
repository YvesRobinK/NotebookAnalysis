#!/usr/bin/env python
# coding: utf-8

# ### **Titanic Competition**
# 
# this is my first published notebook about the kaggle competition, I have synthesized the results from various notebooks and experimented on my own on the data and reached **0.799 score**, I hope this notebook is useful for someone
# 
# * **Your feedback is welcome**
# * **Commented code is things i have tried that hasn't worked**

# In[1]:


#importing some useful libraries
import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 
from sklearn.metrics import f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
sns.set()

scaler = StandardScaler()


# In[2]:


#reading the data
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train.head()


# In[3]:


train.shape


# In[4]:


test.shape


# # Exploring missing values

# In[5]:


train.isnull().sum() #to_show_null_data(Age and cabin embarked)


# In[6]:


test.isnull().sum() #to_show_null_data(AGE and cabin)


# # Plotting some useful visualizations about the features

# In[7]:


def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[8]:


bar_chart('Sex')
bar_chart('Pclass')
bar_chart('Embarked')
bar_chart('Parch')
bar_chart('SibSp')


# In[9]:


all_data = [train,test]
for data in all_data :
    data['Status'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[10]:


train.Status.unique()


# In[11]:


for dataset in all_data:
    dataset['Status'] = dataset['Status'].replace(['Lady', 'Countess','Capt', 'Col',\
     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Status'] = dataset['Status'].replace('Mlle', 'Miss')
    dataset['Status'] = dataset['Status'].replace('Ms', 'Miss')
    dataset['Status'] = dataset['Status'].replace('Mme', 'Mrs')
    


# In[12]:


train.isnull().sum()


# In[13]:


test.head(10)
bar_chart('Status')


# In[14]:


status_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in all_data:
    dataset['Status'] = dataset['Status'].map(status_mapping)
    dataset['Status'] = dataset['Status'].fillna(0)


# In[15]:


train.Status.unique()


# In[16]:


bar_chart('Status')


# # Feature engineering

# In[17]:


train['FamilySize'] = train ['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test ['SibSp'] + test['Parch'] + 1


# In[18]:


train.head()


# In[19]:


sex_mapping = {"male": 0, "female": 1}
for dataset in all_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
    
for dataset in all_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1


# In[20]:


train['Cabin_category'] = train['Cabin'].astype(str).str[0]
train['Cabin_category'] = train['Cabin_category'].map({'A':1,'B':2,'C':2,'D':3,'E':4,'F':5,'G':6,'T':7})
train['Cabin_category'] = train['Cabin_category'].fillna(0)
# Cabin Grouping 
train['HasCabin'] = train['Cabin'].apply(lambda x:0 if x is np.nan else 1)


test['Cabin_category'] = test['Cabin'].astype(str).str[0]
test['Cabin_category'] = test['Cabin_category'].map({'A':1,'B':2,'C':2,'D':3,'E':4,'F':5,'G':6,'T':7})
test['Cabin_category'] = test['Cabin_category'].fillna(0)
# Cabin Grouping 
test['HasCabin'] = test['Cabin'].apply(lambda x:0 if x is np.nan else 1)

#train['Name_length'] = train['Name'].apply(len)
#test['Name_length'] = test['Name'].apply(len)


# In[21]:


train.head()


# # Filling in missing data

# In[22]:


train.isnull().sum()


# In[23]:


train["Age"].fillna(train.groupby("Status")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Status")["Age"].transform("median"), inplace=True)
train['Fare'].fillna(7.5, inplace = True)
test['Fare'].fillna(7.5, inplace = True)
train['Embarked'].fillna('S', inplace = True)
test['Embarked'].fillna('S', inplace = True)


# In[24]:


train.isnull().sum()


# In[25]:


train.isnull().sum()


# In[26]:


train.groupby("Status")["Age"].transform("median")


# In[27]:


#train['Minor'] = (train['Age'] < 14.0) & (train['Age']>= 0)
#test['Minor'] = (test['Age'] < 14.0) & (test['Age']>= 0)


# In[28]:


def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(train)


# # Handling Outliers

# In[29]:


train.Fare = train.Fare.apply(lambda l: np.log(l+1))
test.Fare = test.Fare.apply(lambda l: np.log(l+1))

train.Fare = scaler.fit_transform(train.Fare.values.reshape(-1,1))
test.Fare = scaler.transform(test.Fare.values.reshape(-1,1))


train.Age = scaler.fit_transform(train.Age.values.reshape(-1,1))
test.Age = scaler.transform(test.Age.values.reshape(-1,1))


# In[30]:


train.head()


# # Training our model and making predictions

# In[31]:


y_full = train["Survived"]
features = ["Pclass","Sex","Age","IsAlone", "FamilySize", "Status","Embarked","Fare","Cabin_category","HasCabin"]

X_full = pd.get_dummies(train[features])
X_test_full = pd.get_dummies(test[features])
X_train, X_valid, y_train, y_valid = train_test_split(X_full, y_full, train_size=0.8, test_size=0.2,random_state=42)


# In[32]:


rf_model = RandomForestClassifier(criterion = 'gini', n_estimators = 100, max_depth = 3, min_samples_split=6, min_samples_leaf=6, random_state=3, oob_score = True)


# In[33]:


rf_model.fit(X_train, y_train)
rf_val_predictions = rf_model.predict(X_valid)


# In[34]:


feature_importances = pd.Series(rf_model.feature_importances_, X_full.columns)
feature_importances.sort_values(inplace=True)
feature_importances.plot(kind = "barh",figsize = (7,6))


# * **Scoring our model**

# In[35]:


def evaluation(model):
    
    model.fit(X_train, y_train)
    ypred = model.predict(X_valid)
    
    lr_probs = model.predict_proba(X_valid)
    lr_probs = lr_probs[:, 1]
    lr_auc = roc_auc_score(y_valid, lr_probs)
    
    print(confusion_matrix(y_valid, ypred))
    print(classification_report(y_valid, ypred))
    
    N, train_score, val_score = learning_curve(model, X_train, y_train,
                                              cv=4, scoring='accuracy',
                                               train_sizes=np.linspace(0.1, 1, 10))
    
    plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.legend()
    plt.show()


# In[36]:


evaluation(rf_model)


# In[37]:


rf_model.fit(X_full, y_full)
predictions = rf_model.predict(X_test_full)


# # Preparing and formatting our submissions

# In[38]:


output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)


# # **Credits** :
# 
# * https://www.kaggle.com/chapagain/titanic-solution-a-beginner-s-guide
# * https://www.kaggle.com/startupsci/titanic-data-science-solutions
# * https://www.kaggle.com/rafalplis/my-approach-to-titanic-competition
# * https://www.kaggle.com/brendan45774/titanic-top-solution
# * https://www.kaggle.com/khkuggle/simple-and-intermediate-eda-modeling-for-titanic
