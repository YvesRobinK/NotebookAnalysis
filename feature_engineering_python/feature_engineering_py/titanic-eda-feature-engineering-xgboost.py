#!/usr/bin/env python
# coding: utf-8

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


# <h1>Titanic - Machine Learning from Disaster: Analysis, Feature-engineering, Modelling</h1>

# <h2>Table of Contents</h2>
# 
# * [Introduction](#introduction)
# * [Columns of the Dataset](#metadata)
# * [First look at the dataset and some initial thoughts](#firstlook)
# * [EDA and Feature Engineering](#EDA)
# * [Creating Training and Test Sets, Preprocessing](#preprocessing)
# * [Modelling](#modelling)
# * [Making the Submission](#sub)
# * [Conclusion](#conclusion)

# <a id="introduction"></a>
# 
# <h2>Introduction</h2>
# 
# 
# In this notebook, we will be working with the Titanic - Machine Learning from Disaster dataset. We will first do some exploratory data analysis, with some feature-engineering along the way. Then, after some preprocessing we will do modelling using XGBoost. 

# <a id="metadata"></a>
# 
# <h2>Columns of the Dataset</h2>
# 
# The columns present in the dataset are as follows: 
# 1. PassengerId: This column assigns a unique identifier for each passenger.
# 2. Survived: Specifies whether the given passenger survived or not (1 - survived, 0 - didn't survive)
# 3. Pclass: The passenger's class. (1, 2, 3)
# 4. Name: The name of the passenger. 
# 5. Sex: The sex of the passenger (male, female)
# 6. Age: The age of the passenger in years. 
# 7. SibSp: How many siblings or spouses the passenger had on board with them. 
# 8. Parch: How many parents or children the passenger had on boad with them. 
# 9. Ticket: The ticket of the passenger. 
# 10. Fare: The fare amount paid by the passenger for the trip. 
# 11. Cabin: The cabin in which the passenger stayed. 
# 12. Embarked: The place from which the passenger embarked (S, C, Q)

# <a id="firstlook"></a>
# Let us have a first look of the dataset. 

# In[2]:


df = pd.read_csv('/kaggle/input/titanic/train.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# Some initial thoughts: <br>
# -> The PassengerId column should not provide us with any useful information regarding survival. I will remove it. <br>
# -> Two columns - Age and Cabin have null values. The Cabin column especially has most values as null. We will try to examine the cabin data and see if we can extract some useful information from it and convert it into a better feature. <br>
# -> The Fare column may have outliers. The max value is around 512 while the 75th percentile is only 31. <br>
# -> Perhaps we can use Passenger class and sex to impute missing values for Age. <br>
# -> Taking an intuitive glance at each column, it seems that (except for Passenger ID) each column provides some useful information in regards to deciding whether the passenger would survive or not. 

# Let me drop the PassengerId column. 

# In[6]:


df = df.drop('PassengerId', axis=1)
df.head(1)


# <h2>EDA and Feature Engineering</h2>
# 
# <a id="EDA"></a>

# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns


# Let's look at the target class first. 

# In[8]:


df['Survived'].value_counts() / df.shape[0]


# In[9]:


plt.pie([0.61, 0.38], labels=['Did Not Survive', 'Survived'])
plt.legend()
plt.show()


# It seems that about 62% belong to the negative class (0), and 38% belong to the positive class (1). This is certainly an imbalance, though not very severe. Upsampling the positive class may help here. XGBoost, which we will be using later on, has a parameter scale_pos_weight which has the effect of giving a higher relative weight to the minority class as compared to the majority class. We will use that to deal with this imbalance.  

# Let's look at Pclass. 

# In[10]:


fig, ax = plt.subplots()
sns.countplot(data=df, x='Pclass', ax = ax, palette='Set1')
ax.bar_label(ax.containers[0])
ax.set_ylim((0, 650))
ax.set_ylabel("Count")
plt.show()


# It seems that most of the passengers belonged to 3rd class. Less than half of that quantity is of 1st class, followed by a slightly less in quantity 2nd class.

# In[11]:


fig, axs = plt.subplots(1, 3)

sns.countplot(data=df[df['Pclass'] == 1], x='Survived', ax=axs[0], palette='Set1')
sns.countplot(data=df[df['Pclass'] == 2], x='Survived', ax=axs[1], palette='Set1')
sns.countplot(data=df[df['Pclass'] == 3], x='Survived', ax=axs[2], palette='Set1')

axs[0].set_title("For Pass. Class-1")
axs[1].set_title("For Pass. Class-2")
axs[2].set_title("For Pass. Class-3")

axs[0].set_ylabel("Count")
axs[1].set_ylabel("")
axs[2].set_ylabel("")

plt.tight_layout()
plt.show()


# It seems that most of the passengers from class 1 survived, while the opposite is true for class 3. For class 2, it is somewhat even. 

# In[12]:


fig, axs = plt.subplots(1, 3)

sns.countplot(data=df[df['Pclass'] == 1], x='Sex', ax=axs[0])
sns.countplot(data=df[df['Pclass'] == 2], x='Sex', ax=axs[1])
sns.countplot(data=df[df['Pclass'] == 3], x='Sex', ax=axs[2])

axs[0].set_title("For Pass. Class-1")
axs[1].set_title("For Pass. Class-2")
axs[2].set_title("For Pass. Class-3")

axs[0].set_ylabel("Count")
axs[1].set_ylabel("")
axs[2].set_ylabel("")

plt.tight_layout()
plt.show()


# For each class, most of the passengers were male. The gap is the most for 3rd class. 

# In[13]:


fig, axs = plt.subplots(1, 3, figsize=(10, 3))

sns.histplot(data=df[df['Pclass'] == 1], x='Age', kde=True, ax=axs[0], color='red')
sns.histplot(data=df[df['Pclass'] == 2], x='Age', kde=True, ax=axs[1], color='blue')
sns.histplot(data=df[df['Pclass'] == 3], x='Age', kde=True, ax=axs[2], color='green')

axs[0].set_title("For Pass. Class-1")
axs[1].set_title("For Pass. Class-2")
axs[2].set_title("For Pass. Class-3")

plt.tight_layout()
plt.show()


# The above shows the distribution of age for each passenger class.

# In[14]:


df.groupby('Pclass')['SibSp'].mean().to_frame().reset_index().rename(columns={'SibSp': 'Av. value of SibSp'})


# In[15]:


df.groupby('Pclass')['Parch'].mean().to_frame().reset_index().rename(columns={'Parch': 'Av. value of Parch'})


# 3rd class passengers on average had more siblings or spouses with them. <br>
# Each class has about the same average amount of parents or children with them. 

# In[16]:


df.groupby('Pclass')['Fare'].median().to_frame().reset_index().rename(columns={'Fare': 'Median val. of Fare'})


# 1st class passengers paid a much higher median fare for their travel, followed by 2nd class passengers then 3rd class passengers. 

# Let's look at Name. 

# In[17]:


df['Name'].nunique()


# We have all unique values for Name. Before deciding to remove this feature, let's see if we can extract some useful info. Maybe we'll find something useful. 

# In[18]:


df.head()['Name']


# Notice how each name has a format: Title, Full Name. Let's extract the title. 

# In[19]:


df['Title'] = df['Name'].str.split(',', expand=True)[0]
df.head(3)


# In[20]:


df.drop('Name', axis=1, inplace=True)


# In[21]:


df['Title'].value_counts().to_frame().reset_index().iloc[:7].rename(columns={'index':'Title', 'Title':'Frequency'})


# The above titles seem to come most frequently. Let's see the survival ratios for each of them. 

# In[22]:


for title in ['Andersson', 'Sage', 'Panula', 'Skoog', 'Carter', 'Goodwin', 'Johnson']:
    print(f"Survival ratio for {title}: {df[df['Title'] == title]['Survived'].mean()}")


# Notice how none of the people with titles Sage, Panula, Skoog or Goodwin survived. We can create a binary feature which checks whether the given passenger has one of these titles or not. Such passengers will be modelled to have a lower chance of survival. 

# In[23]:


df['Title_less_chance'] = df['Title'].apply(lambda title: 1 if title in ['Goodwin', 'Panula', 'Sage', 'Skoog'] else 0)


# In[24]:


df.drop('Title', axis=1, inplace=True)


# In[25]:


df.head()


# Let's look at Sex. 

# In[26]:


fig, ax = plt.subplots(figsize=(3, 3))
sns.countplot(data=df, x='Sex', ax=ax, palette='Set1')
ax.bar_label(ax.containers[0])
ax.set_ylabel("Count")
ax.set_ylim((0, 800))
plt.show()


# Most of the passengers were male. Females constitute about half of the male amount. 

# In[27]:


fig, axs = plt.subplots(1, 2)

sns.countplot(data=df[df['Sex'] == 'male'], x='Survived', ax=axs[0], palette='Set2')
sns.countplot(data=df[df['Sex'] == 'female'], x='Survived', ax=axs[1], palette='Set2')

axs[0].set_title("For Male Passengers")
axs[1].set_title("For Female Passengers")

plt.tight_layout()
plt.show()


# Most of the female passengers survived, while most of the male passengers didn't. 

# In[28]:


fig, ax = plt.subplots()

sns.barplot(data=df, x='Sex', y='Age', estimator=np.median, ax=ax, palette='Set1', ci=None)
ax.set_ylim((0, 50))
ax.bar_label(ax.containers[0])
ax.set_title("Median Age for Males and Females")
plt.show()


# Males and females have about the same median age. 29 for males, 27 for females. 

# Let's look at Age.

# In[29]:


fig, ax = plt.subplots()
sns.histplot(data=df, x='Age', kde=True, color='red', ax=ax)
ax.set_title("Distribution of Age")
plt.show()


# Let's look at SibSp and Parch.

# In[30]:


df['SibSp'].value_counts().to_frame().reset_index().rename(columns={'index':'SibSp', 'SibSp':'Count'})


# Most people did not have any siblings or spouses with them. 

# In[31]:


df.groupby('SibSp')['Survived'].mean().plot(kind='bar', rot=0, ylabel='Proportion Survived', ylim=[0,1], color='brown')


# In[32]:


df['Parch'].value_counts().to_frame().reset_index().rename(columns={'index':'Parch', 'Parch':'Count'})


# Most people did not have any parents or children with them.

# In[33]:


df.groupby('Parch')['Survived'].mean().plot(kind='bar', rot=0, ylabel='Proportion Survived', ylim=[0,1], color='maroon')


# Let's use Parch and SibSp to create a feature called Is_Alone which will take value 1 if the passenger was alone, and 0 if the passenger was not alone. 

# In[34]:


df['Alongside'] = df['Parch'] + df['SibSp']
df['Is_Alone'] = df['Alongside'].apply(lambda x: 1 if x == 0 else 0)


# In[35]:


df.drop('Alongside', axis=1, inplace=True)
df.head()


# I am not quite sure what to do with the Ticket column. I'm going to drop it entirely. 

# In[36]:


df.drop('Ticket', axis=1, inplace=True)


# Let's look at Fare.

# In[37]:


fig, ax = plt.subplots()
sns.histplot(data=df, x='Fare', kde=True, color='red', ax=ax)
ax.set_title("Distribution of Fare")
plt.show()


# Due to outliers in Fare, the distribution is heavily skewed. Let's look at a boxplot. 

# In[38]:


sns.boxplot(data=df, x='Fare')


# Let's deal with the outliers since their presence can affect our model's performance. 

# In[39]:


iqr = df['Fare'].quantile(0.75) - df['Fare'].quantile(0.25)
upper_cap = df['Fare'].quantile(0.75) + 1.5*iqr
lower_cap = max(df['Fare'].quantile(0.25) - 1.5*iqr, 0)
print(upper_cap, lower_cap)


# In[40]:


df['Fare'] = df['Fare'].apply(lambda x: x if x<=upper_cap else upper_cap)


# In[41]:


sns.boxplot(data=df, x='Fare')


# In[42]:


df.head()


# Let's look at Cabin now.

# In[43]:


df['Cabin'].isna().sum()


# There are a lot of null values for Cabin. 

# In[44]:


df['Cabin'].unique()[:5]


# Cabin seems to follow the format of an Alphabet followed by a number. Let me work with these alphabets (perhaps it's Deck?) and remove the numbers. 

# In[45]:


df['Cabin'] = df['Cabin'].apply(lambda x: str(x)[0])
df.head(3)


# In[46]:


df[df['Cabin']!='n'].groupby('Cabin')['Survived'].mean()


# The above shows survival ratios for each Cabin. This information may prove useful to our model. 

# In[47]:


df['Cabin'].value_counts()


# There is only one instance for Cabin T. Let me designate it as null. 

# In[48]:


df.loc[df['Cabin']=='T', 'Cabin'] = 'n'


# Cabins B, D and E have a decent survival proportion. I will create a binary feature for them. 

# In[49]:


df['Cabin_more_chance'] = df['Cabin'].apply(lambda cabin: 1 if cabin in ['B', 'D', 'E'] else 0)
df.drop('Cabin', axis=1, inplace=True)


# In[50]:


df.head()


# Finally, let's look at Embarked. 

# In[51]:


df['Embarked'].value_counts().to_frame()


# In[52]:


sns.countplot(data=df, x='Embarked', hue='Survived', palette='Set1')
plt.legend(['Didn\'t Survive', 'Survived'])


# Let's impute missing values for Age. I will use Pclass and Sex to do so. 

# In[53]:


median_age_groupby_Pclass_sex = df.groupby(['Pclass', 'Sex'])['Age'].median()
median_age_groupby_Pclass_sex


# In[54]:


for Pclass in [1, 2, 3]:
    for Sex in ['female', 'male']:
        df.loc[(df['Pclass']==Pclass)&(df['Sex']==Sex), 'Age'] = df.loc[(df['Pclass']==Pclass)&(df['Sex']==Sex), 'Age'].fillna(median_age_groupby_Pclass_sex[Pclass, Sex])


# In[55]:


df.info()


# Two rows (with indices 61 and 829) have missing embarked values. I'll drop them. 

# In[56]:


df.drop(61, axis=0, inplace=True)
df.drop(829, axis=0, inplace=True)


# <h2> Creating Training and Test Sets, Preprocessing </h2>
# 
# <a id="preprocessing"></a>

# Our dataset now looks like this: 

# In[57]:


df.head()


# In[58]:


X = df.drop('Survived', axis=1)
y = df['Survived']


# We will one-hot encode Sex and Embarked. Pclass is an ordinal attribute so we'll not ohe it. 

# In[59]:


X = pd.get_dummies(X)


# In[60]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape


# In[61]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
X_train[['Age',  'Fare']] = sc.fit_transform(X_train[['Age', 'Fare']])


# In[62]:


X_train.head()


# In[63]:


X_test[['Age',  'Fare']] = sc.transform(X_test[['Age', 'Fare']])


# <h2> Modelling </h2>
# 
# <a id="modelling"></a>

# We will do our modelling using XGBoost, and will use GridSearch to select optimal hyperparameters. 

# In[64]:


import xgboost as xgb
from sklearn.model_selection import GridSearchCV


# In[65]:


y_train.value_counts()


# For our first round of grid search, the param_grid is as in the below cell. Also, a good value of scale_pos_weight (which we discussed before) is (number of majority class instances)/(number of minority class instances) = 439/272 which is approximately 1.5, hence we have included 1.5 as one of the values to try in scale_pos_weight's list. 

# In[66]:


param_grid_1 = {
    'max_depth': [3, 6, 9, 15],
    'learning_rate': [0.3, 0.1, 0.01], 
    'gamma': [0, 0.25, 1.0], 
    'lambda': [0, 1.0, 10.0],   
    'scale_pos_weight': [1, 1.5, 2]
}

gs_model = GridSearchCV(estimator=xgb.XGBClassifier(objective='binary:logistic', seed=42), param_grid=param_grid_1, scoring='roc_auc', n_jobs=-1, cv=5)

gs_model.fit(X_train, y_train)


# In[67]:


gs_model.best_params_


# In[68]:


gs_model.best_score_


# For our second round of grid search, the param_grid is as follows:

# In[69]:


param_grid_2 = {
    'max_depth': [15, 20],
    'learning_rate': [0.3, 0.5, 0.7], 
    'gamma': [1.0, 1.5, 3.0], 
    'lambda': [10.0, 15.0, 20.0],   
    'scale_pos_weight': [1.5]
}

gs_model = GridSearchCV(estimator=xgb.XGBClassifier(objective='binary:logistic', seed=42), param_grid=param_grid_2, scoring='roc_auc', n_jobs=-1, cv=5)

gs_model.fit(X_train, y_train)


# In[70]:


gs_model.best_params_


# In[71]:


gs_model.best_score_


# In[72]:


clf_xgb = xgb.XGBClassifier(seed=42, objective='binary:logistic', gamma=1.0, reg_lambda=10.0, learning_rate=0.3, max_depth=15, scale_pos_weight=1.5)


# In[73]:


clf_xgb.fit(X_train, y_train)


# In[74]:


y_prob = clf_xgb.predict_proba(X_test)[:, 1]


# In[75]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_prob)
print(roc_auc_score(y_test, y_prob)) 


# In[76]:


plt.subplots(1, figsize=(6,6))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# <h2> Making the Submission </h2>
# 
# <a id="sub"></a>

# Now let's predict on the given test set and make our submission. 

# In[77]:


test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.head()


# In[78]:


test['Title'] = test['Name'].apply(lambda x: x.split(',')[0])
test['Title_less_chance'] = test['Title'].apply(lambda title: 1 if title in ['Goodwin', 'Panula', 'Sage', 'Skoog'] else 0)


# In[79]:


test.drop('Name', axis=1, inplace=True)
test.drop('Title', axis=1, inplace=True)


# In[80]:


test.drop('Ticket', axis=1, inplace=True)


# In[81]:


test['Cabin'].fillna('n', inplace=True)
test['Cabin_more_chance'] = test['Cabin'].apply(lambda cabin: 1 if cabin in ['B', 'D', 'E'] else 0)
test.drop('Cabin', axis=1, inplace=True)


# In[82]:


test_noid = test.drop('PassengerId', axis=1)
for Pclass in [1, 2, 3]:
    for Sex in ['female', 'male']:
        test_noid.loc[(test_noid['Pclass']==Pclass)&(test_noid['Sex']==Sex), 'Age'] = test_noid.loc[(test_noid['Pclass']==Pclass)&(test_noid['Sex']==Sex), 'Age'].fillna(median_age_groupby_Pclass_sex[Pclass, Sex])


# In[83]:


test_noid['Fare'] = test_noid['Fare'].fillna(df['Fare'].median())


# In[84]:


test_noid.info()


# In[85]:


test_noid['Is_Alone'] = test_noid['SibSp'] + test_noid['Parch']
test_noid['Is_Alone'] = test_noid['Is_Alone'].apply(lambda total: 1 if total==0 else 0)

test_noid = pd.get_dummies(test_noid)

test_noid[['Age',  'Fare']] = sc.transform(test_noid[['Age', 'Fare']])

#Reordering feature names to match those of our X_train
test_noid = test_noid[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Title_less_chance', 'Is_Alone', 'Cabin_more_chance', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
test_noid.head()


# In[86]:


predictions = clf_xgb.predict(test_noid)
predictions


# In[87]:


submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})


# In[88]:


submission


# In[89]:


submission.to_csv('submission.csv', index=False)


# <h2> Conclusion </h2>
# 
# <a id="conclusion"></a>
# 
# 

# In this notebook, we worked with the Titanic - Machine Learning from Disaster dataset. We performed EDA, along with feature-engineering, followed by modelling using XGBoost whose optimal hyperparameters we obtained using grid search.
# 
# <br>
# 
# <b>Thank you for reading this notebook. Do upvote it if you liked it. I would be very happy to receive any suggestions on how I could improve the notebook!</b>

# In[ ]:




