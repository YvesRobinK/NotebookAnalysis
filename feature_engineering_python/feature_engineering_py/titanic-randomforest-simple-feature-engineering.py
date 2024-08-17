#!/usr/bin/env python
# coding: utf-8

# # Importing The Dataset

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble  import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score


# In[2]:


df_train = pd.read_csv('../input/titanic/train.csv')
df_test = pd.read_csv('../input/titanic/test.csv')


# In[3]:


print(df_train.shape)
print(df_test.shape)


# In[4]:


print(df_train.info())
df_train.head(5)


# In[5]:


print(df_test.info())
df_test.head(5)


# In[6]:


print(df_train.isnull().sum())


# In[7]:


print(df_test.isnull().sum())


# ## Missing values in Age

# It can be seen that the correlation between Pclass and age is high.
# Group ages by passenger class.

# In[8]:


df_corr = df_train.corr()
df_corr["Age"].sort_values(ascending=False)


# It can be seen that the median age for each Pclass differs depending on the gender.

# In[9]:


# Copied from https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial?scriptVersionId=27280410&cellId=14

age_by_pclass_sex = df_train.groupby(['Sex', 'Pclass']).median()['Age']

for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
print('Median age of all passengers: {}'.format(df_train['Age'].median()))


# In[10]:


df_train['Age'] = df_train.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))


# In[11]:


df_test['Age'] = df_test.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))


# ## Missing values in Embarked

# In[12]:


df_train[df_train['Embarked'].isnull()]


# According to google, Mrs.George Nelson (Martha Evelyn), who embarked from S (Southampton) with Amelie.

# In[13]:


df_train['Embarked'] = df_train['Embarked'].fillna('S')


# ## Missing values in Fare

# Fare has a large correlation with Pclass.

# In[14]:


df_corr["Fare"].sort_values(ascending=False)


# In[15]:


df_test[df_test['Fare'].isnull()]


# In[16]:


mid_fare = df_train.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
mid_fare


# In[17]:


df_test['Fare'] = df_test['Fare'].fillna(mid_fare)


# ## Missing values in Cabin

# In[18]:


df_train = df_train.drop('Cabin', axis=1)
df_test = df_test.drop('Cabin', axis=1)


# In[19]:


print(df_train.isnull().sum())


# In[20]:


print(df_test.isnull().sum())


# ## Target Distribution

# In[21]:


survived_count = df_train['Survived'].value_counts()[1]
not_survived_count = df_train['Survived'].value_counts()[0]


# In[22]:


print('survived : ' + str(survived_count))
print('not_survived : ' + str(not_survived_count))


# In[23]:


survived_per = survived_count / df_train.shape[0] * 100
not_survived_per = not_survived_count / df_train.shape[0] * 100


# In[24]:


print('survived % : ' + str(survived_per))
print('not_survived % : ' + str(not_survived_per))


# In[25]:


plt.figure(figsize=(10, 8))
sns.countplot(df_train['Survived'])

plt.xlabel('Survival', size=15, labelpad=15)
plt.ylabel('Passenger Count', size=15, labelpad=15)
plt.xticks((0, 1), ['Not Survived ({0:.2f}%)'.format(not_survived_per), 'Survived ({0:.2f}%)'.format(survived_per)])
plt.tick_params(axis='x', labelsize=13)
plt.tick_params(axis='y', labelsize=13)

plt.show()


# # EDA

# ## Correlations

# In[26]:


df_train = df_train.drop(['PassengerId'], axis=1)
df_test = df_test.drop(['PassengerId'], axis=1)


# In[27]:


plt.figure(figsize = (15,9))
sns.heatmap(df_train.corr(), annot = True, cmap='coolwarm')


# In[28]:


plt.figure(figsize = (15,9))
sns.heatmap(df_test.corr(), annot = True, cmap='coolwarm')


# ## Target Distribution in Features

# ### Copied from https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial?scriptVersionId=27280410&cellId=38

# In[29]:


cont_features = ['Age', 'Fare']
surv = df_train['Survived'] == 1

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 20))
plt.subplots_adjust(right=1.5)

for i, feature in enumerate(cont_features):    
    # Distribution of survival in feature
    sns.distplot(df_train[~surv][feature], label='Not Survived', hist=True, color='#e74c3c', ax=axs[0][i])
    sns.distplot(df_train[surv][feature], label='Survived', hist=True, color='#2ecc71', ax=axs[0][i])
    
    # Distribution of feature in dataset
    sns.distplot(df_train[feature], label='Training Set', hist=False, color='#e74c3c', ax=axs[1][i])
    sns.distplot(df_test[feature], label='Test Set', hist=False, color='#2ecc71', ax=axs[1][i])
    
    axs[0][i].set_xlabel('')
    axs[1][i].set_xlabel('')
    
    for j in range(2):        
        axs[i][j].tick_params(axis='x', labelsize=20)
        axs[i][j].tick_params(axis='y', labelsize=20)
    
    axs[0][i].legend(loc='upper right', prop={'size': 20})
    axs[1][i].legend(loc='upper right', prop={'size': 20})
    axs[0][i].set_title('Distribution of Survival in {}'.format(feature), size=20, y=1.05)

axs[1][0].set_title('Distribution of {} Feature'.format('Age'), size=20, y=1.05)
axs[1][1].set_title('Distribution of {} Feature'.format('Fare'), size=20, y=1.05)
        
plt.show()


# In[30]:


cat_features = ['Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp']

fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(20, 20))
plt.subplots_adjust(right=1.5, top=1.25)

for i, feature in enumerate(cat_features, 1):    
    plt.subplot(2, 3, i)
    sns.countplot(x=feature, hue='Survived', data=df_train)
    
    plt.xlabel('{}'.format(feature), size=20, labelpad=15)
    plt.ylabel('Passenger Count', size=20, labelpad=15)    
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    
    plt.legend(['Not Survived', 'Survived'], loc='upper center', prop={'size': 18})
    plt.title('Count of Survival in {} Feature'.format(feature), size=20, y=1.05)

plt.show()


# ## Feature Engineering

# ### This section is Copied from https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial?scriptVersionId=27280410&cellId=45

# In[31]:


# df_feature = df_train


# In[32]:


# df_feature['Fare'] = pd.qcut(df_feature['Fare'], 13)

# fig, axs = plt.subplots(figsize=(22, 9))
# sns.countplot(x='Fare', hue='Survived', data=df_feature)

# plt.xlabel('Fare', size=15, labelpad=20)
# plt.ylabel('Passenger Count', size=15, labelpad=20)
# plt.tick_params(axis='x', labelsize=10)
# plt.tick_params(axis='y', labelsize=15)

# plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})
# plt.title('Count of Survival in {} Feature'.format('Fare'), size=15, y=1.05)

# plt.show()


# In[33]:


# df_feature['Age'] = pd.qcut(df_feature['Age'], 10)

# fig, axs = plt.subplots(figsize=(22, 9))
# sns.countplot(x='Age', hue='Survived', data=df_feature)

# plt.xlabel('Age', size=15, labelpad=20)
# plt.ylabel('Passenger Count', size=15, labelpad=20)
# plt.tick_params(axis='x', labelsize=15)
# plt.tick_params(axis='y', labelsize=15)

# plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})
# plt.title('Survival Counts in {} Feature'.format('Age'), size=15, y=1.05)

# plt.show()


# ### Encoding Ticket 

# According to the graph below, groups with 2,3 and 4 members had a higher survival rate.

# In[34]:


df_train['Ticket_Frequency'] = df_train.groupby('Ticket')['Ticket'].transform('count')
df_test['Ticket_Frequency'] = df_test.groupby('Ticket')['Ticket'].transform('count')


# In[35]:


df_train = df_train.drop(['Ticket'], axis=1)
df_test = df_test.drop(['Ticket'], axis=1)


# In[36]:


df_train.head()


# In[37]:


df_test.head()


# In[38]:


fig, axs = plt.subplots(figsize=(12, 9))
sns.countplot(x='Ticket_Frequency', hue='Survived', data=df_train)

plt.xlabel('Ticket Frequency', size=15, labelpad=20)
plt.ylabel('Passenger Count', size=15, labelpad=20)
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})
plt.title('Count of Survival in {} Feature'.format('Ticket Frequency'), size=15, y=1.05)

plt.show()


# ### Title & Is Married

# In[39]:


df_train['Title'] = df_train['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
df_train['Is_Married'] = 0
df_train['Is_Married'].loc[df_train['Title'] == 'Mrs'] = 1


# In[40]:


df_test['Title'] = df_test['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
df_test['Is_Married'] = 0
df_test['Is_Married'].loc[df_test['Title'] == 'Mrs'] = 1


# In[41]:


df_train['Title'] = df_train['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
df_train['Title'] = df_train['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')


# In[42]:


df_test['Title'] = df_test['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
df_test['Title'] = df_test['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')


# In[43]:


df_train = df_train.drop(['Name'], axis=1)
df_test = df_test.drop(['Name'], axis=1)


# In[44]:


df_train.head()


# # OneHotEncoding

# In[45]:


encoder_sex = OneHotEncoder(handle_unknown='ignore') #"ignore"にするの大事
encoder_sex.fit(df_train["Sex"].values.reshape(-1, 1))

encoder_train_sex = encoder_sex.transform(df_train["Sex"].values.reshape(-1, 1))
encoder_test_sex = encoder_sex.transform(df_test["Sex"].values.reshape(-1, 1))

encoder_train_sex = pd.DataFrame(encoder_train_sex.toarray().astype('int64'), columns=encoder_sex.categories_)
encoder_test_sex = pd.DataFrame(encoder_test_sex.toarray().astype('int64'), columns=encoder_sex.categories_)


# In[46]:


encoder_Embarked = OneHotEncoder(handle_unknown='ignore') #"ignore"にするの大事
encoder_Embarked.fit(df_train["Embarked"].values.reshape(-1, 1))

encoder_train_Embarked = encoder_Embarked.transform(df_train["Embarked"].values.reshape(-1, 1))
encoder_test_Embarked = encoder_Embarked.transform(df_test["Embarked"].values.reshape(-1, 1))

encoder_train_Embarked = pd.DataFrame(encoder_train_Embarked.toarray().astype('int64'), columns=encoder_Embarked.categories_)
encoder_test_Embarked = pd.DataFrame(encoder_test_Embarked.toarray().astype('int64'), columns=encoder_Embarked.categories_)


# In[47]:


encoder_Title = OneHotEncoder(handle_unknown='ignore') #"ignore"にするの大事
encoder_Title.fit(df_train["Title"].values.reshape(-1, 1))

encoder_train_Title = encoder_Title.transform(df_train["Title"].values.reshape(-1, 1))
encoder_test_Title = encoder_Title.transform(df_test["Title"].values.reshape(-1, 1))

encoder_train_Title = pd.DataFrame(encoder_train_Title.toarray().astype('int64'), columns=encoder_Title.categories_)
encoder_test_Title = pd.DataFrame(encoder_test_Title.toarray().astype('int64'), columns=encoder_Title.categories_)


# In[48]:


encoder_Pclass = OneHotEncoder(handle_unknown='ignore') #"ignore"にするの大事
encoder_Pclass.fit(df_train["Pclass"].values.reshape(-1, 1))

encoder_train_Pclass = encoder_Pclass.transform(df_train["Pclass"].values.reshape(-1, 1))
encoder_test_Pclass = encoder_Pclass.transform(df_test["Pclass"].values.reshape(-1, 1))

columns = ['Pclass_1', 'Pclass_2', 'Pclass_3']

encoder_train_Pclass = pd.DataFrame(encoder_train_Pclass.toarray().astype('int64'), columns=columns)
encoder_test_Pclass = pd.DataFrame(encoder_test_Pclass.toarray().astype('int64'), columns=columns)


# In[49]:


encoder_test_Pclass.tail()


# In[50]:


encoder_test_Pclass.shape


# In[51]:


df_train = pd.concat([df_train, encoder_train_Pclass, encoder_train_Title, encoder_train_Embarked, encoder_train_sex], axis=1)
df_test = pd.concat([df_test, encoder_test_Pclass, encoder_test_Title, encoder_test_Embarked, encoder_test_sex], axis=1)


# In[52]:


df_train = df_train.drop(['Pclass', 'Sex', 'Embarked', 'Title'], axis=1)
df_test = df_test.drop(['Pclass', 'Sex', 'Embarked', 'Title'], axis=1)


# In[53]:


df_train.head()


# In[54]:


df_test.tail()


# In[55]:


print(df_train.isnull().sum())


# In[56]:


print(df_test.isnull().sum())


# # Model

# In[57]:


df_train.shape


# In[58]:


df_test.shape


# In[59]:


st = StandardScaler()
columns= ['Fare', 'Age', 'SibSp', 'Parch', 'Ticket_Frequency']
st = st.fit(df_train[columns])
df_train[columns] = st.transform(df_train[columns])
df_test[columns] = st.transform(df_test[columns])


# In[60]:


df_train.head()


# In[61]:


df_test.head()


# In[62]:


train_x = df_train.drop(['Survived'], axis = 1)
train_y = df_train['Survived']


# In[63]:


train_x.shape


# In[64]:


rf = RandomForestClassifier(criterion='gini',
                            n_estimators=1750,
                            max_depth=7,
                            min_samples_split=6,
                            min_samples_leaf=6,
                            max_features='auto',
                            oob_score=True,
                            random_state=0,
                            n_jobs=-1,
                            verbose=1)
rf.fit(train_x, train_y)


# In[65]:


rf_pred = rf.predict(df_test)


# In[66]:


df_submission = pd.read_csv('../input/titanic/test.csv')


# In[67]:


submission = pd.DataFrame(columns=['PassengerId', 'Survived'])


# In[68]:


submission['PassengerId'] = df_submission['PassengerId']
submission['Survived'] = rf_pred


# In[69]:


submission.head()


# In[70]:


submission.shape


# In[71]:


submission.to_csv('submissions.csv', header=True, index=False)


# In[ ]:




