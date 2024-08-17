#!/usr/bin/env python
# coding: utf-8

# ### Import libraries and check the data

# In[1]:


import numpy as np                                   
import pandas as pd                                    
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score           # Metric : Accuracy 

import re
import os
import gc
import warnings
warnings.filterwarnings('ignore')

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ![](https://cdn.pixabay.com/photo/2018/05/14/20/46/ship-3401500_960_720.jpg)

# 
# ### 1. Load the train and test data

# #### An Explanation of each column<br><br>
# - **PassengerId:** the index for each passenger<br>
# - **Pclass:** the first/second/third class (first class is the best quality room)<br>
# - **Name:** the name of each person (including titles)<br>
# - **Sex:** the gender of passenger<br>
# - **Age:** each person's age<br>
# - **SibSp:** the number of siblings or spouse<br>
# - **Parch:** the number of Parents or child<br>
# - **Ticket:** the printed words in ticket<br>
# - **Fare:** paid money<br>
# - **Cabin:** the room number<br>
# - **Embarked:** the location of embarked<br>
# - **Survived:** whether the person live<br>

# In[2]:


PATH = "/kaggle/input/titanic/"
train_df = pd.read_csv(PATH + 'train.csv')
test_df = pd.read_csv(PATH + 'test.csv')


# In[3]:


## Check the train data and columns
print(train_df.shape)
print()
train_df.head()


# In[4]:


## Check the test data and columns
print(test_df.shape)
print()
test_df.head()


# - The 'train_df' has 891 rows and 12 columns, but 'test_df' has 418 rows and 11 columns!<br><br> The difference in rows is okay due to the limitation of number of the data, but columns..?? hmm.. Which column disappeared from the 'test_df'?

# In[5]:


for col in train_df.columns:
    if not col in test_df.columns:
        print('The column', str(col), "is not exist in the test_df.")


# Only the 'train_df' has the column 'Survived'.<br>Thus, the 'Survived' column is a target column that we should be predicted, the label.<br><br>
# As many people know, not much survivors in the Titanic disaster.<br>
# Is 'Survived' balanced?

# In[6]:


plt.hist(train_df['Survived'], bins=3, align='mid',histtype='bar')
plt.title('the # of Not-Survived and Survived')
plt.xlim(-0.2, 1.2)
plt.ylim(0, 600)
plt.show()


# In[7]:


print('The number of Not-Survived is ', len(train_df[train_df['Survived']==0]))
print('The number of Survived is ', len(train_df[train_df['Survived']==1]))
print('The apart between label 0 and 1 is ',len(train_df[train_df['Survived']==0]) - len(train_df[train_df['Survived']==1]))


# The difference between not-survived and survived is 207 apart. It's not a large imbalance and we have some sampling algorithm for solve this problem!<br>
# We will do the Upsampling to solve the imbalance after preprocessing.

# In[8]:


# this part of code from : https://www.youtube.com/watch?v=35j4pCe-fQk
change_age_range_survival_ratio = []

for i in range(1,80):
    change_age_range_survival_ratio.append(train_df[train_df['Age'] < i]['Survived'].sum() / len(train_df[train_df['Age'] < i]['Survived']))
    
plt.figure(figsize=(7,7))
plt.plot(change_age_range_survival_ratio)
plt.title('Survival rate change depending on range of age', y=1.02)
plt.ylabel('survival rate')
plt.xlabel('range of age(0-x)')
plt.show()


# ### 2. Check the null / duplicates values
# - Since the train and test data have to be processed in the same way, we will conduct the exact same process for those two dataframes.

# In[9]:


train_df.isnull().sum()


# In[10]:


test_df.isnull().sum()


# - two nulls in 'Embarked' in train_df and one null in 'Fare' in test_df, but we can handle these little nulls.<br><br>
# However, the null values in 'Age' and 'Cabin', they look quite large. How many are they?

# In[11]:


embark_dict = {}
for em in train_df['Embarked']:
    if em not in embark_dict.keys():
        embark_dict[em] = 1
    else:
        embark_dict[em] += 1
        
embark_dict                   # the most frequent embarked place is 'S'


# In[12]:


train_df['Embarked'].fillna('S', inplace=True)
train_df['Embarked'].isnull().sum()


# In[13]:


print('------The percentage of null in train_df------')
print('Null in train_df.Age :', str(train_df['Age'].isnull().sum()/len(train_df) * 100).split('.')[0], "%")
print('Null in train_df.Cabin :', str(train_df['Cabin'].isnull().sum()/len(train_df) * 100).split('.')[0], "%")
print()
print('-------The percentage of null in test_df-------')
print('Null in test_df.Age :', str(test_df['Age'].isnull().sum()/len(test_df) * 100).split('.')[0], "%")
print('Null in test_df.Cabin :', str(test_df['Cabin'].isnull().sum()/len(test_df) * 100).split('.')[0], "%")


# The null values are dominant in 'Cabin' column while 'Age' column has a fewer amount of NaN.<br>
# Moreover, we have a reference column for 'Age' that is the 'Name', but not for 'Cabin'.<br><br>
# - the column 'PassengerId' is only an index for the row. That looks not very helpful to analysis.<br>
# - the column 'Ticket' is a sequence of letters that has no meaning.<br>
# - the column 'Fare' looks not helpful,too. Because we can classify them with 'Pclass'.
# <br><br>
# So we simply drop out them.<br>

# In[14]:


train_df.drop(columns=['Cabin','PassengerId','Ticket','Fare'], inplace=True)
test_df.drop(columns=['Cabin','PassengerId','Ticket','Fare'], inplace=True)


# In[15]:


print('the former shape of train/test: ',train_df.shape, test_df.shape)
train_df = train_df.drop_duplicates()
test_df = test_df.drop_duplicates()
print('the latter shape of train/test: ', train_df.shape, test_df.shape)


# Good news! They have no duplicate value.

# ### 3. Change categorical values to numeric values
# In model, the type of input data should be numeric values.<br>
# - As you can see below information, column 'Sex','Embarked', and 'Name' have object values. Let's change them!

# In[16]:


train_df.info()


# In[17]:


test_df.info()


# In[18]:


# female: 1, male: 0
train_df['Sex'] = train_df['Sex'].apply(lambda x: 1 if x == 'female' else 0)
test_df['Sex'] = test_df['Sex'].apply(lambda x: 1 if x == 'female' else 0)
train_df['Sex'].unique()


# In[19]:


# sklearn.preprocessing.LabelEncoder change the type of categories, str to int
encoder = LabelEncoder()
train_df['Embarked'] = encoder.fit_transform(train_df['Embarked'].astype(str))
test_df['Embarked'] = encoder.transform(test_df['Embarked'].astype(str))


# In[20]:


train_df.head()


# ### 4. Pre-process 'Name' for change null values to others in 'Age'

# In[21]:


def get_title(name):
    title = ''
    # Major = Mr. Col = Mr. Capt = Mr. 
    titles = ['Mr','Mrs','Miss','Master','Rev','Dr','Ms','Mme','Major','Col',
              'Capt','Countess','Mlle','Sir','Jonkheer']
    
    name = name.split()
    for n in name:
        n = n.strip()
        n = re.sub(r"[^A-Za-z]", " ", n)
        n = n.strip()
        if n in titles:
            if n in ['Mr','Rev','Dr','Major','Col','Capt','Sir']:
                title = n
                return 'Mr'
            elif n in ['Mrs','Mme','Ms','Countess']: # Mme: Madame
                title = n
                return 'Mrs'
            elif n in ['Miss','Mlle']:                # Mlle: Mademoiselle
                title = n
                return 'Miss'
            else:
                return n.strip()
        
    if title == '':
        return 'Mr'     #  the remained names not checked a title are 'John','Don'..etc


# In[22]:


tmp_titles = [0] * len(train_df)
for idx, name in enumerate(train_df.Name):  
    title = get_title(name)
    tmp_titles[idx] = title

train_df['title'] = tmp_titles
    
tmp_titles = [0] * len(test_df)
for idx, name in enumerate(test_df['Name']):
    title = get_title(name)
    tmp_titles[idx] = title
    
test_df['title'] = tmp_titles
del tmp_titles
gc.collect()


# In[23]:


train_df.drop(columns=['Name'], inplace=True)
test_df.drop(columns=['Name'], inplace=True)
train_df['title'].unique()


# In[24]:


train_df.head()


# Finally, we can change the null to integers in 'Age' with 'title'!

# In[25]:


print('the number of nulls in train_df:', train_df['Age'].isnull().sum())
print('the number of nulls in test_df:', test_df['Age'].isnull().sum())


# Let's fill the null with the mean value of each 'Mr', 'Mrs', 'Miss', and 'Master'.

# In[26]:


tr_title_mean = train_df['Age'].groupby(train_df['title']).mean()
te_title_mean = test_df['Age'].groupby(test_df['title']).mean()
title_mean = (te_title_mean + tr_title_mean)/2


# In[27]:


train_df.loc[(train_df['Age'].isnull()) & (train_df['title']=='Mr'), 'Age'] = int(title_mean['Mr'])
train_df.loc[(train_df['Age'].isnull()) & (train_df['title']=='Mrs'), 'Age'] = int(title_mean['Mrs'])
train_df.loc[(train_df['Age'].isnull()) & (train_df['title']=='Master'), 'Age'] = int(title_mean['Master'])
train_df.loc[(train_df['Age'].isnull()) & (train_df['title']=='Miss'), 'Age'] = int(title_mean['Miss'])

test_df.loc[(test_df['Age'].isnull()) & (test_df['title']=='Mr'), 'Age'] = int(title_mean['Mr'])
test_df.loc[(test_df['Age'].isnull()) & (test_df['title']=='Mrs'), 'Age'] = int(title_mean['Mrs'])
test_df.loc[(test_df['Age'].isnull()) & (test_df['title']=='Master'), 'Age'] = int(title_mean['Master'])
test_df.loc[(test_df['Age'].isnull()) & (test_df['title']=='Miss'), 'Age'] = int(title_mean['Miss'])


# In[28]:


train_df.isnull().sum()


# All the null values are gone!<br>
# Make the categories in the column 'title' to integers!

# In[29]:


encoder = LabelEncoder()
train_df['title'] = encoder.fit_transform(train_df['title'].astype(str))
test_df['title'] = encoder.transform(test_df['title'].astype(str))


# ### 5. Feature Engineering

# In[30]:


train_df['familysize'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['familysize'] = test_df['SibSp'] + test_df['Parch'] + 1


# In[31]:


train_df.drop(columns=['SibSp','Parch'], inplace=True)
test_df.drop(columns=['SibSp','Parch'], inplace=True)


# Let's check the correlation between the features :)

# In[32]:


sns.heatmap(train_df.corr(), annot=True)


# In a linear correlation, the column 'Sex' affects 0.54% to predict whether survive.<br>
# After modeling, we will see the non-liear correlation of the features in model.<br>
# Now apply np.log1p(scaling) to train and test data. (This part is an optional)

# In[33]:


train_df['Age'] = np.log1p(train_df['Age'])
test_df['Age'] = np.log1p(test_df['Age'])


# ### 6. Upsampling for solve the imbalance
# Let's upsampling with 'Survived'==1

# In[34]:


concat_train = train_df.loc[train_df['Survived']==1].iloc[:207, :]
train_df = pd.concat([train_df, concat_train], axis=0)


# In[35]:


len(train_df[train_df['Survived']==1]), len(train_df[train_df['Survived']==0])


# In[36]:


y_train = train_df['Survived']
train_df.drop(columns=['Survived'], inplace=True)


# In[37]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(train_df, y_train)


# ### 7. Modeling
# The basic modeling of XGBoost and LGBM<br>
# The result submission of XGBoost is slight better than LGBM's.

# In[38]:


xgb_pbounds = {'n_estimators':2000,
              'learning_rate':0.001,
              'max_depth':10,
              'gamma':0.9,
              'subsample':0.8, 
              'colsample_bytree':0.8,
              'boosting_type':'rf',
              'reg_lambda':10}


# In[39]:


xgb_model = xgb.XGBClassifier(**xgb_pbounds)


# In[40]:


xgb_model.fit(train_df, y_train)
preds = xgb_model.predict(test_df)


# In[41]:


sub_df = pd.read_csv(PATH + 'gender_submission.csv')
sub_df['Survived'] = preds


# In[42]:


sub_df.to_csv('xgb_submission.csv', index=False)


# In[43]:


lgb_pbounds = {'n_estimators':2000,
              'learning_rate':0.001,
              'max_depth':10,
              'subsample':0.8,
              'colsample_bytree':1.0,
              'reg_lambda':10}


# In[44]:


lgb_model = lgb.LGBMClassifier(**lgb_pbounds)


# In[45]:


lgb_model.fit(train_df, y_train)
preds = lgb_model.predict(test_df)


# In[46]:


sub_df['Survived'] = preds
sub_df.to_csv('lgb_submission.csv', index=False)


# ### 8. Feature importance
# The result of LGBM feature importance is almost the same while the result of XGBoost is very different. The one reason in StackOverflow was 'plot_importance' uses 'weight' but 'XGBoost' uses 'gain'. If you want to more about this please check here out. (https://stackoverflow.com/questions/58984474/how-to-get-correct-feature-importance-plot-in-xgboost) 

# In[47]:


# https://steadiness-193.tistory.com/264
feat = pd.DataFrame(xgb_model.feature_importances_, index=train_df.columns, columns=['importances']).sort_values(by='importances', ascending=False)
feat = feat.reset_index()
fig, ax = plt.subplots()
ax = sns.barplot(y=feat['index'], x=feat['importances'])
fig.set_size_inches(10, 5)


# In[48]:


from xgboost import plot_importance
fig, ax = plt.subplots(figsize=(10,5))
plot_importance(xgb_model, ax=ax)
plt.show()


# In[49]:


# https://steadiness-193.tistory.com/264
feat = pd.DataFrame(lgb_model.feature_importances_, index=train_df.columns, columns=['importances']).sort_values(by='importances', ascending=False)
feat = feat.reset_index()
fig, ax = plt.subplots()
ax = sns.barplot(y=feat['index'], x=feat['importances'])
fig.set_size_inches(10, 5)


# In[50]:


from lightgbm import plot_importance
fig, ax = plt.subplots(figsize=(10,5))
plot_importance(lgb_model, ax=ax)
plt.show()

