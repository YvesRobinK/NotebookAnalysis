#!/usr/bin/env python
# coding: utf-8

# ### Progress logs
# 
# * Rank 2000+: My first submission after learning some(a lot of) courses
# * Rank 1000+: Switching model from DesionTree -> XgBoost
# * Rank 800+: Using Pipeline to fill missing value with most_frequent and do standardize number features
# * Rank 600+: Switching to lightgbm and using gridsearch to find the parameters automatically
# * Rank 500+: Add new features like age category and traveling group size after checking [spaceship-titanic-a-complete-guide](https://www.kaggle.com/code/samuelcortinhas/spaceship-titanic-a-complete-guide)
# * Rank 88: Do more EDA using seaborn and add more features about consumption and cabin

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


# In[2]:


train = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
test = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')
combined = pd.concat([train, test], axis=0)


# In[3]:


train.shape, test.shape, combined.shape


# In[4]:


train.head()


# In[5]:


train_X = train.copy()
train_Y = train_X.pop('Transported')


# ## View the data

# In[6]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.4)


# ### Passenger Group

# In[7]:


def addGroupSizeFeat (df):
    my_df = df.copy()
    group = my_df.PassengerId.str.split('_', expand=True)[0]
    group_value_counts = group.value_counts()
    my_df['GroupSize'] = group.apply(lambda x: group_value_counts[x]).astype(int)
    return my_df

plt.figure(figsize=(10,4))
sns.histplot(data=addGroupSizeFeat(train), x='GroupSize', hue='Transported', binwidth=1, kde=True)
plt.xlabel('Group Size')


# In[8]:


def addIsSoleFeat (df):
    my_df = df.copy()
    group = my_df.PassengerId.str.split('_', expand=True)[0]
    group_value_counts = group.value_counts()
    my_df['IsSole'] = group.apply(lambda x: group_value_counts[x] == 1).astype('category')
    return my_df

plt.figure(figsize=(10,4))
sns.countplot(data=addIsSoleFeat(train), x='IsSole', hue='Transported')
plt.xlabel('Is Traveling Alone')


# ### Other Category features

# In[9]:


for feat in ['HomePlanet', 'Destination', 'CryoSleep', 'VIP']:  
    plt.figure(figsize=(10,4))
    sns.countplot(data=train, x=feat, hue='Transported')
    plt.xlabel(feat)


# In[10]:


train.groupby(['Destination','HomePlanet'])['HomePlanet'].size().unstack()


# ### Age

# In[11]:


plt.figure(figsize=(10,4))
sns.histplot(data=train, x='Age', hue='Transported', binwidth=2, kde=True)
plt.xlabel('Age (years)')


# In[12]:


def addYoungerThan20 (df):
    my_df = df.copy()
    my_df['IsYougerThan20'] = my_df['Age'] < 20
    return my_df

plt.figure(figsize=(10,4))
sns.countplot(data=addYoungerThan20(train), x='IsYougerThan20', hue='Transported')
plt.xlabel('Is Younger Than 20')


# ### Consumption

# In[13]:


exp_feats=['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Plot expenditure features
fig=plt.figure(figsize=(10,20))
for i, var_name in enumerate(exp_feats):
    # Left plot
    ax=fig.add_subplot(5,2,2*i+1)
    sns.histplot(data=train, x=var_name, axes=ax, bins=30, kde=False, hue='Transported')
    ax.set_title(var_name)
    
    # Right plot (truncated)
    ax=fig.add_subplot(5,2,2*i+2)
    sns.histplot(data=train, x=var_name, axes=ax, binwidth=2000, kde=False, hue='Transported')
    plt.ylim([0,20])
    ax.set_title(var_name)
fig.tight_layout()  # Improves appearance a bit
plt.show()


# In[14]:


consume_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
consume_more_for_alive = ['FoodCourt', 'ShoppingMall']
consume_more_for_die = ['RoomService', 'Spa', 'VRDeck']

def addTotalConsumption (df):
    my_df = df.copy()
    my_df['ConsumptionTotal'] = my_df.loc[:, consume_feats].sum(axis=1)
    return my_df

def addAverageConsumption (df):
    my_df = df.copy()
    my_df['ConsumptionAverage'] = my_df.loc[:, consume_feats].mean(axis=1)
    return my_df

def addTotalMoreAliveConsumption (df):
    my_df = df.copy()
    my_df['ConsumptionTotalMoreAlive'] = my_df.loc[:, consume_more_for_alive].sum(axis=1)
    return my_df

def addTotalMoreDieConsumption (df):
    my_df = df.copy()
    my_df['ConsumptionTotalMoreDie'] = my_df.loc[:, consume_more_for_die].sum(axis=1)
    return my_df

def addTotalMoreAliveConsumptionOver3k (df):
    my_df = df.copy()
    my_df['ConsumptionTotalMoreAliveMore3K'] = my_df.loc[:, consume_more_for_alive].sum(axis=1) >= 3000
    return my_df

def addIsConsumedMoreDie (df):
    my_df = df.copy()
    my_df['IsConsumedMoreDie'] = my_df.loc[:, consume_more_for_die].sum(axis=1) >0
    return my_df

def addIsConsumed (df):
    my_df = df.copy()
    my_df['IsConsumed'] = my_df.loc[:, consume_feats].sum(axis=1) >0
    return my_df

def addFoodCourtConsumeLevel (df):
    my_df = df.copy()
    my_df.loc[my_df['FoodCourt'] <= 8000, 'FoodCourtConsume'] = 'fc<=8k'
    my_df.loc[(my_df['FoodCourt'] > 8000) & (my_df['FoodCourt'] <= 18000), 'FoodCourtConsume'] = 'fc<=18k'
    my_df.loc[my_df['FoodCourt'] > 18000, 'FoodCourtConsume'] = 'fc>18k'
    return my_df

def addShoppingMallConsumeLevel (df):
    my_df = df.copy()
    my_df.loc[my_df['ShoppingMall'] <= 6000, 'ShoppingMallConsume'] = 'sm<=6k'
    my_df.loc[(my_df['ShoppingMall'] > 6000) & (my_df['ShoppingMall'] <= 8000), 'ShoppingMallConsume'] = 'sm<=8k'
    my_df.loc[(my_df['ShoppingMall'] > 8000) & (my_df['ShoppingMall'] <= 10000), 'ShoppingMallConsume'] = 'sm<=10k'
    my_df.loc[(my_df['ShoppingMall'] > 10000) & (my_df['ShoppingMall'] <= 12000), 'ShoppingMallConsume'] = 'sm<=12k'
    my_df.loc[my_df['ShoppingMall'] > 12000, 'ShoppingMallConsume'] = 'sm>12k'
    return my_df

fig=plt.figure(figsize=(10,20))

ax=fig.add_subplot(5,2,1)
sns.histplot(data=addTotalConsumption(train), axes=ax, bins=40, x='ConsumptionTotal', hue='Transported')
plt.xlim([0,10000])
ax.set_title('Total Consumption')

ax=fig.add_subplot(5,2,2)
sns.histplot(data=addAverageConsumption(train), axes=ax, bins=40, x='ConsumptionAverage', hue='Transported')
plt.xlim([0,5000])
ax.set_title('Average Consumption')

ax=fig.add_subplot(5,2,3)
sns.histplot(data=addTotalMoreAliveConsumption(train), axes=ax, bins=40, x='ConsumptionTotalMoreAlive', hue='Transported')
plt.xlim([0,10000])
plt.ylim([0,200])
ax.set_title('More for live')

ax=fig.add_subplot(5,2,4)
sns.histplot(data=addTotalMoreDieConsumption(train), axes=ax, bins=40, x='ConsumptionTotalMoreDie', hue='Transported')
plt.xlim([0,10000])
plt.ylim([0,200])
ax.set_title('More for Die')

ax=fig.add_subplot(5,2,5)
sns.countplot(data=addTotalMoreAliveConsumptionOver3k(train), axes=ax, x='ConsumptionTotalMoreAliveMore3K', hue='Transported')
ax.set_title('More for live over 3k')

ax=fig.add_subplot(5,2,6)
sns.countplot(data=addIsConsumedMoreDie(train), axes=ax, x='IsConsumedMoreDie', hue='Transported')
ax.set_title('Is More Die Consumed')

ax=fig.add_subplot(5,2,7)
sns.countplot(data=addIsConsumed(train), axes=ax, x='IsConsumed', hue='Transported')
ax.set_title('Is Consumed')

ax=fig.add_subplot(5,2,8)
sns.countplot(data=addFoodCourtConsumeLevel(train), axes=ax, x='FoodCourtConsume', hue='Transported')
plt.ylim([0,200])
ax.set_title('Food Court Consume')

ax=fig.add_subplot(5,2,9)
sns.countplot(data=addShoppingMallConsumeLevel(train), axes=ax, x='ShoppingMallConsume', hue='Transported')
plt.ylim([0,10])
ax.set_title('Shopping Mall Consume')

fig.tight_layout()  # Improves appearance a bit


# ### Cabin

# In[15]:


def splitCabinForNewFeatures (df):
    my_df = df.copy()
    split_cabin_df = my_df.Cabin.str.split("/", expand = True)
    my_df['CabinDeck'] = split_cabin_df[0]
    my_df['CabinNum'] = split_cabin_df[1]
    my_df['CabinSide'] = split_cabin_df[2]
    my_df.pop('Cabin')
    return my_df

splitedCabinData = splitCabinForNewFeatures(train)
splitedCabinData['CabinNum'] = splitedCabinData['CabinNum'].fillna(0).astype(int) # only int type can display well in histplot
print('max cabin number {}'.format(splitedCabinData['CabinNum'].max()))

fig=plt.figure(figsize=(60,60))

ax1=fig.add_subplot(10,4,1)
sns.countplot(data=splitedCabinData.fillna('None'), axes=ax1, x='CabinDeck', hue='Transported')
ax1.set_title('Cabin Deck')

ax2=fig.add_subplot(10,4,2)
sns.histplot(data=splitedCabinData, x='CabinNum', binwidth=100, axes=ax2, hue='Transported')
plt.xlim([0,2000])
ax2.set_title('Cabin Number')

ax3=fig.add_subplot(10,4,3)
sns.countplot(data=splitedCabinData.fillna('None'), axes=ax3, x='CabinSide', hue='Transported')
ax3.set_title('Cabin Side')


# In[16]:


def addGroupCarbinNumber(df):
    my_df = df.copy()
    cabin_num = my_df['CabinNum'].fillna(0).astype(int)
    my_df.loc[cabin_num <= 300, 'CabinNumGroup'] = 'CabinGroup0-300'
    my_df.loc[(cabin_num > 300) & (cabin_num <= 700), 'CabinNumGroup'] = 'CabinGroup300-700'
    my_df.loc[(cabin_num > 700) & (cabin_num <= 1100), 'CabinNumGroup'] = 'CabinGroup700-1100'
    my_df.loc[cabin_num > 1100, 'CabinNumGroup'] = 'CabinGroup1100~'
    return my_df

plt.figure(figsize=(10,5))
sns.countplot(data=addGroupCarbinNumber(splitedCabinData), x='CabinNumGroup', hue='Transported')
plt.xlabel('Cabin Number Group')


# ### Family Name

# In[17]:


def splitNameToGenerateFamilyName (df):
    my_df = df.copy()
    split_name_df = my_df.Name.str.split(" ", expand = True)
    my_df['FamilyName'] = split_name_df[1]
    my_df['FamilyName'] = my_df['FamilyName'].fillna('Unknown')
    familyNameValueCounts = my_df['FamilyName'].value_counts()
    my_df['FamilySize'] = my_df['FamilyName'].map(lambda name: familyNameValueCounts[name] if name != 'Unknown' else 0)
    my_df.pop('Name')
    return my_df

plt.figure(figsize=(10,4))
sns.countplot(data=splitNameToGenerateFamilyName(train), x='FamilySize', hue='Transported')
plt.xlabel('Family Size')


# ## Transform the data and generate new features

# In[18]:


train_X.info()


# In[19]:


def displayAllCateFeatInfo (df):
    for column in df.columns:
        if (df[column].dtype == 'object'):
            print("column: {} -> {}, unique values: {}".format(column, df[column].unique(), df[column].nunique()))
            
displayAllCateFeatInfo(train_X)


# In[20]:


def addAgeGroup (df):
    my_df = df.copy()
    my_df.loc[my_df['Age'] <= 18, 'AgeGroup'] = 'Age0-18'
    my_df.loc[(my_df['Age'] > 18) & (my_df['Age'] <= 26), 'AgeGroup'] = 'Age18-26'
    my_df.loc[(my_df['Age'] > 26) & (my_df['Age'] <= 42), 'AgeGroup'] = 'Age26-42'
    my_df.loc[my_df['Age'] > 42, 'AgeGroup'] = 'Age42~'
    return my_df
    
addAgeGroup(train_X).head()


# In[21]:


def addConsumptionRelatedFeats (df):
    my_df = df.copy()
    my_df = addTotalMoreAliveConsumption(my_df)
    my_df = addTotalMoreDieConsumption(my_df)
    my_df = addIsConsumedMoreDie(my_df)
    my_df = addIsConsumed(my_df)
#     my_df = addTotalMoreAliveConsumptionOver3k(my_df)
    my_df = addFoodCourtConsumeLevel(my_df)
    my_df = addShoppingMallConsumeLevel(my_df)
    return my_df
    
addConsumptionRelatedFeats(train_X).head()


# In[22]:


def dropColumns (df, feats):
    my_df = df.copy()
    my_df = my_df.drop(columns = feats)
    return my_df


# In[23]:


def transformFeatures (df):
    my_df = df.copy()
    my_df = addGroupSizeFeat(my_df)
    my_df = addIsSoleFeat(my_df)
    my_df = splitCabinForNewFeatures(my_df)
    my_df = addGroupCarbinNumber(my_df)
    my_df = splitNameToGenerateFamilyName(my_df)
    my_df = addAgeGroup(my_df)
    my_df = addConsumptionRelatedFeats(my_df)
    my_df = dropColumns(my_df, ["PassengerId", 'VIP'])
    return my_df


# In[24]:


transformed_train_X = transformFeatures(train_X)
transformed_train_X.info()


# ## Fill in missing values

# In[25]:


transformed_train_X.isna().mean(axis=0)


# In[26]:


def getNullInfo(df):
    for column in df.columns:
        print("column: {} -> {}".format(column, df[column].isnull().sum()))
getNullInfo(transformed_train_X)


# In[27]:


transformed_train_X.CabinSide.value_counts()


# ### Pipeline for scale numeric features and encode category features

# In[28]:


from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

categorical_pipeline = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("oh-encode", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ]
)


# In[29]:


from sklearn.preprocessing import StandardScaler

numeric_pipeline = Pipeline(
    steps=[("impute", SimpleImputer(strategy="mean")), 
           ("scale", StandardScaler())]
)


# ## Function to transform the input data

# In[30]:


def transform_input (df):
    my_df = df.copy()
    my_df = transformFeatures(my_df)
    return my_df


# In[31]:


feat_transformed_train_X = transform_input(train_X)
feat_transformed_train_X.shape


# In[32]:


from sklearn.compose import ColumnTransformer

num_feats = feat_transformed_train_X.select_dtypes(include='number').columns
cat_feats = feat_transformed_train_X.select_dtypes(exclude='number').columns

full_processor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, num_feats),
        ("categorical", categorical_pipeline, cat_feats),
    ]
)

ready_train_X = full_processor.fit_transform(feat_transformed_train_X)
ready_train_X.shape


# ## Split train and test

# In[33]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[34]:


ready_train_y = SimpleImputer(strategy="most_frequent").fit_transform(train_Y.values.reshape(-1, 1).astype(int))

X_split_train, X_split_test, y_split_train, y_split_test = train_test_split(
    ready_train_X,
    ready_train_y,
    stratify=ready_train_y, # so the split contains the same proportion of categories in both sets
    random_state=1121218
)


# In[35]:


from lightgbm import LGBMClassifier

clf = LGBMClassifier()
clf.fit(X_split_train, y_split_train.ravel())
preds = clf.predict(X_split_test)
accuracy_score(preds, y_split_test)


# ## GridSearchCV

# In[36]:


from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score


# In[37]:


lgb_model = LGBMClassifier(objective="binary")


# In[38]:


ready_train_y = SimpleImputer(strategy="most_frequent").fit_transform(train_Y.values.reshape(-1, 1).astype(int))

param_grid = {
    'n_estimators': [100, 200, 300],
    "max_depth": [6, 8, 12],
    "learning_rate": [0.01, 0.02, 0.05]
}

grid_cv = GridSearchCV(
    lgb_model,
    param_grid,
    n_jobs=-1,
    cv=StratifiedKFold(n_splits=5, shuffle=True),
    scoring="roc_auc",
    verbose=2,
    refit=True
)

_ = grid_cv.fit(ready_train_X, ready_train_y.ravel())


# In[39]:


grid_cv.best_score_, grid_cv.best_params_


# In[40]:


clf = LGBMClassifier(
    **grid_cv.best_params_,
    objective="binary",
)

ready_train_y = SimpleImputer(strategy="most_frequent").fit_transform(train_Y.values.reshape(-1, 1).astype(int))
clf.fit(ready_train_X, ready_train_y.ravel())


# ## Submit

# In[41]:


test.info()


# In[42]:


feat_transformed_test = transform_input(test)
ready_test = full_processor.transform(feat_transformed_test)
ready_test.shape


# In[43]:


predection = clf.predict(ready_test).astype(bool)
my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Transported': predection})
my_submission


# In[44]:


my_submission.to_csv('submission.csv', index=False)


# In[ ]:




