#!/usr/bin/env python
# coding: utf-8

# ## 0.1. References
# https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial
# ## 0.2. Import libraries and data

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

import string
import warnings
from pprint import PrettyPrinter

warnings.filterwarnings('ignore')
pprint = PrettyPrinter().pprint

SEED = 42


# In[2]:


def concat_df(train_data, test_data):
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

def divide_df(all_data):
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_all = concat_df(df_train, df_test)

df_train.name = 'Training Set'
df_test.name = 'Test Set'
df_all.name = 'All Set'

dfs = [df_train, df_test]

print('Number of Training Examples = {}'.format(df_train.shape[0]))
print('Number of Test Examples = {}\n'.format(df_test.shape[0]))
print('Training X Shape = {}'.format(df_train.shape))
print('Training y Shape = {}\n'.format(df_train['Survived'].shape[0]))
print('Test X Shape = {}'.format(df_test.shape))
print('Test y Shape = {}\n'.format(df_test.shape[0]))
print(df_train.columns)
print(df_test.columns)


# ## 0.3. Table of contents
# 
# 1. Exploratory Data Analysis
#     1. Overview
#     2. Missing Values
#         1. Age column
#         2. Embarked column
#         3. Fare column
#         4. Cabin column
#         5. Result of filling the missing values
#     3. Target Distribution
#     4. Correlations
#     5. Target Distribution in Feature
#         1. Continuous Features
#         2. Categorical Features
# 2. Feature Engineering
#     1. Binding Continous Features
#         1. Fare column
#         2. Age column
#     2. Frequency Encoding
#     3. Title & is Married
#     4. Target Encoding
#     5. Feature Transformation
#         1. Label Encoding Non-Numerical Features
#         2. One-Hot Encoding the Categorical Features
#     6. Conclusion
# 3. Model
#     1. Random Forest with StratifiedKFold
#     2. Feature Importance
#     3. ROC Curve
#     4. Submission

# # 1. Exploratory Data Analysis
# 
# ## 1.1. Overview

# In[3]:


print(df_train.info())
df_train.sample(3)


# In[4]:


print(df_test.info())
df_test.sample(3)


# ## 1.2. Missing Values

# In[5]:


def display_missing(df):
    for col in df.columns.tolist():
        print('{:>11} column missing values: {:>3}'.format(col, df[col].isnull().sum()))
    print('\n')
    
for df in dfs:
    print('{}'.format(df.name))
    display_missing(df)


# ### 1.2.1. Age column
# 
# #### display correlation for Age column

# In[6]:


# corr(): Compute pairwise correlation of columns, excluding NA/null values
# abs(): Return a Series/DataFrame with absolute numeric value of each element
# unstack(): Pivot a level of the (necessarily hierarchical) index labels
df_all_corr = df_all.corr() \
    .abs() \
    .unstack() \
    .sort_values(kind='quicksort', ascending=False) \
    .reset_index()

df_all_corr.rename(columns={
    'level_0': 'Feature 1',
    'level_1': 'Feature 2',
    0: 'Correlation Coefficient'
}, inplace=True)

df_all_corr[df_all_corr['Feature 1'] == 'Age']


# #### fill na with median value for Age column

# In[7]:


age_by_pclass_sex = df_all.groupby(['Sex', 'Pclass']).median()['Age']
age_by_pclass_sex

for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Median age of Pclass {} {:>6}s: {}'.format(
            pclass,
            sex,
            age_by_pclass_sex[sex][pclass]
        ))
print('Median age of all: {}'.format(df_all['Age'].median()))

# Filling the missing value in Age with the medians of Sex and Pclass groups
df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'] \
    .apply(lambda x: x.fillna(x.median()))


# ### 1.2.2. Embarked column

# In[8]:


df_all[df_all['Embarked'].isnull()]


# In[9]:


#  When I googled Stone, Mrs. George Nelson (Martha Evelyn),
# I found that she embarked from S (Southampton) with her maid Amelie Icard,
# in this page Martha Evelyn Stone: Titanic Survivor.  
#  "Mrs Stone boarded the Titanic in Southampton on 10 April 1912
# and was travelling in first class with her maid Amelie Icard.
# She occupied cabin B-28."
#  Missing values in Embarked are filled with S with this information.

df_all['Embarked'] = df_all['Embarked'].fillna('S')


# ### 1.2.3. Fare column

# In[10]:


df_all[df_all['Fare'].isnull()]


# In[11]:


med_fare = df_all.groupby(['Pclass', 'Parch', 'SibSp'])['Fare'].median()
med_fare = med_fare[3][0][0]
df_all['Fare'] = df_all['Fare'].fillna(med_fare)


# ### 1.2.4. Cabin column
# #### passenger class distribution in decks

# In[12]:


# M stands for Missing
df_all['Deck'] = df_all['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

drop_columns = [
    'Survived', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin',
    'PassengerId', 'Ticket'
]
# transpose(): Transpose index and columns
df_all_decks = df_all.groupby(['Deck', 'Pclass']) \
    .count() \
    .drop(columns=drop_columns) \
    .rename(columns={'Name': 'Count'}) \
    .transpose()

def get_pclass_dist(df):
    deck_counts = {
        'A': {}, 'B': {}, 'C': {},
        'D': {}, 'E': {}, 'F': {},
        'G': {}, 'M': {}, 'T': {}
    }
    decks = df.columns.levels[0]
    
    for deck in decks:
        for pclass in range(1, 4):
            try:
                count = df[deck][pclass][0]
                deck_counts[deck][pclass] = count
            except KeyError:
                deck_counts[deck][pclass] = 0
                
    df_decks = pd.DataFrame(deck_counts)
    deck_percentages = {}
    
    for col in df_decks.columns:
        deck_percentages[col] = [
            (count / df_decks[col].sum()) * 100 for count in df_decks[col]
        ]
        
    return deck_counts, deck_percentages

def display_pclass_dist(percentages):
    df_percentage = pd.DataFrame(percentages).transpose()
    deck_names = (
        'A', 'B', 'C',
        'D', 'E', 'F',
        'G', 'M', 'T'
    )
    bar_count = np.arange(len(deck_names))
    bar_width = 0.85
    
    pclass1 = df_percentage[0]
    pclass2 = df_percentage[1]
    pclass3 = df_percentage[2]
    
    plt.figure(figsize=(20, 10))
    plt.bar(bar_count, pclass1, color='#b5ffb9', edgecolor='white', width=bar_width, label='Passenger Class 1')
    plt.bar(bar_count, pclass2, bottom=pclass1, color='#f9bc86', edgecolor='white', width=bar_width, label='Passenger Class 2')
    plt.bar(bar_count, pclass3, bottom=pclass1 + pclass2, color='#a3acff', edgecolor='white', width=bar_width, label='Passenger Class 3')

    plt.xlabel('Deck', size=15, labelpad=20)
    plt.ylabel('Passenger Class Percentage', size=15, labelpad=20)
    plt.xticks(bar_count, deck_names)    
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 15})
    plt.title('Passenger Class Distribution in Decks', size=18, y=1.05)   
    
    plt.show()
    
all_deck_count, all_deck_per = get_pclass_dist(df_all_decks)
pprint(all_deck_count)
pprint(all_deck_per)
display_pclass_dist(all_deck_per)


# In[13]:


#  There is one person on the boat deck in T cabin
# and he is a 1st class passenger.
#  T cabin passenger has the closest resemblance to A deck passengers
# so he is grouped with A deck
idx = df_all[df_all['Deck'] == 'T'].index
df_all.loc[idx, 'Deck'] = 'A'


# #### survival percentage in decks

# In[14]:


drop_columns_all_decks_survived = [
    'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass',
    'Cabin', 'PassengerId', 'Ticket'
]
df_all_decks_survived = df_all \
    .groupby(['Deck', 'Survived']) \
    .count() \
    .drop(columns=drop_columns_all_decks_survived) \
    .rename(columns={'Name': 'Count'}) \
    .transpose()

def get_survived_dist(df):
    surv_counts = {
        'A': {}, 'B': {}, 'C': {},
        'D': {}, 'E': {}, 'F': {},
        'G': {}, 'M': {}
    }
    decks = df.columns.levels[0]
    
    for deck in decks:
        for survive in range(0, 2):
            surv_counts[deck][survive] = df[deck][survive][0]
            
    df_surv = pd.DataFrame(surv_counts)
    surv_percentages = {}
    
    for col in df_surv.columns:
        surv_percentages[col] = [
            (count / df_surv[col].sum()) * 100 for count in df_surv[col]
        ]
        
    return surv_counts, surv_percentages

def display_surv_dist(percentages):
    
    df_survived_percentages = pd.DataFrame(percentages).transpose()
    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M')
    bar_count = np.arange(len(deck_names))  
    bar_width = 0.85    

    not_survived = df_survived_percentages[0]
    survived = df_survived_percentages[1]
    
    plt.figure(figsize=(20, 10))
    plt.bar(bar_count, not_survived, color='#b5ffb9', edgecolor='white', width=bar_width, label="Not Survived")
    plt.bar(bar_count, survived, bottom=not_survived, color='#f9bc86', edgecolor='white', width=bar_width, label="Survived")
 
    plt.xlabel('Deck', size=15, labelpad=20)
    plt.ylabel('Survival Percentage', size=15, labelpad=20)
    plt.xticks(bar_count, deck_names)    
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 15})
    plt.title('Survival Percentage in Decks', size=18, y=1.05)
    
    plt.show()
    
all_surv_count, all_surv_per = get_survived_dist(df_all_decks_survived)
pprint(all_surv_count)
pprint(all_surv_per)
display_surv_dist(all_surv_per)


# - Passenger Class Distribution 과 Survival Percentage 데이터를 통해 연관이 있는 걸로 판단되는 `Cabin` 데이터들을 `Deck` 컬럼으로 그룹화한다.

# In[15]:


df_all['Deck'] = df_all['Deck'].replace(['A', 'B', 'C'], 'ABC')
df_all['Deck'] = df_all['Deck'].replace(['D', 'E'], 'DE')
df_all['Deck'] = df_all['Deck'].replace(['F', 'G'], 'FG')

df_all['Deck'].value_counts()


# ### 1.2.5. Result of filling the missing values

# In[16]:


df_all.drop(['Cabin'], inplace=True, axis=1)

df_train, df_test = divide_df(df_all)
dfs = [df_train, df_test]

for df in dfs:
    display_missing(df)


# ## 1.3. Target Distribution

# In[17]:


survived = df_train['Survived'].value_counts()[1]
not_survived = df_train['Survived'].value_counts()[0]
survived_per = survived / df_train.shape[0]
not_survived_per = not_survived / df_train.shape[0]

print(
    '{} of {} passengers survived and it is the {:.2f}% of the training set.'.format(
        survived,
        df_train.shape[0],
        survived_per
    )
)
print(
    '{} of {} passengers didnt survive and it is the {:.2f}% of the training set.'.format(
        not_survived,
        df_train.shape[0],
        not_survived_per
    )
)

plt.figure(figsize=(10, 8))
sns.countplot(df_train['Survived'])

plt.xlabel('Survival', size=15, labelpad=15)
plt.ylabel('Passenger Count', size=15, labelpad=15)
plt.xticks((0, 1), [
    'Not Survived ({0:.2f}%)'.format(not_survived_per),
    'Survived ({0:.2f}%)'.format(survived_per)
])
plt.tick_params(axis='x', labelsize=13)
plt.tick_params(axis='y', labelsize=13)

plt.title('Training Set Survival Distribution', size=15, y=1.05)

plt.show()


# ## 1.4. Correlations

# In[18]:


df_train_corr = df_train \
    .drop(['PassengerId'], axis=1) \
    .corr() \
    .abs() \
    .unstack() \
    .sort_values(kind='quicksort', ascending=False) \
    .reset_index()

df_train_corr.rename(columns={
    'level_0': 'Feature 1',
    'level_1': 'Feature 2',
    0: 'Correlation Coefficient'
}, inplace=True)

df_train_corr.drop(df_train_corr.iloc[1::2].index, inplace=True)
df_train_corr_nd = df_train_corr.drop(
    df_train_corr[df_train_corr['Correlation Coefficient'] == 1.0].index
)

df_test_corr = df_test \
    .drop(['PassengerId'], axis=1) \
    .corr() \
    .abs() \
    .unstack() \
    .sort_values(kind='quicksort', ascending=False) \
    .reset_index()

df_test_corr.rename(columns={
    'level_0': 'Feature 1',
    'level_1': 'Feature 2',
    0: 'Correlation Coefficient'
}, inplace=True)

df_test_corr.drop(df_test_corr.iloc[1::2].index, inplace=True)
df_test_corr_nd = df_test_corr.drop(
    df_train_corr[df_train_corr['Correlation Coefficient'] == 1.0].index
)
df_test_corr_nd


# In[19]:


corr = df_train_corr_nd['Correlation Coefficient'] > 0.1
df_train_corr_nd[corr]


# In[20]:


corr = df_test_corr_nd['Correlation Coefficient'] > 0.1
df_test_corr_nd[corr]


# In[21]:


fig, axs = plt.subplots(nrows=2, figsize=(15, 15))

sns.heatmap(
    df_train.drop(['PassengerId'], axis=1).corr(),
    ax=axs[0],
    annot=True,
    square=True,
    cmap='coolwarm',
    annot_kws={'size': 14}
)
sns.heatmap(
    df_test.drop(['PassengerId'], axis=1).corr(),
    ax=axs[1],
    annot=True,
    square=True,
    cmap='coolwarm',
    annot_kws={'size': 14}
)

for i in range(1):    
    axs[i].tick_params(axis='x', labelsize=14)
    axs[i].tick_params(axis='y', labelsize=14)
    
axs[0].set_title('Training Set Correlations', size=15)
axs[1].set_title('Test Set Correlations', size=15)

plt.show()


# ## 1.5. Target Distribution In Feature
# 
# ### 1.5.1. Continuous Features

# In[22]:


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


# ### 1.5.2. Categorical Features

# In[23]:


cat_features = ['Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp', 'Deck']

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


# # 2. Feature Engineering
# ## 2.1. Binning Continuous Features
# ### 2.1.1. Fare column

# In[24]:


df_all['Fare'] = pd.qcut(df_all['Fare'], 13)


# In[25]:


fig, axs = plt.subplots(figsize=(22, 9))
sns.countplot(x='Fare', hue='Survived', data=df_all)

plt.xlabel('Fare', size=15, labelpad=20)
plt.ylabel('Passenger Count', size=15, labelpad=20)
plt.tick_params(axis='x', labelsize=10)
plt.tick_params(axis='y', labelsize=15)

plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})
plt.title('Count of Survival in {} Feature'.format('Fare'), size=15, y=1.05)

plt.show()


# ### 2.1.2. Age column

# In[26]:


df_all['Age'] = pd.qcut(df_all['Age'], 10)


# In[27]:


fig, axs = plt.subplots(figsize=(22, 9))
sns.countplot(x='Age', hue='Survived', data=df_all)

plt.xlabel('Age', size=15, labelpad=20)
plt.ylabel('Passenger Count', size=15, labelpad=20)
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})
plt.title('Survival Counts in {} Feature'.format('Age'), size=15, y=1.05)

plt.show()


# ## 2.2. Frequency Encoding
#  - SibSp 와 Parch 를 사용해 Family_Size 생성
#  - Family_Size 를 가지고 Family_Size_Grouped 생성
#  - Ticket 을 가지고 Ticket_Frequency 생성

# In[28]:


df_all['Family_Size'] = df_all['SibSp'] + df_all['Parch'] + 1


# In[29]:


family_map = {
    1: 'Alone',
    2: 'Small',
    3: 'Small',
    4: 'Small',
    5: 'Medium',
    6: 'Medium',
    7: 'Large',
    8: 'Large',
    11: 'Large'
}
df_all['Family_Size_Grouped'] = df_all['Family_Size'].map(family_map)


# In[30]:


fig, axs = plt.subplots(figsize=(20, 20), ncols=2, nrows=2)
plt.subplots_adjust(right=1.5)

sns.barplot(x=df_all['Family_Size'].value_counts().index, y=df_all['Family_Size'].value_counts().values, ax=axs[0][0])
sns.countplot(x='Family_Size', hue='Survived', data=df_all, ax=axs[0][1])

axs[0][0].set_title('Family Size Feature Value Counts', size=20, y=1.05)
axs[0][1].set_title('Survival Counts in Family Size ', size=20, y=1.05)

sns.barplot(x=df_all['Family_Size_Grouped'].value_counts().index, y=df_all['Family_Size_Grouped'].value_counts().values, ax=axs[1][0])
sns.countplot(x='Family_Size_Grouped', hue='Survived', data=df_all, ax=axs[1][1])

axs[1][0].set_title('Family Size Feature Value Counts After Grouping', size=20, y=1.05)
axs[1][1].set_title('Survival Counts in Family Size After Grouping', size=20, y=1.05)

for i in range(2):
    axs[i][1].legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 20})
    for j in range(2):
        axs[i][j].tick_params(axis='x', labelsize=20)
        axs[i][j].tick_params(axis='y', labelsize=20)
        axs[i][j].set_xlabel('')
        axs[i][j].set_ylabel('')

plt.show()


# In[31]:


df_all['Ticket_Frequency'] = df_all.groupby('Ticket')['Ticket'].transform('count')


# In[32]:


fig, axs = plt.subplots(figsize=(12, 9))
sns.countplot(x='Ticket_Frequency', hue='Survived', data=df_all)

plt.xlabel('Ticket Frequency', size=15, labelpad=20)
plt.ylabel('Passenger Count', size=15, labelpad=20)
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})
plt.title('Count of Survival in {} Feature'.format('Ticket Frequency'), size=15, y=1.05)

plt.show()


# ## 2.3. Title & is Married
# 
#  - Name 으로 Title 생성
#  - Title 로 Is_Married 생성

# In[33]:


df_all['Title_Original'] = df_all['Name'] \
    .str.split(', ', expand=True)[1] \
    .str.split('.', expand=True)[0]
df_all['Title'] = df_all['Name'] \
    .str.split(', ', expand=True)[1] \
    .str.split('.', expand=True)[0]

df_all['Title'] = df_all['Title'].replace(
    ['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'],
    'Miss/Mrs/Ms'
)
df_all['Title'] = df_all['Title'].replace(
    ['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'],
    'Dr/Military/Noble/Clergy'
)

df_all['Is_Married'] = 0
df_all['Is_Married'].loc[df_all['Title'] == 'Mrs'] = 1


# In[34]:


fig, axs = plt.subplots(nrows=2, figsize=(20, 20))
sns.barplot(
    x=df_all['Title_Original'].value_counts().index,
    y=df_all['Title_Original'].value_counts().values,
    ax=axs[0]
)

axs[0].tick_params(axis='x', labelsize=10)
axs[1].tick_params(axis='x', labelsize=15)

for i in range(2):    
    axs[i].tick_params(axis='y', labelsize=15)

axs[0].set_title('Title Feature Value Counts', size=20, y=1.05)

sns.barplot(
    x=df_all['Title'].value_counts().index,
    y=df_all['Title'].value_counts().values,
    ax=axs[1]
)
axs[1].set_title('Title Feature Value Counts After Grouping', size=20, y=1.05)

plt.show()
df_all.drop('Title_Original', axis=1, inplace=True)


# ## 2.4. Target Encoding
# 
#  - Name 으로 Family 생성 (family name)
#  - Family, Family_Size 값을 사용해 가족 당 생존 확률 구함
#  - Ticket, Ticket_Frequency 값을 사용해 티켓 당 생존 확률 구함
#  - Family_Survival_Rate
#     - 훈련 데이터 / 테스트 데이터에 모두 있는 Family 라면 가족 당 생존 확률을 저장
#     - 어느 한 쪽에라도 없는 데이터라면 훈련 데이터 전체의 평균 생존 확률을 저장
#  - Family_Survival_Rate_NA
#     - 훈련 데이터 / 테스트 데이터에 모두 있는 Family 인지 저장 (1, 0)
#  - Ticket_Survival_Rate
#     - 훈련 데이터 / 테스트 데이터에 모두 있는 Ticket 이라면 티켓 당 생존 확률을 저장
#     - 어느 한 쪽에라도 없는 데이터라면 훈련 데이터 전체의 평균 생존 확률을 저장
#  - Ticket_Survival_Rate_NA
#     - 훈련 데이터 / 테스트 데이터에 모두 있는 Ticket 인지 저장 (1, 0)
#  - Survaival_Rate = (Ticket_Survival_Rate + Family_Survival_Rate) / 2
#  - Survaival_Rate_NA = (Ticket_Survival_Rate_NA + Family_Survival_Rate_NA) / 2
#  
# #### get family name

# In[35]:


def extract_surname(data):    
    
    families = []
    
    for i in range(len(data)):        
        name = data.iloc[i]

        if '(' in name:
            name_no_bracket = name.split('(')[0] 
        else:
            name_no_bracket = name
            
        family = name_no_bracket.split(',')[0]
        title = name_no_bracket.split(',')[1].strip().split(' ')[0]
        
        for c in string.punctuation:
            family = family.replace(c, '').strip()
            
        families.append(family)
            
    return families

df_all['Family'] = extract_surname(df_all['Name'])
df_train = df_all.loc[:890]
df_test = df_all.loc[891:]
dfs = [df_train, df_test]


# #### get survival rate by family, family size, tickt and ticket frequency

# In[36]:


non_unique_families = [
    x for x in df_train['Family'].unique() if x in df_test['Family'].unique()
]
non_unique_tickets = [
    x for x in df_train['Ticket'].unique() if x in df_test['Ticket'].unique()
]

df_family_survival_rate = df_train \
    .groupby('Family')['Survived', 'Family', 'Family_Size'] \
    .median()
df_ticket_survival_rate = df_train \
    .groupby('Ticket')['Survived', 'Ticket', 'Ticket_Frequency'] \
    .median()
    
family_rates = {}
ticket_rates = {}

for i in range(len(df_family_survival_rate)):
    if df_family_survival_rate.index[i] in non_unique_families \
        and df_family_survival_rate.iloc[i, 1] > 1:
        family_rates[df_family_survival_rate.index[i]] = \
            df_family_survival_rate.iloc[i, 0]

for i in range(len(df_ticket_survival_rate)):
    if df_ticket_survival_rate.index[i] in non_unique_tickets \
        and df_ticket_survival_rate.iloc[i, 1] > 1:
        ticket_rates[df_ticket_survival_rate.index[i]] = \
            df_ticket_survival_rate.iloc[i, 0]


# #### add survlval rate columns

# In[37]:


mean_survival_rate = np.mean(df_train['Survived'])

train_family_survival_rate = []
train_family_survival_rate_NA = []
test_family_survival_rate = []
test_family_survival_rate_NA = []

for i in range(len(df_train)):
    if df_train['Family'][i] in family_rates:
        train_family_survival_rate.append(family_rates[df_train['Family'][i]])
        train_family_survival_rate_NA.append(1)
    else:
        train_family_survival_rate.append(mean_survival_rate)
        train_family_survival_rate_NA.append(0)
        
for i in range(len(df_test)):
    if df_test['Family'].iloc[i] in family_rates:
        test_family_survival_rate.append(family_rates[df_test['Family'].iloc[i]])
        test_family_survival_rate_NA.append(1)
    else:
        test_family_survival_rate.append(mean_survival_rate)
        test_family_survival_rate_NA.append(0)

df_train['Family_Survival_Rate'] = train_family_survival_rate
df_train['Family_Survival_Rate_NA'] = train_family_survival_rate_NA
df_test['Family_Survival_Rate'] = test_family_survival_rate
df_test['Family_Survival_Rate_NA'] = test_family_survival_rate_NA

train_ticket_survival_rate = []
train_ticket_survival_rate_NA = []
test_ticket_survival_rate = []
test_ticket_survival_rate_NA = []

for i in range(len(df_train)):
    if df_train['Ticket'][i] in ticket_rates:
        train_ticket_survival_rate.append(ticket_rates[df_train['Ticket'][i]])
        train_ticket_survival_rate_NA.append(1)
    else:
        train_ticket_survival_rate.append(mean_survival_rate)
        train_ticket_survival_rate_NA.append(0)
        
for i in range(len(df_test)):
    if df_test['Ticket'].iloc[i] in ticket_rates:
        test_ticket_survival_rate.append(ticket_rates[df_test['Ticket'].iloc[i]])
        test_ticket_survival_rate_NA.append(1)
    else:
        test_ticket_survival_rate.append(mean_survival_rate)
        test_ticket_survival_rate_NA.append(0)
        
df_train['Ticket_Survival_Rate'] = train_ticket_survival_rate
df_train['Ticket_Survival_Rate_NA'] = train_ticket_survival_rate_NA
df_test['Ticket_Survival_Rate'] = test_ticket_survival_rate
df_test['Ticket_Survival_Rate_NA'] = test_ticket_survival_rate_NA


# In[38]:


for df in [df_train, df_test]:
    df['Survival_Rate'] = \
        (df['Ticket_Survival_Rate'] + df['Family_Survival_Rate']) / 2
    df['Survival_Rate_NA'] = \
        (df['Ticket_Survival_Rate_NA'] + df['Family_Survival_Rate_NA']) / 2


# ## 2.5. Feature Transformation
# 
# ### 2.5.1. Label Encoding Non-Numerical Features

# In[39]:


non_numeric_features = [
    'Embarked', 'Sex', 'Deck', 'Title', 'Family_Size_Grouped', 'Age', 'Fare'
]

for df in dfs:
    for feature in non_numeric_features:
        df[feature] = LabelEncoder().fit_transform(df[feature])


# ### 2.5.2. One-Hot Encoding the Categorical Features

# In[40]:


cat_features = [
    'Pclass', 'Sex', 'Deck', 'Embarked', 'Title', 'Family_Size_Grouped'
]
encoded_features = []

for df in dfs:
    for feature in cat_features:
        encoded_feature = OneHotEncoder() \
            .fit_transform(df[feature].values.reshape(-1, 1)) \
            .toarray()
        n = df[feature].nunique()
        cols = ['{}_{}'.format(feature, n) for n in range(1, n+1)]
        encoded_df = pd.DataFrame(encoded_feature, columns=cols)
        encoded_df.index = df.index
        encoded_features.append(encoded_df)

df_train = pd.concat([df_train, *encoded_features[:6]], axis=1)
df_test = pd.concat([df_test, *encoded_features[6:]], axis=1)


# ## 2.6. Conclusion

# In[41]:


df_all = concat_df(df_train, df_test)
drop_cols = [
    'Deck', 'Embarked', 'Family', 'Family_Size', 'Family_Size_Grouped',
    'Survived', 'Name', 'Parch', 'PassengerId', 'Pclass', 'Sex', 'SibSp',
    'Ticket', 'Title', 'Ticket_Survival_Rate', 'Family_Survival_Rate',
    'Ticket_Survival_Rate_NA', 'Family_Survival_Rate_NA'
]
df_all.drop(columns=drop_cols, inplace=True)
df_all.columns


# # 3. Model

# In[42]:


X_train = StandardScaler() \
    .fit_transform(df_train.drop(columns=drop_cols))
y_train = df_train['Survived'].values
X_test = StandardScaler() \
    .fit_transform(df_test.drop(columns=drop_cols))

print('X_train shape: {}'.format(X_train.shape))
print('y_train shape: {}'.format(y_train.shape))
print('X_test shape: {}'.format(df_test.shape))


# ## 3.1. Random Forest with StratifiedKFold

# In[43]:


#  single_best_model is a good model to start experimenting
# and learning about decision trees.

single_best_model = RandomForestClassifier(
    criterion='gini',
    n_estimators=1100,
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=5,
    max_features='auto',
    oob_score=True,
    random_state=SEED,
    n_jobs=-1,
    verbose=1
)

#  leaderboard_model overfits to test set
# so it's not suggested to use models like this in real life projects.

leaderboard_model = RandomForestClassifier(
    criterion='gini',
    n_estimators=1750,
    max_depth=7,
    min_samples_split=6,
    min_samples_leaf=6,
    max_features='auto',
    oob_score=True,
    random_state=SEED,
    n_jobs=-1,
    verbose=1
)


# In[44]:


N = 5
oob = 0
probs = pd.DataFrame(
    np.zeros((len(X_test), N*2)),
    columns=[
        'Fold_{}_Prob_{}'.format(i, j) for i in range(1, N+1) for j in range(2)
    ]
)
importances = pd.DataFrame(
    np.zeros((X_train.shape[1], N)),
    columns=[
        'Fold_{}'.format(i) for i in range(1, N+1)
    ],
    index=df_all.columns
)
fprs = []
tprs = []
scores = []

skf = StratifiedKFold(
    n_splits=N,
    random_state=N,
    shuffle=True
)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print('Fold {}\n'.format(fold))
    
    leaderboard_model.fit(
        X_train[trn_idx],
        y_train[trn_idx]
    )
    
    trn_fpr, trn_tpr, trn_thresholds = \
        roc_curve(
            y_train[trn_idx],
            leaderboard_model.predict_proba(X_train[trn_idx])[:, 1]
        )
    trn_auc_score = auc(trn_fpr, trn_tpr)
    
    val_fpr, val_tpr, val_thresholds = \
        roc_curve(
            y_train[val_idx],
            leaderboard_model.predict_proba(X_train[val_idx])[:, 1]
        )
    val_auc_score = auc(val_fpr, val_tpr)
    
    scores.append((trn_auc_score, val_auc_score))
    fprs.append(val_fpr)
    tprs.append(val_tpr)
    
    probs.loc[:, 'Fold_{}_Prob_0'.format(fold)] = \
        leaderboard_model.predict_proba(X_test)[:, 0]
    probs.loc[:, 'Fold_{}_Prob_1'.format(fold)] = \
        leaderboard_model.predict_proba(X_test)[:, 1]
    importances.iloc[:, fold-1] = leaderboard_model.feature_importances_
    
    oob += leaderboard_model.oob_score_ / N
    print('Fold {} OOB Score: {}\n'.format(fold, leaderboard_model.oob_score_))   
    
print('Average OOB Score: {}'.format(oob))


# ## 3.2. Feature Importance

# In[45]:


importances['Mean_Importance'] = importances.mean(axis=1)
importances.sort_values(by='Mean_Importance', inplace=True, ascending=False)
importances

plt.figure(figsize=(15, 20))
sns.barplot(
    x='Mean_Importance',
    y=importances.index,
    data=importances
)

plt.xlabel('')
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.title('Random Forest Classifier Mean Feature Importance Between Folds', size=15)

plt.show()


# ## 3.3. ROC Curve

# In[46]:


def plot_roc_curve(fprs, tprs):
    
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(15, 15))
    
    # Plotting ROC for each fold and computing AUC scores
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs), 1):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC Fold {} (AUC = {:.3f})'.format(i, roc_auc))
        
    # Plotting ROC for random guessing
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8, label='Random Guessing')
    
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    # Plotting the mean ROC
    ax.plot(mean_fpr, mean_tpr, color='b', label='Mean ROC (AUC = {:.3f} $\pm$ {:.3f})'.format(mean_auc, std_auc), lw=2, alpha=0.8)
    
    # Plotting the standard deviation around the mean ROC Curve
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label='$\pm$ 1 std. dev.')
    
    ax.set_xlabel('False Positive Rate', size=15, labelpad=20)
    ax.set_ylabel('True Positive Rate', size=15, labelpad=20)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    ax.set_title('ROC Curves of Folds', size=20, y=1.02)
    ax.legend(loc='lower right', prop={'size': 13})
    
    plt.show()

plot_roc_curve(fprs, tprs)


# ## 3.4. Submission

# In[47]:


probs


# In[48]:


class_survived = [col for col in probs.columns if col.endswith('Prob_1')]
class_survived


# In[49]:


probs['1'] = probs[class_survived].sum(axis=1) / N
probs['0'] = probs.drop(columns=class_survived).sum(axis=1) / N


# In[50]:


probs['pred'] = 0
pos = probs[probs['1'] >= 0.5].index
probs.loc[pos, 'pred'] = 1
probs[['1','0','pred']].head(10)


# In[51]:


y_pred = probs['pred'].astype(int)

submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])
submission_df['PassengerId'] = df_test['PassengerId']
submission_df['Survived'] = y_pred.values
submission_df.to_csv('submissions.csv', header=True, index=False)
submission_df.head()

