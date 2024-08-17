#!/usr/bin/env python
# coding: utf-8

# <br>
# <h1 style="color:pink; text-align:center; font-size:30px; font-family:Arial Black; border-radius:30px 30px; background-color:black; line-height: 50px; padding: 15px 15px 15px 2.5%;">💥Titanic - LGBM with advanced feature engineering💥</h1>
# <br>

# ### Hi everyone, this is a beginner friendly notebook for the famous Titanic dataset.

# <div class="alert alert-block alert-info"><p style='color:black;'>Thanks to Gunes Evitan @gunesevitan for his wonderful notebook on EDA and feature engineering. Please do check his notebook. I am taking the insights and the fetures from his notebook.<br>
# 
#     https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial</p></div>

# # Approach

# <div class="alert alert-block alert-info"><p style='color:black;'>
# 1. Import libraries<br>
# 2. Read the data<br>
# 3. Basic checks<br>
# 4. Feature engineering<br>
# 5. Model<br>
#     6. Submission file</p></div>

# # ✅ Importing Required Libraries

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from lightgbm import LGBMClassifier

import string
import warnings
warnings.filterwarnings('ignore')


# # ✅Reading the Data

# In[2]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
ss = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")


# # 🔍Basic Data Checks

# In[4]:


print(f'Shape of Train dataset is : {train.shape}')
print(f'Shape of Test dataset is : {test.shape}')
print(f'Shape of Sample Submission dataset is : {ss.shape}')


# * Training set has 891 rows and test set has 418 rows
# * Training set have 12 features and test set have 11 features
# * One extra feature in training set is Survived feature, which is the target variable

# In[5]:


train.columns


# * PassengerId is the unique id of the row and it doesn't have any effect on target
# * Survived is the target variable we are trying to predict (0 or 1):
#     * 1 = Survived
#     * 0 = Not Survived
# * Pclass (Passenger Class) is the socio-economic status of the passenger and it is a categorical ordinal feature which has 3 unique values (1, 2 or 3):
#     * 1 = Upper Class
#     * 2 = Middle Class
#     * 3 = Lower Class
# * Name, Sex and Age are self-explanatory
# * SibSp is the total number of the passengers' siblings and spouse
# * Parch is the total number of the passengers' parents and children
# * Ticket is the ticket number of the passenger
# * Fare is the passenger fare
# * Cabin is the cabin number of the passenger
# * Embarked is port of embarkation and it is a categorical feature which has 3 unique values (C, Q or S):
#     * C = Cherbourg
#     * Q = Queenstown
#     * S = Southampton

# In[6]:


train.isnull().sum()/train.shape[0]


# In[7]:


test.isnull().sum()/test.shape[0]


# * Training set have missing values in Age, Cabin and Embarked columns
# * Test set have missing values in Age, Cabin and Fare columns

# In[8]:


train.describe().T


# In[9]:


test.describe().T


# # Feature Engineering

# In[10]:


total = pd.concat([train,test],axis=0)


# ### Missing value treatment

# In[11]:


age_by_pclass_sex = total.groupby(['Sex', 'Pclass']).median()['Age']

for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
print('Median age of all passengers: {}'.format(total['Age'].median()))

# Filling the missing values in Age with the medians of Sex and Pclass groups
total['Age'] = total.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median())).reset_index(drop=True)


# In[12]:


# Filling the missing values in Embarked with S
total['Embarked'] = total['Embarked'].fillna('S')


# In[13]:


med_fare = total.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
# Filling the missing value in Fare with the median Fare of 3rd class alone passenger
total['Fare'] = total['Fare'].fillna(med_fare)


# In[14]:


# Creating Deck column from the first letter of the Cabin column (M stands for Missing)
total['Deck'] = total['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

total_decks = total.groupby(['Deck', 'Pclass']).count().drop(columns=['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 
                                                                        'Fare', 'Embarked', 'Cabin', 'PassengerId', 'Ticket']).rename(columns={'Name': 'Count'}).transpose()

def get_pclass_dist(df):
    
    # Creating a dictionary for every passenger class count in every deck
    deck_counts = {'A': {}, 'B': {}, 'C': {}, 'D': {}, 'E': {}, 'F': {}, 'G': {}, 'M': {}, 'T': {}}
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

    # Creating a dictionary for every passenger class percentage in every deck
    for col in df_decks.columns:
        deck_percentages[col] = [(count / df_decks[col].sum()) * 100 for count in df_decks[col]]
        
    return deck_counts, deck_percentages

def display_pclass_dist(percentages):
    
    df_percentages = pd.DataFrame(percentages).transpose()
    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M', 'T')
    bar_count = np.arange(len(deck_names))  
    bar_width = 0.85
    
    pclass1 = df_percentages[0]
    pclass2 = df_percentages[1]
    pclass3 = df_percentages[2]
    
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

all_deck_count, all_deck_per = get_pclass_dist(total_decks)
display_pclass_dist(all_deck_per)


# In[15]:


# Passenger in the T deck is changed to A
idx = total[total['Deck'] == 'T'].index
total.loc[idx, 'Deck'] = 'A'


# In[16]:


total_decks_survived = total.groupby(['Deck', 'Survived']).count().drop(columns=['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                                                                                   'Embarked', 'Pclass', 'Cabin', 'PassengerId', 'Ticket']).rename(columns={'Name':'Count'}).transpose()

def get_survived_dist(df):
    
    # Creating a dictionary for every survival count in every deck
    surv_counts = {'A':{}, 'B':{}, 'C':{}, 'D':{}, 'E':{}, 'F':{}, 'G':{}, 'M':{}}
    decks = df.columns.levels[0]    

    for deck in decks:
        for survive in range(0, 2):
            surv_counts[deck][survive] = df[deck][survive][0]
            
    df_surv = pd.DataFrame(surv_counts)
    surv_percentages = {}

    for col in df_surv.columns:
        surv_percentages[col] = [(count / df_surv[col].sum()) * 100 for count in df_surv[col]]
        
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

all_surv_count, all_surv_per = get_survived_dist(total_decks_survived)
display_surv_dist(all_surv_per)


# In[17]:


total['Deck'] = total['Deck'].replace(['A', 'B', 'C'], 'ABC')
total['Deck'] = total['Deck'].replace(['D', 'E'], 'DE')
total['Deck'] = total['Deck'].replace(['F', 'G'], 'FG')

total['Deck'].value_counts()


# In[18]:


total.drop(['Cabin'], inplace=True, axis=1)


# ### Feature engineering

# In[19]:


total['Fare'] = pd.qcut(total['Fare'], 13)


# In[20]:


total['Age'] = pd.qcut(total['Age'], 9)


# In[21]:


total['Ticket_Frequency'] = total.groupby('Ticket')['Ticket'].transform('count')


# In[22]:


total['Title'] = total['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
total['Is_Married'] = 0
total['Is_Married'].loc[total['Title'] == 'Mrs'] = 1


# In[23]:


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

total['Family'] = extract_surname(total['Name'])


# In[24]:


total['Family_Size'] = total['SibSp'] + total['Parch'] + 1

family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
total['Family_Size_Grouped'] = total['Family_Size'].map(family_map)

total['Title'] = total['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
total['Title'] = total['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')


# In[25]:


train = total.iloc[:891,:]
test = total.iloc[891:,:]

dfs = [train, test]


# In[26]:


# Creating a list of families and tickets that are occuring in both training and test set
non_unique_families = [x for x in train['Family'].unique() if x in test['Family'].unique()]
non_unique_tickets = [x for x in train['Ticket'].unique() if x in test['Ticket'].unique()]

df_family_survival_rate = train.groupby('Family')['Survived', 'Family','Family_Size'].median()
df_ticket_survival_rate = train.groupby('Ticket')['Survived', 'Ticket','Ticket_Frequency'].median()

family_rates = {}
ticket_rates = {}

for i in range(len(df_family_survival_rate)):
    # Checking a family exists in both training and test set, and has members more than 1
    if df_family_survival_rate.index[i] in non_unique_families and df_family_survival_rate.iloc[i, 1] > 1:
        family_rates[df_family_survival_rate.index[i]] = df_family_survival_rate.iloc[i, 0]

for i in range(len(df_ticket_survival_rate)):
    # Checking a ticket exists in both training and test set, and has members more than 1
    if df_ticket_survival_rate.index[i] in non_unique_tickets and df_ticket_survival_rate.iloc[i, 1] > 1:
        ticket_rates[df_ticket_survival_rate.index[i]] = df_ticket_survival_rate.iloc[i, 0]


# In[27]:


mean_survival_rate = np.mean(train['Survived'])

train_family_survival_rate = []
train_family_survival_rate_NA = []
test_family_survival_rate = []
test_family_survival_rate_NA = []

for i in range(len(train)):
    if train['Family'][i] in family_rates:
        train_family_survival_rate.append(family_rates[train['Family'][i]])
        train_family_survival_rate_NA.append(1)
    else:
        train_family_survival_rate.append(mean_survival_rate)
        train_family_survival_rate_NA.append(0)
        
for i in range(len(test)):
    if test['Family'].iloc[i] in family_rates:
        test_family_survival_rate.append(family_rates[test['Family'].iloc[i]])
        test_family_survival_rate_NA.append(1)
    else:
        test_family_survival_rate.append(mean_survival_rate)
        test_family_survival_rate_NA.append(0)
        
train['Family_Survival_Rate'] = train_family_survival_rate
train['Family_Survival_Rate_NA'] = train_family_survival_rate_NA
test['Family_Survival_Rate'] = test_family_survival_rate
test['Family_Survival_Rate_NA'] = test_family_survival_rate_NA

train_ticket_survival_rate = []
train_ticket_survival_rate_NA = []
test_ticket_survival_rate = []
test_ticket_survival_rate_NA = []

for i in range(len(train)):
    if train['Ticket'][i] in ticket_rates:
        train_ticket_survival_rate.append(ticket_rates[train['Ticket'][i]])
        train_ticket_survival_rate_NA.append(1)
    else:
        train_ticket_survival_rate.append(mean_survival_rate)
        train_ticket_survival_rate_NA.append(0)
        
for i in range(len(test)):
    if test['Ticket'].iloc[i] in ticket_rates:
        test_ticket_survival_rate.append(ticket_rates[test['Ticket'].iloc[i]])
        test_ticket_survival_rate_NA.append(1)
    else:
        test_ticket_survival_rate.append(mean_survival_rate)
        test_ticket_survival_rate_NA.append(0)
        
train['Ticket_Survival_Rate'] = train_ticket_survival_rate
train['Ticket_Survival_Rate_NA'] = train_ticket_survival_rate_NA
test['Ticket_Survival_Rate'] = test_ticket_survival_rate
test['Ticket_Survival_Rate_NA'] = test_ticket_survival_rate_NA


# In[28]:


for df in dfs:
    df['Survival_Rate'] = (df['Ticket_Survival_Rate'] + df['Family_Survival_Rate']) / 2
    df['Survival_Rate_NA'] = (df['Ticket_Survival_Rate_NA'] + df['Family_Survival_Rate_NA']) / 2    


# In[29]:


non_numeric_features = ['Embarked', 'Sex', 'Deck', 'Title', 'Family_Size_Grouped', 'Age', 'Fare']

for df in dfs:
    for feature in non_numeric_features:        
        df[feature] = LabelEncoder().fit_transform(df[feature])


# In[30]:


cat_features = ['Pclass', 'Sex', 'Deck', 'Embarked', 'Title', 'Family_Size_Grouped']
encoded_features = []

for df in dfs:
    for feature in cat_features:
        encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()
        n = df[feature].nunique()
        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
        encoded_df = pd.DataFrame(encoded_feat, columns=cols)
        encoded_df.index = df.index
        encoded_features.append(encoded_df)

train = pd.concat([train, *encoded_features[:6]], axis=1)
test = pd.concat([test, *encoded_features[6:]], axis=1)


# In[31]:


total = pd.concat([train, test],axis=0)
drop_cols = ['Deck', 'Embarked', 'Family', 'Family_Size', 'Family_Size_Grouped', 'Survived',
             'Name', 'Parch', 'PassengerId', 'Pclass', 'Sex', 'SibSp', 'Ticket', 'Title',
            'Ticket_Survival_Rate', 'Family_Survival_Rate', 'Ticket_Survival_Rate_NA', 'Family_Survival_Rate_NA']

total.drop(columns=drop_cols, inplace=True, axis=1)

total.head()


# # Model 

# In[32]:


X = StandardScaler().fit_transform(train.drop(columns=drop_cols))
y = train['Survived'].values
X_test = StandardScaler().fit_transform(test.drop(columns=drop_cols))

print('X_train shape: {}'.format(X.shape))
print('y_train shape: {}'.format(y.shape))
print('X_test shape: {}'.format(X_test.shape))


# In[33]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y , test_size=0.3, stratify=y, random_state=42)


# In[34]:


clf = LGBMClassifier(random_state=42)

clf.fit(X_train, y_train)


# In[35]:


valid_preds = clf.predict(X_valid)
print(accuracy_score(y_valid, valid_preds))


# # 📁 Submission file

# In[36]:


ss['Survived'] = clf.predict(X_test).astype("int")
ss.to_csv("/kaggle/working/submission.csv", index=False)


# # Kindly Upvote, if you like this notebook.
