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


# ## Titanic Competition Notebook
# Hi this is my first attempt on publishing a public notebook. I will be doing in depth data analysis and feature engineering of the Titanic dataset, I will also apply some well-known machine learning model and pick the best model for the submission.  
# Feel free to comments your thoughts and opinion in my analysis, i will add some additional footnote on how to improve this model.  
# Thank you for your time, hope this helps you!

# In[2]:


# Import Packages
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Read the Dataset
train_path = '/kaggle/input/titanic/train.csv'
test_path = '/kaggle/input/titanic/test.csv'

dftrain = pd.read_csv(train_path)
dftest = pd.read_csv(test_path)


# In[4]:


# Get column and row data for training and testing
print('train data have '+str(len(dftrain.index))+' rows')
print('train data have '+str(len(dftrain.columns))+' column')
print('test data have '+str(len(dftest.index))+' rows')
print('test data have '+str(len(dftest.columns))+' column')


# In[5]:


dftrain.head(5)


# ---

# ## Step 1. Initial Analysis
# The dataset consists of 891 entries of Training data and 418 entries of Testing data, we will not use validation split on training so we can fully utilize our limited data

# In[6]:


dftrain.info()


# In[7]:


dftest.info()


# In[8]:


# Find missing values
print('Training missing values : ')
for column in dftrain.columns:
    if dftrain[column].count() < dftrain.shape[0]:
        print( 'Column {} has {} missing values'.format(column, dftrain.shape[0]-dftrain[column].count()) )
print('-'*40)
print('Test missing values : ')
for column in dftest.columns:
    if dftest[column].count() < dftest.shape[0]:
        print( 'Column {} has {} missing values'.format(column, dftest.shape[0]-dftest[column].count()) )


# Data type and Empty Value analysis
# 
# 1. **Age** column uses float64 instead of integer
# 2. **Age** and **Cabin** column have the most missing data for both training and testing
# 3. We will explore the missing values further and determine the best way to handle them, some common methods are (**dropping** the row, **replace** with another value, and **leaving** as null value)

# ---

# ## Step 2. EDA & Feature Engineering
# We will do further analysis by exploring relationship and behaviour of our data, we will use this information to help determined the best model

# In[9]:


# Identify object and numerical columns
obj_cols = dftrain.describe(include='object').columns.tolist()
num_cols = dftrain.describe(include='number').columns.tolist()
print('There are {} object columns'.format(len(obj_cols)))
print(obj_cols)
print('There are {} numerical columns'.format(len(num_cols)))
print(num_cols)


# In[10]:


dftrain.describe(include='object')


# In[11]:


dftrain.describe(include='number')


# Behaviour of the data analysis
# 
# 1. There are no duplicate entries
# 2. Most passangers are males **(64.76%)**
# 3. The survivability rate is low **(38.38%)**
# 4. There are possible outliers in **SibSp**, **Parch**, and **Fare** column
# 5. There are possible abnormal data in column **Age**

# We will get better insight by using visualization

# In[12]:


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

sns.histplot(dftrain['Pclass'].astype('category'), ax=axes[0,0])
axes[0,0].set_title('Pclass')

sns.histplot(dftrain['Age'], bins = 10, ax=axes[0,1])
axes[0,1].set_title('Age')

sns.histplot(dftrain['Fare'], bins = 10, ax=axes[0,2])
axes[0,2].set_title('Fare')

sns.histplot(dftrain['Survived'].astype('category'), ax=axes[1,0])
axes[1,0].set_title('Survived')

sns.histplot(dftrain['SibSp'].astype('category'), ax=axes[1,1])
axes[1,1].set_title('SibSp')

sns.histplot(dftrain['Parch'].astype('category'), ax=axes[1,2])
axes[1,2].set_title('Parch')

plt.tight_layout()

plt.show()


# There are an apparent outliers in **SibSp**, **Fare**, and **Parch** columns, let's analyze them further

# In[13]:


# Outlier detection
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

sns.boxplot(dftrain['Fare'],ax=axes[0])
axes[0].set_title('Fare')

sns.boxplot(dftrain['SibSp'],ax=axes[1])
axes[1].set_title('SibSp')

sns.boxplot(dftrain['Parch'],ax=axes[2])
axes[2].set_title('Parch')

plt.tight_layout()

plt.show()


# based on observation above :
# 1. **Fare** column have max value of 512.33 while the median is 14.45
# 2. **SibSp** column have max value of 8 while the median is 0
# 3. **Parch** column have max value of 6 while median is 0  
# 
# the outliers in **Fare** and **SibSp** columns are significant, we might have to handle this to reduce noise in our training data

# ### Analyze SibSp and Parch columns
# SibSp and Parch of a passanger correlate to the number of how many other passangers are abroad which have familial bond wether ts Sibling/Spouse or Parent/Children.  

# In[14]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
sns.countplot(x=dftrain['SibSp'].astype('category'),
              hue=dftrain['Survived'], 
              palette='deep',ax=axes[0])
sns.countplot(x=dftrain['Parch'].astype('category'),
              hue=dftrain['Survived'], 
              palette='deep',ax=axes[1])
plt.show()
plt.tight_layout()


# In[15]:


frequency = dftrain.groupby(['SibSp', 'Parch']).size().reset_index(name='Frequency')
sns.scatterplot(x=frequency['SibSp'].astype('category'),
                y=frequency['Parch'].astype('category'),
                size=frequency['Frequency'], 
                sizes=(30,400)).set_title('Combination of SibSp and Parch Frequency plot')


# Based on observation above :
# 1. There are some portion of passangers that **is alone** (SibSp = 0, Parch = 0)
# 2. The distribution of SibSp and Parch are **similiar**
# 3. We handle outliers by **removing** the row
# 3. We can use both column to **generate** a more useful feature

# Our new feature is as followed :
# 1. When there are 0 SibSp and 0 Parch labeled **Is_Alone**
# 2. When passangers have 1 or more SibSp labeled **Have_SibSp**
# 3. When passangers have 1 or more Parch labeled **Have_Parch**
# 4. SibSp + Parch + 1 labeled **Group_Size**

# In[16]:


# Generating feature for train and test
for df in [dftrain, dftest] :
    df['Have_Parch'] = df['Parch'].apply(lambda x: 1 if x > 0 else 0).astype(int)
    df['Have_SibSp'] = df['SibSp'].apply(lambda x: 1 if x > 0 else 0).astype(int)
    df['Is_Alone'] = df.apply(lambda row: 1 if row['SibSp'] + row['Parch'] == 0 else 0, axis=1).astype(int)
    df['Group_Size'] = (df['SibSp']+df['Parch']+1).astype(int)


# to handle the outliers, we will bin the **Group_Size** feature into bins to remove noise in the data

# In[17]:


# Bin the group to 4 different categories
group_bins = [0, 1, 4, 6, 12]
for df in [dftrain, dftest]:
    df['Group_Category'] = pd.cut(df['Group_Size'], 
                                  bins=group_bins, 
                                  labels=['Alone', 'Small', 'Medium', 'Large'])
    df['Group_Category'] = df['Group_Category'].map({'Alone':0,'Small':1,'Medium':2,'Large':3}).astype(int)


# In[18]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

sns.countplot(x=dftrain['Have_Parch'], hue=dftrain['Survived'], ax=axes[0,0], palette='deep')
axes[0,0].set_title('Have_Parch')

sns.countplot(x=dftrain['Have_SibSp'], hue=dftrain['Survived'], ax=axes[0,1], palette='deep')
axes[0,1].set_title('Have_SibSp')

sns.countplot(x=dftrain['Is_Alone'], hue=dftrain['Survived'], ax=axes[1,0], palette='deep')
axes[1,0].set_title('Is_Alone')

sns.countplot(x=dftrain['Group_Category'], hue=dftrain['Survived'], ax=axes[1,1], palette='deep')
axes[1,1].set_title('Group_Category')

plt.tight_layout()

plt.show()


# - Passangers with small group size **(2-4 people)** tends to survive compared to other group size ranges
# - Passangers that are **Alone** is most likely did not survive

# In[19]:


columns = ['Survived','Is_Alone','Have_Parch','Have_SibSp','Group_Category']
sns.heatmap(dftrain[columns].corr(), vmax=1,vmin=-1, annot=True, cmap='coolwarm')


# ### Analyze Age columns
# The Age value of a passanger might affect their survivability. In a disaster, most life saving equipments are prioritized to children and older people

# In[20]:


# Bin age to remove abnormal values and remove noise
age_bins = [0, 5, 20, 30, 40, 50, 60, 100]
for df in [dftrain, dftest]:
    df['Age_Category'] = pd.cut(df['Age'], 
                                bins=age_bins, 
                                labels=['Infant', 'Teen', '20s', '30s', '40s', '50s', 'Elderly'])


# In[21]:


sns.countplot(x=dftrain['Age_Category'], hue=dftrain['Sex'], palette='deep')


# In[22]:


# Check unique values 
dftrain['Age_Category'].unique() 


# There are still missing values denoted by **NaN**

# In[23]:


sns.barplot(x=dftrain['Age_Category'], y=dftrain['Survived'], hue=dftrain['Sex'], palette='deep')


# We can conclude that :
# 1. Most passangers are Grown Ups (Age range 20-30)
# 2. Infants have the most survivability rate
# 3. In all range of ages, male have higher count
# 4. In all range of ages, female have higher survivability

# In[24]:


grid = sns.FacetGrid(dftrain, row='Pclass',col='Sex')
grid.map(sns.countplot, 'Age_Category', palette='deep')
grid.add_legend()


# Seperated by the Pclass, we can conclude that :
# 
# 1. There are different **Age_Category** with the most count in different Pclass, and Is_Alone values
# 2. We can use this to replace the missing value of **Age_Category** column
# 3. We can assume that **Is_Alone** might cause the count of **Age_Category** differ even more

# **Handling Missing Values**  
# the common option is to replace the missing values with the mean or median of combined dataset.   However, we might argue that for different distribution of passangers, will have different mean or median.  
# We propose to replace the missing values by the mean or median of combined dataset based on a **Selected Column** to better specify the most likely value

# In[25]:


# Using both data combined, create a table consisting of combination of selected columns, then find the most frequent age category
combined = pd.concat([dftrain, dftest], sort=True).reset_index(drop=True)

selected_column = ['Pclass','Sex','Is_Alone']
frequent_category = combined.groupby(selected_column)['Age_Category'].apply(lambda x: x.value_counts().idxmax()).to_frame().reset_index()
missing_age = pd.DataFrame(dftrain[dftrain['Age_Category'].isna()][selected_column].value_counts()).sort_values(selected_column)
pd.merge(frequent_category, missing_age, on=selected_column).reset_index(drop=True)


# - This table represents the most count of **Age_Category** for each possible **Pclass**,**Sex**, and **Is_Alone** value.  
# - The 0 column represents the sum of missing entries in training data
# - We can see that lower **Pclass** value tends to have higher **Age_Category**
# - Passangers that travel **alone** is more likely to be older

# In[26]:


# Get bottom 5 entries with missing age category
dftrain[dftrain['Age_Category'].isna()][['Pclass','Sex','Is_Alone','Age_Category']].tail(5)


# We can fill the last 5 entries of Age_Category using the table above. Assume (Pclass, Sex, Is Alone)
# 
# - (3,male,1) then Age_Category will be 20s
# - (3,female, 0) then Age_Category will be Teen
# - (3,male,1) then Age_Category will be 20s
# - (3,male,1) then Age_Category will be 20s
# - (3,female,0) then Age_Category will be Teen

# In[27]:


# Replace missing values on both train and test data with the most frequent age category
print('Missing values before replacement, training : {}, test : {}'.format(dftrain['Age_Category'].isna().sum(),dftest['Age_Category'].isna().sum()))

dftrain = dftrain.merge(frequent_category, on=selected_column, how='left', suffixes=('', '_Replacement'))
dftrain['Age_Category'] = dftrain['Age_Category'].fillna(dftrain['Age_Category_Replacement'])

dftest = dftest.merge(frequent_category, on=selected_column, how='left', suffixes=('', '_Replacement'))
dftest['Age_Category'] = dftest['Age_Category'].fillna(dftest['Age_Category_Replacement'])

print('Missing values after replacement, training : {}, test : {}'.format(dftrain['Age_Category'].isna().sum(),dftest['Age_Category'].isna().sum()))


# In[28]:


dftrain = dftrain.drop(columns=['Age','Age_Category_Replacement'])
dftest = dftest.drop(columns=['Age','Age_Category_Replacement'])


# ### Analyze Fare columns
# the Fare value of a passanger denotes the price of the trip for each passangers. There are some Passangers that have the same Fares, this shows that for the price paid might be a Total Fare instead of an Individual Fare.

# **The test data have a missing value and we will handle by replace with median of combined data**  
# The difference between the method for the age column is we will replace it manually since there are only 1 missing value

# In[29]:


# Get the entry with missing Fare values
dftest[dftest['Fare'].isna()][['Pclass','Sex','Is_Alone','Age_Category']]


# We will calculate the median using the four column **Pclass**, **Sex**, **Is_Alone**, and **Age_Category**

# In[30]:


# Combine both train and test data to find the mean value of fare for combination of selected columns
combined = pd.concat([dftrain, dftest], sort=True).reset_index(drop=True)

selected_column = ['Pclass','Sex','Is_Alone','Age_Category']
mean_fare = combined.groupby(selected_column)['Fare'].mean().to_frame().reset_index()


# In[31]:


mean_fare


# We can conclude that Fare value is higher for **lower Pclass**, **females**, **not alone**, and **younger age category**

# In[32]:


# We manually replace the missing values by finding the correct mean value
replacement_fare = mean_fare[(mean_fare['Pclass']==3) & (mean_fare['Sex']=='male') & (mean_fare['Is_Alone']==1) & (mean_fare['Age_Category']=='Elderly')]['Fare']
print('We will replace the missing value with {}'.format(replacement_fare.values[0]))


# In[33]:


# Replace the said missing values
dftest['Fare'] = dftest['Fare'].fillna(replacement_fare.values[0])
print('There are {} missing values in test set'.format(dftest['Fare'].isna().sum()))


# In[34]:


# Generate Fare Individual column
for df in [dftrain,dftest]:
    df['Fare_Individual'] = df['Fare']/df['Group_Size']
    df.rename(columns={'Fare': 'Fare_Total'},inplace=True)
    df[['Fare_Total','Fare_Individual']] = df[['Fare_Total','Fare_Individual']].round(2).astype(float)


# In[35]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
sns.histplot(dftrain['Fare_Total'],  bins=20, ax=axes[0,0])
sns.histplot(dftrain['Fare_Individual'],  bins=20, ax=axes[0,1])
sns.barplot(x=dftrain['Group_Size'], y=dftrain['Fare_Total'], ax=axes[1,0])
sns.barplot(x=dftrain['Group_Size'], y=dftrain['Fare_Individual'], ax=axes[1,1])


# We will bin the fare values to remove noise and handle outliers

# In[36]:


dftrain['Fare_Total_Bins'] = pd.qcut(dftrain['Fare_Total'], 6)
pd.qcut(dftrain['Fare_Total'], 6).unique()


# In[37]:


dftrain['Fare_Indiv_Bins'] = pd.qcut(dftrain['Fare_Individual'], 6)
pd.qcut(dftrain['Fare_Individual'], 6).unique()


# In[38]:


# Bin the Fare Total and Fare Individual columns
for df in [dftrain,dftest] :
    df.loc[ df['Fare_Total'] <= 7.78, 'Fare_Total'] = 0
    df.loc[(df['Fare_Total'] > 7.78) & (df['Fare_Total'] <= 8.66), 'Fare_Total'] = 1
    df.loc[(df['Fare_Total'] > 8.66) & (df['Fare_Total'] <= 14.45), 'Fare_Total'] = 2
    df.loc[(df['Fare_Total'] > 14.45) & (df['Fare_Total'] <= 26.0), 'Fare_Total'] = 3
    df.loc[(df['Fare_Total'] > 26.0) & (df['Fare_Total'] <= 52.37), 'Fare_Total'] = 4
    df.loc[ df['Fare_Total'] > 52.37, 'Fare_Total'] = 5
    df['Fare_Total'] = df['Fare_Total'].astype(int)

    df.loc[ df['Fare_Individual'] <= 6.75, 'Fare_Individual'] = 0
    df.loc[(df['Fare_Individual'] > 6.75) & (df['Fare_Individual'] <= 7.78), 'Fare_Individual'] = 1
    df.loc[(df['Fare_Individual'] > 7.78) & (df['Fare_Individual'] <= 8.3), 'Fare_Individual'] = 2
    df.loc[(df['Fare_Individual'] > 8.3) & (df['Fare_Individual'] <= 13.0), 'Fare_Individual'] = 3
    df.loc[(df['Fare_Individual'] > 13.0) & (df['Fare_Individual'] <= 29.7), 'Fare_Individual'] = 4
    df.loc[ df['Fare_Individual'] > 29.7, 'Fare_Individual'] = 5
    df['Fare_Individual'] = df['Fare_Individual'].astype(int)


# In[39]:


dftrain = dftrain.drop(['Fare_Total_Bins','Fare_Indiv_Bins'], axis=1)


# In[40]:


grid = sns.FacetGrid(dftrain, col='Is_Alone')
grid.map(sns.barplot, 'Fare_Total', 'Survived', palette='deep')
grid.add_legend()


# In[41]:


grid = sns.FacetGrid(dftrain, col='Is_Alone')
grid.map(sns.barplot, 'Fare_Individual', 'Survived', palette='deep')
grid.add_legend()


# In[42]:


columns = ['Survived','Fare_Total','Fare_Individual','Is_Alone','Group_Size']
sns.heatmap(dftrain[columns].corr(), vmax=1,vmin=-1, annot=True, cmap='coolwarm')


# Using the heatmap and analysis above we can see that
# 
# 1. both **Total Fare** and **Individual Fare** are good predictors for **Survived** columns
# 2. **Total Fare** and **Individual Fare** are correlated but have different relationship on **Group Size** or **Is Alone**

# ### Analyze Categorical Feature (Name, Sex, Ticket, Cabin, and Embarked )

# ### Extract Title from Name
# A title can be represented with a single word with a dot (.) in the end, we will use **RegEx** to find them  
# We can use Pandas **str.extract** method to apply RegEx on entire column

# In[43]:


# Apply RegEx pattern to extract title
pattern = ' ([A-Za-z]+)\.'
for df in [dftrain, dftest] :
    df['Title'] = df['Name'].str.extract(pattern)


# In[44]:


print(dftrain['Title'].unique())
print(dftest['Title'].unique())


# In[45]:


# Replace unique titles with selected titles
for df in [dftrain,dftest] :
    df['Title'] = df['Title'].replace(['Capt', 'Col','Don','Major', 'Rev', 'Sir','Lady', 'Countess', 'Dona','Dr', 'Rev','Jonkheer'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle','Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')


# In[46]:


fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x=dftrain['Title'],y=dftrain['Survived'])


# In[47]:


dftrain = dftrain.drop(columns=['Name'])
dftest = dftest.drop(columns=['Name'])


# The **Title** feature extracted from passanger names is a good predictor
# 1. Each entries title are successfully extracted
# 2. Mr. is the title with the most count have the lowest survivability
# 3. The Rare title consist of combination of male and female titles

# ### Analyze Ticket features
# For ticket column, we will use RegEx to extract information  
# The number will ignored and the first character of a word will be extracted

# In[48]:


# Appy RegEx pattern to extract the ticket type
# We ignore numbers and extract every first character on a single entries
# We Replace words like STON or PARIS with their first letter S or P
pattern = '([A-Z\.\/]*).* [0-9]*'
for df in [dftrain, dftest] :
    Ticket_Extract = df['Ticket'].str.extract(pattern).reset_index()
    Ticket_Clean = Ticket_Extract[0].astype(str).str.split('[./]').apply(lambda x: [i.strip() for i in x if i.strip()])
    Ticket_Clean = Ticket_Clean.apply(lambda x: ['S' if item in ['STON', 'SOTON'] else item for item in x])
    Ticket_Clean = Ticket_Clean.apply(lambda x: ['P' if item in ['PARIS'] else item for item in x])
    Ticket_Clean = Ticket_Clean.apply(lambda x: list(set(''.join(x))))
    
    # Generate features corresponding to each ticket type
    df.reset_index(drop=True, inplace=True)
    df['TicketA'] = Ticket_Clean.apply(lambda x: int('A' in x))
    df['TicketP'] = Ticket_Clean.apply(lambda x: int('P' in x))
    df['TicketC'] = Ticket_Clean.apply(lambda x: int('C' in x))
    df['TicketS'] = Ticket_Clean.apply(lambda x: int('S' in x))
    df['TicketQ'] = Ticket_Clean.apply(lambda x: int('Q' in x))
    df['TicketO'] = Ticket_Clean.apply(lambda x: int('O' in x))
    df['TicketW'] = Ticket_Clean.apply(lambda x: int('W' in x))
    df['Have_Ticket'] = Ticket_Clean.apply(lambda x: int(any(['A' in x, 'P' in x, 'C' in x, 'S' in x, 'Q' in x, 'O' in x, 'W' in x])))


# In[49]:


# Get the count of every unique ticket type
columns = ['TicketQ','TicketW','TicketO','TicketA','TicketS','TicketP','TicketC','Have_Ticket']
for column in columns:
    print('column {} have {} entries'.format(column,dftrain[column].sum()))


# Because of small number of entries we will group the tickets types

# In[50]:


# Relation of each ticket type to Survivability
dftrain[['TicketQ','TicketW','TicketO','TicketA','TicketS','TicketP','TicketC','Have_Ticket','Survived']].corr()['Survived'].sort_values()


# Based on the correlation, we can group ticket A,O,Q,S,W into **type1** and ticket P and C into **type2**

# In[51]:


# Group ticket into type1 and type2
for df in [dftrain,dftest] :
    df['Ticket_Type1'] = df['TicketQ'] | df['TicketW'] | df['TicketO'] | df['TicketA'] | df['TicketS']
    df['Ticket_Type2'] = df['TicketP'] | df['TicketC']


# In[52]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
sns.countplot(x=dftrain['Ticket_Type1'],hue=dftrain['Survived'],ax=axes[0], palette='deep')
sns.countplot(x=dftrain['Ticket_Type2'],hue=dftrain['Survived'],ax=axes[1], palette='deep')


# In[53]:


fig, ax = plt.subplots(figsize=(8, 6))
columns = ['Survived','Ticket_Type1','Ticket_Type2','Have_Ticket']
correlation_matrix = dftrain[columns].corr()
sns.heatmap(correlation_matrix, vmax=1,vmin=-1, annot=True, cmap='coolwarm')

plt.show()


# In[54]:


dftrain = dftrain.drop(columns=['Ticket','TicketA', 'TicketP', 'TicketC', 'TicketS', 'TicketQ', 'TicketO', 'TicketW','Have_Ticket'])
dftest = dftest.drop(columns=['Ticket','TicketA', 'TicketP', 'TicketC', 'TicketS', 'TicketQ', 'TicketO', 'TicketW','Have_Ticket'])


# The **Ticket** feature might not provide meaningful information
# - There are only 206 entries which the ticket type can be extracted
# - The correlation between each ticket type and survivability is low
# - We grouped ticket type1 as (Q,W,O,A,S) and ticket type2 as (P,C)

# #### Analyze Cabin feature
# - The cabin feature have the most missing values in both training and testing data, dropping the column is a good option
# - Some entries have multiple cabins listed
# - The structure of the data is the cabin type and the cabin ID, e.g. C85

# In[55]:


sns.barplot(x=dftrain['Cabin'].astype(str).apply(lambda x: x[0]),y=dftrain['Survived'])


# In[56]:


# Extract the first letter from the Cabin column
for df in[dftrain,dftest] :
    df['Cabin'] = df['Cabin'].astype(str).apply(lambda x: x[0])
    df['Cabin'] = df['Cabin'].replace(['A','F','G','T'],'Other')
    df['Cabin'] = df['Cabin'].replace(['n'],'Unlisted')


# In[57]:


sns.countplot(x=dftrain['Cabin'],hue=dftrain['Survived'], palette='deep')


# The **Cabin** feature might provide meaningful information
# - There are 671 unlisted entries, mostly not survive
# - The cabin with most entries are C,B,D, and E. Each have rather high survivability
# - The cabin with lesser entries are grouped into other, is not a good predictor

# #### Analyze Embarked feature
# - Embarked values are mostly categorical with 3 distinct values
# - Training data have a missing value, can be replaced by mode

# In[58]:


# Combine both train and test data then replace missing value with the mode
combined = pd.concat([dftrain, dftest], sort=True).reset_index(drop=True)

frequent_embarked = combined['Embarked'].mode()[0]
print('We will replace the missing value with {}'.format(frequent_embarked))


# In[59]:


dftrain['Embarked'] = dftrain['Embarked'].fillna(frequent_embarked)


# In[60]:


sns.countplot(x=dftrain['Embarked'],hue=dftrain['Survived'], palette='deep')


# In[61]:


# Map the Sex and Age Category columns into numerical value
# We use discrete value because the age category is still a continuous value
for df in [dftrain,dftest] :
    df['Age_Category'] = df['Age_Category'].map({'Infant':0,'Teen':1,'20s':2,'30s':3,'40s':4,'50s':5,'Elderly':6})


# In[62]:


# We one-hot-encode categorical values into separate columns
dftrain = pd.get_dummies(dftrain, columns=['Cabin','Embarked','Title','Sex'])
dftest = pd.get_dummies(dftest, columns=['Cabin','Embarked','Title','Sex'])


# In[63]:


# Drop unnecessary columns
dftrain = dftrain.drop(columns=['PassengerId','Parch','SibSp','Group_Size'])
dftest = dftest.drop(columns=['Parch','SibSp','Group_Size'])


# In[64]:


from sklearn.feature_selection import f_classif
X = dftrain.drop(columns='Survived')
y = dftrain['Survived']
f_stats, p_val = f_classif(X,y)
data = pd.DataFrame(data=p_val.round(2),columns=['p_val'],index=X.columns).sort_values(by='p_val',ascending=False).T
data


# In[65]:


cols = [c for c in X if data[c].mean() < 0.5 and c in list(dftest.columns)]
dftrain = dftrain[cols+['Survived']]
dftest = dftest[cols+['PassengerId']]


# In[66]:


fig, ax = plt.subplots(figsize=(20, 16))
sns.heatmap(round(dftrain.corr(),2),vmax=1,vmin=-1, annot=True, cmap='coolwarm')


# ---

# ## Step 3. Create a prediction Model

# In[67]:


# machine learning packages
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# In[68]:


# Split into training and testing data
X_train = dftrain.drop("Survived", axis=1).astype(float)
Y_train = dftrain["Survived"].astype(int)
X_test  = dftest.drop(columns=['PassengerId']).copy().astype(float)
X_train.shape, Y_train.shape, X_test.shape


# In[69]:


# Normalize both train and test input
scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[70]:


# Split into train and validation for model selection
x_train, x_valid, y_train, y_valid = train_test_split(X_train, 
                                                      Y_train, 
                                                      train_size=0.8, 
                                                      random_state=1)


# In[71]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score as acc


# ### Model 1 **RFC**

# In[72]:


rfc = RandomForestClassifier(random_state=42,
                            n_estimators=90,
                            min_samples_split=4,
                            max_depth=6)

scores = cross_val_score(rfc, X_train, Y_train, cv=5, scoring='accuracy', verbose=True)
rfc_scores=scores.mean()
print(scores)
print("Kfold on rfc: %0.4f (+/- %0.4f)" % (rfc_scores, scores.std()))


# ### Model 2 **XGBoost**

# In[73]:


xgb = XGBClassifier(learning_rate= 0.005, 
                    subsample= 1,
                    n_estimators= 850,
                    max_depth = 6)

scores = cross_val_score(xgb, X_train, Y_train, cv=5, scoring='accuracy', verbose=True)
xgb_scores=scores.mean()
print(scores)
print("Kfold on xgb: %0.4f (+/- %0.4f)" % (xgb_scores, scores.std()))


# ### Model 3 **LightGBM**

# In[74]:


lgb = LGBMClassifier(learning_rate=0.1,
                     subsample=1,
                     n_estimators= 400,
                     max_depth = 6,
                     objective='binary')

scores = cross_val_score(lgb, X_train, Y_train, cv=5, scoring='accuracy', verbose=True)
lgb_scores=scores.mean()
print(scores)
print("Kfold on lgb: %0.4f (+/- %0.4f)" % (lgb_scores, scores.std()))


# ### Model performance summary

# In[75]:


models = pd.DataFrame({
    'Model': ['Random Forest Classifier', 'XGboost','LightGBM'],
    'Acc': [rfc_scores, xgb_scores, lgb_scores]})
models.sort_values(by='Acc', ascending=False)


# ### Create a submission

# In[76]:


highest_score = models.sort_values(by='Acc', ascending=False).reset_index().iloc[0]['Model']
highest_score


# In[77]:


if highest_score == 'Random Forest Classifier' :
    Submission_Model = rfc.fit(X_train,Y_train)
elif highest_score == 'XGboost' :
    Submission_Model = xgb.fit(X_train,Y_train)
elif highest_score == 'LightGBM' :
    Submission_Model = lgb.fit(X_train,Y_train)


# In[78]:


print(classification_report(Y_train,Submission_Model.predict(X_train)))


# In[79]:


cm = confusion_matrix(Y_train,Submission_Model.predict(X_train))

sns.heatmap(cm, annot = True)


# In[80]:


submission = pd.DataFrame({
        "PassengerId": dftest["PassengerId"],
        "Survived": Submission_Model.predict(X_test)
    })
submission.to_csv('submission.csv', index=False)


# ## Footnote
# Approach on improving the model :
# - I want to experiment the effect of feature data. Currently, most of the data are discrete bins and categorical data, something to try is to treat discrete bins as categorical data which results in more features or did not use discrete bins at all
# - I want to try to add feature column of SibSp and Parch as discrete bins data
# - I want to try to seperate the Rare columns to Rare_Males and Rare_Females
# - I want to try to seperate the Other cabin columns to Other_Class1, Other_Class2, and Other_Class3
# - I want to try to extract the numerical value of the ticket data and how it relates to other feature
# - I might try to use some more advanced machine learning models, some of them are Catboost, LightGBM, and XGBoost.
# - I might try a neural machine learning approach with Artificial Neural Network
# - Some method i want to try is ensemble learning to combine multiple model and hopefully make a better predictor
