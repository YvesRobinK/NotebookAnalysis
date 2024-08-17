#!/usr/bin/env python
# coding: utf-8

# # Data Field Descriptions
# 
# - `PassengerId` - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
# - `HomePlanet` - The planet the passenger departed from, typically their planet of permanent residence.
# - `CryoSleep` - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
# - `Cabin` - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
# - `Destination` - The planet the passenger will be debarking to.
# - `Age` - The age of the passenger.
# - `VIP` - Whether the passenger has paid for special VIP service during the voyage.
# - `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
# - `Name` - The first and last names of the passenger.
# - `Transported` - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

# ----

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# ----

# # Load Datasets

# In[2]:


data = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
test = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')


# In[3]:


data.head()


# In[4]:


data.shape


# We have 8693 observations and 14 columns.

# Lets get some general information about the DataFrame. First of all check if we have any missing values in our dataset.

# In[5]:


print('Missing values in TRAINSET')
print(data.isnull().sum())
print('')
print('Missing values in TESTSET')
print(test.isnull().sum())


# Almost every column has missing values.

# In[6]:


data.info()


# We have 6 continuous features and 5 object features. LightGBM can handle categorical features, we will see if we can convert the objects to categoricals. 

# ----

# # Exploratory Data Analysis
# 
# ## Target Distribution
# First we have a closer look on our target, since it is the main focus of our analysis. We begin by checking the distribution of its classes.

# In[7]:


plt.figure(figsize=(8, 4))
sns.countplot(data['Transported'], palette='Set1')

# Count number of observations in each class
true, false = data['Transported'].value_counts()
print('No. Transported: ', true)
print('No. not Transported : ', false)
print('')
print('% of persons labeled Transported', round(true / len(data) * 100, 2), '%')
print('% of persons labeled not Transported', round(false / len(data) * 100, 2), '%')


# The two classes are almost ballanced. No need to apply some imbalanced dataset techniques like SMOTE.

# ## Categorical Features

# In[8]:


cats=['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
fig=plt.figure(figsize=(10,16))
for i, var_name in enumerate(cats):
    ax=fig.add_subplot(5,1,i+1)
    sns.countplot(data=data, x=var_name, axes=ax, hue='Transported', palette='Set1')
    ax.set_title(var_name)
fig.tight_layout()
plt.show()


# `VIP` does not appear to be a useful feature, the target split is more or less equal. We will drop the `VIP` column later.

# ## Continuous Features

# In[9]:


cont_features = ['Transported', 'Age', 'RoomService' , 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']


# In[10]:


sns.pairplot(data=data[cont_features], hue='Transported', palette='Set1')


# In[11]:


plt.figure(figsize=(20,8))
sns.histplot(data=data, x='Age', hue='Transported', binwidth=1, kde=True, palette='Set1')
plt.title('Age distribution')
plt.xlabel('Age (years)')
plt.show();


# From the above diagram we can see some interesting insights. People between 0 and 18 are more likely to be transported than not. Between 18-27 year olds were less likely to be transported than not. The rest is almost equally distributed. One idea would be to generate a new feature `age_group`. This feature indicates the passenger is wheather a child, teenager or adult. 

# In[12]:


features = ['RoomService' , 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for feature in features:
    plt.subplot(1, 2, 1)
    plt.hist(data[feature], bins=20)
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.title(f'Distribution {feature}')
    plt.tight_layout()
    plt.show()


# The features are very skewed. Probably log transform will reduce the skew. 

# Let´s plot a correlation matrix to see if there exists some correlations between the variables. 

# In[13]:


corr = data.corr().round(2)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(20, 20))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
plt.tight_layout()


# No high correlation between variables is in the dataset. We should keep all columns. 

# We still haven´t investigated the variables `PassengerId`, `Cabin` and `Name`. We can´t plot this data yet. We need to transform them into meaningful features. 
# 
# The data field description says:
# - `PassengerId` - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
# - `Cabin` - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
# - `Name` - The first and last names of the passenger.

# ## PassengerId

# In[14]:


data['Group'] = data['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
test['Group'] = test['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)


# In[15]:


data['GroupSize'] = data['Group'].map(lambda x: pd.concat([data['Group'], test['Group']]).value_counts()[x])
test['GroupSize'] = test['Group'].map(lambda x: pd.concat([data['Group'], test['Group']]).value_counts()[x])


# In[16]:


plt.figure(figsize=(20,4))
plt.subplot(1,2,1)
sns.histplot(data=data, x='Group', hue='Transported', binwidth=1, palette='Set1')
plt.title('Group')
plt.subplot(1,2,2)
sns.countplot(data=data, x='GroupSize', hue='Transported', palette='Set1')
plt.title('Group size')
plt.show();


# We can not use `Group` as a feature. This would explode the number of dimension if we whould like to one-hot encode this variable. The extracted `GroupSize` on the other hand could be a very useful feature. From that we can maybe extract another feature by creating a `Solo` variable which indicates if the passengers are traveling by their own. The figure shows that groups with size = 1 is less likely to be transported than groups with more than one person. 

# ## Cabin

# Now let´s have a look at the `Cabin` column. We can extract the deck, num and side from that variable. By the way, this step and the step before should be part of the feature engineering since we extract new features from existing variables. But I want to have all visualizations and explainations in the EDA part. We have missing data in this column, so first of all let´s mark the NaNs with a default value. 

# In[17]:


data['Cabin'].fillna('ZZZ/-999/ZZZ', inplace=True)
test['Cabin'].fillna('ZZZ/-999/ZZZ', inplace=True)


# In[18]:


data['CabinDeck'] = data['Cabin'].apply(lambda x: x.split('/')[0])
data['CabinNum'] = data['Cabin'].apply(lambda x: x.split('/')[1]).astype(int)
data['CabinSide'] = data['Cabin'].apply(lambda x: x.split('/')[2])

test['CabinDeck'] = test['Cabin'].apply(lambda x: x.split('/')[0])
test['CabinNum'] = test['Cabin'].apply(lambda x: x.split('/')[1]).astype(int)
test['CabinSide'] = test['Cabin'].apply(lambda x: x.split('/')[2])


# And now let´s put the NaN´s back in place. 

# In[19]:


data.loc[data['CabinDeck'] == 'ZZZ', 'CabinDeck'] = np.nan
data.loc[data['CabinNum'] == -999, 'CabinNum'] = np.nan
data.loc[data['CabinSide'] == 'ZZZ', 'CabinSide'] = np.nan

test.loc[data['CabinDeck'] == 'ZZZ', 'CabinDeck'] = np.nan
test.loc[data['CabinNum'] == -999, 'CabinNum'] = np.nan
test.loc[data['CabinSide'] == 'ZZZ', 'CabinSide'] = np.nan


# In[20]:


decks = sorted(list(data['CabinDeck'].value_counts().index))


# Now we are ready to plot our new features. 

# In[21]:


plt.figure(figsize=(20,8))
sns.countplot(data=data, x='CabinDeck', hue='Transported', order=decks, palette='Set1')
plt.show();


# In[22]:


plt.figure(figsize=(20,8))
sns.histplot(data=data, x='CabinNum', hue='Transported', palette='Set1', binwidth=20)
plt.show();


# In[23]:


plt.figure(figsize=(20,8))
sns.countplot(data=data, x='CabinSide', hue='Transported', palette='Set1')
plt.show();


# ## Name
# 
# And finally we can extract the lastname from the `Name` variable. From the lastname we can get the family size. 

# In[24]:


data['Name'].fillna('Max Mustermann', inplace=True)
test['Name'].fillna('Max Mustermann', inplace=True)


# In[25]:


data['Name'] = data['Name'].str.split().str[-1]
test['Name'] = test['Name'].str.split().str[-1]


# In[26]:


data['FamilySize']=data['Name'].map(lambda x: pd.concat([data['Name'],test['Name']]).value_counts()[x])
test['FamilySize']=test['Name'].map(lambda x: pd.concat([data['Name'],test['Name']]).value_counts()[x])


# In[27]:


data['FamilySize'].value_counts()


# We will set the lastname for all passengers to default `Mustermann` if the `FamilySize` for this name is bigger than 100. 

# In[28]:


data.loc[data['Name']=='Mustermann','Name']=np.nan
data.loc[data['FamilySize']>100,'FamilySize']=np.nan
test.loc[test['Name']=='Mustermann','Name']=np.nan
test.loc[test['FamilySize']>100,'FamilySize']=np.nan


# In[29]:


plt.figure(figsize=(20,8))
sns.countplot(data=data, x='FamilySize', hue='Transported', palette='Set1')
plt.show();


# **Summary**
# 
# - We already created a few new features in the EDA part (We did this here to have all visualizations together)
# - We need to log transform the numeric features because they are very skewed
# - We can drop the `VIP` variable

# ----

# # Feature Engineering
# 
# ## Age
# 
# First lets create The `AgeGroup` 
# - 0-18 years --> Child
# - 19-27 years --> Teenager
# - gt 28 years --> Adult

# In[30]:


data['AgeGroup'] = np.nan
test['AgeGroup'] = np.nan


# In[31]:


data.loc[data['Age'] <= 18,'AgeGroup'] = 'Child'
data.loc[(data['Age'] > 18) & (data['Age'] <= 27),'AgeGroup'] = 'Teenager'
data.loc[data['Age'] > 27,'AgeGroup'] = 'Adult'

test.loc[test['Age'] <= 18,'AgeGroup'] = 'Child'
test.loc[(test['Age'] > 18) & (test['Age'] <= 27),'AgeGroup'] = 'Teenager'
test.loc[test['Age'] > 27,'AgeGroup'] = 'Adult'


# In[32]:


sns.countplot(data=data, x='AgeGroup', hue='Transported', palette='Set1')
plt.show();


# ## Expenses
# 
# We can calculate the total expenses and create a new feature if a passenger spend some money or not. 

# In[33]:


expenses = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']


# In[34]:


data['TotalExpenses'] = data[expenses].sum(axis=1)
test['TotalExpenses'] = test[expenses].sum(axis=1)


# In[35]:


data['HasExpenses'] = data['TotalExpenses'].apply(lambda x: 1 if x > 0 else 0)
test['HasExpenses'] = test['TotalExpenses'].apply(lambda x: 1 if x > 0 else 0)


# In[36]:


plt.hist(data['TotalExpenses'], bins=50)
plt.show();


# In[37]:


sns.countplot(data=data, x='HasExpenses', hue='Transported', palette='Set1')
plt.show();


# ----

# # CleanUp
# 
# We created a lot of new features from existing variables. Before going on we should clean up our dataset a bit. 

# In[38]:


data = data.drop(columns=['PassengerId', 'Cabin', 'VIP', 'Name', 'Group', 'CabinNum'])
test_sub = test['PassengerId']
test = test.drop(columns=['PassengerId', 'Cabin', 'VIP', 'Name', 'Group', 'CabinNum'])


# # Missing values
# 
# As we saw earlier, we have a lot of missing values in our dataset. The easiest way to deal with missing values is to just use median for continuous features and the mode for categorical features.

# In[39]:


cats = ['HomePlanet', 'CryoSleep', 'Destination', 'CabinDeck', 'CabinSide', 'GroupSize', 'AgeGroup', 'HasExpenses']


# In[40]:


data[cats].mode()


# In[41]:


data['HomePlanet'].fillna('Earth', inplace=True)
data['CryoSleep'].fillna(False, inplace=True)
data['Destination'].fillna('TRAPPIST-1e', inplace=True)
data['CabinDeck'].fillna('F', inplace=True)
data['CabinSide'].fillna('S', inplace=True)
data['GroupSize'].fillna('1', inplace=True)
data['AgeGroup'].fillna('Adult', inplace=True)
data['HasExpenses'].fillna('1', inplace=True)

test['HomePlanet'].fillna('Earth', inplace=True)
test['CryoSleep'].fillna(False, inplace=True)
test['Destination'].fillna('TRAPPIST-1e', inplace=True)
test['CabinDeck'].fillna('F', inplace=True)
test['CabinSide'].fillna('S', inplace=True)
test['GroupSize'].fillna('1', inplace=True)
test['AgeGroup'].fillna('Adult', inplace=True)
test['HasExpenses'].fillna('1', inplace=True)


# In[42]:


for cat in cats:
    data[cat] = data[cat].astype('category')
    test[cat] = test[cat].astype('category')


# In[43]:


data['Age'].fillna(data['Age'].median(), inplace=True)
data['RoomService'].fillna(data['RoomService'].median(), inplace=True)
data['FoodCourt'].fillna(data['FoodCourt'].median(), inplace=True)
data['ShoppingMall'].fillna(data['ShoppingMall'].median(), inplace=True)
data['Spa'].fillna(data['Spa'].median(), inplace=True)
data['VRDeck'].fillna(data['VRDeck'].median(), inplace=True)
data['FamilySize'].fillna(data['FamilySize'].median(), inplace=True)

test['Age'].fillna(data['Age'].median(), inplace=True)
test['RoomService'].fillna(data['RoomService'].median(), inplace=True)
test['FoodCourt'].fillna(data['FoodCourt'].median(), inplace=True)
test['ShoppingMall'].fillna(data['ShoppingMall'].median(), inplace=True)
test['Spa'].fillna(data['Spa'].median(), inplace=True)
test['VRDeck'].fillna(data['VRDeck'].median(), inplace=True)
test['FamilySize'].fillna(data['FamilySize'].median(), inplace=True)


# In[44]:


print(data.isnull().sum())


# In[45]:


print(test.isnull().sum())


# Now we have no missing values in our dataset. Be careful, there are way better techniques to handle missing values. 

# ----

# # Preprocessing

# First apply the log transform on our numerical featuers. 

# In[46]:


for col in ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck', 'TotalExpenses']:
    data[col]=np.log(1+data[col])
    test[col]=np.log(1+test[col])


# Split target and features. 

# In[47]:


X = data.drop(columns=['Transported'])
y = data['Transported']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=1702)


# ----

# # Modeling

# In[48]:


# Create lightGBM Classifier and train model
clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)


# In[49]:


# Run prediction
y_pred = clf.predict(X_test)


# In[50]:


accuracy_score(y_test, y_pred)


# In[51]:


print(classification_report(y_test, y_pred))


# In[52]:


cm = confusion_matrix(y_test, y_pred)


# In[53]:


cm


# ----

# # Submission

# In[54]:


sub_pred = clf.predict(test)


# In[55]:


sub_df = pd.DataFrame({'PassengerId': test_sub, 'Transported': sub_pred})


# In[56]:


sub_df.to_csv('/kaggle/working/submission.csv', index=False)


# In[ ]:




