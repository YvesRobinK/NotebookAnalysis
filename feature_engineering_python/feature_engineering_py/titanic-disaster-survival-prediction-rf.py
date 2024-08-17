#!/usr/bin/env python
# coding: utf-8

# # Importing Packages and Reading the Data

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns

#Data preparation, model building and accuracy checking libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import os #For creating and removing submission files


# In[2]:


#Reading the training data from the CSV file:
train = pd.read_csv("/kaggle/input/titanic/train.csv") 
train.head()


# In[3]:


#Reading the testing data from the CSV file:
test = pd.read_csv("/kaggle/input/titanic/test.csv")
test.head()


# **We pre-emptively extract and then remove the target variable from the training set. This allows us to concatenate the training and testing sets and work with a single dataframe for data cleaning instead of having to separately clean both the training and testing sets. It also allows us to standardise both the training and test data together as we will see later.**

# In[4]:


#Extracting the target variable from the training dataset and then dropping it.
train1 = train.copy()
y_train = train1.Survived
train1.drop(['Survived'], axis = 1, inplace = True)


# In[5]:


#Concatenating the training and testing set:
comb = pd.concat([train1,test], axis = 0, ignore_index= True)
comb.shape


# In[6]:


#Information about the combined dataset:
comb.info()


# # Handling Null Values

# In[7]:


#Checking null values:
comb.isnull().sum()


# In[8]:


#Checking null percentages:
((comb.isnull().sum()/comb.isnull().count())*100).round(2)


# **We see that the 'Cabin' column has a very high percentage of missing values yet we do not drop the column and instead replace the null values with 'N/A' strings so that we can later explore the possibility of the presence of a cabin value as a predictor of survival.**

# In[9]:


#Replacing the null values with 'N/A' strings in the 'Cabin' column:
comb['Cabin'].fillna('N/A', inplace = True)


# **Since there are only 2 missing values in the 'Embarked' column, we replace them with the city where most passengers embarked.**

# In[10]:


#Filling the missing values in the embarked column with the mode of the column:
comb['Embarked'].fillna((comb['Embarked'].mode()[0]), inplace = True)


# **To fill in the missing age values, we use the median age of the passenger's honorific instead of using mean or median of the 'Age' column.**

# In[11]:


#Defining a function to extract the honorific from a name:
def extract_honorific(name):
    record = False
    honorific = ''
    for i, char in enumerate(name):
        if char == ',':
            record = True
        if char == '.':
            record = False
        if record == True:
            honorific += name[i + 2]
    return honorific[:-1]

#Finding the honorifics of all the passengers:
honorifics = [extract_honorific(name) for name in comb.Name]

#Creating a new "Honorific" column:
comb.insert(3, "Honorific", honorifics)
comb.head()


# In[12]:


#Checking the count of each unique honorific:
comb.Honorific.value_counts()


# In[13]:


#Checking the honorific-wise median age:
median_ages = pd.Series(comb.groupby(by = 'Honorific')['Age'].median())
median_ages.sort_values(ascending = False)


# In[14]:


#Grouping the data by honorifics and filling the missing age values:
comb1 = pd.DataFrame(columns = comb.columns)

honorificGroup = comb.groupby(by = 'Honorific')
for _, df_honorific in honorificGroup:
    df_honorific['Age'].fillna(df_honorific['Age'].median(), inplace = True)
    comb1 = pd.concat([comb1, df_honorific], axis = 0)


# In[15]:


#Checking correlation between Pclass and Fare:
plt.figure(figsize = (8, 4))
sns.boxplot(y = comb1.Pclass, x = comb1.Fare, orient = 'h', showfliers = False, palette = 'gist_heat')
plt.ylabel('Passenger Class')
plt.yticks([0,1,2], ['First Class','Second Class', 'Third Class'])
plt.show()


# **As we can see, there is a clear (negative) correlation between the passenger class (Pclass) and the ticket fare (Fare). Therefore, to fill in the missing fare values, we use the median fares of their respective passenger class instead of using mean or median of the 'Fare' column.**

# In[16]:


#Checking the passenger-class-wise median fare:
median_fares = pd.Series(comb1.groupby(by = 'Pclass')['Fare'].median())
plt.figure(figsize = (5,3))
median_fares.plot(kind = 'bar', color = 'teal')
plt.text(x = -0.1, y = median_fares.loc[1] + 0.5, s = "${}".format(median_fares.loc[1].round(2)), fontsize = 12)
plt.text(x = -0.1 + 1, y = median_fares.loc[2] + 0.5, s = "${}".format(median_fares.loc[2].round(2)), fontsize = 12)
plt.text(x = -0.1 + 2, y = median_fares.loc[3] + 0.5, s = "${}".format(median_fares.loc[3].round(2)), fontsize = 12)
plt.xlabel('Passenger Class', fontsize = 12)
plt.ylabel('Fare', fontsize = 12)
plt.title('Passenger Class wise Median Fare', fontsize = 15)
plt.xticks([0,1,2], ['First Class', 'Second Class', 'Third Class'], rotation = 'horizontal', fontsize = 11)
plt.tight_layout(pad = -5)
plt.show()


# In[17]:


#Grouping the data by passenger-class and filling the missing 'Fare' values:
comb2 = pd.DataFrame(columns = comb1.columns)

pclassGroup = comb1.groupby(by = 'Pclass')
for _, df_pclass in pclassGroup:
    df_pclass['Fare'].fillna(df_pclass['Fare'].median(), inplace = True)
    comb2 = pd.concat([comb2, df_pclass], axis = 0)


# In[18]:


#Re-checking for null values:
comb2.isnull().sum()


# **All null values have been eliminated or imputed.**

# # Feature Engineering

# ## PassengerID  
# **PassengerID is not a predictor of survival.**

# ## Name
# **The length of the name may help predict survival, but that would be entirely coincidental. Instead we have extracted the honorifics of each passenger from their name which is a much more meaningful predictor of survival.**

# ## Ticket
# **Ticket contains ticket numbers and in some cases contains some special alphanumeric words which may help predict survival, but that would require a certain amount of domain knowledge of the Titanic ship which we do not possess. Therefore, we will not use this feature.**

# ## Sex

# In[19]:


#Visualising Sex w.r.t Survival:
temp = pd.concat([comb2.sort_index().iloc[:891], y_train], axis = 1)
sexSurvival = temp.groupby(by = 'Sex')['Survived'].value_counts()
plt.figure(figsize = (8, 5))
sns.countplot(data = temp, x = 'Sex',  hue = 'Survived', palette = 'viridis')
plt.text(x = -0.25, y = sexSurvival['male'][0] + 3, s = "{}%".format(((sexSurvival['male'][0]/sexSurvival['male'].sum())*100).round(2)), fontsize = 12)
plt.text(x = 0.13, y = sexSurvival['male'][1] + 3, s = "{}%".format(((sexSurvival['male'][1]/sexSurvival['male'].sum())*100).round(2)), fontsize = 12)
plt.text(x = -0.25 + 1, y = sexSurvival['female'][0] + 3, s = "{}%".format(((sexSurvival['female'][0]/sexSurvival['female'].sum())*100).round(2)), fontsize = 12)
plt.text(x = 0.13 + 1, y = sexSurvival['female'][1] + 3, s = "{}%".format(((sexSurvival['female'][1]/sexSurvival['female'].sum())*100).round(2)), fontsize = 12)
plt.title('Survival Distribution among Men and Women', fontsize = 18)
plt.xticks([0, 1], ['Male', 'Female'], fontsize = 12)
plt.xlabel('Sex', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.tight_layout(pad = -5)
plt.show()


# **It is clear from the above visualisation that more than 80% of men died whereas almost 75% of the women survived. Therefore, sex is a distinct predictor of survival and no feature engineering is required.**

# ## Passenger Class (Pclass)

# In[20]:


#Visualising Pclass w.r.t Survival:
pcSurvival = temp.groupby(by = 'Pclass')['Survived'].value_counts()
plt.figure(figsize = (8, 5))
sns.countplot(data = temp, x = 'Pclass',  hue = 'Survived', palette = 'viridis')
plt.text(x = -0.27, y = pcSurvival[1][0] + 3, s = "{}%".format(((pcSurvival[1][0]/pcSurvival[1].sum())*100).round(2)), fontsize = 12)
plt.text(x = 0.12, y = pcSurvival[1][1] + 3, s = "{}%".format(((pcSurvival[1][1]/pcSurvival[1].sum())*100).round(2)), fontsize = 12)
plt.text(x = -0.27 + 1, y = pcSurvival[2][0] + 3, s = "{}%".format(((pcSurvival[2][0]/pcSurvival[2].sum())*100).round(2)), fontsize = 12)
plt.text(x = 0.12 + 1, y = pcSurvival[2][1] + 3, s = "{}%".format(((pcSurvival[2][1]/pcSurvival[2].sum())*100).round(2)), fontsize = 12)
plt.text(x = -0.27 + 2, y = pcSurvival[3][0] + 3, s = "{}%".format(((pcSurvival[3][0]/pcSurvival[3].sum())*100).round(2)), fontsize = 12)
plt.text(x = 0.12 + 2, y = pcSurvival[3][1] + 3, s = "{}%".format(((pcSurvival[3][1]/pcSurvival[3].sum())*100).round(2)), fontsize = 12)
plt.title('Survival Distribution among different Passenger Classes', fontsize = 18)
plt.xticks([0, 1, 2], ['First-Class', 'Second-Class', 'Third-Class'], fontsize = 12)
plt.xlabel('Passenger Class', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.tight_layout(pad = -5)
plt.show()


# **It is clear from the above visualisation that more than 75% of third-class passengers died whereas more than 60% of first-class passengers survived. The higher the passenger class, the higher is the survival rate and vice-versa. Therefore, passenger class is a distinct predictor of survival and no feature engineering is required.**

# ## City of Embarkment (Embarked)

# In[21]:


#Visualising Embarked w.r.t Survival:
ebSurvival = temp.groupby(by = 'Embarked')['Survived'].value_counts()
plt.figure(figsize = (8, 4))
sns.countplot(data = temp, x = 'Embarked',  hue = 'Survived', palette = 'viridis')
plt.text(x = -0.27, y = ebSurvival['S'][0] + 3, s = "{}%".format(((ebSurvival['S'][0]/ebSurvival['S'].sum())*100).round(2)), fontsize = 12)
plt.text(x = 0.12, y = ebSurvival['S'][1] + 3, s = "{}%".format(((ebSurvival['S'][1]/ebSurvival['S'].sum())*100).round(2)), fontsize = 12)
plt.text(x = -0.27 + 1, y = ebSurvival['C'][0] + 3, s = "{}%".format(((ebSurvival['C'][0]/ebSurvival['C'].sum())*100).round(2)), fontsize = 12)
plt.text(x = 0.12 + 1, y = ebSurvival['C'][1] + 3, s = "{}%".format(((ebSurvival['C'][1]/ebSurvival['C'].sum())*100).round(2)), fontsize = 12)
plt.text(x = -0.27 + 2, y = ebSurvival['Q'][0] + 3, s = "{}%".format(((ebSurvival['Q'][0]/ebSurvival['Q'].sum())*100).round(2)), fontsize = 12)
plt.text(x = 0.12 + 2, y = ebSurvival['Q'][1] + 3, s = "{}%".format(((ebSurvival['Q'][1]/ebSurvival['Q'].sum())*100).round(2)), fontsize = 12)
plt.title('Survival Distribution based on City of Embarkment', fontsize = 18)
plt.xticks([0, 1, 2], ['Southampton', 'Cherbourg', 'Queenstown'], fontsize = 12)
plt.xlabel('City of Embarkment', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.tight_layout(pad = -5)
plt.show()


# **It is clear from the visualisation that almost twice as many passengers from Southampton died as compared to those who survived. Passengers from Queenstown also have a poor survival rate at around 39%. Passengers from Cherbourg have the highest survival rate at 55%.  
# Embarkment also seems like a distinct predictor of survival and no feature engineering is required.**

# ## Cabin

# In[22]:


#Checking the unique cabin counts:
cabin_values = comb2.Cabin.value_counts()
cabin_values


# **There are too many unique cabin values to make separate features out of them. Instead, let's create a feature based on the presence of a cabin value.**

# In[23]:


#Creating a column based on presence of a cabin value:
comb2['IsCabinPresent'] = ['Present' if cabin != 'N/A' else 'Not Present' for cabin in comb2.Cabin]
comb2.sample(5)


# In[24]:


#Visualising presence of cabin w.r.t Survival
temp = pd.concat([comb2.sort_index().iloc[:891], y_train], axis = 1)
cabSurvival = temp.groupby(by = 'IsCabinPresent')['Survived'].value_counts()
plt.figure(figsize = (8, 4))
sns.countplot(data = temp, x = 'IsCabinPresent',  hue = 'Survived', palette = 'viridis')
plt.text(x = -0.27, y = cabSurvival['Not Present'][0] + 3, s = "{}%".format(((cabSurvival['Not Present'][0]/cabSurvival['Not Present'].sum())*100).round(2)), fontsize = 12)
plt.text(x = 0.12, y = cabSurvival['Not Present'][1] + 3, s = "{}%".format(((cabSurvival['Not Present'][1]/cabSurvival['Not Present'].sum())*100).round(2)), fontsize = 12)
plt.text(x = -0.27 + 1, y = cabSurvival['Present'][0] + 3, s = "{}%".format(((cabSurvival['Present'][0]/cabSurvival['Present'].sum())*100).round(2)), fontsize = 12)
plt.text(x = 0.12 + 1, y = cabSurvival['Present'][1] + 3, s = "{}%".format(((cabSurvival['Present'][1]/cabSurvival['Present'].sum())*100).round(2)), fontsize = 12)
plt.title('Survival Distribution based on Presence of Cabin Values', fontsize = 18)
plt.xticks([0, 1], ['No Cabin Value', 'Cabin Value Present'], fontsize = 12)
plt.xlabel('Cabin Value Presence', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.tight_layout(pad = -5)
plt.show()


# **It's clear from the above visualisation that over 70% of the passengers with no cabin value died, whereas only one-third of the passengers with a cabin value present died. Therefore, presence of a cabin value is a distinct predictor of survival and no futher feature engineering is required.**

# ## Fare

# In[25]:


#Visualising the fare distribution w.r.t survival:
plt.figure(figsize = (20, 8))
sns.histplot(x = 'Fare', data = temp, hue = 'Survived', multiple = 'stack', palette = 'viridis')
plt.xlabel('Fare', fontsize = 18)
plt.ylabel('Count', fontsize = 15)
plt.show()


# **Let's try grouping the fares to see if fare groups provide a more distinct correlation with survival.**

# In[26]:


#Calculating the quartiles of 'Fare':
Q1 = comb2.Fare.quantile(0.25)
Q2 = comb2.Fare.quantile(0.50)
Q3 = comb2.Fare.quantile(0.75)

print("\n'Very Low Fare' Range:", 0, "-", Q1.round(2))
print("\n'Low Fare' Range:", Q1.round(2), "-", Q2.round(2))
print("\n'Medium Fare' Range:", Q2.round(2), "-", Q3.round(2))
print("\n'High Fare' Range:", Q3, "-", round(max(comb2.Fare), 2), '\n\n')

#Creating fare groups:
comb2.insert(10, 'FareGroup', np.nan)

comb2.loc[(comb2.Fare <= Q1), 'FareGroup'] = 'VeryLowFare'
comb2.loc[(comb2.Fare > Q1) & (comb2.Fare <= Q2), 'FareGroup'] = 'LowFare'
comb2.loc[(comb2.Fare > Q2) & (comb2.Fare <= Q3), 'FareGroup'] = 'MediumFare'
comb2.loc[(comb2.Fare > Q3), 'FareGroup'] = 'HighFare'

comb2.sample(8)


# In[27]:


#Visualising the fare group distribution w.r.t survival:
temp = pd.concat([comb2.sort_index().iloc[:891], y_train], axis = 1)
fgSurvival = temp.groupby(by = 'FareGroup')['Survived'].value_counts()
plt.figure(figsize = (8, 4))
sns.countplot(data = temp, x = 'FareGroup',  hue = 'Survived', order = ['VeryLowFare', 'LowFare', 'MediumFare', 'HighFare'], palette = 'viridis')
plt.text(x = -0.30, y = fgSurvival['VeryLowFare'][0] + 2, s = "{}%".format(((fgSurvival['VeryLowFare'][0]/fgSurvival['VeryLowFare'].sum())*100).round(2)), fontsize = 12)
plt.text(x = 0.08, y = fgSurvival['VeryLowFare'][1] + 2, s = "{}%".format(((fgSurvival['VeryLowFare'][1]/fgSurvival['VeryLowFare'].sum())*100).round(2)), fontsize = 12)
plt.text(x = -0.30 + 1, y = fgSurvival['LowFare'][0] + 2, s = "{}%".format(((fgSurvival['LowFare'][0]/fgSurvival['LowFare'].sum())*100).round(2)), fontsize = 12)
plt.text(x = 0.08 + 1, y = fgSurvival['LowFare'][1] + 2, s = "{}%".format(((fgSurvival['LowFare'][1]/fgSurvival['LowFare'].sum())*100).round(2)), fontsize = 12)
plt.text(x = -0.30 + 2, y = fgSurvival['MediumFare'][0] + 2, s = "{}%".format(((fgSurvival['MediumFare'][0]/fgSurvival['MediumFare'].sum())*100).round(2)), fontsize = 12)
plt.text(x = 0.08 + 2, y = fgSurvival['MediumFare'][1] + 2, s = "{}%".format(((fgSurvival['MediumFare'][1]/fgSurvival['MediumFare'].sum())*100).round(2)), fontsize = 12)
plt.text(x = -0.30 + 3, y = fgSurvival['HighFare'][0] + 2, s = "{}%".format(((fgSurvival['HighFare'][0]/fgSurvival['HighFare'].sum())*100).round(2)), fontsize = 12)
plt.text(x = 0.08 + 3, y = fgSurvival['HighFare'][1] + 2, s = "{}%".format(((fgSurvival['HighFare'][1]/fgSurvival['HighFare'].sum())*100).round(2)), fontsize = 12)
plt.xlabel('Fare Group', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.xticks([0, 1, 2, 3], ['Very Low Fare', 'Low Fare', 'Medium Fare', 'High Fare'], fontsize = 12)
plt.title('Survival Distribution among different Fare Groups', fontsize = 18)
plt.tight_layout(pad = -5)
plt.show()


# **It's abundantly clear from the above visualisation that the higher the fare group, the higher the survival rate. Therefore, fare groups are a distinct predictor of surival and no further feature engineering is required.**

# ## Number of Parents or Children (Parch) and Number of Siblings or Spouses (SibSp) 

# In[28]:


#Visualising number of parents or children (Parch) and number of siblings or spouses (SibSp) w.r.t Survival:
temp = pd.concat([comb2.sort_index().iloc[:891], y_train], axis = 1)
cols = ['Parch', 'SibSp']
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 5))
for col, subplot in zip(cols, ax.flatten()):
    sns.countplot(data = temp, x = col,  hue = 'Survived', ax = subplot, palette = 'viridis')
    subplot.legend(loc = 'upper right', title = 'Survived')
plt.show()


# **Let's create another feature called FamilyCount which is the number of family members the passenger has aboard the ship (including themselves).**

# In[29]:


#Creating FamilyCount out of Parch and SibSp:
comb2['FamilyCount'] = 1 + comb2['SibSp'] + comb2['Parch']
comb2.head()


# In[30]:


#Visualising number of family members against survival:
temp = pd.concat([comb2.sort_index().iloc[:891], y_train], axis = 1)
plt.figure(figsize = (12, 4))
sns.countplot(data = temp, x = 'FamilyCount', hue = 'Survived', palette = 'viridis')
plt.legend(loc = 'upper right', title = 'Survived')
plt.show()


# **From the above visualisation, we can see that when the passenger is alone, there's a very high chance of dying. When the family count is 2, 3 or 4, the chances of surviving are slightly higher. For large families (FamilyCount > 4), the chances of dying are again higher than the chances of surviving. Therefore, we split the family count accordingly into 3 distinct groups.**

# In[31]:


#Creating another feature FamilySize based on groups of family count:
comb2.insert(8, 'FamilySize', np.nan)

comb2.loc[(comb2.FamilyCount == 1), 'FamilySize'] = 'Alone'
comb2.loc[(comb2.FamilyCount > 1) & (comb2.FamilyCount <= 4), 'FamilySize'] = 'Medium'
comb2.loc[(comb2.FamilyCount > 4), 'FamilySize'] = 'Large'

comb2.sample(5)


# In[32]:


#Visualising family sizes against survival:
temp = pd.concat([comb2.sort_index().iloc[:891], y_train], axis = 1)
fsSurvival = temp.groupby(by = 'FamilySize')['Survived'].value_counts()
plt.figure(figsize = (8, 4))
sns.countplot(data = temp, x = 'FamilySize', hue = 'Survived', order = ['Alone', 'Medium', 'Large'], palette = 'viridis')
plt.text(x = -0.27, y = fsSurvival['Alone'][0] + 3, s = "{}%".format(((fsSurvival['Alone'][0]/fsSurvival['Alone'].sum())*100).round(2)), fontsize = 12)
plt.text(x = 0.11, y = fsSurvival['Alone'][1] + 3, s = "{}%".format(((fsSurvival['Alone'][1]/fsSurvival['Alone'].sum())*100).round(2)), fontsize = 12)
plt.text(x = -0.27 + 1, y = fsSurvival['Medium'][0] + 3, s = "{}%".format(((fsSurvival['Medium'][0]/fsSurvival['Medium'].sum())*100).round(2)), fontsize = 12)
plt.text(x = 0.11 + 1, y = fsSurvival['Medium'][1] + 3, s = "{}%".format(((fsSurvival['Medium'][1]/fsSurvival['Medium'].sum())*100).round(2)), fontsize = 12)
plt.text(x = -0.27 + 2, y = fsSurvival['Large'][0] + 3, s = "{}%".format(((fsSurvival['Large'][0]/fsSurvival['Large'].sum())*100).round(2)), fontsize = 12)
plt.text(x = 0.11 + 2, y = fsSurvival['Large'][1] + 3, s = "{}%".format(((fsSurvival['Large'][1]/fsSurvival['Large'].sum())*100).round(2)), fontsize = 12)
plt.xlabel('Family Size', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.xticks([0, 1, 2], ['No Family (Alone)', 'Medium Sized Family', 'Large Family'], fontsize = 12)
plt.title('Survival Distribution among different Family Sizes', fontsize = 18)
plt.tight_layout(pad = -5)
plt.show()


# **The relationship between family sizes and survival becomes much more distinct using family size groups (as opposed to using Parch and SibSp). A passenger from a large family has more than 80% chance of dying and passenger with no family aboard has almost 70% chance of dying. Medium sized families have the highest survival rate at almost 58%.  
# Therefore, family sizes (independently) are distinct predictors of survival and no further feature engineering is required.**

# ## Age

# In[33]:


#Visualising the age distribution w.r.t survival:
temp = pd.concat([comb2.sort_index().iloc[:891], y_train], axis = 1)
plt.figure(figsize = (15,8))
sns.swarmplot(data = temp, x = 'Survived', y = 'Age', palette = 'viridis')
plt.show()


# In[34]:


#Visualising ages w.r.t survival:
plt.figure(figsize = (15, 8))
sns.histplot(data = temp, x = 'Age', hue = 'Survived', multiple = 'stack', bins = 80, palette = 'viridis')
plt.show()


# **From the above visualisation, the survival rates seem too irregular at different ages and so it's hard to make meaningful age groups out of them.**

# In[35]:


#Dropping columns that are no longer required:
comb2.drop(['PassengerId', 'Name', 'Ticket', 'Parch', 'SibSp', 'FamilyCount', 'Fare', 'Cabin'], axis  = 1, inplace = True)
comb2.head()


# In[36]:


#Checking the datatypes of the features in the dataframe:
comb2.dtypes


# In[37]:


#Dummy Encoding the categorical variables:
categoricals = comb2.select_dtypes(exclude = ['int64', 'float64'])
categorical_dummies = pd.get_dummies(categoricals, drop_first = False)
categorical_dummies.head()


# In[38]:


#Fetching the numerical columns:
numericals = comb2.drop(categoricals, axis = 1)
numericals.head()


# ## Honorific

# **Creating honorific groups by combining honorifics with similar properties.**

# In[39]:


#Combining unmarried women into a single feature:
categorical_dummies['YoungWomen'] = categorical_dummies['Honorific_Miss.'] + categorical_dummies['Honorific_Mlle.']

#Combining married women into a single feature:
categorical_dummies['MarriedWomen'] = categorical_dummies['Honorific_Mrs.'] + categorical_dummies['Honorific_Mme.'] + categorical_dummies['Honorific_Ms.']

#Combining the rarer honorifics into a single feature (grouping them into further subsets would just create more noise for the model):
categorical_dummies['RareHonorific'] = categorical_dummies['Honorific_Capt.'] + categorical_dummies['Honorific_Col.'] + categorical_dummies['Honorific_Don.'] + categorical_dummies['Honorific_Dona.'] + categorical_dummies['Honorific_Dr.'] + categorical_dummies['Honorific_Jonkheer.']  + categorical_dummies['Honorific_Lady.'] + categorical_dummies['Honorific_Major.'] + categorical_dummies['Honorific_Sir.'] + categorical_dummies['Honorific_the Countess.'] + categorical_dummies['Honorific_Rev.']


# In[40]:


#Dropping all the features that have since been combined into a new feature:
categorical_dummies.drop(['Honorific_Miss.', 'Honorific_Mlle.', 'Honorific_Mrs.', 'Honorific_Mme.', 'Honorific_Ms.', 'Honorific_Dona.', 'Honorific_Lady.', 'Honorific_the Countess.', 'Honorific_Rev.', 'Honorific_Jonkheer.', 'Honorific_Capt.', 'Honorific_Col.', 'Honorific_Major.', 'Honorific_Don.', 'Honorific_Sir.', 'Honorific_Dr.'], 
                         axis = 1, inplace = True)

categorical_dummies.head()


# In[41]:


#Re-combining the numerical and categorical variables:
x = pd.concat([numericals, categorical_dummies], axis = 1)
x.head()


# In[42]:


#Sorting the whole data by index:
x = x.sort_index(ascending = True)
x.tail()


# # Checking Correlation

# In[43]:


#Correlation check:
temp = pd.concat([x.iloc[:891], y_train], axis = 1)
corr = temp.corr()
plt.figure(figsize = (24,18))
sns.heatmap(corr, cbar = True, annot = True, linewidths = 0.5)
plt.show()


# In[44]:


#Removing Sex_male and IsCabinPresent_Not Present features as they're dummy encoded from features that contained only binary classes:
x.drop(['Sex_male', 'IsCabinPresent_Not Present'], axis = 1, inplace = True)


# **There's high levels of multicollinearity present in this data yet we don't need to remove it as we'll be using a Random Forest (Random Forest models select features based on information gain and also implement 'bagging' due to which they're unaffected by multicollinearity).**

# # Scaling the Features

# In[45]:


#Scaling the independent features:
scaler = StandardScaler()
scaler.fit(x)
X = scaler.fit_transform(x)


# #  Splitting the Data back into Training and Testing Sets

# In[46]:


#Splitting the transformed data back into training and testing sets:
X_train = X[:891]
X_test = X[891:]


# # Random Forest

# In[47]:


#Creating search parameters for the GridSearchCV
search_parameters = [{'n_estimators': [1000],
                     'criterion': ['gini', 'entropy'],
                     'max_depth': [10, 11, 12],
                     'max_leaf_nodes': [18, 19, 20],
                     'min_samples_leaf': [1],
                     'min_samples_split': [2]}]


# In[48]:


#Creating a random forest instance and using GridSearchCV to find the optimal parameters:
rf_cls_CV = RandomForestClassifier(oob_score = True, random_state = 10)

grid = GridSearchCV(estimator = rf_cls_CV, param_grid = search_parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)

rf_grid = grid.fit(X_train, y_train)

print('Best parameters for random forest classifier: ', rf_grid.best_params_, '\n')


# In[49]:


#Creating a random forest model based on the optimal paramters given by GridSearchCV:
rf_grid_model = RandomForestClassifier(n_estimators = rf_grid.best_params_.get('n_estimators'),
                                       criterion = rf_grid.best_params_.get('criterion'),
                                       max_depth = rf_grid.best_params_.get('max_depth'),
                                       max_leaf_nodes = rf_grid.best_params_.get('max_leaf_nodes'),
                                       min_samples_leaf = rf_grid.best_params_.get('min_samples_leaf'),
                                       min_samples_split = rf_grid.best_params_.get('min_samples_split'),
                                       oob_score = True,
                                       random_state = 10, 
                                       n_jobs = -1)

rf_grid_model = rf_grid_model.fit(X_train, y_train)


# In[50]:


#Making predictions using the optimal random forest:
y_pred_RFGSCV = rf_grid_model.predict(X_test)


# # Compiling and Submitting the Results

# In[51]:


#Final result compilation:
def final_result(model_prediction):
    if os.path.exists("/kaggle/working/submission.csv"):
        os.remove("/kaggle/working/submission.csv")
    Passengers = test.PassengerId
    Survived = pd.Series(model_prediction)
    final = pd.concat([Passengers, Survived], axis = 1)
    final.rename(columns = {0:'Survived'}, inplace = True)
    final.to_csv("submission.csv", index = False)


# In[52]:


final_result(y_pred_RFGSCV)

