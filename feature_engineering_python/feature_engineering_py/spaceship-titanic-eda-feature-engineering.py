#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
1) PassengerId is a unique identifier for each passenger. Each identifier has the form gggg_pp,
where gggg indicates the group with which the passenger is traveling, and pp is his number in the group.
People in a group are often family members, but not always.

2) HomePlanet -the planet from which the passenger took off, usually the planet of his permanent residence
3) CryoSleep - indicates whether the passenger has decided to go into suspended animation for the duration of the trip.
Passengers in cryosleep are chained to their cabins.
4) Cabin - the number of the cabin in which the passenger is located.
Takes the form deck/num/side, where side can be either P for port or S for starboard.
5) Destination - the planet on which the passenger will disembark
6) Age - the age of the passenger.
7) VIP - Whether the passenger has paid for a special VIP service during the flight.
8) RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - the amount that the passenger paid for using
all the numerous luxury amenities
9) Name - the passenger's first and last name
10) Transported - whether the passenger was transported to another dimension. This is the goal, the column that we are trying to predict.
'''


# # Importing libraries

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from sklearn.metrics import accuracy_score ,classification_report ,confusion_matrix
from sklearn import tree
# %matplotlib notebook


# In[3]:


df=pd.read_csv(r'/kaggle/input/spaceship-titanic/train.csv')
df.head()


# In[4]:


# Look at the basic information on the dataset
df.info()
# We see that some columns contain data in the format required for interpretation
# In the Transported column, it is best to replace the values with Yes/No by changing the data type to for ease of interpretation
# In the CryoSleep,VIP columns, we can also replace the False/True values with Yes/No but it's optional


# In[5]:


# Replacing Boolean values with string values
df['Transported'].replace(False,'No',inplace=True)
df['Transported'].replace(True,'Yes',inplace=True)

df['CryoSleep'].replace(False,'No',inplace=True)
df['CryoSleep'].replace(True,'Yes',inplace=True)

df['VIP'].replace(False,'No',inplace=True)
df['VIP'].replace(True,'Yes',inplace=True)

df.head()


# # Working with missing values

# In[6]:


# Looking at the number of missing values
df.isna().sum()


# In[7]:


#The nature of the omissions is random
sns.heatmap(df.isna(),yticklabels=False,cbar = False,cmap="viridis")
plt.tight_layout()
plt.show()


# ##  HomePlanet

# In[8]:


# Let's analyze the HomePlanet column - this is the planet from which the passenger departs
df.HomePlanet


# In[9]:


fig, ax = plt.subplots(figsize=(7,6))
ax=sns.countplot(data=df,x='HomePlanet',palette='Spectral')

#Adding numbers to the graph
for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, int(height), ha='center', va='bottom')
    
plt.xlabel('Departure planet')
plt.ylabel('Аmount')
plt.title('Amount of people from each planet')


# **It can be seen that a larger number of people from the planet Earth, it would be possible to fill in the empty values with this planet.
# But we will conduct a deeper analysis and find out whether those people are from earth or not.
# We can check the planet of empty values by looking at the columns:CryoSleep , Destination,Age,VIP,Transporter.
# Based on these columns, we will try to determine the planet of departure of people** 

# In[10]:


# To begin with, consider the CryoSleep column
df.groupby(['HomePlanet','CryoSleep']).agg({'CryoSleep':'count'})
# It can be seen from this column that it is difficult to make a conclusion about the planet, since the numbers +- are the same
# Consider this data as a percentage, it should give more information


# In[11]:


# Let's write a function to calculate the percentage ratio
# x = Dataset name
# y =  X 
# z = hue
def percent(x, y, z):
    cot = x[y].value_counts()
    q = x.groupby([y, z])[z].count()
    t = []
    for i in cot.index:
        t.append([i,round(q[i]*100/cot[i], 2)])
    return t

percent(df, 'HomePlanet', 'CryoSleep')
# It can be seen that, as a percentage, people from earth are mostly not in krypton
# Only 30% sleep


# The percent(x, y, z) function takes three arguments:
# 
# 1. x is a DataFrame object that contains the data on which the analysis will be performed.
# 2. y is a string that represents the name of the column containing the categorical data that we want to analyze.
# 3. z is a string that represents the name of the column containing the numeric data that we want to analyze.
# The function first calculates the number of each category in column y using the value_counts() method. Then it groups the data by columns y and z, counts the number of values in column z for each group, and calculates the percentage for each group of the total number of values in column y. The results are saved to the list t

# In[12]:


#Output empty HomePlanet values and check how many of them have No\Yes values in the CryoSleep column

HomePlanet_nan_no=df[(df.HomePlanet.isna())&(df.CryoSleep== 'No')]
HomePlanet_nan_yes=df[(df.HomePlanet.isna())&(df.CryoSleep== 'Yes')]
print('CryoSleep = No:',HomePlanet_nan_no.shape[0],'\n','CryoSleep = Yes:',HomePlanet_nan_yes.shape[0])

#According to the CryoSleep column , we can conclude that 124 people are most likely from earth , and 75 from any of the 2 planets
# But based on the data of this column, without confirmation by other columns, it is impossible to draw a conclusion
# Since the values obtained are extremely ambiguous


# **According to the CryoSleep column, we can conclude that 124 people are most likely from earth, and 75 from any of the 2 planets.
# But based on the data of this column, without confirmation by other columns, it is impossible to draw a conclusion.
# Since the values obtained are extremely ambiguous.**

# In[13]:


fig, ax = plt.subplots(figsize=(7,6))
sns.histplot(data=df,x='Age')
plt.xlabel('Age')
plt.ylabel('Amount of people')
plt.title('Age distribution among passengers')


# In[14]:


fig, ax = plt.subplots(figsize=(7,6))
sns.histplot(data=df,x='Age',palette='Spectral',hue='HomePlanet',multiple='stack')
plt.xlabel('Age')
plt.ylabel('Amount of people')
plt.title('Age distribution among passengers depending on the planet')
##It can be seen that most of the passengers from earth are aged from 15 to 35


# In[15]:


#  Calculate the average age of the planet
df.groupby(['HomePlanet'])['Age'].mean()
# The youngest passengers are from Earth, and the oldest are from Europe
# Knowing this data, you can already try to determine the planet of departure, but you will consider a few more columns


# **Knowing this data, you can already try to determine the planet of departure, but we will consider a few more columns.**

# In[16]:


# Consider the Destination column
ax=sns.countplot(data=df,x='Destination',palette='Spectral')
for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, int(height), ha='center', va='bottom')
    
plt.xlabel('Destination Planet')
plt.ylabel('Amount')
plt.title('Amount of people who went to these planets' )


# In[17]:


ax=sns.countplot(data=df,x='Destination',palette='Spectral',hue='HomePlanet')
for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, int(height), ha='center', va='bottom')
    
plt.xlabel('Destination Planet')
plt.ylabel('Amount')
plt.title('Amount of people who went to these planets from their home planets' )
# As we can see, most of the Earthlings went to TRAPPIST-1e
# # Most likely most of the Martians also went to TRAPPIST-1e
# And the Europeans were distributed between TRAPPIST-1e and  
# # For a more objective consideration, let's look at the percentage ratio of 55 Cancri e


# In[18]:


percent(df,'HomePlanet','Destination')
# As we can see, 67% of Earthlings went to TRAPPIST-1e
#
# # 41.58% of Europeans went to 55 Cancri e and 55.80% to TRAPPIST-1e
# 83.85% of Martians went to TRAPPIST-1e


# According to the Destination column , we can conclude:
# 1. Earthlings with a high probability went to TRAPPIST-1e
# 2. Martians are also very likely to have gone to TRAPPIST-1e
# 3. The Europeans were distributed between TRAPPIST-1e and 55 Cancri e

# In[19]:


# Consider the VIP column by which we can draw a conclusion on the account of passes in the HomePlanet
ax=sns.countplot(data=df,x='VIP',palette='Spectral')

for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, int(height), ha='center', va='bottom')
plt.xlabel('VIP status')
plt.ylabel('Amount')
plt.title('Amount of people with VIP status' )


# In[20]:


ax=sns.countplot(data=df,x='HomePlanet',hue='VIP',palette='Spectral')

for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, int(height), ha='center', va='bottom')
plt.xlabel('VIP status')
plt.ylabel('Amount')
plt.title('Amount of people with VIP status' )
# # These data give us a lot, no Earthling has VIP status.This will help to distribute the data correctly


# In[21]:


percent(df,'HomePlanet','VIP')


# General output before filling in empty values:
# 1. Earthlings for the most part are not in cryosleep, but this column is extremely ambiguous.Therefore, in the output, it is taken in the last turn. 100% of Martians in Krypton do not have VIP status. Only 20 of Europeans are in crypto sleep and have VIP Status
# 2. Earthlings have an average age of 26 years. Europeans are 34 years old. The Martians are 29 years old
# 3. 67% of Earthlings went to TRAPPIST-1e, 41.58% of Europeans went to 55 Cancri e and 55.80% to TRAPPIST-1e,83.85% of Martians went to TRAPPIST-1e
# 4. 100% of Earthlings do not have VIP status. 6% of Europeans have VIP status. 3.5% of Martians have VIP status

# # HomePlanet output

# In[22]:


#To begin with, let's select the records that most likely belong to Earthlings
Earth_index=df[(df.HomePlanet.isna())\
               &(df.VIP == 'No')\
               &(df.Destination == 'TRAPPIST-1e' )\
               &(df.Age <=26)\
               &(df.CryoSleep=='No')].index
# Martian Indexes
Mars_index=df[(df.HomePlanet.isna())\
              &(df.VIP == 'No')\
              &(df.Destination == 'TRAPPIST-1e' )\
              &(df.Age <=29)\
              &(df.CryoSleep=='Yes')].index


# In[23]:


# Fill in the values according to the received indexes
for i in Earth_index:
    df.loc[i, 'HomePlanet'] = 'Earth'
    
for i in Mars_index:
    df.loc[i, 'HomePlanet'] = 'Mars'

df['HomePlanet']=df['HomePlanet'].fillna('Europa')


# # CryoSleep

# In[24]:


# Analyzing the HomePlanet, we received a lot of information on the basis of which we can draw conclusions
ax=sns.countplot(data=df,x='Destination',palette='Spectral',hue='CryoSleep')

for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, int(height), ha='center', va='bottom')
    
plt.xlabel('Destination Planet')
plt.ylabel('Amount')
plt.title('Amount of people in cryosleep depending on the destination planet')
# # According to the schedule, we can conclude that 65% of people who went to TRAPPIST-1e are mostly not in cryosleep


# In[25]:


percent(df,'Destination','CryoSleep')


# In[26]:


# Consider the age of people in Krypton
df.groupby('CryoSleep')['Age'].mean()


# In[27]:


# People with VIP status are most likely not in cryosleep
df.groupby('CryoSleep')['VIP'].value_counts()


# In[28]:


ax=sns.countplot(data=df,x='HomePlanet',palette='Spectral',hue='CryoSleep')

for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, int(height), ha='center', va='bottom')
    
plt.xlabel('Planet of departure')
plt.ylabel('Amount')
plt.title('Amount of people in cryosleep depending on the planet of departure')


# In[29]:


percent(df,'HomePlanet','CryoSleep')
# As mentioned earlier, Earthlings are not in Krypton


# General conclusion before filling in the gaps:
# 1. 65% of people who went to TRAPPIST-1e are mostly not in the crypt
# 2. People with an age of 27 are most likely in a crypto dream, people with an average age of 29 are not in a crypto dream (this column is extremely ambiguous, it should be taken into account last of all)
# 3. 67% of Earthlings are not in Krypton, 42% of Europeans are in Krypton

# # Output by CryoSleep

# In[30]:


# # First we will select records for people who are not in krypton
CryoSleep_no_index=df[(df.CryoSleep.isna())&(df.Destination == 'TRAPPIST-1e')&(df.HomePlanet=='Earth')].index


# In[31]:


#Fill in by indexes
for i in CryoSleep_no_index:
    df.loc[i,'CryoSleep'] = 'No'
df['CryoSleep']=df['CryoSleep'].fillna('Yes')


# # Destination

# In[32]:


ax=sns.countplot(data=df,x='Destination',palette='Spectral')
for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, int(height), ha='center', va='bottom')
    
plt.xlabel('Destination Planet')
plt.ylabel('Amount')
plt.title('Amount of people who went to these planets' )


# We have already received some information about the Destination Planet earlier 
# * 67% of Earthlings went to TRAPPIST-1e 
# * 41.58% of Europeans went to 55 Cancri e and 55.80% to TRAPPIST-1e
# * 83.85% of Martians went to TRAPPIST-1e

# In[33]:


sns.countplot(data=df,x='Destination',palette='Spectral',hue='CryoSleep')


# In[34]:


percent(df,'Destination','CryoSleep')
# 67.3 of those who went to TRAPPIST-1e are in krypton


# In[35]:


ax=sns.countplot(data=df,x='Destination',palette='Spectral',hue='VIP')
for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, int(height), ha='center', va='bottom')
    
plt.xlabel('Destination Planet')
plt.ylabel('Amount')
plt.title('Amount of people who went to these planets with VIP status ' )
#Most of the people who went to TRAPPIST-1e do not have VIP status


# In[36]:


percent(df,'Destination','VIP')


# Preliminary output by Destination:
# 1. 67% of Earthlings went to TRAPPIST-1e, 41.58% of Europeans went to 55 Cancri e and 55.80% to TRAPPIST-1e,83.85% of Martians went to TRAPPIST-1e
# 2. 67.3% of those who went to TRAPPIST-1e are not in krypton
# 3. Most of the people who went to TRAPPIST-1e do not have VIP status

# # Output by Destination

# In[37]:


TRAPPIST_index=df[(df.Destination.isna())&(df.HomePlanet == 'Earth')&(df.CryoSleep == 'No')].index
TRAPPIST_index_Mars=df[(df.Destination.isna())&(df.HomePlanet == 'Mars')].index


# In[38]:


for i in TRAPPIST_index:
    df.loc[i,'Destination']='TRAPPIST-1e'
for i in TRAPPIST_index_Mars:
    df.loc[i,'Destination']='TRAPPIST-1e'
df['Destination']=df['Destination'].fillna('55 Cancri e')


# In[39]:


df.isna().sum()


# # Age

# In[40]:


df.head()


# In[41]:


df.groupby(['HomePlanet']).agg({'Age':'mean'})


# In[42]:


#In this case, I will use the average age on the planet as a basis, since there are quite a variety of data there. 
#That is, there are no very strong distortions in one of the categories 
Europa_34=df[(df.Age.isna())&(df.HomePlanet=='Europa')].index
Earth_df_26=df[(df.Age.isna())&(df.HomePlanet=='Earth')].index
Earth_df_29=df[(df.Age.isna())&(df.HomePlanet=='Mars')].index


# In[43]:


for i in Europa_34:
    df.loc[i,'Age'] = 34.0
for i in Earth_df_26:
    df.loc[i,'Age'] = 26.0
for i in Earth_df_29:
    df.loc[i,'Age'] = 29.0


# In[44]:


df.isna().sum()


# # VIP

# In[45]:


# As we can see from the graphs, only 2% of people have a very large VIP status between classes
# For this reason, filling in the data will be difficult
ax=sns.countplot(data=df,x='VIP',palette='Spectral')
plt.ylabel('Amount')
for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, int(height), ha='center', va='bottom')


# #  Output by VIP

# In[46]:


# Since we have very unbalanced data, we will simply scatter them on 2 planets.
# We selected the rows with nan values and divided them into 2
data=df[(df.VIP.isna())]
len(data)//2
VIP_Europa = data.iloc[:101]
VIP_Mars = data.iloc[101:]

VIP_Europa_index=VIP_Europa.index
VIP_Mars_index=VIP_Mars.index
for i in VIP_Europa_index:
    df.loc[i,'VIP']='Yes'
for i in VIP_Mars_index:
    df.loc[i,'VIP']='Yes'


# # RoomService,FoodCourt,ShoppingMall,Spa,VRDeck

# In[47]:


df.describe()


# In[48]:


df.head()


# In[49]:


# Since a person cannot spend money while in crypto, we are looking for empty indexes and fill in the gaps with zeros
RoomService_index=df[(df.CryoSleep=='Yes')&(df.RoomService.isna())].index
FoodCourt_index=df[(df.FoodCourt.isna())&(df.CryoSleep=='Yes')].index
ShoppingMall_index=df[(df.ShoppingMall.isna())&(df.CryoSleep=='Yes')].index
Spa_index=df[(df.Spa.isna())&(df.CryoSleep=='Yes')].index
VRDeck_index=df[(df.VRDeck.isna())&(df.CryoSleep=='Yes')].index


# In[50]:


for i in range(len(df)):
    if i in RoomService_index:
        df.loc[i,'RoomService'] = 0.0
    elif i in FoodCourt_index:
        df.loc[i,'FoodCourt'] = 0.0
    elif i in ShoppingMall_index:
        df.loc[i,'ShoppingMall'] = 0.0
    elif i in Spa_index:
        df.loc[i,'Spa'] = 0.0
    elif i in VRDeck_index:
        df.loc[i,'VRDeck'] = 0.0

# # Now let's fill in the empty values where people are not in a dream with an average value
for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
    df[col] = df[col].fillna(df[col].mean())


# In[51]:


def remove_outliers(df, columns, k=1.5):
    """
A function to remove outliers from the specified dataframe columns.

    Parameters:
    df (pandas.DataFrame): the original dataframe
    columns (list): list of columns to remove outliers for
    k (float): coefficient for calculating range boundaries
    """
    for column in columns:
        # Calculate the first and third quartiles
        q1 = df[column].quantile(0.25) # We find the 25 smallest values
        q3 = df[column].quantile(0.75) # We find the 25 largest values

        # Calculate the interquartile range (the difference between the largest and smallest values)
        iqr = q3 - q1

        #Replace values outside the range [q1 * qr, q3 + k * ir] with boundary values
        #clip -used to crop values 
        #lower: minimum value for cropping. If the value of the element is less than lower, it is replaced by lower.
        #upper: maximum value for cropping. If the value of the element is greater than upper, it is replaced by upper.
        df[column] = df[column].clip(lower=q1 - k * iqr, upper=q3 + k * iqr)

    return df


# In[52]:


remove_outliers(df,['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], k=1.5)


# # Cabin

# **At the moment, we have filled in the values that can be filled in without any column transformations.
# Now let's work with the Cabin column - cabins in which passengers live. This column contains a lot of useful information
# , in particular the deck and side of the ship.**

# In[53]:


# # Creating new columns by splitting the column
df[['Deck', 'RoomNumber', 'Bot Side']]=df['Cabin'].str.split('/',expand=True)
# # Delete the column to, since it no longer carries useful information
df.drop('Cabin',axis=1,inplace=True)


# **We have a certain number of classes on the ship. 
# There are only a certain number of cabins in each class and the sides of the boat also have a certain number.
# Therefore, after analyzing, we can fill in the empty values with 99% accuracy.
# We cannot achieve 100% accuracy due to the individual design of the ship.**

# # Deck

# In[54]:


## Let's see the decks
clas=pd.Series(df.Deck.unique())
clas.drop(clas[clas.isna()].index,inplace=True)
clas.sort_values(ascending=True,inplace=True)
clas
# # There are 9 decks on our ship


# In[55]:


ax=sns.countplot(data=df,x='Deck',palette='Spectral')
for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, int(height), ha='center', va='bottom')
plt.xlabel('Deck')
plt.ylabel('Number of people living')
plt.title('Distribution of passengers by decks')


# We see that most of the passengers are distributed between decks A and P. This is one of the cheapest tickets 
# You can conditionally divide the classes as follows:
# 1. A,B - first class (the most expensive tickets, these passengers are on the upper decks)
# 2. C,D,E - second class (medium-priced tickets, the decks are in the middle of the ship)
# 3. F,G,T - third class (tickets of the lowest cost, the decks are at the bottom of the ship)
# We will confirm and refute this hypothesis during the analysis of this column

# In[56]:


ax=sns.countplot(data=df,x='Deck',hue='HomePlanet',palette='Spectral')

for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, int(height), ha='center', va='bottom')
plt.xlabel('Deck')
plt.ylabel('Number of people living')
plt.title('Distribution of passengers by decks')


# In[57]:


percent(df,'HomePlanet','Deck')


# Analysis:
# 1. The lands were mainly distributed between the G54% deck and the F-35% deck. These decks belong to the 3rd class and are located in the lower part of the ship. Tickets are inexpensive. If we recall the fact that no Earthlings have VIP status, then we can draw a preliminary conclusion that, in general, Earthlings are not rich. But we will confirm or refute this fact when analyzing the amounts spent on services and entertainment.
# 2. Europeans were distributed between decks B-34%, C-33%, A-11%. These are the best decks on the ship, are at the top of the ship and the article is also expensive. We will make the final conclusion when analyzing the amounts spent.
# 3. The Martians were distributed among classes F-62, E-18%, D-15%. It's practically the middle of the ship.Although most of them are located at the bottom.

# In[58]:


#Разброс значенйи велин, во внимание столбец не принимаем
percent(df,'CryoSleep','Deck')


# In[59]:


ax=sns.countplot(data=df,x='Deck',hue='CryoSleep',palette='Spectral')

for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, int(height), ha='center', va='bottom')
plt.xlabel('Deck')
plt.ylabel('Amount of people staying')


# In[60]:


#The spread of values is large, we do not take into account the column
percent(df,'Destination','Deck')


# In[61]:


#The spread of values is large, we do not take into account the column
df.groupby('Deck')['Age'].mean()


# # Output by Deck

# 1. The lands were mainly distributed between the G54% deck and the F-35% deck. These decks belong to the 3rd class and are located in the lower part of the ship.
# 2. Europeans were distributed between decks B-34%, C-33%, A-11%. These are the best decks on the ship, are located at the top of the ship and are also expensive.
# 3. The Martians were distributed among classes F-62, E-18%, D-15%. It's practically the middle of the ship.Although most of them are located at the bottom.

# In[62]:


Deck_Earth=df[(df.Deck.isna())&(df.HomePlanet=='Earth')].index
Deck_Europa= df[(df.Deck.isna())&(df.HomePlanet=='Europa')].index
Deck_Mars= df[(df.Deck.isna())&(df.HomePlanet=='Mars')].index


# In[63]:


for i in Deck_Earth:
    df.loc[i,'Deck']= 'G'
for i in Deck_Europa:
    df.loc[i,'Deck']= 'C'
for i in Deck_Mars:
    df.loc[i,'Deck']= 'F'


# # Bot Side	

# In[64]:


ax=sns.countplot(data=df,x='Bot Side',hue='HomePlanet',palette='Spectral')

for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, int(height), ha='center', va='bottom')
plt.xlabel('Deck')
plt.ylabel('Number of people staying')


# In[65]:


df['Bot Side'].value_counts()


# In[66]:


# Fill in the empty Europa values for a uniform distribution of P
df['Bot Side']=df['Bot Side'].fillna('P')


# In[67]:


df.head()


# # Creating new columns

# In[68]:


# # Before creating a column with the amount spent, we will delete all the extra columns that are not suitable for analytics
df.drop(['PassengerId','Name','RoomNumber'],axis=1,inplace=True)

#Swap the columns to simplify the perception
df = df.reindex(columns=['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService',
       'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Deck',
       'Bot Side', 'Transported'])


# # TotalAmount

# In[69]:


# Creating a column with the total amount spent
df = df.assign(TotalAmount=df.RoomService + df.FoodCourt + df.ShoppingMall + df.Spa + df.VRDeck)


# # Analytics

# In[70]:


df.head()


# **Let's continue writing the characteristics of people from each of the 3 planets**

# Earthlings:
# * Prevail on the ship
# * Do not have VIP status at all 
# * Distributed between decks: G-54% and F-35%
# * Almost equally distributed between the sides
# * 96% of Earthlings have a family
# * 69% of Earthlings went TRAPPIST-1e
# * Only 42% of Earthlings have reached their destination
# * On average, Earthlings spent the least money on the ship, although they spent the most on the ship
# * 70% of earthlings are not in Krypton
# * The average age of an earthling is 25 years

# Europeans:
# * The second largest on the ship
# * In general, 8% have VIP status, this is the largest percentage on the ship.Speaks of their wealth
# * Distributed between decks: B-34%, C-33%, A-11%. These decks are at the top
# * Almost equally distributed between the sides
# * 88% of Europeans have a family
# * 55% of Europeans went TRAPPIST-1e , 42% went 55 Cancri e 
# * 65% of Europeans have reached their destination
# * On average, Europeans spent the most money on the ship, although they are the 2nd largest on the ship
# * 54% of Europeans are not in Krypton
# * The average age of a European is 34 years

# The Martians:
# * 3 in number on the ship
# * In general, 5% have VIP status.You can't call them poor
# * Distributed between decks: F-62, E-18%,D-15%. These decks are almost in the middle
# * Almost equally distributed between the sides
# * 92% of Martians have a family
# * 86% of Martians went TRAPPIST-1e , 10% went 55 Cancri e 
# * 52% of Martians have reached their destination
# * On average, Martians are second in spending on a ship, but they are not much different from Earthlings
# * 58% of Martians are not in Krypton
# * The average age of a European is 29 years

# In[71]:


sns.boxplot(df)


# In[72]:


#First, let's change the data types to the correct ones
df['Age']=df['Age'].astype('int')


# In[73]:


df.head()

