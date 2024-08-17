#!/usr/bin/env python
# coding: utf-8

# What we will explore:
# 
# 1. [Introduction](#1)
# 2. [Exploratory Data Analysis and Feature Engineering](#2)
# 3. [Filling missing values by finding patterns of it with other features](#3)
# 4. [Preprocessing](#4)
# 5. [Modeling](#5)
# 6. [Reference](#6)

# <a id="1"></a>
# # **<center><span style="color:#00BFC4;">Introduction  </span></center>**

# Through this notebook, we'll embark on a mission to decipher the clues hidden within the ship's damaged computer records. With the fate of thousands hanging in the balance, Our analytical skills will play a pivotal role in identifying the passengers who were ensnared by the anomaly. As we explore the data and develop predictive models.
# 
# Guided by a comprehensive array of tools and methodologies, we'll delve into data preprocessing, feature engineering, model selection, and more. Whether you're a seasoned data scientist or an enthusiastic beginner, this notebook offers a introductory guide to all the analysis with detailed explanation.
# 
# Join us as we embark on a thrilling voyage of exploration, problem-solving, and discovery. The "Spaceship Titanic" competition competition awaits, ready to immerse you in a world of cosmic intrigue and data-driven innovation. Let's work together to unlock the mysteries of the ship.

# # **<center><span style="color:#00BFC4;">Import the Data  </span></center>**

# In[1]:


import warnings

warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score


# In[2]:


df_train = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')


# # **<center><span style="color:#00BFC4;">Visualise the dataset  </span></center>**

# In[3]:


df_train.head()


# In[4]:


df_train.info()


# In[5]:


df_train.describe()


# There are 12 features we need to use in order to predict whether a passenger is rescued or not. We have 6 categorical features leaving out the passenger_id and 6 numerical features.

# <a id="2"></a>
# # **<center><span style="color:#00BFC4;"> Exploratory Data Analysis <br> and <br> Feature Engineering  </span></center>**

# In[6]:


plot_df = df_train.Transported.value_counts()
plot_df.plot(kind="bar")


# It's evenly balanced so we don't need to worry about sampling techniques.

# ## Missing Values

# Check amounts of missing values and percentage of Missing Values every columns have.

# In[7]:


na_cols=df_train.columns[df_train.isna().any()].tolist()

mv=pd.DataFrame(df_train[na_cols].isna().sum(), columns=['Number_missing'])
mv['Percentage_missing']=np.round(100*mv['Number_missing']/len(df_train),2)
mv


# In[8]:


before_drop = df_train.shape[0]


# Since numbers of missing values is less than 2% let's try dropping the missing values and check the size of the dataset after dropping it.

# In[9]:


df_dropped = df_train.dropna()


# In[10]:


df_dropped.isnull().sum()


# In[11]:


after_drop = df_dropped.shape[0]


# In[12]:


print("Numbers of rows we lost after dropping the missing values: {}".format(before_drop-after_drop))


# We lost around 2087 rows, so dropping missing values might not be the solution.

# ## Numerical Columns Analysis

# In[13]:


numerical_columns = df_train.select_dtypes(include=['number']).columns.tolist()


# In[14]:


fig, ax = plt.subplots(len(numerical_columns),1,  figsize=(10, 10))
plt.subplots_adjust(top = 2)

for index,column in enumerate(numerical_columns):
    sns.histplot(df_train[column], color='b', bins=50, ax=ax[index]);


# **Insight:**
# 
# *Age:*
# 
# * The mean age of the individuals is approximately 28.83 years.
# * The age distribution ranges from 0 to 79 years.
# * The age distribution appears to be positively skewed, as the mean is higher than the median (50th percentile).
# 
# *Other Features:*
# 
# * "RoomService," "FoodCourt," "ShoppingMall," "Spa," and "VRDeck" represent the amount each passenger has billed at each of the luxury amenities on the Spaceship Titanic. 
# 
# * The majority of passengers seem to have spent little or nothing on these amenities, as evidenced by the large counts at lower billing amounts (likely 0).
# 
# * The histogram suggests only a small subset of passengers made significant expenditures at these amenities. These passengers might have been more interested in utilizing these luxury services.
# 
# * The distribution could provide insights into the popularity of each amenity. For example, the lower spending might indicate that not all passengers were interested in or had the opportunity to use amenities like the spa or VR deck.
# 
# * The presence of higher billing amounts could indicate potential outliers—passengers who spent significantly more on these amenities than the majority. These outliers might represent a specific group of passengers or individuals with unique preferences.

# In[15]:


sns.pairplot(df_train, vars=['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], hue = 'Transported')
plt.show()


# **Insights:**
# 
# *High Billing in Food Court and Transportation:*
# 
# * Observation: Passengers with high billings in the Food Court but lower or zero billings in other amenities are more likely to be transported.
# * Insight: This suggests that passengers who heavily utilized the Food Court services without equally engaging with other amenities were more prone to being transported. There might be a connection between specific spending patterns and the anomalies causing transportation.
# 
# *High Billing in Shopping Mall and Transportation:*
# 
# * Observation: Passengers with high billings in the Shopping Mall and lower billings in other amenities are more likely to be transported.
# * Insight: A similar pattern emerges for the Shopping Mall, indicating that passengers who predominantly used this facility without extensively using other amenities faced a higher likelihood of transportation.
# 
# *High Billing in Room Service and Survival:*
# 
# * Observation: Passengers with high billings in Room Service but lower or zero billings in other amenities are less likely to survive.
# * Insight: This observation suggests that passengers who focused on Room Service without engaging in other luxury amenities were less likely to survive the spacetime anomaly event. Their spending behavior could indicate certain vulnerabilities.
# 
# *High Billing in VRDeck and Room Service:*
# 
# * Observation: Passengers with high billings in both VRDeck and Room Service have lower survival rates.
# * Insight: Combining spending on both VRDeck and Room Service appears to correlate with lower survival rates. This could indicate a potential risk associated with overindulgence or distraction by these particular amenities during the event.
# 
# *Zero Billing in Spa and Room Service:*
# 
# * Observation: Passengers with zero billings in Spa and high billings in Room Service (or vice versa) have lower survival rates.
# * Insight: Passengers who had a significant imbalance in spending between Spa and Room Service seemed to have lower survival rates. This could suggest a connection between these two amenities and certain survival factors.

# Check how many passenger has been transported who have bought the luxury Amenities.

# In[16]:


# Set the threshold value for the luxury amenities
thresholds = {
    'RoomService': 0,
    'FoodCourt': 0,
    'ShoppingMall': 0,
    'Spa': 0,
    'VRDeck': 0
}

# Create subplots for each luxury amenity
num_rows = len(thresholds)
fig, axes = plt.subplots(nrows=num_rows, ncols=1, figsize=(8, 5 * num_rows))
fig.suptitle('Transportation by Luxury Amenities Exceeding Thresholds', fontsize=16)

for idx, (amenity, threshold) in enumerate(thresholds.items()):
    # Create a subset of the data where the luxury amenity purchases exceed the threshold
    data_subset = df_train[df_train[amenity] > threshold]
    
    # Group the data and create a bar plot
    grouped = data_subset.groupby('Transported')[amenity].count()
    
    # Plot in the respective subplot
    bars = axes[idx].bar(grouped.index, grouped.values)
    axes[idx].set_title(f'{amenity} > {threshold}')
    axes[idx].set_xlabel('Transported')
    axes[idx].set_ylabel('Count')
    
    # Set tick positions and labels for x-axis
    axes[idx].set_xticks(grouped.index)
    axes[idx].set_xticklabels(['Not Transported', 'Transported'])
    
    # Annotate bars with value counts
    for bar in bars:
        yval = bar.get_height()
        axes[idx].text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# Check how many passenger has been transported who have not bought the luxury Amenities.

# In[17]:


# Create subplots for each luxury amenity
num_rows = len(thresholds)
fig, axes = plt.subplots(nrows=num_rows, ncols=1, figsize=(8, 5 * num_rows))
fig.suptitle('Transportation by Luxury Amenities Exceeding Thresholds', fontsize=16)

for idx, (amenity, threshold) in enumerate(thresholds.items()):
    # Create a subset of the data where the luxury amenity purchases exceed the threshold
    data_subset = df_train[df_train[amenity] == threshold]
    
    # Group the data and create a bar plot
    grouped = data_subset.groupby('Transported')[amenity].count()
    
    # Plot in the respective subplot
    bars = axes[idx].bar(grouped.index, grouped.values)
    axes[idx].set_title(f'{amenity} = {threshold}')
    axes[idx].set_xlabel('Transported')
    axes[idx].set_ylabel('Count')
    
    # Set tick positions and labels for x-axis
    axes[idx].set_xticks(grouped.index)
    axes[idx].set_xticklabels(['Not Transported', 'Transported'])
    
    # Annotate bars with value counts
    for bar in bars:
        yval = bar.get_height()
        axes[idx].text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# *Insights:*
# 
# *Passenger Behavior and Dimensional Transport:*
# 
# * Insight: Passengers who purchased luxury amenities on the spaceship are less likely to be transported (1900-2200), and those who were transported tend to have lower counts (700-900).
# 
# * Implication: This could suggest a potential relationship between luxury amenities and dimensional transport. Passengers who indulged in amenities might have been less affected by the spacetime anomaly or chosen a different fate. The smaller count of transported passengers with luxury amenities purchases might indicate specific attributes that made them more resilient.
# 
# *Transportation Without Luxury Amenity Purchases:*
# 
# * Insight: Passengers who didn't buy luxury amenities but were transported have a significant count (3200-3500).
# 
# * Implication: This observation raises intriguing questions. What factors beyond luxury amenities influence dimensional transport? Could certain passenger groups or demographics have characteristics that make them more prone to transportation? Further investigation is needed to uncover these factors.
# 
# *Non-Transported Passengers:*
# 
# * Insight: Passengers who neither bought luxury amenities nor were transported have a notable count (1900-2200).
# 
# * Implication: This group could hold important clues. Their decision not to indulge in luxury amenities and their lack of transportation might indicate a pattern or a specific characteristic that protected them from the anomaly. Understanding what distinguishes this group could be key to preventing further incidents.

# ### Feature Engineering of Luxury Amenities

# I will add 3 new features. 
# 
# 1. Total_Spending: Total amount the passenger spend on Amenities.
# 2. No_Spending: If the passenger used either None of the Amenities.
# 3. UsedAmenities: Number of Amenities they used.
# 4. Service_Spending: Amount of money spend on this service.
# 5. Shopping_Spending: Amount of money spent on shopping.

# In[18]:


luxury_amenities = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
shopping_features = ['FoodCourt', 'ShoppingMall']
service_features = ['RoomService', 'Spa', 'VRDeck']

df_train['Total_Spending']=df_train[luxury_amenities].sum(axis=1)
df_train['No_spending']=(df_train['Total_Spending']==0).astype(int)
df_train['UsedAmenities'] = df_train[luxury_amenities].gt(0).sum(axis=1)
df_train['Service_Spending'] = df_train[service_features].sum(axis=1)
df_train['Shopping_Spending'] = df_train[shopping_features].sum(axis=1)

df_test['Total_Spending']=df_test[luxury_amenities].sum(axis=1)
df_test['No_spending']=(df_test['Total_Spending']==0).astype(int)
df_test['UsedAmenities'] = df_test[luxury_amenities].gt(0).sum(axis=1)
df_test['Service_Spending'] = df_test[service_features].sum(axis=1)
df_test['Shopping_Spending'] = df_test[shopping_features].sum(axis=1)

df_train.drop(luxury_amenities, axis = 1, inplace=True)
df_test.drop(luxury_amenities, axis = 1, inplace=True)


# In[19]:


df_train.head()


# ### Feature Engineering of Age Columns

# Check Distribution of Age with the hue of Transported.

# In[20]:


# Figure size
plt.figure(figsize=(10,4))

# Histogram
sns.histplot(data=df_train, x='Age', hue='Transported', binwidth=1, kde=True)

# Aesthetics
plt.title('Age distribution')
plt.xlabel('Age (years)')


# Insights:
# 
# 0-18 year olds were more likely to be transported than not.
# 18-25 year olds were less likely to be transported than not.
# Over 25 year olds were about equally likely to be transported than not.
# 
# Based on this we create a new feature to which describe whether a passanger is a child, adolescent or adult.

# In[21]:


df_train['Age_group']=np.nan
df_train.loc[df_train['Age']<=12,'Age_group']='Age_0-12'
df_train.loc[(df_train['Age']>12) & (df_train['Age']<18),'Age_group']='Age_13-17'
df_train.loc[(df_train['Age']>=18) & (df_train['Age']<=25),'Age_group']='Age_18-25'
df_train.loc[(df_train['Age']>25) & (df_train['Age']<=30),'Age_group']='Age_26-30'
df_train.loc[(df_train['Age']>30) & (df_train['Age']<=50),'Age_group']='Age_31-50'
df_train.loc[df_train['Age']>50,'Age_group']='Age_51+'

df_test['Age_group']=np.nan
df_test.loc[df_test['Age']<=12,'Age_group']='Age_0-12'
df_test.loc[(df_test['Age']>12) & (df_test['Age']<18),'Age_group']='Age_13-17'
df_test.loc[(df_test['Age']>=18) & (df_test['Age']<=25),'Age_group']='Age_18-25'
df_test.loc[(df_test['Age']>25) & (df_test['Age']<=30),'Age_group']='Age_26-30'
df_test.loc[(df_test['Age']>30) & (df_test['Age']<=50),'Age_group']='Age_31-50'
df_test.loc[df_test['Age']>50,'Age_group']='Age_51+'


# In[22]:


df_train.head()


# Visualise the Count plot of both of the features.

# In[23]:


plt.figure(figsize=(10,4))
g=sns.countplot(data=df_train, x='Age_group', hue='Transported')
plt.title('Age group distribution')


# ## Categorical Column Analysis

# In[24]:


categorical_columns = df_train.select_dtypes(include=['object']).columns.tolist()


# In[25]:


unique_values_dict = {}
for column in categorical_columns:
    unique_values_dict[column] = len(df_train[column].unique())

# Print the unique values for each categorical column
for column, values in unique_values_dict.items():
    print(f"Unique values in '{column}': {values}")


# Visualise the count plot of the features with less Unique Values.

# In[26]:


cat_feats=['HomePlanet', 'CryoSleep', 'Destination', 'VIP']

# Plot categorical features
fig=plt.figure(figsize=(10,16))
for i, var_name in enumerate(cat_feats):
    ax=fig.add_subplot(4,1,i+1)
    sns.countplot(data=df_train, x=var_name, axes=ax, hue='Transported')
    ax.set_title(var_name)
fig.tight_layout()  
plt.show()


# **Inisghts**
# 
# *Home Planet and Transportation:*
# 
# * Passengers with the home planet Earth being slightly less likely to be transported.
# * Passengers from other planets, like Europa, being slightly more likely to be transported.
# 
# *Cryosleep and Transportation:*
# 
# * Passengers in cryosleep are more likely to be transported.
# 
# *Destination and Transportation:*
# 
# * Passengers destined for 55 Cancri E being more likely to be transported.
# * The slight decrease in likelihood of being transported for passengers with a destination of Trappist.

# Check the remaining Categorical Features

# In[27]:


cols = ['PassengerId', 'Cabin' ,'Name']

df_train[cols].head()


# In[28]:


len(df_train['Name'].unique())


# In[29]:


surnames = df_train['Name'].str.split().str[-1]


# In[30]:


len(surnames.unique())


# * The first four digit of passenger id describe the group and last 2 digit after the undersciore describes the number of the passenger within the group.
# 
# * The feature cabin describeds deck/num/side, where side can be either P for Port or S for Starboard.
# 
# Feature Engineering Goal:
# 
# * We can extract the group and group size from the Passenger feature.
# 
# * we can create a feature name Solo describing if a person don't belong to any group.
# 
# * We can create three seperate features of cabin for Deck, Num and Side. 
# 
# * We can create the new feature name where we extract the Passenger Surname to add information of which Family they belongs to since there are 2218 unique values of surnames among 8474 names.
# 
# * We can create the feature name Family Size describing whats the size of the family the passenger belongs to.

# ### Feature Engineering of Categorical Variables

# **Make feature of Group, Group Size and Solo.**

# In[31]:


# Extract Group and Group Size from PassengerId
df_train['Group'] = df_train['PassengerId'].apply(lambda x: x[:4]).astype(int)
group_size = df_train['Group'].value_counts().to_dict()
df_train['Group_Size'] = df_train['Group'].map(group_size)

df_test['Group'] = df_test['PassengerId'].apply(lambda x: x[:4]).astype(int)
group_size = df_test['Group'].value_counts().to_dict()
df_test['Group_Size'] = df_test['Group'].map(group_size)


# Create Solo feature
df_train['Solo'] = df_train['Group_Size'].apply(lambda x: x == 1)
df_test['Solo'] = df_test['Group_Size'].apply(lambda x: x == 1)


# In[32]:


len(df_train['Group'].unique())


# We can't really use the Group feature in our models because it has too big of a cardinality (6217) and would explode the number of dimensions with one-hot encoding.

# Visualise the survival rate of Group Size and Solo.

# In[33]:


plt.figure(figsize=(16,5))

plt.subplot(1,2,1)
sns.countplot(data=df_train, x='Group_Size', hue='Transported')
plt.title('Passenger travelling solo or not')

plt.subplot(1,2,2)
sns.countplot(data=df_train, x='Solo', hue='Transported')
plt.title('Passenger travelling solo or not')


# Passenger traveling Solo have less chance of survival compares to the passenger traveling in the group. But the passengers belonging to groupsize 8 is less likely to survive. 

# **Extract features from Cabin.**
# 
# We start by filling missing values in the 'Cabin' column of both the training and test datasets with a placeholder value 'Z/9999/Z'. 
# 
# We then create new features based on the 'Cabin' information for both the training and test datasets:
# 
# 'Cabin_deck': This feature captures the deck of the cabin where a passenger stayed. It's extracted by splitting the 'Cabin' value at the '/' character and taking the first part.
# 'Cabin_number': This feature represents the cabin number. It's extracted similarly but converted to an integer.
# 'Cabin_side': This feature indicates whether the cabin was on the port side (P) or starboard side (S). It's extracted similarly by taking the third part of the 'Cabin' value.
# 
# After creating these new features, we put the missing values (previously filled with 'Z/9999/Z') back as actual NaN (missing) values for further processing.

# In[34]:


df_train['Cabin'].fillna('Z/9999/Z', inplace=True)
df_test['Cabin'].fillna('Z/9999/Z', inplace=True)

# New features - training set
df_train['Cabin_deck'] = df_train['Cabin'].apply(lambda x: x.split('/')[0])
df_train['Cabin_number'] = df_train['Cabin'].apply(lambda x: x.split('/')[1]).astype(int)
df_train['Cabin_side'] = df_train['Cabin'].apply(lambda x: x.split('/')[2])

# New features - test set
df_test['Cabin_deck'] = df_test['Cabin'].apply(lambda x: x.split('/')[0])
df_test['Cabin_number'] = df_test['Cabin'].apply(lambda x: x.split('/')[1]).astype(int)
df_test['Cabin_side'] = df_test['Cabin'].apply(lambda x: x.split('/')[2])

# Put Nan's back in (we will fill these later)
df_train.loc[df_train['Cabin_deck']=='Z', 'Cabin_deck']=np.nan
df_train.loc[df_train['Cabin_number']==9999, 'Cabin_number']=np.nan
df_train.loc[df_train['Cabin_side']=='Z', 'Cabin_side']=np.nan
df_test.loc[df_test['Cabin_deck']=='Z', 'Cabin_deck']=np.nan
df_test.loc[df_test['Cabin_number']==9999, 'Cabin_number']=np.nan
df_test.loc[df_test['Cabin_side']=='Z', 'Cabin_side']=np.nan

df_train = df_train.drop(['Cabin'], axis=1)
df_test = df_test.drop(['Cabin'], axis=1)


# **After that we visualise each of the feature we extracted from cabin with Transported.**

# In[35]:


fig=plt.figure(figsize=(10,12))
plt.subplot(3,1,1)
sns.countplot(data=df_train, x='Cabin_deck', hue='Transported')
plt.title('Cabin deck')

plt.subplot(3,1,2)
sns.histplot(data=df_train, x='Cabin_number', hue='Transported',binwidth=20)
plt.vlines(300, ymin=0, ymax=200, color='black')
plt.vlines(600, ymin=0, ymax=200, color='black')
plt.vlines(900, ymin=0, ymax=200, color='black')
plt.vlines(1200, ymin=0, ymax=200, color='black')
plt.vlines(1500, ymin=0, ymax=200, color='black')
plt.vlines(1800, ymin=0, ymax=200, color='black')
plt.title('Cabin number')
plt.xlim([0,2000])

plt.subplot(3,1,3)
sns.countplot(data=df_train, x='Cabin_side', hue='Transported')
plt.title('Cabin side')
fig.tight_layout()


# **Insights:**
# 
# * T seems to be an outlier.
# 
# * Cabin number is been grouped in 300 chunks. At ever 300 divisible cabin number the count is high.
# 
# * Cabin side Starboard is more likely to survive compares to the port.
# 
# * Passenger in Cabin Deck B and C are more likely to survive.

# **Analyse the 'T' cabin Deck.**

# In[36]:


df_train.loc[df_train['Cabin_deck']=='T']


# In[37]:


df_train.loc[df_train['Cabin_deck']=='T']['Transported']


# **Insights:**
# 
# *Age and Grouping:*
# 
# * All passengers in cabin T are adults, which aligns with your observation that they are all 18 years old or older.
# 
# *Solo Travelers:*
# 
# * All passengers in cabin T are traveling alone (solo), as you mentioned.
# 
# *Destination and Home Planet:*
# 
# * Passengers in cabin T have a common destination of TRAPPIST-1e and a common home planet of Europa.
# 
# *Transportation:*
# 
# * Out of the 5 passengers in cabin T, 4 were not transported.
# 
# 

# **Create Cabin Regions**

# We will create new binary features based on the 'Cabin_number' feature. These binary features indicate whether a passenger's cabin number falls within specific numerical ranges. Each binary feature corresponds to a different "cabin region," and a value of 1 indicates that the cabin number falls within the specified range, while a value of 0 indicates that it does not.
# 
# By creating these binary features based on the ranges of 'Cabin_number', we've transformed a continuous numerical feature into a set of categorical binary features that capture specific intervals of cabin numbers. This can help our model capture potential relationships or patterns associated with different ranges of cabin numbers and their impact on survival.

# In[38]:


df_train['Cabin_region1']=(df_train['Cabin_number']<300).astype(int)  
df_train['Cabin_region2']=((df_train['Cabin_number']>=300) & (df_train['Cabin_number']<600)).astype(int)
df_train['Cabin_region3']=((df_train['Cabin_number']>=600) & (df_train['Cabin_number']<900)).astype(int)
df_train['Cabin_region4']=((df_train['Cabin_number']>=900) & (df_train['Cabin_number']<1200)).astype(int)
df_train['Cabin_region5']=((df_train['Cabin_number']>=1200) & (df_train['Cabin_number']<1500)).astype(int)
df_train['Cabin_region6']=((df_train['Cabin_number']>=1500) & (df_train['Cabin_number']<1800)).astype(int)
df_train['Cabin_region7']=(df_train['Cabin_number']>=1800).astype(int)

df_test['Cabin_region1']=(df_test['Cabin_number']<300).astype(int)  
df_test['Cabin_region2']=((df_test['Cabin_number']>=300) & (df_test['Cabin_number']<600)).astype(int)
df_test['Cabin_region3']=((df_test['Cabin_number']>=600) & (df_test['Cabin_number']<900)).astype(int)
df_test['Cabin_region4']=((df_test['Cabin_number']>=900) & (df_test['Cabin_number']<1200)).astype(int)
df_test['Cabin_region5']=((df_test['Cabin_number']>=1200) & (df_test['Cabin_number']<1500)).astype(int)
df_test['Cabin_region6']=((df_test['Cabin_number']>=1500) & (df_test['Cabin_number']<1800)).astype(int)
df_test['Cabin_region7']=(df_test['Cabin_number']>=1800).astype(int)


# Visualise distribution of this new features.

# In[39]:


df_train['Cabin_regions_plot'] = (
    df_train['Cabin_region1'] +
    2 * df_train['Cabin_region2'] +
    3 * df_train['Cabin_region3'] +
    4 * df_train['Cabin_region4'] +
    5 * df_train['Cabin_region5'] +
    6 * df_train['Cabin_region6'] +
    7 * df_train['Cabin_region7']
).astype(int)

# Create the plot
plt.figure(figsize=(10, 4))
sns.countplot(data=df_train, x='Cabin_regions_plot', hue='Transported')

# Set title and labels
plt.title('Count of Passengers in Combined Cabin Regions by Transportation')
plt.xlabel('Combined Cabin Region')
plt.ylabel('Count')

# Add legend
plt.legend(title='Transported', loc='upper right', labels=['Not Transported', 'Transported'])

# Show the plot
plt.tight_layout()
plt.show()

# Drop the temporary column
df_train.drop('Cabin_regions_plot', axis=1, inplace=True)


# **Insight:**
# 
# * Passengers belonging to Cabin regions 1,3,4 are more likely to be transported.

# **Extract Features from Name Column**

# **The Feature Engineering we gonna perform in Name column are:**
# 
# *Surname Extraction:*
# 
# * By extracting the surname from the 'Name' column and creating a new 'Surname' feature, you are aiming to capture potential family relationships among passengers. Surnames are often indicative of family groups, and analyzing them can provide insights into family-based survival patterns or other characteristics that might affect the passengers' outcomes.
# 
# *Family Size Calculation:*
# 
# * The 'Family_size' feature is derived from the calculated count of occurrences of each surname in both the training and test datasets. This feature provides an estimate of the size of the passenger's family group. Larger family groups might have different survival probabilities or other behaviors compared to smaller groups or solo passengers.
# 
# *Handling Missing Values:*
# 
# * By filling missing values in the 'Name' column with 'Unknown Unknown' and handling extreme values in the 'Family_size' column, we ensure that these features are complete and consistent. This enables you to use them effectively in your analysis and modeling without introducing bias due to missing data.
# 
# *Family-based Patterns:*
# 
# * The extracted surname and calculated family size can help uncover patterns related to survival, behavior, or other factors that might be associated with families traveling together. These patterns can contribute valuable insights into the relationships between passengers and how they might have influenced their outcomes.

# In[40]:


df_train['Name'].fillna('Unknown Unknown', inplace=True)
df_test['Name'].fillna('Unknown Unknown', inplace=True)

# New feature - Surname
df_train['Surname']=df_train['Name'].str.split().str[-1]
df_test['Surname']=df_test['Name'].str.split().str[-1]

# New feature - Family size
df_train['Family_size']=df_train['Surname'].map(lambda x: pd.concat([df_train['Surname'],df_test['Surname']]).value_counts()[x])
df_test['Family_size']=df_test['Surname'].map(lambda x: pd.concat([df_train['Surname'],df_test['Surname']]).value_counts()[x])

# Put Nan's back in (we will fill these later)
df_train.loc[df_train['Surname']=='Unknown','Surname']=np.nan
df_train.loc[df_train['Family_size']>100,'Family_size']=np.nan
df_test.loc[df_test['Surname']=='Unknown','Surname']=np.nan
df_test.loc[df_test['Family_size']>100,'Family_size']=np.nan

df_train = df_train.drop(['Name'], axis=1)
df_test = df_test.drop(['Name'], axis=1)


# Visualise the plot of family size.

# In[41]:


plt.figure(figsize=(12,4))
sns.countplot(data=df_train, x='Family_size', hue='Transported')
plt.title('Family size')


# <a id="3"></a>
# # **<center><span style="color:#00BFC4;"> Handling Missing Values  </span></center>**

# Create a data which have combination for both training and testing data frame

# In[42]:


X=df_train.drop('Transported', axis=1).copy()
y=df_train['Transported'].astype(int).copy()

data=pd.concat([X, df_test], axis=0).reset_index(drop=True)


# In[43]:


na_cols=data.columns[data.isna().any()].tolist()

mv=pd.DataFrame(data[na_cols].isna().sum(), columns=['Number_missing'])
mv['Percentage_missing']=np.round(100*mv['Number_missing']/len(data),2)
mv


# **Strategy:**
# 
# We have two approach:
# 
# 1. Simple Imputation: One straightforward method for handling missing values is to replace them with some representative value. For continuous features (numerical), this passage suggests using the median value of the available data, and for categorical features (non-numerical), it suggests using the mode (most frequent value). While this method is easy to implement and can work reasonably well, it might not capture complex relationships in the data and could potentially lead to biased results.
# 
# 2. Pattern-Based Imputation: To maximize the accuracy of predictive models, it's important to explore patterns within the missing data. This involves investigating the relationships between different features in the dataset and using this information to make more informed imputations. In the context of the example provided (passenger data), the passage suggests looking at joint distributions of features. For instance, it proposes examining whether passengers from the same group (a categorical feature) tend to come from the same family (another categorical feature). This can help identify relationships or dependencies that might guide more accurate imputations.
# 
# The main idea behind pattern-based imputation is that by considering relationships between features, you can potentially make more accurate estimates of missing values. For instance, if passengers from the same group are indeed more likely to come from the same family, you could use this information to impute missing family information based on the group. This approach requires careful analysis and understanding of the data, as well as domain knowledge to make meaningful imputations.

# ### Home Planet

# We will impute (fill in) missing values in the "Home Planet" column using information from the different column. The assumption here is that passengers who belong to the same feature are more likely to come from the same home planet.
# 
# The logic behind this approach is that if most of the passengers within a specific feature share the same home planet, it's reasonable to assume that a passenger with a missing home planet in that group might also belong to the same home planet.

# #### Home Planet and Group

# In[44]:


pd.crosstab(data['Group'], data['HomePlanet'])


# From the above crosstab visualisation we can see that people from the same group comes from the same planet. So, we can replace the missing value of the home planet with the record of that group.

# In[45]:


HP_bef=data['HomePlanet'].isna().sum()

for index, row in data.iterrows():
    if pd.isnull(row['HomePlanet']):
        group = row['Group']
        try:
            most_common_home_planet = data[data['Group'] == group]['HomePlanet'].mode().values[0]
        except IndexError:
            continue
        data.at[index, 'HomePlanet'] = most_common_home_planet

print('#HomePlanet missing values before:',HP_bef)
print('#HomePlanet missing values after:',data['HomePlanet'].isna().sum())


# We are able to Impute 137 values.

# #### Home Planet and Cabin Deck

# In[46]:


crosstab = pd.crosstab(data['Cabin_deck'], data['HomePlanet'])


# In[47]:


plt.figure(figsize=(8, 6))
sns.heatmap(crosstab, annot=True, fmt='d')
plt.title('Cross-Tabulation: Group vs HomePlanet')
plt.xlabel('HomePlanet')
plt.ylabel('Group')
plt.show()


# From the above crosstab visualisation we can see that Passengers from Deck A,B,C and T belongs to Home Planet Europa and Passengers from Deck G belongs to Earth. Meanwhile, passengers from Deck D,E and F belongs to multiple planets

# In[48]:


HP_bef=data['HomePlanet'].isna().sum()
print('#HomePlanet missing values before:',HP_bef)

def impute_home_planet(row):
    if pd.isnull(row['HomePlanet']):
        if row['Cabin_deck'] in ['A', 'B', 'C', 'T']:
            return 'Europa'
        elif row['Cabin_deck'] == 'G':
            return 'Earth'
    else:
        return row['HomePlanet']

data['HomePlanet'] = data.apply(impute_home_planet, axis=1)
print('#HomePlanet missing values after:',data['HomePlanet'].isna().sum())


# We are able to impute 63 values

# #### Home Planet and Family

# In[49]:


pd.crosstab(data['Surname'], data['HomePlanet'])


# From this we can see that passengers from the same family belongs to the same planet.

# In[50]:


HP_bef=data['HomePlanet'].isna().sum()

for index, row in data.iterrows():
    if pd.isnull(row['HomePlanet']):
        surname = row['Surname']
        try:
            most_common_home_planet = data[data['Surname'] == surname]['HomePlanet'].mode().values[0]
        except IndexError:
            continue
        data.at[index, 'HomePlanet'] = most_common_home_planet

print('#HomePlanet missing values before:',HP_bef)
print('#HomePlanet missing values after:',data['HomePlanet'].isna().sum())


# #### Remaining values in Home Planet

# We are able to impute 84 values and only 10 values in Home Planet is remaining to impute. Let's check them.

# In[51]:


data[data['HomePlanet'].isnull()]


# From this we can see the only useful pattern we see is that all the remaining missing values in Home Planet have destination as 'TRAPPIST-1e'. Let's check it's relation of destination of with Home Planet

# In[52]:


plt.figure(figsize=(10, 8))
sns.heatmap(pd.crosstab(data['Destination'], data['HomePlanet']), annot=True, fmt='d')
plt.title('Cross-Tabulation: Destination vs HomePlanet')
plt.xlabel('HomePlanet')
plt.ylabel('Destination')
plt.show()


# As we can see that most of the passenger with destination to TRAPPIST-1e belongs to Earth. We can replace the remaining missing values with Earth. But as we show earlier that no from the Cabin Deck as Home Planet as Earth. Let's check Cabin Deck column

# In[53]:


data[data['HomePlanet'].isnull()]['Cabin_deck']


# As we see here, some of the passenger is on Deck D. We can't assing Home Planet as Earth to them. So, we will add Mars as Home Planet to them.

# In[54]:


HP_bef=data['HomePlanet'].isna().sum()
print('#HomePlanet missing values before:',HP_bef)

def impute_home_planet(row):
    if pd.isnull(row['HomePlanet']):
        if row['Cabin_deck'] == 'D':
            return 'Mars'
        else:
            return 'Earth'
    else:
        return row['HomePlanet']

data['HomePlanet'] = data.apply(impute_home_planet, axis=1)
print('#HomePlanet missing values after:',data['HomePlanet'].isna().sum())


# We are done with Imputing values of Home Planet. 
# 

# ### Destination

# We can't find any patterns of Destination with other columns because of large amount of records belonging to TRAPPIST-1e. Whichever visualisation we would do, most of them willhave high records belonging with TRAPPIST-1e. So, we will fill the missing values of Destination with TRAPPIST-1e

# In[55]:


print('#Destination missing values before:',data['Destination'].isna().sum())

data['Destination'].fillna('TRAPPIST-1e', inplace=True)

print('#Destination missing values after:',data['Destination'].isna().sum())


# ### Surname

# Even though we gonna drop Surname Later, we still try to fill some of the missing values so family size feature gets better.

# #### Surname and Group

# In[56]:


crosstab = pd.crosstab(data['Group'], data['Surname'], margins=True)
plt.figure(figsize=(8, 6))
sns.countplot(x=crosstab['All'].index, data=crosstab['All'])
plt.ylabel('Count')
plt.title('Number of Unique Surname by Group')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# From this we can see that most of the family members belongs to the same group. So, using this we can impute values of Surname using Group.

# In[57]:


HP_bef=data['Surname'].isna().sum()

for index, row in data.iterrows():
    if pd.isnull(row['Surname']):
        group = row['Group']
        try:
            most_common = data[data['Group'] == group]['Surname'].mode().values[0]
        except IndexError:
            continue
        data.at[index, 'Surname'] = most_common

print('#Surname missing values before:',HP_bef)
print('#Surname missing values after:',data['Surname'].isna().sum())


# This is the best we can do for the imputation of Surname columns and now we will update the 'Family_size' feature based on the frequency of surnames while handling missing values and outliers in the 'Surname' column. 

# In[58]:


# Replace NaN's with outliers (so we can use map)
data['Surname'].fillna('Unknown', inplace=True)

# Update family size feature
data['Family_size']=data['Surname'].map(lambda x: data['Surname'].value_counts()[x])

# Put NaN's back in place of outliers
data.loc[data['Surname']=='Unknown','Surname']=np.nan

# Say unknown surname means no family
data.loc[data['Family_size']>100,'Family_size']=0


# ### Cabin Features

# Check the unique values of Cabin features compares to Group.

# In[59]:


crosstab_deck = pd.crosstab(data['Group'], data['Cabin_deck'], margins=True)
crosstab_side = pd.crosstab(data['Group'], data['Cabin_side'], margins=True)

plt.figure(figsize=(18, 6))  # Set the overall figure size

# Create subplots
plt.subplot(1, 3, 1)  # 1 row, 3 columns, subplot 1
sns.countplot(x=crosstab_deck['All'].index, data=crosstab_deck['All'])
plt.ylabel('Count')
plt.title('Number of Unique Cabin Deck by Group')
plt.xticks(rotation=45)

plt.subplot(1, 3, 3)  # 1 row, 3 columns, subplot 3
sns.countplot(x=crosstab_side['All'].index, data=crosstab_side['All'])
plt.ylabel('Count')
plt.title('Number of Unique Cabin Side by Group')
plt.xticks(rotation=45)

plt.tight_layout()  # Adjust spacing between subplots
plt.show()


# We can also impute this three features of Cabin using Group

# In[60]:


HP_bef=data['Cabin_deck'].isna().sum()

for index, row in data.iterrows():
    if pd.isnull(row['Cabin_deck']):
        group = row['Group']
        try:
            most_common = data[data['Group'] == group]['Cabin_deck'].mode().values[0]
        except IndexError:
            continue
        data.at[index, 'Cabin_deck'] = most_common

print('#Cabin_deck missing values before:',HP_bef)
print('#Cabin_deck missing values after:',data['Cabin_deck'].isna().sum())

HP_bef=data['Cabin_side'].isna().sum()

for index, row in data.iterrows():
    if pd.isnull(row['Cabin_side']):
        group = row['Group']
        try:
            most_common = data[data['Group'] == group]['Cabin_side'].mode().values[0]
        except IndexError:
            continue
        data.at[index, 'Cabin_side'] = most_common

print('#Cabin_deck missing values before:',HP_bef)
print('#Cabin_deck missing values after:',data['Cabin_side'].isna().sum())


# #### Cabin Deck with Home Planet, Destination and SOlo

# In[61]:


data.groupby(['HomePlanet','Destination','Solo','Cabin_deck'])['Cabin_deck'].size().unstack().fillna(0)


# **Insights**
# 
# * Passenger who are from Home planet Earth, HaveDestination as TRAPPIST-1e are travelling solo are more likely in Deck G.
# 
# * Passenger who are from Home planet Europa, HaveDestination as TRAPPIST-1e are not travelling solo are more likely in Deck B.
# 
# * Passenger who are from Home planet Mars, HaveDestination as TRAPPIST-1e are not travelling solo are more likely in Deck D.

# We will fill values of Deck according to where mode appears on this combination and we will do same for Cabin Side as well

# In[62]:


# Missing values before
CD_bef=data['Cabin_deck'].isna().sum()

# Fill missing values using the mode
na_rows_CD=data.loc[data['Cabin_deck'].isna(),'Cabin_deck'].index
data.loc[data['Cabin_deck'].isna(),'Cabin_deck']=data.groupby(['HomePlanet','Destination','Solo'])['Cabin_deck'].transform(lambda x: x.fillna(pd.Series.mode(x)[0]))[na_rows_CD]

# Print number of missing values left
print('#Cabin_deck missing values before:',CD_bef)
print('#Cabin_deck missing values after:',data['Cabin_deck'].isna().sum())


# In[63]:


# Missing values before
CD_bef=data['Cabin_side'].isna().sum()

# Fill missing values using the mode
na_rows_CD=data.loc[data['Cabin_side'].isna(),'Cabin_side'].index
data.loc[data['Cabin_side'].isna(),'Cabin_side']=data.groupby(['HomePlanet','Destination','Solo'])['Cabin_side'].transform(lambda x: x.fillna(pd.Series.mode(x)[0]))[na_rows_CD]

# Print number of missing values left
print('#Cabin_side missing values before:',CD_bef)
print('#Cabin_side missing values after:',data['Cabin_side'].isna().sum())


# #### Cabin Number and Group

# In[64]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Cabin_number', y='Group', hue='Cabin_deck', data=data)
plt.title('Scatter Plot with Hue')
plt.xlabel('Cabin Number')
plt.ylabel('Group')
plt.legend(title='Deck')

# Adjust y-axis tick labels
plt.yticks(rotation=45)  # Rotate y-axis labels for better readability

plt.tight_layout()
plt.show()


# There is a relationship between the cabin_number and Group columns, and this relationship holds true for each specific deck on the ship. In other words, for each deck, there seems to be a linear correlation between the cabin numbers and the group.
# 
# To leverage this pattern and impute missing cabin_number values, we can perform linear regression separately for each deck. Linear regression is a statistical technique that models the relationship between two variables by fitting a linear equation to the observed data.
# 
# In this context, we would do the following:
# 
# Group the Data: Group the data by the Cabin_deck column, creating subsets of data for each deck.
# 
# Within-Deck Linear Regression: For each deck's subset of data, perform a linear regression where you predict the cabin_number based on the group_number.
# 
# Impute Missing Values: For the rows where cabin_number is missing and belongs to a specific deck, we will use the linear regression equation to estimate and impute a missing cabin_number value based on the corresponding group_number.
# 
# By performing this process for each deck, we are essentially capturing the linear relationship between cabin_number and group_number specific to each deck. This allows you to make reasonable estimations for missing cabin_number values based on the observed pattern within each deck.
# 
# Keep in mind that while this approach might provide reasonable estimates for missing values, it's still an approximation and might not be accurate in all cases. Additionally, performing linear regression within each deck assumes that the relationship between cabin_number and group_number is linear, which might not always be the case.

# In[65]:


# Missing values before
CN_bef=data['Cabin_number'].isna().sum()

# Extrapolate linear relationship on a deck by deck basis
for deck in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
    # Features and labels
    X_CN=data.loc[~(data['Cabin_number'].isna()) & (data['Cabin_deck']==deck),'Group']
    y_CN=data.loc[~(data['Cabin_number'].isna()) & (data['Cabin_deck']==deck),'Cabin_number']
    X_test_CN=data.loc[(data['Cabin_number'].isna()) & (data['Cabin_deck']==deck),'Group']

    # Linear regression
    model_CN=LinearRegression()
    model_CN.fit(X_CN.values.reshape(-1, 1), y_CN)
    preds_CN=model_CN.predict(X_test_CN.values.reshape(-1, 1))
    
    # Fill missing values with predictions
    data.loc[(data['Cabin_number'].isna()) & (data['Cabin_deck']==deck),'Cabin_number']=preds_CN.astype(int)

# Print number of missing values left
print('#Cabin_number missing values before:',CN_bef)
print('#Cabin_number missing values after:',data['Cabin_number'].isna().sum())


# Let's update the Cabin Region Feature again

# In[66]:


data['Cabin_region1']=(data['Cabin_number']<300).astype(int)
data['Cabin_region2']=((data['Cabin_number']>=300) & (data['Cabin_number']<600)).astype(int)
data['Cabin_region3']=((data['Cabin_number']>=600) & (data['Cabin_number']<900)).astype(int)
data['Cabin_region4']=((data['Cabin_number']>=900) & (data['Cabin_number']<1200)).astype(int)
data['Cabin_region5']=((data['Cabin_number']>=1200) & (data['Cabin_number']<1500)).astype(int)
data['Cabin_region6']=((data['Cabin_number']>=1500) & (data['Cabin_number']<1800)).astype(int)
data['Cabin_region7']=(data['Cabin_number']>=1800).astype(int)


# ### VIP

# Given this highly unbalanced distribution, we should impute the the VIP value with Mode, we should fill them with the most common value that appears in the non-missing data. In this case, since the feature is highly unbalanced, the most common value is likely to be the majority class, which is the value that occurs more frequently in the dataset.

# In[67]:


V_bef=data['VIP'].isna().sum()

data.loc[data['VIP'].isna(),'VIP']=False

print('#VIP missing values before:',V_bef)
print('#VIP missing values after:',data['VIP'].isna().sum())


# ### Age

# In[68]:


data.groupby(['HomePlanet','No_spending','Solo','Cabin_deck'])['Age'].median().unstack().fillna(0)


# Age varies across several features, including:
# 
# * HomePlanet
# * Group Size
# * No_Spending
# * Cabin Deck
# 
# Imputing missing ages based on subgroup medians helps to retain the variability in age within each subgroup. This is important because age can vary significantly based on these different features.By using subgroup-specific medians, the imputed ages are more likely to reflect real-world patterns and relationships. For example, if passengers from a particular HomePlanet tend to be younger or older, this pattern is captured in the subgroup median. Using subgroup medians reduces bias that might arise from imputing missing ages with a single overall median. This approach provides a more nuanced and accurate imputation method.

# In[69]:


A_bef=data['Age'].isna().sum().sum()

na_rows_A=data.loc[data['Age'].isna(),'Age'].index
data.loc[data['Age'].isna(),'Age']=data.groupby(['HomePlanet','No_spending','Solo','Cabin_deck'])['Age'].transform(lambda x: x.fillna(x.median()))[na_rows_A]

print('#Age missing values before:',A_bef)
print('#Age missing values after:',data['Age'].isna().sum())


# Update Age Group again

# In[70]:


data['Age_group']


# In[71]:


data.loc[data['Age']<=18,'Age_group']='Child'
data.loc[(data['Age']>18) & (data['Age']<=25),'Age_group']='Adolescent'
data.loc[(data['Age']>25),'Age_group']='Adult'


# ### Cryosleep

# The presence of Total_Spending might indicate whether a passenger is in CryoSleep or not. If a passenger has spent money, they might not be in CryoSleep, and if they haven't spent anything, they might be in CryoSleep.

# In[72]:


crosstab = pd.crosstab(data['CryoSleep'], data['No_spending'])


# In[73]:


plt.figure(figsize=(8, 6))
sns.heatmap(crosstab, annot=True, fmt='d')
plt.title('Cross-Tabulation: CryoSleep vs No_Spending')
plt.xlabel('CryoSleep')
plt.ylabel('No_Spending')
plt.show()


# In[74]:


HP_bef=data['CryoSleep'].isna().sum()

for index, row in data.iterrows():
    if pd.isnull(row['CryoSleep']):
        spending = row['No_spending']
        try:
            most_common = data[data['No_spending'] == spending]['CryoSleep'].mode().values[0]
        except IndexError:
            continue
        data.at[index, 'CryoSleep'] = most_common

print('#CryoSleep missing values before:',HP_bef)
print('#CryoSleep missing values after:',data['CryoSleep'].isna().sum())


# Check for missing values.

# In[75]:


data.isnull().sum()


# I think we are good to go and can move to Modeling.

# <a id="4"></a>
# # **<center><span style="color:#00BFC4;"> Preprocessing </span></center>**

# We are done with handling missing values, now we just gonna do the preprocessing. First, we will get the test and train records using Passenger ID.

# In[76]:


X=data[data['PassengerId'].isin(df_train['PassengerId'].values)].copy()
df_test=data[data['PassengerId'].isin(df_test['PassengerId'].values)].copy()


# ## Drop Unwanted Features

# In[77]:


X.drop(['PassengerId', 'Group', 'Surname', 'Cabin_number'], axis=1, inplace=True)
df_test.drop(['PassengerId', 'Group', 'Surname', 'Cabin_number'], axis=1, inplace=True)


# ## Encoding and Scaling

# To handle numerical data effectively, the StandardScaler is applied. On the other hand, categorical data requires special treatment, and this is achieved using the OneHotEncoder.  To streamline these preprocessing steps, the ColumnTransformer is employed. After applying the preprocessing transformations, we check the new shape of the transformed training data. 

# In[78]:


numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]

numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(drop='if_binary', handle_unknown='ignore',sparse=False))])

ct = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols)],
        remainder='passthrough')

X = ct.fit_transform(X)
df_test = ct.transform(df_test)

print('Training set shape:', X.shape)


# <a id="5"></a>
# # **<center><span style="color:#00BFC4;"> Modeling  </span></center>**

# In[79]:


X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2)


# I have applied gridsearchCV to find the best suitable parameters and saved the model using pickle. I will just load those models and we will use it to fit and predict

# In[80]:


with open('/kaggle/input/pickle-files/catboost.pkl', 'rb') as model_file:
    catboost_clf = pickle.load(model_file)


# In[81]:


catboost_clf.fit(X_train, y_train)

y_pred = catboost_clf.predict(X_test)
    
accuracy = accuracy_score(y_test, y_pred)


# Let's check the accuracy of our model in splitted test set.

# In[82]:


accuracy


# # Submitting

# We will submit the model which works best and we will fit entire training dataset into it.

# In[83]:


catboost_clf.fit(X, y)

prediction = catboost_clf.predict(df_test)

submission = pd.read_csv('/kaggle/input/spaceship-titanic/sample_submission.csv')
submission['Transported'] = prediction.astype(bool)
submission.to_csv('Submission.csv', index = False)


# <a id="6"></a>
# # **<center><span style="color:#00BFC4;"> Reference  </span></center>**

# I would like to express my gratitude to this [notebook](https://www.kaggle.com/code/samuelcortinhas/spaceship-titanic-a-complete-guide) by Samuel Cortinhas, whose insightful feature engineering strategies and innovative approaches to handling missing values, as presented in their remarkable notebook, served as an invaluable reference and source of inspiration for my own work. Some of the code implementations have been adapted from their notebook, enabling me to navigate complex challenges and achieve meaningful insights in this cosmic data exploration.
