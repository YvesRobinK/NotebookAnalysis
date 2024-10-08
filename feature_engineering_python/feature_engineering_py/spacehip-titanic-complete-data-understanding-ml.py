#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip -q --disable-pip-version-check install mplcyberpunk')


# # <p style="background-color:#a782ec;font-family:newtimeroman;color:#74006f;font-size:150%;text-align:center;border-radius:20px 40px;">SPACESHIP TITANIC</p>

# <h1 align='center'>Introduction 📝</h1>
# The goal of the competition is to predict whether a passenger was transported to an alternate dimension during the Spaceship Titanic's collision with the spacetime anomaly.This notebook will contain almost all the necessary steps and methods which will be helpful in this competition.

# <h1 align='center'>Dataset Info 📈</h1>
# <b>Columns of the train data-</b> 
# 
# * ```PassengerId``` - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
# * ```HomePlanet``` - The planet the passenger departed from, typically their planet of permanent residence.
# * ```CryoSleep``` - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
# * ```Cabin``` - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
# * ```Destination``` - The planet the passenger will be debarking to.
# * ```Age``` - The age of the passenger.
# * ```VIP``` - Whether the passenger has paid for special VIP service during the voyage.
# * ```RoomService, FoodCourt, ShoppingMall, Spa, VRDeck``` - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
# * ```Name``` - The first and last names of the passenger.
# * ```Transported``` -  Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

# <h1 align='center'>Evaluation Metric 📐</h1>
# Submissions are evaluated based on their classification accuracy, the percentage of predicted labels that are correct.
# 
# <img src='https://miro.medium.com/max/1400/1*Ymyg5nHVy-FG429oMkKHkA.jpeg' width=600px>

# <div class="alert alert-block alert-warning">
#     <h2 align='center'>Please consider upvoting the kernel if you found it useful.</h2>
# </div>

# # <p style="background-color:#a782ec;font-family:newtimeroman;color:#74006f;font-size:150%;text-align:center;border-radius:20px 40px;">TABLE OF CONTENTS</p>
# <ul style="list-style-type:square">
#     <li><a href="#1">Importing Libraries</a></li>
#     <li><a href="#2">Reading the data</a></li>
#     <li><a href="#3">Exploratory Data Analysis</a></li>
#     <ul style="list-style-type:disc">
#         <li><a href="#3.1">Missing Values</a></li>
#         <li><a href="#3.2">Transported</a></li>
#         <li><a href="#3.3">PassengerId</a></li>
#         <li><a href="#3.4">HomePlanet</a></li>
#         <li><a href="#3.5">CyroSleep</a></li>
#         <li><a href="#3.6">Cabin</a></li>
#         <li><a href="#3.7">Destination</a></li>
#         <li><a href="#3.8">VIP</a></li>
#         <li><a href="#3.9">Age</a></li>
#         <li><a href="#3.10">RoomService, FoodCourt, ShoppingMall, Spa, VRDeck</a></li>
#     </ul>
#     <li><a href="#4">Data Pre-Processing</a></li>
#     <ul style="list-style-type:disc">
#         <li><a href="#4.1"> Handling Missing Values</a></li>
#         <li><a href="#4.2">Dropping columns</a></li>
#         <li><a href="#4.3">Encoding</a></li>
#         <li><a href="#4.4">Splitting the data</a></li>
#     </ul>
#     <li><a href="#5">ML Models</a></li>
#     <ul style="list-style-type:disc">
#         <li><a href="#5.1">Logistic Regression</a></li>
#         <li><a href="#5.2">Support Vector Classifier</a></li>
#         <li><a href="#5.3">Naive Bayes</a></li>
#         <li><a href="#5.4">Random Forest Classifier</a></li>
#         <li><a href="#5.5">Gradient Boosting Classifier</a></li>
#         <li><a href="#5.6">XGBoost Classifier</a></li>
#         <li><a href="#5.7">SHAP</a></li>
#         <li><a href="#5.8">Ensembling</a></li>
#     </ul>
# </ul>

# <a id='1'></a>
# # <p style="background-color:#a782ec;font-family:newtimeroman;color:#74006f;font-size:150%;text-align:center;border-radius:20px 40px;">IMPORTING LIBRARIES</p>

# In[2]:


import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplcyberpunk
from termcolor import colored
plt.style.use('cyberpunk')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.simplefilter('ignore')


# <a id='2'></a>
# # <p style="background-color:#a782ec;font-family:newtimeroman;color:#74006f;font-size:150%;text-align:center;border-radius:20px 40px;">READING THE DATA</p>

# In[3]:


df_train = pd.read_csv('../input/spaceship-titanic/train.csv')
df_test = pd.read_csv('../input/spaceship-titanic/test.csv')


# In[4]:


df_train.head()


# In[5]:


df_test.head()


# In[6]:


print(colored(f'Number of rows in train data: {df_train.shape[0]}', 'red'))
print(colored(f'Number of columns in train data: {df_train.shape[1]}', 'red'))
print(colored(f'Number of rows in test data: {df_test.shape[0]}', 'blue'))
print(colored(f'Number of columns in test data: {df_test.shape[1]}', 'blue'))


# In[7]:


df_train.describe()


# In[8]:


df_test.describe()


# <a id='3'></a>
# # <p style="background-color:#a782ec;font-family:newtimeroman;color:#74006f;font-size:150%;text-align:center;border-radius:20px 40px;">EXPLORATORY DATA ANALYSIS</p>
# ### We perform EDA to analyse and gain insights of the data which will help in better understanding the problem and will bring an advantage when creating models.

# <a id='3.1'></a>
# ## 1. Missing Values
# ### First of all, we will start by analying the missing values in the dataset. 

# In[9]:


plt.figure(figsize=(20,6))

na = pd.DataFrame(df_train.isna().sum())

sns.barplot(y=na[0], x=na.index)
plt.title('Missing Values Distribution', size = 20, weight='bold')
print(colored("Missing values column wise -", 'magenta'))
print(colored(df_train.isna().sum(), 'magenta'))

plt.show()


# <h2><u>INSIGHTS FROM THE GRAPH</u></h2>
# 
# * We can observe that almost all the columns have Null values. This is a problem that can be solved in multiple ways depeding on the situation which will we see later.
# * Let us also analyse how much these Null values of each column affects Transported.

# In[10]:


fig, ax = plt.subplots(4, 3, figsize=(20, 20))
fig.suptitle("Missing Values Distribution By Transported", size = 20, weight='bold')
fig.subplots_adjust(top=0.95)
i = 0
for x in df_train.columns:
    if len(df_train[df_train[x].isna()==True])>0:
        sns.countplot(x='Transported', data=df_train[df_train[x].isna()==True], ax=fig.axes[i], palette='turbo')
        fig.axes[i].set_title(x, weight='bold')
        i += 1
        
plt.show()


# <h2><u>INSIGHTS FROM THE GRAPH</u></h2>
# 
# * We can observe that all the columns have almost equal distribution of target class. 
# * So one thing we can say that even if we drop rows with missing values according to a particular column, it would not create any bias in the data(but we will not drop the rows right now).

# <a id='3.2'></a>
# ## 2. Transported
# ### Transported is the target column which we have to predict. Let us analyse its distribution.

# In[11]:


plt.figure(figsize=(9,6))

sns.countplot(x='Transported', data=df_train, palette = 'winter')
plt.title("Transported Distribution", size = 20, weight='bold')

print(colored(f"Percentage of Passengers Transported - {(len(df_train[df_train['Transported']==True]) / df_train.shape[0])*100:.2f}%", 'cyan'))
print(colored(f"Percentage of Passengers Not Transported - {(len(df_train[df_train['Transported']==False]) / df_train.shape[0])*100:.2f}%", 'cyan'))

plt.show()


# <h2><u>INSIGHTS FROM THE GRAPH</u></h2>
# 
# * We can observe that the target column is almost perfectly balanced, so we don't have to worry about unequal distribution.

# <a id='3.3'></a>
# ## 3. PassengerId
# All the passengers have unique id so we can't use this feature directly for modelling. But we will not discard this feature as we can perform some feature engineering to extract useful information from it. Also it does not contain any null values so extracting features from it might be very helpful.
# 
# Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group, so we can get to know the size of the group and can check how it affects Transported.

# In[12]:


fig, ax = plt.subplots(1, 2, figsize=(20,7))
fig.suptitle("GroupSize Distribution", size = 20, weight='bold')

df_train['Group'] = df_train['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
df_train['GroupSize']=df_train['Group'].map(lambda x: df_train['Group'].value_counts()[x])

df_temp = df_train.drop_duplicates(subset = ["Group"], keep='last')

sns.countplot(x='GroupSize', data=df_temp, ax=ax[0])
sns.countplot(x='GroupSize', data=df_temp, hue='Transported', ax=ax[1])

print(colored(f"Number of unique groups - {len(df_temp)}",'blue'))
data = pd.DataFrame(df_temp['GroupSize'].value_counts()).reset_index().rename(columns={'index': 'GroupSize', 'GroupSize':'Count'})
print(colored("Group Size Distribution - ",'blue'))
print(colored(data, 'blue'))
plt.show()

print(colored(f"Total number of individual passengers - {len(df_temp[df_temp['GroupSize']==1])}", 'blue'))
print(colored(f"Number of individual passengers transported - {len(df_temp[(df_temp['GroupSize']==1) & (df_temp['Transported']==True)])}", 'blue'))
print(colored(f"Number of individual passengers not transported - {len(df_temp[(df_temp['GroupSize']==1) & (df_temp['Transported']==False)])}", 'blue'))
print(colored(f"Toal number of non individual passengers - {len(df_temp[df_temp['GroupSize']!=1])}", 'red'))
print(colored(f"Number of non individual passengers transported - {len(df_temp[(df_temp['GroupSize']!=1) & (df_temp['Transported']==True)])}", 'red'))
print(colored(f"Number of non individual passengers not transported - {len(df_temp[(df_temp['GroupSize']!=1) & (df_temp['Transported']==False)])}", 'red'))


# <h2><u>INSIGHTS FROM THE GRAPH</u></h2>
# 
# * We can observe that most of the passengers are individual passengers. 
# * Apart from that the maximum size of the group is 8. 
# * An interesting observation is that there is lesser chance of passenger to be transported if he/she is an individual than in a group.

# <a id='3.4'></a>
# ## 4. HomePlanet
# <b> This is one of the categorical feature. There are 3 unique values and this feature contains some null values which we need to take care but first let's check its distribution and dependence on Transported.</b>

# In[13]:


fig, ax = plt.subplots(1, 2, figsize=(20,6))
fig.suptitle('HomePlanet Distribution', size = 20, weight='bold')

sizes = list(df_train['HomePlanet'].value_counts(sort=False))

labels = df_train['HomePlanet'].dropna().unique()
colors = ['#099FFF', '#CC00FF', '#13CA91']
explode = (0.05,0.05,0.05) 

ax[0].pie(sizes, colors=colors, explode=explode, startangle=90, labels=labels,
       autopct='%1.2f%%', pctdistance=0.6,textprops={'fontsize':12})
sns.countplot(x='HomePlanet', data=df_train, hue='Transported', ax=ax[1])

print(colored("HomePlanet Distribution - ",'green'))
data = pd.DataFrame(df_train['HomePlanet'].value_counts()).reset_index().rename(columns={'index': 'HomePlanet', 'HomePlanet':'Count'})
print(colored(data, 'green'))

plt.show()


# <h2><u>INSIGHTS FROM THE GRAPH</u></h2>
# 
# * We can observe that most of the passengers are from Earth. 
# * There is a higher chance of residents from Europa to be transported than others.
# * The chances of the residents of Earth of getting transported is less.
# * There is equal probability for the residents of Mars.

# <a id='3.5'></a>
# ## 5. CyroSleep
# <b>This is another categorical feature with value either True or False and this feature contains maximum null values which we need to take care but first let's check its distribution and dependence on Transported.</b>

# In[14]:


fig, ax = plt.subplots(1, 2, figsize=(20,6))
fig.suptitle('CryoSleep Distribution', size = 20, weight='bold')

sizes = list(df_train['CryoSleep'].value_counts())

labels = df_train['CryoSleep'].dropna().unique()
colors = ['#099FFF', '#CC00FF']
explode = (0.05,0.05) 

ax[0].pie(sizes, colors=colors, explode=explode, startangle=90, labels=labels,
       autopct='%1.2f%%', pctdistance=0.6,textprops={'fontsize':12})
sns.countplot(x='CryoSleep', data=df_train, hue='Transported', ax=ax[1])

print(colored("CryoSleep Distribution - ",'magenta'))
data = pd.DataFrame(df_train['CryoSleep'].value_counts()).reset_index().rename(columns={'index': 'CryoSleep', 'CryoSleep':'Count'})
print(colored(data, 'magenta'))

plt.show()


# <h2><u>INSIGHTS FROM THE GRAPH</u></h2>
# 
# * We can observe that CyroSleep has a high false percentage. 
# * One of the best thing is that this feature has a direct relationship with the Transported. 
# * If CyroSleep is False, then the chances of Transported is less (Transported=False) whereas if CyroSleep is True, then the chances of Transported is high (Transported=True).

# <a id='3.6'></a>
# ## 6. Cabin
# Just like PassengerId, we can't directly use Cabin feature. But we will not discard this feature as we can perform some feature engineering to extract useful information from it. 
# 
# Each cabin takes the form deck/num/side, where there are 8 unique deck values and side can be either P for Port or S for Starboard. So we can extract these features to know their distribution and can check how it affects Transported.

# In[15]:


df_train['Cabin'].fillna('Z/9999/Z', inplace=True)
df_train['deck'] = df_train['Cabin'].apply(lambda x : x.split('/')[0])
df_train['side'] = df_train['Cabin'].apply(lambda x : x.split('/')[2])

fig, ax = plt.subplots(2, 2, figsize=(20,12))
fig.suptitle('Cabin Distribution', size = 20, weight='bold')

sns.countplot(x='deck', data=df_train, order=['A','B','C','D','E','F','G','T','Z'], ax=ax[0][0], palette='turbo')
sns.countplot(x='deck', data=df_train, order=['A','B','C','D','E','F','G','T','Z'], hue='Transported', ax=ax[0][1])

sns.countplot(x='side', data=df_train, ax=ax[1][0], palette='turbo')
sns.countplot(x='side', data=df_train, hue='Transported', ax=ax[1][1])

print(colored("Cabin Deck Distribution - ",'red'))
data = pd.DataFrame(df_train['deck'].value_counts()).reset_index().rename(columns={'index': 'Deck', 'deck':'Count'})
print(colored(data, 'red'))

print(colored("Cabin Side Distribution - ",'blue'))
data = pd.DataFrame(df_train['side'].value_counts()).reset_index().rename(columns={'index': 'Side', 'side':'Count'})
print(colored(data, 'blue'))

plt.show()


# <h2><u>INSIGHTS FROM THE GRAPH</u></h2>
# 
# * The distribution of deck is very unequal whereas there is an almost equal distribution of side.
# * Also, there are only 5 samples of deck 'T'.
# * Apart from that, there is no proper conclusion on how the deck affects Transported as few classes have almost equal distribution whereas some has huge difference.
# * But if you look at the Cabin's Side, you’ll notice that passenger with side 'S' has higher chance of getting transported than side 'P'.

# <a id='3.7'></a>
# ## 7. Destination
# <b> This is also a categorical feature. There are 3 unique values and this feature also contains some null values which we need to take care but first let's check its distribution and dependence on Transported.</b>

# In[16]:


fig, ax = plt.subplots(1, 2, figsize=(20,6))
fig.suptitle('Destination Distribution', size = 20, weight='bold')

sizes = list(df_train['Destination'].value_counts(sort=False))

labels = df_train['Destination'].dropna().unique()
colors = ['#099FFF', '#CC00FF', '#13CA91']
explode = (0.05,0.05,0.05) 

ax[0].pie(sizes, colors=colors, explode=explode, startangle=90, labels=labels,
       autopct='%1.2f%%', pctdistance=0.6,textprops={'fontsize':12})
sns.countplot(x='Destination', data=df_train, hue='Transported', ax=ax[1])

print(colored("Destination Distribution - ",'cyan'))
data = pd.DataFrame(df_train['Destination'].value_counts()).reset_index().rename(columns={'index': 'Destination', 'Destination':'Count'})
print(colored(data, 'cyan'))

plt.show()


# <h2><u>INSIGHTS FROM THE GRAPH</u></h2>
# 
# * We can observe that most of the passenger's destination is TRAPPIST-1e.
# * Apart from that, the chances of getting transported is maximum for the passengers having destination as 55 Cancri e, but the distribution is very much equal for the other two destinations.

# <a id='3.8'></a>
# ## 8. VIP
# <b>This is the last categorical feature with value either True or False and this feature also contains some null values which we need to take care but first let's check its distribution and dependence on Transported.</b>

# In[17]:


fig, ax = plt.subplots(1, 2, figsize=(20,6))
fig.suptitle('VIP Distribution', size = 20, weight='bold')

sizes = list(df_train['VIP'].value_counts(sort=False))

labels = df_train['VIP'].dropna().unique()
colors = ['#099FFF', '#CC00FF']
explode = (0.25,0.25) 

ax[0].pie(sizes, colors=colors, explode=explode, startangle=90, labels=labels,
       autopct='%1.2f%%', pctdistance=0.6,textprops={'fontsize':12})
sns.countplot(x='VIP', data=df_train, hue='Transported', ax=ax[1])

print(colored("VIP Distribution - ",'green'))
data = pd.DataFrame(df_train['VIP'].value_counts()).reset_index().rename(columns={'index': 'VIP', 'VIP':'Count'})
print(colored(data, 'green'))

plt.show()


# <h2><u>INSIGHTS FROM THE GRAPH</u></h2>
# 
# * We can observe that VIP has a high false percentage. 
# * Also, this feature doesn't look useful as the transported distribution is almost equal for both VIP and non VIP passengers.

# <a id='3.9'></a>
# ## 9. Age
# <b>This is a continuos value feature with values ranging from 0 to 79. This feature also contains some null values which we need to take care but first let's check its distribution and dependence on Transported.</b>

# In[18]:


fig, ax = plt.subplots(1, 2, figsize=(20,6))
fig.suptitle('Age Distribution', size = 20, weight='bold')

sns.boxplot(x='Transported', y='Age', data=df_train, ax=ax[0])
sns.histplot(x='Age', element="step", kde=True, data=df_train, hue='Transported', ax=ax[1])

print(colored("Transported Passengers Age Distribution - ", 'magenta'))
print(colored(f"Minimum Age - {df_train[df_train['Transported']==True]['Age'].describe()['min']}", 'magenta'))
print(colored(f"Maximum Age - {df_train[df_train['Transported']==True]['Age'].describe()['max']}", 'magenta'))
print(colored(f"Average Age - {df_train[df_train['Transported']==True]['Age'].describe()['mean']}", 'magenta'))

print(colored("Non Transported Passengers Age Distribution - ", 'blue'))
print(colored(f"Minimum Age - {df_train[df_train['Transported']==False]['Age'].describe()['min']}", 'blue'))
print(colored(f"Maximum Age - {df_train[df_train['Transported']==False]['Age'].describe()['max']}", 'blue'))
print(colored(f"Average Age - {df_train[df_train['Transported']==False]['Age'].describe()['mean']}", 'blue'))

mplcyberpunk.make_lines_glow()

plt.show()


# <h2><u>INSIGHTS FROM THE GRAPH</u></h2>
# 
# * From the first look we can observe that the distribution of age is almost similar for both transported and non transported passengers. 
# * But if we look more carefully, we can oberve that initially the age distribution of transported passengers is high indicating that passengers having age less than 10 have higher chances of getting transported but it is quite opposite for the passengers who are in their 20s.
# * But rest the age distribution is quite similar for both classes. 

# <a id='3.10'></a>
# ## 10. RoomService, FoodCourt, ShoppingMall, Spa, VRDeck
# <b>These are the columns that contains the amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.These all are continuos feature and we will check their distribution and dependence on Transported.</b>

# In[19]:


fig, ax = plt.subplots(1, 2, figsize=(20,6))
fig.suptitle('RoomService Distribution', size = 20, weight='bold')

sns.boxplot(x='Transported', y='RoomService', data=df_train, ax=ax[0])
sns.histplot(x='RoomService', element="step", kde=True, data=df_train, hue='Transported', bins=100, ax=ax[1])

print(colored(f"Percentage of Passengers with no RoomService Billing - {len(df_train[df_train['RoomService']==0.0])/len(df_train)*100:.2f}%", 'red'))
print(colored("Transported Passengers RoomService Billing Distribution - ", 'magenta'))
print(colored(f"Minimum RoomService Billing - {df_train[df_train['Transported']==True]['RoomService'].describe()['min']}", 'magenta'))
print(colored(f"Maximum RoomService Billing - {df_train[df_train['Transported']==True]['RoomService'].describe()['max']}", 'magenta'))
print(colored(f"Average RoomService Billing - {df_train[df_train['Transported']==True]['RoomService'].describe()['mean']}", 'magenta'))

print(colored("Non Transported Passengers RoomService Billing Distribution - ", 'blue'))
print(colored(f"Minimum RoomService Billing - {df_train[df_train['Transported']==False]['RoomService'].describe()['min']}", 'blue'))
print(colored(f"Maximum RoomService Billing - {df_train[df_train['Transported']==False]['RoomService'].describe()['max']}", 'blue'))
print(colored(f"Average RoomService Billing - {df_train[df_train['Transported']==False]['RoomService'].describe()['mean']}", 'blue'))

mplcyberpunk.make_lines_glow()

plt.show()

fig, ax = plt.subplots(1, 2, figsize=(20,6))
fig.suptitle('FoodCourt Distribution', size = 20, weight='bold')

sns.boxplot(x='Transported', y='FoodCourt', data=df_train, ax=ax[0])
sns.histplot(x='FoodCourt', element="step", kde=True, data=df_train, hue='Transported', bins=100, ax=ax[1])

print(colored(f"Percentage of Passengers with no FoodCourt Billing - {len(df_train[df_train['FoodCourt']==0.0])/len(df_train)*100:.2f}%", 'red'))
print(colored("Transported Passengers FoodCourt Billing Distribution - ", 'magenta'))
print(colored(f"Minimum FoodCourt Billing - {df_train[df_train['Transported']==True]['FoodCourt'].describe()['min']}", 'magenta'))
print(colored(f"Maximum FoodCourt Billing - {df_train[df_train['Transported']==True]['FoodCourt'].describe()['max']}", 'magenta'))
print(colored(f"Average FoodCourt Billing - {df_train[df_train['Transported']==True]['FoodCourt'].describe()['mean']}", 'magenta'))

print(colored("Non Transported Passengers FoodCourt Billing Distribution - ", 'blue'))
print(colored(f"Minimum FoodCourt Billing - {df_train[df_train['Transported']==False]['FoodCourt'].describe()['min']}", 'blue'))
print(colored(f"Maximum FoodCourt Billing - {df_train[df_train['Transported']==False]['FoodCourt'].describe()['max']}", 'blue'))
print(colored(f"Average FoodCourt Billing - {df_train[df_train['Transported']==False]['FoodCourt'].describe()['mean']}", 'blue'))


mplcyberpunk.make_lines_glow()

plt.show()

fig, ax = plt.subplots(1, 2, figsize=(20,6))
fig.suptitle('ShoppingMall Distribution', size = 20, weight='bold')

sns.boxplot(x='Transported', y='ShoppingMall', data=df_train, ax=ax[0])
sns.histplot(x='ShoppingMall', element="step", kde=True, data=df_train, hue='Transported', bins=100, ax=ax[1])

print(colored(f"Percentage of Passengers with no ShoppingMall Billing - {len(df_train[df_train['ShoppingMall']==0.0])/len(df_train)*100:.2f}%", 'red'))
print(colored("Transported Passengers ShoppingMall Billing Distribution - ", 'magenta'))
print(colored(f"Minimum ShoppingMall Billing - {df_train[df_train['Transported']==True]['ShoppingMall'].describe()['min']}", 'magenta'))
print(colored(f"Maximum ShoppingMall Billing - {df_train[df_train['Transported']==True]['ShoppingMall'].describe()['max']}", 'magenta'))
print(colored(f"Average ShoppingMall Billing - {df_train[df_train['Transported']==True]['ShoppingMall'].describe()['mean']}", 'magenta'))

print(colored("Non Transported Passengers ShoppingMall Billing Distribution - ", 'blue'))
print(colored(f"Minimum ShoppingMall Billing - {df_train[df_train['Transported']==False]['ShoppingMall'].describe()['min']}", 'blue'))
print(colored(f"Maximum ShoppingMall Billing - {df_train[df_train['Transported']==False]['ShoppingMall'].describe()['max']}", 'blue'))
print(colored(f"Average ShoppingMall Billing - {df_train[df_train['Transported']==False]['ShoppingMall'].describe()['mean']}", 'blue'))


mplcyberpunk.make_lines_glow()

plt.show()

fig, ax = plt.subplots(1, 2, figsize=(20,6))
fig.suptitle('Spa Distribution', size = 20, weight='bold')

sns.boxplot(x='Transported', y='Spa', data=df_train, ax=ax[0])
sns.histplot(x='Spa', element="step", kde=True, data=df_train, hue='Transported', bins=100, ax=ax[1])

print(colored(f"Percentage of Passengers with no Spa Billing - {len(df_train[df_train['Spa']==0.0])/len(df_train)*100:.2f}%", 'red'))
print(colored("Transported Passengers Spa Billing Distribution - ", 'magenta'))
print(colored(f"Minimum Spa Billing - {df_train[df_train['Transported']==True]['Spa'].describe()['min']}", 'magenta'))
print(colored(f"Maximum Spa Billing - {df_train[df_train['Transported']==True]['Spa'].describe()['max']}", 'magenta'))
print(colored(f"Average Spa Billing - {df_train[df_train['Transported']==True]['Spa'].describe()['mean']}", 'magenta'))

print(colored("Non Transported Passengers Spa Billing Distribution - ", 'blue'))
print(colored(f"Minimum Spa Billing - {df_train[df_train['Transported']==False]['Spa'].describe()['min']}", 'blue'))
print(colored(f"Maximum Spa Billing - {df_train[df_train['Transported']==False]['Spa'].describe()['max']}", 'blue'))
print(colored(f"Average Spa Billing - {df_train[df_train['Transported']==False]['Spa'].describe()['mean']}", 'blue'))

mplcyberpunk.make_lines_glow()

plt.show()

fig, ax = plt.subplots(1, 2, figsize=(20,6))
fig.suptitle('VRDeck Distribution', size = 20, weight='bold')

sns.boxplot(x='Transported', y='VRDeck', data=df_train, ax=ax[0])
sns.histplot(x='VRDeck', element="step", kde=True, data=df_train, hue='Transported', bins=100, ax=ax[1])

print(colored(f"Percentage of Passengers with no VRDeck Billing - {len(df_train[df_train['VRDeck']==0.0])/len(df_train)*100:.2f}%", 'red'))
print(colored("Transported Passengers VRDeck Billing Distribution - ", 'magenta'))
print(colored(f"Minimum VRDeck Billing - {df_train[df_train['Transported']==True]['VRDeck'].describe()['min']}", 'magenta'))
print(colored(f"Maximum VRDeck Billing - {df_train[df_train['Transported']==True]['VRDeck'].describe()['max']}", 'magenta'))
print(colored(f"Average VRDeck Billing - {df_train[df_train['Transported']==True]['VRDeck'].describe()['mean']}", 'magenta'))

print(colored("Non Transported Passengers VRDeck Billing Distribution - ", 'blue'))
print(colored(f"Minimum VRDeck Billing - {df_train[df_train['Transported']==False]['VRDeck'].describe()['min']}", 'blue'))
print(colored(f"Maximum VRDeck Billing - {df_train[df_train['Transported']==False]['VRDeck'].describe()['max']}", 'blue'))
print(colored(f"Average VRDeck Billing - {df_train[df_train['Transported']==False]['VRDeck'].describe()['mean']}", 'blue'))

mplcyberpunk.make_lines_glow()

plt.show()


# <h2><u>INSIGHTS FROM THE GRAPH</u></h2>
# 
# * We can clearly observe that all the distributions are centered towards zero. 
# * There are more than 60% passengers in each distribution who have not paid for that service.
# * Also, there are few cases with very high billings (looks like an outlier)
# * It looks like using these features directly for modelling won't help and we might need to create new features from these to have better performance.

# ### We have observed the distribution of these continuos features and looked how these affect the Transported. But still we need more clarity on these features. So next we will check their correlation with some other features like Age and VIP and can get to know how much they are correlated and affect Transported.

# In[20]:


fig, ax = plt.subplots(3, 2, figsize=(20, 15))
fig.suptitle('Continuous Features VS Age', size = 20, weight='bold')
fig.delaxes(ax[2][1])

df_temp = df_train.iloc[:, 5:12]
columns = df_temp.columns[2:]
for i, col in enumerate(columns):
    sns.scatterplot(x='Age', y=col, hue='Transported', data=df_train, ax=fig.axes[i], palette='turbo')
    fig.axes[i].set_title(f'{col} VS Age', weight='bold')
plt.show()


# <h2><u>INSIGHTS FROM THE GRAPH</u></h2>
# 
# * As expected, the expendicture of each feature is almost zero for passengers with age less than 10.
# * The distribution of the transported and non-transported is quite similar with respect to Age in the case FoodCourt and ShoppingMall.
# * Whereas we can observe that non-transported passengers have spent more in the case of RoomService, Spa and VRDeck.

# In[21]:


fig, ax = plt.subplots(3, 2, figsize=(20, 15))
fig.suptitle('Continuous Features VS VIP', size = 20, weight='bold')
fig.delaxes(ax[2][1])

for i, col in enumerate(columns):
    sns.stripplot(x="VIP", y=col, hue='Transported', data=df_train, dodge=True, ax=fig.axes[i], palette='winter')
    fig.axes[i].set_title(f'{col} VS VIP', weight='bold')
plt.show()


# <h2><u>INSIGHTS FROM THE GRAPH</u></h2>
# 
# * Although, it is expected that VIP passengers must have spent more than normal passengers, but as the count of VIP passengers is very less, we can't observe the same from the above plots.
# * Apart from that the distribution looks quite similar for each feature for both VIP and non-VIP passengers.

# ### As I stated before, we can create new features from these 5 continuous features to get more valuable dataset. 
# ### New Features :- 
# 1) Total Expenses - Sum of all the 5 expenses.<br>
# 2) NoSpent - Whether the passenger has spent anything or not.

# In[22]:


df_train['Total_Expenses'] = df_train[df_temp.columns[2:]].sum(axis=1)
df_train['NoSpent'] = df_train['Total_Expenses']==0

fig, ax = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('Total Expenses Distribution', size = 20, weight='bold')

sns.boxplot(x='Transported', y='Total_Expenses', data=df_train, ax=ax[0], palette='turbo')
sns.histplot(x='Total_Expenses', element="step", kde=True, data=df_train, hue='Transported', bins=100, ax=ax[1], palette='turbo')

print(colored("Total_Expenses Distribution - ", 'cyan'))
print(colored(f"Minimum Total_Expenses - {df_train['Total_Expenses'].describe()['min']}", 'cyan'))
print(colored(f"Maximum Total_Expenses - {df_train['Total_Expenses'].describe()['max']}", 'cyan'))
print(colored(f"Average Total_Expenses - {df_train['Total_Expenses'].describe()['mean']}", 'cyan'))

plt.show()

fig, ax = plt.subplots(1, 2, figsize=(20,6))
fig.suptitle('NoSpent Distribution', size = 20, weight='bold')

sizes = list(df_train['NoSpent'].value_counts(sort=False))

labels = df_train['NoSpent'].dropna().unique()
colors = ['#13CA91', '#e5ab09']
explode = (0.0,0.05) 

ax[0].pie(sizes, colors=colors, explode=explode, startangle=90, labels=labels,
       autopct='%1.2f%%', pctdistance=0.6,textprops={'fontsize':12})
sns.countplot(x='NoSpent', data=df_train, hue='Transported', ax=ax[1], palette='turbo')

print(colored("NoSpent Distribution - ",'cyan'))
data = pd.DataFrame(df_train['NoSpent'].value_counts()).reset_index().rename(columns={'index': 'NoSpent', 'NoSpent':'Count'})
print(colored(data, 'cyan'))

plt.show()


# <h2><u>INSIGHTS FROM THE GRAPH</u></h2>
# 
# * We can observe that the total expenses is still mostly near to zero and the distribution is almost same for both transported and non-transported passengers.
# * On the other hand we can observe a direct relationship with the NoSpent and Transported feature such that the person who hasn't spent anything has higher chance of getting transported.

# <a id='4'></a>
# # <p style="background-color:#a782ec;font-family:newtimeroman;color:#74006f;font-size:150%;text-align:center;border-radius:20px 40px;">DATA PRE-PROCESSING</p>
# After exploring the data, now it's time to process the data so that it can be used to train the model. <br>
# I have already performed all the feature engineering in the EDA, so it is not included in the processing.

# In[23]:


# Let's have a quick look at the data
df_train.head()


# <a id='4.1'></a>
# ## Handling Missing Values
# As we already know, there are missing values in almost all the columns. So either we can drop all the rows with the null values or we can fill these null values. <br>
# There aren't many missing values, but it's a good idea to enter these values. This gives you more data and allows you to create better models. There are multiple ways to fill these null values depending upon each column. But I will follow an easier approach by filling the null values by either it's median or mode depending upon the column.

# In[24]:


print(colored(f"Number of missing values before - {df_train.isna().sum().sum()}", 'red'))

for col in df_train.columns:
    if col == 'Age':
        df_train[col].fillna(df_train[col].median(), inplace=True)
    else:
        df_train[col].fillna(df_train[col].mode()[0], inplace=True)
        
print(colored(f"Number of missing values after - {df_train.isna().sum().sum()}", 'blue'))


# <a id='4.2'></a>
# ## Dropping unwanted columns.

# In[25]:


df_train.drop(['PassengerId', 'Cabin', 'Group', 'Name'], axis=1, inplace=True)


# <a id='4.3'></a>
# ## Encoding
# In machine learning models, all input and output variables must be numeric. This means that if our data contains values in string or boolean format, they must be converted into numerical format before training the model.

# In[26]:


for col in df_train.columns[df_train.dtypes == object]:
    encoder = LabelEncoder()
    df_train[col] = encoder.fit_transform(df_train[col])
    
for col in df_train.columns[df_train.dtypes == bool]:
    df_train[col] = df_train[col].astype('int')


# <a id='4.4'></a>
# ## Splitting the data
# Lastly, we assign our input and the target variables and then split the data into train and validation set.

# In[27]:


X = df_train.drop('Transported', axis=1)
y = df_train['Transported']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, test_size=0.2, random_state=5)

print(colored(f"Number of rows in training set - {len(X_train)}", 'cyan'))
print(colored(f"Number of rows in validation set - {len(X_valid)}", 'magenta'))


# <a id='5'></a>
# # <p style="background-color:#a782ec;font-family:newtimeroman;color:#74006f;font-size:150%;text-align:center;border-radius:20px 40px;">ML MODELS</p>
# It is now time to develop some models following all of the experiments and preprocessing. Let's begin by developing a variety of primary machine learning models.

# In[28]:


acc_plot = {} # For plotting purpose
preds = [] # For ensembling


# <a id='5.1'></a>
# ### Logistic Regression

# In[29]:


model_lor = LogisticRegression(random_state=5).fit(X_train, y_train)
y_pred = model_lor.predict(X_valid)
acc = accuracy_score(y_valid, y_pred)
acc_plot['LogisticRegression'] = acc
print(f"Model Name: LogisticRegression ====>>> Validation Accuracy: {acc}")


# <a id='5.2'></a>
# ### Support Vector Classifier

# In[30]:


model_svc = SVC(random_state=5).fit(X_train, y_train)
y_pred = model_svc.predict(X_valid)
acc = accuracy_score(y_valid, y_pred)
acc_plot['SVC'] = acc
print(f"Model Name: SVC ====>>> Validation Accuracy: {acc}")


# <a id='5.3'></a>
# ### Naive Bayes

# In[31]:


model_nb = GaussianNB().fit(X_train, y_train)
y_pred = model_nb.predict(X_valid)
acc = accuracy_score(y_valid, y_pred)
acc_plot['NaiveBayes'] = acc
print(f"Model Name: NaiveBayes ====>>> Validation Accuracy: {acc}")


# <a id='5.4'></a>
# ### Random Forest Classifier

# In[32]:


model_rfc = RandomForestClassifier(random_state=5).fit(X_train, y_train)
y_pred = model_rfc.predict(X_valid)
acc = accuracy_score(y_valid, y_pred)
acc_plot['RFClassifier'] = acc
print(f"Model Name: RFClassifier ====>>> Validation Accuracy: {acc}")


# <a id='5.5'></a>
# ### Gradient Boosting Classifier

# In[33]:


model_gbc = GradientBoostingClassifier(random_state=5).fit(X_train, y_train)
y_pred = model_gbc.predict(X_valid)
acc = accuracy_score(y_valid, y_pred)
acc_plot['GBTClassifier'] = acc
print(f"Model Name: GBTClassifier ====>>> Validation Accuracy: {acc}")


# <a id='5.6'></a>
# ### XGBoost Classifier

# In[34]:


model_xgb = XGBClassifier(eval_metric='logloss', random_state=5).fit(X_train, y_train)
y_pred = model_xgb.predict(X_valid)
acc = accuracy_score(y_valid, y_pred)
acc_plot['XGBoost'] = acc
print(f"Model Name: XGBoost Classifier ====>>> Validation Accuracy: {acc}")


# In[35]:


plt.figure(figsize=(15, 6))
sns.barplot(list(acc_plot.keys()), list(acc_plot.values()))
plt.xlabel("ML Models")
plt.ylabel("Validation Accuracy")
plt.title("Comparison Graph", fontsize=20)
plt.show()


# <a id='5.7'></a>
# ### SHAP (SHapley Additive exPlanations)

# Let's now use SHAP to depict the summary plot to understand which feature was most important to the model. The shap value is a measure of how well that single data point was able to anticipate the results. The data point makes relatively little contribution to predictions if the shap value is extremely close to zero. Depending on whether the shap value is very positive or strongly negative, we may claim that the data point significantly influences predictions.

# In[36]:


# Initialize object that can calculate shap values
explainer = shap.TreeExplainer(model_xgb)

# Calculate Shap values
shap_values = explainer.shap_values(X_valid)


# In[37]:


shap.summary_plot(shap_values, X_valid, plot_size=(20, 15), axis_color='#ff0090')


# In[38]:


shap.summary_plot(shap_values, X_valid, plot_type='bar', plot_size=(20, 15), axis_color='#ff0090')


# <a id='5.8'></a>
# ### Ensembling
# In order to deliver the most precise predictions for a specific situation, we create machine learning models. However, a single model might not produce the most accurate results and might make certain mistakes. We may mix the findings of several models to obtain better outcomes in order to decrease these mistakes and enhance the predictions. This is referred to as ensemble learning. The aggregate findings balance out the noise and can be more accurate than any one approach. In this case, I've shown this by ensembling three of the best performing models and evaluated their valid set results.

# In[39]:


y_pred = model_svc.predict(X_valid)
preds.append(y_pred)

y_pred = model_gbc.predict(X_valid)
preds.append(y_pred)

y_pred = model_xgb.predict(X_valid)
preds.append(y_pred)


# In[40]:


final_preds = np.array(preds).mean(axis = 0)
final_preds = np.where(final_preds > 0.5, 1, 0)


# In[41]:


acc = accuracy_score(y_valid, final_preds)
print(f"Ensembling ====>>> Validation Accuracy: {acc}")


# <div class="alert alert-block alert-warning">
#     <h2 align='center'>⚠ WORK IN PROGRESS ⚠</h2>
#     <h2 align='center'>Please consider upvoting the kernel if you found it useful.</h2>
# </div>
