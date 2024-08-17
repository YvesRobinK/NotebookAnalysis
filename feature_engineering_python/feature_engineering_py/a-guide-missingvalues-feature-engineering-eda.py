#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import missingno as msno
import warnings
warnings.filterwarnings('ignore')
# sns.set_style('darkgrid')


# # **Reading CSV**

# In[2]:


sample = pd.read_csv('../input/spaceship-titanic/sample_submission.csv')
train = pd.read_csv('../input/spaceship-titanic/train.csv')
test = pd.read_csv('../input/spaceship-titanic/test.csv')


# # **Exploring Train Dataset**
# * This is the data that will help train our model so we have to make sure what features(inputs variables) are worth using for traning the model.

# In[3]:


train.head(10)


# In[4]:


test.head(10)


# # File and Data Field Descriptions
# * **train.csv** - Personal records for about two-thirds (~8700) of the passengers, to be used as training data.
# * **PassengerId** - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
# * **HomePlanet** - The planet the passenger departed from, typically their planet of permanent residence.
# * **CryoSleep** - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
# * **Cabin** - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
# * **Destination** - The planet the passenger will be debarking to.
# * **Age** - The age of the passenger.
# * **VIP** - Whether the passenger has paid for special VIP service during the voyage.
# * **RoomService, FoodCourt, ShoppingMall, Spa, VRDeck** - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
# * **Name** - The first and last names of the passenger.
# * **Transported** - Whether the passenger was transported to another dimension. This is the **target**, the column you are trying to predict.
# * **test.csv** - Personal records for the remaining one-third (~4300) of the passengers, to be used as test data. Your task is to predict the value of Transported for the passengers in this set.
# * **sample_submission.csv** - A submission file in the correct format.
# * **PassengerId** - Id for each passenger in the test set.
# * **Transported** - The target. For each passenger, predict either True or False.

# ## First we will see how many variables are **numerical** and **categorical** in our training data.
# * For numerical data we will perform few steps of EDA to make sure that our data is perfect to be trained.
# * For categorical data we will make dummy variables to make model more accurate.

# In[5]:


col_n = train.select_dtypes(include=np.number) # the dataframe of columns related to numbered data
col_n.columns # returns all the columns name


# In[6]:


col_c = train.select_dtypes(include=np.object) # the dataframe of columns related to categorical data
col_c.columns # returns all the columns name


# ## Data Cleaning
# * There are different approaches to deal with missing values.
# * One of the many approaches is to drop the null values which is not the best approach because you lose a healty pecentage of your data.
# * For starters we just try to drop null values and cover the loses with a bit of feature engineering.

# #### Checking the information of dataset i.e. total no of rows and total no of columns and the type of data rows are having

# In[7]:


train.info()


# In[8]:


test.info()


# #### Checking some statistics measures.

# In[9]:


train.describe()


# #### Checking how many values are null in each column and then adding all the null values of each column.

# In[10]:


train.isnull().sum()


# In[11]:


test.isnull().sum()


# #### Checking the null values using Matrix graph of Missingno Library.
# * White spaces are representing the null values.
# * Gray area is representing the not null values(available values).
# * The bar on the right side is representing (on the left side of the bar, 11) how many values are not present in each row and how many values are present (on the right side of the bar , 14).

# In[12]:


msno.matrix(train,labels = True)


# #### Calculating the data loss(in %age).

# In[13]:


(1 - train.dropna().shape[0]/train.shape[0]) * 100


# #### Dropping the null values.

# In[14]:


train = train.dropna()
test = test.dropna()


# #### Rechecking the Matrix for null values.

# In[15]:


msno.matrix(train,labels = True)


# ## Feature Engineering
# * Let's cover our losses of the data that we lost to null values.
# * Here we try to do a bit of feature engineering and add some new variable in our dataset from the ones we are provided.
# * **PassengerId** is having ggg_pp string, ggg means grp_no and pp means no of ppl in a grp, we will divide them and add them in the dataset.
# * **Cabin** is having deck/num/side string, we can divide deck , num , side separately and add into our dataset.
# * Transported feature is our target variable that we are going to predict, if transported value is True or 1 means that the person is transported successfully to the other dimension elsewise False or 0.

# In[16]:


train[['GrpNo','PplInGrp']] = train.PassengerId.str.split('_',expand = True)
train[['Deck','Num','Side']] = train.Cabin.str.split('/',expand = True)


# In[17]:


test[['GrpNo','PplInGrp']] = test.PassengerId.str.split('_',expand = True)
test[['Deck','Num','Side']] = test.Cabin.str.split('/',expand = True)


# In[18]:


train['VIP'] = np.where(train['VIP'] == True , 1, 0)
train['CryoSleep'] = np.where(train['CryoSleep'] == True , 1, 0)
train['Transported'] = np.where(train['Transported'] == True , 1, 0)


# In[19]:


train.head(5)


# In[20]:


train.head(5)


# ## Exploratory Data Analysis
# * Lets try and find what information can we extract from different visuals.

# ### **Heatmap** correlation tells us how different variables are correlated with each other.

# In[21]:


sns.heatmap(train.corr(),annot=True,center=0,vmin = -1 , vmax = 1 , linewidths = 1 , cmap = 'Greens')


# ### **Jointplot** tells us the relationship between two variables and their individual distribution.

# In[22]:


sns.jointplot('Age','RoomService',data=train , kind = 'reg')
# there is strong relationship btw these two variables


# In[23]:


sns.jointplot('Age','VRDeck',data=train , kind = 'reg' , color = 'green')


# ### **Catplot** tells the relationship between the categorical variables.

# In[24]:


sns.catplot('CryoSleep','VIP',data = train , hue = 'Destination' , )


# ### **Distplot** tells the distribution of our data.
# * When working with models distributions of features give us the good idea of what transformation we should apply.

# In[25]:


fig, axes = plt.subplots(nrows = 2, ncols = 5)    # axes is 2d array (3x3)
axes = axes.flatten()         # Convert axes to 1d array of length 9
fig.set_size_inches(15, 15)
for ax, col_n in zip(axes, col_n):
    sns.distplot(train[col_n], ax = ax)
    ax.set_title(col_n)


# ### **Q-Q Plot** helps understanding the normality of our data.
# * If the dots lined up equally with line we can say our data is following gaussian distribution.

# In[26]:


col_n = train.select_dtypes(include=np.number) # the dataframe of columns related to numbered data
col_n.columns # returns all the columns name


plt.figure(figsize=[20,15], dpi=1000)
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
for i,v in enumerate(col_n,start=1):
    stats.probplot(train[v] , plot=plt.subplot(2,5,i), fit=stats.norm)
    plt.xlabel("Quantiles")
    plt.ylabel(v)


# ### **Pairplot** plot a pairwise relationships in a dataset.

# In[27]:


sns.pairplot(col_n)

