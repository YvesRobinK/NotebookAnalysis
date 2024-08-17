#!/usr/bin/env python
# coding: utf-8

# <center>
# <img src="https://2.bp.blogspot.com/-mNmwKeTuZZg/XR76sYkpbpI/AAAAAAAAELA/5DTQjesqBqMdMIlYwe1uOYTmdLUoBRpvACKgBGAs/w1920-h1080-c/spaceship-minimalist-sci-fi-digital-art-uhdpaper.com-4K-138.jpg" />
# </center>

# # Spaceship Titanic Solution Walkthrough 
# 
# The purpose of the notebook is to practice Exploratory Data Analysis, Visualization, and Machine Learning as well as show you how I have applied a systematic Data Science workflow as I navigate through this project.
# 
# 
# ## Data Science Workflow
# > The foundation of this workflow was based on the author's citations in this [notebook](https://www.kaggle.com/code/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy/notebook). I figured there was still room for more C's so I bothered making a more sophisticated and obssessive-compulsive framework.
# - **Comprehend.** *Exploratory Data Analysis.* Understand the nature and relationships among each features in the datasets through data analyses and visualization.
# - **Correlate.** *Feature Selection* Validate the strength of association across features with the appopriate statistical tools and metrics, and to select the features that are significantly relevant with the solution goal.
# - **Clean.** *Data Cleaning.* Identify and remedy missing/null values by imputing them with reasonable inputs.  
# - **Create.** *Feature Engineering.* Create new features out of the existing ones which can make better predictions while also reducing noise in the number of features.
# - **Convert.** *Data Preprocessing.* Perform the necessary adjustments (one-hot encoding) and data transformations (i.e. sqrt, log trasformations) to make the data fit for modelling.
# - **Complete.** *Training Model.* Completion of a working and cleaned dataset in preparation for training the model and predicting solutions out of it. 
# - **Configure.** *Hyperparameter Tuning.* Further optimize our learning algorithms by determining and running the optimal parameters. 
# - **Combine.** *Ensemble Learning.* Combine multiple algorithms into one that can leverage the strengths and compensates the weaknesses of the tested models.

# Credits to the creator who made this awesome package *mplcyberpunk*, which allows us to create visualizations surrounding the 'cyberpunk' theme. You can check more about the repository [HERE](https://github.com/dhaitz/mplcyberpunk/blob/master/README.md).

# In[ ]:


pip install mplcyberpunk


# In[ ]:


# data analysis
import pandas as pd
import numpy as np

# data visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')

# Breathtaking visuals
import mplcyberpunk


# In[ ]:


plt.style.use("cyberpunk")


# In[ ]:


train_df = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')
test_df_copy = test_df.copy()
df = [train_df, test_df]


# In[ ]:


train_df.head(10)


# In[ ]:


test_df.head()


# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# In[ ]:


train_df.describe()


# In[ ]:


def missing_values(df):
    # Calculate missing value and their percentage for each feature
    missing_percent = df.isnull().sum() * 100 / df.shape[0]
    df_missing_percent = pd.DataFrame(missing_percent).round(2)
    df_missing_percent = df_missing_percent.reset_index().rename(
                    columns={
                            'index':'Feature',
                            0:'Missing Percentage (%)'
                    }
                )
    df_missing_value = df.isnull().sum()
    df_missing_value = df_missing_value.reset_index().rename(
                    columns={
                            'index':'Feature',
                            0:'Missing Values'
                    }
                )

    Final = df_missing_value.merge(df_missing_percent, how = 'inner', left_on = 'Feature', right_on = 'Feature')
    Final = Final.sort_values(by = 'Missing Percentage (%)',ascending = False)
    return Final

missing_values(train_df)


# In[ ]:


missing_values(test_df)


# **Missing Values**
# - The proportion of missing values to the total entries in each feature are relatively small, ranging from 0% to 2.5%.
# 
# **Data Types**
# - Numerical. *Age, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck*
# - Categorical. *HomePlanet, CryoSleep, Destination, VIP*
# - Mixed/Alphanumeric. *Cabin, Name*
# - Target Categorical. *Transported*
# 
# ### Interesting Questions and Hypotheses
# - **Cabin vs Cryosleep.** Does a cryosleep facility have its designated cabin area?
# - **PassengerId group** (gggg=group, pp=number) **vs HomePlanet and Destination.** Did passengers within their groups travel together, which means coming from the same HomePlanet and debarking to the same Destination.
# - **CryoSleep vs Services.** Did passengers who elected to cryosleep have lower expenditures?
# - **CryoSleep vs PassengerId group.** Were those who traveled alone in the group likely to undergo CryoSleep?
# - **VIP vs Services.** How the services and expenditures from VIP members differ from non-VIPs?

# In[ ]:


df_num = train_df[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']]
df_cat = train_df[['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Transported']]


# # Numerical Variables

# In[ ]:


sns.pairplot(df_num, hue='Transported')
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
ax = sns.histplot(data=train_df, x="Age", hue="Transported", binwidth=1, kde=True)
plt.title('Age',
          fontsize = 18,
          fontweight = 'bold',
          fontfamily = 'serif',
          loc = 'left')

ax.set(xlabel=None, ylabel=None)


# #### Observations
# - Passengers (<18) were more likely to be transported than other age groups.
# - Passengers (21-28) were less likely to be transported.
# - Age seems to follow a normal distribution, but a little skewed to the right.
# 
# #### Decisions
# - Complete the missing values in 'Age'
# - Normalize our 'Age' distribution through data transformations.

# In[ ]:


def strip_plot(df, x, y):
    ax = sns.stripplot(x=df[x], y=df[y])
    plt.title(str(y),
          fontsize = 18,
          fontweight = 'bold',
          fontfamily = 'serif',
          loc = 'left')
    
    ax.set(xlabel=None, ylabel=None)


# In[ ]:


fig = plt.figure(figsize=(15, 15))

plt.subplot(3, 2, 1)
strip_plot(train_df, 'Transported', 'RoomService')

plt.subplot(3, 2, 2)
strip_plot(train_df, 'Transported', 'FoodCourt')

plt.subplot(3, 2, 3)
strip_plot(train_df, 'Transported', 'Spa')

plt.subplot(3, 2, 4)
strip_plot(train_df, 'Transported', 'ShoppingMall')

plt.subplot(3, 2, 5)
strip_plot(train_df, 'Transported', 'VRDeck')


# #### Observations
# 
# - The distributions on RoomService, Spa, and VRDeck follow very similar patterns, the same is observed with those of FoodCourt and ShoppingMall.
# - The bills spent by transported passengers appear to be concentrated and approaching to 0. Either they spent very little or spent no amount at all.
# - Passengers who spent more on **RoomService, Spa,** and **VRDeck** services were less likely to get transported.
# - Passengers who spent more on **FoodCourt** and **ShoppingMall** services were more likely to get transported.
# 
# #### Decisions
# 
# - Create a new feature called *Premium* that sums the bills spent on **RoomService, Spa,** and **VRDeck**
# - Create a new feature called *Basic* feature that sums those from **FoodCourt** and **ShoppingMall**.

# ## Correlating Numerical Variables
# This corr matrix will mark as our baseline to understand our numerical variables and see what we can play with for feature engineering. 

# In[ ]:


plt.subplots(figsize=(10,10))
mask = np.triu(np.ones_like(df_num.corr()))
sns.heatmap(df_num.corr(), mask=mask, cmap='cool', annot=True, annot_kws={"fontsize":13}, square=True)


# #### Observations
# - Despite following a normal distribution, *Age* has an underwhelmingly low correlation with 'Transported'.
# - So far, *RoomService, Spa, and VRDeck* have some of the highest correlation with our target variable.
# 
# #### Decisions
# - Create a categorical feature *AgeGroup* as a function of *Age*.

# ## Feature Engineering Numerical Features

# In[ ]:


# Create Basic, Premium, and All Spent
for dataset in df:
    dataset['Premium'] = dataset['RoomService'] + dataset['Spa'] + dataset['VRDeck']
    dataset['Basic'] = dataset['FoodCourt'] + dataset['ShoppingMall']
    dataset['All_Services'] = dataset['RoomService'] + dataset['Spa'] + dataset ['VRDeck'] + dataset['FoodCourt'] + dataset['ShoppingMall']

df_num = train_df[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Basic', 'Premium', 'All_Services', 'Transported']]

plt.subplots(figsize=(15,10))
mask = np.triu(np.ones_like(df_num.corr()))
sns.heatmap(df_num.corr(), mask=mask, cmap='cool', annot=True, annot_kws={"fontsize":13}, center=0, square=True)


# #### Decisions
# - I am rooting *Premium* for now, since its correlation -0.36 seems promising. Should I choose to keep it, other features that are multicolliner with *Premium* will be removed.
# - Let's try normalizing the highly skewed numerical features and see if it improves the correlation.

# ## Missing Numerical Values
# Normally, the easier and simpler way is to resort to Simple Imputation technique usually through the measures of central tendency: **mean, median, or mode**, but a more advanced technique can be employed by filling up missing values based on the insights I can find out by conducting Exploratory Data Analysis on my features.

# In[ ]:


def scatter(df, x, y, h):
    ax = sns.scatterplot(x=df[x], y=df[y], hue=df[h])
    plt.title(str(y),
          fontsize = 18,
          fontweight = 'bold',
          fontfamily = 'serif',
          loc = 'left')
    
    ax.set(xlabel=None, ylabel=None)


fig = plt.figure(figsize=(15, 15))

plt.subplot(3, 2, 1)
scatter(train_df, 'Age', 'RoomService', 'Transported')

plt.subplot(3, 2, 2)
scatter(train_df, 'Age', 'FoodCourt', 'Transported')

plt.subplot(3, 2, 3)
scatter(train_df, 'Age', 'Spa', 'Transported')

plt.subplot(3, 2, 4)
scatter(train_df, 'Age', 'ShoppingMall', 'Transported')

plt.subplot(3, 2, 5)
scatter(train_df, 'Age', 'VRDeck', 'Transported')


# In[ ]:


def box(df, x, y):
    ax = sns.boxplot(x=df[x], y=df[y], width = .3)
    plt.title(str(y) + ' by ' + str(x),
          fontsize = 18,
          fontweight = 'bold',
          fontfamily = 'serif',
          loc = 'left')
    
    ax.set(xlabel=None, ylabel=None)

    
fig = plt.figure(figsize=(15, 15))

plt.subplot(3, 2, 1)
box(train_df, 'VIP', 'Age')

plt.subplot(3, 2, 2)
box(train_df, 'HomePlanet', 'Age')


# ### Insights
# - Easily read like a book! It shows that children (<12) don't have buying power as their expenditures amounted to 0 across every service there is.
# - It is also likely that an age requirement is laid out when applying for *VIP*, this explains the slight variation in the distribution of *Age* by *VIP*.
# - Visible variation in the age distributions is also evident across *HomePlanet.*
# - Impute missing age based from *VIP* and *HomePlanet*.

# Let's now examine *Expenditures* by *CryoSleep*.

# In[ ]:


fig = plt.figure(figsize=(15, 15))

plt.subplot(3, 2, 1)
strip_plot(train_df, 'CryoSleep', 'RoomService')

plt.subplot(3, 2, 2)
strip_plot(train_df, 'CryoSleep', 'FoodCourt')

plt.subplot(3, 2, 3)
strip_plot(train_df, 'CryoSleep', 'Spa')

plt.subplot(3, 2, 4)
strip_plot(train_df, 'CryoSleep', 'ShoppingMall')

plt.subplot(3, 2, 5)
strip_plot(train_df, 'CryoSleep', 'VRDeck')


# ### Insights
# - As expected, Cryosleeping members don't spend much on services.
# - We can impute *Expenditures* based from *Age* and *CryoSleep*.

# In[ ]:


for dataset in df:
    dataset['IsChild'] = 0
    dataset.loc[dataset.Age <= 12, 'IsChild'] = 1
    dataset.loc[dataset.Age > 12, 'IsChild'] = 0
    
services = ['RoomService', 'FoodCourt', 'Spa', 'ShoppingMall', 'VRDeck']
for s in services: 
    print(train_df.groupby(['IsChild', 'CryoSleep'])[s].median().fillna(0))


# In[ ]:


# Impute Expenditures
for dataset in df:
    dataset['RoomService'] = dataset['RoomService'].fillna(dataset.groupby(['IsChild', 'CryoSleep'])['RoomService'].transform('median'))
    dataset['FoodCourt'] = dataset['FoodCourt'].fillna(dataset.groupby(['IsChild', 'CryoSleep'])['FoodCourt'].transform('median'))
    dataset['Spa'] = dataset['Spa'].fillna(dataset.groupby(['IsChild', 'CryoSleep'])['Spa'].transform('median'))
    dataset['VRDeck'] = dataset['VRDeck'].fillna(dataset.groupby(['IsChild', 'CryoSleep'])['VRDeck'].transform('median'))
    dataset['ShoppingMall'] = dataset['ShoppingMall'].fillna(dataset.groupby(['IsChild', 'CryoSleep'])['ShoppingMall'].transform('median'))


# In[ ]:


missing_values(train_df)


# In[ ]:


# Update Basic, Premium, and All_Services
for dataset in df:
    dataset['Premium'] = dataset['RoomService'] + dataset['Spa'] + dataset['VRDeck']
    dataset['Basic'] = dataset['FoodCourt'] + dataset['ShoppingMall']
    dataset['All_Services'] = dataset['RoomService'] + dataset['Spa'] + dataset ['VRDeck'] + dataset['FoodCourt'] + dataset['ShoppingMall']


# Reference table for imputing *Age*:
# - it appears that there were no VIPs that aboarded from Earth.

# In[ ]:


train_df.groupby(['HomePlanet', 'VIP'])['Age'].median().fillna(0)


# In[ ]:


# Impute Age
for dataset in df:
    dataset.Age = dataset.groupby(['HomePlanet', 'VIP']).Age.apply(lambda x: x.fillna(x.median()))


# In[ ]:


missing_values(train_df)


# ## Transforming Numerical Variables
# The **probability plot** or [**quantile-quantile plot (QQplot)**](https://www.statisticshowto.com/q-q-plots/) allows us to plot our sample data against the quantiles of a normal distribution. In a nutshell, the objective is to have all the points lie along the line in the QQplot.
# 
# Before doing that, let's first fill up our missing numerical values with median.

# In[ ]:


import scipy.stats as stats

# Defining the function to generate the distribution plot alongside QQplot
def QQplot(df, col):
    plt.figure(figsize = (15, 5))
    plt.subplot(1,2,1)
    ax = sns.histplot(x=df[col], kde=True)
    plt.title(str(col),
          fontsize = 18,
          fontweight = 'bold',
          fontfamily = 'serif',
          loc = 'left')
    
    ax.set(xlabel=None, ylabel=None)
    
    plt.subplot(1,2,2)
    stats.probplot(df[col].dropna(), dist="norm", plot=plt)


# I will present the analysis the *Premium, Basic,* and *All_Services* features as working examples for visualizing and interpreting the QQplots. As shown below is the baseline of the their distributions in the form of QQplot.

# In[ ]:


QQplot(train_df, 'Premium')
QQplot(train_df, 'All_Services')


# Afterwards, we can proceed to transform our data and assess its fit in the QQplots once more. Here, I chose to try the following data transformations.
# - square root
# - cube root
# - logarithmic ( log(x+1))

# In[ ]:


_ = train_df[['Premium', 'Transported']]

_["sqrt"] = _['Premium']**(1./2)
_["4rt"] = _['Premium']**(1./4)
_["log"] = np.log(_['Premium']+1)

QQplot(_, 'sqrt')
QQplot(_, '4rt')
QQplot(_, 'log')


# After transformation, it is usually a good practice to check the correlation once more if there are any improvements. As shown below, both 4th root and log transformations improved the correlation from **-0.35 to -0.56.**

# In[ ]:


plt.subplots(figsize=(8,6))
mask = np.triu(np.ones_like(_.corr()))
sns.heatmap(_.corr(), mask=mask, cmap='cool', annot=True, annot_kws={"fontsize":13}, center=0, square=True)


# In[ ]:


_ = train_df[['All_Services', 'Transported']]

_["sqrt"] = _['All_Services']**(1./2)
_["4rt"] = _['All_Services']**(1./4)
_["log"] = np.log(_['All_Services']+1)

QQplot(_, 'sqrt')
QQplot(_, '4rt')
QQplot(_, 'log')


# In[ ]:


plt.subplots(figsize=(8,6))
mask = np.triu(np.ones_like(_.corr()))
sns.heatmap(_.corr(), mask=mask, cmap='cool', annot=True, annot_kws={"fontsize":13}, center=0, square=True)


# Creating *Spent*.

# In[ ]:


for dataset in df:
    dataset['Spent'] = 0
    dataset.loc[dataset['All_Services'] > 0, 'Spent'] = 1


# # Categorical Features
# 
# Before we visualize them, let's fill up our missing values in each feature with their corresponding **mode**, which is the most common label in the existing feature.

# In[ ]:


def count_plot(df, x, y):
    plt.subplots(1,2, figsize = (15, 5))
    plt.subplot(1,2,1)
    ax = sns.countplot( x=df[x].dropna(), hue=df[y])
    plt.title(str(x),
          fontsize = 18,
          fontweight = 'bold',
          fontfamily = 'serif',
          loc = 'left')
    ax.set(xlabel=None, ylabel=None)
    
    plt.subplot(1,2,2)
    plt.ylim(0,1)
    ax = sns.lineplot( x=df[x], y=df[y], data=df, ci=None, linewidth=3, marker="o")
    ax.set(xlabel=None, ylabel=None)
    plt.show()


count_plot(train_df, 'HomePlanet', 'Transported')
count_plot(train_df, 'CryoSleep', 'Transported')
count_plot(train_df, 'Destination', 'Transported')
count_plot(train_df, 'VIP', 'Transported')
count_plot(train_df, 'IsChild', 'Transported')
count_plot(train_df, 'Spent', 'Transported')


# # Mixed/Alphanumeric Features
# Let's extract relevant information from 'PassengerId', 'Cabin', and 'Name'
# - **PassengerId** = gggg_pp (gggg=group, pp=number within group)
# - **Cabin** = deck/num/side (side: P=Port, S=Starboard)
# - **Name** = First Name + Last Name

# In[ ]:


# Splitting PassengerId into 'Group' and 'GroupSize'

for dataset in df:
    dataset['Group'] = dataset['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
    dataset['GroupSize'] = dataset['Group'].map(lambda x: dataset['Group'].value_counts()[x])
    
    dataset['withGroup'] = 1
    dataset.loc[dataset['GroupSize'] == 1, 'withGroup'] = 0

count_plot(train_df, 'GroupSize', 'Transported')
count_plot(train_df, 'withGroup', 'Transported')


# It is more likely that larger group sizes have children with them because they are family. Let's try exploring that.

# In[ ]:


count_plot(train_df, 'GroupSize', 'IsChild')


# **GroupSize**
# - Large number of passengers traveled alone, the volume exponentially decreases with larger group sizes.
# - GroupSize of 4 followed by 6 had the highest transport rates.
# - GroupSize of 8 followed by alone passengers had the lowest transport rates.
# - As expected, group sizes (>1) tend to have children with them, but with the exception of 7, and 8. This likely explains the transport rate pattern in *GroupSize*.

# In[ ]:


# Splitting Cabin into Deck, Num, and Side

for dataset in df:
    dataset['Deck'] = dataset['Cabin'].apply(lambda x: x.split('/')[0] if (str(x)) != 'nan' else x)
    dataset['Num'] = dataset['Cabin'].apply(lambda x: x.split('/')[1] if (str(x)) != 'nan' else x)
    dataset['Side'] = dataset['Cabin'].apply(lambda x: x.split('/')[2] if (str(x)) != 'nan' else x)

count_plot(train_df, 'Deck', 'Transported')
count_plot(train_df, 'Side', 'Transported')


# **Deck**
# - Majority of passengers resided in Cabins F and G.
# - Highest transport rate among Cabin B passengers, followed by those of Cabin C.
# - Lowest transort rate among Cabin T passengers, but data is not representative enough.
# 
# **Side**
# - Side S dominates transport rate by a small margin over Side P.

# In[ ]:


# Splitting Name into First and Last Names
for dataset in df:
    dataset['FirstName'] = dataset['Name'].apply(lambda x: x.split(' ')[0] if (str(x)) != 'nan' else x)
    dataset['LastName'] = dataset['Name'].apply(lambda x: x.split(' ')[1] if (str(x)) != 'nan' else x)
    
    # Creating 'FamilySize' from 'LastName'
    dataset['FamilySize'] = dataset['LastName'].map(lambda x: dataset['LastName'].value_counts()[x] if (str(x)) != 'nan' else x)

    
def count_plot_adj(df, x, y):
    plt.subplots(1,2, figsize = (15, 5))
    plt.subplot(1,2,1)
    ax = sns.countplot( x=df[x].dropna(), hue=df[y])
    plt.title(str(x),
          fontsize = 18,
          fontweight = 'bold',
          fontfamily = 'serif',
          loc = 'left')
    ax.set(xlabel=None, ylabel=None)
    
    plt.subplot(1,2,2)
    plt.ylim(0,1)
    ax = sns.lineplot( x=df[x], y=df[y], data=df, ci=None, linewidth=3, marker="o")
    ax.set(xlabel=None, ylabel=None, xlim = (1, 18))
    
    plt.show()
    
count_plot_adj(train_df, 'FamilySize', 'Transported')


# In[ ]:


count_plot_adj(train_df, 'FamilySize', 'IsChild')


# **FamilySize**
# - *FamilySize* of 3, 4, and 5 comprise the majority.
# - Transport rate appears to steadily decline with increasing FamilySize, but with the exception of 15.
# - Similar pattern was observed in the increasing distribution of children with larger *FamilySize*, although there seems to be exceptions as it goes greater than 10.

# ## Associating Categorical Variables
# 
# In terms of statistics terminologies, it is not usually appropriate to use the term *correlation* when testing with categorical variables. We can't really assess the magnitude or the strength of correlation between predictor and response categorical variables, unless if they are either dichotomous (categorical variables having 2 categories like 'Sex') or ordinal variables then it is allowed to use **Pearson's correlation.**
# > This means that we can still evaluate *CryoSleep, VIP* against our target *Transported* with Pearson's correlation.
# 
# The rest of our categorical features are either non-ordinal or contain more than 2 categories. Instead of conducting correlation tests, it is more appropriate to use tests of independency to assess the strength of association between cateorical variables. The one I will use is Cramer's V which is based from **[Chi-square test](https://towardsdatascience.com/chi-square-test-for-feature-selection-in-machine-learning-206b1f0b8223#:~:text=In%20feature%20selection%2C%20we%20aim,hypothesis%20of%20independence%20is%20incorrect.).**
# 
# Before we can conduct Chi-square tests, we must ensure that our categorical data are numerically encoded first using `LabelEncoder()`.

# In[ ]:


# Encoding categorical labels
from sklearn.preprocessing import LabelEncoder
label_encode = LabelEncoder()

df1 = train_df.copy()

df_cat = df1[['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'IsChild', 'Spent', 'Group', 'withGroup', 'Deck', 'Num', 'Side', 'LastName', 'Transported']]

label = LabelEncoder()
df_cat_encoded = pd.DataFrame()

for i in df_cat.columns:
    df_cat_encoded[i] = label.fit_transform(df_cat[i])
    
df_cat_encoded.head()


# In[ ]:


from scipy.stats.contingency import association       
    
def Cramers_V(var1, var2) :
  crosstab = np.array(pd.crosstab(index=var1, columns=var2)) # Cross Tab
  return (association(crosstab, method='cramer'))            # Return Cramer's V

# Create the dataFrame matrix with the returned Cramer's V
rows = []

for var1 in df_cat_encoded:
    col = []

    for var2 in df_cat_encoded:
        V = Cramers_V(df_cat_encoded[var1], df_cat_encoded[var2]) # Return Cramer's V
        col.append(V)                                             # Store values to subsequent columns  
    rows.append(col)                                              # Store values to subsequent rows
  
CramersV_results = np.array(rows)
CramersV_df = pd.DataFrame(CramersV_results, columns = df_cat_encoded.columns, index = df_cat_encoded.columns)


# In[ ]:


plt.subplots(figsize=(20,15))
corr = np.corrcoef(np.random.randn(13, 13))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(CramersV_df, mask=mask, cmap='cool', annot=True, annot_kws={"fontsize":13}, center=0, square=True, cbar=False)


# **Observations.** There is a lot to dig in here because my approach to feature engineering was rather exhaustive. As a result, multicollinearity tend to occur, this happens when two or more independent variables are highly associated with one another, rendering them redundant.

# ## Missing Categorical Values
# 
# By answering these questions, we also stand to gain relevant insights and patterns which we can apply to our imputation method. Guess what, since we created our association matrix above, we already know which features and other correspoding interrelated features to look out for. Below are just some questions that we are particularly interested in.
# 
# - **PassengerId group vs Name** (last name). Assuming they are family, it is more probable that most have similar last names.
# - **PassengerId group** (gggg=group, pp=number) **vs HomePlanet and Destination.** Did passengers within their groups travel together, which means coming from the same HomePlanet and debarking to the same Destination.
# - **PassengerId group vs Cabin**. Do people of the same group stay in the same cabin?
# - **Cryosleep vs Cabin.** Does a cryosleep facility have its designated cabin area?
# - **CryoSleep vs Services.** Did passengers who elected to cryosleep have lower expenditures?
# - **CryoSleep vs withGroup.** Were those who traveled alone in the group likely to undergo CryoSleep?
# - **CryoSleep vs HomePlanet and Destination.** Is there a pattern among passengers who cryoslept with respect to longer travels (can possibly infer that distances are farther between HomePlanet vs Destination)

# In[ ]:


# Define function to impute based on a feature
def impute_cat(var1, var2):
    print('Before %s Train:' %var2, train_df[var2].isnull().sum())
    print('Before %s Test:' %var2, test_df[var2].isnull().sum())

    test_df['Transported'] = np.NaN
    df_full = pd.concat([train_df, test_df])

    reference = df_full.groupby([var1, var2])[var2].size().unstack().fillna(0)

    for dataset in df:          
        index = dataset[dataset[var2].isnull()][(dataset.loc[dataset[var2].isnull()][var1]).isin(reference.index)].index
        dataset.loc[index, var2] = dataset.loc[index, var1].map(lambda x: reference.idxmax(axis=1)[x])
    
    print('After %s Train:' %var2, train_df[var2].isnull().sum())
    print('After %s Test:' %var2, test_df[var2].isnull().sum())
    print('\n')


# ### Imputing CryoSleep

# Since Cryosleep is strongly associted with Spent, we will impute Cryosleep as False if Spent is 1, then True otherwise.

# In[ ]:


print('Before Train:', train_df['CryoSleep'].isnull().sum())
print('Before Test:', test_df['CryoSleep'].isnull().sum())

for dataset in df:
    dataset.loc[(dataset.CryoSleep.isnull()) & (dataset.Spent == 0), 'CryoSleep' ] = True
    dataset.loc[(dataset.CryoSleep.isnull()) & (dataset.Spent == 1), 'CryoSleep' ] = False


print('After Train:', train_df['CryoSleep'].isnull().sum())
print('After Test:', test_df['CryoSleep'].isnull().sum())


# In[ ]:


# Impute Remaining Expenditures from CryoSleep
for dataset in df:
    dataset['RoomService'] = dataset['RoomService'].fillna(dataset.groupby(['IsChild', 'CryoSleep'])['RoomService'].transform('median'))
    dataset['FoodCourt'] = dataset['FoodCourt'].fillna(dataset.groupby(['IsChild', 'CryoSleep'])['FoodCourt'].transform('median'))
    dataset['Spa'] = dataset['Spa'].fillna(dataset.groupby(['IsChild', 'CryoSleep'])['Spa'].transform('median'))
    dataset['VRDeck'] = dataset['VRDeck'].fillna(dataset.groupby(['IsChild', 'CryoSleep'])['VRDeck'].transform('median'))
    dataset['ShoppingMall'] = dataset['ShoppingMall'].fillna(dataset.groupby(['IsChild', 'CryoSleep'])['ShoppingMall'].transform('median'))
    
    # Update Basic, Premium, and All_Services
    dataset['Premium'] = dataset['RoomService'] + dataset['Spa'] + dataset['VRDeck']
    dataset['Basic'] = dataset['FoodCourt'] + dataset['ShoppingMall']
    dataset['All_Services'] = dataset['RoomService'] + dataset['Spa'] + dataset ['VRDeck'] + dataset['FoodCourt'] + dataset['ShoppingMall']
    
    # Update Spent
    dataset['Spent'] = 0
    dataset.loc[dataset['All_Services'] > 0, 'Spent'] = 1


# ### Imputing VIP

# In[ ]:


# How VIP spend on services
plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
sns.violinplot(data=train_df, x="VIP", y='Premium')


# **Observation.** It appears that VIP passengers tend to spend more on premium services. For now, I will impute by mode.

# In[ ]:


print('Before Train:', train_df['VIP'].isnull().sum())
print('Before Test:', test_df['VIP'].isnull().sum())

for dataset in df:
    dataset['VIP'].fillna(False, inplace=True)
    
print('After Train:', train_df['VIP'].isnull().sum())
print('After Test:', test_df['VIP'].isnull().sum())


# ### Imputing Cabin Deck and Side

# In[ ]:


CD_PG = train_df.groupby(['Group', 'Deck'])['Deck'].size().unstack().fillna(0)
CD_PG.head(10)


# **Conclusion.** We discovered that passengers of the same group stay in the same cabin deck.

# In[ ]:


# Imputing Deck
impute_cat('Group', 'Deck')
impute_cat('Group', 'Side')


# **Decision.** Impute *Deck* based on 2nd most highly associated feature *LastName*. Navigate [here](#impute-deck-lastname).

# ### Imputing Homeplanet and Destination

# In[ ]:


HP_PG = train_df.groupby(['Group', 'HomePlanet'])['HomePlanet'].size().unstack().fillna(0)
HP_PG.head(10)


# In[ ]:


D_PG = train_df.groupby(['Group', 'Destination'])['Destination'].size().unstack().fillna(0)
D_PG.head(10)


# **Conclusion.** The passengers within each group have the same HomePlanet and Destinations. Given this premise, we can impute missing values in HomePlanet by returning the column name (Earth, Europa, Mars) with the positive number of passengers, as a function of 'Group'. The same imputation process can be done for missing values in 'Destination'.

# In[ ]:


# Impute HomePlanet and Destination
impute_cat('Group', 'HomePlanet')
impute_cat('Group', 'Destination')


# There are still remaining missing values that weren't filled, so my strategy now is to  impute them based on the succeeding features that are highly associated to *HomePlanet* and *Destination.*

# **Decision.** Impute *HomePlanet* and *Destination* based from 2nd most associated feature *LastName*

# In[ ]:


LN_HP = train_df.groupby(['LastName', 'HomePlanet'])['HomePlanet'].size().unstack().fillna(0)
LN_HP.head(10)


# In[ ]:


impute_cat('LastName', 'HomePlanet')
impute_cat('LastName', 'Destination')


# **Decision.** Impute *HomePlanet* and *Destination* based from 3rd most associated feature *Deck*

# In[ ]:


CD_HP = train_df.groupby(['Deck', 'HomePlanet'])['HomePlanet'].size().unstack().fillna(0)
CD_HP.head(10)


# In[ ]:


impute_cat('Deck', 'HomePlanet')
impute_cat('Deck', 'Destination')


# ### Analyzing Cabin Deck, Side vs HomePlanet and Destination

# In[ ]:


HP_D_CS_CD = train_df.groupby(['HomePlanet', 'Destination', 'Spent', 'Deck'])['Deck'].size().unstack().fillna(0)
HP_D_CS_CD.head(20)


# In[ ]:


plt.subplots(figsize=(15,18))
sns.heatmap(HP_D_CS_CD, cmap='cool', annot=True, annot_kws={"fontsize":13}, fmt='g', center=0, square=True)


# **Conclusion.** Out of all the comparisons, HomePlanet-Destination-Spent vs CabinDeck yielded the best patterns.
# - Cabins E, F, and G are mostly reserved by passengers that embarked from Earth.
# - Cabins A, B, C, D, and E resereved by those that embarked from Europa.
# - Cabins D, E, and F resereved by those that embarked from Mars.
# 
# If you are interested, you may check the proposed solution in Stackoverflow [HERE](https://stackoverflow.com/questions/45741879/can-i-replace-nans-with-the-mode-of-a-column-in-a-grouped-data-frame) if you want to proceed the imputation. No matter how I editted and approached, I wasn't able to make it work : (

# ### Imputing Surname and Family Size
# - Passengers of the same group are likely to be families (having the same surnames)
# - The purpose of filling up *LastName* is to update the missing data in *FamilySize* later on.

# In[ ]:


PG_SN = train_df.groupby(['Group', 'LastName'])['LastName'].size().fillna(0)
PG_SN.head(20)


# **Decision.** It appears that most, but not all the passengers in the group have the same surnames, as in the case of Group 20, so we can just impute as the *LastName* with the highest occurences.

# In[ ]:


impute_cat('Group', 'LastName')


# **Decision.** Impute *LastName* based on 2nd most highly associated feature *HomePlanet*

# In[ ]:


impute_cat('HomePlanet', 'LastName')


# In[ ]:


# Update the 'FamilySize' column
for dataset in df:
    dataset['FamilySize'] = dataset['LastName'].map(lambda x: dataset['LastName'].value_counts()[x] if (str(x)) != 'nan' else x)


# <a id="impute-deck-lastname"></a>
# **Decision.** Impute remaining missing *Deck* and *Side* based on *LastName*, followed by *HomePlanet*.

# In[ ]:


impute_cat('LastName', 'Deck')
impute_cat('HomePlanet', 'Deck')

impute_cat('LastName', 'Side')
impute_cat('HomePlanet', 'Side')


# We want finalize if our missing data have all been filled before we proceed to preprocessing.

# In[ ]:


missing_values(train_df)


# In[ ]:


missing_values(test_df)


# # Data Preprocessing

# Now, I will be dropping features that are highly cardinal, weakly correlated/associated with the target variable, and multicollinearity-inducing.

# In[ ]:


# Change VIP data type from bool to object
for dataset in df:
    dataset['VIP'] = dataset['VIP'].astype(object)
    
    # 4th Root transform Premium
    dataset['Premium'] = dataset['Premium']**(1./4)
    dataset['All_Services'] = dataset['All_Services']**(1./4)
    dataset['Basic'] = dataset['Basic']**(1./4)


# In[ ]:


test_df.drop('Transported', axis=1, inplace=True)


# In[ ]:


y_train = train_df['Transported']
X_train = train_df[['CryoSleep', 'Premium', 'Basic', 'IsChild', 'withGroup', 'HomePlanet', 'Destination', 'Deck', 'Side']]

X_test = test_df[['CryoSleep', 'Premium', 'Basic', 'IsChild', 'withGroup', 'HomePlanet', 'Destination', 'Deck', 'Side']]

X = [X_train, X_test]


# In[ ]:


from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

# Label encode categorical variables
for dataset in X:
    dataset['Side'] = label.fit_transform(dataset['Side'])
    dataset['CryoSleep'] = label.fit_transform(dataset['CryoSleep'])

# Scale num features
X_train[['Premium']] = scale.fit_transform(X_train[['Premium']])
X_test[['Premium']] = scale.transform(X_test[['Premium']])

X_train[['Basic']] = scale.fit_transform(X_train[['Basic']])
X_test[['Basic']] = scale.transform(X_test[['Basic']])


# In[ ]:


# Creating dummy indicator columns for categorical variables
X_train = pd.get_dummies(X_train, columns=['HomePlanet', 'Destination', 'Deck'])
X_test = pd.get_dummies(X_test, columns=['HomePlanet', 'Destination', 'Deck'])


# In[ ]:


X_train


# # Model Training
# I will run my model through some common supervised model algorithms.
# - Logistic Regression
# - Support Vector Classifier
# - Decision Tree
# - Random Forest
# - XGBoost

# In[ ]:


from sklearn.model_selection import cross_val_score

#Common Model Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Defining a list of Machine Learning Algorithms I will be running
MLA = [
    LogisticRegression(max_iter = 2000),
    SVC(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    XGBClassifier(),
    LGBMClassifier(),
#     CatBoostClassifier(verbose=False)
]

row_index = 0

# Setting up the table to compare the performances of each model
MLA_cols = ['Model', 'Accuracy']
MLA_compare = pd.DataFrame(columns = MLA_cols)

# Iterate and store scores in the table
for model in MLA:
    MLA_compare.loc[row_index, 'Model'] = model.__class__.__name__
    cv_results = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
    MLA_compare.loc[row_index, 'Accuracy'] = cv_results.mean()
    
    row_index+=1

# Present table
MLA_compare.sort_values(by=['Accuracy'], ascending=False, inplace=True)
MLA_compare


# **Conclusion.** Among the model algorithms with default parameters, the top 3 best perorming classifiers turned out to be LGBM, followed by SVC, followed by XGB.

# # Hyperparameters Tuning
# The top 3 best performing model algorithms will be chosen as candidates for hyperparameter tuning. I already ran the code below for hypertuning parameters and found the optimal parameters as listed below, so the code was commented out in the interest of time. 

# In[ ]:


import optuna
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

# Hypertune XGBoost
def objective(trial, data=X_train , target=y_train):

    param = {
        'n_estimators': 1000,
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_categorical('max_depth', [3, 4, 5, 6, 7, 8, 9, 10]),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 10),
        'subsample': trial.suggest_categorical('subsample', [0.5, 0.6, 0.7, 0.8, 0.9])
    }    
    
    cv = StratifiedKFold( n_splits=5, shuffle=True, random_state=42)
    
    for idx, (train_idx, test_idx) in enumerate(cv.split(X_train, y_train)):
        
        x_trn, x_val = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_trn, y_val = y_train.iloc[train_idx], y_train.iloc[test_idx]
    
    
        model = XGBClassifier(**param)
        
        model.fit(x_trn,
                  y_trn,
                  eval_set = [(x_val, y_val)],
                  early_stopping_rounds = 100,
                  eval_metric = 'logloss',
                  verbose = False
                 )
    
        preds = model.predict(x_val)
        scores = accuracy_score(y_val, preds)
        
    return np.mean(scores)


# In[ ]:


# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=10)

# print(f"\tBest score: {study.best_value:.5f}")
# print(f"\tBest params:", study.best_trial.params)


# In[ ]:


# Hypertune LGBM
from optuna.integration import LightGBMPruningCallback
from lightgbm import early_stopping

def objective(trial, data=X_train , target=y_train):

    param = {
        'n_estimators': 5000,
        'num_leaves': trial.suggest_int('num_leaves', 20, 3000, step=20),
#         'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 200, 10000, step=100),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 15),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10),
        'subsample': trial.suggest_categorical('subsample', [0.5, 0.6, 0.7, 0.8, 0.9])
    }    
    
    cv = StratifiedKFold( n_splits=5, shuffle=True, random_state=42)
    
    for idx, (train_idx, test_idx) in enumerate(cv.split(X_train, y_train)):
        
        x_trn, x_val = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_trn, y_val = y_train.iloc[train_idx], y_train.iloc[test_idx]

    
        model = LGBMClassifier(**param)
        
        model.fit(x_trn,
                  y_trn,
                  eval_set = [(x_val, y_val)],
                  eval_metric = 'logloss',
                  callbacks = [early_stopping(stopping_rounds=100,
                                              verbose = False)],
                  verbose = False
                 )
    
        preds = model.predict(x_val)
        scores = accuracy_score(y_val, preds)
        
    return np.mean(scores)


# In[ ]:


# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=30)

# print(f"\tBest score: {study.best_value:.5f}")
# print(f"\tBest params:", study.best_trial.params)


# In[ ]:


# Tuned models
xgb_params = {'learning_rate': 0.03824797490742583, 'max_depth': 10, 'min_child_weight': 9, 'subsample': 0.5}
lgbm_params = {'num_leaves': 940, 'learning_rate': 0.02496339867162098, 'max_depth': 5, 'min_child_samples': 7, 'reg_alpha': 1.1993698456266415, 'reg_lambda': 0.0012967298146930883, 'subsample': 0.6}


# # Ensemble Model

# In[ ]:


from sklearn.ensemble import VotingClassifier

xgb_optimal = XGBClassifier(**xgb_params)
lgbm_optimal = LGBMClassifier(**lgbm_params)

# Create Hard Voting Classifier
Ensemble_HV = VotingClassifier(estimators= [('SVC', SVC()),
                                           ('XBG', xgb_optimal),
                                           ('LGBM', lgbm_optimal)],
                              voting = 'hard')

# Create Soft Voting Classifier
Ensemble_SV = VotingClassifier(estimators= [('SVC', SVC()),
                                           ('XBG', xgb_optimal),
                                           ('LGBM', lgbm_optimal)],
                              voting = 'soft')

# Return Accuracy Scores
cv_HV = cross_val_score(Ensemble_HV, X_train, y_train, scoring='accuracy')
cv_SV = cross_val_score(Ensemble_SV, X_train, y_train, scoring='accuracy')

print('Hard Voting Classifier:' , cv_HV.mean())
print('Soft Voting Classifier:' , cv_SV.mean())


# # Submission

# In[ ]:


# Defining a function to predict solutions
def predict(model):
    model.fit(X_train, y_train)
    Y_pred = model.predict(X_test)
    pred = pd.DataFrame({
    'PassengerId': test_df_copy['PassengerId'],
    'Transported': Y_pred
})
    return pred


# In[ ]:


xgb = XGBClassifier()
lgbm = LGBMClassifier()
cat = CatBoostClassifier()


# In[ ]:


predict(lgbm).to_csv('submission_lgbm.csv', index=False)
predict(xgb).to_csv('submission_xgb.csv', index=False)
predict(lgbm_optimal).to_csv('submission_lgbm_optimal.csv', index=False)
predict(xgb_optimal).to_csv('submission_xgb_optimal.csv', index=False)
predict(Ensemble_HV).to_csv('submission_Ensemble_HV.csv', index=False)
predict(Ensemble_SV).to_csv('submission_Ensemble_SV.csv', index=False)


# ## My Other Works
# Check out my KAGGLE profile and my other contributions
# - [Attack-on-Titanic Solution Walkthrough | Kaggle](https://www.kaggle.com/code/shilongzhuang/attack-on-titanic-solution-walkthrough-top-8)
# - [Plotly Advanced Charts: EDA on Unicorn Startups](https://www.kaggle.com/code/shilongzhuang/plotly-advanced-charts-eda-on-unicorn-startups)

# ---
# # References
# 
# Special thanks and credits to these awesome and comprehensively informative resources (notebooks) and guides created by talented professionals in the field, which were able to guide me while creating this work.
# 
# - [A Data Science Framework: To achieve 99% Accuracy](https://www.kaggle.com/code/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy)
# - [Spaceship Titanic: A complete guide](https://www.kaggle.com/code/samuelcortinhas/spaceship-titanic-a-complete-guide)
