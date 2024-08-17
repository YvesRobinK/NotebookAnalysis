#!/usr/bin/env python
# coding: utf-8

# # Data Analysis and Feature engineering for House Price modelling
# 
# In this notebook I mix exploratory data analysis, to understand the nature of the data with feature engineering and feature selection. I make heavy use of the open source Python library [Feature-engine](https://feature-engine.readthedocs.io/en/latest/), because it removes quite a lot of coding from my hands and also.
# 
# [Feature-engine](https://feature-engine.readthedocs.io/en/latest/)'s transformers come along with fit and transform funcionality, to learn the necessary parameters from the data, and then transform the data accordinly. So it is very easy to learn and use. And it can also detect automatically the nature of the variables, or engineer a selected group of variables if we so desire.
# 
# 
# ## Table of contents
# 
# **0. Feature Creation**
# 
#     0.1 Binary indicators
#     
#     0.2 Variable Aggregations    
#     
# 
# **1. Categorical variables**
# 
#     1.1 Missing data
# 
#     1.2 Cardinality
#     
#     1.3 Rare Labels
#     
#     1.4 Label grouping
#     
#     1.5 Variable selection
#     
#     1.6 Encoding
#     
#     
# **2. Time variables**
# 
#     2.1 Time elapsed calculation
#     
#     2.2 Missing Data
#     
#     
# **3. Discrete variables**
# 
#     3.1 Exploratory data analysis
#     
#     3.2 Missing Data
#     
#     
# **4. Numerical variables**
# 
#     4.1 Distributions
#     
#     4.2 Relation with house sale price
#     
#     4.3 Missing Data
#     
#     
# **5. End to end pipeline with cross validation**
# 
# 
# This notebook is based on the following resources:
# 
# - [Feature Engineering for Machine Learning](https://www.courses.trainindata.com/p/feature-engineering-for-machine-learning), online course.
# - [Feature Selection for Machine Learning](https://www.courses.trainindata.com/p/feature-selection-for-machine-learning), online course.
# - [Packt Feature Engineering Cookbook](https://www.packtpub.com/data/python-feature-engineering-cookbook)
# - [Feature-engine](https://feature-engine.readthedocs.io/en/latest/), Python open source library
# 
# ## If you find this notebook useful, I will appreciate if you could upvote it :)

# In[1]:


# let's install Feature-engine

get_ipython().system('pip install feature_engine')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from lightgbm import LGBMRegressor

# Feature-engine's modules for feature egineering
from feature_engine import creation
from feature_engine import discretisation as disc
from feature_engine import encoding as enc
from feature_engine import imputation as imp
from feature_engine import selection as sel


# In[3]:


# load training data
data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

# load data for competition submission
# this data does not have the target
submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[4]:


# split training data into train and test

X_train, X_test, y_train, y_test = train_test_split(data.drop(
    ['Id', 'SalePrice'], axis=1),
    data['SalePrice'],
    test_size=0.05,
    random_state=0)

X_train.shape, X_test.shape


# In[5]:


# drop id
id_ = submission['Id']

submission.drop('Id', axis=1, inplace=True)

submission.shape


# In[6]:


# function to create a dataset with all the train, test and subimssion, 
# so I can easily compare the variable distributions

def create_master_data(train, test, submission, y_train, y_test):
    
    train = train.copy()
    test = test.copy()
    submission = submission.copy()
    
    train['target'] = y_train
    train['data'] = 'train'
    train.reset_index(drop=True, inplace=True)
    
    test['target'] = y_test
    test['data'] = 'test'
    test.reset_index(drop=True, inplace=True)
    
    submission['target'] = np.nan
    submission['data'] = 'submission'
    submission.reset_index(drop=True, inplace=True)
    
    master_data = pd.concat([train, test, submission], axis=0)
    master_data.reset_index(drop=True, inplace=True)
    
    return master_data


# ## Target
# 
# Let's explore the target variable.

# In[7]:


# histogran to evaluate target distribution

y_train.hist(bins=50, density=True)
y_test.hist(bins=50, density=True)

plt.legend(["Train", "Test"])
plt.ylabel('Number of houses')
plt.xlabel('Sale Price')
plt.show()


# As expected, the 2 distributions are very similar, because the datasets come from a random division of the original data.

# In[8]:


# let's transform the target, we are optimising for the log 
# of the target as per competition rules

np.log(y_train).hist(bins=50, density=True)
np.log(y_test).hist(bins=50, density=True)

plt.ylabel('Number of houses')
plt.xlabel('Log of Sale Price')
plt.show()


# In[9]:


# density plot instead of histogram
np.log(y_train).plot.density()
np.log(y_test).plot.density()

plt.xlabel('Log of Sale Price')
plt.show()


# In[10]:


# let's transform the target with the log

y_train = np.log(y_train)

y_test = np.log(y_test)


# ## Feature Creation
# 
# In this section, I create new variables, from those already in the dataset.

# In[11]:


# if the house has a pool

X_train['HasPool'] = np.where(X_train['PoolArea']>0,1,0)
X_test['HasPool'] = np.where(X_test['PoolArea']>0,1,0)
submission['HasPool'] = np.where(submission['PoolArea']>0,1,0)


# if the house has an Alley

X_train['HasAlley'] = np.where(X_train['Alley'].isnull(), 0, 1)
X_test['HasAlley'] = np.where(X_test['Alley'].isnull(), 0, 1)
submission['HasAlley'] = np.where(submission['Alley'].isnull(), 0, 1)


# In[12]:


# create function to plot the median sale price vs a list of variables

def plot_median_price(variables, limits):
    
    tmp = pd.concat([X_train[variables].reset_index(drop=True),
                 y_train.reset_index(drop=True)], axis=1)

    # plot mean sale price per null value or otherwise
    g = sns.PairGrid(tmp, x_vars=variables, y_vars=['SalePrice'])
    g.map(sns.barplot)
    plt.ylim(limits)   
    plt.show()


# In[13]:


# plot the mean house sale price for the created variables
# to see if they add some value

plot_median_price(['HasAlley', 'HasPool'], (11.5,12.5))


# Not clear if these variables add a lot of value. The difference in price is small, but then the price is in log scale.

# In[14]:


# The variables Condition1 and Condition2 indicate Proximity to various conditions. 

# Create a variable that aggregates both to indicate if the house is
# close to more than 1 ammenity. If the house is close to more than 1
# ammenity, the conditions take different values.

X_train['Condition_total'] = np.where(X_train['Condition2'] == X_train['Condition1'], 0, 1)
X_test['Condition_total'] = np.where(X_test['Condition2'] == X_test['Condition1'], 0, 1)
submission['Condition_total'] = np.where(submission['Condition2'] == submission['Condition1'], 0, 1)

# inspect the variable
X_train['Condition_total'].value_counts()


# Roughly 10% of the houses are proximal to 2 different ammenities.

# In[15]:


# Exterior 1s and 2nd indicate the Exterior material covering on house:
# Create a variable that aggregates both, like we did for contition

X_train['Exterior_total'] = np.where(X_train['Exterior1st'] == X_train['Exterior2nd'], 0, 1)
X_test['Exterior_total'] = np.where(X_test['Exterior1st'] == X_test['Exterior2nd'], 0, 1)
submission['Exterior_total'] = np.where(submission['Exterior1st'] == submission['Exterior2nd'], 0, 1)

# inspect the variable
X_train['Exterior_total'].value_counts()


# Roughly 20% of the houses ashow 2 materials in the exterior.

# In[16]:


# plot the mean house sale price for the created variables

plot_median_price(['Condition_total', 'Exterior_total'], (11.5,12.5))


# Again, it is unclear if these new variables add a lot of value. The difference in price seems minimal.

# ## Categorical Variables

# In[17]:


# let's identify the categorical variables

categorical = [var for var in data.columns if data[var].dtype == 'O']

# MSSubClass is also categorical by definition, despite its numeric values
categorical = categorical + ['MSSubClass']

# number of categorical variables
len(categorical)


# In[18]:


# cast all variables as categorical

X_train[categorical] = X_train[categorical].astype('O')
X_test[categorical] = X_test[categorical].astype('O')
submission[categorical] = submission[categorical].astype('O')


# ### Missing Data

# In[19]:


# which categorical variables have missing data?

# capture categorical variables with NA in a dictionary
null_cat = {var: data[var].isnull().mean() for var in categorical if data[var].isnull().mean()>0}

# plot
pd.Series(null_cat).sort_values().plot.bar(figsize=(10,4))
plt.ylabel('Percentage of missing data')
plt.axhline(y = 0.90, color = 'r', linestyle = '-') 
plt.axhline(y = 0.80, color = 'g', linestyle = '-') 

plt.show()


# There are a few variables with a lot of data missing, and a few variables with few missing data. In particular, there are 3 variables for which > 90% of the values are missing (above red line)
# 
# Let's examine if the variables with NA are somewhat informative of the house sale price.

# In[20]:


# create a temporary dataset with the interest variables

tmp = pd.concat([X_train[['Alley', 'MiscFeature', 'PoolQC']].reset_index(drop=True),
                 y_train.reset_index(drop=True)], axis=1)

# replace null values by 1, or 0 otherwise
for var in ['Alley', 'MiscFeature', 'PoolQC']:
    tmp[var] = np.where(tmp[var].isnull(),1,0)


# plot mean sale price per null value or otherwise
g = sns.PairGrid(tmp, x_vars=['Alley', 'MiscFeature', 'PoolQC'], y_vars=['SalePrice'])
g.map(sns.barplot)
plt.ylim(10,13)   
plt.show()


# They don't seem to be hughly predictive so I will drop them (I already captured 2 of them in a binary feature anyways, at the beginning of the notebook).
# 
# [DropFeatures](https://feature-engine.readthedocs.io/en/latest/selection/DropFeatures.html)

# In[21]:


# DropFeatures allows me to drop selected feature groups from data

drop_features = sel.DropFeatures(features_to_drop = ['Alley', 'MiscFeature', 'PoolQC'])

X_train = drop_features.fit_transform(X_train)
X_test = drop_features.transform(X_test)
submission = drop_features.transform(submission)

X_train.shape, X_test.shape, submission.shape


# Let's impute the missing data in categorical variables.
# 
# [CategoricalImputer](https://feature-engine.readthedocs.io/en/latest/imputation/CategoricalImputer.html)

# In[22]:


# impute missing data, categorical variables are detected automatically
# the imputer replaces missing data with the string 'Missing'

cat_imputer = imp.CategoricalImputer(return_object=True)

cat_imputer.fit(X_train)

# the variables to impute are stored in the variables attribute
cat_imputer.variables_[0:10]


# In[23]:


# the number of categorical variables detected by Feature-engine

len(cat_imputer.variables_)


# In[24]:


# the imputer will add a string 'Missing' to each categorical variable

cat_imputer.imputer_dict_


# In[25]:


# remove missing data

X_train = cat_imputer.transform(X_train)
X_test = cat_imputer.transform(X_test)
submission = cat_imputer.transform(submission)

X_train.shape, X_test.shape, submission.shape


# In[26]:


# check that we do not have more missing data in categorical variables
# if we do, the list should not be empty

[c for c in cat_imputer.variables_ if X_train[c].isnull().sum()>0]


# ## Quasi-constant variables
# 
# Let's inspect if there are some variables that show predominantly 1 value  in all observations. We can find them automatically with the folliwing class from Feature-engine:
# 
# [DropConstantFeatures](https://feature-engine.readthedocs.io/en/latest/selection/DropConstantFeatures.html)

# In[27]:


# assign the categorical variable list to the categorical variable name

categorical = cat_imputer.variables_

# we still have 41 categorical variables in the dataset
len(categorical)


# In[28]:


# I ask the transformer to remove all variables that show the same value in more than
# 94% of the observations (tol=0.94)
constant = sel.DropConstantFeatures(tol=0.94, variables=categorical)

# find constant features
constant.fit(X_train)

# the quasi-constant features are stored in this attribute
constant.features_to_drop_


# In[29]:


# put data together for analysis

data = create_master_data(X_train, X_test, submission, y_train, y_test)

data.shape


# In[30]:


# plot number of observations per category, per variable
# so that we see that these variables show mostly 1 value in most of the 
# observations

for variable in constant.features_to_drop_:
    
    data.groupby(variable)['data'].value_counts().unstack().plot.bar(figsize=(10,5))
    plt.title(variable)
    plt.ylabel('Number of houses')
    plt.show()


# We see that in most of these variables, most observations show the same value. I will re-map these variables to group all the categories that are less frequent into a new category called 'Other'.

# In[31]:


# I will re-group these variables into either the majoritarian category
# or "Other"

# if a category is present in less than 10% of the observations, we group it with other
# infrequent categories (tol param)

rare_enc = enc.RareLabelEncoder(tol = 0.1,
                                n_categories=1, # number of minimum categories per variable for the grouping to proceed
                                variables=constant.features_to_drop_, # the variables to pre-process
                                replace_with='Other', # the label to use to replace the original category
                               )

rare_enc.fit(X_train)


# In[32]:


# make the grouping

X_train = rare_enc.transform(X_train)
X_test = rare_enc.transform(X_test)
submission = rare_enc.transform(submission)


# In[33]:


# Now, when I plot the data again, I should see only 2 categories in each variable

# put data together for analysis
data = create_master_data(X_train, X_test, submission, y_train, y_test)

for variable in constant.features_to_drop_:
    
    data.groupby(variable)['data'].value_counts().unstack().plot.bar(figsize=(10,5))
    plt.title(variable)
    plt.ylabel('Number of houses')
    plt.show()


# ## Quality variables
# 
# There are a number of variables that refer to the quality of some aspect of the house, for example the garage, or the fence, or the kitchen. I will replace these categories by numbers increasing with the quality of the place or room.
# 
# - Ex = Excellent
# - Gd = Good
# - TA = Average/Typical
# - Fa =	Fair
# - Po = Poor

# In[34]:


qual_mappings = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5, 'Missing': 0, 'NA': 0}

qual_vars = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
             'HeatingQC', 'KitchenQual', 'FireplaceQu',
             'GarageQual', 'GarageCond',
            ]

for var in qual_vars:
    X_train[var] = X_train[var].map(qual_mappings)
    X_test[var] = X_test[var].map(qual_mappings)
    submission[var] = submission[var].map(qual_mappings)


# In[35]:


exposure_mappings = {'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4, 'Missing': 0, 'NA': 0}

var = 'BsmtExposure'

X_train[var] = X_train[var].map(exposure_mappings)
X_test[var] = X_test[var].map(exposure_mappings)
submission[var] = submission[var].map(exposure_mappings)


# In[36]:


finish_mappings = {'Missing': 0, 'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}

finish_vars = ['BsmtFinType1', 'BsmtFinType2']

for var in finish_vars:
    X_train[var] = X_train[var].map(finish_mappings)
    X_test[var] = X_test[var].map(finish_mappings)
    submission[var] = submission[var].map(finish_mappings)


# In[37]:


garage_mappings = {'Missing': 0, 'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}

var = 'GarageFinish'

X_train[var] = X_train[var].map(garage_mappings)
X_test[var] = X_test[var].map(garage_mappings)
submission[var] = submission[var].map(garage_mappings)


# In[38]:


fence_mappings = {'Missing': 0, 'NA': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}

var = 'Fence'

X_train[var] = X_train[var].map(fence_mappings)
X_test[var] = X_test[var].map(fence_mappings)
submission[var] = submission[var].map(fence_mappings)


# In[39]:


# capture all quality variables

qual_vars  = qual_vars + finish_vars + ['BsmtExposure','GarageFinish','Fence']


# In[40]:


# now let's plot the house mean sale price based on the quality of the 
# various attributes

# put data together for analysis
data = create_master_data(X_train, X_test, submission, y_train, y_test)

for var in qual_vars:
    # make boxplot with Catplot
    sns.catplot(x=var, y='target', data=data[data['data']=='train'], kind="box", height=4, aspect=1.5)
    # add data points to boxplot with stripplot
    sns.stripplot(x=var, y='target', data=data[data['data']=='train'], jitter=0.1, alpha=0.3, color='k')
    plt.show()
    


# For many quality variables, we see an increase in the house price, with the quality.
# 
# ### Cardinality
# 
# Cardinality indicates the number of unique values or categories per variable. The highest the cardinality, the more difficult the variable is to handle. Although it could provide more information.

# In[41]:


# let's collect the remaining categorical variables

categorical = [c for c in categorical if c not in constant.features_to_drop_]

categorical = [c for c in categorical if c not in qual_vars]

len(categorical)


# In[42]:


# with nunique from pandas we count the number of unique categories per variable

data[categorical].nunique().sort_values().plot.bar(figsize=(15,5))
plt.ylabel('Number of categories per variable')
plt.show()


# Some variables have few categories (low cardinality), others have a lot of categories (high cardinality). 
# 
# In addition to the number of unique categories, it is important to know how many observations in the dataset show each category. Rare categories tend to bring problems when building the models, with little reliable information, precisely, because there are not a lot of observations to learn from.

# In[43]:


# plot number of observations / houses per category, per variable

for variable in categorical:
    data.groupby(variable)['data'].value_counts().unstack().plot.bar(figsize=(10,5))
    plt.title(variable)
    plt.ylabel('Number of houses')
    plt.show()


# We see that pretty much for every variable, some categories are shared by many houses, and some are shown only by very few houses. Then, we wouldn't know if we should trust the Sale Price of houses with these few categories, because we have few houses to learn from.
# 
# Another problem that comes up with rare categories, is that they may appear only in the train set, or only in the test set, or maybe even only in the submission. So if it appears only in the train set, they may cause over-fitting. But if it appears only on the test set, then the model does not know what to make of that category, and it may crash.
# 
# **I will group infrequent labels together into 1 umbrella category called 'Rare'**. But before that, let's have a look at the mean sale price per category, to see if we find some value in any of these variables that we can capitalize.

# In[44]:


# the following function calculates:

# 1) the percentage of houses per category
# 2) the mean SalePrice per category

# and returns a dataframe with those 2 variables

def calculate_mean_target_per_category(df, var):
    
    df = pd.concat([df, y_train], axis=1)
    
    # total number of houses
    total_houses = len(df)

    # percentage of houses per category
    temp_df = pd.Series(df[var].value_counts() / total_houses).reset_index()
    temp_df.columns = [var, 'perc_houses']

    # add the mean SalePrice
    temp_df = temp_df.merge(df.groupby([var])['SalePrice'].mean().reset_index(),
                            on=var,
                            how='left')

    return temp_df


# In[45]:


# now we use the function for the variable 'Neighborhood'
temp_df = calculate_mean_target_per_category(X_train, 'Neighborhood')

temp_df.head()


# In[46]:


# Now I create a function to plot of the
# category frequency and mean SalePrice.

# This will help us visualise the relationship between the
# target and the labels of the  categorical variable

def plot_categories(df, var):
    
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.xticks(df.index, df[var], rotation=90)

    ax2 = ax.twinx()
    ax.bar(df.index, df["perc_houses"], color='lightgrey')
    ax2.plot(df.index, df["SalePrice"], color='green', label='Seconds')
    ax.axhline(y=0.05, color='red')
    ax.set_ylabel('percentage of houses per category')
    ax.set_xlabel(var)
    ax2.set_ylabel('Average Sale Price per category')
    plt.show()


# In[47]:


# plot house count and sale price for Neighbourhood.

plot_categories(temp_df, 'Neighborhood')


# From the above plot, we can see that there are expensive neighbourhoods and cheap neighbourhoods.

# In[48]:


temp_df['SalePrice'].describe()


# In[49]:


# make a list of the least expensive neighbourhoods

cheap_neighbourhoods = temp_df[temp_df['SalePrice']<11.85]['Neighborhood'].unique()

cheap_neighbourhoods


# In[50]:


# make a list of the most expensive neighbourhoods

expensive_neighbourhoods = temp_df[temp_df['SalePrice']>12.2]['Neighborhood'].unique()

expensive_neighbourhoods


# In[51]:


# new variable that segregates neigbourhoods as per the previous lists

X_train['Neigh_Price'] = np.where(X_train['Neighborhood'].isin(cheap_neighbourhoods),
                                  0, np.where(X_train['Neighborhood'].isin(expensive_neighbourhoods),
                                              2, 1))

X_test['Neigh_Price'] = np.where(X_test['Neighborhood'].isin(cheap_neighbourhoods),
                                  0, np.where(X_test['Neighborhood'].isin(expensive_neighbourhoods),
                                              2, 1))

submission['Neigh_Price'] = np.where(submission['Neighborhood'].isin(cheap_neighbourhoods),
                                  0, np.where(submission['Neighborhood'].isin(expensive_neighbourhoods),
                                              2, 1))

# let's inspect the new variable
X_train['Neigh_Price'].value_counts()


# In[52]:


# let's inspect the value of our new variable

# put data together for analysis
data = create_master_data(X_train, X_test, submission, y_train, y_test)

sns.catplot(x='Neigh_Price', y='target', data=data[data['data']=='train'], kind="box", height=4, aspect=1.5)
# add data points to boxplot with stripplot
sns.stripplot(x='Neigh_Price', y='target', data=data[data['data']=='train'], jitter=0.1, alpha=0.3, color='k')
plt.show()


# Looks good to me.

# In[53]:


# let's plot for the remaining categorical variables
# the count of houses per category and the mean sale price

for col in categorical:
    
    # we plotted this variable already
    if col !='Neighborhood':
        
        # re using the functions I created
        temp_df = calculate_mean_target_per_category(data, col)
        plot_categories(temp_df, col)


# The other variables, don't show a clear increase in house for certain categories. So I will not derive new features from them.

# In[54]:


# Now I will group infrequent labels together:

# if a category is present in less than 5% of the observations, we group it with other
# infrequent categories (tol param)

rare_enc = enc.RareLabelEncoder(tol = 0.05,
                                n_categories=4, # number of minimum categories per variable for the grouping to proceed
                                variables=categorical
                               )

rare_enc.fit(X_train)


# The warnings tell me that there are a few categorical variables that I indicated to the variables parameter, that have less than 4 unique categories. So the transformer will not pre-process those.

# In[55]:


X_train = rare_enc.transform(X_train)
X_test = rare_enc.transform(X_test)
submission = rare_enc.transform(submission)

X_train.shape, X_test.shape, submission.shape


# In[56]:


# put data together for analysis

data = create_master_data(X_train, X_test, submission, y_train, y_test)


# In[57]:


# now we can plot the variables with grouped categories

for variable in categorical:
    data.groupby(variable)['data'].value_counts().unstack().plot.bar(figsize=(10,5))
    plt.title(variable)
    plt.ylabel('Number of houses')
    plt.show()


# We see that after grouping the variables, the categories appear in all the 3 datasets, train, test and submission. And also, they are shared by at least more than 5% of the observations in the datasets.
# 
# ### Categorical variable importance
# 
# Let's plot the House Sale Price distribution per category per variable to understand if there is a difference.
# 
# **Boxplot**: indicates the median value of the house, and the interquantal range distance, which contains most of the houses. The rombos above and below are outliers for that distribution.
# 
# **Jitter**: on top of the box plot I plot the houses individually as dots, this gives us an idea of how many houses show that category. More dots, more houses.
# 
# If the box plots are at the same height, then the categories or the variable do not show predictive power. But if the show different heights for the different categories, then they might.

# In[58]:


for variable in categorical:
    # make boxplot with Catplot
    sns.catplot(x=variable, y='target', data=data[data['data']=='train'], kind="box", height=4, aspect=1.5)
    # add data points to boxplot with stripplot
    sns.stripplot(x=variable, y='target', data=data[data['data']=='train'], jitter=0.1, alpha=0.3, color='k')
    plt.show()


# Some categories seem to correlate with higher or lower house sale prices. And some, not at all. See for example the plot for the variable **Foundation**. The different foundations seem to correlate with lower sale prices. The same is true for the variable **SaleCondition** and others.
# 
# We can try and determine which of the variables are useful using an approach used in the KDD2009 data science competition. The approach consists in replacing the category by the mean of the target. And then using that replacement as a prediction, and evaluate the performance comparing the prediction to the real value of the house price.
# 
# We can perform all of this, very easily with a class from Feature-engine.
# 
# Check [SelectByTargetMeanPerformance](https://feature-engine.readthedocs.io/en/latest/selection/SelectByTargetMeanPerformance.html) for more details.

# In[59]:


selector = sel.SelectByTargetMeanPerformance(
    variables=categorical, # the variables to examine
    scoring='r2_score', # the metric to use for the performance evaluation
    threshold=0.1, # the minimum performance threshold for a variable to be selected
    cv=2, # the cross-validation fold
    random_state=0)

# find the variables that are important
selector.fit(X_train, y_train)


# In[60]:


# in the attribute feature_performance_ the class stores the performance
# of each feature

pd.Series(selector.feature_performance_).sort_values().plot.bar(figsize=(15,5))
plt.ylabel('r2')

# the red line is the threshold we selected.
plt.axhline(y = 0.1, color = 'r', linestyle = '-') 
plt.show()


# I selected the threshold somewhat arbitrarily. We can play with this a bit to select more or less variables. If we leave the parameter to None, then the class will select those features which performance is above the mean performance of the group. An S2 of 0.1 indicates that the variable explains 10% of the total variability in the data. So it is not too bad.

# In[61]:


# number of variables to remove

len(selector.features_to_drop_)


# In[62]:


# remove non predictive categorical variables

X_train = selector.transform(X_train)
X_test = selector.transform(X_test)
submission = selector.transform(submission)

X_train.shape, X_test.shape, submission.shape


# ### Encode categorical variables
# 
# To use categorical variables in machine learning models, we need to encode them into numbers. At least for the Gradient Boosting Classifier from Sklearn.
# 
# We are going to assign numbers to the variables, but these numbers will be assigned from smaller to bigger, based on the mean sale price per category. This way, we create (hopefully) a monotonic relationship between the encoded variable and the target, which may boost the performance of the model.
# 
# [OrdinalEncoder](https://feature-engine.readthedocs.io/en/latest/encoding/OrdinalEncoder.html)

# In[63]:


# let's identify all remaining categorical variables (remember that we encoded
# some into numbers already)

categorical = [var for var in X_train.columns if X_train[var].dtype == 'O']

# MSSubClass is also categorical by definition, despite its numeric values
categorical = categorical + ['MSSubClass']

# number of categorical variables
len(categorical)


# In[64]:


# cast all variables as categorical

X_train[categorical] = X_train[categorical].astype('O')
X_test[categorical] = X_test[categorical].astype('O')
submission[categorical] = submission[categorical].astype('O')


# In[65]:


# set up the ordinal encoder from Feature-engine

encoder = enc.OrdinalEncoder(encoding_method='ordered',
                             variables=categorical)

encoder.fit(X_train, y_train)


# In[66]:


# in this attribute we find the numbers that will replace each category in each variable

encoder.encoder_dict_


# In[67]:


# encode the variables

X_train = encoder.transform(X_train)
X_test = encoder.transform(X_test)
submission = encoder.transform(submission)

X_train.shape, X_test.shape, submission.shape


# In[68]:


# Now let's examine the monotonic relationships

for var in categorical:
    plt.scatter(X_train[var], y_train, alpha=0.2)
    plt.ylabel('Log of Sale Price')
    plt.xlabel(var)
    plt.show()


# For some variables, we see an increase in the house price with the encoded variable value. 

# In[69]:


# check that NAN were not introduced during the encoding

[c for c in categorical if X_train[c].isnull().sum()>0]


# In[70]:


# check that NAN were not introduced during the encoding

[c for c in categorical if X_test[c].isnull().sum()>0]


# In[71]:


[c for c in categorical if submission[c].isnull().sum()>0]


# ## Time variables
# 
# There are a few variables that show time, for example the year in which the house was sold, or the garage was built.

# In[72]:


year_vars = [var for var in data.columns if 'Yr' in var or 'Year' in var]

year_vars


# Let's plot the sale price over the years.

# In[73]:


data = create_master_data(X_train, X_test, submission, y_train, y_test)


# In[74]:


# make boxplot with Catplot
sns.catplot(x='YrSold', y='target', data=data[data['data']=='train'], kind="box", height=4, aspect=1.5)
# add data points to boxplot with stripplot
sns.stripplot(x='YrSold', y='target', data=data[data['data']=='train'], jitter=0.1, alpha=0.3, color='k')
plt.show()


# We see that the sale price seems stable over the years for which we have data.
# 
# Now let's plot the house price vs the last time where something was remodelled in the house. I expect more expensive houses if they were recently remodelled.

# In[75]:


for variable in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    plt.scatter(X_train[variable], y_train)
    plt.ylabel('Log of sale Price')
    plt.xlabel(variable)
    plt.show()


# There seems to be a slight linear trend to increase in house prices, for those houses that were remodelled in later years.
# 
# Let's capture the difference between the year in which something was remodelled or built and the sale time.
# 
# [CombineWithFeatureReference](https://feature-engine.readthedocs.io/en/latest/creation/CombineWithReferenceFeature.html)

# In[76]:


# this transformer will substract all the variables in the reference list, from YrSold
# to determine the age of remodelling at point of sale

create = creation.CombineWithReferenceFeature(
    variables_to_combine = ['YrSold'],
    reference_variables = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt'],
    operations=['sub'],
)

create.fit(X_train)


# In[77]:


X_train = create.transform(X_train)
X_test = create.transform(X_test)
submission = create.transform(submission)

X_train.shape, X_test.shape, submission.shape


# In[78]:


# we can see the new variables at the right of the dataframe
X_train.head()


# In[79]:


# if the variable 'YearRemodAdd' shows the same value as the
# variable 'YearBuilt', that means that the house has not been remodelled

# so let's create a new feature that captures this.

remodelled = creation.CombineWithReferenceFeature(
    variables_to_combine = ['YearRemodAdd'],
    reference_variables = ['YearBuilt'],
    operations=['sub'],
)

remodelled.fit(X_train)


# In[80]:


X_train = remodelled.transform(X_train)
X_test = remodelled.transform(X_test)
submission = remodelled.transform(submission)

X_train.shape, X_test.shape, submission.shape


# In[81]:


# we can see the new variable at the right of the dataframe
X_train.head()


# In[82]:


# now we need to remove the original time features

drop_features = sel.DropFeatures(features_to_drop =['YearBuilt', 'YearRemodAdd', 'GarageYrBlt','YrSold'])

X_train = drop_features.fit_transform(X_train)
X_test = drop_features.transform(X_test)
submission = drop_features.transform(submission)

X_train.shape, X_test.shape, submission.shape


# ### Missing data

# In[83]:


# let's check if the variables have missing data

X_train[[
    'YrSold_sub_YearBuilt', 'YrSold_sub_YearRemodAdd' ,'YrSold_sub_GarageYrBlt', 'YearRemodAdd_sub_YearBuilt']
].isnull().mean()


# In[84]:


X_test[[
    'YrSold_sub_YearBuilt', 'YrSold_sub_YearRemodAdd' ,'YrSold_sub_GarageYrBlt', 'YearRemodAdd_sub_YearBuilt']
].isnull().mean()


# In[85]:


submission[[
    'YrSold_sub_YearBuilt', 'YrSold_sub_YearRemodAdd' ,'YrSold_sub_GarageYrBlt', 'YearRemodAdd_sub_YearBuilt']
].isnull().mean()


# YrSold_sub_GarageYrBlt shows missing data. I will impute this altogether when I impute numerical variables later on in the notebook.
# 
# The final temporal variable is the month in which the house was sold. Let's check if there is some price change depending on the month of the sale.

# In[86]:


# make boxplot with Catplot
sns.catplot(x='MoSold', y='target', data=data[data['data']=='train'], kind="box", height=4, aspect=1.5)
# add data points to boxplot with stripplot
sns.stripplot(x='MoSold', y='target', data=data[data['data']=='train'], jitter=0.1, alpha=0.3, color='k')
plt.show()


# The mean house sale price seems similar throughout the mosnts, but there is a difference in the number of houses sold, with less houses sold in Jan and Feb, and more houses sold towards the summer months of the northern hemisphere.

# ## Discrete variables

# In[87]:


discrete = [
    var for var in X_train.columns if X_train[var].dtype != 'O'
    and var not in year_vars
    and var not in qual_vars+categorical
    and len(X_train[var].unique()) < 20 
]

# number of discrete variables
len(discrete)


# Let's see if there is a relationship between the values of discrete houses and the mean sale price.

# In[88]:


for var in discrete:
    # make boxplot with Catplot
    sns.catplot(x=var, y='target', data=data[data['data']=='train'], kind="box", height=4, aspect=1.5)
    # add data points to boxplot with stripplot
    sns.stripplot(x=var, y='target', data=data[data['data']=='train'], jitter=0.1, alpha=0.3, color='k')
    plt.show()


# For most of these variables, there is a correlation between the number assigned to the house and the sale price. Particularly for those variables which values are determined by people, for example OverallQual, or OverallCond. But also for variables like number of bathrooms or rooms.
# 
# The only variable that seems to have only 1 value predominantly is PoolArea, so I will remove it from the dataset.

# In[89]:


# remove original feature

drop_features = sel.DropFeatures(features_to_drop =['PoolArea'])

X_train = drop_features.fit_transform(X_train)
X_test = drop_features.transform(X_test)
submission = drop_features.transform(submission)

X_train.shape, X_test.shape, submission.shape


# In[90]:


### Missing data

# let's capture the discrete variables with missing data

null_disc = {var: data[var].isnull().mean() for var in discrete if data[var].isnull().mean()>0}

# plot
pd.Series(null_disc).sort_values().plot.bar(figsize=(10,4))
plt.ylabel('Percentage of missing data')
plt.show()


# There are 3 discrete variables with NA. I will impute these altogether with numerical variables 

# In[91]:


# get variable names

pd.Series(null_disc).sort_values().index


# ## Numerical variables

# In[92]:


numerical = [
    var for var in X_train.columns 
    if var not in categorical + qual_vars + discrete + year_vars
]

len(numerical)


# In[93]:


# let's examine the distribution of the numerical continuous variables

X_train[numerical].hist(bins=50, figsize=(15,15))

plt.show()


# There are some variables that show predominantly 1 value, like MiscVal, ScreenPorch, LowQualFinSF. I could remove them with the constant features selector. But I could also let the model decide if they are important or not. So I will do this, this time.

# In[94]:


# let's plot the sale price vs the numerical variables

tmp = pd.concat([X_train, y_train], axis=1)

sns.pairplot(data=tmp,
             y_vars='SalePrice',
             x_vars=['LotFrontage',
                     'LotArea',
                     'MasVnrArea',
                     'BsmtFinSF1',
                     'BsmtFinSF2', ])
plt.show()


# We se that the higher the value of the variable, the higher the sale price, for most of these variables.

# In[95]:


sns.pairplot(data=tmp,
             y_vars=['SalePrice'],
             x_vars=['BsmtUnfSF',
                     'TotalBsmtSF',
                     '1stFlrSF',
                     '2ndFlrSF',
                     'LowQualFinSF', ])
plt.show()


# Same here, the higher the value of the variable, the higher the sale price, in general.

# In[96]:


sns.pairplot(data=tmp,
             y_vars=['SalePrice'],
             x_vars=['GrLivArea',
                     'GarageArea',
                     'WoodDeckSF',
                     'OpenPorchSF',
                     'EnclosedPorch', ])

plt.show()


# In[97]:


sns.pairplot(data=tmp,
             y_vars=['SalePrice'],
             x_vars=['3SsnPorch',
                     'ScreenPorch',
                     'MiscVal'])

plt.show()


# For these variables, there does not seem to be a clear tendency.

# In[98]:


# and now the time variables we created before

sns.pairplot(data=tmp,
             y_vars=['SalePrice'],
             x_vars=['YrSold_sub_YearBuilt', 
                     'YrSold_sub_YearRemodAdd' ,
                     'YrSold_sub_GarageYrBlt',
                     'YearRemodAdd_sub_YearBuilt'])

plt.show()


# There is some promise in these variables
# 
# ### Missing Data

# In[99]:


# concatenate data sources
data = create_master_data(X_train, X_test, submission, y_train, y_test)

# capture numerical variables with NA
null_num = {var: data[var].isnull().mean() for var in numerical if data[var].isnull().mean()>0}

# plot
pd.Series(null_num).sort_values().plot.bar(figsize=(5,4))
plt.ylabel('Percentage of missing data')
plt.show()


# There are a few variables that show missing data.
# 
# Let's add a missing indicator first and then impute the missing data with the median value of the variable. 
# 
# [MeanMedianImputer](https://feature-engine.readthedocs.io/en/latest/imputation/MeanMedianImputer.html)
# 
# [AddMissingIndicator](https://feature-engine.readthedocs.io/en/1.0.x/imputation/AddMissingIndicator.html)

# In[100]:


vars_to_impute = [c for c in null_num.keys()]

# add the discrete variables with NA
vars_to_impute = vars_to_impute + ['GarageCars', 'BsmtFullBath', 'BsmtHalfBath']

indicator = imp.AddMissingIndicator(missing_only = False,
                                variables = vars_to_impute,
                               )

indicator.fit(X_train)


# In[101]:


# in this attribute we find the variables for which
# the missing indicators will be added

indicator.variables_


# In[102]:


X_train = indicator.transform(X_train)
X_test = indicator.transform(X_test)
submission = indicator.transform(submission)

X_train.shape, X_test.shape, submission.shape


# In[103]:


# now replace NA by the median value

num_imputer = imp.MeanMedianImputer(imputation_method = 'median',
                                    variables = vars_to_impute,
                                   )

num_imputer.fit(X_train)

# the median for imputation for each numerical variable
num_imputer.imputer_dict_


# In[104]:


X_train = num_imputer.transform(X_train)
X_test = num_imputer.transform(X_test)
submission = num_imputer.transform(submission)

X_train.shape, X_test.shape, submission.shape


# In[105]:


X_train.head()


# In[106]:


[c for c in X_train.columns if X_train[c].isnull().sum()>0]


# In[107]:


[c for c in X_test.columns if X_test[c].isnull().sum()>0]


# In[108]:


[c for c in submission.columns if submission[c].isnull().sum()>0]


# In[109]:


# create some more features

# total number of basement bathrooms
bath_bsmt = creation.MathematicalCombination(
    variables_to_combine=['BsmtHalfBath', 'BsmtFullBath'],
    math_operations=['sum'],
    new_variables_names=['BsmtBath_total'],
)

# total number of bathrooms
bath_ground = creation.MathematicalCombination(
    variables_to_combine=['FullBath', 'HalfBath'],
    math_operations=['sum'],
    new_variables_names=['Bath_total'],
)

X_train = bath_bsmt.fit_transform(X_train)
X_test = bath_bsmt.transform(X_test)
submission = bath_bsmt.transform(submission)

X_train = bath_ground.fit_transform(X_train)
X_test = bath_ground.transform(X_test)
submission = bath_ground.transform(submission)

X_train.shape, X_test.shape, submission.shape


# ## Model with nested cross-validation

# In[110]:


X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)


# In[111]:


def nested_cross_val(model, grid):
    
    # configure the cross-validation procedure
    cv_outer = KFold(n_splits=5, shuffle=True, random_state=1)

    # enumerate splits
    outer_results = list()

    for train_ix, test_ix in cv_outer.split(X_train):

        # split data
        xtrain, xtest = X_train.loc[train_ix, :], X_train.loc[test_ix, :]
        ytrain, ytest = y_train[train_ix], y_train[test_ix]

        # configure the cross-validation procedure
        cv_inner = KFold(n_splits=5, shuffle=True, random_state=1)

        # define search
        search = GridSearchCV(model, grid, scoring='neg_root_mean_squared_error', cv=cv_inner, refit=True)

        # execute search
        search.fit(xtrain, ytrain)

        # evaluate model on the hold out dataset
        yhat = search.predict(xtest)

        # evaluate the model
        rmse = mean_squared_error(ytest, yhat, squared=False)

        # store the result
        outer_results.append(rmse)

        # report progress
        print('>rmse_outer=%.3f, rmse_inner=%.3f, cfg=%s' % (rmse, search.best_score_, search.best_params_))

    # summarize the estimated performance of the model
    print('rmse_outer: %.3f +- %.3f' % (np.mean(outer_results), np.std(outer_results)))
    
    return search.fit(X_train, y_train)


# ### GBM

# In[112]:


gbm = GradientBoostingRegressor(
    loss='ls',
    n_estimators=100,
    criterion='friedman_mse',
    min_samples_split=2,
    max_depth=3,
    random_state=0,
    n_iter_no_change=2,
    tol=0.0001,
    )

param_grid = dict(
    loss=['ls', 'huber'],
    n_estimators=[10, 20, 50, 100, 200, 500, 1000, 2000],
    min_samples_split=[0.1, 0.3, 0.5],
    max_depth=[1,2,3,4,None],
    )

search = nested_cross_val(gbm, param_grid)


# In[113]:


results = pd.DataFrame(search.cv_results_)

results.sort_values(by='mean_test_score', ascending=False, inplace=True)

results.reset_index(drop=True, inplace=True)

results[['params','mean_test_score', 'std_test_score']]

results['mean_test_score'].plot(yerr=[results['std_test_score'], results['std_test_score']], subplots=True)

plt.ylabel('Mean test score')


# In[114]:


# let's get the predictions

X_train_preds = search.predict(X_train)
X_test_preds = search.predict(X_test)

submission_preds = search.predict(submission)

print('Train rmse: ', mean_squared_error(y_train, X_train_preds, squared=False))
print('Test rmse: ', mean_squared_error(y_test, X_test_preds, squared=False))
print()
print('Train r2: ', r2_score(y_train, X_train_preds))
print('Test r2: ', r2_score(y_test, X_test_preds))


# In[115]:


np.exp(submission_preds)


# In[116]:


# my_submission = pd.DataFrame({'Id': id_, 'SalePrice': np.exp(submission_preds)})

# # you could use any filename. We choose submission here
# my_submission.to_csv('submission_gbm_full.csv', index=False)


# ### Light GBM

# In[117]:


# Light GBM

lgbm_param = {
    "num_leaves": [6, 8, 20, 30],
    "max_depth": [2, 4, 6, 8, 10],
    "n_estimators": [50, 100, 200, 500],
    'colsample_bytree': [0.3, 1.0],
}

lgbm = LGBMRegressor(
    learning_rate = 0.1, 
    min_child_weight = 0.4,
    objective='regression', 
    random_state=0)

search = nested_cross_val(lgbm, lgbm_param)


# In[118]:


results = pd.DataFrame(search.cv_results_)

results.sort_values(by='mean_test_score', ascending=False, inplace=True)

results.reset_index(drop=True, inplace=True)

results[['params','mean_test_score', 'std_test_score']]

results['mean_test_score'].plot(yerr=[results['std_test_score'], results['std_test_score']], subplots=True)

plt.ylabel('Mean test score')


# In[119]:


# let's get the predictions

X_train_preds = search.predict(X_train)
X_test_preds = search.predict(X_test)

submission_preds = search.predict(submission)

print('Train rmse: ', mean_squared_error(y_train, X_train_preds, squared=False))
print('Test rmse: ', mean_squared_error(y_test, X_test_preds, squared=False))
print()
print('Train r2: ', r2_score(y_train, X_train_preds))
print('Test r2: ', r2_score(y_test, X_test_preds))


# In[120]:


np.exp(submission_preds)


# In[121]:


my_submission = pd.DataFrame({'Id': id_, 'SalePrice': np.exp(submission_preds)})

# you could use any filename. We choose submission here
my_submission.to_csv('submission_lgbm_full.csv', index=False)


# ## Feature selection

# In[122]:


sel_perf = sel.SelectBySingleFeaturePerformance(
    estimator=DecisionTreeRegressor(random_state=2, max_depth=2),
    scoring='r2',
    cv=3,
    threshold=0.01,
    variables=None,
)

sel_perf.fit(X_train, y_train)


# In[123]:


# in the attribute feature_performance_ the class stores the performance
# of each feature

pd.Series(sel_perf.feature_performance_).sort_values().plot.bar(figsize=(15,5))
plt.ylabel('r2')

# the red line is the threshold we selected.
plt.axhline(y = pd.Series(sel_perf.feature_performance_).mean(),
            color = 'g', linestyle = '-')

# the red line is the threshold we selected.
plt.axhline(y = pd.Series(sel_perf.threshold).mean(),
            color = 'r', linestyle = '-') 

plt.show()


# In[124]:


# number of features to drop

print('total features: ', X_train.shape[1])
print('features to drop: ', len(sel_perf.features_to_drop_))
print('remaining features: ', X_train.shape[1] - len(sel_perf.features_to_drop_))


# In[125]:


# let's try  different selection method

rfm = RandomForestRegressor(
    n_estimators=100,
    random_state=0,
    max_depth=2,
    )

rfe = sel.RecursiveFeatureElimination(
    estimator = rfm,
    scoring ='neg_root_mean_squared_error',
    cv=3,
    threshold=0.001,
    variables=None,
)

rfe.fit(X_train, y_train)


# In[126]:


# performance of the gbm built using all features

rfe.initial_model_performance_


# In[127]:


# in the attribute performance_drifts_ the class stores the performance
# of each feature the drop in performance when the feature was removed

pd.Series(rfe.performance_drifts_).sort_values().plot.bar(figsize=(15,5))
plt.ylabel('neg rmse')

# the red line is the threshold we selected.
plt.axhline(y = pd.Series(rfe.performance_drifts_).mean(),
            color = 'g', linestyle = '-')

plt.axhline(y = rfe.threshold,
            color = 'r', linestyle = '-') 

plt.show()


# In[128]:


# in the attribute feature_importances_ the class stores the importance
# of each feature, derived from the gbm

pd.Series(rfe.feature_importances_).sort_values().plot.bar(figsize=(15,5))
plt.ylabel('neg rmse')

# the red line is the threshold we selected.
plt.axhline(y = pd.Series(rfe.feature_importances_).mean(),
            color = 'r', linestyle = '-') 
plt.show()


# In[129]:


# number of features to drop

print('total features: ', X_train.shape[1])
print('features to drop: ', len(rfe.features_to_drop_))
print('remaining features: ', X_train.shape[1] - len(rfe.features_to_drop_))


# In[130]:


X_train = sel_perf.transform(X_train) 
X_test = sel_perf.transform(X_test) 
submission = sel_perf.transform(submission) 

X_train.shape, X_test.shape, submission.shape


# ## Train model with cross-validation, searching for best parameters
# 
# In the rest of the notebook, I will perform a Random Search with cross-validation for the best parameters of a GradientBoostingClassifier.

# In[131]:


gbm = GradientBoostingRegressor(
    loss='ls',
    n_estimators=100,
    criterion='friedman_mse',
    min_samples_split=2,
    max_depth=3,
    random_state=0,
    n_iter_no_change=2,
    tol=0.0001,
    )


# In[132]:


param_grid = dict(
    loss=['ls', 'huber'],
    n_estimators=[10, 20, 50, 100, 200, 500, 1000, 2000],
    min_samples_split=[0.1, 0.3, 0.5],
    max_depth=[1,2,3,4,None],
    )


# In[133]:


search = nested_cross_val(gbm, param_grid)


# In[134]:


# reg = GridSearchCV(gbm, param_grid, scoring='neg_mean_squared_error')

# search = reg.fit(X_train, y_train)

# search.best_params_


# In[135]:


results = pd.DataFrame(search.cv_results_)

results.head()


# In[136]:


results.sort_values(by='mean_test_score', ascending=False, inplace=True)

results.reset_index(drop=True, inplace=True)

results[['params','mean_test_score', 'std_test_score']]


# In[137]:


results['params'][0]


# In[138]:


results['mean_test_score'].plot(yerr=[results['std_test_score'], results['std_test_score']], subplots=True)

plt.ylabel('Mean test score')


# In[139]:


# let's get the predictions

X_train_preds = search.predict(X_train)
X_test_preds = search.predict(X_test)

submission_preds = search.predict(submission)


# In[140]:


print('Train rmse: ', mean_squared_error(y_train, X_train_preds, squared=False))
print('Test rmse: ', mean_squared_error(y_test, X_test_preds, squared=False))


# In[141]:


print('Train r2: ', r2_score(y_train, X_train_preds))
print('Test r2: ', r2_score(y_test, X_test_preds))


# In[142]:


np.exp(submission_preds)


# In[143]:


my_submission = pd.DataFrame({'Id': id_, 'SalePrice': np.exp(submission_preds)})

# you could use any filename. We choose submission here
my_submission.to_csv('submission_gbm_small.csv', index=False)


# ## Light GBM

# In[144]:


# Light GBM

lgbm_param = {
    "num_leaves": [6, 8, 20, 30],
    "max_depth": [2, 4, 6, 8, 10],
    "n_estimators": [50, 100, 200, 500],
    'colsample_bytree': [0.3, 1.0],
}

lgbm = LGBMRegressor(
    learning_rate = 0.1, 
    min_child_weight = 0.4,
    objective='regression', 
    random_state=0)


# In[145]:


search = nested_cross_val(lgbm, lgbm_param)


# In[146]:


# reg = GridSearchCV(lgbm, lgbm_param, scoring='neg_mean_squared_error')

# search = reg.fit(X_train, y_train)

# search.best_params_


# In[147]:


results.sort_values(by='mean_test_score', ascending=False, inplace=True)

results.reset_index(drop=True, inplace=True)

results['mean_test_score'].plot(yerr=[results['std_test_score'], results['std_test_score']], subplots=True)

plt.ylabel('Mean test score')


# In[148]:


# let's get the predictions
X_train_preds = search.predict(X_train)
X_test_preds = search.predict(X_test)

submission_preds = search.predict(submission)


# In[149]:


print('Train rmse: ', mean_squared_error(y_train, X_train_preds,squared=False))
print('Test rmse: ', mean_squared_error(y_test, X_test_preds,squared=False))


# In[150]:


print('Train r2: ', r2_score(y_train, X_train_preds))
print('Test r2: ', r2_score(y_test, X_test_preds))


# In[151]:


np.exp(submission_preds)


# In[152]:


my_submission = pd.DataFrame({'Id': id_, 'SalePrice': np.exp(submission_preds)})

# you could use any filename. We choose submission here
my_submission.to_csv('submission_lgbm_small.csv', index=False)

