#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import norm, skew

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Understanding the Data

# **From the data description, the dataset is the Ames Housing dataset, which has about 79 features describing the aspects of a dream home. We will explore the data, before anything**

# In[2]:


#Loading the data
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df.head()


# **From the initial look of it, we see:**
# * There are lots of missing values for some fields
# * General number of features are too high.
# 
# **So let's try understanding the features, by reading the data description**

# In[3]:


with open("/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt",'r') as f:
    print(f.read())


# **By exploring the features, we get to know that the features explain every aspect from the exterior structure to plot structure to interior details.**
# 
# **So we must be able to split the features, and do a exploratory data analysis of each aspect, and before that let's check do some preliminary analysis on the entire data**

# **Let's start with the target variable, 'SalesPrice', we will do some exploration**

# # SalesPrice - Target Variable

# In[4]:


#Plotting the correlation
corr_df = df.corr()['SalePrice'].sort_values(ascending=False)
plt.figure(figsize=(5,8))
sns.barplot( y=corr_df.index, x=corr_df)
plt.xlabel("Correlation with Salesprice")


# **Overall Quality, Living area, Garage area, basement area becomes some of the most correlated features, and that makes complete sense**

# In[5]:


#Plotting a relationship b/w GrLivArea and Salesprice
fig, ax = plt.subplots()
ax.scatter(x = df['GrLivArea'], y = df['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# **We see 2 outlier values, down to the bottom, which has living area, and is available for a lower salesprice. We will remove these outliers, before performing any analysis, as that would mess with the later performance of the model**

# In[6]:


df = df.drop(df[(df['GrLivArea']>4000) & (df['SalePrice']<300000)].index)


# In[7]:


#Lets plot the curve again
fig, ax = plt.subplots()
ax.scatter(x = df['GrLivArea'], y = df['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# This looks pretty clean!

# Now, for simplicity, let's analyse the data, by splitting the features to several sub-components 
# 
# From the detailed study of the data, from data description, the features can be clearly split into
# 1. Lot/Plot details
# 2. Location details
# 3. Building whereabouts
# 4. Building structure details
# 5. Basement details
# 6. Temperature comfort details
# 7. Living space details
# 8. Exterior details

# Let's see, which all features, come into each category

# # 1. Lot/Plot details

# **It includes all the features which says about the plot, so, from the data, we've added below features:**
# 
# * MSSubClass: The building class
# * MSZoning: The general zoning classification
# * LotFrontage: Linear feet of street connected to property
# * LotArea: Lot size in square feet
# * Alley: Type of alley access
# * LotShape: General shape of property
# * LandContour: Flatness of the property
# * Utilities: Type of utilities available
# * LotConfig: Lot configuration
# * LandSlope: Slope of property

# **We create a dataframe with these features, and for analysis purpose we add salesprice to all the seperate dataframes we create**

# In[8]:


#Creating a dataframe for the plot analysis
df_plot = df[['MSSubClass','MSZoning','LotFrontage','LotArea','Street','Alley', 'LotShape','LandContour','LotConfig',
                'LandSlope','SalePrice']]


# As per our common intuition, Lot Area feature will be important, but from our correlation plot, Lot frontage also has a impact on SalePrice. So let's plot and explore

# In[9]:


df_plot.head()


# In[10]:


df_plot.isna().sum()


# ### Exploring the Categorical variables

# In[11]:


#Seperating categorical columns from the data
cat_col = df_plot.loc[:,df_plot.dtypes==np.object].columns
cat_col


# In[12]:


df_plot.describe(include=['O'])


# In[13]:


#Let's do some plotting on the categorical features
for col in cat_col:
    fig, axes = plt.subplots(figsize=(5,3), dpi=150)
    sns.barplot(data=df_plot, x=col, y='SalePrice', palette ='husl')


# **From the plots, 'LotConfig' seems to have a less say, in terms of sales price, and hence we drop that feature from the original dataset**
# 
# *Note: We will use the split feature dataset only to visualize the data better, while we do the tranformations to the original dataset*

# In[14]:


df.drop('LotConfig', axis=1, inplace=True)
df.drop('Alley', axis=1, inplace=True)


# **We drop the 'alley' feature also, as most number of homes does not have an alley, and hence would not contribute much to our cause**

# In[15]:


#Exploring the correlation of numerical features
plt.figure(figsize=(5,3), dpi=150)
sns.heatmap(df_plot.corr(), annot=True)


# In[16]:


fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(9,4), dpi=150)
ax[0].scatter(x = df_plot['LotFrontage'], y = df['SalePrice'])
ax[1].scatter(x = df_plot['LotArea'], y = df['SalePrice'])
ax[0].set_ylabel('SalePrice', fontsize=13)
ax[0].set_xlabel('LotFrontage', fontsize=13)
ax[1].set_xlabel('LotArea', fontsize=13)
fig.subplots_adjust(wspace=0.3)
plt.show()


# **LotFrontage and LotArea seems to have a good relationship with the Salesprice, as implied by the correlation matrix**

# **But, we have 259 values missing from the Lot Frontage, and we replace that with the median of the values.**
# 
# 
# *Note: We use 'median instead of 'mean' as we have outlier values, and mean is too sensitive to outliers*

# In[17]:


df['LotFrontage'].fillna(df['LotFrontage'].median(), inplace=True)


# # 2. Location details

# * **Neighborhood**: Physical locations within Ames city limits
# * **Condition1**: Proximity to main road or railroad
# * **Condition2**: Proximity to main road or railroad (if a second is present)
# * **Utilities**: Type of utilities available

# In[18]:


df_location = df[['Neighborhood','Condition1','Condition2','Utilities','SalePrice']]


# Neighborhood should be the most important feature in terms of location, lets explore that

# In[19]:


plt.figure(figsize=(8,3), dpi=150)
sns.barplot(data=df_location, x='Neighborhood', y='SalePrice')
plt.xticks(rotation=90);


# **Northridge, Northridge Heights and Stone Brook are the key hotspots in the city, and obviously the prices are high**

# **To negate any outlier influence, lets try grouping them as median of Salesprice**

# In[20]:


df.groupby('Neighborhood')['SalePrice'].median().sort_values(ascending=False)


# **And this also shows nothing different as these 3 places remain the hotspot in Ames city**

# In[21]:


df_location.isna().sum()


# **We dont have any missing values here!**

# # 3. Building whereabouts

# * BldgType: Type of dwelling
# * HouseStyle: Style of dwelling
# * OverallQual: Overall material and finish quality
# * OverallCond: Overall condition rating
# * YearBuilt: Original construction date
# * YearRemodAdd: Remodel date

# In[22]:


df_building = df[['BldgType','HouseStyle','OverallQual','OverallCond','YearBuilt','YearRemodAdd','SalePrice']]


# **From our correlation plot, Overall Quality and Year built had the highest correlation, let's visualize and explore**

# In[23]:


df_building.head()


# In[24]:


sns.barplot(data=df_building, x='OverallQual', y='SalePrice')


# **That's a perfect correlation!**

# **Now, let's see if we can get such a perfection in year built and year remodelled**

# In[25]:


fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(10,4), dpi=150)
ax[0].scatter(x = df_building['YearBuilt'], y = df_building['SalePrice'])
ax[1].scatter(x = df_building['YearRemodAdd'], y = df_building['SalePrice'])
ax[0].set_ylabel('SalePrice', fontsize=13)
ax[0].set_xlabel('YearBuilt', fontsize=13)
ax[1].set_xlabel('YearRemodAdd', fontsize=13)
fig.subplots_adjust(wspace=0.3)
plt.show()


# **The plot looks good, with a sudden increase in salesprice close to 2000 and further**

# In[26]:


df_building.isna().sum()


# **No missing values here too!**

# # 4. Building structure details

# * RoofStyle: Type of roof
# * RoofMatl: Roof material
# * Exterior1st: Exterior covering on house
# * Exterior2nd: Exterior covering on house (if more than one material)
# * MasVnrType: Masonry veneer type
# * MasVnrArea: Masonry veneer area in square feet
# * ExterQual: Exterior material quality
# * ExterCond: Present condition of the material on the exterior
# * Foundation: Type of foundation

# In[27]:


df_structure = df[['RoofStyle', 'RoofMatl', 'Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual','ExterCond',
                  'Foundation','SalePrice']]
df_structure.head()


# In[28]:


#Checking for missing values
df_structure.isna().sum()


# In[29]:


cat_cols = df_structure[df_structure.columns[df_structure.dtypes=='object']]
cat_cols


# In[30]:


for col in cat_cols.columns:
    plt.figure(figsize=(5,3), dpi=150)
    sns.barplot(data=df,x=col, y='SalePrice')
    plt.xticks(rotation=90)


# **Let's explore the Masonry veneer area, which has a decent correlation with Salesprice**

# In[31]:


plt.figure(figsize=(7,4), dpi=150)
sns.scatterplot(data=df_structure, x='MasVnrArea', y='SalePrice')


# In[32]:


df_structure['MasVnrType'].value_counts()


# In[33]:


#Filling MasVnrType as 'None' for most values
df['MasVnrType'].fillna(df['MasVnrType'].mode()[0], inplace=True)


# In[34]:


#Filling missing MasVnrArea with its median
df['MasVnrArea'].fillna(df['MasVnrArea'].median(), inplace=True)


# # 5. Basement details

# * BsmtQual: Height of the basement
# * BsmtCond: General condition of the basement
# * BsmtExposure: Walkout or garden level basement walls
# * BsmtFinType1: Quality of basement finished area
# * BsmtFinSF1: Type 1 finished square feet
# * BsmtFinType2: Quality of second finished area (if present)
# * BsmtFinSF2: Type 2 finished square feet
# * BsmtUnfSF: Unfinished square feet of basement area
# * TotalBsmtSF: Total square feet of basement area
# * BsmtFullBath: Basement full bathrooms
# * BsmtHalfBath: Basement half bathrooms

# In[35]:


df_basement = df[['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2',
                  'BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','SalePrice']]
df_basement.head()


# In[36]:


fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(9,3), dpi=150)
ax[0].scatter(x = df_basement['TotalBsmtSF'], y = df_basement['SalePrice'])
ax[1].scatter(x = df_basement['BsmtFinSF1'], y = df_basement['SalePrice'])
ax[0].set_ylabel('SalePrice', fontsize=13)
ax[0].set_xlabel('TotalBsmtSF', fontsize=13)
ax[1].set_xlabel('BsmtFinSF1', fontsize=13)
fig.subplots_adjust(wspace=0.3)
plt.show()


# **The total basement area has a good correlation with Salesprice, and hence that becomes the most important feature in terms of basement, along with BsmtFinSF1**

# In[37]:


df_basement.isna().sum()


# In[38]:


#We replace basement quality and basement condition to 'None' for No basement
df['BsmtQual'].fillna('None', inplace=True)
df['BsmtCond'].fillna('None', inplace=True)

#Removing the remaining columns, which we feel will not be of significance
df['BsmtExposure'].fillna('None', inplace=True)
df.drop(['BsmtFinType1','BsmtFinType2','BsmtFinSF2','BsmtHalfBath'], axis=1, inplace=True)


# # 6. Comfort

# * Heating: Type of heating
# * HeatingQC: Heating quality and condition
# * CentralAir: Central air conditioning
# * Electrical: Electrical system
# * Fireplaces: Number of fireplaces
# * FireplaceQu: Fireplace quality

# **This section deals with the thermal comfort aspect regarding the house, and let's see how that will affect the salesprice of the house**

# In[39]:


df_comfort = df[['Heating','HeatingQC','CentralAir','Electrical','Fireplaces','FireplaceQu','SalePrice']]
df_comfort.head()


# In[40]:


df_comfort.isna().sum()


# In[41]:


for col in df_comfort.drop(['Fireplaces','SalePrice'], axis=1).columns:
    plt.figure(figsize=(5,3), dpi=150)
    sns.barplot(data=df_comfort,x=col, y='SalePrice')
    plt.xticks(rotation=90)


# Heating Quality, and air conditioning have good impact on the salesprice, no wonder thermal comfort is of utmost importance!

# In[42]:


#Replacing the missing values of Quality as 'None' as fireplaces are missing from the remaining homes
df['FireplaceQu'].fillna('None', inplace=True)


# In[43]:


#Filling the 'Electrical' missing value with the mode-Standard Circuit Breakers & Romex
df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)


# # 7. Living space details

# * 1stFlrSF: First Floor square feet
# * 2ndFlrSF: Second floor square feet
# * LowQualFinSF: Low quality finished square feet (all floors)
# * GrLivArea: Above grade (ground) living area square feet
# * FullBath: Full bathrooms above grade
# * HalfBath: Half baths above grade
# * Bedroom: Number of bedrooms above basement level
# * Kitchen: Number of kitchens
# * KitchenQual: Kitchen quality
# * TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# * Functional: Home functionality rating

# **Now comes one of the most important aspects while purchasing a home, the living space, the interior. Let's explore this in detail, as we expect some deep insights from here**

# In[44]:


df_livspace = df[['1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','FullBath','HalfBath','KitchenQual',
                  'TotRmsAbvGrd','Functional','SalePrice']]
df_livspace.head()


# **We have already seen the visualizations for the Living area square feet, so we will explore the remaining features**
# 
# **Let's dive deep into some dimensional features, and see how it can help us**

# In[45]:


fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(9,3), dpi=150)
ax[0].scatter(x = df_livspace['1stFlrSF'], y = df_livspace['SalePrice'])
ax[1].scatter(x = df_livspace['2ndFlrSF'], y = df_livspace['SalePrice'])
ax[0].set_ylabel('SalePrice', fontsize=13)
ax[0].set_xlabel('1stFlrSF', fontsize=13)
ax[1].set_xlabel('2ndFlrSF', fontsize=13)
fig.subplots_adjust(wspace=0.3)
plt.show()


# **Of the features, low quality finished sq feet, doesn't provide any valuable insight as most values are zero, hence we drop that feature**

# In[46]:


#Dropping the feature
df.drop('LowQualFinSF', axis=1, inplace=True)


# In[47]:


#plt.figure(fig)
sns.barplot(data=df_livspace, x='KitchenQual', y='SalePrice')


# In[48]:


sns.barplot(data=df_livspace, x='FullBath', y='SalePrice')


# **Kitchen Quality and Full bathrooms are of utmost priority too, while buying a house!!**

# **Again, we drop 'HalfBath' feature, as the correlation is too less for that, with Saleprice**

# In[49]:


#Dropping the feature
df.drop('HalfBath', axis=1, inplace=True)


# In[50]:


df_livspace.isna().sum()


# There are no missing values here...

# # 8. Exterior

# * GarageType: Garage location
# * GarageYrBlt: Year garage was built
# * GarageFinish: Interior finish of the garage
# * GarageCars: Size of garage in car capacity
# * GarageArea: Size of garage in square feet
# * GarageQual: Garage quality
# * GarageCond: Garage condition
# * PavedDrive: Paved driveway
# * WoodDeckSF: Wood deck area in square feet
# * OpenPorchSF: Open porch area in square feet
# * EnclosedPorch: Enclosed porch area in square feet
# * 3SsnPorch: Three season porch area in square feet
# * ScreenPorch: Screen porch area in square feet
# * PoolArea: Pool area in square feet
# * PoolQC: Pool quality
# * Fence: Fence quality
# * MiscFeature: Miscellaneous feature not covered in other categories

# In[51]:


df_exterior = df[['GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond',
                 'PavedDrive','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea',
                 'PoolQC','Fence','MiscFeature','SalePrice']]
df_exterior.head()


# In[52]:


fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(9,6), dpi=150)
ax[0][0].scatter(x = df_exterior['GarageArea'], y = df_exterior['SalePrice'])
ax[0][1].scatter(x = df_exterior['PoolArea'], y = df_exterior['SalePrice'])
ax[1][0].scatter(x = df_exterior['GarageYrBlt'], y = df_exterior['SalePrice'])
ax[1][1].scatter(x = df_exterior['GarageCars'], y = df_exterior['SalePrice'])
ax[0][0].set_ylabel('SalePrice', fontsize=13)
ax[0][0].set_xlabel('GarageArea', fontsize=13)
ax[0][1].set_xlabel('PoolArea', fontsize=13)
ax[1][0].set_xlabel('GarageYrBlt', fontsize=13)
ax[1][1].set_xlabel('GarageCars', fontsize=13)
fig.subplots_adjust(wspace=0.3,hspace=0.3)
plt.show()


# Garage area and Garage cars have a good correlation as per our original correlation plot, in most cases. Garage Year built and pool area doesn' seem to contribute much

# In[53]:


df_exterior.isna().sum()


# **Pool quality, Fence, Misc Features are missing from most of the observations, maybe these facilities are absent in most of the houses we covered, hence we drop these features**
# 
# **Pool Area also have lots of values as 0, and hence would not contribute to our cause, we will drop that too.**

# In[54]:


df.drop(['MiscFeature','Fence','PoolArea','PoolQC','GarageYrBlt'], axis=1,inplace=True)


# **Let's fill the remaining missing values with None, implying no garages**

# In[55]:


df['GarageType'].fillna('None',inplace=True)
df['GarageFinish'].fillna('None',inplace=True)
df['GarageQual'].fillna('None',inplace=True)
df['GarageCond'].fillna('None',inplace=True)


# **We are also dropping some columns from the dataset, which we feel will have minimum impact**

# In[56]:


df.drop(['MiscVal','MoSold','YrSold','SaleType','Id'], axis=1, inplace=True)


# In[57]:


#Checking for any missing values in our dataset
df.isna().sum().sort_values(ascending=False)[:5]


# **And the last step would be to do a label encoding to the categorical features**

# In[58]:


cat_columns = df[df.columns[df.dtypes=='object']]

from sklearn.preprocessing import LabelEncoder
for i in cat_columns:
    label = LabelEncoder()
    label.fit(df[i].values)
    df[i] = label.transform(df[i].values)


# # Skewness in Target

# **One final thing to check before proceeding to the predictions would be to check the skewness of the target variable. We will have a check and try to use logarithmic transformation**

# In[59]:


plt.figure(figsize=(6,3), dpi=150)
sns.distplot(df['SalePrice'] , fit=norm);

# The fitted parameters used by the function
(mu, sigma) = norm.fit(df['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Plot the QQ-plot
fig = plt.figure(figsize=(6,3), dpi=150)
res = stats.probplot(df['SalePrice'], plot=plt)
plt.show()


# As per the plots, we can see a right skew to the target variable. Let's do a log transformation on the variable to make it normally distributed 

# In[60]:


#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
df["SalePrice"] = np.log1p(df["SalePrice"])

#Check the new distribution 
plt.figure(figsize=(6,3), dpi=150)
sns.distplot(df['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure(figsize=(6,3), dpi=150)
res = stats.probplot(df['SalePrice'], plot=plt)
plt.show()


# **The data now seems like a normally distributed one**
