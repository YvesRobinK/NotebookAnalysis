#!/usr/bin/env python
# coding: utf-8

# # **House Prices Advanced Regression Techniques:**
# 
# #### **If you find the notebook useful , feel free to upvote**

# ## **Importing necessary libraries**

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')


# ### **Loading Data**

# In[2]:


train_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv",index_col=0)
test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv",index_col=0)
sample_sub = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")


# In[3]:


train_data.head()


# In[4]:


test_data.head()


# #### **Describing the data**

# In[5]:


print("No of rows and columns in training data:",train_data.shape)
print("No of rows and columns in testing data:",test_data.shape)


# In[6]:


#creating a individual variable to store the target feature
y = train_data[['SalePrice']]


# In[7]:


train_data.info()


# ### **Splitting into numerical and categorical variables:**

# In[8]:


n_val = train_data.select_dtypes(exclude=['object']).drop(['MSSubClass'], axis=1).copy()
print(n_val.columns)
l=[]
for i in n_val:
  l.append(i)
print("Total no of numerical variables:",len(l))


# In[9]:


c_val = train_data.select_dtypes(include=['object']).copy()
c_val['MSSubClass'] = train_data['MSSubClass']
print(c_val.columns)
l=[]
for i in c_val:
  l.append(i)
print("Total no of categorical variables:",len(l))


# ## **Univariate Analysis:**
# 
# ### **<u>Numerical variables:</u>**

# In[10]:


fig = plt.figure(figsize=(18,16))
for index,col in enumerate(n_val):
    plt.subplot(6,6,index+1)
    sns.distplot(n_val.loc[:,col].dropna(), kde=False,color='r')
fig.tight_layout(pad=1.0)


# From the given plots we can tell that certain numerical variables have only one kind of value:
# 
# 
# -  **BsmtFinSF2**
# -  **EnclosedPorch**
# - **LowQualFinSF**
# -  **3SsnPorch**
# -  **ScreenPorch**
# - **PoolArea**
# -  **MiscVal**
# 
# Most of the values present in these variables are 0, thus these features can be cleared during the data preprocessing step.
# 
# 

# ### **Categorical Variables:**

# In[11]:


fig = plt.figure(figsize=(18,20))
for index in range(len(c_val.columns)):
    plt.subplot(9,5,index+1)
    sns.countplot(x=c_val.iloc[:,index], data=c_val.dropna())
    plt.xticks(rotation=90)
fig.tight_layout(pad=1.2)


# From the given plots we can tell that certain categorical features have only one kind of value:
# 
# 
# *   **Condition2**
# *   **Heating**
# *   **RoofMatl**
# *   **Utilites**
# *   **Functional**
# 
# 
# These consist of only one value hence we can remove these features in the data preprocessing step
# 
# 

# ### **Bivariate Analysis:**

# In[12]:


plt.figure(figsize=(14,12))
correlation = n_val.corr()
sns.heatmap(correlation, mask = correlation <0.8, linewidth=0.5)


# From the given correlation matrix we can infer the highly correlated features:
# 
# -  **GarageYrBlt and YearBuilt**
# - **TotRmsAbvGrd and GrLivArea**
# - **1stFlrSF and TotalBsmtSF**
# - **GarageArea and GarageCars**
# 
# 
# 
# 

# We need to find the numerical features which have high correlation to the target variable 'SalePrice'

# In[13]:


nfeatures_corr = n_val.corr()['SalePrice'][:-1]
high_fealist = nfeatures_corr[abs(nfeatures_corr)> 0.5].sort_values(ascending=False)
print("HIGHLY CORRELATED FEATURES:\n")
print(high_fealist)


# Finding the relation between the numerical features and the target variables:

# In[14]:


fig = plt.figure(figsize=(20,20))
for index in range(len(n_val.columns)):
    plt.subplot(10,5,index+1)
    sns.scatterplot(x=n_val.iloc[:,index], y='SalePrice', data=n_val.dropna())
fig.tight_layout(pad=1.0)


# ### **Data Processing:**
# 
# We will first drop the highly correlated features which we already found out

# In[15]:


train_data.drop(['GarageYrBlt','TotRmsAbvGrd','1stFlrSF','GarageCars'], axis=1, inplace=True)


# From the above scatterplots for various features we can see that
# - MoSold
# - YrSold
# 
# does not have any linear relationship with the target variable hence we can drop them

# In[16]:


train_data.drop(['MoSold','YrSold'], axis=1, inplace=True)


# Dropping the features with consists of lots of missing values cause they are redundant

# In[17]:


plt.figure(figsize=(25,8))
plt.title('Number of missing rows')
missing_count = pd.DataFrame(train_data.isnull().sum(), columns=['sum']).sort_values(by=['sum'],ascending=False).head(18).reset_index()
missing_count.columns = ['features','sum']
sns.barplot(x='features',y='sum', data = missing_count)


# From the given distribution we can infer that the features named:
# *   PoolQC
# *   MiscFeature
# - Alley
# 
# consists of lots of missing values and hence can be removed
# 
# 

# In[18]:


train_data.drop(['PoolQC','MiscFeature','Alley'], axis=1, inplace=True)


# ### **Removing Constant features:**
# 
# We will be dropping both
# - Numerical Variables
# - Categorical Variables
# 
# which consists of large number of one kind of value's or 0.

# In[19]:


#for numerical variables
nval_col = train_data.select_dtypes(exclude=['object']).drop(['MSSubClass'], axis=1).columns
overfit_nvals = []
for i in nval_col:
    counts = train_data[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(train_data) * 100 > 95:
        overfit_nvals.append(i)
overfit_nvals = list(overfit_nvals)
train_data = train_data.drop(overfit_nvals, axis=1)


# In[20]:


#for categorical variables
cval_col = train_data.select_dtypes(include=['object']).columns
overfit_cvals = []
for i in cval_col:
    counts = train_data[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(train_data) * 100 > 95:
        overfit_cvals.append(i)
overfit_cvals = list(overfit_cvals)
train_data = train_data.drop(overfit_cvals, axis=1)


# In[21]:


#printing constant features
print("Dropped the following Constant numerical features:\n",overfit_nvals) 
print("Dropped the following Constant categorical features:\n",overfit_cvals) 


# ### **Dealing with outliers:**

# In[22]:


#checking the total amount of null values in columns present in the training data
train_data.isnull().sum().T


# In[23]:


#the total number of numerical features after dropping certain features based on certain criteria
print(len(nval_col))


# Creating box-plots to find the outliers for the numerical features

# In[24]:


nval_col1 = train_data.select_dtypes(exclude=['object']).drop(['MSSubClass'], axis=1).copy()
fig = plt.figure(figsize=(14,15))
for index,col in enumerate(nval_col1):
    plt.subplot(5,6,index+1)
    sns.boxplot(nval_col1.loc[:,col].dropna())
fig.tight_layout(pad=1.0)


# From the given boxplots , we can infer that certain numerical features have extreme outliers thus we would be removing those outliers.

# In[25]:


train_data = train_data.drop(train_data[train_data['LotFrontage'] > 200].index)
train_data = train_data.drop(train_data[train_data['LotArea'] > 100000].index)
train_data = train_data.drop(train_data[train_data['BsmtFinSF1'] > 4000].index)
train_data = train_data.drop(train_data[train_data['TotalBsmtSF'] > 5000].index)
train_data = train_data.drop(train_data[train_data['GrLivArea'] > 4000].index)
train_data = train_data.drop(train_data[train_data['Fireplaces'] > 2.5].index)


# ### **Missing values:**

# #### Finding the missing values:

# In[26]:


#find the total no of missing values
train_data.isnull().sum().sum()


# In[27]:


#finding the missing values in features
missval = train_data.isnull().sum()
missval = missval[missval>0]
missval.sort_values(ascending=False)


# In[28]:


#view of the total number of missing values and the percentage of missing values in each column
total = train_data.isnull().sum().sort_values(ascending=False)
percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(15)


# In[29]:


print('Features size:', train_data.shape)


# ### **Imputing the missing values:**

# **Ordinal Features:**
# Replacing the missing values with 'NA'

# In[30]:


#we are filling
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual',
            'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','Fence','FireplaceQu','MasVnrType']:
    train_data[col] = train_data[col].fillna('NA')


# **Nominal Features:**
# Replacing the missing value of the categorical features with the most frequently occuring values

# In[31]:


cols = ["MasVnrArea","MasVnrType", "Exterior1st", "Exterior2nd", "SaleType", "Electrical"]
train_data[cols] = train_data.groupby("Neighborhood")[cols].transform(lambda x: x.fillna(x.mode()[0]))


# **Numerical Feature:**
# Replacing the missing values with the mean or median depending upon the distribution of the variable

# In[32]:


sns.distplot(train_data['LotFrontage'])


# From the given distribution we are able to infer that the distribution is slightly right skewed and hence the missing values of the feature 'LotFrontage' can be replaced with the median of the values.

# In[33]:


train_data['LotFrontage'] = train_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))


# In[34]:


#checking the null values in dataset after imputation
total = train_data.isnull().sum().sort_values(ascending=False)
percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data


# ### **Feature Engineering:**

# **Chaning the datatype of the feature MSSubClass:**

# In[35]:


train_data['MSSubClass'] = train_data['MSSubClass'].apply(str)


# **Mapping ordinal features:**

# In[36]:


ordinal_map = {'Ex': 5,'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA':0}
fin_map = {'GLQ': 6,'ALQ': 5,'BLQ': 4,'Rec': 3,'LwQ': 2,'Unf': 1, 'NA': 0}
exposure_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}
fence_map = {'GdPrv': 4,'MnPrv': 3,'GdWo': 2, 'MnWw': 1,'NA': 0}


# In[37]:


ord_col = ['ExterQual','ExterCond','BsmtQual', 'BsmtCond','HeatingQC','KitchenQual','GarageQual','GarageCond', 'FireplaceQu']
for col in ord_col:
    train_data[col] = train_data[col].map(ordinal_map)
    
fin_col = ['BsmtFinType1','BsmtFinType2']
for col in fin_col:
    train_data[col] = train_data[col].map(fin_map)
    
train_data['BsmtExposure'] = train_data['BsmtExposure'].map(exposure_map)
train_data['Fence'] = train_data['Fence'].map(fence_map)


# **Converting Categorical to Numerical Features:**

# In[38]:


train_data = pd.get_dummies(train_data)


# **Distribution of Saleprice:**

# In[39]:


plt.figure(figsize=(10,6))
plt.title("Distribution of SalePrice")
dist = sns.distplot(train_data['SalePrice'],norm_hist=False)


# * From the above plot we can infer that the target variable has a skewed distribution ,which would affect the model.
# * Hence we would apply log transformation on the target variable thus reducing the skewness of the distribution.

# In[40]:


plt.figure(figsize=(10,6))
plt.title("After transformation of SalePrice")
dist = sns.distplot(np.log(train_data['SalePrice']),norm_hist=False)


# In[41]:


y["SalePrice"] = np.log(y['SalePrice'])


# #### **Well Thanks for viewing the notebook and if you find the notebook useful ,feel free to upvote!**
