#!/usr/bin/env python
# coding: utf-8

# ## Introduction: 
# In This kernel I will go to solve House Pricing with Advanced Regression Analysis.
# If there are any recommendations or changes you would like to see in my notebook, please leave a comment at the end of this kernel, I will be glad to answer any questions you may have in the comments. If you like this notebook, Please UPVOTE.

# ## What we want?
# 1. Gathering Data
# 2. Analysis the target and understand what is the important features
# 3. Looking for missing values
# 4. Feature Engineering
# 5. Converting categorical to numerical
# 6. Modeling

# In[1]:


## Import Libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sklearn.metrics as metrics
import math
from scipy.stats import norm, skew
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1. Gathering Data

# In[2]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



# In[3]:


train.shape , test.shape 


# In[4]:


train.head()


# ### 2. Let's know more about the Target and make some analysis
# You may wonder what the target is? 
# It's the 'SalePrice' column. 

# In[5]:


print(train['SalePrice'].describe())


# In[6]:


sns.distplot(train['SalePrice'], color= 'red')


# As we see, we have a positive sekew, we must fix it.

# In[7]:


print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())


# In[8]:


train['SalePrice'] = np.log1p(train['SalePrice'])
sns.distplot(train['SalePrice'], color= 'red', fit=norm, );


# Now we fixed it.

# ### Are we need a specialist or a broker to know what are the most important features that affect home prices?
# Of course not, we can know the important features by sea. So let's go and explore the data.
# 

# In[9]:


corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# Ok, now as you see the correlation between features.. The colours show to us the strong and weak correlation.
# But what we really need? we need the highest correlation between features and SalesPrice, so let's do it.

# In[10]:


corr = train.corr()
highest_corr_features = corr.index[abs(corr["SalePrice"])>0.5]
plt.figure(figsize=(10,10))
g = sns.heatmap(train[highest_corr_features].corr(),annot=True,cmap="RdYlGn")


# #### What we note?
# * It's important to know what you do and how benefit from it. We can see 'OverQual' in the top of highest correlation it's 0.79!
# * 'GarageCars' & 'GarageArea' like each other (correlation between them is 0.88) 
# * 'TotalBsmtSF' & '1stFlrSF' also like each other (correlation betwwen them is 0.82), so we can keep either one of them or add the1stFlrSF to the Toltal.
# * 'TotRmsAbvGrd' & 'GrLivArea' also has a strong correlation (0.83), I decided to keep 'GrLivArea' because it's correlation with 'SalePrice' is higher.
# 

# In[11]:


corr["SalePrice"].sort_values(ascending=False)


# #### ok let's focus on the features have highest correlation.

# In[12]:


cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols])


# Now, We explored the data and know the important features.

# ## 2. Looking for Missing Data 

# #### Before looking for Missing data: 
# We can concatenate train and test datasets, preprocess, and then divide them again. I think it will be easy for us.

# In[13]:


y_train = train['SalePrice']
test_id = test['Id']
all_data = pd.concat([train, test], axis=0, sort=False)
all_data = all_data.drop(['Id', 'SalePrice'], axis=1)


# In[14]:


Total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum() / all_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([Total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(25)


# Well, if we look at these features that have many missing values, we will note that they are not important features, none of them has (correlation > 0.5), so if we delete them we will not miss the data.

# In[15]:


all_data.drop((missing_data[missing_data['Total'] > 5]).index, axis=1, inplace=True)
print(all_data.isnull().sum().max())


# Let's show the features and the number of Missing values

# In[16]:


total = all_data.isnull().sum().sort_values(ascending=False)
total.head(19)


# ### Now, Filling the missing Data

# In[17]:


# filling the numeric data
numeric_missed = ['BsmtFinSF1',
                  'BsmtFinSF2',
                  'BsmtUnfSF',
                  'TotalBsmtSF',
                  'BsmtFullBath',
                  'BsmtHalfBath',
                  'GarageArea',
                  'GarageCars']

for feature in numeric_missed:
    all_data[feature] = all_data[feature].fillna(0)


# In[18]:


#filling categorical data
categorical_missed = ['Exterior1st',
                  'Exterior2nd',
                  'SaleType',
                  'MSZoning',
                   'Electrical',
                     'KitchenQual']

for feature in categorical_missed:
    all_data[feature] = all_data[feature].fillna(all_data[feature].mode()[0])


# In[19]:


#Fill in the remaining missing values with the values that are most common for this feature.

all_data['Functional'] = all_data['Functional'].fillna('Typ')


# In[20]:


all_data.drop(['Utilities'], axis=1, inplace=True)


# let's check if we have another missing values.

# In[21]:


all_data.isnull().sum().max() #just checking that there's no missing data missing...


# ## 4. Feature Engineering

# #### Fix The Skewness in the other features
# 

# In[22]:


numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skewed_feats[abs(skewed_feats) > 0.5]
high_skew


# In[23]:


for feature in high_skew.index:
    all_data[feature] = np.log1p(all_data[feature])


# #### Let's add a new features

# In[24]:


all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']



# ## 5. Converting the categorical to numerical.

# In[25]:


all_data = pd.get_dummies(all_data)
all_data.head()


# #### We cleaned the data very well, and now let's separate the data to its origin (train, test)

# In[26]:


x_train =all_data[:len(y_train)]
x_test = all_data[len(y_train):]


# In[27]:


x_test.shape , x_train.shape


# ## 5. Apply ML Model

# In[28]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(x_train, y_train)


# In[29]:


y_pred = np.floor(np.expm1(regressor.predict(x_test)))
y_pred


# In[30]:


sub = pd.DataFrame()
sub['Id'] = test_id
sub.to_csv('mysubmission.csv',index=False)

