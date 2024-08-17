#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Importing Libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings   # remove all warnings from the output
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 500)  ## this will display all the columns in the output of .csv file
pd.set_option("display.max_rows", 500)


# ##### This is a helper function which plot countplot and percentage of null values available in a feature.

# In[3]:


def null(df, feature, plot=False):
    t = df[feature].isna().mean() * 100
    print(f'% of null --> {t}')
    if plot:
        sns.countplot(dataset[feature], palette='cool')


# In[4]:


train_path = '../input/house-prices-advanced-regression-techniques/train.csv'
test_path = "../input/house-prices-advanced-regression-techniques/test.csv"


# ## Extrapolatory Data Analysis

# ### Looking into the data

# In[5]:


train_data = pd.read_csv(train_path)
train_data.head()


# In[6]:


train_data.info()


# In[7]:


dataset = train_data.copy()  # copying the data, so that if bymistakely we did something wrong which cannot be revert,
                             # then we can simply run this cell and we get back our origional data.
dataset.tail()


# ### Looking for null values

# In[8]:


dataset.isnull().sum()


# #### features which contains null values

# In[9]:


null_feat = [i for i in dataset.columns if dataset[i].isnull().sum() != 0]
null_feat


# In[10]:


for i in null_feat:
    print(f"{i} --> {dataset[i].unique()}")


# ## Feature Engineering

# ### Looking into each feature containing null values

# ### LotFrontage

# In[11]:


dataset['LotFrontage'].fillna(dataset['LotFrontage'].mean(), inplace=True)
dataset['LotFrontage'].isnull().sum()


# ### Alley

# In[12]:


dataset['Alley'].value_counts()


# In[13]:


null(dataset, 'Alley', True) ## --> 93.7671% are missing.


# ### Drop this column is good bcz it is 93% null

# In[14]:


dataset.drop('Alley', axis=1, inplace = True)


# ### MasVnrType

# In[15]:


null(dataset, 'MasVnrType', True)


# ### put nan as 'None'

# In[16]:


dataset['MasVnrType'].fillna('None', inplace=True)


# ### MasVnrArea

# In[17]:


dataset['MasVnrArea'].unique()


# In[18]:


null(dataset, 'MasVnrArea')


# In[19]:


dataset['MasVnrArea'].fillna(dataset['MasVnrArea'].mean(), inplace=True)


# ### BsmtQual

# In[20]:


null(dataset, 'BsmtQual', True)


# In[21]:


dataset['BsmtQual'].fillna('TA', inplace=True)


# ### BsmtCond

# In[22]:


null(dataset, 'BsmtCond', True)


# In[23]:


dataset['BsmtCond'].fillna('TA', inplace=True)


# In[24]:


null_feat


# ### BsmtExposure

# In[25]:


null(dataset, 'BsmtExposure', True)


# In[26]:


dataset['BsmtExposure'].fillna('No', inplace=True)


# ### BsmtFinType1

# In[27]:


null(dataset, 'BsmtFinType1', True)


# In[28]:


dataset['BsmtFinType1'].fillna('Unf', inplace=True)


# ### 'BsmtFinType2'

# In[29]:


null(dataset, 'BsmtFinType2', True)


# In[30]:


dataset['BsmtFinType2'].fillna('Unf', inplace=True)


# ### 'Electrical'

# In[31]:


null(dataset, 'Electrical', True)


# In[32]:


dataset['Electrical'].unique()


# In[33]:


dataset['Electrical'].fillna('SBrkr', inplace=True)


# ### FireplaceQu

# In[34]:


null(dataset, "FireplaceQu", True)


# In[35]:


dataset['FireplaceQu'].fillna('None', inplace=True)


# ### GarageType

# In[36]:


null(dataset, 'GarageType', True)


# In[37]:


val = dataset['GarageType'].unique()[0]
dataset['GarageType'].fillna(val, inplace=True)


# ### GarageYrBlt

# In[38]:


null(dataset, 'GarageYrBlt')


# In[39]:


plt.figure(figsize=(18,8))
sns.countplot(dataset['GarageYrBlt'], palette='rainbow')
plt.xticks(rotation=45)
plt.show()


# In[40]:


dataset['GarageYrBlt'].fillna('2005.0', inplace= True)


# ### GarageFinish

# In[41]:


null(dataset, 'GarageFinish', True)


# In[42]:


dataset['GarageFinish'].fillna(dataset['GarageFinish'].unique()[1], inplace=True)


# ### GarageQual

# In[43]:


null(dataset, 'GarageQual', True)


# In[44]:


dataset['GarageQual'].fillna(dataset['GarageQual'].unique()[0], inplace=True)


# ### GarageCond

# In[45]:


null(dataset, 'GarageCond', True)


# In[46]:


dataset['GarageCond'].fillna(dataset['GarageCond'].unique()[0], inplace=True)


# ### PoolQC

# In[47]:


null(dataset, 'PoolQC', True)


# #### droping...

# In[48]:


dataset.drop('PoolQC', axis=1, inplace= True)


# ### Fence

# In[49]:


null(dataset, 'Fence', True)


# In[50]:


dataset['Fence'].fillna('None', inplace=True)


# ### MiscFeature

# In[51]:


null(dataset, 'MiscFeature', True)


# #### droping...

# In[52]:


dataset.drop('MiscFeature', axis=1, inplace=True)


# In[53]:


dataset.head(2)


# In[54]:


dataset.shape


# In[55]:


dropped_col = list(set(train_data.columns) - set(dataset.columns))
dropped_col


# ## dropped columns are : ['Alley', 'MiscFeature', 'PoolQC']

# In[56]:


plt.scatter(dataset['LotFrontage'], dataset['SalePrice'])


# In[57]:


plt.scatter(dataset['MasVnrArea'], dataset['SalePrice'])


# In[58]:


plt.scatter(dataset['GarageCond'], dataset['SalePrice'])


# In[59]:


dataset['GarageCond'].value_counts()


# #### basically we have two types of columns 1.Numerical 2.Categorical.
# #### putting all numerical columns into a list

# In[60]:


num_feat = [i for i in dataset.columns if dataset[i].dtypes != 'O']
len(num_feat)


# In[61]:


dataset[set(dataset.columns) - set(num_feat)].head(2)


# In[62]:


dataset[set(dataset.columns) - set(num_feat)].dtypes


# In[63]:


dataset['GarageYrBlt'] = dataset['GarageYrBlt'].astype(float)
dataset['GarageYrBlt'] = dataset['GarageYrBlt'].astype(int)


# #### Time to work with categorical data

# In[64]:


cat_feat = dataset[set(dataset.columns) - set(num_feat)]


# In[65]:


dataset[num_feat].head(2)


# In[66]:


float_feat = [i for i in num_feat if dataset[i].dtypes == 'float']
float_feat


# In[67]:


for i in cat_feat:
    print(i)
    print(dataset[i].unique())
    print("------------------------------------")


# ## Encoding categorical data
# ##### Beacause machine learning algorithm are compatible with numerical data only.

# In[68]:


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
for i in cat_feat:
    dataset[i] = label.fit_transform(dataset[i])


# In[69]:


dataset.head(2)


# In[70]:


dataset.dtypes


# In[71]:


dataset.to_csv('final_dataset',header=True, index=False)


# In[72]:


final_dataset = pd.read_csv('./final_dataset')
final_dataset.head(2)


# ## Variable Seperation

# In[73]:


X_train = final_dataset.drop('SalePrice', axis=1)
y_train = final_dataset['SalePrice']


# ## Scaling the data

# In[74]:


columns = X_train.columns
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scaled = scale.fit(X_train)
X_train_scaled = pd.DataFrame(scale.transform(X_train), columns=columns)


# In[75]:


X_train_scaled.head(2)


# ## Working with Test Data

# In[76]:


test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test_data.head(2)


# In[77]:


test_data.drop(dropped_col, axis=1, inplace=True)


# In[78]:


label = LabelEncoder()
for i in cat_feat:
    test_data[i] = label.fit_transform(test_data[i])


# In[79]:


test_data.head()


# In[80]:


test_data['LotFrontage'].fillna(test_data['LotFrontage'].mean(), inplace=True)

test_data['MasVnrType'].fillna('None', inplace=True)

test_data['MasVnrArea'].fillna(test_data['MasVnrArea'].mean(), inplace=True)

test_data['BsmtQual'].fillna('TA', inplace=True)

test_data['BsmtCond'].fillna('TA', inplace=True)

test_data['BsmtExposure'].fillna('No', inplace=True)

test_data['BsmtFinType1'].fillna('Unf', inplace=True)

test_data['BsmtFinType2'].fillna('Unf', inplace=True)

test_data['Electrical'].fillna('SBrkr', inplace=True)

test_data['FireplaceQu'].fillna('None', inplace=True)

val = test_data['GarageType'].unique()[0]
test_data['GarageType'].fillna(val, inplace=True)

test_data['GarageYrBlt'].fillna('2005.0', inplace= True)

test_data['GarageFinish'].fillna(test_data['GarageFinish'].unique()[1], inplace=True)

test_data['GarageQual'].fillna(test_data['GarageQual'].unique()[0], inplace=True)

test_data['GarageCond'].fillna(test_data['GarageCond'].unique()[0], inplace=True)

test_data['Fence'].fillna('None', inplace=True)


# In[81]:


test_data.isnull().sum()


# In[82]:


test_null = [i for i in test_data.columns if test_data[i].isnull().sum() > 0]
test_null


# In[83]:


test_data[test_null].dtypes


# In[84]:


for i in test_null:
    test_data[i].fillna(test_data[i].mean(), inplace = True)


# In[85]:


test_data.isnull().sum()


# In[86]:


test_data.shape


# ## Model Fitting

# In[87]:


X_test = scale.fit(test_data)
X_test_scaled = pd.DataFrame(scale.transform(test_data), columns=columns)


# In[88]:


X_test_scaled.head(2)


# In[89]:


from sklearn.ensemble import AdaBoostRegressor
lin_reg = AdaBoostRegressor(random_state = 24, n_estimators = 35)
model = lin_reg.fit(X_train_scaled, y_train)
model.score(X_train_scaled, y_train)


# ## Submission

# In[90]:


sample = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
y_pred = model.predict(X_test_scaled)
result = pd.DataFrame({sample.columns[0] : sample['Id'],
                        sample.columns[1] : y_pred})
result.to_csv('submission.csv', index=False)


# In[ ]:




