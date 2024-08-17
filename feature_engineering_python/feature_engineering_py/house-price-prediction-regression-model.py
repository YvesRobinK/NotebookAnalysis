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


# In[2]:


#Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

pd.pandas.set_option('display.max_columns',None)


# In[3]:


data=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
data.head()


# In[4]:


data.describe()


# In[5]:


#Identifying columns with missing values


# In[6]:


features_nan=[features for features in data.columns if data[features].isnull().sum()>1]

for features in features_nan:
    print(features," has ", np.round((data[features].isnull().mean())*100,4),"% missing values")


# In[7]:


# Identifying the relationship between missing values and target variable

for features in features_nan:
    data_new=data.copy()
    
    data_new[features]=np.where(data_new[features].isnull(),1,0)
    
    data_new.groupby(features)['SalePrice'].mean().plot.bar()
    plt.title(features)
    plt.show()


# In[8]:


#removing columns that are not required
#data=data.drop('Id',axis=1,inplace=True)


# In[9]:


#identifying numerical features
features_num = [features for features in data.columns if data[features].dtypes != 'O']
print(len(features_num))
data[features_num].head()


# In[10]:


#Temporal Variable (Eg: Datetime)
features_yr=[features for features in features_num if 'Yr' in features or 'Year' in features]
features_yr


# In[11]:


#exploring the content in the temporal features
for features in features_yr:
    print(features, data[features].unique())


# In[12]:


#Relationship between Year Sold and Sales Price
data.groupby('YrSold')['SalePrice'].mean().plot()
plt.xlabel('YrSold')
plt.ylabel('Sales Price')
plt.title('House Price vs. Year Sold')


# In[13]:


#average Sales Price is decreasing with time


# In[14]:


for features in features_yr:
    if features != 'YrSold':
        data_yr=data.copy()
        data_yr[features]=data_yr['YrSold']-data_yr[features]
        
        plt.scatter(data_yr[features],data['SalePrice'])
        plt.xlabel(features)
        plt.ylabel('SalePrice')
        plt.show()


# In[15]:


#Numerical variables are of two types 1. Discrete 2. Continuous

features_dis= [features for features in features_num if len(data[features].unique())<25 and features not in features_yr ]
print("Discrete Variables Count: {}".format(len(features_dis)))


# In[16]:


features_dis


# In[17]:


for a in features_dis:
    data_dis=data.copy()
    data_dis.groupby(a)['SalePrice'].median().plot.bar()
    plt.xlabel(a)
    plt.ylabel('Sales Price')
    plt.title(a)
    plt.show()   


# In[18]:


features_con=[features for features in features_num if features not in features_dis + features_yr+ ['Id']]
print("Continuous feature Count {}".format(len(features_con)))


# In[19]:


features_con


# In[20]:


for feature in features_con:
    data_con=data.copy()
    data_con[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()


# In[21]:


#applying log transformation
for feature in features_con:
    data_con=data.copy()
    if 0 in data_con[feature].unique():
        pass
    else:
        data_con[feature]=np.log(data_con[feature])
        data_con['SalePrice']=np.log(data_con['SalePrice'])
        plt.scatter(data_con[feature],data_con['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalesPrice')
        plt.title(feature)
        plt.show()


# ### Outliers

# In[22]:


for feature in features_con:
    data_con=data.copy()
    if 0 in data_con[feature].unique():
        pass
    else:
        data_con[feature]=np.log(data_con[feature])
        data_con.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()


# In[23]:


features_cat=[feature for feature in data.columns if data[feature].dtype=='O']
features_cat


# In[24]:


for feature in features_cat:
    print('{} contains {} unique values'.format(feature,len(data[feature].unique())))


# In[25]:


## Find out the relationship between categorical variable and dependent feature SalesPrice


# In[26]:


for feature in features_cat:
    data_cat=data.copy()
    data_cat.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# # Feature Engineering

# In[27]:


#train data


# In[28]:


dataset=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')


# In[29]:


dataset.head(5)


# In[30]:


features_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>0 and dataset[feature].dtypes == 'O']

for i in features_nan:
    print(i," has",np.round(dataset[i].isnull().mean()*100,0)," % of null values")


# In[31]:


#replacing nans with "missing"

def replace_nan(dataset,features_nan):
    data1=dataset.copy()
    data1[features_nan]= data1[features_nan].fillna("Missing")
    return data1

dataset=replace_nan(dataset,features_nan)
dataset[features_nan].isnull().sum()


# In[32]:


features_num_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>0 and dataset[feature].dtypes != 'O']

for i in features_num_nan:
    print("{}:{}% missing values".format(i,np.round(dataset[i].isnull().mean()*100,0)))


# In[33]:


for i in features_num_nan:
    median_value=dataset[i].median()
    
    dataset[i+'nan']=np.where(dataset[i].isnull(),1,0)
    dataset[i].fillna(median_value,inplace=True)


# In[34]:


dataset.head(5)


# In[35]:


features_temp= [feature for feature in dataset.columns if (dataset[feature].dtypes != 'O') and ('Yr' in feature or 'Year' in feature)]
features_temp


# In[36]:


for i in features_temp:
    dataset[i]=dataset['YrSold']-dataset[i]


# In[37]:


dataset.head(5)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




