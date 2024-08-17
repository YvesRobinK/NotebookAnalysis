#!/usr/bin/env python
# coding: utf-8

# # Opening
# This is my exercise to explore a dataset based on courses I've learned for few months. I'm beginner on Data Science and I have strong passionate to learn it! Your suggestions and advices is important for me!
# 
# I'm only work in dataset train

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


# # Objective
# From this dataset, I would like to figure out the kind of house type which has high price.

# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# # Dataset Identification

# In[3]:


df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.shape


# Dataset df which is dataset train has total 1460 rows and 81 columns.

# In[7]:


numeric_columns = df.select_dtypes(exclude = ['object'])
numeric_columns


# In[8]:


numeric_columns.columns


# In[9]:


len(numeric_columns.columns)


# Dataset has 38 numeric columns

# In[10]:


categoric_columns = df.select_dtypes(include = ['object'])
categoric_columns


# In[11]:


categoric_columns.columns


# In[12]:


len(categoric_columns.columns)


# Dataset df has 43 categorical columns

# In[13]:


categoric_columns.nunique()


# In[14]:


numeric_columns.nunique()


# In[15]:


df.isna().sum().sum()


# Dataset df has a lot of missing values in its various columns

# In[16]:


categoric_columns.isna().sum()


# There are several columns which have too much missing values with total amount most likely 50% even 90%.

# In[17]:


numeric_columns.isna().sum()


# In[18]:


df.duplicated().sum()


# In[19]:


numeric_columns.describe().T


# In[20]:


categoric_columns.describe().T


# In[21]:


df.corr().T


# In[22]:


plt.figure(figsize=(20, 20))
sns.heatmap(df.corr(),
			cmap = 'BrBG',
			fmt = '.2f',
			linewidths = 2,
			annot = True)


# # Data Preprocessing

# Delete columns where have too much unique value.

# In[23]:


df_clean = df.drop(['Id'], axis=1)


# In[24]:


list_of_numeric = df_clean.select_dtypes(exclude = ['object']).columns.tolist()
list_of_numeric


# In[25]:


list_of_categoric = df_clean.select_dtypes(include = ['object']).columns.tolist()
list_of_categoric


# In[26]:


#numeric_columns = df_clean.select_dtypes(exclude = ['object'])
#categoric_columns = df_clean.select_dtypes(include = ['object'])


# Filling missing values

# In[27]:


df_clean[list_of_numeric].isna().sum()


# In[28]:


df_clean[list_of_numeric].isna().any()


# In[29]:


for i in list_of_numeric:
    if df_clean[i].isna().any() == True:
        df_clean[i] = df_clean[i].fillna(df_clean[i].mean())
        print('Is {} has missing value? {}'.format(i, df_clean[i].isna().any()))
        
df_clean[list_of_numeric].isna().any()


# In[30]:


#df_clean['LotFrontage'] = df_clean['LotFrontage'].fillna(df_clean['LotFrontage'].mean())
#df_clean['MasVnrArea'] = df_clean['MasVnrArea'].fillna(df_clean['MasVnrArea'].mean())
#df_clean['GarageYrBlt'] = df_clean['GarageYrBlt'].fillna(df_clean['GarageYrBlt'].mean())

#df_clean[list_of_numeric].isna().any()


# In[31]:


df_clean[list_of_categoric].isna().sum()


# In[32]:


df_clean[list_of_categoric].isna().any()


# Remove columns which have too many missing values (likely 50% to above) and fill missing values on other columns with mode.

# In[33]:


for i in list_of_categoric:
    if df_clean[i].isna().sum() > 600:
        del df_clean[i]
    else:
        df_clean[i] = df_clean[i].fillna(df_clean[i].mode()[0])
        print('Is {} has missing value? {}'.format(i, df_clean[i].isna().any()))


# In[34]:


list_of_categoric = df_clean.select_dtypes(include = 'object').columns.tolist()
list_of_categoric


# In[35]:


df_clean[list_of_categoric].isna().any()


# In[36]:


df_clean.isna().sum().sum()


# In[37]:


df_clean.shape


# Normalization numeric columns in dataset df_clean.

# In[38]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[39]:


df_clean[list_of_numeric] = scaler.fit_transform(df_clean[list_of_numeric])


# In[40]:


df_clean.describe().T


# Encoding categorical columns

# In[41]:


df_clean = pd.get_dummies(df_clean)


# In[42]:


df_clean


# In[43]:


#from sklearn.preprocessing import LabelEncoder


# In[44]:


#len(list_of_categoric)


# In[45]:


#for col in list_of_categoric:
#    df_clean[col] = LabelEncoder().fit_transform(df_clean[col]) 


# In[46]:


#df_clean[list_of_categoric]


# In[47]:


#df_clean


# # Train Test Split

# In[48]:


from sklearn.model_selection import train_test_split

X = df_clean.drop(['SalePrice'], axis=1)
y = df_clean['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[49]:


print("Dataset train contains:", X_train.shape[0], "rows and ", X_train.shape[1], "columns")
print("Dataset test contains:", X_test.shape[0], "rows and ", X_test.shape[1], "columns")

print("Variable target to train contains:", y_train.shape[0], "rows")
print("Variable target to test contains:", y_test.shape[0], "rows")


# # Modeling with PCA

# In[50]:


#Here I decompose each row into 10 principal components
from sklearn.decomposition import PCA

def pca_dec(data, n):
  pca = PCA(n)
  X_dec = pca.fit_transform(data)
  return X_dec, pca

#Decomposing the train set:
pca_train_results, pca_train = pca_dec(X_train, 10)

#Decomposing the test set:
pca_test_results, pca_test = pca_dec(X_test, 10)

#Creating a table with the explained variance ratio
names_pcas = [f"PCA Component {i}" for i in range(1, 11, 1)]
scree = pd.DataFrame(list(zip(names_pcas, pca_train.explained_variance_ratio_)), columns=["Component", "Explained Variance Ratio"])
print(scree)


# In[51]:


#Sorting the values of the first principal component by how large each one is
df2 = pd.DataFrame({'PCA':pca_train.components_[0], 'Variable Names':list(X_train.columns)})
df2 = df2.sort_values('PCA', ascending=False)

#Sorting the absolute values of the first principal component by magnitude
df3 = pd.DataFrame(df2)
df3['PCA']=df3['PCA'].apply(np.absolute)
df3 = df3.sort_values('PCA', ascending=False)
#print(df2['Variable Names'][0:11])

df2.head()


# What is conclusion? I dunno and I'll learn it more to know the answer!
