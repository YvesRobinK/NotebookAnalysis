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


# # WiDS - Temp Feature Engineering

# In the WiDS dataset we have 36 temperature columns, 3 (min, avg, max) for each month. In this notebook we will try to see the effectiveness of all the columns in predicting our target variable. We will also see whether is it safe to drop redundant columns to prevent multicollinearity.

# ## Steps performed:

# 1. Slice the dataset to read just the temperature columns
# 2. Plotting the correlation matrix to see the effectiveness
# 3. We also fetch the spearman correlation values
# 4. Plotting various seaborn charts:
#     - Barplot to check the count of each feature with respect to target variable
#     - Scatterplot to detect outliers
#     - Regplot to visualize linear regression model for the target and each feature
# 5. Cleaning Data according to analysis
# 6. Binning Data to categorize different months into 4 seasons

# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# In[3]:


train_df = pd.read_csv('../input/widsdatathon2022/train.csv')
train_df.head()


# ### **Considering features from 9 throgh 40**

# In[4]:


sliced_df = train_df.iloc[:,8:44]
sliced_df['site_eui'] = train_df['site_eui']


# In[5]:


sliced_df.head(10)


# In[6]:


for i in sliced_df.columns:
    print(i,"   ", sliced_df[i].mean())


# In[7]:


corr = sliced_df.corr()
# corr.style.background_gradient(cmap='coolwarm')
plt.figure(figsize=(22,15))
plt.title('Correlation Matrix', fontsize=25 )
sns.heatmap(corr, cmap="Greens",annot=True )
plt.show()


# In[8]:


sliced_df.corr(method='spearman').loc[:,'site_eui']


# In[9]:


cols = ['january_avg_temp', 'february_avg_temp', 'march_avg_temp', 'april_avg_temp' ,'may_avg_temp', 
        'june_avg_temp', 'july_avg_temp', 'august_avg_temp','september_avg_temp', 'october_avg_temp', 'november_avg_temp', 'december_avg_temp']
for i in cols:
    plt.figure(figsize=(20,5))
    sns.barplot(x=i, y="site_eui", data=sliced_df.sort_values(by=i, ascending=False))
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# ### **Scatter Plot for temp vs target** 

# In[10]:


for i in sliced_df.iloc[:,:36]:
    fig = plt.figure(figsize=(10,6))
    sns.scatterplot(data=sliced_df, x=i, y="site_eui")
    plt.title(str(i)+' vs site EUI', fontsize=20)


# ### **Line Plot Overlaid on Scatter Plot for Average Temp**

# In[11]:


cols = ['january_avg_temp', 'february_avg_temp', 'march_avg_temp', 'april_avg_temp' ,'may_avg_temp', 
        'june_avg_temp', 'july_avg_temp', 'august_avg_temp','september_avg_temp', 'october_avg_temp', 'november_avg_temp', 'december_avg_temp']
for i in cols:   
    plt.figure(figsize=(10,6))
    sns.regplot(x = "site_eui", 
                y = i, 
                data = sliced_df)
    plt.show()


# ### We conclude that the temperature attributes do not have much effect on the site_eui. For the initial analysis dropping the min and max temp attributes and going forward with just the avg temp attribute.

# In[12]:


sliced_df.head()


# In[13]:


avg_temp_df = sliced_df[cols]
avg_temp_df.head()


# **We are going to categorize the temp attributes into 4 seasons based on:**
# 1. spring runs from March 1 to May 31
# 2. summer runs from June 1 to August 31
# 3. fall (autumn) runs from September 1 to November 30; and
# 4. winter runs from December 1 to February 28
# 
# We have taken the mean of 3 temp months at a time and combined them under one season based on the above logic.

# In[14]:


avg_temp_df['spring'] = (avg_temp_df['march_avg_temp'] + avg_temp_df['april_avg_temp'] + avg_temp_df['may_avg_temp'])/3
avg_temp_df['summer'] = (avg_temp_df['june_avg_temp'] + avg_temp_df['july_avg_temp'] + avg_temp_df['august_avg_temp'])/3
avg_temp_df['fall'] = (avg_temp_df['september_avg_temp'] + avg_temp_df['october_avg_temp'] + avg_temp_df['november_avg_temp'])/3
avg_temp_df['winter'] = (avg_temp_df['december_avg_temp'] + avg_temp_df['january_avg_temp'] + avg_temp_df['february_avg_temp'])/3


# In[15]:


avg_temp_df.tail(10)


# ### Now, we can work with the seasons' attributes. Hence, dropping the rest of the columns.

# In[16]:


avg_temp_df.drop(cols, axis=1, inplace=True)
avg_temp_df.head()


# # Conclusion

# After analysis and few operations on our temperature features we brought down 36 columns to just 4. We will use this to build our initial model and later we can add features to see whether our model performs better.
