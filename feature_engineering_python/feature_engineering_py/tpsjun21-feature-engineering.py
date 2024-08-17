#!/usr/bin/env python
# coding: utf-8

# In this notebook I will try to gain insights from data. This notebook will help in later deciding what model would be best for this problem.

# # Initialization

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train_data = pd.read_csv('/kaggle/input/tabular-playground-series-jun-2021/train.csv')
test_data = pd.read_csv('/kaggle/input/tabular-playground-series-jun-2021/test.csv')


# # Feature Engineering and EDA

# ## Observing datasets

# First, we will observe the dataset by printing
#  - features info
#  - some dataset rows
#  - statistical information
#  
# Then, we will try to gather insights using these.

# In[3]:


train_data.info()
train_data.head()


# In[4]:


test_data.info()
test_data.head()


# From above two cells, we can conclude that there are **no null values** in training data or in test data both. In total, there are 75 features and all of these contain integer values.
# 
# There are **200000 records** in training data and **100000 records** in test data. Since the dataset is huge, models which are computationally expensive like Support Vector Machine cannot be used for this problem.
# 
# Another observation we can make is that a lot of values in the dataset are **zero**. To confirm it we can count the number of zeros in both datasets and find the percentage of zeros.

# In[5]:


# Counting the number of zeros in training dataset
zeros = 0
for field in train_data.drop(['id', 'target'], axis=1).columns:
    zeros += train_data[field].value_counts().loc[0]

print('Training data')
print('Zero Count: ', zeros)
print(f'Percent of Zeros: {zeros/(75*200000)*100}%')

# Counting the number of zeros in test dataset
zeros = 0
for field in test_data.drop(['id'], axis=1).columns:
    zeros += test_data[field].value_counts().loc[0]

print('\nTest data')
print('Zero Count: ', zeros)
print(f'Percent of Zeros: {zeros/(75*100000)*100}%')


# In both training data and test data, the percent of zero values is **about 65%**. So, the dataset is sparse.

# In[6]:


# removing id as it will not provide any statistical insights
train_data.drop(['id'], axis=1).describe()


# In[7]:


# removing id as it will not provide any statistical insights
test_data.drop(['id'], axis=1).describe()


# As expected, the 25 percentile and 50 percentile values are 0.
# 
# The maximum value of each feature is a **lot greater** than the 75 percentile.
# 
# Both of these observations are valid for both training data and test data. So, considering the huge values as outliers and removing them might degrade the model performance.
# 
# We can also observe that all statistical values for all features for both the datasets **are quite similar** to each other, so it might not be necessary to apply regularization in the model.

# ## Finding Correlation between features

# Next, we will find correlation between features to check for any redundant features.
# 
# The **basic concept** is that if the correlation value between two features is **close to -1 or 1**, the more features are like each other. So, keeping both such features provides no extra information and we can remove one of these features.

# In[8]:


# Finding correlation between features
corr = train_data.drop(['id'], axis=1).corr() # we are again removing the id column due to the same reason
corr.info()
corr


# Since, the correlation matrix is of size 75x75, it is not possible gain any insights by observing it.
# 
# So, we will be creating a **correlation plot**.

# In[9]:


# Creating a correlation plot of size 20x20
fig, ax =plt.subplots(figsize=(20, 20))

plt.title("Correlation Plot")
sns.heatmap(corr,
            cmap=sns.diverging_palette(230, 10, as_cmap=True),
            square=True,
            ax=ax)
plt.show()


# The scale on the right of the plot shows that values **close to 0 are plotted as blue** and the **close to 1 are plotted as red**.
# 
# The only red boxes in the plot are along the diagonal which is expected as each feature will definitely be like itself. So, correlation between a feature with itself will be equal to 1.
# 
# The whole plot is blue, which suggests that the correlation values between each features are close to 0. Thus, we can conclude that the features are **not correlated** to each other at all.
# 
# So, all the features should be used for model training and feature reduction techniques will do harm to model accuracy.

# ## Finding class occurence percentage

# Now, we will check whether the occurence of each target class in the training data is equal or not.
# 
# If the number of examples of each class in the training data are not close to each other, then a poorly trained model might generalize this as the property of the feature set, which might not be the actual case. So, in such cases it is beneficial to measure **precision** and **recall** values also in addition to accuracy.

# In[10]:


class_counts = train_data.target.value_counts().sort_index()
class_percents = class_counts/len(train_data)*100
class_percents


# In[11]:


plt.figure(figsize=(10,6))
plt.bar(class_percents.index, class_percents.values)

plt.show()


# **Class_6** and **Class_8** each occur for about **26%** of the cases. On the other hand, **Class_4** and **Class_5** each occur for about **2%** of the cases.
# 
# So, in such case a good strategy would be to maintain a good ratio of precision and recall values.
