#!/usr/bin/env python
# coding: utf-8

# **In this notebook, I studied the following relationships:**
# 
# *Association between categorical features (feature-feature)*
# 
# *Association between categorical features and target (feature-target)*
# 
# *Correlation between numerical features (feature-feature)*
# 
# *Correlation between numerical features and target (feature-target)*
# 
# **I highly recommend reading this medium post which explains my approach in this notebook.**
# 
# The Search for Categorical Correlation: https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
# 
# 
# **To calculate the association and correlations, I used Dython which is a set of data analysis tools in PYTHON 3.x, which can let you get more insights into your data.**
# 
# dython website: http://shakedzy.xyz/dython/
# 
# 
# **The output of this notebook:**
# 
# *cat_f_f.csv* Association between categorical features (feature-feature)
# 
# *cat_f_t.csv* Association between categorical features and target (feature-target)
# 
# *num_f_f.csv* Correlation between numerical features (feature-feature)
# 
# *num_f_t.csv* Correlation between numerical features and target (feature-target)
# 
# 
# **Feel free to download and use the output of this notebook for your feature engineering studies.**
# 
# Good Luck!

# In[1]:


# The easiest way to install dython is using pip install:
get_ipython().system('pip install dython')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dython.nominal import associations

import warnings
warnings.filterwarnings("ignore")


# In[3]:


DEBUG = False

train_data_org = pd.read_feather('../input/amex-default-prediction-feather/train.feather').set_index('customer_ID')
if DEBUG:
    train_data = train_data_org.iloc[:20000, :].copy()
else:
    train_data = train_data_org.copy()  
del train_data_org
train_labels = pd.read_csv('../input/amex-default-prediction/train_labels.csv', index_col='customer_ID').loc[train_data.index]
categorical_features = ['B_30', 'B_31', 'B_38', 'D_114', 
                        'D_116', 'D_117', 'D_120', 'D_126', 
                        'D_63', 'D_64', 'D_66', 'D_68']


# In[4]:


train_data.shape, train_labels.shape


# In[5]:


association_dictionary = associations(train_data[categorical_features], nominal_columns = categorical_features, mark_columns = True, 
                nom_nom_assoc = 'theil', nan_strategy = 'drop_samples', figsize= (15, 15), vmin = 0, vmax=0.8, compute_only = False)


# In[6]:


print('associations ranking:')
acff = association_dictionary['corr'].stack()
acff = acff[acff.index.get_level_values(0) < acff.index.get_level_values(1)]
acff = acff.sort_values(ascending = False)
acff.to_csv('cat_f_f.csv')
print(acff)


# In[7]:


train_feature_target = pd.concat([train_data.groupby('customer_ID').tail(1), train_labels.groupby('customer_ID').tail(1)], axis=1)
train_feature_target.head()


# In[8]:


print('\nThe association between a categorical target and categorical features:')
association_dictionary = associations(train_feature_target[categorical_features + ['target']], 
                                      nominal_columns = categorical_features + ['target'], mark_columns = True,
                                      display_rows = ['target'], nan_strategy = 'drop_samples',figsize= (15, 15),
                                      vmin = 0, vmax=0.8, compute_only = False, cbar = False)


# In[9]:


acft = association_dictionary['corr'].stack().sort_values(ascending = False)
print(acft)
acft.to_csv('cat_f_t.csv')


# In[10]:


numerical_features = list(set(train_data.columns).difference(set(categorical_features)))
numerical_features.remove('S_2')
train_feature_target[numerical_features] = train_feature_target[numerical_features].fillna(train_feature_target[numerical_features].median())


# In[11]:


# compute_only = True means that we don't want to plot the heatmap 
correlation_dictionary = associations(train_feature_target[numerical_features], numerical_columns = numerical_features, mark_columns = False, 
                num_num_assoc = 'pearson', nan_strategy = 'drop_samples',  figsize= (15, 15), vmin = 0, vmax=0.8, compute_only = True)


# In[12]:


print('correlations ranking:')
cnff = correlation_dictionary['corr'].stack()
cnff = cnff[cnff.index.get_level_values(0) < cnff.index.get_level_values(1)]
cnff = cnff.sort_values(ascending = False)
cnff.to_csv('num_f_f.csv')
print(cnff)


# In[13]:


correlation_dictionary = associations(train_feature_target[numerical_features + ['target']], nominal_columns = 'target',
                                        numerical_columns = numerical_features, mark_columns = True, 
                                        display_rows = ['target'], nan_strategy = 'drop_samples', 
                                        figsize= (15, 15), vmin = 0, vmax=0.8, compute_only = True, cbar = False)


# In[14]:


cnft = correlation_dictionary['corr'].stack().sort_values(ascending = False)
cnft.to_csv('num_f_t.csv')
print(cnft)

