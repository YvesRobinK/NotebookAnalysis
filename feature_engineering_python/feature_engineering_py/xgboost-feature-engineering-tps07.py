#!/usr/bin/env python
# coding: utf-8

# # **TPS July 2021** ♨

# Hello everyone,
# 
# This a fork of [this](https://www.kaggle.com/jarupula/eda-rf-model-tps-july-21) notebook to use XGBoost instead of a Random Forest and use a feature engineered dataset (see https://www.kaggle.com/okyanusoz/tps07-feature-engineering for the feature engineering).
# 
# This should (hopefully) give better accuracy.
# 
# Enjoy!

# ***

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#5642C5;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#               color:white;
#           text-align:center;">
# Loading requirements
#              
# </p>
# </div>

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from xgboost import XGBRegressor


# #### Loading dataset

# In[15]:


train = pd.read_csv('../input/tps07-feature-engineering/train.csv')
test = pd.read_csv('../input/tps07-feature-engineering/test.csv')
sample_submission = pd.read_csv('../input/tabular-playground-series-jul-2021/sample_submission.csv')


# In[16]:


train.head()


# In[17]:


train.info()


# We need to convert date_time column to int.

# #### looking at the statistics

# In[18]:


train.describe()


# #### **NOTES**
# 
# - There no missing values
# - There are NO negative observations
# - Feature scaling may be required

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#5642C5;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#               color:white;
#           text-align:center;">
# EDA
#              
# </p>
# </div>

# Let's visualize the dataset to better understand it.

# In[19]:


carbon_monoxide = train['target_carbon_monoxide']
benzene = train['target_benzene']
nitrogen_oxides = train['target_nitrogen_oxides']


# In[20]:


f = plt.figure(figsize=(15,8))

ax = f.add_subplot(131)
stats.probplot(carbon_monoxide, plot=ax)
ax.set_title('Carbon monoxide probability distribution')

ax1 = f.add_subplot(132)
stats.probplot(benzene, plot=ax1)
ax1.set_title('Benzene probability distribution')

ax2 = f.add_subplot(133)
stats.probplot(nitrogen_oxides, plot=ax2)
ax2.set_title('Nitrogen_oxides probability distribution')

plt.show()


# #### **NOTES**
# 
# - The target values are NOT Gaussian distributed as the red line does't overlap with the blue points.
# - There are EXTREME target observations(Outliers) in the data set.
# - There is an OFFSET at 0 for 'benzene' plot.

# #### Scatter matrix

# In[21]:


targets = train[['target_carbon_monoxide', 'target_benzene', 'target_nitrogen_oxides']].copy()


# In[22]:


pd.plotting.scatter_matrix(targets, alpha=0.5,figsize=(15, 8))
plt.show()


# #### **NOTES**
# 
# - All the distributions are tail heavy(Right skewed).
# - There is a STRONG CORRELATION among target variables
# - We can see the OFFSET line in the 'benzene' feature, which we have seen in probability plot. we need to UNDER SAMPLE this data points/remove those to minimize the effect.

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#5642C5;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#               color:white;
#           text-align:center;">
# Model
#              
# </p>
# </div>

# ##### Preprocessing

# In[23]:


train['date_time'] = train['date_time'].astype('datetime64[ns]').astype(np.int64)/10**9
test['date_time'] = test['date_time'].astype('datetime64[ns]').astype(np.int64)/10**9


# In[24]:


train.head()


# In[25]:


X_train = train.drop(['target_carbon_monoxide', 'target_benzene', 'target_nitrogen_oxides'], axis=1).copy()


# In[26]:


model = XGBRegressor(random_state=3)
model.fit(X_train, carbon_monoxide)
sample_submission['target_carbon_monoxide'] = model.predict(test)

model = XGBRegressor(random_state=3)
model.fit(X_train, benzene)
sample_submission['target_benzene'] = model.predict(test)

model = XGBRegressor(random_state=3)
model.fit(X_train, nitrogen_oxides)
sample_submission['target_nitrogen_oxides'] = model.predict(test)


# In[27]:


sample_submission.to_csv('submission.csv', index=False)

