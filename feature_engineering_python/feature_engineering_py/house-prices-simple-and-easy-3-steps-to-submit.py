#!/usr/bin/env python
# coding: utf-8

# # <u>House Prices: Simple and easy 3 steps to submit</u>
# This notebook will guide you through three simple and effective steps to submit.

# * [1. Preparations](#1)
#     * [1.1 Import libraries](#1.1)
#     * [1.2 Load dataset](#1.2)
#     * [1.3 Check data](#1.3)
#     * [1.4 Combine train and test](#1.4)
# * [2. Feature Engineering](#2)
#     * [2.1 Transform numeric into logarithms](#2.1)
#     * [2.2 Transform categorical into one-hot vector](#2.2)
# * [3. Prediction and submission](#3)
#     * [3.1 Format data](#3.1)
#     * [3.2 Prediction](#3.2)
#     * [3.3 Create submission](#3.3)

# <a id="1"></a><h1 style='background:slateblue; border:.; color:white'><center>1. Preparations</center></h1>

# ## 1.1 Import libraries<a id="1.1"></a>
# **Import all required libraries**

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb


# ## 1.2 Load dataset<a id="1.2"></a>
# **Load each data as a Pandas DataFrame**

# In[2]:


submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
submission.head()


# In[3]:


train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
train.head()


# In[4]:


test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
test.head()


# ## 1.3 Check data<a id="1.3"></a>
# **check data shape, count and dtype of each column**

# In[5]:


print('=========== train infomation ===========')
train.info()
print('\n\n=========== test infomation ===========')
test.info()


# ## 1.4 Combine train and test<a id="1.4"></a>
# **Combine train and test so that you can do each future operation once**

# In[6]:


data = pd.concat([train, test])
data.shape


# <a id="2"></a><h1 style='background:slateblue; border:.; color:white'><center>2. Feature Engineering</center></h1>

# ## 2.1 Transform numeric into logarithms<a id="2.1"></a>
# **transform all numeric columns of the data frame into logarithms to reduce skewness**<br><br>
# Approximating to a normal distribution often improves the accuracy of machine learning.<br>Logarithmic conversion reduces the range when the feature scale is large and expands it when the feature scale is small.<br>This often allows you to get closer to a mountainous distribution as if you were crushing a long-tailed distribution.

# In[7]:


# current numeric data
numerics = data.loc[:,data.dtypes != 'object'].drop('Id', axis=1)
numerics.head()


# In[8]:


# numeric data after conversion to logarithm
log_numerics = np.log1p(numerics)
log_numerics.head()


# In[9]:


# compare skewnesses before with after of logarithmization
skewness = pd.concat([numerics.apply(lambda x: skew(x.dropna())),
                      log_numerics.apply(lambda x: skew(x.dropna()))],
                     axis=1).rename(columns={0:'original', 1:'logarithmization'}).sort_values('original')
skewness.plot.barh(figsize=(12,10), title='Comparison of skewness of original and logarithmized', width=0.8);


# The skewnesses of many features are reduced by logarithmic conversion. Some skewnesses have increased, but overall it has decreased.

# ## 2.2 Transform categorical into one-hot vector<a id="2.2"></a>
# **transform all categorical columns of the data frame into one-hot vector**<br><br>
# Since GBDT treats features as numerical data, it is necessary to convert categorical data to numerical values. Label encoding is fine, but one-hot encoding often has better performance.

# In[10]:


cat_cols = data.loc[:,data.dtypes == 'object'].columns
data.loc[:,cat_cols].head()


# In[11]:


# categorical data after conversion to one-hot vector
# cat_data = pd.get_dummies(data.loc[:, cat_cols], drop_first=True, dummy_na=True)
cat_data = pd.get_dummies(data.loc[:, cat_cols], drop_first=True)
cat_data.head()


# <a id="3"></a>
# <h1 style='background:slateblue; border:.; color:white'><center>3. Prediction and submission</center></h1>

# ## 3.1 Format data<a id="3.1"></a>
# **Format data for training**

# In[12]:


# merge categorical and numeric columns
optimized_data = pd.concat([data['Id'], cat_data, log_numerics], axis=1)
optimized_data.head()


# In[13]:


# split data into X_train, y_train and test
train = optimized_data[:train.shape[0]]
test = optimized_data[train.shape[0]:].drop(['Id', 'SalePrice'], axis=1)
X_train = train.drop(['Id', 'SalePrice'], axis=1)
y_train = train['SalePrice']


# ## 3.2 Prediction<a id="3.2"></a>
# **fit and predict using lightGBM**

# In[14]:


# train
lgb_train = lgb.Dataset(X_train, y_train)
params = {
        'objective' : 'regression',
        'metric' : {'rmse'}
}
gbm = lgb.train(params, lgb_train)
# predict
pred = gbm.predict(test)


# ## 3.3 Create submission<a id="3.3"></a>
# **convert prediction into exponent and export CSV file**

# In[15]:


# convert logarithms into exponent
pred = np.expm1(pred)
# create submission file
results = pd.Series(pred, name='SalePrice')
submission = pd.concat([submission['Id'], results], axis=1)
submission.to_csv('submission.csv', index=False)
submission.head()


# This notebook has a score of 0.13086 and is in the top 30%.<br>Based on this notebook, Feature engineering, Hyper-parameter tuning and ensemble will give you a better score.
