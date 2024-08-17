#!/usr/bin/env python
# coding: utf-8

# <div style="background-color:rgba(215, 159, 21, 0.5);">
#     <h1><center>Importing Libraries and Data</center></h1>
# </div>

# Please scroll to the end to see the key take-aways and my action points for the first model run.

# In[ ]:


import random
random.seed(123)

import pandas as pd
import numpy as np
import datatable as dt
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency
import numpy as np


# In[ ]:


# using datatable for faster loading

train = dt.fread(r'../input/tabular-playground-series-oct-2021/train.csv').to_pandas()
test = dt.fread(r'../input/tabular-playground-series-oct-2021/test.csv').to_pandas()
sub = dt.fread(r'../input/tabular-playground-series-oct-2021/sample_submission.csv').to_pandas()


# <div style="background-color:rgba(215, 159, 21, 0.5);">
#     <h1><center>Reduce Memory Usage</center></h1>
# </div>

# In[ ]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                else:
                    df[col] = df[col].astype(np.float32)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


train_1 = reduce_mem_usage(train)
test_1= reduce_mem_usage(test)


# <div style="background-color:rgba(215, 159, 21, 0.5);">
#     <h1><center>Basic Data Check</center></h1>
# </div>

# In[ ]:


train_data = train_1.copy()
test_data = test_1.copy()


# In[ ]:


# our data has 287 columns (including id and target)

train_data.head()


# In[ ]:


# 1 million rows, with 240 numerical columns and 46 boolean ones

train_data.info() 


# In[ ]:


# no null values in train and test data

print('Number of nulls in train data: ',train_data.isna().sum().sum())
print('Number of nulls in test data: ',test_data.isna().sum().sum())


# In[ ]:


# segregating train data as per different data types - id is the only int32 column

train_data_float = train_data.select_dtypes(include = 'float16')
train_data_boolean = train_data.select_dtypes(include = 'bool')

train_data_float= pd.concat([train_data_float,train_data['target']],axis=1)


# <div style="background-color:rgba(215, 159, 21, 0.5);">
#     <h1><center>Check for float data</center></h1>
# </div>

# In[ ]:


# overall, all features have values between 0 and 1, so no need to normalise them
# I checked min, max and mean values to come to this conclusion

train_data_float.min().sort_values(ascending=False)


# In[ ]:


# low correlation with the target variable - these are the top 10 most correlated

np.abs(train_data_float.corr()['target']).sort_values(ascending=False).head(10)


# In[ ]:


# plotting a few of the top correlation variables

fig = plt.figure(figsize=(30,14))
fig, ax = plt.subplots(2,2)
fig.tight_layout()

sns.distplot(train_data_float['f58'],ax=ax[0][0])
sns.distplot(train_data_float['f69'],ax=ax[0][1])
sns.distplot(train_data_float['f156'],ax=ax[1][0]) # only this is left skewed
sns.distplot(train_data_float['f58'],ax=ax[1][1])


# <div style="background-color:rgba(215, 159, 21, 0.5);">
#     <h1><center>Check for boolean data</center></h1>
# </div>

# In[ ]:


# checking % of 1's in our boolean data - top 10 as per number of 1's
# our target variable is very balanced

(train_data_boolean.sum()/len(train_data)*100.0).sort_values(ascending=False).head(10)


# In[ ]:


# target variable - highly balanced

sns.countplot(train_data_boolean['target'])


# In[ ]:


# scatterplot of the target variable

sns.scatterplot(train_data_boolean['f22'])


# In[ ]:


# top 10 correlated boolean variables with our target - only f22 seems really significant

np.abs(train_data_boolean.corr()['target']).sort_values(ascending=False).head(10)


# In[ ]:


# checking if any other variables are strongly related with f22
# It is highly independent of all other variables. Interesting

np.abs(train_data_boolean.corr()['f22']).sort_values(ascending=False).head(10)


# In[ ]:


# Cramer's V correlation matrix - building a function

def cramers_V(var1,var2):
    crosstab =np.array(pd.crosstab(var1,var2, rownames=None, colnames=None)) # Cross table building
    stat = chi2_contingency(crosstab)[0] # Keeping of the test statistic of the Chi2 test
    obs = np.sum(crosstab) # Number of observations
    mini = min(crosstab.shape)-1 # Take the minimum value between the columns and the rows of the cross table
    return (stat/(obs*mini))


# In[ ]:


rows= []

for var1 in train_data_boolean:
    col = []
    for var2 in train_data_boolean :
        cramers =cramers_V(train_data_boolean[var1], train_data_boolean[var2]) # Cramer's V test
        col.append(round(cramers,2)) # Keeping of the rounded value of the Cramer's V  
    rows.append(col)

cramers_results = np.array(rows)
df = pd.DataFrame(cramers_results,columns = train_data_boolean.columns,index = train_data_boolean.columns)
df 


# In[ ]:


# we see nothing useful here, other than with f22

df['target'].sort_values(ascending=False)


# In[ ]:


#  plotting f22 with our target variable - we see the negative correlation here
# even those with majority 1s seem to have some relation with our target - quite balanced

fig = plt.figure(figsize=(30,14))
fig, ax = plt.subplots(2,2)
fig.tight_layout()
sns.countplot(train_data_boolean['f22'],hue=train_data_boolean['target'],ax=ax[0][0])
sns.countplot(train_data_boolean['f253'],hue=train_data_boolean['target'],ax=ax[0][1])
sns.countplot(train_data_boolean['f246'],hue=train_data_boolean['target'],ax=ax[1][0])
sns.countplot(train_data_boolean['f247'],hue=train_data_boolean['target'],ax=ax[1][1])


# <div style="background-color:rgba(215, 159, 21, 0.5);">
#     <h1><center>Feature Engineering Experiment</center></h1>
# </div>

# In[ ]:


# I will work with the boolean features (other than f22) to come up with something
# ones is simply the sum of Trues across the rows, for variables other than f22
# vote is majority voting of all those columns (if most are 1's, then vote will be 1)

train_data_boolean['ones'] = train_data_boolean.drop(['f22','target'],axis=1).sum(axis=1)
train_data_boolean['vote'] = train_data_boolean['ones'].apply(lambda x: 1 if (x/47*100.0)>50 else 0)


# In[ ]:


# a-ha, 'ones' may be a decent predictor too (+0.04), 'vote' not so much
# I had created "ones", once with f22 and once without it, excluding f22 it seems better.

np.abs(train_data_boolean.corr()['target']).sort_values(ascending=False).head(10)


# In[ ]:


# checking for continuous variables - memory issues always

train_data_float['min'] = train_data_float.min(axis=1)
train_data_float['max'] = train_data_float.max(axis=1)
train_data_float['sum'] = train_data_float.sum(axis=1)
train_data_float['std'] = train_data_float.std(axis=1)


# In[ ]:


np.abs(train_data_float.corr()['target']).sort_values(ascending=False).head(10)


# <div style="background-color:rgba(215, 159, 21, 0.5);">
#     <h1><center>Take-aways and Action Points</center></h1>
# </div>

# **Take-aways**
# 
# 1. The data is big - **1 million rows** and 280+ columns (numerical and binary both)
# 2. The target is **highly balanced**.
# 3. There are **no missing values** to be treated -all continuous features are already scaled (0 to 1)
# 4. The variables have **low correlation** with the target and with each other
# 5. **f179 and f22** have decent correlation - f22 is something worth looking at more.
# 6. Variables with high no. of 1s are **distributed very evenly** with respect to the target variable.
# 
# **Action Points**
# 
# 1. **No feature engineering** required for my first run - but may include 'ones' to see if better.
# 2. Check feature importances through my model - may use for **selection** in the future runs.
# 3. Look out for f22 and f179 in the feature importances
# 4. Use **StratifiedKFold** for creating balanced multiple folds while model run.
# 5. Will see the performance of XGB, CatBoost, LightGBM and RandomForest and decide ensemble.

# **I have done some experiments with the new feature in the following notebook -
# https://www.kaggle.com/raahulsaxena/tps-oct-21-feature-added-catboost-baseline
# Please upvote if you find it useful.**

# <div style="background-color:rgba(215, 159, 21, 0.5);">
#     <h1><center>Please upvote if you like my work. Thanks :)</center></h1>
# </div>
