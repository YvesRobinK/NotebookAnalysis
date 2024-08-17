#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgbm
from lightgbm import *


# In[2]:


df = pd.read_parquet('../input/ubiquant-parquet/train_low_mem.parquet')


# In[3]:


def reduce_mem_usage(df):
  
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
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
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
     
    return df

df = reduce_mem_usage(df)


# As EDA, many people are checking the distribution of features. For example, you can make a histogram like this. This is a histogram of 'f14'.

# In[4]:


df['f_14'].hist(bins = 100)


# Looking at the distributions of these feature values, some distributions are beautifully symmetrical, while others are asymmetrical. I don't know the actual calculation formulas, but it seems that these characteristics appear due to the way the feature values are calculated. Therefore, I thought of incorporating how much the values of these features are out of the middle as a feature value. The specific formula for this is
# 
# abs(feature value - median value of feature value) 
# 
# 

# I will implement the code and test the hypothesis in the following.

# show 'f_146' 's histogram

# In[5]:


df['f_146'].hist(bins = 100)


# As you can see, this is a beautiful symmetrical distribution.
# Next, let's process the features.
# This time, we will see if the processing of 'f146' is working, but since we are here, we will calculate the difference from the median from the median all together.

# In[6]:


features = [f'f_{i}' for i in range(300)]
df_median = df[features].head(30000).median()
print(df_median)


# In[7]:


for i in range(300):
    df[f'f_median_{i}'] = abs(df[f'f_{i}']-df_median[f'f_{i}'])

print(df)


# To check if this feature is actually working, we will use LightGBM to check the feature importance.
# 
# We will split the data into train, val, and test data, making sure that the rows used for the median calculation are not included in the training data.

# In[8]:


from sklearn.model_selection import KFold, train_test_split

#
df = df.tail(3101410)
features = [f'f_{i}' for i in range(300)] + [f'f_median_146']
target = 'target'

df_features = df[features]

X_train, X, Y_train, Y = train_test_split(df_features, df[target], train_size=0.6, shuffle=False)

df = [[]]
df_features = [[]]

X_val, X_test, Y_val, Y_test = train_test_split(X, Y, train_size=0.5, shuffle=False)


# 

# In[9]:


import warnings
import numpy as np
import lightgbm as lgb
from scipy.stats import pearsonr

warnings.simplefilter('ignore')

lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_val, Y_val, reference=lgb_train)


params = {'seed': 1,
          'verbose' : -1,
           'objective': "regression",
           'learning_rate': 0.02,
           'bagging_fraction': 0.2,
           'bagging_freq': 1,
           'feature_fraction': 0.3,
           'max_depth': 5,
           'min_child_samples': 50,
           'num_leaves': 64}


        
        
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=lgb_eval,
                verbose_eval=False,
                early_stopping_rounds=5,
                )


Y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

score_tuple = pearsonr(Y_test, Y_pred)
score = score_tuple[0]
print(f"Validation Pearsonr score : {score:.4f}")


# In[10]:


import matplotlib.pyplot as plt





feature = gbm.feature_importance(importance_type='gain')


f = pd.DataFrame({'number': range(0, len(feature)),
             'feature': feature[:]})
f2 = f.sort_values('feature',ascending=False)


label = X_train.columns[0:]


indices = np.argsort(feature)[::-1]

for i in range(len(feature)):
   print(str(i + 1) + "   " + str(label[indices[i]]) + "   " + str(feature[indices[i]]))



# When we check the 246th position of the feature importance, we can see that "f_median_146" is indeed effective.
# 
# There are other features with symmetrical distributions, so it is likely that there are features that can be processed in a similar way.
