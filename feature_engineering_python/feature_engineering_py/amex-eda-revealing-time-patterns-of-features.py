#!/usr/bin/env python
# coding: utf-8

# In this notebook I want to test if features are stationary over time.
# 
# Looks like vast amount of features have seasonal patterns - it may affect the model in many aspects: such as data drift and bad performance on unseen data. Also even simple feature engineering may be less accurate - if feature has stable positive trend, then max value will be closer to the last payment.
# We may consider normalize data on daily level to avoid those mistakes. It also may open additional dimension for feature extraction: for example we may use seasonality or trend features as additional features.
# 
# I will calculate an average value for each feature for each day resulting a univariate time series for each feature. I will plot them only if the max autocorrelated component is over a threshold (0.6).

# In[1]:


import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import gc


# In[2]:


train_agg = pd.read_parquet('../input/amex-data-integer-dtypes-parquet-format/train.parquet').assign(S_2=lambda dx: pd.to_datetime(dx.S_2)).groupby('S_2').mean()
end_of_train = pd.to_datetime(train_agg.index).max()


# In[3]:


test_agg = []
for cols2use in train_agg.columns.values.reshape(-1, 47):
    test_agg.append(pd.read_parquet('../input/amex-data-integer-dtypes-parquet-format/test.parquet', columns=cols2use.tolist() + ['S_2']).assign(S_2=lambda dx: pd.to_datetime(dx.S_2)).groupby('S_2').mean())


test_agg = pd.concat(test_agg, axis=1)


# In[4]:


agg_data = pd.concat([train_agg, test_agg])


# In[5]:


for first_letter in list(set([col.split('_')[0] for col in agg_data.columns])):
    break


# In[6]:


for first_letter in list(set([col.split('_')[0] for col in agg_data.columns])):
    for feature_name in agg_data.columns:
        if feature_name[0] != first_letter:
            continue
        s = agg_data.loc[:, feature_name]
        max_acf = np.abs(acf(s, nlags=agg_data.index.size - 1))[1:].max()
        if max_acf > 0.6:
            print(feature_name) # for Ctrl + F
            fig = plt.figure(figsize=(16, 6))
            sub_pacf = fig.add_subplot(2,2,4)
            sub_acf = fig.add_subplot(2,2,3) 
            mn = fig.add_subplot(2,2,(1,2)) 
            plot_pacf(s, lags=agg_data.index.size/2-1, ax=sub_acf)
            plot_acf(s, lags=agg_data.index.size-1, ax=sub_pacf)
            s.plot(color='green', ax=mn)
            mn.axvline(end_of_train, color='red', linestyle='--')
            mn.set_title(feature_name)
            plt.subplots_adjust(wspace= 0.25, hspace= 0.25)
            plt.show()


# In[ ]:




