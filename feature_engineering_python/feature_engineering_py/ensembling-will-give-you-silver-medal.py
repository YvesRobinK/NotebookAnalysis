#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob


# This notebook ensembles four different submissions and weights them according to the model's different performance. It is based on the following notebook:
# https://www.kaggle.com/code/finlay/amex-rank-ensemble

# In[2]:


paths = ['../input/lag-features-are-all-you-need/submission.csv',
 '../input/amex-lgbm-dart-cv-0-7963-improved/submission.csv',
 '../input/amex-lightautoml-starter/lightautoml_tabularautoml.csv',
 '../input/expressions-of-gluttony/submission.csv']


# In[3]:


#['../input/lag-features-are-all-you-need/submission.csv', --> 0.797 0
# '../input/amex-lgbm-dart-cv-0-7963-improved/submission.csv', --> 0.799 1
# '../input/amex-lightautoml-starter/lightautoml_tabularautoml.csv', --> 0.795 2
# '../input/expressions-of-gluttony/submission.csv'] --> 0.796 3


# In[4]:


dfs = [pd.read_csv(x) for x in paths]
dfs = [x.sort_values(by='customer_ID') for x in dfs]


# In[5]:


for df in dfs:
    df['prediction'] = np.clip(df['prediction'], 0, 1)


# In[6]:


pred_ensembled = 0.529 * dfs[1]['prediction'] + 0.441 * dfs[0]['prediction'] + 0.02 * dfs[3]['prediction'] + 0.01 * dfs[2]['prediction']


# In[7]:


submit = pd.read_csv('../input/amex-default-prediction/sample_submission.csv')


# In[8]:


submit['prediction'] = pred_ensembled


# In[9]:


submit.to_csv('submission', index=False)

