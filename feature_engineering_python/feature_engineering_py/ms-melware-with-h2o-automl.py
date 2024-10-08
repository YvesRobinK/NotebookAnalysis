#!/usr/bin/env python
# coding: utf-8

# [H2O AutoML](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) is an automated machine learning meta-algorithm that is part of the [H2O software library](http://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/intro.html#what-is-h2o). (It shold not be confused with [H2O DriverlessAI](https://www.h2o.ai/products/h2o-driverless-ai/), which is a commercial product and built from an entirely different code base.) H2O’s AutoML can be used for automating the machine learning workflow, which includes automatic training and tuning of many models within a user-specified time-limit. Stacked Ensembles – one based on all previously trained models, another one on the best model of each family – will be automatically trained on collections of individual models to produce highly predictive ensemble models which, in most cases, will be the top performing models in the AutoML Leaderboard.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


import h2o
from h2o.automl import H2OAutoML
print(h2o.__version__)

h2o.init(max_mem_size='17G')


# In[3]:


train = h2o.import_file("../input/malware-feature-engineering-only/new_train.csv")
train.head()


# In[4]:


x = train.columns[1:-1]
y = 'HasDetections'


# In[5]:


# For binary classification, response should be a factor
train[y] = train[y].asfactor()


# In[6]:


# Run AutoML for 20 base models (limited to 1 hour max runtime by default)
aml = H2OAutoML(max_runtime_secs=30000, seed=13)
aml.train(x=x, y=y, training_frame=train)


# In[7]:


# View the AutoML Leaderboard
lb = aml.leaderboard
lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)


# In[8]:


h2o.remove(train.frame_id)
test = h2o.import_file("../input/malware-feature-engineering-only/new_test.csv")
preds = aml.predict(test)
h2o.remove(test.frame_id)


# In[9]:


sample_submission = pd.read_csv('../input/microsoft-malware-prediction/sample_submission.csv')
sample_submission.head()


# In[10]:


sample_submission['HasDetections'] = preds['p1'].as_data_frame().values
sample_submission.to_csv('submission.csv', index=False)


# In[ ]:




