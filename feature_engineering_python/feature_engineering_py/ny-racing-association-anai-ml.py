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


# #ANAI - Automated ML Library
# 
# "ANAI is an Automated Machine Learning Python Library that works with tabular data. It is intended to save time when performing data analysis. It will assist you with everything right from the beginning i.e Ingesting data using the inbuilt connectors, preprocessing, feature engineering, model building, model evaluation, model tuning and much more."
# 
# https://www.kaggle.com/code/d4rklucif3r/introducing-anai-stroke-prediction
# 
# Kaggler: Arsh Anwar (a.k.a. d4rklucif3r)
# 
# https://github.com/Revca-ANAI/ANAI
# 
# www.anai.io

# ![](https://revca-assets.s3.ap-south-1.amazonaws.com/Blue+Yellow+Futuristic+Virtual+Technology+Blog+Banner.png)https://www.kaggle.com/code/d4rklucif3r/introducing-anai-stroke-prediction

# In[2]:


get_ipython().run_line_magic('pip', 'install anai-opensource==0.1.2-alpha-5')


# In[3]:


import anai
from anai.preprocessing import Preprocessor
import plotly.express as px


# In[4]:


df4 = anai.load(df_filepath = '../input/big-data-derby-2022/nyra_start_table.csv')


# In[5]:


df4.head(2)


# In[ ]:


#Save for next time since we don't have Missing Values

df1 = prep.impute(method = 'mean')


# In[ ]:


#Save for next time since we don't have Missing Values

df1.isna().sum()


# In[6]:


prep = Preprocessor(dataset = df4, target = 'odds')


# In[7]:


features, labels = prep.encode(split = True)


# In[8]:


features.head(4)


# In[9]:


X_train, X_val, y_train, y_val, scaler = prep.prepare(features, labels, test_size = 0.2, random_state = 42, smote = False, k_neighbors = 3)


# In[10]:


X_train.shape, X_val.shape, y_val.shape, y_train.shape


# In[ ]:


#Code by Arsh Anwar https://www.kaggle.com/code/d4rklucif3r/introducing-anai-stroke-prediction

ai = anai.run(filepath = '../input/big-data-derby-2022/nyra_start_table.csv', suppress_task_detection=True, task = 'regression')


# #ANAI's AutoML Pipeline
# 
# Below, I deleted xgb, so that it would takes less time since I was just trying it for the 1st time. 

# In[11]:


#Code by Arsh Anwar https://www.kaggle.com/code/d4rklucif3r/introducing-anai-stroke-prediction

ai = anai.run(filepath = '../input/big-data-derby-2022/nyra_start_table.csv', target = 'odds', suppress_task_detection=True, task = 'regression',  predictor = ['ada', 'cat', 'knn', 'lgbm'])


# #Explainable ANAI

# In[14]:


ai.explain('shap')


# #Explainable ANAI

# In[12]:


ai.explain('perm')


# #LB of the chosen models

# In[13]:


ai.result()


# #Acknowledgements:
# 
# Arsh Anwar (d4rklucif3r)
# 
# https://www.kaggle.com/code/d4rklucif3r/introducing-anai-stroke-prediction
# 
# https://www.kaggle.com/code/d4rklucif3r/dangerous-or-not-anai-viz
# 
# www.anai.io
# 
# https://github.com/Revca-ANAI/ANAI
