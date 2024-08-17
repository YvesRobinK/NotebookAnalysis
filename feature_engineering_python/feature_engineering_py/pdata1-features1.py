#!/usr/bin/env python
# coding: utf-8

# 1. pdata1-features1:**current notebook** feature engineering https://www.kaggle.com/quincyqiang/pdata1-features1
# 2. pdata1-lgb-train：https://www.kaggle.com/quincyqiang/pdata1-lgb-train
# 2. pdata1-lgb-inference：：inference https://www.kaggle.com/quincyqiang/pdata1-features1

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm 
from glob import glob
from imp import reload
import copy as cp
import sys
sys.path.append('../input/features-util')
# from utils import util
import util
reload(util)
import joblib


# In[2]:


get_ipython().system('ls /kaggle/input')


# In[3]:


import os
path_lst = glob('../input/optiver-realized-volatility-prediction/book_train.parquet/*')
stock_lst = [os.path.basename(path).split('=')[-1] for path in path_lst]


# In[4]:


print(len(stock_lst))


# In[5]:


temp = util.gen_data_train(0)


# In[6]:


data_type = 'train'
fe_df = util.gen_data_multi(stock_lst, data_type)


# In[7]:


fe_df.to_pickle('train_stock_df.pkl')


# In[8]:


fe_df.columns


# In[ ]:




