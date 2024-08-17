#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 

TRAIN_PATH = "../input/titanic/train.csv"
TEST_PATH = "../input/titanic/test.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)


# In[2]:


train.isnull().sum()


# In[3]:


def autoNullDataFeatureEnginering(df):
    null_list = []
    for col in df.columns:
        if df[col].isnull().sum() != 0:
            null_list.append(col)

    for col in null_list:
        df["Has_" + col] = (df[col].isnull() == False).astype(int)
        
        if df[col].dtype == "float64" or df[col].dtype == "int64":
            df.loc[df[col].isnull() == True,col] = df[col].median()
        else:
            df.loc[df[col].isnull() == True,col] = "Missing"
       
    return df


# In[4]:


train = autoNullDataFeatureEnginering(train)
test = autoNullDataFeatureEnginering(test)


# In[5]:


train.columns


# In[6]:


test.columns


# In[7]:


train.isnull().sum()


# In[8]:


train.head()

