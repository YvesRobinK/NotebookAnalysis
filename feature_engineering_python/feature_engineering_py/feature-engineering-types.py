#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

TRAIN_PATH = "/kaggle/input/titanic/train.csv"
TEST_PATH = "/kaggle/input/titanic/test.csv"


# In[2]:


train = pd.read_csv(TRAIN_PATH)
train.head()


# # New Feature(Convert Type)

# ## SexType

# In[3]:


unique = train["Sex"].unique()
print("unique = ",unique)

dictionary = {value : index for index,value in enumerate(unique)}
print("dictionary = ",dictionary)

train = pd.read_csv(TRAIN_PATH)
train["SexType"] = train["Sex"].map(dictionary)
train[["Sex","SexType"]].head()


# ## EmbarkedType

# In[4]:


unique = train["Embarked"].unique()
print("unique = ",unique)

dictionary = {value : index for index,value in enumerate(unique)}
print("dictionary = ",dictionary)

train = pd.read_csv(TRAIN_PATH)
train["EmbarkedType"] = train["Embarked"].map(dictionary)
train[["Embarked","EmbarkedType"]].head()


# # New Feature(Sum Type)

# ## FamilyCount = SibSp + Parch

# In[5]:


train = pd.read_csv(TRAIN_PATH)
train["FamilyCount"] = train["SibSp"] + train["Parch"]
train[["SibSp","Parch","FamilyCount"]].head()


# # New Feature (Cut Type)

# ## FareLevel

# In[6]:


train = pd.read_csv(TRAIN_PATH)
DEVIDE_NUM = 10
train["FareLevel"] = pd.cut(train["Fare"].values,DEVIDE_NUM,labels=np.arange(0,DEVIDE_NUM))
train[["Fare","FareLevel"]].head()


# # New Feature(Log Type)

# ## FareLog

# In[7]:


train = pd.read_csv(TRAIN_PATH)
train["FareLog"] = np.log10(train["Fare"],where=train["Fare"]>0)
train[["Fare","FareLog"]].head()


# In[8]:


train = pd.read_csv(TRAIN_PATH)
train["FareLog"] = np.log1p(train["Fare"])
train[["Fare","FareLog"]].head()

