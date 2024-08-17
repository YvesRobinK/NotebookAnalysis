#!/usr/bin/env python
# coding: utf-8

# I  just began this competition two days ago. I have been having a rough time creating numeric features I could feed through a machine learning algorithm from the pandas Series of lists contained in train['features']. So I've been doing some studying to improve my meager Python skills and the solution below is what I have arrived at. It quickly and easily makes a new feature out of any item from the train.features/test.features column that you might want to look at.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.mode.chained_assignment = None

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# One quick note: because of the limiting computing power available in Kaggle kernels I have chopped the dataframe down to just ten rows. This allows me to build and then demonstrate the usefulness of my little function here without causing my kernel to die an inglorious death.
# 
# Viva kernels!

# In[2]:


train = pd.read_json('../input/train.json')
train = train.iloc[:10, :]

def newfeat(name, df, series):
    """Create a Series for my feature building loop to fill"""
    feature = pd.Series(0, df.index, name=name)
    """Now populate the new Series with numeric values"""
    for row, word in enumerate(series):
        if name in word:
            feature.iloc[row] = 1
    df[name] = feature
    return(df)
   
train = newfeat('Elevator', train, train.features)
train = newfeat('Dogs Allowed', train, train.features)
train = newfeat('Cats Allowed', train, train.features)

print(train)


# So now I have numeric features corresponding to whether a given listing has an elevator or allows dogs and cats. 
# 
# Of course this kernel doesn't address the obvious issue of misspellings, use of hyphens, alternate word choices, or anything else that might throw off the accuracy of my new features. That is a project for another kernel.
# 
# On a side note, I find it humorous that I felt compelled to figure out how to do this because I wanted to answer a single simple question: Is any given apartment pet friendly or not? And now I can find out!

# In[3]:


train['pet_friendly'] = train['Cats Allowed'] + train['Dogs Allowed']
print(train['pet_friendly'])


# Well, that's all I have. Please upvote this kernel if you decide to use my code or you find what I've done here useful. I'm always in the hunt for medals, even if I am a beginner at this stuff. My observation to date has been that a combination of thoughtful feature engineering and hyperparameter tuning are the key to improving machine learning models. Hopefully this provides some fodder for the former.
