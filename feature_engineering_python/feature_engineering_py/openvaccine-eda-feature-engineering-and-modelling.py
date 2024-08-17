#!/usr/bin/env python
# coding: utf-8

# ## General information
# ## Copy of https://www.kaggle.com/artgor/openvaccine-eda-feature-engineering-and-modelling
# 
# This competition may help with defeating COVID-19. On of the most promising approaches are mRNA (messenger RNA) vaccines. One of the biggest challenges is developing stable mRNA molecules because they degrade spontaneously abd rapidly.
# 
# Eterna - online video game platform - has many challenges with solving scientific problems and helped to make many advances.
# 
# In this competition we have a subset of Eterna dataset with 3000 RNA molecules and their degradation rates. Our task is to predict those degradation rates. Our models will be scores on a new generation of molecules.
# 
# 
# ![](https://www.ddw-online.com/library/sid32/64-figure-3.jpg)
# 
# 
# ![](https://i.imgur.com/cVMlp16.png)

# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import numpy as np
from collections import Counter
from sklearn.model_selection import RepeatedKFold
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
import random


# ## Step1: Data overview
# 
# This competition has unique and interesting data, let's analyze it.
# 
# First of all - main data is in `json` format, not `csv` which is more common on Kaggle.

# In[13]:


path = '/kaggle/input/stanford-covid-vaccine'
train = pd.read_json(f'{path}/train.json',lines=True)
test = pd.read_json(f'{path}/test.json', lines=True)
sub = pd.read_csv(f'{path}/sample_submission.csv')


# # Size of the train, test and submission - Did you notice that test and submission have different number of rows? This is because test data contains sequences and the submission is flattened. 
# 
# 
# * 629 from 3029 are in public test datased. 2400 other sequences are in train;

# In[14]:


train.shape, train['id'].nunique(), test.shape, sub.shape


# In[15]:


train.head()


# ## One Sample - So we have 2400 samples in train dataset. Let's look at all available information about one sample.
# 
# ## It is very important to remember that both `seq_length` and `seq_scored` are different in train and test. This means that we have to be able to work with sequences of different lengths.
# 
# ## TEST Data `seq_scored`
# * 91   --  3005
# * 68    --  629

# In[16]:


train.info()


# In[17]:


sample = train.loc[train['id'] == 'id_001f94081']
sample


# ## Sequence - This is RNA sequence, a combination of A, G, U, and C for each sample. There is a separate column showing sequence length - `seq_length`. Also notice that there is column `seq_scored` - it shows that this number of positions has target values.

# In[18]:


sample['sequence'].values[0]


# In[19]:


Counter(sample['sequence'].values[0])


# 

# ### Structure An - array of (, ), and . characters. e.g. (....) means that base 0 is paired to base 5, and bases 1-4 are unpaired.

# In[20]:


sample['structure'].values[0]


# In[21]:


Counter(sample['structure'].values[0])


# ### Predicted loop type: Describes the structural context of each character in sequence. Loop types assigned by bpRNA from Vienna RNAfold 2 structure. From the bpRNA_documentation: S: paired "Stem" M: Multiloop I: Internal loop B: Bulge H: Hairpin loop E: dangling End X: eXternal loop

# In[22]:


sample['predicted_loop_type'].values[0]


# In[23]:


Counter(sample['predicted_loop_type'].values[0])


# ## How to filter the data!
# 
# Here is information from "data" tab of this competition:
# 
# * 1. there were 3029 RNA sequences of length 107;
# * 2. measurements can be done only on the first 68 points of sequences;
# * 3. measurements were done in 5 conditions (reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C);
# * 
# * they were filtered using the following criteria:
# 
# > 1. minimal value of conditions > -0.5
# > 2. Mean signal/noise across all conditions > 1.0. calculated as mean value divided by mean error
# > 3. The resulting sequences were clustered into clusters with less than 50% sequence similarity, and the 629 test set sequences were chosen from clusters with 3 or fewer members. That is, any sequence in the test set should be sequence similar to at most 2 other sequences.
# 
# * train data wasn't filtered, so it could make sense to apply first two filters to training data. on the other hand the final scoring will be done on non-filtered data. So I suppose it would be better not to do any manual filtering in the end;

# ## Regression Targets: The rest of the values are our targets and their errors. Note that for scoring we need only to predict `reactivity`, `deg_Mg_pH10` and `deg_Mg_50C`.

# In[24]:


len(sample['reactivity'].values[0])


# ### BBPS
# 
# I'm not sure what this is, but it seems to be some representations of the data.

# In[25]:


mol = np.load('/kaggle/input/stanford-covid-vaccine/bpps/id_001f94081.npy')
plt.imshow(mol);


# ## Train, public test and private test

# In[26]:


train['seq_scored'].value_counts()


# In[27]:


test['seq_scored'].value_counts()


# Do you see that test has 2 unique values in `seq_scored`? rows with `68` are public test, `91` is private test.

# ## Preparing the data
# 
# My idea is the following: let's try working with this data as tabular. To do this, we need to flatten the data. Let's try.

# In[28]:


train.head()


# In[29]:


train_data = []
for mol_id in train['id'].unique():
    sample_data = train.loc[train['id'] == mol_id]
    for i in range(68):
        sample_tuple = (sample_data['id'].values[0], sample_data['sequence'].values[0][i],
                        sample_data['structure'].values[0][i], sample_data['predicted_loop_type'].values[0][i],
                        sample_data['reactivity'].values[0][i], sample_data['reactivity_error'].values[0][i],
                        sample_data['deg_Mg_pH10'].values[0][i], sample_data['deg_error_Mg_pH10'].values[0][i],
                        sample_data['deg_pH10'].values[0][i], sample_data['deg_error_pH10'].values[0][i],
                        sample_data['deg_Mg_50C'].values[0][i], sample_data['deg_error_Mg_50C'].values[0][i],
                        sample_data['deg_50C'].values[0][i], sample_data['deg_error_50C'].values[0][i])
        train_data.append(sample_tuple)


# In[30]:


train_data = pd.DataFrame(train_data, columns=['id', 'sequence', 'structure', 'predicted_loop_type', 'reactivity', 'reactivity_error', 'deg_Mg_pH10', 'deg_error_Mg_pH10',
                                  'deg_pH10', 'deg_error_pH10', 'deg_Mg_50C', 'deg_error_Mg_50C', 'deg_50C', 'deg_error_50C'])
train_data.head()


# In[31]:


fig, ax = plt.subplots(figsize = (24, 10))
for i, col in enumerate(['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C',
       'reactivity_error', 'deg_error_Mg_pH10', 'deg_error_pH10', 'deg_error_Mg_50C', 'deg_error_50C']):
    plt.subplot(2, 5, i + 1);
    plt.hist(train_data[col])
    plt.title(f'{col} histogram');
    plt.xticks(rotation=45)


# In[32]:


train_data.sort_values('reactivity_error')


# In[33]:


train.loc[train['id'] == 'id_a1719ebbc']


# What are these huge errors???

# In[34]:


test_data = []
for mol_id in test['id'].unique():
    sample_data = test.loc[test['id'] == mol_id]
    for i in range(sample_data['seq_scored'].values[0]):
        sample_tuple = (sample_data['id'].values[0] + f'_{i}', sample_data['sequence'].values[0][i],
                        sample_data['structure'].values[0][i], sample_data['predicted_loop_type'].values[0][i])
        test_data.append(sample_tuple)


# In[35]:


test_data = pd.DataFrame(test_data, columns=['id', 'sequence', 'structure', 'predicted_loop_type'])
test_data.head()


# ## Baseline submission
# 
# Let's submit a baseline - mean value by categorical columns

# In[36]:


train_data.groupby(['sequence', 'structure', 'predicted_loop_type'])['reactivity'].mean().reset_index().head()


# In[37]:


test_data = pd.merge(test_data, train_data.groupby(['sequence', 'structure', 'predicted_loop_type'])['reactivity'].mean().reset_index(),
                     on=['sequence', 'structure', 'predicted_loop_type'])
test_data = pd.merge(test_data, train_data.groupby(['sequence', 'structure', 'predicted_loop_type'])['deg_Mg_pH10'].mean().reset_index(),
                     on=['sequence', 'structure', 'predicted_loop_type'])
test_data = pd.merge(test_data, train_data.groupby(['sequence', 'structure', 'predicted_loop_type'])['deg_pH10'].mean().reset_index(),
                     on=['sequence', 'structure', 'predicted_loop_type'])
test_data = pd.merge(test_data, train_data.groupby(['sequence', 'structure', 'predicted_loop_type'])['deg_Mg_50C'].mean().reset_index(),
                     on=['sequence', 'structure', 'predicted_loop_type'])
test_data = pd.merge(test_data, train_data.groupby(['sequence', 'structure', 'predicted_loop_type'])['deg_50C'].mean().reset_index(),
                     on=['sequence', 'structure', 'predicted_loop_type'])


# In[38]:


test_data.head()


# In[39]:


sub.head()


# In[40]:


sub.shape, test_data.shape


# In[41]:


sub1 = pd.merge(sub[['id_seqpos']], test_data, left_on='id_seqpos', right_on='id', how='left').drop(['id', 'sequence', 'structure', 'predicted_loop_type'], axis=1)
sub1.head()


# In[42]:


sub1[['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']] = sub1[['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']].fillna(0) * 0.9


# In[43]:


sub1.to_csv('submission.csv', index=False)

