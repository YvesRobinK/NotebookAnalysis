#!/usr/bin/env python
# coding: utf-8

# <h1><center>OpenVaccine || EDA || Feature engineering || Modeling</center></h1>
# 
# <center><img src="https://daslab.stanford.edu/site_data/news_img/openvaccine_lores.png"></center>

# ### In this kernel I am going to present some basic data overview, feature engineering and prepare keras neural network model. Let's fo it and have fun!
# 
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:black; border:0; color:white' role="tab" aria-controls="home"><center>Quick navigation</center></h2>
# 
# * [1. Quick Data Overview](#1)
# * [2. Sample Analysis](#2)
# * [3. Feature Engineering](#3)
# * [4. Keras Neural Network Model](#4)
# * [5. Prepare submission file](#5)

# <a id="1"></a>
# <h2 style='background:black; border:0; color:white'><center>1. Quick Data Overview</center><h2>

# In[1]:


import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import plotly.express as px
from collections import Counter as count

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import KFold


# In[2]:


train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)
test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)
sub = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')

print('Train shapes: ', train.shape)
print('Test shapes: ', test.shape)


# #### So we have 2400 sequences in training set and 3634 in test set.

# In[3]:


train.head(10)


# In[4]:


test.head(5)


# In[5]:


sub


# #### Let's check all training features

# In[6]:


train.info()


# In[7]:


train.describe()


# In[8]:


train["seq_scored"].value_counts()


# #### As we can see every sequence from training set has only 68 scored bases (first 68 bases).

# In[9]:


fig = px.histogram(
    train, 
    "signal_to_noise", 
    nbins=25, 
    title='signal_to_noise column distribution', 
    width=700,
    height=500
)
fig.show()


# In[10]:


ds = train['SN_filter'].value_counts().reset_index()
ds.columns = ['SN_filter', 'count']
fig = px.pie(
    ds, 
    values='count', 
    names="SN_filter", 
    title='SN_filter bar chart', 
    width=500, 
    height=500
)
fig.show()


# In[11]:


train['seq_length'].value_counts()


# In[12]:


test['seq_length'].value_counts()


# In[13]:


train['seq_scored'].value_counts()


# In[14]:


test['seq_scored'].value_counts()


# <a id="2"></a>
# <h2 style='background:black; border:0; color:white'><center>2. Sample Analysis</center><h2>

# #### Let's explore 1 sample from train set. We will focus on some columns and see values.

# #### We have 3 columns that represent structure of sequence: sequence, structure and predicted_loop_type

# In[15]:


sample = train.iloc[0]
sample


# ### 1. Sequence

# ### We have 4 possible nitrogeneous bases for RNA:
# 
# 1) Guanine (G) <br>
# 2) Adenine (A) <br>
# 3) Cytosine (C) <br>
# 4) Uracil (U) <br>
# 
# #### For more details you can check <a href="https://en.wikipedia.org/wiki/Nucleobase">here.</a>
# 
# <center><img src="https://www.thoughtco.com/thmb/jnQVk0_RZ4TRJHeFKR7xxqSV1Pk=/1500x1000/filters:fill(auto,1)/dna-versus-rna-608191_sketch_Final-54acdd8f8af04c73817e8811c32905fa.png" width="700" height="500"></center>

# In[16]:


sample['sequence']


# In[17]:


dict(count(sample['sequence']))


# In[18]:


bases = []

for j in range(len(train)):
    counts = dict(count(train.iloc[j]['sequence']))
    bases.append((
        counts['A'] / 107,
        counts['G'] / 107,
        counts['C'] / 107,
        counts['U'] / 107
    ))
    
bases = pd.DataFrame(bases, columns=['A_percent', 'G_percent', 'C_percent', 'U_percent'])
bases


# #### The length of sequence should be equal to value in ```seq_length``` column

# In[19]:


len(sample['sequence']) == sample['seq_length']


# ### 2. Structure

# In[20]:


sample['structure']


# #### Here we can see 3 different types of characters. Chaacter ```.``` means that base is without pair. ```(``` - is start of pair, ```)``` - the end for current pair. So the number of ```(``` should be equal to ```)```.

# In[21]:


dict(count(sample['structure']))


# In[22]:


pairs_rate = []

for j in range(len(train)):
    res = dict(count(train.iloc[j]['structure']))
    pairs_rate.append(res['('] / 53.5)
    
pairs_rate = pd.DataFrame(pairs_rate, columns=['pairs_rate'])
pairs_rate


# #### Let's check all pairs for our sample

# In[23]:


pairs_dict = {}
queue = []
for i in range(0, len(sample['structure'])):
    if sample['structure'][i] == '(':
        queue.append(i)
    if sample['structure'][i] == ')':
        first = queue.pop()
        try:
            pairs_dict[(sample['sequence'][first], sample['sequence'][i])] += 1
        except:
            pairs_dict[(sample['sequence'][first], sample['sequence'][i])] = 1
pairs_dict


# In[24]:


pairs = []
all_partners = []
for j in range(len(train)):
    partners = [-1 for i in range(130)]
    pairs_dict = {}
    queue = []
    for i in range(0, len(train.iloc[j]['structure'])):
        if train.iloc[j]['structure'][i] == '(':
            queue.append(i)
        if train.iloc[j]['structure'][i] == ')':
            first = queue.pop()
            try:
                pairs_dict[(train.iloc[j]['sequence'][first], train.iloc[j]['sequence'][i])] += 1
            except:
                pairs_dict[(train.iloc[j]['sequence'][first], train.iloc[j]['sequence'][i])] = 1
                
            partners[first] = i
            partners[i] = first
    
    all_partners.append(partners)
    
    pairs_num = 0
    pairs_unique = [('U', 'G'), ('C', 'G'), ('U', 'A'), ('G', 'C'), ('A', 'U'), ('G', 'U')]
    for item in pairs_dict:
        pairs_num += pairs_dict[item]
    add_tuple = list()
    for item in pairs_unique:
        try:
            add_tuple.append(pairs_dict[item]/pairs_num)
        except:
            add_tuple.append(0)
    pairs.append(add_tuple)
    
pairs = pd.DataFrame(pairs, columns=['U-G', 'C-G', 'U-A', 'G-C', 'A-U', 'G-U'])
pairs


# In[25]:


train['partners'] = all_partners


# #### Let's do it for all samples

# In[26]:


pairs_dict = {}
queue = []
for j in range(len(train)):
    sam = train.iloc[j]
    for i in range(0, len(sam['structure'])):
        if sam['structure'][i] == '(':
            queue.append(i)
        if sam['structure'][i] == ')':
            first = queue.pop()
            try:
                pairs_dict[(sam['sequence'][first], sam['sequence'][i])] += 1
            except:
                pairs_dict[(sam['sequence'][first], sam['sequence'][i])] = 1
                
pairs_dict


# #### Basically I don't know now is ('C', 'G') and ('G', 'C') the same - so I leave it as is.

# In[27]:


names = []
values = []
for item in pairs_dict:
    names.append(item)
    values.append(pairs_dict[item])
    
df = pd.DataFrame()
df['pair'] = names
df['count'] = values
df['pair'] = df['pair'].astype(str)

fig = px.bar(
    df, 
    x='pair', 
    y="count", 
    orientation='v', 
    title='Pair types', 
    height=400, 
    width=800
)
fig.show()


# #### We can see that the most popular pair is with G and C, the less popular with U and G. And there is only 3 possible combinations of pairs - G and C, U and G, U and A.

# ### 3. Predicted loop type

# In[28]:


sample['predicted_loop_type']


# S: paired "Stem" <br>
# M: Multiloop  <br>
# I: Internal loop <br>
# B: Bulge <br>
# H: Hairpin loop <br>
# E: dangling End <br>
# X: eXternal loop <br>

# In[29]:


dict(count(sample['predicted_loop_type']))


# In[30]:


loops = []
for j in range(len(train)):
    counts = dict(count(train.iloc[j]['predicted_loop_type']))
    available = ['E', 'S', 'H', 'B', 'X', 'I', 'M']
    row = []
    for item in available:
        try:
            row.append(counts[item] / 107)
        except:
            row.append(0)
    loops.append(row)
    
loops = pd.DataFrame(loops, columns=available)
loops


# #### Let's check for all samples

# In[31]:


res_dict = {}
for j in range(len(train)):
    sam = train.iloc[j]
    prom = dict(count(sam['predicted_loop_type']))
    for item in prom:
        try:
            res_dict[item] += prom[item]
        except:
            res_dict[item] = prom[item]
res_dict


# In[32]:


names = []
values = []
for item in res_dict:
    names.append(item)
    values.append(res_dict[item])
    
df = pd.DataFrame()
df['loop_type'] = names
df['count'] = values


# In[33]:


fig = px.bar(
    df, 
    x='loop_type', 
    y="count", 
    orientation='v', 
    title='Predicted loop types', 
    height=400, 
    width=600
)
fig.show()


# <a id="3"></a>
# <h2 style='background:black; border:0; color:white'><center>3. Feature Engineering</center><h2>

# ### From documentation:
# 
# ```
# At the beginning of the competition, Stanford scientists have data on 3029 RNA sequences of length 107. 
# For technical reasons, measurements cannot be carried out on the final bases of these RNA sequences, so we have experimental data (ground truth) in 5 conditions for the first 68 bases.
# ```

# In[34]:


train = pd.concat([train, bases, pairs, loops, pairs_rate], axis=1)
train


# In[35]:


train.columns


# In[36]:


train_data = []
for mol_id in train['id'].unique():
    sample_data = train.loc[train['id'] == mol_id]
    for i in range(68): 
        if i < 3:
            previousA = -1
            previousB = -1
            previousC = -1
        else:
            if i%3 == 0:
                previousA = sample_data['sequence'].values[0][i - 3]
                previousB = sample_data['sequence'].values[0][i - 2]
                previousC = sample_data['sequence'].values[0][i - 1]
            if i%3 == 1:
                previousA = sample_data['sequence'].values[0][i - 4]
                previousB = sample_data['sequence'].values[0][i - 3]
                previousC = sample_data['sequence'].values[0][i - 2]
            if i%3 == 2:
                previousA = sample_data['sequence'].values[0][i - 5]
                previousB = sample_data['sequence'].values[0][i - 4]
                previousC = sample_data['sequence'].values[0][i - 3]
            
            
        if i%3 == 0:
            a = sample_data['sequence'].values[0][i]
            b = sample_data['sequence'].values[0][i + 1]
            c = sample_data['sequence'].values[0][i + 2]
            
            nextA = sample_data['sequence'].values[0][i + 3]
            nextB = sample_data['sequence'].values[0][i + 4]
            nextC = sample_data['sequence'].values[0][i + 5]
            next2A = sample_data['sequence'].values[0][i + 6]
            next2B = sample_data['sequence'].values[0][i + 7]
            next2C = sample_data['sequence'].values[0][i + 8]
            next3A = sample_data['sequence'].values[0][i + 9]
            next3B = sample_data['sequence'].values[0][i + 10]
            next3C = sample_data['sequence'].values[0][i + 11]
            
        if i%3 == 1:
            a = sample_data['sequence'].values[0][i - 1]
            b = sample_data['sequence'].values[0][i]
            c = sample_data['sequence'].values[0][i + 1]
            
            nextA = sample_data['sequence'].values[0][i + 2]
            nextB = sample_data['sequence'].values[0][i + 3]
            nextC = sample_data['sequence'].values[0][i + 4]
            next2A = sample_data['sequence'].values[0][i + 5]
            next2B = sample_data['sequence'].values[0][i + 6]
            next2C = sample_data['sequence'].values[0][i + 7]
            next3A = sample_data['sequence'].values[0][i + 8]
            next3B = sample_data['sequence'].values[0][i + 9]
            next3C = sample_data['sequence'].values[0][i + 10]
            
        if i%3 == 2:
            a = sample_data['sequence'].values[0][i - 2]
            b = sample_data['sequence'].values[0][i - 1]
            c = sample_data['sequence'].values[0][i]
            
            nextA = sample_data['sequence'].values[0][i + 1]
            nextB = sample_data['sequence'].values[0][i + 2]
            nextC = sample_data['sequence'].values[0][i + 3]
            next2A = sample_data['sequence'].values[0][i + 4]
            next2B = sample_data['sequence'].values[0][i + 5]
            next2C = sample_data['sequence'].values[0][i + 6]
            next3A = sample_data['sequence'].values[0][i + 7]
            next3B = sample_data['sequence'].values[0][i + 8]
            next3C = sample_data['sequence'].values[0][i + 9]
            
        if a==b and b==c:
            all_the_same = 1
        else:
            all_the_same = 0
            
        if sample_data['structure'].values[0][i] == ')' or sample_data['structure'].values[0][i] == '(':
            isPair = 1
        else:
            isPair = 0
        
        partner_index = sample_data['partners'].values[0][i]
        if partner_index != -1:
            partner =  sample_data['sequence'].values[0][partner_index]
        else:
            partner = -1
        
        sample_tuple = (
            sample_data['id'].values[0], 
            sample_data['sequence'].values[0][i],
            sample_data['structure'].values[0][i], 
            sample_data['predicted_loop_type'].values[0][i],
            sample_data['reactivity'].values[0][i], 
            sample_data['reactivity_error'].values[0][i],
            sample_data['deg_Mg_pH10'].values[0][i], 
            sample_data['deg_error_Mg_pH10'].values[0][i],
            sample_data['deg_pH10'].values[0][i], 
            sample_data['deg_error_pH10'].values[0][i],
            sample_data['deg_Mg_50C'].values[0][i], 
            sample_data['deg_error_Mg_50C'].values[0][i],
            sample_data['deg_50C'].values[0][i], 
            sample_data['deg_error_50C'].values[0][i],
            sample_data['A_percent'].values[0], 
            sample_data['G_percent'].values[0],
            sample_data['C_percent'].values[0], 
            sample_data['U_percent'].values[0],
            sample_data['U-G'].values[0], 
            sample_data['C-G'].values[0],
            sample_data['U-A'].values[0], 
            sample_data['G-C'].values[0],
            sample_data['A-U'].values[0], 
            sample_data['G-U'].values[0], 
            sample_data['E'].values[0],
            sample_data['S'].values[0], 
            sample_data['H'].values[0],
            sample_data['B'].values[0], 
            sample_data['X'].values[0],
            sample_data['I'].values[0], 
            sample_data['M'].values[0],
            sample_data['pairs_rate'].values[0],
            i%3,
            a,
            b,
            c,
            (i%107) / 68,
            all_the_same, 
            isPair,
            previousA,
            previousB,
            previousC,
            nextA,
            nextB,
            nextC,
            next2A,
            next2B,
            next2C,
            next3A,
            next3B,
            next3C,
            partner
        )
        train_data.append(sample_tuple)


# In[37]:


train_data = pd.DataFrame(
    train_data, 
    columns=[
        'id', 
        'sequence', 
        'structure', 
        'predicted_loop_type', 
        'reactivity', 
        'reactivity_error', 
        'deg_Mg_pH10', 
        'deg_error_Mg_pH10',
        'deg_pH10', 
        'deg_error_pH10', 
        'deg_Mg_50C', 
        'deg_error_Mg_50C', 
        'deg_50C', 
        'deg_error_50C',
        'A_percent',
        'G_percent',
        'C_percent',
        'U_percent',
        'U-G', 
        'C-G',
        'U-A', 
        'G-C',
        'A-U', 
        'G-U', 
        'E',
        'S', 
        'H',
        'B', 
        'X',
        'I', 
        'M',
        'pairs_rate',
        'codon_position',
        'base_0',
        'base_1',
        'base_2',
        'general_position',
        'all_bases_same',
        'isPair',
        'prevCodon_0',
        'prevCodon_1',
        'prevCodon_2',
        'nextCodon_0',
        'nextCodon_1',
        'nextCodon_2',
        'next2Codon_0',
        'next2Codon_1',
        'next2Codon_2',
        'next3Codon_0',
        'next3Codon_1',
        'next3Codon_2',
        'partner'
    ])
train_data


# In[38]:


bases = []
for j in range(len(test)):
    counts = dict(count(test.iloc[j]['sequence']))
    bases.append((
        counts['A'] / test.iloc[j]['seq_length'],
        counts['G'] / test.iloc[j]['seq_length'],
        counts['C'] / test.iloc[j]['seq_length'],
        counts['U'] / test.iloc[j]['seq_length']
    ))
    
bases = pd.DataFrame(bases, columns=['A_percent', 'G_percent', 'C_percent', 'U_percent'])
bases


# In[39]:


pairs = []
all_partners = []
for j in range(len(test)):
    partners = [-1 for i in range(130)]
    pairs_dict = {}
    queue = []
    for i in range(0, len(test.iloc[j]['structure'])):
        if test.iloc[j]['structure'][i] == '(':
            queue.append(i)
        if test.iloc[j]['structure'][i] == ')':
            first = queue.pop()
            try:
                pairs_dict[(test.iloc[j]['sequence'][first], test.iloc[j]['sequence'][i])] += 1
            except:
                pairs_dict[(test.iloc[j]['sequence'][first], test.iloc[j]['sequence'][i])] = 1
                
            partners[first] = i
            partners[i] = first
    
    all_partners.append(partners)
    
    pairs_num = 0
    pairs_unique = [('U', 'G'), ('C', 'G'), ('U', 'A'), ('G', 'C'), ('A', 'U'), ('G', 'U')]
    for item in pairs_dict:
        pairs_num += pairs_dict[item]
    add_tuple = list()
    for item in pairs_unique:
        try:
            add_tuple.append(pairs_dict[item]/pairs_num)
        except:
            add_tuple.append(0)
    pairs.append(add_tuple)
    
pairs = pd.DataFrame(pairs, columns=['U-G', 'C-G', 'U-A', 'G-C', 'A-U', 'G-U'])
pairs


# In[40]:


test['partners'] = all_partners


# In[41]:


pairs_rate = []
for j in range(len(test)):
    res = dict(count(test.iloc[j]['structure']))
    pairs_rate.append(res['('] / (test.iloc[j]['seq_length']/2))
    
pairs_rate = pd.DataFrame(pairs_rate, columns=['pairs_rate'])
pairs_rate


# In[42]:


loops = []
for j in range(len(test)):
    counts = dict(count(test.iloc[j]['predicted_loop_type']))
    available = ['E', 'S', 'H', 'B', 'X', 'I', 'M']
    row = []
    for item in available:
        try:
            row.append(counts[item] / test.iloc[j]['seq_length'])
        except:
            row.append(0)
    loops.append(row)
    
loops = pd.DataFrame(loops, columns=available)
loops


# In[43]:


test = pd.concat([test, bases, pairs, loops, pairs_rate], axis=1)
test


# In[44]:


test_data = []
for mol_id in test['id'].unique():
    sample_data = test.loc[test['id'] == mol_id]
    for i in range(sample_data['seq_scored'].values[0]):
        if i < 3:
            previousA = -1
            previousB = -1
            previousC = -1
        else:
            if i%3 == 0:
                previousA = sample_data['sequence'].values[0][i - 3]
                previousB = sample_data['sequence'].values[0][i - 2]
                previousC = sample_data['sequence'].values[0][i - 1]
            if i%3 == 1:
                previousA = sample_data['sequence'].values[0][i - 4]
                previousB = sample_data['sequence'].values[0][i - 3]
                previousC = sample_data['sequence'].values[0][i - 2]
            if i%3 == 2:
                previousA = sample_data['sequence'].values[0][i - 5]
                previousB = sample_data['sequence'].values[0][i - 4]
                previousC = sample_data['sequence'].values[0][i - 3]
                    
        if i%3 == 0:
            a = sample_data['sequence'].values[0][i]
            b = sample_data['sequence'].values[0][i + 1]
            c = sample_data['sequence'].values[0][i + 2]
            
            nextA = sample_data['sequence'].values[0][i + 3]
            nextB = sample_data['sequence'].values[0][i + 4]
            nextC = sample_data['sequence'].values[0][i + 5]
            next2A = sample_data['sequence'].values[0][i + 6]
            next2B = sample_data['sequence'].values[0][i + 7]
            next2C = sample_data['sequence'].values[0][i + 8]
            next3A = sample_data['sequence'].values[0][i + 9]
            next3B = sample_data['sequence'].values[0][i + 10]
            next3C = sample_data['sequence'].values[0][i + 11]
            
        if i%3 == 1:
            a = sample_data['sequence'].values[0][i - 1]
            b = sample_data['sequence'].values[0][i]
            c = sample_data['sequence'].values[0][i + 1]
            
            nextA = sample_data['sequence'].values[0][i + 2]
            nextB = sample_data['sequence'].values[0][i + 3]
            nextC = sample_data['sequence'].values[0][i + 4]
            next2A = sample_data['sequence'].values[0][i + 5]
            next2B = sample_data['sequence'].values[0][i + 6]
            next2C = sample_data['sequence'].values[0][i + 7]
            next3A = sample_data['sequence'].values[0][i + 8]
            next3B = sample_data['sequence'].values[0][i + 9]
            next3C = sample_data['sequence'].values[0][i + 10]
            
        if i%3 == 2:
            a = sample_data['sequence'].values[0][i - 2]
            b = sample_data['sequence'].values[0][i - 1]
            c = sample_data['sequence'].values[0][i]
            
            nextA = sample_data['sequence'].values[0][i + 1]
            nextB = sample_data['sequence'].values[0][i + 2]
            nextC = sample_data['sequence'].values[0][i + 3]
            next2A = sample_data['sequence'].values[0][i + 4]
            next2B = sample_data['sequence'].values[0][i + 5]
            next2C = sample_data['sequence'].values[0][i + 6]
            next3A = sample_data['sequence'].values[0][i + 7]
            next3B = sample_data['sequence'].values[0][i + 8]
            next3C = sample_data['sequence'].values[0][i + 9]
            
        if a==b and b==c:
            all_the_same = 1
        else:
            all_the_same = 0
            
        if sample_data['structure'].values[0][i] == ')' or sample_data['structure'].values[0][i] == '(':
            isPair = 1
        else:
            isPair = 0
            
        partner_index = sample_data['partners'].values[0][i]
        if partner_index != -1:
            partner =  sample_data['sequence'].values[0][partner_index]
        else:
            partner = -1
            
        sample_tuple = (
            sample_data['id'].values[0] + f'_{i}', 
            sample_data['sequence'].values[0][i],
            sample_data['structure'].values[0][i], 
            sample_data['predicted_loop_type'].values[0][i],
            sample_data['A_percent'].values[0], 
            sample_data['G_percent'].values[0],
            sample_data['C_percent'].values[0], 
            sample_data['U_percent'].values[0],
            sample_data['U-G'].values[0], 
            sample_data['C-G'].values[0],
            sample_data['U-A'].values[0], 
            sample_data['G-C'].values[0],
            sample_data['A-U'].values[0], 
            sample_data['G-U'].values[0], 
            sample_data['E'].values[0],
            sample_data['S'].values[0], 
            sample_data['H'].values[0],
            sample_data['B'].values[0], 
            sample_data['X'].values[0],
            sample_data['I'].values[0], 
            sample_data['M'].values[0],
            sample_data['pairs_rate'].values[0],
            i%3,
            a,
            b,
            c,
            (i%sample_data['seq_scored'].values[0]) / sample_data['seq_scored'].values[0],
            all_the_same, 
            isPair,
            previousA,
            previousB,
            previousC,
            nextA,
            nextB,
            nextC,
            next2A,
            next2B,
            next2C,
            next3A,
            next3B,
            next3C,
            partner
        )
        test_data.append(sample_tuple)


# In[45]:


test_data = pd.DataFrame(
    test_data, 
    columns=[
        'id', 
        'sequence', 
        'structure', 
        'predicted_loop_type', 
        'A_percent',
        'G_percent',
        'C_percent',
        'U_percent',
        'U-G', 
        'C-G',
        'U-A', 
        'G-C',
        'A-U', 
        'G-U', 
        'E',
        'S', 
        'H',
        'B', 
        'X',
        'I', 
        'M',
        'pairs_rate',
        'codon_position',
        'base_0',
        'base_1',
        'base_2',
        'general_position',
        'all_bases_same',
        'isPair',
        'prevCodon_0',
        'prevCodon_1',
        'prevCodon_2',        
        'nextCodon_0',
        'nextCodon_1',
        'nextCodon_2',        
        'next2Codon_0',
        'next2Codon_1',
        'next2Codon_2',
        'next3Codon_0',
        'next3Codon_1',
        'next3Codon_2',
        'partner'
    ])
test_data


# In[46]:


seq = pd.get_dummies(train_data['sequence'], prefix='Base')
struc = pd.get_dummies(train_data['structure'], prefix='Structure')
loop = pd.get_dummies(train_data['predicted_loop_type'], prefix='Loop')
position = pd.get_dummies(train_data['codon_position'], prefix='Position')
base0 = pd.get_dummies(train_data['base_0'], prefix='Base0')
base1 = pd.get_dummies(train_data['base_1'], prefix='Base1')
base2 = pd.get_dummies(train_data['base_2'], prefix='Base2')
codon0 = pd.get_dummies(train_data['prevCodon_0'], prefix='prevCodon0')
codon1 = pd.get_dummies(train_data['prevCodon_1'], prefix='prevCodon1')
codon2 = pd.get_dummies(train_data['prevCodon_2'], prefix='prevCodon2') 
next_codon0 = pd.get_dummies(train_data['nextCodon_0'], prefix='nextCodon0')
next_codon1 = pd.get_dummies(train_data['nextCodon_1'], prefix='nextCodon1')
next_codon2 = pd.get_dummies(train_data['nextCodon_2'], prefix='nextCodon2')
next2_codon0 = pd.get_dummies(train_data['next2Codon_0'], prefix='next2Codon0')
next2_codon1 = pd.get_dummies(train_data['next2Codon_1'], prefix='next2Codon1')
next2_codon2 = pd.get_dummies(train_data['next2Codon_2'], prefix='next2Codon2')
next3_codon0 = pd.get_dummies(train_data['next3Codon_0'], prefix='next3Codon0')
next3_codon1 = pd.get_dummies(train_data['next3Codon_1'], prefix='next3Codon1')
next3_codon2 = pd.get_dummies(train_data['next3Codon_2'], prefix='next3Codon2')
part = pd.get_dummies(train_data['partner'], prefix='partner')

train_set = pd.concat([seq, struc, loop, position, base0, base1, base2, codon0, codon1, codon2, 
                       next_codon0, next_codon1, next_codon2, next2_codon0, next2_codon1, next2_codon2, next3_codon0, next3_codon1, next3_codon2, part, train_data], 
                      axis=1).drop(['sequence', 'structure', 'predicted_loop_type', 'codon_position', 'base_0', 
                                    'base_1', 'base_2', 'prevCodon_0', 'prevCodon_1', 'prevCodon_2', 
                                    'nextCodon_0', 'nextCodon_1', 'nextCodon_2', 'next2Codon_0', 'next2Codon_1', 'next2Codon_2',
                                    'next3Codon_0', 'next3Codon_1', 'next3Codon_2', 'partner'], axis=1)
train_set


# In[47]:


seq = pd.get_dummies(test_data['sequence'], prefix='Base')
struc = pd.get_dummies(test_data['structure'], prefix='Structure')
loop = pd.get_dummies(test_data['predicted_loop_type'], prefix='Loop')
position = pd.get_dummies(test_data['codon_position'], prefix='Position')
base0 = pd.get_dummies(test_data['base_0'], prefix='Base0')
base1 = pd.get_dummies(test_data['base_1'], prefix='Base1')
base2 = pd.get_dummies(test_data['base_2'], prefix='Base2')
codon0 = pd.get_dummies(test_data['prevCodon_0'], prefix='prevCodon0')
codon1 = pd.get_dummies(test_data['prevCodon_1'], prefix='prevCodon1')
codon2 = pd.get_dummies(test_data['prevCodon_2'], prefix='prevCodon2') 
next_codon0 = pd.get_dummies(test_data['nextCodon_0'], prefix='nextCodon0')
next_codon1 = pd.get_dummies(test_data['nextCodon_1'], prefix='nextCodon1')
next_codon2 = pd.get_dummies(test_data['nextCodon_2'], prefix='nextCodon2') 
next2_codon0 = pd.get_dummies(test_data['next2Codon_0'], prefix='next2Codon0')
next2_codon1 = pd.get_dummies(test_data['next2Codon_1'], prefix='next2Codon1')
next2_codon2 = pd.get_dummies(test_data['next2Codon_2'], prefix='next2Codon2')
next3_codon0 = pd.get_dummies(test_data['next3Codon_0'], prefix='next3Codon0')
next3_codon1 = pd.get_dummies(test_data['next3Codon_1'], prefix='next3Codon1')
next3_codon2 = pd.get_dummies(test_data['next3Codon_2'], prefix='next3Codon2')
part = pd.get_dummies(test_data['partner'], prefix='partner')

test_set = pd.concat([seq, struc, loop, position, base0, base1, base2, codon0, codon1, codon2, 
                       next_codon0, next_codon1, next_codon2, next2_codon0, next2_codon1, next2_codon2, next3_codon0, next3_codon1, next3_codon2, part, test_data], 
                      axis=1).drop(['sequence', 'structure', 'predicted_loop_type', 'codon_position', 'base_0', 
                                    'base_1', 'base_2', 'prevCodon_0', 'prevCodon_1', 'prevCodon_2', 
                                    'nextCodon_0', 'nextCodon_1', 'nextCodon_2', 'next2Codon_0', 'next2Codon_1', 'next2Codon_2',
                                    'next3Codon_0', 'next3Codon_1', 'next3Codon_2', 'partner'], axis=1)
test_set


# In[48]:


train_target = train_set[['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']]
train_set = train_set.drop(['id', 'reactivity', 'reactivity_error', 'deg_Mg_pH10', 'deg_error_Mg_pH10', 'deg_pH10', 'deg_error_pH10',
                            'deg_Mg_50C', 'deg_error_Mg_50C', 'deg_50C', 'deg_error_50C'], axis=1)


# In[49]:


test_id = test_set['id']
test_set = test_set.drop(['id'], axis=1)


# In[50]:


test_set


# In[51]:


drop_columns = ['partner_-1', 'prevCodon1_-1', 'prevCodon2_-1', 'isPair', 'pairs_rate']

train_set = train_set.drop(drop_columns, axis=1)
test_set = test_set.drop(drop_columns, axis=1)


# <a id="4"></a>
# <h2 style='background:black; border:0; color:white'><center>4. Keras Neural Network Model</center><h2>

# In[52]:


def MCRMSE(y_true, y_pred):
    colwise_mse = K.mean(K.square(y_true - y_pred))
    return K.mean(K.sqrt(colwise_mse))


# In[53]:


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(101),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(500, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3, activation="elu")
    ])
    model.compile(optimizer='adam', loss=MCRMSE)
    return model


# In[54]:


from sklearn.metrics import mean_squared_error as mse
import math

def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred)) / 3


# In[55]:


train_target.columns


# In[56]:


target = train_target[['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']]


# In[57]:


preds_df = pd.DataFrame()
preds_df['id'] = test_id
preds_df.loc[:, ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']] = 0
res = target.copy()
for n, (tr, te) in enumerate(KFold(n_splits=10, random_state=666, shuffle=True).split(target)):
    print(f'Fold {n}')
    
    model = create_model()
    
    model.fit(
        train_set.values[tr],
        target.values[tr],
        epochs=45, 
        batch_size=64
    )
    
    preds_df.loc[:, ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']] += model.predict(test_set)
    res.loc[te, ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']] = model.predict(train_set.values[te])
    
preds_df.loc[:, ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']] /= (n+1)


# In[58]:


metrics = []
for _target in target.columns:
    metrics.append(rmse(target.loc[:, _target], res.loc[:, _target]))


# #### Let's check our cross validation score

# In[59]:


print(f'OOF Metric: {np.mean(metrics)}')


# <a id="5"></a>
# <h2 style='background:black; border:0; color:white'><center>5. Prepare submission file</center><h2>

# #### So the final step is to prepare submission file

# In[60]:


preds_df


# In[61]:


sub = pd.merge(sub[['id_seqpos']], preds_df, left_on='id_seqpos', right_on='id', how='left').drop(['id'],axis=1)
sub = sub.fillna(0)
sub['deg_pH10'] = 0
sub['deg_50C'] = 0
sub = sub[['id_seqpos', 'reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']]
sub


# In[62]:


sub.to_csv('submission.csv', index=False)


# In[ ]:




