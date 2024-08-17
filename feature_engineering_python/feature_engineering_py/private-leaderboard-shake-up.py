#!/usr/bin/env python
# coding: utf-8

# ## Summary
# In this notebook, we visualize and analyze the difference of some best scored public kernels' stats on private set.

# In[1]:


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error as mae


# In[2]:


DIR_NAME = {'DeeperGCN':'../input/openvaccine-deepergcn/',
           'RNN':'../input/gru-lstm-with-feature-engineering-and-augmentation/',
           'AE_TF':'../input/covid-ae-pretrain-gnn-attn-cnn/',
           'AE_PT':'../input/openvaccine-pytorch-ae-pretrain/'}


# ## Getting private's id's of sequence + position

# In[3]:


test  = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json",lines=True)
test_pri = test[test["seq_length"] == 130]
test_pri.head()


# In[4]:


sub = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")
id_pri = []
for i, uid in enumerate(test_pri['id']):
    id_seqpos = [f'{uid}_{x}' for x in range(130)]
    id_pri += id_seqpos
id_pri = pd.DataFrame(id_pri, columns=['id_seqpos'])


# In[5]:


sub.loc[sub['id_seqpos'].isin(id_pri['id_seqpos'])].head()


# ## Read a bunch of high scored public kernels

# In[6]:


DIR_NAME[key[1]]


# In[7]:


subs = [] 
target_cols = ['reactivity', 'deg_Mg_50C','deg_Mg_pH10']

for key in enumerate(DIR_NAME.keys()):
    df = pd.read_csv(DIR_NAME[key[1]]+'submission.csv',
                   index_col=False, 
                   usecols=['id_seqpos']+target_cols)
    df.sort_values('id_seqpos',inplace=True, ascending=True)
    df['model'] = key[1]
    print(set(sub.id_seqpos) == set(df.id_seqpos))
    subs.append(df)

    


# ## Computing MAE for different subs
# The original computation computation is wrong, the submissions have different `seqpos_id`. Thanks to Marcello Susanto pointed out.

# In[8]:


for t in target_cols:
    print(f'\nMean abs difference in {t}:\n')
    for i in range(len(DIR_NAME)):
        for j in range(i+1, len(DIR_NAME)):
            df_i, df_j = subs[i], subs[j]
            abs_diff= mae(subs[i][t], subs[j][t])
            print(f'submission {i} and {j}: {abs_diff:.5f}')


# ### This difference is quite huge considering the gold and nothing is less than 0.03 as of right now on public

# ## Randomly choosing 200 predictions for pairplot between different positions after 68-th base

# In[9]:


N_VIS = 200
RANGE_VIS = [[68, 75], [76, 83], [84, 91]]


# In[10]:


def plot_pairplot(positions):
    id_pri = []
    for i, uid in enumerate(test_pri['id']):
        id_seqpos = [f'{uid}_{x}' for x in range(positions[0], positions[1])]
        id_pri += id_seqpos
    id_pri = pd.DataFrame(id_pri, columns=['id_seqpos'])

    subs_pri = []
    for i in range(len(DIR_NAME)): 
        df_tmp = subs[i]
        subs_pri.append(df_tmp.loc[df_tmp['id_seqpos'].isin(id_pri['id_seqpos'])])
        
    idx = np.random.randint(0,len(subs_pri[0]), N_VIS)
    subs_vis = []
    df_vis = pd.DataFrame()
    for i in range(len(DIR_NAME)):
        df_vis = subs_pri[i].iloc[idx].copy()
        df_vis.loc[:,target_cols] = df_vis[target_cols].values
        subs_vis.append(df_vis)
    df_vis = pd.concat(subs_vis)
    
    sns.set_style(style="ticks")
    sns.pairplot(df_vis, hue="model");
    
def plot_lineplot(positions):
    id_pri = []
    for i, uid in enumerate(test_pri['id']):
        id_seqpos = [f'{uid}_{x}' for x in range(positions[0], positions[1])]
        id_pri += id_seqpos
    id_pri = pd.DataFrame(id_pri, columns=['id_seqpos'])

    subs_pri = []
    for i in range(len(DIR_NAME)): 
        df_tmp = subs[i]
        subs_pri.append(df_tmp.loc[df_tmp['id_seqpos'].isin(id_pri['id_seqpos'])])
        
    idx = np.random.randint(0,len(subs_pri[0]), N_VIS)
    subs_vis = []
    df_vis = pd.DataFrame()
    for i in range(len(DIR_NAME)):
        df_vis = subs_pri[i].iloc[idx].copy()
        df_vis.loc[:,target_cols] = df_vis[target_cols].values
        subs_vis.append(df_vis)
    df_vis = pd.concat(subs_vis)

    fig, axes = plt.subplots(3,1,figsize=(10, 15))
    
    for i, col in enumerate(target_cols):
        g = sns.lineplot(data=df_vis, x="id_seqpos", y=col, hue='model', ax=axes[i])
        g.set(xticklabels=[]) 


# In[11]:


plot_pairplot(RANGE_VIS[0])


# In[12]:


plot_pairplot(RANGE_VIS[1])


# In[13]:


plot_pairplot(RANGE_VIS[2])


# In[14]:


plot_lineplot(RANGE_VIS[0])


# In[15]:


plot_lineplot(RANGE_VIS[1])


# In[16]:


plot_lineplot(RANGE_VIS[2])


# ## Randomly choose an mRNA in private set

# In[17]:


def plot_rna_preds(seq_ids=None, n_sample=1, positions=(68,91)):
    if seq_ids is None:
        ids = test_pri['id'].sample(n=n_sample)
    else:
        ids= seq_ids
    id_pri = []
    for i, uid in enumerate(ids):
        id_seqpos = [f'{uid}_{x}' for x in range(positions[0],positions[1])]
        id_pri += id_seqpos
    id_pri = pd.DataFrame(id_pri, columns=['id_seqpos'])
    subs_pri = []
    for i in range(len(DIR_NAME)): 
        df_tmp = subs[i]
        subs_pri.append(df_tmp.loc[df_tmp['id_seqpos'].isin(id_pri['id_seqpos'])])
        
    subs_vis = []
    df_vis = pd.DataFrame()
    for i in range(len(DIR_NAME)):
        df_tmp = subs_pri[i]
        df_vis = df_tmp.loc[df_tmp['id_seqpos'].isin(id_pri['id_seqpos'])].copy()
        df_vis.loc[:,target_cols] = df_vis[target_cols].values
        subs_vis.append(df_vis)
    df_vis = pd.concat(subs_vis)
    
    fig, axes = plt.subplots(3*n_sample,1,figsize=(10, 15*n_sample))
    for j, id_seq in enumerate(ids):
        for i, col in enumerate(target_cols):
            g = sns.lineplot(data=df_vis, x="id_seqpos", y=col, hue='model', ax=axes[i+j*3])
            g.set(xticklabels=[]) 
            g.set_title(f'{id_seq}')


# In[18]:


plot_rna_preds(seq_ids=['id_9085aafc1'])


# In[19]:


plot_rna_preds(seq_ids=['id_4cc792927'])


# In[20]:


plot_rna_preds(seq_ids=['id_2dc15cef2'])


# In[ ]:




