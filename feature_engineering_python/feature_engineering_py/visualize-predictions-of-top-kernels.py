#!/usr/bin/env python
# coding: utf-8

# # Visualize predictions of popular kernels

# - K1: https://www.kaggle.com/mrkmakr/covid-ae-pretrain-gnn-attn-cnn/
# - K2: https://www.kaggle.com/its7171/gru-lstm-with-feature-engineering-and-augmentation
# - K3: https://www.kaggle.com/symyksr/openvaccine-deepergcn
# - K4: https://www.kaggle.com/thedrcat/openvaccine-lstm-fastai

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


n1 = '../input/covid-ae-pretrain-gnn-attn-cnn/submission.csv'
n2 = '../input/gru-lstm-with-feature-engineering-and-augmentation/submission.csv'
n3 = '../input/openvaccine-deepergcn/submission.csv'
n4 = '../input/openvaccine-lstm-fastai/submission.csv'

s1 = pd.read_csv(n1)
s2 = pd.read_csv(n2)
s3 = pd.read_csv(n3)
s4 = pd.read_csv(n4)

s1['id'], s1['seqpos'] = s1['id_seqpos'].str.rsplit('_', 1).str
s1['seqpos'] = s1['seqpos'].astype(int)
s1 = s1.sort_values(by=['id', 'seqpos']).reset_index(drop=True)

s2['id'], s2['seqpos'] = s2['id_seqpos'].str.rsplit('_', 1).str
s2['seqpos'] = s2['seqpos'].astype(int)
s2 = s2.sort_values(by=['id', 'seqpos']).reset_index(drop=True)

s3['id'], s3['seqpos'] = s3['id_seqpos'].str.rsplit('_', 1).str
s3['seqpos'] = s3['seqpos'].astype(int)
s3 = s3.sort_values(by=['id', 'seqpos']).reset_index(drop=True)

s4['id'], s4['seqpos'] = s4['id_seqpos'].str.rsplit('_', 1).str
s4['seqpos'] = s4['seqpos'].astype(int)
s4 = s4.sort_values(by=['id', 'seqpos']).reset_index(drop=True)


# In[3]:


r1a = s1.groupby('id')['reactivity'].apply(list)[0]
r1b = s1.groupby('id')['reactivity'].apply(list)[1000]
r1c = s1.groupby('id')['reactivity'].apply(list)[2000]
r1d = s1.groupby('id')['reactivity'].apply(list)[3000]

r2a = s2.groupby('id')['reactivity'].apply(list)[0]
r2b = s2.groupby('id')['reactivity'].apply(list)[1000]
r2c = s2.groupby('id')['reactivity'].apply(list)[2000]
r2d = s2.groupby('id')['reactivity'].apply(list)[3000]

r3a = s3.groupby('id')['reactivity'].apply(list)[0]
r3b = s3.groupby('id')['reactivity'].apply(list)[1000]
r3c = s3.groupby('id')['reactivity'].apply(list)[2000]
r3d = s3.groupby('id')['reactivity'].apply(list)[3000]

r4a = s4.groupby('id')['reactivity'].apply(list)[0]
r4b = s4.groupby('id')['reactivity'].apply(list)[1000]
r4c = s4.groupby('id')['reactivity'].apply(list)[2000]
r4d = s4.groupby('id')['reactivity'].apply(list)[3000]


# In[4]:


fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(21,15), sharex=True, sharey=True)
fig.suptitle('Reactivity', fontsize=24, color='blue')
viz = [r1a, r1b, r1c, r1d, r2a, r2b, r2c,r2d, r3a, r3b,r3c,r3d,r4a,r4b,r4c,r4d]
tits = ['k1a', 'k1b', 'k1c', 'k1d', 'k2a', 'k2b', 'k2c','k2d', 'k3a', 'k3b','k3c','k3d','k4a','k4b','k4c','k4d']
for i, ax in enumerate(axes.flatten()):
    ax.plot(viz[i])
    ax.set_title(tits[i], color='blue', fontsize=20)
    ax.axvline(x=68, color='red')
    ax.axvline(x=91, color='blue')
    ax.axvline(x=107, color='green')
plt.show()


# In[5]:


r1a = s1.groupby('id')['deg_Mg_pH10'].apply(list)[0]
r1b = s1.groupby('id')['deg_Mg_pH10'].apply(list)[1000]
r1c = s1.groupby('id')['deg_Mg_pH10'].apply(list)[2000]
r1d = s1.groupby('id')['deg_Mg_pH10'].apply(list)[3000]

r2a = s2.groupby('id')['deg_Mg_pH10'].apply(list)[0]
r2b = s2.groupby('id')['deg_Mg_pH10'].apply(list)[1000]
r2c = s2.groupby('id')['deg_Mg_pH10'].apply(list)[2000]
r2d = s2.groupby('id')['deg_Mg_pH10'].apply(list)[3000]

r3a = s3.groupby('id')['deg_Mg_pH10'].apply(list)[0]
r3b = s3.groupby('id')['deg_Mg_pH10'].apply(list)[1000]
r3c = s3.groupby('id')['deg_Mg_pH10'].apply(list)[2000]
r3d = s3.groupby('id')['deg_Mg_pH10'].apply(list)[3000]

r4a = s4.groupby('id')['deg_Mg_pH10'].apply(list)[0]
r4b = s4.groupby('id')['deg_Mg_pH10'].apply(list)[1000]
r4c = s4.groupby('id')['deg_Mg_pH10'].apply(list)[2000]
r4d = s4.groupby('id')['deg_Mg_pH10'].apply(list)[3000]

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(21,15), sharex=True, sharey=True)
fig.suptitle('deg_Mg_pH10', fontsize=24, color='blue')
viz = [r1a, r1b, r1c, r1d, r2a, r2b, r2c,r2d, r3a, r3b,r3c,r3d,r4a,r4b,r4c,r4d]
tits = ['k1a', 'k1b', 'k1c', 'k1d', 'k2a', 'k2b', 'k2c','k2d', 'k3a', 'k3b','k3c','k3d','k4a','k4b','k4c','k4d']
for i, ax in enumerate(axes.flatten()):
    ax.plot(viz[i])
    ax.set_title(tits[i], color='blue', fontsize=20)
    ax.axvline(x=68, color='red')
    ax.axvline(x=91, color='blue')
    ax.axvline(x=107, color='green')
plt.show()


# In[6]:


r1a = s1.groupby('id')['deg_Mg_50C'].apply(list)[0]
r1b = s1.groupby('id')['deg_Mg_50C'].apply(list)[1000]
r1c = s1.groupby('id')['deg_Mg_50C'].apply(list)[2000]
r1d = s1.groupby('id')['deg_Mg_50C'].apply(list)[3000]

r2a = s2.groupby('id')['deg_Mg_50C'].apply(list)[0]
r2b = s2.groupby('id')['deg_Mg_50C'].apply(list)[1000]
r2c = s2.groupby('id')['deg_Mg_50C'].apply(list)[2000]
r2d = s2.groupby('id')['deg_Mg_50C'].apply(list)[3000]

r3a = s3.groupby('id')['deg_Mg_50C'].apply(list)[0]
r3b = s3.groupby('id')['deg_Mg_50C'].apply(list)[1000]
r3c = s3.groupby('id')['deg_Mg_50C'].apply(list)[2000]
r3d = s3.groupby('id')['deg_Mg_50C'].apply(list)[3000]

r4a = s4.groupby('id')['deg_Mg_50C'].apply(list)[0]
r4b = s4.groupby('id')['deg_Mg_50C'].apply(list)[1000]
r4c = s4.groupby('id')['deg_Mg_50C'].apply(list)[2000]
r4d = s4.groupby('id')['deg_Mg_50C'].apply(list)[3000]

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(21,15), sharex=True, sharey=True)
fig.suptitle('deg_Mg_50C', fontsize=24, color='blue')
viz = [r1a, r1b, r1c, r1d, r2a, r2b, r2c,r2d, r3a, r3b,r3c,r3d,r4a,r4b,r4c,r4d]
tits = ['k1a', 'k1b', 'k1c', 'k1d', 'k2a', 'k2b', 'k2c','k2d', 'k3a', 'k3b','k3c','k3d','k4a','k4b','k4c','k4d']
for i, ax in enumerate(axes.flatten()):
    ax.plot(viz[i])
    ax.set_title(tits[i], color='blue', fontsize=20)
    ax.axvline(x=68, color='red')
    ax.axvline(x=91, color='blue')
    ax.axvline(x=107, color='green')
plt.show()


# In[ ]:




