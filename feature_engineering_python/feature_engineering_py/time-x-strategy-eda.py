#!/usr/bin/env python
# coding: utf-8

# # Time and investment strategy
# 
# This notebooks aims to explore target across time id and investment ids.

# ## Other Feature Exploration / Feature engineering for Ubiquant:
# 
# - [Complete Feature Exploration](https://www.kaggle.com/lucasmorin/complete-feature-exploration)
# - [Weird pattern in unique values](https://www.kaggle.com/lucasmorin/weird-patterns-in-unique-values-across-time-ids/)
# - [Time x Strategy EDA](https://www.kaggle.com/lucasmorin/time-x-strategy-eda)  
# - [UMAP Data Analysis & Applications](https://www.kaggle.com/lucasmorin/umap-data-analysis-applications)   
# - [LB probing Notebook  ](https://www.kaggle.com/lucasmorin/don-t-mind-me-just-probing-the-lb)
# - On-Line Feature Engineering (in progress)

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
import matplotlib as mpl
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


# Using @slawekbiel Feather dataset: https://www.kaggle.com/slawekbiel/ubiquant-trainfeather-32-bit

# In[2]:


get_ipython().run_cell_magic('time', '', "\ntraining_path = '../input/ubiquant-market-prediction/train.csv'\n\ndtypes = {\n    'row_id': 'str',\n    'time_id': 'uint16',\n    'investment_id': 'uint16',\n    'target': 'float32',\n}\n\ntrain_data = pd.read_csv(training_path,usecols=list(dtypes.keys()),dtype=dtypes)\n")


# In[3]:


len(train_data.target.unique())


# number of unique values per time ids:

# In[4]:


pivoted_data = train_data[['time_id','investment_id','target']].pivot(index='investment_id',columns='time_id',values='target')


# # Target

# In[5]:


plt.figure(figsize = (200,20))
plt.imshow(np.nan_to_num(pivoted_data),vmin=-0.5,vmax=.5)


# # Average target by time

# In[6]:


plt.plot(pivoted_data.mean(axis=0))


# # Average target by investment_id

# In[7]:


plt.plot(pivoted_data.mean(axis=1))


# # Volatility

# In[8]:


plt.figure(figsize = (200,20))
plt.imshow(np.square(np.nan_to_num(pivoted_data)),vmin=0,vmax=2)


# # Missing target values

# In[9]:


plt.figure(figsize = (200,20))
plt.imshow(np.isnan(pivoted_data))


# # Average volatility over time

# In[10]:


plt.plot(np.mean(np.square(np.nan_to_num(pivoted_data)),axis=0))


# # Observing time

# In[11]:


emb = PCA(n_components=2).fit_transform(np.transpose(np.nan_to_num(pivoted_data)))

plt.figure(figsize=(10, 8))
plt.scatter(emb[:, 0], emb[:, 1], s=10, c=np.mean(pivoted_data,axis=0), edgecolors='none', cmap='viridis',vmin=-.25,vmax=0.25);
cb = plt.colorbar(label='average target', format=mpl.ticker.ScalarFormatter())
cb.ax.yaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())

plt.title('PCA strategy embeddings');


# # Observing strategies

# In[12]:


emb = PCA(n_components=2).fit_transform(np.nan_to_num(pivoted_data))

plt.figure(figsize=(10, 8))
plt.scatter(emb[:, 0], emb[:, 1], s=10, c=np.mean(pivoted_data,axis=1), edgecolors='none', cmap='viridis',vmin=-.25,vmax=0.25);
cb = plt.colorbar(label='average target', format=mpl.ticker.ScalarFormatter())

cb.ax.yaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
plt.title('PCA strategy embeddings');

