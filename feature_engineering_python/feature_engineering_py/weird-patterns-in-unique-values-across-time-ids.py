#!/usr/bin/env python
# coding: utf-8

# # Weird pattern in unique values across time ids
# 
# There seems to be some weird pattern in unique values across time ids. This notebook aims to explore that.

# ## Other Feature Exploration / Feature engineering for Ubiquant:
# 
# - [Complete Feature Exploration](https://www.kaggle.com/lucasmorin/complete-feature-exploration)
# - [Weird pattern in unique values](https://www.kaggle.com/lucasmorin/weird-patterns-in-unique-values-across-time-ids/)
# - [Time x Strategy EDA](https://www.kaggle.com/lucasmorin/time-x-strategy-eda)  
# - [UMAP Data Analysis & Applications](https://www.kaggle.com/lucasmorin/umap-data-analysis-applications)   
# - [LB probing Notebook  ](https://www.kaggle.com/lucasmorin/don-t-mind-me-just-probing-the-lb)
# - On-Line Feature Engineering (in progress)

# ## Weird patterns :
# 
# - [Counting Values](#Counting_Values)
# - [All Patterns](#All_Patterns)
# - [Unitary Values](#Unitary_Values) (🔥🔥🔥)

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
warnings.filterwarnings("ignore")

DEBUG = False


# Using @slawekbiel Feather dataset: https://www.kaggle.com/slawekbiel/ubiquant-trainfeather-32-bit

# In[2]:


get_ipython().run_cell_magic('time', '', "train_data = pd.read_feather('../input/ubiquant-trainfeather-32-bit/train32.feather')\n")


# number of unique values per time ids:

# In[3]:


train_data.head()


# <a id='Counting_Values'></a>
# # Counting values

# In[4]:


mean_problem = []
n = 5 if DEBUG else 300

features_of_interest = {}
count = train_data[['time_id','investment_id']].groupby(['time_id']).count().values.flatten()

for i in range(n):
    f_name = 'f_'+str(i)
    unique = train_data[['time_id',f_name]].groupby(['time_id']).nunique().values.flatten()
    id_pattern = (np.log(count)-np.log(unique)>1)
    mean = id_pattern.mean()
    mean_problem.append(mean)
    features_of_interest[f_name] = unique[id_pattern].mean()

print(f'proportion of features with a problem - above 1%: {np.mean([m>0.01 for m in mean_problem])}')
print(f'average proportion of feature values impacted: {np.mean([m for m in mean_problem if m>0.01])}')
    


# In[5]:


mode_max = 1000000

cat_filter = {k: v for k, v in features_of_interest.items() if v<mode_max}
small_cat_dict = sorted(cat_filter.items(), key=lambda x: x[1])
small_cat_dict


# <a id='All_Patterns'></a>
# # All Patterns

# In[6]:


for i in range(300):
    feature_name = 'f_'+str(i)
    print('f_'+str(i))
    plt.plot(np.log(train_data[['time_id','investment_id']].groupby(['time_id']).count()))
    plt.plot(np.log(train_data[['time_id',feature_name]].groupby(['time_id']).nunique()))
    plt.show()


# <a id='Unitary_Values'></a>
# # Unitary values (Market Features)

# In[7]:


# small number of values:

smalls = ['f_170','f_272','f_182','f_124','f_200','f_175']

count = train_data[['time_id','investment_id']].groupby(['time_id']).count().values.flatten()


for f_name in smalls :
    #f_name = cat[0]
    print(f_name)
    unique = train_data[['time_id',f_name]].groupby(['time_id']).nunique().values.flatten()
    id_pattern = (np.log(count)-np.log(unique)>1)
    mean = id_pattern.mean()
    #plt.plot()
    fig, axs = plt.subplots(2, 3)
    fig.set_size_inches(30, 12)
    
    f_mean = train_data[['time_id',f_name]].groupby('time_id').agg(np.mean)[f_name]
    train_data[f_name+'c'] = train_data.time_id.map(round(f_mean))
    train_data[f_name+'n'] = train_data[f_name] - train_data[f_name+'c']
    
    axs[0, 0].plot(np.log(train_data[['time_id',f_name+'n']].groupby(['time_id']).nunique()))
    axs[0, 0].plot(np.log(train_data[['time_id','investment_id']].groupby(['time_id']).count()))
    axs[0, 0].set_title('Pattern')
    
    axs[1, 0].plot(train_data[['time_id',f_name+'c']].groupby('time_id').agg(np.mean)[f_name+'c'])
    axs[1, 0].set_title('retrieved Categorical')
    
    axs[0, 1].plot(train_data[['time_id',f_name+'n']].groupby('time_id').agg(np.mean)[f_name+'n'])
    axs[0, 1].set_title('Average Noise')
    
    axs[1, 1].plot(train_data[['time_id',f_name+'n']].groupby('time_id').agg(np.std)[f_name+'n'])
    axs[1, 1].set_title('Noise std')
    
    axs[0, 2].plot(train_data[['time_id',f_name+'n']].groupby('time_id').agg(np.min)[f_name+'n'])
    axs[0, 2].plot(train_data[['time_id',f_name+'n']].groupby('time_id').agg(np.max)[f_name+'n'])
    axs[0, 2].set_title('Noise min/max')
    
    axs[1, 2].plot(train_data[['time_id',f_name]].groupby('time_id').agg(np.mean)[f_name].cumsum())
    axs[1, 2].plot(train_data[['time_id',f_name+'c']].groupby('time_id').agg(np.mean)[f_name+'c'].cumsum())
    axs[1, 2].set_title('Cumsum of market data')
    
    plt.show()


# In[8]:


smalls = ['f_124']

for f_name in smalls :
    #f_name = cat[0]
    print(f_name)
    unique = train_data[['time_id',f_name]].groupby(['time_id']).nunique().values.flatten()
    id_pattern = (np.log(count)-np.log(unique)>1)
    mean = id_pattern.mean()
    #plt.plot()
    fig, axs = plt.subplots(2, 3)
    fig.set_size_inches(30, 12)
    
    f_mean = train_data[['time_id',f_name]].groupby('time_id').agg(np.mean)[f_name]
    train_data[f_name+'c'] = train_data.time_id.map(round(f_mean))
    train_data[f_name+'n'] = train_data[f_name] - train_data[f_name+'c']
    
    axs[0, 0].plot(np.log(train_data[['time_id',f_name+'n']].groupby(['time_id']).nunique()))
    axs[0, 0].plot(np.log(train_data[['time_id','investment_id']].groupby(['time_id']).count()))
    axs[1, 0].plot(train_data[['time_id',f_name+'c']].groupby('time_id').agg(np.mean)[f_name+'c'])
    axs[0, 1].plot(train_data[['time_id',f_name+'n']].groupby('time_id').agg(np.mean)[f_name+'n'])
    axs[1, 1].plot(train_data[['time_id',f_name+'n']].groupby('time_id').agg(np.std)[f_name+'n'])
    axs[0, 2].plot(train_data[['time_id',f_name+'n']].groupby('time_id').agg(np.min)[f_name+'n'])
    axs[1, 2].plot(train_data[['time_id',f_name+'n']].groupby('time_id').agg(np.max)[f_name+'n'])
    plt.show()


# In[9]:


plt.plot(np.clip(train_data[['time_id',f_name+'n']].groupby('time_id').agg(np.mean)[f_name+'n'],-0.00001,0.00001))

