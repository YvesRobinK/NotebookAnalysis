#!/usr/bin/env python
# coding: utf-8

# **If you are interested in this notebook, please check [EDA about Pressure with Colored Charts](https://www.kaggle.com/marutama/eda-about-pressure-with-colored-charts) out as well.It will be the latest and most beautiful update.**
# 
# **Note: I added the explanation of the latter half. Oct 15th**
# 
# This is Part 1 of the series notebook, Pressure main mode EDA. Part 2 is [here](https://www.kaggle.com/marutama/eda-about-pressure-part-2), dealing with other than main mode.
# 
# It's long, so I'll write an overview.
# - I decided the conditions from the chart shape of u_in and puressure with the feeling that I became AI. I hope it will be a hint for Feature engineering.
# - Feature addition has been sped up by avoiding groupby as much as possible.
# - Part 1 is specialized for main mode. After 1 second, the u_in graph has the same shape. It accounts for 92% of the total.
# - If you look at the u_in and pressure graphs in the main mode, you can see a faint layered pattern.
# - Classified by R_C to make the layered pattern clearer.
# - As a layered feature point of Pressure, the Pressure value when u_out becomes 1 is referred to. If you make a histogram, multiple peaks will appear neatly.
# - For a relatively simple pattern, it seems possible to predict the number of layers corresponding to the average value (mean) of u_in. Not all works, but ...
# - I also made the vibration(diff_vib) coefficient of u_in. Take the diff of u_in and count how many times the sign of the diff is inverted. It's quite convenient.
# 
# Continue to [Part 2](https://www.kaggle.com/marutama/eda-about-pressure-part-2).
# 
# In [Part 2](https://www.kaggle.com/marutama/eda-about-pressure-part-2)., you can see that the mode is clearly divided depending on whether the end time of time_step is larger or smaller than 2.65. This is also interesting!
# 
# I think there are many places where the explanation is insufficient. Please comment if you request.
# 
# 
# This notebook is a continuation of:
# - [EDA about time_step and u_out](https://www.kaggle.com/marutama/eda-about-time-step-and-u-out).
# - [EDA about u_in](https://www.kaggle.com/marutama/eda-about-u-in)
# 
# If you find it useful, please upvote it as well.。
# 
# Chart Plot referred to [Ventilator Pressure Prediction: EDA, FE and models](https://www.kaggle.com/artgor/ventilator-pressure-prediction-eda-fe-and-models). 
# 
# For the R_C distribution part, I referred to [Ventilator Pressure simple EDA](https://www.kaggle.com/currypurin/ventilator-pressure-simple-eda).
# 
# 
# - [EDA about: LSTM Feature Importance](https://www.kaggle.com/marutama/eda-about-lstm-feature-importance)
# And [finetune of Tensorflow Bi-LSTM EDA about](https://www.kaggle.com/marutama/finetune-of-tensorflow-bi-lstm-eda-about) is for Modeling.
# 
# Thank you very much.
# 
# The importance of the features introduced in the "EDA about" series below:
# - [EDA about: LSTM Feature Importance](https://www.kaggle.com/marutama/eda-about-lstm-feature-importance)
# 
# And [finetune of Tensorflow Bi-LSTM EDA about](https://www.kaggle.com/marutama/finetune-of-tensorflow-bi-lstm-eda-about) is for Modeling.
# 
# 
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Load CSV

# In[2]:


oj = os.path.join
path = '../input/ventilator-pressure-prediction'
train = pd.read_csv(oj(path, 'train.csv'))
test  = pd.read_csv(oj(path, 'test.csv'))
sub   = pd.read_csv(oj(path, 'sample_submission.csv'))


# # Add features

# Groupby is slow, so I don't use it as much as possible.

# bid_list = list(train['breath_id'].unique())

# In[3]:


get_ipython().run_cell_magic('time', '', "train['R_C'] = [f'{r:02}_{c:02}' for r, c in zip(train['R'], train['C'])]\nRCorder = list(np.sort(train['R_C'].unique()))\n#RCorder\n")


# In[4]:


get_ipython().run_cell_magic('time', '', "# fast area calculation\ntrain['time_delta'] = train['time_step'].diff()\ntrain['time_delta'].fillna(0, inplace=True)\ntrain['time_delta'].mask(train['time_delta'] < 0, 0, inplace=True)\ntrain['tmp'] = train['time_delta'] * train['u_in']\ntrain['area'] = train.groupby('breath_id')['tmp'].cumsum()\n")


# In[5]:


get_ipython().run_cell_magic('time', '', "# u_in: max, min, mean, std \nu_in_max_dict = train.groupby('breath_id')['u_in'].max().to_dict()\ntrain['u_in_max'] = train['breath_id'].map(u_in_max_dict)\nu_in_min_dict = train.groupby('breath_id')['u_in'].min().to_dict()\ntrain['u_in_min'] = train['breath_id'].map(u_in_min_dict)\nu_in_mean_dict = train.groupby('breath_id')['u_in'].mean().to_dict()\ntrain['u_in_mean'] = train['breath_id'].map(u_in_mean_dict)\nu_in_std_dict = train.groupby('breath_id')['u_in'].std().to_dict()\ntrain['u_in_std'] = train['breath_id'].map(u_in_std_dict)\n")


# In[6]:


# u_in_half is time:0 - time point of u_out:1 rise (almost 1.0s)
train['tmp'] = train['u_out']*(-1)+1 # inversion of u_out
train['u_in_half'] = train['tmp'] * train['u_in']


# In[7]:


get_ipython().run_cell_magic('time', '', "# u_in_half: max, min, mean, std\nu_in_half_max_dict = train.groupby('breath_id')['u_in_half'].max().to_dict()\ntrain['u_in_half_max'] = train['breath_id'].map(u_in_half_max_dict)\nu_in_half_min_dict = train.groupby('breath_id')['u_in_half'].min().to_dict()\ntrain['u_in_half_min'] = train['breath_id'].map(u_in_half_min_dict)\nu_in_half_mean_dict = train.groupby('breath_id')['u_in_half'].mean().to_dict()\ntrain['u_in_half_mean'] = train['breath_id'].map(u_in_half_mean_dict)\nu_in_half_std_dict = train.groupby('breath_id')['u_in_half'].std().to_dict()\ntrain['u_in_half_std'] = train['breath_id'].map(u_in_half_std_dict)\n")


# In[8]:


# Groupby is slow, do not use it.
# All entries are first point of each breath_id
first_df = train.loc[0::80,:]
# All entries are first point of each breath_id
last_df = train.loc[79::80,:]


# In[9]:


get_ipython().run_cell_magic('time', '', "# The Main mode DataFrame and flag\nmain_df= last_df[(last_df['u_in']>4.8)&(last_df['u_in']<5.1)]\nmain_mode_dict = dict(zip(main_df['breath_id'], [1]*len(main_df)))\ntrain['main_mode'] = train['breath_id'].map(main_mode_dict)\ntrain['main_mode'].fillna(0, inplace=True)\n")


# In[10]:


get_ipython().run_cell_magic('time', '', "# u_out1_timing flag and DataFrame: speed up\n# 高速版 uout1_df 作成\ntrain['u_out_diff'] = train['u_out'].diff()\ntrain['u_out_diff'].fillna(0, inplace=True)\ntrain['u_out_diff'].replace(-1, 0, inplace=True)\nuout1_df = train[train['u_out_diff']==1]\n")


# In[11]:


main_uout1 = uout1_df[uout1_df['main_mode']==1]
nomain_uout1 = uout1_df[uout1_df['main_mode']==1]


# In[12]:


# Register Area when u_out becomes 1
uout1_area_dict = dict(zip(first_df['breath_id'], first_df['u_in']))
train['area_uout1'] = train['breath_id'].map(uout1_area_dict) 


# In[13]:


get_ipython().run_cell_magic('time', '', "# u_in: first point, last point\nu_in_first_dict = dict(zip(first_df['breath_id'], first_df['u_in']))\ntrain['u_in_first'] = train['breath_id'].map(u_in_first_dict)\nu_in_last_dict = dict(zip(first_df['breath_id'], last_df['u_in']))\ntrain['u_in_last'] = train['breath_id'].map(u_in_last_dict)\n# time(sec) of end point\ntime_end_dict = dict(zip(last_df['breath_id'], last_df['time_step']))     \ntrain['time_end'] = train['breath_id'].map(time_end_dict)\n")


# In[14]:


get_ipython().run_cell_magic('time', '', "# time(sec) when u_out becomes 1\nuout1_dict = dict(zip(uout1_df['breath_id'], uout1_df['time_step']))\ntrain['time_uout1'] = train['breath_id'].map(uout1_dict)\n")


# In[15]:


get_ipython().run_cell_magic('time', '', "# u_in when u_out becomes1\nu_in_uout1_dict = dict(zip(uout1_df['breath_id'], uout1_df['u_in']))\ntrain['u_in_uout1'] = train['breath_id'].map(u_in_uout1_dict)\n")


# In[16]:


get_ipython().run_cell_magic('time', '', "# Dict that puts 0 at the beginning of the 80row cycle\nfirst_0_dict = dict(zip(first_df['id'], [0]*len(uout1_df)))\n\n# Faster version u_in_diff creation, faster than groupby\ntrain['u_in_diff'] = train['u_in'].diff()\ntrain['tmp'] = train['id'].map(first_0_dict) # put 0, the 80row cycle\ntrain.iloc[0::80, train.columns.get_loc('u_in_diff')] = train.iloc[0::80, train.columns.get_loc('tmp')]\n")


# In[17]:


get_ipython().run_cell_magic('time', '', "# Create u_in vibration\ntrain['diff_sign'] = np.sign(train['u_in_diff'])\ntrain['sign_diff'] = train['diff_sign'].diff()\ntrain['tmp'] = train['id'].map(first_0_dict) # put 0, the 80row cycle\ntrain.iloc[0::80, train.columns.get_loc('sign_diff')] = train.iloc[0::80, train.columns.get_loc('tmp')]\n\n# Count the number of inversions, so take the absolute value and sum\ntrain['sign_diff'] = abs(train['sign_diff']) \nsign_diff_dict = train.groupby('breath_id')['sign_diff'].sum().to_dict()\ntrain['diff_vib'] = train['breath_id'].map(sign_diff_dict)\n")


# In[18]:


get_ipython().run_cell_magic('time', '', "if 'diff_sign' in train.columns:\n    train.drop(['diff_sign', 'sign_diff'], axis=1, inplace=True)\n")


# In[19]:


train.head()


# In[20]:


train.columns


# # Recreate each DataFrame when all the features are available

# In[21]:


get_ipython().run_cell_magic('time', '', "################################################################\nfirst_df = train.loc[0::80,:]\nlast_df = train.loc[79::80,:]\nmain_df= last_df[(last_df['u_in']>4.8)&(last_df['u_in']<5.1)]\nnomain_df = last_df[(last_df['u_in']<=4.8)|(last_df['u_in']>=5.1)]\nuout1_df = train[train['u_out_diff']==1]\nmain_uout1 = uout1_df[uout1_df['main_mode']==1]\nnomain_uout1 = uout1_df[uout1_df['main_mode']==1]\n################################################################\n")


# # Functions for plot

# In[22]:


def plot_bid(bid, col1='', col2=''):
    fig, ax1 = plt.subplots(figsize = (6, 4)) 
    
    tmp = train.loc[train['breath_id'] == bid].reset_index(drop=True)
    ax2 = ax1.twinx()

    ax1.plot(tmp['time_step'], tmp['pressure'], 'm-', label='pressure')
    ax1.plot(tmp['time_step'], tmp['u_in'], 'g-', label='u_in')
    ax2.plot(tmp['time_step'], tmp['u_out'], 'b-', label='u_out')

    ax1.set_xlabel('Timestep')
    
    R = tmp['R'][0]
    C = tmp['C'][0]
    mean = tmp['diff_mean'][0]
    std = tmp['diff_std'][0]
    vib = tmp['diff_vib'][0]
    title_str = f'breath_id:{bid}, R:{R}, C:{C}, mean:{mean:.2f}, std:{std:.2f}, vib:{vib:.1f}'
    if col1 != '':
        c1 = tmp[col1][0]
        title_str += f'{col1}: {c1}'
    if col2 != '':
        c2 = tmp[col2][0]
        title_str += f'{col2}: {c2}'
    ax1.set_title(title_str)

    ax1.set_ylim(0, 100)
    
    ax1.legend(loc=(1.1, 0.8))
    ax2.legend(loc=(1.1, 0.7))
    plt.show()

def plot_uin(bid):
    fig, ax1 = plt.subplots(figsize = (6, 4)) 

    tmp = train.loc[train['breath_id'] == bid].reset_index(drop=True)
    #ax2 = ax1.twinx()

    ax1.plot(tmp['time_step'], tmp['u_in'], 'g-', label='u_in')

    ax1.set_xlabel('Timestep')
    
    R = tmp['R'][0]
    C = tmp['C'][0]
    mean = tmp['diff_mean'][0]
    std = tmp['diff_std'][0]
    vib = tmp['diff_vib'][0]
    title_str = f'breath_id:{bid}, R:{R}, C:{C}, mean:{mean:.2f}, std:{std:.2f}, vib:{vib:.1f}'
    ax1.set_title(title_str)

    ax1.set_ylim(0, 100)
    
    plt.show()

def plot_time_step(bid):
    plt.figure()
    tmp = train.loc[train['breath_id'] == bid].reset_index(drop=True)
    R = tmp['R'][0]
    C = tmp['C'][0]
    plt.title(f'breath_id:{bid}, R:{R}, C:{C}')
    plt.ylabel('Timestep')
    plt.xlabel('Row No.')

    plt.plot(train.loc[train['breath_id'] == bid]['time_step'].tolist())
    plt.show()

def plot_uin_list(bid_list, ylim=100, u_low=0, u_high=100, pos=79, alpha=False):
    
    fig, ax1 = plt.subplots(figsize = (6, 4)) # original (12, 8)

    if alpha:
        a = alpha
    else:
        if (len(bid_list)):
            a = max(1.0/len(bid_list), 0.01)
        else:
            a = 1
    
    for bid in tqdm(bid_list):
        tmp = train.loc[train['breath_id'] == bid].reset_index(drop=True)
        u = tmp['u_in'][pos]
        if (u >= u_low) and (u <= u_high):  
            ax1.plot(tmp['time_step'], tmp['u_in'], 'g-', alpha=a)

    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('u_in')
    ax1.set_ylim(0,ylim)
    #ax1.legend(loc=(1.1, 0.8))
    plt.show()

def df_from_to(df, f=0.0, t=6.0):
    # 「0.0のみ」と「0.0より大きく1以下」を実現したいので、この不等号の形
    bid_list = df.loc[(df['u_in'] > f)&(df['u_in'] <= t)]['breath_id'].tolist()
    return bid_list   


# In[23]:


def plot_double_time_bid(bid, time_delta=False, col1='', col2=''):
    fig = plt.figure(figsize = (12, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    
    tmp = train.loc[train['breath_id'] == bid].reset_index(drop=True)

    ts = []
    td = []
    if time_delta:
        outlier = tmp.loc[tmp['time_delta'] > 0.15]
        
        rw = outlier['id'].tolist()
        ts = outlier['time_step'].tolist()
        td = outlier['time_delta'].tolist()
        
    
    R = tmp['R'][0]
    C = tmp['C'][0]
    title_str = f'breath_id:{bid}, R:{R}, C:{C}'
    if col1 != '':
        c1 = tmp[col1][0]
        title_str += f'{col1}: {c1}'
    if col2 != '':
        c2 = tmp[col2][0]
        title_str += f'{col2}: {c2}'
    ax1.set_title(title_str)
    
    ax1.set_ylabel('Timestep')
    ax1.set_xlabel('Row No.')

    ymax = 3.0
    ax1.set_ylim(0, ymax)

    if time_delta:
        rows = []
        for a in rw:
            aa = a % 80 - 2
            if aa < 0:
                aa += 80
            rows.append(aa)
            aa = a % 80 - 1
            if aa < 0:
                aa += 80
            rows.append(aa)
        ax1.vlines(rows, 0, ymax, "red", linestyles='dashed', alpha=0.2)

    
    ax1.plot(train.loc[train['breath_id'] == bid]['time_step'].tolist())

    ##############################
    ax3 = ax2.twinx()

    ax2.plot(tmp['time_step'], tmp['pressure'], 'm-', label='pressure')
    ax2.plot(tmp['time_step'], tmp['u_in'], 'g-', label='u_in')
    ax3.plot(tmp['time_step'], tmp['u_out'], 'b-', label='u_out')

    ax2.set_xlabel('Timestep')
    
    R = tmp['R'][0]
    C = tmp['C'][0]
    mean = tmp['diff_mean'][0]
    std = tmp['diff_std'][0]
    vib = tmp['diff_vib'][0]
    title_str = f'breath_id:{bid}, R:{R}, C:{C}, mean:{mean:.2f}, std:{std:.2f}, vib:{vib:.1f}'
    ax2.set_title(title_str)

    ymax = 100
    ax2.set_ylim(0, ymax)
    
    if time_delta:
        lines = []
        for a, b in zip(ts, td):
            lines.append(a-b)
            lines.append(a)
        ax2.vlines(lines, 0, ymax, "red", linestyles='dashed', alpha=0.2)
    
    ax2.legend(loc=(1.1, 0.8))
    ax3.legend(loc=(1.1, 0.7))
    
    fig.tight_layout()
    plt.show()


# In[24]:


def plot_pre_list(bid_list, ylim=100, low=0, high=100, pos=79, alpha=False):
    
    fig, ax1 = plt.subplots(figsize = (6, 4)) # original (12, 8)

    if alpha:
        a = alpha
    else:
        if (len(bid_list)):
            a = max(1.0/len(bid_list), 0.01)
        else:
            a = 1
    
    for bid in tqdm(bid_list):
        tmp = train.loc[train['breath_id'] == bid].reset_index(drop=True)
        u = tmp['pressure'][pos]
        if (u >= low) and (u <= high):  
            ax1.plot(tmp['time_step'], tmp['pressure'], 'm-', alpha=a)

    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Pressure')
    ax1.set_ylim(0,ylim)
    #ax1.legend(loc=(1.1, 0.8))
    plt.show()


# In[25]:


def plot_double_pre_list(bid_list, max_plots=False, ylim=100, alpha=False):
    fig = plt.figure(figsize = (12, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
       
    title_str = f'time - u_in'
    ax1.set_title(title_str)
    
    ax1.set_ylabel('u_in')
    ax1.set_xlabel('Timestep')

    ax1.set_ylim(0, ylim)
   
    ##############################
    ax2.set_ylabel('Pressure')
    ax2.set_xlabel('Timestep')
    
    title_str = f'time - pressure'
    ax2.set_title(title_str)

    ax2.set_ylim(0, ylim)

    ##############################
    if alpha:
        a = alpha
    else:
        if (len(bid_list)):
            a = max(1.0/len(bid_list), 0.01)
        else:
            a = 1
    if not max_plots:
        max_plots = len(bid_list)
        
    for bid in tqdm(bid_list[:max_plots]):
        tmp = train.loc[train['breath_id'] == bid].reset_index(drop=True)
        ax1.plot(tmp['time_step'], tmp['u_in'], 'g-', label='u_in', alpha=a)
        ax2.plot(tmp['time_step'], tmp['pressure'], 'm-', label='pressure', alpha=a)
    
    fig.tight_layout()
    plt.show()


# In[26]:


def plot_double_area_bid(bid):
    fig = plt.figure(figsize = (12, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    
    tmp = train.loc[train['breath_id'] == bid].reset_index(drop=True)

    R = tmp['R'][0]
    C = tmp['C'][0]
    title_str = f'Area'
    ax1.set_title(title_str)
    
    ax1.set_ylabel('Area')
    ax1.set_xlabel('Timestep')

    ymax = 100
    ax1.set_ylim(0, ymax)

    ax1.plot(tmp['time_step'], tmp['area'],  'r-', label='area')
    #ax1.plot(tmp['time_step'], tmp['area2'], 'g-', label='area2')

    ##############################
    ax3 = ax2.twinx()

    ax2.plot(tmp['time_step'], tmp['pressure'], 'm-', label='pressure')
    ax2.plot(tmp['time_step'], tmp['u_in'], 'g-', label='u_in')
    ax3.plot(tmp['time_step'], tmp['u_out'], 'b-', label='u_out')

    ax2.set_xlabel('Timestep')
    
    R = tmp['R'][0]
    C = tmp['C'][0]
    mean = tmp['diff_mean'][0]
    std = tmp['diff_std'][0]
    vib = tmp['diff_vib'][0]
    title_str = f'breath_id:{bid}, R:{R}, C:{C}, mean:{mean:.2f}, std:{std:.2f}, vib:{vib:.1f}'
    ax2.set_title(title_str)

    ymax = 100
    ax2.set_ylim(0, ymax)
    
    ax2.legend(loc=(1.1, 0.8))
    ax3.legend(loc=(1.1, 0.7))
    
    fig.tight_layout()
    plt.show()


# In[27]:


def plot_bid_stats(list, indiv=3, df=last_df, max_plots=False, no_uin=False,
                   alpha=False, time_delta=False, col1='', col2=''):
    tmpdf= df[df['breath_id'].isin(list)]
    bid_list = tmpdf['breath_id']
    print('Number of plots:', len(bid_list))
    
    if indiv:
        for bid in bid_list[:indiv]: # 最大3個、個別表示
            if time_delta:
                plot_double_bid(bid, time_delta=True, col1=col1, col2=col2)
            else:
                plot_bid(bid, col1=col1, col2=col2)

    #plt.hist(tmpdf['R_C'], bins=17) # 棒グラフの順番指定できないので見にくい
    sns.countplot(x="R_C", data=tmpdf, order=RCorder)
    plt.show()
    
    if not no_uin:
        if not max_plots:
            max_plots = len(bid_list)
            print(f'Number of plots: {max_plots}')
        else:
            print(f'Number of plots: {max_plots}/{len(bid_list)}')
        plot_uin_list(bid_list[:max_plots], alpha=alpha)


# # The Main mode and others

# According to [notebook of mine](https://www.kaggle.com/marutama/eda-about-u-in), u_in has a main mode that accounts for 70018/75450=92%. It is 0 for 1 to 1.5 seconds and ends near 5 (4.965-4.995).

# In[28]:


print('The main mode:')
plot_bid_stats(main_df['breath_id'], indiv=0, no_uin=True)
plot_double_pre_list(main_df['breath_id'][:1000])
print('Other than the main mode:')
plot_bid_stats(nomain_df['breath_id'], indiv=0, no_uin=True)
plot_double_pre_list(nomain_df['breath_id'][:1000])


# In "other than main mode", there are only R = 50 and C = 10.

# # Distribution of diff_vib: "u_in" diff vibration (Number of sign inversions)¶

# diff_vib indicates the number of vibrations.

# In[29]:


plt.hist(last_df['diff_vib'], bins=100)
plt.title("diff_vib global distribution")
plt.show()


# In[30]:


# The main mode and ohters
fig = plt.figure(figsize = (12, 4))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.hist(main_df['diff_vib'], bins=100)
ax1.set_title('The main mode')

ax2.hist(nomain_df['diff_vib'], bins=100)
ax2.set_title('Other than the main mode')

plt.show()


# The pattern is very different between The main mode and others.

# # EDA of the main mode

# There are several modes, but it seems that they can be roughly classified by u_in_mean.

# ## Overview

# In[31]:


df = main_df
print('Number of the main mode:', len(df) )
plot_double_pre_list(df['breath_id'], max_plots=1000)


# In the "pressure" graph, you can see a faint layer.

# In[32]:


print('Total:', len(uout1_df))
print('Main mode:', len(main_uout1))


# In[33]:


def between(list):
    between = []
    for i, a in enumerate(list):
        if i == 0:
            prev = a
            continue
        b = (prev + a)/2
        between.append(b)
        prev = a
    
    return between


# In[34]:


pattern = {}


# From here on, we will look at each R_C separately. They have the perspectives of'Overview',' Peak point of pressure at u_out=1', and 'Classification by u_in_mean', respectively.

# # R_C: 05_10, 20_10, 50_10

# The peak distributions of the '05_10', '20_10' and '50_10' pressure histograms are similar.

# ## Overview

# In[35]:


df=main_uout1
for a in ['05_10', '20_10', '50_10']:
    print(a)
    df2=df[df['R_C']==a]
    plot_double_pre_list(df2['breath_id'], max_plots=1000)


# It seems that 2 to 3 modes are mixed. Let's disassemble it.

# ## Peak point of pressure at u_out=1

# In[36]:


df=main_uout1
ymax=600
xmax=60
for a in ['05_10', '20_10', '50_10']:
    print(a)
    plt.figure(figsize=(12,4))
    plt.title(f'R_C: {a}')
    plt.ylim(0,ymax)
    plt.xlim(0,xmax)
    plt.hist(df[df['R_C']==a]['pressure'], bins=100)
    pattern[a] = [10, 15, 20, 25, 30, 35]
    plt.vlines(pattern[a], 0, ymax, "red", linestyles='dashed')
    plt.vlines(between(pattern[a]), 0, ymax, "green", linestyles='dashed', alpha=0.5)
    plt.show()


# In[37]:


def plot_uin_pre_hist(RC_list, df, double=True, pre=True, mean=True):
    ymax=600
    xmax=60
    for a in RC_list:
        print('R_C:', a)
        df2=df[df['R_C']==a]
        print('Number:', len(df2))
        
        if double:
            plot_double_pre_list(df2['breath_id'], max_plots=1000)    

        if pre:
            plt.figure(figsize=(12,4))
            plt.title(f'pressure: R_C: {a}')
            plt.ylim(0,ymax)
            plt.xlim(0,xmax)
            plt.hist(df2['pressure'], bins=100)
            plt.vlines(pattern[a], 0, ymax, "red", linestyles='dashed')
            plt.vlines(between(pattern[a]), 0, ymax, "green", linestyles='dashed', alpha=0.5)
            plt.show()

        if mean:
            plt.figure(figsize=(12,4))
            plt.title(f'u_in_mean: R_C: {a}')
            plt.hist(df2['u_in_mean'], bins=100)
            plt.show()


# ## Classification by u_in_mean 05_10, 20_10

# Let's take a look at the histogram of u_in_mead.

# In[38]:


df=main_uout1
plot_uin_pre_hist(['05_10', '20_10'], df, double=False, pre=False)


# It seems that it will not be possible to classify as it is.

# The peak distributions of the '05_10' and '20_10' u_in_mean histograms are similar. Then, in order to classify the modes more clearly, we divide them into the following three conditions.
# 
# - u_in_max >= 30
# - u_in_max < 30 & u_in_first > 0
# - u_in_max < 30 & u_in_first == 0

# ### u_in_max >= 30

# In[39]:


df=main_uout1
df2=df[(df['u_in_max']>=30)] ### condition
plot_uin_pre_hist(['05_10', '20_10'], df2)


# ### u_in_max < 30 & u_in_first > 0

# In[40]:


df=main_uout1
df2=df[(df['u_in_max']<30)&(df['u_in_first']>0)] ### condition
plot_uin_pre_hist(['05_10', '20_10'], df2)


# ### u_in_max <30 & u_in_first == 0

# In[41]:


df=main_uout1
df2=df[(df['u_in_max']<30)&(df['u_in_first']==0)] ### condition
plot_uin_pre_hist(['05_10', '20_10'], df2)


# u_in_mean makes it easier to classify.

# ## Classification by u_in_mean 50_10 

# Think of 50_10 in a different story. Now let's take a look at diff_vib.

# In[42]:


plt.figure(figsize=(12, 4))
plt.hist(df[df['R_C']==a]['diff_vib'], bins=100)
plt.vlines([10], 0, 2000, "red", linestyles='dashed')
plt.title('Main mode, R_C=50_10: histgram of diff_vib')
plt.show()


# Since it seems that you can classify with diff_vib, consider the following three conditions. Another u_in_first: I'm also paying attention to the first u_in.
# 
# - u_in_first < 1 & diff_vib < 10
# - u_in_first < 1 & diff_vib >= 10
# - u_in_first >= 1

# ### u_in_first < 1 & diff_vib < 10

# In[43]:


df=main_uout1
df2=df[(df['u_in_first']<1)&(df['diff_vib']<10)] ### condition
plot_uin_pre_hist(['50_10'], df2)


# ### u_in_first < 1 & diff_vib >= 10

# In[44]:


df=main_uout1
df2=df[(df['u_in_first']<1)&(df['diff_vib']>=10)] ### condition
plot_uin_pre_hist(['50_10'], df2)


# ### u_in_first >= 1

# In[45]:


df=main_uout1
df2=df[(df['u_in_first']>=1)] ### condition
plot_uin_pre_hist(['50_10'], df2)


# It seems that peak has come out that seems to be divided by u_in_mean.

# # R_C: 20_20

# R_C: 20_20 seems to be a different mode.

# ## Overview

# In[46]:


df=main_uout1
df2=df[df['R_C']=='20_20']
plot_double_pre_list(df2['breath_id'], max_plots=1000)


# ## Peak point of pressure at u_out=1

# In[47]:


df=main_uout1
ymax=600
xmax=60
for a in ['20_20']:
    print(a)
    plt.figure(figsize=(12,4))
    plt.title(f'R_C: {a}')
    plt.ylim(0,ymax)
    plt.xlim(0,xmax)
    plt.hist(df[df['R_C']==a]['pressure'], bins=100)
    pattern[a] = [10, 15.5, 21, 26, 31, 36]
    plt.vlines(pattern[a], 0, ymax, "red", linestyles='dashed')
    plt.vlines(between(pattern[a]), 0, ymax, "green", linestyles='dashed', alpha=0.5)
    plt.show()


# ## Classification by u_in_mean 20_20

# Let's take a look at the histogram of u_in_mean.

# In[48]:


df=main_uout1
plot_uin_pre_hist(['20_20'], df, double=False, pre=False)


# It seems better to classify it a little more. It seems better to classify it a little more. Let's take a look at diff_vib.

# In[49]:


plt.figure(figsize=(12, 4))
plt.hist(df[df['R_C']=='20_20']['diff_vib'], bins=100)
plt.vlines([10], 0, 2000, "red", linestyles='dashed')
plt.show()


# diff_vib: It seems to be divided at 10.

# ### diff_vib > 10

# In[50]:


df=main_uout1
df2 = df[(df['diff_vib']>10)]
plot_uin_pre_hist(['20_20'], df2)


# ### diff_vib <= 10

# In[51]:


df=main_uout1
df2 = df[df['diff_vib']<=10]
plot_uin_pre_hist(['20_20'], df2)


# In particular, 20_20 is easier to classify with u_in_mead.

# # R_C: 05_20, 20_50

# ## Overview

# 05_20 and 20_50 are in the form of peak in pressure histgram.

# In[52]:


df=main_uout1
for a in ['05_20', '20_50']:
    print(a)
    df2=df[df['R_C']==a]
    plot_double_pre_list(df2['breath_id'], max_plots=1000)


# ## Peak point of pressure at u_out=1

# In[53]:


df=main_uout1
ymax=600
xmax=60
for a in ['05_20', '20_50']:
    print(a)
    plt.figure(figsize=(12,4))
    plt.title(f'R_C: {a}')
    plt.ylim(0,ymax)
    plt.xlim(0,xmax)
    plt.hist(df[df['R_C']==a]['pressure'], bins=100)
    pattern[a] = [10, 15.5, 20.5, 25.5, 30, 34] 
    plt.vlines(pattern[a], 0, ymax, "red", linestyles='dashed')
    plt.vlines(between(pattern[a]), 0, ymax, "green", linestyles='dashed', alpha=0.5)
    plt.show()


# ## Classification by u_in_mean 05_20, 20_50

# Let's take a look at the histogram of u_in_mean.

# In[54]:


df=main_uout1
plot_uin_pre_hist(['05_20', '20_50'], df, double=False, pre=False)


# 20_50 seems to be divided by u_in_mean as it is, but 05_20 seems to be a little difficult. Let's consider 05_20 under the following conditions.
# - u_in_first > 30
# - u_in_first > 0 & u_in_first <= 30
# - u_in_first ==0

# ### 05_20: u_in_first > 30

# In[55]:


df=main_uout1
df2 = df[(df['u_in_first']>30)]
plot_uin_pre_hist(['05_20'], df2)


# ### 05_20: u_in_first > 0 & u_in_first <= 30

# In[56]:


df=main_uout1
df2 = df[(df['u_in_first']>0)&(df['u_in_first']<=30)]
plot_uin_pre_hist(['05_20'], df2)


# ### 05_20: u_in_first ==0

# In[57]:


df=main_uout1
df2 = df[(df['u_in_first']==0)]
plot_uin_pre_hist(['05_20'], df2)


# By classifying by the value of u_in_first, it became easier to classify by u_in_mead. Still, u_in_first == 0 seems difficult.

# # R_C: 05_50

# 05_50 is considered alone.

# ## Overview

# In[58]:


df=main_uout1
for a in ['05_50']:
    print(a)
    df2=df[df['R_C']==a]
    plot_double_pre_list(df2['breath_id'], max_plots=1000)


# ## Peak point of pressure at u_out=1

# In[59]:


df=main_uout1
ymax=600
xmax=60
for a in ['05_50']:
    print(a)
    plt.figure(figsize=(12,4))
    plt.title(f'R_C: {a}')
    plt.ylim(0,ymax)
    plt.xlim(0,xmax)
    plt.hist(df[df['R_C']==a]['pressure'], bins=100)
    pattern[a] = [10, 15, 19, 23.5, 27, 31.5]
    plt.vlines(pattern[a], 0, ymax, "red", linestyles='dashed')
    plt.vlines(between(pattern[a]), 0, ymax, "green", linestyles='dashed', alpha=0.5)
    plt.show()


# ## Classification by u_in_mean 05_50

# In[60]:


df=main_uout1
plot_uin_pre_hist(['05_50'], df, double=False, pre=False)


# 05_50 seems to be able to be classified by u_in_mead as it is.

# # R_C: 50_20, 50_50

# 50_20, 50_50 are very difficult cases.

# ## Overview

# In[61]:


df=main_uout1
for a in ['50_20', '50_50']:
    print(a)
    df2=df[df['R_C']==a]
    plot_double_pre_list(df2['breath_id'], max_plots=1000)


# ## Peak point of pressure at u_out=1

# In[62]:


df=main_uout1
ymax=600
xmax=60
for a in ['50_20', '50_50']:
    plt.figure(figsize=(12,4))
    plt.title(f'R_C: {a}')
    plt.ylim(0,ymax)
    plt.xlim(0,xmax)
    plt.hist(df[df['R_C']==a]['pressure'], bins=100)
    pattern[a]=[]
    plt.vlines(pattern[a], 0, ymax, "red", linestyles='dashed')
    plt.vlines(between(pattern[a]), 0, ymax, "green", linestyles='dashed', alpha=0.5)
    plt.show()


# The pressure in uout1 is terrible, and there seems to be no classification.

# ## Classification by u_in_mean 50_20, 50_50

# In[63]:


df=main_uout1
plot_uin_pre_hist(['50_20', '50_50'], df, double=False, pre=False)


# It seems to be difficult at this rate. Let's take a look at diff_vib.

# In[64]:


plt.figure(figsize=(12, 4))
plt.hist(df[df['R_C']=='50_20']['diff_vib'], bins=100)
plt.vlines([25], 0, 2000, "red", linestyles='dashed')
plt.title('50_20: histgram of diff_vib')
plt.show()


# In[65]:


plt.figure(figsize=(12, 4))
plt.hist(df[df['R_C']=='50_50']['diff_vib'], bins=100)
plt.vlines([25], 0, 2000, "red", linestyles='dashed')
plt.title('50_50: histgram of diff_vib')
plt.show()


# It seems that diff_vib can be divided into 25 or less and above ...

# ### diff_vib < 25

# In[66]:


df=main_uout1
df2 = df[(df['diff_vib']<25)]
plot_uin_pre_hist(['50_20', '50_50'], df2)


# ### dif_vib >= 25

# In[67]:


df=main_uout1
df2 = df[(df['diff_vib']>=25)]
plot_uin_pre_hist(['50_20', '50_50'], df2)


# 50_20 and 50_50 seem difficult.

# In[ ]:




