#!/usr/bin/env python
# coding: utf-8

# # This EDA specializes in u_in

# **Note: I made a mistake in calculating the area of u_in and time_delta. I was multiplying by time_step (which means real time in this competition) instead of time_delta. Fixed. October 3rd.**
# 
# This notebook is a continuation of [EDA about time_step and u_out](https://www.kaggle.com/marutama/eda-about-time-step-and-u-out). If you find it useful, please upvote it as well.。
# 
# Chart Plot referred to [Ventilator Pressure Prediction: EDA, FE and models](https://www.kaggle.com/artgor/ventilator-pressure-prediction-eda-fe-and-models). 
# 
# For the R_C distribution part, I referred to [Ventilator Pressure simple EDA](https://www.kaggle.com/currypurin/ventilator-pressure-simple-eda).
# 
# Thank you very much.
# 
# The importance of the features introduced in the "EDA about" series below:
# - [EDA about: LSTM Feature Importance](https://www.kaggle.com/marutama/eda-about-lstm-feature-importance)
# 
# And [finetune of Tensorflow Bi-LSTM EDA about](https://www.kaggle.com/marutama/finetune-of-tensorflow-bi-lstm-eda-about) is for Modeling.
# 
# 

# # TL;DR
# - About half of u_in starts at 0. The next most common is u_in, which starts from 100.
# - u_in has a main mode that accounts for 70018/75450=92%. It is 0 for 1 to 1.5 seconds and ends near 5 (4.965-4.995).
# - All R_Cs other than the main mode are 50_10.
# - The pressure correlates with Area (the sum of the products of u_in and time_step up to that point) and R_C.

# # Import

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


# # Functions

# In[2]:


def plot_bid(bid):
    fig, ax1 = plt.subplots(figsize = (6, 4)) 
    
    tmp = train.loc[train['breath_id'] == bid].reset_index(drop=True)
    ax2 = ax1.twinx()

    ax1.plot(tmp['time_step'], tmp['pressure'], 'r-', label='pressure')
    ax1.plot(tmp['time_step'], tmp['u_in'], 'g-', label='u_in')
    ax2.plot(tmp['time_step'], tmp['u_out'], 'b-', label='u_out')

    ax1.set_xlabel('Timestep')
    
    R = tmp['R'][0]
    C = tmp['C'][0]
    ax1.set_title(f'breath_id:{bid}, R:{R}, C:{C}')

    ax1.set_ylim(0, 100)
    
    ax1.legend(loc=(1.1, 0.8))
    ax2.legend(loc=(1.1, 0.7))
    plt.show()


# In[3]:


def plot_uin(bid):
    fig, ax1 = plt.subplots(figsize = (6, 4)) 

    tmp = train.loc[train['breath_id'] == bid].reset_index(drop=True)
    #ax2 = ax1.twinx()

    ax1.plot(tmp['time_step'], tmp['u_in'], 'g-', label='u_in')

    ax1.set_xlabel('Timestep')
    
    R = tmp['R'][0]
    C = tmp['C'][0]
    ax1.set_title(f'breath_id:{bid}, R:{R}, C:{C}')

    ax1.set_ylim(0, 100)
    
    plt.show()


# In[4]:


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


# In[5]:


'''
def plot_double_bid(bid):
    fig = plt.figure(figsize = (12, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    tmp = train.loc[train['breath_id'] == bid].reset_index(drop=True)

    R = tmp['R'][0]
    C = tmp['C'][0]
    ax1.set_title(f'breath_id:{bid}, R:{R}, C:{C}')
    ax1.set_ylabel('Timestep')
    ax1.set_xlabel('Row No.')

    ax1.plot(train.loc[train['breath_id'] == bid]['time_step'].tolist())

    ##############################
    ax3 = ax2.twinx()

    ax2.plot(tmp['time_step'], tmp['pressure'], 'r-', label='pressure')
    ax2.plot(tmp['time_step'], tmp['u_in'], 'g-', label='u_in')
    ax3.plot(tmp['time_step'], tmp['u_out'], 'b-', label='u_out')

    ax2.set_xlabel('Timestep')
    
    R = tmp['R'][0]
    C = tmp['C'][0]
    ax2.set_title(f'breath_id:{bid}, R:{R}, C:{C}')

    ax2.set_ylim(0, 100)
    
    ax2.legend(loc=(1.1, 0.8))
    ax3.legend(loc=(1.1, 0.7))
    
    fig.tight_layout()
    plt.show()
'''


# # Load CSV

# In[6]:


oj = os.path.join
path = '../input/ventilator-pressure-prediction'
train = pd.read_csv(oj(path, 'train.csv'))
test  = pd.read_csv(oj(path, 'test.csv'))
sub   = pd.read_csv(oj(path, 'sample_submission.csv'))


# # Add features

# In[7]:


get_ipython().run_cell_magic('time', '', "train['time_delta'] = train.groupby('breath_id')['time_step'].diff()\n")


# In[8]:


bid_list = list(train['breath_id'].unique())


# In[9]:


def plot_uin_list(bid_list, ylim=100, u_low=0, u_high=100, pos=79, alpha=False):
    
    fig, ax1 = plt.subplots(figsize = (6, 4)) # original (12, 8)

    if alpha:
        a = alpha
    else:
        a = max(1.0/len(bid_list), 0.01)
    
    for bid in tqdm(bid_list):
        tmp = train.loc[train['breath_id'] == bid].reset_index(drop=True)
        u = tmp['u_in'][pos]
        if (u >= u_low) and (u <= u_high):  
            ax1.plot(tmp['time_step'], tmp['u_in'], 'g-', alpha=a)

    ax1.set_xlabel('Timestep')
    ax1.set_ylim(0,ylim)
    #ax1.legend(loc=(1.1, 0.8))
    plt.show()


# In[10]:


get_ipython().run_cell_magic('time', '', "#train['R_C'] = [f'{r:02}_{c:02}' for r, c in zip(train['R'], train['C'])]\ntrain['R_C'] = [f'{r:02}_{c}' for r, c in zip(train['R'], train['C'])]\n#RCorder = ['05_10', '05_20', '05_50', '20_10', '20_20', '20_50', '50_10', '50_20', '50_50']\nRCorder =  sorted(train['R_C'].unique())\nRCorder\n")


# Plot the u_in chart for the first 3

# # Overview

# In[11]:


for bid in bid_list[:3]:
    plot_uin(bid)


# Combination distribution of R and C

# In[12]:


sns.countplot(x="R_C", data=train, order=RCorder)


# I will stack the first 1000 pieces and plot

# In[13]:


plot_uin_list(bid_list[:1000])


# Let's expand the y-axis further.

# In[14]:


plot_uin_list(bid_list[:1000], ylim=7)


# It seems that there are many graphs where 1.0 to 1.5 is 0, and it starts from 1.5 and gradually becomes about 5 at the end. Let's call this graph the main mode of u_in.

# # u_in End point value EDA

# In[15]:


last_df = train.loc[79::80,:]
last_df


# In[16]:


plt.hist(last_df['u_in'], bins=20)
plt.show()


# Approximately 70,000 u_ins end in 5 hits. Let's increase bins and expand.

# In[17]:


ymin=0
ymax=500
plt.hist(last_df['u_in'], bins=100)
plt.vlines([0.0, 0.75, 1.75, 2.1, 4.8, 5.1], ymin, ymax, "red", linestyles='dashed')
plt.ylim(ymin,ymax)
plt.show()


# Let's divide it into this zone. Around 5.0 is the final value of u_in main mode.

# In[18]:


ymin=0
ymax=5
plt.hist(last_df['u_in'], bins=100)
plt.vlines([0.0, 0.75, 1.75, 2.1, 3, 4.8, 5.1], ymin, ymax, "red", linestyles='dashed')
plt.ylim(ymin, ymax)
plt.xlim(4,)
plt.show()


# If you expand it, there is only one data for 5 or more.

# In[19]:


close_up_u_in = [u for u in last_df['u_in'] if u > 4.6]
plt.hist(close_up_u_in, bins=1000)
plt.xlim(4.96, 5.0)
plt.show()


# Further close-up around 5.0, there are three peaks in the final u_in value in main mode, which is between 4.965-4.995 in detail.

# In[20]:


def plot_double_bid(bid, time_delta=False):
    fig = plt.figure(figsize = (12, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    
    tmp = train.loc[train['breath_id'] == bid].reset_index(drop=True)

    ts = []
    td = []
    if time_delta:
        outlier = tmp.loc[tmp['time_delta'] > 0.15]
        rw = list(outlier['id'])
        ts = list(outlier['time_step'])
        td = list(outlier['time_delta'])
        
    
    R = tmp['R'][0]
    C = tmp['C'][0]
    ax1.set_title(f'breath_id:{bid}, R:{R}, C:{C}')
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
    ax2.set_title(f'breath_id:{bid}, R:{R}, C:{C}')

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


# In[21]:


def df_from_to(df, f=0.0, t=6.0):
    # 「0.0のみ」と「0.0より大きく1以下」を実現したいので、この不等号の形
    bid_list = df.loc[(df['u_in'] > f)&(df['u_in'] <= t)]['breath_id'].tolist()
    return bid_list   


# In[22]:


def plot_uin_stats(list, indiv=3, df=last_df, alpha=False, time_step=False):
    tmpdf= df[df['breath_id'].isin(list)]
    bid_list = tmpdf['breath_id']
    print('Number of plots:', len(bid_list))
    
    if indiv:
        for bid in bid_list[:indiv]: # 最大3個、個別表示
            if time_step:
                plot_double_bid(bid, time_delta=True)
            else:
                plot_bid(bid)
            
    #plt.hist(tmpdf['R_C'], bins=17) # 棒グラフの順番指定できないので見にくい
    sns.countplot(x="R_C", data=tmpdf, order=RCorder)
    plt.show()
    
    plot_uin_list(bid_list, alpha=alpha)


# # Other than the Main mode: u_in ends 5.0 - 5.4

# In[23]:


plot_uin_stats(df_from_to(last_df, 5, 5.4))


# Only one. 'R_C' is '50_10'.

# # The Main mode : u_in ends 4.8 - 5.0

# In[24]:


plot_uin_stats(df_from_to(last_df, 4.8, 5.0))


# 4.8-5.1 is the main mode and there are many. From 1.0 second to 1.5 seconds, u_in = 0.0, and from 1.5 seconds, it starts up with the same shape and the final value is 4.965-4.995.
# The R_C distribution clearly has a decrease of '50_10' compared to the overall distribution. It seems that the ratio of '50_10' is high except for 4.8-5.1.

# In[25]:


plot_uin_list(df_from_to(last_df[:1000], 4.8, 5.0))


# If you plot only 1000 pieces, you can see the graph shape between 0 seconds and 1 second like a pattern. 
# 
# So far, only in the Main mode, there are some 'time_steps' that are severely broken.

# In[26]:


no_prop_list = list(train.loc[train['time_delta']>0.15]['breath_id'].unique())
print(len(no_prop_list))


# In[27]:


plot_uin_stats(no_prop_list, indiv=10, time_step=True)


# # Other than the Main mode: u_in ends 2.1 - 4.8

# In[28]:


plot_uin_stats(df_from_to(last_df, 2.1, 4.8))


# 'R_C' is only 50_10. The u_in graph looks like a vibrating shape in much the same way.

# # Other than the Main mode: u_in ends 1.75 - 2.1

# In[29]:


plot_uin_stats(df_from_to(last_df, 1.75, 2.1))


# 'R_C' is only 50_10. The u_in graph seems to have a vibration shape and a hilly shape.

# # Other than the Main mode: u_in ends 0.75 - 1.75

# In[30]:


plot_uin_stats(df_from_to(last_df, 0.75, 1.75))


# 'R_C' is only 50_10. The u_in graph seems to have more hill shapes than vibration shapes.

# # Other than the Main mode: 0.0 - 0.75

# In[31]:


plot_uin_stats(df_from_to(last_df, 0.0, 0.75))


# 'R_C' is only 50_10. Is the vibration shape of the u_in graph increasing again?

# # Other than the Main mode: 0.0 only

# In[32]:


plot_uin_stats(df_from_to(last_df, -1, 0.0))# ０のみ


# 'R_C' is only 50_10. The u_in graph also looks like it has returned to its vibrating shape.

# # u_in start point value EDA

# In[33]:


first_df = train.loc[0::80,:]
first_df


# In[34]:


plt.hist(first_df['u_in'], bins=100)
plt.show()


# In[35]:


plt.hist(first_df['u_in'], bins=100)
plt.ylim(0, 3000)
plt.show()


# In[36]:


plt.hist(first_df['u_in'], bins=100)
plt.ylim(0, 200)
plt.show()


# # EDA with u_in starting point value of 100

# In[37]:


first100_list = first_df[first_df['u_in']==100]['breath_id'].tolist()
plot_uin_stats(first100_list)


# Most seem to be the main mode, but there seem to be other modes as well. Let's separate.

# # The Main mode

# In[38]:


# u_in last : Main mode
lastM_list = last_df[(last_df['u_in']>4.8)&(last_df['u_in']<5.1)]['breath_id'].tolist()
print(len(lastM_list))


# In[39]:


get_ipython().run_cell_magic('time', '', 'plot_uin_stats(lastM_list, indiv=False)\n')


# There are 70018 main modes. It's 70018/75450=92%. Use a set operation to separate the other modes.

# # Main mode with 100 at the beginning

# In[40]:


first100_set = set(first100_list)
lastM_set = set(lastM_list)


# In[41]:


first100_lastM_set = first100_set & lastM_set
len(first100_lastM_set)
plot_uin_stats(first100_lastM_set, indiv=5)


# The main mode, which starts with 100, seems to have a vibrating graph, a slide-like descending graph, and a combination graph. The R_C distribution is also biased.

# # The starting point value is 100 and other than main mode

# In[42]:


plot_uin_stats(first100_set - first100_lastM_set)


# There are only 23 graphs other than the main mode, where u_in starts with 100, and they have the same graph shape. R_C is also only 50_10.

# # The starting point value is 0 u_in EDA

# In[43]:


first0_list = first_df[first_df['u_in']==0]['breath_id'].tolist()
plot_uin_stats(first0_list)


# This also seems to have a main mode and other modes. Let's separate.

# In[44]:


first0_set = set(first0_list)
first0_lastM_set = first0_set & lastM_set


# In[45]:


plot_uin_stats(first0_lastM_set)


# In[46]:


plot_uin_stats(first0_set - lastM_set, alpha=0.1)


# Graphs with u_in starting at 0, excluding the main mode, are unique and have a distinctly different shape from other graphs. R_C is only 50_10.

# # Let's calculate the area from u_in

# In[47]:


#train = pd.read_csv(oj(path, 'train.csv'))
#train['R_C'] = [f'{r}_{c}' for r, c in zip(train['R'], train['C'])]
#RCorder = ['5_10', '5_20', '5_50', '20_10', '20_20', '20_50', '50_10', '50_20', '50_50']
#train['R_C'].unique()


# In[48]:


#%%time
#train['time_delta'] = train.groupby('breath_id')['time_step'].diff()
#train.fillna(0, inplace=True)
#train['delta'] = train['time_delta'] * train['u_in']
#train['area'] = train.groupby('breath_id')['delta'].cumsum()


# speed up for area calculation

# In[49]:


get_ipython().run_cell_magic('time', '', "train['time_delta'] = train['time_step'].diff()\ntrain['time_delta'].fillna(0, inplace=True)\ntrain['time_delta'].mask(train['time_delta'] < 0, 0, inplace=True)\ntrain['delta'] = train['time_delta'] * train['u_in']\ntrain['area'] = train.groupby('breath_id')['delta'].cumsum()\n")


# In[50]:


train


# # Check the area of the point where u_out rises

# In[51]:


# u_out1_timing
# generate empty df
#df = pd.DataFrame(columns=['id', 'breath_id', 'R', 'C', 'time_step', 'u_in', 'u_out', 'pressure',
#                           'R_C', 'time_delta', 'delta', 'area'])
#for i in tqdm(bid_list):
#    breath_one = train[train['breath_id']==i].reset_index(drop = True)
#    tmp_df=breath_one[breath_one['u_out']==1].head(1)
#    df = df.append(tmp_df)

#uout1_df = df
#uout1_df


# speed up version

# In[52]:


get_ipython().run_cell_magic('time', '', "# u_out1_timing : speed up\n# 高速版 uout1_df 作成\ntrain['u_out_diff'] = train['u_out'].diff()\ntrain['u_out_diff'].fillna(0, inplace=True)\ntrain['u_out_diff'].replace(-1, 0, inplace=True)\nuout1_df = train[train['u_out_diff']==1]\n")


# In[53]:


fig, ax1 = plt.subplots(figsize = (6, 4)) 

tmp = uout1_df
for rc in RCorder:
    t = tmp[tmp['R_C']==rc]
    if len(t) == 0:
        continue
    ax1.hist(t['pressure'], bins=100, label=rc, alpha=0.5)

ax1.legend()
plt.show()


# In[54]:


uout1_list =  uout1_df['breath_id'].tolist()
uout1_set = set(uout1_list)
uout1_notM = uout1_set - lastM_set
uout1_Main = uout1_set - uout1_notM


# In[55]:


print(len(uout1_notM))
print(len(uout1_Main))


# In[56]:


uout1_notM_df = uout1_df[uout1_df['breath_id'].isin(uout1_notM)]
uout1_Main_df = uout1_df[uout1_df['breath_id'].isin(uout1_Main)]


# In[57]:


fig, ax1 = plt.subplots(figsize = (6, 4)) 

tmp = uout1_notM_df

for rc in RCorder:
    t = tmp[tmp['R_C']==rc]
    if len(t) == 0:
        continue
    ax1.scatter(t['area'], t['pressure'], label=rc, alpha=0.1)

ax1.set_ylabel('pressure')
ax1.set_xlabel('area')

ax1.set_title('Other than Main mode')

ax1.legend(loc=(1.1, 0.8))
plt.show()


# In[58]:


get_ipython().run_cell_magic('time', '', "fig = plt.figure(figsize = (12, 4))\nax1 = fig.add_subplot(1, 2, 1)\nax2 = fig.add_subplot(1, 2, 2)\n\ntmp = uout1_Main_df\n\nfor rc in RCorder:\n    t = tmp[tmp['R_C']==rc]\n    if len(t) == 0:\n        continue\n    ax1.scatter(t['area'], t['pressure'], label=rc, alpha=1)\nax1.set_ylabel('pressure')\nax1.set_xlabel('area')\nax1.set_title('Main mode: alpha=1')\nax1.legend()\n\nfor rc in RCorder:\n    t = tmp[tmp['R_C']==rc]\n    if len(t) == 0:\n        continue\n    ax2.scatter(t['area'], t['pressure'], label=rc, alpha=0.01)\nax2.set_ylabel('pressure')\nax2.set_xlabel('area')\nax2.set_title('Main mode: alpha=0.01')\nax2.legend()\n\nplt.show()\n")


# The pressure correlates with Area (the sum of the products of u_in and time_step up to that point) and R_C.

# In[ ]:




