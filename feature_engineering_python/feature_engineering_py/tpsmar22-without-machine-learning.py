#!/usr/bin/env python
# coding: utf-8

# # Solving the March TPS Without Machine Learning
# 
# We don't need machine learning to predict congestions for September 30. It suffices to determine the median congestion for every place and time of day and submit these medians.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
from cycler import cycler
from IPython import display

oldcycler = plt.rcParams['axes.prop_cycle']
plt.rcParams['axes.facecolor'] = '#0057b8' # blue
plt.rcParams['axes.prop_cycle'] = cycler(color=['#ffd700'] +
                                         oldcycler.by_key()['color'][1:])


# In[2]:


# Read the data
train = pd.read_csv('../input/tabular-playground-series-mar-2022/train.csv', index_col='row_id', parse_dates=['time'])
test = pd.read_csv('../input/tabular-playground-series-mar-2022/test.csv', index_col='row_id', parse_dates=['time'])


# In[3]:


# Feature Engineering
for df in [train, test]:
    df['weekday'] = df.time.dt.weekday
    df['hour'] = df.time.dt.hour
    df['minute'] = df.time.dt.minute
    


# In[4]:


# Compute the median congestion for every place and time of week
medians = train.groupby(['x', 'y', 'direction', 'weekday', 'hour', 'minute']).congestion.median().astype(int)
medians


# In[5]:


# Write the submission file
sub = test.merge(medians, 
                 left_on=['x', 'y', 'direction', 'weekday', 'hour', 'minute'],
                 right_index=True)[['congestion']]
sub.reset_index(inplace=True)
sub.to_csv('submission_no_machine_learning.csv', index=False)
sub


# In[6]:


# Plot the distribution of the test predictions
# compared to the other Monday afternoons
plt.figure(figsize=(16,3))
plt.hist(train.congestion[((train.time.dt.weekday == 0) &
                           (train.time.dt.hour >= 12)).values],
         bins=np.linspace(-0.5, 100.5, 102),
         density=True, label='Train',
         color='#ffd700')
plt.hist(sub['congestion'], np.linspace(-0.5, 100.5, 102),
         density=True, rwidth=0.5, label='Test predictions',
         color='r')
plt.xlabel('Congestion')
plt.ylabel('Frequency')
plt.title('Congestion on Monday afternoons')
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
plt.legend()
plt.show()


# In[ ]:




