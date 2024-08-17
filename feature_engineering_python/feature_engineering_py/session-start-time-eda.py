#!/usr/bin/env python
# coding: utf-8

# # Session start time EDA🕰️

# ### This EDA has been made possible thanks to [this](https://www.kaggle.com/code/pdnartreb/session-id-reverse-engineering) notebook from Bertrand P, the current N°1 in this competition (03/01/2023) 
# He found that the session_id wasn't a sequence of randomly selected number but rather contained information about the session starting time and date, here is a **picture breaking down that notebook**: ![Session Id.drawio.png](attachment:35d78dc2-655c-41e4-b301-dde4d1b494d6.png)
# 
# The last two digits are either noise or a feature we currently have no consensus about, if any idea comes to you mind and it happens to verify, i'll gladly add it to this notebook (with credits)

# ## 🛑If you appreciate the notebook, don't hesitate to **give me feedback** ! :)

# #### This is what we learn from the [Jo Wilder's official website](https://pbswisconsineducation.org/jowilder/about/):

# ![image.png](attachment:a6fdf2d2-2bfd-4511-8e7e-ee846d3b4510.png)

# #### For those like me who don't live in the land of the free: 
# 
# **In the U.S, a typical day of high school starts at about 7:30 a.m. and ends around 3:00 p.m.,from Monday to Friday**

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import seaborn as sns
sns.set_theme(style="ticks", palette="pastel")
pd.set_option('display.max_column', 200)


dtypes={'session_id':np.int64, 
'elapsed_time':np.int32,
    'event_name':'category',
    'name':'category',
    'level':np.uint8,
    'page':'category',
    'room_coor_x':np.float32,
    'room_coor_y':np.float32,
    'screen_coor_x':np.float32,
    'screen_coor_y':np.float32,
    'hover_duration':np.float32,
     'text':'category',
     'fqid':'category',
     'room_fqid':'category',
     'text_fqid':'category',
     'fullscreen':'category',
     'hq':'category',
     'music':'category',
     'level_group':'category'}


# In[2]:


df_sessions =  pd.read_csv("/kaggle/input/predict-student-performance-from-game-play/train.csv", dtype = dtypes)
drop = ['num_events','lots_events','few_events','has_finished','last_elapsed_time','long_session','short_session','Unnamed: 0']
df_accuracy = pd.read_csv('/kaggle/input/student-perf-eda-feature-engineering/df_sessions.csv').drop(columns = drop)


# In[3]:


def feature_eng(df_sessions):
    df_final = pd.DataFrame()
    df_final['session_id'] = df_sessions['session_id'].unique()
    df_final['year'] = df_final['session_id'].apply(lambda x: int(str(x)[:2])).astype(np.uint8)
    df_final['month'] = df_final['session_id'].apply(lambda x: int(str(x)[2:4]) + 1).astype(np.uint8)
    df_final['weekday'] = df_final['session_id'].apply(lambda x: int(str(x)[4:6])).astype(np.uint8)
    df_final['hour'] = df_final['session_id'].apply(lambda x: int(str(x)[6:8])).astype(np.uint8)
    df_final['minute'] = df_final['session_id'].apply(lambda x: int(str(x)[8:10])).astype(np.uint8)
    df_final['second'] = df_final['session_id'].apply(lambda x: int(str(x)[10:12])).astype(np.uint8)
    df_final['ms'] = df_final['session_id'].apply(lambda x: int(str(x)[12:15])).astype(np.uint16)
    df_final['noise'] = df_final['session_id'].apply(lambda x: int(str(x)[15:17])).astype(np.uint8)
    return df_final

df_final = feature_eng(df_sessions)
df_final = df_final.set_index(['session_id'])
df_final.index.get_level_values('session_id')
day_map = {0: 'monday', 1: 'tuesday', 2:'wednesday', 3:'thursday', 4:'friday', 5:'saturday', 6:'sunday'}
df_final = pd.merge(df_accuracy, df_final, on= 'session_id')
#df_final.weekday = df_final.weekday.map(day_map)
df_final.head()


# In[4]:


plt.figure(figsize=(10, 5))
sns.countplot(x='year', data=df_final, palette=['#F96167'])
plt.title('year countplot')
plt.grid(True)
plt.show()


# In[5]:


plt.figure(figsize=(20, 5))
sns.countplot(x='month', data=df_final, palette=['#F96167', '#F96167', '#F96167', '#F96167', '#F96167', '#F96167', '#FCE77D', '#FCE77D', '#F96167', '#F96167', '#F96167'])
plt.legend(labels=['school time'], loc='upper right')
plt.title('months countplot')
plt.grid(True)
plt.show()


# In[6]:


plt.figure(figsize=(20, 5))
sns.countplot(x='weekday', data=df_final, palette = ['#F96167','#F96167',"#F96167","#F96167","#F96167","#FCE77D","#FCE77D","#F96167","#F96167","#F96167",'#F96167'], order = [1,2,3,4,5,6,0])
plt.title('weekdays countplot')
plt.legend(labels=['school time'], loc='upper right')
plt.grid(True)
plt.show()


# In[7]:


plt.figure(figsize=(20, 5))
sns.countplot(x='hour', data=df_final, palette=(['#FCE77D','#FCE77D',"#F96167","#F96167","#F96167","#F96167","#F96167","#F96167","#F96167","#F96167",'#FCE77D','#FCE77D','#FCE77D','#FCE77D','#FCE77D','#FCE77D','#FCE77D','#FCE77D','#FCE77D','#FCE77D']), order = [i for i in range(6,24)] + [i for i in range(0,6)])
plt.title('hours countplot')
plt.legend(labels=['school time'], loc='upper right')
plt.grid(True)
plt.show()


# #### Coral represents school time, we see that the game was clearly played mainly during that time span
# #### It is suprising to see that children from 8 to 11 years old play a game so late into the day, they might have been other age groups in this dataset.
# 
# *Weekend, night and season realated features are left for the reader to experiment with.*

# In[8]:


df_final['during_school']  = (df_final.weekday.isin([6,0])) | (df_final.month.isin([7,8])) | ((df_final.hour < 7) | (df_final.hour > 16 ))
df_final.during_school.value_counts()


# In[9]:


cols_to_drop = ['session_id','year','month', 'weekday','hour','minute','second','ms','noise','during_school']
# separate data into two sets based on during_school flag
data1 = df_final[df_final.during_school == True].drop(cols_to_drop, axis=1).sum() / df_final[df_final.during_school == True].shape[0]
data2 = df_final[df_final.during_school == False].drop(cols_to_drop, axis=1).sum() / df_final[df_final.during_school == False].shape[0]

# combine the data into a single DataFrame
combined_data = pd.concat([data1, data2], axis=1, keys=['During school', 'Outside of school'])

# plot the grouped bar plot
fig, ax = plt.subplots(figsize=(20, 5))
combined_data.plot(kind='bar', ax=ax, color = ['#F96167','#FCE77D'])

# set titles and labels
ax.set_title('Accuracy per question for sessions')
ax.set_ylabel('Accuracy')
ax.set_xlabel('Questions')
plt.grid(True)

plt.show()


# #### It looks like students played a little bit better when they were in class, which is a resonable assuption to make, however, due to the little sample size, this might be a coincidence.

# In[10]:


df_final['session_time'] = df_sessions.groupby('session_id')['elapsed_time'].max().reset_index(drop=True)
# calculate the mean session time for each hour of the day in minutes
hourly_data = df_final.groupby('hour')['session_time'].median()/ (1000 * 60)

# reindex the hourly_data DataFrame to sort the hours from 6 to 23 and then 0 to 5 in reverse order
hourly_data = hourly_data.reindex(index=[*range(6, 24), *range(0, 6)], fill_value=0)[::-1]

# create a horizontal bar plot
plt.figure(figsize=(10,10))
plt.barh(hourly_data.index, hourly_data.values, height=0.7, color='#8DDDD0')

# set the y-axis limits
plt.ylim(-0.5, 23.5)

# set the y-tick labels
hour_labels = [f'{h:02d}' for h in range(6, 24)] + [f'{h:02d}' for h in range(0, 6)]
plt.yticks(range(24), hour_labels[::-1])

# set the titles and labels
plt.title('Median session time by hour of the day')
plt.xlabel('Median session time (minutes)')
plt.ylabel('Hour of the day')
plt.grid(True)

plt.show()


# #### One would tend to believe that late session have a higher median for session time since people might have felt tired and went to sleep but it doesn't seem to be the case that much here. 
# 
# (We use median not mean because of the number of outliers, you can try for yourseld tho !)

# ## Overall and question focused accuracy over time:

# In[11]:


df_final['month_str'] = df_final['month'].apply(lambda x: f'{x:02d}')
df_final['year_full'] = 2000+ df_final['year']
df_final['datetime'] = pd.to_datetime(df_final['year_full'].astype(str) + df_final['month_str'], format='%Y%m')
mean_accuracy_by_month = df_final.groupby('datetime')['accuracy'].mean()
# plot the mean accuracy vs. time
plt.figure(figsize = (15,7))
plt.plot(mean_accuracy_by_month.index, mean_accuracy_by_month.values, color ='red')
plt.xlabel('Date')
plt.ylabel('Mean Accuracy')
plt.title('Accuracy Over Time')
plt.grid(True)
plt.show()


# In[12]:


fig, axes = plt.subplots(6, 3, figsize = (30,40))
for i, ax in enumerate(axes.ravel()):
    df_final.groupby('datetime')['accuracy'].mean()
    ax.plot(mean_accuracy_by_month.index, df_final.groupby('datetime')[f'q_{i+1}'].mean().values, color ='red')
    ax.set_title(f'accuracy for question {i+1} over time')
    ax.grid(True)
    ax.set_xlabel(f'Date')
    ax.set_ylabel(f'mean accuracy')
plt.show()


# ### Here is the same but with a 5window rolling mean:

# In[13]:


import pandas as pd
import matplotlib.pyplot as plt

fig, axes = plt.subplots(6, 3, figsize = (30,40))

for i, ax in enumerate(axes.ravel()):
# calculate rolling mean accuracy over a window of 3 months
    ax.plot(mean_accuracy_by_month.index, df_final.groupby('datetime')[f'q_{i+1}'].mean().rolling(window=5).mean(), color ='red')
    ax.set_title(f'accuracy for question {i+1} over time')
    ax.grid(True)
    ax.set_xlabel(f'Date')
    ax.set_ylabel(f'mean accuracy')
plt.show()



# In[ ]:




