#!/usr/bin/env python
# coding: utf-8

# # EDA :

# ### Goal:
# * **Understand the data the best we can**
# * **Develop a first modeling strategy**
# 
# ### Starting checklist (train.csv):
# * **Rows and columns**: (13174211, 20)
# * **Variable types**: 7 categorical features and 13 numerical
# * **NaN value analysis**: 3 Columns full of NaN (fullscreen, hq, music), some Column have NaN due to their dependency to a defined event type, page has a lot of NaN for unknown reasons at the moment
# 
# ### Starting checklist (train_labels.csv):
# * **Target Column**: "correct"
# * **Rows and columns**: (212022, 2) 
# * **Variable types**: 2 categorical features
# * **NaN value analysis**: No NaN
# * **Target Visualisation:** 70.4% of positive rows
# 
# 
# ### Column explaination (Most of it from original Explaination) :
# * **session_id**: The session identifier, each student can have multiples but can't share them **Categorical**
# * **id**: the event identifier **Categorical**
# * **elapsed_time** :how much time has passed (in milliseconds) between the start of the session and when the event was recorded **Numerical**
# * **event_name** : the name of the event type **Categorical**
# * **name** : the event name (e.g. identifies whether a notebook_click is is opening or closing the notebook) **Categorical**
# * **level** : what level of the game the event occurred in (0 to 22) **Numerical**
# * **page** : the page number of the event (only for notebook-related events) **Numerical**
# * **room_coor_x** : the coordinates of the click in reference to the in-game room (only for click events) **Numerical**
# * **room_coor_y** : the coordinates of the click in reference to the in-game room (only for click events) **Numerical**
# * **screen_coor_x** : the coordinates of the click in reference to the player’s screen (only for click events) **Numerical**
# * **screen_coor_y** : the coordinates of the click in reference to the player’s screen (only for click events) **Numerical**
# * **hover_duration** : how long (in milliseconds) the hover happened for (only for hover events) **Numerical**
# * **text** : the text the player sees during this event **Categorical/Text**
# * **fqid** : the fully qualified ID of the event **Categorical**
# * **room_fqid** : the fully qualified ID of the room the event took place in **Categorical**
# * **text_fqid** : the fully qualified ID of the room the text appeared in **Categorical**
# * **fullscreen** : whether the player is in fullscreen mode (all NaN) **Categorical**
# * **hq** : whether the game is in high-quality (all NaN) **Categorical**
# * **music** : whether the game music is on or off  (all NaN) **Categorical**
# * **level_group** : which group of levels - and group of questions - this row belongs to (0-4, 5-12, 13-22) **Categorical**
# * **correct** : Target column, whether or not the right answer was given **Classification**
# 

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
sns.set()
pd.set_option('display.max_column', 100)


# In[2]:


dtypes={'session_id':'category', 
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


# In[3]:


#data_test = pd.read_csv("/kaggle/input/predict-student-performance-from-game-play/test.csv")
data_train =  pd.read_csv("/kaggle/input/predict-student-performance-from-game-play/train.csv")
data_train_label = pd.read_csv("/kaggle/input/predict-student-performance-from-game-play/train_labels.csv")


# In[4]:


df =  data_train.copy()
df_labels =  data_train_label.copy()

df = df.sort_values(['session_id','elapsed_time'])


# In[5]:


df.head(10)


# In[6]:


(df.isna().sum()/df.shape[0]).sort_values(ascending=False)


# **The music, fullscreen and full quality columns seems to all contain NaN values, the 9 other Nan containing columns seem to be due to their dependecy to a type of event (ig: click only event for screen_coor_y)**
# 
# To check that we can look at an heatmap using the following code: **sns.heatmap(df.isna(), cbar=False)** but that tends to take to much RAM for Kaggle Notebooks

# In[7]:


for col in df.select_dtypes('object'):
    if col not in ['text','fqid', 'room_fqid','text_fqid']: print(f'{col :-<20}: {df[col].unique()}')


# In[8]:


for col in df.select_dtypes('object'):
    if col not in ['text','fqid', 'room_fqid','text_fqid']:plt.title(col); df[col].value_counts().plot.pie(autopct='%1.1f%%'); plt.show()


# In[9]:


df['isclick'] = df['event_name'].astype('str').str[-5:] == 'click'
plt.figure(figsize=(5,5))
plt.title('Click event ratio')
(df['isclick'].value_counts(normalize=True)*100).plot.pie(labels = ["Click", "other"], autopct='%1.1f%%')
plt.show()


# **Lots of clicks, see [here](https://www.kaggle.com/code/cdeotte/game-room-click-eda) for click EDA by Chris Deotte**

# In[10]:


sns.displot(df['level'])


# In[11]:


df.session_id.value_counts().mean()


# **Each session as a mean of 1118 events**

# ## Let's now take a look at df_labels and change it a little bit:

# In[12]:


df_labels.tail(10)


# In[13]:


plt.figure(figsize=(5,5))
plt.title('Correct_rate')
(df_labels['correct'].value_counts(normalize=True)*100).plot.pie(labels = ["correct", "Not correct"], autopct='%1.1f%%')


# In[14]:


df_labels = df_labels.rename(columns={"session_id": "session_id_q"})
df_labels['session_id'] = df_labels['session_id_q'].astype(str).str[:17].astype(int)
df_labels['question'] = df_labels['session_id_q'].astype(str).str[19:].astype(int)


# In[15]:


df_labels = df_labels.loc[:,["session_id","question","correct"]]


# In[16]:


df_labels.head()


# In[17]:


df_labels[df_labels['session_id'] == 21100511290882536]


# ## Lets create a new df that informs us of each session information

# In[18]:


df_sessions = pd.DataFrame(df.session_id.astype(int))
df_sessions =df_sessions.drop_duplicates().reset_index(drop=True)

df_sessions.head(5)


# In[19]:


temp=pd.DataFrame(df.session_id.value_counts()).rename_axis('session_id0').reset_index()
temp = temp.rename(columns={"session_id0": "session_id","session_id": "num_events"})


# In[20]:


df_sessions = pd.merge(temp,df_sessions, on = ['session_id'])
df_sessions.head()


# In[21]:


plt.figure(figsize=(20,5))
sns.histplot(df_sessions.num_events, kde=True)
plt.title('num_events histogram')
plt.show()


# ## Above 2500 clicks can be considered an execption

# In[22]:


df_sessions['lots_events'] = df_sessions.num_events > 2500
df_sessions['few_events'] = df_sessions.num_events < 700
print(f"Few events count: {len(df_sessions[df_sessions.few_events])}/{len(df_sessions)}")
print(f"Lots of events count: {len(df_sessions[df_sessions.lots_events])}/{len(df_sessions)}")


# In[23]:


temp1 = pd.pivot_table(df_labels, values = 'correct', index=['session_id'], columns = 'question')
df_sessions = pd.merge(temp1,df_sessions, on = ['session_id'])
df_sessions = df_sessions.rename(columns={1 : "q_1",
 2 : "q_2",
 3 : "q_3",
 4 : "q_4",
 5 : "q_5",
 6 : "q_6",
 7 : "q_7",
 8 : "q_8",
 9 : "q_9",
 10 : "q_10",
 11 : "q_11",
 12 : "q_12",
 13 : "q_13",
 14 : "q_14",
 15 : "q_15",
 16 : "q_16",
 17 : "q_17",
 18 : "q_18",})


# In[24]:


df_sessions['accuracy'] = (df_sessions.q_1 + df_sessions.q_2 + df_sessions.q_3 + df_sessions.q_4 + df_sessions.q_5+ df_sessions.q_6 + df_sessions.q_7 +df_sessions.q_8 + df_sessions.q_9+ df_sessions.q_10 + df_sessions.q_11 + df_sessions.q_12 + df_sessions.q_13+ df_sessions.q_14 + df_sessions.q_15 + df_sessions.q_16 + df_sessions.q_17+ df_sessions.q_18)/18


# In[25]:


sns.histplot(df_sessions.accuracy)
plt.title('accuracy distribution for all sessions')
plt.show()
print(f"Average accuracy for all sessions : {df_sessions.accuracy.mean()} ")


# In[26]:


fig, ax = plt.subplots()
sns.histplot(df_sessions.accuracy[df_sessions.lots_events == True])
ax.set_xlim(0,1)
plt.title('accuracy distribution for sessions with lots of events')
plt.show()
print(f"Average accuracy for sessions with lots of events : {df_sessions.accuracy[df_sessions.lots_events == True].mean()} ")


# In[27]:


fig, ax = plt.subplots()
sns.histplot(df_sessions.accuracy[df_sessions.few_events == True])
ax.set_xlim(0,1)
plt.title('accuracy distribution for sessions with few events')
plt.show()
print(f"Average accuracy for sessions with few events : {df_sessions.accuracy[df_sessions.few_events == True].mean()} ")


# In[28]:


plt.figure(figsize=(10,5))
cols_to_drop = ['session_id', 'num_events', 'accuracy','few_events','lots_events']
data=df_sessions.drop(cols_to_drop, axis=1).sum() / df_sessions.shape[0]
plt.bar(data.index, data.values)
plt.title('accuracy per question')
plt.show()


# In[29]:


plt.figure(figsize=(10,5))
cols_to_drop = ['session_id', 'num_events', 'accuracy','few_events','lots_events']

data=df_sessions[df_sessions.lots_events == True].drop(cols_to_drop, axis=1).sum() / df_sessions[df_sessions.lots_events == True].shape[0]
plt.bar(data.index, data.values)
plt.title('accuracy per question for sessions with lots of events')
plt.show()


# In[30]:


plt.figure(figsize=(10,5))
cols_to_drop = ['session_id', 'num_events', 'accuracy','few_events','lots_events']

data=df_sessions[df_sessions.few_events == True].drop(cols_to_drop, axis=1).sum() / df_sessions[df_sessions.few_events == True].shape[0]
plt.bar(data.index, data.values)
plt.title('accuracy per question for sessions with few events')
plt.show()


# In[31]:


plt.figure(figsize=(20,5))
cols_to_drop = ['session_id', 'num_events', 'accuracy','few_events','lots_events']
data1=df_sessions.drop(cols_to_drop, axis=1).sum() / df_sessions.shape[0]
data2=df_sessions[df_sessions.few_events == True].drop(cols_to_drop, axis=1).sum() / df_sessions[df_sessions.few_events == True].shape[0]
data3=df_sessions[df_sessions.lots_events == True].drop(cols_to_drop, axis=1).sum() / df_sessions[df_sessions.lots_events == True].shape[0]
plt.bar(data1.index, data1.values, alpha = 0.5,color=['red'])
plt.bar(data2.index, data1.values,alpha = 0.5,color = ['green'])
plt.bar(data3.index, data1.values,alpha = 0.5, color = ['blue'])
plt.show()


# **We see a clear difference for sessions with a lot of clicks, we could check if specific questions are answered wrong**

# In[32]:


grouped = df.groupby('session_id')
last_elapsed_time = grouped['session_id','event_name','fqid'].tail(1)

result = last_elapsed_time.reset_index().rename(columns={'elapsed_time': 'last_elapsed_time'})
result['has_finished'] = (result.event_name == 'checkpoint') & (result.fqid == 'chap4_finale_c')
df_sessions = df_sessions.merge(result[['session_id','has_finished']], on='session_id')
df_sessions.head()


# In[33]:


print(f"Average accuracy for all sessions : {df_sessions.accuracy.mean()} ")
print(f"Average accuracy for unfinished sessions : {df_sessions.accuracy[df_sessions.has_finished == False].mean()} ")


# In[34]:


grouped = df.groupby('session_id')
last_elapsed_time = grouped['session_id','elapsed_time'].tail(1)
result = last_elapsed_time.reset_index().rename(columns={'elapsed_time': 'last_elapsed_time'})
df_sessions = df_sessions.merge(result[['session_id','last_elapsed_time']], on='session_id')
df_sessions.last_elapsed_time = df_sessions.last_elapsed_time // 1000

df_sessions.head()


# In[35]:


df_sessions.last_elapsed_time.max()


# In[36]:


fig, ax = plt.subplots()
sns.histplot(df_sessions.last_elapsed_time)
ax.set_ylim(0,1000)
ax.set_xlim(0,df_sessions.last_elapsed_time.quantile(0.95))
plt.title('last_elapsed_time histogram')
plt.show()


# In[37]:


print(f"Average accuracy for all sessions : {df_sessions.accuracy.mean()} ")
df_sessions['long_session'] = df_sessions.last_elapsed_time >= df_sessions.last_elapsed_time.quantile(0.90)
df_sessions['short_session'] = df_sessions.last_elapsed_time <= df_sessions.last_elapsed_time.quantile(0.10) 
print(f"Average accuracy for sessions with long last elapsed : {df_sessions.accuracy[df_sessions.long_session == True].mean()} ")
print(f"Average accuracy for sessions with short last elapsed : {df_sessions.accuracy[df_sessions.short_session == True].mean()} ")


# In[38]:


df_sessions.head()


# In[39]:


df_sessions.to_csv("/kaggle/working/df_sessions.csv")
df_labels.to_csv("/kaggle/working/df_labels_updated.csv")


# 

# # To be continued..
