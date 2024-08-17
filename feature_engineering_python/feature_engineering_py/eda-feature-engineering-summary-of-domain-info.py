#!/usr/bin/env python
# coding: utf-8

# # **Back ground info**

# Ref: https://www.amc.seoul.kr/asan/healthinfo/disease/diseaseDetail.do?contentId=31884  
# 'Freezing of Gait' refers to a condition in which the **center of gravity moves excessively** when walking or **the movement of both sides is unbalanced**, making it impossible to walk normally.  
#   
# It is caused by a disorder in the extrapyramidal system of the brain. Movement disorders such as hand tremor, muscle stiffness, posture disorders, and gait disorders appear. **Typical Parkinson's disease patients have difficulty** bending their bodies and **starting walking**. Once you start walking, **it is difficult to change direction, avoid obstacles, or stop**. During walking, **the movement of the upper extremity or the movement of the torso and pelvis decreases**, and the posture response is also impaired. Therefore, even if the **center of the body shakes a little, it easily falls down**.

# # **Library Load**

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import random
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


# In[2]:


Base = '/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/'


# # **Data Load & EDA**
# ### **→ Availiable csv files**

# ### **1. Subjcets**

# In[3]:


subject = pd.read_csv(Base + 'subjects.csv')
subject.head()


# In[4]:


subject.isna().sum()


# In[5]:


subject.describe()


# In[6]:


# replace Nan with 1 in Visit
subject["Visit_"] = subject["Visit"].fillna(1)

# replace Nan with 43.0 in UPDRSIII_Off
subject["UPDRSIII_Off_"] = subject["UPDRSIII_Off"].fillna(43.0)

# replace Nan with 35.0 in UPDRSIII_On
subject["UPDRSIII_On_"] = subject["UPDRSIII_On"].fillna(35.0)

subject.head()


# In[7]:


fig,ax = plt.subplots(nrows=2, ncols=1,figsize=(5,5))

sns.histplot(data=subject, x="Age",ax=ax[0])
ax[0].set_title("Distribution of Age")

sns.histplot(data=subject, x="YearsSinceDx",ax=ax[1])
ax[1].set_title("Distribution of YearsSinceDx")

fig.tight_layout()


# In[8]:


fig,ax = plt.subplots(nrows=2, ncols=1,figsize=(5,5))

sns.boxplot(data=subject, x="UPDRSIII_On",ax=ax[0])
ax[0].set_title("Distribution of UPDRSIII_On")

sns.boxplot(data=subject, x="UPDRSIII_Off",ax=ax[1])
ax[1].set_title("Distribution of UPDRSIII_Off")

fig.tight_layout()


# In[9]:


sns.histplot(data = subject, x= 'NFOGQ')


# In[10]:


sns.countplot(data = subject, x='Visit')


# In[11]:


# Check the Subjects data by Sex,Age
subject["Age"].hist(by=subject['Sex'])


# Man has more larg boundary for age

# In[12]:


# Check the Subjects data by Sex,YearsSinceDx(how long from desease)
subject["YearsSinceDx"].hist(by=subject['Sex'])


# Check NFOGQ 

# In[13]:


# Check the Subjects data by Sex,YearsSinceDx(how long from desease)
subject["NFOGQ"].hist(by=subject['Sex'])


# In[14]:


# Grouping Age and Checking wit GroupAge,NFOGQ
bins = 3
subject["Group_Age"] = pd.cut(subject.Age, bins,labels=['0','1','2'])
subject.head()


# In[15]:


# Check the Subjects data by Group_Age,NFOGQ
fig,ax = plt.subplots(nrows=1,ncols=1)
subject["NFOGQ"].hist(by=subject['Group_Age'],ax=ax)
fig.tight_layout()


# NFOGQ has no relationship with Group_Age

# Check UPDRSIII_On

# In[16]:


# Check the Subjects data by Sex, UPDRSIII_On
subject["UPDRSIII_On"].hist(by=subject['Sex'])


# Man has more boundary for UPDRSIII_On  
# Is this situation from age?

# In[17]:


# Check the Subjects data by Group_Age,UPDRSIII_On
fig,ax = plt.subplots(nrows=1,ncols=1)
subject["UPDRSIII_On"].hist(by=subject['Group_Age'],ax=ax)
fig.tight_layout()


# It seems old person has more large boundary for UPDRSIII_On

# Check UPDRSIII_Off

# In[18]:


# Check the Subjects data by Sex, UPDRSIII_Off
subject["UPDRSIII_Off"].hist(by=subject['Sex'])


# In[19]:


# Check the Subjects data by Group_Age,UPDRSIII_Off
fig,ax = plt.subplots(nrows=1,ncols=1)
subject["UPDRSIII_Off"].hist(by=subject['Group_Age'],ax=ax)
fig.tight_layout()


# It seems age has relationship with UPDRSIII_Off but middle(1) has largest boundary in the group

# #### **2. task**

# In[20]:


task = pd.read_csv(Base + 'tasks.csv')
task.head()


# In[21]:


task.isna().sum()


# In[22]:


task["Between_Begin_End"] = task["End"] - task["Begin"]
print(task.shape)
task.head()


# In[23]:


task.Task.value_counts()


# In[24]:


task.groupby("Task")["Between_Begin_End"].mean().sort_values(ascending=False)


# # **Data Load & EDA**
# ### **→ train data**

# ### 1. defog

# In[25]:


defog_list = os.listdir(Base + "/train/defog")
tdcsfog_list = os.listdir(Base + "/train/tdcsfog")


# In[26]:


defog_list[0]


# In[27]:


defog_df = pd.read_csv(Base + "/train/defog/"+defog_list[0])
defog_df.head()


# In[28]:


sns.lineplot(data = defog_df,x="Time" , y= "AccV")


# In[29]:


sns.lineplot(data = defog_df,x="Time" , y= "AccML")


# In[30]:


sns.lineplot(data = defog_df,x="Time" , y= "AccAP")


# Some signal processing skill need...

# In[31]:


print("StartHesitation: ",defog_df.StartHesitation.value_counts())
print("-----"*10)
print("Turn: ",defog_df.Turn.value_counts())
print("-----"*10)
print("Walking: ",defog_df.Walking.value_counts())


# In[32]:


defog_df.groupby(["Turn","Walking"]).count()


# In[33]:


print("Valid: ",defog_df.Valid.value_counts())
print("-----"*10)
print("Task: ",defog_df.Task.value_counts())


# In[34]:


print("Turn status 1")
defog_df[defog_df["Turn"]==1].describe().iloc[:,:4]


# In[35]:


print("Turn status 0")
defog_df[defog_df["Turn"]==0].describe().iloc[:,:4]


# In[36]:


fig,ax = plt.subplots(ncols=1,nrows=2,figsize=(10,5))

sns.lineplot(data = defog_df.iloc[80000:,:],x="Time" , y= "AccAP",ax=ax[0])
sns.lineplot(data = defog_df.iloc[80000:,:],x="Time" , y= "Turn",ax=ax[0])

sns.lineplot(data = defog_df.iloc[80000:,:],x="Time" , y= "AccAP",ax=ax[1])
sns.lineplot(data = defog_df.iloc[80000:,:],x="Time" , y= "Walking",ax=ax[1])


# In[37]:


fig,ax = plt.subplots(ncols=1,nrows=2,figsize=(10,5))

sns.lineplot(data = defog_df.iloc[80000:,:],x="Time" , y= "AccML",ax=ax[0])
sns.lineplot(data = defog_df.iloc[80000:,:],x="Time" , y= "Turn",ax=ax[0])

sns.lineplot(data = defog_df.iloc[80000:,:],x="Time" , y= "AccML",ax=ax[1])
sns.lineplot(data = defog_df.iloc[80000:,:],x="Time" , y= "Walking",ax=ax[1])


# In[38]:


fig,ax = plt.subplots(ncols=1,nrows=2,figsize=(10,5))

sns.lineplot(data = defog_df.iloc[80000:,:],x="Time" , y= "AccV",ax=ax[0])
sns.lineplot(data = defog_df.iloc[80000:,:],x="Time" , y= "Turn",ax=ax[0])

sns.lineplot(data = defog_df.iloc[80000:,:],x="Time" , y= "AccV",ax=ax[1])
sns.lineplot(data = defog_df.iloc[80000:,:],x="Time" , y= "Walking",ax=ax[1])


# In[39]:


defog_df[defog_df["Turn"]==1]


# Index at 93448, change in Turn status  
# Compare before 93448 and after 93448 to 93552

# #### **Check Turn status change with each value(AccV, AccML, AccAP)**

# In[40]:


slice_defog_df = defog_df.loc[93343:93352]
slice_defog_df


# In[41]:


fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(10,8))

sns.lineplot(data = slice_defog_df,x="Time" , y= "AccV",ax=ax[0])
ax[0].set_title("AccV")

sns.lineplot(data = slice_defog_df,x="Time", y= "AccML",ax=ax[1])
ax[1].set_title("AccML")

sns.lineplot(data = slice_defog_df,x="Time", y= "AccAP",ax=ax[2])
ax[2].set_title("AccAP")

fig.tight_layout()


# Nothing dramaticaly change at index 93348

# In[42]:


# Moving average
defog_df.loc[93341:93352][["AccV","AccML","AccAP"]].rolling(window=2).mean()


# In[43]:


tmp_df = defog_df.loc[93341:93352][["AccV","AccML","AccAP"]].rolling(window=2).mean()
tmp_df["Time"] = defog_df.loc[93341:93352]["Time"]

fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(10,8))

sns.lineplot(data = tmp_df,x="Time" , y= "AccV",ax=ax[0])
ax[0].set_title("AccV moving average")

sns.lineplot(data = tmp_df,x="Time", y= "AccML",ax=ax[1])
ax[1].set_title("AccML moving average")

sns.lineplot(data = tmp_df,x="Time", y= "AccAP",ax=ax[2])
ax[2].set_title("AccAP moving average")

fig.tight_layout()


# Hmm, It seems Moving average about AccML, AccAP has litle bit strange factor with this graph 

# In[44]:


tmp_df = defog_df.copy()
tmp_df["AccV_delta"] = (tmp_df.AccV - tmp_df.AccV.shift()).fillna(0)
tmp_df["AccML_delta"] = (tmp_df.AccML - tmp_df.AccML.shift()).fillna(0)
tmp_df["AccAP_delta"] = (tmp_df.AccAP - tmp_df.AccAP.shift()).fillna(0)

fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(10,8))

sns.lineplot(data = tmp_df.loc[93341:110000],x="Time" , y= "AccV_delta",ax=ax[0])
sns.lineplot(data = tmp_df.loc[93341:110000],x="Time" , y= "Turn",ax=ax[0])
ax[0].set_title("AccV delta")

sns.lineplot(data = tmp_df.loc[93341:110000],x="Time", y= "AccML_delta",ax=ax[1])
sns.lineplot(data = tmp_df.loc[93341:110000],x="Time" , y= "Turn",ax=ax[1])
ax[1].set_title("AccML delta")

sns.lineplot(data = tmp_df.loc[93341:110000],x="Time", y= "AccAP_delta",ax=ax[2])
sns.lineplot(data = tmp_df.loc[93341:110000],x="Time" , y= "Turn",ax=ax[2])
ax[2].set_title("AccAP delta")

fig.tight_layout()


# #### **Check Turn status change with complex(AccV, AccML, AccAP)**
# ref : https://www.mdpi.com/347228

# ![image.png](attachment:d69344df-c5dd-414e-b6d7-590015c2f06a.png)

# ### **2.Feature Engineering**

# In[45]:


defog_df["Stride"] = defog_df["AccV"] + defog_df["AccML"] + defog_df["AccAP"]
defog_df["Stride"]


# In[46]:


fig,ax = plt.subplots(ncols=1,nrows=2,figsize=(10,5))

sns.lineplot(data = defog_df.iloc[80000:,:],x="Time" , y= "Stride",ax=ax[0])
sns.lineplot(data = defog_df.iloc[80000:,:],x="Time" , y= "Turn",ax=ax[0])

sns.lineplot(data = defog_df.iloc[80000:,:],x="Time" , y= "Stride",ax=ax[1])
sns.lineplot(data = defog_df.iloc[80000:,:],x="Time" , y= "Walking",ax=ax[1])


# In[47]:


tmp_df = defog_df.loc[93341:93352][["Stride"]]
tmp_df["Time"] = defog_df.loc[93341:93352]["Time"]

fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(5,5))

sns.lineplot(data = tmp_df,x="Time" , y= "Stride",ax=ax)
ax.set_title("Stride")


# In[48]:


def sqrt_df(x):
    return np.sqrt(abs(x))

defog_df["Step"] = defog_df["Stride"].apply(sqrt_df)


# In[49]:


defog_df

