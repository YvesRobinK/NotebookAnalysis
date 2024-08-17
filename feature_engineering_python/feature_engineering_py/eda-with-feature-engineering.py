#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm
from xgboost import XGBRegressor
import gc


# In[2]:


train_db = pd.read_csv('../input/ventilator-pressure-prediction/train.csv')
test_db = pd.read_csv('../input/ventilator-pressure-prediction/test.csv')
train_db.head()


# In[3]:


train_db = train_db.drop(columns = 'id')
test_db = test_db.drop(columns = 'id')


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns
corrmat= train_db.corr()
plt.figure(figsize=(15,15))  

cmap = sns.diverging_palette(250, 10, s=80, l=55, n=9, as_cmap=True)

sns.heatmap(corrmat,annot=True, cmap=cmap, center=0)


# In[5]:


shades =["#f7b2b0","#c98ea6","#8f7198","#50587f", "#003f5c"]
plt.figure(figsize=(20,10))
sns.boxenplot(data = train_db,palette = shades)
plt.xticks(rotation=90)
plt.show()


# In[6]:


data_hist_plot = train_db.hist(figsize = (20,20), color = "#9F1EA0")


# In[7]:


fig, axes = plt.subplots(1, 7, figsize=(18, 5))
sns.boxplot(ax=axes[0], data=train_db, x='breath_id')
sns.boxplot(ax=axes[1], data=train_db, x='R')
sns.boxplot(ax=axes[2], data=train_db, x='C')
sns.boxplot(ax=axes[3], data=train_db, x='time_step')
sns.boxplot(ax=axes[4], data=train_db, x='u_in')
sns.boxplot(ax=axes[5], data=train_db, x='u_out')
sns.boxplot(ax=axes[6], data=train_db, x='pressure')


# In[8]:


train_db.groupby("breath_id")["time_step"].count().unique().item()


# In[9]:


test_db.groupby("breath_id")["time_step"].count().unique().item()   


# In[10]:


train_db.isnull().sum(axis = 0).to_frame()


# In[11]:


train_db.time_step.max()


# In[12]:


train_db.query('u_out == 0').time_step.max()


# In[13]:


breath_one = train_db.query('breath_id == 1').reset_index(drop = True)
breath_one


# In[14]:


breath_one.nunique().to_frame()


# In[15]:


train_db['u_in_cumsum'] = (train_db['u_in']).groupby(train_db['breath_id']).cumsum()
test_db['u_in_cumsum']  = (test_db['u_in']).groupby(test_db['breath_id']).cumsum()


# In[16]:


import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
plt.style.use('fivethirtyeight')
import seaborn as sns
import warnings
breath_928 = train_db.query('breath_id == 928').reset_index(drop = True)
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
ax.plot(breath_928["time_step"],breath_928["u_in"], lw=2, label='u_in')
ax.plot(breath_928["time_step"],breath_928["pressure"], lw=2, label='pressure')
ax.set(xlim=(0,1))
ax.legend(loc="upper right")
ax.set_xlabel("time_id", fontsize=14)
plt.show();


# In[17]:


train_db['u_in_shifted'] = train_db.groupby('breath_id')['u_in'].shift(2).fillna(method="backfill")
test_db['u_in_shifted']  = test_db.groupby('breath_id')['u_in'].shift(2).fillna(method="backfill")


# In[18]:


for df in (train_db, test_db):
    df['u_in_first']  = df.groupby('breath_id')['u_in'].transform('first')
    df['u_in_min']    = df.groupby('breath_id')['u_in'].transform('min')
    df['u_in_mean']   = df.groupby('breath_id')['u_in'].transform('mean')
    df['u_in_median'] = df.groupby('breath_id')['u_in'].transform('median')
    df['u_in_max']    = df.groupby('breath_id')['u_in'].transform('max')
    df['u_in_last']   = df.groupby('breath_id')['u_in'].transform('last')


# In[19]:


sample=pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')
X_train = train_db.drop(['pressure'], axis=1)
y_train = train_db['pressure']
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble     import HistGradientBoostingRegressor
regressor  =  HistGradientBoostingRegressor(max_iter=100,
     loss="least_absolute_deviation",early_stopping=False)
regressor.fit(X_train, y_train)
sample["pressure"] = regressor.predict(test_db)
sample.to_csv('submission.csv',index=False)


# # Please UPVOTE
