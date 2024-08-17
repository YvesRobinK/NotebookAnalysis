#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# ## Load Dataset

# In[2]:


train_logs = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/train_logs.csv')
train_scores = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/train_scores.csv')

train_logs.head()


# In[3]:


train_data = train_scores.set_index('id')


# In[4]:


train_logs.info()


# In[5]:


for col in train_logs.select_dtypes('object').columns:
    display(train_logs[col].describe())
    print('-----------------\n')


# In[6]:


# for col in train_logs.select_dtypes('object').columns:
#     display(train_logs[col].value_counts())
#     print('-----------------\n')


# In[7]:


train_logs.describe()


# In[8]:


sns.countplot(train_scores, x='score')


# In[9]:


def plot_features(train_data, label, last_n_features=-1):
    for i, col in enumerate(train_data.columns[-last_n_features:].values):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(train_data, x=col, kde=True)
        plt.title('feature distribution')

        plt.subplot(1, 2, 2)
        sns.lineplot(train_data, x=label, y=col)
        plt.title(f'relation with {label}')

        plt.tight_layout()
        plt.suptitle(col)
        plt.show()


# # Feature Engineering

# ## Keystroke Measures
# 
# > as given in the [data-collection-procedure](http://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/overview/data-collection-procedure)

# #### First and last logs of every id

# In[10]:


train_logs_initial = train_logs.loc[train_logs.groupby('id')['event_id'].idxmin()].set_index('id')
train_logs_final = train_logs.loc[train_logs.groupby('id')['event_id'].idxmax()].set_index('id')


# ### Production Stats
# 
# * total time taken to write it
# * final number of words
# * number of characters (including spaces) produced per minute

# In[11]:


train_data['total_time'] = train_logs_final['up_time'] - train_logs_initial['down_time']
train_data['word_count'] = train_logs_final.word_count
train_data['prod_rate'] = train_logs[train_logs.activity == 'Input']['id'].value_counts() * 60 * 1000 / train_data['total_time']


# In[12]:


plot_features(train_data, 'score', 3)


# ### Pause Related
# 
# * Time from the start of the keystroke logging until frst key press
# * IKI (inter-keystroke intervals) stats: Metrics of the time from a key press until the next key press
# 

# In[13]:


train_data['initial_pause_time'] = train_logs_initial['down_time']
train_logs_grouped = train_logs.groupby('id')


# In[14]:


train_data['mean_IKI'] = train_logs_grouped.down_time.agg(lambda x: (x - x.shift(1)).mean())
train_data['median_IKI'] = train_logs_grouped.down_time.agg(lambda x: (x - x.shift(1)).median())
train_data['std_IKI'] = train_logs_grouped.down_time.agg(lambda x: (x - x.shift(1)).std())


# In[15]:


plot_features(train_data, 'score', 4)


# In[16]:


for i, j in [(0.5, 1), (1, 1.5), (1.5, 2), (2, 3)]:
    train_data[f'num_IKI_{str(i)}_{str(j)}'] = train_logs_grouped.down_time.agg(
        lambda x: (((x - x.shift(1)) > i*1000) & ((x - x.shift(1)) < j*1000)).sum()
    )

train_data['num_IKI_large_3'] = train_logs_grouped.down_time.agg(lambda x: ((x - x.shift(1)) > 3000).sum())


# In[17]:


train_data['num_pauses'] = train_logs_grouped.down_time.agg(lambda x: ((x - x.shift(1)) > 1000).sum())


# In[18]:


plot_features(train_data, 'score', 6)


# more features will be added soon !!!

# # Correlation Heatmaps

# In[19]:


px.imshow(train_data.corr(method='pearson'), title='Pearson Correlation Heatmap')


# In[20]:


px.imshow(train_data.corr(method='spearman'), title='Spearman Correlation Heatmap')


# # Feature Importance

# In[21]:


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

X = train_data.sample(frac=1).reset_index().drop(['id', 'score'], axis=1)
y = train_data.sample(frac=1).reset_index()['score']

rfr = RandomForestRegressor(n_estimators=100, random_state=42)
rfr.fit(X, y)

xgbr = xgb.XGBRegressor(random_state=42)
xgbr.fit(X, y);


# In[22]:


for name, model in [('RF', rfr), ('XGB', xgbr)]:
    feature_importance = pd.DataFrame()
    feature_importance['feature'] = model.feature_names_in_
    feature_importance['importance'] = model.feature_importances_
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    sns.barplot(feature_importance, x='importance', y='feature')
    plt.suptitle(f'{name} Feature Importance')
    plt.show()


# > *This Notebook is still under progress !!*
# >
# > **Please upvote if you found it helpful !!!!**
