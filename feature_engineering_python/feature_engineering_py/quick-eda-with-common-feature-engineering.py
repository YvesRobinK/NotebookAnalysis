#!/usr/bin/env python
# coding: utf-8

# We make use to [**TubesML**](https://pypi.org/project/tubesml/) as it contains some plotting functions that I find convenient for quick explorations of the data. [In this other notebook](https://www.kaggle.com/lucabasa/march-madness-model-validation-strategies), you can see how to use it to build and validate your models.

# In[1]:


get_ipython().system('pip install tubesml==0.2.0')


# In[2]:


import numpy as np 
import pandas as pd 

import tubesml as tml

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import mm_data_manipulation as dm

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Submission file
# 
# We are required to predict the probability of winning of one team against the other
# 
# ## NCAAM

# In[3]:


df = pd.read_csv('/kaggle/input/ncaam-march-mania-2021/MSampleSubmissionStage1.csv')
df[['year', 'Team_1', 'Team_2']] = pd.DataFrame(df['ID'].str.split('_').values.tolist(), index=df.index)
df['year'] = pd.to_numeric(df.year)
df.head()


# We have this many games per year, indicating all the possible combinations of games each year. We will be evaluated only on the games that actually happened (naturally)

# In[4]:


df.year.value_counts(dropna=False).sort_index()


# # NCAAW

# In[5]:


df = pd.read_csv('/kaggle/input/ncaaw-march-mania-2021/WSampleSubmissionStage1.csv')
df[['year', 'Team_1', 'Team_2']] = pd.DataFrame(df['ID'].str.split('_').values.tolist(), index=df.index)
df['year'] = pd.to_numeric(df.year)
df.head()


# In[6]:


df.year.value_counts(dropna=False).sort_index()


# # Teams.csv
# 
# ## NCAAM

# In[7]:


df = pd.read_csv('/kaggle/input/ncaam-march-mania-2021/MTeams.csv')
print(f'Shape: {df.shape}')
df.head()


# In[8]:


df['years_in_d1'] = df['LastD1Season'] - df['FirstD1Season']

df['years_in_d1'].hist(bins=20, figsize=(12,8))
plt.grid(False)
plt.title('Number of Years in Division 1', fontsize=16)
plt.show()


# In[9]:


df.FirstD1Season.hist(bins=20, alpha=0.7, label='First Season', figsize=(12,8))
df.LastD1Season.hist(bins=20, alpha=0.7, label='Last Season', figsize=(12,8))
plt.grid(False)
plt.legend()
plt.title('Distribution of First and Last Season in D1', fontsize=16)
plt.show()


# In[10]:


yr_count = pd.DataFrame({'year': np.arange(1985, 2022)})

for year in yr_count.year:
    df['is_in'] = 0
    df.loc[(df.FirstD1Season <= year) & (df.LastD1Season >= year), 'is_in'] = 1
    tot_teams = df.is_in.sum()
    yr_count.loc[yr_count.year == year, 'n_teams'] = tot_teams
    
yr_count = yr_count.set_index('year')
yr_count.n_teams.plot(figsize=(12,8))
plt.title('Number of teams per year', fontsize=16)
plt.show()


# # NCAAW
# 
# There is no information about first and last NCAA season

# In[11]:


df = pd.read_csv('/kaggle/input/ncaaw-march-mania-2021/WTeams.csv')
print(f'Shape: {df.shape}')
df.head()


# # Seasons.csv
# 
# ## NCAAM

# In[12]:


df = pd.read_csv('/kaggle/input/ncaam-march-mania-2021/MSeasons.csv')
print(f'Shape: {df.shape}')
df.head()


# In[13]:


df.RegionW.value_counts()


# In[14]:


df.RegionX.value_counts()


# In[15]:


df.RegionY.value_counts()


# In[16]:


df.RegionZ.value_counts()


# ## NCAAW

# In[17]:


df = pd.read_csv('/kaggle/input/ncaaw-march-mania-2021/WSeasons.csv')
print(f'Shape: {df.shape}')
df.head()


# In[18]:


df.RegionW.value_counts()


# In[19]:


df.RegionX.value_counts()


# In[20]:


df.RegionY.value_counts()


# In[21]:


df.RegionZ.value_counts()


# # Regular Season Compact Results
# 
# These are files that simply summarize when a game happened, where it was, and how it ended.
# 
# ## NCAAM

# In[22]:


df = pd.read_csv('/kaggle/input/ncaam-march-mania-2021/MRegularSeasonCompactResults.csv')
print(f'Shape: {df.shape}')
df.head()


# In[23]:


df['point_diff'] = df.WScore - df.LScore
df.point_diff.hist(bins=30, figsize=(12,8))
plt.grid(False)
plt.title('Point difference distribution in the regular season', fontsize=16)
plt.show()


# Half of the games were won by less than 10 points of difference, one game had 94 points of difference (!!!)

# In[24]:


df.describe()


# Creating some overall statistics

# In[25]:


summaries = df[['Season', 
    'WScore', 
    'LScore', 
    'NumOT', 
    'point_diff']].groupby('Season').agg(['min', 'max', 'mean', 'median'])

summaries.columns = ['_'.join(col).strip() for col in summaries.columns.values]
summaries


# In[26]:


summaries[[col for col in summaries.columns if 'point_diff' in col and 'sum' not in col]].plot(figsize=(12,8))
plt.title('Point difference over time', fontsize=16)
plt.show()


# Taking into account the court where the games were played

# In[27]:


summaries = df[['Season', 'WLoc',
    'WScore', 
    'LScore', 
    'NumOT', 
    'point_diff']].groupby(['Season', 'WLoc']).agg(['min', 'max', 'mean', 'median', 'count'])

summaries.columns = ['_'.join(col).strip() for col in summaries.columns.values]
summaries


# In[28]:


summaries[['point_diff_mean']].unstack().plot(figsize=(12,8))
plt.title('Point difference over time', fontsize=16)
plt.show()


# ## NCAAW

# In[29]:


df = pd.read_csv('/kaggle/input/ncaaw-march-mania-2021/WRegularSeasonCompactResults.csv')
print(f'Shape: {df.shape}')
df.head()


# In[30]:


df['point_diff'] = df.WScore - df.LScore
df.point_diff.hist(bins=30, figsize=(12,8))
plt.grid(False)
plt.title('Point difference distribution in the regular season', fontsize=16)
plt.show()


# On average, games in the Women's tournament end with a larger point differential. The biggest one being a game ended with 108 points of difference.

# In[31]:


df.describe()


# In[32]:


summaries = df[['Season', 
    'WScore', 
    'LScore', 
    'NumOT', 
    'point_diff']].groupby('Season').agg(['min', 'max', 'mean', 'median'])

summaries.columns = ['_'.join(col).strip() for col in summaries.columns.values]
summaries


# In[33]:


summaries[[col for col in summaries.columns if 'point_diff' in col and 'sum' not in col]].plot(figsize=(12,8))
plt.title('Point difference over time', fontsize=16)
plt.show()


# In[34]:


summaries = df[['Season', 'WLoc',
    'WScore', 
    'LScore', 
    'NumOT', 
    'point_diff']].groupby(['Season', 'WLoc']).agg(['min', 'max', 'mean', 'median', 'count'])

summaries.columns = ['_'.join(col).strip() for col in summaries.columns.values]
summaries


# In[35]:


summaries[['point_diff_mean']].unstack().plot(figsize=(12,8))
plt.title('Point difference over time', fontsize=16)
plt.show()


# # Playoff compact result
# 
# Similar to the previous section, but for Playoff games. We are going to predict games from this group (well, that will be in this group next year as the games did not happen yet).
# 
# ## NCAAM

# In[36]:


df = pd.read_csv('/kaggle/input/ncaam-march-mania-2021/MNCAATourneyCompactResults.csv')
print(f'Shape: {df.shape}')
df.head()


# In[37]:


df['point_diff'] = df.WScore - df.LScore
df.point_diff.hist(bins=30, figsize=(12,8))
plt.grid(False)
plt.title('Point difference distribution in the playoff', fontsize=16)
plt.show()


# The distribution is very similar to the one in the regular season, but with less extreme outcomes

# In[38]:


df.describe()


# In[39]:


summaries = df[['Season', 
    'WScore', 
    'LScore', 
    'NumOT', 
    'point_diff']].groupby('Season').agg(['min', 'max', 'mean', 'median'])

summaries.columns = ['_'.join(col).strip() for col in summaries.columns.values]
summaries


# In[40]:


summaries[[col for col in summaries.columns if 'point_diff' in col and 'sum' not in col]].plot(figsize=(12,8))
plt.title('Point difference over time', fontsize=16)
plt.show()


# ## NCAAW

# In[41]:


df = pd.read_csv('/kaggle/input/ncaaw-march-mania-2021/WNCAATourneyCompactResults.csv')
print(f'Shape: {df.shape}')
df.head()


# In[42]:


df['point_diff'] = df.WScore - df.LScore
df.point_diff.hist(bins=30, figsize=(12,8))
plt.grid(False)
plt.title('Point difference distribution in the playoff', fontsize=16)
plt.show()


# Even though the extreme results are less extreme, on average the Women's tournament games end with a bigger point differential during the playoff than the regular season

# In[43]:


df.describe()


# In[44]:


summaries = df[['Season', 
    'WScore', 
    'LScore', 
    'NumOT', 
    'point_diff']].groupby('Season').agg(['min', 'max', 'mean', 'median'])

summaries.columns = ['_'.join(col).strip() for col in summaries.columns.values]
summaries


# In[45]:


summaries[[col for col in summaries.columns if 'point_diff' in col and 'sum' not in col]].plot(figsize=(12,8))
plt.title('Point difference over time', fontsize=16)
plt.show()


# # Regular Season Detailed results
# 
# These datasets give more information about the games by adding some boxscore stats
# 
# ## NCAAM

# In[46]:


reg_season_m = pd.read_csv('/kaggle/input/ncaam-march-mania-2021/MRegularSeasonDetailedResults.csv')
print(f'Original shape: {reg_season_m.shape}')
stats = [col for col in reg_season_m.columns if 'W' in col and 'ID' not in col and 'Loc' not in col]

reg_season_m = dm.process_details(reg_season_m)
print(f'Processed shape: {reg_season_m.shape}')
reg_season_m.head()


# In[47]:


not_sum = ['WTeamID', 'DayNum', 'LTeamID']
to_sum = [col for col in reg_season_m.columns if col not in not_sum]

summaries = reg_season_m[to_sum].groupby(['Season', 'WLoc']).agg(['min', 'max', 'mean', 'median', 'count'])

summaries.columns = ['_'.join(col).strip() for col in summaries.columns.values]


fig, ax= plt.subplots(6,2, figsize=(15, 6*6))

i = 0

for col in [c for c in summaries.columns if '_perc_mean' in c and c.startswith('W')]:
    name = col.split('_perc_')[0][1:]
    summaries[col].unstack().plot(title='Mean percenteage of '+name+', Winners',ax=ax[i][0])
    summaries['L'+name+'_perc_mean'].unstack().plot(title='Mean percenteage of '+name+', Losers',ax=ax[i][1])
    ax[i][0].legend(labels=['Away', 'Home', 'Neutral'])
    ax[i][1].legend(labels=['Away', 'Home', 'Neutral'])
    ax[i][0].set_ylim(0,1)
    ax[i][1].set_ylim(0,1)
    i += 1


# In[48]:


fig, ax= plt.subplots(7,2, figsize=(15, 6*7))

i, j = 0, 0

for col in stats:
    name = col[1:]
    summaries[[c for c in summaries.columns if name+'_diff_mean' in c]].unstack().plot(title='Difference in mean '+name,ax=ax[i][j])
    ax[i][j].legend(labels=['Away', 'Home', 'Neutral'])
    if j == 0:
        j = 1
    else:
        j = 0
        i += 1


# ## NCAAW

# In[49]:


reg_season_w = pd.read_csv('/kaggle/input/ncaaw-march-mania-2021/WRegularSeasonDetailedResults.csv')
print(f'Original shape: {reg_season_w.shape}')
stats = [col for col in reg_season_w.columns if 'W' in col and 'ID' not in col and 'Loc' not in col]

reg_season_w = dm.process_details(reg_season_w)
print(f'Processed shape: {reg_season_w.shape}')
reg_season_w.head()


# In[50]:


not_sum = ['WTeamID', 'DayNum', 'LTeamID']
to_sum = [col for col in reg_season_w.columns if col not in not_sum]

summaries = reg_season_w[to_sum].groupby(['Season', 'WLoc']).agg(['min', 'max', 'mean', 'median', 'count'])

summaries.columns = ['_'.join(col).strip() for col in summaries.columns.values]


fig, ax= plt.subplots(6,2, figsize=(15, 6*6))

i = 0

for col in [c for c in summaries.columns if '_perc_mean' in c and c.startswith('W')]:
    name = col.split('_perc_')[0][1:]
    summaries[col].unstack().plot(title='Mean percenteage of '+name+', Winners',ax=ax[i][0])
    summaries['L'+name+'_perc_mean'].unstack().plot(title='Mean percenteage of '+name+', Losers',ax=ax[i][1])
    ax[i][0].legend(labels=['Away', 'Home', 'Neutral'])
    ax[i][1].legend(labels=['Away', 'Home', 'Neutral'])
    ax[i][0].set_ylim(0,1)
    ax[i][1].set_ylim(0,1)
    i += 1


# In[51]:


fig, ax= plt.subplots(7,2, figsize=(15, 6*7))

i, j = 0, 0

for col in stats:
    name = col[1:]
    summaries[[c for c in summaries.columns if name+'_diff_mean' in c]].unstack().plot(title='Difference in mean '+name,ax=ax[i][j])
    ax[i][j].legend(labels=['Away', 'Home', 'Neutral'])
    if j == 0:
        j = 1
    else:
        j = 0
        i += 1


# # Playoff Detailed Results
# 
# As in the previous section, but for the playoff
# 
# ## NCAAM

# In[52]:


playoff_m = pd.read_csv('/kaggle/input/ncaam-march-mania-2021/MNCAATourneyDetailedResults.csv')
print(f'Original shape: {playoff_m.shape}')
playoff_m = dm.process_details(playoff_m)
print(f'Processed shape: {playoff_m.shape}')
playoff_m.head()


# In[53]:


not_sum = ['WTeamID', 'DayNum', 'LTeamID']
to_sum = [col for col in playoff_m.columns if col not in not_sum]

summaries = playoff_m[to_sum].groupby(['Season']).agg(['min', 'max', 'mean', 'median', 'count'])

summaries.columns = ['_'.join(col).strip() for col in summaries.columns.values]
summaries


# In[54]:


fig, ax= plt.subplots(7,2, figsize=(15, 6*7))

i, j = 0, 0

for col in stats:
    name = col[1:]
    summaries[[c for c in summaries.columns if name+'_diff_mean' in c]].plot(title='Difference in mean '+name,ax=ax[i][j])
    if j == 0:
        j = 1
    else:
        j = 0
        i += 1


# In[55]:


fig, ax= plt.subplots(6,2, figsize=(15, 6*6))

i = 0

for col in [c for c in summaries.columns if '_perc_mean' in c and c.startswith('W')]:
    name = col.split('_perc_')[0][1:]
    summaries[col].plot(title='Mean percenteage of '+name+', Winners',ax=ax[i][0])
    summaries['L'+name+'_perc_mean'].plot(title='Mean percenteage of '+name+', Losers',ax=ax[i][1])
    ax[i][0].set_ylim(0,1)
    ax[i][1].set_ylim(0,1)
    i += 1


# ## NCAAW

# In[56]:


playoff_w = pd.read_csv('/kaggle/input/ncaaw-march-mania-2021/WNCAATourneyDetailedResults.csv')
print(f'Original shape: {playoff_w.shape}')
playoff_w = dm.process_details(playoff_w)
print(f'Processed shape: {playoff_w.shape}')
playoff_w.head()


# In[57]:


not_sum = ['WTeamID', 'DayNum', 'LTeamID']
to_sum = [col for col in playoff_w.columns if col not in not_sum]

summaries = playoff_w[to_sum].groupby(['Season']).agg(['min', 'max', 'mean', 'median', 'count'])

summaries.columns = ['_'.join(col).strip() for col in summaries.columns.values]
summaries


# In[58]:


fig, ax= plt.subplots(7,2, figsize=(15, 6*7))

i, j = 0, 0

for col in stats:
    name = col[1:]
    summaries[[c for c in summaries.columns if name+'_diff_mean' in c]].plot(title='Difference in mean '+name,ax=ax[i][j])
    if j == 0:
        j = 1
    else:
        j = 0
        i += 1


# In[59]:


fig, ax= plt.subplots(6,2, figsize=(15, 6*6))

i = 0

for col in [c for c in summaries.columns if '_perc_mean' in c and c.startswith('W')]:
    name = col.split('_perc_')[0][1:]
    summaries[col].plot(title='Mean percenteage of '+name+', Winners',ax=ax[i][0])
    summaries['L'+name+'_perc_mean'].plot(title='Mean percenteage of '+name+', Losers',ax=ax[i][1])
    ax[i][0].set_ylim(0,1)
    ax[i][1].set_ylim(0,1)
    i += 1


# # Putting things together
# 
# Here we make use of some functions made for the past competitions to create more boxscore stats
# 
# ## NCAAM

# In[60]:


reg_m = dm.full_stats(reg_season_m)
reg_m.head()


# In[61]:


reg_m.hist(bins=50, figsize=(18, 18), grid=False)
plt.show()


# In[62]:


summary_reg = reg_m.groupby('Season')[[col for col in reg_m if col not in ['TeamID', 'Season']]].agg(['mean', 'max', 'min'])
summary_reg.columns = ['_'.join(col).strip() for col in summary_reg.columns.values]
summary_reg


# In[63]:


stats = [col.split('_mean')[0] for col in summary_reg if '_mean' in col and 'diff_' not in col and 'advantage' not in col]

fig, ax= plt.subplots(int(np.ceil(len(stats)/2)),2, figsize=(15, 6*int(np.ceil(len(stats)/2))))

i, j = 0, 0

for col in stats:
    summary_reg[[col+'_mean', col+'_max', col+'_min']].plot(title=col,ax=ax[i][j])
    ax[i][j].legend(labels=['Mean', 'Max', 'Min'])
    if j == 0:
        j = 1
    else:
        j = 0
        i += 1

plt.show()


# ## NCAAW

# In[64]:


reg_w = dm.full_stats(reg_season_w)
reg_w.head()


# In[65]:


reg_w.hist(bins=50, figsize=(18, 18), grid=False)
plt.show()


# In[66]:


summary_reg = reg_w.groupby('Season')[[col for col in reg_w if col not in ['TeamID', 'Season']]].agg(['mean', 'max', 'min'])
summary_reg.columns = ['_'.join(col).strip() for col in summary_reg.columns.values]
summary_reg


# In[67]:


stats = [col.split('_mean')[0] for col in summary_reg if '_mean' in col and 'diff_' not in col and 'advantage' not in col]

fig, ax= plt.subplots(int(np.ceil(len(stats)/2)),2, figsize=(15, 6*int(np.ceil(len(stats)/2))))

i, j = 0, 0

for col in stats:
    summary_reg[[col+'_mean', col+'_max', col+'_min']].plot(title=col,ax=ax[i][j])
    ax[i][j].legend(labels=['Mean', 'Max', 'Min'])
    if j == 0:
        j = 1
    else:
        j = 0
        i += 1

plt.show()


# # Training data
# 
# To make the training data we start from the datasets of the previous section and we make sure each game is present twice, once as is and once with the 2 teams in opposite order. This doubles the training instances and allows the model to train on recognizing losses as well (the original data only shows the wins).
# 
# Each row will be a game and the features are going to be the statistics of each team during the corresponding season.
# 
# We also add a few more features, like the Seed and the Round the game will take place
# 
# ## NCAAM

# In[68]:


def make_training_data(details, targets):
    tmp = details.copy()
    tmp.columns = ['Season', 'Team1'] + \
                ['T1_'+col for col in tmp.columns if col not in ['Season', 'TeamID']]
    total = pd.merge(targets, tmp, on=['Season', 'Team1'], how='left')

    tmp = details.copy()
    tmp.columns = ['Season', 'Team2'] + \
                ['T2_'+col for col in tmp.columns if col not in ['Season', 'TeamID']]
    total = pd.merge(total, tmp, on=['Season', 'Team2'], how='left')
    
    if total.isnull().any().any():
        raise ValueError('Something went wrong')
        
    stats = [col[3:] for col in total.columns if 'T1_' in col and 'region' not in col]

    for stat in stats:
        total['delta_'+stat] = total['T1_'+stat] - total['T2_'+stat]
        
    try:
        total['delta_off_edge'] = total['T1_off_rating'] - total['T2_def_rating']
        total['delta_def_edge'] = total['T2_off_rating'] - total['T1_def_rating']
    except KeyError:
        pass
        
    return total


def add_seed(seed_location, total):
    seed_data = pd.read_csv(seed_location)
    seed_data['region'] = seed_data['Seed'].apply(lambda x: x[0])
    seed_data['Seed'] = seed_data['Seed'].apply(lambda x: int(x[1:3]))
    total = pd.merge(total, seed_data, how='left', on=['TeamID', 'Season'])
    return total


def add_stage(data):
    data.loc[(data.T1_region == 'W') & (data.T2_region == 'X'), 'stage'] = 'finalfour'
    data.loc[(data.T1_region == 'X') & (data.T2_region == 'W'), 'stage'] = 'finalfour'
    data.loc[(data.T1_region == 'Y') & (data.T2_region == 'Z'), 'stage'] = 'finalfour'
    data.loc[(data.T1_region == 'Z') & (data.T2_region == 'Y'), 'stage'] = 'finalfour'
    data.loc[(data.T1_region == 'W') & (data.T2_region.isin(['Y', 'Z'])), 'stage'] = 'final'
    data.loc[(data.T1_region == 'X') & (data.T2_region.isin(['Y', 'Z'])), 'stage'] = 'final'
    data.loc[(data.T1_region == 'Y') & (data.T2_region.isin(['W', 'X'])), 'stage'] = 'final'
    data.loc[(data.T1_region == 'Z') & (data.T2_region.isin(['W', 'X'])), 'stage'] = 'final'
    data.loc[(data.T1_region == data.T2_region) & (data.T1_Seed + data.T2_Seed == 17), 'stage'] = 'Round1'
    
    fil = data.stage.isna()
    
    data.loc[fil & (data.T1_Seed.isin([1, 16])) & (data.T2_Seed.isin([8, 9])), 'stage'] = 'Round2'
    data.loc[fil & (data.T1_Seed.isin([8, 9])) & (data.T2_Seed.isin([1, 16])), 'stage'] = 'Round2'
    data.loc[fil & (data.T1_Seed.isin([5, 12])) & (data.T2_Seed.isin([4, 13])), 'stage'] = 'Round2'
    data.loc[fil & (data.T1_Seed.isin([4, 13])) & (data.T2_Seed.isin([5, 12])), 'stage'] = 'Round2'
    data.loc[fil & (data.T1_Seed.isin([6, 11])) & (data.T2_Seed.isin([3, 14])), 'stage'] = 'Round2'
    data.loc[fil & (data.T1_Seed.isin([3, 14])) & (data.T2_Seed.isin([6, 11])), 'stage'] = 'Round2'
    data.loc[fil & (data.T1_Seed.isin([7, 10])) & (data.T2_Seed.isin([2, 15])), 'stage'] = 'Round2'
    data.loc[fil & (data.T1_Seed.isin([2, 15])) & (data.T2_Seed.isin([7, 10])), 'stage'] = 'Round2'
    
    fil = data.stage.isna()
    
    data.loc[fil & (data.T1_Seed.isin([1, 16, 8, 9])) & (data.T2_Seed.isin([4, 5, 12, 13])), 'stage'] = 'Round3'
    data.loc[fil & (data.T1_Seed.isin([4, 5, 12, 13])) & (data.T2_Seed.isin([1, 16, 8, 9])), 'stage'] = 'Round3'
    data.loc[fil & (data.T1_Seed.isin([3, 6, 11, 14])) & (data.T2_Seed.isin([2, 7, 10, 15])), 'stage'] = 'Round3'
    data.loc[fil & (data.T1_Seed.isin([2, 7, 10, 15])) & (data.T2_Seed.isin([3, 6, 11, 14])), 'stage'] = 'Round3'
    
    fil = data.stage.isna()
    
    data.loc[fil & (data.T1_Seed.isin([1, 16, 8, 9, 4, 5, 12, 13])) & 
             (data.T2_Seed.isin([3, 6, 11, 14, 2, 7, 10, 15])), 'stage'] = 'Round4'
    data.loc[fil & (data.T1_Seed.isin([3, 6, 11, 14, 2, 7, 10, 15])) & 
             (data.T2_Seed.isin([1, 16, 8, 9, 4, 5, 12, 13])), 'stage'] = 'Round4'
    
    data.loc[data.stage.isna(), 'stage'] = 'impossible'
    
    #data = pd.get_dummies(data, columns=['stage'])
    
    del data['T1_region']
    del data['T2_region']
    
    return data


def make_teams_target(data, league):
    if league == 'men':
        limit = 2003
    else:
        limit = 2010

    df = data[data.Season >= limit].copy()

    df['Team1'] = np.where((df.WTeamID < df.LTeamID), df.WTeamID, df.LTeamID)
    df['Team2'] = np.where((df.WTeamID > df.LTeamID), df.WTeamID, df.LTeamID)
    df['target'] = np.where((df['WTeamID'] < df['LTeamID']),1,0)
    df['target_points'] = np.where((df['WTeamID'] < df['LTeamID']),df.WScore - df.LScore,df.LScore - df.WScore)
    df.loc[df.WLoc == 'N', 'LLoc'] = 'N'
    df.loc[df.WLoc == 'H', 'LLoc'] = 'A'
    df.loc[df.WLoc == 'A', 'LLoc'] = 'H'
    df['T1_Loc'] = np.where((df.WTeamID < df.LTeamID), df.WLoc, df.LLoc)
    df['T2_Loc'] = np.where((df.WTeamID > df.LTeamID), df.WLoc, df.LLoc)
    df['T1_Loc'] = df['T1_Loc'].map({'H': 1, 'A': -1, 'N': 0})
    df['T2_Loc'] = df['T2_Loc'].map({'H': 1, 'A': -1, 'N': 0})

    reverse = data[data.Season >= limit].copy()
    reverse['Team1'] = np.where((reverse.WTeamID > reverse.LTeamID), reverse.WTeamID, reverse.LTeamID)
    reverse['Team2'] = np.where((reverse.WTeamID < reverse.LTeamID), reverse.WTeamID, reverse.LTeamID)
    reverse['target'] = np.where((reverse['WTeamID'] > reverse['LTeamID']),1,0)
    reverse['target_points'] = np.where((reverse['WTeamID'] > reverse['LTeamID']),
                                        reverse.WScore - reverse.LScore,
                                        reverse.LScore - reverse.WScore)
    reverse.loc[reverse.WLoc == 'N', 'LLoc'] = 'N'
    reverse.loc[reverse.WLoc == 'H', 'LLoc'] = 'A'
    reverse.loc[reverse.WLoc == 'A', 'LLoc'] = 'H'
    reverse['T1_Loc'] = np.where((reverse.WTeamID > reverse.LTeamID), reverse.WLoc, reverse.LLoc)
    reverse['T2_Loc'] = np.where((reverse.WTeamID < reverse.LTeamID), reverse.WLoc, reverse.LLoc)
    reverse['T1_Loc'] = reverse['T1_Loc'].map({'H': 1, 'A': -1, 'N': 0})
    reverse['T2_Loc'] = reverse['T2_Loc'].map({'H': 1, 'A': -1, 'N': 0})
    
    df = pd.concat([df, reverse], ignore_index=True)

    to_drop = ['WScore','WTeamID', 'LTeamID', 'LScore', 'WLoc', 'LLoc', 'NumOT']
    for col in to_drop:
        del df[col]
    
    df.loc[:,'ID'] = df.Season.astype(str) + '_' + df.Team1.astype(str) + '_' + df.Team2.astype(str)
    return df


def prepare_data(league):
    save_loc = 'processed_data/' + league + '/'

    if league == 'women':
        regular_season = '/kaggle/input/ncaaw-march-mania-2021-spread/WRegularSeasonDetailedResults.csv'
        playoff = '/kaggle/input/ncaaw-march-mania-2021/WNCAATourneyDetailedResults.csv'
        playoff_compact = '/kaggle/input/ncaaw-march-mania-2021/WNCAATourneyCompactResults.csv'
        seed = '/kaggle/input/ncaaw-march-mania-2021/WNCAATourneySeeds.csv'
        save_loc = 'data/processed_women/'
    else:
        regular_season = '/kaggle/input/ncaam-march-mania-2021-spread/MRegularSeasonDetailedResults.csv'
        playoff = '/kaggle/input/ncaam-march-mania-2021/MNCAATourneyDetailedResults.csv'
        playoff_compact = '/kaggle/input/ncaam-march-mania-2021/MNCAATourneyCompactResults.csv'
        seed = '/kaggle/input/ncaam-march-mania-2021/MNCAATourneySeeds.csv'
        save_loc = 'data/processed_men/'
    
    # Season stats
    reg = pd.read_csv(regular_season)
    reg = dm.process_details(reg)
    regular_stats = dm.full_stats(reg)
    
    regular_stats = add_seed(seed, regular_stats)    
    
    # Target data generation 
    target_data = pd.read_csv(playoff_compact)
    target_data = make_teams_target(target_data, league)
    
    all_reg = make_training_data(regular_stats, target_data)
    all_reg = all_reg[all_reg.DayNum >= 136]  # remove pre tourney 
    all_reg = add_stage(all_reg)
    
    return all_reg


# In[69]:


train_men = prepare_data('men')
train_men.head()


# In[70]:


high_corr = tml.plot_correlations(train_men, target='target_points', limit=20)


# In[71]:


tml.corr_target(train_men, 'target_points', list(high_corr[2:].index), x_estimator=None)


# In[72]:


tmp = train_men[train_men.stage.isin(['Round1', 'Round2'])].groupby(['T1_Seed', 'stage'])['target'].agg('mean').unstack()

fig, ax = plt.subplots(1, 1, figsize=(13, 8))

tmp.plot(kind='barh', ax=ax)
ax.axvline(0.5, color='k', linestyle='--')
ax.set_title('Percentage of victory in the first 2 rounds', fontsize=16)
ax.set_xlabel('% of victory', fontsize=12)
ax.set_ylabel('Seed', fontsize=12)

plt.show()


# In[73]:


tml.segm_target(data=train_men, target='target_points', cat='stage')


# ## NCAAW

# In[74]:


train_women = prepare_data('women')
train_women.head()


# In[75]:


high_corr = tml.plot_correlations(train_women, target='target_points', limit=20)


# In[76]:


tml.corr_target(train_women, 'target_points', list(high_corr[2:].index), x_estimator=None)


# In[77]:


tmp = train_women[train_women.stage.isin(['Round1', 'Round2'])].groupby(['T1_Seed', 'stage'])['target'].agg('mean').unstack()

fig, ax = plt.subplots(1, 1, figsize=(13, 8))

tmp.plot(kind='barh', ax=ax)
ax.axvline(0.5, color='k', linestyle='--')
ax.set_title('Percentage of victory in the first 2 rounds', fontsize=16)
ax.set_xlabel('% of victory', fontsize=12)
ax.set_ylabel('Seed', fontsize=12)

plt.show()


# In[78]:


tml.segm_target(data=train_women, target='target_points', cat='stage')


# # To Be continued

# In[ ]:




