#!/usr/bin/env python
# coding: utf-8

# # Adapting the 2nd Place Solution of the 2021 NCAAW Competition
# 
# - Solution write-up is available here : https://www.kaggle.com/c/ncaaw-march-mania-2021/discussion/230705
# - 2nd place code : https://www.kaggle.com/theoviel/2nd-place-ncaaw-2021?scriptVersionId=58780991
# - It was adapted from this notebook : https://www.kaggle.com/theoviel/ncaa-starter-the-simpler-the-better 
# 
# 
#   - Version 6: Women
#   - Version 10: Men

# In[1]:


import os
import re
import sklearn
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.metrics import *
from sklearn.linear_model import *
from sklearn.model_selection import *

pd.set_option('display.max_columns', None)


# In[2]:


DATA_PATH = '../input/mens-march-mania-2022/MDataFiles_Stage1/'

for filename in os.listdir(DATA_PATH):
    print(filename)


# # Data preparation

# ## Seeds
# > This file identifies the seeds for all teams in each NCAA® tournament, for all seasons of historical data. Thus, there are between 64-68 rows for each year, depending on whether there were any play-in games and how many there were. In recent years the structure has settled at 68 total teams, with four "play-in" games leading to the final field of 64 teams entering Round 1 on Thursday of the first week (by definition, that is DayNum=136 each season). We will not know the seeds of the respective tournament teams, or even exactly which 68 teams it will be, until Selection Sunday on March 15, 2020 (DayNum=132).
# 
# > The seed is a 3/4-character :
# - First character : Region (W, X, Y, or Z)
# - Next two digits : Seed within the region (01 to 16)
# - Last character (optional): Distinguishes teams between play-ins ( a or b)

# In[3]:


df_seeds = pd.read_csv(DATA_PATH + "MNCAATourneySeeds.csv")
df_seeds.head()


# ## Season results
# > This file identifies the game-by-game results for many seasons of historical data, starting with the 1985 season (the first year the NCAA® had a 64-team tournament). For each season, the file includes all games played from DayNum 0 through 132. It is important to realize that the "Regular Season" games are simply defined to be all games played on DayNum=132 or earlier (DayNum=132 is Selection Sunday, and there are always a few conference tournament finals actually played early in the day on Selection Sunday itself). Thus a game played on or before Selection Sunday will show up here whether it was a pre-season tournament, a non-conference game, a regular conference game, a conference tournament game, or whatever.

# In[4]:


df_season_results = pd.read_csv(DATA_PATH + "MRegularSeasonCompactResults.csv")
df_season_results.drop(['NumOT', 'WLoc'], axis=1, inplace=True)


# In[5]:


df_season_results['ScoreGap'] = df_season_results['WScore'] - df_season_results['LScore']


# In[6]:


df_season_results.head()


# ### Features 
# For each team at each season, I compute : 
# - Number of wins
# - Number of losses
# - Average score gap of wins
# - Average score gap of losses
# 
# And use the following features : 
# - Win Ratio
# - Average score gap

# In[7]:


num_win = df_season_results.groupby(['Season', 'WTeamID']).count()
num_win = num_win.reset_index()[['Season', 'WTeamID', 'DayNum']].rename(columns={"DayNum": "NumWins", "WTeamID": "TeamID"})


# In[8]:


num_loss = df_season_results.groupby(['Season', 'LTeamID']).count()
num_loss = num_loss.reset_index()[['Season', 'LTeamID', 'DayNum']].rename(columns={"DayNum": "NumLosses", "LTeamID": "TeamID"})


# In[9]:


gap_win = df_season_results.groupby(['Season', 'WTeamID']).mean().reset_index()
gap_win = gap_win[['Season', 'WTeamID', 'ScoreGap']].rename(columns={"ScoreGap": "GapWins", "WTeamID": "TeamID"})


# In[10]:


gap_loss = df_season_results.groupby(['Season', 'LTeamID']).mean().reset_index()
gap_loss = gap_loss[['Season', 'LTeamID', 'ScoreGap']].rename(columns={"ScoreGap": "GapLosses", "LTeamID": "TeamID"})


# Merge

# In[11]:


df_features_season_w = df_season_results.groupby(['Season', 'WTeamID']).count().reset_index()[['Season', 'WTeamID']].rename(columns={"WTeamID": "TeamID"})
df_features_season_l = df_season_results.groupby(['Season', 'LTeamID']).count().reset_index()[['Season', 'LTeamID']].rename(columns={"LTeamID": "TeamID"})


# In[12]:


df_features_season = pd.concat([df_features_season_w, df_features_season_l], 0).drop_duplicates().sort_values(['Season', 'TeamID']).reset_index(drop=True)


# In[13]:


df_features_season = df_features_season.merge(num_win, on=['Season', 'TeamID'], how='left')
df_features_season = df_features_season.merge(num_loss, on=['Season', 'TeamID'], how='left')
df_features_season = df_features_season.merge(gap_win, on=['Season', 'TeamID'], how='left')
df_features_season = df_features_season.merge(gap_loss, on=['Season', 'TeamID'], how='left')


# In[14]:


df_features_season.fillna(0, inplace=True)  


# Compute features

# In[15]:


df_features_season['WinRatio'] = df_features_season['NumWins'] / (df_features_season['NumWins'] + df_features_season['NumLosses'])
df_features_season['GapAvg'] = (
    (df_features_season['NumWins'] * df_features_season['GapWins'] - 
    df_features_season['NumLosses'] * df_features_season['GapLosses'])
    / (df_features_season['NumWins'] + df_features_season['NumLosses'])
)


# In[16]:


df_features_season.drop(['NumWins', 'NumLosses', 'GapWins', 'GapLosses'], axis=1, inplace=True)


# ## Tourney results
# 
# > This file identifies the game-by-game NCAA® tournament results for all seasons of historical data. The data is formatted exactly like the MRegularSeasonCompactResults data. All games will show up as neutral site (so WLoc is always N). Note that this tournament game data also includes the play-in games (which always occurred on day 134/135) for those years that had play-in games. Thus each season you will see between 63 and 67 games listed, depending on how many play-in games there were.

# In[17]:


df_tourney_results = pd.read_csv(DATA_PATH + "MNCAATourneyCompactResults.csv")
df_tourney_results.drop(['NumOT', 'WLoc'], axis=1, inplace=True)


# ## FiveThirtyEight Ratings
# > Thanks to Raddar ! https://www.kaggle.com/raddar/ncaa-men-538-team-ratings

# In[18]:


df_538 = pd.read_csv("../input/ncaa-men-538-team-ratings/538ratingsMen.csv")
df_538.drop('TeamName', axis=1, inplace=True)

df_538.head()


# # Feature Engineering

# ## Train data

# In[19]:


df = df_tourney_results.copy()
df = df[df['Season'] >= 2016].reset_index(drop=True)

df.head()


# ### Seeds
# - `SeedW` is the seed of the winning team
# - `SeedL` is the seed of the losing team

# In[20]:


df = pd.merge(
    df, 
    df_seeds, 
    how='left', 
    left_on=['Season', 'WTeamID'], 
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedW'})


# In[21]:


df = pd.merge(
    df, 
    df_seeds, 
    how='left', 
    left_on=['Season', 'LTeamID'], 
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedL'})


# In[22]:


def treat_seed(seed):
    return int(re.sub("[^0-9]", "", seed))


# In[23]:


df['SeedW'] = df['SeedW'].apply(treat_seed)
df['SeedL'] = df['SeedL'].apply(treat_seed)


# In[24]:


df.head()


# ### Season Stats
# - `WinRatioW` is the win ratio of the winning team during the season
# - `WinRatioL` is the win ratio of the losing team during the season

# In[25]:


df = pd.merge(
    df,
    df_features_season,
    how='left',
    left_on=['Season', 'WTeamID'],
    right_on=['Season', 'TeamID']
).rename(columns={
    'NumWins': 'NumWinsW',
    'NumLosses': 'NumLossesW',
    'GapWins': 'GapWinsW',
    'GapLosses': 'GapLossesW',
    'WinRatio': 'WinRatioW',
    'GapAvg': 'GapAvgW',
}).drop(columns='TeamID', axis=1)


# In[26]:


df = pd.merge(
    df,
    df_features_season,
    how='left',
    left_on=['Season', 'LTeamID'],
    right_on=['Season', 'TeamID']
).rename(columns={
    'NumWins': 'NumWinsL',
    'NumLosses': 'NumLossesL',
    'GapWins': 'GapWinsL',
    'GapLosses': 'GapLossesL',
    'WinRatio': 'WinRatioL',
    'GapAvg': 'GapAvgL',
}).drop(columns='TeamID', axis=1)


# In[27]:


df.head()


# ### FiveThirtyEight Ratings
# - `538ratingW` is the rating of the winning team during the season
# - `538ratingL` is the rating of the losing team during the season

# In[28]:


df = pd.merge(
    df,
    df_538,
    how='left',
    left_on=['Season', 'WTeamID'],
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'538rating': '538ratingW'})


# In[29]:


df = pd.merge(
    df, 
    df_538, 
    how='left', 
    left_on=['Season', 'LTeamID'], 
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'538rating': '538ratingL'})


# In[30]:


df.head()


# ### Add symetrical
# - Right now our data only consists of won matches
# - We duplicate our data, get rid of the winner loser 

# In[31]:


def add_loosing_matches(df):
    win_rename = {
        "WTeamID": "TeamIdA", 
        "WScore" : "ScoreA", 
        "LTeamID" : "TeamIdB",
        "LScore": "ScoreB",
     }
    win_rename.update({c : c[:-1] + "A" for c in df.columns if c.endswith('W')})
    win_rename.update({c : c[:-1] + "B" for c in df.columns if c.endswith('L')})
    
    lose_rename = {
        "WTeamID": "TeamIdB", 
        "WScore" : "ScoreB", 
        "LTeamID" : "TeamIdA",
        "LScore": "ScoreA",
    }
    lose_rename.update({c : c[:-1] + "B" for c in df.columns if c.endswith('W')})
    lose_rename.update({c : c[:-1] + "A" for c in df.columns if c.endswith('L')})
    
    win_df = df.copy()
    lose_df = df.copy()
    
    win_df = win_df.rename(columns=win_rename)
    lose_df = lose_df.rename(columns=lose_rename)
    
    return pd.concat([win_df, lose_df], 0, sort=False)


# In[32]:


df = add_loosing_matches(df)


# In[33]:


df.head()


# ### Differences
# - We compute the difference between the team for each feature.
# - This helps further assessing how better (or worse) team A is from team B

# In[34]:


cols_to_diff = [
    'Seed', 'WinRatio', 'GapAvg', '538rating'
]

for col in cols_to_diff:
    df[col + 'Diff'] = df[col + 'A'] - df[col + 'B']


# ## Test Data

# ### Preparing

# In[35]:


df_test = pd.read_csv(DATA_PATH + "MSampleSubmissionStage1.csv")


# In[36]:


df_test['Season'] = df_test['ID'].apply(lambda x: int(x.split('_')[0]))
df_test['TeamIdA'] = df_test['ID'].apply(lambda x: int(x.split('_')[1]))
df_test['TeamIdB'] = df_test['ID'].apply(lambda x: int(x.split('_')[2]))


# In[37]:


df_test.head()


# ### Seeds

# In[38]:


df_test = pd.merge(
    df_test,
    df_seeds,
    how='left',
    left_on=['Season', 'TeamIdA'],
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedA'})


# In[39]:


df_test = pd.merge(
    df_test, 
    df_seeds, 
    how='left', 
    left_on=['Season', 'TeamIdB'], 
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedB'})


# In[40]:


df_test['SeedA'] = df_test['SeedA'].apply(treat_seed)
df_test['SeedB'] = df_test['SeedB'].apply(treat_seed)


# ### Season Stats

# In[41]:


df_test = pd.merge(
    df_test,
    df_features_season,
    how='left',
    left_on=['Season', 'TeamIdA'],
    right_on=['Season', 'TeamID']
).rename(columns={
    'NumWins': 'NumWinsA',
    'NumLosses': 'NumLossesA',
    'GapWins': 'GapWinsA',
    'GapLosses': 'GapLossesA',
    'WinRatio': 'WinRatioA',
    'GapAvg': 'GapAvgA',
}).drop(columns='TeamID', axis=1)


# In[42]:


df_test = pd.merge(
    df_test,
    df_features_season,
    how='left',
    left_on=['Season', 'TeamIdB'],
    right_on=['Season', 'TeamID']
).rename(columns={
    'NumWins': 'NumWinsB',
    'NumLosses': 'NumLossesB',
    'GapWins': 'GapWinsB',
    'GapLosses': 'GapLossesB',
    'WinRatio': 'WinRatioB',
    'GapAvg': 'GapAvgB',
}).drop(columns='TeamID', axis=1)


# ### Ratings

# In[43]:


df_test = pd.merge(
    df_test,
    df_538,
    how='left',
    left_on=['Season', 'TeamIdA'],
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'538rating': '538ratingA'})


# In[44]:


df_test = pd.merge(
    df_test,
    df_538,
    how='left',
    left_on=['Season', 'TeamIdB'],
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'538rating': '538ratingB'})


# ### Differences

# In[45]:


for col in cols_to_diff:
    df_test[col + 'Diff'] = df_test[col + 'A'] - df_test[col + 'B']


# In[46]:


df_test.head()


# ## Target

# In[47]:


df['ScoreDiff'] = df['ScoreA'] - df['ScoreB']
df['WinA'] = (df['ScoreDiff'] > 0).astype(int)


# # Modeling

# In[48]:


features = [
    'SeedDiff',
    '538ratingDiff',
    'WinRatioDiff', 
    'GapAvgDiff', 
]


# In[49]:


def rescale(features, df_train, df_val, df_test=None):
    min_ = df_train[features].min()
    max_ = df_train[features].max()
    
    df_train[features] = (df_train[features] - min_) / (max_ - min_)
    df_val[features] = (df_val[features] - min_) / (max_ - min_)
    
    if df_test is not None:
        df_test[features] = (df_test[features] - min_) / (max_ - min_)
        
    return df_train, df_val, df_test


# ### Cross Validation
# - Validate on season `n`, for `n` in the 3 last seasons. 
# - Train on earlier seasons. First available season is 2016 because of the FiveThirtyEight ratings.
# - Pipeline support classification (predict the team that wins) and regression (predict the score gap), but classification worked better on CV.

# In[50]:


def kfold(df, df_test_=None, plot=False, verbose=0, mode="reg"):
    seasons = df['Season'].unique()
    cvs = []
    pred_tests = []
    target = "ScoreDiff" if mode == "reg" else "WinA"
    
    for season in seasons[1:]:
        if verbose:
            print(f'\nValidating on season {season}')
        
        df_train = df[df['Season'] < season].reset_index(drop=True).copy()
        df_val = df[df['Season'] == season].reset_index(drop=True).copy()
        df_test = df_test_.copy()
        
        df_train, df_val, df_test = rescale(features, df_train, df_val, df_test)
        
        if mode == "reg":
            model = ElasticNet(alpha=1, l1_ratio=0.5)
        else:
            model = LogisticRegression(C=100)

        model.fit(df_train[features], df_train[target])
        
        if mode == "reg":
            pred = model.predict(df_val[features])
            pred = (pred - pred.min()) / (pred.max() - pred.min())
        else:
            pred = model.predict_proba(df_val[features])[:, 1]
        
        if df_test is not None:
            if mode == "reg":
                pred_test = model.predict(df_test[features])
                pred_test = (pred_test - pred_test.min()) / (pred_test.max() - pred_test.min())
            else:
                pred_test = model.predict_proba(df_test[features])[:, 1]
                
            pred_tests.append(pred_test)
            
        if plot:
            plt.figure(figsize=(15, 6))
            plt.subplot(1, 2, 1)
            plt.scatter(pred, df_val['ScoreDiff'].values, s=5)
            plt.grid(True)
            plt.subplot(1, 2, 2)
            sns.histplot(pred)
            plt.show()
        
        loss = log_loss(df_val['WinA'].values, pred)
        cvs.append(loss)

        if verbose:
            print(f'\t -> Scored {loss:.3f}')
        
    print(f'\n Local CV is {np.mean(cvs):.3f}')
    
    return pred_tests


# In[51]:


pred_tests = kfold(df, df_test, plot=False, verbose=1, mode="cls")


# In[52]:


pred_test = np.mean(pred_tests, 0)

_ = sns.displot(pred_test)


# # Overriding predictions
# - Without overridees, the 4 features model achieved 22nd, which is good but enough !

# ## Prepare

# In[53]:


sub = df_test[['ID', 'Season', 'Pred', 'TeamIdA', 'TeamIdB', 'SeedA', 'SeedB']].copy()
sub['Pred'] = pred_test


# In[54]:


df_teams = pd.read_csv(DATA_PATH + "MTeams.csv")
sub = sub.merge(df_teams, left_on="TeamIdA", right_on="TeamID").drop('TeamID', axis=1).rename(columns={"TeamName": "TeamA"})
sub = sub.merge(df_teams, left_on="TeamIdB", right_on="TeamID").drop('TeamID', axis=1).rename(columns={"TeamName": "TeamB"})


# In[55]:


df_seeds['Seed'] = df_seeds['Seed'].apply(lambda x:x[0])

sub = sub.merge(df_seeds, left_on=["TeamIdA", "Season"], right_on=["TeamID", "Season"]).drop('TeamID', axis=1).rename(columns={"Seed": "RegionA"})
sub = sub.merge(df_seeds, left_on=["TeamIdB", "Season"], right_on=["TeamID", "Season"]).drop('TeamID', axis=1).rename(columns={"Seed": "RegionB"})


# ## Strategies
# 
# #### Risky strategy
# 
# - Picked 11 teams that would win their first match
# - Stanford and Baylor beat every team seeded 3 or higher
# - Connecticut and South Carolina  beat every team seeded 4 or higher
# - Maryland wins beats every team seeded 7 or higher
# - use p=0.99999 for overriding
# 
# #### Safe strategy
# 
# - Picked 7 teams that would win their first match
# - Stanford, Connecticut and South Carolina beat every team seeded 6 or higher
# - Baylor wins beats every team seeded 7 or higher
# - use p=0.99 for overriding
# 
# *This parts needs to be updated using this year's best teams. I'm waiting to get analysts' insights for this.*
# Also, this was for the women teams so don't be this aggressive with the men competition as there usually are more upsets.

# In[56]:


best_teams = ['Stanford', 'South Carolina', 'Connecticut', 'Baylor', 'Maryland']  # considered for buff

strong_teams_safe = best_teams + ['NC State', 'Louisville']  # win 1st round
strong_teams_risky = strong_teams_safe + ['Texas A&M', 'Arizona', 'Georgia', 'UCLA']  # win 1st round


# In[57]:


def overwrite_pred_risky(sub, eps=1e-5):

    new_sub = []
    
    for i, row in sub.iterrows():
        
        # Buff Stanford
        if row['TeamA'] == 'Stanford' and row['SeedB'] >= 3:
            row['Pred'] = 1 - eps
        elif row['TeamB'] == 'Stanford' and row['SeedA'] >= 3:
            row['Pred'] = eps
    
        # Buff South Carolina
        if row['TeamA'] == 'South Carolina' and row['SeedB'] >= 4:
            row['Pred'] = 1 - eps
        elif row['TeamB'] == 'South Carolina' and row['SeedA'] >= 4:
            row['Pred'] = eps
            
        # Buff Connecticut
        if row['TeamA'] == 'Connecticut' and row['SeedB'] >= 4:
            row['Pred'] = 1 - eps
        elif row['TeamB'] == 'Connecticut' and row['SeedA'] >= 4:
            row['Pred'] = eps
            
        # Buff Baylor
        if row['TeamA'] == 'Baylor' and row['SeedB'] >= 3:
            row['Pred'] = 1 - eps
        elif row['TeamB'] == 'Baylor' and row['SeedA'] >= 3:
            row['Pred'] = eps
            
        # Buff Maryland
        if row['TeamA'] == 'Maryland' and row['SeedB'] >= 7:
            row['Pred'] = 1 - eps
        elif row['TeamB'] == 'Maryland' and row['SeedA'] >= 7:
            row['Pred'] = eps
        
        # Strong teams (risky) win their first round
        if row['TeamA'] in strong_teams_risky and row['SeedB'] >= 13:
            row['Pred'] = 1 - eps
        elif row['TeamB'] in strong_teams_risky and row['SeedA'] >= 13:
            row['Pred'] = eps

        new_sub.append(row)
        
    return pd.DataFrame(np.array(new_sub), columns=sub.columns)


# In[58]:


def overwrite_pred_safe(sub, eps=1e-2):    
    new_sub = []
    
    for i, row in sub.iterrows():
        row['Pred'] = np.clip(row['Pred'], 0.1, 0.9) # clip for safety
        
        # Buff Stanford
        if row['TeamA'] == 'Stanford' and row['SeedB'] >= 6:
            row['Pred'] = 1 - eps
        elif row['TeamB'] == 'Stanford' and row['SeedA'] >= 6:
            row['Pred'] = eps
    
        # Buff South Carolina
        if row['TeamA'] == 'South Carolina' and row['SeedB'] >= 6:
            row['Pred'] = 1 - eps
        elif row['TeamB'] == 'South Carolina' and row['SeedA'] >= 6:
            row['Pred'] = eps
            
        # Buff Connecticut
        if row['TeamA'] == 'Connecticut' and row['SeedB'] >= 6:
            row['Pred'] = 1 - eps
        elif row['TeamB'] == 'Connecticut' and row['SeedA'] >= 6:
            row['Pred'] = eps

        # Buff Baylor
        if row['TeamA'] == 'Baylor' and row['SeedB'] >= 7:
            row['Pred'] = 1 - eps
        elif row['TeamB'] == 'Baylor' and row['SeedA'] >= 7:
            row['Pred'] = eps
        
        # Strong teams (safe) win their first rounds
        if row['TeamA'] in strong_teams_safe and row['SeedB'] >= 13:
            row['Pred'] = 1 - eps
        elif row['TeamB'] in strong_teams_safe and row['SeedA'] >= 13:
            row['Pred'] = eps

        new_sub.append(row)
        
    return pd.DataFrame(np.array(new_sub), columns=sub.columns)


# In[59]:


# sub_pp = overwrite_pred_safe(sub)
# sub_pp = overwrite_pred_risky(sub)

sub_pp = sub


# ## Submission

# In[60]:


final_sub = sub_pp[['ID', 'Pred']].copy()
final_sub.to_csv('submission.csv', index=False)


# In[61]:


final_sub.head()

