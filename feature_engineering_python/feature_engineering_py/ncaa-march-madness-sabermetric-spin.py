#!/usr/bin/env python
# coding: utf-8

# # <center><b>NCAA March Madness 2022 Notebook</b></center>

# Special thanks to the following notebook:
# 
# https://www.kaggle.com/theoviel/ncaa-starter-the-simpler-the-better

# Table of Contents:
#     
#     1.) Loading Necessities
#         a.) Libraries
#         b.) Dataset
#     2.) Exploratory Data Analysis
#         a.) Seeds
#         b.) Season Results
#         c.) Tourney Results
#         d.) Team Statistics
#         e.) Computer Ratings
#     3.) Feature Engineering
#         a.) Seeds
#         b.) Team Statistics
#         c.) Computer Ratings
#         d.) Sabermetrics
#     4.) Create Train and Test Sets
#         a.) Add Symmetrical
#         b.) Scoring Margins
#         c.) Target Variable
#     5.) Modeling and Verifying Model
#         a.) Scale Variables
#         b.) Logistic Regression
#         c.) XGBoost
#         d.) LightGBM
#         e.) XGBoostRF
#         f.) Combined Model
#     6.) Creating Submission
#         a.) Predicting on Test Set
#         b.) Preparing for Automatic Submission

# # <h2>1.) Loading Necessities</h2>

# # <h3>1A.) Load Libraries</h3>

# In[1]:


import os
import re
import sklearn
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
from collections import Counter
from lightgbm import LGBMClassifier
from optuna.visualization import plot_slice
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.linear_model import *
from sklearn.metrics import classification_report, log_loss
from sklearn.model_selection import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew, kurtosis
from pandas import CategoricalDtype

pd.set_option('display.max_columns', None)


# # <h3>1B.) Load Dataset</h3>

# In[2]:


DATA_PATH = '../input/mens-march-mania-2022/MDataFiles_Stage2/'
# DATA_PATH = '../input/ncaaw-march-mania-2022/'
# DATA_PATH_M = '../input/ncaam-march-mania-2022/'

for filename in os.listdir(DATA_PATH):
    print(filename)


# In[3]:


df_computer_ranks = pd.read_csv(DATA_PATH + "MMasseyOrdinals_thruDay128.csv")


# # <h2>2.) Exploratory Data Analysis</h2>

# # <h3>2A.) Seeds</h3>

# The Tournament Selection Committee seeds every team in the NCAA Tournament ranging from 1 (the best teams) to 16 (the worst ones). The tournament started with 8 teams in 1939 and has since expanded to 68 teams today (4 play-in games). On Selection Sunday (March 13, 2022), the committee releases the bracket with the seeds.
# 
# * First character : Region (W, X, Y, or Z)
# * Next two digits : Seed within the region (01 to 16)
# * Last character (optional): Distinguishes teams between play-ins ( a or b)

# In[4]:


df_seeds = pd.read_csv(DATA_PATH + 'MNCAATourneySeeds.csv')
df_seeds.head()


# # <h3>2B.) Season Results</h3>

# In[5]:


df_season_results = pd.read_csv(DATA_PATH + "MRegularSeasonDetailedResults.csv")
# df_season_results = pd.read_csv(DATA_PATH + "WRegularSeasonCompactResults.csv")
df_season_results.drop(['NumOT', 'WLoc'], axis=1, inplace=True)


# In[6]:


# df_tourney_results = pd.read_csv(DATA_PATH + 'MNCAATourneyDetailedResults.csv')
# df_tourney_results.drop(['NumOT', 'WLoc'], axis=1, inplace=True)


# In[7]:


def concat_row(r):
    if r['WTeamID'] < r['LTeamID']:
        res = str(r['Season'])+"_"+str(r['WTeamID'])+"_"+str(r['LTeamID'])
    else:
        res = str(r['Season'])+"_"+str(r['LTeamID'])+"_"+str(r['WTeamID'])
    return res

# Delete leaked from train
def delete_leaked_from_df_train(df_train, df_test):
    df_train['Concats'] = df_train.apply(concat_row, axis=1)
    df_train_duplicates = df_train[df_train['Concats'].isin(df_test['ID'].unique())]
    df_train_idx = df_train_duplicates.index.values
    df_train = df_train.drop(df_train_idx)
    df_train = df_train.drop('Concats', axis=1)

    return df_train 

def read_data(inFile, sep=','):
    df_op = pd.read_csv(filepath_or_buffer=inFile, low_memory=False, encoding='utf-8', sep=sep)
    return df_op

PATH = "../input/mens-march-mania-2022/MDataFiles_Stage1/"
df_test = read_data(PATH+"MSampleSubmissionStage1.csv")
df_train = read_data(PATH+"MNCAATourneyCompactResults.csv")


print("SIZE TRAIN BEFORE :")
print(df_train.shape)
df_train = delete_leaked_from_df_train(df_train, df_test)

print("SIZE TRAIN AFTER :")
print(df_train.shape)


# In[8]:


df_season_results['ScoreMargin'] = df_season_results['WScore'] - df_season_results['LScore']


# In[9]:


sabermetrics = pd.DataFrame()

sabermetrics['Season'] = df_season_results['Season']
sabermetrics['WTeamID'] = df_season_results['WTeamID']
sabermetrics['LTeamID'] = df_season_results['LTeamID']

# Number of Possessions
sabermetrics['WPossessions'] = (df_season_results['WFGA'] - df_season_results['WOR']) + df_season_results['WTO'] + .44 * df_season_results['WFTA']
sabermetrics['LPossessions'] = (df_season_results['LFGA'] - df_season_results['LOR']) + df_season_results['LTO'] + .44 * df_season_results['LFTA']

df_season_results['WPossessions'] = sabermetrics['WPossessions']
df_season_results['LPossessions'] = sabermetrics['LPossessions']

# Points Per Possession
sabermetrics['WPtsPerPoss'] = df_season_results['WScore'] / df_season_results['WPossessions']
sabermetrics['LPtsPerPoss'] = df_season_results['LScore'] / df_season_results['LPossessions']

# Effective Field Goal Percentage
sabermetrics['WEffectiveFGPct'] = ((df_season_results['WScore'] - df_season_results['WFTM']) / 2) / df_season_results['WFGA']
sabermetrics['LEffectiveFGPct'] = ((df_season_results['LScore'] - df_season_results['LFTM']) / 2) / df_season_results['LFGA']

# Percentage of Field Goals Assisted
sabermetrics['WAssistRate'] = df_season_results['WAst'] / df_season_results['WFGM']
sabermetrics['LAssistRate'] = df_season_results['LAst'] / df_season_results['LFGM']

# Offensive Rebound Percentage
sabermetrics['WOReboundPct'] = df_season_results['WOR'] / (df_season_results['WFGA'] - df_season_results['WFGM'])
sabermetrics['LOReboundPct'] = df_season_results['LOR'] / (df_season_results['LFGA'] - df_season_results['LFGM'])

# Rebound Percentage
sabermetrics['WReboundPct'] = (df_season_results['WDR'] + df_season_results['WOR']) / (df_season_results['LFGA'] - df_season_results['LFGM'])
sabermetrics['LReboundPct'] = (df_season_results['LDR'] + df_season_results['LOR']) / (df_season_results['WFGA'] - df_season_results['WFGM'])

# Assist to Turnover Ratio
sabermetrics['WATORatio'] = df_season_results['WAst'] / df_season_results['WTO']
sabermetrics['LATORatio'] = df_season_results['LAst'] / df_season_results['LTO']

# Turnover Rate
sabermetrics['WTORate'] = df_season_results['WTO'] / df_season_results['WPossessions']
sabermetrics['LTORate'] = df_season_results['LTO'] /  df_season_results['LPossessions']

# Percentage of Shots Beyond the Arc
sabermetrics['WBArcPct'] = df_season_results['WFGA3'] / df_season_results['WFGA']
sabermetrics['LBArcPct'] = df_season_results['LFGA3'] /  df_season_results['LFGA']

# Free Throw Rate
sabermetrics['WFTRate'] = df_season_results['WFTA'] / df_season_results['WFGA']
sabermetrics['LFTRate'] = df_season_results['LFTA'] /  df_season_results['LFGA']


# In[10]:


winning_columns = sabermetrics[[col for col in sabermetrics.columns if col[0] == 'W']]
losing_columns = sabermetrics[[col for col in sabermetrics.columns if col[0] == 'L']]
winning_columns.loc[:, 'Season'] = sabermetrics['Season']
losing_columns.loc[:, 'Season'] = sabermetrics['Season']


# In[11]:


winning_columns.groupby(['Season', 'WTeamID']).mean().head()


# In[12]:


losing_columns.groupby(['Season', 'LTeamID']).mean().head()


# In[13]:


sabermetrics_season_w = winning_columns.groupby(['Season', 'WTeamID']).count().reset_index()[['Season', 'WTeamID']].rename(columns={"WTeamID": "TeamID"})
sabermetrics_season_l = losing_columns.groupby(['Season', 'LTeamID']).count().reset_index()[['Season', 'LTeamID']].rename(columns={"LTeamID": "TeamID"})


# # <h3>Compute Wins, Losses, ScoreMarginWin, ScoreMarginLoss</h3>

# In[14]:


num_win = df_season_results.groupby(['Season', 'WTeamID']).count()
num_win = num_win.reset_index()[['Season', 'WTeamID', 'DayNum']].rename(columns={"DayNum": "NumWins", "WTeamID": "TeamID"}).fillna(0)


# In[15]:


num_win.isna().sum()


# In[16]:


num_loss = df_season_results.groupby(['Season', 'LTeamID']).count()
num_loss = num_loss.reset_index()[['Season', 'LTeamID', 'DayNum']].rename(columns={"DayNum": "NumLosses", "LTeamID": "TeamID"}).fillna(0)


# In[17]:


win_score_margin = df_season_results.groupby(['Season', 'WTeamID']).mean().reset_index()
win_score_margin = win_score_margin[['Season', 'WTeamID', 'ScoreMargin']].rename(columns={"ScoreMargin": "AvgWinningScoreMargin", "WTeamID": "TeamID"}).fillna(0)


# In[18]:


lose_score_margin = df_season_results.groupby(['Season', 'LTeamID']).mean().reset_index()
lose_score_margin = lose_score_margin[['Season', 'LTeamID', 'ScoreMargin']].rename(columns={"ScoreMargin": "AvgLosingScoreMargin", "LTeamID": "TeamID"}).fillna(0)


# # <h3>Merge the results together</h3>

# In[19]:


df_features_season_w = df_season_results.groupby(['Season', 'WTeamID']).count().reset_index()[['Season', 'WTeamID']].rename(columns={"WTeamID": "TeamID"})
df_features_season_l = df_season_results.groupby(['Season', 'LTeamID']).count().reset_index()[['Season', 'LTeamID']].rename(columns={"LTeamID": "TeamID"})


# In[20]:


df_features_season = pd.concat([df_features_season_w, df_features_season_l], axis=0).drop_duplicates().sort_values(['Season', 'TeamID']).reset_index(drop=True)


# In[21]:


df_features_season = df_features_season.merge(num_win, on=['Season', 'TeamID'], how='left')
df_features_season = df_features_season.merge(num_loss, on=['Season', 'TeamID'], how='left')
df_features_season = df_features_season.merge(win_score_margin, on=['Season', 'TeamID'], how='left')
df_features_season = df_features_season.merge(lose_score_margin, on=['Season', 'TeamID'], how='left')


# In[22]:


df_features_season['NumWins'] = df_features_season['NumWins'].fillna(0)
df_features_season['NumLosses'] = df_features_season['NumLosses'].fillna(0)
df_features_season['AvgWinningScoreMargin'] = df_features_season['AvgWinningScoreMargin'].fillna(0)
df_features_season['AvgLosingScoreMargin'] = df_features_season['AvgLosingScoreMargin'].fillna(0)


# In[23]:


df_features_season['WinPercentage'] = df_features_season['NumWins'] / (df_features_season['NumWins'] + df_features_season['NumLosses'])
df_features_season['AvgScoringMargin'] = (
    (df_features_season['NumWins'] * df_features_season['AvgWinningScoreMargin'] - 
    df_features_season['NumLosses'] * df_features_season['AvgLosingScoreMargin'])
    / (df_features_season['NumWins'] + df_features_season['NumLosses'])
)


# In[24]:


df_features_season.drop(['AvgWinningScoreMargin', 'AvgLosingScoreMargin'], axis=1, inplace=True)


# # <h3>2C.) Tourney Results</h3>
# 

# In[25]:


df_tourney_results = pd.read_csv(DATA_PATH + "MNCAATourneyDetailedResults.csv")
df_tourney_results.drop(['NumOT', 'WLoc'], axis=1, inplace=True)


# # <h3>2D.) Computer Ratings</h3>

# In[26]:


df_massey = pd.read_csv(DATA_PATH + "MMasseyOrdinals_thruDay128.csv")
df_massey = df_massey[df_massey['RankingDayNum'] == 128].drop('RankingDayNum', axis=1).reset_index(drop=True) # use first day of the tournament


# In[27]:


df_massey = pd.read_csv(DATA_PATH + "MMasseyOrdinals_thruDay128.csv")


# In[28]:


systems = []
for year in range(2003, 2019):
    r = df_massey[df_massey['Season'] == year]
    systems.append(r['SystemName'].unique())
    
all_systems = list(set(list(np.concatenate(systems))))


# In[29]:


common_systems = []  
for system in all_systems:
    common = True
    for system_years in systems:
        if system not in system_years:
            common = False
    if common:
        common_systems.append(system)
        
common_systems


# ### 2E: Elo Ratings

# In[30]:


# Chat GPT To the Rescue: Update dictionary of Elo Scores

df_season_condensed = df_season_results[['Season', 'WTeamID', 'LTeamID']].copy()

def update_elo_scores(df, scores_dict, k=40):
    for index, row in df.iterrows():
        player1, player2 = str(row['WTeamID']), str(row['LTeamID'])
        if player1 not in scores_dict:
            scores_dict[player1] = 1000
        if player2 not in scores_dict:
            scores_dict[player2] = 1000
        score1, score2 = scores_dict[player1], scores_dict[player2]
        expected_score1 = 1 / (1 + 10 ** ((score2 - score1) / 400))
        expected_score2 = 1 / (1 + 10 ** ((score1 - score2) / 400))
        scores_dict[player1] = score1 + k * (1 - expected_score1)
        scores_dict[player2] = score2 + k * (0 - expected_score2)
    return scores_dict

all_seasons = dict()
scores_dict = dict()

for season in range(2003, 2023):
    df_season_condensed_year = df_season_condensed[df_season_condensed['Season'] == season]
    scores_dict = update_elo_scores(df_season_condensed_year, scores_dict)
    all_seasons[str(season)] = scores_dict
    scores_dict = dict() # Reset dict for next season
    


# In[31]:


elo_scores = pd.DataFrame(all_seasons)


# In[32]:


elo_scores_unstacked = elo_scores.unstack().reset_index()
elo_scores_unstacked.columns = ['Season', 'TeamID', 'Elo']


# In[33]:


elo_scores_unstacked


# # <h2>3.) Feature Engineering</h2>

# In[34]:


df = df_tourney_results.copy()
df = df[df['Season'] >= 2003].reset_index(drop=True)

df.head()


# # <h3>3A.) Seeds</h3>

# In[35]:


df = pd.merge(
    df, 
    df_seeds, 
    how='left', 
    left_on=['Season', 'WTeamID'], 
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedW'})


# In[36]:


df = pd.merge(
    df, 
    df_seeds, 
    how='left', 
    left_on=['Season', 'LTeamID'], 
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedL'})


# In[37]:


def treat_seed(seed):
    return int(re.sub("[^0-9]", "", seed))


# In[38]:


df['SeedW'] = df['SeedW'].apply(treat_seed)
df['SeedL'] = df['SeedL'].apply(treat_seed)


# # <h3>3B.) Season Stats</h3>

# In[39]:


df = pd.merge(
    df,
    df_features_season,
    how='left',
    left_on=['Season', 'WTeamID'],
    right_on=['Season', 'TeamID']
).rename(columns={
    'NumWins': 'NumWinsW',
    'NumLosses': 'NumLossesW',
    'AvgWinningScoreMargin': 'AvgWinningScoreMarginW',
    'AvgLosingScoreMargin': 'AvgLosingScoreMarginW',
    'WinPercentage': 'WinPercentageW',
    'AvgScoringMargin': 'AvgScoringMarginW',
}).drop(columns='TeamID', axis=1)

df = pd.merge(
    df,
    df_features_season,
    how='left',
    left_on=['Season', 'LTeamID'],
    right_on=['Season', 'TeamID']
).rename(columns={
    'NumWins': 'NumWinsL',
    'NumLosses': 'NumLossesL',
    'AvgWinningScoreMargin': 'AvgWinningScoreMarginL',
    'AvgLosingScoreMargin': 'AvgLosingScoreMarginL',
    'WinPercentage': 'WinPercentageL',
    'AvgScoringMargin': 'AvgScoringMarginL',
}).drop(columns='TeamID', axis=1)


# ### 3C.) Computer Ratings

# In[40]:


avg_ranking = df_massey.groupby(['Season', 'TeamID'])['OrdinalRank'].mean().reset_index()

df = pd.merge(
     df,
     avg_ranking,
     how='left',
     left_on=['Season', 'WTeamID'],
     right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'OrdinalRank': 'ComputerRankW'})

df = pd.merge(
    df, 
    avg_ranking, 
    how='left', 
    left_on=['Season', 'LTeamID'], 
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'OrdinalRank': 'ComputerRankL'})


# In[41]:


conferences = pd.read_csv(DATA_PATH + 'Conferences.csv')
team_conferences = pd.read_csv(DATA_PATH + 'MTeamConferences.csv')
avg_ranking = df_massey.groupby(['Season', 'TeamID']).mean().reset_index()


# In[42]:


conference_features = avg_ranking.merge(team_conferences, on=['Season', 'TeamID']).groupby(['Season', 'ConfAbbrev'])['OrdinalRank'].agg([np.mean])


# # <h3>3D.) Sabermetrics</h3>

# In[43]:


df_season_results.head()


# In[44]:


sabermetrics = pd.DataFrame()

sabermetrics['Season'] = df_season_results['Season']
sabermetrics['WTeamID'] = df_season_results['WTeamID']
sabermetrics['LTeamID'] = df_season_results['LTeamID']

# Number of Possessions
sabermetrics['WPossessions'] = (df_season_results['WFGA'] - df_season_results['WOR']) + df_season_results['WTO'] + .44 * df_season_results['WFTA']
sabermetrics['LPossessions'] = (df_season_results['LFGA'] - df_season_results['LOR']) + df_season_results['LTO'] + .44 * df_season_results['LFTA']

df_season_results['WPossessions'] = sabermetrics['WPossessions']
df_season_results['LPossessions'] = sabermetrics['LPossessions']

# Points Per Possession
sabermetrics['WPtsPerPoss'] = df_season_results['WScore'] / df_season_results['WPossessions']
sabermetrics['LPtsPerPoss'] = df_season_results['LScore'] / df_season_results['LPossessions']

# Effective Field Goal Percentage
sabermetrics['WEffectiveFGPct'] = ((df_season_results['WScore'] - df_season_results['WFTM']) / 2) / df_season_results['WFGA']
sabermetrics['LEffectiveFGPct'] = ((df_season_results['LScore'] - df_season_results['LFTM']) / 2) / df_season_results['LFGA']

# Percentage of Field Goals Assisted
sabermetrics['WAssistRate'] = df_season_results['WAst'] / df_season_results['WFGM']
sabermetrics['LAssistRate'] = df_season_results['LAst'] / df_season_results['LFGM']

# Rebound Percentage
sabermetrics['WReboundPct'] = (df_season_results['WOR'] + df_season_results['WDR']) / (df_season_results['WFGA'] - df_season_results['WFGM'])
sabermetrics['LReboundPct'] = (df_season_results['LOR'] + df_season_results['LDR']) / (df_season_results['LFGA'] - df_season_results['LFGM'])

# Assist to Turnover Ratio
sabermetrics['WATORatio'] = df_season_results['WAst'] / df_season_results['WTO']
sabermetrics['LATORatio'] = df_season_results['LAst'] / df_season_results['LTO']

# Turnover Rate
sabermetrics['WTORate'] = df_season_results['WTO'] / df_season_results['WPossessions']
sabermetrics['LTORate'] = df_season_results['LTO'] /  df_season_results['LPossessions']

# Percentage of Shots Beyond the Arc
sabermetrics['WBArcPct'] = df_season_results['WFGA3'] / df_season_results['WFGA']
sabermetrics['LBArcPct'] = df_season_results['LFGA3'] /  df_season_results['LFGA']

# Free Throw Rate
sabermetrics['WFTRate'] = df_season_results['WFTA'] / df_season_results['WFGA']
sabermetrics['LFTRate'] = df_season_results['LFTA'] /  df_season_results['LFGA']

# Block to Foul Percentage
sabermetrics['WBlockFoul'] = df_season_results['WBlk'] / (df_season_results['WPF'] + df_season_results['WBlk'])
sabermetrics['LBlockFoul'] = df_season_results['LBlk'] / (df_season_results['LPF'] + df_season_results['LBlk'])

# Steal to Foul Percentage
sabermetrics['WStealFoul'] = df_season_results['WStl'] / (df_season_results['WPF'] + df_season_results['WStl'])
sabermetrics['LStealFoul'] = df_season_results['LStl'] / (df_season_results['LPF'] + df_season_results['LStl'])


# In[45]:


winning_columns = sabermetrics[[col for col in sabermetrics.columns if col[0] == 'W']]
losing_columns = sabermetrics[[col for col in sabermetrics.columns if col[0] == 'L']]
winning_columns.loc[:, 'Season'] = sabermetrics['Season']
losing_columns.loc[:, 'Season'] = sabermetrics['Season']


# In[46]:


winning_sabermetrics = winning_columns.groupby(['Season', 'WTeamID']).mean()
losing_sabermetrics = losing_columns.groupby(['Season', 'LTeamID']).mean()


# In[47]:


winning_sabermetrics = winning_sabermetrics \
                        .reset_index() \
                        .merge(df_features_season[['Season', 'TeamID', 'NumWins']], left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left') \
                        .set_index(['Season', 'WTeamID']) \
                        .drop(['TeamID'], axis=1)

losing_sabermetrics = losing_sabermetrics \
                        .reset_index() \
                        .merge(df_features_season[['Season', 'TeamID', 'NumLosses']], left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left') \
                        .set_index(['Season', 'LTeamID']) \
                        .drop(['TeamID'], axis=1)


# In[48]:


weighted_sabermetrics_wins = winning_sabermetrics[[col for col in winning_sabermetrics.columns if col[0] == 'W']].multiply(winning_sabermetrics['NumWins'], axis=0)
weighted_sabermetrics_losses = losing_sabermetrics[[col for col in losing_sabermetrics.columns if col[0] == 'L']].multiply(losing_sabermetrics['NumLosses'], axis=0)

weighted_sabermetrics = pd.DataFrame()
weighted_sabermetrics['Possessions'] = (weighted_sabermetrics_wins['WPossessions'] + weighted_sabermetrics_losses['LPossessions']) /  \
                                       (winning_sabermetrics['NumWins'] + losing_sabermetrics['NumLosses'])

combined_df = winning_sabermetrics.reset_index().merge(losing_sabermetrics.reset_index(), left_on=['WTeamID', 'Season'], right_on=['LTeamID', 'Season'], how='outer')

def weighted_metric(metric, df=combined_df.set_index(['Season', 'WTeamID'], inplace=True)):
    """Computes the weighted stat from winning and losing metric"""
        
    weighted_df = ((combined_df[f'W{metric}'].mul(combined_df['NumWins']) + combined_df[f'L{metric}'].mul(combined_df['NumLosses'])) \
    / (combined_df['NumWins'] + combined_df['NumLosses']))
    return weighted_df


# In[49]:


combined_df.reset_index(inplace=True)
combined_df['WTeamID'].fillna(combined_df['LTeamID'], inplace=True)
combined_df['LTeamID'].fillna(combined_df['WTeamID'], inplace=True)
combined_df.set_index(['Season', 'WTeamID'], inplace=True)
combined_df.fillna(0, inplace=True)


# In[50]:


metrics_list = ['Possessions', 'PtsPerPoss', 'EffectiveFGPct', 'AssistRate', 'ReboundPct', 'ATORatio', 'TORate', 'BArcPct', 'FTRate', 'BlockFoul', 'StealFoul']
season_sabermetrics = pd.concat([weighted_metric(metric) for metric in metrics_list], axis=1)
season_sabermetrics.columns=metrics_list


# In[51]:


season_sabermetrics.index.columns = ['Season', 'TeamID']


# In[52]:


df = df.merge(season_sabermetrics, left_on=['Season', 'WTeamID'], right_on=['Season', 'WTeamID'], how='left', suffixes=[None, 'W'])
df = df.merge(season_sabermetrics, left_on=['Season', 'LTeamID'], right_on=['Season', 'WTeamID'], how='left', suffixes=[None, 'L'])


# ### 3E.) Elo Scores

# In[53]:


elo_scores_unstacked.head()


# In[54]:


df['Season'] = df['Season'].astype('str')
df['WTeamID'] = df['WTeamID'].astype('str')
df['LTeamID'] = df['LTeamID'].astype('str')


# In[55]:


df = df.merge(elo_scores_unstacked, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left', suffixes=[None, 'W'])
df = df.merge(elo_scores_unstacked, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left', suffixes=[None, 'L'])


# In[56]:


df.head()


# # <h2>4.) Create Train and Test Sets</h2>

# # <h3>4A.) Add Symmetrical</h3>

# ### Add symmetrical
# - Right now our data only consists of won matches
# - We duplicate our data, get rid of the winner loser 

# In[57]:


def add_losing_matches(win_df):
    win_rename = {
        "WTeamID": "TeamIDA", 
        "WScore" : "ScoreA", 
        "LTeamID" : "TeamIDB",
        "LScore": "ScoreB",
        "SeedW": "SeedA", 
        "SeedL": "SeedB",
        'WinPercentageW' : 'WinPercentageA',
        'WinPercentageL' : 'WinPercentageB',
        'AvgScoringMarginW' : 'AvgScoringMarginA',
        'AvgScoringMarginL' : 'AvgScoringMarginB',
        "ComputerRankW": "ComputerRankA",
        "ComputerRankL": "ComputerRankB",
        'EffectiveFGPct': 'EffectiveFGPctA',
         'PtsPerPoss': 'PtsPerPossA',
         'Possessions': 'PossessionsA',
         'AssistRate': 'AssistRateA',
         'ReboundPct': 'ReboundPctA',
         'ATORatio':'ATORatioA', 
         'TORate': 'TORateA',
         'BArcPct': 'BArcPctA',
         'FTRate': 'FTRateA',
         'StealFoul': 'StealFoulA',
         'BlockFoul': 'BlockFoulA',
         'PossessionsL': 'PossessionsB',
         'PtsPerPossL': 'PtsPerPossB',
         'EffectiveFGPctL': 'EffectiveFGPctB',
         'AssistRateL': 'AssistRateB',
         'ReboundPctL': 'ReboundPctB',
         'ATORatioL': 'ATORatioB',
         'TORateL': 'TORateB',
         'BArcPctL': 'BArcPctB',
         'FTRateL': 'FTRateB',
         'StealFoulL': 'StealFoulB',
         'BlockFoulL': 'BlockFoulB',
         'Elo': 'EloA',
         'EloL': 'EloB'
     }
    
    lose_rename = {
        "WTeamID": "TeamIDB", 
        "WScore" : "ScoreB", 
        "LTeamID" : "TeamIDA",
        "LScore": "ScoreA",
        "SeedW": "SeedB", 
        "SeedL": "SeedA",
        'WinPercentageW' : 'WinPercentageB',
        'WinPercentageL' : 'WinPercentageA',
        'AvgScoringMarginW' : 'AvgScoringMarginB',
        'AvgScoringMarginL' : 'AvgScoringMarginA',
        "ComputerRankW": "ComputerRankB",
        "ComputerRankL": "ComputerRankA",
        'EffectiveFGPct': 'EffectiveFGPctB',
         'PtsPerPoss': 'PtsPerPossB',
         'Possessions': 'PossessionsB',
         'AssistRate': 'AssistRateB',
         'ReboundPct': 'ReboundPctB',
         'ATORatio':'ATORatioB', 
         'TORate': 'TORateB',
         'BArcPct': 'BArcPctB',
         'FTRate': 'FTRateB',
         'StealFoul': 'StealFoulB',
         'BlockFoul': 'BlockFoulB',
         'PossessionsL': 'PossessionsA',
         'PtsPerPossL': 'PtsPerPossA',
         'EffectiveFGPctL': 'EffectiveFGPctA',
         'AssistRateL': 'AssistRateA',
         'ReboundPctL': 'ReboundPctA',
         'ATORatioL': 'ATORatioA',
         'TORateL': 'TORateA',
         'BArcPctL': 'BArcPctA',
         'FTRateL': 'FTRateA',
         'StealFoulL': 'StealFoulA',
         'BlockFoulL': 'BlockFoulA',
         'Elo': 'EloB',
         'EloL': 'EloA'
    }
    
    win_df = win_df.copy()
    lose_df = win_df.copy()
    
    win_df = win_df.rename(columns=win_rename)
    lose_df = lose_df.rename(columns=lose_rename)
    
    merged_df = pd.concat([win_df, lose_df], axis=0, join='inner')
    return merged_df


# In[58]:


df = add_losing_matches(df)


# In[59]:


df.loc[0] 


# # <h3>4E.) Metric Differences</h3>

# In[60]:


df['SeedDiff'] = df['SeedA'] - df['SeedB']
df['WinPercentageDiff'] = df['WinPercentageA'] - df['WinPercentageB']
df['AvgScoringMarginDiff'] = df['AvgScoringMarginA'] - df['AvgScoringMarginB']
df['ComputerRankDiff'] = df['ComputerRankA'] - df['ComputerRankB']
df['PossessionsDiff'] = df['PossessionsA'] - df['PossessionsB']
df['PtsPerPossDiff'] = df['PtsPerPossA'] - df['PtsPerPossB']
df['EffectiveFGPctDiff'] = df['EffectiveFGPctA'] - df['EffectiveFGPctB'] 
df['AssistRateDiff'] = df['AssistRateA'] - df['AssistRateB']
df['ReboundPctDiff'] = df['ReboundPctA'] - df['ReboundPctB'] 
df['TORateDiff'] = df['TORateA'] - df['TORateB'] 
df['BArcPctDiff'] = df['BArcPctA'] - df['BArcPctB']
df['FTRateDiff'] = df['FTRateA'] - df['FTRateB']
df['BlockFoulDiff'] = df['BlockFoulA'] - df['BlockFoulB']
df['StealFoulDiff'] = df['StealFoulA'] - df['StealFoulB']
df['EloDiff'] = df['EloA'] - df['EloB']


# In[61]:


conferences = pd.read_csv(DATA_PATH + 'Conferences.csv')
team_conferences = pd.read_csv(DATA_PATH + 'MTeamConferences.csv')


# In[62]:


team_conferences.merge(conferences, on='ConfAbbrev', how='left').head()


# # <h3>4F.) Test Data</h3>

# In[63]:


df_test = pd.read_csv(DATA_PATH + "MSampleSubmissionStage2.csv")
df_test['Season'] = df_test['ID'].apply(lambda x: int(x.split('_')[0]))
df_test['TeamIDA'] = df_test['ID'].apply(lambda x: int(x.split('_')[1]))
df_test['TeamIDB'] = df_test['ID'].apply(lambda x: int(x.split('_')[2]))


# <h3>Seeds</h3>

# In[64]:


df_test = pd.merge(
    df_test,
    df_seeds,
    how='left',
    left_on=['Season', 'TeamIDA'],
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedA'})

df_test = pd.merge(
    df_test, 
    df_seeds, 
    how='left', 
    left_on=['Season', 'TeamIDB'], 
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedB'})

df_test['SeedA'] = df_test['SeedA'].apply(treat_seed)
df_test['SeedB'] = df_test['SeedB'].apply(treat_seed)

# df_test['SeedA'] = df_test['SeedA'].astype(pd.CategoricalDtype(categories=[16 - x for x in range(16)], ordered=True))
# df_test['SeedB'] = df_test['SeedB'].astype(pd.CategoricalDtype(categories=[16 - x for x in range(16)], ordered=True))


# <h3>Season Stats</h3>

# In[65]:


df_test = pd.merge(
    df_test,
    df_features_season,
    how='left',
    left_on=['Season', 'TeamIDA'],
    right_on=['Season', 'TeamID']
).rename(columns={
    'NumWins': 'NumWinsA',
    'NumLosses': 'NumLossesA',
    'AvgWinningScoreMargin': 'AvgWinningScoreMarginA',
    'AvgLosingScoreMargin': 'AvgLosingScoreMarginA',
    'WinPercentage': 'WinPercentageA',
    'AvgScoringMargin': 'AvgScoringMarginA',
    'EffectiveFGPct': 'EffectiveFGPctA',
    'PtsPerPoss': 'PtsPerPossA',
    'Possessions': 'PossessionsA',
    'AssistRate': 'AssistRateA',
    'ReboundPct': 'ReboundPctA',
    'ATORatio':'ATORatioA', 
    'TORate': 'TORateA',
    'BArcPct': 'BArcPctA',
    'FTRate': 'FTRateA',
    'StealFoul': 'StealFoulA',
    'BlockFoul': 'BlockFoulA'
}).drop(columns='TeamID', axis=1)

df_test = pd.merge(
    df_test,
    df_features_season,
    how='left',
    left_on=['Season', 'TeamIDB'],
    right_on=['Season', 'TeamID']
).rename(columns={
    'NumWins': 'NumWinsB',
    'NumLosses': 'NumLossesB',
    'AvgWinningScoreMargin': 'AvgWinningScoreMarginB',
    'AvgLosingScoreMargin': 'AvgLosingScoreMarginB',
    'WinPercentage': 'WinPercentageB',
    'AvgScoringMargin': 'AvgScoringMarginB',
    'PossessionsL': 'PossessionsB',
    'PtsPerPossL': 'PtsPerPossB',
    'EffectiveFGPctL': 'EffectiveFGPctB',
    'AssistRateL': 'AssistRateB',
    'ReboundPctL': 'ReboundPctB',
    'ATORatioL': 'ATORatioB',
    'TORateL': 'TORateB',
    'BArcPctL': 'BArcPctB',
    'FTRateL': 'FTRateB',
    'StealFoulL': 'StealFoulB',
    'BlockFoulL': 'BlockFoulB'
    
}).drop(columns='TeamID', axis=1)


# # <h3>Computer Ratings</h3>

# In[66]:


df_test = pd.merge(
    df_test,
    avg_ranking,
    how='left',
    left_on=['Season', 'TeamIDA'],
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'OrdinalRank': 'ComputerRankA'})

df_test = pd.merge(
    df_test,
    avg_ranking,
    how='left',
    left_on=['Season', 'TeamIDB'],
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'OrdinalRank': 'ComputerRankB'})



# # <h3>Differences</h3>

# In[67]:


df_test.head()


# In[68]:


season_sabermetrics.head()


# # <h3>Sabermetrics</h3>

# In[69]:


df_test = df_test.merge(season_sabermetrics, left_on=['Season', 'TeamIDA'], right_on=['Season', 'WTeamID'], how='left', suffixes=[None, 'A'])
df_test = df_test.merge(season_sabermetrics, left_on=['Season', 'TeamIDB'], right_on=['Season', 'WTeamID'], how='left', suffixes=[None, 'B'])


# In[70]:


df_test.head()


# In[71]:


df_test['Season'] = df_test['Season'].astype('str')
df_test['TeamIDA'] = df_test['TeamIDA'].astype('str')
df_test['TeamIDB'] = df_test['TeamIDB'].astype('str')


# In[72]:


df_test = df_test.merge(elo_scores_unstacked, left_on=['Season', 'TeamIDA'], right_on=['Season', 'TeamID'], how='left', suffixes=[None, 'A'])
df_test = df_test.merge(elo_scores_unstacked, left_on=['Season', 'TeamIDB'], right_on=['Season', 'TeamID'], how='left', suffixes=[None, 'B'])


# In[73]:


df_test.head()


# In[74]:


df_test = df_test.rename(columns={'EffectiveFGPct': 'EffectiveFGPctA',
                                 'PtsPerPoss': 'PtsPerPossA',
                                 'Possessions': 'PossessionsA',
                                 'AssistRate': 'AssistRateA',
                                 'ReboundPct': 'ReboundPctA',
                                 'ATORatio':'ATORatioA', 
                                 'TORate': 'TORateA',
                                 'BArcPct': 'BArcPctA',
                                 'FTRate': 'FTRateA',
                                 'StealFoul': 'StealFoulA',
                                 'BlockFoul': 'BlockFoulA',
                                 'Elo': 'EloA',
                                 'PossessionsL': 'PossessionsB',
                                 'PtsPerPossL': 'PtsPerPossB',
                                 'EffectiveFGPctL': 'EffectiveFGPctB',
                                 'AssistRateL': 'AssistRateB',
                                 'ReboundPctL': 'ReboundPctB',
                                 'ATORateL': 'ATORatioB',
                                 'TORateL': 'TORateB',
                                 'BArcPctL': 'BArcPctB',
                                 'FTRateL': 'FTRateB',
                                 'StealFoulL': 'StealFoulB',
                                 'BlockFoulL': 'BlockFoulB'})


# In[75]:


df_test.head()


# In[76]:


df_test['SeedDiff'] = df_test['SeedA'].astype(int) - df_test['SeedB'].astype(int)
df_test['WinPercentageDiff'] = df_test['WinPercentageA'] - df_test['WinPercentageB']
df_test['AvgScoringMarginDiff'] = df_test['AvgScoringMarginA'] - df_test['AvgScoringMarginB']
df_test['ComputerRankDiff'] = df_test['ComputerRankA'] - df_test['ComputerRankB'] 
df_test['PossessionsDiff'] = df_test['PossessionsA'] - df_test['PossessionsB']
df_test['PtsPerPossDiff'] = df_test['PtsPerPossA'] - df_test['PtsPerPossB']
df_test['EffectiveFGPctDiff'] = df_test['EffectiveFGPctA'] - df_test['EffectiveFGPctB']
df_test['AssistRateDiff'] = df_test['AssistRateA'] - df_test['AssistRateB']
df_test['ReboundPctDiff'] = df_test['ReboundPctA'] - df_test['ReboundPctB'] 
df_test['TORateDiff'] = df_test['TORateA'] - df_test['TORateB'] 
df_test['BArcPctDiff'] = df_test['BArcPctA'] - df_test['BArcPctB'] 
df_test['FTRateDiff'] = df_test['FTRateA'] - df_test['FTRateB'] 
df_test['BlockFoulDiff'] = df_test['BlockFoulA'] - df_test['BlockFoulB'] 
df_test['StealFoulDiff'] = df_test['StealFoulA'] - df_test['StealFoulB'] 
df_test['EloDiff'] = df_test['EloA'] - df_test['EloB']


# In[77]:


df_test.head()


# # <h3>4F.) Target Variable</h3>

# In[78]:


df['ScoreDiff'] = df['ScoreA'] - df['ScoreB']
df['WinA'] = (df['ScoreDiff'] > 0).astype(int)


# # <h2>5.) Modeling and Verifying Model</h2>

# # <h3>5A.) Scaling Variables<h3>

# In[79]:


def standard_scale(features, df_train, df_val, df_test=None):    
    mm = MinMaxScaler(feature_range=(-1, 1))
    
    transformed_train = pd.DataFrame(mm.fit_transform(df_train[features]), columns = features)
    transformed_val = pd.DataFrame(mm.transform(df_val[features]), columns = features)
    
    if df_test is not None:
        transformed_test = pd.DataFrame(mm.transform(df_test[features]), columns = features)
    
    return transformed_train, transformed_val, transformed_test


# # <h3>5B.) Modeling</h3>

# 

# In[80]:


season = 2016
df_train_2016 = df[df['Season'].astype('int') < season].copy()
df_val_2016 = df[df['Season'].astype('int') >= season].copy()  
df_test_2016 = df[df['Season'].astype('int') == season].copy() 


# In[81]:


features = [
'SeedDiff',
'WinPercentageDiff',
'AvgScoringMarginDiff',
'ComputerRankDiff',
'PossessionsDiff',
'PtsPerPossDiff',
'EffectiveFGPctDiff',
'ReboundPctDiff',
'TORateDiff',
'BArcPctDiff',
'FTRateDiff',
'BlockFoulDiff',
'StealFoulDiff',
'AssistRateDiff',
'EloDiff'
]

target = ['WinA']


# In[82]:


df_test_2016[features].describe()


# In[83]:


scale = standard_scale(features, df_train_2016, df_val_2016, df_test_2016)
X_train = scale[0]
X_validation = scale[1]
X_test = scale[2]
y_train = df_train_2016['WinA'].values
y_validation = df_val_2016['WinA'].values


# In[84]:


X_test.describe()


# In[85]:


f, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(X_train.corr(), annot=True)



# In[86]:


from sklearn.feature_selection import mutual_info_classif

def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    
mi_scores = make_mi_scores(X_train, y_train)
f, ax = plt.subplots(figsize=(10, 10))
plot_mi_scores(mi_scores[:20])


# In[87]:


features = [
'SeedDiff',
'WinPercentageDiff',
'AvgScoringMarginDiff',
'ComputerRankDiff',
'PtsPerPossDiff',
'EffectiveFGPctDiff',
'ReboundPctDiff',
'TORateDiff',
'BArcPctDiff',
'FTRateDiff',
'BlockFoulDiff',
'StealFoulDiff',
'EloDiff'
]


# In[88]:


def objective(trial):
    xgb_params = dict(
        max_depth=trial.suggest_int("max_depth", 2, 64),
        learning_rate=trial.suggest_float("learning_rate", 0, 1),
        n_estimators=trial.suggest_int("n_estimators", 30, 80),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.01, 1),
        colsample_bynode=trial.suggest_float("colsample_bynode", 0.01, 1),
        colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.01, 1),
        max_delta_step=trial.suggest_int("max_delta_step", 1, 10),
        subsample=trial.suggest_float("subsample", 0.5, 1),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1e1, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 1e1, log=True),
        gamma=trial.suggest_float("gamma", 1e-6, 1, log=True)
    )
    xg_cl = xgb.XGBClassifier(**xgb_params, objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
    xg_cl.fit(X_train, y_train)
    predictions = xg_cl.predict_proba(X_validation)[:, 1]
    score = sklearn.metrics.log_loss(y_validation, predictions)
    return score

study = optuna.create_study()
study.optimize(objective, n_trials=100)
xgb_params = study.best_params
print(xgb_params)

plot_slice(study)


# In[89]:


def objective(trial):
    lgbm_params = {
        'reg_alpha': trial.suggest_float("reg_alpha", 1e-6, 1e2, log=True),
        'reg_lambda': trial.suggest_float("reg_lambda", 1e0, 1e3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 200, 500),
        'min_child_samples': trial.suggest_int('min_child_samples', 2, 100),
        'max_depth': trial.suggest_int('max_depth', 2, 64),
        'learning_rate': trial.suggest_float("learning_rate", 0, .5),
        'colsample_bytree': trial.suggest_float('colsample_bytree', .5, 1),
        'colsample_bynode': trial.suggest_float('colsample_bynode', .5, 1),
        'n_estimators': trial.suggest_int('n_estimators', 10, 80),
    }
    
    lgbm_cl = LGBMClassifier(**lgbm_params, boosting_type='gbdt', objective='binary', random_state=42)
    lgbm_cl.fit(X_train, y_train)
    predictions = lgbm_cl.predict_proba(X_validation)[:, 1]
    score = sklearn.metrics.log_loss(y_validation, predictions)
    return score

study = optuna.create_study()
study.optimize(objective, n_trials=100)
lgbm_params = study.best_params
print(lgbm_params)
plot_slice(study)


# In[90]:


def objective(trial):
    xgb_rf_params = dict(
        max_depth=trial.suggest_int("max_depth", 2, 64),
        eta=trial.suggest_float("eta", 0, 1),
        colsample_bytree=trial.suggest_float("colsample_bytree", .01, 1),
        colsample_bynode=trial.suggest_float("colsample_bynode", .01, 1),
        colsample_bylevel=trial.suggest_float("colsample_bylevel", .01, 1),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1e1, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 1e1, log=True),
        gamma=trial.suggest_float("gamma", 1e-6, 1e-1, log=True),
        n_estimators=trial.suggest_int("n_estimators", 30, 100),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 5),
        subsample=trial.suggest_float("subsample", 0.5, 1),
    )
    xgbrf = xgb.XGBRFClassifier(**xgb_rf_params, objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
    xgbrf.fit(X_train, y_train)
    predictions = xgbrf.predict_proba(X_validation)[:, 1]
    score = sklearn.metrics.log_loss(y_validation, predictions)
    return xgbrf.score(X_validation, y_validation)

study = optuna.create_study()
study.optimize(objective, n_trials=100)
xgb_rf_params = study.best_params
plot_slice(study)


# In[91]:


model1 = xgb.XGBClassifier(**xgb_params, objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
model2 = xgb.XGBRFClassifier(**xgb_rf_params, objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
model3 = LGBMClassifier(**lgbm_params, boosting_type='gbdt', objective='binary', random_state=42)
model = StackingClassifier(estimators=[('XGB', model1), ('LGBM', model3), ('XGBRF', model2)], 
                                   final_estimator=LogisticRegression(), passthrough=True, cv=5)
model.fit(X_train, y_train)
print(model.score(X_validation, y_validation))


# In[92]:


print(xgb_params)
print(lgbm_params)
print(xgb_rf_params)


# <h3>5C.) Cross Validation</h3>

# In[93]:


def kfold_reg(df, df_test_=None, plot=False, verbose=0, mode="reg"):
    seasons = df['Season'].unique()
    cvs = []
    pred_tests = []
    target = "ScoreDiff" if mode == "reg" else "WinA"
    
    
    for season in seasons[13:]:
        if verbose:
            print(f'\nValidating on season {season}')
        
        df_train = df[df['Season'] < season].copy()
        df_val = df[df['Season'] == season].copy()
        df_test = df_test_.copy()
        
        X_train, X_val, X_test = standard_scale(features, df_train, df_val, df_test)
        y_train, y_val = df_train[target], df_val[target]
        
        if mode == "reg":
            model = ElasticNet(alpha=1, l1_ratio=0.5)
            
        else:
            model1 = xgb.XGBClassifier(**xgb_params, objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
            model2 = xgb.XGBRFClassifier(**xgb_rf_params, objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
            model3 = LGBMClassifier(**lgbm_params, boosting_type='gbdt', objective='binary', random_state=42)
            model4 = ExtraTreesClassifier(max_depth=10, max_features=None, class_weight='balanced')
            model5 = RandomForestClassifier(max_depth=10, max_features=None, class_weight='balanced')
            model6 = LogisticRegression(class_weight='balanced', max_iter=10000, C=.025)
            model7 = KNeighborsClassifier(n_neighbors=5)
            model8 = HistGradientBoostingClassifier(loss='binary_crossentropy', scoring='log_loss')
            model9 = SGDClassifier(loss='modified_huber', class_weight='balanced', early_stopping=True, alpha=1e-6, fit_intercept=False)
            model10 = LogisticRegression(class_weight='balanced', max_iter=10000)
            model = StackingClassifier(estimators=[('XGB', model1),
                                                   ('XGBRF', model2), 
                                                   ('LGBM', model3),
                                                   ('ET', model4),
                                                   ('RF', model5),
                                                   ('LR', model6),
                                                   ('KNN', model7),
                                                   ('HB', model8),
                                                   ('SGD', model9)
                                                  ], 
                                                   final_estimator=model10, 
                                                   passthrough=True, cv=5)
            
            
        model.fit(X_train, y_train)
        
        
        calibrated = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
        calibrated.fit(X_train, y_train)
        
        if mode == "reg":
            pred = model.predict(X_val)
        else:
            pred = calibrated.predict_proba(X_val)[:, 1]

        
        if X_test is not None:
            if mode == "reg":
                pred_test = model.predict(X_test)
            else:
                pred_test = calibrated.predict_proba(X_test)[:, 1]
            
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


# # <h2>6.) Create Submission</h2>

# # <h3>6A.) Predict on Test Set</h3>

# In[94]:


pred_tests = kfold_reg(df, df_test[features], plot=True, verbose=1, mode="cls")


# # <h3>6B.) Preparing for Automatic Submission</h3>

# In[95]:


pred_test = np.mean((pred_tests), 0)
sub = df_test[['ID', 'Pred']].copy()
sub['Pred'] = pred_test
sub.to_csv('submission.csv', index=False)


# In[96]:


sub['Pred'].mean()


# In[97]:


for col in df_test[features].columns:
    fig, ax = plt.subplots()
    sns.histplot(df_test[col], ax=ax)
    ax.set_title(f"Histogram of {col}")
    plt.show()


# In[98]:


sns.histplot(sub['Pred'])


# In[99]:


# Create human readable output for tournament brackets

human_sub = sub.copy()
human_sub['Season'] = human_sub['ID'].apply(lambda x: int(x.split('_')[0]))
human_sub['TeamIDA'] = human_sub['ID'].apply(lambda x: int(x.split('_')[1]))
human_sub['TeamIDB'] = human_sub['ID'].apply(lambda x: int(x.split('_')[2]))

team_names = 'MTeams.csv'
teams = pd.read_csv(DATA_PATH + team_names)
team_names_1 = pd.merge(human_sub, teams[['TeamID', 'TeamName']], left_on='TeamIDA', right_on='TeamID')
team_names_2 = pd.merge(team_names_1, teams[['TeamID', 'TeamName']], left_on='TeamIDB', right_on='TeamID')

# To make human predictions, export the following columns into Excel and make a pivot table
team_names_2[['Pred', 'TeamName_x', 'TeamName_y']].to_csv('Train2022.csv')

