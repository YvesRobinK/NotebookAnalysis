#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering for March Madness

# ****This kernel originally used data from [Google Cloud & NCAA® ML Competition 2018-Men's](https://www.kaggle.com/c/mens-machine-learning-competition-2018), but it might still be useful for this years' competition. 

# In[1]:


from collections import Counter
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

curr_dir = '../input/mens-machine-learning-competition-2018/' 
tourney_games = pd.read_csv(curr_dir + 'NCAATourneyCompactResults.csv')
regular_games = pd.read_csv(curr_dir + 'RegularSeasonCompactResults.csv')
test_games = pd.read_csv(curr_dir + 'SampleSubmissionStage2.csv')
seeds = pd.read_csv(curr_dir + 'NCAATourneySeeds.csv')
massey_ordinals = pd.read_csv(curr_dir + 'MasseyOrdinals_thruSeason2018_Day128.csv')


# In[2]:


test_games.head()


# In[3]:


test_games.shape 


# Each row in `test_games` corresponds to every possible match up among the 68 teams selected for the 2018 March Madness tournament.  

# In[4]:


68*67/2 # 68 choose 2


# The first column of ID consists of the season and the numerical codes for the two teams in the match-ups.

# In[5]:


def test_features(games):
    (games['Season'], games['Team1'], games['Team2']) = zip(*games.ID.apply(lambda x: tuple(map(int, x.split('_')))))
    games['DayNum'] = 134
    cols_to_keep = ['Season', 'Team1', 'Team2', 'DayNum']
    games = games[cols_to_keep]
    return games
test_games = test_features(test_games)


# The exact day of the games in the `test_games` is unknown if they take place at all. The feature `DayNum` corresponding to the exact day when the games happens is not used in the final model but is used to derive other useful features. For that purpose, it suffices to set it to 134, that is the starting day of tournaments, for all games in the `test_games`. 

# In[6]:


test_games.head()


# In[7]:


tourney_games.head()


# In[8]:


regular_games.head()


# The dataframes for both the tournament and regular games are in the exactly same format. To bring them in sync with that of the test games, they are processed as follows:
# * We generate the features Team 1 and Team 2 out of the winning and losing team in the game such that Team 1 is the numerically smaller one of two and Team 2 is larger to keep it consistent with the `test_games`.
# * Prediction is 1 if Team 1 wins else 0.
# * The score difference given by Team 1's score - Team 2's score.

# In[9]:


def basic_features(games):
    (games['Team1'], games['Team2']) = np.where(games.WTeamID < games.LTeamID, (games.WTeamID, games.LTeamID), (games.LTeamID, games.WTeamID))
    games['Prediction'] = np.where(games.WTeamID==games.Team1, 1, 0)
    games['Score_difference'] = np.where(games.WTeamID==games.Team1, games.WScore - games.LScore, games.LScore - games.WScore)
    cols = ['Season', 'Team1', 'Team2', 'DayNum', 'Score_difference', 'Prediction']
    games = games[cols]
    return games
tourney_games = basic_features(tourney_games)
regular_games = basic_features(regular_games)


# In[10]:


tourney_games.head()


# In[11]:


regular_games.head()


# In[12]:


tourney_games.shape, regular_games.shape


# In[13]:


print("Total number of games to be used in training:", tourney_games.shape[0] + regular_games.shape[0])


# Each row in the dataframe corresponding to a particular game contains score difference that was not known prior to the game.
# There are many possible features that can be engineered, but here we cover the following four main features:
# * Difference in the seed of the two competing teams
# * Difference in the ranking of the two teams just prior to the match
# * Difference in the number of games played by the two teams in the tournaments until previous season. Since it is a single elimination tournaments, this indicates how many times in the past, the teams were selected in the tournament and how far ahead they went
# * Difference in the scores in the games played by the two teams in the past is calculated using [exponentially weighted averages](https://www.coursera.org/learn/deep-neural-network/lecture/Ud7t0/understanding-exponentially-weighted-averages)
# 
# The data consists of a time series. We have to pay attention not to violate what Claudia Perlich calls the NTMC (*[No Time Machine Condition](https://medium.com/@colin.fraser/the-treachery-of-leakage-56a2d7c4e931): If X happens after Y, we should not use X to predict Y*) and only use the information that was known prior to the game while features engineering. 

# In case of large tabular datasets that needs feature engineering, the efficiency of code can be increased using a few tricks. [Here](https://engineering.upside.com/a-beginners-guide-to-optimizing-pandas-code-for-speed-c09ef2c6a4d6) is a blog that discuss the efficiency of operations on pandas dataframes. It states 
# > The efficiency of several methodologies for applying a function to a Pandas DataFrame, from slowest to fastest:
# 1. Crude looping over DataFrame rows using indices
# 2. Looping with iterrows()
# 3. Looping with apply()
# 4. Vectorization with Pandas series
# 5. Vectorization with NumPy arrays
# 
# The feature engineering for this dataset has plenty of scope to use the fundamental computer science concepts such as avoiding repeated computations and nested loops using hash tables to improve speed drastically. For information that needs to be retrieved frequently, using a python dictionary (implemented as a hash table) will take some additional space but will bring down the computation time complexity from O(n) to O(1).
# 

# In[14]:


seeds.head()


# In[15]:


seeds = seeds.set_index(['Season', 'TeamID'])
seeds = seeds['Seed'].to_dict()
type(seeds)


# The seed dataframe is converted to a python dictionary, which is then used to create the feature calculating the difference in the seed of the teams in the tournament match-ups. This feature is used only for tournament games and not regular games.

# In[16]:


def seed_features(games):
    games['Seed_diff'] = games.apply(lambda row: int(seeds[(row.Season, row.Team1)][1:3]) -
                                                int(seeds[(row.Season, row.Team2)][1:3]), axis=1)
    return games
test_games = seed_features(test_games)
tourney_games = seed_features(tourney_games)


# In[17]:


tourney_games.head()


# In[18]:


massey_ordinals.head()


# There are multiple rankings corresponding to a Team on a particular day of a season.

# In[19]:


massey_ordinals[(massey_ordinals.TeamID ==1101) & (massey_ordinals.Season == 2014) & (massey_ordinals.RankingDayNum ==9)] 


# The ordinals are grouped by (Team, Season, Day) and their median is taken as the ranking.

# In[20]:


massey_ordinals = massey_ordinals.groupby(['TeamID', 'Season', 'RankingDayNum']).median()
massey_ordinals.head()


# The rankings are stored in a dictionary for fast retrieval. If the ranking for a team on a particular day of the season is missing, we take the latest available ranking.

# In[21]:


ordinals_dict = massey_ordinals['OrdinalRank'].to_dict()

def massey_ranking_difference(Team1, Team2, Season, DayNum):
    if Season < 2003:
        return np.nan
    try:
        Ranking1 = ordinals_dict[(Team1, Season, DayNum)]
    except:
        try:
            RankingDays1 = massey_ordinals.loc[Team1, Season].index
            LatestDayTeam1 = RankingDays1[RankingDays1 <= DayNum][-1]
            Ranking1 = ordinals_dict[(Team1, Season, LatestDayTeam1)]
        except: return np.nan
    try:
        Ranking2 = ordinals_dict[(Team2, Season, DayNum)]
    except:
        try:
            RankingDays2 = massey_ordinals.loc[Team2, Season].index
            LatestDayTeam2 = RankingDays2[RankingDays2 <= DayNum][-1]
            Ranking2 = ordinals_dict[(Team2, Season, LatestDayTeam2)]
        except: return np.nan
    return Ranking1 - Ranking2

def ranking_feature(games, test=False):
    if test:
        games['Ranking_diff'] = games.apply(lambda row: 
                    massey_ranking_difference(row.Team1, row.Team2, 2018, 128), axis=1)
        
    else:
        games['Ranking_diff'] = games.apply(lambda row: 
                    massey_ranking_difference(row.Team1, row.Team2, row.Season, row.DayNum), axis=1)
    return games

tourney_games = ranking_feature(tourney_games)
regular_games = ranking_feature(regular_games)
test_games = ranking_feature(test_games, test=True)


# In[22]:


tourney_games.tail()


# Since it is a single-elimination tournament, the number of games played by a team in the past tournaments is an indication of its selection and wins in the past tournaments. We calculate the difference in the games played in previous tournament between the two competing teams and normalize it by dividing with (current season - 1984). Thus, the `Tourney_games_played_diff` is difference between the average number of games played per season by the two teams in the match-ups in the previous seasons.

# We first group the tournament games as per the seasons and then for each group/season, we count the number of games played by each qualifying team in the tournament and add it to the previous count of total number of games played by the teams. The count is stored in `tourney_games_count` which is a dictionary of season-wise dictionaries. 

# In[23]:


games = tourney_games.set_index('Season').groupby('Season')


# In[24]:


games.describe()


# In[25]:


tourney_games_count = {1985: {}} # dictionary of season-wise dictionaries

for grp in games:
    season = grp[0]+1
    df = pd.concat([grp[1].Team1.value_counts(), grp[1].Team2.value_counts()], axis=1).fillna(0)
    df['Season'] = season
    df['games_played'] = df.Team1 + df.Team2
    current_count = df['games_played'].to_dict()
    total_count = Counter(tourney_games_count[season-1]) + Counter(current_count)
    tourney_games_count[season] = total_count

def games_played_difference(Team1, Team2, season):
    games_played_Team1 = tourney_games_count[season].get(Team1, 0)
    games_played_Team2 = tourney_games_count[season].get(Team2, 0)
    return round((games_played_Team1 - games_played_Team2)/(season-1984), 2)

def games_played_feature(games):
    games['Tourney_games_played_diff'] = games.apply(lambda row: 
                            games_played_difference(row.Team1, row.Team2, row.Season), axis=1)
    games['Tournament'] = 1
    return games

tourney_games = games_played_feature(tourney_games)
test_games = games_played_feature(test_games)
regular_games['Tournament'] = 0


# To calculate the average score difference between two teams, we consider all the games that the two teams played with each other including both regular and tournament games. We use hash table to reduce the computation.

# In[26]:


all_games = pd.concat([tourney_games, regular_games]).copy()
all_games.sort_values(['Season', 'DayNum'], inplace=True)
hash_scores = {}
b = 0.8
def scores(row):
    if (row.Team1, row.Team2) in hash_scores:
        previous_average_score_difference = hash_scores[(row.Team1, row.Team2)]
        average_score_difference = b*previous_average_score_difference + (1-b)*row.Score_difference
    else: 
        previous_average_score_difference = np.nan 
        average_score_difference = row.Score_difference
    hash_scores[(row.Team1, row.Team2)] = average_score_difference
    return previous_average_score_difference


# For every row (game), the previous average score difference is used as a feature and the current score difference is used to update the average score difference in the hash table corresponding to the particular pair of teams. The average score is updated using [exponentially weighted averages](https://www.coursera.org/learn/deep-neural-network/lecture/Ud7t0/understanding-exponentially-weighted-averages).

# In[27]:


all_games['Average_score_difference'] = all_games.apply(lambda row: scores(row), axis=1)
all_games.set_index(['Team1', 'Team2', 'Season', 'DayNum'], inplace=True)
all_games.sample(10)


# In[28]:


all_games = all_games['Average_score_difference'].to_dict()
def score_difference_feature(games, test=False):  
    if test:
        games['Average_score_diff'] = games.apply(lambda row: 
                            hash_scores.get((row.Team1, row.Team2), np.nan), axis=1)
    else:
        games['Average_score_diff'] = games.apply(lambda row: 
                            all_games[(row.Team1, row.Team2, row.Season, row.DayNum)], axis=1)
    return games

tourney_games = score_difference_feature(tourney_games)
regular_games = score_difference_feature(regular_games)
test_games = score_difference_feature(test_games, test=True)


# In[29]:


tourney_games.columns


# In[30]:


def final_features(games):
    games.fillna(0, inplace=True)
    games['Team1'] = games['Team1'].astype('category', ordered=False)
    games['Team2'] = games['Team2'].astype('category', ordered=False)
    features_to_keep = ['Team1', 'Team2', 'Seed_diff', 'Average_score_diff', 'Tourney_games_played_diff', 
                        'Ranking_diff', 'Tournament'] 
    games = games[features_to_keep]
    return games


# In[31]:


train = pd.concat([tourney_games, regular_games])
prediction = train.Prediction
test = test_games

train = final_features(train)
test = final_features(test)


# In[32]:


train.iloc[2000:2010]


# In[33]:


test.head()


# ### References:
# * https://www.kaggle.com/juliaelliott/basic-starter-kernel-ncaa-men-s-dataset
# * https://www.kaggle.com/the1owl/ridge-huber-3-pointer-m

# In[34]:




