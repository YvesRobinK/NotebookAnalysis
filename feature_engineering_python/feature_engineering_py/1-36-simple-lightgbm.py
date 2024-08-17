#!/usr/bin/env python
# coding: utf-8

# Credit to @columbia2131 - I started with his notebook and then added an external data set with descriptive statistics of the targets for each player.

# ## About Dataset

# Train.csv is stored as a csv file with each column as follows.  
# train.csvを以下のようにして各カラムをcsvファイルとして保管しています。
# 
# To use many data, I used fruction of "reduce_mem_usage" to reduce CPU load.
# CPU負荷を抑えるためにreduce_mem_usageという関数を使っています。
# 
# Params are tuned by Light GBM tuner. 
# パラメータはLight GBM tunerで調整しています。
# 
# I want to continue feature engineering, because there are other features not used.
# 特徴量エンジニアリングを続けたい、まだ使っていない特徴量があるため。

# In[1]:


get_ipython().run_cell_magic('capture', '', '"""\n!pip install pandarallel \n\nimport gc\n\nimport numpy as np\nimport pandas as pd\nfrom pathlib import Path\n\nfrom pandarallel import pandarallel\npandarallel.initialize()\n\nBASE_DIR = Path(\'../input/mlb-player-digital-engagement-forecasting\')\ntrain = pd.read_csv(BASE_DIR / \'train.csv\')\n\nnull = np.nan\ntrue = True\nfalse = False\n\nfor col in train.columns:\n\n    if col == \'date\': continue\n\n    _index = train[col].notnull()\n    train.loc[_index, col] = train.loc[_index, col].parallel_apply(lambda x: eval(x))\n\n    outputs = []\n    for index, date, record in train.loc[_index, [\'date\', col]].itertuples():\n        _df = pd.DataFrame(record)\n        _df[\'index\'] = index\n        _df[\'date\'] = date\n        outputs.append(_df)\n\n    outputs = pd.concat(outputs).reset_index(drop=True)\n\n    outputs.to_csv(f\'{col}_train.csv\', index=False)\n    outputs.to_pickle(f\'{col}_train.pkl\')\n\n    del outputs\n    del train[col]\n    gc.collect()\n"""\n')


# ## Training

# In[2]:


import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from datetime import timedelta
from functools import reduce
from tqdm import tqdm
import lightgbm as lgbm
import mlb
import gc

pd.options.display.max_rows = 200
pd.options.display.max_columns = 100


# ## Fruction to reduce CPU load

# In[3]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[4]:


BASE_DIR = Path('../input/mlb-player-digital-engagement-forecasting')
TRAIN_DIR = Path('../input/mlb-pdef-train-dataset')


# ## Select columns

# In[5]:


targets_cols = [
    'playerId', 
    'target1', 
    'target2', 
    'target3', 
    'target4', 
    'date'
]

players_cols = [
    'playerId', 
    'primaryPositionName'
]

teams_cols = [
    'id', 
#     'name', 
#     'teamName', 
#     'teamCode', 
#     'shortName', 
#     'abbreviation', 
#     'locationName', 
    'leagueId', 
#     'leagueName', 
    'divisionId', 
#     'divisionName', 
#     'venueId', 
#     'venueName'
]

rosters_cols = [
    'playerId', 
    'teamId', 
    'status', 
    'date'
]

scores_cols = [
    'playerId', 
    'battingOrder', 
    'gamesPlayedBatting', 
    'flyOuts',
    'groundOuts', 
    'runsScored', 
    'doubles', 
    'triples', 
    'homeRuns',
    'strikeOuts', 
    'baseOnBalls', 
    'intentionalWalks', 
    'hits', 
    'hitByPitch',
    'atBats', 
    'caughtStealing', 
    'stolenBases', 
    'groundIntoDoublePlay',
    'groundIntoTriplePlay', 
    'plateAppearances', 
    'totalBases', 
    'rbi',
    'leftOnBase', 
    'sacBunts', 
    'sacFlies', 
    'catchersInterference',
    'pickoffs', 
    'gamesPlayedPitching', 
    'gamesStartedPitching',
    'completeGamesPitching', 
    'shutoutsPitching', 
    'winsPitching',
    'lossesPitching', 
    'flyOutsPitching', 
    'airOutsPitching',
    'groundOutsPitching', 
    'runsPitching', 
    'doublesPitching',
    'triplesPitching', 
    'homeRunsPitching', 
    'strikeOutsPitching',
    'baseOnBallsPitching', 
    'intentionalWalksPitching', 
    'hitsPitching',
    'hitByPitchPitching', 
    'atBatsPitching', 
    'caughtStealingPitching',
    'stolenBasesPitching', 
    'inningsPitched', 
    'saveOpportunities',
    'earnedRuns', 
    'battersFaced', 
    'outsPitching', 
    'pitchesThrown', 
    'balls',
    'strikes', 
    'hitBatsmen', 
    'balks', 
    'wildPitches', 
    'pickoffsPitching',
    'rbiPitching', 
    'gamesFinishedPitching', 
    'inheritedRunners',
    'inheritedRunnersScored', 
    'catchersInterferencePitching',
    'sacBuntsPitching', 
    'sacFliesPitching', 
    'saves', 
    'holds', 
    'blownSaves',
    'assists', 
    'putOuts', 
    'errors', 
    'chances', 
    'date'
]

awards_cols = [
    'date', 
    'playerId',
    'awardId'
]

playerTwitterFollowers_cols = [
    'playerId', 
    'numberOfFollowers'
]

teamTwitterFollowers_cols = [
    'teamId', 
    'numberOfFollowers'
]

standings_cols = [
    'teamId', 
#     'wildCardRank', 
    'wins', 
    'losses', 
#     'divisionChamp', 
#     'divisionLeader', 
#     'wildCardLeader', 
    'lastTenWins',
    'lastTenLosses',
    'date'
]

feature_cols = [
    'label_playerId', 
    'label_primaryPositionName', 
    'label_teamId',
    'label_status',
    'playerId', 
    'battingOrder', 
    'gamesPlayedBatting', 
    'flyOuts',
    'groundOuts', 
    'runsScored', 
    'doubles', 
    'triples', 
    'homeRuns',
    'strikeOuts', 
    'baseOnBalls', 
    'intentionalWalks', 
    'hits', 
    'hitByPitch',
    'atBats', 
    'caughtStealing', 
    'stolenBases', 
    'groundIntoDoublePlay',
    'groundIntoTriplePlay', 
    'plateAppearances', 
    'totalBases', 
    'rbi',
    'leftOnBase', 
    'sacBunts', 
    'sacFlies', 
    'catchersInterference',
    'pickoffs', 
    'gamesPlayedPitching', 
    'gamesStartedPitching',
    'completeGamesPitching', 
    'shutoutsPitching', 
    'winsPitching',
    'lossesPitching', 
    'flyOutsPitching', 
    'airOutsPitching',
    'groundOutsPitching', 
    'runsPitching', 
    'doublesPitching',
    'triplesPitching', 
    'homeRunsPitching', 
    'strikeOutsPitching',
    'baseOnBallsPitching', 
    'intentionalWalksPitching', 
    'hitsPitching',
    'hitByPitchPitching', 
    'atBatsPitching', 
    'caughtStealingPitching',
    'stolenBasesPitching', 
    'inningsPitched', 
    'saveOpportunities',
    'earnedRuns', 
    'battersFaced', 
    'outsPitching', 
    'pitchesThrown', 
    'balls',
    'strikes', 
    'hitBatsmen', 
    'balks', 
    'wildPitches', 
    'pickoffsPitching',
    'rbiPitching', 
    'gamesFinishedPitching', 
    'inheritedRunners',
    'inheritedRunnersScored', 
    'catchersInterferencePitching',
    'sacBuntsPitching', 
    'sacFliesPitching', 
    'saves', 
    'holds', 
    'blownSaves',
    'assists', 
    'putOuts', 
    'errors', 
    'chances', 
    'target1_mean',
    'target1_median',
    'target1_std',
    'target1_min',
    'target1_max',
    'target1_prob',
    'target2_mean',
    'target2_median',
    'target2_std',
    'target2_min',
    'target2_max',
    'target2_prob',
    'target3_mean',
    'target3_median',
    'target3_std',
    'target3_min',
    'target3_max',
    'target3_prob',
    'target4_mean',
    'target4_median',
    'target4_std',
    'target4_min',
    'target4_max',
    'target4_prob',
    'awardId_count',
    'playernumberOfFollowers',               
    'teamnumberOfFollowers',
    'label_leagueId',
    'label_divisionId',
    'wins', 
    'losses', 
    'lastTenWins',
    'lastTenLosses'
]


# ## Read data and groupby

# In[6]:


players = pd.read_csv(BASE_DIR / 'players.csv', usecols = players_cols)
players = reduce_mem_usage(players)


teams = pd.read_csv(BASE_DIR / 'teams.csv', usecols = teams_cols)
teams = teams.rename(columns = {'id':'teamId'})
teams = reduce_mem_usage(teams)


rosters = pd.read_csv(TRAIN_DIR / 'rosters_train.csv', usecols = rosters_cols)
rosters = reduce_mem_usage(rosters)


targets = pd.read_csv(TRAIN_DIR / 'nextDayPlayerEngagement_train.csv', usecols = targets_cols)
targets = reduce_mem_usage(targets)


scores = pd.read_csv(TRAIN_DIR / 'playerBoxScores_train.csv', usecols = scores_cols)
scores = scores.groupby(['playerId', 'date']).sum().reset_index()
scores = reduce_mem_usage(scores)


awards = pd.read_csv(TRAIN_DIR / 'awards_train.csv', usecols = awards_cols)
# awards = awards.groupby(['playerId', 'date']).count().reset_index()


awards_count = awards[['playerId', 'awardId']].groupby('playerId').count().reset_index()
awards_count = awards_count.rename(columns = {'awardId':'awardId_count'})
awards_count = reduce_mem_usage(awards_count)


playerTwitterFollowers = pd.read_csv(TRAIN_DIR / 'playerTwitterFollowers_train.csv', usecols = playerTwitterFollowers_cols)
playerTwitterFollowers = playerTwitterFollowers.groupby('playerId').sum().reset_index()
playerTwitterFollowers = playerTwitterFollowers.rename(columns = {'numberOfFollowers':'playernumberOfFollowers'})
playerTwitterFollowers = reduce_mem_usage(playerTwitterFollowers)


teamTwitterFollowers = pd.read_csv(TRAIN_DIR / 'teamTwitterFollowers_train.csv', usecols = teamTwitterFollowers_cols)
teamTwitterFollowers = teamTwitterFollowers.groupby('teamId').sum().reset_index()
teamTwitterFollowers = teamTwitterFollowers.rename(columns = {'numberOfFollowers':'teamnumberOfFollowers'})
teamTwitterFollowers = reduce_mem_usage(teamTwitterFollowers)


standings = pd.read_csv(TRAIN_DIR / 'standings_train.csv', usecols = standings_cols)
standings = reduce_mem_usage(standings)

gc.collect()


# In[7]:


player_target_stats = pd.read_csv("../input/player-target-stats/player_target_stats.csv")
data_names=player_target_stats.columns.values.tolist()
data_names


# ## Make train data

# In[8]:


# creat dataset

train = targets.copy()[targets_cols]

print(targets[targets_cols].shape)

train = train.merge(
    players, 
    on=['playerId'], 
    how='left'
)
gc.collect()

print(train.shape, 'after_players')
print('--------------------------------------')

train = train.merge(
    rosters, 
    on=['playerId', 'date'], 
    how='left'
)
gc.collect()

print(train.shape, 'after_rosters')
print('--------------------------------------')

train = train.merge(
    scores, 
    on=['playerId', 'date'], 
    how='left'
)
gc.collect()

print(train.shape, 'after_scores')
print('--------------------------------------')

train = train.merge(
    player_target_stats, 
    how='inner', 
    on= "playerId",
)
gc.collect()

print(train.shape, 'after_player_target_stats')


print('--------------------------------------')

train = train.merge(
    teams,
    on = 'teamId',
    how='left'
)
# del rosters
gc.collect()

print(train.shape, 'after_teams')
print('--------------------------------------')

train = train.merge(
    awards_count,
    on = 'playerId',
    how = 'left'
)

train['awardId_count'] = train['awardId_count'].fillna(0)

print(train.shape, 'after_awards_count')
print('--------------------------------------')

train = train.merge(
    playerTwitterFollowers, 
    how = 'left', 
    on = 'playerId'
)
gc.collect()

print(train.shape, 'after_playerTwitter')
print('--------------------------------------')


train = train.merge(
    teamTwitterFollowers, 
    how = 'left', 
    on = 'teamId'
)
gc.collect()

print(train.shape, 'after_taemTwitter')
print('--------------------------------------')

train = train.merge(
    standings, 
    how = 'left', 
    on = ['teamId', 'date']
)
gc.collect()

print(train.shape, 'after_standings')
print('--------------------------------------')


# label encoding
player2num = {c: i for i, c in enumerate(train['playerId'].unique())}
position2num = {c: i for i, c in enumerate(train['primaryPositionName'].unique())}
teamid2num = {c: i for i, c in enumerate(train['teamId'].unique())}
status2num = {c: i for i, c in enumerate(train['status'].unique())}
leagueId2num = {c: i for i, c in enumerate(train['leagueId'].unique())}
divisionId2num = {c: i for i, c in enumerate(train['divisionId'].unique())}


train['label_playerId'] = train['playerId'].map(player2num)
train['label_primaryPositionName'] = train['primaryPositionName'].map(position2num)
train['label_teamId'] = train['teamId'].map(teamid2num)
train['label_status'] = train['status'].map(status2num)
train['label_leagueId'] = train['leagueId'].map(leagueId2num)
train['label_divisionId'] = train['divisionId'].map(divisionId2num)


# In[9]:


train.info()


# In[10]:


print(train.shape)
train.isnull().sum()


# ## Divide train and valid data

# In[11]:


train_X = train[feature_cols]
train_y = train[['target1', 'target2', 'target3', 'target4']]

_index = (train['date'] < 20210401)
x_train = train_X.loc[_index].reset_index(drop=True)
y_train = train_y.loc[_index].reset_index(drop=True)
x_valid = train_X.loc[~_index].reset_index(drop=True)
y_valid = train_y.loc[~_index].reset_index(drop=True)


# In[12]:


def fit_lgbm(x_train, y_train, x_valid, y_valid, params: dict=None, verbose=100):
    oof_pred = np.zeros(len(y_valid), dtype=np.float32)
    model = lgbm.LGBMRegressor(**params)
    model.fit(x_train, y_train, 
        eval_set=[(x_valid, y_valid)],  
        early_stopping_rounds=verbose, 
        verbose=verbose)
    oof_pred = model.predict(x_valid)
    score = mean_absolute_error(oof_pred, y_valid)
    print('mae:', score)
    return oof_pred, model, score

"""
# training lightgbm before param
params = {
 'objective':'mae',
 'reg_alpha': 0.1,
 'reg_lambda': 0.1, 
 'n_estimators': 100000,
 'learning_rate': 0.1,
 'random_state': 42,
}
"""

params1 = {'objective': 'mae', 'metric': 'l1', 'feature_pre_filter': False, 'lambda_l1': 3.485822021802935e-08, 'lambda_l2': 4.230468117096112e-06, 'num_leaves': 253, 'feature_fraction': 0.8, 'bagging_fraction': 0.550250698524785, 'bagging_freq': 1, 'min_child_samples': 20, 'num_iterations': 10000, 'early_stopping_round': 100}

params2 = {'objective': 'mae', 'metric': 'l1', 'feature_pre_filter': False, 'lambda_l1': 3.731605225849285, 'lambda_l2': 0.02803980626777797, 'num_leaves': 8, 'feature_fraction': 0.5, 'bagging_fraction': 0.5262728428461787, 'bagging_freq': 3, 'min_child_samples': 20, 'num_iterations': 10000, 'early_stopping_round': 100}

params3 = {'objective': 'mae', 'metric': 'l1', 'feature_pre_filter': False, 'lambda_l1': 7.654830305013684, 'lambda_l2': 4.14748542765967e-07, 'num_leaves': 252, 'feature_fraction': 0.7200000000000001, 'bagging_fraction': 1.0, 'bagging_freq': 0, 'min_child_samples': 20, 'num_iterations': 10000, 'early_stopping_round': 100}

params4 = {'objective': 'mae', 'metric': 'l1', 'feature_pre_filter': False, 'lambda_l1': 9.486880706514734e-08, 'lambda_l2': 0.005143767850872896, 'num_leaves': 246, 'feature_fraction': 0.5479999999999999, 'bagging_fraction': 0.5238463354446826, 'bagging_freq': 5, 'min_child_samples': 20, 'num_iterations': 10000, 'early_stopping_round': 100}



oof1, model1, score1 = fit_lgbm(
    x_train, y_train['target1'],
    x_valid, y_valid['target1'],
    params1
)
oof2, model2, score2 = fit_lgbm(
    x_train, y_train['target2'],
    x_valid, y_valid['target2'],
    params2
)
oof3, model3, score3 = fit_lgbm(
    x_train, y_train['target3'],
    x_valid, y_valid['target3'],
    params3
)
oof4, model4, score4 = fit_lgbm(
    x_train, y_train['target4'],
    x_valid, y_valid['target4'],
    params4
)

score = (score1+score2+score3+score4) / 4
print(f'score: {score}')


# ## Example for tuning
import optuna.integration.lightgbm as lgbm

def fit_lgbm(x_train, y_train, x_valid, y_valid, params: dict=None, verbose=100):
        
    oof_pred = np.zeros(len(y_valid), dtype=np.float32)    
    
    trains = lgbm.Dataset(x_train, y_train)
    valids = lgbm.Dataset(x_valid, y_valid)
    
    model = lgbm.train(
        params, 
        trains,
        valid_sets = valids,
        num_boost_round = 10000,
        verbose_eval = False,
        early_stopping_rounds = 100
    )
    
    best_params = model.params
    
    oof_pred = model.predict(x_valid)
    score = mean_absolute_error(oof_pred, y_valid)
    print('mae:', score)
    return oof_pred, model, score, best_params
    
params = {
    'objective':'mae',
    'metric':'mae'
}



oof4, model4, score4, best_params4 = fit_lgbm(
        x_train, y_train['target4'],
        x_valid, y_valid['target4'],
        params
)

print(best_params4)
# ## Predict

# In[13]:


rosters_cols.remove('date')
scores_cols.remove('date')
standings_cols = [
    'teamId', 
    'wins', 
    'losses', 
    'lastTenWins',
    'lastTenLosses'
]

null = np.nan
true = True
false = False

env = mlb.make_env() # initialize the environment
iter_test = env.iter_test() # iterator which loops over each date in test set

for (test_df, sample_prediction_df) in iter_test: # make predictions here
    
    sample_prediction_df = sample_prediction_df.reset_index(drop=True)
    
    # creat dataset
    sample_prediction_df['playerId'] = sample_prediction_df['date_playerId']\
                                        .map(lambda x: int(x.split('_')[1]))
    # Dealing with missing values
    if test_df['rosters'].iloc[0] == test_df['rosters'].iloc[0]:
        test_rosters = pd.DataFrame(eval(test_df['rosters'].iloc[0]))
    else:
        test_rosters = pd.DataFrame({'playerId': sample_prediction_df['playerId']})
        for col in rosters.columns:
            if col == 'playerId': continue
            test_rosters[col] = np.nan
            
    if test_df['playerBoxScores'].iloc[0] == test_df['playerBoxScores'].iloc[0]:
        test_scores = pd.DataFrame(eval(test_df['playerBoxScores'].iloc[0]))
    else:
        test_scores = pd.DataFrame({'playerId': sample_prediction_df['playerId']})
        for col in scores.columns:
            if col == 'playerId': continue
            test_scores[col] = np.nan
            
    if test_df['standings'].iloc[0] == test_df['standings'].iloc[0]:
        test_standings = pd.DataFrame(eval(test_df['standings'].iloc[0]))
    else:
        test_standings = pd.DataFrame({'playerId': sample_prediction_df['playerId']})
        for col in standings.columns:
            if col == 'playerId': continue
            test_scores[col] = np.nan
            
            
    test_scores = test_scores.groupby('playerId').sum().reset_index()
    test = sample_prediction_df[['playerId']].copy()
    test = test.merge(players[players_cols], on='playerId', how='left')
    test = test.merge(test_rosters[rosters_cols], on='playerId', how='left')
    test = test.merge(test_scores[scores_cols], on='playerId', how='left')
    test = test.merge(player_target_stats, how='inner', left_on=["playerId"],right_on=["playerId"])
    test = test.merge(awards_count, how = 'left', on = 'playerId')
    test = test.merge(teams, how = 'left', on = 'teamId')
    test['awardId_count'] = test['awardId_count'].fillna(0)
    test = test.merge(playerTwitterFollowers, how = 'left', on ='playerId')
    test = test.merge(teamTwitterFollowers, how = 'left', on ='teamId')
    test = test.merge(test_standings[standings_cols], how = 'left', on = 'teamId')

    

    test['label_playerId'] = test['playerId'].map(player2num)
    test['label_primaryPositionName'] = test['primaryPositionName'].map(position2num)
    test['label_teamId'] = test['teamId'].map(teamid2num)
    test['label_status'] = test['status'].map(status2num)
    test['label_leagueId'] = test['leagueId'].map(leagueId2num)
    test['label_divisionId'] = test['divisionId'].map(divisionId2num)
    
    test_X = test[feature_cols]
    
    # predict
    pred1 = model1.predict(test_X)
    pred2 = model2.predict(test_X)
    pred3 = model3.predict(test_X)
    pred4 = model4.predict(test_X)
    
    # merge submission
    sample_prediction_df['target1'] = np.clip(pred1, 0, 100)
    sample_prediction_df['target2'] = np.clip(pred2, 0, 100)
    sample_prediction_df['target3'] = np.clip(pred3, 0, 100)
    sample_prediction_df['target4'] = np.clip(pred4, 0, 100)
    sample_prediction_df = sample_prediction_df.fillna(0.)
    del sample_prediction_df['playerId']
    
    env.predict(sample_prediction_df)


# In[14]:


sample_prediction_df


# In[ ]:




