#!/usr/bin/env python
# coding: utf-8

# # NCAAM Prediction
# ---
# Previous [notebook](https://www.kaggle.com/readoc/ncaam-prediction-v3).
# 
# The 2019 best solution was based on @raddar's solution
# 
# Changes: 
# 1. Removed bad bets 
# 2. Tuned some hyperparameters
# 3. Removed prediction clipping

# # If you fork, do leave an upvote!

# In[1]:


import numpy as np
import pandas as pd
import os
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from scipy.interpolate import UnivariateSpline
import statsmodels.api as sm
import matplotlib.pyplot as plt
import collections
from dataclasses import dataclass
import dataclasses


# In[2]:


@dataclass
class Config: 
    stage_2 = True # True for Stage 2 submission
    debug = False # True for fast debug run

config = Config()


# ### Load the data

# In[3]:


tourney_results = pd.read_csv('../input/ncaam-march-mania-2021/MDataFiles_Stage2/MNCAATourneyDetailedResults.csv')
seeds = pd.read_csv('../input/ncaam-march-mania-2021/MDataFiles_Stage2/MNCAATourneySeeds.csv')
regular_results = pd.read_csv('../input/ncaam-march-mania-2021/MDataFiles_Stage2/MRegularSeasonDetailedResults.csv')
kenpom = pd.read_csv('../input/kenpom-2020/Mkenpom2021.csv')


# In[4]:


all(regular_results.columns == tourney_results.columns)


# ### Preparing the data

# In[5]:


regular_results.columns


# In[6]:


regular_results_swap = regular_results[[
    'Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc', 'NumOT', 
    'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 
    'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']]


# In[7]:


regular_results_swap.head()


# In[8]:


regular_results_swap.loc[regular_results['WLoc'] == 'H', 'WLoc'] = 'A'
regular_results_swap.loc[regular_results['WLoc'] == 'A', 'WLoc'] = 'H'
regular_results.columns.values[6] = 'location'
regular_results_swap.columns.values[6] = 'location'


# In[9]:


regular_results.columns = [x.replace('W','T1_').replace('L','T2_') for x in list(regular_results.columns)]
regular_results_swap.columns = [x.replace('L','T1_').replace('W','T2_') for x in list(regular_results.columns)]


# In[10]:


regular_results.head()


# In[11]:


regular_data = pd.concat([regular_results, regular_results_swap]).sort_index().reset_index(drop = True)


# In[12]:


regular_data.head(10)


# In[13]:


tourney_results = pd.read_csv('../input/ncaam-march-mania-2021/MDataFiles_Stage2/MNCAATourneyDetailedResults.csv')
seeds = pd.read_csv('../input/ncaam-march-mania-2021/MDataFiles_Stage2/MNCAATourneySeeds.csv')
regular_results = pd.read_csv('../input/ncaam-march-mania-2021/MDataFiles_Stage2/MRegularSeasonDetailedResults.csv')

def prepare_data(df):
    dfswap = df[['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc', 'NumOT', 
    'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 
    'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']]

    dfswap.loc[df['WLoc'] == 'H', 'WLoc'] = 'A'
    dfswap.loc[df['WLoc'] == 'A', 'WLoc'] = 'H'
    df.columns.values[6] = 'location'
    dfswap.columns.values[6] = 'location'    
      
    df.columns = [x.replace('W','T1_').replace('L','T2_') for x in list(df.columns)]
    dfswap.columns = [x.replace('L','T1_').replace('W','T2_') for x in list(dfswap.columns)]

    output = pd.concat([df, dfswap]).reset_index(drop=True)
    output.loc[output.location=='N','location'] = '0'
    output.loc[output.location=='H','location'] = '1'
    output.loc[output.location=='A','location'] = '-1'
    output.location = output.location.astype(int)
    
    output['PointDiff'] = output['T1_Score'] - output['T2_Score']
    
    return output


# In[14]:


regular_data = prepare_data(regular_results)
tourney_data = prepare_data(tourney_results)


# In[15]:


regular_data.shape


# In[16]:


tourney_data.shape


# ### Feature engineering

# In[17]:


tourney_data.columns


# In[18]:


boxscore_cols = ['T1_Score', 'T2_Score', 
        'T1_FGM', 'T1_FGA', 'T1_FGM3', 'T1_FGA3', 'T1_FTM', 'T1_FTA', 'T1_OR', 'T1_DR', 'T1_Ast', 'T1_TO', 'T1_Stl', 'T1_Blk', 'T1_PF', 
        'T2_FGM', 'T2_FGA', 'T2_FGM3', 'T2_FGA3', 'T2_FTM', 'T2_FTA', 'T2_OR', 'T2_DR', 'T2_Ast', 'T2_TO', 'T2_Stl', 'T2_Blk', 'T2_PF', 
        'PointDiff']

boxscore_cols = [
        'T1_FGM', 'T1_FGA', 'T1_FGM3', 'T1_FGA3', 'T1_OR', 'T1_Ast', 'T1_TO', 'T1_Stl', 'T1_PF', 
        'T2_FGM', 'T2_FGA', 'T2_FGM3', 'T2_FGA3', 'T2_OR', 'T2_Ast', 'T2_TO', 'T2_Stl', 'T2_Blk',  
        'PointDiff']

funcs = [np.mean]


# In[19]:


season_statistics = regular_data.groupby(["Season", 'T1_TeamID'])[boxscore_cols].agg(funcs)
season_statistics.head()


# In[20]:


season_statistics = regular_data.groupby(["Season", 'T1_TeamID'])[boxscore_cols].agg(funcs).reset_index()
season_statistics.head()


# In[21]:


season_statistics.columns = [''.join(col).strip() for col in season_statistics.columns.values]
season_statistics.head()


# In[22]:


season_statistics_T1 = season_statistics.copy()
season_statistics_T2 = season_statistics.copy()

season_statistics_T1.columns = ["T1_" + x.replace("T1_","").replace("T2_","opponent_") for x in list(season_statistics_T1.columns)]
season_statistics_T2.columns = ["T2_" + x.replace("T1_","").replace("T2_","opponent_") for x in list(season_statistics_T2.columns)]
season_statistics_T1.columns.values[0] = "Season"
season_statistics_T2.columns.values[0] = "Season"


# In[23]:


season_statistics_T1.head()


# In[24]:


season_statistics_T2.head()


# In[25]:


tourney_data = tourney_data[['Season', 'DayNum', 'T1_TeamID', 'T1_Score', 'T2_TeamID' ,'T2_Score']]
tourney_data.head()


# In[26]:


tourney_data = pd.merge(tourney_data, season_statistics_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_data = pd.merge(tourney_data, season_statistics_T2, on = ['Season', 'T2_TeamID'], how = 'left')


# In[27]:


tourney_data.head()


# In[28]:


last14days_stats_T1 = regular_data.loc[regular_data.DayNum>118].reset_index(drop=True)
last14days_stats_T1['win'] = np.where(last14days_stats_T1['PointDiff']>0,1,0)
last14days_stats_T1 = last14days_stats_T1.groupby(['Season','T1_TeamID'])['win'].mean().reset_index(name='T1_win_ratio_14d')

last14days_stats_T2 = regular_data.loc[regular_data.DayNum>118].reset_index(drop=True)
last14days_stats_T2['win'] = np.where(last14days_stats_T2['PointDiff']<0,1,0)
last14days_stats_T2 = last14days_stats_T2.groupby(['Season','T2_TeamID'])['win'].mean().reset_index(name='T2_win_ratio_14d')


# In[29]:


tourney_data = pd.merge(tourney_data, last14days_stats_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_data = pd.merge(tourney_data, last14days_stats_T2, on = ['Season', 'T2_TeamID'], how = 'left')


# In[30]:


regular_season_effects = regular_data[['Season','T1_TeamID','T2_TeamID','PointDiff']].copy()
regular_season_effects['T1_TeamID'] = regular_season_effects['T1_TeamID'].astype(str)
regular_season_effects['T2_TeamID'] = regular_season_effects['T2_TeamID'].astype(str)
regular_season_effects['win'] = np.where(regular_season_effects['PointDiff']>0,1,0)
march_madness = pd.merge(seeds[['Season','TeamID']],seeds[['Season','TeamID']],on='Season')
march_madness.columns = ['Season', 'T1_TeamID', 'T2_TeamID']
march_madness.T1_TeamID = march_madness.T1_TeamID.astype(str)
march_madness.T2_TeamID = march_madness.T2_TeamID.astype(str)
regular_season_effects = pd.merge(regular_season_effects, march_madness, on = ['Season','T1_TeamID','T2_TeamID'])
regular_season_effects.shape


# In[31]:


def team_quality(season):
    formula = 'win~-1+T1_TeamID+T2_TeamID'
    glm = sm.GLM.from_formula(formula=formula, 
                              data=regular_season_effects.loc[regular_season_effects.Season==season,:], 
                              family=sm.families.Binomial()).fit()
    
    quality = pd.DataFrame(glm.params).reset_index()
    quality.columns = ['TeamID','quality']
    quality['Season'] = season
    quality['quality'] = np.exp(quality['quality'])
    quality = quality.loc[quality.TeamID.str.contains('T1_')].reset_index(drop=True)
    quality['TeamID'] = quality['TeamID'].apply(lambda x: x[10:14]).astype(int)
    return quality


# In[32]:


seeds.head()


# In[33]:


seeds['seed'] = seeds['Seed'].apply(lambda x: int(x[1:3]))
seeds.head()


# In[34]:


seeds_T1 = seeds[['Season','TeamID','seed']].copy()
seeds_T2 = seeds[['Season','TeamID','seed']].copy()
seeds_T1.columns = ['Season','T1_TeamID','T1_seed']
seeds_T2.columns = ['Season','T2_TeamID','T2_seed']


# In[35]:


tourney_data = pd.merge(tourney_data, seeds_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_data = pd.merge(tourney_data, seeds_T2, on = ['Season', 'T2_TeamID'], how = 'left')


# In[36]:


tourney_data["Seed_diff"] = tourney_data["T1_seed"] - tourney_data["T2_seed"]


# ### Model Building

# In[37]:


y = tourney_data['T1_Score'] - tourney_data['T2_Score']
y.describe()


# You can also add Kenpom data, 538 ratings data and spread data in the features

# In[38]:


features = list(season_statistics_T1.columns[2:999]) + \
    list(season_statistics_T2.columns[2:999]) + \
    list(seeds_T1.columns[2:999]) + \
    list(seeds_T2.columns[2:999]) + \
    list(last14days_stats_T1.columns[2:999]) + \
    list(last14days_stats_T2.columns[2:999]) + \
    ["Seed_diff"]

len(features)


# In[39]:


tourney_data


# In[40]:


X = tourney_data[features].values
dtrain = xgb.DMatrix(X, label = y)


# In[41]:


def cauchyobj(preds, dtrain):
    labels = dtrain.get_label()
    c = 5000 
    x =  preds-labels    
    grad = x / (x**2/c**2+1)
    hess = -c**2*(x**2-c**2)/(x**2+c**2)**2
    return grad, hess


# In[42]:


param = {} 
#param['objective'] = 'reg:linear'
param['eval_metric'] =  'mae'
param['booster'] = 'gbtree'
param['eta'] = 0.02
param['subsample'] = 0.35
param['colsample_bytree'] = 0.7
param['num_parallel_tree'] = 10
param['min_child_weight'] = 40
param['gamma'] = 10
param['max_depth'] =  3
param['silent'] = 1

print(param)


# In[43]:


xgb_cv = []
repeat_cv = 20
n_splits = 5
if config.debug: 
    repeat_cv = 2
    n_splits = 2

for i in range(repeat_cv): 
    print(f"Fold repeater {i}")
    xgb_cv.append(
        xgb.cv(
          params = param,
          dtrain = dtrain,
          obj = cauchyobj,
          num_boost_round = 3500,
          folds = KFold(n_splits = n_splits, shuffle = True, random_state = i),
          early_stopping_rounds = 25,
          verbose_eval = 50
        )
    )


# In[44]:


iteration_counts = [np.argmin(x['test-mae-mean'].values) for x in xgb_cv]
val_mae = [np.min(x['test-mae-mean'].values) for x in xgb_cv]
iteration_counts, val_mae


# In[45]:


oof_preds = []
for i in range(repeat_cv):
    print(f"Fold repeater {i}")
    preds = y.copy()
    kfold = KFold(n_splits = 5, shuffle = True, random_state = i)    
    for train_index, val_index in kfold.split(X,y):
        dtrain_i = xgb.DMatrix(X[train_index], label = y[train_index])
        dval_i = xgb.DMatrix(X[val_index], label = y[val_index])  
        model = xgb.train(
              params = param,
              dtrain = dtrain_i,
              num_boost_round = iteration_counts[i],
              verbose_eval = 50
        )
        preds[val_index] = model.predict(dval_i)
    oof_preds.append(np.clip(preds,-30,30))


# In[46]:


plot_df = pd.DataFrame({"pred":oof_preds[0], "label":np.where(y>0,1,0)})
plot_df["pred_int"] = plot_df["pred"].astype(int)
plot_df = plot_df.groupby('pred_int')['label'].mean().reset_index(name='average_win_pct')

plt.figure()
plt.plot(plot_df.pred_int,plot_df.average_win_pct)


# In[47]:


spline_model = []

for i in range(repeat_cv):
    dat = list(zip(oof_preds[i],np.where(y>0,1,0)))
    dat = sorted(dat, key = lambda x: x[0])
    datdict = {}
    for k in range(len(dat)):
        datdict[dat[k][0]]= dat[k][1]
        
    spline_model.append(UnivariateSpline(list(datdict.keys()), list(datdict.values())))
    spline_fit = spline_model[i](oof_preds[i])
    
    print(f"logloss of cvsplit {i}: {log_loss(np.where(y>0,1,0),spline_fit)}") 


# In[48]:


plot_df = pd.DataFrame({"pred":oof_preds[0], "label":np.where(y>0,1,0), "spline":spline_model[0](oof_preds[0])})
plot_df["pred_int"] = (plot_df["pred"]).astype(int)
plot_df = plot_df.groupby('pred_int')['spline','label'].mean().reset_index()

plt.figure()
plt.plot(plot_df.pred_int,plot_df.spline)
plt.plot(plot_df.pred_int,plot_df.label)


# In[49]:


spline_model = []

for i in range(repeat_cv):
    dat = list(zip(oof_preds[i],np.where(y>0,1,0)))
    dat = sorted(dat, key = lambda x: x[0])
    datdict = {}
    for k in range(len(dat)):
        datdict[dat[k][0]]= dat[k][1]
    spline_model.append(UnivariateSpline(list(datdict.keys()), list(datdict.values())))
    spline_fit = spline_model[i](oof_preds[i])
    
    print(f"adjusted logloss of cvsplit {i}: {log_loss(np.where(y>0,1,0),spline_fit)}") 


# In[50]:


spline_model = []

for i in range(repeat_cv):
    dat = list(zip(oof_preds[i],np.where(y>0,1,0)))
    dat = sorted(dat, key = lambda x: x[0])
    datdict = {}
    for k in range(len(dat)):
        datdict[dat[k][0]]= dat[k][1]
    spline_model.append(UnivariateSpline(list(datdict.keys()), list(datdict.values())))
    spline_fit = spline_model[i](oof_preds[i])

    
    print(f"adjusted logloss of cvsplit {i}: {log_loss(np.where(y>0,1,0),spline_fit)}") 


# In[51]:


spline_model = []

for i in range(repeat_cv):
    dat = list(zip(oof_preds[i],np.where(y>0,1,0)))
    dat = sorted(dat, key = lambda x: x[0])
    datdict = {}
    for k in range(len(dat)):
        datdict[dat[k][0]]= dat[k][1]
    spline_model.append(UnivariateSpline(list(datdict.keys()), list(datdict.values())))
    spline_fit = spline_model[i](oof_preds[i])
    
    print(f"adjusted logloss of cvsplit {i}: {log_loss(np.where(y>0,1,0),spline_fit)}") 


# In[52]:


val_cv = []
spline_model = []

for i in range(repeat_cv):
    dat = list(zip(oof_preds[i],np.where(y>0,1,0)))
    dat = sorted(dat, key = lambda x: x[0])
    datdict = {}
    for k in range(len(dat)):
        datdict[dat[k][0]]= dat[k][1]
    spline_model.append(UnivariateSpline(list(datdict.keys()), list(datdict.values())))
    spline_fit = spline_model[i](oof_preds[i])
    val_cv.append(pd.DataFrame({"y":np.where(y>0,1,0), "pred":spline_fit, "season":tourney_data.Season}))
    print(f"adjusted logloss of cvsplit {i}: {log_loss(np.where(y>0,1,0),spline_fit)}") 
    
val_cv = pd.concat(val_cv)
val_cv.groupby('season').apply(lambda x: log_loss(x.y, x.pred))


# ### Submission

# In[53]:


if config.stage_2: 
    sub = pd.read_csv('../input/ncaam-march-mania-2021/MDataFiles_Stage2/MSampleSubmissionStage2.csv')
    sub["Season"] = 2021
    sub["T1_TeamID"] = sub["ID"].apply(lambda x: x[5:9]).astype(int)
    sub["T2_TeamID"] = sub["ID"].apply(lambda x: x[10:14]).astype(int)
    sub = pd.merge(sub, season_statistics_T1, on = ['Season', 'T1_TeamID'])
    sub = pd.merge(sub, season_statistics_T2, on = ['Season', 'T2_TeamID'])
#     sub = pd.merge(sub, glm_quality_T1, on = ['Season', 'T1_TeamID'])
#     sub = pd.merge(sub, glm_quality_T2, on = ['Season', 'T2_TeamID'])
    sub = pd.merge(sub, seeds_T1, on = ['Season', 'T1_TeamID'])
    sub = pd.merge(sub, seeds_T2, on = ['Season', 'T2_TeamID'])
    sub = pd.merge(sub, last14days_stats_T1, on = ['Season', 'T1_TeamID'])
    sub = pd.merge(sub, last14days_stats_T2, on = ['Season', 'T2_TeamID'])
    sub["Seed_diff"] = sub["T1_seed"] - sub["T2_seed"]
    Xsub = sub[features].values
    dtest = xgb.DMatrix(Xsub)
    sub_models = []
    for i in range(repeat_cv):
        print(f"Fold repeater {i}")
        sub_models.append(
            xgb.train(
              params = param,
              dtrain = dtrain,
              num_boost_round = int(iteration_counts[i] * 1.05),
              verbose_eval = 50
            )
        )
    sub_preds = []
    for i in range(repeat_cv):
        sub_preds.append(
            spline_model[i](np.clip(sub_models[i].predict(dtest),-30,30))
        )

    sub["Pred"] = pd.DataFrame(sub_preds).mean(axis=0)
    #sub[['ID','Pred']].to_csv("submission.csv", index = None)


# In[54]:


def add_gameround(game_df):
    rnds = pd.read_csv(DATAPATH+'MNCAATourneySeedRoundSlots.csv', sep=',')
    rnds['ASeed'] = rnds['Seed'].apply(lambda x: int(x[1:3]))
    rnds['DayNum'] = rnds['EarlyDayNum']
    temp = rnds.copy()
    temp['DayNum'] = temp['LateDayNum']
    rnds = rnds[['ASeed','GameRound','DayNum']].append(temp[['ASeed','GameRound','DayNum']], ignore_index=True, sort=False)
    rnds.drop_duplicates(inplace=True)
    game_df = game_df.merge(rnds, on=['ASeed','DayNum'], how='left')
    return game_df


# In[55]:


rnds = pd.read_csv('../input/ncaam-march-mania-2021/MDataFiles_Stage2/MNCAATourneySeedRoundSlots.csv', sep=',')
rnds['ASeed'] = rnds['Seed'].apply(lambda x: int(x[1:3]))
rnds['DayNum'] = rnds['EarlyDayNum']
temp = rnds.copy()
temp['DayNum'] = temp['LateDayNum']
rnds = rnds[['ASeed','GameRound','DayNum']].append(temp[['ASeed','GameRound','DayNum']], ignore_index=True, sort=False)
rnds.drop_duplicates(inplace=True)


# In[56]:


sub.head()


# In[57]:


sub_final = sub[['ID','Pred']]
sub_final.to_csv("submission.csv", index = None)


# In[58]:


sub


# In[59]:




