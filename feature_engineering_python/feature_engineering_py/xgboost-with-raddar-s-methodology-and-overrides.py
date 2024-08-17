#!/usr/bin/env python
# coding: utf-8

# # This is where I got the idea for this notebook
# 
# This notebook is ready to submit and you just have to tweak it and make it yours. I would love it if someone lands a medal with it. Don't forget to mention @verracodeguacas at the end if you do.
# 
# The notebook roughly follows the guidelines of @raddar - Darius Barušauskas as shown in his youtube presentation: https://www.youtube.com/watch?v=KmhGNc7gcCM&t=18s&ab_channel=Kaggle
# 
# He won this tournament in 2018 using these ideas. I implemented in python but it won't be exactly the same because I added my own stuff. I basically changed some of the features and modified his "quality" measure to something that made more sense to me.
# 
# I dedicated the last few cells to create overrides - In case you like or dislike some particular university just for the hell of it. You can do that.
# 
# Two CSV's are created at the end. You can submit both since we have two final submission to be scored. Killing two birds with one stone!

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

pd.set_option("display.max_column", 999)
print(os.listdir("../input"))


# # Data preparation. 
# 
# A lot of this has to do with duplicating the data. Each game is seen once from the winner's and once from the loser's perspective. This is easy to understand if you watch the youtube video linked at the top

# In[2]:


tourney_results = pd.read_csv('../input/womens-march-mania-2022/WDataFiles_Stage2/WNCAATourneyDetailedResults.csv')
seeds = pd.read_csv('../input/womens-march-mania-2022/WDataFiles_Stage2/WNCAATourneySeeds.csv')
regular_results = pd.read_csv('../input/womens-march-mania-2022/WDataFiles_Stage2/WRegularSeasonDetailedResults.csv')

regular_results['WEFFG'] = regular_results['WFGM'] / regular_results['WFGA']
regular_results['WEFFG3'] = regular_results['WFGM3'] / regular_results['WFGA3']
regular_results['WDARE'] = regular_results['WFGM3'] / regular_results['WFGM']
regular_results['WTOQUETOQUE'] = regular_results['WAst'] / regular_results['WFGM']

regular_results['LEFFG'] = regular_results['LFGM'] / regular_results['LFGA']
regular_results['LEFFG3'] = regular_results['LFGM3'] / regular_results['LFGA3']
regular_results['LDARE'] = regular_results['LFGM3'] / regular_results['LFGM']
regular_results['LTOQUETOQUE'] = regular_results['LAst'] / regular_results['LFGM']
tourney_results.Season.unique()

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


# In[3]:


regular_data = prepare_data(regular_results)
tourney_data = prepare_data(tourney_results)


# # Feature engineering!

# In[4]:


tourney_data.columns


# # Feature Choices:
# 
# Features chosen from my personal knowledge of basketball. Feel free to perform your own, erase features, add features of your own. This is where you can have most of the fun! - Even raddar chose some features just because he "felt" like it.
# 

# In[5]:


boxscore_cols = ['T1_Score', 'T2_Score', 
        'T1_FGM', 'T1_FGA', 'T1_FGM3', 'T1_FGA3', 'T1_FTM', 'T1_FTA', 'T1_OR', 'T1_DR', 'T1_Ast', 'T1_TO', 'T1_Stl', 'T1_Blk', 'T1_PF', 
        'T2_FGM', 'T2_FGA', 'T2_FGM3', 'T2_FGA3', 'T2_FTM', 'T2_FTA', 'T2_OR', 'T2_DR', 'T2_Ast', 'T2_TO', 'T2_Stl', 'T2_Blk', 'T2_PF', 
        'PointDiff']

boxscore_cols = [
        'T1_FGM', 'T1_FGA', 'T1_FGM3', 'T1_FGA3', 'T1_OR', 'T1_Ast', 'T1_TO', 'T1_Stl', 'T1_PF', 
        'T2_FGM', 'T2_FGA', 'T2_FGM3', 'T2_FGA3', 'T2_OR', 'T2_Ast', 'T2_TO', 'T2_Stl', 'T2_Blk',  
        'PointDiff', 'T1_EFFG', 'T1_EFFG3', 'T1_DARE', 'T1_TOQUETOQUE', 'T2_EFFG', 'T2_EFFG3', 'T2_DARE', 'T2_TOQUETOQUE']
# Choose a function to aggregate
funcs = [np.mean]


# The idea is to be able to take a picture of the teams right before the tournament

# In[6]:


season_statistics = regular_data.groupby(["Season", 'T1_TeamID'])[boxscore_cols].agg(funcs).reset_index()
season_statistics.columns = [''.join(col).strip() for col in season_statistics.columns.values]
season_statistics.head()


# In[7]:


#Make two copies of the data
season_statistics_T1 = season_statistics.copy()
season_statistics_T2 = season_statistics.copy()

season_statistics_T1.columns = ["T1_" + x.replace("T1_","").replace("T2_","opponent_") for x in list(season_statistics_T1.columns)]
season_statistics_T2.columns = ["T2_" + x.replace("T1_","").replace("T2_","opponent_") for x in list(season_statistics_T2.columns)]
season_statistics_T1.columns.values[0] = "Season"
season_statistics_T2.columns.values[0] = "Season"


# In[8]:


# We don't have the box score statistics in the prediction bank. So drop it.
tourney_data = tourney_data[['Season', 'DayNum', 'T1_TeamID', 'T1_Score', 'T2_TeamID' ,'T2_Score']]
tourney_data.head()


# In[9]:


tourney_data = pd.merge(tourney_data, season_statistics_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_data = pd.merge(tourney_data, season_statistics_T2, on = ['Season', 'T2_TeamID'], how = 'left')


# Notice that there are Team 1 statistics, team 1 opponent's statistics, team 2 statistics and team 2 opponent statistics

# In[10]:


tourney_data.head()


# Darius likes to include a bit of extra information. I don't like it, but I put it in the comments here if you think it's useful.

# In[11]:


# These statistics are created because in the last 2 weeks some stuff may happen (injuries just before the tournament and such)
#last14days_stats_T1 = regular_data.loc[regular_data.DayNum>118].reset_index(drop=True)
#last14days_stats_T1['win'] = np.where(last14days_stats_T1['PointDiff']>0,1,0)
#last14days_stats_T1 = last14days_stats_T1.groupby(['Season','T1_TeamID'])['win'].mean().reset_index(name='T1_win_ratio_14d')

#last14days_stats_T2 = regular_data.loc[regular_data.DayNum>118].reset_index(drop=True)
#last14days_stats_T2['win'] = np.where(last14days_stats_T2['PointDiff']<0,1,0)
#last14days_stats_T2 = last14days_stats_T2.groupby(['Season','T2_TeamID'])['win'].mean().reset_index(name='T2_win_ratio_14d')


# In[12]:


#tourney_data = pd.merge(tourney_data, last14days_stats_T1, on = ['Season', 'T1_TeamID'], how = 'left')
#tourney_data = pd.merge(tourney_data, last14days_stats_T2, on = ['Season', 'T2_TeamID'], how = 'left')


# Extract the teams that make it to the tournament and see how they do with respect to the others

# In[13]:


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


# This formula is "kind of" what Darius calls the "quality" measure. I created my own because I didn't like his. Some issues with infinities and etc. Feel free to dial it back to what he shows. It's around minute 40 of his presentation

# In[14]:


def normalize_column(values):
  themean = np.mean(values)
  thestd = np.std(values)
  norm = (values - themean)/(thestd) 
  return(pd.DataFrame(norm))

def team_quality(season):
    formula = 'win~-1+T1_TeamID+T2_TeamID'
    glm = sm.GLM.from_formula(formula=formula, 
                              data=regular_season_effects.loc[regular_season_effects.Season==season,:], 
                              family=sm.families.Binomial()).fit()
    quality = pd.DataFrame(glm.params).reset_index()
    quality.columns = ['TeamID','quality']
    quality['Season'] = season
    quality['quality'] = normalize_column(quality['quality'])
    quality['quality'] = np.exp(quality['quality'])
    quality = quality.loc[quality.TeamID.str.contains('T1_')].reset_index(drop=True)
    quality['TeamID'] = quality['TeamID'].apply(lambda x: x[10:14]).astype(int)
    print(quality['quality'].mean(), quality['quality'].std())
    return quality


# In[15]:


# This is metric to measure the team's strength, in this case, this is a logistic regression and we
# the coefficients
glm_quality = pd.concat([team_quality(2010),
                         team_quality(2011),
                         team_quality(2012),
                         team_quality(2013),
                         team_quality(2014),
                         team_quality(2015),
                         team_quality(2016),
                         team_quality(2017),
                         team_quality(2018),
                         team_quality(2019),
                         team_quality(2021)]).reset_index(drop=True)


# In[16]:


glm_quality_T1 = glm_quality.copy()
glm_quality_T2 = glm_quality.copy()
glm_quality_T1.columns = ['T1_TeamID','T1_quality','Season']
glm_quality_T2.columns = ['T2_TeamID','T2_quality','Season']


# In[17]:


glm_quality_T1.shape


# In[18]:


tourney_data = pd.merge(tourney_data, glm_quality_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_data = pd.merge(tourney_data, glm_quality_T2, on = ['Season', 'T2_TeamID'], how = 'left')


# In[19]:


tourney_data.head()
tourney_data['T1_quality'].fillna(0.2, inplace = True)
tourney_data['T2_quality'].fillna(0.2, inplace = True)
tourney_data.T2_quality.isnull().sum()


# In[20]:


seeds.head()


# In[21]:


seeds['seed'] = seeds['Seed'].apply(lambda x: int(x[1:3]))
seeds.head()


# In[22]:


seeds_T1 = seeds[['Season','TeamID','seed']].copy()
seeds_T2 = seeds[['Season','TeamID','seed']].copy()
seeds_T1.columns = ['Season','T1_TeamID','T1_seed']
seeds_T2.columns = ['Season','T2_TeamID','T2_seed']


# In[23]:


tourney_data = pd.merge(tourney_data, seeds_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_data = pd.merge(tourney_data, seeds_T2, on = ['Season', 'T2_TeamID'], how = 'left')


# In[24]:


#Optional but not relevant
tourney_data["Seed_diff"] = tourney_data["T1_seed"] - tourney_data["T2_seed"]


# # Time to build some models!

# In[25]:


# The descriptive feature is the score, not the winner
y = tourney_data['T1_Score'] - tourney_data['T2_Score']
y.describe()


# In[26]:


features = list(season_statistics_T1.columns[2:999]) + \
    list(season_statistics_T2.columns[2:999]) + \
    list(seeds_T1.columns[2:999]) + \
    list(seeds_T2.columns[2:999]) + \
    ["Seed_diff"] + ["T1_quality","T2_quality"]

features


# In[27]:


X = tourney_data[features].values
dtrain = xgb.DMatrix(X, label = y)


# # Loss function
# 
# Darius didn't like the original loss function. He changed it a bit. This is the objective loss function provided to xgboost. Notice that it's smooth
# 

# In[28]:


def cauchyobj(preds, dtrain):
    labels = dtrain.get_label()
    c = 5000 
    x =  preds-labels    
    grad = x / (x**2/c**2+1)
    hess = -c**2*(x**2-c**2)/(x**2+c**2)**2
    return grad, hess


# In[29]:


param = {} 
#param['objective'] = 'reg:linear'
param['eval_metric'] =  'mae'
param['booster'] = 'gbtree'
param['eta'] = 0.02 #recommend change to ~0.02 for final run
param['subsample'] = 0.35
param['colsample_bytree'] = 0.7
param['num_parallel_tree'] = 10 #recommend 10 (this is very important for kagglers)
param['min_child_weight'] = 40
param['gamma'] = 10
param['max_depth'] =  3
param['silent'] = 1

print(param)


# In[30]:


xgb_cv = []
repeat_cv = 13 # recommend 10

for i in range(repeat_cv): 
    print(f"Fold repeater {i}")
    xgb_cv.append(
        xgb.cv(
          params = param,
          dtrain = dtrain,
          obj = cauchyobj,
          num_boost_round = 3000,
          folds = KFold(n_splits = 5, shuffle = True, random_state = i),
          early_stopping_rounds = 25,
          verbose_eval = 50
        )
    )


# In[31]:


iteration_counts = [np.argmin(x['test-mae-mean'].values) for x in xgb_cv]
val_mae = [np.min(x['test-mae-mean'].values) for x in xgb_cv]
iteration_counts, val_mae


# In[32]:


#This is to get out-of-fold predictions
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


# In[33]:


plot_df = pd.DataFrame({"pred":oof_preds[0], "label":np.where(y>0,1,0)})
plot_df["pred_int"] = plot_df["pred"].astype(int)
plot_df = plot_df.groupby('pred_int')['label'].mean().reset_index(name='average_win_pct')

plt.figure()
plt.plot(plot_df.pred_int,plot_df.average_win_pct)


# In[34]:


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


# In[35]:


plot_df = pd.DataFrame({"pred":oof_preds[0], "label":np.where(y>0,1,0), "spline":spline_model[0](oof_preds[0])})
plot_df["pred_int"] = (plot_df["pred"]).astype(int)
plot_df = plot_df.groupby('pred_int')['spline','label'].mean().reset_index()

plt.figure()
plt.plot(plot_df.pred_int,plot_df.spline)
plt.plot(plot_df.pred_int,plot_df.label)


# # Submission time!

# In[36]:


sub = pd.read_csv('../input/womens-march-mania-2022/WDataFiles_Stage2/WSampleSubmissionStage2.csv')
sub.shape


# In[37]:


sub["Season"] = sub["ID"].apply(lambda x: x[0:4]).astype(int)
sub["T1_TeamID"] = sub["ID"].apply(lambda x: x[5:9]).astype(int)
sub["T2_TeamID"] = sub["ID"].apply(lambda x: x[10:14]).astype(int)
sub.shape


# In[38]:


sub = pd.merge(sub, season_statistics_T1, on = ['Season', 'T1_TeamID'])
sub = pd.merge(sub, season_statistics_T2, on = ['Season', 'T2_TeamID'])
print(sub.shape)
sub = pd.merge(sub, glm_quality_T1, on = ['Season', 'T1_TeamID'], how = 'left') # This is because some teams didn't face off in the regular season
sub = pd.merge(sub, glm_quality_T2, on = ['Season', 'T2_TeamID'], how = 'left')
print(sub.shape)
sub = pd.merge(sub, seeds_T1, on = ['Season', 'T1_TeamID'])
sub = pd.merge(sub, seeds_T2, on = ['Season', 'T2_TeamID'])
print(sub.shape)
sub["Seed_diff"] = sub["T1_seed"] - sub["T2_seed"]
sub.shape


# In[39]:


sub.head()
print(sub.T2_quality.isnull().sum())
sub['T1_quality'].fillna(0.2, inplace = True)
sub['T2_quality'].fillna(0.2, inplace = True)
sub.T2_quality.isnull().sum()


# In[40]:


Xsub = sub[features].values
dtest = xgb.DMatrix(Xsub)


# In[41]:


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


# # Overrides in the end:
# 
# You can include some crazy bets in the end to get you more points and separate from the crowd. BEWARE! You can ruin everything here. These are just some universities that I like

# In[42]:


teamdata = pd.read_csv('../input/womens-march-mania-2022/WDataFiles_Stage2/WTeams.csv')
sub = pd.merge(sub, teamdata, left_on = 'T1_TeamID', right_on = 'TeamID', how = 'left')
sub = pd.merge(sub, teamdata, left_on = 'T2_TeamID', right_on = 'TeamID', how = 'left')
sub_preds = []
for i in range(repeat_cv):
    subm = sub_models[i].predict(dtest)
    subm[(sub['TeamName_x']=='Stanford') & (sub.T2_seed >= 4)] += 8.0 # Bet hard on Stanford
    subm[(sub['TeamName_y']=='Stanford') & (sub.T1_seed >= 4)] -= 8.0
    subm[(sub['TeamName_x']=='Stanford') & (sub.T2_seed >= 2)] += 1.435 # Bet hard on Stanford
    subm[(sub['TeamName_y']=='Stanford') & (sub.T1_seed >= 2)] -= 1.435
    subm[(sub['TeamName_x']=='Connecticut') & (sub.T2_seed >= 4)] += 2.435 # Bet hard on Connecticut
    subm[(sub['TeamName_y']=='Connecticut') & (sub.T1_seed >= 4)] -= 2.435
    subm[(sub.T1_seed <= 2) & (sub.T2_seed >= 3)] += 5.0 # The top 2 seeds seem to advance to elite 8 everytime
    subm[(sub.T2_seed <= 2) & (sub.T1_seed >= 3)] -= 5.0
    subm[(sub.T1_seed <= 1) & (sub.T2_seed >= 3)] += 4.0 # The top seed seems to advance to elite 8 everytime
    subm[(sub.T2_seed <= 1) & (sub.T1_seed >= 3)] -= 4.0
    sub_preds.append(np.clip(spline_model[i](np.clip(subm,-30,30)),0.025,0.975))
sub['Pred'] = pd.DataFrame(sub_preds).mean(axis = 0)


# These overlays below could easily backfire. Be careful!

# In[43]:


sub.loc[sub['Pred'] > 0.5, "Pred"] *= 1.01 # This is calibrated by trial and error
sub.loc[sub['Pred'] < 0.5, "Pred"] /= 1.03
print(sub['Pred'].mean())

spline_fit[(tourney_data.T1_seed==2) & (tourney_data.T2_seed==15) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 0.97
spline_fit[(tourney_data.T1_seed==3) & (tourney_data.T2_seed==14) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 0.97
spline_fit[(tourney_data.T1_seed==4) & (tourney_data.T2_seed==13) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 0.96
spline_fit[(tourney_data.T1_seed==15) & (tourney_data.T2_seed==2) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.03
spline_fit[(tourney_data.T1_seed==14) & (tourney_data.T2_seed==3) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.03
spline_fit[(tourney_data.T1_seed==13) & (tourney_data.T2_seed==4) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.04
sub["Pred"] = np.clip(sub["Pred"], 0.015, 0.985)
spline_fit[(tourney_data.T1_seed==1) & (tourney_data.T2_seed==16) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 0.97
spline_fit[(tourney_data.T1_seed==16) & (tourney_data.T2_seed==1) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.03
sub.to_csv("submission_map.csv", index = None)
sub[['ID','Pred']].to_csv("submission.csv", index = None)


# In[44]:


#print(sub.loc[sub['Pred'] > 0.5, "Pred"].mean())
#sub.loc[sub['Pred'] > 0.5, "Pred"] *= 1.008
#sub.loc[sub['Pred'] > 0.5, "Pred"].mean()


# In[45]:


#tourney_results2018 = pd.read_csv('../input/NCAA_2018_Solution_Womens.csv')
#tourney_results2018 = tourney_results2018[tourney_results2018.Pred!=-1].reset_index(drop=True)
#tourney_results2018.columns = ['ID', 'label']
#tourney_results2018 = pd.merge(tourney_results2018, sub, on = 'ID')
#log_loss(tourney_results2018.label, tourney_results2018.Pred)

