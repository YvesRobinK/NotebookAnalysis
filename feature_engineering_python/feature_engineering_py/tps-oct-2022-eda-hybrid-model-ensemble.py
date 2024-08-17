#!/usr/bin/env python
# coding: utf-8

# # TPS Oct 2022
# The Oct edition of the 2022 Tabular Playground Series has as argument Rocket League matchs!!!. Given a snapshot from a Rocket League match, we need to predict the probability of each team scoring within the next 10 seconds of the game.
# 
# <img src="https://i.postimg.cc/cJZcdTGn/Tournaments-Prematch-Lobby-v2.webp"/>

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
get_ipython().system('pip install mrmr_selection')

from mrmr import mrmr_classif

import matplotlib.pyplot as plt
import random 
from numpy import dtype
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns
import gc
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

#ignore warning messages 
import warnings
warnings.filterwarnings('ignore') 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Train files
# - we've 10 big train files. In order to speed up the EDA we are going to use only the first one.
# - instead the **model that we are going to train will use ALL available files**

# In[2]:


dtypes_df = pd.read_csv('/kaggle/input/tabular-playground-series-oct-2022/train_dtypes.csv')
dtypes = {k: v for (k, v) in zip(dtypes_df.column, dtypes_df.dtype)}
print(f"Loading train set 0")
train0_df = pd.read_csv(f"/kaggle/input/tabular-playground-series-oct-2022/train_0.csv", dtype=dtypes)
train_df = train0_df
print(f"Loading test set")
test_df = pd.read_csv(f'/kaggle/input/tabular-playground-series-oct-2022/test.csv', dtype=dtypes)
submission_df = pd.read_csv(f'/kaggle/input/tabular-playground-series-oct-2022/sample_submission.csv')
gc.collect()


# In[3]:


train_df.head(5)


# In[4]:


train_df.shape


# In[5]:


test_df.head(5)


# In[6]:


test_df.shape


# ## Columns description:
# 
# - **game_num** (train only): Unique identifier for the game from which the event was taken.
# - **event_id** (train only): Unique identifier for the sequence of consecutive frames.
# - **event_time** (train only): Time in seconds before the event ended, either by a goal being scored or simply when we decided to truncate the timeseries if a goal was not scored.
# - **ball_pos_[xyz]**: Ball's position as a 3d vector.
# - **ball_vel_[xyz]**: Ball's velocity as a 3d vector.
# - For i in [0, 6]:
#     - **p{i}_pos_[xyz]**: Player i's position as a 3d vector.
#     - **p{i}_vel_[xyz]**: Player i's velocity as a 3d vector.
#     - **p{i}_boost**: Player i's boost remaining, in [0, 100]. A player can consume boost to substantially increase their speed, and is required to fly up into the z dimension (besides driving up a wall, or the small air gained by a jump).
#     - **boost{i}_timer**: Time in seconds until big boost orb i respawns, or 0 if it's available. Big boost orbs grant a full 100 boost to a player driving over it. The orb (x, y) locations are roughly [ (-61.4, -81.9), (61.4, -81.9), (-71.7, 0), (71.7, 0), (-61.4, 81.9), (61.4, 81.9) ] with z = 0. (Players can also gain boost from small boost pads across the map, but we do not capture those pads in this dataset).
# - **player_scoring_next** (train only): Which player scores at the end of the current event, in [0, 6], or -1 if the event does not end in a goal.
# - **team_scoring_next** (train only): Which team scores at the end of the current event (A or B), or NaN if the event does not end in a goal.
# - **team_[A|B]_scoring_within_10sec** (train only): [Target columns] Value of 1 if team_scoring_next == [A|B] and time_before_event is in [-10, 0], otherwise 0.
# - **id** (test and submission only): Unique identifier for each test row. Your submission should be a pair of team_A_scoring_within_10sec and team_B_scoring_within_10sec probability predictions for each id, where your predictions can range the real numbers from [0, 1].
# - Players 0, 1, and 2 make up team A and players 3, 4, and 5 make up team B.
# - The orientation vector of the player's car (which way the car is facing) does not necessarily match the player's velocity vector, and this dataset does not capture orientation data.
# 
# 
# ## Note
# - There are two features (UNAVAILABLE_FEATURES_ON_TEST) on train set not available on test set. for the moment we are going to skip them
# 

# In[7]:


UNAVAILABLE_FEATURES_ON_TEST = ['player_scoring_next', 'team_scoring_next']
train_df = train_df.drop(UNAVAILABLE_FEATURES_ON_TEST, axis=1)
train0_df = train0_df.drop(UNAVAILABLE_FEATURES_ON_TEST, axis=1)


# In[8]:


# Seed all
seed = 12
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
gc.collect()


# # EDA - Exploratory Data Analysis

# In[9]:


train_df.info(null_counts=True)


# # NaN Values
# There are some features with NaN values; 
# All p{i} columns will be NaN if and only if the player is demolished (destroyed by an enemy player; will respawn within a few seconds).
# 
# **How Manage them?**
# - we could impute them in some way like: mean, constant value, knn, linear regression or other
# - we could use a Tree-Model in order to manage them without impute
# 
# We choose the second one, so we'll train a LGBM Model

# ## Events distribution for single Games

# In[10]:


f, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (25, 10))
labels = [f"Games with {i+1} events" for i in range(10)]
train_df.groupby(['game_num'])['event_id'].nunique().value_counts().plot(kind="pie", ax=axes[0], labels=labels)
train_df.groupby(['game_num'])['event_id'].nunique().value_counts().plot(kind="bar", ax=axes[1])


# ## Frames count distribution for a Game

# In[11]:


train_df.groupby(['game_num'])['event_id'].count().plot(kind="hist", figsize=(30,5), bins=200, color="b")


# ## Frames count distribution for Game/Event

# In[12]:


train_df.groupby(['game_num'])['event_id'].value_counts().plot(kind="hist", figsize=(30,5), bins=500, color="b")


# ## Average and Minimum Event Time Duration

# In[13]:


train_df.groupby(['game_num'])['event_time'].min().plot(kind="hist", figsize=(30,5), bins=200, color="b")
train_df.groupby(['game_num'])['event_time'].mean().plot(kind="hist", figsize=(30,5), bins=200, color="r")


# ## Average Event Time difference between events_id on the same Game

# In[14]:


train_df['event_time_diff'] = abs(train_df['event_time'].shift(1).fillna(method='ffill')-train_df['event_time'])
average_event_time_diff_by_game = train_df.groupby(['game_num'])['event_time_diff'].mean()
average_event_time_diff_by_game.plot(kind="hist") 


# # Ball Position in the field when Team A or B make a goal within 10 sec.

# In[15]:


# Team A field vs Team B Field
f, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 10))
axes[0].title.set_text(f'Ball Position when Team A scoring within 10 sec')
axes[0].scatter(train_df[train_df['team_A_scoring_within_10sec']==1]['ball_pos_x'], train_df[train_df['team_A_scoring_within_10sec']==1]['ball_pos_y'], s=0.1)
axes[1].title.set_text(f'Ball Position when Team B scoring within 10 sec')
axes[1].scatter(train_df[train_df['team_B_scoring_within_10sec']==1]['ball_pos_x'], train_df[train_df['team_B_scoring_within_10sec']==1]['ball_pos_y'], s=0.1, c="red")


# # Team A and B Position in the field when itself score within 10 sec

# In[16]:


f, axes = plt.subplots(nrows = 1, ncols = 4, figsize = (20, 7))
f.suptitle("Team A positions distribution when itself scoring within 10 sec.")
for p in range(3):
    axes[p].title.set_text(f'Player {p}')
    axes[p].scatter(train_df[train_df['team_A_scoring_within_10sec']==1][f'p{p}_pos_x'], train_df[train_df['team_A_scoring_within_10sec']==1][f'p{p}_pos_y'], s=0.05)
axes[3].title.set_text(f'Ball Position')
axes[3].scatter(train_df[train_df['team_A_scoring_within_10sec']==1]['ball_pos_x'], train_df[train_df['team_A_scoring_within_10sec']==1]['ball_pos_y'], s=0.05)
    
f, axes = plt.subplots(nrows = 1, ncols = 4, figsize = (20, 7))
f.suptitle("Team B positions distribution when itself scoring within 10 sec.")
for p in range(3):
    axes[p].title.set_text(f'Player {3+p}')
    axes[p].scatter(train_df[train_df['team_B_scoring_within_10sec']==1][f'p{3+p}_pos_x'], train_df[train_df['team_B_scoring_within_10sec']==1][f'p{3+p}_pos_y'], s=0.05, c="red")
axes[3].title.set_text(f'Ball Position')
axes[3].scatter(train_df[train_df['team_B_scoring_within_10sec']==1]['ball_pos_x'], train_df[train_df['team_B_scoring_within_10sec']==1]['ball_pos_y'], s=0.05, c="red")


# # Team A and B Position in the field when the adversarial score within 10 sec

# In[17]:


f, axes = plt.subplots(nrows = 1, ncols = 4, figsize = (20, 7))
f.suptitle("Team A positions distribution when Team B scoring within 10 sec.")
for p in range(3):
    axes[p].title.set_text(f'Player {p}')
    axes[p].scatter(train_df[train_df['team_B_scoring_within_10sec']==1][f'p{p}_pos_x'], train_df[train_df['team_B_scoring_within_10sec']==1][f'p{p}_pos_y'], s=0.02)
axes[3].title.set_text(f'Ball Position')
axes[3].scatter(train_df[train_df['team_B_scoring_within_10sec']==1]['ball_pos_x'], train_df[train_df['team_B_scoring_within_10sec']==1]['ball_pos_y'], s=0.05, c="red")
    
f, axes = plt.subplots(nrows = 1, ncols = 4, figsize = (20, 7))
f.suptitle("Team B positions distribution when Team A scoring within 10 sec.")
for p in range(3):
    axes[p].title.set_text(f'Player {3+p}')
    axes[p].scatter(train_df[train_df['team_A_scoring_within_10sec']==1][f'p{3+p}_pos_x'], train_df[train_df['team_A_scoring_within_10sec']==1][f'p{3+p}_pos_y'], s=0.05, c="red")
axes[3].title.set_text(f'Ball Position')
axes[3].scatter(train_df[train_df['team_A_scoring_within_10sec']==1]['ball_pos_x'], train_df[train_df['team_A_scoring_within_10sec']==1]['ball_pos_y'], s=0.05)


# # Players Positions Analysis
# The attack of a team that makes a goal within 10 sec. is probably brought ahead by individuals players (attack team positions distributed randomly in the field) instead the defense is managed by all players of a team (defense team distributed near its gate)
# 
# **Idea.** This behavior could be translated as a new feature engineering

# # Player Positions Analysis in the Test set
# we try to visualize the players position in the test set

# In[18]:


f, axes = plt.subplots(nrows = 1, ncols = 7, figsize = (40, 9))
f.suptitle("Team A, B and Ball positions on Test set")
for p in range(3):
    axes[p].title.set_text(f'Player {p}')
    axes[p].scatter(test_df[f'p{p}_pos_x'], test_df[f'p{p}_pos_y'], s=0.002)
axes[3].title.set_text(f'Ball Position')
axes[3].scatter(test_df['ball_pos_x'], test_df['ball_pos_y'], s=0.002, c="green")
for p in range(3):
    axes[4+p].title.set_text(f'Player {3+p}')
    axes[4+p].scatter(test_df[f'p{3+p}_pos_x'], test_df[f'p{3+p}_pos_y'], s=0.002, c="red")


# # Targets Distribution (imbalanced)

# In[19]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 7))
labels_a = [f"{p:.2f}%" for p in train_df['team_A_scoring_within_10sec'].value_counts()/train_df['team_A_scoring_within_10sec'].value_counts().sum()*100]
labels_b = [f"{p:.2f}%" for p in train_df['team_B_scoring_within_10sec'].value_counts()/train_df['team_A_scoring_within_10sec'].value_counts().sum()*100]
train_df['team_A_scoring_within_10sec'].value_counts().plot(kind='pie', ax=ax1, labels=labels_a, startangle=-140)
train_df['team_B_scoring_within_10sec'].value_counts().plot(kind='pie', ax=ax2, labels=labels_b, startangle=-140)


# ## StratifiedKFold and imbalanced datasets
# 
# In unbalanced datasets, the minority class, often of greatest interest and whose predictions are most valuable, is more difficult to predict because in practice there are few examples available.
# 
# Furthermore, most machine learning algorithms for classification assume an equal distribution of classes. This means that in an unbalanced scenario the model focuses only on learning the characteristics of the most abundant observations, neglecting the examples of the minority class.
# 
# The most common approaches used for model evaluation are train / test splitting and the k-fold cross-validation procedure. Both approaches can be very effective in general, although they can lead to misleading results and potentially fail when used on classification problems with severe class imbalance. The techniques that must be used to stratify the sampling according to the class label are: the subdivision of the stratified train test and the stratified **k-fold cross-validation**
# 
# ### How to manage imbalanced dataset?
# - the first option is to check if your model algo support class weight option
# - another option is to use https://imbalanced-learn.org algorithms in order to under-sampling or over-sampling your dataset
# - use ensemble/bagging techniques help to reduce the effect of imbalanced dataset
# - naturally we need to validate our model using a cross-validation or train/test split which maintains the same percentage of target weight (ex. Stratified)
# 
# #### Note for sub-sampling
# - The problem with subsampling techniques is that you may strip out some valuable information and alter the overall distribution of the dataset that is representative in a specific domain
# - **Idea to Test** : trying to sub-sampling 0 label group in order to reduce the complete dataset and train on that

# In[20]:


# imbalanced ratio factor about 16
train_df['team_A_scoring_within_10sec'].value_counts()[0]/train_df['team_A_scoring_within_10sec'].value_counts()[1]


# => Our dataset has an imbalance ratio of about 16, so we might consider our case as medium unbalanced.
# 
# ### Note
# In real-world applications such as fraud detection we can also find imbalance ratios ranging from 1: 1000 up to 1: 5000. So, these would have a serious imbalance

# ## Features and Columns considerations
# 
# ### Games and Events
# - **game_num** (train only): 737x10 games.
# - **event_id** (train only): 3000 average events in each game.
# - **event_time** (train only): Time in seconds before the event ended, either by a goal being scored or simply when we decided to truncate the timeseries if a goal was not scored.
# 
# ### Ball and Players position
# - **[xyz] position (ball and players)**: Ball's position as a 3d vector.
#     * => **Match field measures** 
#         - X : [-80:80]
#         - Y : [-100:100]
#         - Z : [0:40]
#         
# ### Some Important Features are not available on Test set, so we are going to drop also on train set
# - **player_scoring_next** (train only): Important feature we need to transform with others (Which player scores at the end of the current event, in [0, 6], or -1 if the event does not end in a goal.)
# - **team_scoring_next** (train only): Important feature (Which team scores at the end of the current event (A or B), or NaN if the event does not end in a goal.)
# 

# # Features Correlation and HeatMap

# In[21]:


# features correlations values
matrix = train_df.corr()


# In[22]:


import seaborn as sns

# we use the absolute because negative correlation are good in the same way
matrix_correlations = matrix.abs()
mask = np.triu(np.ones_like(matrix_correlations, dtype=bool))
matrix_correlations = matrix_correlations.mask(mask)

# visualize correlation through heatmap chart
plt.figure(figsize=(30, 20))
mask_tri = np.triu(np.ones_like(matrix_correlations, dtype=bool))
sns.heatmap(matrix_correlations, cmap="RdYlBu_r", annot=True, fmt=".1f")
plt.show()


# ## Correlation beetween features and the target

# In[23]:


# correlation beetween features and the target
corr=matrix_correlations.round(2)  
corr=corr.iloc[-1,:-1].sort_values(ascending=False)
pal=sns.color_palette("RdYlBu",32).as_hex()
titles=[i for i in corr.index]
corr.index=titles
corr.plot.bar(color=pal, figsize=(20,5))


# # Features Engineering

# In[24]:


TARGETS = ['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec']
DROP_FEATURES = ['id', 'game_num', 'event_id', 'event_time', 'team_scoring_next', 'player_scoring_next']


# In[25]:


def columns_exchanger(df_in, a, b):
    df = df_in.copy()
    tmp = df[a].copy()
    df[a] = df[b].copy()
    df[b] = tmp.copy()
    return df
    
def feature_engineering(df_in, flip=False):
    """
    Work in progress
    """
    df = df_in.copy()
    # flipping the ball and the players position for "Data Augmentation" how suggest by @pietromaldini1 in discussion https://www.kaggle.com/competitions/tabular-playground-series-oct-2022/discussion/357577
    # for the moment I'm going to try to flip X position and changing players in the same Team
    # Mirroring all (x|y-axis and teams) doesn't fit well (I'm investigationg why)
    if flip:
        # flipping the ball only X
        df["ball_pos_x"] = -df["ball_pos_x"]
        #df["ball_pos_y"] = -df["ball_pos_y"]
        df["ball_vel_x"] = -df["ball_vel_x"]
        #df["ball_vel_y"] = -df["ball_vel_y"]
        # flipping the player position
        for player in range(6):
            df[f"p{player}_pos_x"] = -df[f"p{player}_pos_x"]
            #df[f"p{player}_pos_y"] = -df[f"p{player}_pos_y"]
            df[f"p{player}_vel_x"] = -df[f"p{player}_vel_x"]
            #df[f"p{player}_vel_y"] = -df[f"p{player}_vel_y"]
        # exchange players [0,2] to [3,5] in order to avoid to flip the targets 
        #for player in range(3):
        for template in ['p{PLAYER}_pos_x', 'p{PLAYER}_pos_y', 'p{PLAYER}_pos_z', 'p{PLAYER}_vel_x', 'p{PLAYER}_vel_y', 'p{PLAYER}_vel_z', 'p{PLAYER}_boost', 'boost{PLAYER}_timer']:
            df = columns_exchanger(df, template.format(PLAYER=0), template.format(PLAYER=2))
            df = columns_exchanger(df, template.format(PLAYER=3), template.format(PLAYER=5))
    # simgle player missing in the field
    for player in range(6):        
        df[f"p{player}_missing"] = df[f"p{player}_pos_x"].isna().astype('int8')
    df.fillna(0, inplace=True)
    # two simple new features rapresenting the distance between the ball and the gate of a team
    df[f"goal_a_distance"] = ((df["ball_pos_x"]-0)**2 + (df["ball_pos_y"]-100)**2 + (df["ball_pos_z"]-20)**2)**0.5
    df[f"goal_b_distance"] = ((df["ball_pos_x"]-0)**2 + (df["ball_pos_y"]+100)**2 + (df["ball_pos_z"]-20)**2)**0.5
    # two other features rapresenting the distance between the team and its gate indicates when the team is defending (see analysis above)
    df[f"team_a_defending"] = pd.Series(np.zeros(df.shape[0]), index=df.index)
    for player in range(3):
        df[f"team_a_defending"] += ((df[f"p{player}_pos_x"]-0)**2 + (df[f"p{player}_pos_y"]+100)**2 + (df[f"p{player}_pos_z"]-20)**2)**0.5
    df[f"team_b_defending"] = pd.Series(np.zeros(df.shape[0]), index=df.index)
    for player in range(3):
        df[f"team_b_defending"] += ((df[f"p{3+player}_pos_x"]-0)**2 + (df[f"p{3+player}_pos_y"]-100)**2 + (df[f"p{3+player}_pos_z"]-20)**2)**0.5
    # we calcolate 3 new features as the position [x,y,z] in the next future considering the velocity [x,y,z] and the average Event Time difference between events_id on the same Game see specific plot above
    df['ball_pos_x_next'] = df['ball_pos_x'] + df['ball_vel_x']*0.2
    df['ball_pos_y_next'] = df['ball_pos_y'] + df['ball_vel_y']*0.2
    df['ball_pos_z_next'] = df['ball_pos_z'] + df['ball_vel_z']*0.2
    df = df.drop(DROP_FEATURES, axis=1, errors="ignore")   
    return df


# # Hybrid Model Class
# 
# The following class permit us to encapsulate the logic of ensembling N models making a weighted average of their predictions (Soft-Voting)
# 
# We are going to choose several models in order to ensemble them and try to get better results
# 
# ## Hybrid Model Schema
# 
# ![https://i.postimg.cc/prVWrkCg/hybrid-model-drawio.png](https://i.postimg.cc/prVWrkCg/hybrid-model-drawio.png)
# 
# ### Note
# For a models comparison see my notebook ["TPS Oct 2022 | PyCaret Model Analysis"](https://www.kaggle.com/code/infrarosso/tps-oct-2022-pycaret-model-analysis)

# In[26]:


class EnsembleHybrid:
   def __init__(self, models=[], weights=[]):
       self.models = models
       self.weights = weights

   def fit(self, X, y):
       # Train models
       for m in self.models:
           print(f"Training {m}...")
           m.fit(X, y)

   def predict_proba(self, X_test):
       y_pred = pd.Series(np.zeros(X_test.shape[0]), index=X_test.index)
       for i, m in enumerate(self.models):
           y_pred += pd.Series(m.predict_proba(X_test)[:,1], index=X_test.index) * self.weights[i]
       return y_pred


# # Cross-validation with StratifiedKFold
# 
# I'm going to use StratifiedKFold in order to better fit imbalanced dataset
# We ensemble N models

# In[27]:


FOLDS = 5
    
def model_it(X_train, Y_train, X_test):
    ZERO_AS_MISSING = True
    SCALE_WEIGHT=16
    Y_validations, model_val_preds, model_test_preds, ensemble_models, scores=[],[],[],[],[]
    feat_importance=pd.DataFrame(index=X_train.columns)
        
    lgbm = LGBMClassifier(objective='binary',
                      metric='logloss',
                      importance_type='gain',
                      random_state=seed,
                      zero_as_missing=ZERO_AS_MISSING,
                      learning_rate=0.1,
                      max_depth=10,
                      min_child_samples=340,
                      min_child_weight=1e-05,
                      n_estimators=100,
                      num_leaves=130,
                      reg_alpha=50,
                      reg_lambda=50,
                      subsample=0.7865007820901366)   
    
    cat_boost = CatBoostClassifier(random_seed=seed,
                               eval_metric='Logloss',
                               logging_level='Silent',
                               learning_rate=0.05,
                               iterations=100)
    
    xgbm = XGBClassifier(objective='binary:logistic',
                     random_state=seed,
                     learning_rate=0.1,
                     n_estimators=100,
                     max_depth=8, 
                     #tree_method='gpu_hist')
                     tree_method='hist')
    
    #models = [lgbm, xgbm]
    #weights=[0.7, 0.3]
    models = [lgbm]
    weights=[1]
    
    k_fold = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(k_fold.split(X_train, Y_train)):
        print("\nFold {}".format(fold+1))
        X_fold_train, Y_fold_train = X_train.iloc[train_idx,:], Y_train[train_idx]
        X_fold_val, Y_fold_val = X_train.iloc[val_idx,:], Y_train[val_idx]
        print("Train shape: {}, {}, Valid shape: {}, {}".format(
            X_fold_train.shape, Y_fold_train.shape, X_fold_val.shape, Y_fold_val.shape))
        ensemble_model = EnsembleHybrid(models=models, weights=weights)
        ensemble_model.fit(X_fold_train, Y_fold_train)
        model_prob = ensemble_model.predict_proba(X_fold_val)
        Y_validations.append(Y_fold_val)
        model_val_preds.append(model_prob)
        model_test_preds.append(ensemble_model.predict_proba(X_test))
        ensemble_models.append(ensemble_model)
        feat_importance["Importance_Fold"+str(fold)]=lgbm.feature_importances_
        score=log_loss(Y_fold_val, model_prob)
        scores.append(score)
        print("Validation Log Loss = {:.4f}".format(score))
        del X_fold_train, Y_fold_train, X_fold_val, Y_fold_val
        gc.collect()
    return model_test_preds, feat_importance, ensemble_models


# In[28]:


# set to 10 to use all train dataset
# I'm going to use 9 for train set and I leave 1 for final validation
VALIDATION=True
if VALIDATION:
    TRAIN_SET = 1
    final_validation_df = pd.read_csv("/kaggle/input/tabular-playground-series-oct-2022/train_9.csv", dtype=dtypes)
    print("Loading train 9 as Validation set")
else:
    TRAIN_SET = 10


# # Feature Selection with MRMR (optional)
# Maximum Relevance — Minimum Redundancy” (aka MRMR) is an algorithm used by Uber’s machine learning platform for finding the “minimal-optimal” subset of features.
# - Enabling FEATURE_SELECTION_ENABLED the train will be done with the features selection algorithm
# - Disabling FEATURE_SELECTION_ENABLED the train will be done with all features availables in the train set (all columns)

# In[29]:


FEATURE_SELECTION_ENABLED = False
# FEATURE SELECTION MRMR
def feature_selection(X, y):
    if not FEATURE_SELECTION_ENABLED:
        return X.columns
    out = mrmr_classif(X, y, K=40)
    print("Features selection:", out)
    return out


# # Ensembling Hybrid Models 
# 
# The following schema describe how we are going to use single train_df file as a fold in order to train several hybrid models and soft voting their predictions for the final submission 
# 
# ## Ensemble Schema
# 
# ![https://i.postimg.cc/RhMkBk8r/tps-2022-10-drawio.png](https://i.postimg.cc/RhMkBk8r/tps-2022-10-drawio.png)
# 
# 

# ## Training All

# In[30]:


X_test = feature_engineering(test_df)
X_test.drop(TARGETS+DROP_FEATURES, axis=1, errors='ignore', inplace=True)
model_test_preds_a_all = []
model_test_preds_b_all = []
models_a = []
models_b = []
#FLIP_ITERATIONS_VALUES=[False, True]
FLIP_ITERATIONS_VALUES=[False]
for flip in FLIP_ITERATIONS_VALUES:
    print(f"Elaborating dataset flipped:{flip}")
    for i in range(TRAIN_SET):
        print(f"Loading train set {i}")
        if i==0:
            train_df = train0_df
        else:
            train_df = pd.read_csv(f"/kaggle/input/tabular-playground-series-oct-2022/train_{i}.csv", dtype=dtypes)
        # FEATURE ENGINEERING
        y_train_a = train_df[TARGETS[0]]
        y_train_b = train_df[TARGETS[1]]
        X_train = feature_engineering(train_df, flip)
        X_train.drop(TARGETS, axis=1, errors='ignore', inplace=True)
        X_train.reset_index().drop("index", axis=1, inplace=True)
        y_train_a = y_train_a.reset_index()['team_A_scoring_within_10sec']
        y_train_b = y_train_b.reset_index()['team_B_scoring_within_10sec']
        # FEATURE SELECTION
        features_selection_a = feature_selection(X_train, y_train_a)
        features_selection_b = feature_selection(X_train, y_train_b)
        # MODELLING
        model_test_preds_a, feat_importance_a, model_a_folds = model_it(X_train[features_selection_a], y_train_a, X_test[features_selection_a])
        model_test_preds_b, feat_importance_b, model_a_folds = model_it(X_train[features_selection_b], y_train_b, X_test[features_selection_b])
        model_test_preds_a_all.append(model_test_preds_a)
        model_test_preds_b_all.append(model_test_preds_b)
        models_a.append(model_a_folds)
        models_b.append(model_a_folds)
        gc.collect()


# ## Final Validation 

# In[31]:


# when flip is enabled we have 2 x folds and trainset
FLIP_FOLDS = len(FLIP_ITERATIONS_VALUES)

if VALIDATION:
    X_validation = feature_engineering(final_validation_df)
    y_validation_a = final_validation_df[TARGETS[0]]
    y_validation_b = final_validation_df[TARGETS[1]]
    y_validations = [y_validation_a, y_validation_b]
    teams = ['a', 'b']
    X_validation.drop(TARGETS+DROP_FEATURES, axis=1, errors='ignore', inplace=True)
    for i, models in enumerate([models_a, models_b]):
        model_fold_validation_preds = np.zeros(X_validation.shape[0])
        for folds_models in models:
            for fold_model in folds_models:
                model_fold_validation_preds += fold_model.predict_proba(X_validation)
        model_fold_validation_preds = model_fold_validation_preds/(FOLDS*TRAIN_SET*FLIP_FOLDS)
        score=log_loss(y_validations[i], model_fold_validation_preds)
        print(f"Final Validation Log Loss for Team {teams[i]} = {score:.4f}")


# ## Feature Importance by LGBM

# In[32]:


def feature_importance(feat_importance):
    feat_importance['avg']=feat_importance.mean(axis=1)
    feat_importance=feat_importance.sort_values(by='avg',ascending=False)

    pal=sns.color_palette("RdYlBu",32).as_hex()
    titles=[i for i in feat_importance.index]
    feat_importance.index=titles
    feat_importance['avg'].plot.bar(color=pal, figsize=(20,5))


# In[33]:


feature_importance(feat_importance_a)


# In[34]:


feature_importance(feat_importance_b)


# # Submission

# In[35]:


submission_a = np.zeros(test_df.shape[0])
submission_b = np.zeros(test_df.shape[0])
for i in range(len(model_test_preds_a_all)):
    for j in range(FOLDS):
        submission_a += model_test_preds_a_all[i][j]
        submission_b += model_test_preds_b_all[i][j]
# 2 if we train with flipped version of train data
submission_a = submission_a/(FOLDS*TRAIN_SET*FLIP_FOLDS)
submission_b = submission_b/(FOLDS*TRAIN_SET*FLIP_FOLDS)


# In[36]:


submission_df = pd.read_csv(f'/kaggle/input/tabular-playground-series-oct-2022/sample_submission.csv')
submission_df['team_A_scoring_within_10sec'] = submission_a
submission_df['team_B_scoring_within_10sec'] = submission_b
submission_df.to_csv('submission_lgbm.csv', index=False)
submission_df


# -------------------------------------------------------------------------
# <div style="text-align: center;">
#     <h3>Thanks for watching till the end ;)  </h3>
#     <h2>If you liked this notebook, upvote it ! </h2>
# <img src="https://i.postimg.cc/SsChkSJv/upvote.png" width="70"/>
# </div>
# 
# 
# 
# 
# ---
# 
