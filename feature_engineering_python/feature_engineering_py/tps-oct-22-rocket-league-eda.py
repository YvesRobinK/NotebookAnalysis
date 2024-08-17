#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction
# 
# <div style="color:white;display:fill;
#             background-color:#deb500;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>1.1 Background</b></p>
# </div>
# 
# This months TPS competition we are working with **Rocket League** data! This is the game where you play **football with cars**. 
# 
# <center>
# <img src='https://pm1.narvii.com/6922/56ece18e5e61afcb8c6dc13f797322f063d190f2r1-1280-640v2_hq.jpg' width=600>
# </center>
# <br>
# 
# > The goal of the competition is to predict -- from a given snapshot in the game -- for each team, the probability that they will score within the next 10 seconds of game time.
# 
# <br>
# 
# *Initial thoughts:*
# * This competition is definitely going to be a **challenge**; the dataset is massive (10GB) and the relationships between the features are very complex. 
# * The exciting part though is that there is lots of potential for **feature engineering**. 
# 
# <br>
# 
# <div style="color:white;display:fill;
#             background-color:#deb500;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>1.2 Libraries</b></p>
# </div>

# In[1]:


# Core
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.6)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from itertools import combinations
import math
import statistics
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import shapiro
from scipy.stats import chi2
from scipy.stats import poisson
import time
from datetime import datetime
import matplotlib.dates as mdates
import plotly.express as px
from termcolor import colored
import warnings
warnings.filterwarnings("ignore")

# Sklearn
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, TimeSeriesSplit, GroupKFold, cross_validate
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

# UMAP
import umap
import umap.plot

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB


# # 2. Data
# 
# <div style="color:white;display:fill;
#             background-color:#deb500;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>2.1 Load data</b></p>
# </div>
# 
# * The dataset is **very large** (10GB) - the train set is broken down into **10 parts**.
# * We will just use **small subset** of the train set to perform this EDA.
# 

# In[2]:


# Data types
dtypes_dict_train = dict(pd.read_csv('../input/tabular-playground-series-oct-2022/train_dtypes.csv').values)
dtypes_dict_test = dict(pd.read_csv('../input/tabular-playground-series-oct-2022/test_dtypes.csv').values)

# Data (only 10% of train set)
train = pd.read_csv("/kaggle/input/tabular-playground-series-oct-2022/train_0.csv", dtype=dtypes_dict_train)
test = pd.read_csv("../input/tabular-playground-series-oct-2022/test.csv", dtype=dtypes_dict_test)


# In[3]:


# Shape and preview
print('Train set shape:', train.shape)
display(train.head(3))

print('Test set shape:', test.shape)
display(test.head(3))


# In[4]:


# To speed up analysis, we only use a subset of the data
train = train.iloc[:200000]


# <div style="color:white;display:fill;
#             background-color:#deb500;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>2.2 File information</b></p>
# </div>
# 
# **Files**
# 
# > * **train_[0-9].csv**: Train set split into 10 files. Rows are sorted by game_num, event_id, and event_time, and each event is entirely contained in one file.
# > * **test.csv**: Test set. Unlike the train set, the rows are scrambled.
# > * **[train|test]_dtypes.csv**: pandas dtypes for the columns in the train / test set.
# > * **sample_submission.csv**: A sample submission in the correct format.
# 
# **Feature descriptions**
# 
# > * **game_num** (train only): Unique identifier for the game from which the event was taken.
# > * **event_id** (train only): Unique identifier for the sequence of consecutive frames.
# > * **event_time** (train only): Time in seconds before the event (e.g. goal or game end) ended.
# > * **ball_pos_[xyz]**: Ball's position as a 3d vector.
# > * **ball_vel_[xyz]**: Ball's velocity as a 3d vector.
# > * **p{i}_pos_[xyz]**: Player i's position as a 3d vector.
# > * **p{i}_vel_[xyz]**: Player i's velocity as a 3d vector.
# > * **p{i}_boost**: Player i's boost remaining, in [0, 100]. Boost temporarily increases player speed.
# > * **boost{i}_timer**: Time in seconds until big boost orb (resets boost to 100) i respawns, or 0 if it's available.
# > * **player_scoring_next** (train only): Which player scores at end of event, in [0, 5], or -1 if no goal.
# > * **team_scoring_next (train only)**: Which team scores at the end of event (A or B), or NaN if no goal.
# > * **team_[A|B]_scoring_within_10sec** (train only): 1 if team [A|B] scores in next 10 seconds, 0 otherwise.
# > * **id** (test and submission only): Unique identifier for each test row. 
# 
# **Additional notes**
# 
# > * All p{i} columns will be **NaN if and only if the player is demolished** (will respawn within a few seconds).
# > * Players 0, 1, and 2 make up **team A** and players 3, 4, and 5 make up **team B**.

# <div style="color:white;display:fill;
#             background-color:#deb500;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>2.3 Missing values</b></p>
# </div>
# 
# * Missing values appear when players have been **demolished**. They also appear in the feature 'team_scoring_next' if a team doesn't score at the end of the event.
# * Missing values represent **less than 1%** of the data.

# In[5]:


# Heatmap of missing values (subset of data)
plt.figure(figsize=(15,8))
sns.heatmap(train.loc[:,train.columns[train.isnull().any()]].isna().T, cmap='summer')
plt.title('Heatmap of missing values')
plt.show()


# In[6]:


# Missing values summary
mv=pd.DataFrame(train[train.columns[train.isnull().any()]].isna().sum(), columns=['Number_missing (TRAIN)'])
mv['Percentage_missing (TRAIN)']=np.round(100*mv['Number_missing (TRAIN)']/len(train),2)
mv['Number_missing (TEST)']=test[test.columns[test.isnull().any()]].isna().sum()
mv['Percentage_missing (TEST)']=np.round(100*mv['Number_missing (TEST)']/len(test),2)
mv.head()


# *Observations:*
# * Missing values indicate **demolitions**, which means one of the teams is outnumbered for a few seconds.
# * We can **create features to indicate missing values** (e.g. number of players on the pitch) as this may affect the target.

# # 3. EDA
# 
# <div style="color:white;display:fill;
#             background-color:#deb500;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>3.1 Targets</b></p>
# </div>
# 
# First, we'll visualise the **target variables**.
# 

# In[7]:


plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
sns.countplot(data=train, x='team_A_scoring_within_10sec')
plt.title('Target A')

plt.subplot(1,2,2)
sns.countplot(data=train, x='team_B_scoring_within_10sec')
plt.title('Target B')
plt.show()


# In[8]:


print('Team A target mean', train['team_A_scoring_within_10sec'].mean())
print('Team B target mean', train['team_B_scoring_within_10sec'].mean())


# In[9]:


plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
sns.countplot(data=train, x='team_scoring_next')
plt.title('Team scoring next')

plt.subplot(1,2,2)
sns.countplot(data=train, x='player_scoring_next')
plt.title('Player scoring next')
plt.show()


# * -1 corresponds to no player scoring next.

# <div style="color:white;display:fill;
#             background-color:#deb500;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>3.2 Temporal features</b></p>
# </div>
# 
# Now we'll look at the **features that depend on time**. Note that these only appear in the train set.

# In[10]:


plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
sns.histplot(data=train, x='game_num', binwidth=1)
plt.title('Game number')

plt.subplot(1,2,2)
sns.histplot(data=train, x='event_id', binwidth=1000)
plt.title('Frame number in event')
plt.show()


# In[11]:


plt.figure(figsize=(12,4))
sns.histplot(data=train, x='event_time')
plt.title('Time before event (s)')
plt.show()


# <div style="color:white;display:fill;
#             background-color:#deb500;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>3.3 Gameplay variables</b></p>
# </div>
# 
# We'll visualise position and velocity of the ball and players using **pairplots**. These show the **pairwise relationships** in a dataset.

# **Ball position & velocity**

# In[12]:


g = sns.PairGrid(train[['ball_pos_x','ball_pos_y','ball_pos_z']], height=4, aspect=1.3, corner=True)
g.map_lower(sns.scatterplot, s=10, alpha=0.02, color='green')
g.map_diag(sns.histplot)
plt.suptitle('Ball position', y=1.02, fontsize=22)
plt.show()


# *Observations:*
# * This plot gives us the **dimensions of the pitch**.
# * Notice how the ball spends **more time** on the **floor** and at the **edge** of the pitch.
# * There is also a **spike at the center** of the pitch due to kick-offs.

# In[13]:


g = sns.PairGrid(train[['ball_vel_x','ball_vel_y','ball_vel_z']], height=4, aspect=1.3, corner=True)
g.map_lower(sns.scatterplot, s=10, alpha=0.02, color='green')
g.map_diag(sns.histplot)
plt.suptitle('Ball velocity', y=1.02, fontsize=22)
plt.show()


# *Observations:*
# * The pairplots are roughly **spherical**, meaning they can attain any combination of velocities (up to a limit).
# * There is also a **spike at 0** due to kick-offs.

# **Player position & velocity**

# In[14]:


g = sns.PairGrid(train[['p0_pos_x','p0_pos_y','p0_pos_z']], height=4, aspect=1.3, corner=True)
g.map_lower(sns.scatterplot, s=10, alpha=0.02, color='orange')
g.map_diag(sns.histplot)
plt.suptitle('Player0 position', y=1.02, fontsize=22)
plt.show()


# *Observations:*
# * The **position** of the players covers the **whole pitch**.
# * **Some routes are used more than others**, e.g. to pick up boosts, or climb walls.
# * Most of the time the player is **on/close to the ground**. 

# In[15]:


g = sns.PairGrid(train[['p0_vel_x','p0_vel_y','p0_vel_z']], height=4, aspect=1.3, corner=True)
g.map_lower(sns.scatterplot, s=10, alpha=0.02, color='orange')
g.map_diag(sns.histplot)
plt.suptitle('Player0 velocity', y=1.02, fontsize=22)
plt.show()


# *Observations:*
# * The circular plot shows a players **speed is bounded**, i.e. cannot exceed a certain amount.
# * There are **2 rings** corresponding to **maximum speed** with and without boost.

# **Boost variables**

# In[16]:


plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
sns.histplot(data=train, x='p0_boost', binwidth=1)
plt.title('Boost amount for player 0')

plt.subplot(1,2,2)
sns.histplot(data=train, x='boost0_timer')
plt.title('Time until boost pad 0 resets')
plt.show()


# *Observations*:
# * **Spikes at 0 and 100** on the left plot are from players **running out/picking up big boost pads**.
# * There are **spikes at multiples of 12's** on the left plot because of '**mini boost pads**'.
# * The plot on the right shows big boost pads take **10 seconds to respawn**.

# <div style="color:white;display:fill;
#             background-color:#deb500;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>3.4 Correlations</b></p>
# </div>
# 
# * There are many **feature interactions** in the data.
# * Every players position and velocity is correlated, **usually positively**, with the ball's features.
# * Players' boost variables tend to be **negatively correlated** with other features.

# In[17]:


# Heatmap of correlations
plt.figure(figsize=(10,7))
sns.heatmap(train.corr(), cmap='bwr', vmin=-1, vmax=1)
plt.title('Correlations')


# <div style="color:white;display:fill;
#             background-color:#deb500;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>3.5 Violinplots</b></p>
# </div>
# 
# We will use violinplots to show how variables are **related to the target**.

# **Ball position vs target**

# In[18]:


for i in ['x','y','z']:
    plt.figure(figsize=(18,4))
    plt.subplot(1,2,1)
    sns.violinplot(data=train, x='team_A_scoring_within_10sec', y=f'ball_pos_{i}')

    plt.subplot(1,2,2)
    sns.violinplot(data=train, x='team_B_scoring_within_10sec', y=f'ball_pos_{i}')
    plt.show()


# *Observations*:
# * The ball is **more central** (in the x-axis) before a goal.
# * The ball is **closer to the goal** (in the y-axis) before a goal.
# * The ball tends to be **more elevated** (in the z-axis) before a goal. 

# **Player0 position vs target**

# In[19]:


for i in ['x','y','z']:
    plt.figure(figsize=(18,4))
    plt.subplot(1,2,1)
    sns.violinplot(data=train, x='team_A_scoring_within_10sec', y=f'p0_pos_{i}')

    plt.subplot(1,2,2)
    sns.violinplot(data=train, x='team_B_scoring_within_10sec', y=f'p0_pos_{i}')
    plt.show()


# *Observations*:
# * **Similar relationships** can be seen with 'ball position vs target' (above).

# # 4. Feature Engineering
# 
# Some initial ideas to get started with feature engineering.
# 
# <div style="color:white;display:fill;
#             background-color:#deb500;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>4.1 Speed</b></p>
# </div>
# 
# We can calculate the ball's and player's speed using the formula:
# 
# $$
# \text{speed} = \sqrt{v_x^2+v_y^2+v_z^2}
# $$

# In[20]:


# Calculate absolute speed of ball and players
def calc_speeds(df):
    df['ball_speed'] = np.sqrt((df['ball_vel_x']**2)+(df['ball_vel_y']**2)+(df['ball_vel_z']**2))
    for i in range(6):
        df[f'p{i}_speed'] = np.sqrt((df[f'p{i}_vel_x']**2)+(df[f'p{i}_vel_y']**2)+(df[f'p{i}_vel_z']**2))
    return df

train = calc_speeds(train)


# In[21]:


g = sns.PairGrid(train[['p0_speed','p1_speed','p2_speed']], height=4, aspect=1.3, corner=True)
g.map_lower(sns.scatterplot, s=10, alpha=0.02, color='purple')
g.map_diag(sns.histplot)
plt.suptitle('Team A speeds', y=1.02, fontsize=22)
plt.show()


# *Observations*:
# * There is a clear **positive correlation** between player's speeds.
# * The sharp lines correspond to **maximum speeds** with and without boost.

# <div style="color:white;display:fill;
#             background-color:#deb500;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>4.2 Demolitions</b></p>
# </div>
# 
# Demolitions result in a team having a **temporary numbers advantage**. We can keep track of these via the missing values and create additional features that could be helpful the our models.

# In[22]:


def demolitions(df):
    for i in range(6):
        df[f'p{i}_demo'] = (df[f'p{i}_pos_x'].isna()).astype(int)
    df['active_players_A'] = 3-df['p0_demo']-df['p1_demo']-df['p2_demo']
    df['active_players_B'] = 3-df['p3_demo']-df['p4_demo']-df['p5_demo']
    return df

train = demolitions(train)


# In[23]:


plt.figure(figsize=(10,5))
sns.histplot(train, x='active_players_A', hue='team_B_scoring_within_10sec')
plt.yscale('log')
plt.show()


# *Observations:*
# * Teams with **less active players** are more likely to **conceed a goal** in the next 10 seconds.

# <div style="color:white;display:fill;
#             background-color:#deb500;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>4.3 Distance to goal</b></p>
# </div>
# 
# We can calculate the distance of the ball/players to the goal using different metrics, like **euclidean distance** or **manhattan distance**.

# In[24]:


def dist_to_goal(df):
    # Estimates
    goal1_coord = (0,-102.5,1.2)
    goal2_coord = (0,102.5,1.2)
    
    # Euclidean distance
    df['ball_dist_to_goal1_euclid'] = np.sqrt((df['ball_pos_x']-goal1_coord[0])**2 + (df['ball_pos_y']-goal1_coord[1])**2 + (df['ball_pos_z']-goal1_coord[2])**2)
    df['ball_dist_to_goal2_euclid'] = np.sqrt((df['ball_pos_x']-goal2_coord[0])**2 + (df['ball_pos_y']-goal2_coord[1])**2 + (df['ball_pos_z']-goal2_coord[2])**2)
    
    # Manhattan distance
    df['ball_dist_to_goal1_manhat'] = np.absolute(df['ball_pos_x']-goal1_coord[0]) + np.absolute(df['ball_pos_y']-goal1_coord[1]) + np.absolute(df['ball_pos_z']-goal1_coord[2])
    df['ball_dist_to_goal2_manhat'] = np.absolute(df['ball_pos_x']-goal2_coord[0]) + np.absolute(df['ball_pos_y']-goal2_coord[1]) + np.absolute(df['ball_pos_z']-goal2_coord[2])
        
    for i in range(6):
        # Euclidean distance
        df[f'p{i}_dist_to_goal1_euclid'] = np.sqrt((df[f'p{i}_pos_x']-goal1_coord[0])**2 + (df[f'p{i}_pos_y']-goal1_coord[1])**2 + (df[f'p{i}_pos_z']-goal1_coord[2])**2)
        df[f'p{i}_dist_to_goal2_euclid'] = np.sqrt((df[f'p{i}_pos_x']-goal2_coord[0])**2 + (df[f'p{i}_pos_y']-goal2_coord[1])**2 + (df[f'p{i}_pos_z']-goal2_coord[2])**2)
        
        # Manhattan distance
        df[f'p{i}_dist_to_goal1_manhat'] = np.absolute(df[f'p{i}_pos_x']-goal1_coord[0]) + np.absolute(df[f'p{i}_pos_y']-goal1_coord[1]) + np.absolute(df[f'p{i}_pos_z']-goal1_coord[2])
        df[f'p{i}_dist_to_goal2_manhat'] = np.absolute(df[f'p{i}_pos_x']-goal2_coord[0]) + np.absolute(df[f'p{i}_pos_y']-goal2_coord[1]) + np.absolute(df[f'p{i}_pos_z']-goal2_coord[2])
    
    return df

train = dist_to_goal(train)


# In[25]:


for i in ['A','B']:
    plt.figure(figsize=(18,4))
    plt.subplot(1,2,1)
    sns.violinplot(data=train, x=f'team_{i}_scoring_within_10sec', y='ball_dist_to_goal1_euclid')

    plt.subplot(1,2,2)
    sns.violinplot(data=train, x=f'team_{i}_scoring_within_10sec', y='ball_dist_to_goal2_euclid')
    plt.show()


# *Observations:*
# * The **closer** the ball is to the goal, the **more likely** a goal is scored in that goal.

# <div style="color:white;display:fill;
#             background-color:#deb500;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>4.4 Min/max/mean distance to goal</b></p>
# </div>
# 
# The minimum distance of a team to a goal can indicate whether there is a **goal keeper** defending the goal. Similar features with max (is there a **stricker**) and mean can also be computed. 

# In[26]:


def min_dist_to_goal(df):
    # Team A
    df['min_dist_to_goal1_A'] = df[[f'p{i}_dist_to_goal1_euclid' for i in range(3)]].min(axis=1)
    df['min_dist_to_goal2_A'] = df[[f'p{i}_dist_to_goal2_euclid' for i in range(3)]].min(axis=1)
    
    # Team B
    df['min_dist_to_goal1_B'] = df[[f'p{i}_dist_to_goal1_euclid' for i in range(3,6)]].min(axis=1)
    df['min_dist_to_goal2_B'] = df[[f'p{i}_dist_to_goal2_euclid' for i in range(3,6)]].min(axis=1)
    return df

def max_dist_to_goal(df):
    # Team A
    df['max_dist_to_goal1_A'] = df[[f'p{i}_dist_to_goal1_euclid' for i in range(3)]].max(axis=1)
    df['max_dist_to_goal2_A'] = df[[f'p{i}_dist_to_goal2_euclid' for i in range(3)]].max(axis=1)
    
    # Team B
    df['max_dist_to_goal1_B'] = df[[f'p{i}_dist_to_goal1_euclid' for i in range(3,6)]].max(axis=1)
    df['max_dist_to_goal2_B'] = df[[f'p{i}_dist_to_goal2_euclid' for i in range(3,6)]].max(axis=1)
    return df

def mean_dist_to_goal(df):
    # Team A
    df['mean_dist_to_goal1_A'] = df[[f'p{i}_dist_to_goal1_euclid' for i in range(3)]].mean(axis=1)
    df['mean_dist_to_goal2_A'] = df[[f'p{i}_dist_to_goal2_euclid' for i in range(3)]].mean(axis=1)
    
    # Team B
    df['mean_dist_to_goal1_B'] = df[[f'p{i}_dist_to_goal1_euclid' for i in range(3,6)]].mean(axis=1)
    df['mean_dist_to_goal2_B'] = df[[f'p{i}_dist_to_goal2_euclid' for i in range(3,6)]].mean(axis=1)
    return df

train = min_dist_to_goal(train)
train = max_dist_to_goal(train)
train = mean_dist_to_goal(train)


# # 5. Next steps
# 
# <div style="color:white;display:fill;
#             background-color:#deb500;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 4px;color:white;"><b>5.1 Ideas</b></p>
# </div>
# 
# Here are some of the next steps that you can consider taking in your own notebooks.
# 
# * Fill **missing values** (with 0 or penalise in some other way)
# * More **feature engineering** (angles, distances, trajectories, etc - get creative)
# * Set up a **cross-validation** scheme to validate models (train-test split or group-k-fold)
# * Build **models** (tree-based like LGBM, neural networks or others)
# * **Compress the data** to be able to train the model on a bigger proportion of the dataset (parquet, feather, etc)
