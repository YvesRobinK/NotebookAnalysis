#!/usr/bin/env python
# coding: utf-8

# In this notebook, I will demonstrate a full walkthrough of attacking this challenge. If you like any part of the notebook, please give it an upvote! 🎉
# ------
# 
# For this challenge, we will be using LightGBM as the primary ML model and optuna for hyperparameter tuning. I will make use of functions as the primary vehicle for feature engineering and to keep everything organized

# In[1]:


get_ipython().system(' pip install textstat')


# In[2]:


import os, random, optuna, textstat, umap
from functools import reduce
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from pandas_profiling import ProfileReport

from scipy.stats import mode
from sklearn.model_selection import cross_validate, StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.linear_model import LinearRegression

import category_encoders as ce

import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, Dataset, log_evaluation

import seaborn as sns
from matplotlib import pyplot as plt

plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)


# In[3]:


'''Set the directory for the data'''

ROOT_DIR = "../input/scrabble-player-rating"


# # 1. Exploratory Data Analysis
# - look at the dataset basics (size of the data, data types, look at a few examples etc.)
# - look for any missing data
# - look at target value
# - look at elements of the games (examine patterns, examine types of players, etc.)
# - look at elements of the `turns` data

# ## 1.a Profile the different datasets present in the data
# 
# __note__: I used PandasProfling for some additional profiling of the datasets, but commented those out in this notebook to save on space.

# In[4]:


train = pd.read_csv(os.path.join(ROOT_DIR, "train.csv"))
test = pd.read_csv(os.path.join(ROOT_DIR, "test.csv"))
turns = pd.read_csv(os.path.join(ROOT_DIR, "turns.csv"))
games = pd.read_csv(os.path.join(ROOT_DIR, "games.csv"))
sub = pd.read_csv(os.path.join(ROOT_DIR, "sample_submission.csv"))

display(train.head())
display(test.head())
display(turns.head())
display(games.head())
display(sub.head())


# In[5]:


print("train shape:", train.shape)
print("test shape:", test.shape)
print("turns shape:", turns.shape)
print("games shape:", games.shape)
print("submission shape:", sub.shape)


# In[6]:


#ProfileReport(train)


# In[7]:


#ProfileReport(turns)


# In[8]:


#ProfileReport(games)


# From the initial dataset profiling
# - There are some missing values that we need to look into
# - There are both categorical and real-valued variables, so we need to consider encoding schemes
# - There are some obvious correlations between some variables (i.e. 'lexicon' and 'rating_mode')

# ## 1.b look at the missing values
# - look for any patterns in missing data
# - look at some examples of missing data

# In[9]:


turns.isna().sum()


# Begin with looking at the missing data in the 'rack' variable

# In[10]:


turns[turns['rack'].isna()].head()


# In[11]:


'''Display counts and types of data point where rack is blank'''

np.unique(turns[turns['rack'].isna()]['turn_type'], return_counts=True)


# In[12]:


np.unique(turns[turns['rack'].isna()]['move'], return_counts=True)


# Next, look at the missing data in turn_type

# In[13]:


turns[turns['turn_type'].isna()].head()


# In[14]:


games.isna().sum()


# In[15]:


test.isna().sum()


# From looking at the missing data:
# - the missing data in the test file appears to the be missing target variable (i.e. the rating of the player before that game)
# - In both the move and location have have NA's for games in the test set. Location and rack are also NA if a player did not place pieces. <s>We must be careful in handling these NA's and keep in mind we need to use __previous__ games to the test games </s> (we should also look to see if the test games are the final game in the string of games for all of the players).

# ## 1.c Look at the target variable

# In[16]:


plt.figure(figsize=(14,10))
sns.histplot(train['rating'], ax=plt.gca())


# In[17]:


print("primary mode of the target variable: {}".format(mode(train['rating'])))
print("secondary mode of the target variable: {}".format(mode(train[train["rating"]!=1500]['rating'])))


# The ratings have a couple of different peaks of values and a big peak at 1500. Perhaps this is the inital rating assigned to a new user? We will need to look more into this particular value, espeically if it occurs for all users, and if so, when.

# ## 1.d EDA of the Games
# 
# Based on the observattions from the first part of the EDA, lets look to see the following
# - How many games does each non-bot player have?
# - What do the course of a players games' look like? Is it different between bots?
# - When does the '1500' player rating occur?
# - Are there any patterns in the variables that we might be able to exploit for determining a player's ratings over the course of their games

# In[18]:


bot_names =["BetterBot", "STEEBot", "HastyBot"]

names, counts= np.unique(train[~train['nickname'].isin(bot_names)]['nickname'], return_counts=True)
plot = sns.displot(counts, kind='ecdf')
plot.fig.suptitle("Cummulative distibution of number of games per player")


# In[19]:


'''Look at Game Stats per User'''

print("most games per account: {}".format(np.sort(counts)[-10:]))
print("mean games per account: {}".format(np.mean(counts)))
print("median games per account: {}".format(np.median(counts)))


# In[20]:


'''create a dataframe of the bot information for each of the games'''

bot_df = train[["game_id", "nickname", "score", "rating"]].copy()
bot_df['bot_name'] = bot_df['nickname'].apply(lambda x: x if x in bot_names else np.nan)
bot_df = bot_df[["game_id", "score", "rating", "bot_name"]].dropna(subset=["bot_name"])
bot_df.columns = ["game_id", "bot_score", "bot_rating", "bot_name"]


# In[21]:


full_df = train[~train['nickname'].isin(bot_names)] #take out the bots
full_df = full_df.merge(bot_df, on="game_id") #add in bot information
full_df = full_df.merge(games, on="game_id") # add in game information
full_df["created_at"] = pd.to_datetime(full_df["created_at"]) #transform the date and time to a format pandas understands


# In[22]:


'''Check for any non-bot games (i.e. human on human)'''

full_df[full_df['bot_name'].isna()]


# In[23]:


''' Pick a random nickname and look at their ratings over the course of their games'''

nickname = full_df["nickname"].sample(1).values[0]
print(nickname)

full_df[full_df["nickname"]==nickname].sort_values(by="created_at")[["created_at", "rating", "bot_name"]]


# In[24]:


fig, axes = plt.subplots(1, 2, figsize=(16,8))

sns.scatterplot(data = full_df[full_df["nickname"]==nickname].sort_values(by="created_at")[["created_at", "rating", "bot_name"]], x="created_at", y="rating", hue="bot_name", ax=axes[0])
axes[0].set_title("Player {} ratings over the course of thier games, by bot".format(nickname))

sns.scatterplot(data = full_df[full_df["nickname"]==nickname].sort_values(by="created_at")[["created_at", "score", "bot_name"]], x="created_at", y="score", hue="bot_name", ax=axes[1])
axes[1].set_title("Player {} scores over the course of thier games, by bot".format(nickname))


# So, most players have very few games (less than 12), and some have a lot (pareto type of phenomenon). And, players only play against bots. Also, some players do play different bots over the course of their games. Players' ratings are also dynamic with time, with various patterns in that dyamicity. And, all of the games area against some kind of bot. __However__, we also need to note that we are not given any of the ratings nor any of the moves or locations for any of the players in the test set.
# 
# ### 1.d.1 Now, let's turn to looking at Players with only one rating
# - Are there cases where players just stay at one rating?
# - When does the rating of '1500' and '1640' occur for players?
# - Are they present in the test set

# In[25]:


'''How many users have a 1500 rating ever'''

users, counts = np.unique(full_df[full_df['rating'] == 1500]['nickname'], return_counts=True)

print("number of users with 1500 rating: {}".format(len(users)))
print("counts per user")
print(counts)


# In[26]:


'''look at those users that have more than 1 1500 ratings'''

print("users with a high number of 1500 ratings: ")
print(users[np.argsort(counts)[::-1][:34]])
print("...and their counts of 1500 scores: ")
print(counts[np.argsort(counts)[::-1][:34]])


# In[27]:


'''Look at the user with the most 1500 ratings'''

nickname = 'BB-8'
full_df[full_df["nickname"]=='BB-8'].sort_values(by="created_at")[["created_at", "rating", "bot_name"]]


# In[28]:


fig, axes = plt.subplots(1, 2, figsize=(16,8))

sns.scatterplot(data = full_df[full_df["nickname"]==nickname].sort_values(by="created_at")[["created_at", "rating", "bot_name"]], x="created_at", y="rating", hue="bot_name", ax=axes[0])
axes[0].set_title("Player {} ratings over the course of their games, by bot".format(nickname))

sns.scatterplot(data = full_df[full_df["nickname"]==nickname].sort_values(by="created_at")[["created_at", "score", "bot_name"]], x="created_at", y="score", hue="bot_name", ax=axes[1])
axes[1].set_title("Player {} scores over the course of their games, by bot".format(nickname))


# In[29]:


'''Look at the user with the next number of 1500 ratings'''

nickname = 'stevy'
full_df[full_df["nickname"]=='BB-8'].sort_values(by="created_at")[["created_at", "rating", "bot_name"]]


# In[30]:


fig, axes = plt.subplots(1, 2, figsize=(16,8))

sns.scatterplot(data = full_df[full_df["nickname"]==nickname].sort_values(by="created_at")[["created_at", "rating", "bot_name"]], x="created_at", y="rating", hue="bot_name", ax=axes[0])
axes[0].set_title("Player {} ratings over the course of their games, by bot".format(nickname))

sns.scatterplot(data = full_df[full_df["nickname"]==nickname].sort_values(by="created_at")[["created_at", "score", "bot_name"]], x="created_at", y="score", hue="bot_name", ax=axes[1])
axes[1].set_title("Player {} scores over the course of their games, by bot".format(nickname))


# In[31]:


'''Look at a different player with more than one, but not all, 1500 ratings'''

nickname = 'BethMix'
full_df[full_df["nickname"]==nickname].sort_values(by="created_at")[["created_at", "rating", "bot_name"]]


# In[32]:


fig, axes = plt.subplots(1, 2, figsize=(16,8))

sns.scatterplot(data=full_df[full_df["nickname"]==nickname].sort_values(by="created_at")[["created_at", "rating", "bot_name"]], x="created_at", y="rating", hue="bot_name", ax=axes[0])
axes[0].set_title("Player {} ratings over the course of thier games, by bot".format(nickname))

sns.scatterplot(data=full_df[full_df["nickname"]==nickname].sort_values(by="created_at")[["created_at", "score", "bot_name"]], x="created_at", y="score", hue="bot_name", ax=axes[1])
axes[1].set_title("Player {} scores over the course of thier games, by bot".format(nickname))


# In[33]:


df = full_df[full_df['nickname'].isin(users)][['nickname', 'rating']].groupby('nickname').agg({"nickname":"count",
                                                                                         "rating" : lambda x: np.sum(x == 1500)
                                                                                         })
df["ratio"] = df["rating"]/df["nickname"]
print("number of accounts that only have 1500 ratings and more than one game: {}".format(len(df[(df["ratio"] >=1.0) & (df["nickname"]>1)])))


# In[34]:


df[(df["ratio"] >=1.0) & (df["nickname"]>1)]


# So, there are definitely some anamolous accounts, like 'BB-8' that only ever have a 1500 rating. <s>I am not sure what these are, but they should probably be excluded, since we are not given any of the ratings for a player in the test set and we'll just asusme none of them are these anamolous players.</s> We now need to look more into these one-rating players.
# 
# ### 1.d.3. Looking at non-rated games and one-game players
# 
# Based on observation from [HP](https://www.kaggle.com/jaunedeau)
# 
# - What are the ratings given to players that only play one game. Are these possible starting values for ratings?
# - What ratings are given to those that all or mostly play non-rated games? do these ratings change at all?
# - Are there casual-only players in both train and test sets?
# - What is the proportion of rated to non-rated games for players?

# In[35]:


game_count_and_rating_df = full_df.groupby("nickname").agg({"nickname":"count", "rating":"mean"})
game_count_and_rating_df.columns = ["counts_of_games", "average_rating"]

plt.figure(figsize=(16,8))
sns.displot(game_count_and_rating_df[game_count_and_rating_df["counts_of_games"] ==1]["average_rating"])
plt.title("Ratings for Players with Only One Game")


# In[36]:


game_count_and_rating_df.corr()


# So, players with only one game can vary in their rating. Thus, 1500 is not the one-game-rating rating. Now, lets turn to those players that only play casual games. Do their ratings vary ever?

# In[37]:


casual_df = full_df.copy()
casual_df["rating_mode"] = casual_df["rating_mode"].apply(lambda x: 1 if x=="CASUAL" else 0)

user_casual_game_frac_df = casual_df.groupby("nickname").agg({"rating_mode":"mean"})
casual_only_users = user_casual_game_frac_df[user_casual_game_frac_df["rating_mode"] >= 1.0].index


# In[38]:


np.unique(casual_df[casual_df["nickname"].isin(casual_only_users)].groupby("nickname").agg({"rating":"std"}).sort_values("rating", ascending=False) >0,
          return_counts=True)


# In[39]:


'''Look at an example of these users that only play casual, but
have variable ratings'''

full_df[full_df["nickname"] == "J-Oriola"].sort_values("created_at")


# In[40]:


sns.displot(casual_df[casual_df["nickname"].isin(casual_only_users)].groupby("nickname").agg({"nickname":"count"})).set(title="Distribution of Numbers of Games for Casual Only Players")
plt.show()


# In[41]:


'''Are there players in the test set that only play casual games?'''

test_df = test[~test['nickname'].isin(bot_names)] #take out the bots
test_df = test_df.merge(games, on="game_id") # add in game information

test_df["rating_mode"] = test_df["rating_mode"].apply(lambda x: 1 if x=="CASUAL" else 0)
user_casual_game_frac_df = test_df.groupby("nickname").agg({"rating_mode":"mean"})
casual_only_users = user_casual_game_frac_df[user_casual_game_frac_df["rating_mode"] >= 1.0].index


# In[42]:


print(casual_only_users)


# In[43]:


len(casual_only_users)


# In[44]:


'''Look at an example of these users that only play casual in the test set'''

test_df[test_df["nickname"] == 'Theus'].sort_values("created_at")


# In[45]:


sns.displot(test_df[test_df["nickname"].isin(casual_only_users)].groupby("nickname").agg({"nickname":"count"})).set(title="Distribution of Numbers of Games for Casual Only Players\n in the Test set")
plt.show()


# ### 1.d.4. Clustering of the types of players
# 
# For this we know that some players only play casual games. We also now that the `lexicon` `game_control_type` and `rating_mode` all have different ratings. So, lets investigate if there are patterns, or clusters, of the users based on their game styles.

# In[46]:


'''what is the fraction of games that are rated for a given user'''

rated_games_by_user_train = full_df.groupby("nickname").agg({'rating_mode': [('rated_count', lambda x: np.sum(x=='RATED')), 'count']})
rated_games_by_user_train['rated_fraction'] = rated_games_by_user_train[('rating_mode', 'rated_count')]/rated_games_by_user_train[('rating_mode',       'count')]


test_df = test[~test['nickname'].isin(bot_names)] #take out the bots
test_df = test_df.merge(games, on="game_id") # add in game information
rated_games_by_user_test = test_df.groupby("nickname").agg({'rating_mode': [('rated_count', lambda x: np.sum(x=='RATED')), 'count']})
rated_games_by_user_test['rated_fraction'] = rated_games_by_user_test[('rating_mode', 'rated_count')]/rated_games_by_user_test[('rating_mode',       'count')]


# In[47]:


fig, axs = plt.subplots(1,2 , sharey=True, figsize=(20,8))
sns.histplot(rated_games_by_user_train['rated_fraction'], ax=axs[0], stat='density').set(title="Ditribution of Ratio of rated to total games\n by user in the train set")
sns.histplot(rated_games_by_user_test['rated_fraction'], ax=axs[1], stat='density').set(title="Ditribution of Ratio of rated to total games\n by user in the test set")
plt.show()


# Now, lets look at the general types (or, clusters of players) that exist in the dataset. We will look across both train and test

# In[48]:


train_test_df = pd.concat([train.copy(), test.copy()])

bot_df = train_test_df[["game_id", "nickname", "score", "rating"]].copy()
bot_df['bot_name'] = bot_df['nickname'].apply(lambda x: x if x in bot_names else np.nan)
bot_df = bot_df[["game_id", "score", "rating", "bot_name"]].dropna(subset=["bot_name"])
bot_df.columns = ["game_id", "bot_score", "bot_rating", "bot_name"]

train_test_df = train_test_df[~train_test_df['nickname'].isin(bot_names)] #take out the bots
train_test_df = train_test_df.merge(bot_df, on="game_id") #add in bot information
train_test_df = train_test_df.merge(games, on="game_id") # add in game information
train_test_df["created_at"] = pd.to_datetime(train_test_df["created_at"]) #transform the date and time to a format pandas understands


# In[49]:


'''
create a df of all of the players and their game information. Normalize that game information by player and by the
variable (i.e. game_control_type, rating_mode, etc.)
'''
features_to_include = ['time_control_name', 'lexicon', 'rating_mode']
game_types_df = train_test_df[['nickname']+features_to_include].copy()
game_types_df = game_types_df[['nickname']].join(ce.OneHotEncoder().fit_transform(game_types_df[features_to_include]))

game_types_df = game_types_df.groupby("nickname").agg("sum")
for col_set in [['time_control_name_1','time_control_name_2','time_control_name_3','time_control_name_4'],
                ['lexicon_1','lexicon_2','lexicon_3','lexicon_4'], ['rating_mode_1','rating_mode_2']
               ]:

    game_types_df[col_set] = game_types_df[col_set].div(game_types_df[col_set].sum(axis=1), axis=0)


# In[50]:


'''
Now, get a low dimensional embedding the players by their game stats
'''

reducer = umap.UMAP()
embedding = pd.DataFrame(reducer.fit_transform(game_types_df), columns=['x','y'])
embedding['casual_ratio'] = game_types_df['rating_mode_1'].values
embedding['regular_game_ratio'] = game_types_df['time_control_name_1'].values


# In[51]:


fig, axs = plt.subplots(1,2 , sharey=True, figsize=(14,6))

sns.scatterplot(data=embedding, x='x', y='y', hue='casual_ratio', hue_norm=(0,1), alpha=0.2, ax=axs[0])
sns.scatterplot(data=embedding, x='x', y='y', hue='regular_game_ratio', hue_norm=(0,1), alpha=0.2, ax=axs[1])
plt.show()


# Som we can observe that there are some distinct clusters of user by the types of games they tend to play. Most of these clusters look like they can be sorted by the ratio of rated to casual games they play

# In[52]:


'''
Now, lets bin the types of players by their games using clustering
'''

cluster = AgglomerativeClustering(n_clusters=9)
cluster_labels = cluster.fit_predict(game_types_df)


# In[53]:


embedding['cluster_label'] = cluster_labels
sns.scatterplot(data=embedding, x='x', y='y', hue='cluster_label', palette="deep", alpha=0.2)
plt.show()


# In[54]:


'''
create a df of all of the players and their winning information. 
'''
win_types_df = ce.OneHotEncoder().fit_transform(train_test_df['game_end_reason'])
win_types_df['nickname'] = train_test_df['nickname']
win_types_df['winner'] = train_test_df['winner']
win_types_df['score'] = train_test_df['score']
win_types_df['score_diff'] = train_test_df['score']-train_test_df['bot_score']

win_types_df = win_types_df.groupby(['nickname']).agg({'score_diff':'mean', 'score': 'mean', 'game_end_reason_1': 'sum', 'game_end_reason_2': 'sum', 'game_end_reason_3':'sum', 
                                                       'game_end_reason_4': 'sum', 'winner': lambda x: np.sum(x==1)/len(x)})

win_types_df[['score_diff', 'score']] = MinMaxScaler().fit_transform(win_types_df[['score_diff', 'score']]) #norm to 0,1
col_set = ['game_end_reason_1', 'game_end_reason_2', 'game_end_reason_3', 'game_end_reason_4']
win_types_df[col_set] = win_types_df[col_set].div(win_types_df[col_set].sum(axis=1), axis=0) #create distribution across types


# In[55]:


'''
Now, get a low dimensional embedding the players by their game stats
'''

reducer = umap.UMAP()
embedding = pd.DataFrame(reducer.fit_transform(win_types_df), columns=['x','y'])
embedding['win_ratio'] = win_types_df['winner'].values
embedding['score'] = win_types_df['score'].values


# In[56]:


fig, axs = plt.subplots(1,2 , sharey=True, figsize=(14,6))

sns.scatterplot(data=embedding, x='x', y='y', hue='win_ratio', hue_norm=(0,1), alpha=0.2, ax=axs[0])
sns.scatterplot(data=embedding, x='x', y='y', hue='score', hue_norm=(0,1), alpha=0.2, ax=axs[1])
plt.show()


# The users can also be roughly grouped by their skill, or their ability to win

# In[57]:


'''
Now, lets bin the types of players by their games using clustering
'''

cluster = AgglomerativeClustering(n_clusters=7)
cluster_labels = cluster.fit_predict(embedding[['x', 'y']])


# In[58]:


embedding['cluster_label'] = cluster_labels
sns.scatterplot(data=embedding, x='x', y='y', hue='cluster_label', palette="deep", alpha=0.2)
plt.show()


# Alright from looking at some of these data points, some observations. First, casual only players *do have* changes in their ratings. <s>From looking at exmaples of those that do change their ratings and those that don't it looks like the rating change is a discrete thing that happens whenever a player changes their game settings from the previous game (i.e. `bot_name` to `HastyBot`, `time_control_name`, or `lexicon`)</s>. It looks like the changes in rating that occurs for casual-only players is a result of missing data; players have played rating games with a particular `bot_name` to `HastyBot`, `time_control_name`, or `lexicon` combination that did not get recorded in the dataset, and then went on to play that same combination in casual games. Also, it looks like there are casual only players in both the train and test sets. And, from looking at the ratio of games, most users are either all rated or aall casual players in both the train and test sets, with the preponderence being toward all rated.
# 
# When taking these game type variables into account it looks like they can be binned into roughly 9 types iof players, who mostly differ on the ratio of rated games they play.
# 
# At any rate, it seems like we need to include some kind of comparions between the game settings from the last game to the current game as a feature

# ### 1.d.5 Now, turning to any possible positive correlations between variables.
# __Note__: that when looking at these correlations, we need to keep in mind that the rating for a game is the rating __before__ that game is played

# In[59]:


sns.displot(full_df[["rating", "score", "bot_name"]], x="score", y="rating", hue="bot_name", kind="kde")


# In[60]:


sns.displot(full_df[["rating", "score", "winner"]], x="score", y="rating", hue="winner", kind="kde")


# We can observe that higher scores are generally correlate with higher ratings (same correlation seen in profiling). Higher scores do come with winning, which makes sense, however, it doesn't really seem to have much bearing on rating. Finally, the ratings do seem to congregate into some areas, depending on the bot.

# In[61]:


'''Take a look at some of the player performance stats'''

player_avg_performance = full_df[["nickname","score","rating", "winner", "game_duration_seconds"]].groupby("nickname").agg({"score":"mean", 
                                                                                                                            "rating":"mean", 
                                                                                                                            "winner":"sum", 
                                                                                                                            "nickname":"count",
                                                                                                                            "game_duration_seconds":"mean",
                                                                                                                            
                                                                                                                           })
player_avg_performance["win_ratio"] = player_avg_performance["winner"] / player_avg_performance["nickname"]


# In[62]:


plot = sns.displot(player_avg_performance, x="score", y="rating", kind="kde")
plot.fig.suptitle("Distribution of scores by ratings, averaged over each player")


# In[63]:


plot = sns.displot(player_avg_performance, x="win_ratio", y="rating", kind="kde")
plot.fig.suptitle("Distribution of the win ratio by ratings, averaged over each player")


# In[64]:


plot = sns.displot(player_avg_performance, x="game_duration_seconds", y="rating", kind="kde")
plot.fig.suptitle("Distribution of game duration by ratings, averaged over each player")


# In[65]:


'''Calculation of correlations of player stats with their ratings'''

print("Correlation between a players average score and their average rating {:.3f}".format(player_avg_performance["score"].corr(player_avg_performance["rating"])))
print("Correlation between a players number of wins and their average rating {:.3f}".format(player_avg_performance["score"].corr(player_avg_performance["winner"])))
print("Correlation between a players average win ratio and their average rating {:.3f}".format(player_avg_performance["score"].corr(player_avg_performance["win_ratio"])))
print("Correlation between a players average game duration and their average rating {:.3f}".format(player_avg_performance["score"].corr(player_avg_performance["game_duration_seconds"])))


# We observe some strong correlations between a player's average score as well as their winning ratio and their rating. Bascially, as a player scores more (and, presumably, wins more) they get a higehr rating. This make sense.

# ## 1(e) EDA of the Turns
# 
# For the Turns, lets investigate if there any patterns in the data in the turns relative to the rating variable. 

# In[66]:


def fe_turns(df):
    '''
    Function from https://www.kaggle.com/code/hasanbasriakcay/scrabble-eda-fe-modeling
    '''
    df["rack_len"] = df["rack"].str.len()
    df["rack_len_less_than_7"] = df["rack_len"].apply(lambda x : x <7)
    df["move_len"] = df["move"].str.len()
    df["move"].fillna("None",inplace=True)
    df["difficult_word"] = df["move"].apply(textstat.difficult_words)
    difficult_letters = ["Z", "Q", "J", "X", "K"]
    df["difficult_letters"] = df["move"].apply(lambda x: len([letter for letter in x if letter in difficult_letters]))
    medium_letters = ["F", "H", "V", "W", "Y", "B", "C", "M", "P"]
    df["medium_letters"] = df["move"].apply(lambda x: len([letter for letter in x if letter in medium_letters]))
    easy_letters = ["D", "G", "A", "E", "I", "O", "U", "L", "N", "S", "T", "R"]
    df["easy_letters"] = df["move"].apply(lambda x: len([letter for letter in x if letter in easy_letters]))
    df["points_per_letter"] = df["points"]/df["move_len"]
    
    df["turn_type"].fillna("None",inplace=True)
    turn_type_unique = df["turn_type"].unique()
    df = pd.get_dummies(df, columns=["turn_type"])
    dummy_features = [f"turn_type_{value}" for value in turn_type_unique]
    
    char_map = {
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
        'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15,
    }
    df['y'] = df["location"].str.extract('(\d+)')[0].values
    df['y'].fillna("0",inplace=True)
    df["y"] = df["y"].astype(int)
    
    df["x"] = df["location"].str.extract('([A-Z])')[0].values
    df["x"].replace(char_map, inplace=True)
    df['x'].fillna("0",inplace=True)
    df["x"] = df["x"].astype(int)
    
    df["direction_of_play"] = df["location"].apply(lambda x: 1 if str(x)[0].isdigit() else 0)
    
    df["curr_board_pieces_used"] = df["move"].apply(lambda x: str(x).count(".") + sum(int(c.islower()) for c in str(x)))
    
    return df, dummy_features


# In[67]:


turns_fe, dummy_features = fe_turns(turns.copy())
agg_func_counts = {feature:'sum' for feature in dummy_features}
turns_grouped_counts = turns_fe.groupby(["game_id", "nickname"], as_index=False).agg(agg_func_counts)

agg_func_stats = {
    "points":["mean", "max"],
    "move_len":["mean", "max"],
    "difficult_word":["mean", "sum"],
    "difficult_letters":["mean", "sum"],
    "medium_letters":["mean", "sum"],
    "easy_letters":["mean", "sum"],
    "points_per_letter":"mean",
    "curr_board_pieces_used": "mean",
    "direction_of_play": "mean",
    "rack_len_less_than_7" : "sum",
    "turn_number" : "count"
}
# Only take those turns where a play is made
turns_grouped_stats = turns_fe[turns_fe["turn_type_Play"]==1].groupby(["game_id", "nickname"], as_index=False).agg(agg_func_stats)
turns_grouped_stats.columns = ["_".join(a) if a[0] not in ["game_id", "nickname"] else a[0] for a in turns_grouped_stats.columns.to_flat_index()]
turns_grouped = turns_grouped_counts.merge(turns_grouped_stats, how="outer", on =["game_id", "nickname"])
# Fill in games where no play is ever done (about 46 of them)
turns_grouped.fillna(value=0, inplace=True)


# In[68]:


full_df = full_df.merge(turns_grouped, how="left", on=["game_id", "nickname"])


# In[69]:


plt.figure(figsize=(12,8))
sns.heatmap(full_df[['rating', 'bot_rating'] +list(turns_grouped.columns[2:])].corr(), cmap="vlag")
plt.show()


# So, it looks like both the turns points stats, move length stats and difficult word stats have some correaltion to player ratings. Thank you to [Hasan Basri Akçay](https://www.kaggle.com/hasanbasriakcay) for doing most of the work for these turn-based features

# # 2. Import and Preprocess Data
# 
# As part of the import, we'll also do some feature engineering based on insights from the EDA

# In[70]:


def create_turn_features(df):
    '''
    Function based on function from :
    https://www.kaggle.com/code/hasanbasriakcay/scrabble-eda-fe-modeling
    '''
    
    df["rack_len"] = df["rack"].str.len()
    df["rack_len_less_than_7"] = df["rack_len"].apply(lambda x : x <7)
    df["move_len"] = df["move"].str.len()
    df["move"].fillna("None",inplace=True)
    df["difficult_word"] = df["move"].apply(textstat.difficult_words)
    difficult_letters = ["Z", "Q", "J", "X", "K"]
    df["difficult_letters"] = df["move"].apply(lambda x: len([letter for letter in x if letter in difficult_letters]))
    medium_letters = ["F", "H", "V", "W", "Y", "B", "C", "M", "P"]
    df["medium_letters"] = df["move"].apply(lambda x: len([letter for letter in x if letter in medium_letters]))
    easy_letters = ["D", "G", "A", "E", "I", "O", "U", "L", "N", "S", "T", "R"]
    df["easy_letters"] = df["move"].apply(lambda x: len([letter for letter in x if letter in easy_letters]))
    df["points_per_letter"] = df["points"]/df["move_len"]
    
    df["turn_type"].fillna("None",inplace=True)
    turn_type_unique = df["turn_type"].unique()
    df = pd.get_dummies(df, columns=["turn_type"])
    dummy_features = [f"turn_type_{value}" for value in turn_type_unique]
    
    char_map = {
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
        'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15,
    }
    df['y'] = df["location"].str.extract('(\d+)')[0].values
    df['y'].fillna("0",inplace=True)
    df["y"] = df["y"].astype(int)
    
    df["x"] = df["location"].str.extract('([A-Z])')[0].values
    df["x"].replace(char_map, inplace=True)
    df['x'].fillna("0",inplace=True)
    df["x"] = df["x"].astype(int)
    
    df["direction_of_play"] = df["location"].apply(lambda x: 1 if str(x)[0].isdigit() else 0)
    
    df["curr_board_pieces_used"] = df["move"].apply(lambda x: str(x).count(".") + sum(int(c.islower()) for c in str(x)))
    
    agg_func_counts = {feature:'sum' for feature in dummy_features}
    turns_grouped_counts = df.groupby(["game_id", "nickname"], as_index=False).agg(agg_func_counts)

    agg_func_stats = {
        "points":["mean", "max"],
        "move_len":["mean", "max"],
        "difficult_word":["mean", "sum"],
        "difficult_letters":["mean", "sum"],
        "medium_letters":["mean", "sum"],
        "easy_letters":["mean", "sum"],
        "points_per_letter":"mean",
        "curr_board_pieces_used": "mean",
        "direction_of_play": "mean",
        "rack_len_less_than_7" : "sum",
        "turn_number" : "count"
    }
    
    # Only take those turns where a play is made
    turns_grouped_stats = df[df["turn_type_Play"]==1].groupby(["game_id", "nickname"], as_index=False).agg(agg_func_stats)
    turns_grouped_stats.columns = ["_".join(a) if a[0] not in ["game_id", "nickname"] else a[0] for a in turns_grouped_stats.columns.to_flat_index()]
    turns_grouped = turns_grouped_counts.merge(turns_grouped_stats, how="outer", on =["game_id", "nickname"])
    # Fill in games where no play is ever done (about 46 of them)
    turns_grouped.fillna(value=0, inplace=True)
    
    return turns_grouped


# In[71]:


# Wrapper function to read in, encode and impute missing values for the data

def load_data(bot_names =["BetterBot", "STEEBot", "HastyBot"], cat_features=[]):
    train = pd.read_csv(os.path.join(ROOT_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(ROOT_DIR, "test.csv"))
    turns = pd.read_csv(os.path.join(ROOT_DIR, "turns.csv"))
    games = pd.read_csv(os.path.join(ROOT_DIR, "games.csv"))
    
    # Merge the splits so we can process them together
    df = pd.concat([train, test])
    
    # Preprocessing
    
    # Add in turn features
    turns_fe_df = create_turn_features(turns)
    df = df.merge(turns_fe_df, how="left", on=["game_id", "nickname"])
    
    # Create the bot matrix
    bot_turns_columns = [i for i in turns_fe_df.columns.tolist() if i not in ["game_id", "nickname"]]
    bot_df = df[["game_id", "nickname", "score", "rating"]+bot_turns_columns].copy()
    bot_df['bot_name'] = bot_df['nickname'].apply(lambda x: x if x in bot_names else np.nan)
    bot_df = bot_df[["game_id", "bot_name", "score", "rating"]+bot_turns_columns].dropna(subset=["bot_name"])
    bot_df.columns = ["game_id", "bot_name", "bot_score", "bot_rating"]+["bot_"+i for i in bot_turns_columns]
    
    # Bring all of the data together
    df = df[~df['nickname'].isin(bot_names)] #take out the bots
    df = df.merge(bot_df, on="game_id") #add in bot information
    df = df.merge(games, on="game_id") # add in game information
    df["created_at"] = pd.to_datetime(df["created_at"]) #convert to datetime
    df["first"] = df["first"].apply(lambda x: 'bot' if x in bot_names else "player")
    
    # Create the binned labels of the players by their game preferences and add in those labels
    features_to_include = ['time_control_name', 'lexicon', 'rating_mode']
    game_types_df = df[['nickname']+features_to_include]
    game_types_df = game_types_df[['nickname']].join(ce.OneHotEncoder().fit_transform(game_types_df[features_to_include]))

    game_types_df = game_types_df.groupby("nickname").agg("sum")
    for col_set in [['time_control_name_1','time_control_name_2','time_control_name_3','time_control_name_4'],
                    ['lexicon_1','lexicon_2','lexicon_3','lexicon_4'], ['rating_mode_1','rating_mode_2']
                   ]:

        game_types_df[col_set] = game_types_df[col_set].div(game_types_df[col_set].sum(axis=1), axis=0)
    
    cluster = AgglomerativeClustering(n_clusters=9)
    game_types_df['game_cluster_label'] = cluster.fit_predict(game_types_df)
    df = df.merge(game_types_df['game_cluster_label'], how='left', left_on='nickname', right_index=True) #add in play type cluster
    
    # Create the binned labels of the players by their game results and add in those labels
    win_types_df = ce.OneHotEncoder().fit_transform(df['game_end_reason'])
    win_types_df['nickname'] = df['nickname']
    win_types_df['winner'] = df['winner']
    win_types_df['score'] = df['score']
    win_types_df['score_diff'] = df['score']-df['bot_score']

    win_types_df = win_types_df.groupby(['nickname']).agg({'score_diff':'mean', 'score': 'mean', 'game_end_reason_1': 'sum', 'game_end_reason_2': 'sum', 'game_end_reason_3':'sum', 
                                                           'game_end_reason_4': 'sum', 'winner': lambda x: np.sum(x==1)/len(x)})

    win_types_df[['score_diff', 'score']] = MinMaxScaler().fit_transform(win_types_df[['score_diff', 'score']]) #norm to 0,1
    col_set = ['game_end_reason_1', 'game_end_reason_2', 'game_end_reason_3', 'game_end_reason_4']
    win_types_df[col_set] = win_types_df[col_set].div(win_types_df[col_set].sum(axis=1), axis=0) #create distribution across types
    reducer = umap.UMAP()
    embedding = pd.DataFrame(reducer.fit_transform(win_types_df), columns=['x','y']) #reduce the dimension for better clusters
    cluster = AgglomerativeClustering(n_clusters=7)
    win_types_df['result_cluster_label'] = cluster.fit_predict(embedding)
    df = df.merge(win_types_df['result_cluster_label'], how='left', left_on='nickname', right_index=True) #add in play type cluster
    

    # Specify categorical variables
    for name in cat_features:
        df[name] = df[name].astype("category")
        # Add a None category for missing values
        if "None" not in df[name].cat.categories:
            df[name].cat.add_categories("None", inplace=True)

    # Reform splits
    train = df[df["game_id"].isin(train["game_id"])].set_index("game_id")
    test = df[df["game_id"].isin(test["game_id"])].set_index("game_id")
    return train, test


# In[72]:


'''Now, load in the data'''

train, test = load_data(cat_features = ["nickname","bot_name", "time_control_name", "first", "game_end_reason", "winner", "lexicon", "rating_mode"])


# # 3. Create a Baseline
# - Specify a baseline scoring function. We will use StratifiedGroupKFold splitting on the nicknames to try and better simulate the actual test set up. Namely, we have whole users excluded in the test dataset, but also want to make sure all the diffferent types of users, as described by the games they tend to play, are represented in the folds like they are in the test dataset.
# - Create a baseline model. Since the player performances were correlated with ratings, we'll start with that. We need to transform the data to match the scenario. <s> Namely, the rating for a user's game is their rating __prior__ to that game being played.</s> Namely, given a player's performance in a game, what would there rating have been.

# In[73]:


def score_dataset(X, y, 
                  model=LGBMRegressor(n_estimators=1000, verbose=-1, random_state=42)
                 ):
    
    X = X.copy()
    groups = X.pop('nickname')
    folds = StratifiedGroupKFold().split(X, X[['rating_mode', 'lexicon', 'time_control_name']].astype(str).agg('-'.join, axis=1), groups)
    
    scores = cross_validate(
        model, X, y, cv=folds, groups=groups, n_jobs=-1, scoring='neg_root_mean_squared_error', return_train_score=True
    )
    
    return {"Training":-1*np.mean(scores["train_score"]), "Validation":-1*np.mean(scores["test_score"])}


# In[74]:


base_features = ['nickname', 'score', 'turn_type_Play', 'turn_type_End', 'turn_type_Exchange', 'turn_type_Pass', 'turn_type_Timeout',
                 'turn_type_Challenge', 'turn_type_Six-Zero Rule', 'turn_type_None','points_mean', 'points_max', 'move_len_mean', 'move_len_max',
                 'difficult_word_mean', 'difficult_word_sum', 'difficult_letters_mean', 'difficult_letters_sum', 'points_per_letter_mean',
                 'curr_board_pieces_used_mean', 'direction_of_play_mean', 'rack_len_less_than_7_sum', 'turn_number_count', 'bot_name',
                 'bot_score', 'bot_rating', 'bot_turn_type_Play', 'bot_turn_type_End', 'bot_turn_type_Exchange', 'bot_turn_type_Pass', 'bot_turn_type_Timeout',
                 'bot_turn_type_Challenge', 'bot_turn_type_Six-Zero Rule', 'bot_turn_type_None', 'bot_points_mean', 'bot_points_max',
                 'bot_move_len_mean', 'bot_move_len_max', 'bot_difficult_word_mean', 'bot_difficult_word_sum', 'bot_difficult_letters_mean',
                 'bot_difficult_letters_sum', 'bot_points_per_letter_mean', 'bot_curr_board_pieces_used_mean', 'bot_direction_of_play_mean',
                 'bot_rack_len_less_than_7_sum', 'bot_turn_number_count', 'first', 'time_control_name', 'game_end_reason', 'winner',
                 'lexicon', 'initial_time_seconds', 'increment_seconds', 'rating_mode', 'max_overtime_minutes', 'game_duration_seconds',
                 'game_cluster_label', 'result_cluster_label']


# In[75]:


X = train[base_features].copy()
X = ce.OrdinalEncoder().fit_transform(X)
y = train["rating"]


# In[76]:


score_dataset(X, y)


# # 4. Featurize the Data
# - We'll start with taking a player's past performance into account
#     - We'll start with the games data. Let's also try breking these stats down by the bot_types and lexicons.
#     - We'll also try adding the game types across the player (i.e. how many times each player plays each lexicon, bot, etc.)
#     - We'll also try adding in some of the turn features we constructed earlier
# - Add in the game stats for the last game by the player, based on insights from the EDA (i.e. differences in game set up seem to affect playe rating, even for non-rated games)

# ## 4.a Creating cumulative player statistics
# 
# We'll try overall player statistics and break down statistics by various game types

# In[77]:


def create_cum_player_features_overall(df):
    '''
    Get the running average of player scores and win ratio over the course of all of their games up to the current rating
    '''
    
    df = df[["nickname", "created_at","score","winner", "game_duration_seconds", "bot_score"]]
    
    #sort by the times of the games so that we aggregate over time in the ensuing steps
    df= df.sort_values(by="created_at")

    #Initialize our new variables with 0's
    df["cum_avg_player_score"] = np.zeros(len(df))
    df["cum_max_player_score"] = np.zeros(len(df))
    df["cum_min_player_score"] = np.zeros(len(df))
    df["cum_total_player_score"] = np.zeros(len(df))
    df["cum_player_wins"] = np.zeros(len(df))
    df["cum_avg_player_win_ratio"] = np.zeros(len(df))
    df["cum_avg_game_duration_seconds"] = np.zeros(len(df))
    df["cum_avg_player_score_ratio"] = np.zeros(len(df))
    df["cum_total_player_score_difference"] = np.zeros(len(df))

    for nickname in df["nickname"].unique():
        '''
        Create the running averages of the player game features. Very important note with these, I am shifting the averages up by one ([:-1]) and
        adding in a starting zero. this is because 'expanding' takes into account the current value, and we do not actually know the current
        values of the game prior to it being played. We must remember that the rating for each game is the rating *prior* to that game being played.
        '''
        df.loc[df["nickname"]==nickname, "cum_avg_player_score"]= np.append(0, df[df["nickname"]==nickname]["score"].expanding(min_periods=1).mean().values[:-1])
        df.loc[df["nickname"]==nickname, "cum_max_player_score"]= np.append(0, df[df["nickname"]==nickname]["score"].expanding(min_periods=1).max().values[:-1])
        df.loc[df["nickname"]==nickname, "cum_min_player_score"]= np.append(0, df[df["nickname"]==nickname]["score"].expanding(min_periods=1).min().values[:-1])
        df.loc[df["nickname"]==nickname, "cum_total_player_score"]= np.append(0, df[df["nickname"]==nickname]["score"].expanding(min_periods=1).sum().values[:-1])
        
        df.loc[df["nickname"]==nickname, "cum_player_wins"]= np.append(0, df[df["nickname"]==nickname]["winner"].expanding(min_periods=1).apply(lambda x: np.sum(x==1)).values[:-1])
        
        df.loc[df["nickname"]==nickname, "cum_avg_player_win_ratio"]= \
        df[df["nickname"]==nickname]["cum_player_wins"] / np.append(0, df[df["nickname"]==nickname]["winner"].expanding(min_periods=1).count().values[:-1])
        
        df.loc[df["nickname"]==nickname, "cum_avg_game_duration_seconds"]= \
        np.append(0, df[df["nickname"]==nickname]["game_duration_seconds"].expanding(min_periods=1).mean().values[:-1])
        
        df.loc[df["nickname"]==nickname, "cum_avg_player_score_ratio"] =\
        np.append(0, (df[df["nickname"]==nickname]["score"] - df[df["nickname"]==nickname]["bot_score"]).expanding(min_periods=1).mean().values[:-1])
        
        df.loc[df["nickname"]==nickname, "cum_total_player_score_difference"] =\
        np.append(0, (df[df["nickname"]==nickname]["score"] - df[df["nickname"]==nickname]["bot_score"]).expanding(min_periods=1).sum().values[:-1])
        
    #fill in any missing values with 0
    df[["cum_avg_player_score", "cum_max_player_score", "cum_min_player_score", "cum_total_player_score", "cum_player_wins",  "cum_avg_player_win_ratio", "cum_avg_game_duration_seconds", "cum_avg_player_score_ratio", "cum_total_player_score_difference"]]\
    = df[["cum_avg_player_score", "cum_max_player_score", "cum_min_player_score", "cum_total_player_score", "cum_player_wins",  "cum_avg_player_win_ratio", "cum_avg_game_duration_seconds", "cum_avg_player_score_ratio", "cum_total_player_score_difference"]].fillna(0)
    
    # resort the data by the the index (i.e. game number)
    df = df.sort_index()
    
    return df[["cum_avg_player_score", "cum_max_player_score", "cum_min_player_score", "cum_total_player_score", "cum_player_wins",  "cum_avg_player_win_ratio", "cum_avg_game_duration_seconds", "cum_avg_player_score_ratio", "cum_total_player_score_difference"]]


# In[78]:


X = train[base_features].copy()
X = ce.OrdinalEncoder().fit_transform(X)
X = X.join(create_cum_player_features_overall(train.copy()))
y = train["rating"]


# In[79]:


score_dataset(X, y)


# In[80]:


'''Splitting the features between rated and casual games'''

X = train[base_features].copy()
X = ce.OrdinalEncoder().fit_transform(X)

feat_rated = create_cum_player_features_overall(train[train["rating_mode"]=='RATED'].copy())
feat_rated.columns = [i+"_rated" for i in feat_rated.columns]
feat_casual = create_cum_player_features_overall(train[train["rating_mode"]=='CASUAL'].copy())
feat_casual.columns = [i+"_casual" for i in feat_casual.columns]
df = reduce(lambda  left,right: pd.merge(left,right, left_index=True, right_index=True, how='outer'), 
           [feat_rated, feat_casual])
df['nickname'] = train['nickname'] # Add in the nicknames
df = df.groupby(['nickname']).fillna(method='ffill').fillna(0) #Fill forward from last valid game and zero otherwise
X = X.join(df)


y = train["rating"]


# In[81]:


score_dataset(X, y)


# In[82]:


'''Splitting the features between bot types'''

X = train[base_features].copy()
X = ce.OrdinalEncoder().fit_transform(X)

feat_betterbot = create_cum_player_features_overall(train[train["bot_name"]=='BetterBot'].copy())
feat_betterbot.columns = [i+"_betterbot" for i in feat_betterbot.columns]
feat_hastybot = create_cum_player_features_overall(train[train["bot_name"]=='HastyBot'].copy())
feat_hastybot.columns = [i+"_hastybot" for i in feat_hastybot.columns]
feat_steebot = create_cum_player_features_overall(train[train["bot_name"]=='STEEBot'].copy())
feat_steebot.columns = [i+"_steebot" for i in feat_steebot.columns]
df = reduce(lambda  left,right: pd.merge(left,right, left_index=True, right_index=True, how='outer'), 
           [feat_betterbot, feat_hastybot, feat_steebot])
df['nickname'] = train['nickname']
df = df.groupby(['nickname']).fillna(method='ffill').fillna(0)
X = X.join(df)


y = train["rating"]


# In[83]:


score_dataset(X, y)


# In[84]:


'''Splitting the features between lexicon'''

X = train[base_features].copy()
X = ce.OrdinalEncoder().fit_transform(X)

feat_CSW21 = create_cum_player_features_overall(train[train["lexicon"]=='CSW21'].copy())
feat_CSW21.columns = [i+"_CSW21" for i in feat_CSW21.columns]
feat_ECWL = create_cum_player_features_overall(train[train["lexicon"]=='ECWL'].copy())
feat_ECWL.columns = [i+"_ECWL" for i in feat_ECWL.columns]
feat_NSWL20 = create_cum_player_features_overall(train[train["lexicon"]=='NSWL20'].copy())
feat_NSWL20.columns = [i+"_NSWL20" for i in feat_NSWL20.columns]
feat_NWL20 = create_cum_player_features_overall(train[train["lexicon"]=='NWL20'].copy())
feat_NWL20.columns = [i+"_NWL20" for i in feat_NWL20.columns]
df = reduce(lambda  left,right: pd.merge(left,right, left_index=True, right_index=True, how='outer'), 
           [feat_CSW21, feat_ECWL, feat_NSWL20, feat_NWL20])
df['nickname'] = train['nickname']
df = df.groupby(['nickname']).fillna(method='ffill').fillna(0)
X = X.join(df)


y = train["rating"]


# In[85]:


score_dataset(X, y)


# In[86]:


'''Splitting the features between game types'''

X = train[base_features].copy()
X = ce.OrdinalEncoder().fit_transform(X)

feat_regular = create_cum_player_features_overall(train[train["time_control_name"]=='regular'].copy())
feat_regular.columns = [i+"_regular" for i in feat_regular.columns]
feat_rapid = create_cum_player_features_overall(train[train["time_control_name"]=='rapid'].copy())
feat_rapid.columns = [i+"_rapid" for i in feat_rapid.columns]
feat_blitz = create_cum_player_features_overall(train[train["time_control_name"]=='blitz'].copy())
feat_blitz.columns = [i+"_blitz" for i in feat_blitz.columns]
feat_ultrablitz = create_cum_player_features_overall(train[train["time_control_name"]=='ultrablitz'].copy())
feat_ultrablitz.columns = [i+"_ultrablitz" for i in feat_ultrablitz.columns]
df = reduce(lambda  left,right: pd.merge(left,right, left_index=True, right_index=True, how='outer'), 
           [feat_regular, feat_rapid, feat_blitz, feat_ultrablitz])
df['nickname'] = train['nickname']
df = df.groupby(['nickname']).fillna(method='ffill').fillna(0)
X = X.join(df)

y = train["rating"]


# In[87]:


score_dataset(X, y)


# It looks like cumulative features from player performance improves upon the baseline pretty significantly. However, breaking down the player performances by things like rated or casual games does not seem to help much, with the exception of splitting by lexicon. Perhaps this is because most players really just do one game-style, and so breaking down between game-styles doesn't change much.

# ## 4.b Creating cumulative game types by player
# 
# We'll try adding in some statistics on the types of games being played by each player

# In[88]:


def create_cum_player_game_features(df):
    '''
    Get the cumulative counts of bots, rating_modes, and lexicons by each player up to the current game
    '''
    
    df = df[["nickname", "created_at", "bot_name", "rating_mode", "lexicon", "game_end_reason"]]
    
    encoder = ce.OneHotEncoder(cols=["bot_name", "rating_mode", "lexicon", "game_end_reason"], use_cat_names=True)
    df = df.join(encoder.fit_transform(df[["bot_name", "rating_mode", "lexicon", "game_end_reason"]]))
    
    df= df.sort_values(by="created_at")
    
    for feature_name in encoder.get_feature_names():
        df["cum_"+str(feature_name)+"_counts"] = np.zeros(len(df))

    for nickname in df["nickname"].unique():
        for feature_name in encoder.get_feature_names():
            '''
            Create the running counts of the types of games by player. Very important note with these, I am shifting the averages up by one ([:-1]) and
            adding in a starting zero. this is because 'expanding' takes into account the current value, and we do not actually know the current
            values of the game prior to it being played. We must remember that the rating for each game is the rating *prior* to that game being played.
            '''
            df.loc[df["nickname"]==nickname, "cum_"+str(feature_name)+"_counts"]= \
            np.append(0, df[df["nickname"]==nickname][feature_name].expanding(min_periods=1).sum().values[:-1])

    for feature_name in encoder.get_feature_names():
        df["cum_"+str(feature_name)+"_counts"] = df["cum_"+str(feature_name)+"_counts"].fillna(0)
        
    df['cum_rated_game_ratio'] = df['cum_rating_mode_CASUAL_counts'] / (df['cum_rating_mode_CASUAL_counts'] + df['cum_rating_mode_RATED_counts'])
        
    df = df.sort_index()
    
    return df[df.columns.difference(["nickname", "created_at", "bot_name", "rating_mode", "lexicon", "game_end_reason"]+encoder.get_feature_names())]


# In[89]:


'''Adding in Base features'''

X = train[base_features].copy()
X = ce.OrdinalEncoder().fit_transform(X)
X = X.join(create_cum_player_game_features(train.copy()))
y = train["rating"]


# In[90]:


score_dataset(X, y)


# In[91]:


'''
Try adding the game type features with player performance features
'''

X = train[base_features].copy()
X = ce.OrdinalEncoder().fit_transform(X)
X = X.join(create_cum_player_game_features(train.copy()))

feat_CSW21 = create_cum_player_features_overall(train[train["lexicon"]=='CSW21'].copy())
feat_CSW21.columns = [i+"_CSW21" for i in feat_CSW21.columns]
feat_ECWL = create_cum_player_features_overall(train[train["lexicon"]=='ECWL'].copy())
feat_ECWL.columns = [i+"_ECWL" for i in feat_ECWL.columns]
feat_NSWL20 = create_cum_player_features_overall(train[train["lexicon"]=='NSWL20'].copy())
feat_NSWL20.columns = [i+"_NSWL20" for i in feat_NSWL20.columns]
feat_NWL20 = create_cum_player_features_overall(train[train["lexicon"]=='NWL20'].copy())
feat_NWL20.columns = [i+"_NWL20" for i in feat_NWL20.columns]
df = reduce(lambda  left,right: pd.merge(left,right, left_index=True, right_index=True, how='outer'), 
           [feat_CSW21, feat_ECWL, feat_NSWL20, feat_NWL20])
df['nickname'] = train['nickname']
df = df.groupby(['nickname']).fillna(method='ffill').fillna(0)
X = X.join(df)

y = train["rating"]


# In[92]:


score_dataset(X, y)


# So, it looks like doing cumulative averages of the player stats and combining that with game-type variable counts, and then add in the base features gets a small performance improvement.

# ## 4.c Creating cummulative bot statistics against each player
# 
# We'll try adding in the cumulative bot features, by player, for each of the games

# In[93]:


def create_cum_bot_features(df):
    '''
    Get the running average of bot ratings and scores, broken down by bot, for each player for each of their gammes
    '''

    df= df[["nickname", "created_at","bot_name", "bot_score", "bot_rating", "winner"]]
    df['score_rating_ratio'] = df['bot_score']/df['bot_rating']

    df= df.sort_values(by="created_at")

    for bot_name in df["bot_name"].unique():
        df["cum_avg_bot_score_"+str(bot_name)] = np.zeros(len(df))
        df["cum_avg_bot_rating_"+str(bot_name)] = np.zeros(len(df))
        df["cum_min_bot_rating_"+str(bot_name)] = np.zeros(len(df))
        df["cum_max_bot_rating_"+str(bot_name)] = np.zeros(len(df))
        df["cum_avg_bot_wins_"+str(bot_name)] = np.zeros(len(df))
        df["cum_avg_bot_win_ratio_"+str(bot_name)] = np.zeros(len(df))
        df["cum_avg_bot_score_rating_ratio_"+str(bot_name)] = np.zeros(len(df))


    for nickname in df["nickname"].unique():
        for bot_name in df["bot_name"].unique():
            '''
            Create the running averages of bot performances, by player, and by bot. Very important note with these, I am shifting the averages up by one ([:-1]) and
            adding in a starting zero. this is because 'expanding' takes into account the current value, and we do not actually know the current
            values of the game prior to it being played. We must remember that the rating for each game is the rating *prior* to that game being played.
            This, however, does not apply to the bot rating, which we do know before the game is played (ratings are known before the game is played!)
            '''
            df.loc[(df["nickname"]==nickname) & (df["bot_name"]==bot_name), "cum_avg_bot_score_"+str(bot_name)]= \
            np.append(0, df[(df["nickname"]==nickname) & (df["bot_name"]==bot_name)]["bot_score"].expanding(min_periods=1).mean().values[:-1])
            
            df.loc[(df["nickname"]==nickname) & (df["bot_name"]==bot_name), "cum_avg_bot_rating_"+str(bot_name)]= \
            df[(df["nickname"]==nickname) & (df["bot_name"]==bot_name)]["bot_rating"].expanding(min_periods=1).mean().values
            
            df.loc[(df["nickname"]==nickname) & (df["bot_name"]==bot_name), "cum_min_bot_rating_"+str(bot_name)]= \
            df[(df["nickname"]==nickname) & (df["bot_name"]==bot_name)]["bot_rating"].expanding(min_periods=1).min().values
            
            df.loc[(df["nickname"]==nickname) & (df["bot_name"]==bot_name), "cum_max_bot_rating_"+str(bot_name)]= \
            df[(df["nickname"]==nickname) & (df["bot_name"]==bot_name)]["bot_rating"].expanding(min_periods=1).max().values
            
            df.loc[(df["nickname"]==nickname) & (df["bot_name"]==bot_name), "cum_avg_bot_wins_"+str(bot_name)]= \
            np.append(0, df[(df["nickname"]==nickname) & (df["bot_name"]==bot_name)]["winner"].expanding(min_periods=1).apply(lambda x: np.sum(x==0)).values[:-1])
            
            df.loc[(df["nickname"]==nickname) & (df["bot_name"]==bot_name), "cum_avg_bot_win_ratio_"+str(bot_name)]= \
            df.loc[(df["nickname"]==nickname) & (df["bot_name"]==bot_name), "cum_avg_bot_wins_"+str(bot_name)]/ np.append(0, df[(df["nickname"]==nickname) & (df["bot_name"]==bot_name)]["winner"].expanding(min_periods=1).count().values[:-1])

            df.loc[(df["nickname"]==nickname) & (df["bot_name"]==bot_name), "cum_avg_bot_score_rating_ratio_"+str(bot_name)]= \
            np.append(0, df[(df["nickname"]==nickname) & (df["bot_name"]==bot_name)]['score_rating_ratio'].expanding(min_periods=1).mean().values[:-1])
            
    for bot_name in df["bot_name"].unique():
        df[["cum_avg_bot_score_"+str(bot_name), "cum_avg_bot_rating_"+str(bot_name), "cum_min_bot_rating_"+str(bot_name), "cum_max_bot_rating_"+str(bot_name), "cum_avg_bot_wins_"+str(bot_name), "cum_avg_bot_win_ratio_"+str(bot_name), "cum_avg_bot_score_rating_ratio_"+str(bot_name)]] = \
        df[["cum_avg_bot_score_"+str(bot_name), "cum_avg_bot_rating_"+str(bot_name), "cum_min_bot_rating_"+str(bot_name), "cum_max_bot_rating_"+str(bot_name), "cum_avg_bot_wins_"+str(bot_name), "cum_avg_bot_win_ratio_"+str(bot_name), "cum_avg_bot_score_rating_ratio_"+str(bot_name)]].fillna(0)
    
    df = df.sort_index()
    
    return df[df.columns.difference(["nickname", "created_at","bot_name", "bot_score", "bot_rating", "winner", 'score_rating_ratio'])]


# In[94]:


X = train[base_features].copy()
X = ce.OrdinalEncoder().fit_transform(X)
X = X.join(create_cum_bot_features(train.copy()))
y = train["rating"]


# In[95]:


score_dataset(X, y)


# In[96]:


'''
Try adding the priorly found good player features to the bot features
'''

X = train[base_features].copy()
X = ce.OrdinalEncoder().fit_transform(X)
X = X.join(create_cum_bot_features(train.copy()))

feat_CSW21 = create_cum_player_features_overall(train[train["lexicon"]=='CSW21'].copy())
feat_CSW21.columns = [i+"_CSW21" for i in feat_CSW21.columns]
feat_ECWL = create_cum_player_features_overall(train[train["lexicon"]=='ECWL'].copy())
feat_ECWL.columns = [i+"_ECWL" for i in feat_ECWL.columns]
feat_NSWL20 = create_cum_player_features_overall(train[train["lexicon"]=='NSWL20'].copy())
feat_NSWL20.columns = [i+"_NSWL20" for i in feat_NSWL20.columns]
feat_NWL20 = create_cum_player_features_overall(train[train["lexicon"]=='NWL20'].copy())
feat_NWL20.columns = [i+"_NWL20" for i in feat_NWL20.columns]
df = reduce(lambda  left,right: pd.merge(left,right, left_index=True, right_index=True, how='outer'), 
           [feat_CSW21, feat_ECWL, feat_NSWL20, feat_NWL20])
df['nickname'] = train['nickname']
df = df.groupby(['nickname']).fillna(method='ffill').fillna(0)
X = X.join(df)

y = train["rating"]


# In[97]:


score_dataset(X, y)


# Adding in the cumulative bot stuff seems to improve performance. In particular, Adding all previous features with the bot features seems to improve performance

# ## 4.d Creating cumulative Features from the Turn-based Features
# 
# We'll try adding in the cumulative bot features, by player, for each of the games

# In[98]:


def create_cum_turns_features(df):
    turn_features = ['turn_type_Play', 'turn_type_End',
       'turn_type_Exchange', 'turn_type_Pass', 'turn_type_Timeout',
       'turn_type_Challenge', 'turn_type_Six-Zero Rule', 'turn_type_None',
       'points_mean', 'points_max', 'move_len_mean', 'move_len_max',
       'difficult_word_mean', 'difficult_word_sum', 'difficult_letters_mean',
       'difficult_letters_sum', 'points_per_letter_mean',
       'curr_board_pieces_used_mean', 'direction_of_play_mean',
       'rack_len_less_than_7_sum', 'turn_number_count']
    
    # Create some features looking at the difference in performance between player and bot
    df['play_counts_diff'] = df['turn_type_Play'] - df['bot_turn_type_Play']
    df['avg_points_diff'] = df['points_mean'] - df['bot_points_mean']
    df['avg_move_len_diff'] = df['move_len_mean'] - df['bot_move_len_mean']
    df['avg_points_per_letter_diff'] = df['points_per_letter_mean'] - df['bot_points_per_letter_mean']
    df['difficult_words_count_diff'] = df['difficult_word_sum'] - df['bot_difficult_word_sum']
    df['difficult_letters_count_diff'] = df['difficult_letters_sum'] - df['bot_difficult_letters_sum']
    
    df = df[["nickname", "created_at", 'play_counts_diff', 'avg_points_diff', 'avg_move_len_diff',
            'avg_points_per_letter_diff', 'difficult_words_count_diff', 'difficult_letters_count_diff']+turn_features]
    
    df= df.sort_values(by="created_at")
    
    for nickname in df["nickname"].unique():
        for feature_name in turn_features:
            '''
            Very important note with these, I am shifting the averages up by one ([:-1]) and
            adding in a starting zero. this is because 'expanding' takes into account the current value, and we do not actually know the current
            values of the game prior to it being played. We must remember that the rating for each game is the rating *prior* to that game being played.
            '''
            df.loc[df["nickname"]==nickname, "cum_"+str(feature_name)+"_average"]= \
            np.append(0, df[df["nickname"]==nickname][feature_name].expanding(min_periods=1).mean().values[:-1])

    for feature_name in turn_features:
        df["cum_"+str(feature_name)+"_average"] = df["cum_"+str(feature_name)+"_average"].fillna(0)
    
    df = df.sort_index()
    
    return df[df.columns.difference(["nickname", "created_at"]+turn_features)]


# In[99]:


X = train[base_features].copy()
X = ce.OrdinalEncoder().fit_transform(X)
X = X.join(create_cum_turns_features(train.copy()))
y = train["rating"]


# In[100]:


score_dataset(X, y)


# In[101]:


'''
Try adding the priorly found good player features to the Turn-based features
'''

X = train[base_features].copy()
X = ce.OrdinalEncoder().fit_transform(X)
X = X.join(create_cum_turns_features(train.copy()))

feat_CSW21 = create_cum_player_features_overall(train[train["lexicon"]=='CSW21'].copy())
feat_CSW21.columns = [i+"_CSW21" for i in feat_CSW21.columns]
feat_ECWL = create_cum_player_features_overall(train[train["lexicon"]=='ECWL'].copy())
feat_ECWL.columns = [i+"_ECWL" for i in feat_ECWL.columns]
feat_NSWL20 = create_cum_player_features_overall(train[train["lexicon"]=='NSWL20'].copy())
feat_NSWL20.columns = [i+"_NSWL20" for i in feat_NSWL20.columns]
feat_NWL20 = create_cum_player_features_overall(train[train["lexicon"]=='NWL20'].copy())
feat_NWL20.columns = [i+"_NWL20" for i in feat_NWL20.columns]
df = reduce(lambda  left,right: pd.merge(left,right, left_index=True, right_index=True, how='outer'), 
           [feat_CSW21, feat_ECWL, feat_NSWL20, feat_NWL20])
df['nickname'] = train['nickname']
df = df.groupby(['nickname']).fillna(method='ffill').fillna(0)
X = X.join(df)

y = train["rating"]


# In[102]:


score_dataset(X, y)


# Adding in the cumulative turn features does seem to improve things a little.

# # 4.e Add in results from last game played
# 
# Add in the game results from the last game played by a player

# In[103]:


def create_previous_game_features(df, features_to_include = ['rating_mode','lexicon','bot_name','time_control_name', 'score']):
    '''
    Add in the base features from the last game
    '''
    df= df.sort_values(by="created_at")
    time_diff = df.groupby("nickname")["created_at"].shift(periods=0) - df.groupby("nickname")["created_at"].shift(periods=1)
    df = df.groupby("nickname")[features_to_include].shift(periods=1)
    df = df.add_suffix("_prev_game")
    # Get the time between the last game a player played and the current one
    df["time_between_games"] = time_diff.dt.total_seconds().fillna(0)
    df = df.fillna(value = {"score":0})
    df = df.sort_index()
    
    return df


# In[104]:


X = train[base_features].copy()
X = X.join(create_previous_game_features(train.copy()))
X = ce.OrdinalEncoder().fit_transform(X)


# In[105]:


score_dataset(X, y)


# Adding in the features from the last game barely improves performance, if at all.

# # 5. Finalize Features for Final Model

# In[106]:


def create_features(df, df_test=None):
    X_raw = df.copy()
    y = df['rating'].copy()
    
    enc = ce.m_estimate.MEstimateEncoder(cols=["bot_name", "time_control_name", "first", "game_end_reason", "winner", "lexicon", "rating_mode"])
    X_enc = enc.fit_transform(X_raw, y)
    
    if df_test is not None:
        X_test = df_test.copy()
        X_test_enc = enc.transform(X_test)
        X_enc = pd.concat([X_enc, X_test_enc])
        X_raw = pd.concat([X_raw, X_test])
        
    # Add in engineered features
    X = X_raw[base_features].copy()
    #X = X.join(create_previous_game_features(X_raw.copy()))
    X = ce.OrdinalEncoder().fit_transform(X)
    
    feat_CSW21 = create_cum_player_features_overall(train[train["lexicon"]=='CSW21'].copy())
    feat_CSW21.columns = [i+"_CSW21" for i in feat_CSW21.columns]
    feat_ECWL = create_cum_player_features_overall(train[train["lexicon"]=='ECWL'].copy())
    feat_ECWL.columns = [i+"_ECWL" for i in feat_ECWL.columns]
    feat_NSWL20 = create_cum_player_features_overall(train[train["lexicon"]=='NSWL20'].copy())
    feat_NSWL20.columns = [i+"_NSWL20" for i in feat_NSWL20.columns]
    feat_NWL20 = create_cum_player_features_overall(train[train["lexicon"]=='NWL20'].copy())
    feat_NWL20.columns = [i+"_NWL20" for i in feat_NWL20.columns]
    df = reduce(lambda  left,right: pd.merge(left,right, left_index=True, right_index=True, how='outer'), 
               [feat_CSW21, feat_ECWL, feat_NSWL20, feat_NWL20])
    df['nickname'] = train['nickname']
    df = df.groupby(['nickname']).fillna(method='ffill').fillna(0)
    X = X.join(df)

    X = X.join(create_cum_turns_features(X_raw.copy()))
    X = X.join(create_cum_player_game_features(X_raw.copy()))
    X = X.join(create_cum_bot_features(X_raw.copy()))

    # Add in the differently encoded features
    X[["bot_name", "time_control_name", "first", "game_end_reason", "winner", "lexicon", "rating_mode"]] =\
    X_enc[["bot_name", "time_control_name", "first", "game_end_reason", "winner", "lexicon", "rating_mode"]]
    
    # Reform splits
    if df_test is not None:
        X_test = X.loc[df_test.index, :]
        X.drop(df_test.index, inplace=True)
    

    if df_test is not None:
        return X, X_test
    else:
        return X


# In[107]:


X, X_test = create_features(train, test)
y = train['rating'].copy()


# In[108]:


score_dataset(X, y)


# That's the ticket - this looks like a decent featurization of the data 🙌

# # 6. Hyperparameter Tuning

# In[109]:


def objective(trial, X, y):
    # Specify a search space using distributions across plausible values of hyperparameters.
    param = {
        "objective": "regression",
        "verbosity": -1,
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 512),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 0, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
    }
    
    # Run LightGBM CV for the hyperparameter values
    groups = X.pop('nickname')
    folds = StratifiedGroupKFold(shuffle=True).split(X, X[['rating_mode', 'lexicon', 'time_control_name']].astype(str).agg('-'.join, axis=1), groups)
    lgbcv = lgb.cv(param,
                   lgb.Dataset(X, label=y),
                   folds= folds,
                   num_boost_round=1000,
                   callbacks =[log_evaluation(period=0)]                                  
                  )
    cv_score = lgbcv['l2-mean'][-1]
    
    # Return metric of interest averaged across the CV folds
    return cv_score


# In[110]:


optuna.logging.set_verbosity(optuna.logging.WARNING) 
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, X.copy(), y.copy()), timeout=1800, n_trials=200) 


# In[111]:


print(study.best_params)


# In[112]:


print(study.best_value**0.5)


# # 7. Fit final model and make predictions
# 
# We are going to fit more than one model and just use averaging to ensemble between them.

# In[113]:


test_preds = []
train_preds = []
scores = []
groups = X.pop('nickname') #remove the player nicknames from the train set and make them groups for CV
test_groups = X_test.pop('nickname') #remove the player nicknames from the test set

for repeat in range(4):
    skf = StratifiedGroupKFold(shuffle=True).split(X, X[['rating_mode', 'lexicon', 'time_control_name']].astype(str).agg('-'.join, axis=1), groups)
    for fold_idx, (train_index, valid_index) in enumerate(skf):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
        lgb_params = {
            'objective': 'regression',
            'verbose': -1,
            'n_estimators': 1000,
            **study.best_params
        }
        model = lgb.train(lgb_params, lgb_train, valid_sets=lgb_eval, callbacks=[lgb.log_evaluation(0)])

        y_pred = model.predict(X_valid)
        score = mean_squared_error(y_valid, y_pred, squared=False)
        scores.append(score)
        print("Fold {} MSE Score: {}".format(fold_idx, score))
        print("----------------------")
        test_preds.append( model.predict(X_test))
        train_preds.append( model.predict(X))


# In[114]:


print("Avg. and Std. of Validation MSE's: {} +/- {}".format(np.mean(scores), np.std(scores)))


# In[115]:


'''
Use stacking with a simple linear regession model to combine the different predictions
'''

final_estimator= LinearRegression()
final_estimator.fit(np.array(train_preds).transpose(), y)
final_test_preds = final_estimator.predict(np.array(test_preds).transpose())
final_train_preds = final_estimator.predict(np.array(train_preds).transpose())

print("R^2 value for stacking regressor: {}".format(final_estimator.score(np.array(train_preds).transpose(), y)))


# In[116]:


print("Overall Train MSE: {}".format(mean_squared_error(final_train_preds, train["rating"], squared=False)))


# In[117]:


'''Take a look at the distribution of the produced ratings versus the given ratings'''

fig, axs = plt.subplots(2, 2, sharey=True, figsize=(20,8))
sns.histplot(train['rating'], ax=axs[0,0], stat="density")
axs[0,0].set_title("Distribution of Train Ratings")
sns.histplot(final_train_preds , ax=axs[0,1], stat="density")
axs[0,1].set_title("Distribution of Predicted Ratings on Train")
sns.histplot(final_test_preds , ax=axs[1,0], stat="density")
axs[1,0].set_title("Distribution of Predicted Ratings on Test")
plt.show()


# In[118]:


# Create the submission
test['rating'] = final_test_preds
submission = test['rating']
submission.to_csv("submission.csv")


# In[ ]:




