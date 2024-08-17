#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Hey, thanks for viewing my Kernel!
# 
# If you like my work, please, leave an upvote: it will be really appreciated and it will motivate me in offering more content to the Kaggle community ! :)

# In[1]:


get_ipython().system('pip install textstat')


# In[2]:


get_ipython().system('pip install featdist')


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
import datetime as dt
import math

from featdist import numerical_ttt_dist
from featdist import categorical_ttt_dist
import textstat

np.random.seed(0)
warnings.simplefilter("ignore")


# In[4]:


train = pd.read_csv("../input/scrabble-player-rating/train.csv")
test = pd.read_csv("../input/scrabble-player-rating/test.csv")
turns = pd.read_csv("../input/scrabble-player-rating/turns.csv")
games = pd.read_csv("../input/scrabble-player-rating/games.csv")
sub = pd.read_csv("../input/scrabble-player-rating/sample_submission.csv")

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
print("sub shape:", sub.shape)


# In[6]:


print("train nan value sum:", train.isna().sum().sum())
print("test nan value sum:", test.isna().sum().sum())
print("turns nan value sum:", turns.isna().sum().sum())
print("games nan value sum:", games.isna().sum().sum())


# In[7]:


turns.isna().sum()


# In[8]:


print("train dublicated value sum:", train.duplicated().sum().sum())
print("test dublicated value sum:", test.duplicated().sum().sum())
print("turns dublicated value sum:", turns.duplicated().sum().sum())
print("games dublicated value sum:", games.duplicated().sum().sum())


# # Exploratory Data Analysis

# ## Feature Engineering

# In[9]:


def fe_turns(df):
    df["rack_len"] = df["rack"].str.len()
    df["move_len"] = df["move"].str.len()
    df["move"].fillna("None",inplace=True)
    df["difficult_word"] = df["move"].apply(textstat.difficult_words)
    
    # FE ref: https://www.kaggle.com/code/ijcrook/full-walkthrough-eda-fe-model-tuning
    df["rack_len_less_than_7"] = df["rack_len"].apply(lambda x : x <7)
    rare_letters = ["Z", "Q", "J", "X", "K", "V", "Y", "W", "G"]
    df["difficult_letters"] = df["move"].apply(lambda x: len([letter for letter in x if letter in rare_letters]))
    df["points_per_letter"] = df["points"]/df["move_len"]
    df["direction_of_play"] = df["location"].apply(lambda x: 1 if str(x)[0].isdigit() else 0)
    df["curr_board_pieces_used"] = df["move"].apply(lambda x: str(x).count(".") + sum(int(c.islower()) for c in str(x)))
    #
    
    df["turn_type"].fillna("None",inplace=True)
    turn_type_unique = df["turn_type"].unique()
    df = pd.get_dummies(df, columns=["turn_type"])
    dummy_features = [f"turn_type_{value}" for value in turn_type_unique]
    
    char_map = {
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
        'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15,
    }
    df["loc_nan"] = df["location"].isna()
    df['y'] = df["location"].str.extract('(\d+)')[0].values
    df['y'].fillna("0",inplace=True)
    df["y"] = df["y"].astype(int)
    
    df["x"] = df["location"].str.extract('([A-Z])')[0].values
    df["x"].replace(char_map, inplace=True)
    df['y'].fillna("0",inplace=True)
    df["y"] = df["y"].astype(int)
    
    return df, dummy_features


# In[10]:


turns_fe, dummy_features = fe_turns(turns.copy())
turns_fe["game_count"] = turns_fe["game_id"]

agg_features = ["points", "score", "rack_len", "move_len", "difficult_word", "loc_nan", "x", "y", 
                 "rack_len_less_than_7", "difficult_letters", "points_per_letter", "direction_of_play", 
                 "curr_board_pieces_used"] + dummy_features
agg_func = {feature:['mean', 'sum', 'max'] for feature in agg_features}
agg_func["game_count"] = "count"
    
turns_group = turns_fe.groupby(["game_id", "nickname"], as_index=False).agg(agg_func)
turns_group.columns = ['_'.join(col) for col in turns_group.columns]
turns_group_bot = turns_group[turns_group["nickname_"].isin(["BetterBot", "STEEBot", "HastyBot"])]
turns_group_bot.columns = [col+'_bot' for col in turns_group_bot.columns]
turns_group_player = turns_group[~turns_group["nickname_"].isin(["BetterBot", "STEEBot", "HastyBot"])]
print("turns_group_bot.shape:",turns_group_bot.shape)
print("turns_group_player.shape:",turns_group_player.shape)
turns_group_all = pd.concat([turns_group_player.reset_index(), turns_group_bot.reset_index()], axis=1, ignore_index=False)
turns_group_all.drop("index", axis=1, inplace=True)
print("turns_group_all.shape:",turns_group_all.shape)
turns_group_all.head()


# In[11]:


def feature_agg(df, turns_group, games):
    df_bot = df[df["nickname"].isin(["BetterBot", "STEEBot", "HastyBot"])]
    
    df["nickname_count"] = df["nickname"].replace(df["nickname"].value_counts())
    df = df.merge(turns_group, left_on=["game_id", "nickname"], right_on=["game_id_", "nickname_"], how='inner', suffixes=["", "_agg"])
    df = df.merge(df_bot, left_on=["game_id"], right_on=["game_id"], how='left', suffixes=["", "_bot"])
    df.drop(["game_id_", "nickname_", "nickname_bot", "game_id__bot"], axis=1,inplace=True)
    
    games["created_at"] = pd.to_datetime(games["created_at"])
    df = df.merge(games, left_on="game_id", right_on='game_id', how='left')
    df["first"] = df["first"] == df["nickname"]
    df["first"] = df["first"].astype(int)
    df["winner"] = df["winner"] == df["first"]
    df["winner"] = df["winner"].astype(int)
    return df


# In[12]:


train_fe = feature_agg(train, turns_group_all, games)
test_fe = feature_agg(test, turns_group_all, games)
train_fe.head()


# In[13]:


bad_features = []
for f in train_fe.columns:
    if train_fe[f].nunique() < 2:
        bad_features.append(f)

print("bad_features len:",len(bad_features))
train_fe.drop(bad_features, axis=1, inplace=True)
test_fe.drop(bad_features, axis=1, inplace=True)


# In[14]:


train_fe.fillna(0, inplace=True)
test_fe.fillna(0, inplace=True)


# In[15]:


train_fe.to_feather("train_fe.feather")
test_fe.to_feather("test_fe.feather")


# ## Distributions

# In[16]:


fig, ax = plt.subplots(figsize=(16,8))
sns.histplot(data=train_fe, x="rating", ax=ax);
ax.axvline(1500, color="tab:red", linewidth=5, linestyle="--");
ax.text(1510, 5000, "Anomaly", fontsize=20,color="tab:red");
ax.set_title("Target Hist");


# In[17]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'uint8']
num_features = train_fe.select_dtypes(include=numerics).columns.tolist()
num_features.remove("rating")
bool_features = train_fe.select_dtypes(include=["bool"]).columns.tolist()
cat_features = train_fe.select_dtypes(include=["object"]).columns.tolist()

print("num_features len:",len(num_features))
print("bool_features len:",len(bool_features))
print("cat_features len:",len(cat_features))


# In[18]:


NBINS = 11
num_features_bins = []
less_bins = []
for f in num_features:
    if train_fe[f].nunique() >= NBINS:
        num_features_bins.append(f)
    else:
        less_bins.append(f)
print("num_features_bins len:",len(num_features_bins))
print("less_bins:", less_bins)


# In[19]:


df_stats = numerical_ttt_dist(train=train_fe, test=test_fe, features=num_features_bins, target='rating', ncols=5, nbins=NBINS)
df_stats


# * nickname_count is not a well distributed feature

# In[20]:


df_stats = categorical_ttt_dist(train=train_fe, test=test_fe, features=cat_features, target='rating', ncols=3)
df_stats


# # Validation

# In[21]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import optuna

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[22]:


train_fe = pd.read_feather("../input/advanced-dataset/scrabble/train_fe.feather")
test_fe = pd.read_feather("../input/advanced-dataset/scrabble/test_fe.feather")

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'uint8']
num_features = train_fe.select_dtypes(include=numerics).columns.tolist()
num_features.remove("game_id")
num_features.remove("rating")

# Less correlated dist between the train and the test
#num_features.remove("initial_time_seconds")
#num_features.remove("rating_bot")
#num_features.remove("nickname_count")

bool_features = train_fe.select_dtypes(include=["bool"]).columns.tolist()
cat_features = train_fe.select_dtypes(include=["object"]).columns.tolist()
cat_features.remove("nickname")
cat_features.remove("nickname__bot")
#features = num_features + cat_features + bool_features
features = num_features + cat_features

print("num_features len:",len(num_features))
print("bool_features len:",len(bool_features))
print("cat_features len:",len(cat_features))

X = train_fe[features]
y = train_fe[["rating"]]
X_test = test_fe[features]

for feature in cat_features:
    X[feature] = X[feature].replace(X[feature].value_counts(normalize=True))
    X_test[feature] = X_test[feature].replace(X_test[feature].value_counts(normalize=True))
    
print("X.shape:", X.shape)
print("y.shape:", y.shape)
print("X_test.shape:", X_test.shape)


# In[23]:


def get_scores(model_dict, X, y, X_test, nfolds=5):
    df_score_details = {
        'model':[],
        'r2':[],
        'rmse':[],
    }
    fig, axes = plt.subplots(len(model_dict), figsize=(16, 4*len(model_dict)))
    for fig_index, model_key in enumerate(model_dict.keys()):
        kf = KFold(n_splits=nfolds)
        np_test = np.zeros((len(X_test)))
        val_r2_scores = []
        val_rmse_scores = []
        for fold, (train_index, val_index) in enumerate(kf.split(X)):
            X_train, X_val = X.loc[train_index,:], X.loc[val_index,:]
            y_train, y_val = y.loc[train_index,:], y.loc[val_index,:]
            
            model = model_dict[model_key]
            model.fit(X_train, y_train)
            
            val_preds = model.predict(X_val).reshape(-1)
            test_preds = model.predict(X_test).reshape(-1)
            
            val_r2_scores.append(r2_score(y_val, val_preds))
            val_rmse_scores.append(mean_squared_error(y_val, val_preds, squared=False))
            
            np_test += test_preds / nfolds
        df_score_details["model"].append(model_key)
        df_score_details["r2"].append(np.mean(val_r2_scores))
        df_score_details["rmse"].append(np.mean(val_rmse_scores))
        
        sub = pd.read_csv("../input/scrabble-player-rating/sample_submission.csv")
        sub.loc[sub["rating"].isna(), "rating"] = np_test
        sub.to_csv(f"{model_key}_baseline_sub.csv",index=False)
        
        axes[fig_index].hist(y, color='tab:blue', label='train', bins=50, density=True, alpha=0.5)
        axes[fig_index].hist(sub["rating"], color='tab:red', label='test', bins=50, density=True, alpha=0.5)
        axes[fig_index].set_title(model_key)
        axes[fig_index].legend()
    df_score = pd.DataFrame(df_score_details)
    gc.collect()
    plt.show()
    return df_score


# In[24]:


model_dict = {
    'lr':LinearRegression(),
    'ridge':Ridge(),
    'dt':DecisionTreeRegressor(),
    'rf':RandomForestRegressor(),
    'lgb':lgb.LGBMRegressor(),
}
#df_score = get_scores(model_dict, X, y, X_test, nfolds=5)


# In[25]:


df_score_details = {'model': {0: 'lr', 1: 'ridge', 2: 'dt', 3: 'rf', 4: 'lgb'},
 'r2': {0: 0.6130722539459688,
  1: 0.6130003820326145,
  2: 0.5079049576294109,
  3: 0.7554189363524964,
  4: 0.7482953578278093},
 'rmse': {0: 142.86110682061616,
  1: 142.87432546020736,
  2: 161.09238311693016,
  3: 113.56680458868478,
  4: 115.20888248520865},
  'lb':{0:160.95957,
  1: 161.01134,
  2: 172.17746,
  3: 157.34657,
  4: 156.36289}}
df_score = pd.DataFrame(df_score_details)
df_score.sort_values("lb",inplace=True,ascending=True)
df_score


# # Modeling

# ## Hyperparameter Tuning

# In[26]:


def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.01,
        "max_depth":6,
        'random_state': 42,
        "n_jobs":-1,
        "seed":123,
        "verbose":-1,
        "num_leaves":trial.suggest_int('num_leaves', 10, 200, step=10),
        "min_data_in_leaf":trial.suggest_int('min_data_in_leaf', 10, 100, step=10),
        'lambda_l1': trial.suggest_float('alpha', 0.0001, 10.0),
        'lambda_l2': trial.suggest_float('lambda', 0.0001, 10.0),
        'subsample': trial.suggest_float('subsample', 0.2, 1.0, step=0.1),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.2, 1.0, step=0.1),
    }
    
    skf = KFold(n_splits=5, random_state=0, shuffle=True)
    trainlist, validlist = [], []
    for train_index, val_index in skf.split(X, y):
        trainlist.append(train_index)
        validlist.append(val_index)
    folds = zip(trainlist, validlist)
    
    dataset = lgb.Dataset(X, y)
    res = lgb.cv(
        params, dataset, num_boost_round=1000, verbose_eval=0,
        early_stopping_rounds=10,
        folds=folds,
        stratified=False,
    )
    
    return res["rmse-mean"][-1] + res["rmse-stdv"][-1]


# In[27]:


'''
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, timeout=18000)
print('Number of finished trials: ', len(study.trials))
print('Best trial: ', study.best_trial.params)
print('Best value: ', study.best_value)
'''


# In[28]:


def get_scores_lgbm(params, X, y, X_test, nfolds=5):
    df_score_details = {
        'model':[],
        'r2':[],
        'rmse':[],
    }
    fig, ax = plt.subplots(figsize=(16, 4))
    for fig_index in range(1):
        model_key = "lgbm"
        kf = KFold(n_splits=nfolds)
        np_test = np.zeros((len(X_test)))
        val_r2_scores = []
        val_rmse_scores = []
        for fold, (train_index, val_index) in enumerate(kf.split(X)):
            X_train, X_val = X.loc[train_index,:], X.loc[val_index,:]
            y_train, y_val = y.loc[train_index,:], y.loc[val_index,:]
            
            dtrain = lgb.Dataset(X_train, y_train)
            dval = lgb.Dataset(X_val, y_val)
            
            evals_result = {}
            model = lgb.train(params=params, train_set=dtrain, valid_sets=[dval], num_boost_round=10000, 
                          early_stopping_rounds=50, verbose_eval=1000, evals_result=evals_result)
            
            val_preds = model.predict(X_val).reshape(-1)
            test_preds = model.predict(X_test).reshape(-1)
            
            val_r2_scores.append(r2_score(y_val, val_preds))
            val_rmse_scores.append(mean_squared_error(y_val, val_preds, squared=False))
            
            np_test += test_preds / nfolds
        df_score_details["model"].append(model_key)
        df_score_details["r2"].append(np.mean(val_r2_scores))
        df_score_details["rmse"].append(np.mean(val_rmse_scores))
        
        sub = pd.read_csv("../input/scrabble-player-rating/sample_submission.csv")
        sub["rating"] = np_test
        sub.to_csv(f"{model_key}_baseline_sub.csv",index=False)
        
        ax.hist(y, color='tab:blue', label='train', bins=50, density=True, alpha=0.5)
        ax.hist(sub["rating"], color='tab:red', label='test', bins=50, density=True, alpha=0.5)
        ax.set_title(model_key)
        ax.legend()
    df_score = pd.DataFrame(df_score_details)
    gc.collect()
    plt.show()
    return df_score


# In[29]:


params_lgbm = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.01,
        "max_depth":6,
        'random_state': 42,
        "n_jobs":-1,
        "seed":123,
        "verbose":-1,
        'num_leaves': 170, 
        'min_data_in_leaf': 10, 
        'alpha': 6.213527693669625, 
        'lambda': 0.1701794289469321, 
        'subsample': 0.7, 
        'feature_fraction': 0.8,
    }

df_score = get_scores_lgbm(params_lgbm, X, y, X_test, nfolds=5)


# In[30]:


df_score

