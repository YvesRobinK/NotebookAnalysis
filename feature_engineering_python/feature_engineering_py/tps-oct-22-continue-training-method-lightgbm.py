#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import gc
from lightgbm import LGBMClassifier
from lightgbm import early_stopping, log_evaluation
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style ('darkgrid')
sns.palplot(sns.color_palette('rainbow'))
sns.set_palette('rainbow')

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class config:
    SEED = 777
    test_size = 0.20


# In[2]:


TARGETS = ['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec']
DROP_FEATURES = ['id', 'game_num', 'event_id', 'event_time', 'team_scoring_next', 'player_scoring_next']

def feature_engineering(df):
    df = df.drop(DROP_FEATURES, axis=1, errors="ignore")    
    return df

# https://www.kaggle.com/code/mattop/rocket-league-tps-eda
def add_features(df):
    df[f"goal_a_distance"] = ((df["ball_pos_x"]-0)**2 + (df["ball_pos_y"]-100)**2 + (df["ball_pos_z"]-20)**2)**0.5
    df[f"goal_b_distance"] = ((df["ball_pos_x"]-0)**2 + (df["ball_pos_y"]+100)**2 + (df["ball_pos_z"]-20)**2)**0.5
    df = df.drop(DROP_FEATURES, axis=1, errors="ignore")   
    return df


# In[3]:


# https://www.kaggle.com/code/gazu468/feather-to-compress-your-data-8x-faster

test = pd.read_feather("../input/tpsoct22-feather-files/test.feather")
test = feature_engineering(test)
test = add_features(test)


# ### Data Descriptions 
# *   **`game_num`** _(train only)_: Unique identifier for the game from which the event was taken.
#     
# *   **`event_id`** _(train only)_: Unique identifier for the sequence of consecutive frames.
#     
# *   **`event_time`** _(train only)_: Time in seconds before the event ended, either by a goal being scored or simply when we decided to truncate the timeseries if a goal was not scored.
#     
# *   **`ball_pos_[xyz]`**: Ball's position as a 3d vector.
#     
# *   **`ball_vel_[xyz]`**: Ball's velocity as a 3d vector.
#     
# *   For `i` in `[0, 6)`:
#     
#     *   **`p{i}_pos_[xyz]`**: Player `i`'s position as a 3d vector.
#     *   **`p{i}_vel_[xyz]`**: Player `i`'s velocity as a 3d vector.
#     *   **`p{i}_boost`**: Player `i`'s boost remaining, in `[0, 100]`. A player can consume boost to substantially increase their speed, and is required to fly up into the `z` dimension (besides driving up a wall, or the small air gained by a jump).
#     *   All `p{i}` columns will be `NaN` if and only if the player is demolished (destroyed by an enemy player; will respawn within a few seconds).
#     *   Players 0, 1, and 2 make up team `A` and players 3, 4, and 5 make up team `B`.
#     *   The orientation vector of the player's car (which way the car is facing) does not necessarily match the player's velocity vector, and this dataset does not capture orientation data.
# *   For `i` in `[0, 6)`:
#     
#     *   **`boost{i}_timer`**: Time in seconds until big boost orb `i` respawns, or `0` if it's available. Big boost orbs grant a full 100 boost to a player driving over it. The orb `(x, y)` locations are roughly `[ (-61.4, -81.9), (61.4, -81.9), (-71.7, 0), (71.7, 0), (-61.4, 81.9), (61.4, 81.9) ]` with `z = 0`. (Players can also gain boost from small boost pads across the map, but we do not capture those pads in this dataset).
# *   **`player_scoring_next`** _(train only)_: Which player scores at the end of the current event, in `[0, 6)`, or `-1` if the event does not end in a goal.
#     
# *   **`team_scoring_next`** _(train only)_: Which team scores at the end of the current event (`A` or `B`), or `NaN` if the event does not end in a goal.
#     
# *   **`team_[A|B]_scoring_within_10sec`** _(train only)_: **\[Target columns\]** Value of `1` if `team_scoring_next == [A|B]` and `time_before_event` is in `[-10, 0]`, otherwise `0`.
#     
# *   **`id`** _(test and submission only)_: Unique identifier for each test row. Your submission should be a pair of `team_A_scoring_within_10sec` and `team_B_scoring_within_10sec` probability predictions for each `id`, where your predictions can range the real numbers from `[0, 1]`.

# In[4]:


dtypes_df = pd.read_csv('/kaggle/input/tabular-playground-series-oct-2022/train_dtypes.csv')
dtypes = {k: v for (k, v) in zip(dtypes_df.column, dtypes_df.dtype)}

train = pd.read_csv('/kaggle/input/tabular-playground-series-oct-2022/train_0.csv',nrows=10000, dtype = dtypes)
#train = pd.read_csv('/kaggle/input/tabular-playground-series-oct-2022/train_0.csv', dtype = dtypes).sample(10000)
train.shape


# In[5]:


train.head()


# # **<span style="color:#e76f51;">Missings</span>**

# In[6]:


print('Number of missing values in training set:',train.isna().sum().sum())
print('')
print('Number of missing values in test set:',test.isna().sum().sum())


# In[7]:


trn_null = train.isnull().sum()
tst_null = test.isnull().sum()

print('Train columns with null values:\n', trn_null[trn_null>0])
print("-"*10)

print('Test/Validation columns with null values:\n', tst_null[tst_null>0])
print("-"*10)


# In[8]:


# per sample
plt.figure(figsize=(10, 8))
plt.imshow(train.isna(), aspect="auto", interpolation="nearest", cmap="gray")
plt.xlabel("Column Number")
plt.ylabel("Sample Number")
plt.show()


# In[9]:


# per feature
train.isna().mean().sort_values().plot(
    kind="bar", figsize=(15, 4),
    title="Percentage of missing values per feature",
    ylabel="Ratio of missing values per feature")
plt.show()


# # **<span style="color:#e76f51;">Duplicates</span>**

# In[10]:


ignore_cols = ['game_num','event_id','event_time']
n_duplicates = train.drop(labels=ignore_cols, axis=1).duplicated().sum()
n_duplicates_test = test.duplicated().sum()
print(f"You have {n_duplicates} duplicates in train.")
print(f"You have {n_duplicates_test} duplicates in test.")


# # **<span style="color:#e76f51;">EDA</span>**

# In[11]:


unique_values = train.select_dtypes(include="number").nunique().sort_values()

# Plot information with y-axis in log-scale
unique_values.plot.bar(logy=True, figsize=(15, 4), title="Unique values per feature")
plt.show()


# In[12]:


train.plot(lw=0, marker=".", subplots=True, layout=(-1, 4),
          figsize=(15, 30), markersize=1);


# In[13]:


# Feature distribution
train.hist(bins=25, figsize=(15, 25), layout=(-1, 5), edgecolor="black")
plt.tight_layout();


# In[14]:


# Most frequent entries
most_frequent_entry = train.mode()

df_freq = train.eq(most_frequent_entry.values, axis=1)
df_freq = df_freq.mean().sort_values(ascending=False)


display(df_freq.head())

df_freq.plot.bar(figsize=(15, 4));


# ## **<span style="color:#e76f51;">Correlation</span>**

# In[15]:


drop_cols = ['game_num','event_id','event_time']
df_corr = train.drop(columns=drop_cols).corr()
labels = np.where(np.abs(df_corr)>0.75, "S",
                  np.where(np.abs(df_corr)>0.5, "M",
                           np.where(np.abs(df_corr)>0.25, "W", "")))

plt.figure(figsize=(30, 20))
sns.heatmap(df_corr, mask=np.eye(len(df_corr)), square=True,
            center=0, annot=labels, fmt='', linewidths=.5,
            cmap="YlGnBu", cbar_kws={"shrink": 0.8});


# In[16]:


del train
del dtypes_df
gc.collect()


# # **<span style="color:#e76f51;">Continue Training LightGBM Model</span>**
# 
# Using LightGBM model or Booster instance for continue training with **init_model** option. **init_model** will perform gradient boosting for num_iterations additional rounds. This will allow us to use all 10 train files with LightGBM model training.
# 
# Reference: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html

# In[17]:


params = {
    'force_col_wise' : True,
    'objective': 'binary',
    'random_state' : config.SEED,
    #'importance_type': 'gain',
    'metric': 'logloss',
    'n_jobs': -1,
}



es = early_stopping(stopping_rounds=10, first_metric_only=False)
le = log_evaluation(-1)

lr = lgb.reset_parameter(learning_rate=lambda iter: 0.05 * (0.99 ** iter))
#lr = lgb.reset_parameter(learning_rate=lambda iter: 0.1 - 0.001 * iter)


# In[18]:


features = [f for f in test.columns]
init = 1

for split in range(10):
    gc.collect()

    filename = "../input/tpsoct22-feather-files/train_"+str(split)+".feather"
    train = pd.read_feather(filename)
    # drop unwanted features
    train = feature_engineering(train)
    # add new features
    train = add_features(train)
    
    label_a = 'team_A_scoring_within_10sec'
    label_b = 'team_B_scoring_within_10sec'

    train[label_a] = train[label_a].astype('int8')
    train[label_b] = train[label_b].astype('int8')
       
    X_train, X_val, y_train_a, y_val_a = train_test_split(train[features], train[label_a], test_size = config.test_size, random_state = config.SEED)
    X_train, X_val, y_train_b, y_val_b = train_test_split(train[features], train[label_b], test_size = config.test_size, random_state = config.SEED)
    
    if (init):
        print(f'Start a new training job....split {split} with file {filename}')
              
        # Team A
        lgb_a = LGBMClassifier(**params)
        lgb_a.fit(X_train,y_train_a, eval_set=[(X_val, y_val_a)], eval_metric = ['logloss'],  callbacks=[es,le])
        
        # Team B
        lgb_b = LGBMClassifier(**params)
        lgb_b.fit(X_train,y_train_b, eval_set=[(X_val, y_val_b)], eval_metric = ['logloss'],  callbacks=[es,le])
        
        init = 0
        #break
    else:
        print(f'continue training....split {split} with file {filename}')
        lgb_train_a = lgb.Dataset(X_train, y_train_a)
        lgb_eval_a = lgb.Dataset(X_val, y_val_a, reference=lgb_train_a)
        
        lgb_train_b = lgb.Dataset(X_train, y_train_b)
        lgb_eval_b = lgb.Dataset(X_val, y_val_b, reference=lgb_train_b)
        
        # Team A
        lgb_a = lgb.train(params,lgb_train_a, num_boost_round=100, valid_sets=lgb_eval_a, callbacks=[es,le,lr], init_model=lgb_a)
        
        # Team B
        lgb_b = lgb.train(params,lgb_train_b, num_boost_round=100, valid_sets=lgb_eval_b, callbacks=[es,le,lr], init_model=lgb_b) 


# In[19]:


fig, ax = plt.subplots(figsize=(20, 10))
lgb.plot_importance(lgb_a,ax=ax)
plt.title('Feature Importance - Model A')
plt.show()


# In[20]:


fig, ax = plt.subplots(figsize=(20, 10))
lgb.plot_importance(lgb_b,ax=ax)
plt.title('Feature Importance - Model B')
plt.show()


# In[21]:


import shap

shap.initjs()

df_shap = X_val[:1000]
explainer = shap.TreeExplainer(lgb_a)
shap_values = explainer.shap_values(df_shap)


# In[22]:


shap.summary_plot(shap_values, df_shap)


# In[23]:


# visualize a single prediction
shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], df_shap.iloc[0,:])


# In[24]:


shap.force_plot(explainer.expected_value[1], shap_values[1][:1000,:], df_shap.iloc[:1000,:])


# In[25]:


# visualize tree 0 for lgb_a
fig, ax = plt.subplots(figsize=(20, 10))
lgb.plot_tree(lgb_a, ax=ax, show_info = ['internal_value','leaf_count'])
plt.show()


# # **<span style="color:#e76f51;">Submission</span>**

# In[26]:


dtypes_df = pd.read_csv('/kaggle/input/tabular-playground-series-oct-2022/test_dtypes.csv')
dtypes = {k: v for (k, v) in zip(dtypes_df.column, dtypes_df.dtype)}

submission = pd.read_csv('/kaggle/input/tabular-playground-series-oct-2022/sample_submission.csv',dtype = dtypes)


# In[27]:


predictions_a = lgb_a.predict(test[features])
predictions_b = lgb_b.predict(test[features])

submission['team_A_scoring_within_10sec'] = predictions_a
submission['team_B_scoring_within_10sec'] = predictions_b


# In[28]:


submission.to_csv('submission.csv', index = False)
submission.head()


# # **<span style="color:#e76f51;">Work in progress</span>**
# 
# **TODO:**
# - Dynamic learning rate
# - more EDA and Feature Engineering
