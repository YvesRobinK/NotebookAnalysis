#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import polars as pl


# ## Source for recreating essays
# https://www.kaggle.com/code/iurigabriel/lgbm-xgboost

# In[ ]:





# In[ ]:





# ## Goal
# My goal for this project is to develop the most efficient solution for the task. To do so, I'm going to keep only the necessary portions of my code for my model to run effectively. 
# 

# **This notebook is currently ranked 168 on the efficiency LB** 

# ### 1. Load the data

# In[2]:


test_logs=pl.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/test_logs.csv').to_pandas()


test_logs['score']=np.nan
test_logs['index']=test_logs['id'].astype(str)+'test'

test_scores=test_logs[['id', 'score']].groupby('id',as_index=False).max()
test_scores['index']=test_scores['id'].astype(str)+'test'

test_logs=test_logs[['index', 'id', 'event_id', 'down_time', 'up_time', 'action_time',
       'activity', 'down_event', 'up_event', 'text_change',
       'cursor_position', 'word_count']]



# In[3]:


train_scores=pl.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/train_scores.csv').to_pandas()
train_scores['index']=train_scores['id'].astype(str)+'train'
train_logs=pl.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/train_logs.csv').to_pandas()
train_logs['index']=train_logs['id'].astype(str)+'train'



# In[4]:


##concat the train and test logs
df_logs = pd.concat([train_logs, test_logs], axis=0)
df_scores = pd.concat([train_scores, test_scores], axis=0)


# In[5]:


import re
def q1(x):
    return x.quantile(0.25)
def q3(x):
    return x.quantile(0.75)

AGGREGATIONS = ['count', 'mean', 'min', 'max', 'first', 'last', pd.Series.mode, q1, 'median', q3, 'sum']
def reconstruct_essay(currTextInput):
    essayText = ""
    for Input in currTextInput.values:
        if Input[0] == 'Replace':
            replaceTxt = Input[2].split(' => ')
            essayText = essayText[:Input[1] - len(replaceTxt[1])] + replaceTxt[1] + essayText[Input[1] - len(replaceTxt[1]) + len(replaceTxt[0]):]
            continue
        if Input[0] == 'Paste':
            essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
            continue
        if Input[0] == 'Remove/Cut':
            essayText = essayText[:Input[1]] + essayText[Input[1] + len(Input[2]):]
            continue
        if "M" in Input[0]:
            croppedTxt = Input[0][10:]
            splitTxt = croppedTxt.split(' To ')
            valueArr = [item.split(', ') for item in splitTxt]
            moveData = (int(valueArr[0][0][1:]), int(valueArr[0][1][:-1]), int(valueArr[1][0][1:]), int(valueArr[1][1][:-1]))
            if moveData[0] != moveData[2]:
                if moveData[0] < moveData[2]:
                    essayText = essayText[:moveData[0]] + essayText[moveData[1]:moveData[3]] + essayText[moveData[0]:moveData[1]] + essayText[moveData[3]:]
                else:
                    essayText = essayText[:moveData[2]] + essayText[moveData[0]:moveData[1]] + essayText[moveData[2]:moveData[0]] + essayText[moveData[1]:]
            continue
        essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
    return essayText


def get_essay_df(df):
    df       = df[df.activity != 'Nonproduction']
    temp     = df.groupby('id').apply(lambda x: reconstruct_essay(x[['activity', 'cursor_position', 'text_change']]))
    essay_df = pd.DataFrame({'id': df['id'].unique().tolist()})
    essay_df = essay_df.merge(temp.rename('essay'), on='id')
    return essay_df


def word_feats(df):
    essay_df = df
    df['word'] = df['essay'].apply(lambda x: re.split(' |\\n|\\.|\\?|\\!',x))
    df = df.explode('word')
    df['word_len'] = df['word'].apply(lambda x: len(x))
    df = df[df['word_len'] != 0]

    word_agg_df = df[['id','word_len']].groupby(['id']).agg(AGGREGATIONS)
    word_agg_df.columns = ['_'.join(x) for x in word_agg_df.columns]
    word_agg_df['id'] = word_agg_df.index
    word_agg_df = word_agg_df.reset_index(drop=True)
    return word_agg_df

train_essays           = get_essay_df(df_logs)
train_feats            = word_feats(train_essays)


# ### 2. Summarizing data and feature engineering
#  - Calculate streaks for activities
#  - Group by student
#  - Word length
#  - Sentences, questions, quotes, commas
#  - Group by activity (for input, nonproduction, remove/cut, and replace)
#  - 2 minute warning for nonproduction
#  - Production rate - number of characters (including spaces) produced per minute during the process
#  - Binary flags based on number of words, number of paragraphs, and sentences

# In[6]:


def generate_streak_info(df):

    
    data = df[['id', 'activity']]
    data['start_of_streak'] = data['activity'].ne(data['activity'].shift())
    data['streak_id'] = data.start_of_streak.cumsum()
    data['streak_counter'] = data.groupby('streak_id').cumcount() + 1
 
    logs_with_streaks = pd.concat([df, data[['streak_counter', 'start_of_streak']]], axis=1)
    logs_with_streaks['end_of_streak']= logs_with_streaks['start_of_streak'].shift(-1, fill_value=True)

    return logs_with_streaks

df_logs=generate_streak_info(df_logs)
df_logs


# In[7]:


df_logs_space=df_logs[df_logs['up_event']=='Space']
df_logs_space.sort_values(['id', 'event_id'], inplace=True)
df_logs_space['time_betweenwords'] = df_logs_space.groupby(['id'])['up_time'].diff()
df_logs_by_id=df_logs_space.groupby(by='index').agg({'time_betweenwords':  ['mean', 'max', 'min',  'std', 'skew']}).reset_index()
df_logs_by_id.columns = df_logs_by_id.columns.map(''.join).str.strip('|')


# In[8]:


#interkeystroke interval IKI and time between keys TBK
def iki_tbk(data):
    
    data['previous_uptime'] = data.groupby('id')['up_time'].shift()
    data['iki']=data['up_time']-data['previous_uptime']
    data['tbk']=data['down_time']-data['previous_uptime']
    return data
df_logs=iki_tbk(df_logs)
df_logs['iki.5-1']=np.where((df_logs['iki']>=500 ) & (df_logs['iki']<=1000 ),1,0)
df_logs['iki1.5']=np.where((df_logs['iki']>1000 ) & (df_logs['iki']<=1500 ),1,0)
df_logs['iki2']=np.where((df_logs['iki']>1500 ) & (df_logs['iki']<=2000 ),1,0)
df_logs['iki2.5']=np.where((df_logs['iki']>2000 ) & (df_logs['iki']<=3000 ),1,0)
df_logs['iki3']=np.where((df_logs['iki']>3000 ),1,0)


# Resource: https://people.eng.unimelb.edu.au/baileyj/papers/paper249-EDM.pdf
# 

# In[9]:


df_logs['space_time']=np.where(df_logs['up_event']=='Space', df_logs['action_time'], np.nan)
df_logs['revision_time']=np.where((df_logs['activity']=='Remove/Cut' )|(df_logs['activity']=='Replace' ), df_logs['action_time'], np.nan)


# In[10]:


df_logs['2min_warning']=np.where((df_logs['action_time']>=120000) & (df_logs['activity']=='Nonproduction'), 1, 0)

df_logs['word_length']=np.where((df_logs['activity']=='Input') & (df_logs['end_of_streak']==True), df_logs['streak_counter'], np.nan)
df_logs['character']=np.where((df_logs['activity']=='Input') & (df_logs['text_change']!='NoChange'), 1, 0)
df_logs['enter']=np.where((df_logs['down_event']=='Enter'), 1, 0)
df_logs['sentence']=np.where((df_logs['text_change']=='.'), 1, 0)
df_logs['semicolon']=np.where((df_logs['text_change']==';'), 1, 0)
df_logs['comma']=np.where((df_logs['text_change']==','), 1, 0)

df_logs['question']=np.where((df_logs['text_change']=='?'), 1, 0)
df_logs['parenthesis']=np.where((df_logs['text_change']=='('), 1, 0)
df_logs['quotes']=np.where((df_logs['text_change']=='"'), 1, 0)

df_logs['shift']=np.where((df_logs['up_event']=='Shift'), 1, 0)
df_logs['semicolon']=np.where((df_logs['text_change']==';'), 1, 0)


# In[11]:


df_logs_agg = df_logs.groupby(by=['index', 'id']).agg({'event_id': 'count', 'space_time': ['mean', 'min', 'max', 'std'], 'revision_time':['mean', 'min', 'max', 'std'],'iki':['mean', 'min', 'max', 'std'] , 'iki2':'count','iki2.5':'count','iki3':'count','iki1.5':'count','iki.5-1':'count', 'tbk':['mean', 'min', 'max', 'std'], 'action_time': ['mean', 'max', 'min', 'sum', 'std', 'skew'], 'character': 'sum', 'enter': 'sum', 'comma':'sum','question':'sum', 'parenthesis':'sum', 'quotes':'sum', 'shift':'sum', 'semicolon':'sum','sentence':'sum','up_time':'max', 'word_count': 'max','cursor_position': ['max','mean', 'std','skew'],   'word_length': ['max','mean', 'std','skew'], 'streak_counter': ['max','mean', 'std','skew'],   'word_length': ['max','mean', 'std','skew'], '2min_warning': ['max','sum'],   'word_count':'max'}).reset_index()


# In[12]:


df_logs_agg.columns = df_logs_agg.columns.map('|'.join).str.strip('|')


# In[13]:


#input group
df_input=df_logs[df_logs['activity']=='Input'].groupby(by='index').agg({'event_id': 'count', 'action_time': ['mean', 'max', 'min', 'sum', 'std', 'skew'], 'streak_counter': ['max','mean', 'std','skew']}).reset_index()
df_input.columns = df_input.columns.map('_input'.join).str.strip('|')
#pause group
df_pause=df_logs[df_logs['activity']=='Nonproduction'].groupby(by='index').agg({'event_id': 'count', 'action_time': ['mean', 'max', 'min', 'sum', 'std', 'skew'],'streak_counter': ['max','mean', 'std','skew'],}).reset_index()
df_pause.columns = df_pause.columns.map('_pause'.join).str.strip('|')
#remove cut group
df_remove_cut=df_logs[df_logs['activity']=='Remove/Cut'].groupby(by='index').agg({'event_id': 'count', 'action_time': ['mean', 'max', 'min', 'sum', 'std', 'skew'],'streak_counter': ['max','mean', 'std','skew'],}).reset_index()
df_remove_cut.columns = df_remove_cut.columns.map('_remove'.join).str.strip('|')
#replace group
df_replace=df_logs[df_logs['activity']=='Replace'].groupby(by='index').agg({'event_id': 'count', 'action_time': ['mean', 'max', 'min', 'sum', 'std', 'skew'],'streak_counter': ['max','mean', 'std','skew'],}).reset_index()
df_replace.columns = df_replace.columns.map('_replace'.join).str.strip('|')


# In[14]:


df_logs_agg1 = pd.merge(df_logs_agg, df_input, left_on='index', right_on='index_input' , how='left')

df_logs_agg2=pd.merge(df_logs_agg1, df_pause,  left_on=['index', 'index_input'], right_on=['index_pause', 'index_pause'],  how='left')
df_logs_agg3=pd.merge(df_logs_agg2, df_remove_cut,  left_on=['index', 'index_pause'], right_on=['index_remove', 'index_remove'] , how='left')
df_logs_agg4=pd.merge(df_logs_agg3, df_replace,  left_on=['index', 'index_remove'],right_on=['index_replace', 'index_replace'] , how='left')
df_logs_agg5=pd.merge(df_logs_agg4, df_logs_by_id, left_on=['index'], right_on=['index'], how='left')


# ## Features from prior research

# https://link.springer.com/article/10.1007/s11145-019-09953-8#Sec6

# In[15]:


df=pd.merge(df_logs_agg5, df_scores, on=['index', 'id'], how='left')
df['time_in_minutes']=df['up_time|max']/60000
#production measures
df['words_per_minute']=df['word_count|max']/df['time_in_minutes']
df['characters_per_minute']=df['character|sum']/df['time_in_minutes']

df['sentence_length']=df['word_count|max']/(df['sentence|sum']+1)
df['3_or_more_paragraphs']=np.where(df['enter|sum']>=2, 1, 0)
df['no_paragraphs']=np.where(df['enter|sum']<=0, 1, 0)
df['word_count_250']=np.where(abs(df['word_count|max']-250)<=50, 1, 0)
df['word_count_800']=np.where(df['word_count|max']>800, 1, 0)

df['word_count_100']=np.where(df['word_count|max']<100, 1, 0)
df['paragraph_length']=round(df['sentence|sum']/(df['enter|sum']-1))
df['5_sentences']=np.where(df['sentence|sum']<5, 1, 0)
df['efficiency']=(df['character|sum'].astype(float))/(df['event_id|count'].astype(float))




# In[16]:


df_features=pd.merge(df, train_feats, on='id')


# ### 3. Remove unnecessary index columns & create training and testing set based on missing scores

# In[17]:


#train data is when there is a score
df_train=df_features[df_features['score'].isna()==False]



# In[18]:


df_test=df_features[df_features['score'].isna()==True]


# In[19]:


df_numeric=df_train.select_dtypes(include='number')


# ### 4. Select features with corr >.04 or <-.04

# In[20]:


features=[]
score_corr=df_numeric.corr().T
score_corr
score_corr=score_corr[score_corr['score']!=1]
for value in  (score_corr[score_corr['score']>=.04].T.columns.values):
    features.append(value)
for value in  (score_corr[score_corr['score']<=-.04].T.columns.values):
    features.append(value)



# In[21]:


features


# ### 5. Create X and y variables

# In[22]:


from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error
df_train=df_train[df_train['score']>=.5]

X=df_train[features]
X.replace(-np.Inf, np.nan, inplace=True)
X.replace(np.Inf, np.nan, inplace=True)

X.fillna(0, inplace=True)



y=df_train['score']


#X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=.2, random_state=2)


# In[23]:


# Compare Algorithms
import matplotlib.pyplot as plt
import pandas
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgbm
from catboost import CatBoostRegressor
from  xgboost import XGBRegressor
"""

# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
#transform target variable and predictors




models.append(('LGBM', lgbm.LGBMRegressor()))
models.append(('RF', RandomForestRegressor()))
models.append(('Cat', CatBoostRegressor(logging_level='Silent')))
models.append(('XG', XGBRegressor() ))
# evaluate each model in turn
results = []
names = []
scoring='neg_root_mean_squared_error'

for name, model in models:
	kfold = KFold(n_splits=10, shuffle=True)
	cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring,verbose=0)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
"""


# ### 6. Train test split and create model
# - I tried a few different models including a random forest regressor and lightgbm and a I did some tuning of the hyperparameters.
# - I landed on a model that yielded a .638 RMSE for the public leaderboard
# - This notebook is currently  168 on the efficiency LB

# In[24]:


## from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import lightgbm as lgbm
from catboost import CatBoostRegressor

#transform target variable and predictors

#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.2, random_state=2)





# In[25]:


# Define objective function
def objective(trial):
    # Suggest values for hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 10, 200, log=True)
    max_depth = trial.suggest_int("max_depth", 2, 32)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    # Create and fit random forest model
    model = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=42,
    )
    model.fit(X_train, y_train)

    # Make predictions and calculate RMSE
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
     
    return rmse


# In[26]:


# RF study
import optuna
#study = optuna.create_study(direction="minimize")

# Run optimization process
#study.optimize(objective, n_trials=20, show_progress_bar=True)


# ## RF Best Params

# In[27]:


#print("Best params ", study.best_params)
#print("Best score: ", study.best_value)


# In[28]:


# Define objective function
def objective(trial):
    param = {
        #'tree_method':'gpu_hist',  # this parameter means using the GPU when training our model to speedup the training process
        #'sampling_method': 'gradient_based',
        'lambda': trial.suggest_loguniform('lambda', 7.0, 17.0),
        'alpha': trial.suggest_loguniform('alpha', 7.0, 17.0),
        'eta': trial.suggest_categorical('eta', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'gamma': trial.suggest_categorical('gamma', [18, 19, 20, 21, 22, 23, 24, 25]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.008,0.01,0.012,0.014,0.016,0.018, 0.02]),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        
        'colsample_bynode': trial.suggest_categorical('colsample_bynode', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'n_estimators': trial.suggest_int('n_estimators', 400, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 8, 600),  
        'max_depth': trial.suggest_categorical('max_depth', [3, 4, 5, 6, 7]),  
        'subsample': trial.suggest_categorical('subsample', [0.5,0.6,0.7,0.8,1.0]),
        'random_state': 42
    }


    model = XGBRegressor(**param
    )
    model.fit(X_train, y_train)

    # Make predictions and calculate RMSE
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
     
    return rmse
    


# In[29]:


# XGB study
#study = optuna.create_study(direction="minimize")

# Run optimization process
#study.optimize(objective, n_trials=30, show_progress_bar=True)


# ## XGBoost Best Params

# In[30]:


#print("Best params ", study.best_params)
#print("Best score: ", study.best_value)


# In[31]:


import optuna
from sklearn.model_selection import cross_validate


def objective(trial):
    param = {
                'metric': 'rmse', 
                'random_state': 48,
                'n_estimators': 500,
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
                'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
                'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
                'learning_rate': trial.suggest_categorical('learning_rate', [0.006,0.008,0.01,0.02]),
                'max_depth': trial.suggest_categorical('max_depth', [5, 10,20,100]),
                'num_leaves' : trial.suggest_int('num_leaves', 1, 1000),
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
                'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100)
            }
    model = lgbm.LGBMRegressor(**param)    

    model.fit(X_train, y_train)

    # Make predictions and calculate RMSE
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
     
    return rmse
    


# In[32]:


## LGB study
#study = optuna.create_study(direction="minimize")

# Run optimization process
#study.optimize(objective, n_trials=30, show_progress_bar=True)

#print("Best params ", study.best_params)
#print("Best score: ", study.best_value)


# In[33]:


def objective(trial):
    param = {
        "iterations": 400,
        "learning_rate": trial.suggest_float("learning_rate", 5e-3, 5e-2, log=True),
        "depth": trial.suggest_int("depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }
    model = CatBoostRegressor(**param)  

    model.fit(X_train, y_train, verbose=0)

    # Make predictions and calculate RMSE
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
     
    return rmse

 
    


# In[34]:


## catboost study
#study = optuna.create_study(direction="minimize")

# Run optimization process
#study.optimize(objective, n_trials=100, show_progress_bar=True)
#print("Best params ", study.best_params)
#print("Best score: ", study.best_value)


# In[35]:


#xgbparams= {'lambda': 16.649892927590216, 'alpha': 7.693884294507179, 'eta': 0.5, 'gamma': 18, 'learning_rate': 0.018, 'colsample_bytree': 0.8, 'colsample_bynode': 0.7, 'n_estimators': 515, 'min_child_weight': 80, 'max_depth': 7, 'subsample': 1.0}

#rfparams={'n_estimators': 141, 'max_depth': 32, 'min_samples_split': 10, 'min_samples_leaf': 4}

#lgparams={'reg_alpha': 5.44507429393035, 'reg_lambda': 0.0177242808475022, 'colsample_bytree': 0.4, 'subsample': 0.6, 'learning_rate': 0.01, 'max_depth': 20, 'num_leaves': 77, 'min_child_samples': 22, 'min_data_per_groups': 28}
catparams=  {'learning_rate': 0.01779311171344571, 'depth': 6, 'subsample': 0.4231297035401239, 'colsample_bylevel': 0.386115563954346, 'min_data_in_leaf': 100}


# In[36]:


#xgregressor=XGBRegressor(**xgbparams)
#rfregressor=RandomForestRegressor(**rfparams)
#lgbregressor=lgbm.LGBMRegressor(**lgparams)
catregressor=CatBoostRegressor(**catparams)


# In[37]:


df_train=df_train[df_train['score']>=1]

X=df_train[features]
X.replace(-np.Inf, np.nan, inplace=True)
X.replace(np.Inf, np.nan, inplace=True)

X.fillna(0, inplace=True)



y=df_train['score']


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.2, random_state=2)


#lgbregressor.fit(X_train, y_train)
catregressor.fit(X_train, y_train, verbose=0)
#xgregressor.fit(X_train, y_train)
#rfregressor.fit(X_train, y_train)
#y_pred1=lgbregressor.predict(X_test)
y_pred2=catregressor.predict(X_test)
#y_pred3=xgregressor.predict(X_test)
#y_pred4=rfregressor.predict(X_test)



# In[38]:


#predictions=pd.DataFrame(y_test)
#predictions['lgbscore']=y_pred1
#predictions['catscore']=y_pred2
#predictions['xgbscore']=y_pred3
#predictions['rfscore']=y_pred4


# In[39]:


#mse = mean_squared_error(predictions['score'], predictions['catscore'])
#rmse = math.sqrt(mse)
#rmse


# In[40]:


from sklearn.model_selection import ShuffleSplit
from catboost import CatBoostRegressor
import warnings 
warnings.filterwarnings("ignore")



# ### 7. Identify testing data and make predictions

# In[41]:


X_test=df_test[features]

X_test.fillna(0, inplace=True)





# ### 8. Create submission file

# In[42]:


df_test['score']=catregressor.predict(X_test)
submission=df_test[['id', 'score']]


submission.to_csv('submission.csv', index=False)


# In[43]:


submission


# In[ ]:




