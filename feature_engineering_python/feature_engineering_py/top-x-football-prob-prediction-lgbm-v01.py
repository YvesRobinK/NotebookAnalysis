#!/usr/bin/env python
# coding: utf-8

# ## Football Match Probability Prediction
# 
# > **In case you fork/copy/like this notebook:**
# > **Upvote it s'il te plaît/Per favore/Please**

# ## Import modules and loading the data
# First let's import modules, the training and the test set. The test set contains the same columns than the training set without the target.

# In[1]:


# Import libraries
# import gc
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
#
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, precision_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
#
import lightgbm as lgb


# In[2]:


#loading the data
train = pd.read_csv('/kaggle/input/football-match-probability-prediction/train.csv')
test = pd.read_csv('/kaggle/input/football-match-probability-prediction/test.csv')
submission = pd.read_csv('/kaggle/input/football-match-probability-prediction/sample_submission.csv')


# ## Parameters
# Set some parameters that will be used later.

# In[3]:


# Set seed
# np.random.seed(123)
# tf.random.set_seed(123)

#Some parameters
MASK = -666 # fill NA with -666 (the number of the beast)
T_HIST = 10 # time history, last 10 games
CLASS = 3 #number of classes (home, draw, away)

DEBUG = False
# Run on a small sample of the data
if DEBUG:
    train = train[:10000]


# In[4]:


# exclude matches with no history at date 1 - full of NA (1159 rows excluded)
train.dropna(subset=['home_team_history_match_date_1'], inplace = True)


# In[5]:


print(f"Train: {train.shape} \n Submission: {submission.shape}")
train.head()


# ## Feature Engineering

# In[6]:


# for cols "date", change to datatime 
for col in train.filter(regex='date', axis=1).columns:
    train[col] = pd.to_datetime(train[col])
    test[col] = pd.to_datetime(test[col])

# Some feature engineering

def add_features(df):
    df['home_team_history_match_date_0'] = df['match_date']
    df['away_team_history_match_date_0'] = df['match_date']
    for i in range(1, 11): # range from 1 to 10
        # Feat. difference of days
        df[f'home_team_history_match_DIFF_days_{i}'] = (df['match_date'] - df[f'home_team_history_match_date_{i}']).dt.days
        df[f'away_team_history_match_DIFF_days_{i}'] = (df['match_date'] - df[f'away_team_history_match_date_{i}']).dt.days
        # date diff 2
        # df[f'home_team_history_match_DIFF2_days_{i}'] = (df[f'home_team_history_match_date_{i}'] - df[f'home_team_history_match_date_{i-1}']).dt.days
        # df[f'away_team_history_match_DIFF2_days_{i}'] = (df[f'away_team_history_match_date_{i}'] - df[f'away_team_history_match_date_{i-1}']).dt.days
    # Feat. difference of scored goals
        df[f'home_team_history_DIFF_goal_{i}'] = df[f'home_team_history_goal_{i}'] - df[f'home_team_history_opponent_goal_{i}']
        df[f'away_team_history_DIFF_goal_{i}'] = df[f'away_team_history_goal_{i}'] - df[f'away_team_history_opponent_goal_{i}']
    # Feat dummy winner x loser
        df[f'home_winner_{i}'] = np.where(df[f'home_team_history_DIFF_goal_{i}'] > 0, 1., 0.) 
        df[f'home_loser_{i}'] = np.where(df[f'home_team_history_DIFF_goal_{i}'] < 0, 1., 0.)
        df[f'away_winner_{i}'] = np.where(df[f'away_team_history_DIFF_goal_{i}'] > 0, 1., 0.)
        df[f'away_loser_{i}'] = np.where(df[f'away_team_history_DIFF_goal_{i}'] < 0, 1., 0.)
    # Results: multiple nested where # away:0, draw:1, home:2
        df[f'home_team_result_{i}'] = np.where(df[f'home_team_history_DIFF_goal_{i}'] > 0., 2.,
                         (np.where(df[f'home_team_history_DIFF_goal_{i}'] == 0., 1,
                                   np.where(df[f'home_team_history_DIFF_goal_{i}'].isna(), np.nan, 0))))
        df[f'away_team_result_{i}'] = np.where(df[f'away_team_history_DIFF_goal_{i}'] > 0., 2.,
                         (np.where(df[f'away_team_history_DIFF_goal_{i}'] == 0., 1.,
                                   np.where(df[f'away_team_history_DIFF_goal_{i}'].isna(), np.nan, 0.))))
    # Feat. difference of rating ("modified" ELO RATING)
        df[f'home_team_history_ELO_rating_{i}'] = 1/(1+10**((df[f'home_team_history_opponent_rating_{i}']-df[f'home_team_history_rating_{i}'])/10))
        df[f'away_team_history_ELO_rating_{i}'] = 1/(1+10**((df[f'away_team_history_opponent_rating_{i}']-df[f'away_team_history_rating_{i}'])/10))
        df[f'home_away_team_history_ELO_rating_{i}'] = 1/(1+10**((df[f'away_team_history_rating_{i}']-df[f'home_team_history_rating_{i}'])/10))
    # Dif
       #  df[f'home_team_history_DIF_rating_{i}'] = -df[f'home_team_history_opponent_rating_{i}']+df[f'home_team_history_rating_{i}']
       #  df[f'away_team_history_DIF_rating_{i}'] = -df[f'away_team_history_opponent_rating_{i}']+df[f'away_team_history_rating_{i}']
       #  df[f'home_away_team_history_DIF_rating_{i}'] = -df[f'away_team_history_rating_{i}']+df[f'home_team_history_rating_{i}']
    # Feat. same coach id
        df[f'home_team_history_SAME_coaX_{i}'] = np.where(df['home_team_coach_id']==df[f'home_team_history_coach_{i}'],1,0)
        df[f'away_team_history_SAME_coaX_{i}'] = np.where(df['away_team_coach_id']==df[f'away_team_history_coach_{i}'],1,0) 
    # Feat. same league id
        df[f'home_team_history_SAME_leaG_{i}'] = np.where(df['league_id']==df[f'home_team_history_league_id_{i}'],1,0)
        df[f'away_team_history_SAME_leaG_{i}'] = np.where(df['league_id']==df[f'away_team_history_league_id_{i}'],1,0) 
    # Features...
        df[f'feature_A_{i}'] = df[f'home_team_history_ELO_rating_{i}'] * df[f'home_team_history_is_play_home_{i}']
        df[f'feature_B_{i}'] = df[f'away_team_history_ELO_rating_{i}'] * df[f'away_team_history_is_play_home_{i}']
        df[f'feature_C_{i}'] = df[f'home_away_team_history_ELO_rating_{i}'] * df[f'home_team_history_is_play_home_{i}']
        df[f'feature_X_{i}'] = df[f'home_team_history_ELO_rating_{i}'] * df[f'home_team_history_SAME_leaG_{i}']
        df[f'feature_Y_{i}'] = df[f'away_team_history_ELO_rating_{i}'] * df[f'away_team_history_SAME_leaG_{i}']
        df[f'feature_Z_{i}'] = df[f'home_away_team_history_ELO_rating_{i}'] * df[f'home_team_history_SAME_leaG_{i}']
    # Fill NA with -666
    # df.fillna(MASK, inplace = True)
    return df

train = add_features(train)
test = add_features(test)


# In[7]:


# The feature based on target do not reduce the log loss 
'''
league_pivot = train.pivot_table(index='league_id', columns='target',
                          aggfunc='size',  fill_value=0).reset_index()
league_pivot['total'] = league_pivot['away']+ league_pivot['draw'] + league_pivot['home']
league_pivot['away'] /= league_pivot['total']
league_pivot['draw'] /= league_pivot['total']
league_pivot['home'] /= league_pivot['total']

train_league = pd.merge(train[['id', 'league_id']], league_pivot, left_on="league_id", right_on="league_id").copy()
test_league = pd.merge(test[['id', 'league_id']], league_pivot, left_on="league_id", right_on="league_id").copy()
'''


# In[8]:


# league_pivot.head(5)


# ## Scaling and Reshape

# In[9]:


# save targets
# train_id = train['id'].copy()
train_y = train['target'].copy()
#keep only some features
train_x = train.drop(['target', 'home_team_name', 'away_team_name'], axis=1) #, inplace=True) # is_cup EXCLUDED
# Exclude all date, league, coach columns
train_x.drop(train.filter(regex='date').columns, axis=1, inplace = True)
train_x.drop(train.filter(regex='league').columns, axis=1, inplace = True)
train_x.drop(train.filter(regex='coach').columns, axis=1, inplace = True)
#
# Test set
# test_id = test['id'].copy()
test_x = test.drop(['home_team_name', 'away_team_name'], axis=1)#, inplace=True) # is_cup EXCLUDED
# Exclude all date, league, coach columns
test_x.drop(test.filter(regex='date').columns, axis=1, inplace = True)
test_x.drop(test.filter(regex='league').columns, axis=1, inplace = True)
test_x.drop(test.filter(regex='coach').columns, axis=1, inplace = True)

# Deal with the is_cup feature
# There are NA in 'is_cup'
train_x=train_x.fillna({'is_cup':False})
train_x['is_cup'] = pd.get_dummies(train_x['is_cup'], drop_first=True)
#
test_x=test_x.fillna({'is_cup':False})
test_x['is_cup']= pd.get_dummies(test_x['is_cup'], drop_first=True)


# In[10]:


# Target, train and test shape
print(f"Target: {train_y.shape} \n Train shape: {train_x.shape} \n Test: {test_x.shape}")
print(f"Column names: {list(train_x.columns)}")


# In[11]:


train_x.head()


# In[12]:


train_y.head()


# In[13]:


# Store feature names
# feature_names = list(train.columns)
# Pivot dataframe to create an input array for the LSTM network
feature_groups = ["home_team_history_is_play_home", "home_team_history_is_cup",
    "home_team_history_goal", "home_team_history_opponent_goal",
    "home_team_history_rating", "home_team_history_opponent_rating",  
    "away_team_history_is_play_home", "away_team_history_is_cup",
    "away_team_history_goal", "away_team_history_opponent_goal",
    "away_team_history_rating", "away_team_history_opponent_rating",  
    "home_team_history_match_DIFF_days", "away_team_history_match_DIFF_days",
    # "home_team_history_match_DIFF2_days", "away_team_history_match_DIFF2_days",
    "home_team_history_DIFF_goal","away_team_history_DIFF_goal",
    "home_team_history_ELO_rating","away_team_history_ELO_rating",
    "home_away_team_history_ELO_rating",
    "home_team_history_SAME_coaX", "away_team_history_SAME_coaX",
    "home_team_history_SAME_leaG", "away_team_history_SAME_leaG",
    "home_team_result", "away_team_result",
    "home_winner", "home_loser", "away_winner", "away_loser",
    "feature_A", "feature_B", "feature_C",
    "feature_X", "feature_Y", "feature_Z",
    # "home_team_history_DIF_rating","away_team_history_DIF_rating","home_away_team_history_DIF_rating"
                 ]      
# Pivot dimension (id*features) x time_history
train_x_pivot = pd.wide_to_long(train_x, stubnames=feature_groups, 
                i=['id','is_cup'], j='time', sep='_', suffix='\d+')
test_x_pivot = pd.wide_to_long(test_x, stubnames=feature_groups, 
                i=['id','is_cup'], j='time', sep='_', suffix='\d+')
#
print(f"Train pivot shape: {train_x_pivot.shape}")  
print(f"Test pivot shape: {test_x_pivot.shape}") 


# In[14]:


# create columns based on index
train_x_pivot = train_x_pivot.reset_index()
test_x_pivot = test_x_pivot.reset_index()
# Deal with the is_cup feature
# There are NA in 'is_cup'
train_x_pivot=train_x_pivot.fillna({'is_cup':False})
train_x_pivot['is_cup'] = pd.get_dummies(train_x_pivot['is_cup'], drop_first=True)
#
test_x_pivot=test_x_pivot.fillna({'is_cup':False})
test_x_pivot['is_cup']= pd.get_dummies(test_x_pivot['is_cup'], drop_first=True)


# In[15]:


train_x_pivot.head(20)


# In[16]:


# x_train['league_id'] = train[['league_id']]
# x_test['league_id'] = test[['league_id']]


# In[17]:


# Feature engineering again!!! Group by id, stats over time
def agg_features(df):
    # vol_cols = ['log_return1_realized_volatility', 'log_return2_realized_volatility']
    # Group by match id
    df_agg = df.groupby(['id'])[feature_groups].agg(['mean', 'median', 'std', 'max', 'min', 'sum',]).reset_index()
    # Rename columns joining suffix
    df_agg.columns = ['_'.join(col) for col in df_agg.columns]
    return df_agg

train_x_agg = agg_features(train_x_pivot)
test_x_agg = agg_features(test_x_pivot)

train_x_last = train_x_pivot.loc[train_x_pivot['time'] == 1]
test_x_last = test_x_pivot.loc[train_x_pivot['time'] == 1]

# keep order
# Merge and drop columns
x_train = pd.merge(train_x_last, train_x_agg, left_on="id", right_on="id_").drop(['id', 'id_', 'time'], axis = 1).copy()
x_test = pd.merge(test_x_last, test_x_agg, left_on="id", right_on="id_").drop(['id', 'id_', 'time'], axis = 1).copy()


# In[18]:


# Merge with the feature based on target
# with the code below the Score: 1.00849 (no good!)
'''
if False:
    x_train_ = pd.merge(train_x_last, train_x_agg, left_on="id", right_on="id_").drop(['id_', 'time'], axis = 1)
    x_test_ = pd.merge(test_x_last, test_x_agg, left_on="id", right_on="id_").drop(['id_', 'time'], axis = 1)
    # Merge keep original order
    x_train = x_train_.reset_index().merge(train_league, how='left', on='id', sort=False).sort_index()
    x_test = x_test_.reset_index().merge(test_league, how='left', on='id', sort=False).sort_index()
    #
    x_train = x_train.drop(['id', 'index'], axis = 1)
    x_test = x_test.drop(['id', 'index'], axis = 1)
'''


# In[19]:


x_train.head(20)


# In[20]:


print(x_train.shape)
print(x_test.shape)


# In[21]:


# x_train = train_x_agg.drop(['id_'], axis=1).copy() 
# x_test = test_x_agg.drop(['id_'], axis=1).copy() 

# Fill NA with mean
# fill_mean = False
# if fill_mean:
#     x_train = np.where(np.isnan(x_train), np.nanmean(x_train, axis=0), x_train)
#     x_test = np.where(np.isnan(x_test), np.nanmean(x_test, axis=0), x_test)

'''
else:
    # Fill NA with MASK
    x_train = np.nan_to_num(x_train, nan=MASK)
    x_test = np.nan_to_num(x_test, nan=MASK)

# Scale features using statistics that are robust to outliers
RS = RobustScaler()
x_train = RS.fit_transform(x_train)
x_test = RS.transform(x_test)
# Reshape 
x_train = x_train.reshape(-1, T_HIST, x_train.shape[-1])
x_test = x_test.reshape(-1, T_HIST, x_test.shape[-1])
'''
# Back to dataframe ids x columns_1_10


# In[22]:


# Deal with targets
# encode class values as integers
'''encoder = LabelEncoder()
encoder.fit(train_y)
encoded_y = encoder.transform(train_y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = to_categorical(encoded_y)
# 
print(encoded_y.shape)
print(dummy_y.shape)
'''
target2int = {'away': 0, 'draw': 1, 'home': 2}
# encode target
dummy_y = train_y.map(target2int)

# multi task
# target2int_2 = {'away': 0, 'draw': 1, 'home': 0}
# encode target
# dummy2_y = train_y.map(target2int_2)


# In[23]:


# encoding away: 0 draw: 1 home: 2 
dummy_y.head()


# In[24]:


print(dummy_y.shape)


# ## Make a model
# XGBOOST model with aggregated time features.

# ## Fit the model and a make submission
# Once the model is fitted we make the **probabilities prediction**. Then we make the submission dataframe with **4 columns**. The columns home, away and draw contain probability while the column id contains the match id.

# In[25]:


# N_SPLITS of the traning set for validation using KFold
# Parameters
# Train model and predict
# models = []
# aucs = []
# preds = np.zeros(len(train))
#
num_boost_round = 1000
seed = 123
N_SPLITS = 5 # 10 (5 is better than 10 Folds)
FIRST_OOF_ONLY = False
'''
# Set LGBM hyper parameters
lgbm_params = { "objective":"multiclass"
    , "boosting_type":"gbdt"
    , 'num_class':CLASS
    , 'metric': 'multi_logloss'
    , "random_seed":SEED
    , "max_depth":5
    , "colsample_bytree":0.5
    , "subsample":0.5
    , "lambda_l1":0.1 #reg_alpha(L1)
    , "lambda_l2":0.9 #reg_lambda(L2)
    , "learning_rate":0.1
    , "num_leaves": 5
} 
'''
# Params 2
lgbm_params = {
    "objective":"multiclass" #"binary"
    , "boosting_type":"gbdt"
    , 'num_class':CLASS
    , 'metric': "multi_logloss" #''binary_logloss
    ,        'learning_rate': 0.05,        
            'lambda_l1': 2,
            'lambda_l2': 7,
            'num_leaves': 400,
            'min_sum_hessian_in_leaf': 20,
            'feature_fraction': 0.5,
            'feature_fraction_bynode': 0.5,
            'bagging_fraction': 0.5,
            'bagging_freq': 42,
            'min_data_in_leaf': 700,
            'max_depth': 5,
            'seed': seed,
            'feature_fraction_seed': seed,
            'bagging_seed': seed,
            'drop_seed': seed,
            'data_random_seed': seed,
            'verbosity': -1
        }

test_preds = []

kf = KFold(n_splits=N_SPLITS, random_state=seed, shuffle=True) # KFold

for fold, (train_idx, test_idx) in enumerate(kf.split(x_train, dummy_y)):
    print('-'*15, '>', f'Fold {fold+1}/{N_SPLITS}', '<', '-'*15)
    X_train, X_valid =  x_train.iloc[train_idx], x_train.iloc[test_idx]
    Y_train, Y_valid =  dummy_y.iloc[train_idx], dummy_y.iloc[test_idx]
    train_dataset = lgb.Dataset(X_train, Y_train) #, categorical_feature = ['is_cup'])
    val_dataset = lgb.Dataset(X_valid, Y_valid) # , categorical_feature = ['is_cup'])
    model = lgb.train(params = lgbm_params, 
                        train_set = train_dataset, 
                        valid_sets = [train_dataset, val_dataset], 
                        num_boost_round = num_boost_round, 
                        callbacks=[lgb.early_stopping(stopping_rounds=50)],
                        verbose_eval = 50)
    # Model validation    
    y_true = Y_valid.squeeze()
    y_pred = model.predict(X_valid).squeeze()
    score1 = log_loss(y_true, y_pred)
    print(f"Fold-{fold+1} | OOF LogLoss Score: {score1}")
    # print(roc_auc_score(y_true, y_pred))
    # Predictions
    test_preds.append(model.predict(x_test).squeeze())
    lgb.plot_importance(model,max_num_features=20)
    # df_importance = pd.DataFrame({'Value':model.feature_importance(),'Feature':X_train.columns.tolist()}).sort_values(by="Value",ascending=False)
    # print(df_importance.head(30))
    if FIRST_OOF_ONLY: break


# In[26]:


# weights
    # away 0 draw 1 home 2
    # train_weights = np.where(Y_train == 0,1.,np.where(Y_train == 1,1.25,1.))
    # val_weights =  np.where(Y_valid == 0,1.,np.where(Y_valid == 1,1.25,1.))
    # train_dataset = lgb.Dataset(X_train, Y_train, weight = train_weights) #, categorical_feature = ['is_cup'])
    # val_dataset = lgb.Dataset(X_valid, Y_valid, weight = val_weights)# , categorical_feature = ['is_cup'])


# ## Second LightGB model
# Only the last 5 matches...
# 
# Cancelled due to poor results.

# In[27]:


'''

# Do it again with up to 5 matches

train_x_last_5 = train_x_pivot.loc[train_x_pivot['time'] < 6]
test_x_last_5 = test_x_pivot.loc[train_x_pivot['time'] < 6]

train_x_agg = agg_features(train_x_last_5)
test_x_agg = agg_features(test_x_last_5)

# train_x_last = train_x_pivot.loc[train_x_pivot['time'] == 1]
# test_x_last = test_x_pivot.loc[train_x_pivot['time'] == 1]

# keep order
# Merge and drop columns
x_train = pd.merge(train_x_last, train_x_agg, left_on="id", right_on="id_").drop(['id', 'id_', 'time'], axis = 1).copy()
x_test = pd.merge(test_x_last, test_x_agg, left_on="id", right_on="id_").drop(['id', 'id_', 'time'], axis = 1).copy()
'''


# In[28]:


'''
# N_SPLITS of the traning set for validation using KFold
# Parameters
# Train model and predict
# models = []
# aucs = []
# preds = np.zeros(len(train))
#
num_boost_round = 1000
SEED = 123
N_SPLITS = 5
#
# Set LGBM hyper parameters
lgbm_params = { "objective":"multiclass"
    , "boosting_type":"gbdt"
    , 'num_class':CLASS
    , 'metric': 'multi_logloss'
    , "random_seed":SEED
    , "max_depth":5
    , "colsample_bytree":0.5
    , "subsample":0.5
    , "lambda_l1":0.1 #reg_alpha(L1)
    , "lambda_l2":0.9 #reg_lambda(L2)
    , "learning_rate":0.1
    , "num_leaves": 5
} 
'''

'''
# Params 2
seed = 2022
lgbm_params = {
    "objective":"multiclass" #"binary"
    , "boosting_type":"gbdt"
    , 'num_class':CLASS
    , 'metric': "multi_logloss" #''binary_logloss
    ,        'learning_rate': 0.1,        
            'lambda_l1': 2,
            'lambda_l2': 7,
            'num_leaves': 400,
            'min_sum_hessian_in_leaf': 20,
            'feature_fraction': 0.5,
            'feature_fraction_bynode': 0.5,
            'bagging_fraction': 0.25,
            'bagging_freq': 42,
            'min_data_in_leaf': 700,
            'max_depth': 10,
            'seed': seed,
            'feature_fraction_seed': seed,
            'bagging_seed': seed,
            'drop_seed': seed,
            'data_random_seed': seed,
            'verbosity': -1
        }



test_preds_2 = []

kf = StratifiedKFold(n_splits=N_SPLITS, random_state=seed, shuffle=True) # KFold

for fold, (train_idx, test_idx) in enumerate(kf.split(x_train, dummy_y)):
    print('-'*15, '>', f'Fold {fold+1}/{N_SPLITS}', '<', '-'*15)
    X_train, X_valid = x_train.iloc[train_idx], x_train.iloc[test_idx]
    Y_train, Y_valid = dummy_y.iloc[train_idx], dummy_y.iloc[test_idx]
    # 
    train_dataset = lgb.Dataset(X_train, Y_train) #, categorical_feature = ['is_cup'])
    val_dataset = lgb.Dataset(X_valid, Y_valid)# , categorical_feature = ['is_cup'])
    model = lgb.train(params = lgbm_params, 
                        train_set = train_dataset, 
                        valid_sets = [train_dataset, val_dataset], 
                        num_boost_round = num_boost_round, 
                        callbacks=[lgb.early_stopping(stopping_rounds=50)],
                        verbose_eval = 50)
    # Model validation    
    y_true = Y_valid.squeeze()
    y_pred = model.predict(X_valid).squeeze()
    score1 = log_loss(y_true, y_pred)
    print(f"Fold-{fold+1} | OOF LogLoss Score: {score1}")
    # print(roc_auc_score(y_true, y_pred))
    # Predictions
    test_preds_2.append(model.predict(x_test).squeeze())
    lgb.plot_importance(model,max_num_features=20)
'''


# In[29]:


preds = np.argmax(y_pred, axis=1) # preds = np.where(y_pred > 0.5, 1, 0)
cm = confusion_matrix(y_true, preds)
# # encoding away: 0 draw: 1 home:2
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels={'away', 'draw','home',})
disp.plot()
report = classification_report(y_true, preds)
print(report)


# In[30]:


predictions = sum(test_preds)/N_SPLITS

# away, draw, home
submission = pd.DataFrame(predictions,columns=['away', 'draw', 'home'])

# Round
round_num = False
if round_num:
    submission = submission.round(2)
    submission['draw'] = 1 - (submission['home'] + submission['away'])  
    
#do not forget the id column
submission['id'] = test[['id']]

#submit!
submission[['id', 'home', 'away', 'draw']].to_csv('submission.csv', index=False)


# In[31]:


submission[['id', 'home', 'away', 'draw']].head()


# ## Conclusion
# **Good luck!**
# 
# Report any error that you will probably find.
