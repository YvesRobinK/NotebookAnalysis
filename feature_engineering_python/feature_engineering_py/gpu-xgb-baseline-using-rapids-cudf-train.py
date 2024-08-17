#!/usr/bin/env python
# coding: utf-8

# # Using GPU - Part 1
# ## Feature Engineering and XGBoost Training using GPU
# 
# This is Part 1 of the implementation of Chris's idea to use GPU Kaggle notebook for feature engineering during training and then CPU for inference from the discussion [here][1]. Part 2 Notebook is used to make submission using CPU.
# 
# Changes made-
# 
# * I modified Chris's XGBoost Baseline [notebook][2] to use RAPIDS cuDF. Feature Engineering gets really fast with the help of RAPIDS cuDF. In the CPU notebook, feature engineering takes about 1 min but utilizing the power of GPU takes the time down to like **3 seconds**!!!
# * Changed the XGBoost `tree_method` from `'hist'` to `'gpu hist'`. This makes training faster too.
# * Added some features to get a better score.
# 
# I wasn't able to import cudf in the latest Kaggle Notebook environment (don't know why), so I copied [this][3] notebook and used it's environment instead.
# 
# I have written a comment wherever I made a change.
# 
# ## RAPIDS cuDF
# 
# cuDF is a Python GPU DataFrame library which provides a pandas-like API. So just importing it as pd allows us to use cuDF without changing the pandas code much. 
# 
# But to use scikit learn functions on cudf dataframes, we have to convert the cudf df to pandas df by calling `to_pandas()`.
# 
# To learn more about how cuDF work, check out cuDF’s documentation [here][4]. 
# 
# [1]: https://www.kaggle.com/competitions/predict-student-performance-from-game-play/discussion/386218
# [2]: https://www.kaggle.com/code/cdeotte/xgboost-baseline-0-676
# [3]: https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575?scriptVersionId=111214204
# [4]: https://docs.rapids.ai/api/cudf/stable/

# ## Version Updates:
# 
# **Version 4:**
# Changed the structure of the notebook and added features with the help of [this][1] amazing notebook by @takanashihumbert
# 
# [1]: https://www.kaggle.com/code/takanashihumbert/magic-bingo-train-part-lb-0-687

# In[1]:


import cudf as pd #Change1
import numpy as np
from sklearn.model_selection import KFold, GroupKFold
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from collections import defaultdict
import warnings
from itertools import combinations
import gc
import pickle

print('We will use RAPIDS version',pd.__version__)


# ## Load Train Data and Labels

# In[2]:


train = pd.read_csv('/kaggle/input/predict-student-performance-from-game-play/train.csv')
print( train.shape )
train.head()


# In[3]:


targets = pd.read_csv('/kaggle/input/predict-student-performance-from-game-play/train_labels.csv')
targets['session'] = pd.to_numeric( targets.session_id.str.split('_').list.get(0) ) #Change2
targets['q'] = pd.to_numeric( targets.session_id.str.split('_q').list.get(1) ) #Change3


# In[4]:


print(targets.shape)
targets.head()


# ## Feature Engineer

# In[5]:


#Calculate Elapsed Time Difference Column
trainp = train.to_pandas()
trainp['time_diff'] = (trainp['elapsed_time'] - trainp.groupby(['session_id','level_group'])['elapsed_time'].shift(1)).clip(0,1e7)
train = pd.DataFrame(trainp)
train.loc[train['time_diff']<0] = 0


# In[6]:


train1 = train[train["level_group"]=='0-4']
train2 = train[train["level_group"]=='5-12']
train3 = train[train["level_group"]=='13-22']


# In[7]:


CATS = ['event_name','name','fqid','room_fqid','text_fqid']

NUMS = ['page', 'room_coor_x','room_coor_y','screen_coor_x','screen_coor_y','hover_duration','time_diff']

EVENTS = ['cutscene_click', 'person_click', 'navigate_click',
       'observation_click', 'notification_click', 'object_click',
       'object_hover', 'map_hover', 'map_click', 'checkpoint',
       'notebook_click']

NAMES = ['basic', 'undefined', 'close', 'open', 'prev', 'next']


# In[8]:


def feature_engineer(x, grp):
    
    x['elapsed_time'] = x['elapsed_time'] / 1000
    x['time_diff'] = x['time_diff'] / 1000
    
    #session duration
    df_final = x.groupby('session_id')['index'].agg('count')
    df_final.name = 'num_events'
    df_final = df_final.reset_index()
    df_final = df_final.set_index('session_id')
    
    #Bingo Features
    if grp == '5-12':
        
        df_final['logbingo-logbook'] = x[(x['fqid']=='logbook.page.bingo')&(x['event_name']=='object_click')].groupby('session_id')['index'].agg('first') - x[x['fqid']=='logbook'].groupby('session_id')['index'].agg('first')
        df_final['readerbingo-reader'] = x[(x['fqid']=='reader.paper2.bingo')&(x['event_name']=='object_click')].groupby('session_id')['index'].agg('first') - x[x['fqid']=='reader'].groupby('session_id')['index'].agg('first')
        df_final['jourbingo-journalspic'] = x[(x['fqid']=='journals.pic_2.bingo')&(x['event_name']=='object_click')].groupby('session_id')['index'].agg('first') - x[x['fqid']=='journals.pic_0.next'].groupby('session_id')['index'].agg('first')
        
        df_final['logbingo-logbook_time'] = x[(x['fqid']=='logbook.page.bingo')&(x['event_name']=='object_click')].groupby('session_id')['elapsed_time'].agg('first') - x[x['fqid']=='logbook'].groupby('session_id')['elapsed_time'].agg('first')
        df_final['readerbingo-reader_time'] = x[(x['fqid']=='reader.paper2.bingo')&(x['event_name']=='object_click')].groupby('session_id')['elapsed_time'].agg('first') - x[x['fqid']=='reader'].groupby('session_id')['elapsed_time'].agg('first')
        df_final['jourbingo-journalspic_time'] = x[(x['fqid']=='journals.pic_2.bingo')&(x['event_name']=='object_click')].groupby('session_id')['elapsed_time'].agg('first') - x[x['fqid']=='journals.pic_0.next'].groupby('session_id')['elapsed_time'].agg('first')
        
    if grp=='13-22':
        
        df_final['readerbingo-reader_flag'] = x[(x['fqid']=='reader_flag.paper2.bingo')&(x['event_name']=='object_click')].groupby('session_id')['index'].agg('first') - x[x['fqid']=='reader_flag'].groupby('session_id')['index'].agg('first')
        df_final['journalbingo-journals_flag'] = x[(x['fqid']=='journals_flag.pic_0.bingo')&(x['event_name']=='object_click')].groupby('session_id')['index'].agg('first') - x[x['fqid']=='journals_flag'].groupby('session_id')['index'].agg('first')
        
        df_final['readerbingo-reader_flag_time'] = x[(x['fqid']=='reader_flag.paper2.bingo')&(x['event_name']=='object_click')].groupby('session_id')['elapsed_time'].agg('first') - x[x['fqid']=='reader_flag'].groupby('session_id')['elapsed_time'].agg('first')
        df_final['journalbingo-journals_flag_time'] = x[(x['fqid']=='journals_flag.pic_0.bingo')&(x['event_name']=='object_click')].groupby('session_id')['elapsed_time'].agg('first') - x[x['fqid']=='journals_flag'].groupby('session_id')['elapsed_time'].agg('first')
        
    df_final['first_elapsed_time'] = x.groupby('session_id')['elapsed_time'].agg('first')
    df_final['elapsed_time'] = x.groupby('session_id')['elapsed_time'].agg('last') - df_final['first_elapsed_time']
    
    for c in CATS:
        df_final[f'{c}_nuniques'] = x.groupby('session_id')[c].agg('nunique')
    
    for c in NUMS:
        df_final[f'{c}_mean'] = x.groupby('session_id')[c].agg('mean')
        df_final[f'{c}_min'] = x.groupby('session_id')[c].agg('min')
        df_final[f'{c}_max'] = x.groupby('session_id')[c].agg('max')
        
    for c in EVENTS:
        x[c] = (x.event_name == c).astype('int8')
    for c in EVENTS:
        df_final[f'{c}_sum'] = x.groupby('session_id')[c].agg('sum')
    x.drop(EVENTS, axis=1, inplace=True)
    
    for c in EVENTS:
        df_final[f'{c}_time_mean'] = x[x['event_name']==c].groupby('session_id')['time_diff'].mean()
        df_final[f'{c}_time_min'] = x[x['event_name']==c].groupby('session_id')['time_diff'].min()
        df_final[f'{c}_time_max'] = x[x['event_name']==c].groupby('session_id')['time_diff'].max()
    
    for c in NAMES:
        x[c] = (x.name == c).astype('int8')
    for c in NAMES:
        df_final[f'{c}_sum'] = x.groupby('session_id')[c].agg('sum')
    x.drop(NAMES, axis=1, inplace=True)
    
    for c in NAMES:
        df_final[f'{c}_time_mean'] = x[x['name']==c].groupby('session_id')['time_diff'].mean()
        df_final[f'{c}_time_min'] = x[x['name']==c].groupby('session_id')['time_diff'].min()
        df_final[f'{c}_time_max'] = x[x['name']==c].groupby('session_id')['time_diff'].max()

    return df_final


# In[9]:


get_ipython().run_cell_magic('time', '', "df1 = feature_engineer(train1.copy(), grp='0-4')\nprint('df1 done')\ndf2 = feature_engineer(train2.copy(), grp='5-12')\nprint('df2 done')\ndf3 = feature_engineer(train3.copy(), grp='13-22')\nprint('df3 done')\n")


# In[10]:


null1 = df1.isnull().sum().sort_values(ascending=False) / len(df1)
null2 = df2.isnull().sum().sort_values(ascending=False) / len(df1)
null3 = df3.isnull().sum().sort_values(ascending=False) / len(df1)

drop1 = list(null1[null1>0.9].index.to_pandas())
drop2 = list(null2[null2>0.9].index.to_pandas())
drop3 = list(null3[null3>0.9].index.to_pandas())
print(len(drop1), len(drop2), len(drop3))

for col in tqdm(df1.columns):
    if df1[col].nunique()==1:
        print(col)
        drop1.append(col)
print("*********df1 DONE*********")
for col in tqdm(df2.columns):
    if df2[col].nunique()==1:
        print(col)
        drop2.append(col)
print("*********df2 DONE*********")
for col in tqdm(df3.columns):
    if df3[col].nunique()==1:
        print(col)
        drop3.append(col)
print("*********df3 DONE*********")


# ## Train XGBoost Model

# In[11]:


FEATURES1 = [c for c in df1.columns if c not in drop1+['level_group']]
FEATURES2 = [c for c in df2.columns if c not in drop2+['level_group','first_index']]
FEATURES3 = [c for c in df3.columns if c not in drop3+['level_group']]
print('We will train with', len(FEATURES1), len(FEATURES2), len(FEATURES3) ,'features')
ALL_USERS = df1.index.unique()
print('We will train with', len(ALL_USERS) ,'users info')


# In[12]:


gkf = GroupKFold(n_splits=5)
oof = pd.DataFrame(data=np.zeros((len(ALL_USERS), 18)),index=ALL_USERS)


# In[13]:


get_ipython().run_cell_magic('time', '', 'gkf = GroupKFold(n_splits=5)\noof_xgb = pd.DataFrame(data=np.zeros((len(ALL_USERS),18)), index=ALL_USERS, columns=[f\'meta_{i}\' for i in range(1, 19)])\n#models = {}\nbest_iteration_xgb = defaultdict(list)\nimportance_dict = {}\n\n# ITERATE THRU QUESTIONS 1 THRU 18\nfor t in range(1,19):\n\n    # USE THIS TRAIN DATA WITH THESE QUESTIONS\n    if t<=3: \n        grp = \'0-4\'\n        df = df1\n        FEATURES = FEATURES1\n    elif t<=13: \n        grp = \'5-12\'\n        df = df2\n        FEATURES = FEATURES2\n    elif t<=22: \n        grp = \'13-22\'\n        df = df3\n        FEATURES = FEATURES3\n        \n    print(\'#\'*25)\n    print(\'### question\', t, \'with features\', len(FEATURES))\n    print(\'#\'*25)\n    \n    xgb_params = {\n        \'booster\': \'gbtree\',\n        \'objective\': \'binary:logistic\',\n        \'tree_method\': \'gpu_hist\', #Change4\n        \'eval_metric\':\'logloss\',\n        \'learning_rate\': 0.02,\n        \'alpha\': 8,\n        \'max_depth\': 4,\n        \'n_estimators\': 9999,\n        \'early_stopping_rounds\': 90,\n        \'subsample\':0.8,\n        \'colsample_bytree\': 0.5,\n        \'seed\': 42\n    }\n\n    feature_importance_df = pd.DataFrame()\n    # COMPUTE CV SCORE WITH 5 GROUP K FOLD\n    P1 = df.iloc[:,0].to_pandas() #Change5\n    P2 = df.index.to_pandas() #Change6\n    for i, (train_index, test_index) in enumerate(gkf.split(X=P1, groups=P2)): #Change7\n        \n        # TRAIN DATA\n        train_x = df.iloc[train_index]\n        train_users = train_x.index.values\n        train_y = targets.loc[targets.q==t].set_index(\'session\').loc[train_users]\n        \n        # VALID DATA\n        valid_x = df.iloc[test_index]\n        valid_users = valid_x.index.values\n        valid_y = targets.loc[targets.q==t].set_index(\'session\').loc[valid_users]\n        \n        # TRAIN MODEL        \n        clf =  XGBClassifier(**xgb_params)\n        clf.fit(train_x[FEATURES].astype(\'float32\'), train_y[\'correct\'],\n                eval_set=[(valid_x[FEATURES].astype(\'float32\'), valid_y[\'correct\'])],\n                verbose=0)\n        print(i+1, \', \', end=\'\')\n        best_iteration_xgb[str(t)].append(clf.best_ntree_limit)\n        \n        fold_importance_df = pd.DataFrame()\n        fold_importance_df["feature"] = FEATURES\n        fold_importance_df["importance"] = clf.feature_importances_\n        fold_importance_df["fold"] = i + 1\n        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)\n        \n        # SAVE MODEL, PREDICT VALID OOF\n        oof_xgb.loc[valid_users, f\'meta_{t}\'] = clf.predict_proba(valid_x[FEATURES].astype(\'float32\'))[:,1]\n            \n    print()\n    feature_importance_df = feature_importance_df.groupby([\'feature\'])[\'importance\'].agg([\'mean\']).sort_values(by=\'mean\', ascending=False)\n')


# ## Compute CV Score

# In[14]:


true = oof_xgb.copy()
for i in range(1, 19):
    # GET TRUE LABELS
    tmp = targets.loc[targets.q==i].set_index('session').loc[ALL_USERS]
    true[f'meta_{i}'] = tmp.correct.values

# FIND BEST THRESHOLD TO CONVERT PROBS INTO 1s AND 0s
scores = []; thresholds = []
best_score_xgb = 0; best_threshold_xgb = 0

for threshold in np.arange(0.4,0.81,0.005):
    print(f'{threshold:.03f}, ',end='')
    preds = (oof_xgb.to_pandas().values.reshape((-1))>threshold).astype('int') #Change8
    m = f1_score(true.to_pandas().values.reshape((-1)), preds, average='macro') #Change9
    scores.append(m)
    thresholds.append(threshold)
    if m>best_score_xgb:
        best_score_xgb = m
        best_threshold_xgb = threshold

# PLOT THRESHOLD VS. F1_SCORE
plt.figure(figsize=(20,5))
plt.plot(thresholds,scores,'-o',color='blue')
plt.scatter([best_threshold_xgb], [best_score_xgb], color='blue', s=300, alpha=1)
plt.xlabel('Threshold',size=14)
plt.ylabel('Validation F1 Score',size=14)
plt.title(f'Threshold vs. F1_Score with Best F1_Score = {best_score_xgb:.5f} at Best Threshold = {best_threshold_xgb:.4}',size=18)
plt.show()


# In[15]:


print('When using optimal threshold...')
for k in range(18):
    m = f1_score(true[f'meta_{k+1}'].to_pandas().values, (oof_xgb[f'meta_{k+1}'].to_pandas().values>best_threshold_xgb).astype('int'), average='macro') #Change10
    print(f'Q{k}: F1 =',m)
    
m = f1_score(true.to_pandas().values.reshape((-1)), (oof_xgb.to_pandas().values.reshape((-1))>best_threshold_xgb).astype('int'), average='macro') #Change11
print('==> Overall F1 =',m)


# In[16]:


get_ipython().run_cell_magic('time', '', "# ITERATE THRU QUESTIONS 1 THRU 18\nfor t in range(1,19):\n\n    # USE THIS TRAIN DATA WITH THESE QUESTIONS\n    if t<=3: \n        grp = '0-4'\n        df = df1\n        FEATURES = FEATURES1\n    elif t<=13: \n        grp = '5-12'\n        df = df2\n        FEATURES = FEATURES2\n    elif t<=22: \n        grp = '13-22'\n        df = df3\n        FEATURES = FEATURES3\n    \n    n_estimators = int(np.median(best_iteration_xgb[str(t)]) + 1)\n    xgb_params = {\n        'objective': 'binary:logistic',\n        'tree_method': 'gpu_hist',\n        'eval_metric':'logloss',\n        'learning_rate': 0.02,\n        'alpha': 8,\n        'max_depth': 4,\n        'n_estimators': n_estimators,\n        'subsample':0.8,\n        'colsample_bytree': 0.5,\n    }\n    \n    print('#'*25)\n    print(f'### question {t} features {len(FEATURES)}')\n        \n    # TRAIN DATA\n    train_users = df.index.values\n    train_y = targets.loc[targets.q==t].set_index('session').loc[train_users]\n\n    # TRAIN MODEL        \n    clf =  XGBClassifier(**xgb_params)\n    clf.fit(df[FEATURES].astype('float32'), train_y['correct'], verbose=0)\n    clf.save_model(f'XGB_question{t}.xgb')\n    \n    print()\n")


# In[17]:


importance_dict = {}
for t in range(1, 19):
    if t<=3: 
        importance_dict[str(t)] = FEATURES1
    elif t<=13: 
        importance_dict[str(t)] = FEATURES2
    elif t<=22:
        importance_dict[str(t)] = FEATURES3

f_save = open('importance_dict.pkl', 'wb')
pickle.dump(importance_dict, f_save)
f_save.close()


# ### *Check the Part 2 - Inference Notebook to make a submission!!*
