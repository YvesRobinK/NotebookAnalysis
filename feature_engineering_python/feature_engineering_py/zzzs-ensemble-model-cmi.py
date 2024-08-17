#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from itertools import groupby
from sklearn.model_selection import train_test_split
from pandas.api.types import is_datetime64_ns_dtype
import numba

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, delayed
import gc
import plotly.express as px
from metric import score 

import warnings
warnings.filterwarnings("ignore")


# In[2]:


column_names = {
    'series_id_column_name': 'series_id',
    'time_column_name': 'step',
    'event_column_name': 'event',
    'score_column_name': 'score',
}

tolerances = {
    'onset': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360], 
    'wakeup': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360]
}


# In[3]:


def reduce_mem_usage(df):
    
    """ 
    Iterate through all numeric columns of a dataframe and modify the data type
    to reduce memory usage.        
    """
    
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and not is_datetime64_ns_dtype(df[col]) and not 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int32)  
            else:
                df[col] = df[col].astype(np.float16)
        
    return df


# # Feature Engineering
# 
# **Explanation for initial features ideas**: [notebook](https://www.kaggle.com/code/renatoreggiani/zzzs-feat-eng-ideas-60-memory-reduction) 

# In[4]:


def get_macd(s, hfast=4, hslow=48, macd=15, center=False):
    
    sma_fast = s.rolling(hfast*12*60, min_periods=1, center=center).agg('mean')
    sma_slow = s.rolling(hslow*12*60, min_periods=1, center=center).agg('mean')
    macd = (sma_fast - sma_slow).rolling(12*macd, min_periods=1, center=center).mean().astype(np.float32)
    
    return macd


def cls_zone(df, col='anglezdiffabs'):
    
    s = df[f'{col}']
    df['anglezdiffabs_macd'] = get_macd(s)

    s = df[f'{col}'].sort_index(ascending=False)
    df['anglezdiffabs_macd_rev'] = get_macd(s).sort_index()
    
    df['macd_spred'] = df['anglezdiffabs_macd']-df['anglezdiffabs_macd_rev']
    df['macd_spread_diff'] =df['macd_spred'].diff(60)
    
#     df.loc[df['macd_spred']<3, 'wakeup_zone'] = 1
#     df['wakeup_zone'].fillna(0, inplace=True)
#     df.loc[df['macd_spred']>-3, 'onset_zone'] = 1
#     df['onset_zone'].fillna(0, inplace=True)
    
    return df


def feat_eng(df):
    
    df['series_id'] = df['series_id'].astype('category')
    df['timestamp'] = pd.to_datetime(df['timestamp']).apply(lambda t: t.tz_localize(None))
    df['hour'] = df["timestamp"].dt.hour
    df['month'] = df["timestamp"].dt.month
    
    df.sort_values(['timestamp'], inplace=True)
    df.set_index('timestamp', inplace=True)
    
    # limit outiliers in enmo
    df['enmo'] = df['enmo'].clip(upper=4)
    
    df['lids'] = np.maximum(0., df['enmo'] - 0.02)
    df['lids'] = df['lids'].rolling(f'{120*5}s', center=True, min_periods=1).agg('sum')
    df['lids'] = 100 / (df['lids'] + 1)
    df['lids'] = df['lids'].rolling(f'{360*5}s', center=True, min_periods=1).agg('mean').astype(np.float32)
    
    df['lids_sma_fast'] = df['lids'].rolling('2h', center=True, min_periods=1).agg('mean').astype(np.float32)
    df['lids_sma_macd'] = (df['lids'].rolling('18h', center=True, min_periods=1).agg('mean') - df['lids_sma_fast']
                                 ).rolling(f'1h', center=True, min_periods=1).agg('mean').astype(np.float32)
    
    df["enmo"] = (df["enmo"]*1000).astype(np.int16)
    df["anglez"] = df["anglez"].astype(np.int16)
    df["anglezdiffabs"] = df["anglez"].diff().abs().astype(np.float32)
    
    df = cls_zone(df, col='anglezdiffabs')

    for col in ['enmo', 'anglez', 'anglezdiffabs']:
        
        df[f'{col}_sma_fast'] = df[f'{col}'].rolling(f'4h', center=True, min_periods=1).agg('mean').astype(np.float32)
        df[f'{col}_sf_macd'] = (df[f'{col}_sma_fast'] - df[f'{col}'].rolling(f'48h', center=True, min_periods=1).agg('mean')
                                   ).rolling(f'1h', center=True, min_periods=1).agg('mean').astype(np.float32)
        
        # periods in seconds        
        periods = [60, 300, 720, 1440, 3600] 
        
        for n in periods:
            
            rol_args = {'window':f'{n+5}s', 'min_periods':10, 'center':True}
            
            for agg in ['median', 'mean', 'max', 'min', 'std']:
                df[f'{col}_{agg}_{n}'] = df[col].rolling(**rol_args).agg(agg).astype(np.float32).values
                gc.collect()
            
            if n == max(periods):
                df[f'{col}_mad_{n}'] = (df[col] - df[f'{col}_median_{n}']).abs().rolling(**rol_args).median().astype(np.float32)
            
            df[f'{col}_amplit_{n}'] = df[f'{col}_max_{n}']-df[f'{col}_min_{n}']
            df[f'{col}_amplit_{n}_min'] = df[f'{col}_amplit_{n}'].rolling(**rol_args).min().astype(np.float32).values
            
            df[f'{col}_diff_{n}_max'] = df[f'{col}_max_{n}'].diff().abs().rolling(**rol_args).max().astype(np.float32)
            df[f'{col}_diff_{n}_mean'] = df[f'{col}_max_{n}'].diff().abs().rolling(**rol_args).mean().astype(np.float32)

    
            gc.collect()
        
#     df.drop(columns=[
#             'anglezdiffabs_min_720', 'anglezdiffabs_min_3600',
#             'anglezdiffabs_amplit_720', 'anglezdiffabs_min_60', 'anglezdiffabs_diff_60_max',
#             'anglezdiffabs','enmo_diff_60_max',
#             'anglez_median_60', 'anglezdiffabs_amplit_3600',
#             'enmo_max_60', 'enmo_amplit_720',
#             'anglezdiffabs_diff_60_mean', 'anglezdiffabs_max_720', 'enmo_median_60',
#             'anglez_amplit_720', 'enmo_mean_60', 
#             'anglez_mean_720', 'anglez_mean_60',
#             'anglez_var_720', 'anglez_diff_60_max', 'anglez_diff_720_max',
#             'enmo_min_60', 'enmo_max_720', 'enmo_min_720', 'anglez_diff_720_mean',
#             'anglez_diff_60_mean', 'anglez_max_720', 'anglezdiffabs_var_720',
#             'anglezdiffabs_diff_720_mean', 'enmo_diff_720_mean', 'anglezdiffabs_min_1440', 'anglezdiffabs_min_300',
#             'anglezdiffabs_amplit_1440', 'anglezdiffabs_amplit_300',
#             'anglezdiffabs_diff_300_mean', 'enmo_diff_300_mean',
#             'anglezdiffabs_max_300', 'anglezdiffabs_diff_300_max',
#             'anglez_diff_300_max', 'enmo_diff_300_max', 'anglez_amplit_300',
#             'anglez_diff_300_mean', 'enmo_amplit_300', 'anglezdiffabs_var_300',
#             'enmo_diff_60_mean', 'anglez_mean_300', 'anglez_max_60', 'enmo_var_300',
#             'anglez_median_300', 'enmo_median_300', 'enmo_amplit_1440',
#             'enmo_median_720', 'anglez_amplit_1440', 'anglez_mean_1440',
#             'enmo_max_1440', 'enmo_max_300', 'anglez_max_300', 'enmo_min_1440',
#             'anglez_var_300', 'enmo_min_3600'
#                     ], inplace=True)
    
    rol_args = {'window':f'8h', 'min_periods':15, 'center':True}
    df['not_wear'] = df[f'anglez_std_720'].rolling(**rol_args).max().astype(np.float32)
    
    
    df.reset_index(inplace=True)
    df.dropna(inplace=True)
    
#     df = df[(df['wakeup_zone']==1)|(df['onset_zone']==1)]
    

    return df


# In[5]:


file = '/kaggle/input/zzzs-lightweight-training-dataset-target/Zzzs_train.parquet'

def feat_eng_by_id(idx):
    
    from warnings import simplefilter 
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    
    df  = pd.read_parquet(file, filters=[('series_id','=',idx)])
    df['awake'] = df['awake'].astype(np.int8)
    df = feat_eng(df)
    
    return df


# In[6]:


DEV = False

series_id  = pd.read_parquet(file, columns=['series_id'])
series_id = series_id.series_id.unique()

print('nunique id', len(series_id))

if DEV:
    series_id = series_id[::10]
    print('DEV nunique ids:', len(series_id))


# In[7]:


weird_series = ['31011ade7c0a', 'a596ad0b82aa']

series_id = [s for s in series_id if s not in weird_series]


# In[8]:


get_ipython().run_cell_magic('time', '', '\ntrain = Parallel(n_jobs=-1)(delayed(feat_eng_by_id)(i) for i in series_id)\ntrain = pd.concat(train, ignore_index=True)\n')


# In[9]:


train.shape


# In[10]:


train = train.loc[train['not_wear']>22]
train.shape


# In[11]:


drop_cols = ['series_id', 'step', 'timestamp']

X, y = train.drop(columns=drop_cols+['awake']), train['awake']
# reduce train dataset by half
step = 60
X, y = X.iloc[::step], y[::step]
step


# In[12]:


y.value_counts()


# In[13]:


del train
gc.collect()


# # Ensemble Model

# In[14]:


class EnsembleAvgProba():
    
    def __init__(self, classifiers):
        
        self.classifiers = classifiers
    
    def fit(self,X,y):
        
        for classifier in self.classifiers:                
            classifier.fit(X, y)
            gc.collect()
     
    def predict_proba(self, X):
        
        probs = []
        
        for m in self.classifiers:
            probs.append(m.predict_proba(X))
        
        probabilities = np.stack(probs)
        p = np.mean(probabilities, axis=0)
        
        return p
    
    def predict(self, X):
        
        probs = []
        
        for m in self.classifiers:
            probs.append(m.predict(X))
        
        probabilities = np.stack(probs)
        p = np.median(probabilities, axis=0)
        
        return p.round()


# In[15]:


X.head()


# In[16]:


lgb_params1 = {    
    'boosting_type':'gbdt',
    'num_leaves':31,
    'max_depth':6,
    'learning_rate':0.03,
    'n_estimators':500,
    'subsample_for_bin':200000,
    'min_child_weight':0.001,
    'min_child_samples':20,
    'subsample':0.6,
#     'colsample_bytree':0.7,
    'reg_alpha':0.025,
    'reg_lambda':0.025,
             }

model = EnsembleAvgProba(classifiers=[
                    lgb.LGBMClassifier(random_state=42, **lgb_params1),
                    GradientBoostingClassifier(n_estimators=100,max_depth=5,min_samples_leaf=300,random_state=42),
                    RandomForestClassifier(n_estimators=500, min_samples_leaf=300, random_state=42, n_jobs=-1),
                    xgb.XGBClassifier(n_estimators=520,objective="binary:logistic", learning_rate=0.02, max_depth=7, random_state=42)    
])


# In[17]:


get_ipython().run_cell_magic('time', '', '\nmodel.fit(X, y)\n')


# # Features importance
# 
# We can vizualize all fetures in this [notebook](https://www.kaggle.com/code/renatoreggiani/zzzs-interactive-plot-all-feat-ideas), is helpfull to get more insights

# In[18]:


feats = []

for m in model.classifiers:
    feat_imp = m.feature_importances_
    
    feat_imp = MinMaxScaler().fit_transform(feat_imp.reshape(-1, 1))
    feat_imp = pd.Series(pd.Series(feat_imp.reshape(1, -1)[0], index=X.columns).sort_values(), index=X.columns).sort_values()
    feats.append(feat_imp)


# In[19]:


feat_imp = pd.Series(pd.concat(feats, axis=1).mean(axis=1), index=feats[0].index).sort_values()
print('Columns with poor contribution', feat_imp[feat_imp<0.005].index, sep='\n')
fig = px.bar(x=feat_imp, y=feat_imp.index, orientation='h')
fig.show()


# In[20]:


feat_imp.sort_values().head(10)


# In[21]:


# del X, y
gc.collect()


# # Function to get events

# In[22]:


# def get_events(idx, classifier, file='test_series.parquet') :
    
#     test  = pd.read_parquet(f'/kaggle/input/child-mind-institute-detect-sleep-states/{file}',
#                     filters=[('series_id','=',idx)])
    
#     test = feat_eng(test)
#     test = test[test[f'anglez_std_max_4h']>21]
    
# #     test_daily = test.groupby(test['timestamp'].dt.date)[['anglez', 'enmo']].std()
# #     min_daily_enmo = test_daily['enmo']<5
# #     days_to_drop = test_daily[min_daily_enmo].index
# #     test = test[~test['timestamp'].dt.date.isin(days_to_drop)]
        
# #     test = test[test['enmo_max_360']>2]
# #     test = test[test['anglez_std_360']>2]
    
#     X_test = test.drop(columns=drop_cols)
#     test = test[drop_cols]
    
#     if test.shape[0]>0:

#         preds, probs = classifier.predict(X_test), classifier.predict_proba(X_test)[:, 1]

#         test['prediction'] = preds
#         test['prediction'] = test['prediction'].rolling(120+1, center=True).median()
#         test['probability'] = probs

# #         test = test[test['prediction']!=2]

#         test.loc[test['prediction']==0, 'probability'] = 1-test.loc[test['prediction']==0, 'probability']
#         test['score'] = test['probability'].rolling(60*12*5, center=True, min_periods=10).mean().bfill().ffill()
#         test['pred_diff'] = test['prediction'].diff()
#         test['event'] = test['pred_diff'].replace({1:'wakeup', -1:'onset', 0:np.nan})

#         test_wakeup = test[test['event']=='wakeup'].groupby(test['timestamp'].dt.date).agg('first')
#         test_onset = test[test['event']=='onset'].groupby(test['timestamp'].dt.date).agg('last')
#         test = pd.concat([test_wakeup, test_onset], ignore_index=True).sort_values('timestamp')
#         if test.shape[0]>0:
#             if test['event'].values[0]=='wakeup':
#                 test = test[1:]

#     return test

# get_events(series_id[0], model)


# In[23]:


def get_events(idx, classifier, file='test_series.parquet') :
    
    test  = pd.read_parquet(f'/kaggle/input/child-mind-institute-detect-sleep-states/{file}',
                    filters=[('series_id','=',idx)])
    
    test = feat_eng(test)
    
    X_test = test.drop(columns=drop_cols)
    
    if test.shape[0]>0:

        preds, probs = classifier.predict(X_test), classifier.predict_proba(X_test)[:, 1]

        test['prediction'] = preds
        test['prediction'] = test['prediction'].rolling(60+1, center=True).median()
        test['probability'] = probs

        test = test[test['prediction']!=2]

        test.loc[test['prediction']==0, 'probability'] = 1-test.loc[test['prediction']==0, 'probability']
        test['score'] = test['probability'].rolling(60*12*5, center=True, min_periods=10).mean().bfill().ffill()
        test['pred_diff'] = test['prediction'].diff()
        
        test['event'] = test['pred_diff'].replace({1:'wakeup', -1:'onset', 0:np.nan})
        test.dropna(subset='event', inplace=True)
        test['step_diff'] = test['step'].diff()
        test = test.loc[test['not_wear']>20]
        test.sort_values('step_diff', inplace=True)
        

        test['day'] = ((test['step']//17280)+1).astype(np.int32)
        
        
        mask_wk= (test['event']=='wakeup') #& (test['wakeup_zone']==1)
        test_wakeup = test[mask_wk].groupby('day', as_index=False).agg('last')
        
        mask_os= (test['event']=='onset') #& (test['onset_zone']==1)
        test_onset = test[mask_os].groupby('day', as_index=False).agg('last')

        
        df_events = pd.concat([test_wakeup, test_onset], ignore_index=True).sort_values('step')
        
        
        if df_events.shape[0]>0:
    
            if df_events.iloc[0]['event'] == 'wakeup':
                df_events = df_events.iloc[1:]

    return df_events


# In[24]:


get_ipython().run_cell_magic('time', '', "\ncols_sub = ['series_id','step','event','score']\n\nseries_id  = pd.read_parquet('/kaggle/input/child-mind-institute-detect-sleep-states/test_series.parquet', columns=['series_id'])\nseries_id = series_id.series_id.unique()\n\ntests = Parallel(n_jobs=-1)(delayed(get_events)(i, model) for i in series_id)\n")


# # Submission

# In[25]:


submission = pd.concat(tests, ignore_index=True)
submission = submission[cols_sub].reset_index(names='row_id')
submission.to_csv('submission.csv', index=False)
submission


# In[26]:


# %%time

# if DEV:
    
#     print('validation')

#     series_id  = pd.read_parquet('/kaggle/input/child-mind-institute-detect-sleep-states/train_series.parquet', columns=['series_id'])
#     series_id = series_id.series_id.unique()

#     path = '/kaggle/input/child-mind-institute-detect-sleep-states/'

#     train_events = pd.read_csv(path + 'train_events.csv')
#     train_events['timestamp'] = pd.to_datetime(train_events['timestamp']).apply(lambda t: t.tz_localize(None))

#     val_solution = train_events[train_events['series_id'].isin(series_id)]

#     vals = Parallel(n_jobs=-1)(delayed(get_events)(i, model, file='train_series.parquet') for i in series_id)

#     val=pd.concat(vals, ignore_index=True).reset_index(names='row_id')

#     print(f"LGBM score: {score(val_solution, val, tolerances, **column_names)}")

#     val['score'] = 1
#     print(f"LGBM score: {score(val_solution, val, tolerances, **column_names)}")


# In[ ]:




