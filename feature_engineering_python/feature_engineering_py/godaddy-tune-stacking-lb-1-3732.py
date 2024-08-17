#!/usr/bin/env python
# coding: utf-8

# ### **Thanks to great works from Prem, Jasonczh, Chris, Jasonczh, Giba, Nin7a1, judith007,samu2505, tetsutani.**

# ## **This notebook mainly tuned the models and explain the reasons of doing these things. Hope my words can help you in this competition!**

# ### **Thanks to great works from Prem, Jasonczh, Chris, Giba, Nin7a1, judith007,samu2505, tetsutani, DANGNGUYEN1997, .**
# 
# ### **This notebook mainly tuned the models and explain the reasons of doing these things. Hope my words can help you in this competition!**
# 
# ### This version is for content. The best score can be found [here](https://www.kaggle.com/code/batprem/godaddy-tune-stacking?scriptVersionId=121080529)
# 
# #### <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#9E3F00; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #9E3F00">CHANGE LOG</p>
# * ** Added tuning from https://www.kaggle.com/code/vadimkamaev/godaddy-stacking-xgb-lgbm-cat**
# * ** Added stacking from https://www.kaggle.com/code/samu2505/godaddy-stacking-xgb-lgbm-cat**
# * ** Added ensemble with https://www.kaggle.com/code/dangnguyen97/lb-1-3803-simple-baseline-with-eda-and-smape?scriptVersionId=120991448**
# * **Added data from https://www2.census.gov/programs-surveys/popest/datasets/2020-2021/counties/totals/ please see [Feature engineeering](#Feature-Engineering)**
# > It didn't work before the data was adjusted. It now somehow works. - Prem
# 
# #### Mixing technique
# This technique is a mixing technique of
# 
# * @judith007
#     * https://www.kaggle.com/code/judith007/location-feature-single-model-improvement
#     * https://www.kaggle.com/code/judith007/tuning-methods-included-better-score-lb1-3946
# * @samu2505 https://www.kaggle.com/code/samu2505/adding-location-features
# * @tetsutani https://www.kaggle.com/code/tetsutani/catboost-only-tune-score-lb1-3845
# <br>Also thank to greate works from Jasonczh, Chris, Jasonczh, Giba, Nin7a1

# In[1]:


import gc
import numpy as np
import pandas as pd
import os
from tqdm.notebook import tqdm
BASE = '../input/godaddy-microbusiness-density-forecasting/'

def smape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    
    pos_ind = (y_true!=0)|(y_pred!=0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    
    return 100 * np.mean(smap)

def vsmape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    
    pos_ind = (y_true!=0)|(y_pred!=0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    
    return 100 * smap


# In[2]:


census = pd.read_csv(BASE + 'census_starter.csv')
train = pd.read_csv(BASE + 'train.csv')
reaveal_test = pd.read_csv(BASE + 'revealed_test.csv')
train = pd.concat([train, reaveal_test]).sort_values(by=['cfips','first_day_of_month']).reset_index()
test = pd.read_csv(BASE + 'test.csv')
drop_index = (test.first_day_of_month == '2022-11-01') | (test.first_day_of_month == '2022-12-01')
test = test.loc[~drop_index,:]

sub = pd.read_csv(BASE + 'sample_submission.csv')
coords = pd.read_csv("/kaggle/input/usa-counties-coordinates/cfips_location.csv")
print(train.shape, test.shape, sub.shape)

train['istest'] = 0
test['istest'] = 1
raw = pd.concat((train, test)).sort_values(['cfips','row_id']).reset_index(drop=True)
raw = raw.merge(coords.drop("name", axis=1), on="cfips")

raw['state_i1'] = raw['state'].astype('category')
raw['county_i1'] = raw['county'].astype('category')
raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])
raw['county'] = raw.groupby('cfips')['county'].ffill()
raw['state'] = raw.groupby('cfips')['state'].ffill()
raw["dcount"] = raw.groupby(['cfips'])['row_id'].cumcount()
raw['county_i'] = (raw['county'] + raw['state']).factorize()[0]
raw['state_i'] = raw['state'].factorize()[0]
raw['scale'] = (raw['first_day_of_month'] - raw['first_day_of_month'].min()).dt.days
raw['scale'] = raw['scale'].factorize()[0]
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# # There are some anomalies, specially at timestep 18

# In[3]:


for o in tqdm(raw.cfips.unique()): 
    indices = (raw['cfips'] == o) 
    tmp = raw.loc[indices].copy().reset_index(drop=True)
    var = tmp.microbusiness_density.values.copy()
    for i in range(37, 2, -1):
        thr = 0.10 * np.mean(var[:i]) 
        difa = var[i] - var[i - 1] 
        if (difa >= thr) or (difa <= -thr):              
            if difa > 0:
                var[:i] += difa - 0.003 
            else:
                var[:i] += difa + 0.003  
    var[0] = var[1] * 0.99
    raw.loc[indices, 'microbusiness_density'] = var


# In[4]:


lag = 1
raw[f'mbd_lag_{lag}'] = raw.groupby('cfips')['microbusiness_density'].shift(lag).bfill()
raw['dif'] = (raw['microbusiness_density'] / raw[f'mbd_lag_{lag}']).fillna(1).clip(0, None) - 1
raw.loc[(raw[f'mbd_lag_{lag}']==0), 'dif'] = 0
raw.loc[(raw[f'microbusiness_density']>0) & (raw[f'mbd_lag_{lag}']==0), 'dif'] = 1
raw['dif'] = raw['dif'].abs()
# raw.groupby('dcount')['dif'].sum().plot()


# # SMAPE is a relative metric so target must be converted.

# In[5]:


raw['target'] = raw.groupby('cfips')['microbusiness_density'].shift(-1)
raw['target'] = raw['target']/raw['microbusiness_density'] - 1


raw.loc[raw['cfips']==28055, 'target'] = 0.0
raw.loc[raw['cfips']==48269, 'target'] = 0.0


# In[6]:


raw['lastactive'] = raw.groupby('cfips')['active'].transform('last')

# dt = raw.loc[raw.dcount==40].groupby('cfips')['microbusiness_density'].agg('last')
# raw['lastactive'].clip(0, 8000).hist(bins=30)


# # Feature Engineering
# > Try tuning this part

# In[7]:


def build_features(raw, target='microbusiness_density', target_act='active_tmp', lags = 6):
    feats = []   

    for lag in range(1, lags):
        raw[f'mbd_lag_{lag}'] = raw.groupby('cfips')[target].shift(lag)
        raw[f'act_lag_{lag}'] = raw.groupby('cfips')[target_act].diff(lag)
        feats.append(f'mbd_lag_{lag}')
        feats.append(f'act_lag_{lag}')
        
    lag = 1
    for window in [2, 4, 6, 8, 10]:
        raw[f'mbd_rollmea{window}_{lag}'] = raw.groupby('cfips')[f'mbd_lag_{lag}'].transform(lambda s: s.rolling(window, min_periods=1).sum())        
        feats.append(f'mbd_rollmea{window}_{lag}')
    
    census_columns = list(census.columns)
    census_columns.remove( "cfips")
    
    raw = raw.merge(census, on="cfips", how="left")
    feats += census_columns
    
    co_est = pd.read_csv("/kaggle/input/us-indicator/co-est2021-alldata.csv", encoding='latin-1')
    co_est["cfips"] = co_est.STATE*1000 + co_est.COUNTY
    co_columns = [
        'SUMLEV',
        'DIVISION',
        'ESTIMATESBASE2020',
        'POPESTIMATE2020',
        'POPESTIMATE2021',
        'NPOPCHG2020',
        'NPOPCHG2021',
        'BIRTHS2020',
        'BIRTHS2021',
        'DEATHS2020',
        'DEATHS2021',
        'NATURALCHG2020',
        'NATURALCHG2021',
        'INTERNATIONALMIG2020',
        'INTERNATIONALMIG2021',
        'DOMESTICMIG2020',
        'DOMESTICMIG2021',
        'NETMIG2020',
        'NETMIG2021',
        'RESIDUAL2020',
        'RESIDUAL2021',
        'GQESTIMATESBASE2020',
        'GQESTIMATES2020',
        'GQESTIMATES2021',
        'RBIRTH2021',
        'RDEATH2021',
        'RNATURALCHG2021',
        'RINTERNATIONALMIG2021',
        'RDOMESTICMIG2021',
        'RNETMIG2021'
    ]
    raw = raw.merge(co_est, on="cfips", how="left")
    feats +=  co_columns
    return raw, feats


# In[8]:


# Build Features based in lag of target
raw, feats = build_features(raw, 'target', 'active', lags = 9)
features = ['state_i']
features += feats
features += ['lng','lat','scale']
# print(features)
# raw.loc[raw.dcount==40, features].head(10)


# Latitude and Longitude feature engineering from samu2505.

# In[9]:


coordinates = raw[['lng', 'lat']].values

# Encoding tricks
emb_size = 20
precision = 1e6

latlon = np.expand_dims(coordinates, axis=-1)

m = np.exp(np.log(precision)/emb_size)
angle_freq = m ** np.arange(emb_size)
angle_freq = angle_freq.reshape(1,1, emb_size)
latlon = latlon * angle_freq
latlon[..., 0::2] = np.cos(latlon[..., 0::2])


# In[10]:


def rot(df):
    for angle in [15, 30, 45]:
        df[f'rot_{angle}_x'] = (np.cos(np.radians(angle)) * df['lat']) + \
                                (np.sin(np.radians(angle)) * df['lng'])
        
        df[f'rot_{angle}_y'] = (np.cos(np.radians(angle)) * df['lat']) - \
                                (np.sin(np.radians(angle)) * df['lng'])
        
    return df

raw = rot(raw)


# In[11]:


features += ['rot_15_x', 'rot_15_y', 'rot_30_x', 'rot_30_y', 'rot_45_x', 'rot_45_y']


# In[12]:


def get_model():
    from sklearn.ensemble import VotingRegressor
    import lightgbm as lgb
    import xgboost as xgb
    import catboost as cat
    from sklearn.pipeline import Pipeline
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.impute import KNNImputer    

# we should decrease the num_iterations of catboost
    cat_model = cat.CatBoostRegressor(
        iterations=2000,
        loss_function="MAPE",
        verbose=0,
        grow_policy='SymmetricTree',
        learning_rate=0.035,
        colsample_bylevel=0.8,
        max_depth=5,
        l2_leaf_reg=0.2,
        subsample=0.70,
        max_bin=4096,
    )

    return cat_model


def base_models():
    from sklearn.ensemble import VotingRegressor
    import lightgbm as lgb
    import xgboost as xgb
    import catboost as cat
    from sklearn.pipeline import Pipeline
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.impute import KNNImputer    
    
    # LGBM model
    params = {
    'n_iter': 300,
    'boosting_type': 'dart',
    'verbosity': -1,
    'objective': 'l1',
    'random_state': 42,
    'colsample_bytree': 0.8841279649367693,
    'colsample_bynode': 0.10142964450634374,
    'max_depth': 8,
    'learning_rate': 0.003647749926797374,
    'lambda_l2': 0.5,
    'num_leaves': 61,
    "seed": 42,
    'min_data_in_leaf': 213}

    lgb_model = lgb.LGBMRegressor(**params)
    
    xgb_model = xgb.XGBRegressor(
    objective='reg:pseudohubererror',
    tree_method="hist",
    n_estimators=795,
    learning_rate=0.0075,
    max_leaves = 17,
    subsample=0.50,
    colsample_bytree=0.50,
    max_bin=4096,
    n_jobs=2)

    # we should decrease the num_iterations of catboost
    cat_model = cat.CatBoostRegressor(
        iterations=2500,
        loss_function="MAPE",
        verbose=0,
        grow_policy='SymmetricTree',
        learning_rate=0.035,
        colsample_bylevel=0.8,
        max_depth=5,
        l2_leaf_reg=0.2,
        subsample=0.70,
        max_bin=4096,
    )
    
    models = {}
    models['xgb'] = xgb_model
    models['lgbm'] = lgb_model
    models['cat'] = cat_model

    return models


# In[13]:


ACT_THR = 150
MONTH_1 = 39
MONTH_last = 40


# In[14]:


# raw['ypred_last'] = np.nan
# raw['ypred'] = np.nan
# raw['k'] = 1.
# raw['microbusiness_density'].fillna(0, inplace = True)

# for TS in range(MONTH_1, MONTH_last): #40):
#     print(f'TS: {TS}')
   
#     model = get_model()
            
#     train_indices = (raw.istest==0) & (raw.dcount  < TS) & (raw.dcount >= 1) & (raw.lastactive>ACT_THR) 
#     valid_indices = (raw.istest==0) & (raw.dcount == TS) 
#     model.fit(
#         raw.loc[train_indices, features],
#         raw.loc[train_indices, 'target'].clip(-0.002, 0.006),

#     )

#     ypred = model.predict(raw.loc[valid_indices, features])
#     raw.loc[valid_indices, 'k'] = ypred + 1
#     raw.loc[valid_indices,'k'] = raw.loc[valid_indices,'k'] * raw.loc[valid_indices,'microbusiness_density']

#     # Validate
#     lastval = raw.loc[raw.dcount==TS, ['cfips', 'microbusiness_density']].set_index('cfips').to_dict()['microbusiness_density']
#     dt = raw.loc[raw.dcount==TS, ['cfips', 'k']].set_index('cfips').to_dict()['k']
    
#     df = raw.loc[raw.dcount==(TS+1), 
#                  ['cfips', 'microbusiness_density', 'state', 'lastactive', 'mbd_lag_1']].reset_index(drop=True)
#     df['pred'] = df['cfips'].map(dt)
#     df['lastval'] = df['cfips'].map(lastval)
    
# #     df.loc[df['lastval'].isnull(), 'lastval'] = df.loc[df['lastval'].isnull(), 'microbusiness_density']    
    
#     df.loc[df['lastactive']<=ACT_THR, 'pred'] = df.loc[df['lastactive']<=ACT_THR, 'lastval']
        
#     raw.loc[raw.dcount==(TS+1), 'ypred'] = df['pred'].values
#     raw.loc[raw.dcount==(TS+1), 'ypred_last'] = df['lastval'].values
    
#     print('Last Value SMAPE:', smape(df['microbusiness_density'], df['lastval']) )
#     print('SMAPE:', smape(df['microbusiness_density'], df['pred']))
#     print()


# ind = (raw.dcount > MONTH_1)&(raw.dcount <= MONTH_last)

# print( 'SMAPE:', smape( raw.loc[ind, 'microbusiness_density'],  raw.loc[ind, 'ypred'] ) )
# print( 'Last Value SMAPE:', smape( raw.loc[ind, 'microbusiness_density'],  raw.loc[ind, 'ypred_last'] ) )


# In[15]:


raw[raw.istest == 0].shape


# In[16]:


raw['ypred_last'] = np.nan
raw['ypred'] = np.nan
raw['k'] = 1.
raw['microbusiness_density'].fillna(0, inplace = True)


for TS in range(MONTH_1, MONTH_last): #40):
    print(f'TS: {TS}')
   
    # model = get_model()
    models = base_models()
    model0 = models['xgb']
    model1 = models['lgbm']
    model2 = models['cat']
            
    train_indices = (raw.istest==0) & (raw.dcount  < TS) & (raw.dcount >= 1) & (raw.lastactive>ACT_THR) 
    valid_indices = (raw.istest==0) & (raw.dcount == TS) 
    
    # Train each of the models on the current TS
    model0.fit(
        raw.loc[train_indices, features],
        raw.loc[train_indices, 'target'].clip(-0.002, 0.006))
    
    model1.fit(
        raw.loc[train_indices, features],
        raw.loc[train_indices, 'target'].clip(-0.002, 0.006))
    
    model2.fit(
        raw.loc[train_indices, features],
        raw.loc[train_indices, 'target'].clip(-0.002, 0.006))
    

    tr_pred0 = model0.predict(raw.loc[train_indices, features])
    tr_pred1 = model1.predict(raw.loc[train_indices, features])
    tr_pred2 = model2.predict(raw.loc[train_indices, features])
    train_preds = np.column_stack((tr_pred0, tr_pred1, tr_pred2))
    
    meta_model = get_model() 
    meta_model.fit(train_preds, raw.loc[train_indices, 'target'].clip(-0.002, 0.006))
    
    val_preds0 = model0.predict(raw.loc[valid_indices, features])
    val_preds1 = model1.predict(raw.loc[valid_indices, features])
    val_preds2 = model2.predict(raw.loc[valid_indices, features])
    valid_preds = np.column_stack((val_preds0, val_preds1, val_preds2))
    
    ypred = meta_model.predict(valid_preds)
    raw.loc[valid_indices, 'k'] = ypred + 1
    raw.loc[valid_indices,'k'] = raw.loc[valid_indices,'k'] * raw.loc[valid_indices,'microbusiness_density']

    # Validate
    lastval = raw.loc[raw.dcount==TS, ['cfips', 'microbusiness_density']].set_index('cfips').to_dict()['microbusiness_density']
    dt = raw.loc[raw.dcount==TS, ['cfips', 'k']].set_index('cfips').to_dict()['k']
    
    df = raw.loc[raw.dcount==(TS+1), 
                 ['cfips', 'microbusiness_density', 'state', 'lastactive', 'mbd_lag_1']].reset_index(drop=True)
    df['pred'] = df['cfips'].map(dt)
    df['lastval'] = df['cfips'].map(lastval)
    
#     df.loc[df['lastval'].isnull(), 'lastval'] = df.loc[df['lastval'].isnull(), 'microbusiness_density']    
    
    df.loc[df['lastactive']<=ACT_THR, 'pred'] = df.loc[df['lastactive']<=ACT_THR, 'lastval']
        
    raw.loc[raw.dcount==(TS+1), 'ypred'] = df['pred'].values
    raw.loc[raw.dcount==(TS+1), 'ypred_last'] = df['lastval'].values
    
    print('Last Value SMAPE:', smape(df['microbusiness_density'], df['lastval']) )
    print('SMAPE:', smape(df['microbusiness_density'], df['pred']))
    print()

ind = (raw.dcount > MONTH_1)&(raw.dcount <= MONTH_last)

print('SMAPE:', smape( raw.loc[ind, 'microbusiness_density'],  raw.loc[ind, 'ypred']))
print('Last Value SMAPE:', smape( raw.loc[ind, 'microbusiness_density'],  raw.loc[ind, 'ypred_last']))


# In[17]:


raw[raw['microbusiness_density'].isnull()]


# In[18]:


TS = 40
print(TS)

model0 = get_model()

train_indices = (raw.istest==0) & (raw.dcount  < TS) & (raw.dcount >= 1) & (raw.lastactive>ACT_THR) 
valid_indices = (raw.dcount == TS)

model0.fit(
        raw.loc[train_indices, features],
        raw.loc[train_indices, 'target'].clip(-0.002, 0.006))
    
model1.fit(
    raw.loc[train_indices, features],
    raw.loc[train_indices, 'target'].clip(-0.002, 0.006))

model2.fit(
    raw.loc[train_indices, features],
    raw.loc[train_indices, 'target'].clip(-0.002, 0.006))


tr_pred0 = model0.predict(raw.loc[train_indices, features])
tr_pred1 = model1.predict(raw.loc[train_indices, features])
tr_pred2 = model2.predict(raw.loc[train_indices, features])
train_preds = np.column_stack((tr_pred0, tr_pred1, tr_pred2))

meta_model = get_model() 
meta_model.fit(train_preds, raw.loc[train_indices, 'target'].clip(-0.002, 0.006))

val_preds0 = model0.predict(raw.loc[valid_indices, features])
val_preds1 = model1.predict(raw.loc[valid_indices, features])
val_preds2 = model2.predict(raw.loc[valid_indices, features])
valid_preds = np.column_stack((val_preds0, val_preds1, val_preds2))

ypred = meta_model.predict(valid_preds)

# ypred = model0.predict(raw.loc[valid_indices, features])
raw.loc[valid_indices, 'k'] = ypred + 1.
raw.loc[valid_indices,'k'] = raw.loc[valid_indices,'k'] * raw.loc[valid_indices,'microbusiness_density']

# Validate
lastval = raw.loc[raw.dcount==TS, ['cfips', 'microbusiness_density']].set_index('cfips').to_dict()['microbusiness_density']
dt = raw.loc[raw.dcount==TS, ['cfips', 'k']].set_index('cfips').to_dict()['k']


# In[19]:


df = raw.loc[raw.dcount==(TS+1), ['cfips', 'microbusiness_density', 'state', 'lastactive', 'mbd_lag_1']].reset_index(drop=True)


# In[20]:


df['pred'] = df['cfips'].map(dt)
df['lastval'] = df['cfips'].map(lastval)

df.loc[df['lastactive']<=ACT_THR, 'pred'] = df.loc[df['lastactive']<=ACT_THR, 'lastval']

raw.loc[raw.dcount==(TS+1), 'ypred'] = df['pred'].values
raw.loc[raw.dcount==(TS+1), 'ypred_last'] = df['lastval'].values


# In[21]:


raw.loc[raw['cfips']==28055, 'microbusiness_density'] = 0
raw.loc[raw['cfips']==48269, 'microbusiness_density'] = 1.762115


# In[22]:


COLS = ['GEO_ID','NAME','S0101_C01_026E']
df2020 = pd.read_csv('/kaggle/input/census-data-for-godaddy/ACSST5Y2020.S0101-Data.csv', usecols=COLS, dtype = 'object')
df2021 = pd.read_csv('/kaggle/input/census-data-for-godaddy/ACSST5Y2021.S0101-Data.csv',usecols=COLS, dtype = 'object')

df2020 = df2020.iloc[1:]
df2020 = df2020.astype({'S0101_C01_026E':'int'})

df2021 = df2021.iloc[1:]
df2021 = df2021.astype({'S0101_C01_026E':'int'})

df2020['cfips'] = df2020.GEO_ID.apply(lambda x: int(x.split('US')[-1]))
adult2020 = df2020.set_index('cfips').S0101_C01_026E.to_dict()

df2021['cfips'] = df2021.GEO_ID.apply(lambda x: int(x.split('US')[-1]))
adult2021 = df2021.set_index('cfips').S0101_C01_026E.to_dict()

df2020['mnoshitel'] = df2020['S0101_C01_026E'] / df2021['S0101_C01_026E']

df2020 = df2020[['cfips','mnoshitel']]
df2020.set_index('cfips', inplace=True)


# In[23]:


raw = raw.join(df2020, on='cfips')
maska = (raw["first_day_of_month"]=='2023-01-01')
raw.loc[maska, 'microbusiness_density'] = raw.loc[maska, 'ypred'] * raw.loc[maska, 'mnoshitel']
raw.drop(columns = 'mnoshitel' , inplace = True)


# In[24]:


test = raw[raw.first_day_of_month >= '2022-11-01'].copy()
test = test[['row_id', 'cfips', 'microbusiness_density']]
test = test[['row_id', 'microbusiness_density']]
test.to_csv('submission.csv', index=False)


# In[25]:


sub = pd.read_csv("/kaggle/input/godaddy-lb-13803/submission.csv")
for i, row in sub.iterrows():
    test.iat[i,1] = (
        0.6*test.iat[i,1] +
        0.4*row["microbusiness_density"]
    )
test.to_csv('submission.csv', index=False)
test.head(40)


# In[ ]:




