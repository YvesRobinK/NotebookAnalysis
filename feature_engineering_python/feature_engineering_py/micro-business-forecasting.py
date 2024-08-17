#!/usr/bin/env python
# coding: utf-8

# ### **Thanks to great works from Vitaly Kudelya, Prem, Jasonczh, Chris, Jasonczh, Giba, Nin7a1, judith007,samu2505, tetsutani.**

# Aim :
# To predict / forecast monthly microbusiness density in a given area.
# It is a Time Series Analysis problem.
# 
# Dataset Attributes :
# This competition has provided 3 csv files :
# 
# train.csv :
# 
# row_id : An ID code for the row.
# cfips : A unique identifier for each county using the Federal Information Processing System. The first two digits correspond to the state FIPS code, while the following 3 represent the county.
# county_name : The written name of the county.
# state_name : The name of the state.
# first_day_of_month : The date of the first day of the month.
# microbusiness_density : Microbusinesses per 100 people over the age of 18 in the given county. This is the target variable. The population figures used to calculate the density are on a two-year lag due to the pace of update provided by the U.S. Census Bureau, which provides the underlying population data annually. 2021 density figures are calculated using 2019 population figures, etc.
# active : The raw count of microbusinesses in the county. Not provided for the test set.
# 
# census_starter.csv : Examples of useful columns from the Census Bureau's American Community Survey (ACS) at data.census.gov. The percentage fields were derived from the raw counts provided by the ACS. All fields have a two year lag to match what information was avaiable at the time a given microbusiness data update was published.
# 
# pctbb[year] : The percentage of households in the county with access to broadband of any type. Derived from ACS table B28002: PRESENCE AND TYPES OF INTERNET SUBSCRIPTIONS IN HOUSEHOLD.
# cfips : The CFIPS code.
# pctcollege[year] : The percent of the population in the county over age 25 with a 4-year college degree. Derived from ACS table S1501: EDUCATIONAL ATTAINMENT.
# pct_foreignborn[year] : The percent of the population in the county born outside of the United States. Derived from ACS table DP02: SELECTED SOCIAL CHARACTERISTICS IN THE UNITED STATES.
# pct_itworkers[year] : The percent of the workforce in the county employed in information related industries. Derived from ACS table S2405: INDUSTRY BY OCCUPATION FOR THE CIVILIAN EMPLOYED POPULATION 16 YEARS AND OVER.
# median_hhinc[year] : The median household income in the county. Derived from ACS table S1901: INCOME IN THE PAST 12 MONTHS (IN 2021 INFLATION-ADJUSTED DOLLARS).
# test.csv :
# 
# row_id : An ID code for the row.
# cfips : A unique identifier for each county using the Federal Information Processing System. The first two digits correspond to the state FIPS code, while the following 3 represent the county.
# first_day_of_month : The date of the first day of the month.
# sample_submission.csv : A valid sample submission. This file will remain unchanged throughout the competition.
# 
# row_id : An ID code for the row.
# microbusiness_density : The target variable.
# 

# We use WGET to download additional data

# In[1]:


get_ipython().system('wget http://bit.ly/3ZLyF82 -O CSS.css -q')
    
from IPython.core.display import HTML
with open('./CSS.css', 'r') as file:
    custom_css = file.read()

HTML(custom_css)


# In[2]:


import gc
import numpy as np
import pandas as pd
import os
from tqdm.notebook import tqdm
BASE = '../input/godaddy-microbusiness-density-forecasting/'


# In[3]:


# error calculation function smape
def smape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    
    pos_ind = (y_true!=0)|(y_pred!=0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    
    return 100 * np.mean(smap)

# error vector calculation function
def vsmape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    
    pos_ind = (y_true!=0)|(y_pred!=0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    
    return 100 * smap


# In[4]:


# loading data
census = pd.read_csv(BASE + 'census_starter.csv')
train = pd.read_csv(BASE + 'train.csv')
test = pd.read_csv(BASE + 'test.csv')
sub = pd.read_csv(BASE + 'sample_submission.csv')
# loading additional data
reaveal_test = pd.read_csv(BASE + 'revealed_test.csv') 
# attaching additional data
train = pd.concat([train, reaveal_test]).sort_values(by=['cfips','first_day_of_month']).reset_index()
drop_index = (test.first_day_of_month == '2022-11-01') | (test.first_day_of_month == '2022-12-01')
test = test.loc[~drop_index,:]
# loading geodata
coords = pd.read_csv("/kaggle/input/usa-counties-coordinates/cfips_location.csv")
# data aggregation
train['istest'] = 0
test['istest'] = 1
raw = pd.concat((train, test)).sort_values(['cfips','row_id']).reset_index(drop=True)
raw = raw.merge(coords.drop("name", axis=1), on="cfips")
# creating new columns
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
# use GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# There are some anomalies. We remove emissions by more than 10%

# In[5]:


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


# In[6]:


lag = 1
raw[f'mbd_lag_{lag}'] = raw.groupby('cfips')['microbusiness_density'].shift(lag).bfill()
raw['dif'] = (raw['microbusiness_density'] / raw[f'mbd_lag_{lag}']).fillna(1).clip(0, None) - 1
raw.loc[(raw[f'mbd_lag_{lag}']==0), 'dif'] = 0
raw.loc[(raw[f'microbusiness_density']>0) & (raw[f'mbd_lag_{lag}']==0), 'dif'] = 1
raw['dif'] = raw['dif'].abs()
raw.groupby('dcount')['dif'].sum().plot()


# SMAPE is a relative metric so target must be converted.

# In[7]:


raw['target'] = raw.groupby('cfips')['microbusiness_density'].shift(-1)
raw['target'] = raw['target']/raw['microbusiness_density'] - 1


# In[8]:


raw['lastactive'] = raw.groupby('cfips')['active'].transform('last')


# # Feature Engineering
# > Try tuning this part

# In[9]:


def build_features(raw, target='microbusiness_density', target_act='active_tmp', lags = 8):
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
    # loading additional data
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


# In[10]:


# Build Features based in lag of target
raw, feats = build_features(raw, 'target', 'active', lags = 8)
features = ['state_i'] # state number
features += feats
features += ['lng','lat','scale'] # geodata coordinates and day number
print(features)
raw.loc[raw.dcount==40, features].head(10)


# Latitude and Longitude feature engineering from samu2505.

# In[11]:


# geodata processing
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


# In[12]:


def rot(df):
    for angle in [15, 30, 45]:
        df[f'rot_{angle}_x'] = (np.cos(np.radians(angle)) * df['lat']) + \
                                (np.sin(np.radians(angle)) * df['lng'])
        
        df[f'rot_{angle}_y'] = (np.cos(np.radians(angle)) * df['lat']) - \
                                (np.sin(np.radians(angle)) * df['lng'])
        
    return df

raw = rot(raw)


# In[13]:


features += ['rot_15_x', 'rot_15_y', 'rot_30_x', 'rot_30_y', 'rot_45_x', 'rot_45_y']


# In[14]:


def get_model():
    import catboost as cat

# we should decrease the num_iterations of catboost
    cat_model = cat.CatBoostRegressor(
        iterations=800,
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
    


# In[15]:


ACT_THR = 140 
MONTH_1 = 30
MONTH_last = 40


# In[16]:


raw['ypred_last'] = np.nan
raw['ypred'] = np.nan
raw['k'] = 1.
raw['microbusiness_density'].fillna(0, inplace = True)

for TS in range(MONTH_1, MONTH_last): # monthly cycle
    print(TS)
   
    model = get_model()
            
    train_indices = (raw.istest==0) & (raw.dcount  < TS) & (raw.dcount >= 1) & (raw.lastactive>ACT_THR) 
    valid_indices = (raw.istest==0) & (raw.dcount == TS) 
    model.fit(
        raw.loc[train_indices, features],
        raw.loc[train_indices, 'target'].clip(-0.0043, 0.0045),

    )

    ypred = model.predict(raw.loc[valid_indices, features])
    # get value 'microbusiness_density' from target
    raw.loc[valid_indices, 'k'] = ypred + 1
    raw.loc[valid_indices,'k'] = raw.loc[valid_indices,'k'] * raw.loc[valid_indices,'microbusiness_density']

    # Validate
    lastval = raw.loc[raw.dcount==TS, ['cfips', 'microbusiness_density']].set_index('cfips').to_dict()['microbusiness_density']
    dt = raw.loc[raw.dcount==TS, ['cfips', 'k']].set_index('cfips').to_dict()['k']
    
    df = raw.loc[raw.dcount==(TS+1), 
                 ['cfips', 'microbusiness_density', 'state', 'lastactive', 'mbd_lag_1']].reset_index(drop=True)
    df['pred'] = df['cfips'].map(dt)
    df['lastval'] = df['cfips'].map(lastval)    
   
    df.loc[df['lastactive']<=ACT_THR, 'pred'] = df.loc[df['lastactive']<=ACT_THR, 'lastval']
        
    raw.loc[raw.dcount==(TS+1), 'ypred'] = df['pred'].values
    raw.loc[raw.dcount==(TS+1), 'ypred_last'] = df['lastval'].values
    
    print(f'TS: {TS}')
    print('Last Value SMAPE:', smape(df['microbusiness_density'], df['lastval']) )
    print('SMAPE:', smape(df['microbusiness_density'], df['pred']))
    print()


ind = (raw.dcount > MONTH_1)&(raw.dcount <= MONTH_last)

print( 'SMAPE:', smape( raw.loc[ind, 'microbusiness_density'],  raw.loc[ind, 'ypred'] ) )
print( 'Last Value SMAPE:', smape( raw.loc[ind, 'microbusiness_density'],  raw.loc[ind, 'ypred_last'] ) )


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
    raw.loc[train_indices, 'target'].clip(-0.0044, 0.0046),
)
ypred = model0.predict(raw.loc[valid_indices, features])
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


mask = (raw.lastactive > 150) & (raw.dcount==(TS+1))
k = 0.5
raw.loc[mask, 'ypred'] = k * 1.005 * raw.loc[mask, 'ypred_last'] + (1-k) * raw.loc[mask, 'ypred'] 


# In[22]:


dt = raw.loc[raw.dcount==41, ['cfips', 'ypred']].set_index('cfips').to_dict()['ypred']
test = raw.loc[raw.istest==1, ['row_id', 'cfips','microbusiness_density']].copy()
test['microbusiness_density'] = test['cfips'].map(dt)
test = test[['row_id','microbusiness_density']]


# In[23]:


sample_sub = pd.read_csv(BASE + 'revealed_test.csv')
sub_index = (sample_sub.first_day_of_month == '2022-11-01') | (sample_sub.first_day_of_month  == '2022-12-01')
test1 = pd.concat([sample_sub.loc[sub_index, :].drop([i for i  in sample_sub.columns if i !='row_id' and i != 'microbusiness_density'],axis=1).fillna(2), test])


# In[24]:


test1 = test1.fillna(2)  # fill anything you want 


# In[25]:


COLS = ['GEO_ID','NAME','S0101_C01_026E']
df2020 = pd.read_csv('/kaggle/input/census-data-for-godaddy/ACSST5Y2020.S0101-Data.csv',usecols=COLS, dtype = 'object')
df2020 = df2020.iloc[1:]
df2020['S0101_C01_026E'] = df2020['S0101_C01_026E'].astype('int')


# In[26]:


df2021 = pd.read_csv('/kaggle/input/census-data-for-godaddy/ACSST5Y2021.S0101-Data.csv',usecols=COLS, dtype = 'object')
df2021 = df2021.iloc[1:]
df2021['S0101_C01_026E'] = df2021['S0101_C01_026E'].astype('int')


# In[27]:


test1['cfips'] = test1.row_id.apply(lambda x: int(x.split('_')[0]))


# In[28]:


df2020['cfips'] = df2020.GEO_ID.apply(lambda x: int(x.split('US')[-1]) )
adult2020 = df2020.set_index('cfips').S0101_C01_026E.to_dict()

df2021['cfips'] = df2021.GEO_ID.apply(lambda x: int(x.split('US')[-1]) )
adult2021 = df2021.set_index('cfips').S0101_C01_026E.to_dict()

test1['adult2020'] = test1.cfips.map(adult2020)
test1['adult2021'] = test1.cfips.map(adult2021)


# In[29]:


test1.microbusiness_density = test1.microbusiness_density * test1.adult2020 / test1.adult2021
test1 = test1.drop(['adult2020','adult2021','cfips'],axis=1)
test1.to_csv('submission.csv',index=False)

