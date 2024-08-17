#!/usr/bin/env python
# coding: utf-8

# <div style="color:#FF1F26;
#            display:fill;
#            border-style: solid;
#            border-color:#C1C1C1;
#            font-size:16px;
#            font-family:Calibri;
#            background-color:#B75351;">
# <h2 style="text-align: center;
#            padding: 10px;
#            color:#FFFFFF;">
# ======= AutoML FLAML for Time-series Forecasting =======
# </h2>
# </div>
# 
# 

# <div style=" background-color:#4b371c;text-align:left; padding: 13px 13px; border-radius: 8px; color: white; font-size: 16px">
# Time-series forecasting plays a crucial role in various industries, enabling organizations to make accurate predictions based on historical data trends. However, developing effective forecasting models can be a complex and time-consuming task. That's where FLAML AutoML comes in.
# <br><br>
# FLAML AutoML is an automated machine learning library specifically designed to streamline the process of time-series forecasting. It empowers data scientists and analysts to rapidly develop robust and accurate forecasting models, even without extensive expertise in machine learning algorithms and techniques.
# <br><br>
# In this guide, we will take you through the steps of utilizing FLAML AutoML to unlock the potential of your time-series data and make data-driven predictions. Whether you are new to time-series forecasting or an experienced practitioner, FLAML AutoML offers a user-friendly interface and powerful capabilities to enhance your forecasting projects.
# </div>

# <div style="padding:10px;
#             color:white;
#             margin:5;
#             font-size:170%;
#             text-align:left;
#             display:fill;
#             border-radius:15px;
#             background-color:#294B8E;
#             overflow:hidden;
#             font-weight:700"> Table of Contents</div>

# <a id="toc"></a>
# - [1. Notebook Setup](#1)
# - [2. Time Dependency Variables](#2)
# - [3. Anomaly Analysis](#3)
# - [4. Lag Feature](#4)
# - [5. Feature Engineering](#5)
# - [6. FLAML Time-series Forecasting Model](#6)
# 

# <a id="1"></a>
# # <b><span style='color:#FF0000'>1. Notebook Setup</span></b>

# In[1]:


import gc
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import random

# Install  the FLAML package
get_ipython().system('pip install -q flaml')
from flaml import AutoML

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

# Hide convergence warning for now
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# In[2]:


# Parameters
p_train_time = 8000


# In[3]:


BASE = '../input/godaddy-microbusiness-density-forecasting/'

def smape(y_true, y_pred):
    # Initializes an array of zeros with the length of y_true.
    smap = np.zeros(len(y_true))
    
    # Calculates the average of the absolute values of y_true and y_pred.
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    
    # Calculates SMAPE only for indices where pos_ind is True.
    pos_ind = (y_true!=0)|(y_pred!=0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    
    return 100 * np.mean(smap)

def vsmape(y_true, y_pred):
    # Initializes an array of zeros with the length of y_true.
    smap = np.zeros(len(y_true))
    
    # Calculates the average of the absolute values of y_true and y_pred.
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    
    # Calculates SMAPE only for indices where pos_ind is True.
    pos_ind = (y_true!=0)|(y_pred!=0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    
    return 100 * smap

# Lists the files and directories in the specified path.
get_ipython().system('ls ../input/godaddy-microbusiness-density-forecasting')


# In[4]:


# Import external data to improve the performance of the forecasting
census = pd.read_csv(BASE + 'census_starter.csv')
print(census.columns)
census.head()


# In[5]:


# Reads the data
train = pd.read_csv(BASE + 'train.csv')
reaveal_test = pd.read_csv(BASE + 'revealed_test.csv')

# Concatenates 'train' and 'revealed_test' data, sorts them by 'cfips' and 'first_day_of_month', and resets the index.
train = pd.concat([train, reaveal_test]).sort_values(by=['cfips','first_day_of_month']).reset_index()
test = pd.read_csv(BASE + 'test.csv')
# Creates a boolean array to identify rows with '2022-11-01' or '2022-12-01'.
drop_index = (test.first_day_of_month == '2022-11-01') | (test.first_day_of_month == '2022-12-01')
# Filters out rows
test = test.loc[~drop_index,:]
sub = pd.read_csv(BASE + 'sample_submission.csv')
print(train.shape, test.shape, sub.shape)

train['istest'] = 0
test['istest'] = 1

# Concatenates 'train' and 'test' DataFrames, sorts them by 'cfips' and 'row_id', and resets the index.
raw = pd.concat((train, test)).sort_values(['cfips','row_id']).reset_index(drop=True)


# <a id="2"></a>
# # <b><span style='color:#FF0000'>2. Time Dependency Variables</span></b>

# <div style=" background-color:#4b371c;text-align:left; padding: 13px 13px; border-radius: 8px; color: white; font-size: 16px">
# Time dependency refers to the inherent relationship and dependence of the target variable  on time. Time-series data consists of observations recorded at different time intervals, such as hourly, daily, monthly, or yearly. In such data, the order and sequence of observations are crucial as they often exhibit patterns, trends, and seasonality that change over time.
# <br><br>
# Time dependency implies that the current or future values of the target variable are influenced by past values of the same variable. The past observations in a time series serve as valuable information for predicting future values.    
# </div>

# In[6]:


# Generate list of date dependence features for time-series analysis

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
#raw['population'] = np.round(np.mean(raw['active']*100/raw['microbusiness_density']))

raw.tail(5)


# <a id="3"></a>
# # <b><span style='color:#FF0000'>3. Anomaly Analysis</span></b>

# <div style=" background-color:#4b371c;text-align:left; padding: 13px 13px; border-radius: 8px; color: white; font-size: 16px">
# Anomaly analysis in time-series forecasting refers to the identification and detection of unusual or abnormal patterns, events, or observations in a time series data. Anomalies, also known as outliers, are data points that deviate significantly from the expected or normal behavior of the time series. 
# </div>

# In[7]:


# to calculate and visualize the relative changes in microbusiness density over time
lag = 1
raw[f'mbd_lag_{lag}'] = raw.groupby('cfips')['microbusiness_density'].shift(lag).bfill()
raw['dif'] = (raw['microbusiness_density'] / raw[f'mbd_lag_{lag}']).fillna(1).clip(0, None) - 1
raw.loc[(raw[f'mbd_lag_{lag}']==0), 'dif'] = 0
raw.loc[(raw[f'microbusiness_density']>0) & (raw[f'mbd_lag_{lag}']==0), 'dif'] = 1
raw['dif'] = raw['dif'].abs()
raw.groupby('dcount')['dif'].sum().plot()


# In[8]:


# to identify and adjusts outliers in the 'microbusiness_density'.  
# the variance between data points and the mean is used as the measure to detect outliers
outliers = []
cnt = 0
for o in tqdm(raw.cfips.unique()):
    indices = (raw['cfips']==o)
    tmp = raw.loc[indices].copy().reset_index(drop=True)
    var = tmp.microbusiness_density.values.copy()
    #vmax = np.max(var[:38]) - np.min(var[:38])
    
    for i in range(40, 0, -1):
        thr = 0.20*np.mean(var[:i])
        difa = abs(var[i]-var[i-1])
        if (difa>=thr):
            var[:i] *= (var[i]/var[i-1])
            outliers.append(o)
            cnt+=1
    var[0] = var[1]*0.99
    raw.loc[indices, 'microbusiness_density'] = var
    
outliers = np.unique(outliers)
# len(outliers), cnt


# <a id="4"></a>
# # <b><span style='color:#FF0000'>4. Lag Feature</span></b>

# <div style=" background-color:#4b371c;text-align:left; padding: 13px 13px; border-radius: 8px; color: white; font-size: 16px">
# Lag features refer to the values of the target variable or other relevant variables at previous time steps. A lag feature captures the relationship between the current observation and past observations in the time series.
# <br><br>
# By incorporating lag features into a time-series analysis, we can leverage the temporal patterns and dependencies present in the data. The idea is that the past values of a variable can provide valuable information for predicting its future values. Lag features help capture trends, seasonality, and other temporal patterns that exist in the time series.
# </div>

# In[9]:


# to analyze microbusiness density over time by calculating relative changes and visualizing trends 
# by creating lagged values for microbusiness density within each group
# And then to compute the relative change compared to the lagged value.
lag = 1
raw[f'mbd_lag_{lag}'] = raw.groupby('cfips')['microbusiness_density'].shift(lag).bfill()
raw['dif'] = (raw['microbusiness_density'] / raw[f'mbd_lag_{lag}']).fillna(1).clip(0, None) - 1
raw.loc[(raw[f'mbd_lag_{lag}']==0), 'dif'] = 0
raw.loc[(raw[f'microbusiness_density']>0) & (raw[f'mbd_lag_{lag}']==0), 'dif'] = 1
raw['dif'] = raw['dif'].abs()
raw.groupby('dcount')['dif'].sum().plot()


# In[10]:


raw.loc[raw.cfips == 1015].plot(x='dcount', y='microbusiness_density')
raw.loc[raw.cfips == 21215].plot(x='dcount', y='microbusiness_density')


# In[11]:


raw['target'] = raw.groupby('cfips')['microbusiness_density'].shift(-1)
raw['target'] = raw['target']/raw['microbusiness_density'] - 1


raw.loc[raw['cfips']==28055, 'target'] = 0.0
raw.loc[raw['cfips']==48269, 'target'] = 0.0

raw.iloc[:5,:]


# In[12]:


raw['target'].clip(-0.05, 0.05).hist(bins=100)


# In[13]:


raw['lastactive'] = raw.groupby('cfips')['active'].transform('last')

dt = raw.loc[raw.dcount==28].groupby('cfips')['microbusiness_density'].agg('last')
raw['lasttarget'] = raw['cfips'].map(dt)

raw['lastactive'].clip(0, 8000).hist(bins=30)


# <a id="5"></a>
# # <b><span style='color:#FF0000'>5. Feature Engineering</span></b>

# <div style=" background-color:#4b371c;text-align:left; padding: 13px 13px; border-radius: 8px; color: white; font-size: 16px">
# Feature engineering in time-series analysis refers to the process of creating and selecting relevant features from the raw time-series data to improve the performance of predictive models. It involves transforming the raw data into meaningful representations that capture the underlying patterns and relationships in the time series.
# </div>

# In[14]:


# to generate lagged and rolling features based on specified target variables and lags, providing a broader perspective on the dataset.
# Additionally, to compute rolling mean features by applying a rolling window function to the lagged features, summing the values within the specified window.

def build_features(raw, target='microbusiness_density', target_act='active_tmp', lags = 6):
    feats = []
    for lag in tqdm(range(1, lags)):
        raw[f'mbd_lag_{lag}'] = raw.groupby('cfips')[target].shift(lag)
        raw[f'act_lag_{lag}'] = raw.groupby('cfips')[target_act].diff(lag)
        feats.append(f'mbd_lag_{lag}')
        feats.append(f'act_lag_{lag}')
        
    lag = 1
    for window in [2, 4, 6, 8, 10]:
        raw[f'mbd_rollmea{window}_{lag}'] = raw.groupby('cfips')[f'mbd_lag_{lag}'].transform(lambda s: s.rolling(window, min_periods=1).sum())        
        #raw[f'mbd_rollmea{window}_{lag}'] = raw[f'mbd_lag_{lag}'] - raw[f'mbd_rollmea{window}_{lag}']
        feats.append(f'mbd_rollmea{window}_{lag}')
      

    census_columns = list(census.columns)
    census_columns.remove( "cfips")
    
    raw = raw.merge(census, on="cfips", how="left")
    feats += census_columns
        
    return raw, feats


# In[15]:


# Build Features based in lag of target
raw, feats = build_features(raw, 'target', 'active', lags = 5)
features = ['state_i']
features += feats
# features += ['lng','lat']
print(features)
# raw.loc[raw.dcount==40, features].head(10)


# In[16]:


#state=raw.groupby(['state'])['target'].mean()
#raw['state_m'] = raw['state'].map(state)

#features += ['state_m']
features += ['scale']


# In[17]:


raw['lasttarget'].clip(0,10).hist(bins=100)


# In[18]:


# Manual adjustment

blacklist = [
    'North Dakota', 'Iowa', 'Kansas', 'Nebraska', 'South Dakota','New Mexico', 'Alaska', 'Vermont'
]
blacklistcfips = [
1019,1027,1029,1035,1039,1045,1049,1057,1067,1071,1077,1085,1091,1099,1101,1123,1131,1133,4001,4012,4013,4021,4023,5001,5003,5005,5017,5019,5027,5031,5035,5047,5063,5065,5071,5081,5083,5087,5091,5093,5107,5109,5115,5121,5137,5139,5141,5147,6003,6015,6027,6033,6053,6055,6057,6071,6093,6097,6103,6105,6115,8003,8007,8009,8019,8021,8023,8047,8051,8053,8055,8057,8059,8061,8065,8067,8069,8071,8073,8075,8085,8091,8093,8097,8099,8103,8105,8107,8109,8111,8115,8117,8121,9007,9009,9015,12009,12017,12019,12029,12047,12055,12065,12075,12093,12107,12127,13005,13007,13015,13017,13019,13027,13035,13047,13065,13081,13083,13099,13107,13109,13117,13119,13121,13123,13125,13127,13135,13143,13147,13161,13165,13171,13175,13181,13193,13201,13221,13225,13229,13231,13233,13245,13247,13249,13257,13279,13281,13287,13289,13293,13301,13319,15001,15005,15007,16001,16003,16005,16007,16013,16015,16017,16023,16025,16029,16031,16033,16035,16037,16043,16045,16049,16061,16063,16067,17001,17003,17007,17009,17013,17015,17023,17025,17031,17035,17045,17051,17059,17061,17063,17065,17067,17069,17075,17077,17081,17085,17087,17103,17105,17107,17109,17115,17117,17123,17127,17133,17137,17141,17143,17147,17153,17167,17169,17171,17177,17179,17181,17185,17187,17193,18001,18007,18009,18013,18015,18019,18021,18025,18035,18037,18039,18041,18053,18061,18075,18079,18083,18087,18099,18103,18111,18113,18115,18137,18139,18145,18153,18171,18179,21001,21003,21013,21017,21023,21029,21035,21037,21039,21045,21047,21055,21059,21065,21075,21077,21085,21091,21093,21097,21099,21101,21103,21115,21125,21137,21139,21141,21149,21155,21157,21161,21165,21179,21183,21191,21197,21199,21215,21217,21223,21227,21237,21239,22019,22021,22031,22039,22041,22047,22069,22085,22089,22101,22103,22109,22111,22115,22119,22121,23003,23009,23021,23027,23029,24011,24027,24029,24031,24035,24037,24039,24041,25011,25015,26003,26007,26011,26019,26021,26025,26027,26033,26037,26041,26043,26051,26053,26057,26059,26061,26065,26071,26077,26079,26083,26089,26097,26101,26103,26109,26111,26115,26117,26119,26127,26129,26131,26135,26141,26143,26155,26161,26165,27005,27011,27013,27015,27017,27021,27023,27025,27029,27047,27051,27055,27057,27065,27069,27073,27075,27077,27079,27087,27091,27095,27101,27103,27105,27107,27109,27113,27117,27119,27123,27125,27129,27131,27133,27135,27141,27147,27149,27155,27159,27167,27169,28017,28019,28023,28025,28035,28045,28049,28061,28063,28093,28097,28099,28125,28137,28139,28147,28159,29001,29015,29019,29031,29033,29041,29049,29051,29055,29057,29063,29065,29069,29075,29085,29089,29101,29103,29111,29121,29123,29125,29135,29137,29139,29143,29157,29159,29161,29167,29171,29173,29175,29177,29183,29195,29197,29199,29203,29205,29207,29209,29213,29215,29217,29223,29227,29229,30005,30009,30025,30027,30033,30035,30037,30039,30045,30049,30051,30053,30055,30057,30059,30069,30071,30073,30077,30079,30083,30085,30089,30091,30093,30101,30103,30105,30107,30109,32005,32009,32017,32023,32027,32029,32510,33005,33007,34021,34027,34033,34035,36011,36017,36023,36033,36043,36047,36049,36051,36057,36061,36067,36083,36091,36097,36103,36107,36113,36115,36121,36123,37005,37009,37011,37017,37023,37029,37031,37049,37061,37075,37095,37117,37123,37131,37137,37151,37187,37189,37197,39005,39009,39015,39017,39019,39023,39037,39039,39043,39049,39053,39057,39063,39067,39071,39077,39085,39087,39091,39097,39105,39107,39113,39117,39119,39125,39127,39129,39135,39137,39151,39153,39157,40003,40013,40015,40023,40025,40027,40035,40039,40043,40045,40053,40055,40057,40059,40065,40067,40073,40077,40079,40099,40105,40107,40111,40115,40123,40127,40129,40133,40141,40147,40151,40153,41001,41007,41013,41015,41017,41021,41025,41031,41033,41037,41051,41055,41063,41067,41069,42005,42007,42011,42013,42015,42019,42027,42029,42031,42035,42053,42057,42067,42071,42083,42085,42093,42097,42105,42111,42113,42115,42123,42125,42127,42129,44005,44007,44009,45001,45009,45021,45025,45031,45059,45067,45071,45073,45089,47001,47005,47013,47015,47019,47021,47023,47027,47035,47039,47041,47047,47055,47057,47059,47061,47069,47073,47075,47077,47083,47087,47099,47105,47121,47127,47131,47133,47135,47137,47147,47151,47153,47159,47161,47163,47169,47177,47183,47185,48001,48011,48017,48019,48045,48057,48059,48063,48065,48073,48077,48079,48081,48083,48087,48095,48101,48103,48107,48109,48115,48117,48119,48123,48125,48129,48149,48151,48153,48155,48159,48161,48165,48175,48189,48191,48195,48197,48211,48221,48229,48233,48235,48237,48239,48241,48243,48245,48255,48261,48263,48265,48267,48269,48275,48277,48283,48293,48299,48305,48311,48313,48319,48321,48323,48327,48333,48345,48347,48355,48369,48377,48379,48383,48387,48389,48401,48403,48413,48417,48431,48433,48437,48443,48447,48453,48455,48457,48461,48463,48465,48469,48471,48481,48483,48485,48487,48495,48499,49001,49009,49013,49019,49027,49031,49045,51005,51017,51025,51029,51031,51036,51037,51043,51057,51059,51065,51071,51073,51077,51079,51083,51091,51095,51097,51101,51111,51115,51119,51121,51127,51135,51147,51155,51159,51165,51167,51171,51173,51181,51183,51191,51197,51530,51590,51610,51620,51670,51678,51720,51735,51750,51770,51810,51820,53013,53019,53023,53031,53033,53037,53039,53041,53047,53065,53069,53071,53075,54013,54019,54025,54031,54033,54041,54049,54055,54057,54063,54067,54071,54077,54079,54085,54089,54103,55001,55003,55005,55007,55011,55017,55021,55025,55029,55037,55043,55047,55049,55051,55061,55065,55067,55075,55077,55091,55097,55101,55103,55109,55117,55123,55125,55127,56007,56009,56011,56015,56017,56019,56021,56027,56031,56037,56043,56045,
12061,  6095, 49025, 18073, 29029, 29097, 48419, 51830, 30067, 26095, 18159, 32001, 54065, 54027, 13043, 48177, 55069, 48137, 30087, 29007, 13055, 48295, 28157, 29037, 45061, 22053, 13199, 47171, 53001, 55041, 51195, 18127, 29151, 48307, 51009, 16047, 29133,  5145, 17175, 21027, 48357, 29179, 13023, 16077, 48371, 21057, 16039, 21143, 48435, 48317, 48475,  5129, 36041, 48075, 29017, 47175, 39167, 47109, 17189, 17173, 28009, 39027, 48133, 18129, 48217, 40081, 36021,  6005, 42099, 18051, 36055, 53051, 6109, 21073, 27019,  6051, 48055,  8083, 48503, 17021, 10003, 41061, 22001, 22011, 21205, 48223, 51103, 51047, 16069, 17033, 41011,  6035, 47145, 27083, 18165, 36055, 12001, 26159,  8125, 34017,
28141, 55119, 48405, 40029, 18125, 21135, 29073, 55115, 37149,55039, 26029, 12099, 13251, 48421, 39007, 41043, 22015, 37115,54099, 51137, 22049, 55131, 17159, 56001, 40005, 18017, 28091,47101, 27037, 29005, 13239, 21019, 55085, 48253, 51139, 40101,13283, 18049, 39163, 45049, 51113,
]


# In[19]:


# to perform training and prediction using an AutoML model to generate forecasts for microbusiness density

ACT_THR = 140
ABS_THR = 0
raw['ypred_last'] = np.nan
raw['ypred'] = np.nan
raw['k'] = 1.
VAL = []

for TS in range(39, 40):
#     print(TS)
#     if TS != 37:
#          continue
    

    # Initialize an AutoML instance
    model = AutoML()    
            
    train_indices = (raw.istest==0) & (raw.dcount  < TS) & (raw.dcount >= 1) & (raw.lastactive>ACT_THR)  & (raw.lasttarget>ABS_THR) 
    valid_indices = (raw.istest==0) & (raw.dcount == TS)
    

    # Specify automl goal and constraint
    FL_automl_settings = {
        "time_budget": p_train_time,  # in seconds
        "metric": 'r2',
        "task": 'regression',
        "verbose": 0,
        "n_jobs": -1,
        "eval_method": 'cv',
        "n_splits":5
    }

    # Train with labeled input data
    model.fit(raw.loc[train_indices, features], raw.loc[train_indices, 'target'].clip(-0.0043, 0.0045), **FL_automl_settings)

    ypred = model.predict(raw.loc[valid_indices, features])
    

    raw.loc[valid_indices, 'k'] = ypred + 1
    raw.loc[valid_indices,'k'] = raw.loc[valid_indices,'k'] * raw.loc[valid_indices,'microbusiness_density']

    # Validate
    lastval = raw.loc[raw.dcount==TS, ['cfips', 'microbusiness_density']].set_index('cfips').to_dict()['microbusiness_density']
    dt = raw.loc[raw.dcount==TS, ['cfips', 'k']].set_index('cfips').to_dict()['k']
    
    df = raw.loc[raw.dcount==(TS+1), ['cfips', 'microbusiness_density', 'state', 'lastactive', 'mbd_lag_1']].reset_index(drop=True)
    df['pred'] = df['cfips'].map(dt)
    df['lastval'] = df['cfips'].map(lastval)
    
    
    df.loc[df['lastactive']<=ACT_THR, 'pred'] = (
        df.loc[df['lastactive']<=ACT_THR, 'pred'] +
        df.loc[df['lastactive']<=ACT_THR, 'lastval']
    )/2
    df.loc[df['lastval']<=ABS_THR, 'pred'] = (
        df.loc[df['lastval']<=ABS_THR, 'pred'] +
        df.loc[df['lastval']<=ABS_THR, 'lastval']
    )/2
    
    #df.loc[df['state'].isin(blacklist), 'pred'] = (df.loc[df['state'].isin(blacklist), 'lastval']+
                                                   #df.loc[df['state'].isin(blacklist), 'pred'])/2
    
    
    #df.loc[df['cfips'].isin(blacklistcfips), 'pred'] = (df.loc[df['cfips'].isin(blacklistcfips), 'lastval']+
                                                   #df.loc[df['cfips'].isin(blacklistcfips), 'pred'])/2
        
        
    #df.loc[df['state'].isin(blacklist), 'pred'] = df.loc[df['state'].isin(blacklist), 'lastval']
                                                   
    
    
    #df.loc[df['cfips'].isin(blacklistcfips), 'pred'] = df.loc[df['cfips'].isin(blacklistcfips), 'lastval']
                                                   
        
        
    raw.loc[raw.dcount==(TS+1), 'ypred'] = df['pred'].values
    raw.loc[raw.dcount==(TS+1), 'ypred_last'] = df['lastval'].values
    
#     print(f'TS: {TS}')
#     print('Last Value SMAPE:', smape(df['microbusiness_density'], df['lastval']) )
#     print('SMAPE:', smape(df['microbusiness_density'], df['pred']))
#     print()


# ind = (raw.dcount>=30)&(raw.dcount<=40)
# print( 'SMAPE:', smape( raw.loc[ind, 'microbusiness_density'],  raw.loc[ind, 'ypred'] ) )
# print( 'Last Value SMAPE:', smape( raw.loc[ind, 'microbusiness_density'],  raw.loc[ind, 'ypred_last'] ) )


# In[20]:


raw['error'] = vsmape(raw['microbusiness_density'], raw['ypred'])
raw['error_last'] = vsmape(raw['microbusiness_density'], raw['ypred_last'])
raw.loc[(raw.dcount==30), ['microbusiness_density', 'ypred', 'error', 'error_last'] ]


# In[21]:


dt = raw.loc[(raw.dcount>=30)&(raw.dcount<=38) ].groupby(['cfips','dcount'])['error', 'error_last'].last()
dt['miss'] = dt['error'] > dt['error_last']
dt = dt.groupby('cfips')['miss'].mean()
dt = dt.loc[dt>=0.50]
dt.shape


# In[22]:


raw.loc[raw.dcount==(41), ['cfips', 'microbusiness_density', 'state', 'lastactive', 'mbd_lag_1']].reset_index(drop=True)


# <a id="6"></a>
# # <b><span style='color:#FF0000'>6. FLAML Time-series Forecasting Model</span></b>

# <div style=" background-color:#4b371c;text-align:left; padding: 13px 13px; border-radius: 8px; color: white; font-size: 16px">
# FLAML (Fast and Lightweight AutoML) is an open-source Python library developed by Microsoft Research that provides automated machine learning capabilities. While FLAML is primarily designed for general machine learning tasks, including classification and regression, it can also be utilized for time-series forecasting.
# </div>

# In[23]:


TS = 40
print(TS)

# model0 = get_model()

# Initialize an AutoML instance
model0 = AutoML()

train_indices = (raw.istest==0) & (raw.dcount  < TS) & (raw.dcount >= 1) & (raw.lastactive>ACT_THR)  & (raw.lasttarget>ABS_THR) 
valid_indices = (raw.dcount == TS)

# Hide convergence warning for now
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Specify automl goal and constraint
FL_automl_settings = {
    "time_budget": p_train_time,  # in seconds
    "metric": 'r2',
    "task": 'regression',
    "verbose": 1,
    "n_jobs": -1,
    "eval_method": 'cv',
    "n_splits":5
}

# Train with labeled input data
model0.fit(raw.loc[train_indices, features], raw.loc[train_indices, 'target'].clip(-0.0044, 0.0046), **FL_automl_settings)


ypred = model0.predict(raw.loc[valid_indices, features])
raw.loc[valid_indices, 'k'] = ypred + 1.
raw.loc[valid_indices,'k'] = raw.loc[valid_indices,'k'] * raw.loc[valid_indices,'microbusiness_density']

# Validate
lastval = raw.loc[raw.dcount==TS, ['cfips', 'microbusiness_density']].set_index('cfips').to_dict()['microbusiness_density']
dt = raw.loc[raw.dcount==TS, ['cfips', 'k']].set_index('cfips').to_dict()['k']


# In[24]:


df = raw.loc[raw.dcount==(TS+1), ['cfips', 'microbusiness_density', 'state', 'lastactive', 'mbd_lag_1']].reset_index(drop=True)
df


# In[25]:


# to perform post-processing for prediction results
df['pred'] = df['cfips'].map(dt)
df['lastval'] = df['cfips'].map(lastval)

#df.loc[df['lastactive']<=ACT_THR, 'pred'] = df.loc[df['lastactive']<=ACT_THR, 'lastval']
#df.loc[df['lastval']<=ABS_THR, 'pred'] = df.loc[df['lastval']<=ABS_THR, 'lastval']
df.loc[df['lastactive']<=ACT_THR, 'pred'] = df.loc[df['lastactive']<=ACT_THR, 'lastval']
df.loc[df['lastval']<=ABS_THR, 'pred'] = df.loc[df['lastval']<=ABS_THR, 'lastval']

#df.loc[df['state'].isin(blacklist), 'pred'] = df.loc[df['state'].isin(blacklist), 'lastval']
#df.loc[df['cfips'].isin(blacklistcfips), 'pred'] = df.loc[df['cfips'].isin(blacklistcfips), 'lastval']
raw.loc[raw.dcount==(TS+1), 'ypred'] = df['pred'].values
raw.loc[raw.dcount==(TS+1), 'ypred_last'] = df['lastval'].values


# In[26]:


raw.loc[raw['cfips']==28055, 'microbusiness_density'] = 0
raw.loc[raw['cfips']==48269, 'microbusiness_density'] = 1.762115

dt = raw.loc[raw.dcount==41, ['cfips', 'ypred']].set_index('cfips').to_dict()['ypred']
test = raw.loc[raw.istest==1, ['row_id', 'cfips','microbusiness_density']].copy()
test


# In[27]:


test['microbusiness_density'] = test['cfips'].map(dt)

test = test[['row_id','microbusiness_density']]
test


# In[28]:


sample_sub = pd.read_csv(BASE + 'revealed_test.csv')
sub_index = (sample_sub.first_day_of_month == '2022-11-01') | (sample_sub.first_day_of_month  == '2022-12-01')
test1 = pd.concat([sample_sub.loc[sub_index, :].drop([i for i  in sample_sub.columns if i !='row_id' and i != 'microbusiness_density'],axis=1).fillna(2), test])


# In[29]:


test1 = test1.fillna(1.7569) 


# In[30]:


COLS = ['GEO_ID','NAME','S0101_C01_026E']
df2020 = pd.read_csv('/kaggle/input/census-data-for-godaddy/ACSST5Y2020.S0101-Data.csv',usecols=COLS)
df2020 = df2020.iloc[1:]
df2020['S0101_C01_026E'] = df2020['S0101_C01_026E'].astype('int')
print( df2020.shape )
df2020.head()


# In[31]:


df2021 = pd.read_csv('/kaggle/input/census-data-for-godaddy/ACSST5Y2021.S0101-Data.csv',usecols=COLS)
df2021 = df2021.iloc[1:]
df2021['S0101_C01_026E'] = df2021['S0101_C01_026E'].astype('int')
print( df2021.shape )
df2021.head()


# In[32]:


test1['cfips'] = test1.row_id.apply(lambda x: int(x.split('_')[0]))
#sub = sub.drop('cfips',axis=1)
#sub.to_csv('submission.csv',index=False)
test1.head()


# In[33]:


df2020['cfips'] = df2020.GEO_ID.apply(lambda x: int(x.split('US')[-1]) )
adult2020 = df2020.set_index('cfips').S0101_C01_026E.to_dict()

df2021['cfips'] = df2021.GEO_ID.apply(lambda x: int(x.split('US')[-1]) )
adult2021 = df2021.set_index('cfips').S0101_C01_026E.to_dict()

test1['adult2020'] = test1.cfips.map(adult2020)
test1['adult2021'] = test1.cfips.map(adult2021)
test1.head()


# ### Submission

# In[34]:


test1.microbusiness_density = test1.microbusiness_density * test1.adult2020  / test1.adult2021
test1 = test1.drop(['adult2020','adult2021','cfips'],axis=1)
test1.to_csv('submission.csv',index=False)
test1.head()


# # Reference

# #### [1] [Better XGB Baseline](https://www.kaggle.com/code/titericz/better-xgb-baseline) by @titericz
# #### [2] [kNN impute + Voting XGB+LGBM+CAT](https://www.kaggle.com/code/batprem/knn-impute-voting-xgb-lgbm-cat) by @batprem
# #### [3] [New Census+ kNN+ XGB+LGBM+CAT[LB=1.4121]](https://www.kaggle.com/code/jasonczh/new-census-knn-xgb-lgbm-cat-lb-1-4121/notebook) by @Jasonczh
# 
