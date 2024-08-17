#!/usr/bin/env python
# coding: utf-8

# <b><span style="color:green; font-size:150%">Micro-Business Density forecast:
# with focus on feature engineering of census data and time-series based data [simple version]</span></b>
# 

# ![image.png](attachment:7f65804f-68f8-4044-a99c-d6ae85af12f6.png)

# # **<span style="color:#F7B2B0;">Problem Statment & workflow summary</span>**
# In this competition we are given 3,135 (the unique number of county) time-series of length 39 (time span). With those data, we should predict We must predict microbusiness density for the 8 months November 2022 thru June 2023. 
# 
# This is not a easy task. **Time span is too short** for making a prediction. As there are **only 39 data points for each county**, our ML model couldn't learn a lot from historical data. 
# 
# The workflow would be like this flowchart:
# ![cv_timeseries.png](attachment:f9b9ffb5-75e1-4a80-9d1d-0527899ce520.png)
# 
# 

# ### It means we train 3,135 * (the number of different models you will train) !!!

# 
# # **<span style="color:#F7B2B0;">feature engineering overview</span>**
# 
# Even though this is a tough task, we should find the best model. I think the key is 'feature engineering'. What kind of variables can we add to this model? 
# ![image.png](attachment:8605b1ca-e155-4a95-8225-8fb7c3d9ed84.png)
# 
# 
# As we have time-series data, we can make **time-series based variables**. 
# 1) lag data : The lag time is the time between the two time series you are correlating. It is reasonable to think that yesterday's stock price would affect today's stock price. :)
# 
# 2) moving average : The moving average (MA) is a simple technical analysis tool that smooths out price data by creating a constantly updated average price. The average is taken over a specific period of time, like 2days, 3days, a week etc...
# 
# 
# 3) weighted moving average / Exponetioanl moving Average: Exponential Moving Average (EMA) is similar to Simple Moving Average (SMA), measuring trend direction over a period of time. However, whereas SMA simply calculates an average of price data, EMA applies more weight to data that is more current.
# 
# 
# Also, the census data is given. As it noted,all fields have a two year lag to match what information was avaiable at the time a given microbusiness data update was published.
# 
# - pct_bb_[year] - The percentage of households in the county with access to broadband of any type. 
# - pct_college_[year] - The percent of the population in the county over age 25 with a 4-year college degree.
# - pct_foreign_born_[year] - The percent of the population in the county born outside of the United States. 
# - pct_it_workers_[year] - The percent of the workforce in the county employed in information related industries.
# - median_hh_inc_[year] - The median household income in the county.
# 
# As it is reasonble that these factors could affect the number of micro-business and the number of young people in each county, I will include those variables. 
# 
# Finally,we should consider **'national-wide' and 'state-wide' factors**, like average micro-business density in each state on each month or national average micro-business density each month. Therefore, I will also create these variables and try to use it for prediction.
# 

# ### 🗨️ if you want to check more on EDA precess, please visit below notetook:
# https://www.kaggle.com/code/kimtaehun/complete-baseline-code-with-various-ml-model

# In[1]:


# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', 500)

import plotly.figure_factory as ff
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import TimeSeriesSplit


# In[2]:


# create function for MA and EWM
#SMA and EMA are both commonly-used trend indicators. SMA gives equal weight to all data points,  while EMA applies more weight to recent data points. 

def moving_average(df,i, n):
    MA = pd.Series(df[i].rolling(n, min_periods=n).mean(), name = 'MA_' + str(n))
    df = df.join(MA)
    return df

def weighted_moving_average(df,i, n):
    EMA = pd.Series(df[i].ewm(span=n, adjust=False, min_periods=n).mean(), name = 'EMA_' + str(n))
    df = df.join(EMA)
    return df


# In[3]:


# load dataset

df_train = pd.read_csv('/kaggle/input/godaddy-microbusiness-density-forecasting/train.csv')
df_test = pd.read_csv('/kaggle/input/godaddy-microbusiness-density-forecasting/test.csv')
df_census = pd.read_csv('/kaggle/input/godaddy-microbusiness-density-forecasting/census_starter.csv')


# In[4]:


# combine train and testset
df_train['dataset'] = 'train'
df_test['dataset'] = 'test'
df = pd.concat((df_train, df_test)).sort_values('row_id').reset_index(drop=True)


# In[5]:


# merge census data with train-test dataset. 
df_all = df.merge(df_census, on = 'cfips', how='left')


# In[6]:


df_all['first_day_of_month'] = pd.to_datetime(df_all["first_day_of_month"])
df_all["year"] = df_all["first_day_of_month"].dt.year
df_all['month'] = df_all["first_day_of_month"].dt.month


# In[7]:


# add 2-year lag census data to each rows 

conditions = [df_all['year']==2019,df_all['year']==2020,df_all['year']==2021, df_all['year']==2022, df_all['year']==2023]
choices_bb = [df_all['pct_bb_2017'],df_all['pct_bb_2018'],df_all['pct_bb_2019'],df_all['pct_bb_2020'],df_all['pct_bb_2021'] ]
choices_college = [df_all['pct_college_2017'],df_all['pct_college_2018'],df_all['pct_college_2019'],
                   df_all['pct_college_2020'],df_all['pct_college_2021']]
choices_foreign = [df_all['pct_foreign_born_2017'],df_all['pct_foreign_born_2018'],df_all['pct_foreign_born_2019'],
                   df_all['pct_foreign_born_2020'],df_all['pct_foreign_born_2021']]
choices_workers = [df_all['pct_it_workers_2017'],df_all['pct_it_workers_2018'],df_all['pct_it_workers_2019'],
                   df_all['pct_it_workers_2020'],df_all['pct_it_workers_2021']]
choices_inc = [df_all['median_hh_inc_2017'],df_all['median_hh_inc_2018'],df_all['median_hh_inc_2019'],
               df_all['median_hh_inc_2020'],df_all['median_hh_inc_2021']]


# In[8]:


df_all["pct_bb"] = np.select(conditions, choices_bb, default=np.nan)
df_all["pct_college"] = np.select(conditions, choices_college, default=np.nan)
df_all["pct_foreign"] = np.select(conditions, choices_foreign, default=np.nan)
df_all["pct_workers"] = np.select(conditions, choices_workers, default=np.nan)
df_all["pct_inc"] = np.select(conditions, choices_inc, default=np.nan)


# In[9]:


# drop unnecessary columns
df_all.drop(['pct_bb_2017', 'pct_bb_2018',
       'pct_bb_2019', 'pct_bb_2020', 'pct_bb_2021', 'pct_college_2017',
       'pct_college_2018', 'pct_college_2019', 'pct_college_2020',
       'pct_college_2021', 'pct_foreign_born_2017', 'pct_foreign_born_2018',
       'pct_foreign_born_2019', 'pct_foreign_born_2020',
       'pct_foreign_born_2021', 'pct_it_workers_2017', 'pct_it_workers_2018',
       'pct_it_workers_2019', 'pct_it_workers_2020', 'pct_it_workers_2021',
       'median_hh_inc_2017', 'median_hh_inc_2018', 'median_hh_inc_2019',
       'median_hh_inc_2020', 'median_hh_inc_2021'], axis=1, inplace=True)


# In[10]:


# using pivot table to check the trend of micro-biz
train_pivoted = df_all.pivot(index='cfips',columns='first_day_of_month',values='microbusiness_density')


# In[11]:


# micro-business density 

plt.plot(train_pivoted.mean(axis=0));
plt.xticks(rotation=90);


# Microbusinesses per 100 people over the age of 18 in the given county. This is the target variable. The population figures used to calculate the density are on a two-year lag due to the pace of update provided by the U.S. Census Bureau, which provides the underlying population data annually.
# 
# It seems like micro-business was affected by Covid-19 (where it dropped dramatically). However, 4 out of 100 people over the age of 18 in the given county running their own business Now.

# In[12]:


# generate time-series based variables.
# create Moving Average variable.
df_all = moving_average(df_all, 'microbusiness_density', 3)
df_all = moving_average(df_all, 'microbusiness_density', 6)

# actually it's Exponential Moving Average. 
df_all = weighted_moving_average(df_all, 'microbusiness_density', 3)
df_all = weighted_moving_average(df_all, 'microbusiness_density', 6)


# #### please note that you can add any MA or EMA that you think reasonable, like 5days, 10days...

# In[13]:


# add 'pct_change_before' variable. This variable shows that how the density changed previous month.
df_all['pct_change_before'] = df_all['microbusiness_density'].pct_change().shift(1)


# In[14]:


# gnerate lag date


def lag_feature(df):
    for lag in range(1, 6):
        df[f'lag_density_{lag}'] = df.groupby('cfips')['microbusiness_density'].shift(lag)
        df[f'lag_density_{lag}'] = df.groupby('cfips')[f'lag_density_{lag}'].bfill()
        
    return df
    
df_all = lag_feature(df_all)


# #### As I mentioned, we should consider **'national-wide' and 'state-wide' factors**, like average micro-business density in each state on each month or national average micro-business density each month. 

# In[15]:


# the average microbusiness density in each state by month
# each month's national average microbusines_density 
df_all['national_avg'] = df_all.groupby(['year','month'])['microbusiness_density'].transform('mean')
df_all['state_avg'] = df_all.groupby(['state','year','month'])['microbusiness_density'].transform('mean')


# ### create target variable
# Now, we need to create target column. 
# As we are trying to predict the next month's micro-biz density,
# we should shift the 'microbusiness_density' variables like below.

# In[16]:


df_all['target'] = df_all.groupby('cfips')['microbusiness_density'].shift(-1)


# In[17]:


# check columns
df_all.columns


# **Not all columns are need for prediction**. 
# Definetely, we should exclude row_id, state etc. 
# 
# So the next step is exclude unnecessary columns.

# In[18]:


#SMAPE formula : Symmetric mean absolute percentage error
# https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
# SMAPE = (1/n) * Σ(|forecast – actual| / ((|actual| + |forecast|)/2) * 100
'''
advantages: 
Expressed as a percentage.
Fixes the shortcoming of the original MAPE — it has both the lower (0%) and the upper (200%) bounds.

'''
def SMAPE(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)


# In[19]:


# replace inf values to NaN, otherwise you will encounter error on the next step.
df_all = df_all.replace([np.inf, -np.inf], np.nan)


# In[20]:


# feature_list

feature_list = ['year', 'month', 'pct_bb','microbusiness_density',
       'pct_college', 'pct_foreign', 'pct_workers', 'pct_inc', 'MA_3', 'MA_6',
       'EMA_3', 'EMA_6', 'lag_density_1', 'lag_density_2',
       'lag_density_3', 'lag_density_4', 'lag_density_5', 'pct_change_before',
       'national_avg', 'state_avg']


# In[21]:


# separate train dataset
sample_train = df_all[df_all['dataset']=='train']


# In[22]:


train_X = sample_train[feature_list]
train_y = sample_train['target']


# #### please note that we are going to use timeseires split, so that we could aviod to cheating feature data (data leakage).
# ![image.png](attachment:2333eeb2-0293-4d05-a698-b183219fb2c0.png)

# In[23]:


def smape_cv(model):
    tscv = TimeSeriesSplit(n_splits=7)
    smape_list = []
    model_name = model.__class__.__name__
    for _, (train_index, test_index) in tqdm(enumerate(tscv.split(train_X), start=1), 
                                             desc=f'{model_name} Cross Validations', total=7):
        X_train, X_test = train_X.iloc[train_index], train_X.iloc[test_index]
        y_train, y_test = train_y.iloc[train_index], train_y.iloc[test_index]
        clf = model.fit(X_train, y_train)
        pred = clf.predict(X_test)
        smape = SMAPE(y_test, pred) 
        smape_list.append(smape)
    return model_name, smape_list

def print_smape_score(model):
    model_name, score = smape_cv(model)
    for i, r in enumerate(score, start=1):
        print(f'{i} FOLDS: {model_name} smape: {r:.4f}')
    print(f'\n{model_name} mean smape: {np.mean(score):.4f}')
    print('='*30)
    return model_name, np.mean(score)


# In[24]:


# Train 8 models for each county !!! Please note that this is just a sample code. You Should modify or tweak below parameters to get better performance.
reg = LinearRegression(n_jobs=-1)
ridge = Ridge(alpha=0.8, random_state=1)
lasso = Lasso(alpha = 0.01, random_state=1)
Enet = ElasticNet(alpha=0.03, l1_ratio=0.01, random_state=1)
DTree = DecisionTreeRegressor(max_depth=5, min_samples_split=2, min_samples_leaf=2, random_state=1)
rf = RandomForestRegressor(n_estimators=500, criterion='mse', max_depth=5, min_samples_split=2,
                           min_samples_leaf=2, random_state=1, n_jobs=-1)
model_xgb = xgb.XGBRegressor(n_estimators=500, max_depth=5, min_child_weight=5, gamma=0.1, n_jobs=-1)
model_lgb = lgb.LGBMRegressor(n_estimators=500, max_depth=5, min_child_weight=5, n_jobs=-1)


# In[25]:


model_list = [reg, ridge, lasso, Enet, DTree, rf, model_xgb, model_lgb]


# <span style = 'color: #008000'> **Due to time contraints, I just test it on the sample list (randomly select 4 counties)** </span>

# In[26]:


test_list = [9011, 9007, 40055, 29107]

#to get all the county list, change test_list to df_all.cfips.unique()


# In[27]:


submit_df = pd.DataFrame()
model_results = {}
county_and_model = {}

for i in test_list:
    print(f'======================= county {i} modeling =====================')
    sample = df_all[df_all['cfips']==i]
    sample_train = sample[sample['dataset']=='train']
    train_X = sample_train[['year', 'month', 'pct_bb','microbusiness_density',
       'pct_college', 'pct_foreign', 'pct_workers', 'pct_inc', 'MA_3', 'MA_6',
       'EMA_3', 'EMA_6', 'lag_density_1', 'lag_density_2',
       'lag_density_3', 'lag_density_4', 'lag_density_5', 'pct_change_before',
       'national_avg', 'state_avg']]
    train_y = sample_train['target']
    train_X.fillna(method='bfill', inplace=True)
    train_X.fillna(method='ffill', inplace=True)
    
    model_dict = {}
    for model in [reg, ridge, lasso, Enet, DTree, rf, model_xgb, model_lgb]:
        model_name, mean_score = print_smape_score(model)
        model_dict[model_name] = mean_score
        model_results[i] = min(model_dict.values())
        
    final_model = model_list[list(model_dict.keys()).index(min(model_dict, key=model_dict.get))]
    
    sample.fillna(method='ffill', inplace=True)
    
    # select testset
    sample_test = sample[sample['dataset']=='test']
    submit_X = sample_test[['year', 'month', 'pct_bb','microbusiness_density',
       'pct_college', 'pct_foreign', 'pct_workers', 'pct_inc', 'MA_3', 'MA_6',
       'EMA_3', 'EMA_6', 'lag_density_1', 'lag_density_2',
       'lag_density_3', 'lag_density_4', 'lag_density_5', 'pct_change_before',
       'national_avg', 'state_avg']]
    #predict with final_model
    county_and_model[i] = min(model_dict, key=model_dict.get)
    #print(county_and_model)
    
    predict_result = final_model.predict(submit_X).tolist()
    
    df = pd.DataFrame(list(zip(sample_test['row_id'].values.tolist(), predict_result)),
              columns=['row_id','microbusiness_density'])
    submit_df = submit_df.append(df)


# In[28]:


#model results is a dictionary that the key is county number and value is each county's the lowest SMAPE score.
model_results


# In[29]:


# county_and_model is a dictionary that the key is county number and value is each county's selected ML model.
county_and_model 


# In[30]:


# this is submit file :) !!!
submit_df.head()


# #### Please note that this code still have a room for improvement. Especially when it comes to hyper-parameter tuning. Due to time constraints, I try to make it as simple as possible.
# 
