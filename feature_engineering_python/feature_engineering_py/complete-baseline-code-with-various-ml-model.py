#!/usr/bin/env python
# coding: utf-8

# <b><span style="color:green; font-size:150%">Micro-Business Density forecast:
# with focus on feature engineering of census data and time-series based data</span></b>
# 

# ![image.png](attachment:42a2fd50-df7d-44f4-82a6-ca120a723a03.png)

# <b><span style="color:green; font-size:120%">If you want to check simple version with full modeling code, please refer to 
# https://www.kaggle.com/code/kimtaehun/simple-version-predict-3-135-county-one-by-one</span></b> 

# # **<span style="color:#F7B2B0;">Problem Statment & workflow summary</span>**
# In this competition we are given 3,135 (the unique number of county) time-series of length 39 (time span). With those data, we should predict We must predict microbusiness density for the 8 months November 2022 thru June 2023. 
# 
# This is not a easy task. **Time span is too short** for making a prediction. As there are **only 39 data points for each county**, our ML model couldn't learn a lot from historical data. 
# 
# The workflow would be like this diagram:
# ![Blank diagram (2).png](attachment:415b7928-3e18-48df-9070-eb3da614a6d7.png)
# 
# Even though this is a tough task, we should find the best model. I think the key is 'feature engineering'. What kind of variables can we add to this model? 
# 
# 
# 
# # **<span style="color:#F7B2B0;">feature engineering overview</span>**
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
# Here's choropleth map of The percentage of households in the county with access to broadband of any type (pct_bb_[year]) in 2019. You can find the related code below!
# ![image.png](attachment:22eb7088-10e3-4560-b1f9-ad1bd7d04d88.png)
# 
# As it is reasonble that these factors could affect the number of micro-business and the number of young people in each county, I will include those variables. 
# 
# 
# Finally,we should consider **'national-wide' and 'state-wide' factors**, like average micro-business density in each state on each month or national average micro-business density each month. Therefore, I will also create these variables and try to use it for prediction.
# 
# ![image.png](attachment:a6e42157-e4b6-4095-90df-7dc27dc337f3.png)
# 
# It seems like micro-business was affected by Covid-19 (where it dropped dramatically). However, 4 out of 100 people over the age of 18 in the given county running their own business Now.

# # **<span style="color:#F7B2B0;">Modeling overviews</span>**
# 
# There are multiple candidate models for this time series forecasting.
# As time span is relatively short, complicated model would be avoided.
# I will test simple linear regression, regression with regularizations like Ridge, Lasso or both (Enet).
# At the same time, I will try tree-based model like Random Forest, XGB and LGBM.
# 
# the below results is one of the example of density forecasting. the average SMAPE is quite low in this case (New London County)
# ![image.png](attachment:6d3269b3-b9d2-4499-8c76-4c95540cba25.png)
# 
# 
# So I believe you could build strong model based on this code. Hope it will be helpful for you guys, and please **UPVOTE** this post for the next version!
# 
# In terms of modeling, I'd like to point one thing here. YOU should be careful when it comes to train/test split for CV.
# TimeSeries split method should be used for avoiding data leakage (cheating).
# 
# I will show you how to use this technic easily in the following code.
# 
# Time Series Split Example:
# ![image.png](attachment:ffa89fea-3415-4689-8ab7-7845c75223d8.png)

# ## What's Next?
# Each county's time series might has different trend. 
# Theoretically, the best way is test each time series and choose the best model.
# (It might take a lot of time, as it should train tons of models)
# 
# ![cv_timeseries.png](attachment:2a964afe-a97e-4ed3-b734-659c3fb86493.png)

# In[1]:


# Plz install these packages if you want to make choropleth map.
get_ipython().system('pip install plotly-geo')
get_ipython().system('pip install pyshp')
get_ipython().system('pip install geopandas')


# In[ ]:


# if Pip does not work, please try conda install as below
# conda install -c plotly plotly-geo


# In[2]:


# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', 500)

import plotly.figure_factory as ff
import geopandas
import shapely
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


df_all.head()


# In[7]:


df_all['first_day_of_month'] = pd.to_datetime(df_all["first_day_of_month"])
df_all["year"] = df_all["first_day_of_month"].dt.year
df_all['month'] = df_all["first_day_of_month"].dt.month


# In[8]:


# you can change the value under here to see the national trend in each elements.
fig = ff.create_choropleth(fips=df_census.cfips,title = 'pct_bb_2019', values=df_census.pct_bb_2019.values)
fig.layout.template = None
fig.show();


# In[9]:


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


# In[10]:


df_all["pct_bb"] = np.select(conditions, choices_bb, default=np.nan)
df_all["pct_college"] = np.select(conditions, choices_college, default=np.nan)
df_all["pct_foreign"] = np.select(conditions, choices_foreign, default=np.nan)
df_all["pct_workers"] = np.select(conditions, choices_workers, default=np.nan)
df_all["pct_inc"] = np.select(conditions, choices_inc, default=np.nan)


# In[11]:


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


# In[12]:


# let's see how '(2year-lag)pct_college' is related with micro-biz density
import matplotlib.pyplot as plt

test = df_all.set_index('first_day_of_month')
test = test[test['cfips']==56001]
fig = plt.figure(figsize = (12, 4))
ax1 = fig.add_subplot(1,1,1)
ax1.plot(test['microbusiness_density'], color='blue' , label='microbusiness_density')
ax2 = ax1.twinx()
ax2.plot(test['pct_college'], color='red' , label='college')


# In[13]:


# randomly choose cfips == 1001 and 56001 (above)
test = df_all.set_index('first_day_of_month')
test = test[test['cfips']==1001]
fig = plt.figure(figsize = (12, 4))
ax1 = fig.add_subplot(1,1,1)
ax1.plot(test['microbusiness_density'], color='blue' , label='microbusiness_density')
ax2 = ax1.twinx()
ax2.plot(test['pct_college'], color='red' , label='college')


# It's not easy to figure out how these variables are related with each other at a glance. Anyway, Let's move on to the next step.

# In[14]:


# using pivot table to check the trend of micro-biz
train_pivoted = df_all.pivot(index='cfips',columns='first_day_of_month',values='microbusiness_density')


# In[15]:


# micro-business density 

plt.plot(train_pivoted.mean(axis=0));
plt.xticks(rotation=90);


# Microbusinesses per 100 people over the age of 18 in the given county. This is the target variable. The population figures used to calculate the density are on a two-year lag due to the pace of update provided by the U.S. Census Bureau, which provides the underlying population data annually. 
#  
# It seems like micro-business was affected by Covid-19 (where it dropped dramatically). However, 4 out of 100 people over the age of 18 in the given county running their own business Now.

# In[16]:


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


# In[17]:


# generate time-series based variables.
# create Moving Average variable.
df_all = moving_average(df_all, 'microbusiness_density', 3)
df_all = moving_average(df_all, 'microbusiness_density', 6)

# actually it's Exponential Moving Average. 
df_all = weighted_moving_average(df_all, 'microbusiness_density', 3)
df_all = weighted_moving_average(df_all, 'microbusiness_density', 6)


# In[18]:


# add 'pct_change_before' variable. This variable shows that how the density changed previous month.
df_all['pct_change_before'] = df_all['microbusiness_density'].pct_change().shift(1)


# In[19]:


df_all.head(10)


# #### generate lagged data , lag1 means previous (one-month) month's mciro-biz density.

# In[20]:


def lag_feature(df):
    for lag in range(1, 6):
        df[f'lag_density_{lag}'] = df.groupby('cfips')['microbusiness_density'].shift(lag)
        df[f'lag_density_{lag}'] = df.groupby('cfips')[f'lag_density_{lag}'].bfill()
        
    return df
    
df_all = lag_feature(df_all)


# As I mentioned, we should consider **'national-wide' and 'state-wide' factors**, like average micro-business density in each state on each month or national average micro-business density each month. 

# In[21]:


# the average microbusiness density in each state by month
# each month's national average microbusines_density 
df_all['national_avg'] = df_all.groupby(['year','month'])['microbusiness_density'].transform('mean')
df_all['state_avg'] = df_all.groupby(['state','year','month'])['microbusiness_density'].transform('mean')


# In[22]:


# example 
df_all[df_all['cfips']==47081].head(10)


# ### create target variable
# Now, we need to create target column. 
# As we are trying to predict the next month's micro-biz density,
# we should shift the 'microbusiness_density' variables like below.

# In[23]:


df_all['target'] = df_all.groupby('cfips')['microbusiness_density'].shift(-1)


# #### create function to check the table at a glance

# In[24]:


def summary(df):
    print(f'data shape: {df.shape}')
    summ = pd.DataFrame(df.dtypes, columns=['data type'])
    summ['#missing'] = df.isnull().sum().values * 100
    summ['%missing'] = df.isnull().sum().values / len(df)
    summ['#unique'] = df.nunique().values
    desc = pd.DataFrame(df.describe(include='all').transpose())
    summ['min'] = desc['min'].values
    summ['max'] = desc['max'].values
    summ['first value'] = df.loc[0].values
    summ['second value'] = df.loc[1].values
    summ['third value'] = df.loc[2].values
    
    return summ


# In[25]:


summary_table = summary(df_all)


# In[26]:


# now you can check all variables at a glance.
summary_table


# ### Modeling

# In[27]:


# check columns
df_all.columns


# **Not all columns are need for prediction**. 
# Definetely, we should exclude row_id, state, microbusiness_density etc. 
# 
# So the next step is exclude unnecessary columns.

# #### let's choose one county as a sample and build a prediction model.

# In[28]:


# choose one county 
sample = df_all[df_all['cfips']==9011]


# In[29]:


# choose train data only
sample_train = sample[sample['dataset']=='train']


# In[30]:


### split train X and Y
train_X = sample_train[['year', 'month', 'pct_bb',
       'pct_college', 'pct_foreign', 'pct_workers', 'pct_inc', 'MA_3', 'MA_6',
       'EMA_3', 'EMA_6', 'lag_density_1', 'lag_density_2',
       'lag_density_3', 'lag_density_4', 'lag_density_5', 'pct_change_before',
       'national_avg', 'state_avg']]
train_y = sample_train['target']


# In[31]:


# fill na with 'bfill'
train_X.fillna(method='bfill', inplace=True)


# ### Time Series Split.
# #### before we start, let's see the overview of time-series split.

# In[32]:


# Function modified from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
from matplotlib.patches import Patch
cmap_cv = plt.cm.seismic
def plot_cv_indices(cv, n_splits, X, y, date_col = None):
    """Create a sample plot for indices of a cross-validation object."""
    
    fig, ax = plt.subplots(1, 1, figsize = (11, 7))
    
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=10, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)


    # Formatting
#     yticklabels = list(range(n_splits))
    
    if date_col is not None:
        tick_locations  = ax.get_xticks()
        tick_dates = [" "] + date_col.iloc[list(tick_locations[1:-1])].astype(str).tolist() + [" "]

        tick_locations_str = [str(int(i)) for i in tick_locations]
        new_labels = ['\n\n'.join(x) for x in zip(list(tick_locations_str), tick_dates) ]
        ax.set_xticks(tick_locations)
        ax.set_xticklabels(new_labels)
    
    ax.set(yticks=np.arange(n_splits+2) + .5,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+0.2, -.2])
    ax.legend([Patch(color=cmap_cv(.8)), Patch(color=cmap_cv(.02))],
              ['Testing set', 'Training set'], loc=(1.02, .8))
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)


# In[33]:


# time series split 
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit, TimeSeriesSplit
cvs = [KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit, TimeSeriesSplit]
n_points = 100
n_splits = 8
X = np.random.randn(100, 10)
percentiles_classes = [.1, .3, .6]
y = np.hstack([[ii] * int(100 * perc) for ii, perc in enumerate(percentiles_classes)])

for i, cv in enumerate(cvs):
    this_cv = cv(n_splits=n_splits)
    plot_cv_indices(this_cv, n_splits, X, y, date_col=None)


# ### If you are not using TimeSeriesSplit, It means that you are cheating future data during the training.
# ### To avoid data leakage like this, data scientist prefer using TimeSeriesSlit method, even though the size of traning data would become smaller than normal method.

# In[34]:


from sklearn.model_selection import TimeSeriesSplit
n_splits = 7
tscv = TimeSeriesSplit(n_splits)


# In[35]:


# In our case, data point is 39, so it would be splited like below
# (example taken from https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py )

for fold, (train_index, test_index) in enumerate(tscv.split(sample[sample['dataset']=='train'])):
    print("Fold: {}".format(fold))
    print("TRAIN indices:", train_index, "\n", "TEST indices:", test_index)
    print("\n")
    X_train, X_test = train_X.iloc[train_index], train_X.iloc[test_index]
    y_train, y_test = train_y.iloc[train_index], train_y.iloc[test_index]


plot_cv_indices(tscv,n_splits, X, y)


# ### create evaluation metrics
# Symmetric mean absolute percentage error (SMAPE or sMAPE) is an accuracy measure based on percentage (or relative) errors. 
# ![image.png](attachment:a597a9c9-be5e-4629-9f1a-b52e42998aba.png)![image.png](attachment:ce2c4879-8a0d-49e8-80c4-ea8bb420d978.png)
# 
# Advantages of SMAPE :
# - Expressed as a percentage, therefore easy to interpret
# - Fixes the shortcoming of the original MAPE. it has both the lower (0%) and the upper (200%) bounds.

# In[36]:


# create custom function
def SMAPE(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)


# In[37]:


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
    # print cv and save the average
    model_name, score = smape_cv(model)
    for i, r in enumerate(score, start=1):
        print(f'{i} FOLDS: {model_name} smape: {r:.4f}')
    print(f'\n{model_name} mean smape: {np.mean(score):.4f}')
    print('='*30)
    return model_name, np.mean(score)


# In[38]:


reg = LinearRegression(n_jobs=-1)
ridge = Ridge(alpha=0.8, random_state=1)
lasso = Lasso(alpha = 0.01, random_state=1)
Elastic = ElasticNet(alpha=0.03, l1_ratio=0.01, random_state=1)
RF = RandomForestRegressor(n_estimators=500, criterion='mse', max_depth=5, min_samples_split=2,
                           min_samples_leaf=2, random_state=1, n_jobs=-1)
XGB = xgb.XGBRegressor(n_estimators=500, max_depth=5, min_child_weight=5, gamma=0.1, n_jobs=-1)
LGBM = lgb.LGBMRegressor(n_estimators=500, max_depth=5, min_child_weight=5, n_jobs=-1)


# In[39]:


models = []
scores = []
for model in [reg, ridge, lasso, Elastic, RF, XGB, LGBM]:
    model_name, mean_score = print_smape_score(model)
    models.append(model_name)
    scores.append(mean_score)


# #### For New London County in Connecticut, test score looks not bad. 

# ## What's Next?
# Each county's time series might has different trend. 
# Theoretically, the best way is test each time series and choose the best model.
# (It might take a lot of time, as it should train tons of models)
# 
# ![image.png](attachment:e9fdd1cb-f418-4a12-bc3b-100513e9f60f.png)
# 
# 

# In[ ]:





# In[ ]:




