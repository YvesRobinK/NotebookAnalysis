#!/usr/bin/env python
# coding: utf-8

# # ASHRAE - Great Energy Predictor III
# ### *How much energy will a building consume?*
# 
# ----
# 
# <a href="https://www.kaggle.com/c/ashrae-energy-prediction/overview"><img src="https://i.ibb.co/rp01Ngb/Screenshot-from-2019-10-16-17-39-18.png" alt="Screenshot-from-2019-10-16-17-39-18" border="0"></a>
# 
# <br>
# 
# ### starter Content:
# 
# > <span style="color:red">IMPORTANT</span> : I will keep updating this starter kernel these days :)
# 
# - EDA
# - Feature Engineering
# - Basic LGBM Model
# 
# ### References:
# 
# - My baseline was **[Simple LGBM Solution](https://www.kaggle.com/ryches/simple-lgbm-solution)**, an amazing kernel by @ryches
# - My post [Must read material: similar comps, models, github ...](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/112958#latest-650382)
# 
# <br>

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import lightgbm as lgb
import math
from tqdm import tqdm_notebook as tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

PATH = '../input/ashrae-energy-prediction/'
get_ipython().system('ls ../input/ashrae-energy-prediction')


# **Reduce Memory function**

# In[2]:


#Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",df[col].dtype)            
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            print("min for this col: ",mn)
            print("max for this col: ",mx)
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                NAlist.append(col)
                df[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",df[col].dtype)
            print("******************************")
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df, NAlist


# **RMSLE calculation** 

# In[3]:


def rmsle(y, y_pred):
    '''
    A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
    source: https://www.kaggle.com/marknagelberg/rmsle-function
    '''
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5


# In[4]:


# from: https://www.kaggle.com/bejeweled/ashrae-catboost-regressor
def RMSLE(y_true, y_pred, *args, **kwargs):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


# # Data

# In[5]:


building_df = pd.read_csv(PATH+"building_metadata.csv")
weather_train = pd.read_csv(PATH+"weather_train.csv")
train = pd.read_csv(PATH+"train.csv")


# **building_meta.csv**
# - ```site_id``` - Foreign key for the weather files.
# - ```building_id``` - Foreign key for ```training.csv```
# - ```primary_use``` - Indicator of the primary category of activities for the building based on [EnergyStar property type definitions](https://www.energystar.gov/buildings/facility-owners-and-managers/existing-buildings/use-portfolio-manager/identify-your-property-type)
# - ```square_feet``` - Gross floor area of the building
# - ```year_built``` - Year building was opened
# - ```floor_count``` - Number of floors of the building
# 

# In[6]:


building_df.head()


# **weather_[train/test].csv**
# - ```site_id```
# - ```air_temperature``` - Degrees Celsius
# - ```cloud_coverage``` - Portion of the sky covered in clouds, in [oktas](https://en.wikipedia.org/wiki/Okta)
# - ```dew_temperature``` - Degrees Celsius
# - ```precip_depth_1_hr``` - Millimeters
# - ```sea_level_pressure``` - Millibar/hectopascals
# - ```wind_direction``` - Compass direction (0-360)
# - ```wind_speed``` - Meters per second
# 

# In[7]:


weather_train.head()


# **train.csv**
# - ```building_id``` - Foreign key for the building metadata.
# - ```meter``` - The meter id code. Read as ```{0: electricity, 1: chilledwater, 2: steam, hotwater: 3}```. Not every building has all meter types.
# - ```timestamp``` - When the measurement was taken
# - ```meter_reading``` - The target variable. Energy consumption in kWh (or equivalent). Note that this is real data with measurement error, which we expect will impose a baseline level of modeling error.
# 

# In[8]:


train.head()


# ### Prepare training and test

# In[9]:


train = train.merge(building_df, left_on = "building_id", right_on = "building_id", how = "left")
train = train.merge(weather_train, left_on = ["site_id", "timestamp"], right_on = ["site_id", "timestamp"])


# In[10]:


#test = test.merge(weather_test, left_on = ["timestamp"], right_on = ["timestamp"])
#del weather_test


# # Simple FE: Timestamp
# 
# - **Break** ```timestamp``` into: year, month, day, weekday, hour.

# In[11]:


train.timestamp[0]


# In[12]:


train["timestamp"] = pd.to_datetime(train["timestamp"])
train["hour"] = train["timestamp"].dt.hour
train["day"] = train["timestamp"].dt.day
train["weekend"] = train["timestamp"].dt.weekday
train["month"] = train["timestamp"].dt.month
train["year"] = train["timestamp"].dt.year
print ('TRAIN: ', train.shape)
train.head(3)


# # EDA

# In[13]:


train.head(8)


# ### Dates
# 
# **Train:** from ```2016-01-01 00:00:00``` to ```2016-12-31 23:00:00```
# 
# **Test:** from ```'2017-01-01 00:00:00'``` to ```'2018-05-09 07:00:00'```

# In[14]:


print ('START : ', train.timestamp[0] )
print ('END : ', train.timestamp[train.shape[0]-1])
print ('MONTHS :', train.month.unique())


# ### Missing data x Column

# In[15]:


for col in train.columns:
    if train[col].isna().sum()>0:
        print (col,train[col].isna().sum())


# ### Meter type
# > Not every building has all meter types.

# In[16]:


sns.countplot(x='meter', data=train).set_title('{0: electricity, 1: chilledwater, 2: steam, hotwater: 3}\n\n')


# **Buildings with all meter types**: (building, site)
# 
# ```
# [(1232, 14), (1241, 14), (1249, 14), (1258, 14), (1259, 14), (1293, 14), (1294, 14), (1295, 14), (1296, 14), (1297, 14), (1298, 14), (1301, 14), (1331, 15)]
# ```
# 
# If you want to check, just run the code bellow

# In[17]:


'''
building_4 = []
for b in train.building_id.unique():
    cond = train[train.building_id==b]['meter'].nunique()
    place = train[train.building_id==b]['site_id'].unique()[0]
    if cond == 4:
        building_4.append((b,place))
        
print (building_4)
'''


# ### Buildings and sites
# 
# Each building is at only one site!

# In[18]:


print ('We have {} buildings'.format(train.building_id.nunique()))
print ('We have {} sites'.format(train.site_id.nunique()))
print ('More information about each site ...')
for s in train.site_id.unique():
    print ('Site ',s, '\tobservations: ', train[train.site_id == s].shape[0], '\tNum of buildings: ',train[train.site_id == s].building_id.nunique())


# In[19]:


# Prove that each building is only at one site
for b in train.building_id.unique():
    if train[train.building_id == b].site_id.nunique() >1:
        print (train[train.building_id == b].site_id.nunique())


# **Top 5 consuming buildings**

# In[20]:


top_buildings = train.groupby("building_id")["meter_reading"].mean().sort_values(ascending = False).iloc[:5]
for value in top_buildings.index:
    train[train["building_id"] == value]["meter_reading"].rolling(window = 24).mean().plot()
    pyplot.title('Building {} at site: {}'.format(value,train[train["building_id"] == value]["site_id"].unique()[0]))
    pyplot.show()


# ### Old buildings
# 
# I'm not an expert in the field but probably old buildings consume more!

# In[21]:


print ('Buildings built before 1900: ', train[train.year_built <1900].building_id.nunique())
print ('Buildings built before 2000: ', train[train.year_built <2000].building_id.nunique())
print ('Buildings built after 2010: ', train[train.year_built >=2010].building_id.nunique())
print ('Buildings built after 2015: ', train[train.year_built >=2015].building_id.nunique())


# In[22]:


build_corr = train[['building_id','year_built','meter_reading']].corr()
print (build_corr)
del build_corr


# ### primary_use

# In[23]:


fig, ax = pyplot.subplots(figsize=(10, 8))
sns.countplot(y='primary_use', data=train)


# In[24]:


fig, ax = pyplot.subplots(figsize=(10, 8))
sns.countplot(y='primary_use', data=train, hue= 'month')


# ## is site_id the key?

# In[25]:


train.groupby('site_id')['meter_reading'].describe()


# **Click ```output``` to see the plots**

# In[26]:


for s in train.site_id.unique():
    train[train["site_id"] == s].plot("timestamp", "meter_reading")


# ## Consume x Site

# In[27]:


for s in train.site_id.unique():
    np.log1p(train[train['site_id']==s].meter_reading).plot.hist(figsize=(6, 4), bins=10, title='Dist. of Electricity Power Consumption on Site {}'.format(s))
    plt.xlabel('LOG Power (kWh)')
    plt.show()


# ## Correlation plot

# In[28]:


fig, ax = plt.subplots(figsize = (17,8))
corr = train.corr()
ax = sns.heatmap(corr, annot=True,
            xticklabels = corr.columns.values,
            yticklabels = corr.columns.values)
plt.show()


# # Understanding the target: meter_reading. 
# 
# ```meter_reading``` - The target variable. Energy consumption in **kWh** (or equivalent). Note that this is real data with measurement error, which we expect will impose a baseline level of modeling error.
# 
# ![](https://www.solarschools.net/build/img/learn/energy/electricity/kw-kwh-explained//kwh-explained-diagram_400_resize_q95.jpg)
# 
# **How do they measure this?**
# 
# <img src="https://modernsurvivalblog.com/wp-content/uploads/2015/04/kill-a-watt-kilowatt-hour-meter.jpg" width="200" height="200"> 
# 
# <br>
# ### Differences between kWh and KW:
# 
# <img src="https://www.boilerguide.co.uk/data/imagecache/content_images/wpimages-boilerguide.co.uk/2018/10/15095900/kW-and-kWh-Explained.png" width="300" height="300"> 
# 
# ![](https://www.onetemp.com.au/images/thumbs/0003743_what-is-the-difference-between-kw-and-kwh_510.png)
# 
# <br>
# 
# ### Target = 0?
# commented at the post: [what does 0.0 means in target variable](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/113054#latest-651232)
# 
# Understanding the difference between kW (Power) and kWh (energy), why don't some buildings consume (meter_reading ==0)? some reasons:
# - Those buildings don't consume energy (weird).
# - The stuff they use for measuring was broken.
# - **Missing data!** See the next example with the building ```0```. Spoiler: They started to measure Building ```0``` at June 2016! (month=6)

# In[29]:


print ('Dataset with meter_reading = 0')
df0 = train[train.meter_reading==0]
print (df0.shape, df0.shape[0]/train.shape[0] ,'% of total data')
df0.head(3)


# Let's take the 1st day ```2016-01-01``` of the building ```0```

# In[30]:


# I only show from 0 to 12 am.
print ('Month with no consume: ', df0[(df0.building_id==0)].month.unique())
df0[(df0.building_id==0) & (df0.year== 2016) & (df0.month== 1) & (df0.day== 1)].head(12)


# The months: 1 to 5 the building ```0``` didn't consume nothing!
# Let's see the followinf months (6 to 12):

# In[31]:


train[(train.building_id==0) & (train.year== 2016) & (train.month== 6) & (train.day== 1)].head(2)


# In[32]:


train[(train.building_id==0) & (train.year== 2016) & (train.month== 12) & (train.day== 1)].head(2)


# ### Conclusion
# They started to measure Building ```0``` at June 2016! (month=6)
# 
# In my opinion they started to measure the consume of that specific building in June 2016.
# In order to complete the database from 2016, they filled the other months with 0.
# Note that they can do it because the meteorological variables aren't a problem, you can access to historical data from external sources, and other variables about the building are constant like the year_built, how big is the building, floors etc… And that's why they could create the database and include new buildings since 2016 :)

# In[33]:


# dirty and fast
build_info = pd.DataFrame (columns = ['building_id', 'start'])
info = [] # (building, start)


# In[34]:


for b in tqdm(train.building_id.unique()):
    if b in df0.building_id.unique():
        start = df0[(df0.building_id==b) & (df0.meter_reading==0)]['month'].unique()[-1]+1
        info.append((b, start))
    else:
        # those buildings with no metric_reading=0 --> they have measurements from 2016-1-1
        info.append((b, 1))


# In[35]:


build_info ['building_id'] = [x[0] for x in info]
build_info ['start'] = [x[1] for x in info]
build_info.head()


# In[36]:


print (build_info[build_info.start == 1].shape)


# In[37]:


build_info[build_info.start == 13].shape
build_info[build_info.start == 13].head()


# **This is not right yest!**

# In[38]:


build_info.to_csv('build_info.csv', index=False)


# **Buildings where the proportion of missing >= 0.5**
# 
# > 53, 799, 815 , 817 , 853 , 857 , 1113 , 1221 , 754 , 954 , 1446 , 783
# 
# 

# In[39]:


# Buildings where the proportion of missing >= 0.5

'''
for build in tqdm(train.building_id.unique()):
    a = train[(train.building_id==build) & (train.meter_reading==0)].shape[0] 
    b = train[(train.building_id==build)].shape[0]
    if a/b >= 0.5:
        print (build)
'''


# In[40]:


del df0
gc.collect()


# <br>
# # Training

# In[41]:


del weather_train, building_df
gc.collect()


# **Delete time stamp and encode ```primary_use```**

# In[42]:


train = train.drop("timestamp", axis = 1)
le = LabelEncoder()
train["primary_use"] = le.fit_transform(train["primary_use"])


# In[43]:


train.head(3)


# In[44]:


categoricals = ["building_id", "primary_use", "hour", "day", "weekend", "month", "meter", 'year']

drop_cols = ["precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed"]

numericals = ["square_feet", "year_built", "air_temperature", "cloud_coverage",
              "dew_temperature"]

feat_cols = categoricals + numericals


# In[45]:


target = np.log1p(train["meter_reading"])


# In[46]:


train = train.drop(drop_cols + ["site_id","floor_count","meter_reading"], axis = 1)
#train.fillna(-999, inplace=True)
train.head()


# In[47]:


train, NAlist = reduce_mem_usage(train)


# ## Validation

# **Initial features**

# In[48]:


# Features
print (train.shape)
train[feat_cols].head(3)


# In[49]:


# target = np.log1p(train["meter_reading"])
# raw_target = np.expm1(target)


# In[50]:


num_folds = 5
kf = KFold(n_splits = num_folds, shuffle = True, random_state = 42)
error = 0

for fold, (train_index, val_index) in enumerate(kf.split(train, target)):

    print ('Training FOLD ',fold,'\n')
    print('Train index:','\tfrom:',train_index.min(),'\tto:',train_index.max())
    print('Valid index:','\tfrom:',val_index.min(),'\tto:',val_index.max(),'\n')
    
    train_X = train[feat_cols].iloc[train_index]
    val_X = train[feat_cols].iloc[val_index]
    train_y = target.iloc[train_index]
    val_y = target.iloc[val_index]
    lgb_train = lgb.Dataset(train_X, train_y)
    lgb_eval = lgb.Dataset(val_X, val_y)
    
    params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'rmse'},
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9
            }
    
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=(lgb_train, lgb_eval),
               early_stopping_rounds=20,
               verbose_eval = 20)

    y_pred = gbm.predict(val_X, num_iteration=gbm.best_iteration)
    error += np.sqrt(mean_squared_error(y_pred, (val_y)))/num_folds
    
    print('\nFold',fold,' Score: ',np.sqrt(mean_squared_error(y_pred, val_y)))
    #print('RMSLE: ', rmsle(y_pred, val_y))
    #print('RMSLE_2: ', np.sqrt(mean_squared_log_error(y_pred, (val_y))))

    del train_X, val_X, train_y, val_y, lgb_train, lgb_eval
    gc.collect()

    print (20*'---')
    break
    
print('CV error: ',error)


# In[51]:


# memory allocation
del train, target
gc.collect()


# ### Plot importance

# In[52]:


import matplotlib.pyplot as plt
feature_imp = pd.DataFrame(sorted(zip(gbm.feature_importance(), gbm.feature_name()),reverse = True), columns=['Value','Feature'])
plt.figure(figsize=(10, 5))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()


# ## Prepare Test

# In[53]:


#preparing test data
building_df = pd.read_csv(PATH+"building_metadata.csv")
test = pd.read_csv("../input/ashrae-energy-prediction/test.csv")
test = test.merge(building_df, left_on = "building_id", right_on = "building_id", how = "left")
del building_df
gc.collect()


# In[54]:


weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")
weather_test = weather_test.drop(drop_cols, axis = 1)
test = test.merge(weather_test, left_on = ["site_id", "timestamp"], right_on = ["site_id", "timestamp"], how = "left")
del weather_test
gc.collect()


# In[55]:


test.head()


# **Reduce Memory**

# In[56]:


test["primary_use"] = le.transform(test["primary_use"])
test, NAlist = reduce_mem_usage(test)


# **Change dates type**

# In[57]:


test["timestamp"] = pd.to_datetime(test["timestamp"])
test["hour"] = test["timestamp"].dt.hour.astype(np.uint8)
test["day"] = test["timestamp"].dt.day.astype(np.uint8)
test["weekend"] = test["timestamp"].dt.weekday.astype(np.uint8)
test["month"] = test["timestamp"].dt.month.astype(np.uint8)
test["year"] = test["timestamp"].dt.year.astype(np.uint8)
test = test[feat_cols]
test.head()


# ### Inference

# In[58]:


from tqdm import tqdm
i=0
res=[]
step_size = 50000 
for j in tqdm(range(int(np.ceil(test.shape[0]/50000)))):
    res.append(np.expm1(gbm.predict(test.iloc[i:i+step_size])))
    i+=step_size


# In[59]:


del test
gc.collect()


# # Submission

# In[60]:


res = np.concatenate(res)
sub = pd.read_csv(PATH+"sample_submission.csv")
sub["meter_reading"] = res
sub.to_csv("submission.csv", index = False)
sub.head(10)

