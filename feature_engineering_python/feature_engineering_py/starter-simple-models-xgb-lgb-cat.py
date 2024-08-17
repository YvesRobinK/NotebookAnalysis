#!/usr/bin/env python
# coding: utf-8

# 
# 
# 

# <center><h1>Tabular Playground Series - Jul 2021<h1> <center> 
#     <center> <h5> I hope you find this helpful 😊 <h5> <center>

# # Problem 
# In this competition we are predicting the values of air pollution measurements over time, based on basic weather information (temperature and humidity) and the input values of 5 sensors. we will first do exploratory analysis and after that we will build a model.

# ### Data source 

# The data is available at [this link](https://www.kaggle.com/c/tabular-playground-series-jul-2021/data) and it contains this files.
# *  train.csv - the training data, including the weather data, sensor data, and values for the 3 targets
# *  test.csv - the same format as train.csv, but without the target value; your task is to predict the value for each of these targets.
# * sample_submission.csv - a sample submission file in the correct format.

# ### Notebook setup 
# let's start by loading the diffrent libraries and packages.

# In[1]:


#importing common libraries  
import os , random
import datetime
import numpy as np # linear algebra
import pandas as pd # data processing, 
import matplotlib.pyplot as plt # ploting
import seaborn as sns # visualisation 
import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
import catboost as cat
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
sns.set_style("ticks")


# **Loading the Data**

# In[2]:


train_data = pd.read_csv("../input/tabular-playground-series-jul-2021/train.csv"  ) # reading the train data to a data frame 
test_data = pd.read_csv('../input/tabular-playground-series-jul-2021/test.csv' ) # reading the test data into a data frame 
sample_submission = pd.read_csv('../input/tabular-playground-series-jul-2021/sample_submission.csv') # reading the test data into a data frame
print(" data imported keep going....")


# In[3]:


train_data.head()
train_data.tail()


# In[4]:


train_data.shape #7111,12
train_data.info()


# **Checking missing or null values**

# In[5]:


train_data.isnull().sum() # no null values so we can continue


# **Basic summary statistics**

# In[6]:


# describing the data beautifully
train_data.describe().T.style.bar().background_gradient().background_gradient()


# 
# 
# 
# **Checking Correlation**

# In[7]:


corrMatrix =train_data.corr(method='pearson', min_periods=1)
corrMatrix 


# In[8]:


sns.set(rc={"figure.figsize":(10, 8)})
sns.heatmap(corrMatrix, cmap="YlGnBu",annot=True)
plt.show()


# 
# Looks some features are more related to targets than other features 

# In[9]:


#Variation
train_data.var()


# In[10]:


#Standard  deviation  
train_data.std()


# **Variables Histogram**

# In[11]:


train_data.hist(bins=10,color='#A0E8AF',figsize=(16,12))
plt.show()


# **Variables's Boxplot**

# In[12]:


sns.set(rc={"figure.figsize":(14, 6)})
plot = train_data.iloc[:,:9]
sns.boxplot(data=plot)


# As we can see sensor ( 1,2,3,4,5) come with much outliers. le's invesitage sensors columns one more time by ploting their distribution.

# In[13]:


features = train_data.iloc[:,1:9]
features 


# In[14]:


# we will look into the features distribution now, to get insight into the data
i = 1
plt.figure()
fig, ax = plt.subplots(5, 3,figsize=(14, 24))
for feature in features:
    plt.subplot(3, 3,i)
    sns.distplot(train_data[feature],color="blue", kde=True,bins=120, label='train')
    sns.distplot(test_data[feature],color="red", kde=True,bins=120, label='test')
    plt.xlabel(feature, fontsize=9); plt.legend()
    i += 1
plt.show()



# **Target distrubitions**

# In[15]:


targets = ["target_carbon_monoxide", "target_benzene", "target_nitrogen_oxides"]
plt.rcParams['figure.dpi'] = 600
fig = plt.figure(figsize=(5, 1), facecolor='#f6f6f4')
gs = fig.add_gridspec(1, 3)
gs.update(wspace=0.2, hspace=0.5)

background_color = "#f6f5f5"

run_no = 0
for row in range(0, 1):
    for col in range(0, 3):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        locals()["ax"+str(run_no)].tick_params(axis='y', which=u'both',length=0)
        locals()["ax"+str(run_no)].set_yticklabels([])
        for s in ["top","right"]:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

run_no = 0
for col in targets:
    sns.kdeplot(train_data[col], ax=locals()["ax"+str(run_no)], shade=True, color='darkblue', alpha=0.95, linewidth=0, zorder=2)
    locals()["ax"+str(run_no)].set_ylabel('')
    locals()["ax"+str(run_no)].set_xlabel(col, fontsize=5, fontweight='bold')
    locals()["ax"+str(run_no)].tick_params(labelsize=5, width=0.5, length=1.5)
    locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='#EEEEEF', linewidth=0.7)
    locals()["ax"+str(run_no)].grid(which='major', axis='y', zorder=0, color='#EEEEEF', linewidth=0.7)
    run_no += 1
    
ax0.text(-1.2, 0.44, 'Target Distribution', fontsize=8, fontweight='bold')
plt.show()


# **A quick look at the test data**

# In[16]:


test_data.head()
test_data.tail()


# In[17]:


test_data.shape #2247,9
test_data.info()


# In[18]:


test_data.describe().T.style.bar().background_gradient().background_gradient()


# ### Building The Model
# For this problem we are going to use ensemnling predictions of three models (  XGB,LGB and CatB ).

# #### Data preparation and some Features Engineering

# In[19]:


train_data.set_index('date_time')
test_data.set_index('date_time' )


# In[20]:


#concating the data 
all_df = pd.concat([train_data, test_data]).reset_index(drop = True)
print(all_df.shape)


# In[21]:


# Some Features Engineering check discussion for more .
all_df['sensor_7'] = (all_df['sensor_3'] - all_df['sensor_4']) / all_df['sensor_4']
all_df['Dew_Point'] = 243.12*(np.log(all_df['relative_humidity'] * 0.01) + (17.62 * all_df['deg_C'])/(243.12+all_df['deg_C']))/(17.62-(np.log(all_df['relative_humidity'] * 0.01)+17.62*all_df['deg_C']/(243.12+all_df['deg_C'])))
all_df['SMC'] = (all_df['absolute_humidity'] * 100) / all_df['relative_humidity']
all_df['temperature_lag_3'] = all_df['deg_C'] - all_df['deg_C'].shift(periods=3, fill_value=0)
all_df['temperature_lag_6'] = all_df['deg_C'] - all_df['deg_C'].shift(periods=6, fill_value=0)
all_df['Partial_pressure'] = 243.12*(np.log(all_df['absolute_humidity'] * 0.01) + (17.62 * all_df['deg_C'])/(243.12+all_df['deg_C']))/(17.62-(np.log(all_df['relative_humidity'] * 0.01)+17.62*all_df['deg_C']/(243.12+all_df['deg_C'])))
all_df ['Saturated_wvd'] = (all_df ['absolute_humidity'] * 100) / all_df ['relative_humidity']
all_df['humidity_lag_3'] = all_df['absolute_humidity'] - all_df['absolute_humidity'].shift(periods=3, fill_value=0)
all_df['humidity_lag_6'] = all_df['absolute_humidity'] - all_df['absolute_humidity'].shift(periods=6, fill_value=0)
#droping the sensor3 since it has the minimal correlaion with the targets 
all_df.drop(['sensor_3'],axis=1,inplace=True)
# date_time_features 
all_df['date_time'] = pd.to_datetime(all_df['date_time'])
all_df['month'] = all_df['date_time'].dt.month
all_df['week'] = all_df['date_time'].dt.week
all_df['day'] = all_df['date_time'].dt.day
all_df['hour'] = all_df['date_time'].dt.hour
all_df["working_hours"] =  all_df["hour"].isin(np.arange(8, 21, 1)).astype("int")
all_df["quarter"] = all_df["date_time"].dt.quarter
all_df["is_weekend"] = (all_df["date_time"].dt.dayofweek >= 5).astype("int")
all_df.shape


# In[22]:


# seperating the train and test data again 
train, test = all_df.iloc[:(len(all_df) - len(test_data)), :], all_df.iloc[(len(all_df) - len(test_data)):, :]
print(train.shape, test.shape)


# In[23]:


# converting date time to unix time
train['date_time'] = train['date_time'].astype('datetime64[ns]').astype(np.int64)/10**9
test['date_time'] = test['date_time'].astype('datetime64[ns]').astype(np.int64)/10**9
# preparing targets 
labels = ['target_carbon_monoxide','target_benzene','target_nitrogen_oxides'] 
test = test.drop(columns= labels)
X_train= train.drop(columns=labels)  
y_carbon_monoxide = train['target_carbon_monoxide']
y_benzene = train['target_benzene']
y_nitrogen_oxides = train['target_nitrogen_oxides']


# **XGB Model prediction**

# In[24]:


#paramaters found using gridsearch 
xgb_submission = sample_submission.copy()
#######define fit and predict for carbon_monoxide ######
xgboost = XGBRegressor( colsample_bytree=0.7, 
                       learning_rate = 0.03 ,
                       n_estimators=500, 
                       subsample=0.7,
                       alpha=0.9) # define 
xgboost.fit(X_train, y_carbon_monoxide) #fit
xgb_submission['target_carbon_monoxide'] = xgboost.predict(test) #predict 
######### fit and predict for benzen ########
xgboost.fit(X_train, y_benzene) #fit
xgb_submission['target_benzene'] = xgboost.predict(test) #predict
######fit and predict for nitrogen_oxide#######
xgboost.fit(X_train, y_nitrogen_oxides) #fit
xgb_submission['target_nitrogen_oxides'] = xgboost.predict(test) #predict 
xgb_submission.head()


# **lGB model prediction**

# **Gridsearch to find the best parameters** 

# In[25]:


lgb1 = LGBMRegressor()
parameters = { 
               'objective' :['regression'], 
                'max_depth' : [5,7,9],
                'learning_rate': [.001, 0.05, .07], 
                'n_estimators': [500,700,300],
                'num_leaves':[30,40,50],
                ' max_bin':[55,35,75],
                'bagging_seed' :[7,9,5],
                                    }

lgb_grid = GridSearchCV(lgb1,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)


# In[26]:


lgb_grid.fit(X_train,
         y_nitrogen_oxides)
print(lgb_grid.best_score_)
print(lgb_grid.best_params_)


# In[27]:


lgb_submission = sample_submission.copy()
lightgbm =LGBMRegressor(    
                                       objective='regression', 
                                       max_depth = 9,
                                       num_leaves=30,
                                       learning_rate=0.07, 
                                       n_estimators=300,
                                       max_bin=75, 
                                       bagging_seed=7,
                                       feature_fraction_seed=7,
                                       verbose=-1,
                                       )
lightgbm.fit(X_train, y_carbon_monoxide)
lgb_submission['target_carbon_monoxide'] = lightgbm.predict(test) #predict
#####
lightgbm.fit(X_train, y_benzene) #fit
lgb_submission['target_benzene'] = lightgbm.predict(test) #predict
####
lightgbm.fit(X_train, y_nitrogen_oxides) #fit
lgb_submission['target_nitrogen_oxides'] = lightgbm.predict(test) #predict 
lgb_submission.head()


# **CatBoost model**

# **GridSearch to find the best paramaters**

# In[28]:


catb1 = CatBoostRegressor()
params = {'iterations': [500,300,100],
          'depth': [4, 5, 6],
         'learning_rate': [.001, .04, .07], 
          'l2_leaf_reg': np.logspace(-20, -19, 3),
          'leaf_estimation_iterations': [10],
          'eval_metric': ['RMSE'],
          'logging_level':['Silent'],
          'random_seed': [42]
         }

catb_grid = GridSearchCV(catb1,
                        params,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)
catb_grid.fit(X_train,
         y_benzene)


# In[29]:


print(catb_grid.best_score_)
print(catb_grid.best_params_)


# In[30]:


cat_submission = sample_submission.copy()
cat_boost = CatBoostRegressor(
                 learning_rate = 0.04,
                  iterations =300,
                 l2_leaf_reg = 1e-20,
                 bagging_temperature=4,
                 leaf_estimation_iterations=10,
                 random_strength = 1.5,
                 depth= 4,
                 random_seed = 42,
                 eval_metric = 'RMSE',
                 logging_level='Silent' ,           
               
)
         
cat_boost.fit(X_train, y_carbon_monoxide)
cat_submission['target_carbon_monoxide'] = cat_boost.predict(test) #predict
#####
cat_boost.fit(X_train, y_benzene) #fit
cat_submission['target_benzene'] = cat_boost.predict(test) #predict
####
cat_boost.fit(X_train, y_nitrogen_oxides) #fit
cat_submission['target_nitrogen_oxides'] = cat_boost.predict(test) #predict 
cat_submission.head()


# **Ensemble predictions**

# In[31]:


ensembe_sub = sample_submission.copy()
ensembe_sub['target_carbon_monoxide'] = 0.4*lgb_submission['target_carbon_monoxide'] + 0.2*cat_submission['target_carbon_monoxide'] + 0.4*xgb_submission['target_carbon_monoxide'] 
ensembe_sub['target_benzene'] = 0.4*lgb_submission['target_benzene'] + 0.2*cat_submission['target_benzene'] + 0.4*xgb_submission['target_benzene'] 
ensembe_sub['target_nitrogen_oxides'] = 0.4*lgb_submission['target_nitrogen_oxides'] + 0.2*cat_submission['target_nitrogen_oxides'] + 0.4*xgb_submission['target_nitrogen_oxides']
ensembe_sub.head()


# In[32]:


# saving a submissions
#xgb_prediction
xgb_submission.to_csv('xgb_submission.csv', index=False)
#lgb_prediction
lgb_submission.to_csv('lgb_submission.csv', index=False)
#cat_prediction
cat_submission.to_csv('cat_submission.csv', index=False)
#essemble prediction
ensembe_sub.to_csv('ensembled_submission.csv', index=False)



# ### Recommendation
# XGB ,LGB and CatB showed a good performance in predicting the targets but it still need to be improved,for that you may play with Hyperparameters . However I recommend spending most of your time on feature-engineering.And ofcourse you can experiment with other methods / autoMl libraries.

# <center> <h3> If you find this usefull you can UPvote , Thank you😊 <h3> <center> 
# 

# In[ ]:




