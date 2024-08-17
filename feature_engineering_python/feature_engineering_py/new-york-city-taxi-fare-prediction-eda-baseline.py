#!/usr/bin/env python
# coding: utf-8

# # What did I learnt from this notebook
# 1. Always create a outline of the project as it `give us direction`
# 2. How to handle **Large dataset**
# 
# # What is the main objective of this Notebook
# 1. Perform EDA and create a Baseline Model on sample data (20% of training set)
# 2. Reduce the size of training set and than train the a model better than this Notebook
#     a. In next notebook use stacking and blending to achieve more respectable result

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

pd.set_option('display.max_columns', None)


# ## About the data:
# Data fields
# ID
# * **key** - Unique string identifying each row in both the training and test sets. Comprised of pickup_datetime plus a unique integer, but this doesn't matter, it should just be used as a unique ID field. Required in your submission CSV. Not necessarily needed in the training set, but could be useful to simulate a 'submission file' while doing cross-validation within the training set.
# ### Features**
# * **pickup_datetime** - timestamp value indicating when the taxi ride started.
# * **pickup_longitude** - float for longitude coordinate of where the taxi ride started.
# * **pickup_latitude** - float for latitude coordinate of where the taxi ride started.
# * **dropoff_longitude** - float for longitude coordinate of where the taxi ride ended.
# * **dropoff_latitude** - float for latitude coordinate of where the taxi ride ended.
# * **passenger_count** - integer indicating the number of passengers in the taxi ride.
# ### Target
# fare_amount - float dollar amount of the cost of the taxi ride. This value is only in the training set; this is what you are predicting in the test set and it is required in your submission CSV.

# In[2]:


#%%time
#sub_df= pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv')
#test_df= pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/test.csv')
#df= pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv')
#df.sample(5)


# training data is 5.5GB so it fails to load 
# so lets analyse data with shell commands

# ## Basic Terminal Navigation Commands: 
# 
# **ls** : To get the list of all the files or folders.  
# **ls -l**: Optional flags are added to ls to modify default behavior, listing contents in extended form -l is used for “long” output  
# **ls -a**: Lists of all files including the hidden files, add -a  flag   
# **cd**: Used to change the directory.  
# **du**: Show disk usage.  
# **pwd**: Show the present working directory.  
# **man**: Used to show the manual of any command present in Linux.  
# **rmdir**: It is used to delete a directory if it is empty.  
# **ln file1 file2**: Creates a physical link.  
# **ln -s file1 file2**: Creates a symbolic link.  
# **locate**: It is used to locate a file in Linux System  
# **echo**:  This command helps us move some data, usually text into a file.      
# **df**: It is used to see the available disk space in each of the partitions in your system.      
# **tar**: Used to work with tarballs (or files compressed in a tarball archive)       
# 
# ### [For more details...](https://www.geeksforgeeks.org/basic-shell-commands-in-linux/)

# In[3]:


data_dir = '../input/new-york-city-taxi-fare-prediction'
get_ipython().system('ls -lh {data_dir}')


# In[4]:


get_ipython().run_cell_magic('time', '', '!wc -l {data_dir}/train.csv\n')


# In[5]:


get_ipython().run_cell_magic('time', '', '!wc -l {data_dir}/test.csv\n')


# In[6]:


get_ipython().run_cell_magic('time', '', '!wc -l {data_dir}/sample_submission.csv\n')


# Test and Submission have a difference of 1 row... we will look into this but this could me mostly an empty line

# ### Lets look at the 1st few lines of each dataset

# In[7]:


# Training set
get_ipython().system('head {data_dir}/train.csv')


# In[8]:


# Test set
get_ipython().system('head {data_dir}/test.csv')


# In[9]:


#  Sample sub
get_ipython().system('head {data_dir}/sample_submission.csv')


# ### Observations:
# 
# - This is a supervised learning regression problem
# - Training data is 5.5 GB in size
# - Training data has 55 million rows (`55,423,856 rows`) 
# - Test set is much smaller (`9,914 rows`)
# - The training set has 8 columns:
#     - `key` (a unique identifier)
#     - `fare_amount` (target column)
#     - `pickup_datetime`
#     - `pickup_longitude`
#     - `pickup_latitude`
#     - `dropoff_longitude`
#     - `dropoff_latitude`
#     - `passenger_count`
# - The test set has all columns except the target column `fare_amount`.
# - The submission file should contain the `key` and `fare_amount` for each test sample.
# - Evaluation is donw with **RMSE**
# 

# ## Loading data
# Since we can't load training data full 
# lets load it in pieces

# In[10]:


df_test = pd.read_csv(data_dir+'/test.csv',parse_dates=['pickup_datetime'])
df_test


# In[11]:


df_test.info()


# `Why key is an object data type to me this these number in the dataframe?`

# In[12]:


df_test.key.nunique()


# ### Now lets load training data
# I will avoid key column

# In[13]:


import random
## to select random index no from training dataset


# In[14]:


# Change this
sample_frac = 0.20
# we are loading 20% data : Wall time: 28min 35s ; memory usage: 327.6 MB
# 10% data loading : Wall time: 14min 38s


# * numpy.finfo(numpy.float16).precision **> 3**
# * numpy.finfo(numpy.float32).precision **> 6** (about 8 digit)
# * numpy.finfo(numpy.float64).precision **> 15**
# * numpy.finfo(numpy.float128).precision **> 18**
# * A UINT8 is an **8-bit `unsigned integer` (range: `0 through 255` decimal)** > Generally Taxi can accommodate single digit passangers so 255 is still over kill.

# In[15]:


get_ipython().run_cell_magic('time', '', 'selected_cols = \'fare_amount,pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count\'.split(\',\')\ndtypes = {\n    \'fare_amount\': \'float16\',\n    \'pickup_longitude\': \'float32\',\n    \'pickup_latitude\': \'float32\',\n    \'dropoff_longitude\': \'float32\',\n    \'passenger_count\': \'uint8\'\n}\n## this function will return True for (1 -sample_frac) thus these rows will be skipped\ndef skip_row(row_idx):\n    if row_idx == 0:\n        return False\n    return random.random() > sample_frac  ## \n\nrandom.seed(7)\ndf = pd.read_csv(data_dir+"/train.csv", \n                 usecols=selected_cols, \n                 dtype=dtypes, \n                 parse_dates=[\'pickup_datetime\'], \n                 skiprows=skip_row)\ndf_original =df.copy()\ndf\n')


# In[16]:


## Exporting 20% data to csv for further prediction
df.to_csv('20% ofnew-york-city-taxi-fare-predicition.csv')


# In[17]:


df.info()


# # 2. Explore the Dataset
# 
# - Basic info about training set
# - Basic info about test set
# - Exploratory data analysis & visualization
# - Ask & answer questions

# In[18]:


df.isnull().sum()


# Dataset has no missing values

# In[19]:


get_ipython().run_cell_magic('time', '', '#### Checking for duplicates\ndf.duplicated().sum()\n')


# In[20]:


get_ipython().run_cell_magic('time', '', '# There are 42 duplicates, Lets remove them\ndf.drop_duplicates()  ## dropes duplicates\ndf.duplicated().sum()\n')


# In[21]:


df.describe()


# Why fare has min value as -ve and max value = infinite

# # 3. Feature Engineering
# After some exploraation I realised I sholud perform **Feature Engineering** to understand the data properly  
# as I saw fare to be -ve in around 463 rows and fare value more than 500 few time and also infinite twice or trice
# 1. add a feature for distance between pickup place and drop place
# 2. Need to perform feature extraction for time

# ### Add Distance Between Pickup and Drop
# 
# We can use the haversine distance: 
# - https://en.wikipedia.org/wiki/Haversine_formula
# - https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas

# In[22]:


import numpy as np

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


# In[23]:


def add_trip_distance(df):
    df['trip_distance'] = haversine_np(df['pickup_longitude'], df['pickup_latitude'], df['dropoff_longitude'], df['dropoff_latitude'])


# In[24]:


get_ipython().run_cell_magic('time', '', 'add_trip_distance(df)\n')


# ### Extract Parts of Date
# 
# - Year
# - Month
# - Day
# - Weekday
# - Hour
# 

# In[25]:


def add_dateparts(df, col):
    df['year'] = df[col].dt.year
    df['month'] = df[col].dt.month
    df['day'] = df[col].dt.day
    df['weekday'] = df[col].dt.weekday
    df[col + '_hour'] = df[col].dt.hour


# In[26]:


get_ipython().run_cell_magic('time', '', "add_dateparts(df, 'pickup_datetime')\n")


# ### Add Distance From Popular Landmarks
# 
# - JFK Airport
# - LGA Airport
# - EWR Airport
# - Times Square
# - Met Meuseum
# - World Trade Center
# 
# We'll add the distance from drop location. 

# In[27]:


jfk_lonlat = -73.7781, 40.6413
lga_lonlat = -73.8740, 40.7769
ewr_lonlat = -74.1745, 40.6895
met_lonlat = -73.9632, 40.7794
wtc_lonlat = -74.0099, 40.7126


# In[28]:


def add_landmark_dropoff_distance(df, landmark_name, landmark_lonlat):
    lon, lat = landmark_lonlat
    df[landmark_name + '_drop_distance'] = haversine_np(lon, lat, df['dropoff_longitude'], df['dropoff_latitude'])


# In[29]:


get_ipython().run_cell_magic('time', '', "for name, lonlat in [('jfk', jfk_lonlat), ('lga', lga_lonlat), ('ewr', ewr_lonlat), ('met', met_lonlat), ('wtc', wtc_lonlat)]:\n    add_landmark_dropoff_distance(df, name, lonlat)\n")


# # 3.1 Removing Outliers
# We'll use the following ranges:
# 
# - `fare_amount`: 1 to 500
# - `longitudes`: -75 to -72
# - `latitudes`: 40 to 42
# - `passenger_count`: 1 to 6

# In[30]:


def remove_outliers(df):
    return df[(df['fare_amount'] >= 1.) & 
              (df['fare_amount'] <= 500.) &
              (df['pickup_longitude'] >= -75) & 
              (df['pickup_longitude'] <= -72) & 
              (df['dropoff_longitude'] >= -75) & 
              (df['dropoff_longitude'] <= -72) & 
              (df['pickup_latitude'] >= 40) & 
              (df['pickup_latitude'] <= 42) & 
              (df['dropoff_latitude'] >=40) & 
              (df['dropoff_latitude'] <= 42) & 
              (df['passenger_count'] >= 1) & 
              (df['passenger_count'] <= 6)]


# In[31]:


get_ipython().run_cell_magic('time', '', 'df = remove_outliers(df)\n')


# # 2. **EDA** cont...

# I wanted to see relationship between -ve fare value with distance thus decided to perform feature engineering before hand 

# ### Why fare has min value as -ve and max value = infinite

# In[32]:


## Checking for fare value to be -ve
df[df['fare_amount']<0]


# There are 463 incidents with fare less than 0, I assume they might have used some **coupons** or might carry from past when they paid in surplus... 
# But since we don't have customer i.d. or cab id we can't infer these

# In[33]:


## Checking for fare value to be greater than 500
df[df['fare_amount']>500]


# why there are log and lat with 0,0

# In[34]:


df[df['pickup_longitude']==0]


# There are 210749 rows with log lat = 0,0

# In[35]:


## Lets check for distance = 0
df[df['trip_distance']==0]


# There are 210749 rows with log lat = 0,0
# and 200326 rows with 0 trip distance 

# In[36]:


df[df['passenger_count']>6]


# In[37]:


df.pickup_datetime.min(), df.pickup_datetime.max()


# In[38]:


get_ipython().run_cell_magic('time', '', "df = df.drop('pickup_datetime', axis=1)\ndf.head(2)\n")


# In[39]:


df.hist(figsize=(22,21), bins=20);


# # Ask & answer questions about the dataset: 
# 
# 1. What is the busiest day of the week?
# 2. What is the busiest time of the day?
# 3. In which month are fares the highest?
# 4. Which pickup locations have the highest fares?
# 5. Which drop locations have the highest fares?
# 6. What is the average ride distance?
# 
# EDA + asking questions will help you develop a deeper understand of the data and give you ideas for feature engineering.

# In[40]:


# 1. What is the busiest day of the week?
df.weekday.mode()


# In[41]:


# 2. What is the busiest time of the day?
df.pickup_datetime_hour.mode()


# In[42]:


# 3. In which month are fares the highest? >>> winters have high fare and Jan has highest fare
df_fare= df.sort_values(ascending=False, by= 'fare_amount').head(50)
df_fare.month.hist();


# In[43]:


# 4. Which pickup locations have the highest fares?
sns.scatterplot(x='pickup_longitude', y= 'pickup_latitude', hue='fare_amount',data=df_fare);


# In[44]:


# 5. Which drop locations have the highest fares?
sns.scatterplot(x='dropoff_longitude', y= 'dropoff_latitude',hue='fare_amount',data=df_fare);


# In[45]:


# 6. What is the average ride distance?
df.trip_distance.mean()


# In[46]:


### Plotting log , lat for pickup
sns.scatterplot(x='pickup_longitude', y= 'pickup_latitude', data=df )


# In[47]:


### Plotting log , lat for pickup
#sns.scatterplot(x='dropoff_longitude', y= 'dropoff_latitude', data=df , hue='fare_amount')


# ### Just wanna check if this pattern is present in test dataset.

# In[48]:


df_test.hist(figsize=(8,7), bins=20);


# In[49]:


df_test[df_test['pickup_longitude']==0]


# In[50]:


df_test[df_test['passenger_count']>6]


# In[51]:


df_test.pickup_datetime.min(), df_test.pickup_datetime.max()


# Fortunatly this doesn't exit in test dataset so we can **remove** these data from training set

# ## 4. Prepare Dataset for Training
# 
# - Split Training & Validation Set
# - Fill/Remove Missing Values
# - Extract Inputs & Outputs
#    - Training
#    - Validation
#    - Test

# In[52]:


get_ipython().run_cell_magic('time', '', 'df.corr()\n')


# * I was curious if distance and log, lat had any correlation between them
# * I also want to check for correlation between extracts of time
# * `Fare` had a correlation of **0.819645** with `trip distance`

# In[53]:


df.describe()


# In[54]:


import seaborn as sns


# In[55]:


#%%time
#sns.lineplot(y='fare_amount', x='pickup_datetime', data= df_original)


# ### Split Training & Validation Set
# 
# We'll set aside 20% of the training data as the validation set, to evaluate the models we train on previously unseen data. 
# 
# Since the test set and training set have the same date ranges, we can pick a random 20% fraction.

# In[56]:


from sklearn.model_selection import train_test_split


# In[57]:


train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)


# In[58]:


len(train_df), len(val_df)


# ### Extract Inputs and Outputs

# In[59]:


df.columns


# In[60]:


input_cols = ['pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'passenger_count',
       'trip_distance', 'year', 'month', 'day', 'weekday',
       'pickup_datetime_hour', 'jfk_drop_distance', 'lga_drop_distance',
       'ewr_drop_distance', 'met_drop_distance', 'wtc_drop_distance']
target_col = 'fare_amount'


# ## Training

# In[61]:


train_inputs = train_df[input_cols]
train_targets = train_df[target_col]
train_inputs.head(3)


# In[62]:


train_targets.head(3)


# ## Validation

# In[63]:


val_inputs = val_df[input_cols]
val_targets = val_df[target_col]


# In[64]:


display(val_inputs.head(3))
display(val_targets.head(3))


# ## Test

# In[65]:


### Feature enigeerning on Test dataset
add_dateparts(df_test, 'pickup_datetime')
add_trip_distance(df_test)

for name, lonlat in [('jfk', jfk_lonlat), ('lga', lga_lonlat), ('ewr', ewr_lonlat), ('met', met_lonlat), ('wtc', wtc_lonlat)]:
    add_landmark_dropoff_distance(df_test, name, lonlat)
df_test.head(2)


# In[66]:


test_inputs = df_test[input_cols]
test_inputs.head(3)


# # 5 Modeling

# ## 5.1. Train Hardcoded & Baseline Models
# 
# - Hardcoded model: always predict average fare
# - Baseline model: Linear regression 
# 
# For evaluation the dataset uses RMSE error: 
# https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/overview/evaluation

# ### Train & Evaluate Hardcoded Model
# 
# general approach is to create a simple model that always predicts the average.
# But we will use **linear regression **

# In[67]:


from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


# In[68]:


#%%time
linreg_model = LinearRegression()
linreg_model.fit(train_inputs, train_targets)
train_preds = linreg_model.predict(train_inputs)
val_preds = linreg_model.predict(val_inputs)
train_rmse = mean_squared_error(train_targets, train_preds, squared=False)
val_rmse = mean_squared_error(val_targets, val_preds, squared=False)
print('RMSE Score on Validation data',val_rmse)
print('RMSE Score on Validation data',train_rmse)


# * Rmse = 5.15384799403991 mean our prediction is off by 5.153 per prediction which is not good as **fare.median is 8.5**
# * our base model isn't overfitting as validation score is similar to training set

# In[69]:


df.fare_amount.median()


# In[70]:


from sklearn.metrics import r2_score
r2_train= r2_score(train_targets, train_preds)
r2_val = r2_score(val_targets, val_preds)
print('R2 Score on Validation data',r2_val)
print('R2 Score on Validation data',r2_train)


# ## 5.2 Train & Evaluate Different Models
# 
# We'll train each of the following & submit predictions to Kaggle:
# 
# - Ridge Regression
# - Random Forests
# - Gradient Boosting

# In[71]:


def evaluate(model):
    train_preds = model.predict(train_inputs)
    train_rmse = mean_squared_error(train_targets, train_preds, squared=False)
    val_preds = model.predict(val_inputs)
    val_rmse = mean_squared_error(val_targets, val_preds, squared=False)
    return train_rmse, val_rmse, train_preds, val_preds


# In[72]:


def predict_and_submit(model, fname):
    test_preds = model.predict(test_inputs)
    sub_df = pd.read_csv(data_dir+'/sample_submission.csv')
    sub_df['fare_amount'] = test_preds
    sub_df.to_csv(fname, index=None)
    return sub_df


# ### Ridge Regression

# In[73]:


from sklearn.linear_model import Ridge


# In[74]:


model1 = Ridge(random_state=42)


# In[75]:


get_ipython().run_cell_magic('time', '', 'model1.fit(train_inputs, train_targets)\n')


# In[76]:


evaluate(model1)


# Time taken by model : Wall time: 3.18 s
# * Model is not overfitting as both RMSE score is 5.138
# * This mean fare prediction is off by $ 5.138 which is `similar to Linear Regression` 
# 

# In[77]:


predict_and_submit(model1, 'ridge_submission.csv')


# In[78]:


predict_and_submit(linreg_model, 'Linear_submission.csv')


# ### iii) Random Forest
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

# In[79]:


from sklearn.ensemble import RandomForestRegressor


# In[80]:


get_ipython().run_cell_magic('time', '', 'model2 = RandomForestRegressor(max_depth=10, n_jobs=-1, random_state=7, n_estimators=50)\nmodel2.fit(train_inputs, train_targets)\n')


# In[81]:


evaluate(model2)


# RF Model Execution time :: Wall time: 41min 6s   :: CPU times: user 2h 37min 31s  
# Wow Random Forest is giving so accurate results with RMSE score of **0.01** and **0.0131** which means prediction is off by few cents and model is not overfitting the data

# In[82]:


predict_and_submit(model2, 'rf_submission.csv')


# ### iv) XGradient Boosting
# 
# https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn

# In[83]:


from xgboost import XGBRegressor


# In[84]:


get_ipython().run_cell_magic('time', '', "model3 = XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror')\nmodel3.fit(train_inputs, train_targets)\n")


# In[85]:


evaluate(model3)


# Time taken to execute XGBoost :: Wall time: 41min 23s :: CPU times: user 2h 28min 42s  
# XGBoost performed worse than Random forest it has a std of 1.2 to 1.35 usd per prediction

# In[86]:


predict_and_submit(model3, 'xgb_submission.csv')


# ## 8. Tune Hyperparmeters
# 
# https://towardsdatascience.com/mastering-xgboost-2eb6bce6bc76
# 
# 
# We'll train parameters for the XGBoost model. Here’s a strategy for tuning hyperparameters:
# 
# - Tune the most important/impactful hyperparameter first e.g. n_estimators
# 
# - With the best value of the first hyperparameter, tune the next most impactful hyperparameter
# 
# - And so on, keep training the next most impactful parameters with the best values for previous parameters...
# 
# - Then, go back to the top and further tune each parameter again for further marginal gains
# 
# - Hyperparameter tuning is more art than science, unfortunately. Try to get a feel for how the parameters interact with each other based on your understanding of the parameter…
# 
# Let's define a helper function for trying different hyperparameters.

# In[87]:


import matplotlib.pyplot as plt

def test_params(ModelClass, **params):
    """Trains a model with the given parameters and returns training & validation RMSE"""
    model = ModelClass(**params).fit(train_inputs, train_targets)
    train_rmse = mean_squared_error(model.predict(train_inputs), train_targets, squared=False)
    val_rmse = mean_squared_error(model.predict(val_inputs), val_targets, squared=False)
    return train_rmse, val_rmse

def test_param_and_plot(ModelClass, param_name, param_values, **other_params):
    """Trains multiple models by varying the value of param_name according to param_values"""
    train_errors, val_errors = [], [] 
    for value in param_values:
        params = dict(other_params)
        params[param_name] = value
        train_rmse, val_rmse = test_params(ModelClass, **params)
        train_errors.append(train_rmse)
        val_errors.append(val_rmse)
    
    plt.figure(figsize=(10,6))
    plt.title('Overfitting curve: ' + param_name)
    plt.plot(param_values, train_errors, 'b-o')
    plt.plot(param_values, val_errors, 'r-o')
    plt.xlabel(param_name)
    plt.ylabel('RMSE')
    plt.legend(['Training', 'Validation'])


# In[88]:


best_params = {
    'random_state': 7,
    'n_jobs': -1,
    'objective': 'reg:squarederror'
}


# In[ ]:


get_ipython().run_cell_magic('time', '', "### No of trees\ntest_param_and_plot(XGBRegressor, 'n_estimators', [100, 250, 500], **best_params)\n")


# Seems like 500 estimators has the lowest validation loss. However, it also takes a long time. Let's stick with 250 for now.

# In[ ]:


best_params['n_estimators'] = 250


# In[ ]:


get_ipython().run_cell_magic('time', '', "#### Max Depth\ntest_param_and_plot(XGBRegressor, 'max_depth', [3, 4, 5], **best_params)\n")


# In[ ]:


best_params['max_depth'] = 5


# In[ ]:


get_ipython().run_cell_magic('time', '', "#### Learning Rate\ntest_param_and_plot(XGBRegressor, 'learning_rate', [0.05, 0.1, 0.25], **best_params)\n")


# In[ ]:


best_params['learning_rate'] = 0.25


# In[ ]:


# Final Model Creation
xgb_model_final = XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42,
                               n_estimators=500, max_depth=5, learning_rate=0.1, 
                               subsample=0.8, colsample_bytree=0.8)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'xgb_model_final.fit(train_inputs, train_targets)\n')


# In[ ]:


evaluate(xgb_model_final)


# In[ ]:


predict_and_submit(xgb_model_final, 'xgb_tuned_submission.csv')


# In[ ]:





# In[ ]:




