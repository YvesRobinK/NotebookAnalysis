#!/usr/bin/env python
# coding: utf-8

# # Intro Bike Sharing Demand Competition
# Source: https://www.kaggle.com/c/bike-sharing-demand
# 
# This notebook is a starter code for all beginners and easy to understand. We will give an introduction to analysis and feature engineering.<br> 
# Therefore we focus on
# * a simple analysis of the data,
# * create new features,
# * encoding and
# * scale data.
# 
# We use categorical feature encoding techniques, compare <br>
# https://www.kaggle.com/drcapa/categorical-feature-encoding-challenge-xgb <br>
# 
# For this competition, the training set is comprised of the first 19 days of each month, while the test set is the 20th to the end of the month. You must predict the total count of bikes rented during each hour covered by the test set, using only information available prior to the rental period.

# # Load Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import calendar


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# In[3]:


from keras.models import Sequential
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import RMSprop,Adam


# In[4]:


import warnings
warnings.filterwarnings("ignore")


# # Input path

# In[5]:


path_in = '../input/'
print(os.listdir(path_in))


# # Load Data

# In[6]:


train_data = pd.read_csv(path_in + 'train.csv', parse_dates = ['datetime'],
                         index_col='datetime', infer_datetime_format=True)
test_data = pd.read_csv(path_in + 'test.csv', parse_dates = ['datetime'],
                        index_col='datetime', infer_datetime_format=True)
samp_subm = pd.read_csv(path_in+'sampleSubmission.csv', parse_dates = ['datetime'],
                        index_col='datetime', infer_datetime_format=True)


# # Functions

# In[7]:


def plot_bar(data, feature):
    """ Plot distribution """
    
    fig = plt.figure(figsize=(5,3))
    sns.barplot(x=feature, y='count', data=data, palette='Set3',orient='v')


# In[8]:


def plot_timeseries(data, feature):
    """ Plot timeseries """
    
    fig = plt.figure(figsize=(16,9))
    plt.plot(data.index, data[feature])
    plt.title(feature)
    plt.grid()


# In[9]:


def plot_timeseries_train_and_predict(train, predict, year, month):
    """ Compare train and predict data for a month """
    
    start_date = datetime.datetime(year, month, 1, 0, 0, 0)
    last_day_of_month = calendar.monthrange(year, month)[1]
    end_date = datetime.datetime(year, month, last_day_of_month, 23, 0, 0)
    
    fig = plt.figure(figsize=(16,9))
    plt.plot(train[start_date: end_date].index, train.loc[start_date:end_date, 'count'], 'b', label = 'train')
    plt.plot(predict[start_date: end_date].index, predict.loc[start_date:end_date, 'count'], 'r', label = 'predict')
    plt.title('Train and Predict')
    plt.legend()
    plt.grid()


# In[10]:


def rmse(y_true, y_pred):
    """ root_mean_squared_error """
    
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# # EDA
# You are provided hourly rental data spanning two years. For this competition, the training set is comprised of the first 19 days of each month, while the test set is the 20th to the end of the month. You must predict the total count of bikes rented during each hour covered by the test set, using only information available prior to the rental period.

# In[11]:


# Parameters
num_months_per_year = 12
year_list = [2011, 2012]


# In[12]:


train_data.head()


# In[13]:


test_data.head()


# In[14]:


samp_subm.head()


# # Missing Timestamps
# There are losts of missing hours in the train dataset. We expect $ 2 years*12 months*19 days *24 hours = 10944 timesteps$. We count 10886 timesteps so there are 58 missing. Every month in the train data set hast 456 timestamps.  

# Fill missing timestamps:

# In[15]:


train_data_temp = pd.DataFrame(columns=train_data.columns)

for year in year_list:
    for month in range(num_months_per_year):
        start_date = datetime.datetime(year, month+1, 1, 0, 0, 0)
        end_date = datetime.datetime(year, month+1, 19, 23, 0, 0)
        # Fill missing timestamps
        temp = train_data[start_date:end_date].resample('H').asfreq()
        # Handle missing values
        features_fill_zero = ['casual', 'registered', 'count']
        temp[features_fill_zero] = temp[features_fill_zero].fillna(0)
        features_fill_bbfil = ['season', 'holiday', 'workingday', 'weather']
        temp[features_fill_bbfil] = temp[features_fill_bbfil].fillna(method='bfill')
        features_fill_linear = ['temp', 'atemp', 'humidity', 'windspeed']
        temp[features_fill_linear] = temp[features_fill_linear].interpolate(method='linear')
        
        train_data_temp = train_data_temp.append(temp)
        
train_data = train_data_temp


# Test Data:

# In[16]:


test_data_temp = pd.DataFrame(columns=test_data.columns)
for year in year_list:
    for month in range(num_months_per_year):
        start_date = datetime.datetime(year, month+1, 20, 0, 0, 0)
        last_day_of_month = calendar.monthrange(year,month+1)[1]
        end_date = datetime.datetime(year, month+1, last_day_of_month, 23, 0, 0)
        # Fill missing timestamps
        temp = test_data[start_date:end_date].resample('H').asfreq()
        # Handle missing values
        features_fill_bbfil = ['season', 'holiday', 'workingday', 'weather']
        temp[features_fill_bbfil] = temp[features_fill_bbfil].fillna(method='bfill')
        features_fill_linear = ['temp', 'atemp', 'humidity', 'windspeed']
        temp[features_fill_linear] = temp[features_fill_linear].interpolate(method='linear')
        
        test_data_temp = test_data_temp.append(temp)
        
test_data = test_data_temp


# The datetime and the seasons are cyclic features. So we can use a cyclic encoding for it.

# # Create new features
# Based on the datetime we create new features for the month, the weekday the hour and the year. These are also cyclic features.

# In[17]:


train_data['weekday'] = train_data.index.weekday
train_data['hour'] = train_data.index.hour
# train_data['month'] = train_data.index.month
# train_data['year'] = train_data.index.year
test_data['weekday'] = test_data.index.weekday
test_data['hour'] = test_data.index.hour
# test_data['month'] = test_data.index.month
# test_data['year'] = test_data.index.year


# ## Count Monthly Mean

# In[18]:


for year in year_list:
    for month in range(num_months_per_year): 
        start_date = datetime.datetime(year, month+1, 1, 0, 0, 0)
        end_date = datetime.datetime(year, month+1, 19, 23, 0, 0)
        count_mean = train_data[start_date:end_date]['count'].mean()
        train_data.loc[start_date:end_date, 'count_mean'] = count_mean
        
        start_date = datetime.datetime(year, month+1, 20, 0, 0, 0)
        last_day_of_month = calendar.monthrange(year,month+1)[1]
        end_date = datetime.datetime(year, month+1, last_day_of_month, 23, 0, 0)
        test_data.loc[start_date:end_date, 'count_mean'] = count_mean


# In[19]:


test_data.head()


# ## Feature Season

# In[20]:


plot_bar(train_data, 'season')
plt.grid()


# ## Feature Weekday

# In[21]:


plot_bar(train_data, 'weekday')
plt.grid()


# ## Feature Hour
# Add new feature hour_group by group of hours.

# In[22]:


plot_bar(train_data, 'hour')
plt.grid()


# In[23]:


def hour_group(s):
    if((0<=s) & (s<=6)):
        return 1
    elif((s==7) | (s==9)):
        return 2
    elif((s==8) | (s==16) | (s==19)):
        return 3
    elif((10<=s) & (s<=15)):
        return 4
    elif((s==17) | (s==18)):
        return 5
    elif(20<=s):
        return 6


# In[24]:


train_data['hour_group'] = train_data['hour'].apply(hour_group)
test_data['hour_group'] = test_data['hour'].apply(hour_group)


# # Encoding 
# ## Cyclic features

# In[25]:


# features_cyc = ['hour', 'weekday']
# for feature in features_cyc:
#     train_data[feature+'_sin'] = np.sin((2*np.pi*train_data[feature])/max(train_data[feature]))
#     train_data[feature+'_cos'] = np.cos((2*np.pi*train_data[feature])/max(train_data[feature]))
#     test_data[feature+'_sin'] = np.sin((2*np.pi*test_data[feature])/max(test_data[feature]))
#     test_data[feature+'_cos'] = np.cos((2*np.pi*test_data[feature])/max(test_data[feature]))
# train_data = train_data.drop(features_cyc, axis=1)
# test_data = test_data.drop(features_cyc, axis=1)


# ## One Hot Encoding for categorical variables

# In[26]:


# features_one_hot = ['weekday', 'weather']
# train_data[features_one_hot] = train_data[features_one_hot].astype(int).astype(str)
# test_data[features_one_hot] = test_data[features_one_hot].astype(int).astype(str)


# In[27]:


# train_data = pd.get_dummies(train_data)
# test_data = pd.get_dummies(test_data)


# # Scale Data

# In[28]:


scale_features = ['temp', 'atemp', 'humidity', 'hour', 'windspeed']


# In[29]:


# scaler = MinMaxScaler()
# train_data[scale_features] = scaler.fit_transform(train_data[scale_features])
# test_data[scale_features] = scaler.transform(test_data[scale_features])


# # Predict Monthly

# In[30]:


# Features
feature_list = ['holiday', 'workingday', 'weather', 'temp', 'atemp',
                'humidity', 'windspeed', 'hour_group',
                'hour', 'weekday', 'count_mean']
no_features = ['casual', 'registered']#, 'count']


# # Predict One Month

# In[31]:


train_data['workingday'].value_counts()


# In[32]:


# Parameters
lookback = 3
horizon = 2
month = 1
year = 2011


# In[33]:


def create_data(data, lookback, horizon, start, end, kind):
    X, y = [], []
    
    if kind == 'train':
        start_shifted = start
    else:
        start_shifted = start - datetime.timedelta(hours=lookback)
        
    temp = data[start_shifted:end].copy()
    temp.index = range(len(temp.index))
    
    start_ix = lookback
    
    n_samples = int((len(temp.index)-lookback)/horizon)
    
    for i in range(n_samples):
        end_ix = start_ix+horizon
        seq_X = temp[(start_ix-lookback):start_ix][data.columns.difference(no_features)].values
        seq_y = temp[start_ix:end_ix]['count'].values
        
        X.append(seq_X)
        y.append(seq_y)
        
        start_ix = end_ix
    
    return np.array(X), np.array(y)


# In[34]:


# Parameters
lookback = 24
horizon = 1
month = 12
year = 2012

# define time range
start_train = datetime.datetime(year, month, 1, 0, 0, 0)
end_train = datetime.datetime(year, month, 19, 23, 0, 0)
start_test = datetime.datetime(year, month, 20, 0, 0, 0)
last_day_of_month = calendar.monthrange(year,month)[1]
end_test = datetime.datetime(year, month, last_day_of_month, 23, 0, 0)


# concate train and test data
data = pd.concat([train_data[start_train:end_train],
                  test_data[start_test:end_test]])

# scale data
features_no_scale = ['season', 'holiday', 'workingday', 'count_mean']
data_scaled = data.copy()
data_mean = data_scaled.mean(axis=0)
data_scaled[data.columns.difference(features_no_scale)] -= data_mean
data_std = data_scaled.std(axis=0)
data_scaled[data.columns.difference(features_no_scale)] /= data_std

# create train and test data
X_train, y_train = create_data(data_scaled, lookback, horizon, start_train, end_train, 'train')
print(X_train[0])
# define model
n_steps = X_train.shape[1]
n_features = X_train.shape[2]
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(n_steps, n_features)))
model.add(SimpleRNN(64, return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(32))
model.add(Dense(horizon))

# define optimizer and compile model
optimizer = Adam(lr=1e-4)
model.compile(optimizer=optimizer, loss='mse', metrics = ['mse'])

# fit model
history = model.fit(X_train, y_train,
                    epochs=1, verbose=1)

# predict y_test
start_test_temp = start_test

n_days = last_day_of_month-19
for i in range(n_days*24+1):
    end_test_temp = start_test+datetime.timedelta(hours=i)
    #print(start_test_temp, end_test_temp)
    X_test, y_test = create_data(data_scaled, lookback, horizon, start_test_temp, end_test_temp, 'test')
    if X_test.size != 0: 
        y_test = model.predict(X_test, verbose=0)

        data_scaled.loc[start_test_temp:(end_test_temp-datetime.timedelta(hours=1)), 'count'] = y_test[0]
        start_test_temp = end_test_temp

# write in submission data        
#samp_subm[start_test:end_test]['count'] = data_scaled[start_test:end_test]['count']*data_std['count']+data_mean['count']
samp_subm.loc[start_test:end_test, 'count'] = data_scaled.loc[samp_subm[start_test:end_test].index]['count']*data_std['count']+data_mean['count']
samp_subm['count'] = np.where(samp_subm['count']<0, 0, samp_subm['count'])
samp_subm['count'] = samp_subm['count'].interpolate()


# In[35]:


X_train.shape, y_train.shape


# In[36]:


loss = history.history['loss']
acc = history.history['mse']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, 'ro', label='loss_train')
plt.plot(epochs, acc, 'b', label='accuracy_train')
plt.title('value of the loss function')
plt.xlabel('epochs')
plt.ylabel('value of the loss function')
plt.legend()
plt.grid()
plt.show()


# In[37]:


plot_timeseries_train_and_predict(train_data, samp_subm, year, month)


# # Predict All Months

# In[38]:


# parameters
lookback = 24
horizon = 1
n_months = 12
year_list = [2011, 2012]


# ## Fit Model

# In[39]:


n_samples = 0#int(19*24)-lookback
n_features = len(train_data[train_data.columns.difference(no_features)].columns)
print('features used: ', train_data[train_data.columns.difference(no_features)].columns)
X_train, y_train = np.empty(shape=(n_samples, lookback, n_features)), np.empty(shape=(n_samples, horizon))

for year in year_list:
    for month in range(1,n_months+1):
        #print('year: ', year, ' month: ', month)
        start_train = datetime.datetime(year, month, 1, 0, 0, 0)
        end_train = datetime.datetime(year, month, 19, 23, 0, 0)
        
        data = train_data[start_train:end_train]
        
        # scale data
        features_no_scale = ['season', 'holiday', 'workingday', 'count_mean']
        data_scaled = data.copy()
        data_mean = data_scaled.mean(axis=0)
        data_scaled[data.columns.difference(features_no_scale)] -= data_mean
        data_std = data_scaled.std(axis=0)
        data_scaled[data.columns.difference(features_no_scale)] /= data_std
                
        # create train and test data
        X_temp, y_temp = create_data(data_scaled, lookback, horizon, start_train, end_train, 'train')
    
        X_train = np.vstack((X_train, X_temp))
        y_train = np.vstack((y_train, y_temp))


# In[40]:


n_steps = X_train.shape[1]
n_features = X_train.shape[2]
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(n_steps, n_features), dropout=0.0, recurrent_dropout=0.2,))
model.add(LSTM(64, return_sequences=True, input_shape=(n_steps, n_features), dropout=0.0, recurrent_dropout=0.2,))
#model.add(LSTM(164, return_sequences=True, input_shape=(n_steps, n_features)))
#model.add(LSTM(164, return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(32))
model.add(Dense(horizon))

# define optimizer and compile model
optimizer = Adam(lr=1e-4)
model.compile(optimizer=optimizer, loss='mse', metrics = ['mse'])

# fit model
history = model.fit(X_train, y_train,
                    validation_split=0.1,
                    epochs=150, verbose=1)


# In[41]:


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, 'ro', label='loss_train')
plt.plot(epochs, val_loss, 'b', label='loss_val')
plt.title('value of the loss function')
plt.xlabel('epochs')
plt.ylabel('value of the loss function')
plt.legend()
plt.grid()
plt.show()


# ## Predict Test Data

# In[42]:


for year in year_list:
    for month in range(1,n_months+1):
        #print('year: ', year, ' month: ', month)
        start_train = datetime.datetime(year, month, 1, 0, 0, 0)
        end_train = datetime.datetime(year, month, 19, 23, 0, 0)
        start_test = datetime.datetime(year, month, 20, 0, 0, 0)
        last_day_of_month = calendar.monthrange(year,month)[1]
        end_test = datetime.datetime(year, month, last_day_of_month, 23, 0, 0)

        # concate train and test data
        data = pd.concat([train_data[start_train:end_train],
                          test_data[start_test:end_test]])
        
        # scale data
        features_no_scale = ['season', 'holiday', 'workingday', 'count_mean']
        data_scaled = data.copy()
        data_mean = data_scaled.mean(axis=0)
        data_scaled[data.columns.difference(features_no_scale)] -= data_mean
        data_std = data_scaled.std(axis=0)
        data_scaled[data.columns.difference(features_no_scale)] /= data_std
        
        # predict y_test
        start_test_temp = start_test

        n_days = last_day_of_month-19
        for i in range(n_days*24+1):
            end_test_temp = start_test+datetime.timedelta(hours=i)
        
            X_test, y_test = create_data(data_scaled, lookback, horizon, start_test_temp, end_test_temp, 'test')
            if X_test.size != 0: 
                y_test = model.predict(X_test, verbose=0)

                data_scaled.loc[start_test_temp:(end_test_temp-datetime.timedelta(hours=1)), 'count'] = y_test[0]
                start_test_temp = end_test_temp
        
        # write submission file
        #samp_subm.loc[start_test:end_test, 'count'] = data_scaled[start_test:end_test]['count']*data_std['count']+data_mean['count']
        samp_subm.loc[start_test:end_test, 'count'] = data_scaled.loc[samp_subm[start_test:end_test].index]['count']*data_std['count']+data_mean['count']
        samp_subm['count'] = np.where(samp_subm['count']<0, 0, samp_subm['count'])
        samp_subm['count'] = samp_subm['count'].interpolate()        


# # Generate Output

# In[43]:


output = pd.DataFrame({'datetime': samp_subm.index,
                       'count': samp_subm['count']})
output.to_csv('submission.csv', index=False)


# # Analyse Results

# In[44]:


plot_timeseries_train_and_predict(train_data, samp_subm, 2012, 12)

