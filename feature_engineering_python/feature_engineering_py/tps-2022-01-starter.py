#!/usr/bin/env python
# coding: utf-8

# # Intro
# Welcome to the [Tabular Playground Series - Jan 2022](https://www.kaggle.com/c/tabular-playground-series-jan-2022) competition.
# 
# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/33101/logos/header.png)
# 
# We use the dates of the holidays in Finland, Norway and Sweden from here:
# * Source [Finland](https://date.nager.at/PublicHoliday/Country/FI), 
# * Source [Norway](https://date.nager.at/PublicHoliday/Country/NO), 
# * Source [Sweden](https://date.nager.at/PublicHoliday/Country/SE). 
# 
# The file with all holidays you can find [here](https://www.kaggle.com/drcapa/holidays-finland-norway-sweden-20152019). 
# 
# **Table of content:**
# 1. [Exploratory Data Analysis](#EDA)
# 2. [Plot Time Series](#PTS)
# 3. [Time Series Analysis](#TSA)
# 4. [Feature Engineering](#FE)
# 5. [Define Model](#DE)
# 6. [Export](#Export)
# 
# 
# <font size="4"><span style="color: royalblue;">Please vote the notebook up if it helps you. Feel free to leave a comment above the notebook. Thank you. </span></font>

# # Libraries

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer


# # Path

# In[2]:


path = '../input/tabular-playground-series-jan-2022/'
path_holidays = '../input/holidays-finland-norway-sweden-20152019/'

print('Compedition data:', os.listdir(path))
print('Holiday data:', os.listdir(path_holidays))


# # Load Data

# In[3]:


train_data = pd.read_csv(path+'train.csv')
test_data = pd.read_csv(path+'test.csv')
samp_subm = pd.read_csv(path+'sample_submission.csv')

holidays = pd.read_csv(path_holidays+'Holidays_Finland_Norway_Sweden_2015-2019.csv')


# Set feature date to datetime format:

# In[4]:


train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])

holidays['Date'] = pd.to_datetime(holidays['Date'])


# # Overview
# In this compedition we have to predict number of sales of products in the kaggle stores.

# In[5]:


print('Number train samples:', len(train_data.index))
print('Number test samples:', len(test_data.index))
print('Number holiday samples:', len(holidays))


# Train data:

# In[6]:


train_data.head()


# Holiday data:

# In[7]:


holidays.head()


# # Exploratory Data Analysis <a name="EDA"></a>

# Countries:

# In[8]:


country_list = list(train_data['country'].unique())
train_data['country'].value_counts()


# Stores:

# In[9]:


store_list = list(train_data['store'].unique())
train_data['store'].value_counts()


# Products

# In[10]:


product_list = ['Kaggle Hat', 'Kaggle Mug', 'Kaggle Sticker']
train_data['product'].value_counts()


# # Plot Time Series <a name="PTS"></a>
# We plot the target value **num_sold** countrywise for every shop and every product

# In[11]:


def plot_timeseries(country):
    fig, axs = plt.subplots(2, 3, figsize=(30, 10))
    fig.subplots_adjust(hspace = .2, wspace=.2)
    for row in range(len(store_list)):
        store = store_list[row]
        for col in range(len(product_list)):
            product = product_list[col]
            temp = train_data[(train_data['country']==country) &
                              (train_data['store']==store) &
                              (train_data['product']==product)]
            temp.index = temp['date']
            axs[row][col].plot(temp.index, temp['num_sold'])
            axs[row][col].set_title('Store:'+store+', Product:'+product)
            axs[row][col].grid()


# ## Finland

# In[12]:


plot_timeseries(country='Finland')


# ## Norway

# In[13]:


plot_timeseries(country='Norway')


# ## Sweden

# In[14]:


plot_timeseries(country='Sweden')


# **Observation:** The products have an individually structure. It could be helpful to use different models for different products.

# # Time Series Analysis <a name="TSA"></a>

# In[15]:


country = 'Finland'
store = 'KaggleMart'
product = 'Kaggle Mug'


# In[16]:


df_temp = train_data[(train_data['country']==country) &
                     (train_data['store']==store) &
                     (train_data['product']==product)]

df_temp.index = df_temp['date']


# In[17]:


decompose = seasonal_decompose(df_temp['num_sold'], model="multiplicative")


# In[18]:


def plot_timeseries(decompose):
    fig, axs = plt.subplots(1, 4, figsize=(40, 10))
    fig.subplots_adjust(hspace = .2, wspace=.2)
    part = ['observed', 'trend', 'saisonal', 'resid']
    axs = axs.ravel()
    axs[0].plot(decompose.observed.index, decompose.observed.values)
    axs[1].plot(decompose.trend.index, decompose.trend.values)
    axs[2].plot(decompose.seasonal.index, decompose.seasonal.values)
    axs[3].plot(decompose.resid.index, decompose.resid.values)
    
    for i in range(4):
        axs[i].set_title(part[i])
        axs[i].grid()


# In[19]:


plot_timeseries(decompose)


# # Feature Engineering <a name="FE"></a>
# We create new features based on the date.

# Create feature year:

# In[20]:


def extract_year(s):
    return s.year

train_data['year'] = train_data['date'].apply(extract_year)
test_data['year'] = test_data['date'].apply(extract_year)

year_list = list(train_data['year'].unique()) + list(test_data['year'].unique())


# Create feature month:

# In[21]:


def extract_month_name(s):
    return s.month_name()

train_data['month_name'] = train_data['date'].apply(extract_month_name)
test_data['month_name'] = test_data['date'].apply(extract_month_name)

month_list = list(train_data['month_name'].unique())

def extract_month(s):
    return s.month

train_data['month'] = train_data['date'].apply(extract_month)
test_data['month'] = test_data['date'].apply(extract_month)


# Create feature day of week:

# In[22]:


def extract_day_name(s):
    return s.day_name()

train_data['day_name'] = train_data['date'].apply(extract_day_name)
test_data['day_name'] = test_data['date'].apply(extract_day_name)

day_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def extract_day(s):
    return s.dayofweek

train_data['day'] = train_data['date'].apply(extract_day)
test_data['day'] = test_data['date'].apply(extract_day)


# Weekend:

# In[23]:


def extract_weekend(s):
    if s == 'Saturday' or s == 'Sunday':
        return 1
    else:
        return 0
    
train_data['weekend'] = train_data['day'].apply(extract_weekend)
test_data['weekend'] = test_data['day'].apply(extract_weekend)


# Holidays:

# In[24]:


rename_dict = {'Date': 'date', 'Country': 'country'}
holidays = holidays.rename(rename_dict, axis=1)

train_data = pd.merge(train_data, holidays, how='left', on=['date', 'country'])
test_data = pd.merge(test_data, holidays, how='left', on=['date', 'country'])


# In[25]:


def extract_holiday(s):
    if type(s)==float:
        return 0
    else:
        return 1
    
train_data['holiday'] = train_data['Name'].apply(extract_holiday)
test_data['holiday'] = test_data['Name'].apply(extract_holiday)


# Now we encode the categorical features. Therefor we recommend this [notebook](https://www.kaggle.com/drcapa/categorical-feature-engineering-xgb).

# In[26]:


train_data[pd.get_dummies(train_data['country']).columns] = pd.get_dummies(train_data['country'])
train_data[pd.get_dummies(train_data['store']).columns] = pd.get_dummies(train_data['store'])
train_data[pd.get_dummies(train_data['product']).columns] = pd.get_dummies(train_data['product'])
train_data[pd.get_dummies(train_data['day_name']).columns] = pd.get_dummies(train_data['day_name'])
train_data[pd.get_dummies(train_data['month_name']).columns] = pd.get_dummies(train_data['month_name'])
train_data[pd.get_dummies(train_data['year']).columns] = pd.get_dummies(train_data['year'])
train_data[2019] = 0

test_data[pd.get_dummies(test_data['country']).columns] = pd.get_dummies(test_data['country'])
test_data[pd.get_dummies(test_data['store']).columns] = pd.get_dummies(test_data['store'])
test_data[pd.get_dummies(test_data['product']).columns] = pd.get_dummies(test_data['product'])
test_data[pd.get_dummies(test_data['day_name']).columns] = pd.get_dummies(test_data['day_name'])
test_data[pd.get_dummies(test_data['month_name']).columns] = pd.get_dummies(test_data['month_name'])
test_data[pd.get_dummies(test_data['year']).columns] = pd.get_dummies(test_data['year'])
test_data[[2015, 2016, 2017, 2018]]=0


# In[27]:


features_cyclic = ['day', 'month']
for feature in features_cyclic:
    train_data[feature+'_sin'] = np.sin((2*np.pi*train_data[feature])/max(train_data[feature]))
    train_data[feature+'_cos'] = np.cos((2*np.pi*train_data[feature])/max(train_data[feature]))
    test_data[feature+'_sin'] = np.sin((2*np.pi*test_data[feature])/max(test_data[feature]))
    test_data[feature+'_cos'] = np.cos((2*np.pi*test_data[feature])/max(test_data[feature]))
    
feature_cyc_list = ['day_sin', 'day_cos', 'month_sin', 'month_cos']


# # Define Model <a name="DM"></a>

# The [symmetric mean absolute percentage error](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error) (smape) is used to measure the predictive accuracy of the submission results:

# In[28]:


def smape_error(y_true, y_pred):
    smape = 1/len(y_true) * np.sum(2 * np.abs(y_pred-y_true) / (np.abs(y_true) + np.abs(y_pred))*100)
    return smape

score = make_scorer(smape_error, greater_is_better=False)


# Because of the different structure of the products we use different models to train and predict the data.

# In[29]:


def get_params(X, y):
    params = {'objective': ['reg:squarederror'],
              'max_depth': [4, 5, 6, 7],
              'learning_rate': [0.1],
              'n_estimators': [50, 100, 125, 150],
              'colsample_bytree': [0.8, 0.9, 1.0]}
    
    #params = {'objective': ['reg:squarederror'],
    #          'max_depth': [6],
    #          'learning_rate': [0.1],
    #          'n_estimators': [50, 75, 100],
    #          'colsample_bytree': [0.8, 0.9]}

    xgb = XGBRegressor(seed = 2022)

    clf = GridSearchCV(cv=10,
                       estimator=xgb, 
                       param_grid=params,
                       scoring=score, 
                       verbose=0,
                       n_jobs=4)
    
    clf.fit(X, y)
    print('Best SMAPE Score:', round(-(clf.best_score_), 2))
    print('Best Params:', clf.best_params_)
    print('\n')
    return clf.best_params_


# In[30]:


for country in country_list:
    for store in store_list:
        for product in product_list:
            print(country, ' - ', store, ' - ', product)
            train_temp = train_data[(train_data[country]==1)&
                                    (train_data[store]==1)&
                                    (train_data[product]==1)]
            test_temp = test_data[(test_data[country]==1)&
                                  (test_data[store]==1)&
                                  (test_data[product]==1)]
            
            features = ['year', 'month', 'day', 'weekend', 'holiday']+day_list+month_list+year_list+feature_cyc_list
            
            X = train_temp[features]
            y = train_temp['num_sold']
            X_test = test_temp[features]
            
            #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state=2022)
            X_train = X
            y_train = y
            params = get_params(X_train, y_train)
            
            #model = XGBRegressor(objective='reg:squarederror',
            #                     n_estimators=50,
            #                     learning_rate=0.1,
            #                     colsample_bytree=0.8,
            #                     max_depth=5)
            model = XGBRegressor(**params)
            #model = LinearRegression(normalize=False)
            model.fit(X_train, y_train)
            #y_val_pred = model.predict(X_val)
            #print('SMAPE:', round(smape_error(y_val, y_val_pred), 2))
            
            y_test = model.predict(X_test)
            samp_subm.loc[X_test.index, 'num_sold'] = y_test
        


# # Export <a name="Export"></a>

# In[31]:


samp_subm.to_csv('submission.csv', index=False)

