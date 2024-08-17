#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In TPS Jan 2022, the aim of this competition is to predict the number of items sold by two different companies. The features that are available to us are date, country, store and product. My initialy impression is that this is a very small dataset (features and samples) and so a gradient boosted model like XGBoost could do well here.
# 
# **Note** that I made the first version of this notebook before I looked at other peoples solutions. After this I build on other peoples ideas to improve my own score and learn new techniques.
# 
# **Ver. 2 Acknowledgments:**
# * [Rounding up predictions](https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/298201#1642988) by [Carl McBride Ellis](https://www.kaggle.com/carlmcbrideellis).
# * [LGBM](https://www.kaggle.com/ambrosm/tpsjan22-06-lightgbm-quickstart/notebook) by [Ambros M](https://www.kaggle.com/ambrosm).
# * [Holidays](https://www.kaggle.com/mfedeli/tabular-playground-series-jan-2022) by [Matteo Fedeli](https://www.kaggle.com/mfedeli).
# 
# **Ver. 3 Acknowledgments:**
# * [GDP](https://www.kaggle.com/carlmcbrideellis/gdp-of-finland-norway-and-sweden-2015-2019/data) by [Carl McBride Ellis](https://www.kaggle.com/carlmcbrideellis).
# 
# **Ver. 5 Acknowledgments:**
# * [Date feat. eng.](https://www.kaggle.com/lucamassaron/kaggle-merchandise-eda-with-baseline-linear-model) by [Luca Massaron](https://www.kaggle.com/lucamassaron).
# 
# **Ver. 6 Acknowledgments:**
# * [TimesSeriesSplit + weekend feature](https://www.kaggle.com/adamwurdits/tps-01-2022-catboost-w-optuna-seed-averaging?scriptVersionId=84848139) by [Adam Wurdits](https://www.kaggle.com/adamwurdits).
# * [Unofficial holidays](https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/298990) by [Vincent Pallares](https://www.kaggle.com/vpallares).
# 
# **Ver. 8 Acknowledgments:**
# * [CPI](https://www.kaggle.com/sardorabdirayimov/consumer-price-index-20152019-nordic-countries) by [Sardor Abdirayimov](https://www.kaggle.com/sardorabdirayimov).
# * [Fourier features](https://www.kaggle.com/ryanholbrook/seasonality) from the [kaggle time series course](https://www.kaggle.com/learn/time-series).
# 
# **Ver. 9 Acknowledgments:**
# * [Geometric rounding](https://www.kaggle.com/fergusfindley/ensembling-and-rounding-techniques-comparison) by [Fergus Findley](https://www.kaggle.com/fergusfindley).
# 
# **Ver. 10 Acknowledgments:**
# * [Fourier Features](https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/301629) by [AmbrosM](https://www.kaggle.com/ambrosm).

# # Libraries

# In[1]:


# Core
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.4)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from itertools import combinations
import statistics
import time
from datetime import datetime
import matplotlib.dates as mdates

# Sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

# Models
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

# Tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks


# # Data

# **Load data**

# In[2]:


# Save to df
train_data=pd.read_csv('../input/tabular-playground-series-jan-2022/train.csv', index_col='row_id')
test_data=pd.read_csv('../input/tabular-playground-series-jan-2022/test.csv', index_col='row_id')

# Shape and preview
print('Training data df shape:',train_data.shape)
print('Test data df shape:',test_data.shape)
train_data.head()


# *Initial thoughts:*
# * Date data can be tricky to deal with but luckily here it is already converted to international date time format. This means entries have an ordering, which makes it easier to sort and plot.
# * Country, store and product features are not currently numeric. The first pre-processing step could be encoding these to categorical numeric using a one-hot scheme. We'll check below for missing values and number of unique entries to see if we can do this. 

# **Missing values**

# In[3]:


print('Number of missing values in training set:',train_data.isna().sum().sum())
print('')
print('Number of missing values in test set:',test_data.isna().sum().sum())


# There are no missing values. That's very nice of kaggle.

# **Cardinality of features**

# In[4]:


print('Training cardinalities: \n', train_data.nunique())
print('')
print('Test cardinalities: \n', test_data.nunique())


# The cardinalities of country, store and product are very small and so one-hot encoding this categorical data is justified. It would now be good to see what the range of date values is to understand the timescale we are dealing with.

# **Timeframe**

# In[5]:


print('Training data:')
print('Min date', train_data['date'].min())
print('Max date', train_data['date'].max())
print('')
print('Test data:')
print('Min date', test_data['date'].min())
print('Max date', test_data['date'].max())


# Ok, so this tells us that the training data spans 3 years from 2015 to 2018 and the test data picks up from where we left off and spans a further 1 year, i.e. 2019. 

# In[6]:


# Convert date to datetime
train_data.date=pd.to_datetime(train_data.date)
test_data.date=pd.to_datetime(test_data.date)


# # EDA

# **Store sales**

# Let's begin by plotting the target data.  

# In[7]:


# Figure
plt.figure(figsize=(12,5))

# Groupby
aa=train_data.groupby(['date','store']).agg(num_sold=('num_sold','sum'))

# Lineplot
sns.lineplot(data=aa, x='date', y='num_sold', hue='store')

# Aesthetics
plt.title('num_sold by store')


# *Observations:*
# * Kaggle Rama is consistently selling more products than Kaggle Mart. 
# * The number of products sold for both companies oscillates depending on the time of year (season) and fluctuates rapidly (this is probably due to weekday vs weekend sales).
# * There are big spikes towards the end of each year (likely due to christmas) and also some other smaller seasonal spikes (perhaps easter holidays etc).

# **Store sales by country**

# This is to look at whether the stores sell more products in certain countries or not.

# In[8]:


# Subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Groupby
KR=train_data[train_data.store=='KaggleRama']
KM=train_data[train_data.store=='KaggleMart']
bb=KR.groupby(['date','country']).agg(num_sold=('num_sold','sum'))
cc=KM.groupby(['date','country']).agg(num_sold=('num_sold','sum'))

# Lineplots
ax1=sns.lineplot(ax=axes[0], data=bb, x='date', y='num_sold', hue='country')
ax2=sns.lineplot(ax=axes[1], data=cc, x='date', y='num_sold', hue='country')

# Aesthetics
ax1.title.set_text('KaggleRama')
ax2.title.set_text('KaggleMart')


# *Observations:*
# * We see that both stores sell more products in Norway than the other two countries. 
# * Finland and Sweden perform very similarly but maybe Sweden has a slight edge in general. 

# **Store sales by product type**

# In[9]:


# Subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Groupby
dd=KR.groupby(['date','product']).agg(num_sold=('num_sold','sum'))
ee=KM.groupby(['date','product']).agg(num_sold=('num_sold','sum'))

# Lineplots
ax1=sns.lineplot(ax=axes[0], data=dd, x='date', y='num_sold', hue='product')
ax2=sns.lineplot(ax=axes[1], data=ee, x='date', y='num_sold', hue='product')

# Aesthetics
ax1.title.set_text('KaggleRama')
ax2.title.set_text('KaggleMart')


# *Observations:*
# * We see that both stores sell Hats the most, then Mugs and finally Stickers the least. 
# * Sales of stickers is fairly constant throughout the year, whereas hat (especially) and mug sales is more affected by seasonality. 

# # Pre-processing & Feat. Eng.

# **Labels and features**

# In[10]:


# Labels
y=train_data.num_sold

# Features
X=train_data.drop('num_sold', axis=1)


# **Public holidays**

# In[11]:


holiday_path = '../input/holidays-finland-norway-sweden-20152019/Holidays_Finland_Norway_Sweden_2015-2019.csv'

def GetHoliday(holiday_path, df):
    """
    Get a boolean feature of whether the current row is a holiday sale
    """
    
    holiday = pd.read_csv(holiday_path)
    fin_holiday = holiday.loc[holiday.Country == 'Finland']
    swe_holiday = holiday.loc[holiday.Country == 'Sweden']
    nor_holiday = holiday.loc[holiday.Country == 'Norway']
    df['fin holiday'] = df.date.isin(fin_holiday.Date).astype(int)
    df['swe holiday'] = df.date.isin(swe_holiday.Date).astype(int)
    df['nor holiday'] = df.date.isin(nor_holiday.Date).astype(int)
    
    df['holiday'] = np.zeros(df.shape[0]).astype(int)
    df.loc[df.country == 'Finland', 'holiday'] = df.loc[df.country == 'Finland', 'fin holiday']
    df.loc[df.country == 'Sweden', 'holiday'] = df.loc[df.country == 'Sweden', 'swe holiday']
    df.loc[df.country == 'Norway', 'holiday'] = df.loc[df.country == 'Norway', 'nor holiday']
    df.drop(['fin holiday', 'swe holiday', 'nor holiday'], axis=1, inplace=True)
    return df

#X = GetHoliday(holiday_path, X)
#test_data = GetHoliday(holiday_path, test_data)


# **All holidays (inc. unofficial)**

# In[12]:


hol_path = '../input/public-and-unofficial-holidays-nor-fin-swe-201519/holidays.csv'

def unofficial_hol(hol_path, df):
    countries = {'Finland': 1, 'Norway': 2, 'Sweden': 3}
    stores = {'KaggleMart': 1, 'KaggleRama': 2}
    products = {'Kaggle Mug': 1,'Kaggle Hat': 2, 'Kaggle Sticker': 3}
    
    # load holiday info.
    holiday = pd.read_csv(hol_path)
    
    fin_holiday = holiday.loc[holiday.country == 'Finland']
    swe_holiday = holiday.loc[holiday.country == 'Sweden']
    nor_holiday = holiday.loc[holiday.country == 'Norway']
    df['fin holiday'] = df.date.isin(fin_holiday.date).astype(int)
    df['swe holiday'] = df.date.isin(swe_holiday.date).astype(int)
    df['nor holiday'] = df.date.isin(nor_holiday.date).astype(int)
    df['holiday'] = np.zeros(df.shape[0]).astype(int)
    df.loc[df.country == 'Finland', 'holiday'] = df.loc[df.country == 'Finland', 'fin holiday']
    df.loc[df.country == 'Sweden', 'holiday'] = df.loc[df.country == 'Sweden', 'swe holiday']
    df.loc[df.country == 'Norway', 'holiday'] = df.loc[df.country == 'Norway', 'nor holiday']
    df.drop(['fin holiday', 'swe holiday', 'nor holiday'], axis=1, inplace=True)
    
    return df

X = unofficial_hol(hol_path, X)
test = unofficial_hol(hol_path, test_data)


# **Include day of week, month, year etc**

# In[13]:


def date_feat_eng(df):
    df['day_of_week']=df['date'].dt.dayofweek       # 0 to 6
    df['day_of_month']=df['date'].dt.day            # 1 to 31
    df['weekend']=(df['day_of_week']//5 == 1)       # 0 or 1
    df['weekend']=df['weekend'].astype('int')       # int64
    df['week']=df['date'].dt.isocalendar().week     # 1 to 53
    df['week'][df['week']>52]=52                    # 1 to 52
    df['week']=df['week'].astype('int')             # int64
    df['month']=df['date'].dt.month                 # 1 to 12
    df['quarter']=df['date'].dt.quarter             # 1 to 4
    df['year']=df['date'].dt.year                   # 2015 to 2019
    return df

X= date_feat_eng(X)
test=date_feat_eng(test)


# **Drop 29th Feb**

# This date isn't useful for prediction because it doesn't appear in the test set. (2019 is not a leap year) It will also make the Fourier analysis easier later.

# In[14]:


# drop 29th Feb
#y.drop(X[(X.month==2) & (X.day_of_month==29)].index, axis=0, inplace=True)
#X.drop(X[(X.month==2) & (X.day_of_month==29)].index, axis=0, inplace=True)


# **Gross Domestic Product (GDP)**

# In[15]:


# Load data
GDP_data = pd.read_csv("../input/gdp-20152019-finland-norway-and-sweden/GDP_data_2015_to_2019_Finland_Norway_Sweden.csv",index_col="year")

# Rename the columns in GDP df 
GDP_data.columns = ['Finland', 'Norway', 'Sweden']

# Plot data
plt.figure(figsize=(8,5))

# Heatmap with annotations
sns.heatmap(GDP_data, annot=True, fmt='g', cmap='Blues')

# Aesthetics
plt.title('Heatmap of GDP in nordic countries')


# We see that the GDP in 2019 is lower than in 2018 so this could suggest sales will actually decrease in the test set.

# Now for some absolute [wizardry](https://www.kaggle.com/carlmcbrideellis/gdp-of-finland-norway-and-sweden-2015-2019/comments) with the help of Carl...

# In[16]:


# Create a dictionary
GDP_dictionary = GDP_data.unstack().to_dict()

# Create new GDP column
#X['GDP'] = X.set_index(['country', 'year']).index.map(GDP_dictionary.get)
#test['GDP'] = test.set_index(['country', 'year']).index.map(GDP_dictionary.get)


# **GDP per capita**

# In[17]:


# Load data
GDP_PC=pd.read_csv('../input/gdp-per-capita-finland-norway-sweden-201519/GDP_per_capita_2015_to_2019_Finland_Norway_Sweden.csv',index_col="year")

# Create a dictionary
GDP_PC_dictionary = GDP_PC.unstack().to_dict()

# Create new GDP_PC column
X['GDP_PC'] = X.set_index(['country', 'year']).index.map(GDP_PC_dictionary.get)
test['GDP_PC'] = test.set_index(['country', 'year']).index.map(GDP_PC_dictionary.get)

# Preview df
X.head()


# **CPI (inflation)**

# In[18]:


# Does not improve score
'''
# load data
CPI_data = pd.read_csv('../input/consumer-price-index-20152019-nordic-countries/consumer_price_index.csv')

# format data
CPI_data=CPI_data.T.iloc[2:,:]
CPI_data.columns = ['Finland', 'Norway', 'Sweden']
CPI_data.index=[2015,2016,2017,2018,2019]
CPI_data.index.name='year'

# Round to 2 d.p
CPI_data=CPI_data.astype(float).round(2)

# Create a dictionary
CPI_dictionary = CPI_data.unstack().to_dict()

# Create new CPI column
X['CPI'] = X.set_index(['country', 'year']).index.map(CPI_dictionary.get)
test['CPI'] = test.set_index(['country', 'year']).index.map(CPI_dictionary.get)

# Preview df
X.head()
'''


# # Fourier features

# These work best for linear models. Check out my other [notebook](https://www.kaggle.com/samuelcortinhas/tps-jan-22-hybrid-model) that uses these with a hybrid model. 

# In[19]:


# From https://www.kaggle.com/ambrosm/tpsjan22-03-linear-model#Simple-feature-engineering-(without-holidays)
def FourierFeatures(df):
    # temporary one hot encoding
    for product in ['Kaggle Mug', 'Kaggle Hat']:
        df[product] = df['product'] == product
    
    # The three products have different seasonal patterns
    dayofyear = df.date.dt.dayofyear
    for k in range(1, 3):
        df[f'sin{k}'] = np.sin(dayofyear / 365 * 2 * math.pi * k)
        df[f'cos{k}'] = np.cos(dayofyear / 365 * 2 * math.pi * k)
        df[f'mug_sin{k}'] = df[f'sin{k}'] * df['Kaggle Mug']
        df[f'mug_cos{k}'] = df[f'cos{k}'] * df['Kaggle Mug']
        df[f'hat_sin{k}'] = df[f'sin{k}'] * df['Kaggle Hat']
        df[f'hat_cos{k}'] = df[f'cos{k}'] * df['Kaggle Hat']
        df=df.drop([f'sin{k}', f'cos{k}'], axis=1)
    
    # drop temporary one hot encoding
    df=df.drop(['Kaggle Mug','Kaggle Hat'], axis=1)
    
    return df

# add fourier features
#X=fourier_features(X)
#test=fourier_features(test)


# **Drop date**

# In[20]:


X.drop('date',axis=1, inplace=True)
test.drop('date',axis=1, inplace=True)


# **Encode categorical variables**

# In[21]:


X=pd.get_dummies(X, columns=['store', 'country', 'product'])
test=pd.get_dummies(test, columns=['store', 'country', 'product'])


# # Modelling

# Base model for exploration and evaluation of new ideas

# In[22]:


'''
# Break off a validation set (in time-series-split style)
X_train=X.iloc[:3*len(X)//4,:]
X_valid=X.iloc[3*len(X)//4:,:]
y_train=y.iloc[:3*len(X)//4]
y_valid=y.iloc[3*len(X)//4:]

# Base model
model=LGBMRegressor(random_state=0, n_estimators=200, max_depth=6)

# Train model
model.fit(X_train,y_train)

# Predict
preds = model.predict(X_valid)

# Calcaculate smape
def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

# Evaluate smape
smape(preds,y_valid)
'''


# In[23]:


# Store results from experiments
smape_results=pd.DataFrame.from_dict({'Method':['base','include holidays','date feat. eng. (FE)', 'holidays + date FE', 
                                                'prev. row + GDP (model A)', 'model A + weekend', 'model A + day dummy',
                                                'model A + unofficial holidays', 'prev. row + GDP per capita', 'GDP per capita instead of GDP (Model B)',
                                                'Model B + CPI', 'Model B + drop feb 29', 'Model B + drop feb 29 + Fourier feats.'],
                                      'SMAPE': [16.52,16.46,9.06, 8.94, 9.02, 9.02, 21.97, 9.00, 8.97, 7.82, 7.83, 7.93, 7.93]})
smape_results


# In[24]:


# Parameter grid
grid = {'n_estimators': [50, 75, 100, 125, 150, 175, 200, 225, 250],
        'max_depth': [2, 4, 6, 8, 10, 12],
        'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15]}

# XGBoost model
model=LGBMRegressor(random_state=0)

# Grid Search with n-fold cross validation
grid_model = GridSearchCV(model,grid,cv=5)

# Train classifier with optimal parameters
grid_model.fit(X,y)


# **Results from Grid Search**

# In[25]:


print("The best parameters across ALL searched params:\n",grid_model.best_params_)
print("\n The best score across ALL searched params:\n",grid_model.best_score_) # r^2 score


# # Prediction

# In[26]:


# from https://www.kaggle.com/fergusfindley/ensembling-and-rounding-techniques-comparison
def geometric_round(arr):
    result_array = arr
    result_array = np.where(result_array < np.sqrt(np.floor(arr)*np.ceil(arr)), np.floor(arr), result_array)
    result_array = np.where(result_array >= np.sqrt(np.floor(arr)*np.ceil(arr)), np.ceil(arr), result_array)
    return result_array

# Make predictions
preds_test = geometric_round(grid_model.predict(test))

# Save predictions to file
output = pd.DataFrame({'row_id': test.index,
                       'num_sold': preds_test})

# Check format
output.head()


# In[27]:


output.to_csv('submission.csv', index=False)


# # Plot predictions

# In[28]:


def plot_predictions(SS, CC, PP):
    '''
    SS=store
    CC=country
    PP=product
    '''
    
    # uncomment if your dataframes have different names
    #train_data=train_df
    #test_data=test_df
    #output=preds
    
    # Training set target
    train_subset=train_data[(train_data.store==SS)&(train_data.country==CC)&(train_data['product']==PP)]

    # Predictions
    plot_index=test_data[(test_data.store==SS)&(test_data.country==CC)&(test_data['product']==PP)].index
    pred_subset=output[output.row_id.isin(plot_index)].reset_index(drop=True)

    # Plot
    plt.figure(figsize=(12,5))
    n1=len(train_subset['num_sold'])
    n2=len(pred_subset['num_sold'])
    plt.plot(np.arange(n1),train_subset['num_sold'], label='Training')
    plt.plot(np.arange(n1,n1+n2),pred_subset['num_sold'], label='Predictions')
    plt.title('\n'+f'Store:{SS}, Country:{CC}, Product:{PP}')
    plt.legend()
    plt.xlabel('Days since 2015-01-01')
    plt.ylabel('num_sold')


# In[29]:


for SS in ['KaggleMart','KaggleRama']:
    for CC in ['Finland', 'Norway', 'Sweden']:
        for PP in ['Kaggle Mug', 'Kaggle Hat', 'Kaggle Sticker']:
            plot_predictions(SS, CC, PP)

