#!/usr/bin/env python
# coding: utf-8

# # Introduction

# See my previous notebook for EDA and a LGBM model: [tps-jan-22-eda-modelling](https://www.kaggle.com/samuelcortinhas/tps-jan-22-eda-modelling). This notebook improves on my previous one and attempts a hybrid model. 
# 
# **Acknowledgements:**
# * Kaggle's [time series course](https://www.kaggle.com/learn/time-series).
# * This [notebook](https://www.kaggle.com/teckmengwong/tps2201-hybrid-time-series/notebook#Data/Feature-Engineering) by [Teck Meng Wong](https://www.kaggle.com/teckmengwong).
# * Many of [AmbrosM's](https://www.kaggle.com/ambrosm) great notebooks.
# * This [notebook](https://www.kaggle.com/lucamassaron/kaggle-merchandise-eda-with-baseline-linear-model/notebook) by [Luca Massaron](https://www.kaggle.com/lucamassaron).

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
import math
import statistics
import scipy.stats
from scipy.stats import pearsonr
import time
from datetime import datetime
import matplotlib.dates as mdates
import dateutil.easter as easter

# Sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge

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


# **Datetime and drop 29th Feb**

# In[3]:


# Convert date to datetime
train_data.date=pd.to_datetime(train_data.date)
test_data.date=pd.to_datetime(test_data.date)

# drop 29th Feb
train_data.drop(train_data[(train_data.date.dt.month==2) & (train_data.date.dt.day==29)].index, axis=0, inplace=True)


# # Quick EDA

# **num_sold by store**

# In[4]:


# Figure
plt.figure(figsize=(12,5))

# Groupby
aa=train_data.groupby(['date','store']).agg(num_sold=('num_sold','sum'))

# Lineplot
sns.lineplot(data=aa, x='date', y='num_sold', hue='store')

# Aesthetics
plt.title('num_sold by store')


# * KaggleRama consistently sells more products than KaggleMart. 
# * There are big spikes towards the end of each year.

# **num_sold by product**

# In[5]:


# Subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Groupby
KR=train_data[train_data.store=='KaggleRama']
KM=train_data[train_data.store=='KaggleMart']
bb=KR.groupby(['date','product']).agg(num_sold=('num_sold','sum'))
cc=KM.groupby(['date','product']).agg(num_sold=('num_sold','sum'))

# Lineplots
ax1=sns.lineplot(ax=axes[0], data=bb, x='date', y='num_sold', hue='product')
ax2=sns.lineplot(ax=axes[1], data=cc, x='date', y='num_sold', hue='product')

# Aesthetics
ax1.title.set_text('KaggleRama')
ax2.title.set_text('KaggleMart')


# * The Hat and Mug show strong yearly seasonal trends whereas the Sticker remains fairly constant. We can use Fourier Features to model these trends.
# * Hats sells the most, then Mugs and finally Stickers.

# **num_sold by country**

# In[6]:


# Subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Groupby
dd=KR.groupby(['date','country']).agg(num_sold=('num_sold','sum'))
ee=KM.groupby(['date','country']).agg(num_sold=('num_sold','sum'))

# Lineplots
ax1=sns.lineplot(ax=axes[0], data=dd, x='date', y='num_sold', hue='country')
ax2=sns.lineplot(ax=axes[1], data=ee, x='date', y='num_sold', hue='country')

# Aesthetics
ax1.title.set_text('KaggleRama')
ax2.title.set_text('KaggleMart')


# * Most products are sold in Norway, then Sweden followed by Finland.
# * Some spikes accur at different times for each country, perhaps because of different holidays.

# # Feature Engineering

# **Labels and features**

# In[7]:


# Labels
y=train_data.num_sold

# Features
X=train_data.drop('num_sold', axis=1)


# **Public holidays (including unofficial ones)**

# In[8]:


# From https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/298990
def unofficial_hol(df):
    countries = {'Finland': 1, 'Norway': 2, 'Sweden': 3}
    stores = {'KaggleMart': 1, 'KaggleRama': 2}
    products = {'Kaggle Mug': 1,'Kaggle Hat': 2, 'Kaggle Sticker': 3}
    
    # load holiday info.
    hol_path = '../input/public-and-unofficial-holidays-nor-fin-swe-201519/holidays.csv'
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


# **Holidays (from AmbrosM)**

# In[9]:


def get_holidays(df):
    # End of year
    df = pd.concat([df, pd.DataFrame({f"dec{d}":
                      (df.date.dt.month == 12) & (df.date.dt.day == d)
                      for d in range(24, 32)}),
        pd.DataFrame({f"n-dec{d}":
                      (df.date.dt.month == 12) & (df.date.dt.day == d) & (df.country == 'Norway')
                      for d in range(24, 32)}),
        pd.DataFrame({f"f-jan{d}":
                      (df.date.dt.month == 1) & (df.date.dt.day == d) & (df.country == 'Finland')
                      for d in range(1, 14)}),
        pd.DataFrame({f"jan{d}":
                      (df.date.dt.month == 1) & (df.date.dt.day == d) & (df.country == 'Norway')
                      for d in range(1, 10)}),
        pd.DataFrame({f"s-jan{d}":
                      (df.date.dt.month == 1) & (df.date.dt.day == d) & (df.country == 'Sweden')
                      for d in range(1, 15)})], axis=1)
    
    # May
    df = pd.concat([df, pd.DataFrame({f"may{d}":
                      (df.date.dt.month == 5) & (df.date.dt.day == d) 
                      for d in list(range(1, 10))}),
        pd.DataFrame({f"may{d}":
                      (df.date.dt.month == 5) & (df.date.dt.day == d) & (df.country == 'Norway')
                      for d in list(range(19, 26))})], axis=1)
    
    # June and July
    df = pd.concat([df, pd.DataFrame({f"june{d}":
                   (df.date.dt.month == 6) & (df.date.dt.day == d) & (df.country == 'Sweden')
                   for d in list(range(8, 14))})], axis=1)
    
    #Swedish Rock Concert
    #Jun 3, 2015 – Jun 6, 2015
    #Jun 8, 2016 – Jun 11, 2016
    #Jun 7, 2017 – Jun 10, 2017
    #Jun 6, 2018 – Jun 10, 2018
    #Jun 5, 2019 – Jun 8, 2019
    swed_rock_fest  = df.date.dt.year.map({2015: pd.Timestamp(('2015-06-6')),
                                         2016: pd.Timestamp(('2016-06-11')),
                                         2017: pd.Timestamp(('2017-06-10')),
                                         2018: pd.Timestamp(('2018-06-10')),
                                         2019: pd.Timestamp(('2019-06-8'))})

    df = pd.concat([df, pd.DataFrame({f"swed_rock_fest{d}":
                                      (df.date - swed_rock_fest == np.timedelta64(d, "D")) & (df.country == 'Sweden')
                                      for d in list(range(-3, 3))})], axis=1)
    
    # Last Wednesday of June
    wed_june_date = df.date.dt.year.map({2015: pd.Timestamp(('2015-06-24')),
                                         2016: pd.Timestamp(('2016-06-29')),
                                         2017: pd.Timestamp(('2017-06-28')),
                                         2018: pd.Timestamp(('2018-06-27')),
                                         2019: pd.Timestamp(('2019-06-26'))})
    
    df = pd.concat([df, pd.DataFrame({f"wed_june{d}": 
                   (df.date - wed_june_date == np.timedelta64(d, "D")) & (df.country != 'Norway')
                   for d in list(range(-4, 6))})], axis=1)
    
    # First Sunday of November
    sun_nov_date = df.date.dt.year.map({2015: pd.Timestamp(('2015-11-1')),
                                         2016: pd.Timestamp(('2016-11-6')),
                                         2017: pd.Timestamp(('2017-11-5')),
                                         2018: pd.Timestamp(('2018-11-4')),
                                         2019: pd.Timestamp(('2019-11-3'))})
    
    df = pd.concat([df, pd.DataFrame({f"sun_nov{d}": 
                   (df.date - sun_nov_date == np.timedelta64(d, "D")) & (df.country != 'Norway')
                   for d in list(range(0, 9))})], axis=1)
    
    # First half of December (Independence Day of Finland, 6th of December)
    df = pd.concat([df, pd.DataFrame({f"dec{d}":
                   (df.date.dt.month == 12) & (df.date.dt.day == d) & (df.country == 'Finland')
                   for d in list(range(6, 14))})], axis=1)

    # Easter
    easter_date = df.date.apply(lambda date: pd.Timestamp(easter.easter(date.year)))
    df = pd.concat([df, pd.DataFrame({f"easter{d}":
                   (df.date - easter_date == np.timedelta64(d, "D"))
                   for d in list(range(-2, 11)) + list(range(40, 48)) + list(range(50, 59))})], axis=1)
    
    return df


# **Include day of week, month, year etc**

# In[10]:


def date_feat_eng_X1(df):
    df['year']=df['date'].dt.year                   # 2015 to 2019
    return df

def date_feat_eng_X2(df):
    df['day_of_week']=df['date'].dt.dayofweek       # 0 to 6
    df['day_of_month']=df['date'].dt.day            # 1 to 31
    df['dayofyear'] = df['date'].dt.dayofyear       # 1 to 366
    df.loc[(df.date.dt.year==2016) & (df.dayofyear>60), 'dayofyear'] -= 1   # 1 to 365
    df['week']=df['date'].dt.isocalendar().week     # 1 to 53
    df['week']=df['week'].astype('int')             # int64
    df['month']=df['date'].dt.month                 # 1 to 12
    return df


# **GDP**

# In[11]:


def get_GDP(df):

    # Load data
    GDP_data = pd.read_csv("../input/gdp-20152019-finland-norway-and-sweden/GDP_data_2015_to_2019_Finland_Norway_Sweden.csv",index_col="year")

    # Rename the columns in GDP df 
    GDP_data.columns = ['Finland', 'Norway', 'Sweden']

    # Create a dictionary
    GDP_dictionary = GDP_data.unstack().to_dict()
    
    # Create GDP column
    df['GDP']=df.set_index(['country', 'year']).index.map(GDP_dictionary.get)
    
    # Log transform (only if the target is log-transformed too)
    df['GDP']=np.log(df['GDP'])
    
    # Split GDP by country (for linear model)
    df['GDP_Finland']=df['GDP'] * (df['country']=='Finland')
    df['GDP_Norway']=df['GDP'] * (df['country']=='Norway')
    df['GDP_Sweden']=df['GDP'] * (df['country']=='Sweden')
    
    # Drop column
    df=df.drop(['GDP','year'],axis=1)
    
    return df


# **GDP per capita**

# In[12]:


def GDP_PC(df):
    # Load data
    GDP_PC_data = pd.read_csv("../input/gdp-per-capita-finland-norway-sweden-201519/GDP_per_capita_2015_to_2019_Finland_Norway_Sweden.csv",index_col="year")
    
    # Create a dictionary
    GDP_PC_dictionary = GDP_PC_data.unstack().to_dict()

    # Create new GDP_PC column
    df['GDP_PC'] = df.set_index(['country', 'year']).index.map(GDP_PC_dictionary.get)
    
    return df


# **GDP vs GDP per capita**

# In[13]:


def GDP_corr(df):
    
    # Load data
    GDP_data = pd.read_csv("../input/gdp-20152019-finland-norway-and-sweden/GDP_data_2015_to_2019_Finland_Norway_Sweden.csv",index_col="year")
    GDP_PC_data = pd.read_csv("../input/gdp-per-capita-finland-norway-sweden-201519/GDP_per_capita_2015_to_2019_Finland_Norway_Sweden.csv",index_col="year")

    # Rename the columns
    GDP_data.columns = ['Finland', 'Norway', 'Sweden']
    
    # Create dictionary
    GDP_dictionary = GDP_data.unstack().to_dict()
    GDP_PC_dictionary = GDP_PC_data.unstack().to_dict()
    
    # Add year column
    df['year']=df.date.dt.year
    
    # Make new column
    df['GDP']=df.set_index(['country', 'year']).index.map(GDP_dictionary.get)
    df['GDP_PC'] = df.set_index(['country', 'year']).index.map(GDP_PC_dictionary.get)

    # Initialise output
    feat_corr=[]
    
    # Compute pairwise correlations
    for SS in ['KaggleMart', 'KaggleRama']:
        for CC in ['Finland', 'Norway', 'Sweden']:
            for PP in ['Kaggle Mug', 'Kaggle Hat', 'Kaggle Sticker']:
                subset=df[(df.store==SS)&(df.country==CC)&(df['product']==PP)].groupby(['year']).agg(num_sold=('num_sold','sum'), GDP=('GDP','mean'), GDP_PC=('GDP_PC','mean'))
                v1=subset.num_sold
                v2=subset.GDP
                v3=subset.GDP_PC
                
                r1, _ = pearsonr(v1,v2)
                r2, _ = pearsonr(v1,v3)
                
                feat_corr.append([f'{SS}, {CC}, {PP}', r1, r2])

    return pd.DataFrame(feat_corr, columns=['Features', 'GDP_corr', 'GDP_PC_corr'])
    
corr_df=GDP_corr(train_data)
corr_df.head()


# * In general, both GDP and GDP_PC are very highly correlated to the num_sold aggregate each year. 
# * GDP tends to have a slightly higher correlation than GDP_PC.

# **Fourier features**

# In[14]:


# From https://www.kaggle.com/ambrosm/tpsjan22-03-linear-model#Simple-feature-engineering-(without-holidays)
def FourierFeatures(df):
    # temporary one hot encoding
    for product in ['Kaggle Mug', 'Kaggle Hat']:
        df[product] = df['product'] == product
    
    # The three products have different seasonal patterns
    dayofyear = df.date.dt.dayofyear
    for k in range(1, 2):
        df[f'sin{k}'] = np.sin(dayofyear / 365 * 2 * math.pi * k)
        df[f'cos{k}'] = np.cos(dayofyear / 365 * 2 * math.pi * k)
        df[f'mug_sin{k}'] = df[f'sin{k}'] * df['Kaggle Mug']
        df[f'mug_cos{k}'] = df[f'cos{k}'] * df['Kaggle Mug']
        df[f'hat_sin{k}'] = df[f'sin{k}'] * df['Kaggle Hat']
        df[f'hat_cos{k}'] = df[f'cos{k}'] * df['Kaggle Hat']
        df=df.drop([f'sin{k}', f'cos{k}'], axis=1)
    
    # drop temporary one hot encoding
    df=df.drop(['Kaggle Mug', 'Kaggle Hat'], axis=1)
    
    return df


# **Interactions**

# In[15]:


# Help linear model find the right height of trends for each combination of features
def get_interactions(df):
    df['KR_Sweden_Mug']=(df.country=='Sweden')*(df['product']=='Kaggle Mug')*(df.store=='KaggleRama')
    df['KR_Sweden_Hat']=(df.country=='Sweden')*(df['product']=='Kaggle Hat')*(df.store=='KaggleRama')
    df['KR_Sweden_Sticker']=(df.country=='Sweden')*(df['product']=='Kaggle Sticker')*(df.store=='KaggleRama')
    df['KR_Norway_Mug']=(df.country=='Norway')*(df['product']=='Kaggle Mug')*(df.store=='KaggleRama')
    df['KR_Norway_Hat']=(df.country=='Norway')*(df['product']=='Kaggle Hat')*(df.store=='KaggleRama')
    df['KR_Norway_Sticker']=(df.country=='Norway')*(df['product']=='Kaggle Sticker')*(df.store=='KaggleRama')
    df['KR_Finland_Mug']=(df.country=='Finland')*(df['product']=='Kaggle Mug')*(df.store=='KaggleRama')
    df['KR_Finland_Hat']=(df.country=='Finland')*(df['product']=='Kaggle Hat')*(df.store=='KaggleRama')
    df['KR_Finland_Sticker']=(df.country=='Finland')*(df['product']=='Kaggle Sticker')*(df.store=='KaggleRama')
    
    df['KM_Sweden_Mug']=(df.country=='Sweden')*(df['product']=='Kaggle Mug')*(df.store=='KaggleMart')
    df['KM_Sweden_Hat']=(df.country=='Sweden')*(df['product']=='Kaggle Hat')*(df.store=='KaggleMart')
    df['KM_Sweden_Sticker']=(df.country=='Sweden')*(df['product']=='Kaggle Sticker')*(df.store=='KaggleMart')
    df['KM_Norway_Mug']=(df.country=='Norway')*(df['product']=='Kaggle Mug')*(df.store=='KaggleMart')
    df['KM_Norway_Hat']=(df.country=='Norway')*(df['product']=='Kaggle Hat')*(df.store=='KaggleMart')
    df['KM_Norway_Sticker']=(df.country=='Norway')*(df['product']=='Kaggle Sticker')*(df.store=='KaggleMart')
    df['KM_Finland_Mug']=(df.country=='Finland')*(df['product']=='Kaggle Mug')*(df.store=='KaggleMart')
    df['KM_Finland_Hat']=(df.country=='Finland')*(df['product']=='Kaggle Hat')*(df.store=='KaggleMart')
    df['KM_Finland_Sticker']=(df.country=='Finland')*(df['product']=='Kaggle Sticker')*(df.store=='KaggleMart')
    
    return df


# **Drop date and one hot encoding**

# In[16]:


def dropdate(df):
    df=df.drop('date',axis=1)
    return df

def onehot(df,columns):
    df=pd.get_dummies(df, columns)
    return df


# **Put pieces together**

# In[17]:


# Feature set for trend model
def FeatEng_X1(df):
    df=date_feat_eng_X1(df)
    df=get_GDP(df)
    df=FourierFeatures(df)
    df=get_interactions(df)
    df=dropdate(df)
    df=onehot(df,['store', 'product', 'country'])
    return df

# Feature set for interactions model
def FeatEng_X2(df):
    df=date_feat_eng_X2(df)
    df=unofficial_hol(df)
    df=get_holidays(df)
    df=dropdate(df)
    df=onehot(df,['store', 'product', 'country'])
    return df

# Apply feature engineering
X_train_1=FeatEng_X1(X)
X_train_2=FeatEng_X2(X)
X_test_1=FeatEng_X1(test_data)
X_test_2=FeatEng_X2(test_data)


# # Hybrid model

# The idea is as follows. Linear interpolation is good at extrapolating trends but poor at learning interactions. Conversely, decision tree algorithms like XGBoost are very good at learning interactions but can't extrapolate trends. A hybrid model tries to take the best of both worlds by first learning the trend with linear interpolation and then learning the interactions on the detrended time series.

# In[18]:


# A class is a collection of properties and methods (like models from Sklearn)
class HybridModel:
    def __init__(self, model_1, model_2, grid=None):
        self.model_1 = model_1
        self.model_2 = model_2
        self.grid=grid
        
    def fit(self, X_train_1, X_train_2, y):
        # Train model 1
        self.model_1.fit(X_train_1, y)
        
        # Predictions from model 1 (trend)
        y_trend = self.model_1.predict(X_train_1)

        if self.grid:
            # Grid search
            tscv = TimeSeriesSplit(n_splits=3)
            grid_model = GridSearchCV(estimator=self.model_2, cv=tscv, param_grid=self.grid)
        
            # Train model 2 on detrended series
            grid_model.fit(X_train_2, y-y_trend)
            
            # Model 2 preditions (for residual analysis)
            y_resid = grid_model.predict(X_train_2)
            
            # Save model
            self.grid_model=grid_model
        else:
            # Train model 2 on residuals
            self.model_2.fit(X_train_2, y-y_trend)
            
            # Model 2 preditions (for residual analysis)
            y_resid = self.model_2.predict(X_train_2)
        
        # Save data
        self.y_train_trend = y_trend
        self.y_train_resid = y_resid
        
    def predict(self, X_test_1, X_test_2):
        # Predict trend using model 1
        y_trend = self.model_1.predict(X_test_1)
        
        if self.grid:
            # Grid model predictions
            y_resid = self.grid_model.predict(X_test_2)
        else:
            # Model 2 predictions
            y_resid = self.model_2.predict(X_test_2)
        
        # Add predictions together
        y_pred = y_trend + y_resid
        
        # Save data
        self.y_test_trend = y_trend
        self.y_test_resid = y_resid
        
        return y_pred


# # Prediction

# **Ensembling**

# In[19]:


# Choose models
model_1=LinearRegression()
models_2=[LGBMRegressor(random_state=0), CatBoostRegressor(random_state=0, verbose=False), XGBRegressor(random_state=0)]

# Parameter grid
param_grid = {'n_estimators': [100, 150, 200, 225, 250, 275, 300],
        'max_depth': [4, 5, 6, 7],
        'learning_rate': [0.1, 0.12, 0.13, 0.14, 0.15]}

# Initialise output vectors
y_pred=np.zeros(len(test_data))
train_preds=np.zeros(len(y))

# Ensemble predictions
for model_2 in models_2:
    # Start timer
    start = time.time()
    
    # Construct hybrid model
    model = HybridModel(model_1, model_2, grid=param_grid)

    # Train model
    model.fit(X_train_1, X_train_2, np.log(y))

    # Save predictions
    y_pred += np.exp(model.predict(X_test_1,X_test_2))
    
    # Training set predictions (for residual analysis)
    train_preds += np.exp(model.y_train_trend+model.y_train_resid)
    
    # Stop timer
    stop = time.time()
    
    print(f'Model_2:{model_2} -- time:{round((stop-start)/60,2)} mins')
    
    if model.grid:
        print('Best parameters:',model.grid_model.best_params_,'\n')
    
# Scale
y_pred = y_pred/len(models_2)
train_preds = train_preds/len(models_2)


# **Post-processing**

# In[20]:


# From https://www.kaggle.com/fergusfindley/ensembling-and-rounding-techniques-comparison
def geometric_round(arr):
    result_array = arr
    result_array = np.where(result_array < np.sqrt(np.floor(arr)*np.ceil(arr)), np.floor(arr), result_array)
    result_array = np.where(result_array >= np.sqrt(np.floor(arr)*np.ceil(arr)), np.ceil(arr), result_array)
    return result_array

y_pred=geometric_round(y_pred)

# Save predictions to file
output = pd.DataFrame({'row_id': test_data.index, 'num_sold': y_pred})

# Check format
output.head()


# In[21]:


output.to_csv('submission.csv', index=False)


# # Plot predictions

# In[22]:


def plot_predictions(SS, CC, PP, series=output):
    '''
    SS=store
    CC=country
    PP=product
    '''
    
    # uncomment if your dataframes have different names
    #train_data=train_df
    #test_data=test_df
    
    # Training set target
    train_subset=train_data[(train_data.store==SS)&(train_data.country==CC)&(train_data['product']==PP)]
    
    # Predictions
    plot_index=test_data[(test_data.store==SS)&(test_data.country==CC)&(test_data['product']==PP)].index
    pred_subset=series[series.row_id.isin(plot_index)].reset_index(drop=True)
    
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


# **Plot trends**

# In[23]:


# Put into dataframes
y_trend=pd.DataFrame({'row_id': test_data.index, 'num_sold': np.exp(model.y_test_trend)})
y_resid=pd.DataFrame({'row_id': test_data.index, 'num_sold': np.exp(model.y_test_resid)})
y_pred=pd.DataFrame({'row_id': test_data.index, 'num_sold': np.exp(model.y_test_trend+model.y_test_resid)})

# Choose parameters
SS='KaggleMart'
CC='Norway'

# Plot trends (model 1 predictions)
plot_predictions(SS, CC, 'Kaggle Hat', series=y_trend)
plot_predictions(SS, CC, 'Kaggle Mug', series=y_trend)
plot_predictions(SS, CC, 'Kaggle Sticker', series=y_trend)


# **All predictions**

# In[24]:


for SS in ['KaggleMart','KaggleRama']:
    for CC in ['Finland', 'Norway', 'Sweden']:
        for PP in ['Kaggle Mug', 'Kaggle Hat', 'Kaggle Sticker']:
            plot_predictions(SS, CC, PP)


# # Residual Analysis

# **Plot residuals**

# In[25]:


# need to ensemble
train_preds = np.exp(model.y_train_trend+model.y_train_resid)

# Residuals on training set (SMAPE)
residuals = 200 * (train_preds - y) / (train_preds + y)

# Plot residuals
plt.figure(figsize=(12,4))
plt.scatter(np.arange(len(residuals)),residuals, s=1)
plt.hlines([0], 0, residuals.index.max(), color='k')
plt.title('Residuals on training set')
plt.xlabel('Sample')
plt.ylabel('SMAPE')


# **Plot histogram of residuals**

# In[26]:


mu, std = scipy.stats.norm.fit(residuals)

plt.figure(figsize=(12,4))
plt.hist(residuals, bins=100, density=True)
x = np.linspace(plt.xlim()[0], plt.xlim()[1], 200)
plt.plot(x, scipy.stats.norm.pdf(x, mu, std), 'r', linewidth=2)
plt.title(f'Histogram of residuals; mean = {residuals.mean():.4f}, '
          f'$\sigma = {residuals.std():.1f}$, SMAPE = {residuals.abs().mean():.5f}')
plt.xlabel('Residual (percent)')
plt.ylabel('Density')
plt.show()

