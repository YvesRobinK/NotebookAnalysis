#!/usr/bin/env python
# coding: utf-8

# # Summary
# 
# 
# #### About the dataset
# 
# Data found in this dataset was collected from the [Climate Data Online (CDO)](https://www.ncdc.noaa.gov/cdo-web/) of the National Centers For Environmental Information (NCEI). It contains daily country average precipitation and air temperature data (in metric units). The original dataset collected from the CDO's site consisted of around 4.9 million individual observations from 1306 distinct weather stations throughout the three countries. Missing data points were imputed with the daily mean and averaged across all weather stations within the country.
# 
# The dataset can be accessed [here](https://www.kaggle.com/adamwurdits/finland-norway-and-sweden-weather-data-20152019) along with some more information. For updates, suggestions and general discussion, please visit [this](https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/301486) thread.
# 
# ### Findings
# 
# Neither the base features nor any feature engineering seems to improve our predictions.
# * The use of some features improves cross-validation SMAPE slightly, but makes the public score worse
# * Correlations in weather and traing data coincide with the effects of seasonality

# # Importing libraries

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns
import optuna
from catboost import CatBoostRegressor
import shap


# # Loading the data

# In[2]:


df_train = pd.read_csv('../input/tabular-playground-series-jan-2022/train.csv', parse_dates=['date'])
df_test = pd.read_csv('../input/tabular-playground-series-jan-2022/test.csv', parse_dates=['date'])
sample_submission = pd.read_csv('../input/tabular-playground-series-jan-2022/sample_submission.csv')
df_weather = pd.read_csv('../input/finland-norway-and-sweden-weather-data-20152019/nordics_weather.csv', parse_dates=['date'])


# # Preprocessing

# Let's merge the weather dataset with our training and test sets and do some basic datetime feature engineering. The latter will make parts of the EDA easier.

# In[3]:


# Adding weather data to our dataframes.

df_train = df_train.merge(df_weather, on=['date', 'country'], how='left')
df_test = df_test.merge(df_weather, on=['date', 'country'], how='left')

weather_features = ['precipitation', 'snow_depth', 'tavg', 'tmax', 'tmin']

# Creating new features from the 'date' column.
df_train['year'] = df_train['date'].dt.year
df_train['quarter'] = df_train['date'].dt.quarter
df_train['month'] = df_train['date'].dt.month
df_train['week'] = df_train['date'].dt.isocalendar().week.astype(int)
df_train['day'] = df_train['date'].dt.day
df_train['dayofyear'] = df_train['date'].dt.dayofyear
df_train['daysinmonth'] = df_train['date'].dt.days_in_month
df_train['dayofweek'] = df_train['date'].dt.dayofweek
df_train['weekend'] = ((df_train['date'].dt.dayofweek) // 5 == 1).astype(int)

df_test['year'] = df_test['date'].dt.year
df_test['quarter'] = df_test['date'].dt.quarter
df_test['month'] = df_test['date'].dt.month
df_test['week'] = df_test['date'].dt.isocalendar().week.astype(int)
df_test['day'] = df_test['date'].dt.day
df_test['dayofyear'] = df_test['date'].dt.dayofyear
df_test['daysinmonth'] = df_test['date'].dt.days_in_month
df_test['dayofweek'] = df_test['date'].dt.dayofweek
df_test['weekend'] = ((df_test['date'].dt.dayofweek) // 5 == 1).astype(int)


# # Summary statistics and trends
# 
# We will take a quick look at the data and create a few plots - just enough to get a feel for the data.

# In[4]:


df_weather.head()


# Our dataframe contains 7 columns of which 5 are weather features. These are:
# 
# - Precipitation - How much rain, snow, hail, etc has fallen. Measured in centimeters (cm).
# - Snow depth - How much snow has collected on the ground. Measured in millimeters (mm).
# - Temperature average - Country average of daily mean temperatures. Measured in degrees Celsius (°C).
# - Temperature maximum - Country average of daily maximum temperatures. Measured in degrees Celsius (°C).
# - Temperature minimum - Country average of daily minimum temperatures. Measured in degrees Celsius (°C).

# In[5]:


df_train.groupby(['country', 'month'])[weather_features].mean().style.set_caption('Daily averages by month')


# How to read this table? Taking the very first row as an example, we learn that the daily average precipiation in Finland in January is 1.4 centimeters. This means we can expect 1.4 centimeters of rain, snow or other precipitation every day of the month. Average snow depth is 327 millimeters, which means snow accumulates this high on the ground. Average temperature is self-explanatory, and the last two features show the averages of all the minimum and maximum weather station measurements.
# 
# Let's create a few plots next to identify trends in the data.

# In[6]:


fig = plt.figure(figsize=(12, 10))

fig.add_subplot(211)
plt.title('Daily mean precipitation')
sns.lineplot(data=df_train[df_train['country'] == 'Finland'], x='month', y='precipitation', label='Finland')
sns.lineplot(data=df_train[df_train['country'] == 'Norway'], x='month', y='precipitation', label='Norway')
sns.lineplot(data=df_train[df_train['country'] == 'Sweden'], x='month', y='precipitation', label='Sweden')
sns.lineplot(data=df_train, x='month', y='precipitation', label='Average', color='grey')

fig.add_subplot(212)
plt.title('Daily mean snow depth')
sns.lineplot(data=df_train[df_train['country'] == 'Finland'], x='month', y='snow_depth', label='Finland')
sns.lineplot(data=df_train[df_train['country'] == 'Norway'], x='month', y='snow_depth', label='Norway')
sns.lineplot(data=df_train[df_train['country'] == 'Sweden'], x='month', y='snow_depth', label='Sweden')
sns.lineplot(data=df_train, x='month', y='snow_depth', label='Average', color='grey')


# We can see that countries receive more precipitation in the second half of the year and Norway receives about as much as the other two countries. Snow starts to accumulate in October, starts melting in middle of spring and melts completely by the end on May. On first look, this may seem late. Let's take a look at the temperatures as a sanity check.

# In[7]:


fig = plt.figure(figsize=(18, 6))

fig.add_subplot(131)
plt.title('Mean temperature')
plt.xlabel('Months')
sns.lineplot(data=df_train[df_train['country'] == 'Finland'], x='month', y='tavg', label='Finland')
sns.lineplot(data=df_train[df_train['country'] == 'Norway'], x='month', y='tavg', label='Norway')
sns.lineplot(data=df_train[df_train['country'] == 'Sweden'], x='month', y='tavg', label='Sweden')

fig.add_subplot(132)
plt.title('Maximum temperature')
plt.xlabel('Months')
sns.lineplot(data=df_train[df_train['country'] == 'Finland'], x='month', y='tmax', label='Finland')
sns.lineplot(data=df_train[df_train['country'] == 'Norway'], x='month', y='tmax', label='Norway')
sns.lineplot(data=df_train[df_train['country'] == 'Sweden'], x='month', y='tmax', label='Sweden')

fig.add_subplot(133)
plt.title('Minimum temperature')
plt.xlabel('Months')
sns.lineplot(data=df_train[df_train['country'] == 'Finland'], x='month', y='tmin', label='Finland')
sns.lineplot(data=df_train[df_train['country'] == 'Norway'], x='month', y='tmin', label='Norway')
sns.lineplot(data=df_train[df_train['country'] == 'Sweden'], x='month', y='tmin', label='Sweden')


# Mean temperatures creep above freezing point in April and are still in the single digits in May, so it makes sense that we still see some snow.
# 
# Now let's examine our sales by country and product and see if we identify similar trends.

# In[8]:


fig = plt.figure(figsize=(18, 6))

fig.add_subplot(131)
plt.title('Kaggle Hat sales')
plt.xlabel('Months')
sns.lineplot(data=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Hat'], x='month', y='num_sold', label='Finland')
sns.lineplot(data=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Hat'], x='month', y='num_sold', label='Norway')
sns.lineplot(data=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Hat'], x='month', y='num_sold', label='Sweden')

fig.add_subplot(132)
plt.title('Kaggle Mug sales')
plt.xlabel('Months')
sns.lineplot(data=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Mug'], x='month', y='num_sold', label='Finland')
sns.lineplot(data=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Mug'], x='month', y='num_sold', label='Norway')
sns.lineplot(data=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Mug'], x='month', y='num_sold', label='Sweden')

fig.add_subplot(133)
plt.title('Kaggle Sticker sales')
plt.xlabel('Months')
sns.lineplot(data=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Sticker'], x='month', y='num_sold', label='Finland')
sns.lineplot(data=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Sticker'], x='month', y='num_sold', label='Norway')
sns.lineplot(data=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Sticker'], x='month', y='num_sold', label='Sweden')


# We have seen so far that:
# 
# 1. Hat sales start increasing in October and start decreasing after the peak in April. We saw the same trend with snow depth.
# 2. Mug sales start increasing in July as temperatures drop and start decreasing after the peak in December. We saw the same trend with precipitation.
# 3. Sticker sales peak in December and April but are otherwise even throughout the year.
# 
# The first two points require more investigation.

# # Identifying relationships
# 
# In this section, we will look at relationships of two variables - number of products sold and one of our weather features. Regression plots work well for this purpose. They are just like scatter plots in that they show the relationship between two variables, but also fit a regression line on the points. This shows the strength and nature of the relationship. Let's do for precipitation, snow depth and temperatures combined and we will summarize our findings in the end.

# In[9]:


fig = plt.figure(figsize=(18, 16))

fig.add_subplot(331)
plt.title('Kaggle Hat sales in Finland')
plt.xlabel('Months')
sns.regplot(x=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Hat']['num_sold'], y=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Hat']['precipitation'])

fig.add_subplot(332)
plt.title('Kaggle Mug sales in Finland')
plt.xlabel('Months')
sns.regplot(x=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Mug']['num_sold'], y=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Mug']['precipitation'])

fig.add_subplot(333)
plt.title('Kaggle Sticker sales in Finland')
plt.xlabel('Months')
sns.regplot(x=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Sticker']['num_sold'], y=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Sticker']['precipitation'])

fig.add_subplot(334)
plt.title('Kaggle Hat sales in Norway')
plt.xlabel('Months')
sns.regplot(x=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Hat']['num_sold'], y=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Hat']['precipitation'])

fig.add_subplot(335)
plt.title('Kaggle Mug sales in Norway')
plt.xlabel('Months')
sns.regplot(x=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Mug']['num_sold'], y=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Mug']['precipitation'])

fig.add_subplot(336)
plt.title('Kaggle Sticker sales in Norway')
plt.xlabel('Months')
sns.regplot(x=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Sticker']['num_sold'], y=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Sticker']['precipitation'])

fig.add_subplot(337)
plt.title('Kaggle Hat sales in Sweden')
plt.xlabel('Months')
sns.regplot(x=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Hat']['num_sold'], y=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Hat']['precipitation'])

fig.add_subplot(338)
plt.title('Kaggle Mug sales in Sweden')
plt.xlabel('Months')
sns.regplot(x=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Mug']['num_sold'], y=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Mug']['precipitation'])

fig.add_subplot(339)
plt.title('Kaggle Sticker sales in Sweden')
plt.xlabel('Months')
sns.regplot(x=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Sticker']['num_sold'], y=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Sticker']['precipitation'])


# In[10]:


fig = plt.figure(figsize=(18, 16))

fig.add_subplot(331)
plt.title('Kaggle Hat sales in Finland')
plt.xlabel('Months')
sns.regplot(x=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Hat']['num_sold'], y=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Hat']['snow_depth'])

fig.add_subplot(332)
plt.title('Kaggle Mug sales in Finland')
plt.xlabel('Months')
sns.regplot(x=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Mug']['num_sold'], y=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Mug']['snow_depth'])

fig.add_subplot(333)
plt.title('Kaggle Sticker sales in Finland')
plt.xlabel('Months')
sns.regplot(x=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Sticker']['num_sold'], y=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Sticker']['snow_depth'])

fig.add_subplot(334)
plt.title('Kaggle Hat sales in Norway')
plt.xlabel('Months')
sns.regplot(x=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Hat']['num_sold'], y=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Hat']['snow_depth'])

fig.add_subplot(335)
plt.title('Kaggle Mug sales in Norway')
plt.xlabel('Months')
sns.regplot(x=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Mug']['num_sold'], y=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Mug']['snow_depth'])

fig.add_subplot(336)
plt.title('Kaggle Sticker sales in Norway')
plt.xlabel('Months')
sns.regplot(x=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Sticker']['num_sold'], y=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Sticker']['snow_depth'])

fig.add_subplot(337)
plt.title('Kaggle Hat sales in Sweden')
plt.xlabel('Months')
sns.regplot(x=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Hat']['num_sold'], y=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Hat']['snow_depth'])

fig.add_subplot(338)
plt.title('Kaggle Mug sales in Sweden')
plt.xlabel('Months')
sns.regplot(x=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Mug']['num_sold'], y=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Mug']['snow_depth'])

fig.add_subplot(339)
plt.title('Kaggle Sticker sales in Sweden')
plt.xlabel('Months')
sns.regplot(x=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Sticker']['num_sold'], y=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Sticker']['snow_depth'])


# In[11]:


fig = plt.figure(figsize=(18, 16))

fig.add_subplot(331)
plt.title('Kaggle Hat sales in Finland')
sns.regplot(x=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Hat']['num_sold'], y=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Hat']['tavg'])
sns.regplot(x=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Hat']['num_sold'], y=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Hat']['tmax'])
sns.regplot(x=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Hat']['num_sold'], y=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Hat']['tmin'])

fig.add_subplot(332)
plt.title('Kaggle Mug sales in Finland')
sns.regplot(x=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Mug']['num_sold'], y=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Mug']['tavg'])
sns.regplot(x=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Mug']['num_sold'], y=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Mug']['tmax'])
sns.regplot(x=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Mug']['num_sold'], y=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Mug']['tmin'])

fig.add_subplot(333)
plt.title('Kaggle Sticker sales in Finland')
sns.regplot(x=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Sticker']['num_sold'], y=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Sticker']['tavg'])
sns.regplot(x=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Sticker']['num_sold'], y=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Sticker']['tmax'])
sns.regplot(x=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Sticker']['num_sold'], y=df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Sticker']['tmin'])

fig.add_subplot(334)
plt.title('Kaggle Hat sales in Norway')
sns.regplot(x=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Hat']['num_sold'], y=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Hat']['tavg'])
sns.regplot(x=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Hat']['num_sold'], y=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Hat']['tmax'])
sns.regplot(x=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Hat']['num_sold'], y=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Hat']['tmin'])

fig.add_subplot(335)
plt.title('Kaggle Mug sales in Norway')
sns.regplot(x=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Mug']['num_sold'], y=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Mug']['tavg'])
sns.regplot(x=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Mug']['num_sold'], y=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Mug']['tmax'])
sns.regplot(x=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Mug']['num_sold'], y=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Mug']['tmin'])

fig.add_subplot(336)
plt.title('Kaggle Sticker sales in Norway')
sns.regplot(x=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Sticker']['num_sold'], y=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Sticker']['tavg'])
sns.regplot(x=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Sticker']['num_sold'], y=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Sticker']['tmax'])
sns.regplot(x=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Sticker']['num_sold'], y=df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Sticker']['tmin'])

fig.add_subplot(337)
plt.title('Kaggle Hat sales in Sweden')
sns.regplot(x=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Hat']['num_sold'], y=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Hat']['tavg'])
sns.regplot(x=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Hat']['num_sold'], y=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Hat']['tmax'])
sns.regplot(x=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Hat']['num_sold'], y=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Hat']['tmin'])

fig.add_subplot(338)
plt.title('Kaggle Mug sales in Sweden')
sns.regplot(x=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Mug']['num_sold'], y=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Mug']['tavg'])
sns.regplot(x=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Mug']['num_sold'], y=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Mug']['tmax'])
sns.regplot(x=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Mug']['num_sold'], y=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Mug']['tmin'])

fig.add_subplot(339)
plt.title('Kaggle Sticker sales in Sweden')
sns.regplot(x=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Sticker']['num_sold'], y=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Sticker']['tavg'])
sns.regplot(x=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Sticker']['num_sold'], y=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Sticker']['tmax'])
sns.regplot(x=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Sticker']['num_sold'], y=df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Sticker']['tmin'])


# The last few plots are a bit too busy, but the regression lines make interpreting them easier.
# 
# * Precipitation does not seem to have a meaningful effect on product sales in any of the countries.
# * All product sales increase with the accumulation of snow depth, though Stickers less so.
# * All product sales decrease with the increase of temperature.
# 
# Next, let's look at the correlation matrices by country and product.

# In[12]:


weather_features += ['num_sold']
df_train[weather_features].corr()


# Looking at the last column, we can see that product sales are positively correlated with precipitation and snow depth, and inversely correlated with temperatures. Both correlations are weak. Let's plot these relationships on heatmaps.

# In[13]:


fig = plt.figure(figsize=(20, 18))

fig.add_subplot(331)
plt.title('Finland - Kaggle Hat')
sns.heatmap(df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Hat'][weather_features].corr())

fig.add_subplot(332)
plt.title('Finland - Kaggle Mug')
sns.heatmap(df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Mug'][weather_features].corr())

fig.add_subplot(333)
plt.title('Finland - Kaggle Sticker')
sns.heatmap(df_train[df_train['country'] == 'Finland'][df_train['product'] == 'Kaggle Sticker'][weather_features].corr())

fig.add_subplot(334)
plt.title('Norway - Kaggle Hat')
sns.heatmap(df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Hat'][weather_features].corr())

fig.add_subplot(335)
plt.title('Norway - Kaggle Mug')
sns.heatmap(df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Mug'][weather_features].corr())

fig.add_subplot(336)
plt.title('Norway - Kaggle Sticker')
sns.heatmap(df_train[df_train['country'] == 'Norway'][df_train['product'] == 'Kaggle Sticker'][weather_features].corr())

fig.add_subplot(337)
plt.title('Sweden - Kaggle Hat')
sns.heatmap(df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Hat'][weather_features].corr())

fig.add_subplot(338)
plt.title('Sweden - Kaggle Mug')
sns.heatmap(df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Mug'][weather_features].corr())

fig.add_subplot(339)
plt.title('Sweden - Kaggle Sticker')
sns.heatmap(df_train[df_train['country'] == 'Sweden'][df_train['product'] == 'Kaggle Sticker'][weather_features].corr())


# The weak relationship (coral) between snow depth and sales is interesting, but could be just seasonality/coincidence.
# We're indifferent to the strong relationship (navy) between snow depth and temperatures of course.

# # Model comparisons
# 
# Arguably the ultimate test for the usefulness of features is seeing how a model performs with and without them. In this section, we will do a simple comparison training a model by itself and then with the weather features added. We will average our model predictions over 5 seeds to get more consistent results. We can standardize our weather features and encode our categoricals at this point. 

# In[14]:


weather_features = ['precipitation', 'snow_depth', 'tavg', 'tmax', 'tmin']

scaler = StandardScaler()
df_train[weather_features] = scaler.fit_transform(df_train[weather_features])
df_test[weather_features] = scaler.transform(df_test[weather_features])

cat_features = ['country', 'store', 'product']

ordinal_encoder = OrdinalEncoder()
df_train[cat_features] = ordinal_encoder.fit_transform(df_train[cat_features])
df_test[cat_features] = ordinal_encoder.fit_transform(df_test[cat_features])

tss = TimeSeriesSplit(n_splits=4)


# ### Model 1 - No weather features

# In[15]:


# Excluding all weather features
features = [c for c in df_test.columns if c not in ('row_id', 'date', 'precipitation', 'snow_depth', 'tavg', 'tmax', 'tmin')]
seeds = 5 # set the number of seeds you want to average

seed_valid_preds = []
seed_scores = []
seed_test_preds = []

for s in range(seeds):
    seed_valid_ids = []
    fold_valid_preds = {}
    fold_scores = []
    fold_test_preds = []

    for fold, (i_train, i_test) in enumerate(tss.split(df_train)):
        X_train = df_train.iloc[i_train]
        y_train = df_train['num_sold'].iloc[i_train]

        X_test = df_test.copy()

        X_valid = df_train.iloc[i_test]
        y_valid = df_train['num_sold'].iloc[i_test]

        fold_valid_ids = X_valid.row_id.values.tolist()
        seed_valid_ids += fold_valid_ids

        X_train = X_train[features]
        X_valid = X_valid[features]

        model = CatBoostRegressor(iterations=5000,
                                  loss_function='MAE',
                                  eval_metric='SMAPE',
                                  random_seed=s)

        model.fit(X_train,
                  y_train,
                  early_stopping_rounds=200,
                  eval_set=[(X_valid, y_valid)],
                  verbose=0)

        fold_valid_pred = model.predict(X_valid)
        fold_valid_preds.update(dict(zip(fold_valid_ids, fold_valid_pred)))

        fold_test_pred = model.predict(X_test)
        fold_test_preds.append(fold_test_pred)

        fold_score = np.mean(np.abs(fold_valid_pred - y_valid) / ((np.abs(y_valid) + np.abs(fold_valid_pred)) / 2)) * 100
        fold_scores.append(fold_score)
        print(f'Seed {s} fold {fold} SMAPE: {fold_score}')

    print(f'Seed {s} SMAPE {np.mean(fold_scores)}, std {np.std(fold_scores)}')
    
    seed_valid_pred = np.array(list(fold_valid_preds.values()))
    seed_valid_preds.append(seed_valid_pred)
        
    seed_score = np.mean(fold_scores)
    seed_scores.append(seed_score)

    seed_test_pred = np.mean(np.column_stack(fold_test_preds), axis=1)
    seed_test_preds.append(seed_test_pred)
    
print(f'SMAPE of {s+1} seeds: {np.mean(seed_scores)}, std {np.std(seed_scores)}')

# Submission
sample_submission.columns = ['row_id', 'num_sold']
sample_submission.num_sold = np.round(np.mean(np.column_stack(seed_test_preds), axis=1))
sample_submission.to_csv('submission_baseline.csv', index=False)


# ### Model 2 - Precipitation

# In[16]:


# Excluding all weather features but precipitation
features = [c for c in df_test.columns if c not in ('row_id', 'date', 'snow_depth', 'tavg', 'tmax', 'tmin')]
seeds = 5

seed_valid_preds = []
seed_scores = []
seed_test_preds = []

for s in range(seeds):
    seed_valid_ids = []
    fold_valid_preds = {}
    fold_scores = []
    fold_test_preds = []

    for fold, (i_train, i_test) in enumerate(tss.split(df_train)):
        X_train = df_train.iloc[i_train]
        y_train = df_train['num_sold'].iloc[i_train]

        X_test = df_test.copy()

        X_valid = df_train.iloc[i_test]
        y_valid = df_train['num_sold'].iloc[i_test]

        fold_valid_ids = X_valid.row_id.values.tolist()
        seed_valid_ids += fold_valid_ids

        X_train = X_train[features]
        X_valid = X_valid[features]

        model = CatBoostRegressor(iterations=5000,
                                  loss_function='MAE',
                                  eval_metric='SMAPE',
                                  random_seed=s)
        model.fit(X_train,
                  y_train,
                  early_stopping_rounds=200,
                  eval_set=[(X_valid, y_valid)],
                  verbose=0)

        fold_valid_pred = model.predict(X_valid)
        fold_valid_preds.update(dict(zip(fold_valid_ids, fold_valid_pred)))

        fold_test_pred = model.predict(X_test)
        fold_test_preds.append(fold_test_pred)

        fold_score = np.mean(np.abs(fold_valid_pred - y_valid) / ((np.abs(y_valid) + np.abs(fold_valid_pred)) / 2)) * 100
        fold_scores.append(fold_score)
        print(f'Seed {s} fold {fold} SMAPE: {fold_score}')

    print(f'Seed {s} SMAPE {np.mean(fold_scores)}, std {np.std(fold_scores)}')
    
    seed_valid_pred = np.array(list(fold_valid_preds.values()))
    seed_valid_preds.append(seed_valid_pred)
        
    seed_score = np.mean(fold_scores)
    seed_scores.append(seed_score)

    seed_test_pred = np.mean(np.column_stack(fold_test_preds), axis=1)
    seed_test_preds.append(seed_test_pred)
    
print(f'SMAPE of {s+1} seeds: {np.mean(seed_scores)}, std {np.std(seed_scores)}')

# Submission
sample_submission.columns = ['row_id', 'num_sold']
sample_submission.num_sold = np.round(np.mean(np.column_stack(seed_test_preds), axis=1))
sample_submission.to_csv('submission_precipitation.csv', index=False)


# ### Model 3 - Snow depth

# In[17]:


# Excluding all weather features but snow_depth
features = [c for c in df_test.columns if c not in ('row_id', 'date', 'precipitation', 'tavg', 'tmax', 'tmin')]
seeds = 5

seed_valid_preds = []
seed_scores = []
seed_test_preds = []

for s in range(seeds):
    seed_valid_ids = []
    fold_valid_preds = {}
    fold_scores = []
    fold_test_preds = []

    for fold, (i_train, i_test) in enumerate(tss.split(df_train)):
        X_train = df_train.iloc[i_train]
        y_train = df_train['num_sold'].iloc[i_train]

        X_test = df_test.copy()

        X_valid = df_train.iloc[i_test]
        y_valid = df_train['num_sold'].iloc[i_test]

        fold_valid_ids = X_valid.row_id.values.tolist()
        seed_valid_ids += fold_valid_ids

        X_train = X_train[features]
        X_valid = X_valid[features]

        model = CatBoostRegressor(iterations=5000,
                                  loss_function='MAE',
                                  eval_metric='SMAPE',
                                  random_seed=s)
        model.fit(X_train,
                  y_train,
                  early_stopping_rounds=200,
                  eval_set=[(X_valid, y_valid)],
                  verbose=0)

        fold_valid_pred = model.predict(X_valid)
        fold_valid_preds.update(dict(zip(fold_valid_ids, fold_valid_pred)))

        fold_test_pred = model.predict(X_test)
        fold_test_preds.append(fold_test_pred)

        fold_score = np.mean(np.abs(fold_valid_pred - y_valid) / ((np.abs(y_valid) + np.abs(fold_valid_pred)) / 2)) * 100
        fold_scores.append(fold_score)
        print(f'Seed {s} fold {fold} SMAPE: {fold_score}')

    print(f'Seed {s} SMAPE {np.mean(fold_scores)}, std {np.std(fold_scores)}')
    
    seed_valid_pred = np.array(list(fold_valid_preds.values()))
    seed_valid_preds.append(seed_valid_pred)
        
    seed_score = np.mean(fold_scores)
    seed_scores.append(seed_score)

    seed_test_pred = np.mean(np.column_stack(fold_test_preds), axis=1)
    seed_test_preds.append(seed_test_pred)
    
print(f'SMAPE of {s+1} seeds: {np.mean(seed_scores)}, std {np.std(seed_scores)}')

# Submission
sample_submission.columns = ['row_id', 'num_sold']
sample_submission.num_sold = np.round(np.mean(np.column_stack(seed_test_preds), axis=1))
sample_submission.to_csv('submission_snow.csv', index=False)


# ### Model 4 - Temperature average

# In[18]:


# Excluding all weather features but tavg
features = [c for c in df_test.columns if c not in ('row_id', 'date', 'precipitation', 'snow_depth', 'tmax', 'tmin')]
seeds = 5

seed_valid_preds = []
seed_scores = []
seed_test_preds = []

for s in range(seeds):
    seed_valid_ids = []
    fold_valid_preds = {}
    fold_scores = []
    fold_test_preds = []

    for fold, (i_train, i_test) in enumerate(tss.split(df_train)):
        X_train = df_train.iloc[i_train]
        y_train = df_train['num_sold'].iloc[i_train]

        X_test = df_test.copy()

        X_valid = df_train.iloc[i_test]
        y_valid = df_train['num_sold'].iloc[i_test]

        fold_valid_ids = X_valid.row_id.values.tolist()
        seed_valid_ids += fold_valid_ids

        X_train = X_train[features]
        X_valid = X_valid[features]

        model = CatBoostRegressor(iterations=5000,
                                  loss_function='MAE',
                                  eval_metric='SMAPE',
                                  random_seed=s)
        model.fit(X_train,
                  y_train,
                  early_stopping_rounds=200,
                  eval_set=[(X_valid, y_valid)],
                  verbose=0)

        fold_valid_pred = model.predict(X_valid)
        fold_valid_preds.update(dict(zip(fold_valid_ids, fold_valid_pred)))

        fold_test_pred = model.predict(X_test)
        fold_test_preds.append(fold_test_pred)

        fold_score = np.mean(np.abs(fold_valid_pred - y_valid) / ((np.abs(y_valid) + np.abs(fold_valid_pred)) / 2)) * 100
        fold_scores.append(fold_score)
        print(f'Seed {s} fold {fold} SMAPE: {fold_score}')

    print(f'Seed {s} SMAPE {np.mean(fold_scores)}, std {np.std(fold_scores)}')
    
    seed_valid_pred = np.array(list(fold_valid_preds.values()))
    seed_valid_preds.append(seed_valid_pred)
        
    seed_score = np.mean(fold_scores)
    seed_scores.append(seed_score)

    seed_test_pred = np.mean(np.column_stack(fold_test_preds), axis=1)
    seed_test_preds.append(seed_test_pred)
    
print(f'SMAPE of {s+1} seeds: {np.mean(seed_scores)}, std {np.std(seed_scores)}')

# Submission
sample_submission.columns = ['row_id', 'num_sold']
sample_submission.num_sold = np.round(np.mean(np.column_stack(seed_test_preds), axis=1))
sample_submission.to_csv('submission_tavg.csv', index=False)


# ### Model 5 - Temperature maximum

# In[19]:


# Excluding all weather features but tmax
features = [c for c in df_test.columns if c not in ('row_id', 'date', 'precipitation', 'snow_depth', 'tavg', 'tmin')]
seeds = 5

seed_valid_preds = []
seed_scores = []
seed_test_preds = []

for s in range(seeds):
    seed_valid_ids = []
    fold_valid_preds = {}
    fold_scores = []
    fold_test_preds = []

    for fold, (i_train, i_test) in enumerate(tss.split(df_train)):
        X_train = df_train.iloc[i_train]
        y_train = df_train['num_sold'].iloc[i_train]

        X_test = df_test.copy()

        X_valid = df_train.iloc[i_test]
        y_valid = df_train['num_sold'].iloc[i_test]

        fold_valid_ids = X_valid.row_id.values.tolist()
        seed_valid_ids += fold_valid_ids

        X_train = X_train[features]
        X_valid = X_valid[features]

        model = CatBoostRegressor(iterations=5000,
                                  loss_function='MAE',
                                  eval_metric='SMAPE',
                                  random_seed=s)
        model.fit(X_train,
                  y_train,
                  early_stopping_rounds=200,
                  eval_set=[(X_valid, y_valid)],
                  verbose=0)

        fold_valid_pred = model.predict(X_valid)
        fold_valid_preds.update(dict(zip(fold_valid_ids, fold_valid_pred)))

        fold_test_pred = model.predict(X_test)
        fold_test_preds.append(fold_test_pred)

        fold_score = np.mean(np.abs(fold_valid_pred - y_valid) / ((np.abs(y_valid) + np.abs(fold_valid_pred)) / 2)) * 100
        fold_scores.append(fold_score)
        print(f'Seed {s} fold {fold} SMAPE: {fold_score}')

    print(f'Seed {s} SMAPE {np.mean(fold_scores)}, std {np.std(fold_scores)}')
    
    seed_valid_pred = np.array(list(fold_valid_preds.values()))
    seed_valid_preds.append(seed_valid_pred)
        
    seed_score = np.mean(fold_scores)
    seed_scores.append(seed_score)

    seed_test_pred = np.mean(np.column_stack(fold_test_preds), axis=1)
    seed_test_preds.append(seed_test_pred)
    
print(f'SMAPE of {s+1} seeds: {np.mean(seed_scores)}, std {np.std(seed_scores)}')

# Submission
sample_submission.columns = ['row_id', 'num_sold']
sample_submission.num_sold = np.round(np.mean(np.column_stack(seed_test_preds), axis=1))
sample_submission.to_csv('submission_tmax.csv', index=False)


# ### Model 6 - Temperature minimum

# In[20]:


# Excluding all weather features but tmin
features = [c for c in df_test.columns if c not in ('row_id', 'date', 'precipitation', 'snow_depth', 'tavg', 'tmax')]
seeds = 5

seed_valid_preds = []
seed_scores = []
seed_test_preds = []

for s in range(seeds):
    seed_valid_ids = []
    fold_valid_preds = {}
    fold_scores = []
    fold_test_preds = []

    for fold, (i_train, i_test) in enumerate(tss.split(df_train)):
        X_train = df_train.iloc[i_train]
        y_train = df_train['num_sold'].iloc[i_train]

        X_test = df_test.copy()

        X_valid = df_train.iloc[i_test]
        y_valid = df_train['num_sold'].iloc[i_test]

        fold_valid_ids = X_valid.row_id.values.tolist()
        seed_valid_ids += fold_valid_ids

        X_train = X_train[features]
        X_valid = X_valid[features]

        model = CatBoostRegressor(iterations=5000,
                                  loss_function='MAE',
                                  eval_metric='SMAPE',
                                  random_seed=s)
        model.fit(X_train,
                  y_train,
                  early_stopping_rounds=200,
                  eval_set=[(X_valid, y_valid)],
                  verbose=0)

        fold_valid_pred = model.predict(X_valid)
        fold_valid_preds.update(dict(zip(fold_valid_ids, fold_valid_pred)))

        fold_test_pred = model.predict(X_test)
        fold_test_preds.append(fold_test_pred)

        fold_score = np.mean(np.abs(fold_valid_pred - y_valid) / ((np.abs(y_valid) + np.abs(fold_valid_pred)) / 2)) * 100
        fold_scores.append(fold_score)
        print(f'Seed {s} fold {fold} SMAPE: {fold_score}')

    print(f'Seed {s} SMAPE {np.mean(fold_scores)}, std {np.std(fold_scores)}')
    
    seed_valid_pred = np.array(list(fold_valid_preds.values()))
    seed_valid_preds.append(seed_valid_pred)
        
    seed_score = np.mean(fold_scores)
    seed_scores.append(seed_score)

    seed_test_pred = np.mean(np.column_stack(fold_test_preds), axis=1)
    seed_test_preds.append(seed_test_pred)
    
print(f'SMAPE of {s+1} seeds: {np.mean(seed_scores)}, std {np.std(seed_scores)}')

# Submission
sample_submission.columns = ['row_id', 'num_sold']
sample_submission.num_sold = np.round(np.mean(np.column_stack(seed_test_preds), axis=1))
sample_submission.to_csv('submission_tmin.csv', index=False)


# ### Model 7 - All weather features

# In[21]:


# Including all weather features
features = [c for c in df_test.columns if c not in ('row_id', 'date')]
seeds = 5

seed_valid_preds = []
seed_scores = []
seed_test_preds = []

for s in range(seeds):
    seed_valid_ids = []
    fold_valid_preds = {}
    fold_scores = []
    fold_test_preds = []

    for fold, (i_train, i_test) in enumerate(tss.split(df_train)):
        X_train = df_train.iloc[i_train]
        y_train = df_train['num_sold'].iloc[i_train]

        X_test = df_test.copy()

        X_valid = df_train.iloc[i_test]
        y_valid = df_train['num_sold'].iloc[i_test]

        fold_valid_ids = X_valid.row_id.values.tolist()
        seed_valid_ids += fold_valid_ids

        X_train = X_train[features]
        X_valid = X_valid[features]

        model = CatBoostRegressor(iterations=5000,
                                  loss_function='MAE',
                                  eval_metric='SMAPE',
                                  random_seed=s)
        model.fit(X_train,
                  y_train,
                  early_stopping_rounds=200,
                  eval_set=[(X_valid, y_valid)],
                  verbose=0)

        fold_valid_pred = model.predict(X_valid)
        fold_valid_preds.update(dict(zip(fold_valid_ids, fold_valid_pred)))

        fold_test_pred = model.predict(X_test)
        fold_test_preds.append(fold_test_pred)

        fold_score = np.mean(np.abs(fold_valid_pred - y_valid) / ((np.abs(y_valid) + np.abs(fold_valid_pred)) / 2)) * 100
        fold_scores.append(fold_score)
        print(f'Seed {s} fold {fold} SMAPE: {fold_score}')

    print(f'Seed {s} SMAPE {np.mean(fold_scores)}, std {np.std(fold_scores)}')
    
    seed_valid_pred = np.array(list(fold_valid_preds.values()))
    seed_valid_preds.append(seed_valid_pred)
        
    seed_score = np.mean(fold_scores)
    seed_scores.append(seed_score)

    seed_test_pred = np.mean(np.column_stack(fold_test_preds), axis=1)
    seed_test_preds.append(seed_test_pred)
    
print(f'SMAPE of {s+1} seeds: {np.mean(seed_scores)}, std {np.std(seed_scores)}')

# Submission
sample_submission.columns = ['row_id', 'num_sold']
sample_submission.num_sold = np.round(np.mean(np.column_stack(seed_test_preds), axis=1))
sample_submission.to_csv('submission_all_weather.csv', index=False)


# ### Model 8 - All but tmin

# In[22]:


# Including all weather features but tmin
features = [c for c in df_test.columns if c not in ('row_id', 'date', 'tmin')]
seeds = 5

seed_valid_preds = []
seed_scores = []
seed_test_preds = []

for s in range(seeds):
    seed_valid_ids = []
    fold_valid_preds = {}
    fold_scores = []
    fold_test_preds = []

    for fold, (i_train, i_test) in enumerate(tss.split(df_train)):
        X_train = df_train.iloc[i_train]
        y_train = df_train['num_sold'].iloc[i_train]

        X_test = df_test.copy()

        X_valid = df_train.iloc[i_test]
        y_valid = df_train['num_sold'].iloc[i_test]

        fold_valid_ids = X_valid.row_id.values.tolist()
        seed_valid_ids += fold_valid_ids

        X_train = X_train[features]
        X_valid = X_valid[features]

        model = CatBoostRegressor(iterations=5000,
                                  loss_function='MAE',
                                  eval_metric='SMAPE',
                                  random_seed=s)
        model.fit(X_train,
                  y_train,
                  early_stopping_rounds=200,
                  eval_set=[(X_valid, y_valid)],
                  verbose=0)

        fold_valid_pred = model.predict(X_valid)
        fold_valid_preds.update(dict(zip(fold_valid_ids, fold_valid_pred)))

        fold_test_pred = model.predict(X_test)
        fold_test_preds.append(fold_test_pred)

        fold_score = np.mean(np.abs(fold_valid_pred - y_valid) / ((np.abs(y_valid) + np.abs(fold_valid_pred)) / 2)) * 100
        fold_scores.append(fold_score)
        print(f'Seed {s} fold {fold} SMAPE: {fold_score}')

    print(f'Seed {s} SMAPE {np.mean(fold_scores)}, std {np.std(fold_scores)}')
    
    seed_valid_pred = np.array(list(fold_valid_preds.values()))
    seed_valid_preds.append(seed_valid_pred)
        
    seed_score = np.mean(fold_scores)
    seed_scores.append(seed_score)

    seed_test_pred = np.mean(np.column_stack(fold_test_preds), axis=1)
    seed_test_preds.append(seed_test_pred)
    
print(f'SMAPE of {s+1} seeds: {np.mean(seed_scores)}, std {np.std(seed_scores)}')

# Submission
sample_submission.columns = ['row_id', 'num_sold']
sample_submission.num_sold = np.round(np.mean(np.column_stack(seed_test_preds), axis=1))
sample_submission.to_csv('submission_select_weather.csv', index=False)


# # Feature engineering
# 
# We are going to create a few new features from what we have in the hopes of a score improvement. These are:
# * Diurnal air temperature variation - this is the difference of the maximum and minimum daily temperatures
# * Lag and forecast features of our 5 weather features - these are backward and forward looking features
# * Moving averages
# * Daily differences from the moving averages
# 
# I want to thank @khbreslauer and @cv13j0 for their feature engineering ideas.

# In[23]:


# Adding diurnal air temperature variation.
df_train['temp_var'] = df_train['tmax'] - df_train['tmin']
df_test['temp_var'] = df_test['tmax'] - df_test['tmin']

# Adding lag and forecast features.
for day in range(1, 4):
    for feature in weather_features:
        df_train[f'{feature}_{day}_day_lag'] = df_train.groupby(['country', 'store', 'product'])[feature].shift(day)
        df_test[f'{feature}_{day}_day_lag'] = df_test.groupby(['country', 'store', 'product'])[feature].shift(day)
        df_train[f'{feature}_{day}_day_forecast'] = df_train.groupby(['country', 'store', 'product'])[feature].shift(-day)
        df_test[f'{feature}_{day}_day_forecast'] = df_test.groupby(['country', 'store', 'product'])[feature].shift(-day)

# Adding moving averages. This doesn't make much sense when doing for snow_depth, but let's see.

days = [3, 7, 15, 30]
for day in days:
    for feature in weather_features:
        df_train[f'{feature}_{day}_day_mov_avg'] = df_train.groupby([
            'country',
            'store',
            'product'])[feature].transform(lambda x: x.shift(1).rolling(
            window=day,
            min_periods=0,
            center=False,
        ).mean())
        df_test[f'{feature}_{day}_day_mov_avg'] = df_test.groupby([
            'country',
            'store',
            'product'])[feature].transform(lambda x: x.shift(1).rolling(
            window=day,
            min_periods=0,
            center=False,
        ).mean())
        
# Adding differences from the moving averages to identify sudden dips and hikes.
df_train['precipitation_diff_3'] = df_train['precipitation'] - df_train['precipitation_3_day_mov_avg']
df_train['precipitation_diff_7'] = df_train['precipitation'] - df_train['precipitation_7_day_mov_avg']
df_train['snow_depth_diff_3'] = df_train['snow_depth'] - df_train['snow_depth_3_day_mov_avg']
df_train['snow_depth_diff_7'] = df_train['snow_depth'] - df_train['snow_depth_3_day_mov_avg']
df_train['tavg_diff_3'] = df_train['tavg'] - df_train['tavg_3_day_mov_avg']
df_train['tavg_diff_7'] = df_train['tavg'] - df_train['tavg_7_day_mov_avg']
df_train['tmax_diff_3'] = df_train['tmax'] - df_train['tmax_3_day_mov_avg']
df_train['tmax_diff_7'] = df_train['tmax'] - df_train['tmax_7_day_mov_avg']
df_train['tmin_diff_3'] = df_train['tmin'] - df_train['tmin_3_day_mov_avg']
df_train['tmin_diff_7'] = df_train['tmin'] - df_train['tmin_7_day_mov_avg']

df_test['precipitation_diff_3'] = df_test['precipitation'] - df_test['precipitation_3_day_mov_avg']
df_test['precipitation_diff_7'] = df_test['precipitation'] - df_test['precipitation_7_day_mov_avg']
df_test['snow_depth_diff_3'] = df_test['snow_depth'] - df_test['snow_depth_3_day_mov_avg']
df_test['snow_depth_diff_7'] = df_test['snow_depth'] - df_test['snow_depth_3_day_mov_avg']
df_test['tavg_diff_3'] = df_test['tavg'] - df_test['tavg_3_day_mov_avg']
df_test['tavg_diff_7'] = df_test['tavg'] - df_test['tavg_7_day_mov_avg']
df_test['tmax_diff_3'] = df_test['tmax'] - df_test['tmax_3_day_mov_avg']
df_test['tmax_diff_7'] = df_test['tmax'] - df_test['tmax_7_day_mov_avg']
df_test['tmin_diff_3'] = df_test['tmin'] - df_test['tmin_3_day_mov_avg']
df_test['tmin_diff_7'] = df_test['tmin'] - df_test['tmin_7_day_mov_avg']

# Filling missing values is not important with Catboost. If you're using a different model, you may need to.

# country_features = [c for c in df_test.columns if 'Finland' in c or 'Norway' in c or 'Sweden' in c]
# df_test[country_features].fillna(0)


# In[24]:


# for feature in features:
#     fig = plt.figure(figsize=(4, 3))
#     sns.regplot(x=df_train['num_sold'], y=df_train[feature])


# All our engineered features are as weakly correlated with the target as the base weather features. These deteriorate both our CV and LB scores significantly and are not worth keeping.

# # Feature importances

# ### Model 9 - All weather features including engineered features

# In[25]:


# Including all weather features
features = [c for c in df_test.columns if c not in ('row_id', 'date')]
seeds = 5

seed_valid_preds = []
seed_scores = []
seed_test_preds = []

for s in range(seeds):
    seed_valid_ids = []
    fold_valid_preds = {}
    fold_scores = []
    fold_test_preds = []

    for fold, (i_train, i_test) in enumerate(tss.split(df_train)):
        X_train = df_train.iloc[i_train]
        y_train = df_train['num_sold'].iloc[i_train]

        X_test = df_test.copy()

        X_valid = df_train.iloc[i_test]
        y_valid = df_train['num_sold'].iloc[i_test]

        fold_valid_ids = X_valid.row_id.values.tolist()
        seed_valid_ids += fold_valid_ids

        X_train = X_train[features]
        X_valid = X_valid[features]

        model = CatBoostRegressor(iterations=5000,
                                  loss_function='MAE',
                                  eval_metric='SMAPE',
                                  random_seed=s)
        model.fit(X_train,
                  y_train,
                  early_stopping_rounds=200,
                  eval_set=[(X_valid, y_valid)],
                  verbose=0)

        fold_valid_pred = model.predict(X_valid)
        fold_valid_preds.update(dict(zip(fold_valid_ids, fold_valid_pred)))

        fold_test_pred = model.predict(X_test)
        fold_test_preds.append(fold_test_pred)

        fold_score = np.mean(np.abs(fold_valid_pred - y_valid) / ((np.abs(y_valid) + np.abs(fold_valid_pred)) / 2)) * 100
        fold_scores.append(fold_score)
        print(f'Seed {s} fold {fold} SMAPE: {fold_score}')

    print(f'Seed {s} SMAPE {np.mean(fold_scores)}, std {np.std(fold_scores)}')
    
    seed_valid_pred = np.array(list(fold_valid_preds.values()))
    seed_valid_preds.append(seed_valid_pred)
        
    seed_score = np.mean(fold_scores)
    seed_scores.append(seed_score)

    seed_test_pred = np.mean(np.column_stack(fold_test_preds), axis=1)
    seed_test_preds.append(seed_test_pred)
    
print(f'SMAPE of {s+1} seeds: {np.mean(seed_scores)}, std {np.std(seed_scores)}')

# Submission
sample_submission.columns = ['row_id', 'num_sold']
sample_submission.num_sold = np.round(np.mean(np.column_stack(seed_test_preds), axis=1))
sample_submission.to_csv('submission_all_weather_fe.csv', index=False)


# In[26]:


feature_importances = model.get_feature_importance(prettified=True)
feature_importances.head(50)


# In[27]:


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df_train)
shap.summary_plot(shap_values, df_train)


# Our weather features are not deemed important by either method and are scored consistently low by Shap.

# # Final results
# 
# 

# | Features | SMAPE | Public LB |
# | --- | --- | --- |
# | No weather features | 7.4664912617172380 | 6.00421 |
# | precipitation | 7.5020382959417430 | n/a |
# | snow_depth | 7.4622390996710210 | 6.11984 |
# | tavg | 7.5011167684512190 | n/a |
# | tmax | 7.4754759828989590 | 6.11840 |
# | tmin | 7.5467655794719875 | n/a |
# | All weather features | 7.4596059127535440 | 6.25309 |
# | All but tmin (best) | 7.4480535771435420 | 6.21853 |
# 
# Unfortunately, the features just make our predictions worse.
# 
# Thank you for reading this notebook. Please share with me if you find a use for this dataset!
