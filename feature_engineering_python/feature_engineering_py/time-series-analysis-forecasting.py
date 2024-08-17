#!/usr/bin/env python
# coding: utf-8

# <div style="padding:20px;color:black;margin:0;font-size:350%;text-align:center;display:fill;border-radius:5px;background-color:#aaf0f0;overflow:hidden;font-weight:700;border: 5px solid #21a3a3;"> ⌛️ Time Series Analysis</div>

# # ⌛️ 1. Time Series Analysis

# In mathematics, a time series is a series of data points indexed in time order. Most commonly, a time series is a sequence taken at successive equally spaced points in time. Thus it is a sequence of discrete-time data. [*Wikipedia*](https://en.wikipedia.org/wiki/Time_series)
# 
# Time series analysis is a specific way of analyzing a sequence of data points collected over an interval of time. In time series analysis, analysts record data points at consistent intervals over a set period of time rather than just recording the data points intermittently or randomly. [*Tableau*](https://www.tableau.com/learn/articles/time-series-analysis)

# ![](https://www.researchgate.net/profile/Ben-Fulcher/publication/320032800/figure/fig1/AS:542676766007296@1506395631636/Time-series-modeling-and-forecasting-The-figure-shows-a-uniformly-sampled-time-series.png)

# - A Time-Series represents a series of time-based orders. It would be Years, Months, Weeks, Days, Horus, Minutes, and Seconds
# - A time series is an observation from the sequence of discrete-time of successive intervals.
# - A time series is a running chart.
# - The time variable/feature is the independent variable and supports the target variable to predict the results.
# - Time Series Analysis (TSA) is used in different fields for time-based predictions – like Weather Forecasting, Financial, Signal processing, Engineering domain – Control Systems, Communications Systems.
# - Since TSA involves producing the set of information in a particular sequence, it makes a distinct from spatial and other analyses.
# - Using AR, MA, ARMA, and ARIMA models, we could predict the future.

# ![](https://www.simplilearn.com/ice9/free_resources_article_thumb/Time_Series_Analysis_In_Python_2.png)

# <div style="padding:20px;color:black;margin:0;font-size:250%;text-align:center;display:fill;border-radius:5px;background-color:#aaf0f0;overflow:hidden;font-weight:700;border: 5px solid #21a3a3;"> 📒 Important Terms to understand for Time Series Analysis </div>

# # 📒 2. Important Terms to understand for Time Series Analysis 
# 
# A Time-Series is a sequence of data points collected at different timestamps. These are essentially successive measurements collected from the same data source at the same time interval. Further, we can use these chronologically gathered readings to monitor trends and changes over time. The time-series models can be univariate or multivariate. The univariate time series models are implemented when the dependent variable is a single time series, like room temperature measurement from a single sensor. On the other hand, a multivariate time series model can be used when there are multiple dependent variables, i.e., the output depends on more than one series. An example for the multivariate time-series model could be modelling the GDP, inflation, and unemployment together as these variables are linked to each other.
# 
# ### 1. Stationary and Non-Stationary Time Series
# **Stationarity is a property of a time series.** A stationary series is one where the values of the series is not a function of time. That is, the statistical properties of the series like mean, variance and autocorrelation are constant over time. Autocorrelation of the series is nothing but the correlation of the series with its previous values, more on this coming up. **A stationary time series id devoid of seasonal effects as well.**
# 
# ### 2. Trend
# The trend shows a general direction of the time series data over a long period of time. A trend can be increasing(upward), decreasing(downward), or horizontal(stationary).
# 
# ### 3.Seasonality 
# The seasonality component exhibits a trend that repeats with respect to timing, direction, and magnitude. Some examples include an increase in water consumption in summer due to hot weather conditions.
# 
# ### 4. Cyclical Component
# These are the trends with no set repetition over a particular period of time. A cycle refers to the period of ups and downs, booms and slums of a time series, mostly observed in business cycles. These cycles do not exhibit a seasonal variation but generally occur over a time period of 3 to 12 years depending on the nature of the time series.
# 
# ###  5. Irregular Variation 
# These are the fluctuations in the time series data which become evident when trend and cyclical variations are removed. These variations are unpredictable, erratic, and may or may not be random.
# 
# ###  6. ETS Decomposition
# ETS Decomposition is used to separate different components of a time series. The term ETS stands for Error, Trend and Seasonality.
# 
# ###  7. Dependence
# It refers to the association of two observations of the same variable at prior time periods.
# 
# ###  8. Differencing
# Differencing is used to make the series stationary and to control the auto-correlations. There may be some cases in time series analyses where we do not require differencing and over-differenced series can produce wrong estimates.
# 
# ###  9. Specification 
# It may involve the testing of the linear or non-linear relationships of dependent variables by using time series models such as ARIMA models.
# 
# ###  10. ARIMA 
# ARIMA stands for Auto Regressive Integrated Moving Average.

# ![](https://www.simplilearn.com/ice9/free_resources_article_thumb/Time_Series_Analysis_In_Python_3.png)

# <div style="padding:20px;color:black;margin:0;font-size:250%;text-align:center;display:fill;border-radius:5px;background-color:#aaf0f0;overflow:hidden;font-weight:700;border: 5px solid #21a3a3;"> 📋 Helpful Libraries in Python for Time Series</div>

# # 📋 3. Helpful Libraries in Python
# 
# - ### **1) Tsfresh**
# The name of this library, Tsfresh, is based on the acronym “Time Series Feature Extraction Based on Scalable Hypothesis Tests.” It is a Python package that automatically calculates and extracts several time series features for classification and regression tasks. Hence, this library is mainly used for feature engineering in time series problems and other packages like sklearn to analyze the time series.
# 
# - ### **2) Darts**
# Darts is another time series Python library developed by Unit8 for easy manipulation and forecasting of time series. This idea was to make darts as simple to use as sklearn for time-series. Darts attempts to smooth the overall process of using time series in machine learning. Darts has two models: Regression models (predicts output with time as input) and Forecasting models (predicts future output based on past values).
# 
# - ### **3)  Kats**
# Kats (Kits to Analyze Time Series) is an open-source Python library developed by researchers at Facebook (now Meta). This library is easy to use and is helpful for time series problems. This is due to its very light weighted library of generic time series analysis which allows to set up the models quicker without spending so much time processing time series and calculations in different models.
# 
# - ### **4) GreyKite**
# GreyKite is a time-series forecasting library released by LinkedIn to simplify prediction for data scientists. This library offers automation in forecasting tasks using the primary forecasting algorithm ‘Silverkite.’ This library also helps interpret outputs making it a go-to tool for most time-series forecasting projects.
# 
# - ### **5) AutoTS**
# AutoTS, another Python time series tool, stands for Automatic Time Series, quickly providing high-accuracy forecasts at scale. It offers many different forecasting models and functions directly compatible with pandas’ data frames. The models from this library can be used for deployment. Some noticeable features of this library are – it works well with both univariate and multivariate time series data, it can handle missing or messy data with outliers, it helps to identify the best time series forecasting model based on the input data type

# <div style="padding:20px;color:black;margin:0;font-size:250%;text-align:center;display:fill;border-radius:5px;background-color:#aaf0f0;overflow:hidden;font-weight:700;border: 5px solid #21a3a3;"> 📇 Data and Modules</div>

# # 📇 4. Data and Modules

# In[1]:


import numpy as np 
import pandas as pd 

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

import os
train=pd.read_csv("/kaggle/input/tabular-playground-series-sep-2022/train.csv")
test=pd.read_csv("/kaggle/input/tabular-playground-series-sep-2022/test.csv")


# In[2]:


train.head()


# <div style="padding:20px;color:black;margin:0;font-size:250%;text-align:center;display:fill;border-radius:5px;background-color:#aaf0f0;overflow:hidden;font-weight:700;border: 5px solid #21a3a3;"> 🧮 Data Vizualization</div>

# # 🧮 5. Data Vizualization

# In[3]:


sns.set(rc={'figure.figsize':(24,8)})
ax=sns.lineplot(data=train,x='date',y='num_sold',hue='product')
ax.axes.set_title("\nBasic Time Series of Sales\n",fontsize=20);


# -  We can spot some spikes here!

# ### There are several types of patterns that can be found.
# 
# ![](https://www.machinelearningplus.com/wp-content/uploads/2019/02/5_Patterns_in_Time_Series-min-865x207.png)

# <div style="padding:20px;color:black;margin:0;font-size:250%;text-align:center;display:fill;border-radius:5px;background-color:#aaf0f0;overflow:hidden;font-weight:700;border: 5px solid #21a3a3;"> 📉 Patterns Recognition - Trend</div>

# # 📉 6. Patterns Recognition - Trend

# A trend is observed when there is an increasing or decreasing slope observed in the time series.
# 
# - **Linear Trend** : Maintains a straight line.
# - **Polynomial Trend** : Shows polynomial curves.
# - **Exponential Trend** : Shows exponential growths or fall.
# - **Log-based Trend** : Shows log-based growths or fall.

# ![](https://www.oraylis.de/sites/default/files/styles/optimize/public/inline-images/2015_10_Rplot04_thumb.png?itok=rvwgk3N9)
# 
# - source: oraylis.de

# ## How to detrend a time series?
# 
# Detrending a time series is to remove the trend component from a time series. But how to extract the trend? There are multiple approaches.
# 
# - Subtract the line of best fit from the time series. The line of best fit may be obtained from a linear regression model with the time steps as the predictor. For more complex trends, you may want to use quadratic terms (x^2) in the model.
# - Subtract the trend component obtained from time series decomposition we saw earlier.
# - Subtract the mean
# - Apply a filter like Baxter-King filter(statsmodels.tsa.filters.bkfilter) or the Hodrick-Prescott Filter (statsmodels.tsa.filters.hpfilter) to remove the moving average trend lines or the cyclical components.

# In[4]:


from scipy import signal
df = train[(train['country']=='Belgium')&(train['product']=='Kaggle Advanced Techniques')&(train['store']=='KaggleMart')]
detrended = signal.detrend(df.num_sold.values)
plt.plot(detrended)


# <div style="padding:20px;color:black;margin:0;font-size:250%;text-align:center;display:fill;border-radius:5px;background-color:#aaf0f0;overflow:hidden;font-weight:700;border: 5px solid #21a3a3;"> 🗃 Patterns Recognition - Seasonality</div>

# # 🗃 7. Patterns Recognition - Seasonality
# A seasonality is observed when there is a distinct repeated pattern observed between regular intervals due to seasonal factors. It could be because of the month of the year, the day of the month, weekdays or even time of the day.
# 
# - **Yearly** : Example - Black Friday and chrismas Sales
# - **Monthly** : Example - We may find big sales on first week of the month as slalry is paid that time.

# ![](https://i2.wp.com/radacad.com/wp-content/uploads/2017/07/trendseasonal.png) 
# - source: radacad.com

# ## How to deseasonalize a time series?
# There are multiple approaches to deseasonalize a time series as well. Below are a few:
# 
# - 1. Take a moving average with length as the seasonal window. This will smoothen in series in the process.
# - 2. Seasonal difference the series (subtract the value of previous season from the current value)
# - 3. Divide the series by the seasonal index obtained from STL decomposition

# In[5]:


df.info()


# In[6]:


df = train[(train['country']=='Belgium')&(train['product']=='Kaggle Advanced Techniques')&(train['store']=='KaggleMart')]

df=df.set_index('date')
df.index = pd.to_datetime(df.index)

from statsmodels.tsa.seasonal import seasonal_decompose
result_mul = seasonal_decompose(df['num_sold'], model='addtive', extrapolate_trend='freq')
deseasonalized = df.num_sold.values / result_mul.seasonal
plt.plot(deseasonalized)
plt.title('Deseasonalized', fontsize=16)
plt.plot()


# ## How to test for seasonality of a time series?
# 
# The common way is to plot the series and check for repeatable patterns in fixed time intervals. So, the types of seasonality is determined by the clock or the calendar:
# - Hour of day
# - Day of month
# - Weekly
# - Monthly
# - Yearly

# In[7]:


from pandas.plotting import autocorrelation_plot
df = train[(train['country']=='France')&(train['product']=='Kaggle Advanced Techniques')&(train['store']=='KaggleMart')]

plt.rcParams.update({'figure.figsize':(9,5), 'figure.dpi':120})
autocorrelation_plot(df.num_sold.tolist())
plt.title('Autocorrelation', fontsize=16)
plt.plot()


# <div style="padding:20px;color:black;margin:0;font-size:250%;text-align:center;display:fill;border-radius:5px;background-color:#aaf0f0;overflow:hidden;font-weight:700;border: 5px solid #21a3a3;"> 🚲 Patterns Recognition - Cyclic behaviour</div>

# # 🚲 8. Patterns Recognition - Cyclic behaviour
# If the trend shows a cyclic behavior, not necessarily maintaining a calender.
# 
# 
# ![](https://robjhyndman.com/hyndsight/2011-12-14-cyclicts_files/figure-html/unnamed-chunk-1-1.png)
# 
# - source: robjhyndman

# <div style="padding:20px;color:black;margin:0;font-size:250%;text-align:center;display:fill;border-radius:5px;background-color:#aaf0f0;overflow:hidden;font-weight:700;border: 5px solid #21a3a3;"> 🗓 Additive and Multiplicative Time Series </div>

# # 🗓 9. Additive and Multiplicative Time Series 
# 
# We may have different combinations of trends and seasonality. Depending on the nature of the trends and seasonality, a time series can be modeled as an additive or multiplicative time series. Each observation in the series can be expressed as either a sum or a product of the components.
# 
# ## Additive time series:
# **Value = Base Level + Trend + Seasonality + Error**
# 
# ## Multiplicative Time Series:
# **Value = Base Level x Trend x Seasonality x Error**

# <div style="padding:20px;color:black;margin:0;font-size:250%;text-align:center;display:fill;border-radius:5px;background-color:#aaf0f0;overflow:hidden;font-weight:700;border: 5px solid #21a3a3;"> 📟  Stationaity</div>

# # 📟 10. Stationaity
# A stationary time series has **statistical properties or moments** (e.g., mean and variance) that do not vary in time. Stationarity, then, is the status of a stationary time series. Conversely, nonstationarity is the status of a time series whose statistical properties are changing through time.
# 
# Data points are often non-stationary or have means, variances, and covariances that change over time. **Non-stationary behaviors can be trends, cycles, random walks, or combinations of the three**. Non-stationary data, as a rule, are unpredictable and cannot be modeled or forecasted.

# ![](https://www.investopedia.com/thmb/HtFNw1xKlp-MdCrmblCoER9zV3w=/660x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/AnIntroductiontoStationaryandNon-StationaryProcesses1_4-babcf20c2229411f8da42e24e7aaa18f.png)
# 
# ![](https://www.investopedia.com/thmb/o8j3j__1rfqHhd2PniUgECjsNz8=/660x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/AnIntroductiontoStationaryandNon-StationaryProcesses2_4-f26acdca6bb14ed185223475e779508f.png)

# ## How to make a time series stationary? 
# 
# We can apply some sort of transformation to make the time-series stationary. These transformation may include:
# - Differencing the Series (once or more)
# - Take the log of the series
# - Take the nth root of the series
# - Combination of the above

# ## Why make a non-stationary series stationary before forecasting?
# 
# Forecasting a stationary series is relatively easy and the forecasts are more reliable.
# An important reason is, autoregressive forecasting models are essentially linear regression models that utilize the lag(s) of the series itself as predictors.
# We know that linear regression works best if the predictors (X variables) are not correlated against each other. So, stationarizing the series solves this problem since it removes any persistent autocorrelation, thereby making the predictors(lags of the series) in the forecasting models nearly independent.

# ##  How to test for stationarity?
# The stationarity of a series can be established by looking at the plot of the series like we did earlier.
# 
# Another method is to split the series into 2 or more contiguous parts and computing the summary statistics like the mean, variance and the autocorrelation. If the stats are quite different, then the series is not likely to be stationary.
# 
# Nevertheless, you need a method to quantitatively determine if a given series is stationary or not. This can be done using statistical tests called ‘Unit Root Tests’. There are multiple variations of this, where the tests check if a time series is non-stationary and possess a unit root.
# 
# There are multiple implementations of Unit Root tests like:
# - Augmented Dickey Fuller test (ADH Test)
# - Kwiatkowski-Phillips-Schmidt-Shin – KPSS test (trend stationary)
# - Philips Perron test (PP Test)
# 
# The most commonly used is the ADF test, where the null hypothesis is the time series possesses a unit root and is non-stationary. So, id the P-Value in ADH test is less than the significance level (0.05), you reject the null hypothesis.
# 
# The KPSS test, on the other hand, is used to test for trend stationarity. The null hypothesis and the P-Value interpretation is just the opposite of ADH test. The below code implements these two tests using statsmodels package in python.

# In[8]:


df = train[(train['country']=='Belgium')&(train['product']=='Kaggle Advanced Techniques')&(train['store']=='KaggleMart')]


# In[9]:


sns.set(rc={'figure.figsize':(24,8)})
ax=sns.lineplot(data=df,x='date',y='num_sold')
ax.axes.set_title("\nSales of Kaggle Advanced Techniques on KaggleMart in Belgium\n",fontsize=20);


# ### Augmented Dickey Fuller test (ADH Test)

# In[10]:


from statsmodels.tsa.stattools import adfuller
result = adfuller(df.num_sold.values, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')


# ## Kwiatkowski-Phillips-Schmidt-Shin – KPSS test (trend stationary)

# In[11]:


from statsmodels.tsa.stattools import kpss
result = kpss(df.num_sold.values, regression='c')
print('\nKPSS Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[3].items():
    print('Critial Values:')
    print(f'   {key}, {value}');


# ## What is the difference between white noise and a stationary series?
# 
# Like a stationary series, the white noise is also not a function of time, that is its mean and variance does not change over time. But the difference is, the white noise is completely random with a mean of 0.
# In white noise there is no pattern whatsoever. If you consider the sound signals in an FM radio as a time series, the blank sound you hear between the channels is white noise.
# Mathematically, a sequence of completely random numbers with mean zero is a white noise.

# <div style="padding:20px;color:black;margin:0;font-size:250%;text-align:center;display:fill;border-radius:5px;background-color:#aaf0f0;overflow:hidden;font-weight:700;border: 5px solid #21a3a3;"> ✂️  Treating missing values</div>

# # ✂️ 11. How to treat missing values in a time series?
# 
# Some effective alternatives to imputation are:
# 
# - Backward Fill
# - Linear Interpolation
# - Quadratic interpolation
# - Mean of nearest neighbors
# - Mean of seasonal couterparts

# <div style="padding:20px;color:black;margin:0;font-size:250%;text-align:center;display:fill;border-radius:5px;background-color:#aaf0f0;overflow:hidden;font-weight:700;border: 5px solid #21a3a3;"> 📑 Autocorrelation and Partial Autocorrelation</div>

# # 📑 12. Autocorrelation and Partial Autocorrelation

# The term autocorrelation refers to the **degree of similarity between A) a given time series, and B) a lagged version of itself, over C) successive time intervals**. In other words, autocorrelation is intended to measure the relationship between a variable's present value and any past values that you may have access to.
# 
# When regression is performed on time series data, the errors may not be independent. Often errors are autocorrelated; that is, each error is correlated with the error immediately before it. Autocorrelation is also a symptom of systematic lack of fit. The DW option provides the Durbin-Watson d statistic to test that the autocorrelation is zero:
# d = \frac{ \sum_{i=2}^n (e_i - e_{i-1})^2}{\sum_{i=1}^n e_i^2} 
# The value of d is close to 2 if the errors are uncorrelated. The distribution of d is reported by Durbin and Watson (1951). Tables of the distribution are found in most econometrics textbooks, such as Johnston (1972) and Pindyck and Rubinfeld (1981).
# 
# The sample autocorrelation estimate is displayed after the Durbin-Watson statistic. The sample is computed as
# 
# r = \frac{\sum_{i=2}^n e_i e_{i-1}}{\sum_{i=1}^n e_i^2} 
# This autocorrelation of the residuals may not be a very good estimate of the autocorrelation of the true errors, especially if there are few observations and the independent variables have certain patterns. If there are missing observations in the regression, these measures are computed as though the missing observations did not exist.
# 
# Positive autocorrelation of the errors generally tends to make the estimate of the error variance too small, so confidence intervals are too narrow and true null hypotheses are rejected with a higher probability than the stated significance level. Negative autocorrelation of the errors generally tends to make the estimate of the error variance too large, so confidence intervals are too wide and the power of significance tests is reduced. With either positive or negative autocorrelation, least-squares parameter estimates are usually not as efficient as generalized least-squares parameter estimates.

# Autocorrelation is the correlation between two observations at different points in a time series. For example, values that are separated by an interval might have a strong positive or negative correlation. When these correlations are present, they indicate that past values influence the current value. Analysts use the autocorrelation and partial autocorrelation functions to understand the properties of time series data, fit the appropriate models, and make forecasts.
# 
# In this post, I cover both the autocorrelation function and partial autocorrelation function. You’ll learn about the differences between these functions and what they can tell you about your data. In later posts, I’ll show you how to incorporate this information in regression models of time series data and other time-series analyses.
# 
# #### Autocorrelation and Partial Autocorrelation Basics
# Autocorrelation is the correlation between two values in a time series. In other words, the time series data correlate with themselves—hence, the name. We talk about these correlations using the term “lags.” Analysts record time-series data by measuring a characteristic at evenly spaced intervals—such as daily, monthly, or yearly. The number of intervals between the two observations is the lag. For example, the lag between the current and previous observation is one. If you go back one more interval, the lag is two, and so on.
# 
# In mathematical terms, the observations at yt and yt–k are separated by k time units. K is the lag. This lag can be days, quarters, or years depending on the nature of the data. When k=1, you’re assessing adjacent observations. For each lag, there is a correlation.
# 
# #### Autocorrelation Function (ACF)
# Use the autocorrelation function (ACF) to identify which lags have significant correlations, understand the patterns and properties of the time series, and then use that information to model the time series data. From the ACF, you can assess the randomness and stationarity of a time series. You can also determine whether trends and seasonal patterns are present.
# 
# In an ACF plot, each bar represents the size and direction of the correlation. Bars that extend across the red line are statistically significant.
# 
# #### Partial Autocorrelation Function (PACF)
# The partial autocorrelation function is similar to the ACF except that it displays only the correlation between two observations that the shorter lags between those observations do not explain. For example, the partial autocorrelation for lag 3 is only the correlation that lags 1 and 2 do not explain. In other words, the partial correlation for each lag is the unique correlation between those two observations after partialling out the intervening correlations.
# 
# As you saw, the autocorrelation function helps assess the properties of a time series. In contrast, the partial autocorrelation function (PACF) is more useful during the specification process for an autoregressive model. Analysts use partial autocorrelation plots to specify regression models with time series data and Auto Regressive Integrated Moving Average (ARIMA) models. I’ll focus on that aspect in posts about those methods.
# 
# *https://statisticsbyjim.com/time-series/autocorrelation-partial-autocorrelation/*

# In[12]:


from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = train[(train['country']=='Belgium')&(train['product']=='Kaggle Advanced Techniques')&(train['store']=='KaggleMart')]
df['value']=df['num_sold']

acf_50 = acf(df.value, nlags=50)
pacf_50 = pacf(df.value, nlags=50)

# Draw Plot
fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
plot_acf(df.value.tolist(), lags=50, ax=axes[0])
plot_pacf(df.value.tolist(), lags=50, ax=axes[1])


# <div style="padding:20px;color:black;margin:0;font-size:250%;text-align:center;display:fill;border-radius:5px;background-color:#aaf0f0;overflow:hidden;font-weight:700;border: 5px solid #21a3a3;"> 📖 Lag plots</div>

# # 📖 13. Lag Plots
# 
# A Lag plot is a scatter plot of a time series against a lag of itself. It is normally used to check for autocorrelation. 

# In[13]:


from pandas.plotting import lag_plot
plt.rcParams.update({'ytick.left' : False, 'axes.titlepad':10})


ss = train[(train['country']=='Belgium')&(train['product']=='Kaggle Advanced Techniques')&(train['store']=='KaggleMart')]
ss['value']=ss['num_sold']
a10 = train[(train['country']=='France')&(train['product']=='Kaggle Advanced Techniques')&(train['store']=='KaggleMart')]
a10['value']=a10['num_sold']


fig, axes = plt.subplots(1, 4, figsize=(10,3), sharex=True, sharey=True, dpi=100)
for i, ax in enumerate(axes.flatten()[:4]):
    lag_plot(ss.value, lag=i+1, ax=ax, c='firebrick')
    ax.set_title('Lag ' + str(i+1))

fig.suptitle('Lag Plots of Sun Spots Area \n(Points get wide and scattered with increasing lag -> lesser correlation)\n', y=1.15)    

fig, axes = plt.subplots(1, 4, figsize=(10,3), sharex=True, sharey=True, dpi=100)
for i, ax in enumerate(axes.flatten()[:4]):
    lag_plot(a10.value, lag=i+1, ax=ax, c='firebrick')
    ax.set_title('Lag ' + str(i+1))

fig.suptitle('Lag Plots of Drug Sales', y=1.05)    
plt.show()


# <div style="padding:20px;color:black;margin:0;font-size:250%;text-align:center;display:fill;border-radius:5px;background-color:#aaf0f0;overflow:hidden;font-weight:700;border: 5px solid #21a3a3;"> ⏳ Forecastability</div>

# # ⏳ 14. Forecastability
# 
# **The more regular and repeatable patterns a time series has, the easier it is to forecast.** The Approximate Entropy can be used to quantify the regularity and unpredictability of fluctuations in a time series. The higher the approximate entropy, the more difficult it is to forecast it.
# 
# If you measure the CV of any demand pattern, be it actual sales, forecasts or anything like it, you will get a number that represents both forecastable signal and unforecastable noise. It doesn't tell you which part of that is forecastable.

# The more regular and repeatable patterns a time series has, the easier it is to forecast. The ‘Approximate Entropy’ can be used to quantify the regularity and unpredictability of fluctuations in a time series.
# 
# The higher the approximate entropy, the more difficult it is to forecast it.
# 
# Another better alternate is the ‘Sample Entropy’.
# 
# Sample Entropy is similar to approximate entropy but is more consistent in estimating the complexity even for smaller time series. For example, a random time series with fewer data points can have a lower ‘approximate entropy’ than a more ‘regular’ time series, whereas, a longer random time series will have a higher ‘approximate entropy’.

# <div style="padding:20px;color:black;margin:0;font-size:250%;text-align:center;display:fill;border-radius:5px;background-color:#aaf0f0;overflow:hidden;font-weight:700;border: 5px solid #21a3a3;"> 🔮 Smoothing</div>

# # 🔮 15. Smoothing

# ### Smoothening of a time series may be useful in:
# 
# - Reducing the effect of noise in a signal get a fair approximation of the noise-filtered series.
# - The smoothed version of series can be used as a feature to explain the original series itself.
# - Visualize the underlying trend better
# 
# ### So how to smoothen a series? Let’s discuss the following methods:
# 
# - Take a moving average
# - Do a LOESS smoothing (Localized Regression)
# - Do a LOWESS smoothing (Locally Weighted Regression)
# - Moving average is nothing but the average of a rolling window of defined width. But you must choose the window-width wisely, because, large window-size will over-smooth the series. For example, a window-size equal to the seasonal duration (ex: 12 for a month-wise series), will effectively nullify the seasonal effect.

# In[14]:


from statsmodels.nonparametric.smoothers_lowess import lowess
plt.rcParams.update({'xtick.bottom' : False, 'axes.titlepad':5})

df = train[(train['country']=='Belgium')&(train['product']=='Kaggle Advanced Techniques')&(train['store']=='KaggleMart')]
df['value']=df['num_sold']
df=df.set_index('date')
df.index = pd.to_datetime(df.index)

df_orig=df.copy()

df_ma = df_orig.value.rolling(3, center=True, closed='both').mean()


df_loess_5 = pd.DataFrame(lowess(df_orig.value, np.arange(len(df_orig.value)), frac=0.05)[:, 1], index=df_orig.index, columns=['value'])
df_loess_15 = pd.DataFrame(lowess(df_orig.value, np.arange(len(df_orig.value)), frac=0.15)[:, 1], index=df_orig.index, columns=['value'])

# Plot
fig, axes = plt.subplots(4,1, figsize=(7, 7), sharex=True, dpi=120)
df_orig['value'].plot(ax=axes[0], color='k', title='Original Series')
df_loess_5['value'].plot(ax=axes[1], title='Loess Smoothed 5%')
df_loess_15['value'].plot(ax=axes[2], title='Loess Smoothed 15%')
df_ma.plot(ax=axes[3], title='Moving Average (3)')
fig.suptitle('How to Smoothen a Time Series', y=0.95, fontsize=14)
plt.show()


# <div style="padding:20px;color:black;margin:0;font-size:250%;text-align:center;display:fill;border-radius:5px;background-color:#aaf0f0;overflow:hidden;font-weight:700;border: 5px solid #21a3a3;"> 📄 ARIMA Model</div>

# # 📄 16. ARIMA Model

# ARIMA Model stands for Auto-Regressive Integrated Moving Average. It is used to predict the future values of a time series using its past values and forecast errors. The below diagram shows the components of an ARIMA model: 
# 
# ![](https://www.simplilearn.com/ice9/free_resources_article_thumb/Time_Series_Analysis_In_Python_4.png)
# 
# ### Auto Regressive Model
# Auto-Regressive models predict future behavior using past behavior where there is some correlation between past and future data. The formula below represents the autoregressive model. It is a modified version of the slope formula with the target value being expressed as the sum of the intercept, the product of a coefficient and the previous output, and an error correction term.
# 
# ![](https://www.simplilearn.com/ice9/free_resources_article_thumb/Time_Series_Analysis_In_Python_5.png)
# 
# ### Moving Average
# Moving Average is a statistical method that takes the updated average of values to help cut down on noise. It takes the average over a specific interval of time. You can get it by taking different subsets of your data and finding their respective averages.
# 
# You first consider a bunch of data points and take their average. You then find the next average by removing the first value of the data and including the next value of the series.
# 
# ![](https://www.simplilearn.com/ice9/free_resources_article_thumb/Time_Series_Analysis_In_Python_6.png)
# 
# ### Integration
# Integration is the difference between present and previous observations. It is used to make the time series stationary. 
# 
# Each of these values acts as a parameter for our ARIMA model. Instead of representing the ARIMA model by these various operators and models, you use parameters to represent them. These parameters are: 
# 
# - p: Previous lagged values for each time point. Derived from the Auto-Regressive Model.
# - q: Previous lagged values for the error term. Derived from the Moving Average.
# - d: Number of times data is differenced to make it stationary. It is the number of times it performs integration.

# In[15]:


df = train[(train['country']=='Belgium')&(train['product']=='Kaggle Advanced Techniques')&(train['store']=='KaggleMart')]
series=pd.DataFrame()
series['value']=df['num_sold']
series=series.set_index(df['date'])
series.index = pd.to_datetime(series.index)

series


# In[16]:


from statsmodels.tsa.arima.model import ARIMA

model = ARIMA (series, order=(5,1,0))
model_fit = model.fit()
print(model_fit.summary())

# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print (residuals.describe())


# <div style="padding:20px;color:black;margin:0;font-size:250%;text-align:center;display:fill;border-radius:5px;background-color:#aaf0f0;overflow:hidden;font-weight:700;border: 5px solid #21a3a3;"> 📄 SARIMA Model</div>

# # 📄 17. SARIMA Model

# The ARIMA model is great, but to include seasonality and exogenous variables in the model can be extremely powerful. Since the ARIMA model assumes that the time series is stationary, we need to use a different model.
# 
# SARIMA stands for Seasonal-ARIMA and it includes seasonality contribution to the forecast. The importance of seasonality is quite evident and ARIMA fails to encapsulate that information implicitly.
# 
# The Autoregressive (AR), Integrated (I), and Moving Average (MA) parts of the model remain as that of ARIMA. The addition of Seasonality adds robustness to the SARIMA model. It’s represented as:
# ![](https://i0.wp.com/neptune.ai/wp-content/uploads/Sarima-seasonality.png?resize=425%2C145&ssl=1)
# 
# Enter SARIMA (Seasonal ARIMA). This model is very similar to the ARIMA model, except that there is an additional set of autoregressive and moving average components.The additional lags are offset by the frequency of seasonality (ex. 12 — monthly, 24 — hourly).
# 
# SARIMA models allow for differencing data by seasonal frequency, yet also by non-seasonal differencing. Knowing which parameters are best can be made easier through automatic parameter search frameworks such as pmdarina.

# A seasonal ARIMA model uses differencing at a lag equal to the number of seasons (s) to remove additive seasonal effects. As with lag 1 differencing to remove a trend, the lag s differencing introduces a moving average term. The seasonal ARIMA model includes autoregressive and moving average terms at lag s.
# 
# — Page 142, Introductory Time Series with R, 2009.

# <div style="padding:20px;color:black;margin:0;font-size:250%;text-align:center;display:fill;border-radius:5px;background-color:#aaf0f0;overflow:hidden;font-weight:700;border: 5px solid #21a3a3;">  📒 SARIMAX Model</div>

# SARIMAX(Seasonal Auto-Regressive Integrated Moving Average with eXogenous factors) is an updated version of the ARIMA model. we can say SARIMAX is a seasonal equivalent model like SARIMA and Auto ARIMA. it can also deal with external effects. This feature of the model differs from other models. - Yugesh Verma.

# ![](https://365datascience.com/resources/blog/2020-07-sarimax-model-equation-explained-1024x404.png)
# 
# [*source*](https://365datascience.com/tutorials/python-tutorials/sarimax/)

# ## What is Sarimax used for?
# 
# Naproxen Sodium is used in musculoskeletal and joint disorders such as ankylosing spondylitis, osteoarthritis, and rheumatoid arthritis including juvenile idiopathic arthritis. It is also used in dysmenorrhea, headache including migraine, postoperative pain, soft tissue disorders, acute gout, and to reduce fever.
# 
# ## What is difference between sarima and Sarimax?
# 
# The implementation is called SARIMAX instead of SARIMA because the “X” addition to the method name means that the implementation also supports exogenous variables. These are parallel time series variates that are not modeled directly via AR, I, or MA processes, but are made available as a weighted input to the model.
# 
# ## What is seasonal order in Sarimax?
# A SARIMA model can be tuned with two kinds of orders: (p,d,q) order, which refers to the order of the time series. This order is also used in the ARIMA model (which does not consider seasonality); (P,D,Q,M) seasonal order, which refers to the order of the seasonal component of the time series.

# <div style="padding:20px;color:black;margin:0;font-size:250%;text-align:center;display:fill;border-radius:5px;background-color:#aaf0f0;overflow:hidden;font-weight:700;border: 5px solid #21a3a3;"> 📇 Rolling Statistics</div>

# #  📇 19. Rolling Statistics
# 
# A rolling analysis of a time series model is often used to assess the model's stability over time. When analyzing financial time series data using a statistical model, a key assumption is that the parameters of the model are constant over time.
# 
# A rolling average is a great way to visualize how the dataset is trending. As the dataset provides counts by month, a window size of 12 will give us the annual rolling average.
# 
# We will also include the rolling standard deviation to see how much the data varies from the rolling average.

# <div style="padding:20px;color:black;margin:0;font-size:250%;text-align:center;display:fill;border-radius:5px;background-color:#aaf0f0;overflow:hidden;font-weight:700;border: 5px solid #21a3a3;">  📒 Pros and Cons</div>

# #  📒 20. Pros and Cons
# 
# ## Pros of ARIMA & SARIMA
# - **Easy to understand and interpret:** The one thing that your fellow teammates and colleagues would appreciate is the simplicity and interpretability of the models. Focusing on both of these things while also maintaining the quality of the results will help with presentations with the stakeholders.
# - **Limited variables:** There are fewer hyperparameters so the config file will be easily maintainable if the model goes into production.
# 
# ## Cons of ARIMA & SARIMA
# - **Exponential time complexity:** When the value of p and q increases there are equally more coefficients to fit hence increasing the time complexity manifold if p and q are high. This makes both of these algorithms hard to put into production and makes Data Scientists look into Prophet and other algorithms. Then again, it depends on the complexity of the dataset too.
# - **Complex data:** There can be a possibility where your data is too complex and there is no optimal solution for p and q. Although highly unlikely that ARIMA and SARIMA would fail but if this occurs then unfortunately you may have to look elsewhere.
# - **Amount of data needed:** Both the algorithms require considerable data to work on, especially if the data is seasonal. For example, using three years of historical demand is likely not to be enough (Short Life-Cycle Products) for a good forecast.

# <div style="padding:20px;color:black;margin:0;font-size:250%;text-align:center;display:fill;border-radius:5px;background-color:#aaf0f0;overflow:hidden;font-weight:700;border: 5px solid #21a3a3;"> 📓 References</div>

# # 📓 21. References
# 
# - [Time Series Analysis in Python – A Comprehensive Guide with Examples](https://www.machinelearningplus.com/time-series/time-series-analysis-python/)
# - [Understanding Time Series Analysis in Python](https://www.simplilearn.com/tutorials/python-tutorial/time-series-analysis-in-python)
# - [A Guide to Time Series Analysis in Python](https://builtin.com/data-science/time-series-python)
# - [Complete Guide on Time Series Analysis in Python by @prashant111 ](https://www.kaggle.com/code/prashant111/complete-guide-on-time-series-analysis-in-python)
# - [A Guide to Obtaining Time Series Datasets in Python](https://machinelearningmastery.com/a-guide-to-obtaining-time-series-datasets-in-python/)
# - [Time Series Forecasting with ARIMA , SARIMA and SARIMAX](https://towardsdatascience.com/time-series-forecasting-with-arima-sarima-and-sarimax-ee61099e78f6)
# - [ARIMA & SARIMA: Real-World Time Series Forecasting](https://neptune.ai/blog/arima-sarima-real-world-time-series-forecasting-guide)

# <div style="padding:20px;color:black;margin:20;font-size:350%;text-align:center;display:fill;border-radius:5px;background-color:
# #f0c999;overflow:hidden;font-weight:800;border: 10px solid #eb8b15;"> Upvote, Share, Support</div>
