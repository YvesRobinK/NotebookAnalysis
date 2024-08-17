#!/usr/bin/env python
# coding: utf-8

# # How to Detect Random Walk and White Noise in Time Series Forecasting
# ## Find out if the target is worth predicting!
# ![](https://miro.medium.com/max/2000/1*8WHjedcCTZtWsKZTk8aSnw.jpeg)
# <figcaption style="text-align: center;">
#     <strong>
#         Photo by 
#         <a href='https://www.pexels.com/@pripicart?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels'>Tobi</a>
#         on 
#         <a href='https://www.pexels.com/photo/person-stands-on-brown-pathway-631986/?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels'>Pexels</a>
#     </strong>
# </figcaption>

# ## Setup

# In[1]:


import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("ggplot")

warnings.filterwarnings("ignore")


# In[2]:


tps_july = pd.read_csv(
    "../input/tabular-playground-series-jul-2021/train.csv",
    parse_dates=["date_time"],
    index_col="date_time",
)


# ## Introduction <small id='intro'></small>

# No matter how powerful, machine learning cannot predict everything. A well-known area where it can become pretty helpless is related to time series forecasting.
# 
# Despite the availability of a large suite of autoregressive models and many other algorithms for time series, you cannot predict the target distribution if it is **white noise** or follows a **random walk**.
# 
# So, you must detect such distributions before you make further efforts.
# 
# In this article, you will learn what white noise and random walk are and explore proven statistical techniques to detect them.

# ## Table of Contents <small id='toc'></small>

# #### [1. Brief notes on Autocorrelation](#1)
# #### [2. What is white noise?](#2)
# #### [3. Importance of White Noise in Forecasting](#3)
# #### [4. Random Walks](#4)
# #### [5. Random Walks with drift](#5)
# #### [6. Detecting random walks statistically](#6)
# #### [7. Summary](#7)

# > You can read the Medium article of this notebook [here](https://towardsdatascience.com/how-to-detect-random-walk-and-white-noise-in-time-series-forecasting-bdb5bbd4ef81).

# ## Before we start...

# This is my third article on the time series forecasting series (you can check out the whole series from this [list](https://ibexorigin.medium.com/list/time-series-forecast-from-scratch-c72ccf7a9229), a new Medium feature).
# 
# While the first one was about [every single Pandas function to manipulate TS data](https://medium.com/r/?url=https%3A%2F%2Ftowardsdatascience.com%2Fevery-pandas-function-you-can-should-use-to-manipulate-time-series-711cb0c5c749%3Fsource%3Dyour_stories_page-------------------------------------), the second was about [time series decomposition and autocorrelation](https://www.kaggle.com/bextuychiev/advanced-time-series-analysis-decomposition/comments).
# 
# To get the most out of this post, you need to understand at least what autocorrelation is. Here, I will give a brief explanation, but check out my last article if you want to go deeper.

# ## 1. Brief notes on Autocorrelation <small id='1'></small>

# <a href='#toc'>Back to top🔝</a>
# 
# Autocorrelation involves finding the correlation between a time series and a lagged version of itself. Consider this distribution:

# In[3]:


deg_C = tps_july["deg_C"].to_frame("temperature")

deg_C.head()


# Lagging a time series means shifting it 1 or more periods backward:

# In[4]:


deg_C["lag_1"] = deg_C["temperature"].shift(periods=1)
deg_C["lag_2"] = deg_C["temperature"].shift(periods=2)
deg_C["lag_3"] = deg_C["temperature"].shift(periods=3)

deg_C.head(6)


# The Autocorrelation Function (ACF) finds the correlation coefficient between a time series and its lagged version at each lag *k*. You can plot it using the `plot_acf` function from `statsmodels`. Here is what it looks like:

# In[5]:


from matplotlib import rcParams
from statsmodels.graphics.tsaplots import plot_acf

rcParams["figure.figsize"] = 9, 4
# ACF function up to 50 lags
fig = plot_acf(deg_C["temperature"], lags=50)

plt.show();


# The XAxis is the lag *k*, and the YAxis is the Pearson's correlation coefficient at each lag. The red shaded region is a confidence interval. If the height of the bars is outside this region, it means the correlation is statistically significant.

# ## 2. What is white noise? <small id='2'></small>

# <a href='#toc'>Back to top🔝</a>
# 
# In short, white noise distribution is any distribution that has:
# 
# - Zero mean
# - A constant variance/standard deviation (does not change over time)
# - Zero autocorrelation at all lags
# 
# Essentially, it is a series of random numbers, and by definition, no algorithm can reasonably model its behavior.
# 
# There are special types of white noise. If the noise is normal (follows a [normal distribution](https://medium.com/r/?url=https%3A%2F%2Ftowardsdatascience.com%2Fhow-to-use-normal-distribution-like-you-know-what-you-are-doing-1cf4c55241e3%3Fsource%3Dyour_stories_page-------------------------------------)), it is called Gaussian white noise. Let's see an example of this visually:

# In[6]:


# Generate Gaussian white noise dist with mean 0 and 0.5 std
noise = np.random.normal(loc=0, scale=0.5, size=1000)

plt.figure(figsize=(12, 4))
plt.plot(noise);


# > Gaussian white noise distribution with a standard deviation of 0.5
# 
# Even though there are occasional spikes, there are no discernible patterns visible, i.e., the distribution is completely random.
# 
# The best way you can validate this is to create the ACF plot:

# In[7]:


fig = plot_acf(noise, lags=40)

plt.title("Autocorrelation of a White Noise Series")
plt.show()


# > White noise distributions have approximately 0 autocorrelation at all lags.
# 
# There are also "strict" white noise distributions - these have strictly 0 serial correlation. This is different from [brown/pink](https://medium.com/r/?url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FBrownian_noise) noise or other natural random phenomena where there is a weak serial correlation but still remain memory-free.

# ## 3. Importance of White Noise in Forecasting <small id='3'></small>

# <a href='#toc'>Back to top🔝</a>
# 
# Even though white noise distributions are considered dead ends, they can be quite useful in other contexts.
# 
# For example, in time series forecasting, if the differences between predictions and actual values represent a white noise distribution, you can pat yourself on the back for a job well done.
# 
# When the residual errors show any pattern, whether seasonal or trending or have a non-zero mean, this suggests there is still room for improvement. In contrast, if the residuals are purely white noise, you maxed out the abilities of the chosen model.
# 
# In other words, the algorithm managed to capture all the important signals and properties of the target. What's left are the random fluctuations and inconsistent data points that could not be attributed to anything.
# 
# For example, we will predict the amount of carbon monoxide in the air using the July Kaggle playground competition. We will leave the inputs "as-is" - we won't perform any feature engineering, and we will choose a baseline model with default parameters:

# In[8]:


X = tps_july.drop(
    ["target_carbon_monoxide", "target_benzene", "target_nitrogen_oxides"], axis=1
)
y = tps_july["target_carbon_monoxide"].values.reshape(-1, 1)


# In[9]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=1121218, test_size=0.1
)

forest = RandomForestRegressor()
forest.fit(X_train, y_train)

preds = forest.predict(X_test)
residuals = y_test.flatten() - preds

plt.plot(residuals)
plt.title("Plot of the Error residuals");


# In[10]:


fig = plot_acf(residuals, lags=50)

plt.title("Autocorrelation Plot of the Error residuals")
plt.show();


# There is some pattern in the ACF plot, but they are within the confidence interval. These two plots suggest that Random Forests could capture almost all the important signals from the training data even with default parameters.

# ## 4. Random Walks <small id='4'></small>

# <a href='#toc'>Back to top🔝</a>
# 
# A more challenging but equally unpredictable distribution in time series forecasting is a random walk. Unlike white noise, it has non-zero mean, non-constant std/variance, and when plotted, looks a lot like a regular distribution:
# 
# ![](https://miro.medium.com/max/1400/1*Ypc9yU6wX8yg6-gCKhFf4w.png)

# > Doesn't it resemble a plot of stocks from Yahoo Finance?

# Random walk series are always cleverly disguised in this manner, but still, they are unpredictable as ever. The best guess for today's value is yesterday's.
# 
# A common confusion among beginners is thinking of a random walk as a simple sequence of random numbers. This is not the case because, in a random walk, each step is dependent on the previous step. 
# 
# For this reason, the Autocorrelation function of random walks does return non-zero correlations.
# The formula of a random walk is simple:
# 
# ![](https://miro.medium.com/proxy/1*5m_X6KTNGar2K8oLd92XMg.png)

# Whatever the previous data point is, add some random value to it and continue for as long as you like. Let's generate this in Python with a starting value of, let's say, 99:

# In[11]:


walk = [99]

for i in range(1000):
    # Create random noise
    noise = -1 if np.random.random() < 0.5 else 1
    walk.append(walk[-1] + noise)

rcParams["figure.figsize"] = 14, 4
plt.plot(walk);


# Let's also plot the ACF:

# In[12]:


fig = plot_acf(walk, lags=50)
plt.show();


# As you can see, the first ~40 lags yield statistically significant correlations.
# 
# So, how do we detect a random walk when a visualization is not an option?
# 
# Because of how they are created, differencing the time series should isolate the random addition of each step. Taking the first-order difference is done by lagging the series by 1 and subtracting it from the original. Pandas has a convenient `diff` function to do this:

# In[13]:


walk_diff = pd.Series(walk).diff()

plt.plot(walk_diff);


# If you plot the first-order difference of a time series and the result is white noise, then it is a random walk.

# ## 5. Random Walks with drift <small id='5'></small>

# <a href='#toc'>Back to top🔝</a>
# 
# A slight modification to regular random walks is adding a constant value called a drift at random step:
# 
# ![](https://miro.medium.com/max/696/1*keplCGTu2KDh9ksHFPtDWA.png)

# Drift is often denoted with μ, and in terms of values changing over time, drift means gradually changing into something.
# 
# For example, even though stocks fluctuate constantly, they might have a positive drift, i.e., gain an overall gradual increase over time.
# 
# Now, let's see how to simulate this in Python. We will first create the regular random walk with a start value of 25:

# In[14]:


walk = [25]

for i in range(1000):
    # Create random noise
    noise = -1 if np.random.random() < 0.5 else 1
    # Add the drift too
    walk.append(walk[-1] + noise)


# From the above formula, we see that we need to add the desired drift at each step. Let's add a drift of 5 and look at the plot:

# In[15]:


drift = 5
drifty_walk = pd.Series(walk) + 5

drifty_walk.plot(title="A Random Walk with Drift");


# Despite the wild fluctuations, the series has a discernible upward drift. If we perform differencing, we will see that the series is still a random walk:

# In[16]:


drifty_walk.diff().plot();


# ## 6. Detecting random walks statistically <small id='6'></small>

# <a href='#toc'>Back to top🔝</a>
# 
# You might ask if there are better methods of identifying random walks than just "eyeballing" them from plots.
# 
# As an answer, there is a hypothesis test outlined in 1979 by Dicker D. A. and Fuller W. A., and it is called the augmented Dickey-Fuller test.
# 
# Essentially, it tries to test the null hypothesis that a series follows a random walk. Under the hood, it regresses the difference in prices on the lagged price.
# 
# ![](https://miro.medium.com/max/992/1*Ver7NOe_-5lA1bJBWxhFdA.png)
# 
# If the found slope (β) is equal to 0, the series is a random walk. If the slope is significantly different from 0, we reject the null hypothesis that the series follows a random walk.
# 
# Fortunately, you don't have to worry about the math because the test is already implemented in Python. 
# 
# We import the `adfuller` function from `statsmodels` and use it on the drifty random walk created in the last section:

# In[17]:


from statsmodels.tsa.stattools import adfuller

results = adfuller(drifty_walk)

print(f"ADF Statistic: {results[0]}")
print(f"p-value: {results[1]}")
print("Critical Values:")
for key, value in results[4].items():
    print("\t%s: %.3f" % (key, value))


# We look at the p-value, which is ~0.26. Since 0.05 is the significance threshold, we fail to reject the null hypothesis that `drifty_walk` is a random walk, i.e., it is a random walk.
# 
# Let's perform another test on a distribution we know isn't a random walk. We will use the carbon monoxide target from the TPS July Kaggle playground competition:

# In[18]:


results = adfuller(tps_july["target_carbon_monoxide"])

print(f"ADF Statistic: {results[0]}")
print(f"p-value: {results[1]}")
print("Critical Values:")
for key, value in results[4].items():
    print("\t%s: %.3f" % (key, value))


# The p-value is extremely small, suggesting we can easily reject the null hypothesis that `target_carbon_monoxide` follows a random walk.

# ## Summary <small id='7'></small>

# <a href='#toc'>Back to top🔝</a>
# 
# We only finished the third part of this Time Series "series," and you already know a ton.
# 
# From here on, things are only going to get more and more interesting as we draw closer to the actual "forecasting" part in the series. There are some interesting articles planned on key time series topics such as stationarity and time-series cross-validation.
# 
# Besides, I will dedicate a post solely on feature engineer specific to time series - this is something to be excited about! [Stay tuned](https://ibexorigin.medium.com/)!

# ## My other kernels you might be interested...
# - [Comprehensive Guide to Multiclass Classification](https://www.kaggle.com/bextuychiev/comprehensive-guide-to-mutliclass-classification)
# - [Advanced Time Series Analysis in Python: Decomposition, Autocorrelation](https://www.kaggle.com/bextuychiev/advanced-time-series-analysis-decomposition)
# - [Master the Subtle Art of Train/Test Generation](https://www.kaggle.com/bextuychiev/master-the-subtle-art-of-train-test-set-generation)
