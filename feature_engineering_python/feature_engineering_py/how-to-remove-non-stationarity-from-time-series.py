#!/usr/bin/env python
# coding: utf-8

# # How to Remove Non-Stationarity in Time Series Forecasting
# ## Algorithms can't handle non-stationary. They need static relationships.
# ![](https://cdn-images-1.medium.com/max/1200/1*qhSTGmSC69HxBM8V3Jh7bA.jpeg)
# <figcaption style="text-align: center;">
#     <strong>
#         Photo by 
#         <a href='https://unsplash.com/@jonathanpielmayer?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText'>Jonathan Pielmayer</a>
#         on 
#         <a href='https://unsplash.com/s/photos/still?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText'>Unsplash.</a> All images are by author unless specified otherwise.
#     </strong>
# </figcaption>

# ## Setup

# In[1]:


import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
plt.style.use("ggplot")


# ## Introduction <small id='intro'></small>
# 
# Unlike ordinary machine learning problems, time series forecasting requires *extra* preprocessing steps.
# 
# On top of the normality assumptions, most ML algorithms expect a *static relationship* between the input features and the output.
# 
# A static relationship requires inputs and outputs with constant parameters such as mean, median, and variance. In other words, algorithms perform best when the inputs and outputs are **stationary**.
# 
# This is not the case in time series forecasting. Distributions that change over time can have unique properties such as seasonality and trend. These, in turn, cause the mean and variance of the series to fluctuate, making it hard to model their behavior.
# 
# So, making a distribution stationary is a strict requirement in time series forecasting. In this article, we will explore several techniques to detect non-stationary distributions and convert them into stationary data.

# ## Table of Contents <small id='toc'></small>
# #### [1. Why is stationarity important?](#1)
# #### [2. Examples of non-stationary series](#2)
# #### [3. Detecting non-stationarity statistically](#3)
# #### [4. Transforming non-stationary series to make it stationary](#4)
# #### [5. Summary](#5)
# 
# 

# 
# > Spoiler: Surprisingly, all the target variables turned out to be stationary. I made this notebook anyway as a tutorial on detecting and transforming non-stationary time series data. It may not be useful in this competition but this topic is extremely important in other forecasting problems since almost all real-world time-series data is non-stationary.

# ## Why is stationarity important? <small id='1'></small>

# If a distribution is not stationary, then it becomes tough to model. Algorithms build relationships between inputs and outputs by estimating the core parameters of the underlying distributions.
# 
# When these parameters are all time-dependent, algorithms will face different values at each point in time. And if the time series is granular enough (such as minutes or seconds frequencies), models may even end up with more parameters than actual data.
# 
# This type of variable relationship between inputs and outputs will seriously compromise the decision function of any model. If the relationship keeps changing through time, models end up using an outdated relationship or one that does not contribute to its predictive power.
# 
# Therefore, you must dedicate a certain amount of time to detecting non-stationarity and removing its effects during your workflow.
# 
# We will see an example of this in the coming sections.

# ## Examples of non-stationary series <small id='2'></small>

# <a href='#toc'>Back to Top 🔝</a>
# 
# Take a look at these plots and try to guess which of the lines represent a stationary series:
# 
# ![](https://otexts.com/fpp2/fpp_files/figure-html/stationary-1.png)
# <figcaption style="text-align: center;">
#     <strong>
#         Image by <a href='https://otexts.com/fpp2/'>otexts.com</a> [1]
#     </strong>
# </figcaption>

# Since stationary series have constant variance, we can rule out **a, c, e, f**, and **i**. These plots show a clear upward or downward trend or changing levels like in f.
# 
# Similarly, as **d** and **h** show seasonal patterns, we can rule them out too. 
# 
# But how about **g** - the pattern does look it is seasonal.
# 
# **g** is the plot of the [lynx](https://en.wikipedia.org/wiki/Lynx) population growth. When food becomes scarce, they stop breeding, causing the population numbers to plummet. When the food sources replenish, they start reproducing again, making the population grow.
# 
# This cyclic behavior is not the same as seasonality. When seasonality exists, you know exactly what will happen after a certain period of time. In contrast, the cyclic behavior of lynx population growth is unpredictable. You can't guess the timing of the food cycles and this makes the series stationary.
# So, the only stationary series are b and g.
# 
# Tricky distributions like **d, f**, and **h** can make you question whether identifying non-stationary data visually is the best option. As you observed, it is pretty easy to confuse seasonality with random cycles or trends with white noise.
# 
# For this reason, the next section will be about statistical methods of detecting non-stationary time series.

# ## Detecting non-stationarity statistically <small id='3'></small>

# <a href='#toc'>Back to Top 🔝</a>
# 
# In the statistics world, there are several tests under the label of [unit root tests](https://medium.com/r/?url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FUnit_root_test%23%3A~%3Atext%3DIn%2520statistics%252C%2520a%2520unit%2520root%2Cdepending%2520on%2520the%2520test%2520used.). The augmented Dickey-Fuller test may be the most popular one, and we have already seen how to use it to detect random walks in [my last kernel](https://www.kaggle.com/bextuychiev/how-to-detect-white-noise-and-random-walks-in-ts).
# 
# Here, we will see how to use it to check if a series is stationary or not.
# 
# Simply put, here are the null and alternative hypotheses of this test:
# - **The null hypothesis**: the distribution is non-stationary, time-dependent (it has a unit root)
# - **The alternative hypothesis**: the distribution is stationary, not time-dependent (can't be represented by a unit root).
# 
# The p-value determines the result of the test. If it is smaller than a critical threshold of 0.05 or 0.01, we **reject** the null hypothesis and conclude that the series is stationary. Otherwise, we **fail to reject** the null and conclude the series is non-stationary.
# 
# The full test is conveniently implemented as `adfuller` function under `statsmodels`. First, let's use it on a distribution we know is stationary and get familiarized with its output:

# In[2]:


import seaborn as sns
from statsmodels.tsa.stattools import adfuller

diamonds = sns.load_dataset("diamonds")

test_results = adfuller(diamonds["price"])

print(f"ADF test statistic: {test_results[0]}")
print(f"p-value: {test_results[1]}")
print("Critical thresholds:")
for key, value in test_results[4].items():
    print(f"\t{key}: {value}")


# We look at the p-value, which is almost 0. This means we can easily reject the null and consider the distribution as stationary.
# 
# Now, let's load the TPS July Playground dataset from Kaggle and check if the target carbon monoxide is stationary:

# In[3]:


tps_july = pd.read_csv(
    "../input/tabular-playground-series-jul-2021/train.csv", parse_dates=["date_time"], index_col="date_time"
)

test_results = adfuller(tps_july["target_carbon_monoxide"])

test_results[1]


# Surprisingly, carbon monoxide is found to be stationary. This might be because the data is recorded over a short period of time, which diminishes the effect of the time component. In fact, all other variables in the data are completely stationary.
# 
# Now, let's load some stocks data that is more likely to be non-stationary:

# In[4]:


amzn = pd.read_csv(
    "https://raw.githubusercontent.com/BexTuychiev/medium_stories/master/2021/july/6_stationarity_ts/data/AMZN.csv",
    parse_dates=["Date"],
    index_col="Date",
)

amzn.plot(figsize=(14, 4));


# As you can see, Amazon stocks show a clear upward trend. Let's perform the Dickey-Fuller test:

# In[5]:


amzn_results = adfuller(amzn["Close"])

amzn_results[1]


# We get a perfect p-value of 1 - this is 100% non-stationary time series data. Let's perform a final test on Microsoft stocks, and we will move on to different techniques you can deal with this type of data:

# In[6]:


msft = pd.read_csv(
    "https://raw.githubusercontent.com/BexTuychiev/medium_stories/master/2021/july/6_stationarity_ts/data/MSFT.csv",
    parse_dates=["Date"],
    index_col="Date",
).dropna()

msft.plot(figsize=(14, 4));


# In[7]:


msft_results = adfuller(msft["Close"])

msft_results[1]


# The p-value is close to 1. No interpretation is necessary.

# ## Transforming non-stationary series to make it stationary <small id='4'></small>
# 
# <a href='#toc'>Back to Top 🔝</a>

# One method for transforming the simplest non-stationary data is differencing. This process involves taking the differences of consecutive observations. Pandas has a `diff` function to do this:

# In[8]:


msft["diff_1"] = msft["Close"].diff(periods=1)
msft["diff_2"] = msft["Close"].diff(periods=2)
msft["diff_3"] = msft["Close"].diff(periods=3)

msft.head(6)


# The output above shows the results of first, second, and third-order differencing.
# 
# For simple distributions, taking the first-order difference is enough to make it stationary. Let's check this by using the `adfuller` function on the `diff_1` (first-order difference of Microsoft stocks):

# In[9]:


results = adfuller(msft["diff_1"].dropna())

results[1]


# When we run `adfuller` on the original distribution of Microsoft stocks, the p-value was close to 1. After differencing, the p-value is flat 0, suggesting we reject the null and conclude the series is now stationary.
# 
# However, some distributions may not be so easy to deal with. Going back to Amazon stocks:

# In[10]:


amzn.plot(figsize=(14, 4));


# Before taking the difference, we have to account for that obvious non-linear trend. Otherwise, the series will still be non-stationary.
# 
# To remove non-linearity, we will use the logarithmic function `np.log` and then, take the first-order difference:

# In[11]:


transformed_amzn = pd.Series(np.log(amzn["Close"])).diff().dropna()

transformed_amzn.plot(figsize=(14, 4));


# In[12]:


results = adfuller(transformed_amzn)

results[1]


# As you can see, the distribution that returned a perfect p-value before transformation is now completely stationary.
# 
# Let's look at another example. Below is the plot of monthly antibiotics sales in Australia:

# In[13]:


drugs = pd.read_csv("https://raw.githubusercontent.com/BexTuychiev/medium_stories/master/2021/july/6_stationarity_ts/data/australia_drug_sales.csv", index_col=0)

drugs.plot("time", "value", figsize=(20, 5))
plt.xlabel("Year")
plt.ylabel("Anibiotics Sold (mln)");


# As you can see, the series shows both an upward trend and a strong seasonality. We will again apply a log transform and, this time, take a yearly difference (12 months) to remove the seasonality.
# 
# Here is what each step looks like:

# In[14]:


fig, ax = plt.subplots(3, 1, figsize=(20, 15))

ax[0].plot(drugs["time"], drugs["value"], label="Original series")
ax[1].plot(drugs["time"], np.log(drugs["value"]), label="After log transform")

drugs["transformed"] = pd.Series(np.log(drugs["value"])).diff(periods=12)  # 12 months
drugs.dropna(inplace=True)
ax[2].plot(drugs["time"], drugs["transformed"], label="After differencing");


# We can confirm the stationarity with `adfuller`:

# In[15]:


results = adfuller(drugs["transformed"])

results[1]


# The p-value is extremely small, proving that the transformation steps have shown their effect.
# 
# In general, every distribution is different, and to achieve stationarity, you might end up chaining multiple operations. Most of these involve taking logarithms, first/second-order, or seasonal differencing.

# ## Summary <small id='5'></small>

# Another topic is done and dusted in this Time Series forecasting series!
# 
# We have already covered a lot, even though we haven't gotten around to the actual forecasting part. By now, you should be able to:
# 
# - Manipulate time-series data like a pro using pandas ([link](https://www.kaggle.com/bextuychiev/every-pandas-function-to-manipulate-time-series)).
# - Dissect any time series into core components such as seasonality and trend ([link](https://www.kaggle.com/bextuychiev/advanced-time-series-analysis-decomposition)).
# - Analyze time-series signals using autocorrelation ([link](https://www.kaggle.com/bextuychiev/advanced-time-series-analysis-decomposition)).
# - Identify if the target you want to predict is white noise or follows a random walk ([link](https://www.kaggle.com/bextuychiev/how-to-detect-white-noise-and-random-walks-in-ts)).
# 
# And finally, you learned how to remove the effect of non-stationary from any time series. All of these are must-know topics and important building blocks to forecasting. The next post is on **time-series feature engineering**. Don't miss it!
