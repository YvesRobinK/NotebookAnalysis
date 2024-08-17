#!/usr/bin/env python
# coding: utf-8

# # An EDA which makes sense
# 
# Among others, this EDA shows:
# - variation per season and day of the week
# - the effect of Easter
# - zoom-in on end-of-year peak
# - yearly growth (which is higher for the end-of-year peak than for the rest of the year)
# - why this competition is scored by SMAPE
# 
# What next? Look at [the notebook](https://www.kaggle.com/ambrosm/tpsjan22-03-linear-model) which presents feature engineering (based on the insights of this EDA) and a linear model which makes use of the features.

# In[1]:


import pandas as pd
import numpy as np
import dateutil.easter as easter
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor


# In[2]:


train_df = pd.read_csv('../input/tabular-playground-series-jan-2022/train.csv')
test_df = pd.read_csv('../input/tabular-playground-series-jan-2022/test.csv')

# The dates are read as strings and must be converted
for df in [train_df, test_df]:
    df['date'] = pd.to_datetime(df.date)
    df.set_index('date', inplace=True, drop=False)
train_df


# # Completeness of the data
# 
# There are three countries, two stores and three products. For all 18 combinations of these, we have the sales data for 1461 days. The 1461 days are all days of the four years 2015, 2016, 2017, 2018. There are no missing values.

# In[3]:


print(train_df.groupby(['country', 'store', 'product']).date.count())

print("First day:", train_df.date.min(), "   Last day:", train_df.date.max())
print("Number of days in four years:", 365 * 4 + 1) # four years including one leap year
print(18 * 1461, train_df.shape, train_df.date.isna().sum())


# Every product is sold in every store on every day. num_sold is always positive:

# In[4]:


train_df.groupby(['country', 'store', 'product']).num_sold.agg(['min', 'max', 'mean'])


# Whereas the training data covers the year 2015 through 2018, the test data requires us to predict the year 2019:

# In[5]:


test_df.date.min(), test_df.date.max()


# # KaggleRama sells more
# 
# For every country and product, KaggleRama on average sells 1.74 times as much as KaggleMart.
# 
# **Insight:** Maybe it suffices to model KaggleMart and multiply all predictions by 1.74 to get the KaggleRama predictions.

# In[6]:


kk = train_df.groupby(['country', 'store', 'product']).num_sold.mean().unstack(level='store')
kk['KaggleRama:KaggleMart'] = kk.KaggleRama / kk.KaggleMart
kk


# # Products
# 
# If we group the data by country, store, product and year, the ratio Sticker:Mug:Hat is always 1:1.97:3.5 and depends neither on country nor on store nor on year. If we group the data by month, however, the ratio is not constant. This implies that the products have different seasonal variations.
# 
# **Insight:**
# We have to model seasonal effects which depend on the product.

# In[7]:


# Group by year
kk = train_df.groupby(['country', 'store', 'product', train_df.date.dt.year]).num_sold.mean().unstack(level='product')
kk['Mugs/Sticker'] = kk['Kaggle Mug'] / kk['Kaggle Sticker']
kk['Hats/Sticker'] = kk['Kaggle Hat'] / kk['Kaggle Sticker']
kk


# In[8]:


# Group by month
kk = train_df.groupby(['product', train_df.date.dt.month]).num_sold.mean().unstack(level='product')
kk['Mugs/Sticker'] = kk['Kaggle Mug'] / kk['Kaggle Sticker']
kk['Hats/Sticker'] = kk['Kaggle Hat'] / kk['Kaggle Sticker']
kk


# # Histograms and SMAPE
# 
# The histograms for every country-store-product combination show that all histograms are skewed. For every product, there are some days with sales far above the mean. For these outliers, predictions will be much less accurate than for the regular days. This is why the competition is scored by Symmetric mean absolute percentage error ([SMAPE](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error)) rather than MAE or MSE.
# 
# Of course, in a competition scored by SMAPE, we have to carefully choose a loss function for training our models. MSE or MAE are not the ideal loss functions here.
# 
# Every product's histogram has a slightly different shape. The histograms of the Kaggle Hat looks wider than the histograms of the other two products.
# 
# **Insight**
# - Choosing a suitable loss function is important.
# - It may be advantageous to predict log(num_sold) rather than num_sold directly.
# - We need more analysis to understand why the histograms have different shapes. Maybe it is because of the seasonal variations, maybe there is something else.

# In[9]:


plt.figure(figsize=(18, 12))
for i, (combi, df) in enumerate(train_df.groupby(['country', 'store', 'product'])):
    ax = plt.subplot(6, 3, i+1, ymargin=0.5)
    ax.hist(df.num_sold, bins=50, color='pink')
    #ax.set_xscale('log')
    ax.set_title(combi)
plt.suptitle('Histograms of num_sold', y=1.03)
plt.tight_layout(h_pad=3.0)
plt.show()


# # Time series
# 
# ## Daily sales and the year-end peak
# 
# A plot of the daily values of the 18 four_year time series clearly shows high peaks at the end of every year. If we look at the diagrams closely, we see slight waveforms and discern more seasonal effects:
# - (left column:) Kaggle sells more hats in the first half of the year than in the second half (maybe because buyers want to wear Kaggle hats during summer).
# - (middle column:) Demand for mugs is higher in the (northern hemisphere) winter than in summer.
# - (right column:) Sticker sales don't depend on season, except for some small spikes. (All three products have these spikes in the second quarter.)
# 
# **Insight**
# - We have to model seasonal effects which depend on the product.
# - We have to model waves with a wavelength of a year as well as short spikes.

# In[10]:


plt.figure(figsize=(18, 12))
for i, (combi, df) in enumerate(train_df.groupby(['country', 'store', 'product'])):
    ax = plt.subplot(6, 3, i+1, ymargin=0.5)
    #print(df.num_sold.values.shape, df.num_sold.values)
    ax.plot(df.num_sold)
    ax.set_title(combi)
    #if i == 6: break
plt.tight_layout(h_pad=3.0)
plt.suptitle('Daily sales for 2015-2018', y=1.03)
plt.show()


# Let's zoom in on the year-end peak. We plot only the 31 days of December, averaged over the four years. The plots show that sales start growing after Christmas and peak on the 30th of December:
# 

# In[11]:


plt.figure(figsize=(18, 12))
for i, (combi, df) in enumerate(train_df.groupby(['country', 'store', 'product'])):
    ax = plt.subplot(6, 3, i+1, ymargin=0.5)
    ax.bar(range(1, 32),
           df.num_sold[df.date.dt.month==12].groupby(df.date.dt.day).mean(),
           color=['b'] * 25 + ['orange'] * 6)
    ax.set_title(combi)
    ax.set_xticks(ticks=range(5, 31, 5))
plt.tight_layout(h_pad=3.0)
plt.suptitle('Daily sales for December', y=1.03)
plt.show()


# ## Monthly sales and seasonal variation
# 
# A plot of the monthly totals shows the seasonal variation and a growing trend. The growth looks more pronounced fo the stickers than for the hats.
# 
# **Insight**
# - We must ensure that our models can extrapolate the growth to the fifth year.

# In[12]:


plt.figure(figsize=(18, 12))
for i, (combi, df) in enumerate(train_df.groupby(['country', 'store', 'product'])):
    ax = plt.subplot(6, 3, i+1, ymargin=0.5)
    #print(df.resample('MS').num_sold.sum())
    resampled = df.resample('MS').num_sold.sum()
    ax.bar(range(len(resampled)), resampled)
    ax.set_title(combi)
    ax.set_ylim(resampled.min(), resampled.max())
    ax.set_xticks(range(0, 48, 12), [f"Jan {y}" for y in range(2015, 2019)])
plt.suptitle('Monthly sales for 2015-2018', y=1.03)
plt.tight_layout(h_pad=3.0)
plt.show()


# We can see the seasonal variation more clearly if we average over the four years and show only 12 bars for the 12 months:
# - Hats have the maximum in April or May and the minimum in September. They have another (local) maximum in December / January.
# - Mugs have the maximum in December / January and the minimum in June or July. They have a small local maximum in March.
# - Stickers have their maximum in December / January, minimum in February and second maximum May.

# In[13]:


plt.figure(figsize=(18, 12))
for i, (combi, df) in enumerate(train_df.groupby(['country', 'store', 'product'])):
    ax = plt.subplot(6, 3, i+1, ymargin=0.5)
    resampled = df.resample('MS').sum()
    resampled = resampled.groupby(resampled.index.month).mean()
    ax.bar(range(1, 13), resampled.num_sold)
    ax.set_xticks(ticks=range(1, 13), labels='JFMAMJJASOND')
    ax.set_title(combi)
    ax.set_ylim(resampled.num_sold.min(), resampled.num_sold.max())
plt.suptitle('Monthly sales for 2015-2018', y=1.03)
plt.tight_layout(h_pad=3.0)
plt.show()


# ## Growth
# 
# Aggregating the sales per year shows the growth trend. All country-store-product combinations show growth, but there are subtle differences:
# - In Norway, 2016 was a bad year with lower sales than 2015.
# - Sweden had no growth from 2017 to 2018.
# - Almost everywhere, the end-of-year rush grew more than the rest of the year.
# 
# **Insight**
# - We have to model a growth rate which depends on the country.
# - The growth is neither linear nor exponential.
# - We have to model a different growth rate for the end-of-year rush (and maybe other seasonal effects).
# 
# For a broader analysis of the topic, see [this discussion](https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/298318). The outcome was that the Kaggle sales figures depend on the country's GDP.

# In[14]:


plt.figure(figsize=(18, 12))
for i, (combi, df) in enumerate(train_df.groupby(['country', 'store', 'product'])):
    ax = plt.subplot(6, 3, i+1, ymargin=0.5)
    resampled = df.resample('AS').sum()
    ax.bar(range(2015, 2019), resampled.num_sold, color='brown')
    ax.set_title(combi)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # only integer labels
    ax.set_ylim(0, resampled.num_sold.max())
plt.suptitle('Growth of yearly sales for 2015-2018', y=1.03)
plt.tight_layout(h_pad=3.0)
plt.show()


# In[15]:


plt.figure(figsize=(12, 90))
for i, (combi, df) in enumerate(train_df.groupby(['country', 'product', 'store'])):
    ax = plt.subplot(18, 1, i+1, ymargin=0.5)

    # Bar charts (scaled so that 2015 is 1.0)
    resampled = df[(df.date.dt.month<12) | (df.date.dt.day<25)].resample('AS').num_sold.sum()
    resampled /= resampled.iloc[0]
    resampled_end_of_year = df[(df.date.dt.month==12) & (df.date.dt.day>=25)].resample('AS').num_sold.sum()
    resampled_end_of_year /= resampled_end_of_year.iloc[0]
    ax.bar(range(2015, 2019), resampled, color='brown')
    ax.bar(range(2015, 2019), resampled_end_of_year, color='orange', width=0.4)
    
    # Fit exponential growth curves and determine percent growth per year
    X = np.arange(2015, 2019).reshape(-1, 1)
    lr = TransformedTargetRegressor(LinearRegression(), func=np.log, inverse_func=np.exp)
    lr.fit(X, resampled)
    ax.plot(range(2015, 2019), lr.predict(X), color='brown', label=f"whole year: {lr.predict([[2016]]).squeeze() - 1:.1%}")
    lr.fit(X, resampled_end_of_year)
    ax.plot(range(2015, 2019), lr.predict(X), color='orange', label=f"end of year: {lr.predict([[2016]]).squeeze() - 1:.1%}")
    
    ax.legend()
    ax.set_title(f"Yearly sales for {combi}")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # only integer labels
plt.tight_layout(h_pad=3.0)
plt.show()


# ## Weekdays
# 
# Saturdays and Sundays are the best days (highest sales) for all products. Friday seems to be better than Monday through Thursday.
# 
# **Insight**
# - Our model needs to distinguish at least three phases of the week: Mon-Thu, Fri, Sat-Sun.

# In[16]:


plt.figure(figsize=(18, 12))
for i, (combi, df) in enumerate(train_df.groupby(['country', 'store', 'product'])):
    ax = plt.subplot(6, 3, i+1, ymargin=0.5)
    resampled = df.groupby(df.index.dayofweek).mean()
    ax.bar(range(7), resampled.num_sold, 
           color=['b']*4 + ['g'] + ['orange']*2)
    ax.set_title(combi)
    ax.set_xticks(ticks=range(7), labels=['M', 'T', 'W', 'T', 'F', 'S', 'S'])
    ax.set_ylim(0, resampled.num_sold.max())
plt.suptitle('Sales per day of the week', y=1.03)
plt.tight_layout(h_pad=3.0)
plt.show()


# # Easter
# 
# The following diagram shows that during the week after Easter, sales are higher than normal. The diagram shows daily sales for April of the four years. Weekends are colored orange, Easter Sunday is marked red, the week after Easter is colored light blue.
# 
# Easter of 2016 was on the 27th of March; the diagram shows only the last days of the week after Easter.
# 
# **Insight**
# - The model must know the date of Easter and account for higher demand in the week after Easter.

# In[17]:


plt.figure(figsize=(18, 12))
for i, (year, df) in enumerate(train_df.groupby(train_df.date.dt.year)):
    df = df.reset_index(drop=True)
    ax = plt.subplot(4, 1, i+1, ymargin=0.5)
    april = df.num_sold[(df.date.dt.month==4)].groupby(df.date.dt.day).mean()
    date_range = pd.date_range(start=f'{year}-04-01', end=f'{year}-04-30', freq='D')
    easter_date = easter.easter(year)
    color = ['r' if d == easter_date else 'lightblue' if (d.date() - easter_date).days in range(6) else 'b' if d.dayofweek < 5 else 'orange' for d in date_range]
    ax.bar(range(1, 31),
           april,
           color=color)
    ax.set_title(str(year))
    ax.set_xticks(ticks=range(5, 31, 5))
plt.tight_layout(h_pad=3.0)
plt.suptitle('Daily sales for April', y=1.03)
plt.show()


# What next? Look at [a notebook](https://www.kaggle.com/ambrosm/tpsjan22-03-linear-model) which presents feature engineering (based on the insights of this EDA) and a linear model which makes use of the features.

# In[ ]:




