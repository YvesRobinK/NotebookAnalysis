#!/usr/bin/env python
# coding: utf-8

# ## [TPS-Jan] Happy New Year 🌅
# 
# > Thank you to Kaggle management for conducting TPS in 2022 following 2021!
# 
# **Keywords**
# 
# - Time Series 
# - Regression
#     - targe value : `num_sold`

# ## Import Library & Dataset

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# reference : https://www.kaggle.com/subinium/dark-mode-visualization-apple-version
from cycler import cycler

mpl.rcParams['figure.dpi'] = 120
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
# mpl.rcParams['font.family'] = 'serif'

raw_light_palette = [
    (0, 122, 255), # Blue
    (255, 149, 0), # Orange
    (52, 199, 89), # Green
    (255, 59, 48), # Red
    (175, 82, 222),# Purple
    (255, 45, 85), # Pink
    (88, 86, 214), # Indigo
    (90, 200, 250),# Teal
    (255, 204, 0)  # Yellow
]

light_palette = np.array(raw_light_palette)/255


mpl.rcParams['axes.prop_cycle'] = cycler('color',light_palette)

survived_palette = ['#dddddd', mpl.colors.to_hex(light_palette[2])]
sex_palette = [light_palette[0], light_palette[3]]


# In[3]:


# https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side
from IPython.display import display, HTML

CSS = """
.output {
    flex-direction: row;
}
"""

HTML('<style>{}</style>'.format(CSS))


# In[4]:


train = pd.read_csv('../input/tabular-playground-series-jan-2022/train.csv')
test = pd.read_csv('../input/tabular-playground-series-jan-2022/test.csv')
print('train shape : ', train.shape)
print('test shape : ', test.shape)
train.head()


# The date column type is text. Convert to datetime type for easy handling in pandas.

# In[5]:


train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])


# ## Simple Stats
# 
# There are a total of 3 columns: country, store, and product.

# In[6]:


train.describe(include='O')


# In[7]:


for col in ['country', 'store', 'product']:
    print(col, train[col].unique())


# Train dataset is data for 3 years from 2015 to 2018, and test dataset is data for 1 year from 2019.

# In[8]:


print('train date duration : ', train['date'].min(), train['date'].max())
print('test date duration : ', test['date'].min(), test['date'].max())


# As a result of counting for each column, it appears to be a (very) balanced dataset.

# In[9]:


for col in ['country', 'store', 'product']:
    display(pd.DataFrame(train[col].value_counts()))


# In[10]:


for col in ['country', 'store', 'product']:
    display(pd.DataFrame(test[col].value_counts()))


# ## Preprocessing & Visualization
# 
# Time series data needs to be preprocessed for data analysis, feature engineering, visualization, etc.
# 
# ### Pivot Table (time)
# 
# - by date

# In[11]:


train_date = train.set_index('date').pivot(columns=['country', 'store', 'product'], values='num_sold')
train_date.head()


# - by month
#     - For grouping by month, I recommend using pandas' latest feature grouper.

# In[12]:


train_month = train.set_index('date').groupby([pd.Grouper(freq='M'), 'country', 'store', 'product'])['num_sold'].mean().unstack([1, 2, 3])
train_month_country = train.set_index('date').groupby([pd.Grouper(freq='M'), 'country'])['num_sold'].mean().unstack()
train_month.head()


# Grouping by month makes it easier to see overall trends. You can see the trend of peaking at the beginning and end of the year and decreasing before and after.

# In[13]:


fig, ax = plt.subplots(1, 1, figsize=(12, 7))
train_monthly = train.set_index('date').groupby([pd.Grouper(freq='M')])[['num_sold']].mean()

sns.lineplot(x="date", y="num_sold", data=train, ax=ax, label='daily')
sns.lineplot(x="date", y="num_sold", data=train_monthly, ax=ax, label='monthly mean', color='black')
ax.set_title('Monthly Trend', fontsize=20, fontweight='bold', loc='left', y=1.03)
ax.grid(alpha=0.5)
ax.legend()
plt.show()


# In[14]:


country_daily = train.groupby(['date','country'])[['num_sold']].sum().reset_index(level=[0,1])
for country in train['country'].unique():
    display(country_daily[country_daily['country']==country].sort_values(by=['num_sold'], ascending=False).head(30))


# In[15]:


fig, ax = plt.subplots(1, 1, figsize=(12, 7))
train_monthly_country = train.set_index('date').groupby([pd.Grouper(freq='M'),'country'])[['num_sold']].mean()
sns.lineplot(x="date", y='num_sold', hue='country', data=train_monthly_country, ax=ax)

ax.set_ylabel('num_sold')
ax.set_title('Monthly Trend by Country', fontsize=15, fontweight='bold', loc='left')
ax.grid(alpha=0.5)
plt.show()


# ### Day of Week
# 
# Time-series data, such as product sales, often have different distributions on weekends and weekdays. Using the day of the week as a feature is often very effective.

# In[16]:


train['dayofweek'] = train['date'].dt.dayofweek
test['dayofweek'] = test['date'].dt.dayofweek


# Here's a visualization of the average of the days of the week by month to see the weekend trends:
# 
# **fyi**
# 
# - `0` : mon
# - `1` : tue
# - `2` : wed
# - `3` : thu
# - `4` : fri
# - `5` : sat
# - `6` : sun

# In[17]:


fig, ax = plt.subplots(1, 1, figsize=(12, 7))
train_dayofweek = train.set_index('date').groupby([pd.Grouper(freq='M'), 'dayofweek'])[['num_sold']].mean()

sns.lineplot(x="date", y='num_sold', hue='dayofweek', data=train_dayofweek, ax=ax)
ax.set_title('Trend by day of the week', fontsize=15, fontweight='bold', loc='left')
ax.grid(alpha=0.5)
plt.show()


# The following is a comparison by creating a weekend column in more detail.

# In[18]:


train['weekend'] = train['dayofweek'].apply(lambda x : x >= 5)
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
train_weekend = train.set_index('date').groupby([pd.Grouper(freq='M'), 'weekend'])[['num_sold']].mean()
sns.lineplot(x="date", y="num_sold", hue='weekend', data=train_weekend, ax=ax)
ax.set_title('Weekend vs. Weekday Trend Comparison', fontsize=15, fontweight='bold', loc='left')
ax.grid(alpha=0.5)
plt.show()


# In[19]:


fig, ax = plt.subplots(figsize=(12, 9))
country_dayofweek = pd.pivot_table(train, index='country', columns='dayofweek', values='num_sold', aggfunc=np.mean)
country_dayofweek = pd.DataFrame(country_dayofweek.divide(country_dayofweek.sum(axis=1), axis=0).unstack()).reset_index(level=[0,1])
country_dayofweek.rename(columns={0:'num_sold'}, inplace=True)
# country_dayofweek.reset_index(level=[0,1])
sns.barplot(x='dayofweek', y='num_sold', hue='country',data=country_dayofweek, ax=ax)
ax.grid(axis='y',alpha=0.5, )
ax.set_xticklabels(['MON', 'TUE', 'WED','THU','FRI','SAT','SUN'])
ax.set_title('Percentage by day of the week by country', fontsize=15, fontweight='bold', loc='left')
plt.show()


# It can be seen that even Friday has a higher percentage compared to other days.

# ### Pivot (etc)
# 
# - product ratio by country
# 
# You can check the following to see if there is a preference for each country, and there does not seem to be a significant difference.

# In[20]:


country_product = pd.pivot_table(train, index='country', columns='product', values='num_sold', aggfunc=np.mean)
country_product.divide(country_product.sum(axis=1), axis=0)


# You can check the following to see if there is a preference for each day of week, and there does not seem to be a significant difference.

# In[21]:


country_product_dayofweek = pd.pivot_table(train, index='dayofweek', columns='product', values='num_sold', aggfunc=np.mean)
country_product_dayofweek.divide(country_product_dayofweek.sum(axis=1), axis=0)


# ## Animation(Bar Chart Race)

# In[22]:


get_ipython().system('pip install -qqq bar_chart_race')


# In[23]:


import bar_chart_race as bcr


# ### Bar Chart Race (Country)
# 
# For time series data, you can use bar chart races for fun.

# In[24]:


bcr.bar_chart_race(df=train_month_country,
                   n_bars=3,
                   period_length=800,
                   filename=None)


# ### Bar Chart Race (Detail)

# In[25]:


bcr.bar_chart_race(df=train_month,
                   n_bars=9,
                   period_length=800,
                   filename=None)


# In[ ]:




