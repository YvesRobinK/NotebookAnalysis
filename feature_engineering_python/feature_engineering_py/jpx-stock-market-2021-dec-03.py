#!/usr/bin/env python
# coding: utf-8

# # What Is Technical Analysis?
# 
# There are plenty of ways to predict stock price, or when you should buy or sell it. The overall market, economic data, financial statements and fundamentals can all be beneficial factors to examine when looking at a new investment -- whether a stock or another kind of security.
# 
# But one of the major ways predict stock price is by using technical analysis.
# 
# 
# Unlike its counterpart fundamental analysis, technical analysis examines things like trends and price movement to analyze the viability of a potential investment. But what actually is technical analysis and what are some examples? 
# 
# 
# ![](https://g.foolcdn.com/image/?url=https%3A//g.foolcdn.com/editorial/images/578667/tiny-bull-on-stock-market-key-on-pc-keyboard.jpg&w=2000&op=resize)
# 
# Now take a look at the data we have

# In[1]:


import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

get_ipython().system('pip install -q mplfinance')

import mplfinance as mpl

from ipywidgets import HTML

# Turn off the max column width so the images won't be truncated
pd.set_option('display.max_colwidth', None)
 
# Turning off the max column will display all the data
# if gathering into sets / array we might want to restrict to a few items
pd.set_option('display.max_seq_items', 3)

from multiprocessing import Pool, cpu_count

import warnings
warnings.simplefilter("ignore")


# Parallel df groupby func
def applyParallel(dfGrouped, func):
    with Pool(cpu_count()) as p:
        ret_list = p.map(func, [group for name, group in dfGrouped])
    return pd.concat(ret_list)



# Utility function
def sparkline(data, isBar=False, figsize=(7, 0.75), **kwargs):
    """
    creates a sparkline charts
    """
    data = list(data)
    
    *_, ax = plt.subplots(1, 1, figsize=figsize, **kwargs)
    if isBar:
        ax.bar(list(range(len(data))),data)
    else:
        c= 'green' if data[-1]> data[1] else 'red'
        ax.plot(data, color = c)
        ax.fill_between(range(len(data)), data, len(data)*[min(data)], alpha=0.1, color=c)
    ax.set_axis_off()
    plt.margins(0)

    img = BytesIO()
    plt.savefig(img, pad_inches = 0,bbox_inches= 'tight')
    img.seek(0)
    plt.close()
    return '<img src="data:image/png;base64,{}"/>'.format(base64.b64encode(img.read()).decode())


# In[2]:


df = pd.read_csv('../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv',
                 parse_dates=['Date'])\
        .drop('RowId', axis= 1).merge(
            pd.read_csv('../input/jpx-tokyo-stock-exchange-prediction/stock_list.csv')[['SecuritiesCode', 'Name', '33SectorName', '17SectorName', 'IssuedShares', 'MarketCapitalization']],
            on = 'SecuritiesCode', how = 'left')

df.set_index('Date', inplace= True)

df.head()


# # 2021-Dec-03  JPX Market 
# JPX Tokyo Stock Market on date 2021-Dec-03

# In[3]:


def show_table(df= df, date= '2021-12-03', sort_by= 'Target', asc= False, num_of_rows=10):
    
    # get data in between the date range
    a = df[df.index.isin(np.sort(df.index[df.index <= '2021-12-03'].unique())[-70:])]\
        .groupby('SecuritiesCode').agg(
            Name= pd.NamedAgg(column= 'Name', aggfunc='last'),
            Close= pd.NamedAgg('Close','last'),
            Target= pd.NamedAgg('Target', 'last'),
            Close_list= pd.NamedAgg('Close', pd.Series.tolist),
            Volume_list= pd.NamedAgg('Volume', pd.Series.tolist))\
        .sort_values(sort_by, ascending= asc).iloc[:num_of_rows]
    
    # add close line chart
    a['70_D_Close'] = a['Close_list'].apply(sparkline)
    
    # add volume bar chart
    a['70_D_Volume'] = a['Volume_list'].apply(lambda x: sparkline(x, isBar=True))
    
    
    return display(
        a[['Name', '70_D_Close', 'Close', '70_D_Volume', 'Target']]\
            .style.bar(subset=['Target'], align= 'zero', color= '#C0C0C0')\
            .set_properties(subset=['70_D_Close', '70_D_Volume'], **{'width': '250px'})\
            .set_properties(subset=['Name'], **{'width': '200px'})\
            .format('{:.2f}', subset= ['Close', 'Target']))


display(HTML('<h3>Top Gaining Targets</h3>'))
show_table(asc= False)

display(HTML('<h3>Top Loosing Targets</h3>'))
show_table(asc= True)


# # Sectors
# below we can See a crisp summary of sector-wise action   happening across
# * **Target**
# * **Turover:** (close* volume) Volume measures the number of shares traded in a stock. Volume can indicate market strength.
# * **Market Capitalization:** Market capitalization, commonly called market cap, is the market value of a publicly traded company's outstanding shares.

# In[4]:


plt.style.use('fivethirtyeight')

fig, ax = plt.subplots(1,3, figsize=(20,12), sharey=True)
a = df.loc[df.index=='2021-12-02']
a['Turnover']= a.Close* a.Volume

sns.barplot(
    x='Target',
    y='17SectorName',
    data=a, ax=ax[0],
    errwidth=1.7,
    order= a.groupby('17SectorName').Target.mean().sort_values(ascending=False).index)

sns.barplot(
    x='Turnover',
    y='17SectorName',
    data=a, ax=ax[1],
    estimator= np.sum,
    ci=None,
    errwidth=1.7,
    order= a.groupby('17SectorName').MarketCapitalization.mean().sort_values(ascending=False).index)

sns.barplot(
    x='MarketCapitalization',
    y='17SectorName',
    data=a, ax=ax[2],
    #estimator= np.sum,
    #ci=None,
    errwidth=1.7,
    order= a.groupby('17SectorName').Target.mean().sort_values(ascending=False).index)


for i in ax:
    i.grid(False)
    i.set(ylabel=None)

fig.suptitle('Sectors')
fig.show()


# ## Securities per Sector

# In[5]:


df['17SectorName'].value_counts().plot.pie(
    explode = [0.07 for i in range(17)],
    autopct='%1.1f%%',
    ylabel='',
    figsize=(20,10))


plt.show()


# # AUTOMOBILES & TRANSPORTATION EQUIPMENT Sector
# 
# AUTOMOBILES & TRANSPORTATION EQUIPMENT contribute Highest MarketCap in JPX markets.
# So we See all the stocks ups/downs in AUTOMOBILES & TRANSPORTATION EQUIPMENT sector...

# In[6]:


show_table(df =df[df['17SectorName'] =='AUTOMOBILES & TRANSPORTATION EQUIPMENT '], num_of_rows=100)


# # AdjustmentFactor & Missing Values
# 
# from above table we can clearly see that sudden decreass in price and increas in volume TOYOTA MOTOR CORPORATION stock, this randomness in movement is due to [stock splitting, Dividend](https://www.investopedia.com/terms/a/adjusted_closing_price.asp#:~:text=Key%20Takeaways-,The%20adjusted%20closing%20price%20amends%20a%20stock's%20closing%20price%20to,price%20before%20the%20market%20closes.)
# 
# Adjusted prices are essential when working with historical stock prices. Any time there is a corporate split or dividend, all stock prices prior to that event need to be adjusted to reflect the change.
# 

# In[7]:


def calculate_adjusted_prices(df_security):
    
    # fill missing values
    df_security.fillna(method='ffill', inplace=True)
    
    df_security[['Open', 'High', 'Low', 'Close']]= df_security[['Open', 'High', 'Low', 'Close']]\
        .multiply(
            df_security.AdjustmentFactor.sort_index(ascending=False)\
                .cumprod()\
                .sort_index(ascending=True), 
            axis=0)
    return df_security


def plot_close(df, securities_to_plot, ax=None, name=None, figsize= None):
    df.loc[df.SecuritiesCode.isin(securities_to_plot)].pivot_table(columns= 'Date', index=['SecuritiesCode', 'Name']).Close.T.plot(ax=ax, figsize=figsize)
    if name != None: ax.set_title(name, fontdict ={'fontsize': 30})


# In[8]:


fig, ax = plt.subplots(2,1,figsize=(20,20), sharex=True)

# get 7 SecuritiesCode usefull for plotting
securities_to_plot = df[df.AdjustmentFactor<1].sort_values('Close', ascending = False).SecuritiesCode[70:77].values

# plot close befor adjusting values
plot_close(df, securities_to_plot, ax[0], 'Close Before Adjusting')

# apply adjusting func
df = applyParallel(dfGrouped= df.groupby('SecuritiesCode'), func= calculate_adjusted_prices)

# plot close After adjusting values
plot_close(df, securities_to_plot, ax[1], 'Close After Adjusting')
ax[1].get_legend().remove()

plt.tight_layout()
plt.show()


# # Normalization
# Normalizing time series data is benefitial when we want to compare multiple time series/stock price trends.
# to normalized data we divide security values with startig values.

# In[9]:


def normalizer(df_security):
    # get first nonNull close value as Face Value
    face_value= df_security.loc[df_security.first_valid_index()].Close
    
    # Divide each feature by Face value
    df_security[['Open', 'High', 'Low', 'Close', 'Volume']] = df_security[['Open', 'High', 'Low', 'Close', 'Volume']] / face_value
    return df_security

df= applyParallel(dfGrouped= df.groupby('SecuritiesCode'), func= normalizer)

plot_close(df, securities_to_plot, figsize=(20,10))
plt.title('Close after Normaziation', fontsize= 30)
plt.show()


# In[10]:


df.describe().T


# # Feature Engineering

# In[11]:


a= df[df.SecuritiesCode==7203].iloc[-200:]
mpl.plot(
    a,
    type="candle",
    volume=True,
    mav =(9,20,50),
    title = f" Price",  
    style='yahoo',
    figsize=(18,8))

a.Volume= a.Open.pct_change()
a[['Open', 'High', 'Low', 'Close']] = a[['Open', 'High', 'Low', 'Close']].divide(a.Open, axis= 0)
#a.Volume =a.Volume.pct_change()

mpl.plot(
    a,
    type="candle",
    volume=True,
    mav =(9,20, 50),
    title = f" Price",  
    style='yahoo',
    figsize=(18,8))

