#!/usr/bin/env python
# coding: utf-8

# # EDA & Feature Engineering - Options
# 
# The goal of this notebook is to study options prices and provide a framework for (online) feature engineering. 
# 
# The practical aspect of the notebook mainly rely on my previous work in previous competitions. (Janestreet: https://www.kaggle.com/lucasmorin/running-algos-fe-for-fast-inference, G-research: https://www.kaggle.com/code/lucasmorin/on-line-feature-engineering)
# 
# The theoretical aspect is rather new here on kaggle. I think I can say we didn't really have option content in any of the four previous financial comp. This notebook is designed as an intro to Options from a ML perspective (i.e. not getting too much in details in the quantitative side).

# ## EDA & Features engineering techniques :
# 
# - [Intro](#Intro) 
# - [Imports](#Imports) 
# - [Start with the end](#Start) 
# - [Study Data](#Study_Data)
# - [Option Style](#Option_Style)
# - [Underlying](#Underlying)
# - [Put / Call](#Put_Call)
# - [Horizon / Strike](#Horizon_Strike)
# - [Moneyness](#Moneyness)
# - [OLHC Data](#OLHC)
# - [Price Drivers](#Price_Drivers)
# - [Price Sensitivities (The Greeks)](#Greeks)
# - [Options Features Engineering](#Options_FE) (<- All The Magic)
# - [Complete Feature Exploration](#FE_exploration)

# <a id='Intro'></a>
# # Intro
# 
# An option is a traded contract that gives you a right to buy or sell something (called the 'underlying'). The final transaction is optional, not mandatory (hence the name). Options to buy are called 'Call' and option to sell are called 'Put'. 
# 
# There is usually a contractual price to buy or sell (called the strike) and a limit date (maturity or horizon). The price of the underlying will move with the market and the option will be exercised (use the right to buy or sell) depending on the position of the price realtively to the strike. If you have a call - right to buy - with a strike of 40 and the underlying is 30 you don't exercise, if the price is 50 you exercise, buy for 40 then sell for 50, make 10 profit. 
# 
# Those contract allows to take a leveraged position. In the previous exemple, the option would be worth around 10, significantly lower than the underlying. And if the price move, the option will have more important move (if the underlying move +/-10%, from 50 to 55 or 45, the options profit will move to 15/5, i.e. +/-50%). 

# <a id='Imports'></a>
# # Imports

# In[1]:


import jpx_tokyo_market_prediction
env = jpx_tokyo_market_prediction.make_env()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import pickle

def timestamp_to_date(timestamp):
    return(datetime.fromtimestamp(timestamp))


# <a id='Start'></a>
# # Start with the end
# 
# The goal of this notebook is to work on provided options data. We start with the end by looking at an iteration of the data, so that we know what we will have to deal with for submission. We also load the whole option training data. 

# In[2]:


iter_test = env.iter_test()
(prices, options, financials, trades, secondary_prices, sample_prediction) = next(iter_test)



options_train = pd.read_csv('../input/jpx-tokyo-stock-exchange-prediction/train_files/options.csv')
options_train_sup = pd.read_csv('../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/options.csv')


# <a id='Study_Data'></a>
# # Study Data
# 
# Let's have a look at the data:

# In[3]:


options.head()


# We have some nice specifications:

# In[4]:


options_spec = pd.read_csv('../input/jpx-tokyo-stock-exchange-prediction/data_specifications/options_spec.csv')
options_spec


# So we have:
# 
#     - Date used as reference
#     - Identification Data: OptionsCode (gather other informations)
#     - Caracteristics for the contract: horizon (ContractMonth), Strike (StrikePrice), side (Put-call), and technical dates
#     - OLCH data for different sessions (Night, Day, Whole Sessions)
#     - Other Trading information (Volume, Available Volume, SettlementPrice, BaseVolatility, Interest rate, dividend rate, dividend)
#     - Some already calculated features: TheoreticalPrice, ImpliedVolatility

# # Option style
# 
# **There is mainly two options styles: american and european.** They differ mainly on the way to exercise: european options are only exercised at the end (maturity or horizon), while american options can be exercised at any time up to the end date. American options are more powerfull (hence more expensive), but also more difficult to deal with. 
# 
# Basically for modelling and pricing an european contract you have to build a probabilistic model for the underlying price at the end date. Then you can weight profit associated with each price by the probability to reach that price. For American options you need to build a probabilistic model for each time step you can exercise then build recurring relationships between time steps to know when it's better to exercice. American option modelling usually require adding a complex optimisation layer to a probabilistic modelling framework.
# 
# **As per: https://www.jpx.co.jp/english/derivatives/products/domestic/225options/01.html we have european option style.** Good new as there is a lot of material available online.

# # Underlying

# The last digits of options code refer to the undelying:

# In[5]:


options_train.OptionsCode.astype('str').str[-2:].value_counts()


# **We only have one underlying: the nikkei 225 average.** kind of a bad new (we don't have options data for the tickers we need to predict) only the main indice. Kind of a good new too (data for the whole perimeter would be hard to deal with) and we need to avoid volatility to maximise sharpe. As nikkei is an indice and not directly traded we don't have a corresponding securities code. However It seems we have ETF data.

# In[6]:


stock_list = pd.read_csv('../input/jpx-tokyo-stock-exchange-prediction/stock_list.csv')

s = stock_list['Name'].str.lower()
name = 'nikkei'
number = '225'

# Those filter were designed after reviewing the plots below to remove ETF that are not tracking the index directly
# Inverse and leveraged might be kept to better estimate returns if needed
lv = ['bear','bull','leveraged','double','inverse','futures','50','mini','smdam','nzam']

Nikkei_ETFs = stock_list[s.str.contains(name) & s.str.contains(number) & ~(s.str.contains('|'.join(lv)))]
Nikkei_ticker_list = Nikkei_ETFs.SecuritiesCode.unique()
Nikkei_ETFs.Name


# In[7]:


Nikkei_ticker_list


# In[8]:


ETF_train = pd.read_csv('../input/jpx-tokyo-stock-exchange-prediction/train_files/secondary_stock_prices.csv')
ETF_train = ETF_train[ETF_train.SecuritiesCode.isin(Nikkei_ticker_list)]
ETF_train['ret'] = ETF_train['Close']/ETF_train['Open']-1

ETF_close = ETF_train.pivot(index='Date',columns='SecuritiesCode')['Close']
log_ETF_close = np.log(ETF_close/ETF_close.iloc[-1])

plt.plot(log_ETF_close)
plt.legend(log_ETF_close.columns,loc="upper left");


# In[9]:


ETF_train[['Date','SecuritiesCode','Close']].pivot(index='Date',columns='SecuritiesCode').corr()


# Look at the volume to see if one is "standard":

# In[10]:


ETF_train.groupby('SecuritiesCode')['Volume'].mean()


# **As it has a significantly more important average volume (at 5 times others), we might start with security 1321 as a proxy for the underlying prices.**

# # Put / call

# As the contracts appears standardised at the end of the exchange. There is a balanced number of available contracts:

# In[11]:


plt.plot(options_train.groupby('Date')['Putcall'].mean())


# But volume imbalance between call and put traded (+ a downward global trend):

# In[12]:


put_call_volume = options_train.groupby(['Date','Putcall'])['TradingVolume'].sum().unstack().rename(columns={1:'Put',2:'Call'})

plt.plot(put_call_volume.Put + put_call_volume.Call);
plt.title('Total volume traded')
plt.show();

plt.plot(put_call_volume.Put / put_call_volume.Call);
plt.title('Put/Call ratio')
plt.show();


# The imbalance towards puts is exepected: puts are used by investors to protect against big downwards moves. We also have day volume. We can observe the proportion of trading that happen in the daily session (usually only institutionnal investor can trade in the night session). 

# In[13]:


put_call_day_volume = options_train.groupby(['Date','Putcall'])['WholeDayVolume'].sum().unstack().rename(columns={1:'Put',2:'Call'})

plt.plot((put_call_day_volume.Put + put_call_day_volume.Call)/(put_call_volume.Put + put_call_volume.Call));
plt.title('Day/WholeDay volume ratio')
plt.show();


# # Horizon / strike
# 
# Visualise option chain (available options at different time horizons).

# In[14]:


current_date = options.Date[0]
underlying_close = secondary_prices[secondary_prices.SecuritiesCode==1321].Close
options['horizon'] = pd.to_datetime(options.LastTradingDay.astype('str'))

plt.scatter(pd.to_datetime(options['horizon'].astype('str')),options.StrikePrice,color=options['Putcall'].map({1:'red',2:'blue'}),s=options['OpenInterest']/100)
plt.scatter(pd.to_datetime(current_date),underlying_close,color='black')
plt.xticks(rotation=90);


# Most options are not traded on the day. We migh want to keep (and weight) those that are trade:

# In[15]:


plt.scatter(pd.to_datetime(options['horizon'].astype('str')),options.StrikePrice,color=options['Putcall'].map({1:'red',2:'blue'}),s=options['TradingVolume']/100)
plt.scatter(pd.to_datetime(current_date),underlying_close,color='black')
plt.xticks(rotation=90);


# # Moneyness
# 
# An option is said to be 'in the money' if the current price is above the strike for call (and below fo put). That is, 'in the money' options would make money if exercised right now.

# In[16]:


options['sign'] = options['Putcall'].map({1:-1,2:1})
(options['StrikePrice'].values>underlying_close.values)
options['Moneyness'] = (underlying_close.values-options['StrikePrice'])*options['sign']>0
ITM_options = options[options['Moneyness']]


# We can check our calculation by plotting: we only have put (red) where price is below current strike and call (blue) that have current price above strike.
# The colours are well separated by the current price in the graph below.

# In[17]:


plt.scatter(pd.to_datetime(ITM_options['horizon'].astype('str')),ITM_options.StrikePrice,color=ITM_options['Putcall'].map({1:'red',2:'blue'}),s=ITM_options['OpenInterest']/100)
plt.hlines(underlying_close,xmin=pd.to_datetime(current_date),xmax=pd.to_datetime(ITM_options['horizon'].max()),color='black')
plt.xticks(rotation=90);


# # OLHC Data
# 
# We get 3 series of OLHC data for each options...

# In[18]:


options.iloc[0]


# OLHC data is set to 0 if there is no volume. We take one exemple with volume to see how it works.

# In[19]:


options.loc[4148]


# The night sessions comes before the day session. The Whole day is indeed night + day session. But there seems to be a gap between night and day (from 510 to 490), maybe some premarket session ? On-going work as the functionning of day/night & the settlement is not entirely clear. 
# We might look at unusual prices, but for that we need to have a baseline price. 

# # Price Drivers - Volatility, Dividend, Interest Rate, Dividend Rate

# We get some price and volatility for each options. For European options there are simple standard models that can give a theoretical price depending on some caracteristics of the underlying. One of the main drivers is volatility. As volatility is difficult to observe precisely (we only have some lagged estimation), the pricing framework maybe used the other way around: given some observed price, we can guess the underlying volatility. Each option traded then give us an estimation of the price-implied underlying volatility.

# We can compare market prices to theoretical prices:

# In[20]:


(options['SettlementPrice'] - options['TheoreticalPrice']).hist()


# Or in terms of implied volatilities:

# In[21]:


np.log((options['ImpliedVolatility']/options['BaseVolatility'])).hist()


# This gives an idea of the volatily inferred by market:

# In[22]:


plt.plot(options_train.groupby('Date')[['BaseVolatility','ImpliedVolatility']].mean());


# We have other price drivers from the underlying.

# In[23]:


plt.plot(options_train.groupby('Date')[['Dividend','InterestRate','DividendRate']].mean());


# # Sensitivities (The Greeks)
# 
# **An important serie of tools for dealing with options are the sensitivities of options price to its main drivers.** Main sensitivities are called by a greek letter (hence the name). Those sensitivities are usually constructed using models. For European options we usually have simple models.
# 
# - Delta: sensitivity of option price to underlying price.
# - Gamma: sensitivity of Delta to underlying price.
# - Theta: sensitivity of option price to time passing by.
# - Vega: sensitivity of option price to underlying volatility.
# 
# There is a lot more... see for exemple (https://en.wikipedia.org/wiki/Greeks_(finance)). For simple models of the underlying we have analytical formula for options prices and derivatives. **Fortunately there are some standard python library to build greeks; one of them is mibian.**

# In[24]:


get_ipython().system('pip install mibian')
import mibian


# Looking at documentation (https://github.com/yassinemaaroufi/MibianLib), we can start with a simple (Merton) model.
# 
# ------
# 
# Me - Merton               Used for pricing European options on stocks with dividends

# In[25]:


exemple_option = options.loc[4148]
exemple_option


# In[26]:


underlyingPrice = secondary_prices[secondary_prices.SecuritiesCode==1321].Close
strikePrice = exemple_option.StrikePrice 
interestRate = exemple_option.InterestRate 
annualDividends = exemple_option.DividendRate
daysToExpiration = (exemple_option.horizon - pd.to_datetime(exemple_option.Date)).days
volatility = exemple_option.BaseVolatility
Putcall = exemple_option.Putcall
CallPrice = exemple_option.SettlementPrice


# In[27]:


get_ipython().run_cell_magic('time', '', '\noption_pricer = mibian.Me([underlyingPrice, strikePrice, interestRate, annualDividends, daysToExpiration], volatility=volatility)\n\nGreeks = (\noption_pricer.callPrice,\noption_pricer.callDelta,\noption_pricer.callDelta2,\noption_pricer.callTheta,\noption_pricer.callRho,\noption_pricer.vega,\noption_pricer.gamma,\n)\n')


# In[28]:


Greeks


# Price are quite far of for the moment. I am investigating what is wrong.
# 
# Given anobserved price we can get the implied volatility:

# In[29]:


option_pricer = mibian.Me([underlyingPrice, strikePrice, interestRate, annualDividends, daysToExpiration], volatility=volatility, callPrice=CallPrice)
option_pricer.impliedVolatility


# As you might have noticed there are some symetries between calls and puts. We can find those symmetries in the formulas above (vega and gamma are independant of the put or call side); There is a nice formula exploiting this directly: [the put-call parity](https://en.wikipedia.org/wiki/Put%E2%80%93call_parity). Providing a call and a put corresponding to the same strike and horizon we can calculate the put_call parity associated value:

# In[30]:


strikePrice = options.loc[4148].StrikePrice 
interestRate = options.loc[4148].InterestRate 
annualDividends = options.loc[4148].DividendRate
daysToExpiration = (options.loc[4148].horizon - pd.to_datetime(options.loc[4148].Date)).days / 365 * 250
volatility = options.loc[4148].BaseVolatility
Putcall = options.loc[4148].Putcall

filter_strike = options.StrikePrice == strikePrice
filter_put_call = options.Putcall == 1
filter_horizon = options.horizon == options.loc[4148].horizon

#price is 0 so we filter out
#PutPrice = options[filter_strike&filter_put_call&filter_horizon].WholeDayClose.values
PutPrice = 1

option_pricer = mibian.Me([underlyingPrice, strikePrice, interestRate, annualDividends, daysToExpiration], volatility=volatility, callPrice=CallPrice, putPrice=PutPrice)
option_pricer.putCallParity


# <a id='Options_FE'></a>
# # Options Feature Engineering

# In[31]:


df_day = options_train[options_train.Date=='2017-01-04']


# In[32]:


def options_FE(df_day,date):
    total_volume = df_day.TradingVolume.sum()
    df_put = df_day[df_day.Putcall==1]
    put_volume = df_put.TradingVolume.sum()
    call_volume = total_volume-put_volume
    put_ratio = put_volume/total_volume
    total_volume_day = df_day.WholeDayVolume.sum()
    put_volume_day = df_put.WholeDayVolume.sum()
    call_volume_day = total_volume_day-put_volume_day
    put_ratio_day = put_volume_day/total_volume_day
    call_call_ratio = call_volume_day/call_volume
    put_put_ratio = put_volume_day/put_volume
    df_day['rel_vol'] = df_day['ImpliedVolatility']/df_day['BaseVolatility']
    avg_vol = np.nanmean(df_day['rel_vol'])
    med_vol = np.nanmedian(df_day['rel_vol'])
    std_vol = np.nanstd(df_day['rel_vol'])    
    dividend, IR, DR, vol = df_day[['Dividend','InterestRate','DividendRate','BaseVolatility']].mean()
    
    return [date, total_volume, 
            put_ratio,
            total_volume_day,
            put_ratio_day,
            call_call_ratio,
            put_put_ratio,
            avg_vol,
            med_vol,
            std_vol, 
            dividend, IR, DR, vol]

options_feature_names = ['date','total_volume', 
            'put_ratio',
            'total_volume_day',
            'put_ratio_day',
            'call_call_ratio',
            'put_put_ratio',
            'avg_vol',
            'med_vol',
            'std_vol', 
            'dividend', 'IR', 'DR', 'vol']


# In[33]:


df_result = pd.DataFrame()

options_train_grouped = options_train.groupby('Date')
options_train_grouped_sup = options_train_sup.groupby('Date')

df_result = pd.DataFrame()
list_df = [] 

df_result_sup = pd.DataFrame()
list_df_sup = []  

for date, df in tqdm(options_train_grouped):
    list_df.append(options_FE(df,date))
    
df_result = pd.DataFrame(np.array(list_df),columns=options_feature_names).set_index('date').astype('float32')
del list_df

for date, df in tqdm(options_train_grouped_sup):
    list_df_sup.append(options_FE(df,date))
 
df_result_sup = pd.DataFrame(np.array(list_df_sup),columns=options_feature_names).set_index('date').astype('float32')
del list_df_sup


# In[34]:


df_result.to_parquet('options_train_FE.parquet')
df_result_sup.to_parquet('options_train_FE_sup.parquet')


# <a id='FE_exploration'></a>
# # Complete Feature Exploration

# In[35]:


for c in df_result.columns:
    print(c)
    ((df_result[c]-df_result[c].mean())/df_result[c].std()).plot();
    plt.show()


# In[ ]:




