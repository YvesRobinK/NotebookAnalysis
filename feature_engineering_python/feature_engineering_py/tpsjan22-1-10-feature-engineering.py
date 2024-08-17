#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import Markdown as md
from IPython.display import HTML

TITLE = 'feature engineering'
EMOTICON = ' 🧰'

PART_NO = 1
TOTAL_PARTS_NO = 10

ORD_LIST = ['0th', '1st', '2nd', '3rd', '4th'
            '5th', '6th', '7th', '8th', '9th', '10th']

NOTEBOOK_NAME = f"TPSJan22 {PART_NO}/{TOTAL_PARTS_NO}: {TITLE+EMOTICON}"
NOTEBOOK_URL = f"https://www.kaggle.com/fergusfindley/tpsjan22-{PART_NO}-{TOTAL_PARTS_NO}-{TITLE.replace(' ', '-')}"

HTML(f'''<div class="alert alert-block alert-info">
If you find <a href={NOTEBOOK_URL}>this notebook</a> useful or you just like it, please upvote ▲.<br>
If you are using any part of this notebook, please link to <a href={NOTEBOOK_URL}>{NOTEBOOK_NAME}</a> notebook.<br>
In case of any question/feedback don't hesitate to <a href={NOTEBOOK_URL}/comments>comment</a> below.
</div>''')


# <hr>

# In[2]:


HTML(f'''<h1><center>{NOTEBOOK_NAME}</center></h1>
<h2><center><b>TL;DR</b> Very first step on a road to victory.</center></h2>
<center>
<h3><span style="color:#20BEFF;">This is the {ORD_LIST[PART_NO]} part of <a href="https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/301276">TPSJan22 series</a></span></h3></center>''')


# <a id="references"></a>
# <h1 id="references" class="list-group-item list-group-item-action active" data-toggle="list" style='background:#20BEFF; border:0; color:white' role="tab" aria-controls="home"><center>References</center></h1>
#     
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="Go to TableOfContents">Go to TOC</a>

# 
# > * [Kaggle's Time Series Tutorial: Seasonality](https://www.kaggle.com/ryanholbrook/seasonality)
# > * [Accessing World Bank data](https://blogs.worldbank.org/opendata/introducing-wbgapi-new-python-package-accessing-world-bank-data)
# > * [Meteostat library](https://github.com/meteostat/meteostat-python)

# <a id="toc"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h1 id="toc" class="list-group-item list-group-item-action active" data-toggle="list" style='background:#20BEFF; border:0; color:white' role="tab" aria-controls="home"><center>Table of contents</center></h1>

# 0. [References](#references) 🎓
# 1. [Libraries](#libraries) 📚
# 1. [Load Datasets](#load-datasets) 🧱
# 1. [Feature engineering](#feature-engineering) 🧰
#     1. [Basic date-based](#basic-date-based)
#     1. [Time-step, trend and fourier](#time-step-trend-and-fourier)
#     1. [Holidays](#holidays)
#     1. [Easter and Christmas breaks](#easter-xmas)
#     1. [May-June-Nov](#may-june-nov)
#     1. [Exchange rate](#exchange-rate)
#     1. [World Bank's data](#world-bank)
#     1. [Weather](#weather)
#     1. [Kaggle data](#kaggle-data)
# 1. [Save](#save) 💾

# <a id="libraries"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h1 id="libraries" class="list-group-item list-group-item-action active" data-toggle="list" style='background:#20BEFF; border:0; color:white' role="tab" aria-controls="home"><center>Libraries</center></h1>
#     
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="Go to TableOfContents">Go to TOC</a>

# In[3]:


import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 500)

from kaggle_colors_util import *

import gc


# In[4]:


RANDOM_STATE = 42

DIRECTORY_PATH = "../input/tabular-playground-series-jan-2022"
TRAIN_CSV = DIRECTORY_PATH + "/train.csv"
TEST_CSV = DIRECTORY_PATH + "/test.csv"
SUBMISSION_CSV = DIRECTORY_PATH + "/sample_submission.csv"

ID = 'row_id'
TARGET = 'num_sold'
DATE = 'date'

countries = ['Finland', 'Norway', 'Sweden']
countries_iso = ['FIN', 'NOR', 'SWE']
currencies = ['EUR', 'NOK', 'SEK']
stores = ['KaggleMart', 'KaggleRama']
products = ['Kaggle Mug', 'Kaggle Hat', 'Kaggle Sticker']


# In[5]:


# time series data common new feature  
YEAR = "year"
QUARTER = "quarter"
MONTH = "month"
WEEK = "week"
DAY = "day"

DAYOFYEAR = "dayofyear"
DAYOFMONTH = "dayofmonth"
DAYOFWEEK = "dayofweek"
DAY_NAME = "day_name"
MONTH_NAME = "month_name"


# <a id="load-datasets"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h1 id="load-datasets" class="list-group-item list-group-item-action active" data-toggle="list" style='background:#20BEFF; border:0; color:white' role="tab" aria-controls="home"><center>Load Datasets</center></h1>
#     
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="Go to TableOfContents">Go to TOC</a>

# In[6]:


train_df = pd.read_csv(TRAIN_CSV, parse_dates=[DATE], index_col=[ID])
test_df = pd.read_csv(TEST_CSV, parse_dates=[DATE], index_col=[ID])
submission_df = pd.read_csv(SUBMISSION_CSV)

test_ids = test_df.index

dfs_dict = {"train":train_df, "test":test_df}


# In[7]:


df = train_df.copy()
df[DATE] = df[DATE].dt.to_period('D')
df = df.set_index([DATE, 'country', 'store', 'product'])


# In[8]:


Y = df.unstack(['country', 'store', 'product'])


# In[9]:


X = pd.concat([train_df.drop(columns=[TARGET]), test_df])  # easier feature generation for both test&train DataFrames at the same time
test_min_date = test_df[DATE].min()
val_min_date = pd.to_datetime('2018-01-01')  # whole 2018 as a validation set
train_max_date = pd.to_datetime('2017-12-31')


# <a id="feature-engineering"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h1 id="feature-engineering" class="list-group-item list-group-item-action active" data-toggle="list" style='background:#20BEFF; border:0; color:white' role="tab" aria-controls="home"><center>Feature engineering</center></h1>
#     
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="Go to TableOfContents">Go to TOC</a>

# <a id="basic-date-based"></a>
# ## **<span style="color:#58355E;">Basic date-based features</span>**
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="Go to TableOfContents">Go to TOC</a> 

# In[10]:


def get_basic_ts_features(df):
    df[YEAR] = df[DATE].dt.year
    df[QUARTER] = 'Q' + df[DATE].dt.quarter.astype(str)
    df[MONTH] = df[DATE].dt.month
    df[MONTH_NAME] = df[DATE].dt.month_name()
    df[WEEK]= df[DATE].dt.isocalendar().week.astype(int)
    
    df[DAY] = df[DATE].dt.day
    df[DAY_NAME] = df[DATE].dt.day_name()  
    df[DAYOFWEEK] = df[DATE].dt.dayofweek
    df['is_weekend'] = (df[DAYOFWEEK]>=5).astype(int)
    
    df[DAYOFMONTH] = df[DATE].dt.days_in_month

    df[DAYOFYEAR] = df[DATE].dt.dayofyear
    df.loc[(df[DATE].dt.is_leap_year) & (df[DAYOFYEAR] >= 60), DAYOFYEAR] -= 1
    
    return df  


# In[11]:


X = get_basic_ts_features(X)


# <a id="time-step-trend-and-fourier"></a>
# ## **<span style="color:#58355E;">Time-step, trend and Fourier features</span>**
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="Go to TableOfContents">Go to TOC</a> 

# In[12]:


from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

min_date = min(X[DATE])
max_date = max(X[DATE])
date_range = pd.date_range(min_date, max_date)
fourier = CalendarFourier(freq="A", order=14)  # 14 sin/cos pairs for "A"nnual seasonality

dp = DeterministicProcess(
    index=date_range,            # as date_range is with daily frequency weekdays will be on-hot encoded 
    constant=True,               # dummy feature for bias (y-intercept)
    order=3,                     # trend (order 3 means cubed)
    seasonal=True,               # weekly seasonality (indicators)
    additional_terms=[fourier],  # annual seasonality (fourier)
    drop=True,                   # drop terms to avoid collinearity
)

time_df = dp.in_sample().reset_index().rename(columns={'index':DATE})  # create features for dates in date_range

# renaming one-hot encoded weekdays for better interpretability 
# (order is determined by the first day in the dataset - 2015-01-01 is Thursday)

days_dict = dict(zip(list(time_df.filter(regex='s\(\d,\d\)').columns), ['Fri','Sat','Sun','Mon','Tue','Wed']))
time_df = time_df.rename(columns=days_dict)

time_df.loc[:, 'const':'Wed'] = time_df.loc[:, 'const':'Wed'].astype(int)  # memory optimization


# In[13]:


X = X.merge(time_df, on=[DATE])


# <a id="holidays"></a>
# ## **<span style="color:#58355E;">Holidays</span>**
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="Go to TableOfContents">Go to TOC</a> 

# In[14]:


from holidays import CountryHoliday


# In[15]:


def get_country_holidays(country, years_list):
    festivities = CountryHoliday(country, years=years_list)
    festivities_df = pd.DataFrame.from_dict(festivities, orient='index').reset_index().rename(columns={'index':DATE, 0:'festivity_name'})
    festivities_df[DATE] = pd.to_datetime(festivities_df[DATE])
    if country == 'Sweden':
        festivities_df = festivities_df[festivities_df['festivity_name']!='Söndag']  # for Sweden all Sundays are set as holidays... :/
    
    additional_dates = [[pd.to_datetime(f'{year}-12-24'), 'Christmas Eve'] for year in years_list]
    additional_dates += [[pd.to_datetime(f'{year}-12-31'), 'Saint Sylvester'] for year in years_list]
    additional_dates += [[pd.to_datetime(f'{year}-01-01'), 'New Year'] for year in years_list]
    additional_festivities_df = pd.DataFrame(additional_dates, columns=[DATE, 'festivity_name'])    
        
    festivities_df = festivities_df.append(additional_festivities_df, ignore_index=True)
    return festivities_df.sort_values(DATE)


# In[16]:


years_list = list(range(min_date.year, max_date.year+1))

for country_iso in countries_iso:
    X[f'is_festivity_in_{country_iso}'] = X[DATE].isin(get_country_holidays(country_iso, years_list)[DATE]).astype(int)


# In[17]:


def days_till_next_holiday(country, date):
    country_holidays_dates = get_country_holidays(country, [date.year, date.year+1])[DATE]
    next_date = min([holidays_date for holidays_date in country_holidays_dates if holidays_date >= date])
    return (next_date - date).days


# In[18]:


get_ipython().run_cell_magic('time', '', "# TODO how to optimize below as it's over 3 mins run\nX['days_till_next_holiday'] = X.apply(lambda x: days_till_next_holiday(x['country'], x[DATE]), axis=1)\n")


# <a id="easter-xmas"></a>
# ## **<span style="color:#58355E;">Easter and Christmas breaks</span>**
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="Go to TableOfContents">Go to TOC</a> 

# In[19]:


from dateutil.easter import easter
from dateutil.relativedelta import relativedelta


# In[20]:


# The model must know the date of Easter and account for higher demand in the week after Easter.

easter_week_df = pd.DataFrame()
easter_week_df[DATE] = date_range
easter_week_df['easter_week'] = 0

for year in range(min_date.year, max_date.year+1):
    easter_week_dates = pd.date_range(easter(year) - relativedelta(days=1), easter(year) + relativedelta(days=8))
    easter_week_df.loc[easter_week_df[DATE].isin(easter_week_dates), 'easter_week'] = 1

easter_timestamp = easter_week_df[DATE].apply(lambda date: pd.Timestamp(easter(date.year)))
easter_week_df['days_from_easter'] = (easter_week_df[DATE] - easter_timestamp).dt.days.clip(-4, 7)

whitsunday_timestamp = easter_week_df[DATE].apply(lambda date: pd.Timestamp(easter(date.year) + relativedelta(days=49)))
easter_week_df['days_from_whitsunday'] = (easter_week_df[DATE] - whitsunday_timestamp).dt.days.clip(-3, 8)


# In[21]:


X = X.merge(easter_week_df, on=[DATE])


# In[22]:


# There's a clearly higher demand after Christmas and till the end of the first week of January

xmas_newyear_df = pd.DataFrame()
xmas_newyear_df[DATE] = date_range
xmas_newyear_df['xmas_newyear'] = 0

xmas_newyear_peak_df = pd.DataFrame()
xmas_newyear_peak_df[DATE] = date_range
xmas_newyear_peak_df['xmas_newyear_peak'] = 0

for year in range(min_date.year-1, max_date.year+1):  # we need to take into consideration also 2014
    xmas_eve = pd.to_datetime(f'{year}-12-24')
    xmas_newyear_break = pd.date_range(xmas_eve, xmas_eve + relativedelta(days=14))
    xmas_newyear_df.loc[xmas_newyear_df[DATE].isin(xmas_newyear_break), 'xmas_newyear'] = 1
    
    xmas_newyear_peak = pd.date_range(xmas_eve + relativedelta(days=4), xmas_eve + relativedelta(days=8))
    xmas_newyear_peak_df.loc[xmas_newyear_peak_df[DATE].isin(xmas_newyear_peak), 'xmas_newyear_peak'] = 1


# In[23]:


X = X.merge(xmas_newyear_df, on=[DATE])
X = X.merge(xmas_newyear_peak_df, on=[DATE])


# <a id="may-june-nov"></a>
# ## **<span style="color:#58355E;">May-June-Nov</span>**
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="Go to TableOfContents">Go to TOC</a> 

# In[24]:


may_june_nov_df = pd.DataFrame()
may_june_nov_df[DATE] = date_range

# Last Sunday of May
sun_may_date = may_june_nov_df.date.dt.year.map({2015: pd.Timestamp(('2015-05-31')),
                                                 2016: pd.Timestamp(('2016-05-29')),
                                                 2017: pd.Timestamp(('2017-05-28')),
                                                 2018: pd.Timestamp(('2018-05-27')),
                                                 2019: pd.Timestamp(('2019-05-26'))})
may_june_nov_df['days_from_sun_may'] = (may_june_nov_df[DATE] - sun_may_date).dt.days.clip(-7, 3)

# Last Wednesday of June
wed_june_timestamp = may_june_nov_df[DATE].dt.year.map({2015: pd.Timestamp(('2015-06-24')),
                                                        2016: pd.Timestamp(('2016-06-29')),
                                                        2017: pd.Timestamp(('2017-06-28')),
                                                        2018: pd.Timestamp(('2018-06-27')),
                                                        2019: pd.Timestamp(('2019-06-26'))})
may_june_nov_df['days_from_wed_jun'] = (may_june_nov_df[DATE] - wed_june_timestamp).dt.days.clip(-5, 5)

# First Sunday of November (second Sunday is Father's Day)
sun_nov_timestamp = may_june_nov_df[DATE].dt.year.map({2015: pd.Timestamp(('2015-11-1')),
                                                       2016: pd.Timestamp(('2016-11-6')),
                                                       2017: pd.Timestamp(('2017-11-5')),
                                                       2018: pd.Timestamp(('2018-11-4')),
                                                       2019: pd.Timestamp(('2019-11-3'))})
may_june_nov_df['days_from_sun_nov'] = (may_june_nov_df[DATE] - sun_nov_timestamp).dt.days.clip(-1, 9)


# In[25]:


X = X.merge(may_june_nov_df, on=[DATE])


# <a id="exchange-rate"></a>
# ## **<span style="color:#58355E;">Exchange rate</span>**
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="Go to TableOfContents">Go to TOC</a> 

# [forex-python](https://forex-python.readthedocs.io/en/latest/index.html) free foreign exchange rates, bitcoin prices and currency conversion.

# In[26]:


get_ipython().system('pip -qqq install forex-python')

from forex_python.converter import get_rates
# https://ratesapi.io is a free API for current and historical foreign exchange rates published by European Central Bank. The rates are updated daily 3PM CET.


# In[27]:


from tqdm import tqdm

exchange_rate_df = pd.DataFrame()
exchange_rate_df[DATE] = date_range

for day in tqdm(date_range):
    try:
        xr = get_rates("USD", day)
    except RatesNotAvailableError:
        print(f"{day} RatesNotAvailableError: Currency Rates Source Not Ready")
    for currency in currencies:
        exchange_rate_df.loc[exchange_rate_df[DATE]==day, 'xr_'+currency] = xr[currency]


# In[28]:


X = X.merge(exchange_rate_df, on=[DATE])


# <a id="world-bank"></a>
# ## **<span style="color:#58355E;">World Bank's data - macro, financial and sector databases under your fingertips</span>**
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="Go to TableOfContents">Go to TOC</a> 

# [wbgapi](https://pypi.org/project/world-bank-data/) provide access to huge compilation of relevant, high-quality, and internationally comparable statistics about global development and the fight against poverty. The database contains 1,400 time series indicators for 217 economies and more than 40 country groups, with data for many indicators going back more than 50 years.

# In[29]:


get_ipython().system('pip -qqq install wbgapi')

import wbgapi as wb  # pythonic access to the World Bank's data API 


# In[30]:


# with following command one can check variety of topics included in World Bank's data

wb.topic.Series().to_frame().tail()


# In[31]:


# more detailed view for particular country and time frame

wb.data.DataFrame(wb.topic.members(21), 'FIN', time=range(2015, 2018), labels=True, skipBlanks=True).head()  


# In[32]:


# easy to use search feature

# wb.search('Price Index')


# To query World Bank (WB) database you'll need so called `topic` which is string type code like for example *'SP.POP.TOTL'* (Population, total). 
# [Here](http://databank.worldbank.org/data/download/site-content/WDI_CETS.xls) you’ll find a file listing the indicators in the WDI database (which includes the IDS indicators), with the second worksheet containing code breakdowns and descriptions for each component.
# 
# By providing additional information you can get data for specific time frame and/or country (in WB called `economy`)

# In[33]:


def get_topic_data(topic, column_name='values', start_year=2015, end_year=2019, return_log=False):
    
    if return_log:
            return wb.data.DataFrame(topic, economy=countries_iso, time=range(start_year, end_year+1),
                             labels=True, numericTimeKeys=True)\
                             .set_index('Country')\
                             .unstack().reset_index()\
                             .rename(columns={'level_0':'year', 'Country':'country', 0:column_name+'_log'})\
                             .set_index(['year', 'country']).apply(np.log1p)
        
    return wb.data.DataFrame(topic, economy=countries_iso, time=range(start_year, end_year+1),
                             labels=True, numericTimeKeys=True)\
                             .set_index('Country')\
                             .unstack().reset_index()\
                             .rename(columns={'level_0':'year', 'Country':'country', 0:column_name})\
                             .set_index(['year', 'country'])


# In[34]:


total_pop_df = get_topic_data('SP.POP.TOTL', 'wb_total_population', return_log=True)  # SP.POP.TOTL - Population, total

labor_force_df = get_topic_data('SL.TLF.TOTL.IN', 'wb_labor_force', return_log=True)  # SL.TLF.TOTL.IN - Labor force, total

imports_from_hi_df = get_topic_data('TM.VAL.MRCH.HI.ZS', 'wb_imports_from_hi')  
# TM.VAL.MRCH.HI.ZS - Merchandise imports from high-income economies (% of total merchandise imports)

gdp_df = get_topic_data('NY.GDP.MKTP.KD', 'wb_gdp', return_log=True)  # NY.GDP.MKTP.KD - GDP (constant 2015 US$)

gdp_per_capita_df = get_topic_data('NY.GDP.PCAP.KD', 'wb_gdp_per_capita')  # NY.GDP.PCAP.KD - GDP per capita (constant 2015 US$)

price_index_df = get_topic_data('FP.CPI.TOTL', 'wb_price_index')  # FP.CPI.TOTL - Consumer price index (2010 = 100)

comm_com_df = get_topic_data('BM.GSR.CMCP.ZS', 'wb_comm_com')  # BM.GSR.CMCP.ZS - Communications, computer, etc. (% of service imports, BoP)


# In[35]:


wb_df = pd.concat([total_pop_df, labor_force_df, imports_from_hi_df, 
                   gdp_df, gdp_per_capita_df, price_index_df, comm_com_df], 
                  axis=1)


# In[36]:


X = X.merge(wb_df.reset_index(), on=['year', 'country'])


# In[37]:


geo_df = wb.economy.DataFrame(countries_iso)[['name', 'capitalCity', 'latitude', 'longitude']]  # will be needed later for weather features


# <a id="weather"></a>
# ## **<span style="color:#58355E;">Weather</span>**
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="Go to TableOfContents">Go to TOC</a> 

# In[38]:


get_ipython().system('pip -qqq install meteostat')


# In[39]:


from meteostat import Daily, Point


# In[40]:


meteo_df = pd.DataFrame()
for country in geo_df['name']:
    lat = geo_df.loc[geo_df['name']==country, 'latitude'].values[0]
    long = geo_df.loc[geo_df['name']==country, 'longitude'].values[0]
    
    station = Point(lat, long)

    data = Daily(station, min_date, max_date)
    data = data.fetch()
    data = data.reset_index()[['time', 'tavg', 'snow']].rename(columns={'time':DATE, 'tavg':'meteo_temp_avg', 'snow':'meteo_snow'})
    data['meteo_snow'] = data['meteo_snow'].fillna(0).astype(int)
    data['country'] = country
    meteo_df = meteo_df.append(data)
    
meteo_df = meteo_df.fillna(method='ffill')


# In[41]:


X = X.merge(meteo_df, on=[DATE, 'country'])


# <a id="kaggle-data"></a>
# ## **<span style="color:#58355E;">Kaggle data</span>**
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="Go to TableOfContents">Go to TOC</a> 

# In[42]:


kaggle_comp = pd.read_csv('../input/meta-kaggle/Competitions.csv', parse_dates=['EnabledDate', 'DeadlineDate'])

competition_cols = ['EnabledDate', 'DeadlineDate', 'HostSegmentTitle', 'TotalTeams', 'TotalCompetitors', 'TotalSubmissions', 'Slug', 'Title', 'Id']

kaggle_comp = kaggle_comp.loc[(kaggle_comp['EnabledDate']>'2014-06-01') & (kaggle_comp['EnabledDate']<'2020-06-01'), competition_cols]\
                         .rename(columns={'Id':'CompetitionId'})


# In[43]:


kaggle_teams = pd.read_csv('../input/meta-kaggle/Teams.csv', parse_dates=['LastSubmissionDate'], low_memory=False)

teams_cols = ['Id', 'CompetitionId']

kaggle_teams = kaggle_teams[teams_cols].rename(columns={'Id':'TeamId'})


# In[44]:


kaggle_subs = pd.read_csv('../input/meta-kaggle/Submissions.csv', parse_dates=['SubmissionDate'])

submissions_cols = ['Id', 'SubmittedUserId', 'TeamId', 'SubmissionDate', 'IsAfterDeadline']

kaggle_subs = kaggle_subs.loc[(kaggle_subs['SubmissionDate']>='2015-01-01') & (kaggle_subs['SubmissionDate']<'2020-01-01') & (kaggle_subs['IsAfterDeadline']==False), 
                              submissions_cols]\
                              .rename(columns={'Id':'SubmissionId'})


# In[45]:


kaggle_all_df = kaggle_comp.merge(kaggle_teams, how='left', on='CompetitionId')\
                           .merge(kaggle_subs, how='left', on='TeamId')\


# In[46]:


kaggle_subs_num_df = kaggle_all_df.groupby('SubmissionDate')['SubmissionId'].nunique()\
                                  .to_frame().reset_index()\
                                  .rename(columns={'SubmissionDate':DATE, 'SubmissionId':'kaggle_subs_num'})


# In[47]:


kaggle_comp_num_df = kaggle_all_df.groupby('SubmissionDate')['CompetitionId'].nunique()\
                                  .to_frame().reset_index()\
                                  .rename(columns={'SubmissionDate':DATE, 'CompetitionId':'kaggle_comp_num'})


# In[48]:


X = X.merge(kaggle_subs_num_df, on=[DATE])

X = X.merge(kaggle_comp_num_df, on=[DATE])


# In[49]:


del kaggle_all_df, kaggle_comp, kaggle_subs, kaggle_teams, kaggle_subs_num_df, kaggle_comp_num_df
gc.collect()


# <a id="one-hot-encode"></a>
# ## **<span style="color:#58355E;">One-Hot encoding</span>**
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="Go to TableOfContents">Go to TOC</a> 

# In[50]:


IGNORE_COLS = ['year', 'month', 'dayofweek', 'day_name']
FEATURES = [col for col in X.columns if col not in IGNORE_COLS+[TARGET]+[ID]]


# In[51]:


X = pd.get_dummies(X[FEATURES], drop_first=True)


# <a id="reduce-mem-usage"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h1 id="save" class="list-group-item list-group-item-action active" data-toggle="list" style='background:#20BEFF; border:0; color:white' role="tab" aria-controls="home"><center>Convert types to reduce memory usage</center></h1>
#     
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="Go to TableOfContents">Go to TOC</a>

# In[52]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32',
                'int64', 'uint64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if 'int' in str(col_type).lower():
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
 
    return df


# In[53]:


X = reduce_mem_usage(X)


# In[54]:


X = pd.merge(left=X, left_index=True,
             right=pd.concat([train_df.drop(columns=[TARGET]), test_df]).drop(columns=[DATE]), right_index=True,
             how='left')


# <a id="train-val-test"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h1 id="save" class="list-group-item list-group-item-action active" data-toggle="list" style='background:#20BEFF; border:0; color:white' role="tab" aria-controls="home"><center>Split to train, val and test</center></h1>
#     
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="Go to TableOfContents">Go to TOC</a>

# In[55]:


X_train_df = X.loc[X[DATE]<val_min_date]  # 2015-2017
X_val_df = X.loc[(val_min_date<=X[DATE]) & (X[DATE]<test_min_date)]  # 2018
X_test_df = X.loc[X[DATE]>=test_min_date]  # 2019


# <a id="save"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h1 id="save" class="list-group-item list-group-item-action active" data-toggle="list" style='background:#20BEFF; border:0; color:white' role="tab" aria-controls="home"><center>Save</center></h1>
#     
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="Go to TableOfContents">Go to TOC</a>

# In[56]:


X_train_df.to_pickle("X_train_df.pkl")
X_val_df.to_pickle("X_val_df.pkl")
X_test_df.to_pickle("X_test_df.pkl")


# In[57]:


Y.loc[:train_max_date].to_pickle("Y_train_df.pkl")
Y.loc[val_min_date:].to_pickle("Y_val_df.pkl")

df.loc[(slice(None,train_max_date), slice(None), slice(None)), :].to_pickle("Y_train_stacked_df.pkl")
df.loc[(slice(val_min_date,None), slice(None), slice(None)), :].to_pickle("Y_val_stacked_df.pkl")


# ***

# <div class="alert alert-block alert-success">  
# Yupi! We've reached the end of this notebook 📝 <br>Hope it was helpful 😊
# </div>

# In[58]:


HTML(f'''
<div style="color:white;
           display:fill;
           border-radius:5px;
           background-color:{COLOR_GREY};
           font-size:110%;
           font-family:Verdana;
           letter-spacing:0.5px">
    <p style="padding: 10px; color:white;">
If you find this notebook useful or you just like it, please upvote ▲.<br>
        Use this link <a href={NOTEBOOK_URL}>{NOTEBOOK_NAME}</a> to cite.
        Questions/feedback? → <a href={NOTEBOOK_URL}/comments>comment</a>.
    </p>
</div>''')


# ***
