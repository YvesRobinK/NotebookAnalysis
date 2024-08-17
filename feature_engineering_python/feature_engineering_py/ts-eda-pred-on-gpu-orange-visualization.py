#!/usr/bin/env python
# coding: utf-8

# In[2]:


# --- CSS STYLE ---
from IPython.core.display import HTML
def css_styling():
    styles = open("../input/2020-cost-of-living/alerts.css", "r").read()
    return HTML("<style>"+styles+"</style>")
css_styling()


# <a id='1'></a>
# # <p style="background-color:orange; font-family:Comic Sans MS; font-size:150%; text-align:center"> 💥 <b>Introduction</b> 💥
#     
# <p style="font-family:Comic Sans MS; font-size:200%; color: #ff7f00; text-align:center;"><b>In this notebook, we will explore variable time-series concepts and build a model that can predict sales.</b></p>
# 
# <br>
# <br>
# 
# <div class="alert warning-alert">
# 📌 <b>Competition Goal : </b>
# 
# <br>&nbsp; This challenge serves as final project for the "How to win a data science competition" Coursera course.
# 
# <br>&nbsp; In this competition you will work with a challenging time-series dataset consisting of daily sales data, kindly provided by one of the largest Russian software firms - 1C Company.
# 
# <br>&nbsp; We are asking you to predict total sales for every product and store in the next month. By solving this competition you will be able to apply and enhance your data science skills.
# </div>
# 
# 
# <div class="alert warning-alert">
# 📌 <b>Data fields : </b>
#     
# <br>&nbsp; <b>ID</b> - an Id that represents a (Shop, Item) tuple within the test set<br>
# &nbsp; <b>shop_id</b> - unique identifier of a shop<br>
# &nbsp; <b>item_id</b> - unique identifier of a product<br>
# &nbsp; <b>item_category_id</b> - unique identifier of item category<br>
# &nbsp; <b>item_cnt_day</b> - number of products sold. You are predicting a monthly amount of this measure<br>
# &nbsp; <b>item_price</b> - current price of an item<br>
# &nbsp; <b>date</b> - date in format dd/mm/yyyy<br>
# &nbsp; <b>date_block_num</b> - a consecutive month number, used for convenience. January 2013 is 0, February 2013 is 1,..., October 2015 is 33<br>
# &nbsp; <b>item_name</b> - name of item<br>
# &nbsp; <b>shop_name</b> - name of shop<br>
# &nbsp; <b>item_category_name</b> - name of item category   
# </div>

# <a id='1'></a>
# # <p style="background-color:orange; font-family:Comic Sans MS ; font-size:150%; text-align:center"> 💥 <b>Data and Packages Imports</b> 💥

# In[3]:


get_ipython().system('pip install pandarallel')


# In[4]:


# BASIC PACKAGES LOAD
import numpy as np
import pandas as pd
import random as rd
import datetime
import calendar
import os
import gc
from pandarallel import pandarallel

# VISUALIZATION
import matplotlib.pyplot as plt
import seaborn as sns
import holoviews as hv
from holoviews import opts
pandarallel.initialize()
hv.extension('bokeh')


# TIME SERIES
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic, kpss
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs


# PREPROCESSING
from sklearn.preprocessing import StandardScaler, LabelEncoder
from itertools import product

# Modelling
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")
plt.style.use('seaborn-whitegrid')


# In[29]:


sns.color_palette("YlOrRd",  as_cmap=True)


# In[55]:


YlOrRd_palette_5 = sns.color_palette("YlOrRd", 50)
sns.palplot(YlOrRd_palette_5)


# In[4]:


# DATA LOAD

sales_train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv", parse_dates = ['date'])
item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")
items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")
submission = pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")
shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")
test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")


# <a id='1'></a>
# # <p style="background-color:orange; font-family:Comic Sans MS; font-size:150%; text-align:center"> 💥 <b>Exploratory Data Analysis</b> 💥

# In[5]:


print(f'items.csv : {items.shape}')
items.head(3)


# In[6]:


print(f'item_categories.csv : {item_categories.shape}')
item_categories.head(3)


# In[7]:


print(f'shops.csv : {shops.shape}')
shops.head(3)


# In[8]:


print(f'sales_train.csv : {sales_train.shape}')
sales_train.head(3)


# In[9]:


print(f'sample_submission.csv : {submission.shape}')
submission.head(3)


# In[10]:


print(f'test.csv : {test.shape}')
test.head(3)


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Data preprocessing for EDA</b></p>

# In[11]:


shops['city_name'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
shops['city_name'].unique()


# In[12]:


shops.loc[shops['city_name']=='!Якутск', 'city_name'] = 'Якутск'
shops['city_code'] = LabelEncoder().fit_transform(shops['city_name']).astype(np.int8)
shops.head(3)


# In[13]:


item_categories['item_maincategory_name'] = item_categories['item_category_name'].str.split(' - ').map(lambda x: x[0])
item_categories['item_maincategory_name'].unique()


# In[14]:


item_categories['item_subcategory_name'] = item_categories['item_category_name'].str.split('-').map(lambda x: '-'.join(x[1:]).strip() if len(x) > 1 else x[0].strip())
item_categories['item_subcategory_name'].unique()


# In[15]:


item_categories.loc[item_categories['item_maincategory_name']=='Игры Android', 'item_maincategory_name'] = 'Игры'
item_categories.loc[item_categories['item_maincategory_name']=='Игры MAC', 'item_maincategory_name'] = 'Игры'
item_categories.loc[item_categories['item_maincategory_name']=='Игры PC', 'item_maincategory_name'] = 'Игры'
item_categories.loc[item_categories['item_maincategory_name']=='Карты оплаты (Кино, Музыка, Игры)', 'item_maincategory_name'] = 'Карты оплаты'
item_categories.loc[item_categories['item_maincategory_name']=='Чистые носители (шпиль)', 'item_maincategory_name'] = 'Чистые носители'
item_categories.loc[item_categories['item_maincategory_name']=='Чистые носители (штучные)', 'item_maincategory_name'] = 'Чистые носители'
item_categories['item_maincategory_id'] = LabelEncoder().fit_transform(item_categories['item_maincategory_name']).astype(np.int8)
item_categories['item_subcategory_id'] = LabelEncoder().fit_transform(item_categories['item_subcategory_name']).astype(np.int8)
item_categories.head(3)


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Merge train and test data</b></p>

# In[16]:


item_info = pd.merge(items, item_categories, on='item_category_id', how='inner')
train_tmp = pd.merge(sales_train,item_info, on='item_id', how='inner')
train = pd.merge(train_tmp, shops, on='shop_id', how='inner')
train.head(3)


# In[17]:


test_tmp = pd.merge(test,item_info, on='item_id', how='inner')
test = pd.merge(test_tmp, shops, on='shop_id', how='inner')
test.head(3)


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Calculating the amount of sales per a day</b></p>

# In[18]:


train['total_sales'] = train['item_price'] * train['item_cnt_day']
train.head(3)


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Converting class type to reduce memory load</b></p>

# In[19]:


# Train
train['date_block_num'] =train['date_block_num'].astype(np.int8)
train['shop_id'] = train['shop_id'].astype(np.int8)
train['item_id'] = train['item_id'].astype(np.int16)
train['item_category_id'] = train['item_category_id'].astype(np.int16)

# Test
test['date_block_num'] = 34
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)
test['item_category_id'] = test['item_category_id'].astype(np.int16)


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Which store sold the most?</b></p>

# In[20]:


shop_rank_df = train.shop_name.value_counts().sort_values(ascending=False)
hv.Bars(shop_rank_df[0:20]).opts(title="Shop Count top20", color="orangered", xlabel="Shop Name", ylabel="Count")\
                            .opts(opts.Bars(width=700, height=500,tools=['hover'],xrotation=45,show_grid=True))


# In[21]:


shop_rank_df = train.shop_name.value_counts().sort_values(ascending=False)
hv.Bars(shop_rank_df[-20:]).opts(title="Worst Sales Shop20", color="orangered", xlabel="Shop Name", ylabel="Count")\
                            .opts(opts.Bars(width=700, height=500,tools=['hover'],xrotation=45,show_grid=True))


# In[22]:


hv.Bars(train['city_name'].value_counts()).opts(title="City Count", color="orangered", xlabel="City Name", ylabel="Count")\
                                            .opts(opts.Bars(width=700, height=500,tools=['hover'],xrotation=45,show_grid=True))


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>Insights</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>👉 Looking at the number of shops,  the shops in Moscow stand out.</b></div>

# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Which item sold the most?</b></p>

# In[23]:


item_rank_df = train.item_name.value_counts().sort_values(ascending=False)
hv.Bars(item_rank_df[0:20]).opts(title="Item Count top20", color="orangered", xlabel="Item Name", ylabel="Count")\
                            .opts(opts.Bars(width=700, height=500,tools=['hover'],xrotation=45,show_grid=True))


# In[24]:


item_rank_df = train.item_name.value_counts().sort_values(ascending=False)
hv.Bars(item_rank_df[-10:]).opts(title="Worst Sales Item20", color="orangered", xlabel="Item Name", ylabel="Count")\
                            .opts(opts.Bars(width=700, height=500,tools=['hover'],xrotation=45,show_grid=True))


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>Insights</b></p>
# 
# ><div class="alert warning-alert" role="alert">👉 <b>The item has a long name, so I can't see the best-selling item. The best-selling item is "ионана нана 1 нана (34*42) 45".</b></div>

# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Which item category sold the most?</b></p>

# In[25]:


item_cat_rank_df = train.item_category_name.value_counts().sort_values(ascending=False)
hv.Bars(item_cat_rank_df[0:20]).opts(title="Item Category Count top20", color="orangered" ,xlabel="Item categories", ylabel="Count")\
                                .opts(opts.Bars(width=700, height=500,tools=['hover'],xrotation=45,show_grid=True))


# In[26]:


item_cat_rank_df = train.item_category_name.value_counts().sort_values(ascending=False)
hv.Bars(item_cat_rank_df[-20:]).opts(title="Worst Sales Item Category20", color="orangered" ,xlabel="Item categories", ylabel="Count")\
                                .opts(opts.Bars(width=700, height=500,tools=['hover'],xrotation=45,show_grid=True))


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>Insights</b></p>
# 
# ><div class="alert warning-alert" role="alert">👉 <b>Movie DVDs, game-related items, and music devices are among the most popular.</b></div>

# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Monthly Aggregation</b></p>

# In[27]:


train[["date_block_num","shop_id","item_id","date","item_price","item_cnt_day","total_sales"]].groupby(["date_block_num","shop_id","item_id"])\
            .agg({"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum","total_sales":"sum"}).head(10)


# In[28]:


monthly_ts = train.groupby(["date_block_num"])["total_sales","item_cnt_day"].sum()
month_ts_sales = hv.Curve(monthly_ts["total_sales"]).opts(title="Monthly Sales Time Series", xlabel="Month", ylabel="Total Sales")
month_ts_cnt = hv.Curve(monthly_ts["item_cnt_day"]).opts(title="Monthly Item Count Time Series", xlabel="Month", ylabel="Item Count")
(month_ts_sales + month_ts_cnt).opts(opts.Curve(width=400, height=300,color="orangered",tools=['hover'],show_grid=True,line_width=5,line_dash='dotted'))


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>Insights</b></p>
# 
# ><div class="alert warning-alert" role="alert">👉 <b>Movie DVDs, game-related items, and music devices are among the most popular.</b></div>

# <a id='1'></a>
# # <p style="background-color:orange; font-family:Comic Sans MS; font-size:150%; text-align:center"> 💥 <b>Single time-series</b> 💥

# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Moving Averages</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>👉 Moving averages are techniques for averaging by moving to a window of defined width. You have to choose this area wisely. If you average the movement with a very wide window, it will be an excessively smooth time series. This is because it nullifies the seasonal effect.</b></div>

# In[29]:


TS = train.groupby(["date_block_num"])["item_cnt_day"].sum()
Rolling_Mean = hv.Curve(TS.rolling(window=12,center=False).mean(), label = 'Mean').opts(color="orange")
Rolling_std = hv.Curve(TS.rolling(window=12,center=False).std(), label = 'std').opts(color="orangered")
(Rolling_Mean * Rolling_std).opts(title="Moving Mean and Std", xlabel="date_block_num", ylabel="item_cnt_day").opts(opts.Curve(width=600, height=300,tools=['hover'],show_grid=True,line_width=5)).opts(legend_position='top_left')


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>Insights</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>👉 We can confirm that "Seasonality" and "Trend" are evident.</b></div>

# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 About the seasonality, trends, and residuals in Sales and Items</b></p>

# In[30]:


sales_dec = sm.tsa.seasonal_decompose(monthly_ts["total_sales"].values,period=12,model="multiplicative").plot()


# In[31]:


item_cnt_dec = sm.tsa.seasonal_decompose(monthly_ts["item_cnt_day"].values,period=12,model="multiplicative").plot()


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>Insights</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>👉 Both data have seasonality. Prices tend to rise and fall in the middle, but the number of items continues to decline.</b></div>

# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 What is a Stationary?</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>👉 Stationary is a time series with constant statistical characteristics even when time changes. Therefore, regardless of the trend of time, the mean, variance, etc. are invariant.</b></div>

# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Why can't Non-Stationary data be used?</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>1. Most statistical prediction methods are designed for normality time series data.<br>
# 2. Predicting Stationary data is relatively easy and stable.<br>
# 3. AR models are essentially linear regression models. It uses its own lags as predictors.<br>
# 4. Linear regression performs well when explanatory variables are uncorrelated.<br>
# 5. Stationarization eliminates autocorrelation, creating explanatory variables in the prediction model independently.<br><br>
#     👉 Therefore, the first step in predicting time series data can be said to be converting nonnormality data into normality data.</b></div>

# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 What are the Stationarization Test methods?</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>1. ADF(Augmented Dicky-Fuller)<br>
#     2. KPSS(Kwiatkowski–Phillips–Schmidt–Shin)<br>3. PP (Phillips-Perron)<br><br>
#     👉 All right, so let's start testing.</b></div>

# In[32]:


# ADF Test
dftest = adfuller(monthly_ts["total_sales"].values, autolag='AIC')
df_output = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lag Used', 'Number of Observation Used'])
for key, value in dftest[4].items():
    df_output['Critical value (%s)'%key] = value
print("Sales")    
print(df_output)
print()

dftest = adfuller(monthly_ts["item_cnt_day"].values, autolag='AIC')
df_output = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lag Used', 'Number of Observation Used'])
for key, value in dftest[4].items():
    df_output['Critical value (%s)'%key] = value
print("Item")    
print(df_output)


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>Insights</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>👉 Sales has a P-value of less than 5%, but Item is above. Therefore, trends and seasonality need to be eliminated.</b></div>

# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Remove Trend and Seasonality</b></p>

# In[33]:


from pandas import Series as Series

# to remove trend
# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced forecast
def inverse_difference(last_ob, value):
    return value + last_ob


# In[34]:


ts = train.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.astype('float')
plt.figure(figsize=(16,16))
plt.subplot(311)
plt.title('Original')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts, color = "orangered")
plt.subplot(312)
plt.title('After De-trend')
plt.xlabel('Time')
plt.ylabel('Sales')
new_ts=difference(ts)
plt.plot(new_ts, color = "orangered")
plt.plot()

plt.subplot(313)
plt.title('After De-seasonalization')
plt.xlabel('Time')
plt.ylabel('Sales')
new_ts=difference(ts,12)       # assuming the seasonality is 12 months long
plt.plot(new_ts, color = "orangered")
plt.plot()


# In[35]:


dftest = adfuller(new_ts, autolag='AIC')
df_output = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lag Used', 'Number of Observation Used'])
for key, value in dftest[4].items():
    df_output['Critical value (%s)'%key] = value
print("Item")    
print(df_output)


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>Insights</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>👉 Now after the transformations, our P-value for the ADF test is well within 5 %. Hence we can assume Stationarity of the series.</b></div>

# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 ACF & PCAF</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>👉 ACF : The ACF is self-correlated with its own time difference. If the autocorrelation is large, the time series data will help predict the future with previous values. The big correlation is because there is a certain pattern.<br><br> 👉 PACF : PACF also delivers similar information, but provides autocorrelation for pure time series and lags, except for autocorrelation contributed from intermediate lags.
# In other words, PACF is used when looking at correlations, except for the factors that affect them.</b></div>

# In[36]:


from matplotlib.collections import PolyCollection, LineCollection

fig, curr_ax = plt.subplots(1, 2, figsize=(16, 3))
sales_acf = sm.graphics.tsa.plot_acf(monthly_ts["item_cnt_day"].values, lags=24,color="#E56717",vlines_kwargs={"colors": 'orange'}, ax = curr_ax[0])
sales_pacf = sm.graphics.tsa.plot_pacf(monthly_ts["item_cnt_day"].values, lags=16, color="#E56717",vlines_kwargs={"colors": 'orange'},ax = curr_ax[1])

for i in range(2):
    for item in curr_ax[i].collections:
        if type(item)==PolyCollection:
            item.set_facecolor("orangered")
fig.suptitle('ACF & PCAF of Items', fontsize = 20, y=1.2)
plt.show()


# In[37]:


fig, curr_ax = plt.subplots(1, 2, figsize=(16, 3))
sales_acf = sm.graphics.tsa.plot_acf(monthly_ts["total_sales"].values, lags=24,color="#E56717",vlines_kwargs={"colors": 'orange'}, ax = curr_ax[0])
sales_pacf = sm.graphics.tsa.plot_pacf(monthly_ts["total_sales"].values, lags=16, color="#E56717",vlines_kwargs={"colors": 'orange'},ax = curr_ax[1])

for i in range(2):
    for item in curr_ax[i].collections:
        if type(item)==PolyCollection:
            item.set_facecolor("orangered")
fig.suptitle('ACF & PCAF of Sales', fontsize = 20, y=1.2)
plt.show()


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>Insights</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>👉 We can see that the ACF and PACF of both data converge to zero relatively quickly.</b></div>

# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Lag Plots</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>👉 Lag Plots is a scatterplot of the time series for its own Lag. It is commonly used to check autocorrelation. If any pattern exists, the time series means that autocorrelation exists. If there is no pattern, the time series is likely to be random white noise.</b></div>

# In[38]:


from pandas.plotting import lag_plot

ax_idcs = [(x, y) for x in range(8) for y in range(4)]
plt.rcParams.update({'ytick.left' : False, 'axes.titlepad' : 10})

fig, axes = plt.subplots(8, 4, figsize=(25, 20),sharex=True, sharey=True, dpi=100)
# for lag, ax_coords in enumerate(ax_idcs):
#     ax_row, ax_col = ax_coords
#     axis = axes[ax_row][ax_col]
#     lag_plot(monthly_ts["item_cnt_day"], lag=lag+1, ax=axis, c="orangered")
#     ax.set_title('Lag' + str(i+1))
for i, ax in enumerate(axes.flatten()[:32]):
    lag_plot(monthly_ts["item_cnt_day"], lag=i+1, ax=ax, c="orangered")
    
    ax.set_title('Lag' + str(i+1))
fig.suptitle('Lag Plots of item', y=1)
plt.tight_layout()
plt.show()


# In[39]:


fig, axes = plt.subplots(8, 4, figsize=(25, 22), sharex=True, sharey=True, dpi=100)
plt.rcParams.update({'ytick.left' : False, 'axes.titlepad' : 10})
for i, ax in enumerate(axes.flatten()[:32]):
    lag_plot(monthly_ts["total_sales"], lag=i+1, ax=ax, c="orangered")
    ax.set_title('Lag' + str(i+1))
fig.suptitle('Lag Plots of Sales', y=1)
plt.tight_layout()
plt.show()


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>Insights</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>👉 We can see that the tightest linear graph appears when lag is 1.</b></div>

# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Checking for outliers</b></p>

# In[40]:


price_bx = hv.BoxWhisker(train[['item_price']].sort_values('item_price',ascending=False)[0:500].values,label='Item Price BoxPlot',vdims='Price').opts(box_fill_color="orangered")
cnt_bx = hv.BoxWhisker(train[['item_cnt_day']].sort_values('item_cnt_day',ascending=False)[0:500].values,label='Item Count Day BoxPlot',vdims='Count').opts(box_fill_color="orangered")
(price_bx + cnt_bx).opts(opts.BoxWhisker(width=400, height=400,show_grid=True,tools=['hover']))


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>Insights</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>👉 Outliers are found in prices and items. It needs to be removed through a detailed search.</b></div>

# <a id='1'></a>
# # <p style="background-color:orange; font-family:Comic Sans MS; font-size:150%; text-align:center"> 💥 <b>Feature engineering for Modelling</b> 💥

# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Load Data for LGBM</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>👉 Let's bring the data back to create a clean modeling dataset.</b></div>

# In[10]:


# DATA LOAD

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
sales = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
item_cats = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Remove Outliers</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>👉 On the boxplot, we found an outlier. With reference to other kernels, the company decided to remove the price of 100,000 or more, and sales of 1001 or more.</b></div>

# In[11]:


train = sales[(sales.item_price < 100000) & (sales.item_price > 0)]
train = train[sales.item_cnt_day < 1001]


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Detect same shops</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>👉 We can find several shop_name duplicates. Therefore, duplication is needed.</b></div>

# In[12]:


print(shops[shops.shop_id == 0]['shop_name'].unique(), shops[shops.shop_id == 57]['shop_name'].unique())
print(shops[shops.shop_id == 1]['shop_name'].unique(), shops[shops.shop_id == 58]['shop_name'].unique())
print(shops[shops.shop_id == 40]['shop_name'].unique(), shops[shops.shop_id == 39]['shop_name'].unique())
print(shops[shops.shop_id == 10]['shop_name'].unique(), shops[shops.shop_id == 11]['shop_name'].unique())


# In[13]:


# Deduplication

# Якутск Орджоникидзе, 56
train.loc[train.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57

# Якутск ТЦ "Центральный"
train.loc[train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58

# Жуковский ул. Чкалова 39м²
train.loc[train.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11

# PостовНаДону ТРК "Мегацентр Горизонт" Островной
train.loc[train.shop_id == 40, 'shop_id'] = 39
test.loc[test.shop_id == 40, 'shop_id'] = 39


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Monthly sales</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>👉 Monthly sales are needed for next month's forecast. Let's change it to a simple train dataset for future predictions.</b></div>

# In[14]:


# Add shop_id, item_id, date_block_num

index_cols = ['shop_id', 'item_id', 'date_block_num']

df = [] 
for block_num in train['date_block_num'].unique():
    cur_shops = train.loc[sales['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = train.loc[sales['date_block_num'] == block_num, 'item_id'].unique()
    df.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

df = pd.DataFrame(np.vstack(df), columns = index_cols,dtype=np.int32)

#Add month sales
group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
group.columns = ['item_cnt_month']
group.reset_index(inplace=True)

df = pd.merge(df, group, on=index_cols, how='left')
df['item_cnt_month'] = (df['item_cnt_month']
                                .fillna(0)
                                .clip(0,20)
                                .astype(np.float16))
df.head(5)


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Concat train and test</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>👉 Train and test data set are combined into one for future feature engineering.</b></div>

# In[15]:


test['date_block_num'] = 34
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)
df = pd.concat([df, test], ignore_index=True, sort=False, keys=index_cols)
df.fillna(0, inplace=True)
df.info()


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Add coordinates in Shop</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>👉 Store the longitude and latitude by taking the city coordinates of the shop. It also maps values between 0 and 4 to a country part.</b></div>

# In[16]:


shops['city'] = shops['shop_name'].apply(lambda x: x.split()[0].lower())
shops.loc[shops.city == '!якутск', 'city'] = 'якутск'
shops['city_code'] = LabelEncoder().fit_transform(shops['city'])

coords = dict()
coords['якутск'] = (62.028098, 129.732555, 4)
coords['адыгея'] = (44.609764, 40.100516, 3)
coords['балашиха'] = (55.8094500, 37.9580600, 1)
coords['волжский'] = (53.4305800, 50.1190000, 3)
coords['вологда'] = (59.2239000, 39.8839800, 2)
coords['воронеж'] = (51.6720400, 39.1843000, 3)
coords['выездная'] = (0, 0, 0)
coords['жуковский'] = (55.5952800, 38.1202800, 1)
coords['интернет-магазин'] = (0, 0, 0)
coords['казань'] = (55.7887400, 49.1221400, 4)
coords['калуга'] = (54.5293000, 36.2754200, 4)
coords['коломна'] = (55.0794400, 38.7783300, 4)
coords['красноярск'] = (56.0183900, 92.8671700, 4)
coords['курск'] = (51.7373300, 36.1873500, 3)
coords['москва'] = (55.7522200, 37.6155600, 1)
coords['мытищи'] = (55.9116300, 37.7307600, 1)
coords['н.новгород'] = (56.3286700, 44.0020500, 4)
coords['новосибирск'] = (55.0415000, 82.9346000, 4)
coords['омск'] = (54.9924400, 73.3685900, 4)
coords['ростовнадону'] = (47.2313500, 39.7232800, 3)
coords['спб'] = (59.9386300, 30.3141300, 2)
coords['самара'] = (53.2000700, 50.1500000, 4)
coords['сергиев'] = (56.3000000, 38.1333300, 4)
coords['сургут'] = (61.2500000, 73.4166700, 4)
coords['томск'] = (56.4977100, 84.9743700, 4)
coords['тюмень'] = (57.1522200, 65.5272200, 4)
coords['уфа'] = (54.7430600, 55.9677900, 4)
coords['химки'] = (55.8970400, 37.4296900, 1)
coords['цифровой'] = (0, 0, 0)
coords['чехов'] = (55.1477000, 37.4772800, 4)
coords['ярославль'] = (57.6298700, 39.8736800, 2) 

shops['city_coord_1'] = shops['city'].apply(lambda x: coords[x][0])
shops['city_coord_2'] = shops['city'].apply(lambda x: coords[x][1])
shops['country_part'] = shops['city'].apply(lambda x: coords[x][2])

shops = shops[['shop_id', 'city_code', 'city_coord_1', 'city_coord_2', 'country_part']]

# Merge
df = pd.merge(df, shops, on=['shop_id'], how='left')


# In[17]:


df.head(5)


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Create derived features in items</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>👉 We will extract the category of the item and the category code here.</b></div>

# In[18]:


map_dict = {
            'Чистые носители (штучные)': 'Чистые носители',
            'Чистые носители (шпиль)' : 'Чистые носители',
            'PC ': 'Аксессуары',
            'Служебные': 'Служебные '
            }

items = pd.merge(items, item_cats, on='item_category_id')

items['item_category'] = items['item_category_name'].apply(lambda x: x.split('-')[0])
items['item_category'] = items['item_category'].apply(lambda x: map_dict[x] if x in map_dict.keys() else x)
items['item_category_common'] = LabelEncoder().fit_transform(items['item_category'])

items['item_category_code'] = LabelEncoder().fit_transform(items['item_category_name'])
items = items[['item_id', 'item_category_common', 'item_category_code']]

# Merge
df = pd.merge(df, items, on=['item_id'], how='left')


# In[19]:


df.head(5)


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Create derived features in date</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>👉 We will extract weekend, day and year-end data from date data. Weekends are set at 4 or 5.</b></div>

# In[20]:


def count_days(date_block_num):
    year = 2013 + date_block_num // 12
    month = 1 + date_block_num % 12
    weeknd_count = len([1 for i in calendar.monthcalendar(year, month) if i[6] != 0])
    days_in_month = calendar.monthrange(year, month)[1]
    return weeknd_count, days_in_month, month

map_dict = {i: count_days(i) for i in range(35)}

df['weeknd_count'] = df['date_block_num'].apply(lambda x: map_dict[x][0])
df['days_in_month'] = df['date_block_num'].apply(lambda x: map_dict[x][1])
df['month'] = df['date_block_num'].apply(lambda x: map_dict[x][2])
df['christmas'] = df['date_block_num'].apply(lambda x: 1 if map_dict[x][2] == 12 else 0)


# In[21]:


df.head(5)


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Additional Sales Data</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>👉 Let's get data on whether the customer first purchased it or not and whether they had previously purchased it.</b></div>

# In[22]:


first_item_block = df.groupby(['item_id'])['date_block_num'].min().reset_index()
first_item_block['item_first_interaction'] = 1

first_shop_item_buy_block = df[df['date_block_num'] > 0].groupby(['shop_id', 'item_id'])['date_block_num'].min().reset_index()
first_shop_item_buy_block['first_date_block_num'] = first_shop_item_buy_block['date_block_num']


# In[23]:


df = pd.merge(df, first_item_block[['item_id', 'date_block_num', 'item_first_interaction']], on=['item_id', 'date_block_num'], how='left')
df = pd.merge(df, first_shop_item_buy_block[['item_id', 'shop_id', 'first_date_block_num']], on=['item_id', 'shop_id'], how='left')

df['first_date_block_num'].fillna(100, inplace=True)
df['shop_item_sold_before'] = (df['first_date_block_num'] < df['date_block_num']).astype('int8')
df.drop(['first_date_block_num'], axis=1, inplace=True)

df['item_first_interaction'].fillna(0, inplace=True)
df['shop_item_sold_before'].fillna(0, inplace=True)
 
df['item_first_interaction'] = df['item_first_interaction'].astype('int8')  
df['shop_item_sold_before'] = df['shop_item_sold_before'].astype('int8') 


# In[24]:


df.head(5)


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Traget lags feature</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>👉 We will extract the lags by last 3.</b></div>

# In[25]:


def lag_feature(df, lags, col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
        df[col+'_lag_'+str(i)] = df[col+'_lag_'+str(i)].astype('float16')
    return df


# In[26]:


#Add sales lags for last 3 months
df = lag_feature(df, [1, 2, 3], 'item_cnt_month')

#Critical point: True or False (Affects qmean calculation)
df['qmean'] = df[['item_cnt_month_lag_1', 
                    'item_cnt_month_lag_2', 
                    'item_cnt_month_lag_3']].mean(skipna=True, axis=1)


# In[27]:


#Add avg shop/item price

index_cols = ['shop_id', 'item_id', 'date_block_num']
group = train.groupby(index_cols)['item_price'].mean().reset_index().rename(columns={"item_price": "avg_shop_price"}, errors="raise")
df = pd.merge(df, group, on=index_cols, how='left')

df['avg_shop_price'] = (df['avg_shop_price']
                                .fillna(0)
                                .astype(np.float16))

index_cols = ['item_id', 'date_block_num']
group = train.groupby(['date_block_num','item_id'])['item_price'].mean().reset_index().rename(columns={"item_price": "avg_item_price"}, errors="raise")


df = pd.merge(df, group, on=index_cols, how='left')
df['avg_item_price'] = (df['avg_item_price']
                                .fillna(0)
                                .astype(np.float16))

df['item_shop_price_avg'] = (df['avg_shop_price'] - df['avg_item_price']) / df['avg_item_price']
df['item_shop_price_avg'].fillna(0, inplace=True)

df = lag_feature(df, [1, 2, 3], 'item_shop_price_avg')
df.drop(['avg_shop_price', 'avg_item_price', 'item_shop_price_avg'], axis=1, inplace=True)


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Traget encoding</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>👉 We will target encodings using items, cities and shops.</b></div>

# In[28]:


#Add target encoding for items for last 3 months 
item_id_target_mean = df.groupby(['date_block_num','item_id'])['item_cnt_month'].mean().reset_index().rename(columns={"item_cnt_month": "item_target_enc"}, errors="raise")
df = pd.merge(df, item_id_target_mean, on=['date_block_num','item_id'], how='left')

df['item_target_enc'] = (df['item_target_enc']
                                .fillna(0)
                                .astype(np.float16))

df = lag_feature(df, [1, 2, 3], 'item_target_enc')
df.drop(['item_target_enc'], axis=1, inplace=True)


# In[30]:


#Add target encoding for item/city for last 3 months 
item_id_target_mean = df.groupby(['date_block_num','item_id', 'city_code'])['item_cnt_month'].mean().reset_index().rename(columns={
    "item_cnt_month": "item_loc_target_enc"}, errors="raise")
df = pd.merge(df, item_id_target_mean, on=['date_block_num','item_id', 'city_code'], how='left')

df['item_loc_target_enc'] = (df['item_loc_target_enc']
                                .fillna(0)
                                .astype(np.float16))

df = lag_feature(df, [1, 2, 3], 'item_loc_target_enc')
df.drop(['item_loc_target_enc'], axis=1, inplace=True)


# In[31]:


#Add target encoding for item/shop for last 3 months 
item_id_target_mean = df.groupby(['date_block_num','item_id', 'shop_id'])['item_cnt_month'].mean().reset_index().rename(columns={
    "item_cnt_month": "item_shop_target_enc"}, errors="raise")

df = pd.merge(df, item_id_target_mean, on=['date_block_num','item_id', 'shop_id'], how='left')

df['item_shop_target_enc'] = (df['item_shop_target_enc']
                                .fillna(0)
                                .astype(np.float16))

df = lag_feature(df, [1, 2, 3], 'item_shop_target_enc')
df.drop(['item_shop_target_enc'], axis=1, inplace=True)


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Extra interaction features</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>👉 We will target encodings using category sales.</b></div>

# In[32]:


#For new items add avg category sales for last 3 months
item_id_target_mean = df[df['item_first_interaction'] == 1].groupby(['date_block_num','item_category_code'])['item_cnt_month'].mean().reset_index().rename(columns={
    "item_cnt_month": "new_item_cat_avg"}, errors="raise")

df = pd.merge(df, item_id_target_mean, on=['date_block_num','item_category_code'], how='left')

df['new_item_cat_avg'] = (df['new_item_cat_avg']
                                .fillna(0)
                                .astype(np.float16))

df = lag_feature(df, [1, 2, 3], 'new_item_cat_avg')
df.drop(['new_item_cat_avg'], axis=1, inplace=True)


# In[33]:


#For new items add avg category sales in a separate store for last 3 months
item_id_target_mean = df[df['item_first_interaction'] == 1].groupby(['date_block_num','item_category_code', 'shop_id'])['item_cnt_month'].mean().reset_index().rename(columns={
    "item_cnt_month": "new_item_shop_cat_avg"}, errors="raise")

df = pd.merge(df, item_id_target_mean, on=['date_block_num','item_category_code', 'shop_id'], how='left')

df['new_item_shop_cat_avg'] = (df['new_item_shop_cat_avg']
                                .fillna(0)
                                .astype(np.float16))

df = lag_feature(df, [1, 2, 3], 'new_item_shop_cat_avg')
df.drop(['new_item_shop_cat_avg'], axis=1, inplace=True)


# In[34]:


# Add sales for the last three months for similar item 
# item with id = item_id - 1; kinda tricky feature, but increased the metric significantly
def lag_feature_adv(df, lags, col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)+'_adv']
        shifted['date_block_num'] += i
        shifted['item_id'] -= 1
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
        df[col+'_lag_'+str(i)+'_adv'] = df[col+'_lag_'+str(i)+'_adv'].astype('float16')
    return df

df = lag_feature_adv(df, [1, 2, 3], 'item_cnt_month')


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Remove data for the first three months</b></p>

# In[35]:


df.fillna(0, inplace=True)
df = df[(df['date_block_num'] > 2)]
df.head(5)


# In[36]:


#Save dataset
df.drop(['ID'], axis=1, inplace=True, errors='ignore')
df.to_pickle('df.pkl')


# <a id='1'></a>
# # <p style="background-color:orange; font-family:Comic Sans MS; font-size:150%; text-align:center"> 💥 <b>Modelling</b> 💥

# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 ARIMA Model</b></p>

# In[31]:


ts = train.groupby(["date_block_num"])["item_cnt_day"].sum()


# In[32]:


tslogdiffshifting = ts - ts.shift()


# In[33]:


get_ipython().system('pip install pmdarima')


# In[34]:


from pmdarima import auto_arima

stepwise_fit = auto_arima(ts, trace=True,
suppress_warnings=True)


# In[35]:


model = ARIMA(ts, order=(3,1,1))
results_ARIMA = model.fit(disp=-1)
plt.plot(tslogdiffshifting, color = 'orange')
plt.plot(results_ARIMA.fittedvalues, color='orangered')
print('Plotting ARIMA model')


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>Insights</b></p>
# 
# ><div class="alert warning-alert" role="alert"><b>👉 The Arima model seemed to have a very low prediction rate, with little profit...</b></div>

# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Load data for LGBM</b></p>

# In[37]:


df = pd.read_pickle('df.pkl')
df.info()


# <p style="font-family: Comic Sans MS; line-height: 1;font-size: 20px; letter-spacing: 1px;  color: #ff7f00"><b>📣 Separate Train, Valid and TEST Set
# </b></p>

# In[38]:


X_train = df[df.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = df[df.date_block_num < 33]['item_cnt_month']
X_valid = df[df.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = df[df.date_block_num == 33]['item_cnt_month']
X_test = df[df.date_block_num == 34].drop(['item_cnt_month'], axis=1)
del df


# In[38]:


# !nvidia-smi


# In[39]:


feature_name = X_train.columns.tolist()

params = {
    'objective': 'mse',
    'metric': 'rmse',
    'num_leaves': 15,
    'learning_rate': 0.005,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'bagging_freq': 5,
    'seed': 1,
    'verbose': 1,
    'device': 'gpu',
#     'gpu_platform_id': 0,
#     'gpu_device_id': 0,
    'force_row_wise' : True
}

feature_name_indexes = [ 
                        'country_part',
                        'month',
                        'item_category_common',
                        'item_category_code', 
                        'city_code',
]

lgb_train = lgb.Dataset(X_train[feature_name], Y_train)
lgb_eval = lgb.Dataset(X_valid[feature_name], Y_valid, reference=lgb_train)

evals_result = {}
gbm = lgb.train(
        params, 
        lgb_train,
        num_boost_round= 3000,
        valid_sets=(lgb_train, lgb_eval), 
        feature_name = feature_name,
        categorical_feature = feature_name_indexes,
        verbose_eval=50, 
        evals_result = evals_result,
        early_stopping_rounds = 100)


# In[68]:


YlOrRd_palette_5 = sns.color_palette("YlOrRd", 30 )
sns.palplot(YlOrRd_palette_5)


# In[73]:


fig, ax = plt.subplots(1,1, figsize=(8,6))
lgb.plot_importance(gbm, max_num_features=50,color=YlOrRd_palette_5, importance_type='gain', ax=ax)
ax.set_title("Result ( type : gain )   ", fontweight="bold", fontsize=15)
ax.patch.set_alpha(0) 
plt.show()


# <a id='1'></a>
# # <p style="background-color:orange; font-family:Comic Sans MS; font-size:150%; text-align:center"> 💥 <b>Submission</b> 💥

# In[70]:


test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
Y_test = gbm.predict(X_test[feature_name]).clip(0, 20)

submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test
})
submission.to_csv('gbm_submission.csv', index=False)
submission.head(5)


#     
# <p style="font-family:Comic Sans MS; font-size:200%; color: #ff7f00; text-align:center;"><b><br>Pls, "UPVOTE" if this code helped ! 👀</b></p>

# <a id='1'></a>
# # <p style="background-color:orange; font-family:Comic Sans MS; font-size:150%; text-align:center"> 💥 <b>References</b> 💥
#     
#  <div class="alert warning-alert">   
# 📌 <b>Thank you for always sharing good data🙏🏻 </b><br>
#     
# <br>&nbsp; <b>[Time series Basics : Exploring traditional TS]<br> 👉 https://www.kaggle.com/jagangupta/time-series-basics-exploring-traditional-ts</b><br><br>
# &nbsp; <b>[Prophet/LightGBM - EDA&Feature Engineering&Tuning]<br>👉 https://www.kaggle.com/koheimuramatsu/prophet-lightgbm-eda-feature-engineering-tuning</b><br><br>
# &nbsp; <b>[lsmmay322.log]<br>
#     👉 https://velog.io/@lsmmay322/Kaggle-AirPassnegerst</b><br><br>
# &nbsp; <b>[Modelling]<br>
#     👉 https://www.kaggle.com/uladzimirkapeika/feature-engineering-lightgbm-top-1?select=gbm_submission.csv<br><br>
#     👉 https://www.kaggle.com/uladzimirkapeika/feature-engineering-lightgbm-top-1?select=gbm_submission.csv</b><br>
#      </div>
