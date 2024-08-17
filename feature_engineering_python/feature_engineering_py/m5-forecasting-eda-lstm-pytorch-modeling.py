#!/usr/bin/env python
# coding: utf-8

# ## Introduction

# <img src="https://i.imgur.com/bOhLgbV.jpg" height="100px" width="400px">

# # Contents
# 
# * [<font size=4>Understand Business Problem</font>](#1)
# 
# * [<font size=4>Data Overview</font>](#2)
# * [<font size=4>Data Preprocessing</font>](#3)
#   * [Loading Data](#3.1)
#   * [Preparing Data](#3.2)
# * [<font size=4>EDA</font>](#4)
#   * [For each day](#4.1)
#   * [For each item](#4.2)
#   * [For week_days vs week_ends](#4.3)
#   * [For events days](#4.4)
#   * [For each Category](#4.5)
#   * [ For each Department](#4.6)
# * [<font size=4>Modeling</font>](#5)
#   * [Model Formation](#5.1)
#   * [Feature Engineering](#5.2)
#   * [Preparing Train & Validation df](#5.3)
#   * [Pytorch Data Loader](#5.4)
#   * [LSTM+NN Model](#5.5)
#   * [Train and Eval functions](#5.6)
#   * [Run Function](#5.7)
# 

# ## Understand Business Problem <a id="1"></a>
# 
# In the challenge, you are predicting item sales at stores in various locations for two 28-day time periods. Information about the data is found in the [M5 Participants Guide.](https://mofc.unic.ac.cy/m5-competition/)
# 

# ## Data Overview <a id="2"></a>
# 
# The dataset consists of five .csv files.
# 
# ### File 1: calendar.csv
# - Contains the dates on which products are sold. The dates are in a <code>yyyy/dd/mm</code> format.
# 
# - `date`: The date in a “y-m-d” format.
# - `wm_yr_wk`: The id of the week the date belongs to.
# - `weekday`: The type of the day (Saturday, Sunday, ..., Friday).
# - `wday`: The id of the weekday, starting from Saturday.
# - `month`: The month of the date.
# - `year`: The year of the date.
# - `event_name_1`: If the date includes an event, the name of this event.
# - `event_type_1`: If the date includes an event, the type of this event.
# - `event_name_2`: If the date includes a second event, the name of this event.
# - `event_type_2`: If the date includes a second event, the type of this event.
# - snap_CA, snap_TX, and snap_WI: A binary variable (0 or 1) indicating whether the stores of CA, TX or WI allow SNAP 3 purchases on the examined date. 1 indicates that SNAP purchases are allowed.
# 
# 
# 
# ### File 2: sales_train_validation.csv
# - Contains the historical daily unit sales data per product and store <code>[d_1 - d_1913]</code>.
# 
# - `item_id`: The id of the product.
# - `dept_id`: The id of the department the product belongs to.
# - `cat_id`: The id of the category the product belongs to.
# - `store_id`: The id of the store where the product is sold.
# - `state_id`: The State where the store is located.
# - `d_1, d_2, ..., d_i, ... d_1941`: The number of units sold at day i, starting from 2011-01-29.
# 
# ### File 3: sell_prices.csv
# - Contains information about the price of the products sold per store and date.
# 
# - `store_id`: The id of the store where the product is sold.
# - `item_id`: The id of the product.
# - `wm_yr_wk`: The id of the week.
# - `sell_price`: The price of the product for the given week/store. The price is provided per week (average across seven days). If not available, this means that the product was not sold during the examined week. Note that although prices are constant at weekly basis, they may change through time (both training and test set).
# 
# 
# ### File 4: submission.csv
# - Demonstrates the correct format for submission to the competition.
# 
# - Each row contains an `id` that is a concatenation of an `item_id` and a `store_id`, which is either `validation` (corresponding to the Public leaderboard), or `evaluation` (corresponding to the Private leaderboard). You are predicting 28 forecast days `(F1-F28)` of items sold for each row. For the `validation` rows, this corresponds to `d_1914 - d_1941`, and for the `evaluation` rows, this corresponds to `d_1942 - d_1969`. (Note: a month before the competition close, the ground truth for the `validation` rows will be provided.)
# 
# ### File 5: sales_train_evaluation.csv
# 
# - Available one month before the competition deadline. It will include sales for <code>[d_1 - d_1941]</code>.
# 
# In this competition, we need to forecast the sales for <code>[d_1942 - d_1969]</code>. These rows form the evaluation set. The rows <code>[d_1914 - d_1941]</code> form the validation set, and the remaining rows form the training set. Now, since we understand the dataset and know what to predict, let us visualize the dataset.

# In[1]:


import os
import gc
import time
import math
import datetime
from math import log, floor
from sklearn.neighbors import KDTree

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle
from tqdm.notebook import tqdm as tqdm

import seaborn as sns
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import pywt
from statsmodels.robust import mad

import scipy
import statsmodels
from scipy import signal
import statsmodels.api as sm
from fbprophet import Prophet
from scipy.signal import butter, deconvolve
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

import joblib
from tqdm import tqdm_notebook as tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
import sklearn

import warnings
warnings.filterwarnings("ignore")


# In[2]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
    


# ## Data Preprocessing <a id="3"></a>

# ### Load Data <a id="3.1"></a>

# In[3]:


def read_data(PATH):
    print('Reading files...')
    calendar = pd.read_csv(f'{PATH}/calendar.csv')
    calendar = reduce_mem_usage(calendar)
    print('Calendar has {} rows and {} columns'.format(calendar.shape[0], calendar.shape[1]))
    sell_prices = pd.read_csv(f'{PATH}/sell_prices.csv')
    sell_prices = reduce_mem_usage(sell_prices)
    print('Sell prices has {} rows and {} columns'.format(sell_prices.shape[0], sell_prices.shape[1]))
    sales_train_validation = pd.read_csv(f'{PATH}/sales_train_validation.csv')
    print('Sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))
    submission = pd.read_csv(f'{PATH}/sample_submission.csv')
    return calendar, sell_prices, sales_train_validation, submission

calendar, sell_prices, sales_train_validation, submission = read_data("../input/m5-forecasting-accuracy")


# ### Preprocessing Data
# 
# - We need to merge all dataframes as a single dataframe then it's eassy to do some EDA and Modeling
# - Using `pd.melt()` function convert `sales_train_validation` columns `[d_1 - d_1941]` to rows as `demand` column 
# - then we mege `calendar` , `sell_prices`, `sales_train_validation`

# In[4]:


sales_train_validation_melt = pd.melt(sales_train_validation, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name='day', value_name='demand')


# - In this data, The products are sold across `ten stores`, located in `three States` (CA, TX and WI)
# - For each `State` we have some `Stores`
# - In each `Store` we have `3,049 products`
# - For each `Product` belongs to one of three `Category` are `[Hobbies, Foods, Household]`
# - For each `Category` we have some `Department` 
# - For each `Department` we have some `Products`
# 
# **So we need to extract features for each `Product`**
# 
# - For EDA we are going to use one of `Store` 
# - We are going to use `CA_1`

# ### Store -> CA_1

# In[5]:


sales_CA_1 = sales_train_validation_melt[sales_train_validation_melt.store_id == "CA_1"]
new_CA_1 = pd.merge(sales_CA_1, calendar, left_on="day", right_on="d", how="left")
new_CA_1 = pd.merge(new_CA_1, sell_prices, left_on=["store_id", "item_id", "wm_yr_wk"],right_on=["store_id", "item_id", "wm_yr_wk"], how="left")
new_CA_1["day_int"] = new_CA_1.day.apply(lambda x: int(x.split("_")[-1]))


# In[6]:


new_CA_1.head()


# ## EDA <a id="4"></a>

# ### For each day <a id="4.1"></a>
# 
# - For each day we `sum` over products `sell_price` and `demand`
# - For each day we `count_nonzeros` over products `sell_price` and `demand`
# 

# In[7]:


day_sum = new_CA_1.groupby("day_int")[["sell_price", "demand"]].agg("sum").reset_index()


# In[8]:


fig = make_subplots(rows=2, cols=1)

fig.add_trace(go.Scatter(x=day_sum.day_int, 
                         y=day_sum.demand,
                         #showlegend=False,
                         mode="lines",
                         name="demand",
                         #marker=dict(color="mediumseagreen"),
                         ),

              row=1,col=1         
              )

fig.add_trace(go.Scatter(x=day_sum.day_int, 
                         y=day_sum.sell_price,
                         #showlegend=False,
                         mode="lines",
                         name="sell_price",
                         #marker=dict(color="mediumseagreen")
                         ),
             
              row=2,col=1           
              )

fig.update_layout(height=1000, title_text="SUM -> Demand  and Sell_price")
fig.show()


# **Observation :** 
# * From the above sum over product `demand` we observe that some days are "Zeros" because those are `Christmas` days, I think in chirstmas day the store was closed. and we observe some patterns over the years. 
# * From the above sum over product `sell_price` we observe that day-by-day the sells are increasing. at the end its becaming constant.

# In[9]:


# For each day we count_nonzeros over products sell_price and demand

day_sum = new_CA_1.groupby("day_int")[["demand","event_name_1" ]].agg({"demand": np.count_nonzero, "event_name_1": "first"}).reset_index()
def count_nulls(series):
    return len(series) - series.count()

cout_null = new_CA_1.groupby("day_int")["sell_price"].agg(count_nulls).reset_index()


# In[10]:


fig = make_subplots(rows=2, cols=1)

fig.add_trace(go.Scatter(x=cout_null.day_int, 
                         y=cout_null.sell_price,
                         #showlegend=False,
                         mode="lines",
                         name="sell_price",
                         #marker=dict(color="mediumseagreen")
                        ),

              row=1,col=1         
              )

fig.add_trace(go.Scatter(x=day_sum.day_int, 
                         y=day_sum.demand,
                         #showlegend=False,
                         mode="lines",
                         name="demand",
                         #marker=dict(color="mediumseagreen")
                        ),
             
              row=2,col=1           
              )

fig.update_layout(height=1000, title_text="Count_Nonzero -> Sell_price  and Demand")
fig.show()


# **Observation :** 
# * From the above cout_nonzeros over product `demand` we observe that non_zeros count increasing day-by-day
# * From the above cout_null_values over product `sell_price` we observe that day-by-day null values decreasing and at the end we observe that count close to 0

# ### For each item <a id="4.2"></a>
# 
# - For each item we `[max, mean, min]` over days `sell_price` and `demand`
# 

# In[11]:


item_id = new_CA_1.groupby("item_id")[["sell_price", "demand"]].agg({
    "sell_price": ["max", "mean", "min"],
    "demand" : ["max", "mean", "min"]
}).reset_index()


# In[12]:


fig = make_subplots(rows=1, cols=1)

item_id = item_id.sort_values(("sell_price", "max"))
fig.add_trace(go.Scatter(x=item_id["item_id"], 
                         y=item_id["sell_price", "max"],
                         #showlegend=Ture,
                         mode="lines",
                         name="max",
                         #marker=dict(color="mediumseagreen")
                         ),

              row=1,col=1         
              )

fig.add_trace(go.Scatter(x=item_id["item_id"], 
                         y=item_id["sell_price", "mean"],
                         #showlegend=Ture,
                         mode="lines",
                         name="mean",
                         #marker=dict(color="yellow")
                         ),
             
              row=1,col=1           
              )

fig.add_trace(go.Scatter(x=item_id["item_id"], 
                         y=item_id["sell_price", "min"],
                         #showlegend=Ture,
                         mode="lines",
                         name="min",
                         #marker=dict(color="blue")
                         ),
             
              row=1,col=1           
              )

fig.update_layout(height=500, title_text="Sell_price")
fig.show()


# **Observation :** 
# * From the above plot we observe that for every product the prices change over time

# In[13]:


fig = make_subplots(rows=1, cols=1)

item_id = item_id.sort_values(("demand", "max"))
fig.add_trace(go.Scatter(x=item_id["item_id"], 
                         y=item_id["demand", "max"],
                         #showlegend=Ture,
                         mode="lines",
                         name="max",
                         #marker=dict(color="mediumseagreen")
                         ),

              row=1,col=1         
              )

fig.add_trace(go.Scatter(x=item_id["item_id"], 
                         y=item_id["demand", "mean"],
                         #showlegend=Ture,
                         mode="lines",
                         name="mean",
                         #marker=dict(color="yellow")
                         ),
             
              row=1,col=1           
              )

fig.add_trace(go.Scatter(x=item_id["item_id"], 
                         y=item_id["demand", "min"],
                         #showlegend=Ture,
                         mode="lines",
                         name="min",
                         #marker=dict(color="blue")
                         ),
             
              row=1,col=1           
              )

fig.update_layout(height=500, title_text="Demand")
fig.show()


# **Observation :** 
# * From the above plot we observe that some products demend high in some days the min is zero because of Christmas day

# ### For week_days vs week_ends <a id="4.3"></a>
# 
# - For each item week_days vs week_ends over days sell_price and demand

# In[14]:


# For each item week_days vs week_ends over days sell_price and demand

week_end = new_CA_1[new_CA_1.weekday == "Sunday"]
week_day = new_CA_1[new_CA_1.weekday != "Sunday"]

week_end = week_end.groupby("item_id")[["demand", "sell_price"]].agg(["mean", "max"]).reset_index()
week_end.columns = ['_'.join(col).strip() for col in week_end.columns.values]

week_day = week_day.groupby("item_id")[["demand", "sell_price"]].agg(["mean", "max"]).reset_index()
week_day.columns = ['_'.join(col).strip() for col in week_day.columns.values]


# In[15]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=week_end["item_id_"],
                         y=week_end["demand_mean"],
                         mode="lines",
                         name="week_day"

))

fig.add_trace(go.Scatter(x=week_end["item_id_"],
                         y=week_day["demand_mean"],
                         mode="lines",
                         name="normal_day"

))

fig.update_layout(height=500, title_text="Demand")
fig.show()


# **Observation :** 
# * From the above plot we observe that in week days some more demand on some items

# In[16]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=week_end["item_id_"],
                         y=week_end["sell_price_mean"],
                         mode="lines",
                         name="week_day"

))

fig.add_trace(go.Scatter(x=week_end["item_id_"],
                         y=week_day["sell_price_mean"],
                         mode="lines",
                         name="normal_day"

))

fig.update_layout(height=500,title_text="Sell_price")
fig.show()


# **Observation :** 
# * From the above plot we observe that in weekdays and weekends are probably same prices

# ## For events days <a id="4.4"></a>

# In[17]:


events = new_CA_1[~new_CA_1.event_name_1.isna()]
events = events.groupby("event_name_1")[["demand", "sell_price"]].agg(["mean", "max"]).reset_index()
events.columns = ['_'.join(col).strip() for col in events.columns.values]


# In[18]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=events["event_name_1_"],
                         y=events["demand_mean"],
                         mode="lines",
                         name="week_day"
))

fig.update_layout(height=500, title_text="Demand")
fig.show()


# **Observation :** 
# * From the above plot we observe that in some event days the demand increase and decrease 

# In[19]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=events["event_name_1_"],
                         y=events["sell_price_mean"],
                         mode="lines",
                         name="week_day"

))

fig.update_layout(height=500, title_text="Sell_price")
fig.show()


# ### For each Category <a id="4.5"></a>
# 
# - In the data there are three categorys

# In[20]:


## Number of items contain each Category

def n_unique(series):
    return series.nunique()

Category_count = new_CA_1.groupby("cat_id")["item_id"].agg(n_unique).reset_index()


# In[21]:


fig = px.bar(Category_count, y="item_id", x="cat_id", color="cat_id", title="Category Item Count")

fig.update_layout(height=500, width=600)
fig.show()


# **Observation :** From the above bar plot we observe thet each category `FOOD`: 1437 , `HOBBIES` : 565 , `HOUSEHOLD` : 1047 items

# In[22]:


## For each category mean of deman and sell_price

Category = new_CA_1.groupby(["day_int","cat_id"])[["demand", "sell_price"]].agg(["mean", "max"]).reset_index()
Category.columns = ['_'.join(col).strip() for col in Category.columns.values]

FOODS = Category[Category.cat_id_ == "FOODS"]
HOBBIES = Category[Category.cat_id_ == "HOBBIES"]
HOUSEHOLD = Category[Category.cat_id_ == "HOUSEHOLD"]


# In[23]:


fig = make_subplots(rows=1, cols=1)


fig.add_trace(go.Scatter(x=FOODS["day_int_"], 
                         y=FOODS["demand_mean"],
                         #showlegend=Ture,
                         mode="lines",
                         name="FOODS",
                         #marker=dict(color="mediumseagreen")
                         ),

              row=1,col=1         
              )

fig.add_trace(go.Scatter(x=HOBBIES["day_int_"], 
                         y=HOBBIES["demand_mean"],
                         #showlegend=Ture,
                         mode="lines",
                         name="HOBBIES",
                         #marker=dict(color="yellow")
                         ),
             
              row=1,col=1           
              )

fig.add_trace(go.Scatter(x=HOUSEHOLD["day_int_"], 
                         y=HOUSEHOLD["demand_mean"],
                         #showlegend=Ture,
                         mode="lines",
                         name="HOUSEHOLD",
                         #marker=dict(color="blue")
                         ),
             
              row=1,col=1           
              )

fig.update_layout(height=500, title_text="Demand Mean Over Category by day-by-day")
fig.show()


# In[24]:


fig = make_subplots(rows=1, cols=1)


fig.add_trace(go.Scatter(x=FOODS["day_int_"], 
                         y=FOODS["demand_max"],
                         #showlegend=Ture,
                         mode="lines",
                         name="FOODS",
                         #marker=dict(color="mediumseagreen")
                         ),

              row=1,col=1         
              )

fig.add_trace(go.Scatter(x=HOBBIES["day_int_"], 
                         y=HOBBIES["demand_max"],
                         #showlegend=Ture,
                         mode="lines",
                         name="HOBBIES",
                         #marker=dict(color="yellow")
                         ),
             
              row=1,col=1           
              )

fig.add_trace(go.Scatter(x=HOUSEHOLD["day_int_"], 
                         y=HOUSEHOLD["demand_max"],
                         #showlegend=Ture,
                         mode="lines",
                         name="HOUSEHOLD",
                         #marker=dict(color="blue")
                         ),
             
              row=1,col=1           
              )

fig.update_layout(height=500, title_text="Demand Max Over Category by day-by-day")
fig.show()


# **Observation :** From the above plots we observe that `FOOD` category have more demand

# In[25]:


fig = make_subplots(rows=1, cols=1)


fig.add_trace(go.Scatter(x=FOODS["day_int_"], 
                         y=FOODS["sell_price_mean"],
                         #showlegend=Ture,
                         mode="lines",
                         name="FOODS",
                         #marker=dict(color="mediumseagreen")
                         ),

              row=1,col=1         
              )

fig.add_trace(go.Scatter(x=HOBBIES["day_int_"], 
                         y=HOBBIES["sell_price_mean"],
                         #showlegend=Ture,
                         mode="lines",
                         name="HOBBIES",
                         #marker=dict(color="yellow")
                         ),
             
              row=1,col=1           
              )

fig.add_trace(go.Scatter(x=HOUSEHOLD["day_int_"], 
                         y=HOUSEHOLD["sell_price_mean"],
                         #showlegend=Ture,
                         mode="lines",
                         name="HOUSEHOLD",
                         #marker=dict(color="blue")
                         ),
             
              row=1,col=1           
              )

fig.update_layout(height=500, title_text="Sell_price Mean Over Category by day-by-day")
fig.show()


# **Observation :** From the above plot we observe `HOBBIES` and `HOUSEHOLD` have high sell_prices

# In[26]:


fig = go.Figure()

fig.add_trace(go.Box(x=FOODS.cat_id_, y=FOODS.demand_mean, name="FOODS"))

fig.add_trace(go.Box(x=HOUSEHOLD.cat_id_, y=HOUSEHOLD.demand_mean, name="HOUSEHOLD"))

fig.add_trace(go.Box(x=HOBBIES.cat_id_, y=HOBBIES.demand_mean, name="HOBBIES"))


fig.update_layout(yaxis_title="Demand", xaxis_title="Time", title="Demand Mean vs. Category")


# In[27]:


fig = go.Figure()

fig.add_trace(go.Box(x=FOODS.cat_id_, y=FOODS.sell_price_mean, name="FOODS"))

fig.add_trace(go.Box(x=HOUSEHOLD.cat_id_, y=HOUSEHOLD.sell_price_mean, name="HOUSEHOLD"))

fig.add_trace(go.Box(x=HOBBIES.cat_id_, y=HOBBIES.sell_price_mean, name="HOBBIES"))


fig.update_layout(yaxis_title="Sell Price", xaxis_title="Time", title="Sell Price Mean vs. Category")


# ### For each Department <a id="4.6"></a>
# 
# - For each Category we have some Deportments
# 
# - There are total 7 Deportments

# In[28]:


## Number of items contain each Deportments

def n_unique(series):
    return series.nunique()

dep_count = new_CA_1.groupby("dept_id")["item_id"].agg(n_unique).reset_index()


# In[29]:


px.bar(dep_count, y="item_id", x="dept_id", color="dept_id", title="Deportment Item Count")


# **Observation :** 
# - From the above plot we observe that `FOODS` Category have 3Deportments, `HOBBIES` have 2Deportments, `HOUSEHOLD` have 2Deportments total 7Deportments
# - Deportment `FOODS_3` have more items

# In[30]:


## For each Deportment mean of deman and sell_price

dep = new_CA_1.groupby(["day_int","dept_id"])[["demand", "sell_price"]].agg(["mean", "max"]).reset_index()
dep.columns = ['_'.join(col).strip() for col in dep.columns.values]


# In[31]:


fig = make_subplots(rows=1, cols=1)

for each_dep in dep.dept_id_.unique():
    dep_df = dep[dep.dept_id_ == each_dep]
    fig.add_trace(go.Scatter(x=dep_df["day_int_"], 
                             y=dep_df["demand_mean"],
                             #showlegend=Ture,
                             mode="lines",
                             name=each_dep,
                             #marker=dict(color="mediumseagreen")
                             ),

                  row=1,col=1         
                  )
    
fig.update_layout(title_text="Demand Mean Over Deportments by day-by-day")
fig.show()


# **Observation :** From the above plot we observe that `FOOD_3` have more demand and it have some picks 

# In[32]:


fig = make_subplots(rows=1, cols=1)

for each_dep in dep.dept_id_.unique():
    dep_df = dep[dep.dept_id_ == each_dep]
    fig.add_trace(go.Scatter(x=dep_df["day_int_"], 
                             y=dep_df["sell_price_mean"],
                             #showlegend=Ture,
                             mode="lines",
                             name=each_dep,
                             #marker=dict(color="mediumseagreen")
                             ),

                  row=1,col=1         
                  )
    
fig.update_layout(title_text="Sell Prices Mean Over Deportments by day-by-day")
fig.show()


# **Observation :** From the above plot we observe `FOODS_1` and `HOUSEHOLD_2` have high sell_prices

# In[33]:


fig = go.Figure()

for each_dep in dep.dept_id_.unique():
    dep_df = dep[dep.dept_id_ == each_dep]

    fig.add_trace(go.Box(x=dep_df.dept_id_, y=dep_df.demand_mean, name=each_dep))
    


fig.update_layout(yaxis_title="Demand", xaxis_title="Time", title="Demand Mean vs. Deportment")


# **Observation :** From the above plots we can observe `FOOD3` have hight demand

# In[34]:


fig = go.Figure()

for each_dep in dep.dept_id_.unique():
    dep_df = dep[dep.dept_id_ == each_dep]

    fig.add_trace(go.Box(x=dep_df.dept_id_, y=dep_df.sell_price_mean, name=each_dep))
    
    
fig.update_layout(yaxis_title="Sell Price", xaxis_title="Time", title="Sell Price Mean vs. Deportment")


# **Observation :** From the above plots we can observe `HOBBIES` have hight sell Price

# ## Modeling

# ### Model Formation <a id="5.1"></a>
# 
# - ###### State -> Store -> Category -> Depoartment -> Item
# - ###### Total Stores -> 10
# - ###### For one Store -> 3049
# - ###### For one Item -> 1913 days
# - ###### For one Item -> Category -> Deportment
# 
# - ###### We need to predict 28d "Demand" for every Item in Every Store
# 
# ### Pytorch DataLoader
# 
# - ###### In data we have 10 Stores * 3049 Items => 30490 Store Items
# - ###### For each Item we need to create Features
# - ###### Because of its lots of data we need to create dataloders to load
# - ###### For that assume items are independet and for each item we have 1914 days
# - ###### For each item we need to create one npy file
# - ###### Each npy file contain 1913 days
# - ###### Each npy file contain 1913 days

# In[35]:


# take only one stor for demo

CA1 = new_CA_1
CA1.head()


# In[36]:


CA1 = CA1[["item_id","day_int", "demand", "sell_price", "date"]]
CA1.fillna(0, inplace=True)
print(CA1.shape)
CA1.head()


# ### Feature Engineering <a id="5.2"></a>
# 
# - For every item we need to extract features

# In[37]:


def date_features(df):
    
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df.date.dt.day
    df["month"] = df.date.dt.month
    df["week_day"] = df.date.dt.weekday

    df.drop(columns="date", inplace=True)

    return df

def sales_features(df):

    df.sell_price.fillna(0, inplace=True)

    return df

def demand_features(df):

    df["lag_t28"] = df["demand"].transform(lambda x: x.shift(28))
    df["rolling_mean_t7"] = df["demand"].transform(lambda x:x.shift(28).rolling(7).mean())
    df['rolling_mean_t30'] = df['demand'].transform(lambda x: x.shift(28).rolling(30).mean())
    df['rolling_mean_t60'] = df['demand'].transform(lambda x: x.shift(28).rolling(60).mean())
    df['rolling_mean_t90'] = df['demand'].transform(lambda x: x.shift(28).rolling(90).mean())
    df['rolling_mean_t180'] = df['demand'].transform(lambda x: x.shift(28).rolling(180).mean())
    df['rolling_std_t7'] = df['demand'].transform(lambda x: x.shift(28).rolling(7).std())
    df['rolling_std_t30'] = df['demand'].transform(lambda x: x.shift(28).rolling(30).std())

    df.fillna(0, inplace=True)

    return df


# In[38]:


get_ipython().system('mkdir "something_spl"')
# Saving each item with there item name.npy

for item in tqdm(CA1.item_id.unique()):
    one_item = CA1[CA1.item_id == item][["demand", "sell_price", "date"]]
    item_df = date_features(one_item)
    item_df = sales_features(item_df)
    item_df = demand_features(item_df)
    joblib.dump(item_df.values, f"something_spl/{item}.npy")


# ### Preparing Train & Validation df <a id="5.3"></a>

# In[39]:


# create dataframe for loading npy files and  train valid split

data_info = CA1[["item_id", "day_int"]]

# total number of days -> 1913
# for training we are taking data between 1800 < train <- 1913-28-28 = 1857

train_df = data_info[(1800 < data_info.day_int) &( data_info.day_int < 1857)]

# valid data is given last day -> 1885 we need to predict next 28days

valid_df = data_info[data_info.day_int == 1885]


# ### Pytorch Data Loader <a id="5.4"></a>
# 
# - we are taking last 28 days data features to predict next 28days of demand for each item
# - `train_window = 28` and `predicting_window=28`

# In[40]:


label = preprocessing.LabelEncoder()
label.fit(train_df.item_id)
label.transform(["FOODS_3_827"])


# In[41]:


class DataLoading:
    def __init__(self, df, train_window = 28, predicting_window=28):
        self.df = df.values
        self.train_window = train_window
        self.predicting_window = predicting_window

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, item):
        df_item = self.df[item]
        item_id = df_item[0]
        day_int = df_item[1]
        
        item_npy = joblib.load(f"something_spl/{item_id}.npy")
        item_npy_demand = item_npy[:,0]
        features = item_npy[day_int-self.train_window:day_int]
    

        predicted_demand = item_npy_demand[day_int:day_int+self.predicting_window]

        item_label = label.transform([item_id])
        item_onehot = [0] * 3049
        item_onehot[item_label[0]] = 1

        list_features = []
        for f in features:
            one_f = []
            one_f.extend(item_onehot)
            one_f.extend(f)
            list_features.append(one_f)

        return {
            "features" : torch.Tensor(list_features),
            "label" : torch.Tensor(predicted_demand)
        }


# In[42]:


## for exaple one item

datac = DataLoading(train_df)
n = datac.__getitem__(100)
n["features"].shape, n["label"].shape


# - we observe that at one item -> one day step we are taking last 28days of features to predict next 28days demand.
# - targets are 28days demand from that particular day

# ### LSTM+NN Model <a id="5.5"></a>

# In[43]:


class LSTM(nn.Module):
    def __init__(self, input_size=3062, hidden_layer_size=100, output_size=28):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))
        
    def forward(self, input_seq):

        lstm_out, self.hidden_cell = self.lstm(input_seq)

        lstm_out = lstm_out[:, -1]

        predictions = self.linear(lstm_out)

        return predictions


# ### Train and Eval functions<a id="5.6"></a>

# In[44]:


# loss function
def criterion1(pred1, targets):
    l1 = nn.MSELoss()(pred1, targets)
    return l1


# In[45]:


def train_model(model,train_loader, epoch, optimizer, scheduler=None, history=None):
    model.train()
    total_loss = 0
    
    t = tqdm(train_loader)
    
    for i, d in enumerate(t):
        
        item = d["features"].cuda().float()
        y_batch = d["label"].cuda().float()

        optimizer.zero_grad()

        out = model(item)
        loss = criterion1(out, y_batch)

        total_loss += loss
        
        t.set_description(f'Epoch {epoch+1} : , LR: %6f, Loss: %.4f'%(optimizer.state_dict()['param_groups'][0]['lr'],total_loss/(i+1)))

        if history is not None:
            history.loc[epoch + i / len(X), 'train_loss'] = loss.data.cpu().numpy()
            history.loc[epoch + i / len(X), 'lr'] = optimizer.state_dict()['param_groups'][0]['lr']

        loss.backward()
        optimizer.step()
        

def evaluate_model(model, val_loader, epoch, scheduler=None, history=None):
    model.eval()
    loss = 0
    pred_list = []
    real_list = []
    RMSE_list = []
    with torch.no_grad():
        for i,d in enumerate(tqdm(val_loader)):
            item = d["features"].cuda().float()
            y_batch = d["label"].cuda().float()

            o1 = model(item)
            l1 = criterion1(o1, y_batch)
            loss += l1
            
            o1 = o1.cpu().numpy()
            y_batch = y_batch.cpu().numpy()
            
            for pred, real in zip(o1, y_batch):
                rmse = np.sqrt(sklearn.metrics.mean_squared_error(real, pred))
                RMSE_list.append(rmse)
                pred_list.append(pred)
                real_list.append(real)

    loss /= len(val_loader)
    
    if scheduler is not None:
        scheduler.step(loss)

    print(f'\n Dev loss: %.4f RMSE : %.4f'%(loss, np.mean(RMSE_list)))
    


# ### Run Function <a id="5.7"></a>

# In[46]:


DEVICE = "cuda"
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 128
EPOCHS = 1
start_e = 1


model = LSTM()
model.to(DEVICE)

train_dataset = DataLoading(train_df)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size= TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    drop_last=True
)


valid_dataset = DataLoading(valid_df)

valid_loader = torch.utils.data.DataLoader(
    dataset=valid_dataset,
    batch_size= TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    drop_last=True
)


optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, mode='min', factor=0.7, verbose=True, min_lr=1e-5)

for epoch in range(start_e, EPOCHS+1):
    train_model(model, train_loader, epoch, optimizer, scheduler=scheduler, history=None)
    evaluate_model(model, valid_loader, epoch, scheduler=scheduler, history=None)


# ## stay tuned compitation metric and custom loss parts coming soon :)
# 
# <h2 style="color:red;"> Please upvote if you like it. It motivates me. Thank you ☺️ .</h2>
