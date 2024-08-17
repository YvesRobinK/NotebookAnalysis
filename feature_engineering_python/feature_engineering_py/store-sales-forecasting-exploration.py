#!/usr/bin/env python
# coding: utf-8

# ## Store Sales Forecasting -  Exploration Notebook
# 
# In this competition, the task is to predict the sales for the thousands of product families sold at Favorita stores located in Ecuador. The training data includes dates, store and product information, whether that item was being promoted, as well as the sales numbers. In this notebook, I have shared a started exploration / eda for the dataset. I will keep updating
# 
# ### 1. Load Dataset 
# 
# Let's first load the dataset files

# In[1]:


import numpy as np 
import pandas as pd 
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

train = pd.read_csv("/kaggle/input/store-sales-time-series-forecasting/train.csv")
test = pd.read_csv("/kaggle/input/store-sales-time-series-forecasting/test.csv")
oil_df = pd.read_csv("/kaggle/input/store-sales-time-series-forecasting/oil.csv")
holidays_events = pd.read_csv("/kaggle/input/store-sales-time-series-forecasting/holidays_events.csv")
stores = pd.read_csv("/kaggle/input/store-sales-time-series-forecasting/stores.csv")
txn = pd.read_csv("/kaggle/input/store-sales-time-series-forecasting/transactions.csv")

print ("Training Data Shape: ", train.shape)
print ("Testing Data Shape", test.shape)

train.head()


# ### 2. Combine Dataset files
# 
# A number of supplement files are provided which contain addition features, we can combine them to our original training and test sets. 

# In[2]:


## combine datasets
train1 = train.merge(oil_df, on = 'date', how='left')
train1 = train1.merge(holidays_events, on = 'date', how='left')
train1 = train1.merge(stores, on = 'store_nbr', how='left')
train1 = train1.merge(txn, on = ['date', 'store_nbr'], how='left')
train1 = train1.rename(columns = {"type_x" : "holiday_type", "type_y" : "store_type"})

test1 = test.merge(oil_df, on = 'date', how='left')
test1 = test1.merge(holidays_events, on = 'date', how='left')
test1 = test1.merge(stores, on = 'store_nbr', how='left')
test1 = test1.merge(txn, on = ['date', 'store_nbr'], how='left')
test1 = test1.rename(columns = {"type_x" : "holiday_type", "type_y" : "store_type"})

train1.head()


# ### 3. Exploratory Analysis 
# 
# Let's take a look at the time series patterns in the dataset such as Average Sales over time, average sales over time by store type, by store name etc.  

# In[3]:


agg = train1.groupby('date').agg({"sales" : "mean"}).reset_index()
fig = px.line(agg, x='date', y="sales")
fig.update_layout(title = "Average Sales by Date")
fig.show()

agg = train1.groupby('date').agg({"transactions" : "mean"}).reset_index()
fig = px.line(agg, x='date', y="transactions")
fig.update_layout(title = "Average Transactions by Date")
fig.show()


# In[4]:


agg = train1.groupby(['date', 'store_type']).agg({"sales" : "mean"}).reset_index()
fig = px.line(agg, x='date', y="sales", color='store_type')
fig.update_layout(title = "Average Sales by Date and Store Type")
fig.show()

agg = train1.groupby(['date', 'store_type']).agg({"transactions" : "mean"}).reset_index()
fig = px.line(agg, x='date', y="transactions", color='store_type')
fig.update_layout(title = "Average Transactions by Date and Store Type")
fig.show()


# In[5]:


agg = train1.groupby(['date', 'cluster']).agg({"sales" : "mean"}).reset_index()
fig = px.line(agg, x='date', y="sales", color='cluster')
fig.update_layout(title = "Average Sales by Date and Store Number")
fig.show()


agg = train1.groupby(['date', 'cluster']).agg({"transactions" : "mean"}).reset_index()
fig = px.line(agg, x='date', y="transactions", color='cluster')
fig.update_layout(title = "Average Transactions by Date and Cluster")
fig.show()


# Let's now look at various other columns and their related average sales 

# In[6]:


def vbar(col):
    temp = train1.groupby(col).agg({"sales" : "mean"}).reset_index()
    temp = temp.sort_values('sales', ascending = False)
    c = {
        'x' : list(temp['sales'])[:15][::-1], 
        'y' : list(temp[col])[:15][::-1],
        'title' : "Average sales by "+col
    }
    trace = go.Bar(y=[str(_) + "    " for _ in c['y']], x=c['x'], orientation="h", marker=dict(color="#f77e90"))
    return trace 

    layout = go.Layout(title=c['title'], 
                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           xaxis_title="", yaxis_title="", width=650)
    fig = go.Figure([trace], layout=layout)
    fig.update_xaxes(tickangle=45, tickfont=dict(color='crimson'))
    fig.update_yaxes(tickangle=0, tickfont=dict(color='crimson'))
    fig.show()
    
trace1 = vbar('family') 
trace2 = vbar('store_type') 
trace3 = vbar('state') 
trace4 = vbar('city') 

titles = ['Store Family', 'Store Type', 'State', 'City']
titles = ['Top ' + _ + " by Average Sales" for _ in titles]
fig = make_subplots(rows=2, cols=2, subplot_titles = titles)

fig.add_trace(trace1, row=1, col=1)
fig.add_trace(trace2, row=1, col=2)
fig.add_trace(trace3, row=2, col=1)
fig.add_trace(trace4, row=2, col=2)

fig.update_layout(height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend = False)
fig.show()


# In[7]:


trace1 = vbar('cluster') 
trace2 = vbar('store_nbr') 

titles = ['Cluster Number', 'Store Number']
titles = ['Top ' + _ + " by Average Sales" for _ in titles]
fig = make_subplots(rows=1, cols=2, subplot_titles = titles)

fig.add_trace(trace1, row=1, col=1)
fig.add_trace(trace2, row=1, col=2)

fig.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend = False)
fig.show()


# ### 4. Feature Engineering 
# 
# We can create some additional features from the date column such as dayofweek, month, year etc. 

# In[8]:


def create_ts_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    return df
    
train1 = create_ts_features(train1)
test1 = create_ts_features(test1)
train1.head()


# In[9]:


def hbar(col):
    temp = train1.groupby(col).agg({"sales" : "mean"}).reset_index()
    temp = temp.sort_values(col, ascending = False)
    c = {
        'y' : list(temp['sales']), 
        'x' : list(temp[col]),
        'title' : "Average sales by "+col
    }
    trace = go.Bar(y=c['y'], x=c['x'], orientation="v", marker=dict(color="#bbe070"))
    return trace 

    layout = go.Layout(title=c['title'], 
                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           xaxis_title="", yaxis_title="", width=650)
    fig = go.Figure([trace], layout=layout)
    fig.update_xaxes(tickangle=45, tickfont=dict(color='crimson'))
    fig.update_yaxes(tickangle=0, tickfont=dict(color='crimson'))
    fig.show()
    
trace1 = hbar('dayofweek') 
trace2 = hbar('dayofmonth') 
trace3 = hbar('dayofyear') 
trace4 = hbar('month') 
trace5 = hbar('quarter') 
trace6 = hbar('year') 

titles = ['Day of Week', 'Day of Month', 'Day of Year', 'Month', 'Quarter', 'Year']
titles = ['Avg Sales by ' + _ for _ in titles]
fig = make_subplots(rows=3, cols=2, subplot_titles = titles)

fig.add_trace(trace1, row=1, col=1)
fig.add_trace(trace2, row=1, col=2)
fig.add_trace(trace3, row=2, col=1)
fig.add_trace(trace4, row=2, col=2)
fig.add_trace(trace5, row=3, col=1)
fig.add_trace(trace6, row=3, col=2)

fig.update_layout(height=1200, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend = False)
fig.show()


# In[10]:


agg = train1.groupby(["year", "month"]).agg({"sales"  :"mean", "transactions" : "mean"}).reset_index()
fig = px.box(agg, y="sales", facet_col="month", color="month",
             boxmode="overlay", points='all')
fig.update_layout(title = "Average Sales Distribution by Store Type")
fig.show()


# In[11]:


agg = train1.groupby(["year", "store_type"]).agg({"sales"  :"mean", "transactions" : "mean"}).reset_index()
fig = px.box(agg, y="sales", facet_col="store_type", color="store_type",
             boxmode="overlay", points='all')
fig.update_layout(title = "Average Sales Distribution by Store Type")
fig.show()


# In[12]:


train1['holiday_type'] = train1['holiday_type'].fillna("No Holiday/Event")
train1['holiday_type'].value_counts()

def convert_to_size(x):
    if x < 50:
        return 6
    elif x < 100:
        return 10
    elif x < 150:
        return 15
    elif x < 250:
        return 18 
    elif x < 300:
        return 24 
    elif x < 500:
        return 30 
    else:
        return 40

def bubble(col1, col2):
    vc = train1.groupby([col1, col2]).agg({"sales" : "mean"}).reset_index()
    vc = vc.sort_values(col2)    
    fig = px.scatter(vc, x=col1, y=col2, 
                     size='sales', color='sales', size_max=40)
    fig.update_layout(title = "Average Sales by "+col1+" and " + col2)
    fig.show()
    
bubble('month', 'holiday_type')
bubble('month', 'store_type')


# In[13]:


train1.to_csv("train_complete.csv", index = False)
test1.to_csv("test_complete.csv", index = False)

