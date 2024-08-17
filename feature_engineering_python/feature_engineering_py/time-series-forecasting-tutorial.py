#!/usr/bin/env python
# coding: utf-8

# # <b>1 <span style='color:#F1C40F'>|</span> Introduction to Date and Time</b>
# 
# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>1.1 | How to import data ?</b></p>
# </div>
# 
# First, we import all the datasets needed for this kernel. The required time series column is imported as a datetime column using **<span style='color:#F1C40F'>parse_dates</span>** parameter and is also selected as index of the dataframe using **<span style='color:#F1C40F'>index_col</span>** parameter.
# 
# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>1.2 | Timestamps and Periods</b></p>
# </div>
# 
# Timestamps are used to represent a point in time. Periods represent an interval in time. Periods can used to check if a specific event in the given period. They can also be converted to each other's form.
# 
# 📌 Video: [How to use dates and times with pandas](https://campus.datacamp.com/courses/manipulating-time-series-data-in-python/working-with-time-series-in-pandas?ex=1): explain **<span style='color:#F1C40F'>TimeStamp</span>** and **<span style='color:#F1C40F'>Period</span>** data. 
# 
# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>1.3 | Using date_range</b></p>
# </div>
# 
# date_range is a method that returns a fixed **<span style='color:#F1C40F'>frequency datetimeindex</span>**. It is quite useful when creating your own time series attribute for pre-existing data or arranging the whole data around the time series attribute created by you.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import datetime
from learntools.time_series.style import *

from pathlib import Path
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.offline as offline
import plotly.graph_objs as go

comp_dir = Path('../input/store-sales-time-series-forecasting')
train = pd.read_csv(comp_dir / 'train.csv')
test = pd.read_csv(comp_dir / 'test.csv')
stores = pd.read_csv(comp_dir / 'stores.csv')
oil = pd.read_csv(comp_dir / 'oil.csv')
transactions =  pd.read_csv(comp_dir / 'transactions.csv')
holidays_events = pd.read_csv(comp_dir / 'holidays_events.csv')

def seasonal_plot(X, y, period, freq, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
    return ax

def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax

def lagplot(x, y=None, lag=1, standardize=False, ax=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    x_ = x.shift(lag)
    if standardize:
        x_ = (x_ - x_.mean()) / x_.std()
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y
    else:
        y_ = x
    corr = y_.corr(x_)
    if ax is None:
        fig, ax = plt.subplots()
    scatter_kws = dict(
        alpha=0.75,
        s=3,
    )
    line_kws = dict(color='C3', )
    ax = sns.regplot(x=x_,
                     y=y_,
                     scatter_kws=scatter_kws,
                     line_kws=line_kws,
                     lowess=True,
                     ax=ax,
                     **kwargs)
    at = AnchoredText(
        f"{corr:.2f}",
        prop=dict(size="large"),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)
    return ax

def plot_lags(x, y=None, lags=6, nrows=1, lagplot_kwargs={}, **kwargs):
    import math
    kwargs.setdefault('nrows', nrows)
    kwargs.setdefault('ncols', math.ceil(lags / nrows))
    kwargs.setdefault('figsize', (kwargs['ncols'] * 2, nrows * 2 + 0.5))
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
    for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):
        if k + 1 <= lags:
            ax = lagplot(x, y, lag=k + 1, ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lag {k + 1}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis('off')
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    return fig


# We will break down the date into different columns: 
# * One for the year
# * One for the month
# * One for the week
# * One for the quarter of the year
# * One for the day of the week

# In[2]:


df_data = pd.concat([train, test], sort=True)
df_data = df_data.merge(stores, how="left", on='store_nbr')   
df_data = df_data.merge(oil, how="left", on='date')      
df_data = df_data.merge(transactions, how="left", on=['date','store_nbr'])  
df_data = df_data.merge(holidays_events,on='date',how='left')
df_data = df_data.rename(columns={'type_x' : 'store_type','type_y':'holiday_type'})

df_data.date = pd.to_datetime(df_data.date)
df_data['year'] = df_data['date'].dt.year
df_data['month'] = df_data['date'].dt.month
df_data['week'] = df_data['date'].dt.isocalendar().week
df_data['quarter'] = df_data['date'].dt.quarter
df_data['day_of_week'] = df_data['date'].dt.day_name()
df_data.head()


# # <b>2 <span style='color:#F1C40F'>|</span> Missing Values</b>
# 
# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.1 | Oil Price</b></p>
# </div>
# 
# Let's start with oil missing values. Firstly, we are going to plot oil price during the years together with trending graph.

# In[3]:


moving_average_oil = oil.rolling(
    window=365,       # 365-day window
    center=True,      # puts the average at the center of the window
    min_periods=183,  # choose about half the window size
).median()              # compute the mean (could also do median, std, min, max, ...)
moving_average_oil['date'] = oil['date']
moving_average_oil.loc[[0,1],'dcoilwtico'] = moving_average_oil.loc[2,'dcoilwtico']
moving_average_oil.date = pd.to_datetime(moving_average_oil.date)

df_yr_oil = oil[['date','dcoilwtico']]
fig = make_subplots(rows=1, cols=1, vertical_spacing=0.08,                    
                    subplot_titles=("Oil price during time"))
fig.add_trace(go.Scatter(x=df_yr_oil['date'], y=df_yr_oil['dcoilwtico'], mode='lines', fill='tozeroy', fillcolor='#c6ccd8',
                     marker=dict(color= '#496595'), name='Oil price'), 
                     row=1, col=1)
fig.add_trace(go.Scatter(x=moving_average_oil.date,y=moving_average_oil.dcoilwtico,mode='lines',name='Trend'))
fig.update_layout(height=350, bargap=0.15,
                  margin=dict(b=0,r=20,l=20), 
                  title_text="Oil price trend during time",
                  template="plotly_white",
                  title_font=dict(size=25, color='#8a8d93', family="Lato, sans-serif"),
                  font=dict(color='#8a8d93'),
                  hoverlabel=dict(bgcolor="#f2f2f2", font_size=13, font_family="Lato, sans-serif"),
                  showlegend=False)
fig.show()


# 📌 **Interpret:** As can be seen in the graph above, we can divide the oil price trend into **<span style='color:#F1C40F'>three phases</span>**. The first and last of these, Jan2013-Jul2014 and Jan2015-Jul2107 respectively, show stabilised trends with ups and downs. However, in the second phase, Jul2014-Jan2015, oil prices decrease considerably.
# 
# Now, taking into account the issue of missing values for oil price, we are going to fill them by **<span style='color:#F1C40F'>backward fill technique</span>**. That means filling missing values with next data point (Forward filling means fill missing values with previous data).

# In[4]:


df_data['dcoilwtico'] = df_data['dcoilwtico'].fillna(method='bfill')
df_data.dcoilwtico.isnull().sum()


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.2 | Transactions</b></p>
# </div>
# 
# With respect to transactions, we understand that since there is no data recorded, this is 0.

# In[5]:


df_data.transactions = df_data.transactions.replace(np.nan,0)


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.3 | Holidays</b></p>
# </div>
# 
# As we can see above the `holidays_events` DataFrame contains a row for each of the national, regional or local holidays. The transferred column refers to whether the holiday has been moved or not. We assume then that the missing data corresponding to this DataFrame in the training set correspond to those days for which no public holiday has been recorded. Therefore, we will replace the `type` by **<span style='color:#F1C40F'>Working day</span>**. The rest of the categorical variables in this DataFrame will be changed to the empty string, and in `transferred` we will set all values to `false`.

# In[6]:


df_data[['locale','locale_name', 'description']] = df_data[['locale','locale_name', 'description']].replace(np.nan,'')
df_data['holiday_type'] = df_data['holiday_type'].replace(np.nan,'Working Day')
df_data['transferred'] = df_data['transferred'].replace(np.nan,False)


# # <b>3 <span style='color:#F1C40F'>|</span> Data Visualization</b>
# 
# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>3.1 | Average Sales Analysis</b></p>
# </div>
# 
# In this section we are going to carry out various studies of the data obtained above, using various graphs. We will focus on seeing: 
# * The shops with the **<span style='color:#F1C40F'>highest percentage of sales</span>**
# * The **<span style='color:#F1C40F'>types of products most sold</span>**. 
# * The sales of each **<span style='color:#F1C40F'>cluster</span>**.
# * The **<span style='color:#F1C40F'>sales history</span>** for each of the months of the year. 
# * The percentages of sales per **<span style='color:#F1C40F'>quarter</span>** of the year.
# * **<span style='color:#F1C40F'>Average sales per week</span>**.

# In[7]:


# data
# Agrupamos por tipo de tienda, y al DataFrame le añadimos un único campo 'sales' con la media de los precios de venta ordenados ascendentemente
df_st_sa = df_data[:train.shape[0]].groupby('store_type').agg({"sales" : "mean"}).reset_index().sort_values(by='sales', ascending=False)
df_fa_sa = df_data[:train.shape[0]].groupby('family').agg({"sales" : "mean"}).reset_index().sort_values(by='sales', ascending=False)[:10]
df_cl_sa = df_data[:train.shape[0]].groupby('cluster').agg({"sales" : "mean"}).reset_index() 
# chart color
df_fa_sa['color'] = '#496595'
df_fa_sa['color'][2:] = '#c6ccd8'
df_cl_sa['color'] = '#c6ccd8'

# chart
fig = make_subplots(rows=2, cols=2, 
                    specs=[[{"type": "bar"}, {"type": "pie"}],
                           [{"colspan": 2}, None]],
                    column_widths=[0.7, 0.3], vertical_spacing=0, horizontal_spacing=0.02,
                    subplot_titles=("Top 10 Highest Product Sales", "Highest Sales in Stores", "Clusters Vs Sales"))

fig.add_trace(go.Bar(x=df_fa_sa['sales'], y=df_fa_sa['family'], marker=dict(color= df_fa_sa['color']),
                     name='Family', orientation='h'), 
                     row=1, col=1)
fig.add_trace(go.Pie(values=df_st_sa['sales'], labels=df_st_sa['store_type'], name='Store type',
                     marker=dict(colors=['#334668','#496595','#6D83AA','#91A2BF','#C8D0DF']), hole=0.7,
                     hoverinfo='label+percent+value', textinfo='label'), 
                    row=1, col=2)
fig.add_trace(go.Bar(x=df_cl_sa['cluster'], y=df_cl_sa['sales'], 
                     marker=dict(color= df_cl_sa['color']), name='Cluster'), 
                     row=2, col=1)

# styling
fig.update_yaxes(showgrid=False, ticksuffix=' ', categoryorder='total ascending', row=1, col=1)
fig.update_xaxes(visible=False, row=1, col=1)
fig.update_xaxes(tickmode = 'array', tickvals=df_cl_sa.cluster, ticktext=[i for i in range(1,17)], row=2, col=1)
fig.update_yaxes(visible=False, row=2, col=1)
fig.update_layout(height=500, bargap=0.2,
                  margin=dict(b=0,r=20,l=20), xaxis=dict(tickmode='linear'),
                  title_text="Average Sales Analysis",
                  template="plotly_white",
                  title_font=dict(size=29, color='#8a8d93', family="Lato, sans-serif"),
                  font=dict(color='#8a8d93'), 
                  hoverlabel=dict(bgcolor="#f2f2f2", font_size=13, font_family="Lato, sans-serif"),
                  showlegend=False)
fig.show()


# 📌 **Interpret:** Highest sales are made by the products like **<span style='color:#F1C40F'>grocery and beverages</span>**.
# Store A has the highest sales which is 38%.

# In[8]:


# data 
df_2013 = df_data[df_data['year']==2013][:train.shape[0]][['month','sales']]
df_2013 = df_2013.groupby('month').agg({"sales" : "mean"}).reset_index().rename(columns={'sales':'s13'})
df_2014 = df_data[df_data['year']==2014][:train.shape[0]][['month','sales']]
df_2014 = df_2014.groupby('month').agg({"sales" : "mean"}).reset_index().rename(columns={'sales':'s14'})
df_2015 = df_data[df_data['year']==2015][:train.shape[0]][['month','sales']]
df_2015 = df_2015.groupby('month').agg({"sales" : "mean"}).reset_index().rename(columns={'sales':'s15'})
df_2016 = df_data[df_data['year']==2016][:train.shape[0]][['month','sales']]
df_2016 = df_2016.groupby('month').agg({"sales" : "mean"}).reset_index().rename(columns={'sales':'s16'})
df_2017 = df_data[df_data['year']==2017][:train.shape[0]][['month','sales']]
df_2017 = df_2017.groupby('month').agg({"sales" : "mean"}).reset_index()
df_2017_no = pd.DataFrame({'month': [9,10,11,12], 'sales':[0,0,0,0]})
df_2017 = df_2017.append(df_2017_no).rename(columns={'sales':'s17'})
df_year = df_2013.merge(df_2014,on='month').merge(df_2015,on='month').merge(df_2016,on='month').merge(df_2017,on='month')

# top levels
top_labels = ['2013', '2014', '2015', '2016', '2017']

colors = ['rgba(38, 24, 74, 0.8)', 'rgba(71, 58, 131, 0.8)',
          'rgba(122, 120, 168, 0.8)', 'rgba(164, 163, 204, 0.85)',
          'rgba(190, 192, 213, 1)']

# X axis value 
df_year = df_year[['s13','s14','s15','s16','s17']].replace(np.nan,0)
x_data = df_year.values

# y axis value (Month)
df_2013['month'] =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
y_data = df_2013['month'].tolist()

fig = go.Figure()
for i in range(0, len(x_data[0])):
    for xd, yd in zip(x_data, y_data):
        fig.add_trace(go.Bar(
            x=[xd[i]], y=[yd],
            orientation='h',
            marker=dict(
                color=colors[i],
                line=dict(color='rgb(248, 248, 249)', width=1)
            )
        ))

fig.update_layout(title='Avg Sales for each Year',
    xaxis=dict(showgrid=False, 
               zeroline=False, domain=[0.15, 1]),
    yaxis=dict(showgrid=False, showline=False,
               showticklabels=False, zeroline=False),
    barmode='stack', 
    template="plotly_white",
    margin=dict(l=0, r=50, t=100, b=10),
    showlegend=False, 
)

annotations = []
for yd, xd in zip(y_data, x_data):
    # labeling the y-axis
    annotations.append(dict(xref='paper', yref='y',
                            x=0.14, y=yd,
                            xanchor='right',
                            text=str(yd),
                            font=dict(family='Arial', size=14,
                                      color='rgb(67, 67, 67)'),
                            showarrow=False, align='right'))
    # labeling the first Likert scale (on the top)
    if yd == y_data[-1]:
        annotations.append(dict(xref='x', yref='paper',
                                x=xd[0] / 2, y=1.1,
                                text=top_labels[0],
                                font=dict(family='Arial', size=14,
                                          color='rgb(67, 67, 67)'),
                          showarrow=False))
    space = xd[0]
    for i in range(1, len(xd)):
            # labeling the Likert scale
            if yd == y_data[-1]:
                annotations.append(dict(xref='x', yref='paper',
                                        x=space + (xd[i]/2), y=1.1,
                                        text=top_labels[i],
                                        font=dict(family='Arial', size=14,
                                                  color='rgb(67, 67, 67)'),
                                        showarrow=False))
            space += xd[i]
fig.update_layout(
    annotations=annotations)
fig.show()


# 📌 **Interpret:** Highest sales are made in **<span style='color:#F1C40F'>December</span>** month and then decreases in January. Sales are **<span style='color:#F1C40F'>increasing gradually</span>** from 2013 to 2017. Note: We don't have data for 2017: 9th to 12th month.

# In[9]:


# data
df_m_sa = df_data[:train.shape[0]].groupby('month').agg({"sales" : "mean"}).reset_index()
df_m_sa['sales'] = round(df_m_sa['sales'],2)
df_m_sa['month_text'] = df_m_sa['month'].apply(lambda x: calendar.month_abbr[x])
df_m_sa['text'] = df_m_sa['month_text'] + ' - ' + df_m_sa['sales'].astype(str) 

df_w_sa = df_data[:train.shape[0]].groupby('week').agg({"sales" : "mean"}).reset_index() 
df_q_sa = df_data[:train.shape[0]].groupby('quarter').agg({"sales" : "mean"}).reset_index() 
# chart color
df_m_sa['color'] = '#496595'
df_m_sa['color'][:-1] = '#c6ccd8'
df_w_sa['color'] = '#c6ccd8'

# chart
fig = make_subplots(rows=2, cols=2, vertical_spacing=0.08,
                    row_heights=[0.7, 0.3], 
                    specs=[[{"type": "bar"}, {"type": "pie"}],
                           [{"colspan": 2}, None]],
                    column_widths=[0.7, 0.3],
                    subplot_titles=("Month wise Avg Sales Analysis", "Quarter wise Avg Sales Analysis", 
                                    "Week wise Avg Sales Analysis"))

fig.add_trace(go.Bar(x=df_m_sa['sales'], y=df_m_sa['month'], marker=dict(color= df_m_sa['color']),
                     text=df_m_sa['text'],textposition='auto',
                     name='Month', orientation='h'), 
                     row=1, col=1)
fig.add_trace(go.Pie(values=df_q_sa['sales'], labels=df_q_sa['quarter'], name='Quarter',
                     marker=dict(colors=['#334668','#496595','#6D83AA','#91A2BF','#C8D0DF']), hole=0.7,
                     hoverinfo='label+percent+value', textinfo='label+percent'), 
                     row=1, col=2)
fig.add_trace(go.Scatter(x=df_w_sa['week'], y=df_w_sa['sales'], mode='lines+markers', fill='tozeroy', fillcolor='#c6ccd8',
                     marker=dict(color= '#496595'), name='Week'), 
                     row=2, col=1)

# styling
fig.update_yaxes(visible=False, row=1, col=1)
fig.update_xaxes(visible=False, row=1, col=1)
fig.update_xaxes(tickmode = 'array', tickvals=df_w_sa.week, ticktext=[i for i in range(1,53)], 
                 row=2, col=1)
fig.update_yaxes(visible=False, row=2, col=1)
fig.update_layout(height=750, bargap=0.15,
                  margin=dict(b=0,r=20,l=20), 
                  title_text="Average Sales Analysis",
                  template="plotly_white",
                  title_font=dict(size=25, color='#8a8d93', family="Lato, sans-serif"),
                  font=dict(color='#8a8d93'),
                  hoverlabel=dict(bgcolor="#f2f2f2", font_size=13, font_family="Lato, sans-serif"),
                  showlegend=False)
fig.show()


# 📌 **Interpret:** Highest sales are made in the **<span style='color:#F1C40F'>last quarter</span>** of the year, followed by the third. The one with less saling is the first one.

# In[10]:


# data
df_dw_sa = df_data[:train.shape[0]].groupby('day_of_week').agg({"sales" : "mean"}).reset_index()
df_dw_sa.sales = round(df_dw_sa.sales, 2)

# chart
fig = px.bar(df_dw_sa, y='day_of_week', x='sales', title='Avg Sales vs Day of Week',
             color_discrete_sequence=['#c6ccd8'], text='sales',
             category_orders=dict(day_of_week=["Monday","Tuesday","Wednesday","Thursday", "Friday","Saturday","Sunday"]))
fig.update_yaxes(showgrid=False, ticksuffix=' ', showline=False)
fig.update_xaxes(visible=False)
fig.update_layout(margin=dict(t=60, b=0, l=0, r=0), height=350,
                  hovermode="y unified", 
                  yaxis_title=" ", template='plotly_white',
                  title_font=dict(size=25, color='#8a8d93', family="Lato, sans-serif"),
                  font=dict(color='#8a8d93'),
                  hoverlabel=dict(bgcolor="#c6ccd8", font_size=13, font_family="Lato, sans-serif"))


# 📌 **Interpret:** Highest sales are made in the **<span style='color:#F1C40F'>weekend</span>**. Surprisingly, Mondays are the third day with most sales.

# In[11]:


df_train = df_data[:train.shape[0]][['state','sales','store_type','year']]
fig = plt.figure(figsize=(22,8))
sns.set_style('whitegrid')
my_palette = ['#C8D0DF','#91A2BF','#6D83AA','#496595','#334668']
sns.barplot(x='state',y='sales',hue = 'year', palette = my_palette, data=df_train[df_train['store_type'] == 'A'])
plt.title("State vs Sales of Store A (per year)")


# 📌 **Interpret:** Highest sales are made in **<span style='color:#F1C40F'>Pichincha</span>** state. Sales have been increasing during the recorded period in every state, except Manabi where Store A is new in 2017.
# # <b>4 <span style='color:#F1C40F'>|</span> Time Series Components</b>
# 
# If we assume an **<span style='color:#F1C40F'>additive decomposition</span>**, then we can write $𝑦_𝑡=𝑆_𝑡+𝑇_𝑡+𝑅_𝑡$, where $𝑦_𝑡$ is the data, $S_t$ is the seasonal component, $𝑇_𝑡$ is the trend-cycle component and $𝑅_𝑡$ is the residual component, all at period 𝑡. Also,for a **<span style='color:#F1C40F'>multiplicative decomposition</span>**, we have
# $𝑦_𝑡=𝑆_𝑡∗𝑇_𝑡∗𝑅_𝑡$.
# 
# The additive decomposition is the most appropriate if the magnitude of the seasonal fluctuations, or the variation around the trend-cycle, does not vary with the level of the time series. When the variation in the seasonal pattern, or the variation around the trend-cycle, appears to be proportional to the level of the time series, then a multiplicative decomposition is more appropriate. Multiplicative decompositions are common with economic time series.
# 
# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>4.1 | Trend</b></p>
# </div>
# 
# ### **What is Trend ?**
# 
# The trend component of a time series represents a **<span style='color:#F1C40F'>persistent, long-term change in the mean of the series</span>**. The trend is the slowest-moving part of a series, the part representing the largest time scale of importance. In a time series of product sales, an increasing trend might be the effect of a market expansion as more people become aware of the product year by year.
# 
# ### **Moving Average Plot**
# To see what kind of trend a time series might have, we can use a moving average plot. To compute a moving average of a time series, we compute the average of the values within a **<span style='color:#F1C40F'>sliding window</span>** of some defined width. Each point on the graph represents the average of all the values in the series that fall within the window on either side. The idea is to **<span style='color:#F1C40F'>smooth out</span>** any short-term **<span style='color:#F1C40F'>fluctuations</span>** in the series so that only long-term changes remain.

# In[12]:


sales = df_data[:train.shape[0]].groupby('date').agg({"sales" : "mean"}).reset_index()
sales.set_index('date',inplace=True)
moving_average = sales.rolling(
    window=365,       # 365-day window
    center=True,      # puts the average at the center of the window
    min_periods=183,  # choose about half the window size
).mean()              # compute the mean (could also do median, std, min, max, ...)
moving_average['date'] = sales.index

fig = make_subplots(rows=1, cols=1, vertical_spacing=0.08,                    
                    subplot_titles=("Sales 365 - Day Moving Average"))
fig.add_trace(go.Scatter(x=sales.index, y=sales['sales'], mode='lines', fill='tozeroy', fillcolor='#c6ccd8',
                     marker=dict(color= '#334668'), name='365-Day Moving Average'))
fig.add_trace(go.Scatter(x=moving_average.date,y=moving_average.sales,mode='lines',name='Trend'))
fig.update_layout(height=350, bargap=0.15,
                  margin=dict(b=0,r=20,l=20), 
                  title_text="Sales trend during years",
                  template="plotly_white",
                  title_font=dict(size=25, color='#8a8d93', family="Lato, sans-serif"),
                  font=dict(color='#8a8d93'),
                  hoverlabel=dict(bgcolor="#f2f2f2", font_size=13, font_family="Lato, sans-serif"),
                  showlegend=False)
fig.show()


# 📌 **Interpret:** As we can appreeciate, sales has an constantly increasing trend during recorded years. 
# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>4.2 | Seasonality </b></p>
# </div>
# 
# We say that a time series exhibits seasonality whenever there is a **<span style='color:#F1C40F'>regular, periodic change</span>** in the mean of the series. Seasonal changes generally follow the clock and calendar - repetitions over a day, a week, or a year are common. Seasonality is often driven by the cycles of the natural world over days and years or by conventions of social behavior surrounding dates and times.

# In[13]:


store_sales = pd.read_csv(
    comp_dir / 'train.csv',
    usecols=['store_nbr', 'family', 'date', 'sales'],
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
store_sales['date'] = store_sales.date.dt.to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()
average_sales = (
    store_sales
    .groupby('date').mean()
    .squeeze()
    #.loc['2017']
)

X = average_sales.to_frame()
X["week"] = X.index.week
X["day"] = X.index.dayofweek
X['year'] = X.index.year
X['dayofyear'] = X.index.dayofyear
fig,(ax0,ax1) = plt.subplots(1, 2, figsize=(22, 10))
seasonal_plot(X.loc['2017'], y='sales', period="week", freq="day", ax=ax0)
ax0.set_title('Seasonal Plot (week/day) 2017')
seasonal_plot(X, y="sales", period="year", freq="dayofyear", ax=ax1);
ax1.set_title('Seasonal Plot (year/dayofyear)')


# In[14]:


plot_periodogram(average_sales.loc['2017']);


# 📌 **Interpret:** both the seasonal plot and the periodogram suggest a **<span style='color:#F1C40F'>strong weekly seasonality</span>**, and a weak annual seasonality. From the periodogram, it appears there may be some **<span style='color:#F1C40F'>monthly</span>** and **<span style='color:#F1C40F'>biweekly</span>** components as well. In fact, the notes to the Store Sales dataset say wages in the public sector are paid out biweekly, on the 15th and last day of the month -- a possible origin for these seasons.
# 
# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>4.3 | Decomposition </b></p>
# </div>
# 
# Let's now combine all above time series features in aan **<span style='color:#F1C40F'>unique</span>** graph. 

# In[15]:


from statsmodels.tsa.seasonal import seasonal_decompose
decomp = seasonal_decompose(sales['sales'], period=365, model='additive', extrapolate_trend='freq')
fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True, figsize=(22,10))
ax[0,0].set_title('Observed values for Sales', fontsize=16)
decomp.observed.plot(ax = ax[0,0], legend=False, color='dodgerblue')

ax[0,1].set_title('Sales Trend', fontsize=16)
decomp.trend.plot(ax = ax[0,1],legend=False, color='dodgerblue')

ax[1,0].set_title('Sales Seasonality', fontsize=16)
decomp.seasonal.plot(ax = ax[1,0],legend=False, color='dodgerblue')

ax[1,1].set_title('Noise', fontsize=16)
decomp.resid.plot(ax = ax[1,1],legend=False, color='dodgerblue')


# 📌 **Interpret:** The three components are shown separately in the bottom three panels. These components can be **<span style='color:#F1C40F'>added/multiplied</span>** together to reconstruct the data shown in the top panel. We can see that the seasonal component changes slowly over time. But this doesn't mean years far apart won't have different seasonal patterns.
# 
# The residual component shown in the bottom panel is what is left over when the seasonal and trend-cycle components have been subtracted from the data.
# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>4.4 | Stationarity </b></p>
# </div>
# 
# ### **What is Stationarity ?**
# 
# A stationary Time Series is one whose properties **<span style='color:#F1C40F'>do not depend</span>** on the time at which the series is observed. Thus, time series with trends, or with seasonality, are not stationary. A time series with **<span style='color:#F1C40F'>cyclic behaviour</span>** (but with no trend or seasonality) is stationary.
# 
# * Strong stationarity: is a **<span style='color:#F1C40F'>stochastic process</span>** whose unconditional joint probability distribution does not change when shifted in time. Consequently, parameters such as mean and variance also do not change over time.
# * Weak stationarity: is a process where mean, variance, autocorrelation are constant throughout the time
# 
# Stationarity is important as non-stationary series that depend on time have too many parameters to account for when modelling the time series. **<span style='color:#F1C40F'>diff method()</span>** can easily convert a non-stationary series to a stationary series.
# 
# ### **What is stationarity used for ?**
# 
# Most statistical forecasting methods are designed to work on a stationary time series. The **<span style='color:#F1C40F'>first step</span>** in the forecasting process is typically to do some transformation to **<span style='color:#F1C40F'>convert a non-stationary series to stationary</span>**. Forecasting a stationary series is relatively easier and the forecasts are more reliable. We know that linear regression works best if the predictors (X variables) are not correlated against each other. So, stationarizing the series solves this problem since it removes any persistent autocorrelation, thereby making the predictors (lags of the series) in the forecasting models nearly independent.
# 
# ### **How to make a Time Series stationary ?**
# 
# There are several ways to do that: 
# 
# * Difference the series once or more times (subtracting the next value by the current value)
# * Take the log of the series (helps to stabilize the variance of a time series.)
# * Take the 𝑛𝑡ℎ root of the series Combinations of the above
# 
# But first, to test if a time series is stationary we can:
# 
# * Look at the time plot.
# * Split the series into 2 parts and compute descriptive statistics. If they differ, then it is not stationary.
# * Perform statistical tests called Unit Root Tests like Augmented Dickey Fuller test (ADF Test), Kwiatkowski-Phillips-Schmidt-Shin — KPSS test (trend stationary), and Philips Perron test (PP Test).
# 
# The most commonly used is the ADF test, where the **<span style='color:#F1C40F'>null hypothesis</span>** is that the time series possesses a unit root (or random walk with drift) and is non-stationary. So, if the P-Value in ADF test is less than the significance level (0.05), you reject the null hypothesis and the series is stationary.

# In[16]:


# check for stationarity
def adf_test(series, title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print('Augmented Dickey-Fuller Test: {}'.format(title))
    # .dropna() handles differenced data
    result = adfuller(series.dropna(),autolag='AIC') 
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out['critical value ({})'.format(key)]=val
        
    # .to_string() removes the line "dtype: float64"
    print(out.to_string())          
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")


# In[17]:


# Aggregating the Time Series to a monthly scaled index
y = df_data[['date','sales']].copy()
y.set_index('date', inplace=True)
y.index = pd.to_datetime(y.index)
y = y.resample('1M').mean()
        
adf_test(y['sales'],title='') 


# If the data is not stationary but we want to use a model such as ARIMA (that requires this characteristic), the data has to be transformed. The two most common methods to transform series into stationarity ones are:
# 
# * Transformation: e.g. log or square root to stabilize non-constant variance
# * Differencing: subtracts the current value from the previous
# 
# Hereafter, we are going to transform sales trend from non-stationarity to stationarity using diff method: 

# In[18]:


fig = plt.figure(figsize=(22,8))
decomp.trend.diff().plot()


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>4.5 | Autocorrelation Analysis</b></p>
# </div>
# 
# After a time series has been stationarized by differencing, the next step in fitting an ARIMA model is to **<span style='color:#F1C40F'>determine whether AR or MA terms</span>** are needed to correct any autocorrelation that remains in the differenced series. Of course, with software like Statgraphics, you could just try some different combinations of terms and see what works best. But there is a more systematic way to do this. By looking at the **<span style='color:#F1C40F'>autocorrelation function </span>**and **<span style='color:#F1C40F'>partial autocorrelation function</span>** plots of the differenced series, you can tentatively identify the numbers of AR and/or MA terms that are needed.
# 
# * Autocorrelation Function (ACF): P = Periods to lag for eg: (if P= 3 then we will use the three previous periods of our time series in the autoregressive portion of the calculation) P helps adjust the line that is being fitted to forecast the series. P corresponds with MA parameter
# * Partial Autocorrelation Function (PACF): D = In an ARIMA model we transform a time series into stationary one(series without trend or seasonality) using differencing. D refers to the number of differencing transformations required by the time series to get stationary. D corresponds with AR parameter.
# 
# 
# 
# 

# In[19]:


#from statsmodels.tsa.stattools import acf
#fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(15, 6))
#plot_lags(df_data[df_data.date >= datetime.datetime(2017,1,1)]['sales'], lags=12, nrows=2)
#plot_acf(df_data[df_data.date >= datetime.datetime(2017,1,1)]['sales'].tolist(), lags=12, ax=ax[0], fft=False);
#plot_pacf(df_data[df_data.date >= datetime.datetime(2017,1,1)]['sales'].tolist(), lags=12, ax=ax[1]);


# 📌 **Interpret:** For autocorrelation, the y-axis is the value for the correlation between a value and its lag. The lag is on the x-axis. The zero-lag has a correlation of 1 because it correlates with itself perfectly.

# # <b>5 <span style='color:#F1C40F'>|</span> Feature Transformation</b>
# 
# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>5.1 | Labeling Non-Numerical Features</b></p>
# </div>
# 
# Using **<span style='color:#F1C40F'>LabelEncoder</span>**, we are going to convert non-numerical features to numerical type. LabelEncoder basically labels the classes from **<span style='color:#F1C40F'>0 to n</span>**. This process is necessary for models to learn from those features.

# In[20]:


non_numerical_cols =  [col for col in df_data.columns if df_data[col].dtype == 'object']
for feature in non_numerical_cols:        
    df_data[feature] = LabelEncoder().fit_transform(df_data[feature])
df_data.head().style.set_properties(subset=non_numerical_cols, **{'background-color': '#F1C40F'})


# In[21]:


df_data.dtypes


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>5.2 | One Hot Encoding</b></p>
# </div>
# 
# To finish with, we are going to one hot encoded non-ordinal features. **<span style='color:#F1C40F'>All</span>** labeled features above are **<span style='color:#F1C40F'>non-ordinal</span>** features. Therefore, we are going to one hot encoded those which have a low cardinality. 

# In[22]:


df_data.dtypes


# In[23]:


low_card_cols = [col for col in non_numerical_cols if len(df_data[col].unique()) < 15]


# In[24]:


encoded_features = []

for feature in low_card_cols:
    encoded_feat = OneHotEncoder().fit_transform(df_data[feature].values.reshape(-1, 1)).toarray()
    n = df_data[feature].nunique()
    cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
    encoded_df = pd.DataFrame(encoded_feat, columns=cols)
    encoded_df.index = df_data.index
    encoded_features.append(encoded_df)

df_data = pd.concat([df_data, *encoded_features[:9]], axis=1)


# In[25]:


df_data.head().style.set_properties(subset=low_card_cols, **{'background-color': '#F1C40F'})


# In[26]:


df_data = df_data.drop(low_card_cols,axis=1)


# # <b>6 <span style='color:#F1C40F'>|</span> Modeling</b>
# 
# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>6.1 | Cross Validation</b></p>
# </div>
# 
# Time series can be either **<span style='color:#F1C40F'>univariate</span>** or **<span style='color:#F1C40F'>multivariate</span>**:
# 
# * Univariate time series only has a single time-dependent variable.
# * Multivariate time series have a multiple time-dependent variable.
# 
# But, first of all we are going to see how does **<span style='color:#F1C40F'>cross validation</span>** technic works in TimeSeries Analysis.

# In[27]:


from sklearn.model_selection import TimeSeriesSplit
N_SPLITS = 3

X = df_data['date']
y = df_data['sales']

folds = TimeSeriesSplit(n_splits=N_SPLITS)


# In[28]:


f, ax = plt.subplots(nrows=N_SPLITS, ncols=2, figsize=(22, 10))

for i, (train_index, valid_index) in enumerate(folds.split(X)):
    X_train, X_valid = X[train_index], X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]

    sns.lineplot(
        x=X_train, 
        y=y_train, 
        ax=ax[i,0], 
        color='dodgerblue', 
        label='train'
    )
    sns.lineplot(
        x=X_train[len(X_train) - len(X_valid):(len(X_train) - len(X_valid) + len(X_valid))], 
        y=y_train[len(X_train) - len(X_valid):(len(X_train) - len(X_valid) + len(X_valid))], 
        ax=ax[i,1], 
        color='dodgerblue', 
        label='train'
    )

    for j in range(2):
        sns.lineplot(x= X_valid, y= y_valid, ax=ax[i, j], color='darkorange', label='validation')
    ax[i, 0].set_title(f"Rolling Window with Adjusting Training Size (Split {i+1})", fontsize=16)
    ax[i, 1].set_title(f"Rolling Window with Constant Training Size (Split {i+1})", fontsize=16)

plt.tight_layout()
plt.show()


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>6.2 | Univariate Time Series Models</b></p>
# </div>
# 
# **<span style='color:#F1C40F'>Univariate time series:</span>** Only one variable is varying over time. For example, data collected from a sensor measuring the temperature of a room every second. Therefore, each second, you will only have a one-dimensional value, which is the temperature.
# 
# ### Prophet
# 
# The first model (which also can handle multivariate problems) we are going to try is Facebook Prophet. Prophet, or “Facebook Prophet,” is an open-source library for univariate (one variable) time series forecasting developed by Facebook. Prophet implements what they refer to as an **<span style='color:#F1C40F'>additive</span>** time series forecasting model, and the implementation supports **<span style='color:#F1C40F'>trends, seasonality, and holidays</span>**. In our case, we are going to use it to show **<span style='color:#F1C40F'>average sales per day</span>** (it is an univariate time series)

# In[29]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

from fbprophet import Prophet

train = df_data[df_data['date']<= datetime.datetime(2017,8,15)][['date','sales']].groupby('date').mean().reset_index('date')
train.columns = ['ds', 'y']
x_valid = pd.DataFrame(df_data[df_data['date']>= datetime.datetime(2017,8,16)]['date'])
x_valid.columns = ['ds']

# Train the model
model = Prophet()
model.fit(train)
y_pred = model.predict(x_valid)


# In[30]:


f, ax = plt.subplots(1)
f.set_figheight(10)
f.set_figwidth(22)

model.plot(y_pred, ax=ax)
sns.lineplot(x=train['ds'], y=train['y'], ax=ax, color='darkorange') #navajowhite

#ax.set_title(f'Prediction \n MAE: {score_mae:.2f}, RMSE: {score_rmse:.2f}', fontsize=14)
ax.set_xlabel(xlabel='Date', fontsize=14)
ax.set_ylabel(ylabel='Sales', fontsize=14)
ax.set_title('Average Sales per Day')

plt.show()


# ### ARIMA
# 
# The Auto-Regressive Integrated Moving Average (ARIMA) model describes the autocorrelations in the data. The model assumes that the time-series is stationary. It consists of three main parts:
# 
# * Auto-Regressive (AR) filter (long term):
# $yt=c+\alpha_1 y_t−1+ \cdots \alpha_n y_{t−n}+\epsilon_t=c+\sum_{i=1}^p\alpha_i y_{t−i}+\epsilon_t \rightarrow p$
# 
# * Integration filter (stochastic trend)
# $\rightarrow d$
# 
# * Moving Average (MA) filter (short term):
# 
# $y_t=c+\epsilon_t+\beta_1 \epsilon_t−1+ \cdots +\beta_q \epsilon_t−q=c+\epsilon_t+\sum_{i=1}^q \beta_i \epsilon_t−i \rightarrow q$
# 
# ARIMA: $y_t=c+\alpha_1 y_{t−1}+\cdots+\alpha_p y_{t−p}+\epsilon_t+ \beta_1 \epsilon_{t−1}+\cdots+\beta_q\epsilon_{t−q}$
# 
# ARIMA( p, d, q)
# 
# * p: Lag order (reference PACF in Autocorrelation Analysis)
# * d: Degree of differencing. (reference Differencing in Stationarity)
# * q: Order of moving average (check out ACF in Autocorrelation Analysis)
# 
# Steps to analyze ARIMA
# 
# * **<span style='color:#F1C40F'>Step 1 — Check stationarity:</span>** If a time series has a trend or seasonality component, it must be made stationary before we can use ARIMA to forecast. .
# * **<span style='color:#F1C40F'>Step 2 — Difference:</span>** If the time series is not stationary, it needs to be stationarized through differencing. Take the first difference, then check for stationarity. Take as many differences as it takes. Make sure you check seasonal differencing as well.
# * **<span style='color:#F1C40F'>Step 3 — Filter out a validation sample:</span>** This will be used to validate how accurate our model is. Use train test validation split to achieve this
# * **<span style='color:#F1C40F'>Step 4 — Select AR and MA terms:</span>**Step 4 — Select AR and MA terms: Use the ACF and PACF to decide whether to include an AR term(s), MA term(s), or both.
# * **<span style='color:#F1C40F'>Step 5 — Build the model:</span>** Build the model and set the number of periods to forecast to N (depends on your needs).
# * **<span style='color:#F1C40F'>Step 6 — Validate model:</span>** Compare the predicted values to the actuals in the validation sample.

# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>6.3 | Multivariate Time Series Models</b></p>
# </div>
# 
# Finnally, we are going to analize multivariate TimeSeries forecasting.
# 
# **<span style='color:#F1C40F'>Multivariate time series:</span>** Multiple variables are varying over time. For example, a tri-axial accelerometer. There are three accelerations, one for each axis (x,y,z) and they vary simultaneously over time.

# In[31]:


train_multivariate = df_data[df_data['date']<= datetime.datetime(2017,8,15)][['date','sales','dcoilwtico']].groupby('date').mean().reset_index('date')
train_multivariate.columns = ['ds', 'y','dcoilwtico']
x_valid = pd.DataFrame(df_data[df_data['date']>= datetime.datetime(2017,8,16)][['date','dcoilwtico']])
x_valid.columns = ['ds','dcoilwtico']

# Train the model
model_multivariate = Prophet()
model_multivariate.add_regressor('dcoilwtico')

model_multivariate.fit(train_multivariate)
y_pred_multivariate = model_multivariate.predict(x_valid)


# In[32]:


f, ax = plt.subplots(1)
f.set_figheight(10)
f.set_figwidth(22)

model.plot(y_pred_multivariate, ax=ax)
sns.lineplot(x=train_multivariate['ds'], y=train_multivariate['y'], ax=ax, color='darkorange') #navajowhite

#ax.set_title(f'Prediction \n MAE: {score_mae:.2f}, RMSE: {score_rmse:.2f}', fontsize=14)
ax.set_xlabel(xlabel='Date', fontsize=14)
ax.set_ylabel(ylabel='Sales', fontsize=14)
ax.set_title('Average Sales per Day')

plt.show()


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>6.4 | Boosted Hybrid Model (Extension)</b></p>
# </div>
# 
# We'll create a **<span style='color:#F1C40F'>boosted hybrid</span>** for the Store Sales dataset by implementing a new Python class. We'll start by defining the new class. Then, we'll add **<span style='color:#F1C40F'>fit and predict</span>** methods to give it a scikit-learn like interface.

# In[33]:


class BoostedHybrid:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2
        self.y_columns = None  # store column names from fit method
        
def fit(self, X_1, X_2, y):
    # Train model_1
    self.model_1.fit(X_1, y)

    # Make predictions
    y_fit = pd.DataFrame(
        self.model_1.predict(X_1), 
        index=X_1.index, columns=y.columns,
    )

    # Compute residuals
    y_resid = y - y_fit
    y_resid = y_resid.stack().squeeze() # wide to long

    # Train model_2 on residuals
    self.model_2.fit(X_2, y_resid)

    # Save column names for predict method
    self.y_columns = y.columns
    # Save data for question checking
    self.y_fit = y_fit
    self.y_resid = y_resid

def predict(self, X_1, X_2):
    # Predict with model_1
    y_pred = pd.DataFrame(
        self.model_1.predict(X_1), 
        index=X_1.index, columns=self.y_columns,
    )
    y_pred = y_pred.stack().squeeze()  # wide to long

    # Add model_2 predictions to model_1 predictions
    y_pred += self.model_2.predict(X_2)

    return y_pred

BoostedHybrid.fit = fit
BoostedHybrid.predict = predict


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>4.2 | Linear Regression Features</b></p>
# </div>
# 
# Now, we are going to create our training dataset for linear regression algorithm. 

# In[34]:


#df_train_2017 = df_data[df_data['date'] >= datetime.datetime(2017,1,1)]
#df_train_2017 = df_train_2017[df_train_2017['date'] <= datetime.datetime(2017,8,15)]
df_train_2017 = df_data[df_data['date'] <= datetime.datetime(2017,8,15)]
y_train = df_train_2017[['sales','date']]
y_train['date'] = y_train.date.dt.to_period('D')
y_train = y_train.set_index('date')
y_train.head(2)


# In[35]:


fourier = CalendarFourier(freq='M', order=4)
dp = DeterministicProcess(
    index=y_train.index,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True,
)
x_train = dp.in_sample()
x_train['NewYear'] = (x_train.index.dayofyear == 1)
x_test = dp.out_of_sample(steps=28512)
x_test.index.name = 'date'
x_test['NewYear'] = (x_test.index.dayofyear == 1)
x_test


# In[36]:


x_test = x_test.reset_index()
x_test


# In[37]:


fechas = []
for i in range(0,16):
    fechas.append(x_test.loc[i,'date'])


# In[38]:


for i in range(0,16):
    x_test.loc[1782*i:1782*(i+1)-1,'date'] = fechas[i]
x_test = x_test.set_index('date')
x_test


# <div style="color:white;display:fill;border-radius:8px;
#             background-color:#323232;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>4.3 | XGBoost Features</b></p>
# </div>
# 
# Now, we are going to create our training dataset for XGBoost algorithm. 

# In[39]:


x_train_2 = df_train_2017.drop('sales',axis=1)
x_train_2['date'] = x_train_2.date.dt.to_period('D')
x_train_2 = x_train_2.set_index('date')
x_train_2 = x_train_2.drop('week',axis=1)
x_train_2


# In[40]:


df_test = df_data[df_data['date'] >= datetime.datetime(2017,8,16)]
x_test_2 = df_test.drop('sales',axis=1)
x_test_2['date'] = x_test_2.date.dt.to_period('D')
x_test_2 = x_test_2.set_index('date')
x_test_2 = x_test_2.drop('week',axis=1)
x_test_2


# In[41]:


model = BoostedHybrid(
    model_1=LinearRegression(),
    model_2=XGBRegressor(),
)
model.fit(x_train, x_train_2, y_train)

y_pred = model.predict(x_test, x_test_2)
y_pred = y_pred.clip(0.0)


# In[42]:


y_pred = pd.DataFrame(y_pred).reset_index()
y_submit = pd.DataFrame({'id':test.id,'sales':y_pred.loc[:,0]}).set_index('id')
y_submit.to_csv('./submission.csv')


# In[ ]:





# In[ ]:




