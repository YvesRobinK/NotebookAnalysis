#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('wget https://raw.githubusercontent.com/JoseCaliz/dotfiles/main/css/gruvbox.css 2>/dev/null 1>&2')
get_ipython().system('pip install feature_engine 2>/dev/null 1>&2')
    
from IPython.core.display import HTML
with open('./gruvbox.css', 'r') as file:
    custom_css = file.read()

HTML(custom_css)


# # Table Of Content
# 
# 1. [Introduction](#Introduction)
# 1. [Library Import](#Library-Import)
# 1. [Holidays](#Holidays)
# 1. [Read Competition Data](#Read-Competition-Data)
# 1. [Feature Engineering](#Feature-Engineering)
# 1. [Cross-Validation](#Cross-Validation)
# 1. [Modeling](#Modeling)
# 1. [Residuals Plot](#Residuals-Plot)
# 1. [Adding Workers' Day](#Adding-Workers'-Day)
# 1. [Re-train and compare results](#Re-train-and-compare-results)
#     1. [Additional Holidays/Events](#Additional-Holidays/Events)
# 
# # Introduction
# 
# 
# By the time, there has been multiple approaches to reduce the LB score
# 
# 1. GDP per capita
# 2. Lockdown-dates due to COVID pandemic.
# 3. CCI
# 
# I tried some of them but not all seems to be working, at least for me (I'm open to discussions related to what has worked an what not).
# 
# One approach that always make sense to investigate is holidays, and not only the traditional holidays like easter and christmas, I'm talking about local holidays, black fridays, mother's day, etc, etc.
# 
# This notebook provides an interactive way to explore holidays, predictions residuals and fix them in order to improve your CV score. I'm using and aggregated models that predicts the sum of all `num_sold` grouped by date.
# 
# 
# **What is not covered**:
# 1. Dissagregate forecast by store / country. Here is a list of other awesome resources:
#     * [@samuelcortinhas](https://www.kaggle.com/samuelcortinhas) | [📈 TPS Sept 22 - Timeseries Analysis](https://www.kaggle.com/code/samuelcortinhas/tps-sept-22-timeseries-analysis)
#     * [@kaggleqrdl](https://www.kaggle.com/kaggleqrdl) | [disaggregate_forecast_cabaxiom_fork](https://www.kaggle.com/code/kaggleqrdl/disaggregate-forecast-cabaxiom-fork)
#     
#     
# * v1: Initial commit
# * v2: Adding additional non-local holidays.
# * v3: TOC.
# * v4: Adding a section to show that ratios are also affected by holidays
# * v5: Adding brand new css -ignore-

# # Library Import

# In[2]:


from rich import print
from plotly.subplots import make_subplots

from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import dateutil.easter as easter
import holidays as holidays
import matplotlib as mpl
import matplotlib.cm as cmap
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

cluster_colors_hex = ['#b4d2b1', '#568f8b', '#1d4a60', '#cd7e59', '#ddb247', '#d15252']
cluster_colors_rgb = [hex_to_rgb(x) for x in cluster_colors_hex]
cmap = mpl_colors.ListedColormap(cluster_colors_rgb)
colors = cmap.colors
bg_color= '#fdfcf6'

custom_params = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    'grid.alpha':0.3,
    'figure.figsize': (16, 6),
    'axes.titlesize': 'Large',
    'axes.labelsize': 'Large',
    'figure.facecolor': bg_color,
    'axes.facecolor': bg_color
}

sns.set_theme(
    style='whitegrid',
    palette=sns.color_palette(cluster_colors_hex),
    rc=custom_params
)

pio.templates.default = "plotly_white"
pio.templates['plotly_white'].layout.colorway = cluster_colors_hex
pio.templates['plotly_white'].layout.plot_bgcolor = bg_color
pio.templates['plotly_white'].layout.paper_bgcolor = bg_color


# # Holidays
# 
# I'm going to use `holidays` package to create a dataframe with date, holiday name, and country

# In[3]:


def get_holidays_by_country(holidays_object, years, country):
    holidays_list = []
    for date, name in sorted(holidays_object(years=years).items()):
        holidays_list.append([country, date, name])
    return holidays_list

holidays_objects = {
    'Belgium': holidays.BE,
    'Italy': holidays.IT,
    'Poland': holidays.PL,
    'Germany': holidays.DE,
    'France': holidays.FR,
    'Spain': holidays.ES
}

holidays_list = []
for country, holidays_object in holidays_objects.items():
    holidays_list += get_holidays_by_country(
        holidays_object, [x for x in range(2017, 2022)], country
    )
    
holidays_df = pd.DataFrame.from_records(
    holidays_list, columns=['Country', 'date', 'name']
)

holidays_df['date'] = pd.to_datetime(holidays_df.date)
print('[blue][bold]Holidays[/bold][/blue] Dataframe:')
holidays_df.head()


# # Read Competition Data

# In[4]:


train_df = pd.read_csv(
    "../input/tabular-playground-series-sep-2022/train.csv",
    parse_dates=['date'],
)

test_df = pd.read_csv(
    "../input/tabular-playground-series-sep-2022/test.csv",
    parse_dates=['date'],
)


# # Feature Engineering
# 
# Only Christmas holidays and initial days of year are included as boolean features. Next we'll see there are other important holidays that worth taking into consideration

# In[5]:


def engineer(df):
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayofyear'] = df['date'].dt.dayofyear

    df['friday'] = df.date.dt.weekday.eq(4).astype(np.uint8)
    df['saturday'] = df.date.dt.weekday.eq(5).astype(np.uint8)
    df['sunday'] = df.date.dt.weekday.eq(6).astype(np.uint8)
    
    df["month_sin"] = np.sin(df['month'] * (2 * np.pi / 12))
    df["month_cos"] = np.cos(df['month'] * (2 * np.pi / 12))
    
    #X-mas Holidays Indicator
    for day in range(24, 32):
        df[f'Dec_{day}'] = df.date.dt.day.eq(day) & df.date.dt.month.eq(12)
        
    #New year starting days
    for day in range(1, 9):
        df[f'Jan_{day}'] = df.date.dt.day.eq(day) & df.date.dt.month.eq(1)
    
    df.set_index('date', inplace=True)
    df.drop(columns=['month'], inplace=True)
    return df

train_df_ = train_df.groupby(["date"])["num_sold"].sum().reset_index()
test_df_ = test_df.groupby(["date"])["row_id"].first().reset_index().drop(columns="row_id")
test_preds = test_df_[["date"]]

train_df_ = engineer(train_df_)
test_df_ = engineer(test_df_)

target = train_df_.pop('num_sold')
target_ = np.log(target)


# # Cross-Validation
# 
# 
# It is not surprise that we need some CV scheme to avoid overfitting the public leaderboard, this is an [excellent post](https://www.kaggle.com/competitions/tabular-playground-series-sep-2022/discussion/350505) talking about all possibles schemes. I'll be using GroupKFold with `year` as groups. `train` function is just a cross_validate shortcut.

# In[6]:


years = [x for x in range(2017, 2022)]
train_df_['idx'] = range(train_df_.shape[0])
cv_idx = []

for year in years[:-1]:
    tr_idx = train_df_.loc[train_df_.year.ne(year), 'idx'].to_list()
    vl_idx = train_df_.loc[train_df_.year.eq(year), 'idx'].to_list()
    cv_idx.append((tr_idx, vl_idx))
    
train_df_.drop(columns='idx', inplace=True)

def smape(y_true, y_pred, overwrite=False):
    y_true = np.exp(y_true)
    y_pred = np.exp(y_pred)   
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)

scorer_smape = make_scorer(smape)
def train(pipeline, train_df_, target):
    cv_results = cross_validate(
        pipeline,
        train_df_,
        target,
        cv=cv_idx,
        return_train_score=True,
        return_estimator=True,
        scoring={
            'smape': scorer_smape,
            'msq': 'neg_mean_squared_error'
        }
    )
    return cv_results


# # Modeling

# In[7]:


pipeline = make_pipeline(StandardScaler(), Ridge(random_state=0))
initial_cv_results = train(pipeline, train_df_, target_)
initial_cv_results_df = pd.DataFrame.from_dict(initial_cv_results)
initial_cv_results_df.index = years[:-1]
initial_cv_results_df.index.name = 'year'
initial_cv_results_df


# # Residuals Plot
# 
# Every vertical line is a holiday, high peaks means our prediction is far away than the real values. **Hover top of vertical line for holiday name**

# In[8]:


fig = make_subplots(rows=4, cols=1)
hovertemplate = '%{x}<br>%{hovertext}<br>'

for i, estimator in enumerate(initial_cv_results['estimator']):
    curr_year = years[i]
    pred_df = train_df_.iloc[cv_idx[i][1]]
    targets = target.iloc[cv_idx[i][1]]
    preds = np.exp(estimator.predict(pred_df).flatten())
    residuals =  (preds - targets).abs()
    
    # Residuals Plot
    fig.append_trace(
        go.Scatter(
            x=pred_df.index, y=residuals,
            name=curr_year, line=dict(color=cluster_colors_hex[i])
        ),
        row=i+1, col=1
    )

    lines = holidays_df.query('date.dt.year == @curr_year')
    
    # Scatter plot to add hover text to holidays
    fig.append_trace(
        go.Scatter(
            x=lines.date, y=[residuals.max()]*lines.shape[0],
            hovertext=lines.Country + ': ' + lines.name,
            mode='markers',
            showlegend=False,
            hovertemplate=hovertemplate,
            marker=dict(color=cluster_colors_hex[i], size=3)
        ),
        row=i+1, col=1
    )
    
    # Vertical lines marking holidays
    for idx, line in lines.iterrows():
        fig.add_shape(
            go.layout.Shape(
                type='line',
                xref='x',
                x0=line['date'],
                y0=0,
                x1=line['date'],
                y1=residuals.max(),
                line={'dash': 'dot', 'width':1}
            ),row=i+1, col=1
        )        

fig.update_layout(height=800, title_text="Residuals per Year")
fig.show()


# We can see than in all plot there is a peak around may 1st, further inspection show that there is indeed a holiday in all countries!! **Worker's Day**

# In[9]:


holidays_df[
    holidays_df.date.dt.month.eq(5) &
    holidays_df.date.dt.day.between(1, 5)
].T


# # Adding Workers' Day

# In[10]:


train_df_.reset_index(inplace=True)
test_df_.reset_index(inplace=True)

#Worker's Day
for day in range(1, 10):
    train_df_[f'May_{day}'] = train_df_.date.dt.day.eq(day) & train_df_.date.dt.month.eq(5)
    test_df_[f'May_{day}'] = test_df_.date.dt.day.eq(day) & test_df_.date.dt.month.eq(5)
    
train_df_.set_index('date', inplace=True)
test_df_.set_index('date', inplace=True)


# # Re-train and compare results

# In[11]:


second_cv_results = train(pipeline, train_df_, target_)
second_cv_results_df = pd.DataFrame.from_dict(second_cv_results)
second_cv_results_df.index = years[:-1]
second_cv_results_df.index.name = 'year'

smapes = pd.merge(
    initial_cv_results_df,
    second_cv_results_df,
    suffixes=('_inital', '_+workers'),
    on='year'
)[['test_smape_inital', 'test_smape_+workers']]


smapes.loc['mean', :] = smapes.mean(axis=0)
display(
    smapes.style.background_gradient(axis=1, cmap=sns.dark_palette('green', as_cmap=True, reverse=True))
)

print("[green]2017, 2019, and 2020 [/green] benefits on having Worker's day as a holiday")


# I hope this notebook is useful. If you find other holidays that might impact the CV I'm always looking forward to discuss them.

# ## Additional Holidays/Events
# 
# 1. Eurocup 2021: Jun 11, 2021 – Jul 11, 2021
# 2. Mother's day: second sunday of each May.
# 3. Black Friday:
#     * 2017: November 23
#     * 2018: November 22
#     * 2019: November 28
#     * 2020: November 26
#     * 2021: November 25
# 

# # Country Ratios
# 
# Contry ratios are also affected by special dates, a model should be able to predict the right ration for 2021

# In[12]:


train_df_country = train_df.copy()
train_df_country = train_df_country.groupby(['date', 'country'])['num_sold'].sum()
train_df_country = train_df_country.groupby(level=0).transform(lambda s: s/s.sum())
train_df_country = train_df_country.reset_index()
train_df_country.rename(columns={'num_sold': 'ratio'}, inplace=True)

fig = make_subplots(6, 1)
for i, country in enumerate(train_df_country.country.unique()):
    df = train_df_country[
        train_df_country.date.dt.year.eq(2020) & 
        train_df_country.country.eq(country)
    ]
    
    fig.add_trace(
        go.Scatter(
            x=df.date, y=df.ratio, name=country,
            line=dict(color=cluster_colors_hex[i])
        ), row=i+1, col=1
    )

    lines = holidays_df.query('date.dt.year == 2020 and Country == @country')
    # Scatter plot to add hover text to holidays
    fig.add_trace(
        go.Scatter(
            x=lines.date,
            y=[0.2]*lines.shape[0],
            hovertext=lines.Country + ': ' + lines.name,
            mode='markers',
            showlegend=False,
            hovertemplate=hovertemplate,
            marker=dict(color=cluster_colors_hex[i], size=3)
        ),
        row=i+1, col=1
    )

    # Vertical lines marking holidays
    for idx, line in lines.iterrows():
        fig.add_shape(
            go.layout.Shape(
                type='line',
                xref='x',
                x0=line['date'],
                y0=0.1,
                x1=line['date'],
                y1=0.2,
                line={'dash': 'dot', 'width':1}
            ),
            row=i+1, col=1
        )

fig.update_layout(height=800, title='Contry Ratios for 2020')

