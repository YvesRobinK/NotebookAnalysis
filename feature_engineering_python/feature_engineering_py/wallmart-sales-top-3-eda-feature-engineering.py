#!/usr/bin/env python
# coding: utf-8

# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> - | Introduction </div>
# 
# <p style="line-height: 100%; margin-left:20px; font-size:22.5px; font-family:cabin;">   
# In this notebook, my goal is to attain a high ranking in the Wallmart Sales competition, hopefully placing in the top 10% to earn a medal. With my most recent update, I have managed to reach the top 3% of competitors.
# </p>
# 

# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> - | Table of contents</div>
# 
# * [1-Libraries](#section-one)
# * [2-Data loading](#section-two)
# * [3-EDA](#section-three)
#     * [3.1-Sales analysis](#subsection-three-one)
#     * [3.2-Other feature analysis](#subsection-three-two)
#     * [3.3-Heatmap and correlation between features](#subsection-three-three)
# * [4-Feature engineering](#section-four)
#     - [4.1-Holidays](#subsection-four-one)
#     - [4.2-Markdowns](#subsection-four-two)
# * [5-Preprocessing](#section-five)
#     - [5.1-Filling missing values](#subsection-five-one)
#     - [5.2-Encoding categorical data](#subsection-five-two)
# * [6-Feature Selection](#section-six)
# * [7-Modeling](#section-seven)
#     * [7.1-Baseline prediction](#subsection-seven-one)
#     * [7.2-Simple blend](#subsection-seven-two)

# <a id="section-one"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 1 | Libraries</div>

# In[1]:


# Data handling
import pandas as pd
import numpy as np

# Viz
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff

# Sklearn
from sklearn import model_selection, metrics

# Feature selection
import eli5
from eli5.sklearn import PermutationImportance

# Models
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn import linear_model, ensemble
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

# Remove warnings
import warnings
warnings.filterwarnings('ignore') 


# <a id="section-two"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 2 | Data loading</div>

# In[2]:


features = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/features.csv.zip')
train = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/train.csv.zip')
stores = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/stores.csv')
test = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/test.csv.zip')
sample_submission = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip')


# In[3]:


feature_store = features.merge(stores, how='inner', on = "Store")


# In[4]:


train_df = train.merge(feature_store, how='inner', on = ['Store','Date','IsHoliday']).sort_values(by=['Store','Dept','Date']).reset_index(drop=True)


# In[5]:


test_df = test.merge(feature_store, how='inner', on = ['Store','Date','IsHoliday']).sort_values(by = ['Store','Dept','Date']).reset_index(drop=True)


# In[6]:


feature_store = features.merge(stores, how='inner', on = "Store")

# Converting date column to datetime 
feature_store['Date'] = pd.to_datetime(feature_store['Date'])
train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])

# Adding some basic datetime features
feature_store['Day'] = feature_store['Date'].dt.day
feature_store['Week'] = feature_store['Date'].dt.week
feature_store['Month'] = feature_store['Date'].dt.month
feature_store['Year'] = feature_store['Date'].dt.year


# In[7]:


train_df = train.merge(feature_store, how='inner', on = ['Store','Date','IsHoliday']).sort_values(by=['Store','Dept','Date']).reset_index(drop=True)


# In[8]:


test_df = test.merge(feature_store, how='inner', on = ['Store','Date','IsHoliday']).sort_values(by = ['Store','Dept','Date']).reset_index(drop=True)


# <a id="section-three"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 3 | Exploratory Data Analysis (EDA)</div>
# 
# <p style="line-height: 100%; margin-left:20px; font-size:22.5px; font-family:cabin;">
# The EDA is one of the most important parts of the process, because will gives you an idea about the relationship of the features, your distribution, and so on.
#     <br><br>

# In[9]:


train_df.describe().T.style.bar(subset=['mean'], color='#205ff2')\
                            .set_caption("Stats Summary of Numeric Variables")\
                            .background_gradient(subset=['min'], cmap='Reds')\
                            .background_gradient(subset=['max'], cmap='Greens')\
                            .background_gradient(subset=['std'], cmap='GnBu')\
                            .background_gradient(subset=['50%'], cmap='GnBu')


# In[10]:


palletes = {
   'continuos':{'blues': ['#03045E', '#023E8A', '#0077B6', '#0077B6', '#0096C7', '#00B4D8', '#48CAE4', '#90E0EF', '#ADE8F4', '#CAF0F8'],
                'green_n_blues': ['#D9ED92', '#B5E48C', '#99D98C', '#76C893', '#52B69A', '#34A0A4', '#168AAD', '#1A759F', '#1E6091', '#184E77']
               }
            }


# In[11]:


template = dict(layout=go.Layout(font=dict(family="Enriqueta", size=12))) # Cabin | Franklin Bold


# <a id="subsection-three-one"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#4AC9FE;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%;margin-left:10px"> 3.1 | Sales analysis</div>

# In[12]:


df_weeks = train_df.groupby('Week').sum()

fig = px.line(data_frame=df_weeks, x=df_weeks.index, y='Weekly_Sales', 
              template='simple_white', 
              labels={'Weekly_Sales' : 'Total Sales', 'x' : 'Weeks'})

fig.update_layout(
    template=template, 
    title={'text':'<b>Sales over the year across every week</b>', 'x': 0.075},
    xaxis=dict(tickmode='linear', showline=True), 
    yaxis=dict(showline=True))

fig.add_annotation(
    x=0, y=-0.2, 
    align='left', 
    font=dict(size=12),
    textangle=0, 
    xref="paper", 
    yref="paper", 
    showarrow=False,
    text="<span style='font-size:16px;'><b><i>Findings</b></i>: <i>Sales remain relatively stable throughout the year, experiencing a notable dip around week 42 and a subsequent resurgence during the holiday season.</i>", )


# <p style="font-size:25px; font-family:cabin; line-height: 1.7em; margin-left:20px">
# <b> Markdowns relationship with sales </b></p>

# In[13]:


legend_names = {'MarkDown1': "MD 1",
                'MarkDown2': 'MD 2',
                'MarkDown3': 'MD 3',
                'MarkDown4': 'MD 4',
                'MarkDown5': 'MD 5',
                'Weekly_Sales': 'Sales'}


fig = px.line(df_weeks, x=df_weeks.index, y=['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'Weekly_Sales'], 
              color_discrete_sequence=palletes['continuos']['green_n_blues'],
              template='simple_white', 
              labels={'value' : 'Total Sales', 'x' : 'Weeks'})

for trace_index, trace in enumerate(fig.data):
    trace.name = legend_names[trace.name]

fig.update_layout(
    template=template, 
    title={'text':'<b>Markdowns (MD) vs Sales</b><br><sup>Impact of Markdowns in Sales over the year across every week</sup>', 'x': 0.075},
    legend_title_text='<b>MDs & Sales</b>',
    xaxis=dict(tickmode='linear', showline=True), 
    yaxis=dict(showline=True))

fig.add_annotation(
    x=0, y=-0.2, 
    align='left', 
    font=dict(size=12),
    textangle=0, 
    xref="paper", 
    yref="paper", 
    showarrow=False,
    text="<span style='font-size:16px;'><b><i>Findings</b></i>: <i>Markdowns (MDs) play a significant role in boosting sales during the beginning and end of the year, contributing to overall sales stability.</i>")


# <p style="font-size:25px; font-family:cabin; line-height: 1.7em; margin-left:20px">
# <b> Mean sales comparassion across the years
#          </b></p>

# In[14]:


weekly_sales = train_df.groupby(['Year','Week'], as_index = False).agg({'Weekly_Sales': ['mean', 'median']})
weekly_sales2010 = train_df.loc[train_df['Year']==2010].groupby(['Week']).agg({'Weekly_Sales': ['mean', 'median']})
weekly_sales2011 = train_df.loc[train_df['Year']==2011].groupby(['Week']).agg({'Weekly_Sales': ['mean', 'median']})
weekly_sales2012 = train_df.loc[train_df['Year']==2012].groupby(['Week']).agg({'Weekly_Sales': ['mean', 'median']})

weekly_sales_data = {
    '2010': weekly_sales2010['Weekly_Sales']['mean'].to_dict(),
    '2011': weekly_sales2011['Weekly_Sales']['mean'].to_dict(),
    '2012': weekly_sales2012['Weekly_Sales']['mean'].to_dict()
}

weekly_sales_df = pd.DataFrame(weekly_sales_data)


# In[15]:


line_columns = ['2010', '2011', '2012']

weekly_sales_df_sorted = weekly_sales_df.sort_index()

fig = px.line(weekly_sales_df_sorted, x=weekly_sales_df_sorted.index, y=line_columns, 
              labels={'x': 'Week', 'value': 'Total Sales'},
              color_discrete_sequence=palletes['continuos']['blues'])


fig.update_layout(
    template=template, 
    margin=dict(b=95),
    title={'text':'<b>Sales across the years by weeks<b><br><sup>Comparassion between 2010,2011 and 2012</sup>', 'x': 0.075}, xaxis_title='Week',
    legend_title_text='<b>Year</b>',
    xaxis=dict(tickmode='linear', showline=True), 
    yaxis=dict(showline=True))

fig.add_annotation(
    x=47, y=25000,
    text="<b>Thanksgiving</b>", 
    bordercolor="#585858",
    showarrow=False, 
    borderpad=2.5, 
    bgcolor='white')

fig.add_annotation(
    x=51, y=29000,
    text="<b>Christmas</b>",  
    bordercolor="#585858", 
    showarrow=False, 
    borderpad=2.5,
    bgcolor='white')

fig.add_annotation(
    x=0, y=-0.25, 
    align='left', 
    font=dict(size=12),
    textangle=0, 
    xref="paper", 
    yref="paper", 
    showarrow=False,
    text="<span style='font-size:16px;'><b><i>Findings</b></i>: <i>A distinct sales pattern emerges across the years, characterized by a dramatic surge around Thanksgiving and Christmas, followed by<br>a sharp decline post-holidays.</i>")


# <a id="subsection-three-two"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#4AC9FE;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%;margin-left:10px"> 3.2 | Other features analysis</div>

# In[16]:


# Converting the temperature to celsius for a better interpretation
train_df['Temperature'] = train_df['Temperature'].apply(lambda x :  (x - 32) / 1.8)
train_df['Temperature'] = train_df['Temperature'].apply(lambda x :  (x - 32) / 1.8)


# In[17]:


train_plt = train_df.sample(frac=.1, random_state=42)


# <p style="font-size:25px; font-family:cabin; line-height: 1.7em; margin-left:20px">
# <b> Temperature
#          </b></p>

# In[18]:


fig = px.histogram(train_plt, x='Temperature', y='Weekly_Sales', color='IsHoliday', marginal='box', opacity=0.55,
                   facet_col='IsHoliday', facet_col_spacing=0.05,
                   color_discrete_sequence=palletes['continuos']['blues'])

fig.update_layout(
    template=template, 
    title={'text':'<b>Behaviour of Temperature and Sales by Holiday<br><sup>Is Temperature a key factor in determining Sales?</sup>', 'x': 0.075}, 
    yaxis_title='Total Sales', xaxis_title=' ',
    legend_title_text='<b>Holidays</b>',
    legend=dict(orientation="h", yanchor="top", x=0.7, y=1.2))

fig.for_each_xaxis(
    lambda x: x.update(title=''))

fig.add_annotation(
    x=0.5, y=-0.125, 
    align='center', 
    font=dict(size=12),
    textangle=0, 
    xref="paper", 
    yref="paper", 
    showarrow=False,
    text="<span style='font-size:16px;'>Temperature<sup><b> (celsius)</b>")

fig.add_annotation(
    x=0, y=-0.225, 
    align='left', 
    font=dict(size=12),
    textangle=0, 
    xref="paper", 
    yref="paper", 
    showarrow=False,
    text="<span style='font-size:16px;'><b><i>Findings</b></i>: <i>A noticeable correlation exists between Temperature and Sales, likely attributed to the United States' location in the Northern Hemisphere,<br>where a significant portion of the country experiences cold temperatures during this period.</i>")


# <p style="font-size:25px; font-family:cabin; line-height: 1.7em; margin-left:20px">
# <b> Fuel Price
#          </b></p>

# In[19]:


fig=px.histogram(train_plt, x='Fuel_Price', y ='Weekly_Sales', color='IsHoliday', marginal='box', opacity= 0.55,
                 facet_col='IsHoliday', facet_col_spacing=0.05,
                 color_discrete_sequence=palletes['continuos']['blues'])

fig.update_layout(template=template, 
                  title={'text':'<b>Fuel Price behaviour and Sales by Holiday</b><br><sup>Is Fuel Price causing an impact on Sales?</sup>', 'x': 0.075}, 
                  yaxis_title='Total Sales', xaxis_title='',
                  legend_title_text='<b>Holidays</b>',
                  legend=dict(orientation="h", yanchor="top", x=0.7, y=1.2))

fig.for_each_xaxis(
    lambda x: x.update(title=''))

fig.for_each_annotation(
    lambda x: x.update(text=''))

fig.add_annotation(
    x=0.5, y=-0.125, 
    align='center', 
    font=dict(size=12),
    textangle=0, 
    xref="paper", 
    yref="paper", 
    showarrow=False,
    text="<span style='font-size:16px;'>Fuel Price")

fig.add_annotation(
    x=0, y=-0.225, 
    align='left', 
    font=dict(size=12),
    textangle=0, 
    xref="paper", 
    yref="paper", 
    showarrow=False,
    text="<span style='font-size:16px;'><b><i>Findings</b></i>: <i>It is difficult to identify a clear pattern in this case.</i>")


# <p style="font-size:25px; font-family:cabin; line-height: 1.7em; margin-left:20px">
# <b> Consumer Price Index (CPI)
#          </b></p>

# In[20]:


fig = px.histogram(train_plt, x='CPI', y ='Weekly_Sales', color='IsHoliday', marginal='box', opacity= 0.55,
                   facet_col='IsHoliday', facet_col_spacing=0.05,
                   title='CPI and sales by holiday',color_discrete_sequence=palletes['continuos']['blues'])

fig.update_layout(
    template=template, 
    title={'text':'<b>Inflation (CPI) impact in Sales by Holiday</b><br><sup>The rise in consumer prices afect Sales?</sup>', 'x': 0.075}, 
    yaxis_title='Total Sales', xaxis_title='',
    legend_title_text='<b>Holidays</b>',
    legend=dict(orientation="h", yanchor="top", x=0.7, y=1.2))

fig.for_each_xaxis(
    lambda x: x.update(title=''))

fig.for_each_annotation(
    lambda x: x.update(text=''))

fig.add_annotation(
    x=0.5, y=-0.125, 
    align='center', 
    font=dict(size=12),
    textangle=0, 
    xref="paper", 
    yref="paper", 
    showarrow=False,
    text="<span style='font-size:16px;'>Consumer Price Index")

fig.add_annotation(
    x=0, y=-0.225, 
    align='left', 
    font=dict(size=12),
    textangle=0, 
    xref="paper", 
    yref="paper", 
    showarrow=False,
    text="<span style='font-size:16px;'><b><i>Findings</b></i>: <i>While CPI does influence Sales behavior, the impact is not as significant as I anticipated. Additionally, for some reason, Sales Holidays tend to be lower<br>for products with moderate CPI compared to those with higher CPI values. This observation warrants further investigation.</i>")


# <p style="font-size:25px; font-family:cabin; line-height: 1.7em; margin-left:20px">
# <b> Unemployment
#          </b></p>

# In[21]:


fig = px.histogram(train_plt, x='Unemployment', y ='Weekly_Sales', color='IsHoliday', marginal='box', opacity= 0.6,
                   facet_col='IsHoliday', facet_col_spacing=0.05,
                   color_discrete_sequence=palletes['continuos']['blues'])

fig.update_layout(
    template=template, 
    title={'text':'<b>Unemployment Rate and Sales by Holiday</b><br><sup>How Unemployment afect Sales?</sup>', 'x': 0.075}, 
    yaxis_title='Total Sales', xaxis_title='',
    legend_title_text='<b>Holidays</b>',
    legend=dict(orientation="h", yanchor="top", x=0.7, y=1.2))

fig.for_each_xaxis(
    lambda x: x.update(title=''))

fig.for_each_annotation(
    lambda x: x.update(text=''))

fig.add_annotation(
    x=0.5, y=-0.125, 
    align='center', 
    font=dict(size=12),
    textangle=0, 
    xref="paper", 
    yref="paper", 
    showarrow=False,
    text="<span style='font-size:16px;'>Unemployment Rate")

fig.add_annotation(
    x=0, y=-0.225, 
    align='left', 
    font=dict(size=12),
    textangle=0, 
    xref="paper", 
    yref="paper", 
    showarrow=False,
    text="<span style='font-size:16px;'><b><i>Findings</b></i>: <i>The observed relationship between unemployment and Sales indicates that lower Unemployment levels correspond to higher Sales figures,<br>which aligns with expectation.</i>")


# <p style="font-size:25px; font-family:cabin; line-height: 1.7em; margin-left:20px">
# <b> Store Sizes, Types & Sales
#          </b></p>

# In[22]:


sizes = train_plt.groupby('Size').mean()
fig = px.line(sizes, x = sizes.index, y = sizes.Weekly_Sales, template='simple_white',
              labels={'Weekly_Sales' : 'Total Sales', 'Size' : 'Store Size'})

fig.update_layout(
    template=template, 
    title={'text':'<b>Sales across different Store sizes</b>', 'x': 0.075},
    yaxis=dict(showline=True))

fig.add_annotation(
    x=0, y=-0.2, 
    align='left', 
    font=dict(size=12),
    textangle=0, 
    xref="paper", 
    yref="paper", 
    showarrow=False,
    text="<span style='font-size:16px;'><b><i>Findings</b></i>: <i>Store Size can significantly impact Sales, as evidenced by the data presented here.</i>", )


# In[23]:


store_type = pd.concat([stores['Type'], stores['Size']], axis=1)

fig = px.box(store_type, x='Type', y='Size', color='Type', 
             title='Store size and Store type',
             color_discrete_sequence=palletes['continuos']['blues'])

fig.update_layout(
    template=template, 
    title={'text':'<b>Store Size and Store Type</b>', 'x': 0.075},
    yaxis_title='Size', 
    xaxis_title='Type',
    yaxis=dict(showline=True))

fig.add_annotation(
    x=0, y=-0.225, 
    align='left', 
    font=dict(size=12),
    textangle=0, 
    xref="paper", 
    yref="paper", 
    showarrow=False,
    text="<span style='font-size:16px;'><b><i>Findings</b></i>: <i>In terms of Size, three distinct store types emerge, with Type A stores being the most prevalent.</i>")


# In[24]:


store_sale = pd.concat([stores['Type'], train_df['Weekly_Sales']], axis=1)

fig = px.box(store_sale.dropna(), x='Type', y='Weekly_Sales', color='Type', 
             color_discrete_sequence=palletes['continuos']['blues'])

fig.update_layout(
    template=template, 
    title={'text':'<b>Store Type and Sales</b>', 'x': 0.075},
    yaxis_title='Total Sales', 
    xaxis_title='Type',
    yaxis=dict(showline=True))

fig.add_annotation(
    x=0, y=-0.2, 
    align='left', 
    font=dict(size=12),
    textangle=0, 
    xref="paper", 
    yref="paper", 
    showarrow=False,
    text="<span style='font-size:16px;'><b><i>Findings</b></i>: <i>Despite being the least common store type, Store Type C exhibits remarkable Sales performance, with a median sale exceeding the other two store types.</i>")


# <p style="font-size:25px; font-family:cabin; line-height: 1.7em; margin-left:20px">
# <b> Departaments
#          </b></p>

# In[25]:


depts = train_plt.groupby('Dept').mean().sort_values(by='Weekly_Sales', ascending=False)

fig = px.bar(depts, x=depts.index, y=depts.Weekly_Sales, color=depts.Weekly_Sales, 
             color_continuous_scale=palletes['continuos']['green_n_blues'])

fig.update_layout(
    template=template, 
    title={'text':'<b>Sales across Departaments</b>', 'x': 0.075},
    legend_title_text='<b>Sales</b>',
    yaxis=dict(showline=True))

fig.add_annotation(
    x=0, y=-0.20, 
    align='left', 
    font=dict(size=12),
    textangle=0, 
    xref="paper", 
    yref="paper", 
    showarrow=False,
    text="<span style='font-size:16px;'><b><i>Findings</b></i>: <i>Sales distribution across departments is uneven, with some departments generating a disproportionately high share of total sales.</i>")


# <a id="subsection-three-three"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#4AC9FE;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%;margin-left:10px"> 3.3 | Heatmap and correlation between features</div>

# In[26]:


corr = train_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
df_mask = corr.mask(mask).round(2)

fig = ff.create_annotated_heatmap(z=df_mask.to_numpy(), 
                                  x=df_mask.columns.tolist(),
                                  y=df_mask.columns.tolist(),
                                  colorscale=palletes['continuos']['green_n_blues'],
                                  hoverinfo="none", 
                                  showscale=True, ygap=1, xgap=1)

fig.update_xaxes(side="bottom")

fig.update_layout(
    template=template, 
    width=900, 
    height=700,
    margin=dict(l=100),
    title={'text':'<b>Feature correlation (Heatmap)</b>', 'x': 0.075},
    title_x=0.5, 
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    xaxis_zeroline=False,
    yaxis_zeroline=False,
    yaxis_autorange='reversed',
)

for i in range(len(fig.layout.annotations)):
    if fig.layout.annotations[i].text == 'nan':
        fig.layout.annotations[i].text = ""

fig.show()


# In[27]:


weekly_sales_corr = train_df.corr().iloc[2,:]
corr_df = pd.DataFrame(data = weekly_sales_corr, index = weekly_sales_corr.index ).sort_values (by='Weekly_Sales', ascending=False)
corr_df = corr_df.iloc[1:]

fig = px.bar(corr_df, x=corr_df.index, y='Weekly_Sales', color=corr_df.index, labels={'index':'Features'},
             color_discrete_sequence=palletes['continuos']['green_n_blues'])

fig.update_traces(showlegend=False)

fig.update_layout(
    template=template, 
    title={'text':'<b>Features and his correlation with Sales</b>', 'x': 0.075},
    yaxis_title='Sales Increase',
    yaxis=dict(showline=True))


# <a id="section-four"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%;margin-left:0px"> 4 | Feature engineering </div>

# In[28]:


data_train = train_df.copy()
data_test = test_df.copy()


# <a id="subsection-four-one"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#4AC9FE;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%;margin-left:10px"> 4.1 | Holidays</div>
# 
# <p style="line-height: 100%; margin-left:30px; font-size:22.5px; font-family:cabin;">
# Since Thanksgiving and Christmas are the most importarnt holidays, I'm going to try some feature engineering on this features, and also Superbowl and Laborday.
#     <br><br>

# In[29]:


data_train['Days_to_Thansksgiving'] = (pd.to_datetime(train_df["Year"].astype(str)+"-11-24", format="%Y-%m-%d") - pd.to_datetime(train_df["Date"], format="%Y-%m-%d")).dt.days.astype(int)
data_train['Days_to_Christmas'] = (pd.to_datetime(train_df["Year"].astype(str)+"-12-24", format="%Y-%m-%d") - pd.to_datetime(train_df["Date"], format="%Y-%m-%d")).dt.days.astype(int)


# In[30]:


data_test['Days_to_Thansksgiving'] = (pd.to_datetime(test_df["Year"].astype(str)+"-11-24", format="%Y-%m-%d") - pd.to_datetime(test_df["Date"], format="%Y-%m-%d")).dt.days.astype(int)
data_test['Days_to_Christmas'] = (pd.to_datetime(test_df["Year"].astype(str)+"-12-24", format="%Y-%m-%d") - pd.to_datetime(test_df["Date"], format="%Y-%m-%d")).dt.days.astype(int)


# In[31]:


data_train['SuperBowlWeek'] = train_df['Week'].apply(lambda x: 1 if x == 6 else 0)
data_train['LaborDay'] = train_df['Week'].apply(lambda x: 1 if x == 36 else 0)
data_train['Tranksgiving'] = train_df['Week'].apply(lambda x: 1 if x == 47 else 0)
data_train['Christmas'] = train_df['Week'].apply(lambda x: 1 if x == 52 else 0)


# In[32]:


data_test['SuperBowlWeek'] = test_df['Week'].apply(lambda x: 1 if x == 6 else 0)
data_test['LaborDay'] = test_df['Week'].apply(lambda x: 1 if x == 36 else 0)
data_test['Tranksgiving'] = test_df['Week'].apply(lambda x: 1 if x == 47 else 0)
data_test['Christmas'] = test_df['Week'].apply(lambda x: 1 if x == 52 else 0)


# <a id="subsection-four-two"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#4AC9FE;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%;margin-left:10px"> 4.2 | Markdowns</div>

# In[33]:


data_train['MarkdownsSum'] = train_df['MarkDown1'] + train_df['MarkDown2'] + train_df['MarkDown3'] + train_df['MarkDown4'] + train_df['MarkDown5'] 


# In[34]:


data_test['MarkdownsSum'] = test_df['MarkDown1'] + test_df['MarkDown2'] + test_df['MarkDown3'] + test_df['MarkDown4'] + test_df['MarkDown5']


# <a id="section-five"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%;margin-left:0px"> 5 | Preprocessing</div>
# 
# <a id="subsection-five-one"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#4AC9FE;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%;margin-left:10px"> 5.1 | Filling missing values</div>

# In[35]:


data_train.isna().sum()[data_train.isna().sum() > 0].sort_values(ascending=False)


# In[36]:


data_test.isna().sum()[data_test.isna().sum() > 0].sort_values(ascending=False)


# In[37]:


data_train.fillna(0, inplace = True)


# In[38]:


data_test['CPI'].fillna(data_test['CPI'].mean(), inplace = True)
data_test['Unemployment'].fillna(data_test['Unemployment'].mean(), inplace = True)


# In[39]:


data_test.fillna(0, inplace = True)


# <a id="subsection-five-two"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#4AC9FE;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%;margin-left:10px"> 5.2 | Encoding categorical data</div>

# In[40]:


data_train['IsHoliday'] = data_train['IsHoliday'].apply(lambda x: 1 if x == True else 0)
data_test['IsHoliday'] = data_test['IsHoliday'].apply(lambda x: 1 if x == True else 0)


# In[41]:


data_train['Type'] = data_train['Type'].apply(lambda x: 1 if x == 'A' else (2 if x == 'B' else 3))
data_test['Type'] = data_test['Type'].apply(lambda x: 1 if x == 'A' else (2 if x == 'B' else 3))


# <a id="section-six"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%;margin-left:0px"> 6 | Feature selection</div>

# In[42]:


features = [feature for feature in data_train.columns if feature not in ('Date','Weekly_Sales')]


# In[43]:


X = data_train[features].copy()
y = data_train.Weekly_Sales.copy()


# In[44]:


data_sample = data_train.copy().sample(frac=.25)
X_sample = data_sample[features].copy()
y_sample = data_sample.Weekly_Sales.copy()


# In[45]:


X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X_sample, y_sample, random_state=0, test_size=0.15)


# In[46]:


feat_model = xgb.XGBRegressor(random_state=0).fit(X_train, y_train)


# In[47]:


perm = PermutationImportance(feat_model, random_state=1).fit(X_valid, y_valid)
features = eli5.show_weights(perm, top=len(X_train.columns), feature_names = X_valid.columns.tolist())


# In[48]:


features_weights = eli5.show_weights(perm, top=len(X_train.columns), feature_names = X_valid.columns.tolist())
features_weights


# In[49]:


f_importances = pd.Series(dict(zip(X_valid.columns.tolist(), perm.feature_importances_))).sort_values(ascending=False)


# In[50]:


weights = eli5.show_weights(perm, top=len(X_train.columns), feature_names=X_valid.columns.tolist())
result = pd.read_html(weights.data)[0]
result


# <div class="alert alert-info" style="border-radius:10px; line-height: 100%; margin-left:20px; font-size:20px; font-family:cabin;">
# Seems to be Dept, Store, Size, CPI, Week, are the top 5 features.
# </div>

# <a id="section-seven"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%;margin-left:0px"> 7 | Modeling</div>

# In[51]:


# Eval metric for the competition
def WMAE(dataset, real, predicted):
    weights = dataset.IsHoliday.apply(lambda x: 5 if x else 1)
    return np.round(np.sum(weights*abs(real-predicted))/(np.sum(weights)), 2)


# In[52]:


models = {
          '    LGBM': lgb.LGBMRegressor(random_state = 0),
          ' XGBoost': xgb.XGBRegressor(random_state = 0, objective = 'reg:squarederror'),
          'Catboost': cb.CatBoostRegressor(random_state = 0, verbose=False),          
          '    HGBR': HistGradientBoostingRegressor(random_state = 0),
          ' ExtraTr': ensemble.ExtraTreesRegressor(bootstrap = True, random_state = 0),
          ' RandomF': ensemble.RandomForestRegressor(random_state = 0),
         }


# In[53]:


def model_evaluation (name, model, models, X_train, y_train, X_valid, y_valid):
   
    rmses = []
    
    for i in range(len(models)):
    
        # Model fit
        model.fit(X_train, y_train)
        
        # Model predict
        y_preds = model.predict(X_valid)

        # RMSE
        rmse = np.sqrt(np.mean((y_valid - y_preds)**2))
        rmses.append(rmse)
        
    return np.mean(rmses)


# In[54]:


for name, model in models.items():
    print(name + ' Valid RMSE {:.4f}'.format(model_evaluation(name, model, models, X_train, y_train, X_valid, y_valid)) )


# <div class="alert alert-info" style="border-radius:10px; line-height: 100%; margin-left:20px; font-size:20px; font-family:cabin;">
# Seems to be RandomForest it's the best baseline model by default, followed by ExtraTrees, but we can improve the score of boosting models by doing hyperparameter optimization. Also, for a more generalizable model you can do a blend of the best models at the end.
# </div>

# <a id="subsection-seven-one"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#4AC9FE;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%;margin-left:10px"> 7.1 | Baseline predictions</div>
# 
# <p style="line-height: 100%; margin-left:20px; font-size:22.5px; font-family:cabin;">
# Establishing a baseline with the best model. </p>
# <br>

# In[55]:


X_baseline = X[['Store','Dept','IsHoliday','Size','Week','Type','Year','Day']].copy()


# In[56]:


X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X_baseline, y, random_state=0, test_size=0.1)


# In[57]:


RF = ensemble.RandomForestRegressor(n_estimators=60, max_depth=25, min_samples_split=3, min_samples_leaf=1)
RF.fit(X_train, y_train)


# In[58]:


test = data_test[['Store','Dept','IsHoliday','Size','Week','Type','Year','Day']].copy()
predict_rf = RF.predict(test)


# In[59]:


sample_submission['Weekly_Sales'] = predict_rf
sample_submission.to_csv('submission.csv',index=False)


# ![rf baseline.jpg](attachment:101c4a8c-8c07-4c1b-8c0d-dfcd698931ce.jpg)

# <a id="subsection-seven-two"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#4AC9FE;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%;margin-left:10px"> 7.2 | Simple Blend </div>
# 
# <p style="line-height: 100%; margin-left:30px; font-size:22.5px; font-family:cabin;">
# Blend baseline with the two best model Random Forest and Extra Trees.
# </p><br>

# In[60]:


ETR = ensemble.ExtraTreesRegressor(n_estimators=50, bootstrap = True, random_state = 0)
ETR.fit(X_train, y_train)


# In[61]:


predict_etr = ETR.predict(test)


# In[62]:


avg_preds = (predict_rf + predict_etr) / 2


# In[63]:


sample_submission['Weekly_Sales'] = avg_preds
sample_submission.to_csv('submission.csv',index=False)


# ![blend baseline.jpg](attachment:bc7b59b8-a926-4943-a8f0-3176ed6ebe6a.jpg)

# <p style="font-size:22.5px; font-family:cabin; line-height: 1.7em; margin-left:30px">
# Thanks for taking the time to read my notebook </p>
