#!/usr/bin/env python
# coding: utf-8

# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/34377/logos/header.png?t=2022-02-11-21-53-06")
# 
# [Image credit: kaggle_comp_spaceship-titanic](https://www.kaggle.com/c/spaceship-titanic/overview)
# 
# ---
# 
# <blockquote style="margin-right:auto; margin-left:auto; color:white; background-color: lightseagreen; padding: 1em; margin:24px;">
# 
# <font color="white" size=+3.0><b> <u> Spaceship Titanic </u> </b></font>  
#         
# <li><font color="white" size=+2.0><b>Problem Statement</b></font> 
#     
# <ul> The year is <strong>2912</strong>. We've received a transmission from four lightyears away and things aren't looking good.
# The Spaceship Titanic was an interstellar passenger liner launched a month ago. With almost 13,000 passengers on board, the vessel set out on its maiden voyage transporting emigrants from our solar system to three newly habitable exoplanets orbiting nearby stars.
#     
# While rounding Alpha Centauri en route to its first destination—the torrid 55 Cancri E—the unwary Spaceship Titanic collided with a spacetime anomaly hidden within a dust cloud. Sadly, it met a similar fate as its namesake from 1000 years before. Though the ship stayed intact, almost half of the passengers were transported to an alternate dimension! </ul> 
# 
#     
# <li><font color="white" size=+2.0><b>Task</b></font> 
#     
# <ul> Our task is to predict whether a passenger was transported to an alternate dimension during the Spaceship Titanic's collision with the spacetime anomaly. To help us make these predictions, we're given a set of personal records recovered from the ship's damaged computer system. </ul>
#     
# <li><font color="white" size=+2.0><b>Evaluation Metric</b></font>
#     
# <ul> Submissions are evaluated based on their classification accuracy, the percentage of predicted labels that are correct. </ul>
#         
#        
# </blockquote>
#     
# <blockquote style="margin-right:auto; margin-left:auto; color:black; background-color: lightgray; padding: 1em; margin:24px;">
# 
# <li><font color="black" size=+2.0><b>Approach</b></font>  
#         
# <ul>First, I'll perform an EDA (exploratory data analysis) to explore and study our dataset using the awesome python data visualization library called plotly. Then following the insight/observations of EDA I'll try to derive some useful features which I'll use in our models. Last but not least I will build a model to predict and classify who will be transported or not to the space. </ul>                                                                                                                               
#        
# </blockquote>
# 
# ---
# ## **Table of Contents**
# <a id="top"></a>
# * [1. Data Overview](#1)
#     * [1.1 Data Shape](#1.1)
#     * [1.2 Head of the dataFrames](#1.2)
#     * [1.3 Columns datatype](#1.3)
#     * [1.4 Missing Values](#1.4)
# * [2. Features & Correlations](#2)
#     * [2.1 Target variable distribution](#2.1)
#     * [2.2 Passengers Age](#2.2)
#     * [2.3 Home Planet](#2.3)
#     * [2.4 CryoSleep](#2.4)
#     * [2.5 VIP](#2.5)
#     * [2.6 Destination](#2.6)
#     * [2.7 Cabin](#2.7)
#     * [2.8 PassangerId](#2.8)
#     * [2.9 RoomService, FoodCourt, ShoppingMall, Spa and VRDeck](#2.9)
#     * [2.10 Correlation](#2.10)
# * [3. Feature Engineering](#3)
#     * [3.1 Data Pre-Processing](#3.1)
#     * [3.2 Feature Engineering](#3.2)
# * [4. Modeling and Prediction](#4)
#     * [4.1 AutoML_H20](#4.1)
#     * [4.2 Gradient Boosting Machines](#4.2)
#         * [4.2.1 LGBM](#4.2.1)
#             * [4.2.1.1 Hyperparameter optimization (optuna)](#4.2.1.1)
#         * [4.2.2 Xgboost](#4.2.2)
#             * [4.2.2.1 Hyperparameter optimization (optuna)](#4.2.2.1)
#     * [4.3 Model Explainablity](#4.3)
# * [5. Reference](#5)   
#     
# ---
# 
# 

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import torch
import seaborn as sns
from termcolor import colored

import plotly.io as pio
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from colorama import Fore, Back, Style
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold

from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
pio.templates.default = "none"

import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## 1. Data Overview <a class="anchor" id="1"></a>

# In[2]:


train = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
test = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')
submission = pd.read_csv('/kaggle/input/spaceship-titanic/sample_submission.csv')


# ### 1.1 Data shape <a class="anchor" id="1.1"></a>
# * The train dataset has 8693 rows and 22 columns inclusing the target column.
# * Test dataset has 4277 rows and 21 columns.

# In[3]:


print(colored(f'Shape of the train dataset is', 'blue'), colored(train.shape, 'blue'))
print(colored(f'Shape of the test dataset is', 'yellow'), colored(test.shape, 'yellow'))


# ### 1.2 Head of the dataFrames <a class="anchor" id="1.2"></a>

# In[4]:


train.head()


# ### 1.3 Columns datatype <a class="anchor" id="1.3"></a>

# In[5]:


object_cols = [col for col in train.columns if train[col].dtype =='object']
numerical_cols = [col for col in train.columns if train[col].dtype !='object']

print(f'Numerical columns in the datasets are: ',colored(numerical_cols, 'blue'))
print(f'Object columns in the datasets are: ',colored(object_cols, 'yellow'))


# ### 1.4 Missing values <a class="anchor" id="1.4"></a>
# 
# - Except columns `PassengerID` and the target variable `Transported`, all columns have missing values in them.
# - Missing values range from 1.87 to around 2.5% of rows.
# - Missing values in train and test datasets are fairly consistent.
# 
# <a href="#top">Back to top</a>     

# In[6]:


def null_value_df(data):    
    null_values_df = []    
    for col in data.columns:
        
        if data[col].isna().sum() != 0:
            pct_na = np.round((100 * (data[col].isna().sum())/len(data)), 2) 
            
            dict1 ={
                'Features' : col,
                'NA (count)': data[col].isna().sum(),
                'NA (%)': (pct_na)
            }
            null_values_df.append(dict1)
    return pd.DataFrame(null_values_df, index=None).sort_values(by='NA (count)',ascending=False)


DF1 = null_value_df(train)
DF2 = null_value_df(test)

fig = go.Figure(data=[go.Bar(x=DF1['Features'],
                             y=DF1["NA (%)"],                              
                             name='train', marker_color='lightseagreen'),
                      go.Bar(x=DF2['Features'],
                             y=DF2["NA (%)"], 
                             name='test', marker_color='lightgray')])
fig.update_layout(title_text='<b> Features with missing values: train & test data <b>',
                  font_family="San Serif",
                  template='simple_white',
                  width=750, height=500,
                  xaxis_title='Features', 
                  yaxis_title='Missing Values (%)',
                  titlefont={'color':'black', 'size': 24, 'family': 'San-Serif'})
fig.update_yaxes(showgrid=False, showline=False, showticklabels=True)
fig.update_xaxes(showgrid=False, showline=True, showticklabels=True)
fig.show()


# #### Do the missing values have some correlation with target?
# - Looking at the charts plotted below, only missing values in columns `RoomService`, `FoodCourt` and `ShoppingMall` seem to have little correlations. Around 54% of Passengers whose `RoomService` is NA did not transport. Whereas around 54% of the passengers whose `FoodCourt` is NA and aound 55% with `ShoppingMall` NA columns did transport. For the rest it is close call.
# - With only around 2.5% of missing data in each column, trying to create new feature based on the missging values is may be hopefull thinking than substatiative expectations. Neverthless, we will try to create new columns and see if they help boost the score. 

# In[7]:


na_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'Cabin', 'VRDeck', 
          'HomePlanet', 'CryoSleep',  'Destination', 'VIP', 'Name']


fig = make_subplots(
    rows=4, cols=3,
    subplot_titles=('Age', 'Spa', 'CryoSleep','RoomService','Cabin','Destination',             
                    'FoodCourt', 'VRDeck', 'VIP','ShoppingMall','HomePlanet', 'Name')
)

for i, col in enumerate(na_cols[0:4]):
    df_sf = train[train[col].isna()]

    neg = df_sf[df_sf['Transported'] == False]
    pos = df_sf[df_sf['Transported'] == True]

    label = ['Transported', 'Not_transported']
    value = [pos.shape[0], neg.shape[0]] 
    pct = [value[0]*100/len(df_sf), value[1]*100/len(df_sf)]

    fig.add_trace(go.Bar(
                y=value, x=label,
                name=col,
                text=(np.round(pct,1)),
                textposition=['inside', 'inside'],
                texttemplate = ["<b style='color: #f'>%{text}%</b>"]*2,
                textfont=dict(  family="sans serif",
                                size=16,
                                color="black"),
                orientation='v',
                marker_color=['lightseagreen', 'lightgray'],
                opacity=1.0,
                        ),
                         row=i+1, col=1                   
                   )
    
for j, col in enumerate(na_cols[4:8]):
    df_sf = train[train[col].isna()]

    neg = df_sf[df_sf['Transported'] == False]
    pos = df_sf[df_sf['Transported'] == True]

    label = ['Transported', 'Not_transported']
    value = [pos.shape[0], neg.shape[0]] 
    pct = [value[0]*100/len(df_sf), value[1]*100/len(df_sf)]

    fig.add_trace(go.Bar(
                y=value, x=label,
                name=f'</b> col',
                text=(np.round(pct,1)),
                textposition=['inside', 'inside'],
                texttemplate = ["<b style='color: #f'>%{text}%</b>"]*2,
                textfont=dict(  family="sans serif",
                                size=16,
                                color="black"),
                orientation='v',
                marker_color=['lightseagreen', 'lightgray'],
                opacity=1.0,
                        ),
                         row=j+1, col=2                   
                   )
for k, col in enumerate(na_cols[8:]):
    df_sf = train[train[col].isna()]

    neg = df_sf[df_sf['Transported'] == False]
    pos = df_sf[df_sf['Transported'] == True]

    label = ['Transported', 'Not_transported']
    value = [pos.shape[0], neg.shape[0]] 
    pct = [value[0]*100/len(df_sf), value[1]*100/len(df_sf)]

    fig.add_trace(go.Bar(
                y=value, x=label,
                name=col,
                text=(np.round(pct,1)),
                textposition=['inside', 'inside'],
                texttemplate = ["<b style='color: #f'>%{text}%</b>"]*2,
                textfont=dict(  family="sans serif",
                                size=16,
                                color="black"),
                orientation='v',
                marker_color=['lightseagreen', 'lightgray'],
                opacity=1.0,
                        ),
                         row=k+1, col=3                   
                   )
fig.update_layout(title='<b> Target Distribution of Missing Values', 
                  font_family="San Serif",
                  yaxis_linewidth=2.5,
                  width=900, 
                  height=1000,
                  bargap=0.2,
                  barmode='group',
                  titlefont={'size': 24},
                  showlegend=False
                  )
fig.update_xaxes(showgrid=False, showline=True)
fig.update_yaxes(showgrid=False, showline=False, showticklabels=False)
fig.show() 


# ## 2. Features & Correlation <a class="anchor" id="2"></a>
# Now let's explore individual features

# ### 2.1 Target variable distribution <a class="anchor" id="2.1"></a>
# 
# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:24px;">
#     <strong>Transported</strong>: Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.
# </blockquote>
# 
# **Observations**: 
# 
# - Target distribution is balanced!

# In[8]:


neg = train[train['Transported'] == False]
pos = train[train['Transported'] == True]

label = ['True', 'False']
value = [pos.shape[0], neg.shape[0]] 
pct = [value[0]*100/len(train), value[1]*100/len(train)]


fig = go.Figure(data=[go.Bar(
            y=value, x=label,
            text=(np.round(pct,2)),
            textposition=['inside', 'inside'],
            texttemplate = ["<b style='color: #f'>%{text}%</b>"]*2,
            textfont=dict(  family="sans serif",
                            size=16,
                            color="black"),
            orientation='v',
            marker_color=['lightseagreen', 'lightgray'],
            opacity=1.0,
                    )])
fig.update_layout(title='<b>Target: Transported (False/True) <b>', 
                  font_family="San Serif",
                  xaxis_title='Target (Transported)',
                  yaxis_linewidth=2.5,
                  width=550, 
                  height=400,
                  bargap=0.2,
                  barmode='group',
                  titlefont={'size': 24},
                  )
fig.update_xaxes(showgrid=False, showline=True)
fig.update_yaxes(showgrid=False, showline=False, showticklabels=False)
fig.show()


# ### Unique Values/ Cardinality 
# - Of all the categorical variables, only `HomePlanet` ,`VIP` `CryoSleep` and `Destination` have low cardinality. `PassengetId` and `Name` have very high cardinality.
# 
# #### Low cardinal features distribution (train/test)
# Below the distribution of the low_cardinal features for both datasets are shown.
# > Train and test datasets have *similar* distributions
# - `HomePalnet`: Passengers from Earth are the majority.
# - `CryoSleep`: Majority of the passengers did not pay for CryoSleep.
# - `Destination`: TRAPPIST-1e is the favorite destination with around 70% of the passengers booked for it.
# - `VIP`: VIP passengers amount to around 2%
# 
# <a href="#top">Back to top</a>     

# In[9]:


fig = make_subplots(rows=4, cols=2,
                    specs=[[{'type':'domain'}, {'type':'domain'}],
                           [{'type':'domain'}, {'type':'domain'}], 
                           [{'type':'domain'}, {'type':'domain'}], 
                           [{'type':'domain'}, {'type':'domain'}], 
                           ])
fig.add_trace(
    go.Pie(
        labels=train['HomePlanet'],
        values=None,scalegroup='one',
        hole=.4,
        title='HomePlanet (train)',
        titlefont={'size': 24},         

        ),
    row=1,col=1
    )
fig.update_traces(
    hoverinfo='label+value',
    textinfo='label+percent',
    textfont_size=12,
#     marker=dict(
#         colors=colors0, 
#         line=dict(color='#000000',
#                   width=2)
#         )
    )

fig.add_trace(
    go.Pie(
        labels=test['HomePlanet'],
        values=None,#scalegroup='one',
        hole=.4,
        title='HomePlanet (test)',
        titlefont={'size': 24},
        ),
    row=1,col=2
    )
fig.update_traces(
    hoverinfo='label+value',
    textinfo='label+percent',
    textfont_size=12,
#     marker=dict(
#         colors=colors0,
#         line=dict(color='#000000',
#                   width=2)
#         )
    )

fig.add_trace(
    go.Pie(
        labels=train['CryoSleep'],
        values=None,#scalegroup='one',
        hole=.4,
        title='CryoSleep (train)',
        titlefont={'size': 24},
        ),
    row=2,col=1
    )
fig.update_traces(
    hoverinfo='label+value',
    textinfo='label+percent',
    textfont_size=12,
#     marker=dict(
#         colors=colors1,
#         line=dict(color='#000000',
#                   width=2)
#         )
    )

fig.add_trace(
    go.Pie(
        labels=test['CryoSleep'],
        values=None,#scalegroup='one',
        hole=.4,
        title='CryoSleep (test)',
        titlefont={'size': 24},
        ),
    row=2,col=2
    )
fig.update_traces(
    hoverinfo='label+value',
    textinfo='label+percent',
    textfont_size=12,
#     marker=dict(
#         colors=colors1,
#         line=dict(color='#000000',
#                   width=2)
#         )
    )

fig.add_trace(
    go.Pie(
        labels=train['Destination'],
        values=None,#scalegroup='one',
        hole=.4,
        title='Destination (train)',
        titlefont={'size': 24},
       ),
    row=3,col=1
    )
fig.update_traces(
    hoverinfo='label+value',
    textinfo='label+percent',
    textfont_size=12,
#     marker=dict(
#         colors=colors2,
#         line=dict(color='#000000',
#                   width=2)
#         )
    )

fig.add_trace(
    go.Pie(
        labels=test['Destination'],
        values=None,#scalegroup='one',
        hole=.4,
        title='Destination (test)',
        titlefont={'size': 24},
       ),
    row=3,col=2
    )
fig.update_traces(
    hoverinfo='label+value',
    textinfo='label+percent',
    textfont_size=12,
#     marker=dict(
#         colors=colors2,
#         line=dict(color='#000000',
#                   width=2)
#         )
    )

fig.add_trace(      
    go.Pie(
        labels=train['VIP'],
        values=None,#scalegroup='one',
        hole=.4,
        title='VIP(train)',
        titlefont={'size': 24},
       ),
    row=4,col=1
    )
fig.update_traces(
    hoverinfo='label+value',
    textinfo='label+percent',
    textfont_size=12,
#     marker=dict(
#         colors=colors3,
#         line=dict(color='#000000',
#                   width=2)
#         )
    )

fig.add_trace(
    go.Pie(
        labels=test['VIP'],
        values=None,#scalegroup='one',
        hole=.4,
        title='VIP(test)',
        titlefont={'size': 24},
#         marker=dict(
#         colors=colors3,
#         line=dict(color='#000000',
#                   width=2)
#         )
       ),
    row=4,col=2
    )
fig.update_traces(
    hoverinfo='label+value',
    textinfo='label+percent',
    textfont_size=12,
    )
fig.layout.update(title="Low cardinal features Distribution (train/test)", showlegend=False, height=1200, width=1000, 
                  titlefont={'size': 24, 'family': 'San-Serif'}
                 )

fig.show()


# ### 2.2 Passengers Age <a class="anchor" id="2.2"></a>
# 
# 
# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:24px;">
#     <strong>Age</strong>: The age of the passenger
# </blockquote>
# 
# **Observations**: 
# 
# 
# - Train and test dataset have similar age distribution.
# - The youngest, the oldest and the average passenger ages are 0, 79 and ~29 years repectively. 
# - Passengers aged less than 5 years seem to have a better chance of being transported.
# 
# 

# In[10]:


print('Train data Age stats')
print('Minumum Age in {}' .format(train['Age'].min()))
print('Maximum Age in {}' .format(train['Age'].max()))
print('Average Age in {}' .format(train['Age'].mean()))

print(' ')
print('Test data Age stats')
print('Minumum Age in {}' .format(test['Age'].min()))
print('Maximum Age in {}' .format(test['Age'].max()))
print('Average Age in {}' .format(test['Age'].mean()))


# In[11]:


fig = go.Figure()

fig.add_trace(go.Histogram(x=train['Age'],
                           name='train', 
                           histnorm='percent',
                           xbins=dict(
                               start=0,
                               end=100,
                               size=2
                           ),
                           marker_color='lightseagreen',
                           
                          )
             ) 
fig.add_trace(go.Histogram(x=test['Age'],
                           name='test', 
                           histnorm='percent',
                           xbins=dict(
                               start=0,
                               end=100,
                               size=2
                           ),
                           marker_color='lightgray',
                           
                          )
             ) 
fig.update_layout(title='Passengers Age Distribution (train-test)',
                  xaxis_title='Age [years]', 
                  yaxis_title='Percent (%)',
                  titlefont={'size': 24},
                  font_family = 'San Serif',
                  width=700,height=400,
                  #template="plotly_dark",
                  showlegend=True,
)
fig.update_yaxes(showgrid=False, showline=False, showticklabels=True)
fig.show()


# In[12]:


trans_y = train[train['Transported'] == True]['Age']
trans_n = train[train['Transported'] == False]['Age']

fig = go.Figure()

fig.add_trace(go.Histogram(x=trans_y,
                           name='Transported-(True)', 
                           histnorm='percent',
                           xbins=dict(
                               start=0,
                               end=100,
                               size=20
                           ),
                           marker_color='crimson',
                           
                          )
             ) 

fig.add_trace(go.Histogram(x=trans_n,
                           name='Transported-(No)', 
                           histnorm='percent',
                           xbins=dict(
                               start=0,
                               end=100,
                               size=20
                           ),
                           marker_color='gray',
                           
                          )
             ) 

fig.update_layout(title='Passengers Age Distribution (target_based)',
                  xaxis_title='Age [years]', 
                  yaxis_title='Percent (%)',
                  titlefont={'size': 24},
                  font_family = 'San Serif',
                  width=700,height=400,
                  #template="plotly_dark",
                  showlegend=True,
)
fig.update_yaxes(showgrid=False, showline=False, showticklabels=True)
fig.show()


# ### 2.3 Home Planet <a class="anchor" id="2.3"></a>
# 
# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:24px;">
#     <strong>HomePlanet</strong>: The planet the passenger departed from, typically their planet of permanent residence.
# </blockquote>
# 
# **Observations**: 
# 
# - Passengers from `Europa` has more chance of being transported
# - While those from `Earth` had lesser chance of being transported
# - Mars passengers have close to 50-50% chance of being transported
# 
# <a href="#top">Back to top</a>     

# In[13]:


trans_y = train[train['Transported'] == True]['HomePlanet']
trans_n = train[train['Transported'] == False]['HomePlanet']

fig = go.Figure()
fig.add_trace(go.Histogram(x=trans_y,histnorm='',
              name='Transported (True)', marker_color = 'crimson'),
              )
fig.add_trace(go.Histogram(x=trans_n,histnorm='',
              name='Transported (False)', marker_color = 'gray', opacity=0.85),
             )  

fig.update_layout(title="Home Planet", 
                  font_family="San Serif",
                  titlefont={'size': 20},
                  template='simple_white',
                  xaxis_title='Home Planet',
                  width=600, 
                  height=400,
                  legend=dict(
                  orientation="v", y=1, yanchor="top", x=1.0, xanchor="right" )                 
                 ).update_xaxes(categoryorder='total descending') 

fig.show()


# ### 2.4 CryoSleep <a class="anchor" id="2.4"></a>
# 
# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:24px;">
#     <strong>CryoSleep</strong>: Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
# </blockquote>
# 
# **Observations**:
# - Opting for the `CryoSleep` gives more chance of being transported
# - 82% (2483/3037) passengers who opted for cyrosleep did transport.
# - 67% (3650/5439) passengers who did not elect the cryosleep **did not** transport.

# In[14]:


trans_y = train[train['Transported'] == True]['CryoSleep']
trans_n = train[train['Transported'] == False]['CryoSleep']

fig = go.Figure()
fig.add_trace(go.Histogram(x=trans_y,histnorm='',
              name='Transported (True)', marker_color = 'crimson'),
              )
fig.add_trace(go.Histogram(x=trans_n,histnorm='',
              name='Transported (False)', marker_color = 'gray', opacity=0.85),
             )  

fig.update_layout(title="CryoSleep", 
                  font_family="San Serif",
                  titlefont={'size': 20},
                  template='simple_white',
                  xaxis_title='Cryo Sleep',
                  width=600, 
                  height=400,
                  legend=dict(
                  orientation="v", y=1, yanchor="top", x=1.0, xanchor="right" )                 
                 ).update_xaxes(categoryorder='total descending') 

fig.show()


# ### 2.5 VIP <a class="anchor" id="2.5"></a>
# 
# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:24px;">
#     <strong>VIP</strong>: Whether the passenger has paid for special VIP service during the voyage.
# </blockquote>
# 
# **Observation**: 
# - Paying for special VIP does little to nothing to increase chances of being transported.

# In[15]:


trans_y = train[train['Transported'] == True]['VIP']
trans_n = train[train['Transported'] == False]['VIP']

fig = go.Figure()
fig.add_trace(go.Histogram(x=trans_y,histnorm='',
              name='Transported (True)', marker_color = 'crimson'),
              )
fig.add_trace(go.Histogram(x=trans_n,histnorm='',
              name='Transported (False)', marker_color = 'gray', opacity=0.85),
             )  

fig.update_layout(title="VIP Passenger", 
                  font_family="San Serif",
                  titlefont={'size': 20},
                  template='simple_white',
                  xaxis_title='VIP - Passenger',
                  width=600, 
                  height=400,
                  legend=dict(
                  orientation="v", y=1, yanchor="top", x=1.0, xanchor="right" )                 
                 ).update_xaxes(categoryorder='total descending') 

fig.show()


# ### 2.6 Destination <a class="anchor" id="2.6"></a>
# 
# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:24px;">
#     <strong>Destination</strong>: The planet the passenger will be debarking to.
# </blockquote>
# 
# **Observation**: 
# - Booking for `55 Cancrie e` has a better chance of being transported 

# In[16]:


trans_y = train[train['Transported'] == True]['Destination']
trans_n = train[train['Transported'] == False]['Destination']

fig = go.Figure()
fig.add_trace(go.Histogram(x=trans_y,histnorm='',
              name='Transported (True)', marker_color = 'crimson'),
              )
fig.add_trace(go.Histogram(x=trans_n,histnorm='',
              name='Transported (False)', marker_color = 'gray', opacity=0.85),
             )  

fig.update_layout(title="Destination", 
                  font_family="San Serif",
                  titlefont={'size': 20},
                  template='simple_white',
                  xaxis_title='Passenger Destination',
                  width=600, 
                  height=400,
                  legend=dict(
                  orientation="v", y=1, yanchor="top", x=1.0, xanchor="right" )                 
                 ).update_xaxes(categoryorder='total descending') 

fig.show()


# ### 2.7. Cabin <a class="anchor" id="2.7"></a>
#  
# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:24px;">
#     <strong>Cabin</strong>: The cabin number where the passenger is staying. Takes the form <strong>deck/num/side</strong>, where side can be either <strong>P</strong> for Port or <strong>S</strong> for Starboard.
# </blockquote>
# 
# - Let's separate deck/num/side of the cabin and make new features.
# 
# **Observations**
# - Most passengers are in Deck F and G; T contains almost no passangers.
# - Deck B and C seem to be favorable to being transported; more than 73% of the passangers in Deck B and around 68% in Deck C did transport.
# - On the other hand, passengers in Decks E and F seem to be have unfavorable chance of being transported.
# 
# <a href="#top">Back to top</a>     

# In[17]:


train[['Deck', 'Num', 'Side']] = train['Cabin'].str.split('/', 2, expand=True)
test[['Deck', 'Num', 'Side']] = test['Cabin'].str.split('/', 2, expand=True)


# In[18]:


trans_y = train[train['Transported'] == True]['Deck']
trans_n = train[train['Transported'] == False]['Deck']

fig = go.Figure()
fig.add_trace(go.Histogram(x=trans_y,histnorm='',
              name='Transported (True)', marker_color = 'crimson'),
              )
fig.add_trace(go.Histogram(x=trans_n,histnorm='',
              name='Transported (False)', marker_color = 'gray', opacity=0.85),
             )  

fig.update_layout(title="Cabin 1st code (Deck)", 
                  font_family="San Serif",
                  titlefont={'size': 20},
                  template='simple_white',
                  xaxis_title='Cabin (Deck)',
                  width=600, 
                  height=400,
                  legend=dict(
                  orientation="v", y=1, yanchor="top", x=1.0, xanchor="right" )                 
                 ).update_xaxes(categoryorder='total descending') 

fig.show()


# In[19]:


trans_y = train[train['Transported'] == True]['Num']
trans_n = train[train['Transported'] == False]['Num']

trans_y = trans_y.fillna(trans_y.mode().iloc[0]).astype(int)
trans_n = trans_n.fillna(trans_n.mode().iloc[0]).astype(int)

fig = go.Figure()


fig.add_trace(go.Violin(x=trans_y.sort_values(axis=0, ascending=True), line_color='crimson', name='Transported (True)',))
fig.add_trace(go.Violin(x=trans_n.sort_values(axis=0, ascending=True), line_color='gray', name= 'Transported (False)', ))


fig.update_traces(orientation='h', side='positive', width=3, points=False, meanline_visible=True,)
fig.update_layout(xaxis_showgrid=True, xaxis_zeroline=False)

fig.update_layout(title='Cabin 2st code (Num)',
                  xaxis_title='Cabin 2nd code (Num)',
                  font_family="San Serif",
                  width=600,height=350,
    template="plotly_dark",
    showlegend=False,
    titlefont={'size': 24},
    paper_bgcolor="black",
    font=dict(
        color ='white', 
    )
 )

fig.show()


# In[20]:


trans_y = train[train['Transported'] == True]['Side']
trans_n = train[train['Transported'] == False]['Side']

fig = go.Figure()
fig.add_trace(go.Histogram(x=trans_y,histnorm='',
              name='Transported (True)', marker_color = 'crimson'),
              )
fig.add_trace(go.Histogram(x=trans_n,histnorm='',
              name='Transported (False)', marker_color = 'gray', opacity=0.85),
             )  

fig.update_layout(title="Cabin 3rd code (Side)", 
                  font_family="San Serif",
                  titlefont={'size': 20},
                  template='simple_white',
                  xaxis_title='Cabin 3rd code (Side)',
                  width=600, 
                  height=400,
                  legend=dict(
                  orientation="v", y=1, yanchor="top", x=1.2, xanchor="right" )                 
                 ).update_xaxes(categoryorder='total descending') 

fig.show()


# ### 2.8. PassengerID <a class="anchor" id="2.8"></a>
# 
# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:24px;">
# <strong>PassengerId</strong>: 
# Is a unique Id for each passenger. Each Id takes the form <strong>gggg_pp</strong> where gggg indicates a group the passenger is travelling with and <strong>pp</strong> is their number within the group. People in a group are often family members, but not always.
# </blockquote>
# 
# **Observations**
# 
# - The first part seem to separate the passenger who transported and those who did not. It looks like most of the transported passengers have lower digits in their ID's.
# - We notice that most of the passengers have `01` in the second part of their ID.
# 
# <a href="#top">Back to top</a>    

# In[21]:


train[['ID_0', 'ID_1']] = train['PassengerId'].str.split('_', 1, expand=True)
test[['ID_0', 'ID_1']] = test['PassengerId'].str.split('_', 1, expand=True)


# In[22]:


trans_y = train[train['Transported'] == True]['ID_0']
trans_n = train[train['Transported'] == False]['ID_0']

trans_y = trans_y.fillna(trans_y.mode().iloc[0]).astype(int)
trans_n = trans_n.fillna(trans_n.mode().iloc[0]).astype(int)

fig = go.Figure()

# fig.add_trace(go.Histogram(x=trans_y,
#                            name='Transported-(True)', 
#                            histnorm='percent',
#                            marker_color='crimson',
                           
#                           )
#              ) 

# fig.add_trace(go.Histogram(x=trans_n,
#                            name='Transported-(No)', 
#                            histnorm='percent',
#                            marker_color='gray',
                           
#                           )
#              ) 


fig.add_trace(go.Violin(x=trans_y, line_color='crimson', name='Transported (True)',))
fig.add_trace(go.Violin(x=trans_n, line_color='gray', name= 'Transported (False)', ))


fig.update_traces(orientation='h', side='positive', width=3, points=False, meanline_visible=True,)
fig.update_layout(xaxis_showgrid=True, xaxis_zeroline=False)

fig.update_layout(title='Passenger ID (1st code)',
                  xaxis_title='Passenger ID 1st code (ID_0)',
                  font_family="San Serif",
                  width=600,height=350,
    template="plotly_dark",
    showlegend=False,
    titlefont={'size': 24},
    paper_bgcolor="black",
    font=dict(
        color ='white', 
    )
 )

fig.show()


# In[23]:


trans_y = train[train['Transported'] == True]['ID_1']
trans_n = train[train['Transported'] == False]['ID_1']

fig = go.Figure()
fig.add_trace(go.Histogram(x=trans_y,histnorm='',
              name='Transported (True)', marker_color = 'crimson'),
              )
fig.add_trace(go.Histogram(x=trans_n,histnorm='',
              name='Transported (False)', marker_color = 'gray', opacity=0.85),
             )  

fig.update_layout(title="Passenger ID (2nd code)", 
                  font_family="San Serif",
                  titlefont={'size': 20},
                  template='simple_white',
                  xaxis_title='Passenger ID 2nd code (ID_1)',
                  width=600, 
                  height=400,
                  legend=dict(
                  orientation="v", y=1, yanchor="top", x=1.2, xanchor="right" )                 
                 ).update_xaxes(categoryorder='total descending') 

fig.show()


# ### 2.9 RoomService, FoodCourt, ShoppingMall, Spa and VRDeck <a class="anchor" id="2.9"></a>
# 
# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:24px;">
# <strong>RoomService, FoodCourt, ShoppingMall, Spa, VRDeck :</strong><br> 
# Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
# </blockquote>
# 
# **Observations**
# - On average passengers who billed more at `VRDeck`, `Spa` and `RoomService` did not transport
# - On the other hand passengers who billed more at `ShoppingMall` and `FoodCourt` did transport
# 

# #### | Room Service

# In[24]:


RS = train.groupby(['Transported']).agg({'RoomService': ['mean', 'max', 'std']}).reset_index()
RS


# In[25]:


trans_y = train[train['Transported'] == True]['RoomService']
trans_n = train[train['Transported'] == False]['RoomService']

fig = go.Figure()

fig.add_trace(go.Box(x=trans_y, line_color='crimson', name='Transported (True)',))
fig.add_trace(go.Box(x=trans_n, line_color='gray', name='Transported (False)',))

fig.update_layout(title='Passengers billed at RoomService',
                  xaxis_title='RoomService', 
                  titlefont={'size': 24},
                  font_family = 'San Serif',
                  width=700,height=400,
                  template="plotly_dark",
                  paper_bgcolor="black",
                  showlegend=False,
)
fig.update_yaxes(showgrid=False, showline=False, showticklabels=True)
fig.show()


# #### | Food Court

# In[26]:


FC = train.groupby(['Transported']).agg({'FoodCourt': ['mean', 'max', 'std']}).reset_index()
FC


# In[27]:


trans_y = train[train['Transported'] == True]['FoodCourt']
trans_n = train[train['Transported'] == False]['FoodCourt']

fig = go.Figure()

fig.add_trace(go.Box(x=trans_y, line_color='crimson', name='Transported (True)',))
fig.add_trace(go.Box(x=trans_n, line_color='gray', name='Transported (False)',))

fig.update_layout(title='Passengers billed at FoodCourt',
                  xaxis_title='Billed @ FoodCourt', 
                  titlefont={'size': 24},
                  font_family = 'San Serif',
                  width=700,height=400,
                  template="plotly_dark",
                  paper_bgcolor="black",
                  showlegend=False,
)
fig.update_yaxes(showgrid=False, showline=False, showticklabels=True)
fig.show()


# <a href="#top">Back to top</a>    
# 
# #### | Shopping Mall

# In[28]:


SM = train.groupby(['Transported']).agg({'ShoppingMall': ['mean', 'max', 'std']}).reset_index()
SM


# In[29]:


trans_y = train[train['Transported'] == True]['ShoppingMall']
trans_n = train[train['Transported'] == False]['ShoppingMall']

fig = go.Figure()

fig.add_trace(go.Box(x=trans_y, line_color='crimson', name='Transported (True)',))
fig.add_trace(go.Box(x=trans_n, line_color='gray', name='Transported (False)',))

fig.update_layout(title='Passengers billed at Shopping Mall',
                  xaxis_title='Billed @ ShoppingMall',
                  titlefont={'size': 24},
                  font_family = 'San Serif',
                  width=700,height=400,
                  template="plotly_dark",
                  paper_bgcolor="black",
                  showlegend=False,
)
fig.update_yaxes(showgrid=False, showline=False, showticklabels=True)
fig.show()


# #### | Spa

# In[30]:


Spa = train.groupby(['Transported']).agg({'Spa': ['mean', 'max', 'std']}).reset_index()
Spa


# In[31]:


trans_y = train[train['Transported'] == True]['Spa']
trans_n = train[train['Transported'] == False]['Spa']

fig = go.Figure()

fig.add_trace(go.Box(x=trans_y, line_color='crimson', name='Transported (True)',))
fig.add_trace(go.Box(x=trans_n, line_color='gray', name='Transported (False)',))

fig.update_layout(title='Passengers billed at Spa',
                  xaxis_title='Billed @ Spa', 
                  titlefont={'size': 24},
                  font_family = 'San Serif',
                  width=700,height=400,
                  template="plotly_dark",
                  paper_bgcolor="black",
                  showlegend=False,
)
fig.update_yaxes(showgrid=False, showline=False, showticklabels=True)
fig.show()


# #### VRDeck

# In[32]:


VRD = train.groupby(['Transported']).agg({'VRDeck': ['mean', 'max', 'std']}).reset_index()
VRD


# In[33]:


trans_y = train[train['Transported'] == True]['VRDeck']
trans_n = train[train['Transported'] == False]['VRDeck']

fig = go.Figure()

fig.add_trace(go.Box(x=trans_y, line_color='crimson', name='Transported (True)',))
fig.add_trace(go.Box(x=trans_n, line_color='gray', name='Transported (False)',))


fig.update_layout(title='Passengers billed at VRDeck',
                  xaxis_title='Billed @ VRDeck', 
                  titlefont={'size': 24},
                  font_family = 'San Serif',
                  width=700,height=400,
                  template="plotly_dark",
                  paper_bgcolor="black",
                  showlegend=False,
)
fig.update_yaxes(showgrid=False, showline=False, showticklabels=True)
fig.show()


# ### 2.10 Correlation <a class="anchor" id="2.10"></a>
# 
# Before we do the correlations, let's first make separate dataframes one with the numerical features and another with the categorical features. We use the pearson's correlation coefficient for the numericals and Cramer's V with the categorical features.
# 
# **Observations** 
# - There seems to be no feature with strong correlation with target.
# - All the `Cabin derivatives` are fairly correlated with `ID_0`.

# In[34]:


train.drop(['PassengerId','Cabin'], axis=1, inplace=True)

corr = train.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
corr = corr.mask(mask)


fig = go.Figure(data= go.Heatmap(z=corr,
                                 x=corr.index.values,
                                 y=corr.columns.values,
                                 colorscale='deep',                                  
                                 )
                )
fig.update_layout(title_text='<b>Correlation Heatmap<b>',
                  font_family="San Serif",
                  title_x=0.5,
                  titlefont={'size': 24},
                  width=750, height=750,
                  xaxis_showgrid=False,
                  xaxis={'side': 'bottom'},
                  yaxis_showgrid=False,
                  yaxis_autorange='reversed',                   
                                    autosize=False,
                  margin=dict(l=150,r=50,b=150,t=70,pad=0),
                  )
fig.show()


# In[35]:


# the cramers_v function is copied from https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9

def cramers_v(x, y): 
    confusion_matrix = pd.crosstab(x,y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


def plot_carmersV_corr(df):
    rows= []
    for x in df:
        col = []
        for y in df :
            cramers =cramers_v(df[x], df[y])
            col.append(round(cramers,2))
        rows.append(col)

    cramers_results = np.array(rows)
    df_corr = pd.DataFrame(cramers_results, columns = df.columns, index = df.columns)

    mask = np.triu(np.ones_like(df_corr, dtype=np.bool))
    df_corr = df_corr.mask(mask)


    fig = go.Figure(data= go.Heatmap(z=df_corr,
                                     x=df_corr.index.values,
                                     y=df_corr.columns.values,
                                     colorscale='deep',                                  
                                     )
                    )
    fig.update_layout(title_text='<b>Correlation Heatmap (Categorical features) <b>',
                      font_family="San Serif",
                      title_x=0.5,
                      titlefont={'size': 20},
                      width=750, height=700,
                      xaxis_showgrid=False,
                      xaxis={'side': 'bottom'},
                      yaxis_showgrid=False,
                      yaxis_autorange='reversed',                   
                                        autosize=False,
                      margin=dict(l=150,r=50,b=150,t=70,pad=0),
                      )
    fig.show()
    
plot_carmersV_corr(train.drop(['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService',
       'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name',], axis=1))


# ## 3. Feature Engineering and Data PreProcessing <a class="anchor" id="3"></a>
# 
# - We have already seen that we can create new features by splitting the `Cabin` and `PassengerID` columns to create extra features. 
# - We can also split the `Name` column to first_name and last_names and create a family feature based on relations (but I will keep this one for later versions, may be)
# - Another possibility is to add new features based on statistics such as `mean`, `max`, and `std` of the numerical features.
# - From section 2.9 we saw that (on average) passengers who billed more at `VRDeck`, `Spa` and `RoomService` did not transport but on the other hand passengers who billed more at `ShoppingMall` and `FoodCourt` did transport. Se we can aggregate these features into two groups and create new features.
# - Other ideas could also include `Age binnig`
# 
# <a href="#top">Back to top</a>   
# 

# ### 3.1 Data Pre-Processing <a class="anchor" id="3.1"></a>
# #### 3.1.1 Missing data imputation

# In[36]:


# load fresh data
train = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
test = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')

# let's drop the `Name` feature 

train.drop(['Name'], axis=1, inplace=True)
test.drop(['Name'], axis=1, inplace=True)


# In[37]:


impute_cols_cat = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP']
impute_cols_num = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

from sklearn.impute import SimpleImputer

imputer_cat = SimpleImputer(strategy="most_frequent" )
imputer_cat.fit(train[impute_cols_cat])
train[impute_cols_cat] = imputer_cat.transform(train[impute_cols_cat])
test[impute_cols_cat] = imputer_cat.transform(test[impute_cols_cat])

imputer_num = SimpleImputer(strategy="mean")
imputer_num.fit(train[impute_cols_num])
train[impute_cols_num] = imputer_num.transform(train[impute_cols_num])
test[impute_cols_num] = imputer_num.transform(test[impute_cols_num])


# #### 3.1.2 Age binning
# - [In section 2.2](#2.2) we have seen that, the Age feature can be binned into four bins (under20, under40, under60 and above60). This is just one example as this can also be binned into more/less bins. It is a bit of trial and error what works best in terms of accuracy of results.

# In[38]:


def make_bins(df):    
    label_names = [0, 1, 2, 3, 4]    
    cut_points = [0, 5, 21, 41, 64, 100]
    df['Binned_Age'] = pd.cut(df["Age"], cut_points, labels=label_names)
    return df


# In[39]:


train = make_bins(train)
test = make_bins(test)
train.drop(['Age'], axis=1, inplace=True)
test.drop(['Age'], axis=1, inplace=True)


# ### 3.2 Feature Engineering <a class="anchor" id="3.2"></a>
# #### 3.2.1 Split `Cabin` and `PassengerId` and Create Extra Features

# In[40]:


# split Cabin to three new columns
train[['Deck', 'Num', 'Side']] = train['Cabin'].str.split('/', 2, expand=True)
test[['Deck', 'Num', 'Side']] = test['Cabin'].str.split('/', 2, expand=True)

# split PassengerId into two new columns
train[['ID_0', 'ID_1']] = train['PassengerId'].str.split('_', 1, expand=True)
test[['ID_0', 'ID_1']] = test['PassengerId'].str.split('_', 1, expand=True)

# drop Cabin and PassengerId cols
train.drop(['Cabin', 'PassengerId', 'VIP'], axis=1, inplace=True)
test.drop(['Cabin', 'PassengerId', 'VIP'], axis=1, inplace=True)


# #### 3.2.2 Create new feature from numerical columns statistics
# - Let's create new features from the numerical features statistics such as the mean, min and standard deviations

# In[41]:


def feats_derived_from_stats(df, features, text=None):
    '''Given a dataframe and a list of features (numerical), 
    this function creates new features with the mean, min, and standard deviations of the 
    selected features and returns the full dataframe
    '''
    df[f'{text}_avg'] = df[features].mean(axis=1)
    df[f'{text}_std'] = df[features].std(axis=1)
    
    return df       


# In[42]:


f_VSR = ['VRDeck', 'Spa', 'RoomService']
train = feats_derived_from_stats(train, f_VSR, 'VSR')
test = feats_derived_from_stats(test, f_VSR, 'VSR')


# In[43]:


f_SF = ['ShoppingMall', 'FoodCourt']
train = feats_derived_from_stats(train, f_SF, 'SF')
test = feats_derived_from_stats(test, f_SF, 'SF')


# <!-- #### Family: Derive new feature from `ID_0` and `Cabin (Side)`
# - It seem that we can create a new feature (family) by combining `ID_0` and `Cabine (side)` assuming that people with the same last name are related.
# - See the two random examples below. Passengers with the same last names share the same `ID_0` and `Cabine (side)` information.
# - Note also these two `families` did not transport.
# 
# train[['FirstName', 'LastName']] = train['Name'].str.split(' ', 1, expand=True)
# test[['FirstName', 'LastName']] = test['Name'].str.split(' ', 1, expand=True)
# train[(train['ID_0'] == '0003') & (train['Side'] == 'S')]
# train[(train['ID_0'] == '0607') & (train['Side'] == 'P')]
#  -->

# ## 4. Modeling and Predictions <a class="anchor" id="4"></a>

# ### 4.1 AutoML_H2O <a class="anchor" id="4.1"></a>
# 
# H2O is an open source machine learning and predictive analytics platform developed by [H2O.ai](https://h2o.ai/platform/h2o-automl/). I used @vopani's notebook as a reference to buid this AutoML model [1].
# 

# In[44]:


# IMPORT THE PACKAGE
import h2o
from h2o.automl import H2OAutoML


# In[45]:


# INITIALIZE H2O CLUSTER (LOCALLY)
h2o.init()


# In[46]:


# PRESET

RANDOM_STATE = 42
RUN_TIME = 60 # run time in seconds

TARGET_NAME = 'Transported'

np.random.seed(RANDOM_STATE)

feature_columns = train.drop(['Transported'], axis=1).columns


train_hf = h2o.H2OFrame(train)
test_hf = h2o.H2OFrame(test)


# In[47]:


# BUILD THE MODEL
h2o_automl = H2OAutoML(
    seed=42,
    stopping_metric='logloss',
    max_runtime_secs = RUN_TIME,
    nfolds = 5,
    exclude_algos = ["DeepLearning"]
)

# FIT (TRAIN) THE MODEL
h2o_automl.train(
    x=list(feature_columns), 
    y=TARGET_NAME, 
    training_frame=train_hf
)


# In[48]:


# MODEL'S LEADERBOARD
leaderBoard = h2o_automl.leaderboard
leaderBoard.head(rows = 20)#leaderBoard.nrows)


# <!-- # https://stackoverflow.com/questions/51640086/is-it-possible-to-get-a-feature-importance-plot-from-a-h2o-automl-model
# #at the moment getting feature importance information is possible only for non-stacked models.
# 
# # feature_importance = h2o.get_model(leaderBoard[4,"model_id"])
# # df = feature_importance.varimp(use_pandas=True)
# # df.head() 
# 
# # fig = go.Figure(data=[go.Bar(
# #     y=df['variable'],
# #     x=df['percentage'],
# #     marker_color='lightseagreen',
# #     orientation='h'
# # )]
# 
# # ).update_yaxes(categoryorder='total ascending')
# # fig.update_layout(title_text='<b> Feature Importance',
# #                   font_family="San Serif",
# #                   titlefont={'size': 24},
# #                   width=600, height=700,
# #                   template='plotly_dark',
# #                   paper_bgcolor="darkgray",
# #                  )
# # fig.show() -->

# In[49]:


# MAKE PREDICTIONS
preds_h2o = h2o_automl.leader.predict(test_hf).as_data_frame()['True']


# In[50]:


# MAKE SUBMISSION FILE
submission[TARGET_NAME] = (preds_h2o.values > 0.5).astype(bool)
submission.to_csv('AutoML_H2O_nominnovip.csv', index=False)
submission.head()


# <a href="#top">Back to top</a>    

# ### 4.2 Gradient Boosting Machines <a class="anchor" id="4.2"></a>
# 
# "In machine learning, boosting is an ensemble meta-algorithm for primarily reducing bias, and also variance in supervised learning, and **a family of machine learning algorithms that convert weak learners to strong ones.** Boosting is based on the question posed by Kearns and Valiant "Can a set of weak learners create a single strong learner?" A weak learner is defined to be a classifier that is only slightly correlated with the true classification (it can label examples better than random guessing). In contrast, a strong learner is a classifier that is arbitrarily well-correlated with the true classification." [[7]](https://en.wikipedia.org/wiki/Boosting_%28machine_learning%29)
# 
# > Gradient boosting is a method that goes through cycles to iteratively add models into an ensemble.
# > It begins by initializing the ensemble with a single model, whose predictions can be pretty naive. (Even if its predictions are wildly inaccurate, subsequent additions to the ensemble will address those errors.)
# > Then, we start the cycle:
# > - First, we use the current ensemble to generate predictions for each observation in the dataset. To make a prediction, we add the predictions from all models in the ensemble.
# > - These predictions are used to calculate a loss function.
# > - Then, we use the loss function to fit a new model that will be added to the ensemble. Specifically, we determine model parameters so that adding this new model to the ensemble will reduce the loss. (Side note: The "gradient" in "gradient boosting" refers to the fact that we'll use gradient descent on the loss function to determine the parameters in this new model.)
# > - Finally, we add the new model to ensemble, and ...
# > - ... repeat!
# "
# 
# While there are more boosting algorithms used in machine learning, in this notebook we will consider only two of the boosting algorithms, Light gradient boosting machines (LGBM) and extreem gradient boosting (XGBoost).
# #### 4.2.1 LGBM <a class="anchor" id="4.2.1"></a>
# 
# "**[LightGBM](https://lightgbm.readthedocs.io/en/latest/index.html)** is a gradient boosting framework that uses **tree based learning algorithms**. It is designed to be distributed and efficient with the following advantages:
# 
# - Faster training speed and higher efficiency.
# - Lower memory usage.
# - Better accuracy.
# - Support of parallel, distributed, and GPU learning.
# - Capable of handling large-scale data.

# In[51]:


# first we label encode the categorical features we are going to use in our model

# categorical feature to use
catcols = ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Num', 'Side', 'ID_0', 'ID_1', 'Binned_Age',]

le = LabelEncoder()
def LE(train_df, test_df):
    for col in catcols:
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.fit_transform(test_df[col])
    return train_df, test_df

train, test = LE(train, test)

# separate target variable from the train dataset
y = train.pop('Transported')
X, test = train, test

# a configuration class 
class CFG:
    SEED = 42
    FOLDS = 5
    ESR = 150  


# In[52]:


# we first start with the default parameters and then (in later version of the notebook) we'll tune the hyperparameters later
lgbm_params = {'objective': 'binary'}


y_oof_pred = np.zeros(train.shape[0])
y_test_pred_lgbm = np.zeros(test.shape[0])

kf = StratifiedKFold(n_splits = CFG.FOLDS, shuffle= True, random_state=CFG.SEED)
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    
    X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
    lgbm= LGBMClassifier(**lgbm_params)

    lgbm.fit(X_train, y_train,
                 eval_set = [(X_train, y_train),(X_val, y_val)],
                 verbose = False, early_stopping_rounds=CFG.ESR)

    y_val_pred = lgbm.predict(X_val)

    print(f"Fold {fold + 1} Accuracy: {round(accuracy_score(y_val, y_val_pred), 5)}")

    y_oof_pred[val_idx] = y_val_pred
    y_test_pred_lgbm += lgbm.predict(test)


y_test_pred_lgbm= y_test_pred_lgbm/ CFG.FOLDS

#print(f"-- Overall OOF Accuracy: {accuracy_score(y, y_oof_pred)}")
print(f"\n{Fore.GREEN}{Style.BRIGHT} Overall OOF Accuracy = {round(accuracy_score(y, y_oof_pred), 5)}{Style.RESET_ALL}")


# #### 4.2.1.1 Hyperparameter optimization (optuna) <a class="anchor" id="4.2.1.1"></a>
# 
# [OPTUNA :](https://optuna.org/#key_features) An open source hyperparameter optimization framework to automate hyperparameter search. The key feature of optuna as described by the official webpage are:
# - Automated search for optimal hyperparameters using Python conditionals, loops, and syntax
# - Efficiently search large spaces and prune unpromising trials for faster results
# - Parallelize hyperparameter searches over multiple threads or processes without modifying code
# 
# We can use optuna with a long list of machine learning or deep learning framework which include XGBoost and LGBM for example. 
# 
# An oprtimazation problem using optuna follows three simple steps:
# - **Define objective function** to be optimized.
# - **Suggest hyperparameter values** using trial object. 
# - **Create a study object** and invoke the optimize method over number of trials.

# In[53]:


y.replace([False, True], [0,1], inplace = True)
from sklearn.model_selection import train_test_split

def objective(trial,data=X,target=y):
    
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2,random_state=42)
    param = {
        'metric': 'binary_logloss', 
        'random_state': CFG.SEED,
        'n_estimators': trial.suggest_int('n_estimators', 500, 5000),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 50.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 50.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.5,0.7,0.9,1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.005,0.01,0.02,0.04]),
        'max_depth': trial.suggest_categorical('max_depth', [10,20,1000]),
        'num_leaves' : trial.suggest_int('num_leaves', 1, 256),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        #'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100)
    }
    model = LGBMClassifier(**param)  
    model.fit(train_x,train_y,
              eval_set=[(test_x,test_y)],
              early_stopping_rounds=CFG.ESR,
              verbose=False)
    
    y_preds = model.predict(test_x)    
    score = accuracy_score(test_y, y_preds)
        
    return score


# In[54]:


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)


# In[55]:


print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)


# In[56]:


params = study.best_trial.params
# we first start with the default parameters and then (in later version of the notebook) we'll tune the hyperparameters later
lgbm_params = {'objective': 'binary'}


y_oof_pred = np.zeros(train.shape[0])
y_test_pred_lgbm = np.zeros(test.shape[0])

kf = StratifiedKFold(n_splits = CFG.FOLDS, shuffle= True, random_state=CFG.SEED)
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    
    X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
    lgbm= LGBMClassifier(**params)

    lgbm.fit(X_train, y_train,
                 eval_set = [(X_train, y_train),(X_val, y_val)],
                 verbose = False, early_stopping_rounds=CFG.ESR)

    y_val_pred = lgbm.predict(X_val)

    print(f"Fold {fold + 1} Accuracy: {round(accuracy_score(y_val, y_val_pred), 5)}")

    y_oof_pred[val_idx] = y_val_pred
    y_test_pred_lgbm += lgbm.predict(test)


y_test_pred_lgbm= y_test_pred_lgbm/ CFG.FOLDS

#print(f"-- Overall OOF Accuracy: {accuracy_score(y, y_oof_pred)}")
print(f"\n{Fore.GREEN}{Style.BRIGHT} Overall OOF Accuracy = {round(accuracy_score(y, y_oof_pred), 5)}{Style.RESET_ALL}")


# In[57]:


# MAKE SUBMISSION FILE (lightgbm model)
submission[TARGET_NAME] = y_test_pred_lgbm.astype(bool)
submission.to_csv('submission_lgbm.csv', index=False)
submission.head()


# #### 4.2.2 Xgboost <a class="anchor" id="4.2.2"></a>
# 
# "**XGBoost** is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way. The same code runs on major distributed environment (Hadoop, SGE, MPI) and can solve problems beyond billions of examples. https://xgboost.readthedocs.io/en/latest/

# In[58]:


# we first start with the default parameters and then (in later version of the notebook) we'll tune the hyperparameters later
xgb_params = {'eval_metric' : 'logloss'}

y_oof_pred = np.zeros(train.shape[0])
y_test_pred_xgb = np.zeros(test.shape[0])

kf = StratifiedKFold(n_splits = CFG.FOLDS, shuffle= True, random_state=CFG.SEED)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    
    X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
    xgb= XGBClassifier(**xgb_params)

    xgb.fit(X_train, y_train,
                 eval_set = [(X_train, y_train),(X_val, y_val)],
                 verbose = False, early_stopping_rounds=CFG.ESR)

    y_val_pred = xgb.predict(X_val)

    print(f"Fold {fold + 1} Accuracy: {round(accuracy_score(y_val, y_val_pred), 5)}")

    y_oof_pred[val_idx] = y_val_pred
    y_test_pred_xgb += xgb.predict(test)


y_test_pred_xgb= y_test_pred_xgb/ CFG.FOLDS
print(f"\n{Fore.GREEN}{Style.BRIGHT} Overall OOF Accuracy = {round(accuracy_score(y, y_oof_pred), 5)}{Style.RESET_ALL}")


# #### 4.2.2.1 Hyperparameter optimization (optuna) <a class="anchor" id="4.2.2.1"></a>

# In[59]:


y.replace([False, True], [0,1], inplace = True)
from sklearn.model_selection import train_test_split

def objective(trial,data=X,target=y):
    
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2,random_state=42)
    param = {
        #'eval_metric': 'binary:logistic', 
        'random_state': CFG.SEED,
        'n_estimators': trial.suggest_int('n_estimators', 500, 5000),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 50.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 50.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.5,0.7,0.9,1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.005,0.01,0.02,0.04]),
        'max_depth': trial.suggest_categorical('max_depth', [10,20,1000]),
        #'num_leaves' : trial.suggest_int('num_leaves', 1, 256),
        #'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        #'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100)
    }
    model = XGBClassifier(**param)  
    model.fit(train_x,train_y,
              eval_set=[(test_x,test_y)],
              early_stopping_rounds=CFG.ESR,
              verbose=False)
    
    y_preds = model.predict(test_x)    
    score = accuracy_score(test_y, y_preds)
        
    return score


# In[60]:


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)


# In[61]:


print('Number of finished trials:', len(study.trials))
print('Best trial parameters:', study.best_trial.params)


# In[62]:


params = study.best_trial.params
y_oof_pred = np.zeros(train.shape[0])
y_test_pred_xgb = np.zeros(test.shape[0])

kf = StratifiedKFold(n_splits = CFG.FOLDS, shuffle= True, random_state=CFG.SEED)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    
    X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
    xgb= XGBClassifier(**params)

    xgb.fit(X_train, y_train,
                 eval_set = [(X_train, y_train),(X_val, y_val)],
                 verbose = False, early_stopping_rounds=CFG.ESR)

    y_val_pred = xgb.predict(X_val)

    print(f"Fold {fold + 1} Accuracy: {round(accuracy_score(y_val, y_val_pred), 5)}")

    y_oof_pred[val_idx] = y_val_pred
    y_test_pred_xgb += xgb.predict(test)

y_test_pred_xgb= y_test_pred_xgb/ CFG.FOLDS
print(f"\n{Fore.GREEN}{Style.BRIGHT} Overall OOF Accuracy = {round(accuracy_score(y, y_oof_pred), 5)}{Style.RESET_ALL}")


# In[63]:


# MAKE SUBMISSION FILE (xgboost model)
submission[TARGET_NAME] = y_test_pred_xgb.astype(bool)
submission.to_csv('submission_xgb.csv', index=False)
submission.head()


# **Remark**: For both the `xgboost` and `lighbgm` models the model with default parameters resulted in a slightly better score than the Optuna tuned models. Two possible explanation are:
# - may be the search space used for hyperparameter tuning is not big enough
# - may be hyperparameter tuning has less potential than feature engineering (for example) in terms of improving the accuracy of the classifiers for this classification challenge
# 
# We shall explore more of this in the future, for now let's proceed to the model explainability part.
# 

# ### 4.3 Model Explainablity  <a class="anchor" id="4.3"></a>
# 
# 
# #### **4.3.1 Permutation importance**
# 
# Permutation importance is defined as "the decrease in a model score when a single feature value is randomly shuffled". The procedure breaks the relationship between the feature and the target, thus the drop in the model score is indicative of how much the model depends on the feature [2, 3, 4]. In other words, permutation importance tell us what features have the biggest impact on our model predictions.

# In[64]:


import eli5
from eli5.sklearn import PermutationImportance

perm_imp = PermutationImportance(lgbm, random_state=42).fit(X_val, y_val)
eli5.show_weights(perm_imp, feature_names = X_val.columns.tolist())


# **Remark**: The above table can be interpreated as:
# 
# - Features which are at or near the top of the table are more important than features down the bottom. Thus our derived feature `VSR_avg` (which is the average of VRDeck, ShoppingMall and RoomService expenditure) is at the top of the list followed by `Deck` and `CryoSleep`. On the other hand, `ID_1` and `VRDeck` are not so important for the model to determine which passenger is likely to be transported or not.
# - The numbers corresponding to each feature is the weight given to the features by the model. We see that the first feature is way ahead of the rest (it is not a contest even) signaling that this feature is so special to the model. The `+/-` value takes care of the randomness.
# - On a side note, we also see that `feature interaction?` can sometimes give us much more information than an individual feature can. `VRDeck` and `ShoppingMall` are right at the foot of the table when considered alone. Whereas their combined average (together with `RoomService`) is deemed so important by our ML model.
# 

# #### **4.3.2 SHAP**
# 
# SHAP, a short name for SHapely Additive ExPlanations, is a method used to explain the output of a machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions [5]. SHAP has a rich functionality (methods) by which we can visualize/interpret the output of our models. Below we use the shap.summary_plot() to identify the impact each feature has on the predicted output.
# 

# In[65]:


import shap
shap.initjs()
explainer = shap.TreeExplainer(lgbm)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values[:], X_train)


# In[66]:


explainer = shap.Explainer(lgbm, X_train)
shap_values = explainer(X_train)
shap.plots.beeswarm(shap_values[:], max_display=len(X_train))


# **Remark**
# 
#  - We see that **SHAP** has picked `VSR_avg` , `CryoSleep` , and `Deck` as the top three most important features for the model. Except for the ordering these features are the same as the `eli5` feature importance. More or less the same story when it comes to the least important features as well. Since the algorithms/mathematics of the two models (eli5 and SHAP) are not exactly the same, we don't expect them to match with each other feature-for-feature. Nevertheless, the global picture is the same.
# 

# ### 
# 
# ## 5. Reference  <a class="anchor" id="5"></a>
# 
# * [1] https://www.kaggle.com/code/rohanrao/automl-tutorial-tps-may-2021
# * [2] https://www.kaggle.com/code/dansbecker/permutation-importance
# * [3] https://scikit-learn.org/stable/modules/permutation_importance.html?highlight=permutation
# * [4] https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html
# * [5] https://shap.readthedocs.io/en/latest/index.html
# * [6] https://h2o.ai/platform/h2o-automl/
# * [7] https://en.wikipedia.org/wiki/Boosting_%28machine_learning%29
# * [8] https://www.kaggle.com/code/alexisbcook/xgboost
# * [9] https://lightgbm.readthedocs.io/en/latest/index.html

# ### End of notebook!
# 
# #### Thank you for reading!
# ___
# 
# ___

# In[ ]:




