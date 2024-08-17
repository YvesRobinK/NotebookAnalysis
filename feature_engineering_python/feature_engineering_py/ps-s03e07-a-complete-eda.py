#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('wget http://bit.ly/3ZLyF82 -O CSS.css -q')
    
from IPython.core.display import HTML
with open('./CSS.css', 'r') as file:
    custom_css = file.read()

HTML(custom_css)


# # PS S03E07: A Complete EDA ⭐️
# 
# This EDA gives useful insights when designig a machine learning pipeline for this episode competition
# 
# **Versions**
# - v1: Inital EDA
# - v2: Some feature egineering on dates
# - v3: Added Demand-Booking-Datasets
# - v4: Added Booking-date Feature Engineering
# - v5: Removed feature engineering as per [this discussion](https://www.kaggle.com/competitions/playground-series-s3e7/discussion/388086)
# 
# # Table of Content
# 
# 1. [The Data](#The-Data)
#     1. [Demand Bookings](#Demand-Bookings)
#         1. [Meal Plans Mappings:](#Meal-Plans-Mappings:)
# 1. [The Label](#The-Label)
# 1. [EDA](#EDA)
#     1. [Data Size](#Data-Size)
# 1. [Distributions](#Distributions)
#     1. [Numerical + Ordinal Features](#Numerical-+-Ordinal-Features)
#     1. [Categorical Columns](#Categorical-Columns)
# 1. [Arrival Date Feature Engineering](#Arrival-Date-Feature-Engineering)
# 1. [Arrival Date Features](#Arrival-Date-Features)
#     1. [Wrong dates](#Wrong-dates)
#     1. [Arrival Date-based Features](#Arrival-Date-based-Features)
# 1. [Booking Date Feature Engineering](#Booking-Date-Feature-Engineering)
# 1. [Booking-Date-Related Features](#Booking-Date-Related-Features)
# 1. [Missing Values](#Missing-Values)
# 1. [Duplicates](#Duplicates)
# 1. [Correlations](#Correlations)
# 1. [Basic Baseline](#Basic-Baseline)
# 1. [Submission](#Submission)
# 

# In[2]:


from time import time
from datetime import timedelta
from colorama import Fore, Style
from lightgbm import early_stopping
from lightgbm import log_evaluation

import math
import matplotlib
import matplotlib as mpl
import matplotlib.cm as cmap
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import lightgbm as lgbm
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import warnings
from cycler import cycler
from matplotlib.ticker import MaxNLocator

palette = ['#3c3744', '#048BA8', '#EE6352', '#E1BB80', '#78BC61']
grey_palette = [
    '#8e8e93', '#636366', '#48484a', '#3a3a3c', '#2c2c2e', '#1c1c27'
]

bg_color = '#F6F5F5'
white_color = '#d1d1d6'

custom_params = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.spines.left": False,
    'grid.alpha':0.2,
    'figure.figsize': (16, 6),
    'axes.titlesize': 'large',
    'axes.labelsize': 'large',
    'ytick.labelsize': 'medium',
    'xtick.labelsize': 'medium',
    'legend.fontsize': 'large',
    'lines.linewidth': 1,
    'axes.prop_cycle': cycler('color',palette),
    'figure.facecolor': bg_color,
    'figure.edgecolor': bg_color,
    'axes.facecolor': bg_color,
    'text.color':grey_palette[1],
    'axes.labelcolor':grey_palette[1],
    'axes.edgecolor':grey_palette[1],
    'xtick.color':grey_palette[1],
    'ytick.color':grey_palette[1],
    'figure.dpi':150,
}

sns.set_theme(
    style='whitegrid',
    palette=sns.color_palette(palette),
    rc=custom_params
)


# # The Data
# 
# The dataset was generated using [Reservation Cancellation Prediction](https://www.kaggle.com/datasets/gauravduttakiit/reservation-cancellation-prediction). The task is to predict the right `booking_status` which is an binary column indicating whether the reservation was cancelled or not.
# 
# Some key aspects are:
# 
# 1 Competition dataset generated from Reservation Cancellation Prediction dataset
# 2 Deep learning model used to generate both train and test dataset
# 3 Feature distributions are similar but not exactly the same as the original dataset
# 4 Original dataset can be used as part of competition to explore differences and improve model performance
# 
# ---
# Columns description from original [Reservation Cancellation Prediction](https://www.kaggle.com/datasets/gauravduttakiit/reservation-cancellation-prediction)
# 
# The file contains the different attributes of customers' reservation details. The detailed data dictionary is given below
# * id: unique identifier of each booking
# * no_of_adults: Number of adults
# * no_of_children: Number of Children
# * no_of_weekend_nights: Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel
# * no_of_week_nights: Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel
# * type_of_meal_plan: Type of meal plan booked by the customer:
# * required_car_parking_space: Does the customer require a car parking space? (0 - No, 1- Yes)
# * room_type_reserved: Type of room reserved by the customer. The values are ciphered (encoded) by INN Hotels.
# * lead_time: Number of days between the date of booking and the arrival date
# * arrival_year: Year of arrival date
# * arrival_month: Month of arrival date
# * arrival_date: Date of the month
# * market_segment_type: Market segment designation.
# * repeated_guest: Is the customer a repeated guest? (0 - No, 1- Yes)
# * no_of_previous_cancellations: Number of previous bookings that were canceled by the customer prior to the current booking
# * no_of_previous_bookings_not_canceled: Number of previous bookings not canceled by the customer prior to the current booking
# * avg_price_per_room: Average price per day of the reservation; prices of the rooms are dynamic. (in euros)
# * no_of_special_requests: Total number of special requests made by the customer (e.g. high floor, view from the room, etc)
# 
# Output variable:
# - booking_status (0 or 1)

# In[3]:


train_df = pd.read_csv('/kaggle/input/playground-series-s3e7/train.csv', index_col=0)
test_df = pd.read_csv('/kaggle/input/playground-series-s3e7/test.csv', index_col=0)

# load original dataset
original_train_df = pd.read_csv(
    '/kaggle/input/reservation-cancellation-prediction/train__dataset.csv'
)
original_test_df = pd.read_csv(
    '/kaggle/input/reservation-cancellation-prediction/test___dataset.csv'
)


original_train_df.index.name = 'id'
original_test_df.index.name = 'id'


# ## Demand Bookings
# 
# This data is complementary to the original and synthetic dataset, in can help the model by augmenting the data. See this [discussion](https://www.kaggle.com/competitions/playground-series-s3e7/discussion/386788) for more information.
# 
# ### Meal Plans Mappings:
# 
# Taken from [here](https://www.logitravel.co.uk/frequently-asked-questions/hotels-what-does-ro-bb-hb-fb-ai-on-my-hotel-voucher-8_172.html)
# 1. BB - Bed and Breakfass
# 2. HB - Half Board
# 3. SC - Self Catering
# 4. FB - Full Board
# 5. Undefined

# In[4]:


demand_bookings_df = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')
demand_bookings_df.rename(columns = {
    'adults': 'no_of_adults',
    'children': 'no_of_children',
    'stays_in_weekend_nights': 'no_of_weekend_nights',
    'stays_in_week_nights': 'no_of_week_nights',
    'meal': 'type_of_meal_plan',
    'required_car_parking_spaces': 'required_car_parking_space',
    'reserved_room_type': 'room_type_reserved',
    'lead_time': 'lead_time',
    'arrival_date_year': 'arrival_year',
    'arrival_date_month': 'arrival_month',
    'arrival_date_day_of_month': 'arrival_date',
    'market_segment': 'market_segment_type',
    'is_repeated_guest': 'repeated_guest',
    'previous_cancellations': 'no_of_previous_cancellations',
    'previous_bookings_not_canceled': 'no_of_previous_bookings_not_canceled',
    'adr': 'avg_price_per_room',
    'total_of_special_requests': 'no_of_special_requests',
    'is_canceled': 'booking_status'
}, inplace=True)

demand_bookings_df['arrival_month'] = demand_bookings_df['arrival_month'].map({
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
})

demand_bookings_df['type_of_meal_plan'] = \
    demand_bookings_df['type_of_meal_plan'].map({
        'BB': 0, 'HB': 2, 'SC': 1, 'Undefined': 1, 'FB': 3
    })

demand_bookings_df['market_segment_type'] = \
    demand_bookings_df['market_segment_type'].map({
        "'Online TA": 1, "Offline TA/TO": 0, "Corporate": 2, "Complementary": 4, "Aviation": 3
    })

demand_bookings_df['room_type_reserved'] = \
    demand_bookings_df['room_type_reserved'].map({
        'A':  0, 'D':  1, 'E':  3, 'F':  2, 'G':  4, 'B':  5, 'C':  6
    })

demand_bookings_df['market_segment_type'].fillna(5, inplace=True)
demand_bookings_df['room_type_reserved'].fillna(7, inplace=True)


# # The Label
# 
# 
# The label target seems to follow the same distribution, this is good in the sense that augmenting the dataset will have great power as we have seen in prevous competitions. Apart from having the same distribution, the label is no imbalaced, so this competition won't involve any oversampling, undersampling nor smote.
# 
# **Insights**
# - StratifiedKFold is recommended as the initial cross validations strategy.
# - Further is explained that dates have very little influence on cross validation as train and test shares the same dates.

# In[5]:


print('Train')
display(train_df.booking_status.value_counts(True))

print('\nOriginal')
display(original_train_df.booking_status.value_counts(True))

print('\nDemand')
display(demand_bookings_df.booking_status.value_counts(True))


# # EDA
# 
# ## Data Size
# 
# **Insights**:
# - This a fairly light dataset, but compared to episode 5, we have enough data to validate if CV results correlates with LB. In this competition if test dataset has similar distributions than train, CV is a proxy of LB.
# - Original dataset has less records that synthetic dataset, beware of duplicates in both datasets.

# In[6]:


print('Train shape:            ', train_df.shape)
print('Test shape:             ', test_df.shape)
print('Original Train shape:   ', original_train_df.shape)
print('Original Test shape:    ', original_test_df.shape)
print('Demand shape:           ', demand_bookings_df.shape)


# # Distributions
# 
# It's important to check the distribution of data, as it gives us information about any anomalies in the variables and whether preprocessing is necessary. The next section will use basic plots to show relationships between the features, the target, and any differences between the synthetic and original datasets.
# 
# ## Numerical + Ordinal Features
# 
# **Insights**:
# 
# - Most of the features are counts, so they can be threated as ordinal features that is why we can use lineplots to visualize the relationship.
# - At first glance there are no discernable difference between train-test and original dataset distributions, supporting the idea of including them as part of our training exampels.
# - The relationship between features and target is different in the synthetic and original datasets, with the feature `no_of_week_nights` being the most concerning difference for outliers. The model that include this feature and original dataset should handle outliers or original dataset will polute the training dataset.
# - `no_of_previous_bookings_not_canceled` is mostly zero everywhere either because most people that repeat this hotel won't cancel the booking or beacuse this is the first time this group of people goes to the hotel. People exhibits a repetetive pattern the more cancellations they do.
# - If the number of special request increases, the less likely are the people to cancel their booking.

# In[7]:


# Some helper Function
def plot_dots_ordinal(feature, ax):
    dots = total_df.groupby(['set', feature])['booking_status'].mean().reset_index(level=1)
    dots.sort_values(feature, inplace=True)
    train_containers = ax.containers[0]
    original_containers = ax.containers[2]
    demand_containers = ax.containers[4]
    
    containers = [train_containers, original_containers, demand_containers]
    sets = ['train', 'original_train', 'demand']
    colors = ['#78BC61', '#FF7F50', '#FF69B4']
    counter = 0
    
    for set_, container in zip(sets, containers):
        dots_subset = dots.loc[set_]
        
        x_s = [bar.get_x() + bar.get_width()/2 for bar in container]
        y_s = dots_subset.booking_status
        x_s = x_s[:y_s.shape[0]]
        
        ax.plot(x_s, y_s, marker='.', alpha=0.8, 
                linestyle=line_style, markersize=10,
                color=colors[counter]
        )
        
        counter += 1

def plot_ordinals(feature, ax):
    percentage = total_df.groupby('set')[feature].value_counts(True)
    percentage = percentage.rename('%').reset_index()
    sns.barplot(data=percentage, x=feature, y='%',
                hue='set',ax=ax, hue_order=labels)
    
    
    plot_dots_ordinal(feature, ax)
    ticks = ax.get_xticklabels()
    ticks = [str(int(float(x.get_text()))) for x in ticks]
    ax.set_xticklabels(ticks)
    
    if percentage.shape[0] > 10:
        ax.xaxis.set_major_locator(MaxNLocator(10))
        
def plot_continous(feature, ax):
    temp = total_df.copy()
    temp[feature] = temp[feature].clip(upper=temp[feature].quantile(0.99))
    
    sns.histplot(data=temp, x=feature,
                hue='set',ax=ax, hue_order=labels,
                common_norm=False, **histplot_hyperparams)
    
    ax_2 = ax.twinx()
    ax_2 = plot_dot_continous(
        total_df.query('set=="train"'),
        feature, 'booking_status', ax_2,
        color='#78BC61'
    )
    
    ax_2 = plot_dot_continous(
        total_df.query('set=="original_train"'),
        feature, 'booking_status', ax_2,
        color='#FF7F50'
    )
#     ax.legend(handles, legend_labels)
        
    
def plot_dot_continous(
    df, column, target, ax,
    show_yticks=False, color='green'
):

    bins = pd.cut(df[column], bins=n_bins)
    bins = pd.IntervalIndex(bins)
    bins = (bins.left + bins.right) / 2
    target = df[target]
    target = target.groupby(bins).mean()
    target.plot(
        ax=ax, linestyle='',
        marker='.', color=color,
        label=f'Mean {target.name}'
    )
    ax.grid(visible=False)
    
    if not show_yticks:
        ax.get_yaxis().set_ticks([])
        
    return ax


total_df = pd.concat([
    train_df.assign(set='train'),
    test_df.assign(set='test'),
    original_train_df.assign(set='original_train'),
    original_test_df.assign(set='original_test'),
    demand_bookings_df.assign(set='demand')
], ignore_index=True)

total_df.reset_index(drop=True, inplace=True)
labels = ['train', 'test', 'original_train', 'original_test', 'demand']

ordinal_features = [
    'no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
    'no_of_special_requests', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled'
]

numeric_features = [
    'lead_time', 'avg_price_per_room'
]

n_bins = 50
histplot_hyperparams = {
    'kde':True,
    'alpha':0.4,
    'stat':'percent',
    'bins':n_bins
}
line_style='--'

columns =  ordinal_features + numeric_features
n_cols = 3
n_rows = math.ceil(len(columns)/n_cols)
fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, n_rows*5))
ax = ax.flatten()

for i, column in enumerate(columns):
    plot_axes = [ax[i]]
    
    if column in ordinal_features:
        plot_ordinals(column, ax[i])
    else:
        plot_continous(column, ax[i])

    # titles
    ax[i].set_title(f'{column} Distribution', pad=35);
    ax[i].set_xlabel(None)
    ax[i].legend(fontsize=9, bbox_to_anchor=(0.5, 1.12), ncol=4,
                loc='upper center')

for i in range(i+1, len(ax)):
    ax[i].axis('off')

plt.tight_layout()


# ## Categorical Columns
# 
# **Insights**:
# 
# - Categorical features exhibits the same discrepancy between synthetic and original dataset when analyzing the relantionship with the target column.
# - `type_of_meal_plan` should not be included in a linear model with augmented dataset becasuse the distributions are quite different and the rank of categories by target-mean is different between synthetic and original.
# - `room_type_reserved` has clear relationship with the target.

# In[8]:


# Some helper Function
def plot_dots_categorical(feature, ax):
    dots = total_df.groupby(['set', feature])['booking_status'].mean().reset_index(level=1)
    dots.sort_values(feature, inplace=True)
    train_containers = ax.containers[0]
    original_containers = ax.containers[2]
    demand_containers = ax.containers[4]
    
    containers = [train_containers, original_containers, demand_containers]
    sets = ['train', 'original_train', 'demand']
    colors = ['#78BC61', '#FF7F50', '#FF69B4']
    counter = 0
    
    for set_, container in zip(sets, containers):
        dots_subset = dots.loc[set_]
        
        x_s = [bar.get_x() + bar.get_width()/2 for bar in container]
        y_s = dots_subset.booking_status
        x_s = x_s[:y_s.shape[0]]
        
        ax.plot(x_s, y_s, marker='.', alpha=0.8, 
                linestyle=line_style, markersize=10,
                color=colors[counter]
        )
        
        counter += 1
        
def plot_categorical(feature, ax):
    percentage = total_df.groupby('set')[feature].value_counts(True)
    percentage = percentage.rename('%').reset_index()
    sns.barplot(data=percentage, x=feature, y='%',
                hue='set',ax=ax, hue_order=labels)
    
    if percentage.shape[0] > 100:
        ticks = ax.get_xticks()
        text = ax.get_xticklabels()
        
        step = len(ticks)//8
        ax.set_xticks(ticks[::step], text[::step])
        
    plot_dots_categorical(feature, ax)

categorical_columns = [
    'market_segment_type', 'repeated_guest', 'required_car_parking_space',
    'room_type_reserved', 'type_of_meal_plan'
]
line_style=''

columns = categorical_columns
n_cols = 3
n_rows = math.ceil(len(columns)/n_cols)
fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, n_rows*5))
ax = ax.flatten()

for i, column in enumerate(columns):
    plot_axes = [ax[i]]
    plot_categorical(column, ax[i])

    # titles
    ax[i].set_title(f'{column} Distribution');
    ax[i].set_xlabel(None)
    ax[i].legend(fontsize=8, loc='upper right')
    
for i in range(i+1, len(ax)):
    ax[i].axis('off')


# # Arrival Date Feature Engineering

# In[9]:


def process_arrival_date(df):
    df.drop(columns=['year', 'month', 'day'], inplace=True, errors='ignore')
    
    temp = df.rename(columns={
        'arrival_year': 'year',
        'arrival_month': 'month',
        'arrival_date': 'day'
    })

    df['date'] = pd.to_datetime(temp[['year', 'month', 'day']], errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week.astype(float)
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['dayofyear'] = df['date'].dt.dayofyear
    
    df.drop(columns='date', inplace=True)
    return df


# # Arrival Date Features
# 
# **Insights**:
# 
# - Datetime should not be used as a feature because of hight cardinality. We can either create Fourier-based features by checking seasonality trends.
# - There are some pronounced pikes on the first and latest month of the year, this is because holidays.
# - Clear seasonality trend over the cancellations.
# - Year 2017 is really noisy because we don't have a lot of data compared to 2018.

# In[10]:


temp = total_df.rename(columns={
    'arrival_year': 'year',
    'arrival_month': 'month',
    'arrival_date': 'day'
}).copy()

total_df['date'] = pd.to_datetime(temp[['year', 'month', 'day']], errors='coerce')
to_plot = total_df.groupby(['date', 'set']).size().rename('booking_count').reset_index()

fig, ax = plt.subplots(2, 1, figsize=(16, 12))
sns.lineplot(data=to_plot, x='date', y='booking_count', hue='set',
            hue_order=labels, ax=ax[0])

to_plot = total_df.groupby(['date', 'set'])['booking_status'].mean().reset_index()
sns.lineplot(data=to_plot, x='date', y='booking_status', hue='set',
            hue_order=labels, ax=ax[1])


ax[0].set_title('Count of Arrivals')
ax[1].set_title('Mean Booking Cancellations Based On Arrival Date')
plt.tight_layout()


# ## Wrong dates
# 
# The following code provides all dates that are wrong if we consider a traditional calendar.

# In[11]:


pd.set_option('display.max_colwidth', None)
wrong_dates = total_df[
    ['arrival_year', 'arrival_month', 'arrival_date', 'set']
].loc[total_df.date.isnull()]

display(
    wrong_dates.groupby('set').apply(
        lambda df: df[
            ['arrival_year', 'arrival_month', 'arrival_date']
        ].apply(tuple, axis=1).unique()
    )
)


# ## Arrival Date-based Features

# In[12]:


feature_dates = [
    'year', 'month', 'week', 'day', 'dayofweek', 'quarter', 'dayofyear'
]
line_style=''
total_df = process_arrival_date(total_df)

columns = feature_dates
n_cols = 3
n_rows = math.ceil(len(columns)/n_cols)
fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, n_rows*5))
ax = ax.flatten()

for i, column in enumerate(columns):
    plot_axes = [ax[i]]
    plot_ordinals(column, ax[i])
    ax[i].legend(fontsize=9)

    # titles
    ax[i].set_title(f'{column} Distribution');
    ax[i].set_xlabel(None)
    
for i in range(i+1, len(ax)):
    ax[i].axis('off')


# # Booking Date Feature Engineering

# In[13]:


def process_booking_date(df):
    df.drop(columns=['year', 'month', 'day'], inplace=True, errors='ignore')
    
    temp = df.rename(columns={
        'arrival_year': 'year',
        'arrival_month': 'month',
        'arrival_date': 'day'
    })

    df['booking_date'] = pd.to_datetime(temp[['year', 'month', 'day']], errors='coerce')
    df['booking_date'] = df['booking_date'] - pd.Series(
        [pd.Timedelta(i, 'd') for i in df.lead_time],
        index=df.index
    )
    
    df['booking_year'] = df['booking_date'].dt.year
    df['booking_month'] = df['booking_date'].dt.month
    df['booking_week'] = df['booking_date'].dt.isocalendar().week.astype(float)
    df['booking_day'] = df['booking_date'].dt.day
    df['booking_dayofweek'] = df['booking_date'].dt.dayofweek
    df['booking_quarter'] = df['booking_date'].dt.quarter
    df['booking_dayofyear'] = df['booking_date'].dt.dayofyear
    
    df.drop(columns='booking_date', inplace=True)
    return df


# # Booking-Date-Related Features
# 
# **Insigths**:
# 
# - Most of the cancels are from booking that were purchased the first weeks of the year. The model by itself won't learn this interaction so we may improve our score by adding booking-date-related features.

# In[14]:


def plot_dots_ordinal(feature, ax):
    dots = total_df.groupby(['set', feature])['booking_status'].mean().reset_index(level=1)
    dots.sort_values(feature, inplace=True)
    train_containers = ax.containers[0]
    
    containers = [train_containers]
    sets = ['train', 'original_train', 'demand']
    colors = ['#78BC61', '#FF7F50', '#FF69B4']
    counter = 0
    
    ax2 = ax.twinx()
    for set_, container in zip(sets, containers):
        dots_subset = dots.loc[set_]
        
        x_s = [bar.get_x() + bar.get_width()/2 for bar in container]
        y_s = dots_subset.booking_status
        x_s = x_s[:y_s.shape[0]]
        
        ax2.plot(x_s, y_s, marker='.', alpha=0.8, 
                linestyle=line_style, markersize=10,
                color=colors[counter]
        )
        counter += 1
        
    ax2.grid(visible=False)
    ax2.set_yticks([], [])
    

feature_dates = [
    'booking_month', 'booking_week', 'booking_dayofyear',
    'booking_day', 'booking_dayofweek', 'booking_quarter',
    'booking_dayofyear', 'booking_year'
]
line_style=''

bck = total_df.copy()
total_df = process_booking_date(total_df)
total_df = total_df.query('set == "train"')
columns = feature_dates
n_cols = 3
n_rows = math.ceil(len(columns)/n_cols)
fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, n_rows*5))
ax = ax.flatten()

for i, column in enumerate(columns):
    column = columns[i]
    plot_axes = [ax[i]]
    sns.countplot(data=total_df, x=column, ax=ax[i],
                  color=palette[0], alpha=0.2)
    plot_dots_ordinal(column, ax[i])
    
    # titles
    ax[i].set_title(f'{column} Distribution');
    ax[i].set_xlabel(None)
    
    xticks = ax[i].get_xticklabels()
    xticks = [int(float(x.get_text())) for x in xticks]
    ax[i].set_xticklabels(xticks)
    
    if len(ax[i].get_xticklabels()) > 10:
        ax[i].xaxis.set_major_locator(MaxNLocator(10))
    
for i in range(i+1, len(ax)):
    ax[i].axis('off')
    
total_df = bck.copy()


# # Missing Values
# 
# **Insights**
# - There are no null values, this competition won't involve any null filling technique. We can use directly any model out of the box.

# In[15]:


train_null = train_df.isnull().sum().rename('train')
test_null = test_df.isnull().sum().rename('test')
original_train_null = original_train_df.isnull().sum().rename('original train')
original_test_null = original_test_df.isnull().sum().rename('original test')
demand_booking_null = demand_bookings_df.isnull().sum().rename('Demand Bokking')

pd.concat([train_null, test_null, original_train_null, original_test_null, demand_booking_null], axis=1)


# # Duplicates
# 
# **Insights**:
# 
# - There are lot of duplicates on test set. By Probbing the leaderboard one can assign the "right" value to this records.
# - Orginal train and original test also have lots of duplicates, if augmenting the dataset the value can lead to noisy predictions.

# In[16]:


train_dups = train_df.duplicated().sum()
test_dups = test_df.duplicated().sum()
original_train_dups = original_train_df.duplicated().sum()
original_test_dups = original_test_df.duplicated().sum()

print(f'''-------------------------
train:           {train_dups}
test:            {test_dups}
original_train:  {original_train_dups}
original_test:   {original_test_dups}
''')


# # Correlations

# In[17]:


fig, ax = plt.subplots(1, 3, figsize=(20, 20))
float_types = [np.int64, np.float16, np.float32, np.float64]
float_columns = train_df.select_dtypes(include=float_types).columns
cbar_ax = fig.add_axes([.91, .39, .01, .2])

names = ['Train', 'Original']
for i, df in enumerate([train_df, original_train_df]):
    
    corr = df[float_columns].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr, mask=mask, cmap='inferno',
        vmax=0.8, vmin=-1,
        center=0, annot=False, fmt='.3f',
        square=True, linewidths=.5,
        ax=ax[i],
        cbar=False,
        cbar_ax=None
    );

    ax[i].set_title(f'Correlation matrix for {names[i]} df', fontsize=14)

df = test_df
float_columns = test_df.select_dtypes(include=float_types).columns
corr = test_df[float_columns].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(
    corr, mask=mask, cmap='inferno',
    vmax=0.8, vmin=-1,
    center=0, annot=False, fmt='.3f',
    square=True, linewidths=.5,
    cbar_kws={"shrink":.5, 'orientation':'vertical'},
    ax=ax[2],
    cbar=True,
    cbar_ax=cbar_ax
);
ax[2].set_title(f'Correlation matrix for Test', fontsize=14)
fig.tight_layout(rect=[0, 0, .9, 1]);


# # Basic Baseline

# In[18]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

lgbm_params = {
    'max_depth':9,
    'n_estimators': 300,
    'min_child_samples':46,
    'num_leaves':80,
    'reg_alpha':0.01,
    'reg_lambda':0.6,
    'subsample':0.45,
    'colsample_bytree': 0.3
}

cv = StratifiedKFold(5, shuffle=True, random_state=42)
features = train_df.columns.difference(['booking_status'])

X = train_df[features]
X_ts = test_df[features]
y = train_df.booking_status

res = []
aucs = []
test_preds = []
models = []
oof_preds = pd.Series(0, index=train_df.index)
start = time()

for fold, (tr_ix, vl_ix) in enumerate(cv.split(train_df, train_df.booking_status)):
    start_fold = time()
    X_tr, y_tr = X.loc[tr_ix].copy(), y.loc[tr_ix]
    X_vl, y_vl = X.loc[vl_ix].copy(), y.loc[vl_ix]
    X_ts = test_df[features].copy()

#     concat orginal df
#     X_tr = pd.concat([X_tr, original_train_df[features]], ignore_index=True)
#     y_tr = pd.concat([y_tr, original_train_df.booking_status], ignore_index=True)

#     X_tr = pd.concat([X_tr, demand_bookings_df[features]], ignore_index=True)
#     y_tr = pd.concat([y_tr, demand_bookings_df.booking_status], ignore_index=True)

    model = LGBMClassifier(**lgbm_params, random_state=42)
    model.fit(X_tr, y_tr,
              eval_set=(X_vl, y_vl),
              eval_metric='auc',
              callbacks=[
                  log_evaluation(0), 
                  early_stopping(
                      50, verbose=False,
                      first_metric_only=True
                  )
              ]
             )
    y_pred = model.predict_proba(X_vl)[:, 1]
    oof_preds.iloc[vl_ix] = y_pred

    test_preds.append(model.predict_proba(X_ts)[:, 1])
    aucs.append(roc_auc_score(y_vl, y_pred))
    models.append(model)

    print('_' * 30)
    print(f'Fold: {fold} - {timedelta(seconds=int(time()-start))}')
    print(f'Fold roc AUC  : ', aucs[-1])
    print(f'Train Time taken :  {timedelta(seconds=int(time()-start_fold))}')
    print()

print(f'Mean ROC AUC:  {Fore.GREEN}{np.mean(aucs)}{Style.RESET_ALL}')
print(f'OFF ROC AUC:   {Fore.GREEN}{roc_auc_score(y, oof_preds)}{Style.RESET_ALL}')


# # Submission

# In[19]:


submission = pd.read_csv('/kaggle/input/playground-series-s3e7/sample_submission.csv', index_col=0)
submission.booking_status = np.mean(test_preds, axis=0)


# In[20]:


fig, ax = plt.subplots()
sns.histplot(oof_preds, alpha=0.3, color=palette[0], ax=ax, bins=50, stat='percent', label='oof')
sns.histplot(submission.booking_status, ax=ax, alpha=0.3, color=palette[1], bins=50, stat='percent', label='test')

plt.legend()

