#!/usr/bin/env python
# coding: utf-8

# <img src="https://i.imgur.com/de3OrxO.png">
# 
# <center><h1> Detect Sleep States </h1></center>
# <center><h1>- time patterns explore & spectrograms -</h1></center>
# 
# > 📌 **Competition Scope**: detect the occurrence of *onset* (the beginning of sleep) and *wakeup* (the end of sleep) in the accelerometer series.
# 
# ### Data
# 
# As the dataframes are very large (especially `train_series.parquet`) I will be using [this customized dataset](https://www.kaggle.com/datasets/andradaolteanu/detect-sleep-states-memory-decrease) I made that occupies less memory than the original ones.
# 
# The notebook that contains the steps made to create the datasets: [Zzz💤: good night sleep with 80% Memory Reduction](https://www.kaggle.com/code/andradaolteanu/zzz-good-night-sleep-with-80-memory-reduction)
# 
# > 📌 **Disclaimer**: in this notebook `series_id` has been remapped to `id_map`; as there are only 277 unique ids and the original id *took a lot of space in terms of memory* (object containing numbers and letters) I chose to remap it using simple integers for **development purposes**.
# 
# ### ○ Libraries

# In[1]:


# libraries
import os
import re
import gc
import wandb
import random
import math
from glob import glob
from tqdm import tqdm
from time import time
import warnings
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import numpy as np
from scipy.signal import spectrogram

# visuals
import seaborn as sns
import matplotlib as mpl
from matplotlib import cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
plt.rcParams.update({'font.size': 18})

# env check
warnings.filterwarnings('ignore')
os.environ["WANDB_SILENT"] = "true"
CONFIG = {'competition': '2023_sleep', '_wandb_kernel': 'aot', "source_type": "artifact"}

# color
class clr:
    S = '\033[1m' + '\033[90m'
    E = '\033[0m'
    
my_colors = ["#f79256", "#fbd1a2", "#7dcfb6", "#00b2ca"]

print(clr.S+"Notebook Color Schemes:"+clr.E)
sns.palplot(sns.color_palette(my_colors))
plt.show()


# ### 🐝 W&B Fork & Run
# 
# In order to run this notebook you will need to input your own **secret API key** within the `! wandb login $secret_value_0` line. 
# 
# 🐝**How do you get your own API key?**
# 
# Super simple! Go to **https://wandb.ai/site** -> Login -> Click on your profile in the top right corner -> Settings -> Scroll down to API keys -> copy your very own key (for more info check [this amazing notebook for ML Experiment Tracking on Kaggle](https://www.kaggle.com/ayuraj/experiment-tracking-with-weights-and-biases)).
# 
# <center><img src="https://i.imgur.com/fFccmoS.png" width=500></center>

# In[2]:


# 🐝 secrets
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("wandb")

get_ipython().system(' wandb login $secret_value_0')


# ### ○ Helper Functions Below

# In[3]:


# === data discover ===

def jitter(values,j):
    return values + np.random.normal(j,0.05,values.shape)


def find_rectangles(arr):
    '''
    return indices where the rectangle starts and ends
    '''
    rectangles = []
    start = None
    for i, val in enumerate(arr):
        if val == 3:
            if start is None:
                start = i
        elif start is not None:
            rectangles.append((start, i - 1))
            start = None
    if start is not None:
        rectangles.append((start, len(arr) - 1))
    return rectangles


def get_general_info(df, desc=None):
    
    # 🐝 new exp
    run = wandb.init(project='2023_sleep', name=f'{desc}_data_summary', config=CONFIG)

    print(clr.S+"--- General Info ---"+clr.E)
    print(clr.S+"Data Shape:"+clr.E, df.shape)
    print(clr.S+"Data Cols:"+clr.E, df.columns.tolist())
    print(clr.S+"Total No. of Cols:"+clr.E, len(df.columns.tolist()))
    print(clr.S+"No. Missing Values:"+clr.E, df.isna().sum().sum())
    print(clr.S+"Columns with missing data:"+clr.E, "\n",
          df.isna().sum()[df.isna().sum() != 0], "\n")

    for col in df.columns:
        if is_string_dtype(df[col]):
            print(clr.S+f"--- {col} --- is type string"+clr.E)
            print(clr.S+f"[nunique] {col}:"+clr.E, 
                  df[col].nunique())
        
        elif is_numeric_dtype(df[col]):
            print(clr.S+f"--- {col} --- is type numeric"+clr.E)
            print(clr.S+f"[describe] {col}:"+clr.E, "\n",
                  df[col].describe())
        
    # log data
    wandb.log
    (
        {"data_shape": len(df),
         "missing_values": df.isna().sum().sum()
        }
    )
    wandb.finish()
    print("🐝 Info saved to dashboard.")
            

def get_missing_values_plot(df):
    '''
    Plots missing values barchart for a given dataframe.
    '''
    
    # count missing values
    missing_counts = df.isnull().sum().reset_index()\
                            .sort_values(0, ascending=False)\
                            .reset_index(drop=True)
    missing_counts.columns = ["col_name", "missing_count"]

    # plot
    plt.figure(figsize=(24, 16))
    axs = sns.barplot(y=missing_counts.col_name, x=missing_counts.missing_count, 
                      color=my_colors[0])
    show_values_on_bars(axs, h_v="h", space=0.4)
    plt.xlabel('no. missing values', size=20, weight="bold")
    plt.ylabel('column name', size=20, weight="bold")
    plt.title('Missing Values', size=22, weight="bold")
    plt.show();
            
            
# === plots ===
def show_values_on_bars(axs, h_v="v", space=0.4):
    '''Plots the value at the end of the a seaborn barplot.
    axs: the ax of the plot
    h_v: weather or not the barplot is vertical/ horizontal'''
    
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(_x, _y, format(value, ','), ha="center") 
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = int(p.get_width())
                ax.text(_x, _y, format(value, ','), ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
        
        
# === 🐝 w&b ===
def save_dataset_artifact(run_name, artifact_name, path, data_type="dataset"):
    '''Saves dataset to W&B Artifactory.
    run_name: name of the experiment
    artifact_name: under what name should the dataset be stored
    path: path to the dataset'''
    
    run = wandb.init(project='2023_sleep', 
                     name=run_name, 
                     config=CONFIG)
    artifact = wandb.Artifact(name=artifact_name, 
                              type=data_type)
    artifact.add_file(path)

    wandb.log_artifact(artifact)
    wandb.finish()
    print(f"🐝Artifact {artifact_name} has been saved successfully.")
    
    
def create_wandb_plot(x_data=None, y_data=None, x_name=None, y_name=None, title=None, log=None, plot="line"):
    '''Create and save lineplot/barplot in W&B Environment.
    x_data & y_data: Pandas Series containing x & y data
    x_name & y_name: strings containing axis names
    title: title of the graph
    log: string containing name of log'''
    
    data = [[label, val] for (label, val) in zip(x_data, y_data)]
    table = wandb.Table(data=data, columns = [x_name, y_name])
    
    if plot == "line":
        wandb.log({log : wandb.plot.line(table, x_name, y_name, title=title)})
    elif plot == "bar":
        wandb.log({log : wandb.plot.bar(table, x_name, y_name, title=title)})
    elif plot == "scatter":
        wandb.log({log : wandb.plot.scatter(table, x_name, y_name, title=title)})
        
        
def create_wandb_hist(x_data=None, x_name=None, title=None, log=None):
    '''Create and save histogram in W&B Environment.
    x_data: Pandas Series containing x values
    x_name: strings containing axis name
    title: title of the graph
    log: string containing name of log'''
    
    data = [[x] for x in x_data]
    table = wandb.Table(data=data, columns=[x_name])
    wandb.log({log : wandb.plot.histogram(table, x_name, title=title)})


# In[4]:


# 🐝 log cover
run = wandb.init(project='2023_sleep', name='cover', config=CONFIG)
cover = plt.imread("/kaggle/input/detect-sleep-states-memory-decrease/de3OrxO.png")
wandb.log({"cover": wandb.Image(cover)})
wandb.finish()


# # 1. Datasets first look

# In[5]:


# events
events = pd.read_parquet("/kaggle/input/detect-sleep-states-memory-decrease/train_events.parquet")
events.head()


# In[6]:


get_general_info(df=events, desc="events")


# In[7]:


# series
series = pd.read_parquet("/kaggle/input/detect-sleep-states-memory-decrease/train_series.parquet")
series.head()


# In[8]:


get_general_info(df=series, desc="series")


# # 2. Exploratory Analysis
# 
# ## 2.1 Are people consistent when they wake up/ fall asleep
# 
# **Question**: I was curious to see if people are consistent on falling asleep every night at around the same hour. Or is the deviation high?
# 
# **Answer**: it looks like the *onset* has a much higher deviation (the hour when one person falls asleep can vary by a lot for more than half of the patients), while the *wakeup* hour is quite consistent (meaning that regularly a person wakes up at around the same hour).
# 
# This would align with the fact that people can go to sleep at any time (depending on the day), but daily activities/routines make us be more diciplined on the hour when we wake up.
# 
# *TODO: Check weekends*

# In[9]:


# 🐝
run = wandb.init(project='2023_sleep', name='eda_consistency', config=CONFIG)


# In[10]:


# get the hour
events["hour"] = events.timestamp.dt.hour
# compute the std of the hour each night per id
onset = events[events.event==1].groupby(by=["id_map"]).hour.std().reset_index()
wakeup = events[events.event==2].groupby(by=["id_map"]).hour.std().reset_index()


# In[11]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
fig.suptitle('Falling asleep/ waking up consistency', 
             weight="bold", size=25)

sns.distplot(onset.hour, rug=True, hist=False, 
             rug_kws={"color": my_colors[2]},
             kde_kws={"color": my_colors[2], "lw": 5, "alpha": 1},
             ax=ax1)
ax1.set_title('Onset Deviation', weight="bold", size=20)

sns.distplot(wakeup.hour, rug=True, hist=False, 
             rug_kws={"color": my_colors[0]},
             kde_kws={"color": my_colors[0], "lw": 5, "alpha": 1},
             ax=ax2)
ax2.set_title('Wakeup Deviation', weight="bold", size=20)

sns.despine(right=True, top=True, left=True);


# In[12]:


create_wandb_hist(x_data=onset.hour, 
                  x_name="hour",
                  title="Onset Deviation",
                  log="onset_deviation")

create_wandb_hist(x_data=wakeup.hour, 
                  x_name="hour",
                  title="Wakeup Deviation",
                  log="wakeup_deviation")


# In[13]:


wandb.finish()


# ## 2.2 Regular times for onset & wakeup
# 
# **Question**: what are the hours when people usually go to sleep or wake up?
# 
# **Answer**: I really thought this would not look like this - so observants can:
# * wake up any time between 00:00 and 14:00 (with the vast majority between 04:00 -> 10:00)
# * fall asleep between 00:00 -> 04:00 and between 19:00 -> 00:00.

# In[14]:


events["time"] = events.timestamp.dt.time
df = events.copy()
df.time = pd.to_datetime(df.time, format='%H:%M:%S')
df.set_index(['time'],inplace=True)


# In[15]:


plt.figure(figsize=(24, 12))
ax = sns.scatterplot(x=df.index, y=jitter(df["event"], 0.01), hue=df["event"], 
                     palette=[my_colors[2], my_colors[0]],
                     size=df["event"], alpha=0.7, sizes=(500, 500),
                     )

plt.title('Times for onset & wakeup', weight="bold", size=25)
ax.set(xlabel="time", ylabel="event type")
ax.xaxis.set_major_locator(mdates.HourLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.tick_params(axis="x", rotation=45)
ax.set_ylim(0, 3)
ax.legend(labels=["wakeup", "onset"]);


# In[16]:


del df
gc.collect()


# ## 2.3 Length of sleep
# 
# **Question**: Now that we know the variations in sleep and what are usually the hours, I want to see how long is usually the *longest* session of sleep (as per competition description, only the longest period has been recorded).
# 
# **Answer**: On average *between 7 and 10 hours*. We have a few quite extreme outliers when some nights they slept only ~2 hrs and some other outliers with nights when they slept 15+ hours.

# In[17]:


# 🐝
run = wandb.init(project='2023_sleep', name='eda_sleep_duration', config=CONFIG)


# In[18]:


# compute the time diff
events = events.sort_values(by=["id_map", "night"]).reset_index(drop=True)
events["time_diff"] = events.timestamp.diff()
# drop value at first night as it's not correct
events.loc[(events.night == 1) & (events.event==1), 'time_diff'] = np.nan
# select only length of sleep
time_diffs = events[events.event==2].dropna()


# In[19]:


# plot
f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(24, 15))
sns.distplot(time_diffs.time_diff.dt.components['hours'],
             rug=True, hist=False,
             rug_kws={"color": my_colors[2]},
             kde_kws={"color": my_colors[2], "lw": 5, "alpha": 0.7},
             ax=a0)

sns.boxplot(x=time_diffs.time_diff.dt.components['hours'],
            ax=a1, notch=False, showcaps=True, 
            flierprops={"marker": "x"},
            boxprops={"facecolor": my_colors[2]},
            medianprops={"color": my_colors[0]},)

plt.suptitle("How long does a sleep session take?", weight="bold", size=25)
sns.despine(right=True, top=True, left=True);


# In[20]:


create_wandb_hist(x_data=time_diffs.time_diff.dt.components['hours'], 
                  x_name="hour",
                  title="How long does a sleep session take?",
                  log="sleep_duration")


# In[21]:


wandb.finish()


# ## 2.4 enmo&anglez vs onset&waking time
# 
# **Question**: I also wanted to see how the waking and onset times look for an individual when they sleep, in regards to the features that we get from the accelerometer data (`angles` and `enmo`).
# 
# **Answer**: This is actually SO COOL! The plot below show anglez and enmo on a window in time for a few ids. The vertical shade showcases when the observant is asleep and we can see with the naked eye the difference in data during this period vs the rest of the day:
# * angles - more sparse
# * enmo - lower values

# In[22]:


def get_accelerometer_plot_data(id_map, night_max=5):
    
    # --- events ---
    # get one id_map and the dates for 5 nights
    df_e = events[(events["id_map"]==id_map) & 
                  (events["night"]>=1) & 
                  (events["night"]<=night_max)]\
            .reset_index(drop=True)
    df_e["date"] = df_e.timestamp.dt.date.astype(str)

    # --- series ---
    # filter on id_map
    df_s = series[series["id_map"]==id_map].reset_index(drop=True)
    df_s["date"] = df_s.timestamp.dt.date.astype(str)
    df_s["time"] = df_s.timestamp.dt.time
    # now retrieve only 1 day
    df_s = df_s[df_s["date"].isin(df_e.date.unique())]

    # grab event (target) info
    df_s = df_s.merge(right=events[["event", "step", "id_map"]],
                      on=["id_map", "step"], how="left")
    # fill between 1 and 2 with value 3
    # 3 meaning that the person is sleeping
    fill_value = 3
    where_value = 1
    df_s['event'] = df_s['event'].fillna(fill_value)\
                    .where(df_s['event']\
                           .fillna(method='ffill').eq(where_value) | df_s['event'].notna())
    
    return df_s


# In[23]:


id_maps = [26, 32, 100, 105, 133, 150, 190, 195]

for id_map in id_maps:
    df_s = get_accelerometer_plot_data(id_map)
    features = ["anglez", "enmo"]

    for feat in features:
        # plot
        plt.figure(figsize=(24, 12))
        plt.title(f"id [{id_map}]: {feat}", weight="bold", size=25)
        sns.lineplot(data=df_s, x="timestamp", y=feat,
                     color="#00B2CA", lw=2)
        plt.xlabel("Time", size = 18, weight="bold")
        plt.ylabel(f"{feat}", size = 18, weight="bold")
        sns.despine(right=True, top=True, left=True)

        # sleeping windows
        rectangles = find_rectangles(df_s['event'])
        for start, end in rectangles:
            plt.axvspan(df_s.loc[start, "timestamp"], 
                        df_s.loc[end, "timestamp"], 
                        color=my_colors[2], alpha=0.5)


# # 3. Spectrogram analysis
# 
# This competition can be tackled from multiple angles:
# * it can be a tabular classification problem
# * it can be an LSTM-RNN classifier using the sequential (time) data
# * it can be a computer vision problem! if we convert the time data to spectrograms :)
# 
# I always wanted to do that and this competition's data looks perfect to try.
# 
# *TODO: this area is still WIP*

# In[24]:


def create_spectrogram(value):
    
    # defines the width of each chunk in terms of samples
    # large window size -> better frequency resolution -> poor time localization
    window_size = 256 
    # ensures that each chunk has a certain number of samples 
    # that are overlapping for each chunk
    # default is 0.5*window
    overlap = 128
    # how many FFT points are desired to be computed per chunk
    # how fine-grained the frequency resolution will be
    nfft = 256 
    # sampling frequency of your signal
    # default is 1
    fs = 1.0 

    frequencies, times, Sxx = spectrogram(value, fs=fs, nperseg=window_size, 
                                          noverlap=overlap, nfft=nfft)
    return frequencies, times, Sxx


# ### What is a Spectrogram
# 
# [A spectrogram is](https://stackoverflow.com/questions/29321696/what-is-a-spectrogram-and-how-do-i-set-its-parameters) a visual representation of the Short-Time Fourier Transform. Think of this as taking chunks of an input signal and applying a local Fourier Transform on each chunk. Each chunk has a specified width and you apply a Fourier Transform to this chunk. You should take note that each chunk has an associated frequency distribution. For each chunk that is centred at a specific time point in your time signal, you get a bunch of frequency components. The collection of all of these frequency components at each chunk and plotted all together is what is essentially a spectrogram.
# 
# The spectrogram is a 2D visual heat map where the horizontal axis represents the time of the signal and the vertical axis represents the frequency axis. What is visualized is an image where darker colours means that for a particular time point and a particular frequency, the lower in magnitude the frequency component is, the darker the colour. Similarly, the higher in magnitude the frequency component is, the lighter the colour.

# In[25]:


id_maps = [26, 32, 100, 105, 133, 150, 190, 195]

for id_map in id_maps:
    spec_data = get_accelerometer_plot_data(id_map, night_max=1).drop(columns="event")
    
    features = ["anglez", "enmo"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    axs = [ax1, ax2]
    fig.suptitle(f'A night for id: {id_map}')

    for k, feat in enumerate(features):
        f, t, s = create_spectrogram(spec_data[feat].values)
        axs[k].pcolormesh(t, f, 10 * np.log10(s), cmap="RdYlBu")
        axs[k].set_xlabel('Time (s)')
        axs[k].set_ylabel('Frequency (Hz)')
        axs[k].set_title(f'Spectrogram of {feat}')


# # 4. Advanced feature engineering for time
# 
# Found this amazing module called `add_datepart` from `fastai` that enables me to create in an instant time features from my `timestamp` columns (without me having to compute them).
# 
# The new features added:
# * **date features**: 'Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'
# * **time features**: 'Hour', 'Minute', 'Second'
# 
# *TODO: add more time features like day/nigh or dusk/morning/noon/afternoon/evening/night.*

# In[26]:


get_ipython().system(' pip install fastai')

from fastai.tabular.core import add_datepart


# In[27]:


# first we need a dataframe with one single column
# the timestamp
# ! it is necessary to be THIS format for it to work
series_timestamp = pd.DataFrame(series["timestamp"])

# now we overwrite this dataframe
# and explode the timestamp into multiple features
series_timestamp = add_datepart(series_timestamp, 
                                field_name='timestamp', 
                                prefix='fast_', 
                                drop=True, 
                                time=True)

series_timestamp.head()


# In[28]:


# repeat for events dataset
events_timestamp = pd.DataFrame(events["timestamp"])

# now we overwrite this dataframe
# and explode the timestamp into multiple features
events_timestamp = add_datepart(events_timestamp, 
                                field_name='timestamp', 
                                prefix='fast_', 
                                drop=True, 
                                time=True)

events_timestamp.head()


# In[29]:


# save datasets
series_timestamp.to_parquet("train_series_time_fe.parquet", index=False)
events_timestamp.to_parquet("train_events_time_fe.parquet", index=False)


# In[30]:


# 🐝 save to W&B
save_dataset_artifact(run_name="train_series_time_fe",
                      artifact_name="train_series_time_fe",
                      path="/kaggle/input/detect-sleep-states-memory-decrease/feature_engineering/train_series_time_fe.parquet", 
                      data_type="dataset")

save_dataset_artifact(run_name="train_events_time_fe",
                      artifact_name="train_events_time_fe",
                      path="/kaggle/input/detect-sleep-states-memory-decrease/feature_engineering/train_events_time_fe.parquet", 
                      data_type="dataset")


# ### 🐝 [my W&B dash](https://wandb.ai/andrada/2023_sleep?workspace=user-andrada)
#     
# <center><img src="https://i.imgur.com/SDicyoE.png"></center>
# 
# ------
# 
# <center><img src="https://i.imgur.com/knxTRkO.png"></center>
# 
# ### My Specs
# 
# * 🖥 Z8 G4 Workstation
# * 💾 2 CPUs & 96GB Memory
# * 🎮 2x NVIDIA A6000
# * 💻 Zbook Studio G9 on the go
