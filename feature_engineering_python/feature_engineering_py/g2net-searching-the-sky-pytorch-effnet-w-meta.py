#!/usr/bin/env python
# coding: utf-8

# In[1]:


# --- CSS STYLE ---
from IPython.core.display import HTML
def css_styling():
    styles = open("../input/2020-cost-of-living/alerts.css", "r").read()
    return HTML("<style>"+styles+"</style>")
css_styling()


# <img src="https://i.imgur.com/k8NA44c.png">
# 
# <center><h1>🌌 Searching the Sky - Explore & Understand 🌌</h1></center>
# 
# # 1. Introduction
# 
# Uiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii!
# 
# I was hoping to have some fun with a notebook. Haven't done some proper artistic EDA in a while, and the theme of this competition is absolute perfection.
# 
# <div class="alert simple-alert">
# 🚀 <b>Competition Goal</b>: detect GW <i>(Gravitational Wave)</i> signals from the mergers of binary black holes from simulated GW time-series data, created from a network of Earth-based detectors.
# </div>
# 
# 💜 Let's get started!
# 
# ### Libraries ⬇

# In[2]:


get_ipython().system('pip install -q nnAudio -qq')

# Libraries
import os
import re
import gc
import wandb
import time
from tqdm import tqdm
import glob
import pickle
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from pylab import text
import torch
from nnAudio.Spectrogram import CQT1992v2

# Librosa
import librosa
from librosa.feature import melspectrogram
import librosa.display

# Environment check
warnings.filterwarnings("ignore")
os.environ["WANDB_SILENT"] = "true"
CONFIG = {'competition': 'g2net', '_wandb_kernel': 'aot'}

# Secrets 🤫
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("wandb")

# Custom colors
class color:
    S = '\033[1m' + '\033[93m'
    E = '\033[0m'
    
my_colors = ["#E7C84B", "#4EE4EA", "#4EA9EA", "#242179", "#AB51E9", "#E051E9"]
print(color.S+"Notebook Color Scheme:"+color.E)
sns.palplot(sns.color_palette(my_colors))

# Set Style
sns.set_style("white")
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.spines.left'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
plt.rcParams.update({'font.size': 17})


# > 🚀 **Note**: If this line throws an error, try using wandb.login() instead. It will ask for the API key to login, which you can get from your [W&B profile](https://wandb.ai/site) (click on Profile -> Settings -> scroll to API keys).
# 
# ***You can find my W&B Dashboard here -> https://wandb.ai/andrada/g2net?workspace=user-andrada***

# In[3]:


get_ipython().system(' wandb login $secret_value_0')


# ### Custom Functions ⬇

# In[4]:


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
        
         
def offset_png(x, y, path, ax, zoom, offset):
    '''For adding other .png images to the graph.
    source: https://stackoverflow.com/questions/61971090/how-can-i-add-images-to-bars-in-axes-matplotlib'''
    
    img = plt.imread(f"../input/g2net-gravitational-wave-dataset/pngs/{path}.png")
    im = OffsetImage(img, zoom=zoom)
    im.image.axes = ax
    x_offset = offset
    ab = AnnotationBbox(im, (x, y), xybox=(x_offset, 0), frameon=False,
                        xycoords='data', boxcoords="offset points", pad=0)
    ax.add_artist(ab)
    
    
def save_dataset_artifact(run_name, artifact_name, path):
    '''Saves dataset to W&B Artifactory.
    run_name: name of the experiment
    artifact_name: under what name should the dataset be stored
    path: path to the dataset'''
    
    run = wandb.init(project='g2net', 
                     name=run_name, 
                     config=CONFIG, anonymous="allow")
    artifact = wandb.Artifact(name=artifact_name, 
                              type='dataset')
    artifact.add_file(path)

    wandb.log_artifact(artifact)
    wandb.finish()
    print("Artifact has been saved successfully.")
    
    
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


# # 2. 🛫 The Data
# 
# The `training_labels.csv` file contains the file id and the `target`, meaning a flag that is:
# * 0: if there is no signal
# * 1: is there is any signal

# In[5]:


# Read in the training data
train = pd.read_csv("../input/g2net-gravitational-wave-detection/training_labels.csv")

# Print some useful information
print(color.S+"Train Data has:"+color.E, "{:,}".format(train.shape[0]), "observations.", "\n" +
      color.S+"Number of Missing Values:"+color.E, train.isna().sum()[0], "\n" +
      "\n" +
      color.S+"Head of Training Data:"+color.E)
train.head(5)


# In[6]:


# Save data to W&B Dashboard
save_dataset_artifact(run_name='save-training_labels',
                      artifact_name='training_labels', 
                      path="../input/g2net-gravitational-wave-detection/training_labels.csv")


# ## 2.1 The Target - is there a black hole?
# 
# > 🚀 **Note**: The targets are splitted almost 50% - 50%. This is because the **data itself is simulated**, so there's the benefit that you can purposely simulate a black hole as many times as you want. However, as the description tells us, signals of black holes are **very rare**.

# In[7]:


run = wandb.init(project='g2net', name='explore', config=CONFIG, anonymous="allow")


# In[8]:


plt.figure(figsize=(20, 12))
ax = sns.countplot(data=train, y="target", palette=my_colors)

show_values_on_bars(ax, h_v="h", space=0.4)
ax.set_xlabel("Frequency", size = 22)
ax.set_ylabel("Target", size = 22)
ax.set_title("- Frequency of Target variable -", size = 26, weight='bold')
plt.yticks(ticks=[0, 1], labels=["signal not present", "signal present"])
plt.xticks([])
sns.despine(left=True, bottom=True)

offset_png(x=243000, y=1, path="black_hole", ax=ax, zoom=0.3, offset=0)


# In[9]:


# Create W&B Plot
create_wandb_plot(x_data=["signal not present", "signal present"], 
                  y_data=train["target"].value_counts().values, 
                  x_name="Target", y_name="Frequency", 
                  title="- Frequency of Target variable -", 
                  log="target_plot", plot="bar")


# In[10]:


# Add info about total number of observations
wandb.log({"total_obs" : np.int(train.shape[0])})


# ## 2.2 The .npy files
# 
# > 🚀 **Note**: The **simulated GW** (Gravitational Waves) are coming from 3 different Observatories:
# * LIGO Hanford: below in purple
# * LIGO Livingston: below in yellow
# * VIRGO: below in green
# 
# <center><img src="https://i.imgur.com/IJZyBGJ.jpg" width=900></center>

# In[11]:


# Get the full paths to the files and create a df
paths = glob.glob("../input/g2net-gravitational-wave-detection/train/*/*/*/*")
ids = [path.split("/")[-1].split(".")[0] for path in paths]
paths_df = pd.DataFrame({"path":paths, "id": ids})

# Append the full path as a new column
train_df = pd.merge(left=train, right=paths_df, on="id")

print(color.S+"train_df:"+color.E)
train_df.head(5)


# OK!
# 
# Each file has a shape of **`(3, 4096)`** - meaning 3 different GW coming from the 3 sites around the globe, of a length of 4096. The length of 4096 spans for 2 seconds and it is sampled at 2,048 Hz.
# 
# > 🚀 **Note**: Keep in mind this data is becoming pretty big. We have **560,000 observation x 3 sites x 4,096 time series length** => **6,881,280,000** (that's 6 billion datapoints ... with a B)
# 
# Hence, I will rename these 3 as `Site1`, `Site2` and `Site3`, like the one and only [Heads or Tails](https://www.kaggle.com/headsortails) did in his notebook [right here](https://www.kaggle.com/headsortails/when-stars-collide-g2net-eda).

# In[12]:


def get_npy_df(path):
    '''Returns a df of the 3 site information for a particular file.
    path: a string containing the full path to the file'''
    
    df = pd.DataFrame({"Site1" : np.load(path)[0],
                       "Site2" : np.load(path)[1],
                       "Site3" : np.load(path)[2]})
    
    return df


# # 3. 👩‍🚀 Explore ...
# 
# ## 3.1 The Gravitational Waves
# 
# Let's take a look at the Gravitational Waves and see what insights we can find about them, before starting creating an actual model.
# 
# ### 🚀 GW when there is **NO** signal present:
# 
# The 3 sites have fairly similar distribution, with the third one having fewer outliers than the rest.

# In[13]:


# Get a sample data with TARGET == 0
no_target = list(train_df.loc[train_df["target"] == 0, "path"])[23]
no_target = get_npy_df(path = no_target)

# Plot
fig = plt.figure(figsize=(20, 12))
outer = gridspec.GridSpec(1, 3, wspace=0.2, hspace=0.2)
fig.suptitle('- GW Fluctuation: Target = 0 -', size = 26, weight='bold')
sites = ["Site1", "Site2", "Site3"]
colors = [my_colors[2], my_colors[3], my_colors[4]]
peaks = [8e19, 6.2e19, 2.3e20]
pngs = ["blue", "yellow", "pink"]
size = [0.08, 0.07, 0.05]

for i, site, col, p, png, s in zip(range(3), sites, colors, peaks, pngs, size):
    inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                                             subplot_spec=outer[i], 
                                             wspace=0.1, hspace=0.1,
                                             height_ratios= (.15, .85))
    ax1 = plt.Subplot(fig, inner[0])
    ax2 = plt.Subplot(fig, inner[1])
    mean = no_target[site].mean()
    
    sns.boxplot(no_target[site], ax=ax1, color=col)
    sns.kdeplot(data=no_target, x=site, ax=ax2, color=col, shade=True, 
                lw=2, alpha=0.5)
    ax2.axvline(x=mean, color=col, lw=3, ls="--")
    ax2.text(x=mean, y=p, s=f'{mean}', size=13, color=col, weight='bold')
    
    fig.add_subplot(ax1)
    fig.add_subplot(ax2)
    ax1.set(xlabel='')
    axs = [ax1, ax2]
    for ax in axs:
        ax.set_xticks([])
        ax.set_ylabel("")
    sns.despine(bottom=True, left=True)
    offset_png(x=mean, y=p/5, path=png, ax=ax2, zoom=s, offset=0)


# ### 🚀 GW when there **IS** a signal present:
# 
# The distributions look similar, however there is some more fluctuation at the peak of density and and the extremes, especially for Site 2.

# In[14]:


# Get a sample data with TARGET == 1
no_target = list(train_df.loc[train_df["target"] == 1, "path"])[23]
no_target = get_npy_df(path = no_target)

# Plot
fig = plt.figure(figsize=(20, 12))
outer = gridspec.GridSpec(1, 3, wspace=0.2, hspace=0.2)
fig.suptitle('- GW Fluctuation: Target = 1 -', size = 26, weight='bold')
sites = ["Site1", "Site2", "Site3"]
colors = [my_colors[0], my_colors[1], my_colors[2]]
peaks = [6.5e19, 9e19, 2.3e20]
pngs = ["newborn", "earth", "blue2"]
size = [0.02, 0.045, 0.045]

for i, site, col, p, png, s in zip(range(3), sites, colors, peaks, pngs, size):
    inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                                             subplot_spec=outer[i], 
                                             wspace=0.1, hspace=0.1,
                                             height_ratios= (.15, .85))
    ax1 = plt.Subplot(fig, inner[0])
    ax2 = plt.Subplot(fig, inner[1])
    mean = no_target[site].mean()
    
    sns.boxplot(no_target[site], ax=ax1, color=col)
    sns.kdeplot(data=no_target, x=site, ax=ax2, color=col, shade=True, 
                lw=2, alpha=0.5)
    ax2.axvline(x=mean, color=col, lw=3, ls="--")
    ax2.text(x=mean, y=p, s=f'{mean}', size=13, color=col, weight='bold')
    
    fig.add_subplot(ax1)
    fig.add_subplot(ax2)
    ax1.set(xlabel='')
    axs = [ax1, ax2]
    for ax in axs:
        ax.set_xticks([])
        ax.set_ylabel("")
    sns.despine(bottom=True, left=True)
    offset_png(x=mean, y=p/5, path=png, ax=ax2, zoom=s, offset=0)


# ## 3.2 Signals in time
# 
# Ok, now we can look at how these waves look in time, comparing the 3 sites we already know and are familiar with: LIGO, Hanford, LIGO Livingston and VIRGO.
# 
# > 🚀 **Note**: There are **indeed** some differences, that can be seen a bit more clear than by looking only at histograms. The signals with no target have bigger fluctuations, while the other ones have smaller more consistent ones. However, these diferences are **very tiny**.

# In[15]:


# Target Sample
with_target = list(train_df.loc[train_df["target"] == 1, "path"])[23]
with_target = get_npy_df(path = with_target)
# No Target Sample
no_target = list(train_df.loc[train_df["target"] == 0, "path"])[23]
no_target = get_npy_df(path = no_target)
# size = [6.5e19, 9e19, 2.3e20]

# Plot
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(20, 12), sharey=True, sharex=True)
fig.suptitle('- GW Signals in Time -', size = 26, weight='bold')

sns.lineplot(y=with_target["Site1"], x=range(len(with_target)), ax=ax1,
             lw=3, color=my_colors[1])
sns.lineplot(y=no_target["Site1"], x=range(len(no_target)), ax=ax2,
             lw=3, color=my_colors[2])
sns.lineplot(y=with_target["Site2"], x=range(len(with_target)), ax=ax3,
             lw=3, color=my_colors[5])
sns.lineplot(y=no_target["Site2"], x=range(len(no_target)), ax=ax4,
             lw=3, color=my_colors[4])
sns.lineplot(y=with_target["Site3"], x=range(len(with_target)), ax=ax5,
             lw=3, color=my_colors[0])
sns.lineplot(y=no_target["Site3"], x=range(len(no_target)), ax=ax6,
             lw=3, color=my_colors[3])

ax1.title.set_text('With Target')
ax2.title.set_text('No Target');

# Images
offset_png(x=2700, y=2e-20, path="astronaut", ax=ax2, zoom=0.15, offset=0)
offset_png(x=1700, y=1.7e-20, path="satellite", ax=ax3, zoom=0.2, offset=0)
offset_png(x=500, y=-1.3e-20, path="spaceshut", ax=ax6, zoom=0.065, offset=0)


# ##  3.3 The MEL Spectrogram
# 
# 🚀 **What is a Spectrogram?** - A spectrogram is a **visual representation** of the spectrum of frequencies of a signal as it varies with *time*.
# 
# 🚀 **What is a Mel Spectrogram?** - A mel spectrogram is a spectrogram where the **frequencies are converted to the mel scale**.
# 
# 🚀 **Why should we use it?** - In sound processing, the mel-frequency cepstrum (MFC) is a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency. This frequency warping *can allow for better representation of sound*.

# In[16]:


def make_spectrogram(path, prints=False):
    '''Creates a MEL spectrogram.'''
    
    # Get the waves from the 3 sites
    waves = np.load(path).astype(np.float32)
    if prints:
        print(color.S+"Waves Shape:"+color.E, waves.shape)
    
    # Loop and make spectrogram
    spectrograms = []
    
    for i in range(3):
        # Compute a mel-scaled spectrogram.
        spec = melspectrogram(waves[i] / max(waves[i]), sr=4096, 
                              n_mels=128, fmin=20, fmax=2048)
        # Convert a power spectrogram (amplitude squared) to decibel (dB) units
        spec = librosa.power_to_db(spec).transpose((1, 0))
        spectrograms.append(spec)
        
    return spectrograms


# ### Sample
# 
# First let's look at how the function above works on a simple sample from the data.

# In[17]:


path = train_df["path"][0]

# Get the spectrogram
spectrogram = make_spectrogram(path, prints=True)
    
# Plot it
img = np.vstack(spectrogram)

plt.figure(figsize=(22, 10))
plt.title('Sample Mel Spectrogram', size = 20, weight='bold')
plt.imshow(img, cmap="cool")
plt.axis("off");


# ### 🌍 GW Signals Spectrogram - With Target vs No Target
# 
# Good! Looks nice! Now we can do a proper comparison between a few samples that contain the Target Signal vs samples that don't.
# 
# > 🚀 **Note**: You can notice that it is very **hard to observe any kind of difference** between the images, as the fluctuation is so unperceptable by the naked eye.

# In[18]:


# Samples per category
n=3

# Sample 6 paths with target and no target available
paths_no_target = train_df[train_df["target"] == 0]["path"].sample(n, random_state=23).values
paths_with_target = train_df[train_df["target"] == 1]["path"].sample(n, random_state=23).values

all_paths = np.append(paths_no_target, paths_with_target)

# Plot
fig, axes = plt.subplots(nrows=2, ncols=n, figsize=(21,5))
wandb_logs = []

# Enumerate & plot
for i, path in enumerate(all_paths):
    if i < n: title = "No Target" 
    else: title="With Target"
    
    spec = make_spectrogram(path, prints=False)
    img = np.vstack(spec)
    
    x = i // n
    y = i % n
    
    axes[x, y].imshow(img, cmap="cool")
    axes[x, y].set_title(title)
    axes[x, y].axis('off');
    
    # Save to W&B
    wandb_logs.append(wandb.Image(img, caption=f"{title}_{i}"))
    
    
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.07, hspace=0.0)
wandb.log({"spectrograms": wandb_logs})


# ## 3.4 nnAudio()
# 
# **`nnAudio`**: is an audio processing toolbox using `PyTorch` CNN as its backend. By doing so, spectrograms can be generated from audio on-the-fly during neural network training and the Fourier kernels (e.g. or CQT kernels) can be trained ([more info on this pachage here](https://github.com/KinWaiCheuk/nnAudio)).
# 
# <div class="alert simple-alert">
# 🚀 Special thanks to <b>Y.Nakama</b> and <a href="https://www.kaggle.com/yasufuminakama/g2net-efficientnet-b7-baseline-training">his notebook here</a>, from where I took my inspiration to use this library.
# </div>
# 
# > 🚀 **Note**: This function will be used to quickly and efficiently convert the signals from the 3 sites to spectrograms.

# In[19]:


def create_nnAudio_graph(path, title=None):
    '''The full path to an numpy array.'''
    
    plt.figure(figsize=(21,5))

    # This function is to calculate the CQT of the input signal.
    file_ex = np.load(path)
    TRANSFORM = CQT1992v2(sr=2048, fmin=20, fmax=1024, hop_length=32)
    titles = ["Hanford", "Livingston", "Virgo"]

    for i in range(3):
        waves = file_ex[i] / np.max(file_ex[i])
        waves = torch.from_numpy(waves).float()
        image = TRANSFORM(waves)

        plt.subplot(1, 3, i + 1)
        plt.suptitle(title)
        plt.imshow(image.squeeze(), cmap="cool")
        plt.title(titles[i], fontsize=20)
        plt.axis('off');


# In[20]:


path_no_target = train_df[train_df["target"] == 0]["path"].sample(1, random_state=23).values[0]
create_nnAudio_graph(path=path_no_target,
                     title="Sample Spectrograms - No Target -")


# In[21]:


path_with_target = train_df[train_df["target"] == 1]["path"].sample(1, random_state=22).values[0]
create_nnAudio_graph(path=path_with_target,
                     title="Sample Spectrograms - With Target -")


# # 4. 🛸 Basic Feature Engineering & Site Comparisons
# 
# Now let's see some **differences/similarities** between our 3 main sites: *LIGO Hanford, LIGO Livingston* and *VIRGO*.
# 
# > To do that, we are going to take some **basic metrics** and compute them for each observation and site:
# * `mean()`
# * `std()`
# * `var()`
# * `min()`
# * `mode()`
# * `max()`

# In[22]:


def get_site_metrics(df):
    '''Compute for each id the metrics for each site.
    df: the complete df'''
    
    # List of all metrics we want to compute for each site
    sites = ["Site1", "Site2", "Site3"]
    metrics = ["mean", "std", "var", "minim", "maxim", "mode"]

    # Create empty columns of the metrics
    for site in sites:
        for metric in metrics:
            df[f"{site}_{metric}"] = 0

            
    # Compute for each ID these metrics
    for ID, path in tqdm(zip(df["id"].values, df["path"].values)):

        # First extract the cronological info
        info = get_npy_df(path = path)

        # For each site compute the metrics
        for site in sites:
            mean = info[site].mean()
            std = info[site].std()
            var = info[site].var()
            minim = info[site].min()
            maxim = info[site].max()
            mode = info[site].mode()

            # Add it to the dataframe
            df.loc[df["id"] == ID, f"{site}_mean"] = mean
            df.loc[df["id"] == ID, f"{site}_std"] = std
            df.loc[df["id"] == ID, f"{site}_var"] = var
            df.loc[df["id"] == ID, f"{site}_minim"] = minim
            df.loc[df["id"] == ID, f"{site}_maxim"] = maxim
            df.loc[df["id"] == ID, f"{site}_mode"] = mode
            
    return df


# In[23]:


# Process the entire data
# This took a while and cannot be done in the Kaggle Environment
# So I made it locally
# processed = get_site_metrics(df=train_df)
# processed.to_csv("training_labels_features.csv", index=False)


# In[24]:


# Import the data with basic features
train_fe = pd.read_csv("../input/g2net-gravitational-wave-dataset/training_labels_features.csv")
print(color.S + "train with FE: " + color.E, train_fe.shape)
train_fe.head(3)

# Save data to W&B Dashboard
save_dataset_artifact(run_name='save-training_fe',
                      artifact_name='training_fe', 
                      path="../input/g2net-gravitational-wave-dataset/training_labels_features.csv")


# ## 4.1 In depth analysis
# 
# ### Overall Means
# 
# > 🚀 **Note**: We can now explore the means of all observations per site and the differences between them. Besides the bigger values between Site1, Site2 vs Site3, the distributions look very similar and uniform.

# In[25]:


# Plot
fig = plt.figure(figsize=(20, 12))
outer = gridspec.GridSpec(1, 3, wspace=0.2, hspace=0.2)
fig.suptitle('- GW Per Site: All Data -', size = 26, weight='bold')
sites = ["Site1", "Site2", "Site3"]
colors = [my_colors[3], my_colors[4], my_colors[5]]
peaks = [5.66e21, 5.62e21, 2.3e22]
pngs = ["moon2", "mars", "neutral"]
size = [0.1, 0.07, 0.08]

for i, site, col, p, png, s in zip(range(3), sites, colors, peaks, pngs, size):
    inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                                             subplot_spec=outer[i], 
                                             wspace=0.1, hspace=0.1,
                                             height_ratios= (.15, .85))
    ax1 = plt.Subplot(fig, inner[0])
    ax2 = plt.Subplot(fig, inner[1])
    mean = train_fe[f"{site}_mean"].mean()
    
    sns.boxplot(train_fe[f"{site}_mean"], ax=ax1, color=col)
    sns.kdeplot(x=train_fe[f"{site}_mean"], ax=ax2, color=col, shade=True, 
                lw=2, alpha=0.5)
    ax2.axvline(x=mean, color=col, lw=3, ls="--")
    ax2.text(x=mean, y=p, s=f'{mean}', size=13, color=col, weight='bold')
    
    fig.add_subplot(ax1)
    fig.add_subplot(ax2)
    ax1.set(xlabel='')
    axs = [ax1, ax2]
    for ax in axs:
        ax.set_xticks([])
        ax.set_ylabel("")
    sns.despine(bottom=True, left=True)
    offset_png(x=mean, y=p/5, path=png, ax=ax2, zoom=s, offset=0)


# ### Overall Minim and Maxim

# In[26]:


# Separate minim & maxim values
minims = train_fe["Site1_minim"].sort_values(ascending=False).reset_index(drop=True)
maxims = train_fe["Site1_maxim"].sort_values(ascending=True).reset_index(drop=True)

minims = pd.DataFrame({"val":minims, "Category": "minim", "range":range(len(minims))})
maxims = pd.DataFrame({"val":maxims, "Category": "maxim", "range":range(len(maxims))})

data = pd.concat([minims, maxims]).reset_index(drop=True)


# Plot
plt.figure(figsize=(21, 12))
plt.title('- Minim & Maxim per each Observation: All Data -', size = 26, weight='bold')

plot = sns.lineplot(data=data, y=data["val"].sort_values(), x=data["range"], hue="Category", 
                    sort=False, lw=7, palette="cool", style="Category")
plot.legend(["Minim", "Maxim"], loc="center right", title="Category:")
plot.set_xticks([])
plot.set_ylabel("");

# Images
offset_png(x=2700, y=0, path="milkyway", ax=plot, zoom=0.35, offset=0)


# In[27]:


# End this experiment
wandb.finish()


# <center><img src="https://i.imgur.com/MAerCzs.png"></center>
# 
# <center><h1>🌌 PyTorch EffNet Model + Feature Metadata 🌌</h1></center>
# 
# ### Competition Metric
# 
# > 🚀 **AUC - ROC curve**: is a performance measurement for the *classification problems* at various threshold settings. **ROC is a probability curve** and **AUC represents the degree or measure of separability**. *Higher the AUC, the better the model is at predicting the classes*.
# 
# Below is a sample example: the goal is tu have the "Area Under the Curve" (AUC) be as big as possible - meaning that the line should aim to be as closer to the X and Y axis as possible.

# In[28]:


# Libraries
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score 

# Generate Sample Dataset
X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2)

# "no skill" prediction
ns_probs = [0 for _ in range(len(y_test))]
# Fit Ensemble
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict probabilities
rf_probs = model.predict_proba(X_test)
rf_probs = rf_probs[:, 1]

# Comparison
ns_auc = roc_auc_score(y_test, ns_probs)
rf_auc = roc_auc_score(y_test, rf_probs)

print(color.S+'No Skill: ROC AUC=%.3f' % (ns_auc)+color.E)
print(color.S+'Random Forest: ROC AUC=%.3f' % (rf_auc)+color.E)

# Plot
fig, ax = plt.subplots(figsize=(21, 10))
plt.title('- Example of ROC AUC -', size = 26, weight='bold')
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)

ax.plot(ns_fpr, ns_tpr, ls="dotted", label='NoSkill', lw=6, color=my_colors[0])
ax.plot(rf_fpr, rf_tpr, ls="dashdot", label='RandomForest', lw=6, color=my_colors[4])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend();

offset_png(x=0.88, y=0.3, path="astronaut2", ax=ax, zoom=0.35, offset=0)


# ### ⬇ More Libraries & Functions

# In[29]:


get_ipython().system('pip install efficientnet_pytorch -qq')

import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from efficientnet_pytorch import EfficientNet

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import model_selection as sk_model_selection

# Set 
def set_seed(seed = 23):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device available now:', device)


# ~~~~~~~~~~~~~~~~~~~
# ~~~~~FUNCTIONS~~~~~
# ~~~~~~~~~~~~~~~~~~~
def plot_loss_graph(train_losses, valid_losses, epoch, fold):
    '''Lineplot of the training/validation losses.'''
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2.5))
    fig.suptitle(f"Fold {fold} | Epoch {epoch}", fontsize=12, y=1.05)
    axes = [ax1, ax2]
    data = [train_losses, valid_losses]
    sns.lineplot(y=train_losses, x=range(len(train_losses)),
                 lw=2.3, ls=":", color=my_colors[3], ax=ax1)
    sns.lineplot(y=valid_losses, x=range(len(valid_losses)),
                 lw=2.3, ls="-", color=my_colors[5], ax=ax2)
    for ax, t, d in zip(axes, ["Train", "Valid"], data):
        ax.set_title(f"{t} Evolution", size=12, weight='bold')
        ax.set_xlabel("Iteration", weight='bold', size=9)
        ax.set_ylabel("Loss", weight='bold', size=9)
        ax.tick_params(labelsize=9)
    plt.show()
    
    
def get_auc_score(valid_preds, valid_targets, gpu=True):
    '''Compute ROC AUC score.'''
    if gpu:
        predictions = torch.cat(valid_preds).cpu().detach().numpy().tolist()
    else:
        predictions = torch.cat(valid_preds).detach().numpy().tolist()
    actuals = [int(x) for x in valid_targets]

    roc_auc = roc_auc_score(actuals, predictions)
    return roc_auc


# In[30]:


train = pd.read_csv("../input/g2net-gravitational-wave-dataset/training_labels_features.csv")


# # 5. 👨‍🚀 PyTorch Dataset
# 
# First we must create the `PyTorch Dataset`, which will be a class that will take the paths and targets, compute the numpy arrays' spectrograms and return the result.
# 
# This class is also helpful within the `Dataloader` tool, so we can iterate through multiple files at once.
# 
# > 🚀 **Bonus**: I added the features from the 3 sites too on a later iteration - now we can use the additional information for better serults. ;)

# In[31]:


class G2Dataset(Dataset):
    
    def __init__(self, path, features, target=None, test=False, prints=False):
        '''Initiate the arguments & import the numpy file/s.'''
        self.path = path
        self.features = features
        self.target = target
        self.test = test
        self.prints = prints
        
    def __len__(self):
        return len(self.path)
    
    def __transform__(self, np_file):
        '''Transforms the np_file into spectrogram.'''
        spectrogram = []
        TRANSFORM = CQT1992v2(sr=2048, fmin=20, 
                              fmax=1024, hop_length=32, 
                              verbose=False)
        
        # Create an image with 3 channels - for the 3 sites
        for i in range(3):
            waves = np_file[i] / np.max(np_file[i])
            waves = torch.from_numpy(waves).float()
            channel = TRANSFORM(waves).squeeze().numpy()
            spectrogram.append(channel)
            
        spectrogram = torch.tensor(spectrogram).float()
        
        if self.prints:
            plt.figure(figsize=(5, 5))
            plot = spectrogram.detach().cpu().numpy()
            plot = np.transpose(plot, (1, 2, 0))
            plt.imshow(plot)
            plt.axis("off")
            plt.show();

        return spectrogram
    
    def __getitem__(self, i):
        
        # Load the numpy file
        np_file = np.load(self.path[i])
        # Create the spectrograms
        spectrograms = self.__transform__(np_file)
        # Select the features
        metadata = np.array(self.features.iloc[i].values, dtype=np.float32)
        
        # Return the images & target if available
        if self.test==False:
            y = torch.tensor(self.target[i], dtype=torch.float)
            return {"spectrogram": spectrograms,
                    "metadata": metadata,
                    "targets": y}
        else:
            return {"spectrogram": spectrograms,
                    "metadata": metadata}


# ### ~ Test the Dataset function ~
# 
# Good! Now that we've created our `Dataset` class, we can test it by using a simple sample of 4 observations:
# * 4 distinct paths pointing to the numpy arrays (split in 2 batches of size 2)
# * 4 distinct targets, which are the labels of these paths
# 
# > 🚀 **Note**: I am also lotting here the 3 channel spectrogram that is created from the 3 sites: hence, we're making 1 image with 3 channels, instead of 3 images with only 1 channel. This is how they look!
# 
# <center><img src="https://i.imgur.com/IHHFq75.png" width=900></center>

# In[32]:


# Sample
path = train["path"][:4].values
features = train.iloc[:4, 3:]
target = train["target"][:4].values

# Initiate the Dataset
dataset = G2Dataset(path=path, target=target, features=features,
                    test=False, prints=True)

# Initiate the Dataloader
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

# Output of the Dataloader
for k, data in enumerate(dataloader):
    spectrograms, features, targets = data.values()
    print(color.S + f"Batch: {k}" + color.E, "\n" +
          color.S + "Spectrograms:" + color.E, spectrograms.shape, "\n" +
          color.S + f"Features:" + color.E, features.shape, "\n" +
          color.S + "Target:" + color.E, targets, "\n" +
          "="*50)


# # 6. 🌑 PyTorch EfficientNet
# 
# Now we need to create a `Module` class which will help us take the output from the `Dataset` class and train it to predict out target variable.

# In[33]:


class G2EffNet(nn.Module):
    
    def __init__(self, no_features, no_neurons=250):
        super().__init__()
        
        # NN for the spectrogram - out layer = 2560
        self.spectrogram = EfficientNet.from_pretrained('efficientnet-b7')
        
        # NN for the features
        self.metadata = nn.Sequential(nn.Linear(no_features, no_neurons),
                                      nn.BatchNorm1d(no_neurons),
                                      nn.ReLU(),
                                      nn.Dropout(p=0.2),
                                      
                                      nn.Linear(no_neurons, no_neurons),
                                      nn.BatchNorm1d(no_neurons),
                                      nn.ReLU(),
                                      nn.Dropout(p=0.2))
        
        # Final NN for classification
        # Combination of spectrogram + features
        self.classification = nn.Sequential(nn.Linear(2560 + no_neurons, 1))
        
    def forward(self, spectrogram, features, prints=False):
        
        if prints: print(color.S+'Spectrogram In:'+color.E, spectrogram.shape, '\n'+
                         color.S+'Features In:'+color.E, features.shape, '\n' +
                         '='*40)
        
        # Spectrogram
        spectrogram = self.spectrogram.extract_features(spectrogram)
        if prints: print(color.S+'Spectrogram Out:'+color.E, spectrogram.shape)
            
        spectrogram = F.avg_pool2d(spectrogram, spectrogram.size()[2:]).reshape(-1, 2560)
        if prints: print(color.S+'Spectrogram Reshaped:'+color.E, spectrogram.shape)
            
        # Features
        features = self.metadata(features)
        if prints: print(color.S+'Features Out:'+color.E, features.shape)
            
        # Combine Layers
        concatenated = torch.cat((spectrogram, features), dim=1)
        out = self.classification(concatenated)
        if prints: print(color.S+'Concat shape:'+color.E, concatenated.shape, "\n" + 
                         color.S+'Out shape:'+color.E, out.shape)
        
        return torch.sigmoid(out)


# ### ~ How it works? ~
# 
# Goooood! :) Let's see how it works! Below is a schema to help you better grasp how the model works:
# 
# <center><img src="https://i.imgur.com/Kir64Dy.png" width=800></center>

# In[34]:


# Create an example model - Effnet
model_example = G2EffNet(no_features=15, no_neurons=250)


# In[35]:


# We'll use previous datasets & dataloader
# example for 1 batch
for k, data in enumerate(dataloader):
    spectrograms, features, targets = data.values()
    break
    
# Outputs
out = model_example(spectrograms, features, prints=True)

# Criterion
criterion_example = nn.BCEWithLogitsLoss()
# Unsqueeze(1) from shape=[3] => shape=[3, 1]
loss = criterion_example(out, targets.unsqueeze(1))   
print(color.S+'LOSS:'+color.E, loss.item())


# # 7. 🌠 Training ...
# 
# ## 7.1 Training Function
# 
# Usually this part can get quite long and weird; this is why I usually choose to visualize it with a schema, so I can better know at a later date what I did here.
# 
# 🚀 **As a summary**:
# 
# * First we initiate a new **W&B experiment**, where we store all the hyperparameters we'll be using - this way we know how to reproduce everything afterwards.
# * Then we split the data into folds
# * For each fold:
#     * We initiate a `G2Dataset()`, the model, loss criterion, optimizer etc.
#     * We start the training loop (epochs):
#         * train on the training data (`model.train()`), we compute the loss and then optimize
#         * evaluate how the model did (`model.eval()`)
#         * Compute a `roc_auc` score and, if better than the last one, we save the model
# * Repeat
# 
# <center><img src="https://i.imgur.com/6Pme9rZ.png" width = 800></center>
# 
# ### Full Training Function below ⬇

# In[36]:


def train_effnet(name, epochs, splits, batch_size, no_neurons, lr, weight_decay, sample):

    # === W&B Experiment ===
    s = time.time()
    params = dict(model=name, epochs=epochs, split=splits, 
                  batch=batch_size, neurons=no_neurons, 
                  lr=lr, weight_decay=weight_decay, sample=sample)
    CONFIG.update(params)
    run = wandb.init(project='g2net', name=f"effnet_{name}", config=CONFIG, anonymous="allow")


    # === CV Split ===
    df = train.sample(sample, random_state=23)
    cv = StratifiedKFold(n_splits=splits)
    cv_splits = cv.split(X=df, y=df['target'].values)



    for fold, (train_i, valid_i) in enumerate(cv_splits):

        print("~"*25)
        print("~"*8, color.S+f"FOLD {fold}"+color.E, "~"*8)
        print("~"*25)

        train_df = df.iloc[train_i, :]
        # To go quicker through validation
        valid_df = df.iloc[valid_i, :].sample(int(sample*(splits/10)*0.6),
                                              random_state=23)

        # Datasets & Dataloader
        train_dataset = G2Dataset(path=train_df["path"].values, target=train_df["target"].values,
                                  features=train_df.iloc[:, 3:], test=False)
        valid_dataset = G2Dataset(path=valid_df["path"].values, target=valid_df["target"].values,
                                  features=valid_df.iloc[:, 3:], test=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        # Model/ Optimizer/ Criterion/ Scheduler
        model = G2EffNet(no_features=15, no_neurons=no_neurons).to(device)
        optimizer = Adam(model.parameters(), lr=lr, 
                         weight_decay=weight_decay, amsgrad=False)
        criterion = nn.BCEWithLogitsLoss()
        # scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', verbose=True,
        #                               patience=VAR.patience, factor=VAR.factor)
        scaler = GradScaler()

        # ~~~~~~~~~~~~
        # ~~~ LOOP ~~~
        # ~~~~~~~~~~~~
        BEST_SCORE = 0.0

        for epoch in range(epochs):
            print("="*8, color.S+f"Epoch {epoch}"+color.E, "="*8)

            # === TRAIN ===
            model.train()
            train_losses = []
            for k, data in enumerate(train_loader):
                spectrograms, features, targets = data.values()
                spectrograms, features, targets = spectrograms.to(device), features.to(device), targets.to(device)

                with autocast():
                    out = model(spectrograms, features)
                    loss = criterion(out, targets.unsqueeze(1))
                    train_losses.append(loss.cpu().detach().numpy().tolist())

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            mean_train_loss = np.mean(train_losses)
            print(color.S+"Mean Train Loss:"+color.E, mean_train_loss)
            wandb.log({"mean_train_loss": np.float(mean_train_loss)}, step=epoch)


            # === EVAL ===
            model.eval()
            valid_losses, valid_preds, valid_targets = [], [], []
            with torch.no_grad():
                for k, data in enumerate(valid_loader):
                    spectrograms, features, targets = data.values()
                    valid_targets.extend(targets.detach().numpy().tolist())
                    spectrograms, features, targets = spectrograms.to(device), features.to(device), targets.to(device)

                    out = model(spectrograms, features)

                    valid_preds.extend(out)
                    loss = criterion(out, targets.unsqueeze(1))
                    valid_losses.append(loss.cpu().detach().numpy().tolist())

            mean_valid_loss = np.mean(valid_losses)
            print(color.S+"Mean Valid Loss:"+color.E, mean_valid_loss)
            wandb.log({"mean_valid_loss": np.float(mean_valid_loss)}, step=epoch)
            plot_loss_graph(train_losses, valid_losses, epoch, fold)


            # === UPDATES ===
            roc_auc = get_auc_score(valid_preds, valid_targets, gpu=torch.cuda.is_available())
            print(color.S+"ROC AUC:"+color.E, roc_auc)
            wandb.log({"roc_auc": np.float(roc_auc)}, step=epoch)

            if roc_auc > BEST_SCORE:        
                print("! Saving model in fold {} | epoch {} ...".format(fold, epoch), "\n")
                torch.save(model.state_dict(), f"Baseline_fold_{fold}_auc_{round(roc_auc, 5)}.pt")

                BEST_SCORE = roc_auc


        del model, optimizer, criterion, spectrograms, features, targets
        torch.cuda.empty_cache()
        gc.collect()

    wandb.finish()
    print(color.S+f"Time to run: {round((time.time() - s)/60, 2)} minutes"+color.E)


# ## 7.2 Experiments

# In[37]:


# class VAR:
#     name = "60k_samples"
#     splits = 3
#     epochs = 2
#     batch_size = 64
#     no_neurons = 250
#     lr = 0.0001
#     weight_decay = 0.000001
#     patience = 1
#     factor = 0.01
#     sample=60000
    
    
# train_effnet(name=VAR.name, epochs=VAR.epochs, splits=VAR.splits, 
#              batch_size=VAR.batch_size, no_neurons=VAR.no_neurons, lr=VAR.lr, 
#              weight_decay=VAR.weight_decay, sample=VAR.sample)


# In[38]:


# === TEST CELL - runs faster ===
class VAR:
    name = "test"
    splits = 3
    epochs = 2
    batch_size = 64
    no_neurons = 250
    lr = 0.0001
    weight_decay = 0.000001
    patience = 1
    factor = 0.01
    sample=2000
    
    
train_effnet(name=VAR.name, epochs=VAR.epochs, splits=VAR.splits, 
             batch_size=VAR.batch_size, no_neurons=VAR.no_neurons, lr=VAR.lr, 
             weight_decay=VAR.weight_decay, sample=VAR.sample)


# ### W&B Dashboard
# 
# You can check the evolution of the experiments here -> https://wandb.ai/andrada/g2net?workspace=user-andrada
# 
# > 🌠 Below is a *sneak peak* of the dashboard:
# <center><video src="https://i.imgur.com/9JN5eUq.mp4" width=700 controls></center>

# # 8. 🪐 Submission
# 
# We're at the end of the line folks!
# 
# I've put here a simple submission code for this notebook.
# 
# 🚀 **Steps to submission**:
# * Retrieve the pretrained model/s
# * Create a new `Dataset` & `Dataloader` - careful here; you don't have the target anymore
# * Predict using the trained models
# * Blend the predictions if you want into a final output
# * Submit
# * Have a snack, you're done 💜

# In[39]:


# Sample submission containing extracted features
test = pd.read_csv("../input/g2net-gravitational-wave-dataset/sample_submission_features.csv")
test = test.head(20) ### SMALLER TO RUN FASTER - ERASE LINE TO GET FULL SUBMISSION


# In[40]:


# Retrieve all pretrained models
names = ["Baseline_fold_0_auc_0.79091", "Baseline_fold_1_auc_0.78462",
         "Baseline_fold_2_auc_0.7886"]
models = []

for i in range(len(names)):
    model = G2EffNet(no_features=15, no_neurons=250).to(device)
    model.load_state_dict(torch.load(f"../input/g2net-gravitational-wave-dataset/{names[i]}.pt",
                                     map_location=torch.device(device)))
    model.eval()
    models.append(model)


# In[41]:


# Test Dataset & Dataloader
dataset = G2Dataset(path=test["path"].values, target=None,
                    features=test.iloc[:, 3:], test=True)
dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

# === Loop ===
all_preds = []

# Disable gradients
with torch.no_grad():
    for k, data in enumerate(dataloader):
        
        spectrograms, features = data.values()
        spectrograms, features = spectrograms.to(device), features.to(device)
        
        # Predict with each of the 3 models
        out0 = models[0](spectrograms, features).cpu().numpy().squeeze()
        out1 = models[1](spectrograms, features).cpu().numpy().squeeze()
        out2 = models[2](spectrograms, features).cpu().numpy().squeeze()
        
        # Blend the predictions
        all_preds.extend((out0 + out1 + out2)/3)


# ### Submission
# 
# > 🚀 **Note**: For the purpose of this notebook running faster, I'll make the prediction on only the first 20 observations within the test data. Delete `test = test.head(20)` line to get the full prediction; which is save within my G2net dataset as well. :)

# In[42]:


ss = pd.read_csv("../input/g2net-gravitational-wave-detection/sample_submission.csv").head(20)
ss["target"] = all_preds

ss.head()


# In[43]:


actual_submission = pd.read_csv("../input/g2net-gravitational-wave-dataset/60k_submission.csv")
actual_submission.to_csv("60k_submission.csv", index=False)


# <center><img src="https://i.imgur.com/LzL1Srh.png" width=700></center>

# In[44]:


# TODO: make KFold Validation - DONE
# TODO: Change graph representation - DONE
# TODO: create train() function - DONE
# TODO: create schema for train() function - DONE
# TODO: preprocess submission data - DONE
# TODO: train on more data & submit with model - DONE
# TODO: dataset add spectrogram augmentation


# <img src="https://i.imgur.com/cUQXtS7.png">
# 
# # My Specs
# 
# * 🖥 **Z8 G4** Workstation
# * 💾 2 CPUs & 96GB Memory
# * 🎮 **NVIDIA** Quadro RTX 8000
# * 💻 **Zbook** Studio G7 on the go
