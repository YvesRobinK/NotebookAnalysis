#!/usr/bin/env python
# coding: utf-8

# <p align="right">
#   <img src="https://drive.google.com/uc?export=view&id=1JydOvj55Xgv_T-7LM3jqsOVac2PhQrJ-"/>
# </p>
# 
# <a id="Title"></a>

# # <p style="background-color:#1B03A3; font-family:newtimeroman; color:white; font-size:180%; text-align:center; border-radius: 24px 0;">Bacterial Species Prediction, EDA + Model</p>
# 
# >This notebook is a walk through guide for dealing with TPS Feb 2022 competition.
# >* The **objective** of this notebook is to apply step-by-step approach to solve tabular data competition.
# >* The **subject** of this notebook is a multi-classification task, based on the idea from the following [paper](https://www.frontiersin.org/articles/10.3389/fmicb.2020.00257/full).  "Bacterial antibiotic resistance is becoming a significant health threat, and rapid identification of antibiotic-resistant bacteria is essential to save lives and reduce the spread of antibiotic resistance." Our task is to classify 10 different bacteria species using data from a genomic analysis technique that has some data compression and data loss.

# ## <p style="background-color:#1B03A3; font-family:newtimeroman; color:white; font-size:120%; text-align:center;border-radius: 24px 0;">Table of Contents</p>
# * [1. Import of Libraries](#1)
# * [2. Data Loading and Initial Visualization](#2)
# * [3. Exploratory Data Analysis](#34)
# * [4. Feature engineering](#4)
# * [6. Feature Importance](#6)
# * [5. Modeling PyTorch NN Model](#5)  
# 

# <a id='1'></a>
# # <p style="background-color:#1B03A3; font-family:newtimeroman; color:white; font-size:120%; text-align:center;border-radius: 24px 0;">1. Import of Libraries</p>

# In[1]:


import numpy as np # Linear algebra.
import pandas as pd # Data processing, CSV file I/O (e.g. pd.read_csv).
import datatable as dt # Data processing, CSV file I/O (e.g. dt.fread).

import seaborn as sns # Visualization.
import matplotlib.pyplot as plt # Visualization.

# Machine Learning block.
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import random
from tqdm import tqdm_notebook as tqdm
from collections import defaultdict
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

print(f'\n[INFO] Libraries set up has been completed.')


# <a id='2'></a>
# # <p style="background-color:#1B03A3; font-family:newtimeroman; color:white; font-size:120%; text-align:center;border-radius: 24px 0;">2. Data Loading and Initial Visualization</p>
# >
# > **Let's read the data first** (I strongly recommend using 'datatable' to for faster data reading). It reads the data 3x time faster than pandas:

# In[2]:


get_ipython().run_cell_magic('time', '', "df_train = dt.fread('../input/tabular-playground-series-feb-2022/train.csv').to_pandas()\ndf_test = dt.fread('../input/tabular-playground-series-feb-2022/test.csv').to_pandas()\ndf_sub = pd.read_csv('../input/tabular-playground-series-feb-2022/sample_submission.csv')\n\n# Datatable reads target as bool by default.\nmask_bool = df_train.dtypes == bool\nbool_train = df_train.dtypes[mask_bool].index\nbool_test = df_test.dtypes[mask_bool].index\n\ndf_train[bool_train] = df_train[bool_train].astype('int8')\ndf_test[bool_train] = df_test[bool_train].astype('int8')\n\ndf_train.info(verbose=False)\n")


# >**Let's have a sanity check if we have any missing values:**

# In[3]:


miss_val_train =df_train.isna().any().sum()
miss_val_test = df_test.isna().any().sum()

print(f'\n[INFO] {miss_val_train} missing value(s) has/have been detected in the train dataset.')
print(f'[INFO] {miss_val_test} missing value(s) has/have been detected in the test dataset.')


# > We do not really need **"row_id"** column. It will help us to reduce memory usage even more.
# > Let's fix it and cast our dtypes to the smaller ones (references: [**link**](https://www.kaggle.com/c/tabular-playground-series-dec-2021/discussion/294356)):

# In[4]:


def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum()/1024**2
    numerics = ['int8', 'int16', 'int32', 'int64',
                'float16', 'float32', 'float64']

    for col in df.columns:
        col_type = df[col].dtypes
        limit = abs(df[col]).max()

        for tp in numerics:
            cond1 = str(col_type)[0] == tp[0]
            if tp[0] == 'i': cond2 = limit <= np.iinfo(tp).max
            else: cond2 = limit <= np.finfo(tp).max

            if cond1 and cond2:
                df[col] = df[col].astype(tp)
                break

    end_mem = df.memory_usage().sum()/1024**2
    reduction = (start_mem - end_mem)*100/start_mem
    if verbose:
        print(f'[INFO] Mem. usage decreased to {end_mem:.2f}'
              f' MB {reduction:.2f}% reduction.')
    return df

target = df_train.target
df_train.drop(columns=['row_id', 'target'], inplace=True)
df_test.drop(columns='row_id', inplace=True)

df_train = reduce_mem_usage(df_train, verbose=True)
df_train['target'] = target
df_test = reduce_mem_usage(df_test, verbose=True)

print('\n')
df_train.head(5)


# In[5]:


rc = {
    "axes.facecolor":"#FFF2D9",
    "figure.facecolor":"#FFF2D9",
    "axes.edgecolor":"#383838",
    "axes.spines.right" : False,
    "axes.spines.top" : False,
}

sns.set(rc=rc)

df_target_count = df_train.target.value_counts()
s1 = df_target_count[:3]
s2 = pd.Series(sum(df_target_count[3:]), index=["rest"])
s3 = s1.append(s2)

f, axes = plt.subplots(ncols=2, figsize=(15, 4))
plt.subplots_adjust(wspace=0)

outer_sizes = s3
inner_sizes = s3/4
outer_colors = ['#1B03A3', '#1B03A3', '#1B03A3', '#0D0151']
inner_colors = ['#6A02A3', '#6A02A3', '#6A02A3']

axes[0].pie(
    outer_sizes,colors=outer_colors, 
    labels=s3.index.tolist(), 
    startangle=90,frame=True, radius=1.3, 
    explode=(.05,.05,.05,.5),
    wedgeprops={ 'linewidth' : 1, 'edgecolor' : 'white'}, 
    textprops={'fontsize': 12, 'weight': 'bold'}
)

textprops = {
    'size':13, 
    'weight': 'bold', 
    'color':'white'
}

axes[0].pie(
    inner_sizes, colors=inner_colors,
    radius=1, startangle=90,
    autopct='%1.f%%',explode=(.1,.1,.1, -.5),
    pctdistance=0.8, textprops=textprops
)

center_circle = plt.Circle((0,0), .68, color='black', 
                           fc='#FFF2D9', linewidth=0)
axes[0].add_artist(center_circle)

x = df_target_count
y = df_target_count.index.astype(str)
sns.barplot(
    x=x, y=y, ax=axes[1],
    color='#1B03A3', orient='horizontal'
)

axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].tick_params(
    axis='x',         
    which='both',      
    bottom=False,      
    labelbottom=False
)

for i, v in enumerate(df_target_count):
    axes[1].text(v, i+0.1, str(v), color='black', 
                 fontweight='bold', fontsize=12)
 
plt.tight_layout()    
plt.show()


# > Good news: **`df_train`** is **class-balanced**.
# >
# > **Next**, let's get 30000 samples and plot it:

# In[6]:


seed = 322
df_train_sample = df_train.sample(n=30000, random_state=seed)
df_test_sample = df_test.sample(n=30000, random_state=seed)

np.random.seed(seed) 
features_choice = np.random.choice(
    df_train_sample.keys()[1:-1], size=3, replace=False
)

mask = sorted(features_choice.tolist()) + ['target']
df_sample_three = df_train_sample[mask]
df_sample_three.head(3)


# In[7]:


fig, ax = plt.subplots(nrows=3, figsize=(24, 24))

for i, feature in enumerate(sorted(features_choice)):
     sns.scatterplot(
         ax=ax[i], x=df_sample_three.index,
         y=feature,data=df_sample_three,
         hue='target',palette='magma',
         legend=True,
     )


# > **Basic Statistics Train Set head(5) + tail(5)**:

# In[8]:


pd.set_option('display.float_format', lambda x: '%.2f' % x)
descr_tr = df_train.iloc[:, :-1].describe().T\
                     .sort_values(by='std', ascending=False)


pd.concat([descr_tr.iloc[:5,:], descr_tr.iloc[-5:,:]])\
                     .style.background_gradient(cmap='magma')\
                     .bar(subset=["mean",], color='green')\
                     .bar(subset=["max"], color='#BB0000')


# In[9]:


descr_ts = df_test.describe().T\
                     .sort_values(by='std', ascending=False)


pd.concat([descr_ts.iloc[:5,:], descr_ts.iloc[-5:,:]])\
                     .style.background_gradient(cmap='magma')\
                     .bar(subset=["mean",], color='green')\
                     .bar(subset=["max"], color='#BB0000')


# > Some observations from the tables above:
# > * Head and Tail of **`df_train`** and **`df_test`** share the statistics with negligible difference.
# > It is nice to have train and test dataset from the same distribution. It's been a while since we were introduced to Tabular Dataset of such quality.

# <a href="#Title" role="button" aria-pressed="true" >Back to the beginning 🔙</a>

# <a id='3'></a>
# # <p style="background-color:#1B03A3; font-family:newtimeroman; color:white; font-size:120%; text-align:center;border-radius: 24px 0;">3. Exploratory Data Analysis</p>
# >
# > Let's take a closer look at the distribution of the features:

# In[10]:


get_ipython().run_cell_magic('time', '', "figsize = (6*6, 6*6)\nfig = plt.figure(figsize=figsize)\ntitle = 'Probability Density Function Estimation'\nfor idx, col in enumerate(df_test.columns[:20]):\n    ax = plt.subplot(4, 5, idx + 1)\n    sns.kdeplot(\n        data=df_train_sample, hue='target', fill=True,\n        x=col, palette='cividis'\n    )\n            \n    ax.set_ylabel(''); ax.spines['top'].set_visible(False), \n    ax.set_xlabel(''); ax.spines['right'].set_visible(False)\n    ax.set_title(f'{col}', loc='right', \n                 weight='bold', fontsize=10)\n\nfig.supxlabel(f'\\n\\n{title} Train\\n\\n', ha='center', \n              fontweight='bold', fontsize=30)\nplt.tight_layout()\nplt.show()\n\nfig = plt.figure(figsize=figsize)\nfor idx, col in enumerate(df_test.columns[:20]):\n    ax = plt.subplot(4, 5, idx + 1)\n    sns.kdeplot(\n    data=df_train_sample, fill=True,\n    x=col, color='#1B03A3', label='Train'\n    )\n    sns.kdeplot(\n        data=df_test_sample, fill=False,\n        x=col, color='#E54232', label='Test'\n    )\n\n    ax.set_xticks([]); ax.set_xlabel(''); \n    ax.set_ylabel(''); ax.spines['right'].set_visible(False)\n    ax.set_yticks([]); ax.spines['top'].set_visible(False)\n    ax.set_title(f'{col}', loc='right', \n                 weight='bold', fontsize=10)\n    \nfig.supxlabel(f'\\n\\n{title} Train vs Test set', ha='center', \n              fontweight='bold', fontsize=30)\n       \nplt.tight_layout()\nplt.show()\n")


# > **We have plotted Probability Density Function estimation for each feature. What does it tell us?**
# >* The features are distributed differently;
# >* The data is not perfectly symmetrical. The most of the features right-skewed.
# >* There is no bell-shaped-like (e.g., Gaussian distribution) plots.
# >* The plot supports the assumption that the train and test data are from the same distribution.

# > Let's us take a look at features correlation matrix:

# In[11]:


corr_ = df_train_sample.corr().abs()

fig, axes = plt.subplots(figsize=(20, 10))
mask1 = np.zeros_like(corr_)
mask1[np.triu_indices_from(mask1)] = True
sns.heatmap(corr_, mask=mask1, linewidths=.5, cmap='magma_r')

plt.show()


# > If we wish to label the strength of the features association, for absolute values of correlation, 0-0.19 is regarded as very weak (the most of our examples are: [0.00-0.20].

# 
# ## <p style="background-color:#1B03A3; font-family:newtimeroman; color:white; font-size:120%; text-align:center;border-radius: 24px 0;">Highly Correlated Features</p>
# >
# > Let's zoom in and take a closer look at the highly correlated pairs of features, taking an arbitrary threshold of correlation as 0.7:

# In[12]:


# Fill df diagonal with zeros
np.fill_diagonal(corr_.values, 0)
pivot = corr_.unstack()
corr_pairs = pivot.sort_values(kind="quicksort", ascending=False)
high_corr_pairs = corr_pairs[corr_pairs > .7]
pd.DataFrame(high_corr_pairs[::2], columns=['corr']).head(3)


# In[13]:


fig, axes = plt.subplots(figsize=(10, 6))
df_hcorr_pairs = pd.DataFrame(high_corr_pairs).unstack()
sns.heatmap(df_hcorr_pairs, linewidths=.5, cmap='magma_r')
plt.show()


# **Assumption**: 
# >
#     * We can use this knowledge for the future feature engineering.
#     * We need to check feature importance of the correlated pairs separately (preferably).

# <a href="#Title" role="button" aria-pressed="true" >Back to the beginning 🔙</a>

# <a id='4'></a>
# # <p style="background-color:#1B03A3; font-family:newtimeroman; color:white; font-size:120%; text-align:center;border-radius: 24px 0;">4. Feature engineering</p>
# 
# > [**Feature engineering**](https://www.omnisci.com/technical-glossary/feature-engineering#:~:text=Feature%20engineering%20refers%20to%20the,machine%20learning%20or%20statistical%20modeling.) refers to the process of using domain knowledge to select and transform the most relevant variables from raw data.
# >
# > One of the naive approach to engineer features, is to aggregate them.  
# >
# > The problem with aggregation is that we might encounter **multicollinearity** (e.g., the high correlation of the explanatory variables). "It should be noted that the presence of multicollinearity does not mean that the model is
# misspecified. You only start to talk about it when you think that it is
# affecting the regression results seriously." [[1]](#9.1)
# 

# ## <p style="background-color:#1B03A3; font-family:newtimeroman; color:white; font-size:120%; text-align:center;border-radius: 24px 0;">Naive approach</p>

# In[14]:


agg_features = ['sum','mean','std','max','min']
features = df_test.columns
for ft in agg_features:
    
    class_method = getattr(pd.DataFrame, ft)
    df_train_sample[ft] = class_method(df_train_sample[features], axis=1)
    df_test_sample[ft] = class_method(df_test_sample[features], axis=1)

df_test_sample.head(3)


# ## <p style="background-color:#1B03A3; font-family:newtimeroman; color:white; font-size:120%; text-align:center;border-radius: 24px 0;">Greedy Elimination</p>

# 
# The idea of this approach is to eliminate highly correlated features with respect to their pairs. 
# The features will be eliminated based on their feature importance iteratively during the training.
# >
# The complete list of the features to be iteratively eliminated:

# In[15]:


pairs = pd.DataFrame(high_corr_pairs[::2], columns=['corr'])
pairs


# ## <p style="background-color:#1B03A3; font-family:newtimeroman; color:white; font-size:120%; text-align:center;border-radius: 24px 0;">Correlated Pairs Stack and Elimination</p>

# In[16]:


def stack_elimination(train, test, pairs, elim=False):
    """

    Creates combined mean feature based on the pair.
    Takes highly correlated feature pairs and eliminate them.
    
    :param train: (pd.DataFrame)
    :param test: (pd.DataFrame)
    :param pairs: (multiIndex pd.Dataframe)
    :param elim: bool
    :return: train (pd.DataFrame), test (pd.DataFrame)
    """
    
    
    for i, pair in enumerate(pairs.index): 
        
       
        # Creates combined mean feature.
        train[f'pair{i}'] = train[list(pair)].mean(axis=1)
        test[f'pair{i}'] = test[list(pair)].mean(axis=1)

    if elim:
        # Eliminates the paired features.       
        flat_pairs_list = [j for i in pairs.index for j in i]
        ft_train = train.columns
        ft_test = test.columns
        diff_train = set(ft_train).difference(flat_pairs_list)
        diff_test = set(ft_test).difference(flat_pairs_list)
        train = train[sorted(diff_train)]
        test = test[sorted(diff_test)]
        
    return train, test

train, test = stack_elimination(df_train_sample, df_test_sample, pairs)
train.head(3)


# <a id='5'></a>
# # <p style="background-color:#1B03A3; font-family:newtimeroman; color:white; font-size:120%; text-align:center;border-radius: 24px 0;">5. PyTorch NN Model</p>
# 
# > **Versions notes**:
# > * Version1: Model hidden layers + Batch_Norm [300, BN, 200, BN, 100, BN, 50 BN].
# 
# > **Things to try**:
# > * Feature Engineering.
# > * Add evaluation plots and confusion matrices.
# 

# <a id='5.1'></a>
# ## <p style="background-color:#1B03A3; font-family:newtimeroman; color:white; font-size:120%; text-align:left;border-radius: 24px 0px;">__5.1 Reload Data</p>

# In[17]:


train_csv_path = '../input/tabular-playground-series-feb-2022/train.csv'
test_csv_path = '../input/tabular-playground-series-feb-2022/test.csv'

train = dt.fread(train_csv_path).to_pandas()
test = dt.fread(test_csv_path).to_pandas()

# Encode target labels with value between 0 and n_classes.
le = LabelEncoder()
target = le.fit_transform(train.target)

col_drop=['row_id']    
if col_drop:
    train.drop(columns=col_drop, inplace=True)
    test.drop(columns=col_drop, inplace=True)
    print(f'\n[INFO] "Id" columns have been removed successfully.\n')

# Applies encoded target.
train = train.iloc[:, :-1]
train['target'] = target
train.head()


# <a id='5.2'></a>
# ## <p style="background-color:#1B03A3; font-family:newtimeroman; color:white; font-size:120%; text-align:left;border-radius: 24px 0px;">__5.2 Dataset</p>

# In[18]:


class TabularDataset(Dataset):
    def __init__(self, x, y):
        """
        Defines PyTorch dataset.
        :param x: np.ndarray
        :param y: np.ndarray
        """

        self.len = x.shape[0]
        self.x = torch.Tensor(x).float()
        self.y = torch.LongTensor(y).long().flatten()

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len


# <a id='5.3'></a>
# ## <p style="background-color:#1B03A3; font-family:newtimeroman; color:white; font-size:120%; text-align:left;border-radius: 24px 0px;">__5.3 Model and weights initialization</p>

# In[19]:


class Model(nn.Module):
    def __init__(self, in_features, num_cls):
        super().__init__()

        self.fc1 = nn.Linear(in_features, 300)
        self.bn1 = nn.BatchNorm1d(300)
        self.fc2 = nn.Linear(300, 200)
        self.bn2 = nn.BatchNorm1d(200)
        self.fc3 = nn.Linear(200, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.fc4 = nn.Linear(100, 50)
        self.bn4 = nn.BatchNorm1d(50)
        self.fc_out = nn.Linear(50, num_cls)

        self.activation = nn.ReLU()
        self.classifier = nn.Sigmoid()

    def forward(self, x):

        x = self.activation(self.fc1(x))
        x = self.bn1(x)
        x = self.activation(self.fc2(x))
        x = self.bn2(x)
        x = self.activation(self.fc3(x))
        x = self.bn3(x)
        x = self.activation(self.fc4(x))
        x = self.bn4(x)
        x = self.fc_out(x)

        return x

def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight.data)


# <a id='5.4'></a>
# ## <p style="background-color:#1B03A3; font-family:newtimeroman; color:white; font-size:120%; text-align:left;border-radius: 24px 0px;">__5.4 Device and model summary</p>

# In[20]:


train.iloc[:, :-1].shape[1]


# In[21]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Available device: {device}.\n\n")

n_ft = train.iloc[:, :-1].shape[1]
n_cls = len(set(target))
model = Model(in_features=n_ft, num_cls=n_cls).to(torch.device(device))

try:
    from torchsummary import summary
except:
    print("Installing Torchsummary..........")
    get_ipython().system(' pip install torchsummary -q')
    from torchsummary import summary
    
summary(model, (n_ft,))


# <a id='5.5'></a>
# ## <p style="background-color:#1B03A3; font-family:newtimeroman; color:white; font-size:120%; text-align:left;border-radius: 24px 0px;">__5.5 Utils</p>

# In[22]:


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def set_seed(seed):
    """
    Fixes seed for the reproducible results.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# <a id='5.6'></a>
# ## <p style="background-color:#1B03A3; font-family:newtimeroman; color:white; font-size:120%; text-align:left;border-radius: 24px 0px;">__5.6 Train and valid loops with tqdm bar</p>

# In[23]:


def train_loop(train_loader, model, criterion, optimizer, epoch, device):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    for i, (x, y) in enumerate(stream, start=1):
        features = x.to(device)
        target = y.to(device)
        output = model(features)
        loss = criterion(output, target)
        acc_train = accuracy(output, target)
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", acc_train[0].item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        desc = "Epoch: {epoch}. Train.      {metric_monitor}"
        stream.set_description(
          desc.format(epoch=epoch, metric_monitor=metric_monitor)
        )
    
    loss_avg = metric_monitor.metrics["Loss"]['avg']
    acc_avg = metric_monitor.metrics["Accuracy"]['avg']

    return loss_avg, acc_avg


def val_loop(val_loader, model, criterion, epoch, device):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (x, y) in enumerate(stream, start=1):
            features = x.to(device)
            target = y.to(device)
            output = model(features)
            loss = criterion(output, target)
            acc_val = accuracy(output, target)

            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", acc_val[0].item())
            desc = "Epoch: {epoch}. Validation.      {metric_monitor}"
            stream.set_description(
                desc.format(epoch=epoch, metric_monitor=metric_monitor)
            )
            
    loss_avg = metric_monitor.metrics["Loss"]['avg']
    acc_avg = metric_monitor.metrics["Accuracy"]['avg']

    return loss_avg, acc_avg


# <a id='5.7'></a>
# ## <p style="background-color:#1B03A3; font-family:newtimeroman; color:white; font-size:120%; text-align:left;border-radius: 24px 0px;">__5.7 CFG</p>

# In[24]:


param = {
        'seed': 1,
        'nfold': 10,
        'lr': 9e-5,
        'wd': 1e-5,
        'plateau_factor': .5,
        'plateau_patience': 4,
        'batch': 1024,
        'epochs': 40,
        'early_stopping': 9
    }


# <a id='5.8'></a>
# ## <p style="background-color:#1B03A3; font-family:newtimeroman; color:white; font-size:120%; text-align:left;border-radius: 24px 0px;">__5.8 Training Main</p>

# In[25]:


n_ft = train.iloc[:, :-1].shape[1]
n_cls = len(set(target))

X = train.iloc[:, :-1].values
y = train.iloc[:, -1:].values

# StratifiedKfold data split.
skf = StratifiedKFold(
    n_splits=param['nfold'],
    shuffle=True,
    random_state=param['seed']
)
    

for fold, (idx_train, idx_val) in enumerate(skf.split(X, y)):
    
    if fold == 1:
        break
    # Model, weights and seed init.
    model = Model(in_features=n_ft, num_cls=n_cls)
    model.apply(init_weights)
    model = model.to(torch.device(device))
    set_seed(param['seed'])

    # Loss and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=param['lr'],
        weight_decay=param['wd']
    )

    scheduler = ReduceLROnPlateau(
                optimizer=optimizer,
                factor=param['plateau_factor'],
                patience=param['plateau_patience'],
                mode='max', verbose=True
            )
    
   
    scaler = StandardScaler()
    X_train, y_train = scaler.fit_transform(X[idx_train, :]), y[idx_train]
    X_val, y_val = scaler.transform(X[idx_val, :]), y[idx_val]
    print(
        f'\n[INFO] Fold: {fold+1}, '
        f'X_train shape: {X_train.shape}, '
        f'X_val shape: {X_val.shape}.\n'
    )

    trainset = TabularDataset(X_train, y_train)
    valset = TabularDataset(X_val, y_val)
    train_loader = DataLoader(trainset, batch_size=param['batch'], shuffle=True)
    val_loader = DataLoader(valset, batch_size=param['batch'], shuffle=True)
    wait_counter = 0
    valid_acc_best = 0
    
    for epoch in range(1, param['epochs'] + 1):
        train_loss, train_acc = train_loop(train_loader, model, criterion, optimizer, epoch, device)
        valid_loss, valid_acc = val_loop(val_loader, model, criterion, epoch, device)
        
        if valid_acc > valid_acc_best:
            valid_acc_best = valid_acc
            wait_counter = 0
            best_model = deepcopy(model)
            print(f'\n[INFO] The best model has been saved.\n')
        else:
            wait_counter += 1
            if wait_counter > param['early_stopping']:
                print(f"\n[INFO] There's been no improvement "
                      f"in val_acc. Early stopping has been invoked.")
                break


# In[26]:


X_test = scaler.transform(test)
testset = TabularDataset(X_test, np.ones((X_test.shape[0], 1)))
test_loader = DataLoader(testset, batch_size=1024)
y_pred_list = []

best_model.eval()
with torch.no_grad():
    for X_batch, _ in tqdm(test_loader):
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch.float())
        _, y_pred_tags = torch.max(y_test_pred, dim=1)
        y_pred_list.extend(y_pred_tags.cpu().numpy())


# Rearranges classes back (e.g. [0,1,2,3,4,5] -> arr of strings;
# Creates mapped test_y preds list.
test_y = le.inverse_transform(y_pred_list)
            
df_sub['target'] = test_y
df_sub.to_csv('submission.csv', index=False)
df_sub  


# <a href="#Title" role="button" aria-pressed="true" >Back to the beginning 🔙</a>

# <a id=''></a>
# # <p style="background-color:#1B03A3; font-family:newtimeroman; color:white; font-size:120%; text-align:center;border-radius: 24px 0px;">Work in progress...</p>
# 

# <a id=''></a>
# # <p style="background-color:#1B03A3; font-family:newtimeroman; color:white; font-size:120%; text-align:center;border-radius: 24px 0;">Any suggestions to improve this notebook will be greatly appreciated. P/s If I have forgotten to reference someone's work, please, do not hesitate to leave your comment. Any questions, suggestions or complaints are most welcome. Upvotes keep me motivated... Thank you.</p>
# 

# <a href="#Title" role="button" aria-pressed="true" >Back to the beginning 🔙</a>
