#!/usr/bin/env python
# coding: utf-8

# Published on September 20, 2023. By Marília Prata, mpwolke

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# #Competition Citation
# 
# @misc{optiver-trading-at-the-close,
# 
#     author = {Tom Forbes, John Macgillivray, Matteo Pietrobon, Sohier Dane, Maggie Demkin},
#     title = {Optiver - Trading At The Close},
#     
#     publisher = {Kaggle},
#     
#     year = {2023},
#     
#     url = {https://kaggle.com/competitions/optiver-trading-at-the-close}
# }

# #Kaggle Project: Optiver-Realized-Volatility-Prediction
# 
# "In financial markets, volatility captures the amount of fluctuation in prices. For trading firms like Optiver, accurately predicting volatility is essential for the trading of options, whose price is directly related to the volatility of the underlying product.
# 
# "In this Kaggle competition,we had built models that predict short-term volatility for hundreds of stocks across different sectors. Our models will be evaluated against real market data collected in the three-month evaluation period after training."
# 
# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSNi_wcsJ4zQ2zCCtdWeNZMA_B4LeG_WzI-ymK75mt_N9dMNF5N6Y_Hwlsx74jYN1OfYzU&usqp=CAU)https://github.com/Taher-web-dev/Optiver-Realized-Volatility-Prediction

# #Market Volatility and its role on Global Economic Narrative.
# 
# "Stock exchanges are fast-paced, high-stakes environments where every second counts. The intensity escalates as the trading day approaches its end, peaking in the critical final ten minutes. These moments, often characterised by heightened volatility and rapid price fluctuations, play a pivotal role in shaping the global economic narrative for the day."
# 
# "In the last ten minutes of the Nasdaq exchange trading session, market makers like Optiver merge traditional order book data with auction book data. This ability to consolidate information from both sources is critical for providing the best prices to all market participants."
# 
# https://www.kaggle.com/competitions/optiver-trading-at-the-close/overview

# In[2]:


train = pd.read_csv('/kaggle/input/optiver-trading-at-the-close/train.csv')
test =  pd.read_csv('/kaggle/input/optiver-trading-at-the-close/example_test_files/test.csv')
rev =  pd.read_csv('/kaggle/input/optiver-trading-at-the-close/example_test_files/revealed_targets.csv')
sample_submission = pd.read_csv('/kaggle/input/optiver-trading-at-the-close/example_test_files/sample_submission.csv')


# #Bid/Ask Prices
# 
# bid/ask_price - Price of the most competitive buy/sell level in the non-auction book.
# 
# bid/ask_size - The dollar notional amount on the most competitive buy/sell level in the non-auction book.
# 
# https://www.kaggle.com/competitions/optiver-trading-at-the-close/data

# In[6]:


train.tail()


# #Volatility Matters
# 
# Global Market Volatility Hurts Everyone. Trade Law Can Help. 
# By Jeffrey Kucik on March 23, 2023
# 
# "Economists have recognized trade volatility’s dangers for decades. Unexpected trade shocks—to imports or exports—can harm all levels of the market. For countries, volatility drives up government budget deficits, amplifies business cycles, and deters inward investment. Sustained volatility can be so toxic that it reduces long-term growth."
# 
# "All those “macro” effects matter for workers. Firms in traded industries rely on predictability to make business decisions. Yet, market shocks complicate economic planning."
# 
# "Some market shocks are impossible to prevent. However, political unrest and war also contribute to instability. Seemingly isolated domestic conflicts can generate instability that radiates outward through the marketplace to business partners around the world."
# 
# "The most common cause of trade volatility is policy volatility. For example, when countries introduce new protectionist barriers, they disrupt trade relations in ways that may generate more volatility, not less."
# 
# "Volatility will never disappear entirely from the marketplace, and countries need domestic policy solutions to address free trade’s uneven impact on local communities. But, while social safety nets at home might help extinguish the fires volatility lights, effective trade rules can prevent the match from ever being struck."
# 
# https://www.wilsoncenter.org/article/global-market-volatility-hurts-everyone-trade-law-can-help

# In[7]:


test.head()


# In[8]:


rev.head()


# #Visualizing trading sessions
# 
# "Let's visualise some of the trading data for a few different stocks and sessions. We will randomly pick stock_id and time_id values so we can run the cell multiple times to start to get a sense of the dataset."
# 
# "We will use relplots from seaborn, where the size of the bubbles indicates the number of shares traded and the colour of the bubble indicates the number of orders traded at that time period."

# In[11]:


#By Sam Maule https://www.kaggle.com/code/semaule/eda-visualising-trading-data-optiver

from random import sample

# Get some randomly sampled time_ids and stock_ids
time_id_sample = sample(list(train["time_id"].unique()), 3)
stock_id_sample = sample(list(train["stock_id"].unique()), 5)

mask = (train["time_id"].isin(time_id_sample)) & (train["stock_id"].isin(stock_id_sample))
train_sample = train[mask]

# Create relplot on the subset of data
sns.relplot(x="seconds_in_bucket", y="bid_price", hue="target", size="bid_size",
            col="time_id", row="stock_id", sizes=(40,400), data=train_sample);


# In[12]:


#By Sam Maule https://www.kaggle.com/code/semaule/eda-visualising-trading-data-optiver

# Group on stock_id and get the mean and standard deviation of price, order_count and size
grouped_trade = train.groupby(["stock_id"]).agg({"bid_price": ["mean", "std", "count"],  # Count here gives the number of trades
                                                    "target": ["mean", "std", "sum"],
                                                    "bid_size": ["mean", "std", "sum"]}).reset_index()


# In[13]:


grouped_trade.hist(figsize=(15, 15));


# In[14]:


grouped_trade.describe()


# #WAP
# 
# "We'll compute the weighted average price and the log return. Since the log return formula uses diff we should apply it to each stock / time_id individually."

# In[22]:


#By Sam Maule https://www.kaggle.com/code/semaule/eda-visualising-trading-data-optiver

# WAP based on the most competitive bid / ask prices
train['wap'] = (train['bid_price'] * train['ask_size'] + train['ask_price'] * train['bid_size']) / \
                 (train['bid_size']+ train['ask_size'])


# #Compute log return
# 
# "The normalization happens at the beginning of each time bucket by scaling the WAP to 1, so small deviations in price can be explained because of it."
# 
# By Matteo Pietrobon https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/249474#1400435

# In[ ]:


#By Sam Maule https://www.kaggle.com/code/semaule/eda-visualising-trading-data-optiver

# Compute the log return for each unique stock_id and time_id pairing
# NOTE: this takes some time to run
updated_data = []
stock_time_pairs = train.groupby(["time_id", "stock_id"]).size().reset_index().sort_values(["stock_id", "time_id"])
for i, row in stock_time_pairs.iterrows():
    if i % 2000 == 0:
        print(f"Completed {i} of {stock_time_pairs.shape[0]} rows.")
    mask = (train["stock_id"] == row["stock_id"]) & (train["time_id"] == row["time_id"])
    subset = train[mask].copy(deep=True)
    subset['log_return'] = np.log(subset['wap']).diff()
    updated_data.append(subset)
    
train = pd.concat(updated_data)


# #Above, I gave up that Never Ending list of rows and tried Gunes Evitan perfect charts.

# In[3]:


df_train = pd.read_csv('/kaggle/input/optiver-trading-at-the-close/train.csv')


# In[4]:


#By Gunes Evitan https://www.kaggle.com/code/gunesevitan/optiver-realized-volatility-prediction-eda#8.-Time-Buckets

def visualize_target(target):
    
    print(f'{target}\n{"-" * len(target)}')
        
    print(f'Mean: {df_train[target].mean():.4f}  -  Median: {df_train[target].median():.4f}  -  Std: {df_train[target].std():.4f}')
    print(f'Min: {df_train[target].min():.4f}  -  25%: {df_train[target].quantile(0.25):.4f}  -  50%: {df_train[target].quantile(0.5):.4f}  -  75%: {df_train[target].quantile(0.75):.4f}  -  Max: {df_train[target].max():.4f}')
    print(f'Skew: {df_train[target].skew():.4f}  -  Kurtosis: {df_train[target].kurtosis():.4f}')
    missing_values_count = df_train[df_train[target].isnull()].shape[0]
    training_samples_count = df_train.shape[0]
    print(f'Missing Values: {missing_values_count}/{training_samples_count} ({missing_values_count * 100 / training_samples_count:.4f}%)')

    fig, axes = plt.subplots(ncols=2, figsize=(24, 8), dpi=100)
    sns.kdeplot(df_train[target], label=target, fill=True, ax=axes[0])
    axes[0].axvline(df_train[target].mean(), label=f'{target} Mean', color='r', linewidth=2, linestyle='--')
    axes[0].axvline(df_train[target].median(), label=f'{target} Median', color='b', linewidth=2, linestyle='--')
    probplot(df_train[target], plot=axes[1])
    axes[0].legend(prop={'size': 16})
    
    for i in range(2):
        axes[i].tick_params(axis='x', labelsize=12.5, pad=10)
        axes[i].tick_params(axis='y', labelsize=12.5, pad=10)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
    axes[0].set_title(f'{target} Distribution in Training Set', fontsize=20, pad=15)
    axes[1].set_title(f'{target} Probability Plot', fontsize=20, pad=15)

    plt.show()


# In[6]:


from scipy.stats import probplot


# In[7]:


visualize_target('target')


# In[8]:


#By Gunes Evitan https://www.kaggle.com/code/gunesevitan/optiver-realized-volatility-prediction-eda#8.-Time-Buckets

target_means = df_train.groupby('stock_id')['target'].mean()
target_stds = df_train.groupby('stock_id')['target'].std()

target_means_and_stds = pd.concat([target_means, target_stds], axis=1)
target_means_and_stds.columns = ['mean', 'std']
target_means_and_stds.sort_values(by='mean', ascending=True, inplace=True)

fig, ax = plt.subplots(figsize=(32, 48))
ax.barh(
    y=np.arange(len(target_means_and_stds)),
    width=target_means_and_stds['mean'],
    xerr=target_means_and_stds['std'],
    align='center',
    ecolor='black',
    capsize=3
)

ax.set_yticks(np.arange(len(target_means_and_stds)))
ax.set_yticklabels(target_means_and_stds.index)
ax.set_xlabel('target', size=20, labelpad=15)
ax.set_ylabel('stock_id', size=20, labelpad=15)
ax.tick_params(axis='x', labelsize=20, pad=10)
ax.tick_params(axis='y', labelsize=20, pad=10)
ax.set_title('Mean Realized Volatility of Stocks', size=25, pad=20)

plt.show()

del target_means, target_stds, target_means_and_stds


# #What didn't work on the last Optiver Competition:
# 
# By Yirun Zhang  https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/275466
# 
# On Feature Engineering:
# 
# Advanced Realized Volatility and Quarticity features; KMeans + PCA features;
# Microstructural features for HFT such as Volume synchronized probability of informed trading, Hasbrouck's flow, Hasbrouck's lands, Amihuds Lamda, Kyle’s Lamda, Becker Parkinson volatility, Corwin Shultz spread, etc.
# Median, Median Absolute Deviation, Entropy, skewness, kurtosis of original features
# Ranking features in each time id.;;Exponential decay RV;; 
# RV feature calculated by optimising the number of seconds in the bucket (or percentage of length) considered for calculating the RV value for each stock;
# 
# On Modelling:
# 
# Stacking; Tabular Transformer Autoencoder + NN/Ridge Regression; AE-MLP, ResDTNet, NODE, GrowNet, and lots of fancy NN model architectures; RNN, CNN, Transformer models on time-span aggregated features or raw book features (special case); 
# Training separate model for each stock id.SVM, KNN, XGB, HGB (Sklearn); 
# Post-processing by shifting mean and std of prediction; 
# Removing outlier time_ids by isolation forest trained on the feature importance of each time id. Adding Dropout or Gaussian noise after the input layer of NN.
# 
# By Leo(calibrator):
# 
# Feature Engineering:
# Feature selection: Filtering methods; sklearn's VarianceThreshold; sklearn's mutual_info_regression; sklearn's f_regression; BorutaShap (too computational expensive)
# Catboost builtin method; Wrapper methods (CV improved a little, LB worsened); Forward/backward sequential selection by mlxtend; Embedding methods: (these worked when including too many noise features, but they didn't lead to the best result);LASSO
# MLP with L1 regularization on the first dense layer;
# 
# Modelling or Training:
# 
# Traditional econometrics models: HAR-RV and HAR-RV-J, etc.; Data augmentation: At each epoch, randomly select a subset of stock_ids/time_ids for aggregation.; Each stock has its own emsembling weights.; Semi-supervised learning; 
# Directly add prediction to test set, then retrain; 300-sec model.
# 
# #The "Magic" according to Resistance0108
# 
# Make aggregated features within time_id & stock_id groups more wisely
# 
# Capture relation between stocks
# 
# Use NN instead of LGB (CNN, LSTM etc. "But I'm sceptic to use LSTM because I think 10 min data is not sufficient to predict next 10 min volatility.")
# 
# https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/256356

# #Acknowledgements:
# 
# Sam Maule https://www.kaggle.com/code/semaule/eda-visualising-trading-data-optiver
# 
# Fritz Cremer https://www.kaggle.com/code/fritzcremer/how-are-the-buckets-shifted-in-time
# 
# Gunes Evitan https://www.kaggle.com/code/gunesevitan/optiver-realized-volatility-prediction-eda#8.-Time-Buckets
