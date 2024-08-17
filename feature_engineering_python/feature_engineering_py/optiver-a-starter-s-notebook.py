#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Welcome to Optiver competition! In this competition, we have to predict the closing price of hundreds Nasdaq-listed stocks based on the closing auction and order book of the stock. This is a time-series competition, and it seems that most of the test set will only be used after the submission period ends. Metrics used in this competition is Mean Absolute Error or MAE.
# 
# This is the first time I'm doing this kind of competition, so I appreciate any feedbacks. Before reading this notebook though, I recommend you to read these two first:
# - https://www.kaggle.com/code/tomforbes/optiver-trading-at-the-close-introduction
# - https://www.kaggle.com/code/sohier/optiver-2023-basic-submission-demo

# # Loading Libraries and Datasets

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import gc

from sklearn import set_config
from sklearn.base import clone
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

sns.set_theme(style = 'white', palette = 'viridis')
pal = sns.color_palette('viridis')

pd.set_option('display.max_rows', 100)
set_config(transform_output = 'pandas')
pd.options.mode.chained_assignment = None


# Due to the size of dataset, we can load all integral features into bytes (int 8) and short (int 16). We can also try to load float features into float 32 or float 16 if we want. You have to keep in mind that this sacrifices the accuracy of our calculation though.

# In[2]:


dtypes = {
    'stock_id' : np.uint8,
    'date_id' : np.uint16,
    'seconds_in_bucket' : np.uint16,
    'imbalance_buy_sell_flag' : np.int8,
    'time_id' : np.uint16,
}

train = pd.read_csv(r'/kaggle/input/optiver-trading-at-the-close/train.csv', dtype = dtypes).drop(['row_id', 'time_id'], axis = 1)
test = pd.read_csv(r'/kaggle/input/optiver-trading-at-the-close/example_test_files/test.csv', dtype = dtypes).drop(['row_id', 'time_id'], axis = 1)

gc.collect()


# # Descriptive Statistics
# 
# Let's begin by taking a peek at our original training dataset first

# In[3]:


train.head(10)


# In[4]:


desc = pd.DataFrame(index = list(train))
desc['count'] = train.count()
desc['nunique'] = train.nunique()
desc['%unique'] = desc['nunique'] / len(train) * 100
desc['null'] = train.isnull().sum()
desc['type'] = train.dtypes
desc = pd.concat([desc, train.describe().T], axis = 1)
desc


# So we have about 5.238 million rows in the train dataset. Some features have about half of the entire rows being missing value. Disturbingly, there are also 88 rows where the target is missing too.

# In[5]:


test.head(10)


# There are only three rows in public test set. The rest might only come after the submission period ends.

# In[6]:


desc = pd.DataFrame(index = list(test))
desc['count'] = test.count()
desc['nunique'] = test.nunique()
desc['%unique'] = desc['nunique'] / len(test) * 100
desc['null'] = test.isnull().sum()
desc['type'] = test.dtypes
desc = pd.concat([desc, test.describe().T], axis = 1)
desc


# Let's try grouping the categorical and numerical features now.

# In[7]:


temporal_features = ['date_id', 'seconds_in_bucket']
categorical_features = ['imbalance_buy_sell_flag', 'stock_id']
numerical_features = train.drop(temporal_features + categorical_features + ['target'], axis = 1).columns


# # Distribution of Numerical Features
# 
# Since the feature description above can be too confusing to read, we can try using visualization to make things simpler, such as using KDE Plot to visualize the distribution of features.

# In[8]:


fig, ax = plt.subplots(5, 2, figsize = (15, 20), dpi = 300)
ax = ax.flatten()

for i, column in enumerate(numerical_features):
    
    sns.kdeplot(train[column], ax=ax[i], color=pal[0], fill = True)
    
    ax[i].set_title(f'{column} Distribution', size = 14)
    ax[i].set_xlabel(None)
    
fig.suptitle('Distribution of Numerical Features\nper Dataset\n', fontsize = 24, fontweight = 'bold')
plt.tight_layout()


# # Distribution of Categorical Features
# 
# There are two categorical features. However, we only need to check one of them, which is `imbalance_buy_sell_flag`.

# In[9]:


fig, ax = plt.subplots(1, 2, figsize = (20, 10), dpi = 300)
ax = ax.flatten()

ax[0].pie(
    train['imbalance_buy_sell_flag'].value_counts(), 
    shadow = True, 
    explode = [.1 for i in range(train['imbalance_buy_sell_flag'].nunique())], 
    autopct = '%1.f%%',
    textprops = {'size' : 14, 'color' : 'white'}
)

sns.countplot(data = train, y = 'imbalance_buy_sell_flag', ax = ax[1], palette = 'viridis', order = train['imbalance_buy_sell_flag'].value_counts().index)
ax[1].yaxis.label.set_size(20)
plt.yticks(fontsize = 12)
ax[1].set_xlabel('Count in Train', fontsize = 15)
ax[1].set_ylabel('Imbalance Flag', fontsize = 15)
plt.xticks(fontsize = 12)

fig.suptitle('Distribution of Imbalance Flag\nin Train Dataset\n\n\n\n', fontsize = 30, fontweight = 'bold')
plt.tight_layout()


# We can see that sell imbalance flag came out the most, followed by buy imbalance flag, with the rarest being the no imbalance flag.

# # Target Distribution
# 
# Last but not least, we can visualize the distribution of the target.

# In[10]:


plt.figure(figsize = (20, 10), dpi = 300)

sns.kdeplot(train.target, fill = True)

plt.title('Target Distribution', weight = 'bold', fontsize = 30)
plt.show()


# It looks very balanced, but certainly not normal due to the high kurtosis. Not that it actually matters in this competition unless you want to use Linear Regression...

# # Target Over Time
# 
# Since there are notebooks that visualizes the target value over date and time per stocks, we'll just visualize the average target instead. We'll also separate the line based on imbalance flag.

# In[11]:


plt.figure(figsize = (20, 10), dpi = 300)

sns.lineplot(data = train, x = 'date_id', y = 'target', hue = 'imbalance_buy_sell_flag', errorbar = None, palette = 'viridis')

plt.title('Average Target Over Days', weight = 'bold', fontsize = 30)
plt.show()


# The lines above looks like a mess, but we can see that stock with buy flag tends to have upspike, while stock with sell flag tend to have downspikes. Actually, no imbalance flag also looks the same as sell flag, except a bit less frequent.
# 
# Now let's see the target over seconds in buckets.

# In[12]:


plt.figure(figsize = (20, 10), dpi = 300)

sns.lineplot(data = train, x = 'seconds_in_bucket', y = 'target', hue = 'imbalance_buy_sell_flag', errorbar = None, palette = 'viridis')

plt.title('Average Target Over Seconds in Buckets', weight = 'bold', fontsize = 30)
plt.show()


# Now we can clearly see the difference between each flag. It's obvious that buy and sell stocks go opposite direction, while no imbalance flag tends to stay in the middle. At the end of closing period, we can also see that stock with buy flag goes up while stock with sell flag goes down for the target.

# # Preparation
# 
# This is where we start preparing everything if we want to start building machine learning models. We will use Time Series Split for our cross validation process. We will also drop any rows where the missing target values are located.

# In[13]:


X = train[~train.target.isna()]
y = X.pop('target')

seed = 42
tss = TimeSeriesSplit(10)

os.environ['PYTHONHASHSEED'] = '42'
tf.keras.utils.set_random_seed(seed)


# # Feature Engineering
# 
# For feature engineering, we will use Yuanzhe Zhou's code.
# 
# You can see their code here: https://www.kaggle.com/code/yuanzhezhou/baseline-lgb-xgb-and-catboost

# In[14]:


def imbalance_calculator(x):
    
    features = ['seconds_in_bucket', 'imbalance_buy_sell_flag',
               'imbalance_size', 'matched_size', 'bid_size', 'ask_size',
                'reference_price','far_price', 'near_price', 'ask_price', 'bid_price', 'wap',
                'imb_s1', 'imb_s2'
               ]
    
    x_copy = x.copy()
    
    x_copy['imb_s1'] = x.eval('(bid_size - ask_size) / (bid_size + ask_size)')
    x_copy['imb_s2'] = x.eval('(imbalance_size - matched_size) / (matched_size + imbalance_size)')
    
    prices = ['reference_price','far_price', 'near_price', 'ask_price', 'bid_price', 'wap']
    
    for i,a in enumerate(prices):
        for j,b in enumerate(prices):
            if i>j:
                x_copy[f'{a}_{b}_imb'] = x.eval(f'({a} - {b}) / ({a} + {b})')
                features.append(f'{a}_{b}_imb')
                    
    for i,a in enumerate(prices):
        for j,b in enumerate(prices):
            for k,c in enumerate(prices):
                if i>j and j>k:
                    max_ = x[[a,b,c]].max(axis=1)
                    min_ = x[[a,b,c]].min(axis=1)
                    mid_ = x[[a,b,c]].sum(axis=1)-min_-max_

                    x_copy[f'{a}_{b}_{c}_imb2'] = (max_-mid_)/(mid_-min_)
                    features.append(f'{a}_{b}_{c}_imb2')
    
    return x_copy[features]

ImbalanceCalculator = FunctionTransformer(imbalance_calculator)


# # Cross-Validation
# 
# Honestly, there is only one usable non-neural network model I can think of for this data: LightGBM. First, there are a lot of missing values in the dataset, so we either have to impute them all, or just use model that can take care of it implicitly, which are XGBoost, LightGBM, and CatBoost. Second, due to the size of the dataset, we want to use GPU to increase the speed, and guess what, CatBoost's MAE loss function can't be optimized with GPU.

# In[15]:


def cross_val_score(estimator, cv = tss, label = ''):
    
    X = train[~train.target.isna()]
    y = X.pop('target')
    
    #initiate prediction arrays and score lists
    val_predictions = np.zeros((len(X)))
    #train_predictions = np.zeros((len(sample)))
    train_scores, val_scores = [], []
    
    #training model, predicting prognosis probability, and evaluating metrics
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        
        model = clone(estimator)
        
        #define train set
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        
        #define validation set
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        #train model
        model.fit(X_train, y_train)
        
        #make predictions
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
                  
        val_predictions[val_idx] += val_preds
        
        #evaluate model for a fold
        train_score = mean_absolute_error(y_train, train_preds)
        val_score = mean_absolute_error(y_val, val_preds)
        
        #append model score for a fold to list
        train_scores.append(train_score)
        val_scores.append(val_score)
    
    print(f'Val Score: {np.mean(val_scores):.5f} ± {np.std(val_scores):.5f} | Train Score: {np.mean(train_scores):.5f} ± {np.std(train_scores):.5f} | {label}')
    
    return val_scores, val_predictions


# To prevent any leakage during cross-validation, we will put the feature engineering inside the pipeline.

# In[16]:


models = [
    ('XGBoost', XGBRegressor(random_state = seed, objective = 'reg:absoluteerror', tree_method = 'gpu_hist')),
    ('LightGBM', LGBMRegressor(random_state = seed, objective = 'mae', device_type = 'gpu')),
    ('CatBoost', CatBoostRegressor(random_state = seed, objective = 'MAE', verbose = 0))
]

for (label, model) in models:
    _ = cross_val_score(
        make_pipeline(
            ImbalanceCalculator,
            model
        ),
        label = label
    )


# # Prediction and Submission
# 
# Finally, let's train our chosen model on the whole train dataset and do inference on the test dataset. The submission in this competition is a bit special since we have to use the competition host's API.

# In[17]:


import optiver2023
env = optiver2023.make_env()
iter_test = env.iter_test()

model = make_pipeline(
    ImbalanceCalculator,
    LGBMRegressor(random_state = seed, objective = 'mae', device_type = 'gpu')
)

model.fit(X, y)

counter = 0
for (test, revealed_targets, sample_prediction) in iter_test:
    sample_prediction['target'] = model.predict(test.drop('row_id', axis = 1))
    env.predict(sample_prediction)
    counter += 1


# Thanks for reading!
