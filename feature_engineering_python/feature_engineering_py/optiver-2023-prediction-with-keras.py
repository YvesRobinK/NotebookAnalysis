#!/usr/bin/env python
# coding: utf-8

# # Optiver 2023 Prediction with keras
# 
# In this notebook, I will create model using Keras DNN architecture trained on [Optiver - Trading at the Close competetion dataset](https://www.kaggle.com/competitions/optiver-trading-at-the-close/code). 
# 
# The target we need to predict equals 60 second future move in the wap of the stock, less the 60 second future move of the synthetic index. As shown in following:
# $Target = (\frac{StockWAP_{t+60}}{StockWAP_{t}} - \frac{IndexWAP_{t+60}}{IndexWAP_{t}}) * 10000$
# 
# We could either build model to make prediction on target directly or make prediction on wap, synthetic index and intermediate variables. For simplicity I will make prediciton on target directly.
# 
# Credits to notebook https://www.kaggle.com/code/kaito510/goto-conversion-baseline-lgb-xgb-and-catboost

# ## Imports

# In[1]:


import pandas as pd
import optiver2023
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import gc
import tensorflow as tf
import keras_tuner as kt
from sklearn.metrics import mean_absolute_error


# ## Configuration

# In[2]:


class CFG:
    is_tuning = False
    is_training = False


# ## Load data

# In[3]:


train = pd.read_csv("/kaggle/input/optiver-trading-at-the-close/train.csv")
train.head()


# In[4]:


test = pd.read_csv("/kaggle/input/optiver-trading-at-the-close/example_test_files/test.csv")
test.head()


# ## Feature understanding
# **stock_id**: A unique identifier for the stock. Not all stock IDs exist in every time bucket.
# 
# **date_id**: A unique identifier for the date. Date IDs are sequential & consistent across all stocks.
# 
# **imbalance_size**: The amount unmatched at the current reference price (in USD).
# 
# **imbalance_buy_sell_flag**: An indicator reflecting the direction of auction imbalance.
# 
# - buy-side imbalance: 1
# - sell-side imbalance; -1
# - no imbalance; 0
# 
# **reference_price**: The price at which paired shares are maximized, the imbalance is minimized and the distance from the bid-ask midpoint is minimized, in that order. Can also be thought of as being equal to the near price bounded between the best bid and ask price.
# 
# **matched_size**: The amount that can be matched at the current reference price (in USD).
# 
# **far_price**: The crossing price that will maximize the number of shares matched based on auction interest only. This calculation excludes continuous market orders.
# 
# **near_price**: The crossing price that will maximize the number of shares matched based auction and continuous market orders.
# 
# **[bid/ask]_price**: Price of the most competitive buy/sell level in the non-auction book.
# 
# **[bid/ask]_size**: The dollar notional amount on the most competitive buy/sell level in the non-auction book.
# 
# **wap:** The weighted average price in the non-auction book
# 
# $\frac{BidPrice*AskSize+AskPrice*BidSize}{BidSize+AskSize}$.
# 
# 
# **seconds_in_bucket**: The number of seconds elapsed since the beginning of the day's closing auction, always starting from 0.
# 
# **target**: The 60 second future move in the wap of the stock, less the 60 second future move of the synthetic index. Only provided for the train set.
# 
# * The synthetic index is a custom weighted index of Nasdaq-listed stocks constructed by Optiver for this competition.
# 
# * The unit of the target is basis points, which is a common unit of measurement in financial markets. A 1 basis point price move is equivalent to a 0.01% price move.
# 
# * Where t is the time at the current observation, we can define the target:
# 
# $Target = (\frac{StockWAP_{t+60}}{StockWAP_{t}} - \frac{IndexWAP_{t+60}}{IndexWAP_{t}}) * 10000$

# ## Exploratory Data Analysis

# In[5]:


numeric_columns = ["imbalance_size", "reference_price", "matched_size","bid_price", "bid_size", "ask_price", "ask_size", "wap", "imbalance_buy_sell_flag", "seconds_in_bucket"]
categorical_columns = ["imbalance_buy_sell_flag", "stock_id"]
feature_columns = numeric_columns + categorical_columns
label_column = "target"
stock_ids = list(train.stock_id.unique())


# ### Show descriptive statistics

# In[6]:


train[numeric_columns + [label_column]].describe().transpose()


# Visualize a few of stocks.

# In[7]:


def visualize(stock_id, date_id):
    df = train[(train.stock_id == stock_id) & (train.date_id == date_id)].copy()
    df.plot(x="seconds_in_bucket", y=["bid_price", "ask_price", "far_price", "near_price"])
    plt.title("bid_price, ask_price, far_price, near_price")
    plt.show()
    df.plot(x="seconds_in_bucket", y=["bid_size", "ask_size"])
    plt.title("bid_size, ask_size")
    plt.show()
    df.plot(x="seconds_in_bucket", y=["target"])
    plt.title("target")
    plt.show()


# In[8]:


visualize(3, 1)


# In[9]:


visualize(3, 5)


# In[10]:


visualize(33, 5)


# Check null values, remove data with missing value except for far_price and near_price.

# In[11]:


train.isnull().sum()


# In[12]:


train = train[train.wap.isnull() == False]
train.head()


# In[13]:


train.isnull().sum()


# ## Feature Engineering
# 

# In[14]:


def feature_engineering(df):
    features = [
        'imb_s1', 'imb_s2','bid_size_x_ask_price',  
        'ask_size_x_bid_price',
        'bid_size_x_ask_price_over_ask_size_x_bid_price'
    ]
    df['imb_s1'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)')
    df['imb_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)')
    df['bid_size_x_ask_price'] = df.eval('bid_size * ask_price')
    df['ask_size_x_bid_price'] = df.eval('ask_size * bid_price')
    df['bid_size_x_ask_price_over_ask_size_x_bid_price'] = df.eval('bid_size_x_ask_price / ask_size_x_bid_price')
    prices = ['reference_price', 'ask_price', 'bid_price']
    for i,a in enumerate(prices):
        for j,b in enumerate(prices):
            if i>j:
                df[f'{a}_{b}_imb'] = df.eval(f'({a}-{b})/({a}+{b})')
                df[f'{a}_over_{b}'] = df.eval(f'{a}/{b}')
                features.append(f'{a}_{b}_imb') 
                features.append(f'{a}_over_{b}')   
                    
    for i,a in enumerate(prices):
        for j,b in enumerate(prices):
            for k,c in enumerate(prices):
                if i>j and j>k:
                    max_ = df[[a,b,c]].max(axis=1)
                    min_ = df[[a,b,c]].min(axis=1)
                    mid_ = df[[a,b,c]].sum(axis=1)-min_-max_
                    df[f'{a}_{b}_{c}_imb2'] = (max_-mid_)/(mid_-min_)
                    features.append(f'{a}_{b}_{c}_imb2')
    return df, features


# In[15]:


train, other_features = feature_engineering(train)
numeric_columns += other_features


# In[16]:


train.head()


# In[17]:


train[numeric_columns].describe()


# In[18]:


## Remove this column since it somehow has infinitive values
numeric_columns.remove("bid_price_ask_price_reference_price_imb2")


# ## Normalization

# In[19]:


normalization = keras.layers.Normalization()
normalization.adapt(train[numeric_columns])


# ## Cross Validation
# I will use 10% latest data as hold out set.

# In[20]:


validation_split = 0.1
unique_dates = sorted(train.date_id.unique())
train_dates = unique_dates[:int(len(unique_dates) * (1 - validation_split))]
valid_dates = unique_dates[int(len(unique_dates) * (1 - validation_split)):]
train_features = train[train.date_id.isin(train_dates)][numeric_columns]
train_label = train[train.date_id.isin(train_dates)][label_column]
valid_features = train[train.date_id.isin(valid_dates)][numeric_columns]
valid_label =  train[train.date_id.isin(valid_dates)][label_column]


# ## Modeling

# In[21]:


def build_model(hp):
    use_dropout = hp.Choice("use_dropout", [True, False])
    dropout_value = hp.Float("dropout", min_value=0.1, max_value=0.3)
    learning_rate = hp.Float("learing_rate", min_value=1e-5, max_value=1e-3, sampling="log")
    depth = 6
    params = {
        "depth": depth,
        "use_dropout": use_dropout,
        "dropout": dropout_value,
        "learning_rate": learning_rate,
    }
    for i in range(depth):
        params[f"unit_{i}"] = hp.Int(f"unit_{i}", min_value=16, max_value=256, step=16)
    params["activation"] = hp.Choice("activation", ["relu", "swish", "tanh"])
    params["l2"] = hp.Choice("l2", [1e-5, 3e-5, 5e-5, 1e-6, 5e-6])
    params["loss"] = hp.Choice("loss", ["mae", "huber_loss", "mse"])
    return get_model(params)

def get_model(params):
    use_dropout = params["use_dropout"]
    dropout_value = params["dropout"]
    learning_rate = params["learning_rate"]
    depth = params["depth"]
    units = [params[f"unit_{i}"] for i in range(depth)]
    activation = params["activation"]
    l2_factor = params["l2"]
    inputs = tf.keras.Input(shape=(len(numeric_columns)), dtype=tf.float32)
    x = normalization(inputs)
    for i in range(depth):
        kernel_regularizer = tf.keras.regularizers.l2(l2_factor)
        x = tf.keras.layers.Dense(units[i], activation=activation, kernel_regularizer=kernel_regularizer)(x)
        if use_dropout:
            x = tf.keras.layers.Dropout(dropout_value)(x)
    output = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    loss = params["loss"]
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=["mae"])
    return model


def goto_conversion(listOfOdds, total = 1, eps = 1e-6, isAmericanOdds = False):

    #Convert American Odds to Decimal Odds
    if isAmericanOdds:
        for i in range(len(listOfOdds)):
            currOdds = listOfOdds[i]
            isNegativeAmericanOdds = currOdds < 0
            if isNegativeAmericanOdds:
                currDecimalOdds = 1 + (100/(currOdds*-1))
            else: #Is non-negative American Odds
                currDecimalOdds = 1 + (currOdds/100)
            listOfOdds[i] = currDecimalOdds

    #Error Catchers
    if len(listOfOdds) < 2:
        raise ValueError('len(listOfOdds) must be >= 2')
    if any(x < 1 for x in listOfOdds):
        raise ValueError('All odds must be >= 1, set isAmericanOdds parameter to True if using American Odds')

    #Computation
    listOfProbabilities = [1/x for x in listOfOdds] #initialize probabilities using inverse odds
    listOfSe = [pow((x-x**2)/x,0.5) for x in listOfProbabilities] #compute the standard error (SE) for each probability
    step = (sum(listOfProbabilities) - total)/sum(listOfSe) #compute how many steps of SE the probabilities should step back by
    outputListOfProbabilities = [min(max(x - (y*step),eps),1) for x,y in zip(listOfProbabilities, listOfSe)]
    return outputListOfProbabilities

def zero_sum(listOfPrices, listOfVolumes):
    listOfSe = [x**0.5 for x in listOfVolumes] #compute standard errors assuming standard deviation is same for all stocks
    step = sum(listOfPrices)/sum(listOfSe)
    outputListOfPrices = [x - (y*step) for x,y in zip(listOfPrices, listOfSe)]
    return outputListOfPrices

def inference(models, valid_features):
    predictions = []
    for model in models:
        pred = model.predict(valid_features, batch_size=512).reshape(-1)
        predictions.append(pred)
    if len(models) == 1:
        return predictions[0]
    return 0.5 * np.mean(predictions, axis=0) + 0.5 * np.median(predictions, axis=0)
    
def evaluate(valid_features, valid_label, models):
    y_pred = inference(models, valid_features)
    mae = mean_absolute_error(valid_label, y_pred)
    print(f"MAE:", mae)
    return {
        "mae": mae
    }


# In[22]:


models = []
for i in range(5):
    models.append(tf.keras.models.load_model(f"/kaggle/input/optiver-keras-models-v2/optiver_keras_models/model{i}.h5"))


# ## Hyper Parameter Tuning

# In[23]:


if CFG.is_tuning:
    tuner = kt.BayesianOptimization(
        build_model,
        objective=kt.Objective("val_mae", direction="min"),
        max_trials=20,
        overwrite=True
    )
    tuner.search(
        x=train_features,
        y=train_label, 
        batch_size=512,
        epochs=5, 
        validation_data=(valid_features, valid_label),
        verbose=2
    )
    tuner.results_summary()
    if i in range(5):
        if i < len(tuner.get_best_models()):
            model = tuner.get_best_models()[i]
            models.append(model)
            model.summary()
            model.save(F"model{i}.h5")


# ## Training model

# In[24]:


if CFG.is_training:
    model_path = "model.h5"
    best_hps = tuner.get_best_hyperparameters()
    model = build_model(best_hps[0])
    model.fit(
        x=train_features,
        y=train_label, 
        epochs=30, 
        validation_data=(valid_features, valid_label),
        batch_size=512,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                model_path, 
                monitor="val_mae", 
                mode="min", 
                restore_best_weights=True, 
                save_best_only=True
            ),
            keras.callbacks.EarlyStopping(patience=5)
        ]
    )
    model = tf.keras.models.load_model(model_path)
    models.append(model)


# ## Model Evaluation

# In[25]:


evaluate(valid_features, valid_label, models)


# ## Create Submission

# In[26]:


env = optiver2023.make_env()
iter_test = env.iter_test()
counter = 0
for (test, revealed_targets, sample_prediction) in iter_test:
    if counter == 0:
        print(test.head(3))
        print(revealed_targets.head(3))
        print(sample_prediction.head(3))
    test.fillna(1.0, inplace=True)
    df, _ = feature_engineering(test)
    df['target'] = inference(models, df[numeric_columns])
    df['target'] = zero_sum(df['target'], df.loc[:,'bid_size'] + df.loc[:,'ask_size'])
    sample_prediction['target'] = df['target']
    env.predict(sample_prediction)
    counter += 1

