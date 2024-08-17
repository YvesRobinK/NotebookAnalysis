#!/usr/bin/env python
# coding: utf-8

# # Import required libraries

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Reshape
from keras import backend as K
from keras import regularizers 
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from tensorflow.keras.losses import Loss


# # Load Dataset

# In[2]:


stock_price_df = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")


# # A glance at the data

# In[3]:


print('(rows, columns) =', stock_price_df.shape)
stock_price_df.tail()


# In[4]:


stock_price_df['ExpectedDividend'] = stock_price_df['ExpectedDividend'].fillna(0)
stock_price_df['SupervisionFlag'] = stock_price_df['SupervisionFlag'].map({True: 1, False: 0})
stock_price_df['Date'] = pd.to_datetime(stock_price_df['Date'])
stock_price_df.info()


# # Import stock list
# 

# In[5]:


stock_list = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/stock_list.csv")


# In[6]:


stock_list = stock_list[['SecuritiesCode','NewMarketSegment','33SectorCode','17SectorCode','Universe0','Section/Products','NewIndexSeriesSize']]
stock_list = stock_list.replace(np.nan,'-')
stock_list['Universe0'] = np.where(stock_list['Universe0'], 1, 0)
stock_list = stock_list.drop_duplicates()
stock_list


# # Some Feature Engineering

# In[7]:


def FE(stock_price_df):
    stock_price_df['BOP'] = (stock_price_df['Open']-stock_price_df['Close'])/(stock_price_df['High']-stock_price_df['Low'])
    stock_price_df['wp'] = (stock_price_df['Open']+stock_price_df['High']+stock_price_df['Low'])/3
    stock_price_df['TR'] = stock_price_df['High'] - stock_price_df['Low']
    # stock_price_df['AD'] = ta.AD(High, Low, Close, Volume)
    # stock_price_df['OBV']  = ta.OBV(Close, Volume)
    stock_price_df['OC'] = stock_price_df['Open'] * stock_price_df['Close']
    stock_price_df['HL'] = stock_price_df['High'] * stock_price_df['Low']
    stock_price_df['logC'] = np.log(stock_price_df['Close']+1)
    stock_price_df['OHLCstd'] = stock_price_df[['Open','Close','High','Low']].std(axis=1)
    stock_price_df['OHLCskew'] = stock_price_df[['Open','Close','High','Low']].skew(axis=1)
    stock_price_df['OHLCkur'] = stock_price_df[['Open','Close','High','Low']].kurtosis(axis=1)
    stock_price_df['Cpos'] = (stock_price_df['Close']-stock_price_df['Low'])/(stock_price_df['High']-stock_price_df['Low']) -0.5
    stock_price_df['bsforce'] = stock_price_df['Cpos'] * stock_price_df['Volume']
    stock_price_df['Opos'] = (stock_price_df['Open']-stock_price_df['Low'])/(stock_price_df['High']-stock_price_df['Low']) -0.5
    stock_price_df['Date'] = pd.to_datetime(stock_price_df['Date'])
    stock_price_df['weekday'] = stock_price_df['Date'].dt.weekday+1
    stock_price_df['Monday'] = np.where(stock_price_df['weekday']==1,1,0)
    stock_price_df['Tuesday'] = np.where(stock_price_df['weekday']==2,1,0)
    stock_price_df['Wednesday'] = np.where(stock_price_df['weekday']==3,1,0)
    stock_price_df['Thursday'] = np.where(stock_price_df['weekday']==4,1,0)
    stock_price_df['Friday'] = np.where(stock_price_df['weekday']==5,1,0)
    return stock_price_df
stock_price_df = FE(stock_price_df)
stock_price_df = pd.merge(stock_price_df,stock_list, on='SecuritiesCode')


# In[8]:


subf = ['Open', 'High', 'Low', 'Close',
       'Volume', 'ExpectedDividend',
       'SupervisionFlag', 'BOP', 'wp', 'TR', 'OC', 'HL', 'logC',
       'OHLCstd', 'OHLCskew', 'OHLCkur', 'Cpos', 'bsforce', 'Opos']


# # standardize features according to daily stock data (at the same day only)

# In[9]:


def daily_standardize(df,col):
    avg = df[[col,'Date']].groupby('Date').mean()
    avg.columns = ['avg']
    avg['Date'] = avg.index
    avg = avg.reset_index(drop=True)
    std = df[[col,'Date']].groupby('Date').std()
    std.columns = ['std']
    std['Date'] = std.index
    std = std.reset_index(drop=True)
    df = pd.merge(df, avg, on='Date')
    df = pd.merge(df,std,on='Date')
    df[col] = (df[col] - df['avg'])/df['std']
    df = df.drop(['avg','std'],axis=1)
    df[col] = df[col].fillna(0)
    return df


# In[10]:


from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()


# In[11]:


for i in subf:
    stock_price_df = daily_standardize(stock_price_df,i)


# In[12]:


stock_price_df = stock_price_df.fillna(0)
stock_price_df.head()


# # standardize "Target" as well

# In[13]:


stock_price_df = daily_standardize(stock_price_df,'Target')


# In[14]:


stock_price_df['Target'].head()


# # Tranform the "SecuritiesCode" of stock as well

# In[15]:


investment_ids = list(stock_price_df['SecuritiesCode'].unique())
investment_id_size = len(investment_ids) + 1
investment_id_lookup_layer = layers.IntegerLookup(max_tokens=investment_id_size)
with tf.device("cpu"):
    investment_id_lookup_layer.adapt(stock_price_df['SecuritiesCode'])


# ## Define the DNN model that separately trains "SecuritiesCode" & features, and then concatenate them together for the final training

# ## You can adjust number of layers, neurons, and etc.

# In[16]:


def get_model():
    investment_id_inputs = tf.keras.Input((1, ), dtype=tf.uint16)
    features_inputs = tf.keras.Input((len(subf), ), dtype=tf.float16)
    
    investment_id_x = investment_id_lookup_layer(investment_id_inputs)
    investment_id_x = layers.Embedding(investment_id_size, 4, input_length=1)(investment_id_x)
    investment_id_x = layers.Reshape((-1, ))(investment_id_x)
    investment_id_x = layers.Dense(16, activation='swish')(investment_id_x)
    investment_id_x = layers.Dense(16, activation='swish')(investment_id_x)
    
    feature_x = layers.Dense(16, activation='swish')(features_inputs)
    feature_x = layers.Dense(16, activation='swish')(feature_x)
    
    x = layers.Concatenate(axis=1)([investment_id_x, feature_x])
    x = layers.Dense(16, activation='swish', kernel_regularizer="l2")(x)
    x = layers.Dense(16, activation='swish', kernel_regularizer="l2")(x)
    output = layers.Dense(1)(x)
    rmse = keras.metrics.RootMeanSquaredError(name="rmse")
    model = tf.keras.Model(inputs=[investment_id_inputs, features_inputs], outputs=[output])
    model.compile(optimizer=tf.optimizers.Adam(0.05), loss='mse', metrics=['mse', "mae", "mape", rmse])
    return model


# ## Transform the train data so that it can used for DNN model training.

# In[17]:


def preprocess(X, y):
    print(X)
    print(y)
    return X, y
def make_dataset(feature, investment_id, y, batch_size=32, mode="train"):
    ds = tf.data.Dataset.from_tensor_slices(((investment_id, feature), y))
    ds = ds.map(preprocess)
    if mode == "train":
        ds = ds.shuffle(256)
    ds = ds.batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
    return ds


# ## Use GroupKFold to train 5 DNN models (the same "Sector" stock should always be in the same group)

# In[18]:


from sklearn.model_selection import GroupKFold
kfold = GroupKFold(n_splits = 5)


# In[19]:


count=0
df_x = stock_price_df[subf]
df_y=stock_price_df['Target']
time_id = stock_price_df['33SectorCode']
dnn_models = []
for train_index, val_index in kfold.split(df_x, df_y,time_id):
    # Split training dataset.
    train_x, train_y = df_x.iloc[train_index], df_y.iloc[train_index]
    train_inv =stock_price_df['SecuritiesCode'].iloc[train_index]
    # Split validation dataset.
    val_x, val_y = df_x.iloc[val_index], df_y.iloc[val_index]
    val_inv =  stock_price_df['SecuritiesCode'].iloc[val_index]
    # Make tensor dataset.
    tf_train = make_dataset(train_x, train_inv, train_y, batch_size=4096, mode="train")
    tf_val = make_dataset(val_x, val_inv, val_y, batch_size=4096, mode="train")
    # Load model
    model = get_model()
  
    model.fit(tf_train, epochs = 3,
             validation_data = (tf_val), shuffle=True)
    model.save_weights('my_dnn_'+str(count)+'.tf')
    dnn_models.append(model)
    count+=1

    del tf_train
    del tf_val
    del model, train_x, train_y, val_x, val_y


# ## average model predictions

# In[20]:


def infer(models, ds):
    y_preds = []
    for model in models:
        y_pred = model.predict(ds)
        y_preds.append(y_pred)
    return np.mean(y_preds, axis=0)


# ## Transform test set so that it can be used for prediction

# In[21]:


def preprocess_test(investment_id, feature):
    return (investment_id, feature), 0

def make_test_dataset(feature, investment_id, batch_size=1024):
    ds = tf.data.Dataset.from_tensor_slices(((investment_id, feature)))
    ds = ds.map(preprocess_test)
    ds = ds.batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
    return ds


# ## Create features for the test set, standardize them, and use 5 DNN models averaging to make final prediction.

# In[22]:


import jpx_tokyo_market_prediction
env = jpx_tokyo_market_prediction.make_env()   # initialize the environment
iter_test = env.iter_test()    # an iterator which loops over the test files
for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    
    prices['ExpectedDividend'] = prices['ExpectedDividend'].fillna(0)
    prices['SupervisionFlag'] = prices['SupervisionFlag'].map({True: 1, False: 0})
    prices = pd.merge(prices,stock_list, on='SecuritiesCode')
    
    prices = FE(prices)
    
    prices[subf] = StandardScaler().fit_transform(prices[subf])
    prices = prices.fillna(0)


    prices['inference'] = -infer(dnn_models, make_test_dataset(prices[subf],prices['SecuritiesCode']))
    prices['rank'] = prices['inference'].rank(method='first')-1
    prices['rank'] = prices['rank'].apply(lambda x: int(x))
    prices = prices.drop('Date',axis=1)

    sample_prediction = pd.merge(sample_prediction, prices, on=['SecuritiesCode'])[['Date','SecuritiesCode','rank']]
    sample_prediction['rank'] = sample_prediction['rank'].fillna(1000)
    sample_prediction.columns = ['Date','SecuritiesCode','Rank']
    print(sample_prediction)
    
    env.predict(sample_prediction)   # register your predictions


# In[23]:


len(np.unique(sample_prediction['Rank']))


# In[24]:


alt = prices.copy()
alt.head()

