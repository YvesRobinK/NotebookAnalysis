#!/usr/bin/env python
# coding: utf-8

# # Simple Recurrent Neural Networks (RNN) and LSTM
# 
# Hi Kagglers,
# This notebook is continuation of my first notebook for this competition, I couldn't finished everything on first [notebook](https://www.kaggle.com/godzill22/tps-07-eda-statistical-analysis) (which was my intetion ), however for some reason when I run my notebook my computer is slow and freezes. So I decided that I leave EDA notebook as it is now and focuse on modeling in this one.

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


import scipy as sp

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from sklearn.metrics import mean_squared_log_error

print(f"Tensorflow version {tf.__version__}")


# In[2]:


train_df = pd.read_csv("/kaggle/input/tabular-playground-series-jul-2021/train.csv")
test_df = pd.read_csv("/kaggle/input/tabular-playground-series-jul-2021/test.csv")
sample_df = pd.read_csv("/kaggle/input/tabular-playground-series-jul-2021/sample_submission.csv")


# In[3]:


train_df['date_time'] = pd.to_datetime(train_df['date_time'])
test_df['date_time'] = pd.to_datetime(test_df['date_time'])


# In[4]:


# Feature engineering
train_df['year'] = train_df['date_time'].dt.year
train_df['month'] = train_df['date_time'].dt.month
train_df['hour'] = train_df['date_time'].dt.hour
train_df['day'] = train_df['date_time'].dt.day

test_df['year'] = test_df['date_time'].dt.year
test_df['month'] = test_df['date_time'].dt.month
test_df['hour'] = test_df['date_time'].dt.hour
test_df['day'] = test_df['date_time'].dt.day


# In[5]:


# Set data_time column as index as it is needed for RNN
train = train_df.set_index("date_time").copy()
test = test_df.set_index("date_time").copy()


# In[6]:


target_cols = [col for col in train.columns if col.startswith('target')]
feat_cols = [col for col in train.columns if col not in target_cols]


# In[7]:


# Calculate percentage of a dataset to be a test set
test_percent = 0.1
test_point = np.round(len(train)*test_percent)
test_idx = int(len(train)-test_point)


# In[8]:


# Devide a dataset into train and test sets
Xtrain = train.drop(target_cols[:0], axis=1).iloc[:test_idx]
Xtest = train.drop(target_cols[:0], axis=1).iloc[test_idx:]


# In[9]:


Xtrain


# In[10]:


def plot_predictions(col_idx, predictions):
    plt.figure(figsize=(25,5))
    sns.lineplot(x=Xtest[LENGTH:].index, y=Xtest[LENGTH:][target_cols[col_idx]],label="True labels")
    sns.lineplot(x=Xtest[LENGTH:].index, y=predictions.reshape(-1), label="Predictions")
    plt.title(f"Prediction for {target_cols[col_idx]}    RMSLE={np.sqrt(mean_squared_log_error(Xtest[target_cols[col_idx]][LENGTH:], np.abs(predictions.reshape(-1))))}")
    plt.legend();


# In[11]:


LENGTH = 48  # use 48 observation to test_generator 49
BATCH_SIZE = 1 # usually this batch size works well
TARGET_IDX = 0


# In[12]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

Xtrain_scaled = scaler.fit_transform(Xtrain.drop(target_cols[TARGET_IDX],axis=1))
Xtest_scaled = scaler.transform(Xtest.drop(target_cols[TARGET_IDX],axis=1))


# In[13]:


N_FEATURES = Xtrain_scaled.shape[1]


# In[14]:


train_generator = TimeseriesGenerator(data=Xtrain_scaled,
                                      targets=Xtrain[target_cols[TARGET_IDX]],
                                      length=LENGTH,
                                      batch_size=BATCH_SIZE)
test_generator = TimeseriesGenerator(data=Xtest_scaled,
                                     targets=Xtest[target_cols[TARGET_IDX]],
                                     length=LENGTH,
                                     batch_size=BATCH_SIZE)


# ### Simple RNN

# In[15]:


tf.random.set_seed(45)
rnn_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(48, input_shape=(LENGTH, N_FEATURES)),
    tf.keras.layers.Dense(48),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

rnn_model.compile(optimizer='adam',loss='mse', metrics=['mse'])

rnn_history = rnn_model.fit(train_generator, epochs=10, validation_data=test_generator)


# In[16]:


rnn_df = pd.DataFrame(rnn_history.history)


# In[17]:


rnn_df[['loss','val_loss']].plot()


# In[18]:


rnn_preds = rnn_model.predict(test_generator)


# In[19]:


plot_predictions(col_idx=0,
                 predictions=rnn_preds)


# ### Stacked LSTM

# In[20]:


lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(48, return_sequences=True, input_shape=(LENGTH, N_FEATURES), dropout=0.2),
    tf.keras.layers.LSTM(48),
    tf.keras.layers.Dense(1)
])

lstm_model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.mean_squared_error, metrics=['mse'])

lstm_history = lstm_model.fit(train_generator, epochs=10, validation_data=test_generator)


# In[21]:


lstm_df = pd.DataFrame(lstm_history.history)
lstm_df[['loss','val_loss']].plot()


# In[22]:


lstm_preds = lstm_model.predict(test_generator)


# In[23]:


plot_predictions(col_idx=0,
                 predictions=lstm_preds)


# ### LSTM with big guns
# 
# Now is time to build our LSTM model with everything we can to improve our models predictions.

# In[24]:


def rmsle_custom(y_true, y_pred):
    msle = tf.keras.losses.MeanSquaredLogarithmicError()
    return K.sqrt(msle(y_true, y_pred))


es = tf.keras.callbacks.EarlyStopping(monitor='val_rmsle_custom', 
                                      mode='min',patience=4, 
                                      restore_best_weights=True)

plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', # val_rmsle_custom try
                                               mode='min',
                                               patience=2, 
                                               verbose=1)

weights_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=45)


# In[25]:


def lstm_train_test_model():
    
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(48, return_sequences=True, 
                             input_shape=(LENGTH, N_FEATURES),
                             dropout=0.1, 
                             kernel_initializer=weights_initializer),
        tf.keras.layers.LSTM(48, dropout=0.1, 
                             kernel_initializer=weights_initializer),
        tf.keras.layers.Dense(1)  
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=rmsle_custom)
    
    history = model.fit(train_generator,
                        epochs=30,
                        validation_data=test_generator,
                        callbacks=[es,plateau],
                        verbose=1)
    
    return history, model


# In[26]:


lstm_2_history, lstm_model = lstm_train_test_model()


# In[27]:


lstm_2_history_df = pd.DataFrame(lstm_2_history.history)
lstm_2_history_df[['loss','val_loss']].plot()


# In[28]:


lstm_2_preds = lstm_model.predict(test_generator)


# In[29]:


plot_predictions(col_idx=0,
                 predictions=lstm_2_preds)


# In[30]:


K.clear_session()


# ### LSTM Autoencoder

# In[31]:


def rmsle_custom(y_true, y_pred):
    msle = tf.keras.losses.MeanSquaredLogarithmicError()
    return K.sqrt(msle(y_true, y_pred))


es = tf.keras.callbacks.EarlyStopping(monitor='val_rmsle_custom', 
                                      mode='min',patience=6, 
                                      restore_best_weights=True)

plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', # val_rmsle_custom try
                                               mode='min',
                                               patience=2, 
                                               verbose=1)

weights_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=45)


# In[32]:


def lstm_autoencoder():
    
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(48, return_sequences=True, 
                             input_shape=(LENGTH, N_FEATURES),
                             kernel_initializer=weights_initializer),
        tf.keras.layers.LSTM(24, return_sequences=True,
                             kernel_initializer=weights_initializer),
        tf.keras.layers.LSTM(12, kernel_initializer=weights_initializer),
        tf.keras.layers.RepeatVector(LENGTH),
        tf.keras.layers.LSTM(12, return_sequences=True, kernel_initializer=weights_initializer),
        tf.keras.layers.LSTM(24, return_sequences=True, kernel_initializer=weights_initializer),
        tf.keras.layers.LSTM(48, return_sequences=True,  kernel_initializer=weights_initializer),
        tf.keras.layers.TimeDistributed(Dense(N_FEATURES)),
        ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=rmsle_custom)
    
    history = model.fit(train_generator,
                        epochs=30,
                        validation_data=test_generator,
                        callbacks=[plateau,es],
                        verbose=1)
    
    return history, model


# In[33]:


history_autoencoder, lstm_autoenc = lstm_autoencoder()


# In[34]:


autoenc_history_df = pd.DataFrame(history_autoencoder.history)
autoenc_history_df[['loss','val_loss']].plot()


# In[35]:


autoenc_preds = lstm_autoenc.predict(test_generator batch_size=BATCH_SIZE)


# In[37]:


lK.clear_session()


# ## Time to create a model for submission

# In[ ]:




