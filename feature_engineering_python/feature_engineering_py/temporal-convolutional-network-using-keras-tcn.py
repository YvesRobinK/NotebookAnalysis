#!/usr/bin/env python
# coding: utf-8

# # [Ventilator Pressure Prediction](https://www.kaggle.com/c/ventilator-pressure-prediction): 
# # Temporal Convolutional Network using Keras-TCN
# 
# In this simple "starter" notebook we shall be using a **Temporal Convolutional Network** layer, thanks to the [Keras-TCN](https://github.com/philipperemy/keras-tcn) package written by Philippe Rémy, which is based on the work in the paper ["*An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*"](https://arxiv.org/pdf/1803.01271.pdf).
# 
# ![](https://raw.githubusercontent.com/philipperemy/keras-tcn/master/misc/Dilated_Conv.png)
# 
# Firstly, install `keras-tcn`:

# In[1]:


get_ipython().system('pip install -q keras-tcn --no-dependencies')
from tcn import TCN, tcn_full_summary


# This notebook is heavily based on the following two LSTM notebooks:
# * [Tensorflow LSTM Baseline](https://www.kaggle.com/ryanbarretto/tensorflow-lstm-baseline), written by [Ryan Barretto](https://www.kaggle.com/ryanbarretto)
# * [Tensorflow Bidirectional LSTM (0.234)](https://www.kaggle.com/tolgadincer/tensorflow-bidirectional-lstm-0-234), by [Tolga Dincer](https://www.kaggle.com/tolgadincer)

# In[2]:


import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tensorflow import keras
import tensorflow as tf


# # Read in the data

# In[3]:


train_data = pd.read_csv('../input/ventilator-pressure-prediction/train.csv')
test_data  = pd.read_csv('../input/ventilator-pressure-prediction/test.csv')
submission = pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')


# # Some feature engineering
# These ideas are from various sources, not all of them are necessarily useful and are just here for demonstration purposes:

# In[4]:


for df in (train_data, test_data):
    df['u_in_lag'] = df.groupby('breath_id')['u_in'].shift(2).fillna(method="backfill")
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    df['last_value_u_in'] = df.groupby('breath_id')['u_in'].transform('last')
    df['u_in_mean'] = df.groupby('breath_id')['u_in'].transform('mean')
    df['u_in_median'] = df.groupby('breath_id')['u_in'].transform('median')
    df['first_value_u_in'] = df.groupby('breath_id')['u_in'].transform('first')
    df['u_in_min'] = df.groupby('breath_id')['u_in'].transform('min')
    df['u_in_max'] = df.groupby('breath_id')['u_in'].transform('max')
    df['u_in_delta'] = df['u_in_max'] - df['u_in_min']


# # Get things ready

# In[5]:


targets = train_data[['pressure']].to_numpy().reshape(-1, 80)

# drop the unwanted features
train_data.drop(['pressure', 'id', 'breath_id', 'u_out'], axis=1, inplace=True)
test_data =  test_data.drop(['id', 'breath_id', 'u_out'], axis=1)


# In[6]:


from sklearn.preprocessing import RobustScaler
RS = RobustScaler()
train_data = RS.fit_transform(train_data)
test_data  = RS.transform(test_data)


# In[7]:


n_features = train_data.shape[-1]

train_data = train_data.reshape(-1, 80, n_features)
test_data  = test_data.reshape(-1, 80, n_features)

n_epochs = 50
n_splits =  5


# # Calculation

# In[8]:


kf = KFold(n_splits=n_splits, shuffle=False)
test_preds = []

for fold, (train_idx, test_idx) in enumerate(kf.split(train_data, targets)):
    print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)
    X_train, X_valid = train_data[train_idx], train_data[test_idx]
    y_train, y_valid = targets[train_idx], targets[test_idx]
    
    scheduler = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, 200*((len(test_data)*0.8)/1024), 1e-5)
    
    model = keras.models.Sequential([
        TCN(input_shape=(80, n_features), nb_filters=256, return_sequences=True, dilations=[1, 2, 4, 8, 16, 32]),
        keras.layers.Dense(1)
    ])
    
    model.compile(optimizer="adam", loss="mae",
                  metrics=keras.metrics.MeanAbsoluteError())
    
    history = model.fit(X_train, y_train, 
                        validation_data=(X_valid, y_valid), 
                        epochs=n_epochs, 
                        batch_size=1024, 
                        callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler)])
    
    model.save(f'Fold{fold+1} weights')
    test_preds.append(model.predict(test_data).squeeze().reshape(-1, 1).squeeze())


# # Plot a learning curve

# In[9]:


logs = pd.DataFrame(history.history)

plt.figure(figsize=(14, 4))
plt.subplot(1, 2, 1)
plt.plot(logs.loc[1:,"loss"], lw=2, label='training loss')
plt.plot(logs.loc[1:,"val_loss"], lw=2, label='validation loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(logs.loc[1:,"mean_absolute_error"], lw=2, label='training MAE')
plt.plot(logs.loc[1:,"val_mean_absolute_error"], lw=2, label='validation MAE')
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend(loc='upper right')
plt.show()


# # Submission
# The submission is created from the average over each fold

# In[10]:


submission["pressure"] = sum(test_preds)/n_splits
submission.to_csv('submission.csv', index=False)


# # Related kaggle notebooks
# * ["Temporal CNN"](https://www.kaggle.com/christofhenkel/temporal-cnn) by [Dieter](https://www.kaggle.com/christofhenkel)
# * ["Temporal Convolutional Network"](https://www.kaggle.com/christofhenkel/temporal-convolutional-network) by [Dieter](https://www.kaggle.com/christofhenkel)
# * ["(PyTorch) Temporal Convolutional Networks"](https://www.kaggle.com/ceshine/pytorch-temporal-convolutional-networks) by [Ceshine Lee](https://www.kaggle.com/ceshine)
