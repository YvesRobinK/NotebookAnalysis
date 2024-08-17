#!/usr/bin/env python
# coding: utf-8

# ![logo](https://cdn.freelogovectors.net/wp-content/uploads/2018/07/tensorflow-logo.png)

# This notebook implements the optimized model from [Keras model tuning with Optuna](https://www.kaggle.com/mistag/keras-model-tuning-with-optuna).

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import gc
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from pickle import load
get_ipython().system('cp ../input/ventilator-feature-engineering/VFE.py .')


# # Dataset creation
# Training dataset is prepared with functions in the [feature engineering notebook](https://www.kaggle.com/mistag/ventilator-feature-engineering), which is based on [Improvement base on Tensor Bidirect LSTM](https://www.kaggle.com/kensit/improvement-base-on-tensor-bidirect-lstm-0-173/notebook) by [Ken Sit](https://www.kaggle.com/kensit).

# In[2]:


from VFE import add_features

train_ori = pd.read_csv('../input/ventilator-pressure-prediction/train.csv')
targets = train_ori['pressure'].to_numpy().reshape(-1, 80)
train_ori.drop(labels='pressure', axis=1, inplace=True)
train = add_features(train_ori)
# normalise the dataset
RS = RobustScaler()
train = RS.fit_transform(train)

# Reshape to group 80 timesteps for each breath ID
train = train.reshape(-1, 80, train.shape[-1])


# The test set is created below, using the feature engineering function from the above mentioned notebook:

# In[3]:


test_ori = pd.read_csv('../input/ventilator-pressure-prediction/test.csv')
test = add_features(test_ori)
test = RS.transform(test)
test = test.reshape(-1, 80, test.shape[-1])


# # Model creation
# Model parameters are from [Keras model tuning with Optuna](https://www.kaggle.com/mistag/keras-model-tuning-with-optuna). (The "optimal parameters" will not be exactly the same every time the optimization study is run, so the parameters used below might differ from the model tuning notebook).

# In[4]:


# model creation
def create_lstm_model():

    x0 = tf.keras.layers.Input(shape=(train.shape[-2], train.shape[-1]))  

    lstm_layers = 4 # number of LSTM layers
    lstm_units = [320, 305, 304, 229]
    lstm = Bidirectional(keras.layers.LSTM(lstm_units[0], return_sequences=True))(x0)
    for i in range(lstm_layers-1):
        lstm = Bidirectional(keras.layers.LSTM(lstm_units[i+1], return_sequences=True))(lstm)    
    lstm = Dropout(0.001)(lstm)
    lstm = Dense(100, activation='relu')(lstm)
    lstm = Dense(1)(lstm)

    model = keras.Model(inputs=x0, outputs=lstm)
    model.compile(optimizer="adam", loss="mae")
    
    return model


# # Training

# In[5]:


# Function to get hardware strategy
def get_hardware_strategy():
    try:
        # TPU detection. No parameters necessary if TPU_NAME environment variable is
        # set: this is always the case on Kaggle.
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        tf.config.optimizer.set_jit(True)
    else:
        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
        strategy = tf.distribute.get_strategy()

    return tpu, strategy

tpu, strategy = get_hardware_strategy()


# In[6]:


EPOCH = 350
BATCH_SIZE = 512
NFOLDS = 5

with strategy.scope():
    kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=2021)
    history = []
    test_preds = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(train, targets)):
        print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)
        X_train, X_valid = train[train_idx], train[test_idx]
        y_train, y_valid = targets[train_idx], targets[test_idx]
        model = create_lstm_model()
        model.compile(optimizer="adam", loss="mae", metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])

        scheduler = ExponentialDecay(1e-3, 400*((len(train)*0.8)/BATCH_SIZE), 1e-5)
        lr = LearningRateScheduler(scheduler, verbose=0)

        history.append(model.fit(X_train, y_train, 
                                 validation_data=(X_valid, y_valid), 
                                 epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=[lr]))
        test_pred = model.predict(test).squeeze().reshape(-1, 1).squeeze()
        test_preds.append(test_pred)    
        
        # save model
        #model.save("lstm_model_fold_{}".format(fold))
        
        del X_train, X_valid, y_train, y_valid, model
        gc.collect()


# Plot the learning curves:

# In[7]:


colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
plt.figure(figsize=(16,16))
for i in range(NFOLDS):
    plt.plot(history[i].history['loss'], linestyle='-', color=colors[i], label='Train, fold #{}'.format(str(i)))
for i in range(NFOLDS):
    plt.plot(history[i].history['val_loss'], linestyle='--', color=colors[i], label='Validation, fold #{}'.format(str(i)))
plt.ylim(top=1)
plt.title('Model Loss')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend()
plt.grid(which='major', axis='both')
plt.show();


# Also look at the MAE for the different folds:

# In[8]:


def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i], ha = 'center')

fold_mae = np.zeros(NFOLDS, dtype=np.float)
for i in range(1):
    fold_mae[i] = history[i].history['val_loss'][-1]
plt.figure(figsize = (10, 5))
names = ['Fold #0', 'Fold #1', 'Fold #2', 'Fold #3', 'Fold #4']
plt.bar(names, fold_mae, color ='royalblue', width = 0.4)
addlabels(names, np.round(fold_mae, 3))
plt.ylabel("MAE")
plt.title("Fold scores")
plt.show();


# # Submission

# In[9]:


submission = pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')
submission["pressure"] = sum(test_preds)/5
submission.to_csv('submission.csv', index=False)


# In[ ]:




