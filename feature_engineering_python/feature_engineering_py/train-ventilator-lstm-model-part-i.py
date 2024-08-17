#!/usr/bin/env python
# coding: utf-8

# ![logo](https://cdn.freelogovectors.net/wp-content/uploads/2018/07/tensorflow-logo.png)

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import gc
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from pickle import load
import matplotlib.pyplot as plt
import json
get_ipython().system('cp ../input/ventilator-feature-engineering/VFE.py .')


# # Training - LSTM based model
# This notebook is part of a series:  
#   * [Ventilator: Feature engineering](https://www.kaggle.com/mistag/ventilator-feature-engineering)
#   * [Keras model tuning with Optuna](https://www.kaggle.com/mistag/keras-model-tuning-with-optuna)
#   * [train] Ventilator LSTM Model - part I
#   * [[train] Ventilator LSTM Model - part 2](https://www.kaggle.com/mistag/train-ventilator-lstm-model-part-ii)
#   * [[pred] Ventilator LSTM Model](https://www.kaggle.com/mistag/pred-ventilator-lstm-model)
#   
# ## References
# The code is based on these references:  
#   * [Improvement base on Tensor Bidirect LSTM](https://www.kaggle.com/kensit/improvement-base-on-tensor-bidirect-lstm-0-173/notebook) by [Ken Sit](https://www.kaggle.com/kensit)
#   * [Ensemble Folds with MEDIAN - [0.153]](https://www.kaggle.com/cdeotte/ensemble-folds-with-median-0-153) by [Chris Deotte](https://www.kaggle.com/cdeotte)

# # Dataset

# In[2]:


train = np.load('../input/ventilator-feature-engineering/x_train.npy')
targets = np.load('../input/ventilator-feature-engineering/y_train.npy')


# # Model

# In[3]:


# model creation
def create_lstm_model():

    x0 = tf.keras.layers.Input(shape=(train.shape[-2], train.shape[-1]))  

    lstm_layers = 4 # number of LSTM layers
    lstm_units = [940, 540, 462, 316]
    lstm = Bidirectional(keras.layers.LSTM(lstm_units[0], return_sequences=True))(x0)
    for i in range(lstm_layers-1):
        lstm = Bidirectional(keras.layers.LSTM(lstm_units[i+1], return_sequences=True))(lstm)    
    lstm = Dropout(0.002)(lstm)
    lstm = Dense(lstm_units[-1], activation='swish')(lstm)
    lstm = Dense(1)(lstm)

    model = keras.Model(inputs=x0, outputs=lstm)
    model.compile(optimizer="adam", loss="mae")
    
    return model


# # Training
# First define a few parameters that will also be used in other notebooks:

# In[4]:


BATCH_SIZE = 256
NFOLDS = 5
SEED = 777
EPOCHS = 300

params = {}
params['BATCH_SIZE'] = BATCH_SIZE
params['NFODLS'] = NFOLDS
params['SEED'] = SEED
params['EPOCHS'] = EPOCHS
with open('train_params.json', 'w') as fp:
    json.dump(params, fp, indent=4)


# HW strategy:

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


hist = []
folds = [0,1] # folds to train

with strategy.scope():
    kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(train, targets)):
        if fold in folds:
            print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)
            folds.append(fold)
            X_train, X_valid = train[train_idx], train[test_idx]
            y_train, y_valid = targets[train_idx], targets[test_idx]
            
            model = create_lstm_model()
            model.compile(optimizer="adam", loss="mae")
            
            #checkpoint_filepath = f"lstm_fold_{fold}.hdf5"
            checkpoint_filepath = '/kaggle/working/lstm_fold{}.hdf5'.format(fold)

            scheduler = ExponentialDecay(1e-3, 400*((len(train)*0.8)/BATCH_SIZE), 1e-5)
            #lr = LearningRateScheduler(scheduler, verbose=0)
            lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=1)
            es = EarlyStopping(monitor="val_loss", patience=60, verbose=1, mode="min", restore_best_weights=True)
            sv = keras.callbacks.ModelCheckpoint(
                checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                save_weights_only=False, mode='auto', save_freq='epoch',
                options=None
            )
            hist.append(model.fit(X_train, y_train, 
                                  validation_data=(X_valid, y_valid), 
                                  epochs=EPOCHS, batch_size=BATCH_SIZE, 
                                  callbacks=[lr, es, sv]))
        
            del X_train, X_valid, y_train, y_valid, model
            gc.collect()


# Let's take a look at the learning curves.

# In[7]:


colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
plt.figure(figsize=(16,16))
for i in range(len(hist)):
    plt.plot(hist[i].history['loss'], linestyle='-', color=colors[i], label='Train, fold #{}'.format(str(folds[i])))
for i in range(len(hist)):
    plt.plot(hist[i].history['val_loss'], linestyle='--', color=colors[i], label='Validation, fold #{}'.format(str(folds[i])))
plt.ylim(top=1)
plt.title('Model Loss')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend()
plt.grid(which='major', axis='both')
plt.show();

