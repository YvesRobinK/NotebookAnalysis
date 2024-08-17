#!/usr/bin/env python
# coding: utf-8

# ![logo](https://keras.io/img/logo.png)

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
import json
get_ipython().system('cp ../input/ventilator-feature-engineering/VFE.py .')


# # Prediction - LSTM based model
# This notebook is part of a series:  
#   * [Ventilator: Feature engineering](https://www.kaggle.com/mistag/ventilator-feature-engineering)
#   * [Keras model tuning with Optuna](https://www.kaggle.com/mistag/keras-model-tuning-with-optuna)
#   * [[train] Ventilator LSTM Model - part I](https://www.kaggle.com/mistag/train-ventilator-lstm-model-part-i)
#   * [[train] Ventilator LSTM Model - part II](https://www.kaggle.com/mistag/train-ventilator-lstm-model-part-ii)
#   * [[train] Ventilator LSTM Model - part III](https://www.kaggle.com/mistag/train-ventilator-lstm-model-part-iii)
#   * [[train] Ventilator LSTM Model - part IV](https://www.kaggle.com/mistag/train-ventilator-lstm-model-part-iv)
#   
# ## References
# The code is based on these references:  
#   * [Improvement base on Tensor Bidirect LSTM](https://www.kaggle.com/kensit/improvement-base-on-tensor-bidirect-lstm-0-173/notebook) by [Ken Sit](https://www.kaggle.com/kensit)
#   * [Ensemble Folds with MEDIAN - [0.153]](https://www.kaggle.com/cdeotte/ensemble-folds-with-median-0-153) by [Chris Deotte](https://www.kaggle.com/cdeotte)
# 

# # Test dataset

# In[2]:


from VFE import add_features

# test set
test_ori = pd.read_csv('../input/ventilator-pressure-prediction/test.csv')
test = add_features(test_ori)
test.drop(['id', 'breath_id'], axis=1, inplace=True)

RS = load(open('../input/ventilator-feature-engineering/RS.pkl', 'rb'))
test = RS.transform(test)
test = test.reshape(-1, 80, test.shape[-1])


# # Prediction

# Fetch batch size from training session:

# In[3]:


with open('../input/train-ventilator-lstm-model-part-i/train_params.json', 'r') as fp:
    config = json.load(fp)


# In[4]:


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


# In[5]:


test_preds = []
models = ['../input/train-ventilator-lstm-model-part-ii/lstm_fold2.hdf5', 
          '../input/train-ventilator-lstm-model-part-iii/lstm_fold3.hdf5', 
          '../input/train-ventilator-lstm-model-part-iv/lstm_fold4.hdf5',
          '../input/train-ventilator-lstm-model-part-v/lstm_fold0.hdf5',
          '../input/train-ventilator-lstm-model-part-v-b/lstm_fold1.hdf5',
          '../input/train-ventilator-lstm-model-part-i/lstm_fold0.hdf5',
          '../input/train-ventilator-lstm-model-part-i/lstm_fold1.hdf5',
          '../input/train-ventilator-lstm-model-part-v-d/lstm_fold3.hdf5',
          '../input/train-ventilator-lstm-model-part-v-c/lstm_fold2.hdf5']

with strategy.scope():
    for m in models:
        print('Loading model {}'.format(m))
        model = keras.models.load_model(m)
            
        test_preds.append(model.predict(test, batch_size=config['BATCH_SIZE'], verbose=2).squeeze().reshape(-1, 1).squeeze())


# # Submission
# Here we will do rounding to discrete target values, as discussed [here](https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/276083) and several other places.

# In[6]:


pressure = np.load('../input/ventilator-feature-engineering/y_train.npy')
P_MIN = np.min(pressure)
P_MAX = np.max(pressure)
P_STEP = pressure[0][1] - pressure[0][0]
print('Min pressure: {}'.format(P_MIN))
print('Max pressure: {}'.format(P_MAX))
print('Pressure step: {}'.format(P_STEP))
print('Unique values:  {}'.format(np.unique(pressure).shape[0]))


# In[7]:


submission = pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')
submission["pressure"] = np.median(np.vstack(test_preds),axis=0)
submission["pressure"] = np.round((submission.pressure - P_MIN)/P_STEP) * P_STEP + P_MIN
submission.pressure = np.clip(submission.pressure, P_MIN, P_MAX)
submission.to_csv('submission.csv', index=False)


# In[ ]:




