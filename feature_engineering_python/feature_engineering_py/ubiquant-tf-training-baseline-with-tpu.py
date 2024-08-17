#!/usr/bin/env python
# coding: utf-8

# # Quick Notes
# * This is a time series problem, we should split the data based on time. In my experiments it does not provide good cv and lb
# * Best version is splitting data with GroupKFold
# * With this in mind, im still looking for a good cv strategy that can fit models in the correct spot
# * It seems the data is already normalized
# * The competition metric is the mean pearson correlation for each time_id, previous notebook computes the metric at the end of each epoch but for speed reason I only compute it at the end of the training
# * The dataset is very large, use pickle, parquet or others format to save memory
# * Probably a sequence model is much better than a fully connected model (RNN, CNN, Transformer)
# 
# Experiment configs:
# 
# KFOLD_STRAT -> GroupKFold using investiment_id CV -> 0.1936 LB -> 0.144 EPOCHS -> 15
# 
# KFOLD_STRAT -> GroupKFold using investiment_id CV -> 0.1822 LB -> 0.144 EPOCHS -> 8
# 
# KFOLD_STRAT -> GroupKFold using time_id CV -> 0.1461 LB -> 0.145 EPOCHS -> 15
# 
# Last experiment seems to be the best way of validation, cv lb gap is small

# In[1]:


import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import mixed_precision
from tensorflow.keras import backend as K
from tqdm.notebook import tqdm
import random
import warnings
import gc
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 100)


# In[2]:


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
        policy = mixed_precision.Policy('mixed_bfloat16')
        mixed_precision.set_global_policy(policy)
    else:
        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
        strategy = tf.distribute.get_strategy()

    print("REPLICAS: ", strategy.num_replicas_in_sync)
    return tpu, strategy

tpu, strategy = get_hardware_strategy()
# Configuration
EPOCHS = 15
BATCH_SIZE = 32 * strategy.num_replicas_in_sync
# Model Seed 
MODEL_SEED = 42
# Learning rate
LR = 0.0008
# Folds
FOLDS = 5
# Verbosity
VERBOSE = 2
# For tf.dataset
AUTO = tf.data.experimental.AUTOTUNE


# In[3]:


# Function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    
def correlationLoss(x, y, axis = -2):
    """Loss function that maximizes the pearson correlation coefficient between the predicted values and the labels,
    while trying to have the same mean and variance"""
    x = tf.convert_to_tensor(x)
    y = tf.cast(y, x.dtype)
    n = tf.cast(tf.shape(x)[axis], x.dtype)
    xsum = tf.reduce_sum(x, axis = axis)
    ysum = tf.reduce_sum(y, axis = axis)
    xmean = xsum / n
    ymean = ysum / n
    xsqsum = tf.reduce_sum(tf.math.squared_difference(x, xmean), axis = axis)
    ysqsum = tf.reduce_sum(tf.math.squared_difference(y, ymean), axis = axis)
    cov = tf.reduce_sum((x - xmean) * (y - ymean), axis = axis)
    corr = cov / tf.sqrt(xsqsum * ysqsum)
    sqdif = tf.reduce_sum(tf.math.squared_difference(x, y), axis = axis) / n / tf.sqrt(ysqsum / n)
    return tf.convert_to_tensor(K.mean(tf.constant(1.0, dtype = x.dtype) - corr + (0.01 * sqdif)))
    
def build_model(shape, steps):
    with strategy.scope(): 
        def fc_block(x, units):
            x = tf.keras.layers.Dropout(0.35)(x)
            x = tf.keras.layers.Dense(units, activation = 'relu')(x)
            return x
        
        inp = tf.keras.layers.Input((shape))
        x = fc_block(inp, units = 768)
        x = fc_block(x, units = 384)
        x = fc_block(x, units = 192)
        output = tf.keras.layers.Dense(1, activation = 'linear')(x)
        model = tf.keras.models.Model(inputs = [inp], outputs = [output])
        scheduler = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate = LR, decay_steps = steps, end_learning_rate = 0.000005)
        opt = tf.keras.optimizers.Adam(learning_rate = scheduler)
        model.compile(
            optimizer = opt,
            loss = [tf.keras.losses.MeanSquaredError()],
        )
        return model

# Custom callback for mean pearson correlation coefficient with early stopping
class mpcc_metric(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, indices, targets, patience = 7):
        super(mpcc_metric, self).__init__()
        self.patience = patience
        # Store best weights
        self.best_weights = None
        self.x_val, self.y_val = validation_data
        self.indices = indices
        self.targets = targets
        
    def on_train_begin(self, logs = None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = 0
        
    def on_epoch_end(self, epoch, logs = None):
        prediction = self.model.predict(self.x_val).astype(np.float32).reshape(-1)
        p_corr_co = []
        for i in range(len(self.targets)):
            p_corr_co.append(pearsonr(self.targets[i], prediction[self.indices[i]])[0])
        p_corr_co_score = np.average(np.array(p_corr_co))
        print("Mean Pearson correlation coefficient  - epoch: {:d} - score: {:.6f}".format(epoch + 1, p_corr_co_score))
        if p_corr_co_score > self.best:
            print('Validation score improved!')
            self.best = p_corr_co_score
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print('Restoring the best weights from the best epoch')
            self.model.set_weights(self.best_weights)
            print('Saving best weights to disk')
            self.model.save_weights('simple_fc.h5')
    def on_train_end(self, logs = None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: Early Stopping' % (self.stopped_epoch + 1))

# Calculate pearson correlation coefficient
def pearson_coef(data):
    return data.corr()['target']['prediction']

# Calculate mean pearson correlation coefficient
def comp_metric(valid_df):
    return np.mean(valid_df.groupby(['time_id']).apply(pearson_coef))

# Function to train and evaluate
def train_and_evaluate():
    # Seed everything
    seed_everything(MODEL_SEED)
    # Read data
    train = pd.read_pickle('../input/ubiquant-market-prediction-half-precision-pickle/train.pkl')
    # Feature list
    features = [col for col in train.columns if col not in ['row_id', 'time_id', 'investment_id', 'target']]
    # Some feature engineering
    # Get the correlations with the target to encode time_id
    corr1 = train[features[0:100] + ['target']].corr()['target'].reset_index()
    corr2 = train[features[100:200] + ['target']].corr()['target'].reset_index()
    corr3 = train[features[200:] + ['target']].corr()['target'].reset_index()
    corr = pd.concat([corr1, corr2, corr3], axis = 0, ignore_index = True)
    corr['target'] = abs(corr['target'])
    corr.sort_values('target', ascending = False, inplace = True)
    best_corr = corr.iloc[3:103, 0].to_list()
    del corr1, corr2, corr3, corr
    # Add time id related features (market general features to relate time_ids)
    time_id_features = []
    for col in tqdm(best_corr):
        mapper = train.groupby(['time_id'])[col].mean().to_dict()
        train[f'time_id_{col}'] = train['time_id'].map(mapper)
        train[f'time_id_{col}'] = train[f'time_id_{col}'].astype(np.float16)
        time_id_features.append(f'time_id_{col}')
    print(f'We added {len(time_id_features)} features related to time_id')
    # Update feature list
    features += time_id_features
    np.save('features.npy', np.array(features))
    np.save('best_corr.npy', np.array(best_corr))
    # Store out of folds predictions
    oof_predictions = np.zeros(len(train))
    # Initiate GroupKFold (all investment_id should be in the same fold, we want to predict new investment_id)
    kfold = GroupKFold(n_splits = FOLDS)
    # Create groups based on time_id
    train.loc[(train['time_id'] >= 0) & (train['time_id'] < 280), 'group'] = 0
    train.loc[(train['time_id'] >= 280) & (train['time_id'] < 585), 'group'] = 1
    train.loc[(train['time_id'] >= 585) & (train['time_id'] < 825), 'group'] = 2
    train.loc[(train['time_id'] >= 825) & (train['time_id'] < 1030), 'group'] = 3
    train.loc[(train['time_id'] >= 1030) & (train['time_id'] < 1400), 'group'] = 4
    train['group'] = train['group'].astype(np.int16)
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(train, groups = train['group'])):
        print(f'Training fold {fold + 1}')
        x_train, x_val = train[features].loc[trn_ind], train[features].loc[val_ind]
        y_train, y_val = train['target'].loc[trn_ind], train['target'].loc[val_ind]
        # Reset keras session and tpu
        K.clear_session()
        if tpu:
            tf.tpu.experimental.initialize_tpu_system(tpu)
        n_training_rows = x_train.shape[0]
        n_validation_rows = x_val.shape[0]
        STEPS_PER_EPOCH = n_training_rows  // BATCH_SIZE
        # Build simple fc model
        print('Building model...')
        model = build_model(len(features), STEPS_PER_EPOCH * EPOCHS)
        print(f'Training with {n_training_rows} rows')
        print(f'Validating with {n_validation_rows} rows')
        print(f'Training model with {len(features)} features...')
        # Callbacks
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            f'simple_fc_dnn_{fold + 1}.h5', 
            monitor = 'val_loss', 
            verbose = VERBOSE, 
            save_best_only = True,
            save_weights_only = True, 
            mode = 'min', 
            save_freq = 'epoch'
        )
        # Train and evaluate
        history = model.fit(
            x = x_train,
            y = y_train,
            batch_size = BATCH_SIZE,
            epochs = EPOCHS,
            verbose = VERBOSE,
            callbacks = [checkpoint],
            validation_data = (x_val, y_val),
        )
        # Predict validation set
        val_pred = model.predict(x_val, batch_size = BATCH_SIZE).astype(np.float32).reshape(-1)
        # Add validation prediction to out of folds array
        oof_predictions[val_ind] = val_pred
    # Compute out of folds Pearson Correlation Coefficient (for each time_id)
    oof_df = pd.DataFrame({'time_id': train['time_id'], 'target': train['target'], 'prediction': oof_predictions})
    # Save out of folds csv for blending
    oof_df.to_csv('simple_fc_dnn.csv', index = False)
    score = comp_metric(oof_df)
    print(f'Our out of folds mean pearson correlation coefficient is {score}')    
    
train_and_evaluate()

