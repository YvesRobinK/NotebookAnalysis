#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from tqdm.notebook import tqdm
import random
import warnings
import gc
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 100)


# In[2]:


# Configuration
EPOCHS = 20
BATCH_SIZE = 256
# Model Seed 
MODEL_SEED = 42
# Learning rate
LR = 0.001
# Folds
FOLDS = 5
# Verbosity
VERBOSE = 2


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
    return tf.convert_to_tensor(K.mean(tf.constant(1.0, dtype = x.dtype) - corr + (0.01 * sqdif)) , dtype = tf.float32 )

# Function to build our model
def build_model(shape, steps):
    def fc_block(x, units, dropout):
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(units, activation = 'swish')(x)
        return x
    # Input layer
    inp = tf.keras.layers.Input(shape = (shape))
    # Encoder block
    encoder = tf.keras.layers.GaussianNoise(0.015)(inp)
    encoder = tf.keras.layers.Dense(96)(encoder)
    encoder = tf.keras.layers.Activation('swish')(encoder)
    # Decoder block to predict the input to generate more features
    decoder = tf.keras.layers.Dropout(0.03)(encoder)
    decoder = tf.keras.layers.Dense(shape, activation = 'linear', name = 'decoder')(decoder)
    # Autoencoder
    autoencoder = tf.keras.layers.Dense(96)(decoder)
    autoencoder = tf.keras.layers.Activation('swish')(autoencoder)
    autoencoder = tf.keras.layers.Dropout(0.40)(autoencoder)
    out_autoencoder = tf.keras.layers.Dense(1, activation = 'linear', name = 'autoencoder')(autoencoder)
    # Concatenate input and encoder output for extra features
    x = tf.keras.layers.Concatenate()([inp, encoder])
    x = fc_block(x, units = 1024, dropout = 0.4)
    x = fc_block(x, units = 512, dropout = 0.4)
    x = fc_block(x, units = 256, dropout = 0.4)
    output = tf.keras.layers.Dense(1, activation = 'linear', name = 'mlp')(x)
    model = tf.keras.models.Model(inputs = [inp], outputs = [decoder, out_autoencoder, output])
    scheduler = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate = LR, decay_steps = steps, end_learning_rate = 0.00001)
    opt = tf.keras.optimizers.Adam(learning_rate = scheduler)
    model.compile(
        optimizer = opt,
        loss = [tf.keras.losses.MeanSquaredError(), tf.keras.losses.MeanSquaredError(), tf.keras.losses.MeanSquaredError()],
    )
    return model

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
    best_corr = corr.iloc[3:53, 0].to_list()
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
    # Initiate GroupKFold
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
        # Reset keras session
        K.clear_session()
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
            monitor = 'val_mlp_loss', 
            verbose = VERBOSE, 
            save_best_only = True,
            save_weights_only = True, 
            mode = 'min', 
            save_freq = 'epoch'
        )
        # Train and evaluate
        history = model.fit(
            x = x_train,
            y = (x_train, y_train, y_train),
            batch_size = BATCH_SIZE,
            epochs = EPOCHS,
            verbose = VERBOSE,
            callbacks = [checkpoint],
            validation_data = (x_val, (x_val, y_val, y_val)),
        )
        # Load best weights
        model.load_weights(f'simple_fc_dnn_{fold + 1}.h5')
        # Predict validation set
        val_pred = model.predict(x_val, batch_size = BATCH_SIZE)[2].reshape(-1)
        # Add validation prediction to out of folds array
        oof_predictions[val_ind] = val_pred
        del x_train, x_val, y_train, y_val
        gc.collect()
    # Compute out of folds Pearson Correlation Coefficient (for each time_id)
    oof_df = pd.DataFrame({'time_id': train['time_id'], 'target': train['target'], 'prediction': oof_predictions})
    # Save out of folds csv for blending
    oof_df.to_csv('simple_fc_dnn.csv', index = False)
    score = comp_metric(oof_df)
    print(f'Our out of folds mean pearson correlation coefficient is {score}')
    
train_and_evaluate()

