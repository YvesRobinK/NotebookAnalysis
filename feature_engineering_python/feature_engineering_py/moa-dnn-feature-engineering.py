#!/usr/bin/env python
# coding: utf-8

# # Comments
# 
# Add more seed to each model, as this takes more than 2 hours I save the trained models in kaggle datasets and load this model to make inference
# 
# Note: I check the out of folds score and is the same as my independent kernels, in other words inference is deterministic and is getting the same results as the training pipeline

# In[1]:


import sys
sys.path.append('../input/iterative-stratification/iterative-stratification-master')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from sklearn.metrics import log_loss
import random
import os
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


# In[2]:


FOLDS = 10
# Number of epochs to train each model
EPOCHS = 80
# Batch size
BATCH_SIZE = 124
# Learning rate
LR = 0.001
# Verbosity
VERBOSE = 0
# Seed for deterministic results
# Seed for deterministic results
SEEDS1 = [1, 2, 3, 4, 5, 6, 7]
SEEDS2 = [8, 9, 10, 11, 12, 13, 14]
SEEDS3 = [15, 16, 17, 18, 19, 20, 21]
SEEDS4 = [22, 23, 24, 25, 26, 27, 28]
SEEDS5 = [29, 30, 31, 32, 33, 34, 35]

# Function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


# In[3]:


# Function to map an filter out control group
def mapping_and_filter(train, train_targets, test):
    cp_type = {'trt_cp': 0, 'ctl_vehicle': 1}
    cp_dose = {'D1': 0, 'D2': 1}
    for df in [train, test]:
        df['cp_type'] = df['cp_type'].map(cp_type)
        df['cp_dose'] = df['cp_dose'].map(cp_dose)
    train_targets = train_targets[train['cp_type'] == 0].reset_index(drop = True)
    train = train[train['cp_type'] == 0].reset_index(drop = True)
    train_targets.drop(['sig_id'], inplace = True, axis = 1)
    return train, train_targets, test

# Function to scale our data
def scaling(train, test):
    features = train.columns[2:]
    scaler = RobustScaler()
    scaler.fit(pd.concat([train[features], test[features]], axis = 0))
    train[features] = scaler.transform(train[features])
    test[features] = scaler.transform(test[features])
    return train, test, features

# Function to extract pca features
def fe_pca(train, test, n_components_g = 70, n_components_c = 10, SEED = 123):
    
    features_g = list(train.columns[4:776])
    features_c = list(train.columns[776:876])
    
    def create_pca(train, test, features, kind = 'g', n_components = n_components_g):
        train_ = train[features].copy()
        test_ = test[features].copy()
        data = pd.concat([train_, test_], axis = 0)
        pca = PCA(n_components = n_components,  random_state = SEED)
        data = pca.fit_transform(data)
        columns = [f'pca_{kind}{i + 1}' for i in range(n_components)]
        data = pd.DataFrame(data, columns = columns)
        train_ = data.iloc[:train.shape[0]]
        test_ = data.iloc[train.shape[0]:].reset_index(drop = True)
        train = pd.concat([train, train_], axis = 1)
        test = pd.concat([test, test_], axis = 1)
        return train, test
    
    train, test = create_pca(train, test, features_g, kind = 'g', n_components = n_components_g)
    train, test = create_pca(train, test, features_c, kind = 'c', n_components = n_components_c)
    return train, test

# Function to extract common stats features
def fe_stats(train, test):
    
    features_g = list(train.columns[4:776])
    features_c = list(train.columns[776:876])
    
    for df in [train, test]:
        df['g_sum'] = df[features_g].sum(axis = 1)
        df['g_mean'] = df[features_g].mean(axis = 1)
        df['g_std'] = df[features_g].std(axis = 1)
        df['g_kurt'] = df[features_g].kurtosis(axis = 1)
        df['g_skew'] = df[features_g].skew(axis = 1)
        df['c_sum'] = df[features_c].sum(axis = 1)
        df['c_mean'] = df[features_c].mean(axis = 1)
        df['c_std'] = df[features_c].std(axis = 1)
        df['c_kurt'] = df[features_c].kurtosis(axis = 1)
        df['c_skew'] = df[features_c].skew(axis = 1)
        df['gc_sum'] = df[features_g + features_c].sum(axis = 1)
        df['gc_mean'] = df[features_g + features_c].mean(axis = 1)
        df['gc_std'] = df[features_g + features_c].std(axis = 1)
        df['gc_kurt'] = df[features_g + features_c].kurtosis(axis = 1)
        df['gc_skew'] = df[features_g + features_c].skew(axis = 1)
        
    return train, test

def c_squared(train, test):
    
    features_c = list(train.columns[776:876])
    for df in [train, test]:
        for feature in features_c:
            df[f'{feature}_squared'] = df[feature] ** 2
    return train, test

# Function to calculate the mean log loss of the targets including clipping
def mean_log_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    metrics = []
    for target in range(206):
        metrics.append(log_loss(y_true[:, target], y_pred[:, target]))
    return np.mean(metrics)

def create_model_rs(shape1, shape2):    
    input_1 = tf.keras.layers.Input(shape = (shape1))
    input_2 = tf.keras.layers.Input(shape = (shape2))
    
    head_1 = tf.keras.layers.BatchNormalization()(input_1)
    head_1 = tf.keras.layers.Dropout(0.2)(head_1)
    head_1 = tf.keras.layers.Dense(512, activation = "elu")(head_1)
    head_1 = tf.keras.layers.BatchNormalization()(head_1)
    input_3 = tf.keras.layers.Dense(256, activation = "elu")(head_1)

    input_3_concat = tf.keras.layers.Concatenate()([input_2, input_3])
    
    head_2 = tf.keras.layers.BatchNormalization()(input_3_concat)
    head_2 = tf.keras.layers.Dropout(0.3)(head_2)
    head_2 = tf.keras.layers.Dense(512, "relu")(head_2)
    head_2 = tf.keras.layers.BatchNormalization()(head_2)
    head_2 = tf.keras.layers.Dense(512, "elu")(head_2)
    head_2 = tf.keras.layers.BatchNormalization()(head_2)
    head_2 = tf.keras.layers.Dense(256, "relu")(head_2)
    head_2 = tf.keras.layers.BatchNormalization()(head_2)
    input_4 = tf.keras.layers.Dense(256, "elu")(head_2)

    input_4_avg = tf.keras.layers.Average()([input_3, input_4]) 
    
    head_3 = tf.keras.layers.BatchNormalization()(input_4_avg)
    head_3 = tf.keras.layers.Dense(256, kernel_initializer = 'lecun_normal', activation = 'selu')(head_3)
    head_3 = tf.keras.layers.BatchNormalization()(head_3)
    head_3 = tf.keras.layers.Dense(206, kernel_initializer = 'lecun_normal', activation = 'selu')(head_3)
    head_3 = tf.keras.layers.BatchNormalization()(head_3)
    output = tf.keras.layers.Dense(206, activation = "sigmoid")(head_3)

    model = tf.keras.models.Model(inputs = [input_1, input_2], outputs = output)
    opt = tf.optimizers.Adam(learning_rate = LR)
    model.compile(optimizer = opt, 
                  loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.0015), 
                  metrics = tf.keras.metrics.BinaryCrossentropy())
    
    return model


# Function to create our 5 layer dnn model
def create_model_5l(shape):
    inp = tf.keras.layers.Input(shape = (shape))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(2560, activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(2048, activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1524, activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1012, activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(780, activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(206, activation = 'sigmoid')(x)
    model = tf.keras.models.Model(inputs = inp, outputs = out)
    opt = tf.optimizers.Adam(learning_rate = LR)
    opt = tfa.optimizers.SWA(opt)
    model.compile(optimizer = opt, 
                  loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.0020),
                  metrics = tf.keras.metrics.BinaryCrossentropy())
    return model

# Function to create our 4 layer dnn model
def create_model_4l(shape):
    inp = tf.keras.layers.Input(shape = (shape))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(2048, activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1524, activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1012, activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1012, activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(206, activation = 'sigmoid')(x)
    model = tf.keras.models.Model(inputs = inp, outputs = out)
    opt = tf.optimizers.Adam(learning_rate = LR)
    model.compile(optimizer = opt, 
                  loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.0020),
                  metrics = tf.keras.metrics.BinaryCrossentropy())
    return model

# Function to create our 3 layer dnn model
def create_model_3l(shape):
    inp = tf.keras.layers.Input(shape = (shape))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(0.4914099166744246)(x)
    x = tfa.layers.WeightNormalization(tf.keras.layers.Dense(1159, activation = 'relu'))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.18817607797795838)(x)
    x = tfa.layers.WeightNormalization(tf.keras.layers.Dense(960, activation = 'relu'))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.12542057776853896)(x)
    x = tfa.layers.WeightNormalization(tf.keras.layers.Dense(1811, activation = 'relu'))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.20175242230280122)(x)
    out = tfa.layers.WeightNormalization(tf.keras.layers.Dense(206, activation = 'sigmoid'))(x)
    model = tf.keras.models.Model(inputs = inp, outputs = out)
    opt = tf.optimizers.Adam(learning_rate = LR)
    opt = tfa.optimizers.Lookahead(opt, sync_period = 10)
    model.compile(optimizer = opt, 
                  loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.0015),
                  metrics = tf.keras.metrics.BinaryCrossentropy())
    return model

# Function to create our 2 layer dnn model
def create_model_2l(shape):
    inp = tf.keras.layers.Input(shape = (shape))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(0.2688628097505064)(x)
    x = tfa.layers.WeightNormalization(tf.keras.layers.Dense(1292, activation = 'relu'))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4598218403250696)(x)
    x = tfa.layers.WeightNormalization(tf.keras.layers.Dense(983, activation = 'relu'))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4703144018483698)(x)
    out = tfa.layers.WeightNormalization(tf.keras.layers.Dense(206, activation = 'sigmoid'))(x)
    model = tf.keras.models.Model(inputs = inp, outputs = out)
    opt = tf.optimizers.Adam(learning_rate = LR)
    opt = tfa.optimizers.Lookahead(opt, sync_period = 10)
    model.compile(optimizer = opt, 
                  loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.0015),
                  metrics = tf.keras.metrics.BinaryCrossentropy())
    return model


# Function to train our dnn
def train_and_evaluate(train, test, train_targets, features, start_predictors, SEED = 123, MODEL = '3l'):
    seed_everything(SEED)
    oof_pred = np.zeros((train.shape[0], 206))
    test_pred = np.zeros((test.shape[0], 206))   
    for fold, (trn_ind, val_ind) in enumerate(MultilabelStratifiedKFold(n_splits = FOLDS, 
                                                                        random_state = SEED, 
                                                                        shuffle = True)\
                                              .split(train_targets, train_targets)):
        K.clear_session()
        if MODEL == '5l':
            model = create_model_5l(len(features))
        elif MODEL == '4l':
            model = create_model_4l(len(features))
        elif MODEL == '3l':
            model = create_model_3l(len(features))
        elif MODEL == '2l':
            model = create_model_2l(len(features))
        elif MODEL == "rs":
            model = create_model_rs(len(features), len(start_predictors))
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_binary_crossentropy',
                                                          mode = 'min',
                                                          patience = 10,
                                                          restore_best_weights = False,
                                                          verbose = VERBOSE)
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_binary_crossentropy',
                                                         mode = 'min',
                                                         factor = 0.3,
                                                         patience = 3,
                                                         verbose = VERBOSE)
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(f'{MODEL}_{fold}_{SEED}.h5',
                                                        monitor = 'val_binary_crossentropy',
                                                        verbose = VERBOSE,
                                                        save_best_only = True,
                                                        save_weights_only = True)
        
        x_train, x_val = train[features].values[trn_ind], train[features].values[val_ind]
        y_train, y_val = train_targets.values[trn_ind], train_targets.values[val_ind]
        
        if MODEL == "rs":
            x_train_, x_val_ = train[start_predictors].values[trn_ind], train[start_predictors].values[val_ind]

            model.fit([x_train, x_train_], y_train,
                  validation_data = ([x_val, x_val_], y_val),
                  epochs = EPOCHS, 
                  batch_size = BATCH_SIZE,
                  callbacks = [early_stopping, reduce_lr,  checkpoint],
                  verbose = VERBOSE)
        
            model.load_weights(f'{MODEL}_{fold}_{SEED}.h5')

            oof_pred[val_ind] = model.predict([x_val, x_val_])
            test_pred += model.predict([test[features].values, test[start_predictors].values]) / FOLDS
            
        else:
            model.fit(x_train, y_train,
                  validation_data = (x_val, y_val),
                  epochs = EPOCHS, 
                  batch_size = BATCH_SIZE,
                  callbacks = [early_stopping, reduce_lr,  checkpoint],
                  verbose = VERBOSE)
        
            model.load_weights(f'{MODEL}_{fold}_{SEED}.h5')

            oof_pred[val_ind] = model.predict(x_val)
            test_pred += model.predict(test[features].values) / FOLDS


    oof_score = mean_log_loss(train_targets.values, oof_pred)
    print(f'Our out of folds mean log loss score is {oof_score}')
    
    return test_pred, oof_pred

# Function to train our dnn
def inference(train, test, train_targets, features, start_predictors, SEED = 123, MODEL = '3l', PATH = '../input/moa-3layer'):
    seed_everything(SEED)
    oof_pred = np.zeros((train.shape[0], 206))
    test_pred = np.zeros((test.shape[0], 206))   
    for fold, (trn_ind, val_ind) in enumerate(MultilabelStratifiedKFold(n_splits = FOLDS, 
                                                                        random_state = SEED, 
                                                                        shuffle = True)\
                                              .split(train_targets, train_targets)):
        K.clear_session()
        if MODEL == '5l':
            model = create_model_5l(len(features))
        elif MODEL == '4l':
            model = create_model_4l(len(features))
        elif MODEL == '3l':
            model = create_model_3l(len(features))
        elif MODEL == '2l':
            model = create_model_2l(len(features))
        elif MODEL == "rs":
            model = create_model_rs(len(features), len(start_predictors))


        x_train, x_val = train[features].values[trn_ind], train[features].values[val_ind]
        y_train, y_val = train_targets.values[trn_ind], train_targets.values[val_ind]
        
        model.load_weights(f'{PATH}/{MODEL}_{fold}_{SEED}.h5')
        
        if MODEL == "rs":
            x_train_, x_val_ = train[start_predictors].values[trn_ind], train[start_predictors].values[val_ind]
            oof_pred[val_ind] = model.predict([x_val, x_val_])
            test_pred += model.predict([test[features].values, test[start_predictors].values]) / FOLDS
        else:
            oof_pred[val_ind] = model.predict(x_val)
            test_pred += model.predict(test[features].values) / FOLDS

    oof_score = mean_log_loss(train_targets.values, oof_pred)
    print(f'Our out of folds mean log loss score is {oof_score}')
    
    return test_pred, oof_pred
    
    

# Function to train our model with multiple seeds and average the predictions
def run_multiple_seeds(train, test, train_targets, features, start_predictors, SEEDS = [123], MODEL = '3l', PATH = '../input/moa-3layer'):
    
    test_pred = []
    oof_pred = []
    
    for SEED in SEEDS:
        print(f'Using model {MODEL} with seed {SEED} for inference')
        print(f'Trained with {len(features)} features')
        test_pred_, oof_pred_ = inference(train, test, train_targets, features, start_predictors, SEED = SEED, MODEL = MODEL, PATH = PATH)
        test_pred.append(test_pred_)
        oof_pred.append(oof_pred_)
        print('-'*50)
        print('\n')
        
    test_pred = np.average(test_pred, axis = 0)
    oof_pred = np.average(oof_pred, axis = 0)
        
    seed_log_loss = mean_log_loss(train_targets.values, oof_pred)
    print(f'Our out of folds log loss for our seed blend model is {seed_log_loss}')
    
    return test_pred, oof_pred

def submission(test_pred):
    sample_submission.loc[:, train_targets.columns] = test_pred
    sample_submission.loc[test['cp_type'] == 1, train_targets.columns] = 0
    sample_submission.to_csv('submission.csv', index = False)
    return sample_submission

# Got this predictors from public kernels for the resnet type model
start_predictors = ["g-0", "g-7", "g-8", "g-10", "g-13", "g-17", "g-20", "g-22", "g-24", "g-26", "g-28", "g-29", "g-30", "g-31", "g-32", "g-34", "g-35", "g-36", "g-37", "g-38",
                    "g-39","g-41", "g-46", "g-48", "g-50", "g-51", "g-52", "g-55", "g-58", "g-59", "g-61", "g-62", "g-63", "g-65", "g-66", "g-67", "g-68", "g-70", "g-72", "g-74", 
                    "g-75", "g-79", "g-83", "g-84", "g-85", "g-86", "g-90", "g-91", "g-94", "g-95", "g-96", "g-97", "g-98", "g-100", "g-102", "g-105", "g-106", "g-112", "g-113", 
                    "g-114", "g-116", "g-121", "g-123", "g-126", "g-128", "g-131", "g-132", "g-134", "g-135", "g-138", "g-139", "g-140", "g-142", "g-144", "g-145", "g-146", 
                    "g-147", "g-148", "g-152", "g-155", "g-157", "g-158", "g-160", "g-163", "g-164", "g-165", "g-170", "g-173", "g-174", "g-175", "g-177", "g-178", "g-181", 
                    "g-183", "g-185", "g-186", "g-189", "g-192", "g-194", "g-195", "g-196", "g-197", "g-199", "g-201", "g-202", "g-206", "g-208", "g-210", "g-213", "g-214", 
                    "g-215", "g-220", "g-226", "g-228", "g-229", "g-235", "g-238", "g-241", "g-242", "g-243", "g-244", "g-245", "g-248", "g-250", "g-251", "g-254", "g-257", 
                    "g-259", "g-261", "g-266", "g-270", "g-271", "g-272", "g-275", "g-278", "g-282", "g-287", "g-288", "g-289", "g-291", "g-293", "g-294", "g-297", "g-298",
                    "g-301", "g-303", "g-304", "g-306", "g-308", "g-309", "g-310", "g-311", "g-314", "g-315", "g-316", "g-317", "g-320", "g-321", "g-322", "g-327", "g-328", 
                    "g-329", "g-332", "g-334", "g-335", "g-336", "g-337", "g-339", "g-342", "g-344", "g-349", "g-350", "g-351", "g-353", "g-354", "g-355", "g-357", "g-359", 
                    "g-360", "g-364", "g-365", "g-366", "g-367", "g-368", "g-369", "g-374", "g-375", "g-377", "g-379", "g-385", "g-386", "g-390", "g-392", "g-393", "g-400", 
                    "g-402", "g-406", "g-407", "g-409", "g-410", "g-411", "g-414", "g-417", "g-418", "g-421", "g-423", "g-424", "g-427", "g-429", "g-431", "g-432", "g-433", 
                    "g-434", "g-437", "g-439", "g-440", "g-443", "g-449", "g-458", "g-459", "g-460", "g-461", "g-464", "g-467", "g-468", "g-470", "g-473", "g-477", "g-478", 
                    "g-479", "g-484", "g-485", "g-486", "g-488", "g-489", "g-491", "g-494", "g-496", "g-498", "g-500", "g-503", "g-504", "g-506", "g-508", "g-509", "g-512", 
                    "g-522", "g-529", "g-531", "g-534", "g-539", "g-541", "g-546", "g-551", "g-553", "g-554", "g-559", "g-561", "g-562", "g-565", "g-568", "g-569", "g-574", 
                    "g-577", "g-578", "g-586", "g-588", "g-590", "g-594", "g-595", "g-596", "g-597", "g-599", "g-600", "g-603", "g-607", "g-615", "g-618", "g-619", "g-620", 
                    "g-625", "g-628", "g-629", "g-632", "g-634", "g-635", "g-636", "g-638", "g-639", "g-641", "g-643", "g-644", "g-645", "g-646", "g-647", "g-648", "g-663", 
                    "g-664", "g-665", "g-668", "g-669", "g-670", "g-671", "g-672", "g-673", "g-674", "g-677", "g-678", "g-680", "g-683", "g-689", "g-691", "g-693", "g-695", 
                    "g-701", "g-702", "g-703", "g-704", "g-705", "g-706", "g-708", "g-711", "g-712", "g-720", "g-721", "g-723", "g-724", "g-726", "g-728", "g-731", "g-733", 
                    "g-738", "g-739", "g-742", "g-743", "g-744", "g-745", "g-749", "g-750", "g-752", "g-760", "g-761", "g-764", "g-766", "g-768", "g-770", "g-771", "c-0", 
                    "c-1", "c-2", "c-3", "c-4", "c-5", "c-6", "c-7", "c-8", "c-9", "c-10", "c-11", "c-12", "c-13", "c-14", "c-15", "c-16", "c-17", "c-18", "c-19", "c-20", 
                    "c-21", "c-22", "c-23", "c-24", "c-25", "c-26", "c-27", "c-28", "c-29", "c-30", "c-31", "c-32", "c-33", "c-34", "c-35", "c-36", "c-37", "c-38", "c-39", 
                    "c-40", "c-41", "c-42", "c-43", "c-44", "c-45", "c-46", "c-47", "c-48", "c-49", "c-50", "c-51", "c-52", "c-53", "c-54", "c-55", "c-56", "c-57", "c-58", 
                    "c-59", "c-60", "c-61", "c-62", "c-63", "c-64", "c-65", "c-66", "c-67", "c-68", "c-69", "c-70", "c-71", "c-72", "c-73", "c-74", "c-75", "c-76", "c-77", 
                    "c-78", "c-79", "c-80", "c-81", "c-82", "c-83", "c-84", "c-85", "c-86", "c-87", "c-88", "c-89", "c-90", "c-91", "c-92", "c-93", "c-94", "c-95", "c-96", 
                    "c-97", "c-98", "c-99"]


train = pd.read_csv('../input/lish-moa/train_features.csv')
train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
test = pd.read_csv('../input/lish-moa/test_features.csv')
sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')
train, train_targets, test = mapping_and_filter(train, train_targets, test)
train, test = fe_stats(train, test)
train, test = c_squared(train, test)
train, test = fe_pca(train, test, n_components_g = 70, n_components_c = 10, SEED = 123)
train, test, features = scaling(train, test)

# Inference time
test_pred_5l, oof_pred_5l = run_multiple_seeds(train, test, train_targets, features, start_predictors, SEEDS = SEEDS1, MODEL = '5l', PATH = '../input/moa-5layer')
test_pred_4l, oof_pred_4l = run_multiple_seeds(train, test, train_targets, features, start_predictors, SEEDS = SEEDS2, MODEL = '4l', PATH = '../input/moa-4layer')
test_pred_3l, oof_pred_3l = run_multiple_seeds(train, test, train_targets, features, start_predictors, SEEDS = SEEDS3, MODEL = '3l', PATH = '../input/moa-3layer')
test_pred_2l, oof_pred_2l = run_multiple_seeds(train, test, train_targets, features, start_predictors, SEEDS = SEEDS4, MODEL = '2l', PATH = '../input/moa-2layer')
test_pred_rs, oof_pred_rs = run_multiple_seeds(train, test, train_targets, features, start_predictors, SEEDS = SEEDS5, MODEL = 'rs', PATH = '../input/moa-rs')

# Blend 5l, 4l, 3l and l2 dnn model
oof_pred = np.average([oof_pred_5l, oof_pred_4l, oof_pred_3l, oof_pred_2l], axis = 0)
seed_log_loss = mean_log_loss(train_targets.values, oof_pred)
print(f'Our final out of folds log loss for our classic dnn blend is {seed_log_loss}')
test_pred = np.average([test_pred_5l, test_pred_4l, test_pred_3l, test_pred_2l], axis = 0)

# Blend the result of the previous model with the dnn resnet type model
oof_pred = np.average([oof_pred, oof_pred_rs], axis = 0)
seed_log_loss = mean_log_loss(train_targets.values, oof_pred)
print(f'Our final out of folds log loss for our classic dnn + dnn resnet type model is {seed_log_loss}')
test_pred = np.average([test_pred, test_pred_rs], axis = 0)

sample_submission = submission(test_pred)
sample_submission.head()

