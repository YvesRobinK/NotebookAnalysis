#!/usr/bin/env python
# coding: utf-8

# # Keras Quickstart for the AMEX Competition: Training and Inference
# 
# This notebook shows
# - how to do space-efficient feature engineering
# - how to implement a simple Keras model
# - how to train and cross-validate the model
# - how to understand the competition metric graphically
# 
# The notebook is based on insights of the [EDA which makes sense ⭐️⭐️⭐️⭐️⭐️](https://www.kaggle.com/code/ambrosm/amex-eda-which-makes-sense).

# In[1]:


import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
import random
import datetime
import math
from matplotlib.ticker import MaxNLocator
from colorama import Fore, Back, Style
import gc

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OneHotEncoder
from sklearn.metrics import roc_curve, roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from tensorflow.keras.layers import Dense, Input, InputLayer, Add, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.utils import plot_model

INFERENCE = True


# In[2]:


# Plot training history
def plot_history(history, *, n_epochs=None, plot_lr=False, title=None, bottom=None, top=None):
    """Plot (the last n_epochs epochs of) the training history
    
    Plots loss and optionally val_loss and lr."""
    plt.figure(figsize=(15, 6))
    from_epoch = 0 if n_epochs is None else max(len(history['loss']) - n_epochs, 0)
    
    # Plot training and validation losses
    plt.plot(np.arange(from_epoch, len(history['loss'])), history['loss'][from_epoch:], label='Training loss')
    try:
        plt.plot(np.arange(from_epoch, len(history['loss'])), history['val_loss'][from_epoch:], label='Validation loss')
        best_epoch = np.argmin(np.array(history['val_loss']))
        best_val_loss = history['val_loss'][best_epoch]
        if best_epoch >= from_epoch:
            plt.scatter([best_epoch], [best_val_loss], c='r', label=f'Best val_loss = {best_val_loss:.5f}')
        if best_epoch > 0:
            almost_epoch = np.argmin(np.array(history['val_loss'])[:best_epoch])
            almost_val_loss = history['val_loss'][almost_epoch]
            if almost_epoch >= from_epoch:
                plt.scatter([almost_epoch], [almost_val_loss], c='orange', label='Second best val_loss')
    except KeyError:
        pass
    if bottom is not None: plt.ylim(bottom=bottom)
    if top is not None: plt.ylim(top=top)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower left')
    if title is not None: plt.title(title)
        
    # Plot learning rate
    if plot_lr and 'lr' in history:
        ax2 = plt.gca().twinx()
        ax2.plot(np.arange(from_epoch, len(history['lr'])), np.array(history['lr'][from_epoch:]), color='g', label='Learning rate')
        ax2.set_ylabel('Learning rate')
        ax2.legend(loc='upper right')
        
    plt.show()


# In[3]:


# From https://www.kaggle.com/code/inversion/amex-competition-metric-python
def amex_metric(y_true, y_pred, return_components=False) -> float:
    """Amex metric for ndarrays"""
    def top_four_percent_captured(df) -> float:
        """Corresponds to the recall for a threshold of 4 %"""
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
    def weighted_gini(df) -> float:
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(df) -> float:
        """Corresponds to 2 * AUC - 1"""
        df2 = pd.DataFrame({'target': df.target, 'prediction': df.target})
        df2.sort_values('prediction', ascending=False, inplace=True)
        return weighted_gini(df) / weighted_gini(df2)

    df = pd.DataFrame({'target': y_true.ravel(), 'prediction': y_pred.ravel()})
    df.sort_values('prediction', ascending=False, inplace=True)
    g = normalized_weighted_gini(df)
    d = top_four_percent_captured(df)

    if return_components: return g, d, 0.5 * (g + d)
    return 0.5 * (g + d)


# # Reading and preprocessing the training data
# 
# We read the data from @raddar's [dataset](https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format). @raddar has [denoised the data](https://www.kaggle.com/competitions/amex-default-prediction/discussion/328514) so that we can achieve better results with his dataset than with the original competition csv files.
# 
# Then we create several groups of features:
# - Selected features averaged over all statements of a customer
# - Minimum / maximum of selected features over all statements of a customer
# - Selected features taken from the last statement of a customer
# 
# We one-hot encode the categorical features and fill all missing values with 0.
# 
# The code has been optimized for memory efficiency rather than readability. In particular, `.iloc[mask_array, columns]` needs much less RAM than the groupby construction used in previous versions of the notebook. Deleting the index of the train dataframe frees another 0.2 GByte.
# 

# In[4]:


get_ipython().run_cell_magic('time', '', 'features_avg = [\'B_11\', \'B_13\', \'B_14\', \'B_15\', \'B_16\', \'B_17\', \'B_18\', \'B_19\', \'B_2\', \n                \'B_20\', \'B_28\', \'B_29\', \'B_3\', \'B_33\', \'B_36\', \'B_37\', \'B_4\', \'B_42\', \n                \'B_5\', \'B_8\', \'B_9\', \'D_102\', \'D_103\', \'D_105\', \'D_111\', \'D_112\', \'D_113\', \n                \'D_115\', \'D_118\', \'D_119\', \'D_121\', \'D_124\', \'D_128\', \'D_129\', \'D_131\', \n                \'D_132\', \'D_133\', \'D_139\', \'D_140\', \'D_141\', \'D_143\', \'D_144\', \'D_145\', \n                \'D_39\', \'D_41\', \'D_42\', \'D_43\', \'D_44\', \'D_45\', \'D_46\', \'D_47\', \'D_48\', \n                \'D_49\', \'D_50\', \'D_51\', \'D_52\', \'D_56\', \'D_58\', \'D_62\', \'D_70\', \'D_71\', \n                \'D_72\', \'D_74\', \'D_75\', \'D_79\', \'D_81\', \'D_83\', \'D_84\', \'D_88\', \'D_91\', \n                \'P_2\', \'P_3\', \'R_1\', \'R_10\', \'R_11\', \'R_13\', \'R_18\', \'R_19\', \'R_2\', \'R_26\', \n                \'R_27\', \'R_28\', \'R_3\', \'S_11\', \'S_12\', \'S_22\', \'S_23\', \'S_24\', \'S_26\', \n                \'S_27\', \'S_5\', \'S_7\', \'S_8\', ]\nfeatures_min = [\'B_13\', \'B_14\', \'B_15\', \'B_16\', \'B_17\', \'B_19\', \'B_2\', \'B_20\', \'B_22\', \n                \'B_24\', \'B_27\', \'B_28\', \'B_29\', \'B_3\', \'B_33\', \'B_36\', \'B_4\', \'B_42\', \n                \'B_5\', \'B_9\', \'D_102\', \'D_103\', \'D_107\', \'D_109\', \'D_110\', \'D_111\', \n                \'D_112\', \'D_113\', \'D_115\', \'D_118\', \'D_119\', \'D_121\', \'D_122\', \'D_128\', \n                \'D_129\', \'D_132\', \'D_133\', \'D_139\', \'D_140\', \'D_141\', \'D_143\', \'D_144\', \n                \'D_145\', \'D_39\', \'D_41\', \'D_42\', \'D_45\', \'D_46\', \'D_48\', \'D_50\', \'D_51\', \n                \'D_53\', \'D_54\', \'D_55\', \'D_56\', \'D_58\', \'D_59\', \'D_60\', \'D_62\', \'D_70\', \n                \'D_71\', \'D_74\', \'D_75\', \'D_78\', \'D_79\', \'D_81\', \'D_83\', \'D_84\', \'D_86\', \n                \'D_88\', \'D_96\', \'P_2\', \'P_3\', \'P_4\', \'R_1\', \'R_11\', \'R_13\', \'R_17\', \'R_19\', \n                \'R_2\', \'R_27\', \'R_28\', \'R_4\', \'R_5\', \'R_8\', \'S_11\', \'S_12\', \'S_23\', \'S_25\', \n                \'S_3\', \'S_5\', \'S_7\', \'S_9\', ]\nfeatures_max = [\'B_1\', \'B_11\', \'B_13\', \'B_15\', \'B_16\', \'B_17\', \'B_18\', \'B_19\', \'B_2\', \n                \'B_22\', \'B_24\', \'B_27\', \'B_28\', \'B_29\', \'B_3\', \'B_31\', \'B_33\', \'B_36\', \n                \'B_4\', \'B_42\', \'B_5\', \'B_7\', \'B_9\', \'D_102\', \'D_103\', \'D_105\', \'D_109\', \n                \'D_110\', \'D_112\', \'D_113\', \'D_115\', \'D_121\', \'D_124\', \'D_128\', \'D_129\', \n                \'D_131\', \'D_139\', \'D_141\', \'D_144\', \'D_145\', \'D_39\', \'D_41\', \'D_42\', \n                \'D_43\', \'D_44\', \'D_45\', \'D_46\', \'D_47\', \'D_48\', \'D_50\', \'D_51\', \'D_52\', \n                \'D_53\', \'D_56\', \'D_58\', \'D_59\', \'D_60\', \'D_62\', \'D_70\', \'D_72\', \'D_74\', \n                \'D_75\', \'D_79\', \'D_81\', \'D_83\', \'D_84\', \'D_88\', \'D_89\', \'P_2\', \'P_3\', \n                \'R_1\', \'R_10\', \'R_11\', \'R_26\', \'R_28\', \'R_3\', \'R_4\', \'R_5\', \'R_7\', \'R_8\', \n                \'S_11\', \'S_12\', \'S_23\', \'S_25\', \'S_26\', \'S_27\', \'S_3\', \'S_5\', \'S_7\', \'S_8\', ]\nfeatures_last = [\'B_1\', \'B_11\', \'B_12\', \'B_13\', \'B_14\', \'B_16\', \'B_18\', \'B_19\', \'B_2\', \n                 \'B_20\', \'B_21\', \'B_24\', \'B_27\', \'B_28\', \'B_29\', \'B_3\', \'B_30\', \'B_31\', \n                 \'B_33\', \'B_36\', \'B_37\', \'B_38\', \'B_39\', \'B_4\', \'B_40\', \'B_42\', \'B_5\', \n                 \'B_8\', \'B_9\', \'D_102\', \'D_105\', \'D_106\', \'D_107\', \'D_108\', \'D_110\', \n                 \'D_111\', \'D_112\', \'D_113\', \'D_114\', \'D_115\', \'D_116\', \'D_117\', \'D_118\', \n                 \'D_119\', \'D_120\', \'D_121\', \'D_124\', \'D_126\', \'D_128\', \'D_129\', \'D_131\', \n                 \'D_132\', \'D_133\', \'D_137\', \'D_138\', \'D_139\', \'D_140\', \'D_141\', \'D_142\', \n                 \'D_143\', \'D_144\', \'D_145\', \'D_39\', \'D_41\', \'D_42\', \'D_43\', \'D_44\', \'D_45\', \n                 \'D_46\', \'D_47\', \'D_48\', \'D_49\', \'D_50\', \'D_51\', \'D_52\', \'D_53\', \'D_55\', \n                 \'D_56\', \'D_59\', \'D_60\', \'D_62\', \'D_63\', \'D_64\', \'D_66\', \'D_68\', \'D_70\', \n                 \'D_71\', \'D_72\', \'D_73\', \'D_74\', \'D_75\', \'D_77\', \'D_78\', \'D_81\', \'D_82\', \n                 \'D_83\', \'D_84\', \'D_88\', \'D_89\', \'D_91\', \'D_94\', \'D_96\', \'P_2\', \'P_3\', \n                 \'P_4\', \'R_1\', \'R_10\', \'R_11\', \'R_12\', \'R_13\', \'R_16\', \'R_17\', \'R_18\', \n                 \'R_19\', \'R_25\', \'R_28\', \'R_3\', \'R_4\', \'R_5\', \'R_8\', \'S_11\', \'S_12\', \n                 \'S_23\', \'S_25\', \'S_26\', \'S_27\', \'S_3\', \'S_5\', \'S_7\', \'S_8\', \'S_9\', ]\nfeatures_categorical = [\'B_30_last\', \'B_38_last\', \'D_114_last\', \'D_116_last\',\n                        \'D_117_last\', \'D_120_last\', \'D_126_last\',\n                        \'D_63_last\', \'D_64_last\', \'D_66_last\', \'D_68_last\']\n\nfor i in [\'train\', \'test\'] if INFERENCE else [\'train\']:\n    df = pd.read_parquet(f\'../input/amex-data-integer-dtypes-parquet-format/{i}.parquet\')\n    cid = pd.Categorical(df.pop(\'customer_ID\'), ordered=True)\n    last = (cid != np.roll(cid, -1)) # mask for last statement of every customer\n    if \'target\' in df.columns:\n        df.drop(columns=[\'target\'], inplace=True)\n    print(\'Read\', i)\n    gc.collect()\n    df_avg = (df\n              .groupby(cid)\n              .mean()[features_avg]\n              .rename(columns={f: f"{f}_avg" for f in features_avg})\n             )\n    print(\'Computed avg\', i)\n    gc.collect()\n    df_max = (df\n              .groupby(cid)\n              .max()[features_max]\n              .rename(columns={f: f"{f}_max" for f in features_max})\n             )\n    print(\'Computed max\', i)\n    gc.collect()\n    df_min = (df\n              .groupby(cid)\n              .min()[features_min]\n              .rename(columns={f: f"{f}_min" for f in features_min})\n             )\n    print(\'Computed min\', i)\n    gc.collect()\n    df_last = (df.loc[last, features_last]\n               .rename(columns={f: f"{f}_last" for f in features_last})\n               .set_index(np.asarray(cid[last]))\n              )\n    df = None # we no longer need the original data\n    print(\'Computed last\', i)\n    \n    df_categorical = df_last[features_categorical].astype(object)\n    features_not_cat = [f for f in df_last.columns if f not in features_categorical]\n    if i == \'train\':\n        ohe = OneHotEncoder(drop=\'first\', sparse=False, dtype=np.float32, handle_unknown=\'ignore\')\n        ohe.fit(df_categorical)\n        with open("ohe.pickle", \'wb\') as f: pickle.dump(ohe, f)\n    df_categorical = pd.DataFrame(ohe.transform(df_categorical).astype(np.float16),\n                                  index=df_categorical.index).rename(columns=str)\n    print(\'Computed categorical\', i)\n    \n    df = pd.concat([df_last[features_not_cat], df_categorical, df_avg, df_min, df_max], axis=1)\n    \n    # Impute missing values\n    df.fillna(value=0, inplace=True)\n    \n    del df_avg, df_max, df_min, df_last, df_categorical, cid, last, features_not_cat\n    \n    print(f"{i} shape: {df.shape}")\n    if i == \'train\': # train\n        # Free the memory\n        df.reset_index(drop=True, inplace=True) # frees 0.2 GByte\n        df.to_feather(\'train_processed.ftr\')\n        df = None\n        gc.collect()\n\ntrain = pd.read_feather(\'train_processed.ftr\')\n!rm train_processed.ftr\ntest = df\ndel df, ohe\n\ntarget = pd.read_csv(\'../input/amex-default-prediction/train_labels.csv\').target.values\nprint(f"target shape: {target.shape}")\n')


# # The model
# 
# Our model has four hidden layers, enriched by a skip connection and a Dropout layer.

# In[5]:


LR_START = 0.01

features = [f for f in train.columns if f != 'target' and f != 'customer_ID']

def my_model(n_inputs=len(features)):
    """Sequential neural network with a skip connection.
    
    Returns a compiled instance of tensorflow.keras.models.Model.
    """
    activation = 'swish'
    reg = 4e-4
    inputs = Input(shape=(n_inputs, ))
    x0 = Dense(256, kernel_regularizer=tf.keras.regularizers.l2(reg),
              activation=activation,
             )(inputs)
    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(reg),
              activation=activation,
             )(x0)
    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(reg),
              activation=activation,
             )(x)
    x = Concatenate()([x, x0])
    x = Dropout(0.1)(x)
    #x = BatchNormalization()(x)
    x = Dense(16, kernel_regularizer=tf.keras.regularizers.l2(reg),
              activation=activation,
             )(x)
    x = Dense(1, #kernel_regularizer=tf.keras.regularizers.l2(4e-4),
              activation='sigmoid',
             )(x)
    model = Model(inputs, x)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR_START),
                  loss=tf.keras.losses.BinaryCrossentropy())
    return model

plot_model(my_model(), show_layer_names=False, show_shapes=True)


# # Cross-validation
# 
# We use a standard cross-validation loop. In the loop, we scale the data and train a model. We use a StratifiedKFold because the data is imbalanced.
# 

# In[6]:


get_ipython().run_cell_magic('time', '', '# Cross-validation of the classifier\n\nONLY_FIRST_FOLD = False\nEPOCHS_EXPONENTIALDECAY = 100\nVERBOSE = 0 # set to 0 for less output, or to 2 for more output\nLR_END = 1e-5 # learning rate at the end of training\nCYCLES = 1\nEPOCHS = 200\nDIAGRAMS = True\nUSE_PLATEAU = False # set to True for early stopping, or to False for exponential learning rate decay\nBATCH_SIZE = 2048\n\nnp.random.seed(1)\nrandom.seed(1)\ntf.random.set_seed(1)\n\ndef fit_model(X_tr, y_tr, X_va=None, y_va=None, fold=0, run=0):\n    """Scale the data, fit a model, plot the training history and optionally validate the model\n    \n    Saves a trained instance of tensorflow.keras.models.Model.\n    \n    As a side effect, updates y_va_pred, history_list, y_pred_list and score_list.\n    """\n    global y_va_pred\n    gc.collect()\n    start_time = datetime.datetime.now()\n    \n    scaler = StandardScaler()\n    X_tr = scaler.fit_transform(X_tr)\n    \n    if X_va is not None:\n        X_va = scaler.transform(X_va)\n        validation_data = (X_va, y_va)\n    else:\n        validation_data = None\n    # Define the learning rate schedule and EarlyStopping\n    if USE_PLATEAU and X_va is not None: # use early stopping\n        epochs = EPOCHS\n        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.7, \n                               patience=4, verbose=VERBOSE)\n        es = EarlyStopping(monitor="val_loss",\n                           patience=12, \n                           verbose=1,\n                           mode="min", \n                           restore_best_weights=True)\n        callbacks = [lr, es, tf.keras.callbacks.TerminateOnNaN()]\n\n    else: # use exponential learning rate decay rather than early stopping\n        epochs = EPOCHS_EXPONENTIALDECAY\n\n        def exponential_decay(epoch):\n            # v decays from e^a to 1 in every cycle\n            # w decays from 1 to 0 in every cycle\n            # epoch == 0                  -> w = 1 (first epoch of cycle)\n            # epoch == epochs_per_cycle-1 -> w = 0 (last epoch of cycle)\n            # higher a -> decay starts with a steeper decline\n            a = 3\n            epochs_per_cycle = epochs // CYCLES\n            epoch_in_cycle = epoch % epochs_per_cycle\n            if epochs_per_cycle > 1:\n                v = math.exp(a * (1 - epoch_in_cycle / (epochs_per_cycle-1)))\n                w = (v - 1) / (math.exp(a) - 1)\n            else:\n                w = 1\n            return w * LR_START + (1 - w) * LR_END\n\n        lr = LearningRateScheduler(exponential_decay, verbose=0)\n        callbacks = [lr, tf.keras.callbacks.TerminateOnNaN()]\n        \n    # Construct and compile the model\n    model = my_model(X_tr.shape[1])\n    # Train the model\n    history = model.fit(X_tr, y_tr, \n                        validation_data=validation_data, \n                        epochs=epochs,\n                        verbose=VERBOSE,\n                        batch_size=BATCH_SIZE,\n                        shuffle=True,\n                        callbacks=callbacks)\n    del X_tr, y_tr\n    with open(f"scaler_{fold}.pickle", \'wb\') as f: pickle.dump(scaler, f)\n    model.save(f"model_{fold}")\n    history_list.append(history.history)\n    callbacks, es, lr, history = None, None, None, None\n    \n    if X_va is None:\n        print(f"Training loss: {history_list[-1][\'loss\'][-1]:.4f}")\n    else:\n        lastloss = f"Training loss: {history_list[-1][\'loss\'][-1]:.4f} | Val loss: {history_list[-1][\'val_loss\'][-1]:.4f}"\n        \n        # Inference for validation\n        y_va_pred = model.predict(X_va, batch_size=len(X_va), verbose=0).ravel()\n        \n        # Evaluation: Execution time, loss and metrics\n        score = amex_metric(y_va, y_va_pred)\n        print(f"{Fore.GREEN}{Style.BRIGHT}Fold {run}.{fold} | {str(datetime.datetime.now() - start_time)[-12:-7]}"\n              f" | {len(history_list[-1][\'loss\']):3} ep"\n              f" | {lastloss} | Score: {score:.5f}{Style.RESET_ALL}")\n        score_list.append(score)\n        \n        if DIAGRAMS and fold == 0 and run == 0:\n            # Plot training history\n            plot_history(history_list[-1], \n                         title=f"Learning curve",\n                         plot_lr=True)\n\n            # Plot prediction histogram\n            plt.figure(figsize=(16, 5))\n            plt.hist(y_va_pred[y_va == 0], bins=np.linspace(0, 1, 21),\n                     alpha=0.5, density=True)\n            plt.hist(y_va_pred[y_va == 1], bins=np.linspace(0, 1, 21),\n                     alpha=0.5, density=True)\n            plt.xlabel(\'y_pred\')\n            plt.ylabel(\'density\')\n            plt.title(\'OOF Prediction Histogram\')\n            plt.show()\n\n        # Scale and predict\n        y_pred_list.append(model.predict(scaler.transform(test), batch_size=128*1024, verbose=0).ravel())\n        with np.printoptions(linewidth=150, precision=2, suppress=True):\n            print(f"Test pred {fold}", y_pred_list[-1])\n\n\nprint(f"{len(features)} features")\nhistory_list = []\nscore_list = []\ny_pred_list = []\nkf = StratifiedKFold(n_splits=10)\nfor fold, (idx_tr, idx_va) in enumerate(kf.split(train, target)):\n    y_va = target[idx_va]\n    tf.keras.backend.clear_session()\n    gc.collect()\n    fit_model(train.iloc[idx_tr][features], target[idx_tr], \n              train.iloc[idx_va][features], y_va, fold=fold)\n    if ONLY_FIRST_FOLD: break # we only need the first fold\n\nprint(f"{Fore.GREEN}{Style.BRIGHT}OOF Score:                       {np.mean(score_list):.5f}{Style.RESET_ALL}")\n')


# # Understanding the competition metric
# 
# Assuming you know the [ROC (receiver operating characteristic) curve](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics), the competition metric has a simple graphical explanation. The following diagram shows the ROC curve for the last fold in dark red. The area under the curve (AUC) is filled with light red. The green line corresponds to 4 % of all samples.

# In[7]:


g, d, amex = amex_metric(y_va, y_va_pred, return_components=True)
total_positive = (y_va == 1).sum()
total_negative = (y_va == 0).sum() * 20
fourpercent = int(0.04 * (total_positive + total_negative))

plt.figure(figsize=(6, 6))
fpr, tpr, _ = roc_curve(y_va, y_va_pred)
plt.plot(fpr, tpr, color='#c00000', lw=3) # curve
plt.fill_between(fpr, tpr, color='#ffc0c0') # area under the curve
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--") # diagonal
plt.plot([fourpercent / total_negative, 0], [0, fourpercent / total_positive],
         color="green", lw=3, linestyle="-") # four percent line
four_percent_index = np.argmax((fpr * total_negative + tpr * total_positive >= fourpercent))
plt.scatter([fpr[four_percent_index]],
            [tpr[four_percent_index]], 
            s=100) # intersection of roc curve with four percent line
plt.gca().set_aspect('equal')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Receiver operating characteristic")
plt.show()

print(f"Area under the curve (AUC):         {roc_auc_score(y_va, y_va_pred):.5f}")
print(f"2*AUC-1:                            {2 * roc_auc_score(y_va, y_va_pred) - 1:.5f}")
print(f"Normalized Gini coefficient:        {g:.5f} (same as 2*AUC-1)")
print()
print(f"Positive samples in validation set: {total_positive:7}")
#print(f"Negative samples in validation set: {total_negative // 20:7}")
print(f"Negative samples weighted:          {total_negative:7} (unweighted: {total_negative // 20})")
print(f"Total samples weighted:             {total_positive + total_negative:7}")
print(f"4 % of Total samples weighted:      {fourpercent:7}")
print(f"True positives at this threshold:   {int(tpr[four_percent_index] * total_positive):7}")
print(f"False positives at this threshold:  {int(fpr[four_percent_index] * total_negative):7}")
print(f"Default rate captured at 4%:        {d:7.5f} (= {int(tpr[four_percent_index] * total_positive)} / {total_positive})")
print()
print(f"Competition score:                  {amex:7.5f} (= ({g:7.5f} + {d:7.5f}) / 2)")


# The competition metric has two components: the normalized Gini coefficient and the default rate captured at 4 %:
# - The *normalized Gini coefficient* is simply a scaled AUC: AUC is the light red area under the curve and can be between 0 and 1. The normalized Gini coefficient is equal to 2\*AUC-1 and is always between -1 and 1. The larger the light red area, the better is the score.
# - The *default rate captured at 4 %* is the true positive rate (recall) for a threshold set at 4 % of the total (weighted) sample count. It corresponds to the y coordinate of the intersection between the green line and the red roc curve (marked with a green dot) and is always between 0 and 1. The higher the intersection point, the better is the score.
# 
# The competition metric is the average of these two components. In other words: They want us to simultaneously optimize for a large red area under the curve and a high intersection point with the green line.
# 

# # Submission
# 
# We submit the mean of the ten predictions.

# In[8]:


# Ensemble the predictions of all folds
sub = pd.DataFrame({'customer_ID': test.index,
                    'prediction': np.mean(y_pred_list, axis=0)})
sub.to_csv('submission.csv', index=False)
sub


# As a plausibility test, we plot a histogram of the predictions. The histogram should resemble the OOF histogram (see above), and the majority of the predictions should be near 0 (because the classes are imbalanced).

# In[9]:


plt.figure(figsize=(16, 5))
plt.hist(sub.prediction, bins=np.linspace(0, 1, 21), density=True)
plt.title("Plausibility check", fontsize=20)
plt.xlabel('Prediction')
plt.ylabel('Density')
plt.show()


# In[ ]:




