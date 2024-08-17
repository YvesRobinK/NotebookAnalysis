#!/usr/bin/env python
# coding: utf-8

# ![logo](https://optuna.org/assets/img/bg.jpg)

# There are many hyperparameter optimization frameworks available, and in this notebook we will give [Optuna](https://optuna.org/) a spin.

# In[1]:


import numpy as np
import pandas as pd
import optuna
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_contour
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.model_selection import train_test_split
from pickle import load
get_ipython().system('cp ../input/ventilator-feature-engineering/VFE.py .')


# # Dataset creation
# Training dataset is loaded from the [feature engineering notebook](https://www.kaggle.com/mistag/ventilator-feature-engineering).
# Feature engineering is based on [Ensemble Folds with MEDIAN](https://www.kaggle.com/cdeotte/ensemble-folds-with-median-0-153) by [Chris Deotte](https://www.kaggle.com/cdeotte). The optimization is run on a smaller subset of the dataset.

# In[2]:


from VFE import add_features

train = np.load('../input/ventilator-feature-engineering/x_train.npy')
targets = np.load('../input/ventilator-feature-engineering/y_train.npy')

BATCH_SIZE = 1024

# test set
test_ori = pd.read_csv('../input/ventilator-pressure-prediction/test.csv')
test = add_features(test_ori)
test.drop(['id', 'breath_id'], axis=1, inplace=True)

RS = load(open('../input/ventilator-feature-engineering/RS.pkl', 'rb'))
test = RS.transform(test)
test = test.reshape(-1, 80, test.shape[-1])


# Finally we split the data into train and test sets. We keep a large holdout set for model evaluation.

# In[3]:


X_train, X_test, y_train, y_test = train_test_split(train, targets, test_size=0.59284, random_state=21)
X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.79395, random_state=21)
X_train.shape, X_test.shape, X_valid.shape


# # Model building
# The model below is from  [Ensemble Folds with MEDIAN](https://www.kaggle.com/cdeotte/ensemble-folds-with-median-0-153) by [Chris Deotte](https://www.kaggle.com/cdeotte).. Hopefully Optuna will be able to figure out the optimal parameters in the model. All parameters that we want to explore are created with a trial.suggest_() function. 

# In[4]:


# model creation
def create_lstm_model(trial):

    x0 = tf.keras.layers.Input(shape=(train.shape[-2], train.shape[-1]))  

    lstm_layers = 4
    lstm_units = np.zeros(lstm_layers, dtype=np.int)
    lstm_units[0] = trial.suggest_int("lstm_units_L1", 768, 1536)
    lstm = Bidirectional(keras.layers.LSTM(lstm_units[0], return_sequences=True))(x0)
    for i in range(lstm_layers-1):
        lstm_units[i+1] = trial.suggest_int("lstm_units_L{}".format(i+2), lstm_units[i]//2, lstm_units[i])
        lstm = Bidirectional(keras.layers.LSTM(lstm_units[i+1], return_sequences=True))(lstm)    
    dropout_rate = trial.suggest_float("lstm_dropout", 0.0, 0.3)
    lstm = Dropout(dropout_rate)(lstm)
    dense_units = lstm_units[-1]
    # try different activations
    activation = trial.suggest_categorical("activation", ["relu", "selu", "elu", "swish"])
    lstm = Dense(dense_units, activation=activation)(lstm)
    lstm = Dense(1)(lstm)

    model = keras.Model(inputs=x0, outputs=lstm)
    metrics = ["mae"]
    model.compile(optimizer="adam", loss="mae", metrics=metrics)
    
    return model


# ## Objective function
# Here we define the Optuna objective function. The number of epochs per trial is a balance between execution time per trial and confidence in the result of each trial.

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


EPOCHS = 30 # number of epocs per trial

def objective(trial):
    
    # Clear clutter from previous session graphs.
    keras.backend.clear_session()
    
    with strategy.scope():
        # Generate our trial model.
        model = create_lstm_model(trial)

        # learning rate scheduler
        scheduler = ExponentialDecay(1e-3, 400*((len(train)*0.8)/BATCH_SIZE), 1e-5)
        lr = LearningRateScheduler(scheduler, verbose=0)
    
        # Fit the model on the training data.
        # The TFKerasPruningCallback checks for pruning condition every epoch.
        model.fit(
            X_train,
            y_train,
            batch_size=BATCH_SIZE,
            callbacks=[TFKerasPruningCallback(trial, "val_loss")],
            epochs=EPOCHS,
            validation_data=(X_test, y_test),
            verbose=1,
        )

        # Evaluate the model accuracy on the validation set.
        score = model.evaluate(X_valid, y_valid, verbose=0)
        return score[1]


# # Run optimization
# There are different samplers and pruners to choose from, here we go for TPESampler and HyperbandPruner.

# In[7]:


study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner())
study.optimize(objective, n_trials=100)
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])


# # Result
# Now we can create a few interesting plots with the Optuna builtin visualization functions, starting with optimization history:

# In[8]:


plot_optimization_history(study)


# Visualize the loss curves of the trials:

# In[9]:


plot_intermediate_values(study)


# Parameter contour plots - useful or confusing?

# In[10]:


plot_contour(study)


# The parameter importance plot is really interesting:

# In[11]:


plot_param_importances(study)


# Finally list the optimized model parameters:

# In[12]:


print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# # Summary
# Using Optuna we found a set of optimal model parameters. Next step is to [test the optimal model](https://www.kaggle.com/mistag/optuna-optimized-base-keras-model).

# In[ ]:




