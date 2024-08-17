#!/usr/bin/env python
# coding: utf-8

# # Keras quickstart model for the January TPS
# 
# This notebook shows how to use a Keras network in the January 2022 TPS competition.
# 
# On the first of January, I implemented a Keras network for the TPS competition. The network fluctuated between overfitting and divergence. This let me realize that I had to understand the data before implementing the network. I did an [EDA](https://www.kaggle.com/ambrosm/tpsjan22-01-eda-which-makes-sense), implemented a [linear model](https://www.kaggle.com/ambrosm/tpsjan22-03-linear-model) and a [LightGBM model](https://www.kaggle.com/ambrosm/tpsjan22-06-lightgbm-quickstart). Now I'm returning to Keras.
# 
# The network consists of a single dense layer, i.e. it is a linear model. So what is the **advantage of a single-layer neural network** compared to the scikit-learn regressors? The main advantage is that the network is more flexible for experimentation and ready for future improvements: We can play with various regularization schemes or add a hidden layer.
# 
# Some points to note:
# - Although Keras could handle SMAPE as a custom loss, I'm using an MSE loss and a log-transformed target. See [this post](https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/298473) and in particular [this post](https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/300611#1649132) for an explanation why MSE with a log-transformed target is the best choice.
# - The network basically uses the same features as my linear model. This contrasts with the LightGBM model, which works with completely other features.
# - Initializing the bias to a suitable non-zero value reduces training time massively.
# - Cross-validation uses a 4-fold GroupKFold with the years as groups.
# - For good test predictions, we need to retrain the network on the full training data (all four years). In this retraining, we cannot use early stopping because there is no data left for validation. I train the network for a fixed number of epochs and use cosine learning rate decay.
# 
# Experiment with the notebook - I wish you good luck!
# 
# Bug reports: Please report all bugs in the comments section of the notebook.
# 
# Release notes:
# - V2: Retrain on full data
# - V3: Feature engineering
# - V4: Quarterly GDP -> cv is much too high -> don't use this dataset!
# - V5: Other scaling
# - V6: Residuals analysis
# - V7: Residuals grouped by country and product
# 

# In[1]:


import pandas as pd
import numpy as np
import pickle
import itertools
import gc
import math
import matplotlib.pyplot as plt
import dateutil.easter as easter
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter
from datetime import datetime, date
import scipy.stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from tensorflow.keras.layers import Dense, Input, InputLayer, Add


# In[2]:


# Plot training history
def plot_history(history, *, n_epochs=None, plot_lr=False, plot_acc=True, title=None, bottom=None, top=None):
    """Plot (the last n_epochs epochs of) the training history"""
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


# Read the data
original_train_df = pd.read_csv('../input/tabular-playground-series-jan-2022/train.csv')
original_test_df = pd.read_csv('../input/tabular-playground-series-jan-2022/test.csv')

# The dates are read as strings and must be converted
for df in [original_train_df, original_test_df]:
    df['date'] = pd.to_datetime(df.date)
original_train_df.head(2)

gdp_df = pd.read_csv('../input/gdp-20152019-finland-norway-and-sweden/GDP_data_2015_to_2019_Finland_Norway_Sweden.csv')
gdp_df.set_index('year', inplace=True)

# qgdp_df = pd.read_csv('../input/tsp-jan2022-gdp-per-quarter/GDP_Quarterly.csv')
# qgdp_df['GDP'] = qgdp_df['GDP'].apply(lambda s: int(s.replace(',', '')))
# qgdp_df.set_index('Base_Key', inplace=True)
# qgdp_df


# In[4]:


def smape_loss(y_true, y_pred):
    """SMAPE Loss"""
    return tf.abs(y_true - y_pred) / (y_true + tf.abs(y_pred)) * 200


# # Feature engineering

# In[5]:


# Feature engineering
def engineer(df):
    """Return a new dataframe with the engineered features"""
    
    def get_gdp(row):
        """GDP from yearly GDP dataset"""
        country = 'GDP_' + row.country
        return gdp_df.loc[row.date.year, country]
        
#     def get_gdp(row):
#         """GDP from quarterly GDP dataset"""
#         key = f"{row.country}_{row.date.year}_Q{(row.date.month+2)//3}"
#         return qgdp_df.loc[key, 'GDP']
        
    new_df = pd.DataFrame({'gdp': np.log(df.apply(get_gdp, axis=1)),
                           'wd4': df.date.dt.weekday == 4, # Friday
                           'wd56': df.date.dt.weekday >= 5, # Saturday and Sunday
                          })

    # One-hot encoding (no need to encode the last categories)
    for country in ['Finland', 'Norway']:
        new_df[country] = df.country == country
    new_df['KaggleRama'] = df.store == 'KaggleRama'
    for product in ['Kaggle Mug', 'Kaggle Hat']:
        new_df[product] = df['product'] == product
        
    # Seasonal variations (Fourier series)
    # The three products have different seasonal patterns
    dayofyear = df.date.dt.dayofyear
    for k in range(1, 3):
        new_df[f'sin{k}'] = np.sin(dayofyear / 365 * 2 * math.pi * k)
        new_df[f'cos{k}'] = np.cos(dayofyear / 365 * 2 * math.pi * k)
        new_df[f'mug_sin{k}'] = new_df[f'sin{k}'] * new_df['Kaggle Mug']
        new_df[f'mug_cos{k}'] = new_df[f'cos{k}'] * new_df['Kaggle Mug']
        new_df[f'hat_sin{k}'] = new_df[f'sin{k}'] * new_df['Kaggle Hat']
        new_df[f'hat_cos{k}'] = new_df[f'cos{k}'] * new_df['Kaggle Hat']

    # End of year
    new_df = pd.concat([new_df,
                        pd.DataFrame({f"dec{d}":
                                      (df.date.dt.month == 12) & (df.date.dt.day == d)
                                      for d in range(24, 32)}),
                        pd.DataFrame({f"n-dec{d}":
                                      (df.date.dt.month == 12) & (df.date.dt.day == d) &
                                      (df.country == 'Norway')
                                      for d in range(24, 32)}),
                        pd.DataFrame({f"f-jan{d}":
                                      (df.date.dt.month == 1) & (df.date.dt.day == d) & 
                                      (df.country == 'Finland')
                                      for d in range(1, 14)}),
                        pd.DataFrame({f"jan{d}":
                                      (df.date.dt.month == 1) & (df.date.dt.day == d) &
                                      (df.country == 'Norway')
                                      for d in range(1, 10)}),
                        pd.DataFrame({f"s-jan{d}":
                                      (df.date.dt.month == 1) & (df.date.dt.day == d) & 
                                      (df.country == 'Sweden')
                                      for d in range(1, 15)})],
                       axis=1)
    
    # May
    new_df = pd.concat([new_df,
                        pd.DataFrame({f"may{d}":
                                      (df.date.dt.month == 5) & (df.date.dt.day == d) 
                                      for d in list(range(1, 10))}),
                        pd.DataFrame({f"may{d}":
                                      (df.date.dt.month == 5) & (df.date.dt.day == d) & 
                                      (df.country == 'Norway')
                                      for d in list(range(18, 28))})],
                       axis=1)
    
    # June and July
    new_df = pd.concat([new_df,
                        pd.DataFrame({f"june{d}":
                                      (df.date.dt.month == 6) & (df.date.dt.day == d) & 
                                      (df.country == 'Sweden')
                                      for d in list(range(8, 14))}),
                       ],
                       axis=1)
    
    # Last Wednesday of June
    wed_june_date = df.date.dt.year.map({2015: pd.Timestamp(('2015-06-24')),
                                         2016: pd.Timestamp(('2016-06-29')),
                                         2017: pd.Timestamp(('2017-06-28')),
                                         2018: pd.Timestamp(('2018-06-27')),
                                         2019: pd.Timestamp(('2019-06-26'))})
    new_df = pd.concat([new_df,
                        pd.DataFrame({f"wed_june{d}": 
                                      (df.date - wed_june_date == np.timedelta64(d, "D")) & 
                                      (df.country != 'Norway')
                                      for d in list(range(-4, 6))})],
                       axis=1)
    
    # First Sunday of November
    sun_nov_date = df.date.dt.year.map({2015: pd.Timestamp(('2015-11-1')),
                                         2016: pd.Timestamp(('2016-11-6')),
                                         2017: pd.Timestamp(('2017-11-5')),
                                         2018: pd.Timestamp(('2018-11-4')),
                                         2019: pd.Timestamp(('2019-11-3'))})
    new_df = pd.concat([new_df,
                        pd.DataFrame({f"sun_nov{d}": 
                                      (df.date - sun_nov_date == np.timedelta64(d, "D")) &
                                      (df.country != 'Norway')
                                      for d in list(range(0, 9))})],
                       axis=1)
    
    # First half of December (Independence Day of Finland, 6th of December)
    new_df = pd.concat([new_df,
                        pd.DataFrame({f"dec{d}":
                                      (df.date.dt.month == 12) & (df.date.dt.day == d) &
                                      (df.country == 'Finland')
                                      for d in list(range(6, 14))})],
                       axis=1)

    # Easter
    easter_date = df.date.apply(lambda date: pd.Timestamp(easter.easter(date.year)))
    new_df = pd.concat([new_df,
                        pd.DataFrame({f"easter{d}": 
                                      (df.date - easter_date == np.timedelta64(d, "D"))
                                      for d in list(range(-2, 11)) + list(range(40, 48)) +
                                      list(range(50, 59))})],
                       axis=1)
    
    return new_df.astype(np.float32)

train_df = engineer(original_train_df)
train_df['date'] = original_train_df.date
train_df['num_sold'] = original_train_df.num_sold.astype(np.float32)
test_df = engineer(original_test_df)

features = list(test_df.columns)
print(list(features))


# # Training and validation
# 
# We validate using a 4-fold GroupKFold with the years as groups. We show
# - The execution time and the SMAPE
# - The training and validation loss curves with the learning rate
# - A scatterplot y_true vs. y_pred (ideally all points should lie near the diagonal)
# 

# In[6]:


#%%time
EPOCHS = 300
EPOCHS_COSINEDECAY = 120
VERBOSE = 0 # set to 0 for less output, or to 2 for more output
RUNS = 5 # set to 1 for quick experiments
DIAGRAMS = True
USE_PLATEAU = True
INFERENCE = False

# We split the features into subsets so that we can apply different
# regularization schemes for the subsets
wd_features = [f for f in features if f.startswith('wd')]
other_features = [f for f in features if f not in wd_features]

# def tpsjan_model():
#     """Linear model with flexible regularization
    
#     The model is to be used with a log-transformed target.
#     """
#     wd = Input(shape=(len(wd_features), ))
#     other = Input(shape=(len(other_features), ))
#     wd_contribution = Dense(1, kernel_regularizer=tf.keras.regularizers.l2(1e-7),
#                             use_bias=False)(wd)
#     other_contribution = Dense(1, kernel_regularizer=tf.keras.regularizers.l2(1e-7),
#                                use_bias=True,
#                                bias_initializer=tf.keras.initializers.Constant(value=5.7))(other)
#     output = Add()([wd_contribution, other_contribution])
#     model = Model([wd, other], output)
#     return model


def tpsjan_model_2():
    """Linear model
    
    The model is to be used with a log-transformed target.
    """
    other = Input(shape=(len(features), ))
    output = Dense(1, #kernel_regularizer=tf.keras.regularizers.l2(1e-6),
                   use_bias=True,
                   bias_initializer=tf.keras.initializers.Constant(value=5.74))(other)
    model = Model(other, output)
    return model


def fit_model(X_tr, X_va=None):
    """Scale the data, fit a model, plot the training history and validate the model"""
    start_time = datetime.now()

    # Preprocess the data (select columns and scale)
    preproc = make_pipeline(MinMaxScaler(), StandardScaler(with_std=False))
    X_tr_f = pd.DataFrame(preproc.fit_transform(X_tr[features]), columns=features, index=X_tr.index)
    y_tr = X_tr.num_sold.values.reshape(-1, 1)

    if X_va is not None:
        # Preprocess the validation data
        X_va_f = pd.DataFrame(preproc.transform(X_va[features]), columns=features, index=X_va.index)
        y_va = X_va.num_sold.values.reshape(-1, 1)
        validation_data = ([X_va_f[features]], np.log(y_va))
    else:
        validation_data = None

    # Define the learning rate schedule and EarlyStopping
    if USE_PLATEAU and X_va is not None:
        epochs = EPOCHS
        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.7, 
                               patience=4, verbose=VERBOSE) # 4
        es = EarlyStopping(monitor="val_loss",
                           patience=25, 
                           verbose=1,
                           mode="min", 
                           restore_best_weights=True)
        callbacks = [lr, es, tf.keras.callbacks.TerminateOnNaN()]

    else:
        epochs = EPOCHS_COSINEDECAY
        lr_start=0.02
        lr_end=0.00001
        def cosine_decay(epoch):
            if epochs > 1:
                w = (1 + math.cos(epoch / (epochs-1) * math.pi)) / 2
            else:
                w = 1
            return w * lr_start + (1 - w) * lr_end

        lr = LearningRateScheduler(cosine_decay, verbose=0)
        callbacks = [lr, tf.keras.callbacks.TerminateOnNaN()]
        
    # Construct and compile the model
    model = tpsjan_model_2()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    #model.compile(optimizer=tf.keras.optimizers.SGD(), loss='mse')

    # Train the model
    history = model.fit([X_tr_f[features]], np.log(y_tr), 
                        validation_data=validation_data, 
                        epochs=epochs,
                        verbose=VERBOSE,
                        batch_size=512,
                        shuffle=True,
                        callbacks=callbacks)

    history_list.append(history.history)
    callbacks, es, lr, history = None, None, None, None
    #print(f"Loss:            {history_list[-1]['loss'][-1]:.6f}")
    #print(f"Bias:  {model.get_layer(index=-1).get_weights()[1]}")
    
    if X_va is not None:
        # Inference for validation
        y_va_pred = np.exp(model.predict([X_va_f[features]]))
        oof_list[run][val_idx] = y_va_pred
        
        # Evaluation: Execution time and SMAPE
        smape = np.mean(smape_loss(y_va, y_va_pred))
        print(f"Fold {run}.{fold} | {str(datetime.now() - start_time)[-12:-7]}"
              f" | SMAPE: {smape:.5f} validated on {X_va.iloc[0].date.year}")
        score_list.append(smape)
        
        if DIAGRAMS and fold == 0 and run == 0:
            # Plot training history
            plot_history(history_list[-1], title=f"Validation SMAPE = {smape:.5f}",
                         plot_lr=True, n_epochs=110)

            # Plot y_true vs. y_pred
            plt.figure(figsize=(10, 10))
            plt.scatter(y_va, y_va_pred, s=1, color='r')
            #plt.scatter(np.log(y_va), np.log(y_va_pred), s=1, color='g')
            plt.plot([plt.xlim()[0], plt.xlim()[1]], [plt.xlim()[0], plt.xlim()[1]], '--', color='k')
            plt.gca().set_aspect('equal')
            plt.xlabel('y_true')
            plt.ylabel('y_pred')
            plt.title('OOF Predictions')
            plt.show()

    return preproc, model


# Make the results reproducible
np.random.seed(2022)

total_start_time = datetime.now()
history_list, score_list, test_pred_list = [], [], []
oof_list = [np.full((len(train_df), 1), -1.0, dtype='float32') for run in range(RUNS)]
for run in range(RUNS):
    preproc, model = None, None
    kf = GroupKFold(n_splits=4)
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df, groups=train_df.date.dt.year)):
        X_tr = train_df.iloc[train_idx]
        X_va = train_df.iloc[val_idx]
        print(f"Fold {run}.{fold}")
        preproc, model = fit_model(X_tr, X_va)
        if INFERENCE:
            test_df_f = pd.DataFrame(preproc.transform(test_df[features]), columns=features, index=test_df.index)
            test_pred_list.append(np.exp(model.predict([test_df_f[wd_features], test_df_f[other_features]])))


print(f"Average SMAPE: {sum(score_list) / len(score_list):.5f}") # Average over all runs and folds
with open('oof.pickle', 'wb') as handle: pickle.dump(oof_list, handle) # for further analysis
    
if RUNS > 1:
    y_va = train_df.num_sold
    print(f"Ensemble SMAPE: {np.mean(smape_loss(y_va, sum(oof_list).ravel() / len(oof_list))):.5f}")
print(f"Total time: {str(datetime.now() - total_start_time)[:-7]}")


# # Feature importance

# In[7]:


w = pd.Series(model.get_layer(index=-1).get_weights()[0].ravel(), index=features)
ws = w * preproc.named_steps['minmaxscaler'].scale_

def plot_feature_weights_numbered(prefix):
    prefix_features = [f for f in features if f.startswith(prefix)]
    plt.figure(figsize=(12, 2))
    plt.bar([int(f[len(prefix):]) for f in prefix_features], ws[prefix_features])
    plt.title(f'Feature weights for {prefix}')
    plt.ylabel('weight')
    plt.xlabel('day')
    plt.show()
    
plot_feature_weights_numbered('easter')


# # Demonstration

# In[8]:


# Plot all num_sold_true and num_sold_pred (five years) for one country-store-product combination
def plot_five_years_combination(engineer, country='Norway', store='KaggleMart', product='Kaggle Hat'):
    demo_df = pd.DataFrame({'row_id': 0,
                            'date': pd.date_range('2015-01-01', '2019-12-31', freq='D'),
                            'country': country,
                            'store': store,
                            'product': product})
    demo_df.set_index('date', inplace=True, drop=False)
    demo_df = engineer(demo_df)
    demo_df_f = pd.DataFrame(preproc.transform(demo_df[features]), columns=features, index=demo_df.index)
    demo_df['num_sold'] = np.exp(model.predict([demo_df_f[features]]))
    plt.figure(figsize=(20, 6))
    plt.plot(np.arange(len(demo_df)), demo_df.num_sold, label='prediction')
    train_subset = train_df[(original_train_df.country == country) & (original_train_df.store == store) & (original_train_df['product'] == product)]
    plt.scatter(np.arange(len(train_subset)), train_subset.num_sold, label='true', alpha=0.5, color='red', s=3)
    plt.legend()
    plt.title('Predictions and true num_sold for five years')
    plt.show()

plot_five_years_combination(engineer)



# # Retraining and submission

# In[9]:


# Retrain the network on the full training data several times
RETRAIN_RUNS = 33
if RETRAIN_RUNS > 0:
    total_start_time = datetime.now()
    test_pred_list = []
    for run in range(RETRAIN_RUNS):
        preproc, model = None, None
        print(f"Retraining {run}")
        preproc, model = fit_model(train_df)
        print(f"Training loss:            {history_list[-1]['loss'][-1]:.6f}")
        test_df_f = pd.DataFrame(preproc.transform(test_df[features]), columns=features, index=test_df.index)
        test_pred_list.append(np.exp(model.predict([test_df_f[features]])))
    print(f"Total time: {str(datetime.now() - total_start_time)[:-7]}")


# In[10]:


# Ensemble the test predictions
sub = None
if len(test_pred_list) > 0:
    # Create the submission file
    print(f"Ensembling {len(test_pred_list)} predictions...")
    sub = original_test_df[['row_id']].copy()
    sub['num_sold'] = sum(test_pred_list) / len(test_pred_list)
    sub.to_csv('submission_keras.csv', index=False)
    
    # Plot the distribution of the test predictions
    plt.figure(figsize=(16,3))
    plt.hist(train_df['num_sold'], bins=np.linspace(0, 3000, 201), density=True, label='Training')
    plt.hist(sub['num_sold'], bins=np.linspace(0, 3000, 201), density=True, rwidth=0.5, label='Test predictions')
    plt.xlabel('num_sold')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

sub


# In[11]:


# Create a rounded submission file
sub_rounded = None
if sub is not None:
    sub_rounded = sub.copy()
    sub_rounded['num_sold'] = sub_rounded['num_sold'].round()
    sub_rounded.to_csv('submission_keras_rounded.csv', index=False)
sub_rounded


# # Analyzing the residuals
# 
# A residuals analysis can give an indication how the model might be improved. We start by computing the residuals and plotting all 26298 residuals. The plot looks quite homogeneous, except maybe between row_id 18000 and 20000, where the residuals are slightly above average.

# In[12]:


# Compute the residuals using the SMAPE formula without abs() and mean()
train_preds = np.exp(model.predict([preproc.transform(train_df[features])])).ravel()
residuals = (train_df.num_sold - train_preds) / (train_df.num_sold + train_preds) * 200

plt.figure(figsize=(20,6))
plt.scatter(residuals.index, residuals, s=1, color='b')
plt.hlines([0], 0, residuals.index.max(), color='k')
plt.title('Residuals for all 26298 training samples')
plt.ylabel('Residual (percent)')
plt.xlabel('row_id')
plt.show()


# As a second step, we plot the histogram of the residuals. The histogram looks like a normal distribution which has a standard deviation of 5.2 and is (almost) centered at 0. Almost all 26298 residuals are contained within ±4 standard deviations, i.e. there are no outliers.

# In[13]:


mu, std = scipy.stats.norm.fit(residuals)

plt.figure(figsize=(20,4))
plt.hist(residuals, bins=100, color='b', density=True)
x = np.linspace(plt.xlim()[0], plt.xlim()[1], 200)
plt.plot(x, scipy.stats.norm.pdf(x, mu, std), 'r', linewidth=2)
plt.title(f'Histogram of residuals; mean = {residuals.mean():.4f}, '
          f'$\sigma = {residuals.std():.1f}$, SMAPE = {residuals.abs().mean():.5f}')
plt.xlabel('Residual (percent)')
plt.ylabel('Density')
plt.show()


# ## Seasonality
# 
# In the third step of our residual analysis, we want to test whether the seasonality of the data is well modeled. For this purpose, we calculate the average residual
# - for every day of the month (31 values)
# - for every week of the year (53 values)
# - for every month of the year (12 values)
# 
# The standard deviation of these averages should be much smaller than the standard deviation of the individual residuals (which was 5.2). The diagrams show that all averages are between -0.6 and +0.6.
# 
# We now could do a [one-sample t-test](https://en.wikipedia.org/wiki/Student%27s_t-test#One-sample_t-test) to decide whether any average significantly deviates from zero, but the sample size is large enough to approximate the t-test by a [z-test](https://en.wikipedia.org/wiki/Z-test). 
# 
# In the diagrams, the bars are colored red if the average significantly deviates from zero, whereby the significance level is chosen so that only one or two bars per diagram should be red if the residuals are independent. Indeed most of the bars are blue, which shows that the data's seasonality is captured well by the model.

# In[14]:


def plot_unexplained(residuals, groups, labels=None, label_z_score=False, title=None):
    residuals_grouped = residuals.groupby(groups)
    means = residuals_grouped.mean()
    counts = residuals_grouped.count()
    z_score = np.sqrt(counts) * means / residuals.std()
    z_threshold = scipy.stats.norm.ppf(1 - 0.25 / len(means))
    m_threshold = z_threshold * residuals.std() / np.sqrt(counts.mean())
    outliers = np.abs(z_score) > z_threshold
    plt.figure(figsize=(17, 4))
    #plt.hlines([-z_threshold, +z_threshold] if label_z_score else [-m_threshold, +m_threshold], 0, len(means)-1, color='k')
    plt.bar(range(len(means)), z_score if label_z_score else means,
            color=outliers.apply(lambda b: 'r' if b else 'b'), width=0.6)
    if labels is not None: plt.xticks(ticks=range(len(means)), labels=labels)
    plt.ylabel('z score' if label_z_score else 'percent')
    plt.title(title)
    plt.show()    

plot_unexplained(residuals, [train_df.date.dt.day],
                 labels=np.arange(1, 32),
                 title='Residuals for the 31 days of the month')
plot_unexplained(residuals, [(train_df.date.dt.dayofyear) // 7],
                 labels=None,
                 title='Residuals for the 53 weeks of a year')
plot_unexplained(residuals, [train_df.date.dt.month],
                 labels="JFMAMJJASOND",
                 title='Residuals for the 12 months')


# ## Other trends
# 
# Having seen that the model captures the seasonality well, we'll do a similar test to see how the model deals with trends which do not depend on season. Again we average residuals and do a z-test. This time we see lots of red bars, most conspicuously in the last quarter of 2017, where sales are 2 % higher than expected. 
# 
# The result of this analysis means that the sales figures are affected by some time-dependent influence which is unknown to our model. This influence might be, for instance:
# - Some macro-economic indicator with monthly or better granularity, perhaps customer confidence index, unemployment rate or an exchange rate
# - Marketing campaigns of Kaggle or its competitors - maybe Google Trends knows more
# - The weather, as [investigated by](https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/301486) @adamwurdits

# In[15]:


plot_unexplained(residuals, [(train_df.date - train_df.date.min()).dt.days // 7],
                 labels=None,
                 title='Mean residuals of all 213 weeks of the training data')
plot_unexplained(residuals, [train_df.date.dt.year, train_df.date.dt.month],
                 labels="JFMAMJJASOND" * 4,
                 title='Mean residuals of all 48 months')
plot_unexplained(residuals, [train_df.date.dt.year, train_df.date.dt.quarter],
                 labels=[f"{q//4}Q{q%4+1}" for q in range(60, 76)],
                 title='Mean residuals of all 16 Quarters')


# We can group the same residuals by country or product to show that we are looking for an influence which affects all countries and products equally.

# In[16]:


def plot_unexplained(residuals, groups, labels=None, label_z_score=False, title=None, label=None):
    residuals_grouped = residuals.groupby(groups)
    means = residuals_grouped.mean()
    plt.plot(range(len(means)), z_score if label_z_score else means,
            label=label)
    if labels is not None: plt.xticks(ticks=range(len(means)), labels=labels)


plt.figure(figsize=(17, 6))
plt.subplot(2, 1, 1)
for i, c in enumerate(original_train_df.country.unique()):
    selection = original_train_df.country == c
    plot_unexplained(residuals[selection], [train_df.date.dt.year[selection], train_df.date.dt.month[selection]],
                     labels="JFMAMJJASOND" * 4,
                     title='Mean residuals of all 48 months',
                     label=c)
plt.legend()

plt.subplot(2, 1, 2)
for i, c in enumerate(original_train_df['product'].unique()):
    selection = original_train_df['product'] == c
    plot_unexplained(residuals[selection], [train_df.date.dt.year[selection], train_df.date.dt.month[selection]],
                     labels="JFMAMJJASOND" * 4,
                     title='Mean residuals of all 48 months',
                     label=c)
plt.legend()
plt.show()


# In[ ]:




