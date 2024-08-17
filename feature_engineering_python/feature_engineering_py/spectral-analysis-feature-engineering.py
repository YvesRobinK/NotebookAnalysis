#!/usr/bin/env python
# coding: utf-8

# # Spectral Analysis & Feature Engineering
# 
# From niwashi (@marutama) numerous EDA notebooks we can see that a lot of u_in and pressure exhibit oscillating patterns. This is something we also see in my [error analysis / clusterings notebooks]( https://www.kaggle.com/lucasmorin/u-in-mae-exploration-with-umap-hdbscan ): clusters of MAE often correspond to highly oscillating patterns. I wanted to explore these oscillating aspects so I tried a spectral approach, mainly relying on fourrier transformations.
# 
# This approach mainly result in:
# 
# - EDA tools for further spectral exploration
# 
# - Some tricks for better spectral analysis (windowsing)
# 
# - Some interesting ts features for LSTM model
# 
# - An idea to use whole spectrum in 2D architectures
# 
# - An attempt at machine re-identification
# 
# You can find everything in this notebook.

# In[1]:


import numpy as np
import pandas as pd

from IPython.display import display

import pickle
import matplotlib.pyplot as plt


# In[2]:


DEBUG = False

dict_types = {
'id': np.int32,
'breath_id': np.int32,
'R': np.int8,
'C': np.int8,
'time_step': np.float32,
'u_in': np.float32,
'u_out': np.int8, #np.bool ?
'pressure': np.float32,
} 

train = pd.read_csv('../input/ventilator-pressure-prediction/train.csv', dtype=dict_types)
test = pd.read_csv('../input/ventilator-pressure-prediction/test.csv', dtype=dict_types)

submission = pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')

all_pressure = np.sort(train.pressure.unique())
PRESSURE_MIN = all_pressure[0]
PRESSURE_MAX = all_pressure[-1]
PRESSURE_STEP = (all_pressure[1] - all_pressure[0])

if DEBUG:
    train = train[:80*1000]
    test = test[:80*1000]


# # Fourrier transform
# 
# Get a "weird" input / output. 

# In[3]:


idb = train.breath_id.unique()[31]

t1 = train[train.breath_id==idb].u_in
p1 = train[train.breath_id==idb].pressure

plt.plot(t1);
plt.plot(p1);
plt.legend(['u_in', 'pressure']);
plt.show();


# # Base Fourrier Transform

# In[4]:


import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

Pper_spec_t =  fft(np.append(t1.values,t1.values[0]))
Pper_spec_p =  fft(np.append(p1.values,p1.values[0]))

plt.semilogy(np.log(np.abs(Pper_spec_t))[1:40]);
plt.semilogy(np.log(np.abs(Pper_spec_p))[1:40]);
plt.legend(['FFT_u_in', 'FFT_pressure']);


# We can see local maxima. 

# # Windowsing

# We have relatively short time series, thus we get very noisy fft. One option is to use windows. 

# In[5]:


# mostly from the scipy documentation
from scipy.signal import blackman

# Number of sample points
N = 80
# sample spacing
T = 1

x = np.linspace(0.0, N*T+1, N, endpoint=False)
y = train[train.breath_id==idb].u_in

y = np.append(y.values,y.values[0])


yf = fft(y)
w = blackman(N+1)
ywf = fft(y*w)
xf = fftfreq(N, T)[:N//2]

plt.semilogy(xf[1:N//2], 2.0/N * np.abs(yf[1:N//2]), '-b')
plt.semilogy(xf[1:N//2], 2.0/N * np.abs(ywf[1:N//2]), '-r')
plt.legend(['FFT_u_in', 'FFT_u_in w. window'])
plt.grid()
plt.show()

x = np.linspace(0.0, N*T+1, N, endpoint=False)
y = train[train.breath_id==idb].pressure
y = np.append(y.values,y.values[0])

yfp = fft(y)
w = blackman(N+1)
ywfp = fft(y*w)
xf = fftfreq(N, T)[:N//2]

plt.semilogy(xf[1:N//2], 2.0/N * np.abs(yfp[1:N//2]), '-b')
plt.semilogy(xf[1:N//2], 2.0/N * np.abs(ywfp[1:N//2]), '-r')
plt.legend(['FFT_pressure', 'FFT_pressure w. window'])
plt.grid()
plt.show()


# Seems better.

# # Envellope, instantaneous phase / frequency

# Oscillations aren't uniform. We want to better identify change in oscillations. The idea here is to get an envellope and instantaneous frequency.

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp

duration = 80
fs = 1
samples = int(fs*duration)
t = np.arange(samples) / fs
#We create a chirp of which the frequency increases from 20 Hz to 100 Hz and apply an amplitude modulation.

signal = train[train.breath_id==idb].u_in
#The amplitude envelope is given by magnitude of the analytic signal. The instantaneous frequency can be obtained by differentiating the instantaneous phase in respect to time. The instantaneous phase corresponds to the phase angle of the analytic signal.

analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = (np.diff(instantaneous_phase) /
                           (2.0*np.pi) * fs)

fig, (ax0, ax1) = plt.subplots(nrows=2)
ax0.plot(t, signal, label='signal')
ax0.plot(t, amplitude_envelope, label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()
ax1.plot(t[1:], instantaneous_frequency)
ax1.set_xlabel("time in seconds")
ax1.set_ylim(0.0, 1)
fig.tight_layout()


# # TS Feature Engineering

# Adding envellope and instantaneous frequency in a LSTM make sense to me. It's seems a bit more difficult to add FFT transformation as tim series, but why not try it and see if it works ?

# In[7]:


get_ipython().run_cell_magic('time', '', "\nffta = lambda x: np.abs(fft(np.append(x.values,x.values[0]))[:80])\nffta.__name__ = 'ffta'\n\nfftw = lambda x: np.abs(fft(np.append(x.values,x.values[0])*w)[:80])\nfftw.__name__ = 'fftw'\n\ntrain['fft_u_in'] = train.groupby('breath_id')['u_in'].transform(ffta)\ntrain['fft_u_in_w'] = train.groupby('breath_id')['u_in'].transform(fftw)\ntrain['analytical'] = train.groupby('breath_id')['u_in'].transform(hilbert)\ntrain['envelope'] = np.abs(train['analytical'])\ntrain['phase'] = np.angle(train['analytical'])\ntrain['unwrapped_phase'] = train.groupby('breath_id')['phase'].transform(np.unwrap)\ntrain['phase_shift1'] = train.groupby('breath_id')['unwrapped_phase'].shift(1).astype(np.float32)\ntrain['IF'] = train['unwrapped_phase'] - train['phase_shift1'].astype(np.float32)\n")


# # Complete spectrum

# If oscillations aren't uniform over time, this means we are interested in the whole spectrum. However using these might require a completely different model. 

# In[8]:


from scipy import signal
import matplotlib.pyplot as plt

fs = 1
N = 80
 
x = train[train.breath_id==idb].u_in

f, t, Zxx = signal.stft(x, 1, nperseg=16)

amp = np.max(np.log(np.abs(Zxx)))

plt.pcolormesh(t, f, np.log(np.abs(Zxx)), vmin=0, vmax=amp)#, shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [id]')
plt.show()


# Not sure how to use that. Maybe some as a feature ? in a 2D NN architecture ?

# # Error Analysis

# (please upvote the hdbscan wheel data set)

# In[9]:


get_ipython().system('mkdir -p /tmp/pip/cache/')
get_ipython().system('cp ../input/hdbscan0827-whl/hdbscan-0.8.27-cp37-cp37m-linux_x86_64.whl /tmp/pip/cache/')
get_ipython().system('pip install --no-index --find-links /tmp/pip/cache/ hdbscan')


# In[10]:


get_ipython().run_cell_magic('time', '', "\nimport hdbscan\nimport umap\nimport pickle\nimport matplotlib.colors as colors\n\n\nMAE_id = pickle.load(open('../input/u-in-mae-exploration-with-umap-hdbscan/MAE_id.pkl', 'rb'))\n\ntrain['time_id'] = [e  for i in range(len(train.breath_id.unique())) for e in range(80)] \nX = train[['breath_id','fft_u_in_w','time_id']].pivot(index='breath_id',columns='time_id',values='fft_u_in_w')\nMAE = MAE_id[:X.shape[0]]\n\nreducer = umap.UMAP(random_state=42, n_components=2)\nembedding = reducer.fit_transform(X)\nclusterer = hdbscan.HDBSCAN(prediction_data=True, min_cluster_size = 50).fit(embedding)\nu, counts = np.unique(clusterer.labels_, return_counts=True)\n\nprint(u)\nprint(counts)\n\nplt.figure(figsize=(10, 8));\nplt.scatter(embedding[:, 0], embedding[:, 1], s=5, c=clusterer.labels_, edgecolors='none', cmap='jet');\nplt.show();\n\nplt.figure(figsize=(10, 8));\nplt.scatter(embedding[:, 0], embedding[:, 1], s=5, c=MAE, edgecolors='none', cmap='jet', norm=colors.LogNorm(vmin=MAE.quantile(0.05), vmax=MAE.quantile(0.95)));\nplt.colorbar();\nplt.show();\n\ndel X\n")


# # Machine identification ?

# if we assume some sort of constant behavior, the ratio FFT(pressure)/FFT(u_in) should give the behavior of the machine (up to some shift).

# In[11]:


plt.semilogy(xf[1:N//2], np.abs(yfp[1:N//2])/np.abs(yf[1:N//2]), '-b')
plt.semilogy(xf[1:N//2], np.abs(ywfp[1:N//2])/np.abs(ywf[1:N//2]), '-r')
plt.legend(['FFT_pressure / FFT_u_in', 'FFT_pressure / FFT_u_in w. window'])
plt.grid()
plt.show()


# # Machine transformation - whole spectrum

# In[12]:


import matplotlib.pyplot as plt

fs = 1
N = 80

for i in range(1,8):
    
    x = train[train.breath_id==i].u_in
    y = train[train.breath_id==i].pressure
    
    f, t, Zxx = signal.stft(x, fs, nperseg=16)
    f, t, Zxy = signal.stft(y, fs, nperseg=16)
    
    plt.pcolormesh(t, f, np.log(np.abs(Zxy)) - np.log(np.abs(Zxx)))
    plt.title('STFT p / SFT u_in log Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


# # Machine Clustering

# In[13]:


train['fft_u_in_w'] = train.groupby('breath_id')['u_in'].transform(fftw)
train['fft_u_in_w'] = train['fft_u_in_w'].replace(0,1e-6)

train['fft_pressure_w'] = train.groupby('breath_id')['pressure'].transform(fftw)

train['fft_machine_w'] = np.log(train['fft_pressure_w']/train['fft_u_in_w'])


# In[14]:


get_ipython().run_cell_magic('time', '', "\nimport hdbscan\nimport umap\nimport pickle\nimport matplotlib.colors as colors\n\n\nMAE_id = pickle.load(open('../input/u-in-mae-exploration-with-umap-hdbscan/MAE_id.pkl', 'rb'))\n\ntrain['time_id'] = [e  for i in range(len(train.breath_id.unique())) for e in range(80)] \nX = train[['breath_id','fft_machine_w','time_id']].pivot(index='breath_id',columns='time_id',values='fft_machine_w')\nMAE = MAE_id[:X.shape[0]]\n\nreducer = umap.UMAP(random_state=42, n_components=2)\nembedding = reducer.fit_transform(X)\nclusterer = hdbscan.HDBSCAN(prediction_data=True, min_cluster_size = 50).fit(embedding)\nu, counts = np.unique(clusterer.labels_, return_counts=True)\n\nprint(u)\nprint(counts)\n\nplt.figure(figsize=(10, 8));\nplt.scatter(embedding[:, 0], embedding[:, 1], s=5, c=clusterer.labels_, edgecolors='none', cmap='jet');\nplt.show();\n\nplt.figure(figsize=(10, 8));\nplt.scatter(embedding[:, 0], embedding[:, 1], s=5, c=MAE, edgecolors='none', cmap='jet', norm=colors.LogNorm(vmin=MAE.quantile(0.05), vmax=MAE.quantile(0.95)));\nplt.colorbar();\nplt.show();\n\ndel X\n")


# Cool idea, not sure if exploitable as we don't have well separated clusters (and pressure is not available in test). 

# # Feature importance
# 
# Using @cdeotte LTSM Feature importance: https://www.kaggle.com/cdeotte/lstm-feature-importance
# 
# Which Rely on @tenffe: https://www.kaggle.com/tenffe/finetune-of-tensorflow-bidirectional-lstm

# In[15]:


import numpy as np, os
import pandas as pd

import optuna

# https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/274717 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.model_selection import train_test_split, GroupKFold, KFold

from IPython.display import display

DEBUG = False
TRAIN_MODEL = True
INFER_TEST = True
ONE_FOLD_ONLY = True
COMPUTE_LSTM_IMPORTANCE = True

train = pd.read_csv('../input/ventilator-pressure-prediction/train.csv')
pressure_values = np.sort( train.pressure.unique() )
test = pd.read_csv('../input/ventilator-pressure-prediction/test.csv')
submission = pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')

if DEBUG:
    train = train[:80*1000]
    test = test[:80*1000]


# In[16]:


from scipy.signal import hilbert, chirp
from scipy.signal import blackman
from scipy.fft import fft, fftfreq

N = 80
w = blackman(N+1)

ffta = lambda x: np.abs(fft(np.append(x.values,x.values[0]))[:80])
ffta.__name__ = 'ffta'

fftw = lambda x: np.abs(fft(np.append(x.values,x.values[0])*w)[:80])
fftw.__name__ = 'fftw'

def add_features(df):
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    
    df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1)
    #df['u_out_lag1'] = df.groupby('breath_id')['u_out'].shift(1)
    df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-1)
    #df['u_out_lag_back1'] = df.groupby('breath_id')['u_out'].shift(-1)
    df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2)
    #df['u_out_lag2'] = df.groupby('breath_id')['u_out'].shift(2)
    df['u_in_lag_back2'] = df.groupby('breath_id')['u_in'].shift(-2)
    #df['u_out_lag_back2'] = df.groupby('breath_id')['u_out'].shift(-2)
    df['u_in_lag3'] = df.groupby('breath_id')['u_in'].shift(3)
    #df['u_out_lag3'] = df.groupby('breath_id')['u_out'].shift(3)
    df['u_in_lag_back3'] = df.groupby('breath_id')['u_in'].shift(-3)
    #df['u_out_lag_back3'] = df.groupby('breath_id')['u_out'].shift(-3)
    df['u_in_lag4'] = df.groupby('breath_id')['u_in'].shift(4)
    #df['u_out_lag4'] = df.groupby('breath_id')['u_out'].shift(4)
    df['u_in_lag_back4'] = df.groupby('breath_id')['u_in'].shift(-4)
    #df['u_out_lag_back4'] = df.groupby('breath_id')['u_out'].shift(-4)
    df = df.fillna(0)
    
    df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
    #df['breath_id__u_out__max'] = df.groupby(['breath_id'])['u_out'].transform('max')
    
    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
    df['u_out_diff1'] = df['u_out'] - df.groupby('breath_id')['u_out'].shift(1)
    df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
    df['u_out_diff2'] = df['u_out'] - df.groupby('breath_id')['u_out'].shift(2)
    
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    
    df['u_in_diff3'] = df['u_in'] - df['u_in_lag3']
    #df['u_out_diff3'] = df['u_out'] - df['u_out_lag3']
    df['u_in_diff4'] = df['u_in'] - df['u_in_lag4']
    #df['u_out_diff4'] = df['u_out'] - df['u_out_lag4']
    df['cross']= df['u_in']*df['u_out']
    df['cross2']= df['time_step']*df['u_out']
    
    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df['R__C'] = df["R"].astype(str) + '__' + df["C"].astype(str)
    df = pd.get_dummies(df)

    df['fft_u_in'] = df.groupby('breath_id')['u_in'].transform(ffta)
    df['fft_u_in_w'] = df.groupby('breath_id')['u_in'].transform(fftw)
    df['analytical'] = df.groupby('breath_id')['u_in'].transform(hilbert)
    df['envelope'] = np.abs(df['analytical'])
    df['phase'] = np.angle(df['analytical'])
    df['unwrapped_phase'] = df.groupby('breath_id')['phase'].transform(np.unwrap)
    df['phase_shift1'] = df.groupby('breath_id')['unwrapped_phase'].shift(1).astype(np.float32)
    df['IF'] = df['unwrapped_phase'] - df['phase_shift1'].astype(np.float32)
    df = df.fillna(0)
    
    df = df.drop('analytical',axis=1)
    
    return df

train = add_features(train)
test = add_features(test)

print('Train dataframe shape',train.shape)
train.head()


# In[17]:


targets = train[['pressure']].to_numpy().reshape(-1, 80)
train.drop(['pressure', 'id', 'breath_id'], axis=1, inplace=True)
test = test.drop(['id', 'breath_id'], axis=1)

COLS = list(train.columns)
print('Number of feature columns =', len(COLS) )

RS = RobustScaler()
train = RS.fit_transform(train)
test = RS.transform(test)

train = train.reshape(-1, 80, train.shape[-1])
test = test.reshape(-1, 80, train.shape[-1])

train = np.float32(train)
test = np.float32(test)


# In[ ]:





# In[18]:


EPOCH = 10 if DEBUG else 300
BATCH_SIZE = 1024
NUM_FOLDS = 10

# detect and init the TPU
#tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()

# instantiate a distribution strategy
#tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# GET GPU STRATEGY
gpu_strategy = tf.distribute.get_strategy()

with gpu_strategy.scope():
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2021)
    test_preds = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(train, targets)):
        K.clear_session()
        
        print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)
        X_train, X_valid = train[train_idx], train[test_idx]
        y_train, y_valid = targets[train_idx], targets[test_idx]
        
        checkpoint_filepath = f"folds{fold}.hdf5"
        if TRAIN_MODEL:
            model = keras.models.Sequential([
                keras.layers.Input(shape=train.shape[-2:]),
                keras.layers.Bidirectional(keras.layers.LSTM(1024, return_sequences=True)),
                keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True)),
                keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)),
                keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
                keras.layers.Dense(128, activation='selu'),
                keras.layers.Dense(1),
            ])
            model.compile(optimizer="adam", loss="mae")

            lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=1)
            es = EarlyStopping(monitor="val_loss", patience=60, verbose=1, mode="min", restore_best_weights=True)
            sv = keras.callbacks.ModelCheckpoint(
                checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                save_weights_only=False, mode='auto', save_freq='epoch',
                options=None
            )
            
            model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=[lr, es, sv])
            
        else:
            model = keras.models.load_model('../input/finetune-of-tensorflow-bidirectional-lstm/'+checkpoint_filepath)

        if INFER_TEST:
            print(' Predicting test data...')
            test_preds.append(model.predict(test,verbose=0).squeeze().reshape(-1, 1).squeeze())
                    
        if COMPUTE_LSTM_IMPORTANCE:
            results = []
            print(' Computing LSTM feature importance...')
            
            # COMPUTE BASELINE (NO SHUFFLE)
            oof_preds = model.predict(X_valid, verbose=0).squeeze() 
            baseline_mae = np.mean(np.abs( oof_preds-y_valid ))
            results.append({'feature':'BASELINE','mae':baseline_mae})           

            for k in tqdm(range(len(COLS))):
                
                # SHUFFLE FEATURE K
                save_col = X_valid[:,:,k].copy()
                np.random.shuffle(X_valid[:,:,k])
                        
                # COMPUTE OOF MAE WITH FEATURE K SHUFFLED
                oof_preds = model.predict(X_valid, verbose=0).squeeze() 
                mae = np.mean(np.abs( oof_preds-y_valid ))
                results.append({'feature':COLS[k],'mae':mae})
                X_valid[:,:,k] = save_col
         
            # DISPLAY LSTM FEATURE IMPORTANCE
            print()
            df = pd.DataFrame(results)
            df = df.sort_values('mae')
            plt.figure(figsize=(10,20))
            plt.barh(np.arange(len(COLS)+1),df.mae)
            plt.yticks(np.arange(len(COLS)+1),df.feature.values)
            plt.title('LSTM Feature Importance',size=16)
            plt.ylim((-1,len(COLS)+1))
            plt.plot([baseline_mae,baseline_mae],[-1,len(COLS)+1], '--', color='orange',
                     label=f'Baseline OOF\nMAE={baseline_mae:.3f}')
            plt.xlabel(f'Fold {fold+1} OOF MAE with feature permuted',size=14)
            plt.ylabel('Feature',size=14)
            plt.legend()
            plt.show()
                               
            # SAVE LSTM FEATURE IMPORTANCE
            df = df.sort_values('mae',ascending=False)
            df.to_csv(f'lstm_feature_importance_fold_{fold+1}.csv',index=False)
                               
        # ONLY DO ONE FOLD
        if ONE_FOLD_ONLY: break


# In[ ]:





# In[19]:


if not DEBUG:
    if INFER_TEST:
        PRESSURE_MIN = pressure_values[0]
        PRESSURE_MAX = pressure_values[-1]
        PRESSURE_STEP = pressure_values[1] - pressure_values[0]

        # NAME POSTFIX
        postfix = ''
        if ONE_FOLD_ONLY: 
            NUM_FOLDS = 1
            postfix = '_fold_1'

        # ENSEMBLE FOLDS WITH MEAN
        submission["pressure"] = sum(test_preds)/NUM_FOLDS
        submission.to_csv(f'submission_mean{postfix}.csv', index=False)

        # ENSEMBLE FOLDS WITH MEDIAN
        submission["pressure"] = np.median(np.vstack(test_preds),axis=0)
        submission.to_csv(f'submission_median{postfix}.csv', index=False)

        # ENSEMBLE FOLDS WITH MEDIAN AND ROUND PREDICTIONS
        submission["pressure"] =\
            np.round( (submission.pressure - PRESSURE_MIN)/PRESSURE_STEP ) * PRESSURE_STEP + PRESSURE_MIN
        submission.pressure = np.clip(submission.pressure, PRESSURE_MIN, PRESSURE_MAX)
        submission.to_csv(f'submission_median_round{postfix}.csv', index=False)

        # DISPLAY SUBMISSION.CSV
        print(f'submission{postfix}.csv head')
        display( submission.head() )

