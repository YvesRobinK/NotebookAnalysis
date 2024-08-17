#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# In this notebook, I show you how to generate time-series signals and image spectrogram data from audio. This notebook contains several functions making it easy for you to use to generate a dataset. 
# 
# You can use the amplitude signal or fft in an RNN, mel spectrogram with a CNN (the most popular and usually successful approach), or try some uncommon feature engineering with perhaps the tempogram to see if it works. 
# 
# In this notebook, I show you how to generate and visualize several features, and hopefully, this will help you understand audio data and preprocessing better.
# 
# ### Features
# The following features will be explored in this notebook
# - amplitude signal
# - power spectrum - fft
# - spectrogram - stft
# - mel spectrogram 
# - tempo
# - tempogram
# - mfcc with different deltas
# 
# ### Sources
# Some code was inspired by or borrowed from the following sources
# - https://github.com/musikalkemist/DeepLearningForAudioWithPython
# - https://www.kaggle.com/kaerunantoka/birdclef2022-create-image-data-from-audio-data/notebook
# - https://librosa.org/doc/latest/index.html

# # Imports and Config

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats
import json
import glob
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from soundfile import SoundFile
import IPython.display as ipd
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
plt.style.use('ggplot')


# In[2]:


class CFG:
    SR = 32_000
    duration = 5
    
    n_fft = 2048
    hop_length = n_fft // 4
    
    n_mels = 224
    fmin = 20
    fmax = 16000


# In[3]:


train = pd.read_csv("../input/birdclef-2022/train_metadata.csv")
train_paths = glob.glob("../input/birdclef-2022/train_audio/*")
print(f"example path: {train_paths[0]}\n")
print(f"number of birds: {len(train_paths)}\n")
print(f"metadata columns: {train.columns.values}")


# In[4]:


train.head(3)


# In[5]:


# This is just a random file I chose, you can look at any
EX_FILE = "../input/birdclef-2022/train_audio/cacgoo1/XC144036.ogg"


# # Listen

# In[6]:


ipd.Audio(EX_FILE)


# # Audio Signal - Time-Amplitude
# 
# Amplitude Signal - get the audio signal as a list of floating point values representing the amplitude of the signal at any given time (Amplitude being the sound wave measured from its equilibrium position).

# In[7]:


def get_signal(path):
    """ Get audio signal from librosa """
    signal, sr = librosa.load(path, sr=CFG.SR)
    return signal

def plot_signal(signal):
    """ Plots the time-amplitude graph of the audio signal """
    plt.figure(figsize=(12,8))
    librosa.display.waveshow(signal, sr=CFG.SR)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Audio Signal - Time-Amplitude")
    plt.show()


# In[8]:


signal = get_signal(EX_FILE)
plot_signal(signal)


# # Power Spectrum - FFT
# Fast-Fourier Transform - used to create a power spectrum by computing the magnitude and frequency of a signal

# In[9]:


def calculate_mag_freq(signal):
    """ Computes the magnitude and frequency given a signal """
    fft = np.fft.fft(signal)

    # calculate abs values on complex numbers to get magnitude
    spectrum = np.abs(fft)

    # create frequency variable
    f = np.linspace(0, CFG.SR, len(spectrum))

    # only half the plot is needed because the graph is symetrical
    left_spectrum = spectrum[:int(len(spectrum)/2)]
    left_freq = f[:int(len(spectrum)/2)]
    
    return left_spectrum, left_freq

def plot_spectrum(magnitude, freq):
    """ Plots a spectrum given a magnitude and frequency """
    plt.figure(figsize=(12, 8))
    plt.plot(freq, magnitude)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.title("Power Spectrum")
    plt.show()


# In[10]:


magnitude, frequency = calculate_mag_freq(signal)
plot_spectrum(magnitude, frequency)


# # Spectrogram - STFT
# Short-Time Fourier Transform - used to create a spectrogram, then I convert the spectrogram from amplitude to decibels to calculate the log spectrogram

# In[11]:


def calculate_spectrogram(signal):
    """ Computes a log spectrogram given a signal """
    stft = librosa.stft(signal, n_fft=CFG.n_fft, hop_length=CFG.hop_length)
    spectrogram = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    return log_spectrogram

def plot_spectrogram(log_spectrogram):
    """ Plots a spectrogram """
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(log_spectrogram, sr=CFG.SR, hop_length=CFG.hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram (dB)")
    plt.show()


# In[12]:


log_spec = calculate_spectrogram(signal)
plot_spectrogram(log_spec)


# # Mel Spectrogram
# Mel Spectrogram - very popular and effective image data that provides our models with sound information similar to what a human would perceive

# In[13]:


def calculate_melspec(signal):
    """ Computes a mel spectrogram """
    melspec = librosa.feature.melspectrogram(
        y=signal, sr=CFG.SR, n_mels=CFG.n_mels, fmin=CFG.fmin, fmax=CFG.fmax,
    )

    melspec = librosa.power_to_db(melspec).astype(np.float32)
    return melspec

def plot_melspec(melspec):
    """ Plots a mel spectrogram """
    plt.figure(figsize=(12,8))
    img = librosa.display.specshow(melspec, x_axis="time",
                                   y_axis="mel", sr=CFG.SR,
                                   fmax=CFG.fmax) 
    plt.colorbar(img, format="%+2.0f dB")
    plt.title("Mel-Frequency Spectrogram")
    plt.show()


# In[14]:


melspec = calculate_melspec(signal)
plot_melspec(melspec)


# # Additional Features
# Tempo - the speed which the sound is played as a tempo
# 
# Tempogram - a image representation of tempo information
# 
# mfcc delta - Mel-frequency cepstral coefficients at different deltas computed based on Savitsky-Golay filtering.

# In[15]:


def calculate_tempo(signal):
    """ Computes the tempo of an audio signal """
    onset_env = librosa.onset.onset_strength(y=signal, sr=CFG.SR)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=CFG.SR)
    return tempo

def plot_tempogram(signal):
    """ PLots a autocorrelation tempogram """
    oenv = librosa.onset.onset_strength(y=signal, sr=CFG.SR, hop_length=CFG.hop_length)
    tempogram = librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=CFG.SR,
                                                  hop_length=CFG.hop_length)
    # Compute the auto-correlation tempogram, unnormalized to make comparison easier
    ac_tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=CFG.SR,
                                             hop_length=CFG.hop_length, norm=None)
                             
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(ac_tempogram, sr=CFG.SR, hop_length=CFG.hop_length,
                         x_axis="time", y_axis="tempo", cmap="magma")
    plt.title("Autocorrelation Tempogram")
    plt.show()
    
def mfcc_delta(signal):
    """ Plots the mfcc, mfcc delta, and mfcc delta2 """
    mfcc = librosa.feature.mfcc(y=signal, sr=CFG.SR)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    fig, ax = plt.subplots(nrows=3, figsize=(12, 8), sharex=True, sharey=True)
    img1 = librosa.display.specshow(mfcc, ax=ax[0], x_axis='time')
    ax[0].set(title='MFCC')
    ax[0].label_outer()
    img2 = librosa.display.specshow(mfcc_delta, ax=ax[1], x_axis='time')
    ax[1].set(title=r'MFCC-$\Delta$')
    ax[1].label_outer()
    img3 = librosa.display.specshow(mfcc_delta2, ax=ax[2], x_axis='time')
    ax[2].set(title=r'MFCC-$\Delta^2$')
    fig.colorbar(img1, ax=[ax[0]])
    fig.colorbar(img2, ax=[ax[1]])
    fig.colorbar(img3, ax=[ax[2]])
    fig.show()


# In[16]:


print("tempo:", calculate_tempo(signal))


# In[17]:


plot_tempogram(signal)


# In[18]:


mfcc_delta(signal)


# In[ ]:




