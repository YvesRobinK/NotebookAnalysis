#!/usr/bin/env python
# coding: utf-8

# # Feature engineering

# ## The feature engineering stage (with vectorization implementation) was added. Vectorization will help increase the calculation speed to get more features.

# ## This notebook is based in Giba's Notebook 
# 
# ### ["Tabular XGboost GPU + FFT GPU + Cuml = FAST"](https://www.kaggle.com/titericz/0-525-tabular-xgboost-gpu-fft-gpu-cuml-fast)

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import os
import gc
import time
from scipy.interpolate import interp1d
import lightgbm as lgb
import xgboost as xgb
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
from scipy.stats import rankdata
import IPython.display as ipd  # To play sound in the notebook

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, label_ranking_average_precision_score
import soundfile as sf
import seaborn as sns

# Librosa Libraries
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt

import cuml as cm
import cupy as cp
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


trainfiles = glob.glob( '../input/rfcx-species-audio-detection/train/*.flac' )
testfiles = glob.glob( '../input/rfcx-species-audio-detection/test/*.flac' )
len(trainfiles), len(testfiles), trainfiles[0]


# In[ ]:





# In[3]:


traint = pd.read_csv( '../input/rfcx-species-audio-detection/train_tp.csv' )
trainf = pd.read_csv( '../input/rfcx-species-audio-detection/train_fp.csv' )
traint.shape, trainf.shape


# In[4]:


traint.head()


# In[5]:


# Number of species
len(traint['species_id'].unique())


# In[6]:


trainf.head()


# In[7]:


trainf.describe()


# In[8]:


#Extra information
train_general = pd.concat([traint, trainf])
train_general['t_diff'] = train_general['t_max'] - train_general['t_min']
train_general['f_diff'] = train_general['f_max'] - train_general['f_min']


# In[9]:


train_general.describe()


# In[10]:


# Species
sns.countplot(train_general['species_id'])


# In[11]:


# Frequency domain
def figurecpFTT(data,samplerate):
    # Frequency domain representation
    data = cp.array(data)
    fourierTransform = cp.fft.fft(data)/len(data)           # Normalize amplitude
    fourierTransform = fourierTransform[:len(data)//2] # Exclude sampling frequency

    tpCount     = len(data)
    values      = cp.arange(int(tpCount/2))
    timePeriod  = tpCount/samplerate
    frequencies = cp.asnumpy(values/timePeriod)
    
    absFFT = cp.asnumpy(abs(fourierTransform)) 
#     print(frequencies)
    plt.plot(frequencies,absFFT)
        


# In[12]:


def frequeciesVec(data,samplerate):
    tpCount     = 2*len(data)
    values      = cp.arange(int(tpCount/2))
    timePeriod  = tpCount/samplerate
    frequencies = cp.asnumpy(values/timePeriod)
    return cp.array(frequencies)


# In[13]:


#FFT
filesound = trainfiles[0]
data, samplerate = sf.read(filesound)
figurecpFTT(data, samplerate)


# # Feature engineering using Vectorization Implementation

# In[14]:


# Vectorization

def meanF(x): 
    return x.mean(axis=1)

def varianceF(x):
    return x.var(axis=1)

def skewnessF(x):
    skw = 3 * (x.mean(axis=1) - x[:,x.shape[1]/2])
    skw = skw / x.std(axis=1)
    return skw

def kurtosisF(x):
    z = ((x - x.mean(axis=1,keepdims=True))**4).sum(axis=1)
    n = x.shape[1]
    s = n*(x.std(axis=1))**4
    kur = z/s
    return kur

def totalpowerF(x):
    return (x**2).sum(axis=1)

def rmsF(x):
    x = x**2
    return cp.sqrt(x.mean(axis=1))

def stdF(x):
    return x.std(axis=1)

def centroidF(x,frequencies):     
    n = x * frequencies
    s = x.sum(axis=1)    
    centroid = n / s[:,None]
    return centroid.sum(axis=1)

def entropyF(x):
    px = x / (x.sum(axis=1))[:,None]
    r = px*cp.log2(px)
    return -r.sum(axis=1)

def peakF(x):    
    return x.max(axis=1)


# In[15]:


def featuresextractionVec(signalFFT):
    frequecies = frequeciesVec(signalFFT,samplerate).reshape( (1000,1440) )
    varfft = signalFFT.reshape( (1000,1440) )
    features = cp.array([meanF(varfft), varianceF(varfft), skewnessF(varfft), kurtosisF(varfft), totalpowerF(varfft), stdF(varfft), rmsF(varfft), entropyF(varfft), peakF(varfft), centroidF(varfft,frequecies) ])
    L=features.shape[0]*features.shape[1]
    features = features.reshape(1,L)[0]
    return features


# In[16]:


def extract_fft(fn):
    data, samplerate = sf.read(fn)
    data = cp.array(data)    
    varfft = cp.abs( cp.fft.fft(data)[:(len(data)//2)] )
    features = featuresextractionVec(varfft)
    return features


# In[17]:


FT = []
for fn in tqdm(traint.recording_id.values):
    FT.append( extract_fft( '../input/rfcx-species-audio-detection/train/'+fn+'.flac') )
FT = np.stack(FT)
gc.collect()

FT.shape


# In[18]:


# This loop runs in 7min using cupy(GPU) and 40min on numpy(CPU). ~7x Faster in GPU

FF = []
for fn in tqdm(trainf.recording_id.values):
    FF.append( extract_fft( '../input/rfcx-species-audio-detection/train/'+fn+'.flac' ) )
FF = np.stack(FF)
gc.collect()

FF.shape


# In[19]:


#Combine True Positives and False Positives

TRAIN = np.vstack( (FT, FF) )


del FT, FF
gc.collect()
TRAIN.shape


# In[20]:


TEST = []
for fn in tqdm(testfiles):
    TEST.append( extract_fft(fn) )
TEST = np.stack(TEST)
gc.collect()

TEST.shape


# In[21]:


#To Numpy format
TRAIN = cp.asnumpy(TRAIN)
TEST = cp.asnumpy(TEST)


# In[ ]:





# In[22]:


tt = traint[['recording_id','species_id']].copy()
tf = trainf[['recording_id','species_id']].copy()
tf['species_id'] = -1

TRAIN_TAB = pd.concat( (tt, tf) )

for i in range(24):
    TRAIN_TAB['s'+str(i)] = 0
    TRAIN_TAB.loc[TRAIN_TAB.species_id==i,'s'+str(i)] = 1

TRAIN_TAB.head()


# In[23]:


def saveFile(data, name):
    pickle_out = open(name,"wb")
    pickle.dump(data, pickle_out)
    pickle_out.close() 


# In[24]:


# Save TRAIN, TEST, TRAIN_TAB
saveFile(TRAIN,"TRAIN.pickle")
saveFile(TEST,"TEST.pickle")
saveFile(TRAIN_TAB,"TRAIN_TAB.pickle")


# In[25]:


# # To Open
# pickle_in = open("TRAIN.pickle","rb")
# TRAIN = pickle.load(pickle_in)
# pickle_in = open("TEST.pickle","rb")
# TEST = pickle.load(pickle_in)


# In[26]:


#1000 random features was selected to avoid a long training time

import random

random.seed(30)
imp_indx = random.sample(range(0, 10000), 1000) 
TRAIN = TRAIN[:,imp_indx]
TEST = TEST[:,imp_indx]


# In[ ]:





# In[27]:


TRAIN.shape, TEST.shape


# # Modeling

# In[28]:


from sklearn.preprocessing import StandardScaler

std = StandardScaler()
std.fit( np.vstack((TRAIN,TEST)) )

TRAIN = std.transform(TRAIN)
TEST  = std.transform(TEST)
gc.collect()


# In[29]:


TRAIN_TAB.shape


# In[30]:


sub = pd.DataFrame({'recording_id': [f.split('/')[-1].split('.')[0] for f in testfiles] })
gkf = GroupKFold(5)

SCORE = []
groups = TRAIN_TAB['recording_id'].values
for tgt in range(0,24):
    starttime = time.time()
    target = TRAIN_TAB['s'+str(tgt)].values

    ytrain = np.zeros(TRAIN.shape[0])
    ytest = np.zeros(TEST.shape[0])
    for ind_train, ind_valid in gkf.split( TRAIN, target, groups ):
        
        # Define 4 models
        model1 = xgb.XGBClassifier(n_estimators=1000,
                                   max_depth=6,
                                   learning_rate=0.09,
                                   verbosity=0,
                                   min_child_weight=1,
                                   objective='binary:logistic',
                                   subsample=0.95,
                                   colsample_bytree=0.95,
                                   random_state=2021,
                                   tree_method='gpu_hist',
                                   predictor='gpu_predictor',
                                   n_jobs=2,
                                   scale_pos_weight =  np.sum(target==0) / np.sum(target==1),
                                  )
#         scale_pos_weight = np.sum(target==0) / np.sum(target==1)
        model2 = cm.linear_model.LogisticRegression( C=1, max_iter=5000 )
        model3 = cm.svm.SVC(C=1.0, class_weight='balanced', probability=True, kernel='rbf', gamma='auto')
        model4 = cm.neighbors.KNeighborsClassifier(n_neighbors=10)
        
        # Train using GPUs
        model1.fit( X=TRAIN[ind_train], y=target[ind_train], eval_set=[(TRAIN[ind_valid], target[ind_valid])], eval_metric='auc', early_stopping_rounds=30, verbose=False )
        model2.fit( TRAIN[ind_train], target[ind_train] )
        model3.fit( TRAIN[ind_train], target[ind_train] )
        model4.fit( TRAIN[ind_train], target[ind_train] )
        
        # Predict valid and test sets
        yvalid1 = model1.predict_proba(TRAIN[ind_valid])[:,1]
        yvalid2 = model2.predict_proba(TRAIN[ind_valid])[:,1]
        yvalid3 = model3.predict_proba(TRAIN[ind_valid])[:,1]
        yvalid4 = model4.predict_proba(TRAIN[ind_valid])[:,1]
        ytest1 = model1.predict_proba(TEST)[:,1]
        ytest2 = model2.predict_proba(TEST)[:,1]
        ytest3 = model3.predict_proba(TEST)[:,1]
        ytest4 = model4.predict_proba(TEST)[:,1]
        
        #Rank predictions
        SZ = len(ind_valid) + len(ytest1)
        yvalid1 = rankdata( np.concatenate((yvalid1,ytest1)) )[:len(ind_valid)] / SZ
        yvalid2 = rankdata( np.concatenate((yvalid2,ytest2)) )[:len(ind_valid)] / SZ
        yvalid3 = rankdata( np.concatenate((yvalid3,ytest3)) )[:len(ind_valid)] / SZ
        yvalid4 = rankdata( np.concatenate((yvalid4,ytest4)) )[:len(ind_valid)] / SZ
        ytest1 = rankdata( np.concatenate((yvalid1,ytest1)) )[len(ind_valid):] / SZ
        ytest2 = rankdata( np.concatenate((yvalid2,ytest2)) )[len(ind_valid):] / SZ
        ytest3 = rankdata( np.concatenate((yvalid3,ytest3)) )[len(ind_valid):] / SZ
        ytest4 = rankdata( np.concatenate((yvalid4,ytest4)) )[len(ind_valid):] / SZ
        
        #Weighted average models
        ytrain[ind_valid] = (0.40*yvalid1+0.20*yvalid2+0.20*yvalid3+0.20*yvalid4) / 4.
        ytest += (0.40*ytest1+0.20*ytest2+0.20*ytest3+0.20*ytest4) / (4.*5)

    score = roc_auc_score(target, ytrain)
    print( 'Target AUC', tgt, score, time.time()-starttime )
    SCORE.append(score)
    
    TRAIN_TAB['y'+str(tgt)] = ytrain
    sub['s'+str(tgt)] = ytest

print('Overall Score:', np.mean(SCORE) )


# In[31]:


sub.head()


# In[32]:


sub.to_csv('submission_vec.csv', index=False)


# In[33]:


get_ipython().system('ls')


# # Thanks for sharing!
