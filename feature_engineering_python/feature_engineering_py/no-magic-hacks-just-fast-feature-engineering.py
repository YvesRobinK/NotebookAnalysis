#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random, os, pickle, scipy, math, time, joblib

import numpy as np
import pandas as pd
from scipy import signal
from scipy import interpolate

from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from sklearn.preprocessing import RobustScaler


# I was not able to discover the magic this time, as my data analysis skills (or lack thereof) still leave much to be desired. However, I was able to engineer a pretty sweet set of features. I think many of these were already publicly shared, but some of them are my own novel feature sets.
# 
# I do all my FE using numpy. On my machine it takes about a minute to create the 40-some features. Let's see how long it takes Kaggle Kernels...

# In[2]:


train = pd.read_csv('../input/ventilator-pressure-prediction/train.csv')
test  = pd.read_csv('../input/ventilator-pressure-prediction/test.csv')

# This will be explained later
CHOP = 33


# In[3]:


RC_TimeConstant = {
    (R, C): 400 * R * C
    for R in [5,20,50]
    for C in [10,20,50]
}
RC_Intercept = train.groupby(['R','C']).pressure.mean().to_dict()


# # Methods

# In[4]:


def numpy_ewma_vectorized_v2(data, window):
    # https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
        
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out


# In[5]:


def autocorr(x):
    # https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation
    result = np.correlate(x, x, mode='full')
    result = result[result.shape[0]//2:] / x.var() / len(x)
    result[np.isnan(result)] = 0
    result[np.isinf(result)] = 0
    return result


# In[6]:


def rolling_window(a, window):
    # https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


# In[7]:


def integrate(x, v, a, dt):
    # https://en.wikipedia.org/wiki/Fourth,_fifth,_and_sixth_derivatives_of_position
    outp = x + v * dt + 0.5 * a * dt ** 2
    return np.concatenate(([outp[0]], outp[:-1]))


# # Featurizer

# In[8]:


scalers = []


# In[9]:


def featurize(df, no_labels=False, transform=True, verbose=False):
    global scalers
    
    # Don't mess it up!
    df = df.copy()
    
    if not no_labels:
        pressure = df.pressure.values
        delta_pressure = np.concatenate(([0],np.diff(pressure)))
        delta_delta_pressure = np.concatenate(([0],np.diff(delta_pressure)))
        
    drop_cols = [
        col for col in df if col in ['id','breath_id','pressure']
    ]
    df = df[['u_out','R','C','time_step','u_in']].values.reshape(-1, 80, 5)
    
    # I transpose the df so that I can do sample[feature], rather than sample[:, feature]
    # We will un-do this operation at the end...
    df = df.transpose(0,2,1)
    
    new_cols = 41
    enew_cols = 1
    out  = np.zeros((df.shape[0], new_cols, df.shape[2]), dtype=np.float32)*np.nan
    eout = np.zeros((df.shape[0], enew_cols), dtype=np.int64)
    
    feature_uout = 0
    feature_R = 1
    feature_C = 2
    feature_ts = 3
    feature_uin = 4

    mask = (df[:, feature_uout] == 0) & (df[:, feature_ts] >= 0)
    
    #######################################################
    # Embedding Features:
    # MODE: https://www.kaggle.com/marutama/eda-about-pressure-with-colored-charts
    eout[:, 0] = ((df[:, feature_uin, -1] > 4.8) & (df[:, feature_uin, -1] < 5.1)).astype(np.int64)
    
    # I don't embed R or C. I just pass those in as continuous values. 
    # That was probably a mistake..
    
    #######################################################

    it = tqdm(df) if verbose else df
    for idx, sample in enumerate(it):
        
        # CACHE:
        cached_uin = sample[feature_uin].copy()
        
        # Before anything else:
        # https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/282370
        sample[feature_uin] = np.abs(sample[feature_uin])**0.25 * np.sign(sample[feature_uin])
        
        feature_num = 0
        # 5. Transformation
        # The resistance of the valve is R_in ∝ 1/d 4 (Poiseuille’s law) where d, the opening of the valve
        # https://arxiv.org/abs/2102.06779
        out[idx, feature_num] = np.sqrt(np.abs(sample[feature_uin])) * np.sign(sample[feature_uin])
        feature_num += 1
        
        # 6. ts derivative (delta_ts)
        out[idx, feature_num] = np.concatenate(([0],np.diff(sample[feature_ts])))
        out[idx, feature_num, 0] = out[idx, feature_num, 1:].min()
        feature_delta_ts = feature_num
        feature_num += 1

        # 7. u_in derivative (delta_uin)
        out[idx, feature_num] = np.concatenate(([0],np.diff(sample[feature_uin])))
        out[idx, feature_num, 0] = out[idx, feature_num, 1:].min()
        feature_uin_derivative = feature_num
        feature_num += 1
        
        # 8. u_in 2nd order derivative (delta_delta_uin)
        out[idx, feature_num] = np.concatenate(([0],np.diff(out[idx, feature_uin_derivative])))
        out[idx, feature_num, 0] = out[idx, feature_num, 1:].min()
        feature_uin_derivative_derivative = feature_num
        feature_num += 1
        
        # 9. u_in ema3,4
        out[idx, feature_num] = numpy_ewma_vectorized_v2(sample[feature_uin], 3)
        feature_num += 1
        out[idx, feature_num] = numpy_ewma_vectorized_v2(sample[feature_uin], 4)
        feature_num += 1
        
        # 10. feature_uin_derivative ema3
        out[idx, feature_num] = numpy_ewma_vectorized_v2(out[idx, feature_uin_derivative], 3)
        feature_num += 1
        
        # 11. u_in autocorr
        out[idx,feature_num] = autocorr(sample[feature_uin])
        feature_num += 1
        
        # 12. integrated uin w.r.t. time
        out[idx,feature_num] = np.cumsum(sample[feature_uin] * out[idx, feature_delta_ts])
        feature_uin_integrated = feature_num
        feature_num += 1
        
        # 13. d_uin / d_ts
        out[idx,feature_num] = out[idx, feature_uin_derivative] / out[idx, feature_delta_ts]
        out[idx,feature_num][np.isnan(out[idx,feature_num])] = 0  # TODO: Check if this is a good empty fill....
        out[idx,feature_num][np.isinf(out[idx,feature_num])] = 0
        feature_num += 1
        
        ######################################
        window = 4
        rolling = rolling_window(sample[feature_uin], window)
        
        # 14. uin_max
        tmp = rolling.max(axis=-1)
        out[idx,feature_num] = np.concatenate(([tmp[0]]*(window-1), tmp))
        feature_num += 1
        
        # 15. uin_std
        tmp = rolling.std(axis=-1)
        tmp[np.isnan(tmp)] = 0
        out[idx,feature_num] = np.concatenate(([tmp[0]]*(window-1), tmp))
        feature_num += 1
        
        # 16. uin_mean - do we need this? similar to ema...
        tmp = rolling.mean(axis=-1)
        out[idx,feature_num] = np.concatenate(([tmp[0]]*(window-1), tmp))
        feature_num += 1
        
        # 17. uin_slopes
        tmp = sample[feature_ts, window-1:].reshape(-1,1)
        tmp = ((tmp*rolling).mean(axis=1) - tmp.mean()*rolling.mean(axis=1)) / ((tmp**2).mean() - (tmp.mean())**2)
        out[idx,feature_num] = np.concatenate(([tmp[0]]*(window-1), tmp))
        feature_num += 1
        
        #############################
            
        # 18. uin_first value
        out[idx,feature_num] = sample[feature_uin, 0] * np.ones(sample.shape[-1])
        feature_num += 1
        
        # 19. sum(uin[44:])
        out[idx,feature_num] = sample[feature_uin, 44:].sum() * np.ones(sample.shape[-1])
        feature_num += 1
        
        # 20. ts_last_value
        out[idx,feature_num] = sample[feature_ts, -1] * np.ones(sample.shape[-1])
        feature_num += 1
        
        # 21/ u_in right before first u_out=1
        tmp_last_val = np.nonzero(sample[feature_uin] * (1-sample[feature_uout]))[0]
        if len(tmp_last_val) > 0:
            out[idx,feature_num] = sample[feature_uin, tmp_last_val[-1]] * np.ones(sample.shape[-1])
        else:
            out[idx,feature_num] = 0
        feature_num += 1
        
        # 22. u_in_mean <-- where u_out = 0
        out[idx,feature_num] = (sample[feature_uin] * (1-sample[feature_uout])).sum() / (1-sample[feature_uout]).sum() * np.ones(sample.shape[-1])
        feature_num += 1

        # 23. uin_max
        out[idx,feature_num] = sample[feature_uin].max() * np.ones(sample.shape[-1])
        feature_num += 1

        # 24. uin_std
        out[idx,feature_num] = sample[feature_uin].std() * np.ones(sample.shape[-1])
        feature_num += 1
        
        # 25. uin_skew
        out[idx,feature_num] = scipy.stats.skew(sample[feature_uin]) * np.ones(sample.shape[-1])
        feature_num += 1
        
        #############################
        # 26. uin_dist2mean
        out[idx,feature_num] = sample[feature_uin].mean() - sample[feature_uin]
        feature_uin_dist2mean = feature_num
        feature_num += 1
        
        # 27. Dist2slope integration...
        out[idx,feature_num] = np.cumsum(out[idx, feature_uin_dist2mean] * out[idx, feature_delta_ts]) 
        feature_num += 1
        
        # 28-30. uin_lags (3)
        feature_uin_lags = feature_num
        for lag in range(1,4):
            out[idx,feature_num] = np.concatenate((
                [0]*lag,
                sample[feature_uin,:-lag]
            ))
            feature_num += 1
            
        # 31-33. uin_integrated_lags (3)
        for lag in range(1,4):
            out[idx,feature_num] = np.concatenate((
                [0]*lag,
                out[idx, feature_uin_integrated,:-lag]
            ))
            feature_num += 1
            
        # 34-36. uin_derivative_lags (3)
        feature_uin_derivative_lags = feature_num
        for lag in range(1,4):
            out[idx,feature_num] = np.concatenate((
                [0]*lag,
                out[idx, feature_uin_derivative,:-lag]
            ))
            feature_num += 1
        
        # 37-39. uin_seeks (3)
        feature_uin_seeks = feature_num
        for seek in range(1,4):
            out[idx,feature_num] = np.concatenate((
                sample[feature_uin,seek:],
                [0]*seek
            ))
            feature_num += 1
            
        
        # 40. Physics Integration
        out[idx,feature_num] = integrate(
            x=sample[feature_uin],
            v=out[idx, feature_uin_derivative],
            a=out[idx, feature_uin_derivative_derivative],
            dt=out[idx, feature_delta_ts],
        )
        feature_num += 1
        
        # # Real Physics: Inhale Volume
        # # https://www.kaggle.com/motloch/vpp-pip-analysis-and-new-features
        # R = sample[feature_R, 0]
        # C = sample[feature_C, 0]
        # inhale_factor = np.exp(-sample[feature_ts] / RC_TimeConstant[(R,C)])
        # vf = cached_uin.cumsum() * R / inhale_factor  #  <-- feature1
        # out[idx,feature_num] = vf / 450# + RC_Intercept[(R,C)]
        # feature_num += 1

        # 41-42. Weird uin cumsums
        # I forgot who I stole these from
        # Find the index of the first u_out=1
        if len(tmp_last_val) ==0:
            # We cannot compute this feature...
            out[idx,feature_num+0] = 0
            out[idx,feature_num+1] = 0
        else:
            idx_end = 1 + tmp_last_val[-1]
            rev_cumsum = sample[feature_uin, :idx_end].cumsum()[::-1]
            
            out[idx,feature_num+0, :idx_end] = rev_cumsum - sample[feature_uin, :idx_end]
            out[idx,feature_num+1, :idx_end] = rev_cumsum.sum() - sample[feature_uin, :idx_end]
            out[idx,feature_num+0, idx_end:] = out[idx,feature_num+0, idx_end-1]
            out[idx,feature_num+1, idx_end:] = out[idx,feature_num+1, idx_end-1]
        feature_num += 2
        
        # 43. uin * timestep
        out[idx,feature_num] = sample[feature_uin] * sample[feature_ts]
        feature_num += 1
        
        # 44. deltats**2
        out[idx,feature_num] =  out[idx, feature_delta_ts] ** 2
        feature_num += 1
        

    # For my models, I only train on the first ~33 samples
    # Chop off u_out
    df = np.concatenate((df[:, 1:],out), axis=1).astype(np.float32)
    df = df.transpose(0,2,1)
    df = df[:,:CHOP] #chop
    mask = mask.reshape(-1, 80).astype(np.float32)
    mask = mask[:, :CHOP] #chop
    
    # My models regress pressure, and its first and second derivatives
    if not no_labels:
        pressure = pressure.reshape(-1, 80).astype(np.float32)
        delta_pressure = delta_pressure.reshape(-1, 80).astype(np.float32)
        delta_delta_pressure = delta_delta_pressure.reshape(-1, 80).astype(np.float32)
        
        pressure = pressure[:, :CHOP] #chop
        delta_pressure = delta_pressure[:, :CHOP] #chop
        delta_delta_pressure = delta_delta_pressure[:, :CHOP] #chop
        
    else:
        pressure = 1
        delta_pressure = 1
        delta_delta_pressure = 1
    
    ###
    # Baked Transform
    if transform:
        for col in range(df.shape[-1]):
            df[:,:,col] = scalers[col].transform(df[:,:,col].reshape(-1,1)).reshape(df[:,:,col].shape)

    return df, eout, pressure, delta_pressure, delta_delta_pressure, mask


# # Run it 

# In[10]:


df, eout, pressure, delta_pressure, delta_delta_pressure, mask = featurize(train.append(test), transform=False, verbose=True)

print(df.shape)
print(eout.shape)
print(pressure.shape)
print(delta_pressure.shape)
print(delta_delta_pressure.shape)
print(mask.shape)


# < 2min. Not bad Kaggle. This is fast enough that we can actually perform augmentations on uin and ts and regenerate the dataset live......... but that's for another notebook.

# # Target Inspection

# In[11]:


plt.title('Main Mode')
plt.hist(eout[:,0].flatten(), bins=60)
plt.show()

plt.title('Pressure')
plt.hist(pressure[:,0].flatten(), bins=100)
plt.show()

plt.title('dPressure')
plt.hist(delta_pressure[:,0].flatten(), bins=100)
plt.show()

plt.title('ddPressure')
plt.hist(delta_delta_pressure[:,0].flatten(), bins=100)
plt.show()


# # Feature Inspection

# In[12]:


scalers = []
for col in tqdm(range(df.shape[-1])):
    scalers.append(
        RobustScaler().fit(df[:,:,col].reshape(-1,1))
    )
    
    plt.title(str(col))
    plt.hist(df[:,:,col].flatten(), bins=100)
    plt.show()


# # Notes

# Some of these features look like they could benefit from `** 0.x` or even `np.log1p` type transformations (especially the last feature, which you really have to zoom into). Explore around! Also, when you wish to apply the features to a train or test, or a subset fold of train, leave `transform=True` to apply RobustScaler.

# In[ ]:




