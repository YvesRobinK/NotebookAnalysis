#!/usr/bin/env python
# coding: utf-8

# # Running algorithms - Feature Engineering for fast inference
# 
# 
# One of the main problem in this competition is about the inference : we have to predict outcomes one by one, which take quite some time.
# This bottleneck also impact what we can do with feature engineering as we are limited to things that calculate fast during the inference process. 
# I looked what could be used iteratively to stay within this time constraint. 
# It appears that there is a whole class of algorithms - running algorithms (or sometimes streaming algorithms) - that are designed exactly for this kind of problems.
# 
# I was about to publish a notebook about financial feature engineering alone, but I figured this wouldn't be very useful without inference implementations. So I decided to fuse both and here we are : in this notebook I give you some of the usual financial feature engineering tools and, along some standard python implementation, I try to provide you with streaming algorithms that will allow you to calculate them during inference. And it turns out that using numpy is **EXTREMELY FAST**. What is not in this notebook : some way to decide what parameter value to use (lag) or on which features to use this techniques (this might be the topic of another notebook 😉). 
# 
# Please keep in mind that those implementations are my owns. So there might be some problems (well there are some problems, I even point them out), feel free to comment with a correction. 
# 
# If you want to go further, you can check my other works (about [Intraday Feature Exploration](https://www.kaggle.com/lucasmorin/complete-intraday-feature-exploration),[Target Engineering](https://www.kaggle.com/lucasmorin/target-engineering-patterns-denoising), and [using yfinance to download financial data in Ptyhon](https://www.kaggle.com/lucasmorin/downloading-market-data)). Feel free to upvote / share my notebooks.
# Lucas
# 
# ## updates :
# 
# v.11 : added a complete exemple with a model
# 
# v.13 : 
# 
# - Did some testing and modified edge case
# - Added a dummy environnement for testing
# - will probably split training and submission
# 
# ## Features engineering techniques :
# 
# - [Moving Average (starter)](#Moving_Average) 🏃🏃🏃
# - [Moving Moments](#Moving_Moments) (variance, skew, kurtosis) 🏃🏃
# - [Exponentially Weighted Moving Average](#EWMA) 🏃🏃🏃
# - [Past day average ](#PDA)🏃🏃
# - [Paste Trade Information](#PTI) 🏃🏃🏃
# - [Differentiation ](#DIFF)
# - [Fractional differentiation](#FDIFF)
# - [Entropy](#Entropy)
# 
# # Application :
# 
# - [Bottleneck encoder + MLP + Keras Tuner](#MLP)
# - [Dummy Environnement](#Dummy_Env)
# - [Submission](#Submission)

# # Loading base packages
# 
# Nothing too surprising here. Collections deque data structure will help us keep track of past data.

# In[1]:


import numpy as np 
import pandas as pd 
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit

import collections
import math

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# # Load data
# 
# Loading a pickle file. Check this notebook [pickling](https://www.kaggle.com/quillio/pickling) if you haven't pickled your data set yet. Check this notebook [one liner to halve your memory usage](https://www.kaggle.com/jorijnsmit/one-liner-to-halve-your-memory-usage) if you want to reduce memory usage before pickling.

# In[2]:


get_ipython().run_cell_magic('time', '', "train_pickle_file = '/kaggle/input/pickling/train.csv.pandas.pickle'\ntrain_data = pickle.load(open(train_pickle_file, 'rb'))\ntrain_data.info()\n")


# <a id='Moving_Average'></a>
# # Moving Average (starter)
# 
# It is a very standard indicator for financial time series. The goal here is to build a demo. Honestly from what I have seen so far, as we have different securities in the data running windows mean doesn't appears to be that usefull for lower windows.
# If you want to test their importance the usual pandas way of doing that is simply :

# In[3]:


# Don't launch that as it may consume a lot of memory)
#rw = 10000
#train_data_rolled = train_data.rolling(window=rw).mean()


# For a streaming algorithm the idea is to build a class that allows to keep track of past values. Largely inspired from this [Stack exchange answer](https://stackoverflow.com/questions/5147378/rolling-variance-algorithm). 

# In[4]:


from collections import deque

class RunningMean:
    def __init__(self, WIN_SIZE=20, n_size = 1):
        self.n = 0
        self.mean = np.zeros(n_size)
        self.cum_sum = 0
        self.past_value = 0
        self.WIN_SIZE = WIN_SIZE
        self.windows = collections.deque(maxlen=WIN_SIZE+1)
        
    def clear(self):
        self.n = 0
        self.windows.clear()

    def push(self, x):
        
        x = fillna_npwhere_njit(x, self.past_value)
        self.past_value = x
        
        self.windows.append(x)
        self.cum_sum += x
        
        if self.n < self.WIN_SIZE:
            self.n += 1
            self.mean = self.cum_sum / float(self.n)
            
        else:
            self.cum_sum -= self.windows.popleft()
            self.mean = self.cum_sum / float(self.WIN_SIZE)

    def get_mean(self):
        return self.mean if self.n else np.zeros(n_size)

    def __str__(self):
        return "Current window values: {}".format(list(self.windows))

# Temporary removing njit as it cause many bugs down the line
# Problems mainly due to data types, I have to find where I need to constraint types so as not to make njit angry
#@njit
def fillna_npwhere_njit(array, values):
    if np.isnan(array.sum()):
        array = np.where(np.isnan(array), values, array)
    return array


# **We can check that it run fast (iterrrows allow to loop trough rows of a data frame as an interable). Here using numpy array instead of pandas datframe allow to go from 1600 it/sec to 9000+ it/seconds when keeping track of ALL 10000 tick lagged means. Using @gogo827jz fillna method allow to breach 10000 it/sec.** 
# 
# **It also shows you how easy it is to use for inference.**

# In[5]:


a = RunningMean(WIN_SIZE=10000)

for index, row in tqdm(train_data[:100000].iterrows()): 
    a.push(np.array(row))
    
a.get_mean()


# I still have two main problems here :
#  - <s> It doesn't seem to converge properly due to some rounding error (see below)</s> Thanks to @magokecol for pointing the mistake out !
#  - It does not handle na as is. I propose some code to use the last value at the moment, which might be pertinent for some fetaure but probably not all of them. It is also a bit slower when activated (from 1800 it/sec to 1400 with it/sec).
#  
# The second point is not problematic as is, but if we want to use it properly we might either want to replicate exactly what standards library (rolling) does or we want to apply the streaming algo to our whole train set, so as not to create a discrepency between the two.

# In[6]:


a = RunningMean(WIN_SIZE=10)

for index, row in pd.DataFrame({'col1':range(1,100)}).iterrows(): 
    a.push(np.array(row))
    
print(a.get_mean())
print((90+91+92+93+94+95+96+97+98+99)/10)


# # 😍

# <a id='Moving_Moments'></a>
# # Moving Moments (variance, skew, kurtosis)
# 
# The aforementionned stack exchange post also implement the variance :

# In[7]:


from __future__ import division
import collections
import math


class RunningStats:
    def __init__(self, WIN_SIZE=20, n_size = 1):
        self.n = 0
        self.mean = 0
        self.run_var = 0
        self.WIN_SIZE = WIN_SIZE
        self.past_value = 0
        self.windows = collections.deque(maxlen=WIN_SIZE+1)

    def clear(self):
        self.n = 0
        self.windows.clear()

    def push(self, x):
        
        x = fillna_npwhere_njit(x, self.past_value)
        self.past_value = x

        self.windows.append(x)

        if self.n < self.WIN_SIZE:
            # Calculating first variance
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            self.run_var += delta * (x - self.mean)
        else:
            # Adjusting variance
            x_removed = self.windows.popleft()
            old_m = self.mean
            self.mean += (x - x_removed) / self.WIN_SIZE
            self.run_var += (x + x_removed - old_m - self.mean) * (x - x_removed)

    def get_mean(self):
        return self.mean if self.n else np.zeros(n_size)

    def get_var(self):
        return self.run_var / (self.n) if self.n > 1 else np.zeros(n_size)

    def get_std(self):
        return math.sqrt(self.get_var())

    def get_all(self):
        return list(self.windows)

    def __str__(self):
        return "Current window values: {}".format(list(self.windows))


# It isn't really slower than the mean approach. So we might get the variance for almost free.

# In[8]:


a = RunningStats(WIN_SIZE=10000)

for index, row in tqdm(train_data[:100000].iterrows()): 
    a.push(np.array(row))
    
a.get_mean()


# I implement the modifications suggested in comments + add some way to handle missing values. As above this is not problematic for the mean. I think it might get a bit more problematic for the variance, as replacing with last values will systematically lower the variance.
# 
# Note : 
# - I haven't tested the variance toroughfully yet
# - The post refers to a blog wich refer to another post that gives a solution for [skew and kurtosis in C++](https://www.johndcook.com/blog/skewness_kurtosis/), I'll see what I can implement myself in Python.
# 

# For reference I'll leave the pandas implementation :

# In[9]:


#rw = 10000
#train_data_rolled_mean = train_data.rolling(window=rw).mean()
#train_data_rolled_var = train_data.rolling(window=rw).var()
#train_data_rolled_skew = train_data.rolling(window=rw).skew()
#train_data_rolled_kurt = train_data.rolling(window=rw).kurt()


# <a id='EWMA'></a>
# # Exponentially Weighted Moving Average
# 
# Python reference implementation :

# In[10]:


#train_data_ewm = train_data.ewm(span=rw, adjust=True).mean()


# Given that exponentially weighted moving average can be calculated iteratively without any memory, I feel like it would generally be better to use such Features. I use the formula alpha = 2 / (N+1) that give the same 'center of mass' as the traditional mean. My implementation :

# In[11]:


class RunningEWMean:
    def __init__(self, WIN_SIZE=20, n_size = 1, lt_mean = None):
        if lt_mean is not None:
            self.s = lt_mean
        else:
            self.s = np.zeros(n_size)
        self.past_value = np.zeros(n_size)
        self.alpha = 2 /(WIN_SIZE + 1)

    def clear(self):
        self.s = 0

    def push(self, x):
        
        x = fillna_npwhere_njit(x, self.past_value)
        self.past_value = x
        self.s = self.alpha * x + (1 - self.alpha) * self.s
        
    def get_mean(self):
        return self.s


# Somehow it seems to also work better than the standard average :

# In[12]:


a = RunningEWMean(WIN_SIZE=10)

for index, row in pd.DataFrame({'col1':range(1,100)}).iterrows(): 
    a.push(np.array(row))
    
print(a.get_mean())
print(pd.DataFrame({'col1':range(1,100)}).ewm(span=10, adjust=True).mean().iloc[98])


# And it is also a bit faster (11000 it/second):

# In[13]:


a = RunningEWMean(WIN_SIZE=10000)

for index, row in tqdm(train_data[:100000].iterrows()): 
    a.push(np.array(row))
    
a.get_mean()


# <a id='PDA'></a>
# # Past day average
# 
# Given that some long term moving average appears to have some gain, I figured it would probably make sense to calculate past day average.
# I haven't given a lot of attention with a pythonic way but I think I can give it a shot in a streaming way :

# In[14]:


class RunningPDA:
    def __init__(self):
        self.day = -1
        self.past_mean = 0
        self.cum_sum = 0
        self.day_instances = 0
        self.past_value = 0

    def clear(self):
        self.n = 0
        self.windows.clear()

    def push(self, x, date):
        
        x = fillna_npwhere_njit(x, self.past_value)
        self.past_value = x
        
        # change of day
        if date>self.day:
            self.day = date
            if self.day_instances > 0:
                self.past_mean = self.cum_sum/self.day_instances
            else:
                self.past_mean = 0
            self.day_instances = 1
            self.cum_sum = x
            
        else:
            self.day_instances += 1
            self.cum_sum += x

    def get_mean(self):
        return self.cum_sum/self.day_instances

    def get_past_mean(self):
        return self.past_mean


# A test seems to run pretty fast (10000 it/s)

# In[15]:


a = RunningPDA()

for index, row in tqdm(train_data[:100000].iterrows()): 
    date=row['date']
    a.push(np.array(row),date)


# In[16]:


a.get_past_mean()


# Which seems to match (weight 2.7143664 match the first value on the first row) :

# In[17]:


train_data[:200000].groupby('date').mean().iloc[30,]


# Great news !

# <a id='PTI'></a>
# # Previous trade information from the same underlying
# 
# As mentionned [here](https://www.kaggle.com/c/jane-street-market-prediction/discussion/207709) feature_41 being constant over the day allow to find the previous instance with the same feature_41 caracteristic. As we don't exactly know what are in those feature we can't really know how the trade opportunities relate exactly but I speculate that they relate to the same underlying or are pretty close and that information about the previous trade of the day relating to the same underlying might be usefull. At the moment the implementation rely on a simple dictionnary. I suspect I can't really get below keeping 800-900 instances in memory for that feature engineering technique as I need at least one example of each trade. It is not a problem and the method is really fast (10000 it/s).

# In[18]:


class RunningPTI:
    def __init__(self,base_value=0):
        self.dictionnary = {}
        self.base_value = base_value
        self.day = -1

    def clear(self):
        self.dictionnary = {}
        self.base_value = 0

    def push(self, x, value, date):
        
                # change of day
        if date>self.day:
            self.day = date
            self.dictionnary = {}
        
        self.past_value = self.dictionnary.get(value)
        self.dictionnary.update({value:x})
        
    def get_past_value(self):
        if self.past_value is None:
            self.past_value = self.base_value
        return self.past_value
    
    def get_dict(self):
        return self.dictionnary


# In[19]:


a = RunningPTI(base_value=np.nan * np.empty((1, 138)))

for index, row in tqdm(train_data[:10000].iterrows()): 
    f_41 = row['feature_41']
    date = row['date']
    a.push(np.array(row),f_41,date)
    
a.get_past_value()


# Note : I haven't toroughfully tested it yet.

# <a id='DIFF'></a>
# # Differentiation
# 
# Direct and second order differentation of averaged variables seems to have some importance, as is, change in overall trends have an importance for the problem of dealing with multiple securities. To be more clear :

# In[20]:


#rw = 10000
#train_data_diff = train_data.rolling(window=rw).mean().diff(rw)
#train_data_diff_diff = train_data.rolling(window=rw).mean().diff(rw).diff(rw)


# Seems to retain some importance when an xgboost is calibrated on them, especially for higher rw. I haven't built a running algo for them yet, but I'll probably try.

# <a id='FDIFF'></a>
# # Fractional differentiation
# 
# Very important feature engineering tool, especially for time series *. As illustrated by Marcos Lopez de Prado in his book Advances in Financial Machine Learning, It is a great tool to remove noise without removing information. 
# 
# The idea is to generalise differentiation to non integer. Applying multiple stationnarity tests while slowly increasing the fractionnal differentiation level, you can get an 'optimal' level (enough differentiation to remove noise, without removing information). 
# 
# However, I havea found that for optimal level of around -0.75 - that I found in the dataset ** - we would need longer series to make the calculation meaningful. This is illustrated in the example below : the weight for the millionth instance is still above 0.02 for the first one. This is also problematic as it means that the first instances would lack a lot of information.
# 
# \* : it is rather difficult to even use time series tools here as we have many underlying securities
# 
# **  : when the adf test would even converge after lenghty calculations

# In[21]:


def get_weights(d, size):
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


# In[22]:


plt.plot(get_weights(-0.75,1000000))
plt.ylim(0,0.1)


# So, unless someone points out a way to make fractionnal differentiation work in our data set (or that an approximation allows for some information gain), I don't really plan to make a streaming algorithm for building it.

# <a id='Entropy'></a>
# # Entropy rate
# 
# Also mentionned in lopez de prado's book. It relate to some physical mesure of order. I am still not entirely convinced this could work here. Especially because there is a lot of different notions of entropy and all of them are rather calculatory. For reference (outside of inference) I was able to calculate entropy on sliding windows with the pyinform package (see code below). But this is rather slow and doesn't seems to provide any information gain in my early tests.
# 
# However I have found [a streaming implementation](https://github.com/ajcr/rolling) that could be reimplemented for our problem so I am mentionning it.

# In[23]:


#!pip install pyinform
#from pyinform import entropy_rate

#entropy_r = lambda x: entropy_rate(x,k=2)

#df[feature] = (df[feature] > df[feature].mean())
#df[feature] = df[feature].rolling(window=rw[i]).apply(entropy_r)


# In[24]:


del train_data


# <a id='MLP'></a>
# # Application

# In this section I give you an idea of one would apply the algo for submission.
# The model used come from this [notebook](https://www.kaggle.com/aimind/bottleneck-encoder-mlp-keras-tuner-8601c5). I only show where feature engineering appears.

# Loading the packages :

# In[25]:


from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from tqdm import tqdm
from random import choices


import kerastuner as kt

physical_devices = tf.config.list_physical_devices('GPU')
try:
          tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
          # Invalid device or cannot modify virtual devices once initialized.
    pass


# PurgedTimeSeries CV

# In[26]:


import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args

# modified code for group gaps; source
# https://github.com/getgaurav2/scikit-learn/blob/d4a3af5cc9da3a76f0266932644b884c99724c57/sklearn/model_selection/_split.py#L2243
class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Allows for a gap in groups to avoid potentially leaking info from
    train into test if the model has windowed or lag features.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_group_size : int, default=Inf
        Maximum group size for a single training set.
    group_gap : int, default=None
        Gap between train and test
    max_test_group_size : int, default=Inf
        We discard this number of groups from the end of each train split
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_group_size=np.inf,
                 max_test_group_size=np.inf,
                 group_gap=None,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))

        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_test_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]
                
                train_array = np.sort(np.unique(
                                      np.concatenate((train_array,
                                                      train_array_tmp)),
                                      axis=None), axis=None)

            train_end = train_array.size
 
            for test_group_idx in unique_groups[group_test_start:
                                                group_test_start +
                                                group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                                              np.concatenate((test_array,
                                                              test_array_tmp)),
                                     axis=None), axis=None)

            test_array  = test_array[group_gap:]
            
            
            if self.verbose > 0:
                    pass
                    
            yield [int(i) for i in train_array], [int(i) for i in test_array]


# In[27]:


class CVTuner(kt.engine.tuner.Tuner):
    def run_trial(self, trial, X, y, splits, batch_size=32, epochs=1,callbacks=None):
        val_losses = []
        for train_indices, test_indices in splits:
            X_train, X_test = [x[train_indices] for x in X], [x[test_indices] for x in X]
            y_train, y_test = [a[train_indices] for a in y], [a[test_indices] for a in y]
            if len(X_train) < 2:
                X_train = X_train[0]
                X_test = X_test[0]
            if len(y_train) < 2:
                y_train = y_train[0]
                y_test = y_test[0]
            
            model = self.hypermodel.build(trial.hyperparameters)
            hist = model.fit(X_train,y_train,
                      validation_data=(X_test,y_test),
                      epochs=epochs,
                        batch_size=batch_size,
                      callbacks=callbacks)
            
            val_losses.append([hist.history[k][-1] for k in hist.history])
        val_losses = np.asarray(val_losses)
        self.oracle.update_trial(trial.trial_id, {k:np.mean(val_losses[:,i]) for i,k in enumerate(hist.history.keys())})
        self.save_model(trial.trial_id, model)


# In[28]:


# From https://medium.com/@micwurm/using-tensorflow-lite-to-speed-up-predictions-a3954886eb98

class LiteModel:
    
    @classmethod
    def from_file(cls, model_path):
        return LiteModel(tf.lite.Interpreter(model_path=model_path))
    
    @classmethod
    def from_keras_model(cls, kmodel):
        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
        tflite_model = converter.convert()
        return LiteModel(tf.lite.Interpreter(model_content=tflite_model))
    
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        input_det = self.interpreter.get_input_details()[0]
        output_det = self.interpreter.get_output_details()[0]
        self.input_index = input_det["index"]
        self.output_index = output_det["index"]
        self.input_shape = input_det["shape"]
        self.output_shape = output_det["shape"]
        self.input_dtype = input_det["dtype"]
        self.output_dtype = output_det["dtype"]
        
    def predict(self, inp):
        inp = inp.astype(self.input_dtype)
        count = inp.shape[0]
        out = np.zeros((count, self.output_shape[1]), dtype=self.output_dtype)
        for i in range(count):
            self.interpreter.set_tensor(self.input_index, inp[i:i+1])
            self.interpreter.invoke()
            out[i] = self.interpreter.get_tensor(self.output_index)[0]
        return out
    
    def predict_single(self, inp):
        """ Like predict(), but only for a single record. The input data can be a Python list. """
        inp = np.array([inp], dtype=self.input_dtype)
        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)
        return out[0]


# Loading the data :

# In[29]:


TRAINING = False
USE_FINETUNE = True     
FOLDS = 5
SEED = 42

train = pd.read_csv('../input/jane-street-market-prediction/train.csv')


# In[30]:


nb_trade = train.groupby('date')['date'].count()
high_volume_days = [i for i, x in enumerate(np.array(nb_trade > 7000)) if x]


# I change the days a bit to remove the day previous day 85 as the continuity of JS strategy is in question (or is that a different market regime ?)

# In[31]:


train = train.query('date > 85').reset_index(drop = True) 
train = train.query('date not in @high_volume_days').reset_index(drop = True) 
train = train.astype({c: np.float32 for c in train.select_dtypes(include='float64').columns}) #limit memory use
train.fillna(train.mean(),inplace=True)
train = train.query('weight > 0').reset_index(drop = True)
#train['action'] = (train['resp'] > 0).astype('int')
train['action'] =  (  (train['resp_1'] > 0.00001 ) & (train['resp_2'] > 0.00001 ) & (train['resp_3'] > 0.00001 ) & (train['resp_4'] > 0.00001 ) &  (train['resp'] > 0.00001 )   ).astype('int')


# Adding some feature (long term exponentially weighted mean of feature_0 and feature_1) :

# In[32]:


EWM_5000 = RunningEWMean(WIN_SIZE = 5000)
EWM_10000 = RunningEWMean(WIN_SIZE = 10000)
EWM_20000 = RunningEWMean(WIN_SIZE = 20000)

train_FE = []

for index, row in tqdm(train[['feature_0','feature_1']].iterrows()): 
    EWM_5000.push(np.float64(np.array(row)))
    EWM_10000.push(np.float64(np.array(row)))
    EWM_20000.push(np.float64(np.array(row)))

    FE = {
        'feature_0_EWM_5000' : EWM_5000.get_mean()[0],
        'feature_1_EWM_5000' : EWM_5000.get_mean()[1],
        'feature_0_EWM_10000' : EWM_10000.get_mean()[0],
        'feature_1_EWM_10000' : EWM_10000.get_mean()[1],
        'feature_0_EWM_20000' : EWM_20000.get_mean()[0],
        'feature_1_EWM_20000' : EWM_20000.get_mean()[1],
    }

    train_FE.append(FE)

train_FE = pd.DataFrame(train_FE)


# merge two dataframes and add columns

# In[33]:


train = pd.concat([train,train_FE],axis=1)

features = [c for c in train.columns if 'feature' in c]

resp_cols = ['resp_1', 'resp_2', 'resp_3', 'resp', 'resp_4']

X = train[features].values
y = np.stack([(train[c] > 0.000001).astype('int') for c in resp_cols]).T #Multitarget

f_mean = np.mean(train[features[1:]].values,axis=0)


# **Create autoencoder, MLP :**

# In[34]:


def create_autoencoder(input_dim,output_dim,noise=0.05):
    i = Input(input_dim)
    encoded = BatchNormalization()(i)
    encoded = GaussianNoise(noise)(encoded)
    encoded = Dense(640,activation='relu')(encoded)
    decoded = Dropout(0.2)(encoded)
    decoded = Dense(input_dim,name='decoded')(decoded)
    x = Dense(320,activation='relu')(decoded)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(output_dim,activation='sigmoid',name='label_output')(x)
    
    encoder = Model(inputs=i,outputs=encoded)
    autoencoder = Model(inputs=i,outputs=[decoded,x])
    
    autoencoder.compile(optimizer=Adam(0.001),loss={'decoded':'mse','label_output':'binary_crossentropy'})
    return autoencoder, encoder


# In[35]:


def create_model(hp,input_dim,output_dim,encoder):
    inputs = Input(input_dim)
    
    x = encoder(inputs)
    x = Concatenate()([x,inputs]) #use both raw and encoded features
    x = BatchNormalization()(x)
    x = Dropout(hp.Float('init_dropout',0.0,0.5))(x)
    
    for i in range(hp.Int('num_layers',1,5)):
        x = Dense(hp.Int('num_units_{i}',128,256))(x)
        x = BatchNormalization()(x)
        x = Lambda(tf.keras.activations.swish)(x)
        x = Dropout(hp.Float(f'dropout_{i}',0.0,0.5))(x)
    x = Dense(output_dim,activation='sigmoid')(x)
    model = Model(inputs=inputs,outputs=x)
    model.compile(optimizer=Adam(hp.Float('lr',0.00001,0.1,default=0.001)),loss=BinaryCrossentropy(label_smoothing=hp.Float('label_smoothing',0.0,0.1)),metrics=[tf.keras.metrics.AUC(name = 'auc')])
    return model


# In[36]:


autoencoder, encoder = create_autoencoder(X.shape[-1],y.shape[-1],noise=0.1)
if TRAINING:
    autoencoder.fit(X,(X,y),
                    epochs=1002,
                    batch_size=16384, 
                    validation_split=0.1,
                    callbacks=[EarlyStopping('val_loss',patience=10,restore_best_weights=True)])
    encoder.save_weights('./encoder.hdf5')
else:
    encoder.load_weights('../input/running-algos-fe-for-fast-inference/encoder.hdf5')
encoder.trainable = False


# Training the model :

# In[37]:


model_fn = lambda hp: create_model(hp,X.shape[-1],y.shape[-1],encoder)

tuner = CVTuner(
        hypermodel=model_fn,
        oracle=kt.oracles.BayesianOptimization(
        objective= kt.Objective('val_auc', direction='max'),
        num_initial_points=4,
        max_trials=60))

FOLDS = 5
SEED = 42
tf.random.set_seed(SEED)

if TRAINING:
    gkf = PurgedGroupTimeSeriesSplit(n_splits = FOLDS, group_gap=20)
    splits = list(gkf.split(y, groups=train['date'].values))
    tuner.search((X,),(y,),splits=splits,batch_size=16384,epochs=300,callbacks=[EarlyStopping('val_auc', mode='max',patience=3)])
    hp  = tuner.get_best_hyperparameters(1)[0]
    pd.to_pickle(hp,f'./best_hp_{SEED}.pkl')
    for fold, (train_indices, test_indices) in enumerate(splits):
        model = model_fn(hp)
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=300,batch_size=16384,callbacks=[EarlyStopping('val_auc',mode='max',patience=10,restore_best_weights=True)])
        model.save_weights(f'./model_{SEED}_{fold}.hdf5')
        model.compile(Adam(hp.get('lr')/100),loss='binary_crossentropy')
        model.fit(X_test,y_test,epochs=6,batch_size=16384)
        model.save_weights(f'./model_{SEED}_{fold}_finetune.hdf5')
    tuner.results_summary()
else:
    models = []
    hp = pd.read_pickle(f'../input/running-algos-fe-for-fast-inference/best_hp_{SEED}.pkl')
    for f in range(FOLDS):
        model = model_fn(hp)
        if USE_FINETUNE:
            model.load_weights(f'../input/running-algos-fe-for-fast-inference/model_{SEED}_{f}_finetune.hdf5')
        else:
            model.load_weights(f'../input/running-algos-fe-for-fast-inference/model_{SEED}_{f}.hdf5')
        model = LiteModel.from_keras_model(model)
        models.append(model)


# <a id='Dummy_Env'></a>
# ## Dummy environnement

# In[38]:


ENV_REAL = True

if (not TRAINING) & (not ENV_REAL):
    
    test_col = pd.read_pickle(f'../input/dummy-environnement/columns_df_test.pickle')

    n_row = 15219

    dummy_df = train.iloc[:n_row]

    for (index, row) in tqdm(dummy_df.iterrows()):

        time.sleep(0.009)
        test_df = pd.DataFrame(row).transpose()[test_col]

        test_df = pd.DataFrame(row).transpose()
        pred_df = pd.DataFrame(columns=['action'], index = [index])

        pred_df.action = 0


# <a id='Submission'></a>
# ## Submission

# In[39]:


if (not TRAINING) & (ENV_REAL):
    
    import janestreet
    env = janestreet.make_env()
    th = 0.5
    
    EWM_5000 = RunningEWMean(WIN_SIZE = 5000,n_size = 2)
    EWM_10000 = RunningEWMean(WIN_SIZE = 10000,n_size = 2)
    EWM_20000 = RunningEWMean(WIN_SIZE = 20000,n_size = 2)

    train_FE = []
    
    
    for (test_df, pred_df) in tqdm(env.iter_test()):
        
        EWM_5000.push(np.float64(np.array(test_df[['feature_0','feature_1']])))
        EWM_10000.push(np.float64(np.array(test_df[['feature_0','feature_1']])))
        EWM_20000.push(np.float64(np.array(test_df[['feature_0','feature_1']])))

        FE = []

        FE = {
            'feature_0_EWM_5000' : EWM_5000.get_mean()[0][0],
            'feature_1_EWM_5000' : EWM_5000.get_mean()[0][1],
            'feature_0_EWM_10000' : EWM_10000.get_mean()[0][0],
            'feature_1_EWM_10000' : EWM_10000.get_mean()[0][1],
            'feature_0_EWM_20000' : EWM_20000.get_mean()[0][0],
            'feature_1_EWM_20000' : EWM_20000.get_mean()[0][1],
        }

        test_df_FE = pd.concat([test_df,pd.DataFrame(FE, index=[test_df.index[0]])],axis=1)

        if test_df_FE['weight'].item() > 0:
            x_tt = test_df_FE.loc[:, features].values
            if np.isnan(x_tt[:, 1:].sum()):
                x_tt[:, 1:] = np.nan_to_num(x_tt[:, 1:]) + np.isnan(x_tt[:, 1:]) * f_mean
            pred = np.mean([model.predict(x_tt) for model in models],axis=0)
            pred = np.mean(pred)
            pred_df.action = np.where(pred >= th, 1, 0).astype(int)
        else:
            pred_df.action = 0
            
        env.predict(pred_df)

