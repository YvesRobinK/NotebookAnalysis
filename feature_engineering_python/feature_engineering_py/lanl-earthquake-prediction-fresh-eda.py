#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# ### In this kernel, I employ a new type of feature generation technique not used my most kernels in this competition.
# ### I will do EDA on new, fresh features and analyze the visualizations with detailed comments and descriptions.
# ### I will do training and inference with a stacked BiLSTM model (on these features) in a future version (you can try it out yourself)
# ### Thanks to Bruno Aquino, whose work this method is based on. Check out [this kernel](https://www.kaggle.com/braquino/5-fold-lstm-attention-fully-commented-0-694) of his. The entropy and fractal dimension functions are from Raphael Vallat's [entropy](https://github.com/raphaelvallat/entropy/blob/master/entropy) repository
# ### PLEASE UPVOTE IF YOU LIKE THIS KERNEL
# 

# <center><img src="https://i.imgur.com/hBPv3fh.png" width="750px"></center>

# ### Import necessary libraries

# In[1]:


import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from numba import jit
from math import log, floor
from sklearn.neighbors import KDTree
from scipy.signal import periodogram, welch

from keras.layers import *
from keras.models import *
from tqdm import tqdm
from sklearn.model_selection import train_test_split 
from keras import backend as K
from keras import optimizers
from sklearn.model_selection import GridSearchCV, KFold
from keras.callbacks import *
from keras import activations
from keras import regularizers
from keras import initializers
from keras import constraints
from keras.engine import Layer
from keras.engine import InputSpec
from keras.objectives import categorical_crossentropy
from keras.objectives import sparse_categorical_crossentropy
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import SVG

import warnings
warnings.filterwarnings('ignore')


# ### Initialize necessay constants

# In[2]:


SIGNAL_LEN = 150000
MIN_NUM = -27
MAX_NUM = 28


# ### Download seismic signal data along with targets (time left for occurance of laboratory earthquake)

# In[3]:


seismic_signals = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})


# ### Extract the acoustic data and targets from the dataframe
# Note : I delete the original dataframe to save memory

# In[4]:


acoustic_data = seismic_signals.acoustic_data
time_to_failure = seismic_signals.time_to_failure
data_len = len(seismic_signals)
del seismic_signals
gc.collect()


# ### Break the data down into parts
# We have one long array of seismic data. We will break it down into chunks of size 150k (**SIGNAL_LEN**) and each chunk will be one signal in our data (this is because each segment in the test data has length 150k). The **time_to_failure** at the last time step of each segment becomes the target associated with that segment.

# In[5]:


signals = []
targets = []

for i in range(data_len//SIGNAL_LEN):
    min_lim = SIGNAL_LEN * i
    max_lim = min([SIGNAL_LEN * (i + 1), data_len])
    
    signals.append(list(acoustic_data[min_lim : max_lim]))
    targets.append(time_to_failure[max_lim])
    
del acoustic_data
del time_to_failure
gc.collect()
    
signals = np.array(signals)
targets = np.array(targets)


# ## Functions for preparing signal features

# ### Scaling the signals
# This function scales the seismic signals from its original range (-27 to 28 : this where 99% of the data lies) to the range (-1, 1)

# In[6]:


def min_max_transfer(ts, min_value, max_value, range_needed=(-1,1)):
    ts_std = (ts - min_value) / (max_value - min_value)

    if range_needed[0] < 0:    
        return ts_std * (range_needed[1] + abs(range_needed[0])) + range_needed[0]
    else:
        return ts_std * (range_needed[1] - range_needed[0]) + range_needed[0]


# ### Extracting features from each part of the segment
# The original long seismic signal has already been broken down into several segments. The segments are scaled using the **min_max_transfer** function. Then, we break down each segment into several parts. Usual features such as mean, standard deviation, range, percentiles etc are calculated over each part of the segment and now, each part of the segment is represented by its own list of such features. Finally, the representations of all the small parts of the segment are strung together into a time series. This time series becomes the representation of that segment.

# In[7]:


def transform_ts(ts, n_dim=160, min_max=(-1,1)):
    ts_std = min_max_transfer(ts, min_value=MIN_NUM, max_value=MAX_NUM)
    bucket_size = int(SIGNAL_LEN / n_dim)
    new_ts = []
    for i in range(0, SIGNAL_LEN, bucket_size):
        ts_range = ts_std[i:i + bucket_size]
        mean = ts_range.mean()
        std = ts_range.std()
        std_top = mean + std
        std_bot = mean - std
        percentil_calc = ts_range.quantile([0, 0.01, 0.25, 0.50, 0.75, 0.99, 1])
        max_range = ts_range.quantile(1) - ts_range.quantile(0)
        relative_percentile = percentil_calc - mean
        new_ts.append(np.concatenate([np.asarray([mean, std, std_top, std_bot, max_range]), percentil_calc, relative_percentile]))
    return np.asarray(new_ts)


# ### Prepare the final signal features
# The time series representations of all segments of the signal are calulated and concatenated. This results in a 3D tensor containing the time series representations of all segments in the signal.

# In[8]:


def prepare_data(start, end):
    train = pd.DataFrame(np.transpose(signals[int(start):int(end)]))
    X = []
    for id_measurement in tqdm(train.index[int(start):int(end)]):
        X_signal = transform_ts(train[id_measurement])
        X.append(X_signal)
    X = np.asarray(X)
    return X


# ### Implement the feature generation process

# In[9]:


X = []

def load_all():
    total_size = len(signals)
    for start, end in [(0, int(total_size))]:
        X_temp = prepare_data(start, end)
        X.append(X_temp)
        
load_all()
X = np.concatenate(X)


# Here is the shape of X. There are a total of 4194 segments. Each segment is divided into 161 parts and each part is represented by a list of 19 features (mean, stddev etc). Therefore, X is a 3D tensor with shape (4194, 161, 19).

# In[10]:


X.shape


# ## Flattening the features and doing basic EDA with seaborn

# Now, we flatten the 2D tensors associated with each segment into 1D arrays. Now, each data point (segment) is represented by a 1D array.
# 
# Here are some flattened 1D arrays (with sparse selection) visualized with **matplotlib**.

# In[11]:


shape = X.shape
new_signals = X.reshape((shape[0], shape[1]*shape[2]))

sparse_signals = []
for i in range(3):
    sparse_signal = []
    for j in range(len(new_signals[i])):
        if j % 3 == 0:
            sparse_signal.append(new_signals[i][j])
    sparse_signals.append(sparse_signal)

plt.plot(sparse_signals[0], 'purple')
plt.show()
plt.plot(sparse_signals[1], 'mediumvioletred')
plt.show()
plt.plot(sparse_signals[2], 'crimson')
plt.show()


# ### Permutation Entropy
# The permutation entropy is a complexity measure for time-series first introduced by Bandt and Pompe in 2002. It represents the information contained
# in comparing *n* consecutive values of the time series. It is a measure of entropy or disorderliness in a time series.

# In[12]:


def _embed(x, order=3, delay=1):
    """Time-delay embedding.
    Parameters
    ----------
    x : 1d-array, shape (n_times)
        Time series
    order : int
        Embedding dimension (order)
    delay : int
        Delay.
    Returns
    -------
    embedded : ndarray, shape (n_times - (order - 1) * delay, order)
        Embedded time-series.
    """
    N = len(x)
    if order * delay > N:
        raise ValueError("Error: order * delay should be lower than x.size")
    if delay < 1:
        raise ValueError("Delay has to be at least 1.")
    if order < 2:
        raise ValueError("Order has to be at least 2.")
    Y = np.zeros((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T

all = ['perm_entropy', 'spectral_entropy', 'svd_entropy', 'app_entropy',
       'sample_entropy']


def perm_entropy(x, order=3, delay=1, normalize=False):
    """Permutation Entropy.
    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times)
    order : int
        Order of permutation entropy
    delay : int
        Time delay
    normalize : bool
        If True, divide by log2(order!) to normalize the entropy between 0
        and 1. Otherwise, return the permutation entropy in bit.
    Returns
    -------
    pe : float
        Permutation Entropy
    Notes
    -----
    The permutation entropy is a complexity measure for time-series first
    introduced by Bandt and Pompe in 2002 [1]_.
    The permutation entropy of a signal :math:`x` is defined as:
    .. math:: H = -\sum p(\pi)log_2(\pi)
    where the sum runs over all :math:`n!` permutations :math:`\pi` of order
    :math:`n`. This is the information contained in comparing :math:`n`
    consecutive values of the time series. It is clear that
    :math:`0 ≤ H (n) ≤ log_2(n!)` where the lower bound is attained for an
    increasing or decreasing sequence of values, and the upper bound for a
    completely random system where all :math:`n!` possible permutations appear
    with the same probability.
    The embedded matrix :math:`Y` is created by:
    .. math:: y(i)=[x_i,x_{i+delay}, ...,x_{i+(order-1) * delay}]
    .. math:: Y=[y(1),y(2),...,y(N-(order-1))*delay)]^T
    References
    ----------
    .. [1] Bandt, Christoph, and Bernd Pompe. "Permutation entropy: a
           natural complexity measure for time series." Physical review letters
           88.17 (2002): 174102.
    Examples
    --------
    1. Permutation entropy with order 2
        >>> from entropy import perm_entropy
        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value in bit between 0 and log2(factorial(order))
        >>> print(perm_entropy(x, order=2))
            0.918
    2. Normalized permutation entropy with order 3
        >>> from entropy import perm_entropy
        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value comprised between 0 and 1.
        >>> print(perm_entropy(x, order=3, normalize=True))
            0.589
    """
    x = np.array(x)
    ran_order = range(order)
    hashmult = np.power(order, ran_order)
    # Embed x and sort the order of permutations
    sorted_idx = _embed(x, order=order, delay=delay).argsort(kind='quicksort')
    # Associate unique integer to each permutations
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    # Return the counts
    _, c = np.unique(hashval, return_counts=True)
    # Use np.true_divide for Python 2 compatibility
    p = np.true_divide(c, c.sum())
    pe = -np.multiply(p, np.log2(p)).sum()
    if normalize:
        pe /= np.log2(factorial(order))
    return pe


# In[13]:


perm_entropies = np.array([perm_entropy(new_signal) for new_signal in new_signals])


# #### Bivariate KDE distribution plot

# In[14]:


plot = sns.jointplot(x=perm_entropies, y=targets, kind='kde', color='orangered')
plot.set_axis_labels('perm_entropy', 'time_to_failure', fontsize=16)
plt.show()


# The KDE plot has highest density (darkness) along a line with positive slope.

# #### Bivariate hexplot

# In[15]:


plot = sns.jointplot(x=perm_entropies, y=targets, kind='hex', color='orangered')
plot.set_axis_labels('perm_entropy', 'time_to_failure', fontsize=16)
plt.show()


# The hexplot is also darkest around a positively-sloped line.

# #### Scatterplot with line of best fit

# In[16]:


plot = sns.jointplot(x=perm_entropies, y=targets, kind='reg', color='orangered')
plot.set_axis_labels('perm_entropy', 'time_to_failure', fontsize=16)
plt.show()


# The line of best fit in the scatterplot has a clear positive slope.

# From the above three plots we can see a somewhat **positive correlation** between the permutation entropy of the flattened feature array and the time left for the laboratory earthquake to occur.

# ### Approximate Entropy
# Approximate entropy is a technique used to quantify the amount of
# regularity and the unpredictability of fluctuations over time-series data.
# Smaller values indicates that the data is more regular and predictable.

# In[17]:


def _app_samp_entropy(x, order, metric='chebyshev', approximate=True):
    """Utility function for `app_entropy`` and `sample_entropy`.
    """
    _all_metrics = KDTree.valid_metrics
    if metric not in _all_metrics:
        raise ValueError('The given metric (%s) is not valid. The valid '
                         'metric names are: %s' % (metric, _all_metrics))
    phi = np.zeros(2)
    r = 0.2 * np.std(x, axis=-1, ddof=1)

    # compute phi(order, r)
    _emb_data1 = _embed(x, order, 1)
    if approximate:
        emb_data1 = _emb_data1
    else:
        emb_data1 = _emb_data1[:-1]
    count1 = KDTree(emb_data1, metric=metric).query_radius(emb_data1, r,
                                                           count_only=True
                                                           ).astype(np.float64)
    # compute phi(order + 1, r)
    emb_data2 = _embed(x, order + 1, 1)
    count2 = KDTree(emb_data2, metric=metric).query_radius(emb_data2, r,
                                                           count_only=True
                                                           ).astype(np.float64)
    if approximate:
        phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))
        phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))
    else:
        phi[0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
        phi[1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))
    return phi


@jit('f8(f8[:], i4, f8)', nopython=True)
def _numba_sampen(x, mm=2, r=0.2):
    """
    Fast evaluation of the sample entropy using Numba.
    """
    n = x.size
    n1 = n - 1
    mm += 1
    mm_dbld = 2 * mm

    # Define threshold
    r *= x.std()

    # initialize the lists
    run = [0] * n
    run1 = run[:]
    r1 = [0] * (n * mm_dbld)
    a = [0] * mm
    b = a[:]
    p = a[:]

    for i in range(n1):
        nj = n1 - i

        for jj in range(nj):
            j = jj + i + 1
            if abs(x[j] - x[i]) < r:
                run[jj] = run1[jj] + 1
                m1 = mm if mm < run[jj] else run[jj]
                for m in range(m1):
                    a[m] += 1
                    if j < n1:
                        b[m] += 1
            else:
                run[jj] = 0
        for j in range(mm_dbld):
            run1[j] = run[j]
            r1[i + n * j] = run[j]
        if nj > mm_dbld - 1:
            for j in range(mm_dbld, nj):
                run1[j] = run[j]

    m = mm - 1

    while m > 0:
        b[m] = b[m - 1]
        m -= 1

    b[0] = n * n1 / 2
    a = np.array([float(aa) for aa in a])
    b = np.array([float(bb) for bb in b])
    p = np.true_divide(a, b)
    return -log(p[-1])


def app_entropy(x, order=2, metric='chebyshev'):
    """Approximate Entropy.
    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times)
    order : int (default: 2)
        Embedding dimension.
    metric : str (default: chebyshev)
        Name of the metric function used with
        :class:`~sklearn.neighbors.KDTree`. The list of available
        metric functions is given by: ``KDTree.valid_metrics``.
    Returns
    -------
    ae : float
        Approximate Entropy.
    Notes
    -----
    Original code from the mne-features package.
    Approximate entropy is a technique used to quantify the amount of
    regularity and the unpredictability of fluctuations over time-series data.
    Smaller values indicates that the data is more regular and predictable.
    The value of :math:`r` is set to :math:`0.2 * std(x)`.
    Code adapted from the mne-features package by Jean-Baptiste Schiratti
    and Alexandre Gramfort.
    References
    ----------
    .. [1] Richman, J. S. et al. (2000). Physiological time-series analysis
           using approximate entropy and sample entropy. American Journal of
           Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.
    1. Approximate entropy with order 2.
        >>> from entropy import app_entropy
        >>> import numpy as np
        >>> np.random.seed(1234567)
        >>> x = np.random.rand(3000)
        >>> print(app_entropy(x, order=2))
            2.075
    """
    phi = _app_samp_entropy(x, order=order, metric=metric, approximate=True)
    return np.subtract(phi[0], phi[1])


# In[18]:


app_entropies = np.array([app_entropy(new_signal) for new_signal in new_signals])


# #### Bivariate KDE distribution plot

# In[19]:


plot = sns.jointplot(x=app_entropies, y=targets, kind='kde', color='magenta')
plot.set_axis_labels('app_entropy', 'time_to_failure', fontsize=16)
plt.show()


# The KDE plot has highest density (darkness) around a line with negative slope.

# #### Bivariate hexplot

# In[20]:


plot = sns.jointplot(x=app_entropies, y=targets, kind='hex', color='magenta')
plot.set_axis_labels('app_entropy', 'time_to_failure', fontsize=16)
plt.show()


# The hexplot is also darkest around a negatively-sloped line.

# #### Scatterplot with line of best fit

# In[21]:


plot = sns.jointplot(x=app_entropies, y=targets, kind='reg', color='magenta')
plot.set_axis_labels('app_entropy', 'time_to_failure', fontsize=16)
plt.show()


# The line of best fit in the scatterplot has a clear negative slope.

# From the above three plots we can see a somewhat **negative correlation** between the approximate entropy of the flattened feature array and the time left for the laboratory earthquake to occur.

# ### Higuchi Fractal Dimension
# The Higuchi fractal dimension is a method to calculate the [fractal dimension](https://en.wikipedia.org/wiki/Fractal_dimension) of any two-dimensional curve. Generally, curves with higher fractal dimension are "rougher" or more "complex" (higher entropy).

# In[22]:


@jit('i8[:](f8, f8, f8)', nopython=True)
def _log_n(min_n, max_n, factor):
    """
    Creates a list of integer values by successively multiplying a minimum
    value min_n by a factor > 1 until a maximum value max_n is reached.
    Used for detrended fluctuation analysis (DFA).
    Function taken from the nolds python package
    (https://github.com/CSchoel/nolds) by Christopher Scholzel.
    Parameters
    ----------
    min_n (float):
        minimum value (must be < max_n)
    max_n (float):
        maximum value (must be > min_n)
    factor (float):
       factor used to increase min_n (must be > 1)
    Returns
    -------
    list of integers:
        min_n, min_n * factor, min_n * factor^2, ... min_n * factor^i < max_n
        without duplicates
    """
    max_i = int(floor(log(1.0 * max_n / min_n) / log(factor)))
    ns = [min_n]
    for i in range(max_i + 1):
        n = int(floor(min_n * (factor ** i)))
        if n > ns[-1]:
            ns.append(n)
    return np.array(ns, dtype=np.int64)

@jit('float64(float64[:], int32)')
def _higuchi_fd(x, kmax):
    """Utility function for `higuchi_fd`.
    """
    n_times = x.size
    lk = np.empty(kmax)
    x_reg = np.empty(kmax)
    y_reg = np.empty(kmax)
    for k in range(1, kmax + 1):
        lm = np.empty((k,))
        for m in range(k):
            ll = 0
            n_max = floor((n_times - m - 1) / k)
            n_max = int(n_max)
            for j in range(1, n_max):
                ll += abs(x[m + j * k] - x[m + (j - 1) * k])
            ll /= k
            ll *= (n_times - 1) / (k * n_max)
            lm[m] = ll
        # Mean of lm
        m_lm = 0
        for m in range(k):
            m_lm += lm[m]
        m_lm /= k
        lk[k - 1] = m_lm
        x_reg[k - 1] = log(1. / k)
        y_reg[k - 1] = log(m_lm)
    higuchi, _ = _linear_regression(x_reg, y_reg)
    return higuchi


def higuchi_fd(x, kmax=10):
    """Higuchi Fractal Dimension.
    Parameters
    ----------
    x : list or np.array
        One dimensional time series
    kmax : int
        Maximum delay/offset (in number of samples).
    Returns
    -------
    hfd : float
        Higuchi Fractal Dimension
    Notes
    -----
    Original code from the mne-features package by Jean-Baptiste Schiratti
    and Alexandre Gramfort.
    The `higuchi_fd` function uses Numba to speed up the computation.
    References
    ----------
    .. [1] Higuchi, Tomoyuki. "Approach to an irregular time series on the
       basis of the fractal theory." Physica D: Nonlinear Phenomena 31.2
       (1988): 277-283.
    Examples
    --------
    Higuchi Fractal Dimension
        >>> import numpy as np
        >>> from entropy import higuchi_fd
        >>> np.random.seed(123)
        >>> x = np.random.rand(100)
        >>> print(higuchi_fd(x))
            2.051179
    """
    x = np.asarray(x, dtype=np.float64)
    kmax = int(kmax)
    return _higuchi_fd(x, kmax)

@jit('UniTuple(float64, 2)(float64[:], float64[:])', nopython=True)
def _linear_regression(x, y):
    """Fast linear regression using Numba.
    Parameters
    ----------
    x, y : ndarray, shape (n_times,)
        Variables
    Returns
    -------
    slope : float
        Slope of 1D least-square regression.
    intercept : float
        Intercept
    """
    n_times = x.size
    sx2 = 0
    sx = 0
    sy = 0
    sxy = 0
    for j in range(n_times):
        sx2 += x[j] ** 2
        sx += x[j]
        sxy += x[j] * y[j]
        sy += y[j]
    den = n_times * sx2 - (sx ** 2)
    num = n_times * sxy - sx * sy
    slope = num / den
    intercept = np.mean(y) - slope * np.mean(x)
    return slope, intercept


# In[23]:


higuchi_fds = np.array([higuchi_fd(new_signal) for new_signal in new_signals])


# #### Bivariate KDE distribution plot

# In[24]:


plot = sns.jointplot(x=higuchi_fds, y=targets, kind='kde', color='crimson')
plot.set_axis_labels('higuchi_fd', 'time_to_failure', fontsize=16)
plt.show()


# The KDE plot has highest density (darkness) around a line with negative slope.

# #### Bivariate hexplot

# In[25]:


plot = sns.jointplot(x=higuchi_fds, y=targets, kind='hex', color='crimson')
plot.set_axis_labels('higuchi_fd', 'time_to_failure', fontsize=16)
plt.show()


# The hexplot is also darkest around a negatively-sloped line.

# In[26]:


plot = sns.jointplot(x=higuchi_fds, y=targets, kind='reg', color='crimson')
plot.set_axis_labels('higuchi_fd', 'time_to_failure', fontsize=16)
plt.show()


# The line of best fit in the scatterplot has a clear negative slope.

# From the above three plots we can see a somewhat **negative correlation** between the Higuchi fractal dimension of the flattened feature array and the time left for the laboratory earthquake to occur.

# ### Katz Fractal Dimension
# The Katz fractal dimension is yet another way to calculate the fractal dimension of a two-dimensional curve.

# In[27]:


def katz_fd(x):
    """Katz Fractal Dimension.
    Parameters
    ----------
    x : list or np.array
        One dimensional time series
    Returns
    -------
    kfd : float
        Katz fractal dimension
    Notes
    -----
    The Katz Fractal dimension is defined by:
    .. math:: FD_{Katz} = \dfrac{log_{10}(n)}{log_{10}(d/L)+log_{10}(n)}
    where :math:`L` is the total length of the time series and :math:`d`
    is the Euclidean distance between the first point in the
    series and the point that provides the furthest distance
    with respect to the first point.
    Original code from the mne-features package by Jean-Baptiste Schiratti
    and Alexandre Gramfort.
    References
    ----------
    .. [1] Esteller, R. et al. (2001). A comparison of waveform fractal
           dimension algorithms. IEEE Transactions on Circuits and Systems I:
           Fundamental Theory and Applications, 48(2), 177-183.
    .. [2] Goh, Cindy, et al. "Comparison of fractal dimension algorithms for
           the computation of EEG biomarkers for dementia." 2nd International
           Conference on Computational Intelligence in Medicine and Healthcare
           (CIMED2005). 2005.
    Examples
    --------
    Katz fractal dimension.
        >>> import numpy as np
        >>> from entropy import katz_fd
        >>> np.random.seed(123)
        >>> x = np.random.rand(100)
        >>> print(katz_fd(x))
            5.1214
    """
    x = np.array(x)
    dists = np.abs(np.ediff1d(x))
    ll = dists.sum()
    ln = np.log10(np.divide(ll, dists.mean()))
    aux_d = x - x[0]
    d = np.max(np.abs(aux_d[1:]))
    return np.divide(ln, np.add(ln, np.log10(np.divide(d, ll))))


# In[28]:


katz_fds = np.array([katz_fd(new_signal) for new_signal in new_signals])


# In[29]:


plot = sns.jointplot(x=katz_fds, y=targets, kind='kde', color='forestgreen')
plot.set_axis_labels('katz_fd', 'time_to_failure', fontsize=16)
plt.show()


# The KDE plot has highest density (darkness) around an almost vertical line.

# In[30]:


plot = sns.jointplot(x=katz_fds, y=targets, kind='hex', color='forestgreen')
plot.set_axis_labels('katz_fd', 'time_to_failure', fontsize=16)
plt.show()


# The hexplot also has highest density around an almost vertical line.

# In[31]:


plot = sns.jointplot(x=katz_fds, y=targets, kind='reg', color='forestgreen')
plot.set_axis_labels('katz_fd', 'time_to_failure', fontsize=16)
plt.show()


# The line of best fit has a very slight positive correlation.

# From the above three plots we can see an **very slight positive correlation (or maybe no correlation)** between the Katz fractal dimension of the flattened feature array and the time left for the laboratory earthquake to occur.

# ## Example model to use for these features

# ### Attention layer
# Attention mechanisms in neural networks serve to orient perception as well as memory access (you might even say perception is just a very short-term subset of all memory). Attention filters the perceptions that can be stored in memory, and filters them again on a second pass when they are to be retrieved from memory. Attention can be aimed at the present and the past.

# In[32]:


# https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-652-lb

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


# ### Example neural network architecture (stacked BiLSTM with Attention)
# 
# #### Create a model that :
# 
# 1). Passes the 3D input through a Bidirectional LSTM layer with 128 units
# 
# 2). Passes the output of the previous layer through another Bidirectional LSTM layer with 64 units
# 
# 3). Then passes the result through an Attention layer
# 
# 4). The Attention layer's is passed through a Dense layer with 64 units
# 
# 5). Finally, the output is obtained by passing the previous layer's output through a Dense layer with one neuron

# In[33]:


def model_lstm(input_shape):
    inp = Input(shape=(input_shape[1], input_shape[2],))

    bi_lstm = Bidirectional(LSTM(128, return_sequences=True))(inp)
    bi_gru = Bidirectional(LSTM(64, return_sequences=True))(bi_lstm)
    
    attention = Attention(input_shape[1])(bi_gru)
    
    x = Dense(64, activation="relu")(attention)
    x = Dense(1)(x)
    
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='mae', optimizer='adam')
    
    return model


# In[34]:


sample_model = model_lstm(X.shape)


# ### Model summary

# In[35]:


sample_model.summary()


# ### Model plot visualization 

# In[36]:


SVG(model_to_dot(sample_model).create(prog='dot', format='svg'))


# ### I will do training and inference with this model in a future version (you can try it out yourself)

# ### That's it ! Thanks for reading this kernel :)

# ### Please post your feedback and suggestions in the comments.
