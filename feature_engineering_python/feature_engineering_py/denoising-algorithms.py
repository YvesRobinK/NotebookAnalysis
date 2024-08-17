#!/usr/bin/env python
# coding: utf-8

# # Denoising algorithms
# 
# ## Background
# Basically any digital or analog process which produces a signal is vulnerable to noise. For example digital microphones get noise from random electron excitement (which occurs at any temperature above absolute 0), and analog film cameras get noise from differences in the sizes of the grains of the exposed film strip. When trying to quantify this variance there is additionally the [measurement error](https://en.wikipedia.org/wiki/Observational_error) of the instrument itself to content with; while the underlying .
# 
# An important preprocessing step for any signal processing task is denoising. There is a broad literature on denoising algorithms and techniques dating back centuries, and a number of fundamental techniques (like Fast Fourier Transforms) that have fallen out of from this.
# 
# In Python the principal library for performing signal processing tasks is the `signal` submodule in the `scipy` package. In this notebook I briefly go through the denoising algorithms available. This notebook is somewhat an elaboration on [xhlulu -- exploring signal processing with scipy](https://www.kaggle.com/xhlulu/exploring-signal-processing-with-scipy).

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


# In[ ]:


train = pd.read_parquet("../input/train.parquet")


# We will use the following demo data:

# In[ ]:


train.iloc[:, 0:3].plot(figsize=(12, 8))
plt.axis('off'); pass


# In[ ]:


df = train.iloc[:, 0:3]


# ## Rolling average (or other statistic)

# In[ ]:


df.rolling(100).mean().plot(figsize=(12, 8))
plt.axis('off'); pass


# The simplest algorithm for denoising time-series data is taking a summary statistic using a **rolling window**. A rolling window collects observations into groups of $n$ size. The groups are shifted one observation at a time, creating a "window" that passes over the dataset, hence the name. Each observation is part of $n-1$ groups, except for observations very near the beginning or the end, which appear in fewer (although you may choose to anneal them). Any summary statistic can be used within the rolling window to aggregate the data, though the average is the most popular default option.
# 
# Rolling windows have the following advantages:
# * They are computationally easy and simple to understand.
# * Varying $n$ can greatly increase or decrease the "smoothing factor".
# * They are flexible, you can use any algorithm you want as the aggregation function (mean, median, min, max, a quantile...).
# * They allow you to extend the algorithms you are already familiar with in record-oriented contexts to the time-series context without requiring any additional work.
# 
# Rolling windows have the following disadvantages:
# * They clip the beginnings and ends of the observations, thus reducing the total number of observations, which can represent a significant chunk of the dataset if the window is large and/or the data is small.
# * They cannot capture macro periodicity without removing micro periodicity.
# 
# The simplest rolling window technique is to apply an average on a neighborhood. However, you can apply any algorithm on the window, including a very complex one. `pandas` has very strong support for rolling window techniques built into the library (taken from `scipy`). You can define a more sophisticated window: e.g. an exponentially weighted window, which greatly reduces the contribution of points further off from the center of the neighborhood. For the most part more sophisticated windows mostly make the result more robust to outliers. For more details see [the docs](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.rolling.html).
# 
# One rolling window technique worth particular mention is the [median filter](https://en.wikipedia.org/wiki/Median_filter). When applied to an image a median filter has the effect of smoothing (color) noise in the image whilst retaining boundaries in the image. This is obviously useful in image processing because noisy images will have many spurious boundaries, bad, but smeared images will have weak boundaries, also bad.

# In[ ]:


df.rolling(100, win_type='gaussian').mean(std=df.std().mean()).plot(figsize=(12, 8))
plt.axis('off'); pass


# ## Convolution
# 
# Convolution is a mathematical operation which generates a new function (or pointwise distribution) that is a function of two prior functions (or pointwise distributions). Convolution is geometrically a graph of the area under the curve of two functions which are moved towards and then away from one another ("convolved"). For example, if the two functions are both squares, their convolution will be a triangle function beginning where the moving squares just begin to touch, maxing out where they overlap completely, and going back to zero where they just touch the other way. In the continuous case it is defined mathematically as:
# 
# $$(f * g)(t) = \int_{-\infty}^{\infty}f(x)g(x-\tau)d\tau$$
# 
# [The visualization](https://en.wikipedia.org/wiki/Convolution#/media/File:Convolution_of_box_signal_with_itself2.gif) from the Wikipedia article for it has a great intuitive explanation of how it works.
# 
# Any operation that can be expressed using a rolling window can be expressed using a convolution. For example, we can re-express a rolling mean with a window size of 10 by asking for the convolution of the function with a square window function (or "kernel") of length 10, divided by 10 (since area / length = average height):

# In[ ]:


# pd.Series(np.repeat([0., 1., 0.], 10)).plot()


# In[ ]:


def apply_convolution(sig, window):
    conv = np.repeat([0., 1., 0.], window)
    filtered = signal.convolve(sig, conv, mode='same') / window
    return filtered

df.apply(lambda srs: apply_convolution(srs, 100)).plot(figsize=(12, 8))
plt.axis('off'); pass


# If this equivalence doesn't jump out at you right away, a good reference on it is [this](https://matthew-brett.github.io/teaching/smoothing_intro.html) and [this](https://matthew-brett.github.io/teaching/smoothing_as_convolution.html). In general you shouldn't need to use convolution to perform smoothing; the rolling window methods are a better "form factor", but you may be performing convolution under the hood, as convolution is an efficient array-linearizable operation and generic rolling window functions are not.

# ## Filters
# 
# A large number of algorithms have been developed and published over the years for denoising signals. These algorithms are generally known as **filters**, or sometimes "digital filters" (to distinguish from electrical analog filters). Denoising is one of the fundamental tasks of signal processing, and it seems that different fiends have developed slightly different approaches, dependent slightly on the particularities of the kinds of problems they tend to run into.
# 
# For a list of filters available in `scipy` see the corresponding entry in the [documentation](https://scipy.github.io/devdocs/signal.html).
# 
# A lot of filters are based on rolling averages or convolutions under the hood because those are the core operations for operating on data. Good filters just make smart mathematical adaptations that, for some subset of tasks, outperforms simply taking the average of the window.
# 
# The next section describes a _relatively_ simple example of such a filter.

# ## Savitzky–Golay filter
# 
# B-splines or "basis splines" approximate data as a smooth sequence of piecewise exponential functions. The functions being _exponential_ means that they have terms of the form:
# 
# $$c_1x^n + c_2x^{n-1} + \ldots + c_n$$
# 
# The functions being _piecewise_ means that each one is only used for a particular subsequence of the data. For example $f(x)$ for $[0, 0.5]$ and $g(x)$ for $(0.5, 1]$. The points where functions intersect are known as _knot points_.
# 
# The last major requirement of a B-spline is that it must be smooth, e.g. continuous, e.g. it must have a derivative defined at every point. Geometrically this means that the area around the knots must be smooth, just like the spline is smooth on the body of the function. So the functions must join together in a way that preserves local topology.
# 
# Approximating a dataset using B-splines means solving for the exponential function which best fits the data within a local area between knots. You can use any general-purpose optimization algorithm to do this; [stochastic gradient descent](https://www.kaggle.com/residentmario/gradient-descent-with-linear-regression/) for example (see the ["Curve fitting"](https://en.wikipedia.org/wiki/B-spline#Curve_fitting) section of the Wikipedia article for a bit more detail). More complex functions can theoretically fit data within a region better, but require more work to optimize, as they have more parameters to tune. In practice cubic curves are sufficient for almost all applications.
# 
# Here is an example of a curve approximated using B-splines:
# 
# ![](https://i.imgur.com/gckO3IS.png)
# 
# B-splines and their slightly more advanced cousin the Bezier curve play an important role in computer graphics.
# 
# B-splines are more powerful and compact than rolling averages, as they generate a output function, not just a bunch of numbers.

# In[ ]:


# TODO: recipe based on https://scipy-cookbook.readthedocs.io/items/Interpolation.html
# I had difficulty getting this working because I did not know how to interpret the result
# of signal.cspline1d, which claims to generate coefficients, not values; but those 
# coefficients cannot be immediately mapped onto data values.

# coef = df.apply(lambda srs: signal.cspline1d(srs[::50].values, 10))
# df_approx = signal.cspline1d_eval(coef.iloc[:, 0].values.astype('float64'), df.index.values.astype('float64'))
# coef.head(100).apply(lambda t: t[0]*t.name**3 + t[1]*t.name**2 + t[2], axis='columns').plot()
# ^ goes to \infty


# The problem with using B-spline approximation directly (besides the fact that it really isn't all that well documented; I don't quite understand the application surface) is that determining where the knots should be is difficult. A good durable procedure for doing B-spline approximation was discovered in the 1960s and applied to the field of analytical chemistry (where it is still very dominant). It's called a Savitzky-Golay filter; you can read about it in [its Wikipedia article](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter).
# 
# Basically all it does is roll a window along the data, build a curve in the left-right neighborhood of the central point (e.g. if the window size is 9, there are 4 points to the left and 4 to the right), and then set the value to that curve's approximation of the value at the middle of the neighborhood. A good animation showing this in action is located [here](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter#/media/File:Lissage_sg3_anim.gif).

# In[ ]:


df.apply(lambda srs: signal.savgol_filter(srs.values, 99, 3)).plot(figsize=(12, 8))
plt.axis('off'); pass


# This algorithm is that it has an analytical solution, which makes it quick to calculate. But it is starting to be a bit difficult to grasp mathematically.

# ## Machine learning models
# 
# Complex filters take the idea of reducing the noise in a dataset far out into the realm of statistical modeling. However almost all of the relevant techniques accessible from e.g. `scipy` and to a regular programmer are relatively ancient, having been well-developed by the 1970s or thereabout. They predate the rise and rise of machine learning over the last few years.
# 
# If you think about it, modeling data is not that much different from denoising it. No useful model captures all of the variance in a dataset, so it learns to find the explainable variance and omit the incidental variance. Building a machine learning model on a dataset is not that different from denoising it. Therefore we can use a machine learning algorithm as an alternative to a filter, e.g. instead of applying a filter to denoise a dataset we can apply a machine learning model. This is a lot like pretraining in neural networks! For example, here is a simple model curve using the K nearest neighbors algorithm:

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor

clf = KNeighborsRegressor(n_neighbors=100, weights='uniform')
clf.fit(df.index.values[:, np.newaxis], 
        df.iloc[:, 0])
y_pred = clf.predict(df.index.values[:, np.newaxis])
ax = pd.Series(df.iloc[:, 0]).plot(color='lightgray')
pd.Series(y_pred).plot(color='black', ax=ax, figsize=(12, 8))


# The `KNeighborsRegressor` algorithm is just a rolling window along the dataset that sets the value of the model at every point to be the (potentially smoothed) average of the surrounding points. If you think about it, that's not all that different from the Savitzky–Golay filter from earlier, which does almost the same thing; the only difference is that Savitzky-Golay builds a cubic spline and uses that instead.
# 
# The biggest disadvantage of using machine learning algorithm to do smoothing like this is that machine learning algorithms (particularly parametric ones like kNN regression) are generally much slower than filters. Filters were designed in an era where compute was many orders of magnitude more expensive. They tend to make statistical assumptions about the underlying data which, although never entirely valid, leads to simple computations and fast implementations. Typical machine learning algorithms can take much longer to do the same work, but don't make assumptions about the data. Additionally they tend to be more familiar to the practioner.
# 
# 
# ## Addendum
# 
# As the notebook [VSB Power Line Faults EDA + Feature Engineering](https://www.kaggle.com/timothycwillard/vsb-power-line-faults-eda-feature-engineering) points out, likely the strongest signal in the VSB Power Line Fault Detection competition dataset is the level of noise in various segments of the dataset. Denoising algorithms are relevant to the dataset because you can back out the level of noise in the dataset by comparing the original dataset with the smoothed version. Try using that as a feature for this competition.
