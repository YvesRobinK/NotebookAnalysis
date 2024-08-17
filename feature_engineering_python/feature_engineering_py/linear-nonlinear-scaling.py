#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <hr style="border: solid 3px blue;">
# 
# # Introduction
# 
# ![](http://49.media.tumblr.com/5944bcf4f7fe9c0c99f7a593f233731a/tumblr_mj7bx09MDo1s5nl47o2_r1_500.gif) 
# 
# Picture Credit: http://49.media.tumblr.com

# **Numerical variables can basically use the model input as it is, but with appropriate transformation or processing, more effective features can be created.**
# 
# In this notebook, various linear and non-linear scalings will be summarized.

# -----------------------------------------------
# ## Why do we need linear scaling?
# > Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without normalization. For example, many classifiers calculate the distance between two points by the Euclidean distance. If one of the features has a broad range of values, the distance will be governed by this particular feature. Therefore, the range of all features should be normalized so that each feature contributes approximately proportionately to the final distance.
# > 
# > Another reason why feature scaling is applied is that gradient descent converges much faster with feature scaling than without it
# > 
# > It's also important to apply feature scaling if regularization is used as part of the loss function (so that coefficients are penalized appropriately).
# 
# Ref: https://en.wikipedia.org/wiki/Feature_scaling
# 
# Linear scaling does not change the shape of the distribution. However, it is important in a model where the distance of features is important. In addition, it is an important process that is almost essential to use in neural networks where parameters are changed in the manner of gradient descent.

# -------------------------------------------------------------
# ## Why do we need non-linear scaling?
# 
# Non-linear scaling changes the distribution of features. If certain features are skewed to one side, non-linear scaling should be considered. For example, let's assume that certain features are clustered in small values. In this case, it is recommended to increase the interval between small values of specific features and decrease the interval between large values through scaling such as log scaling. Through this process, if you make a distribution that is as similar to the normal distribution as possible, you will be able to learn the features evenly while the model trains.
# 
# We can draw a Q-Q plot to visually check the normality after non-linear scaling.

# ## Q-Q plot
# 
# ![](https://miro.medium.com/max/1024/1*_wuWDNGs3hB2K0_kgpoc1A.jpeg)
# 
# Picture Credit: https://miro.medium.com
# 
# > In statistics, a Q–Q (quantile-quantile) plot is a probability plot, which is a graphical method for comparing two probability distributions by plotting their quantiles against each other.First, the set of intervals for the quantiles is chosen. A point (x, y) on the plot corresponds to one of the quantiles of the second distribution (y-coordinate) plotted against the same quantile of the first distribution (x-coordinate). Thus the line is a parametric curve with the parameter which is the number of the interval for the quantile.
# 
# Ref: https://en.wikipedia.org/wiki/

# ------------------------------------------------
# # Setting Up

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scipy.stats as stats

import warnings
warnings.filterwarnings(action='ignore')


# In[3]:


train_x = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
train_org = train_x.copy()


# In[4]:


train_x.loc[:,['SalePrice','LotArea']].describe()


# For explanation, skewed variables are selected.

# In[5]:


num_cols = ['SalePrice','LotArea']


# In[6]:


def display_stat():
    for i in range(2):
        mean = train_x[num_cols[i]].mean()
        std = train_x[num_cols[i]].std()
        skew = train_x[num_cols[i]].skew()
        kurtosis = train_x[num_cols[i]].kurtosis()
        print(num_cols[i]+':')
        print('mean: {0:.4f}, std: {1:.4f}, skew: {2:.4f}, kurtosis: {3:.4f} '.format(mean, std, skew, kurtosis))


# ----------------------------------------------------------------
# # Checking Orignal Distribution

# In[7]:


rcParams['figure.figsize'] = 25,30
sns.set(font_scale = 1.5)
sns.set_style("white")
plt.subplots_adjust(hspace=1)
fig, axes = plt.subplots(3, 2)
for i in range(2):
    sns.distplot(train_x[num_cols[i]],ax = axes[0,i],rug=True,color='green')
    sns.boxplot(train_x[num_cols[i]],ax = axes[1,i],color='green')  
    stats.probplot(train_x[num_cols[i]],plot = axes[2,i])
    sns.despine()


# #### Checking Statistics

# In[8]:


display_stat()


# <span style="color:Blue"> Observation:
# * SalePrice is slightly skewed.
# * LotArea is highly skewed, kurtosis is large and distribution is sharp.

# #### Checking Scatter

# In[9]:


sns.set(font_scale = 1.5)
sns.set_style("white")
sns.set_palette("bright")
rcParams['figure.figsize'] = 7,7
ax1 = plt.subplot(1,1,1)
sns.scatterplot(data=train_org, y="SalePrice", x="LotArea",ax=ax1,color='green')
ax1.set_title('Orginal',fontsize=30)


# Looking at the above figures, it can be seen that the orginal data is skewed to one side. Let's check how the data is changed through each transformation.

# --------------------------------------------
# # Linear Scaling
# 
# ![](https://miro.medium.com/max/1400/1*yR54MSI1jjnf2QeGtt57PA.png)
# 
# Picture Credit: https://miro.medium.com
# 
# **Linear Scaling does not change the shape, but the scale of the variable is changed.**

# --------------------------------------
# ### StandardScaler
# Remove the mean and adjust the data to unit variance. However, if there are outliers, the spread of the transformed data becomes very different by affecting the mean and standard deviation.
# 
# Therefore, a balanced scale cannot be guaranteed if there are outliers.

# In[10]:


from sklearn.preprocessing import StandardScaler
train_x = train_org.copy()
rcParams['figure.figsize'] = 24,40
fig, axes = plt.subplots(4, 2)
sns.set_style("white")
sns.set_palette("bright")
plt.subplots_adjust(hspace=0.35)
for i in range(2):
    scaler = StandardScaler()
    train_x.loc[:,num_cols[i]] = scaler.fit_transform(train_x.loc[:,[num_cols[i]]])
    sns.distplot(train_org.loc[:,num_cols[i]],ax=axes[0,i],rug=True,color='blue')
    axes[0,i].set_title('Orginal '+num_cols[i],fontsize=25)
    sns.distplot(train_x.loc[:,num_cols[i]],ax=axes[1,i],rug=True,color='green')
    axes[1,i].set_title('Standard Scaling '+num_cols[i],fontsize=25)
    sns.boxplot(train_x.loc[:,num_cols[i]],ax=axes[2,i],color='blue')
    axes[2,i].set_title('Standard Scaling '+num_cols[i],fontsize=25)
    stats.probplot(train_x[num_cols[i]],plot = axes[3,i])
    sns.despine()


# #### Checking Statistics

# In[11]:


display_stat()


# <span style="color:Blue"> Observation:
# 
# * The distribution was changed to the standard normal distribution.
# * Skewness and kurtosis did not change from the original distribution.

# #### Checking Scatter

# In[12]:


sns.set(font_scale = 1.5)
sns.set_style("white")
sns.set_palette("bright")
rcParams['figure.figsize'] = 24,10
ax1 = plt.subplot(1,2,1)  
ax2 = plt.subplot(1,2,2)  
sns.scatterplot(data=train_org, y="SalePrice", x="LotArea",ax=ax1,color='green')
ax1.set_title('Orginal',fontsize=25)
sns.scatterplot(data =train_x, y="SalePrice", x="LotArea",ax=ax2)
ax2.set_title('After Standard Scale',fontsize=25)
sns.despine()


# Looking at the above figures, it can be seen that the shape of the distribution is not changed and has been changed to the standard scale.

# --------------------------------
# ### MinMaxScaler
# 
# Rescale the data so that all feature values are between 0 and 1. However, if there is an outlier, the transformed value may be compressed into a very narrow range.
# 
# In other words, MinMaxScaler is also very sensitive to the existence of outliers.

# In[13]:


from sklearn.preprocessing import MinMaxScaler
train_x = train_org.copy()
rcParams['figure.figsize'] = 24,40
fig, axes = plt.subplots(4, 2)
sns.set_style("white")
sns.set_palette("bright")
plt.subplots_adjust(hspace=0.35)

scaler = MinMaxScaler()
for i in range(2):
    train_x.loc[:,num_cols[i]] = scaler.fit_transform(train_x[[num_cols[i]]])
    sns.distplot(train_org.loc[:,num_cols[i]],ax=axes[0,i],rug=True,color='green')
    axes[0,i].set_title('Orginal '+num_cols[i],fontsize=25)
    sns.distplot(train_x.loc[:,num_cols[i]],ax=axes[1,i],rug=True,color='blue')
    axes[1,i].set_title('Standard Scaling '+num_cols[i],fontsize=25)
    sns.boxplot(train_x.loc[:,num_cols[i]],ax=axes[2,i],color='blue')
    axes[2,i].set_title('Standard Scaling '+num_cols[i],fontsize=25)
    stats.probplot(train_x[num_cols[i]],plot = axes[3,i])
    sns.despine()


# #### Checking Statistics

# In[14]:


display_stat()


# <span style="color:Blue"> Observation:
# 
# * The distribution is mapped between 0 and 1 while maintaining the shape of the original distribution.
# * Skewness and kurtosis did not change from the original distribution.

# #### Checking Scatter

# In[15]:


sns.set(font_scale = 1.5)
sns.set_style("white")
sns.set_palette("bright")
rcParams['figure.figsize'] = 25,10
ax1 = plt.subplot(1,2,1)  
ax2 = plt.subplot(1,2,2)  
sns.scatterplot(data=train_org, y="SalePrice", x="LotArea",ax=ax1,color='green')
ax1.set_title('Orginal',fontsize=25)
sns.scatterplot(data =train_x, y="SalePrice", x="LotArea",ax=ax2)
ax2.set_title('After MinMax Scaler',fontsize=25)
sns.despine()


# Looking at the above figures, it can be seen that the shape of the distribution is not changed and the data distribution is rescaled between 0 and 1.

# ----------------------------------------------------------
# ### RobustScaler
# 
# This is a technique that minimizes the influence of outliers. 
# 
# Since the median and IQR (interquartile range) are used, it can be confirmed that the same values ​​are more widely distributed after standardization when compared with the StandardScaler.
# 
# $𝐼𝑄𝑅=𝑄3−𝑄1$: That is, it deals with values in the 25th and 75th percentiles.
# 
# > If your data contains many outliers, scaling using the mean and variance of the data is likely to not work very well. In these cases, you can use RobustScaler as a drop-in replacement instead. It uses more robust estimates for the center and range of your data.
# 
# Ref: https://scikit-learn.org/stable

# In[16]:


from sklearn.preprocessing import RobustScaler
train_x = train_org.copy()

rcParams['figure.figsize'] = 25,40
fig, axes = plt.subplots(4, 2)
sns.set_style("white")
sns.set_palette("bright")
plt.subplots_adjust(hspace=0.35)

robuster = RobustScaler()
for i in range(2):
    train_x.loc[:,num_cols[i]] = robuster.fit_transform(train_x[[num_cols[i]]])
    sns.distplot(train_org.loc[:,num_cols[i]],ax=axes[0,i],rug=True,color='green')
    axes[0,i].set_title('Orginal '+num_cols[i],fontsize=25)
    sns.distplot(train_x.loc[:,num_cols[i]],ax=axes[1,i],rug=True,color='blue')
    axes[1,i].set_title('Robust Scaling '+num_cols[i],fontsize=25)
    sns.boxplot(train_x.loc[:,num_cols[i]],ax=axes[2,i],color='blue')
    axes[2,i].set_title('Robust Scaling '+num_cols[i],fontsize=25)
    stats.probplot(train_x[num_cols[i]],plot = axes[3,i])
    sns.despine()


# #### Checking Statistics

# In[17]:


display_stat()


# <span style="color:Blue"> Observation:
# 
# * The mean and variance were changed, but the shape of the distribution did not. That is, only scaling was changed linearly.
# * Skewness and kurtosis did not change from the original distribution.

# #### Checking Scatter

# In[18]:


sns.set(font_scale = 1.5)
sns.set_style("white")
sns.set_palette("bright")
rcParams['figure.figsize'] = 25,10
ax1 = plt.subplot(1,2,1)  
ax2 = plt.subplot(1,2,2)  
sns.scatterplot(data=train_org, y="SalePrice", x="LotArea",ax=ax1,color='green')
ax1.set_title('Orginal',fontsize=25)
sns.scatterplot(data =train_x, y="SalePrice", x="LotArea",ax=ax2)
ax2.set_title('After Robust Scaling',fontsize=25)
sns.despine()


# ----------------------------------------------------------------------
# # Nonlinear Scaling
# 
# ![](https://www.researchgate.net/profile/Xiao-Li-128/publication/324486223/figure/fig3/AS:614746801860608@1523578467516/Linear-vs-Nonlinear-response.png)
# 
# Picture Credit: https://www.researchgate.net
# 
# **Nonlinear Scaling changes the shape of the distribution.**

# ----------------------------------------
# ### Log Scaling
# 
# 
# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/Logarithm_plots.png/300px-Logarithm_plots.png)
# Picture Credit: https://upload.wikimedia.org
# 
# In other words, the amount of change on the y-axis according to the amount of change on the x-axis is small.
# 
# This feature is useful in the case of a feature with a large deviation of values between the data.
# 
# If the deviation is reduced, the skewness and kurtosis of the graph can be reduced, which also has the advantage of increasing normality.
# It can be said that it plays the role of regularization!!
# 
# Logarithm function increases the spacing between small numbers and reduces the spacing between large numbers. When certain features are dense with values in small values, by increasing these intervals, our models increase the intervals for small values, and we can improve the performance of the model when training and testing using these values.

# In[19]:


train_x = train_org.copy()
rcParams['figure.figsize'] = 25,30
fig, axes = plt.subplots(4, 2)
sns.set_style("white")
sns.set_palette("bright")
plt.subplots_adjust(hspace=0.35)
for i in range(2):
    train_x.loc[:,num_cols[i]] = np.log1p(train_x[[num_cols[i]]])
    sns.distplot(train_org.loc[:,num_cols[i]],ax=axes[0,i],rug=True,color='green')
    axes[0,i].set_title('Orginal '+num_cols[i],fontsize=25)
    sns.distplot(train_x.loc[:,num_cols[i]],ax=axes[1,i],rug=True,color='blue')
    axes[1,i].set_title('Log Scaling '+num_cols[i],fontsize=25)
    sns.boxplot(train_x.loc[:,num_cols[i]],ax=axes[2,i],color='blue')
    axes[2,i].set_title('Log Scaling '+num_cols[i],fontsize=25)
    stats.probplot(train_x[num_cols[i]],plot = axes[3,i])
    sns.despine()


# #### Checking Statistics

# In[20]:


display_stat()


# <span style="color:Blue"> Observation:
# 
# * Skewness and kurtosis were both lowered. The shape was changed to resemble a normal distribution.

# #### Checking Scatter

# In[21]:


sns.set(font_scale = 1.5)
sns.set_style("white")
sns.set_palette("bright")
rcParams['figure.figsize'] = 25,10
ax1 = plt.subplot(1,2,1)  
ax2 = plt.subplot(1,2,2)  
sns.scatterplot(data=train_org, y="SalePrice", x="LotArea",ax=ax1,color='green')
ax1.set_title('Orginal',fontsize=30)
sns.scatterplot(data =train_x, y="SalePrice", x="LotArea",ax=ax2)
ax2.set_title('After Log Scaling',fontsize=30)
sns.despine()


# Log scaling can be considered for data with skewed distribution as shown in the figure above. After scaling, the distribution was changed to a distribution close to the normal distribution.

# -------------------------------------------------------------
# ### Clipping
# 
# After setting the upper and lower limits, outliers outside a certain range can be removed by replacing only the upper limit with the lower limit for values ​​outside the range.
# 
# In this case, Values below 1% point are clipped as 1% point, and values above 99% point are clipped as 99% point.

# In[22]:


train_x = train_org.copy()
p01 = train_x[num_cols].quantile(0.01)
p99 = train_x[num_cols].quantile(0.99)

rcParams['figure.figsize'] = 25,40
sns.set_style("white")
sns.set_palette("bright")
plt.subplots_adjust(hspace=1.5)
# Values below 1% point are clipped to 1% point, and values above 99% point are clipped to 99% point
fig, axes = plt.subplots(4, 2)
for i in range(2):
    train_x.loc[:,num_cols[i]]  = train_x[[num_cols[i]]].clip(p01, p99, axis=1)
    sns.distplot(train_org.loc[:,num_cols[i]],ax=axes[0,i],rug=True,color='green')
    axes[0,i].set_title('Orginal '+num_cols[i],fontsize=25)
    sns.distplot(train_x.loc[:,num_cols[i]],ax=axes[1,i],rug=True,color='blue')
    axes[1,i].set_title('Clipping Scaling '+num_cols[i],fontsize=25)
    sns.boxplot(train_x.loc[:,num_cols[i]],ax=axes[2,i],color='blue')
    axes[2,i].set_title('Clipping Scaling '+num_cols[i],fontsize=25)
    stats.probplot(train_x[num_cols[i]],plot = axes[3,i])
    sns.despine()


# #### Checking Statistics

# In[23]:


display_stat()


# <span style="color:Blue"> Observation:
# * As the outlier is removed, it can be seen that skewness and kurtosis are lowered.
# * In this case, Values below 1% point are clipped as 1% point, and values above 99% point are clipped as 99% point. If it is easier to remove and replace outliers, you can change the clipping points. However, it should be remembered that there is a possibility that the distribution may be distorted by the clipped value as seen in the scatter plot below.

# #### Checking Scatter

# In[24]:


sns.set(font_scale = 1.5)
sns.set_style("white")
sns.set_palette("bright")
rcParams['figure.figsize'] = 25,10
ax1 = plt.subplot(1,2,1)  
ax2 = plt.subplot(1,2,2)  
sns.scatterplot(data=train_org, y="SalePrice", x="LotArea",ax=ax1,color='green')
ax1.set_title('Orginal',fontsize=30)
sns.scatterplot(data =train_x, y="SalePrice", x="LotArea",ax=ax2)
ax2.set_title('After Clipping',fontsize=30)
sns.despine()


# In particular, if you look at the Scatter plot, you can see that clipping is clear.

# -------------------------------------------
# ## Power Transformation
# 
# > In statistics, a power transform is a family of functions applied to create a monotonic transformation of data using power functions. It is a data transformation technique used to stabilize variance, make the data more normal distribution-like, improve the validity of measures of association (such as the Pearson correlation between variables), and for other data stabilization procedures.
# 
# https://en.wikipedia.org/wiki/Power_transform
# 
# ### Yeo-Johnson Transformation
# 
# ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/2a99e24c81226f3d0547c471281197ea265553c5)
# 
# Picture Credit: https://wikimedia.org

# In[25]:


from sklearn.preprocessing import PowerTransformer
train_x = train_org.copy()
rcParams['figure.figsize'] = 25,40
fig, axes = plt.subplots(4, 2)
sns.set_style("white")
sns.set_palette("bright")
plt.subplots_adjust(hspace=0.35)

pt = PowerTransformer(method='yeo-johnson')
for i in range(2):
    train_x.loc[:,num_cols[i]] = pt.fit_transform(train_x[[num_cols[i]]])
    sns.distplot(train_org.loc[:,num_cols[i]],ax=axes[0,i],rug=True,color='green')
    axes[0,i].set_title('Orginal '+num_cols[i],fontsize=25)
    sns.distplot(train_x.loc[:,num_cols[i]],ax=axes[1,i],rug=True,color='blue')
    axes[1,i].set_title('Yeo-johnson Scaling '+num_cols[i],fontsize=25)
    sns.boxplot(train_x.loc[:,num_cols[i]],ax=axes[2,i],color='blue')
    axes[2,i].set_title('Yeo-johnson Scaling '+num_cols[i],fontsize=25)
    stats.probplot(train_x[num_cols[i]],plot = axes[3,i])
    sns.despine()


# #### Checking Statistics

# In[26]:


display_stat()


# <span style="color:Blue"> Observation:
# * The change is very similar to the standard normal distribution. It is strange that the shape of the standard normal distribution is changed while performing non-linear transformation.

# #### Checking Scatter

# In[27]:


sns.set(font_scale = 1.5)
sns.set_style("white")
sns.set_palette("bright")
rcParams['figure.figsize'] = 25,10
ax1 = plt.subplot(1,2,1)  
ax2 = plt.subplot(1,2,2)  
sns.scatterplot(data=train_org, y="SalePrice", x="LotArea",ax=ax1,color='green')
ax1.set_title('Orginal',fontsize=30)
sns.scatterplot(data =train_x, y="SalePrice", x="LotArea",ax=ax2)
ax2.set_title('After Yeo-johnson Scaling',fontsize=30)
sns.despine()


# ### Box–Cox Transformation
# 
# ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/b565ae8f1cce1e4035e2a36213b8c9ce34b5029d)
# 
# Picture Credit: https://wikimedia.org

# In[28]:


from sklearn.preprocessing import PowerTransformer
train_x = train_org.copy()
rcParams['figure.figsize'] = 25,40
fig, axes = plt.subplots(4, 2)
sns.set_style("white")
sns.set_palette("bright")
plt.subplots_adjust(hspace=0.35)

pt = PowerTransformer(method= "box-cox")
for i in range(2):
    train_x.loc[:,num_cols[i]] = pt.fit_transform(train_x[[num_cols[i]]])
    sns.distplot(train_org.loc[:,num_cols[i]],ax=axes[0,i],rug=True,color='green')
    axes[0,i].set_title('Orginal '+num_cols[i],fontsize=25)
    sns.distplot(train_x.loc[:,num_cols[i]],ax=axes[1,i],rug=True,color='blue')
    axes[1,i].set_title('Box-cox Scaling '+num_cols[i],fontsize=25)
    sns.boxplot(train_x.loc[:,num_cols[i]],ax=axes[2,i],color='blue')
    axes[2,i].set_title('Box-cox Scaling '+num_cols[i],fontsize=25)
    stats.probplot(train_x[num_cols[i]],plot = axes[3,i])
    sns.despine()


# #### Checking Statistics

# In[29]:


display_stat()


# <span style="color:Blue"> Observation:
# * The change is very similar to the standard normal distribution. It is strange that the shape of the standard normal distribution is changed while performing non-linear transformation.

# #### Checking Scatter

# In[30]:


sns.set(font_scale = 1.5)
sns.set_style("white")
sns.set_palette("bright")
rcParams['figure.figsize'] = 25,10
ax1 = plt.subplot(1,2,1)  
ax2 = plt.subplot(1,2,2)  
sns.scatterplot(data=train_org, y="SalePrice", x="LotArea",ax=ax1,color='green')
ax1.set_title('Orginal',fontsize=30)
sns.scatterplot(data =train_x, y="SalePrice", x="LotArea",ax=ax2)
ax2.set_title('After Box-cox Scaling',fontsize=30)
sns.despine()


# -----------------------------------
# ### Quantile Transformer
# 
# 
# The quantile function ranks or smooths out the relationship between observations and can be mapped onto other distributions, such as the uniform or normal distribution.

# In[31]:


from sklearn.preprocessing import QuantileTransformer
train_x = train_org.copy()

rcParams['figure.figsize'] = 25,40
fig, axes = plt.subplots(4, 2)
sns.set_style("white")
sns.set_palette("bright")
plt.subplots_adjust(hspace=0.35)

transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
for i in range(2):
    train_x.loc[:,num_cols[i]] = transformer.fit_transform(train_x[[num_cols[i]]])
    sns.distplot(train_org.loc[:,num_cols[i]],ax=axes[0,i],rug=True,color='green')
    axes[0,i].set_title('Orginal '+num_cols[i],fontsize=25)
    sns.distplot(train_x.loc[:,num_cols[i]],ax=axes[1,i],rug=True,color='blue')
    axes[1,i].set_title('QuantileTransform Scaling '+num_cols[i],fontsize=25)
    sns.boxplot(train_x.loc[:,num_cols[i]],ax=axes[2,i],color='blue')
    axes[2,i].set_title('QuantileTransform Scaling '+num_cols[i],fontsize=25)
    stats.probplot(train_x[num_cols[i]],plot = axes[3,i])
    sns.despine()


# #### Checking Statistics

# In[32]:


display_stat()


# <span style="color:Blue"> Observation:
# * It has been changed to a nearly perfect standard normal distribution.

# #### Checking Scatter

# In[33]:


sns.set(font_scale = 1.5)
sns.set_style("white")
sns.set_palette("bright")
rcParams['figure.figsize'] = 25,10
ax1 = plt.subplot(1,2,1)  
ax2 = plt.subplot(1,2,2)  
sns.scatterplot(data=train_org, y="SalePrice", x="LotArea",ax=ax1,color='green')
ax1.set_title('Orginal',fontsize=30)
sns.scatterplot(data =train_x, y="SalePrice", x="LotArea",ax=ax2)
ax2.set_title('After QuantileTransforming',fontsize=30)
sns.despine()


# The distribution is changed by scaling to a shape similar to the normal distribution.

# <hr style="border: solid 3px blue;">
# 
# # Conclusion
# 
# It is not easy to obtain meaningful information from data and process features with it.
# However, in order to increase the predictive power, it is often necessary to properly process the features. To do this, it is necessary to first observe the features and figure out the distribution of the features.
# 
# The distribution of features can be a key to solving a problem and a starting point to better understand the collected data.
# In general, a distribution in the form of a gaussian distribution can be said to be a good distribution. If the distribution of features is not skewed, it is likely that linear scaling should be performed, otherwise, non-linear scaling should be performed.
# 
# However, there may be guides in the work of processing features, but there does not seem to be an answer. Understanding the data well and processing it accordingly will be the best solution.
# 

# In[ ]:




