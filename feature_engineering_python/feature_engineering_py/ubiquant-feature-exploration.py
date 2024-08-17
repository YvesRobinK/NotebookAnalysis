#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:a239c6bb-8ce8-4c66-8c45-c7363aad3e48.png)
# 
# # Exploratory Data Analysis of the Features
# 
# @marketneutral
# 
# This is my second Ubiquant competition notebook. Please also see my other three notebooks.
# 
# - ["Ubiquant Target EDA PCA Magic"](https://www.kaggle.com/marketneutral/ubiquant-target-eda-pca-magic)
# - ["Stacking & Feature Importance"](https://www.kaggle.com/marketneutral/stacking-feature-importance)
# - ["Robust Multi-Target Pytorch"](https://www.kaggle.com/marketneutral/robust-multi-target-pytorch)
# 
# 
# # Updated!
# 
# Version Notes
# 
# 1. Version 4 released Feb 8, 2022
# 2. Version 6 released Feb 9, 2022
#     - added feature to target corr and rolling IC

# <div class="alert alert-warning">
# <b>TL;DR:</b> The features are anonymous but represent "alphas": a vector of dimensionless numbers which should be directly proportionate to the forward return target. Although the features are anonymous, we <b>can</b> uncover some details. Mainly we can determined the "speed" of the alpha: how quickly it turns over aross time. 
# </div>

# In[1]:


import warnings
#warnings.filterwarnings("ignore")


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import multiprocessing

from scipy.stats import spearmanr

plt.style.use('bmh')
plt.rcParams['figure.figsize'] = (16, 5)

N_CPU = multiprocessing.cpu_count()


# # Load and Inspect the Data
# 
# Thanks [Rob Mulla](https://www.kaggle.com/robikscube) for the reduced memory version of the data.

# In[3]:


get_ipython().run_cell_magic('time', '', "train = (pd.read_parquet('../input/ubiquant-parquet/train_low_mem.parquet')\n         .sort_values(['investment_id', 'time_id']));\n")


# Let's take a peak and see what the data looks like.

# In[4]:


train.head()


# In[5]:


train.info()


# In[6]:


train.isnull().sum().sum()


# # Feature Fingerprints
# 
# There are 300 features, so it is difficult to do a full visual EDA, but we can try! The following is what I would call a fingerprint plot. The x-axis is the feature value and the y-axis is the target value. From a univariate perspective, these features (visually) seem like mostly noise.

# In[7]:


def make_multi_scatters(df, nrow, ncol, figsize, start_count=0):
    fig, ax = plt.subplots(ncol, nrow, figsize=figsize, sharey=True)
    fig.patch.set_visible(False)
    fig.suptitle(f"Feature {start_count} to {start_count -1 + nrow*ncol} (target on y-axis)")

    plt.tight_layout()
    sampled = df.sample(frac=0.10)

    for i, axes in enumerate(ax.flatten()):
        axes.scatter(sampled[f'f_{i + start_count}'], sampled['target'], alpha=0.15)
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
        axes.axis('off')
        plt.text(
            0.1,
            0.9,
            f'f_{i+start_count}',
            horizontalalignment='left',
            verticalalignment='top',
            transform = axes.transAxes)  


# In[8]:


make_multi_scatters(train, 10, 10, (12, 12), start_count=0)


# In[9]:


make_multi_scatters(train, 10, 10, (12, 12), start_count=100)


# In[10]:


make_multi_scatters(train, 10, 10, (12, 12), start_count=200)


# # Thinking (and Trading) Fast and Slow
# 
# We've discused in the prior notebook, that the target is some kind of forward return for the `investment_id`. The features represent an "alpha" meant to be informative about the relative forward ranking of the target. We don't know what time horizon the target is for and we don't know anything about the individual alphas. One thing we can ascertain though, is the **turnover** of each alpha. In portfolio management, the turnover means the amount of trading that needs to be done for time *t* to rebalance the portfolio from *t-1* to the *t* target portfolio. We can imagine that an individual feature represents the target portolio for time *t*. If we look at the rank correlation between the feature on time *t* vs the feature on time *t-1* then we will have a proxy for the turnover of that feature. This can be very useful when we look at feature importance in a model later. Features can be **fast** or **slow**. A fast feature means it changes a lot from day-to-day. If the feature is defined by price and volume data for the recent history than it will likely change very quickly as time progresses because the data inputs change a lot. In comparison, if a feature is made from, for example, financial statement data (e.g., sales, asset information, etc.), then this feature will not change a lot from day to day since financial statement data is updated very infrequently (e.g., every 3 months in the United States). Let's get a flavor then of the turnover of the features.
# 
# How do we show visually if an alpha is fast or slow? We can do the following
# 
# - bin the alpha into deciles across stocks per day. We take each feature value and calculate what decile the feature value falls into *for that feature for that day*.
# - we sort the dataframe once by the last day.
# - we then plot this as a heatmap over time.
# 
# Is it better for an alpha to be fast or slow? It depends on the horizon of the target! Intuitively, imagine the target is the forward 6-month return. If your alpha whips around with dramatically different values every day, it is hard to rationalize that it could be predictive of 6-month returns. A slower alpha would likely be better in this case. 
# 
# Let's see two examples...

# In[11]:


train.set_index(['time_id', 'investment_id'], inplace=True)


# In[12]:


all_columns = train.columns
features = all_columns[train.columns.str.contains('f_')]


# In[13]:


def make_turnover_plot(df, feature, start_day=1100):
    div_map = sns.diverging_palette(220, 20, as_cmap=True)
    ax = (sns.heatmap(
        np.round(
            train[feature]
                 .unstack()
                 .T
                 .iloc[:, start_day:]
                 .rank(axis=0, pct=True)
                 .sort_values(1219)*10
            ,0),
        cmap='PuOr',
    ))
    ax.set(yticklabels=[]) 
    plt.title(f'{feature}: `investment_id` Binned by Feature Value vs Time')


# ## A Slow Alpha
# 
# Feature 4 is a classic example of a **very slow** alpha. In fact, we can further see that this alpha is driven by **monthly data inputs** primarily. A month is ~20 business days. You can see that in the last 20 days, stocks barely changed deciles. In fact if you look at the 100-day period plotted, the stocks barely change bins. For a slow alpha, if you know the ranks for day *t*, you can guess closely what those ranks would be for *t-1*.

# In[14]:


make_turnover_plot(train, 'f_4')


# ## A Fast Alpha
# 
# Feature 9 is a **very fast** alpha. You can see that the bin membership for the `investment_id`s flips around dramatically every day. 

# In[15]:


make_turnover_plot(train, 'f_9')


# ## The Target Speed
# 
# The target looks like it is very fast!

# In[16]:


make_turnover_plot(train, 'target')


# ## Measuring Alpha Speed: Rank Autocorrelation
# 
# We've seen it visually, but we want a measure we can apply across all features. The rank autocorrelation does this. For a given feature, we calculate the rank correlation for day *t* and day *t-1*. If the rank autocorrelation is **high** then the ranks don't change much and this is a **slow** alpha. If the rank autocorrelation is low then the ranks whip around a lot and this is a **fast** alpha.

# In[17]:


def factor_rank_autocorrelation(feature_name='f_4',factor_data=train, period=1):
    """
    
    This function is a slightly modified version of the same from:
    https://github.com/stefan-jansen/alphalens-reloaded
    
    Computes autocorrelation of mean factor ranks in specified time spans.
    We must compare period to period factor ranks rather than factor values
    to account for systematic shifts in the factor values of all names or names
    within a group. This metric is useful for measuring the turnover of a
    factor. If the value of a factor for each name changes randomly from period
    to period, we'd expect an autocorrelation of 0.
    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    period: int, optional
        Number of days over which to calculate the turnover.
    Returns
    -------
    autocorr : pd.Series
        Rolling 1 period (defined by time_rule) autocorrelation of
        factor values.
    """
    grouper = [factor_data.index.get_level_values('time_id')]

    ranks = factor_data.groupby(grouper)[feature_name].rank()

    asset_factor_rank = ranks.reset_index().pivot(index='time_id',
                                                  columns='investment_id',
                                                  values=feature_name)

    asset_shifted = asset_factor_rank.shift(period)

    autocorr = asset_factor_rank.corrwith(asset_shifted, axis=1)
    autocorr.name = period
    return autocorr


def rank_mean(feature):
    return factor_rank_autocorrelation(feature).mean()


# In[18]:


def make_feature_rank_corr_plot(factor_data, feature_name, period=1):
    autocorr = factor_rank_autocorrelation(
        feature_name,
        factor_data, 
    )
    autocorr.plot(
        title=f'{feature_name}: Rolling 1-period Rank Autocorrelation',
        ylim=(-0.2, 1.05)
    );
    plt.axhline(autocorr.mean(), c='green', linestyle='--')
    plt.axhline(0)


# We see for the **slow** alpha, the autocorrelation is very close to 1.0 always.

# In[19]:


make_feature_rank_corr_plot(train, 'f_4')


# For the fast alpha, we see that the rolling autocorrelation is lower.

# In[20]:


make_feature_rank_corr_plot(train, 'f_9')


# In[21]:


make_feature_rank_corr_plot(train, 'target')


# Let's get the mean rank correlation for **all** features. We can use this later along with feature importance to tell us about the nature of this competition.

# In[22]:


get_ipython().run_cell_magic('time', '', '\np = multiprocessing.Pool(N_CPU - 1)\n\nfeature_rank_corr_dict = dict(\n    zip(features,\n        p.map(rank_mean, features))\n)\n')


# In[23]:


fra = pd.DataFrame(
    index=feature_rank_corr_dict.keys(),
    data=feature_rank_corr_dict.values(),
    columns=['fra']
)

fra.to_csv('fra.csv')


# In[24]:


fra.describe().T


# In[25]:


fra.sort_values(by='fra').plot(title='Speeds of All Features (1=Slow, 0=Fast)');


# In[26]:


fig, ax = plt.subplots(1, 2)

(fra.sort_values(by='fra')[:15]
 .plot(kind='barh',
       title='Fastest Features',
       legend=False,
       ax=ax[0]))

ax[0].set_xlabel("Mean Daily Rank Autocorrelation")

(fra.sort_values(by='fra')[-15:]
 .plot(kind='barh',
       title='Slowest Features',
       legend=False,
       ax=ax[1]))

ax[1].set_xlabel("Mean Daily Rank Autocorrelation");


# # Target Prediction Correlation
# 
# From the competition Overview -> Evaluation tab
# > Submissions are evaluated on the mean of the Pearson correlation coefficient for each time ID.
# 
# To evaluate the features you need to calculated the per `time_id` correlation with the target and then take the mean over all periods. In quantitative finance, this is called the **mean information coefficient**, or just **mean IC**.
# 

# In[27]:


def target_correlation(feature_name='f_4', factor_data=train):

    grouper = [factor_data.index.get_level_values('time_id')]

    ranks = factor_data.groupby(grouper)[feature_name].rank()

    asset_factor_rank = ranks.reset_index().pivot(
        index='time_id',
        columns='investment_id',
        values=feature_name
    )

    target_ = factor_data['target'].unstack()

    corr = asset_factor_rank.corrwith(target_, axis=1)
    corr.name = 'corr'
    return corr

def ic_mean(feature):
    return target_correlation(feature).mean()


# In[28]:


target_correlation('f_231').plot(title='Rolling IC for feature f_231');


# In[29]:


get_ipython().run_cell_magic('time', '', '\np = multiprocessing.Pool(N_CPU - 1)\n\ntarget_corr_dict = dict(\n    zip(features,\n        p.map(ic_mean, features))\n)\n')


# In[30]:


corr = pd.DataFrame(
    index=target_corr_dict.keys(),
    data=target_corr_dict.values(),
    columns=['corr']
)

corr['corr_abs'] = corr['corr'].abs()
corr.to_csv('corr.csv')


# In[31]:


corr.describe().T


# In[32]:


fig, ax = plt.subplots(1, 2)
corr.sort_values(by='corr_abs')[-15:].plot(kind='barh', title='Most Predictive Features', legend=False, ax=ax[0])
ax[0].set_xlabel("Pearson Corr with Target")

corr.sort_values(by='corr_abs')[:15].plot(kind='barh', title='Least Predictive Features', legend=False, ax=ax[1])
ax[1].set_xlabel("Pearson Corr with Target");


# # Feature Rank Autocorrelation vs. Target Correlation

# In[33]:


sns.lmplot(x='fra',y='corr_abs',data=corr.join(fra).sort_values('corr_abs'), fit_reg=True)
plt.title('Feature Speed vs Feature:Target Corr')


# In[34]:


corr.join(fra).sort_values('corr_abs', ascending=False)


# This is surprising to me... it doesn't appear that alpha turnover is correlated to the abilty of the alpha to predict the target. From a financial perspective, it is surprising because it means that an almost static portfolio allocation can still outperform. From a machine learning perspective, it is suprising beacuse it means that "dropping low variance features" is not a sensible feature selection strategy. 

# # Engineering "Speed" Features
# 
# I had to create a new notebook for utilizing the speed features in modeling due to out-of-memory issues. Please see ["Stacking & Feature Importance"](https://www.kaggle.com/marketneutral/stacking-feature-importance) to see an example of how the speed information can be used. The speed features are both important and stable.
# 

# Thank you for taking a look at this notebook. Please leave comments and suggestions.
