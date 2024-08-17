#!/usr/bin/env python
# coding: utf-8

# # TL'DR
# This is the first part in a series of tutorials showcasing how to automate much of feature engineering. 
# 
# In Part I (this tutorial), we show you ***how to estimate the best performance you can achieve consistently (i.e. in-sample and out-of-sample) using a set of features, in a single line of code, and without training any model.***
# 
# In [Part II](https://www.kaggle.com/leanboosting/automating-feature-engineering-part-ii), we will build on this tutorial to show you *how to seamlessly remove non-informative and redundant features from a set of candidate features.*
# 
# In [Part III](https://www.kaggle.com/leanboosting/automating-feature-engineering-part-iii), we will build on Part II and show you how to seamlessly add shrinkage/feature selection to any regressor in Python.
# 
# *Please upvote and share it you found it useful.*
# 
# # Table of Contents
# 
# - **I. [Math Background](#background)**
#  - **I.1 [Problem Formulation](#problem)**
#  - **I.2 [Solution](#solution)**
# - **II. [Application](#application)**
#  - **II.1 [Getting Started](#setup)**
#    - **II.1.a [How to install the kxy package](#install)**
#    - **II.1.b [How to get and use a kxy API key](#api)**
#    - **II.1.c [The set of features we are using in this tutorial](#features)**
#  - **II.2 [How to Calculate the Highest Performance Achievable in a Single Line of Code](#one-liner)**
#    - **II.2.a [Highest performance achievable by a cross-sectional model](#cross-sectional)**
#    - **II.2.b [Highest performance achievable by single-asset models](#cross-sectional)**
#  - **II.3 [Summary of Findings](#summary)**
# 
# # I. Background <a name="background"></a>
# To set the stage, we formally discuss challenges specific to feature engineering, and we recall new ML findings upon which this tutorial is based.
# 
# 
# ## I.1 Problem Formulation (Memoryless Case) <a name="problem"></a>
# Formally, we are interested in predicting a target variable $y \in \mathbb{R}$ given some raw inputs $x \in \mathbb{R}^d$.
# 
# Although all the *juice* we need to predict $y$ is fully contained in the raw inputs $x$, in practice, the tabular models we use to extract this juice can only learn specific types of input(s)-output patterns. 
# 
# For instance, linear models (e.g. OLS, LASSO, Ridge, ElasticNet etc.) can only capture linear relationships between inputs and the target. Similarly, decision tree regressors work best in situations where a *good global regressor* can be learned by partitioning the input space into $m$ disjoint subsets using simple decision rules, and learning $m$ *very simple local regressors*. Along the same lines, kernel-Ridge regressors and Gaussian process regressors with smooth kernels assume that inputs that are close to each other (in Euclidean distance) will always have targets that are also close to each other (in Euclidean distance). The same holds for nearest neighbors regressors. Beyond smoothness, kernel-based regressors can also be used to learn global patterns in situations where inputs that are close to each other in a given norm, will have outputs that are close to each other in Euclidean norm. The norm can be chosen to learn periodic or seasonal patterns, with or without a trend, to name but a few.
# 
# Oftentimes, the natural representation of raw inputs $x$ is inconsistent with the types of patterns our regression models rely on, even when $x$ is very informative about $y$. 
# 
# For instance, $y$ might not be linear in $x$, but it could be linear in $x^2$. In such a case, a linear regressor trained on $(y, x)$ would perform poorly, but a linear regressor trained on $(y, x^2)$ could perform very well. 
# 
# The same applies to tree-based models. Decision rules are typically as simple as whether a variable is above a certain threshold (i.e. whether an input point falls above or below a hyperplane in the input space). However, the right partition of the raw inputs space can sometimes be hard to generate using hyperplanes. Consider for instance the generative model (a.k.a. the ground truth) $y = \mu_0 + \epsilon ~~\text{  if  }~~ x_1^2 + x_2^2 < 1 ~~\text{  else  }~~ \mu_1 + \epsilon,$ where $\mu_0 \neq \mu_1$ are two constants and $\epsilon$ a noise term. 
# 
# If we train a regression tree directly on inputs $(x_1, x_2)$, then we will need to chain a lot of conditions of the form $x_1 < a$ and $x_2 < b$ to mimic the condition $x_1^2 + x_2^2 < 1$. If it is unclear why, imagine having to draw a circle (the true decision boundary) using intersections of vertical and horizontal lines. You'll end up with a lot of rectangles, each corresponding to a separate local regressor. Training that many regressors could result in overfitting as each rectangle wouldn't have enough data to reliably estimate $\mu_0$ or $\mu_1$. 
# 
# That said, if we introduce the feature $z:= x_1^2+x_2^2$ and train the tree using $z$ as input, then with only a single rule, namely $z < 1$, we can recover the true decision rule and the true model.
# 
# In general, a successful feature engineering pipeline requires three ingredients:
# 
# - **A) Valuable Data -** Raw inputs $x$ need to be informative about the target of interest $y$. No model or feature transformation can compensate for the fact that the raw inputs aren't informative about the target.
# - **B) Valuable Features -** The features $z:=f(x)$ that will be fed as inputs to our models need to be as informative about the target $y$ as possible. By the [data processing inequality](https://en.wikipedia.org/wiki/Data_processing_inequality), any feature transformation is necessarily *juice-reducing*, so we need to keep the reduction of *juice* to a minimum.
# - **C) Features-Model Adequacy -** The relationship between features $z$ and $y$ should be consistent with the types of patterns models in our toolbox can reliably learn.
# 
# Requirement C) is typically met by looking for features that are likely to have a simpler relationship to the target, one that models in our toolbox can reliably learn. In this competition for instance, the open (resp. high, low, close) price is unlikely to reveal much about the target, when used by itself. However, differences or ratios of these quantities indicate magnitudes of certain types of moves that may be informative about the target (in particular during trending markets).
# 
# On the other hand, meeting requirement B) by trial-and-error can be very time-consuming. This usually entails trying our features on a range of models and seeing what performance we get. In this approach, when trained models do not yield the desired performance, it is never clear which of our feature transformation and our model training is to blame. Is the poor performance due to a poor choice of model hyperparameters, using the wrong class of models, or the fact that our feature transformation missed out on a lot of juice that was originally in raw inputs $x$? Even when our trained models performed well enough out-of-sample, it would still be useful to know if we could do better. 
# 
# 
# 
# ## I.2 Solution (Memoryless Case) <a name="solution"></a>
# 
# To meet requirement B) above without resorting to a trial-and-error approach, we need a cost-effective way of estimating the best performance we could reliably achieve when using a set of features $z$ to predict a target $y$ (e.g. the highest Pearson's correlation $\bar{\rho}(P_{y, z})$ between prediction and target). 
# 
# $\bar{\rho}(P_{y, z})$ tells us how much juice our features still have, while comparing $\bar{\rho}(P_{y, z})$ to $\bar{\rho}(P_{y, x})$ (i.e. the highest Pearson's correlation achievable between predictions of $y$ made using $x$, no matter the feature transformation), we may determine how much juice that is in $x$ about $y$ our features $z$ are missing out on. 
# 
# Now, how exactly do we calculate $\bar{\rho}(P_{y, z})$? A *good* regression model $\mathcal{M}$ would usually take the form $y = g(z) + \epsilon,$ where the residual $\epsilon$ is at least mean-zero and decorrelated from the prediction $g(z)$; mean-zero because otherwise the model would be biased, and decorrelated residuals because otherwise there will be residual juice in $z$ about $y$ that $g(z)$ is not accounting for.
# 
# Using such a model, the correlation between the prediction and the target reads $$\text{Corr}(y, g(z)) = \frac{\text{Cov}(y, g(z))}{\sqrt{\text{Var}(y)\text{Var}(g(z))}} = \sqrt{\frac{\text{Var}(g(z))}{\text{Var}(y)}} := \sqrt{R^2(\mathcal{M})},$$ where $R^2(\mathcal{M})$ is the R-squared of our model $\mathcal{M}$. 
# 
# Thus, $$\bar{\rho}(P_{y, z}) := \sqrt{\bar{R}^2(P_{y, z})}$$ where $\bar{R}^2\left(P_{y, z}\right)$ is the highest R-squared achievable using $z$ to predict $y$.
# 
# In [[1]](#paper1), it was shown that $\bar{R}^2\left(P_{y, z}\right) := 1- e^{-2I(y; z)},$ where $I(y; z)$ is the mutual information between $y$ and $z$.  
# 
# The **kxy** Python package we use in this tutorial implements this formula, using the mutual information estimator of [[2]](#paper2).
# 
# 
# ## I.3 Extension to Time Series
# 
# So far we have only considered memoryless problems. These are problems where, although observations might have been collected in a chronological order, the order in which, or times at which, observations were collected is irrelevant to solve the problem. Time series problems on the other hand exhibit memory that could be useful to predict the target.
# 
# Nonetheless, the two formalisms are fairly similar. In effect, most time series regression models (e.g. ARIMA, RNN, LSTM etc.) are conceptually made of two building blocks: 
# 
# - A set of temporal features that encode the full extent to which time matters (i.e. what temporal patterns the model will attempt to exploit).
# - A memoryless or tabular regression model that uses the temporal features above as inputs.
# 
# Good temporal features are typically found either by trial-and-error (e.g. using EDA), or learned implicitly and jointly with the regression model (e.g. LSTM). Either way, once a set of candidate features (temporal and/or cross-sectional) has been found, we may estimate the highest performance achievable using the result of the previous section. 
# 
# 
# ## I.4 Effect of Nonstationarity
# 
# Strictly speaking, the extension to time series above requires the mutual information $I\left(y_t, z_t\right)$ to be the same for any time $t$.
# 
# In practice, good temporal features $z_t$ should be such that their dependency structure with $y_t$  (a.k.a. the copula $C(y_t, z_t)$) does not change over time, or at least changes slowly relative to the amount of observations required to train the regression model. When this is not the case, model performance will decay rapidly in production. 
# 
# The foregoing pratical requirement is stronger than our theoretical requirement that $I\left(y_t, z_t\right)$ be time-invariant because $I\left(y_t, z_t\right)$ is fully determined by $C(y_t, z_t)$.
# 
# Coincidentally, when applied to time series, the mutual information estimator [[2]](#paper2) only requires the joint copula between target and features $C(y_t, z_t)$ to be invariant over time or slow-changing. It does not require $\{y_t, x_t\}$ or $\{y_t, z_t\}$ to be stationary.
# 
# 
# 
# **Reference:**
# 
# - [1]<a name="paper1"></a> Samo, Y.L.K., 2021. LeanML: A Design Pattern To Slash Avoidable Wastes in Machine Learning Projects. arXiv preprint arXiv:2107.08066.
# - [2]<a name="paper2"></a> Samo, Y.L.K., 2021, March. Inductive Mutual Information Estimation: A Convex Maximum-Entropy Copula Approach. In International Conference on Artificial Intelligence and Statistics (pp. 2242-2250). PMLR.
# 
# 

# # II. Application
# Because mutual information estimation is compute-intensive, we implemented [[1]](#paper1) as a serverless Function-As-A-Service tool, the ``kxy`` Python package. 
# 
# All analyses provided in the ``kxy`` package are available as simple method calls from any pandas dataframe object; all you need is to install and import the ``kxy`` Python package. 
# 
# Our backend does the work for you, and you only get charged for the compute resource used to execute your function. 
# 
# ``kxy`` *is compeletely free to use in Kaggle competitions. Just sign up and use the promotional code KAGGLE21.*
# 
# ## II.1 Getting Started <a name="setup"></a>
# ### II.1.a Installation <a name="install"></a>

# In[1]:


get_ipython().system('pip install kxy -U')


# In[2]:


import os
import numpy as np
import pandas as pd
import pprint as pp
import kxy


# ### II.1.b Working with your KXY API key <a name="api"></a>
# 
# To get an API key, simply create an account [here](https://www.kxy.ai/portal). Once done, you'll find your API key [here](https://www.kxy.ai/portal/profile/identity/). 
# 
# Your API key should either be defined through the **KXY_API_KEY** environment variable, or by excecuting the command ``kxy configure <your_api_key>`` after installing the **kxy** Python package. 
# 
# **Note:** API keys should not be shared. If you are using the ``kxy`` package in a public notebook, we recommend recording your API key as a Kaggle secret and setting the ``KXY_API_KEY`` environment variable using the corresponding Kaggle secret. See the code-block below.

# In[3]:


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
kxy_api_key = user_secrets.get_secret('KXY_API_KEY')
os.environ['KXY_API_KEY'] = kxy_api_key


# ### II.1.c Generating Candidate Features <a name="features"></a>
# 
# Here we generate a set of 37 candidate features, temporal and cross-sectional.

# In[4]:


TRAIN_CSV = '/kaggle/input/g-research-crypto-forecasting/train.csv'
df_train = pd.read_csv(TRAIN_CSV)

def nanmaxmmin(a, axis=None, out=None):
    ''' '''
    return np.nanmax(a, axis=axis, out=out)-np.nanmin(a, axis=axis, out=out)


def get_features(df):
    ''' 
    An example function generating a candidate list of features.
    '''
    features = df[['Count', 'Open', 'High', 'Low', 'Close', \
                                    'Volume', 'VWAP','timestamp', 'Target', 'Asset_ID']].copy()
    # Upper shadow
    features['UPS'] = (df['High']-np.maximum(df['Close'], df['Open']))
    features['UPS'] = features['UPS'].astype(np.float16)
    
    # Lower shadow
    features['LOS'] = (np.minimum(df['Close'], df['Open'])-df['Low'])
    features['LOS'] = features['LOS'].astype(np.float16)
    
    # High-Low range
    features['RNG'] = ((features['High']-features['Low'])/features['VWAP'])
    features['RNG'] = features['RNG'].astype(np.float16)
    
    # Daily move
    features['MOV'] = ((features['Close']-features['Open'])/features['VWAP'])
    features['MOV'] = features['MOV'].astype(np.float16)
    
    # Close vs. VWAP
    features['CLS'] = ((features['Close']-features['VWAP'])/features['VWAP'])
    features['CLS'] = features['CLS'].astype(np.float16)
    
    # Log-volume
    features['LOGVOL'] = np.log(1.+features['Volume'])
    features['LOGVOL'] = features['LOGVOL'].astype(np.float16)
    
    # Log-count
    features['LOGCNT'] = np.log(1.+features['Count'])
    features['LOGCNT'] = features['LOGCNT'].astype(np.float16)
    
    # Drop raw inputs
    features.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', \
                           'Count'], errors='ignore', inplace=True)
    
    # Previous target !! WARNING: THIS FEATURE IS NOT TRADEABLE !!
    features['PREVTARGET'] = features.groupby('Asset_ID')['Target'].shift(1)
    
    # Enrich the features dataframe with some temporal feautures 
    # (specifically, some stats on the 4 bars worth of bars)
    features = features.kxy.temporal_features(max_lag=3, \
                exclude=['LOGVOL', 'LOGCNT', 'timestamp', 'Target'],
                groupby='Asset_ID')
    
    # Enrich the features dataframe context around the time
    # (e.g. hour, day of the week, etc.)
    time_features = features.kxy.process_time_columns(['timestamp'])
    features.drop(columns=['timestamp'], errors='ignore', inplace=True)
    features = pd.concat([features, time_features], axis=1)
    
    return features


# In[5]:


try:
    # Reading candidate features from disk
    training_features = pd.read_parquet('../input/cross-asset-featuresparquet/cross_asset_features.parquet')
    
except:
    # Randomly select 10% of days to speed-up features generation
    df_train['DAYSTR'] = pd.to_datetime(df_train['timestamp'], unit='s').apply(lambda x: x.strftime("%Y%m%d"))
    all_days = list(set([_ for _ in df_train['DAYSTR'].values]))
    selected_days = np.random.choice(all_days, size=int(len(all_days)/10), replace=False)
    df_train = df_train[df_train['DAYSTR'].isin(selected_days)]
    df_train.drop(columns=['DAYSTR'], errors='ignore', inplace=True)
    # Generating candidate features
    training_features = get_features(df_train)
    # Saving to disk
    to_save = training_features.astype(np.float32)
    to_save.to_parquet('cross_asset_features.parquet')
    del to_save
training_features


# In[6]:


# Printing all feautures
all_features = sorted([_ for _ in training_features.columns])
pp.pprint(all_features)


# ## II.2 How To Calculate Highest Performance Achievable <a name="one-liner"></a>
# The syntax is ``features_df.kxy.data_valuation(target_column, problem_type='regression')`` and it works on any pandas DataFrame object, so long as you import the ``kxy`` package.
# 
# ### II.2.a Case 1: Cross-Asset Models <a name="cross-sectional"></a>
# This is the maximum performance we may achieve with the features above using a single model to trade all crypto-currencies.

# In[7]:


ca_data_valuation_df = training_features.kxy.data_valuation('Target', problem_type='regression')


# In[8]:


ca_data_valuation_df['Achievable Pearson Correlation'] = '%.2f' % np.sqrt(
    float(ca_data_valuation_df['Achievable R-Squared'].iloc[0]))
ca_data_valuation_df


# ### II.2.b Case 2: Single-Asset Models <a name="single-asset"></a>
# 
# This is the maximum performance we may achieve with the features above using one model per crypto-currency.

# In[9]:


ASSET_CSV = '/kaggle/input/g-research-crypto-forecasting/asset_details.csv'
asset_details = pd.read_csv(ASSET_CSV)
asset_details.set_index(['Asset_ID'], inplace=True)


# In[10]:


results = {}
results['== Cross-Asset =='] = [\
        '%.2f' % np.sqrt(float(ca_data_valuation_df['Achievable R-Squared'].iloc[0])), \
        ca_data_valuation_df['Achievable R-Squared'].iloc[0]]
for asset_id in range(14):
    asset_name = asset_details.loc[asset_id]['Asset_Name']
    df = training_features[training_features['Asset_ID']==asset_id]
    data_valuation_df = df.kxy.data_valuation('Target', problem_type='regression')
    print(asset_name)
    print(data_valuation_df)
    print('\n\n')
    results[asset_name] = [\
        '%.2f' % np.sqrt(float(data_valuation_df['Achievable R-Squared'].iloc[0])), \
        data_valuation_df['Achievable R-Squared'].iloc[0]]
    
breakdown_df = pd.DataFrame.from_dict(results, orient='index', columns=[
    'Achievable Pearson Correlation', 'Achievable R-Squared'])


# ## II.3 Summary of Findings <a name="summary"></a>

# In[11]:


breakdown_df


# ### Update
# 
# **PREVTARGET** features have look-ahead bias, and as such they can't be traded. Check out [Part II](https://www.kaggle.com/leanboosting/automating-feature-engineering-part-ii) to find out more. 
# 
# In the meantime, here's the cross-asset achievable performance without **PREVTARGET** features.

# In[12]:


clean_features = training_features.drop(
    columns=[_ for _ in training_features.columns if 'PREVTARGET' in _], errors='ignore')
clean_feature_names = sorted([_ for _ in clean_features.columns])
pp.pprint(clean_feature_names)
clean_ca_data_valuation_df = clean_features.kxy.data_valuation(
    'Target', problem_type='regression')
clean_ca_data_valuation_df['Achievable Pearson Correlation'] = '%.2f' % np.sqrt(
    float(clean_ca_data_valuation_df['Achievable R-Squared'].iloc[0]))
clean_ca_data_valuation_df

