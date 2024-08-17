#!/usr/bin/env python
# coding: utf-8

# One of the few Quantitative contribution we had in the Jane Street competition was this notebook: https://www.kaggle.com/pcarta/jane-street-time-horizons-and-volatilities. We had returns at different time horizons. The idea was to use pytorch as a an optimisation solver for estimation of parameters of a Brownian Motion via Maximum Likelihood. Some winning teams mentionned it as decisive for their understanding of the competition and at least one team managed to use it for target engineering (using estimated trend as target). I figured this could be used for volatility estimation here. This notebook rely on the code in the linked notebook.
# 
# For the moment I've only managed to make it work on a given time_id x stock_id time series and it take 6 seconds... it might require some additional work (factorization, parallelisation) to be usable. Also the Geometric Brownian motion part require assuming a constant volatility. This might not be optimal for a volatility forecasting competition and might require an upgrade of the model. Worth mentionning too is that I added a bit of stochastic calculus explanation at the beginning to make it more palatable for non-quants.

# In[1]:


# add it to G-research


# In[2]:


import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import random
import seaborn as sns


# # Stochastic Calculus Intro - Brownian Motion - Geometric Brownian Motion - Balck & Scholes model
# Stochastic Calculus is a whole Field centered around stochastic processes and their integration. It is heavily used in industry to deal with financial time series. The main building block of stochastic calculus is Brownian Motion, a random process. Inspired from erratic movements of pollen particles on water (observed by Brown in the 30's ... 1830's). It can be caracterised (and simulated) by random normal increments.
# 
# Taking a simple definition of Brownian motion:
# 
# - $W_0 = 0$
# - $W_t$ is almost surely continuous
# - $W_t$ as random increments 
# - $W_t - W_s = N(0,t-s)$ for $0 \leq s \le t$ 
# 
# 
# 

# Allows for simple simulation:

# In[3]:


n_step = 100

w = np.zeros(n_step)
        
for i in range(1,n_step):
    yi = np.random.normal()
    w[i] = w[i-1]+(yi/np.sqrt(n_step))
    
plt.plot(w)
plt.title('Simulated Brownian Motion')


# With some basic assumptions the price of a stock can be modeled as the exponential of a random walk, usually called Geometric Brownian Motion. This is a standard of financial modelling and might be used for option pricing in the Black & Scholes model. Under these assumptions, at time $t$, the price $S_t$ equals :
# 
# $$ S_0 e^{(\mu -\sigma^2)*t/2 + \sigma W_t} $$
# 
# Where $W_t$ is a Brownian motion. Of course assuming a constant volatility would be a problem for forecasting it. But we might use this model to estimate a smoothed volatility and use it as a feature. 
# 
# Using a vectorised version of the precedent calculation:

# In[4]:


n_step = 100
dt = 0.01
S0 = 1
mu = 0
sigma = 0.05
S = np.ones(n_step)

increments = np.random.normal(0, np.sqrt(dt), size=(1, n_step))
S = np.exp((mu - sigma ** 2 / 2) * dt + sigma * increments).T
S = np.vstack([np.ones(1), S])
S = S0 * S.cumprod(axis=0)

plt.plot(S)
plt.title( "Realizations of Geometric Brownian Motion")


# # Maximum Likelihood Estimation

# We might start by simplifying the model a bit, assuming no drift (and ignoring Ito's formula):
# 
# $$ S_t = S_0 e^{\sigma W_t} $$
# 
# 
# Under our assumptions, $W_t$ increments follow the same law and are uncorrelated. Each increment follow a normal law of mean $0$ and variance $\sigma^2 (T_i - T_{i-1})$. Then therefore have the joint probability density function of these increments, which is the product of pdf for Gaussian random variables. 
# 
# We can obtain the likelihood :
# 
# $$
# L(\sigma, T) = \prod_{i=1}^n \frac{1}{\sqrt{2 \pi \sigma^2 (T_i - T_{i-1})}} \exp \left(-\frac{{\Delta W_i}^2}{2 \sigma^2(T_i - T_{i-1})}\right)
# $$
# 
# By maximizing the value of the likelihood function with respect to the parameters, we can find the parameters which are most likely to have generated the data.
# 
# In practice we have the time and price increments so we can estimate the volatility by finding which volatility maximize the likelihood. Equivalently we can minimize the negative log-likelihood
# 
# $$
# \mathcal{l}(\sigma, T) = \sum_{i = 1}^n \left(\frac{{\Delta W_{i}}^2}{2 \sigma^2 (T_i - T_{i-1})} + \frac{1}{2}\log(T_i - T_{i-1}) + \log(\sigma)\right) + \text{const}
# $$
# 
# We can optimize the log-likelihood numerically, to obtain the estimated values of $\sigma$. The cool stuff here is using Pytorch to do so. 
# 
# 
# ## Caveats:
# This approach assume a lot things (normal increments of log returns, no drift, independence of increments). We know these assumptions are all wrong, but the volatility estimation might still be usefull.

# # Application to Optiver Data

# In[5]:


def calc_wap(df):
    wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1'])/(df['bid_size1'] + df['ask_size1'])
    return wap

def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff()

def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return**2))


# In[6]:


book_example = pd.read_parquet('../input/optiver-realized-volatility-prediction/book_train.parquet/stock_id=0')
trade_example =  pd.read_parquet('../input/optiver-realized-volatility-prediction/trade_train.parquet/stock_id=0')

stock_id = '0'
time_id = book_example.time_id.unique()

book_example = book_example[book_example['time_id'].isin(time_id)]
book_example.loc[:,'stock_id'] = stock_id

trade_example = trade_example[trade_example['time_id'].isin(time_id)]
trade_example.loc[:,'stock_id'] = stock_id

book_example['wap'] = calc_wap(book_example)
book_example['log_wap'] = np.log(book_example['wap'])
book_example.loc[:,'log_return'] = book_example.groupby('time_id')['log_wap'].diff()

book_example = book_example.merge(trade_example, on=['seconds_in_bucket','time_id'],how='left', suffixes=('', '_y'))


# In[7]:


ts_example = book_example[book_example['time_id']==5]


# In[8]:


plt.plot(ts_example.wap)
plt.title('Price - Stock 0 - Time 5')


# # Pytorch for Maximum Likelihood Estimation of Volatility

# In[9]:


import torch

# use cpu if gpu is not available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# get the increments
dW = ts_example.log_return
dW.loc[0] = 0

dT = ts_example.seconds_in_bucket.diff()
dT.loc[0] = 1

# initialize sigma randomly
sigma = torch.tensor(np.random.randn(1)*.01 + 1, device=device)

# the actual parameters are the logarithms of sigma, to enforce positivity and have more stable convergence

with torch.no_grad():
    sigma_log = torch.log(sigma).clone().requires_grad_(True)

# load the data to device (increments)
dW = torch.tensor(dW.values, device=device)
dT_log = torch.log(torch.tensor(dT.values, device=device))


# In[10]:


get_ipython().run_cell_magic('time', '', "\nITERS = 15000\n\n# use Adam as it finds the right learning rates easily\nopt = torch.optim.Adam([sigma_log])\n\niteration = 0\n\nwhile iteration < ITERS:\n    \n    # reset the gradients to 0\n    opt.zero_grad() \n    \n    # compute the log-likelihood\n    logL = 1/2 * torch.sum((dW**2 @ (1/torch.exp(dT_log)).float())) * (1/torch.exp(2*sigma_log)) + 1/2*torch.sum(dT_log) + sigma_log\n        \n    # compute the gradient\n    logL.backward()\n\n    if iteration % 1000 == 0:\n        with torch.no_grad():\n            print(f'iter {iteration:8} {logL} {sigma_log}')\n\n    # execute one step of gradient descent\n    opt.step()\n    \n    iteration+=1\n")


# Cool but too slow. Even with parallelisation we need to estimate that for hundreds of thousands of time series.

# # Estimated volatility v.s. realized volatility

# In[11]:


torch.exp(sigma_log)
np.sqrt(np.sum(log_return(ts_example.wap)[1:]**2))


# # Scipy version
# 
# Going Back to a basic optimisation library we might get faster results.

# In[12]:


get_ipython().run_cell_magic('time', '', "\ndW = ts_example.log_return\ndW.loc[0] = 0\n\ndT = ts_example.seconds_in_bucket.diff()\ndT.loc[0] = 1\n\n# initialize sigma randomly\nsigma_0 = np.random.randn(1)*.001\n\nlog_sigma_0 = np.log(sigma_0)\ndT_log = np.log(dT)\n\ndef neg_log_likelihood(log_sigma):\n     # compute the log-likelihood\n    logL = 1/2 * np.sum((dW**2 @ (1/np.exp(dT_log)))) * (1/np.exp(2*log_sigma)) + 1/2*np.sum(dT_log) + log_sigma\n    return logL\n\nres = minimize(neg_log_likelihood, log_sigma_0, method='nelder-mead',\n               options={'xatol': 1e-8, 'disp': True})\n")


# In[13]:


# 55 ms x 112 stocks x 3830 time ids / 4 threads = 1h30
(55/1000)*112*3830*1/4*1/3600


# In[14]:


np.exp(res.final_simplex[0][0][0])


# # Multivariate approach with scipy
# 
# Refactoring the code to get a multivariate approach might help too.

# In[15]:


def preprocessor_book(file_path_book):

    df_book = pd.read_parquet(file_path_book)
    stock_id = int(file_path_book.split('=')[1])

    df_book['wap'] = calc_wap(df_book)
    df_book['log_wap'] = np.log(df_book['wap'])
    df_book['log_return'] = df_book.groupby('time_id')['log_wap'].diff()

    unique_time_ids = df_book['time_id'].unique()

    res_BS = []

    for time_id in unique_time_ids:

        ts_example = df_book[df_book['time_id']==time_id]

        dW = ts_example.log_return.values
        dW[0] = 0

        dT = ts_example.seconds_in_bucket.diff().values
        dT[0] = 1

        # initialize sigma randomly
        sigma_0 = 0.0001

        log_sigma_0 = np.log(sigma_0)
        dT_log = np.log(dT)

        def neg_log_likelihood(log_sigma):
             # compute the log-likelihood
            logL = 1/2 * np.sum((dW**2 @ (1/np.exp(dT_log)))) * (1/np.exp(2*log_sigma)) + 1/2*np.sum(dT_log) + log_sigma
            return logL

        res = minimize(neg_log_likelihood, log_sigma_0, method='nelder-mead',
                       options={'xatol': 1e-8, 'disp': False})

        
        rv = np.sqrt(np.sum(np.square(dW)))
        
        res_BS.append((stock_id,time_id,np.exp(res.final_simplex[0][0][0]),rv))

    return pd.DataFrame(res_BS, columns=['stock_id', 'time_id', 'vol_BS','rv'])


def preprocessor(list_stock_ids, is_train = True):
    
    def for_joblib(stock_id):

        if is_train:
            file_path_book = data_dir + "book_train.parquet/stock_id=" + str(stock_id)
        else:
            file_path_book = data_dir + "book_test.parquet/stock_id=" + str(stock_id)
            
        df_tmp = preprocessor_book(file_path_book)
        
        return df_tmp
    
    # Use parallel api to call paralle for loop
    df = Parallel(n_jobs = -1, verbose = 1)(delayed(for_joblib)(stock_id) for stock_id in list_stock_ids)
    # Concatenate all the dataframes that return from Parallel
    df = pd.concat(df, ignore_index = True)
    
    return df

def read_train_test():
    train = pd.read_csv('../input/optiver-realized-volatility-prediction/train.csv')
    test = pd.read_csv('../input/optiver-realized-volatility-prediction/test.csv')
    # Create a key to merge with book and trade data
    train['row_id'] = train['stock_id'].astype(str) + '-' + train['time_id'].astype(str)
    test['row_id'] = test['stock_id'].astype(str) + '-' + test['time_id'].astype(str)
    print(f'Our training set has {train.shape[0]} rows')
    return train, test


# # test on all stocks

# In[16]:


get_ipython().run_cell_magic('time', '', "\ndata_dir  ='../input/optiver-realized-volatility-prediction/'\n\ntrain, test = read_train_test()\n\ntrain_stock_ids = train['stock_id'].unique()\ntest_stock_ids = test['stock_id'].unique()\n\ntrain_ = preprocessor(train_stock_ids, is_train = True)\ntest_ = preprocessor(test_stock_ids, is_train = False)\n\ntrain = train.merge(train_, on = ['time_id','stock_id'], how = 'left')\ntest = test.merge(test_, on = ['time_id','stock_id'], how = 'left')\n")


# In[17]:


def rmspe(y_true, y_pred):
    return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))


# In[18]:


print(rmspe(train.target, train.vol_BS))


# In[19]:


print(rmspe(train.target, train.rv))


# # No model Baseline

# In[20]:


test['target'] = test['vol_BS']
test['row_id'] = test['stock_id'].astype(str) + '-' + test['time_id'].astype(str)

test[['row_id', 'target']].to_csv('submission.csv',index = False)

