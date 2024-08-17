#!/usr/bin/env python
# coding: utf-8

# In this kernel, I’m sharing my Gaussian Process Feature Engineering method with celerite used on my 19th place solution ( https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75167 ) 
# 
# Gaussian Process is used to interpolate flux curve, then extract features from the light curves. Some of Gaussian Process model tuning refinement from my original code was done with reference @CPMP 's solution as the following.
# 
# Thanks @CPMP.
# CPMP’s Solution detail: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75050  
# CPMP’s Github repo:  https://github.com/jfpuget/Kaggle_PLAsTiCC  
# 
# Remaining issue:  Some of the flux max can not be captured with current implementation of gaussian process curve.
# 

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Basic Library
import pandas as pd
import pandas.io.sql as psql
import numpy as np
import numpy.random as rd
import gc
import multiprocessing as mpa
import os
import sys
import pickle
from collections import defaultdict
from glob import glob
import math
from datetime import datetime as dt
from pathlib import Path
import scipy.stats as st
import re
from scipy.stats.stats import pearsonr
from itertools import combinations

# Matplotlib
import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc

from matplotlib import animation as ani
from IPython.display import Image

from joblib import Parallel, delayed
import multiprocessing as mp

plt.rcParams["patch.force_edgecolor"] = True
#rc('text', usetex=True)
from IPython.display import display # Allows the use of display() for DataFrames
import seaborn as sns
sns.set(style="whitegrid", palette="muted", color_codes=True)
sns.set_style("whitegrid", {'grid.linestyle': '--'})
red = sns.xkcd_rgb["light red"]
green = sns.xkcd_rgb["medium green"]
blue = sns.xkcd_rgb["denim blue"]

pd.set_option("display.max_colwidth", 100)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.options.display.float_format = '{:,.5f}'.format

get_ipython().run_line_magic('matplotlib', 'inline')
#%config InlineBackend.figure_format='retina'


# In[ ]:


# for gaussian process
from scipy.optimize import minimize
import celerite
from celerite import terms
from celerite.modeling import Model


# In[ ]:


def get_data_for_gp(df_pb, offset = 11):
    # original code of this part of code is CPMP's following notebook
    # https://github.com/jfpuget/Kaggle_PLAsTiCC/blob/master/code/celerite_003.ipynb
    x_min = df_pb.mjd.min()
    x_max = df_pb.mjd.max()

    yerr_mean = df_pb.flux_err.mean()
    x = df_pb.mjd.values
    y = df_pb.flux.values
    yerr = df_pb.flux_err

    x = np.concatenate((np.linspace(x_min-250, x_min -200, offset), x, np.linspace(x_max+200, x_max+250, offset),))
    y = np.concatenate((np.random.randn(offset) * yerr_mean, y, np.random.randn(offset) * yerr_mean))
    yerr = np.concatenate((yerr_mean * np.ones(offset), yerr, yerr_mean * np.ones(offset) ))
    return x, y, yerr


optimizer_list = ["L-BFGS-B",
                  "Nelder-Mead",
                  "Powell",
                  "CG",
                  "BFGS",
                  "Newton-CG",
                  "TNC",
                  "COBYLA",
                  "SLSQP",
                  "dogleg",
                  "trust-ncg",]

def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)

def grad_neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.grad_log_likelihood(y)[1]

def build_gp_model(x, y, yerr, n_param = 2):
    log_sigma = 0
    log_rho = 0
    eps = 0.001
    bounds = dict(log_sigma=(-15, 15), log_rho=(-15, 15))
    kernel = terms.Matern32Term(log_sigma=log_sigma, 
                                log_rho=log_rho, 
                                eps=eps, 
                                bounds=bounds)

    gp = celerite.GP(kernel, mean=0)
    gp.compute(x, yerr) 

    initial_params = gp.get_parameter_vector()
    bounds = gp.get_parameter_bounds()

    # depend on a combination of optimizer and dataset, minimize function throw exception,
    # so trying all type of method.
    for opt in optimizer_list:
        try:
            r = minimize(neg_log_like, 
                         initial_params, 
                         jac=grad_neg_log_like, 
                         method=opt, #"L-BFGS-B", 
                         bounds=bounds, 
                         args=(y, gp))
            return gp

        except Exception as e:
            pass
    raise Exception("[build_gp_model] can`t optimize")


# In[ ]:


ZIP = False
if ZIP:
    train = pd.read_csv('../input/training_set.csv.zip', compression='zip')
else:
    train = pd.read_csv('../input/training_set.csv')
    
meta_train = pd.read_csv('../input/training_set_metadata.csv')
meta_train["gal"] = (meta_train['hostgal_specz'] == 0).astype(int)
classes = np.sort(meta_train.target.unique())
########################################
n_row = 2 # with changing this variable, you can set the number of graphs for each target showing in this notebook.
########################################
import numpy.random as rd
rd.seed(71)
df_list = []
for u in np.sort(meta_train.target.unique()):
    df_list.append(meta_train[meta_train.target==u].sample(frac=1).iloc[:n_row,:]) # with shuffle
df_target_short = pd.concat(df_list)


# # Visualization
# 

# In[ ]:


def draw_gp_result(object_id_list):
    for object_id in object_id_list:
        meta = meta_train[meta_train.object_id==object_id]
        df = train[train.object_id==object_id]

        for pb in range(6):
            df_pb = df[df.passband == pb]

            flux_err_mean = df_pb.flux_err.mean()
            flux_err_std = df_pb.flux_err.std()

            # using flux_err within mean±6std
            df_pb = df_pb[df_pb.flux_err <= flux_err_mean + 6*flux_err_std]

            x, y, yerr = get_data_for_gp(df_pb)
            gp = build_gp_model(x, y, yerr)

            # graph drowing
            colors = ["r","b", "g", "k", "purple", "orange"]
            n_xx = 800
            gp_xx = np.linspace(x.min(), x.max(), n_xx)
            mu, var = gp.predict(y, gp_xx, return_var=True) 
            # N/A interpolation
            mu = np.interp(gp_xx, gp_xx[~np.isnan(mu)], mu[~np.isnan(mu)], )
            
            # Draw Graph
            ax = plt.subplot(111)
            df_pb.plot.scatter("mjd", "flux", c=colors[pb], figsize=(18,4), ax=ax) #cmap=cm.rainbow,
            ax.errorbar(df_pb.mjd, df_pb.flux, df_pb.flux_err, lw=1, fmt="none", ecolor="k", zorder=-1, label="flux_err")
            plt.plot(gp_xx, mu, color="gray")

            plt.fill_between(gp_xx, mu+np.sqrt(var), mu-np.sqrt(var), color="orange", alpha=0.3, edgecolor="none")
            plt.title(f"oid:{object_id}, passband:{pb}, class:{meta.target.values[0]}")
            plt.show()


# In[ ]:


for c in classes:
    print("="*30, f" class_{c} ", "="*30)
    oid_list = df_target_short[df_target_short.target==c].object_id.values
    draw_gp_result(oid_list)


# In[ ]:





# # Create Features

# In[ ]:


def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=mp.cpu_count())(delayed(func)(group) for name, group in dfGrouped)
    return pd.concat(retLst)

def gp_features(df):
    prefix = "gp001:"
    feature = {}
    feature["object_id"] = [df.object_id.values[0]]

    mu_interp_list = []
    for pb in range(6):
        df_pb = df[df.passband == pb]

        flux_err_mean = df_pb.flux_err.mean()
        flux_err_std = df_pb.flux_err.std()

        df_pb = df_pb[df_pb.flux_err <= flux_err_mean + 6*flux_err_std]
        x, y, yerr = get_data_for_gp(df_pb)

        try:
            gp = build_gp_model(x, y, yerr)

            n_xx = 800
            dx = (x.max() - x.min()) / n_xx
            gp_xx = np.linspace(x.min(), x.max(), n_xx)
            mu, var = gp.predict(y, gp_xx, return_var=True) 
            mu = np.interp(gp_xx, gp_xx[~np.isnan(mu)], mu[~np.isnan(mu)], )
            mu_interp_list.append(mu)    

            feature[f"{prefix}mean_var_pb{pb}"] = [np.mean(var)]
            feature[f"{prefix}skew_pb{pb}"] = [st.skew(mu)]
            feature[f"{prefix}range_pb{pb}"] = [mu.max() - mu.min()]

            mu_slope = pd.Series(np.diff(mu) / dx).rolling(window=30).mean()
            feature[f"{prefix}slope_max_pb{pb}"] = [mu_slope.max()]
            feature[f"{prefix}slope_min_pb{pb}"] = [mu_slope.min()]

        except Exception as e:
            print(e)
            mu_interp_list.append(np.zeros_like(gp_xx))

            feature[f"{prefix}skew_pb{pb}"] = [np.nan]
            feature[f"{prefix}range_pb{pb}"] = [np.nan]
            feature[f"{prefix}slope_max_pb{pb}"] = [np.nan]
            feature[f"{prefix}slope_min_pb{pb}"] = [np.nan]

    for c1, c2 in combinations(range(6), 2):
        ratio = mu_interp_list[c1] / mu_interp_list[c2]
        ratio = ratio[~np.isnan(ratio)]
        feature[f"{prefix}ratio_pb{c1}_{c2}"] = [np.mean(ratio)]

        feature[f"{prefix}corr_pb{c1}_{c2}"] = [pearsonr(mu_interp_list[c1], mu_interp_list[c2])[0]]

    return pd.DataFrame(feature)

import pickle
def unpickle(filename):
    with open(filename, 'rb') as fo:
        p = pickle.load(fo)
    return p

def to_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, -1)


# In[ ]:


# Test one object
oid = 23822
df = train[train.object_id==oid]
df_gp_feat = gp_features(df)


# In[ ]:


df_gp_feat


# In[ ]:


get_ipython().run_cell_magic('time', '', '# all object ids\ndf_gp_feature = applyParallel(train.groupby("object_id"), gp_features)\n')


# In[ ]:


df_gp_feature.head()


# # EDA

# In[ ]:


df_gp_feature = df_gp_feature.merge(meta_train, on="object_id", how="left")
df_gp_feature_dropna = df_gp_feature.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()


# In[ ]:


upper_limit = 99999999
df_gp_feature_dropna["gp001:ratio_pb1_3"] = np.where(np.abs(df_gp_feature_dropna["gp001:ratio_pb1_3"])>upper_limit, 
                                                     np.sign(df_gp_feature_dropna["gp001:ratio_pb1_3"])*upper_limit, 
                                                     df_gp_feature_dropna["gp001:ratio_pb1_3"])


# In[ ]:


for i, c in enumerate(df_gp_feature_dropna.loc[:,df_gp_feature_dropna.columns.str.startswith("gp001:")]):
    try:
        print(c)
        df_gp_feature_dropna[c] = np.where(np.abs(df_gp_feature_dropna[c]) > 99999999, 
                                                     np.sign(df_gp_feature_dropna[c])*99999999, 
                                                     df_gp_feature_dropna[c])
        
        plt.figure(figsize=[20, 4])
        plt.subplot(1, 2, 1)
        sns.violinplot(x='target', y=c, data=df_gp_feature_dropna)
        plt.grid()

        plt.subplot(1, 2, 2)
        sns.distplot(df_gp_feature_dropna[c], kde=False)
        plt.yscale('log')
        plt.legend(['train', 'test'])
        plt.grid()
        plt.show();
    except Exception as e:
        print(e)


# In[ ]:




