#!/usr/bin/env python
# coding: utf-8

# # Non ML Approach

# ## I try to predict directory pressure using no machine learning. 
# ## Specificaly, I use first-order lag of a domain of a control engineering.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from tqdm import tqdm

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import warnings
warnings.simplefilter('ignore')


# In[2]:


train_df = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
test_df = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")


# ## Feature engineering using a first-order lag system

# In[3]:


def physics_info(df_breath):
    
    lag = 1

    inhale_ind = np.argmax(df_breath["u_out"])+lag

    df_breath["physics_info"] = 0

    # Time constant
    T = 400 * df_breath['R'].astype("float")*df_breath['C'].astype("float")

    # inhale
    exponent = (- df_breath['time_step'])/T
    factor=np.exp(exponent)
    df_breath['vf']=(df_breath['u_in_cumsum']*df_breath['R'])/factor

    inhale = (df_breath["vf"]/(450) + df_breath["intercept"]).values

    # exhale
    time_exhale_start = df_breath["time_step"].iloc[inhale_ind]
    exponent=(- (df_breath['time_step'] - time_exhale_start)) / T * 4500000
    factor=np.exp(exponent)

    df_breath["intercept"] = 0.00
    for R in df_breath["R"].unique():
        for C in df_breath["C"].unique():
            df_breath.loc[(df_breath["R"]==R) & (df_breath["C"]==C), "intercept"] = get_intercept(R, C)

    K_T = inhale[inhale_ind] - df_breath["area_devided_C"].values[0]
    exhale = K_T * (1 - factor)

    # store
    physics_info = np.zeros(80)
    physics_info[:inhale_ind] = inhale[:inhale_ind]
    physics_info[inhale_ind:] = inhale[inhale_ind] - exhale[inhale_ind:]

    return physics_info.tolist()


# In[4]:


def add_physics_info(df):
    
    _physics_info = []
    
    for i in tqdm(df["breath_id"].unique()):
        _physics_info.extend(physics_info(df.loc[df["breath_id"]==i]))
            
    return _physics_info


# In[5]:


def memory_usage_mb(df, *args, **kwargs):
    """Dataframe memory usage in MB. """
    return df.memory_usage(*args, **kwargs).sum() / 1024**2

def reduce_memory_usage(df, deep=True, verbose=True, categories=True):
    # All types that we want to change for "lighter" ones.
    # int8 and float16 are not include because we cannot reduce
    # those data types.
    # float32 is not include because float16 has too low precision.
    numeric2reduce = ["int16", "int32", "int64", "float64"]
    start_mem = 0
    if verbose:
        start_mem = memory_usage_mb(df, deep=deep)

    for col, col_type in df.dtypes.iteritems():
        best_type = None
        if col_type == "object":
            df[col] = df[col].astype("category")
            best_type = "category"
        elif col_type in numeric2reduce:
            downcast = "integer" if "int" in str(col_type) else "float"
            df[col] = pd.to_numeric(df[col], downcast=downcast)
            best_type = df[col].dtype.name
        # Log the conversion performed.
        if verbose and best_type is not None and best_type != str(col_type):
            print(f"Column '{col}' converted from {col_type} to {best_type}")

    if verbose:
        end_mem = memory_usage_mb(df, deep=deep)
        diff_mem = start_mem - end_mem
        percent_mem = 100 * diff_mem / start_mem
        print(f"Memory usage decreased from"
              f" {start_mem:.2f}MB to {end_mem:.2f}MB"
              f" ({diff_mem:.2f}MB, {percent_mem:.2f}% reduction)")
    
    return df

get_intercept = lambda R, C : train_df.loc[train_df["time_step"]==0].groupby(["R", "C"])["pressure"].mean().loc[(R, C)]


def add_features(df):
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    df['time_step_cumsum'] = df.groupby(['breath_id'])['time_step'].cumsum()
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    print("Step-1...Completed")
    df = reduce_memory_usage(df)
    
    # my features
    ## TV / C
    area_devided_C = df.loc[(df["u_out"]==0), ["breath_id", "area"]].groupby("breath_id").max().values / \
        df.loc[(df["u_out"]==0), ["breath_id", "C"]].groupby("breath_id").mean().values
    _area_devided_C = np.repeat(area_devided_C, 80, axis=1).flatten()
    df["area_devided_C"] = _area_devided_C

    df["intercept"] = 0.00
    for R in df["R"].unique():
        for C in df["C"].unique():
            df.loc[(df["R"]==R) & (df["C"]==C), "intercept"] = get_intercept(R, C)
    
    df["predicted_pressure_by_physics"] = add_physics_info(df)
    
    print("Step-1.5(My-Features)...Completed")
    
    
    return df


# In[6]:


print("Train data...\n")
train = add_features(train_df)
test = add_features(test_df)


# In[7]:


train["mae_pp"] = np.abs(train["pressure"]-train["predicted_pressure_by_physics"])
error=train.loc[:, ["breath_id", "mae_pp"]].groupby("breath_id").mean()
min_error_index = error.sort_values(by="mae_pp").index.values


# ## good results

# In[8]:


for _id in min_error_index[:10]:

    fig, ax1 = plt.subplots(figsize = (12, 8))

    breath_1 = train.loc[train['breath_id'] == _id]

    ax2 = ax1.twinx()
    plt.title(f"breath_id={_id}")
    ax1.plot(breath_1['time_step'], breath_1['pressure'], 'r-', label='pressure')
    ax1.plot(breath_1['time_step'], breath_1['u_in'], 'g-', label='u_in')
    ax2.plot(breath_1['time_step'], breath_1['u_out'], 'b-', label='u_out')
    ax1.plot(breath_1['time_step'], breath_1["predicted_pressure_by_physics"], 'k--', label='ppbp')

    ax1.set_xlabel('Timestep')

    ax1.legend(loc=(1.1, 0.8))
    ax2.legend(loc=(1.1, 0.7))
    plt.show()


# ## bad results

# In[9]:


for _id in min_error_index[-10:]:

    fig, ax1 = plt.subplots(figsize = (12, 8))

    breath_1 = train.loc[train['breath_id'] == _id]

    ax2 = ax1.twinx()
    plt.title(f"breath_id={_id}")
    ax1.plot(breath_1['time_step'], breath_1['pressure'], 'r-', label='pressure')
    ax1.plot(breath_1['time_step'], breath_1['u_in'], 'g-', label='u_in')
    ax2.plot(breath_1['time_step'], breath_1['u_out'], 'b-', label='u_out')
    ax1.plot(breath_1['time_step'], breath_1["predicted_pressure_by_physics"], 'k--', label='ppbp')

    ax1.set_xlabel('Timestep')

    ax1.legend(loc=(1.1, 0.8))
    ax2.legend(loc=(1.1, 0.7))
    plt.show()


# In[10]:


train["predicted_pressure_by_physics"].to_csv("train_predictd_pressure_by_physics.csv", index=False)


# In[11]:


sub = pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')
sub['pressure'] = test["predicted_pressure_by_physics"]
sub.to_csv('submission.csv', index=False)
sub.head(5)

