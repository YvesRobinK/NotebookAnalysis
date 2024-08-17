#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':13})

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import mean_absolute_error as mae

import gc

import warnings
warnings.filterwarnings("ignore")


# In[2]:


class NBConfig:
    general = {
        "seed": 2021,
        "folds": 5
    }
    method = {
        "jobs": 4, 
        "criterion": "mse",
        "samples": 0.7,
        "feat_frac": 0.6,
        "depth": 20,
        "n_trees": 222,
        "leafSize": 3,
        "costs": 0.0
    }


# In[3]:


trainpath = "../input/ventilator-pressure-prediction/train.csv"
testpath = "../input/ventilator-pressure-prediction/test.csv"
samsubpath = "../input/ventilator-pressure-prediction/sample_submission.csv"
train, test, samSub = pd.read_csv(trainpath, index_col="id"), pd.read_csv(testpath, index_col="id"), pd.read_csv(samsubpath)


# In[4]:


print("train shape is: " + str(train.shape))
print("test shape is: " + str(test.shape))


# In[5]:


train.tail()


# In[6]:


train.describe()


# # Feature Engineering

# In[7]:


train["timeDiff"] = train["time_step"].groupby(train["breath_id"]).diff(1).fillna(-1)
test["timeDiff"] = test["time_step"].groupby(test["breath_id"]).diff(1).fillna(-1)


# In[8]:


train["maxu_in"] = train[["breath_id", "u_in"]].groupby("breath_id").transform("max")["u_in"]
test["maxu_in"] = test[["breath_id", "u_in"]].groupby("breath_id").transform("max")["u_in"]


# **Define a function for flexible feature engineering**

# In[9]:


def ByBreath(method: str, DF, lags=None, center=False, fillNas=-1):

    output = pd.DataFrame()
    if center == True:
        c = "c"
    else:
        c = ""
    
    if method == "mean":
        if lags is None:
            sys.exit("specify lags")
        for l in lags:
            agg = \
            DF[["breath_id", "u_in", "u_out"]].groupby("breath_id").rolling(window=l, center=center).mean().fillna(fillNas)
            output[["{0}mu_in_l{1}".format(c, l), "{0}mu_out_l{1}".format(c, l)]] = agg[["u_in", "u_out"]]
            gc.collect()
            
    elif method == "max":
        if lags is None:
            sys.exit("specify lags")
        for l in lags:
            agg = \
            DF[["breath_id", "u_in"]].groupby("breath_id").rolling(window=l, center=center).max().fillna(fillNas)  
            output[["{0}mxu_in_l{1}".format(c, l)]] = agg[["u_in"]]
            gc.collect()
            
    elif method == "min":
        if lags is None:
            sys.exit("specify lags")
        for l in lags:
            agg = \
            DF[["breath_id", "u_in"]].groupby("breath_id").rolling(window=l, center=center).min().fillna(fillNas)  
            output[["{0}miu_in_l{1}".format(c, l)]] = agg[["u_in"]]
            gc.collect()
            
    elif method == "std":
        if lags is None:
            sys.exit("specify lags")
        for l in lags:
            agg = \
            DF[["breath_id", "u_in"]].groupby("breath_id").rolling(window=l, center=center).std().fillna(fillNas)  
            output["{0}su_in_l{1}".format(c, l)] = agg["u_in"]
            gc.collect()
            
    elif method == "shift":
        if lags is None:
            sys.exit("specify lags")
        for l in lags:
            agg = \
            DF[["breath_id", "u_in", "u_out"]].groupby("breath_id").shift(l).fillna(fillNas)  
            output[["sftu_in_l{0}".format(l), "sftu_out_l{0}".format(l)]] = agg[["u_in", "u_out"]]
            gc.collect()     
        
    elif method == "diff":
        if lags is None:
            sys.exit("specify lags")
        for l in lags:
            agg = \
            DF[["breath_id", "u_in"]].groupby("breath_id").diff(l).fillna(fillNas)  
            output["du_in_l{0}".format(l)] = agg["u_in"]
            gc.collect()  
            
    elif method == "log":
        output["lgu_in"] = np.log1p(DF["u_in"].values)
        gc.collect()  
        
    elif method == "cumsum":
            agg = \
            DF[["breath_id", "u_in", "u_out"]].groupby("breath_id").cumsum() 
            output[["csu_in", "csu_out"]] = agg[["u_in", "u_out"]]
            gc.collect()   
            
    elif method == "area":
            agg = \
            DF[["time_step", "u_in", "breath_id"]]
            agg["area"] = agg["time_step"] * agg["u_in"]
            output["area"] = agg.groupby("breath_id")["area"].cumsum()
            gc.collect()   
            
    elif method == "centering":
            agg = \
            DF[["u_in", "breath_id"]].groupby("breath_id").transform('mean')#does not aggregate like just mean()
            output["cenu_in"] = DF["u_in"] - agg["u_in"]
            gc.collect()  

    print(c + method + " created")
    return output


# In[10]:


def assignment(DF, mDF):
    DF = DF.copy()
    colNames = mDF.columns
    for n in colNames:
        DF["{0}".format(n)] = mDF["{0}".format(n)].values
    gc.collect()
    return DF


# In[11]:


train = assignment(train, ByBreath("area", train))
train = assignment(train, ByBreath("mean", train, lags=[6,9]))
train = assignment(train, ByBreath("mean", train, center=True, lags=[6]))
train = assignment(train, ByBreath("max", train, lags=[9]))
train = assignment(train, ByBreath("min", train, lags=[9]))
train = assignment(train, ByBreath("diff", train, lags=[1,2]))
train = assignment(train, ByBreath("log", train))
train = assignment(train, ByBreath("std", train, lags=[6]))
train = assignment(train, ByBreath("shift", train, lags=[-2,-1,1,2]))
train = assignment(train, ByBreath("cumsum", train))
train = assignment(train, ByBreath("centering", train))

test = assignment(test, ByBreath("area", test))
test = assignment(test, ByBreath("mean", test, lags=[6,9]))
test = assignment(test, ByBreath("mean", test, center=True, lags=[6]))
test = assignment(test, ByBreath("max", test, lags=[9]))
test = assignment(test, ByBreath("min", test, lags=[9]))
test = assignment(test, ByBreath("diff", test, lags=[1,2]))
test = assignment(test, ByBreath("log", test))
test = assignment(test, ByBreath("std", test, lags=[6]))
test = assignment(test, ByBreath("shift", test, lags=[-2,-1,1,2]))
test = assignment(test, ByBreath("cumsum", test))
test = assignment(test, ByBreath("centering", test))


# Number of unique values after feature engineering

# In[12]:


train.nunique().to_frame()


# In[13]:


test.nunique().to_frame()


# # Random Forest Approach

# In[14]:


train.reset_index(drop=True, inplace=True)
y = train.pressure
uniTarg = np.array(sorted(y.unique()))
names = [c for c in train.columns if c not in ["breath_id", "u_out", "pressure", "sftu_out_l-1", "sftu_out_l-2"]]
train, test = train[names], test[names]
gc.collect()


# In[15]:


kf = KFold(
    n_splits=NBConfig.general["folds"], 
    random_state=NBConfig.general["seed"], 
    shuffle=True
)


# In[16]:


k=0

importances = []

for train_index, test_index in kf.split(train):
    
    print("Fold: " + str(k+1))
    X_train, X_test = train.loc[train_index], train.loc[test_index]
    print("Train observations: " + str(X_train.shape[0]) + "\n" + 
          "Test observations: " + str(X_test.shape[0]))
    y_train, y_test = y[train_index], y[test_index]
    gc.collect()
    reg = RFR(
        random_state=NBConfig.general["seed"], 
        n_jobs=NBConfig.method["jobs"], 
        criterion=NBConfig.method["criterion"],
        max_samples=NBConfig.method["samples"],
        max_features=NBConfig.method["feat_frac"],
        max_depth=NBConfig.method["depth"],
        n_estimators=NBConfig.method["n_trees"],
        min_samples_leaf=NBConfig.method["leafSize"],
        ccp_alpha=NBConfig.method["costs"],
        verbose=0
    )
    
    reg.fit(X_train, y_train); gc.collect()
    
    pred_KFold = reg.predict(X_test)
    
    score = mae(pred_KFold, y_test)
    print("RF score: " + str(score))
    
    imps = reg.feature_importances_
    importances.append(imps)
    
    if k == 0:
        samSub.pressure = reg.predict(test[names]) / NBConfig.general["folds"] 
    else:
        samSub.pressure = samSub.pressure + reg.predict(test[names]) / NBConfig.general["folds"] 
    
    k+=1
    
del(train, test)


# # Post Processing and Submission

# In[17]:


impts = pd.DataFrame(importances, columns=names)
means = impts.mean(axis=0).to_frame(name="means")
stds = impts.std(axis=0)
means["stds"] = stds
means.sort_values(by="means", ascending=True, inplace=True)

fig, ax = plt.subplots(figsize=(15,15))
means.means.plot.barh(yerr=means.stds.values, ax=ax)
ax.set_title("Mean Feature Importances with Standard Deviation")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()


# In[18]:


samSub["pressure"] = samSub.pressure.map(lambda x: uniTarg[np.argmin(((uniTarg - x)**2))])


# In[19]:


samSub[["id", "pressure"]].to_csv("sampleSubmission", index=False)
samSub.head()

