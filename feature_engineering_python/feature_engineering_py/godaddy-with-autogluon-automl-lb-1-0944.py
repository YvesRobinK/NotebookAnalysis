#!/usr/bin/env python
# coding: utf-8

# # use AutoGluon for automated machine learning^W^
# In this notebook, many of the ideas are based on the [Better XGB Baseline](https://www.kaggle.com/code/titericz/better-xgb-baseline)

# In[1]:


get_ipython().system('pip install autogluon')
import os,warnings,sys
import numpy as np
import torch as tc
from tqdm import tqdm
import pandas as pd
warnings.filterwarnings("ignore")
from autogluon.tabular import TabularDataset, TabularPredictor
import matplotlib.pyplot as plt


# # Load train_data and test_data

# In[2]:


train_data = pd.read_csv('/kaggle/input/godaddy-microbusiness-density-forecasting/train.csv')
test_data = pd.read_csv('/kaggle/input/godaddy-microbusiness-density-forecasting/test.csv')

cfip_data=dict()
for id,c,s in zip(train_data["cfips"],train_data["county"],train_data["state"]):
    cfip_data[id]=(c,s)

cache_c=[]
cache_s=[]
for id in test_data["cfips"]:
    cache_c.append(cfip_data[id][0])
    cache_s.append(cfip_data[id][1])
test_data["county"]=cache_c
test_data["state"]=cache_s

train_data['istest'] = 0
test_data['istest'] = 1
raw = pd.concat((train_data, test_data)).sort_values(['cfips','row_id']).reset_index(drop=True)
del raw["active"]
raw


# In[3]:


cache = pd.to_datetime(raw["first_day_of_month"])
raw["year"]=cache.dt.year
raw["month"]=cache.dt.month

cache=raw["year"].astype("float").to_numpy()
cache=(cache-cache.min())/(cache.max()-cache.min())
raw["year"]=cache# may work, may not work

cache=raw["microbusiness_density"].to_numpy()
cache[np.isnan(cache)]=0
cache[np.isinf(cache)]=1
raw["microbusiness_density"]=cache


# # Fix microbusiness_density

# In[4]:


for o in tqdm(raw.cfips.unique()):
    indices = (raw['cfips']==o)
    tmp = raw.loc[indices].copy().reset_index(drop=True)
    var = tmp.microbusiness_density.values.copy()
    for i in range(37, 2, -1):
        thr = 0.20*np.mean(var[:i])
        difa = abs(var[i]-var[i-1])
        if (difa>=thr):
            var[:i] *= (var[i]/var[i-1])
    var[0] = var[1]*0.99
    raw.loc[indices, 'microbusiness_density'] = var


# # Missing value filling

# In[5]:


raw["time"]=-1
for name,g in raw.groupby('cfips'):
    cache=g[g["istest"]==0]["microbusiness_density"].to_numpy()
    a=np.isnan(cache.sum())
    if a:
        #28055
        #48301
        cache=cache[~np.isnan(cache)]
        cache=cache[~np.isinf(cache)]
    train_data_mean=cache.mean()
    cache=g[g["istest"]==0]["microbusiness_density"].copy()
    cache[np.isnan(cache)]=train_data_mean
    cache[np.isinf(cache)]=train_data_mean
    g.loc[g["istest"]==0,"microbusiness_density"]=cache
    
    g["time"]=list(range(len(g)))
    raw[raw["cfips"]==name]=g


# # SMAPE is a relative metric so target must be converted.

# In[6]:


a=raw.groupby('cfips')['microbusiness_density'].shift(0)
b=raw.groupby('cfips')['microbusiness_density'].shift(1)
raw['target']=a/(b+1e-5)-1

#Set upper and lower limits
n=0.2
cache=raw["target"].to_numpy()
cache[~pd.notna(cache)]=0
cache[cache>n]=n
cache[cache<-n]=-n
raw["target"]=cache


# In[7]:


raw.loc[raw["istest"]==1,"target"]=0
raw=raw[raw["time"]!=0] 

raw = raw.sort_values(['cfips','time']).reset_index(drop=True)
raw


# # Feature Engineering

# In[8]:


def build_features(data:pd.DataFrame):
    key=[]
    data["mean"]=data.groupby('cfips')["microbusiness_density"].transform('mean')
    data["std"]=data.groupby('cfips')["microbusiness_density"].transform('std')#1.1664->1.1421
    key.extend(["mean","std"])
    
    for x in range(1,5):
        str_=f"past_target_{x}"
        data[str_]=data.groupby("cfips")["target"].shift(x)
        cache=data[str_].to_numpy()
        key.append(str_)
        cache[np.isnan(cache)]=0
        data[str_]=cache
    for x in range(1,2):
        str_=f"diff_{x}"
        data[str_]=data.groupby("cfips")["target"].diff(x)
        cache=data[str_].to_numpy()
        key.append(str_)
        cache[np.isnan(cache)]=0
        data[str_]=cache
    
    """k=3
    raw[f"rooling_{k}_mean"]=raw.groupby("cfips")["target"].shift(1).transform(lambda x: x.rolling(window=k,min_periods=3).mean())
    raw[f"rooling_{k}_mean"]=raw[f"rooling_{k}_mean"].fillna(0)
    key.append(f"rooling_{k}_mean")
    raw[f"rooling_{k}_std"]=raw.groupby("cfips")["target"].shift(1).transform(lambda x: x.rolling(window=k,min_periods=3).std())
    raw[f"rooling_{k}_std"]=raw[f"rooling_{k}_std"].fillna(0)
    key.append(f"rooling_{k}_std")"""
    
    return raw,key


# In[9]:


raw,key=build_features(raw)#imporve
test_keys=key+["county","state","first_day_of_month","year","month"]
train_keys=test_keys+["target"]
train_data=raw[raw["istest"]==0]


# # Training with Autogluon

# In[10]:


label="target"
save_path = './auto_model'  # specifies folder to store trained models

t_data=train_data.sample(n=50000,random_state=0)#all119130
#         lb
#10000->1.1007
#30000->higher
#50000->1.0969
#all  ->1.1018

print("start")
#predictor = TabularPredictor(label=label, path=save_path,verbosity=2).fit(train_data,presets='good_quality')
predictor = TabularPredictor(label=label,path=save_path,verbosity=2).fit(t_data[train_keys],presets='good_quality',num_gpus=1)
#         lb
#normal:1.1007
#good  :1.0987
print("end")


# # Inference

# In[11]:


predictor = TabularPredictor.load(save_path)  # unnecessary, just demonstrates how to load previously-trained predictor from file


base=train_data[train_data["time"]==train_data["time"].max()]["microbusiness_density"].to_numpy()

test_data:pd.DataFrame=raw[raw["istest"]==1]
test_data=test_data[test_keys+["row_id","cfips","microbusiness_density","time","istest","target"]]
raw=raw[test_keys+["row_id","cfips","microbusiness_density","time","istest","target"]]

predictor.persist_models()#Save the model to memory
for i in tqdm(range(test_data["time"].min(),test_data["time"].max()+1)):
    test_data=test_data.copy(deep=True)
    x=test_data[test_data["time"]==i].copy(deep=True)
    output=predictor.predict(x[test_keys])
    
    base=(output.to_numpy()+1)*base
    test_data.loc[output.index,"microbusiness_density"]=base
    test_data.loc[output.index,"target"]=output.to_numpy()
    raw[raw["istest"]==1]=test_data
    
    raw,_=build_features(raw)
    test_data=raw[raw["istest"]==1]
    
for name,g in raw.groupby("cfips"):
    m=g["microbusiness_density"].to_numpy()
    t=g["target"].to_numpy()
    plt.plot(m)
    plt.show()
    plt.plot(t)
    plt.show()
    break


sub=test_data[["row_id","microbusiness_density"]]
sub.to_csv("./submission.csv",index=None)
sub

