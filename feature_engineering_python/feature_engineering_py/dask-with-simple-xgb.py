#!/usr/bin/env python
# coding: utf-8

# ### This dataset contains a lot of files. [Dask](https://dask.org/) is a great library to accelerate such workloads in parallel. In this notebook, we show how to use dask to engineer features in parallel and train a xgboost model.
# 
# ### On a machine with 16 cores, the feature engineering time is reduced from 1 hour to 5 minutes.

# In[1]:


from glob import glob
from collections import Counter
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
import xgboost as xgb


# In[2]:


from dataclasses import dataclass

import numpy as np


@dataclass
class ReadData:
    acce: np.ndarray
    acce_uncali: np.ndarray
    gyro: np.ndarray
    gyro_uncali: np.ndarray
    magn: np.ndarray
    magn_uncali: np.ndarray
    ahrs: np.ndarray
    wifi: np.ndarray
    ibeacon: np.ndarray
    waypoint: np.ndarray


def read_data_file(data_filename):
    acce = []
    acce_uncali = []
    gyro = []
    gyro_uncali = []
    magn = []
    magn_uncali = []
    ahrs = []
    wifi = []
    ibeacon = []
    waypoint = []

    with open(data_filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line_data in lines:
        line_data = line_data.strip()
        if not line_data or line_data[0] == '#':
            continue

        line_data = line_data.split('\t')

        if line_data[1] == 'TYPE_ACCELEROMETER':
            acce.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_ACCELEROMETER_UNCALIBRATED':
            acce_uncali.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_GYROSCOPE':
            gyro.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_GYROSCOPE_UNCALIBRATED':
            gyro_uncali.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_MAGNETIC_FIELD':
            magn.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_MAGNETIC_FIELD_UNCALIBRATED':
            magn_uncali.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_ROTATION_VECTOR':
            if len(line_data)>=5:
                ahrs.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_WIFI':
            sys_ts = line_data[0]
            ssid = line_data[2]
            bssid = line_data[3]
            rssi = line_data[4]
            lastseen_ts = line_data[6]
            wifi_data = [sys_ts, ssid, bssid, rssi, lastseen_ts]
            wifi.append(wifi_data)
            continue

        if line_data[1] == 'TYPE_BEACON':
            ts = line_data[0]
            uuid = line_data[2]
            major = line_data[3]
            minor = line_data[4]
            rssi = line_data[6]
            ibeacon_data = [ts, '_'.join([uuid, major, minor]), rssi]
            ibeacon.append(ibeacon_data)
            continue

        if line_data[1] == 'TYPE_WAYPOINT':
            waypoint.append([int(line_data[0]), float(line_data[2]), float(line_data[3])])

    acce = np.array(acce)
    acce_uncali = np.array(acce_uncali)
    gyro = np.array(gyro)
    gyro_uncali = np.array(gyro_uncali)
    magn = np.array(magn)
    magn_uncali = np.array(magn_uncali)
    ahrs = np.array(ahrs)
    wifi = np.array(wifi)
    ibeacon = np.array(ibeacon)
    waypoint = np.array(waypoint)

    return ReadData(acce, acce_uncali, gyro, gyro_uncali, magn, magn_uncali, ahrs, wifi, ibeacon, waypoint)


# In[3]:


import dask
from dask.distributed import Client, wait, LocalCluster


# We use only 2 workers on kaggle kernel. On your local machine, you could increase the number of workers/threads.

# In[4]:


client = Client(n_workers=2, 
                threads_per_worker=1)
client


# The dashboard is a great feature to monitor the progress of dask. Since dask is asynchronous, only the progress bar in the dashboard reflects the real progress.

# ### Functions

# In[5]:


def mpe(yp, y):
    e1 = (yp[:,0] - y[:,0])**2 + (yp[:,1] - y[:,1])**2
    e2 = 15*np.abs(yp[:,2] - y[:,2])
    return np.mean(e1**0.5 + e2)


# In[6]:


def get_building_floor(fname):
    xx = fname.split('/')
    return xx[-3],xx[-2]

def get_test_building(name):
    with open(name) as f:
        for c,line in enumerate(f):
            if c==1:
                x = line.split()[1].split(':')[1]
                return x  

def get_floor_target(floor):
    floor = floor.lower()
    if floor in ['bf','bm']:
        return None
    elif floor == 'b':
        return -1
    if floor.startswith('f'):
        return int(floor[1])
    elif floor.endswith('f'):
        return int(floor[0])
    elif floor.startswith('b'):
        return -int(floor[1])
    elif floor.endswith('b'):
        return -int(floor[0])
    else:
        return None
        
ACOLS = ['timestamp','x','y','z']
        
FIELDS = {
    'acce': ACOLS,
    'acce_uncali': ACOLS,
    'gyro': ACOLS,
    'gyro_uncali': ACOLS,
    'magn': ACOLS,
    'magn_uncali': ACOLS,
    'ahrs': ACOLS,
    'wifi': ['timestamp','ssid','bssid','rssi','last_timestamp'],
    'ibeacon': ['timestamp','code','rssi'],
    'waypoint': ['timestamp','x','y']
}

NFEAS = {
    'acce': 3,
    'acce_uncali': 3,
    'gyro': 3,
    'gyro_uncali': 3,
    'magn': 3,
    'magn_uncali': 3,
    'ahrs': 3,
    'wifi': 1,
    'ibeacon': 1,
    'waypoint': 3
}


# In[7]:


def build_fea_one_file(data):
    feas = []
    target = None
    for k,v in vars(data).items():
        if k == 'waypoint':
            if len(v.shape)==2 and v.shape[1] == 3:
                target = v[:,1:]
            else:
                target = None
            continue
        if k in ['wifi','ibeacon']:
            continue
        if v.shape[0] == 0:
            feas.extend([None]*NFEAS[k]*2)
            continue
        df = pd.DataFrame(v, columns=FIELDS[k])
        for col in df.columns[1:]:
            if df[col].dtype!='O' and 'time' not in col:
                feas.extend([df[col].mean(),df[col].std()])
    return np.array(feas),target

def fe(name):
    data = read_data_file(name)
    x,y = build_fea_one_file(data)
    assert len(x) == 42
    return x,y


# ### Build Features

# In[8]:


get_ipython().system('ls ../input/indoor-location-navigation')


# In[9]:


PATH = '../input/indoor-location-navigation'
train_files = glob(f'{PATH}/train/*/*/*.txt')
len(train_files)


# In[10]:


get_ipython().run_cell_magic('time', '', 'buildings = []\nfloors = []\nused = []\nfor fname in tqdm(train_files):\n    b,f = get_building_floor(fname)\n    f = get_floor_target(f)\n    if f is None:\n        continue\n    used.append(fname)\n    buildings.append(b)\n    floors.append(f)\ny = np.array(floors)\nb = np.array(buildings)\n')


# In[11]:


get_ipython().run_cell_magic('time', '', 'enc = OneHotEncoder()\nbs = enc.fit_transform(np.expand_dims(b,1))\nbs.shape\n')


# In[12]:


get_ipython().run_cell_magic('time', '', 'futures = [] # save the future since dask is lazy, otherwise nothing is executed.\nfor fname in tqdm(used):\n    f = client.submit(fe,fname) \n    futures.append(f) \n')


# In[13]:


get_ipython().run_cell_magic('time', '', 'X = [i.result() for i in futures]\nys = np.vstack([np.mean(i[1],axis=0) for i in X])\nX = np.vstack([i[0] for i in X])\nX.shape,ys.shape\n')


# In[14]:


df = pd.DataFrame(ys,columns=['w_x','w_y'])
df['building'] = b
df['floors'] = y
df.head()


# In[15]:


X = np.hstack([X,bs.toarray()])
print(X.shape)


# In[16]:


test_files = glob(f'{PATH}/test/*.txt')
len(test_files)


# In[17]:


get_ipython().run_cell_magic('time', '', 'test_b = []\nfor name in tqdm(test_files):\n    test_b.append(get_test_building(name))\ntest_b = np.array(test_b)\nbs = enc.transform(np.expand_dims(test_b,1))\n')


# In[18]:


get_ipython().run_cell_magic('time', '', 'futures = [] # save the future since dask is lazy, otherwise nothing is executed.\nfor fname in tqdm(test_files):\n    f = client.submit(fe,fname) \n    futures.append(f) \n')


# In[19]:


get_ipython().run_cell_magic('time', '', 'Xt = [i.result()[0] for i in futures]\nXt = np.vstack(Xt)\nXt = np.hstack([Xt,bs.toarray()])\nXt.shape\n')


# ### Train XGB

# In[20]:


params = {
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'eta':0.1,
        'depth':7,
        'nthread':2,
        'verbosity': 0,
    }


# In[21]:


N = 5
dtest = xgb.DMatrix(data=Xt)
ysub = np.zeros([Xt.shape[0],3])

kf = KFold(n_splits=N,shuffle=True,random_state=42)

msgs = []
for i,(train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    yps = np.zeros([X_test.shape[0],3])
    yrs = yps.copy()
    for c,col in enumerate(['w_x','w_y','floors']):
        y = df[col].values
        y_train, y_test = y[train_index], y[test_index]
        
        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        dvalid = xgb.DMatrix(data=X_test, label=y_test)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')] 

        clf = xgb.train(params, dtrain=dtrain,
                    num_boost_round=70,evals=watchlist,
                    early_stopping_rounds=10,
                    verbose_eval=100)
        yp = clf.predict(dvalid)
        yps[:,c] = yp
        yrs[:,c] = y_test
        ysub[:,c] += clf.predict(dtest)
    msg = f'Fold {i}: MPE {mpe(yps, yrs):.4f}'
    print(msg)
    msgs.append(msg)
ysub = ysub/N


# In[22]:


msgs


# In[23]:


sub = pd.read_csv(f'{PATH}/sample_submission.csv')
sub.head()


# In[24]:


sub.shape


# In[25]:


sub['site'] = sub['site_path_timestamp'].apply(lambda x: x.split('_')[0])
sub.head()


# In[26]:


test_map = {i:j for i,j in zip(test_b, test_files)}
sub['filename'] = sub['site'].apply(lambda x: test_map[x])
sub.head()


# In[27]:


ds = pd.DataFrame(ysub,columns=['x','y','floor'])
ds.head()


# In[28]:


ds['filename'] = test_files
ds.head()


# In[29]:


sub = sub.drop(['x','y','floor'],axis=1).merge(ds,on='filename',how='left')
print(sub.shape)
sub.head()


# In[30]:


for i in sub.columns:
    print(i,sub[i].isnull().sum())


# In[31]:


sub['floor'] = sub['floor'].astype('int')
sub.head()


# In[32]:


sub.drop(['site','filename'],axis=1).to_csv('submission.csv',index=False)

