#!/usr/bin/env python
# coding: utf-8

# # Dynamic Time Warping for Classification, Feature Engineering and Clustering
# 
# Calculation of distance between time series is often difficult as time series might be similar but shifted one from the other. Dynamic time warping aims to correct that. The general idea is that, instead of calculating the euclidian distance on vertical differences, the difference are calculated along inclined lines, as shown below (image from wikipedia).
# 
# <p><a href="https://commons.wikimedia.org/wiki/File:Dynamic_time_warping.png#/media/File:Dynamic_time_warping.png"><img src="https://upload.wikimedia.org/wikipedia/commons/a/ab/Dynamic_time_warping.png" alt="Dynamic time warping.png"></a><br>
# 
# A lot of the code come from Alex Minnaar [Blog Post](http://alexminnaar.com/2014/04/16/Time-Series-Classification-and-Clustering-with-Python.html). I've done some rework, some adaptation to better match the competition data and some benchmarking to better illustrate the interest of Dynamic Time Warping.

# # Imports

# In[1]:


import optuna

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.model_selection import train_test_split, GroupKFold, KFold

from IPython.display import display

import pickle

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# # Data Preparation

# In[2]:


DEBUG = True

if ~DEBUG:
    warnings.filterwarnings("ignore")

dict_types = {
'id': np.int32,
'breath_id': np.int32,
'R': np.int8,
'C': np.int8,
'time_step': np.float32,
'u_in': np.float32,
'u_out': np.int8, #np.bool ?
'pressure': np.float32,
} 

train = pd.read_csv('../input/ventilator-pressure-prediction/train.csv', dtype=dict_types)
test = pd.read_csv('../input/ventilator-pressure-prediction/test.csv', dtype=dict_types)

submission = pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')

all_pressure = np.sort(train.pressure.unique())
PRESSURE_MIN = all_pressure[0]
PRESSURE_MAX = all_pressure[-1]
PRESSURE_STEP = (all_pressure[1] - all_pressure[0])

if DEBUG:
    train = train[:80*1000]


# In[3]:


n_train = int(train.shape[0]/80)
train['time_id'] = [i for j in range(n_train) for i in range(80)]

n_test = int(test.shape[0]/80)
test['time_id'] = [i for j in range(n_test) for i in range(80)]

train_pivot = train.pivot(index='breath_id', columns='time_id', values='u_in')
test_pivot = test.pivot(index='breath_id', columns='time_id', values='u_in')

train_pivot_1000 = train_pivot[:999]
test_pivot_1000 = test_pivot[:999]


# # Euclidian distance

# Standard distance, doesn't account for time shift, but is rather fast.

# In[4]:


ts1 = train_pivot.iloc[0]
ts2 = train_pivot.iloc[1]


# In[5]:


def euclid_dist(t1,t2):
    return np.sqrt(sum((t1-t2)**2))

print(euclid_dist(ts1,ts2))


# In[6]:


print(euclid_dist(ts1,ts1))
print(euclid_dist(ts1,ts1.shift().fillna(0)))
print(euclid_dist(ts1,ts1.shift().shift().fillna(0)))


# In[7]:


get_ipython().run_cell_magic('time', '', '\nfor i in range(1000):\n    euclid_dist(ts1,ts2)\n')


# # Dynamic Time Warping Distance
# 
# DTW account for shift (but is way slower)

# In[8]:


def DTWDistance(s1, s2):
    DTW={}
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
    return np.sqrt(DTW[len(s1)-1, len(s2)-1])

print(DTWDistance(ts1,ts2))


# In[9]:


print(DTWDistance(ts1,ts1))
print(DTWDistance(ts1,ts1.shift().fillna(0)))
print(DTWDistance(ts1,ts1.shift().shift().fillna(0)))


# In[10]:


get_ipython().run_cell_magic('time', '', '\nfor i in range(1000):\n    DTWDistance(ts1,ts2)\n')


# # Speeding up DTW
# 
# DTW need to be speeded up to be exploitable. One way to do this is limit the delay between series to avoid comparison with data point that are too far apart. This can be achieved with a windows w.

# In[11]:


def DTWDistance(s1, s2,w=10):
    DTW={}
    w = max(w, abs(len(s1)-len(s2)))
    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist = (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
    return np.sqrt(DTW[len(s1)-1, len(s2)-1])

DTWDistance(ts1,ts2)


# In[12]:


get_ipython().run_cell_magic('time', '', '\nfor i in range(1000):\n    DTWDistance(ts1,ts2, 5)\n')


# # Keogh Lower Bound
# 
# The DTW calculation being $O(n^2)$ in complexity, a better approach is to consider an approximation. A lower bound for the DTW distance was discovered by Dr. Aemon Keogh. This lower bound is $O(n)$ in complexity and thus help avoid a ton of $O(n^2)$. Say you want to find the closest time serie to one instance. You can loop trough all candidates and check the lower bound. If the lower bound for a new candidate isn't below the lowest distance for your current best you can skip the whole DTW calculation.

# In[13]:


def LB_Keogh(s1,s2,r):
    LB_sum=0
    for ind,i in enumerate(s1):
        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2
    return np.sqrt(LB_sum)

print(LB_Keogh(ts1,ts2,5))


# In[14]:


get_ipython().run_cell_magic('time', '', '\nfor i in range(1000):\n    LB_Keogh(ts1.values,ts2.values, 5)\n')


# So we are down to around 1ms for two time series comparison.

# # Classification and NN Feature Engineering
# 
# Short implementation of 1NN for demonstration. Might be better not to perform the full $n^2$ comparisons. This would allow for Classification (outputing the nearest neighbor prediction) or Feature engineering (outputing feature of the nearest neighbor).

# In[15]:


from sklearn.metrics import classification_report

def knn(train,test,w):
    preds=[]
    for ind,i in enumerate(test):
        min_dist=float('inf')
        closest_seq=[]
        #print ind
        for indj, j in enumerate(train):
            if LB_Keogh(i[:-1],j[:-1],w)<min_dist:
                dist=DTWDistance(i[:-1],j[:-1],w)
                if dist<min_dist:
                    min_dist=dist
                    closest_seq=indj
        preds.append(closest_seq)
    return preds


# In[16]:


get_ipython().run_cell_magic('time', '', 'closest = knn(train_pivot_1000.values,test_pivot[:10].values,5)\n')


# This seems too slow for the current competition. I am currently looking for faster implementations.

# # Clustering with DTW
# 
# Similarly short implementation for demonstration. Clustering has a lower complexity (number of time series x number of clusters). So it might directly be usefull.

# In[17]:


import random

def k_means_clust(data,num_clust,num_iter,w=5):
    centroids=random.sample(data,num_clust)
    counter=0
    for n in range(num_iter):
        counter+=1
        print(counter)
        assignments={}
        #assign data points to clusters
        for ind,i in enumerate(data):
            min_dist=float('inf')
            closest_clust=None
            for c_ind,j in enumerate(centroids):
                if LB_Keogh(i,j,5)<min_dist:
                    cur_dist=DTWDistance(i,j,w)
                    if cur_dist<min_dist:
                        min_dist=cur_dist
                        closest_clust=c_ind
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust]=[]
    
        #recalculate centroids of clusters
        for key in assignments:
            clust_sum=0
            for k in assignments[key]:
                clust_sum=clust_sum+np.array(data[k])
            centroids[key]=[m/len(assignments[key]) for m in clust_sum]
    
    return centroids, assignments


# In[18]:


get_ipython().run_cell_magic('time', '', 'centroids, assignments = k_means_clust(train_pivot_1000.values.tolist(),num_clust=10,num_iter=5,w=5)\n')


# We can look at centroids:

# In[19]:


plt.plot(np.array(centroids).transpose());


# In[20]:


def get_nearest_centroid(ts):
    cluster = -1
    dist = np.inf
    for i in range(len(centroids)):
        if LB_Keogh(ts,centroids[i],5)<dist:
            dist_c = DTWDistance(ts,centroids[i],5)
            if dist_c < dist:
                dist = dist_c.copy()
                cluster = i
    return cluster


# In[21]:


get_ipython().run_cell_magic('time', '', '\ntest_assignements = test_pivot_1000.transpose().apply(get_nearest_centroid)\n')


# In[22]:


with open('train_DTW_clust.pkl', 'wb') as handle:
    pickle.dump(assignments, handle)
    
with open('test_DTW_clust.pkl', 'wb') as handle:
    pickle.dump(test_assignements, handle)


# # Comparison with kmeans

# In[23]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10, random_state=0).fit(train_pivot)

preds = kmeans.transform(test_pivot)

plt.plot(kmeans.cluster_centers_.transpose());


# In[24]:


with open('train_kmeans_clust.pkl', 'wb') as handle:
    pickle.dump(kmeans.labels_, handle)
    
with open('test_kmeans_clust.pkl', 'wb') as handle:
    pickle.dump(preds, handle)

