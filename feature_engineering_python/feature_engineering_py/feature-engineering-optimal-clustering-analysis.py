#!/usr/bin/env python
# coding: utf-8

# ## From chapter 4 - Optimal Clustering
# 
# ## of Machine Learning for Asset Managers by Marcos Lopez de Prado
# 
# <b>Book</b>: https://www.amazon.com/Machine-Learning-Managers-Elements-Quantitative/dp/1108792898 <br>
# <b>Code</b>: https://github.com/emoen/Machine-Learning-for-Asset-Managers <br>
# <b>Paper</b>: "DETECTION OF FALSE INVESTMENT STRATEGIES USING UNSUPERVISED LEARNING METHODS" - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3167017 <br>
# 
# 
# Use unsupervised learning to maximize intragroup similarities and minimize intergroup similarities. Consider matrix X of shape N x F. N objects and F features. Features are used to compute proximity(correlation, mutual information) to N objects in an NxN matrix.
# 
# There are 2 types of clustering algorithms. Partitional and hierarchical:
# 1. Connectivity: hierarchical clustering
# 2. Centroids: like k-means
# 3. Distribution: gaussians
# 4. Density: search for connected dense regions like DBSCAN, OPTICS
# 5. Subspace: modeled on two dimension, feature and observation. [Example](https://quantdare.com/biclustering-time-series/)
# 
# 
# Generating of random block correlation matrices is used to simulate instruments with correlation. The utility for doing this is in code snippet 4.3, and it uses clustering algorithms <i>optimal number of cluster</i> (ONC) defined in snippet 4.1 and 4.2, which does not need a predefined number of clusters (unlike k-means), but uses an 'elbow method' to stop adding clusters. The optimal number of clusters are achieved when there is high intra-cluster correlation and low inter-cluster correlation. The [silhouette score](https://en.wikipedia.org/wiki/Silhouette_(clustering)) is used to minimize within-group distance and maximize between-group distance. 
# 
# The code-snippets: https://github.com/emoen/Machine-Learning-for-Asset-Managers/blob/master/Machine_Learning_for_Asset_Managers/ch4_optimal_clustering.py

# ## Read data
# 
# Train and test data is taken from "LGBM & FFNN" - https://www.kaggle.com/mayangrui/lgbm-ffnn

# In[1]:


from IPython.core.display import display, HTML

import pandas as pd
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import os
import gc

from joblib import Parallel, delayed

from sklearn import preprocessing, model_selection
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt 
import seaborn as sns
import numpy.matlib


# In[2]:


train=pd.read_pickle("../input/pickled/train.pkl")
test=pd.read_pickle("../input/pickled/test.pkl")


# In[3]:


from sklearn.cluster import KMeans
# making agg features
                      
train_p = pd.read_csv('../input/optiver-realized-volatility-prediction/train.csv')
train_p = train_p.pivot(index='time_id', columns='stock_id', values='target')


# ### Install library Machine-Learning-for-Asset-Managers

# In[4]:


get_ipython().system('pip install ../input/machine-learning-for-asset-managers/')
#Installs the library: https://github.com/emoen/Machine-Learning-for-Asset-Managers


# ### Generate Random Matrix with 6 clusters
# and use ONC algorith to find the clusters

# In[5]:


from Machine_Learning_for_Asset_Managers import ch4_optimal_clustering  as oc
import matplotlib.pylab as plt
import matplotlib

from sklearn.cluster import KMeans

nCols, nBlocks = 6, 3
nObs = 8
sigma = 1.
corr0 = oc.randomBlockCorr(nCols, nBlocks)
testGetCovSub = oc.getCovSub(nObs, nCols, sigma, random_state=None) #6x6 matrix

# recreate fig 4.1 colormap of random block correlation matrix
nCols, nBlocks, minBlockSize = 30, 6, 2
print("minBlockSize"+str(minBlockSize))
corr0 = oc.randomBlockCorr(nCols, nBlocks, minBlockSize=minBlockSize) #pandas df

corr1 = oc.clusterKMeansTop(corr0) #corr0 is ground truth, corr1 is ONC

#Draw ground truth
matplotlib.pyplot.matshow(corr0) #invert y-axis to get origo at lower left corner
matplotlib.pyplot.gca().xaxis.tick_bottom()
matplotlib.pyplot.gca().invert_yaxis()
matplotlib.pyplot.colorbar()
matplotlib.pyplot.show()


# ## Apply ONC algorithm to Optiver dataset

# In[6]:


corr0 = train_p.corr()

corr1, clstrs, silh = oc.clusterKMeansTop(corr0) #corr0 is ground truth, corr1 is ONC

#Draw ground truth
matplotlib.pyplot.matshow(corr0) #invert y-axis to get origo at lower left corner
matplotlib.pyplot.gca().xaxis.tick_bottom()
matplotlib.pyplot.gca().invert_yaxis()
matplotlib.pyplot.colorbar()
matplotlib.pyplot.show()


# ### 2 Clusters found

# In[7]:


#ONC found strongest signal for 2 clusters
print("Keys:"+str(clstrs.keys()))
print("Clusters:")
print(clstrs)


# # From Chapter 2 - Marchenko-Pastur analysis - is there signal in the correlations

# The organisers says time_id has been shuffled for each stock_id. Therefore clustering on the correlation of stock_id and time_id should contain no signal. From Random Matrix theory we know that a random matrix - which is symmetric and Hermitian matrix which is positive-definite - follows the Marchenko-Pastur distribution (https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution)
# 
# We can use this to check if there is any signal in the correlation of stock_id/time_id
# 
# Lets get the eigenvalues of the correlation matrix.

# In[8]:


from Machine_Learning_for_Asset_Managers import ch2_marcenko_pastur_pdf as mp

corr0 = train_p.corr()
eigenVal, eigenVec = mp.getPCA(corr0)


# Lets get the theoretical Marchenko pastur distribution (MP-pdf) of this matrix.

# In[9]:


variance = 1.0 # assume variance of correlation matrix is 1
relation_row_col = eigenVal.shape[0]/float(eigenVal.shape[1])
number_of_samples = eigenVal.shape[1]
pdf0 = mp.mpPDF(variance, q=relation_row_col, pts=number_of_samples)


# Lets find the max eigenvalue expected from a random matrix of this dimension. Any signal would typically have larger eigenvalue than this

# In[10]:


np.max(pdf0) 


# Lets plot the pdf together with eigenvalues found from correlation matrix

# In[11]:


fig = plt.figure()
ax  = fig.add_subplot(111)
plt.plot(pdf0.keys(), pdf0, color='r', label="Marcenko-Pastur pdf")
ax.hist(np.diag(eigenVal), density = True, bins=50)


# There is one large eigenvalue. That is the market. Every stock moves to the music of the market. 
# Lets view the largest 10 eigenvalues

# In[12]:


print(eigenVal.flatten().shape)
sortedEigenV = np.sort(eigenVal.flatten())
sortedEigenV[-9:]


# There are 2 eigenvalues larger than the MP-pdf which is not strong support for claiming there is signal in the correlation. This might explain why ONC found only 2 clusters. 
# 
# Below is an example of the MP-pdf and eigenvalues found in a matrix with signal.

# ![RandomMatrixWithSignal](https://raw.githubusercontent.com/emoen/Machine-Learning-for-Asset-Managers/master/img/fig_2_3_mp_with_signal.png)
