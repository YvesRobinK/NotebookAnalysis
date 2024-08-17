#!/usr/bin/env python
# coding: utf-8

# # Jane Street "features.csv" hierarchical clustering
# 
# Update of [this notebook](https://www.kaggle.com/lucasmorin/jane-street-feature-tags-trippy) that showed the correlation matrices (for features and tags) to be trippy. However I show that this trippy aspect is in part due to the ordering the feature. Once you reorder the features you get something that look more realistic. 

# In[1]:


# Import data & libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.max_columns = None

train = pd.read_csv('/kaggle/input/jane-street-market-prediction/train.csv')
feat = pd.read_csv('/kaggle/input/jane-street-market-prediction/features.csv')


# ## Correlation Matrix
# 
# We start by plotting the correlation matrix. A diverging colormap is better to visualise the extreme as being similar (a correlation of -1 is a correlation of 1 with the opposite feature).

# In[2]:


from matplotlib import pyplot as plt
plt.figure(figsize=(10,10))

corr_mat = train.corr()
plt.imshow(corr_mat, cmap='Spectral')
plt.show()


# Nothing unusual for market data.

# # Tag Correlation Matrix
# 
# Shows the correlation between each **feature**, with respect to the True/False values for each tag.

# In[3]:


flip = feat.drop(columns=['feature'])
flip = flip.transpose()
plt.figure(figsize=(10,10))

flip_corr_mat = flip.corr()

plt.imshow(flip_corr_mat, cmap='Spectral')
plt.show()


# # Woah! Looks like there are some interesting patterns. Why do you think this is?
# 
# This is the orginial question asked by the author of the original notebook. It turns out that the structures we can see are partly due to feature ordering.
# So the question is what could be a better ordering ? The answer is given by hierarchical clustering, i.e. using the correlation matrix to look what feature ressemble each-other the most.

# ## Reordering Features using hierarchical clustering 

# # Hierachical Clustering on correlation matrix

# In[4]:


from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

import scipy.cluster.hierarchy as sch

plt.figure(figsize=(10,10))

t = 0.5

pdist = sch.distance.pdist(corr_mat)
linkage = sch.linkage(pdist, method='complete')

# Calculate the cluster
labels = fcluster(linkage, t,'distance')

# Keep the indices to sort labels
labels_order = np.argsort(labels)

corr_mat_reordered = corr_mat.sort_index(0,labels_order).sort_index(1,labels_order)

plt.imshow(corr_mat_reordered,cmap='Spectral')
plt.show()


# It appears that appart some permutation of the main cluster nothing is really changing. It is entirely possible the jane-street team performed as similar clustering before handing the data.
# 
# This approach allows to plot dendograms for more advanced clustering (here with 3 clusters):

# In[5]:


plt.figure(figsize=(12,5))

dendrogram(linkage, labels=train.columns, orientation='top',leaf_rotation=90);


# Below the same work but with the tag. Matrix is not changing, but the dendogram show different level of possible clustering.

# In[6]:


plt.figure(figsize=(10,10))

t = 0.5

flip_corr_mat = flip_corr_mat.iloc[1:].drop(flip_corr_mat.columns[[0]],axis=1)

pdist = sch.distance.pdist(flip_corr_mat)
linkage = sch.linkage(pdist, method='complete')

# Calculate the cluster
labels = fcluster(linkage, t,'distance')

# Keep the indices to sort labels
labels_order = np.argsort(labels)

flip_corr_mat_reordered = flip_corr_mat.sort_index(0,labels_order).sort_index(1,labels_order)

plt.imshow(flip_corr_mat_reordered,cmap='Spectral')
plt.show()


# In[7]:


plt.figure(figsize=(12,5))

dendrogram(linkage, labels=train.columns, orientation='top',leaf_rotation=90);


# In[ ]:




