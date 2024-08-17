#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.utils import shuffle
from sklearn.manifold import TSNE

def load_data():
    train = pd.read_csv('/kaggle/input/playground-series-s3e12/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s3e12/test.csv')
    test_id = test['id']
    
    original = pd.read_csv('/kaggle/input/kidney-stone-prediction-based-on-urine-analysis/kindey stone urine analysis.csv')

    train = pd.concat([train, original], axis =  0)
    train.drop('id', axis = 1, inplace = True)
    test.drop('id', axis = 1, inplace = True)
    train = shuffle(train).reset_index(drop = True)
    return train, test, test_id


# # My approach: two datasets  
# I've used two different feature engineering strategies, creating at the end two different datasets. 
# 
# ## First dataset  
# My strategy was based on the idea that this is a medical dataset, so there must be medical ranges for each feature to be considered healthy. After some googling, I found out that a healthy range of the urine ph is between 4.5 and 8. Therefore, values outside this range are considered abnormal.
# I created binary features from each feature to check if they are in the healthy range.  
# Then, I had another idea: this looks like the perfect dataset to include feature interactions.  
# **ph_normal_x_calc_normal** for example, this feature is 1 if ph **and** calc are normal, otherwise is 0.  
# You can imagine it as a patient who has both values in the normal range.  
# 

# In[2]:


train, test, test_id = load_data()
# Define healthy normal ranges for each feature
normal_ranges = {
    'gravity': (1.005, 1.030),
    'ph': (4.5, 8.0),
    'osmo': (500, 800),
    'cond': (50, 1500),
    'urea': (2.5, 6.7),
    'calc': (1.0, 2.5)
}

# Create new binary features
for feature, (min_value, max_value) in normal_ranges.items():
    train[f'{feature}_normal'] = train[feature].apply(lambda x: 1 if min_value <= x <= max_value else 0)
    test[f'{feature}_normal'] = test[feature].apply(lambda x: 1 if min_value <= x <= max_value else 0)

# Create new features itneractions
for feature1, feature2 in [('gravity_normal', 'ph_normal'), ('gravity_normal', 'cond_normal'), ('gravity_normal', 'calc_normal'), ('ph_normal', 'cond_normal'), ('ph_normal', 'calc_normal'), ('cond_normal', 'calc_normal')]:
    train[f'{feature1}_x_{feature2}'] = train[feature1] * train[feature2]
    test[f'{feature1}_x_{feature2}'] = test[feature1] * test[feature2]
    
train


# 

# # Second dataset
# The motivation for this dataset was very simple: feature selection. I noticed that after dropping urea and osmo, linear models were performing very well. It seems that there are clusters, as confirmed by the following visualization.   
# 
# I clipped some features, this idea wasn't mine, but it was inspired by someone in the competition. Unfortunately, I can't find the notebook anymore, but if I find it, I'll include the source.

# In[3]:


# Second dataset
train, test, test_id = load_data()
to_del=['urea', 'osmo']
train = train.drop(to_del, axis=1)
test = test.drop(to_del, axis=1)

train['ph'] = train['ph'].clip(5.4,7)
test['ph'] = test['ph'].clip(5.4,7)


train['cond'] = train['cond'].clip(16, 30)
test['cond'] = test['cond'].clip(16, 30)

y_train = train['target']
X_train = train.drop('target', axis =1)

X_train


# Let's visualize the new dataset with t-sne.

# In[4]:


import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 

data = X_train
labels = y_train

tsne = TSNE(n_components=3, random_state=0)
tsne_data = tsne.fit_transform(X_train)

import plotly.graph_objs as go 

# Create interactive 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=tsne_data[:,0],
    y=tsne_data[:,1],
    z=tsne_data[:,2],
    mode='markers',
    marker=dict(
        size=8,
        color=labels,
        colorscale='Viridis',
        opacity=0.8
    )
)])

# Add axes labels and title
fig.update_layout(
    scene=dict(
        xaxis_title='PC1',
        yaxis_title='PC2',
        zaxis_title='PC3'
    ),
    title=dict(text='3D PCA Visualization of Digits Dataset')
)

# Show the plot
fig.show()


# The double dataset feature engineering idea was winning because complex tree-based ensembles perform better on the first dataset, which has more features and interactions.
# On the other hand, they performed very bad on the second one. In this case, the best models were simple, linear models. 
# Capturing this diversity allowed to build a better ensemble in the end, which was a stacking model with 16 models.
