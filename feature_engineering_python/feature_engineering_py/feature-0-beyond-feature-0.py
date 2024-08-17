#!/usr/bin/env python
# coding: utf-8

# # Feature-0 Exploration
# As [mentioned previously](https://www.kaggle.com/c/jane-street-market-prediction/discussion/199462), feature 0 seems to be somewhat different from the other features, in that it's a binary indicator of something. At the same time, [another notebook](https://www.kaggle.com/hkailee/eda-lung-shape-umap-clusters-comparison) of the data showed a lung-shaped UMAP of all the features, indicating two major "lungs"/clusters, with potentially some internal structure. 
# 
# ## Experiment 1: TriMap of features 1-130
# What I wanted to check out below is whether if we run dimensionality reduction on all features **except feature 0**, in order to find the two major clusters as from the lung UMAP, will those two clusters correspond to the values of feature 0. I'll use TriMAP instead of UMAP for my investigation, but I expect we'll see similar clusters in the TriMAP space.

# In[1]:


get_ipython().system('pip install trimap --quiet | cat')


# In[2]:


from typing import List

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import trimap

# Read the data
train = pd.read_csv('/kaggle/input/jane-street-market-prediction/train.csv')
print(f">> Done loading data. Shape: {train.shape}")

# Subset the data for processing speed
train = train.fillna(0).sample(frac=0.1, random_state=2020)

# Select all features, except feature_0
FEATURES = [c for c in train.columns if 'feature' in c and c != 'feature_0']

# Scale the features in the data
means = np.nanmean(train[FEATURES], axis=0)
stds = np.nanstd(train[FEATURES], axis=0)
train[FEATURES] = (train[FEATURES] - means) / stds
print(">> Done scaling data")

# Split up the data
X = train[FEATURES]
y = train['feature_0']


# In[3]:


# Create TriMap embeddings
trimap_embedding = trimap.TRIMAP(
  n_iters=1000,
  apply_pca=True,
  weight_adj=50
).fit_transform(X.values)
print(">> Done with TriMap embeddings")

# Put embeddings into dataframe
embedding_df = pd.DataFrame({
    'Component 1': trimap_embedding[:, 0], 
    'Component 2': trimap_embedding[:, 1],
    'Feature 0': train.feature_0.astype(str),
})

# Visualize the embeddings in 2D plot
fig = px.scatter(
    embedding_df, 
    x='Component 1', y='Component 2', 
    color='Feature 0',
    opacity=0.8
)
fig.update_traces(
    marker=dict(size=5, line=dict(width=1, color='DarkSlateGrey')),
    selector=dict(mode='markers')
)    
fig.update_layout(title={
    'text': f'TriMAP Embedding on Features 1-{len(FEATURES)+1}',
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top'}
)
fig.show()


# **My Take:** As expected the features 1-130 are split into two different distributions, and indeed these match perfectly with `feature_0`. Whether that indicates bid/ask, long/short, or something else I don't know enough about finance to say, but it's clear that the value of `feature_0` influences the values of some if not all other features in the dataset.
# 
# ## Experiment 2: Features related with Feature 0
# Clearly `feature_0` splits up the rest of the features in two data distributions, but the plot only tells so much. To investigate a little further what effect `feature_0` has on the other features, I here try to solve the reverse problem of predicting `feature_0` based on features 1-130. I'll then remove the single most important feature (based off feature importance), and again check how well the remaining features can predict `feature_0`. By iteratively removing the most important feature, we can get a sense of how many features are required for predicting `feature_0`, and thereby how many features are 'related' to `feature_0`.
# 
# The reason for this approach rather than just looking at correlations between `feature_0` and `feature_x` is that we also want to consider possible non-linear effects, i.e. `feature_i` through `feature_j` together might in conjunction with each other be able to predict `feature_0`, but not by themselves.

# In[4]:


import lightgbm as lgb
from sklearn.model_selection import cross_val_score

# Reset X & y, just in case I'm running this cell multiple times
# Only run on small subset for processing speed (I checked with 
# larger sample, and results are the same)
subset = train.sample(2000, random_state=2020)
X = subset[FEATURES]
y = subset['feature_0']

# Iteratively remove most important feature & measure AUC score
i, current_auc = 0, 1.0
history = []
while current_auc > 0.5 and len(X.columns) > 1:
    
    # Calculate CV scores
    model = lgb.LGBMRegressor(n_estimators=100, n_jobs=1)
    scores = cross_val_score(model, X, y, scoring='roc_auc', n_jobs=-1)
    current_auc = scores.mean()
    
    # Get the most important feature for the model
    model = lgb.LGBMRegressor(n_estimators=100)
    model.fit(X, y)

    # Remove the best column
    best_column = X.columns[np.argmax(model.feature_importances_)]
    history.append((f'Iteration {i} - {best_column}', scores.mean()))
    X = X.drop([best_column], axis=1)
    
    #print(f">> Current AUC: {current_auc:.2f} -> Removing: {best_column}. X.shape: {X.shape}")
    i += 1
    
# Put history into a dataframe
result = pd.DataFrame(history, columns=['Removed Feature', 'Resulting AUC'])

# Create a plot 
fig = px.bar(result, x='Removed Feature', y='Resulting AUC', title='AUC as a function of removed best feature')

# Rotate tick labels & set y-axis
fig.update_layout(xaxis=dict(tickangle=-90))   
fig.update_yaxes(range=[0.45, 1])


# **My take:** We can remove *a lot* of the most important features, and still be able to perfectly classify `feature_0` - it is not just a distribution of a single feature that is dependent on the value of `feature_0`; rather it is the distribution of a lot of the features that is linked to the value of `feature_0`.
# 
# ## Experiment 3: Relation with each other feature
# In experiment 2 I iteratively removed the most predictive feature for `feature_0` from the other features to get an idea about the minimal set of features that together are related to `feature_0`. In this experiment I'll do it the other way around and see which features by themself are enough to predict the value of `feature_0`.

# In[5]:


import lightgbm as lgb
from sklearn.model_selection import cross_val_score

# Reset X & y, just in case I'm running this cell multiple times
# Only run on small subset for processing speed (I checked with 
# larger sample, and results are the same)
subset = train.sample(5000, random_state=2020)
y = subset['feature_0']

# Go through each feature & fit a model for feature_0
history = []
for i, feature in enumerate(FEATURES):
    
    # Define X as the single feature 
    X = subset[[feature]]
    
    # Calculate CV scores
    model = lgb.LGBMRegressor(n_estimators=100, n_jobs=1)
    scores = cross_val_score(model, X, y, scoring='roc_auc', n_jobs=-1)
    history.append((i+1, scores.mean()))
    
# Put history into a dataframe
result = pd.DataFrame(history, columns=['Feature ID', 'ROC AUC'])

# Create a plot 
fig = px.bar(
    result, 
    x='Feature ID', 
    y='ROC AUC', 
    title='Predicting feature_0 using solo other features'
)

# Rotate tick labels & set y-axis
fig.update_layout(xaxis=dict(tickangle=-90))   
fig.update_yaxes(range=[0.45, 1])


# For good measure, let's see what the AUC is of a model where we use all features with a score < 0.6 when used by themselves.

# In[6]:


# Get all features with an individual score < 0.6
bad_features = [f'feature_{i}' for i, auc in history if auc < 0.6]

# Get the subset of data with these "poor" features
X = subset[bad_features]
    
# Calculate a CV score
model = lgb.LGBMRegressor(n_estimators=100, n_jobs=1)
score = cross_val_score(model, X, y, scoring='roc_auc', n_jobs=-1)
print(f">> AUC score of poor features combined: {score.mean():.2f}")


# **My take:** This looks very interesting - following patterns occur:
# 
# * Pattern 1: Features *17 through 40* are extremely good at separating `feature_0`. Looking at the tags file, this does not seem to match clearly any of the given tags :/ 
# 
# * Pattern 2: Features 65-68 seem to have some relation to `feature_0`. None of these feature have any tag.
# 
# * Pattern 3: Features 72-106 seemt to have a repeating pattern of relation to `feature_0` which matches the the frequency (6) of a similar pattern seen in 6 tags for the same feature (tags 0-5), however in the tags file the pattern extends all the way to feature 118, so it's not a complete match. 
# 
# * Pattern 4: Even though the figure of model scores based on individual features indicate that some features by themselves are not related to `feature_0`, when all these "poor" features are combined, they are able to classify `feature_0` very well, indicating that the value of `feature_0` truly influences the distribution of a lot of the other features.
# 
# ## Experiment 4: Inspecting feature patterns
# With the different patterns identified above, let's look at the actual values of the features as split by `feature_0`, to see if we can actually observe the differences in distribution.
# 
# ### Pattern 1 - Good Predictors

# In[7]:


def plotFeatureSplits(df: pd.DataFrame, feature_list: List[int]) -> None:
    for i in feature_list:

        # Create a plot with original timeseries, and split by feature 0
        _, axes = plt.subplots(1, 2, figsize=(20, 5))

        # Original timeseries
        df[f'feature_{i}'].plot(ax=axes[0])
        axes[0].set_title(f'Feature {i}')
        axes[0].set_ylabel(f'Feature {i}')
        axes[0].set_xlabel(f'Trade ID')

        # Plot by feature 0 split
        df.groupby('feature_0')[f'feature_{i}'].plot(ax=axes[1])
        axes[1].set_title(f'Feature {i}, split by feature_0')
        axes[1].set_ylabel(f'Feature {i}')
        axes[1].set_xlabel(f'Trade ID')

        # Show figure with legend
        plt.legend()    
        plt.show()
        
# Get the first 10k rows, which have not be
ordered_subset = pd.read_csv('/kaggle/input/jane-street-market-prediction/train.csv', nrows=10000)


# In[8]:


# Show pattern 1, features 17-40
plotFeatureSplits(ordered_subset, np.arange(17, 41))


# Cool - makes sense that these features split the data perfectly. Seems like an opportunity for some feature engineering here as well!
# 
# ### Pattern 2 - Weak Predictors

# In[9]:


# Show pattern 2, features 65-68
plotFeatureSplits(ordered_subset, np.arange(65, 69))


# Again this makes sense, seems like `feature_0 = 1` is generally a bit higher than the signal for `feature_0 = -1`.
# 
# ### Pattern 3 - Various strength predictors

# In[10]:


# Show pattern 3
plotFeatureSplits(ordered_subset, np.arange(72,107))


# This also makes sense, it's clear that some features such as 73 split the data into plus/minus almost perfectly, while others are a lot more subtle.
