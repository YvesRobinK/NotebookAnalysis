#!/usr/bin/env python
# coding: utf-8

# # S3E9 Feature Engineering
# Predicting strength of concrete
# 
# [Using lessons learnt from the Kaggle Learn course](https://www.kaggle.com/code/ryanholbrook/feature-engineering-for-house-prices)

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression

import optuna


# In[2]:


train = pd.read_csv('/kaggle/input/playground-series-s3e9/train.csv', index_col=0)
test = pd.read_csv('/kaggle/input/playground-series-s3e9/test.csv', index_col=0)


# In[3]:


print('Shape of train dataset:', train.shape)
print('Shape of test dataset:', test.shape)
train.head()


# ## Short EDA

# In[4]:


train.describe()


# In[5]:


train.dtypes


# In[6]:


print('Any missing data?:',train.isna().any().any())


# In[7]:


sns.heatmap(train.corr(), annot=True)


# In[8]:


# sns.pairplot(train)


# ## Baseline model

# In[9]:


def score_dataset(X, y, model=XGBRegressor()):
    score = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    score = np.sqrt(-1 * score.mean())
    return score


# In[10]:


X = train.copy()
y = X.pop("Strength")
baseline_score = score_dataset(X, y)
print(f"Baseline RMSE score: {baseline_score:.5f}")


# ## Mutual Information Scores

# In[11]:


def make_mi_scores(X, y):
    X = X.copy()
    mi_scores = mutual_info_regression(X, y, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


# In[12]:


X = train.copy()
y = X.pop("Strength")

mi_scores = make_mi_scores(X, y)
mi_scores


# None of them have very low mutual information scores, so let's keep all of them.

# In[13]:


plot_mi_scores(mi_scores)


# ### New features

# In[14]:


features = list(train.columns)[:-1]
features


# Trying out some combinations at random

# In[15]:


def transforms(df):
    X = pd.DataFrame()
    
    # Mathematical Transforms - I found that these three worked
    X['CementToCoarse'] = df.CementComponent / (df.CoarseAggregateComponent + 1e-6)
    X['CementToWater'] = df.FlyAshComponent / (df.WaterComponent + 1e-6)
    X['CoarseMinusFine'] = df.CoarseAggregateComponent - df.FineAggregateComponent
    
    # Group Transforms - Choosing features with high correlation
    X['SuperplasticizerWaterMedian'] = df.groupby('SuperplasticizerComponent')['FlyAshComponent'].transform("median")

    return X


# ## K-Means, PCA

# In[16]:


def cluster_labels(df, features=features, n_clusters=10):
    X = df[features]
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0)
    X_new = pd.DataFrame()
    X_new["Cluster"] = kmeans.fit_predict(X_scaled)
    X_new["Cluster"] = X_new["Cluster"]
    return X_new


def cluster_distance(df, features=features, n_clusters=10):
    X = df[features]
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0)
    X_cd = kmeans.fit_transform(X_scaled)
    # Label features and join to dataset
    X_cd = pd.DataFrame(
        X_cd, columns=[f"Centroid_{i}" for i in range(X_cd.shape[1])]
    )
    return X_cd


# In[17]:


def apply_pca(X, standardize=True):
    # Standardize
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Create principal components
    pca = PCA()
    X_pca = pca.fit_transform(X)
    # Convert to dataframe
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    # Create loadings
    loadings = pd.DataFrame(
        pca.components_.T,  # transpose the matrix of loadings
        columns=component_names,  # so the columns are the principal components
        index=X.columns,  # and the rows are the original features
    )
    return pca, X_pca, loadings


# In[18]:


pca, X_pca, loadings = apply_pca(X)
loadings


# Contrast between BlastFurnaceSlag and Cement Component in PC2

# In[19]:


def from_pca(df):
    X = pd.DataFrame()
    X["BlastToCement"] = df.BlastFurnaceSlag / (df.CementComponent + 1e-6)
    return X

def pca_components(df, features=features):
    X = df.loc[:, features]
    _, X_pca, _ = apply_pca(X)
    return X_pca


# In[20]:


# mi_scores = make_mi_scores(X_pca, y)
# mi_scores


# ## Putting it all altogether and testing which ones work
# The ones that are commented out don't seem to improve the score.

# In[21]:


def create_features(df, df_test=None):
    X = df.copy()
    y = X.pop('Strength')
    
    # Combine splits
    if df_test is not None:
        X_test = df_test.copy()
#         X_test.pop('Strength')
        X = pd.concat([X, X_test])
    
    # Transformations
    X = X.join(transforms(X))
    
    # Add features from KMeans
#     X = X.join(cluster_labels(X))
#     X = X.join(cluster_distance(X))
    
    # Add features from PCA
    X = X.join(from_pca(X))
#     X = X.join(pca_components(X))
    
    # Reform splits
    if df_test is not None:
        X_test = X.loc[df_test.index, :]
        X = X.drop(df_test.index)
        
    # Returns
    if df_test is not None:
        return X, X_test
    return X


# In[22]:


X_train = create_features(train)
y_train = train.loc[:, 'Strength']

score = score_dataset(X_train, y_train)
print(f"Baseline RMSE score: {baseline_score:.5f}")
print(f"Score after feature engineering: {score:.5f}")


# ## Hyperparameter Tuning

# In[23]:


def objective(trial):
    xgb_params = dict(
        max_depth=trial.suggest_int("max_depth", 2, 10),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        n_estimators=trial.suggest_int("n_estimators", 1000, 8000),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 1.0),
        subsample=trial.suggest_float("subsample", 0.2, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1e2, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 1e2, log=True),
    )
    xgb = XGBRegressor(**xgb_params)
    return score_dataset(X_train, y_train, xgb)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)
xgb_params = study.best_params


# ## Training Model and Making Predictions

# In[24]:


X_train, X_test = create_features(train, test)
y_train = train.loc[:, 'Strength']


# ### Comparing the tuned model to the untuned one

# In[25]:


xgb = XGBRegressor(**xgb_params, tree_method='gpu_hist')
print('Score of untuned model:', score)
print('Score of tuned model:', score_dataset(X_train, y_train, xgb))


# ### Making Predictions

# In[26]:


xgb.fit(X_train, y_train)
predictions = xgb.predict(X_test)
assert len(predictions) == len(test)


# In[27]:


output = pd.DataFrame({'Id': test.index, 'Strength': predictions})
output.to_csv('submission.csv', index=False)
print('Successful')

