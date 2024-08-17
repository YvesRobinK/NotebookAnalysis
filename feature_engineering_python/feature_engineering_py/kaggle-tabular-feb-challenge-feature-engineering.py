#!/usr/bin/env python
# coding: utf-8

# # Kaggle Tabular Feb Challenge - Feature Engineering

# In this notebook, I will explore various feature engineering techniques to improve the model performance.

# ## Data Import

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from category_encoders import MEstimateEncoder
import optuna


# In[2]:


def read_data(data_dir):
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'), index_col='id')
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'), index_col='id')
    sample_submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'), index_col='id')
    return train, test, sample_submission


# In[3]:


DAT_DIR = '../input/tabular-playground-series-feb-2021'
train, test, sample_submission = read_data(DAT_DIR)


# ## EDA

# First let's examine the variable's influnce on the target.

# In[4]:


disc_features = [train[c].dtype == 'object' for c in train.drop('target', axis=1).columns]
cat_cols = train.select_dtypes(include='object')
cat_cols


# In[5]:


for c in cat_cols:
    enc = LabelEncoder()
    train[c] = enc.fit_transform(train[c])
    test[c] = enc.transform(test[c])


# In[6]:


X_train = train.drop('target', axis=1)
y_train = train.target


# We also re-introduce utility functions to make model assessment easier.

# In[7]:


xgb_params = {'max_depth': 7, 
              'learning_rate': 0.002368706913117573, 
              'n_estimators': 3842, 
              'min_child_weight': 4, 
              'colsample_bytree': 0.6612496396706031, 
              'subsample': 0.6060764549240347, 
              'reg_alpha': 0.18899174723187226, 
              'reg_lambda': 30.33470416661318}


# In[8]:


def score_dataset(X, y, model=XGBRegressor(), cv_folds=2):
    # Label encoding for categoricals
    #
    # Label encoding is good for XGBoost and RandomForest, but one-hot
    # would be better for models like Lasso or Ridge. The `cat.codes`
    # attribute holds the category levels.
    for colname in X.select_dtypes(["category"]):
        X[colname] = X[colname].cat.codes
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    score = cross_val_score(
        model, X, y, cv=cv_folds, scoring="neg_mean_squared_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score


# In[9]:


def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=disc_features) 
    mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)    
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


# In[10]:


get_ipython().run_cell_magic('time', '', '\nmi_scores = make_mi_scores(X_train, y_train, discrete_features = disc_features)\n')


# In[11]:


mi_scores


# It seems the top 4 variables, i.e., cont8, cat1, cont0, and cat9, are more significant than others. Let us try creating products of these variables. 

# ### Create pairwise product features

# In[12]:


def create_pairwise_product(X):
    X2 = pd.DataFrame()
    for idx1, c1 in enumerate(X.columns):
        for idx2, c2 in enumerate(X.columns):
            if idx1 >= idx2: continue
                
            new_var = pd.Series(X[c1] * X[c2], name=f'{c1}-{c2}', index=X.index)
            X2 = pd.concat([X2, new_var], axis=1)
                
    return X2


# In[13]:


pw_cols = ['cont8', 'cat1', 'cont0', 'cat9']
X_train_pw = create_pairwise_product(X_train[pw_cols])
X_train2 = X_train.join(X_train_pw)


# In[14]:


score = score_dataset(X_train2, y_train, model=XGBRegressor(**xgb_params), cv_folds=2)
print(f'RMSE: = {score:.4f}')


# It does not seem to make a difference. Now, let's try adding a little more variables.

# In[15]:


pw_cols = [c for c in X_train.columns if mi_scores[c] > 0.005]
X_train_pw = create_pairwise_product(X_train[pw_cols])
X_train2 = X_train.join(X_train_pw)


# In[16]:


score = score_dataset(X_train2, y_train, model=XGBRegressor(**xgb_params), cv_folds=2)
print(f'RMSE: = {score:.4f}')


# Pairwise product variables do not seem to make much difference.

# ### PCA Transformation

# Next, we will try PCA transformation approach.

# In[17]:


# Create PCA

pca = PCA()
X_train_pca = pca.fit_transform(X_train)


# In[18]:


component_names = [f'PC{i+1}' for i in range(X_train_pca.shape[1])]
X_train_pca = pd.DataFrame(X_train_pca, columns=component_names)


# In[19]:


X_train_pca


# In[20]:


loadings = pd.DataFrame(pca.components_.T, columns=component_names, index=X_train.columns)


# In[21]:


loadings


# In[22]:


def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1,2)
    n = pca.n_components_
    grid = np.arange(1, n+1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(xlabel='Component', title='% Explained Variance', ylim=(0.0, 1.0))
    
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0,cv], 'o-')
    axs[1].set(xlabel='Component', title='% Cumulative Variance', ylim=(0.0, 1.0))
    
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs


# In[23]:


_ = plot_variance(pca)


# The first 2 PCA components explain 80% of total variances. Next, let's check their MI scores.

# In[24]:


X_train2 = X_train.join(X_train_pca)


# In[25]:


score = score_dataset(X_train2, y_train, model=XGBRegressor(**xgb_params), cv_folds=2)
print(f'RMSE: = {score:.4f}')


# We now confirm that adding new PCA variables do not help.

# ### Target Encoding

# In[26]:


X_train_cat_cnts = list(map(lambda c: X_train[c].value_counts(), cat_cols))


# In[27]:


X_train_cat_cnts = pd.DataFrame(X_train_cat_cnts).T


# In[28]:


X_train_cat_cnts


# It seems variables cat9 has many levels, followed by cat6, cat7, and cat8. Let's use them as candicates for target encoding. To avoid overfitting, we split the training data into encoding and rest.

# In[29]:


# Encoding split
X_train_enc = X_train.sample(frac=0.2, random_state=0)
y_train_enc = y_train.loc[X_train_enc.index]

# Training split
X_train_res = X_train.drop(index=X_train_enc.index)
y_train_res = y_train.drop(X_train_enc.index)


# In[30]:


print(f'X_train_enc.shape = {X_train_enc.shape}, y_train_enc = {y_train_enc.shape}')
print(f'X_train_res.shape = {X_train_res.shape}, y_train_res = {y_train_res.shape}')


# In[31]:


enc = MEstimateEncoder(cols=['cat6', 'cat7', 'cat8', 'cat9'], m=1.0)
enc.fit(X_train_enc, y_train_enc)
X_train2 = enc.transform(X_train_res, y_train_res)


# In[32]:


X_train2


# In[33]:


xgb_params = {'max_depth': 7, 
              'learning_rate': 0.002368706913117573, 
              'n_estimators': 3842, 
              'min_child_weight': 4, 
              'colsample_bytree': 0.6612496396706031, 
              'subsample': 0.6060764549240347, 
              'reg_alpha': 0.18899174723187226, 
              'reg_lambda': 30.33470416661318}


# In[34]:


score = score_dataset(X_train2, y_train_res, model=XGBRegressor(**xgb_params), cv_folds=2)
print(f'RMSE: = {score:.4f}')


# ## Conclusions

# We have tried a few common feature engineering approaches but none of them seem useful. In the next steps, we will survey the existing work to collect new ideas.
