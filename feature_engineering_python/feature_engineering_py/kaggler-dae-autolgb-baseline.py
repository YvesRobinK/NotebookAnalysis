#!/usr/bin/env python
# coding: utf-8

# # Kaggler DAE + AutoLGB Baseline
# 
# ## **UPDATE on 5/2/2021**
# 
# * Feature engineering using target encoding and label encoding from `Kaggler`.
# * Treating all features categorical based on the findings from [Simple yet interesting things about features](https://www.kaggle.com/jeongyoonlee/simple-yet-interesting-things-about-features): i.e. instead of creating two kinds of DAE features (one for categorical, the other for numerical features), creating just DAE features for categorical features.
# 
# ## **UPDATE on 5/1/2021**
# 
# Today, [`Kaggler`](https://github.com/jeongyoonlee/Kaggler) v0.9.4 is released with additional features for DAE as follows:
# * In addition to the swap noise (`swap_prob`), the Gaussian noise (`noise_std`) and zero masking (`mask_prob`) have been added to DAE to overcome overfitting.
# * Stacked DAE is available through the `n_layer` input argument (see Figure 3. in [Vincent et al. (2010), "Stacked Denoising Autoencoders"](https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf) for reference).
# 
# For example, to build a stacking DAE with 3 pairs of encoder/decoder and all three types of noises, you can do:
# ```python
# from kaggler.preprocessing import DAE
# 
# dae = DAE(cat_cols=cat_cols, num_cols=num_cols, n_layer=3, noise_std=.05, swap_prob=.2, masking_prob=.1)
# X = dae.fit_transform(pd.concat([trn, tst], axis=0))
# ```
# 
# If you're using previous versions, please upgrade `Kaggler` using `pip install -U kaggler`.
# 
# ---
# 
# In this notebook, I will show how to create DAE features from both training and test data, then train a LightGBM model with feature selection and hyperparameter optimization using [Kaggler](https://github.com/jeongyoonlee/Kaggler), a Python package for Kaggle competition.
# 
# The contents of the notebook are as follows:
# 1. Simple EDA and Target Transformation
# 2. DAE Feature Generation
# 3. AutoLGB Model Training
# 4. Submission

# ## Part 1. Simple EDA and Target Transformation

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import numpy as np
import lightgbm as lgb
import os
import pandas as pd
from pathlib import Path
import seaborn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from warnings import simplefilter


# In[2]:


get_ipython().system('pip install -U kaggler')


# In[3]:


plt.style.use('fivethirtyeight')
pd.set_option('max_columns', 100)
simplefilter('ignore')


# In[4]:


import kaggler
from kaggler.model import AutoLGB
from kaggler.preprocessing import DAE, TargetEncoder, LabelEncoder
print(kaggler.__version__)


# In[5]:


feature_name = 'dae_te_le'
algo_name = 'lgb'
version = 3
model_name = f'{algo_name}_{feature_name}_v{version}'

data_dir = Path('../input/tabular-playground-series-may-2021')
train_file = data_dir / 'train.csv'
test_file = data_dir / 'test.csv'
sample_file = data_dir / 'sample_submission.csv'

feature_file = f'{feature_name}.h5'
predict_val_file = f'{model_name}.val.txt'
predict_tst_file = f'{model_name}.tst.txt'
submission_file = f'{model_name}.sub.csv'

id_col = 'id'
target_col = 'target'


# In[6]:


encoding_dim = 128
seed = 42
n_fold = 5
n_class = 4


# In[7]:


trn = pd.read_csv(train_file, index_col=id_col)
tst = pd.read_csv(test_file, index_col=id_col)
sub = pd.read_csv(sample_file, index_col=id_col)
print(trn.shape, tst.shape, sub.shape)


# In[8]:


y = trn[target_col].str.split('_').str[1].astype(int) - 1
n_trn = trn.shape[0]
df = pd.concat([trn.drop(target_col, axis=1), tst], axis=0)
feature_cols = df.columns.tolist()
print(y.shape, df.shape)


# In[9]:


y.value_counts()


# In[10]:


df.describe()


# In[11]:


df.nunique()


# ## Part 2. Feature Engineering

# ### DAE

# First, generating DAE features by treating all features as categorical features. Internally, an embedding layer will be added to each feature to convert the categories into an embedding vector.

# In[12]:


dae = DAE(cat_cols=df.columns.to_list(), num_cols=[], encoding_dim=encoding_dim, random_state=seed, 
          swap_prob=.3, n_layer=3)
X = dae.fit_transform(df)
df_dae = pd.DataFrame(X, columns=[f'dae1_{x}' for x in range(X.shape[1])])
print(df_dae.shape)
df_dae.head()


# ### Target Encoding

# Target encoding is a popular feature engineering method for categorical features. However, it is subject to overfitting. `Kaggler` uses cross-validation and smoothing to avoid it.

# In[13]:


cv = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
te = TargetEncoder(cv=cv)
te.fit(trn[feature_cols], y)
df_te = te.transform(df[feature_cols])
df_te.columns = [f'te_{x}' for x in df.columns]
df_te.head()


# ### Label Encoding with Grouping

# Although features are already label-encoded, let's group rare categories with `Kaggler`'s label encoder.

# In[14]:


le = LabelEncoder(min_obs=50)
df_le = le.fit_transform(df[feature_cols])
df_le.columns = [f'le_{x}' for x in df.columns]
df_le.head()


# ## Part 3. AutoLGB Model Training

# In[15]:


params = {'num_class': n_class}


# In[16]:


df_feature = pd.concat([df_le, df_te, df_dae], axis=1)
df_feature.to_hdf(feature_file, key='data')

X = df_feature.iloc[:n_trn]
X_tst = df_feature.iloc[n_trn:]

clf = AutoLGB(objective='multiclass', metric='multi_logloss', params=params, sample_size=X.shape[0], 
              feature_selection=False, random_state=seed)
clf.tune(X, y)

features = clf.features
params = clf.params
n_best = clf.n_best
print(f'{n_best}')
print(f'{params}')
print(f'{features}')

p = np.zeros((X.shape[0], n_class), dtype=float)
p_tst = np.zeros((X_tst.shape[0], n_class), dtype=float)
for i, (i_trn, i_val) in enumerate(cv.split(X, y)):
    trn_data = lgb.Dataset(X.iloc[i_trn], y[i_trn])
    val_data = lgb.Dataset(X.iloc[i_val], y[i_val])
    clf = lgb.train(params, trn_data, n_best, val_data, verbose_eval=100)
    p[i_val] = clf.predict(X.iloc[i_val])
    p_tst += clf.predict(X_tst) / n_fold
    print(f'CV #{i + 1} Loss: {log_loss(y[i_val], p[i_val]):.6f}')


# In[17]:


print(f'CV Log Loss: {log_loss(y, p):.6f}')
np.savetxt(predict_val_file, p, fmt='%.6f')
np.savetxt(predict_tst_file, p_tst, fmt='%.6f')


# ## Part 4. Submission

# In[18]:


sub[sub.columns] = p_tst
sub.to_csv(submission_file)
sub.head()


# If you find this notebook helpful, please upvote it and share your feedback in comments. I really appreciate it.
# 
# You can find my other notebooks in both the current and previous TPS competitions below:
# * [Adversarial Validation with LightGBM](https://www.kaggle.com/jeongyoonlee/adversarial-validation-with-lightgbm): shows how close/different the feature distributions between the training and test data. It's a good exercise to perform it at the begining of the competition to understand the risk of overfitting to the training data.
# * [DAE with 2 Lines of Code with Kaggler](https://www.kaggle.com/jeongyoonlee/dae-with-2-lines-of-code-with-kaggler): shows how to extract DAE features and train the AutoLGB model with TPS4 data.
# * [AutoEncoder + Pseudo Label + AutoLGB](https://www.kaggle.com/jeongyoonlee/autoencoder-pseudo-label-autolgb): shows how to build a basic AutoEncoder using Keras, and perform automated feature selection and hyperparameter optimization using Kaggler's AutoLGB.
# * [Supervised Emphasized Denoising AutoEncoder](https://www.kaggle.com/jeongyoonlee/supervised-emphasized-denoising-autoencoder): shows how to build a more sophiscated version of * AutoEncoder, called supervised emphasized Denoising AutoEncoder (DAE), which trains DAE and a classifier simultaneously.
# * [Stacking Ensemble](https://www.kaggle.com/jeongyoonlee/stacking-ensemble): shows how to perform stacking ensemble.

# In[ ]:




