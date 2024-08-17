#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install xgbfir')
get_ipython().system('pip install openpyxl')


# In[2]:


import pandas as pd
import numpy as np
import random
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import time
import seaborn as sns
import matplotlib.pyplot as plt
import gc
import datetime
import time
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import xgbfir
import xgboost as xgb

#importing plotly and cufflinks in offline mode
import cufflinks as cf
import plotly.offline
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

import plotly 
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import iplot
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
    
N_SPLITS = 5
N_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 200
VERBOSE = 1000
SEED = 42

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything(SEED)


# # **<span style="color:#e76f51;">Overview</span>**
# 
# The dataset is used for this competition is synthetic, but based on a real dataset and generated using a CTGAN. The original dataset deals with predicting identifying spam emails via various extracted features from the email. Features are anonymized.

# # **<span style="color:#e76f51;">Target</span>**
# 
# Our goal is to **predict** whether email is **spam or ham** based on a binary target feature called 'target'. This is a classification task.

# # **<span style="color:#e76f51;">Dataset</span>**
# 
# - In train dataset we have 600K rows and in test dataset 540K.
# - There are 100 features all continous. 
# - Target label is binary. We have a balanced dataset.
# 
# 

# In[3]:


train = pd.read_csv('../input/tabular-playground-series-nov-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-nov-2021/test.csv')


# In[4]:


train.shape, test.shape


# In[5]:


# no missing
train.isnull().any().sum(), test.isnull().any().sum()


# In[6]:


train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)


# In[7]:


features = [col for col in train.columns if 'f' in col]
org_features = features.copy()
TARGET='target'


# # **<span style="color:#e76f51;">Basic Statistics</span>**

# In[8]:


train[features].describe().style.background_gradient(cmap='Pastel1')


# # **<span style="color:#e76f51;">Target Feature</span>**
# 
# Our target is binary and dataset distribution is balanced. We have a classification task here.

# In[9]:


target_1 = train[train[TARGET]==0].shape[0]
target_2 = train[train[TARGET]==1].shape[0]
plt.figure(figsize=(15, 7))
plt.pie([target_1,target_2], labels = ["0" , "1"],autopct='%1.1f%%',colors = ["#17becf", "#1f77b4"])
plt.title('Target Value')


# # **<span style="color:#e76f51;">Target Feature Correlations</span>**
# 
# We see low correlation between target and features.

# In[10]:


cor_1 = train.corr()
cor_1.head()
cor_target = cor_1.loc['target':'target']
cor_2 = cor_target.drop(['target'],axis=1)
cor_3 = abs(cor_2)
cor_4 = cor_3.sort_values(by='target',axis=1, ascending=False)
pd.set_option('display.max_rows', 1)
pd.set_option('display.max_columns', 100)
cor_4.head()


# In[11]:


del cor_1
del cor_2
del cor_3
del cor_4
gc.collect()

pd.set_option('display.max_rows', 20)


# # **<span style="color:#e76f51;">Features Correlation</span>**
# 
# Our correlation plot shows that we have very low correlation between features in our dataset.

# In[12]:


# https://www.kaggle.com/legendsoul/tps-october-21-comprehensive-insight-of-eda

def correlation_matrix(data, features):
    
    fig, ax = plt.subplots(1, 1, figsize = (20, 20))
    plt.title('Pearson Correlation Matrix', fontweight='bold', fontsize=25)
    fig.set_facecolor('#d0d0d0') 
    corr = data[features].corr()

    # Mask to hide upper-right part of plot as it is a duplicate
    mask = np.triu(np.ones_like(corr, dtype = bool))
    sns.heatmap(corr, annot = False, center = 0, cmap = 'jet', mask = mask, linewidths = .5, square = True, cbar_kws = {"shrink": .70})
    ax.set_xticklabels(ax.get_xticklabels(), fontfamily = 'sans', rotation = 90, fontsize = 12)
    ax.set_yticklabels(ax.get_yticklabels(), fontfamily = 'sans', rotation = 0, fontsize = 12)
    plt.tight_layout()
    plt.show()
    
correlation_matrix(train, features)


# We see high variance in features 'f2' and 'f35'

# In[13]:


train.var().iplot(kind='bar')


# # **<span style="color:#e76f51;">Train and Test Distributions</span>**
# 

# In[14]:


sample_size=1000
train_sample = train.sample(n=sample_size, replace=True, random_state=SEED)
test_sample = test.sample(n=sample_size, replace=True, random_state=SEED)

print("Feature distribution of continous features: ")
ncols = 5
nrows = int(len(features) / ncols + (len(features) % ncols > 0))

fig, axes = plt.subplots(nrows, ncols, figsize=(18, 50), facecolor='#EAEAF2')

for r in range(nrows):
    for c in range(ncols):
        col = features[r*ncols+c]
        sns.kdeplot(x=train_sample[col], ax=axes[r, c], color='#58D68D', label='Train data')
        sns.kdeplot(x=test_sample[col], ax=axes[r, c], color='#DE3163', label='Test data')
        axes[r, c].set_ylabel('')
        axes[r, c].set_xlabel(col, fontsize=8, fontweight='bold')
        axes[r, c].tick_params(labelsize=5, width=0.5)
        axes[r, c].xaxis.offsetText.set_fontsize(4)
        axes[r, c].yaxis.offsetText.set_fontsize(4)
plt.show()



# # **<span style="color:#e76f51;">Feature Unique Values</span>**
# 
# Following is a plot showing unique values in features categorized by shapes (Gaussian and Pearson like). Considering the high number of unique values maybe it's worth trying to discretize the features. Also the two features in red (Gaussian) which have similar number of uniques like blue (Pearson) are f1 and f36.
# 

# In[15]:


h_skew = train[org_features].loc[:,train[org_features].skew() >= 2].columns
l_skew = train[org_features].loc[:,train[org_features].skew() < 2].columns

fig, ax = plt.subplots(figsize=(10, 5))
Pearson_val=train[h_skew].nunique().sort_values().values
Gaussian_val=train[l_skew].nunique().sort_values().values
ax.plot(Pearson_val, '.', color='blue',linewidth=1.0, label="Pearson")
ax.plot(Gaussian_val, '.', color='red',linewidth=1.0, label="Gaussian")
ax.set_title('Unique Values')
ax.grid(True)
ax.text(0.6, 0.5, "Features F1 & F36", ha="center", va="center", rotation=15, size=15,bbox=dict(boxstyle="rarrow,pad=0.3", fc="0.9", ec="b", lw=2),transform=ax.transAxes)
ax.legend()
plt.show()


# # **<span style="color:#e76f51;">Feature Engineering</span>**

# # **<span style="color:#e76f51;">Feature Interactions</span>**
# 
# Basic features created with division,multiplication,adding and subtraction operations. I will be using **[xgbfir](https://github.com/limexp/xgbfir)** package to find interacting features. Xgbfir is a XGBoost model dump parser, which ranks features as well as feature interactions by different metrics. Default interaction max value is 2 for xgbfir and can be changed with 'MaxInteractionDepth' parameter.
# 
# <div style="width:100%;text-align: center;"> <img align=middle src="https://i.imgur.com/DNdjqWO.jpg" alt="Heat beating" style="height:200px;margin-top:3rem;"> </div>
# 
# Some xgbfir output metrics
# 
# - Gain: Total gain of each feature or feature interaction
# - FScore: Amount of possible splits taken on a feature or feature interaction
# - wFScore: Amount of possible splits taken on a feature or feature interaction weighted by the probability of the splits to take place
# - Average wFScore: wFScore divided by FScore
# - Average Gain: Gain divided by FScore
# - Expected Gain: Total gain of each feature or feature interaction weighted by the probability to gather the gain
# 

# In[16]:


tmp_df = train.sample(n=100000)
xgb_X = tmp_df[org_features]
xgb_y = tmp_df[TARGET]

xgb_model = xgb.XGBClassifier(random_state=42,tree_method='gpu_hist',eval_metric='auc').fit(xgb_X,xgb_y)

xgbfir.saveXgbFI(xgb_model, feature_names=xgb_X.columns, OutputXlsxFile='xgb.xlsx')
joint_contrib = pd.read_excel('xgb.xlsx')

xls = pd.ExcelFile('xgb.xlsx')
df1 = pd.read_excel(xls, 'Interaction Depth 0')
df2 = pd.read_excel(xls, 'Interaction Depth 1')
df3 = pd.read_excel(xls, 'Interaction Depth 2')

frames = [df2] # I will be using depth1 interactions only for demonstration
joint_contrib = pd.concat(frames)

abs_imp_joint_contrib = (joint_contrib.groupby('Interaction')
                                          .Gain
                                          .apply(lambda x: x.abs().sum())
                                           .sort_values(ascending=False))
# then calculate the % of total joint contribution by dividing by the sum of all absolute vals
rel_imp_join_contrib = abs_imp_joint_contrib / abs_imp_joint_contrib.sum()
rel_imp_join_contrib.head(15)[::-1].iplot(kind='barh', color='#4358C0', title='Joint Feature Importances');


# In[17]:


joint_contrib.sort_values(by='Gain', ascending=False)


# In[18]:


train['f55_ratio_f34'] = train['f55']/train['f34']
test['f55_ratio_f34'] = test['f55']/test['f34']

train['f34_multiply_f8'] = train['f34']*train['f8']
test['f34_multiply_f8'] = test['f34']*test['f8']

train['f34_diff_f55'] = train['f34']-train['f55']
test['f34_diff_f55'] = test['f34']-test['f55']

train['f34_ratio_f80'] = train['f34']/train['f80']
test['f34_ratio_f80'] = test['f34']/test['f80']

train['f43_sum_f34'] = train['f43']+train['f34']
test['f43_sum_f34'] = test['f43']+test['f34']

train['f55_diff_f56'] = train['f55']-train['f56']
test['f55_diff_f56'] = test['f55']-test['f56']

train['f27_diff_f55'] = train['f27']-train['f55']
test['f27_diff_f55'] = test['f27']-test['f55']

train['f41_multiply_f34'] = train['f41']*train['f34']
test['f41_multiply_f34'] = test['f41']*test['f34']

train['f90_sum_f55'] = train['f90']+train['f55']
test['f90_sum_f55'] = test['f90']+test['f55']

train['f60_ratio_f55'] = train['f60']/train['f55']
test['f60_ratio_f55'] = test['f60']/test['f55']

feature_interactions = ['f55_ratio_f34', 'f34_multiply_f8','f34_diff_f55',
                   'f34_ratio_f80', 'f43_sum_f34', 'f55_diff_f56',
                   'f27_diff_f55','f41_multiply_f34','f90_sum_f55',
                   'f60_ratio_f55'
                  ]

features += feature_interactions


# In[19]:


train[feature_interactions].sample(n=2000).iplot(kind='histogram',subplots=True,bins=50)


# In[20]:


seed_everything(SEED)


# # **<span style="color:#e76f51;">Target Correlations (After New Features)</span>**
# We see some improvement with new features in terms of target correlation.

# In[21]:


cor_1 = train.corr()
cor_1.head()
cor_target = cor_1.loc['target':'target']
cor_2 = cor_target.drop(['target'],axis=1)
cor_3 = abs(cor_2)
cor_4 = cor_3.sort_values(by='target',axis=1, ascending=False)
pd.set_option('display.max_rows', 1)
pd.set_option('display.max_columns', 100)
cor_4.head()


# In[22]:


del cor_1
del cor_2
del cor_3
del cor_4
gc.collect()
pd.set_option('display.max_rows', 20)


# # **<span style="color:#e76f51;">K-Means Clustering</span>**
# 
# We create a clustering model based on K-Means. Data first scaled then fit in K-Means clustering.  Later we calculate each row's distance to each cluster and predict cluster number also as another feature. <br>
# 
# Code modified version of snippet reference: https://github.com/mljar/mljar-supervised <br>

# In[23]:


# MiniBatchKMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads
if (os.name=='nt'):
    os.environ['OMP_NUM_THREADS']='4' #Workaround for MKL Windows BUG with K-means++


class KMeansTransformer(object):
    def __init__(self, cluster_number=None):
        self._cluster_number = cluster_number
       
    def fit(self, X, y):
        if X.shape[1] == 0:
            raise Exception("input error")

        if self._cluster_number is None:
            n_clusters = int(np.log10(X.shape[0]) * 8)
            n_clusters = max(8, n_clusters)
            n_clusters = min(n_clusters, X.shape[1])
        else: 
            n_clusters = self._cluster_number
            

        self._input_columns = X.columns.tolist()
        # scale data
        self._scale = StandardScaler(copy=True, with_mean=True, with_std=True)
        X = self._scale.fit_transform(X)

        self._kmeans = kmeans = MiniBatchKMeans(n_clusters=n_clusters, init="k-means++")
        self._kmeans.fit(X)
        self._create_new_features_names()

    def _create_new_features_names(self):
        n_clusters = self._kmeans.cluster_centers_.shape[0]
        self._new_features = [f"Dist_Cluster_{i}" for i in range(n_clusters)]
        self._new_features += ["Cluster"]

    def transform(self, X):
        if self._kmeans is None:
            raise Exception("KMeans not fitted")

        # scale
        X_scaled = self._scale.transform(X[self._input_columns])

        # kmeans
        distances = self._kmeans.transform(X_scaled)
        clusters = self._kmeans.predict(X_scaled)

        X[self._new_features[:-1]] = distances
        X[self._new_features[-1]] = clusters

        return X


# # **<span style="color:#e76f51;">Scaling</span>**

# In[24]:


from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

scaler = StandardScaler()
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])
train.shape, test.shape


# # **<span style="color:#e76f51;">Keras Model with Built-in Discretization Layer</span>**

# In[25]:


import tensorflow as tf
tf.random.set_seed(SEED)
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import layers


es = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=20, verbose=0,
    mode='min',restore_best_weights=True)

plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, verbose=0,
    mode='min')
    
# with Discrete layers
def base_model_disc(activator='relu',summary=False,bin_data=None,shape_size=None):
    
    # define Keras Discretization layer with number of bins parameter 
    disct_layer = tf.keras.layers.Discretization(num_bins=20)
    disct_layer.adapt(bin_data)

    inputs = tf.keras.Input(shape=(shape_size,), name='input_data')
    x = tf.keras.layers.Dense(32, activation=activator)(inputs)
      
    y = disct_layer(inputs)
    y = tf.keras.layers.Dense(32, activation=activator)(y)
  
    x_cnn = tf.keras.layers.concatenate([x, y])

    
    x1 = tf.keras.layers.Dense(32, activation=activator)(x_cnn)
    x1 = tf.keras.layers.Dense(32, activation=activator)(x1)
    x1 = tf.keras.layers.Dropout(0.2)(x1)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x1)
    model = tf.keras.Model(inputs, outputs)
    if (summary):
        model.summary()
    return model


# # **<span style="color:#e76f51;">Training</span>**

# In[26]:


nn_oof = np.zeros(train.shape[0])
nn_pred = np.zeros(test.shape[0])
f_scores = []

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


for fold, (trn_idx, val_idx) in enumerate(skf.split(X=train, y=train[TARGET])):
    print(f"===== fold {fold} =====")
    X_train, y_train = train[features].iloc[trn_idx], train[TARGET].iloc[trn_idx]
    X_valid, y_valid = train[features].iloc[val_idx], train[TARGET].iloc[val_idx]
    X_test = test[features]
       
    new_features = features.copy()
    
    ### kmeans
        
    kmeans = KMeansTransformer()
    kmeans.fit(X_train, y_train)
    X_train = kmeans.transform(X_train)
    X_valid = kmeans.transform(X_valid)
    X_test = kmeans.transform(X_test)
    kmeans_columns = kmeans._new_features
    
    new_features +=kmeans_columns
        
    

    start = time.time()
    
        
    nn_model = base_model_disc(activator='relu',summary=False,bin_data=X_train,shape_size=X_train.shape[1])
    nn_model.compile(
        keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics = ['AUC'])

    history = nn_model.fit(X_train, y_train,      
              batch_size=2048,
              epochs=700,
              validation_data=(X_valid, y_valid),
              callbacks=[es, plateau],
              validation_batch_size=len(y_valid),
              shuffle=True,
             verbose = 0)
    
    scores = pd.DataFrame(history.history)
    scores['folds'] = fold
    
    if fold == 0:
        f_scores = scores 
    else: 
        f_scores = pd.concat([f_scores, scores], axis  = 0)

    nn_oof[val_idx] = nn_model.predict(X_valid).reshape(1,-1)[0]
    nn_pred += nn_model.predict(X_test).reshape(1,-1)[0] / N_SPLITS

    
    elapsed = time.time() - start
    nn_auc = roc_auc_score(y_valid, nn_oof[val_idx])
    print(f"fold {fold} - nn auc: {nn_auc:.6f}, elapsed time: {elapsed:.2f}sec")
    gc.collect()

print(f"oof nn roc = {roc_auc_score(train[TARGET], nn_oof)}")


# # **<span style="color:#e76f51;">History</span>**

# In[27]:


for fold in range(f_scores['folds'].nunique()):
    history_f = f_scores[f_scores['folds'] == fold]

    fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(14,4))
    fig.suptitle('Fold : '+str(fold), fontsize=14)
        
    plt.subplot(1,2,1)
    plt.plot(history_f.loc[:, ['loss', 'val_loss']], label= ['loss', 'val_loss'])
    plt.legend(fontsize=15)
    plt.grid()
    
    plt.subplot(1,2,2)
    plt.plot(history_f.loc[:, ['auc', 'val_auc']],label= ['auc', 'val_auc'])
    plt.legend(fontsize=15)
    plt.grid()
    
    print("Validation Loss: {:0.4f}".format(history_f['val_loss'].min()));


# # **<span style="color:#e76f51;">Submission</span>**

# In[28]:


sample = pd.read_csv("../input/tabular-playground-series-nov-2021/sample_submission.csv")
sample['target'] = nn_pred
sample.to_csv("submission.csv",index=None)


# # **<span style="color:#e76f51;">Work in progress</span>**
# 
# - Keras model needs to be optimized
# - feature selection and importance
# - ...

# In[ ]:




