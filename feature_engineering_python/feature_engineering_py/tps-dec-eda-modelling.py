#!/usr/bin/env python
# coding: utf-8

# **Created by Sanskar Hasija**
# 
# **[TPS-DEC] 📊EDA + Modeling🔥**
# 
# **1 DECEMBER 2021**
# 

# # <center> [TPS-DEC] 📊EDA + MODELING🔥</center>
# ## <center>If you find this notebook useful, support with an upvote👍</center>

# # Table of Contents
# <a id="toc"></a>
# - [1. Introduction](#1)
# - [2. Imports](#2)
# - [3. Data Loading and Preperation](#3)
#     - [3.1 Exploring Train Data](#3.1)
#     - [3.2 Exploring Test Data](#3.2)
#     - [3.3 Submission File](#3.3)
# - [4. EDA](#4)
#     - [4.1 Overview of Data](#4.1)
#     - [4.1 Continuos and Categorical Data Distribution](#4.1)
#     - [4.2 Feature Distribution of Continous Features](#4.2)
#     - [4.3 Feature Distribution of Categorical Features](#4.3)
#     - [4.4 Target Distribution ](#4.4)
# - [5. Feature Engineering](#5)
# - [6. Modeling](#6)
#     - [6.1 LGBM Classifier](#6.1)
#     - [6.2 Catboost Classifier](#6.2)
#     - [6.3 XGBoost Classifier](#6.3)
#     - [6.4 Neural Network](#6.4)
# - [7. Submission](#7)

# <a id="1"></a>
# # Introduction

# **The dataset is used for this competition is synthetic, but based on a real dataset and generated using a CTGAN. This dataset is based off of the original Forest Cover Type Prediction competition.**
# 
# **Submissions are evaluated on multi-class classification accuracy.**

# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>

# <a id="2"></a>
# # Imports

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.stats import mode


from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.utils import to_categorical

from matplotlib import ticker
import time
import warnings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('float_format', '{:f}'.format)
warnings.filterwarnings('ignore')


# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>

# <a id="3"></a>
# # Data Loading and Preperation

# In[2]:


train = pd.read_csv("../input/tabular-playground-series-dec-2021/train.csv")
test = pd.read_csv("../input/tabular-playground-series-dec-2021/test.csv")
submission = pd.read_csv("../input/tabular-playground-series-dec-2021/sample_submission.csv")


train.drop(["Id"] , axis = 1 , inplace = True)
test.drop(["Id"] , axis = 1 , inplace = True)
TARGET = 'Cover_Type'
FEATURES = [col for col in train.columns if col not in ['id', TARGET]]
RANDOM_STATE = 12 


# <a id="3.1"></a>
# ## Exploring Train Data

# ### Quick view of Train Data

# In[3]:


train.head()


# In[4]:


print(f'Number of rows in train data: {train.shape[0]}')
print(f'Number of columns in train data: {train.shape[1]}')
print(f'No of missing values in train data: {sum(train.isna().sum())}')


# ### Basic statistics of training data

# In[5]:


train.describe()


# <a id="3.2"></a>
# ## Exploring Test Data

# ### Quick view of Test Data

# In[6]:


test.head()


# In[7]:


print(f'Number of rows in test data: {test.shape[0]}')
print(f'Number of columns in test data: {test.shape[1]}')
print(f'No of missing values in test data: {sum(test.isna().sum())}')


# ### Basic statistics of test data

# In[8]:


test.describe()


# <a id="3.3"></a>
# ## Submission File

# In[9]:


submission.head()


# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>

# <a id="4"></a>
# # EDA

# <a id="4.1"></a>
# ## Overview of Data

# In[10]:


train.iloc[:, :-1].describe().T.sort_values(by='std' , ascending = False)\
                     .style.background_gradient(cmap='GnBu')\
                     .bar(subset=["max"], color='#BB0000')\
                     .bar(subset=["mean",], color='green')


# <a id="4.2"></a>
# ## Continuos and Categorical Data Distribution

# In[11]:


df = pd.concat([train[FEATURES], test[FEATURES]], axis=0)

cat_features = [col for col in FEATURES if df[col].nunique() < 25]
cont_features = [col for col in FEATURES if df[col].nunique() >= 25]

del df
print(f'Total number of features: {len(FEATURES)}')
print(f'Number of categorical features: {len(cat_features)}')
print(f'Number of continuos features: {len(cont_features)}')

plt.pie([len(cat_features), len(cont_features)], 
        labels=['Categorical', 'Continuos'],
        colors=['#76D7C4', '#F5B7B1'],
        textprops={'fontsize': 13},
        autopct='%1.1f%%')
plt.show()


# <a id="4.3"></a>
# ## Feature Distribution of Continous Features

# In[12]:


ncols = 5
nrows = int(len(cont_features) / ncols + (len(FEATURES) % ncols > 0))-1

fig, axes = plt.subplots(nrows, ncols, figsize=(18, 8), facecolor='#EAEAF2')

for r in range(nrows):
    for c in range(ncols):
        col = cont_features[r*ncols+c]
        sns.kdeplot(x=train[col], ax=axes[r, c], color='#58D68D', label='Train data')
        sns.kdeplot(x=test[col], ax=axes[r, c], color='#DE3163', label='Test data')
        axes[r, c].set_ylabel('')
        axes[r, c].set_xlabel(col, fontsize=8, fontweight='bold')
        axes[r, c].tick_params(labelsize=5, width=0.5)
        axes[r, c].xaxis.offsetText.set_fontsize(4)
        axes[r, c].yaxis.offsetText.set_fontsize(4)
plt.show()


# <a id="4.4"></a>
# ## Feature Distribution of Categorical Features

# In[13]:


if len(cat_features) == 0 :
    print("No Categorical features")
else:
    ncols = 5
    nrows = int(len(cat_features) / ncols + (len(FEATURES) % ncols > 0)) 

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 45), facecolor='#EAEAF2')

    for r in range(nrows):
        for c in range(ncols):
            if r*ncols+c >= len(cat_features):
                break
            col = cat_features[r*ncols+c]
            sns.countplot(x=train[col], ax=axes[r, c], color='#58D68D', label='Train data')
            sns.countplot(x=test[col], ax=axes[r, c], color='#DE3163', label='Test data')
            axes[r, c].set_ylabel('')
            axes[r, c].set_xlabel(col, fontsize=8, fontweight='bold')
            axes[r, c].tick_params(labelsize=5, width=0.5)
            axes[r, c].xaxis.offsetText.set_fontsize(4)
            axes[r, c].yaxis.offsetText.set_fontsize(4)
    plt.show()


# **```Soil_type7``` and ```Soil_Type15``` are all zero values**

# <a id="4.5"></a>
# ## Target Distribution

# In[14]:


target_df = pd.DataFrame(train[TARGET].value_counts()).reset_index()
target_df.columns = [TARGET, 'count']
fig = px.bar(data_frame =target_df, 
             x = 'Cover_Type',
             y = 'count' , 
             color = "count",
             color_continuous_scale="Emrld") 
fig.show()
target_df.sort_values(by =TARGET , ignore_index = True)


# **There are total 7 different output classes**

# ### Removing Unwanted Rows and columns

# In[15]:


train = train.drop(index = int(np.where(train["Cover_Type"] == 5 )[0]))
train = train.drop(labels = ["Soil_Type7" , "Soil_Type15"] ,axis = 1)
FEATURES.remove('Soil_Type7')
FEATURES.remove('Soil_Type15')


# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>

# <a id="5"></a>
# #  Feature Engineering

# In[16]:


train["mean"] = train[FEATURES].mean(axis=1)
train["std"] = train[FEATURES].std(axis=1)
train["min"] = train[FEATURES].min(axis=1)
train["max"] = train[FEATURES].max(axis=1)

test["mean"] = test[FEATURES].mean(axis=1)
test["std"] = test[FEATURES].std(axis=1)
test["min"] = test[FEATURES].min(axis=1)
test["max"] = test[FEATURES].max(axis=1)

FEATURES.extend(['mean', 'std', 'min', 'max'])


# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>

# <a id="6"></a>
# #  Modeling

# In[17]:


scaler = StandardScaler()
for col in FEATURES:
    train[col] = scaler.fit_transform(train[col].to_numpy().reshape(-1,1))
    test[col] = scaler.transform(test[col].to_numpy().reshape(-1,1))
    
X = train[FEATURES].to_numpy().astype(np.float32)
y = train[TARGET].to_numpy().astype(np.float32)
X_test = test[FEATURES].to_numpy().astype(np.float32)

del train, test


# <a id="6.1"></a>
# ## LGBM Classifier

# In[18]:


lgb_params = {
    'objective' : 'multiclass',
    'metric' : 'multi_logloss',
    'device' : 'gpu',
}


lgb_predictions = []
lgb_scores = []

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for fold, (train_idx, valid_idx) in enumerate(kf.split(X = X, y = y)):

    print(10*"=", f"Fold={fold+1}", 10*"=")
    start_time = time.time()
    x_train = X[train_idx, :]
    x_valid = X[valid_idx, :]
    y_train = y[train_idx]
    y_valid = y[valid_idx]
    
    model = LGBMClassifier(**lgb_params)
    model.fit(x_train, y_train,
          early_stopping_rounds=200,
          eval_set=[(x_valid, y_valid)],
          verbose=0)
    
    preds_valid = model.predict(x_valid)
    acc = accuracy_score(y_valid,  preds_valid)
    lgb_scores.append(acc)
    run_time = time.time() - start_time
    print(f"Fold={fold+1}, acc: {acc:.8f}, Run Time: {run_time:.2f}")
    test_preds = model.predict(X_test)
    lgb_predictions.append(test_preds)
    
print("Mean Accuracy :", np.mean(lgb_scores))


# <a id="6.2"></a>
# ## Catboost Classifier

# In[19]:


catb_params = {
    "objective": "MultiClass",
    "task_type": "GPU",
}

catb_predictions = []
catb_scores = []

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for fold, (train_idx, valid_idx) in enumerate(kf.split(X = X, y = y)):

    print(10*"=", f"Fold={fold+1}", 10*"=")
    start_time = time.time()
    x_train = X[train_idx, :]
    x_valid = X[valid_idx, :]
    y_train = y[train_idx]
    y_valid = y[valid_idx]
    
    model = CatBoostClassifier(**catb_params)
    model.fit(x_train, y_train,
          early_stopping_rounds=200,
          eval_set=[(x_valid, y_valid)],
          verbose=0)
    
    preds_valid = model.predict(x_valid)
    acc = accuracy_score(y_valid,  preds_valid)
    catb_scores.append(acc)
    run_time = time.time() - start_time
    print(f"Fold={fold+1}, acc: {acc:.8f}, Run Time: {run_time:.2f}")
    test_preds = model.predict(X_test)
    catb_predictions.append(test_preds)
    
print("Mean Accuracy:", np.mean(catb_scores))


# <a id="6.3"></a>
# ## XGBoost Classifier

# In[20]:


xgb_params = {
    'objective': 'multi:softmax',
    'eval_metric': 'mlogloss',
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    }

xgb_predictions = []
xgb_scores = []

xgb_predictions = []
xgb_scores = []

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for fold, (train_idx, valid_idx) in enumerate(kf.split(X = X, y = y)):

    print(10*"=", f"Fold={fold+1}", 10*"=")
    start_time = time.time()
    x_train = X[train_idx, :]
    x_valid = X[valid_idx, :]
    y_train = y[train_idx]
    y_valid = y[valid_idx]
    
    model = XGBClassifier(**xgb_params)
    model.fit(x_train, y_train,
          early_stopping_rounds=200,
          eval_set=[(x_valid, y_valid)],
          verbose=0)
    preds_valid = model.predict(x_valid)
    acc = accuracy_score(y_valid,  preds_valid)
    xgb_scores.append(acc)
    run_time = time.time() - start_time
    print(f"Fold={fold+1}, acc: {acc:.8f}, Run Time: {run_time:.2f}")
    test_preds = model.predict(X_test)
    xgb_predictions.append(test_preds)
    
print("Mean Accuracy:", np.mean(xgb_scores))


# <a id="6.4"></a>
# ## Neural Network

# In[21]:


LEARNING_RATE = 0.0001
BATCH_SIZE = 2048
EPOCHS = 100
VALIDATION_RATIO = 0.05

LE = LabelEncoder()
y = to_categorical(LE.fit_transform(y))
X_train , X_valid ,y_train ,y_valid  = train_test_split(X,y , test_size = VALIDATION_RATIO , random_state=RANDOM_STATE)


def load_model(): 
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2048, activation = 'swish', input_shape = [X.shape[1]]),
        tf.keras.layers.Dense(1024, activation ='swish'),
        tf.keras.layers.Dense(512, activation ='swish'),
        tf.keras.layers.Dense(6, activation='softmax'),
    ])
    model.compile(
        optimizer= tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['acc'],
    )
    return model
    
    
early_stopping = callbacks.EarlyStopping(
        patience=10,
        min_delta=0,
        monitor='val_loss',
        restore_best_weights=True,
        verbose=0,
        mode='min', 
        baseline=None,
    )
plateau = callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=4, 
            verbose=0,
            mode='min')

nn_model = load_model()
history = nn_model.fit(  X_train , y_train,
                validation_data = (X_valid , y_valid),
                batch_size = BATCH_SIZE, 
                epochs = EPOCHS,
                callbacks = [early_stopping , plateau],
              )
nn_preds = nn_model.predict(X_test , batch_size=BATCH_SIZE)


# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>

# <a id="7"></a>
# #  Submission

# ### LGBM Classifier Submission

# In[22]:


lgb_submission = submission.copy()
lgb_submission['Cover_Type'] = np.squeeze(mode(np.column_stack(lgb_predictions),axis = 1)[0]).astype('int')
lgb_submission.to_csv("lgb-subs.csv",index=None)
lgb_submission.head()


# ### Catboost Classifier Submission

# In[23]:


catb_submission = submission.copy()
catb_submission['Cover_Type'] = np.squeeze(mode(np.column_stack(catb_predictions),axis = 1)[0]).astype('int')
catb_submission.to_csv("catb-subs.csv",index=None)
catb_submission.head()


# ### XGBoost Classifier Submission

# In[24]:


xgb_submission = submission.copy()
xgb_submission['Cover_Type'] = np.squeeze(mode(np.column_stack(xgb_predictions),axis = 1)[0]).astype('int')
xgb_submission.to_csv("xgb-subs.csv",index=None)
xgb_submission.head()


# ### Neural Network Submission

# In[25]:


nn_submission = submission.copy()
nn_submission["Cover_Type"] = LE.inverse_transform(np.argmax((nn_preds), axis=1)).astype(int)
nn_submission.to_csv("nn-sub.csv" , index= False)
nn_submission.head()


# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>
