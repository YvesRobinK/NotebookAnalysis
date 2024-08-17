#!/usr/bin/env python
# coding: utf-8

# # Deep Learning with Feature Engineering
# 
# ## Credits
# This notebook is copied from https://www.kaggle.com/adityasharma01/simple-nn-tps-nov-21  
# The original is https://www.kaggle.com/javiervallejos/simple-nn-with-good-results-tps-nov-21  
# 
# ## What I am doing in this notebook?
# First of all, I am currently learning neural networks, so don't expect too much from me.  
# 
# I was actually following [the deep learning courses](https://www.kaggle.com/learn/intro-to-deep-learning), so I created a model after I have learned couple things. You can find that notebook [here.](https://www.kaggle.com/sfktrkl/tps-nov-2021-nn)  
# I wanted to create a model which produces good results but appearently it was not a perfect model :).
# 
# Then, I have also started to [feature engineering courses](https://www.kaggle.com/learn/feature-engineering).  
# I tried couple things in [this notebook](https://www.kaggle.com/sfktrkl/tps-nov-2021-nn-with-feature-engineering) but it didn't also produce very good results because my model wasn't a good one. Still, I see some improvements. So, I wanted to try my changes in a better model. This is the reason why I have copied this notebook.
# 
# Since my aim is to apply some changes to the feature engineering part, I am not touching the model at all.  
# Originally in this notebook, dataset was splitted according to the distribution of each column and some columns were added (mea, std, var, mean, etc).  
# What I have added is the application of the [mutual information](https://www.kaggle.com/ryanholbrook/mutual-information) before creating those additional columns.

# # Importing Libraries and Loading datasets

# In[1]:


import os
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks


# In[2]:


# Reading the dataset
raw_train = pd.read_csv("../input/tabular-playground-series-nov-2021/train.csv")
raw_test = pd.read_csv("../input/tabular-playground-series-nov-2021/test.csv")

train = raw_train.drop(['id','target'], axis = 1)
test = raw_test.drop('id', axis = 1)

target = raw_train.target
id_train = raw_train.id
id_test = raw_test.id


# ## Feature Engineering
# 
# Split the dataset by distribution of each column and add some basic columns (mea, std, var, mean, etc).

# ## Mutual information
# 
# Before creating those additional columns, select some features so that instead of having new columns which uses all of the data, only use selected features' columns. 

# In[3]:


def make_mi_scores(mi_scores, X, y):
    mi_scores = pd.Series(mi_scores, name="MI Scores")
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


# In[4]:


mi_scores = mutual_info_classif(train, target, random_state=1)
mi_scores_classif = make_mi_scores(mi_scores, train, target)


# In[5]:


plt.figure(dpi=100, figsize=(20, 16))
plot_mi_scores(mi_scores_classif[mi_scores_classif > 1e-4])


# In[6]:


selected_features = mi_scores_classif[mi_scores_classif > 1e-4].index.tolist()
selected_features = [f'f{feature}' for feature in selected_features]
print(f"Selected Features: {selected_features}")


# ## Split the dataset and add new columns

# In[7]:


# The number 2 is just a threshold to split
data = train[selected_features].copy()
h_skew = data.loc[:,data.skew() >= 2].columns  # with Skewed 
l_skew = data.loc[:,data.skew() < 2].columns   # Bimodal

# Skewed distrubutions
train['median_h'] = train[h_skew].median(axis=1)
test['median_h'] = test[h_skew].median(axis=1)

train['var_h'] = train[h_skew].var(axis=1)
test['var_h'] = test[h_skew].var(axis=1)

# Bimodal distributions
train['mean_l'] = train[l_skew].mean(axis=1)
test['mean_l'] = test[l_skew].mean(axis=1)

train['std_l'] = train[l_skew].std(axis=1)
test['std_l'] = test[l_skew].std(axis=1)

train['median_l'] = train[l_skew].median(axis=1)
test['median_l'] = test[l_skew].median(axis=1)

train['skew_l'] = train[l_skew].skew(axis=1)
test['skew_l'] = test[l_skew].skew(axis=1)

train['max_l'] = train[l_skew].max(axis=1)
test['max_l'] = test[l_skew].max(axis=1)

train['var_l'] = train[l_skew].var(axis=1)
test['var_l'] = test[l_skew].var(axis=1)

raw_train = train.copy()
raw_test = test.copy()


# In[8]:


# Scaling and Nomalization
transformer_high_skew = make_pipeline(
    StandardScaler(), 
    MinMaxScaler(feature_range=(0, 1))
)

transformer_low_skew = make_pipeline(
    StandardScaler(),
    MinMaxScaler(feature_range=(0, 1))
)

new_cols = train.columns[-8:]
h_skew = train.iloc[:,:100].loc[:, train.skew() >= 2].columns
l_skew = train.iloc[:,:100].loc[:, train.skew() < 2].columns

transformer_new_cols = make_pipeline(
    StandardScaler(),
    MinMaxScaler(feature_range=(0, 1))
)

preprocessor = make_column_transformer(
    (transformer_high_skew, l_skew),
    (transformer_low_skew, h_skew),
    (transformer_new_cols, new_cols),
)


# # Neural Network

# In[9]:


# Some parameters to config 
EPOCHS = 840
BATCH_SIZE = 2048 
ACTIVATION = 'swish'
LEARNING_RATE = 0.000265713
FOLDS = 5


# In[10]:


# Seed 
my_seed = 42
def seedAll(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
seedAll(my_seed)

# -----------------------------------------------------------------
def load_model(name:str):
    early_stopping = callbacks.EarlyStopping(
        patience=20,
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
            patience=7, 
            verbose=0,
            mode='min')

    model = keras.Sequential([
        layers.Dense(108, activation = ACTIVATION, input_shape = [train.shape[1]]),      
        layers.Dense(64, activation =ACTIVATION), 
        layers.Dense(32, activation =ACTIVATION),
        layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer= keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['AUC'],
    )
    
    return model, early_stopping, plateau


# # Model

# In[11]:


preds_valid_f = {}
preds_test = []
total_auc = []
f_scores = []

kf = StratifiedKFold(n_splits=FOLDS,random_state=0,shuffle=True)
for fold,(train_index, valid_index) in enumerate(kf.split(train,target)):
    X_train,X_valid = train.loc[train_index], train.loc[valid_index]
    y_train,y_valid = target.loc[train_index], target.loc[valid_index]

    # Preprocessing
    index_valid  = X_valid.index.tolist()
    test  = raw_test.copy()
    
    X_train = preprocessor.fit_transform(X_train)
    X_valid = preprocessor.transform(X_valid)
    test = preprocessor.transform(test)
      
    # Model
    model, early_stopping, plateau  = load_model('version1')
    history = model.fit(  X_train, y_train,
                validation_data = (X_valid, y_valid),
                batch_size = BATCH_SIZE, 
                epochs = EPOCHS,
                callbacks = [early_stopping, plateau],
                shuffle = True,
                verbose = 0
              )
    preds_valid = model.predict(X_valid).reshape(1,-1)[0] 
    preds_test.append(model.predict(test).reshape(1,-1)[0])
    
    #  Saving  scores to plot the end  
    scores = pd.DataFrame(history.history)
    scores['folds'] = fold
    if fold == 0:
        f_scores = scores 
    else: 
        f_scores = pd.concat([f_scores, scores], axis  = 0)
        
    # Concatenating valid preds
    preds_valid_f.update(dict(zip(index_valid, preds_valid)))

    # Getting score for a fold model
    fold_auc = roc_auc_score(y_valid, preds_valid)
    print(f"Fold {fold} roc_auc_score: {fold_auc}")

    # Total auc
    total_auc.append(fold_auc)

print(f"mean roc_auc_score: {np.mean(total_auc)}, std: {np.std(total_auc)}")


# # Outcomes

# In[12]:


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


# In[13]:


sub = pd.read_csv("../input/tabular-playground-series-nov-2021/sample_submission.csv")
sub['target'] = np.mean(preds_test, axis = 0)
sub.to_csv('submission.csv', index=False)
sub.head()

