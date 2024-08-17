#!/usr/bin/env python
# coding: utf-8

# # Current best version: 12

# In[1]:


import sys
sys.path.append('../input/iterative-stratification/iterative-stratification-master')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook


# - train_features.csv - Features for the training set. Features `g-` signify gene expression data, and `c-` signify cell viability data. `cp_type` indicates samples treated with a compound (`cp_vehicle`) or with a control perturbation (`ctrl_vehicle`); control perturbations have no MoAs; `cp_time` and `cp_dose` indicate treatment duration (24, 48, 72 hours) and `dose` (high or low).
# - train_targets_scored.csv - The binary MoA targets that are scored.
# - train_targets_nonscored.csv - Additional (optional) binary MoA responses for the training data. These are not predicted nor scored.
# - test_features.csv - Features for the test data. You must predict the probability of each scored MoA for each row in the test data.
# - sample_submission.csv - A submission file in the correct format.

# In[3]:


train=pd.read_csv("/kaggle/input/lish-moa/train_features.csv")
train


# In[4]:


# train.isnull().sum(axis=0).sum(), train.isnull().sum(axis=1).sum()


# In[5]:


# train.nunique(dropna=False).sort_values()


# In[6]:


# # This code is used to check duplicate columns (if any). It runs for a long time: the result is None, so avoid running this cell

# train_factorized = pd.DataFrame(index=train.index)
# for col in tqdm.notebook.tqdm(train.columns):
#     train_factorized[col] = train[col].map(train[col].value_counts())


# dup_cols = {}

# for i, c1 in enumerate(tqdm_notebook(train_factorized.columns)):
#     for c2 in train_factorized.columns[i + 1:]:
#         if c2 not in dup_cols and np.all(train_factorized[c1] == train_factorized[c2]):
#             dup_cols[c2] = c1
            
# dup_cols


# In[7]:


# # Check for classes distribution
train_target = pd.read_csv("../input/lish-moa/train_targets_scored.csv")
# limit = 0
# for col in tqdm_notebook(train_target.columns):
#     if col != "sig_id":
#         print(train_target[col].value_counts())
#     limit+=1
#     if limit >= 15:
#         break


# In[8]:


ctlVehicle_idx = train["cp_type"] != "ctl_vehicle"
train = train.loc[ctlVehicle_idx].reset_index(drop=True)
train = train.drop("cp_type", axis=1)
train_target = train_target.loc[ctlVehicle_idx].reset_index(drop=True)


# In[9]:


gcols = [g for g in train.columns if "g-" in g]
ccols = [c for c in train.columns if "c-" in c]
cpcols = [cp for cp in train.columns if "cp_" in cp]


# In[10]:


train


# In[11]:


train_target


# In[12]:


from sklearn.preprocessing import LabelEncoder

test = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")
ctlVehicle_test = test["cp_type"] == "ctl_vehicle"
test = test.drop("cp_type", axis=1)

# enc = LabelEncoder()
# for col in train[cpcols]:
#     train[col] = enc.fit_transform(train[col])
    
# enc = LabelEncoder()
# for col in test[cpcols]:
#     test[col] = enc.fit_transform(test[col])
train["cp_time"]=train["cp_time"].map({24: 1, 48: 2, 72: 3})
train["cp_dose"] = train["cp_dose"].map({"D1": 1, "D2": 2})
test["cp_time"]= test["cp_time"].map({24: 1, 48: 2, 72: 3})
test["cp_dose"]=test["cp_dose"].map({"D1": 1, "D2": 2})
train


# In[13]:


test


# In[14]:


for col in train.iloc[:, 3:].columns:
    percent = train[col].quantile([0.01, 0.99]).values
    train[col] = np.clip(train[col], percent[0], percent[1])

for col in test.iloc[:, 3:].columns:
    percent = test[col].quantile([0.01, 0.99]).values
    test[col] = np.clip(test[col], percent[0], percent[1])


# In[15]:


from sklearn.decomposition import PCA

g_pca = PCA(n_components=0.99)
c_pca = PCA(n_components=0.99)

train_test_g_concat = pd.concat([train[gcols], test[gcols]], axis=0)
train_test_c_concat = pd.concat([train[ccols], test[ccols]], axis=0)
g_pca.fit(train_test_g_concat)
c_pca.fit(train_test_c_concat)

train_gtrans = pd.DataFrame(g_pca.transform(train[gcols]), columns=["g_PCA" + str(i) for i in range(g_pca.n_components_)], index=train.index)
test_gtrans = pd.DataFrame(g_pca.transform(test[gcols]), columns=["g_PCA" + str(i) for i in range(g_pca.n_components_)], index=test.index)

train_ctrans = pd.DataFrame(c_pca.transform(train[ccols]), columns=["c_PCA" + str(i) for i in range(c_pca.n_components_)], index=train.index)
test_ctrans = pd.DataFrame(c_pca.transform(test[ccols]), columns=["c_PCA" + str(i) for i in range(c_pca.n_components_)], index=test.index)

g_pca.n_components_, c_pca.n_components_


# In[16]:


train = pd.concat([train_gtrans, train_ctrans, train[cpcols]], axis=1)
test = pd.concat([test_gtrans, test_ctrans, test[cpcols]], axis=1)
train


# In[17]:


test


# In[18]:


train_target = train_target.drop("sig_id", axis=1)
train_target


# In[19]:


import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_addons as tfa

def create_shallow_model():
    model = tf.keras.Sequential([
        tfa.layers.WeightNormalization(L.Dense(train.shape[1], input_shape=(train.shape[1],))),
        L.BatchNormalization(),
        L.Dropout(0.2),
        tfa.layers.WeightNormalization(L.Dense(128, activation="relu")),
        L.BatchNormalization(),
        L.Dropout(0.2),
        tfa.layers.WeightNormalization(L.Dense(train_target.shape[1], activation="sigmoid"))
    ])
    
    sgd = tf.keras.optimizers.SGD()
    adamw = tfa.optimizers.AdamW(weight_decay = 0.0001)
    adam = tf.keras.optimizers.Adam()
    radam = tfa.optimizers.RectifiedAdam()
    lookahead_radam = tfa.optimizers.Lookahead(radam)
    lookahead_adamw = tfa.optimizers.Lookahead(adamw)
    
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=1e-6), optimizer=adam)
    return model

def create_mid_model():
    model = tf.keras.Sequential([
        tfa.layers.WeightNormalization(L.Dense(train.shape[1], input_shape=(train.shape[1],))),
        L.BatchNormalization(),
        tfa.layers.WeightNormalization(L.Dense(512, activation="relu")),
        L.BatchNormalization(),
        L.Dropout(0.3),
        tfa.layers.WeightNormalization(L.Dense(512, activation="relu")),
        L.BatchNormalization(),
        L.Dropout(0.3),
        tfa.layers.WeightNormalization(L.Dense(train_target.shape[1], activation="sigmoid"))
    ])
    
    sgd = tf.keras.optimizers.SGD()
    adamw = tfa.optimizers.AdamW(weight_decay = 0.0001)
    adam = tf.keras.optimizers.Adam()
    radam = tfa.optimizers.RectifiedAdam()
    lookahead_radam = tfa.optimizers.Lookahead(radam)
    lookahead_adamw = tfa.optimizers.Lookahead(adamw)
    
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=1e-6), optimizer=lookahead_adamw)
    return model

def create_deep_model():
    model = tf.keras.Sequential([
        tfa.layers.WeightNormalization(L.Dense(train.shape[1], input_shape=(train.shape[1],))),
        L.BatchNormalization(),
        tfa.layers.WeightNormalization(L.Dense(512, activation="selu")),
        L.BatchNormalization(),
        L.Dropout(0.42),
        tfa.layers.WeightNormalization(L.Dense(512, activation="relu")),
        L.BatchNormalization(),
        L.Dropout(0.42),
        tfa.layers.WeightNormalization(L.Dense(256, activation="relu")),
        L.BatchNormalization(),
        L.Dropout(0.42),
        tfa.layers.WeightNormalization(L.Dense(256, activation="elu")),
        L.BatchNormalization(),
        L.Dropout(0.1),
        tfa.layers.WeightNormalization(L.Dense(train_target.shape[1], activation="sigmoid"))
    ])
    
    sgd = tf.keras.optimizers.SGD()
    adamw = tfa.optimizers.AdamW(weight_decay = 0.0001)
    adam = tf.keras.optimizers.Adam()
    radam = tfa.optimizers.RectifiedAdam()
    lookahead_radam = tfa.optimizers.Lookahead(radam)
    lookahead_adamw = tfa.optimizers.Lookahead(adamw)
    
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=1e-6), optimizer=lookahead_radam)
    return model


# In[20]:


# from sklearn.model_selection import KFold

predictions = []
kf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tf.random.set_seed(42)
for fold_id, (train_idx, valid_idx) in enumerate(kf.split(train, train_target)):
    model1 = create_shallow_model()
    model2 = create_mid_model()
    model3 = create_deep_model()
    
    history1 = model1.fit(train.iloc[train_idx], train_target.iloc[train_idx], batch_size=32,
              validation_data=(train.iloc[valid_idx], train_target.iloc[valid_idx]),
             epochs=100,
             verbose=2,
             callbacks=[
    tf.keras.callbacks.ReduceLROnPlateau(),
    tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("model1_fold" + str(fold_id) + ".h5", save_best_only=True, save_weights_only=True)
])
    print("Model 1, Fold ID: {}, train loss: {}, valid loss: {}".format(fold_id, min(history1.history["loss"]), min(history1.history["val_loss"])))
    history2 = model2.fit(train.iloc[train_idx], train_target.iloc[train_idx], batch_size=32,
              validation_data=(train.iloc[valid_idx], train_target.iloc[valid_idx]),
             epochs=100,
             verbose=2,
             callbacks=[
    tf.keras.callbacks.ReduceLROnPlateau(),
    tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("model2_fold" + str(fold_id) + ".h5", save_best_only=True, save_weights_only=True)
])
    print("Model 2, Fold ID: {}, train loss: {}, valid loss: {}".format(fold_id, min(history2.history["loss"]), min(history2.history["val_loss"])))
    history3 = model3.fit(train.iloc[train_idx], train_target.iloc[train_idx], batch_size=32,
              validation_data=(train.iloc[valid_idx], train_target.iloc[valid_idx]),
             epochs=100,
             verbose=2,
             callbacks=[
    tf.keras.callbacks.ReduceLROnPlateau(),
    tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("model3_fold" + str(fold_id) + ".h5", save_best_only=True, save_weights_only=True)
])
    
    print("Model 3, Fold ID: {}, train loss: {}, valid loss: {}".format(fold_id, min(history3.history["loss"]), min(history3.history["val_loss"])))
    
    model1.load_weights("model1_fold" + str(fold_id) + ".h5")
    model2.load_weights("model2_fold" + str(fold_id) + ".h5")
    model3.load_weights("model3_fold" + str(fold_id) + ".h5")
    pred1 = model1.predict(test)
    pred2 = model2.predict(test)
    pred3 = model3.predict(test)
    predictions.append(np.average([pred1, pred2, pred3], weights=[0.15, 0.7, 0.15], axis=0))


# In[21]:


pred = np.mean(predictions, axis=0)
pred = np.clip(pred, 0.001, 0.999)
pred.shape


# In[22]:


sub = pd.read_csv("../input/lish-moa/sample_submission.csv")
sub.loc[:, 1:] = pred
sub.loc[ctlVehicle_test, sub.columns != "sig_id"] = 0

# sub.loc[:, 1:] = tf.keras.utils.normalize(pred)
sub


# In[23]:


sub.to_csv("submission.csv", index=False)


# In[ ]:




