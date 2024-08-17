#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import tensorflow as tf
import time, logging, gc
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import roc_auc_score

from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.callbacks import *
from sklearn.model_selection import KFold, GroupKFold
from tensorflow.keras.metrics import AUC
import matplotlib.pyplot as plt   


# In[2]:


train = pd.read_csv('../input/tabular-playground-series-apr-2022/train.csv')
test = pd.read_csv('../input/tabular-playground-series-apr-2022/test.csv')
submission = pd.read_csv("../input/tabular-playground-series-apr-2022/sample_submission.csv")
labels = pd.read_csv("../input/tabular-playground-series-apr-2022/train_labels.csv")
groups = train["sequence"]


# # EDA

# In[3]:


train.head()


# In[4]:


test.head()


# In[5]:


labels.head()


# ## <h4> Data Description </h4>
# * <b>sequence</b> - a unique id for each sequence 
# * <b>subject</b> - a unique id for the subject in the experiment
# * <b>step</b> - time step of the recording, in one second intervals
# * <b>sensor_00 - sensor_12</b> - the value for each of the thirteen sensors at that time step 
# * <b>state</b> - the value for each of the thirteen sensors at that time step
# ## <h4> Objective 🤾🏻‍♂️ </h4>
# * For each sequence in the test set, we will predict a probability for the state variable

# In[6]:


train.info()


# In[7]:


test.info()


# In[8]:


features  = [col for col in test.columns if col not in ("sequence","step","subject")]


# In[9]:


train[features].describe() 


# In[10]:


# adding labels to train data
train = pd.merge(train, labels,how='left', on="sequence")


# In[11]:


# data for the first 60 seconds
train[train['sequence']==0]


# # Feature Engineering

# In[12]:


def addFeatures(df):  
    for feature in features:
        df[feature + '_lag1'] = df.groupby('sequence')[feature].shift(1)
        df.fillna(0, inplace=True)
        df[feature + '_diff1'] = df[feature] - df[feature + '_lag1']    
    return df

train = addFeatures(train)
test = addFeatures(test)


# In[13]:


Window = 60


# In[14]:


y = train['state'].to_numpy().reshape(-1, Window)
train.drop(["sequence","step","subject","state"], axis=1, inplace=True)
test.drop(["sequence","step","subject"], axis=1, inplace=True)


# In[15]:


sc = StandardScaler()

sc.fit(train)
train = sc.transform(train)
test = sc.transform(test)


# In[16]:


train = train.reshape(-1, Window, train.shape[-1])
test = test.reshape(-1, Window, train.shape[-1])


# In[17]:


train.shape


# In[18]:


# Detect hardware, return appropriate distribution strategy
print(tf.version.VERSION)
tf.get_logger().setLevel(logging.ERROR)
try: # detect TPU
    tpu = None
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError: # detect GPU(s) and enable mixed precision
    strategy = tf.distribute.MirroredStrategy() # works on GPU and multi-GPU
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.config.optimizer.set_jit(True) # XLA compilation
    tf.keras.mixed_precision.experimental.set_policy(policy)
    print('Mixed precision enabled')
print("REPLICAS: ", strategy.num_replicas_in_sync)


# # Modeling

# In[19]:


def plotHist(hist):
    plt.plot(hist.history["auc"])
    plt.plot(hist.history["val_auc"])
    plt.title("model performance")
    plt.ylabel("area_under_curve")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()
    return


# In[20]:


def createModel():   
    with strategy.scope():
    
        input_layer = Input(shape=(train.shape[-2:]))
        x1 = Bidirectional(LSTM(768, return_sequences=True))(input_layer)
        
        x21 = Bidirectional(LSTM(512, return_sequences=True))(x1)
        x22 = Bidirectional(LSTM(512, return_sequences=True))(input_layer)
        l2 = Concatenate(axis=2)([x21, x22])
        
        x31 = Bidirectional(LSTM(384, return_sequences=True))(l2)
        x32 = Bidirectional(LSTM(384, return_sequences=True))(x21)
        l3 = Concatenate(axis=2)([x31, x32])
        
        x41 = Bidirectional(LSTM(256, return_sequences=True))(l3)
        x42 = Bidirectional(LSTM(128, return_sequences=True))(x32)
        l4 = Concatenate(axis=2)([x41, x42])
        
        l5 = Concatenate(axis=2)([x1, l2, l3, l4])
        x7 = Dense(128, activation='selu')(l5)
        x8 = Dropout(0.1)(x7)
        output_layer = Dense(units=1, activation="sigmoid")(x8)
        model = Model(inputs=input_layer, outputs=output_layer, name='DNN_Model')
        
        model.compile(optimizer="adam",loss="binary_crossentropy", metrics=[AUC(name = 'auc')])
    return(model)


# In[21]:


model = createModel()
model.summary()


# In[22]:


utils.plot_model(createModel())


# In[23]:


kf = GroupKFold(n_splits=10)
auc = []
test_preds = []
for fold, (train_idx, test_idx) in enumerate(kf.split(train, y, groups.unique())):
    print(f"** fold: {fold+1} ** ........training ...... \n")
    X_train, X_valid = train[train_idx], train[test_idx]
    y_train, y_valid = y[train_idx], y[test_idx]
    lr = ReduceLROnPlateau(monitor="val_auc", mode='max', factor=0.7, patience=4, verbose=False)
    es = EarlyStopping(monitor='val_auc',mode='max', patience=10, verbose=False,restore_best_weights=True)
    
    model = createModel()
        
    history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=30, batch_size = 64, callbacks = [es,lr],verbose = False)
    
    y_pred = model.predict(X_valid).squeeze()
    auc.append(roc_auc_score(y_valid, y_pred))
    print(f"auc: {auc[fold]} \n")
    test_preds.append(model.predict(test).squeeze())
    plotHist(history)
    del X_train, X_valid, y_train, y_valid, model, history
    gc.collect()  


# In[24]:


print(f"the mean AUC for the {kf.n_splits} folds is : {round(np.mean(auc)*100,3)}")


# # Submission

# In[25]:


submission["state"] = sum(test_preds)/kf.n_splits 
submission.to_csv('submission.csv', index=False)
submission   


# # References:
# * 1.[https://www.kaggle.com/code/ryanbarretto/lstm-baseline](https://www.kaggle.com/code/ryanbarretto/lstm-baseline)
# * 2.[https://www.kaggle.com/code/hamzaghanmi/tensorflow-bi-lstm-with-tpu](https://www.kaggle.com/code/hamzaghanmi/tensorflow-bi-lstm-with-tpu)
