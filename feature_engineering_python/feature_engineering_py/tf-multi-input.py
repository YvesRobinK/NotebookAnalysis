#!/usr/bin/env python
# coding: utf-8

# ### Import libraries

# In[1]:


import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as tfl
from tensorflow.data import Dataset as ds

from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from sklearn.cluster import KMeans

from PIL import Image
import os
import warnings

warnings.filterwarnings('ignore')

np.random.seed(0)
tf.random.set_seed(0)


# In[2]:


strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines
print("Number of accelerators: ", strategy.num_replicas_in_sync)


# ### Read data

# In[3]:


train = pd.read_csv('/kaggle/input/petfinder-pawpularity-score/train.csv')
train['path'] = '/kaggle/input/petfinder-pawpularity-score/train/' + train['Id'] + '.jpg'
train.head(3)


# In[4]:


test = pd.read_csv('/kaggle/input/petfinder-pawpularity-score/test.csv')
test['path'] = '/kaggle/input/petfinder-pawpularity-score/test/' + test['Id'] + '.jpg'


# ### Feature Engineering

# In[5]:


def size_and_shape(row):
    img = Image.open(row['path'])
    return pd.Series([img.size[0], img.size[1], os.path.getsize(row['path'])])


# In[6]:


scale = MinMaxScaler()

train[['width', 'height', 'size']] = pd.DataFrame(scale.fit_transform(train.apply(size_and_shape, axis=1).values))
test[['width', 'height', 'size']] = pd.DataFrame(scale.transform(test.apply(size_and_shape, axis=1).values))


# In[7]:


k = KMeans(8, random_state=0)

train['cluster'] = k.fit_predict(train.drop(['Id', 'Pawpularity', 'path'], axis=1))
test['cluster'] = k.predict(test.drop(['Id', 'path'], axis=1))


# In[8]:


p = PCA(random_state=0)

train = train.join(pd.DataFrame(p.fit_transform(train.drop(['Id', 'Pawpularity', 'path'], axis=1))))
test = test.join(pd.DataFrame(p.transform(test.drop(['Id', 'path'], axis=1))))


# In[9]:


train, val= train_test_split(train, test_size=0.2, random_state=0)


# In[10]:


AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_SIZE = 299
BATCH_SIZE = 64


# In[11]:


train = train[['Pawpularity']].join(train.drop('Pawpularity', axis=1))
val = val[['Pawpularity']].join(val.drop('Pawpularity', axis=1))


# In[12]:


def process_data(path, meta, augment=False, label=True):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.image.central_crop(img, 1.0)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = keras.applications.efficientnet.preprocess_input(img)
    img = tf.cast(img, dtype=tf.float64)
    
    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_saturation(img, 0.9, 1.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        
    if label:
        return (img, meta[1:]), meta[0]
    return (img, meta), 0


# In[13]:


# train_ds = tf.data.Dataset.from_tensor_slices((train['path'], train.drop(['path', 'Id'], axis=1).astype(float))).map(lambda x,y: process_data(x, y, True)).batch(BATCH_SIZE).prefetch(AUTOTUNE)
# val_ds = tf.data.Dataset.from_tensor_slices((val['path'], val.drop(['path', 'Id'], axis=1).astype(float))).map(process_data).batch(BATCH_SIZE).prefetch(AUTOTUNE)
test_ds = ds.from_tensor_slices((test['path'], test.drop(['path', 'Id'], axis=1).astype(float))).map(lambda x,y: process_data(x, y, False, False)).batch(BATCH_SIZE).prefetch(AUTOTUNE)


# ### Build the model

# In[14]:


eff_model = keras.models.load_model('/kaggle/input/keras-applications-models/Xception.h5')
eff_model.trainable = False

def get_model():
    img_input = tfl.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    meta_input = tfl.Input(shape=(32,))

    X = eff_model(img_input)
    X = tfl.BatchNormalization()(X)

    con = tfl.concatenate([X, meta_input])

    X = tfl.Dense(64, activation='relu')(con)
    X = tfl.Dense(64, activation='relu')(X)
    
    X = tfl.Dropout(0.3)(X)

    out = tfl.Dense(1)(X)

    model = keras.Model(inputs=[img_input, meta_input], outputs=out)
    
    return model


# In[15]:


model = get_model()


# In[16]:


tf.keras.utils.plot_model(model, show_shapes=True)


# In[17]:


k = 5
fold = KFold(k,shuffle=True)


# In[18]:


models = []
histories = []

for i, (t_ids, v_ids) in enumerate(fold.split(train)):
    
    keras.backend.clear_session()

    print("\n\n===========================================================================================\n")
    train_ds = ds.from_tensor_slices((train.iloc[t_ids]['path'], train.iloc[t_ids].drop(['path', 'Id'], axis=1).astype(float))).map(lambda x,y: process_data(x, y, True)).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    
    val_ds = ds.from_tensor_slices((train.iloc[v_ids]['path'], train.iloc[v_ids].drop(['path', 'Id'], axis=1).astype(float))).map(process_data).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    
    model = get_model()
    
    early_stop = keras.callbacks.EarlyStopping(
        patience=3,
        restore_best_weights=True)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True)
    
    model.compile(keras.optimizers.Adam(learning_rate=lr_schedule), 
            loss='mse', 
            metrics=[keras.metrics.RootMeanSquaredError()])
    
    history = model.fit(train_ds,
                   validation_data=val_ds,
                   epochs=20,
                   callbacks=[early_stop])
    
    models.append(model)
    histories.append(history)


# In[19]:


# preds = model.predict(test_ds)
preds = models[0].predict(test_ds)/k

for i in range(1,k):
    preds += models[i].predict(test_ds)/k


# In[20]:


preds


# In[21]:


test['Pawpularity'] = preds
test[['Id', 'Pawpularity']].to_csv('submission.csv', index=False)


# In[ ]:




