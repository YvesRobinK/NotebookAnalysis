#!/usr/bin/env python
# coding: utf-8

# # **TPS Dec - Multi Head Attention**
# 
# * This notebook is based on [tensorflow homepage](https://www.tensorflow.org/)

# [](https://data-science-blog.com/wp-content/uploads/2022/01/mha_visualization-930x1030.png)

# In[ ]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, KFold

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


# # **Memory Reduce Func**
# 
# * **If you don't use some memory reducing strategy, you can face some OOM**

# In[ ]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
 
    return df


# # **Pseudolabeling**
# 
# * **[reference](https://www.kaggle.com/remekkinas/tps-12-nn-tpu-pseudolabeling-0-95661/notebook)**

# In[ ]:


train = pd.read_csv('../input/tabular-playground-series-dec-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-dec-2021/test.csv')
pl_df = pd.read_csv('../input/tps12-pseudolabels/tps12-pseudolabels_v1.csv')

train = pd.concat([train, pl_df], axis=0).reset_index(drop = True)
train


# In[ ]:


del pl_df
gc.collect()


# # **Feature Engineering**
# 
# * **Check Missing Values**
# * **Check Target Column**

# * **Soil_Type7 & Soil_Type15 → useless**

# In[ ]:


train = train.drop(columns = ['Id', 'Soil_Type7', 'Soil_Type15'])
test = test.drop(columns = ['Id', 'Soil_Type7', 'Soil_Type15'])


# ## **Check Missing Values**
# 
# * **Looking good!**

# In[ ]:


print('<----------Value Count of Missing Values in Train Data---------->\n', train.isna().sum())
print('<----------Value Count of Missing Values in Test Data---------->\n', test.isna().sum())


# ## **Check Target Column**
# 
# * **There is only one row of class5! We need to drop this**

# In[ ]:


train['Cover_Type'].value_counts()


# In[ ]:


train = train.drop(index = train[train['Cover_Type'] == 5].index).reset_index(drop = True)
train['Cover_Type'].value_counts()


# ## **Aspect**
# 
# * **Aspect means angle. Good for rescaling**
# * **Hillshade needs to rescale to 0 ~ 255**
# 
# From [gulshanmishra Kernel](https://www.kaggle.com/gulshanmishra/tps-dec-21-tensorflow-nn-feature-engineering)
# 
# Thank you for sharing nice notebook :)

# In[ ]:


train["Aspect"][train["Aspect"] < 0] += 360
train["Aspect"][train["Aspect"] > 359] -= 360

test["Aspect"][test["Aspect"] < 0] += 360
test["Aspect"][test["Aspect"] > 359] -= 360


# In[ ]:


train.loc[train["Hillshade_9am"] < 0, "Hillshade_9am"] = 0
test.loc[test["Hillshade_9am"] < 0, "Hillshade_9am"] = 0

train.loc[train["Hillshade_Noon"] < 0, "Hillshade_Noon"] = 0
test.loc[test["Hillshade_Noon"] < 0, "Hillshade_Noon"] = 0

train.loc[train["Hillshade_3pm"] < 0, "Hillshade_3pm"] = 0
test.loc[test["Hillshade_3pm"] < 0, "Hillshade_3pm"] = 0

train.loc[train["Hillshade_9am"] > 255, "Hillshade_9am"] = 255
test.loc[test["Hillshade_9am"] > 255, "Hillshade_9am"] = 255

train.loc[train["Hillshade_Noon"] > 255, "Hillshade_Noon"] = 255
test.loc[test["Hillshade_Noon"] > 255, "Hillshade_Noon"] = 255

train.loc[train["Hillshade_3pm"] > 255, "Hillshade_3pm"] = 255
test.loc[test["Hillshade_3pm"] > 255, "Hillshade_3pm"] = 255


# ## **Interaction Features**
# 
# * **Sum of Hydrology**
# * **Subtraction of Hydrology**

# In[ ]:


train['Sum_Hydrology'] = np.abs(train['Horizontal_Distance_To_Hydrology']) + np.abs(train['Vertical_Distance_To_Hydrology'])
train['Sub_Hydrology'] = np.abs(train['Horizontal_Distance_To_Hydrology']) - np.abs(train['Vertical_Distance_To_Hydrology'])

test['Sum_Hydrology'] = np.abs(test['Horizontal_Distance_To_Hydrology']) + np.abs(test['Vertical_Distance_To_Hydrology'])
test['Sub_Hydrology'] = np.abs(test['Horizontal_Distance_To_Hydrology']) - np.abs(test['Vertical_Distance_To_Hydrology'])


# # **Target Encoding**
# 
# ### **Need to use inverse_transform at the end for submission**

# In[ ]:


from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()
y = le.fit_transform(train['Cover_Type'])

gc.collect()


# # **Scaling**
# 
# * **Robust Scaling**

# In[ ]:


train = train.drop(columns = ['Cover_Type'])

cols = train.columns

# Scaling
rb = RobustScaler()

train[cols] = rb.fit_transform(train[cols].values)
test[cols] = rb.transform(test[cols].values)


# In[ ]:


train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
train = train.values
test = test.values
gc.collect()


# In[ ]:


from tensorflow.keras.utils import to_categorical

target = to_categorical(y)


# # **Modeling**

# ## **Multi-Head Attention**
# 
# * **Removed Batch Size (No Time Step here)**

# In[ ]:


def scaled_dot_product_attention(q, k, v, mask):
  matmul_qk = tf.matmul(q, k, transpose_b=True)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

  output = tf.matmul(attention_weights, v)

  return output, attention_weights


# In[ ]:


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (-1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (-1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights


# ## **Feed Forward Net**

# In[ ]:


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


# ## **Encoder Block**

# In[ ]:


class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask = None):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2


# In[ ]:


d_model = 54 # Embedding Dimension of our data
dropout_rate = 0.1


# ## **Model Builder**

# In[ ]:


def get_model():
    inputs = tf.keras.layers.Input(shape = (54))

    x = EncoderLayer(d_model, 6, 512, dropout_rate)(inputs)
    x = EncoderLayer(d_model, 6, 256, dropout_rate)(x)
    x = EncoderLayer(d_model, 6, 128, dropout_rate)(x)
    x = EncoderLayer(d_model, 6, 32, dropout_rate)(x)
    
    outputs = tf.keras.layers.Dense(6, activation = 'softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="encoder")
    
    return model


# In[ ]:


model = get_model()
tf.keras.utils.plot_model(model, show_shapes = True)


# # **Training**
# 
# ### **Pseudolabeling using [public data](https://www.kaggle.com/remekkinas/tps12-pseudolabels?select=tps12-pseudolabels_v1.csv)**
# #### **Thanks for sharing Remek Kinas!**
# 
# #### **You can increase the EPOCH for better result**

# In[ ]:


EPOCH = 2
BATCH_SIZE = 256
NUM_FOLDS = 5

kf = StratifiedKFold(n_splits = NUM_FOLDS, shuffle = True, random_state=2021)
test_preds = []

for fold, (train_idx, test_idx) in enumerate(kf.split(train, y)):
    print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)

    checkpoint_filepath = f"folds{fold}.hdf5"
    X_train, X_valid = train[train_idx], train[test_idx]
    y_train, y_valid = target[train_idx], target[test_idx]

    model = get_model()
    model.compile(optimizer = "adam",
                  loss = "categorical_crossentropy",
                  metrics = ['accuracy'])

    lr = ReduceLROnPlateau(monitor = "val_loss",
                           factor = 0.5,
                           patience = 1,
                           verbose = 1)

    es = EarlyStopping(monitor = "val_loss",
                       patience = 2,
                       verbose = 1,
                       restore_best_weights = True)

    sv = ModelCheckpoint(checkpoint_filepath,
                         monitor = 'val_loss',
                         verbose = 1,
                         save_best_only = True,
                         save_weights_only = True,
                         mode = 'auto',
                         save_freq = 'epoch',
                         options = None)

    model.fit(X_train,
              y_train,
              validation_data = (X_valid, y_valid),
              epochs = EPOCH,
              batch_size = BATCH_SIZE,
              callbacks = [lr, es, sv])

    test_preds.append(model.predict(test))

    del X_train, X_valid, y_train, y_valid, model
    gc.collect()


# In[ ]:


sub = pd.read_csv('../input/tabular-playground-series-dec-2021/sample_submission.csv')
sub['Cover_Type'] = le.inverse_transform(np.argmax(np.array(test_preds).sum(axis = 0), axis = 1))
sub


# In[ ]:


sub.to_csv('sub.csv', index = 0)

