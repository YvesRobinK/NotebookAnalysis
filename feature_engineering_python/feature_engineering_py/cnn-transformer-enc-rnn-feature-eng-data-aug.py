#!/usr/bin/env python
# coding: utf-8

# This is a essentially a combination of ['Transformer Encoder Implementation'](https://www.kaggle.com/arunprathap/transformer-encoder-implementation) by [Arun P R](https://www.kaggle.com/arunprathap) and [GRU+LSTM with feature engineering and augmentation](https://www.kaggle.com/its7171/gru-lstm-with-feature-engineering-and-augmentation) by [tito](https://www.kaggle.com/its7171), please check out their work as well.
# 
# I have not tuned any hyperparameters. If you do so and find better results, please let me know in the comments.
# 
# So far I have tried the following while working on this:
# 
# 1. Embedding -> CNN -> RNN -> Transformer (did not do as good as current model)
# 2. Added a 'Position' value for each value in sequence - surprisingly helped (ie: 0 for A 1 for B in ABCDE)

# ## 1. Import libraries

# In[1]:


import os 
import sys
import json
import math
import random
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm

import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, KFold,  StratifiedKFold, GroupKFold

from sklearn.cluster import KMeans

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L

import warnings
warnings.filterwarnings("ignore")


# In[2]:


seed = 42


# In[3]:


DEVICE = "TPU"
if DEVICE == "TPU":
    print("connecting to TPU...")
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        print("Could not connect to TPU")
        tpu = None

    if tpu:
        try:
            print("initializing  TPU ...")
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
            print("TPU initialized")
        except _:
            print("failed to initialize TPU")
    else:
        DEVICE = "GPU"

if DEVICE != "TPU":
    #if tf.config.list_physical_devices('gpu'):
    #    strategy = tf.distribute.MirroredStrategy()#if using multiple gpu
    #else:  # use default strategy
    strategy = tf.distribute.get_strategy()

if DEVICE == "GPU":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))  
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


# In[4]:


dropout_model = 0.36
hidden_dim_first = 128
hidden_dim_second = 256
hidden_dim_third = 128


# ## 2. Read & Process Datasets - Including Augmented Data

# In[5]:


# Download datasets
train = pd.read_json('../input/bpps-data-included/out_train (1).json')
test = pd.read_json('../input/bpps-data-included/out_test (1).json')
sample_sub = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")


# In[6]:


train.head()


# In[7]:


# Target columns 
target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}

def preprocess_inputs(df, cols=['sequence','predicted_loop_type','structure']):
    base_fea = np.transpose(
        np.array(
            df[cols]
            .applymap(lambda seq: [token2int[x] for x in seq])
            .values
            .tolist()
        ),
        (0, 2, 1)
    )
    bpps_sum_fea = np.array(df['bpps_sum'].to_list())[:,:,np.newaxis]
    bpps_max_fea = np.array(df['bpps_max'].to_list())[:,:,np.newaxis]
    bpps_nb_fea = np.array(df['bpps_nb'].to_list())[:,:,np.newaxis]
    
    # This kind of helps...
    _, position_fea = np.mgrid[0:bpps_nb_fea.shape[0]:1, 0:bpps_nb_fea.shape[1]:1]/(bpps_nb_fea.shape[1]-1)
    
    return np.concatenate([base_fea,bpps_sum_fea,bpps_max_fea,bpps_nb_fea, position_fea[:,:,np.newaxis]], 2)

# clustering for  GroupKFold
kmeans_model = KMeans(n_clusters=200, random_state=110).fit(preprocess_inputs(train)[:,:,0])
train['cluster_id'] = kmeans_model.labels_


# In[8]:


aug_df = pd.read_csv('../input/augmented-data/aug_data1.csv')
display(aug_df.head())


# Please check out [this notebook](https://www.kaggle.com/its7171/how-to-generate-augmentation-data) as well to see how this was generated

# In[9]:


def aug_data(df):
    target_df = df.copy()
    new_df = aug_df[aug_df['id'].isin(target_df['id'])]
                         
    del target_df['structure']
    del target_df['predicted_loop_type']
    new_df = new_df.merge(target_df, on=['id','sequence'], how='left')

    df['cnt'] = df['id'].map(new_df[['id','cnt']].set_index('id').to_dict()['cnt'])
    df['log_gamma'] = 100
    df['score'] = 1.0
    df = df.append(new_df[df.columns])
    return df
train = aug_data(train)
test = aug_data(test)


# In[10]:


train.shape


# In[11]:


train_inputs_all = preprocess_inputs(train)
train_labels_all = np.array(train[target_cols].values.tolist()).transpose((0, 2, 1))


# In[12]:


train_inputs_all.shape


# ## 3. Model Implementation, Training & Prediction

# In[13]:


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.
  
    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable 
            to (..., seq_len_q, seq_len_k). Defaults to None.
    
    Returns:
     output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

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
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'depth': self.depth,
            'wq': self.wq,
            'qk': self.wk,
            'wv': self.wv,
            'dense': self.dense,
        })
        
        return config
def point_wise_feed_forward_network(d_model, dff):
      return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training):
        #mask made None
        attn_output, _ = self.mha(x, x, x, None)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_heads': self.num_heads,
            'rate': self.rate,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'mha': self.mha,
            'ffn': self.ffn,
        })
        return config
    
def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer = 'orthogonal'))

def lstm_layer(hidden_dim, dropout):
    return L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer = 'orthogonal'))

def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)


# In[14]:


def build_model(model_type=1, seq_len=107, pred_len=68, embed_dim=200, 
                dropout=dropout_model, hidden_dim_first = hidden_dim_first, 
                hidden_dim_second = hidden_dim_second, hidden_dim_third = hidden_dim_third):
    
    inputs = tf.keras.layers.Input(shape=(seq_len, 7))

    # Extract features
    categorical_feat_dim = 3
    categorical_fea = inputs[:, :, :categorical_feat_dim]
    numerical_fea = inputs[:, :, 3:6]
    positional_fea = tf.expand_dims(inputs[:, :, 6], axis=2) 
    
    # Categorical embedding
    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(categorical_fea)
    reshaped = tf.reshape(
        embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))

    # Concatenate with numerical
    reshaped = L.concatenate([reshaped, numerical_fea], axis=2)
    reshaped = tf.keras.layers.SpatialDropout1D(.2)(reshaped)
    
    # Convolve
    conv = L.Conv1D(255, 5, padding='same', activation=tf.keras.activations.swish)(reshaped)

    # Concatenate with positional
    reshaped = L.concatenate([conv, positional_fea], axis=2) 
    
    # Transformer x2 - SWAPPED POSITION WITH RNN
    hidden = EncoderLayer(256, 128, 512)(reshaped)

    hidden = EncoderLayer(256, 128, 512)(hidden)
    
    # RNN
    if model_type == 0:
        hidden = gru_layer(256, 0.3)(hidden)
        hidden = gru_layer(256, 0.3)(hidden)
    elif model_type == 1:
        hidden = lstm_layer(256, 0.3)(hidden)
        hidden = gru_layer(256, 0.3)(hidden)
    elif model_type == 2:
        hidden = gru_layer(256, 0.3)(hidden)
        hidden = lstm_layer(256, 0.3)(hidden)
    elif model_type == 3:
        hidden = lstm_layer(256, 0.3)(hidden)
        hidden = lstm_layer(256, 0.3)(hidden)  

    truncated = hidden[:, :pred_len]

    out = tf.keras.layers.Dense(len(target_cols), activation='linear')(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    adam = tf.optimizers.Adam()
    model.compile(optimizer=adam, loss=MCRMSE, metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    return model


# In[15]:


tf.keras.backend.clear_session()
from tqdm.keras import TqdmCallback
lr_callback = tf.keras.callbacks.ReduceLROnPlateau()
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',restore_best_weights=True,min_delta=0.001, patience=10)


# In[16]:


mse_s = []
rmse_s = []

def train_and_predict(n_folds=5, model_name="model", model_type=0, epochs=100, debug=True,
                      dropout_model=dropout_model, hidden_dim_first = hidden_dim_first, 
                      hidden_dim_second = hidden_dim_second, hidden_dim_third = hidden_dim_third,
                      seed=seed):

    print("Model:", model_name)

    ensemble_preds = pd.DataFrame(index=sample_sub.index, columns=target_cols).fillna(0) # test dataframe with 0 values
    kf = KFold(n_folds, shuffle=True, random_state=seed)
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    gkf = GroupKFold(n_splits=n_folds)
    val_losses = []
    historys = []
    
    
    for i, (train_index, val_index) in enumerate(gkf.split(train, train['reactivity'], train['cluster_id'])):
        print("Fold:", str(i+1))
        with strategy.scope():
            model_train = build_model(model_type=model_type, 
                                      dropout=dropout_model, 
                                      hidden_dim_first = hidden_dim_first, 
                                      hidden_dim_second = hidden_dim_second, 
                                      hidden_dim_third = hidden_dim_third)
            model_short = build_model(model_type=model_type, seq_len=107, pred_len=107,
                                      dropout=dropout_model, 
                                      hidden_dim_first = hidden_dim_first, 
                                      hidden_dim_second = hidden_dim_second, 
                                      hidden_dim_third = hidden_dim_third)
            model_long = build_model(model_type=model_type, seq_len=130, pred_len=130,
                                     dropout=dropout_model, 
                                     hidden_dim_first = hidden_dim_first, 
                                     hidden_dim_second = hidden_dim_second, 
                                     hidden_dim_third = hidden_dim_third)

        train_inputs, train_labels = train_inputs_all[train_index], train_labels_all[train_index]
        
        val = train.iloc[val_index]
        x_val_all = preprocess_inputs(val)
        
        val = val[val.SN_filter == 1]
        
        val_inputs = preprocess_inputs(val)
        val_labels = np.array(val[target_cols].values.tolist()).transpose((0, 2, 1))
        
        w_trn = np.log(train.iloc[train_index].signal_to_noise+1.1)/2

        checkpoint = tf.keras.callbacks.ModelCheckpoint(f'{model_name}_Fold_{str(i+1)}.h5')

        history = model_train.fit(
            train_inputs , train_labels, 
            validation_data=(val_inputs,val_labels),
            batch_size=64,
            sample_weight=w_trn/2,
            epochs=epochs, 
            callbacks=[checkpoint,
                       lr_callback,
                       TqdmCallback(),
                       tf.keras.callbacks.TerminateOnNaN(),
                       es_callback],
            verbose= 0
        )

        holdouts = train.iloc[val_index]
        holdout_preds = model_train.predict(x_val_all)
        holdout_labels = np.array(holdouts[target_cols].values.tolist()).transpose((0, 2, 1))
        
        rmse = ((holdout_labels - holdout_preds) ** 2).mean() ** .5
        mse = ((holdout_labels - holdout_preds) ** 2).mean()

        print(f"{model_name} Min training loss={min(history.history['loss'])}, min validation loss={min(history.history['val_loss'])}")
        
        print(f"{model_name} Holdouts mse ={mse}, Holdouts rmse ={rmse}")
        mse_s.append(mse)
        rmse_s.append(rmse)
        mse_s_t.append(mse)
        rmse_s_t.append(rmse)
        
        val_losses.append(min(history.history['val_loss']))
        historys.append(history)
        
        model_short.load_weights(f'{model_name}_Fold_{str(i+1)}.h5')
        model_long.load_weights(f'{model_name}_Fold_{str(i+1)}.h5')

        public_preds = model_short.predict(public_inputs)
        private_preds = model_long.predict(private_inputs)

        preds_model = []
        for df, preds in [(public_df, public_preds), (private_df, private_preds)]:
            for i, uid in enumerate(df.id):
                single_pred = preds[i]

                single_df = pd.DataFrame(single_pred, columns=target_cols)
                single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

                preds_model.append(single_df)
            
        preds_model_df = pd.concat(preds_model).groupby('id_seqpos').mean().reset_index()
        
        ensemble_preds[target_cols] += preds_model_df[target_cols].values / n_folds

        if debug:
            print("Intermediate ensemble result")
            print(ensemble_preds[target_cols].head())

    ensemble_preds["id_seqpos"] = preds_model_df["id_seqpos"].values
    ensemble_preds = pd.merge(sample_sub["id_seqpos"], ensemble_preds, on="id_seqpos", how="left")

    print("Mean Validation loss:", str(np.mean(val_losses)))

    if debug:
        fig, ax = plt.subplots(1, 3, figsize = (20, 10))
        for i, history in enumerate(historys):
            ax[0].plot(history.history['loss'])
            ax[0].plot(history.history['val_loss'])
            ax[0].set_title('model_'+str(i+1))
            ax[0].set_ylabel('Loss')
            ax[0].set_xlabel('Epoch')
            
            ax[1].plot(history.history['root_mean_squared_error'])
            ax[1].plot(history.history['val_root_mean_squared_error'])
            ax[1].set_title('model_'+str(i+1))
            ax[1].set_ylabel('RMSE')
            ax[1].set_xlabel('Epoch')
            
            ax[2].plot(history.history['lr'])
            ax[2].set_title('model_'+str(i+1))
            ax[2].set_ylabel('LR')
            ax[2].set_xlabel('Epoch')
        plt.show()

    return ensemble_preds


public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()
public_inputs = preprocess_inputs(public_df)
private_inputs = preprocess_inputs(private_df)

ensembles = []

for i in range(1):
    model_name = "model_"+str(i+1)
    
    mse_s_t = []
    rmse_s_t = []

    ensemble = train_and_predict(n_folds=5, model_name=model_name, model_type=i, epochs=60,
                                 dropout_model=dropout_model, hidden_dim_first = hidden_dim_first, 
                                 hidden_dim_second = hidden_dim_second, hidden_dim_third = hidden_dim_third,
                                 seed=seed)
    ensembles.append(ensemble)
    print("RMSE Avg ", np.array(rmse_s_t).mean())
    print("MSE Avg ", np.array(mse_s_t).mean())


# ## 4. Ensembling the solutions and submission
# 

# In[17]:


# Score to beat when making changes
print("RMSE Avg ", np.array(rmse_s).mean())
print("MSE Avg ", np.array(mse_s).mean())


# In[18]:


# Ensembling the solutions
ensemble_final = ensembles[0].copy()
ensemble_final[target_cols] = 0

for ensemble in ensembles:
    ensemble_final[target_cols] += ensemble[target_cols].values / len(ensembles)

ensemble_final.head().T


# In[19]:


blend_preds_df = pd.DataFrame()
blend_preds_df['id_seqpos'] = ensemble_final['id_seqpos']
blend_preds_df['reactivity'] = ensemble_final['reactivity'] 
blend_preds_df['deg_Mg_pH10'] = ensemble_final['deg_Mg_pH10']
blend_preds_df['deg_pH10'] = ensemble_final['deg_Mg_pH10']
blend_preds_df['deg_Mg_50C'] = ensemble_final['deg_Mg_50C']
blend_preds_df['deg_50C'] = ensemble_final['deg_Mg_50C']
blend_preds_df.head().T


# In[20]:


# Submission
blend_preds_df.to_csv('submission.csv', index=False)

