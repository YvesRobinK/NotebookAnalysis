#!/usr/bin/env python
# coding: utf-8

#  # <h1 style='background:#F7B2B0; border:0; color:black'><center>THE GATEDTABTRANSFORMER - AN ENHANCED DEEP LEARNING ARCHITECTURE FOR TABULAR MODELING.</center></h1> 

# This is a two part tutorial series containing the implementation of [GatedTabTransformer](https://arxiv.org/pdf/2201.00199.pdf) paper in both FLAX and TensorFlow (TPU)
# 
# [Part 1 : GatedTabTransformer in FLAX](https://www.kaggle.com/code/usharengaraju/gatedtabtransformer-flax)
# 
# [Part 2 : GatedTabTransformer in TensorFlow + TPU](https://www.kaggle.com/code/usharengaraju/tensorflow-tpu-gatedtabtransformer)

# ### Abstract
# 
# Over the last few years the research towards using deep learning for tabular data has been on the rise .The state of the art TabTransformer incorporates an attention mechanism to better track relationships between categorical features and then makes use of a standard MLP to output its final logits.GatedTabTransformer implements linear projections are implemented in the MLP block and the paper also experiments with several activation functions .
# 
# ### Introduction
# 
# Tabular data is the most commonly used data type in real world applications . Tree based ensemble methods like LightGBM , XGBoost are the current state of the art approaches for tabular data . Over the last few years , there is increasing interest in the usage of deep learning techniques for tabular data primarily because of the eliminating the need for manual embedding and feature engineering . Some of the neural networks architectures which have comparable performance with Tree based ensemble methods are TabNet , DNF-Net etc.
# 
# There is an increasing usage of attention-based architectures like Transformers which was originally used to  handle NLP tasks to solve tabular data problems . TabTransformer is one such architecture which focuses on using Multi-Head Self Attention blocks to model relationships between the categorical features in tabular data, transforming them into robust contextual embeddings.The transformed categorical features are concatenated with continuous values and then fed through a standard multilayer perceptron which makes TabTransformer significantly outperform other deep learning counterparts like TabNet and MLP . GatedTabTransformer further enhances TabTransformer by replacing the final MLP block with a gated multi-layer perceptron (gMLP) , a simple MLP-based network with spatial gating projections, which aims to be on par with Transformers in terms of performance on sequential data
# 
# 
# ### TabTransformer    
# 
# The TabTransformer model, outperforms the other state-of-the-art deep learning methods for tabular data by at least 1.0% on mean AUROC. It consists of a column embedding layer, a stack of N Transformer layers, and a multilayer perceptron . The inputted tabular features are split in two parts for the categorical and continuous values. Column embedding is performed for each categorical feature .It generates parametric embeddings which are inputted to a stack of Transformer layers. Each Transformer layer consists of a multi-head self-attention layer followed by a position-wise feed-forward layer. After the processing of categorical values , they are concatenated along with the continuous values to form a final feature vector which is inputted to a standard multilayer perceptron
# 
# ![](https://i.imgur.com/mnv2bLy.png)
# 
# 
# ### gMLP model
# 
# The gMLP model consists of a stack of multiple identically structured blocks ,activation function and linear projections along the channel dimension and the spatial gating unit which captures spatial cross-token interactions. The weights are initialized as near-zero values and the biases as ones at the beginning of training. This structure does not require positional embeddings because relevant information will be captured in the gating units. gMLP has been proposed as an alternative to Transformers for NLP and vision tasks having up to 66% less trainable parameters. GatedTabTransformers replaces the pure MLP block in the TabTransformer with gMLP 
# 
# ![](https://i.imgur.com/SRyVmYY.png)
# 
# 
# 
# ### GatedTabTransformer
# 
# The column embeddings are generated from categorical data features and continuous values are passed through a normalization layer. The categorical embeddings are then processed by a Transformer block. 
# 
# Transformer block represents the encoder part of a Transformer . It has two sub-layers - a multi-head self-attention mechanism, and a simple, position wise fully connected feed-forward network. In the final layer MLP is replaced by gMLP and the architecture is adapted to output classification logits and works best for optimization of cross entropy or binary cross entropy loss 
# 
# ![](https://i.imgur.com/ROdKcQy.png)
# 

# <img src="https://camo.githubusercontent.com/dd842f7b0be57140e68b2ab9cb007992acd131c48284eaf6b1aca758bfea358b/68747470733a2f2f692e696d6775722e636f6d2f52557469567a482e706e67">
# 
# > I will be integrating W&B for visualizations and logging artifacts!
# > 
# > [GatedTabTransformer in FLAX project on W&B Dashboard](https://wandb.ai/usharengaraju/GatedTabTransformer_FLAX)
# > 
# > - To get the API key, create an account in the [website](https://wandb.ai/site) .
# > - Use secrets to use API Keys more securely

# # **<span style="color:#F7B2B0;">W & B Artifacts</span>**
# 
# An artifact as a versioned folder of data.Entire datasets can be directly stored as artifacts .
# 
# W&B Artifacts are used for dataset versioning, model versioning . They are also used for tracking dependencies and results across machine learning pipelines.Artifact references can be used to point to data in other systems like S3, GCP, or your own system.
# 
# You can learn more about W&B artifacts [here](https://docs.wandb.ai/guides/artifacts)
# 
# ![](https://drive.google.com/uc?id=1JYSaIMXuEVBheP15xxuaex-32yzxgglV)

# In[1]:


import wandb
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
wandb_key = user_secrets.get_secret("api_key")
wandb.login(key = wandb_key)


# In[2]:


# Save training data to W&B Artifacts
run = wandb.init(project='GatedTabTransformer_FLAX', name='processed_data') 
artifact = wandb.Artifact(name='processed_data',type='dataset')
artifact.add_file("/kaggle/input/amex-tfrecords/minidata (1).csv")
wandb.log_artifact(artifact)
wandb.finish()


# ![](https://i.imgur.com/tRDVISy.png)

# In[3]:


get_ipython().system('pip install --quiet tensorflow-addons')


# In[4]:


import tensorflow as tf
from kaggle_datasets import KaggleDatasets
def configure_device():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()  # connect to tpu cluster
        strategy = tf.distribute.TPUStrategy(tpu) # get strategy for tpu
        print('Num of TPUs: ', strategy.num_replicas_in_sync)
        device='TPU'
    except: # otherwise detect GPUs
        tpu = None
        gpus = tf.config.list_logical_devices('GPU') # get logical gpus
        ngpu = len(gpus)
        if ngpu: # if number of GPUs are 0 then CPU
            strategy = tf.distribute.MirroredStrategy(gpus) # single-GPU or multi-GPU
            print("> Running on GPU", end=' | ')
            print("Num of GPUs: ", ngpu)
            device='GPU'
        else:
            print("> Running on CPU")
            strategy = tf.distribute.get_strategy() # connect to single gpu or cpu
            device='CPU'
    return strategy, device, tpu


# In[5]:


strategy, device, tpu = configure_device()
AUTO = tf.data.experimental.AUTOTUNE


# In[6]:


import numpy as np

import pandas as pd
df  = pd.read_csv('../input/amex-tfrecords/minidata (1).csv')


# In[7]:


df.select_dtypes('int').columns


# In[8]:


cat_features  = ['B_30', 'B_38', 'D_63', 'D_64', 'D_66', 'D_68', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126']
cont_features = [x for x in list(df.columns) if x not in cat_features]


# In[9]:


ff = list(df.select_dtypes('int').columns)
tt  = [x for x in list(df.columns) if x not in ff]

def parse_tfr_element(element):
  data = {}
  for col in ff:
    data[col] = tf.io.FixedLenFeature([], tf.int64)

  for col in tt:
    data[col] = tf.io.FixedLenFeature([], tf.float32)
  
    
  content = tf.io.parse_single_example(element, data)
  
  my_arr = []
  for col in list(df.columns):
    my_arr.append(content[col])
  return my_arr

def get_dataset_small(filename):
  dataset = tf.data.TFRecordDataset(filename)
  dataset = dataset.map(
      parse_tfr_element
  )
  return dataset
gspath = KaggleDatasets().get_gcs_path('amex-tfrecords')
filename = tf.io.gfile.glob(gspath + '/*.tfrecord')
dataset_small = get_dataset_small(filename[0])


# In[10]:


from tqdm import tqdm
arr = []
for i in tqdm(dataset_small.take(1000)):
  temp = [j.numpy() for j in i]
  arr.append(temp)


# In[11]:


arr = np.array(arr)
arr.shape


# In[12]:


df1 = pd.DataFrame(arr,columns=list(df.columns))
df1


# In[13]:


def get_X_from_groups(feature_set, groups):
    result = []
    for group in groups:
        result.append(feature_set[group])
    return result

def get_X_from_features(feature_set, cont_features, cat_features):
    groups = [cont_features]
    groups.extend(cat_features)
    return get_X_from_groups(feature_set, groups)


# In[14]:


cont_features.remove('target')
len(cont_features)


# In[15]:


cats = []
for col in cat_features:
  cats.append(df1[col].unique().shape[0])


# In[16]:


X = get_X_from_features(df1.drop(columns=['target'],axis=1),cont_features,cat_features)
y = df1['target']


# In[17]:


import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
class gMLPLayer(layers.Layer):
    def __init__(self, num_patches, embedding_dim, dropout_rate, *args, **kwargs):
        super(gMLPLayer, self).__init__(*args, **kwargs)

        self.channel_projection1 = keras.Sequential(
            [
                layers.Dense(units=embedding_dim * 2),
                tfa.layers.GELU(),
                layers.Dropout(rate=dropout_rate),
            ]
        )

        self.channel_projection2 = layers.Dense(units=embedding_dim)

        self.spatial_projection = layers.Dense(
            units=num_patches, bias_initializer="Ones"
        )

        self.normalize1 = layers.LayerNormalization(epsilon=1e-6)
        self.normalize2 = layers.LayerNormalization(epsilon=1e-6)

    def spatial_gating_unit(self, x):
        # Split x along the channel dimensions.
        # Tensors u and v will in th shape of [batch_size, num_patchs, embedding_dim].
        u, v = tf.split(x, num_or_size_splits=2, axis=2)
        # Apply layer normalization.
        v = self.normalize2(v)
        # Apply spatial projection.
        v_channels = tf.linalg.matrix_transpose(v)
        v_projected = self.spatial_projection(v_channels)
        v_projected = tf.linalg.matrix_transpose(v_projected)
        # Apply element-wise multiplication.
        return u * v_projected

    def call(self, inputs):
        # Apply layer normalization.
        x = self.normalize1(inputs)
        # Apply the first channel projection. x_projected shape: [batch_size, num_patches, embedding_dim * 2].
        x_projected = self.channel_projection1(x)
        # Apply the spatial gating unit. x_spatial shape: [batch_size, num_patches, embedding_dim].
        x_spatial = self.spatial_gating_unit(x_projected)
        # Apply the second channel projection. x_projected shape: [batch_size, num_patches, embedding_dim].
        x_projected = self.channel_projection2(x_spatial)
        # Add skip connection.
        return x + x_projected


# In[18]:


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        # parametreleri
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        # batch-layer
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class GatedTabTransformer(keras.Model):

    def __init__(self, 
            categories,
            num_continuous,
            dim,
            dim_out,
            depth,
            embedding_dim,
            heads,
            attn_dropout,
            ff_dropout,
            gmlp_blocks,
            normalize_continuous = True):

        super(GatedTabTransformer, self).__init__()

        # --> continuous inputs
        self.embedding_dim = embedding_dim
        self.normalize_continuous = normalize_continuous
        if normalize_continuous:
            self.continuous_normalization = layers.LayerNormalization()

        # --> categorical inputs

        # embedding
        self.embedding_layers = []
        for number_of_classes in categories:
            self.embedding_layers.append(layers.Embedding(input_dim = number_of_classes, output_dim = dim))

        # concatenation
        self.embedded_concatenation = layers.Concatenate(axis=1)

        # adding transformers
        self.transformers = []
        for _ in range(depth):
            self.transformers.append(TransformerBlock(dim, heads, dim))
        self.flatten_transformer_output = layers.Flatten()

        # --> MLP
        self.pre_mlp_concatenation = layers.Concatenate()

        # mlp layers
        self.gmlp_layers = []
        for _ in range(gmlp_blocks):
            self.gmlp_layers.append(gMLPLayer(1,self.embedding_dim,0.2))
        self.embedder2 = layers.Dense(self.embedding_dim)
        self.output_layer = layers.Dense(dim_out,activation='sigmoid')

    def call(self, inputs):
        continuous_inputs  = inputs[0]
        categorical_inputs = inputs[1:]
        
        # --> continuous
        if self.normalize_continuous:
            continuous_inputs = self.continuous_normalization(continuous_inputs)

        # --> categorical
        embedding_outputs = []
        for categorical_input, embedding_layer in zip(categorical_inputs, self.embedding_layers):
            embedding_outputs.append(embedding_layer(categorical_input))
        categorical_inputs = self.embedded_concatenation(embedding_outputs)
        # print(embedding_outputs[0].shape)
        
        for transformer in self.transformers:
            categorical_inputs = transformer(categorical_inputs)
        contextual_embedding = self.flatten_transformer_output(categorical_inputs)
        # print(categorical_inputs.shape)
        # --> MLP
        mlp_input = self.pre_mlp_concatenation([continuous_inputs, contextual_embedding])
        gmlp_input = tf.expand_dims(self.embedder2(mlp_input),axis=1)
        for gmlp_layer in self.gmlp_layers:
            gmlp_input = gmlp_layer(gmlp_input)
        gmlp_input = tf.math.reduce_mean(gmlp_input,axis=1)
        return self.output_layer(gmlp_input)
  


# In[19]:


with strategy.scope():
  model = GatedTabTransformer(
      categories = cats, # number of unique elements in each categorical feature
      num_continuous = 177,      # number of numerical features
      dim = 16,                # embedding/transformer dimension
      dim_out = 1,             # dimension of the model output
      depth = 6,  
      embedding_dim=256,             # number of transformer layers in the stack
      heads = 8,               # number of attention heads
      attn_dropout = 0.1,      # attention layer dropout in transformers
      ff_dropout = 0.1,        # feed-forward layer dropout in transformers
      gmlp_blocks = 6 # mlp layer dimensions and activations
  )

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  history = model.fit(X,y,epochs=20,validation_split=0.2,batch_size=32,verbose=1)


# In[20]:


df2 = df1.drop(columns=['target'],axis=1)
y = y.to_numpy()


# In[21]:


from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=True)
acc_per_fold=[]
loss_per_fold = []
# K-fold Cross Validation model evaluation
fold_no = 1
with strategy.scope():
  for train, test in kfold.split(df2, y):
    train = list(train)
    test = list(test)
    train_df = df2.iloc[train]
    test_df = df2.iloc[test]
    train_y = y[train]
    test_y = y[test]
    train_X = get_X_from_features(train_df,cont_features,cat_features)
    test_X = get_X_from_features(test_df,cont_features,cat_features)
    
    model = GatedTabTransformer(
        categories = cats, # number of unique elements in each categorical feature
        num_continuous = 177,      # number of numerical features
        dim = 16,                # embedding/transformer dimension
        dim_out = 1,             # dimension of the model output
        depth = 6,  
        embedding_dim=256,             # number of transformer layers in the stack
        heads = 8,               # number of attention heads
        attn_dropout = 0.1,      # attention layer dropout in transformers
        ff_dropout = 0.1,        # feed-forward layer dropout in transformers
        gmlp_blocks = 6 # mlp layer dimensions and activations
    )

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    history = model.fit(train_X,train_y,epochs=20,validation_split=0.2,batch_size=32,verbose=1)
    

    # Generate generalization metrics
    scores = model.evaluate(test_X, test_y, verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1


# In[22]:


model.save_weights('gatedtab.h5', overwrite=True)


# In[23]:


import matplotlib.pyplot as plt

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[24]:


pip install --quiet shap


# In[25]:


import shap

# print the JS visualization code to the notebook
shap.initjs()


# In[26]:


import matplotlib.pyplot as plt
def f(inp):
    inp = pd.DataFrame(inp,columns=list(df.columns))
    X = get_X_from_features(inp.drop(columns=['target'],axis=1),cont_features,cat_features)
    return model.predict(X).flatten()

explainer = shap.KernelExplainer(f, df1.iloc[:50,:])
shap_values = explainer.shap_values(df1.iloc[299,:], nsamples=500)
shap.force_plot(explainer.expected_value, shap_values, df1.iloc[299,:])


# In[27]:


shap_values50 = explainer.shap_values(df1.iloc[280:330,:], nsamples=500)
fig = shap.summary_plot(shap_values50, df1.iloc[280:330,:],show=False)
plt.savefig('shap.png', dpi=600, bbox_inches='tight')
plt.show()


# TabTransformers are highly robust against missing and noisy data and provide better interpretability . By replacing the final MLP block with gated Multilayer perceptron , GatedTabTranformers are able to achieve high accuracy in binary classification tasks.
# 
# ### References
# 
# https://arxiv.org/pdf/2201.00199.pdf
# 
# https://arxiv.org/pdf/2012.06678.pdf
# 
# https://www.tensorflow.org/
# 
# https://flax.readthedocs.io/
# 
# Pytorch Implementation : https://github.com/radi-cho/GatedTabTransformer
