#!/usr/bin/env python
# coding: utf-8

# ---
# # [Titanic - Machine Learning from Disaster][1]
# 
# **Goal:** To predict if a passenger survived the sinking of the Titanic or not.
# 
# ---
# ### **The aim of this notebook is to**
# - **1. Conduct Exploratory Data Analysis (EDA) and Feature Engineering.**
# - **2. Create and train a Deep Learning model with TensorFlow.**
# - **3. Optimize the neural network architecture with Optuna.**
# - **4. Learn how to use TPU.**
# - **5. Learn how to use AutoML (H2O AutoML).**
# ---
# #### **Note:** 
# - You can run this notebook on CPU, GPU, and TPU without changing codes.
# - In this notebook, training model on TPU takes more time than on GPU or CPU, because of the small batch size, small datasets, ect. Please understand that I didn't optimize the experiment parameters for TPU.
# - It would take much time to run optimization codes with many trials on TPU, so it is recommended to run on CPU or GPU.
# 
# ---
# #### **References:**
#  Thanks to previous great codes, blogs, and notebooks.
# - [How to Use Kaggle: Tensor Processing Units (TPUs)][2]
# - [AutoML: Automatic Machine Learning][5]
# - [H2O AutoML Tutorial][6]
# - [Automated Machine Learning with H2O][7]
# 
# ---
# #### **My Previous Notebooks:**
# - This competition is a basic classification task. If you are also interested in basic regression task, **[my notebook of House Prices competition][3]** would be useful.
# - If you would like to know more about other deep learning models for tabular data, you can find it in **[my notebook of Spaceship Titanic competition][4]**.
# 
# ---
# ### **If you find this notebook useful, or when you copy&edit this notebook, please do give me an upvote. It helps me keep up my motivation.**
# 
# ---
# [1]: https://www.kaggle.com/competitions/titanic
# [2]: https://www.kaggle.com/docs/tpu
# [3]: https://www.kaggle.com/code/masatomurakawamm/houseprices-deeplearning-eda-automl-pycaret
# [4]: https://www.kaggle.com/code/masatomurakawamm/spaceshiptitanic-eda-tabtransformer-tensorflow
# [5]: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
# [6]: https://github.com/h2oai/h2o-tutorials/tree/master/h2o-world-2017/automl
# [7]: https://towardsdatascience.com/automated-machine-learning-with-h2o-258a2f3a203f

# <h1 style="background:#05445E; border:0; border-radius: 12px; color:#D3D3D3"><center>0. TABLE OF CONTENTS</center></h1>
# 
# <ul class="list-group" style="list-style-type:none;">
#     <li><a href="#1" class="list-group-item list-group-item-action">1. Settings</a></li>
#     <li><a href="#2" class="list-group-item list-group-item-action">2. Data Loading</a></li>
#     <li><a href="#3" class="list-group-item list-group-item-action">3. EDA and Feature Engineering</a>
#         <ul class="list-group" style="list-style-type:none;">
#             <li><a href="#3.1" class="list-group-item list-group-item-action">3.1 AutoEDA with Sweetviz</a></li>
#             <li><a href="#3.2" class="list-group-item list-group-item-action">3.2 Feature Selection</a></li>
#             <li><a href="#3.3" class="list-group-item list-group-item-action">3.3 Target Distribution</a></li>
#             <li><a href="#3.4" class="list-group-item list-group-item-action">3.4 Numerical Features</a></li>
#             <li><a href="#3.5" class="list-group-item list-group-item-action">3.5 Categorical Features</a></li>
#             <li><a href="#3.6" class="list-group-item list-group-item-action">3.6 Validation Split</a></li>
#         </ul>
#     </li>
#     <li><a href="#4" class="list-group-item list-group-item-action">4. Deep Learning</a>
#         <ul class="list-group" style="list-style-type:none;">
#             <li><a href="#4.1" class="list-group-item list-group-item-action">4.1 Creating Dataset</a></li>
#             <li><a href="#4.2" class="list-group-item list-group-item-action">4.2 Creating Model</a></li>
#             <li><a href="#4.3" class="list-group-item list-group-item-action">4.3 Training Model</a></li>
#             <li><a href="#4.4" class="list-group-item list-group-item-action">4.4 Inference</a></li>
#         </ul>
#     </li>
#     <li><a href="#5" class="list-group-item list-group-item-action">5. Optimization</a></li>
#     <li><a href="#6" class="list-group-item list-group-item-action">6. Cross Validation and Ensebmling</a></li>
#     <li><a href="#7" class="list-group-item list-group-item-action">7. AutoML</a>
#         <ul class="list-group" style="list-style-type:none;">
#             <li><a href="#7.1" class="list-group-item list-group-item-action">7.1 Set up</a></li>
#             <li><a href="#7.2" class="list-group-item list-group-item-action">7.2 Create Training Data</a></li>
#             <li><a href="#7.3" class="list-group-item list-group-item-action">7.3 Run AutoML</a></li>
#             <li><a href="#7.4" class="list-group-item list-group-item-action">7.4 Explainability</a></li>
#         </ul>
#     </li>
# </ul>

# <a id ="1"></a><h1 style="background:#05445E; border:0; border-radius: 12px; color:#D3D3D3"><center>1. Settings</center></h1>

# In[1]:


## Parameters
data_config = {'train.csv': '../input/titanic/train.csv',
               'test.csv': '../input/titanic/test.csv',
               'gender_submission.csv': '../input/titanic/gender_submission.csv',
              }

exp_config = {'competition_name': 'titanic',
              'n_splits': 5,
              'normalization': 'Robust',
              'encoding': 'one_hot',
              'n_sample_per_TPU_core': 16,
              'batch_size': 128,
              'learning_rate': 5e-4,
              'label_smoothing': 0.01,
              'train_epochs': 100,
              'checkpoint_filepath': './tmp/model/exp.ckpt',
              'cross_validation': True,
             }

model_config = {'model_input_shape': (57, ),
                'model_units': [64, 48, 32],
                'dropout_rates': [0., 0.1, 0.1],
               }

opt_config = {'opt_flg': True,
              'opt_trials': 30,
              'opt_epochs': 60,
              'opt_batch_size': 256}

print('Parameters setted!')


# In[2]:


## Import dependencies 
import numpy as np
import pandas as pd
import scipy as sp

import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import KFold, StratifiedKFold

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

import os
import pathlib
import gc
import sys
import re
import math 
import random
import time 
import datetime as dt
import pprint
from tqdm import tqdm 

print('Import done!')


# In[3]:


## For reproducible results    
def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['PYTHONHASHSEED'] = str(s) 
    print('Seeds setted!')
    
global_seed = 42
seed_all(global_seed)


# ---
# # [TPU] Distribution Strategy #
# 
# A TPU has eight different *cores* and each of these cores acts as its own accelerator. (A TPU is sort of like having eight GPUs in one machine.) We tell TensorFlow how to make use of all these cores at once through a **distribution strategy**. Run the following cell to create the distribution strategy that we'll later apply to our model. We'll use the distribution strategy when we create our neural network model. Then, TensorFlow will distribute the training among the eight TPU cores by creating eight different replicas of the model, one for each core.

# In[4]:


# Detect TPU, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() 
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() 

print("REPLICAS: ", strategy.num_replicas_in_sync)


# ---

# <a id ="2"></a><h1 style="background:#05445E; border:0; border-radius: 12px; color:#D3D3D3"><center>2. Data Loading</center></h1>

# ---
# ## [TPU] Loading the Competition Data ##
# 
# When used with TPUs, datasets need to be stored in a [Google Cloud Storage bucket](https://cloud.google.com/storage/). You can use data from any public GCS bucket by giving its path just like you would data from `'/kaggle/input'`. The following will retrieve the GCS path for this competition's dataset.

# In[5]:


competition_name = exp_config['competition_name']

## Get GCS Path
from kaggle_datasets import KaggleDatasets

if tpu:
    DATA_DIR = KaggleDatasets().get_gcs_path(competition_name) 
    
    save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
    load_locally = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    
    for file_path in tf.io.gfile.glob(os.path.join(DATA_DIR, "*")):
        file_name = file_path.split('/')[-1]
        data_config[file_name] = file_path
    
else:
    DATA_DIR = '/kaggle/input/' + competition_name
    save_locally = None
    load_locally = None

print(f"Data Directory Path: {DATA_DIR}\n")
print("Contents of Data Directory:")
for file in tf.io.gfile.glob(os.path.join(DATA_DIR, "*")):
    print(f"\t{file}")


# After Loading data, we can conduct EDA or Feature Engineering just as like on CPU/GPU.
# 
# ---

# ### [File and Data Field Descriptions](https://www.kaggle.com/competitions/titanic/data)
# 
# - **train.csv** - the training set
# - **test.csv** - the test set
# - **gender_submission.csv** - a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like
# 
# 
# ---
# ### [Submission & Evaluation](https://www.kaggle.com/competitions/titanic/overview/evaluation)
# 
# -  For each in the test set, you must predict a 0 or 1 value for the variable. Your score is the percentage of passengers you correctly predict. This is known as accuracy.

# In[6]:


## Data Loading
train_df = pd.read_csv(data_config['train.csv'])
test_df = pd.read_csv(data_config['test.csv'])
submission_df = pd.read_csv(data_config['gender_submission.csv'])

print(f'train_length: {len(train_df)}')
print(f'test_lenght: {len(test_df)}')
print(f'submission_length: {len(submission_df)}')


# In[7]:


## Null Value Check
print('train_df.info()'); print(train_df.info(), '\n')
print('test_df.info()'); print(test_df.info(), '\n')

## train_df Check
train_df.head()


# <a id ="3"></a><h1 style="background:#05445E; border:0; border-radius: 12px; color:#D3D3D3"><center>3. EDA and Feature Engineering</center></h1>

# <a id ="3.1"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>3.1 AutoEDA with Sweetviz</center></h2>

# In[8]:


## Import dependencies
get_ipython().system('pip install -U -q sweetviz')
import sweetviz
print('import done!')


# In[9]:


#my_report = sweetviz.analyze(train_df, "Survived")
my_report = sweetviz.compare([train_df, "Train"], [test_df, "Test"], "Survived")
my_report.show_notebook()


# <a id ="3.2"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>3.2 Feature Selection</center></h2>

# In[10]:


for column in train_df.columns:
    print(f"# of unique values in {column}: {train_df[column].nunique()}")


# In[11]:


target = 'Survived'
categorical_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Cabin', 'Embarked']
numerical_features = ['Age', 'Fare']


# <a id ="3.3"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>3.3 Target Distribution</center></h2>

# In[12]:


## Interactive Target Distribution Plot with plotly
target_count = train_df.groupby(target)['PassengerId'].count()
target_percent = target_count / target_count.sum()

## 1. Make Figure object
fig = go.Figure()

## 2. Make trace (graph object)
data = go.Bar(x=target_count.index.astype(str).values,
              y=target_count.values)

## 3. Add the trace to the Figure
fig.add_trace(data)

## 4. Setting layouts
fig.update_layout(title=dict(text='Target distribution'),
                  xaxis=dict(title='Survived values'),
                  yaxis=dict(title='counts'))

## 5. Show the Figure
fig.show()


# <a id ="3.4"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>3.4 Numerical Features</center></h2>

# In[13]:


## Statistics of training data
train_df[numerical_features].describe().T.style.bar(subset=['mean'],)\
                        .background_gradient(subset=['std'], cmap='coolwarm')\
                        .background_gradient(subset=['50%'], cmap='coolwarm')\
                        .background_gradient(subset=['max'], cmap='coolwarm')


# In[14]:


## Statistics of test data
test_df[numerical_features].describe().T.style.bar(subset=['mean'],)\
                        .background_gradient(subset=['std'], cmap='coolwarm')\
                        .background_gradient(subset=['50%'], cmap='coolwarm')\
                        .background_gradient(subset=['max'], cmap='coolwarm')


# In[15]:


## Interactive Heatmap of Correlation Matrix with plotly
train_numerical = train_df[numerical_features + ['Survived']]

fig = px.imshow(train_numerical.corr(),
                color_continuous_scale='RdBu_r',
                color_continuous_midpoint=0, 
                aspect='auto')
fig.update_layout(height=300, 
                  width=300,
                  title = "Heatmap",                  
                  showlegend=False)
fig.show()


# In[16]:


## Plotting distribution of numerical features with seaborn
bins = 20
fig = plt.figure(figsize=(10, 5))
for i, nf in enumerate(numerical_features):
    ax = fig.add_subplot(1, 2, i+1)
    sns.histplot(data=train_df,
                 x=nf,
                 bins=bins,
                 kde=True,
                 hue=target,
                 ax=ax)
    plt.title(nf)
    plt.xlabel(None)
fig.tight_layout()


# In[17]:


## Fill NaN in numerical columns with its median
train_df[numerical_features] = train_df[numerical_features].fillna(train_df[numerical_features].median()) 
test_df[numerical_features] = test_df[numerical_features].fillna(train_df[numerical_features].median()) 


# ### Binning for Numerical Features

# In[18]:


## Binning for "Age"
age_bins = np.array([i*5 for i in range(18)])
age_bins


# In[19]:


## Binning for "Fare"
fare_mean = train_df['Fare'].mean()
print(f'fare mean: {fare_mean}\n')

fare_quantiles = train_df['Fare'].quantile([0, 0.05, 0.1, 0.5, 0.8, 0.9, 0.95, 0.99, 1])
print(f'fare quantiles: {fare_quantiles}\n')

fare_uniques = train_df['Fare'].unique()
fare_uniques.sort()
print(f'fare uniques: {fare_uniques}\n')

fare_bins = np.array([0, 10, 20, 30, 40, 50, 75, 100, 200, 500, 1_000])
print(f'fare_bins: {fare_bins}\n')


# In[20]:


def binning(dataframe, column, bins):
    df = dataframe.copy()
    splits = pd.cut(df[column], 
                    bins=bins, 
                    labels=False, ## For return of integer index.
                    right=False)
    df[column] = splits
    return df

## Binning for numerical features
train = binning(train_df, 'Age', age_bins)
train = binning(train, 'Fare', fare_bins)

test = binning(test_df, 'Age', age_bins)
test = binning(test, 'Fare', fare_bins)

## After binning, 'Age' and 'Fare' become categorical features.
categorical_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Cabin', 'Embarked', 'Age', 'Fare']


# <a id ="3.5"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>3.5 Categorical Features</center></h2>

# In[21]:


## Plotting distribution of categorical features with seaborn
fig = plt.figure(figsize=(10, 20))
for i, cf in enumerate(categorical_features):
    ax = fig.add_subplot(4, 2, i+1)
    sns.histplot(data=train_df,
                 x=cf,
                 hue=target,
                 kde=False,
                 ax=ax)
    plt.title(cf)
    plt.xlabel(None)
fig.tight_layout()


# ### Feature Engineering on 'Cabin'

# In[22]:


train_df['Cabin'].unique()


# In[23]:


def get_cabin_alphabet(cabin):
    if cabin is np.NaN:
        return 'NA'
    else:
        return cabin[0]
    
train['Cabin_alphabet'] = train['Cabin'].map(get_cabin_alphabet)
test['Cabin_alphabet'] = test['Cabin'].map(get_cabin_alphabet)

train = train.drop('Cabin', axis=1)
test = test.drop('Cabin', axis=1)

categorical_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Cabin_alphabet', 'Embarked', 'Age', 'Fare']


# In[24]:


## Plotting distribution of 'Cabin_alphabet' with seaborn
sns.histplot(data=train,
             x='Cabin_alphabet',
             hue=target,
             kde=False)


# ### Data Encoding

# In[25]:


## Fill NaN in categorical columns with its mode
train[categorical_features] = train[categorical_features].fillna(train[categorical_features].mode().iloc[0])  
test[categorical_features] = test[categorical_features].fillna(train[categorical_features].mode().iloc[0])  


# In[26]:


def df_encode(categorical_features,
              train,
              test,
              valid=None,
              encoding='one_hot',
              encoder=None,
              return_encoder=False):
    
    if encoder is not None:
        enc = encoder
    else:
        if encoding == 'one_hot':
            enc = preprocessing.OneHotEncoder(handle_unknown='ignore',
                                              sparse=False,
                                              dtype=np.int32)
        elif encoding == 'label':
            enc = preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value',
                                               unknown_value=-1,
                                               dtype=np.int32)
        enc.fit(train[categorical_features])
        
    train_categorical = pd.DataFrame(enc.transform(train[categorical_features]),
                                     columns=enc.get_feature_names())
    test_categorical = pd.DataFrame(enc.transform(test[categorical_features]),
                                    columns=enc.get_feature_names())
    
    if valid is not None:
        valid_categorical = pd.DataFrame(enc.transform(valid[categorical_features]),
                                         columns=enc.get_feature_names())
        if return_encoder:
            return train_categorical, valid_categorical, test_categorical, enc
        else:
            return train_categorical, valid_categorical, test_categorical
        
    else:
        if return_encoder:
            return train_categorical, test_categorical, enc
        else:
            return train_categorical, test_categorical


# In[27]:


## One-Hot Encoding
encoding = exp_config['encoding']

_, _, enc = df_encode(categorical_features,
                      train,
                      test,
                      encoding=encoding,
                      return_encoder=True)

train, test = df_encode(categorical_features,
                        train,
                        test,
                        encoding=encoding,
                        encoder=enc)

train[target] = train_df[target]
print(train.columns)
print(train.shape, test.shape)


# <a id ="3.6"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>3.6 Validation Split</center></h2>

# In[28]:


## K-Fold validation split
n_splits = exp_config['n_splits']

#kf = KFold(n_splits=n_splits)
skf = StratifiedKFold(n_splits=n_splits)

train['k_folds'] = -1

#for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
for fold, (train_idx, valid_idx) in enumerate(skf.split(X=train,
                                                        y=train[target])):
    train['k_folds'][valid_idx] = fold
        
for i in range(n_splits):
    print(f"fold {i}: {len(train.query('k_folds==@i'))} samples")


# In[29]:


## Hold-out validation split
valid_fold = train.query('k_folds == 0').reset_index(drop=True)
train_fold = train.query('k_folds != 0').reset_index(drop=True)

train_fold = train_fold.drop(['k_folds'], axis=1)
valid_fold = valid_fold.drop(['k_folds'], axis=1)

print(len(train_fold), len(valid_fold))


# <a id ="4"></a><h1 style="background:#05445E; border:0; border-radius: 12px; color:#D3D3D3"><center>4. Deep Learning</center></h1>

# <a id ="4.1"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>4.1 Creating Dataset</center></h2>

# ---
# ## [TPU] Batch size ##
# 
# To go fast on a TPU, increase the batch size. The rule of thumb is to use batches of 128 elements per core (ex: batch size of 128*8=1024 for a TPU with 8 cores). At this size, the 128x128 hardware matrix multipliers of the TPU (see hardware section below) are most likely to be kept busy. You start seeing interesting speedups from a batch size of 8 per core though. In the sample above, the batch size is scaled with the core count through this line of code:

# In[30]:


if tpu:
    n_sample_per_TPU_core = exp_config['n_sample_per_TPU_core']
    batch_size = n_sample_per_TPU_core * strategy.num_replicas_in_sync
else:
    batch_size = exp_config['batch_size']


# ---

# In[31]:


def df_to_dataset(data_frame,
                  target_column=None,
                  shuffle=False, repeat=False,
                  batch_size=5, drop_remainder=False):
    
    df = data_frame.copy()
    
    if target_column is not None:
        target = df.pop(target_column) ##PandasArray
        data = df.values
        ds = tf.data.Dataset.from_tensor_slices((data, target))
    else:
        data = df.values
        ds = tf.data.Dataset.from_tensor_slices(data)
        
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(batch_size)
    
    return ds


# In[32]:


## Create datasets
batch_size = exp_config['batch_size']

train_ds = df_to_dataset(train_fold,
                         target_column=target,
                         shuffle=True,
                         repeat=False,
                         batch_size=batch_size,
                         drop_remainder=False,)
    
valid_ds = df_to_dataset(valid_fold,
                         target_column=target,
                         shuffle=False,
                         repeat=False,
                         batch_size=batch_size,
                         drop_remainder=False,)

f, t = next(iter(train_ds))
print(f.shape, t.shape)


# <a id ="4.2"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>4.2 Creating Model</center></h2>

# In[33]:


def create_training_model(input_shape, model_units=[128,], dropout_rates=[0.2]):
    
    model_inputs = layers.Input(shape=input_shape)
    x = model_inputs
    
    for units, dropout_rate in zip(model_units, dropout_rates):
        feedforward = keras.Sequential([
            layers.Dense(units, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(dropout_rate),
        ])
        x = feedforward(x)
        
    final_layer = layers.Dense(units=1, activation=None)
    model_outputs = final_layer(x)
    
    training_model = tf.keras.Model(inputs=model_inputs,
                                    outputs=model_outputs)
    return training_model


# --- 
# ## [TPU] Model on TPUs ##
# 
# The strategy scope instructs Tensorflow to instantiate all the variables of the model in the memory of the TPU. The TPUClusterResolver.connect() call automatically enters the TPU device scope which instructs Tensorflow to run Tensorflow operations on the TPU. 

# In[34]:


## Create training model
input_shape = model_config['model_input_shape']
model_units = model_config['model_units']
dropout_rates = model_config['dropout_rates']

if tpu:
    with strategy.scope():
        training_model = create_training_model(input_shape=input_shape,
                                               model_units=model_units, 
                                               dropout_rates=dropout_rates)
else:
    training_model = create_training_model(input_shape=input_shape,
                                           model_units=model_units, 
                                           dropout_rates=dropout_rates)

## Model compile and build
lr = exp_config['learning_rate']
label_smoothing = exp_config['label_smoothing']
optimizer = keras.optimizers.Adam(learning_rate=lr)
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                             label_smoothing=label_smoothing)

training_model.compile(optimizer=optimizer,
                       loss=loss_fn,
                       metrics=['acc'])

training_model.summary()


# ---

# <a id ="4.3"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>4.3 Training Model</center></h2>

# ---
# ## [TPU] Model saving/loading on TPUs ##
# When loading and saving models TPU models from/to the local disk, the `experimental_io_device `option must be used. It can be omitted if writing to GCS because TPUs have direct access to GCS. This option does nothing on GPUs.
# 
# TPU users will remember that in order to train a model on TPU, you have to instantiate the model in a TPUStrategy scope. The strategy scope instructs Tensorflow to instantiate all the variables of the model in the memory of the TPU. The TPUClusterResolver.connect() call automatically enters the TPU device scope which instructs Tensorflow to run Tensorflow operations on the TPU. Now if you call model.save('./model') when you are connected to a TPU, Tensorflow will try to run the save operations on the TPU and since the TPU is a network-connected accelerator that has no access to your local disk, the operation will fail. Notice that saving to GCS will work though. The TPU does have access to GCS. If you want to save a TPU model to your local disk, you need to run the saving operation on your local machine and that is what the `experimental_io_device='/job:localhost'` flag does.

# In[35]:


def model_training(training_model,
                   train_ds,
                   vali_ds,
                   epochs,
                   batch_size,
                   steps_per_epoch,
                   verbose=1,
                   fold=None,
                   model_save=True):
    
    ## For saving the best model
    checkpoint_filepath = exp_config['checkpoint_filepath'] ## './tmp/model/exp.ckpt'
    if fold is not None:
        l = checkpoint_filepath.split('/')  ## ['.', 'tmp', 'model', 'exp.ckpt']
        l[2] = l[2] + '_' + str(fold)
        checkpoint_filepath = '/'.join(l) ## f'./tmp/model_{fold}/exp.ckpt'
        
    if tpu:
        save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath, 
            save_weights_only=False, 
            monitor='val_loss', 
            mode='min', 
            save_best_only=True,
            options=save_locally)  
    else:
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath, 
            save_weights_only=True, 
            monitor='val_loss', 
            mode='min', 
            save_best_only=True)
        
    ## For the adjustment of learning rate
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        cooldown=10,
        min_lr=1e-5,
        verbose=verbose)
    
    if model_save:
        callbacks = [model_checkpoint_callback, reduce_lr]
    else:
        callbacks = [reduce_lr]
    
    ## Model training
    history = training_model.fit(train_ds,
                                 epochs=epochs,
                                 shuffle=True,
                                 validation_data=valid_ds,
                                 callbacks=callbacks,
                                 verbose=verbose,
                                 )
    
    ## Load the best parameters
    if model_save:
        if tpu:
            with strategy.scope():
                load_locally = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
                training_model = tf.keras.models.load_model(checkpoint_filepath,
                                                            options=load_locally)
        else:
            training_model.load_weights(checkpoint_filepath)
        
    return history


# ---

# In[36]:


## Settings for Training
epochs = exp_config['train_epochs']
batch_size = exp_config['batch_size']
steps_per_epoch = len(train_ds)//batch_size 

history = model_training(training_model,
                         train_ds,
                         valid_ds,
                         epochs,
                         batch_size,
                         steps_per_epoch,
                         verbose=1,
                         fold=None)


# In[37]:


## Plot the train and valid losses
def plot_history(hist, title=None, valid=True):
    plt.figure(figsize=(7, 5))
    plt.plot(np.array(hist.index), hist['loss'], label='Train Loss')
    if valid:
        plt.plot(np.array(hist.index), hist['val_loss'], label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title)
    plt.show()
    
hist = pd.DataFrame(history.history)
plot_history(hist)


# <a id ="4.4"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>4.4 Inference</center></h2>

# In[38]:


## Create test dataset
test_ds = df_to_dataset(test,
                        target_column=None,
                        shuffle=False,
                        repeat=False,
                        batch_size=batch_size,
                        drop_remainder=False,)

f = next(iter(test_ds))
print(f.shape)


# In[39]:


logits = training_model.predict(test_ds)
probs = tf.math.sigmoid(logits)
probs = np.squeeze(probs)
preds = np.where(probs < 0.5, 0, 1)

submission_df[target] = preds

submission_df.to_csv('submission_dnn.csv', index=False)
submission_df.head(10)


# <a id ="5"></a><h1 style="background:#05445E; border:0; border-radius: 12px; color:#D3D3D3"><center>5. Optimization</center></h1>

# <img src="https://www.preferred.jp/wp-content/themes/preferred/assets/img/projects/optuna/pict01.jpg" width="200"/>
# 
# [Optuna™](https://www.preferred.jp/en/projects/optuna/) is an open-source automatic hyperparameter optimization framework. It automatically finds optimal hyperparameter values based on an optimization target.

# In[40]:


def create_trial_model(trial):
    model = keras.Sequential()
    model.add(layers.Input(shape=model_config['model_input_shape']))
    
    activation = trial.suggest_categorical('activation', ['relu', 'gelu', 'selu'])
    n_layers = trial.suggest_int('n_layers', 1, 3)
    for i in range(n_layers):
        n_units = trial.suggest_discrete_uniform(f'units_{i}', 24, 256, 12)
        dropout_rate = trial.suggest_uniform(f'dropout_{i}', 0, 0.5)
        model.add(layers.Dense(n_units, use_bias=True, activation=activation))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(units=1, activation=None))
    
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    label_smoothing = trial.suggest_uniform('label_smoothing', 0, 0.2)
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                 label_smoothing=label_smoothing)
    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=['acc'])
    
    return model


# In[41]:


def objective(trial):
    if tpu:
        with strategy.scope():
            compiled_model = create_trial_model(trial)
    else:
        compiled_model = create_trial_model(trial)
        
    epochs = opt_config['opt_epochs']
    batch_size = opt_config['opt_batch_size']
    steps_per_epoch = len(train_ds)//batch_size 
        
    history = model_training(compiled_model,
                             train_ds,
                             valid_ds,
                             epochs,
                             batch_size,
                             steps_per_epoch,
                             verbose=0,
                             fold=None,
                             model_save=False)
    
    return min(history.history['val_loss'])


# In[42]:


if opt_config['opt_flg']:
    import optuna
    n_trials = opt_config['opt_trials']
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)


# In[43]:


if opt_config['opt_flg']:
    best_params = study.best_params
    pprint.pprint(best_params)


# In[44]:


optuna.visualization.plot_optimization_history(study)


# In[45]:


optuna.visualization.plot_slice(study)


# <a id ="6"></a><h1 style="background:#05445E; border:0; border-radius: 12px; color:#D3D3D3"><center>6. Cross Validation and Ensebmling</center></h1>

# In[46]:


if exp_config['cross_validation']:
    ## Settings for Training
    batch_size = exp_config['batch_size']
    
    cv_results = submission_df.drop('Survived', axis=1)
    cv_results['probs_mean'] = 0.
    
    ## Create test dataset
    test_ds = df_to_dataset(test,
                            target_column=None,
                            shuffle=False,
                            repeat=False,
                            batch_size=batch_size,
                            drop_remainder=False,)
    
    ## Create cross validation samples
    for fold in range(exp_config['n_splits']):
        valid_fold = train.query(f'k_folds == {fold}').reset_index(drop=True)
        train_fold = train.query(f'k_folds != {fold}').reset_index(drop=True)
        
        train_fold = train_fold.drop(['k_folds'], axis=1)
        valid_fold = valid_fold.drop(['k_folds'], axis=1)
        
        ## Create datasets
        train_ds = df_to_dataset(train_fold,
                                 target_column=target,
                                 shuffle=True,
                                 repeat=False,
                                 batch_size=batch_size,
                                 drop_remainder=False,)
        
        valid_ds = df_to_dataset(valid_fold,
                                 target_column=target,
                                 shuffle=False,
                                 repeat=False,
                                 batch_size=batch_size,
                                 drop_remainder=False,)
        
        ## Create training model
        input_shape = model_config['model_input_shape']
        model_units = model_config['model_units']
        dropout_rates = model_config['dropout_rates']
        
        if tpu:
            with strategy.scope():
                training_model = create_training_model(input_shape=input_shape,
                                                       model_units=model_units, 
                                                       dropout_rates=dropout_rates)
        else:
            training_model = create_training_model(input_shape=input_shape,
                                                   model_units=model_units, 
                                                   dropout_rates=dropout_rates)
            
        ## Model compile and build
        lr = exp_config['learning_rate']
        label_smoothing = exp_config['label_smoothing']
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                     label_smoothing=label_smoothing)
        training_model.compile(optimizer=optimizer,
                               loss=loss_fn,
                               metrics=['acc'])
        
        ## Model training
        epochs = exp_config['train_epochs']
        batch_size = exp_config['batch_size']
        steps_per_epoch = len(train_ds)//batch_size 
        
        history = model_training(training_model,
                                 train_ds,
                                 valid_ds,
                                 epochs,
                                 batch_size,
                                 steps_per_epoch,
                                 verbose=0,
                                 fold=fold)
        
        ## Plot the train and valid losses
        hist = pd.DataFrame(history.history)
        plot_history(hist, title=f'fold: {fold}')
        
        ## Inference
        logits = training_model.predict(test_ds)
        probs = tf.math.sigmoid(logits)
        probs = np.squeeze(probs)
        cv_results[f'prods_{fold}'] = probs
        cv_results['probs_mean'] += probs
        
    ## Ensebmle the inferences of cross-validations
    cv_results['probs_mean'] /= exp_config['n_splits']
    probs_mean = cv_results['probs_mean'].values
    preds = np.where(probs_mean > 0.5, 1, 0)
    submission_df[target] = preds
    
    submission_df.to_csv('submission_cv.csv', index=False)


# In[47]:


if exp_config['cross_validation']:
    submission_df.head(10)


# In[48]:


if exp_config['cross_validation']:
    cv_results.head(10)


# <a id ="7"></a><h1 style="background:#05445E; border:0; border-radius: 12px; color:#D3D3D3"><center>7. AutoML</center></h1>

# <img src="https://docs.h2o.ai/h2o/latest-stable/h2o-docs/_images/h2o-automl-logo.jpg" width="200"/>
# 
# AutoML (Automatic Machine Learning) is the process of automating algorithm selection, feature generation, hyperparameter tuning, iterative modeling, and model assessment. [H2O AutoML](https://h2o.ai/platform/h2o-automl/) can be used for automating the machine learning workflow with a simple interface in R, Python, or a web GUI.

# <a id ="7.1"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>7.1 Set up</center></h2>

# In[49]:


## Install and Import dependencies
#!pip install h2o -q
import h2o
from h2o.automl import H2OAutoML

## Initialize the H2O cluster
h2o.init()


# <a id ="7.2"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>7.2 Create Training Data</center></h2>

# In[50]:


## Load the dataset as a H2OFrame
train_h2o_df = h2o.import_file(data_config['train.csv'])
test_h2o_df = h2o.import_file(data_config['test.csv'])

## How to make a H2OFrame from Pandas DataFrame
#train_h2o_df = h2o.H2OFrame(train)
#test_h2o_df = h2o.H2OFrame(test)


# In[51]:


## Describe the dataset
train_h2o_df.describe(chunk_summary=False)


# For classification, target should be encoded as categorical (aka. "factor" or "enum"). As described, `Survived` column is encoded as a 0/1 "int", thus we have to convert the column as follows:

# In[52]:


## Convert the column into categorical
train_h2o_df['Survived'] = train_h2o_df['Survived'].asfactor()
train_h2o_df.describe()


# <a id ="7.3"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>7.3 Run AutoML</center></h2>

# In[53]:


## Create AutoML Models
aml = H2OAutoML(max_models=10,
                exclude_algos=['GBM'],
                max_runtime_secs=120,
                balance_classes=True, ## This option is only applicable for classification.
                seed=42)


# The `max_models` argument specifies the number of individual (or "base") models, and does not include the two ensemble models that are trained at the end. The current version of H2O AutoML trains and cross-validates the models in the following order:
# 
# 1. three pre-specified XGBoost GBM (Gradient Boosting Machine) models,
# 2. a fixed grid of GLMs,
# 3. a default Random Forest (DRF),
# 4. five pre-specified H2O GBMs,
# 5. a near-default Deep Neural Net,
# 6. an Extremely Randomized Forest (XRT),
# 7. a random grid of XGBoost GBMs,
# 8. a random grid of H2O GBMs,
# 9. and a random grid of Deep Neural Nets.
# 
# In addition, it also trains the two ensemble models:
# 
# 1. a stacked ensemble of all the models trained above
# 2. a “Best of Family” Stacked Ensemble that contains the best performing model for each algorithm class
# 
# **Nonte:** Particular algorithms (or groups of algorithms) can be switched on/off using the `include_algos` and`exclude_algos` argument.
# 
# In some cases, there will not be enough time to complete all the algorithms, so some may be missing from the leaderboard.

# In[54]:


## Feature Selection
x = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Age', 'Fare']
y = 'Survived'

aml.train(training_frame=train_h2o_df,
          x=x, y=y)


# By default and when `nfolds` > 1, models will be evaluated using k-fold cross validation. Thus, when you would like to specify validation_frame for holdout validation, run the following codes:

# In[55]:


#train_h2o, valid_h2o = train_h2o_df.split_frame(ratios=[0.8], seed=42)  ## Validation split
#aml = H2OAutoML(max_models=10,
#                exclude_algos=['GBM'],
#                max_runtime_secs=120,
#                balance_classes=True,
#                nfolds=0,
#                seed=42)
#aml.train(training_frame=train_h2o,
#          validation_frame=valid_h2o,
#          leaderboard_frame=valid_h2o,
#          x=x,　y=y,)


# After the models are trained, we can compare the model performance using the leaderboard. When we did not specify a `leaderboard_frame` in the `H2OAutoML.train()` method, the AutoML leaderboard uses cross-validation metrics to score and rank the models.

# In[56]:


lb = aml.leaderboard
lb.head(rows=lb.nrows)


# In the case of binary classification, the default ranking metric is Area Under the ROC Curve (AUC).

# In[57]:


## Get the top model of leaderboard
best_model = aml.leader
#best_model = aml.get_best_model() ## same result

print(best_model)


# In[58]:


## Save and load the model
model_path = h2o.save_model(model=best_model,
                            path='./automl_model', 
                            force=True)
print(model_path)
loaded_model = h2o.load_model(path=model_path)


# In[59]:


## Inference
probs = aml.predict(test_h2o_df)
#probs = best_model.predict(test_h2o_df) ## same result

probs = h2o.as_list(probs) ## Convert to pandas DataFrame
preds = probs['predict'].values
submission_df[target] = preds

submission_df.to_csv('submission_automl.csv', index=False)
submission_df.head(20)


# <a id ="7.4"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>7.4 Explainability</center></h2>

# H2O AutoML provides insights into model’s global explainability (such as variable importance, partial dependence plot, SHAP values, and model correlation) and local explainability for individual records.

# In[60]:


## Global explainability for models
explain_model = aml.explain(frame=train_h2o_df, figsize=(8, 6))


# In[61]:


## Local explainability for individual records
row_index = 1
aml.explain_row(frame=train_h2o_df, row_index=row_index, figsize=(8, 6))

