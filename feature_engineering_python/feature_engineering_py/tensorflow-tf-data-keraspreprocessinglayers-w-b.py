#!/usr/bin/env python
# coding: utf-8

# ![](https://drive.google.com/uc?id=1OAEI8ghsx2CITu_vZVu9X9fnINtp_1oO)
# 
# For Tabular Playground Series - Jan 2022 , the problem deals with sales forecasting for two (fictitious) independent store chains selling Kaggle merchandise that want to become the official outlet for all things Kaggle. .
# 
# # **<span style="color:#e76f51;">Goal</span>**
#  
# The goal is to which of the store chains (KaggleMart or KaggleRama ) would have the best sales going forward
# 
# # **<span style="color:#e76f51;">Data</span>**
# 
# **Training Data**
# 
# > - ```train.csv``` -  the training set, which includes the sales data for each date-country-store-item combination.
# > - ```test.csv``` -  the test set; your task is to predict the corresponding item sales for each date-country-store-item combination. Note the Public leaderboard is scored on the first quarter of the test year, and the Private on the remaining.
# > - ```sample_submission.csv``` - a sample submission file in the correct format
# 
# # **<span style="color:#e76f51;">Metric</span>**
# 
# Submissions are evaluated on SMAPE between forecasts and actual values. SMAPE = 0 when the actual and predicted values are both 0.
# 
# 
# 🎯 The mean absolute percentage error is one of the most commonly used metrics for forecasting . MAPE is expressed as percentage and is scale independent . MAPE is not easily differentiable and asymmetric . MAPE also puts heavy penalties on the negative errors .
# 
# 🎯 Symmetric Mean Absolute Percentage Error (sMAPE) overcomes the shortcomings of MAPE and has both lower and upper bounds .
# 
# 🎯 SMAPE is calculated by taking square root of the squared difference between the forecast and the actual value .
# 
# ### SMAPE = SquareRoot(Squared(F - A))
# 
# Resources to understand SMAPE in detail are [Source1](https://www.brightworkresearch.com/the-problem-with-using-smape-for-forecast-error-measurement/) [Source2](https://towardsdatascience.com/choosing-the-correct-error-metric-mape-vs-smape-5328dec53fac)
# 
# 
# <img src="https://camo.githubusercontent.com/dd842f7b0be57140e68b2ab9cb007992acd131c48284eaf6b1aca758bfea358b/68747470733a2f2f692e696d6775722e636f6d2f52557469567a482e706e67">
# 
# > I will be integrating W&B for visualizations and logging artifacts!
# > 
# > [TPS Jan 2022 Project on W&B Dashboard]
# (https://wandb.ai/usharengaraju/TPSJan2022)
# > 
# > - To get the API key, create an account in the [website](https://wandb.ai/site) .
# > - Use secrets to use API Keys more securely 

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
import seaborn as sns
import wandb
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers


#ignore warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')



# In[2]:


try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    secret_value_0 = user_secrets.get_secret("api_key")
    wandb.login(key=secret_value_0)
    anony=None
except:
    anony = "must"
    print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')
    
CONFIG = dict(competition = 'TPSJan2022',_wandb_kernel = 'tensorgirl')


# In[3]:


train = pd.read_csv("../input/tabular-playground-series-jan-2022/train.csv", parse_dates=True)
test = pd.read_csv("../input/tabular-playground-series-jan-2022/test.csv", index_col=0, parse_dates=True)


# # **<span style="color:#e76f51;">Preprocessing</span>**
# 
# The training dataset is divided in to training and validation dataset . The date column is converted to pandas datetime object .

# In[4]:


train, val = np.split(train.sample(frac=1), [int(0.8*len(train))])
train['date'] = pd.to_datetime(train['date'])
val['date'] = pd.to_datetime(val['date'])
test['date'] = pd.to_datetime(test['date'])


# In[5]:


# code copied from https://www.kaggle.com/vad13irt/tps-jan-2022-exploratory-data-analysis?scriptVersionId=84140623&cellId=11
def hide_spines(ax, spines=["top", "right", "bottom", "left"]):
    for spine in spines:
        ax.spines[spine].set_visible(False)
        
chart_colors = ["#2a9d8f","#ff355d", "#E4916C"]
sns.palplot(chart_colors)
chart_colors1 = ["#2a9d8f","#ff355d"]
sns.palplot(chart_colors1)


# # **<span style="color:#e76f51;">Exploratory Data Analysis</span>**
# 
# [Source1](https://www.kaggle.com/ambrosm/tpsjan22-01-eda-which-makes-sense) [Source2](https://www.kaggle.com/vad13irt/tps-jan-2022-exploratory-data-analysis)
# 
# 📌 Histograms in the below graph are skewed with outliers . Hence choosing log(num_sold over num_sold is preferred .
# 
# 

# In[6]:


# code copied from https://www.kaggle.com/ambrosm/tpsjan22-01-eda-which-makes-sense?scriptVersionId=84561837&cellId=16

plt.figure(figsize=(18, 12))
for i, (combi, df) in enumerate(train.groupby(['country', 'store', 'product'])):
    ax = plt.subplot(6, 3, i+1, ymargin=0.5)
    ax.hist(train.num_sold, bins=50, color='#2a9d8f')
    #ax.set_xscale('log')
    ax.set_title(combi)
plt.suptitle('Histograms of num_sold', y=1.03)
plt.tight_layout(h_pad=3.0)
plt.show()


# 📌 The peaks in the below graph indicates lot of sales happens during January .

# In[7]:


# code copied from https://www.kaggle.com/vad13irt/tps-jan-2022-exploratory-data-analysis?scriptVersionId=84140623&cellId=21

fig = plt.figure(figsize=(25, 7))
fig.set_facecolor("#fff")
ax = fig.add_subplot()
ax.set_facecolor("#fff")
ax.grid(color="lightgrey", alpha=0.7, linewidth=1, axis="both", zorder=0)
sns.lineplot(x="date", y="num_sold", color="#2a9d8f", err_style=None, data=train, linewidth=1, ax=ax, zorder=2)
ax.yaxis.set_tick_params(color="#000", labelsize=12, pad=5, length=0)
ax.set_ylabel("Num Sold", fontsize=15, fontfamily="serif", labelpad=10)
ax.set_xlabel("Date", fontsize=15, fontfamily="serif", labelpad=10)
ax.xaxis.set_tick_params(color="#000", labelsize=12, pad=5, length=0)
ax.yaxis.set_tick_params(color="#000", labelsize=12, pad=5, length=0)
ax.set_title("Number of sales", loc="left", color="#000", fontsize=25, pad=5, fontweight="bold", fontfamily="serif", y=1.05, zorder=3)
hide_spines(ax)
fig.show()


# 📌 Norway has the highest sales followed by Sweden and Finland
# 

# In[8]:


# code copied from https://www.kaggle.com/vad13irt/tps-jan-2022-exploratory-data-analysis?scriptVersionId=84140623&cellId=23

fig = plt.figure(figsize=(25, 7))
fig.set_facecolor("#fff")
ax = fig.add_subplot()
ax.set_facecolor("#fff")
ax.grid(color="lightgrey", alpha=0.7, linewidth=1, axis="both", zorder=0)
sns.lineplot(x="date", y="num_sold", hue="country", color="#FECD00",palette=chart_colors, data=train, err_style=None, linewidth=1, ax=ax, zorder=2)
ax.yaxis.set_tick_params(color="#000", labelsize=12, pad=5, length=0)
ax.set_ylabel("Num Sold", fontsize=15, fontfamily="serif", labelpad=10)
ax.set_xlabel("Date", fontsize=15, fontfamily="serif", labelpad=10)
ax.xaxis.set_tick_params(color="#000", labelsize=12, pad=5, length=0)
ax.yaxis.set_tick_params(color="#000", labelsize=12, pad=5, length=0)
ax.set_title("Countries vs Number of sales", loc="left", color="#000", fontsize=25, pad=5, fontweight="bold", fontfamily="serif", y=1.05, zorder=3)
hide_spines(ax)
ax.legend(loc="upper right", ncol=3, fontsize=15, edgecolor=None, facecolor=None, markerscale=2, labelcolor="#000", handlelength=1, title=None)
fig.show()


# 📌 KaggleRama has higher sales compared to KaggleMart
# 
# 

# In[9]:


# code copied from https://www.kaggle.com/vad13irt/tps-jan-2022-exploratory-data-analysis?scriptVersionId=84140623&cellId=25

fig = plt.figure(figsize=(25, 7))
fig.set_facecolor("#fff")
ax = fig.add_subplot()
ax.set_facecolor("#fff")
ax.grid(color="lightgrey", alpha=0.7, linewidth=1, axis="both", zorder=0)
sns.lineplot(x="date", y="num_sold", data=train, hue="store",palette=chart_colors1, err_style=None, linewidth=1, ax=ax, zorder=2)
ax.yaxis.set_tick_params(color="#000", labelsize=12, pad=5, length=0)
ax.set_ylabel("Num Sold", fontsize=15, fontfamily="serif", labelpad=10)
ax.set_xlabel("Date", fontsize=15, fontfamily="serif", labelpad=10)
ax.xaxis.set_tick_params(color="#000", labelsize=12, pad=5, length=0)
ax.yaxis.set_tick_params(color="#000", labelsize=12, pad=5, length=0)
ax.set_title("Stores vs Number of sales", loc="left", color="#000", fontsize=25, pad=5, fontweight="bold", fontfamily="serif", y=1.05, zorder=3)
hide_spines(ax)
ax.legend(loc="upper right", ncol=3, fontsize=15, edgecolor=None, facecolor=None, markerscale=2, labelcolor="#000", handlelength=1, title=None)
fig.show()


# 📌 KaggleHat has the highest sales followed by KaggleMug and KaggleStickers

# In[10]:


# code copied from https://www.kaggle.com/vad13irt/tps-jan-2022-exploratory-data-analysis?scriptVersionId=84140623&cellId=27

fig = plt.figure(figsize=(25, 7))
fig.set_facecolor("#fff")
ax = fig.add_subplot()
ax.set_facecolor("#fff")
ax.grid(color="lightgrey", alpha=0.7, linewidth=1, axis="both", zorder=0)
sns.lineplot(x="date", y="num_sold", data=train, hue="product",palette=chart_colors, err_style=None, linewidth=1, ax=ax, zorder=2)
ax.yaxis.set_tick_params(color="#000", labelsize=12, pad=5, length=0)
ax.set_ylabel("Num Sold", fontsize=15, fontfamily="serif", labelpad=10)
ax.set_xlabel("Date", fontsize=15, fontfamily="serif", labelpad=10)
ax.xaxis.set_tick_params(color="#000", labelsize=12, pad=5, length=0)
ax.yaxis.set_tick_params(color="#000", labelsize=12, pad=5, length=0)
ax.set_title("Products vs Number of sales", loc="left", color="#000", fontsize=25, pad=5, fontweight="bold", fontfamily="serif", y=1.05, zorder=3)
hide_spines(ax)
ax.legend(loc="upper right", ncol=3, fontsize=15, edgecolor=None, facecolor=None, markerscale=2, labelcolor="#000", handlelength=1, title=None)
fig.show()


# 📌 Monthly trend in the below graph shows the seasonal variations in sales across products

# In[11]:


# code copied from https://www.kaggle.com/subinium/tps-jan-happy-new-year?scriptVersionId=84186421&cellId=22

fig, ax = plt.subplots(1, 1, figsize=(25, 7))
train_monthly = train.set_index('date').groupby([pd.Grouper(freq='M')])[['num_sold']].mean()

sns.lineplot(x="date", y="num_sold", data=train, ax=ax, label='daily',color= '#2a9d8f')
sns.lineplot(x="date", y="num_sold", data=train_monthly, ax=ax, label='monthly mean', color='black')
ax.set_title('Monthly Trend', fontsize=20, fontweight='bold', loc='left', y=1.03)
ax.grid(alpha=0.5)
hide_spines(ax)
ax.legend()
plt.show()


# # **<span style="color:#e76f51;">Feature Engineering</span>**
# 
# New features can be created from the date column like month , year , weekend or weekday .

# In[12]:


def create_features(df):
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['weekday'] = df['date'].dt.weekday   
    df['weekend'] = (df['date'].dt.weekday>=5).astype(int)   
    df.drop(columns=['date'], inplace=True)
    
create_features(train)
create_features(val)
create_features(test)


# In[13]:


train = train.drop('row_id',axis =1)
val = val.drop('row_id',axis =1)


# # **<span style="color:#e76f51;">W & B Artifacts</span>**
# 
# An artifact as a versioned folder of data.Entire datasets can be directly stored as artifacts .
# 
# W&B Artifacts are used for dataset versioning, model versioning . They are also used for tracking dependencies and results across machine learning pipelines.Artifact references can be used to point to data in other systems like S3, GCP, or your own system.
# 
# You can learn more about W&B artifacts [here](https://docs.wandb.ai/guides/artifacts)
# 
# ![](https://drive.google.com/uc?id=1JYSaIMXuEVBheP15xxuaex-32yzxgglV)

# In[14]:


# Save train data to W&B Artifacts
train.to_csv("train_wandb.csv", index = False)
run = wandb.init(project='TPSJan2022', name='training_data', anonymous=anony,config=CONFIG) 
artifact = wandb.Artifact(name='training_data',type='dataset')
artifact.add_file("./train_wandb.csv")

wandb.log_artifact(artifact)
wandb.finish()


# The snapshot of the artifact created is below
# 
# ![](https://drive.google.com/uc?id=16biHK189-q2mhyZAhE-cAvxHb3BIAfFq)

# 
# [Source](https://www.tensorflow.org/guide/data)
# 
# # **<span style="color:#e76f51;">🎯tf.data</span>**
# 
# tf.data API is used for building efficient input pipelines which can handle large amounts of data and perform complex data transformations . tf.data API has provisions for handling different data formats .
# 
# <img src="https://storage.googleapis.com/jalammar-ml/tf.data/images/tf.data.png" />
# 
# [Image Source](https://www.kaggle.com/jalammar/intro-to-data-input-pipelines-with-tf-data)
# 
# Data source is essential for building any input pipeline and tf.data.Dataset.from_tensors() or tf.data.Dataset.from_tensor_slices can be used to construct a dataset from data in memory .The recommended format for the iput data stored in file is TFRecord which can be created using TFRecordDataset() .The different data source formats supported are numpy arrays , python generators , csv files ,image , TFRecords , csv and text files. 
# 
# <img src="https://storage.googleapis.com/jalammar-ml/tf.data/images/tf.data-read-data.png" />
# 
# [Image Source](https://www.kaggle.com/jalammar/intro-to-data-input-pipelines-with-tf-data)
# 
# Construction of tf.data input pipeline consists of three phases namely Extract , Transform and Load . The extraction involves the loading of data from different file format and converting it in to tf.data.Dataset object .
# 
# ## **<span style="color:#e76f51;">🎯tf.data.Dataset</span>**
# 
# tf.data.Dataset is an abstraction introduced by tf.data API and consists of sequence of elements where each element has one or more components . For example , in a tabular data pipeline , an element might be a single training example , with a pair of tensor components representing the input features and its label 
# 
# tf.data.Dataset can be created using two distinct ways
# 
# Constructing a dataset using data stored in memory by a data source
# 
# Constructing a dataset from one or more tf.data.Dataset objects by a data transformation
# 
# <img src="https://storage.googleapis.com/jalammar-ml/tf.data/images/tf.data-simple-pipeline.png" />
# 
# [Image Source](https://www.kaggle.com/jalammar/intro-to-data-input-pipelines-with-tf-data)
# 
# 
# Basic input data pipeline constructed using tf.data API consists of the following steps .
# 
# 📌 Reading input data
# 
# 📌 Processing multiple epochs using **Dataset.repeat()**
# 
# 📌 Randomly shuffling using **Dataset.shuffle()**
# 
# 📌 Batching dataset elements using **Dataset.batch()**
# 
# Additionally preprocessing of dataset can be done using **Dataset.map()** transformation .
# 
# 
# <img src="https://storage.googleapis.com/jalammar-ml/tf.data/images/tf.data-pipeline-1.png" />
# 
# [Image Source](https://www.kaggle.com/jalammar/intro-to-data-input-pipelines-with-tf-data)
# 
# In the first step, tf.data reads the CSV file and creates a Dataset object representing the dataset. If we're to pass this Dataset to the model, it would take one of the rows in each training iteration. It's important to note that the Dataset object does not make these transformations right away -- if the a dataset is 2 TB in size and the CPU tf.data is running on only has 32GBs of RAM available, we'd be in trouble. The Dataset object acknowledges the processing plan and the transformations required, and then applies them when needed on a batch-by-batch basis.
# 
# 
# ## **<span style="color:#e76f51;">Randomly shuffling using Dataset.shuffle()</span>**
# 
# Dataset.shuffle() transformation shuffles the order of elements in the dataset and uniformly chooses the next element from the buffer.
# 
# <img src="https://storage.googleapis.com/jalammar-ml/tf.data/images/tf.data-pipeline-2.png" />
# 
# [Image Source](https://www.kaggle.com/jalammar/intro-to-data-input-pipelines-with-tf-data)
# 
# ## **<span style="color:#e76f51;">Repeating for several epochs</span>**
# 
# Now, models are trained over multiple epochs -- with the training dataset being fed to the model in each epoch. So let's tell tf.data that we want to use the Dataset for two epochs. That's done using the repeat() method:
# 
# <img src="https://storage.googleapis.com/jalammar-ml/tf.data/images/tf.data-pipeline-3.png" />
# 
# [Image Source](https://www.kaggle.com/jalammar/intro-to-data-input-pipelines-with-tf-data)
# 
# You can see that we now have double the number of rows -- the first half would be epoch #1 and the second half is epoch number #2.
# 
# ## **<span style="color:#e76f51;">Creating batches using Dataset.batch()</span>**
# 
# 
# The dataset can be broken down in to stacks or batches of consecutive elements using Dataset.batch() API
# 
# <img src="https://storage.googleapis.com/jalammar-ml/tf.data/images/tf.data-pipeline-4.png" />
# 
# [Image Source](https://www.kaggle.com/jalammar/intro-to-data-input-pipelines-with-tf-data)
# 
# 

# In[15]:


def df_to_dataset(dataframe, shuffle=True, batch_size=32 , train = 1):
  df = dataframe.copy()
  labels = df.pop('num_sold')
  df = {key: value[:,tf.newaxis] for key, value in dataframe.items()}
  ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  ds = ds.prefetch(batch_size)
  return ds


# In[16]:


batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size )
val_ds = df_to_dataset(val, batch_size=batch_size )
test_ds = tf.data.Dataset.from_tensor_slices(dict(test))


# ## **<span style="color:#e76f51;">tf.data.Dataset.take()</span>**

# In[17]:


[(train_features, label_batch)] = train_ds.take(1)
print('Every feature:', list(train_features.keys()))
for feature in list(train_features.keys()):
    print(train_features[feature])
print('A batch of targets:', label_batch )


# ## **<span style="color:#e76f51;">Feature representation using Keras Preprocessing Layers</span>**
# 
# Feature representations can be one of the crucial aspect in model developement workflows . It is a experimental process and there is no perfect solution . Keras preprocessing Layers helps us create more flexible preprocessing pipeline where new data transformations can be applied while changing the model architecture .
# 
# ![](https://drive.google.com/uc?id=1248y8JYTwjnxZnIEaTQHr1xV5jUZotLm)
# 
# [ImageSource](https://blog.tensorflow.org/2021/11/an-introduction-to-keras-preprocessing.html)
# 
# ## **<span style="color:#e76f51;">Keras Preprocessing Layers - Numerical Features</span>**
# 
# The Keras preprocessing layers available for numerical features are below 
# 
# `tf.keras.layers.Normalization`: performs feature-wise normalization of input features.
#   
# `tf.keras.layers.Discretization`: turns continuous numerical features into integer categorical features.
# 
# `adapt():`
# 
# Adapt is an optional utility function which helps in setting the internal state of layers from input data . adapt() is available on all stateful processing layerrs and it computes mean and variance for the layerrs and stores them as layers weights . adapt() is called before fit() , evaluate or predict()
# 
# 
# In this example , we are going to use tf.keras.layers.Normalization for normalizing numeric input features like month , year , weekday and weekend . This normalization layer shifts and scales inputs to a distribution  centered around 0 with standard deviation 1 by precomputing the mean and variance of the data, and calling (input - mean) / sqrt(var) at runtime.
# 

# In[18]:


def get_normalization_layer(name, dataset):
  # Create a Normalization layer for the feature.
  normalizer = tf.keras.layers.Normalization(axis=None)

  # Prepare a Dataset that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the statistics of the data.
  normalizer.adapt(feature_ds)

  return normalizer


# In[19]:


all_inputs = []
encoded_features = []

# Numerical features.
for header in ['month', 'year','weekday','weekend']:
  numeric_col = tf.keras.Input(shape=(1,), name=header)
  normalization_layer = get_normalization_layer(header, train_ds)
  encoded_numeric_col = normalization_layer(numeric_col)
  all_inputs.append(numeric_col)
  encoded_features.append(encoded_numeric_col)


# ## **<span style="color:#e76f51;">Keras Preprocessing Layers - Categorical Features</span>**
# 
# The various keras preprocessing layers available for categorical variables are below .
# 
# `tf.keras.layers.CategoryEncoding:` turns integer categorical features into one-hot, multi-hot, or count dense representations.
# 
# `tf.keras.layers.Hashing:` performs categorical feature hashing, also known as the "hashing trick".
# 
# `tf.keras.layers.StringLookup:` turns string categorical values an encoded representation that can be read by an Embedding layer or Dense layer.
# 
# `tf.keras.layers.IntegerLookup:` turns integer categorical values into an encoded representation that can be read by an Embedding layer or Dense layer.

# In[20]:


def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
  # Create a layer that turns strings into integer indices.
  if dtype == 'string':
    index = tf.keras.layers.StringLookup(max_tokens=max_tokens)
  # Otherwise, create a layer that turns integer values into integer indices.
  else:
    index = tf.keras.layers.IntegerLookup(max_tokens=max_tokens)

  # Prepare a `tf.data.Dataset` that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the set of possible values and assign them a fixed integer index.
  index.adapt(feature_ds)

  # Encode the integer indices.
  encoder = tf.keras.layers.CategoryEncoding(num_tokens=index.vocabulary_size())

  # Apply multi-hot encoding to the indices. The lambda function captures the
  # layer, so you can use them, or include them in the Keras Functional model later.
  return lambda feature: encoder(index(feature))


# In[21]:


categorical_cols = ['country','store','product']
for feature in categorical_cols:
  categorical_col = tf.keras.Input(shape=(1,), name=feature, dtype='string')
  encoding_layer = get_category_encoding_layer(name=feature,
                                               dataset=train_ds,
                                               dtype='string',
                                               max_tokens=5)
  encoded_categorical_col = encoding_layer(categorical_col)
  all_inputs.append(categorical_col)
  encoded_features.append(encoded_categorical_col)


# Prebuilt layers can be mixed and matched with custom layers and other tensorflow functions. Preprocessing can be split from training and applied efficiently with tf.data, and joined later for inference.

# ![](https://drive.google.com/uc?id=1rDqk8wCX9zJXOvyqSi6e5exmz_x03Ji4)
# 
# Models can be created using one of the following API
# 
# `Keras Sequential API`
# 
# `Keras Functional API`
# 
# `Model Subclassing`
# 
# In this tutorial lets explore the usage of Keras Functional API
# 
# 
# # **<span style="color:#e76f51;">Keras Functional API</span>**
# 
# The Keras Functional API gives users more flexibility in model creation by allowing shared layers , non -linear topology and multiple input and output layers. The functional API can be used to be build a graph of layers.
# 
# 

# In[22]:


all_features = tf.keras.layers.concatenate(encoded_features)
x = tf.keras.layers.Dense(32, activation="relu")(all_features)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model( all_inputs, output)


# # **<span style="color:#e76f51;">Custom Loss Functions</span>**
# 
# Custom Loss can be implemented by using functions and the function name has to be passed as value to the loss parameter in the compile() method .

# In[23]:


def smape(y_true, y_pred):
   y_true = tf.cast(y_true, tf.float32)
   y_pred = tf.cast(y_pred, tf.float32)
   num = tf.math.abs(tf.math.subtract(y_true, y_pred))
   denom = tf.math.add(tf.math.abs(y_true), tf.math.abs(y_pred))
   denom = tf.math.divide(denom,200.0)

   val = tf.math.divide(num,denom)
   val = tf.where(denom == 0.0, 0.0, val)
        
   return tf.reduce_mean(val)


# In[24]:


model.compile(optimizer='rmsprop',loss=smape)


# # **<span style="color:#e76f51;">Custom Loss Functions as Classes</span>**
# 
# Custom Loss can also be implemented using class and the class name has to be passed as value to loss parameter in compile() method

# In[25]:


class SMAPE(tf.keras.losses.Loss):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)        

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        num = tf.math.abs(tf.math.subtract(y_true, y_pred))
        denom = tf.math.add(tf.math.abs(y_true), tf.math.abs(y_pred))
        denom = tf.math.divide(denom,200.0)

        val = tf.math.divide(num,denom)
        val = tf.where(denom == 0.0, 0.0, val)
        
        return tf.reduce_mean(val)
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config}


# In[26]:


model.compile(optimizer='rmsprop',loss=SMAPE())


# `Keras.utils.plot_model` converts a Keras model to dot format and save to a file.
# 
# 

# In[27]:


# Use `rankdir='LR'` to make the graph horizontal.
tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")


# In[28]:


model.fit(train_ds, epochs=10, validation_data=val_ds)


# # **<span style="color:#e76f51;">References</span>**
# 
# https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers#categorical_columns
# 
# https://www.kaggle.com/ambrosm/tpsjan22-01-eda-which-makes-sense
# 
# https://www.kaggle.com/subinium/tps-jan-happy-new-year
# 
# https://www.kaggle.com/vad13irt/tps-jan-2022-exploratory-data-analysis

# # Work in progress 🚧
