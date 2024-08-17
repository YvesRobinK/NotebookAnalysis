#!/usr/bin/env python
# coding: utf-8

# <div style="width:700px; margin:0 auto;">
# <img src="https://media4.giphy.com/media/MQuGeeTJXx52ZoyZrP/giphy.gif?cid=790b7611050e8d3a164f9e1fbb60da479cef3bfc72967a6c&rid=giphy.gif&ct=g" width="480px"/>
# </div>

# ## <span style="color:#011936;">This notebook shows how to classify structured data, such as tabular data, using Spaceship Titanic dataset.<br>We will use Keras to define our model, and Keras preprocessing layers as a bridge to map from columns in a CSV file to features used to train the model.<br> The goal is to predict Whether a passenger was transported to another dimension.<span>

# ### This tutorial contains complete code for:
# * [1. Import Libraries.](#chapter1)
# * [2. Load the dataset and read it into a pandas DataFrame](#chapter2)
# * [3. Exploratory Data Analysis](#chapter3)
# * [4. Data Preprocessing](#chapter4)
# * [5. Split tha data](#chapter5)
# * [6. Create an input pipeline using tf.data](#chapter6) 
# * [7. Feature preprocessing with Keras layers](#chapter7)
#     * [7.1. Numerical features](#section71) 
#     * [7.2. categorical features](#section72)
#     * [7.3. Preprocess selected features](#section73)
# * [8. Create, compile, and train the model](#chapter8)
# * [9. Inference](#chapter9)
#     * [9.1 Perform Inference on a random samples](#section91) 
#     * [9.2. categorical features](#section92)

# ## 1. Import Libraries <a class="anchor" id="chapter1"></a>

# In[1]:


import numpy as np
import pandas as pd 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras import layers

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


tf.__version__


# ## 2. Load the dataset and read it into a pandas DataFrame <a class="anchor" id="chapter2"></a>

# #### Pandas is a Python library with many helpful utilities for loading and working with structured data

# In[3]:


train = pd.read_csv("../input/spaceship-titanic/train.csv")
test = pd.read_csv("../input/spaceship-titanic/test.csv")
submission = pd.read_csv("../input/spaceship-titanic/sample_submission.csv")


# ## 3. Exploratory Data Analysis <a class="anchor" id="chapter3"></a>

# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


submission.head()


# In[7]:


train.info()


# In[8]:


test.info()


# #### Let's analyze our features using [pandas.DataFrame.describe API](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html) 

# In[9]:


# Looking at NaN % within the train data
nan = pd.DataFrame(train.isna().sum(), columns = ['NaN_sum'])
nan['Percentage(%)'] = (nan['NaN_sum']/len(train))*100
nan = nan[nan['NaN_sum'] > 0]
nan = nan.sort_values(by = ['NaN_sum'])
nan


# In[10]:


# Plotting Nan

plt.figure(figsize = (15,5))
sns.barplot(x = nan.index, y = nan['Percentage(%)'])
plt.xticks(rotation=45)
plt.title('Features containing Nan')
plt.xlabel('Features')
plt.ylabel('% of Missing Data')
plt.show()


# In[11]:


# features with float64 type
train.describe(include=["float64"]).T


# In[12]:


#features with object and bool types
train.describe(include=[object,bool]).T


# #### Les's plot categorical features

# In[13]:


def plot_pie_chart(dataframe,col):
    _, ax = plt.subplots(figsize=[18,6])
    dataframe.groupby([col]).size().plot(kind='pie',autopct='%.2f%%',ax=ax, title='',label=col) 


# In[14]:


plot_pie_chart(train,col="HomePlanet")


# In[15]:


plot_pie_chart(train,col="CryoSleep")


# In[16]:


plot_pie_chart(train,col="Destination")


# In[17]:


plot_pie_chart(train,col="VIP")


# In[18]:


train['Cabin'].value_counts()


# In[19]:


# Target Distribution
plot_pie_chart(train,col="Transported")


# ## 4.Data Preprocessing <a class="anchor" id="chapter4"></a>

# In[20]:


def bool_to_str(dataframe,columns):
    mask = dataframe[columns].applymap(type) != bool
    d = {True: 'TRUE', False: 'FALSE'}
    dataframe[columns] = dataframe[columns].where(mask, dataframe[columns].replace(d))
    return dataframe

def impute_data(train,test,columns,method):
    for col in columns:
        if method == 'mean':
            value = train[col].mean()
            train[col].fillna(value,inplace=True)
            test[col].fillna(value,inplace=True)
        elif method == 'mode':
            value = train[col].mode()[0]
            train[col].fillna(value,inplace=True)
            test[col].fillna(value,inplace=True)
        elif method == 'median':
            value = train[col].median()
            train[col].fillna(value,inplace=True)
            test[col].fillna(value,inplace=True)
    return train,test


# #### We will delete "PassengerId", "Cabin" and "Name" from our dataset

# In[21]:


del train['PassengerId'], train['Cabin'], train["Name"], test['PassengerId'], test['Cabin'], test["Name"]


# #### Now, we will impute null values in our data using the impute_data function defined earlier
# #### We will impute numerical features with mean and categorical features with mode.

# 
#  <img src="https://d1e4pidl3fu268.cloudfront.net/607026d5-c501-4a3d-910a-a990d735ec37/MeanmedianmodeandrangepostersPage3.PNG" alt="Girl in a jacket" width="380" height="500"> 

# In[22]:


# impute categorical features
train, test = impute_data(train,test,columns=['HomePlanet','CryoSleep','Destination','VIP'],method = "mode")
# impute numerical features
train, test = impute_data(train,test,columns=['Age', 'RoomService','FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'],method="mean")


# #### We will transform 'CryoSleep' and 'VIP' from boolean type to string.

# In[23]:


train, test = bool_to_str(train,['CryoSleep','VIP']), bool_to_str(test,['CryoSleep','VIP'])


# #### We will encode the target column from boolean to numerical type

# In[24]:


le = LabelEncoder()
train["Transported"] = le.fit_transform(train["Transported"])


# In[25]:


train.head()


# ## 5. Split tha data <a class="anchor" id="chapter5"></a>

# #### Split tha data and use 80% for training and 20% for validation

# In[26]:


X_train, X_val = train_test_split(train, test_size=0.2, random_state=42, stratify=train["Transported"])


# In[27]:


print(len(X_train), 'training examples')
print(len(X_val), 'validation examples')
print(len(test), 'test examples')


# ## 6. Create an input pipeline using tf.data  <a class="anchor" id="chapter6"></a>

# #### Next, create a utility function that converts each training, validation, and test set DataFrame into a [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset), then shuffles and batches the data

# In[28]:


def df_to_dataset(dataframe, shuffle=True, batch_size=32, inference=False):
  df = dataframe.copy()
  if inference == False: 
      labels = df.pop('Transported')
  df = {key: value[:,tf.newaxis] for key, value in dataframe.items()}
  if inference == False:
      ds = tf.data.Dataset.from_tensor_slices((df, labels))
  else:
    ds = tf.data.Dataset.from_tensor_slices((df))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  ds = ds.prefetch(batch_size)
  return ds


# #### Now, use the newly created function (df_to_dataset) to check the format of the data the input pipeline helper function returns by calling it on the training data, and use a small batch size to keep the output readable:

# In[29]:


batch_size = 5
train_ds = df_to_dataset(X_train, batch_size=batch_size)


# * Each Dataset yields a tuple (input, target) where input is a dictionary of features and target is the value 0 or 1:

# In[30]:


[(train_features, label_batch)] = train_ds.take(1)
print('Every feature:', list(train_features.keys()))
print('A batch of ages:', train_features['Age'])
print('A batch of targets:', label_batch )


# #### As the output demonstrates, the training set returns a dictionary of column names (from the DataFrame) that map to column values from rows.

# ## 7. Feature preprocessing with Keras layers <a class="anchor" id="chapter7"></a>

# #### The Keras preprocessing layers allow you to build Keras-native input processing pipelines, which can be used as independent preprocessing code in non-Keras workflows, combined directly with Keras models, and exported as part of a Keras SavedModel.
# 
# #### In this kernel, we will use the following two preprocessing layers to demonstrate how to perform preprocessing, structured data encoding, and feature engineering:
# *  tf.keras.layers.Normalization: Performs feature-wise normalization of input features.
# *  tf.keras.layers.StringLookup: Turns string categorical values into integer indices.
# 
# #### You can learn more about the available layers in the Working with [preprocessing layers](https://www.tensorflow.org/guide/keras/preprocessing_layers) guide.
# #### For numerical features, such as 'Age', 'RoomService','FoodCourt', 'ShoppingMall', 'Spa' and 'VRDeck'  we will use a tf.keras.layers.Normalization layer to standardize the distribution of the data.
# #### For categorical features, such as 'HomePlanet','CryoSleep','Destination' and 'VIP', We will transform them to multi-hot encoded tensors with tf.keras.layers.CategoryEncoding.
# 

# ### 7.1. Numerical features <a class="anchor" id="section71"></a>

# #### For each numeric feature we will use a [tf.keras.layers.Normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Normalization) layer to standardize the distribution of the data.
# 
# #### Let's define a new function that returns a layer which applies feature-wise normalization to numerical features using that Keras preprocessing layer:

# In[31]:


def get_normalization_layer(name, dataset):
  # Create a Normalization layer for the feature.
  normalizer = layers.Normalization(axis=None)

  # Prepare a Dataset that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])
  #feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

  # Learn the statistics of the data.
  normalizer.adapt(feature_ds)

  return normalizer


# #### Let's test the new function by calling it on the "Age" and "Spa" features to normalize them:

# In[32]:


photo_count_col = train_features['Age']
layer = get_normalization_layer('Age', train_ds)
layer(photo_count_col)


# In[33]:


photo_count_col = train_features['Spa']
layer = get_normalization_layer('Spa', train_ds)
layer(photo_count_col)


# ### 7.2. categorical features <a class="anchor" id="section72"></a>

# #### Let's define another function that maps values from a vocabulary to integer indices and multi-hot encodes the features

# In[34]:


def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
  # Create a layer that turns strings into integer indices.
  if dtype == 'string' or dtype == 'object':
    index = layers.StringLookup(max_tokens=max_tokens)
  # Otherwise, create a layer that turns integer values into integer indices.
  else:
    index = layers.IntegerLookup(max_tokens=max_tokens)

  # Prepare a `tf.data.Dataset` that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the set of possible values and assign them a fixed integer index.
  index.adapt(feature_ds)

  # Encode the integer indices.
  encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

  # Apply one-hot encoding to the indices. The lambda function captures the
  # layer, so you can use them, or include them in the Keras Functional model later.
  return lambda feature: encoder(index(feature))


# #### Let's test the new function by calling it on the "Destination" and "VIP" features to normalize them:

# In[35]:


test_type_col = train_features['Destination']
test_type_layer = get_category_encoding_layer(name='Destination',
                                              dataset=train_ds,
                                              dtype='object')
test_type_layer(test_type_col)


# In[36]:


test_type_col = train_features['VIP']
test_type_layer = get_category_encoding_layer(name='VIP',
                                              dataset=train_ds,
                                              dtype='object', max_tokens=2)
test_type_layer(test_type_col)


# ### 7.3. Preprocess selected features <a class="anchor" id="section73"></a>

# #### Let's now create a new input pipeline with a larger batch size: 

# In[37]:


batch_size = 32
train_ds = df_to_dataset(X_train, batch_size=batch_size)
val_ds = df_to_dataset(X_val, shuffle=False, batch_size=batch_size)


# #### Now we will :
# ####  1. Normalize the numerical features, and add them to one list of inputs called encoded_features.
# ####  2. Then, encode categorical features and add them to one list of inputs called encoded_categorical_col

# In[38]:


all_inputs = []
encoded_features = []

# Numerical features.
numerical_cols = ['Age', 'RoomService','FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for header in numerical_cols:
  numeric_col = tf.keras.Input(shape=(1,), name=header)
  normalization_layer = get_normalization_layer(header, train_ds)
  encoded_numeric_col = normalization_layer(numeric_col)
  all_inputs.append(numeric_col)
  encoded_features.append(encoded_numeric_col)


# categorical features.
categorical_cols = ['HomePlanet','CryoSleep','Destination','VIP']
for header in categorical_cols:
  categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
  encoding_layer = get_category_encoding_layer(name=header,
                                               dataset=train_ds,
                                               dtype='string',
                                               max_tokens=5)
  encoded_categorical_col = encoding_layer(categorical_col)
  all_inputs.append(categorical_col)
  encoded_features.append(encoded_categorical_col)


# ## 8. Create, compile, and train the model <a class="anchor" id="chapter8"></a>

# #### Now, we will create our deep learning model using [keras API](https://www.tensorflow.org/guide/keras/functional):
# #### For the first layer in your model, merge the list of feature inputs—encoded_features—into one vector via concatenation with [tf.keras.layers.concatenate](https://www.tensorflow.org/api_docs/python/tf/keras/layers/concatenate)

# In[39]:


all_features = tf.keras.layers.concatenate(encoded_features)
x = tf.keras.layers.Dense(64, activation="relu")(all_features)
x = tf.keras.layers.Dense(32, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(all_inputs, output)


# #### Before you do any training, you need to decide on three things:
# 
# #### 1. <b>An optimizer</b>. The job of the optimizer is to decide how much to change each parameter in the model, given the current model prediction. When using the Layers API, you can provide either a string identifier of an existing optimizer (such as 'sgd' or 'adam'), or an instance of the [Optimizer](https://js.tensorflow.org/api/latest/#Training-Optimizers) class.
# #### 2. <b>A loss function</b>. An objective that the model will try to minimize. Its goal is to give a single number for "how wrong" the model's prediction was. The loss is computed on every batch of data so that the model can update its weights. When using the Layers API, you can provide either a string identifier of an existing loss function (such as 'BinaryCrossentropy'), or any function that takes a predicted and a true value and returns a loss. See a [list of available losses](https://js.tensorflow.org/api/latest/#Training-Losses) in the API docs.
# #### 3. <b>List of metrics</b>. Similar to losses, metrics compute a single number, summarizing how well our model is doing. The metrics are usually computed on the whole data at the end of each epoch. At the very least, we want to monitor that our loss is going down over time. However, we often want a more human-friendly metric such as accuracy. When using the Layers API, you can provide either a string identifier of an existing metric (such as 'accuracy'), or any function that takes a predicted and a true value and returns a score. See a list of [available metrics](https://js.tensorflow.org/api/latest/#Training-Losses) in the API docs.
# 
# #### When you've decided, compile a LayersModel by calling model.compile() with the provided options:

# #### Now, we will configure the model with [Model.compile()](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile) with the provided options:

# In[40]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
              metrics=["accuracy"]) 


# #### Now, Let's call [model.summary()](https://www.tensorflow.org/js/guide/models_and_layers#model_summary) to print a useful summary of the model, which includes:
# 
# * Name and type of all layers in the model.
# * Output shape for each layer.
# * Number of weight parameters of each layer.
# * If the model has general topology, the inputs each layer receives
# * The total number of trainable and non-trainable parameters of the model.
# 
# #### For the model we defined above, we get the following output on the console:

# In[41]:


model.summary()


# #### Note the null values in the output shapes of the layers: a reminder that the model expects the input to have a batch size as the outermost dimension,</br> which in this case can be flexible due to the null value.

# #### Now, Let's visualize the connectivity graph:

# In[42]:


# Use `rankdir='LR'` to make the graph horizontal. 
tf.keras.utils.plot_model(model, show_shapes=False, rankdir="LR") 


# #### Let's train our model:

# In[43]:


history = model.fit(train_ds, epochs=20, validation_data=val_ds)


# #### Let's plot the accuracy on the training and validation datasets over training epochs. 

# In[44]:


def plotHistory(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model performance")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()
    return


# In[45]:


plotHistory(history)


# ## 9. Inference <a class="anchor" id="chapter9"></a>

# #### We can now save and reload the Keras model with [Model.save](https://www.tensorflow.org/api_docs/python/tf/keras/Model#save) and [Model.load_model](https://www.tensorflow.org/tutorials/keras/save_and_load) before performing inference on new data:

# In[46]:


# save model
model.save('passengers_classifier')
# load the model
our_model = tf.keras.models.load_model('passengers_classifier')


# ### 9.1 Perform Inference on a random samples <a class="anchor" id="section91"></a>

# #### To get a prediction for a new sample, we can simply call the Keras Model.predict method.<br> There are just two things you need to do:
# 
# * 1. Wrap scalars into a list so as to have a batch dimension (Models only process batches of data, not single samples).
# * 2. Call [tf.convert_to_tensor](https://www.tensorflow.org/api_docs/python/tf/convert_to_tensor) on each feature.

# #### Now I'll choose some random values and test our model on two samples

# In[47]:


sample1 = {
    'HomePlanet': 'Europa',
    'CryoSleep': 'TRUE',
    'Destination': 'TRAPPIST-1e',
    'Age': 33.0,
    'VIP': 'FALSE',
    'RoomService': 10.0,
    'FoodCourt': 17.0,
    'ShoppingMall': 44.0,
    'Spa': 3.0,
    'VRDeck': 11.0
}

input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample1.items()}
predictions = our_model.predict(input_dict)
prob = tf.nn.sigmoid(predictions[0])

print(
    "This particular passenger had a %.1f percent probability "
    "of being transported to another dimension." % (100 * prob)
)


# In[48]:


sample2 = {
    'HomePlanet': 'Earth',
    'CryoSleep': 'TRUE',
    'Destination': 'TRAPPIST-1e',
    'Age': 66.0,
    'VIP': 'FALSE',
    'RoomService': 12.0,
    'FoodCourt': 27.0,
    'ShoppingMall': 34.0,
    'Spa': 13.0,
    'VRDeck': 11.0
}

input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample2.items()}
predictions = our_model.predict(input_dict)
prob = tf.nn.sigmoid(predictions[0])

print(
    "This particular passenger had a %.1f percent probability "
    "of being transported to another dimension." % (100 * prob)
)


# ### 9.2 Make Submission <a class="anchor" id="section92"></a>

# In[49]:


test_ds = df_to_dataset(test, shuffle=False, batch_size=32, inference=True)


# In[50]:


proba = tf.nn.sigmoid(our_model.predict(test_ds))
labels = np.where(proba<0.5,0,1)
preds = le.inverse_transform(labels.reshape(-1))
submission['Transported'] = preds
submission.to_csv('submission.csv', index=False)


# In[51]:


submission.head()


# <h4>References:</h4>
# 
# * [https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers](https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers?hl=eng) <br>
# 
# 

# ## I hope that you find this kernel usefull 🏄

# <div style="width:700px; margin:0 auto;">
# <img src="https://media4.giphy.com/media/26u4lOMA8JKSnL9Uk/200.webp?cid=ecf05e47anilez5fnvfkh2qjsnyzoma4k6grh8wz6lbzzsk0&rid=200.webp&ct=g" width="450px"/>
# </div>
