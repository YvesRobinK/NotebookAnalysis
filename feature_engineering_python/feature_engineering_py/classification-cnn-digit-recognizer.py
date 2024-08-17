#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### Introduction
# 
# This notebook is my first introduction to deep learning. I recently read a wonderful book by Francois Schollet, "Deep Learning with Python".  This book has taught and inspired me. Thank you, Francois. In this competition, the challenge is to correctly recognize handwritten numbers for a known MNIST set. MNIST is often referred to as "Hello  World of Deep Learning".  In this kernel you will find my solution to this classification problem. Please consider voting for if it would be helpful for you.

# ### Table of contents:
# 1.Import
# 
# 1.1.Import of Required Modules
# 
# 1.2.Importing (Reading) Data
# 
# 2.Exploratory Data Analysis (EDA)
# 
# 2.1.Data Visualization 
# 
# 3.Data Cleaning
# 
# 4.Feature Engineering / Feature Selection
# 
# 5.Machine Learning Models
# 
# 6.Creating Submission File
# 
# 7.(if necessary) Define the Question of Interest/Goal

# ### 1.Import

# ### 1.1.Import of Required Modules

# In[2]:


# data analysis libraries 
import numpy as np 
import pandas as pd

# visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

# ignore warnings
import warnings
#warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten


# ### 1.2.Importing (Reading) Data

# In[3]:


df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# ### 2.Exploratory Data Analysis (EDA)

# In[4]:


from sklearn.model_selection import train_test_split # training and testing data split
X = df.drop('label', axis=1)
y = df['label']


# In[5]:


X.shape     # the shape function returns the size - this is a pair (number of rows, number of columns)


# In[6]:


y.shape


# In[7]:


X.dtypes    # the type function returns the types of all data from the Data Frame


# In[8]:


y.dtypes


# In[9]:


X.info()            # the info function provides general information about the DataFrame


# In[10]:


X.describe(include="all")


# In[11]:


y.describe(include="all")


# ### 2.1.Data Visualization

# In[12]:


X.head()


# In[13]:


y.head()


# ### 3.Data Cleaning
# I'm skipping this option here

# ### 4.Feature Engineering / Feature Selection
# I'm skipping this option here

# ### 5.Machine Learning Models

# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=22)


# #### Naive Bayes

# In[15]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
y_model = model.predict(X_test)


# In[16]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_model)


# #### Random Forest

# In[17]:


from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn import model_selection
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_model = model.predict(X_test)


# In[18]:


accuracy_score(y_test, y_model)


# In[19]:


# cross validation
scores = model_selection.cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(scores)
print("Kfold on RandomForestClassifier: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))
#random_forest.fit(X, y)
#random_forest.score(X, y)


# It turned out that a simple random forest, not configured in any special way, gives a very accurate classification of data by handwritten digits.

# ### Neural Networks

# ### Fully connected neural network

# In[20]:


from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras import  backend as K
from keras.preprocessing.image import ImageDataGenerator


# In[21]:


X_train = (df.iloc[:,1:].values).astype('float32') # all pixel values
y_train = df.iloc[:,0].values.astype('int32') # only labels i.e targets digits
X_test = test_df.values.astype('float32')
y_test = test_df.iloc[:,0].values.astype('int32') # only labels i.e targets digits


# In[22]:


X_train


# In[23]:


y_train


# In[24]:


X_test


# #### Data Visualization

# In[25]:


X_train = X_train.reshape(X_train.shape[0], 28, 28)

for i in range(3, 9):
    plt.subplot(330 + (i+1))
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    plt.title(y_train[i]);


# In[26]:


#expand 1 more dimention as 1 for colour channel gray
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_train.shape


# In[27]:


y_train.shape


# In[28]:


X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
X_test.shape


# In[29]:


y_test.shape


# In[30]:


# Normalization of input data
X_train = X_train/255
X_test = X_test/255


# In[31]:


# Converting output values into vectors by category
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)


# In[32]:


# We can look at our data
# For example, the 10th index (this is the 11th element in order) from the training set:
X_train[10] 


# In[33]:


y_train[10]


# In[34]:


X_train.shape


# In[35]:


y_train.shape


# In[36]:


# Displaying the first 25 images from the training sample
plt.figure(figsize=(10,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_train[i], cmap = plt.cm.binary)

plt.show()


# Designing Architecture of Neural  Network 

# In[37]:


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# ### Fully connected neural network
# 
# •	A fully connected neural network consists of a series of fully connected layers that connect every neuron in one layer to every neuron in the other layer.
# 
# •	The major advantage of fully connected networks is that they are “structure agnostic” i.e. there are no special assumptions needed to be made about the input.
# 
# •	While being structure agnostic makes fully connected networks very broadly applicable, such networks do tend to have weaker performance than special-purpose networks tuned to the structure of a problem space.

# The developer chooses the structure of the neural network based on the problem being solved.
# CNN convolutional neural networks have performed well for recognizing graphic images (next I will show how CNN works).
# But first, I'll show you how an ordinary fully connected neural network solves this problem:

# In[38]:


model = keras.Sequential([
    Flatten(input_shape=(28,28,1)),  # 1st layer: 28x28 pixels are fed to the input of this layer, 1 byte is 1 pixel in grayscale from 0 to 255
    Dense(128, activation='relu'),   # hidden layer: of 128 neurons, activation function ’Relu’
    Dense(10, activation='softmax')  # output layer: of 10 neurons, the activation function is ’softmax’ (because we want to interpret the output values in terms of probability)
])
print(model.summary())               # output the neural network structure in the console


# Neural network compilation with Adam optimization and criteria – categorical cross-entropy

# In[39]:


model.compile(optimizer='adam',              # optimization by Adam
            loss='categorical_crossentropy', # loss function ’categorical_crossentropy’ (often taken when solving a classification problem
            metrics=['accuracy'])            # the ‘accuracy’ metric


# In[40]:


model.fit(X_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)
# batch_size=32 means that after every 32 images we will adjust the weights
# epochs=5 – our model will be trained by going through the entire dataset 5 times
# validation_split=0.2 – we split the traning set into the actual training sample (0.8) and validation sample (0.2)


# We evaluate our model.
# evaluate() calculates the loss value and the values of all the metrics that we selected when compiling the model.

# In[41]:


model.evaluate(X_test, y_test_cat)


# Checking digit recognition

# In[42]:


n = 2
X = np.expand_dims(X_test[n], axis=0)
res = model.predict(X)
print(res)
print(f'Recognized digit: {np.argmax(res)}')

plt.imshow(X_test[n], cmap = plt.cm.binary)
plt.show()


# Recognition of the entire test set

# In[43]:


pred = model.predict(X_test)
pred = np.argmax(pred, axis=1)

print(pred.shape)

print(pred[:20])


# ### Convolutional neural network CNN
# 
# 
# •	CNN architectures make the explicit assumption that the inputs are images, which allows encoding certain properties into the model architecture.
# 
# •	A simple CNN is a sequence of layers, and every layer of a CNN transforms one volume of activations to another through a differentiable function. Three main types of layers are used to build CNN architecture: Convolutional Layer, Pooling Layer, and Fully-Connected Layer.

# In[44]:


print(X_train.shape)


# In[45]:


print(y_train.shape)


# Each convolutional layer in the 2-dimensional case is implemented using the following class: keras.layers.Conv2D(filters, kernel_size, strides=(1,1), ...) This class has the following basic parameters: filters – number of cores (channels) kernel_size – the size of the kernel (in the form of a tuple of two numbers) strides=(1,1) – the step of scanning filters along the axes of the plane (by default: one pixel)

# In[46]:


model = keras.Sequential([
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2), strides=2),  # after this layer, the feature map became (14,14)
    Conv2D(64, (3,3), padding='same', activation='relu'),
    MaxPooling2D((2,2), strides=2),  # after this layer, the feature map became (7,7), that is, at the output we have a 7 x 64 tensor
    Flatten(),                       # this layer pulls this tensor into a vector
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # this output layer will define the digit (0,1,...,9)
])


# In[47]:


print(model.summary())  # output of the neural network structure to the console


# Neural network compilation 

# In[48]:


model.compile(optimizer='adam',      # optimization by Adam
              loss='categorical_crossentropy', # loss function 'categorical_crossentropy' (often taken when solving a classification problem
              metrics=['accuracy']) # the 'accuracy' metric


# Training a neural network

# In[49]:


his = model.fit(X_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)


# We evaluate our model.
# evaluate() calculates the loss value and the values of all the metrics that we selected when compiling the model.

# In[50]:


model.evaluate(X_test, y_test_cat)


# Recognition of the entire test set

# In[51]:


pred = model.predict(X_test)
pred = np.argmax(pred, axis=1)

print(pred.shape)

print(pred[:20])


# ### 7.Creating Submission File

# In[52]:


submission=pd.DataFrame({"ImageId": list(range(1,len(pred)+1)),
                         "Label": pred})


# In[53]:


submission.to_csv("Digit_RecognizerSubmission.csv", index=False, header=True)
submission 


# ### 8.(if necessary) Define the Question of Interest/Goal
# I did not perform this item in this notebook.

# I would be grateful for any feedback!
# 
# I would appreciate it if you mark my notebook with your upvote! 
# 
# You can also refer to my other notebooks if you are interested: 
# 
# https://www.kaggle.com/code/igorprikhodko/stock-market-prediction-using-sarimax
# 
# https://www.kaggle.com/code/igorprikhodko/rain-prediction-for-tomorrow-in-australia
# 
# https://www.kaggle.com/code/igorprikhodko/diamonds-price-prediction
# 
# Thank you for your attention to my work!
