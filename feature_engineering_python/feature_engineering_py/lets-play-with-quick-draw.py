#!/usr/bin/env python
# coding: utf-8

# # Lets Play With Quick Draw!!

# ![Imgur](https://i.imgur.com/l7LgSGy.png)

# # &#128225; What is Quick Draw?
# ![Imgur](https://i.imgur.com/HyPUqwF.png)
# “Quick, Draw!” was a game that was initially featured at Google I/O in 2016, as a game where one player would be prompted to draw a picture of an object, and the other player would need to guess what it was. Just like pictionary.
# In 2017, the Magenta team at Google Research took that idea a step further by using this labeled dataset to train the [Sketch-RNN](https://magenta.tensorflow.org/assets/sketch_rnn_demo/index.html) model, to try to predict what the player was drawing, in real time, instead of requiring a second player to do the guessing. The game is [available online,](https://quickdraw.withgoogle.com) and has now collected over 1 billion hand-drawn doodles!

# # &#128220; Overview Of Quick Draw DataSet
# The team has open sourced this data, and in a variety of formats. You can learn more at their [GitHub page.](https://github.com/googlecreativelab/quickdraw-dataset)Now we can get this dataset easily from [Our Data Science House Kaggle.](https://www.kaggle.com)
# 
# **Here is Orginal Google DataSet  demo picture.This picture collect from github. &#128071;** ![Imgur](https://i.imgur.com/MOziCCc.png)
# 
# **Another picture of Kaggle DataSet picture. &#128071; **![Imgur](https://i.imgur.com/36cyzNX.png)

# There are 4 formats: First up are the raw files stored in (.ndjson) format. These files encode the full set of information for each doodle. It contains timing information for each stroke of every picture drawn.
# 
# There is also a simplified version, stored in the same format (.ndjson), which has some preprocessing applied to normalize the data. The simplified version is also available as a binary format for more efficient storage and transfer. There are examples of how to read the files using both Python and NodeJS.
# 
# **This picture  Google Cloud Platfrom of Quick Draw Datasets. &#128071;** ![Imgur](https://i.imgur.com/eYoCpkA.png)
# 
# 
# The fourth format takes the simplified data and renders it into a 28x28 grayscale bitmap in numpy .npy format, which can be loaded using np.load().
# 
# Why is it 28x28? Well, it’s a perfect replacement for any existing code you might have for processing MNIST data. So if you’re looking for something fancier than 10 handwritten digits, you can try processing over 300 different classes of doodles.

# #  &#128202; Data exploration and visualization of Quick Draw Game
# If you want to explore the dataset some more, you can visualize the quickdraw dataset using Facets. The Facets team has even taken the liberty of hosting it online and giving us some presets to play around with! You can [access the page here.](https://pair-code.github.io/facets/quickdraw.html) We can load up some random chairs and see how different players drew chairs from around the world.
# 
# ![Quick Draw gift](https://i.imgur.com/q1h49cE.gif)
# 
# 

# In[ ]:


get_ipython().run_cell_magic('html', '', "<style>\n@import url('https://fonts.googleapis.com/css?family=Ewert|Roboto&effect=3d|ice|');\nbody {background-color: gainsboro;} \na {color: #37c9e1; font-family: 'Roboto';} \nh1 {color: #37c9e1; font-family: 'Orbitron'; text-shadow: 4px 4px 4px #aaa;} \nh2, h3 {color: slategray; font-family: 'Orbitron'; text-shadow: 4px 4px 4px #aaa;}\nh4 {color: #818286; font-family: 'Roboto';}\nspan {font-family:'Roboto'; color:black; text-shadow: 5px 5px 5px #aaa;}  \ndiv.output_area pre{font-family:'Roboto'; font-size:110%; color:lightblue;}      \n</style>\n")


# # 📑 Import Libraries

# In[ ]:


import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
#deep lerning libraries
import keras
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import pickle # Read/Write with Serialization
import requests # Makes HTTP requests
from io import BytesIO # Use When expecting bytes-like objects


# # &#128229; Load and Read DataSets

# In[ ]:


# Classes we will load
categories = ['cannon','eye', 'face', 'nail', 'pear','piano','radio','spider','star','sword']

# Dictionary for URL and class labels
URL_DATA = {}
for category in categories:
    URL_DATA[category] = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/' + category +'.npy'


# In[ ]:


classes_dict = {}
for key, value in URL_DATA.items():
    response = requests.get(value)
    classes_dict[key] = np.load(BytesIO(response.content))


# In[ ]:


for i, (key, value) in enumerate(classes_dict.items()):
    value = value.astype('float32')/255.
    if i == 0:
        classes_dict[key] = np.c_[value, np.zeros(len(value))]
    else:
        classes_dict[key] = np.c_[value,i*np.ones(len(value))]

# Create a dict with label codes
label_dict = {0:'cannon',1:'eye', 2:'face', 3:'nail', 4:'pear', 
              5:'piana',6:'radio', 7:'spider', 8:'star', 9:'sword'}


# In[ ]:


lst = []
for key, value in classes_dict.items():
    lst.append(value[:3000])
doodles = np.concatenate(lst)


# In[ ]:


# Split the data into features and class labels (X & y respectively)
y = doodles[:,-1].astype('float32')
X = doodles[:,:784]

# Split each dataset into train/test splits
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)


# In[ ]:


# Save X_train dataset as a pickle file
with open('xtrain_doodle.pickle', 'wb') as f:
    pickle.dump(X_train, f)
    
# Save X_test dataset as a pickle file
with open('xtest_doodle.pickle', 'wb') as f:
    pickle.dump(X_test, f)
    
# Save y_train dataset as a pickle file
with open('ytrain_doodle.pickle', 'wb') as f:
    pickle.dump(y_train, f)
    
# Save y_test dataset as a pickle file
with open('ytest_doodle.pickle', 'wb') as f:
    pickle.dump(y_test, f)


# # &#128187; Predictive Modeling 

# # &#128204; 1. K-Nearest Neighbors
# ![Imgur](https://i.imgur.com/HyPUqwF.png)
# ![Imgur](https://i.imgur.com/IO58pDy.jpg)

# ### &#128210;  Note:
# By simply randomly guessing, one should be able to reach ~10% accuracy (since there are only ten class labels). A machine learning algorithm will need to obtain > 10% accuracy in order to demonstrate that it has in fact “learned” something (or found an underlying pattern in the data).
# 
# To start, we’ll model the data with the k-Nearest Neighbor (k-NN) classifier, arguably the most simple, easy to understand machine learning algorithm. The k-NN algorithm classifies unknown data points by finding the most common class among the k-closest examples. Each data point in the k closest examples casts a vote and the category with the most votes is chosen.

# ## &#128295;  Base model Of KNN
# 
# Next, I will try out a KNN classifier:

# ### Output:
# 
# ```Accuracy:80.5 ```

# ## &#128290; Tuning number of neighbors
# The KNN classifier looks promising, let's test different values of K:

# ![Imgur](https://i.imgur.com/vwadkbv.png)

# ## &#128201; Plot results of grid search
# 

# ### Output:
# ![Imgur](https://i.imgur.com/DBn5Cyj.png)

# ### &#128210;  Note:
# From examining our plot and using the elbow-method using 3 neighbors seems like the best choice to avoid overfitting. The main advantage of the KNN algorithm is that it performs well with multi-modal classes because the basis of its decision is based on a small neighborhood of similar objects. This is why its results were fairly high with 80%. The main disadvantage is the computational cost are very high and the results take far too long.**

# # &#127966; 2. Random Forest

# ![Imgur](https://i.imgur.com/HyPUqwF.png)
# ![Imgur](https://i.imgur.com/lREy3CV.jpg)
# ![Imgur](https://i.imgur.com/lEuwiKK.jpg)

# ### &#128210; Note:
# Random forests is an ensemble model which means that it uses the results from many different models to calculate a label

# ## &#128295; Base RFC model
# 

# ### Output:
# ```
# accuracy:74.4
# ```

# We then tuned the max_features parameters, which are the maximum number of
# features Random Forest can try in individual tree. By limiting the max features to
# the square root of total features we improved the model and made it computationally
# less expensive.
# We then plotted the pixel importances and saw that the edges of the doodles tend to
# be the most important.

# ## &#128295; Tuning number of estimators in the ensemble method
# 

# ### Output:
# ![Imgur](https://i.imgur.com/Ds6OVNz.png)

# ## &#128201; Plot results of grid search
# 

# ### Output:
# ![Imgur](https://i.imgur.com/wKOndoI.png)

# ## &#128295; Tuning max features

# ### Output:
# ![Imgur](https://i.imgur.com/IbOqyhQ.png)

# ### Output:
# ![Imgur](https://i.imgur.com/MhH5Wnz.png)

# ##  Modeling RFC with best hyper-parameters

# ### Output:
# ```
# accuracy:80.5
# ```

# ### Output:
# **Seeing what pixels are the most important in deciding the label
# **![Imgur](https://i.imgur.com/aJMgZJi.png)

# ### &#128210; Note:
# The random forest had an accuracy score very close to the k-nn model. The features that are most important are on the edge and in the middle of each side.

# # &#128208; 3.Support Vector Machine
# ![Imgur](https://i.imgur.com/HyPUqwF.png)
# ![Imgur](https://i.imgur.com/JHwkkBg.jpg)
# ![Imgur](https://i.imgur.com/mILrxTO.png)
# 

# ### &#128210; Note:
# SVM classification uses planes in space to divide data points. We can compared a linear divider a non-linear divider.

# ## 3.1.LinearSVC

# ### Output:
# ```
# Linear SVC accuracy:71.7
# ```

# ## 3.2. Non-Linear SVM (Radial Basis Function)

# ### Output:
# ```
# Gaussian Radial Basis Function SVC Accuracy:  77.15
# ```

# # &#128218;  4. Multi-Layer Perceptron
# 
# ![Imgur](https://i.imgur.com/HyPUqwF.png)
# 
# ![Imgur](https://i.imgur.com/1wtN0Ln.png)
# 
# ![Imgur](https://i.imgur.com/1S9bC75.jpg)

# ### &#128210; Note
# 
# A perceptron is a neural network with a very basic architecture. This will be good to compare against the convolutional neural network. Neural Networks receive an input (a single vector), and transform it through a series of hidden layers. Each hidden layer is made up of a set of neurons, where each neuron is fully connected to all neurons in the previous layer, and where neurons in a single layer function completely independently and do not share any connections. The last fully-connected layer is called the “output layer” and in classification settings it represents the class scores.
# 
# For tuning the hyper-parameters for a Multi-Layer Perceptron, we can try different number of layers and number of neurons in each layer. The default activiation function (ReLu) is the most effective choice from sklearn and the the defualt optimization algorithm (adam) is the best choice due to the size of our dataset.

# ### Output:
# ```
# mlp accuracy:  81.9
# ```

# ### Output:
# **Mean Test Score of MLP:
# **![Imgur](https://i.imgur.com/lINgGl4.png)

# ### Output:
# ```
# mlp accuracy:  84.6
# ```

# # &#128225; 5.Convolutional Neural Networks (CNN / ConvNet)
# The Convolutional Neural Network architectures make the explicit assumption that the inputs are images, which allows us to encode certain properties into the architecture. These then make the forward function more efficient to implement and vastly reduce the amount of parameters in the network.
# 
# ![Imgur](https://i.imgur.com/HyPUqwF.png)
# ![Imgur](https://i.imgur.com/jR5naeg.png)
# 
# ### How Work Convolutional Neural Networks (CNN)
# ![Imgur](https://i.imgur.com/q5BBj8p.png)
# 

# ### Output:
# **Train on 21000 samples, validate on 9000 samples
# **![Imgur](https://i.imgur.com/QoB4Bft.png)

# ### Output:
# **Model accuracy & Model loss plot:
# **![Imgur](https://i.imgur.com/csePWWo.png)

# ### Output:
# ```
# Test Loss: 41.8
# Test Accuracy: 91
# ```

# # 8. Results

# ### Output:
# Final result of all
# ```
# KNN accuracy:  0.819555555556 
#  Random forest accuracy:  0.804555555556 
#  Linear SVC accuracy:  0.717777777778 
#  Gaussian Radial Basis Function SVC Accuracy:  0.781555555556 
#  Multi-Layer Perceptron accuracy:  0.845666666667 
#  Convolutional Neural Network Score: 0.909555555556 
# ```
#  

# # Conclusion:
# This notebook a demon for this compitions. I will only use around 5000 doodles for each label since the full dataset would be too much for my computer to handle. Then I will explore the drawings and graph random sketches of each
# category. Finally, I will test the dataset with Random Forest, Support-Vector Machine (SVM), KNearest
# Neighbors (KNN) and Multi-Layer Perceptron (MLP) classifiers in scikit-learn as well as
# a Convolutional Neural Network (CNN) in Karas.
# 
# Here is some reference :
# 1. [Quick Draw's Article](https://towardsdatascience.com/quick-draw-the-worlds-largest-doodle-dataset-823c22ffce6b)
# 2. [Quick Draw's Demo video](https://www.youtube.com/watch?list=PLIivdWyY5sqJxnwJhe3etaK7utrBiPBQ2&time_continue=2&v=8DEjphIfeYw)
# 3. [ Quick Draw's Paper](https://github.com/nolanadams1230/Doodle_Classification/blob/master/Final_Report.pdf)
# 
# If you enjoyed reading the kernel , hit the upvote button !
