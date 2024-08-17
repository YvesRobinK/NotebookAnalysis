#!/usr/bin/env python
# coding: utf-8

# # Awesome Deep Learning Basics and Resources with CNN MNIST Classifier
#       This kernel has the CNN MNIST Classifier using TensorFlow Estimators and MIT Deep Learning Basics along with a curated list of collections of Awesome Deep Learning Resources with two different parts as follows
# 
# > #### **Credits**: Thanks to **TensorFlow Team**, **Lex Fridman** for MIT Deep Learning, **Christos Christofidis**, **Guillaume Chevalier**  and other contributers for such wonderful curated collections
# 
# ### Here are some of *my kernel notebooks* for **Machine Learning and Data Science** as follows, ***Upvote*** them if you *like* them
# 
# > * [Data Science with R - Awesome Tutorials](https://www.kaggle.com/arunkumarramanan/data-science-with-r-awesome-tutorials)
# > * [Data Science and Machine Learning Cheetcheets](https://www.kaggle.com/arunkumarramanan/data-science-and-machine-learning-cheatsheets)
# > * [Awesome ML Frameworks and MNIST Classification](https://www.kaggle.com/arunkumarramanan/awesome-machine-learning-ml-frameworks)
# > * [Awesome Data Science for Beginners with Titanic Exploration](https://kaggle.com/arunkumarramanan/awesome-data-science-for-beginners)
# > * [Tensorflow Tutorial and House Price Prediction](https://www.kaggle.com/arunkumarramanan/tensorflow-tutorial-and-examples)
# > * [Practical Machine Learning with PyTorch](https://www.kaggle.com/arunkumarramanan/practical-machine-learning-with-pytorch)
# > * [Awesome Computer Vision Resources (TBU)](https://www.kaggle.com/arunkumarramanan/awesome-computer-vision-resources-to-be-updated)
# > * [Data Scientist's Toolkits - Awesome Data Science Resources](https://www.kaggle.com/arunkumarramanan/data-scientist-s-toolkits-awesome-ds-resources)
# > * [Data Science with Python - Awesome Tutorials](https://www.kaggle.com/arunkumarramanan/data-science-with-python-awesome-tutorials)
# > * [Machine Learning and Deep Learning - Awesome Tutorials](https://www.kaggle.com/arunkumarramanan/awesome-deep-learning-ml-tutorials)
# > * [Machine Learning Engineer's Toolkit with Roadmap](https://www.kaggle.com/arunkumarramanan/machine-learning-engineer-s-toolkit-with-roadmap) 
# > * [Awesome TensorFlow and PyTorch Resources](https://www.kaggle.com/arunkumarramanan/awesome-tensorflow-and-pytorch-resources)
# > * [Hands-on ML with scikit-learn and TensorFlow](https://www.kaggle.com/arunkumarramanan/hands-on-ml-with-scikit-learn-and-tensorflow)
# > * [Awesome Data Science IPython Notebooks](https://www.kaggle.com/arunkumarramanan/awesome-data-science-ipython-notebooks)
# > * [Awesome Deep Learning Basics and Resources](https://www.kaggle.com/arunkumarramanan/awesome-deep-learning-resources)

# # Building a Convolutional Neural Network CNN using Estimators from TensorFlow Docs
# 
# The `tf.layers` module provides a high-level API that makes
# it easy to construct a neural network. It provides methods that facilitate the
# creation of dense (fully connected) layers and convolutional layers, adding
# activation functions, and applying dropout regularization. In this tutorial,
# you'll learn how to use `layers` to build a convolutional neural network model
# to recognize the handwritten digits in the MNIST data set.
# 
# ![handwritten digits 0–9 from the MNIST data set](https://www.tensorflow.org/images/mnist_0-9.png)
# 
# The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) comprises 60,000
# training examples and 10,000 test examples of the handwritten digits 0–9,
# formatted as 28x28-pixel monochrome images.
# 
# ## Get Started
# 
# Let's set up the imports for our TensorFlow program:

# In[1]:


from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)


# As you work through the tutorial, you'll add code to construct, train, and
# evaluate the convolutional neural network. The complete, final code can be
# [found here](https://www.tensorflow.org/code/tensorflow/examples/tutorials/layers/cnn_mnist.py).
# 
# ## Intro to Convolutional Neural Networks
# 
# Convolutional neural networks (CNNs) are the current state-of-the-art model
# architecture for image classification tasks. CNNs apply a series of filters to
# the raw pixel data of an image to extract and learn higher-level features, which
# the model can then use for classification. CNNs contains three components:
# 
# *   **Convolutional layers**, which apply a specified number of convolution
#     filters to the image. For each subregion, the layer performs a set of
#     mathematical operations to produce a single value in the output feature map.
#     Convolutional layers then typically apply a
#     [ReLU activation function](https://en.wikipedia.org/wiki/Rectifier_\(neural_networks\)) to
#     the output to introduce nonlinearities into the model.
# 
# *   **Pooling layers**, which
#     [downsample the image data](https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer)
#     extracted by the convolutional layers to reduce the dimensionality of the
#     feature map in order to decrease processing time. A commonly used pooling
#     algorithm is max pooling, which extracts subregions of the feature map
#     (e.g., 2x2-pixel tiles), keeps their maximum value, and discards all other
#     values.
# 
# *   **Dense (fully connected) layers**, which perform classification on the
#     features extracted by the convolutional layers and downsampled by the
#     pooling layers. In a dense layer, every node in the layer is connected to
#     every node in the preceding layer.
# 
# Typically, a CNN is composed of a stack of convolutional modules that perform
# feature extraction. Each module consists of a convolutional layer followed by a
# pooling layer. The last convolutional module is followed by one or more dense
# layers that perform classification. The final dense layer in a CNN contains a
# single node for each target class in the model (all the possible classes the
# model may predict), with a
# [softmax](https://en.wikipedia.org/wiki/Softmax_function) activation function to
# generate a value between 0–1 for each node (the sum of all these softmax values
# is equal to 1). We can interpret the softmax values for a given image as
# relative measurements of how likely it is that the image falls into each target
# class.
# 
# Note: For a more comprehensive walkthrough of CNN architecture, see Stanford University's [Convolutional Neural Networks for Visual Recognition course material](https://cs231n.github.io/convolutional-networks/).

# ## Building the CNN MNIST Classifier
# 
# Let's build a model to classify the images in the MNIST dataset using the
# following CNN architecture:
# 
# 1.  **Convolutional Layer #1**: Applies 32 5x5 filters (extracting 5x5-pixel
#     subregions), with ReLU activation function
# 2.  **Pooling Layer #1**: Performs max pooling with a 2x2 filter and stride of 2
#     (which specifies that pooled regions do not overlap)
# 3.  **Convolutional Layer #2**: Applies 64 5x5 filters, with ReLU activation
#     function
# 4.  **Pooling Layer #2**: Again, performs max pooling with a 2x2 filter and
#     stride of 2
# 5.  **Dense Layer #1**: 1,024 neurons, with dropout regularization rate of 0.4
#     (probability of 0.4 that any given element will be dropped during training)
# 6.  **Dense Layer #2 (Logits Layer)**: 10 neurons, one for each digit target
#     class (0–9).
# 
# The `tf.layers` module contains methods to create each of the three layer types
# above:
# 
# *   `conv2d()`. Constructs a two-dimensional convolutional layer. Takes number
#     of filters, filter kernel size, padding, and activation function as
#     arguments.
# *   `max_pooling2d()`. Constructs a two-dimensional pooling layer using the
#     max-pooling algorithm. Takes pooling filter size and stride as arguments.
# *   `dense()`. Constructs a dense layer. Takes number of neurons and activation
#     function as arguments.
# 
# Each of these methods accepts a tensor as input and returns a transformed tensor
# as output. This makes it easy to connect one layer to another: just take the
# output from one layer-creation method and supply it as input to another.
# 
# Add the following `cnn_model_fn` function, which
# conforms to the interface expected by TensorFlow's Estimator API (more on this
# later in [Create the Estimator](#create-the-estimator)). This function takes
# MNIST feature data, labels, and mode (from
# `tf.estimator.ModeKeys`: `TRAIN`, `EVAL`, `PREDICT`) as arguments;
# configures the CNN; and returns predictions, loss, and a training operation:

# In[2]:


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# The following sections (with headings corresponding to each code block above)
# dive deeper into the `tf.layers` code used to create each layer, as well as how
# to calculate loss, configure the training op, and generate predictions. If
# you're already experienced with CNNs and [TensorFlow `Estimator`s](../../guide/custom_estimators.md),
# and find the above code intuitive, you may want to skim these sections or just
# skip ahead to ["Training and Evaluating the CNN MNIST Classifier"](#train_eval_mnist).
# 
# ### Input Layer
# 
# The methods in the `layers` module for creating convolutional and pooling layers
# for two-dimensional image data expect input tensors to have a shape of
# <code>[<em>batch_size</em>, <em>image_height</em>, <em>image_width</em>,
# <em>channels</em>]</code> by default. This behavior can be changed using the
# <code><em>data_format</em></code> parameter; defined as follows:
# 
# *   `batch_size` —Size of the subset of examples to use when performing
#     gradient descent during training.
# *   `image_height` —Height of the example images.
# *   `image_width` —Width of the example images.
# *   `channels` —Number of color channels in the example images. For color
#     images, the number of channels is 3 (red, green, blue). For monochrome
#     images, there is just 1 channel (black).
# *   `data_format` —A string, one of `channels_last` (default) or `channels_first`.
#       `channels_last` corresponds to inputs with shape
#       `(batch, ..., channels)` while `channels_first` corresponds to
#       inputs with shape `(batch, channels, ...)`.
# 
# Here, our MNIST dataset is composed of monochrome 28x28 pixel images, so the
# desired shape for our input layer is <code>[<em>batch_size</em>, 28, 28,
# 1]</code>.
# 
# To convert our input feature map (`features`) to this shape, we can perform the
# following `reshape` operation:
# 
# ```
# input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
# ```
# 
# Note that we've indicated `-1` for batch size, which specifies that this
# dimension should be dynamically computed based on the number of input values in
# `features["x"]`, holding the size of all other dimensions constant. This allows
# us to treat `batch_size` as a hyperparameter that we can tune. For example, if
# we feed examples into our model in batches of 5, `features["x"]` will contain
# 3,920 values (one value for each pixel in each image), and `input_layer` will
# have a shape of `[5, 28, 28, 1]`. Similarly, if we feed examples in batches of
# 100, `features["x"]` will contain 78,400 values, and `input_layer` will have a
# shape of `[100, 28, 28, 1]`.
# 
# ### Convolutional Layer #1
# 
# In our first convolutional layer, we want to apply 32 5x5 filters to the input
# layer, with a ReLU activation function. We can use the `conv2d()` method in the
# `layers` module to create this layer as follows:
# 
# ```
# conv1 = tf.layers.conv2d(
#     inputs=input_layer,
#     filters=32,
#     kernel_size=[5, 5],
#     padding="same",
#     activation=tf.nn.relu)
# ```
# 
# The `inputs` argument specifies our input tensor, which must have the shape
# <code>[<em>batch_size</em>, <em>image_height</em>, <em>image_width</em>,
# <em>channels</em>]</code>. Here, we're connecting our first convolutional layer
# to `input_layer`, which has the shape <code>[<em>batch_size</em>, 28, 28,
# 1]</code>.
# 
# Note: `conv2d()` will instead accept a shape of `[<em>batch_size</em>, <em>channels</em>, <em>image_height</em>, <em>image_width</em>]` when passed the argument `data_format=channels_first`.
# 
# The `filters` argument specifies the number of filters to apply (here, 32), and
# `kernel_size` specifies the dimensions of the filters as `[<em>height</em>,
# <em>width</em>]</code> (here, <code>[5, 5]`).
# 
# <p class="tip"><b>TIP:</b> If filter height and width have the same value, you can instead specify a
# single integer for <code>kernel_size</code>—e.g., <code>kernel_size=5</code>.</p>
# 
# The `padding` argument specifies one of two enumerated values
# (case-insensitive): `valid` (default value) or `same`. To specify that the
# output tensor should have the same height and width values as the input tensor,
# we set `padding=same` here, which instructs TensorFlow to add 0 values to the
# edges of the input tensor to preserve height and width of 28. (Without padding,
# a 5x5 convolution over a 28x28 tensor will produce a 24x24 tensor, as there are
# 24x24 locations to extract a 5x5 tile from a 28x28 grid.)
# 
# The `activation` argument specifies the activation function to apply to the
# output of the convolution. Here, we specify ReLU activation with
# `tf.nn.relu`.
# 
# Our output tensor produced by `conv2d()` has a shape of
# <code>[<em>batch_size</em>, 28, 28, 32]</code>: the same height and width
# dimensions as the input, but now with 32 channels holding the output from each
# of the filters.

# ### Pooling Layer #1
# 
# Next, we connect our first pooling layer to the convolutional layer we just
# created. We can use the `max_pooling2d()` method in `layers` to construct a
# layer that performs max pooling with a 2x2 filter and stride of 2:
# 
# ```
# pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
# ```
# 
# Again, `inputs` specifies the input tensor, with a shape of
# <code>[<em>batch_size</em>, <em>image_height</em>, <em>image_width</em>,
# <em>channels</em>]</code>. Here, our input tensor is `conv1`, the output from
# the first convolutional layer, which has a shape of <code>[<em>batch_size</em>,
# 28, 28, 32]</code>.
# 
# Note: As with <code>conv2d()</code>, <code>max_pooling2d()</code> will instead
# accept a shape of <code>[<em>batch_size</em>, <em>channels</em>, 
# <em>image_height</em>, <em>image_width</em>]</code> when passed the argument
# <code>data_format=channels_first</code>.
# 
# The `pool_size` argument specifies the size of the max pooling filter as
# <code>[<em>height</em>, <em>width</em>]</code> (here, `[2, 2]`). If both
# dimensions have the same value, you can instead specify a single integer (e.g.,
# `pool_size=2`).
# 
# The `strides` argument specifies the size of the stride. Here, we set a stride
# of 2, which indicates that the subregions extracted by the filter should be
# separated by 2 pixels in both the height and width dimensions (for a 2x2 filter,
# this means that none of the regions extracted will overlap). If you want to set
# different stride values for height and width, you can instead specify a tuple or
# list (e.g., `stride=[3, 6]`).
# 
# Our output tensor produced by `max_pooling2d()` (`pool1`) has a shape of
# <code>[<em>batch_size</em>, 14, 14, 32]</code>: the 2x2 filter reduces height and width by 50% each.

# ### Convolutional Layer #2 and Pooling Layer #2
# 
# We can connect a second convolutional and pooling layer to our CNN using
# `conv2d()` and `max_pooling2d()` as before. For convolutional layer #2, we
# configure 64 5x5 filters with ReLU activation, and for pooling layer #2, we use
# the same specs as pooling layer #1 (a 2x2 max pooling filter with stride of 2):
# 
# ```
# conv2 = tf.layers.conv2d(
#     inputs=pool1,
#     filters=64,
#     kernel_size=[5, 5],
#     padding="same",
#     activation=tf.nn.relu)
# 
# pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
# ```
# 
# Note that convolutional layer #2 takes the output tensor of our first pooling
# layer (`pool1`) as input, and produces the tensor `conv2` as output. `conv2`
# has a shape of <code>[<em>batch_size</em>, 14, 14, 64]</code>, the same height and width as `pool1` (due to `padding="same"`), and 64 channels for the 64
# filters applied.
# 
# Pooling layer #2 takes `conv2` as input, producing `pool2` as output. `pool2`
# has shape <code>[<em>batch_size</em>, 7, 7, 64]</code> (50% reduction of height and width from `conv2`).

# ### Dense Layer
# 
# Next, we want to add a dense layer (with 1,024 neurons and ReLU activation) to
# our CNN to perform classification on the features extracted by the
# convolution/pooling layers. Before we connect the layer, however, we'll flatten
# our feature map (`pool2`) to shape <code>[<em>batch_size</em>,
# <em>features</em>]</code>, so that our tensor has only two dimensions:
# 
# ```
# pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
# ```
# 
# In the `reshape()` operation above, the `-1` signifies that the *`batch_size`*
# dimension will be dynamically calculated based on the number of examples in our
# input data. Each example has 7 (`pool2` height) * 7 (`pool2` width) * 64
# (`pool2` channels) features, so we want the `features` dimension to have a value
# of 7 * 7 * 64 (3136 in total). The output tensor, `pool2_flat`, has shape
# <code>[<em>batch_size</em>, 3136]</code>.
# 
# Now, we can use the `dense()` method in `layers` to connect our dense layer as
# follows:
# 
# ```
# dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
# ```
# 
# The `inputs` argument specifies the input tensor: our flattened feature map,
# `pool2_flat`. The `units` argument specifies the number of neurons in the dense
# layer (1,024). The `activation` argument takes the activation function; again,
# we'll use `tf.nn.relu` to add ReLU activation.
# 
# To help improve the results of our model, we also apply dropout regularization
# to our dense layer, using the `dropout` method in `layers`:
# 
# ```
# dropout = tf.layers.dropout(
#     inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
# ```
# 
# Again, `inputs` specifies the input tensor, which is the output tensor from our
# dense layer (`dense`).
# 
# The `rate` argument specifies the dropout rate; here, we use `0.4`, which means
# 40% of the elements will be randomly dropped out during training.
# 
# The `training` argument takes a boolean specifying whether or not the model is
# currently being run in training mode; dropout will only be performed if
# `training` is `True`. Here, we check if the `mode` passed to our model function
# `cnn_model_fn` is `TRAIN` mode.
# 
# Our output tensor `dropout` has shape <code>[<em>batch_size</em>, 1024]</code>.

# ### Logits Layer
# 
# The final layer in our neural network is the logits layer, which will return the
# raw values for our predictions. We create a dense layer with 10 neurons (one for
# each target class 0–9), with linear activation (the default):
# 
# ```
# logits = tf.layers.dense(inputs=dropout, units=10)
# ```
# 
# Our final output tensor of the CNN, `logits`, has shape `[batch_size, 10]`.

# ### Generate Predictions {#generate_predictions}
# 
# The logits layer of our model returns our predictions as raw values in a
# <code>[<em>batch_size</em>, 10]</code>-dimensional tensor. Let's convert these
# raw values into two different formats that our model function can return:
# 
# *   The **predicted class** for each example: a digit from 0–9.
# *   The **probabilities** for each possible target class for each example: the
#     probability that the example is a 0, is a 1, is a 2, etc.
# 
# For a given example, our predicted class is the element in the corresponding row
# of the logits tensor with the highest raw value. We can find the index of this
# element using the `tf.argmax`
# function:
# 
# ```
# tf.argmax(input=logits, axis=1)
# ```
# 
# The `input` argument specifies the tensor from which to extract maximum
# values—here `logits`. The `axis` argument specifies the axis of the `input`
# tensor along which to find the greatest value. Here, we want to find the largest
# value along the dimension with index of 1, which corresponds to our predictions
# (recall that our logits tensor has shape <code>[<em>batch_size</em>,
# 10]</code>).
# 
# We can derive probabilities from our logits layer by applying softmax activation
# using `tf.nn.softmax`:
# 
# ```
# tf.nn.softmax(logits, name="softmax_tensor")
# ```
# 
# Note: We use the `name` argument to explicitly name this operation `softmax_tensor`, so we can reference it later. (We'll set up logging for the softmax values in ["Set Up a Logging Hook"](#set-up-a-logging-hook)).
# 
# We compile our predictions in a dict, and return an `EstimatorSpec` object:
# 
# ```
# predictions = {
#     "classes": tf.argmax(input=logits, axis=1),
#     "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
# }
# if mode == tf.estimator.ModeKeys.PREDICT:
#   return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
# ```

# ### Calculate Loss {#calculating-loss}
# 
# For both training and evaluation, we need to define a
# [loss function](https://en.wikipedia.org/wiki/Loss_function)
# that measures how closely the model's predictions match the target classes. For
# multiclass classification problems like MNIST,
# [cross entropy](https://en.wikipedia.org/wiki/Cross_entropy) is typically used
# as the loss metric. The following code calculates cross entropy when the model
# runs in either `TRAIN` or `EVAL` mode:
# 
# ```
# loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
# ```
# 
# Let's take a closer look at what's happening above.
# 
# Our `labels` tensor contains a list of prediction indices for our examples, e.g. `[1,
# 9, ...]`. `logits` contains the linear outputs of our last layer. 
# 
# `tf.losses.sparse_softmax_cross_entropy`, calculates the softmax crossentropy
# (aka: categorical crossentropy, negative log-likelihood) from these two inputs
# in an efficient, numerically stable way.

# ### Configure the Training Op
# 
# In the previous section, we defined loss for our CNN as the softmax
# cross-entropy of the logits layer and our labels. Let's configure our model to
# optimize this loss value during training. We'll use a learning rate of 0.001 and
# [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
# as the optimization algorithm:
# 
# ```
# if mode == tf.estimator.ModeKeys.TRAIN:
#   optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
#   train_op = optimizer.minimize(
#       loss=loss,
#       global_step=tf.train.get_global_step())
#   return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
# ```

# ### Add evaluation metrics
# 
# To add accuracy metric in our model, we define `eval_metric_ops` dict in EVAL
# mode as follows:
# 
# ```
# eval_metric_ops = {
#     "accuracy": tf.metrics.accuracy(
#         labels=labels, predictions=predictions["classes"])
# }
# return tf.estimator.EstimatorSpec(
#     mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
# ```

# <a id="train_eval_mnist"></a>
# ## Training and Evaluating the CNN MNIST Classifier
# 
# We've coded our MNIST CNN model function; now we're ready to train and evaluate
# it.
# 
# ### Load Training and Test Data
# 
# First, let's load our training and test data with the following code:

# In[3]:


# Load training and eval data
((train_data, train_labels),
 (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

train_data = train_data/np.float32(255)
train_labels = train_labels.astype(np.int32)  # not required

eval_data = eval_data/np.float32(255)
eval_labels = eval_labels.astype(np.int32)  # not required


# We store the training feature data (the raw pixel values for 55,000 images of
# hand-drawn digits) and training labels (the corresponding value from 0–9 for
# each image) as [numpy
# arrays](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html)
# in `train_data` and `train_labels`, respectively. Similarly, we store the
# evaluation feature data (10,000 images) and evaluation labels in `eval_data`
# and `eval_labels`, respectively
# 
# ### Create the Estimator {#create-the-estimator}
# 
# Next, let's create an `Estimator` (a TensorFlow class for performing high-level
# model training, evaluation, and inference) for our model. Add the following code
# to `main()`:

# In[4]:


# Create the Estimator
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")


# The `model_fn` argument specifies the model function to use for training,
# evaluation, and prediction; we pass it the `cnn_model_fn` we created in
# ["Building the CNN MNIST Classifier."](#building-the-cnn-mnist-classifier) The
# `model_dir` argument specifies the directory where model data (checkpoints) will
# be saved (here, we specify the temp directory `/tmp/mnist_convnet_model`, but
# feel free to change to another directory of your choice).
# 
# Note: For an in-depth walkthrough of the TensorFlow `Estimator` API, see the tutorial [Creating Estimators in tf.estimator](../../guide/custom_estimators.md).

# ### Set Up a Logging Hook {#set_up_a_logging_hook}
# 
# Since CNNs can take a while to train, let's set up some logging so we can track
# progress during training. We can use TensorFlow's `tf.train.SessionRunHook` to create a
# `tf.train.LoggingTensorHook`
# that will log the probability values from the softmax layer of our CNN. Add the
# following to `main()`:

# In[5]:


# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}

logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)


# We store a dict of the tensors we want to log in `tensors_to_log`. Each key is a
# label of our choice that will be printed in the log output, and the
# corresponding label is the name of a `Tensor` in the TensorFlow graph. Here, our
# `probabilities` can be found in `softmax_tensor`, the name we gave our softmax
# operation earlier when we generated the probabilities in `cnn_model_fn`.
# 
# Note: If you don't explicitly assign a name to an operation via the `name` argument, TensorFlow will assign a default name. A couple easy ways to discover the names applied to operations are to visualize your graph on [TensorBoard](../../guide/graph_viz.md)) or to enable the [TensorFlow Debugger (tfdbg)](../../guide/debugger.md).
# 
# Next, we create the `LoggingTensorHook`, passing `tensors_to_log` to the
# `tensors` argument. We set `every_n_iter=50`, which specifies that probabilities
# should be logged after every 50 steps of training.
# 
# ### Train the Model
# 
# Now we're ready to train our model, which we can do by creating `train_input_fn`
# and calling `train()` on `mnist_classifier`. In the `numpy_input_fn` call, we pass the training feature data and labels to
# `x` (as a dict) and `y`, respectively. We set a `batch_size` of `100` (which
# means that the model will train on minibatches of 100 examples at each step).
# `num_epochs=None` means that the model will train until the specified number of
# steps is reached. We also set `shuffle=True` to shuffle the training data. Then train the model a single step and log the output:

# In[6]:


# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

# train one step and display the probabilties
mnist_classifier.train(
    input_fn=train_input_fn,
    steps=1,
    hooks=[logging_hook])


# Now—without logging each step—set `steps=1000` to train the model longer, but in a reasonable time to run this example. Training CNNs is computationally intensive. To increase the accuracy of your model, increase the number of `steps` passed to `train()`, like 20,000 steps. 

# In[7]:


mnist_classifier.train(input_fn=train_input_fn, steps=1000)


# ### Evaluate the Model
# 
# Once training is complete, we want to evaluate our model to determine its
# accuracy on the MNIST test set. We call the `evaluate` method, which evaluates
# the metrics we specified in `eval_metric_ops` argument in the `model_fn`.
# Add the following to `main()`:

# In[8]:


eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)

eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)


# To create `eval_input_fn`, we set `num_epochs=1`, so that the model evaluates
# the metrics over one epoch of data and returns the result. We also set
# `shuffle=False` to iterate through the data sequentially.
# 
# ## Additional Resources
# 
# To learn more about TensorFlow Estimators and CNNs in TensorFlow, see the
# following resources:
# 
# *   [Creating Estimators in tf.estimator](../../guide/custom_estimators.md)
#     provides an introduction to the TensorFlow Estimator API. It walks through
#     configuring an Estimator, writing a model function, calculating loss, and
#     defining a training op.
# *   [Advanced Convolutional Neural Networks](../../tutorials/images/deep_cnn.md) walks through how to build a MNIST CNN classification model
#     *without estimators* using lower-level TensorFlow operations.

# ## License
# 
# ##### Copyright 2018 The TensorFlow Authors.

# In[9]:


#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# ## Deep Learning Basics with TensorFlow
# 
# This tutorial accompanies the [lecture on Deep Learning Basics](https://www.youtube.com/watch?list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf&v=O5xeyoRL95U) given as part of [MIT Deep Learning](https://deeplearning.mit.edu). Acknowledgement to amazing people involved is provided throughout the tutorial and at the end. You can watch the video on YouTube:
# 
# In this tutorial, we mention seven important types/concepts/approaches in deep learning, introducing the first 2 and providing pointers to tutorials on the others. Here is a visual representation of the seven:
# 
# ![Deep learning concepts](https://i.imgur.com/EAl47rp.png)
# 
# At a high-level, neural networks are either encoders, decoders, or a combination of both. Encoders find patterns in raw data to form compact, useful representations. Decoders generate new data or high-resolution useful infomation from those representations. As the lecture describes, deep learning discovers ways to **represent** the world so that we can reason about it. The rest is clever methods that help use deal effectively with visual information, language, sound (#1-6) and even act in a world based on this information and occasional rewards (#7).
# 
# 1. **Feed Forward Neural Networks (FFNNs)** - classification and regression based on features. See [Part 1](#Part-1:-Boston-Housing-Price-Prediction-with-Feed-Forward-Neural-Networks) of this tutorial for an example.
# 2. **Convolutional Neural Networks (CNNs)** - image classification, object detection, video action recognition, etc. See [Part 2](#Part-2:-Classification-of-MNIST-Dreams-with-Convolution-Neural-Networks) of this tutorial for an example.
# 3. **Recurrent Neural Networks (RNNs)** - language modeling, speech recognition/generation, etc. See [this TF tutorial on text generation](https://www.tensorflow.org/tutorials/sequences/text_generation) for an example.
# 4. **Encoder Decoder Architectures** - semantic segmentation, machine translation, etc. See [our tutorial on semantic segmentation](https://github.com/lexfridman/mit-deep-learning/blob/master/tutorial_driving_scene_segmentation/tutorial_driving_scene_segmentation.ipynb) for an example.
# 5. **Autoencoder** - unsupervised embeddings, denoising, etc.
# 6. **Generative Adversarial Networks (GANs)** - unsupervised generation of realistic images, etc. See [this TF tutorial on DCGANs](https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/eager/python/examples/generative_examples/dcgan.ipynb) for an example.
# 7. **Deep Reinforcement Learning** - game playing, robotics in simulation, self-play, neural arhitecture search, etc. We'll be releasing notebooks on this soon and will link them here.
# 
# ## License
# 
# MIT License
# 
# Copyright (c) 2019 Lex Fridman
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# 
# 
# 

# # Awesome Deep Learning Resources
# 
# ## Awesome Deep Learning - Part I
# 
# ### Table of Content - Part I
# 
# * **[Free Online Books](#free-online-books)**  
# 
# * **[Courses](#courses)**  
# 
# * **[Videos and Lectures](#videos-and-lectures)**  
# 
# * **[Papers](#papers)**  
# 
# * **[Tutorials](#tutorials)**  
# 
# * **[Researchers](#researchers)**  
# 
# * **[Websites](#websites)**  
# 
# * **[Datasets](#datasets)**
# 
# * **[Conferences](#Conferences)**
# 
# * **[Frameworks](#frameworks)**  
# 
# * **[Tools](#tools)**  
# 
# * **[Miscellaneous](#miscellaneous)**  
# 
#  
# ### Free Online Books
# 
# 1.  [Deep Learning](http://www.deeplearningbook.org/) by Yoshua Bengio, Ian Goodfellow and Aaron Courville  (05/07/2015)
# 2.  [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by  Michael Nielsen (Dec 2014)
# 3.  [Deep Learning](http://research.microsoft.com/pubs/209355/DeepLearning-NowPublishing-Vol7-SIG-039.pdf) by Microsoft Research (2013) 
# 4.  [Deep Learning Tutorial](http://deeplearning.net/tutorial/deeplearning.pdf) by LISA lab, University of Montreal (Jan 6 2015)
# 5.  [neuraltalk](https://github.com/karpathy/neuraltalk) by Andrej Karpathy : numpy-based RNN/LSTM implementation
# 6.  [An introduction to genetic algorithms](https://svn-d1.mpi-inf.mpg.de/AG1/MultiCoreLab/papers/ebook-fuzzy-mitchell-99.pdf)
# 7.  [Artificial Intelligence: A Modern Approach](http://aima.cs.berkeley.edu/)
# 8.  [Deep Learning in Neural Networks: An Overview](http://arxiv.org/pdf/1404.7828v4.pdf)
# 9.  [Artificial intelligence and machine learning: Topic wise explanation](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/)
#  
# ### Courses
# 
# 1.  [Machine Learning - Stanford](https://class.coursera.org/ml-005) by Andrew Ng in Coursera (2010-2014)
# 2.  [Machine Learning - Caltech](http://work.caltech.edu/lectures.html) by Yaser Abu-Mostafa (2012-2014)
# 3.  [Machine Learning - Carnegie Mellon](http://www.cs.cmu.edu/~tom/10701_sp11/lectures.shtml) by Tom Mitchell (Spring 2011)
# 2.  [Neural Networks for Machine Learning](https://class.coursera.org/neuralnets-2012-001) by Geoffrey Hinton in Coursera (2012)
# 3.  [Neural networks class](https://www.youtube.com/playlist?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH) by Hugo Larochelle from Université de Sherbrooke (2013)
# 4.  [Deep Learning Course](http://cilvr.cs.nyu.edu/doku.php?id=deeplearning:slides:start) by CILVR lab @ NYU (2014)
# 5.  [A.I - Berkeley](https://courses.edx.org/courses/BerkeleyX/CS188x_1/1T2013/courseware/) by Dan Klein and Pieter Abbeel (2013)
# 6.  [A.I - MIT](http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/lecture-videos/) by Patrick Henry Winston (2010)
# 7.  [Vision and learning - computers and brains](http://web.mit.edu/course/other/i2course/www/vision_and_learning_fall_2013.html) by Shimon Ullman, Tomaso Poggio, Ethan Meyers @ MIT (2013)
# 9.  [Convolutional Neural Networks for Visual Recognition - Stanford](http://vision.stanford.edu/teaching/cs231n/syllabus.html) by Fei-Fei Li, Andrej Karpathy (2017)
# 10.  [Deep Learning for Natural Language Processing - Stanford](http://cs224d.stanford.edu/)
# 11.  [Neural Networks - usherbrooke](http://info.usherbrooke.ca/hlarochelle/neural_networks/content.html)
# 12.  [Machine Learning - Oxford](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/) (2014-2015)
# 13.  [Deep Learning - Nvidia](https://developer.nvidia.com/deep-learning-courses) (2015)
# 14.  [Graduate Summer School: Deep Learning, Feature Learning](https://www.youtube.com/playlist?list=PLHyI3Fbmv0SdzMHAy0aN59oYnLy5vyyTA) by Geoffrey Hinton, Yoshua Bengio, Yann LeCun, Andrew Ng, Nando de Freitas and several others @ IPAM, UCLA (2012)
# 15.  [Deep Learning - Udacity/Google](https://www.udacity.com/course/deep-learning--ud730) by Vincent Vanhoucke and Arpan Chakraborty (2016)
# 16.  [Deep Learning - UWaterloo](https://www.youtube.com/playlist?list=PLehuLRPyt1Hyi78UOkMPWCGRxGcA9NVOE) by Prof. Ali Ghodsi at University of Waterloo (2015)
# 17.  [Statistical Machine Learning - CMU](https://www.youtube.com/watch?v=azaLcvuql_g&list=PLjbUi5mgii6BWEUZf7He6nowWvGne_Y8r) by Prof. Larry Wasserman
# 18.  [Deep Learning Course](https://www.college-de-france.fr/site/en-yann-lecun/course-2015-2016.htm) by Yann LeCun (2016)
# 19. [Designing, Visualizing and Understanding Deep Neural Networks-UC Berkeley](https://www.youtube.com/playlist?list=PLkFD6_40KJIxopmdJF_CLNqG3QuDFHQUm)
# 20. [UVA Deep Learning Course](http://uvadlc.github.io) MSc in Artificial Intelligence for the University of Amsterdam.
# 21. [MIT 6.S094: Deep Learning for Self-Driving Cars](http://selfdrivingcars.mit.edu/)
# 22. [MIT 6.S191: Introduction to Deep Learning](http://introtodeeplearning.com/)
# 23. [Berkeley CS 294: Deep Reinforcement Learning](http://rll.berkeley.edu/deeprlcourse/)
# 24. [Keras in Motion video course](https://www.manning.com/livevideo/keras-in-motion)
# 25. [Practical Deep Learning For Coders](http://course.fast.ai/) by Jeremy Howard - Fast.ai
# 26. [Introduction to Deep Learning](http://deeplearning.cs.cmu.edu/) by Prof. Bhiksha Raj (2017)
# 27. [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/)
# 
# ### Videos and Lectures
# 
# 1.  [How To Create A Mind](https://www.youtube.com/watch?v=RIkxVci-R4k) By Ray Kurzweil
# 2.  [Deep Learning, Self-Taught Learning and Unsupervised Feature Learning](https://www.youtube.com/watch?v=n1ViNeWhC24) By Andrew Ng
# 3.  [Recent Developments in Deep Learning](https://www.youtube.com/watch?v=vShMxxqtDDs&amp;index=3&amp;list=PL78U8qQHXgrhP9aZraxTT5-X1RccTcUYT) By Geoff Hinton
# 4.  [The Unreasonable Effectiveness of Deep Learning](https://www.youtube.com/watch?v=sc-KbuZqGkI) by Yann LeCun
# 5.  [Deep Learning of Representations](https://www.youtube.com/watch?v=4xsVFLnHC_0) by Yoshua bengio
# 6.  [Principles of Hierarchical Temporal Memory](https://www.youtube.com/watch?v=6ufPpZDmPKA) by Jeff Hawkins
# 7.  [Machine Learning Discussion Group - Deep Learning w/ Stanford AI Lab](https://www.youtube.com/watch?v=2QJi0ArLq7s&amp;list=PL78U8qQHXgrhP9aZraxTT5-X1RccTcUYT) by Adam Coates
# 8.  [Making Sense of the World with Deep Learning](http://vimeo.com/80821560) By Adam Coates 
# 9.  [Demystifying Unsupervised Feature Learning ](https://www.youtube.com/watch?v=wZfVBwOO0-k) By Adam Coates 
# 10.  [Visual Perception with Deep Learning](https://www.youtube.com/watch?v=3boKlkPBckA) By Yann LeCun
# 11.  [The Next Generation of Neural Networks](https://www.youtube.com/watch?v=AyzOUbkUf3M) By Geoffrey Hinton at GoogleTechTalks
# 12.  [The wonderful and terrifying implications of computers that can learn](http://www.ted.com/talks/jeremy_howard_the_wonderful_and_terrifying_implications_of_computers_that_can_learn) By Jeremy Howard at TEDxBrussels
# 13.  [Unsupervised Deep Learning - Stanford](http://web.stanford.edu/class/cs294a/handouts.html) by Andrew Ng in Stanford (2011)
# 14.  [Natural Language Processing](http://web.stanford.edu/class/cs224n/handouts/) By Chris Manning in Stanford
# 15.  [A beginners Guide to Deep Neural Networks](http://googleresearch.blogspot.com/2015/09/a-beginners-guide-to-deep-neural.html) By Natalie Hammel and Lorraine Yurshansky
# 16.  [Deep Learning: Intelligence from Big Data](https://www.youtube.com/watch?v=czLI3oLDe8M) by Steve Jurvetson (and panel) at VLAB in Stanford. 
# 17. [Introduction to Artificial Neural Networks and Deep Learning](https://www.youtube.com/watch?v=FoO8qDB8gUU) by Leo Isikdogan at Motorola Mobility HQ
# 18. [NIPS 2016 lecture and workshop videos](https://nips.cc/Conferences/2016/Schedule) - NIPS 2016
# 19. [Deep Learning Crash Course](https://www.youtube.com/watch?v=oS5fz_mHVz0&list=PLWKotBjTDoLj3rXBL-nEIPRN9V3a9Cx07): a series of mini-lectures by Leo Isikdogan on YouTube (2018)
# 
# ### Papers
# *You can also find the most cited deep learning papers from [here](https://github.com/terryum/awesome-deep-learning-papers)*
# 
# 1.  [ImageNet Classification with Deep Convolutional Neural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
# 2.  [Using Very Deep Autoencoders for Content Based Image Retrieval](http://www.cs.toronto.edu/~hinton/absps/esann-deep-final.pdf)
# 3.  [Learning Deep Architectures for AI](http://www.iro.umontreal.ca/~lisa/pointeurs/TR1312.pdf)
# 4.  [CMU’s list of papers](http://deeplearning.cs.cmu.edu/)
# 5.  [Neural Networks for Named Entity Recognition](http://nlp.stanford.edu/~socherr/pa4_ner.pdf) [zip](http://nlp.stanford.edu/~socherr/pa4-ner.zip)
# 6. [Training tricks by YB](http://www.iro.umontreal.ca/~bengioy/papers/YB-tricks.pdf)
# 7. [Geoff Hinton's reading list (all papers)](http://www.cs.toronto.edu/~hinton/deeprefs.html)
# 8. [Supervised Sequence Labelling with Recurrent Neural Networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
# 9.  [Statistical Language Models based on Neural Networks](http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf)
# 10.  [Training Recurrent Neural Networks](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf)
# 11.  [Recursive Deep Learning for Natural Language Processing and Computer Vision](http://nlp.stanford.edu/~socherr/thesis.pdf)
# 12.  [Bi-directional RNN](http://www.di.ufpe.br/~fnj/RNA/bibliografia/BRNN.pdf)
# 13.  [LSTM](http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf)
# 14.  [GRU - Gated Recurrent Unit](http://arxiv.org/pdf/1406.1078v3.pdf)
# 15.  [GFRNN](http://arxiv.org/pdf/1502.02367v3.pdf) [.](http://jmlr.org/proceedings/papers/v37/chung15.pdf) [.](http://jmlr.org/proceedings/papers/v37/chung15-supp.pdf)
# 16.  [LSTM: A Search Space Odyssey](http://arxiv.org/pdf/1503.04069v1.pdf)
# 17.  [A Critical Review of Recurrent Neural Networks for Sequence Learning](http://arxiv.org/pdf/1506.00019v1.pdf)
# 18.  [Visualizing and Understanding Recurrent Networks](http://arxiv.org/pdf/1506.02078v1.pdf)
# 19.  [Wojciech Zaremba, Ilya Sutskever, An Empirical Exploration of Recurrent Network Architectures](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
# 20.  [Recurrent Neural Network based Language Model](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)
# 21.  [Extensions of Recurrent Neural Network Language Model](http://www.fit.vutbr.cz/research/groups/speech/publi/2011/mikolov_icassp2011_5528.pdf)
# 22.  [Recurrent Neural Network based Language Modeling in Meeting Recognition](http://www.fit.vutbr.cz/~imikolov/rnnlm/ApplicationOfRNNinMeetingRecognition_IS2011.pdf)
# 23.  [Deep Neural Networks for Acoustic Modeling in Speech Recognition](http://cs224d.stanford.edu/papers/maas_paper.pdf)
# 24.  [Speech Recognition with Deep Recurrent Neural Networks](http://www.cs.toronto.edu/~fritz/absps/RNN13.pdf)
# 25.  [Reinforcement Learning Neural Turing Machines](http://arxiv.org/pdf/1505.00521v1)
# 26.  [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](http://arxiv.org/pdf/1406.1078v3.pdf)
# 27. [Google - Sequence to Sequence  Learning with Neural Networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
# 28. [Memory Networks](http://arxiv.org/pdf/1410.3916v10)
# 29. [Policy Learning with Continuous Memory States for Partially Observed Robotic Control](http://arxiv.org/pdf/1507.01273v1)
# 30. [Microsoft - Jointly Modeling Embedding and Translation to Bridge Video and Language](http://arxiv.org/pdf/1505.01861v1.pdf)
# 31. [Neural Turing Machines](http://arxiv.org/pdf/1410.5401v2.pdf)
# 32. [Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](http://arxiv.org/pdf/1506.07285v1.pdf)
# 33. [Mastering the Game of Go with Deep Neural Networks and Tree Search](http://www.nature.com/nature/journal/v529/n7587/pdf/nature16961.pdf)
# 34. [Batch Normalization](https://arxiv.org/abs/1502.03167)
# 35. [Residual Learning](https://arxiv.org/pdf/1512.03385v1.pdf)
# 36. [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004v1.pdf) 
# 37. [Berkeley AI Research (BAIR) Laboratory](https://arxiv.org/pdf/1611.07004v1.pdf) 
# 38. [MobileNets by Google](https://arxiv.org/abs/1704.04861)
# 39. [Cross Audio-Visual Recognition in the Wild Using Deep Learning](https://arxiv.org/abs/1706.05739)
# 40. [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)
# 41. [Matrix Capsules With Em Routing](https://openreview.net/pdf?id=HJWLfGWRb)
# 42. [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
# 43. [Collection of Popular Deep Learning Papers](https://github.com/ArunkumarRamanan/Computer-Science-Resources)
# 
# ### Tutorials
# 
# 1.  [UFLDL Tutorial 1](http://deeplearning.stanford.edu/wiki/index.php/UFLDL_Tutorial)
# 2.  [UFLDL Tutorial 2](http://ufldl.stanford.edu/tutorial/supervised/LinearRegression/)
# 3.  [Deep Learning for NLP (without Magic)](http://www.socher.org/index.php/DeepLearningTutorial/DeepLearningTutorial)
# 4.  [A Deep Learning Tutorial: From Perceptrons to Deep Networks](http://www.toptal.com/machine-learning/an-introduction-to-deep-learning-from-perceptrons-to-deep-networks)
# 5.  [Deep Learning from the Bottom up](http://www.metacademy.org/roadmaps/rgrosse/deep_learning)
# 6.  [Theano Tutorial](http://deeplearning.net/tutorial/deeplearning.pdf)
# 7.  [Neural Networks for Matlab](http://uk.mathworks.com/help/pdf_doc/nnet/nnet_ug.pdf)
# 8.  [Using convolutional neural nets to detect facial keypoints tutorial](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/)
# 9.  [Torch7 Tutorials](https://github.com/clementfarabet/ipam-tutorials/tree/master/th_tutorials)
# 10.  [The Best Machine Learning Tutorials On The Web](https://github.com/josephmisiti/machine-learning-module)
# 11. [VGG Convolutional Neural Networks Practical](http://www.robots.ox.ac.uk/~vgg/practicals/cnn/index.html)
# 12. [TensorFlow tutorials](https://github.com/nlintz/TensorFlow-Tutorials)
# 13. [More TensorFlow tutorials](https://github.com/pkmital/tensorflow_tutorials)
# 13. [TensorFlow Python Notebooks](https://github.com/aymericdamien/TensorFlow-Examples)
# 14. [Keras and Lasagne Deep Learning Tutorials](https://github.com/Vict0rSch/deep_learning)
# 15. [Classification on raw time series in TensorFlow with a LSTM RNN](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition)
# 16. [Using convolutional neural nets to detect facial keypoints tutorial](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/)
# 17. [TensorFlow-World](https://github.com/astorfi/TensorFlow-World)
# 18. [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python)
# 19. [Grokking Deep Learning](https://www.manning.com/books/grokking-deep-learning)
# 20. [Deep Learning for Search](https://www.manning.com/books/deep-learning-for-search)
# 21. [Keras Tutorial: Content Based Image Retrieval Using a Convolutional Denoising Autoencoder](https://blog.sicara.com/keras-tutorial-content-based-image-retrieval-convolutional-denoising-autoencoder-dc91450cc511)
# 22. [Pytorch Tutorial by Yunjey Choi](https://github.com/yunjey/pytorch-tutorial)
# 23. [Practical Machine Learning with PyTorch](https://www.kaggle.com/arunkumarramanan/practical-machine-learning-with-pytorch)
# 24. [Hands-On Machine Learning with scikit-learn and TensorFlow](https://www.kaggle.com/arunkumarramanan/hands-on-ml-with-scikit-learn-and-tensorflow)
# 25. [Awesome TensorFlow Tutorials](https://www.kaggle.com/arunkumarramanan/tensorflow-tutorial-and-examples)
# 26. [Awesome TensorFlow Resources](https://www.kaggle.com/arunkumarramanan/awesome-tensorflow-resources)
# 
# ## Researchers
# 
# 1. [Aaron Courville](http://aaroncourville.wordpress.com)
# 2. [Abdel-rahman Mohamed](http://www.cs.toronto.edu/~asamir/)
# 3. [Adam Coates](http://cs.stanford.edu/~acoates/)
# 4. [Alex Acero](http://research.microsoft.com/en-us/people/alexac/)
# 5. [ Alex Krizhevsky ](http://www.cs.utoronto.ca/~kriz/index.html)
# 6. [ Alexander Ilin ](http://users.ics.aalto.fi/alexilin/)
# 7. [ Amos Storkey ](http://homepages.inf.ed.ac.uk/amos/)
# 8. [ Andrej Karpathy ](http://cs.stanford.edu/~karpathy/)
# 9. [ Andrew M. Saxe ](http://www.stanford.edu/~asaxe/)
# 10. [ Andrew Ng ](http://www.cs.stanford.edu/people/ang/)
# 11. [ Andrew W. Senior ](http://research.google.com/pubs/author37792.html)
# 12. [ Andriy Mnih ](http://www.gatsby.ucl.ac.uk/~amnih/)
# 13. [ Ayse Naz Erkan ](http://www.cs.nyu.edu/~naz/)
# 14. [ Benjamin Schrauwen ](http://reslab.elis.ugent.be/benjamin)
# 15. [ Bernardete Ribeiro ](https://www.cisuc.uc.pt/people/show/2020)
# 16. [ Bo David Chen ](http://vision.caltech.edu/~bchen3/Site/Bo_David_Chen.html)
# 17. [ Boureau Y-Lan ](http://cs.nyu.edu/~ylan/)
# 18. [ Brian Kingsbury ](http://researcher.watson.ibm.com/researcher/view.php?person=us-bedk)
# 19. [ Christopher Manning ](http://nlp.stanford.edu/~manning/)
# 20. [ Clement Farabet ](http://www.clement.farabet.net/)
# 21. [ Dan Claudiu Cireșan ](http://www.idsia.ch/~ciresan/)
# 22. [ David Reichert ](http://serre-lab.clps.brown.edu/person/david-reichert/)
# 23. [ Derek Rose ](http://mil.engr.utk.edu/nmil/member/5.html)
# 24. [ Dong Yu ](http://research.microsoft.com/en-us/people/dongyu/default.aspx)
# 25. [ Drausin Wulsin ](http://www.seas.upenn.edu/~wulsin/)
# 26. [ Erik M. Schmidt ](http://music.ece.drexel.edu/people/eschmidt)
# 27. [ Eugenio Culurciello ](https://engineering.purdue.edu/BME/People/viewPersonById?resource_id=71333)
# 28. [ Frank Seide ](http://research.microsoft.com/en-us/people/fseide/)
# 29. [ Galen Andrew ](http://homes.cs.washington.edu/~galen/)
# 30. [ Geoffrey Hinton ](http://www.cs.toronto.edu/~hinton/)
# 31. [ George Dahl ](http://www.cs.toronto.edu/~gdahl/)
# 32. [ Graham Taylor ](http://www.uoguelph.ca/~gwtaylor/)
# 33. [ Grégoire Montavon ](http://gregoire.montavon.name/)
# 34. [ Guido Francisco Montúfar ](http://personal-homepages.mis.mpg.de/montufar/)
# 35. [ Guillaume Desjardins ](http://brainlogging.wordpress.com/)
# 36. [ Hannes Schulz ](http://www.ais.uni-bonn.de/~schulz/)
# 37. [ Hélène Paugam-Moisy ](http://www.lri.fr/~hpaugam/)
# 38. [ Honglak Lee ](http://web.eecs.umich.edu/~honglak/)
# 39. [ Hugo Larochelle ](http://www.dmi.usherb.ca/~larocheh/index_en.html)
# 40. [ Ilya Sutskever ](http://www.cs.toronto.edu/~ilya/)
# 41. [ Itamar Arel ](http://mil.engr.utk.edu/nmil/member/2.html)
# 42. [ James Martens ](http://www.cs.toronto.edu/~jmartens/)
# 43. [ Jason Morton ](http://www.jasonmorton.com/)
# 44. [ Jason Weston ](http://www.thespermwhale.com/jaseweston/)
# 45. [ Jeff Dean ](http://research.google.com/pubs/jeff.html)
# 46. [ Jiquan Mgiam ](http://cs.stanford.edu/~jngiam/)
# 47. [ Joseph Turian ](http://www-etud.iro.umontreal.ca/~turian/)
# 48. [ Joshua Matthew Susskind ](http://aclab.ca/users/josh/index.html)
# 49. [ Jürgen Schmidhuber ](http://www.idsia.ch/~juergen/)
# 50. [ Justin A. Blanco ](https://sites.google.com/site/blancousna/)
# 51. [ Koray Kavukcuoglu ](http://koray.kavukcuoglu.org/)
# 52. [ KyungHyun Cho ](http://users.ics.aalto.fi/kcho/)
# 53. [ Li Deng ](http://research.microsoft.com/en-us/people/deng/)
# 54. [ Lucas Theis ](http://www.kyb.tuebingen.mpg.de/nc/employee/details/lucas.html)
# 55. [ Ludovic Arnold ](http://ludovicarnold.altervista.org/home/)
# 56. [ Marc'Aurelio Ranzato ](http://www.cs.nyu.edu/~ranzato/)
# 57. [ Martin Längkvist ](http://aass.oru.se/~mlt/)
# 58. [ Misha Denil ](http://mdenil.com/)
# 59. [ Mohammad Norouzi ](http://www.cs.toronto.edu/~norouzi/)
# 60. [ Nando de Freitas ](http://www.cs.ubc.ca/~nando/)
# 61. [ Navdeep Jaitly ](http://www.cs.utoronto.ca/~ndjaitly/)
# 62. [ Nicolas Le Roux ](http://nicolas.le-roux.name/)
# 63. [ Nitish Srivastava ](http://www.cs.toronto.edu/~nitish/)
# 64. [ Noel Lopes ](https://www.cisuc.uc.pt/people/show/2028)
# 65. [ Oriol Vinyals ](http://www.cs.berkeley.edu/~vinyals/)
# 66. [ Pascal Vincent ](http://www.iro.umontreal.ca/~vincentp)
# 67. [ Patrick Nguyen ](https://sites.google.com/site/drpngx/)
# 68. [ Pedro Domingos ](http://homes.cs.washington.edu/~pedrod/)
# 69. [ Peggy Series ](http://homepages.inf.ed.ac.uk/pseries/)
# 70. [ Pierre Sermanet ](http://cs.nyu.edu/~sermanet)
# 71. [ Piotr Mirowski ](http://www.cs.nyu.edu/~mirowski/)
# 72. [ Quoc V. Le ](http://ai.stanford.edu/~quocle/)
# 73. [ Reinhold Scherer ](http://bci.tugraz.at/scherer/)
# 74. [ Richard Socher ](http://www.socher.org/)
# 75. [ Rob Fergus ](http://cs.nyu.edu/~fergus/pmwiki/pmwiki.php)
# 76. [ Robert Coop ](http://mil.engr.utk.edu/nmil/member/19.html)
# 77. [ Robert Gens ](http://homes.cs.washington.edu/~rcg/)
# 78. [ Roger Grosse ](http://people.csail.mit.edu/rgrosse/)
# 79. [ Ronan Collobert ](http://ronan.collobert.com/)
# 80. [ Ruslan Salakhutdinov ](http://www.utstat.toronto.edu/~rsalakhu/)
# 81. [ Sebastian Gerwinn ](http://www.kyb.tuebingen.mpg.de/nc/employee/details/sgerwinn.html)
# 82. [ Stéphane Mallat ](http://www.cmap.polytechnique.fr/~mallat/)
# 83. [ Sven Behnke ](http://www.ais.uni-bonn.de/behnke/)
# 84. [ Tapani Raiko ](http://users.ics.aalto.fi/praiko/)
# 85. [ Tara Sainath ](https://sites.google.com/site/tsainath/)
# 86. [ Tijmen Tieleman ](http://www.cs.toronto.edu/~tijmen/)
# 87. [ Tom Karnowski ](http://mil.engr.utk.edu/nmil/member/36.html)
# 88. [ Tomáš Mikolov ](https://research.facebook.com/tomas-mikolov)
# 89. [ Ueli Meier ](http://www.idsia.ch/~meier/)
# 90. [ Vincent Vanhoucke ](http://vincent.vanhoucke.com)
# 91. [ Volodymyr Mnih ](http://www.cs.toronto.edu/~vmnih/)
# 92. [ Yann LeCun ](http://yann.lecun.com/)
# 93. [ Yichuan Tang ](http://www.cs.toronto.edu/~tang/)
# 94. [ Yoshua Bengio ](http://www.iro.umontreal.ca/~bengioy/yoshua_en/index.html)
# 95. [ Yotaro Kubo ](http://yota.ro/)
# 96. [ Youzhi (Will) Zou ](http://ai.stanford.edu/~wzou)
# 97. [ Fei-Fei Li ](http://vision.stanford.edu/feifeili)
# 98. [ Ian Goodfellow ](https://research.google.com/pubs/105214.html)
# 99. [ Robert Laganière ](http://www.site.uottawa.ca/~laganier/)
# 
# 
# ### WebSites
# 
# 1.  [deeplearning.net](http://deeplearning.net/)
# 2.  [deeplearning.stanford.edu](http://deeplearning.stanford.edu/)
# 3.  [nlp.stanford.edu](http://nlp.stanford.edu/)
# 4.  [ai-junkie.com](http://www.ai-junkie.com/ann/evolved/nnt1.html)
# 5.  [cs.brown.edu/research/ai](http://cs.brown.edu/research/ai/)
# 6.  [eecs.umich.edu/ai](http://www.eecs.umich.edu/ai/)
# 7.  [cs.utexas.edu/users/ai-lab](http://www.cs.utexas.edu/users/ai-lab/)
# 8.  [cs.washington.edu/research/ai](http://www.cs.washington.edu/research/ai/)
# 9.  [aiai.ed.ac.uk](http://www.aiai.ed.ac.uk/)
# 10.  [www-aig.jpl.nasa.gov](http://www-aig.jpl.nasa.gov/)
# 11.  [csail.mit.edu](http://www.csail.mit.edu/)
# 12.  [cgi.cse.unsw.edu.au/~aishare](http://cgi.cse.unsw.edu.au/~aishare/)
# 13.  [cs.rochester.edu/research/ai](http://www.cs.rochester.edu/research/ai/)
# 14.  [ai.sri.com](http://www.ai.sri.com/)
# 15.  [isi.edu/AI/isd.htm](http://www.isi.edu/AI/isd.htm)
# 16.  [nrl.navy.mil/itd/aic](http://www.nrl.navy.mil/itd/aic/)
# 17.  [hips.seas.harvard.edu](http://hips.seas.harvard.edu/)
# 18.  [AI Weekly](http://aiweekly.co)
# 19.  [stat.ucla.edu](http://www.stat.ucla.edu/~junhua.mao/m-RNN.html)
# 20.  [deeplearning.cs.toronto.edu](http://deeplearning.cs.toronto.edu/i2t)
# 21.  [jeffdonahue.com/lrcn/](http://jeffdonahue.com/lrcn/)
# 22.  [visualqa.org](http://www.visualqa.org/)
# 23.  [www.mpi-inf.mpg.de/departments/computer-vision...](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/)
# 24.  [Deep Learning News](http://news.startup.ml/)
# 25.  [Machine Learning is Fun! Adam Geitgey's Blog](https://medium.com/@ageitgey/)
# 26.  [Guide to Machine Learning](http://yerevann.com/a-guide-to-deep-learning/)
# 27.  [Deep Learning for Beginners](https://spandan-madan.github.io/DeepLearningProject/)
# 
# ### Datasets
# 
# 1.  [MNIST](http://yann.lecun.com/exdb/mnist/) Handwritten digits
# 2.  [Google House Numbers](http://ufldl.stanford.edu/housenumbers/) from street view
# 3.  [CIFAR-10 and CIFAR-100](http://www.cs.toronto.edu/~kriz/cifar.html)
# 4.  [IMAGENET](http://www.image-net.org/)
# 5.  [Tiny Images](http://groups.csail.mit.edu/vision/TinyImages/) 80 Million tiny images6.  
# 6.  [Flickr Data](https://yahooresearch.tumblr.com/post/89783581601/one-hundred-million-creative-commons-flickr-images) 100 Million Yahoo dataset
# 7.  [Berkeley Segmentation Dataset 500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
# 8.  [UC Irvine Machine Learning Repository](http://archive.ics.uci.edu/ml/)
# 9.  [Flickr 8k](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html)
# 10. [Flickr 30k](http://shannon.cs.illinois.edu/DenotationGraph/)
# 11. [Microsoft COCO](http://mscoco.org/home/)
# 12. [VQA](http://www.visualqa.org/)
# 13. [Image QA](http://www.cs.toronto.edu/~mren/imageqa/data/cocoqa/)
# 14. [AT&T Laboratories Cambridge face database](http://www.uk.research.att.com/facedatabase.html)
# 15. [AVHRR Pathfinder](http://xtreme.gsfc.nasa.gov)
# 16. [Air Freight](http://www.anc.ed.ac.uk/~amos/afreightdata.html) - The Air Freight data set is a ray-traced image sequence along with ground truth segmentation based on textural characteristics. (455 images + GT, each 160x120 pixels). (Formats: PNG)  
# 17. [Amsterdam Library of Object Images](http://www.science.uva.nl/~aloi/) - ALOI is a color image collection of one-thousand small objects, recorded for scientific purposes. In order to capture the sensory variation in object recordings, we systematically varied viewing angle, illumination angle, and illumination color for each object, and additionally captured wide-baseline stereo images. We recorded over a hundred images of each object, yielding a total of 110,250 images for the collection. (Formats: png)
# 18. [Annotated face, hand, cardiac & meat images](http://www.imm.dtu.dk/~aam/) - Most images & annotations are supplemented by various ASM/AAM analyses using the AAM-API. (Formats: bmp,asf)
# 19. [Image Analysis and Computer Graphics](http://www.imm.dtu.dk/image/)  
# 21. [Brown University Stimuli](http://www.cog.brown.edu/~tarr/stimuli.html) - A variety of datasets including geons, objects, and "greebles". Good for testing recognition algorithms. (Formats: pict)
# 22. [CAVIAR video sequences of mall and public space behavior](http://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/) - 90K video frames in 90 sequences of various human activities, with XML ground truth of detection and behavior classification (Formats: MPEG2 & JPEG)
# 23. [Machine Vision Unit](http://www.ipab.inf.ed.ac.uk/mvu/)
# 25. [CCITT Fax standard images](http://www.cs.waikato.ac.nz/~singlis/ccitt.html) - 8 images (Formats: gif)
# 26. [CMU CIL's Stereo Data with Ground Truth](cil-ster.html) - 3 sets of 11 images, including color tiff images with spectroradiometry (Formats: gif, tiff)
# 27. [CMU PIE Database](http://www.ri.cmu.edu/projects/project_418.html) - A database of 41,368 face images of 68 people captured under 13 poses, 43 illuminations conditions, and with 4 different expressions.
# 28. [CMU VASC Image Database](http://www.ius.cs.cmu.edu/idb/) - Images, sequences, stereo pairs (thousands of images) (Formats: Sun Rasterimage)
# 29. [Caltech Image Database](http://www.vision.caltech.edu/html-files/archive.html) - about 20 images - mostly top-down views of small objects and toys. (Formats: GIF)
# 30. [Columbia-Utrecht Reflectance and Texture Database](http://www.cs.columbia.edu/CAVE/curet/) - Texture and reflectance measurements for over 60 samples of 3D texture, observed with over 200 different combinations of viewing and illumination directions. (Formats: bmp)
# 31. [Computational Colour Constancy Data](http://www.cs.sfu.ca/~colour/data/index.html) - A dataset oriented towards computational color constancy, but useful for computer vision in general. It includes synthetic data, camera sensor data, and over 700 images. (Formats: tiff)
# 32. [Computational Vision Lab](http://www.cs.sfu.ca/~colour/)
# 34. [Content-based image retrieval database](http://www.cs.washington.edu/research/imagedatabase/groundtruth/) - 11 sets of color images for testing algorithms for content-based retrieval. Most sets have a description file with names of objects in each image. (Formats: jpg) 
# 35. [Efficient Content-based Retrieval Group](http://www.cs.washington.edu/research/imagedatabase/)
# 37. [Densely Sampled View Spheres](http://ls7-www.cs.uni-dortmund.de/~peters/pages/research/modeladaptsys/modeladaptsys_vba_rov.html) - Densely sampled view spheres - upper half of the view sphere of two toy objects with 2500 images each. (Formats: tiff)
# 38. [Computer Science VII (Graphical Systems)](http://ls7-www.cs.uni-dortmund.de/)
# 40. [Digital Embryos](https://web-beta.archive.org/web/20011216051535/vision.psych.umn.edu/www/kersten-lab/demos/digitalembryo.html) - Digital embryos are novel objects which may be used to develop and test object recognition systems. They have an organic appearance. (Formats: various formats are available on request)
# 41. [Univerity of Minnesota Vision Lab](http://vision.psych.umn.edu/www/kersten-lab/kersten-lab.html) 
# 42. [El Salvador Atlas of Gastrointestinal VideoEndoscopy](http://www.gastrointestinalatlas.com) - Images and Videos of his-res of studies taken from Gastrointestinal Video endoscopy. (Formats: jpg, mpg, gif)
# 43. [FG-NET Facial Aging Database](http://sting.cycollege.ac.cy/~alanitis/fgnetaging/index.htm) - Database contains 1002 face images showing subjects at different ages. (Formats: jpg)
# 44. [FVC2000 Fingerprint Databases](http://bias.csr.unibo.it/fvc2000/) - FVC2000 is the First International Competition for Fingerprint Verification Algorithms. Four fingerprint databases constitute the FVC2000 benchmark (3520 fingerprints in all).
# 45. [Biometric Systems Lab](http://bias.csr.unibo.it/research/biolab) - University of Bologna
# 46. [Face and Gesture images and image sequences](http://www.fg-net.org) - Several image datasets of faces and gestures that are ground truth annotated for benchmarking
# 47. [German Fingerspelling Database](http://www-i6.informatik.rwth-aachen.de/~dreuw/database.html) - The database contains 35 gestures and consists of 1400 image sequences that contain gestures of 20 different persons recorded under non-uniform daylight lighting conditions. (Formats: mpg,jpg)  
# 48. [Language Processing and Pattern Recognition](http://www-i6.informatik.rwth-aachen.de/)
# 50. [Groningen Natural Image Database](http://hlab.phys.rug.nl/archive.html) - 4000+ 1536x1024 (16 bit) calibrated outdoor images (Formats: homebrew)
# 51. [ICG Testhouse sequence](http://www.icg.tu-graz.ac.at/~schindler/Data) -  2 turntable sequences from ifferent viewing heights, 36 images each, resolution 1000x750, color (Formats: PPM)
# 52. [Institute of Computer Graphics and Vision](http://www.icg.tu-graz.ac.at)
# 54. [IEN Image Library](http://www.ien.it/is/vislib/) - 1000+ images, mostly outdoor sequences (Formats: raw, ppm)  
# 55. [INRIA's Syntim images database](http://www-rocq.inria.fr/~tarel/syntim/images.html) - 15 color image of simple objects (Formats: gif)
# 56. [INRIA](http://www.inria.fr/)
# 57. [INRIA's Syntim stereo databases](http://www-rocq.inria.fr/~tarel/syntim/paires.html) - 34 calibrated color stereo pairs (Formats: gif) 
# 58. [Image Analysis Laboratory](http://www.ece.ncsu.edu/imaging/Archives/ImageDataBase/index.html) - Images obtained from a variety of imaging modalities -- raw CFA images, range images and a host of "medical images". (Formats: homebrew)
# 59. [Image Analysis Laboratory](http://www.ece.ncsu.edu/imaging) 
# 61. [Image Database](http://www.prip.tuwien.ac.at/prip/image.html) - An image database including some textures  
# 62. [JAFFE Facial Expression Image Database](http://www.mis.atr.co.jp/~mlyons/jaffe.html) - The JAFFE database consists of 213 images of Japanese female subjects posing 6 basic facial expressions as well as a neutral pose. Ratings on emotion adjectives are also available, free of charge, for research purposes. (Formats: TIFF Grayscale images.) 
# 63. [ATR Research, Kyoto, Japan](http://www.mic.atr.co.jp/)
# 64. [JISCT Stereo Evaluation](ftp://ftp.vislist.com/IMAGERY/JISCT/) - 44 image pairs. These data have been used in an evaluation of stereo analysis, as described in the April 1993 ARPA Image Understanding Workshop paper ``The JISCT Stereo Evaluation'' by R.C.Bolles, H.H.Baker, and M.J.Hannah, 263--274 (Formats: SSI) 
# 65. [MIT Vision Texture](http://www-white.media.mit.edu/vismod/imagery/VisionTexture/vistex.html) - Image archive (100+ images) (Formats: ppm)
# 66. [MIT face images and more](ftp://whitechapel.media.mit.edu/pub/images) - hundreds of images (Formats: homebrew) 
# 67. [Machine Vision](http://vision.cse.psu.edu/book/testbed/images/) - Images from the textbook by Jain, Kasturi, Schunck (20+ images) (Formats: GIF TIFF)
# 68. [Mammography Image Databases](http://marathon.csee.usf.edu/Mammography/Database.html) - 100 or more images of mammograms with ground truth. Additional images available by request, and links to several other mammography databases are provided. (Formats: homebrew)
# 69. [ftp://ftp.cps.msu.edu/pub/prip](ftp://ftp.cps.msu.edu/pub/prip) - many images (Formats: unknown)
# 70. [Middlebury Stereo Data Sets with Ground Truth](http://www.middlebury.edu/stereo/data.html) - Six multi-frame stereo data sets of scenes containing planar regions. Each data set contains 9 color images and subpixel-accuracy ground-truth data. (Formats: ppm)
# 71. [Middlebury Stereo Vision Research Page](http://www.middlebury.edu/stereo) - Middlebury College
# 72. [Modis Airborne simulator, Gallery and data set](http://ltpwww.gsfc.nasa.gov/MODIS/MAS/) - High Altitude Imagery from around the world for environmental modeling in support of NASA EOS program (Formats: JPG and HDF)
# 73. [NIST Fingerprint and handwriting](ftp://sequoyah.ncsl.nist.gov/pub/databases/data) - datasets - thousands of images (Formats: unknown)
# 74. [NIST Fingerprint data](ftp://ftp.cs.columbia.edu/jpeg/other/uuencoded) - compressed multipart uuencoded tar file 
# 75. [NLM HyperDoc Visible Human Project](http://www.nlm.nih.gov/research/visible/visible_human.html) - Color, CAT and MRI image samples - over 30 images (Formats: jpeg)
# 76. [National Design Repository](http://www.designrepository.org) - Over 55,000 3D CAD and solid models of (mostly) mechanical/machined engineerign designs. (Formats: gif,vrml,wrl,stp,sat) 
# 77. [Geometric & Intelligent Computing Laboratory](http://gicl.mcs.drexel.edu) 
# 79. [OSU (MSU) 3D Object Model Database](http://eewww.eng.ohio-state.edu/~flynn/3DDB/Models/) - several sets of 3D object models collected over several years to use in object recognition research (Formats: homebrew, vrml)
# 80. [OSU (MSU/WSU) Range Image Database](http://eewww.eng.ohio-state.edu/~flynn/3DDB/RID/) - Hundreds of real and synthetic images (Formats: gif, homebrew)
# 81. [OSU/SAMPL Database: Range Images, 3D Models, Stills, Motion Sequences](http://sampl.eng.ohio-state.edu/~sampl/database.htm) - Over 1000 range images, 3D object models, still images and motion sequences (Formats: gif, ppm, vrml, homebrew) 
# 82. [Signal Analysis and Machine Perception Laboratory](http://sampl.eng.ohio-state.edu)
# 84. [Otago Optical Flow Evaluation Sequences](http://www.cs.otago.ac.nz/research/vision/Research/OpticalFlow/opticalflow.html) - Synthetic and real sequences with machine-readable ground truth optical flow fields, plus tools to generate ground truth for new sequences. (Formats: ppm,tif,homebrew)
# 85. [Vision Research Group](http://www.cs.otago.ac.nz/research/vision/index.html) 
# 87. [ftp://ftp.limsi.fr/pub/quenot/opflow/testdata/piv/](ftp://ftp.limsi.fr/pub/quenot/opflow/testdata/piv/) - Real and synthetic image sequences used for testing a Particle Image Velocimetry application. These images may be used for the test of optical flow and image matching algorithms. (Formats: pgm (raw)) 
# 88. [LIMSI-CNRS/CHM/IMM/vision](http://www.limsi.fr/Recherche/IMM/PageIMM.html) 
# 89. [LIMSI-CNRS](http://www.limsi.fr/)
# 90. [Photometric 3D Surface Texture Database](http://www.taurusstudio.net/research/pmtexdb/index.htm) - This is the first 3D texture database which provides both full real surface rotations and registered photometric stereo data (30 textures, 1680 images). (Formats: TIFF) 
# 91. [SEQUENCES FOR OPTICAL FLOW ANALYSIS (SOFA)](http://www.cee.hw.ac.uk/~mtc/sofa) - 9 synthetic sequences designed for testing motion analysis applications, including full ground truth of motion and camera parameters. (Formats: gif)
# 92. [Computer Vision Group](http://www.cee.hw.ac.uk/~mtc/research.html)
# 94. [Sequences for Flow Based Reconstruction](http://www.nada.kth.se/~zucch/CAMERA/PUB/seq.html) - synthetic sequence for testing structure from motion algorithms (Formats: pgm)
# 95. [Stereo Images with Ground Truth Disparity and Occlusion](http://www-dbv.cs.uni-bonn.de/stereo_data/) - a small set of synthetic images of a hallway with varying amounts of noise added. Use these images to benchmark your stereo algorithm. (Formats: raw, viff (khoros), or tiff)
# 96. [Stuttgart Range Image Database](http://range.informatik.uni-stuttgart.de) - A collection of synthetic range images taken from high-resolution polygonal models available on the web (Formats: homebrew)
# 97. [Department Image Understanding](http://www.informatik.uni-stuttgart.de/ipvr/bv/bv_home_engl.html) 
# 99. [The AR Face Database](http://www2.ece.ohio-state.edu/~aleix/ARdatabase.html) - Contains over 4,000 color images corresponding to 126 people's faces (70 men and 56 women). Frontal views with variations in facial expressions, illumination, and occlusions. (Formats: RAW (RGB 24-bit))
# 100. [Purdue Robot Vision Lab](http://rvl.www.ecn.purdue.edu/RVL/)
# 101. [The MIT-CSAIL Database of Objects and Scenes](http://web.mit.edu/torralba/www/database.html) - Database for testing multiclass object detection and scene recognition algorithms. Over 72,000 images with 2873 annotated frames. More than 50 annotated object classes. (Formats: jpg)
# 102. [The RVL SPEC-DB (SPECularity DataBase)](http://rvl1.ecn.purdue.edu/RVL/specularity_database/) - A collection of over 300 real images of 100 objects taken under three different illuminaiton conditions (Diffuse/Ambient/Directed). -- Use these images to test algorithms for detecting and compensating specular highlights in color images. (Formats: TIFF )
# 103. [Robot Vision Laboratory](http://rvl1.ecn.purdue.edu/RVL/)
# 105. [The Xm2vts database](http://xm2vtsdb.ee.surrey.ac.uk) - The XM2VTSDB contains four digital recordings of 295 people taken over a period of four months. This database contains both image and video data of faces.
# 106. [Centre for Vision, Speech and Signal Processing](http://www.ee.surrey.ac.uk/Research/CVSSP) 
# 107. [Traffic Image Sequences and 'Marbled Block' Sequence](http://i21www.ira.uka.de/image_sequences) - thousands of frames of digitized traffic image sequences as well as the 'Marbled Block' sequence (grayscale images) (Formats: GIF)
# 108. [IAKS/KOGS](http://i21www.ira.uka.de) 
# 110. [U Bern Face images](ftp://ftp.iam.unibe.ch/pub/Images/FaceImages) - hundreds of images (Formats: Sun rasterfile)
# 111. [U Michigan textures](ftp://freebie.engin.umich.edu/pub/misc/textures) (Formats: compressed raw)
# 112. [U Oulu wood and knots database](http://www.ee.oulu.fi/~olli/Projects/Lumber.Grading.html) - Includes classifications - 1000+ color images (Formats: ppm) 
# 113. [UCID - an Uncompressed Colour Image Database](http://vision.doc.ntu.ac.uk/datasets/UCID/ucid.html) - a benchmark database for image retrieval with predefined ground truth. (Formats: tiff) 
# 115. [UMass Vision Image Archive](http://vis-www.cs.umass.edu/~vislib/) - Large image database with aerial, space, stereo, medical images and more. (Formats: homebrew)
# 116. [UNC's 3D image database](ftp://sunsite.unc.edu/pub/academic/computer-science/virtual-reality/3d) - many images (Formats: GIF)
# 117. [USF Range Image Data with Segmentation Ground Truth](http://marathon.csee.usf.edu/range/seg-comp/SegComp.html) - 80 image sets (Formats: Sun rasterimage)
# 118. [University of Oulu Physics-based Face Database](http://www.ee.oulu.fi/research/imag/color/pbfd.html) - contains color images of faces under different illuminants and camera calibration conditions as well as skin spectral reflectance measurements of each person.
# 119. [Machine Vision and Media Processing Unit](http://www.ee.oulu.fi/mvmp/)
# 121. [University of Oulu Texture Database](http://www.outex.oulu.fi) - Database of 320 surface textures, each captured under three illuminants, six spatial resolutions and nine rotation angles. A set of test suites is also provided so that texture segmentation, classification, and retrieval algorithms can be tested in a standard manner. (Formats: bmp, ras, xv)
# 122. [Machine Vision Group](http://www.ee.oulu.fi/mvg)
# 124. [Usenix face database](ftp://ftp.uu.net/published/usenix/faces) - Thousands of face images from many different sites (circa 994)
# 125. [View Sphere Database](http://www-prima.inrialpes.fr/Prima/hall/view_sphere.html) - Images of 8 objects seen from many different view points. The view sphere is sampled using a geodesic with 172 images/sphere. Two sets for training and testing are available. (Formats: ppm)
# 126. [PRIMA, GRAVIR](http://www-prima.inrialpes.fr/Prima/)
# 127. [Vision-list Imagery Archive](ftp://ftp.vislist.com/IMAGERY/) - Many images, many formats
# 128. [Wiry Object Recognition Database](http://www.cs.cmu.edu/~owenc/word.htm) - Thousands of images of a cart, ladder, stool, bicycle, chairs, and cluttered scenes with ground truth labelings of edges and regions. (Formats: jpg)
# 129. [3D Vision Group](http://www.cs.cmu.edu/0.000000E+003dvision/)
# 131. [Yale Face Database](http://cvc.yale.edu/projects/yalefaces/yalefaces.html) -  165 images (15 individuals) with different lighting, expression, and occlusion configurations.
# 132. [Yale Face Database B](http://cvc.yale.edu/projects/yalefacesB/yalefacesB.html) - 5760 single light source images of 10 subjects each seen under 576 viewing conditions (9 poses x 64 illumination conditions). (Formats: PGM) 
# 133. [Center for Computational Vision and Control](http://cvc.yale.edu/)
# 134. [DeepMind QA Corpus](https://github.com/deepmind/rc-data) - Textual QA corpus from CNN and DailyMail. More than 300K documents in total. [Paper](http://arxiv.org/abs/1506.03340) for reference.
# 135. [YouTube-8M Dataset](https://research.google.com/youtube8m/) - YouTube-8M is a large-scale labeled video dataset that consists of 8 million YouTube video IDs and associated labels from a diverse vocabulary of 4800 visual entities.
# 136. [Open Images dataset](https://github.com/openimages/dataset) - Open Images is a dataset of ~9 million URLs to images that have been annotated with labels spanning over 6000 categories.
# 137. [Visual Object Classes Challenge 2012 (VOC2012)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit) - VOC2012 dataset containing 12k images with 20 annotated classes for object detection and segmentation.
# 138. [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) - MNIST like fashion product dataset consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. 
# 139. [Large-scale Fashion (DeepFashion) Database](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) - Contains over 800,000 diverse fashion images.  Each image in this dataset is labeled with 50 categories, 1,000 descriptive attributes, bounding box and clothing landmarks
# 140. [FakeNewsCorpus](https://github.com/several27/FakeNewsCorpus) - Contains about 10 million news articles classified using [opensources.co](http://opensources.co) types
# 
# ### Conferences
# 
# 1. [CVPR - IEEE Conference on Computer Vision and Pattern Recognition](http://cvpr2018.thecvf.com)
# 2. [AAMAS - International Joint Conference on Autonomous Agents and Multiagent Systems](http://celweb.vuse.vanderbilt.edu/aamas18/)
# 3. [IJCAI - 	International Joint Conference on Artificial Intelligence](https://www.ijcai-18.org/)
# 4. [ICML - 	International Conference on Machine Learning](https://icml.cc)
# 5. [ECML - European Conference on Machine Learning](http://www.ecmlpkdd2018.org)
# 6. [KDD - Knowledge Discovery and Data Mining](http://www.kdd.org/kdd2018/)
# 7. [NIPS - Neural Information Processing Systems](https://nips.cc/Conferences/2018)
# 8. [O'Reilly AI Conference - 	O'Reilly Artificial Intelligence Conference](https://conferences.oreilly.com/artificial-intelligence/ai-ny)
# 9. [ICDM - International Conference on Data Mining](https://www.waset.org/conference/2018/07/istanbul/ICDM)
# 10. [ICCV - International Conference on Computer Vision](http://iccv2017.thecvf.com)
# 11. [AAAI - Association for the Advancement of Artificial Intelligence](https://www.aaai.org)
# 
# ### Frameworks
# 
# 1.  [Caffe](http://caffe.berkeleyvision.org/)  
# 2.  [Torch7](http://torch.ch/)
# 3.  [Theano](http://deeplearning.net/software/theano/)
# 4.  [cuda-convnet](https://code.google.com/p/cuda-convnet2/)
# 5.  [convetjs](https://github.com/karpathy/convnetjs)
# 5.  [Ccv](http://libccv.org/doc/doc-convnet/)
# 6.  [NuPIC](http://numenta.org/nupic.html)
# 7.  [DeepLearning4J](http://deeplearning4j.org/)
# 8.  [Brain](https://github.com/harthur/brain)
# 9.  [DeepLearnToolbox](https://github.com/rasmusbergpalm/DeepLearnToolbox)
# 10.  [Deepnet](https://github.com/nitishsrivastava/deepnet)
# 11.  [Deeppy](https://github.com/andersbll/deeppy)
# 12.  [JavaNN](https://github.com/ivan-vasilev/neuralnetworks)
# 13.  [hebel](https://github.com/hannes-brt/hebel)
# 14.  [Mocha.jl](https://github.com/pluskid/Mocha.jl)
# 15.  [OpenDL](https://github.com/guoding83128/OpenDL)
# 16.  [cuDNN](https://developer.nvidia.com/cuDNN)
# 17.  [MGL](http://melisgl.github.io/mgl-pax-world/mgl-manual.html)
# 18.  [Knet.jl](https://github.com/denizyuret/Knet.jl)
# 19.  [Nvidia DIGITS - a web app based on Caffe](https://github.com/NVIDIA/DIGITS)
# 20.  [Neon - Python based Deep Learning Framework](https://github.com/NervanaSystems/neon)
# 21.  [Keras - Theano based Deep Learning Library](http://keras.io)
# 22.  [Chainer - A flexible framework of neural networks for deep learning](http://chainer.org/)
# 23.  [RNNLM Toolkit](http://rnnlm.org/)
# 24.  [RNNLIB - A recurrent neural network library](http://sourceforge.net/p/rnnl/wiki/Home/)
# 25.  [char-rnn](https://github.com/karpathy/char-rnn)
# 26.  [MatConvNet: CNNs for MATLAB](https://github.com/vlfeat/matconvnet)
# 27.  [Minerva - a fast and flexible tool for deep learning on multi-GPU](https://github.com/dmlc/minerva)
# 28.  [Brainstorm - Fast, flexible and fun neural networks.](https://github.com/IDSIA/brainstorm)
# 29.  [Tensorflow - Open source software library for numerical computation using data flow graphs](https://github.com/tensorflow/tensorflow)
# 30.  [DMTK - Microsoft Distributed Machine Learning Tookit](https://github.com/Microsoft/DMTK)
# 31.  [Scikit Flow - Simplified interface for TensorFlow (mimicking Scikit Learn)](https://github.com/google/skflow)
# 32.  [MXnet - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning framework](https://github.com/dmlc/mxnet/)
# 33.  [Veles - Samsung Distributed machine learning platform](https://github.com/Samsung/veles)
# 34.  [Marvin - A Minimalist GPU-only N-Dimensional ConvNets Framework](https://github.com/PrincetonVision/marvin)
# 35.  [Apache SINGA - A General Distributed Deep Learning Platform](http://singa.incubator.apache.org/)
# 36.  [DSSTNE - Amazon's library for building Deep Learning models](https://github.com/amznlabs/amazon-dsstne)
# 37.  [SyntaxNet - Google's syntactic parser - A TensorFlow dependency library](https://github.com/tensorflow/models/tree/master/syntaxnet)
# 38.  [mlpack - A scalable Machine Learning library](http://mlpack.org/)
# 39.  [Torchnet - Torch based Deep Learning Library](https://github.com/torchnet/torchnet)
# 40.  [Paddle - PArallel Distributed Deep LEarning by Baidu](https://github.com/baidu/paddle)
# 41.  [NeuPy - Theano based Python library for ANN and Deep Learning](http://neupy.com)
# 42.  [Lasagne - a lightweight library to build and train neural networks in Theano](https://github.com/Lasagne/Lasagne)
# 43.  [nolearn - wrappers and abstractions around existing neural network libraries, most notably Lasagne](https://github.com/dnouri/nolearn)
# 44.  [Sonnet - a library for constructing neural networks by Google's DeepMind](https://github.com/deepmind/sonnet)
# 45.  [PyTorch - Tensors and Dynamic neural networks in Python with strong GPU acceleration](https://github.com/pytorch/pytorch)
# 46.  [CNTK - Microsoft Cognitive Toolkit](https://github.com/Microsoft/CNTK)
# 47.  [Serpent.AI - Game agent framework: Use any video game as a deep learning sandbox](https://github.com/SerpentAI/SerpentAI)
# 48.  [Caffe2 - A New Lightweight, Modular, and Scalable Deep Learning Framework](https://github.com/caffe2/caffe2)
# 49.  [deeplearn.js - Hardware-accelerated deep learning and linear algebra (NumPy) library for the web](https://github.com/PAIR-code/deeplearnjs)
# 50.  [TensorForce - A TensorFlow library for applied reinforcement learning](https://github.com/reinforceio/tensorforce)
# 51.  [Coach - Reinforcement Learning Coach by Intel® AI Lab](https://github.com/NervanaSystems/coach)
# 52.  [albumentations - A fast and framework agnostic image augmentation library](https://github.com/albu/albumentations)
# 
# ### Tools
# 
# 1.  [Netron](https://github.com/lutzroeder/netron) - Visualizer for deep learning and machine learning models
# 2.  [Jupyter Notebook](http://jupyter.org) - Web-based notebook environment for interactive computing
# 3.  [TensorBoard](https://github.com/tensorflow/tensorboard) - TensorFlow's Visualization Toolkit
# 4.  [Visual Studio Tools for AI](https://visualstudio.microsoft.com/downloads/ai-tools-vs) - Develop, debug and deploy deep learning and AI solutions
# 
# ### Miscellaneous
# 
# 1.  [Google Plus - Deep Learning Community](https://plus.google.com/communities/112866381580457264725)
# 2.  [Caffe Webinar](http://on-demand-gtc.gputechconf.com/gtcnew/on-demand-gtc.php?searchByKeyword=shelhamer&amp;searchItems=&amp;sessionTopic=&amp;sessionEvent=4&amp;sessionYear=2014&amp;sessionFormat=&amp;submit=&amp;select=+)
# 3.  [100 Best Github Resources in Github for DL](http://meta-guide.com/software-meta-guide/100-best-github-deep-learning/)
# 4.  [Word2Vec](https://code.google.com/p/word2vec/)
# 5.  [Caffe DockerFile](https://github.com/tleyden/docker/tree/master/caffe)
# 6.  [TorontoDeepLEarning convnet](https://github.com/TorontoDeepLearning/convnet)
# 8.  [gfx.js](https://github.com/clementfarabet/gfx.js)
# 9.  [Torch7 Cheat sheet](https://github.com/torch/torch7/wiki/Cheatsheet)
# 10. [Misc from MIT's 'Advanced Natural Language Processing' course](http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-864-advanced-natural-language-processing-fall-2005/)
# 11. [Misc from MIT's 'Machine Learning' course](http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-867-machine-learning-fall-2006/lecture-notes/)
# 12. [Misc from MIT's 'Networks for Learning: Regression and Classification' course](http://ocw.mit.edu/courses/brain-and-cognitive-sciences/9-520-a-networks-for-learning-regression-and-classification-spring-2001/)
# 13. [Misc from MIT's 'Neural Coding and Perception of Sound' course](http://ocw.mit.edu/courses/health-sciences-and-technology/hst-723j-neural-coding-and-perception-of-sound-spring-2005/index.htm)
# 14. [Implementing a Distributed Deep Learning Network over Spark](http://www.datasciencecentral.com/profiles/blogs/implementing-a-distributed-deep-learning-network-over-spark)
# 15. [A chess AI that learns to play chess using deep learning.](https://github.com/erikbern/deep-pink)
# 16. [Reproducing the results of "Playing Atari with Deep Reinforcement Learning" by DeepMind](https://github.com/kristjankorjus/Replicating-DeepMind)
# 17. [Wiki2Vec. Getting Word2vec vectors for entities and word from Wikipedia Dumps](https://github.com/idio/wiki2vec)
# 18. [The original code from the DeepMind article + tweaks](https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner)
# 19. [Google deepdream - Neural Network art](https://github.com/google/deepdream)
# 20. [An efficient, batched LSTM.](https://gist.github.com/karpathy/587454dc0146a6ae21fc)
# 21. [A recurrent neural network designed to generate classical music.](https://github.com/hexahedria/biaxial-rnn-music-composition)
# 22. [Memory Networks Implementations - Facebook](https://github.com/facebook/MemNN)
# 23. [Face recognition with Google's FaceNet deep neural network.](https://github.com/cmusatyalab/openface)
# 24. [Basic digit recognition neural network](https://github.com/joeledenberg/DigitRecognition)
# 25. [Emotion Recognition API Demo - Microsoft](https://www.projectoxford.ai/demo/emotion#detection)
# 26. [Proof of concept for loading Caffe models in TensorFlow](https://github.com/ethereon/caffe-tensorflow)
# 27. [YOLO: Real-Time Object Detection](http://pjreddie.com/darknet/yolo/#webcam)
# 28. [AlphaGo - A replication of DeepMind's 2016 Nature publication, "Mastering the game of Go with deep neural networks and tree search"](https://github.com/Rochester-NRT/AlphaGo)
# 29. [Machine Learning for Software Engineers](https://github.com/ZuzooVn/machine-learning-for-software-engineers)
# 30. [Machine Learning is Fun!](https://medium.com/@ageitgey/machine-learning-is-fun-80ea3ec3c471#.oa4rzez3g)
# 31. [Siraj Raval's Deep Learning tutorials](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A)
# 32. [Dockerface](https://github.com/natanielruiz/dockerface) - Easy to install and use deep learning Faster R-CNN face detection for images and video in a docker container.
# 33. [Awesome Deep Learning Music](https://github.com/ybayle/awesome-deep-learning-music) - Curated list of articles related to deep learning scientific research applied to music
# 34. [Awesome Graph Embedding](https://github.com/benedekrozemberczki/awesome-graph-embedding) - Curated list of articles related to deep learning scientific research on graph structured data
# 
# 
# ## Awesome Deep Learning - Part II
# 
# This is a rough list of my favorite deep learning resources. It has been useful to me for learning how to do deep learning, I use it for revisiting topics or for reference.
# 
# ## Table of Content - Part II
# 
# - [Trends](#trends)
# - [Online classes](#online-classes)
# - [Books](#books)
# - [Posts and Articles](#posts-and-articles)
# - [Practical resources](#practical-resources)
#   - [Librairies and Implementations](#librairies-and-implementations)
#   - [Some Datasets](#some-datasets)
# - [Other Math Theory](#other-math-theory)
#   - [Gradient Descent Algorithms and optimization](#gradient-descent-algorithms-and-optimization)
#   - [Complex Numbers & Digital Signal Processing](#complex-numbers-and-digital-signal-processing)
# - [Papers](#papers)
#   - [Recurrent Neural Networks](#recurrent-neural-networks)
#   - [Convolutional Neural Networks](#convolutional-neural-networks)
#   - [Attention Mechanisms](#attention-mechanisms)
# - [YouTube and Videos](#youtube)
# - [Misc. Hubs and Links](#misc-hubs-and-links)
# - [License](#license)
# 
# <a name="trends" />
# 
# ## Trends
# 
# Here are the all-time [Google Trends](https://www.google.ca/trends/explore?date=all&q=machine%20learning,deep%20learning,data%20science,computer%20programming), from 2004 up to now, September 2017:
# <p align="center">
#   <img src="https://raw.githubusercontent.com/guillaume-chevalier/Awesome-Deep-Learning-Resources/master/google_trends.png" width="792" height="424" />
# </p>
# 
# You might also want to look at Andrej Karpathy's [new post](https://medium.com/@karpathy/a-peek-at-trends-in-machine-learning-ab8a1085a106) about trends in Machine Learning research.
# 
# I believe that Deep learning is the key to make computers think more like humans, and has a lot of potential. Some hard automation tasks can be solved easily with that while this was impossible to achieve earlier with classical algorithms.
# 
# Moore's Law about exponential progress rates in computer science hardware is now more affecting GPUs than CPUs because of physical limits on how tiny an atomic transistor can be. We are shifting toward parallel architectures
# [[read more](https://www.quora.com/Does-Moores-law-apply-to-GPUs-Or-only-CPUs)]. Deep learning exploits parallel architectures as such under the hood by using GPUs. On top of that, deep learning algorithms may use Quantum Computing and apply to machine-brain interfaces in the future.
# 
# I find that the key of intelligence and cognition is a very interesting subject to explore and is not yet well understood. Those technologies are promising.
# 
# 
# <a name="online-classes" />
# 
# ## Online Classes
# 
# - [Machine Learning by Andrew Ng on Coursera](https://www.coursera.org/learn/machine-learning) - Renown entry-level online class with [certificate](https://www.coursera.org/account/accomplishments/verify/DXPXHYFNGKG3). Taught by: Andrew Ng, Associate Professor, Stanford University; Chief Scientist, Baidu; Chairman and Co-founder, Coursera.
# - [Deep Learning Specialization by Andrew Ng on Coursera](https://www.coursera.org/specializations/deep-learning) - New series of 5 Deep Learning courses by Andrew Ng, now with Python rather than Matlab/Octave, and which leads to a [specialization certificate](https://www.coursera.org/account/accomplishments/specialization/U7VNC3ZD9YD8).
# - [Deep Learning by Google](https://www.udacity.com/course/deep-learning--ud730) - Good intermediate to advanced-level course covering high-level deep learning concepts, I found it helps to get creative once the basics are acquired.
# - [Machine Learning for Trading by Georgia Tech](https://www.udacity.com/course/machine-learning-for-trading--ud501) - Interesting class for acquiring basic knowledge of machine learning applied to trading and some AI and finance concepts. I especially liked the section on Q-Learning.
# - [Neural networks class by Hugo Larochelle, Université de Sherbrooke](https://www.youtube.com/playlist?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH) - Interesting class about neural networks available online for free by Hugo Larochelle, yet I have watched a few of those videos.
# - [GLO-4030/7030 Apprentissage par réseaux de neurones profonds](https://ulaval-damas.github.io/glo4030/) - This is a class given by Philippe Giguère, Professor at University Laval. I especially found awesome its rare visualization of the multi-head attention mechanism, which can be contemplated at the [slide 28 of week 13's class](http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/09-Attention.pdf).
# 
# <a name="books" />
# 
# ## Books
# 
# - [How to Create a Mind](https://www.amazon.com/How-Create-Mind-Thought-Revealed/dp/B009VSFXZ4) - The audio version is nice to listen to while commuting. This book is motivating about reverse-engineering the mind and thinking on how to code AI.
# - [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) - This book covers many of the core concepts behind neural networks and deep learning.
# - [Deep Learning - An MIT Press book](http://www.deeplearningbook.org/) - Yet halfway through the book, it contains satisfying math content on how to think about actual deep learning.
# - [Some other books I have read](https://books.google.ca/books?hl=en&as_coll=4&num=10&uid=103409002069648430166&source=gbs_slider_cls_metadata_4_mylibrary_title) - Some books listed here are less related to deep learning but are still somehow relevant to this list.
# 
# <a name="posts-and-articles" />
# 
# ## Posts and Articles
# 
# - [Predictions made by Ray Kurzweil](https://en.wikipedia.org/wiki/Predictions_made_by_Ray_Kurzweil) - List of mid to long term futuristic predictions made by Ray Kurzweil.
# - [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) - MUST READ post by Andrej Karpathy - this is what motivated me to learn RNNs, it demonstrates what it can achieve in the most basic form of NLP.
# - [Neural Networks, Manifolds, and Topology](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) - Fresh look on how neurons map information.
# - [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Explains the LSTM cells' inner workings, plus, it has interesting links in conclusion.
# - [Attention and Augmented Recurrent Neural Networks](http://distill.pub/2016/augmented-rnns/) - Interesting for visual animations, it is a nice intro to attention mechanisms as an example.
# - [Recommending music on Spotify with deep learning](http://benanne.github.io/2014/08/05/spotify-cnns.html) - Awesome for doing clustering on audio - post by an intern at Spotify.
# - [Announcing SyntaxNet: The World’s Most Accurate Parser Goes Open Source](https://research.googleblog.com/2016/05/announcing-syntaxnet-worlds-most.html) - Parsey McParseface's birth, a neural syntax tree parser.
# - [Improving Inception and Image Classification in TensorFlow](https://research.googleblog.com/2016/08/improving-inception-and-image.html) - Very interesting CNN architecture (e.g.: the inception-style convolutional layers is promising and efficient in terms of reducing the number of parameters).
# - [WaveNet: A Generative Model for Raw Audio](https://deepmind.com/blog/wavenet-generative-model-raw-audio/) - Realistic talking machines: perfect voice generation.
# - [François Chollet's Twitter](https://twitter.com/fchollet) - Author of Keras - has interesting Twitter posts and innovative ideas.
# - [Neuralink and the Brain’s Magical Future](http://waitbutwhy.com/2017/04/neuralink.html) - Thought provoking article about the future of the brain and brain-computer interfaces.
# - [Migrating to Git LFS for Developing Deep Learning Applications with Large Files](http://vooban.com/en/tips-articles-geek-stuff/migrating-to-git-lfs-for-developing-deep-learning-applications-with-large-files/) - Easily manage huge files in your private Git projects.
# - [The future of deep learning](https://blog.keras.io/the-future-of-deep-learning.html) - François Chollet's thoughts on the future of deep learning.
# - [Discover structure behind data with decision trees](http://vooban.com/en/tips-articles-geek-stuff/discover-structure-behind-data-with-decision-trees/) - Grow decision trees and visualize them, infer the hidden logic behind data.
# - [Hyperopt tutorial for Optimizing Neural Networks’ Hyperparameters](http://vooban.com/en/tips-articles-geek-stuff/hyperopt-tutorial-for-optimizing-neural-networks-hyperparameters/) - Learn to slay down hyperparameter spaces automatically rather than by hand.
# - [Estimating an Optimal Learning Rate For a Deep Neural Network](https://medium.com/@surmenok/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0) - Clever trick to estimate an optimal learning rate prior any single full training.
# 
# <a name="practical-resources" />
# 
# ## Practical Resources
# 
# <a name="librairies-and-implementations" />
# 
# ### Librairies and Implementations
# - [TensorFlow's GitHub repository](https://github.com/tensorflow/tensorflow) - Most known deep learning framework, both high-level and low-level while staying flexible.
# - [skflow](https://github.com/tensorflow/skflow) - TensorFlow wrapper à la scikit-learn.
# - [Keras](https://keras.io/) - Keras is another intersting deep learning framework like TensorFlow, it is mostly high-level.
# - [carpedm20's repositories](https://github.com/carpedm20) - Many interesting neural network architectures are implemented by the Korean guy Taehoon Kim, A.K.A. carpedm20.
# - [carpedm20/NTM-tensorflow](https://github.com/carpedm20/NTM-tensorflow) - Neural Turing Machine TensorFlow implementation.
# - [Deep learning for lazybones](http://oduerr.github.io/blog/2016/04/06/Deep-Learning_for_lazybones) - Transfer learning tutorial in TensorFlow for vision from high-level embeddings of a pretrained CNN, AlexNet 2012.
# - [LSTM for Human Activity Recognition (HAR)](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition) - Tutorial of mine on using LSTMs on time series for classification.
# - [Deep stacked residual bidirectional LSTMs for HAR](https://github.com/guillaume-chevalier/HAR-stacked-residual-bidir-LSTMs) - Improvements on the previous project.
# - [Sequence to Sequence (seq2seq) Recurrent Neural Network (RNN) for Time Series Prediction](https://github.com/guillaume-chevalier/seq2seq-signal-prediction) - Tutorial of mine on how to predict temporal sequences of numbers - that may be multichannel.
# - [Hyperopt for a Keras CNN on CIFAR-100](https://github.com/Vooban/Hyperopt-Keras-CNN-CIFAR-100) - Auto (meta) optimizing a neural net (and its architecture) on the CIFAR-100 dataset.
# - [ML / DL repositories I starred](https://github.com/guillaume-chevalier?direction=desc&page=1&q=machine+OR+deep+OR+learning+OR+rnn+OR+lstm+OR+cnn&sort=stars&tab=stars&utf8=%E2%9C%93) - GitHub is full of nice code samples & projects.
# - [Smoothly Blend Image Patches](https://github.com/Vooban/Smoothly-Blend-Image-Patches) - Smooth patch merger for [semantic segmentation with a U-Net](https://vooban.com/en/tips-articles-geek-stuff/satellite-image-segmentation-workflow-with-u-net/).
# 
# <a name="some-datasets" />
# 
# ### Some Datasets
# 
# Those are resources I have found that seems interesting to develop models onto.
# 
# - [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.html) - TONS of datasets for ML.
# - [Cornell Movie--Dialogs Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) - This could be used for a chatbot.
# - [SQuAD The Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/) - Question answering dataset that can be explored online, and a list of models performing well on that dataset.
# - [LibriSpeech ASR corpus](http://www.openslr.org/12/) - Huge free English speech dataset with balanced genders and speakers, that seems to be of high quality.
# - [Awesome Public Datasets](https://github.com/caesar0301/awesome-public-datasets) - An awesome list of public datasets.
# 
# 
# <a name="other-math-theory" />
# 
# ## Other Math Theory
# 
# <a name="gradient-descent-algorithms-and-optimization" />
# 
# ### Gradient Descent Algorithms & Optimization Theory
# 
# - [Neural Networks and Deep Learning, ch.2](http://neuralnetworksanddeeplearning.com/chap2.html) - Overview on how does the backpropagation algorithm works.
# - [Neural Networks and Deep Learning, ch.4](http://neuralnetworksanddeeplearning.com/chap4.html) - A visual proof that neural nets can compute any function.
# - [Yes you should understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b#.mr5wq61fb) - Exposing backprop's caveats and the importance of knowing that while training models.
# - [Artificial Neural Networks: Mathematics of Backpropagation](http://briandolhansky.com/blog/2013/9/27/artificial-neural-networks-backpropagation-part-4) - Picturing backprop, mathematically.
# - [Deep Learning Lecture 12: Recurrent Neural Nets and LSTMs](https://www.youtube.com/watch?v=56TYLaQN4N8) - Unfolding of RNN graphs is explained properly, and potential problems about gradient descent algorithms are exposed.
# - [Gradient descent algorithms in a saddle point](http://sebastianruder.com/content/images/2016/09/saddle_point_evaluation_optimizers.gif) - Visualize how different optimizers interacts with a saddle points.
# - [Gradient descent algorithms in an almost flat landscape](https://devblogs.nvidia.com/wp-content/uploads/2015/12/NKsFHJb.gif) - Visualize how different optimizers interacts with an almost flat landscape.
# - [Gradient Descent](https://www.youtube.com/watch?v=F6GSRDoB-Cg) - Okay, I already listed Andrew NG's Coursera class above, but this video especially is quite pertinent as an introduction and defines the gradient descent algorithm.
# - [Gradient Descent: Intuition](https://www.youtube.com/watch?v=YovTqTY-PYY) - What follows from the previous video: now add intuition.
# - [Gradient Descent in Practice 2: Learning Rate](https://www.youtube.com/watch?v=gX6fZHgfrow) - How to adjust the learning rate of a neural network.
# - [The Problem of Overfitting](https://www.youtube.com/watch?v=u73PU6Qwl1I) - A good explanation of overfitting and how to address that problem.
# - [Diagnosing Bias vs Variance](https://www.youtube.com/watch?v=ewogYw5oCAI) - Understanding bias and variance in the predictions of a neural net and how to address those problems.
# - [Self-Normalizing Neural Networks](https://arxiv.org/pdf/1706.02515.pdf) - Appearance of the incredible SELU activation function.
# - [Learning to learn by gradient descent by gradient descent](https://arxiv.org/pdf/1606.04474.pdf) - RNN as an optimizer: introducing the L2L optimizer, a meta-neural network.
# 
# <a name="complex-numbers-and-digital-signal-processing" />
# 
# ### Complex Numbers & Digital Signal Processing
# 
# Okay, signal processing might not be directly related to deep learning, but studying it is interesting to have more intuition in developing neural architectures based on signal.
# 
# - [Window Functions](https://en.wikipedia.org/wiki/Window_function) - Wikipedia page that lists some of the known window functions.
# - [MathBox, Tools for Thought Graphical Algebra and Fourier Analysis](https://acko.net/files/gltalks/toolsforthought/) - New look on Fourier analysis.
# - [How to Fold a Julia Fractal](http://acko.net/blog/how-to-fold-a-julia-fractal/) - Animations dealing with complex numbers and wave equations.
# - [Animate Your Way to Glory, Math and Physics in Motion](http://acko.net/blog/animate-your-way-to-glory/) - Convergence methods in physic engines, and applied to interaction design.
# - [Animate Your Way to Glory - Part II, Math and Physics in Motion](http://acko.net/blog/animate-your-way-to-glory-pt2/) - Nice animations for rotation and rotation interpolation with Quaternions, a mathematical object for handling 3D rotations.
# - [Filtering signal, plotting the STFT and the Laplace transform](https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform) - Simple Python demo on signal processing.
# 
# 
# <a name="papers" />
# 
# ## Papers
# 
# <a name="recurrent-neural-networks" />
# 
# ### Recurrent Neural Networks
# 
# - [Deep Learning in Neural Networks: An Overview](https://arxiv.org/pdf/1404.7828v4.pdf) - You_Again's summary/overview of deep learning, mostly about RNNs.
# - [Bidirectional Recurrent Neural Networks](http://www.di.ufpe.br/~fnj/RNA/bibliografia/BRNN.pdf) - Better classifications with RNNs with bidirectional scanning on the time axis.
# - [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078v3.pdf) - Two networks in one combined into a seq2seq (sequence to sequence) Encoder-Decoder architecture. RNN Encoder–Decoder with 1000 hidden units. Adadelta optimizer.
# - [Sequence to Sequence Learning with Neural Networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) - 4 stacked LSTM cells of 1000 hidden size with reversed input sentences, and with beam search, on the WMT’14 English to French dataset.
# - [Exploring the Limits of Language Modeling](https://arxiv.org/pdf/1602.02410.pdf) - Nice recursive models using word-level LSTMs on top of a character-level CNN using an overkill amount of GPU power.
# - [Neural Machine Translation and Sequence-to-sequence Models: A Tutorial](https://arxiv.org/pdf/1703.01619.pdf) - Interesting overview of the subject of NMT, I mostly read part 8 about RNNs with attention as a refresher.
# - [Exploring the Depths of Recurrent Neural Networks with Stochastic Residual Learning](https://cs224d.stanford.edu/reports/PradhanLongpre.pdf) - Basically, residual connections can be better than stacked RNNs in the presented case of sentiment analysis.
# - [Pixel Recurrent Neural Networks](https://arxiv.org/pdf/1601.06759.pdf) - Nice for photoshop-like "content aware fill" to fill missing patches in images.
# - [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/pdf/1603.08983v4.pdf) - Let RNNs decide how long they compute. I would love to see how well would it combines to Neural Turing Machines. Interesting interactive visualizations on the subject can be found [here](http://distill.pub/2016/augmented-rnns/).
# 
# 
# <a name="convolutional-neural-networks" />
# 
# ### Convolutional Neural Networks
# 
# - [What is the Best Multi-Stage Architecture for Object Recognition?](http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf) - Awesome for the use of "local contrast normalization".
# - [ImageNet Classification with Deep Convolutional Neural Networks](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf) - AlexNet, 2012 ILSVRC, breakthrough of the ReLU activation function.
# - [Visualizing and Understanding Convolutional Networks](https://arxiv.org/pdf/1311.2901v3.pdf) - For the "deconvnet layer".
# - [Fast and Accurate Deep Network Learning by Exponential Linear Units](https://arxiv.org/pdf/1511.07289v1.pdf) - ELU activation function for CIFAR vision tasks.
# - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556v6.pdf) - Interesting idea of stacking multiple 3x3 conv+ReLU before pooling for a bigger filter size with just a few parameters. There is also a nice table for "ConvNet Configuration".
# - [Going Deeper with Convolutions](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) - GoogLeNet: Appearance of "Inception" layers/modules, the idea is of parallelizing conv layers into many mini-conv of different size with "same" padding, concatenated on depth.
# - [Highway Networks](https://arxiv.org/pdf/1505.00387v2.pdf) - Highway networks: residual connections.
# - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167v3.pdf) - Batch normalization (BN): to normalize a layer's output by also summing over the entire batch, and then performing a linear rescaling and shifting of a certain trainable amount.
# - [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf) - The U-Net is an encoder-decoder CNN that also has skip-connections, good for image segmentation at a per-pixel level.
# - [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385v1.pdf) - Very deep residual layers with batch normalization layers - a.k.a. "how to overfit any vision dataset with too many layers and make any vision model work properly at recognition given enough data".
# - [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261v2.pdf) - For improving GoogLeNet with residual connections.
# - [WaveNet: a Generative Model for Raw Audio](https://arxiv.org/pdf/1609.03499v2.pdf) - Epic raw voice/music generation with new architectures based on dilated causal convolutions to capture more audio length.
# - [Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling](https://arxiv.org/pdf/1610.07584v2.pdf) - 3D-GANs for 3D model generation and fun 3D furniture arithmetics from embeddings (think like word2vec word arithmetics with 3D furniture representations).
# - [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://research.fb.com/publications/ImageNet1kIn1h/) - Incredibly fast distributed training of a CNN.
# - [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf) - Best Paper Award at CVPR 2017, yielding improvements on state-of-the-art performances on CIFAR-10, CIFAR-100 and SVHN datasets, this new neural network architecture is named DenseNet.
# - [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326.pdf) - Merges the ideas of the U-Net and the DenseNet, this new neural network is especially good for huge datasets in image segmentation.
# - [Prototypical Networks for Few-shot Learning](https://arxiv.org/pdf/1703.05175.pdf) - Use a distance metric in the loss to determine to which class does an object belongs to from a few examples.
# 
# 
# <a name="attention-mechanisms" />
# 
# ### Attention Mechanisms
# 
# - [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf) - Attention mechanism for LSTMs! Mostly, figures and formulas and their explanations revealed to be useful to me. I gave a talk on that paper [here](https://www.youtube.com/watch?v=QuvRWevJMZ4).
# - [Neural Turing Machines](https://arxiv.org/pdf/1410.5401v2.pdf) - Outstanding for letting a neural network learn an algorithm with seemingly good generalization over long time dependencies. Sequences recall problem.
# - [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf) - LSTMs' attention mechanisms on CNNs feature maps does wonders.
# - [Teaching Machines to Read and Comprehend](https://arxiv.org/pdf/1506.03340v3.pdf) - A very interesting and creative work about textual question answering, what a breakthrough, there is something to do with that.
# - [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf) - Exploring different approaches to attention mechanisms.
# - [Matching Networks for One Shot Learning](https://arxiv.org/pdf/1606.04080.pdf) - Interesting way of doing one-shot learning with low-data by using an attention mechanism and a query to compare an image to other images for classification.
# - [Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144.pdf) - In 2016: stacked residual LSTMs with attention mechanisms on encoder/decoder are the best for NMT (Neural Machine Translation).
# - [Hybrid computing using a neural network with dynamic external memory](http://www.nature.com/articles/nature20101.epdf?author_access_token=ImTXBI8aWbYxYQ51Plys8NRgN0jAjWel9jnR3ZoTv0MggmpDmwljGswxVdeocYSurJ3hxupzWuRNeGvvXnoO8o4jTJcnAyhGuZzXJ1GEaD-Z7E6X_a9R-xqJ9TfJWBqz) - Improvements on differentiable memory based on NTMs: now it is the Differentiable Neural Computer (DNC).
# - [Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/pdf/1703.03906.pdf) - That yields intuition about the boundaries of what works for doing NMT within a framed seq2seq problem formulation.
# - [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram
# Predictions](https://arxiv.org/pdf/1712.05884.pdf) - A [WaveNet](https://arxiv.org/pdf/1609.03499v2.pdf) used as a vocoder can be conditioned on generated Mel Spectrograms from the Tacotron 2 LSTM neural network with attention to generate neat audio from text.
# 
# 
# <a name="youtube" />
# 
# ## YouTube and Videos
# 
# - [Attention Mechanisms in Recurrent Neural Networks (RNNs) - IGGG](https://www.youtube.com/watch?v=QuvRWevJMZ4) - A talk for a reading group on attention mechanisms (Paper: Neural Machine Translation by Jointly Learning to Align and Translate).
# - [Tensor Calculus and the Calculus of Moving Surfaces](https://www.youtube.com/playlist?list=PLlXfTHzgMRULkodlIEqfgTS-H1AY_bNtq) - Generalize properly how Tensors work, yet just watching a few videos already helps a lot to grasp the concepts.
# - [Deep Learning & Machine Learning (Advanced topics)](https://www.youtube.com/playlist?list=PLlp-GWNOd6m4C_-9HxuHg2_ZeI2Yzwwqt) - A list of videos about deep learning that I found interesting or useful, this is a mix of a bit of everything.
# - [Signal Processing Playlist](https://www.youtube.com/playlist?list=PLlp-GWNOd6m6gSz0wIcpvl4ixSlS-HEmr) - A YouTube playlist I composed about DFT/FFT, STFT and the Laplace transform - I was mad about my software engineering bachelor not including signal processing classes (except a bit in the quantum physics class).
# - [Computer Science](https://www.youtube.com/playlist?list=PLlp-GWNOd6m7vLOsW20xAJ81-65C-Ys6k) - Yet another YouTube playlist I composed, this time about various CS topics.
# - [Siraj's Channel](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A/videos?view=0&sort=p&flow=grid) - Siraj has entertaining, fast-paced video tutorials about deep learning.
# - [Two Minute Papers' Channel](https://www.youtube.com/user/keeroyz/videos?sort=p&view=0&flow=grid) - Interesting and shallow overview of some research papers, for example about WaveNet or Neural Style Transfer.
# - [Geoffrey Hinton interview](https://www.coursera.org/learn/neural-networks-deep-learning/lecture/dcm5r/geoffrey-hinton-interview) - Andrew Ng interviews Geoffrey Hinton, who talks about his research and breaktroughs, and gives advice for students.
# 
# <a name="misc-hubs-and-links" />
# 
# ## Misc. Hubs & Links
# 
# - [Hacker News](https://news.ycombinator.com/news) - Maybe how I discovered ML - Interesting trends appear on that site way before they get to be a big deal.
# - [DataTau](http://www.datatau.com/) - This is a hub similar to Hacker News, but specific to data science.
# - [Naver](http://www.naver.com/) - This is a Korean search engine - best used with Google Translate, ironically. Surprisingly, sometimes deep learning search results and comprehensible advanced math content shows up more easily there than on Google search.
# - [Arxiv Sanity Preserver](http://www.arxiv-sanity.com/) - arXiv browser with TF/IDF features.
# 
# 
# <a name="license" />

# ## Credits (Reference)
# 
# > * [Build a Convolutional Neural Network using Estimators](https://www.tensorflow.org/tutorials/estimators/cnn
# > * [MIT Deep Learning](https://github.com/lexfridman/mit-deep-learning)
# > * [GitHub Awesome Lists Topic](https://github.com/topics/awesome)
# > * [Christos Christofidis](https://github.com/ChristosChristofidis/awesome-deep-learning)
# > * [Guillaume Chevalier](https://github.com/guillaume-chevalier/Awesome-Deep-Learning-Resources)
# > * [GitHub Topic - Deep Learning](https://github.com/topics/deep-learning)
# 
# ## License
# 
# [![CC0](http://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
# 
# ### Please ***UPVOTE*** my kernel if you like it or wanna fork it.
# 
# ##### Feedback: If you have any ideas or you want any other content to be added to this curated list, please feel free to make any comments to make it better.
# #### I am open to have your *feedback* for improving this ***kernel***
# ###### Hope you enjoyed this kernel!
# 
# ### Thanks for visiting my *Kernel* and please *UPVOTE* to stay connected and follow up the *further updates!*
