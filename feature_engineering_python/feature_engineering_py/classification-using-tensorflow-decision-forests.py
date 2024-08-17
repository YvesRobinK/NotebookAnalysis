#!/usr/bin/env python
# coding: utf-8

# # [TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests): Classification example
# > "*TensorFlow Decision Forests (TF-DF) is a collection of state-of-the-art algorithms for the training, serving and interpretation of Decision Forest models. The library is a collection of Keras models and supports classification, regression and ranking.*"
# 
# This notebook is heavily based on the official tutorial ["*Build, train and evaluate models with TensorFlow Decision Forests*"](https://www.tensorflow.org/decision_forests/tutorials/beginner_colab).
# 
# First we shall install the `tensorflow_decision_forests` package

# In[1]:


get_ipython().system('pip3 install -q tensorflow_decision_forests')


# In[2]:


import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_decision_forests as tfdf

# Read in the data
train = pd.read_csv('../input/tabular-playground-series-sep-2021/train.csv',index_col=0)
test  = pd.read_csv('../input/tabular-playground-series-sep-2021/test.csv', index_col=0)


# **Feature engineering:** We shall add a new feature which is the count of the number of `Nan` present in each row. This idea is thanks to [Lukasz Borecki](https://www.kaggle.com/lukaszborecki). 
# 
# **Update:** I have now commented out this little piece of "*magic*" (which gives a Pulbic Leaderboard score of `0.80930` using less than 200 trees) as the purpose of this notebook is to be didactic. For a discussion on the topic see ["*Add Number of Nans in A Row as a Feature*"](https://www.kaggle.com/c/tabular-playground-series-sep-2021/discussion/270206) and the associated notebook ["TPS Sep 2021: Simple NaN model = 0.79446*"](https://www.kaggle.com/carlmcbrideellis/tps-sep-2021-simple-nan-model-0-79446)

# In[3]:


# train["nan_count"] = train.isnull().sum(axis=1)
# test["nan_count"]  = test.isnull().sum(axis=1)


# Create a test set using only 20% of the data, and a hold-out validation set that is never used until the end:

# In[4]:


# Use only 20% of the training data in this example
train_data      = train.sample(frac=0.2, random_state=42)
# create a hold-out validation set
validation_data = train.drop(train_data.index).sample(frac=0.05, random_state=42)


# In[5]:


train_data['claim'].value_counts().to_frame().T


# In[6]:


validation_data['claim'].value_counts().to_frame().T


# In general tree models should generally be robust w.r.t. missing data. However, it has been pointed out below in the comments section by [James McNeill](https://www.kaggle.com/datajmcn) (many thanks!) that Neural Nets don't work well with numerical NaNs, not that we are using neural networks here, but nevertheless. Here we shall simply replace them with zeros, although evidently a more sophisticated treatment is obviously preferable, for example using the [missingpy](https://github.com/epsilon-machine/missingpy) package.

# In[7]:


train_data      = train_data.fillna(0)
validation_data = validation_data.fillna(0)
test            = test.fillna(0)


# In[8]:


# Convert the dataset into a TensorFlow dataset.
train_ds      = tfdf.keras.pd_dataframe_to_tf_dataset(train_data, 
                                                      label="claim")                                          
validation_ds = tfdf.keras.pd_dataframe_to_tf_dataset(validation_data, 
                                                      label="claim")                                 
test_ds       = tfdf.keras.pd_dataframe_to_tf_dataset(test)


# We shall use the [`tfdf.keras.GradientBoostedTreesModel`](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/GradientBoostedTreesModel). Other models are the [`tfdf.keras.RandomForestModel`](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/RandomForestModel) and the [`tfdf.keras.CartModel`](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/CartModel)

# In[9]:


get_ipython().run_cell_magic('time', '', '\n# Train a Random Forest model.\n#model = tfdf.keras.RandomForestModel()\n\n# Train a Gradient Boosted Trees model.\nmodel = tfdf.keras.GradientBoostedTreesModel(num_trees=1500)\nmodel.fit(train_ds)\n')


# In[10]:


# Summary of the model structure.
model.summary()


# Plot the progress of our training

# In[11]:


logs = model.make_inspector().training_logs()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Accuracy (out-of-bag)")
plt.subplot(1, 2, 2)
plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Logloss (out-of-bag)")
plt.show()


# Calculate the score of our hold-out validation dataset

# In[12]:


predictions = model.predict(validation_ds)
y_true      = validation_data["claim"]

from sklearn.metrics import roc_auc_score
ROC_AUC = roc_auc_score(y_true, predictions)
print("The ROC AUC score is %.5f" % ROC_AUC )


# Now write out a `submission.csv`

# In[13]:


sample          = pd.read_csv('../input/tabular-playground-series-sep-2021/sample_solution.csv')
sample['claim'] = model.predict(test_ds)
sample.to_csv('submission.csv',index=False)


# # Related reading
# * [Introducing TensorFlow Decision Forests](https://blog.tensorflow.org/2021/05/introducing-tensorflow-decision-forests.html)
# * [TensorFlow Decision Forests](https://github.com/tensorflow/decision-forests) GitHub
# * [Yggdrasil Decision Forests](https://github.com/google/yggdrasil-decision-forests) GitHub
# 
# **Related kaggle notebooks**
# 
# * ["*Decision Forest for dummies*"](https://www.kaggle.com/kritidoneria/decision-forest-for-dummies) written by [KritiDoneria](https://www.kaggle.com/kritidoneria) and [Laurent Pourchot](https://www.kaggle.com/pourchot)
# * ["*Decision Forest fed by Neural Network*"](https://www.kaggle.com/pourchot/decision-forest-fed-by-neural-network) written by [Laurent Pourchot](https://www.kaggle.com/pourchot)
# * ["*Tensorflow decision forests (tfdf) - Titanic*"](https://www.kaggle.com/omidforoqi/tensorflow-decision-forests-tfdf-titanic) written by [Omid Foroqi](https://www.kaggle.com/omidforoqi)
# * ["*TensorFlow Decision Forests on Diabetes Dataset*"](https://www.kaggle.com/kirankunapuli/tensorflow-decision-forests-on-diabetes-dataset) written by [Kiran Kunapuli](https://www.kaggle.com/kirankunapuli)
