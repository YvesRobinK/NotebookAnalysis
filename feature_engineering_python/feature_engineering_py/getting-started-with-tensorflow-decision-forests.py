#!/usr/bin/env python
# coding: utf-8

# # Getting started with TensorFlow Decision Forests
# 

# **How to use this notebook**:
#  - Click on the "copy & edit" button in the top right corner. Run the code cells from top to bottom and save a new version.
#  - Read through and understand both the markdown cells as well as the code cells and their outputs.
#  - Make a submission to the [Tabular Playground Series](https://www.kaggle.com/competitions/tabular-playground-series-may-2022) competition. Experiment and try to increase your score (model selection, hyperparameter choices, feature engineering, feature selection, etc)
# 

# # Introduction

# The goal of this notebook is to help Kagglers to get started with the [TensorFlow Decision Forests (TF-DF)](https://www.tensorflow.org/decision_forests) Python API.  We will use data from the [Tabular Playground Series](https://www.kaggle.com/competitions/tabular-playground-series-may-2022/data) Kaggle competition to train ML models using TF-DF.
# 
# We'll be working with the [Tabular Playground Series May 2022](https://www.kaggle.com/competitions/tabular-playground-series-may-2022/data) Kaggle Dataset.  It is a tabular dataset with 900,000 rows and 33 columns (318MB .CSV training dataset + 247MB .CSV test set) that is suitable for training algorithms to solve binary classification problems (in this case to determine if a machine is in a state of "0" or "1" based off of input sensor data).  
# 
# We'll be using [TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests) (TF-DF) to train our model.  TensorFlow Decision Forests is a TensorFlow wrapper for the [Yggdrasil Decision Forests C++ libraries](https://github.com/google/yggdrasil-decision-forests).  TF-DF makes it very easy to train, serve and interpret various Decision Forest models such as [RandomForests](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/RandomForestModel) and [GrandientBoostedTrees](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/GradientBoostedTreesModel).  These types of decision forest models require minimal pre-processing of the data and are great when working with tabular datasets and/or small datasets (especially if you just want a quick baseline result to compare against).
# 
# By studying this tutorial you will learn how to quickly train a GradientBoostedTrees model to perform a binary classification task using tabular data.

# Step 1: Import Python packages

# In[1]:


get_ipython().system('pip install tensorflow_decision_forests')


# In[2]:


# Import Python packages
import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_decision_forests as tfdf
print("TensorFlow Decision Forests v" + tfdf.__version__)


# In[3]:


# Define helper functions:  
# One for plotting training evaluation curves, and another for expanding feature number 27.
# This bit of code is not particularly important with regards to learning how to use TensorFlow Decision Forests (TF-DF)
# If you are just trying to learn how to use TF-DF then my recommendation would be to skip this code cell and instead focus on understanding all the rest

def plot_tfdf_model_training_curves(model):
    # This function was adapted from the following tutorial:
    # https://www.tensorflow.org/decision_forests/tutorials/beginner_colab
    logs = model.make_inspector().training_logs()
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    # Plot accuracy vs number of trees
    plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy (out-of-bag)")
    plt.subplot(1, 2, 2)
    # Plot loss vs number of trees
    plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Logloss (out-of-bag)")
    plt.show()
    
    
def expand_feature_27(data):
    # This function was adapted from the following notebooks:
    # https://www.kaggle.com/code/cabaxiom/tps-may-22-eda-lgbm-model and
    # https://www.kaggle.com/code/ambrosm/tpsmay22-gradient-boosting-quickstart
    for df in [data]:
        # Extract the 10 letters of f_27 into individual features
        for i in range(10):
            df[f'ch{i}'] = df.f_27.str.get(i).apply(ord) - ord('A')
        df["unique_characters"] = df.f_27.apply(lambda s: len(set(s)))
        # Feature interactions: create three ternary features
        # Every ternary feature can have the values -1, 0 and +1
        df['i_02_21'] = (df.f_21 + df.f_02 > 5.2).astype(int) - (df.f_21 + df.f_02 < -5.3).astype(int)
        df['i_05_22'] = (df.f_22 + df.f_05 > 5.1).astype(int) - (df.f_22 + df.f_05 < -5.4).astype(int)
        i_00_01_26 = df.f_00 + df.f_01 + df.f_26
        df['i_00_01_26'] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)
        return data


# Step 2: Identify the location of the data

# In[4]:


# print list of all data and files attached to this notebook
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Step 3: Load the data

# In[5]:


# load to pandas dataframe (for data exploration)
train_df = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/train.csv')
test_df = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/test.csv')

# load to tensorflow dataset (for model training)
train_tfds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="target")
test_tfds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df)


# Step 4: Explore the data

# In[6]:


# print column names
print(train_df.columns)


# In[7]:


# preview first few rows of data
train_df.head(10)


# In[8]:


# print basic summary statistics
train_df.describe()


# In[9]:


# check for missing values
sns.heatmap(train_df.isnull(), cbar=False)


# Step 5: Feature Engineering

# Here we just expand out feature number 27. There are 10 unique character positions in feature number 27, and the following bit of code expands feature 27 to instead be 10+ features instead of only one feature. Adding in this step boosts our score by >>5%.

# In[10]:


print('Feature number 27 is a string') 
print('with 10 different character positions (1-10)') 
print('where each character position will contain')
print('one of 26 possible characters (A-Z):\n\n')
train_df[['f_27']].head()


# In[11]:


train_df = expand_feature_27(train_df)
test_df = expand_feature_27(test_df)
train_tfds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="target")
test_tfds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df)


# In[12]:


print('\n\nNew features split out from f_27:\n\n')
train_df[['f_27','ch0', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7',
       'ch8', 'ch9', 'unique_characters', 'i_02_21', 'i_05_22',
       'i_00_01_26']].head()


# To get to the top of the leaderboard you will likely want to do a lot more [feature engineering and feature selection](https://www.kaggle.com/learn/feature-engineering), as these steps were intentionally kept to a minimum in this tutorial (for the sake of brevity).

# # RandomForest
# 
# Next we will take our training data and we will use it to train a Random Forest model (to predict whether a given piece of machinery is in a state of "0" or "1").

# Step 6: Train a [Random Forest](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) Model
# 
# 
# 
# > "A Random Forest is a collection of deep CART decision trees trained independently and without pruning. Each tree is trained on a random subset of the original training dataset (sampled with replacement).
# > 
# > The algorithm is unique in that it is robust to overfitting, even in extreme cases e.g. when there is more features than training examples.
# > 
# > It is probably the most well-known of the Decision Forest training algorithms"
# 
# 
# 
# 
#  ~ Quoted from [TFDF RandomForest documentation ](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/RandomForestModel)

# One neat thing about TF-DF is that in addition to having a default set of hyper-parameters, you are also provided with a list of additional hyper-parameter choices to consider.  This makes it a lot easier to optimize model performance because you do not have to do this expensive hyper-parameter optimization step all by yourself.

# In[13]:


print(tfdf.keras.RandomForestModel.predefined_hyperparameters())


# In[14]:


# Train the model
rf_model = tfdf.keras.RandomForestModel(hyperparameter_template="better_default")
rf_model.compile(metrics=[tf.keras.metrics.AUC(curve="ROC")]) 
rf_model.fit(x=train_tfds)


# In[15]:


# Visualize the model
# Currently this step works in the Kaggle Notebook Editor but unfortunately displays an empty/blank visualization in the Notebook Viewer
tfdf.model_plotter.plot_model_in_colab(rf_model, tree_idx=0, max_depth=3)


# Step 7: Evaluate your Random Forest Model

# In[16]:


plot_tfdf_model_training_curves(rf_model)


# In[17]:


inspector = rf_model.make_inspector()
inspector.evaluation()


# In[18]:


rf_model.evaluate(train_tfds)


# In[19]:


print("Model type:", inspector.model_type())
print("Objective:", inspector.objective())
print("Evaluation:", inspector.evaluation())


# 90% accuracy is not a bad baseline result given how quickly we put this together (and with so few lines of code).

# Step 8: Investigate variable importances for the RandomForest model

# 
# Variable importances (VI) describe the impact of each feature to the model.
#  - > VIs generally indicates how much a variable contributes to the model predictions or quality. Different VIs have different semantics and are generally not comparable.
#  - > The VIs returned by variable_importances() depends on the learning algorithm and its hyper-parameters. For example, the hyperparameter compute_oob_variable_importances=True of the Random Forest learner enables the computation of permutation out-of-bag variable importances.
#  - > Variable importances can be obtained with tfdf.inspector.make_inspector(path).variable_importances().
# 
# The available variable importances are:
#  - > Model agnostic
#   - > MEAN_{INCREASE,DECREASE}_IN_{metric}: Estimated metric change from removing a feature using permutation importance . Depending on the learning algorithm and hyper-parameters, the VIs can be computed with validation, cross-validation or out-of-bag. For example, the MEAN_DECREASE_IN_ACCURACY of a feature is the drop in accuracy (the larger, the most important the feature) caused by shuffling the values of a features. For example, MEAN_DECREASE_IN_AUC_3_VS_OTHERS is the expected drop in AUC when comparing the label class "3" to the others.
#  - > Decision Forests specific
#   - > SUM_SCORE: Sum of the split scores using a specific feature. The larger, the most important.
#   - > NUM_AS_ROOT: Number of root nodes using a specific feature. The larger, the most important.
#   - > NUM_NODES: Number of nodes using a specific feature. The larger, the most important.
#   - > MEAN_MIN_DEPTH: Average minimum depth of the first occurence of a feature across all the tree paths. The smaller, the most important.
#   
# 
# ~ Quoted from [TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/inspector/AbstractInspector#variable_importances) documentation and [yggdrasil-decision-forests](https://github.com/google/yggdrasil-decision-forests/blob/main/documentation/user_manual.md#variable-importances) documentation.
# 

# In[20]:


# Adapted from https://www.tensorflow.org/decision_forests/tutorials/advanced_colab
# See list of inspector methods from:
# [field for field in dir(inspector) if not field.startswith("_")]
print(f"Available variable importances:")
for importance in inspector.variable_importances().keys():
  print("\t", importance)


# Variable importances describe how each feature impacts the model. Here we can see that our 4 most important features were "unique_characters","i_00_01_26","i_02_21", and "i_05_22". We created these features during our feature engineering step and it looks like it made a big difference!

# In[21]:


inspector.variable_importances()["SUM_SCORE"]


# # GradientBoostedTrees
# 
# Next we will take our training data and we will use it to train a Gradient Boosted model (to predict whether a given piece of machinery is in a state of "0" or "1").

# Step 9: Train a [GradientBoostedTrees](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) model.  GradientBoostedTrees often perform better than the RandomForests we were using previously.
# 
# 
# 
# > "A GBT (Gradient Boosted Tree) is a set of shallow decision trees trained sequentially. Each tree is trained to predict and then "correct" for the errors of the previously trained trees (more precisely each tree predict the gradient of the loss relative to the model output)"
# 
# 
# 
#  ~ Quoted from [TFDF GradientBoostedTrees documentation ](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/GradientBoostedTreesModel)
# 

# In[22]:


# As mentioned previously, TF-DF gives you lots of different "default" hyper-parameter settings to choose from.
print(tfdf.keras.GradientBoostedTreesModel.predefined_hyperparameters())


# In[23]:


# Train the model
gb_model = tfdf.keras.GradientBoostedTreesModel(hyperparameter_template="benchmark_rank1",num_trees=300)
gb_model.compile(metrics=[tf.keras.metrics.AUC(curve="ROC")])
gb_model.fit(x=train_tfds)


# In[24]:


# Visualize the model
# Currently this step works in the Kaggle Notebook Editor but unfortunately displays an empty/blank visualization in the Notebook Viewer
tfdf.model_plotter.plot_model_in_colab(gb_model, tree_idx=0, max_depth=3)


# Step 10: Evaluate your GradientBoostedTrees  Model

# In[25]:


plot_tfdf_model_training_curves(gb_model)


# In[26]:


inspector = gb_model.make_inspector()
inspector.evaluation()


# In[27]:


gb_model.evaluate(train_tfds)


# In[28]:


print("Model type:", inspector.model_type())
print("Objective:", inspector.objective())
print("Evaluation:", inspector.evaluation())


# In[29]:


gb_model.evaluate(train_tfds)


# 95% accuracy is not a bad baseline result given how quickly we put this together (and with so few lines of code).

# Step 10: Investigate variable importances for the GradientBoostedTrees model

# As mentioned previously, variable importances describe how each feature impacts the model. Variable importances can tell you how much a given variable contributes to the model's predictions. 
# 

# In[30]:


# Adapted from https://www.tensorflow.org/decision_forests/tutorials/advanced_colab
# See list of inspector methods from:
# [field for field in dir(inspector) if not field.startswith("_")]
print(f"Available variable importances:")
for importance in inspector.variable_importances().keys():
  print("\t", importance)


# Variable importances describe how each feature impacts the model. Once again we can see that our most important features were the features that we created during our feature engineering step. 

# In[31]:


inspector.variable_importances()["SUM_SCORE"]


# In[32]:


tfdf.model_plotter.plot_model_in_colab(gb_model, tree_idx=0, max_depth=3)


# Step 11: Submit your results

# In[33]:


sample_submission_df = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv')
sample_submission_df['target'] = gb_model.predict(test_tfds)
sample_submission_df.to_csv('/kaggle/working/submission.csv', index=False)
sample_submission_df.head()


# TF-DF makes it very easy to find lots of useful information about your model.  For example, the following code cell provides a tremendous amount of information with just a single line of code.  You can preview the output of this code cell by clicking on the "show output" button below.

# In[34]:


gb_model.summary()


# # Conclusion

# [TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests) (TF-DF) made it  quick and easy to train our RandomForest and GradientBoostedTrees models.  These types of decision forest models require minimal pre-processing of the data and are great when working with tabular datasets and/or small datasets (especially if you just want a quick baseline result to compare against).  Some of my favorite parts about  working with TF-DF were: (1) I was able to train a GradientBoostedTrees model with only a few lines of code; (2) there were lots of different default hyper-parameter options that I could choose from; (3) it was easy to visualize the structure/architecture of my models; and (4) it was easy to explore what features were most important to my model (to interpret and explain its decisions).
# 
# 
# We worked with the [Tabular Playground Series May 2022](https://www.kaggle.com/competitions/tabular-playground-series-may-2022/data) Kaggle Dataset.  It was a tabular dataset with 900,000 rows and 33 columns that contained data from industrial sensors, designed t be used to determine whether that piece of industrial equipment was in a state of  "0" or "1".
# 
# We were able to solve this task with an accuracy of ~95% which is not a bad baseline result given how quickly we were able to put this together (and with so few lines of code).
# 
# To learn more about TF-DF visit https://www.tensorflow.org/decision_forests.
# 
# Next steps?
#  - Click on the "copy & edit" button in the top right corner of this notebook
#  - Experiment and try to increase the score.  My recommendation would be to focus on the [feature engineering and feature selection](https://www.kaggle.com/learn/feature-engineering) steps, as these steps were omitted from this tutorial (for the sake of brevity)
#  - Make a submission to https://www.kaggle.com/competitions/tabular-playground-series-may-2022

# Works Cited:
#  - [Build, train and evaluate models with TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests/tutorials/beginner_colab) from @[tensorflow](https://www.tensorflow.org/decision_forests/tutorials/)
#   - Code snippets for model training visualization 
#   - See comments in plot_tfdf_model_training_curves() for more detail
#  - [[TPS-MAY-22] EDA & LGBM Model](https://www.kaggle.com/code/cabaxiom/tps-may-22-eda-lgbm-model) from @[cabaxiom](https://www.kaggle.com/cabaxiom)
#   - Feature engineering code snippets
#   - See comments in expand_feature_27() for more detail
#  - [TPSMAY22 Gradient-Boosting Quickstart](https://www.kaggle.com/code/ambrosm/tpsmay22-gradient-boosting-quickstart) from @[ambrosm](https://www.kaggle.com/ambrosm)
#   - Feature engineering code snippets
#   - See comments in expand_feature_27() for more detail
# 
# 
# Other Useful References:
#  - https://www.tensorflow.org/decision_forests/tutorials/beginner_colab
#  - https://www.tensorflow.org/decision_forests/tutorials/intermediate_colab
#  - https://www.tensorflow.org/decision_forests/tutorials/advanced_colab

# In[ ]:




