#!/usr/bin/env python
# coding: utf-8

# # How to make a submission to the Tabular Playground Series (Sep 2022) Using TensorFlow Decision Forests

# **How to use this notebook**:
#  - Click on the "copy & edit" button in the top right corner. Run the code cells from top to bottom and save a new version.
#  - Read through and understand both the markdown cells as well as the code cells and their outputs.
#  - Make a submission to the [Tabular Playground Series](https://www.kaggle.com/competitions/tabular-playground-series-sep-2022) competition. Experiment and try to increase your score (model selection, hyperparameter choices, feature engineering, feature selection, etc)
# 

# # Introduction

# The goal of this notebook is to help Kagglers to: (1) better understand decision forest algorithms; (2) become familiar with the [TensorFlow Decision Forests (TF-DF)](https://www.tensorflow.org/decision_forests) Python API for executing decision forest algorithms; and (3) use decision forest algorithms to make submissions to the [September 2022 Tabular Playground Series](https://www.kaggle.com/competitions/tabular-playground-series-sep-2022/data) Kaggle competition.
# 
# We'll be using [TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests) (TF-DF) to train our model.  TensorFlow Decision Forests is a TensorFlow wrapper for the [Yggdrasil Decision Forests C++ libraries](https://github.com/google/yggdrasil-decision-forests).  TF-DF makes it very easy to train, serve and interpret various Decision Forest models such as [RandomForests](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/RandomForestModel) and [GrandientBoostedTrees](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/GradientBoostedTreesModel).  These types of decision forest models require minimal pre-processing of the data and are great when working with tabular datasets and/or small datasets (especially if you just want a quick baseline result to compare against).
# 
# [Decision Forests ](https://www.tensorflow.org/decision_forests)are a category of supervised machine learning methods that can be used for classification, regression and ranking. When thinking about decision forests, it can be helpful to first think about decision trees. Multiple decision trees can be "[ensembled](https://scikit-learn.org/stable/modules/ensemble.html#)" or analyzed together in order to form a "decision forest" which then should have improved predictive capabilities as compared to each individual tree.  The most popular decision forest methods are the [RandomForest](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) and [GradientBoostedTrees](https://jerryfriedman.su.domains/ftp/trebst.pdf) algorithms.  These algorithms are conceptually very similar to each other but anecdotally I often find that gradient boosted models have superior performance.
# * You can learn more about decision tree models and ensemble methods (such as RandomForest and GradientBoostedTrees) on scikit-learn's [decision tree](https://scikit-learn.org/stable/modules/tree.html#tree) and [ensemble](https://scikit-learn.org/stable/modules/ensemble.html#) pages. Some very nice implementationss of random forest algorithms can be found in popular Python packages such as [TF-DF](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/RandomForestModel),  [scikit-learn](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees), and [xgboost](https://xgboost.readthedocs.io/en/latest/tutorials/rf.html?highlight=random%20forest#random-forests-tm-in-xgboost).  Likewise, you can find great gradient boosted trees algorithms in [TF-DF](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/GradientBoostedTreesModel),  [scikit-learn](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting), and [xgboost](https://xgboost.readthedocs.io/en/latest/tutorials/model.html#introduction-to-boosted-trees).  For an explanation on the differences between random forest models and gradient boosted models, see [here](https://xgboost.readthedocs.io/en/latest/R-package/discoverYourData.html?highlight=random%20forest#special-note-what-about-random-forests) and [here](https://xgboost.readthedocs.io/en/latest/tutorials/rf.html?highlight=random%20forest#random-forests-tm-in-xgboosthttps://xgboost.readthedocs.io/en/latest/tutorials/rf.html?highlight=random%20forest#random-forests-tm-in-xgboost).  
# * For hands-on experience with these same concepts consider completing Kaggle Learn's [Intro to machine learning](https://www.kaggle.com/learn/intro-to-machine-learning) and [Intermediate machine learning](https://www.kaggle.com/learn/intermediate-machine-learning%5C) courses.
# 

# 
# We'll be working with the [Tabular Playground Series September 2022](https://www.kaggle.com/competitions/tabular-playground-series-sep-2022/data) Kaggle Dataset.  It is a tabular dataset with ~70,000 rows and 6 columns (4.49MB .CSV training dataset + 1.06MB .CSV test set) that is suitable for training regression algorithms (in this case to determine the number of units sold ("num_sold")).
# 
# Submissions in this competition will be [evaluated](https://admin.kaggle.com/competitions/tabular-playground-series-sep-2022/overview/evaluation) according to the [SMAPE metric](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error).  The SMAPE metric will be equal to 0 when the predictions are perfect. A few additional metrics that you consider using to evaluate regression models include [root-mean-square error (RMSE)](https://en.wikipedia.org/wiki/Root-mean-square_deviation) and [mean absolute error (mae)](https://en.wikipedia.org/wiki/Mean_absolute_error).  To learn more validating predictions for regression problems, consider reviewing https://www.kaggle.com/code/dansbecker/model-validation.
# 
# By studying this tutorial you will learn how to use [TF-DF](https://www.tensorflow.org/decision_forests) to quickly train a [GradientBoostedTrees](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/GradientBoostedTreesModel) model to perform a binary classification task using tabular data, in order to make a submission to the [September 2022 Tabular Playground Series competition](https://www.kaggle.com/competitions/tabular-playground-series-sep-2022/) on Kaggle.

# Step 1: Import Python packages

# In[1]:


get_ipython().system('pip install tensorflow_decision_forests')


# In[2]:


# Import Python packages
import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import tensorflow as tf
import tensorflow_decision_forests as tfdf
from pandas_profiling import ProfileReport
print("TensorFlow Decision Forests v" + tfdf.__version__)
pd.set_option('display.max_columns', None)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


def plot_tfdf_model_training_curves(model):
    # This function was adapted from the following tutorial:
    # https://www.tensorflow.org/decision_forests/tutorials/beginner_colab
    logs = model.make_inspector().training_logs()
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    # Plot rmse vs number of trees
    plt.plot([log.num_trees for log in logs], [log.evaluation.rmse for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("RMSE (out-of-bag)")
    plt.subplot(1, 2, 2)
    # Plot loss vs number of trees
    plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Logloss (out-of-bag)")
    plt.show()


# Step 2: Identify the location of the data

# In[4]:


# print list of all data and files attached to this notebook
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Step 3: Load the data

# In[5]:


# load to pandas dataframe (for data exploration)
train_df = pd.read_csv('/kaggle/input/tabular-playground-series-sep-2022/train.csv').drop('row_id', axis=1)
test_df = pd.read_csv('/kaggle/input/tabular-playground-series-sep-2022/test.csv').drop('row_id', axis=1)

# load to tensorflow dataset (for model training)
train_tfds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="num_sold", task=tfdf.keras.Task.REGRESSION)
test_tfds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, task=tfdf.keras.Task.REGRESSION)


# Step 4: Explore the data

# In[6]:


# print column names
print(train_df.columns)
print(train_df.shape)


# It is a tabular dataset with ~70,000 rows and 6 columns (4.49MB .CSV training dataset + 1.06MB .CSV test set) that is suitable for training regression algorithms (in this case to determine the number of units sold ("num_sold")).

# In[7]:


# preview first few rows of data
train_df.head(10)


# In[8]:


# print basic summary statistics
train_df.describe().head(3)
# For additional summary statistics you can run the following few lines as well:
# from pandas_profiling import ProfileReport
# # train_profile = ProfileReport(train_df, title="September 2022 Tabular Data Series")
# # train_profile


# In[9]:


# check for missing values
sns.heatmap(train_df.isnull(), cbar=False)


# RandomForest and GradientBoostedTrees do not throw any error messages when presented with missing values, but for improved model performance we might consider handling any missing values in a more thoughtful way (there seem to be too few null values for this to be worthwhile for this particular exercise).  To learn more about data imputation techniques, consider reviewing the relevant Kaggle Learn exercise [here](https://www.kaggle.com/code/alexisbcook/missing-values).

# In[10]:


fig = px.line(train_df, x="date", y="num_sold", color='country', title='Number of Units Sold Over Time')
fig.show()


# Step 5: Feature Engineering and Feature Selection

# In[11]:


#  ADD CONTENT HERE TO IMPROVE YOUR SCORE!


# To get to the top of the leaderboard it will be important to do a lot of clever feature engineering and feature selection. You can learn more about these concepts by reviewing the relevant Kaggle Learn exercise [here]((https://www.kaggle.com/learn/feature-engineering)). These steps were intentionally skipped in this tutorial for the sake of brevity.

# # RandomForest
# 
# Next we will take our training data and we will use it to train a Random Forest model (to predict the number of units sold).

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

# In[12]:


print(tfdf.keras.RandomForestModel.predefined_hyperparameters())


# In[13]:


# Train the model
rf_model = tfdf.keras.RandomForestModel(hyperparameter_template="better_default",num_trees=75, task=tfdf.keras.Task.REGRESSION)
rf_model.compile(metrics=["mae"]) # change to SMAPE eventually
rf_model.fit(x=train_tfds)


# In[14]:


# Visualize the model
# Currently this step works in the Kaggle Notebook Editor but unfortunately displays an empty/blank visualization in the Notebook Viewer
tfdf.model_plotter.plot_model_in_colab(rf_model, tree_idx=0, max_depth=3)


# Step 7: Evaluate your Random Forest Model

# In[15]:


plot_tfdf_model_training_curves(rf_model)


# In[16]:


inspector = rf_model.make_inspector()
inspector.evaluation()


# In[17]:


#rf_model.evaluate(train_tfds)


# In[18]:


print("Model type:", inspector.model_type())
print("Objective:", inspector.objective())
print("Evaluation:", inspector.evaluation())


# RMSE ~20.8 is not a bad baseline result given how few lines of code were required to get that result.  To get to the top of the leaderboard it will be important to do add in a few additional code cells where and spend some time doing some feature engineering and feature selection (these steps were intentionally skipped in this tutorial for the sake of brevity). You can learn more about these concepts by reviewing the relevant Kaggle Learn exercise [here]((https://www.kaggle.com/learn/feature-engineering)).

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

# In[19]:


# Adapted from https://www.tensorflow.org/decision_forests/tutorials/advanced_colab
# See list of inspector methods from:
# [field for field in dir(inspector) if not field.startswith("_")]
print(f"Available variable importances:")
for importance in inspector.variable_importances().keys():
  print("\t", importance)


# Variable importances describe how each feature impacts the model.

# In[20]:


inspector.variable_importances()["SUM_SCORE"]


# Here we can see that our model thinks that the most important feature is the "store" column.
# 
# Using the RandomForest model we were able to achieve RMSE ~20.8. This isn't necessarily a great result.  To get to the top of the leaderboard it will be important to do add in a few additional code cells where and spend some time doing some feature engineering and feature selection (as mentioned previously).  In the meantime, let's evaluate if simply swapping out the RandomForest for a GradientBoostedTrees model is enough to improve our result.

# # GradientBoostedTrees
# 
# Next we will take our training data and we will use it to train a GradientBoostedTrees model (to predict whether a given piece of machinery is in a state of "0" or "1").

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

# In[21]:


# As mentioned previously, TF-DF gives you lots of different "default" hyper-parameter settings to choose from.
print(tfdf.keras.GradientBoostedTreesModel.predefined_hyperparameters())


# In[22]:


# Train the model
gb_model = tfdf.keras.GradientBoostedTreesModel(hyperparameter_template="better_default",num_trees=200,early_stopping='NONE', task=tfdf.keras.Task.REGRESSION)
gb_model.compile(metrics=["mae"]) # change to SMAPE eventually
gb_model.fit(x=train_tfds)


# In[23]:


# Visualize the model
# Currently this step works in the Kaggle Notebook Editor but unfortunately displays an empty/blank visualization in the Notebook Viewer
tfdf.model_plotter.plot_model_in_colab(gb_model, tree_idx=0, max_depth=3)


# Step 10: Evaluate your GradientBoostedTrees  Model

# In[24]:


plot_tfdf_model_training_curves(gb_model)


# In[25]:


inspector = gb_model.make_inspector()
inspector.evaluation()


# In[26]:


gb_model.evaluate(train_tfds)


# In[27]:


print("Model type:", inspector.model_type())
print("Objective:", inspector.objective())
print("Evaluation:", inspector.evaluation())


# In[28]:


gb_model.evaluate(train_tfds)


# RMSE ~14.9 and MAE ~8.9 is not a bad baseline result given how few lines of code were required, but we'll still want to eventually do a bit more with the feature engineering and feature selection steps in order to improve this score a bit.

# Step 10: Investigate variable importances for the GradientBoostedTrees model

# As mentioned previously, variable importances describe how each feature impacts the model. Variable importances can tell you how much a given variable contributes to the model's predictions. 
# 

# In[29]:


# Adapted from https://www.tensorflow.org/decision_forests/tutorials/advanced_colab
# See list of inspector methods from:
# [field for field in dir(inspector) if not field.startswith("_")]
print(f"Available variable importances:")
for importance in inspector.variable_importances().keys():
  print("\t", importance)


# Variable importances describe how each feature impacts the model. Once again we can see that our most important features were the features that we created during our feature engineering step. 

# In[30]:


inspector.variable_importances()["SUM_SCORE"]


# Our GradientBoostedTrees algorithm found that the "store" feature was the most informative, just like our RandomForest model did.  
# 
# Using the GradientBoostedTrees model we were able to achieve RMSE ~14.9 and MAE ~8.9. This isn't necessarily a great result.  To get to the top of the leaderboard it will be important to do add in a few additional code cells and spend some time doing some feature engineering and feature selection (as mentioned previously).  

# In[31]:


# Visualize the model
# Currently this step works in the Kaggle Notebook Editor but unfortunately displays an empty/blank visualization in the Notebook Viewer
tfdf.model_plotter.plot_model_in_colab(gb_model, tree_idx=0, max_depth=3)


# Step 11: Submit your results

# In[32]:


# One submission file using RandomForest
sample_submission_df = pd.read_csv('/kaggle/input/tabular-playground-series-sep-2022/sample_submission.csv')
sample_submission_df['num_sold'] = rf_model.predict(test_tfds)
sample_submission_df.to_csv('/kaggle/working/rf_submission.csv', index=False)
sample_submission_df.head()

# And another using GradientBoostedTrees 
sample_submission_df = pd.read_csv('/kaggle/input/tabular-playground-series-sep-2022/sample_submission.csv')
sample_submission_df['num_sold'] = gb_model.predict(test_tfds)
sample_submission_df.to_csv('/kaggle/working/gb_submission.csv', index=False)
sample_submission_df.head()


# TF-DF makes it very easy to find lots of useful information about your model.  For example, the following code cell provides a tremendous amount of information with just a single line of code.  You can preview the output of this code cell by clicking on the "show output" button below.

# In[33]:


gb_model.summary()


# # Conclusion

# [TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests) (TF-DF) made it  quick and easy to train our RandomForest and GradientBoostedTrees models.  These types of decision forest models require minimal pre-processing of the data and are great when working with tabular datasets and/or small datasets (especially if you just want a quick baseline result to compare against).  Some of my favorite parts about  working with TF-DF were: (1) I was able to train a GradientBoostedTrees model with only a few lines of code; (2) there were lots of different default hyper-parameter options that I could choose from; (3) it was easy to visualize the structure/architecture of my models; and (4) it was easy to explore what features were most important to my model (to interpret and explain its decisions).
# 
# 
# We worked with the [Tabular Playground Series September 2022](https://www.kaggle.com/competitions/tabular-playground-series-sep-2022/data) Kaggle Dataset.  It was a tabular dataset with ~70,000 rows and 6 columns (4.49MB .CSV training dataset + 1.06MB .CSV test set) that was suitable for training regression algorithms (in this case to determine the number of units sold ("num_sold")).
# 
# Submissions in this competition will be [evaluated](https://admin.kaggle.com/competitions/tabular-playground-series-sep-2022/overview/evaluation) according to the [SMAPE metric](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error).  
# 
# 
# 
# We were able to quickly solve this task with an accuracy of RMSE ~14.9 and MAE ~8.9. This isn't necessarily a great result but hopefully can serve as a helpful and informative baseline for the task at hand.  To get to the top of the leaderboard it will be important to do add in a few additional code cells where and spend some time doing some feature engineering and feature selection (as mentioned previously). 
# 
# To learn more about TF-DF visit https://www.tensorflow.org/decision_forests.
# 
# Next steps?
#  - Click on the "copy & edit" button in the top right corner of this notebook
#  - Experiment and try to increase the score.  My recommendation would be to focus on the [feature engineering and feature selection](https://www.kaggle.com/learn/feature-engineering) steps, as these steps were omitted from this tutorial (for the sake of brevity)
#  - Make a submission to https://www.kaggle.com/competitions/tabular-playground-series-sep-2022

# Works Cited:
#  - [Build, train and evaluate models with TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests/tutorials/beginner_colab) from @[tensorflow](https://www.tensorflow.org/decision_forests/tutorials/)
#   - Code snippets for model training visualization 
#   - See comments in plot_tfdf_model_training_curves() for more detail
# 
# Other Useful References:
#  - https://www.tensorflow.org/decision_forests/tutorials/beginner_colab
#  - https://www.tensorflow.org/decision_forests/tutorials/intermediate_colab
#  - https://www.tensorflow.org/decision_forests/tutorials/advanced_colab

# In[ ]:





# In[ ]:




